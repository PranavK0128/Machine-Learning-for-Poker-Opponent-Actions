from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import (
    InstalledResultValidationReceipt,
    ValidatedProducerArchiveSet,
    local_disk_preflight,
    materialize_local_archive_lane_group,
    stage_validated_result_archive,
)
from .common import (
    ALGORITHMS,
    STRATEGY_ABSTRACTION_VERSION,
    STAGES,
    StreamedJSONMapping,
    atomic_write,
    canonical_json_bytes,
    qualify_root,
    require_ephemeral_local_path,
    sha256_bytes,
    sha256_json,
    stream_file_identity,
    validate_self_hash,
    write_self_hashed_json_stream,
    with_self_hash,
)
from .continuation import (
    PolicyRolloutContinuationAuthority,
    load_continuation_authority,
)
from .crossplay import (
    CrossplayEvidenceError,
    validate_root_crossplay_once,
)
from .exact import projected_deal_distribution
from .game import state_from_root, strategy_backoff_keys
from .evaluation import EVALUATION_CONTRACT, EVALUATION_CONTRACT_SHA256, adjudicate_lane
from .exact import EXACT_CONTRACT_SHA256, ProjectedDealSampler
from .solver import SOLVER_CONTRACT_SHA256
from .ranges import build_joint_sampler, build_player_range
from .policy_store import open_staged_policy
from .legacy_results import (
    load_legacy_completed_result_contract,
    validate_legacy_completed_result_contract_target,
    validate_legacy_result_execution_identity_receipt,
    validate_legacy_result_lane_document,
)


class AuthorityBuildError(RuntimeError):
    pass


def _owned_authority_scratch(function):
    """Give authority finalization an exception-safe local scratch lifetime."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        stage_drive_root = Path(kwargs["stage_drive_root"])
        requested_parent = require_ephemeral_local_path(
            Path(kwargs["local_scratch_root"]),
            forbidden_roots=(stage_drive_root,),
        )
        requested_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="authority_owned_", dir=requested_parent
        ) as owned_text:
            owned_root = require_ephemeral_local_path(
                Path(owned_text), forbidden_roots=(stage_drive_root,)
            )
            kwargs["local_scratch_root"] = owned_root
            return function(*args, **kwargs)

    return wrapped


class _ValidatedStagedArchiveView:
    """Small read interface over coordinator-validated local lane staging."""

    def __init__(
        self,
        lane_handles: Mapping[str, Mapping[str, Any]],
        captured_members: Mapping[str, bytes],
    ) -> None:
        self._lane_handles = lane_handles
        self._captured_members = captured_members

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def namelist(self) -> list[str]:
        return [
            *(f"lanes/{lane_id}.json" for lane_id in self._lane_handles),
            *self._captured_members,
        ]

    def read(self, name: str) -> bytes:
        if name in self._captured_members:
            return self._captured_members[name]
        if name.startswith("lanes/") and name.endswith(".json"):
            lane_id = name.removeprefix("lanes/").removesuffix(".json")
            handle = self._lane_handles[lane_id]
            if handle.get("staged_member_released_after_policy_spool") is True:
                metadata = handle.get("document_metadata")
                if not isinstance(metadata, Mapping):
                    raise AuthorityBuildError(
                        "authority released staged lane lacks validated metadata"
                    )
                raw = canonical_json_bytes(metadata)
            else:
                raw = Path(str(handle["staged_path"])).read_bytes()
            if len(raw) != int(handle["size"]) or sha256_bytes(raw) != handle["sha256"]:
                raise AuthorityBuildError("authority staged lane readback mismatch")
            return raw
        raise KeyError(name)


class _SingleRootSamplerCache(dict[str, tuple[Any, Any]]):
    """Deterministic one-root cache for the authority recomputation pass."""

    def __setitem__(self, root_id: str, value: tuple[Any, Any]) -> None:
        if root_id not in self:
            self.clear()
        super().__setitem__(root_id, value)
        if len(self) > 1:
            raise AssertionError("authority sampler cache exceeded one root")


class _AuthorityLaneResourceGuard:
    """Close a lane policy before deleting its temporary materialization."""

    def __init__(self) -> None:
        self.policy = None
        self.temporary_directory = None
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        if self.policy is not None:
            self.policy.close()
            self.policy = None
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
            self.temporary_directory = None
        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _immutable_write(path: Path, raw: bytes) -> dict[str, Any]:
    if path.is_file():
        existing = path.read_bytes()
        if existing != raw:
            raise AuthorityBuildError(
                f"immutable authority collision at {path}; preserve/supersede the prior authority explicitly"
            )
        return {
            "path": str(path),
            "size": len(raw),
            "sha256": sha256_bytes(raw),
            "reused_identical": True,
        }
    return {**atomic_write(path, raw), "reused_identical": False}


def _immutable_install_stream(
    path: Path,
    source: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Install a local canonical spool without materializing its bytes."""

    if path.is_file():
        existing = stream_file_identity(path)
        if (
            existing["size"] != int(expected_identity["size"])
            or existing["sha256"] != expected_identity["sha256"]
        ):
            raise AuthorityBuildError(
                f"immutable authority collision at {path}; preserve/supersede explicitly"
            )
        return {**existing, "reused_identical": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        temporary = None
        installed = stream_file_identity(path)
        if (
            installed["size"] != int(expected_identity["size"])
            or installed["sha256"] != expected_identity["sha256"]
        ):
            raise IOError("stream-installed authority identity mismatch")
        return {**installed, "reused_identical": False}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _source_semantic_hashes(source_root: Path) -> dict[str, str]:
    package = source_root / "source" / "stage21_baseline"
    groups = {
        "game_engine_semantic_sha256": (
            "cards.py",
            "chance.py",
            "game.py",
            "exact.py",
            "restricted_solver.py",
            "restricted_oracle.py",
        ),
        "range_semantic_sha256": ("ranges.py", "coverage.py"),
        "evaluation_semantic_sha256": ("evaluation.py", "crossplay.py"),
        "solver_semantic_sha256": ("solver.py",),
        "continuation_code_semantic_sha256": ("continuation.py",),
    }
    source_hashes = {
        path.name: sha256_bytes(path.read_bytes())
        for path in sorted(package.glob("*.py"))
    }
    result = {}
    for field, names in groups.items():
        for name in names:
            path = package / name
            if not path.is_file():
                raise AuthorityBuildError(f"accepted authority source member is missing: {path}")
        members = {name: source_hashes[name] for name in names}
        result[field] = sha256_json(members)
    result["package_source_semantic_sha256"] = sha256_json(source_hashes)
    return result


def _lane_bindings(execution: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for notebook in execution["notebooks"]:
        if notebook["kind"] == "strict_aggregate":
            continue
        for lane in notebook["lanes"]:
            lane_id = str(lane["lane_id"])
            if lane_id in rows:
                raise AuthorityBuildError(f"duplicate planned lane: {lane_id}")
            rows[lane_id] = {
                **lane,
                "token": notebook["token"],
                "root_kind": notebook.get("root_kind"),
                "target_iterations": notebook.get("target_iterations", 0),
            }
    return rows


def _aggregate_root_evidence(
    *,
    execution: Mapping[str, Any],
    stage_drive_root: Path,
    expected_roots: Mapping[str, Mapping[str, Any]],
    include_validation_receipts: bool = False,
    installed_result_receipt: InstalledResultValidationReceipt | None = None,
    aggregate_lane_handles: Sequence[Mapping[str, Any]] = (),
    recovery_revalidate_archive: bool = False,
) -> Any:
    aggregate_rows = [
        row for row in execution["notebooks"] if row["kind"] == "strict_aggregate"
    ]
    if len(aggregate_rows) != 1:
        raise AuthorityBuildError("stage must contain one strict aggregate notebook")
    row = aggregate_rows[0]
    path = stage_drive_root / "returned_results" / row["result_filename"]
    if not path.is_file():
        raise AuthorityBuildError("strict aggregate result is missing")
    aggregate_staging_context = None
    if installed_result_receipt is not None:
        installed_result_receipt.lightweight_stat_check()
        receipt_document = installed_result_receipt.document
        if (
            receipt_document.get("stage") != execution["stage"]
            or receipt_document.get("token") != row["token"]
            or receipt_document.get("lane_count") != 48
        ):
            raise AuthorityBuildError("installed aggregate result receipt mismatch")
        installed_members = {
            str(member["path"]): member
            for member in receipt_document["validated_members"]
        }
        handle_map = {
            str(handle["lane_id"]): handle for handle in aggregate_lane_handles
        }
        if set(handle_map) != {str(lane["lane_id"]) for lane in row["lanes"]}:
            raise AuthorityBuildError("aggregate lane handle membership mismatch")

        def read_aggregate_lane(lane_id: str) -> bytes:
            handle = handle_map[lane_id]
            raw = Path(str(handle["path"])).read_bytes()
            member = installed_members.get(f"lanes/{lane_id}.json")
            if (
                member is None
                or len(raw) != int(handle["size"])
                or sha256_bytes(raw) != handle["sha256"]
                or int(member["size"]) != len(raw)
                or member["sha256"] != handle["sha256"]
            ):
                raise AuthorityBuildError("aggregate installed member binding mismatch")
            return raw

    elif recovery_revalidate_archive:
        aggregate_staging_context = tempfile.TemporaryDirectory(
            prefix="stage21_authority_aggregate_staging_"
        )
        staged_archive = stage_validated_result_archive(
            path,
            expected_stage=str(execution["stage"]),
            expected_token=str(row["token"]),
            expected_lane_ids=[
                str(lane["lane_id"]) for lane in row["lanes"]
            ],
            staging_root=Path(aggregate_staging_context.name),
        )
        if staged_archive["manifest"].get("lane_count") != 48:
            raise AuthorityBuildError("strict aggregate does not contain 48 shards")

        def read_aggregate_lane(lane_id: str) -> bytes:
            handle = staged_archive["lane_handles"][lane_id]
            if handle.get("staged_member_released_after_policy_spool") is True:
                metadata = handle.get("document_metadata")
                if not isinstance(metadata, Mapping):
                    raise AuthorityBuildError(
                        "aggregate released staged lane lacks validated metadata"
                    )
                raw = canonical_json_bytes(metadata)
            else:
                raw = Path(str(handle["staged_path"])).read_bytes()
            if len(raw) != int(handle["size"]) or sha256_bytes(raw) != handle["sha256"]:
                raise AuthorityBuildError("aggregate staging readback mismatch")
            return raw

    else:
        raise AuthorityBuildError(
            "strict authority requires the typed installed-result validation receipt"
        )
    evidence: dict[str, dict[str, Any]] = {}
    validation_receipts: dict[str, Any] = {}
    expected_lane_bindings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane in _lane_bindings(execution).values():
        expected_lane_bindings[str(lane["root_id"])].append(lane)
    for lane in row["lanes"]:
        lane_id = str(lane["lane_id"])
        raw = read_aggregate_lane(lane_id)
        document = json.loads(raw)
        validate_self_hash(document)
        if (
            document.get("role")
            != "strict_result_member_and_crossplay_validation_shard"
            or document.get("transport_valid") is not True
            or document.get("independent_confirmation_panel") is not False
        ):
            raise AuthorityBuildError(
                f"aggregate shard {lane_id} is not substantive transport/crossplay evidence"
            )
        root_rows = document.get("root_evidence", ())
        if not root_rows:
            raise AuthorityBuildError(f"aggregate shard {lane_id} is empty")
        for root_row in root_rows:
            validate_self_hash(root_row)
            root_id = str(root_row["root_id"])
            if root_id in evidence:
                raise AuthorityBuildError(
                    f"duplicate aggregate root evidence: {root_id}"
                )
            crossplay = root_row.get("crossplay")
            if isinstance(crossplay, Mapping):
                expected_root = expected_roots.get(root_id)
                if expected_root is None or expected_root.get("root_kind") != "approximate":
                    raise AuthorityBuildError(
                        f"unexpected cross-play evidence for {root_id}"
                    )
                try:
                    _, validation_receipts[root_id] = validate_root_crossplay_once(
                        crossplay,
                        expected_root_id=root_id,
                        expected_members=expected_lane_bindings[root_id],
                        deals_per_assignment=int(
                            execution["crossplay_deals_per_assignment"]
                        ),
                        familywise_alpha=float(
                            EVALUATION_CONTRACT["crossplay_familywise_alpha"]
                        ),
                        effective_stack_bb=(
                            min(state_from_root(expected_root).starting_stacks)
                            / 100.0
                        ),
                        limit_effective_stack_fraction_per_hand=float(
                            EVALUATION_CONTRACT[
                                "crossplay_max_absolute_symmetric_advantage_effective_stack_fraction_per_hand"
                            ]
                        ),
                        maximum_missing_policy_reach_fraction=float(
                            EVALUATION_CONTRACT[
                                "maximum_missing_policy_reach_fraction"
                            ]
                        ),
                        pairing_unit=str(
                            EVALUATION_CONTRACT["crossplay_pairing_unit"]
                        ),
                    )
                except (CrossplayEvidenceError, KeyError, TypeError, ValueError) as error:
                    raise AuthorityBuildError(
                        f"aggregate cross-play evidence is invalid for {root_id}: {error}"
                    ) from error
            if root_id in validation_receipts:
                validation_receipt = validation_receipts[root_id]
                evidence[root_id] = with_self_hash(
                    {
                        "schema_version": (
                            "stage21_compact_aggregate_root_evidence_v1"
                        ),
                        "root_id": root_id,
                        "root_kind": root_row["root_kind"],
                        "source_root_evidence_self_sha256": root_row[
                            "self_sha256"
                        ],
                        "member_lane_ids": list(root_row["member_lane_ids"]),
                        "member_document_self_sha256s": list(
                            root_row["member_document_self_sha256s"]
                        ),
                        "transport_archive_sha256": root_row[
                            "transport_archive_sha256"
                        ],
                        "crossplay": {
                            "self_sha256": validation_receipt.evidence_self_sha256
                        },
                        "validated_crossplay_receipt": (
                            validation_receipt.compact_document()
                        ),
                        "raw_crossplay_retained_in_aggregate_result_only": True,
                    }
                )
            else:
                evidence[root_id] = dict(root_row)
    if set(evidence) != set(expected_roots):
        raise AuthorityBuildError("aggregate evidence does not cover every root exactly once")
    if aggregate_staging_context is not None:
        aggregate_staging_context.cleanup()
    return (
        (evidence, validation_receipts)
        if include_validation_receipts
        else evidence
    )


def _iter_lane_documents(
    *,
    source_root: Path,
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
    stage_drive_root: Path,
    validated_producers: ValidatedProducerArchiveSet | None = None,
    recovery_revalidate_archives: bool = False,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    expected_all = _lane_bindings(execution)
    roots = {str(root["root_id"]): root for root in registry["roots"]}
    evaluation_sampler_cache: _SingleRootSamplerCache = _SingleRootSamplerCache()
    coverage_path = source_root / "contracts" / "coverage_contract.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    validate_self_hash(coverage)
    legacy_result_contract_path = (
        source_root / "contracts" / "legacy_completed_result_contract.json"
    )
    legacy_result_contract = (
        load_legacy_completed_result_contract(legacy_result_contract_path)
        if legacy_result_contract_path.is_file()
        else None
    )
    if legacy_result_contract is not None:
        qualification = json.loads(
            (source_root / "contracts" / "qualification_contract.json").read_text(
                encoding="utf-8"
            )
        )
        validate_self_hash(qualification)
        validate_legacy_completed_result_contract_target(
            legacy_result_contract,
            execution=execution,
            registry=registry,
            qualification=qualification,
        )
    range_identities: dict[str, dict[str, str]] = defaultdict(dict)
    marginal_identities: dict[str, dict[str, str]] = {}
    for profile in coverage["range_feature_coverage"]["per_root_player_profiles"]:
        root_id = str(profile["root_id"])
        range_identities[root_id][str(profile["player"])] = str(
            profile["range_identity_sha256"]
        )
        identity_row = {
            "declared_preconditioning": str(
                profile["declared_preconditioning_marginal_joint_identity_sha256"]
            ),
            "sampled_blocker_conditioned": str(
                profile[
                    "sampled_blocker_conditioned_marginal_joint_identity_sha256"
                ]
            ),
        }
        prior = marginal_identities.setdefault(root_id, identity_row)
        if prior != identity_row:
            raise AuthorityBuildError("coverage marginal identities disagree by player")
    expected_identity_base = {
        "execution_manifest_self_sha256": execution["self_sha256"],
        "package_source_semantic_sha256": execution["source_semantic_hashes"][
            "package_source_semantic_sha256"
        ],
        "root_registry_semantic_sha256": registry["semantic_sha256"],
        "coverage_contract_self_sha256": coverage["self_sha256"],
    }
    observed = set()
    authority_staging_context = None
    producer_by_token: dict[str, Mapping[str, Any]] = {}
    if validated_producers is not None:
        validate_self_hash(validated_producers.document)
        if (
            validated_producers.stage != execution["stage"]
            or validated_producers.document.get("factory_validated") is not True
        ):
            raise AuthorityBuildError("typed producer receipt stage binding mismatch")
        producer_by_token = {
            str(item["receipt"]["token"]): item
            for item in validated_producers.producers
        }
    elif recovery_revalidate_archives:
        authority_staging_context = tempfile.TemporaryDirectory(
            prefix="stage21_authority_producer_staging_"
        )
        authority_staging_root = Path(authority_staging_context.name)
    else:
        raise AuthorityBuildError(
            "strict authority requires coordinator-validated producer handles"
        )
    for notebook in execution["notebooks"]:
        if notebook["kind"] == "strict_aggregate":
            continue
        path = stage_drive_root / "returned_results" / notebook["result_filename"]
        if not path.is_file():
            raise AuthorityBuildError(f"missing producer result: {path.name}")
        identity_name = "receipts/execution_identity_receipt.json"
        if validated_producers is not None:
            staged_archive = producer_by_token.get(str(notebook["token"]))
            if staged_archive is None:
                raise AuthorityBuildError(
                    f"typed producer set is missing {notebook['token']}"
                )
        else:
            staged_archive = stage_validated_result_archive(
                path,
                expected_stage=str(execution["stage"]),
                expected_token=str(notebook["token"]),
                expected_lane_ids=[
                    str(lane["lane_id"]) for lane in notebook["lanes"]
                ],
                staging_root=authority_staging_root / str(notebook["token"]),
                capture_members=(identity_name,),
            )
        archive_sha = staged_archive["source_identity"]["sha256"]
        if validated_producers is not None:
            local_archive_path = Path(str(staged_archive["local_archive_path"]))
            local_archive_identity = stream_file_identity(local_archive_path)
            if (
                int(local_archive_identity["size"])
                != int(staged_archive["source_identity"]["size"])
                or local_archive_identity["sha256"] != archive_sha
            ):
                raise AuthorityBuildError(
                    f"producer {notebook['token']} local archive changed before authority"
                )
        manifest = staged_archive["manifest"]
        if manifest.get("lane_count") != 48:
            raise AuthorityBuildError(f"producer {notebook['token']} does not contain 48 lanes")
        expected_token_lanes = {
            str(lane["lane_id"]): lane for lane in notebook["lanes"]
        }
        with _ValidatedStagedArchiveView(
            staged_archive["lane_handles"],
            staged_archive["captured_members"],
        ) as archive:
            lane_names = {
                name for name in archive.namelist() if name.startswith("lanes/")
            }
            expected_names = {
                f"lanes/{lane_id}.json" for lane_id in expected_token_lanes
            }
            if lane_names != expected_names:
                raise AuthorityBuildError(
                    f"producer {notebook['token']} lane membership mismatch"
                )
            if identity_name not in archive.namelist():
                raise AuthorityBuildError(
                    f"producer {notebook['token']} lacks execution identity receipt"
                )
            identity = json.loads(archive.read(identity_name))
            validate_self_hash(identity)
            current_producer_identity = (
                identity.get("execution_manifest_self_sha256")
                == execution["self_sha256"]
                and identity.get("root_registry_semantic_sha256")
                == execution["root_registry_semantic_sha256"]
                and identity.get("package_source_semantic_sha256")
                == execution["source_semantic_hashes"][
                    "package_source_semantic_sha256"
                ]
            )
            legacy_producer_receipt = None
            if not current_producer_identity:
                if legacy_result_contract is None:
                    raise AuthorityBuildError(
                        f"producer {notebook['token']} was created from a different package contract"
                    )
                try:
                    legacy_producer_receipt = (
                        validate_legacy_result_execution_identity_receipt(
                            identity,
                            token=str(notebook["token"]),
                            contract=legacy_result_contract,
                            source_archive_identity=staged_archive[
                                "source_identity"
                            ],
                            result_filename=path.name,
                        )
                    )
                except ValueError as error:
                    raise AuthorityBuildError(
                        f"producer {notebook['token']} recovery identity failed"
                    ) from error
            if legacy_producer_receipt is not None and current_producer_identity:
                raise AssertionError(
                    "producer cannot be both current and legacy identity"
                )
            for lane_id in expected_token_lanes:
                planned = expected_all[lane_id]
                lane_resources = _AuthorityLaneResourceGuard()
                handle = staged_archive["lane_handles"].get(lane_id)
                if not isinstance(handle, Mapping):
                    raise AuthorityBuildError(
                        f"producer {notebook['token']} lacks staged handle {lane_id}"
                    )
                lane_materialization_context = None
                if (
                    handle.get("policy_materialization_deferred") is True
                    and handle.get("policy_mapping_present") is True
                ):
                    materialization_parent = (
                        Path(str(staged_archive["staging_root"]))
                        / "authority_lane_materialization"
                    ).resolve()
                    materialization_parent.mkdir(parents=True, exist_ok=True)
                    lane_materialization_context = tempfile.TemporaryDirectory(
                        prefix="lane_", dir=materialization_parent
                    )
                    lane_resources.temporary_directory = (
                        lane_materialization_context
                    )
                    handle = materialize_local_archive_lane_group(
                        [handle],
                        staging_root=Path(lane_materialization_context.name)
                        / "materialized",
                    )[0]
                metadata = handle.get("document_metadata")
                if (
                    handle.get("canonical_self_hash_verified") is not True
                    or not isinstance(metadata, Mapping)
                ):
                    raise AuthorityBuildError(
                        f"lane {lane_id} staged metadata binding mismatch"
                    )
                if handle.get("staged_member_released_after_policy_spool") is not True:
                    staged_path = Path(str(handle["staged_path"]))
                    if staged_path.stat().st_size != int(handle["size"]):
                        raise AuthorityBuildError(
                            f"lane {lane_id} staged member binding mismatch"
                        )
                document = dict(metadata)
                policy_handle = None
                if handle.get("policy_mapping_present") is True:
                    policy_handle = open_staged_policy(handle, cache_entries=4096)
                    lane_resources.policy = policy_handle
                    document["policy"] = policy_handle
                for field in ("lane_id", "root_id", "algorithm", "seed"):
                    if document.get(field) != planned.get(field):
                        raise AuthorityBuildError(
                            f"lane {lane_id} binding mismatch for {field}"
                        )
                expected_root = roots.get(str(planned["root_id"]))
                if expected_root is None:
                    raise AuthorityBuildError(f"lane {lane_id} references an unknown root")
                expected_root_hash = expected_root.get("root_semantic_sha256")
                execution_identity = document.get("execution_identity")
                current_lane_identity = (
                    expected_root_hash is not None
                    and document.get("root_semantic_sha256")
                    == expected_root_hash
                    and isinstance(execution_identity, Mapping)
                    and not any(
                        execution_identity.get(key) != value
                        for key, value in expected_identity_base.items()
                    )
                )
                legacy_lane_spec = None
                if not current_lane_identity:
                    if legacy_result_contract is None:
                        raise AuthorityBuildError(
                            f"lane {lane_id} package/root identity mismatch"
                        )
                    legacy_lane_spec = legacy_result_contract["lane_bindings"].get(
                        lane_id
                    )
                    if not isinstance(legacy_lane_spec, Mapping):
                        raise AuthorityBuildError(
                            f"lane {lane_id} is not an exact contracted legacy completed result"
                        )
                    if (
                        document.get("root_semantic_sha256")
                        != legacy_lane_spec["legacy_root_semantic_sha256"]
                        or sha256_json(execution_identity)
                        != legacy_lane_spec[
                            "legacy_execution_identity_sha256"
                        ]
                    ):
                        raise AuthorityBuildError(
                            f"lane {lane_id} legacy package/root identity mismatch"
                        )
                else:
                    if execution_identity.get(
                        "range_identity_sha256_by_player"
                    ) != range_identities[str(planned["root_id"])]:
                        raise AuthorityBuildError(
                            f"lane {lane_id} range identity mismatch"
                        )
                    if execution_identity.get(
                        "range_marginal_identity_sha256"
                    ) != marginal_identities[str(planned["root_id"])]:
                        raise AuthorityBuildError(
                            f"lane {lane_id} sampled marginal identity mismatch"
                        )
                if document.get("status") != "COMPLETE":
                    raise AuthorityBuildError(f"lane {lane_id} is not complete")
                if lane_id in observed:
                    raise AuthorityBuildError(f"duplicate imported lane: {lane_id}")
                observed.add(lane_id)
                policy = document.get("policy")
                policy_sha = (
                    str(handle["policy_sha256"])
                    if handle.get("policy_sha256") is not None
                    else None
                )
                legacy_lane_receipt = None
                if legacy_lane_spec is not None:
                    try:
                        legacy_lane_receipt = validate_legacy_result_lane_document(
                            document,
                            current_lane=planned,
                            contract=legacy_result_contract,
                            policy_sha256=policy_sha,
                        )
                    except ValueError as error:
                        raise AuthorityBuildError(
                            f"lane {lane_id} legacy completed-result validation failed"
                        ) from error
                declared_policy_sha = document.get("solver_result", {}).get(
                    "policy_sha256"
                )
                if declared_policy_sha is not None and policy_sha != declared_policy_sha:
                    raise AuthorityBuildError(f"lane {lane_id} policy identity mismatch")
                if planned.get("root_kind") == "forced":
                    expected_count = {"river": 0, "turn": 44, "flop": 45, "preflop": 17_296}[
                        str(execution["stage"])
                    ]
                    mean = document.get("mean_utility_chips", ())
                    diagnostics = document.get("continuation_diagnostics")
                    continuation_ok = diagnostics is None or diagnostics.get(
                        "within_fallback_bound"
                    ) is True
                    if not (
                        document.get("role") == "forced_runout_evaluation"
                        and document.get("solver_iterations") == 0
                        and document.get("physical_runout_count") == expected_count
                        and document.get("expected_physical_runout_count") == expected_count
                        and document.get("per_sample_zero_sum_validated") is True
                        and isinstance(mean, Sequence)
                        and len(mean) == 2
                        and all(math.isfinite(float(value)) for value in mean)
                        and abs(math.fsum(float(value) for value in mean)) <= 1e-9
                        and document.get("qualified") is continuation_ok
                    ):
                        raise AuthorityBuildError(f"forced lane {lane_id} failed strict recomputation")
                else:
                    root_id = str(planned["root_id"])
                    if root_id not in evaluation_sampler_cache:
                        runtime_state = state_from_root(expected_root)
                        joint_sampler = build_joint_sampler(expected_root, runtime_state)
                        if planned.get("root_kind") == "restricted_exact":
                            projected = projected_deal_distribution(
                                joint_sampler.first,
                                joint_sampler.second,
                                combos_per_player=int(
                                    execution.get(
                                        "restricted_exact_combos_per_player", 4
                                    )
                                ),
                            )
                            evaluation_sampler = ProjectedDealSampler(projected)
                        else:
                            evaluation_sampler = joint_sampler
                        evaluation_sampler_cache[root_id] = (
                            runtime_state,
                            evaluation_sampler,
                        )
                    runtime_state, evaluation_sampler = evaluation_sampler_cache[root_id]
                    solver_result = document.get("solver_result")
                    evaluation = document.get("evaluation")
                    adjudication = document.get("adjudication")
                    checkpoint_lifecycle = document.get(
                        "checkpoint_lifecycle"
                    )
                    if not all(
                        isinstance(value, Mapping)
                        for value in (
                            solver_result,
                            evaluation,
                            adjudication,
                            policy,
                            checkpoint_lifecycle,
                        )
                    ):
                        raise AuthorityBuildError(f"solver lane {lane_id} lacks nested evidence")
                    for nested in (solver_result, evaluation, adjudication):
                        validate_self_hash(nested)
                    for panel in evaluation.get("panels", ()):
                        validate_self_hash(panel)
                    document_root_hash = (
                        legacy_lane_spec["legacy_root_semantic_sha256"]
                        if legacy_lane_spec is not None
                        else expected_root_hash
                    )
                    if (
                        solver_result.get("root_id") != planned["root_id"]
                        or solver_result.get("algorithm") != planned.get("algorithm")
                        or solver_result.get("seed") != planned.get("seed")
                        or solver_result.get("root_semantic_sha256")
                        != document_root_hash
                        or evaluation.get("root_id") != planned["root_id"]
                        or evaluation.get("evaluation_contract_sha256")
                        != EVALUATION_CONTRACT_SHA256
                        or solver_result.get("task_semantic_sha256")
                        != document.get("task_semantic_sha256")
                        or solver_result.get(
                            "solver_continuation_semantic_sha256"
                        )
                        != document.get(
                            "solver_continuation_semantic_sha256"
                        )
                    ):
                        raise AuthorityBuildError(f"solver lane {lane_id} nested binding mismatch")
                    exact_oracle = document.get("exact_oracle")
                    if planned.get("root_kind") == "restricted_exact":
                        if (
                            solver_result.get("exact_contract_sha256")
                            != EXACT_CONTRACT_SHA256
                            or not isinstance(exact_oracle, Mapping)
                        ):
                            raise AuthorityBuildError(f"exact lane {lane_id} contract mismatch")
                        validate_self_hash(exact_oracle)
                    else:
                        expected_solver_contract = (
                            legacy_result_contract["source_package"][
                                "solver_contract_sha256"
                            ]
                            if legacy_lane_spec is not None
                            else SOLVER_CONTRACT_SHA256
                        )
                        if (
                            solver_result.get("solver_contract_sha256")
                            != expected_solver_contract
                        ):
                            raise AuthorityBuildError(
                                f"approximate lane {lane_id} solver contract mismatch"
                            )
                    reported_required_iterations = int(
                        adjudication.get("required_terminal_iterations", -1)
                    )
                    reported_target_iterations = int(
                        document.get("target_iterations", -1)
                    )
                    reported_tier = int(document.get("target_tier", -1))
                    reported_requested_tier = int(
                        document.get("requested_target_tier", -1)
                    )
                    reported_requested_iterations = int(
                        document.get("requested_target_iterations", -1)
                    )
                    reported_minimum_tier = int(
                        document.get("minimum_qualification_tier", -1)
                    )
                    if planned.get("root_kind") == "approximate":
                        allowed_tiers = {
                            int(value)
                            for value in execution.get(
                                "allowed_approximate_tiers", ()
                            )
                        }
                        expected_tier_iterations = (
                            int(execution["approximate_base_iterations_per_tier"])
                            * reported_tier
                        )
                        first_tier = int(
                            execution["first_approximate_qualification_tier"]
                        )
                        tier_stride = int(
                            execution["approximate_qualification_tier_stride"]
                        )
                        tier_binding_valid = (
                            reported_tier in allowed_tiers
                            and reported_requested_tier in allowed_tiers
                            and reported_minimum_tier in allowed_tiers
                            and reported_tier >= first_tier
                            and reported_tier % tier_stride == 0
                            and reported_minimum_tier <= reported_tier
                            and reported_tier <= reported_requested_tier
                            and expected_tier_iterations
                            == reported_target_iterations
                            and reported_requested_iterations
                            == int(
                                execution[
                                    "approximate_base_iterations_per_tier"
                                ]
                            )
                            * reported_requested_tier
                        )
                        expected_look_iterations = [
                            int(
                                execution[
                                    "approximate_base_iterations_per_tier"
                                ]
                            )
                            * tier
                            for tier in sorted(allowed_tiers)
                            if reported_minimum_tier
                            <= tier
                            <= reported_requested_tier
                        ]
                        look_receipts = checkpoint_lifecycle.get(
                            "qualification_look_receipts", ()
                        )
                        receipt_iterations = [
                            int(receipt.get("iteration", -1))
                            for receipt in look_receipts
                            if isinstance(receipt, Mapping)
                        ]
                        lifecycle_binding_valid = (
                            isinstance(look_receipts, list)
                            and bool(look_receipts)
                            and len(receipt_iterations) == len(look_receipts)
                            and receipt_iterations
                            == sorted(set(receipt_iterations))
                            and all(
                                iteration in expected_look_iterations
                                and iteration <= reported_target_iterations
                                for iteration in receipt_iterations
                            )
                            and receipt_iterations[-1]
                            == reported_target_iterations
                            and all(
                                receipt.get("qualified") is False
                                for receipt in look_receipts[:-1]
                            )
                            and look_receipts[-1].get("qualified")
                            is document.get("qualified")
                            and checkpoint_lifecycle.get(
                                "qualification_look_iterations"
                            )
                            == expected_look_iterations
                        )
                    else:
                        tier_binding_valid = (
                            reported_tier == 0
                            and reported_requested_tier == 0
                            and reported_minimum_tier == 0
                            and reported_target_iterations
                            == int(planned["target_iterations"])
                            and reported_requested_iterations
                            == reported_target_iterations
                        )
                        lifecycle_binding_valid = (
                            checkpoint_lifecycle.get(
                                "qualification_look_iterations"
                            )
                            == []
                            and checkpoint_lifecycle.get(
                                "qualification_look_receipts"
                            )
                            == []
                        )
                    if (
                        not tier_binding_valid
                        or not lifecycle_binding_valid
                        or reported_required_iterations
                        != reported_target_iterations
                        or reported_target_iterations
                        != int(solver_result.get("iterations", -1))
                        or document.get("stopped_early_after_qualification")
                        is not (
                            reported_target_iterations
                            < reported_requested_iterations
                        )
                        or checkpoint_lifecycle.get(
                            "stopped_early_after_qualification"
                        )
                        is not (
                            reported_target_iterations
                            < reported_requested_iterations
                        )
                        or int(
                            checkpoint_lifecycle.get(
                                "effective_target_iteration", -1
                            )
                        )
                        != reported_target_iterations
                        or int(
                            checkpoint_lifecycle.get(
                                "requested_target_iteration", -1
                            )
                        )
                        != reported_requested_iterations
                        or (
                            reported_target_iterations
                            < reported_requested_iterations
                            and document.get("qualified") is not True
                        )
                    ):
                        raise AuthorityBuildError(
                            f"lane {lane_id} terminal-tier binding mismatch"
                        )
                    released_deals = int(
                        execution["evaluation_deals_per_family_role"]
                    )
                    reported_deals = int(
                        document.get("evaluation_deals_per_family_role", -1)
                    )
                    allowed_deals = {
                        int(value)
                        for value in execution.get(
                            "allowed_evaluation_deals_per_family_role", ()
                        )
                    }
                    if (
                        any(
                            int(panel.get("deals_per_family_role", -1))
                            != reported_deals
                            for panel in evaluation.get("panels", ())
                        )
                        or reported_deals not in allowed_deals
                        or reported_deals < released_deals
                        or (
                            planned.get("root_kind") == "approximate"
                            and int(
                                evaluation.get(
                                    "qualification_look_index", -1
                                )
                            )
                            != reported_tier
                            // int(
                                execution[
                                    "approximate_qualification_tier_stride"
                                ]
                            )
                        )
                        or int(
                            evaluation.get("qualification_look_count", -1)
                        )
                        != (
                            int(
                                execution[
                                    "maximum_approximate_qualification_looks"
                                ]
                            )
                            if planned.get("root_kind") == "approximate"
                            else 1
                        )
                    ):
                        raise AuthorityBuildError(
                            f"lane {lane_id} evaluation sample size is below release floor"
                        )
                    recomputed = adjudicate_lane(
                        root_kind=str(planned["root_kind"]),
                        solver_result=solver_result,
                        evaluation_result=evaluation,
                        exact_oracle_result=exact_oracle,
                        required_terminal_iterations=reported_required_iterations,
                        continuation_diagnostics=document.get("continuation_diagnostics"),
                        runtime_feature_board=runtime_state.board,
                        runtime_feature_sampler=evaluation_sampler,
                        runtime_feature_restricted_projected_game=(
                            planned.get("root_kind") == "restricted_exact"
                        ),
                    )
                    if adjudication != recomputed or document.get("qualified") is not recomputed[
                        "qualified"
                    ]:
                        raise AuthorityBuildError(f"lane {lane_id} adjudication was not reproducible")
                if document.get("artifact_hash_valid") is not True or document.get(
                    "independent_confirmation_panel"
                ) is not True:
                    raise AuthorityBuildError(f"lane {lane_id} lacks required validation flags")
                binding = {
                    **planned,
                    "result_archive_path": str(path),
                    "result_archive_sha256": archive_sha,
                    "lane_document_self_sha256": document["self_sha256"],
                    "policy_sha256": policy_sha,
                    "legacy_result_receipt": legacy_lane_receipt,
                    "legacy_producer_identity_receipt": (
                        legacy_producer_receipt
                    ),
                }
                try:
                    yield binding, document
                finally:
                    lane_resources.close()
                    lane_materialization_context = None
        if validated_producers is not None:
            evaluation_sampler_cache.clear()
            final_local_archive_identity = stream_file_identity(
                Path(str(staged_archive["local_archive_path"]))
            )
            if (
                int(final_local_archive_identity["size"])
                != int(staged_archive["source_identity"]["size"])
                or final_local_archive_identity["sha256"] != archive_sha
            ):
                raise AuthorityBuildError(
                    f"producer {notebook['token']} local archive changed during authority"
                )
            producer_staging_root = Path(
                str(staged_archive["staging_root"])
            ).resolve()
            if (
                producer_staging_root.name != str(notebook["token"])
                or producer_staging_root.parent.name
                != "validated_producer_scratch"
            ):
                raise AuthorityBuildError(
                    "unsafe progressive producer staging cleanup target"
                )
            shutil.rmtree(producer_staging_root)
    if observed != set(expected_all):
        missing = sorted(set(expected_all) - observed)
        raise AuthorityBuildError(f"complete member authority is missing lanes: {missing}")
    if authority_staging_context is not None:
        authority_staging_context.cleanup()


def _forced_qualification(
    root_id: str,
    lane_results: Sequence[Mapping[str, Any]],
    expected_lane_ids: Sequence[str],
    aggregate_root_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relevant = [row for row in lane_results if row.get("root_id") == root_id]
    observed_ids = [str(row["lane_id"]) for row in relevant]
    complete = (
        len(observed_ids) == 6
        and len(set(observed_ids)) == 6
        and set(observed_ids) == set(expected_lane_ids)
    )
    passing = [
        row
        for row in relevant
        if row.get("qualified") is True
        and row.get("artifact_hash_valid") is True
        and row.get("independent_confirmation_panel") is True
    ]
    return with_self_hash(
        {
            "schema_version": "stage21_forced_root_qualification_v2",
            "root_id": root_id,
            "expected_member_count": 6,
            "observed_member_count": len(relevant),
            "complete_member_authority": complete,
            "passing_evaluation_shard_count": len(passing),
            "aggregate_forced_evidence_gate_pass": aggregate_root_evidence
            is not None
            and aggregate_root_evidence.get("forced_member_validation", {}).get(
                "gate_pass"
            )
            is True,
            "accepted": complete
            and len(passing) == 6
            and aggregate_root_evidence is not None
            and aggregate_root_evidence.get("forced_member_validation", {}).get(
                "gate_pass"
            )
            is True,
        }
    )


def _validate_policy_row(row: Mapping[str, Any]) -> dict[str, float]:
    if not row:
        raise AuthorityBuildError("empty policy row")
    normalized = {str(label): float(value) for label, value in row.items()}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise AuthorityBuildError("policy row has invalid probability")
    total = math.fsum(normalized.values())
    if total <= 0:
        raise AuthorityBuildError("policy row has zero probability mass")
    return {label: value / total for label, value in normalized.items()}


class _RiverPolicyBankAccumulator:
    """Exact policy-bank reduction with bounded controller memory."""

    def __init__(self, path: Path, *, maximum_database_bytes: int) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        requested_quota = int(maximum_database_bytes)
        if requested_quota < 16 * 1024 * 1024:
            raise OSError(
                "River policy accumulator requires at least 16 MiB of bounded scratch"
            )
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA mmap_size=0")
        self.page_size = int(
            self.connection.execute("PRAGMA page_size").fetchone()[0]
        )
        requested_max_pages = requested_quota // self.page_size
        self.maximum_page_count = int(
            self.connection.execute(
                f"PRAGMA max_page_count={requested_max_pages}"
            ).fetchone()[0]
        )
        self.maximum_database_bytes = self.maximum_page_count * self.page_size
        self.maximum_observed_database_bytes = 0
        self.maximum_single_batch_growth_bytes = 0
        self.bounded_sql_batch_count = 0
        self.connection.executescript(
            """
            CREATE TABLE lane_signatures (
                projected_key TEXT PRIMARY KEY,
                action_signature TEXT NOT NULL
            );
            CREATE TABLE lane_values (
                projected_key TEXT NOT NULL,
                action_label TEXT NOT NULL,
                probability_sum REAL NOT NULL,
                observation_count INTEGER NOT NULL,
                PRIMARY KEY(projected_key, action_label)
            );
            CREATE TABLE family_signatures (
                algorithm TEXT NOT NULL,
                projected_key TEXT NOT NULL,
                action_signature TEXT NOT NULL,
                PRIMARY KEY(algorithm, projected_key)
            );
            CREATE TABLE family_values (
                algorithm TEXT NOT NULL,
                projected_key TEXT NOT NULL,
                action_label TEXT NOT NULL,
                member_probability_sum REAL NOT NULL,
                member_count INTEGER NOT NULL,
                PRIMARY KEY(algorithm, projected_key, action_label)
            );
            """
        )
        self.connection.commit()
        self._record_database_growth(0)
        self.statement_batch_rows = 2048
        self.family_members: dict[str, list[dict[str, Any]]] = {
            algorithm: [] for algorithm in ALGORITHMS
        }

    def _allocated_database_bytes(self) -> int:
        page_count = int(
            self.connection.execute("PRAGMA page_count").fetchone()[0]
        )
        return page_count * self.page_size

    def _record_database_growth(self, before_bytes: int) -> int:
        after_bytes = self._allocated_database_bytes()
        if after_bytes > self.maximum_database_bytes:
            raise AuthorityBuildError(
                "River policy accumulator exceeded its enforced SQLite quota"
            )
        self.maximum_observed_database_bytes = max(
            self.maximum_observed_database_bytes, after_bytes
        )
        self.maximum_single_batch_growth_bytes = max(
            self.maximum_single_batch_growth_bytes,
            max(0, after_bytes - int(before_bytes)),
        )
        return after_bytes

    def _bounded_executemany(
        self,
        statement: str,
        rows: Sequence[tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        before_bytes = self._allocated_database_bytes()
        try:
            cursor = self.connection.executemany(statement, rows)
        except sqlite3.OperationalError as error:
            if "full" in str(error).lower() or "disk" in str(error).lower():
                raise AuthorityBuildError(
                    "River policy accumulator exhausted its fail-closed SQLite quota"
                ) from error
            raise
        self.bounded_sql_batch_count += 1
        self._record_database_growth(before_bytes)
        return cursor

    def disk_quota_receipt(self) -> dict[str, Any]:
        current_bytes = self._record_database_growth(
            self._allocated_database_bytes()
        )
        return with_self_hash(
            {
                "schema_version": "stage21_river_policy_accumulator_disk_quota_v1",
                "sqlite_path_is_ephemeral_local_scratch": True,
                "sqlite_page_size": self.page_size,
                "sqlite_max_page_count": self.maximum_page_count,
                "enforced_maximum_database_bytes": self.maximum_database_bytes,
                "current_database_bytes": current_bytes,
                "maximum_observed_database_bytes": (
                    self.maximum_observed_database_bytes
                ),
                "maximum_single_batch_growth_bytes": (
                    self.maximum_single_batch_growth_bytes
                ),
                "bounded_sql_batch_count": self.bounded_sql_batch_count,
                "sqlite_max_page_count_enforced_before_schema_and_rows": True,
                "heuristic_policy_size_multiplier_used": False,
                "status": "PASS",
            }
        )

    def _signature(self, row: Mapping[str, float]) -> str:
        return canonical_json_bytes(sorted(row)).decode("ascii")

    def _require_signature(
        self,
        table: str,
        where: tuple[Any, ...],
        signature: str,
    ) -> None:
        if table == "lane_signatures":
            cursor = self.connection.execute(
                "INSERT INTO lane_signatures(projected_key, action_signature) "
                "VALUES (?, ?) ON CONFLICT(projected_key) DO UPDATE SET "
                "action_signature = excluded.action_signature "
                "WHERE lane_signatures.action_signature = excluded.action_signature",
                (*where, signature),
            )
        else:
            cursor = self.connection.execute(
                "INSERT INTO family_signatures(algorithm, projected_key, action_signature) "
                "VALUES (?, ?, ?) ON CONFLICT(algorithm, projected_key) DO UPDATE "
                "SET action_signature = excluded.action_signature WHERE "
                "family_signatures.action_signature = excluded.action_signature",
                (*where, signature),
            )
        if cursor.rowcount != 1:
            raise AuthorityBuildError(
                "policy projection merged different legal-action signatures"
            )

    def _require_signature_batch(
        self,
        table: str,
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        if not rows:
            return
        if table == "lane_signatures":
            cursor = self._bounded_executemany(
                "INSERT INTO lane_signatures(projected_key, action_signature) "
                "VALUES (?, ?) ON CONFLICT(projected_key) DO UPDATE SET "
                "action_signature = excluded.action_signature "
                "WHERE lane_signatures.action_signature = excluded.action_signature",
                rows,
            )
        else:
            cursor = self._bounded_executemany(
                "INSERT INTO family_signatures(algorithm, projected_key, action_signature) "
                "VALUES (?, ?, ?) ON CONFLICT(algorithm, projected_key) DO UPDATE "
                "SET action_signature = excluded.action_signature WHERE "
                "family_signatures.action_signature = excluded.action_signature",
                rows,
            )
        if cursor.rowcount != len(rows):
            raise AuthorityBuildError(
                "policy projection merged different legal-action signatures"
            )

    def add_member(
        self,
        binding: Mapping[str, Any],
        policy: Mapping[str, Mapping[str, Any]],
    ) -> None:
        lane_id = str(binding["lane_id"])
        algorithm = str(binding.get("algorithm"))
        if algorithm not in ALGORITHMS:
            raise AuthorityBuildError("River policy member has an invalid family")
        self.connection.execute("DELETE FROM lane_signatures")
        self.connection.execute("DELETE FROM lane_values")
        lane_signature_rows: list[tuple[Any, ...]] = []
        lane_value_rows: list[tuple[Any, ...]] = []

        def flush_lane_rows() -> None:
            self._require_signature_batch(
                "lane_signatures", lane_signature_rows
            )
            if lane_value_rows:
                self._bounded_executemany(
                    "INSERT INTO lane_values VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(projected_key, action_label) DO UPDATE SET "
                    "probability_sum = lane_values.probability_sum + "
                    "excluded.probability_sum, observation_count = "
                    "lane_values.observation_count + 1",
                    lane_value_rows,
                )
            lane_signature_rows.clear()
            lane_value_rows.clear()

        for full_key in policy:
            row = _validate_policy_row(policy[full_key])
            signature = self._signature(row)
            projected_keys = tuple(dict.fromkeys(strategy_backoff_keys(str(full_key))))
            for projected_key in projected_keys:
                lane_signature_rows.append((projected_key, signature))
                for label, probability in row.items():
                    lane_value_rows.append((projected_key, label, probability))
                if (
                    len(lane_signature_rows) >= self.statement_batch_rows
                    or len(lane_value_rows) >= self.statement_batch_rows * 4
                ):
                    flush_lane_rows()
        flush_lane_rows()
        family_signature_rows: list[tuple[Any, ...]] = []
        family_value_rows: list[tuple[Any, ...]] = []

        def flush_family_rows() -> None:
            self._require_signature_batch(
                "family_signatures", family_signature_rows
            )
            if family_value_rows:
                self._bounded_executemany(
                    "INSERT INTO family_values VALUES (?, ?, ?, ?, 1) "
                    "ON CONFLICT(algorithm, projected_key, action_label) DO UPDATE SET "
                    "member_probability_sum = family_values.member_probability_sum + "
                    "excluded.member_probability_sum, member_count = "
                    "family_values.member_count + 1",
                    family_value_rows,
                )
            family_signature_rows.clear()
            family_value_rows.clear()

        projected_key_cursor = self.connection.execute(
            "SELECT projected_key FROM lane_signatures ORDER BY projected_key"
        )
        for (raw_projected_key,) in projected_key_cursor:
            projected_key = str(raw_projected_key)
            values = list(
                self.connection.execute(
                    "SELECT action_label, probability_sum, observation_count "
                    "FROM lane_values WHERE projected_key = ? ORDER BY action_label",
                    (projected_key,),
                )
            )
            member_row = _validate_policy_row(
                {
                    str(label): float(total) / int(count)
                    for label, total, count in values
                }
            )
            signature = self._signature(member_row)
            family_signature_rows.append((algorithm, projected_key, signature))
            for label, probability in member_row.items():
                family_value_rows.append(
                    (algorithm, projected_key, label, probability)
                )
            if (
                len(family_signature_rows) >= self.statement_batch_rows
                or len(family_value_rows) >= self.statement_batch_rows * 4
            ):
                flush_family_rows()
        flush_family_rows()
        self.family_members[algorithm].append(
            {
                "lane_id": lane_id,
                "root_id": binding["root_id"],
                "seed": binding["seed"],
                "policy_sha256": binding["policy_sha256"],
                "lane_document_self_sha256": binding[
                    "lane_document_self_sha256"
                ],
                "result_archive_sha256": binding["result_archive_sha256"],
            }
        )
        self.connection.commit()
        self._record_database_growth(self._allocated_database_bytes())

    def _family_rows(self, algorithm: str):
        key_cursor = self.connection.execute(
            "SELECT projected_key FROM family_signatures "
            "WHERE algorithm = ? ORDER BY projected_key",
            (algorithm,),
        )
        for (raw_projected_key,) in key_cursor:
            projected_key = str(raw_projected_key)
            values = list(
                self.connection.execute(
                    "SELECT action_label, member_probability_sum, member_count "
                    "FROM family_values WHERE algorithm = ? AND projected_key = ? "
                    "ORDER BY action_label",
                    (algorithm, projected_key),
                )
            )
            yield projected_key, _validate_policy_row(
                {
                    str(label): float(total) / int(count)
                    for label, total, count in values
                }
            )

    def write_policy_bank(
        self, *, stage: str, path: Path, spool_root: Path
    ) -> dict[str, Any]:
        for algorithm in ALGORITHMS:
            if not self.family_members[algorithm]:
                raise AuthorityBuildError(
                    f"policy bank has no accepted {algorithm} members"
                )
        return write_self_hashed_json_stream(
            path,
            document={
                "schema_version": "stage21_policy_bank_v6",
                "stage": stage,
                "street": stage,
                "family_members": self.family_members,
                "accepted_member_count_by_family": {
                    family: len(rows)
                    for family, rows in self.family_members.items()
                },
                "projection_levels": 6,
                "member_equal_weight_before_cross_member_averaging": True,
                "action_labels_are_pot_relative": True,
                "action_labels_are_unique_within_every_legal_set": True,
                "strategy_abstraction_version": STRATEGY_ABSTRACTION_VERSION,
                "solver_primary_keys_bind_exact_canonical_public_board": True,
                "solver_primary_keys_preserve_own_cards_actions_and_exact_payoff_state": True,
                "downstream_backoff_is_separately_bounded_and_not_a_solver_key": True,
                "opponent_private_cards_in_policy_key": False,
                "hard_fallback_policy": "preregistered_low_variance_check_call",
            },
            streamed_mappings={
                "families": (
                    (
                        algorithm,
                        StreamedJSONMapping(self._family_rows(algorithm)),
                    )
                    for algorithm in sorted(ALGORITHMS)
                )
            },
            binding_fields=(
                "schema_version",
                "stage",
                "street",
                "strategy_abstraction_version",
            ),
            spool_root=spool_root,
        )

    def cleanup(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)


def _build_policy_bank(
    *,
    stage: str,
    accepted_lane_ids: set[str],
    documents: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    global_accumulator: dict[str, dict[str, dict[str, list[float]]]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    family_members = {algorithm: [] for algorithm in ALGORITHMS}
    for binding, document in documents:
        lane_id = str(binding["lane_id"])
        algorithm = binding.get("algorithm")
        if lane_id not in accepted_lane_ids or algorithm not in ALGORITHMS:
            continue
        policy = document.get("policy")
        if not isinstance(policy, Mapping):
            raise AuthorityBuildError(f"accepted lane {lane_id} has no policy")
        member_projection: dict[str, dict[str, list[float]]] = {}
        for full_key, raw_row in policy.items():
            row = _validate_policy_row(raw_row)
            for projected_key in strategy_backoff_keys(str(full_key)):
                projected = member_projection.setdefault(
                    projected_key, {label: [0.0, 0.0] for label in row}
                )
                if set(projected) != set(row):
                    raise AuthorityBuildError(
                        "policy projection merged different legal-action signatures"
                    )
                for label, probability in row.items():
                    projected[label][0] += probability
                    projected[label][1] += 1.0
        family_members[str(algorithm)].append(
            {
                "lane_id": lane_id,
                "root_id": binding["root_id"],
                "seed": binding["seed"],
                "policy_sha256": binding["policy_sha256"],
                "lane_document_self_sha256": binding["lane_document_self_sha256"],
                "result_archive_sha256": binding["result_archive_sha256"],
            }
        )
        family_accumulator = global_accumulator[str(algorithm)]
        for key, label_values in member_projection.items():
            member_row = {
                label: values[0] / values[1]
                for label, values in label_values.items()
            }
            member_row = _validate_policy_row(member_row)
            target = family_accumulator.setdefault(
                key, {label: [0.0, 0.0] for label in member_row}
            )
            if set(target) != set(member_row):
                raise AuthorityBuildError("policy bank action membership collision")
            for label, probability in member_row.items():
                target[label][0] += probability
                target[label][1] += 1.0
    families = global_accumulator
    for algorithm in ALGORITHMS:
        if not family_members[algorithm]:
            raise AuthorityBuildError(f"policy bank has no accepted {algorithm} members")
        family_accumulator = families[algorithm]
        for key in sorted(family_accumulator):
            label_values = family_accumulator[key]
            family_accumulator[key] = _validate_policy_row(
                {
                    label: values[0] / values[1]
                    for label, values in label_values.items()
                }
            )
    return with_self_hash(
        {
            "schema_version": "stage21_policy_bank_v6",
            "stage": stage,
            "street": stage,
            "families": families,
            "family_members": family_members,
            "accepted_member_count_by_family": {
                family: len(rows) for family, rows in family_members.items()
            },
            "projection_levels": 6,
            "member_equal_weight_before_cross_member_averaging": True,
            "action_labels_are_pot_relative": True,
            "action_labels_are_unique_within_every_legal_set": True,
            "strategy_abstraction_version": STRATEGY_ABSTRACTION_VERSION,
            "solver_primary_keys_bind_exact_canonical_public_board": True,
            "solver_primary_keys_preserve_own_cards_actions_and_exact_payoff_state": True,
            "downstream_backoff_is_separately_bounded_and_not_a_solver_key": True,
            "opponent_private_cards_in_policy_key": False,
            "hard_fallback_policy": "preregistered_low_variance_check_call",
        }
    )


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _calibrate_policy_rollout(
    *,
    stage: str,
    bank: Mapping[str, Any],
    draft_manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    aggregate_evidence: Mapping[str, Mapping[str, Any]],
    dependency_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    downstream = None
    continuation_binding = dependency_receipt.get("continuation")
    if continuation_binding is not None:
        downstream = load_continuation_authority(
            Path(str(continuation_binding["manifest_path"])),
            Path(str(continuation_binding["payload_path"])),
        )
        if not isinstance(downstream, PolicyRolloutContinuationAuthority):
            raise AuthorityBuildError("nested policy rollout calibration needs policy authority")
    authority = PolicyRolloutContinuationAuthority(
        draft_manifest,
        bank,
        downstream=downstream,
        _allow_pending_calibration=True,
    )
    approximate_roots = [
        root for root in registry["roots"] if root["root_kind"] == "approximate"
    ]
    if len(approximate_roots) < 8:
        raise AuthorityBuildError("continuation calibration lacks eight approximate anchors")
    anchor_indices = sorted(
        {
            round(index * (len(approximate_roots) - 1) / 7)
            for index in range(8)
        }
    )
    small_game_rows = []
    calibration_rollouts = 128
    for root_index in anchor_indices:
        root = approximate_roots[root_index]
        state = state_from_root(root)
        sampler = build_joint_sampler(root, state)
        deal_seed = int(
            sha256_json(
                [bank["self_sha256"], root["root_semantic_sha256"], "calibration_deal"]
            )[:16],
            16,
        )
        deal = sampler.sample(random.Random(deal_seed))
        for family in ALGORITHMS:
            exact = authority.exact_policy_expectation(
                state,
                deal,
                family,
                context=root.get("range_contract"),
            )
            estimated = authority.calibration_rollout_mean(
                state,
                deal,
                family,
                count=calibration_rollouts,
                context=root.get("range_contract"),
            )
            error = max(abs(exact[player] - estimated[player]) for player in (0, 1))
            small_game_rows.append(
                {
                    "root_id": root["root_id"],
                    "root_semantic_sha256": root["root_semantic_sha256"],
                    "family": family,
                    "deal_seed": deal_seed,
                    "exact_fixed_policy_expectation_chips": list(exact),
                    "rollout_mean_chips": list(estimated),
                    "absolute_error_chips": error,
                    "rollout_count": calibration_rollouts,
                    "claim_scope": "fixed_empirical_policy_estimator_error_only",
                }
            )
    small_error_limit = 25.0
    small_error = max(row["absolute_error_chips"] for row in small_game_rows)

    restricted_rows = []
    for root in registry["roots"]:
        if root["root_kind"] != "restricted_exact":
            continue
        state = state_from_root(root)
        projected = projected_deal_distribution(
            build_player_range(root, state, 0),
            build_player_range(root, state, 1),
            combos_per_player=4,
        )
        family_values = {}
        for family in ALGORITHMS:
            per_deal = [
                (
                    row.probability,
                    authority.exact_policy_expectation(
                        state,
                        row.deal,
                        family,
                        context=root.get("range_contract"),
                    ),
                )
                for row in projected
            ]
            family_values[family] = [
                math.fsum(weight * values[player] for weight, values in per_deal)
                for player in (0, 1)
            ]
        mixture = [
            math.fsum(family_values[family][player] for family in ALGORITHMS)
            / len(ALGORITHMS)
            for player in (0, 1)
        ]
        reference = aggregate_evidence[str(root["root_id"])][
            "numerical_order_stability"
        ]["reference_profile_value_chips"]
        error = max(abs(mixture[player] - float(reference[player])) for player in (0, 1))
        restricted_rows.append(
            {
                "root_id": root["root_id"],
                "projected_deal_count": len(projected),
                "bank_family_exact_profile_value_chips": family_values,
                "bank_mixture_exact_profile_value_chips": mixture,
                "restricted_solver_reference_profile_value_chips": reference,
                "absolute_profile_value_error_chips": error,
                "restricted_claim_scope": "four_combo_projected_game_only",
            }
        )
    exact_anchor_limit = 50.0
    if restricted_rows:
        exact_anchor_error = max(
            row["absolute_profile_value_error_chips"] for row in restricted_rows
        )
        exact_anchor_gate = exact_anchor_error <= exact_anchor_limit
        inherited = None
    else:
        inherited = (
            downstream.manifest.get("calibration") if downstream is not None else None
        )
        exact_anchor_error = None
        exact_anchor_gate = (
            isinstance(inherited, Mapping) and inherited.get("gate_pass") is True
        )
    diagnostics = authority.diagnostics()
    backoff_gate = diagnostics.get("within_all_backoff_bounds") is True
    gate = (
        len(small_game_rows) == 16
        and small_error <= small_error_limit
        and exact_anchor_gate
        and backoff_gate
    )
    return with_self_hash(
        {
            "schema_version": "stage21_policy_rollout_calibration_v1",
            "stage": stage,
            "policy_bank_self_sha256": bank["self_sha256"],
            "exact_fixed_policy_small_game_rows": small_game_rows,
            "small_game_anchor_count": len(small_game_rows),
            "maximum_small_game_leaf_absolute_error_chips": small_error,
            "maximum_allowed_small_game_leaf_absolute_error_chips": (
                small_error_limit
            ),
            "restricted_exact_anchor_rows": restricted_rows,
            "restricted_exact_anchor_count": len(restricted_rows),
            "maximum_restricted_anchor_profile_error_chips": exact_anchor_error,
            "maximum_allowed_restricted_anchor_profile_error_chips": (
                exact_anchor_limit
            ),
            "inherited_predecessor_calibration_self_sha256": inherited.get(
                "self_sha256"
            )
            if isinstance(inherited, Mapping)
            else None,
            "backoff_and_fallback_calibration_diagnostics": diagnostics,
            "every_backoff_level_runtime_bound_gate_pass": backoff_gate,
            "safe_solving_or_exploitability_guarantee_claim": False,
            "gate_pass": gate,
        }
    )


@_owned_authority_scratch
def build_stage_authority(
    *,
    source_root: Path,
    stage_drive_root: Path,
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
    dependency_receipt: Mapping[str, Any],
    validated_producers: ValidatedProducerArchiveSet,
    installed_aggregate_result: InstalledResultValidationReceipt,
    aggregate_lane_handles: Sequence[Mapping[str, Any]],
    local_scratch_root: Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Strictly recompute root qualification and publish authority last.

    This is adjudication, not an operator authorization interlock.  The
    function never promotes a partial root set and never overwrites different
    accepted bytes.
    """

    stage = str(execution["stage"])
    if registry.get("stage") != stage:
        raise AuthorityBuildError("registry stage mismatch")
    local_scratch_root = local_scratch_root.resolve()
    maximum_producer_lane_bytes = max(
        int(handle["size"])
        for producer in validated_producers.producers
        for handle in producer["lane_handles"].values()
    )
    maximum_single_lane_policy_materialization_bytes = (
        maximum_producer_lane_bytes * 2
    )
    disk_free_before_authority = int(shutil.disk_usage(local_scratch_root).free)
    authority_non_accumulator_reserve_bytes = (
        maximum_single_lane_policy_materialization_bytes
        + int(execution["estimated_required_disk_bytes_per_notebook"])
        + 512 * 1024 * 1024
    )
    river_accumulator_disk_quota_bytes = (
        math.floor(disk_free_before_authority * 0.90)
        - authority_non_accumulator_reserve_bytes
    )
    if stage == "river" and river_accumulator_disk_quota_bytes < 16 * 1024 * 1024:
        raise OSError(
            "local scratch cannot admit the fail-closed River authority accumulator"
        )
    authority_local_preflight = local_disk_preflight(
        local_scratch_root,
        components={
            "maximum_single_lane_policy_materialization": (
                maximum_single_lane_policy_materialization_bytes
            ),
            "policy_bank_and_authority_output_reserve": int(
                execution["estimated_required_disk_bytes_per_notebook"]
            ),
        },
    )
    authority_accumulator_admission = with_self_hash(
        {
            "schema_version": "stage21_authority_accumulator_admission_v1",
            "stage": stage,
            "disk_free_before_authority": disk_free_before_authority,
            "maximum_usable_fraction": 0.90,
            "maximum_single_lane_policy_materialization_bytes": (
                maximum_single_lane_policy_materialization_bytes
            ),
            "policy_bank_and_authority_output_reserve_bytes": int(
                execution["estimated_required_disk_bytes_per_notebook"]
            ),
            "fixed_emergency_reserve_bytes": 512 * 1024 * 1024,
            "enforced_river_accumulator_quota_bytes": (
                river_accumulator_disk_quota_bytes if stage == "river" else None
            ),
            "quota_derived_from_current_free_disk_not_policy_multiplier": True,
            "status": "PASS",
        }
    )

    def emit(event: str, **extra: Any) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, "phase": "authority", **extra})

    emit("authority_finalization_started")
    aggregate_evidence, crossplay_validation_receipts = _aggregate_root_evidence(
        execution=execution,
        stage_drive_root=stage_drive_root,
        expected_roots={str(root["root_id"]): root for root in registry["roots"]},
        include_validation_receipts=True,
        installed_result_receipt=installed_aggregate_result,
        aggregate_lane_handles=aggregate_lane_handles,
    )
    emit("authority_aggregate_evidence_recomputed")
    policy_spool_root = (local_scratch_root / "policy_projections").resolve()
    policy_spool_root.mkdir(parents=True, exist_ok=False)
    policy_spool_handles: list[tuple[dict[str, Any], Path]] = []
    river_policy_accumulator = (
        _RiverPolicyBankAccumulator(
            local_scratch_root / "river_policy_bank_accumulator.sqlite",
            maximum_database_bytes=river_accumulator_disk_quota_bytes,
        )
        if stage == "river"
        else None
    )
    river_policy_accumulator_disk_quota_receipt = None
    policy_bank_stream_preflight = None

    def cleanup_policy_spool() -> None:
        if (
            policy_spool_root.name == "policy_projections"
            and policy_spool_root.parent == local_scratch_root
        ):
            shutil.rmtree(policy_spool_root, ignore_errors=False)
        else:
            raise RuntimeError("unsafe policy projection cleanup target")

    lane_summaries = []
    member_rows = []
    for binding, document in _iter_lane_documents(
        source_root=source_root,
        execution=execution,
        registry=registry,
        stage_drive_root=stage_drive_root,
        validated_producers=validated_producers,
    ):
        summary = {
            "lane_id": binding["lane_id"],
            "root_id": binding["root_id"],
            "algorithm": binding.get("algorithm"),
            "seed": binding["seed"],
            "qualified": document.get("qualified") is True,
            "artifact_hash_valid": document.get("artifact_hash_valid") is True,
            "independent_confirmation_panel": document.get(
                "independent_confirmation_panel"
            )
            is True,
            "replication_axis": document.get("solver_result", {}).get(
                "replication_axis", "independent_stochastic_solver_seed"
            ),
            "policy_sha256": binding.get("policy_sha256"),
            "lane_document_self_sha256": binding.get(
                "lane_document_self_sha256"
            ),
            "legacy_completed_result": binding.get("legacy_result_receipt")
            is not None,
            "legacy_result_receipt_self_sha256": (
                binding["legacy_result_receipt"]["self_sha256"]
                if binding.get("legacy_result_receipt") is not None
                else None
            ),
        }
        lane_summaries.append(summary)
        member_rows.append(
            {
                **binding,
                "qualified": summary["qualified"],
                "artifact_hash_valid": summary["artifact_hash_valid"],
                "independent_confirmation_panel": summary[
                    "independent_confirmation_panel"
                ],
                "disposition": "ACCEPTED_SUPPORT"
                if summary["qualified"]
                else "COMPLETE_NONQUALIFYING_SUPPORT",
            }
        )
        if (
            summary["qualified"] is True
            and binding.get("algorithm") in ALGORITHMS
            and binding.get("root_kind") == "approximate"
        ):
            policy = document.get("policy")
            if not isinstance(policy, Mapping):
                raise AuthorityBuildError(
                    f"qualified lane {binding['lane_id']} lacks a policy"
                )
            if river_policy_accumulator is not None:
                policy_store_path = getattr(policy, "path", None)
                if policy_store_path is None:
                    raise AuthorityBuildError(
                        "River authority requires a bounded disk-backed policy"
                    )
                river_policy_accumulator.add_member(binding, policy)
            else:
                spool_path = policy_spool_root / f"{binding['lane_id']}.json"
                write_self_hashed_json_stream(
                    spool_path,
                    document={
                        "schema_version": "stage21_validated_policy_projection_spool_v1",
                        "lane_id": binding["lane_id"],
                        "policy_sha256": binding["policy_sha256"],
                        "lane_document_self_sha256": binding[
                            "lane_document_self_sha256"
                        ],
                    },
                    streamed_mappings={
                        "policy": (
                            (str(key), policy[key]) for key in sorted(policy)
                        )
                    },
                    binding_fields=(
                        "schema_version",
                        "lane_id",
                        "policy_sha256",
                        "lane_document_self_sha256",
                    ),
                    spool_root=local_scratch_root / "stream_spools",
                )
                policy_spool_handles.append((dict(binding), spool_path))
        emit(
            "authority_producer_member_adjudication_progress",
            completed_members=len(lane_summaries),
            total_members=registry["root_count"] * 6,
        )
    expected_by_root: dict[str, list[str]] = defaultdict(list)
    for lane_id, binding in _lane_bindings(execution).items():
        expected_by_root[str(binding["root_id"])].append(lane_id)
    registry_root_ids = {str(root["root_id"]) for root in registry["roots"]}
    if set(expected_by_root) != registry_root_ids:
        raise AuthorityBuildError("execution lanes and registry roots do not have exact membership")
    if len(member_rows) != registry["root_count"] * 6:
        raise AuthorityBuildError("complete stage member count is not six per root")
    if any(len(lanes) != 6 for lanes in expected_by_root.values()):
        raise AuthorityBuildError("a stage root does not have exactly six planned members")
    qualifications = []
    for root in registry["roots"]:
        root_id = str(root["root_id"])
        if root["root_kind"] == "forced":
            qualification = _forced_qualification(
                root_id,
                lane_summaries,
                expected_by_root[root_id],
                aggregate_evidence[root_id],
            )
        else:
            qualification = qualify_root(
                root_id,
                lane_summaries,
                root_kind=str(root["root_kind"]),
                aggregate_root_evidence=aggregate_evidence[root_id],
                require_aggregate_evidence=True,
                expected_effective_stack_bb=(
                    min(state_from_root(root).starting_stacks) / 100.0
                ),
                validated_crossplay_receipt=crossplay_validation_receipts.get(
                    root_id
                ),
            )
        qualifications.append(qualification)
        emit(
            "authority_root_qualification_progress",
            completed_roots=len(qualifications),
            total_roots=registry["root_count"],
        )
    all_roots_accepted = all(row["accepted"] for row in qualifications)
    approximate_root_ids = {
        str(root["root_id"])
        for root in registry["roots"]
        if root["root_kind"] == "approximate"
    }
    accepted_lane_ids = {
        row["lane_id"]
        for row in lane_summaries
        if row["qualified"] is True
        and row["algorithm"] in ALGORITHMS
        and row["root_id"] in approximate_root_ids
    }
    adjudication = with_self_hash(
        {
            "schema_version": "stage21_stage_adjudication_v6",
            "stage": stage,
            "root_count": len(qualifications),
            "accepted_root_count": sum(row["accepted"] for row in qualifications),
            "complete_member_count": len(member_rows),
            "expected_complete_member_count": registry["root_count"] * 6,
            "legacy_completed_result_member_count": sum(
                row["legacy_completed_result"] is True for row in lane_summaries
            ),
            "legacy_members_preserved_without_relabelling": True,
            "root_qualifications": qualifications,
            "minimum_distinct_passing_seeds_by_algorithm": {
                algorithm: 2 for algorithm in ALGORITHMS
            },
            "restricted_exact_member_axis": (
                "numerical_accumulation_order_stability_not_independent_seed"
            ),
            "aggregate_root_evidence_self_sha256s": {
                root_id: row["self_sha256"]
                for root_id, row in sorted(aggregate_evidence.items())
            },
            "all_roots_accepted": all_roots_accepted,
            "partial_authority_promotion_allowed": False,
            "status": "ACCEPTED" if all_roots_accepted else "REJECTED_NO_AUTHORITY",
        }
    )
    if not all_roots_accepted:
        if river_policy_accumulator is not None:
            river_policy_accumulator.cleanup()
        cleanup_policy_spool()
        return adjudication

    def spooled_policy_documents():
        for binding, path in policy_spool_handles:
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_self_hash(document)
            if (
                document.get("lane_id") != binding["lane_id"]
                or document.get("policy_sha256") != binding["policy_sha256"]
                or sha256_json(document.get("policy")) != binding["policy_sha256"]
            ):
                raise AuthorityBuildError("policy projection spool binding mismatch")
            try:
                yield binding, document
            finally:
                path.unlink(missing_ok=True)

    stage_number = STAGES[stage].number
    authority_dir = stage_drive_root / "authority"
    policy_bank_name = f"stage{stage_number}_{stage}_policy_bank.json"
    policy_bank_spool = local_scratch_root / "policy_bank_canonical.json"
    if river_policy_accumulator is not None:
        try:
            river_policy_accumulator_disk_quota_receipt = (
                river_policy_accumulator.disk_quota_receipt()
            )
            accumulator_size = river_policy_accumulator.path.stat().st_size
            policy_bank_stream_preflight = local_disk_preflight(
                local_scratch_root,
                components={
                    "canonical_policy_bank_stream_spool": accumulator_size,
                },
            )
            policy_bank_stream_receipt = (
                river_policy_accumulator.write_policy_bank(
                    stage=stage,
                    path=policy_bank_spool,
                    spool_root=local_scratch_root / "stream_spools",
                )
            )
        finally:
            river_policy_accumulator.cleanup()
        policy_bank_self_sha256 = str(
            policy_bank_stream_receipt["self_sha256"]
        )
        policy_bank = None
    else:
        policy_bank = _build_policy_bank(
            stage=stage,
            accepted_lane_ids=accepted_lane_ids,
            documents=spooled_policy_documents(),
        )
        policy_bank_self_sha256 = str(policy_bank["self_sha256"])
        policy_bank_stream_receipt = write_self_hashed_json_stream(
            policy_bank_spool,
            document={
                key: value
                for key, value in policy_bank.items()
                if key not in {"families", "self_sha256"}
            },
            streamed_mappings={
                "families": (
                    (
                        algorithm,
                        StreamedJSONMapping(
                            (
                                (
                                    policy_key,
                                    policy_bank["families"][algorithm][policy_key],
                                )
                                for policy_key in sorted(
                                    policy_bank["families"][algorithm]
                                )
                            )
                        ),
                    )
                    for algorithm in sorted(policy_bank["families"])
                )
            },
            binding_fields=(
                "schema_version",
                "stage",
                "street",
                "strategy_abstraction_version",
            ),
            spool_root=local_scratch_root / "stream_spools",
        )
        if policy_bank_stream_receipt["self_sha256"] != policy_bank_self_sha256:
            raise AuthorityBuildError("streamed policy-bank self hash changed")
    emit("authority_policy_bank_reduced")
    emit(
        "authority_policy_bank_spooled",
        policy_bank_size=policy_bank_stream_receipt["size"],
        policy_bank_sha256=policy_bank_stream_receipt["sha256"],
    )
    policy_bank_receipt = None
    source_hashes = _source_semantic_hashes(source_root)

    continuation = None
    continuation_manifest = None
    if stage in {"turn", "flop"}:
        downstream = None
        depth_limit = None
        if stage == "flop":
            depth_limit = "turn"
            dependency = dependency_receipt.get("continuation")
            if not isinstance(dependency, Mapping):
                raise AuthorityBuildError("Flop authority lacks accepted Turn continuation")
            downstream = {
                "manifest_relative_path": _relative_path(
                    Path(str(dependency["manifest_path"])), authority_dir
                ),
                "policy_bank_relative_path": _relative_path(
                    Path(str(dependency["payload_path"])), authority_dir
                ),
                "manifest_self_sha256": dependency["manifest_self_sha256"],
            }
        draft_manifest = with_self_hash(
            {
                "schema_version": "stage21_empirical_policy_rollout_continuation_manifest_v6",
                "authority_kind": "policy_rollout",
                "status": "ACCEPTED_SCIENCE_AUTHORITY",
                "stage": stage,
                "street": stage,
                "depth_limit": depth_limit,
                "complete_member_authority": True,
                "required_continuation_families": list(ALGORITHMS),
                "preregistered_family_mixture": {
                    algorithm: 1.0 / len(ALGORITHMS) for algorithm in ALGORITHMS
                },
                "claim_scope": "bounded_empirical_zero_sum_policy_rollout_leaf_model",
                "safe_solving_claim": False,
                "exploitability_nonincrease_guarantee": False,
                "opponent_cfv_gadget_present": False,
                "rollouts_per_family": 32,
                "common_random_numbers_by_exact_leaf": True,
                "common_random_numbers_across_families": True,
                "named_rng_streams": ["public_chance", "action_selection"],
                "per_rollout_joint_zero_sum": True,
                "scalar_family_selection_used": False,
                "aggregate_residual_recentering_used": False,
                "hard_fallback_policy": "preregistered_low_variance_check_call",
                "hard_fallback_fraction_limit": 0.05,
                "backoff_fraction_limits_by_level": [
                    1.0,
                    0.35,
                    0.35,
                    0.25,
                    0.15,
                    0.10,
                ],
                "every_backoff_level_is_runtime_gated": True,
                "calibration_requirement": {
                    "exact_policy_expectation_small_game_required": True,
                    "restricted_exact_anchor_required_when_stage_supports_it": True,
                    "maximum_leaf_mean_absolute_error_chips": 25.0,
                    "minimum_anchor_count": 8,
                },
                "calibration_status": "PENDING_BUILD_TIME_FAIL_CLOSED_CALIBRATION",
                "cache_entry_limit": 4096,
                "policy_bank_filename": policy_bank_name,
                "policy_bank_file_sha256": policy_bank_stream_receipt["sha256"],
                "policy_bank_self_sha256": policy_bank_self_sha256,
                "downstream": downstream,
            }
        )
        calibration = _calibrate_policy_rollout(
            stage=stage,
            bank=policy_bank,
            draft_manifest=draft_manifest,
            registry=registry,
            aggregate_evidence=aggregate_evidence,
            dependency_receipt=dependency_receipt,
        )
        emit(
            "authority_continuation_calibration_complete",
            gate_pass=calibration.get("gate_pass") is True,
        )
        if calibration.get("gate_pass") is not True:
            rejected = with_self_hash(
                {
                    **adjudication,
                    "status": "REJECTED_CONTINUATION_CALIBRATION_NO_AUTHORITY",
                    "continuation_calibration": calibration,
                    "policy_bank_published": False,
                }
            )
            cleanup_policy_spool()
            return rejected
        continuation_manifest = with_self_hash(
            {
                **{
                    key: value
                    for key, value in draft_manifest.items()
                    if key != "self_sha256"
                },
                "calibration_status": "PASS",
                "calibration": calibration,
            }
        )
        policy_bank_receipt = _immutable_install_stream(
            authority_dir / policy_bank_name,
            policy_bank_spool,
            expected_identity=policy_bank_stream_receipt,
        )
        continuation_name = (
            f"stage{stage_number}_{stage}_continuation_manifest.json"
        )
        continuation_receipt = _immutable_write(
            authority_dir / continuation_name,
            canonical_json_bytes(continuation_manifest),
        )
        continuation = {
            "authority_kind": "policy_rollout",
            "manifest_filename": continuation_name,
            "payload_filename": policy_bank_name,
            "manifest_self_sha256": continuation_manifest["self_sha256"],
            "manifest_file_sha256": continuation_receipt["sha256"],
            "policy_bank_self_sha256": policy_bank_self_sha256,
            "policy_bank_file_sha256": policy_bank_receipt["sha256"],
        }

    policy_bank = None
    gc.collect()
    emit("authority_policy_bank_memory_released_after_spool_and_calibration")

    if policy_bank_receipt is None:
        policy_bank_receipt = _immutable_install_stream(
            authority_dir / policy_bank_name,
            policy_bank_spool,
            expected_identity=policy_bank_stream_receipt,
        )
    emit("authority_policy_bank_published")

    member_registry = with_self_hash(
        {
            "schema_version": "stage21_complete_member_registry_v3",
            "stage": stage,
            "complete_member_authority": True,
            "member_count": len(member_rows),
            "members": sorted(member_rows, key=lambda row: row["lane_id"]),
        }
    )
    member_name = f"stage{stage_number}_{stage}_complete_member_registry.json"
    member_receipt = _immutable_write(
        authority_dir / member_name, canonical_json_bytes(member_registry)
    )
    authority = with_self_hash(
        {
            "schema_version": "stage21_accepted_stage_authority_v3",
            "stage": stage,
            "stage_number": stage_number,
            "status": "ACCEPTED_SCIENCE_AUTHORITY",
            "complete_member_authority": True,
            "root_count": registry["root_count"],
            "accepted_root_count": registry["root_count"],
            "root_registry_semantic_sha256": registry["semantic_sha256"],
            "execution_manifest_self_sha256": execution["self_sha256"],
            **source_hashes,
            "qualification_semantic_sha256": adjudication["self_sha256"],
            "policy_bank": {
                "filename": policy_bank_name,
                "file_sha256": policy_bank_receipt["sha256"],
                "self_sha256": policy_bank_self_sha256,
            },
            "continuation": continuation,
            "continuation_semantic_sha256": continuation_manifest["self_sha256"]
            if continuation_manifest is not None
            else None,
            "complete_member_registry": {
                "filename": member_name,
                "file_sha256": member_receipt["sha256"],
                "self_sha256": member_registry["self_sha256"],
            },
            "manual_science_authorization_interlock_present": False,
            "teacher_eligible": True,
            "live_or_production_promotion_authorized": False,
            "validated_producer_archive_set_self_sha256": (
                validated_producers.document["self_sha256"]
            ),
            "installed_result_validation_receipt_self_sha256": (
                installed_aggregate_result.document["self_sha256"]
            ),
            "authority_local_disk_preflight": authority_local_preflight,
            "authority_accumulator_admission": authority_accumulator_admission,
            "river_policy_accumulator_disk_quota_receipt": (
                river_policy_accumulator_disk_quota_receipt
            ),
            "policy_bank_stream_preflight": policy_bank_stream_preflight,
        }
    )
    authority_name = f"stage{stage_number}_{stage}_accepted_authority.json"
    authority_receipt = _immutable_write(
        authority_dir / authority_name, canonical_json_bytes(authority)
    )
    result = with_self_hash(
        {
            **adjudication,
            "status": "ACCEPTED_AUTHORITY_PUBLISHED_LAST",
            "authority": authority_receipt,
            "authority_self_sha256": authority["self_sha256"],
            "policy_bank": policy_bank_receipt,
            "member_registry": member_receipt,
            "continuation": continuation,
            "validated_producer_archive_set": validated_producers.document,
            "installed_result_validation_receipt": (
                installed_aggregate_result.document
            ),
            "authority_local_disk_preflight": authority_local_preflight,
            "authority_accumulator_admission": authority_accumulator_admission,
            "river_policy_accumulator_disk_quota_receipt": (
                river_policy_accumulator_disk_quota_receipt
            ),
            "policy_bank_stream_preflight": policy_bank_stream_preflight,
        }
    )
    emit("authority_published_last", authority_self_sha256=authority["self_sha256"])
    cleanup_policy_spool()
    return result
