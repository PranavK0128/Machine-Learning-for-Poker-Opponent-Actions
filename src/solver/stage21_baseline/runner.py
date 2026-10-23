from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    InstalledResultValidationReceipt,
    ValidatedProducerArchiveSet,
    build_and_publish_result_archive,
    file_identity,
    local_disk_preflight,
    materialize_local_archive_lane_group,
    stage_validated_result_archive,
)
from .chance import legal_flops_for_deal, legal_next_cards
from .common import (
    APPROXIMATE_ALLOWED_TIERS,
    CROSSPLAY_DEALS_PER_ASSIGNMENT,
    EVALUATION_ALLOWED_DEALS_PER_FAMILY_ROLE,
    atomic_write,
    canonical_json_bytes,
    release_runtime_caches,
    require_ephemeral_local_path,
    sha256_bytes,
    sha256_json,
    validate_self_hash,
    with_self_hash,
)
from .continuation import (
    ContinuationAuthority,
    PolicyRolloutContinuationAuthority,
    load_continuation_authority,
)
from .crossplay import build_root_crossplay_evidence
from .evaluation import (
    EVALUATION_CONTRACT,
    adjudicate_lane,
    evaluate_policy_crossplay,
    evaluate_policy_panels,
)
from .exact import ProjectedDealSampler
from .game import (
    advance_flop,
    advance_public_card,
    state_from_root,
    terminal_utilities,
)
from .ranges import build_joint_sampler, build_player_range
from .policy_store import open_staged_policy
from .legacy_results import (
    load_legacy_completed_result_contract,
    validate_legacy_completed_result_contract_target,
    validate_legacy_result_execution_identity_receipt,
    validate_legacy_result_lane_document,
)
from .restricted_oracle import IndependentRestrictedProfileOracle
from .restricted_solver import FullChanceRestrictedCFR
from .runtime import (
    _release_process_memory,
    controller_memory_observation,
    lane_progress_task_defaults,
    parse_requested_workers,
    run_lane_queue,
)
from .solver import ExternalSamplingSolver


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_map(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(root["root_id"]): root for root in registry["roots"]}


def _checkpoint_path(lane_root: Path, lane_id: str, suffix: str = ".json") -> Path:
    directory = lane_root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{lane_id}{suffix}"


def _bonferroni_two_sided_z(familywise_alpha: float, comparisons: int) -> float:
    if not 0.0 < familywise_alpha < 1.0 or comparisons <= 0:
        raise ValueError("cross-play confidence inputs are invalid")
    return NormalDist().inv_cdf(
        1.0 - familywise_alpha / (2.0 * comparisons)
    )


def _write_lane_progress(
    progress_path: Path | None,
    task: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    if progress_path is None:
        return
    try:
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        previous = {}
    row = {
        **lane_progress_task_defaults(task),
        **previous,
        **payload,
        "schema_version": "stage21_lane_progress_v2",
        "lane_id": task["lane_id"],
        "root_id": task["root_id"],
        "algorithm": task.get("algorithm"),
        "solver_continuation_semantic_sha256": task.get(
            "solver_continuation_semantic_sha256"
        ),
        "task_semantic_sha256": task.get("task_semantic_sha256"),
        "seed": task["seed"],
        "updated_unix_seconds": time.time(),
    }
    atomic_write(progress_path, canonical_json_bytes(row))


def _run_solver_with_rotating_checkpoint(
    *,
    solver: Any,
    task: Mapping[str, Any],
    checkpoint: Path,
    target: int,
    interval: int,
    progress_path: Path | None,
    qualification_look_iterations: Sequence[int] = (),
    qualification_look_callback: Callable[[Any, int], Mapping[str, Any]]
    | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Mapping[str, float]],
    dict[str, Any],
]:
    if target < 0 or interval <= 0:
        raise ValueError("checkpoint target/interval is invalid")
    resumed_iteration = int(solver.iteration)
    if resumed_iteration > target:
        raise ValueError("checkpoint iteration exceeds lane target")
    saved_iterations: list[int] = []
    checkpoint_publications: list[dict[str, Any]] = []
    qualification_look_receipts: list[dict[str, Any]] = []
    normalized_looks = tuple(
        sorted(
            {
                int(value)
                for value in qualification_look_iterations
                if 0 < int(value) <= target
            }
        )
    )
    if normalized_looks and qualification_look_callback is None:
        raise ValueError("qualification looks require a callback")
    last_checkpoint_iteration = resumed_iteration if checkpoint.is_file() else 0
    started = time.monotonic()
    resumed_qualification_look_rechecked = False

    def emit(row: Mapping[str, Any]) -> None:
        current = int(row.get("iteration", solver.iteration))
        elapsed = max(1e-9, time.monotonic() - started)
        completed_since_resume = max(0, current - resumed_iteration)
        remaining = max(0, target - current)
        rate = completed_since_resume / elapsed
        _write_lane_progress(
            progress_path,
            task,
            {
                **dict(row),
                "status": "RUNNING",
                "current_iteration": current,
                "target_iterations": target,
                "percent_complete": 80.0 * current / max(1, target),
                "lane_phase": "solver",
                "iterations_per_second_since_resume": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "last_checkpoint_iteration": last_checkpoint_iteration,
                "checkpoint_interval_iterations": interval,
            },
        )

    def evaluate_qualification_look(
        iteration: int, *, trigger: str
    ) -> bool:
        if qualification_look_callback is None:
            raise RuntimeError("qualification look callback is absent")
        look = dict(qualification_look_callback(solver, iteration))
        if not isinstance(look.get("qualified"), bool):
            raise RuntimeError(
                "qualification look callback lacks a boolean disposition"
            )
        qualification_look_receipts.append(
            {
                "iteration": iteration,
                "trigger": trigger,
                "qualified": bool(look["qualified"]),
                "target_tier": look.get("target_tier"),
                "evaluation_self_sha256": look.get(
                    "evaluation_self_sha256"
                ),
                "adjudication_self_sha256": look.get(
                    "adjudication_self_sha256"
                ),
                "failed_checks": list(look.get("failed_checks", ())),
            }
        )
        emit(
            {
                "event": "qualification_look_adjudicated",
                "iteration": iteration,
                "qualification_look_trigger": trigger,
                "qualification_look_qualified": bool(look["qualified"]),
                "qualification_look_target_tier": look.get("target_tier"),
                "qualification_look_failed_checks": list(
                    look.get("failed_checks", ())
                ),
            }
        )
        return bool(look["qualified"])

    emit({"event": "lane_started_or_resumed", "iteration": resumed_iteration})
    stopped_after_qualification = False
    if (
        checkpoint.is_file()
        and resumed_iteration in normalized_looks
        and qualification_look_callback is not None
    ):
        # A runtime can be interrupted after the durable checkpoint but before
        # its evaluation is durably returned.  Re-evaluate that exact policy
        # before performing another solver iteration.  The held-out namespace
        # is deterministic for the preregistered look, so this neither changes
        # the solver state nor creates an unregistered extra scientific look.
        resumed_qualification_look_rechecked = True
        stopped_after_qualification = evaluate_qualification_look(
            resumed_iteration,
            trigger="resumed_exact_qualification_checkpoint",
        )
    while int(solver.iteration) < target and not stopped_after_qualification:
        future_looks = [
            value for value in normalized_looks if value > int(solver.iteration)
        ]
        next_boundary = min(
            target,
            ((int(solver.iteration) // interval) + 1) * interval,
            min(future_looks) if future_looks else target,
        )
        solver.run(
            next_boundary,
            progress_callback=emit,
            progress_interval_seconds=60.0,
            finalize_result=False,
        )
        receipt = solver.save_checkpoint(checkpoint)
        if int(receipt.get("checkpoint_iteration", -1)) != int(solver.iteration):
            raise RuntimeError("checkpoint iteration readback mismatch")
        if (
            receipt.get("independent_stream_file_identity_verified") is not True
            or receipt.get("independent_stream_self_hash_verified") is not True
        ):
            raise RuntimeError("checkpoint independent stream readback failed")
        last_checkpoint_iteration = int(solver.iteration)
        saved_iterations.append(last_checkpoint_iteration)
        checkpoint_publications.append(dict(receipt))
        emit(
            {
                "event": "checkpoint_saved_readback_verified",
                "iteration": last_checkpoint_iteration,
                "checkpoint_sha256": receipt["sha256"],
                "checkpoint_bytes": receipt["size"],
                "checkpoint_generations_retained": 1,
                "superseded_checkpoint_removed_by_atomic_replace": True,
            }
        )
        if (
            last_checkpoint_iteration in normalized_looks
            and qualification_look_callback is not None
        ):
            if evaluate_qualification_look(
                last_checkpoint_iteration,
                trigger="newly_saved_exact_qualification_checkpoint",
            ):
                stopped_after_qualification = True
                break
    finalizer = getattr(solver, "finalize_policy_and_release", None)
    if callable(finalizer):
        if "policy_store_path" in inspect.signature(finalizer).parameters:
            policy_store_root = (
                Path(str(task["local_completion_spool_root"]))
                / "disk_policies"
            )
            policy_store_root.mkdir(parents=True, exist_ok=True)
            solver_result, policy, training_state_release = finalizer(
                policy_store_path=(
                    policy_store_root / f"{task['lane_id']}.sqlite"
                )
            )
        else:
            solver_result, policy, training_state_release = finalizer()
    else:
        # Kept only for bounded test doubles and compatible external callers.
        # Every production solver implements the progressive release path.
        policy = solver.policy()
        solver_result = solver.result(materialized_policy=policy)
        release = getattr(solver, "release_training_state", None)
        training_state_release = (
            release()
            if callable(release)
            else {
                "schema_version": "stage21_solver_training_state_release_v1",
                "test_double_without_training_state": True,
                "policy_and_result_retained_outside_solver": True,
            }
        )
    return solver_result, {
        "schema_version": "stage21_rotating_checkpoint_lifecycle_v2",
        "checkpoint_path": str(checkpoint),
        "resumed_iteration": resumed_iteration,
        "requested_target_iteration": target,
        "target_iteration": int(solver_result["iterations"]),
        "effective_target_iteration": int(solver_result["iterations"]),
        "stopped_early_after_qualification": int(solver_result["iterations"])
        < target,
        "qualification_look_iterations": list(normalized_looks),
        "qualification_look_receipts": qualification_look_receipts,
        "resumed_qualification_look_rechecked_before_more_solving": (
            resumed_qualification_look_rechecked
        ),
        "checkpoint_interval_iterations": interval,
        "saved_iterations_this_process": saved_iterations,
        "checkpoint_generations_retained": 1,
        "superseded_checkpoint_removed_by_atomic_replace": True,
        "final_checkpoint_readback_verified": checkpoint.is_file(),
        "checkpoint_publications": checkpoint_publications,
        "checkpoint_serialization_seconds": math.fsum(
            float(row.get("serialization_seconds", 0.0))
            for row in checkpoint_publications
        ),
        "checkpoint_fsync_seconds": math.fsum(
            float(row.get("fsync_seconds", 0.0))
            for row in checkpoint_publications
        ),
        "checkpoint_publication_seconds": math.fsum(
            float(row.get("atomic_publication_seconds", 0.0))
            for row in checkpoint_publications
        ),
        "checkpoint_readback_seconds": math.fsum(
            float(row.get("independent_stream_readback_seconds", 0.0))
            for row in checkpoint_publications
        ),
        "checkpoint_io_queue_wait_seconds": math.fsum(
            float(row.get("io_admission_queue_wait_seconds", 0.0))
            for row in checkpoint_publications
        ),
        "checkpoint_boundary_result_finalizations": 0,
        "final_policy_materializations": 1,
        "final_policy_hash_computations": 1,
    }, policy, training_state_release


def verify_dependency_authority(
    stage: str,
    stage_drive_root: Path,
    dependency_contract: Mapping[str, Any],
    *,
    staging_root: Path | None = None,
    dependency_authority_raw: bytes | None = None,
    continuation_preflight_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if dependency_contract.get("required") is not True:
        return with_self_hash(
            {
                "schema_version": "stage21_dependency_read_receipt_v2",
                "stage": stage,
                "required": False,
                "status": "NOT_APPLICABLE",
            }
        )
    relative = Path(str(dependency_contract["authority_relative_to_current_stage_root"]))
    authority_path = (stage_drive_root / relative).resolve()
    if not authority_path.is_file():
        raise RuntimeError(
            f"{stage} is correctly held: predecessor authority is missing at {authority_path}"
        )
    raw = (
        bytes(dependency_authority_raw)
        if dependency_authority_raw is not None
        else authority_path.read_bytes()
    )
    authority = json.loads(raw)
    validate_self_hash(authority)
    if authority.get("status") != "ACCEPTED_SCIENCE_AUTHORITY":
        raise RuntimeError(f"{stage} predecessor authority is not scientifically accepted")
    if authority.get("stage") != dependency_contract["predecessor_stage"]:
        raise RuntimeError(f"{stage} predecessor stage binding mismatch")
    if authority.get("complete_member_authority") is not True:
        raise RuntimeError(f"{stage} predecessor member authority is incomplete")
    required_hashes = dependency_contract.get("required_semantic_hash_fields", ())
    missing = [name for name in required_hashes if not authority.get(name)]
    if missing:
        raise RuntimeError(f"{stage} predecessor authority missing hashes: {missing}")
    for name, expected in dependency_contract.get(
        "expected_source_semantic_hashes", {}
    ).items():
        if authority.get(name) != expected:
            raise RuntimeError(
                f"{stage} predecessor authority uses incompatible {name}"
            )
    continuation = authority.get("continuation")
    continuation_receipt = None
    if dependency_contract.get("continuation_required") is True:
        if not isinstance(continuation, Mapping):
            raise RuntimeError(f"{stage} predecessor continuation binding is absent")
        manifest_path = authority_path.parent / str(continuation["manifest_filename"])
        payload_name = continuation.get("payload_filename") or continuation.get(
            "rows_filename"
        )
        if not payload_name:
            raise RuntimeError(f"{stage} predecessor continuation payload binding is absent")
        payload_path = authority_path.parent / str(payload_name)
        source_manifest_path = manifest_path
        source_payload_path = payload_path
        staging_receipt = None
        if staging_root is not None:
            staged = _stage_continuation_tree(
                source_model_root=stage_drive_root.parent,
                primary_manifest_path=manifest_path,
                primary_payload_path=payload_path,
                staging_root=staging_root,
                preflight_manifest_bytes_by_path=(
                    continuation_preflight_inventory.get(
                        "manifest_bytes_by_resolved_path", {}
                    )
                    if continuation_preflight_inventory is not None
                    else None
                ),
                preflight_manifest_identities_by_path=(
                    continuation_preflight_inventory.get(
                        "manifest_identities_by_resolved_path", {}
                    )
                    if continuation_preflight_inventory is not None
                    else None
                ),
            )
            manifest_path = Path(staged["primary_manifest_path"])
            payload_path = Path(staged["primary_payload_path"])
            loaded = staged["loaded"]
            staging_receipt = staged["receipt"]
        else:
            loaded = load_continuation_authority(manifest_path, payload_path)
        if continuation.get("manifest_self_sha256") not in {
            None,
            loaded.manifest["self_sha256"],
        }:
            raise RuntimeError(f"{stage} predecessor continuation manifest binding mismatch")
        if staging_receipt is not None:
            staged_source_identities = {
                str(item["source_relative_path"]): item
                for item in staging_receipt["files"]
            }
            manifest_relative = source_manifest_path.resolve().relative_to(
                stage_drive_root.parent.resolve()
            ).as_posix()
            payload_relative = source_payload_path.resolve().relative_to(
                stage_drive_root.parent.resolve()
            ).as_posix()
            manifest_file_sha256 = staged_source_identities[manifest_relative][
                "sha256"
            ]
            payload_file_sha256 = staged_source_identities[payload_relative][
                "sha256"
            ]
        else:
            manifest_file_sha256 = file_identity(manifest_path)["sha256"]
            payload_file_sha256 = file_identity(payload_path)["sha256"]
        if continuation.get("manifest_file_sha256") not in {
            None,
            manifest_file_sha256,
        }:
            raise RuntimeError(f"{stage} predecessor continuation file identity mismatch")
        continuation_receipt = {
            "manifest_path": str(manifest_path),
            "payload_path": str(payload_path),
            "source_manifest_path": str(source_manifest_path),
            "source_payload_path": str(source_payload_path),
            "manifest_self_sha256": loaded.manifest["self_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "payload_file_sha256": payload_file_sha256,
            "payload_self_sha256": getattr(loaded, "bank", {}).get(
                "self_sha256"
            )
            if isinstance(getattr(loaded, "bank", None), Mapping)
            else None,
            "authority_kind": loaded.manifest.get(
                "authority_kind", "exact_hand_specific_rows"
            ),
            "local_staging": staging_receipt,
        }
    return with_self_hash(
        {
            "schema_version": "stage21_dependency_read_receipt_v2",
            "stage": stage,
            "required": True,
            "predecessor_stage": authority["stage"],
            "authority_path": str(authority_path),
            "authority_sha256": sha256_bytes(raw),
            "authority_self_sha256": authority["self_sha256"],
            "continuation": continuation_receipt,
            "status": "ACCEPTED_AND_READBACK_VERIFIED",
        }
    )


def _stage_continuation_tree(
    *,
    source_model_root: Path,
    primary_manifest_path: Path,
    primary_payload_path: Path,
    staging_root: Path,
    preflight_manifest_bytes_by_path: Mapping[str, bytes] | None = None,
    preflight_manifest_identities_by_path: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify once and mirror a continuation tree to local runtime scratch."""

    source_model_root = source_model_root.resolve()
    staging_root = staging_root.resolve()
    expected_parent = staging_root.parent.resolve()
    if staging_root.name != "continuation_staging":
        raise RuntimeError("unsafe continuation staging directory name")
    if staging_root.exists():
        if staging_root.parent.resolve() != expected_parent:
            raise RuntimeError("unsafe continuation staging cleanup target")
        shutil.rmtree(staging_root)
    staged_model_root = staging_root / "Poker Assistant Model"
    staged_model_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    visited: set[Path] = set()

    def destination(source: Path) -> Path:
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(source_model_root)
        except ValueError as error:
            raise RuntimeError("continuation source escaped the model Drive root") from error
        return staged_model_root / relative

    def copy_verified(
        source: Path,
        *,
        capture_source_bytes: bool = False,
        source_raw: bytes | None = None,
        expected_source_identity: Mapping[str, Any] | None = None,
    ) -> tuple[Path, bytes | None, dict[str, Any]]:
        target = destination(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        captured = bytearray() if capture_source_bytes else None
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                if source_raw is None:
                    input_handle = source.open("rb")
                    blocks = iter(lambda: input_handle.read(1024 * 1024), b"")
                else:
                    input_handle = None
                    source_view = memoryview(source_raw)
                    blocks = (
                        source_view[offset : offset + 1024 * 1024]
                        for offset in range(0, len(source_view), 1024 * 1024)
                    )
                try:
                    for block in blocks:
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                        if captured is not None:
                            captured.extend(block)
                finally:
                    if input_handle is not None:
                        input_handle.close()
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        source_identity = {"size": size, "sha256": digest.hexdigest()}
        if (
            expected_source_identity is not None
            and source_identity != dict(expected_source_identity)
        ):
            raise RuntimeError(
                "continuation staging bytes differ from the preflight source identity"
            )
        row = {
            "source_relative_path": source.resolve().relative_to(
                source_model_root
            ).as_posix(),
            "size": source_identity["size"],
            "sha256": source_identity["sha256"],
            "source_read_verified": True,
            "source_stream_pass_count": 1,
            "staged_readback_verified": False,
        }
        receipts.append(row)
        return target, bytes(captured) if captured is not None else None, row

    def stage_pair(manifest_source: Path, payload_source: Path) -> tuple[Path, Path]:
        resolved_manifest = manifest_source.resolve()
        if resolved_manifest in visited:
            raise RuntimeError("continuation staging detected a dependency cycle")
        visited.add(resolved_manifest)
        manifest_source_raw = None
        manifest_source_identity = None
        if preflight_manifest_bytes_by_path is not None:
            manifest_source_raw = preflight_manifest_bytes_by_path.get(
                str(resolved_manifest)
            )
            if manifest_source_raw is None:
                raise RuntimeError(
                    "continuation staging lacks its preflight-captured manifest"
                )
            if preflight_manifest_identities_by_path is None:
                raise RuntimeError(
                    "continuation staging lacks preflight manifest identities"
                )
            manifest_source_identity = preflight_manifest_identities_by_path.get(
                str(resolved_manifest)
            )
            if manifest_source_identity is None:
                raise RuntimeError(
                    "continuation staging lacks its preflight manifest identity"
                )
        staged_manifest, manifest_raw, _ = copy_verified(
            manifest_source,
            capture_source_bytes=True,
            source_raw=manifest_source_raw,
            expected_source_identity=manifest_source_identity,
        )
        if manifest_raw is None:
            raise AssertionError("continuation manifest capture is missing")
        manifest = json.loads(manifest_raw)
        validate_self_hash(manifest)
        staged_payload, _, payload_row = copy_verified(payload_source)
        kind = manifest.get("authority_kind", "exact_hand_specific_rows")
        expected_payload_sha = manifest.get(
            "policy_bank_file_sha256"
            if kind == "policy_rollout"
            else "continuation_file_sha256"
        )
        if expected_payload_sha is not None and payload_row["sha256"] != expected_payload_sha:
            raise RuntimeError("continuation staging payload source hash mismatch")
        binding = manifest.get("downstream")
        if binding is not None:
            stage_pair(
                manifest_source.parent / str(binding["manifest_relative_path"]),
                manifest_source.parent / str(binding["policy_bank_relative_path"]),
            )
        return staged_manifest, staged_payload

    staged_manifest, staged_payload = stage_pair(
        primary_manifest_path, primary_payload_path
    )
    staged_loaded = load_continuation_authority(staged_manifest, staged_payload)
    for row in receipts:
        row["staged_readback_verified"] = True
    receipt = with_self_hash(
        {
            "schema_version": "stage21_local_continuation_staging_receipt_v2",
            "primary_manifest_self_sha256": staged_loaded.manifest["self_sha256"],
            "file_count": len(receipts),
            "files": sorted(receipts, key=lambda row: row["source_relative_path"]),
            "source_files_verified_once_by_coordinator": True,
            "source_stream_passes_per_file": 1,
            "staged_files_read_and_loaded_once": True,
            "worker_tasks_use_local_staged_paths": True,
            "staging_is_not_scientific_authority": True,
        }
    )
    return {
        "primary_manifest_path": str(staged_manifest),
        "primary_payload_path": str(staged_payload),
        "loaded": staged_loaded,
        "receipt": receipt,
    }


def _continuation_tree_preflight_inventory(
    *,
    source_model_root: Path,
    primary_manifest_path: Path,
    primary_payload_path: Path,
) -> dict[str, Any]:
    """Capture manifests once and count every continuation source file."""

    source_model_root = source_model_root.resolve()
    visited_manifests: set[Path] = set()
    visited_files: set[Path] = set()
    manifest_bytes_by_resolved_path: dict[str, bytes] = {}
    manifest_identities_by_resolved_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    def include(path: Path) -> Path:
        nonlocal total_bytes
        resolved = path.resolve()
        try:
            resolved.relative_to(source_model_root)
        except ValueError as error:
            raise RuntimeError("continuation preflight source escaped model root") from error
        if resolved not in visited_files:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            total_bytes += int(resolved.stat().st_size)
            visited_files.add(resolved)
        return resolved

    def visit(manifest_path: Path, payload_path: Path) -> None:
        resolved_manifest = include(manifest_path)
        include(payload_path)
        if resolved_manifest in visited_manifests:
            raise RuntimeError("continuation preflight detected a dependency cycle")
        visited_manifests.add(resolved_manifest)
        captured = bytearray()
        digest = hashlib.sha256()
        captured_size = 0
        with resolved_manifest.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                captured.extend(block)
                digest.update(block)
                captured_size += len(block)
        manifest_raw = bytes(captured)
        if captured_size != int(resolved_manifest.stat().st_size):
            raise RuntimeError("continuation manifest changed during preflight")
        manifest = json.loads(manifest_raw)
        validate_self_hash(manifest)
        manifest_bytes_by_resolved_path[str(resolved_manifest)] = manifest_raw
        manifest_identities_by_resolved_path[str(resolved_manifest)] = {
            "size": captured_size,
            "sha256": digest.hexdigest(),
        }
        downstream = manifest.get("downstream")
        if downstream is not None:
            if not isinstance(downstream, Mapping):
                raise RuntimeError("continuation downstream binding is not an object")
            visit(
                resolved_manifest.parent
                / str(downstream["manifest_relative_path"]),
                resolved_manifest.parent
                / str(downstream["policy_bank_relative_path"]),
            )

    visit(primary_manifest_path, primary_payload_path)
    return {
        "total_bytes": total_bytes,
        "manifest_bytes_by_resolved_path": manifest_bytes_by_resolved_path,
        "manifest_identities_by_resolved_path": (
            manifest_identities_by_resolved_path
        ),
        "manifest_source_stream_pass_count": 1,
        "payload_source_preflight_mode": "stat_only_then_one_staging_stream_pass",
    }


def _load_continuation(
    task: Mapping[str, Any],
) -> ContinuationAuthority | PolicyRolloutContinuationAuthority | None:
    manifest_path = task.get("continuation_manifest_path")
    payload_path = task.get("continuation_payload_path")
    if manifest_path is None and payload_path is None:
        return None
    if not manifest_path or not payload_path:
        raise RuntimeError("partial continuation binding")
    return load_continuation_authority(
        Path(str(manifest_path)), Path(str(payload_path))
    )


def _generic_forced_lane(
    task: Mapping[str, Any], progress_path: Path | None = None
) -> dict[str, Any]:
    root = task["root"]
    state = state_from_root(root)
    sampler = build_joint_sampler(root, state)
    expected_ranges = task.get("execution_identity", {}).get(
        "range_identity_sha256_by_player", {}
    )
    if expected_ranges != {
        "0": sampler.first.identity_sha256,
        "1": sampler.second.identity_sha256,
    }:
        raise RuntimeError("forced task range identity mismatch")
    expected_marginals = task.get("execution_identity", {}).get(
        "range_marginal_identity_sha256", {}
    )
    if expected_marginals != {
        "declared_preconditioning": (
            sampler.declared_preconditioning_marginal_identity_sha256
        ),
        "sampled_blocker_conditioned": (
            sampler.sampled_blocker_conditioned_marginal_identity_sha256
        ),
    }:
        raise RuntimeError("forced task marginal identity mismatch")
    rng = random.Random(
        int(
            sha256_json(
                [
                    task["solver_continuation_semantic_sha256"],
                    task["seed"],
                    "private_deal",
                ]
            )[:16],
            16,
        )
    )
    continuation = _load_continuation(task)
    strategy_context = root.get("range_contract")
    deal = sampler.sample(rng)
    player_values: tuple[list[float], list[float]] = ([], [])
    last_progress = 0.0

    def emit_forced_progress(completed: int, total: int, *, force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 60.0:
            return
        _write_lane_progress(
            progress_path,
            task,
            {
                "event": "forced_runout_progress",
                "status": "EVALUATING_FORCED_RUNOUTS",
                "lane_phase": "forced_runout_evaluation",
                "completed_runouts": completed,
                "total_runouts": total,
                "percent_complete": 95.0 * completed / max(1, total),
                "current_iteration": 0,
                "target_iterations": 0,
                "last_checkpoint_iteration": 0,
            },
        )
        last_progress = now

    def append_pair(pair: tuple[float, float]) -> None:
        if abs(math.fsum(pair)) > 1e-9:
            raise AssertionError("forced evaluation sample is not jointly zero sum")
        player_values[0].append(pair[0])
        player_values[1].append(pair[1])
    if state.street == "turn":
        cards = legal_next_cards(state.board, deal)
        emit_forced_progress(0, len(cards), force=True)
        for index, card in enumerate(cards, start=1):
            child = advance_public_card(state, card)
            append_pair(terminal_utilities(child, deal))
            emit_forced_progress(index, len(cards))
        physical_count = len(cards)
        expected_count = 44
        coverage = "all_legal_rivers_for_fixed_deal"
    elif state.street == "flop":
        if continuation is None:
            raise RuntimeError("forced Flop evaluation lacks accepted Turn continuation")
        cards = legal_next_cards(state.board, deal)
        emit_forced_progress(0, len(cards), force=True)
        for index, card in enumerate(cards, start=1):
            child = advance_public_card(state, card)
            append_pair(
                continuation.resolve_preregistered_mixture(
                    child, deal, context=strategy_context
                )
            )
            emit_forced_progress(index, len(cards))
        physical_count = len(cards)
        expected_count = 45
        coverage = "all_legal_turn_leaves_for_fixed_deal_with_accepted_turn_continuation"
    elif state.street == "preflop":
        if continuation is None:
            raise RuntimeError("forced Preflop evaluation lacks accepted Flop continuation")
        flops = legal_flops_for_deal(deal)
        emit_forced_progress(0, len(flops), force=True)
        for index, flop in enumerate(flops, start=1):
            child = advance_flop(state, flop)
            append_pair(
                continuation.resolve_preregistered_mixture(
                    child, deal, context=strategy_context
                )
            )
            emit_forced_progress(index, len(flops))
        physical_count = len(flops)
        expected_count = 17_296
        coverage = "all_legal_flop_leaves_for_fixed_deal_with_accepted_flop_continuation"
    elif state.street == "river":
        forced_deals = int(task.get("forced_deals", 1024))
        emit_forced_progress(0, forced_deals, force=True)
        for index in range(1, forced_deals + 1):
            sampled = sampler.sample(rng)
            append_pair(terminal_utilities(state, sampled))
            emit_forced_progress(index, forced_deals)
        physical_count = 0
        expected_count = 0
        coverage = "terminal_showdown_range_panel_no_future_public_runout"
    else:
        raise ValueError("generic forced lane received unsupported street")
    if physical_count != expected_count:
        raise AssertionError("forced lane physical runout coverage mismatch")
    if not player_values[0]:
        raise AssertionError("forced evaluation is empty")
    emit_forced_progress(len(player_values[0]), len(player_values[0]), force=True)
    continuation_diagnostics = (
        continuation.diagnostics() if continuation is not None else None
    )
    continuation_ok = (
        continuation_diagnostics is None
        or continuation_diagnostics.get("within_all_backoff_bounds") is True
    )
    continuation_cache_release = (
        continuation.release_runtime_state()
        if continuation is not None
        and hasattr(continuation, "release_runtime_state")
        else None
    )
    cache_release = release_runtime_caches()
    return {
        "lane_id": task["lane_id"],
        "root_id": task["root_id"],
        "algorithm": None,
        "seed": task["seed"],
        "solver_continuation_semantic_sha256": task[
            "solver_continuation_semantic_sha256"
        ],
        "task_semantic_sha256": task["task_semantic_sha256"],
        "execution_identity": task["execution_identity"],
        "root_semantic_sha256": root["root_semantic_sha256"],
        "role": "forced_runout_evaluation",
        "solver_iterations": 0,
        "physical_runout_count": physical_count,
        "expected_physical_runout_count": expected_count,
        "coverage": coverage,
        "mean_utility_chips": [
            math.fsum(player_values[player]) / len(player_values[player])
            for player in range(2)
        ],
        "per_sample_zero_sum_validated": True,
        "continuation_diagnostics": continuation_diagnostics,
        "qualified": continuation_ok,
        "artifact_hash_valid": True,
        "independent_confirmation_panel": True,
        "runtime_cache_release": cache_release,
        "continuation_cache_release": continuation_cache_release,
        "joint_range_sampler_diagnostics": sampler.diagnostics(),
    }


def _generic_solver_lane(
    task: Mapping[str, Any],
    lane_root: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    root = task["root"]
    state = state_from_root(root)
    sampler = build_joint_sampler(root, state)
    expected_ranges = task.get("execution_identity", {}).get(
        "range_identity_sha256_by_player", {}
    )
    actual_ranges = {
        "0": sampler.first.identity_sha256,
        "1": sampler.second.identity_sha256,
    }
    if expected_ranges != actual_ranges:
        raise RuntimeError("task-bound range identity does not match materialized ranges")
    expected_marginals = task.get("execution_identity", {}).get(
        "range_marginal_identity_sha256", {}
    )
    if expected_marginals != {
        "declared_preconditioning": (
            sampler.declared_preconditioning_marginal_identity_sha256
        ),
        "sampled_blocker_conditioned": (
            sampler.sampled_blocker_conditioned_marginal_identity_sha256
        ),
    }:
        raise RuntimeError(
            "task-bound blocker-conditioned marginal identity does not match"
        )
    continuation = _load_continuation(task)
    if continuation is not None:
        identity = task["execution_identity"]
        if continuation.manifest.get("self_sha256") != identity.get(
            "continuation_manifest_self_sha256"
        ):
            raise RuntimeError("task-bound continuation manifest identity mismatch")
        bank = getattr(continuation, "bank", None)
        if isinstance(bank, Mapping) and bank.get("self_sha256") != identity.get(
            "continuation_payload_self_sha256"
        ):
            raise RuntimeError("task-bound continuation payload identity mismatch")
    algorithm = str(task["algorithm"])
    seed = int(task["seed"])
    target = int(task["target_iterations"])
    strategy_context = root.get("range_contract")

    def evaluate_candidate_at_look(
        *,
        candidate_policy: Mapping[str, Mapping[str, float]],
        candidate_solver_result: Mapping[str, Any],
        candidate_sampler: Any,
        exact_oracle: Mapping[str, Any] | None,
        look_index: int,
        look_count: int,
        required_iterations: int,
    ) -> dict[str, Any]:
        current_iteration = int(candidate_solver_result["iterations"])
        _write_lane_progress(
            progress_path,
            task,
            {
                "event": "qualification_look_evaluation_started",
                "status": "EVALUATING_QUALIFICATION_LOOK",
                "current_iteration": current_iteration,
                "target_iterations": target,
                "percent_complete": 80.0 * current_iteration / max(1, target),
                "last_checkpoint_iteration": current_iteration,
                "lane_phase": "evaluation",
                "qualification_look_index": look_index,
                "qualification_look_count": look_count,
                "qualification_look_percent_complete": 0.0,
                "qualification_look_trigger": "EVALUATION_IN_PROGRESS",
                "qualification_look_qualified": None,
                "qualification_look_target_tier": (
                    current_iteration
                    // int(task.get("approximate_base_iterations_per_tier", 1))
                    if root["root_kind"] == "approximate"
                    else 0
                ),
                "qualification_look_failed_checks": [],
            },
        )

        def emit_evaluation_progress(row: Mapping[str, Any]) -> None:
            completed = int(row["completed_evaluation_work_units"])
            total = int(row["total_evaluation_work_units"])
            _write_lane_progress(
                progress_path,
                task,
                {
                    "event": "evaluation_panel_progress",
                    "status": "EVALUATING_QUALIFICATION_LOOK",
                    "lane_phase": "evaluation",
                    "current_iteration": current_iteration,
                    "target_iterations": target,
                    "percent_complete": 80.0
                    * current_iteration
                    / max(1, target),
                    "qualification_look_percent_complete": (
                        100.0 * completed / max(1, total)
                    ),
                    "last_checkpoint_iteration": current_iteration,
                    "qualification_look_index": look_index,
                    "qualification_look_count": look_count,
                    **dict(row),
                },
            )

        evaluation = evaluate_policy_panels(
            root_id=root["root_id"],
            initial_state=state,
            joint_sampler=candidate_sampler,
            policy=candidate_policy,
            expected_policy_sha256=str(
                candidate_solver_result["policy_sha256"]
            ),
            base_seed=seed,
            panel_count=int(task["evaluation_panel_count"]),
            deals_per_family_role=int(
                task["evaluation_deals_per_family_role"]
            ),
            final_deals_per_family_role=int(
                task["final_holdout_deals_per_family_role"]
            ),
            optimization_stream_identity_sha256=str(
                candidate_solver_result[
                    "optimization_stream_identity_sha256"
                ]
            ),
            depth_limit=root["depth_limit"],
            continuation=continuation,
            strategy_context=strategy_context,
            exact_information_sets=root["root_kind"] == "restricted_exact",
            qualification_look_index=look_index,
            qualification_look_count=look_count,
            progress_callback=emit_evaluation_progress,
        )
        diagnostics = (
            continuation.diagnostics() if continuation is not None else None
        )
        adjudication = adjudicate_lane(
            root_kind=root["root_kind"],
            solver_result=candidate_solver_result,
            evaluation_result=evaluation,
            exact_oracle_result=exact_oracle,
            required_terminal_iterations=required_iterations,
            continuation_diagnostics=diagnostics,
            runtime_feature_board=state.board,
            runtime_feature_sampler=candidate_sampler,
            runtime_feature_restricted_projected_game=(
                root["root_kind"] == "restricted_exact"
            ),
        )
        return {
            "solver_result": dict(candidate_solver_result),
            "evaluation": evaluation,
            "adjudication": adjudication,
            "continuation_diagnostics": diagnostics,
        }

    if root["root_kind"] == "restricted_exact":
        solver = FullChanceRestrictedCFR(
            root_id=root["root_id"],
            root_semantic_sha256=root["root_semantic_sha256"],
            initial_state=state,
            first_range=sampler.first,
            second_range=sampler.second,
            algorithm=algorithm,
            evaluation_seed=seed,
            task_semantic_sha256=str(task["task_semantic_sha256"]),
            solver_continuation_semantic_sha256=str(
                task["solver_continuation_semantic_sha256"]
            ),
            combos_per_player=int(task.get("exact_combos_per_player", 4)),
            checkpoint_io_lock_path=Path(
                str(task["local_checkpoint_io_lock_path"])
            ),
            checkpoint_spool_root=Path(
                str(task["local_checkpoint_spool_root"])
            ),
        )
        checkpoint = _checkpoint_path(lane_root, str(task["lane_id"]))
        if checkpoint.is_file():
            solver.load_checkpoint(checkpoint)
        projected_deals = tuple(solver.deals)
        (
            solver_result,
            checkpoint_lifecycle,
            policy,
            training_state_release,
        ) = _run_solver_with_rotating_checkpoint(
            solver=solver,
            task=task,
            checkpoint=checkpoint,
            target=target,
            interval=int(task["checkpoint_interval_iterations"]),
            progress_path=progress_path,
        )
        evaluation_sampler = ProjectedDealSampler(projected_deals)
        solver = None
        training_memory_release = _release_process_memory()

        def emit_oracle_progress(row: Mapping[str, Any]) -> None:
            completed = int(row["completed_oracle_passes"])
            total = int(row["total_oracle_passes"])
            _write_lane_progress(
                progress_path,
                task,
                {
                    "event": "restricted_oracle_progress",
                    "status": "EVALUATING_RESTRICTED_ORACLE",
                    "lane_phase": "restricted_oracle",
                    "current_iteration": target,
                    "target_iterations": target,
                    "percent_complete": 80.0 + 2.0 * completed / max(1, total),
                    "last_checkpoint_iteration": target,
                    **dict(row),
                },
            )

        oracle = IndependentRestrictedProfileOracle(
            state, projected_deals
        ).evaluate(policy, progress_callback=emit_oracle_progress)
        selected_look_evidence = evaluate_candidate_at_look(
            candidate_policy=policy,
            candidate_solver_result=solver_result,
            candidate_sampler=evaluation_sampler,
            exact_oracle=oracle,
            look_index=1,
            look_count=1,
            required_iterations=target,
        )
    else:
        solver = ExternalSamplingSolver(
            root_id=root["root_id"],
            root_semantic_sha256=root["root_semantic_sha256"],
            initial_state=state,
            joint_sampler=sampler,
            algorithm=algorithm,
            seed=seed,
            task_semantic_sha256=str(task["task_semantic_sha256"]),
            solver_continuation_semantic_sha256=str(
                task["solver_continuation_semantic_sha256"]
            ),
            depth_limit=root["depth_limit"],
            continuation_authority=continuation,
            strategy_context=strategy_context,
            checkpoint_io_lock_path=Path(
                str(task["local_checkpoint_io_lock_path"])
            ),
            checkpoint_spool_root=Path(
                str(task["local_checkpoint_spool_root"])
            ),
        )
        checkpoint = _checkpoint_path(lane_root, str(task["lane_id"]))
        if checkpoint.is_file():
            solver.load_checkpoint(checkpoint)
        tier_base = int(task["approximate_base_iterations_per_tier"])
        tier_stride = int(task["approximate_qualification_tier_stride"])
        maximum_looks = int(task["maximum_approximate_qualification_looks"])
        minimum_qualification_tier = int(
            task["minimum_qualification_tier"]
        )
        look_iterations = tuple(
            tier_base * tier_stride * look_index
            for look_index in range(1, maximum_looks + 1)
            if minimum_qualification_tier
            <= tier_stride * look_index
            <= int(task["target_tier"])
        )
        if target not in look_iterations:
            raise RuntimeError(
                "approximate target is not a preregistered qualification look"
            )
        selected_look_evidence: dict[str, Any] | None = None

        def evaluate_qualification_look(
            live_solver: ExternalSamplingSolver,
            iteration: int,
        ) -> dict[str, Any]:
            nonlocal selected_look_evidence
            tier = iteration // tier_base
            look_index = tier // tier_stride
            snapshot_path = (
                Path(str(task["local_completion_spool_root"]))
                / "qualification_look_policies"
                / f"{task['lane_id']}__tier_{tier}.sqlite"
            )
            snapshot_result, snapshot_policy = (
                live_solver.snapshot_policy_to_disk(snapshot_path)
            )
            try:
                evidence = evaluate_candidate_at_look(
                    candidate_policy=snapshot_policy,
                    candidate_solver_result=snapshot_result,
                    candidate_sampler=sampler,
                    exact_oracle=None,
                    look_index=look_index,
                    look_count=maximum_looks,
                    required_iterations=iteration,
                )
            finally:
                snapshot_policy.cleanup()
            adjudication = evidence["adjudication"]
            evaluation_self_sha256 = str(
                evidence["evaluation"]["self_sha256"]
            )
            adjudication_self_sha256 = str(adjudication["self_sha256"])
            qualified = adjudication["qualified"] is True
            if qualified or iteration == target:
                selected_look_evidence = evidence
            else:
                evidence = None
                _release_process_memory()
            return {
                "qualified": qualified,
                "target_tier": tier,
                "evaluation_self_sha256": evaluation_self_sha256,
                "adjudication_self_sha256": adjudication_self_sha256,
                "failed_checks": sorted(
                    name
                    for name, passed in adjudication["checks"].items()
                    if passed is not True
                ),
            }
        (
            solver_result,
            checkpoint_lifecycle,
            policy,
            training_state_release,
        ) = _run_solver_with_rotating_checkpoint(
            solver=solver,
            task=task,
            checkpoint=checkpoint,
            target=target,
            interval=int(task["checkpoint_interval_iterations"]),
            progress_path=progress_path,
            qualification_look_iterations=look_iterations,
            qualification_look_callback=evaluate_qualification_look,
        )
        if selected_look_evidence is None:
            raise RuntimeError("approximate lane lacks final qualification evidence")
        if selected_look_evidence["solver_result"] != solver_result:
            raise RuntimeError(
                "final solver result differs from qualification-look snapshot"
            )
        if (
            selected_look_evidence["evaluation"]["policy_sha256"]
            != solver_result["policy_sha256"]
        ):
            raise RuntimeError(
                "final policy differs from qualification-look policy"
            )
        oracle = None
        evaluation_sampler = sampler
        solver = None
        training_memory_release = _release_process_memory()
    evaluation = selected_look_evidence["evaluation"]
    adjudication = selected_look_evidence["adjudication"]
    continuation_diagnostics = selected_look_evidence[
        "continuation_diagnostics"
    ]
    effective_target_iterations = int(solver_result["iterations"])
    effective_target_tier = (
        effective_target_iterations
        // int(task["approximate_base_iterations_per_tier"])
        if root["root_kind"] == "approximate"
        else 0
    )
    final_failed_checks = sorted(
        name
        for name, passed in adjudication["checks"].items()
        if passed is not True
    )
    look_receipts = list(checkpoint_lifecycle.get("qualification_look_receipts", ()))
    final_look_trigger = (
        look_receipts[-1].get("trigger")
        if look_receipts
        else "FIXED_TERMINAL_EVALUATION"
    )
    continuation_cache_release = (
        continuation.release_runtime_state()
        if continuation is not None
        and hasattr(continuation, "release_runtime_state")
        else None
    )
    _write_lane_progress(
        progress_path,
        task,
        {
            "event": "lane_adjudicated_pending_durable_completion",
            "status": "QUALIFIED_PENDING_DURABLE_WRITE"
            if adjudication["qualified"]
            else "NOT_QUALIFIED_PENDING_DURABLE_WRITE",
            "current_iteration": effective_target_iterations,
            "target_iterations": target,
            "percent_complete": 99.0,
            "last_checkpoint_iteration": effective_target_iterations,
            "checkpoint_interval_iterations": int(
                task["checkpoint_interval_iterations"]
            ),
            "lane_phase": "adjudication_and_serialization",
            "requested_target_tier": int(task.get("target_tier", 0)),
            "effective_target_tier": effective_target_tier,
            "effective_target_iterations": effective_target_iterations,
            "minimum_qualification_tier": int(
                task.get("minimum_qualification_tier", 0)
            ),
            "stopped_early_after_qualification": (
                effective_target_iterations < target
            ),
            "qualification_look_index": int(
                evaluation["qualification_look_index"]
            ),
            "qualification_look_count": int(
                evaluation["qualification_look_count"]
            ),
            "qualification_look_percent_complete": 100.0,
            "qualification_look_trigger": final_look_trigger,
            "qualification_look_qualified": adjudication["qualified"],
            "qualification_look_target_tier": effective_target_tier,
            "qualification_look_failed_checks": final_failed_checks,
            "qualified": adjudication["qualified"],
            "failed_checks": final_failed_checks,
        },
    )
    cache_release = release_runtime_caches()
    return {
        "lane_id": task["lane_id"],
        "root_id": task["root_id"],
        "algorithm": algorithm,
        "seed": seed,
        "solver_continuation_semantic_sha256": task[
            "solver_continuation_semantic_sha256"
        ],
        "task_semantic_sha256": task["task_semantic_sha256"],
        "execution_identity": task["execution_identity"],
        "root_semantic_sha256": root["root_semantic_sha256"],
        "role": "solver_and_independent_evaluation",
        "requested_target_tier": int(task.get("target_tier", 0)),
        "minimum_qualification_tier": int(
            task.get("minimum_qualification_tier", 0)
        ),
        "target_tier": effective_target_tier,
        "requested_target_iterations": target,
        "target_iterations": effective_target_iterations,
        "stopped_early_after_qualification": (
            effective_target_iterations < target
        ),
        "evaluation_deals_per_family_role": int(
            task["evaluation_deals_per_family_role"]
        ),
        "solver_result": solver_result,
        "checkpoint_lifecycle": checkpoint_lifecycle,
        "policy": policy,
        "evaluation": evaluation,
        "exact_oracle": oracle,
        "continuation_diagnostics": continuation_diagnostics,
        "qualified": adjudication["qualified"],
        "adjudication": adjudication,
        "artifact_hash_valid": True,
        "independent_confirmation_panel": evaluation.get(
            "independent_confirmation_panel"
        )
        is True,
        "runtime_cache_release": cache_release,
        "solver_training_state_release": training_state_release,
        "solver_training_memory_release": training_memory_release,
        "continuation_cache_release": continuation_cache_release,
        "joint_range_sampler_diagnostics": sampler.diagnostics(),
    }


def _evaluate_aggregate_root_binding(
    *,
    task: Mapping[str, Any],
    binding: Mapping[str, Any],
    source_handles: Sequence[Mapping[str, Any]],
    legacy_result_contract: Mapping[str, Any] | None,
    continuation: Any,
    starting_staged_member_reads: int,
    emit_aggregate_progress: Callable[..., None],
) -> dict[str, Any]:
    """Validate and evaluate exactly one aggregate root in a short-lived frame."""

    staging_receipt = binding.get("producer_staging_receipt")
    if not isinstance(staging_receipt, Mapping):
        raise ValueError("aggregate root lacks coordinator staging receipt")
    validate_self_hash(staging_receipt)
    if (
        staging_receipt.get("stage") != task["stage"]
        or staging_receipt.get("token") != binding["token"]
        or staging_receipt.get("source_archive_sha256") != binding["sha256"]
        or staging_receipt.get("staged_members_independently_read_back") is not True
    ):
        raise ValueError("aggregate coordinator staging receipt mismatch")
    handles = {str(handle["lane_id"]): handle for handle in source_handles}
    if set(handles) != set(binding["lane_ids"]):
        raise ValueError("aggregate staged lane membership mismatch")
    lane_documents = []
    legacy_lane_receipts = []
    staged_member_reads = int(starting_staged_member_reads)
    for lane_id in binding["lane_ids"]:
        handle = handles[lane_id]
        staged_member_released = (
            handle.get("staged_member_released_after_policy_spool") is True
        )
        if handle.get("canonical_self_hash_verified") is not True:
            raise ValueError(f"aggregate staged member identity mismatch: {lane_id}")
        if not staged_member_released:
            staged_path = Path(str(handle["staged_path"]))
            if staged_path.stat().st_size != int(handle["size"]):
                raise ValueError(f"aggregate staged member identity mismatch: {lane_id}")
        store = handle.get("policy_store")
        if handle.get("policy_mapping_present") is True and (
            not isinstance(store, Mapping)
            or not Path(str(store.get("path"))).is_file()
            or Path(str(store["path"])).stat().st_size
            != int(store.get("sqlite_size", -1))
        ):
            raise ValueError(f"aggregate policy store disappeared: {lane_id}")
        metadata = handle.get("document_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"aggregate staged metadata is absent: {lane_id}")
        document = dict(metadata)
        if document.get("self_sha256") != handle["document_self_sha256"]:
            raise ValueError(f"aggregate staged member self binding mismatch: {lane_id}")
        policy_sha256 = handle.get("policy_sha256")
        if handle.get("policy_mapping_present") is True and not isinstance(
            policy_sha256, str
        ):
            raise ValueError(f"aggregate staged policy binding mismatch: {lane_id}")
        current_binding = (
            document.get("root_id") == binding["root_id"]
            and document.get("root_semantic_sha256")
            == binding["root"]["root_semantic_sha256"]
            and document.get("status") == "COMPLETE"
            and document.get("execution_identity") == binding["execution_identity"]
        )
        if not current_binding:
            if legacy_result_contract is None:
                raise ValueError(f"aggregate root member binding mismatch: {lane_id}")
            spec = legacy_result_contract["lane_bindings"].get(lane_id)
            if not isinstance(spec, Mapping):
                raise ValueError(f"aggregate legacy member is not contracted: {lane_id}")
            legacy_lane_receipts.append(
                validate_legacy_result_lane_document(
                    document,
                    current_lane={
                        "lane_id": lane_id,
                        "root_id": binding["root_id"],
                        "algorithm": spec.get("algorithm"),
                        "seed": spec["seed"],
                    },
                    contract=legacy_result_contract,
                    policy_sha256=(
                        str(policy_sha256) if policy_sha256 is not None else None
                    ),
                )
            )
        lane_documents.append(document)
        staged_member_reads += 1
        emit_aggregate_progress(
            "aggregate_member_read",
            aggregate_member_reads=staged_member_reads,
        )
    if len(lane_documents) != 6:
        raise ValueError("aggregate root shard must verify all six members")
    kind = str(binding["root"]["root_kind"])
    evidence_body: dict[str, Any] = {
        "schema_version": "stage21_aggregate_root_evidence_v3",
        "root_id": binding["root_id"],
        "root_kind": kind,
        "member_lane_ids": sorted(binding["lane_ids"]),
        "member_document_self_sha256s": sorted(
            document["self_sha256"] for document in lane_documents
        ),
        "transport_archive_sha256": binding["sha256"],
        "legacy_completed_result_contract_self_sha256": (
            legacy_result_contract["self_sha256"]
            if legacy_lane_receipts
            else None
        ),
        "legacy_member_count": len(legacy_lane_receipts),
        "legacy_member_receipt_self_sha256s": sorted(
            receipt["self_sha256"] for receipt in legacy_lane_receipts
        ),
        "legacy_members_preserved_without_relabelling": bool(legacy_lane_receipts),
    }
    if kind == "approximate":
        state = state_from_root(binding["root"])
        sampler = build_joint_sampler(binding["root"], state)
        pair_rows = []
        for first_index in range(len(lane_documents)):
            for second_index in range(first_index + 1, len(lane_documents)):
                first = lane_documents[first_index]
                second = lane_documents[second_index]
                pair_seed = int(
                    sha256_json(
                        [
                            task["task_semantic_sha256"],
                            binding["root_id"],
                            first["lane_id"],
                            second["lane_id"],
                        ]
                    )[:16],
                    16,
                )
                first_handle = handles[str(first["lane_id"])]
                second_handle = handles[str(second["lane_id"])]
                with open_staged_policy(
                    first_handle, cache_entries=512
                ) as first_policy, open_staged_policy(
                    second_handle, cache_entries=512
                ) as second_policy:
                    if (
                        first_policy.policy_sha256
                        != first["solver_result"]["policy_sha256"]
                        or second_policy.policy_sha256
                        != second["solver_result"]["policy_sha256"]
                    ):
                        raise ValueError("aggregate pair policy identity mismatch")
                    pair_rows.append(
                        evaluate_policy_crossplay(
                            root_id=str(binding["root_id"]),
                            initial_state=state,
                            joint_sampler=sampler,
                            first_policy=first_policy,
                            second_policy=second_policy,
                            first_member={
                                "lane_id": first["lane_id"],
                                "algorithm": first["algorithm"],
                                "seed": first["seed"],
                                "policy_sha256": first["solver_result"][
                                    "policy_sha256"
                                ],
                            },
                            second_member={
                                "lane_id": second["lane_id"],
                                "algorithm": second["algorithm"],
                                "seed": second["seed"],
                                "policy_sha256": second["solver_result"][
                                    "policy_sha256"
                                ],
                            },
                            panel_seed=pair_seed,
                            deals_per_assignment=int(
                                task["crossplay_deals_per_assignment"]
                            ),
                            depth_limit=binding["root"]["depth_limit"],
                            continuation=continuation,
                            strategy_context=binding["root"].get("range_contract"),
                            progress_callback=lambda progress_row: emit_aggregate_progress(
                                "aggregate_crossplay_progress",
                                crossplay_progress=dict(progress_row),
                            ),
                        )
                    )
        effective_stack_bb = min(state.starting_stacks) / 100.0
        evidence_body["crossplay"] = build_root_crossplay_evidence(
            root_id=str(binding["root_id"]),
            pair_results=pair_rows,
            expected_members=[
                {
                    "lane_id": document["lane_id"],
                    "algorithm": document["algorithm"],
                    "seed": document["seed"],
                    "policy_sha256": document["solver_result"]["policy_sha256"],
                }
                for document in lane_documents
            ],
            deals_per_assignment=int(task["crossplay_deals_per_assignment"]),
            familywise_alpha=float(task["crossplay_familywise_alpha"]),
            effective_stack_bb=effective_stack_bb,
            limit_effective_stack_fraction_per_hand=float(
                task[
                    "crossplay_max_absolute_advantage_effective_stack_fraction_per_hand"
                ]
            ),
            maximum_missing_policy_reach_fraction=float(
                EVALUATION_CONTRACT["maximum_missing_policy_reach_fraction"]
            ),
            pairing_unit=str(EVALUATION_CONTRACT["crossplay_pairing_unit"]),
        )
    elif kind == "restricted_exact":
        oracle_rows = [document["exact_oracle"] for document in lane_documents]
        profile_spread = max(
            max(float(row["profile_values"][player]) for row in oracle_rows)
            - min(float(row["profile_values"][player]) for row in oracle_rows)
            for player in (0, 1)
        )
        per_algorithm_order_stability = {}
        for algorithm in ("cfr_plus", "dcfr"):
            algorithm_documents = [
                document
                for document in lane_documents
                if document["algorithm"] == algorithm
            ]
            algorithm_oracles = [
                document["exact_oracle"] for document in algorithm_documents
            ]
            per_algorithm_order_stability[algorithm] = {
                "numerical_order_ids": sorted(
                    int(document["seed"]) for document in algorithm_documents
                ),
                "maximum_profile_value_spread_chips": max(
                    max(
                        float(row["profile_values"][player])
                        for row in algorithm_oracles
                    )
                    - min(
                        float(row["profile_values"][player])
                        for row in algorithm_oracles
                    )
                    for player in (0, 1)
                ),
            }
        evidence_body["numerical_order_stability"] = {
            "schema_version": "stage21_restricted_order_stability_v1",
            "order_ids": sorted(
                {int(document["seed"]) for document in lane_documents}
            ),
            "member_order_ids": sorted(
                int(document["seed"]) for document in lane_documents
            ),
            "per_algorithm_order_stability": per_algorithm_order_stability,
            "independent_random_seed_claim": False,
            "maximum_profile_value_spread_chips": profile_spread,
            "maximum_allowed_profile_value_spread_chips": 5.0,
            "reference_profile_value_chips": [
                math.fsum(
                    float(row["profile_values"][player]) for row in oracle_rows
                )
                / len(oracle_rows)
                for player in (0, 1)
            ],
            "reference_profile_source": (
                "mean_of_six_full_chance_projected_profile_oracles"
            ),
            "all_projected_nashconv_gates_pass": all(
                float(row["nashconv"]) <= 5.0 for row in oracle_rows
            ),
            "gate_pass": profile_spread <= 5.0
            and all(float(row["nashconv"]) <= 5.0 for row in oracle_rows),
        }
    else:
        evidence_body["forced_member_validation"] = {
            "all_six_zero_sum_runout_evaluations_qualified": all(
                document.get("qualified") is True for document in lane_documents
            ),
            "gate_pass": all(
                document.get("qualified") is True for document in lane_documents
            ),
        }
    member_count = len(lane_documents)
    return {
        "producer_token": str(binding["token"]),
        "staged_member_reads": staged_member_reads - starting_staged_member_reads,
        "root_evidence": with_self_hash(evidence_body),
        "validated_result": {
            "token": binding["token"],
            "sha256": binding["sha256"],
            "lane_count": staging_receipt["staged_lane_count"],
            "root_id": binding["root_id"],
            "validated_member_count": member_count,
        },
    }


def _process_aggregate_root_binding(
    *,
    task: Mapping[str, Any],
    binding: Mapping[str, Any],
    legacy_result_contract: Mapping[str, Any] | None,
    continuation: Any,
    worker_materialization_root: Path | None,
    starting_staged_member_reads: int,
    emit_aggregate_progress: Callable[..., None],
) -> dict[str, Any]:
    """Materialize, evaluate, and fully release one root before returning."""

    source_handles: list[Mapping[str, Any]] | None = list(
        binding.get("staged_lane_handles", ())
    )
    root_materialization_context: tempfile.TemporaryDirectory[str] | None = None
    root_materialization_path: Path | None = None
    root_continuation = None
    outcome: dict[str, Any] | None = None
    continuation_cache_release = None
    try:
        if any(
            handle.get("policy_materialization_deferred") is True
            for handle in source_handles
        ):
            if worker_materialization_root is None:
                raise ValueError("deferred aggregate root lacks owned scratch")
            root_materialization_context = tempfile.TemporaryDirectory(
                prefix="root_", dir=worker_materialization_root
            )
            root_materialization_path = Path(root_materialization_context.name)
            source_handles = materialize_local_archive_lane_group(
                source_handles,
                staging_root=root_materialization_path / "materialized",
            )
        if continuation is not None:
            if not hasattr(continuation, "spawn_runtime_view"):
                raise TypeError("aggregate continuation lacks a root-scoped runtime view")
            root_continuation = continuation.spawn_runtime_view()
        outcome = _evaluate_aggregate_root_binding(
            task=task,
            binding=binding,
            source_handles=source_handles,
            legacy_result_contract=legacy_result_contract,
            continuation=root_continuation,
            starting_staged_member_reads=starting_staged_member_reads,
            emit_aggregate_progress=emit_aggregate_progress,
        )
    finally:
        # The evaluator's frame has already closed all policy-store contexts.
        # Drop its remaining path/metadata handles before deleting this root's
        # materialization and clearing mutable continuation/runtime caches.
        source_handles = None
        if root_materialization_context is not None:
            root_materialization_context.cleanup()
        materialization_removed = (
            root_materialization_path is None
            or not root_materialization_path.exists()
        )
        if not materialization_removed:
            raise RuntimeError("aggregate root materialization survived cleanup")
        if root_continuation is not None:
            continuation_cache_release = root_continuation.release_runtime_state()
        root_continuation = None
        runtime_cache_release = release_runtime_caches()
        process_memory_release = _release_process_memory()
        lifecycle_release = with_self_hash(
            {
                "schema_version": "stage21_aggregate_root_lifecycle_release_v1",
                "root_id": binding["root_id"],
                "policy_store_handles_closed_before_cleanup": True,
                "root_materialization_removed_before_next_root": (
                    materialization_removed
                ),
                "root_local_source_handle_references_released": True,
                "root_local_evaluation_frame_released": True,
                "continuation_immutable_bank_reused_without_drive_reload": (
                    continuation is not None
                ),
                "continuation_root_cache_release": continuation_cache_release,
                "runtime_cache_count": int(runtime_cache_release["cache_count"]),
                "runtime_cache_entries_released": int(
                    runtime_cache_release["entries_released"]
                ),
                "process_memory_release": process_memory_release,
            }
        )
    if outcome is None:
        raise AssertionError("aggregate root evaluation returned no outcome")
    outcome["root_lifecycle_release"] = lifecycle_release
    return outcome


def _aggregate_shard(
    task: Mapping[str, Any], progress_path: Path | None = None
) -> dict[str, Any]:
    root_bindings = list(task.get("root_bindings", ()))
    if not root_bindings:
        raise ValueError("strict aggregate shard cannot be an empty no-op")
    legacy_result_contract = None
    legacy_result_contract_path = task.get(
        "legacy_completed_result_contract_path"
    )
    if legacy_result_contract_path is not None:
        legacy_result_contract = load_legacy_completed_result_contract(
            Path(str(legacy_result_contract_path))
        )
        if legacy_result_contract.get("self_sha256") != task.get(
            "legacy_completed_result_contract_self_sha256"
        ):
            raise ValueError("aggregate legacy completed-result contract mismatch")
    validated = []
    root_evidence = []
    root_lifecycle_releases = []
    last_progress = 0.0
    completed_roots = 0

    def emit_aggregate_progress(
        event: str, *, force: bool = False, **extra: Any
    ) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 60.0:
            return
        _write_lane_progress(
            progress_path,
            task,
            {
                "event": event,
                "status": "VALIDATING_STRICT_AGGREGATE",
                "lane_phase": "aggregate_member_and_crossplay_validation",
                "completed_aggregate_roots": completed_roots,
                "total_aggregate_roots": len(root_bindings),
                "percent_complete": 95.0
                * completed_roots
                / max(1, len(root_bindings)),
                "current_iteration": 0,
                "target_iterations": 0,
                "last_checkpoint_iteration": 0,
                **extra,
            },
        )
        last_progress = now

    emit_aggregate_progress("aggregate_shard_started", force=True)
    continuation = _load_continuation(task)
    producer_tokens: set[str] = set()
    staged_member_reads = 0
    deferred_materialization_required = any(
        handle.get("policy_materialization_deferred") is True
        for binding in root_bindings
        for handle in binding.get("staged_lane_handles", ())
    )
    worker_materialization_root: Path | None = None
    if deferred_materialization_required:
        worker_materialization_root = Path(
            str(task.get("local_aggregate_policy_materialization_root", ""))
        ).resolve()
        if (
            worker_materialization_root.name != str(task["lane_id"])
            or worker_materialization_root.parent.name
            != "worker_policy_materialization"
        ):
            raise ValueError("aggregate worker materialization root is not owned")
        worker_materialization_root.mkdir(parents=True, exist_ok=True)
    for binding in root_bindings:
        outcome = _process_aggregate_root_binding(
            task=task,
            binding=binding,
            legacy_result_contract=legacy_result_contract,
            continuation=continuation,
            worker_materialization_root=worker_materialization_root,
            starting_staged_member_reads=staged_member_reads,
            emit_aggregate_progress=emit_aggregate_progress,
        )
        producer_tokens.add(str(outcome["producer_token"]))
        staged_member_reads += int(outcome["staged_member_reads"])
        root_evidence.append(outcome["root_evidence"])
        validated.append(outcome["validated_result"])
        root_lifecycle_releases.append(outcome["root_lifecycle_release"])
        completed_roots += 1
        emit_aggregate_progress(
            "aggregate_root_complete",
            root_lifecycle_release_self_sha256=outcome[
                "root_lifecycle_release"
            ]["self_sha256"],
        )
        outcome = None
    unique_archive_count = len(producer_tokens)
    binding = None
    continuation_cache_release = (
        continuation.release_runtime_state()
        if continuation is not None
        and hasattr(continuation, "release_runtime_state")
        else None
    )
    continuation = None
    if worker_materialization_root is not None:
        try:
            worker_materialization_root.rmdir()
        except OSError:
            # Exceptional worker exits are cleaned by the owned token scratch
            # lifecycle; a successful shard must not retain policy stores.
            pass
    cache_release = release_runtime_caches()
    emit_aggregate_progress("aggregate_shard_complete", force=True)
    return {
        "lane_id": task["lane_id"],
        "root_id": task["root_id"],
        "algorithm": None,
        "seed": task["seed"],
        "solver_continuation_semantic_sha256": task[
            "solver_continuation_semantic_sha256"
        ],
        "task_semantic_sha256": task["task_semantic_sha256"],
        "execution_identity": task["execution_identity"],
        "role": "strict_result_member_and_crossplay_validation_shard",
        "validated_results": validated,
        "root_evidence": root_evidence,
        "root_lifecycle_releases": root_lifecycle_releases,
        "transport_valid": True,
        "artifact_hash_valid": True,
        "independent_confirmation_panel": False,
        "root_binding_count": len(root_bindings),
        "unique_producer_archive_validation_count": 0,
        "producer_archives_prevalidated_once_by_coordinator": unique_archive_count,
        "staged_lane_member_read_count": staged_member_reads,
        "producer_archive_validation_reuse_count": len(root_bindings),
        "continuation_cache_release": continuation_cache_release,
        "runtime_cache_release": cache_release,
    }


def run_lane_task(
    task: Mapping[str, Any],
    lane_root: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    role = task.get("role")
    if role == "strict_result_member_and_crossplay_validation_shard":
        return _aggregate_shard(task, progress_path)
    if role == "forced_runout_evaluation":
        return _generic_forced_lane(task, progress_path)
    return _generic_solver_lane(task, lane_root, progress_path)


def _read_context(
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = _load(source_root / "contracts" / "execution_manifest.json")
    registry = _load(source_root / "contracts" / "root_registry.json")
    dependency = _load(source_root / "contracts" / "dependency_contract.json")
    coverage = _load(source_root / "contracts" / "coverage_contract.json")
    for document in (execution, dependency, coverage):
        validate_self_hash(document)
    return execution, registry, dependency, coverage


def _partition_aggregate_root_bindings(
    root_bindings: Sequence[Mapping[str, Any]], *, shard_count: int = 48
) -> list[list[Mapping[str, Any]]]:
    """Assign substantive shards while minimizing producer-archive rereads."""

    if shard_count <= 0 or len(root_bindings) < shard_count:
        raise ValueError("strict aggregate requires at least one root per shard")
    can_group = all(
        binding.get("token") is not None
        and binding.get("sha256") is not None
        for binding in root_bindings
    )
    if can_group:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for binding in root_bindings:
            key = (
                str(binding["token"]),
                str(binding["sha256"]),
            )
            grouped.setdefault(key, []).append(binding)
        group_rows = list(grouped.values())
        base, remainder = divmod(shard_count, len(group_rows))
        quotas = [base + int(index < remainder) for index in range(len(group_rows))]
        if base <= 0 or any(
            quota > len(rows) for quota, rows in zip(quotas, group_rows)
        ):
            raise ValueError("aggregate producer groups cannot populate every shard")
        assignments = [
            list(rows[index::quota])
            for rows, quota in zip(group_rows, quotas)
            for index in range(quota)
        ]
    else:
        assignments = [
            list(root_bindings[index::shard_count]) for index in range(shard_count)
        ]
    if any(not rows for rows in assignments):
        raise ValueError("strict aggregate shard cannot be an empty no-op")
    flattened = [id(row) for rows in assignments for row in rows]
    if len(flattened) != len(root_bindings) or len(set(flattened)) != len(root_bindings):
        raise AssertionError("aggregate partition lost or duplicated a root binding")
    return assignments


def _build_aggregate_staging_summary(
    *,
    aggregate_staging_receipts: Sequence[Mapping[str, Any]],
    maximum_root_materialization_bytes: int,
    aggregate_phase_disk_preflight: Mapping[str, Any],
    aggregate_local_archive_post_worker_readback: Mapping[str, Any],
    scratch_deleted: bool,
) -> dict[str, Any]:
    """Build the compact post-authority aggregate staging receipt."""

    if maximum_root_materialization_bytes <= 0:
        raise ValueError("maximum root materialization bytes must be positive")
    if scratch_deleted is not True:
        raise RuntimeError("aggregate staging summary requires confirmed cleanup")
    return with_self_hash(
        {
            "schema_version": "stage21_aggregate_coordinator_staging_summary_v2",
            "producer_archive_count": len(aggregate_staging_receipts),
            "staged_lane_member_count": sum(
                int(receipt["staged_lane_count"])
                for receipt in aggregate_staging_receipts
            ),
            "producer_receipt_self_sha256s": sorted(
                str(receipt["self_sha256"])
                for receipt in aggregate_staging_receipts
            ),
            "producer_archives_validated_once_by_coordinator": True,
            "workers_materialize_only_six_local_disk_backed_policies_per_root": True,
            "compressed_local_archives_retained_until_authority_consumption": True,
            "aggregate_worker_count_is_not_disk_or_ram_capped": True,
            "maximum_root_worker_disk_budget_bytes": (
                int(maximum_root_materialization_bytes)
            ),
            "aggregate_phase_disk_preflight": dict(
                aggregate_phase_disk_preflight
            ),
            "local_archive_post_worker_readback": dict(
                aggregate_local_archive_post_worker_readback
            ),
            "scratch_rebuilt_and_revalidated_for_this_run": True,
            "scratch_deleted_after_finalization": True,
        }
    )


def _build_continuation_staging_summary(
    *,
    local_continuation_staging: Mapping[str, Any],
    scratch_deleted: bool,
) -> dict[str, Any]:
    if scratch_deleted is not True:
        raise RuntimeError("continuation staging summary requires confirmed cleanup")
    return with_self_hash(
        {
            "schema_version": "stage21_continuation_staging_cleanup_v2",
            "staging_receipt_self_sha256": local_continuation_staging[
                "self_sha256"
            ],
            "file_count": local_continuation_staging["file_count"],
            "workers_used_only_hash_bound_local_staging": True,
            "scratch_deleted_only_after_result_and_adjudication_finalization": True,
            "scratch_deleted_after_finalization": True,
        }
    )


def _token_lease_path(runtime_root: Path, token: str) -> Path:
    if not token or token.replace("_", "").replace("-", "").isalnum() is False:
        raise ValueError("unsafe token lease identity")
    return runtime_root / token / ".stage21_token_writer_lease.json"


@contextmanager
def _durable_token_writer_lease(
    *,
    runtime_root: Path,
    binding: Mapping[str, Any],
) -> Any:
    lease_path = _token_lease_path(runtime_root, str(binding["token"]))
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    owner_uuid = str(uuid.uuid4())
    document = with_self_hash(
        {
            "schema_version": "stage21_whole_token_writer_lease_v1",
            **dict(binding),
            "owner_uuid": owner_uuid,
            "created_unix_seconds": time.time(),
            "automatic_pid_reclaim_allowed": False,
            "high_frequency_heartbeat_present": False,
        }
    )
    raw = canonical_json_bytes(document)
    try:
        descriptor = os.open(
            str(lease_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError as error:
        try:
            existing_raw = lease_path.read_bytes()
            existing = json.loads(existing_raw)
            validate_self_hash(existing)
            evidence = (
                f"owner={existing.get('owner_uuid')} "
                f"sha256={sha256_bytes(existing_raw)}"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            evidence = "malformed_existing_lease"
        raise RuntimeError(
            f"whole-token writer lease already exists: {lease_path} ({evidence})"
        ) from error
    try:
        try:
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if lease_path.read_bytes() != raw:
            raise IOError("whole-token writer lease immediate raw readback mismatch")
    except BaseException:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass
        raise
    lifecycle = {
        "schema_version": "stage21_whole_token_writer_lease_lifecycle_v1",
        "lease_path": str(lease_path),
        "lease_document_self_sha256": document["self_sha256"],
        "lease_file_sha256": sha256_bytes(raw),
        "owner_uuid": owner_uuid,
        "exclusive_creation_and_raw_readback_verified": True,
        "released_by_matching_owner": False,
    }
    try:
        yield lifecycle
    finally:
        stored_raw = lease_path.read_bytes()
        stored = json.loads(stored_raw)
        validate_self_hash(stored)
        if (
            stored.get("owner_uuid") != owner_uuid
            or sha256_bytes(stored_raw) != lifecycle["lease_file_sha256"]
        ):
            raise RuntimeError("whole-token lease owner changed before release")
        lease_path.unlink()
        lifecycle["released_by_matching_owner"] = True


def recover_stale_token_writer_lease(
    lease_path: Path,
    *,
    expected_owner_uuid: str,
    expected_lease_file_sha256: str,
    confirmation: str,
) -> dict[str, Any]:
    """Explicit operator-only stale recovery; never infers remote host liveness."""

    if confirmation != "REMOVE_EXACT_STALE_STAGE21_TOKEN_LEASE":
        raise ValueError("explicit stale-token lease recovery confirmation is required")
    lease_path = lease_path.resolve()
    if lease_path.name != ".stage21_token_writer_lease.json":
        raise ValueError("stale-token recovery target is not a token lease")
    raw = lease_path.read_bytes()
    document = json.loads(raw)
    validate_self_hash(document)
    if (
        document.get("schema_version") != "stage21_whole_token_writer_lease_v1"
        or document.get("token") != lease_path.parent.name
        or document.get("owner_uuid") != expected_owner_uuid
        or sha256_bytes(raw) != expected_lease_file_sha256
    ):
        raise RuntimeError("stale-token lease recovery binding mismatch")
    lease_path.unlink()
    return with_self_hash(
        {
            "schema_version": "stage21_explicit_stale_lease_recovery_v1",
            "lease_path": str(lease_path),
            "removed_owner_uuid": expected_owner_uuid,
            "removed_lease_file_sha256": expected_lease_file_sha256,
            "remote_pid_liveness_inference_used": False,
            "status": "EXACT_STALE_LEASE_REMOVED",
        }
    )


def _whole_token_owned(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = str(kwargs["token"])
        source_root = Path(kwargs["source_root"])
        stage_drive_root = Path(kwargs["stage_drive_root"])
        runtime_root = Path(kwargs["runtime_root"])
        execution, registry, _, _ = _read_context(source_root)
        stage = str(execution["stage"])
        requested_local_root = kwargs.get("local_scratch_root")
        local_parent = require_ephemeral_local_path(
            Path(requested_local_root)
            if requested_local_root is not None
            else Path(tempfile.gettempdir()) / "stage21_runtime_scratch",
            forbidden_roots=(stage_drive_root, runtime_root),
        )
        local_parent.mkdir(parents=True, exist_ok=True)
        binding = {
            "revision": execution["revision"],
            "stage": stage,
            "token": token,
            "execution_manifest_self_sha256": execution["self_sha256"],
            "package_source_semantic_sha256": execution["source_semantic_hashes"][
                "package_source_semantic_sha256"
            ],
            "root_registry_semantic_sha256": registry["semantic_sha256"],
        }
        with _durable_token_writer_lease(
            runtime_root=runtime_root,
            binding=binding,
        ) as lease_lifecycle:
            with tempfile.TemporaryDirectory(
                prefix=f"{token}_owned_", dir=local_parent
            ) as owned_text:
                owned_root = require_ephemeral_local_path(
                    Path(owned_text),
                    forbidden_roots=(stage_drive_root, runtime_root),
                )
                kwargs["local_scratch_root"] = owned_root
                result = function(*args, **kwargs)
        body = {key: value for key, value in result.items() if key != "self_sha256"}
        body["whole_token_writer_lease"] = with_self_hash(lease_lifecycle)
        body["owned_local_scratch_cleanup"] = {
            "base": str(local_parent),
            "owned_path_removed_in_finally": not Path(owned_text).exists(),
            "mounted_drive_overlap": False,
        }
        return with_self_hash(body)

    return wrapped


def _legacy_result_contract_for_row(
    *,
    source_root: Path,
    row: Mapping[str, Any],
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load the completed-R9 bridge only for River strict aggregation.

    Fresh producer notebooks neither read nor validate legacy completed results.
    Their sole durable state is the current package's own runtime tree.
    """

    if row.get("kind") != "strict_aggregate":
        return None
    legacy_result_contract_path = (
        source_root / "contracts" / "legacy_completed_result_contract.json"
    )
    if not legacy_result_contract_path.is_file():
        return None
    legacy_result_contract = load_legacy_completed_result_contract(
        legacy_result_contract_path
    )
    qualification = _load(source_root / "contracts" / "qualification_contract.json")
    validate_self_hash(qualification)
    validate_legacy_completed_result_contract_target(
        legacy_result_contract,
        execution=execution,
        registry=registry,
        qualification=qualification,
    )
    return legacy_result_contract


@_whole_token_owned
def run_token(
    *,
    token: str,
    source_root: Path,
    stage_drive_root: Path,
    runtime_root: Path,
    local_scratch_root: Path | None = None,
    requested_workers: Any = 0,
    target_iterations: int | None = None,
    target_tier: int | None = None,
    minimum_qualification_tier: int | None = None,
    evaluation_deals_per_family_role: int | None = None,
) -> dict[str, Any]:
    execution, registry, dependency, coverage = _read_context(source_root)
    stage = str(execution["stage"])
    row = next(
        (item for item in execution["notebooks"] if item["token"] == token),
        None,
    )
    if row is None:
        raise ValueError(f"unknown notebook token: {token}")
    root_kind = row.get("root_kind")
    allowed_evaluation_deals = tuple(
        int(value)
        for value in execution.get(
            "allowed_evaluation_deals_per_family_role",
            EVALUATION_ALLOWED_DEALS_PER_FAMILY_ROLE,
        )
    )
    selected_evaluation_deals = int(
        execution["evaluation_deals_per_family_role"]
        if evaluation_deals_per_family_role is None
        else evaluation_deals_per_family_role
    )
    if selected_evaluation_deals not in allowed_evaluation_deals:
        raise ValueError("evaluation deal count is not preregistered by this package")
    if root_kind == "approximate":
        allowed_tiers = tuple(
            int(value)
            for value in execution.get(
                "allowed_approximate_tiers", APPROXIMATE_ALLOWED_TIERS
            )
        )
        selected_tier = int(
            execution["default_approximate_tier"]
            if target_tier is None
            else target_tier
        )
        if selected_tier not in allowed_tiers:
            raise ValueError("target tier is not preregistered by this package")
        expected_target = (
            int(execution["approximate_base_iterations_per_tier"])
            * selected_tier
        )
        selected_target = int(
            expected_target if target_iterations is None else target_iterations
        )
        if selected_target != expected_target:
            raise ValueError("target iterations do not match the selected tier")
        selected_minimum_qualification_tier = int(
            execution["default_minimum_approximate_qualification_tier"]
            if minimum_qualification_tier is None
            else minimum_qualification_tier
        )
        if selected_minimum_qualification_tier not in allowed_tiers:
            raise ValueError(
                "minimum qualification tier is not preregistered by this package"
            )
        if selected_minimum_qualification_tier > selected_tier:
            raise ValueError(
                "minimum qualification tier exceeds the selected target tier"
            )
    else:
        selected_tier = 0
        if target_tier not in (None, 0):
            raise ValueError("only approximate producers accept a target tier")
        selected_minimum_qualification_tier = 0
        if minimum_qualification_tier not in (None, 0):
            raise ValueError(
                "only approximate producers accept a minimum qualification tier"
            )
        selected_target = int(
            row["target_iterations"]
            if target_iterations is None
            else target_iterations
        )
        if selected_target != int(row["target_iterations"]):
            raise ValueError("non-approximate target iterations are immutable")
    legacy_result_contract = _legacy_result_contract_for_row(
        source_root=source_root,
        row=row,
        execution=execution,
        registry=registry,
    )
    if row["lane_count"] != 48:
        raise ValueError("every Stage 21 notebook must bind exactly 48 work lanes")
    if local_scratch_root is None:
        raise RuntimeError("owned local scratch was not installed by token lease wrapper")
    local_scratch_root = require_ephemeral_local_path(
        Path(local_scratch_root),
        forbidden_roots=(stage_drive_root, runtime_root),
    )
    if row["kind"] == "strict_aggregate":
        returned = stage_drive_root / "returned_results"
        for producer in execution["notebooks"]:
            if producer["kind"] == "strict_aggregate":
                continue
            producer_path = returned / producer["result_filename"]
            if not producer_path.is_file():
                raise RuntimeError(
                    f"aggregate is held: missing producer result {producer_path.name}"
                )
    continuation_payload_bytes = 0
    dependency_authority_raw: bytes | None = None
    continuation_preflight_inventory: dict[str, Any] | None = None
    if (
        dependency.get("required") is True
        and dependency.get("continuation_required") is True
    ):
        authority_path = (
            stage_drive_root
            / Path(str(dependency["authority_relative_to_current_stage_root"]))
        ).resolve()
        authority_buffer = bytearray()
        with authority_path.open("rb") as authority_handle:
            for block in iter(lambda: authority_handle.read(1024 * 1024), b""):
                authority_buffer.extend(block)
        dependency_authority_raw = bytes(authority_buffer)
        authority_preview = json.loads(dependency_authority_raw)
        continuation_preview = authority_preview.get("continuation")
        if not isinstance(continuation_preview, Mapping):
            raise RuntimeError("dependency continuation is absent during local preflight")
        preview_manifest_name = continuation_preview.get("manifest_filename")
        preview_payload_name = continuation_preview.get(
            "payload_filename"
        ) or continuation_preview.get("rows_filename")
        if not preview_manifest_name or not preview_payload_name:
            raise RuntimeError("dependency continuation preflight binding is incomplete")
        continuation_preflight_inventory = _continuation_tree_preflight_inventory(
            source_model_root=stage_drive_root.parent,
            primary_manifest_path=(
                authority_path.parent / str(preview_manifest_name)
            ),
            primary_payload_path=(
                authority_path.parent / str(preview_payload_name)
            ),
        )
        continuation_payload_bytes = int(
            continuation_preflight_inventory["total_bytes"]
        )
    local_scratch_preflight = local_disk_preflight(
        local_scratch_root,
        components={
            "continuation_tree_payloads": continuation_payload_bytes,
            "expected_checkpoint_spools": int(
                execution["estimated_required_disk_bytes_per_notebook"]
            ),
        },
    )
    dependency_receipt = verify_dependency_authority(
        stage,
        stage_drive_root,
        dependency,
        staging_root=local_scratch_root / "continuation_staging",
        dependency_authority_raw=dependency_authority_raw,
        continuation_preflight_inventory=continuation_preflight_inventory,
    )
    roots = _root_map(registry)
    range_identities: dict[str, dict[str, str]] = {}
    marginal_identities: dict[str, dict[str, str]] = {}
    for profile in coverage["range_feature_coverage"]["per_root_player_profiles"]:
        root_id = str(profile["root_id"])
        range_identities.setdefault(root_id, {})[
            str(profile["player"])
        ] = str(profile["range_identity_sha256"])
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
            raise RuntimeError("coverage contract has inconsistent joint marginals")
    if any(set(values) != {"0", "1"} for values in range_identities.values()):
        raise RuntimeError("coverage contract lacks both player range identities")
    execution_identity = {
        "execution_manifest_self_sha256": execution["self_sha256"],
        "package_source_semantic_sha256": execution["source_semantic_hashes"][
            "package_source_semantic_sha256"
        ],
        "root_registry_semantic_sha256": registry["semantic_sha256"],
        "coverage_contract_self_sha256": coverage["self_sha256"],
        "dependency_contract_self_sha256": dependency["self_sha256"],
        "dependency_authority_sha256": dependency_receipt.get("authority_sha256"),
    }
    continuation_identity = dependency_receipt.get("continuation")
    if continuation_identity:
        execution_identity["continuation_manifest_self_sha256"] = (
            continuation_identity["manifest_self_sha256"]
        )
        execution_identity["continuation_manifest_file_sha256"] = (
            continuation_identity["manifest_file_sha256"]
        )
        execution_identity["continuation_payload_file_sha256"] = (
            continuation_identity["payload_file_sha256"]
        )
        execution_identity["continuation_payload_self_sha256"] = (
            continuation_identity.get("payload_self_sha256")
        )
    lane_root = runtime_root / token / "lanes"
    tasks = []
    aggregate_scratch: Path | None = None
    aggregate_staging_receipts: list[dict[str, Any]] = []
    aggregate_staged_archives: list[dict[str, Any]] = []
    validated_producer_set: ValidatedProducerArchiveSet | None = None
    aggregate_phase_disk_preflight: dict[str, Any] | None = None
    maximum_root_materialization_bytes: int | None = None
    if row["kind"] == "strict_aggregate":
        returned = stage_drive_root / "returned_results"
        aggregate_scratch = (
            local_scratch_root / "validated_producer_scratch"
        ).resolve()
        if aggregate_scratch.name != "validated_producer_scratch":
            raise RuntimeError("unsafe aggregate staging root")
        if aggregate_scratch.exists():
            shutil.rmtree(aggregate_scratch)
        aggregate_scratch.mkdir(parents=True, exist_ok=True)
        producer_rows = [item for item in execution["notebooks"] if item["kind"] != "strict_aggregate"]
        root_bindings = []
        for producer in producer_rows:
            result_path = returned / producer["result_filename"]
            if not result_path.is_file():
                raise RuntimeError(f"aggregate is held: missing producer result {result_path.name}")
            contracted_archive = (
                legacy_result_contract.get(
                    "known_complete_returned_archives", {}
                ).get(
                    result_path.name
                )
                if legacy_result_contract is not None
                else None
            )
            staged_archive = stage_validated_result_archive(
                result_path,
                expected_stage=stage,
                expected_token=str(producer["token"]),
                expected_lane_ids=[
                    str(lane["lane_id"]) for lane in producer["lanes"]
                ],
                staging_root=aggregate_scratch / str(producer["token"]),
                capture_members=(
                    "receipts/execution_identity_receipt.json",
                ),
                expected_source_size=(
                    int(contracted_archive["size"])
                    if isinstance(contracted_archive, Mapping)
                    else None
                ),
                expected_source_sha256=(
                    str(contracted_archive["sha256"])
                    if isinstance(contracted_archive, Mapping)
                    else None
                ),
                materialize_policy_stores=False,
            )
            archive_sha = staged_archive["source_identity"]["sha256"]
            identity_raw = staged_archive["captured_members"][
                "receipts/execution_identity_receipt.json"
            ]
            producer_identity = json.loads(identity_raw)
            validate_self_hash(producer_identity)
            current_producer_identity = (
                producer_identity.get("execution_manifest_self_sha256")
                == execution["self_sha256"]
                and producer_identity.get("root_registry_semantic_sha256")
                == execution["root_registry_semantic_sha256"]
                and producer_identity.get("package_source_semantic_sha256")
                == execution["source_semantic_hashes"][
                    "package_source_semantic_sha256"
                ]
            )
            if not current_producer_identity:
                if legacy_result_contract is None:
                    raise RuntimeError(
                        f"producer {producer['token']} has uncontracted legacy identity"
                    )
                staged_archive["legacy_archive_validation_receipt"] = (
                    validate_legacy_result_execution_identity_receipt(
                        producer_identity,
                        token=str(producer["token"]),
                        contract=legacy_result_contract,
                        source_archive_identity=staged_archive[
                            "source_identity"
                        ],
                        result_filename=result_path.name,
                    )
                )
            aggregate_staging_receipts.append(staged_archive["receipt"])
            aggregate_staged_archives.append(staged_archive)
            producer_root_ids = sorted(
                {str(lane["root_id"]) for lane in producer["lanes"]}
            )
            for root_id in producer_root_ids:
                root = roots[root_id]
                lane_ids = sorted(
                    str(lane["lane_id"])
                    for lane in producer["lanes"]
                    if str(lane["root_id"]) == root_id
                )
                if len(lane_ids) != 6:
                    raise RuntimeError("producer root does not bind six members")
                lane_handles = [
                    staged_archive["lane_handles"][lane_id]
                    for lane_id in lane_ids
                ]
                root_bindings.append(
                    {
                        "token": producer["token"],
                        "sha256": archive_sha,
                        "producer_staging_receipt": staged_archive["receipt"],
                        "root_id": root_id,
                        "root": root,
                        "lane_ids": lane_ids,
                        "staged_lane_handles": lane_handles,
                        "execution_identity": {
                            **execution_identity,
                            "range_identity_sha256_by_player": range_identities[
                                root_id
                            ],
                            "range_marginal_identity_sha256": marginal_identities[
                                root_id
                            ],
                        },
                    }
                )
        if len(root_bindings) != registry["root_count"]:
            raise RuntimeError("aggregate root binding does not cover the registry")
        validated_producer_set = ValidatedProducerArchiveSet.create(
            stage=stage,
            aggregate_token=token,
            staged_archives=aggregate_staged_archives,
        )
        maximum_root_serialized_bytes = max(
            sum(int(handle["size"]) for handle in binding["staged_lane_handles"])
            for binding in root_bindings
        )
        # One worker retains at most six SQLite policies for its current root.
        # Twice the exact six-lane serialized size covers the sequential lane
        # extraction plus the bounded SQLite stores without counting any other
        # root or a mutually exclusive authority phase.
        maximum_root_materialization_bytes = max(
            1, maximum_root_serialized_bytes * 2
        )
        aggregate_phase_disk_preflight = local_disk_preflight(
            local_scratch_root,
            components={
                "maximum_single_root_policy_materialization": (
                    maximum_root_materialization_bytes
                ),
                "remaining_aggregate_result_and_checkpoint_spools": int(
                    execution["estimated_required_disk_bytes_per_notebook"]
                ),
            },
        )
        aggregate_assignments = _partition_aggregate_root_bindings(root_bindings)
        continuation_binding = dependency_receipt.get("continuation")
        for lane in row["lanes"]:
            shard = int(lane["shard_index"])
            assigned = aggregate_assignments[shard]
            task = {
                **lane,
                "stage": stage,
                "root_id": f"{stage.upper()}-AGGREGATE-{shard:02d}",
                "root_bindings": assigned,
                "execution_identity": execution_identity,
                "crossplay_deals_per_assignment": int(
                    execution.get(
                        "crossplay_deals_per_assignment",
                        CROSSPLAY_DEALS_PER_ASSIGNMENT,
                    )
                ),
                "crossplay_familywise_alpha": EVALUATION_CONTRACT[
                    "crossplay_familywise_alpha"
                ],
                "crossplay_max_absolute_advantage_effective_stack_fraction_per_hand": (
                    EVALUATION_CONTRACT[
                        "crossplay_max_absolute_symmetric_advantage_effective_stack_fraction_per_hand"
                    ]
                ),
                "role": "strict_result_member_and_crossplay_validation_shard",
                "local_aggregate_policy_materialization_root": str(
                    aggregate_scratch
                    / "worker_policy_materialization"
                    / str(lane["lane_id"])
                ),
            }
            if continuation_binding:
                task["continuation_manifest_path"] = continuation_binding[
                    "manifest_path"
                ]
                task["continuation_payload_path"] = continuation_binding[
                    "payload_path"
                ]
            if legacy_result_contract is not None:
                task["legacy_completed_result_contract_path"] = str(
                    legacy_result_contract_path
                )
                task["legacy_completed_result_contract_self_sha256"] = (
                    legacy_result_contract["self_sha256"]
                )
            tasks.append(task)
    else:
        continuation_binding = dependency_receipt.get("continuation")
        for lane in row["lanes"]:
            root = roots[lane["root_id"]]
            task = {
                **lane,
                "stage": stage,
                "root": root,
                "execution_identity": {
                    **execution_identity,
                    "range_identity_sha256_by_player": range_identities[
                        str(root["root_id"])
                    ],
                    "range_marginal_identity_sha256": marginal_identities[
                        str(root["root_id"])
                    ],
                },
                "target_tier": selected_tier,
                "minimum_qualification_tier": (
                    selected_minimum_qualification_tier
                ),
                "target_iterations": selected_target,
                "qualification_terminal_iterations": selected_target,
                "approximate_base_iterations_per_tier": int(
                    execution["approximate_base_iterations_per_tier"]
                ),
                "first_approximate_qualification_tier": int(
                    execution["first_approximate_qualification_tier"]
                ),
                "approximate_qualification_tier_stride": int(
                    execution["approximate_qualification_tier_stride"]
                ),
                "maximum_approximate_qualification_looks": int(
                    execution["maximum_approximate_qualification_looks"]
                ),
                "checkpoint_interval_iterations": int(
                    execution["restricted_exact_checkpoint_interval_iterations"]
                    if root["root_kind"] == "restricted_exact"
                    else execution["checkpoint_interval_iterations"]
                ),
                "evaluation_deals_per_family_role": selected_evaluation_deals,
                "evaluation_panel_count": execution["evaluation_panel_count"],
                "final_holdout_deals_per_family_role": selected_evaluation_deals,
                "exact_combos_per_player": execution["restricted_exact_combos_per_player"],
            }
            if continuation_binding:
                task["continuation_manifest_path"] = continuation_binding["manifest_path"]
                task["continuation_payload_path"] = continuation_binding["payload_path"]
            tasks.append(task)
    for task in tasks:
        task["token"] = token
        task["local_checkpoint_io_lock_path"] = str(
            local_scratch_root / "checkpoint_spools" / "admission.lock"
        )
        task["local_checkpoint_spool_root"] = str(
            local_scratch_root / "checkpoint_spools"
        )
        task["local_completion_spool_root"] = str(
            local_scratch_root / "lane_completion_spools"
        )
    queue_receipt = run_lane_queue(
        worker_path="stage21_baseline.runner:run_lane_task",
        tasks=tasks,
        lane_root=lane_root,
        requested_workers=requested_workers,
        estimated_required_disk_bytes=int(execution["estimated_required_disk_bytes_per_notebook"]),
        local_scratch_root=local_scratch_root / "telemetry",
    )
    aggregate_local_archive_post_worker_readback = None
    if validated_producer_set is not None:
        verified_local_archives = []
        for producer in validated_producer_set.producers:
            local_archive_path = Path(str(producer["local_archive_path"]))
            identity = file_identity(local_archive_path)
            if (
                int(identity["size"])
                != int(producer["source_identity"]["size"])
                or identity["sha256"]
                != producer["source_identity"]["sha256"]
            ):
                raise IOError(
                    "validated local producer archive changed during aggregate workers"
                )
            verified_local_archives.append(
                {
                    "token": producer["receipt"]["token"],
                    "size": identity["size"],
                    "sha256": identity["sha256"],
                }
            )
        aggregate_local_archive_post_worker_readback = with_self_hash(
            {
                "schema_version": (
                    "stage21_local_producer_archive_post_worker_readback_v1"
                ),
                "archive_count": len(verified_local_archives),
                "archives": verified_local_archives,
                "all_bytes_read_from_owned_local_scratch": True,
                "mounted_drive_reopened": False,
                "status": "PASS",
            }
        )
    identity_receipt = with_self_hash(
        {
            "schema_version": "stage21_execution_identity_receipt_v3",
            "stage": stage,
            "token": token,
            "execution_manifest_self_sha256": execution["self_sha256"],
            "root_registry_semantic_sha256": registry["semantic_sha256"],
            "package_source_semantic_sha256": execution["source_semantic_hashes"][
                "package_source_semantic_sha256"
            ],
            "coverage_contract_self_sha256": coverage["self_sha256"],
            "dependency_contract_self_sha256": dependency["self_sha256"],
            "continuation_content_identity": continuation_identity,
        }
    )
    result_path = stage_drive_root / "returned_results" / row["result_filename"]
    last_token_progress = 0.0

    def emit_token_progress(progress: Mapping[str, Any]) -> None:
        nonlocal last_token_progress
        now = time.monotonic()
        event = str(progress.get("event", "token_progress"))
        force = event in {
            "result_archive_publication_started",
            "result_archive_installed_full_readback_complete",
            "authority_finalization_started",
            "authority_aggregate_evidence_recomputed",
            "authority_policy_bank_reduced",
            "authority_policy_bank_spooled",
            "authority_continuation_calibration_complete",
            "authority_policy_bank_published",
            "authority_published_last",
            "token_finalization_failed",
        }
        if not force and now - last_token_progress < 60.0:
            return
        print(
            json.dumps(
                {
                    "stage21_token_progress": True,
                    "stage": stage,
                    "token": token,
                    **dict(progress),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        last_token_progress = now

    local_continuation_staging = (
        dependency_receipt.get("continuation", {}).get("local_staging")
        if isinstance(dependency_receipt.get("continuation"), Mapping)
        else None
    )
    continuation_scratch = (
        (local_scratch_root / "continuation_staging").resolve()
        if isinstance(local_continuation_staging, Mapping)
        else None
    )
    stage_adjudication = None
    authority_memory_observation = None
    finalization_error: Exception | None = None
    finalization_traceback = None
    cleanup_failures: list[dict[str, str]] = []
    aggregate_scratch_deleted = aggregate_scratch is None
    continuation_scratch_deleted = continuation_scratch is None
    try:
        publication = build_and_publish_result_archive(
            result_path,
            token=token,
            stage=stage,
            lane_handles=queue_receipt["handles"],
            receipt_documents={
                "dependency_read_receipt": dependency_receipt,
                "lane_queue_receipt": queue_receipt,
                "execution_identity_receipt": identity_receipt,
            },
            progress_callback=emit_token_progress,
        )
        installed_result = InstalledResultValidationReceipt.bind(
            result_path,
            publication,
            expected_stage=stage,
            expected_token=token,
        )
        if row["kind"] == "strict_aggregate":
            from .authority import build_stage_authority

            if validated_producer_set is None:
                raise RuntimeError("strict aggregate lacks typed producer validation")
            authority_memory_observation = controller_memory_observation(
                role="river_strict_authority_controller",
            )
            stage_adjudication = build_stage_authority(
                source_root=source_root,
                stage_drive_root=stage_drive_root,
                execution=execution,
                registry=registry,
                dependency_receipt=dependency_receipt,
                validated_producers=validated_producer_set,
                installed_aggregate_result=installed_result,
                aggregate_lane_handles=queue_receipt["handles"],
                local_scratch_root=local_scratch_root / "authority_finalization",
                progress_callback=emit_token_progress,
            )
    except Exception as error:
        finalization_error = error
        finalization_traceback = error.__traceback__
    finally:
        if aggregate_scratch is not None:
            try:
                if (
                    aggregate_scratch.name != "validated_producer_scratch"
                    or aggregate_scratch.parent != local_scratch_root.resolve()
                ):
                    raise RuntimeError("unsafe aggregate staging cleanup target")
                if aggregate_scratch.exists():
                    shutil.rmtree(aggregate_scratch)
                aggregate_scratch_deleted = not aggregate_scratch.exists()
                if not aggregate_scratch_deleted:
                    raise RuntimeError("aggregate staging cleanup did not complete")
            except Exception as cleanup_error:
                cleanup_failures.append(
                    {
                        "scope": "aggregate_staging",
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                )
        if continuation_scratch is not None:
            try:
                if (
                    continuation_scratch.name != "continuation_staging"
                    or continuation_scratch.parent != local_scratch_root.resolve()
                ):
                    raise RuntimeError("unsafe continuation staging cleanup target")
                if continuation_scratch.exists():
                    shutil.rmtree(continuation_scratch)
                continuation_scratch_deleted = not continuation_scratch.exists()
                if not continuation_scratch_deleted:
                    raise RuntimeError("continuation staging cleanup did not complete")
            except Exception as cleanup_error:
                cleanup_failures.append(
                    {
                        "scope": "continuation_staging",
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                )
        if finalization_error is not None:
            emit_token_progress(
                {
                    "event": "token_finalization_failed",
                    "error_type": type(finalization_error).__name__,
                    "error": str(finalization_error),
                    "owned_scratch_cleanup_failures": cleanup_failures,
                }
            )
        elif cleanup_failures:
            emit_token_progress(
                {
                    "event": "token_finalization_failed",
                    "error_type": "OwnedScratchCleanupError",
                    "error": "owned scratch cleanup failed after finalization",
                    "owned_scratch_cleanup_failures": cleanup_failures,
                }
            )
    if finalization_error is not None:
        raise finalization_error.with_traceback(finalization_traceback)
    if cleanup_failures:
        raise RuntimeError(
            "owned scratch cleanup failed after finalization: "
            + json.dumps(cleanup_failures, sort_keys=True)
        )
    try:
        aggregate_staging_summary = None
        if aggregate_scratch is not None:
            if maximum_root_materialization_bytes is None:
                raise RuntimeError("aggregate root materialization bound is absent")
            if not isinstance(aggregate_phase_disk_preflight, Mapping):
                raise RuntimeError("aggregate disk preflight receipt is absent")
            if not isinstance(aggregate_local_archive_post_worker_readback, Mapping):
                raise RuntimeError("aggregate archive readback receipt is absent")
            aggregate_staging_summary = _build_aggregate_staging_summary(
                aggregate_staging_receipts=aggregate_staging_receipts,
                maximum_root_materialization_bytes=(
                    maximum_root_materialization_bytes
                ),
                aggregate_phase_disk_preflight=aggregate_phase_disk_preflight,
                aggregate_local_archive_post_worker_readback=(
                    aggregate_local_archive_post_worker_readback
                ),
                scratch_deleted=aggregate_scratch_deleted,
            )
        continuation_staging_summary = None
        if isinstance(local_continuation_staging, Mapping):
            continuation_staging_summary = _build_continuation_staging_summary(
                local_continuation_staging=local_continuation_staging,
                scratch_deleted=continuation_scratch_deleted,
            )
        run_receipt = with_self_hash(
            {
            "schema_version": "stage21_notebook_run_receipt_v5",
            "stage": stage,
            "token": token,
            "lane_count": 48,
            "requested_workers": parse_requested_workers(requested_workers),
            "requested_workers_mode": queue_receipt["requested_workers_mode"],
            "detected_logical_cpu_count": queue_receipt[
                "detected_logical_cpu_count"
            ],
            "runtime_worker_limit": queue_receipt["runtime_worker_limit"],
            "resolved_live_worker_target": queue_receipt["live_worker_target"],
            "worker_count_clipped_by_cpu": queue_receipt[
                "worker_count_clipped_by_cpu"
            ],
            "worker_resolution_rule": queue_receipt[
                "worker_resolution_rule"
            ],
            "dependency": dependency_receipt,
            "queue": queue_receipt,
            "result": publication,
            "installed_result_validation_receipt": installed_result.document,
            "stage_adjudication": stage_adjudication,
            "authority_memory_observation": authority_memory_observation,
            "aggregate_staging": aggregate_staging_summary,
            "continuation_staging": continuation_staging_summary,
            "owned_scratch_cleanup_completed_before_return": True,
            "finalization_and_summary_failure_event_boundary": True,
            "local_scratch_preflight": local_scratch_preflight,
            "legacy_completed_results": (
                {
                    "enabled_for_runtime_resume": False,
                    "enabled_only_for_strict_aggregate": (
                        row["kind"] == "strict_aggregate"
                    ),
                    "contract_self_sha256": legacy_result_contract[
                        "self_sha256"
                    ],
                    "exact_contracted_archive_count": len(
                        legacy_result_contract[
                            "known_complete_returned_archives"
                        ]
                    ),
                    "legacy_source_files_modified": False,
                    "legacy_runtime_tree_read": False,
                }
                if legacy_result_contract is not None
                else None
            ),
            "status": (
                stage_adjudication["status"]
                if stage_adjudication is not None
                else "COMPLETE_UNADJUDICATED_RESULT_RETURNED"
            ),
            }
        )
    except Exception as error:
        emit_token_progress(
            {
                "event": "token_finalization_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "owned_scratch_cleanup_completed": True,
            }
        )
        raise
    return run_receipt
