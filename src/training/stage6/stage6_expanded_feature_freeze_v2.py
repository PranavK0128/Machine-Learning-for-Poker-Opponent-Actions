"""Additive R2 Notebook 12 runtime for an equivalent, faster feature freeze.

This module does not mutate or identify the historical R1 implementation.  Its
run identity is bound only to the R2 implementation and a newly executed exact
frame-equivalence/material-speedup preflight.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage6_behavior_tournament import MODEL_SOURCE_COLUMNS, feature_matrix, feature_names
from stage6_expanded_history_acceleration_v2 import (
    FeatureHistoryHeartbeatV2,
    PREFERRED_COLAB_HOST_V2,
    build_dyad_history_features_v2,
    build_line_prefix_features_v2,
    build_preflight_equivalence_binding_v2,
    implementation_binding_v2 as history_implementation_binding_v2,
)
from stage6_expanded_layers import (
    DEVELOPMENT_CUTOFF_UTC,
    FEATURE_SOURCE_COLUMNS,
    LOGICAL_LANE_COUNT,
    canonical_json_bytes,
    seal_document,
    temporal_role,
    validate_document,
)
from stage6_notebooks02_07_runtime import (
    NOTEBOOK01_RUN_ID,
    NOTEBOOK01_SPLIT_ROOT,
    _lane_paths,
    materialization_root,
    sha256_file,
)


COMMON_IDENTITY_COLUMNS = (
    "row_id", "game_id", "hand_id", "decision_id", "decision_utc",
    "decision_order", "street", "action_label",
)
SUPPORTED_ACCELERATORS_V2 = ("CPU", "GPU", "TPU")
FEATURE_MANIFEST_SCHEMA_V2 = "stage6_expanded_feature_freeze_manifest_v1"
R2_OUTPUT_AFFECTING_SOURCE_MEMBERS = (
    "src/stage6_expanded_feature_freeze_v2.py",
    "src/stage6_expanded_history_acceleration_v2.py",
    "src/stage6_behavior_tournament.py",
    "src/stage6_expanded_layers.py",
    "src/stage6_notebooks02_07_runtime.py",
    "src/stage6_expanded_feature_freeze.py",
)


def _safe_base_feature_name(name: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"base_{safe.strip('_')}"


BASE_FEATURE_COLUMNS = tuple(_safe_base_feature_name(name) for name in feature_names())
if len(BASE_FEATURE_COLUMNS) != len(set(BASE_FEATURE_COLUMNS)):
    raise RuntimeError("base feature name normalization is not unique")


def runtime_implementation_binding_v2() -> dict[str, Any]:
    stage_root = Path(__file__).resolve().parents[1]
    members = []
    for relative_path in R2_OUTPUT_AFFECTING_SOURCE_MEMBERS:
        path = stage_root / relative_path
        raw = path.read_bytes()
        members.append({
            "relative_path": relative_path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    history = history_implementation_binding_v2()
    history_member = next(
        row for row in members
        if row["relative_path"] == "src/stage6_expanded_history_acceleration_v2.py"
    )
    if (
        history.get("relative_path") != history_member["relative_path"]
        or history.get("bytes") != history_member["bytes"]
        or history.get("sha256") != history_member["sha256"]
    ):
        raise RuntimeError("R2 history implementation binding differs from full source inventory")
    return seal_document({
        "schema_version": "stage6_expanded_feature_freeze_implementation_binding_v5",
        "member_order": list(R2_OUTPUT_AFFECTING_SOURCE_MEMBERS),
        "members": members,
        "history_member_binding_self_sha256": history["self_sha256"],
        "historical_r1_implementation_hash_emitted": False,
        "r1_source_bytes_mutated": False,
    })


R2_IMPLEMENTATION_BINDING = runtime_implementation_binding_v2()
R2_IMPLEMENTATION_DIGEST = R2_IMPLEMENTATION_BINDING["self_sha256"]
R2_EQUIVALENCE_GATE_SPEC = seal_document({
    "schema_version": "stage6_expanded_history_equivalence_gate_spec_v2",
    "oracle": "frozen_r1_exact_frames_without_r1_implementation_hash",
    "adversarial_generator": "cross_lane_gt64_overlap_multiway_v2",
    "benchmark_generator": "80_disjoint_dyads_x20_completed_hands_v2",
    "required_exact_properties": [
        "columns", "column_order", "row_id_order", "dtype", "null_mask", "finite_values",
    ],
    "minimum_speedup": 1.25,
    "maximum_benchmark_rows": 200_000,
    "historical_r1_implementation_hash_emitted": False,
})


def runtime_package_precommit_v5() -> dict[str, Any]:
    return seal_document({
        "schema_version": "stage6_expanded_feature_freeze_precommit_v5",
        "status": "FROZEN_BEFORE_R5_DEVELOPMENT_FEATURE_READ",
        "r2_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
        "r2_implementation_binding": R2_IMPLEMENTATION_BINDING,
        "r2_equivalence_gate_spec": R2_EQUIVALENCE_GATE_SPEC,
        "logical_data_lane_count": LOGICAL_LANE_COUNT,
        "requested_workers_default": "auto",
        "preferred_colab_host": "CPU High-RAM",
        "accepted_runtime_types": ["CPU", "GPU", "TPU"],
        "feature_derivation_execution": "host_cpu_process_work",
        "tpu_acceleration_claimed": False,
        "action_timing_layer_included": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "historical_r1_implementation_hash_emitted": False,
        "json_publication_recovery": (
            "authenticated_claim_then_atomic_tombstone_cleanup_v2"
        ),
    })


R5_PACKAGE_PRECOMMIT = runtime_package_precommit_v5()


def feature_run_id_from_bindings_v5(
    implementation_digest: str,
    equivalence_digest: str,
    package_precommit_digest: str,
) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_layer_feature_freeze_v5",
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_root": NOTEBOOK01_SPLIT_ROOT,
        "r2_implementation": implementation_digest,
        "equivalence_gate_spec_self_sha256": equivalence_digest,
        "r5_package_precommit_self_sha256": package_precommit_digest,
        "development_cutoff_utc_exclusive": DEVELOPMENT_CUTOFF_UTC,
    })).hexdigest()


FEATURE_RUN_ID_V2 = feature_run_id_from_bindings_v5(
    R2_IMPLEMENTATION_DIGEST,
    R2_EQUIVALENCE_GATE_SPEC["self_sha256"],
    R5_PACKAGE_PRECOMMIT["self_sha256"],
)
# Compatibility names consumed by the downstream rebind adapter.
FEATURE_RUN_ID = FEATURE_RUN_ID_V2
SCIENCE_IMPLEMENTATION_DIGEST = R2_IMPLEMENTATION_DIGEST


def runtime_package_binding_v5() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    precommit_path = package_root / "r5_feature_freeze_precommit.json"
    if not manifest_path.is_file() or not precommit_path.is_file():
        raise RuntimeError("R5 runtime package manifest/precommit are required")
    package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    validate_document(package_manifest)
    validate_document(precommit)
    if precommit != R5_PACKAGE_PRECOMMIT:
        raise RuntimeError("R5 runtime package precommit identity changed")
    if (
        package_manifest.get("schema_version")
        != "stage6_expanded_feature_freeze_input_package_manifest_v5"
        or package_manifest.get("revision") != "R5"
        or package_manifest.get("feature_run_id") != FEATURE_RUN_ID_V2
        or package_manifest.get("r2_implementation_self_sha256") != R2_IMPLEMENTATION_DIGEST
        or package_manifest.get("precommit_self_sha256") != precommit["self_sha256"]
        or package_manifest.get("historical_r1_implementation_hash_emitted") is not False
    ):
        raise RuntimeError("R5 runtime package manifest authority is invalid")
    members = package_manifest.get("members")
    order = package_manifest.get("member_order")
    if (
        not isinstance(members, list)
        or not isinstance(order, list)
        or [row.get("path") if isinstance(row, Mapping) else None for row in members] != order
        or len(order) != len(set(order))
    ):
        raise RuntimeError("R5 runtime package member inventory is invalid")
    precommit_identity = None
    for identity in members:
        pure = PurePosixPath(str(identity["path"]))
        if pure.is_absolute() or ".." in pure.parts or "\\" in str(identity["path"]):
            raise RuntimeError("R5 runtime package member path is unsafe")
        artifact = package_root / pure.as_posix()
        raw = artifact.read_bytes()
        if len(raw) != identity.get("bytes") or hashlib.sha256(raw).hexdigest() != identity.get("sha256"):
            raise RuntimeError(f"R5 runtime package member identity changed: {identity['path']}")
        if identity["path"] == "r5_feature_freeze_precommit.json":
            precommit_identity = identity
    expected_precommit_raw = canonical_json_bytes(R5_PACKAGE_PRECOMMIT) + b"\n"
    if (
        precommit_identity is None
        or precommit_identity.get("bytes") != len(expected_precommit_raw)
        or precommit_identity.get("sha256") != hashlib.sha256(expected_precommit_raw).hexdigest()
    ):
        raise RuntimeError("R5 package manifest does not bind the exact precommit")
    return seal_document({
        "schema_version": "stage6_expanded_feature_runtime_package_binding_v5",
        "feature_run_id": FEATURE_RUN_ID_V2,
        "package_manifest_self_sha256": package_manifest["self_sha256"],
        "precommit_self_sha256": precommit["self_sha256"],
        "member_order": order,
        "member_identity_root_sha256": hashlib.sha256(
            canonical_json_bytes(members)
        ).hexdigest(),
    })


def expanded_feature_root(stage_drive_root: Path) -> Path:
    return stage_drive_root / "runtime_state" / "expanded_layers" / "feature_freezes_v2" / FEATURE_RUN_ID_V2


def _file_identity(path: Path, relative_to: Path, row_count: int) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(row_count),
    }


def _json_claim_directory(path: Path) -> Path:
    return path.with_name(f".{path.name}.stage6-json-claim")


def _json_artifact_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"R5 JSON artifact is outside its run root: {path}") from error


def _json_claim_document(path: Path, root: Path, raw: bytes) -> dict[str, Any]:
    claim_directory = _json_claim_directory(path)
    document = json.loads(raw.decode("utf-8"))
    validate_document(document)
    return seal_document({
        "schema_version": "stage6_expanded_feature_json_publication_claim_v1",
        "feature_run_id": FEATURE_RUN_ID_V2,
        "science_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
        "artifact_relative_path": _json_artifact_relative_path(path, root),
        "claim_directory_relative_path": _json_artifact_relative_path(
            claim_directory, root
        ),
        "staged_artifact_name": "staged.json",
        "expected_bytes": len(raw),
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "document_self_sha256": document["self_sha256"],
        "ownership_authority": "same_run_same_artifact_truncated_final_recovery_only",
    })


def _read_authenticated_json_claim(
    path: Path,
    root: Path,
    *,
    expected_raw: bytes | None = None,
) -> tuple[Path, bytes, dict[str, Any]] | None:
    claim_directory = _json_claim_directory(path)
    if not claim_directory.exists():
        return None
    if not claim_directory.is_dir():
        raise RuntimeError(f"conflicting R5 JSON claim path exists: {claim_directory}")
    entries = {entry.name: entry for entry in claim_directory.iterdir()}
    if set(entries) != {"claim.json", "staged.json"}:
        raise RuntimeError(f"R5 JSON claim inventory is invalid: {claim_directory}")
    claim_raw = entries["claim.json"].read_bytes()
    staged_raw = entries["staged.json"].read_bytes()
    try:
        claim = json.loads(claim_raw.decode("utf-8"))
        staged_document = json.loads(staged_raw.decode("utf-8"))
        validate_document(claim)
        validate_document(staged_document)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"R5 JSON claim bytes are invalid: {claim_directory}") from error
    if canonical_json_bytes(claim) + b"\n" != claim_raw:
        raise RuntimeError(f"R5 JSON claim serialization changed: {claim_directory}")
    if canonical_json_bytes(staged_document) + b"\n" != staged_raw:
        raise RuntimeError(f"R5 staged JSON serialization changed: {claim_directory}")
    expected_claim = _json_claim_document(path, root, staged_raw)
    if (
        claim != expected_claim
        or claim.get("feature_run_id") != FEATURE_RUN_ID_V2
        or claim.get("science_implementation_self_sha256")
        != R2_IMPLEMENTATION_DIGEST
        or claim.get("expected_bytes") != len(staged_raw)
        or claim.get("expected_sha256")
        != hashlib.sha256(staged_raw).hexdigest()
        or staged_document.get("self_sha256")
        != claim.get("document_self_sha256")
        or (expected_raw is not None and staged_raw != expected_raw)
    ):
        raise RuntimeError(f"R5 JSON claim identity changed: {claim_directory}")
    return entries["staged.json"], staged_raw, claim


def _json_claim_cleanup_prefix(path: Path, claim: Mapping[str, Any]) -> str:
    return (
        f"{_json_claim_directory(path).name}.cleanup-"
        f"{claim['self_sha256'][:24]}-"
    )


def _after_json_claim_cleanup_event_v5(event: str, tombstone: Path) -> None:
    """Crash-injection seam; production execution deliberately does nothing."""


def _cleanup_json_tombstone_v5(
    tombstone: Path,
    *,
    expected_claim_raw: bytes,
    expected_staged_raw: bytes,
) -> bool:
    """Best-effort cleanup of one exact bound tombstone.

    A partially deleted tombstone is valid after a crash.  Unexpected entries or
    foreign bytes are never removed and, because the active claim was already
    atomically renamed away, never regain publication authority.
    """
    if not tombstone.exists():
        return True
    if not tombstone.is_dir():
        return False
    entries = {entry.name: entry for entry in tombstone.iterdir()}
    if not set(entries).issubset({"claim.json", "staged.json"}):
        return False
    expected = {
        "claim.json": expected_claim_raw,
        "staged.json": expected_staged_raw,
    }
    if any(entry.read_bytes() != expected[name] for name, entry in entries.items()):
        return False
    for name in ("claim.json", "staged.json"):
        member = tombstone / name
        if member.exists():
            member.unlink()
            _after_json_claim_cleanup_event_v5(f"after_{name}_delete", tombstone)
    try:
        tombstone.rmdir()
    except OSError:
        return False
    return True


def _resume_json_claim_cleanup_v5(path: Path, root: Path, raw: bytes) -> None:
    claim = _json_claim_document(path, root, raw)
    prefix = _json_claim_cleanup_prefix(path, claim)
    expected_claim_raw = canonical_json_bytes(claim) + b"\n"
    for tombstone in sorted(path.parent.glob(f"{prefix}*"), key=lambda row: row.name):
        _cleanup_json_tombstone_v5(
            tombstone,
            expected_claim_raw=expected_claim_raw,
            expected_staged_raw=raw,
        )


def _remove_authenticated_json_claim(path: Path, root: Path, raw: bytes) -> None:
    observed = _read_authenticated_json_claim(path, root, expected_raw=raw)
    if observed is None:
        _resume_json_claim_cleanup_v5(path, root, raw)
        return
    if not path.is_file() or path.read_bytes() != raw:
        raise RuntimeError(f"R5 JSON final changed before claim cleanup: {path}")
    _resume_json_claim_cleanup_v5(path, root, raw)
    claim_directory = _json_claim_directory(path)
    claim = observed[2]
    tombstone = claim_directory.with_name(
        _json_claim_cleanup_prefix(path, claim) + uuid.uuid4().hex
    )
    claim_directory.rename(tombstone)
    _after_json_claim_cleanup_event_v5("after_cleanup_rename", tombstone)
    _cleanup_json_tombstone_v5(
        tombstone,
        expected_claim_raw=canonical_json_bytes(claim) + b"\n",
        expected_staged_raw=raw,
    )


def _ensure_authenticated_json_claim(path: Path, root: Path, raw: bytes) -> Path:
    observed = _read_authenticated_json_claim(path, root, expected_raw=raw)
    if observed is not None:
        return observed[0]
    claim_directory = _json_claim_directory(path)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{path.name}.{hashlib.sha256(raw).hexdigest()[:16]}.",
        suffix=".stage6-json-claim-staging",
        dir=path.parent,
    ))
    try:
        claim = _json_claim_document(path, root, raw)
        for name, payload in (
            ("staged.json", raw),
            ("claim.json", canonical_json_bytes(claim) + b"\n"),
        ):
            with (temporary / name).open("xb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
        try:
            temporary.rename(claim_directory)
        except OSError:
            if not claim_directory.exists():
                raise
            (temporary / "claim.json").unlink()
            (temporary / "staged.json").unlink()
            temporary.rmdir()
        observed = _read_authenticated_json_claim(path, root, expected_raw=raw)
        if observed is None:
            raise RuntimeError(f"R5 JSON claim publication failed: {claim_directory}")
        return observed[0]
    except Exception:
        if temporary.exists():
            expected_names = {entry.name for entry in temporary.iterdir()}
            if expected_names.issubset({"claim.json", "staged.json"}):
                for name in expected_names:
                    (temporary / name).unlink()
                temporary.rmdir()
        raise


def _publish_json_staging_to_final_v4(staging: Path, destination: Path) -> None:
    """Create the JSON final from an authenticated staging claim."""
    with staging.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def _recover_claimed_json_if_needed(path: Path, root: Path) -> bool:
    observed = _read_authenticated_json_claim(path, root)
    if observed is None:
        if path.is_file():
            try:
                current = path.read_bytes()
                _resume_json_claim_cleanup_v5(path, root, current)
            except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
                pass
        return False
    staging, raw, _claim = observed
    if path.exists():
        current = path.read_bytes()
        if current == raw:
            _remove_authenticated_json_claim(path, root, raw)
            return True
        if len(current) >= len(raw) or not raw.startswith(current):
            raise RuntimeError(f"foreign bytes block R5 JSON recovery: {path}")
        path.unlink()
    _publish_json_staging_to_final_v4(staging, path)
    if not path.is_file() or path.read_bytes() != raw:
        raise RuntimeError(f"R5 JSON publication readback failed: {path}")
    _remove_authenticated_json_claim(path, root, raw)
    return True


def _exclusive_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    validate_document(document)
    raw = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    run_root = path.parent if root is None else Path(root)
    if path.exists():
        current = path.read_bytes()
        if current == raw:
            if _json_claim_directory(path).exists():
                _remove_authenticated_json_claim(path, run_root, raw)
            else:
                _resume_json_claim_cleanup_v5(path, run_root, raw)
            return
        observed = _read_authenticated_json_claim(
            path, run_root, expected_raw=raw
        )
        if observed is None:
            raise RuntimeError(f"conflicting R5 checkpoint already exists: {path}")
        if len(current) >= len(raw) or not raw.startswith(current):
            raise RuntimeError(f"foreign bytes block R5 JSON recovery: {path}")
        path.unlink()
    staging = _ensure_authenticated_json_claim(path, run_root, raw)
    _publish_json_staging_to_final_v4(staging, path)
    if not path.is_file() or path.read_bytes() != raw:
        raise RuntimeError(f"R5 JSON publication readback failed: {path}")
    _remove_authenticated_json_claim(path, run_root, raw)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    """Compatibility name with create-exclusive, identical-byte resume."""
    _exclusive_json(path, document)


def _publish_staging_to_final_v2(staging: Path, destination: Path) -> None:
    """Create the final from authenticated staging; kept separate for crash tests."""
    with staging.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def _validate_parquet_identity_v2(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_rows: int,
) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == expected_bytes
            and sha256_file(path) == expected_sha256
            and pq.ParquetFile(path).metadata.num_rows == expected_rows
        )
    except (OSError, pa.ArrowInvalid):
        return False


def _publish_parquet_v2(frame: pd.DataFrame, destination: Path, root: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage6-expanded-feature-v2-") as temporary:
        local = Path(temporary) / destination.name
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), local,
            compression="zstd", compression_level=6, use_dictionary=True,
            write_statistics=True,
        )
        expected_bytes = local.read_bytes()
        expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
        expected_rows = len(frame)
        relative_destination = destination.relative_to(root).as_posix()
        staging = destination.with_name(
            f".{destination.name}.{expected_sha256[:16]}.stage6-staging"
        )
        owner_path = destination.with_name(
            f".{destination.name}.stage6-staging-owner.json"
        )
        owner = seal_document({
            "schema_version": "stage6_expanded_feature_staging_owner_v1",
            "feature_run_id": FEATURE_RUN_ID_V2,
            "science_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
            "artifact_relative_path": relative_destination,
            "staging_relative_path": staging.relative_to(root).as_posix(),
            "expected_bytes": len(expected_bytes),
            "expected_sha256": expected_sha256,
            "expected_row_count": expected_rows,
            "ownership_authority": "same_run_same_artifact_crash_recovery_only",
        })

        def exact_owner_receipt() -> bool:
            if not owner_path.is_file():
                return False
            try:
                observed = json.loads(owner_path.read_text(encoding="utf-8"))
                validate_document(observed)
            except (OSError, json.JSONDecodeError, RuntimeError):
                return False
            return observed == owner

        if destination.exists():
            if _validate_parquet_identity_v2(
                destination, expected_bytes=len(expected_bytes),
                expected_sha256=expected_sha256, expected_rows=expected_rows,
            ):
                if exact_owner_receipt():
                    if staging.exists():
                        staging.unlink()
                    owner_path.unlink()
                return _file_identity(destination, root, expected_rows)
            if not exact_owner_receipt():
                raise RuntimeError(f"conflicting R2 parquet already exists: {destination}")
            destination.unlink()
        else:
            _exclusive_json(owner_path, owner, root=root)
        if not exact_owner_receipt():
            raise RuntimeError(f"R2 parquet staging ownership is invalid: {destination}")
        if staging.exists() and not _validate_parquet_identity_v2(
            staging, expected_bytes=len(expected_bytes),
            expected_sha256=expected_sha256, expected_rows=expected_rows,
        ):
            staging.unlink()
        if not staging.exists():
            with staging.open("xb") as target:
                target.write(expected_bytes)
                target.flush()
                os.fsync(target.fileno())
        if not _validate_parquet_identity_v2(
            staging, expected_bytes=len(expected_bytes),
            expected_sha256=expected_sha256, expected_rows=expected_rows,
        ):
            raise RuntimeError(f"R2 parquet staging readback failed: {staging}")
        _publish_staging_to_final_v2(staging, destination)
        if not _validate_parquet_identity_v2(
            destination, expected_bytes=len(expected_bytes),
            expected_sha256=expected_sha256, expected_rows=expected_rows,
        ):
            raise RuntimeError(f"R2 feature publication readback failed: {destination}")
        if not exact_owner_receipt():
            raise RuntimeError(f"R2 parquet ownership changed before cleanup: {destination}")
        staging.unlink()
        owner_path.unlink()
    identity = _file_identity(destination, root, len(frame))
    if pq.ParquetFile(destination).metadata.num_rows != len(frame):
        raise RuntimeError(f"R2 feature publication row mismatch: {destination}")
    return identity


def _load_development_source_v2(
    notebook01_root: Path,
    heartbeat: FeatureHistoryHeartbeatV2 | None = None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys((*FEATURE_SOURCE_COLUMNS, *MODEL_SOURCE_COLUMNS)))
    frames: list[pd.DataFrame] = []
    if heartbeat is not None:
        heartbeat.start_phase("load_48_global_source_lanes")
    for lane in range(LOGICAL_LANE_COUNT):
        decision_path, _, _ = _lane_paths(notebook01_root, lane)
        maximum = pq.read_table(decision_path, columns=["decision_utc"]).column(0)
        maximum_utc = max(str(value.as_py()) for value in maximum) if len(maximum) else ""
        if maximum_utc >= DEVELOPMENT_CUTOFF_UTC:
            raise RuntimeError("fresh adjudication rows are present; R2 Notebook 12 cannot open them")
        frame = pq.read_table(decision_path, columns=columns).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
        if heartbeat is not None:
            heartbeat.update(
                phase="load_48_global_source_lanes", completed=lane + 1,
                denominator=LOGICAL_LANE_COUNT, checkpoint=f"lane-{lane:02d}",
                force=lane + 1 == LOGICAL_LANE_COUNT,
            )
    source = pd.concat(frames, ignore_index=True)
    if source["row_id"].astype(str).duplicated().any():
        raise RuntimeError("R2 expanded feature source duplicates row_id across lanes")
    return source


def build_feature_frames_v2(
    source: pd.DataFrame,
    *,
    requested_workers: str | int = "auto",
    accelerator: str = "CPU",
    allow_fresh_adjudication: bool = False,
    heartbeat: FeatureHistoryHeartbeatV2 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accelerator_name = str(accelerator).strip().upper()
    if accelerator_name not in SUPPORTED_ACCELERATORS_V2:
        raise ValueError(f"accelerator must be one of {SUPPORTED_ACCELERATORS_V2}")
    line = build_line_prefix_features_v2(source)
    dyad, history_receipt = build_dyad_history_features_v2(
        source, requested_workers=requested_workers,
        accelerator=accelerator_name, heartbeat=heartbeat,
    )
    matrix = feature_matrix(source)
    common = source.loc[:, COMMON_IDENTITY_COLUMNS].copy()
    common["source_lane_id"] = source["source_lane_id"].astype(str)
    hand_roles = source.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["decision_utc"].max().astype(str).map(temporal_role).rename("temporal_role").reset_index()
    common = common.merge(
        hand_roles, on=["game_id", "hand_id"], how="left",
        validate="many_to_one", sort=False,
    )
    if common["temporal_role"].isna().any():
        raise RuntimeError("whole-hand temporal role assignment is incomplete")
    if not allow_fresh_adjudication and common["temporal_role"].eq("fresh_adjudication").any():
        raise RuntimeError("fresh target entered the R2 development feature freeze")
    for index, name in enumerate(BASE_FEATURE_COLUMNS):
        common[name] = matrix[:, index]
    if not common["row_id"].astype(str).equals(source["row_id"].astype(str)):
        raise RuntimeError("R2 common feature row order changed")
    return common, line, dyad, history_receipt


def build_feature_frames(
    source: pd.DataFrame,
    *,
    allow_fresh_adjudication: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common, line, dyad, _ = build_feature_frames_v2(
        source, requested_workers="auto", accelerator="CPU",
        allow_fresh_adjudication=allow_fresh_adjudication,
    )
    return common, line, dyad


def whole_hand_role_binding(common: pd.DataFrame) -> dict[str, Any]:
    per_hand_counts = common.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["temporal_role"].nunique()
    if not per_hand_counts.eq(1).all():
        raise RuntimeError("a game hand spans multiple R2 temporal roles")
    assignments = common.loc[:, ["game_id", "hand_id", "temporal_role"]].drop_duplicates().sort_values(
        ["game_id", "hand_id"], kind="mergesort"
    ).reset_index(drop=True)
    records = [[str(row.game_id), str(row.hand_id), str(row.temporal_role)] for row in assignments.itertuples(index=False)]
    return seal_document({
        "schema_version": "stage6_expanded_whole_hand_temporal_assignment_binding_v1",
        "assignment_unit": "game_id_plus_hand_id",
        "assignment_time": "maximum_decision_utc",
        "boundary_behavior": "later_partition_wins",
        "hand_count": len(assignments),
        "role_hand_counts": {
            str(key): int(value) for key, value in assignments["temporal_role"].value_counts(sort=False).items()
        },
        "assignment_root_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    })


def _valid_completed_manifest_v2(
    root: Path,
    runtime_package_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = root / "feature-freeze-manifest.json"
    if not path.is_file():
        _recover_claimed_json_if_needed(path, root)
    if not path.is_file():
        return None
    _recover_claimed_json_if_needed(path, root)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "unowned or foreign truncated R5 feature manifest exists"
        ) from error
    validate_document(manifest)
    if (
        manifest.get("schema_version") != FEATURE_MANIFEST_SCHEMA_V2
        or manifest.get("execution_revision") != "R5"
        or manifest.get("status") != "PASS_DEVELOPMENT_FEATURE_FREEZE"
        or manifest.get("feature_run_id") != FEATURE_RUN_ID_V2
        or manifest.get("science_implementation_self_sha256") != R2_IMPLEMENTATION_DIGEST
        or manifest.get("completed_lane_count") != LOGICAL_LANE_COUNT
        or manifest.get("fresh_targets_opened") is not False
    ):
        raise RuntimeError("completed R2 expanded feature manifest is invalid")
    implementation = manifest.get("r2_implementation_binding")
    equivalence = manifest.get("r2_equivalence_binding")
    stored_package_binding = manifest.get("runtime_package_binding")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(equivalence, Mapping)
        or not isinstance(stored_package_binding, Mapping)
    ):
        raise RuntimeError("completed R2 manifest lacks implementation/equivalence/package binding")
    validate_document(implementation)
    validate_document(equivalence)
    validate_document(stored_package_binding)
    if implementation.get("self_sha256") != R2_IMPLEMENTATION_DIGEST:
        raise RuntimeError("completed R2 implementation binding changed")
    validate_document(runtime_package_binding)
    if stored_package_binding != runtime_package_binding:
        raise RuntimeError("completed R2 runtime package identity changed")
    if (
        equivalence.get("feature_run_id") != FEATURE_RUN_ID_V2
        or equivalence.get("r2_implementation_self_sha256") != R2_IMPLEMENTATION_DIGEST
        or equivalence.get("equivalence_gate_spec") != R2_EQUIVALENCE_GATE_SPEC
        or not isinstance(equivalence.get("preflight_receipt"), Mapping)
        or equivalence.get("exact_equivalence_pass") is not True
        or equivalence.get("material_speedup_pass") is not True
        or equivalence.get("fresh_targets_opened") is not False
    ):
        raise RuntimeError("completed R2 equivalence binding is invalid")
    role_binding = manifest.get("whole_hand_temporal_assignment_binding")
    if not isinstance(role_binding, Mapping):
        raise RuntimeError("completed R2 whole-hand binding is missing")
    validate_document(role_binding)
    role_hand_counts = role_binding.get("role_hand_counts")
    if (
        role_binding.get("schema_version")
        != "stage6_expanded_whole_hand_temporal_assignment_binding_v1"
        or role_binding.get("assignment_unit") != "game_id_plus_hand_id"
        or role_binding.get("assignment_time") != "maximum_decision_utc"
        or role_binding.get("boundary_behavior") != "later_partition_wins"
        or not isinstance(role_binding.get("hand_count"), int)
        or role_binding.get("hand_count") < 0
        or not isinstance(role_hand_counts, Mapping)
        or not all(
            isinstance(role, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for role, count in role_hand_counts.items()
        )
        or sum(role_hand_counts.values()) != role_binding["hand_count"]
        or not isinstance(role_binding.get("assignment_root_sha256"), str)
        or len(role_binding["assignment_root_sha256"]) != 64
    ):
        raise RuntimeError("completed R2 whole-hand binding is invalid")
    expected_lane_ids = [f"lane-{lane:02d}" for lane in range(LOGICAL_LANE_COUNT)]
    lanes = manifest.get("lanes")
    fixed_lane_order = manifest.get("fixed_lane_order")
    if (
        not isinstance(lanes, list)
        or len(lanes) != LOGICAL_LANE_COUNT
        or fixed_lane_order != expected_lane_ids
        or [lane.get("lane_id") if isinstance(lane, Mapping) else None for lane in lanes]
        != expected_lane_ids
    ):
        raise RuntimeError("completed R2 lane inventory is invalid")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or row_count < 0:
        raise RuntimeError("completed R2 row count is invalid")
    seen_paths: set[str] = set()
    row_sums = {"common": 0, "line": 0, "dyad": 0}
    for lane_index, lane in enumerate(lanes):
        lane_id = expected_lane_ids[lane_index]
        lane_rows: list[int] = []
        for key in ("common", "line", "dyad"):
            identity = lane.get(key)
            expected_path = f"{key}/{lane_id}.parquet"
            if (
                not isinstance(identity, Mapping)
                or identity.get("relative_path") != expected_path
                or identity["relative_path"] in seen_paths
                or not isinstance(identity.get("bytes"), int)
                or identity["bytes"] <= 0
                or not isinstance(identity.get("sha256"), str)
                or len(identity["sha256"]) != 64
                or not isinstance(identity.get("row_count"), int)
                or identity["row_count"] < 0
            ):
                raise RuntimeError(f"completed R2 {lane_id} {key} identity is invalid")
            seen_paths.add(identity["relative_path"])
            lane_rows.append(identity["row_count"])
            row_sums[key] += identity["row_count"]
            artifact = root / identity["relative_path"]
            if (
                not artifact.is_file()
                or artifact.stat().st_size != identity["bytes"]
                or sha256_file(artifact) != identity["sha256"]
                or pq.ParquetFile(artifact).metadata.num_rows != identity["row_count"]
            ):
                raise RuntimeError(f"completed R2 feature {key} identity changed")
        if len(set(lane_rows)) != 1:
            raise RuntimeError(f"completed R2 {lane_id} output row counts differ")
    if any(total != row_count for total in row_sums.values()):
        raise RuntimeError("completed R2 lane row sums do not equal manifest row count")
    return manifest


def run_feature_freeze_v2(
    stage_drive_root: Path,
    requested_workers: str | int = "auto",
    accelerator: str = "CPU",
) -> dict[str, Any]:
    """Run an R2 freeze only after exact-equivalence and speedup preflight pass."""
    started = time.monotonic()
    accelerator_name = str(accelerator).strip().upper()
    if accelerator_name not in SUPPORTED_ACCELERATORS_V2:
        raise ValueError(f"accelerator must be one of {SUPPORTED_ACCELERATORS_V2}")
    target_root = expanded_feature_root(stage_drive_root)
    package_binding = runtime_package_binding_v5()
    completed = _valid_completed_manifest_v2(target_root, package_binding)
    if completed is not None:
        return {
            "status": "PASS", "feature_run_id": FEATURE_RUN_ID_V2,
            "manifest_path": str(target_root / "feature-freeze-manifest.json"),
            "manifest_self_sha256": completed["self_sha256"],
            "reused_completed_result": True, "fresh_targets_opened": False,
            "qualification_authority": False,
        }

    heartbeat = FeatureHistoryHeartbeatV2(disk_root=Path("/content") if Path("/content").is_dir() else Path.cwd())
    heartbeat.start()
    try:
        heartbeat.start_phase("r2_equivalence_and_speed_preflight")
        preflight = build_preflight_equivalence_binding_v2(repeats=1, minimum_speedup=1.25)
        equivalence_binding = seal_document({
            "schema_version": "stage6_expanded_feature_equivalence_run_binding_v2",
            "feature_run_id": FEATURE_RUN_ID_V2,
            "r2_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
            "equivalence_gate_spec": R2_EQUIVALENCE_GATE_SPEC,
            "preflight_receipt": preflight,
            "exact_equivalence_pass": True,
            "material_speedup_pass": True,
            "fresh_targets_opened": False,
        })
        heartbeat.update(
            phase="r2_equivalence_and_speed_preflight", completed=1, denominator=1,
            checkpoint="preflight-pass", force=True,
        )
        notebook01_root = materialization_root(stage_drive_root)
        source = _load_development_source_v2(notebook01_root, heartbeat)
        common, line, dyad, history_receipt = build_feature_frames_v2(
            source, requested_workers=requested_workers, accelerator=accelerator_name,
            heartbeat=heartbeat,
        )
        role_binding = whole_hand_role_binding(common)
        lanes: list[dict[str, Any]] = []
        heartbeat.start_phase("publish_48_exact_feature_lanes")
        line_index = line.set_index("row_id", drop=False)
        dyad_index = dyad.set_index("row_id", drop=False)
        for lane in range(LOGICAL_LANE_COUNT):
            lane_id = f"lane-{lane:02d}"
            common_lane = common.loc[common["source_lane_id"].eq(lane_id)].drop(columns="source_lane_id").reset_index(drop=True)
            exact_row_ids = common_lane["row_id"].astype(str).tolist()
            line_lane = line_index.loc[exact_row_ids].reset_index(drop=True)
            dyad_lane = dyad_index.loc[exact_row_ids].reset_index(drop=True)
            if (
                common_lane["row_id"].astype(str).tolist() != line_lane["row_id"].astype(str).tolist()
                or common_lane["row_id"].astype(str).tolist() != dyad_lane["row_id"].astype(str).tolist()
            ):
                raise RuntimeError(f"{lane_id} R2 exact row_id merge/order failed")
            lanes.append({
                "lane_id": lane_id,
                "common": _publish_parquet_v2(common_lane, target_root / "common" / f"{lane_id}.parquet", target_root),
                "line": _publish_parquet_v2(line_lane, target_root / "line" / f"{lane_id}.parquet", target_root),
                "dyad": _publish_parquet_v2(dyad_lane, target_root / "dyad" / f"{lane_id}.parquet", target_root),
            })
            heartbeat.update(
                phase="publish_48_exact_feature_lanes", completed=lane + 1,
                denominator=LOGICAL_LANE_COUNT, checkpoint=lane_id,
                force=lane + 1 == LOGICAL_LANE_COUNT,
            )
        manifest = seal_document({
            "schema_version": FEATURE_MANIFEST_SCHEMA_V2,
            "execution_revision": "R5",
            "status": "PASS_DEVELOPMENT_FEATURE_FREEZE",
            "feature_run_id": FEATURE_RUN_ID_V2,
            "notebook01_run_identity": NOTEBOOK01_RUN_ID,
            "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
            "science_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
            "r2_implementation_binding": R2_IMPLEMENTATION_BINDING,
            "r2_equivalence_binding": equivalence_binding,
            "runtime_package_binding": package_binding,
            "history_acceleration_receipt": history_receipt,
            "development_cutoff_utc_exclusive": DEVELOPMENT_CUTOFF_UTC,
            "row_count": len(common),
            "role_row_counts": {
                str(key): int(value) for key, value in common["temporal_role"].value_counts(sort=False).items()
            },
            "whole_hand_temporal_assignment_binding": role_binding,
            "base_feature_columns": list(BASE_FEATURE_COLUMNS),
            "line_feature_columns": [column for column in line.columns if column != "row_id"],
            "dyad_feature_columns": [column for column in dyad.columns if column not in {"row_id", "dyad_eligible"}],
            "completed_lane_count": len(lanes),
            "fixed_lane_order": [row["lane_id"] for row in lanes],
            "lanes": lanes,
            "requested_workers": requested_workers,
            "accelerator_environment": accelerator_name,
            "preferred_colab_host": PREFERRED_COLAB_HOST_V2,
            "feature_derivation_execution": "host_cpu_process_work",
            "tpu_acceleration_claimed": False,
            "action_timing_layer_included": False,
            "fresh_targets_opened": False,
            "qualification_authority": False,
            "assistant_installation_allowed": False,
            "action_changing_exploit_authority": False,
            "historical_r1_implementation_hash_emitted": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        _exclusive_json(
            target_root / "feature-freeze-manifest.json",
            manifest,
            root=target_root,
        )
        result = {
            "status": "PASS", "feature_run_id": FEATURE_RUN_ID_V2,
            "manifest_path": str(target_root / "feature-freeze-manifest.json"),
            "manifest_self_sha256": manifest["self_sha256"],
            "row_count": len(common), "reused_completed_result": False,
            "fresh_targets_opened": False, "qualification_authority": False,
        }
        del source, common, line, dyad, line_index, dyad_index
        gc.collect()
        return result
    finally:
        heartbeat.close()


def run_feature_freeze(
    stage_drive_root: Path,
    requested_workers: str | int = "auto",
    accelerator: str = "CPU",
) -> dict[str, Any]:
    return run_feature_freeze_v2(stage_drive_root, requested_workers, accelerator)
