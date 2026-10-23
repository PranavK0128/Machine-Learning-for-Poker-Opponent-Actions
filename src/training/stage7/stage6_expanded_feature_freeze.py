"""Notebook 12 runtime: freeze causal features for the expanded Stage 6 layers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage6_behavior_tournament import (
    MODEL_SOURCE_COLUMNS,
    feature_matrix,
    feature_names,
)
from stage6_expanded_layers import (
    DEVELOPMENT_CUTOFF_UTC,
    EXPANDED_LAYER_CONTRACT_DIGEST,
    FEATURE_SOURCE_COLUMNS,
    LOGICAL_LANE_COUNT,
    SCIENCE_IMPLEMENTATION_BINDING,
    SCIENCE_IMPLEMENTATION_DIGEST,
    build_dyad_history_features,
    build_line_prefix_features,
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


FEATURE_RUN_ID = hashlib.sha256(canonical_json_bytes({
    "schema": "stage6_expanded_layer_feature_freeze_v1",
    "notebook01_run_identity": NOTEBOOK01_RUN_ID,
    "notebook01_split_root": NOTEBOOK01_SPLIT_ROOT,
    "expanded_layer_contract": EXPANDED_LAYER_CONTRACT_DIGEST,
    "science_implementation": SCIENCE_IMPLEMENTATION_DIGEST,
    "development_cutoff_utc_exclusive": DEVELOPMENT_CUTOFF_UTC,
})).hexdigest()
COMMON_IDENTITY_COLUMNS = (
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "decision_utc",
    "decision_order",
    "street",
    "action_label",
)


def _safe_base_feature_name(name: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"base_{safe.strip('_')}"


BASE_FEATURE_COLUMNS = tuple(_safe_base_feature_name(name) for name in feature_names())
if len(BASE_FEATURE_COLUMNS) != len(set(BASE_FEATURE_COLUMNS)):
    raise RuntimeError("base feature name normalization is not unique")


def expanded_feature_root(stage_drive_root: Path) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "feature_freezes"
        / FEATURE_RUN_ID
    )


def _file_identity(path: Path, relative_to: Path, row_count: int) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(row_count),
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _publish_parquet(
    frame: pd.DataFrame,
    destination: Path,
    root: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage6-expanded-feature-") as temporary:
        local = Path(temporary) / destination.name
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(
            table,
            local,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        )
        expected = _file_identity(local, local.parent, len(frame))
        if destination.exists():
            observed = _file_identity(destination, root, pq.ParquetFile(destination).metadata.num_rows)
            if (
                observed["bytes"] == expected["bytes"]
                and observed["sha256"] == expected["sha256"]
                and observed["row_count"] == expected["row_count"]
            ):
                return observed
            destination.unlink()
        shutil.copyfile(local, destination)
    identity = _file_identity(destination, root, len(frame))
    if (
        identity["bytes"] != expected["bytes"]
        or identity["sha256"] != expected["sha256"]
        or pq.ParquetFile(destination).metadata.num_rows != len(frame)
    ):
        raise RuntimeError(f"expanded feature publication mismatch: {destination}")
    return identity


def _load_development_source(notebook01_root: Path) -> pd.DataFrame:
    columns = list(dict.fromkeys((*FEATURE_SOURCE_COLUMNS, *MODEL_SOURCE_COLUMNS)))
    frames: list[pd.DataFrame] = []
    total = 0
    for lane in range(LOGICAL_LANE_COUNT):
        decision_path, _, _ = _lane_paths(notebook01_root, lane)
        # The development runtime is permanently sealed before the fresh panel.
        # It must fail rather than open any future target from a refreshed source.
        maximum = pq.read_table(decision_path, columns=["decision_utc"]).column(0)
        maximum_utc = max(str(value.as_py()) for value in maximum) if len(maximum) else ""
        if maximum_utc >= DEVELOPMENT_CUTOFF_UTC:
            raise RuntimeError(
                "fresh adjudication rows are present; Notebook 12 is not allowed to open them"
            )
        frame = pq.read_table(decision_path, columns=columns).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
        total += len(frame)
        print(json.dumps({
            "event": "stage6_expanded_feature_source_lane_loaded",
            "lane": lane,
            "completed": lane + 1,
            "total": LOGICAL_LANE_COUNT,
            "lane_rows": len(frame),
            "loaded_rows": total,
            "fresh_targets_opened": False,
        }, sort_keys=True), flush=True)
    source = pd.concat(frames, ignore_index=True)
    if source["row_id"].astype(str).duplicated().any():
        raise RuntimeError("expanded feature source duplicates row_id across lanes")
    return source


def build_feature_frames(
    source: pd.DataFrame,
    *,
    allow_fresh_adjudication: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return common/base, line-prefix, and dyad-history feature frames."""
    line = build_line_prefix_features(source)
    dyad = build_dyad_history_features(source)
    matrix = feature_matrix(source)
    common = source.loc[:, COMMON_IDENTITY_COLUMNS].copy()
    common["source_lane_id"] = source["source_lane_id"].astype(str)
    hand_roles = source.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["decision_utc"].max().astype(str).map(temporal_role).rename("temporal_role").reset_index()
    common = common.merge(
        hand_roles,
        on=["game_id", "hand_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if common["temporal_role"].isna().any():
        raise RuntimeError("whole-hand temporal role assignment is incomplete")
    if (
        not allow_fresh_adjudication
        and common["temporal_role"].eq("fresh_adjudication").any()
    ):
        raise RuntimeError("fresh target entered the development feature freeze")
    for index, name in enumerate(BASE_FEATURE_COLUMNS):
        common[name] = matrix[:, index]
    if not common["row_id"].astype(str).equals(source["row_id"].astype(str)):
        raise RuntimeError("common feature row order changed")
    return common, line, dyad


def whole_hand_role_binding(common: pd.DataFrame) -> dict[str, Any]:
    per_hand_counts = common.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["temporal_role"].nunique()
    if not per_hand_counts.eq(1).all():
        raise RuntimeError("a game hand spans multiple expanded-layer temporal roles")
    assignments = common.loc[:, ["game_id", "hand_id", "temporal_role"]].drop_duplicates().sort_values(
        ["game_id", "hand_id"], kind="mergesort"
    ).reset_index(drop=True)
    records = [
        [str(row.game_id), str(row.hand_id), str(row.temporal_role)]
        for row in assignments.itertuples(index=False)
    ]
    return seal_document({
        "schema_version": "stage6_expanded_whole_hand_temporal_assignment_binding_v1",
        "assignment_unit": "game_id_plus_hand_id",
        "assignment_time": "maximum_decision_utc",
        "boundary_behavior": "later_partition_wins",
        "hand_count": len(assignments),
        "role_hand_counts": {
            str(key): int(value)
            for key, value in assignments["temporal_role"].value_counts(sort=False).items()
        },
        "assignment_root_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    })


def _valid_completed_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "feature-freeze-manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("schema_version") != "stage6_expanded_feature_freeze_manifest_v1"
        or manifest.get("status") != "PASS_DEVELOPMENT_FEATURE_FREEZE"
        or manifest.get("feature_run_id") != FEATURE_RUN_ID
        or manifest.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or manifest.get("completed_lane_count") != LOGICAL_LANE_COUNT
        or manifest.get("fresh_targets_opened") is not False
    ):
        raise RuntimeError("completed expanded feature manifest is invalid")
    implementation_binding = manifest.get("science_implementation_binding")
    if not isinstance(implementation_binding, Mapping):
        raise RuntimeError("completed expanded feature manifest lacks implementation binding")
    validate_document(implementation_binding)
    if implementation_binding.get("self_sha256") != SCIENCE_IMPLEMENTATION_DIGEST:
        raise RuntimeError("completed expanded feature implementation binding changed")
    whole_hand_binding = manifest.get("whole_hand_temporal_assignment_binding")
    if not isinstance(whole_hand_binding, Mapping):
        raise RuntimeError("completed expanded feature manifest lacks whole-hand binding")
    validate_document(whole_hand_binding)
    for lane in manifest.get("lanes", []):
        for key in ("common", "line", "dyad"):
            identity = lane[key]
            path = root / identity["relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != identity["bytes"]
                or sha256_file(path) != identity["sha256"]
                or pq.ParquetFile(path).metadata.num_rows != identity["row_count"]
            ):
                raise RuntimeError(f"completed expanded feature {key} identity changed")
    return manifest


def run_feature_freeze(stage_drive_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    target_root = expanded_feature_root(stage_drive_root)
    completed = _valid_completed_manifest(target_root)
    if completed is not None:
        return {
            "status": "PASS",
            "feature_run_id": FEATURE_RUN_ID,
            "manifest_path": str(target_root / "feature-freeze-manifest.json"),
            "manifest_self_sha256": completed["self_sha256"],
            "reused_completed_result": True,
            "fresh_targets_opened": False,
            "qualification_authority": False,
        }

    notebook01_root = materialization_root(stage_drive_root)
    source = _load_development_source(notebook01_root)
    print(json.dumps({
        "event": "stage6_expanded_feature_derivation_started",
        "rows": len(source),
        "available_system_ram_bytes": _available_ram(),
        "available_disk_bytes": shutil.disk_usage(
            "/content" if Path("/content").is_dir() else Path.cwd()
        ).free,
    }, sort_keys=True), flush=True)
    common, line, dyad = build_feature_frames(source)
    hand_role_binding = whole_hand_role_binding(common)

    lanes: list[dict[str, Any]] = []
    for lane in range(LOGICAL_LANE_COUNT):
        lane_id = f"lane-{lane:02d}"
        common_lane = common.loc[common["source_lane_id"].eq(lane_id)].drop(
            columns="source_lane_id"
        ).reset_index(drop=True)
        row_ids = set(common_lane["row_id"].astype(str))
        line_lane = line.loc[line["row_id"].astype(str).isin(row_ids)].reset_index(drop=True)
        dyad_lane = dyad.loc[dyad["row_id"].astype(str).isin(row_ids)].reset_index(drop=True)
        if len(common_lane) != len(line_lane) or len(common_lane) != len(dyad_lane):
            raise RuntimeError(f"{lane_id} expanded feature row conservation failed")
        outputs = {
            "lane_id": lane_id,
            "common": _publish_parquet(
                common_lane, target_root / "common" / f"{lane_id}.parquet", target_root
            ),
            "line": _publish_parquet(
                line_lane, target_root / "line" / f"{lane_id}.parquet", target_root
            ),
            "dyad": _publish_parquet(
                dyad_lane, target_root / "dyad" / f"{lane_id}.parquet", target_root
            ),
        }
        lanes.append(outputs)
        print(json.dumps({
            "event": "stage6_expanded_feature_lane_published",
            "lane": lane,
            "completed": lane + 1,
            "total": LOGICAL_LANE_COUNT,
            "rows": len(common_lane),
            "elapsed_seconds": time.monotonic() - started,
            "available_system_ram_bytes": _available_ram(),
        }, sort_keys=True), flush=True)

    role_counts = {
        str(key): int(value)
        for key, value in common["temporal_role"].value_counts(sort=False).items()
    }
    manifest = seal_document({
        "schema_version": "stage6_expanded_feature_freeze_manifest_v1",
        "status": "PASS_DEVELOPMENT_FEATURE_FREEZE",
        "feature_run_id": FEATURE_RUN_ID,
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "science_implementation_binding": SCIENCE_IMPLEMENTATION_BINDING,
        "development_cutoff_utc_exclusive": DEVELOPMENT_CUTOFF_UTC,
        "row_count": len(common),
        "role_row_counts": role_counts,
        "whole_hand_temporal_assignment_binding": hand_role_binding,
        "base_feature_columns": list(BASE_FEATURE_COLUMNS),
        "line_feature_columns": [column for column in line.columns if column != "row_id"],
        "dyad_feature_columns": [
            column for column in dyad.columns if column not in {"row_id", "dyad_eligible"}
        ],
        "completed_lane_count": len(lanes),
        "fixed_lane_order": [row["lane_id"] for row in lanes],
        "lanes": lanes,
        "action_timing_layer_included": False,
        "fresh_targets_opened": False,
        "current_corpus_authority": "development_and_retrospective_only",
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "action_changing_exploit_authority": False,
        "elapsed_seconds": time.monotonic() - started,
    })
    target_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(target_root / "feature-freeze-manifest.json", manifest)
    return {
        "status": "PASS",
        "feature_run_id": FEATURE_RUN_ID,
        "manifest_path": str(target_root / "feature-freeze-manifest.json"),
        "manifest_self_sha256": manifest["self_sha256"],
        "row_count": len(common),
        "role_row_counts": role_counts,
        "reused_completed_result": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
    }


def _available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None
