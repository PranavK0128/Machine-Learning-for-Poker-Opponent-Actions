"""Build exact target-free incumbent contexts before fresh target opening."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stage6_behavior_challenger_finalist_v2 import (
    FRESH_CUTOFF_UTC,
    _science_package_binding,
    aggregate_run_id,
    aggregate_work_root,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    SOURCE_DEPENDENCY_DIGEST,
)
from stage6_behavior_json_publication_v2 import (
    load_json_with_recovery,
    write_json_exclusive,
)
from stage6_drive_fuse_publication import create_or_verify_bytes
from stage6_expanded_feature_freeze_v2 import FEATURE_RUN_ID
from stage6_expanded_layers import LOGICAL_LANE_COUNT, canonical_json_bytes, seal_document, validate_document
from stage6_notebook09_runtime import (
    ACTION_NATIVE_SOURCE_COLUMNS,
    INCUMBENT_ARTIFACT_FILES,
    NOTEBOOK01_ACTION_MEMBER,
    NOTEBOOK01_ACTION_MEMBER_BYTES,
    NOTEBOOK01_ACTION_MEMBER_SHA256,
    NOTEBOOK01_INPUT_BYTES,
    NOTEBOOK01_INPUT_NAME,
    NOTEBOOK01_INPUT_SHA256,
    PREFLOP_NATIVE_FEATURE_NAME,
    PREFLOP_NATIVE_FEATURE_ROWS,
    PREFLOP_NATIVE_REQUIRED_COLUMNS,
    ROLLING_PREFIXES,
    _action_stat_feature_values,
    _empty_action_stats,
    _parse_active_opponents,
    _read_action_native_source,
    _subtract_action_stats,
    _sum_action_stats,
    _update_action_stats,
)
from stage6_notebooks02_07_runtime import (
    NOTEBOOK01_RUN_ID,
    _lane_paths,
    materialization_root,
    sha256_file,
)


SOURCE_MANIFEST_NAME = "stage6_behavior_v2_target_free_native_source_manifest.json"
TARGET_FREE_CONTEXT_MANIFEST_NAME = (
    "stage6_behavior_v2_fresh_incumbent_context_manifest.json"
)
PREFLOP_SOURCE_NAME = "preflop-native-source.parquet"
ACTION_SOURCE_NAME = "action-native-rolling-source.parquet"
NOTEBOOK09_INPUT_NAME = "POKERNOW_ASSISTANT_STAGE6_NOTEBOOK09_INPUTS_R3.zip"
NOTEBOOK09_INPUT_BYTES = 93_176_757
NOTEBOOK09_INPUT_SHA256 = "3ca44fdfd11da02ca3969cc1f5b21d07ec4554ce0909258637633ea5e74d1a61"
NOTEBOOK09_PACKAGE_MANIFEST_SELF_SHA256 = "a0845397af3a8d0d4fb2d5c41a2d4bd5d9cb433f8fa05028180182e56ed86be2"
PREFLOP_NATIVE_MEMBER = (
    "current_qualified_artifact/" + PREFLOP_NATIVE_FEATURE_NAME
)
PREFLOP_NATIVE_MEMBER_BYTES = 37_624_154
PREFLOP_NATIVE_MEMBER_SHA256 = "7014ab96e12679edd5065bb9fb4e77ae573605c9231ec99ed38c934eeed3392c"
PROHIBITED_TARGET_OR_OUTCOME_COLUMNS = frozenset({
    "action_label", "normalized_type", "target", "target_label", "outcome",
    "winner", "profit", "net_won", "pot_won", "is_fold", "is_check",
    "is_call", "is_bet_or_raise", "is_all_in", "action_contribution_delta_raw",
    "stack_delta_raw", "actor_committed_after_bb", "pot_after_bb",
})
PROHIBITED_TARGET_OR_OUTCOME_TOKENS = (
    "action_label", "target_label", "hand_outcome", "showdown", "winner",
    "profit", "net_won", "pot_won", "private_card", "hole_card",
)
PREFLOP_SOURCE_REQUIRED_COLUMNS = frozenset({
    "hand_id", "seat", *PREFLOP_NATIVE_REQUIRED_COLUMNS,
})
ROLLING_REQUIRED_COLUMNS = frozenset().union(*(
    set(_action_stat_feature_values(prefix, _empty_action_stats()))
    for prefix in ROLLING_PREFIXES
))
ACTION_SOURCE_REQUIRED_COLUMNS = frozenset(
    ({"decision_id"} | (set(ACTION_NATIVE_SOURCE_COLUMNS) - {"hand_action_id"}))
    | set(ROLLING_REQUIRED_COLUMNS)
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _context_json_binding(phase: str, fresh_run_identity: str = "pre-freeze") -> dict[str, Any]:
    return seal_document({
        "schema_version": "stage6_behavior_context_json_binding_v3",
        "phase": phase,
        "fresh_materialization_run_identity": fresh_run_identity,
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
    })


def _load_freeze(stage_drive_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = aggregate_work_root(stage_drive_root)
    binding = seal_document({
        "schema_version": "stage6_behavior_finalist_json_binding_v3",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
    })
    freeze = load_json_with_recovery(
        root / "aggregate-freeze-manifest.json", root=root, publication_binding=binding
    )
    custody = load_json_with_recovery(
        root / "fresh-custody-request.json", root=root, publication_binding=binding
    )
    if (
        freeze.get("aggregate_run_id") != aggregate_run_id()
        or freeze.get("science_runtime_package_binding")
        != _science_package_binding()
        or freeze.get("fresh_targets_opened") is not False
        or custody.get("aggregate_freeze_manifest_self_sha256")
        != freeze["self_sha256"]
    ):
        raise RuntimeError("behavior aggregate freeze/custody identity is invalid")
    return freeze, custody


def expected_fresh_custody_binding(
    freeze: Mapping[str, Any], custody: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "stage6_notebook01_fresh_custody_binding_v1",
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_self_sha256": custody["self_sha256"],
    }


def canonical_source_root(stage_drive_root: Path, fresh_run_identity: str) -> Path:
    return (
        stage_drive_root
        / "inputs"
        / "behavior_v2_fresh_native_context"
        / fresh_run_identity
    )


def fresh_context_root(stage_drive_root: Path, fresh_run_identity: str) -> Path:
    return aggregate_work_root(stage_drive_root) / "fresh_context" / fresh_run_identity


def validate_fresh_manifest_metadata_only(
    stage_drive_root: Path,
    root: Path,
    expected_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Validate custody/canonical metadata without hashing target-bearing Parquet."""
    path = root / "stage6_notebook01_materialization_manifest.json"
    if not path.is_file():
        raise RuntimeError("fresh Notebook01 materialization manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_document(manifest)
    run_identity = str(manifest.get("run_identity", ""))
    run_binding = manifest.get("run_binding")
    source_identities = (
        run_binding.get("source_identities") if isinstance(run_binding, Mapping)
        else None
    )
    native_action_identity = (
        source_identities.get("data/exports/ml/action-state-decision-rows.csv")
        if isinstance(source_identities, Mapping) else None
    )
    canonical = (
        stage_drive_root / "runtime_state" / "notebook01" / "materializations"
        / run_identity
    )
    if (
        manifest.get("schema_version") != "stage6_notebook01_materialization_manifest_v1"
        or manifest.get("status")
        != "PASS_FULL_CAUSAL_DECISION_PROFILE_MATERIALIZATION_AND_SEALED_SPLIT_LOCK"
        or manifest.get("completed_lane_count") != LOGICAL_LANE_COUNT
        or manifest.get("private_lockbox_opened") is not False
        or run_identity == NOTEBOOK01_RUN_ID
        or not isinstance(run_binding, Mapping)
        or run_binding.get("schema_version")
        != "stage6_notebook01_materialization_run_binding_v3"
        or hashlib.sha256(canonical_json_bytes(run_binding)).hexdigest() != run_identity
        or run_binding.get("fresh_custody_binding") != dict(expected_binding)
        or not isinstance(native_action_identity, Mapping)
        or native_action_identity.get("bytes") != NOTEBOOK01_ACTION_MEMBER_BYTES
        or native_action_identity.get("sha256") != NOTEBOOK01_ACTION_MEMBER_SHA256
        or root.resolve() != canonical.resolve()
    ):
        raise RuntimeError("fresh Notebook01 metadata-only authority is invalid")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or [row.get("lane_id") for row in lanes] != [
        f"lane-{index:02d}" for index in range(LOGICAL_LANE_COUNT)
    ]:
        raise RuntimeError("fresh Notebook01 lane metadata changed")
    paths = []
    for lane in lanes:
        identity = lane.get("outputs", {}).get("accepted_decisions")
        relative = f"lanes/{lane['lane_id']}/accepted-decisions.parquet"
        path = root / relative
        if (
            not isinstance(identity, Mapping)
            or identity.get("relative_path") != relative
            or not path.is_file()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(path) != identity.get("sha256")
            or pq.ParquetFile(path).metadata.num_rows
            != int(identity.get("row_count", -1))
        ):
            raise RuntimeError("fresh accepted-decision metadata path is invalid")
        paths.append(path)
    return manifest, paths


def _metadata_panel(paths: Sequence[Path]) -> pd.DataFrame:
    columns = [
        "row_id", "game_id", "hand_id", "decision_id", "decision_utc",
        "decision_order", "street", "seat", "actor_id",
        "active_opponent_canonical_player_ids_before",
    ]
    frames = []
    for lane, path in enumerate(paths):
        frame = pq.read_table(path, columns=columns).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    for name in (
        "row_id", "game_id", "hand_id", "decision_id", "street", "actor_id"
    ):
        frame[name] = frame[name].astype(str)
    if frame["row_id"].duplicated().any() or frame["decision_id"].duplicated().any():
        raise RuntimeError("fresh metadata identities are duplicated")
    frame["decision_ns"] = pd.to_datetime(
        frame["decision_utc"], utc=True, errors="raise"
    ).astype("int64")
    cutoff = pd.Timestamp(FRESH_CUTOFF_UTC).value
    hands = frame.groupby(["game_id", "hand_id"], sort=True)["decision_ns"].agg(
        ["min", "max"]
    )
    if ((hands["min"] < cutoff) & (hands["max"] >= cutoff)).any():
        raise RuntimeError("a whole hand crosses the behavior fresh cutoff")
    eligible = hands.index[hands["min"] >= cutoff]
    keys = pd.MultiIndex.from_frame(frame[["game_id", "hand_id"]])
    fresh = frame.loc[keys.isin(eligible)].copy()
    if fresh.empty:
        raise RuntimeError("new materialization has no target-free fresh metadata rows")
    return fresh


def _schema_receipt(path: Path) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in pq.ParquetFile(path).schema_arrow
    ]


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": pq.ParquetFile(path).metadata.num_rows,
        "schema": _schema_receipt(path),
    }


def target_free_native_source_producer_binding(
    fresh_manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    upstream_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact global-order causal derivation and all authenticated inputs."""
    validate_document(upstream_binding)
    return seal_document({
        "schema_version": "stage6_behavior_v2_target_free_native_source_producer_binding_v3",
        "generator": "global_completed_hand_pre_row_causal_context_v3",
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "authenticated_upstream_binding": dict(upstream_binding),
        "global_order": [
            "hand_started_at", "game_id", "hand_id", "decision_order",
            "source_lane_id", "source_original_index",
        ],
        "history_rows": "bound_development_rows_strictly_before_fresh_cutoff",
        "pre_row_capture": True,
        "fresh_rows_update_history": False,
        "development_target_columns_read": ["action_label"],
        "fresh_target_columns_read": [],
        "fresh_outcome_columns_read": [],
    })


def _exact_input(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha256
    ):
        raise RuntimeError(f"authenticated causal producer input changed: {path.name}")
    return {"name": path.name, "bytes": expected_bytes, "sha256": expected_sha256}


def _development_history(stage_drive_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = materialization_root(stage_drive_root)
    manifest_path = root / "stage6_notebook01_materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if manifest.get("run_identity") != NOTEBOOK01_RUN_ID:
        raise RuntimeError("development Notebook01 materialization identity changed")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != LOGICAL_LANE_COUNT:
        raise RuntimeError("development accepted-decision lane inventory changed")
    columns = [
        "decision_id", "decision_utc", "decision_order", "game_id", "hand_id",
        "actor_id", "seat", "street",
        "active_opponent_canonical_player_ids_before", "action_label",
        "amount_to_call_bb", "pot_before_bb", "actor_committed_before_bb",
        "actor_committed_after_bb", "actor_stack_before_bb", "spr_before",
    ]
    frames: list[pd.DataFrame] = []
    identities = []
    for lane in range(LOGICAL_LANE_COUNT):
        path, _, _ = _lane_paths(root, lane)
        identity = lanes[lane]["outputs"]["accepted_decisions"]
        if (
            path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != identity["sha256"]
        ):
            raise RuntimeError("development accepted-decision identity changed")
        frame = pq.read_table(path, columns=columns).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frame["source_original_index"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
        identities.append(dict(identity))
    history = pd.concat(frames, ignore_index=True)
    decision_ns = pd.to_datetime(history["decision_utc"], utc=True).astype("int64")
    history["_decision_ns"] = decision_ns
    cutoff = pd.Timestamp(FRESH_CUTOFF_UTC).value
    hand_max = history.groupby(["game_id", "hand_id"], sort=True)["_decision_ns"].transform("max")
    history = history.loc[hand_max < cutoff].copy()
    if history.empty:
        raise RuntimeError("development causal history is empty")
    binding = seal_document({
        "schema_version": "stage6_behavior_v2_development_history_binding_v1",
        "notebook01_run_id": NOTEBOOK01_RUN_ID,
        "materialization_manifest_self_sha256": manifest["self_sha256"],
        "accepted_lane_identities": identities,
        "accepted_lane_identity_root": _digest(identities),
        "row_count_strictly_before_cutoff": len(history),
        "fresh_cutoff_utc_exclusive": FRESH_CUTOFF_UTC,
    })
    return history, binding


def _causal_fresh_rolling(
    history: pd.DataFrame,
    fresh: pd.DataFrame,
    hand_starts: pd.DataFrame,
    fresh_postflop_ids: set[str],
) -> pd.DataFrame:
    history = history.copy()
    history["is_fresh_source_row"] = False
    fresh_rows = fresh.copy()
    fresh_rows["is_fresh_source_row"] = True
    fresh_rows["source_original_index"] = np.arange(len(fresh_rows), dtype=np.int64)
    for column in history.columns:
        if column not in fresh_rows.columns:
            fresh_rows[column] = (
                0.0 if pd.api.types.is_numeric_dtype(history[column]) else ""
            )
    ordered = pd.concat(
        [history, fresh_rows[history.columns]], ignore_index=True
    ).merge(hand_starts, on="hand_id", how="left", validate="many_to_one")
    if ordered["hand_started_at"].isna().any():
        raise RuntimeError("causal producer lacks native hand start time")
    ordered["_hand_started_at"] = pd.to_numeric(
        ordered["hand_started_at"], errors="raise"
    )
    ordered = ordered.sort_values(
        ["_hand_started_at", "game_id", "hand_id", "decision_order",
         "source_lane_id", "source_original_index"],
        kind="mergesort",
    )
    player_stats: dict[str, dict[str, float]] = {}
    game_stats: dict[str, dict[str, float]] = {}
    game_player_stats: dict[tuple[str, str], dict[str, float]] = {}
    output = []
    for row_tuple in ordered.itertuples(index=False):
        row = row_tuple._asdict()
        actor = str(row["actor_id"])
        game = str(row["game_id"])
        actor_stats = player_stats.setdefault(actor, _empty_action_stats())
        table_stats = game_stats.setdefault(game, _empty_action_stats())
        actor_table = game_player_stats.setdefault((game, actor), _empty_action_stats())
        active_ids = [
            value for value in _parse_active_opponents(
                row["active_opponent_canonical_player_ids_before"]
            ) if value != actor
        ]
        active_stats = (
            _sum_action_stats([
                player_stats.get(value, _empty_action_stats()) for value in active_ids
            ]) if active_ids else _subtract_action_stats(table_stats, actor_table)
        )
        decision_id = str(row["decision_id"])
        if bool(row["is_fresh_source_row"]):
            if decision_id in fresh_postflop_ids:
                values: dict[str, Any] = {"decision_id": decision_id}
                values.update(_action_stat_feature_values("actor_roll", actor_stats))
                values.update(_action_stat_feature_values(
                    "active_opponent_roll", active_stats
                ))
                output.append(values)
            continue
        _update_action_stats(actor_stats, row)
        _update_action_stats(table_stats, row)
        _update_action_stats(actor_table, row)
    result = pd.DataFrame(output)
    if (
        len(result) != len(fresh_postflop_ids)
        or result["decision_id"].astype(str).duplicated().any()
        or set(result["decision_id"].astype(str)) != fresh_postflop_ids
    ):
        raise RuntimeError("pre-row rolling features do not cover exact fresh postflop rows")
    return result


def _preflop_native_from_bound_package(
    package_path: Path,
    fresh_first: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with zipfile.ZipFile(package_path) as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        validate_document(manifest)
        member = next(
            row for row in manifest["members"] if row["path"] == PREFLOP_NATIVE_MEMBER
        )
        raw = archive.read(PREFLOP_NATIVE_MEMBER)
    if (
        manifest["self_sha256"] != NOTEBOOK09_PACKAGE_MANIFEST_SELF_SHA256
        or member["bytes"] != PREFLOP_NATIVE_MEMBER_BYTES
        or member["sha256"] != PREFLOP_NATIVE_MEMBER_SHA256
        or len(raw) != PREFLOP_NATIVE_MEMBER_BYTES
        or hashlib.sha256(raw).hexdigest() != PREFLOP_NATIVE_MEMBER_SHA256
    ):
        raise RuntimeError("bound Notebook09 preflop native member changed")
    native = pq.read_table(pa.BufferReader(raw)).to_pandas()
    if len(native) != PREFLOP_NATIVE_FEATURE_ROWS:
        raise RuntimeError("bound preflop native row count changed")
    native["hand_id"] = native["hand_id"].astype(str)
    native["seat"] = pd.to_numeric(native["seat"], errors="raise").astype(int)
    keys = set(zip(fresh_first["hand_id"].astype(str), fresh_first["seat"].astype(int)))
    selected = native.loc[
        [(str(hand), int(seat)) in keys for hand, seat in zip(native["hand_id"], native["seat"])]
    ].copy()
    merged = fresh_first[["decision_id", "hand_id", "seat"]].merge(
        selected, on=["hand_id", "seat"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(fresh_first):
        raise RuntimeError("bound preflop native source does not exactly cover fresh rows")
    return merged, dict(member)


def produce_target_free_native_source_bundle(
    stage_drive_root: Path,
    source_root: Path,
    fresh_manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    fresh_paths: Sequence[Path],
) -> dict[str, Any]:
    """Derive exact target-free sources from bound upstream decisions and native data."""
    source_root.mkdir(parents=True, exist_ok=True)
    fresh = _metadata_panel(fresh_paths)
    postflop_ids = set(fresh.loc[~fresh["street"].eq("preflop"), "decision_id"])
    notebook01_input = stage_drive_root / "inputs" / NOTEBOOK01_INPUT_NAME
    notebook09_input = stage_drive_root / "inputs" / NOTEBOOK09_INPUT_NAME
    notebook01_identity = _exact_input(
        notebook01_input, NOTEBOOK01_INPUT_BYTES, NOTEBOOK01_INPUT_SHA256
    )
    notebook09_identity = _exact_input(
        notebook09_input, NOTEBOOK09_INPUT_BYTES, NOTEBOOK09_INPUT_SHA256
    )
    with zipfile.ZipFile(notebook01_input) as archive:
        action_info = archive.getinfo(NOTEBOOK01_ACTION_MEMBER)
    if action_info.file_size != NOTEBOOK01_ACTION_MEMBER_BYTES:
        raise RuntimeError("bound Notebook01 native action member size changed")
    native_action, hand_starts = _read_action_native_source(
        notebook01_input, postflop_ids
    )
    history, history_binding = _development_history(stage_drive_root)
    rolling = _causal_fresh_rolling(history, fresh, hand_starts, postflop_ids)
    action = native_action.rename(columns={"hand_action_id": "decision_id"}).merge(
        rolling, on="decision_id", how="inner", validate="one_to_one"
    ).sort_values("decision_id", kind="mergesort").reset_index(drop=True)
    first = fresh.loc[fresh["street"].eq("preflop")].sort_values(
        ["game_id", "hand_id", "seat", "decision_order", "row_id"],
        kind="mergesort",
    ).drop_duplicates(["game_id", "hand_id", "seat"], keep="first")
    preflop, preflop_member = _preflop_native_from_bound_package(
        notebook09_input, first
    )
    preflop = preflop.sort_values("decision_id", kind="mergesort").reset_index(drop=True)
    _write_parquet_exclusive(source_root / PREFLOP_SOURCE_NAME, preflop)
    _write_parquet_exclusive(source_root / ACTION_SOURCE_NAME, action)
    upstream = seal_document({
        "schema_version": "stage6_behavior_v2_causal_context_upstream_binding_v1",
        "fresh_materialization_run_binding": fresh_manifest["run_binding"],
        "fresh_accepted_lane_identities": [
            dict(row["outputs"]["accepted_decisions"])
            for row in fresh_manifest["lanes"]
        ],
        "development_history_binding": history_binding,
        "notebook01_input": notebook01_identity,
        "notebook01_action_member": {
            "path": NOTEBOOK01_ACTION_MEMBER,
            "bytes": NOTEBOOK01_ACTION_MEMBER_BYTES,
            "sha256": NOTEBOOK01_ACTION_MEMBER_SHA256,
        },
        "notebook09_input": notebook09_identity,
        "notebook09_package_manifest_self_sha256": (
            NOTEBOOK09_PACKAGE_MANIFEST_SELF_SHA256
        ),
        "preflop_native_member": preflop_member,
    })
    producer = target_free_native_source_producer_binding(
        fresh_manifest, freeze, upstream
    )
    files = [
        _file_identity(source_root / PREFLOP_SOURCE_NAME),
        _file_identity(source_root / ACTION_SOURCE_NAME),
    ]
    manifest = seal_document({
        "schema_version": "stage6_behavior_v2_target_free_native_source_manifest_v2",
        "status": "PASS_AUTHENTICATED_CAUSAL_TARGET_FREE_NATIVE_SOURCE_FROZEN",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "target_or_outcome_columns_included": False,
        "producer_binding": producer,
        "producer_binding_self_sha256": producer["self_sha256"],
        "authenticated_upstream_binding": upstream,
        "fresh_target_columns_read": [],
        "development_target_columns_read": ["action_label"],
        "pre_row_capture": True,
        "fresh_rows_update_history": False,
        "incumbent_artifact_files": INCUMBENT_ARTIFACT_FILES,
        "files": files,
        "qualification_authority": False,
    })
    _write_json_exclusive(
        source_root / SOURCE_MANIFEST_NAME,
        manifest,
        phase="source",
        fresh_run_identity=str(fresh_manifest["run_identity"]),
    )
    return manifest


def produce_target_free_native_source_for_fresh_materialization(
    stage_drive_root: Path,
    fresh_materialization_root: Path,
) -> dict[str, Any]:
    """Notebook 26 entrypoint: derive, authenticate, and seal the source bundle."""
    freeze, custody = _load_freeze(stage_drive_root)
    fresh_manifest, paths = validate_fresh_manifest_metadata_only(
        stage_drive_root,
        fresh_materialization_root,
        expected_fresh_custody_binding(freeze, custody),
    )
    root = canonical_source_root(
        stage_drive_root, str(fresh_manifest["run_identity"])
    )
    return produce_target_free_native_source_bundle(
        stage_drive_root, root, fresh_manifest, freeze, paths
    )


def _validate_source_file(
    root: Path,
    identity: Mapping[str, Any],
    name: str,
    required_columns: frozenset[str],
) -> Path:
    path = root / name
    schema = pq.ParquetFile(path).schema_arrow if path.is_file() else None
    columns = set(schema.names) if schema is not None else set()
    if (
        identity.get("name") != name
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != identity.get("sha256")
        or pq.ParquetFile(path).metadata.num_rows != int(identity.get("row_count", -1))
        or identity.get("schema") != _schema_receipt(path)
        or not required_columns.issubset(columns)
        or columns & PROHIBITED_TARGET_OR_OUTCOME_COLUMNS
        or any(
            token in column.lower()
            for column in columns
            for token in PROHIBITED_TARGET_OR_OUTCOME_TOKENS
        )
    ):
        raise RuntimeError(f"target-free native source is invalid: {name}")
    return path


def _load_source_bundle(
    stage_drive_root: Path,
    source_root: Path,
    fresh_manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    path = source_root / SOURCE_MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError(
            "authenticated target-free native source producer has not run"
        )
    manifest = load_json_with_recovery(
        path,
        root=source_root,
        publication_binding=_context_json_binding(
            "source", str(fresh_manifest["run_identity"])
        ),
    )
    files = manifest.get("files")
    upstream = manifest.get("authenticated_upstream_binding")
    producer = manifest.get("producer_binding")
    if (
        manifest.get("schema_version")
        != "stage6_behavior_v2_target_free_native_source_manifest_v2"
        or manifest.get("status")
        != "PASS_AUTHENTICATED_CAUSAL_TARGET_FREE_NATIVE_SOURCE_FROZEN"
        or manifest.get("aggregate_run_id") != aggregate_run_id()
        or manifest.get("science_runtime_package_binding")
        != _science_package_binding()
        or manifest.get("aggregate_freeze_manifest_self_sha256")
        != freeze["self_sha256"]
        or manifest.get("fresh_materialization_run_identity")
        != fresh_manifest["run_identity"]
        or manifest.get("fresh_materialization_manifest_self_sha256")
        != fresh_manifest["self_sha256"]
        or manifest.get("target_or_outcome_columns_included") is not False
        or not isinstance(upstream, Mapping)
        or upstream.get("fresh_materialization_run_binding")
        != fresh_manifest["run_binding"]
        or upstream.get("fresh_accepted_lane_identities")
        != [dict(row["outputs"]["accepted_decisions"])
            for row in fresh_manifest["lanes"]]
        or upstream.get("notebook01_input")
        != {"name": NOTEBOOK01_INPUT_NAME, "bytes": NOTEBOOK01_INPUT_BYTES,
            "sha256": NOTEBOOK01_INPUT_SHA256}
        or upstream.get("notebook09_input")
        != {"name": NOTEBOOK09_INPUT_NAME, "bytes": NOTEBOOK09_INPUT_BYTES,
            "sha256": NOTEBOOK09_INPUT_SHA256}
        or upstream.get("notebook09_package_manifest_self_sha256")
        != NOTEBOOK09_PACKAGE_MANIFEST_SELF_SHA256
        or upstream.get("preflop_native_member", {}).get("sha256")
        != PREFLOP_NATIVE_MEMBER_SHA256
        or manifest.get("fresh_target_columns_read") != []
        or manifest.get("development_target_columns_read") != ["action_label"]
        or manifest.get("pre_row_capture") is not True
        or manifest.get("fresh_rows_update_history") is not False
        or not isinstance(producer, Mapping)
        or manifest.get("producer_binding_self_sha256")
        != producer.get("self_sha256")
        or manifest.get("incumbent_artifact_files") != INCUMBENT_ARTIFACT_FILES
        or not isinstance(files, list)
        or {row.get("name") for row in files}
        != {PREFLOP_SOURCE_NAME, ACTION_SOURCE_NAME}
    ):
        raise RuntimeError("target-free native source manifest authority is invalid")
    validate_document(upstream)
    validate_document(producer)
    _exact_input(
        stage_drive_root / "inputs" / NOTEBOOK01_INPUT_NAME,
        NOTEBOOK01_INPUT_BYTES, NOTEBOOK01_INPUT_SHA256,
    )
    _exact_input(
        stage_drive_root / "inputs" / NOTEBOOK09_INPUT_NAME,
        NOTEBOOK09_INPUT_BYTES, NOTEBOOK09_INPUT_SHA256,
    )
    development_manifest = json.loads(
        (materialization_root(stage_drive_root)
         / "stage6_notebook01_materialization_manifest.json").read_text(
             encoding="utf-8"
         )
    )
    validate_document(development_manifest)
    if upstream.get("development_history_binding", {}).get(
        "materialization_manifest_self_sha256"
    ) != development_manifest["self_sha256"]:
        raise RuntimeError("development history parent identity changed")
    expected_producer = target_free_native_source_producer_binding(
        fresh_manifest, freeze, upstream
    )
    if producer != expected_producer:
        raise RuntimeError("target-free native source producer binding changed")
    by_name = {row["name"]: row for row in files}
    preflop = _validate_source_file(
        source_root, by_name[PREFLOP_SOURCE_NAME], PREFLOP_SOURCE_NAME,
        PREFLOP_SOURCE_REQUIRED_COLUMNS,
    )
    action = _validate_source_file(
        source_root, by_name[ACTION_SOURCE_NAME], ACTION_SOURCE_NAME,
        ACTION_SOURCE_REQUIRED_COLUMNS,
    )
    return manifest, preflop, action


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError("stale fresh context temporary file exists")
    pq.write_table(
        table, temporary, compression="zstd", compression_level=9,
        use_dictionary=True, write_statistics=True, row_group_size=100_000,
    )
    try:
        raw = temporary.read_bytes()
        create_or_verify_bytes(
            path,
            raw,
            owner_run_id=hashlib.sha256(raw).hexdigest(),
        )
    except (FileExistsError, RuntimeError) as error:
        raise RuntimeError(
            "conflicting fresh context Parquet already exists"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(
    path: Path,
    document: Mapping[str, Any],
    *,
    phase: str,
    fresh_run_identity: str,
) -> None:
    write_json_exclusive(
        path,
        document,
        root=path.parent,
        publication_binding=_context_json_binding(phase, fresh_run_identity),
    )


def build_fresh_incumbent_context(
    stage_drive_root: Path,
    fresh_materialization_root: Path,
) -> dict[str, Any]:
    """Build two exact contexts without reading target or outcome columns."""
    freeze, custody = _load_freeze(stage_drive_root)
    fresh_manifest, paths = validate_fresh_manifest_metadata_only(
        stage_drive_root,
        fresh_materialization_root,
        expected_fresh_custody_binding(freeze, custody),
    )
    fresh = _metadata_panel(paths)
    source_root = canonical_source_root(
        stage_drive_root, str(fresh_manifest["run_identity"])
    )
    source_manifest, preflop_source, action_source = _load_source_bundle(
        stage_drive_root, source_root, fresh_manifest, freeze
    )
    preflop_metadata = fresh.loc[
        fresh["street"].eq("preflop")
    ].sort_values(
        ["game_id", "hand_id", "seat", "decision_order", "row_id"],
        kind="mergesort",
    ).drop_duplicates(["game_id", "hand_id", "seat"], keep="first")
    preflop_native = pq.read_table(preflop_source).to_pandas()
    preflop_native["hand_id"] = preflop_native["hand_id"].astype(str)
    preflop_native["seat"] = pd.to_numeric(preflop_native["seat"], errors="raise").astype(int)
    if preflop_native.duplicated(["hand_id", "seat"]).any():
        raise RuntimeError("preflop native source duplicates hand/seat")
    preflop = preflop_metadata[["decision_id", "hand_id", "seat"]].merge(
        preflop_native, on=["hand_id", "seat"], how="inner", validate="one_to_one"
    )
    if preflop.empty:
        raise RuntimeError("fresh preflop native domain is empty")
    action_ids = set(
        fresh.loc[~fresh["street"].eq("preflop"), "decision_id"].astype(str)
    )
    action = pq.read_table(
        action_source, filters=[("decision_id", "in", sorted(action_ids))]
    ).to_pandas()
    action["decision_id"] = action["decision_id"].astype(str)
    if action["decision_id"].duplicated().any() or set(action["decision_id"]) != action_ids:
        raise RuntimeError("action native source does not exactly cover fresh postflop rows")
    preflop = preflop.sort_values("decision_id", kind="mergesort").reset_index(drop=True)
    action = action.sort_values("decision_id", kind="mergesort").reset_index(drop=True)
    output_root = fresh_context_root(stage_drive_root, fresh_manifest["run_identity"])
    preflop_path = output_root / "preflop-native.parquet"
    action_path = output_root / "action-native-rolling.parquet"
    _write_parquet_exclusive(preflop_path, preflop)
    _write_parquet_exclusive(action_path, action)
    files = [_file_identity(preflop_path), _file_identity(action_path)]
    manifest = seal_document({
        "schema_version": "stage6_behavior_v2_fresh_incumbent_context_manifest_v2",
        "status": "PASS_TARGET_FREE_EXACT_INCUMBENT_CONTEXT_FROZEN",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "target_free_native_source_manifest_self_sha256": source_manifest["self_sha256"],
        "target_free_native_source_producer_binding_self_sha256": source_manifest[
            "producer_binding_self_sha256"
        ],
        "target_free_native_source_root": source_root.relative_to(stage_drive_root).as_posix(),
        "target_columns_included": False,
        "outcome_columns_included": False,
        "built_before_any_fresh_target_read": True,
        "metadata_columns_read_from_materialization": [
            "row_id", "game_id", "hand_id", "decision_id", "decision_utc",
            "decision_order", "street", "seat",
            "actor_id", "active_opponent_canonical_player_ids_before",
        ],
        "fresh_target_columns_read": [],
        "preflop_row_identity_root": _digest(
            preflop["decision_id"].astype(str).tolist()
        ),
        "postflop_row_identity_root": _digest(
            action["decision_id"].astype(str).tolist()
        ),
        "files": files,
        "incumbent_artifact_files": INCUMBENT_ARTIFACT_FILES,
        "selection_or_tuning_authority": False,
        "qualification_authority": False,
    })
    _write_json_exclusive(
        output_root / TARGET_FREE_CONTEXT_MANIFEST_NAME,
        manifest,
        phase="context",
        fresh_run_identity=str(fresh_manifest["run_identity"]),
    )
    return {
        "status": "PASS",
        "context_root": str(output_root),
        "manifest_path": str(output_root / TARGET_FREE_CONTEXT_MANIFEST_NAME),
        "manifest_self_sha256": manifest["self_sha256"],
        "fresh_targets_opened": False,
        "qualification_authority": False,
    }


__all__ = [
    "ACTION_SOURCE_NAME",
    "PREFLOP_SOURCE_NAME",
    "PROHIBITED_TARGET_OR_OUTCOME_COLUMNS",
    "SOURCE_MANIFEST_NAME",
    "build_fresh_incumbent_context",
    "canonical_source_root",
    "fresh_context_root",
    "produce_target_free_native_source_bundle",
    "produce_target_free_native_source_for_fresh_materialization",
    "target_free_native_source_producer_binding",
    "validate_fresh_manifest_metadata_only",
]
