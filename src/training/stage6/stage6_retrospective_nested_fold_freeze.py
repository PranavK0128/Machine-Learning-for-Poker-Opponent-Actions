"""Freeze leakage-safe retrospective nested whole-game folds for Stage 6.

This module reuses the current processed corpus without describing it as fresh.
Every outer test game is absent from fitting, selection, early stopping,
calibration, and support-rule selection for its fold.  Candidate registries and
the fold plan are sealed before any outer-fold model execution.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


CONTRACT_NAME = "stage6_retrospective_nested_qualification_contract_v1.json"
ELIGIBLE_SOURCE_ROLES = (
    "development_train",
    "development_validation",
    "retrospective_calibration",
)
PROHIBITED_SOURCE_ROLE = "fresh_adjudication"
LOGICAL_DATA_LANES = 48
MAXIMUM_OUTER_FOLDS = 5
MINIMUM_OUTER_TEST_GAMES = 20
MINIMUM_INITIAL_HISTORY_GAMES = 60
MINIMUM_TRAIN_GAMES = 40
INNER_VALIDATION_GAMES = 10
INNER_CALIBRATION_GAMES = 10

EXPANDED_LAYERS = (
    "hero_opponent_interaction",
    "within_hand_line_surprise",
)
BEHAVIOR_SCOPES = ("preflop_behavior", "postflop_behavior")
ADDITIONAL_LAYERS = (
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "opponent_bluff_likelihood",
    "player_strategy_drift",
    "showdown_aware_range_calibration",
    "supported_fold_response_bluff_opportunity",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("self_sha256", None)
    document["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def validate_sealed_document(value: Mapping[str, Any]) -> None:
    document = dict(value)
    expected = str(document.pop("self_sha256", ""))
    observed = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    if not expected or observed != expected:
        raise RuntimeError("sealed document self SHA-256 mismatch")


def _identity_root(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_contract() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "contracts" / CONTRACT_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_sealed_document(document)
    expected = {
        "authority_class": "retrospective_nested_out_of_fold",
        "fresh_or_prospective_claim": False,
        "outer_cluster_unit": "game_id",
        "maximum_outer_fold_count": MAXIMUM_OUTER_FOLDS,
        "minimum_outer_test_game_clusters_per_fold": MINIMUM_OUTER_TEST_GAMES,
        "minimum_initial_history_game_clusters": MINIMUM_INITIAL_HISTORY_GAMES,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "action_timing_layer_included": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"retrospective nested contract changed: {key}")
    return document


def _normalized_identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "game_id", "hand_id", "decision_utc", "temporal_role"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"nested fold source is missing columns: {missing}")
    source = frame.loc[:, sorted(required)].copy()
    for column in required:
        source[column] = source[column].astype(str)
        if source[column].eq("").any():
            raise RuntimeError(f"nested fold source has empty {column}")
    if source["row_id"].duplicated().any():
        raise RuntimeError("nested fold source duplicates row_id")
    roles = set(source["temporal_role"])
    if PROHIBITED_SOURCE_ROLE in roles:
        raise RuntimeError("fresh adjudication rows cannot enter retrospective folds")
    unexpected = sorted(roles.difference(ELIGIBLE_SOURCE_ROLES))
    if unexpected:
        raise RuntimeError(f"nested fold source has unexpected roles: {unexpected}")
    if not roles:
        raise RuntimeError("nested fold source is empty")
    return source


def _game_table(source: pd.DataFrame) -> pd.DataFrame:
    games = (
        source.groupby("game_id", sort=False, observed=True)
        .agg(
            minimum_decision_utc=("decision_utc", "min"),
            maximum_decision_utc=("decision_utc", "max"),
            row_count=("row_id", "size"),
            hand_count=("hand_id", "nunique"),
        )
        .reset_index()
        .sort_values(["maximum_decision_utc", "game_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if games["game_id"].duplicated().any():
        raise RuntimeError("nested fold game aggregation duplicates game_id")
    return games


def _partition_blocks(values: Sequence[str], count: int) -> list[list[str]]:
    if count <= 0 or len(values) < count:
        raise ValueError("invalid nested test block request")
    quotient, remainder = divmod(len(values), count)
    blocks: list[list[str]] = []
    cursor = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        blocks.append([str(value) for value in values[cursor : cursor + size]])
        cursor += size
    if cursor != len(values) or any(not block for block in blocks):
        raise RuntimeError("nested test block partition lost games")
    return blocks


def _partition_summary(games: pd.DataFrame, identities: Sequence[str]) -> dict[str, Any]:
    selected = games.loc[games["game_id"].isin(set(identities))]
    if len(selected) != len(identities):
        raise RuntimeError("nested fold partition lost game identities")
    return {
        "game_count": len(identities),
        "row_count": int(selected["row_count"].sum()),
        "hand_count": int(selected["hand_count"].sum()),
        "minimum_decision_utc": str(selected["minimum_decision_utc"].min()),
        "maximum_decision_utc": str(selected["maximum_decision_utc"].max()),
        "game_identity_root_sha256": _identity_root(identities),
    }


def candidate_registry_bindings() -> dict[str, Any]:
    from stage6_expanded_layers import candidate_registry as expanded_registry
    from stage6_behavior_challenger_tournament_v2 import (
        candidate_registry as behavior_registry,
        registry_digest as behavior_registry_digest,
    )
    from stage6_additional_exploitation_layers import (
        candidate_registry as additional_registry,
        registry_digest as additional_registry_digest,
    )

    return seal_document({
        "schema_version": "stage6_retrospective_nested_candidate_registry_bindings_v1",
        "expanded": {
            layer: {
                "candidate_count": len(expanded_registry(layer)),
                "registry_sha256": hashlib.sha256(
                    canonical_json_bytes(expanded_registry(layer))
                ).hexdigest(),
            }
            for layer in EXPANDED_LAYERS
        },
        "behavior": {
            scope: {
                "candidate_count": len(behavior_registry(scope)),
                "registry_sha256": behavior_registry_digest(scope),
            }
            for scope in BEHAVIOR_SCOPES
        },
        "additional": {
            layer: {
                "candidate_count": len(additional_registry(layer)),
                "registry_sha256": additional_registry_digest(layer),
            }
            for layer in ADDITIONAL_LAYERS
        },
        "all_registry_count": len(EXPANDED_LAYERS) + len(BEHAVIOR_SCOPES)
        + len(ADDITIONAL_LAYERS),
        "candidate_internal_threads": 1,
        "action_timing_layer_included": False,
    })


def build_nested_fold_plan(
    frame: pd.DataFrame,
    *,
    feature_manifest_self_sha256: str,
    registry_bindings: Mapping[str, Any],
    contract_self_sha256: str,
) -> dict[str, Any]:
    """Return a deterministic, sealed expanding-window whole-game fold plan."""
    source = _normalized_identity_frame(frame)
    games = _game_table(source)
    game_ids = games["game_id"].astype(str).tolist()
    available_test_games = max(0, len(game_ids) - MINIMUM_INITIAL_HISTORY_GAMES)
    fold_count = min(
        MAXIMUM_OUTER_FOLDS,
        available_test_games // MINIMUM_OUTER_TEST_GAMES,
    )
    inventory = {
        "decision_row_count": len(source),
        "whole_hand_count": int(
            (source["game_id"] + "\x1f" + source["hand_id"]).nunique()
        ),
        "independent_game_cluster_count": len(game_ids),
        "minimum_decision_utc": str(source["decision_utc"].min()),
        "maximum_decision_utc": str(source["decision_utc"].max()),
        "original_role_row_counts": {
            role: int(source["temporal_role"].eq(role).sum())
            for role in ELIGIBLE_SOURCE_ROLES
        },
    }
    common = {
        "schema_version": "stage6_retrospective_nested_fold_freeze_manifest_v1",
        "authority_class": "retrospective_nested_out_of_fold",
        "fresh_or_prospective_claim": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "contract_self_sha256": str(contract_self_sha256),
        "feature_manifest_self_sha256": str(feature_manifest_self_sha256),
        "candidate_registry_bindings_self_sha256": str(
            registry_bindings["self_sha256"]
        ),
        "inventory": inventory,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "action_timing_layer_included": False,
    }
    if fold_count < 1:
        manifest = seal_document({
            **common,
            "status": "HELD_INSUFFICIENT_RETROSPECTIVE_NESTED_GAME_HISTORY",
            "blocking_gate": {
                "minimum_required_game_clusters": (
                    MINIMUM_INITIAL_HISTORY_GAMES + MINIMUM_OUTER_TEST_GAMES
                ),
                "observed_game_clusters": len(game_ids),
                "gate_met": False,
            },
            "outer_fold_count": 0,
            "outer_test_game_count": 0,
        })
        return {
            "manifest": manifest,
            "assignments": seal_document({
                "schema_version": "stage6_retrospective_nested_fold_assignments_v1",
                "status": manifest["status"],
                "folds": [],
            }),
        }

    test_blocks = _partition_blocks(
        game_ids[MINIMUM_INITIAL_HISTORY_GAMES:], fold_count
    )
    folds: list[dict[str, Any]] = []
    seen_test_games: set[str] = set()
    test_cursor = MINIMUM_INITIAL_HISTORY_GAMES
    for fold_index, test_games in enumerate(test_blocks):
        history = game_ids[:test_cursor]
        calibration_games = history[-INNER_CALIBRATION_GAMES:]
        validation_end = len(history) - INNER_CALIBRATION_GAMES
        validation_start = validation_end - INNER_VALIDATION_GAMES
        validation_games = history[validation_start:validation_end]
        train_games = history[:validation_start]
        if (
            len(train_games) < MINIMUM_TRAIN_GAMES
            or len(validation_games) != INNER_VALIDATION_GAMES
            or len(calibration_games) != INNER_CALIBRATION_GAMES
            or len(test_games) < MINIMUM_OUTER_TEST_GAMES
        ):
            raise RuntimeError("nested fold violates minimum whole-game partitions")
        partitions = {
            "train": train_games,
            "validation": validation_games,
            "calibration": calibration_games,
            "test": test_games,
        }
        sets = {name: set(values) for name, values in partitions.items()}
        names = tuple(partitions)
        if any(sets[names[i]] & sets[names[j]] for i in range(len(names)) for j in range(i + 1, len(names))):
            raise RuntimeError("nested fold partitions overlap game identities")
        if seen_test_games & sets["test"]:
            raise RuntimeError("outer test game was reused")
        seen_test_games.update(sets["test"])
        fold = seal_document({
            "schema_version": "stage6_retrospective_nested_fold_assignment_v1",
            "fold_id": f"outer-fold-{fold_index:02d}",
            "outer_fold_index": fold_index,
            "whole_game_order": "maximum_decision_utc_then_game_id",
            "partitions": partitions,
            "partition_summaries": {
                name: _partition_summary(games, identities)
                for name, identities in partitions.items()
            },
            "future_game_count_excluded": len(game_ids) - test_cursor - len(test_games),
            "outer_test_targets_or_metrics_available_during_selection": False,
            "candidate_registry_bindings_self_sha256": registry_bindings[
                "self_sha256"
            ],
        })
        folds.append(fold)
        test_cursor += len(test_games)

    expected_test_games = set(game_ids[MINIMUM_INITIAL_HISTORY_GAMES:])
    if seen_test_games != expected_test_games:
        raise RuntimeError("nested outer folds do not conserve eligible test games")
    assignments = seal_document({
        "schema_version": "stage6_retrospective_nested_fold_assignments_v1",
        "status": "PASS_RETROSPECTIVE_NESTED_FOLD_FREEZE",
        "whole_game_order": "maximum_decision_utc_then_game_id",
        "initial_history_game_count": MINIMUM_INITIAL_HISTORY_GAMES,
        "outer_test_game_count": len(seen_test_games),
        "outer_test_game_identity_root_sha256": _identity_root(seen_test_games),
        "folds": folds,
        "candidate_registry_bindings": dict(registry_bindings),
    })
    manifest = seal_document({
        **common,
        "status": "PASS_RETROSPECTIVE_NESTED_FOLD_FREEZE",
        "outer_fold_count": len(folds),
        "outer_test_game_count": len(seen_test_games),
        "outer_test_game_identity_root_sha256": _identity_root(seen_test_games),
        "fold_assignment_self_sha256": assignments["self_sha256"],
        "minimum_outer_test_game_clusters_per_fold": MINIMUM_OUTER_TEST_GAMES,
        "minimum_independent_game_clusters_per_claim": MINIMUM_OUTER_TEST_GAMES,
        "candidate_registry_frozen_before_outer_execution": True,
        "outer_test_reuse_count_per_game": 1,
        "selection_isolation": {
            "model_family_hyperparameter_iteration_and_support_selection": "inner_train_and_validation_only",
            "scalar_temperature_selection": "inner_calibration_only",
            "outer_test": "score_once_only",
        },
    })
    return {"manifest": manifest, "assignments": assignments}


def _fixed_zip_bytes(members: Sequence[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return stream.getvalue()


def _validate_requested_workers(value: str) -> str:
    raw = str(value).strip()
    if raw.lower() == "auto" or raw in {"", "0"}:
        return "auto"
    if raw.isascii() and raw.isdecimal() and int(raw) > 0:
        return raw
    raise RuntimeError(
        "REQUESTED_WORKERS must be auto, blank, 0, or a positive integer"
    )


def run_retrospective_nested_fold_freeze(
    stage_drive_root: str | Path,
    *,
    requested_workers: str = "auto",
    accelerator: str = "CPU",
) -> dict[str, Any]:
    """Load all 48 frozen lanes and publish the exact nested fold plan."""
    requested_workers = _validate_requested_workers(requested_workers)
    accelerator = str(accelerator).strip().upper()
    if accelerator not in {"CPU", "GPU", "TPU"}:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    contract = _load_contract()

    import stage6_expanded_feature_freeze_v2 as feature_freeze_v2

    existing = sys.modules.get("stage6_expanded_feature_freeze")
    if existing not in (None, feature_freeze_v2):
        raise RuntimeError("R5 feature release must bind before historical consumers")
    sys.modules["stage6_expanded_feature_freeze"] = feature_freeze_v2

    from stage6_drive_fuse_publication import create_or_verify_bytes
    from stage6_expanded_tournament import load_layer_dataset

    stage_drive_root = Path(stage_drive_root).resolve()
    print(json.dumps({
        "event": "stage6_retrospective_nested_fold_freeze_started",
        "phase": "loading_all_frozen_logical_lanes",
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "requested_workers": requested_workers,
        "accepted_accelerator": accelerator,
        "model_training_performed": False,
    }, sort_keys=True), flush=True)
    dataset = load_layer_dataset(stage_drive_root, "within_hand_line_surprise")
    registries = candidate_registry_bindings()
    result = build_nested_fold_plan(
        dataset["frame"],
        feature_manifest_self_sha256=dataset["feature_manifest_self_sha256"],
        registry_bindings=registries,
        contract_self_sha256=contract["self_sha256"],
    )
    manifest = result["manifest"]
    assignments = result["assignments"]
    run_id = hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_retrospective_nested_fold_freeze_run_v1",
        "manifest_self_sha256": manifest["self_sha256"],
        "assignments_self_sha256": assignments["self_sha256"],
        "implementation_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
    })).hexdigest()
    work_root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "fold_freeze"
        / run_id
    )
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    assignments_raw = canonical_json_bytes(assignments) + b"\n"
    manifest_path = work_root / "nested-fold-freeze-manifest.json"
    assignments_path = work_root / "nested-fold-assignments.json"
    create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=run_id)
    create_or_verify_bytes(assignments_path, assignments_raw, owner_run_id=run_id)
    returned_raw = _fixed_zip_bytes([
        ("nested-fold-freeze-manifest.json", manifest_raw),
        ("nested-fold-assignments.json", assignments_raw),
    ])
    returned_name = (
        "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_FOLD_FREEZE_RESULT_"
        f"{run_id[:16]}.zip"
    )
    returned_path = stage_drive_root / "returned_results" / returned_name
    returned_path.parent.mkdir(parents=True, exist_ok=True)
    create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
    print(json.dumps({
        "event": "stage6_retrospective_nested_fold_freeze_complete",
        "status": manifest["status"],
        "outer_fold_count": manifest["outer_fold_count"],
        "outer_test_game_count": manifest["outer_test_game_count"],
        "checkpoint": str(manifest_path),
    }, sort_keys=True), flush=True)
    return {
        "status": manifest["status"],
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "assignments_path": str(assignments_path),
        "assignments_self_sha256": assignments["self_sha256"],
        "outer_fold_count": manifest["outer_fold_count"],
        "outer_test_game_count": manifest["outer_test_game_count"],
        "qualification_authority": False,
        "fresh_or_prospective_claim": False,
        "returned_result": {
            "path": str(returned_path),
            "name": returned_name,
            "bytes": len(returned_raw),
            "sha256": hashlib.sha256(returned_raw).hexdigest(),
        },
    }
