"""Locked-tail adjudication of frozen Stage 6 candidates on the current corpus.

The user explicitly selected the already-processed corpus instead of waiting for
future hands.  Targets in the retrospective role were opened previously, so this
module never describes the result as fresh.  Frozen model/configuration bytes are
calibrated on earlier whole games and evaluated once on later whole games.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
RETROSPECTIVE_ROLE = "retrospective_calibration"
TEMPERATURE_GRID = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00)
LOGICAL_DATA_LANES = 48
MINIMUM_GAME_CLUSTERS = 20
FAMILY_ALPHA = 0.05

EXPANDED_FREEZE_SELF_SHA256 = (
    "2c2fb61d9cde6b3a3b888b7a708df8215c7ea13bb6e31275a709e84fb1fa474d"
)
BEHAVIOR_FREEZE_SELF_SHA256 = (
    "a4e8e28b13054ee889d5bf45809318b9ce5022ebf15c6a22e7dbd1a437084fc8"
)
BEHAVIOR_AGGREGATE_RUN_ID = (
    "b4c446c74078dbc764625cf3685ef9355e625b3837f1abea493d62b1bf61bba0"
)
EXPANDED_AGGREGATE_RUN_ID = (
    "ae8c18653df5f698e95fed02ab2e2ba1e1b78fbd41c0c9285425e32ac4e6bcb9"
)
ADDITIONAL_FREEZE_SELF_SHA256 = (
    "c3e36953c7eda47277e3d3535af7d458736e655fc490b43bf705c23942f436f2"
)
ADDITIONAL_SOURCE_SELF_SHA256 = (
    "21c9c543a5b0ea43155956e2a0b317285025136f029f744575ffd20dccbfa168"
)
NOTEBOOK01_RUN_ID = (
    "8a6ee5a801b7ae945ddcca7e122d33dd722d25ab5112b6de785965d62e4c6226"
)

EXPANDED_LAYERS = (
    "hero_opponent_interaction",
    "within_hand_line_surprise",
)
BEHAVIOR_SCOPES = ("preflop", "postflop")
ADDITIONAL_LAYERS = (
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "opponent_bluff_likelihood",
    "player_strategy_drift",
    "showdown_aware_range_calibration",
    "supported_fold_response_bluff_opportunity",
)
ALL_CANDIDATE_LAYERS = (
    *EXPANDED_LAYERS,
    "preflop_behavior",
    "postflop_behavior",
    *ADDITIONAL_LAYERS,
)
REQUIRED_BEHAVIOR_BASELINES = (
    "class_frequency",
    "current_qualified_artifact",
    "deterministic_chart_or_heuristic",
    "legal_random_or_uniform_where_semantically_valid",
)
VARIABLE_EXECUTION_TELEMETRY_FIELDS = frozenset({
    "accepted_accelerator",
    "elapsed_seconds",
    "worker_report",
})


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_deterministic_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    from stage6_expanded_layers import seal_document

    overlap = VARIABLE_EXECUTION_TELEMETRY_FIELDS.intersection(payload)
    if overlap:
        raise RuntimeError(
            "variable execution telemetry cannot enter sealed result bytes: "
            + ", ".join(sorted(overlap))
        )
    return seal_document(payload)


def _identity_root(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0))
    logits /= float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _project_legal(
    probabilities: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    probabilities = np.asarray(probabilities, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    priors = np.asarray(priors, dtype=float)
    if probabilities.shape != legal.shape or probabilities.ndim != 2:
        raise RuntimeError("probability and legal-mask shapes differ")
    if legal.shape[1] != len(priors) or not legal.any(axis=1).all():
        raise RuntimeError("visible legal action inventory is invalid")
    raw_illegal_argmax = int(
        (~legal[np.arange(len(legal)), probabilities.argmax(axis=1)]).sum()
    )
    projected = np.where(legal, probabilities, 0.0)
    totals = projected.sum(axis=1)
    fallback_count = 0
    for row in np.flatnonzero(totals <= 0.0):
        choices = np.flatnonzero(legal[row])
        choice = choices[int(np.argmax(priors[choices]))]
        projected[row, choice] = 1.0
        totals[row] = 1.0
        fallback_count += 1
    projected /= totals[:, None]
    served_illegal_count = int(np.count_nonzero(projected[~legal] > 0.0))
    if not np.isfinite(projected).all():
        raise RuntimeError("legal projection emitted non-finite probabilities")
    return projected, raw_illegal_argmax, fallback_count, served_illegal_count


def select_temperature(
    projected: np.ndarray,
    legal: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    priors: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float], dict[str, int]]:
    """Select only a scalar temperature on the supplied calibration rows."""
    target = np.asarray(target, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise RuntimeError("temperature calibration partition is empty")
    rows = np.flatnonzero(mask)
    trials: list[tuple[float, float, np.ndarray, dict[str, int]]] = []
    for temperature in TEMPERATURE_GRID:
        scaled = _temperature_scale(projected, temperature)
        scaled, raw_illegal, fallback, served_illegal = _project_legal(
            scaled, legal, priors
        )
        loss = float(
            -np.log(np.clip(scaled[rows, target[rows]], 1e-15, 1.0)).mean()
        )
        trials.append((loss, temperature, scaled, {
            "raw_preprojection_illegal_argmax_count": raw_illegal,
            "zero_legal_mass_fallback_count": fallback,
            "served_illegal_probability_mass_count": served_illegal,
        }))
    trials.sort(key=lambda row: (row[0], row[1]))
    loss, temperature, scaled, legality = trials[0]
    return (
        float(temperature),
        scaled,
        {str(temp): float(value) for value, temp, _, _ in trials},
        legality,
    )


def locked_tail_split(frame: pd.DataFrame) -> dict[str, Any]:
    """Split a retrospective panel at a deterministic whole-game boundary."""
    required = {"row_id", "game_id", "hand_id", "decision_utc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"locked-tail split is missing columns: {missing}")
    source = frame.loc[:, sorted(required)].copy()
    for column in ("row_id", "game_id", "hand_id", "decision_utc"):
        source[column] = source[column].astype(str)
        if source[column].eq("").any():
            raise RuntimeError(f"locked-tail split has empty {column}")
    if source["row_id"].duplicated().any():
        raise RuntimeError("locked-tail split duplicates row_id")
    games = (
        source.groupby("game_id", sort=False, observed=True)
        .agg(
            maximum_decision_utc=("decision_utc", "max"),
            minimum_decision_utc=("decision_utc", "min"),
            row_count=("row_id", "size"),
            hand_count=("hand_id", "nunique"),
        )
        .reset_index()
        .sort_values(["maximum_decision_utc", "game_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(games) < 2 * MINIMUM_GAME_CLUSTERS:
        raise RuntimeError("locked-tail panel has too few games for both partitions")
    cumulative = games["row_count"].cumsum().to_numpy(dtype=np.int64)
    target = float(len(source)) / 2.0
    candidates = np.arange(MINIMUM_GAME_CLUSTERS - 1, len(games) - MINIMUM_GAME_CLUSTERS)
    if not len(candidates):
        raise RuntimeError("locked-tail split has no valid whole-game boundary")
    boundary = int(
        min(candidates.tolist(), key=lambda index: (abs(float(cumulative[index]) - target), index))
    )
    calibration_games = tuple(games.loc[:boundary, "game_id"].astype(str))
    evaluation_games = tuple(games.loc[boundary + 1:, "game_id"].astype(str))
    calibration_set, evaluation_set = set(calibration_games), set(evaluation_games)
    if calibration_set & evaluation_set:
        raise RuntimeError("locked-tail split overlaps game identities")
    calibration_mask = source["game_id"].isin(calibration_set).to_numpy(dtype=bool)
    evaluation_mask = source["game_id"].isin(evaluation_set).to_numpy(dtype=bool)
    calibration_hands = set(
        source.loc[calibration_mask, "game_id"] + "\x1f"
        + source.loc[calibration_mask, "hand_id"]
    )
    evaluation_hands = set(
        source.loc[evaluation_mask, "game_id"] + "\x1f"
        + source.loc[evaluation_mask, "hand_id"]
    )
    if calibration_hands & evaluation_hands or not np.all(calibration_mask ^ evaluation_mask):
        raise RuntimeError("locked-tail split overlaps hands or loses rows")
    manifest = {
        "schema_version": "stage6_current_corpus_locked_tail_split_v1",
        "source_role": RETROSPECTIVE_ROLE,
        "current_corpus_targets_previously_opened": True,
        "fresh_claim": False,
        "whole_game_order": "maximum_decision_utc_then_game_id",
        "boundary_game_id": calibration_games[-1],
        "boundary_game_maximum_decision_utc": str(
            games.loc[boundary, "maximum_decision_utc"]
        ),
        "total_row_count": int(len(source)),
        "total_game_count": int(len(games)),
        "total_hand_count": int(
            source[["game_id", "hand_id"]].drop_duplicates().shape[0]
        ),
        "calibration_row_count": int(calibration_mask.sum()),
        "calibration_game_count": len(calibration_games),
        "calibration_hand_count": len(calibration_hands),
        "evaluation_row_count": int(evaluation_mask.sum()),
        "evaluation_game_count": len(evaluation_games),
        "evaluation_hand_count": len(evaluation_hands),
        "calibration_game_identity_root_sha256": _identity_root(calibration_games),
        "evaluation_game_identity_root_sha256": _identity_root(evaluation_games),
        "calibration_row_identity_root_sha256": _identity_root(
            source.loc[calibration_mask, "row_id"].tolist()
        ),
        "evaluation_row_identity_root_sha256": _identity_root(
            source.loc[evaluation_mask, "row_id"].tolist()
        ),
        "game_overlap_count": 0,
        "hand_overlap_count": 0,
    }
    manifest["self_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    return {
        "manifest": manifest,
        "calibration_games": calibration_games,
        "evaluation_games": evaluation_games,
        "calibration_mask": calibration_mask,
        "evaluation_mask": evaluation_mask,
    }


def insufficient_cluster_manifest(
    frame: pd.DataFrame,
    implementation_sha256: str,
) -> dict[str, Any]:
    """Seal the current-corpus terminal hold without weakening the 20-game gate."""
    required = {"row_id", "game_id", "hand_id", "decision_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"current-corpus inventory is missing columns: {missing}")
    game_ids = frame["game_id"].astype(str)
    hand_ids = game_ids + "\x1f" + frame["hand_id"].astype(str)
    inventory = {
        "decision_row_count": int(len(frame)),
        "whole_hand_count": int(hand_ids.nunique()),
        "independent_game_cluster_count": int(game_ids.nunique()),
        "minimum_decision_utc": str(frame["decision_utc"].astype(str).min()),
        "maximum_decision_utc": str(frame["decision_utc"].astype(str).max()),
    }
    run_id = hashlib.sha256(_canonical_json_bytes({
        "schema": "stage6_current_corpus_insufficient_clusters_v1",
        "notebook01_run_id": NOTEBOOK01_RUN_ID,
        "minimum_independent_game_clusters_per_claim": MINIMUM_GAME_CLUSTERS,
        "inventory": inventory,
        "implementation_sha256": implementation_sha256,
    })).hexdigest()
    return _seal_deterministic_result({
        "schema_version": "stage6_current_corpus_locked_tail_adjudication_v1",
        "status": "HELD_INSUFFICIENT_INDEPENDENT_GAME_CLUSTERS",
        "run_id": run_id,
        "notebook01_run_id": NOTEBOOK01_RUN_ID,
        "current_corpus_targets_previously_opened": True,
        "fresh_or_never_opened_claim": False,
        "qualification_authority": False,
        "method": "current_corpus_gate_preflight_only",
        "blocking_gate": {
            "cluster_unit": "game_id",
            "minimum_independent_game_clusters_per_claim": MINIMUM_GAME_CLUSTERS,
            "observed_independent_game_clusters": inventory[
                "independent_game_cluster_count"
            ],
            "gate_met": False,
        },
        "inventory": inventory,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "model_evaluation_performed": False,
        "temperature_calibration_performed": False,
        "model_refit_count": 0,
        "implementation_sha256": implementation_sha256,
        "freeze_bindings": {
            "expanded": {
                "aggregate_run_id": EXPANDED_AGGREGATE_RUN_ID,
                "self_sha256": EXPANDED_FREEZE_SELF_SHA256,
            },
            "behavior": {
                "aggregate_run_id": BEHAVIOR_AGGREGATE_RUN_ID,
                "self_sha256": BEHAVIOR_FREEZE_SELF_SHA256,
            },
            "additional": ADDITIONAL_FREEZE_SELF_SHA256,
            "additional_source": ADDITIONAL_SOURCE_SELF_SHA256,
        },
        "candidate_layer_count": len(ALL_CANDIDATE_LAYERS),
        "qualified_layer_count": 0,
        "qualified_layers": [],
        "held_layers": list(ALL_CANDIDATE_LAYERS),
        "assistant_installation_allowed_for_qualified_layers_after_focused_independent_audit": False,
        "automatic_assistant_installation_performed": False,
        "direct_action_authority": False,
        "live_action_authority": False,
    })


def holm_step_down(
    pvalues: Mapping[str, float], alpha: float = FAMILY_ALPHA
) -> dict[str, dict[str, Any]]:
    ordered = sorted(pvalues, key=lambda key: (float(pvalues[key]), key))
    output: dict[str, dict[str, Any]] = {}
    continue_rejecting = True
    count = len(ordered)
    running_adjusted = 0.0
    for rank, key in enumerate(ordered, start=1):
        pvalue = float(pvalues[key])
        threshold = float(alpha) / (count - rank + 1)
        reject = bool(continue_rejecting and pvalue <= threshold)
        if not reject:
            continue_rejecting = False
        running_adjusted = max(running_adjusted, min(1.0, (count - rank + 1) * pvalue))
        output[key] = {
            "holm_rank": rank,
            "raw_p_value": pvalue,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": running_adjusted,
            "reject_no_improvement_null": reject,
        }
    return output


def _read_bound_manifest(path: Path, schema: str, expected_self: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required frozen manifest is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    actual_self = document.get("self_sha256")
    unsealed = dict(document)
    unsealed.pop("self_sha256", None)
    if (
        document.get("schema_version") != schema
        or actual_self != expected_self
        or hashlib.sha256(_canonical_json_bytes(unsealed)).hexdigest() != actual_self
    ):
        raise RuntimeError(f"frozen manifest authority mismatch: {path}")
    return document


def _load_joblib_identity(root: Path, reference: Mapping[str, Any]) -> Any:
    import joblib

    path = root / Path(str(reference["artifact_relative_path"]))
    identity = reference["selected_artifact"]
    if (
        not path.is_file()
        or path.name != identity["name"]
        or path.stat().st_size != int(identity["bytes"])
        or _sha256_file(path) != identity["sha256"]
    ):
        raise RuntimeError(f"frozen artifact identity mismatch: {path}")
    return joblib.load(path)


def _metric_report(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from stage6_metrics import multiclass_metric_report

    truth = [ACTIONS[int(value)] for value in np.asarray(target, dtype=int)]
    return multiclass_metric_report(truth, probabilities, ACTIONS)


def _expanded_claim(
    stage_drive_root: Path,
    layer: str,
    feature_frame: pd.DataFrame,
    split: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    from stage6_expanded_feature_freeze import BASE_FEATURE_COLUMNS
    from stage6_expanded_tournament import (
        _aligned_probabilities,
        _apply_imputer,
        expanded_model_matrix,
    )

    frame = feature_frame.loc[
        feature_frame["temporal_role"].astype(str).eq(RETROSPECTIVE_ROLE)
    ].reset_index(drop=True)
    artifact = _load_joblib_identity(stage_drive_root, freeze["finalists"][layer])
    selected_names, selected_x = expanded_model_matrix(frame, layer)
    if selected_names != list(artifact["selected_feature_names"]):
        raise RuntimeError(f"{layer} selected feature schema changed")
    if layer == "hero_opponent_interaction":
        control_names, control_x = expanded_model_matrix(
            frame, layer, include_dyad_interaction=False
        )
        support = (
            pd.to_numeric(frame["dyad_prior_hand_count"], errors="raise")
            .fillna(0).to_numpy(dtype=float) >= 2.0
        )
    else:
        control_names = list(BASE_FEATURE_COLUMNS)
        control_x = frame[control_names].to_numpy(dtype=np.float32)
        support = (
            pd.to_numeric(frame["line_prefix_decision_count"], errors="raise")
            .to_numpy(dtype=np.int64) >= 1
        )
    if control_names != list(artifact["control_feature_names"]):
        raise RuntimeError(f"{layer} control feature schema changed")
    selected_x = _apply_imputer(
        selected_x, np.asarray(artifact["selected_imputer_medians"], dtype=np.float32)
    )
    control_x = _apply_imputer(
        control_x, np.asarray(artifact["control_imputer_medians"], dtype=np.float32)
    )
    legal = frame[[f"base_legal_{action}" for action in ACTIONS]].to_numpy(dtype=bool)
    target = np.asarray(
        [ACTIONS.index(value) for value in frame["action_label"].astype(str)],
        dtype=np.int8,
    )
    priors = np.asarray(artifact["class_priors"], dtype=float)
    selected_raw = _aligned_probabilities(artifact["selected_model"], selected_x)
    control_raw = _aligned_probabilities(artifact["control_model"], control_x)
    selected_projected, _, _, _ = _project_legal(selected_raw, legal, priors)
    control_projected, _, _, _ = _project_legal(control_raw, legal, priors)
    calibration_games = set(split["calibration_games"])
    evaluation_games = set(split["evaluation_games"])
    calibration = frame["game_id"].astype(str).isin(calibration_games).to_numpy() & support
    evaluation = frame["game_id"].astype(str).isin(evaluation_games).to_numpy() & support
    selected_temperature, selected, selected_trials, selected_scaled_legality = select_temperature(
        selected_projected, legal, target, calibration, priors
    )
    control_temperature, control, control_trials, control_scaled_legality = select_temperature(
        control_projected, legal, target, calibration, priors
    )
    positions = np.flatnonzero(evaluation)
    if not len(positions):
        raise RuntimeError(f"{layer} evaluation claim stratum is empty")
    selected_metrics = _metric_report(target[positions], selected[positions])
    control_metrics = _metric_report(target[positions], control[positions])
    selected_loss = -np.log(
        np.clip(selected[positions, target[positions]], 1e-15, 1.0)
    )
    control_loss = -np.log(
        np.clip(control[positions, target[positions]], 1e-15, 1.0)
    )
    from stage6_expanded_adjudication import _paired_cluster_inference

    inference = _paired_cluster_inference(
        selected_loss - control_loss,
        frame.iloc[positions]["game_id"].astype(str).tolist(),
        int(hashlib.sha256(("current:" + layer).encode()).hexdigest()[:8], 16),
        MINIMUM_GAME_CLUSTERS,
    )
    below = ~support & frame["game_id"].astype(str).isin(evaluation_games).to_numpy()
    evaluation_all = frame["game_id"].astype(str).isin(evaluation_games).to_numpy()
    evaluation_rows = np.flatnonzero(evaluation_all)
    selected_eval_projected, selected_illegal, selected_fallback, selected_served_illegal = (
        _project_legal(selected_raw[evaluation_rows], legal[evaluation_rows], priors)
    )
    _, selected_scaled_illegal, selected_scaled_fallback, selected_scaled_served_illegal = (
        _project_legal(
            _temperature_scale(selected_eval_projected, selected_temperature),
            legal[evaluation_rows], priors,
        )
    )
    control_eval_projected, control_illegal, control_fallback, control_served_illegal = (
        _project_legal(control_raw[evaluation_rows], legal[evaluation_rows], priors)
    )
    _, control_scaled_illegal, control_scaled_fallback, control_scaled_served_illegal = (
        _project_legal(
            _temperature_scale(control_eval_projected, control_temperature),
            legal[evaluation_rows], priors,
        )
    )
    served = selected.copy()
    served[~support] = control[~support]
    exact_control = bool(np.array_equal(served[below], control[below]))
    return {
        "layer": layer,
        "candidate_id": freeze["finalists"][layer]["selected_candidate_id"],
        "calibration_row_count": int(calibration.sum()),
        "evaluation_row_count": int(positions.size),
        "evaluation_game_cluster_count": int(
            frame.iloc[positions]["game_id"].astype(str).nunique()
        ),
        "selected_temperature": selected_temperature,
        "control_temperature": control_temperature,
        "selected_temperature_grid_log_loss": selected_trials,
        "control_temperature_grid_log_loss": control_trials,
        "selected_metrics": selected_metrics,
        "matched_control_metrics": control_metrics,
        "inference": inference,
        "raw_preprojection_illegal_argmax_count": int(
            selected_illegal + selected_scaled_illegal
            + control_illegal + control_scaled_illegal
        ),
        "zero_legal_mass_fallback_count": int(
            selected_fallback + control_fallback
            + selected_scaled_fallback + control_scaled_fallback
        ),
        "served_illegal_probability_mass_count": int(
            selected_served_illegal + control_served_illegal
            + selected_scaled_served_illegal + control_scaled_served_illegal
        ),
        "below_support_evaluation_row_count": int(below.sum()),
        "below_support_served_exact_matched_control": exact_control,
    }


def _behavior_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from stage6_behavior_tournament import BOOLEAN_FEATURES, NUMERIC_FEATURES
    from stage6_expanded_feature_freeze_v2 import _safe_base_feature_name

    result = frame.copy()
    for name in (*BOOLEAN_FEATURES, *NUMERIC_FEATURES):
        safe = _safe_base_feature_name(name)
        if safe in result.columns:
            result[name] = result[safe]
    return result


def _behavior_raw_candidate(
    artifact: Mapping[str, Any], matrix: np.ndarray, legal: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    from stage6_behavior_tournament import _aligned_predict_proba, _apply_imputer

    if artifact.get("artifact_kind") != "single_model":
        raise RuntimeError("current-corpus behavior release expects one frozen model")
    raw = _aligned_predict_proba(
        artifact["model"],
        _apply_imputer(
            matrix, np.asarray(artifact["imputer_medians"], dtype=np.float32)
        ),
    )
    priors = np.asarray(artifact["class_priors"], dtype=float)
    projected, illegal, fallback, served_illegal = _project_legal(raw, legal, priors)
    return projected, {
        "raw_preprojection_illegal_argmax_count": illegal,
        "zero_legal_mass_fallback_count": fallback,
        "served_illegal_probability_mass_count": served_illegal,
    }


def _behavior_native_temperature(
    scope: str,
    frame: pd.DataFrame,
    projected: np.ndarray,
    legal: np.ndarray,
    artifact: Mapping[str, Any],
    notebook09_root: Path,
    context_root: Path,
    hand_sizes: Mapping[str, int],
) -> tuple[float, dict[str, float]]:
    from stage6_notebook09_runtime import _incumbent_comparison
    from stage6_notebook11_runtime import _map_six_action_baseline, _losses

    truth_six = frame["action_label"].astype(str).tolist()
    anchor = _incumbent_comparison(
        frame, projected, legal, truth_six, scope,
        str(notebook09_root), hand_sizes, str(context_root),
    )
    positions = np.asarray(anchor["positions"], dtype=int)
    trials: list[tuple[float, float]] = []
    priors = np.asarray(artifact["class_priors"], dtype=float)
    for temperature in TEMPERATURE_GRID:
        scaled, _, fallback, served_illegal = _project_legal(
            _temperature_scale(projected, temperature), legal, priors
        )
        if fallback or served_illegal:
            raise RuntimeError("behavior temperature projection required a fallback")
        labels, truth, mapped = _map_six_action_baseline(
            scope, scaled, legal, truth_six, positions
        )
        if labels != list(anchor["labels"]) or truth != list(anchor["truth"]):
            raise RuntimeError("behavior calibration native mapping changed")
        trials.append((float(_losses(truth, mapped, labels).mean()), temperature))
    trials.sort(key=lambda row: (row[0], row[1]))
    return float(trials[0][1]), {str(temp): value for value, temp in trials}


def _behavior_evaluation(
    scope: str,
    frame: pd.DataFrame,
    candidate: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
    notebook09_root: Path,
    context_root: Path,
    hand_sizes: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[str]]:
    from stage6_metrics import multiclass_metric_report
    from stage6_notebook09_runtime import (
        _deterministic_public_heuristic,
        _incumbent_comparison,
        _legal_uniform,
    )
    from stage6_notebook11_runtime import _losses, _map_six_action_baseline

    truth_six = frame["action_label"].astype(str).tolist()
    incumbent = _incumbent_comparison(
        frame, candidate, legal, truth_six, scope,
        str(notebook09_root), hand_sizes, str(context_root),
    )
    positions = np.asarray(incumbent["positions"], dtype=int)
    labels = list(incumbent["labels"])
    truth = list(incumbent["truth"])
    class_frequency, _, class_fallback, class_served_illegal = _project_legal(
        np.broadcast_to(priors, candidate.shape).copy(), legal, priors
    )
    if class_fallback or class_served_illegal:
        raise RuntimeError("behavior class-frequency projection failed")
    weak: dict[str, np.ndarray] = {}
    for name, six in (
        ("class_frequency", class_frequency),
        ("deterministic_chart_or_heuristic", _deterministic_public_heuristic(frame, legal)),
        ("legal_random_or_uniform_where_semantically_valid", _legal_uniform(legal)),
    ):
        mapped_labels, mapped_truth, mapped = _map_six_action_baseline(
            scope, six, legal, truth_six, positions
        )
        if mapped_labels != labels or mapped_truth != truth:
            raise RuntimeError(f"{scope} behavior baseline mapping changed")
        weak[name] = mapped
    matrices = {
        "candidate": np.asarray(incumbent["candidate"], dtype=float),
        "current_qualified_artifact": np.asarray(incumbent["incumbent"], dtype=float),
        **weak,
    }
    metrics = {
        name: multiclass_metric_report(truth, probabilities, labels)
        for name, probabilities in matrices.items()
    }
    candidate_losses = _losses(truth, matrices["candidate"], labels)
    differences = {
        name: candidate_losses - _losses(truth, probabilities, labels)
        for name, probabilities in matrices.items() if name != "candidate"
    }
    clusters = frame.iloc[positions]["game_id"].astype(str).tolist()
    sizes = np.asarray([
        hand_sizes[f"{game}\x1f{hand}"]
        for game, hand in zip(
            frame.iloc[positions]["game_id"].astype(str),
            frame.iloc[positions]["hand_id"].astype(str),
        )
    ], dtype=int)
    truth_array = np.asarray(truth, dtype=object)
    per_table_size: dict[str, Any] = {}
    for size in sorted(set(sizes.tolist())):
        mask = sizes == size
        per_table_size[str(size)] = {
            name: multiclass_metric_report(
                truth_array[mask].tolist(), probabilities[mask], labels
            ) for name, probabilities in matrices.items()
        }
    report = {
        "scope": f"{scope}_behavior",
        "evaluation_scope_row_count": len(frame),
        "native_domain_row_count": len(truth),
        "native_domain_game_cluster_count": len(set(clusters)),
        "metrics": metrics,
        "per_table_size_metrics": per_table_size,
        "illegal_postprojection_count": 0,
        "incumbent_illegal_preprojection_count": int(
            incumbent["incumbent_illegal_preprojection"]
        ),
        "incumbent_zero_mass_fallback_count": int(
            incumbent["incumbent_zero_mass_fallback"]
        ),
        "missing_or_defaulted_required_incumbent_input_count": int(
            incumbent["missing_or_defaulted_required_input_count"]
        ),
        "comparison_label_space": incumbent["comparison_label_space"],
    }
    return report, differences, clusters


def _load_behavior_freeze(stage_drive_root: Path) -> dict[str, Any]:
    return _read_bound_manifest(
        stage_drive_root
        / "runtime_state"
        / "behavior_challenger_v2"
        / "aggregate_freezes"
        / BEHAVIOR_AGGREGATE_RUN_ID
        / "aggregate-freeze-manifest.json",
        "stage6_behavior_challenger_aggregate_freeze_manifest_v3",
        BEHAVIOR_FREEZE_SELF_SHA256,
    )


def _load_expanded_freeze(stage_drive_root: Path) -> dict[str, Any]:
    return _read_bound_manifest(
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "aggregate_freezes"
        / EXPANDED_AGGREGATE_RUN_ID
        / "aggregate-freeze-manifest.json",
        "stage6_expanded_aggregate_freeze_manifest_v1",
        EXPANDED_FREEZE_SELF_SHA256,
    )


def _additional_claims(
    stage_drive_root: Path,
    split: Mapping[str, Any],
    row_to_game: pd.Series,
    progress: Any,
    layer_progress: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from stage6_additional_exploitation_execution import (
        FINALIST_MANIFEST_NAME,
        ROLE_TO_INDEX,
        _aligned_probabilities,
        _game_cluster_mean_differences,
        _paired_sign_flip_pvalue,
        load_execution_source,
    )
    from stage6_metrics import multiclass_metric_report

    finalist_root = stage_drive_root / "runtime_state/additional_exploitation/finalist"
    source_root = stage_drive_root / "runtime_state/additional_exploitation/development_source"
    finalist = _read_bound_manifest(
        finalist_root / FINALIST_MANIFEST_NAME,
        "stage6_additional_exploitation_finalist_freeze_v1",
        ADDITIONAL_FREEZE_SELF_SHA256,
    )
    local_cache = Path("/content/stage6-additional-current-cache")
    if not Path("/content").is_dir():
        local_cache = Path(tempfile.gettempdir()) / "stage6-additional-current-cache"
    source = load_execution_source(
        source_root,
        local_cache_root=local_cache,
        progress=progress,
    )
    if source["manifest"].get("self_sha256") != ADDITIONAL_SOURCE_SELF_SHA256:
        raise RuntimeError("additional execution source identity changed")
    calibration_games = set(split["calibration_games"])
    evaluation_games = set(split["evaluation_games"])
    results: list[dict[str, Any]] = []
    raw_pvalues: dict[str, float] = {}
    frozen_by_layer = {row["layer"]: row for row in finalist["finalists"]}
    for layer_index, layer in enumerate(ADDITIONAL_LAYERS, start=1):
        arrays = source["layers"][layer]
        roles = np.asarray(arrays["roles"])
        retro = np.flatnonzero(roles == ROLE_TO_INDEX[RETROSPECTIVE_ROLE])
        row_ids = np.asarray(arrays["row_ids"])[retro].astype(str)
        games = row_to_game.reindex(row_ids).to_numpy(dtype=object)
        if pd.isna(games).any():
            raise RuntimeError(f"{layer} retrospective row identity lacks a game mapping")
        games = games.astype(str)
        cal = np.isin(games, list(calibration_games))
        evaluation = np.isin(games, list(evaluation_games))
        if not np.all(cal ^ evaluation):
            raise RuntimeError(f"{layer} locked-tail game mapping is incomplete")
        x = np.asarray(arrays["x"])[retro]
        y = np.asarray(arrays["y"], dtype=int)[retro]
        legal = np.asarray(arrays["legal"], dtype=bool)[retro]
        names = list(arrays["feature_names"])
        support_columns = [i for i, name in enumerate(names) if name.endswith("_supported")]
        support = (
            np.any(x[:, support_columns] > 0.5, axis=1)
            if support_columns else np.ones(len(x), dtype=bool)
        )
        frozen = frozen_by_layer[layer]
        model_path = finalist_root / frozen["model"]["path"]
        identity = frozen["model"]
        if (
            not model_path.is_file()
            or model_path.stat().st_size != int(identity["bytes"])
            or _sha256_file(model_path) != identity["sha256"]
        ):
            raise RuntimeError(f"additional frozen model identity changed: {layer}")
        model = pickle.loads(model_path.read_bytes())
        priors = np.asarray(frozen["class_priors"], dtype=float)
        raw = _aligned_probabilities(model, x)
        projected, _, _, _ = _project_legal(raw, legal, priors)
        calibration = cal & support
        temperature, candidate, trials, scaled_legality = select_temperature(
            projected, legal, y, calibration, priors
        )
        eval_positions = np.flatnonzero(evaluation)
        supported = np.flatnonzero(evaluation & support)
        candidate_eval_projected, raw_illegal, fallback, served_illegal = _project_legal(
            raw[eval_positions], legal[eval_positions], priors
        )
        _, scaled_raw_illegal, scaled_fallback, scaled_served_illegal = _project_legal(
            _temperature_scale(candidate_eval_projected, temperature),
            legal[eval_positions], priors,
        )
        baseline_raw = np.broadcast_to(priors, candidate.shape).copy()
        baseline, _, _, _ = _project_legal(baseline_raw, legal, priors)
        _, baseline_raw_illegal, baseline_fallback, baseline_served_illegal = (
            _project_legal(
                baseline_raw[eval_positions], legal[eval_positions], priors
            )
        )
        truth = [ACTIONS[int(y[index])] for index in supported]
        candidate_metrics = multiclass_metric_report(
            truth, candidate[supported], ACTIONS
        )
        baseline_metrics = multiclass_metric_report(
            truth, baseline[supported], ACTIONS
        )
        candidate_loss = -np.log(
            np.clip(candidate[supported, y[supported]], 1e-15, 1.0)
        )
        baseline_loss = -np.log(
            np.clip(baseline[supported, y[supported]], 1e-15, 1.0)
        )
        cluster_differences = _game_cluster_mean_differences(
            candidate_loss - baseline_loss, games[supported]
        )
        pvalue = _paired_sign_flip_pvalue(cluster_differences)
        raw_pvalues[layer] = pvalue
        result = {
            "layer": layer,
            "candidate_id": frozen["candidate_id"],
            "calibration_supported_row_count": int(calibration.sum()),
            "evaluation_row_count": int(len(eval_positions)),
            "supported_evaluation_row_count": int(len(supported)),
            "prediction_support_coverage": float(len(supported) / len(eval_positions)),
            "supported_game_cluster_count": int(len(cluster_differences)),
            "selected_temperature": temperature,
            "temperature_grid_log_loss": trials,
            "candidate_metrics": candidate_metrics,
            "frozen_baseline_metrics": baseline_metrics,
            "candidate_log_loss_minus_baseline": float(
                candidate_metrics["log_loss"] - baseline_metrics["log_loss"]
            ),
            "candidate_brier_minus_baseline": float(
                candidate_metrics["multiclass_brier_score"]
                - baseline_metrics["multiclass_brier_score"]
            ),
            "paired_sign_flip_raw_pvalue": pvalue,
            "raw_preprojection_illegal_argmax_count": int(
                raw_illegal + baseline_raw_illegal
                + scaled_raw_illegal
            ),
            "served_illegal_probability_mass_count": int(
                served_illegal + baseline_served_illegal
                + scaled_served_illegal
            ),
            "zero_legal_mass_fallback_count": int(
                fallback + baseline_fallback
                + scaled_fallback
            ),
            "raw_preprojection_illegal_argmax_is_diagnostic_only": True,
        }
        results.append(result)
        model = None
        raw = projected = candidate = baseline = x = None
        gc.collect()
        print(json.dumps({
            "event": "stage6_current_corpus_additional_layer_complete",
            "layer": layer,
            "completed": layer_index,
            "denominator": len(ADDITIONAL_LAYERS),
            "evaluation_rows": len(eval_positions),
            "supported_rows": len(supported),
        }, sort_keys=True), flush=True)
        if layer_progress is not None:
            layer_progress(layer_index, len(ADDITIONAL_LAYERS), layer)
    holm = holm_step_down(raw_pvalues)
    gates = finalist["preregistered_fresh_gates"]
    for row in results:
        layer = row["layer"]
        row["paired_sign_flip_holm"] = holm[layer]
        row["gates"] = {
            "minimum_rows": row["evaluation_row_count"]
            >= int(gates["minimum_fresh_rows_per_layer"]),
            "minimum_supported_rows": row["supported_evaluation_row_count"]
            >= int(gates["minimum_supported_fresh_rows_per_layer"]),
            "prediction_support": row["prediction_support_coverage"]
            >= float(gates["minimum_prediction_support_coverage"]),
            "log_loss": row["candidate_log_loss_minus_baseline"] < 0.0,
            "brier": row["candidate_brier_minus_baseline"] <= 0.0,
            "holm_significance": holm[layer]["reject_no_improvement_null"],
            "legality": row["served_illegal_probability_mass_count"] == 0
            and row["zero_legal_mass_fallback_count"] == 0,
            "coverage": float(gates["minimum_legal_coverage"]) == 1.0,
        }
        row["qualified"] = bool(all(row["gates"].values()))
    return results, finalist


def _fixed_zip_bytes(members: Sequence[tuple[str, bytes]]) -> bytes:
    import io

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return stream.getvalue()


def run_current_corpus_adjudication(
    stage_drive_root: str | Path,
    *,
    requested_workers: str = "auto",
    accelerator: str = "CPU",
    heartbeat_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run the single current-corpus adjudication and publish a compact result."""
    # Notebook 12's completed R5 freeze is the v2 feature release. The
    # historical tournament module imports its provider by the old module
    # name, so bind that name before importing the consumer.
    import sys
    import stage6_expanded_feature_freeze_v2 as feature_freeze_v2

    existing_feature_provider = sys.modules.get("stage6_expanded_feature_freeze")
    if existing_feature_provider not in (None, feature_freeze_v2):
        raise RuntimeError("R5 feature release must bind before the historical provider")
    sys.modules["stage6_expanded_feature_freeze"] = feature_freeze_v2

    from stage6_behavior_challenger_notebook_runtime_v2 import ActiveWorkHeartbeat
    from stage6_drive_fuse_publication import create_or_verify_bytes
    from stage6_expanded_layers import seal_document
    from stage6_expanded_tournament import load_layer_dataset
    from stage6_notebook11_runtime import (
        _extract_notebook09_package,
        _hand_sizes,
        _prepare_incumbent_context,
    )
    from stage6_notebooks02_07_runtime import materialization_root, resolve_workers

    stage_drive_root = Path(stage_drive_root).resolve()
    accelerator = str(accelerator).strip().upper()
    if accelerator not in {"CPU", "GPU", "TPU"}:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    worker_report = resolve_workers(requested_workers)
    started = time.monotonic()
    heartbeat = ActiveWorkHeartbeat(
        scope="current_corpus_locked_tail_adjudication",
        shard=-1,
        interval_seconds=heartbeat_interval_seconds,
    )
    heartbeat.update(
        phase="loading_frozen_current_corpus",
        denominator=48,
        completed=0,
        work_unit="logical_lanes",
    )
    heartbeat.start()
    line_data: dict[str, Any] | None = None
    hero_data: dict[str, Any] | None = None
    try:
        expanded_freeze = _load_expanded_freeze(stage_drive_root)
        behavior_freeze = _load_behavior_freeze(stage_drive_root)
        line_data = load_layer_dataset(stage_drive_root, "within_hand_line_surprise")
        line_frame = line_data["feature_frame"]
        retrospective = line_frame.loc[
            line_frame["temporal_role"].astype(str).eq(RETROSPECTIVE_ROLE)
        ].reset_index(drop=True)
        observed_game_clusters = int(
            retrospective["game_id"].astype(str).nunique()
        )
        if observed_game_clusters < MINIMUM_GAME_CLUSTERS:
            heartbeat.update(
                phase="held_insufficient_independent_game_clusters",
                denominator=MINIMUM_GAME_CLUSTERS,
                completed=observed_game_clusters,
                work_unit="independent_game_clusters",
            )
            manifest = insufficient_cluster_manifest(
                retrospective, _sha256_file(Path(__file__).resolve())
            )
            run_id = manifest["run_id"]
            result_root = (
                stage_drive_root / "runtime_state/current_corpus_locked_tail" / run_id
            )
            result_root.mkdir(parents=True, exist_ok=True)
            manifest_raw = _canonical_json_bytes(manifest) + b"\n"
            manifest_path = result_root / "current-corpus-adjudication-manifest.json"
            create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=run_id)
            returned_raw = _fixed_zip_bytes([
                ("current-corpus-adjudication-manifest.json", manifest_raw),
            ])
            returned_name = (
                "POKERNOW_ASSISTANT_STAGE6_CURRENT_CORPUS_LOCKED_TAIL_RESULT_"
                f"{run_id[:16]}.zip"
            )
            returned_path = stage_drive_root / "returned_results" / returned_name
            returned_path.parent.mkdir(parents=True, exist_ok=True)
            create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
            return {
                "status": manifest["status"],
                "run_id": run_id,
                "manifest_path": str(manifest_path),
                "manifest_self_sha256": manifest["self_sha256"],
                "qualified_layer_count": 0,
                "qualified_layers": [],
                "held_layers": manifest["held_layers"],
                "execution_telemetry": {
                    "accepted_accelerator": accelerator,
                    "elapsed_seconds": time.monotonic() - started,
                    "worker_report": worker_report,
                },
                "returned_result": {
                    "path": str(returned_path),
                    "name": returned_name,
                    "bytes": len(returned_raw),
                    "sha256": hashlib.sha256(returned_raw).hexdigest(),
                },
            }
        split = locked_tail_split(retrospective)
        heartbeat.update(
            phase="expanded_and_behavior_adjudication",
            denominator=4,
            completed=0,
            work_unit="layers",
        )
        expanded_results: dict[str, dict[str, Any]] = {}
        expanded_results["within_hand_line_surprise"] = _expanded_claim(
            stage_drive_root, "within_hand_line_surprise", line_frame,
            split, expanded_freeze,
        )
        heartbeat.update(
            phase="expanded_and_behavior_adjudication", denominator=4,
            completed=1, work_unit="layers",
        )

        behavior_source = _behavior_base_frame(retrospective)
        local_root = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        run_prefix = split["manifest"]["self_sha256"][:16]
        notebook09_root = local_root / f"stage6-current-incumbent-{run_prefix}"
        notebook09_anchor = _extract_notebook09_package(stage_drive_root, notebook09_root)
        context_root = local_root / f"stage6-current-context-{run_prefix}"
        if context_root.exists():
            shutil.rmtree(context_root)
        _prepare_incumbent_context(
            stage_drive_root,
            materialization_root(stage_drive_root),
            behavior_source,
            notebook09_root,
            notebook09_anchor,
            context_root,
        )
        hand_sizes = _hand_sizes(behavior_source)
        from stage6_behavior_challenger_fresh_adjudication_v2 import _load_artifacts
        from stage6_expanded_feature_freeze_v2 import BASE_FEATURE_COLUMNS
        from stage6_expanded_adjudication import _paired_cluster_inference

        behavior_artifacts, behavior_identities = _load_artifacts(
            stage_drive_root, behavior_freeze
        )
        behavior_reports: dict[str, Any] = {}
        behavior_inference: dict[str, Any] = {}
        for scope_index, scope in enumerate(BEHAVIOR_SCOPES, start=1):
            scope_mask = (
                behavior_source["street"].astype(str).eq("preflop")
                if scope == "preflop"
                else behavior_source["street"].astype(str).isin(("flop", "turn", "river"))
            )
            scope_frame = behavior_source.loc[scope_mask].reset_index(drop=True)
            legal = scope_frame[[f"base_legal_{action}" for action in ACTIONS]].to_numpy(dtype=bool)
            matrix = scope_frame[list(BASE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
            artifact = behavior_artifacts[scope]
            normalized_artifact_names = [
                "".join(character if character.isalnum() else "_" for character in name)
                for name in artifact["feature_names"]
            ]
            normalized_artifact_names = [
                "base_" + "_".join(part for part in name.split("_") if part)
                for name in normalized_artifact_names
            ]
            if normalized_artifact_names != list(BASE_FEATURE_COLUMNS):
                raise RuntimeError(f"{scope} behavior frozen feature schema changed")
            projected, _ = _behavior_raw_candidate(artifact, matrix, legal)
            calibration_mask = scope_frame["game_id"].astype(str).isin(
                set(split["calibration_games"])
            ).to_numpy()
            evaluation_mask = scope_frame["game_id"].astype(str).isin(
                set(split["evaluation_games"])
            ).to_numpy()
            temperature, temperature_trials = _behavior_native_temperature(
                scope,
                scope_frame.loc[calibration_mask].reset_index(drop=True),
                projected[calibration_mask],
                legal[calibration_mask],
                artifact,
                notebook09_root,
                context_root,
                hand_sizes,
            )
            priors = np.asarray(artifact["class_priors"], dtype=float)
            evaluation_projected, evaluation_legality = _behavior_raw_candidate(
                artifact, matrix[evaluation_mask], legal[evaluation_mask]
            )
            candidate, scaled_illegal, scaled_fallback, scaled_served_illegal = _project_legal(
                _temperature_scale(evaluation_projected, temperature),
                legal[evaluation_mask], priors,
            )
            report, differences, clusters = _behavior_evaluation(
                scope,
                scope_frame.loc[evaluation_mask].reset_index(drop=True),
                candidate,
                legal[evaluation_mask],
                priors,
                notebook09_root,
                context_root,
                hand_sizes,
            )
            report.update({
                "candidate_id": artifact["candidate_id"],
                "calibration_scope_row_count": int(calibration_mask.sum()),
                "selected_temperature": temperature,
                "temperature_grid_native_log_loss": temperature_trials,
                "raw_preprojection_illegal_argmax_count": int(
                    evaluation_legality["raw_preprojection_illegal_argmax_count"]
                    + scaled_illegal
                ),
                "zero_legal_mass_fallback_count": int(
                    evaluation_legality["zero_legal_mass_fallback_count"]
                    + scaled_fallback
                ),
                "served_illegal_probability_mass_count": int(
                    evaluation_legality["served_illegal_probability_mass_count"]
                    + scaled_served_illegal
                ),
            })
            behavior_reports[f"{scope}_behavior"] = report
            for comparator in REQUIRED_BEHAVIOR_BASELINES:
                key = f"{scope}_behavior:{comparator}"
                behavior_inference[key] = _paired_cluster_inference(
                    differences[comparator],
                    clusters,
                    int(hashlib.sha256(("current:" + key).encode()).hexdigest()[:8], 16),
                    MINIMUM_GAME_CLUSTERS,
                )
            heartbeat.update(
                phase="expanded_and_behavior_adjudication", denominator=4,
                completed=1 + scope_index, work_unit="layers",
            )
        behavior_holm = holm_step_down({
            key: row["paired_sign_flip_two_sided_p_value"]
            for key, row in behavior_inference.items()
        })
        behavior_qualification: dict[str, Any] = {}
        for scope in BEHAVIOR_SCOPES:
            layer = f"{scope}_behavior"
            report = behavior_reports[layer]
            comparison_gates: dict[str, bool] = {}
            for comparator in REQUIRED_BEHAVIOR_BASELINES:
                key = f"{layer}:{comparator}"
                inference = behavior_inference[key]
                comparison_gates[comparator] = bool(
                    inference["independent_cluster_gate_met"]
                    and inference["mean_candidate_minus_control_log_loss"] < 0.0
                    and inference["confidence_interval_95"][1] < 0.0
                    and behavior_holm[key]["reject_no_improvement_null"]
                    and report["metrics"]["candidate"]["multiclass_brier_score"]
                    <= report["metrics"][comparator]["multiclass_brier_score"]
                )
            qualified = bool(
                all(comparison_gates.values())
                and report["illegal_postprojection_count"] == 0
                and report["zero_legal_mass_fallback_count"] == 0
                and report["served_illegal_probability_mass_count"] == 0
                and report["missing_or_defaulted_required_incumbent_input_count"] == 0
            )
            report["preregistered_comparison_gates"] = comparison_gates
            behavior_qualification[layer] = qualified
        heartbeat.update(
            phase="expanded_and_behavior_adjudication", denominator=4,
            completed=3, work_unit="layers",
        )

        del behavior_source, behavior_artifacts
        gc.collect()
        row_to_game = pd.Series(
            retrospective["game_id"].astype(str).to_numpy(),
            index=retrospective["row_id"].astype(str).to_numpy(),
        )
        del line_data, line_frame
        line_data = None
        gc.collect()
        hero_data = load_layer_dataset(stage_drive_root, "hero_opponent_interaction")
        expanded_results["hero_opponent_interaction"] = _expanded_claim(
            stage_drive_root, "hero_opponent_interaction", hero_data["feature_frame"],
            split, expanded_freeze,
        )
        heartbeat.update(
            phase="expanded_and_behavior_adjudication", denominator=4,
            completed=4, work_unit="layers",
        )
        del hero_data
        hero_data = None
        gc.collect()

        expanded_holm = holm_step_down({
            layer: row["inference"]["paired_sign_flip_two_sided_p_value"]
            for layer, row in expanded_results.items()
        })
        expanded_qualification: dict[str, bool] = {}
        for layer in EXPANDED_LAYERS:
            claim = expanded_results[layer]
            inference = claim["inference"]
            qualified = bool(
                inference["independent_cluster_gate_met"]
                and inference["mean_candidate_minus_control_log_loss"] < 0.0
                and inference["confidence_interval_95"][1] < 0.0
                and expanded_holm[layer]["reject_no_improvement_null"]
                and claim["selected_metrics"]["multiclass_brier_score"]
                <= claim["matched_control_metrics"]["multiclass_brier_score"]
                and claim["zero_legal_mass_fallback_count"] == 0
                and claim["served_illegal_probability_mass_count"] == 0
                and claim["below_support_served_exact_matched_control"]
            )
            claim["holm"] = expanded_holm[layer]
            claim["qualified"] = qualified
            expanded_qualification[layer] = qualified

        heartbeat.update(
            phase="additional_exploitation_adjudication", denominator=7,
            completed=0, work_unit="layers",
        )
        last_progress = {"completed": 0, "total": 1}

        def source_progress(completed: int, total: int, path: str) -> None:
            last_progress.update(completed=int(completed), total=int(total))
            print(json.dumps({
                "event": "stage6_current_corpus_additional_source_progress",
                "completed_bytes": int(completed),
                "denominator_bytes": int(total),
                "path": Path(path).name,
            }, sort_keys=True), flush=True)

        def additional_layer_progress(completed: int, total: int, layer: str) -> None:
            heartbeat.update(
                phase="additional_exploitation_adjudication",
                denominator=int(total),
                completed=int(completed),
                work_unit="layers",
            )

        additional_results, additional_freeze = _additional_claims(
            stage_drive_root, split, row_to_game, source_progress,
            additional_layer_progress,
        )
        qualified_layers = [
            *[layer for layer in EXPANDED_LAYERS if expanded_qualification[layer]],
            *[layer for layer in ("preflop_behavior", "postflop_behavior")
              if behavior_qualification[layer]],
            *[row["layer"] for row in additional_results if row["qualified"]],
        ]
        all_layers = [
            *EXPANDED_LAYERS,
            "preflop_behavior",
            "postflop_behavior",
            *ADDITIONAL_LAYERS,
        ]
        implementation_sha256 = _sha256_file(Path(__file__).resolve())
        run_id = hashlib.sha256(_canonical_json_bytes({
            "schema": "stage6_current_corpus_locked_tail_run_v1",
            "notebook01_run_id": NOTEBOOK01_RUN_ID,
            "split_self_sha256": split["manifest"]["self_sha256"],
            "expanded_freeze_self_sha256": EXPANDED_FREEZE_SELF_SHA256,
            "behavior_freeze_self_sha256": BEHAVIOR_FREEZE_SELF_SHA256,
            "additional_freeze_self_sha256": ADDITIONAL_FREEZE_SELF_SHA256,
            "additional_source_self_sha256": ADDITIONAL_SOURCE_SELF_SHA256,
            "implementation_sha256": implementation_sha256,
        })).hexdigest()
        manifest = _seal_deterministic_result({
            "schema_version": "stage6_current_corpus_locked_tail_adjudication_v1",
            "status": (
                "PASS_ALL_CURRENT_CORPUS_LOCKED_TAIL_GATES"
                if len(qualified_layers) == len(all_layers)
                else "HOLD_ONE_OR_MORE_CURRENT_CORPUS_LOCKED_TAIL_GATES"
            ),
            "run_id": run_id,
            "notebook01_run_id": NOTEBOOK01_RUN_ID,
            "current_corpus_targets_previously_opened": True,
            "fresh_or_never_opened_claim": False,
            "method": "earlier_whole_game_temperature_calibration_later_whole_game_evaluation",
            "split": split["manifest"],
            "temperature_grid": list(TEMPERATURE_GRID),
            "model_refit_count": 0,
            "model_family_hyperparameter_or_feature_selection_after_split_count": 0,
            "logical_data_lanes": LOGICAL_DATA_LANES,
            "model_execution_device": "host_cpu",
            "implementation_sha256": implementation_sha256,
            "freeze_bindings": {
                "expanded": {
                    "aggregate_run_id": EXPANDED_AGGREGATE_RUN_ID,
                    "self_sha256": EXPANDED_FREEZE_SELF_SHA256,
                },
                "behavior": {
                    "aggregate_run_id": BEHAVIOR_AGGREGATE_RUN_ID,
                    "self_sha256": BEHAVIOR_FREEZE_SELF_SHA256,
                },
                "additional": ADDITIONAL_FREEZE_SELF_SHA256,
                "additional_source": ADDITIONAL_SOURCE_SELF_SHA256,
            },
            "expanded": {
                "claims": expanded_results,
                "holm_family_size": 2,
                "qualification": expanded_qualification,
            },
            "behavior": {
                "reports": behavior_reports,
                "inference": behavior_inference,
                "holm": behavior_holm,
                "holm_family_size": 8,
                "qualification": behavior_qualification,
                "artifact_receipts": behavior_identities,
            },
            "additional": {
                "layers": additional_results,
                "holm_family_size": 7,
                "freeze_self_sha256": additional_freeze["self_sha256"],
                "raw_preprojection_illegal_argmax_is_diagnostic_only": True,
                "legality_gate": "zero_served_illegal_mass_and_zero_projection_fallback",
            },
            "candidate_layer_count": len(all_layers),
            "qualified_layer_count": len(qualified_layers),
            "qualified_layers": qualified_layers,
            "held_layers": [layer for layer in all_layers if layer not in qualified_layers],
            "assistant_installation_allowed_for_qualified_layers_after_focused_independent_audit": True,
            "automatic_assistant_installation_performed": False,
            "direct_action_authority": False,
            "live_action_authority": False,
        })
        result_root = (
            stage_drive_root / "runtime_state/current_corpus_locked_tail" / run_id
        )
        result_root.mkdir(parents=True, exist_ok=True)
        manifest_raw = _canonical_json_bytes(manifest) + b"\n"
        manifest_path = result_root / "current-corpus-adjudication-manifest.json"
        create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=run_id)
        returned_raw = _fixed_zip_bytes([
            ("current-corpus-adjudication-manifest.json", manifest_raw),
        ])
        returned_name = (
            "POKERNOW_ASSISTANT_STAGE6_CURRENT_CORPUS_LOCKED_TAIL_RESULT_"
            f"{run_id[:16]}.zip"
        )
        returned_path = stage_drive_root / "returned_results" / returned_name
        returned_path.parent.mkdir(parents=True, exist_ok=True)
        create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
        return {
            "status": manifest["status"],
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": manifest["self_sha256"],
            "qualified_layer_count": len(qualified_layers),
            "qualified_layers": qualified_layers,
            "held_layers": manifest["held_layers"],
            "execution_telemetry": {
                "accepted_accelerator": accelerator,
                "elapsed_seconds": time.monotonic() - started,
                "worker_report": worker_report,
            },
            "returned_result": {
                "path": str(returned_path),
                "name": returned_name,
                "bytes": len(returned_raw),
                "sha256": hashlib.sha256(returned_raw).hexdigest(),
            },
        }
    finally:
        heartbeat.stop()
        line_data = None
        hero_data = None
        gc.collect()
