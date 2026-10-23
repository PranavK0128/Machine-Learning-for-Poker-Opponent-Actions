"""Expanded Stage 6 behavior-model tournament on a frozen temporal split.

Notebook 10 uses this module to train and select behavior challengers without
reading the later holdout targets.  Notebook 11 is the only consumer allowed
to open that target partition.  The implementation intentionally reuses the
existing Stage 6 action schema, visible legal projection, proper-score metric,
and hand-level grouping rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_metrics import multiclass_metric_report
from stage6_notebooks02_07_runtime import (
    ACTIONS,
    BEHAVIOR_FEATURE_COLUMNS,
    CONTEXT_LEVELS,
    LEGAL_COLUMNS,
    LANE_COUNT,
    NOTEBOOK01_RUN_ID,
    NOTEBOOK01_SPLIT_ROOT,
    _context_keys,
    _lane_paths,
    _project_legal,
    canonical_json_bytes,
    materialization_root,
    resolve_workers,
    seal_document,
    sha256_file,
)


TRAIN_END_UTC = "2026-08-01T00:00:00.000Z"
HOLDOUT_START_UTC = "2026-08-16T00:00:00.000Z"
TOURNAMENT_SEED = 610_610
TEMPERATURES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00)
CONTEXT_MIN_SUPPORT_GRID = (25, 50, 100, 250)
CONTEXT_PRIOR_STRENGTH_GRID = (5.0, 25.0, 100.0)
CATEGORICAL_SCHEMAS = {
    "street": ("preflop", "flop", "turn", "river"),
    "board_pair_level": (
        "__MISSING__",
        "unpaired",
        "one_pair",
        "two_pair",
        "trips",
        "quads",
        "full_house_possible",
    ),
}
BOOLEAN_FEATURES = (
    *LEGAL_COLUMNS,
    "short_all_in_call",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "street_last_aggressor_is_actor",
    "facing_bet_or_raise",
    "check_raise_opportunity",
    "facing_cbet",
    "facing_donk_lead",
    "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity",
    "turn_barrel_opportunity",
    "river_barrel_opportunity",
    "preflop_facing_open_raise",
    "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity",
    "preflop_limp_opportunity",
    "preflop_actor_has_limped_before",
    "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before",
)
NUMERIC_FEATURES = (
    "amount_to_call_bb",
    "pot_before_bb",
    "actor_committed_before_bb",
    "actor_stack_before_bb",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
    "active_player_count_before",
    "players_with_chips_before",
    "effective_stack_before_bb",
    "spr_before",
    "board_card_count",
    "board_connectedness_score",
    "street_action_count_before",
    "street_aggressive_action_count_before",
    "street_call_count_before",
    "street_check_count_before",
    "street_fold_count_before",
    "street_prior_actor_action_count",
    "street_prior_actor_bet_raise_count",
    "preflop_limper_count_before",
    "preflop_caller_count_before",
    "preflop_raise_count_before",
    "preflop_prior_entry_count_before",
)
MODEL_SOURCE_COLUMNS = tuple(
    dict.fromkeys((
        "street",
        "board_pair_level",
        *BOOLEAN_FEATURES,
        *NUMERIC_FEATURES,
    ))
)
READ_COLUMNS = tuple(dict.fromkeys((
    "row_id",
    "decision_id",
    "decision_utc",
    "game_id",
    "hand_id",
    "decision_order",
    "action_label",
    *BEHAVIOR_FEATURE_COLUMNS,
    *MODEL_SOURCE_COLUMNS,
)))


def _candidate_registry() -> list[dict[str, Any]]:
    """Return the fixed, bounded, multi-family search registry."""
    registry: list[dict[str, Any]] = []

    lightgbm = (
        ("lgbm_leaf31_regularized", dict(num_leaves=31, max_depth=-1, min_child_samples=100, learning_rate=0.05, n_estimators=5000, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=2.0)),
        ("lgbm_leaf63_depth8", dict(num_leaves=63, max_depth=8, min_child_samples=100, learning_rate=0.05, n_estimators=5000, subsample=0.85, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=5.0)),
        ("lgbm_leaf127_depth10", dict(num_leaves=127, max_depth=10, min_child_samples=150, learning_rate=0.04, n_estimators=5000, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.5, reg_lambda=8.0)),
        ("lgbm_leaf63_conservative", dict(num_leaves=63, max_depth=9, min_child_samples=300, learning_rate=0.035, n_estimators=5000, subsample=0.90, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=12.0)),
        ("lgbm_leaf127_fast", dict(num_leaves=127, max_depth=12, min_child_samples=75, learning_rate=0.08, n_estimators=5000, subsample=0.80, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=4.0)),
    )
    xgboost = (
        ("xgb_depth6", dict(max_depth=6, min_child_weight=5.0, learning_rate=0.06, n_estimators=5000, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=3.0, max_bin=256)),
        ("xgb_depth8", dict(max_depth=8, min_child_weight=10.0, learning_rate=0.05, n_estimators=5000, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=6.0, max_bin=256)),
        ("xgb_depth10_regularized", dict(max_depth=10, min_child_weight=20.0, learning_rate=0.04, n_estimators=5000, subsample=0.80, colsample_bytree=0.80, reg_alpha=0.5, reg_lambda=10.0, max_bin=256)),
        ("xgb_depth5_fast", dict(max_depth=5, min_child_weight=5.0, learning_rate=0.10, n_estimators=5000, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.0, reg_lambda=2.0, max_bin=128)),
    )
    catboost = (
        ("catboost_depth6", dict(depth=6, learning_rate=0.07, iterations=5000, l2_leaf_reg=5.0, random_strength=0.5, bagging_temperature=0.5)),
        ("catboost_depth8", dict(depth=8, learning_rate=0.05, iterations=5000, l2_leaf_reg=8.0, random_strength=0.5, bagging_temperature=1.0)),
        ("catboost_depth10_regularized", dict(depth=10, learning_rate=0.035, iterations=5000, l2_leaf_reg=12.0, random_strength=1.0, bagging_temperature=1.0)),
    )
    hist_gradient = (
        ("histgb_leaf31", dict(max_iter=5000, learning_rate=0.07, max_leaf_nodes=31, max_depth=None, min_samples_leaf=100, l2_regularization=2.0)),
        ("histgb_leaf63", dict(max_iter=5000, learning_rate=0.05, max_leaf_nodes=63, max_depth=None, min_samples_leaf=150, l2_regularization=5.0)),
        ("histgb_depth8", dict(max_iter=5000, learning_rate=0.05, max_leaf_nodes=127, max_depth=8, min_samples_leaf=200, l2_regularization=10.0)),
    )
    gaussian = (
        ("gaussian_nb_1e_11", dict(var_smoothing=1e-11)),
        ("gaussian_nb_1e_9", dict(var_smoothing=1e-9)),
        ("gaussian_nb_1e_7", dict(var_smoothing=1e-7)),
    )
    for family, rows in (
        ("lightgbm", lightgbm),
        ("xgboost", xgboost),
        ("catboost", catboost),
        ("hist_gradient_boosting", hist_gradient),
        ("gaussian_nb", gaussian),
    ):
        for candidate_id, parameters in rows:
            registry.append({
                "candidate_id": candidate_id,
                "family": family,
                "parameters": parameters,
            })
    for support in CONTEXT_MIN_SUPPORT_GRID:
        for prior in CONTEXT_PRIOR_STRENGTH_GRID:
            registry.append({
                "candidate_id": f"context_support{support}_prior{int(prior)}",
                "family": "backoff_context_frequency",
                "parameters": {
                    "minimum_context_support": support,
                    "prior_strength": prior,
                },
            })
    return registry


CANDIDATE_REGISTRY = _candidate_registry()
CANDIDATE_REGISTRY_DIGEST = hashlib.sha256(
    canonical_json_bytes(CANDIDATE_REGISTRY)
).hexdigest()
TOURNAMENT_RUN_ID = hashlib.sha256(canonical_json_bytes({
    "schema": "stage6_expanded_behavior_tournament_v1",
    "notebook01_run_identity": NOTEBOOK01_RUN_ID,
    "split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
    "train_end_utc": TRAIN_END_UTC,
    "holdout_start_utc": HOLDOUT_START_UTC,
    "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
    "seed": TOURNAMENT_SEED,
})).hexdigest()


def _role_for_hand(maximum_utc: str) -> str:
    if maximum_utc < TRAIN_END_UTC:
        return "tournament_train"
    if maximum_utc < HOLDOUT_START_UTC:
        return "tournament_validation"
    return "tournament_final_holdout"


def build_temporal_assignment(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Build a whole-hand temporal assignment without reading any targets."""
    hands: dict[str, dict[str, Any]] = {}
    lane_rows: list[dict[str, Any]] = []
    total_rows = 0
    for lane in range(LANE_COUNT):
        decision_path, _, _ = _lane_paths(root, lane)
        frame = pq.read_table(
            decision_path,
            columns=["row_id", "game_id", "hand_id", "decision_utc"],
        ).to_pandas()
        if frame["row_id"].astype(str).duplicated().any():
            raise RuntimeError(f"lane-{lane:02d} temporal source has duplicate rows")
        lane_rows.append({"lane_id": f"lane-{lane:02d}", "row_count": len(frame)})
        total_rows += len(frame)
        grouped = frame.groupby(["game_id", "hand_id"], sort=False)["decision_utc"].agg(["min", "max", "size"])
        for (game_id, hand_id), row in grouped.iterrows():
            key = f"{game_id}\x1f{hand_id}"
            current = hands.get(key)
            candidate = {
                "game_id": str(game_id),
                "hand_id": str(hand_id),
                "minimum_utc": str(row["min"]),
                "maximum_utc": str(row["max"]),
                "row_count": int(row["size"]),
            }
            if current is None:
                hands[key] = candidate
            else:
                current["minimum_utc"] = min(current["minimum_utc"], candidate["minimum_utc"])
                current["maximum_utc"] = max(current["maximum_utc"], candidate["maximum_utc"])
                current["row_count"] += candidate["row_count"]
        print(json.dumps({
            "event": "stage6_temporal_assignment_lane_scanned",
            "lane": lane,
            "completed": lane + 1,
            "total": LANE_COUNT,
            "rows": len(frame),
        }, sort_keys=True), flush=True)
    if total_rows != 2_333_461:
        raise RuntimeError(f"accepted temporal row total changed: {total_rows}")

    hand_id_to_game: dict[str, str] = {}
    hand_roles: dict[str, str] = {}
    role_rows = {name: 0 for name in (
        "tournament_train", "tournament_validation", "tournament_final_holdout"
    )}
    role_hands = {name: 0 for name in role_rows}
    assignment_rows = []
    for key in sorted(hands):
        row = hands[key]
        hand_id = row["hand_id"]
        prior_game = hand_id_to_game.setdefault(hand_id, row["game_id"])
        if prior_game != row["game_id"]:
            raise RuntimeError("hand identity is not globally unique across games")
        role = _role_for_hand(row["maximum_utc"])
        hand_roles[hand_id] = role
        role_rows[role] += row["row_count"]
        role_hands[role] += 1
        assignment_rows.append({
            "cluster": key,
            "minimum_utc": row["minimum_utc"],
            "maximum_utc": row["maximum_utc"],
            "row_count": row["row_count"],
            "role": role,
        })
    assignment_root = hashlib.sha256(canonical_json_bytes(assignment_rows)).hexdigest()
    boundary_crossing = {
        "train_to_validation": sum(
            row["minimum_utc"] < TRAIN_END_UTC <= row["maximum_utc"]
            for row in hands.values()
        ),
        "validation_to_holdout": sum(
            row["minimum_utc"] < HOLDOUT_START_UTC <= row["maximum_utc"]
            for row in hands.values()
        ),
    }
    manifest = seal_document({
        "schema_version": "stage6_expanded_behavior_temporal_assignment_v1",
        "status": "PASS_PRECOMMITTED_WHOLE_HAND_TEMPORAL_ASSIGNMENT",
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "train_end_utc_exclusive": TRAIN_END_UTC,
        "holdout_start_utc_inclusive": HOLDOUT_START_UTC,
        "whole_hand_assignment_rule": "role_selected_from_maximum_decision_utc_so_boundary_crossing_hands_move_to_later_partition",
        "assignment_root_digest": assignment_root,
        "accepted_row_count": total_rows,
        "hand_count": len(hands),
        "role_row_counts": role_rows,
        "role_hand_counts": role_hands,
        "boundary_crossing_hand_counts": boundary_crossing,
        "lane_scan": lane_rows,
        "target_columns_read_during_assignment": [],
        "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
        "candidate_registry_frozen_before_holdout_target_read": True,
        "holdout_targets_opened": False,
        "promotion_authority": False,
    })
    return hand_roles, manifest


def _strict_boolean_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if not values.map(lambda value: type(value) in {bool, np.bool_}).all():
        raise RuntimeError(f"{column} is not strict boolean data")
    return values.to_numpy(dtype=np.float32)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    parts: list[np.ndarray] = []
    for column, categories in CATEGORICAL_SCHEMAS.items():
        values = frame[column].astype("string").fillna("__MISSING__").astype(str)
        if not values.isin(categories).all():
            unknown = sorted(set(values) - set(categories))
            raise RuntimeError(f"{column} contains unknown values: {unknown[:10]}")
        parts.append(np.column_stack([(values == value).to_numpy(dtype=np.float32) for value in categories]))
    for column in BOOLEAN_FEATURES:
        parts.append(_strict_boolean_column(frame, column).reshape(-1, 1))
    for column in NUMERIC_FEATURES:
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float32)
        invalid = raw.notna().to_numpy() & np.isnan(numeric)
        if invalid.any() or np.isinf(numeric).any():
            raise RuntimeError(f"{column} contains invalid numeric values")
        parts.append(numeric.reshape(-1, 1))
    matrix = np.column_stack(parts).astype(np.float32, copy=False)
    if matrix.shape[1] != len(feature_names()):
        raise RuntimeError("expanded behavior feature width changed")
    return matrix


def feature_names() -> list[str]:
    names: list[str] = []
    for column, categories in CATEGORICAL_SCHEMAS.items():
        names.extend(f"{column}=={value}" for value in categories)
    names.extend(BOOLEAN_FEATURES)
    names.extend(NUMERIC_FEATURES)
    return names


def _street_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    streets = frame["street"].astype(str)
    if scope == "preflop":
        return streets.eq("preflop").to_numpy()
    if scope == "postflop":
        return streets.isin(("flop", "turn", "river")).to_numpy()
    raise RuntimeError(f"unknown behavior scope: {scope}")


def load_selection_data(
    root: Path,
    hand_roles: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Read train/validation targets while excluding every holdout hand."""
    holdout_hands = sorted(
        hand_id for hand_id, role in hand_roles.items()
        if role == "tournament_final_holdout"
    )
    accumulators: dict[str, dict[str, list[Any]]] = {
        scope: {
            "train_x": [], "train_y": [], "validation_x": [], "validation_y": [],
            "validation_legal": [], "validation_clusters": [], "validation_roster": [],
            "train_context_counts": [], "validation_context_keys": [],
        }
        for scope in ("preflop", "postflop")
    }
    loaded_rows = 0
    for lane in range(LANE_COUNT):
        decision_path, _, _ = _lane_paths(root, lane)
        frame = pq.read_table(
            decision_path,
            columns=list(READ_COLUMNS),
            filters=[
                ("decision_utc", "<", HOLDOUT_START_UTC),
                ("hand_id", "not in", holdout_hands),
            ],
        ).to_pandas()
        roles = frame["hand_id"].astype(str).map(hand_roles)
        if roles.isna().any() or roles.eq("tournament_final_holdout").any():
            raise RuntimeError("holdout target entered Notebook 10 selection memory")
        loaded_rows += len(frame)
        for scope in ("preflop", "postflop"):
            scoped = frame.loc[_street_mask(frame, scope)].copy()
            if scoped.empty:
                continue
            scoped_roles = scoped["hand_id"].astype(str).map(hand_roles)
            for role, prefix in (
                ("tournament_train", "train"),
                ("tournament_validation", "validation"),
            ):
                selected = scoped.loc[scoped_roles.eq(role)].copy()
                if selected.empty:
                    continue
                labels = selected["action_label"].astype(str)
                if not labels.isin(ACTIONS).all():
                    raise RuntimeError("selection target is outside the registered action schema")
                matrix = feature_matrix(selected)
                target = np.asarray([ACTIONS.index(value) for value in labels], dtype=np.int8)
                accumulators[scope][f"{prefix}_x"].append(matrix)
                accumulators[scope][f"{prefix}_y"].append(target)
                if role == "tournament_train":
                    keys = _context_keys(selected, scope)
                    lane_counts: dict[str, dict[str, list[int]]] = {}
                    for level in CONTEXT_LEVELS:
                        grouped = (
                            pd.DataFrame({"key": keys[level], "action": labels.to_numpy()})
                            .groupby(["key", "action"], sort=True)
                            .size().unstack(fill_value=0).reindex(columns=ACTIONS, fill_value=0)
                        )
                        lane_counts[level] = {
                            str(key): [int(value) for value in row]
                            for key, row in zip(grouped.index, grouped.to_numpy(dtype=np.int64))
                        }
                    accumulators[scope]["train_context_counts"].append(lane_counts)
                else:
                    accumulators[scope]["validation_legal"].append(
                        selected[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
                    )
                    accumulators[scope]["validation_clusters"].extend(
                        (selected["game_id"].astype(str) + "\x1f" + selected["hand_id"].astype(str)).tolist()
                    )
                    accumulators[scope]["validation_roster"].extend(
                        selected["players_with_chips_before"].astype(int).tolist()
                    )
                    accumulators[scope]["validation_context_keys"].append(
                        _context_keys(selected, scope)
                    )
        print(json.dumps({
            "event": "stage6_tournament_selection_lane_loaded",
            "lane": lane,
            "completed": lane + 1,
            "total": LANE_COUNT,
            "selection_rows_loaded": loaded_rows,
            "holdout_targets_opened": False,
        }, sort_keys=True), flush=True)

    expected = sum(
        1 for role in hand_roles.values() if role != "tournament_final_holdout"
    )
    if loaded_rows <= expected:
        # There are normally multiple decisions per hand.  This guards an
        # accidental hand-only load without binding to quarantine-sensitive rows.
        raise RuntimeError("selection data load is unexpectedly small")
    result: dict[str, dict[str, Any]] = {}
    for scope, values in accumulators.items():
        result[scope] = {
            "train_x": np.concatenate(values["train_x"], axis=0),
            "train_y": np.concatenate(values["train_y"], axis=0),
            "validation_x": np.concatenate(values["validation_x"], axis=0),
            "validation_y": np.concatenate(values["validation_y"], axis=0),
            "validation_legal": np.concatenate(values["validation_legal"], axis=0),
            "validation_clusters": list(values["validation_clusters"]),
            "validation_roster": list(values["validation_roster"]),
            "train_context_counts": list(values["train_context_counts"]),
            "validation_context_keys": {
                level: sum(
                    (list(row[level]) for row in values["validation_context_keys"]),
                    [],
                )
                for level in CONTEXT_LEVELS
            },
        }
        if not len(result[scope]["train_y"]) or not len(result[scope]["validation_y"]):
            raise RuntimeError(f"{scope} temporal selection partition is empty")
    return result


def _fit_imputer(train_x: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(train_x, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    return medians


def _apply_imputer(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float32).copy()
    rows, columns = np.nonzero(~np.isfinite(output))
    if len(rows):
        output[rows, columns] = medians[columns]
    return output


def _fit_sklearn_candidate(
    family: str,
    parameters: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    workers: int,
) -> Any:
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTIONS),
            random_state=TOURNAMENT_SEED,
            n_jobs=workers,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
            verbosity=-1,
            **dict(parameters),
        )
        model.fit(
            train_x, train_y,
            eval_set=[(validation_x, validation_y)],
            callbacks=[lgb.early_stopping(100, verbose=False, min_delta=1e-5)],
        )
        return model
    if family == "xgboost":
        import xgboost as xgb
        from xgboost import XGBClassifier
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(ACTIONS),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=TOURNAMENT_SEED,
            n_jobs=workers,
            callbacks=[xgb.callback.EarlyStopping(
                rounds=100,
                metric_name="mlogloss",
                data_name="validation_0",
                save_best=True,
                min_delta=1e-5,
            )],
            **dict(parameters),
        )
        model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        return model
    if family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=TOURNAMENT_SEED,
            thread_count=workers,
            verbose=False,
            allow_writing_files=False,
            od_type="IncToDec",
            od_pval=1e-4,
            od_wait=100,
            **dict(parameters),
        )
        model.fit(
            train_x, train_y,
            eval_set=(validation_x, validation_y),
            use_best_model=True,
            verbose=False,
        )
        return model
    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=TOURNAMENT_SEED,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=100,
            tol=1e-5,
            **dict(parameters),
        )
        model.fit(train_x, train_y)
        return model
    if family == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**dict(parameters))
        model.fit(train_x, train_y)
        return model
    raise RuntimeError(f"unknown model family: {family}")


def _aligned_predict_proba(model: Any, matrix: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=float)
    classes = [int(value) for value in getattr(model, "classes_", range(raw.shape[1]))]
    if set(classes) != set(range(len(ACTIONS))):
        raise RuntimeError(f"candidate class inventory changed: {classes}")
    index = {value: position for position, value in enumerate(classes)}
    return np.column_stack([raw[:, index[value]] for value in range(len(ACTIONS))])


def _merge_context_counts(
    lane_counts: Sequence[Mapping[str, Mapping[str, Sequence[int]]]],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    class_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    merged: dict[str, dict[str, np.ndarray]] = {level: {} for level in CONTEXT_LEVELS}
    for lane in lane_counts:
        legal_table = lane["legal_street"]
        # The most reduced table covers every training row exactly once.
        for counts in legal_table.values():
            class_counts += np.asarray(counts, dtype=np.int64)
        for level in CONTEXT_LEVELS:
            for key, counts in lane[level].items():
                merged[level].setdefault(str(key), np.zeros(len(ACTIONS), dtype=np.int64))
                merged[level][str(key)] += np.asarray(counts, dtype=np.int64)
    return class_counts, merged


def _context_model(
    class_counts: np.ndarray,
    merged: Mapping[str, Mapping[str, np.ndarray]],
    scope: str,
    minimum_support: int,
    prior_strength: float,
) -> dict[str, Any]:
    priors = (class_counts.astype(float) + 1.0) / (class_counts.sum() + len(ACTIONS))
    tables: dict[str, dict[str, Any]] = {}
    for level in CONTEXT_LEVELS:
        table = {}
        for key in sorted(merged[level]):
            counts = np.asarray(merged[level][key], dtype=np.int64)
            support = int(counts.sum())
            if support >= minimum_support:
                probabilities = (counts + prior_strength * priors) / (support + prior_strength)
                table[key] = {
                    "support": support,
                    "probabilities": probabilities.tolist(),
                }
        tables[level] = table
    return {
        "schema_version": "stage6_expanded_backoff_context_behavior_head_v1",
        "street_scope": scope,
        "label_order": list(ACTIONS),
        "minimum_context_support": minimum_support,
        "prior_strength": prior_strength,
        "class_counts": class_counts.tolist(),
        "class_priors": priors.tolist(),
        "context_levels_most_specific_first": list(CONTEXT_LEVELS),
        "context_tables": tables,
    }


def _predict_context_keys(
    keys: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
) -> np.ndarray:
    row_count = len(keys[CONTEXT_LEVELS[0]])
    priors = np.asarray(model["class_priors"], dtype=float)
    probabilities = np.broadcast_to(priors, (row_count, len(ACTIONS))).copy()
    unresolved = np.ones(row_count, dtype=bool)
    for level in CONTEXT_LEVELS:
        table = model["context_tables"][level]
        for row, key in enumerate(keys[level]):
            if unresolved[row] and key in table:
                probabilities[row] = np.asarray(table[key]["probabilities"], dtype=float)
                unresolved[row] = False
    return probabilities


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    powered = np.power(np.clip(probabilities, 1e-15, 1.0), 1.0 / temperature)
    return powered / powered.sum(axis=1, keepdims=True)


def _mean_log_loss(target: np.ndarray, probabilities: np.ndarray) -> float:
    return -float(np.log(np.clip(
        probabilities[np.arange(len(target)), target.astype(int)], 1e-15, 1.0
    )).mean())


def _evaluate_temperatures(
    raw: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float], int, int]:
    projected, illegal, fallback = _project_legal(raw, legal, priors)
    losses: dict[str, float] = {}
    best = None
    for temperature in TEMPERATURES:
        scaled = _temperature_scale(projected, temperature)
        scaled, _, _ = _project_legal(scaled, legal, priors)
        loss = _mean_log_loss(target, scaled)
        losses[str(temperature)] = loss
        if best is None or (loss, temperature) < (best[0], best[1]):
            best = (loss, temperature, scaled)
    assert best is not None
    return best[2], float(best[1]), losses, illegal, fallback


def _predict_artifact_with_keys(
    artifact: Mapping[str, Any],
    matrix: np.ndarray,
    legal: np.ndarray,
    context_keys: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Replay the exact validation legal-projection/calibration order."""
    kind = str(artifact["artifact_kind"])
    if kind == "single_model":
        family = str(artifact["family"])
        model = artifact["model"]
        raw = (
            _predict_context_keys(context_keys, model)
            if family == "backoff_context_frequency"
            else _aligned_predict_proba(
                model,
                _apply_imputer(
                    matrix,
                    np.asarray(artifact["imputer_medians"], dtype=np.float32),
                ),
            )
        )
        priors = np.asarray(artifact["class_priors"], dtype=float)
        projected, _, _ = _project_legal(raw, legal, priors)
        scaled = _temperature_scale(projected, float(artifact["temperature"]))
        scaled, _, _ = _project_legal(scaled, legal, priors)
        return scaled
    if kind == "weighted_ensemble":
        members = artifact.get("members")
        weights = artifact.get("weights")
        if not isinstance(members, Sequence) or not isinstance(weights, Sequence) or len(members) != len(weights):
            raise RuntimeError("selected ensemble artifact is malformed")
        result = np.zeros((len(matrix), len(ACTIONS)), dtype=float)
        for member, weight in zip(members, weights):
            result += float(weight) * _predict_artifact_with_keys(
                member, matrix, legal, context_keys
            )
        result /= result.sum(axis=1, keepdims=True)
        priors = np.asarray(artifact["class_priors"], dtype=float)
        result, _, _ = _project_legal(result, legal, priors)
        result = _temperature_scale(result, float(artifact["temperature"]))
        result, _, _ = _project_legal(result, legal, priors)
        return result
    raise RuntimeError(f"unknown selected artifact kind: {kind}")


def predict_artifact(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    matrix: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    """Predict with the exact validation-selected legal/calibration order."""
    keys = _context_keys(frame, str(artifact["street_scope"]))
    return _predict_artifact_with_keys(artifact, matrix, legal, keys)


def _save_joblib(path: Path, value: Any) -> dict[str, Any]:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path, compress=3)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _candidate_wrapper(
    entry: Mapping[str, Any],
    scope: str,
    model: Any,
    medians: np.ndarray,
    temperature: float,
    class_priors: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": "stage6_expanded_behavior_candidate_artifact_v1",
        "artifact_kind": "single_model",
        "candidate_id": entry["candidate_id"],
        "family": entry["family"],
        "parameters": dict(entry["parameters"]),
        "street_scope": scope,
        "label_order": list(ACTIONS),
        "feature_names": feature_names(),
        "imputer_medians": medians.tolist(),
        "class_priors": class_priors.tolist(),
        "temperature": temperature,
        "model": model,
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "holdout_targets_used": 0,
    }


def run_scope_tournament(
    scope: str,
    data: Mapping[str, Any],
    workers: int,
    work_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import joblib

    train_x_raw = np.asarray(data["train_x"], dtype=np.float32)
    validation_x_raw = np.asarray(data["validation_x"], dtype=np.float32)
    train_y = np.asarray(data["train_y"], dtype=np.int8)
    validation_y = np.asarray(data["validation_y"], dtype=np.int8)
    legal = np.asarray(data["validation_legal"], dtype=bool)
    medians = _fit_imputer(train_x_raw)
    train_x = _apply_imputer(train_x_raw, medians)
    validation_x = _apply_imputer(validation_x_raw, medians)
    class_counts = np.bincount(train_y.astype(int), minlength=len(ACTIONS))
    priors = (class_counts + 1.0) / (class_counts.sum() + len(ACTIONS))
    context_class_counts, context_merged = _merge_context_counts(data["train_context_counts"])
    if not np.array_equal(context_class_counts, class_counts):
        raise RuntimeError(f"{scope} context and matrix training targets are misaligned")

    artifact_dir = work_root / scope / "candidate_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    started = time.monotonic()
    for index, entry in enumerate(CANDIDATE_REGISTRY, start=1):
        candidate_started = time.monotonic()
        family = str(entry["family"])
        if family == "backoff_context_frequency":
            model = _context_model(
                context_class_counts,
                context_merged,
                scope,
                int(entry["parameters"]["minimum_context_support"]),
                float(entry["parameters"]["prior_strength"]),
            )
            raw = _predict_context_keys(data["validation_context_keys"], model)
        else:
            model = _fit_sklearn_candidate(
                family, entry["parameters"], train_x, train_y,
                validation_x, validation_y, workers,
            )
            raw = _aligned_predict_proba(model, validation_x)
        selected, temperature, temperature_losses, illegal, fallback = _evaluate_temperatures(
            raw, validation_y, legal, priors
        )
        report = {
            "candidate_id": entry["candidate_id"],
            "family": family,
            "parameters": dict(entry["parameters"]),
            "selected_temperature": temperature,
            "temperature_validation_log_losses": temperature_losses,
            "validation_metrics": multiclass_metric_report(
                [ACTIONS[index] for index in validation_y], selected, ACTIONS
            ),
            "illegal_preprojection_argmax_count": illegal,
            "illegal_postprojection_count": 0,
            "zero_legal_mass_fallback_count": fallback,
            "fit_seconds": time.monotonic() - candidate_started,
            "holdout_target_rows_used": 0,
        }
        wrapper = _candidate_wrapper(entry, scope, model, medians, temperature, priors)
        path = artifact_dir / f"{entry['candidate_id']}.joblib"
        _save_joblib(path, wrapper)
        reports.append(report)
        predictions[str(entry["candidate_id"])] = selected.astype(np.float32)
        paths[str(entry["candidate_id"])] = path
        elapsed = time.monotonic() - started
        mean = elapsed / index
        print(json.dumps({
            "event": "stage6_tournament_candidate_complete",
            "scope": scope,
            "candidate": entry["candidate_id"],
            "family": family,
            "completed": index,
            "total": len(CANDIDATE_REGISTRY),
            "validation_log_loss": report["validation_metrics"]["log_loss"],
            "selected_temperature": temperature,
            "elapsed_seconds": elapsed,
            "eta_seconds": mean * (len(CANDIDATE_REGISTRY) - index),
            "holdout_targets_opened": False,
        }, sort_keys=True), flush=True)

    reports.sort(key=lambda row: (
        row["validation_metrics"]["log_loss"], row["candidate_id"]
    ))
    top_ids = [str(row["candidate_id"]) for row in reports[:3]]
    ensemble_specs = (
        ("ensemble_top2_equal", top_ids[:2], (0.5, 0.5)),
        ("ensemble_top2_best75", top_ids[:2], (0.75, 0.25)),
        ("ensemble_top3_equal", top_ids[:3], (1 / 3, 1 / 3, 1 / 3)),
        ("ensemble_top3_best50", top_ids[:3], (0.5, 0.25, 0.25)),
    )
    ensemble_reports = []
    ensemble_predictions: dict[str, np.ndarray] = {}
    for candidate_id, member_ids, weights in ensemble_specs:
        raw = sum(float(weight) * predictions[member] for member, weight in zip(member_ids, weights))
        selected, temperature, temperature_losses, illegal, fallback = _evaluate_temperatures(
            raw, validation_y, legal, priors
        )
        ensemble_predictions[candidate_id] = selected.astype(np.float32)
        ensemble_reports.append({
            "candidate_id": candidate_id,
            "family": "weighted_ensemble",
            "member_candidate_ids": list(member_ids),
            "weights": list(weights),
            "selected_temperature": temperature,
            "temperature_validation_log_losses": temperature_losses,
            "validation_metrics": multiclass_metric_report(
                [ACTIONS[index] for index in validation_y], selected, ACTIONS
            ),
            "illegal_preprojection_argmax_count": illegal,
            "illegal_postprojection_count": 0,
            "zero_legal_mass_fallback_count": fallback,
            "holdout_target_rows_used": 0,
        })
    all_reports = reports + ensemble_reports
    all_reports.sort(key=lambda row: (
        row["validation_metrics"]["log_loss"], row["candidate_id"]
    ))
    selected_report = all_reports[0]
    selected_id = str(selected_report["candidate_id"])
    if selected_report["family"] == "weighted_ensemble":
        member_wrappers = [joblib.load(paths[value]) for value in selected_report["member_candidate_ids"]]
        selected_artifact = {
            "schema_version": "stage6_expanded_behavior_candidate_artifact_v1",
            "artifact_kind": "weighted_ensemble",
            "candidate_id": selected_id,
            "family": "weighted_ensemble",
            "street_scope": scope,
            "label_order": list(ACTIONS),
            "feature_names": feature_names(),
            "class_priors": priors.tolist(),
            "members": member_wrappers,
            "member_candidate_ids": list(selected_report["member_candidate_ids"]),
            "weights": list(selected_report["weights"]),
            "temperature": float(selected_report["selected_temperature"]),
            "tournament_run_id": TOURNAMENT_RUN_ID,
            "holdout_targets_used": 0,
        }
    else:
        selected_artifact = joblib.load(paths[selected_id])
    expected_replay = (
        ensemble_predictions[selected_id]
        if selected_id in ensemble_predictions
        else predictions[selected_id]
    )
    replayed = _predict_artifact_with_keys(
        selected_artifact,
        validation_x_raw,
        legal,
        data["validation_context_keys"],
    )
    replay_max_abs_difference = float(np.max(np.abs(
        replayed - np.asarray(expected_replay, dtype=float)
    )))
    if replay_max_abs_difference > 2e-7:
        raise RuntimeError(
            f"{scope} selected artifact replay differs from validation selection: "
            f"{replay_max_abs_difference}"
        )
    selected_path = work_root / scope / f"selected-{selected_id}.joblib"
    selected_identity = _save_joblib(selected_path, selected_artifact)

    for path in paths.values():
        if path != selected_path and path.exists():
            path.unlink()
    shutil.rmtree(artifact_dir, ignore_errors=True)
    baseline = np.broadcast_to(priors, (len(validation_y), len(ACTIONS))).copy()
    baseline, _, _ = _project_legal(baseline, legal, priors)
    report_document = seal_document({
        "schema_version": "stage6_expanded_behavior_scope_validation_report_v1",
        "scope": scope,
        "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
        "base_candidate_count": len(CANDIDATE_REGISTRY),
        "ensemble_candidate_count": len(ensemble_reports),
        "temperature_values": list(TEMPERATURES),
        "training_row_count": len(train_y),
        "validation_row_count": len(validation_y),
        "class_frequency_validation_metrics": multiclass_metric_report(
            [ACTIONS[index] for index in validation_y], baseline, ACTIONS
        ),
        "ranking": all_reports,
        "selected_candidate_id": selected_id,
        "selected_candidate_family": selected_report["family"],
        "selected_validation_metrics": selected_report["validation_metrics"],
        "selected_artifact": selected_identity,
        "selected_artifact_validation_replay_max_abs_difference": replay_max_abs_difference,
        "selection_metric": "multiclass_log_loss_lower_is_better",
        "legal_projection": "exact_visible_six_action_mask",
        "holdout_target_rows_used": 0,
        "holdout_targets_opened": False,
        "promotion_authority": False,
    })
    freeze = seal_document({
        "schema_version": "stage6_expanded_behavior_scope_candidate_freeze_v1",
        "scope": scope,
        "selected_candidate_id": selected_id,
        "selected_candidate_family": selected_report["family"],
        "selected_artifact": selected_identity,
        "validation_report_self_sha256": report_document["self_sha256"],
        "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "candidate_frozen": True,
        "selection_or_tuning_after_freeze_allowed": False,
        "holdout_targets_opened": False,
        "promotion_authority": False,
    })
    return freeze, report_document, selected_path


def publish_selected_artifacts(
    stage_drive_root: Path,
    temporal_manifest: Mapping[str, Any],
    scope_results: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any], Path]],
) -> dict[str, Any]:
    target_root = stage_drive_root / "runtime_state" / "notebook10" / TOURNAMENT_RUN_ID
    target_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    freezes = {}
    validation_reports = {}
    for scope in ("preflop", "postflop"):
        freeze, report, source = scope_results[scope]
        target = target_root / f"{scope}-selected.joblib"
        shutil.copyfile(source, target)
        identity = {"name": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
        if identity["sha256"] != freeze["selected_artifact"]["sha256"]:
            raise RuntimeError(f"{scope} selected artifact changed during Drive publication")
        artifacts[scope] = identity
        freezes[scope] = dict(freeze)
        validation_reports[scope] = dict(report)
    publication = seal_document({
        "schema_version": "stage6_notebook10_candidate_publication_manifest_v1",
        "status": "PASS_VALIDATION_ONLY_EXPANDED_BEHAVIOR_CANDIDATE_FREEZE",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "temporal_assignment_self_sha256": temporal_manifest["self_sha256"],
        "temporal_assignment_root_digest": temporal_manifest["assignment_root_digest"],
        "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
        "selected_artifacts": artifacts,
        "scope_freezes": {scope: freezes[scope]["self_sha256"] for scope in freezes},
        "scope_validation_reports": {
            scope: validation_reports[scope]["self_sha256"] for scope in validation_reports
        },
        "artifact_root_relative": (
            Path("runtime_state") / "notebook10" / TOURNAMENT_RUN_ID
        ).as_posix(),
        "holdout_targets_opened": False,
        "selection_or_tuning_after_freeze_allowed": False,
        "promotion_authority": False,
    })
    manifest_path = target_root / "candidate-publication-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(publication) + b"\n")
    if sha256_file(manifest_path) != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise RuntimeError("Notebook 10 publication manifest write verification failed")
    return {
        "publication": publication,
        "freezes": freezes,
        "validation_reports": validation_reports,
        "target_root": target_root,
    }


def load_holdout_frame(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Notebook 11 only: open the frozen temporal holdout after its claim."""
    hand_roles, temporal = build_temporal_assignment(root)
    holdout_hands = sorted(
        hand_id for hand_id, role in hand_roles.items()
        if role == "tournament_final_holdout"
    )
    frames = []
    for lane in range(LANE_COUNT):
        decision_path, _, _ = _lane_paths(root, lane)
        frame = pq.read_table(
            decision_path,
            columns=list(READ_COLUMNS),
            filters=[("hand_id", "in", holdout_hands)],
        ).to_pandas()
        roles = frame["hand_id"].astype(str).map(hand_roles)
        if not roles.eq("tournament_final_holdout").all():
            raise RuntimeError("Notebook 11 holdout load crossed its frozen assignment")
        frames.append(frame)
        print(json.dumps({
            "event": "stage6_tournament_holdout_lane_opened",
            "lane": lane,
            "completed": lane + 1,
            "total": LANE_COUNT,
            "rows": len(frame),
        }, sort_keys=True), flush=True)
    result = pd.concat(frames, ignore_index=True)
    expected = temporal["role_row_counts"]["tournament_final_holdout"]
    if len(result) != expected or result["row_id"].astype(str).duplicated().any():
        raise RuntimeError("Notebook 11 holdout row identity is incomplete or duplicated")
    return result, temporal


__all__ = [
    "CANDIDATE_REGISTRY",
    "CANDIDATE_REGISTRY_DIGEST",
    "HOLDOUT_START_UTC",
    "MODEL_SOURCE_COLUMNS",
    "READ_COLUMNS",
    "TOURNAMENT_RUN_ID",
    "TRAIN_END_UTC",
    "build_temporal_assignment",
    "feature_matrix",
    "load_holdout_frame",
    "load_selection_data",
    "predict_artifact",
    "publish_selected_artifacts",
    "run_scope_tournament",
]
