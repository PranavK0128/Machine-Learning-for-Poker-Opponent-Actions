"""Execute one leakage-safe retrospective nested Stage 6 layer/fold search.

Each invocation evaluates one frozen 256-candidate registry against one frozen
outer fold. Candidate, hyperparameter, and iteration selection use only the
inner train/validation partitions. Scalar temperature uses only calibration.
The outer test partition is scored once after every choice is fixed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
LOGICAL_DATA_LANES = 48
FOLD_RESULT_NAME = (
    "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_FOLD_FREEZE_RESULT_"
    "6975842b78151725.zip"
)
FOLD_RESULT_BYTES = 15_003
FOLD_RESULT_SHA256 = (
    "229f636e8804ca57ec6edf0e8628c64ea40aa05a402c5d83d533f229f6a3e457"
)
FOLD_RUN_ID = (
    "6975842b78151725c4508072f89b30ca91e698dfd4f231a5e3e3da4f83bb798b"
)
FOLD_ASSIGNMENTS_SELF_SHA256 = (
    "c5c884cf058df70f40442a0ff4cabc84bfb58f6d7be175c2a07b8c33f0b0544e"
)
FOLD_MANIFEST_SELF_SHA256 = (
    "d9acc15a643ea3b94193c7a96d5f2a33eec18b2008850b3f508ebefa8dbf9ac8"
)

EXPANDED_LAYERS = (
    "hero_opponent_interaction",
    "within_hand_line_surprise",
)
BEHAVIOR_LAYERS = ("preflop_behavior", "postflop_behavior")
ADDITIONAL_LAYERS = (
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "opponent_bluff_likelihood",
    "player_strategy_drift",
    "showdown_aware_range_calibration",
    "supported_fold_response_bluff_opportunity",
)
LAYER_FAMILIES = {
    **{layer: "expanded" for layer in EXPANDED_LAYERS},
    **{layer: "behavior" for layer in BEHAVIOR_LAYERS},
    **{layer: "additional" for layer in ADDITIONAL_LAYERS},
}
NESTED_STAGES = (
    {
        "stage": 1,
        "train_row_cap": 60_000,
        "validation_row_cap": 20_000,
        "iteration_cap": 600,
        "patience": 50,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 96,
    },
    {
        "stage": 2,
        "train_row_cap": 180_000,
        "validation_row_cap": 60_000,
        "iteration_cap": 2_000,
        "patience": 100,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 32,
    },
    {
        "stage": 3,
        "train_row_cap": 500_000,
        "validation_row_cap": 150_000,
        "iteration_cap": 6_000,
        "patience": 150,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 32,
    },
)
EXACT_STAGE_CANDIDATE_COUNTS = (256, 96, 32)
EXACT_CANDIDATE_EVALUATIONS_PER_FOLD = sum(EXACT_STAGE_CANDIDATE_COUNTS)
TEMPERATURE_GRID = (0.50, 0.67, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00)
MAX_CANDIDATE_ATTEMPTS = 2
HEARTBEAT_INTERVAL_SECONDS = 60.0
RETROSPECTIVE_NESTED_SCIENCE_COMPATIBILITY_SHA256 = (
    "0d60299b8c47a2bd73942d79affcd1b0899d8248fb54dead3d8819a9b9f0b634"
)
# These are the exact R2 work roots that already contain durable checkpoints for
# the six interrupted searches.  R3 keeps using the same storage roots while
# binding the scientific run independently of Colab's CPU/GPU/TPU platform
# fingerprint.  No other historical run root is eligible for this bridge.
R2_RUNTIME_RESUME_STORAGE_RUN_IDS = {
    ("outer-fold-04", "hero_opponent_interaction"):
        "1b9a573b94fb2b4cb316ae61fbed5770a4110072a775302a68db55f84babdeb5",
    ("outer-fold-04", "within_hand_line_surprise"):
        "5988f17025b6a5aafa28bd522889b4347e8fb7af05da64fd5cdc135070b05b73",
    ("outer-fold-02", "preflop_behavior"):
        "a80633138ec9bcb465180198e1318979b90c5ba2ead3fe768bb3c7dcfdbea8ec",
    ("outer-fold-00", "joint_action_size_surprise"):
        "125766d8658688d4a494e1121358e68b45218a4bd4bf0b6aa6ad18c79f9dfa63",
    ("outer-fold-00", "multiway_target_response"):
        "dff819aacd8c1690277c2bff25cd1cf1f02790ebcc23506e55d9f2053a5d300a",
    ("outer-fold-03", "multiway_target_response"):
        "6bc43e10675f259838dd212eacdeced19a3d06de1583461fc2d521c4e5225f86",
}
RUNTIME_PACKAGE_BINDINGS = {
    "POKERNOW_ASSISTANT_STAGE6_BEHAVIOR_CHALLENGER_V2_SCIENCE_R14.zip": {
        "bytes": 386_673,
        "sha256": "624f8a4623307b49eee4a0e2331df33485709c94a3aad2c5a731ee894a633971",
    },
    "POKERNOW_ASSISTANT_STAGE6_ADDITIONAL_EXPLOITATION_RELEASE_R21.zip": {
        "bytes": 79_218,
        "sha256": "c4e74aa7842d58fdd5b127be9dffff0c5b021a1b843ca7ef92cebd3dcf9c4888",
    },
}


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
    if not expected or expected != observed:
        raise RuntimeError("sealed nested-search document self SHA-256 mismatch")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _fixed_npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    """Serialize NumPy arrays with fixed ZIP metadata for exact restart bytes."""
    members: list[tuple[str, bytes]] = []
    for name, value in arrays.items():
        stream = io.BytesIO()
        np.save(stream, np.asarray(value), allow_pickle=False)
        members.append((f"{name}.npy", stream.getvalue()))
    return _fixed_zip_bytes(members)


def _validate_requested_workers(value: str | int) -> str:
    raw = str(value).strip()
    if raw.lower() == "auto" or raw in {"", "0"}:
        return "auto"
    if raw.isascii() and raw.isdecimal() and int(raw) > 0:
        return raw
    raise RuntimeError(
        "REQUESTED_WORKERS must be auto, blank, 0, or a positive integer"
    )


def _resolve_candidate_workers(
    requested: str, unfinished_candidates: int
) -> dict[str, Any]:
    import psutil

    detected_logical_cpus = max(1, int(os.cpu_count() or 1))
    if requested == "auto":
        requested_capacity = detected_logical_cpus
        mode = "auto"
    else:
        requested_capacity = int(requested)
        mode = "explicit"
    live_workers = max(
        1, min(requested_capacity, int(unfinished_candidates))
    )
    return {
        "mode": mode,
        "requested_workers": requested,
        "detected_logical_cpus": detected_logical_cpus,
        "unfinished_candidate_count": int(unfinished_candidates),
        "live_candidate_workers": live_workers,
        "available_system_ram_bytes_telemetry_only": int(
            psutil.virtual_memory().available
        ),
        "ram_used_as_worker_cap": False,
        "preemptive_worker_reduction_applied": False,
        "worker_reduction_allowed_only_after_confirmed_oom": True,
    }


def _bind_feature_release_v2() -> None:
    import stage6_expanded_feature_freeze_v2 as feature_freeze_v2

    existing = sys.modules.get("stage6_expanded_feature_freeze")
    if existing not in (None, feature_freeze_v2):
        raise RuntimeError("R5 feature release must bind before historical consumers")
    sys.modules["stage6_expanded_feature_freeze"] = feature_freeze_v2


def load_fold_freeze(stage_drive_root: str | Path) -> dict[str, Any]:
    path = Path(stage_drive_root).resolve() / "returned_results" / FOLD_RESULT_NAME
    if (
        not path.is_file()
        or path.stat().st_size != FOLD_RESULT_BYTES
        or sha256_file(path) != FOLD_RESULT_SHA256
    ):
        raise RuntimeError("exact Notebook 29 returned fold result is missing")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected_names = [
            "nested-fold-freeze-manifest.json",
            "nested-fold-assignments.json",
        ]
        if names != expected_names or len(names) != len(set(names)):
            raise RuntimeError("Notebook 29 returned fold inventory changed")
        manifest = json.loads(archive.read(expected_names[0]))
        assignments = json.loads(archive.read(expected_names[1]))
    validate_sealed_document(manifest)
    validate_sealed_document(assignments)
    if (
        manifest.get("self_sha256") != FOLD_MANIFEST_SELF_SHA256
        or assignments.get("self_sha256") != FOLD_ASSIGNMENTS_SELF_SHA256
        or manifest.get("status") != "PASS_RETROSPECTIVE_NESTED_FOLD_FREEZE"
        or manifest.get("outer_fold_count") != 5
        or manifest.get("outer_test_game_count") != 387
        or len(assignments.get("folds", [])) != 5
        or assignments.get("candidate_registry_bindings", {}).get(
            "action_timing_layer_included"
        )
        is not False
    ):
        raise RuntimeError("Notebook 29 fold authority changed")
    return {
        "path": path,
        "manifest": manifest,
        "assignments": assignments,
    }


def _registry(layer: str) -> list[dict[str, Any]]:
    family = LAYER_FAMILIES.get(layer)
    if family == "expanded":
        from stage6_expanded_layers import candidate_registry

        rows = candidate_registry(layer)
    elif family == "behavior":
        from stage6_behavior_challenger_tournament_v2 import candidate_registry

        rows = candidate_registry(layer)
    elif family == "additional":
        from stage6_additional_exploitation_layers import candidate_registry

        rows = candidate_registry(layer)
    else:
        raise ValueError(f"unknown retrospective nested layer: {layer}")
    if len(rows) != 256 or len({row["candidate_id"] for row in rows}) != 256:
        raise RuntimeError(f"{layer} registry is not the exact frozen 256 candidates")
    return rows


def _registry_binding(
    layer: str, rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, Any]
) -> dict[str, Any]:
    family = LAYER_FAMILIES[layer]
    expected = assignments["candidate_registry_bindings"][family][layer]
    if family == "behavior":
        from stage6_behavior_challenger_tournament_v2 import registry_digest

        registry_sha256 = registry_digest(layer)
    elif family == "additional":
        from stage6_additional_exploitation_layers import registry_digest

        registry_sha256 = registry_digest(layer)
    else:
        registry_sha256 = hashlib.sha256(
            canonical_json_bytes(list(rows))
        ).hexdigest()
    observed = {
        "candidate_count": len(rows),
        "registry_sha256": registry_sha256,
    }
    if observed != expected:
        raise RuntimeError(f"{layer} candidate registry changed after fold freeze")
    return observed


def _load_common_identity_frame(stage_drive_root: Path) -> tuple[pd.DataFrame, str]:
    from stage6_expanded_tournament import _load_feature_manifest, _verify_lane_identity

    root, manifest = _load_feature_manifest(stage_drive_root)
    frames: list[pd.DataFrame] = []
    columns = ["row_id", "game_id", "hand_id", "decision_utc", "temporal_role"]
    for lane_number, lane in enumerate(manifest["lanes"]):
        if lane.get("lane_id") != f"lane-{lane_number:02d}":
            raise RuntimeError("common identity lane order changed")
        path = _verify_lane_identity(root, lane["common"])
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame)
        print(json.dumps({
            "event": "stage6_nested_identity_lane_loaded",
            "lane": lane_number,
            "completed": lane_number + 1,
            "denominator": LOGICAL_DATA_LANES,
            "row_count": len(frame),
        }, sort_keys=True), flush=True)
    identity = pd.concat(frames, ignore_index=True)
    if identity["row_id"].astype(str).duplicated().any():
        raise RuntimeError("common identity frame duplicates row_id")
    return identity, str(manifest["self_sha256"])


def _load_dataset(stage_drive_root: Path, layer: str) -> dict[str, Any]:
    family = LAYER_FAMILIES[layer]
    if family == "expanded":
        from stage6_expanded_tournament import load_layer_dataset

        dataset = load_layer_dataset(stage_drive_root, layer)
        dataset["source_manifest_self_sha256"] = dataset[
            "feature_manifest_self_sha256"
        ]
        return dataset
    if family == "behavior":
        from stage6_behavior_challenger_notebook_runtime_v2 import (
            load_behavior_dataset,
        )

        dataset = load_behavior_dataset(stage_drive_root, layer)
        dataset["source_manifest_self_sha256"] = dataset[
            "feature_manifest_self_sha256"
        ]
        return dataset

    from stage6_retrospective_nested_additional_source import (
        load_retrospective_nested_additional_layer,
    )

    cache_parent = Path("/content")
    if not cache_parent.is_dir():
        cache_parent = Path(tempfile.gettempdir())
    source = load_retrospective_nested_additional_layer(
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "additional_raw_source",
        layer,
        local_cache_root=(
            cache_parent / "stage6_nested_additional_raw_source_cache"
        ),
    )
    identity, feature_manifest_self_sha256 = _load_common_identity_frame(
        stage_drive_root
    )
    row_frame = pd.DataFrame({"row_id": source["row_ids"]})
    frame = row_frame.merge(identity, on="row_id", how="left", validate="one_to_one")
    if frame.isna().any().any() or len(frame) != len(row_frame):
        raise RuntimeError(f"{layer} additional row/game identity join is incomplete")
    return {
        "frame": frame,
        "matrix": np.asarray(source["x_raw"], dtype=np.float32),
        "target": np.asarray(source["y"], dtype=np.int8),
        "legal": np.asarray(source["legal"], dtype=bool),
        "feature_names": list(source["feature_names"]),
        "source_manifest_self_sha256": str(source["manifest"]["self_sha256"]),
        "feature_manifest_self_sha256": feature_manifest_self_sha256,
        "whole_hand_temporal_assignment_binding": None,
        "raw_nan_preserved_for_fold_local_imputation": True,
    }


def _fold(layer_dataset: Mapping[str, Any], assignment: Mapping[str, Any]) -> dict[str, Any]:
    frame = layer_dataset["frame"].copy()
    role_by_game: dict[str, str] = {}
    for role, games in assignment["partitions"].items():
        for game in games:
            if game in role_by_game:
                raise RuntimeError("nested fold assigns one game to multiple roles")
            role_by_game[str(game)] = str(role)
    roles = frame["game_id"].astype(str).map(role_by_game)
    mask = roles.notna().to_numpy()
    frame = frame.loc[mask].reset_index(drop=True)
    frame["nested_role"] = roles.loc[mask].astype(str).to_numpy()
    observed = set(frame["nested_role"])
    if observed != {"train", "validation", "calibration", "test"}:
        raise RuntimeError(f"layer fold has incomplete roles: {sorted(observed)}")
    hand_roles = frame.groupby(["game_id", "hand_id"], sort=False)[
        "nested_role"
    ].nunique()
    if (hand_roles != 1).any():
        raise RuntimeError("one whole hand crosses nested roles")
    target = np.asarray(layer_dataset["target"])[mask]
    legal = np.asarray(layer_dataset["legal"])[mask]
    if not legal[np.arange(len(target)), target.astype(int)].all():
        raise RuntimeError("observed nested target is illegal")
    result = {
        **dict(layer_dataset),
        "frame": frame,
        "matrix": np.asarray(layer_dataset["matrix"])[mask],
        "target": target,
        "legal": legal,
    }
    if "feature_frame" in layer_dataset:
        result["feature_frame"] = layer_dataset["feature_frame"].loc[
            mask
        ].reset_index(drop=True)
    for role in ("train", "validation", "calibration", "test"):
        role_games = set(frame.loc[frame["nested_role"].eq(role), "game_id"].astype(str))
        assigned_games = set(
            str(value) for value in assignment["partitions"][role]
        )
        if not role_games or not role_games.issubset(assigned_games):
            raise RuntimeError(
                f"{role} layer rows are empty or escape their assigned games"
            )
    return result


class _NativeSixActionBooster:
    def __init__(
        self,
        booster: Any,
        family: str,
        *,
        best_iteration: int,
    ) -> None:
        self.booster = booster
        self.family = family
        self.classes_ = np.arange(len(ACTIONS), dtype=np.int8)
        self.best_iteration = max(0, int(best_iteration) - 1)
        self.best_iteration_ = max(1, int(best_iteration))

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.family == "xgboost":
            import xgboost as xgb

            values = self.booster.predict(xgb.DMatrix(matrix))
        else:
            values = self.booster.predict(matrix)
        return np.asarray(values, dtype=np.float64).reshape(
            len(matrix), len(ACTIONS)
        )


def _fit_candidate_allowing_action_subsets(
    entry: Mapping[str, Any], paths: Mapping[str, str], stage: Mapping[str, Any]
) -> Any:
    from stage6_expanded_tournament import (
        _CatBoostMeaningfulImprovementStopper,
        _fit_candidate,
        _load_array,
    )

    family = str(entry["family"])
    if family not in {"lightgbm", "xgboost", "catboost"}:
        return _fit_candidate(entry, paths, stage)
    parameters = dict(entry["parameters"])
    train_x = _load_array(paths, "train_x")
    train_y = np.asarray(_load_array(paths, "train_y"), dtype=np.int8)
    validation_x = _load_array(paths, "validation_x")
    validation_y = np.asarray(
        _load_array(paths, "validation_y"), dtype=np.int8
    )
    iteration_cap = int(stage["iteration_cap"])
    patience = int(stage["patience"])
    tolerance = float(stage["minimum_meaningful_improvement"])
    seed = int(
        hashlib.sha256(str(entry["candidate_id"]).encode()).hexdigest()[:8],
        16,
    )
    if family == "xgboost":
        import xgboost as xgb

        booster = xgb.train(
            {
                "objective": "multi:softprob",
                "num_class": len(ACTIONS),
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "nthread": 1,
                "seed": seed,
                **parameters,
            },
            xgb.DMatrix(train_x, label=train_y),
            num_boost_round=iteration_cap,
            evals=[(
                xgb.DMatrix(validation_x, label=validation_y),
                "validation_0",
            )],
            callbacks=[xgb.callback.EarlyStopping(
                rounds=patience,
                metric_name="mlogloss",
                data_name="validation_0",
                save_best=True,
                min_delta=tolerance,
            )],
            verbose_eval=False,
        )
        best = int(getattr(booster, "best_iteration", iteration_cap - 1)) + 1
        return _NativeSixActionBooster(
            booster, family, best_iteration=best
        )
    if family == "lightgbm":
        import lightgbm as lgb

        booster = lgb.train(
            {
                "objective": "multiclass",
                "num_class": len(ACTIONS),
                "verbosity": -1,
                "num_threads": 1,
                "deterministic": True,
                "force_col_wise": True,
                "subsample_freq": 1,
                "seed": seed,
                **parameters,
            },
            lgb.Dataset(train_x, label=train_y, free_raw_data=False),
            num_boost_round=iteration_cap,
            valid_sets=[
                lgb.Dataset(
                    validation_x, label=validation_y, free_raw_data=False
                )
            ],
            valid_names=["validation_0"],
            callbacks=[lgb.early_stopping(
                patience, verbose=False, min_delta=tolerance
            )],
        )
        best = int(booster.best_iteration or iteration_cap)
        return _NativeSixActionBooster(
            booster, family, best_iteration=best
        )
    meaningful_stopper = _CatBoostMeaningfulImprovementStopper(
        patience=patience,
        minimum_improvement=tolerance,
    )
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        classes_count=len(ACTIONS),
        iterations=iteration_cap,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **parameters,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=(validation_x, validation_y),
        use_best_model=False,
        verbose=False,
        callbacks=[meaningful_stopper],
    )
    if meaningful_stopper.best_tree_count < 1:
        raise RuntimeError("CatBoost nested stopper selected no trees")
    if int(model.tree_count_) > meaningful_stopper.best_tree_count:
        model.shrink(ntree_end=meaningful_stopper.best_tree_count)
    return model


def _aligned_six_action_probabilities(
    model: Any, matrix: np.ndarray
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    classes = [
        int(value)
        for value in getattr(estimator, "classes_", range(raw.shape[1]))
    ]
    if (
        len(classes) != raw.shape[1]
        or len(classes) != len(set(classes))
        or not set(classes).issubset(set(range(len(ACTIONS))))
    ):
        raise RuntimeError(f"candidate class inventory is invalid: {classes}")
    aligned = np.zeros((len(raw), len(ACTIONS)), dtype=np.float64)
    for source_column, action_index in enumerate(classes):
        aligned[:, action_index] = raw[:, source_column]
    return aligned


def _fit_fixed_candidate_allowing_action_subsets(
    entry: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    iterations: int,
) -> Any:
    family = str(entry["family"])
    if family not in {"lightgbm", "xgboost"}:
        from stage6_expanded_finalist import _fit_fixed_candidate

        return _fit_fixed_candidate(entry, train_x, train_y, iterations)
    seed = int(
        hashlib.sha256(str(entry["candidate_id"]).encode()).hexdigest()[:8],
        16,
    )
    if family == "lightgbm":
        import lightgbm as lgb

        booster = lgb.train(
            {
                "objective": "multiclass",
                "num_class": len(ACTIONS),
                "verbosity": -1,
                "num_threads": 1,
                "deterministic": True,
                "force_col_wise": True,
                "subsample_freq": 1,
                "seed": seed,
                **dict(entry["parameters"]),
            },
            lgb.Dataset(train_x, label=train_y, free_raw_data=False),
            num_boost_round=max(1, int(iterations)),
        )
        return _NativeSixActionBooster(
            booster, "lightgbm", best_iteration=max(1, int(iterations))
        )
    import xgboost as xgb

    booster = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": len(ACTIONS),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "nthread": 1,
            "seed": seed,
            **dict(entry["parameters"]),
        },
        xgb.DMatrix(train_x, label=train_y),
        num_boost_round=max(1, int(iterations)),
        verbose_eval=False,
    )
    return _NativeSixActionBooster(
        booster, "xgboost", best_iteration=max(1, int(iterations))
    )


def _best_iteration(model: Any, family: str, stage: Mapping[str, Any]) -> int:
    if family == "lightgbm":
        return int(model.best_iteration_ or stage["iteration_cap"])
    if family == "xgboost":
        value = getattr(model, "best_iteration", None)
        return int(value) + 1 if value is not None else int(stage["iteration_cap"])
    if family == "catboost":
        return int(model.tree_count_)
    if family == "hist_gradient_boosting":
        return int(model.n_iter_)
    if family in {"extra_trees", "random_forest"}:
        return int(model.n_estimators)
    if family == "logistic_regression":
        return int(np.asarray(model.steps[-1][1].n_iter_).max())
    return 1


def _evaluate_uncalibrated_candidate(
    entry: Mapping[str, Any], paths: Mapping[str, str], stage: Mapping[str, Any]
) -> dict[str, Any]:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_info, threadpool_limits
    from stage6_expanded_tournament import (
        _load_array,
        _project_legal,
    )
    from stage6_metrics import multiclass_metric_report

    with threadpool_limits(limits=1):
        model = _fit_candidate_allowing_action_subsets(entry, paths, stage)
        validation_x = _load_array(paths, "validation_x")
        validation_y = np.asarray(
            _load_array(paths, "validation_y"), dtype=np.int8
        )
        legal = np.asarray(_load_array(paths, "validation_legal"), dtype=bool)
        train_y = np.asarray(_load_array(paths, "train_y"), dtype=np.int8)
        counts = np.bincount(train_y, minlength=len(ACTIONS)).astype(float)
        priors = (counts + 1.0) / (counts.sum() + len(ACTIONS))
        raw = _aligned_six_action_probabilities(model, validation_x)
        projected, raw_illegal, fallback = _project_legal(raw, legal, priors)
        if fallback:
            raise RuntimeError(
                "candidate produced invalid validation probabilities"
            )
        metrics = multiclass_metric_report(
            [ACTIONS[int(value)] for value in validation_y], projected, ACTIONS
        )
        observed_max_threads = max(
            (int(row.get("num_threads", 0)) for row in threadpool_info()),
            default=1,
        )
        if observed_max_threads > 1:
            raise RuntimeError("candidate native thread pool exceeded one thread")
    result = {
        "status": "PASS_VALIDATION_PROBABILITIES",
        "rank_eligible": True,
        "candidate_id": entry["candidate_id"],
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "parameters": entry["parameters"],
        "stage": int(stage["stage"]),
        "train_row_count": len(train_y),
        "validation_row_count": len(validation_y),
        "validation_temperature": 1.0,
        "validation_temperature_selected": False,
        "validation_metrics": metrics,
        "raw_preprojection_illegal_argmax_count": int(raw_illegal),
        "legal_projection_fallback_count": int(fallback),
        "best_iteration": _best_iteration(model, str(entry["family"]), stage),
        "candidate_internal_threads": 1,
        "observed_native_threadpool_max_threads": observed_max_threads,
        "threadpoolctl_limit_applied": True,
    }
    del raw, projected, model
    gc.collect()
    return result


def _candidate_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_uncalibrated_candidate(
        job["entry"], job["paths"], job["stage"]
    )


class _Progress:
    def __init__(
        self,
        *,
        layer: str,
        fold_id: str,
        stage: int,
        completed: int,
        denominator: int,
        live_workers: int,
        checkpoint: str,
    ) -> None:
        self.layer = layer
        self.fold_id = fold_id
        self.stage = int(stage)
        self.completed = int(completed)
        self.denominator = int(denominator)
        self.live_workers = int(live_workers)
        self.checkpoint = checkpoint
        self.started = time.monotonic()
        self.last_completed_at = self.started
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def update(self, completed: int) -> None:
        with self._lock:
            if int(completed) > self.completed:
                self.completed = int(completed)
                self.last_completed_at = time.monotonic()
        self.emit()

    def payload(self) -> dict[str, Any]:
        with self._lock:
            completed = self.completed
            last = self.last_completed_at
        elapsed = max(0.0, time.monotonic() - self.started)
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (self.denominator - completed) / rate if rate > 0 else None
        return {
            "event": "stage6_retrospective_nested_search_heartbeat",
            "phase": f"successive_halving_stage_{self.stage}",
            "layer": self.layer,
            "fold_id": self.fold_id,
            "completed": completed,
            "denominator": self.denominator,
            "live_workers": self.live_workers,
            "throughput_candidates_per_second": rate,
            "eta_seconds": eta,
            "seconds_since_last_candidate": max(0.0, time.monotonic() - last),
            "checkpoint": self.checkpoint,
        }

    def emit(self) -> None:
        print(json.dumps(self.payload(), sort_keys=True), flush=True)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            self.emit()

    def __enter__(self) -> "_Progress":
        self.emit()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.emit()


def _confirmed_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, MemoryError) or "out of memory" in text or (
        "cannot allocate memory" in text
    )


INVALID_VALIDATION_PROBABILITY_ERROR = (
    "candidate produced invalid validation probabilities"
)


def _is_invalid_validation_probability_error(error: BaseException) -> bool:
    return (
        type(error) is RuntimeError
        and str(error) == INVALID_VALIDATION_PROBABILITY_ERROR
    )


def _rejected_invalid_probability_result(
    entry: Mapping[str, Any], stage: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal a deterministic non-survivor without fabricating model scores."""
    return {
        "status": "REJECTED_INVALID_VALIDATION_PROBABILITIES",
        "rank_eligible": False,
        "candidate_id": entry["candidate_id"],
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "parameters": entry["parameters"],
        "stage": int(stage["stage"]),
        "validation_temperature": 1.0,
        "validation_temperature_selected": False,
        "candidate_internal_threads": 1,
        "rejection_reason": INVALID_VALIDATION_PROBABILITY_ERROR,
        "qualification_authority": False,
    }


def _rank(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = [
        row
        for row in reports
        if row.get("result", {}).get("rank_eligible", True) is not False
    ]
    if not eligible:
        raise RuntimeError("nested stage has no rank-eligible candidates")
    return sorted(
        eligible,
        key=lambda row: (
            float(row["result"]["validation_metrics"]["log_loss"]),
            float(row["result"]["validation_metrics"]["multiclass_brier_score"]),
            str(row["candidate_id"]),
        ),
    )


def _checkpoint_inventory(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    inventory = sorted(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "self_sha256": str(row["self_sha256"]),
            }
            for row in reports
        ],
        key=lambda row: row["candidate_id"],
    )
    if len(inventory) != len({row["candidate_id"] for row in inventory}):
        raise RuntimeError("nested checkpoint inventory duplicates a candidate")
    return inventory, hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _assert_exact_evaluation_contract(
    stage_summaries: Sequence[Mapping[str, Any]],
) -> int:
    candidate_counts = tuple(
        int(row.get("candidate_count", -1)) for row in stage_summaries
    )
    if candidate_counts != EXACT_STAGE_CANDIDATE_COUNTS:
        raise RuntimeError(
            "nested stage candidate counts changed: "
            f"{candidate_counts} != {EXACT_STAGE_CANDIDATE_COUNTS}"
        )
    for index, required in enumerate(EXACT_STAGE_CANDIDATE_COUNTS[1:]):
        observed = int(stage_summaries[index].get("survivor_count", -1))
        if observed != required:
            raise RuntimeError(
                "nested stage survivor shortage changed the 384-evaluation "
                f"contract: stage {index + 1} retained {observed}, "
                f"required {required}"
            )
    return EXACT_CANDIDATE_EVALUATIONS_PER_FOLD


def _validate_checkpoint_inventories(
    work_root: Path, stage_summaries: Sequence[Mapping[str, Any]]
) -> None:
    for summary in stage_summaries:
        stage = int(summary["stage"])
        expected = list(summary["checkpoint_inventory"])
        observed = []
        for identity in expected:
            path = _checkpoint_path(
                work_root, stage, str(identity["candidate_id"])
            )
            from stage6_drive_fuse_publication import recover_claimed_bytes

            recover_claimed_bytes(path)
            if not path.is_file():
                raise RuntimeError(f"completed checkpoint is missing: {path}")
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_sealed_document(document)
            if document.get("self_sha256") != identity["self_sha256"]:
                raise RuntimeError(f"completed checkpoint identity changed: {path}")
            observed.append({
                "candidate_id": str(document["candidate_id"]),
                "self_sha256": str(document["self_sha256"]),
            })
        observed.sort(key=lambda row: row["candidate_id"])
        if (
            observed != expected
            or hashlib.sha256(canonical_json_bytes(observed)).hexdigest()
            != summary["checkpoint_inventory_root_sha256"]
        ):
            raise RuntimeError("completed checkpoint inventory root changed")


def _checkpoint_path(root: Path, stage: int, candidate_id: str) -> Path:
    return root / "candidate_results" / f"stage-{stage}" / f"{candidate_id}.json"


def _load_checkpoint(
    path: Path,
    *,
    run_id: str,
    stage: int,
    entry: Mapping[str, Any],
    data_binding_self_sha256: str,
    compatible_run_ids: Sequence[str] = (),
) -> dict[str, Any] | None:
    from stage6_drive_fuse_publication import recover_claimed_bytes

    recover_claimed_bytes(path)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_sealed_document(document)
    accepted_run_ids = {str(run_id), *(str(value) for value in compatible_run_ids)}
    if (
        document.get("schema_version")
        != "stage6_retrospective_nested_candidate_checkpoint_v1"
        or document.get("run_id") not in accepted_run_ids
        or document.get("stage") != int(stage)
        or document.get("candidate_id") != entry["candidate_id"]
        or document.get("configuration_sha256") != entry["configuration_sha256"]
        or document.get("stage_data_binding_self_sha256")
        != data_binding_self_sha256
        or document.get("result", {}).get("validation_temperature_selected")
        is not False
        or document.get("qualification_authority") is not False
    ):
        raise RuntimeError(f"candidate checkpoint binding changed: {path}")
    return document


def _write_checkpoint(
    path: Path,
    *,
    run_id: str,
    stage: int,
    entry: Mapping[str, Any],
    data_binding_self_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    from stage6_drive_fuse_publication import create_or_verify_bytes

    document = seal_document({
        "schema_version": "stage6_retrospective_nested_candidate_checkpoint_v1",
        "run_id": run_id,
        "stage": int(stage),
        "candidate_id": entry["candidate_id"],
        "configuration_sha256": entry["configuration_sha256"],
        "stage_data_binding_self_sha256": data_binding_self_sha256,
        "result": dict(result),
        "qualification_authority": False,
    })
    raw = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    create_or_verify_bytes(path, raw, owner_run_id=run_id)
    return document


def _search(
    *,
    dataset: Mapping[str, Any],
    layer: str,
    fold_id: str,
    registry: Sequence[Mapping[str, Any]],
    requested_workers: str,
    work_root: Path,
    run_id: str,
    compatible_checkpoint_run_ids: Sequence[str] = (),
) -> tuple[Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from stage6_expanded_tournament import (
        _stage_arrays,
        _write_stage_arrays,
        stage_data_binding,
    )

    search_dataset = dict(dataset)
    search_frame = dataset["frame"].copy()
    search_frame["temporal_role"] = search_frame["nested_role"].map({
        "train": "development_train",
        "validation": "development_validation",
        "calibration": "unused_calibration",
        "test": "unused_outer_test",
    })
    search_dataset["frame"] = search_frame
    by_id = {str(row["candidate_id"]): dict(row) for row in registry}
    active_ids = list(by_id)
    stage_summaries: list[dict[str, Any]] = []
    worker_reports: list[dict[str, Any]] = []

    for stage in NESTED_STAGES:
        stage_number = int(stage["stage"])
        arrays = _stage_arrays(search_dataset, layer, 0, stage)
        data_binding = stage_data_binding(
            arrays, dataset.get("whole_hand_temporal_assignment_binding")
        )
        reports: list[dict[str, Any]] = []
        unfinished: list[Mapping[str, Any]] = []
        for candidate_id in active_ids:
            entry = by_id[candidate_id]
            checkpoint = _load_checkpoint(
                _checkpoint_path(work_root, stage_number, candidate_id),
                run_id=run_id,
                stage=stage_number,
                entry=entry,
                data_binding_self_sha256=data_binding["self_sha256"],
                compatible_run_ids=compatible_checkpoint_run_ids,
            )
            if checkpoint is None:
                unfinished.append(entry)
            else:
                reports.append(checkpoint)

        if unfinished:
            worker_report = _resolve_candidate_workers(
                requested_workers, len(unfinished)
            )
            workers = int(worker_report["live_candidate_workers"])
            worker_reports.append({"stage": stage_number, **worker_report})
            with tempfile.TemporaryDirectory(
                prefix=f"stage6-nested-{layer}-{fold_id}-s{stage_number}-"
            ) as temporary:
                paths = _write_stage_arrays(Path(temporary), arrays)
                pending = list(unfinished)
                attempts = {str(row["candidate_id"]): 0 for row in pending}
                with _Progress(
                    layer=layer,
                    fold_id=fold_id,
                    stage=stage_number,
                    completed=len(reports),
                    denominator=len(active_ids),
                    live_workers=workers,
                    checkpoint=str(
                        work_root / "candidate_results" / f"stage-{stage_number}"
                    ),
                ) as progress:
                    while pending:
                        retry: list[Mapping[str, Any]] = []
                        oom_seen = False
                        with ProcessPoolExecutor(max_workers=workers) as executor:
                            futures = {
                                executor.submit(_candidate_worker, {
                                    "entry": entry,
                                    "paths": paths,
                                    "stage": stage,
                                }): entry
                                for entry in pending
                            }
                            for future in as_completed(futures):
                                entry = futures[future]
                                candidate_id = str(entry["candidate_id"])
                                try:
                                    result = future.result()
                                except BaseException as error:
                                    attempts[candidate_id] += 1
                                    if _confirmed_oom(error):
                                        oom_seen = True
                                    if attempts[candidate_id] < MAX_CANDIDATE_ATTEMPTS:
                                        retry.append(entry)
                                        continue
                                    if _is_invalid_validation_probability_error(
                                        error
                                    ):
                                        checkpoint = _write_checkpoint(
                                            _checkpoint_path(
                                                work_root,
                                                stage_number,
                                                candidate_id,
                                            ),
                                            run_id=run_id,
                                            stage=stage_number,
                                            entry=entry,
                                            data_binding_self_sha256=(
                                                data_binding["self_sha256"]
                                            ),
                                            result=(
                                                _rejected_invalid_probability_result(
                                                    entry, stage
                                                )
                                            ),
                                        )
                                        reports.append(checkpoint)
                                        progress.update(len(reports))
                                        print(json.dumps({
                                            "event": (
                                                "stage6_retrospective_nested_"
                                                "candidate_rejected"
                                            ),
                                            "layer": layer,
                                            "fold_id": fold_id,
                                            "stage": stage_number,
                                            "candidate_id": candidate_id,
                                            "reason": (
                                                INVALID_VALIDATION_PROBABILITY_ERROR
                                            ),
                                            "attempts": attempts[candidate_id],
                                            "checkpoint": str(
                                                _checkpoint_path(
                                                    work_root,
                                                    stage_number,
                                                    candidate_id,
                                                )
                                            ),
                                            "scores_fabricated": False,
                                            "rank_eligible": False,
                                        }, sort_keys=True), flush=True)
                                        continue
                                    raise RuntimeError(
                                        f"candidate failed after retry: {candidate_id}: {error}"
                                    ) from error
                                checkpoint = _write_checkpoint(
                                    _checkpoint_path(
                                        work_root, stage_number, candidate_id
                                    ),
                                    run_id=run_id,
                                    stage=stage_number,
                                    entry=entry,
                                    data_binding_self_sha256=data_binding[
                                        "self_sha256"
                                    ],
                                    result=result,
                                )
                                reports.append(checkpoint)
                                progress.update(len(reports))
                        if oom_seen:
                            if workers <= 1:
                                raise MemoryError(
                                    "confirmed OOM persisted at one candidate worker"
                                )
                            workers = max(1, workers // 2)
                            print(json.dumps({
                                "event": "stage6_nested_confirmed_oom_worker_reduction",
                                "layer": layer,
                                "fold_id": fold_id,
                                "stage": stage_number,
                                "next_live_workers": workers,
                                "model_configuration_changed": False,
                            }, sort_keys=True), flush=True)
                        pending = retry

        if len(reports) != len(active_ids):
            raise RuntimeError("nested candidate checkpoint conservation failed")
        ranking = _rank(reports)
        rejected_ids = sorted(
            str(row["candidate_id"])
            for row in reports
            if row.get("result", {}).get("rank_eligible", True) is False
        )
        checkpoint_inventory, checkpoint_root = _checkpoint_inventory(reports)
        required_survivors = int(stage["survivor_count"])
        if stage_number < len(NESTED_STAGES) and len(ranking) < required_survivors:
            raise RuntimeError(
                "nested stage has too few rank-eligible candidates to preserve "
                f"the next frozen stage: stage {stage_number} has "
                f"{len(ranking)}, requires {required_survivors}"
            )
        survivor_count = min(required_survivors, len(ranking))
        survivors = [str(row["candidate_id"]) for row in ranking[:survivor_count]]
        stage_summaries.append({
            "stage": stage_number,
            "candidate_count": len(active_ids),
            "rank_eligible_candidate_count": len(ranking),
            "rejected_invalid_probability_candidate_count": len(rejected_ids),
            "rejected_invalid_probability_candidate_ids": rejected_ids,
            "survivor_count": survivor_count,
            "survivor_ids": survivors,
            "stage_data_binding_self_sha256": data_binding["self_sha256"],
            "top_validation_log_loss": float(
                ranking[0]["result"]["validation_metrics"]["log_loss"]
            ),
            "top_validation_brier": float(
                ranking[0]["result"]["validation_metrics"][
                    "multiclass_brier_score"
                ]
            ),
            "validation_temperature_selected": False,
            "checkpoint_inventory": checkpoint_inventory,
            "checkpoint_inventory_root_sha256": checkpoint_root,
        })
        active_ids = survivors
    final_ranking = _rank(reports)
    return final_ranking[0], stage_summaries, worker_reports


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _predict_projected(
    model: Any,
    matrix: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, dict[str, int]]:
    from stage6_expanded_tournament import _project_legal

    raw = _aligned_six_action_probabilities(model, matrix)
    raw_illegal = int((~legal[np.arange(len(legal)), raw.argmax(axis=1)]).sum())
    projected, projector_raw_illegal, fallback = _project_legal(
        raw, legal, priors
    )
    if projector_raw_illegal != raw_illegal:
        raise RuntimeError("nested finalist raw legality diagnostic changed")
    scaled = _temperature_scale(projected, temperature)
    scaled, scaled_raw_illegal, scaled_fallback = _project_legal(
        scaled, legal, priors
    )
    if fallback or scaled_fallback:
        raise RuntimeError("nested finalist produced invalid probabilities")
    if (scaled[~legal] > 1e-15).any():
        raise RuntimeError("nested finalist served illegal probability mass")
    return scaled, {
        "raw_preprojection_illegal_argmax_count": int(
            raw_illegal + scaled_raw_illegal
        ),
        "legal_projection_fallback_count": int(fallback + scaled_fallback),
        "served_illegal_probability_mass_count": 0,
    }


def _calibrate_temperature(
    model: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    from stage6_metrics import multiclass_metric_report

    trials = []
    for temperature in TEMPERATURE_GRID:
        probabilities, counts = _predict_projected(
            model, matrix, legal, priors, temperature
        )
        metrics = multiclass_metric_report(
            [ACTIONS[int(value)] for value in target], probabilities, ACTIONS
        )
        trials.append((metrics["log_loss"], temperature, metrics, counts))
    trials.sort(key=lambda row: (row[0], row[1]))
    _, temperature, metrics, counts = trials[0]
    return float(temperature), {
        "metrics": metrics,
        "projection_counts": counts,
        "temperature_grid": list(TEMPERATURE_GRID),
        "selection_partition": "inner_calibration_only",
    }


def _control_matrix(dataset: Mapping[str, Any], layer: str) -> np.ndarray:
    from stage6_expanded_feature_freeze_v2 import BASE_FEATURE_COLUMNS
    from stage6_expanded_tournament import expanded_model_matrix

    feature_frame = dataset["feature_frame"]
    if layer == "hero_opponent_interaction":
        _, matrix = expanded_model_matrix(
            feature_frame, layer, include_dyad_interaction=False
        )
        return matrix
    return feature_frame[list(BASE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)


def _support_mask(dataset: Mapping[str, Any], layer: str) -> np.ndarray:
    if layer == "hero_opponent_interaction":
        return (
            pd.to_numeric(
                dataset["feature_frame"]["dyad_prior_hand_count"],
                errors="raise",
            ).fillna(0).to_numpy(dtype=float)
            >= 2.0
        )
    if layer == "within_hand_line_surprise":
        return (
            pd.to_numeric(
                dataset["feature_frame"]["line_prefix_decision_count"],
                errors="raise",
            ).fillna(0).to_numpy(dtype=float)
            >= 1.0
        )
    if LAYER_FAMILIES[layer] == "additional":
        columns = [
            index
            for index, name in enumerate(dataset["feature_names"])
            if str(name).endswith("_supported")
        ]
        if columns:
            values = np.asarray(dataset["matrix"], dtype=float)[:, columns]
            return np.any(np.nan_to_num(values, nan=0.0) > 0.5, axis=1)
    return np.ones(len(dataset["frame"]), dtype=bool)


def _fit_outer_component(
    *,
    entry: Mapping[str, Any],
    matrix: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
    best_iteration: int,
) -> dict[str, Any]:
    """Fit and score one deterministic one-thread outer-final component."""
    from stage6_expanded_tournament import _apply_imputer, _impute_medians

    medians = _impute_medians(matrix[fit_indices])
    fit_x = _apply_imputer(matrix[fit_indices], medians)
    model = _fit_fixed_candidate_allowing_action_subsets(
        entry, fit_x, target[fit_indices], best_iteration
    )
    temperature, calibration_report = _calibrate_temperature(
        model,
        _apply_imputer(matrix[calibration_indices], medians),
        target[calibration_indices],
        legal[calibration_indices],
        priors,
    )
    probabilities, projection = _predict_projected(
        model,
        _apply_imputer(matrix[test_indices], medians),
        legal[test_indices],
        priors,
        temperature,
    )
    del model, fit_x, medians
    gc.collect()
    return {
        "probabilities": probabilities,
        "temperature": temperature,
        "calibration_report": calibration_report,
        "projection": projection,
    }


def _outer_predictions_under_one_thread(
    *,
    dataset: Mapping[str, Any],
    layer: str,
    entry: Mapping[str, Any],
    best_iteration: int,
) -> tuple[bytes, dict[str, Any]]:
    from stage6_metrics import multiclass_metric_report

    roles = dataset["frame"]["nested_role"].astype(str).to_numpy()
    fit_indices = np.flatnonzero(np.isin(roles, ["train", "validation"]))
    calibration_indices = np.flatnonzero(roles == "calibration")
    test_indices = np.flatnonzero(roles == "test")
    target = np.asarray(dataset["target"], dtype=np.int8)
    legal = np.asarray(dataset["legal"], dtype=bool)
    matrix = np.asarray(dataset["matrix"], dtype=np.float32)
    support = _support_mask(dataset, layer)
    fit_y = target[fit_indices]
    if len(np.unique(fit_y)) < 2:
        raise RuntimeError("nested finalist fit partition has fewer than two classes")
    counts = np.bincount(fit_y, minlength=len(ACTIONS)).astype(float)
    priors = (counts + 1.0) / (counts.sum() + len(ACTIONS))
    component_matrices = {"candidate": matrix}
    if LAYER_FAMILIES[layer] == "expanded":
        component_matrices["matched_control"] = _control_matrix(dataset, layer)

    component_results: dict[str, dict[str, Any]] = {}
    component_workers = len(component_matrices)
    if component_workers == 1:
        component_results["candidate"] = _fit_outer_component(
            entry=entry,
            matrix=matrix,
            fit_indices=fit_indices,
            calibration_indices=calibration_indices,
            test_indices=test_indices,
            target=target,
            legal=legal,
            priors=priors,
            best_iteration=best_iteration,
        )
    else:
        print(json.dumps({
            "event": "stage6_retrospective_nested_outer_final_progress",
            "phase": "parallel_candidate_and_matched_control_fit",
            "layer": layer,
            "completed": 0,
            "total": component_workers,
            "component_workers": component_workers,
            "model_internal_threads": 1,
        }, sort_keys=True), flush=True)
        with ThreadPoolExecutor(max_workers=component_workers) as executor:
            futures = {
                executor.submit(
                    _fit_outer_component,
                    entry=entry,
                    matrix=component_matrix,
                    fit_indices=fit_indices,
                    calibration_indices=calibration_indices,
                    test_indices=test_indices,
                    target=target,
                    legal=legal,
                    priors=priors,
                    best_iteration=best_iteration,
                ): component_name
                for component_name, component_matrix in component_matrices.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                component_name = futures[future]
                component_results[component_name] = future.result()
                print(json.dumps({
                    "event": "stage6_retrospective_nested_outer_final_progress",
                    "phase": "parallel_candidate_and_matched_control_fit",
                    "layer": layer,
                    "completed_component": component_name,
                    "completed": completed,
                    "total": component_workers,
                    "component_workers": component_workers,
                    "model_internal_threads": 1,
                }, sort_keys=True), flush=True)

    candidate_result = component_results["candidate"]
    candidate = np.asarray(candidate_result["probabilities"], dtype=np.float64)
    temperature = float(candidate_result["temperature"])
    calibration_report = candidate_result["calibration_report"]
    projection = candidate_result["projection"]
    truth = [ACTIONS[int(value)] for value in target[test_indices]]
    outer_metrics = multiclass_metric_report(truth, candidate, ACTIONS)

    arrays: dict[str, Any] = {
        "row_ids": dataset["frame"].iloc[test_indices]["row_id"].astype(str).to_numpy(dtype="U128"),
        "game_ids": dataset["frame"].iloc[test_indices]["game_id"].astype(str).to_numpy(dtype="U128"),
        "hand_ids": dataset["frame"].iloc[test_indices]["hand_id"].astype(str).to_numpy(dtype="U128"),
        "target": target[test_indices],
        "legal": legal[test_indices].astype(np.uint8),
        "support_mask": support[test_indices].astype(np.uint8),
        "candidate_probabilities": candidate.astype(np.float32),
    }
    report: dict[str, Any] = {
        "fit_row_count": len(fit_indices),
        "calibration_row_count": len(calibration_indices),
        "outer_test_row_count": len(test_indices),
        "outer_test_game_count": int(
            dataset["frame"].iloc[test_indices]["game_id"].astype(str).nunique()
        ),
        "outer_test_supported_row_count": int(support[test_indices].sum()),
        "outer_test_supported_game_count": int(
            dataset["frame"].iloc[
                test_indices[np.flatnonzero(support[test_indices])]
            ]["game_id"].astype(str).nunique()
        ),
        "selected_temperature": temperature,
        "temperature_selection_partition": "inner_calibration_only",
        "calibration_report": calibration_report,
        "outer_candidate_metrics": outer_metrics,
        "candidate_projection_counts": projection,
        "projection_counter_semantics": (
            "raw_illegal_argmax_diagnostic_separate_from_true_zero_mass_fallback"
        ),
        "matched_control_present": False,
        "candidate_internal_threads": 1,
        "outer_final_component_workers": component_workers,
        "threadpoolctl_limit_applied": True,
    }

    if LAYER_FAMILIES[layer] == "expanded":
        control_result = component_results["matched_control"]
        control = np.asarray(control_result["probabilities"], dtype=np.float64)
        control_temperature = float(control_result["temperature"])
        control_calibration = control_result["calibration_report"]
        control_projection = control_result["projection"]
        arrays["matched_control_probabilities"] = control.astype(np.float32)
        served = candidate.copy()
        served[~support[test_indices]] = control[~support[test_indices]]
        arrays["served_candidate_probabilities"] = served.astype(np.float32)
        report.update({
            "matched_control_present": True,
            "matched_control_temperature": control_temperature,
            "matched_control_calibration_report": control_calibration,
            "outer_matched_control_metrics": multiclass_metric_report(
                truth, control, ACTIONS
            ),
            "matched_control_projection_counts": control_projection,
            "below_support_served_exact_matched_control": bool(
                np.array_equal(
                    served[~support[test_indices]],
                    control[~support[test_indices]],
                )
            ),
        })
        del control, served

    del candidate, matrix, component_matrices, component_results
    gc.collect()
    return _fixed_npz_bytes(arrays), report


def _outer_predictions(
    *,
    dataset: Mapping[str, Any],
    layer: str,
    entry: Mapping[str, Any],
    best_iteration: int,
) -> tuple[bytes, dict[str, Any]]:
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=1):
        raw, report = _outer_predictions_under_one_thread(
            dataset=dataset,
            layer=layer,
            entry=entry,
            best_iteration=best_iteration,
        )
        observed_max_threads = max(
            (int(row.get("num_threads", 0)) for row in threadpool_info()),
            default=1,
        )
        if observed_max_threads > 1:
            raise RuntimeError("finalist native thread pool exceeded one thread")
    report["observed_native_threadpool_max_threads"] = observed_max_threads
    return raw, report


def _outer_checkpoint_path(work_root: Path) -> Path:
    return work_root / "outer-test-checkpoint.zip"


def _load_outer_checkpoint(
    path: Path,
    *,
    run_id: str,
    layer: str,
    selected_checkpoint_self_sha256: str,
    selected_configuration_sha256: str,
    best_iteration: int,
) -> dict[str, Any] | None:
    from stage6_drive_fuse_publication import recover_claimed_bytes

    recover_claimed_bytes(path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != [
            "outer-test-predictions.npz",
            "outer-test-report.json",
        ]:
            raise RuntimeError("outer checkpoint inventory changed")
        prediction_raw = archive.read("outer-test-predictions.npz")
        report_raw = archive.read("outer-test-report.json")
    report = json.loads(report_raw.decode("utf-8"))
    validate_sealed_document(report)
    expected_prediction = report.get("prediction_file", {})
    if (
        report.get("schema_version")
        != "stage6_retrospective_nested_outer_test_checkpoint_v1"
        or report.get("run_id") != run_id
        or report.get("layer") != layer
        or report.get("selected_checkpoint_self_sha256")
        != selected_checkpoint_self_sha256
        or report.get("selected_configuration_sha256")
        != selected_configuration_sha256
        or report.get("selected_best_iteration") != int(best_iteration)
        or expected_prediction.get("name") != "outer-test-predictions.npz"
        or expected_prediction.get("bytes") != len(prediction_raw)
        or expected_prediction.get("sha256")
        != hashlib.sha256(prediction_raw).hexdigest()
        or report.get("qualification_authority") is not False
    ):
        raise RuntimeError("outer checkpoint binding changed")
    required_arrays = [
        "row_ids",
        "game_ids",
        "hand_ids",
        "target",
        "legal",
        "support_mask",
        "candidate_probabilities",
    ]
    if LAYER_FAMILIES[layer] == "expanded":
        required_arrays.extend([
            "matched_control_probabilities",
            "served_candidate_probabilities",
        ])
    with np.load(io.BytesIO(prediction_raw), allow_pickle=False) as payload:
        if payload.files != required_arrays:
            raise RuntimeError("outer checkpoint prediction inventory changed")
        row_count = len(payload["target"])
        if any(len(payload[name]) != row_count for name in required_arrays):
            raise RuntimeError("outer checkpoint prediction row counts changed")
        probabilities = np.asarray(
            payload["candidate_probabilities"], dtype=np.float64
        )
        if probabilities.shape != (row_count, len(ACTIONS)):
            raise RuntimeError("outer checkpoint probability shape changed")
    return {
        "raw": raw,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "prediction_raw": prediction_raw,
        "report": report,
    }


def _write_outer_checkpoint(
    path: Path,
    *,
    run_id: str,
    layer: str,
    selected_checkpoint_self_sha256: str,
    selected_configuration_sha256: str,
    best_iteration: int,
    prediction_raw: bytes,
    prediction_report: Mapping[str, Any],
) -> dict[str, Any]:
    from stage6_drive_fuse_publication import create_or_verify_bytes

    report = seal_document({
        "schema_version": "stage6_retrospective_nested_outer_test_checkpoint_v1",
        "run_id": run_id,
        "layer": layer,
        "selected_checkpoint_self_sha256": selected_checkpoint_self_sha256,
        "selected_configuration_sha256": selected_configuration_sha256,
        "selected_best_iteration": int(best_iteration),
        "prediction_file": {
            "name": "outer-test-predictions.npz",
            "bytes": len(prediction_raw),
            "sha256": hashlib.sha256(prediction_raw).hexdigest(),
        },
        "prediction_report": dict(prediction_report),
        "candidate_and_iteration_selection_partitions": ["train", "validation"],
        "temperature_selection_partition": "calibration",
        "outer_test_used_for_selection": False,
        "qualification_authority": False,
    })
    raw = _fixed_zip_bytes([
        ("outer-test-predictions.npz", prediction_raw),
        ("outer-test-report.json", canonical_json_bytes(report) + b"\n"),
    ])
    create_or_verify_bytes(path, raw, owner_run_id=run_id)
    loaded = _load_outer_checkpoint(
        path,
        run_id=run_id,
        layer=layer,
        selected_checkpoint_self_sha256=selected_checkpoint_self_sha256,
        selected_configuration_sha256=selected_configuration_sha256,
        best_iteration=best_iteration,
    )
    if loaded is None:
        raise RuntimeError("outer checkpoint publication disappeared")
    return loaded


def run_retrospective_nested_layer_fold_search(
    stage_drive_root: str | Path,
    layer: str,
    fold_index: int,
    *,
    requested_workers: str | int = "auto",
    accelerator: str = "TPU",
) -> dict[str, Any]:
    """Run and checkpoint one exact layer/fold nested search."""
    _bind_feature_release_v2()
    from stage6_drive_fuse_publication import (
        create_or_verify_bytes,
        recover_claimed_bytes,
    )
    from stage6_expanded_tournament import runtime_environment

    stage_drive_root = Path(stage_drive_root).resolve()
    if layer not in LAYER_FAMILIES:
        raise ValueError(f"unknown retrospective nested layer: {layer}")
    fold_index = int(fold_index)
    if fold_index not in range(5):
        raise ValueError("fold_index must be 0 through 4")
    requested_workers = _validate_requested_workers(requested_workers)
    accelerator = str(accelerator).strip().upper()
    if accelerator not in {"CPU", "GPU", "TPU"}:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    fold_freeze = load_fold_freeze(stage_drive_root)
    assignment = fold_freeze["assignments"]["folds"][fold_index]
    validate_sealed_document(assignment)
    if assignment.get("outer_fold_index") != fold_index:
        raise RuntimeError("nested fold index/order changed")
    registry = _registry(layer)
    registry_binding = _registry_binding(
        layer, registry, fold_freeze["assignments"]
    )
    environment = runtime_environment()
    import stage6_retrospective_nested_additional_source as additional_source

    implementation_members = {
        Path(__file__).name: hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        Path(additional_source.__file__).name: hashlib.sha256(
            Path(additional_source.__file__).resolve().read_bytes()
        ).hexdigest(),
    }
    pre_run_binding = {
        "schema": "stage6_retrospective_nested_layer_fold_search_v2",
        "fold_run_id": FOLD_RUN_ID,
        "fold_assignment_self_sha256": assignment["self_sha256"],
        "layer": layer,
        "family": LAYER_FAMILIES[layer],
        "registry": registry_binding,
        "science_compatibility_sha256": (
            RETROSPECTIVE_NESTED_SCIENCE_COMPATIBILITY_SHA256
        ),
        "runtime_package_bindings": RUNTIME_PACKAGE_BINDINGS,
        "nested_stages": list(NESTED_STAGES),
        "temperature_grid": list(TEMPERATURE_GRID),
    }
    pre_run_id = hashlib.sha256(canonical_json_bytes(pre_run_binding)).hexdigest()
    print(json.dumps({
        "event": "stage6_retrospective_nested_layer_fold_started",
        "layer": layer,
        "family": LAYER_FAMILIES[layer],
        "fold_id": assignment["fold_id"],
        "fold_index": fold_index,
        "candidate_count": 256,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "requested_workers": requested_workers,
        "accepted_accelerator": accelerator,
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_gpu_or_tpu_acceleration": False,
    }, sort_keys=True), flush=True)
    dataset = _fold(_load_dataset(stage_drive_root, layer), assignment)
    dataset_source_binding = {
        key: dataset.get(key)
        for key in (
            "source_manifest_self_sha256",
            "feature_manifest_self_sha256",
            "raw_nan_preserved_for_fold_local_imputation",
        )
        if dataset.get(key) is not None
    }
    run_binding = seal_document({
        **pre_run_binding,
        "dataset_source_binding": dataset_source_binding,
    })
    run_id = str(run_binding["self_sha256"])
    checkpoint_storage_run_id = R2_RUNTIME_RESUME_STORAGE_RUN_IDS.get(
        (str(assignment["fold_id"]), layer), run_id
    )
    compatible_checkpoint_run_ids = tuple(
        value for value in (checkpoint_storage_run_id,) if value != run_id
    )
    work_root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "searches"
        / str(assignment["fold_id"])
        / LAYER_FAMILIES[layer]
        / layer
        / checkpoint_storage_run_id
    )
    manifest_path = work_root / "nested-search-result-manifest.json"
    work_root.mkdir(parents=True, exist_ok=True)
    recover_claimed_bytes(manifest_path)
    if manifest_path.is_file():
        completed = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_sealed_document(completed)
        outer_checkpoint_file = completed.get("outer_checkpoint_file", {})
        outer_checkpoint_path = work_root / str(
            outer_checkpoint_file.get("name", "")
        )
        outer_checkpoint = _load_outer_checkpoint(
            outer_checkpoint_path,
            run_id=run_id,
            layer=layer,
            selected_checkpoint_self_sha256=str(
                completed.get("selected_checkpoint_self_sha256", "")
            ),
            selected_configuration_sha256=str(
                completed.get("selected_configuration_sha256", "")
            ),
            best_iteration=int(completed.get("selected_best_iteration", 0)),
        )
        if (
            completed.get("status")
            != "PASS_RETROSPECTIVE_NESTED_LAYER_FOLD_SEARCH"
            or completed.get("run_id") != run_id
            or completed.get("pre_run_id") != pre_run_id
            or outer_checkpoint is None
            or outer_checkpoint_file.get("name")
            != "outer-test-checkpoint.zip"
            or outer_checkpoint["bytes"] != outer_checkpoint_file.get("bytes")
            or outer_checkpoint["sha256"]
            != outer_checkpoint_file.get("sha256")
            or outer_checkpoint["report"].get("prediction_report")
            != completed.get("prediction_report")
            or completed.get("qualification_authority") is not False
            or completed.get("run_binding") != run_binding
        ):
            raise RuntimeError("completed nested search manifest is invalid")
        completed_evaluations = _assert_exact_evaluation_contract(
            completed.get("stage_summaries", [])
        )
        if completed.get("candidate_evaluations_per_fold") != completed_evaluations:
            raise RuntimeError("completed nested search evaluation count changed")
        _validate_checkpoint_inventories(
            work_root, completed.get("stage_summaries", [])
        )
        manifest_raw = canonical_json_bytes(completed) + b"\n"
        returned_raw = _fixed_zip_bytes([
            ("nested-search-result-manifest.json", manifest_raw),
            ("outer-test-checkpoint.zip", outer_checkpoint["raw"]),
        ])
        returned_name = (
            "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_SEARCH_RESULT_"
            f"{layer}_{assignment['fold_id']}_{run_id[:12]}.zip"
        )
        returned_path = stage_drive_root / "returned_results" / returned_name
        recover_claimed_bytes(returned_path)
        create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
        return {
            **completed,
            "manifest_path": str(manifest_path),
            "returned_result": {
                "path": str(returned_path),
                "name": returned_name,
                "bytes": len(returned_raw),
                "sha256": hashlib.sha256(returned_raw).hexdigest(),
            },
        }
    winner_checkpoint, stage_summaries, worker_reports = _search(
        dataset=dataset,
        layer=layer,
        fold_id=str(assignment["fold_id"]),
        registry=registry,
        requested_workers=requested_workers,
        work_root=work_root,
        run_id=run_id,
        compatible_checkpoint_run_ids=compatible_checkpoint_run_ids,
    )
    candidate_evaluations_per_fold = _assert_exact_evaluation_contract(
        stage_summaries
    )
    winner = next(
        row for row in registry
        if row["candidate_id"] == winner_checkpoint["candidate_id"]
    )
    best_iteration = int(winner_checkpoint["result"]["best_iteration"])
    outer_checkpoint_path = _outer_checkpoint_path(work_root)
    outer_checkpoint = _load_outer_checkpoint(
        outer_checkpoint_path,
        run_id=run_id,
        layer=layer,
        selected_checkpoint_self_sha256=str(winner_checkpoint["self_sha256"]),
        selected_configuration_sha256=str(winner["configuration_sha256"]),
        best_iteration=best_iteration,
    )
    if outer_checkpoint is None:
        prediction_raw, prediction_report = _outer_predictions(
            dataset=dataset,
            layer=layer,
            entry=winner,
            best_iteration=best_iteration,
        )
        outer_checkpoint = _write_outer_checkpoint(
            outer_checkpoint_path,
            run_id=run_id,
            layer=layer,
            selected_checkpoint_self_sha256=str(
                winner_checkpoint["self_sha256"]
            ),
            selected_configuration_sha256=str(
                winner["configuration_sha256"]
            ),
            best_iteration=best_iteration,
            prediction_raw=prediction_raw,
            prediction_report=prediction_report,
        )
    prediction_report = dict(outer_checkpoint["report"]["prediction_report"])
    if worker_reports:
        print(json.dumps({
            "event": "stage6_retrospective_nested_worker_telemetry",
            "layer": layer,
            "fold_id": assignment["fold_id"],
            "reports": worker_reports,
            "sealed_scientific_manifest": False,
        }, sort_keys=True), flush=True)
    manifest = seal_document({
        "schema_version": "stage6_retrospective_nested_layer_fold_search_result_v1",
        "status": "PASS_RETROSPECTIVE_NESTED_LAYER_FOLD_SEARCH",
        "run_id": run_id,
        "pre_run_id": pre_run_id,
        "fold_run_id": FOLD_RUN_ID,
        "fold_id": assignment["fold_id"],
        "fold_index": fold_index,
        "fold_assignment_self_sha256": assignment["self_sha256"],
        "run_binding": run_binding,
        "execution_environment": environment,
        "execution_implementation_members": implementation_members,
        "checkpoint_storage_run_id": checkpoint_storage_run_id,
        "compatible_checkpoint_run_ids": list(compatible_checkpoint_run_ids),
        "runtime_independent_checkpoint_resume": True,
        "layer": layer,
        "family": LAYER_FAMILIES[layer],
        "registry_binding": registry_binding,
        "candidate_count": 256,
        "candidate_evaluations_per_fold": candidate_evaluations_per_fold,
        "selected_candidate_id": winner["candidate_id"],
        "selected_configuration_sha256": winner["configuration_sha256"],
        "selected_family": winner["family"],
        "selected_parameters": winner["parameters"],
        "selected_best_iteration": best_iteration,
        "selected_checkpoint_self_sha256": winner_checkpoint["self_sha256"],
        "stage_summaries": stage_summaries,
        "dataset_source_binding": dataset_source_binding,
        "outer_checkpoint_file": {
            "name": "outer-test-checkpoint.zip",
            "bytes": outer_checkpoint["bytes"],
            "sha256": outer_checkpoint["sha256"],
        },
        "prediction_report": prediction_report,
        "candidate_and_iteration_selection_partitions": ["train", "validation"],
        "temperature_selection_partition": "calibration",
        "outer_test_scored_once_per_atomic_checkpoint": True,
        "outer_checkpoint_reused_without_model_rescore_on_resume": True,
        "fresh_or_prospective_claim": False,
        "action_timing_layer_included": False,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "candidate_internal_threads": 1,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "direct_action_authority": False,
        "live_action_authority": False,
    })
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=run_id)
    returned_raw = _fixed_zip_bytes([
        ("nested-search-result-manifest.json", manifest_raw),
        ("outer-test-checkpoint.zip", outer_checkpoint["raw"]),
    ])
    returned_name = (
        "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_SEARCH_RESULT_"
        f"{layer}_{assignment['fold_id']}_{run_id[:12]}.zip"
    )
    returned_path = stage_drive_root / "returned_results" / returned_name
    recover_claimed_bytes(returned_path)
    create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
    print(json.dumps({
        "event": "stage6_retrospective_nested_layer_fold_complete",
        "layer": layer,
        "fold_id": assignment["fold_id"],
        "selected_candidate_id": winner["candidate_id"],
        "outer_test_game_count": prediction_report["outer_test_game_count"],
        "outer_test_row_count": prediction_report["outer_test_row_count"],
        "checkpoint": str(manifest_path),
        "returned_result": str(returned_path),
    }, sort_keys=True), flush=True)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "returned_result": {
            "path": str(returned_path),
            "name": returned_name,
            "bytes": len(returned_raw),
            "sha256": hashlib.sha256(returned_raw).hexdigest(),
        },
    }


__all__ = [
    "ADDITIONAL_LAYERS",
    "BEHAVIOR_LAYERS",
    "EXPANDED_LAYERS",
    "FOLD_RESULT_SHA256",
    "LAYER_FAMILIES",
    "NESTED_STAGES",
    "load_fold_freeze",
    "run_retrospective_nested_layer_fold_search",
]
