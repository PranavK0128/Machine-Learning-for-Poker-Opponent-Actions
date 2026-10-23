"""Process-parallel successive-halving tournament for expanded Stage 6 layers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_expanded_feature_freeze import (
    BASE_FEATURE_COLUMNS,
    FEATURE_RUN_ID,
    SCIENCE_IMPLEMENTATION_DIGEST as UPSTREAM_FEATURE_IMPLEMENTATION_DIGEST,
    expanded_feature_root,
)
from stage6_expanded_layers import (
    ACTIONS,
    EXPANDED_LAYER_CONTRACT_DIGEST,
    LOGICAL_LANE_COUNT,
    PINNED_RUNTIME_DISTRIBUTIONS,
    SCIENCE_IMPLEMENTATION_DIGEST,
    assert_no_prohibited_model_features,
    candidate_registry,
    candidate_registry_shard,
    canonical_json_bytes,
    dyad_interaction_feature_columns,
    dyad_main_effect_feature_columns,
    dyad_numeric_feature_columns,
    line_numeric_feature_columns,
    seal_document,
    shard_digest,
    validate_document,
)
from stage6_metrics import multiclass_metric_report
from stage6_notebooks02_07_runtime import _project_legal, sha256_file


HALVING_STAGES = (
    {
        "stage": 1,
        "train_row_cap": 60_000,
        "validation_row_cap": 20_000,
        "iteration_cap": 600,
        "patience": 50,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 48,
    },
    {
        "stage": 2,
        "train_row_cap": 180_000,
        "validation_row_cap": 60_000,
        "iteration_cap": 2_000,
        "patience": 100,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 16,
    },
    {
        "stage": 3,
        "train_row_cap": 500_000,
        "validation_row_cap": 150_000,
        "iteration_cap": 6_000,
        "patience": 150,
        "minimum_meaningful_improvement": 1e-5,
        "survivor_count": 16,
    },
)
TEMPERATURE_GRID = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00)
MINIMUM_WORKER_PEAK_BYTES = 1_250_000_000
RAM_RESERVE_BYTES = 6_000_000_000
MAX_CANDIDATE_ATTEMPTS = 2
PROGRESS_INTERVAL_SECONDS = 60.0


class _CatBoostMeaningfulImprovementStopper:
    """Stop CatBoost only after patience rounds without a meaningful loss drop."""

    def __init__(self, *, patience: int, minimum_improvement: float) -> None:
        self.patience = max(1, int(patience))
        self.minimum_improvement = max(0.0, float(minimum_improvement))
        self.best_loss = math.inf
        self.best_tree_count = 0
        self.rounds_without_meaningful_improvement = 0

    def after_iteration(self, info: Any) -> bool:
        metrics = getattr(info, "metrics", {})
        validation = metrics.get("validation")
        if validation is None:
            validation_keys = sorted(
                key for key in metrics if str(key).startswith("validation")
            )
            if not validation_keys:
                raise RuntimeError("CatBoost validation metrics are unavailable")
            validation = metrics[validation_keys[0]]
        history = validation.get("MultiClass")
        if not history:
            raise RuntimeError("CatBoost MultiClass validation history is unavailable")
        loss = float(history[-1])
        if not math.isfinite(loss):
            raise RuntimeError("CatBoost produced a non-finite validation loss")
        tree_count = len(history)
        if self.best_tree_count == 0 or self.best_loss - loss > self.minimum_improvement:
            self.best_loss = loss
            self.best_tree_count = tree_count
            self.rounds_without_meaningful_improvement = 0
        else:
            self.rounds_without_meaningful_improvement += 1
        return self.rounds_without_meaningful_improvement < self.patience


class _PhaseProgress:
    """Phase-local process-pool progress with bounded silent intervals."""

    def __init__(
        self,
        *,
        phase: str,
        completed: int,
        denominator: int,
        live_workers: int,
        checkpoint: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.phase = str(phase)
        self.baseline_completed = int(completed)
        self.completed = int(completed)
        self.denominator = int(denominator)
        self.live_workers = int(live_workers)
        self.checkpoint = str(checkpoint)
        self.context = dict(context or {})
        self.started = time.monotonic()
        self.last_emitted = self.started

    def seconds_until_heartbeat(self) -> float:
        return max(
            0.0,
            PROGRESS_INTERVAL_SECONDS - (time.monotonic() - self.last_emitted),
        )

    def emit(
        self,
        *,
        event: str = "stage6_expanded_model_training_progress",
        force: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> bool:
        now = time.monotonic()
        if not force and now - self.last_emitted < PROGRESS_INTERVAL_SECONDS:
            return False
        elapsed = max(0.0, now - self.started)
        phase_completions = max(0, self.completed - self.baseline_completed)
        throughput = phase_completions / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.denominator - self.completed)
        disk_root = Path("/content") if Path("/content").is_dir() else Path.cwd()
        payload: dict[str, Any] = {
            "event": event,
            "phase": self.phase,
            "work_unit": "candidates",
            "completed": self.completed,
            "denominator": self.denominator,
            "live_workers": min(self.live_workers, max(0, remaining)),
            "throughput_per_second": throughput,
            "eta_seconds": remaining / throughput if throughput > 0 else None,
            "available_system_ram_bytes": _available_ram(),
            "available_disk_bytes": shutil.disk_usage(disk_root).free,
            "checkpoint": self.checkpoint,
            **self.context,
        }
        if extra:
            payload.update(extra)
        print(json.dumps(payload, sort_keys=True), flush=True)
        self.last_emitted = now
        return True

    def completed_one(
        self,
        checkpoint: Path,
        *,
        event: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.completed += 1
        self.checkpoint = str(checkpoint)
        self.emit(event=event, force=True, extra=extra)


def broad_run_id(layer: str, shard: int) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_layer_broad_tournament_v1",
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation": SCIENCE_IMPLEMENTATION_DIGEST,
        "layer": layer,
        "shard": shard,
        "shard_digest": shard_digest(layer, shard),
        "halving_stages": HALVING_STAGES,
        "temperature_grid": TEMPERATURE_GRID,
        "candidate_internal_threads": 1,
    })).hexdigest()


def broad_work_root(stage_drive_root: Path, layer: str, shard: int) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "broad_tournaments"
        / broad_run_id(layer, shard)
    )


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    from stage6_expanded_feature_freeze import _atomic_json as write

    write(path, document)


def _load_feature_manifest(stage_drive_root: Path) -> tuple[Path, dict[str, Any]]:
    root = expanded_feature_root(stage_drive_root)
    path = root / "feature-freeze-manifest.json"
    if not path.is_file():
        raise RuntimeError("Notebook 12 feature freeze must complete before broad tournaments")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("schema_version") != "stage6_expanded_feature_freeze_manifest_v1"
        or manifest.get("status") != "PASS_DEVELOPMENT_FEATURE_FREEZE"
        or manifest.get("feature_run_id") != FEATURE_RUN_ID
        or manifest.get("science_implementation_self_sha256")
        != UPSTREAM_FEATURE_IMPLEMENTATION_DIGEST
        or manifest.get("completed_lane_count") != LOGICAL_LANE_COUNT
        or manifest.get("fresh_targets_opened") is not False
        or manifest.get("qualification_authority") is not False
    ):
        raise RuntimeError("Notebook 12 feature freeze authority is invalid")
    return root, manifest


def _verify_lane_identity(root: Path, identity: Mapping[str, Any]) -> Path:
    path = root / str(identity["relative_path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(identity["bytes"])
        or sha256_file(path) != str(identity["sha256"])
        or pq.ParquetFile(path).metadata.num_rows != int(identity["row_count"])
    ):
        raise RuntimeError(f"expanded feature lane identity mismatch: {path}")
    return path


def load_layer_dataset(
    stage_drive_root: Path,
    layer: str,
) -> dict[str, Any]:
    root, manifest = _load_feature_manifest(stage_drive_root)
    frames: list[pd.DataFrame] = []
    for lane_number, lane in enumerate(manifest["lanes"]):
        lane_id = f"lane-{lane_number:02d}"
        if lane.get("lane_id") != lane_id:
            raise RuntimeError("expanded feature manifest lane order changed")
        common_path = _verify_lane_identity(root, lane["common"])
        derived_key = "dyad" if layer == "hero_opponent_interaction" else "line"
        derived_path = _verify_lane_identity(root, lane[derived_key])
        common = pd.read_parquet(common_path)
        derived = pd.read_parquet(derived_path)
        joined = common.merge(derived, on="row_id", how="inner", validate="one_to_one")
        if len(joined) != len(common) or len(joined) != len(derived):
            raise RuntimeError(f"{lane_id} common/derived row conservation failed")
        if layer == "hero_opponent_interaction":
            joined = joined.loc[joined["dyad_eligible"].eq(True)].copy()
        frames.append(joined)
        print(json.dumps({
            "event": "stage6_expanded_tournament_lane_loaded",
            "layer": layer,
            "lane": lane_number,
            "completed": lane_number + 1,
            "total": LOGICAL_LANE_COUNT,
            "eligible_rows": len(joined),
        }, sort_keys=True), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty or frame["row_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{layer} dataset is empty or duplicates rows")
    roles = set(frame["temporal_role"].astype(str))
    if not {"development_train", "development_validation"}.issubset(roles):
        raise RuntimeError(f"{layer} training/validation role is missing")
    if "fresh_adjudication" in roles:
        raise RuntimeError("fresh target entered broad tournament memory")
    feature_names, matrix = expanded_model_matrix(frame, layer)
    target = np.asarray([ACTIONS.index(value) for value in frame["action_label"].astype(str)], dtype=np.int8)
    legal_indices = [feature_names.index(f"base_legal_{action}") for action in ACTIONS]
    legal = matrix[:, legal_indices].astype(bool)
    if not legal[np.arange(len(target)), target].all():
        raise RuntimeError("observed action is illegal in expanded feature freeze")
    return {
        "feature_frame": frame,
        "frame": frame.loc[:, [
            "row_id", "game_id", "hand_id", "decision_utc", "temporal_role"
        ]].copy(),
        "matrix": matrix,
        "target": target,
        "legal": legal,
        "feature_names": feature_names,
        "feature_manifest_self_sha256": manifest["self_sha256"],
        "whole_hand_temporal_assignment_binding": manifest[
            "whole_hand_temporal_assignment_binding"
        ],
    }


def expanded_model_matrix(
    frame: pd.DataFrame,
    layer: str,
    *,
    include_dyad_interaction: bool = True,
) -> tuple[list[str], np.ndarray]:
    names = list(BASE_FEATURE_COLUMNS)
    parts = [frame[list(BASE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)]
    if layer == "hero_opponent_interaction":
        derived_names = dyad_main_effect_feature_columns()
        if include_dyad_interaction:
            derived_names = [*derived_names, *dyad_interaction_feature_columns()]
        parts.append(frame[derived_names].to_numpy(dtype=np.float32))
        names.extend(derived_names)
    elif layer == "within_hand_line_surprise":
        raw_names = line_numeric_feature_columns()
        for name in raw_names:
            if name.endswith("_code"):
                values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.int16)
                if not np.isin(values, np.arange(-1, len(ACTIONS))).all():
                    raise RuntimeError(f"invalid line action code: {name}")
                for action_index, action in enumerate(ACTIONS):
                    names.append(f"{name}__{action}")
                    parts.append((values == action_index).astype(np.float32).reshape(-1, 1))
                names.append(f"{name}__missing")
                parts.append((values == -1).astype(np.float32).reshape(-1, 1))
            else:
                names.append(name)
                parts.append(pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1))
    else:
        raise ValueError(f"unknown expanded layer: {layer}")
    assert_no_prohibited_model_features(names)
    matrix = np.column_stack(parts).astype(np.float32, copy=False)
    if matrix.shape != (len(frame), len(names)):
        raise RuntimeError("expanded model feature matrix shape changed")
    return names, matrix


def _group_sample_indices(
    frame: pd.DataFrame,
    role: str,
    row_cap: int,
    salt: str,
) -> np.ndarray:
    selected = frame.loc[frame["temporal_role"].astype(str).eq(role)].copy()
    if selected.empty:
        raise RuntimeError(f"expanded tournament role is empty: {role}")
    clusters = selected["game_id"].astype(str) + "\x1f" + selected["hand_id"].astype(str)
    group_sizes = clusters.value_counts(sort=False)
    ordered_groups = sorted(
        group_sizes.index,
        key=lambda value: hashlib.sha256(f"{salt}\x1f{value}".encode("utf-8")).hexdigest(),
    )
    chosen: list[str] = []
    rows = 0
    for group in ordered_groups:
        chosen.append(group)
        rows += int(group_sizes[group])
        if rows >= row_cap:
            break
    mask = clusters.isin(chosen)
    indices = selected.index[mask].to_numpy(dtype=np.int64)
    if not len(indices):
        raise RuntimeError("expanded tournament group sample is empty")
    return indices


def _impute_medians(train: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    return medians


def _apply_imputer(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    rows, columns = np.nonzero(~np.isfinite(result))
    if len(rows):
        result[rows, columns] = medians[columns]
    return result


def _stage_arrays(
    dataset: Mapping[str, Any],
    layer: str,
    shard: int,
    stage: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    frame = dataset["frame"]
    # Registry shards must be directly comparable, so both shards of a layer
    # use the identical nested hand sample at every rung.
    salt = f"{layer}:nested-rungs"
    train_indices = _group_sample_indices(
        frame, "development_train", int(stage["train_row_cap"]), salt + ":train"
    )
    validation_indices = _group_sample_indices(
        frame, "development_validation", int(stage["validation_row_cap"]), salt + ":validation"
    )
    train_x = np.asarray(dataset["matrix"])[train_indices]
    validation_x = np.asarray(dataset["matrix"])[validation_indices]
    medians = _impute_medians(train_x)
    return {
        "train_x": _apply_imputer(train_x, medians),
        "train_y": np.asarray(dataset["target"])[train_indices],
        "validation_x": _apply_imputer(validation_x, medians),
        "validation_y": np.asarray(dataset["target"])[validation_indices],
        "validation_legal": np.asarray(dataset["legal"])[validation_indices],
        "imputer_medians": medians,
        "train_row_ids": frame.iloc[train_indices]["row_id"].astype(str).to_numpy(dtype="U64"),
        "validation_row_ids": frame.iloc[validation_indices]["row_id"].astype(str).to_numpy(dtype="U64"),
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def stage_data_binding(
    arrays: Mapping[str, np.ndarray],
    whole_hand_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identities = {
        name: {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": _array_sha256(values),
        }
        for name, values in sorted(arrays.items())
    }
    return seal_document({
        "schema_version": "stage6_expanded_halving_stage_data_binding_v1",
        "arrays": identities,
        "train_row_root": identities["train_row_ids"]["sha256"],
        "validation_row_root": identities["validation_row_ids"]["sha256"],
        "whole_hand_temporal_assignment_self_sha256": (
            whole_hand_binding.get("self_sha256") if whole_hand_binding is not None else None
        ),
    })


def _write_stage_arrays(root: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, values in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        paths[name] = str(path)
    return paths


def _load_array(paths: Mapping[str, str], name: str) -> np.ndarray:
    return np.load(paths[name], mmap_mode="r", allow_pickle=False)


def _fit_candidate(
    entry: Mapping[str, Any],
    paths: Mapping[str, str],
    stage: Mapping[str, Any],
) -> Any:
    family = str(entry["family"])
    parameters = dict(entry["parameters"])
    train_x = _load_array(paths, "train_x")
    train_y = _load_array(paths, "train_y")
    validation_x = _load_array(paths, "validation_x")
    validation_y = _load_array(paths, "validation_y")
    iteration_cap = int(stage["iteration_cap"])
    patience = int(stage["patience"])
    tolerance = float(stage["minimum_meaningful_improvement"])
    seed = int(hashlib.sha256(str(entry["candidate_id"]).encode("utf-8")).hexdigest()[:8], 16)

    if family == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTIONS),
            n_estimators=iteration_cap,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
            verbosity=-1,
            **parameters,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            callbacks=[lgb.early_stopping(
                patience, verbose=False, min_delta=tolerance
            )],
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
            n_estimators=iteration_cap,
            random_state=seed,
            n_jobs=1,
            callbacks=[xgb.callback.EarlyStopping(
                rounds=patience,
                metric_name="mlogloss",
                data_name="validation_0",
                save_best=True,
                min_delta=tolerance,
            )],
            **parameters,
        )
        model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        return model
    if family == "catboost":
        from catboost import CatBoostClassifier

        meaningful_stopper = _CatBoostMeaningfulImprovementStopper(
            patience=patience,
            minimum_improvement=tolerance,
        )
        model = CatBoostClassifier(
            loss_function="MultiClass",
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
            raise RuntimeError("CatBoost meaningful-improvement callback selected no trees")
        if int(model.tree_count_) > meaningful_stopper.best_tree_count:
            model.shrink(ntree_end=meaningful_stopper.best_tree_count)
        return model
    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=iteration_cap,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=patience,
            tol=tolerance,
            **parameters,
        )
        model.fit(train_x, train_y)
        return model
    if family in {"extra_trees", "random_forest"}:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        constructor = ExtraTreesClassifier if family == "extra_trees" else RandomForestClassifier
        parameters["n_estimators"] = min(
            int(parameters["n_estimators"]),
            96 if int(stage["stage"]) == 1 else (256 if int(stage["stage"]) == 2 else 768),
        )
        model = constructor(
            random_state=seed,
            n_jobs=1,
            bootstrap=family == "random_forest",
            **parameters,
        )
        model.fit(train_x, train_y)
        return model
    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=100 if int(stage["stage"]) == 1 else (400 if int(stage["stage"]) == 2 else 1_200),
                tol=tolerance,
                random_state=seed,
                n_jobs=1,
                **parameters,
            ),
        )
        model.fit(train_x, train_y)
        return model
    if family == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), GaussianNB(**parameters))
        model.fit(train_x, train_y)
        return model
    raise RuntimeError(f"unknown expanded candidate family: {family}")


def _aligned_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    classes = [int(value) for value in getattr(estimator, "classes_", range(raw.shape[1]))]
    if set(classes) != set(range(len(ACTIONS))):
        raise RuntimeError(f"candidate class inventory changed: {classes}")
    positions = {value: index for index, value in enumerate(classes)}
    return np.column_stack([raw[:, positions[index]] for index in range(len(ACTIONS))])


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _evaluate_candidate(
    entry: Mapping[str, Any],
    paths: Mapping[str, str],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_info, threadpool_limits

    thread_limiter = threadpool_limits(limits=1)
    model = _fit_candidate(entry, paths, stage)
    validation_x = _load_array(paths, "validation_x")
    validation_y = np.asarray(_load_array(paths, "validation_y"), dtype=np.int8)
    legal = np.asarray(_load_array(paths, "validation_legal"), dtype=bool)
    train_y = np.asarray(_load_array(paths, "train_y"), dtype=np.int8)
    class_counts = np.bincount(train_y, minlength=len(ACTIONS)).astype(np.float64)
    priors = (class_counts + 1.0) / (class_counts.sum() + len(ACTIONS))
    raw = _aligned_probabilities(model, validation_x)
    projected, fallback_count, invalid_count = _project_legal(raw, legal, priors)
    if invalid_count:
        raise RuntimeError("candidate produced invalid probability rows")
    truth = [ACTIONS[int(value)] for value in validation_y]
    calibrated = []
    for temperature in TEMPERATURE_GRID:
        scaled = _temperature_scale(projected, temperature)
        scaled, fallback_after, invalid_after = _project_legal(scaled, legal, priors)
        if invalid_after:
            raise RuntimeError("temperature scaling produced invalid rows")
        metrics = multiclass_metric_report(truth, scaled, ACTIONS)
        calibrated.append((metrics["log_loss"], temperature, metrics, fallback_after))
    calibrated.sort(key=lambda row: (row[0], row[1]))
    _, temperature, metrics, fallback_after = calibrated[0]
    family = str(entry["family"])
    if family == "lightgbm":
        best_iteration = int(model.best_iteration_ or stage["iteration_cap"])
    elif family == "xgboost":
        raw_iteration = getattr(model, "best_iteration", None)
        best_iteration = int(raw_iteration) + 1 if raw_iteration is not None else int(stage["iteration_cap"])
    elif family == "catboost":
        best_iteration = int(model.tree_count_)
    elif family == "hist_gradient_boosting":
        best_iteration = int(model.n_iter_)
    elif family in {"extra_trees", "random_forest"}:
        best_iteration = int(model.n_estimators)
    elif family == "logistic_regression":
        best_iteration = int(np.asarray(model.steps[-1][1].n_iter_).max())
    else:
        best_iteration = 1
    observed_threadpools = [
        {
            "internal_api": str(row.get("internal_api", "unknown")),
            "prefix": str(row.get("prefix", "unknown")),
            "num_threads": int(row.get("num_threads", 0)),
        }
        for row in threadpool_info()
    ]
    if any(row["num_threads"] > 1 for row in observed_threadpools):
        raise RuntimeError("candidate native thread pool exceeded one thread")
    return {
        "candidate_id": entry["candidate_id"],
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "parameters": entry["parameters"],
        "stage": int(stage["stage"]),
        "train_row_count": len(train_y),
        "validation_row_count": len(validation_y),
        "selected_temperature": float(temperature),
        "validation_metrics": metrics,
        "legal_projection_fallback_count": int(fallback_count + fallback_after),
        "best_iteration": best_iteration,
        "fit_and_evaluate_seconds": time.monotonic() - started,
        "candidate_internal_threads": 1,
        "observed_threadpool_limits": observed_threadpools,
        "threadpoolctl_limit_applied": thread_limiter is not None,
        "random_seed": int(hashlib.sha256(str(entry["candidate_id"]).encode("utf-8")).hexdigest()[:8], 16),
    }


def _worker(job: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_candidate(job["entry"], job["paths"], job["stage"])


def _candidate_model_peak_bytes(
    entry: Mapping[str, Any],
    stage: Mapping[str, Any],
    train_rows: int,
) -> int:
    family = str(entry["family"])
    parameters = dict(entry["parameters"])
    iteration_cap = int(stage["iteration_cap"])
    if family in {"extra_trees", "random_forest"}:
        stage_number = int(stage["stage"])
        estimator_cap = 96 if stage_number == 1 else (256 if stage_number == 2 else 768)
        estimators = min(int(parameters["n_estimators"]), estimator_cap)
        minimum_leaf = max(1, int(parameters["min_samples_leaf"]))
        depth = int(parameters["max_depth"])
        maximum_leaves = min(2**depth, max(1, math.ceil(train_rows / minimum_leaf)))
        maximum_nodes = 2 * maximum_leaves - 1
        bytes_per_node_with_six_class_values = 112
        return int(1.60 * estimators * maximum_nodes * bytes_per_node_with_six_class_values)
    if family == "catboost":
        leaves = 2 ** int(parameters["depth"])
        return int(2.0 * iteration_cap * leaves * len(ACTIONS) * 8)
    if family in {"lightgbm", "xgboost", "hist_gradient_boosting"}:
        leaves = int(parameters.get("num_leaves", parameters.get("max_leaf_nodes", 127)))
        if family == "xgboost":
            leaves = 2 ** int(parameters["max_depth"])
        return int(2.0 * iteration_cap * (2 * leaves - 1) * 128)
    return 250_000_000


def resolve_candidate_workers(
    requested: str,
    unfinished_candidates: int,
    arrays: Mapping[str, np.ndarray],
    entries: Sequence[Mapping[str, Any]] | None = None,
    stage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = str(requested).strip()
    cpu = max(1, int(os.cpu_count() or 1))
    if raw.lower() == "auto" or raw in {"", "0"}:
        requested_capacity = cpu
        mode = "auto"
    elif raw.isascii() and raw.isdecimal() and int(raw) > 0:
        requested_capacity = int(raw)
        mode = "explicit"
    else:
        raise RuntimeError("REQUESTED_WORKERS must be auto, blank, 0, or a positive integer")
    import psutil

    available_ram = int(psutil.virtual_memory().available)
    matrix_bytes = int(arrays["train_x"].nbytes + arrays["validation_x"].nbytes)
    candidate_peaks = [
        _candidate_model_peak_bytes(entry, stage, len(arrays["train_x"]))
        for entry in (entries or ())
    ] if stage is not None else []
    estimated_model_peak = max(candidate_peaks, default=0)
    estimated_worker_peak = max(
        MINIMUM_WORKER_PEAK_BYTES,
        int(matrix_bytes * 2.75) + estimated_model_peak,
    )
    if estimated_worker_peak > available_ram - RAM_RESERVE_BYTES:
        raise RuntimeError(
            "one candidate cannot fit the RAM-bound peak estimate with the required reserve"
        )
    ram_capacity = max(1, (available_ram - RAM_RESERVE_BYTES) // estimated_worker_peak)
    live = max(1, min(requested_capacity, cpu, int(ram_capacity), unfinished_candidates))
    return {
        "requested_workers": raw or "auto",
        "requested_workers_mode": mode,
        "detected_usable_logical_cpus": cpu,
        "available_system_ram_bytes": available_ram,
        "estimated_worker_peak_bytes": estimated_worker_peak,
        "estimated_model_peak_bytes": estimated_model_peak,
        "ram_bounded_worker_capacity": int(ram_capacity),
        "unfinished_candidates": unfinished_candidates,
        "live_candidate_workers": live,
        "candidate_internal_threads": 1,
        "logical_data_lane_count": LOGICAL_LANE_COUNT,
        "logical_lane_count_changed": False,
    }


def _checkpoint_path(root: Path, stage_number: int, candidate_id: str) -> Path:
    return root / "candidate_results" / f"stage-{stage_number}" / f"{candidate_id}.json"


def _load_checkpoint(
    path: Path,
    run_id: str,
    stage_number: int,
    entry: Mapping[str, Any],
    stage_binding_self_sha256: str,
    environment_self_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    if (
        document.get("schema_version") != "stage6_expanded_broad_candidate_result_v1"
        or document.get("status") != "PASS_CANDIDATE_STAGE_COMPLETE"
        or document.get("broad_run_id") != run_id
        or document.get("stage") != stage_number
        or document.get("candidate_id") != entry["candidate_id"]
        or document.get("configuration_sha256") != entry["configuration_sha256"]
        or document.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or document.get("stage_data_binding_self_sha256") != stage_binding_self_sha256
        or document.get("environment_self_sha256") != environment_self_sha256
        or document.get("candidate_internal_threads") != 1
    ):
        raise RuntimeError(f"candidate checkpoint does not bind this task: {path}")
    return document


def runtime_environment() -> dict[str, Any]:
    import platform
    import sys

    versions = {}
    expected = {}
    for distribution, module_name, version in PINNED_RUNTIME_DISTRIBUTIONS:
        module = __import__(module_name)
        versions[distribution] = str(getattr(module, "__version__", "unknown"))
        expected[distribution] = version
    if versions != expected:
        raise RuntimeError(f"expanded-layer library stack is not pinned: {versions}")
    library_stack = seal_document({
        "schema_version": "stage6_expanded_pinned_library_stack_v1",
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "libraries": versions,
    })
    return seal_document({
        "schema_version": "stage6_expanded_tournament_environment_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": versions,
        "library_stack": library_stack,
        "library_stack_self_sha256": library_stack["self_sha256"],
        "candidate_internal_threads": 1,
    })


def _rank(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        reports,
        key=lambda row: (
            float(row["validation_metrics"]["log_loss"]),
            float(row["validation_metrics"]["multiclass_brier_score"]),
            str(row["candidate_id"]),
        ),
    )


def _completed_manifest(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root / "broad-tournament-manifest.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    if (
        document.get("schema_version") != "stage6_expanded_broad_tournament_manifest_v1"
        or document.get("status") != "PASS_BROAD_SUCCESSIVE_HALVING_COMPLETE"
        or document.get("broad_run_id") != run_id
        or document.get("fresh_targets_opened") is not False
        or document.get("qualification_authority") is not False
    ):
        raise RuntimeError("completed broad tournament manifest is invalid")
    return document


def run_broad_tournament(
    stage_drive_root: Path,
    layer: str,
    shard: int,
    requested_workers: str = "auto",
) -> dict[str, Any]:
    run_id = broad_run_id(layer, shard)
    work_root = broad_work_root(stage_drive_root, layer, shard)
    completed = _completed_manifest(work_root, run_id)
    if completed is not None:
        return {
            "status": "PASS",
            "broad_run_id": run_id,
            "manifest_path": str(work_root / "broad-tournament-manifest.json"),
            "manifest_self_sha256": completed["self_sha256"],
            "terminal_survivor_ids": completed["terminal_survivor_ids"],
            "reused_completed_result": True,
            "qualification_authority": False,
        }

    started = time.monotonic()
    dataset = load_layer_dataset(stage_drive_root, layer)
    environment = runtime_environment()
    registry_by_id = {
        row["candidate_id"]: row for row in candidate_registry_shard(layer, shard)
    }
    active_ids = list(registry_by_id)
    stage_summaries = []
    worker_reports = []
    for stage in HALVING_STAGES:
        stage_number = int(stage["stage"])
        arrays = _stage_arrays(dataset, layer, shard, stage)
        data_binding = stage_data_binding(
            arrays, dataset["whole_hand_temporal_assignment_binding"]
        )
        reports: list[dict[str, Any]] = []
        unfinished = []
        for candidate_id in active_ids:
            entry = registry_by_id[candidate_id]
            checkpoint = _load_checkpoint(
                _checkpoint_path(work_root, stage_number, candidate_id),
                run_id,
                stage_number,
                entry,
                data_binding["self_sha256"],
                environment["self_sha256"],
            )
            if checkpoint is None:
                unfinished.append(entry)
            else:
                reports.append(checkpoint)

        if unfinished:
            worker_report = resolve_candidate_workers(
                requested_workers, len(unfinished), arrays, unfinished, stage
            )
            worker_reports.append({"stage": stage_number, **worker_report})
            print(json.dumps({
                "event": "stage6_expanded_halving_stage_started",
                "layer": layer,
                "shard": shard,
                "stage": stage_number,
                "candidates_total": len(active_ids),
                "candidates_already_complete": len(reports),
                **worker_report,
            }, sort_keys=True), flush=True)
            with tempfile.TemporaryDirectory(prefix=f"stage6-expanded-stage-{stage_number}-") as temporary:
                paths = _write_stage_arrays(Path(temporary), arrays)
                pending = list(unfinished)
                attempts = {str(row["candidate_id"]): 0 for row in pending}
                progress = _PhaseProgress(
                    phase=f"broad_search_stage_{stage_number}",
                    completed=len(reports),
                    denominator=len(active_ids),
                    live_workers=int(worker_report["live_candidate_workers"]),
                    checkpoint=str(work_root / "candidate_results" / f"stage-{stage_number}"),
                    context={"layer": layer, "shard": shard, "stage": stage_number},
                )
                progress.emit(force=True)
                while pending:
                    retry: list[Mapping[str, Any]] = []
                    with ProcessPoolExecutor(max_workers=int(worker_report["live_candidate_workers"])) as executor:
                        futures = {}
                        for entry in pending:
                            candidate_id = str(entry["candidate_id"])
                            attempts[candidate_id] += 1
                            future = executor.submit(_worker, {
                                "entry": entry,
                                "paths": paths,
                                "stage": stage,
                            })
                            futures[future] = entry
                        remaining_futures = set(futures)
                        while remaining_futures:
                            completed_futures, remaining_futures = wait(
                                remaining_futures,
                                timeout=progress.seconds_until_heartbeat(),
                                return_when=FIRST_COMPLETED,
                            )
                            if not completed_futures:
                                progress.emit()
                                continue
                            for future in sorted(
                                completed_futures,
                                key=lambda item: str(futures[item]["candidate_id"]),
                            ):
                                entry = futures[future]
                                candidate_id = str(entry["candidate_id"])
                                try:
                                    payload = future.result()
                                except Exception as error:
                                    print(json.dumps({
                                        "event": "stage6_expanded_candidate_attempt_failed",
                                        "layer": layer,
                                        "shard": shard,
                                        "stage": stage_number,
                                        "candidate_id": candidate_id,
                                        "attempt": attempts[candidate_id],
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                    }, sort_keys=True), flush=True)
                                    if attempts[candidate_id] < MAX_CANDIDATE_ATTEMPTS:
                                        retry.append(entry)
                                        continue
                                    raise RuntimeError(
                                        f"candidate {candidate_id} failed {MAX_CANDIDATE_ATTEMPTS} attempts"
                                    ) from error
                                report = seal_document({
                                    "schema_version": "stage6_expanded_broad_candidate_result_v1",
                                    "status": "PASS_CANDIDATE_STAGE_COMPLETE",
                                    "broad_run_id": run_id,
                                    "feature_run_id": FEATURE_RUN_ID,
                                    "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
                                    "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
                                    "layer": layer,
                                    "shard": shard,
                                    **payload,
                                    "stage_data_binding_self_sha256": data_binding["self_sha256"],
                                    "environment_self_sha256": environment["self_sha256"],
                                    "attempt": attempts[candidate_id],
                                    "fresh_targets_opened": False,
                                    "qualification_authority": False,
                                    "assistant_installation_allowed": False,
                                })
                                checkpoint_path = _checkpoint_path(
                                    work_root, stage_number, candidate_id
                                )
                                _atomic_json(checkpoint_path, report)
                                check = _load_checkpoint(
                                    checkpoint_path,
                                    run_id,
                                    stage_number,
                                    entry,
                                    data_binding["self_sha256"],
                                    environment["self_sha256"],
                                )
                                if check != report:
                                    raise RuntimeError("candidate checkpoint readback changed")
                                reports.append(report)
                                progress.completed_one(
                                    checkpoint_path,
                                    event="stage6_expanded_candidate_complete",
                                    extra={
                                        "candidate_id": candidate_id,
                                        "validation_log_loss": report["validation_metrics"]["log_loss"],
                                        "best_iteration": report["best_iteration"],
                                    },
                                )
                    pending = retry

        if len(reports) != len(active_ids):
            raise RuntimeError("expanded halving stage did not conserve candidates")
        ranked = _rank(reports)
        survivor_count = min(int(stage["survivor_count"]), len(ranked))
        active_ids = [str(row["candidate_id"]) for row in ranked[:survivor_count]]
        stage_summaries.append(seal_document({
            "schema_version": "stage6_expanded_halving_stage_summary_v1",
            "broad_run_id": run_id,
            "layer": layer,
            "shard": shard,
            "stage": stage_number,
            "stage_parameters": dict(stage),
            "stage_data_binding": data_binding,
            "environment_self_sha256": environment["self_sha256"],
            "evaluated_candidate_count": len(ranked),
            "survivor_count": survivor_count,
            "survivor_ids": active_ids,
            "ranked_results": [
                {
                    "rank": index + 1,
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "configuration_sha256": row["configuration_sha256"],
                    "validation_log_loss": row["validation_metrics"]["log_loss"],
                    "validation_brier": row["validation_metrics"]["multiclass_brier_score"],
                    "selected_temperature": row["selected_temperature"],
                    "best_iteration": row["best_iteration"],
                    "checkpoint_self_sha256": row["self_sha256"],
                    "checkpoint_bytes": _checkpoint_path(
                        work_root, stage_number, str(row["candidate_id"])
                    ).stat().st_size,
                    "checkpoint_sha256": sha256_file(_checkpoint_path(
                        work_root, stage_number, str(row["candidate_id"])
                    )),
                }
                for index, row in enumerate(ranked)
            ],
            "fresh_targets_opened": False,
            "qualification_authority": False,
        }))
        _atomic_json(
            work_root / f"stage-{stage_number}-summary.json",
            stage_summaries[-1],
        )

    terminal_reports = stage_summaries[-1]["ranked_results"]
    manifest = seal_document({
        "schema_version": "stage6_expanded_broad_tournament_manifest_v1",
        "status": "PASS_BROAD_SUCCESSIVE_HALVING_COMPLETE",
        "broad_run_id": run_id,
        "feature_run_id": FEATURE_RUN_ID,
        "feature_manifest_self_sha256": dataset["feature_manifest_self_sha256"],
        "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "layer": layer,
        "shard": shard,
        "candidate_registry_digest": hashlib.sha256(
            canonical_json_bytes(candidate_registry(layer))
        ).hexdigest(),
        "candidate_shard_digest": shard_digest(layer, shard),
        "base_candidate_count": 128,
        "halving_stage_summaries": [row["self_sha256"] for row in stage_summaries],
        "halving_stage_data_bindings": [row["stage_data_binding"] for row in stage_summaries],
        "terminal_survivor_ids": [str(row["candidate_id"]) for row in terminal_reports],
        "terminal_ranked_results": terminal_reports,
        "worker_reports": worker_reports,
        "environment": environment,
        "candidate_internal_threads": 1,
        "logical_data_lane_count": LOGICAL_LANE_COUNT,
        "logical_lane_count_changed": False,
        "action_timing_layer_included": False,
        "fresh_targets_opened": False,
        "current_corpus_authority": "development_and_retrospective_only",
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "action_changing_exploit_authority": False,
        "elapsed_seconds": time.monotonic() - started,
    })
    work_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(work_root / "broad-tournament-manifest.json", manifest)
    return {
        "status": "PASS",
        "broad_run_id": run_id,
        "manifest_path": str(work_root / "broad-tournament-manifest.json"),
        "manifest_self_sha256": manifest["self_sha256"],
        "terminal_survivor_ids": manifest["terminal_survivor_ids"],
        "reused_completed_result": False,
        "qualification_authority": False,
    }


def _available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None
