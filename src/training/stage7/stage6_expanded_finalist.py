"""Full-development finalist training and post-refit calibration for Stage 6."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from stage6_expanded_feature_freeze import BASE_FEATURE_COLUMNS, FEATURE_RUN_ID
from stage6_expanded_feature_freeze_v2 import _recover_claimed_json_if_needed
from stage6_expanded_layers import (
    ACTIONS,
    EXPANDED_LAYER_CONTRACT_DIGEST,
    SCIENCE_IMPLEMENTATION_DIGEST,
    candidate_registry,
    canonical_json_bytes,
    seal_document,
    validate_document,
)
from stage6_expanded_tournament import (
    TEMPERATURE_GRID,
    _candidate_model_peak_bytes,
    _aligned_probabilities,
    _apply_imputer,
    _atomic_json,
    _evaluate_candidate,
    _impute_medians,
    _project_legal,
    _PhaseProgress,
    _rank,
    _temperature_scale,
    _worker,
    _write_stage_arrays,
    broad_run_id,
    broad_work_root,
    expanded_model_matrix,
    load_layer_dataset,
    resolve_candidate_workers,
    runtime_environment,
    stage_data_binding,
)
from stage6_metrics import multiclass_metric_report
from stage6_notebooks02_07_runtime import sha256_file


FINALIST_COUNT_PER_SHARD = 12
FINALIST_ITERATION_CAP = 12_000
FINALIST_PATIENCE = 250
FINALIST_MINIMUM_IMPROVEMENT = 1e-5
DYAD_MINIMUM_HAND_SUPPORT = 2
LINE_MINIMUM_PREFIX_ACTIONS = 1


def finalist_run_id(layer: str) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_layer_finalist_v1",
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation": SCIENCE_IMPLEMENTATION_DIGEST,
        "layer": layer,
        "broad_run_ids": [broad_run_id(layer, shard) for shard in (0, 1)],
        "finalist_count_per_shard": FINALIST_COUNT_PER_SHARD,
        "iteration_cap": FINALIST_ITERATION_CAP,
        "patience": FINALIST_PATIENCE,
        "minimum_improvement": FINALIST_MINIMUM_IMPROVEMENT,
        "calibration_role": "retrospective_calibration",
        "candidate_internal_threads": 1,
    })).hexdigest()


def finalist_work_root(stage_drive_root: Path, layer: str) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "finalists"
        / finalist_run_id(layer)
    )


def _broad_finalists(stage_drive_root: Path, layer: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = {row["candidate_id"]: row for row in candidate_registry(layer)}
    selected: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for shard in (0, 1):
        path = broad_work_root(stage_drive_root, layer, shard) / "broad-tournament-manifest.json"
        if not path.is_file():
            raise RuntimeError(f"broad shard {shard} must complete before {layer} finalists")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_document(manifest)
        if (
            manifest.get("schema_version") != "stage6_expanded_broad_tournament_manifest_v1"
            or manifest.get("status") != "PASS_BROAD_SUCCESSIVE_HALVING_COMPLETE"
            or manifest.get("broad_run_id") != broad_run_id(layer, shard)
            or manifest.get("layer") != layer
            or manifest.get("shard") != shard
            or manifest.get("science_implementation_self_sha256")
            != SCIENCE_IMPLEMENTATION_DIGEST
            or manifest.get("fresh_targets_opened") is not False
        ):
            raise RuntimeError(f"broad shard {shard} manifest is invalid")
        manifests.append(manifest)
        ranked = manifest["terminal_ranked_results"][:FINALIST_COUNT_PER_SHARD]
        for row in ranked:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in registry:
                raise RuntimeError("broad finalist is outside the frozen registry")
            selected.append(registry[candidate_id])
    if len({manifest["feature_manifest_self_sha256"] for manifest in manifests}) != 1:
        raise RuntimeError("broad finalist shards used different feature manifests")
    if manifests[0]["halving_stage_data_bindings"] != manifests[1]["halving_stage_data_bindings"]:
        raise RuntimeError("broad finalist shards used different rung data roots")
    if len({manifest["environment"]["library_stack_self_sha256"] for manifest in manifests}) != 1:
        raise RuntimeError("broad finalist shards used different pinned library stacks")
    if len(selected) != 2 * FINALIST_COUNT_PER_SHARD:
        raise RuntimeError("finalist inventory changed")
    return selected, manifests


def _full_selection_arrays(dataset: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frame = dataset["frame"]
    train_indices = frame.index[frame["temporal_role"].eq("development_train")].to_numpy(dtype=np.int64)
    validation_indices = frame.index[
        frame["temporal_role"].eq("development_validation")
    ].to_numpy(dtype=np.int64)
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("full finalist train/validation split is empty")
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


def _checkpoint_path(root: Path, candidate_id: str) -> Path:
    return root / "candidate_results" / f"{candidate_id}.json"


def _load_checkpoint(
    path: Path,
    run_id: str,
    entry: Mapping[str, Any],
    data_binding: str,
    environment: str,
) -> dict[str, Any] | None:
    _recover_claimed_json_if_needed(path, path.parent)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    if (
        document.get("schema_version") != "stage6_expanded_finalist_candidate_result_v1"
        or document.get("status") != "PASS_FINALIST_SELECTION_COMPLETE"
        or document.get("finalist_run_id") != run_id
        or document.get("candidate_id") != entry["candidate_id"]
        or document.get("configuration_sha256") != entry["configuration_sha256"]
        or document.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or document.get("stage_data_binding_self_sha256") != data_binding
        or document.get("environment_self_sha256") != environment
        or document.get("candidate_internal_threads") != 1
    ):
        raise RuntimeError("finalist checkpoint is not bound to this run")
    return document


def _fit_fixed_candidate(
    entry: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    iterations: int,
) -> Any:
    family = str(entry["family"])
    parameters = dict(entry["parameters"])
    seed = int(hashlib.sha256(str(entry["candidate_id"]).encode("utf-8")).hexdigest()[:8], 16)
    iterations = max(1, int(iterations))
    if family == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTIONS),
            n_estimators=iterations,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
            verbosity=-1,
            **parameters,
        )
    elif family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(ACTIONS),
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=iterations,
            random_state=seed,
            n_jobs=1,
            **parameters,
        )
    elif family == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=iterations,
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **parameters,
        )
    elif family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=iterations,
            random_state=seed,
            early_stopping=False,
            **parameters,
        )
    elif family in {"extra_trees", "random_forest"}:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        constructor = ExtraTreesClassifier if family == "extra_trees" else RandomForestClassifier
        parameters["n_estimators"] = iterations
        model = constructor(
            random_state=seed,
            n_jobs=1,
            bootstrap=family == "random_forest",
            **parameters,
        )
    elif family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=iterations,
                tol=FINALIST_MINIMUM_IMPROVEMENT,
                random_state=seed,
                n_jobs=1,
                **parameters,
            ),
        )
    elif family == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), GaussianNB(**parameters))
    else:
        raise RuntimeError(f"unknown finalist family: {family}")
    model.fit(train_x, train_y)
    return model


def _calibrated_report(
    model: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    raw = _aligned_probabilities(model, matrix)
    projected, _, invalid = _project_legal(raw, legal, priors)
    if invalid:
        raise RuntimeError("finalist model produced invalid calibration probabilities")
    truth = [ACTIONS[int(value)] for value in target]
    rows = []
    for temperature in TEMPERATURE_GRID:
        scaled = _temperature_scale(projected, temperature)
        scaled, _, invalid = _project_legal(scaled, legal, priors)
        if invalid:
            raise RuntimeError("finalist calibration produced invalid probabilities")
        rows.append((
            multiclass_metric_report(truth, scaled, ACTIONS)["log_loss"],
            temperature,
            scaled,
        ))
    rows.sort(key=lambda row: (row[0], row[1]))
    _, temperature, probabilities = rows[0]
    return float(temperature), probabilities, multiclass_metric_report(
        truth, probabilities, ACTIONS
    )


def _stratum_report(
    target: np.ndarray,
    selected: np.ndarray,
    control: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"row_count": 0, "selected": None, "control": None}
    truth = [ACTIONS[int(value)] for value in target[mask]]
    return {
        "row_count": count,
        "selected": multiclass_metric_report(truth, selected[mask], ACTIONS),
        "control": multiclass_metric_report(truth, control[mask], ACTIONS),
    }


def _publish_artifact(
    root: Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage6-expanded-finalist-") as temporary:
        local = Path(temporary) / "artifact.joblib"
        joblib.dump(dict(artifact), local, compress=3)
        digest = sha256_file(local)
        destination = root / f"{artifact['layer']}-{digest[:24]}.joblib"
        if destination.exists():
            if destination.stat().st_size != local.stat().st_size or sha256_file(destination) != digest:
                raise RuntimeError("finalist artifact destination conflicts")
        else:
            shutil.copyfile(local, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError("finalist artifact publication changed bytes")
    return {
        "name": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def _completed_manifest(root: Path, run_id: str) -> dict[str, Any] | None:
    path = root / "finalist-manifest.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    if (
        document.get("schema_version") != "stage6_expanded_finalist_manifest_v1"
        or document.get("status") != "PASS_FINALIST_FROZEN_RETROSPECTIVE_ONLY"
        or document.get("finalist_run_id") != run_id
        or document.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or document.get("fresh_targets_opened") is not False
        or document.get("qualification_authority") is not False
    ):
        raise RuntimeError("completed finalist manifest is invalid")
    artifact = root / "artifacts" / document["selected_artifact"]["name"]
    if (
        not artifact.is_file()
        or artifact.stat().st_size != document["selected_artifact"]["bytes"]
        or sha256_file(artifact) != document["selected_artifact"]["sha256"]
    ):
        raise RuntimeError("completed finalist artifact identity changed")
    return document


def run_finalists(
    stage_drive_root: Path,
    layer: str,
    requested_workers: str = "auto",
) -> dict[str, Any]:
    run_id = finalist_run_id(layer)
    root = finalist_work_root(stage_drive_root, layer)
    completed = _completed_manifest(root, run_id)
    if completed is not None:
        return {
            "status": "PASS",
            "finalist_run_id": run_id,
            "manifest_path": str(root / "finalist-manifest.json"),
            "manifest_self_sha256": completed["self_sha256"],
            "selected_candidate_id": completed["selected_candidate_id"],
            "selected_artifact": completed["selected_artifact"],
            "reused_completed_result": True,
            "qualification_authority": False,
        }

    started = time.monotonic()
    finalists, broad_manifests = _broad_finalists(stage_drive_root, layer)
    dataset = load_layer_dataset(stage_drive_root, layer)
    arrays = _full_selection_arrays(dataset)
    binding = stage_data_binding(
        arrays, dataset["whole_hand_temporal_assignment_binding"]
    )
    environment = runtime_environment()
    stage = {
        "stage": 4,
        "iteration_cap": FINALIST_ITERATION_CAP,
        "patience": FINALIST_PATIENCE,
        "minimum_meaningful_improvement": FINALIST_MINIMUM_IMPROVEMENT,
    }
    reports = []
    unfinished = []
    for entry in finalists:
        checkpoint = _load_checkpoint(
            _checkpoint_path(root, str(entry["candidate_id"])),
            run_id,
            entry,
            binding["self_sha256"],
            environment["self_sha256"],
        )
        if checkpoint is None:
            unfinished.append(entry)
        else:
            reports.append(checkpoint)

    worker_report = None
    if unfinished:
        worker_report = resolve_candidate_workers(
            requested_workers, len(unfinished), arrays, unfinished, stage
        )
        print(json.dumps({
            "event": "stage6_expanded_finalists_started",
            "layer": layer,
            "completed": len(reports),
            "total": len(finalists),
            **worker_report,
        }, sort_keys=True), flush=True)
        with tempfile.TemporaryDirectory(prefix="stage6-expanded-finalists-") as temporary:
            paths = _write_stage_arrays(Path(temporary), arrays)
            with ProcessPoolExecutor(max_workers=int(worker_report["live_candidate_workers"])) as executor:
                futures = {
                    executor.submit(_worker, {"entry": entry, "paths": paths, "stage": stage}): entry
                    for entry in unfinished
                }
                progress = _PhaseProgress(
                    phase="full_finalist_selection",
                    completed=len(reports),
                    denominator=len(finalists),
                    live_workers=int(worker_report["live_candidate_workers"]),
                    checkpoint=str(root / "candidate_results"),
                    context={"layer": layer},
                )
                progress.emit(force=True)
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
                        payload = future.result()
                        report = seal_document({
                            "schema_version": "stage6_expanded_finalist_candidate_result_v1",
                            "status": "PASS_FINALIST_SELECTION_COMPLETE",
                            "finalist_run_id": run_id,
                            "feature_run_id": FEATURE_RUN_ID,
                            "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
                            "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
                            "layer": layer,
                            **payload,
                            "stage_data_binding_self_sha256": binding["self_sha256"],
                            "environment_self_sha256": environment["self_sha256"],
                            "fresh_targets_opened": False,
                            "qualification_authority": False,
                        })
                        checkpoint_path = _checkpoint_path(root, str(entry["candidate_id"]))
                        _atomic_json(checkpoint_path, report)
                        reports.append(report)
                        progress.completed_one(
                            checkpoint_path,
                            event="stage6_expanded_finalist_candidate_complete",
                            extra={
                                "candidate_id": entry["candidate_id"],
                                "validation_log_loss": report["validation_metrics"]["log_loss"],
                                "selected_iterations": report["best_iteration"],
                            },
                        )
    if len(reports) != len(finalists):
        raise RuntimeError("finalist candidate inventory was not conserved")
    ranked = _rank(reports)
    winner_report = ranked[0]
    winner = next(
        row for row in finalists if row["candidate_id"] == winner_report["candidate_id"]
    )

    frame = dataset["frame"]
    feature_frame = dataset["feature_frame"]
    refit_indices = frame.index[frame["temporal_role"].isin(
        ["development_train", "development_validation"]
    )].to_numpy(dtype=np.int64)
    calibration_indices = frame.index[
        frame["temporal_role"].eq("retrospective_calibration")
    ].to_numpy(dtype=np.int64)
    if not len(refit_indices) or not len(calibration_indices):
        raise RuntimeError("reserved retrospective calibration slice is empty")

    selected_x = np.asarray(dataset["matrix"])
    if layer == "hero_opponent_interaction":
        control_names, control_x = expanded_model_matrix(
            feature_frame, layer, include_dyad_interaction=False
        )
    else:
        control_names = list(BASE_FEATURE_COLUMNS)
        control_x = feature_frame[control_names].to_numpy(dtype=np.float32)
    selected_medians = _impute_medians(selected_x[refit_indices])
    control_medians = _impute_medians(control_x[refit_indices])
    selected_train = _apply_imputer(selected_x[refit_indices], selected_medians)
    control_train = _apply_imputer(control_x[refit_indices], control_medians)
    target = np.asarray(dataset["target"])
    iterations = int(winner_report["best_iteration"] or 1)
    import psutil

    refit_model_peak = 2 * _candidate_model_peak_bytes(
        winner,
        {"stage": 5, "iteration_cap": iterations},
        len(refit_indices),
    )
    refit_matrix_peak = int(
        selected_train.nbytes
        + control_train.nbytes
        + selected_x[calibration_indices].nbytes
        + control_x[calibration_indices].nbytes
    )
    refit_available_ram = int(psutil.virtual_memory().available)
    if refit_model_peak + refit_matrix_peak > refit_available_ram - 6_000_000_000:
        raise RuntimeError("selected/control final refits cannot fit the RAM-bound peak estimate")
    legal = np.asarray(dataset["legal"])[calibration_indices]
    class_counts = np.bincount(target[refit_indices], minlength=len(ACTIONS)).astype(np.float64)
    priors = (class_counts + 1.0) / (class_counts.sum() + len(ACTIONS))
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=1):
        selected_model = _fit_fixed_candidate(
            winner, selected_train, target[refit_indices], iterations
        )
        control_model = _fit_fixed_candidate(
            winner, control_train, target[refit_indices], iterations
        )
        selected_temperature, selected_probabilities, selected_metrics = _calibrated_report(
            selected_model,
            _apply_imputer(selected_x[calibration_indices], selected_medians),
            target[calibration_indices],
            legal,
            priors,
        )
        control_temperature, control_probabilities, control_metrics = _calibrated_report(
            control_model,
            _apply_imputer(control_x[calibration_indices], control_medians),
            target[calibration_indices],
            legal,
            priors,
        )
        refit_threadpool_receipt = [
            {
                "internal_api": str(row.get("internal_api", "unknown")),
                "prefix": str(row.get("prefix", "unknown")),
                "num_threads": int(row.get("num_threads", 0)),
            }
            for row in threadpool_info()
        ]
        if any(row["num_threads"] > 1 for row in refit_threadpool_receipt):
            raise RuntimeError("final refit native thread pool exceeded one thread")

    strata = None
    thresholds: dict[str, Any]
    if layer == "hero_opponent_interaction":
        calibration_features = feature_frame.iloc[calibration_indices]
        pair_support = calibration_features["dyad_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64)
        actor_support = calibration_features["main_actor_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64)
        counterpart_support = calibration_features["main_counterpart_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64)
        known = pair_support >= DYAD_MINIMUM_HAND_SUPPORT
        unseen_known_players = (pair_support == 0) & (actor_support > 0) & (counterpart_support > 0)
        strata = {
            "known_directed_dyad": _stratum_report(
                target[calibration_indices], selected_probabilities, control_probabilities, known
            ),
            "unseen_directed_dyad_with_known_players": _stratum_report(
                target[calibration_indices], selected_probabilities, control_probabilities, unseen_known_players
            ),
        }
        thresholds = {
            "minimum_directed_dyad_prior_hand_count": DYAD_MINIMUM_HAND_SUPPORT,
            "below_support_behavior": "abstain_from_interaction_and_return_matched_control",
        }
    else:
        calibration_features = feature_frame.iloc[calibration_indices]
        prefix = calibration_features["line_prefix_decision_count"].to_numpy(dtype=np.int64)
        true_probability = selected_probabilities[
            np.arange(len(calibration_indices)), target[calibration_indices]
        ]
        surprise = -np.log(np.clip(true_probability, 1e-15, 1.0))
        thresholds = {
            "minimum_prior_public_action_count": LINE_MINIMUM_PREFIX_ACTIONS,
            "surprise_review_threshold_calibration_quantile": 0.95,
            "surprise_review_threshold": float(np.quantile(
                surprise[prefix >= LINE_MINIMUM_PREFIX_ACTIONS], 0.95
            )),
            "no_prefix_behavior": "abstain_from_line_signal_and_return_matched_control",
        }

    artifact = {
        "schema_version": "stage6_expanded_predictive_layer_artifact_v1",
        "finalist_run_id": run_id,
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "layer": layer,
        "candidate_id": winner["candidate_id"],
        "family": winner["family"],
        "parameters": winner["parameters"],
        "configuration_sha256": winner["configuration_sha256"],
        "selected_iterations": iterations,
        "selected_model": selected_model,
        "control_model": control_model,
        "selected_feature_names": list(dataset["feature_names"]),
        "control_feature_names": control_names,
        "selected_imputer_medians": selected_medians,
        "control_imputer_medians": control_medians,
        "class_priors": priors,
        "selected_temperature": selected_temperature,
        "control_temperature": control_temperature,
        "refit_threadpool_receipt": refit_threadpool_receipt,
        "refit_resource_admission": {
            "available_system_ram_bytes": refit_available_ram,
            "estimated_two_model_peak_bytes": refit_model_peak,
            "estimated_matrix_peak_bytes": refit_matrix_peak,
            "required_reserve_bytes": 6_000_000_000,
        },
        "thresholds": thresholds,
        "authority": ["prediction", "confidence", "abstention", "review"],
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
        "action_timing_layer_included": False,
    }
    artifact_identity = _publish_artifact(root / "artifacts", artifact)
    manifest = seal_document({
        "schema_version": "stage6_expanded_finalist_manifest_v1",
        "status": "PASS_FINALIST_FROZEN_RETROSPECTIVE_ONLY",
        "finalist_run_id": run_id,
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "layer": layer,
        "broad_manifest_self_sha256": [row["self_sha256"] for row in broad_manifests],
        "candidate_count": len(finalists),
        "stage_data_binding": binding,
        "environment": environment,
        "worker_report": worker_report,
        "ranked_finalists": [
            {
                "rank": index + 1,
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "configuration_sha256": row["configuration_sha256"],
                "validation_log_loss": row["validation_metrics"]["log_loss"],
                "validation_brier": row["validation_metrics"]["multiclass_brier_score"],
                "selected_iterations": row["best_iteration"],
                "checkpoint_self_sha256": row["self_sha256"],
                "checkpoint_bytes": _checkpoint_path(root, str(row["candidate_id"])).stat().st_size,
                "checkpoint_sha256": sha256_file(_checkpoint_path(root, str(row["candidate_id"]))),
            }
            for index, row in enumerate(ranked)
        ],
        "selected_candidate_id": winner["candidate_id"],
        "selected_family": winner["family"],
        "selected_configuration_sha256": winner["configuration_sha256"],
        "selected_iterations": iterations,
        "selected_artifact": artifact_identity,
        "retrospective_calibration_row_count": len(calibration_indices),
        "retrospective_selected_metrics": selected_metrics,
        "retrospective_matched_control_metrics": control_metrics,
        "retrospective_strata": strata,
        "frozen_thresholds": thresholds,
        "calibration_frozen_after_refit": True,
        "fresh_targets_opened": False,
        "current_corpus_authority": "development_and_retrospective_only",
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "action_changing_exploit_authority": False,
        "action_timing_layer_included": False,
        "elapsed_seconds": time.monotonic() - started,
    })
    _atomic_json(root / "finalist-manifest.json", manifest)
    return {
        "status": "PASS",
        "finalist_run_id": run_id,
        "manifest_path": str(root / "finalist-manifest.json"),
        "manifest_self_sha256": manifest["self_sha256"],
        "selected_candidate_id": winner["candidate_id"],
        "selected_artifact": artifact_identity,
        "reused_completed_result": False,
        "qualification_authority": False,
    }
