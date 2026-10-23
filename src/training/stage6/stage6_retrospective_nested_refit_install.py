"""Deterministic production-preparation refit for qualified Stage 6 layers.

This installer deliberately separates scientific qualification from runtime
installation.  It reconstructs the frozen outer-fold-04 inner-selected model
configuration for each qualified layer, refits on causally eligible development
roles, freezes the authenticated outer-fold-04 inner-selected temperature, and
uses retrospective_calibration only for non-selecting diagnostics.  It publishes
versioned shadow artifacts, never writes PokerNow, never grants live-action
authority, and never overwrites an incumbent artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_info, threadpool_limits

from stage6_additional_exploitation_layers import (
    _CausalHistoryIndex,
    build_joint_action_size_surprise_features,
    build_latent_player_archetype_shrinkage_features,
    build_multiway_target_response_features,
    build_opponent_bluff_likelihood_features,
    build_player_strategy_drift_features,
    build_showdown_aware_range_calibration_features,
    build_supported_fold_response_bluff_opportunity_features,
)
from stage6_additional_exploitation_source_materializer import (
    DECISION_REQUIRED_COLUMNS,
    EXACT_LABEL_REQUIRED_COLUMNS,
    NOTEBOOK01_QUARANTINE_REQUIRED_COLUMNS,
    OBSERVATION_REQUIRED_COLUMNS,
    REVEALED_CARD_REQUIRED_COLUMNS,
    materialize_additional_exploitation_sources,
)
from stage6_behavior_tournament import (
    MODEL_SOURCE_COLUMNS,
    feature_matrix as behavior_feature_matrix,
    feature_names as behavior_feature_names,
)
from stage6_expanded_feature_freeze_v2 import (
    BASE_FEATURE_COLUMNS,
    COMMON_IDENTITY_COLUMNS,
    _load_development_source_v2,
)
from stage6_expanded_history_acceleration_v2 import build_line_prefix_features_v2
from stage6_expanded_layers import ACTIONS, temporal_role
from stage6_expanded_tournament import (
    _apply_imputer,
    _impute_medians,
    expanded_model_matrix,
)
from stage6_notebooks02_07_runtime import LANE_COUNT, LEGAL_COLUMNS, _lane_paths
from stage6_retrospective_nested_additional_source import _feature_names
from stage6_retrospective_nested_search import (
    _aligned_six_action_probabilities,
    _fit_fixed_candidate_allowing_action_subsets,
    _predict_projected,
    _registry,
)


QUALIFICATION_RUN_ID = "5132a0c8eed12f9e"
QUALIFICATION_AGGREGATE_BYTES = 111_030
QUALIFICATION_AGGREGATE_SHA256 = (
    "c951aa63c2381446c276de42ce012a99a22eb1e96fa1a253820571b333af40a9"
)
QUALIFICATION_AGGREGATE_SELF_SHA256 = (
    "1214874417891be2ce974cffd0b3cabe05f4ceb50ff2ef4c72a88e3ac92c3ab4"
)
QUALIFICATION_INDEX_RELATIVE = Path(
    "qualified_results/stage6_retrospective_nested_qualified_layers_"
    f"{QUALIFICATION_RUN_ID}.json"
)
VERSION_NAME = "stage6_retrospective_nested_v2"
MODEL_LABELS = tuple(ACTIONS)
FIT_ROLES = ("development_train", "development_validation")
CALIBRATION_ROLE = "retrospective_calibration"
PROHIBITED_ROLE = "fresh_adjudication"

QUALIFIED_LAYERS = (
    "within_hand_line_surprise",
    "postflop_behavior",
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "opponent_bluff_likelihood",
    "player_strategy_drift",
    "showdown_aware_range_calibration",
    "supported_fold_response_bluff_opportunity",
)
HELD_LAYERS = {
    "hero_opponent_interaction": "incomplete_outer_fold_search",
    "preflop_behavior": "failed_incumbent_comparison",
    "action_timing": "excluded_from_qualification_and_installation",
}

FROZEN_FOLD4_SELECTION = {
    "within_hand_line_surprise": ("catboost-024-16406dfd7ce2", 4181),
    "postflop_behavior": (
        "postflop_behavior--lightgbm-044-260330195a5f",
        453,
    ),
    "joint_action_size_surprise": ("lightgbm-059-952c898b4f4a", 112),
    "latent_player_archetype_shrinkage": ("xgboost-004-3ddf9836e441", 171),
    "multiway_target_response": ("catboost-022-d53e0c902ef0", 357),
    "opponent_bluff_likelihood": ("xgboost-058-2b3071105630", 133),
    "player_strategy_drift": ("catboost-046-51ff86504734", 252),
    "showdown_aware_range_calibration": ("catboost-018-a4850b7abfb7", 480),
    "supported_fold_response_bluff_opportunity": (
        "extra_trees-013-d0acf877bfb3",
        192,
    ),
}

ADDITIONAL_BUILDERS: dict[str, Callable[..., pd.DataFrame]] = {
    "joint_action_size_surprise": build_joint_action_size_surprise_features,
    "latent_player_archetype_shrinkage": (
        build_latent_player_archetype_shrinkage_features
    ),
    "multiway_target_response": build_multiway_target_response_features,
    "opponent_bluff_likelihood": build_opponent_bluff_likelihood_features,
    "player_strategy_drift": build_player_strategy_drift_features,
    "showdown_aware_range_calibration": (
        build_showdown_aware_range_calibration_features
    ),
    "supported_fold_response_bluff_opportunity": (
        build_supported_fold_response_bluff_opportunity_features
    ),
}

SUPPORT_CONTRACTS: dict[str, dict[str, Any]] = {
    "within_hand_line_surprise": {
        "predicate": "line_prefix_decision_count >= 1",
        "below_support": "abstain_exact_matched_control_not_installed_as_policy",
    },
    "postflop_behavior": {
        "predicate": "street in [flop,turn,river]",
        "below_support": "abstain",
    },
    "joint_action_size_surprise": {
        "predicate": "joint_action_size_supported == true",
        "below_support": "abstain",
    },
    "latent_player_archetype_shrinkage": {
        "predicate": "latent_archetype_supported == true",
        "below_support": "abstain",
        "prior_decisions": 12.0,
    },
    "multiway_target_response": {
        "predicate": "multiway_target_response_supported == true",
        "below_support": "abstain",
    },
    "opponent_bluff_likelihood": {
        "predicate": "one_or_more_selection_supported_flags == true",
        "below_support": "abstain",
        "authority": "bluffShownProxy_not_bluff_truth",
    },
    "player_strategy_drift": {
        "predicate": "strategy_drift_supported == true",
        "below_support": "abstain",
        "recent_completed_hands": 8,
        "reference_completed_hands": 64,
    },
    "showdown_aware_range_calibration": {
        "predicate": "range_calibration_strength_supported == true",
        "below_support": "abstain",
        "minimum_supported_labels": 3,
        "minimum_selection_coverage": 0.5,
    },
    "supported_fold_response_bluff_opportunity": {
        "predicate": "supported_fold_supported == true",
        "below_support": "abstain",
        "minimum_observations": 4,
        "minimum_local_observations": 3,
    },
}


@dataclass(frozen=True)
class FrozenSpec:
    layer: str
    candidate_id: str
    family: str
    parameters: dict[str, Any]
    configuration_sha256: str
    best_iteration: int
    qualification_fold_temperature: float
    qualification_receipt_self_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_new_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == raw:
            return
        raise FileExistsError(f"refusing to overwrite existing bytes: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"installer staging path already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    write_new_bytes(path, canonical_json_bytes(value) + b"\n")


def _assert_canonical(path: Path, workspace: Path) -> Path:
    resolved = path.resolve()
    canonical_workspace = workspace.resolve()
    if "OneDrive" in str(resolved):
        raise RuntimeError(f"historical OneDrive path is forbidden: {resolved}")
    try:
        resolved.relative_to(canonical_workspace)
    except ValueError as error:
        raise RuntimeError(f"path escapes canonical workspace: {resolved}") from error
    return resolved


def _load_aggregate(path: Path) -> dict[str, Any]:
    if path.stat().st_size != QUALIFICATION_AGGREGATE_BYTES:
        raise RuntimeError("qualified aggregate byte count changed")
    if sha256_file(path) != QUALIFICATION_AGGREGATE_SHA256:
        raise RuntimeError("qualified aggregate SHA-256 changed")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["retrospective-nested-aggregate-manifest.json"]:
            raise RuntimeError("qualified aggregate member inventory changed")
        document = json.loads(archive.read(names[0]))
    if (
        document.get("self_sha256") != QUALIFICATION_AGGREGATE_SELF_SHA256
        or not str(document.get("run_id", "")).startswith(QUALIFICATION_RUN_ID)
        or tuple(document.get("qualified_layers", ())) != QUALIFIED_LAYERS
        or document.get("status")
        != "HOLD_ONE_OR_MORE_RETROSPECTIVE_NESTED_LAYER_GATES"
        or document.get("assistant_installation_allowed_for_qualified_layers_after_focused_independent_audit")
        is not True
        or document.get("automatic_assistant_installation_performed") is not False
    ):
        raise RuntimeError("qualified aggregate authority changed")
    return document


def _fold4_receipt(aggregate: Mapping[str, Any], layer: str) -> Mapping[str, Any]:
    if layer == "postflop_behavior":
        return aggregate["behavior"]["reports"][layer]["fold_receipts"][4]
    if layer == "within_hand_line_surprise":
        return aggregate["expanded"]["claims"][layer]["fold_receipts"][4]
    rows = {
        str(row["layer"]): row for row in aggregate["additional"]["layers"]
    }
    return rows[layer]["fold_receipts"][4]


def _frozen_specs(aggregate: Mapping[str, Any]) -> dict[str, FrozenSpec]:
    specs: dict[str, FrozenSpec] = {}
    for layer in QUALIFIED_LAYERS:
        candidate_id, best_iteration = FROZEN_FOLD4_SELECTION[layer]
        matches = [
            row for row in _registry(layer) if row.get("candidate_id") == candidate_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"frozen candidate registry lookup changed: {layer}")
        registry = matches[0]
        receipt = _fold4_receipt(aggregate, layer)
        expected = {
            "fold_id": "outer-fold-04",
            "selected_candidate_id": candidate_id,
            "selected_best_iteration": best_iteration,
            "selected_configuration_sha256": registry["configuration_sha256"],
            "selected_family": registry["family"],
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"qualified fold-4 receipt changed: {layer}/{key}")
        report = receipt["prediction_report"]
        specs[layer] = FrozenSpec(
            layer=layer,
            candidate_id=candidate_id,
            family=str(registry["family"]),
            parameters=dict(registry["parameters"]),
            configuration_sha256=str(registry["configuration_sha256"]),
            best_iteration=best_iteration,
            qualification_fold_temperature=float(report["selected_temperature"]),
            qualification_receipt_self_sha256=str(
                receipt["manifest_self_sha256"]
            ),
        )
    return specs


def _versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "joblib",
        "lightgbm",
        "xgboost",
        "catboost",
        "threadpoolctl",
    )
    return {name: importlib_metadata.version(name) for name in packages}


def _role_frame(frame: pd.DataFrame) -> pd.Series:
    per_hand = (
        frame.groupby(["game_id", "hand_id"], sort=False, observed=True)[
            "decision_utc"
        ]
        .max()
        .astype(str)
        .map(temporal_role)
        .rename("temporal_role")
        .reset_index()
    )
    joined = frame[["game_id", "hand_id"]].merge(
        per_hand,
        on=["game_id", "hand_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined["temporal_role"].isna().any():
        raise RuntimeError("whole-hand temporal roles are incomplete")
    roles = joined["temporal_role"].astype(str)
    if roles.eq(PROHIBITED_ROLE).any():
        raise RuntimeError("fresh adjudication rows entered production refit")
    observed = set(roles)
    required = {*FIT_ROLES, CALIBRATION_ROLE}
    if not required.issubset(observed):
        raise RuntimeError(f"required refit roles are missing: {sorted(required-observed)}")
    return roles


def _schema_hash(feature_names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(feature_names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _native_model_payload(model: Any, spec: FrozenSpec) -> dict[str, Any]:
    native = getattr(model, "booster", model)
    return {
        "schema_version": "stage6_retrospective_nested_model_package_v1",
        "layer": spec.layer,
        "family": spec.family,
        "candidate_id": spec.candidate_id,
        "classes": list(range(len(MODEL_LABELS))),
        "model": native,
        "prediction_api": (
            "native_booster_predict"
            if native is not model
            else "estimator_predict_proba"
        ),
    }


def _package_probabilities(package: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    model = package["model"]
    family = str(package["family"])
    if package["prediction_api"] == "native_booster_predict":
        if family == "xgboost":
            import xgboost as xgb

            raw = model.predict(xgb.DMatrix(matrix))
        else:
            raw = model.predict(matrix)
    else:
        raw = model.predict_proba(matrix)
    values = np.asarray(raw, dtype=np.float64)
    classes = [int(value) for value in package["classes"]]
    if values.shape != (len(matrix), len(classes)):
        raise RuntimeError("serialized model probability shape changed")
    aligned = np.zeros((len(matrix), len(MODEL_LABELS)), dtype=np.float64)
    for source, target in enumerate(classes):
        aligned[:, target] = values[:, source]
    return aligned


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _frozen_temperature_diagnostic(
    model: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    priors: np.ndarray,
    temperature: float,
) -> dict[str, Any]:
    from stage6_metrics import multiclass_metric_report

    probabilities, projection = _predict_projected(
        model, matrix, legal, priors, temperature
    )
    metrics = multiclass_metric_report(
        [MODEL_LABELS[int(value)] for value in target],
        probabilities,
        MODEL_LABELS,
    )
    return {
        "selection_reperformed": False,
        "selection_source": "authenticated_outer_fold_04_inner_validation_receipt",
        "diagnostic_partition": CALIBRATION_ROLE,
        "frozen_temperature": float(temperature),
        "diagnostic": {
            "temperature": float(temperature),
            "log_loss": float(metrics["log_loss"]),
            "multiclass_brier_score": float(metrics["multiclass_brier_score"]),
            "projection": projection,
        },
    }


def _feature_importance(package: Mapping[str, Any], names: Sequence[str]) -> list[dict[str, Any]]:
    model = package["model"]
    family = str(package["family"])
    values: np.ndarray
    if family == "lightgbm" and hasattr(model, "feature_importance"):
        values = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    elif family == "xgboost" and hasattr(model, "get_score"):
        scores = model.get_score(importance_type="gain")
        values = np.asarray([scores.get(f"f{index}", 0.0) for index in range(len(names))])
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.zeros(len(names), dtype=float)
    if len(values) != len(names):
        values = np.zeros(len(names), dtype=float)
    order = np.argsort(-values, kind="stable")
    return [
        {"feature": str(names[index]), "importance": float(values[index])}
        for index in order
    ]


def _support_counts(
    layer: str, matrix: np.ndarray, feature_names: Sequence[str], frame: pd.DataFrame
) -> dict[str, int]:
    if layer == "within_hand_line_surprise":
        supported = (
            pd.to_numeric(frame["line_prefix_decision_count"], errors="raise")
            .fillna(0)
            .to_numpy(dtype=float)
            >= 1.0
        )
    elif layer == "postflop_behavior":
        supported = np.ones(len(matrix), dtype=bool)
    else:
        indices = [
            index
            for index, name in enumerate(feature_names)
            if str(name).endswith("_supported")
        ]
        supported = (
            np.any(np.nan_to_num(matrix[:, indices], nan=0.0) > 0.5, axis=1)
            if indices
            else np.zeros(len(matrix), dtype=bool)
        )
    return {
        "row_count": int(len(matrix)),
        "supported_row_count": int(supported.sum()),
        "abstained_row_count": int((~supported).sum()),
    }


@contextmanager
def _one_native_thread() -> Any:
    environment_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in environment_names}
    for name in environment_names:
        os.environ[name] = "1"
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _train_layer(
    *,
    spec: FrozenSpec,
    matrix: np.ndarray,
    target: np.ndarray,
    legal: np.ndarray,
    roles: Sequence[str],
    feature_names: Sequence[str],
    feature_frame: pd.DataFrame,
    output_dir: Path,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    final_output_dir = output_dir
    if matrix.shape != (len(target), len(feature_names)):
        raise RuntimeError(f"feature matrix shape changed: {spec.layer}")
    if legal.shape != (len(target), len(MODEL_LABELS)):
        raise RuntimeError(f"legal matrix shape changed: {spec.layer}")
    if not legal[np.arange(len(target)), target.astype(int)].all():
        raise RuntimeError(f"observed target is illegal: {spec.layer}")
    role_values = np.asarray([str(value) for value in roles])
    fit_indices = np.flatnonzero(np.isin(role_values, FIT_ROLES))
    calibration_indices = np.flatnonzero(role_values == CALIBRATION_ROLE)
    if not len(fit_indices) or not len(calibration_indices):
        raise RuntimeError(f"fit/calibration partition is empty: {spec.layer}")
    fit_target = np.asarray(target[fit_indices], dtype=np.int8)
    if set(np.unique(fit_target)) != set(range(len(MODEL_LABELS))):
        raise RuntimeError(f"fit partition lacks a six-action class: {spec.layer}")
    medians = _impute_medians(matrix[fit_indices])
    fit_x = _apply_imputer(matrix[fit_indices], medians)
    counts = np.bincount(fit_target, minlength=len(MODEL_LABELS)).astype(float)
    priors = (counts + 1.0) / (counts.sum() + len(MODEL_LABELS))
    entry = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "parameters": spec.parameters,
    }
    started = time.perf_counter()
    with _one_native_thread():
        model = _fit_fixed_candidate_allowing_action_subsets(
            entry,
            fit_x,
            fit_target,
            spec.best_iteration,
        )
        calibration_x = _apply_imputer(matrix[calibration_indices], medians)
        temperature = float(spec.qualification_fold_temperature)
        calibration_report = _frozen_temperature_diagnostic(
            model,
            calibration_x,
            np.asarray(target[calibration_indices], dtype=np.int8),
            legal[calibration_indices],
            priors,
            temperature,
        )
        package = _native_model_payload(model, spec)
        package.update(
            {
                "feature_names": list(feature_names),
                "feature_order_sha256": _schema_hash(feature_names),
                "imputer_medians": [float(value) for value in medians],
                "class_priors_laplace_one": [float(value) for value in priors],
                "temperature": float(temperature),
                "fail_closed_on_missing_features": True,
                "support_contract": SUPPORT_CONTRACTS[spec.layer],
                "direct_action_authority": False,
                "live_action_authority": False,
            }
        )
        native_pools = [
            {
                "user_api": str(row.get("user_api")),
                "internal_api": str(row.get("internal_api")),
                "num_threads": int(row.get("num_threads") or 0),
            }
            for row in threadpool_info()
        ]
        oversized_pools = [
            row for row in native_pools if int(row.get("num_threads", 0)) > 1
        ]
        if oversized_pools:
            raise RuntimeError(
                f"qualified refit exposed a native thread pool above one thread: "
                f"{spec.layer}: {oversized_pools}"
            )
    elapsed = time.perf_counter() - started
    support = _support_counts(spec.layer, matrix, feature_names, feature_frame)

    output_dir = final_output_dir.with_name(
        f".{final_output_dir.name}.staging-{os.getpid()}-{time.time_ns()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    model_name = (
        "lightgbm_baseline.joblib"
        if spec.layer == "postflop_behavior"
        else "model.joblib"
    )
    model_path = output_dir / model_name
    joblib.dump(package, model_path, compress=3)
    loaded = joblib.load(model_path)
    sample_indices = calibration_indices[: min(32, len(calibration_indices))]
    sample_x = _apply_imputer(matrix[sample_indices], medians)
    expected = _aligned_six_action_probabilities(model, sample_x)
    observed = _package_probabilities(loaded, sample_x)
    if not np.allclose(expected, observed, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"serialized model parity failed: {spec.layer}")

    feature_schema = {
        "schema_version": "stage6_retrospective_nested_feature_schema_v1",
        "categorical": [],
        "numeric": list(feature_names),
        "feature_order_sha256": _schema_hash(feature_names),
        "missing_value_policy": "fit_partition_median_then_fail_closed_nonfinite",
        "action_or_decision_timing_feature": False,
        "hidden_or_future_card_feature": False,
    }
    label_schema = {label: index for index, label in enumerate(MODEL_LABELS)}
    calibration = {
        "method": "frozen_authenticated_inner_selected_temperature",
        "temperature": temperature,
        "selection_source": "outer-fold-04 prediction_report.selected_temperature",
        "selection_reperformed": False,
        "retrospective_calibration_used_for_selection": False,
    }
    importance = _feature_importance(package, feature_names)
    receipt = seal_document(
        {
            "schema_version": "stage6_retrospective_nested_refit_receipt_v2",
            "status": "PASS_DETERMINISTIC_ONE_THREAD_REFIT",
            "layer": spec.layer,
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "parameters": spec.parameters,
            "configuration_sha256": spec.configuration_sha256,
            "best_iteration": spec.best_iteration,
            "selection_rule": (
                "latest_chronological_completed_outer_fold_inner_selected_"
                "configuration_without_outer_test_reuse"
            ),
            "selection_fold": "outer-fold-04",
            "qualification_fold_temperature": spec.qualification_fold_temperature,
            "refit_temperature": temperature,
            "temperature_selection_partition": (
                "authenticated_outer_fold_04_inner_validation_receipt"
            ),
            "fit_roles": list(FIT_ROLES),
            "fit_row_count": int(len(fit_indices)),
            "calibration_role": CALIBRATION_ROLE,
            "calibration_row_count": int(len(calibration_indices)),
            "class_counts_fit": {
                label: int(counts[index])
                for index, label in enumerate(MODEL_LABELS)
            },
            "class_priors_laplace_one": {
                label: float(priors[index])
                for index, label in enumerate(MODEL_LABELS)
            },
            "feature_count": len(feature_names),
            "feature_order_sha256": _schema_hash(feature_names),
            "imputer_medians": [float(value) for value in medians],
            "support": support,
            "support_contract": SUPPORT_CONTRACTS[spec.layer],
            "calibration_report": calibration_report,
            "source_binding": dict(source_binding),
            "qualified_aggregate_sha256": QUALIFICATION_AGGREGATE_SHA256,
            "qualified_aggregate_self_sha256": (
                QUALIFICATION_AGGREGATE_SELF_SHA256
            ),
            "qualification_receipt_self_sha256": (
                spec.qualification_receipt_self_sha256
            ),
            "versions": _versions(),
            "candidate_internal_threads": 1,
            "threadpoolctl_limit_applied": True,
            "observed_native_threadpools": native_pools,
            "elapsed_seconds": elapsed,
            "serialized_probability_parity_max_abs": float(
                np.max(np.abs(expected - observed))
            ),
            "action_timing_layer_included": False,
            "direct_action_authority": False,
            "live_action_authority": False,
            "automatic_pokernow_action": False,
        }
    )
    metrics = {
        "schema_version": "stage6_retrospective_nested_refit_metrics_v2",
        "layer": spec.layer,
        "calibration": calibration_report,
        "support": support,
        "outer_test_metrics_reused_for_refit_selection": False,
        "retrospective_calibration_used_for_selection": False,
    }
    model_card = (
        f"# {spec.layer}\n\n"
        f"Version: `{VERSION_NAME}`\n\n"
        f"Qualified frozen candidate: `{spec.candidate_id}` at "
        f"{spec.best_iteration} iterations.\n\n"
        "This is a read-only shadow prediction artifact. It has no direct "
        "betting, sizing, PokerNow write, or live-action authority. Unsupported "
        "inputs abstain under the bound support contract.\n"
    ).encode("utf-8")
    write_new_json(output_dir / "feature_schema.json", feature_schema)
    write_new_json(output_dir / "label_schema.json", label_schema)
    write_new_json(output_dir / "calibration.json", calibration)
    write_new_json(output_dir / "metrics.json", metrics)
    write_new_json(output_dir / "refit_receipt.json", receipt)
    write_new_json(output_dir / "support_contract.json", SUPPORT_CONTRACTS[spec.layer])
    write_new_json(
        output_dir / "feature_importance.json",
        {"importance_type": "model_native_gain_or_equivalent", "features": importance},
    )
    write_new_bytes(output_dir / "model_card.md", model_card)
    if spec.layer == "postflop_behavior":
        write_new_json(output_dir / "lightgbm_calibration.json", calibration)
        write_new_json(
            output_dir / "lightgbm_feature_importance.json",
            {
                "importance_type": "lightgbm_gain",
                "top_features": importance[:50],
            },
        )
        csv_raw = "feature,importance\n" + "\n".join(
            f"{row['feature']},{row['importance']:.17g}" for row in importance[:50]
        ) + "\n"
        write_new_bytes(
            output_dir / "lightgbm_top_features.csv", csv_raw.encode("utf-8")
        )
    if final_output_dir.exists():
        raise FileExistsError(
            f"refusing to replace an existing layer artifact: {final_output_dir}"
        )
    output_dir.replace(final_output_dir)
    identities = {
        path.name: file_identity(path, relative_to=final_output_dir)
        for path in sorted(final_output_dir.iterdir())
        if path.is_file()
    }
    return {
        "layer": spec.layer,
        "artifact_directory": final_output_dir.name,
        "candidate_id": spec.candidate_id,
        "configuration_sha256": spec.configuration_sha256,
        "best_iteration": spec.best_iteration,
        "refit_temperature": temperature,
        "support": support,
        "receipt_self_sha256": receipt["self_sha256"],
        "files": identities,
    }


def _existing_layer_row(
    output_dir: Path, spec: FrozenSpec
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    receipt_path = output_dir / "refit_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"existing layer directory is incomplete: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_self = str(receipt.get("self_sha256", ""))
    candidate = dict(receipt)
    candidate.pop("self_sha256", None)
    if (
        not expected_self
        or hashlib.sha256(canonical_json_bytes(candidate)).hexdigest()
        != expected_self
        or receipt.get("status") != "PASS_DETERMINISTIC_ONE_THREAD_REFIT"
        or receipt.get("layer") != spec.layer
        or receipt.get("candidate_id") != spec.candidate_id
        or receipt.get("configuration_sha256") != spec.configuration_sha256
        or receipt.get("best_iteration") != spec.best_iteration
    ):
        raise RuntimeError(f"existing layer receipt binding is invalid: {spec.layer}")
    files = {
        path.name: file_identity(path, relative_to=output_dir)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    required = {
        "feature_schema.json",
        "label_schema.json",
        "calibration.json",
        "metrics.json",
        "refit_receipt.json",
        "support_contract.json",
        "feature_importance.json",
        "model_card.md",
        (
            "lightgbm_baseline.joblib"
            if spec.layer == "postflop_behavior"
            else "model.joblib"
        ),
    }
    if not required.issubset(files):
        raise RuntimeError(f"existing layer file inventory is incomplete: {spec.layer}")
    return {
        "layer": spec.layer,
        "artifact_directory": output_dir.name,
        "candidate_id": spec.candidate_id,
        "configuration_sha256": spec.configuration_sha256,
        "best_iteration": spec.best_iteration,
        "refit_temperature": receipt["refit_temperature"],
        "support": receipt["support"],
        "receipt_self_sha256": expected_self,
        "files": files,
    }


def _core_datasets(
    materialization_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = _load_development_source_v2(materialization_root)
    source["temporal_role"] = _role_frame(source).to_numpy()
    behavior_matrix = behavior_feature_matrix(source)
    names = behavior_feature_names()
    common = source.loc[:, COMMON_IDENTITY_COLUMNS].copy()
    common["temporal_role"] = source["temporal_role"].astype(str).to_numpy()
    for index, name in enumerate(BASE_FEATURE_COLUMNS):
        common[name] = behavior_matrix[:, index]
    line = build_line_prefix_features_v2(source)
    joined = common.merge(line, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(common) or len(joined) != len(line):
        raise RuntimeError("line/common refit join is incomplete")
    line_names, line_matrix = expanded_model_matrix(
        joined, "within_hand_line_surprise"
    )
    target = np.asarray(
        [MODEL_LABELS.index(value) for value in source["action_label"].astype(str)],
        dtype=np.int8,
    )
    legal = source[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
    postflop = source["street"].astype(str).isin(("flop", "turn", "river")).to_numpy()
    source_binding = {
        "materialization_manifest": file_identity(
            materialization_root / "stage6_notebook01_materialization_manifest.json",
            relative_to=materialization_root,
        ),
        "accepted_row_count": int(len(source)),
        "maximum_decision_utc": str(source["decision_utc"].max()),
        "action_or_decision_timing_feature": False,
    }
    datasets = {
        "postflop_behavior": {
            "matrix": behavior_matrix[postflop],
            "target": target[postflop],
            "legal": legal[postflop],
            "roles": source.loc[postflop, "temporal_role"].astype(str).to_numpy(),
            "feature_names": names,
            "feature_frame": source.loc[postflop, ["street"]].reset_index(drop=True),
        },
        "within_hand_line_surprise": {
            "matrix": line_matrix,
            "target": target,
            "legal": legal,
            "roles": source["temporal_role"].astype(str).to_numpy(),
            "feature_names": line_names,
            "feature_frame": joined,
        },
    }
    return datasets, source_binding


def _load_additional_decisions(materialization_root: Path) -> pd.DataFrame:
    # The causal source materializer needs only its DECISION_REQUIRED_COLUMNS,
    # but the refit identity also carries the six authenticated legal-action
    # masks into projection/evaluation.  Read both sets from the same sealed
    # Notebook 01 decision payloads so legality is never reconstructed later.
    columns = list(dict.fromkeys((*DECISION_REQUIRED_COLUMNS, *LEGAL_COLUMNS)))
    frames: list[pd.DataFrame] = []
    for lane in range(LANE_COUNT):
        decision_path, _, _ = _lane_paths(materialization_root, lane)
        frames.append(pq.read_table(decision_path, columns=columns).to_pandas())
        print(
            json.dumps(
                {
                    "event": "stage6_install_additional_decision_lane_loaded",
                    "completed": lane + 1,
                    "denominator": LANE_COUNT,
                    "lane": lane,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    frame = pd.concat(frames, ignore_index=True)
    if frame["row_id"].astype(str).duplicated().any():
        raise RuntimeError("additional decision source duplicates row_id")
    return frame


def _load_quarantine(materialization_root: Path) -> pd.DataFrame:
    frames = []
    columns = list(NOTEBOOK01_QUARANTINE_REQUIRED_COLUMNS)
    for lane in range(LANE_COUNT):
        path = materialization_root / "lanes" / f"lane-{lane:02d}" / "quarantine.parquet"
        frames.append(pq.read_table(path, columns=columns).to_pandas())
    return pd.concat(frames, ignore_index=True)


def _load_additional_inputs(
    stage6_root: Path, additional_zip: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], bytes]:
    observations = pd.read_csv(
        stage6_root / "data/exports/ml/stage6-hand-profile-observations.csv",
        usecols=list(OBSERVATION_REQUIRED_COLUMNS),
        low_memory=False,
    )
    with zipfile.ZipFile(additional_zip) as archive:
        revealed_name = "stage008_known_hole_cards.csv"
        label_name = "stage6_additional_revealed_hand_labels.jsonl"
        receipt_name = "stage6_additional_revealed_hand_labels_receipt.json"
        label_raw = archive.read(label_name)
        labels = pd.read_json(
            io.BytesIO(label_raw), lines=True, precise_float=True
        ).loc[:, list(EXACT_LABEL_REQUIRED_COLUMNS)]
        revealed = pd.read_csv(
            archive.open(revealed_name),
            usecols=list(REVEALED_CARD_REQUIRED_COLUMNS),
            low_memory=False,
        )
        receipt = json.loads(archive.read(receipt_name))
    return observations, revealed, labels, receipt, label_raw


def _additional_source(
    stage6_root: Path,
    materialization_root: Path,
    additional_zip: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decisions = _load_additional_decisions(materialization_root)
    quarantines = _load_quarantine(materialization_root)
    observations, revealed, labels, label_receipt, label_raw = (
        _load_additional_inputs(stage6_root, additional_zip)
    )

    def progress(event: Mapping[str, Any]) -> None:
        print(json.dumps(dict(event), sort_keys=True), flush=True)

    materialized = materialize_additional_exploitation_sources(
        decisions,
        observations,
        revealed_card_rows=revealed,
        exact_label_rows=labels,
        exact_label_receipt=label_receipt,
        exact_label_raw_bytes=label_raw,
        quarantined_decision_rows=quarantines,
        progress_callback=progress,
    )
    context = materialized["decision_context"]
    history = materialized["completed_history"]
    roles = _role_frame(decisions)
    identity = decisions[["row_id", "action_label", *LEGAL_COLUMNS]].copy()
    identity["temporal_role"] = roles.to_numpy()
    source_binding = {
        "materialization_manifest_self_sha256": materialized["manifest"][
            "self_sha256"
        ],
        "materialization_counts": materialized["manifest"]["counts"],
        "additional_input_zip": file_identity(
            additional_zip, relative_to=stage6_root
        ),
        "action_or_decision_timing_feature": False,
        "current_action_or_outcome_used_as_feature": False,
    }
    del decisions, quarantines, observations, revealed, labels, label_raw, materialized
    gc.collect()
    return context, history, identity, source_binding


def _publish_install_manifest(
    artifact_root: Path,
    evidence_root: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    qualification_index: Path,
    aggregate_zip: Path,
    incumbent_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = seal_document(
        {
            "schema_version": "stage6_retrospective_nested_install_manifest_v2",
            "status": "PASS_VERSIONED_SHADOW_ARTIFACTS_PREPARED",
            "version": VERSION_NAME,
            "qualified_run_id": QUALIFICATION_RUN_ID,
            "qualified_layers": list(QUALIFIED_LAYERS),
            "held_layers": HELD_LAYERS,
            "mapping": {
                "postflop_behavior": {
                    "runtime_head": "action_state_global",
                    "role": "only_qualified_replacement_candidate",
                    "activation": "versioned_pointer_after_focused_runtime_validation",
                    "rollback": "remove_pointer_or_select_incumbent_folder",
                },
                "within_hand_line_surprise": {
                    "runtime_role": "typed_additive_shadow_telemetry",
                    "action_effect": False,
                },
                "additional_seven": {
                    "runtime_role": "typed_additive_shadow_telemetry",
                    "action_effect": False,
                },
            },
            "layers": list(layer_rows),
            "qualification_index": file_identity(
                qualification_index, relative_to=qualification_index.parents[1]
            ),
            "qualification_aggregate": file_identity(
                aggregate_zip, relative_to=aggregate_zip.parents[1]
            ),
            "incumbent_hashes_before_install": dict(incumbent_hashes),
            "preflop_incumbent_preserved": True,
            "bet_size_incumbent_preserved": True,
            "postflop_incumbent_preserved_as_rollback_bytes": True,
            "stage21_strategy_authority_changed": False,
            "stage51_reintroduced": False,
            "action_timing_layer_included": False,
            "direct_action_authority": False,
            "live_action_authority": False,
            "automatic_pokernow_action": False,
            "exact_gto_claim": False,
            "artifact_root": artifact_root.as_posix(),
        }
    )
    write_new_json(artifact_root / "install_manifest.json", manifest)
    evidence_root.mkdir(parents=True, exist_ok=False)
    write_new_json(evidence_root / "install_manifest.json", manifest)
    write_new_json(
        evidence_root / "held_layers.json",
        seal_document(
            {
                "schema_version": "stage6_retrospective_nested_held_layers_v1",
                "held_layers": HELD_LAYERS,
            }
        ),
    )
    readme = (
        "# Stage 6 Retrospective Nested Qualified Layers\n\n"
        f"Qualified run: `{QUALIFICATION_RUN_ID}`\n\n"
        f"Install manifest self hash: `{manifest['self_sha256']}`\n\n"
        "Nine qualified layers were reconstructed from their frozen outer-fold-04 "
        "inner-selected configurations and refit on causally eligible development "
        "roles. Each scalar temperature is frozen from the authenticated outer-fold-04 "
        "inner-selected receipt; retrospective rows are diagnostic only. "
        "These artifacts are read-only shadow inputs: they have no PokerNow write, "
        "direct action, sizing, live-action, scientific requalification, or promotion "
        "authority. `postflop_behavior` is the only `action_state_global` replacement "
        "candidate; its incumbent bytes remain the rollback target. The other eight "
        "layers are typed, source-labelled, support-gated telemetry and abstain when "
        "their exact causal feature record is unavailable.\n\n"
        "Held: `hero_opponent_interaction`, `preflop_behavior`, and `action_timing`.\n"
    )
    write_new_bytes(evidence_root / "README.md", readme.encode("utf-8"))
    for row in layer_rows:
        source = artifact_root / str(row["artifact_directory"]) / "refit_receipt.json"
        destination = evidence_root / f"{row['layer']}-refit-receipt.json"
        write_new_bytes(destination, source.read_bytes())
    return manifest


def _validate_qualification_index(
    path: Path, aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    index = json.loads(path.read_text(encoding="utf-8"))
    qualified = [str(row.get("layer")) for row in index.get("qualified_layers", [])]
    source = index.get("source_aggregate", {})
    if (
        index.get("schema_version")
        != "stage6_retrospective_nested_qualified_layers_index_v1"
        or index.get("status")
        != "PASS_NINE_RETROSPECTIVE_NESTED_LAYER_GATES_PENDING_ASSISTANT_INSTALLATION"
        or index.get("authority_class") != "retrospective_nested_out_of_fold"
        or index.get("qualified_layer_count") != len(QUALIFIED_LAYERS)
        or qualified != list(QUALIFIED_LAYERS)
        or not str(source.get("run_id", "")).startswith(QUALIFICATION_RUN_ID)
        or source.get("run_id") != aggregate.get("run_id")
        or source.get("bytes") != QUALIFICATION_AGGREGATE_BYTES
        or source.get("sha256") != QUALIFICATION_AGGREGATE_SHA256
        or source.get("manifest_self_sha256")
        != QUALIFICATION_AGGREGATE_SELF_SHA256
        or index.get("fresh_or_prospective_claim") is not False
        or index.get("direct_action_authority") is not False
        or index.get("live_action_authority") is not False
        or index.get("automatic_assistant_installation_performed") is not False
        or index.get("action_timing_layer_included") is not False
    ):
        raise RuntimeError("qualified layer index authority binding is invalid")
    return index


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    stage6_root = _assert_canonical(Path(args.stage6_root), workspace)
    assistant_root = _assert_canonical(Path(args.assistant_root), workspace)
    materialization_root = _assert_canonical(
        Path(args.materialization_root), workspace
    )
    aggregate_zip = _assert_canonical(Path(args.aggregate_zip), workspace)
    additional_zip = _assert_canonical(Path(args.additional_input_zip), workspace)
    qualification_index = _assert_canonical(
        stage6_root / QUALIFICATION_INDEX_RELATIVE, workspace
    )
    aggregate = _load_aggregate(aggregate_zip)
    specs = _frozen_specs(aggregate)
    if not qualification_index.is_file():
        raise RuntimeError("qualified index is missing")
    _validate_qualification_index(qualification_index, aggregate)
    materialization_manifest = (
        materialization_root / "stage6_notebook01_materialization_manifest.json"
    )
    if not materialization_manifest.is_file():
        raise RuntimeError("completed causal materialization manifest is missing")

    candidate_root = assistant_root / "runtime/ml/artifacts" / VERSION_NAME
    evidence_root = (
        assistant_root.parents[0]
        / "Qualified Layers"
        / "Stage 6 Retrospective Nested Qualified Layers"
        / QUALIFICATION_RUN_ID
        / VERSION_NAME
    )
    if evidence_root.exists():
        raise FileExistsError("versioned evidence destination already exists")
    if candidate_root.exists():
        if (candidate_root / "install_manifest.json").exists():
            raise FileExistsError("versioned install is already complete")
    else:
        candidate_root.mkdir(parents=True, exist_ok=False)

    incumbent_root = assistant_root / "runtime/ml/artifacts"
    incumbent_paths = {
        "preflop_global": incumbent_root / "preflop_global/lightgbm_baseline.joblib",
        "action_state_global": incumbent_root
        / "action_state_global/lightgbm_baseline.joblib",
        "bet_size_bucket": incumbent_root / "bet_size_bucket/lightgbm_baseline.joblib",
    }
    incumbent_hashes = {
        name: file_identity(path, relative_to=incumbent_root)
        for name, path in incumbent_paths.items()
    }

    layer_rows: dict[str, dict[str, Any]] = {}
    core_layers = ("postflop_behavior", "within_hand_line_surprise")
    for layer in core_layers:
        existing = _existing_layer_row(candidate_root / layer, specs[layer])
        if existing is not None:
            layer_rows[layer] = existing
            print(
                json.dumps(
                    {"event": "stage6_install_refit_resumed", **existing},
                    sort_keys=True,
                ),
                flush=True,
            )
    missing_core = [layer for layer in core_layers if layer not in layer_rows]
    if missing_core:
        core, core_binding = _core_datasets(materialization_root)
        for layer in missing_core:
            print(json.dumps({"event": "stage6_install_refit_started", "layer": layer}), flush=True)
            row = _train_layer(
                spec=specs[layer],
                output_dir=candidate_root / layer,
                source_binding=core_binding,
                **core[layer],
            )
            layer_rows[layer] = row
            print(json.dumps({"event": "stage6_install_refit_complete", **row}, sort_keys=True), flush=True)
            del core[layer]
            gc.collect()
        del core
        gc.collect()

    for layer in ADDITIONAL_BUILDERS:
        existing = _existing_layer_row(candidate_root / layer, specs[layer])
        if existing is not None:
            layer_rows[layer] = existing
            print(
                json.dumps(
                    {"event": "stage6_install_refit_resumed", **existing},
                    sort_keys=True,
                ),
                flush=True,
            )
    missing_additional = [
        layer for layer in ADDITIONAL_BUILDERS if layer not in layer_rows
    ]
    if missing_additional:
        context, history, identity, additional_binding = _additional_source(
            stage6_root, materialization_root, additional_zip
        )
        history_index = _CausalHistoryIndex(history)
        del history
        empty_history = pd.DataFrame()
    for layer in missing_additional:
        builder = ADDITIONAL_BUILDERS[layer]
        print(json.dumps({"event": "stage6_install_refit_started", "layer": layer}), flush=True)
        frame = builder(context, empty_history, _index=history_index)
        joined = identity.merge(frame, on="row_id", how="inner", validate="one_to_one")
        if len(joined) != len(identity) or len(joined) != len(frame):
            raise RuntimeError(f"additional identity/feature join is incomplete: {layer}")
        names = list(_feature_names(frame))
        matrix = joined[names].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )
        if np.isinf(matrix).any():
            raise RuntimeError(f"additional feature contains infinity: {layer}")
        target = np.asarray(
            [MODEL_LABELS.index(value) for value in joined["action_label"].astype(str)],
            dtype=np.int8,
        )
        legal = joined[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
        row = _train_layer(
            spec=specs[layer],
            matrix=matrix,
            target=target,
            legal=legal,
            roles=joined["temporal_role"].astype(str).to_numpy(),
            feature_names=names,
            feature_frame=frame,
            output_dir=candidate_root / layer,
            source_binding=additional_binding,
        )
        layer_rows[layer] = row
        print(json.dumps({"event": "stage6_install_refit_complete", **row}, sort_keys=True), flush=True)
        del frame, joined, matrix, target, legal
        gc.collect()
    if missing_additional:
        del context, identity, history_index
        gc.collect()

    ordered_rows = [layer_rows[layer] for layer in QUALIFIED_LAYERS]
    manifest = _publish_install_manifest(
        candidate_root,
        evidence_root,
        ordered_rows,
        qualification_index,
        aggregate_zip,
        incumbent_hashes,
    )
    observed_incumbents = {
        name: file_identity(path, relative_to=incumbent_root)
        for name, path in incumbent_paths.items()
    }
    if observed_incumbents != incumbent_hashes:
        raise RuntimeError("an incumbent artifact changed during versioned installation")
    return {
        "status": manifest["status"],
        "artifact_root": str(candidate_root),
        "evidence_root": str(evidence_root),
        "install_manifest_self_sha256": manifest["self_sha256"],
        "qualified_layer_count": len(ordered_rows),
        "incumbents_preserved": True,
        "pointer_promoted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage6-root", type=Path, required=True)
    parser.add_argument("--assistant-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--aggregate-zip", type=Path, required=True)
    parser.add_argument("--additional-input-zip", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"event": "stage6_qualified_install_complete", **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
