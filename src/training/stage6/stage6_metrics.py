from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from stage6_common import Stage6ContractError


LOG_LOSS_CLIP_EPSILON = 1e-15


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


def _strict_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise Stage6ContractError(f"{label} must be strict numeric data")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Stage6ContractError(f"{label} must be finite")
    return numeric


def _ordered_values(values: Any, field_name: str) -> list[Any]:
    if isinstance(values, (str, bytes, Mapping, set, frozenset)):
        raise Stage6ContractError(f"{field_name} must be an ordered sequence")
    try:
        return list(values)
    except TypeError as exc:
        raise Stage6ContractError(
            f"{field_name} must be an ordered sequence"
        ) from exc


def _validated_multiclass_inputs(
    y_true: Sequence[Any], probabilities: Any, labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_list = _ordered_values(labels, "labels")
    if (
        not label_list
        or any(
            type(label) is not str
            or not label
            or label != label.strip()
            or label != unicodedata.normalize("NFC", label)
            for label in label_list
        )
        or len(set(label_list)) != len(label_list)
        or len({label.casefold() for label in label_list}) != len(label_list)
    ):
        raise Stage6ContractError("labels must be unique canonical strings")
    truth_values = _ordered_values(y_true, "truth")
    if any(
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        for value in truth_values
    ):
        raise Stage6ContractError("truth must contain canonical strings")
    truth = np.asarray(truth_values, dtype=object)
    try:
        raw_matrix = np.asarray(probabilities)
    except (TypeError, ValueError) as exc:
        raise Stage6ContractError("probabilities must form a rectangular matrix") from exc
    if np.issubdtype(raw_matrix.dtype, np.bool_):
        raise Stage6ContractError("boolean probabilities are not allowed")
    if raw_matrix.ndim == 2 and any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_matrix.ravel()
    ):
        raise Stage6ContractError("probabilities must be strict numeric data")
    try:
        matrix = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise Stage6ContractError("probabilities must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape != (len(truth), len(label_list)):
        raise Stage6ContractError("probability matrix shape does not match rows and labels")
    if len(truth) == 0:
        raise Stage6ContractError("metric input has no rows")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise Stage6ContractError("probabilities must be finite and nonnegative")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-8):
        raise Stage6ContractError("each probability row must sum to one")
    unknown = sorted(set(truth_values) - set(label_list))
    if unknown:
        raise Stage6ContractError(f"truth contains labels outside schema: {unknown}")
    return truth, matrix, label_list


def expected_calibration_error(
    y_true: Sequence[Any],
    probabilities: Any,
    labels: Sequence[str],
    *,
    bin_count: int = 15,
) -> dict[str, Any]:
    if type(bin_count) is not int or bin_count <= 0:
        raise Stage6ContractError("bin_count must be a positive integer")
    truth, matrix, label_list = _validated_multiclass_inputs(
        y_true, probabilities, labels
    )
    predicted_indices = matrix.argmax(axis=1)
    predicted = np.asarray([label_list[index] for index in predicted_indices])
    confidence = matrix.max(axis=1)
    correct = predicted == truth
    bin_indices = np.minimum((confidence * bin_count).astype(int), bin_count - 1)
    bins = []
    ece = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        if count:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            weight = count / len(truth)
            ece += weight * abs(accuracy - mean_confidence)
        else:
            accuracy = None
            mean_confidence = None
        bins.append(
            {
                "bin_index": bin_index,
                "lower_inclusive": lower,
                "upper_exclusive_except_final": upper,
                "row_count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "method": "top_label_fixed_width",
        "bin_count": bin_count,
        "bin_assignment": "floor_confidence_times_bin_count_final_bin_inclusive",
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def multiclass_metric_report(
    y_true: Sequence[Any], probabilities: Any, labels: Sequence[str]
) -> dict[str, Any]:
    truth, matrix, label_list = _validated_multiclass_inputs(
        y_true, probabilities, labels
    )
    predicted = np.asarray([label_list[index] for index in matrix.argmax(axis=1)])
    label_to_index = {label: index for index, label in enumerate(label_list)}
    one_hot = np.zeros_like(matrix)
    one_hot[
        np.arange(len(truth)),
        np.asarray([label_to_index[str(value)] for value in truth]),
    ] = 1.0
    true_probabilities = matrix[
        np.arange(len(truth)),
        np.asarray([label_to_index[str(value)] for value in truth]),
    ]
    mean_log_loss = -float(
        np.mean(np.log(np.clip(true_probabilities, LOG_LOSS_CLIP_EPSILON, 1.0)))
    )
    calibration = expected_calibration_error(truth, matrix, label_list)
    observed_truth_labels = [
        label for label in label_list if np.count_nonzero(truth == label) > 0
    ]
    balanced_accuracy = float(
        np.mean(
            [
                np.mean(predicted[truth == label] == label)
                for label in observed_truth_labels
            ]
        )
    )
    return {
        "row_count": len(truth),
        "label_order": label_list,
        "class_support": {
            label: int(np.count_nonzero(truth == label)) for label in label_list
        },
        "log_loss": mean_log_loss,
        "log_loss_clip_epsilon": LOG_LOSS_CLIP_EPSILON,
        "multiclass_brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=label_list,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_scope": "observed_truth_classes",
        "expected_calibration_error": calibration["expected_calibration_error"],
        "calibration": calibration,
    }


def selective_risk_curve(
    y_true: Sequence[Any],
    probabilities: Any,
    labels: Sequence[str],
    *,
    coverages: Sequence[float],
) -> list[dict[str, Any]]:
    truth, matrix, label_list = _validated_multiclass_inputs(
        y_true, probabilities, labels
    )
    raw_coverages = _ordered_values(coverages, "coverages")
    coverage_values = [
        _strict_real(value, label="coverage") for value in raw_coverages
    ]
    if (
        not coverage_values
        or any(not math.isfinite(value) or value <= 0 or value > 1 for value in coverage_values)
        or coverage_values != sorted(set(coverage_values))
    ):
        raise Stage6ContractError(
            "coverages must be unique ascending finite values in (0, 1]"
        )
    predicted_indices = matrix.argmax(axis=1)
    predicted = np.asarray([label_list[index] for index in predicted_indices])
    confidence = matrix.max(axis=1)
    order = np.lexsort((np.arange(len(truth)), -confidence))
    reports = []
    for requested in coverage_values:
        accepted_count = max(1, int(math.ceil(requested * len(truth))))
        accepted = order[:accepted_count]
        error_rate = float((predicted[accepted] != truth[accepted]).mean())
        reports.append(
            {
                "requested_coverage": requested,
                "actual_coverage": accepted_count / len(truth),
                "accepted_rows": accepted_count,
                "selective_risk": error_rate,
                "minimum_accepted_confidence": float(confidence[accepted].min()),
                "tie_breaker": "original_row_order",
            }
        )
    return reports


def paired_cluster_bootstrap_mean_difference(
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    cluster_ids: Sequence[Any],
    *,
    bootstrap_replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    raw_candidate = np.asarray(candidate_losses)
    raw_baseline = np.asarray(baseline_losses)
    if np.issubdtype(raw_candidate.dtype, np.bool_) or np.issubdtype(
        raw_baseline.dtype, np.bool_
    ):
        raise Stage6ContractError("boolean bootstrap losses are not allowed")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in [*raw_candidate.ravel(), *raw_baseline.ravel()]
    ):
        raise Stage6ContractError("paired bootstrap losses must be strict numeric data")
    try:
        candidate = np.asarray(candidate_losses, dtype=float)
        baseline = np.asarray(baseline_losses, dtype=float)
    except (TypeError, ValueError) as exc:
        raise Stage6ContractError("paired bootstrap losses must be numeric") from exc
    cluster_values = _ordered_values(cluster_ids, "cluster_ids")
    if any(
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        for value in cluster_values
    ):
        raise Stage6ContractError("cluster_ids must contain canonical strings")
    casefold_bindings: dict[str, str] = {}
    for cluster in cluster_values:
        existing = casefold_bindings.setdefault(cluster.casefold(), cluster)
        if existing != cluster:
            raise Stage6ContractError(
                f"case-only cluster aliases are forbidden: {existing!r} versus {cluster!r}"
            )
    clusters = np.asarray(cluster_values, dtype=object)
    if (
        candidate.ndim != 1
        or baseline.shape != candidate.shape
        or clusters.shape != candidate.shape
        or len(candidate) == 0
    ):
        raise Stage6ContractError("paired bootstrap inputs must be aligned nonempty vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise Stage6ContractError("paired bootstrap losses must be finite")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise Stage6ContractError("bootstrap_replicates must be an integer of at least 100")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, int) or seed < 0:
        raise Stage6ContractError("bootstrap seed must be a nonnegative integer")
    normalized_alpha = _strict_real(alpha, label="bootstrap alpha")
    if (
        not math.isfinite(normalized_alpha)
        or normalized_alpha <= 0
        or normalized_alpha >= 1
    ):
        raise Stage6ContractError("bootstrap alpha must be in (0, 1)")
    unique_clusters = np.asarray(sorted(set(clusters)), dtype=object)
    cluster_rows = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    differences = candidate - baseline
    generator = np.random.default_rng(seed)
    replicates = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled_clusters = generator.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        sampled_rows = np.concatenate(
            [cluster_rows[cluster] for cluster in sampled_clusters]
        )
        replicates[replicate] = differences[sampled_rows].mean()
    lower, upper = np.quantile(
        replicates,
        [normalized_alpha / 2, 1 - normalized_alpha / 2],
        method="linear",
    )
    return {
        "method": "paired_nonparametric_cluster_bootstrap_percentile",
        "difference_direction": "candidate_minus_baseline_lower_is_better",
        "row_count": len(candidate),
        "cluster_count": len(unique_clusters),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "alpha": normalized_alpha,
        "mean_difference": float(differences.mean()),
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
    }


def holm_bonferroni_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise Stage6ContractError("p-value mapping is empty")
    normalized = {}
    casefold_names: dict[str, str] = {}
    for name, raw_value in p_values.items():
        name = _canonical_text(name, label="comparison name")
        existing_name = casefold_names.setdefault(name.casefold(), name)
        if existing_name != name:
            raise Stage6ContractError(
                f"case-only comparison aliases are forbidden: {existing_name!r} versus {name!r}"
            )
        value = _strict_real(raw_value, label=f"p-value for {name}")
        if not math.isfinite(value) or value < 0 or value > 1:
            raise Stage6ContractError(f"invalid p-value for {name}")
        normalized[name] = value
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[name] = running_max
    return {name: adjusted[name] for name in sorted(adjusted)}
