from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from stage6_common import Stage6ContractError, validate_document_self_hash
from stage6_profile_oracle import validate_metric_registry


REGISTRY_FIELDS = {
    "schema_version",
    "status",
    "runtime_authority",
    "source_dataset_id",
    "source_relative_path",
    "source_time_semantics",
    "rate_scale",
    "interval_method",
    "interval_confidence_level",
    "support_selection",
    "metrics",
    "observation_bindings",
    "self_sha256",
}
SUPPORT_SELECTION_FIELDS = {
    "selection_notebook_token",
    "selection_partition",
    "materialization_hold_sentinel",
    "sentinel_semantics",
}
BINDING_FIELDS = {
    "metric_id",
    "source_presence_field",
    "numerator_terms",
    "opportunity_terms",
}
HOLD_SENTINEL = 2_147_483_647


def _canonical_identifier(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or not value.replace("_", "a").isalnum()
    ):
        raise Stage6ContractError(f"{label} must be a canonical identifier")
    return value


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage6ContractError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    with path.open("r", encoding="utf-8", errors="strict") as source:
        value = json.load(source, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise Stage6ContractError("profile metric registry must be a JSON object")
    return value


def validate_profile_metric_registry_document(
    raw: Mapping[str, Any],
    *,
    observation_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    document = dict(raw)
    if set(document) != REGISTRY_FIELDS:
        raise Stage6ContractError("profile metric registry wrapper schema mismatch")
    validate_document_self_hash(document)
    if document["schema_version"] != "stage6_profile_metric_registry_hold_v1":
        raise Stage6ContractError("profile metric registry version is invalid")
    if document["status"] != "HOLD_RUNTIME_SUPPORT_SELECTION_PENDING_LOCKED_VALIDATION":
        raise Stage6ContractError("profile metric registry is not in the required hold state")
    if document["runtime_authority"] is not False:
        raise Stage6ContractError("held profile metric registry cannot carry runtime authority")
    if document["source_dataset_id"] != "stage6-hand-profile-observations-v1":
        raise Stage6ContractError("profile metric registry source dataset is invalid")
    if document["source_relative_path"] != "data/exports/ml/stage6-hand-profile-observations.csv":
        raise Stage6ContractError("profile metric registry source path is invalid")
    if document["source_time_semantics"] != "completed_hands_strictly_before_decision_utc":
        raise Stage6ContractError("profile metric registry time semantics are not causal")
    if (
        document["rate_scale"] != "proportion"
        or document["interval_method"] != "wilson_score_interval"
        or document["interval_confidence_level"] != 0.95
    ):
        raise Stage6ContractError("profile metric rate or interval policy is invalid")

    support = document["support_selection"]
    if not isinstance(support, Mapping) or set(support) != SUPPORT_SELECTION_FIELDS:
        raise Stage6ContractError("profile metric support-selection schema mismatch")
    if (
        support["selection_notebook_token"] != "05"
        or support["selection_partition"] != "locked_validation_only"
        or support["materialization_hold_sentinel"] != HOLD_SENTINEL
        or support["sentinel_semantics"]
        != "blocks_all_runtime_support_before_locked_validation_selection"
    ):
        raise Stage6ContractError("profile metric support selection is not fail-closed")

    metrics_raw = document["metrics"]
    metrics = validate_metric_registry(metrics_raw)
    metric_ids_in_document = [metric["metric_id"] for metric in metrics_raw]
    if len(metric_ids_in_document) != len(set(metric_ids_in_document)):
        raise Stage6ContractError("profile metric ids are duplicated")
    if any(
        metric["minimum_runtime_opportunity_support"] != HOLD_SENTINEL
        for metric in metrics
    ):
        raise Stage6ContractError("a profile metric bypasses the runtime support hold")

    bindings_raw = document["observation_bindings"]
    if not isinstance(bindings_raw, list):
        raise Stage6ContractError("profile observation bindings must be a list")
    binding_ids: list[str] = []
    required_columns: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for raw_binding in bindings_raw:
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != BINDING_FIELDS:
            raise Stage6ContractError("profile observation binding schema mismatch")
        binding = dict(raw_binding)
        metric_id = _canonical_identifier(binding["metric_id"], label="binding.metric_id")
        binding_ids.append(metric_id)
        presence_field = _canonical_identifier(
            binding["source_presence_field"],
            label=f"{metric_id}.source_presence_field",
        )
        binding["source_presence_field"] = presence_field
        required_columns.add(presence_field)
        for field in ("numerator_terms", "opportunity_terms"):
            terms = binding[field]
            if not isinstance(terms, list) or not terms:
                raise Stage6ContractError(f"{metric_id}.{field} must be a nonempty list")
            canonical_terms = [
                _canonical_identifier(term, label=f"{metric_id}.{field}") for term in terms
            ]
            if len(canonical_terms) != len(set(canonical_terms)):
                raise Stage6ContractError(f"{metric_id}.{field} contains duplicates")
            binding[field] = canonical_terms
            required_columns.update(canonical_terms)
        bindings.append(binding)
    if binding_ids != metric_ids_in_document:
        raise Stage6ContractError("profile observation bindings do not exactly cover metrics")
    if "raw_level3_raise_numerator" in required_columns:
        raise Stage6ContractError("raw level-3 raises cannot stand in for opener four-bets")
    if observation_columns is not None:
        missing = sorted(required_columns - set(observation_columns))
        if missing:
            raise Stage6ContractError(
                f"profile observation source is missing registry columns: {missing}"
            )

    return {
        "document": document,
        "metrics": metrics,
        "bindings": bindings,
        "required_observation_columns": sorted(required_columns),
    }


def load_profile_metric_registry(
    path: Path,
    *,
    observation_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    return validate_profile_metric_registry_document(
        _strict_json_object(path), observation_columns=observation_columns
    )
