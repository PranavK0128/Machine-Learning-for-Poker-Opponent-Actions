from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    parse_utc_timestamp,
    validate_document_self_hash,
)


REQUIRED_PROFILE_COLUMNS = (
    "canonical_player_id",
    "profile_snapshot_id",
    "profile_snapshot_cutoff_utc",
    "profile_source_max_event_utc",
    "identity_resolution_status",
    "total_hands",
    "games_played",
)
REQUIRED_METRIC_FIELDS = (
    "metric_id",
    "value_field",
    "numerator_field",
    "opportunity_denominator_field",
    "unit",
    "valid_min",
    "valid_max",
    "street_or_context_scope",
    "missing_value_semantics",
    "minimum_runtime_opportunity_support",
    "shrinkage_or_interval_method",
    "reconciliation_absolute_tolerance",
    "uncertainty_lower_field",
    "uncertainty_upper_field",
    "uncertainty_confidence_level",
    "uncertainty_required_for_runtime_support",
)
PROFILE_STATUSES = {"resolved", "cold_start_zero_history"}
METRIC_UNITS = {"percent", "proportion"}
PROFILE_RECEIPT_COLUMNS = (
    "authoritative_source_receipt",
    "alias_resolution_receipt",
)
SOURCE_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "canonical_player_id",
    "profile_snapshot_id",
    "source_dataset_id",
    "source_snapshot_id",
    "source_row_id",
    "source_row_sha256",
    "source_max_event_utc",
    "self_sha256",
}
ALIAS_RECEIPT_FIELDS = {
    "schema_version",
    "canonical_player_id",
    "profile_snapshot_id",
    "identity_resolution_status",
    "source_aliases",
    "resolver_id",
    "resolver_version",
    "self_sha256",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise Stage6ContractError(f"{label} must be a nonnegative integer")
    return int(value)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Stage6ContractError(f"{label} must be a finite numeric value")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Stage6ContractError(f"{label} must be a finite numeric value")
    return numeric


def validate_metric_registry(
    metric_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(metric_registry, (str, bytes)) or not isinstance(
        metric_registry, Sequence
    ):
        raise Stage6ContractError("profile metric registry must be a sequence")
    validated: list[dict[str, Any]] = []
    metric_ids: set[str] = set()
    value_fields: set[str] = set()
    for raw in metric_registry:
        if not isinstance(raw, Mapping):
            raise Stage6ContractError("profile metric registry row is not an object")
        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in raw]
        unknown = sorted(set(raw) - set(REQUIRED_METRIC_FIELDS))
        if missing or unknown:
            raise Stage6ContractError(
                f"profile metric registry schema mismatch: missing={missing}, unknown={unknown}"
            )
        row = dict(raw)
        for field in (
            "metric_id",
            "value_field",
            "numerator_field",
            "opportunity_denominator_field",
            "street_or_context_scope",
            "missing_value_semantics",
            "shrinkage_or_interval_method",
            "uncertainty_lower_field",
            "uncertainty_upper_field",
        ):
            row[field] = _canonical_text(row[field], label=f"metric_registry.{field}")
        metric_id = row["metric_id"]
        value_field = row["value_field"]
        if metric_id in metric_ids or value_field in value_fields:
            raise Stage6ContractError(
                f"duplicate profile metric id or value field: {metric_id}"
            )
        metric_ids.add(metric_id)
        value_fields.add(value_field)
        field_names = {
            value_field,
            row["numerator_field"],
            row["opportunity_denominator_field"],
            row["uncertainty_lower_field"],
            row["uncertainty_upper_field"],
        }
        if len(field_names) != 5:
            raise Stage6ContractError(f"{metric_id}: metric fields must be distinct")
        unit = _canonical_text(row["unit"], label=f"{metric_id}.unit")
        if unit not in METRIC_UNITS:
            raise Stage6ContractError(f"{metric_id}: unsupported unit {unit!r}")
        if row["missing_value_semantics"] != "null_tuple_means_not_observed":
            raise Stage6ContractError(
                f"{metric_id}: unsupported missing-value semantics"
            )
        if row["shrinkage_or_interval_method"].casefold() in {
            "none",
            "raw",
            "raw_rate",
            "not_applicable",
        }:
            raise Stage6ContractError(
                f"{metric_id}: support uncertainty method is not substantive"
            )
        lower = _finite_number(row["valid_min"], label=f"{metric_id}.valid_min")
        upper = _finite_number(row["valid_max"], label=f"{metric_id}.valid_max")
        expected_upper = 100.0 if unit == "percent" else 1.0
        if lower != 0.0 or upper != expected_upper:
            raise Stage6ContractError(
                f"{metric_id}: rate range must be [0, {expected_upper:g}]"
            )
        minimum_support = _nonnegative_integer(
            row["minimum_runtime_opportunity_support"],
            label=f"{metric_id}.minimum_runtime_opportunity_support",
        )
        if minimum_support <= 0:
            raise Stage6ContractError(
                f"{metric_id}: minimum runtime opportunity support must be positive"
            )
        tolerance = _finite_number(
            row["reconciliation_absolute_tolerance"],
            label=f"{metric_id}.reconciliation_absolute_tolerance",
        )
        if tolerance < 0:
            raise Stage6ContractError(
                f"{metric_id}: reconciliation tolerance must be nonnegative"
            )
        maximum_tolerance = 0.05 if unit == "percent" else 0.0005
        if tolerance > maximum_tolerance:
            raise Stage6ContractError(
                f"{metric_id}: reconciliation tolerance is too permissive"
            )
        confidence_level = _finite_number(
            row["uncertainty_confidence_level"],
            label=f"{metric_id}.uncertainty_confidence_level",
        )
        if not 0 < confidence_level < 1:
            raise Stage6ContractError(
                f"{metric_id}: uncertainty confidence level must be inside (0, 1)"
            )
        if row["uncertainty_required_for_runtime_support"] is not True:
            raise Stage6ContractError(
                f"{metric_id}: uncertainty must be required for runtime support"
            )
        row.update(
            {
                "unit": unit,
                "valid_min": lower,
                "valid_max": upper,
                "minimum_runtime_opportunity_support": minimum_support,
                "reconciliation_absolute_tolerance": tolerance,
                "uncertainty_confidence_level": confidence_level,
            }
        )
        validated.append(row)
    if not validated:
        raise Stage6ContractError("profile metric registry is empty")
    return sorted(validated, key=lambda row: row["metric_id"])


def _validate_alias_receipt(
    raw: Any, *, player_id: str, snapshot_id: str, status: str
) -> str:
    if not isinstance(raw, Mapping) or set(raw) != ALIAS_RECEIPT_FIELDS:
        raise Stage6ContractError(f"{player_id}: alias-resolution receipt schema is invalid")
    if raw.get("schema_version") != "stage6_alias_resolution_receipt_v1":
        raise Stage6ContractError(f"{player_id}: alias-resolution receipt version is invalid")
    if (
        raw.get("canonical_player_id") != player_id
        or raw.get("profile_snapshot_id") != snapshot_id
        or raw.get("identity_resolution_status") != status
    ):
        raise Stage6ContractError(f"{player_id}: alias-resolution receipt does not bind profile")
    for field in ("resolver_id", "resolver_version"):
        _canonical_text(raw.get(field), label=f"{player_id}.alias_receipt.{field}")
    aliases_raw = raw.get("source_aliases")
    if not isinstance(aliases_raw, list) or not aliases_raw:
        raise Stage6ContractError(f"{player_id}: alias-resolution receipt has no aliases")
    aliases = [
        _canonical_text(alias, label=f"{player_id}.alias_receipt.source_alias")
        for alias in aliases_raw
    ]
    if aliases != sorted(aliases) or len({alias.casefold() for alias in aliases}) != len(aliases):
        raise Stage6ContractError(f"{player_id}: alias receipt aliases are not canonical and unique")
    if player_id not in aliases:
        raise Stage6ContractError(f"{player_id}: canonical player is absent from alias receipt")
    return validate_document_self_hash(raw)


def _validate_source_receipt(
    raw: Any,
    *,
    player_id: str,
    snapshot_id: str,
    status: str,
    source_time_raw: Any,
) -> str:
    if not isinstance(raw, Mapping) or set(raw) != SOURCE_RECEIPT_FIELDS:
        raise Stage6ContractError(f"{player_id}: authoritative-source receipt schema is invalid")
    if raw.get("schema_version") != "stage6_profile_source_receipt_v1":
        raise Stage6ContractError(f"{player_id}: authoritative-source receipt version is invalid")
    if raw.get("canonical_player_id") != player_id or raw.get("profile_snapshot_id") != snapshot_id:
        raise Stage6ContractError(f"{player_id}: source receipt does not bind profile identity")
    expected_status = "authoritative_history" if status == "resolved" else "zero_history"
    if raw.get("status") != expected_status:
        raise Stage6ContractError(f"{player_id}: source receipt status is invalid")
    source_fields = (
        "source_dataset_id",
        "source_snapshot_id",
        "source_row_id",
        "source_row_sha256",
        "source_max_event_utc",
    )
    if status == "resolved":
        for field in source_fields[:3]:
            _canonical_text(raw.get(field), label=f"{player_id}.source_receipt.{field}")
        digest = raw.get("source_row_sha256")
        if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
            raise Stage6ContractError(f"{player_id}: source-row SHA-256 is invalid")
        if raw.get("source_max_event_utc") != source_time_raw:
            raise Stage6ContractError(f"{player_id}: source receipt timestamp disagrees with profile")
        parse_utc_timestamp(
            raw["source_max_event_utc"],
            label=f"{player_id}.source_receipt.source_max_event_utc",
        )
    elif any(raw.get(field) is not None for field in source_fields):
        raise Stage6ContractError(f"{player_id}: zero-history source receipt carries source data")
    return validate_document_self_hash(raw)


def audit_profile_snapshots(
    rows: Iterable[Mapping[str, Any]],
    metric_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise Stage6ContractError("refreshed profile dataset has no rows")
    registry = validate_metric_registry(metric_registry)
    metric_data_fields: set[str] = set()
    for metric in registry:
        for field in (
            metric["value_field"],
            metric["numerator_field"],
            metric["opportunity_denominator_field"],
            metric["uncertainty_lower_field"],
            metric["uncertainty_upper_field"],
        ):
            if field in metric_data_fields or field in REQUIRED_PROFILE_COLUMNS or field in PROFILE_RECEIPT_COLUMNS:
                raise Stage6ContractError(f"profile metric field collides with schema: {field}")
            metric_data_fields.add(field)
    expected_profile_fields = (
        set(REQUIRED_PROFILE_COLUMNS)
        | set(PROFILE_RECEIPT_COLUMNS)
        | metric_data_fields
    )
    seen_keys: set[tuple[str, str]] = set()
    player_casefold_bindings: dict[str, str] = {}
    snapshot_casefold_bindings: dict[str, str] = {}
    snapshot_owner_bindings: dict[str, str] = {}
    alias_casefold_bindings: dict[str, str] = {}
    alias_owner_bindings: dict[str, str] = {}
    source_row_identity_bindings: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    source_row_owner_bindings: dict[tuple[str, str, str], str] = {}
    source_row_hash_owner_bindings: dict[str, str] = {}
    players: set[str] = set()
    snapshots: set[str] = set()
    status_counts: Counter[str] = Counter()
    metric_counts = {
        row["metric_id"]: {
            "observed_rows": 0,
            "missing_rows": 0,
            "zero_opportunity_rows": 0,
            "runtime_supported_rows": 0,
            "below_runtime_support_rows": 0,
        }
        for row in registry
    }

    for raw in materialized:
        missing_exact = sorted(expected_profile_fields - set(raw))
        unknown_exact = sorted(set(raw) - expected_profile_fields)
        if missing_exact or unknown_exact:
            raise Stage6ContractError(
                "profile exact schema mismatch: "
                f"missing={missing_exact}, unknown={unknown_exact}"
            )
        missing_columns = [
            column for column in REQUIRED_PROFILE_COLUMNS if column not in raw
        ]
        if missing_columns:
            raise Stage6ContractError(
                f"profile row is missing columns {missing_columns}"
            )
        player_id = _canonical_text(
            raw["canonical_player_id"], label="profile.canonical_player_id"
        )
        snapshot_id = _canonical_text(
            raw["profile_snapshot_id"], label=f"{player_id}.profile_snapshot_id"
        )
        normalized_player_id = player_id.casefold()
        existing_player_id = player_casefold_bindings.setdefault(
            normalized_player_id, player_id
        )
        if existing_player_id != player_id:
            raise Stage6ContractError(
                "case-only aliases cannot create distinct canonical players: "
                f"{existing_player_id!r} versus {player_id!r}"
            )
        normalized_snapshot_id = snapshot_id.casefold()
        existing_snapshot_id = snapshot_casefold_bindings.setdefault(
            normalized_snapshot_id, snapshot_id
        )
        if existing_snapshot_id != snapshot_id:
            raise Stage6ContractError(
                "case-only aliases cannot create distinct profile snapshots: "
                f"{existing_snapshot_id!r} versus {snapshot_id!r}"
            )
        existing_snapshot_owner = snapshot_owner_bindings.setdefault(
            normalized_snapshot_id, player_id
        )
        if existing_snapshot_owner != player_id:
            raise Stage6ContractError(
                f"profile snapshot {snapshot_id!r} is assigned to multiple canonical players"
            )
        key = (player_id, snapshot_id)
        if key in seen_keys:
            raise Stage6ContractError(f"duplicate profile snapshot row: {key}")
        seen_keys.add(key)
        players.add(player_id)
        snapshots.add(snapshot_id)
        cutoff = parse_utc_timestamp(
            raw["profile_snapshot_cutoff_utc"],
            label=f"{player_id}.profile_snapshot_cutoff_utc",
        )
        status = raw["identity_resolution_status"]
        if status not in PROFILE_STATUSES:
            raise Stage6ContractError(
                f"{player_id}: invalid identity_resolution_status {status!r}"
            )
        total_hands = _nonnegative_integer(
            raw["total_hands"], label=f"{player_id}.total_hands"
        )
        games_played = _nonnegative_integer(
            raw["games_played"], label=f"{player_id}.games_played"
        )
        source_time_raw = raw["profile_source_max_event_utc"]
        if status == "resolved":
            if total_hands <= 0:
                raise Stage6ContractError(
                    f"{player_id}: resolved profile has no historical hands"
                )
            if games_played <= 0 or games_played > total_hands:
                raise Stage6ContractError(
                    f"{player_id}: resolved games_played is inconsistent with hands"
                )
            source_time = parse_utc_timestamp(
                source_time_raw, label=f"{player_id}.profile_source_max_event_utc"
            )
            if source_time > cutoff:
                raise Stage6ContractError(
                    f"{player_id}: profile source exceeds its snapshot cutoff"
                )
        else:
            if total_hands != 0 or games_played != 0 or source_time_raw is not None:
                raise Stage6ContractError(
                    f"{player_id}: zero-history profile carries historical data"
                )
        status_counts[status] += 1
        _validate_alias_receipt(
            raw["alias_resolution_receipt"],
            player_id=player_id,
            snapshot_id=snapshot_id,
            status=status,
        )
        _validate_source_receipt(
            raw["authoritative_source_receipt"],
            player_id=player_id,
            snapshot_id=snapshot_id,
            status=status,
            source_time_raw=source_time_raw,
        )
        alias_receipt = raw["alias_resolution_receipt"]
        for alias in alias_receipt["source_aliases"]:
            normalized_alias = alias.casefold()
            existing_alias = alias_casefold_bindings.setdefault(
                normalized_alias, alias
            )
            if existing_alias != alias:
                raise Stage6ContractError(
                    "case-only source aliases are forbidden across profile rows: "
                    f"{existing_alias!r} versus {alias!r}"
                )
            existing_owner = alias_owner_bindings.setdefault(
                normalized_alias, player_id
            )
            if existing_owner != player_id:
                raise Stage6ContractError(
                    f"source alias {alias!r} is assigned to multiple canonical players"
                )
        if status == "resolved":
            source_receipt = raw["authoritative_source_receipt"]
            source_identity = (
                source_receipt["source_dataset_id"],
                source_receipt["source_snapshot_id"],
                source_receipt["source_row_id"],
            )
            normalized_source_identity = tuple(
                value.casefold() for value in source_identity
            )
            existing_source_identity = source_row_identity_bindings.setdefault(
                normalized_source_identity, source_identity
            )
            if existing_source_identity != source_identity:
                raise Stage6ContractError(
                    "case-only authoritative source-row identities are forbidden"
                )
            existing_source_owner = source_row_owner_bindings.setdefault(
                normalized_source_identity, player_id
            )
            if existing_source_owner != player_id:
                raise Stage6ContractError(
                    "one authoritative source row is assigned to multiple canonical players"
                )
            source_row_sha256 = source_receipt["source_row_sha256"]
            existing_hash_owner = source_row_hash_owner_bindings.setdefault(
                source_row_sha256, player_id
            )
            if existing_hash_owner != player_id:
                raise Stage6ContractError(
                    "one authoritative source-row hash is assigned to multiple canonical players"
                )

        for metric in registry:
            metric_id = metric["metric_id"]
            value_field = metric["value_field"]
            numerator_field = metric["numerator_field"]
            denominator_field = metric["opportunity_denominator_field"]
            lower_field = metric["uncertainty_lower_field"]
            upper_field = metric["uncertainty_upper_field"]
            for field in (
                value_field,
                numerator_field,
                denominator_field,
                lower_field,
                upper_field,
            ):
                if field not in raw:
                    raise Stage6ContractError(f"{player_id}: missing metric field {field}")
            value_raw = raw[value_field]
            numerator_raw = raw[numerator_field]
            denominator_raw = raw[denominator_field]
            lower_raw = raw[lower_field]
            upper_raw = raw[upper_field]
            if denominator_raw is None:
                if (
                    value_raw is not None
                    or numerator_raw is not None
                    or lower_raw is not None
                    or upper_raw is not None
                ):
                    raise Stage6ContractError(
                        f"{player_id}.{metric_id}: partial missing metric tuple"
                    )
                metric_counts[metric_id]["missing_rows"] += 1
                continue
            denominator = _nonnegative_integer(
                denominator_raw, label=f"{player_id}.{metric_id}.opportunities"
            )
            numerator = _nonnegative_integer(
                numerator_raw, label=f"{player_id}.{metric_id}.numerator"
            )
            if numerator > denominator:
                raise Stage6ContractError(
                    f"{player_id}.{metric_id}: numerator exceeds opportunities"
                )
            if status == "cold_start_zero_history" and denominator != 0:
                raise Stage6ContractError(
                    f"{player_id}.{metric_id}: zero-history profile has observations"
                )
            if denominator == 0:
                if (
                    numerator != 0
                    or value_raw is not None
                    or lower_raw is not None
                    or upper_raw is not None
                ):
                    raise Stage6ContractError(
                        f"{player_id}.{metric_id}: zero opportunities must have null value"
                    )
                metric_counts[metric_id]["zero_opportunity_rows"] += 1
                continue
            value = _finite_number(
                value_raw, label=f"{player_id}.{metric_id}.value"
            )
            if value < metric["valid_min"] or value > metric["valid_max"]:
                raise Stage6ContractError(
                    f"{player_id}.{metric_id}: value is outside the registered range"
                )
            lower = _finite_number(
                lower_raw, label=f"{player_id}.{metric_id}.uncertainty_lower"
            )
            upper = _finite_number(
                upper_raw, label=f"{player_id}.{metric_id}.uncertainty_upper"
            )
            if (
                lower < metric["valid_min"]
                or upper > metric["valid_max"]
                or lower > value
                or value > upper
            ):
                raise Stage6ContractError(
                    f"{player_id}.{metric_id}: uncertainty interval does not contain the value"
                )
            scale = 100.0 if metric["unit"] == "percent" else 1.0
            expected_value = scale * numerator / denominator
            if not math.isclose(
                value,
                expected_value,
                rel_tol=0.0,
                abs_tol=metric["reconciliation_absolute_tolerance"],
            ):
                raise Stage6ContractError(
                    f"{player_id}.{metric_id}: value disagrees with numerator/opportunities"
                )
            metric_counts[metric_id]["observed_rows"] += 1
            if denominator >= metric["minimum_runtime_opportunity_support"]:
                metric_counts[metric_id]["runtime_supported_rows"] += 1
            else:
                metric_counts[metric_id]["below_runtime_support_rows"] += 1

    ordered_rows = sorted(
        materialized,
        key=lambda row: (row["canonical_player_id"], row["profile_snapshot_id"]),
    )
    row_root_digest = hashlib.sha256(canonical_json_bytes(ordered_rows)).hexdigest()
    registry_root_digest = hashlib.sha256(canonical_json_bytes(registry)).hexdigest()
    return {
        "schema_version": "stage6_profile_oracle_report_v1",
        "status": "PASS",
        "runtime_authority": False,
        "promotion_authority": False,
        "row_count": len(materialized),
        "unique_player_count": len(players),
        "snapshot_id_count": len(snapshots),
        "identity_status_counts": dict(sorted(status_counts.items())),
        "profile_row_root_digest": row_root_digest,
        "metric_registry_root_digest": registry_root_digest,
        "metric_support_counts": metric_counts,
    }
