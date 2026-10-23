from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

from stage6_common import Stage6ContractError, canonical_json_bytes
from stage6_data_oracle import audit_decision_rows
from stage6_profile_oracle import audit_profile_snapshots


def audit_joined_decision_profiles(
    decision_rows: Iterable[Mapping[str, Any]],
    profile_rows: Iterable[Mapping[str, Any]],
    metric_registry: Sequence[Mapping[str, Any]],
    *,
    feature_role_manifest: Mapping[str, Any],
    expected_feature_role_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate both datasets and bind each decision to one exact profile row."""

    decisions = [dict(row) for row in decision_rows]
    profiles = [dict(row) for row in profile_rows]
    decision_report = audit_decision_rows(
        decisions,
        feature_role_manifest=feature_role_manifest,
        expected_feature_role_manifest_sha256=(
            expected_feature_role_manifest_sha256
        ),
    )
    profile_report = audit_profile_snapshots(profiles, metric_registry)

    profiles_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for profile in profiles:
        key = (
            str(profile["canonical_player_id"]),
            str(profile["profile_snapshot_id"]),
        )
        if key in profiles_by_key:
            raise Stage6ContractError(f"duplicate joined profile identity: {key}")
        profiles_by_key[key] = profile

    join_receipts: list[dict[str, Any]] = []
    for decision in decisions:
        key = (str(decision["actor_id"]), str(decision["profile_snapshot_id"]))
        profile = profiles_by_key.get(key)
        if profile is None:
            raise Stage6ContractError(
                f"{decision['row_id']}: no exact actor/profile snapshot join for {key}"
            )
        comparisons = {
            "profile_snapshot_cutoff_utc": "profile_snapshot_cutoff_utc",
            "profile_source_max_event_utc": "profile_source_max_event_utc",
            "actor_profile_status": "identity_resolution_status",
        }
        mismatches = [
            decision_field
            for decision_field, profile_field in comparisons.items()
            if decision[decision_field] != profile[profile_field]
        ]
        if mismatches:
            raise Stage6ContractError(
                f"{decision['row_id']}: decision/profile receipt mismatch: {mismatches}"
            )
        join_receipts.append(
            {
                "row_id": decision["row_id"],
                "actor_id": decision["actor_id"],
                "profile_snapshot_id": decision["profile_snapshot_id"],
                "profile_snapshot_cutoff_utc": decision[
                    "profile_snapshot_cutoff_utc"
                ],
                "profile_source_max_event_utc": decision[
                    "profile_source_max_event_utc"
                ],
                "actor_profile_status": decision["actor_profile_status"],
                "profile_row_sha256": hashlib.sha256(
                    canonical_json_bytes(profile)
                ).hexdigest(),
            }
        )
    join_receipts.sort(key=lambda row: str(row["row_id"]))
    return {
        "schema_version": "stage6_joined_decision_profile_oracle_v1",
        "status": "PASS",
        "decision_row_count": len(decisions),
        "profile_row_count": len(profiles),
        "matched_decision_count": len(join_receipts),
        "unmatched_decision_count": 0,
        "decision_row_root_digest": decision_report["decision_row_root_digest"],
        "profile_row_root_digest": profile_report["profile_row_root_digest"],
        "feature_role_manifest_sha256": decision_report[
            "feature_role_manifest_sha256"
        ],
        "metric_registry_root_digest": profile_report[
            "metric_registry_root_digest"
        ],
        "join_receipt_root_digest": hashlib.sha256(
            canonical_json_bytes(join_receipts)
        ).hexdigest(),
        "runtime_authority": False,
        "promotion_authority": False,
    }
