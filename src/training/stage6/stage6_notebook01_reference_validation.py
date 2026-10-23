from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from stage6_causal_profile_snapshot import build_reference_causal_snapshots
from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    sha256_file,
)
from stage6_decision_row_builder import (
    build_decision_row,
    build_feature_role_manifest,
    classify_source_quarantine_reason,
)
from stage6_joined_data_oracle import audit_joined_decision_profiles
from stage6_notebook01_common import (
    recursive_stage6_code_identities,
    regular_file_identity,
)


DEFAULT_DECISION_LIMIT = 30_000
DEFAULT_OBSERVATION_LIMIT = 20_000
DEFAULT_RECEIPT = (
    "evidence/stage6_notebook01_joined_reference_validation_2026-08-21.json"
)


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
        raise Stage6ContractError(f"JSON authority is not an object: {path}")
    return value


def _integerize(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            try:
                row[field] = int(value)
            except (TypeError, ValueError) as error:
                raise Stage6ContractError(f"{field} is not an integer: {value!r}") from error
    return row


def _stream_identity(path: Path, *, relative_path: Path) -> dict[str, Any]:
    return regular_file_identity(path, relative_path=relative_path)


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
        if path.read_text(encoding="utf-8", errors="strict") != payload:
            raise Stage6ContractError("reference receipt durable readback mismatch")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_reference_validation(
    stage6_root: Path,
    *,
    decision_limit: int = DEFAULT_DECISION_LIMIT,
    observation_limit: int = DEFAULT_OBSERVATION_LIMIT,
) -> dict[str, Any]:
    started = time.perf_counter()
    stage6_root = stage6_root.resolve()
    if "OneDrive" in str(stage6_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    if decision_limit <= 0 or observation_limit <= 0:
        raise Stage6ContractError("reference validation limits must be positive")

    action_relative = Path("data/exports/ml/action-state-decision-rows.csv")
    identity_relative = Path(
        "data/exports/ml/stage6-action-decision-time-identity.csv"
    )
    observation_relative = Path(
        "data/exports/ml/stage6-hand-profile-observations.csv"
    )
    registry_relative = Path("contracts/stage6_profile_metric_registry_hold_v1.json")
    feature_manifest_relative = Path(
        "contracts/stage6_decision_feature_role_manifest_v1.json"
    )
    action_path = stage6_root / action_relative
    identity_path = stage6_root / identity_relative
    observation_path = stage6_root / observation_relative
    registry_path = stage6_root / registry_relative
    feature_manifest_path = stage6_root / feature_manifest_relative
    required = (
        action_path,
        identity_path,
        observation_path,
        registry_path,
        feature_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Stage6ContractError(f"reference validation inputs are missing: {missing}")

    registry_document = _strict_json_object(registry_path)
    feature_manifest = _strict_json_object(feature_manifest_path)
    generated_manifest = build_feature_role_manifest()
    if feature_manifest != generated_manifest:
        raise Stage6ContractError("frozen feature-role manifest differs from code-pinned roles")
    terms = sorted(
        {
            term
            for binding in registry_document["observation_bindings"]
            for field in ("numerator_terms", "opportunity_terms")
            for term in binding[field]
        }
    )
    presence_fields = sorted(
        {
            binding["source_presence_field"]
            for binding in registry_document["observation_bindings"]
        }
    )

    observations: list[dict[str, Any]] = []
    with observation_path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as source:
        for index, row in enumerate(csv.DictReader(source)):
            if index >= observation_limit:
                break
            observations.append(
                _integerize(
                    row,
                    (
                        "source_row_ordinal",
                        "hand_completed_epoch_ms",
                        "observed_hand",
                        *presence_fields,
                        *terms,
                    ),
                )
            )
    if len(observations) != observation_limit:
        raise Stage6ContractError("observation source is shorter than the bounded prefix")
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_reference_progress",
                "phase": "observations_loaded",
                "completed": len(observations),
                "total": observation_limit,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    actions: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    causal_decisions: list[dict[str, Any]] = []
    with action_path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as action_source, identity_path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as identity_source:
        action_reader = csv.DictReader(action_source)
        identity_reader = csv.DictReader(identity_source)
        for index, pair in enumerate(zip(action_reader, identity_reader)):
            if index >= decision_limit:
                break
            action, identity = pair
            identity = _integerize(
                identity,
                ("source_row_ordinal", "event_index", "decision_epoch_ms"),
            )
            if identity["source_row_ordinal"] != index:
                raise Stage6ContractError(
                    f"decision identity physical ordinal mismatch at row {index}"
                )
            actions.append(action)
            identities.append(identity)
            causal_decisions.append(
                {
                    "source_row_ordinal": identity["source_row_ordinal"],
                    "canonical_group_player_id": identity[
                        "canonical_group_player_id"
                    ],
                    "actor_pokernow_player_id": identity[
                        "actor_pokernow_player_id"
                    ],
                    "decision_utc": identity["decision_utc"],
                    "decision_epoch_ms": identity["decision_epoch_ms"],
                }
            )
    if len(identities) != decision_limit:
        raise Stage6ContractError("decision sources are shorter than the bounded prefix")

    causal = build_reference_causal_snapshots(
        causal_decisions, observations, registry_document
    )
    binding_by_ordinal = {
        row["source_row_ordinal"]: row for row in causal["decision_bindings"]
    }
    decision_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    quarantine_counts: Counter[str] = Counter()
    for action, identity in zip(actions, identities):
        reason = classify_source_quarantine_reason(action)
        if reason is not None:
            quarantine_counts[reason] += 1
            quarantine_rows.append(
                {
                    "source_row_ordinal": identity["source_row_ordinal"],
                    "decision_id": identity["hand_action_id"],
                    "game_id": identity["game_id"],
                    "hand_id": identity["hand_id"],
                    "decision_order": identity["event_index"],
                    "decision_utc": identity["decision_utc"],
                    "canonical_player_id": identity["canonical_group_player_id"],
                    "reason": reason,
                }
            )
            continue
        decision_rows.append(
            build_decision_row(
                action,
                identity,
                binding_by_ordinal[identity["source_row_ordinal"]],
            )
        )
    if len(decision_rows) + len(quarantine_rows) != decision_limit:
        raise Stage6ContractError("bounded decision work-unit conservation failed")

    joined_report = audit_joined_decision_profiles(
        decision_rows,
        causal["profile_snapshots"],
        registry_document["metrics"],
        feature_role_manifest=feature_manifest,
        expected_feature_role_manifest_sha256=feature_manifest["self_sha256"],
    )
    source_paths = (
        action_relative,
        identity_relative,
        observation_relative,
        registry_relative,
        feature_manifest_relative,
    )
    code_entry_paths = (
        Path("src/stage6_notebook01_reference_validation.py"),
    )
    receipt: dict[str, Any] = {
        "schema_version": "stage6_notebook01_joined_reference_validation_v1",
        "status": "PASS_BOUNDED_JOINED_DECISION_PROFILE_REFERENCE",
        "bounded_reference_only": True,
        "decision_prefix_limit": decision_limit,
        "observation_prefix_limit": observation_limit,
        "source_identities": {
            path.as_posix(): _stream_identity(stage6_root / path, relative_path=path)
            for path in source_paths
        },
        "code_identities": recursive_stage6_code_identities(
            stage6_root, code_entry_paths
        ),
        "decision_source_count": decision_limit,
        "accepted_decision_count": len(decision_rows),
        "quarantined_decision_count": len(quarantine_rows),
        "quarantine_reason_counts": dict(sorted(quarantine_counts.items())),
        "quarantine_root_digest": hashlib.sha256(
            canonical_json_bytes(quarantine_rows)
        ).hexdigest(),
        "work_unit_conservation_verified": (
            len(decision_rows) + len(quarantine_rows) == decision_limit
        ),
        "causal_decision_binding_count": causal["decision_binding_count"],
        "profile_snapshot_count": causal["profile_snapshot_count"],
        "joined_oracle_report": joined_report,
        "runtime_authority": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "drive_publication_authority": False,
        "live_action_authority": False,
    }
    receipt["self_sha256"] = document_self_sha256(receipt)
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_reference_complete",
                "accepted": len(decision_rows),
                "quarantined": len(quarantine_rows),
                "profile_snapshots": causal["profile_snapshot_count"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "receipt_self_sha256": receipt["self_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage6-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--decision-limit", type=int, default=DEFAULT_DECISION_LIMIT)
    parser.add_argument(
        "--observation-limit", type=int, default=DEFAULT_OBSERVATION_LIMIT
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_RECEIPT))
    args = parser.parse_args(argv)
    receipt = run_reference_validation(
        args.stage6_root,
        decision_limit=args.decision_limit,
        observation_limit=args.observation_limit,
    )
    output = args.output
    if not output.is_absolute():
        output = args.stage6_root.resolve() / output
    _atomic_write_json(output.resolve(), receipt)
    if _strict_json_object(output.resolve()) != receipt:
        raise Stage6ContractError("reference receipt JSON readback mismatch")
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_reference_receipt_written",
                "path": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
