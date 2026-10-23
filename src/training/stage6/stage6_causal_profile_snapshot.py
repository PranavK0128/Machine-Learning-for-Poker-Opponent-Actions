from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from numbers import Integral
from typing import Any, Iterable, Mapping

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    parse_utc_timestamp,
)
from stage6_profile_metric_registry import validate_profile_metric_registry_document


COLD_START_CUTOFF_UTC = "1970-01-01T00:00:00.000Z"
WILSON_95_Z = 1.959963984540054


def _canonical_text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise Stage6ContractError(f"{label} must be a nonnegative integer")
    return int(value)


def _binary_integer(value: Any, *, label: str) -> int:
    numeric = _nonnegative_integer(value, label=label)
    if numeric not in (0, 1):
        raise Stage6ContractError(f"{label} must be zero or one")
    return numeric


def _utc_epoch_millis(value: Any, *, label: str) -> int:
    parsed = parse_utc_timestamp(value, label=label)
    delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    if delta.microseconds % 1000:
        raise Stage6ContractError(f"{label} has sub-millisecond precision")
    return (
        delta.days * 86_400_000
        + delta.seconds * 1000
        + delta.microseconds // 1000
    )


def wilson_interval(
    numerator: int, denominator: int, *, z: float = WILSON_95_Z
) -> tuple[float | None, float | None]:
    numerator = _nonnegative_integer(numerator, label="wilson.numerator")
    denominator = _nonnegative_integer(denominator, label="wilson.denominator")
    if numerator > denominator:
        raise Stage6ContractError("wilson numerator exceeds denominator")
    if denominator == 0:
        return None, None
    proportion = numerator / denominator
    z_squared = z * z
    scale = 1.0 + z_squared / denominator
    center = (proportion + z_squared / (2.0 * denominator)) / scale
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + z_squared / (4.0 * denominator * denominator)
        )
        / scale
    )
    lower = 0.0 if numerator == 0 else max(0.0, center - margin)
    upper = 1.0 if numerator == denominator else min(1.0, center + margin)
    return lower, upper


def _metric_fields(
    cumulative: Mapping[str, Mapping[str, int]], metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in metrics:
        metric_id = metric["metric_id"]
        numerator = cumulative[metric_id]["numerator"]
        denominator = cumulative[metric_id]["opportunity"]
        lower, upper = wilson_interval(numerator, denominator)
        output[metric["numerator_field"]] = numerator
        output[metric["opportunity_denominator_field"]] = denominator
        output[metric["value_field"]] = (
            numerator / denominator if denominator else None
        )
        output[metric["uncertainty_lower_field"]] = lower
        output[metric["uncertainty_upper_field"]] = upper
    return output


def build_reference_causal_snapshots(
    decision_rows: Iterable[Mapping[str, Any]],
    observation_rows: Iterable[Mapping[str, Any]],
    metric_registry_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Build bounded oracle snapshots using only completed hands before decisions.

    This is the deterministic reference kernel for Notebook 01 parity tests. The
    full dataset implementation may use a columnar/as-of engine, but it must
    reproduce these exact strict-time and cumulative-count semantics.
    """

    registry = validate_profile_metric_registry_document(metric_registry_document)
    metrics = registry["metrics"]
    bindings_by_id = {
        binding["metric_id"]: binding for binding in registry["bindings"]
    }

    observations_by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases_by_player: dict[str, set[str]] = defaultdict(set)
    observation_ordinals: set[int] = set()
    observation_source_keys: set[tuple[str, str, str]] = set()
    for raw in observation_rows:
        row = dict(raw)
        ordinal = _nonnegative_integer(
            row.get("source_row_ordinal"), label="observation.source_row_ordinal"
        )
        if ordinal in observation_ordinals:
            raise Stage6ContractError(f"duplicate observation source row ordinal: {ordinal}")
        observation_ordinals.add(ordinal)
        player_id = _canonical_text(
            row.get("canonical_group_player_id"),
            label=f"observation[{ordinal}].canonical_group_player_id",
        )
        pokernow_player_id = _canonical_text(
            row.get("pokernow_player_id"),
            label=f"observation[{ordinal}].pokernow_player_id",
        )
        aliases_by_player[player_id].update((player_id, pokernow_player_id))
        game_id = _canonical_text(
            row.get("game_id"), label=f"observation[{ordinal}].game_id"
        )
        hand_id = _canonical_text(
            row.get("hand_id"), label=f"observation[{ordinal}].hand_id"
        )
        source_key = (pokernow_player_id, game_id, hand_id)
        if source_key in observation_source_keys:
            raise Stage6ContractError(f"duplicate PokerNow player-hand observation: {source_key}")
        observation_source_keys.add(source_key)
        completion_utc = _canonical_text(
            row.get("hand_completed_utc"),
            label=f"observation[{ordinal}].hand_completed_utc",
        )
        completion_ms = _nonnegative_integer(
            row.get("hand_completed_epoch_ms"),
            label=f"observation[{ordinal}].hand_completed_epoch_ms",
        )
        if _utc_epoch_millis(
            completion_utc, label=f"observation[{ordinal}].hand_completed_utc"
        ) != completion_ms:
            raise Stage6ContractError(
                f"observation[{ordinal}] completion timestamp representations disagree"
            )
        if row.get("observed_hand") != 1:
            raise Stage6ContractError(
                f"observation[{ordinal}].observed_hand must equal one"
            )
        metric_contributions: dict[str, dict[str, int]] = {}
        for metric in metrics:
            metric_id = metric["metric_id"]
            binding = bindings_by_id[metric_id]
            presence_field = binding["source_presence_field"]
            source_present = _binary_integer(
                row.get(presence_field),
                label=f"observation[{ordinal}].{presence_field}",
            )
            all_terms = set(binding["numerator_terms"]) | set(
                binding["opportunity_terms"]
            )
            if not source_present:
                nonzero_terms: list[str] = []
                for term in all_terms:
                    raw_term = row.get(term)
                    if raw_term in (None, ""):
                        continue
                    if _nonnegative_integer(
                        raw_term, label=f"observation[{ordinal}].{term}"
                    ) != 0:
                        nonzero_terms.append(term)
                if nonzero_terms:
                    raise Stage6ContractError(
                        f"observation[{ordinal}].{metric_id} has values without its source: "
                        f"{sorted(nonzero_terms)}"
                    )
                numerator = 0
                opportunity = 0
            else:
                numerator = sum(
                    _nonnegative_integer(
                        row.get(term), label=f"observation[{ordinal}].{term}"
                    )
                    for term in binding["numerator_terms"]
                )
                opportunity = sum(
                    _nonnegative_integer(
                        row.get(term), label=f"observation[{ordinal}].{term}"
                    )
                    for term in binding["opportunity_terms"]
                )
            if numerator > opportunity:
                raise Stage6ContractError(
                    f"observation[{ordinal}].{metric_id} numerator exceeds opportunities"
                )
            metric_contributions[metric_id] = {
                "numerator": numerator,
                "opportunity": opportunity,
            }
        observations_by_player[player_id].append(
            {
                "source_row_ordinal": ordinal,
                "pokernow_player_id": pokernow_player_id,
                "game_id": game_id,
                "hand_id": hand_id,
                "completion_utc": completion_utc,
                "completion_ms": completion_ms,
                "metric_contributions": metric_contributions,
            }
        )

    decisions_by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_ordinals: set[int] = set()
    for raw in decision_rows:
        row = dict(raw)
        ordinal = _nonnegative_integer(
            row.get("source_row_ordinal"), label="decision.source_row_ordinal"
        )
        if ordinal in decision_ordinals:
            raise Stage6ContractError(f"duplicate decision source row ordinal: {ordinal}")
        decision_ordinals.add(ordinal)
        player_id = _canonical_text(
            row.get("canonical_group_player_id"),
            label=f"decision[{ordinal}].canonical_group_player_id",
        )
        aliases_by_player[player_id].add(player_id)
        actor_pokernow_player_id = row.get("actor_pokernow_player_id")
        if actor_pokernow_player_id not in (None, ""):
            aliases_by_player[player_id].add(
                _canonical_text(
                    actor_pokernow_player_id,
                    label=f"decision[{ordinal}].actor_pokernow_player_id",
                )
            )
        decision_utc = _canonical_text(
            row.get("decision_utc"), label=f"decision[{ordinal}].decision_utc"
        )
        decision_ms = _nonnegative_integer(
            row.get("decision_epoch_ms"),
            label=f"decision[{ordinal}].decision_epoch_ms",
        )
        if _utc_epoch_millis(
            decision_utc, label=f"decision[{ordinal}].decision_utc"
        ) != decision_ms:
            raise Stage6ContractError(
                f"decision[{ordinal}] timestamp representations disagree"
            )
        decisions_by_player[player_id].append(
            {
                "source_row_ordinal": ordinal,
                "player_id": player_id,
                "decision_utc": decision_utc,
                "decision_ms": decision_ms,
            }
        )

    snapshot_rows_by_id: dict[str, dict[str, Any]] = {}
    decision_bindings: list[dict[str, Any]] = []
    all_players = sorted(decisions_by_player)
    for player_id in all_players:
        observations = sorted(
            observations_by_player.get(player_id, []),
            key=lambda row: (
                row["completion_ms"],
                row["game_id"],
                row["hand_id"],
                row["source_row_ordinal"],
            ),
        )
        decisions = sorted(
            decisions_by_player[player_id],
            key=lambda row: (row["decision_ms"], row["source_row_ordinal"]),
        )
        cumulative = {
            metric["metric_id"]: {"numerator": 0, "opportunity": 0}
            for metric in metrics
        }
        hands: set[tuple[str, str]] = set()
        games: set[str] = set()
        observation_index = 0
        source_max_utc: str | None = None
        source_chain = hashlib.sha256(
            canonical_json_bytes(
                {
                    "algorithm": "stage6_ordered_profile_source_chain_v1",
                    "canonical_player_id": player_id,
                    "source_dataset_id": registry["document"]["source_dataset_id"],
                    "registry_self_sha256": registry["document"]["self_sha256"],
                }
            )
        ).digest()
        cached_snapshot_row: dict[str, Any] | None = None

        for decision in decisions:
            source_state_changed = False
            while (
                observation_index < len(observations)
                and observations[observation_index]["completion_ms"]
                < decision["decision_ms"]
            ):
                completion_ms = observations[observation_index]["completion_ms"]
                timestamp_group: list[dict[str, Any]] = []
                while (
                    observation_index < len(observations)
                    and observations[observation_index]["completion_ms"] == completion_ms
                ):
                    timestamp_group.append(observations[observation_index])
                    observation_index += 1
                for observation in timestamp_group:
                    hands.add((observation["game_id"], observation["hand_id"]))
                    games.add(observation["game_id"])
                    for metric in metrics:
                        metric_id = metric["metric_id"]
                        contribution = observation["metric_contributions"][metric_id]
                        cumulative[metric_id]["numerator"] += contribution["numerator"]
                        cumulative[metric_id]["opportunity"] += contribution["opportunity"]
                    source_record = {
                        "source_row_ordinal": observation["source_row_ordinal"],
                        "pokernow_player_id": observation["pokernow_player_id"],
                        "game_id": observation["game_id"],
                        "hand_id": observation["hand_id"],
                        "hand_completed_utc": observation["completion_utc"],
                        "metric_contributions": observation["metric_contributions"],
                    }
                    source_chain = hashlib.sha256(
                        source_chain + canonical_json_bytes(source_record)
                    ).digest()
                    source_state_changed = True
                source_max_utc = timestamp_group[-1]["completion_utc"]

            if cached_snapshot_row is None or source_state_changed:
                total_hands = len(hands)
                status = "resolved" if total_hands else "cold_start_zero_history"
                cutoff_utc = source_max_utc or COLD_START_CUTOFF_UTC
                metric_values = _metric_fields(cumulative, metrics)
                snapshot_body = {
                    "canonical_player_id": player_id,
                    "profile_snapshot_cutoff_utc": cutoff_utc,
                    "profile_source_max_event_utc": source_max_utc,
                    "identity_resolution_status": status,
                    "total_hands": total_hands,
                    "games_played": len(games),
                    **metric_values,
                }
                source_aliases = sorted(aliases_by_player[player_id])
                source_row_sha256 = source_chain.hex() if total_hands else None
                snapshot_digest = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "snapshot_body": snapshot_body,
                            "source_aliases": source_aliases,
                            "source_row_sha256": source_row_sha256,
                        }
                    )
                ).hexdigest()
                snapshot_id = f"stage6-profile-{snapshot_digest}"
                source_receipt = {
                    "schema_version": "stage6_profile_source_receipt_v1",
                    "status": (
                        "authoritative_history" if total_hands else "zero_history"
                    ),
                    "canonical_player_id": player_id,
                    "profile_snapshot_id": snapshot_id,
                    "source_dataset_id": (
                        registry["document"]["source_dataset_id"]
                        if total_hands
                        else None
                    ),
                    "source_snapshot_id": (
                        "stage6-ordered-profile-source-chain-v1-"
                        + registry["document"]["self_sha256"]
                        if total_hands
                        else None
                    ),
                    "source_row_id": (
                        f"stage6-profile-source-{snapshot_digest}"
                        if total_hands
                        else None
                    ),
                    "source_row_sha256": source_row_sha256,
                    "source_max_event_utc": source_max_utc,
                }
                source_receipt["self_sha256"] = document_self_sha256(source_receipt)
                alias_receipt = {
                    "schema_version": "stage6_alias_resolution_receipt_v1",
                    "canonical_player_id": player_id,
                    "profile_snapshot_id": snapshot_id,
                    "identity_resolution_status": status,
                    "source_aliases": source_aliases,
                    "resolver_id": "stage6-finalized-retrospective-identity",
                    "resolver_version": "1",
                }
                alias_receipt["self_sha256"] = document_self_sha256(alias_receipt)
                cached_snapshot_row = {
                    **snapshot_body,
                    "profile_snapshot_id": snapshot_id,
                    "authoritative_source_receipt": source_receipt,
                    "alias_resolution_receipt": alias_receipt,
                }
                existing = snapshot_rows_by_id.setdefault(
                    snapshot_id, cached_snapshot_row
                )
                if existing != cached_snapshot_row:
                    raise Stage6ContractError("profile snapshot digest collision")
            snapshot_row = cached_snapshot_row
            if snapshot_row is None:
                raise Stage6ContractError("profile snapshot cache was not initialized")
            decision_bindings.append(
                {
                    "source_row_ordinal": decision["source_row_ordinal"],
                    "canonical_player_id": player_id,
                    "decision_utc": decision["decision_utc"],
                    "profile_snapshot_id": snapshot_row["profile_snapshot_id"],
                    "profile_snapshot_cutoff_utc": snapshot_row[
                        "profile_snapshot_cutoff_utc"
                    ],
                    "profile_source_max_event_utc": snapshot_row[
                        "profile_source_max_event_utc"
                    ],
                    "actor_profile_status": snapshot_row[
                        "identity_resolution_status"
                    ],
                }
            )

    decision_bindings.sort(key=lambda row: row["source_row_ordinal"])
    snapshot_rows = sorted(
        snapshot_rows_by_id.values(),
        key=lambda row: (row["canonical_player_id"], row["profile_snapshot_id"]),
    )
    return {
        "schema_version": "stage6_reference_causal_profile_snapshots_v1",
        "status": "PASS_REFERENCE_CAUSAL_SNAPSHOT_BUILD",
        "runtime_authority": False,
        "decision_binding_count": len(decision_bindings),
        "profile_snapshot_count": len(snapshot_rows),
        "decision_bindings": decision_bindings,
        "profile_snapshots": snapshot_rows,
    }
