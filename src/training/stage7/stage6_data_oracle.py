from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Mapping

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    finite_nonnegative,
    parse_utc_timestamp,
    validate_document_self_hash,
)


ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
LEGAL_COLUMNS = {action: f"legal_{action}" for action in ACTIONS}
PROFILE_STATUSES = {"resolved", "cold_start_zero_history"}
REQUIRED_COLUMNS = (
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "decision_utc",
    "decision_order",
    "actor_id",
    "street",
    "board_texture_present",
    "board_card_count",
    "board_high_card_value",
    "board_pair_level",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "board_connectedness_score",
    "active_opponent_canonical_player_ids_before",
    "active_opponent_count_before",
    "sequence_context_quality",
    "street_action_count_before",
    "street_aggressive_action_count_before",
    "street_call_count_before",
    "street_check_count_before",
    "street_fold_count_before",
    "street_prior_actor_action_count",
    "street_prior_actor_bet_raise_count",
    "street_last_aggressor_is_actor",
    "facing_bet_or_raise",
    "check_raise_opportunity",
    "facing_cbet",
    "facing_donk_lead",
    "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity",
    "turn_barrel_opportunity",
    "river_barrel_opportunity",
    "preflop_limper_count_before",
    "preflop_caller_count_before",
    "preflop_raise_count_before",
    "preflop_prior_entry_count_before",
    "preflop_facing_open_raise",
    "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity",
    "preflop_limp_opportunity",
    "preflop_actor_has_limped_before",
    "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before",
    "seat",
    "effective_stack_before_bb",
    "spr_before",
    "active_player_count_before",
    "players_with_chips_before",
    "action_label",
    *LEGAL_COLUMNS.values(),
    "amount_to_call_bb",
    "pot_before_bb",
    "actor_committed_before_bb",
    "actor_committed_after_bb",
    "actor_stack_before_bb",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
    "action_to_bb",
    "short_all_in_call",
    "feature_cutoff_utc",
    "profile_snapshot_id",
    "profile_snapshot_cutoff_utc",
    "profile_source_max_event_utc",
    "actor_profile_status",
    "future_reveal_used",
    "post_decision_outcome_used",
    "private_card_features_used",
    "private_card_visibility_utc",
    "private_card_source",
)
FEATURE_ROLE_MANIFEST_FIELDS = {
    "schema_version",
    "decision_columns",
    "model_feature_columns",
    "target_columns",
    "identity_columns",
    "profile_join_columns",
    "prohibited_model_feature_columns",
    "self_sha256",
}
TARGET_COLUMNS = ("action_label", "action_to_bb")
ALLOWED_BOARD_PAIR_LEVELS = {
    "unpaired",
    "one_pair",
    "two_pair",
    "trips",
    "quads",
    "full_house_possible",
}
IDENTITY_COLUMNS = (
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "decision_order",
    "actor_id",
)
PROFILE_JOIN_COLUMNS = (
    "actor_id",
    "profile_snapshot_id",
    "profile_snapshot_cutoff_utc",
    "profile_source_max_event_utc",
    "actor_profile_status",
)
ALLOWED_MODEL_FEATURE_COLUMNS = (
    "street",
    "board_texture_present",
    "board_card_count",
    "board_high_card_value",
    "board_pair_level",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "board_connectedness_score",
    "active_opponent_count_before",
    "sequence_context_quality",
    "street_action_count_before",
    "street_aggressive_action_count_before",
    "street_call_count_before",
    "street_check_count_before",
    "street_fold_count_before",
    "street_prior_actor_action_count",
    "street_prior_actor_bet_raise_count",
    "street_last_aggressor_is_actor",
    "facing_bet_or_raise",
    "check_raise_opportunity",
    "facing_cbet",
    "facing_donk_lead",
    "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity",
    "turn_barrel_opportunity",
    "river_barrel_opportunity",
    "preflop_limper_count_before",
    "preflop_caller_count_before",
    "preflop_raise_count_before",
    "preflop_prior_entry_count_before",
    "preflop_facing_open_raise",
    "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity",
    "preflop_limp_opportunity",
    "preflop_actor_has_limped_before",
    "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before",
    "seat",
    "effective_stack_before_bb",
    "spr_before",
    "active_player_count_before",
    "players_with_chips_before",
    *LEGAL_COLUMNS.values(),
    "amount_to_call_bb",
    "pot_before_bb",
    "actor_committed_before_bb",
    "actor_stack_before_bb",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
    "short_all_in_call",
)
PROHIBITED_MODEL_FEATURE_COLUMNS = (
    "action_label",
    "action_to_bb",
    *IDENTITY_COLUMNS,
    "decision_utc",
    "feature_cutoff_utc",
    "profile_snapshot_id",
    "profile_snapshot_cutoff_utc",
    "profile_source_max_event_utc",
    "actor_profile_status",
    "active_opponent_canonical_player_ids_before",
    "actor_committed_after_bb",
    "future_reveal_used",
    "post_decision_outcome_used",
    "private_card_features_used",
    "private_card_visibility_utc",
    "private_card_source",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _timestamp(value: Any, *, field: str, row_id: str) -> datetime:
    return parse_utc_timestamp(value, label=f"{row_id}.{field}")


def _optional_nonnegative(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return finite_nonnegative(value, label=label)


def _canonical_identifier(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


def _require_bound(
    value: float | None, *, row_id: str, field: str, reason: str
) -> float:
    if value is None:
        raise Stage6ContractError(f"{row_id}: {field} is required when {reason}")
    return value


def _canonical_column_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        type(column) is not str
        or not column
        or column != column.strip()
        or column != unicodedata.normalize("NFC", column)
        for column in value
    ):
        raise Stage6ContractError(f"{label} must be a canonical string list")
    if len(set(value)) != len(value):
        raise Stage6ContractError(f"{label} contains duplicate columns")
    return list(value)


def validate_feature_role_manifest(
    manifest: Mapping[str, Any], *, expected_manifest_sha256: str
) -> str:
    if not isinstance(manifest, Mapping) or set(manifest) != FEATURE_ROLE_MANIFEST_FIELDS:
        raise Stage6ContractError("decision feature-role manifest schema is invalid")
    if manifest.get("schema_version") != "stage6_decision_feature_role_manifest_v1":
        raise Stage6ContractError("decision feature-role manifest version is invalid")
    if (
        type(expected_manifest_sha256) is not str
        or SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None
    ):
        raise Stage6ContractError("expected feature-role manifest SHA-256 is invalid")
    observed_manifest_sha256 = validate_document_self_hash(manifest)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise Stage6ContractError(
            "feature-role manifest does not match the independently frozen input packet"
        )
    decision_columns = _canonical_column_list(
        manifest.get("decision_columns"), label="decision_columns"
    )
    model_columns = _canonical_column_list(
        manifest.get("model_feature_columns"), label="model_feature_columns"
    )
    target_columns = _canonical_column_list(
        manifest.get("target_columns"), label="target_columns"
    )
    identity_columns = _canonical_column_list(
        manifest.get("identity_columns"), label="identity_columns"
    )
    join_columns = _canonical_column_list(
        manifest.get("profile_join_columns"), label="profile_join_columns"
    )
    prohibited_columns = _canonical_column_list(
        manifest.get("prohibited_model_feature_columns"),
        label="prohibited_model_feature_columns",
    )
    if decision_columns != list(REQUIRED_COLUMNS):
        raise Stage6ContractError("feature-role decision columns do not match Stage 6")
    if target_columns != list(TARGET_COLUMNS):
        raise Stage6ContractError("feature-role target columns are invalid")
    if identity_columns != list(IDENTITY_COLUMNS):
        raise Stage6ContractError("feature-role identity columns are invalid")
    if join_columns != list(PROFILE_JOIN_COLUMNS):
        raise Stage6ContractError("feature-role profile join columns are invalid")
    if prohibited_columns != list(PROHIBITED_MODEL_FEATURE_COLUMNS):
        raise Stage6ContractError("feature-role prohibited columns are invalid")
    if (
        not model_columns
        or not set(model_columns).issubset(ALLOWED_MODEL_FEATURE_COLUMNS)
    ):
        raise Stage6ContractError(
            "model features are outside the code-pinned causal decision allowlist"
        )
    return observed_manifest_sha256


def audit_decision_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_role_manifest: Mapping[str, Any],
    expected_feature_role_manifest_sha256: str,
) -> dict[str, Any]:
    feature_role_manifest_sha256 = validate_feature_role_manifest(
        feature_role_manifest,
        expected_manifest_sha256=expected_feature_role_manifest_sha256,
    )
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise Stage6ContractError("refreshed decision dataset has no rows")

    row_ids: set[str] = set()
    decision_keys: set[tuple[str, str, str]] = set()
    decision_order_keys: set[tuple[str, str, int]] = set()
    casefold_bindings: dict[tuple[str, str], str] = {}
    street_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    decision_times_by_hand: dict[tuple[str, str], list[tuple[int, datetime, str]]] = {}

    for row in materialized:
        raw_row_id = row.get("row_id")
        row_id = raw_row_id if type(raw_row_id) is str else ""
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        unknown = sorted(set(row) - set(REQUIRED_COLUMNS))
        if missing or unknown:
            raise Stage6ContractError(
                f"{row_id or '<unknown>'}: exact decision schema mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        row_id = _canonical_identifier(row_id, label="row_id")
        if row_id in row_ids:
            raise Stage6ContractError(f"duplicate or empty row_id: {row_id!r}")
        row_ids.add(row_id)

        identifiers = {
            name: _canonical_identifier(row[name], label=f"{row_id}.{name}")
            for name in ("game_id", "hand_id", "decision_id", "actor_id")
        }
        for field, value in {"row_id": row_id, **identifiers}.items():
            alias_key = (field, value.casefold())
            previous = casefold_bindings.setdefault(alias_key, value)
            if previous != value:
                raise Stage6ContractError(
                    f"{row_id}: case-only {field} aliases are forbidden: "
                    f"{previous!r} versus {value!r}"
                )
        decision_key = (
            identifiers["game_id"],
            identifiers["hand_id"],
            identifiers["decision_id"],
        )
        if decision_key in decision_keys:
            raise Stage6ContractError(f"duplicate decision identity: {decision_key}")
        decision_keys.add(decision_key)

        decision_order = row["decision_order"]
        if (
            isinstance(decision_order, bool)
            or not isinstance(decision_order, int)
            or decision_order < 0
        ):
            raise Stage6ContractError(
                f"{row_id}: decision_order must be a nonnegative integer"
            )
        decision_order_key = (
            identifiers["game_id"],
            identifiers["hand_id"],
            decision_order,
        )
        if decision_order_key in decision_order_keys:
            raise Stage6ContractError(
                f"duplicate decision order within hand: {decision_order_key}"
            )
        decision_order_keys.add(decision_order_key)

        decision_time = _timestamp(row["decision_utc"], field="decision_utc", row_id=row_id)
        decision_times_by_hand.setdefault(
            (identifiers["game_id"], identifiers["hand_id"]), []
        ).append((decision_order, decision_time, row_id))
        feature_time = _timestamp(
            row["feature_cutoff_utc"], field="feature_cutoff_utc", row_id=row_id
        )
        profile_time = _timestamp(
            row["profile_snapshot_cutoff_utc"],
            field="profile_snapshot_cutoff_utc",
            row_id=row_id,
        )
        if feature_time > decision_time:
            raise Stage6ContractError(f"{row_id}: feature cutoff is after decision")
        if profile_time > decision_time:
            raise Stage6ContractError(f"{row_id}: profile cutoff is after decision")

        profile_status = _canonical_identifier(
            row["actor_profile_status"], label=f"{row_id}.actor_profile_status"
        )
        if profile_status not in PROFILE_STATUSES:
            raise Stage6ContractError(
                f"{row_id}: invalid actor_profile_status {profile_status!r}"
            )
        profile_snapshot_id = _canonical_identifier(
            row["profile_snapshot_id"], label=f"{row_id}.profile_snapshot_id"
        )
        snapshot_alias_key = ("profile_snapshot_id", profile_snapshot_id.casefold())
        previous_snapshot_id = casefold_bindings.setdefault(
            snapshot_alias_key, profile_snapshot_id
        )
        if previous_snapshot_id != profile_snapshot_id:
            raise Stage6ContractError(
                f"{row_id}: case-only profile snapshot aliases are forbidden: "
                f"{previous_snapshot_id!r} versus {profile_snapshot_id!r}"
            )
        profile_source = row["profile_source_max_event_utc"]
        if profile_status == "resolved":
            profile_source_time = _timestamp(
                profile_source,
                field="profile_source_max_event_utc",
                row_id=row_id,
            )
            if profile_source_time >= decision_time:
                raise Stage6ContractError(
                    f"{row_id}: profile source is not strictly prior to decision"
                )
            if profile_source_time > profile_time:
                raise Stage6ContractError(
                    f"{row_id}: profile source exceeds snapshot cutoff"
                )
        elif profile_source is not None:
            raise Stage6ContractError(
                f"{row_id}: zero-history profile contains a source event timestamp"
            )

        for flag in (
            "future_reveal_used",
            "post_decision_outcome_used",
            "private_card_features_used",
            "short_all_in_call",
            *LEGAL_COLUMNS.values(),
            "board_texture_present",
            "board_is_paired",
            "board_is_two_tone",
            "board_is_monotone",
            "street_last_aggressor_is_actor",
            "facing_bet_or_raise",
            "check_raise_opportunity",
            "facing_cbet",
            "facing_donk_lead",
            "missed_cbet_probe_opportunity",
            "delayed_cbet_opportunity",
            "turn_barrel_opportunity",
            "river_barrel_opportunity",
            "preflop_facing_open_raise",
            "preflop_facing_three_bet_or_more",
            "preflop_squeeze_opportunity",
            "preflop_limp_opportunity",
            "preflop_actor_has_limped_before",
            "preflop_actor_has_called_before",
            "preflop_actor_has_raised_before",
        ):
            if type(row[flag]) is not bool:
                raise Stage6ContractError(f"{row_id}: {flag} must be a strict boolean")
        if row["future_reveal_used"]:
            raise Stage6ContractError(f"{row_id}: future reveal was used")
        if row["post_decision_outcome_used"]:
            raise Stage6ContractError(f"{row_id}: post-decision outcome was used")
        if row["private_card_features_used"]:
            raise Stage6ContractError(
                f"{row_id}: private-card features are forbidden for behavior heads"
            )
        if row["private_card_visibility_utc"] is not None:
            raise Stage6ContractError(
                f"{row_id}: private-card visibility timestamp exists while features are forbidden"
            )
        if row["private_card_source"] != "none":
            raise Stage6ContractError(
                f"{row_id}: private-card source exists while features are forbidden"
            )

        integer_fields = (
            "active_opponent_count_before",
            "street_action_count_before",
            "street_aggressive_action_count_before",
            "street_call_count_before",
            "street_check_count_before",
            "street_fold_count_before",
            "street_prior_actor_action_count",
            "street_prior_actor_bet_raise_count",
            "preflop_limper_count_before",
            "preflop_caller_count_before",
            "preflop_raise_count_before",
            "preflop_prior_entry_count_before",
            "seat",
            "active_player_count_before",
            "players_with_chips_before",
        )
        for field in integer_fields:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Stage6ContractError(f"{row_id}: {field} must be a nonnegative integer")
        for field in ("board_card_count", "board_high_card_value"):
            value = row[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise Stage6ContractError(f"{row_id}: {field} must be null or a nonnegative integer")
        board_pair_level = row["board_pair_level"]
        if board_pair_level is not None:
            board_pair_level = _canonical_identifier(
                board_pair_level, label=f"{row_id}.board_pair_level"
            )
            if board_pair_level not in ALLOWED_BOARD_PAIR_LEVELS:
                raise Stage6ContractError(f"{row_id}: board_pair_level is invalid")
        for field in ("board_connectedness_score", "effective_stack_before_bb", "spr_before"):
            value = row[field]
            if value is not None:
                finite_nonnegative(value, label=f"{row_id}.{field}")
        _canonical_identifier(
            row["sequence_context_quality"],
            label=f"{row_id}.sequence_context_quality",
        )
        try:
            opponent_ids = json.loads(row["active_opponent_canonical_player_ids_before"])
        except (TypeError, json.JSONDecodeError) as error:
            raise Stage6ContractError(
                f"{row_id}: active opponent identity list is invalid JSON"
            ) from error
        if not isinstance(opponent_ids, list):
            raise Stage6ContractError(f"{row_id}: active opponent identity list must be an array")
        canonical_opponents = [
            _canonical_identifier(value, label=f"{row_id}.active_opponent[{index}]")
            for index, value in enumerate(opponent_ids)
        ]
        if len(canonical_opponents) != len(set(canonical_opponents)):
            raise Stage6ContractError(f"{row_id}: active opponent identities are not unique")
        if canonical_opponents != sorted(canonical_opponents):
            raise Stage6ContractError(f"{row_id}: active opponent identities are not sorted")
        if len(canonical_opponents) != row["active_opponent_count_before"]:
            raise Stage6ContractError(f"{row_id}: active opponent count is inconsistent")
        if row["actor_id"] in canonical_opponents:
            raise Stage6ContractError(f"{row_id}: actor appears in active opponent identities")
        if (
            row["active_player_count_before"] < 1
            or len(canonical_opponents) > row["active_player_count_before"] - 1
        ):
            raise Stage6ContractError(
                f"{row_id}: active opponent identities exceed the physical roster"
            )
        if row["players_with_chips_before"] > row["active_player_count_before"]:
            raise Stage6ContractError(f"{row_id}: visible players-with-chips count is inconsistent")
        if row["board_texture_present"]:
            if row["board_card_count"] not in {3, 4, 5}:
                raise Stage6ContractError(f"{row_id}: visible board card count is invalid")
            if board_pair_level is None:
                raise Stage6ContractError(f"{row_id}: visible board pair level is missing")
        elif row["board_card_count"] is not None:
            raise Stage6ContractError(f"{row_id}: absent board has a card count")
        elif board_pair_level is not None:
            raise Stage6ContractError(f"{row_id}: absent board has a pair level")

        legal = {action: row[column] for action, column in LEGAL_COLUMNS.items()}
        if not any(legal.values()):
            raise Stage6ContractError(f"{row_id}: no legal action is available")
        action = _canonical_identifier(
            row["action_label"], label=f"{row_id}.action_label"
        )
        if action not in LEGAL_COLUMNS:
            raise Stage6ContractError(f"{row_id}: invalid action label {action!r}")
        if not legal[action]:
            raise Stage6ContractError(f"{row_id}: action label {action} is illegal")

        amount_to_call = finite_nonnegative(
            row["amount_to_call_bb"], label=f"{row_id}.amount_to_call_bb"
        )
        pot_before = finite_nonnegative(
            row["pot_before_bb"], label=f"{row_id}.pot_before_bb"
        )
        if pot_before <= 0:
            raise Stage6ContractError(f"{row_id}: pot must be positive before decision")
        actor_committed = finite_nonnegative(
            row["actor_committed_before_bb"],
            label=f"{row_id}.actor_committed_before_bb",
        )
        actor_committed_after = finite_nonnegative(
            row["actor_committed_after_bb"],
            label=f"{row_id}.actor_committed_after_bb",
        )
        actor_stack = finite_nonnegative(
            row["actor_stack_before_bb"], label=f"{row_id}.actor_stack_before_bb"
        )
        if actor_stack <= 0:
            raise Stage6ContractError(f"{row_id}: actor has no stack before decision")
        exact_stack_cap_to = actor_committed + actor_stack
        if not math.isfinite(exact_stack_cap_to):
            raise Stage6ContractError(f"{row_id}: actor stack cap is not finite")
        if actor_committed_after > exact_stack_cap_to + 1e-9:
            raise Stage6ContractError(
                f"{row_id}: observed contribution exceeds the exact stack cap"
            )
        min_bet = _optional_nonnegative(
            row["legal_min_bet_to_bb"], label=f"{row_id}.legal_min_bet_to_bb"
        )
        min_raise = _optional_nonnegative(
            row["legal_min_raise_to_bb"], label=f"{row_id}.legal_min_raise_to_bb"
        )
        max_size = _optional_nonnegative(
            row["legal_max_bet_raise_to_bb"],
            label=f"{row_id}.legal_max_bet_raise_to_bb",
        )
        action_size = _optional_nonnegative(
            row["action_to_bb"], label=f"{row_id}.action_to_bb"
        )

        if amount_to_call == 0:
            if (
                legal["check"] is not True
                or legal["fold"]
                or legal["call"]
                or legal["raise"]
            ):
                raise Stage6ContractError(
                    f"{row_id}: zero-call state must include check and exclude fold/call/raise"
                )
        else:
            if legal["fold"] is not True or legal["check"] or legal["bet"]:
                raise Stage6ContractError(
                    f"{row_id}: facing-bet state must include fold and exclude check/bet"
                )
        short_all_in_call = row["short_all_in_call"]
        expected_short_all_in_call = amount_to_call > actor_stack + 1e-9
        if short_all_in_call is not expected_short_all_in_call:
            raise Stage6ContractError(
                f"{row_id}: short_all_in_call disagrees with call amount and stack"
            )
        if expected_short_all_in_call:
            if legal["call"] or legal["raise"] or legal["all_in"] is not True:
                raise Stage6ContractError(
                    f"{row_id}: short all-in call legality is inconsistent"
                )
        elif amount_to_call > 0 and legal["call"] is not True:
            raise Stage6ContractError(
                f"{row_id}: a full call is affordable but not legal"
            )
        if legal["bet"]:
            required_min_bet = _require_bound(
                min_bet, row_id=row_id, field="legal_min_bet_to_bb", reason="bet is legal"
            )
            required_max = _require_bound(
                max_size,
                row_id=row_id,
                field="legal_max_bet_raise_to_bb",
                reason="bet is legal",
            )
            if required_min_bet > required_max:
                raise Stage6ContractError(f"{row_id}: min bet exceeds max size")
            if required_min_bet <= actor_committed:
                raise Stage6ContractError(f"{row_id}: legal bet has no positive chip increment")
        if legal["raise"]:
            required_min_raise = _require_bound(
                min_raise,
                row_id=row_id,
                field="legal_min_raise_to_bb",
                reason="raise is legal",
            )
            required_max = _require_bound(
                max_size,
                row_id=row_id,
                field="legal_max_bet_raise_to_bb",
                reason="raise is legal",
            )
            if required_min_raise > required_max:
                raise Stage6ContractError(f"{row_id}: min raise exceeds max size")
            if required_min_raise <= actor_committed:
                raise Stage6ContractError(
                    f"{row_id}: legal raise has no positive chip increment"
                )
        if legal["all_in"]:
            _require_bound(
                max_size,
                row_id=row_id,
                field="legal_max_bet_raise_to_bb",
                reason="all-in is legal",
            )
        if not legal["bet"] and min_bet is not None:
            raise Stage6ContractError(
                f"{row_id}: min bet is populated while bet is illegal"
            )
        if not legal["raise"] and min_raise is not None:
            raise Stage6ContractError(
                f"{row_id}: min raise is populated while raise is illegal"
            )
        if not any(legal[action] for action in ("bet", "raise", "all_in")) and max_size is not None:
            raise Stage6ContractError(
                f"{row_id}: max sizing bound is populated without a legal sizing action"
            )
        if max_size is not None and not math.isclose(
            max_size, exact_stack_cap_to, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise Stage6ContractError(
                f"{row_id}: legal maximum is not committed chips plus remaining stack"
            )

        if action in {"bet", "raise", "all_in"}:
            target_size = _require_bound(
                action_size,
                row_id=row_id,
                field="action_to_bb",
                reason=f"observed action is {action}",
            )
            upper = _require_bound(
                max_size,
                row_id=row_id,
                field="legal_max_bet_raise_to_bb",
                reason=f"observed action is {action}",
            )
            if target_size > upper:
                raise Stage6ContractError(f"{row_id}: observed size exceeds legal max")
            if target_size <= actor_committed:
                raise Stage6ContractError(
                    f"{row_id}: observed sizing action adds no chips"
                )
            if not math.isclose(
                target_size, actor_committed_after, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise Stage6ContractError(
                    f"{row_id}: action_to_bb differs from the observed contribution"
                )
            if action == "all_in" and not math.isclose(
                target_size, upper, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise Stage6ContractError(
                    f"{row_id}: all-in size does not equal the visible legal max"
                )
            if action == "bet" and min_bet is not None and target_size < min_bet:
                raise Stage6ContractError(f"{row_id}: observed bet is below legal min")
            if action == "raise" and min_raise is not None and target_size < min_raise:
                raise Stage6ContractError(f"{row_id}: observed raise is below legal min")
        else:
            if action_size is not None:
                raise Stage6ContractError(
                    f"{row_id}: action_to_bb is populated for nonsizing action {action}"
                )
            if action in {"fold", "check"} and not math.isclose(
                actor_committed_after,
                actor_committed,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise Stage6ContractError(
                    f"{row_id}: observed {action} changed the actor contribution"
                )
            if action == "call":
                expected_call_after = actor_committed + min(amount_to_call, actor_stack)
                if not math.isclose(
                    actor_committed_after,
                    expected_call_after,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise Stage6ContractError(
                        f"{row_id}: observed call contribution is not stack-capped"
                    )

        street = _canonical_identifier(row["street"], label=f"{row_id}.street")
        if street not in {"preflop", "flop", "turn", "river"}:
            raise Stage6ContractError(f"{row_id}: invalid street {street!r}")
        street_counts[street] += 1
        action_counts[action] += 1
        profile_counts[profile_status] += 1

    for hand_key, decisions in decision_times_by_hand.items():
        ordered = sorted(decisions, key=lambda item: item[0])
        for left, right in zip(ordered, ordered[1:]):
            if left[1] >= right[1]:
                raise Stage6ContractError(
                    f"{hand_key}: decision timestamps do not increase with decision_order; "
                    f"{left[2]} then {right[2]}"
                )

    ordered_rows = sorted(materialized, key=lambda row: row["row_id"])
    identity_rows = [
        {
            field: row[field]
            for field in (
                "row_id",
                "game_id",
                "hand_id",
                "decision_id",
                "decision_order",
            )
        }
        for row in ordered_rows
    ]
    return {
        "schema_version": "stage6_data_oracle_report_v1",
        "status": "PASS",
        "feature_role_manifest_sha256": feature_role_manifest_sha256,
        "row_count": len(materialized),
        "unique_row_count": len(row_ids),
        "unique_decision_count": len(decision_keys),
        "street_counts": dict(sorted(street_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "profile_status_counts": dict(sorted(profile_counts.items())),
        "decision_row_root_digest": hashlib.sha256(
            canonical_json_bytes(ordered_rows)
        ).hexdigest(),
        "row_identity_root_digest": hashlib.sha256(
            canonical_json_bytes(identity_rows)
        ).hexdigest(),
        "duplicate_row_id_count": 0,
        "duplicate_decision_identity_count": 0,
        "invalid_action_label_count": 0,
        "unknown_legality_mask_count": 0,
        "future_timestamp_violation_count": 0,
        "future_reveal_violation_count": 0,
        "post_decision_outcome_violation_count": 0,
        "private_card_feature_violation_count": 0,
        "legal_size_bound_violation_count": 0,
    }
