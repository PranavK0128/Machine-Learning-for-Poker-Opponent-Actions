from __future__ import annotations

import math
import json
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Mapping

from stage6_common import (
    Stage6ContractError,
    document_self_sha256,
    parse_utc_timestamp,
)
from stage6_data_oracle import (
    ALLOWED_BOARD_PAIR_LEVELS,
    ALLOWED_MODEL_FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    PROFILE_JOIN_COLUMNS,
    PROHIBITED_MODEL_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMNS,
)


EXPECTED_ACTION_DATASET_VERSION = "action_state_v1_stage004_action_history"
EXPECTED_LEGAL_MASK_QUALITY = "action_state_export_exact_v1"
EXPECTED_IDENTITY_GROUPING_STATUS = "resolved_retrospective_group"
ALLOWED_STREETS = {"preflop", "flop", "turn", "river"}
SOURCE_ACTION_LABELS = {"fold", "check", "call", "bet_or_raise"}
ROW_ID_WIDTH = 10
QUARANTINE_DOMINATED_ZERO_COST_FOLD = "dominated_zero_cost_fold"
QUARANTINE_OBSERVED_CHECK_FACING_POSITIVE_CALL = (
    "observed_check_facing_positive_call"
)
QUARANTINE_OBSERVED_BET_BELOW_MINIMUM = "observed_bet_below_exact_minimum"
QUARANTINE_OBSERVED_RAISE_BELOW_MINIMUM = "observed_raise_below_exact_minimum"
QUARANTINE_OBSERVED_CALL_TRANSITION_MISMATCH = (
    "observed_call_contribution_transition_mismatch"
)
ALLOWED_SOURCE_QUARANTINE_REASONS = {
    QUARANTINE_DOMINATED_ZERO_COST_FOLD,
    QUARANTINE_OBSERVED_CHECK_FACING_POSITIVE_CALL,
    QUARANTINE_OBSERVED_BET_BELOW_MINIMUM,
    QUARANTINE_OBSERVED_RAISE_BELOW_MINIMUM,
    QUARANTINE_OBSERVED_CALL_TRANSITION_MISMATCH,
}


def _canonical_text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


def _optional_text(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    return _canonical_text(value, label=label)


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise Stage6ContractError(f"{label} must be a nonnegative integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif type(value) is str and value.isdigit():
        parsed = int(value)
    else:
        raise Stage6ContractError(f"{label} must be a nonnegative integer")
    if parsed < 0:
        raise Stage6ContractError(f"{label} must be a nonnegative integer")
    return parsed


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is str:
        normalized = value.casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise Stage6ContractError(f"{label} must be a strict boolean")


def _nonnegative_decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or value in (None, ""):
        raise Stage6ContractError(f"{label} must be a finite nonnegative number")
    if not isinstance(value, (str, Integral, Real, Decimal)):
        raise Stage6ContractError(f"{label} must be a finite nonnegative number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise Stage6ContractError(
            f"{label} must be a finite nonnegative number"
        ) from error
    if not parsed.is_finite() or parsed < 0:
        raise Stage6ContractError(f"{label} must be a finite nonnegative number")
    return parsed


def _optional_nonnegative_decimal(value: Any, *, label: str) -> Decimal | None:
    if value in (None, ""):
        return None
    return _nonnegative_decimal(value, label=label)


def _optional_nonnegative_integer(value: Any, *, label: str) -> int | None:
    if value in (None, ""):
        return None
    return _nonnegative_integer(value, label=label)


def _canonical_identifier_array_json(value: Any, *, label: str) -> tuple[str, list[str]]:
    if type(value) is not str:
        raise Stage6ContractError(f"{label} must be a JSON array string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise Stage6ContractError(f"{label} must be valid JSON") from error
    if not isinstance(parsed, list):
        raise Stage6ContractError(f"{label} must be a JSON array")
    identifiers = [_canonical_text(item, label=f"{label}[{index}]") for index, item in enumerate(parsed)]
    if len(identifiers) != len(set(identifiers)) or identifiers != sorted(identifiers):
        raise Stage6ContractError(f"{label} must contain unique sorted identifiers")
    return json.dumps(identifiers, separators=(",", ":"), ensure_ascii=True), identifiers


def _bb(raw: Decimal, big_blind: Decimal) -> float:
    return float(raw / big_blind)


def _require_equal_text(
    left: Mapping[str, Any],
    left_field: str,
    right: Mapping[str, Any],
    right_field: str,
) -> str:
    left_value = _canonical_text(left.get(left_field), label=f"action.{left_field}")
    right_value = _canonical_text(right.get(right_field), label=f"identity.{right_field}")
    if left_value != right_value:
        raise Stage6ContractError(
            f"action/identity mismatch: {left_field}={left_value!r}, "
            f"{right_field}={right_value!r}"
        )
    return left_value


def _validate_source_action_flags(action: Mapping[str, Any], source_label: str) -> bool:
    expected = {
        "is_fold": source_label == "fold",
        "is_check": source_label == "check",
        "is_call": source_label == "call",
        "is_bet_or_raise": source_label == "bet_or_raise",
    }
    for field, expected_value in expected.items():
        observed = _strict_bool(action.get(field), label=f"action.{field}")
        if observed is not expected_value:
            raise Stage6ContractError(
                f"action.{field} disagrees with source action label {source_label!r}"
            )
    return _strict_bool(action.get("is_all_in"), label="action.is_all_in")


def _normalize_action_label(
    *, source_label: str, is_all_in: bool, amount_to_call_raw: Decimal
) -> str:
    if is_all_in:
        return "all_in"
    if source_label == "bet_or_raise":
        return "bet" if amount_to_call_raw == 0 else "raise"
    return source_label


def classify_source_quarantine_reason(action_row: Mapping[str, Any]) -> str | None:
    """Return the preregistered strict-contract exclusion, if any.

    The refreshed source is conserved exactly: callers must persist every row
    classified here in the quarantine ledger rather than silently dropping it.
    """

    source_label = _canonical_text(
        action_row.get("action_label"), label="action.action_label"
    )
    amount_to_call_raw = _nonnegative_decimal(
        action_row.get("amount_to_call_raw"), label="action.amount_to_call_raw"
    )
    committed_after_raw = _nonnegative_decimal(
        action_row.get("street_contribution_after_raw"),
        label="action.street_contribution_after_raw",
    )
    committed_before_raw = _nonnegative_decimal(
        action_row.get("street_contribution_before_raw"),
        label="action.street_contribution_before_raw",
    )
    stack_before_raw = _nonnegative_decimal(
        action_row.get("stack_before_raw"), label="action.stack_before_raw"
    )
    is_all_in = _strict_bool(action_row.get("is_all_in"), label="action.is_all_in")

    if (
        source_label == "fold"
        and amount_to_call_raw == 0
        and _strict_bool(action_row.get("can_check"), label="action.can_check")
        and not _strict_bool(action_row.get("can_fold"), label="action.can_fold")
    ):
        return QUARANTINE_DOMINATED_ZERO_COST_FOLD

    if source_label == "check" and amount_to_call_raw > 0:
        return QUARANTINE_OBSERVED_CHECK_FACING_POSITIVE_CALL

    if source_label == "bet_or_raise" and not is_all_in:
        if amount_to_call_raw == 0:
            minimum = _optional_nonnegative_decimal(
                action_row.get("legal_min_bet_to_raw"),
                label="action.legal_min_bet_to_raw",
            )
            if minimum is not None and committed_after_raw < minimum:
                return QUARANTINE_OBSERVED_BET_BELOW_MINIMUM
        else:
            minimum = _optional_nonnegative_decimal(
                action_row.get("legal_min_raise_to_raw"),
                label="action.legal_min_raise_to_raw",
            )
            if minimum is not None and committed_after_raw < minimum:
                return QUARANTINE_OBSERVED_RAISE_BELOW_MINIMUM
    if source_label == "call" and committed_after_raw != (
        committed_before_raw + min(amount_to_call_raw, stack_before_raw)
    ):
        return QUARANTINE_OBSERVED_CALL_TRANSITION_MISMATCH
    return None


def build_decision_row(
    action_row: Mapping[str, Any],
    identity_row: Mapping[str, Any],
    profile_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one refreshed action state into the exact Stage 6 decision schema.

    The action and identity exports are reconciled one-for-one. Profile fields
    must come from the strict pre-decision causal snapshot kernel.
    """

    if _canonical_text(
        action_row.get("dataset_version"), label="action.dataset_version"
    ) != EXPECTED_ACTION_DATASET_VERSION:
        raise Stage6ContractError("unexpected action-state dataset version")
    if _canonical_text(
        action_row.get("legal_mask_quality"), label="action.legal_mask_quality"
    ) != EXPECTED_LEGAL_MASK_QUALITY:
        raise Stage6ContractError("action-state legal mask is not exact")
    if _canonical_text(
        identity_row.get("identity_grouping_status"),
        label="identity.identity_grouping_status",
    ) != EXPECTED_IDENTITY_GROUPING_STATUS:
        raise Stage6ContractError("decision identity is not retrospectively resolved")

    decision_id = _require_equal_text(
        action_row, "hand_action_id", identity_row, "hand_action_id"
    )
    hand_id = _require_equal_text(action_row, "hand_id", identity_row, "hand_id")
    game_id = _require_equal_text(action_row, "game_id", identity_row, "game_id")
    actor_pokernow_id = _require_equal_text(
        action_row,
        "pokernow_player_id",
        identity_row,
        "actor_pokernow_player_id",
    )
    del actor_pokernow_id

    action_event_index = _nonnegative_integer(
        action_row.get("event_index"), label="action.event_index"
    )
    identity_event_index = _nonnegative_integer(
        identity_row.get("event_index"), label="identity.event_index"
    )
    if action_event_index != identity_event_index:
        raise Stage6ContractError("action/identity event_index mismatch")
    source_ordinal = _nonnegative_integer(
        identity_row.get("source_row_ordinal"),
        label="identity.source_row_ordinal",
    )
    binding_ordinal = _nonnegative_integer(
        profile_binding.get("source_row_ordinal"),
        label="profile_binding.source_row_ordinal",
    )
    if binding_ordinal != source_ordinal:
        raise Stage6ContractError("identity/profile source_row_ordinal mismatch")

    actor_id = _canonical_text(
        identity_row.get("canonical_group_player_id"),
        label="identity.canonical_group_player_id",
    )
    binding_actor_id = _canonical_text(
        profile_binding.get("canonical_player_id"),
        label="profile_binding.canonical_player_id",
    )
    if binding_actor_id != actor_id:
        raise Stage6ContractError("identity/profile canonical player mismatch")

    decision_utc = _canonical_text(
        identity_row.get("decision_utc"), label="identity.decision_utc"
    )
    parse_utc_timestamp(decision_utc, label="identity.decision_utc")
    binding_decision_utc = _canonical_text(
        profile_binding.get("decision_utc"),
        label="profile_binding.decision_utc",
    )
    if binding_decision_utc != decision_utc:
        raise Stage6ContractError("identity/profile decision timestamp mismatch")

    street = _canonical_text(action_row.get("street"), label="action.street")
    if street not in ALLOWED_STREETS:
        raise Stage6ContractError(f"unsupported action street: {street!r}")
    source_label = _canonical_text(
        action_row.get("action_label"), label="action.action_label"
    )
    normalized_type = _canonical_text(
        action_row.get("normalized_type"), label="action.normalized_type"
    )
    if source_label not in SOURCE_ACTION_LABELS or normalized_type not in SOURCE_ACTION_LABELS:
        raise Stage6ContractError("action label/type normalization is unsupported")

    big_blind_raw = _nonnegative_decimal(
        action_row.get("big_blind_raw"), label="action.big_blind_raw"
    )
    if big_blind_raw == 0:
        raise Stage6ContractError("action.big_blind_raw must be positive")
    amount_to_call_raw = _nonnegative_decimal(
        action_row.get("amount_to_call_raw"), label="action.amount_to_call_raw"
    )
    pot_before_raw = _nonnegative_decimal(
        action_row.get("pot_before_raw"), label="action.pot_before_raw"
    )
    committed_before_raw = _nonnegative_decimal(
        action_row.get("street_contribution_before_raw"),
        label="action.street_contribution_before_raw",
    )
    committed_after_raw = _nonnegative_decimal(
        action_row.get("street_contribution_after_raw"),
        label="action.street_contribution_after_raw",
    )
    stack_before_raw = _nonnegative_decimal(
        action_row.get("stack_before_raw"), label="action.stack_before_raw"
    )
    if pot_before_raw == 0 or stack_before_raw == 0:
        raise Stage6ContractError("decision state requires positive pot and actor stack")

    can_fold = _strict_bool(action_row.get("can_fold"), label="action.can_fold")
    can_check = _strict_bool(action_row.get("can_check"), label="action.can_check")
    source_can_call = _strict_bool(
        action_row.get("can_call"), label="action.can_call"
    )
    can_bet = _strict_bool(action_row.get("can_bet"), label="action.can_bet")
    can_raise = _strict_bool(action_row.get("can_raise"), label="action.can_raise")
    can_all_in = _strict_bool(
        action_row.get("can_all_in"), label="action.can_all_in"
    )
    short_all_in_call = amount_to_call_raw > stack_before_raw
    if short_all_in_call and not source_can_call:
        raise Stage6ContractError("short all-in state is missing the source call path")
    legal_call = source_can_call and not short_all_in_call

    is_all_in = _validate_source_action_flags(action_row, source_label)
    action_label = _normalize_action_label(
        source_label=source_label,
        is_all_in=is_all_in,
        amount_to_call_raw=amount_to_call_raw,
    )
    if source_label in {"fold", "check"}:
        if is_all_in:
            raise Stage6ContractError(f"observed {source_label} cannot be all-in")
        if committed_after_raw != committed_before_raw:
            raise Stage6ContractError(
                f"observed {source_label} changed the actor street contribution"
            )
    elif source_label == "call":
        expected_call_after_raw = committed_before_raw + min(
            amount_to_call_raw, stack_before_raw
        )
        if committed_after_raw != expected_call_after_raw:
            raise Stage6ContractError(
                "observed call contribution does not equal committed chips plus "
                "the stack-capped call amount"
            )
        expected_call_all_in = amount_to_call_raw >= stack_before_raw
        if is_all_in is not expected_call_all_in:
            raise Stage6ContractError(
                "observed call all-in flag disagrees with the exact stack-capped call"
            )

    min_bet_raw = _optional_nonnegative_decimal(
        action_row.get("legal_min_bet_to_raw"),
        label="action.legal_min_bet_to_raw",
    )
    min_raise_raw = _optional_nonnegative_decimal(
        action_row.get("legal_min_raise_to_raw"),
        label="action.legal_min_raise_to_raw",
    )
    max_size_raw = _optional_nonnegative_decimal(
        action_row.get("legal_max_bet_raise_to_raw"),
        label="action.legal_max_bet_raise_to_raw",
    )
    if can_bet and min_bet_raw is None:
        raise Stage6ContractError("legal bet is missing its exact minimum")
    if can_raise and min_raise_raw is None:
        raise Stage6ContractError("legal raise is missing its exact minimum")
    if (can_bet or can_raise or can_all_in) and max_size_raw is None:
        raise Stage6ContractError("legal sizing state is missing its exact maximum")
    if max_size_raw is not None and max_size_raw != committed_before_raw + stack_before_raw:
        raise Stage6ContractError(
            "legal maximum does not equal committed chips plus remaining stack"
        )

    legal_by_action = {
        "fold": can_fold,
        "check": can_check,
        "call": legal_call,
        "bet": can_bet,
        "raise": can_raise,
        "all_in": can_all_in,
    }
    if not legal_by_action[action_label]:
        raise Stage6ContractError(
            f"observed normalized action {action_label!r} is not legal"
        )
    action_to_bb: float | None = None
    if action_label in {"bet", "raise", "all_in"}:
        if committed_after_raw <= committed_before_raw:
            raise Stage6ContractError("sizing action did not add chips")
        action_to_bb = _bb(committed_after_raw, big_blind_raw)
        if action_label == "all_in" and committed_after_raw != max_size_raw:
            raise Stage6ContractError("observed all-in does not equal exact legal maximum")
        if (
            action_label == "bet"
            and min_bet_raw is not None
            and committed_after_raw < min_bet_raw
        ):
            raise Stage6ContractError("observed bet is below the exact legal minimum")
        if (
            action_label == "raise"
            and min_raise_raw is not None
            and committed_after_raw < min_raise_raw
        ):
            raise Stage6ContractError("observed raise is below the exact legal minimum")

    profile_snapshot_id = _canonical_text(
        profile_binding.get("profile_snapshot_id"),
        label="profile_binding.profile_snapshot_id",
    )
    profile_cutoff_utc = _canonical_text(
        profile_binding.get("profile_snapshot_cutoff_utc"),
        label="profile_binding.profile_snapshot_cutoff_utc",
    )
    profile_source_utc = _optional_text(
        profile_binding.get("profile_source_max_event_utc"),
        label="profile_binding.profile_source_max_event_utc",
    )
    profile_status = _canonical_text(
        profile_binding.get("actor_profile_status"),
        label="profile_binding.actor_profile_status",
    )

    active_opponents_json, active_opponents = _canonical_identifier_array_json(
        action_row.get("active_opponent_canonical_player_ids_before"),
        label="action.active_opponent_canonical_player_ids_before",
    )
    active_opponent_count = _nonnegative_integer(
        action_row.get("active_opponent_count_before"),
        label="action.active_opponent_count_before",
    )
    active_player_count = _nonnegative_integer(
        action_row.get("active_player_count_before"),
        label="action.active_player_count_before",
    )
    if active_opponent_count != len(active_opponents):
        raise Stage6ContractError("action active-player identity counts are inconsistent")
    if active_player_count < 1 or active_opponent_count > active_player_count - 1:
        raise Stage6ContractError("action active-player identity counts exceed the physical roster")
    if actor_id in active_opponents:
        raise Stage6ContractError("action actor is listed as an active opponent")

    def source_bool(field: str) -> bool:
        return _strict_bool(action_row.get(field), label=f"action.{field}")

    def source_int(field: str) -> int:
        return _nonnegative_integer(action_row.get(field), label=f"action.{field}")

    def source_optional_int(field: str) -> int | None:
        return _optional_nonnegative_integer(action_row.get(field), label=f"action.{field}")

    def source_optional_float(field: str) -> float | None:
        value = _optional_nonnegative_decimal(action_row.get(field), label=f"action.{field}")
        return float(value) if value is not None else None

    board_pair_level = _optional_text(
        action_row.get("board_pair_level"), label="action.board_pair_level"
    )
    if board_pair_level is not None and board_pair_level not in ALLOWED_BOARD_PAIR_LEVELS:
        raise Stage6ContractError("action.board_pair_level is outside the registered categorical schema")

    return {
        "row_id": f"stage6-decision-{source_ordinal:0{ROW_ID_WIDTH}d}",
        "game_id": game_id,
        "hand_id": hand_id,
        "decision_id": decision_id,
        "decision_utc": decision_utc,
        "decision_order": action_event_index,
        "actor_id": actor_id,
        "street": street,
        "board_texture_present": source_bool("board_texture_present"),
        "board_card_count": source_optional_int("board_card_count"),
        "board_high_card_value": source_optional_int("board_high_card_value"),
        "board_pair_level": board_pair_level,
        "board_is_paired": source_bool("board_is_paired"),
        "board_is_two_tone": source_bool("board_is_two_tone"),
        "board_is_monotone": source_bool("board_is_monotone"),
        "board_connectedness_score": source_optional_float("board_connectedness_score"),
        "active_opponent_canonical_player_ids_before": active_opponents_json,
        "active_opponent_count_before": active_opponent_count,
        "sequence_context_quality": _canonical_text(
            action_row.get("sequence_context_quality"), label="action.sequence_context_quality"
        ),
        "street_action_count_before": source_int("street_action_count_before"),
        "street_aggressive_action_count_before": source_int("street_aggressive_action_count_before"),
        "street_call_count_before": source_int("street_call_count_before"),
        "street_check_count_before": source_int("street_check_count_before"),
        "street_fold_count_before": source_int("street_fold_count_before"),
        "street_prior_actor_action_count": source_int("street_prior_actor_action_count"),
        "street_prior_actor_bet_raise_count": source_int("street_prior_actor_bet_raise_count"),
        "street_last_aggressor_is_actor": source_bool("street_last_aggressor_is_actor"),
        "facing_bet_or_raise": source_bool("facing_bet_or_raise"),
        "check_raise_opportunity": source_bool("check_raise_opportunity"),
        "facing_cbet": source_bool("facing_cbet"),
        "facing_donk_lead": source_bool("facing_donk_lead"),
        "missed_cbet_probe_opportunity": source_bool("missed_cbet_probe_opportunity"),
        "delayed_cbet_opportunity": source_bool("delayed_cbet_opportunity"),
        "turn_barrel_opportunity": source_bool("turn_barrel_opportunity"),
        "river_barrel_opportunity": source_bool("river_barrel_opportunity"),
        "preflop_limper_count_before": source_int("preflop_limper_count_before"),
        "preflop_caller_count_before": source_int("preflop_caller_count_before"),
        "preflop_raise_count_before": source_int("preflop_raise_count_before"),
        "preflop_prior_entry_count_before": source_int("preflop_prior_entry_count_before"),
        "preflop_facing_open_raise": source_bool("preflop_facing_open_raise"),
        "preflop_facing_three_bet_or_more": source_bool("preflop_facing_three_bet_or_more"),
        "preflop_squeeze_opportunity": source_bool("preflop_squeeze_opportunity"),
        "preflop_limp_opportunity": source_bool("preflop_limp_opportunity"),
        "preflop_actor_has_limped_before": source_bool("preflop_actor_has_limped_before"),
        "preflop_actor_has_called_before": source_bool("preflop_actor_has_called_before"),
        "preflop_actor_has_raised_before": source_bool("preflop_actor_has_raised_before"),
        "seat": source_int("seat"),
        "effective_stack_before_bb": source_optional_float("effective_stack_before_bb"),
        "spr_before": source_optional_float("spr_before"),
        "active_player_count_before": active_player_count,
        "players_with_chips_before": source_int("players_with_chips_before"),
        "action_label": action_label,
        "legal_fold": can_fold,
        "legal_check": can_check,
        "legal_call": legal_call,
        "legal_bet": can_bet,
        "legal_raise": can_raise,
        "legal_all_in": can_all_in,
        "amount_to_call_bb": _bb(amount_to_call_raw, big_blind_raw),
        "pot_before_bb": _bb(pot_before_raw, big_blind_raw),
        "actor_committed_before_bb": _bb(committed_before_raw, big_blind_raw),
        "actor_committed_after_bb": _bb(committed_after_raw, big_blind_raw),
        "actor_stack_before_bb": _bb(stack_before_raw, big_blind_raw),
        "legal_min_bet_to_bb": (
            _bb(min_bet_raw, big_blind_raw) if can_bet and min_bet_raw is not None else None
        ),
        "legal_min_raise_to_bb": (
            _bb(min_raise_raw, big_blind_raw)
            if can_raise and min_raise_raw is not None
            else None
        ),
        "legal_max_bet_raise_to_bb": (
            _bb(max_size_raw, big_blind_raw)
            if (can_bet or can_raise or can_all_in) and max_size_raw is not None
            else None
        ),
        "action_to_bb": action_to_bb,
        "short_all_in_call": short_all_in_call,
        "feature_cutoff_utc": decision_utc,
        "profile_snapshot_id": profile_snapshot_id,
        "profile_snapshot_cutoff_utc": profile_cutoff_utc,
        "profile_source_max_event_utc": profile_source_utc,
        "actor_profile_status": profile_status,
        "future_reveal_used": False,
        "post_decision_outcome_used": False,
        "private_card_features_used": False,
        "private_card_visibility_utc": None,
        "private_card_source": "none",
    }


def build_feature_role_manifest() -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "stage6_decision_feature_role_manifest_v1",
        "decision_columns": list(REQUIRED_COLUMNS),
        "model_feature_columns": list(ALLOWED_MODEL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "identity_columns": list(IDENTITY_COLUMNS),
        "profile_join_columns": list(PROFILE_JOIN_COLUMNS),
        "prohibited_model_feature_columns": list(PROHIBITED_MODEL_FEATURE_COLUMNS),
    }
    document["self_sha256"] = document_self_sha256(document)
    return document


def assert_finite_decision_row(row: Mapping[str, Any]) -> None:
    """Cheap materializer-side guard before the full exact oracle runs."""

    for field in (
        "board_connectedness_score",
        "amount_to_call_bb",
        "pot_before_bb",
        "actor_committed_before_bb",
        "actor_committed_after_bb",
        "actor_stack_before_bb",
        "legal_min_bet_to_bb",
        "legal_min_raise_to_bb",
        "legal_max_bet_raise_to_bb",
        "action_to_bb",
        "effective_stack_before_bb",
        "spr_before",
    ):
        value = row.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise Stage6ContractError(f"decision row {field} is not finite and nonnegative")
