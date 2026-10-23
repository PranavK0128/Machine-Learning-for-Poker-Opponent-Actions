"""Causal features and frozen search registries for two Stage 6 signal layers.

The approved layers are deliberately narrower than an action policy:

* ``hero_opponent_interaction`` learns from prior *completed* heads-up hands
  for an ordered actor/counterpart pair.
* ``within_hand_line_surprise`` models the public action prefix that existed
  before the current decision.

Neither layer uses decision duration, wall-clock gaps, private cards, future
actions, showdowns, or financial outcomes.  Their outputs are prediction,
confidence, abstention, and review signals only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from functools import lru_cache
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
ACTION_TO_INDEX = {value: index for index, value in enumerate(ACTIONS)}
LAYERS = ("hero_opponent_interaction", "within_hand_line_surprise")
SHARDS = (0, 1)
LOGICAL_LANE_COUNT = 48
DEVELOPMENT_CUTOFF_UTC = "2026-08-21T00:00:00.000Z"
BROAD_TRAIN_END_UTC = "2026-08-01T00:00:00.000Z"
BROAD_VALIDATION_END_UTC = "2026-08-16T00:00:00.000Z"
PINNED_RUNTIME_DISTRIBUTIONS = (
    ("numpy", "numpy", "2.2.6"),
    ("pandas", "pandas", "2.3.3"),
    ("pyarrow", "pyarrow", "21.0.0"),
    ("scikit-learn", "sklearn", "1.7.2"),
    ("joblib", "joblib", "1.5.2"),
    ("lightgbm", "lightgbm", "4.6.0"),
    ("xgboost", "xgboost", "3.3.0"),
    ("catboost", "catboost", "1.2.10"),
    ("threadpoolctl", "threadpoolctl", "3.6.0"),
    ("psutil", "psutil", "7.0.0"),
)
TOURNAMENT_SEED = 6_202_608

# Explicitly reject decision-speed and inter-action-time features.  UTC fields
# remain available only for causal ordering and temporal partition assignment.
PROHIBITED_TIMING_FEATURE_TOKENS = (
    "action_timing",
    "decision_duration",
    "decision_latency",
    "elapsed_seconds",
    "inter_action_time",
    "think_time",
    "time_to_act",
)
PROHIBITED_INFORMATION_FEATURE_TOKENS = (
    "private_card",
    "future_",
    "showdown",
    "hand_outcome",
    "financial_",
    "profit",
    "net_won",
)

IDENTITY_AND_ORDER_COLUMNS = (
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "decision_utc",
    "decision_order",
    "actor_id",
)
FEATURE_SOURCE_COLUMNS = (
    *IDENTITY_AND_ORDER_COLUMNS,
    "street",
    "action_label",
    "active_opponent_canonical_player_ids_before",
    "active_opponent_count_before",
    "facing_bet_or_raise",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def validate_document(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping) or not isinstance(document.get("self_sha256"), str):
        raise RuntimeError("self-hashed document is invalid")
    expected = str(document["self_sha256"])
    candidate = dict(document)
    candidate.pop("self_sha256", None)
    if hashlib.sha256(canonical_json_bytes(candidate)).hexdigest() != expected:
        raise RuntimeError("self-hashed document mismatch")


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_exact_counterpart(value: Any, count: Any) -> str | None:
    """Return the only visible counterpart, otherwise fail closed to ``None``."""
    try:
        observed_count = int(count)
    except (TypeError, ValueError):
        return None
    if observed_count != 1:
        return None
    if isinstance(value, str):
        try:
            values = json.loads(value)
        except json.JSONDecodeError:
            return None
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = list(value)
    else:
        return None
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        return None
    return values[0]


def _validate_source_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(FEATURE_SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"expanded-layer source is missing columns: {missing}")
    if frame["row_id"].astype(str).duplicated().any():
        raise RuntimeError("expanded-layer source contains duplicate row_id values")
    if not frame["action_label"].astype(str).isin(ACTIONS).all():
        raise RuntimeError("expanded-layer source contains an invalid action label")
    if frame[list(IDENTITY_AND_ORDER_COLUMNS)].isna().any().any():
        raise RuntimeError("expanded-layer identity/order columns contain nulls")


def build_line_prefix_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build public within-hand prefix features without looking past a decision."""
    _validate_source_frame(frame)
    ordered = frame.loc[:, FEATURE_SOURCE_COLUMNS].copy()
    ordered["decision_order"] = pd.to_numeric(ordered["decision_order"], errors="raise")
    ordered = ordered.sort_values(
        ["game_id", "hand_id", "decision_order", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    hand = ordered.groupby(["game_id", "hand_id"], sort=False, observed=True)
    if hand["decision_order"].diff().fillna(1).le(0).any():
        raise RuntimeError("within-hand decision order is not strictly increasing")

    output = pd.DataFrame({"row_id": ordered["row_id"].astype(str)})
    output["line_prefix_decision_count"] = hand.cumcount().astype(np.int32)
    aggressive = ordered["action_label"].isin(("bet", "raise", "all_in")).astype(np.int32)
    prior_aggressive = aggressive.groupby(
        [ordered["game_id"], ordered["hand_id"]], sort=False
    ).cumsum() - aggressive
    output["line_prefix_aggressive_count"] = prior_aggressive.astype(np.int32)
    denominator = output["line_prefix_decision_count"].replace(0, np.nan)
    output["line_prefix_aggressive_rate"] = (
        prior_aggressive / denominator
    ).fillna(0.0).astype(np.float32)

    actor_hand = ordered.groupby(
        ["game_id", "hand_id", "actor_id"], sort=False, observed=True
    )
    output["line_actor_prior_decision_count"] = actor_hand.cumcount().astype(np.int32)
    actor_prior_aggressive = aggressive.groupby(
        [ordered["game_id"], ordered["hand_id"], ordered["actor_id"]], sort=False
    ).cumsum() - aggressive
    output["line_actor_prior_aggressive_count"] = actor_prior_aggressive.astype(np.int32)
    actor_denominator = output["line_actor_prior_decision_count"].replace(0, np.nan)
    output["line_actor_prior_aggressive_rate"] = (
        actor_prior_aggressive / actor_denominator
    ).fillna(0.0).astype(np.float32)

    street_hand = ordered.groupby(
        ["game_id", "hand_id", "street"], sort=False, observed=True
    )
    output["line_street_prefix_decision_count"] = street_hand.cumcount().astype(np.int32)
    prior_street = hand["street"].shift(1)
    transition = (
        prior_street.notna() & prior_street.astype(str).ne(ordered["street"].astype(str))
    ).astype(np.int32)
    output["line_street_transition_count"] = transition.groupby(
        [ordered["game_id"], ordered["hand_id"]], sort=False
    ).cumsum().astype(np.int32)

    for lag in range(1, 7):
        previous_action = hand["action_label"].shift(lag)
        previous_actor = hand["actor_id"].shift(lag)
        previous_street = hand["street"].shift(lag)
        output[f"line_prev_action_{lag}_code"] = (
            previous_action.map(ACTION_TO_INDEX).fillna(-1).astype(np.int8)
        )
        output[f"line_prev_action_{lag}_same_actor"] = (
            previous_actor.notna() & previous_actor.astype(str).eq(ordered["actor_id"].astype(str))
        )
        output[f"line_prev_action_{lag}_same_street"] = (
            previous_street.notna() & previous_street.astype(str).eq(ordered["street"].astype(str))
        )

    previous_action = hand["action_label"].shift(1)
    run_boundary = previous_action.astype(str).ne(ordered["action_label"].astype(str))
    # This diagnostic is based only on already-observed prior actions: the
    # current label is never exported as a feature.  Compute the run ending at
    # the previous decision by grouping the shifted sequence.
    shifted = previous_action.fillna("__NONE__")
    boundary = shifted.groupby([ordered["game_id"], ordered["hand_id"]], sort=False).transform(
        lambda series: series.ne(series.shift(1)).cumsum()
    )
    output["line_previous_action_run_length"] = shifted.groupby(
        [ordered["game_id"], ordered["hand_id"], boundary], sort=False
    ).cumcount().add(1).where(previous_action.notna(), 0).astype(np.int16)
    del run_boundary

    output = output.set_index("row_id").loc[frame["row_id"].astype(str)].reset_index()
    assert_no_prohibited_model_features(output.columns)
    return output


def _rolling_prior_sum(
    values: pd.Series,
    group_keys: Sequence[pd.Series],
    window: int,
) -> pd.Series:
    frame = pd.DataFrame({"value": values.to_numpy(copy=False)})
    key_names = []
    for index, key in enumerate(group_keys):
        name = f"key_{index}"
        frame[name] = key.to_numpy(copy=False)
        key_names.append(name)
    result = frame.groupby(key_names, sort=False, observed=True)["value"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).sum()
    )
    return result.fillna(0)


def _action_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["is_aggressive"] = rows["action_label"].isin(("bet", "raise", "all_in")).astype(np.int16)
    rows["faced_aggression"] = rows["facing_bet_or_raise"].astype(bool).astype(np.int16)
    rows["faced_fold"] = (rows["faced_aggression"].eq(1) & rows["action_label"].eq("fold")).astype(np.int16)
    rows["faced_call"] = (rows["faced_aggression"].eq(1) & rows["action_label"].eq("call")).astype(np.int16)
    rows["faced_aggressive_response"] = (
        rows["faced_aggression"].eq(1) & rows["action_label"].isin(("raise", "all_in"))
    ).astype(np.int16)
    for action in ACTIONS:
        rows[f"action_{action}"] = rows["action_label"].eq(action).astype(np.int16)
    for street in ("preflop", "flop", "turn", "river"):
        rows[f"street_{street}_decision"] = rows["street"].eq(street).astype(np.int16)
        rows[f"street_{street}_aggressive"] = (
            rows["street"].eq(street) & rows["is_aggressive"].eq(1)
        ).astype(np.int16)
    return rows


def _aggregate_columns() -> list[str]:
    return [
        "is_aggressive",
        "faced_aggression",
        "faced_fold",
        "faced_call",
        "faced_aggressive_response",
        *(f"action_{action}" for action in ACTIONS),
        *(f"street_{street}_decision" for street in ("preflop", "flop", "turn", "river")),
        *(f"street_{street}_aggressive" for street in ("preflop", "flop", "turn", "river")),
    ]


def _entity_state_table(working: pd.DataFrame) -> pd.DataFrame:
    rows = _action_indicators(working)
    aggregate_columns = _aggregate_columns()
    summary = rows.groupby(
        ["actor_id", "game_id", "hand_id"],
        sort=False,
        observed=True,
    ).agg(
        decision_count=("row_id", "size"),
        **{column: (column, "sum") for column in aggregate_columns},
    ).reset_index()
    hand_completion = working.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["decision_utc"].max().rename("completed_utc").reset_index()
    summary = summary.merge(
        hand_completion,
        on=["game_id", "hand_id"],
        how="left",
        validate="many_to_one",
    )
    summary["actor_id"] = summary["actor_id"].astype(str)
    summary["completed_ns"] = pd.to_datetime(
        summary["completed_utc"], utc=True, errors="raise"
    ).astype("int64")
    summary = summary.sort_values(
        ["actor_id", "completed_ns", "game_id", "hand_id"], kind="mergesort"
    ).reset_index(drop=True)
    keys = [summary["actor_id"]]
    grouped = summary.groupby("actor_id", sort=False, observed=True)
    state = summary.loc[:, ["actor_id", "completed_ns"]].copy()
    cumulative: dict[str, Any] = {
        "prior_hand_count": grouped.cumcount().add(1).astype(np.int32),
    }
    for column in ("decision_count", *aggregate_columns):
        current = summary[column].astype(np.float64)
        cumulative[f"prior_{column}"] = current.groupby(
            keys, sort=False
        ).cumsum().astype(np.float32)
        for window in (16, 64):
            cumulative[f"prior_{window}_{column}"] = current.groupby(
                keys, sort=False
            ).transform(lambda series: series.rolling(window, min_periods=1).sum()).astype(np.float32)
    state = pd.concat([state, pd.DataFrame(cumulative, index=state.index)], axis=1)

    rates: dict[str, Any] = {}
    for suffix in ("", "16_", "64_"):
        source = "prior_" if not suffix else f"prior_{suffix}"
        denominator = state[f"{source}decision_count"].replace(0, np.nan)
        rates[f"{suffix}aggression_rate"] = (
            state[f"{source}is_aggressive"] / denominator
        ).astype(np.float32)
        for action in ACTIONS:
            rates[f"{suffix}{action}_rate"] = (
                state[f"{source}action_{action}"] / denominator
            ).astype(np.float32)
        faced = state[f"{source}faced_aggression"].replace(0, np.nan)
        for name in ("fold", "call", "aggressive_response"):
            rates[f"{suffix}facing_aggression_{name}_rate"] = (
                state[f"{source}faced_{name}"] / faced
            ).astype(np.float32)
    for street in ("preflop", "flop", "turn", "river"):
        denominator = state[f"prior_street_{street}_decision"].replace(0, np.nan)
        rates[f"{street}_prior_decision_count"] = state[
            f"prior_street_{street}_decision"
        ].astype(np.float32)
        rates[f"{street}_aggression_rate"] = (
            state[f"prior_street_{street}_aggressive"] / denominator
        ).astype(np.float32)
    state = pd.concat([state, pd.DataFrame(rates, index=state.index)], axis=1)
    return state.loc[:, [
        "actor_id",
        "completed_ns",
        *entity_state_value_columns(),
    ]]


def entity_state_value_columns() -> list[str]:
    columns = ["prior_hand_count"]
    for suffix in ("", "16_", "64_"):
        columns.append(f"{suffix}aggression_rate")
        columns.extend(f"{suffix}{action}_rate" for action in ACTIONS)
        columns.extend(
            f"{suffix}facing_aggression_{name}_rate"
            for name in ("fold", "call", "aggressive_response")
        )
    for street in ("preflop", "flop", "turn", "river"):
        columns.extend((
            f"{street}_prior_decision_count",
            f"{street}_aggression_rate",
        ))
    return columns


def _asof_entity_state(
    query: pd.DataFrame,
    state: pd.DataFrame,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    output_columns = [f"{prefix}_{name}" for name in entity_state_value_columns()]
    result = pd.DataFrame(np.nan, index=query.index, columns=output_columns, dtype=np.float32)
    state_groups = {
        str(entity): group.sort_values("completed_ns", kind="mergesort")
        for entity, group in state.groupby("actor_id", sort=False, observed=True)
    }
    for entity, positions in query.groupby(entity_column, sort=False, observed=True).groups.items():
        history = state_groups.get(str(entity))
        if history is None:
            continue
        position_array = np.asarray(list(positions), dtype=np.int64)
        query_times = query.loc[position_array, "hand_start_ns"].to_numpy(dtype=np.int64)
        completed = history["completed_ns"].to_numpy(dtype=np.int64)
        selected = np.searchsorted(completed, query_times, side="left") - 1
        valid = selected >= 0
        if not valid.any():
            continue
        valid_positions = position_array[valid]
        values = history.iloc[selected[valid]][entity_state_value_columns()].to_numpy(dtype=np.float32)
        result.loc[valid_positions, [f"{prefix}_{name}" for name in entity_state_value_columns()]] = values
    return result


def _dyad_state_table(rows: pd.DataFrame, hand_completion: pd.DataFrame) -> pd.DataFrame:
    aggregate_columns = _aggregate_columns()
    summary = rows.groupby(
        ["actor_id", "counterpart_id", "game_id", "hand_id"],
        sort=False,
        observed=True,
    ).agg(
        decision_count=("row_id", "size"),
        **{column: (column, "sum") for column in aggregate_columns},
    ).reset_index()
    summary = summary.merge(
        hand_completion,
        on=["game_id", "hand_id"],
        how="left",
        validate="many_to_one",
    )
    summary["completed_ns"] = pd.to_datetime(
        summary["completed_utc"], utc=True, errors="raise"
    ).astype("int64")
    summary = summary.sort_values(
        ["actor_id", "counterpart_id", "completed_ns", "game_id", "hand_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    keys = [summary["actor_id"], summary["counterpart_id"]]
    grouped = summary.groupby(
        ["actor_id", "counterpart_id"], sort=False, observed=True
    )
    state = summary.loc[:, ["actor_id", "counterpart_id", "completed_ns"]].copy()
    cumulative: dict[str, Any] = {
        "dyad_prior_hand_count": grouped.cumcount().add(1).astype(np.int32),
    }
    for column in ("decision_count", *aggregate_columns):
        current = summary[column].astype(np.float64)
        cumulative[f"prior_{column}"] = current.groupby(keys, sort=False).cumsum().astype(
            np.float32
        )
        for window in (16, 64):
            cumulative[f"prior_{window}_{column}"] = current.groupby(
                keys, sort=False
            ).transform(
                lambda series: series.rolling(window, min_periods=1).sum()
            ).astype(np.float32)
    state = pd.concat([state, pd.DataFrame(cumulative, index=state.index)], axis=1)

    rates: dict[str, Any] = {}
    for suffix in ("", "16_", "64_"):
        prefix = "prior_" if not suffix else f"prior_{suffix}"
        decision_denominator = state[f"{prefix}decision_count"].replace(0, np.nan)
        rates[f"dyad_{suffix}aggression_rate"] = (
            state[f"{prefix}is_aggressive"] / decision_denominator
        ).astype(np.float32)
        for action in ACTIONS:
            rates[f"dyad_{suffix}{action}_rate"] = (
                state[f"{prefix}action_{action}"] / decision_denominator
            ).astype(np.float32)
        faced_denominator = state[f"{prefix}faced_aggression"].replace(0, np.nan)
        for name in ("fold", "call", "aggressive_response"):
            rates[f"dyad_{suffix}facing_aggression_{name}_rate"] = (
                state[f"{prefix}faced_{name}"] / faced_denominator
            ).astype(np.float32)
    for street in ("preflop", "flop", "turn", "river"):
        denominator = state[f"prior_street_{street}_decision"].replace(0, np.nan)
        rates[f"dyad_{street}_prior_decision_count"] = state[
            f"prior_street_{street}_decision"
        ].astype(np.float32)
        rates[f"dyad_{street}_aggression_rate"] = (
            state[f"prior_street_{street}_aggressive"] / denominator
        ).astype(np.float32)
    state = pd.concat([state, pd.DataFrame(rates, index=state.index)], axis=1)
    return state.loc[:, [
        "actor_id",
        "counterpart_id",
        "completed_ns",
        *dyad_interaction_feature_columns(),
    ]]


def _asof_dyad_state(query: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    value_columns = dyad_interaction_feature_columns()
    result = pd.DataFrame(np.nan, index=query.index, columns=value_columns, dtype=np.float32)
    result["dyad_prior_hand_count"] = np.float32(0)
    state_groups = {
        (str(actor), str(counterpart)): group.sort_values("completed_ns", kind="mergesort")
        for (actor, counterpart), group in state.groupby(
            ["actor_id", "counterpart_id"], sort=False, observed=True
        )
    }
    for (actor, counterpart), positions in query.groupby(
        ["actor_id", "counterpart_id"], sort=False, observed=True
    ).groups.items():
        history = state_groups.get((str(actor), str(counterpart)))
        if history is None:
            continue
        position_array = np.asarray(list(positions), dtype=np.int64)
        query_times = query.loc[position_array, "hand_start_ns"].to_numpy(dtype=np.int64)
        completed = history["completed_ns"].to_numpy(dtype=np.int64)
        selected = np.searchsorted(completed, query_times, side="left") - 1
        valid = selected >= 0
        if not valid.any():
            continue
        result.loc[position_array[valid], value_columns] = history.iloc[
            selected[valid]
        ][value_columns].to_numpy(dtype=np.float32)
    return result


def build_dyad_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build strictly-prior-hand ordered actor/counterpart interaction features."""
    _validate_source_frame(frame)
    working = frame.loc[:, FEATURE_SOURCE_COLUMNS].copy()
    working["counterpart_id"] = [
        _parse_exact_counterpart(value, count)
        for value, count in zip(
            working["active_opponent_canonical_player_ids_before"],
            working["active_opponent_count_before"],
            strict=True,
        )
    ]
    eligible = working["counterpart_id"].notna() & working["counterpart_id"].astype(str).ne(
        working["actor_id"].astype(str)
    )
    output = pd.DataFrame({
        "row_id": working["row_id"].astype(str),
        "dyad_eligible": eligible.astype(bool),
    })
    if not eligible.any():
        empty = pd.DataFrame(
            np.nan,
            index=output.index,
            columns=dyad_numeric_feature_columns(),
            dtype=np.float32,
        )
        return pd.concat([output, empty], axis=1)

    hand_times = working.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["decision_utc"].agg(hand_start_utc="min", completed_utc="max").reset_index()
    rows = working.loc[eligible].copy().merge(
        hand_times.loc[:, ["game_id", "hand_id", "hand_start_utc"]],
        on=["game_id", "hand_id"],
        how="left",
        validate="many_to_one",
    )
    rows["counterpart_id"] = rows["counterpart_id"].astype(str)
    rows["actor_id"] = rows["actor_id"].astype(str)
    rows["decision_utc"] = rows["decision_utc"].astype(str)
    rows = _action_indicators(rows)
    rows["hand_start_ns"] = pd.to_datetime(
        rows["hand_start_utc"], utc=True, errors="raise"
    ).astype("int64")

    # Matched marginal histories contain the same actor and counterpart
    # information without an interaction term.  Qualification must beat this
    # control, not merely a context-only baseline.
    state = _entity_state_table(working)
    actor_main = _asof_entity_state(rows, state, "actor_id", "main_actor")
    counterpart_main = _asof_entity_state(
        rows, state, "counterpart_id", "main_counterpart"
    )
    rows = pd.concat([rows, actor_main, counterpart_main], axis=1)

    dyad_state = _dyad_state_table(
        rows,
        hand_times.loc[:, ["game_id", "hand_id", "completed_utc"]],
    )
    interaction = _asof_dyad_state(rows, dyad_state)
    rows = pd.concat([rows, interaction], axis=1)
    merged = rows.loc[:, [
        "row_id",
        *dyad_main_effect_feature_columns(),
        *dyad_interaction_feature_columns(),
    ]]
    output = output.merge(
        merged,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    assert_no_prohibited_model_features(
        column for column in output.columns if column not in {"row_id", "dyad_eligible"}
    )
    return output


def line_numeric_feature_columns() -> list[str]:
    columns = [
        "line_prefix_decision_count",
        "line_prefix_aggressive_count",
        "line_prefix_aggressive_rate",
        "line_actor_prior_decision_count",
        "line_actor_prior_aggressive_count",
        "line_actor_prior_aggressive_rate",
        "line_street_prefix_decision_count",
        "line_street_transition_count",
        "line_previous_action_run_length",
    ]
    for lag in range(1, 7):
        columns.extend((
            f"line_prev_action_{lag}_code",
            f"line_prev_action_{lag}_same_actor",
            f"line_prev_action_{lag}_same_street",
        ))
    return columns


def dyad_interaction_feature_columns() -> list[str]:
    columns = ["dyad_prior_hand_count"]
    for suffix in ("", "16_", "64_"):
        columns.append(f"dyad_{suffix}aggression_rate")
        columns.extend(f"dyad_{suffix}{action}_rate" for action in ACTIONS)
        columns.extend(
            f"dyad_{suffix}facing_aggression_{name}_rate"
            for name in ("fold", "call", "aggressive_response")
        )
    for street in ("preflop", "flop", "turn", "river"):
        columns.extend((
            f"dyad_{street}_prior_decision_count",
            f"dyad_{street}_aggression_rate",
        ))
    return columns


def dyad_main_effect_feature_columns() -> list[str]:
    return [
        f"{prefix}_{name}"
        for prefix in ("main_actor", "main_counterpart")
        for name in entity_state_value_columns()
    ]


def dyad_numeric_feature_columns() -> list[str]:
    return [
        *dyad_main_effect_feature_columns(),
        *dyad_interaction_feature_columns(),
    ]


def assert_no_prohibited_model_features(columns: Sequence[str]) -> None:
    prohibited = (*PROHIBITED_TIMING_FEATURE_TOKENS, *PROHIBITED_INFORMATION_FEATURE_TOKENS)
    violations = sorted({
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in prohibited)
    })
    if violations:
        raise RuntimeError(f"prohibited expanded-layer model features: {violations}")


def _hash_sample_grid(
    family: str,
    grid: Mapping[str, Sequence[Any]],
    count: int,
) -> list[dict[str, Any]]:
    keys = tuple(grid)
    candidates = [dict(zip(keys, values, strict=True)) for values in itertools.product(
        *(grid[key] for key in keys)
    )]
    candidates.sort(key=lambda parameters: _stable_digest({
        "family": family,
        "parameters": parameters,
        "seed": TOURNAMENT_SEED,
    }))
    if len(candidates) < count:
        raise RuntimeError(f"{family} grid has only {len(candidates)} combinations")
    return candidates[:count]


@lru_cache(maxsize=1)
def _family_parameter_sets() -> dict[str, list[dict[str, Any]]]:
    return {
        "lightgbm": _hash_sample_grid("lightgbm", {
            "num_leaves": (15, 23, 31, 47, 63, 95, 127),
            "max_depth": (-1, 6, 8, 10, 12),
            "min_child_samples": (40, 75, 120, 180, 260, 400),
            "learning_rate": (0.025, 0.035, 0.05, 0.07, 0.09),
            "subsample": (0.75, 0.85, 0.95, 1.0),
            "colsample_bytree": (0.70, 0.80, 0.90, 1.0),
            "reg_alpha": (0.0, 0.1, 0.5, 1.0),
            "reg_lambda": (1.0, 3.0, 6.0, 10.0, 16.0),
        }, 64),
        "xgboost": _hash_sample_grid("xgboost", {
            "max_depth": (3, 4, 5, 6, 7, 8, 10),
            "min_child_weight": (1.0, 3.0, 5.0, 10.0, 20.0),
            "learning_rate": (0.025, 0.035, 0.05, 0.07, 0.09),
            "subsample": (0.75, 0.85, 0.95, 1.0),
            "colsample_bytree": (0.70, 0.80, 0.90, 1.0),
            "reg_alpha": (0.0, 0.1, 0.5, 1.0),
            "reg_lambda": (1.0, 3.0, 6.0, 10.0, 16.0),
            "max_bin": (128, 256, 512),
        }, 64),
        "catboost": _hash_sample_grid("catboost", {
            "depth": (4, 5, 6, 7, 8, 9, 10),
            "learning_rate": (0.025, 0.035, 0.05, 0.07, 0.09),
            "l2_leaf_reg": (2.0, 4.0, 6.0, 9.0, 13.0, 18.0),
            "random_strength": (0.0, 0.25, 0.5, 1.0, 2.0),
            "bagging_temperature": (0.0, 0.5, 1.0, 2.0),
        }, 48),
        "hist_gradient_boosting": _hash_sample_grid("hist_gradient_boosting", {
            "max_leaf_nodes": (15, 23, 31, 47, 63, 95, 127),
            "max_depth": (None, 5, 7, 9, 12),
            "learning_rate": (0.025, 0.035, 0.05, 0.07, 0.09),
            "min_samples_leaf": (40, 75, 120, 200, 320),
            "l2_regularization": (0.0, 1.0, 3.0, 6.0, 10.0, 16.0),
        }, 32),
        "extra_trees": _hash_sample_grid("extra_trees", {
            "n_estimators": (128, 192, 256),
            "max_depth": (10, 14, 18),
            "min_samples_leaf": (8, 16, 32, 64),
            "max_features": (0.35, 0.50, 0.70, 0.90, 1.0),
            "class_weight": (None, "balanced"),
        }, 24),
        "random_forest": _hash_sample_grid("random_forest", {
            "n_estimators": (128, 192, 256),
            "max_depth": (10, 14, 18),
            "min_samples_leaf": (8, 16, 32, 64),
            "max_features": ("sqrt", 0.35, 0.50, 0.70),
            "class_weight": (None, "balanced_subsample"),
        }, 8),
        "logistic_regression": _hash_sample_grid("logistic_regression", {
            "C": (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0),
            "l1_ratio": (0.0, 0.05, 0.15, 0.30, 0.50, 0.75),
            "class_weight": (None, "balanced"),
        }, 8),
        "gaussian_nb": [
            {"var_smoothing": value}
            for value in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)
        ],
    }


@lru_cache(maxsize=len(LAYERS))
def candidate_registry(layer: str) -> list[dict[str, Any]]:
    if layer not in LAYERS:
        raise ValueError(f"unknown expanded layer: {layer}")
    rows: list[dict[str, Any]] = []
    for family, parameter_sets in _family_parameter_sets().items():
        for index, parameters in enumerate(parameter_sets):
            payload = {
                "layer": layer,
                "family": family,
                "parameters": parameters,
                "registry_seed": TOURNAMENT_SEED,
            }
            digest = _stable_digest(payload)
            rows.append({
                "candidate_id": f"{family}-{index:03d}-{digest[:12]}",
                "family": family,
                "parameters": parameters,
                "configuration_sha256": digest,
            })
    if len(rows) != 256 or len({row["candidate_id"] for row in rows}) != 256:
        raise RuntimeError("expanded-layer registry must contain 256 unique candidates")
    return rows


def candidate_registry_shard(layer: str, shard: int) -> list[dict[str, Any]]:
    if shard not in SHARDS:
        raise ValueError(f"unknown expanded-layer shard: {shard}")
    rows = [row for index, row in enumerate(candidate_registry(layer)) if index % 2 == shard]
    if len(rows) != 128:
        raise RuntimeError("expanded-layer shard must contain exactly 128 candidates")
    return rows


def registry_digest(layer: str) -> str:
    return _stable_digest(candidate_registry(layer))


def shard_digest(layer: str, shard: int) -> str:
    return _stable_digest(candidate_registry_shard(layer, shard))


def temporal_role(decision_utc: str) -> str:
    value = str(decision_utc)
    if value < BROAD_TRAIN_END_UTC:
        return "development_train"
    if value < BROAD_VALIDATION_END_UTC:
        return "development_validation"
    if value < DEVELOPMENT_CUTOFF_UTC:
        return "retrospective_calibration"
    return "fresh_adjudication"


def expanded_layer_contract() -> dict[str, Any]:
    return seal_document({
        "schema_version": "stage6_expanded_predictive_layers_contract_v1",
        "status": "USER_APPROVED_DEVELOPMENT_CONTRACT__FRESH_QUALIFICATION_REQUIRED",
        "layers": {
            "hero_opponent_interaction": {
                "eligibility": "exactly_one_visible_active_counterpart",
                "history_cutoff": "strictly_prior_completed_hands_only",
                "raw_identity_as_model_feature": False,
                "matched_main_effects_control_required": True,
                "qualification_strata": [
                    "known_directed_dyad",
                    "unseen_directed_dyad_with_known_players",
                ],
                "multiway_behavior": "abstain_without_one_exact_counterpart",
            },
            "within_hand_line_surprise": {
                "eligibility": "public_ordered_action_prefix_available",
                "history_cutoff": "strictly_before_current_decision",
                "observed_current_action_as_model_feature": False,
            },
        },
        "action_timing_layer_included": False,
        "prohibited_timing_feature_tokens": list(PROHIBITED_TIMING_FEATURE_TOKENS),
        "prohibited_information_feature_tokens": list(PROHIBITED_INFORMATION_FEATURE_TOKENS),
        "decision_utc_use": ["causal_ordering", "temporal_partition_assignment"],
        "decision_utc_model_feature": False,
        "development_cutoff_utc_exclusive": DEVELOPMENT_CUTOFF_UTC,
        "broad_train_end_utc_exclusive": BROAD_TRAIN_END_UTC,
        "broad_validation_end_utc_exclusive": BROAD_VALIDATION_END_UTC,
        "fresh_adjudication_requirement": "never_opened_labeled_rows_at_or_after_development_cutoff",
        "logical_lane_count": LOGICAL_LANE_COUNT,
        "candidate_count_per_layer": 256,
        "candidate_count_per_shard": 128,
        "broad_shard_count_per_layer": 2,
        "temporal_partition_unit": "whole_game_hand_by_max_decision_utc_later_role_wins",
        "candidate_internal_threads": 1,
        "pinned_runtime_distributions": {
            distribution: version
            for distribution, _, version in PINNED_RUNTIME_DISTRIBUTIONS
        },
        "live_candidate_workers": "resource_aware_auto_bounded_by_logical_cpu_ram_and_unfinished_candidates",
        "checkpoint_rule": "publish_exact_self_hashed_result_immediately_after_each_candidate_stage",
        "shard_comparability": "identical_nested_hand_time_rung_row_roots_and_seeds",
        "finalist_calibration": "freeze_on_reserved_retrospective_calibration_slice_after_refit",
        "fresh_claim_multiplicity_control": "holm_two_sided_two_claim_family",
        "fresh_inference_cluster_unit": "game_id",
        "minimum_independent_game_clusters_per_claim": 20,
        "authority": ["prediction", "confidence", "abstention", "review"],
        "action_changing_exploit_authority": False,
        "cfr_or_dcfr_dependency": False,
        "assistant_installation_allowed_before_fresh_qualification": False,
        "assistant_consumption_after_qualification": "byte_bound_diagnostics_adapter_only",
        "assistant_diagnostic_outputs": [
            "prediction",
            "confidence",
            "surprise",
            "support",
            "abstention",
        ],
        "assistant_recommendation_authority": False,
        "assistant_live_action_authority": False,
    })


EXPANDED_LAYER_CONTRACT = expanded_layer_contract()
EXPANDED_LAYER_CONTRACT_DIGEST = EXPANDED_LAYER_CONTRACT["self_sha256"]

SCIENCE_IMPLEMENTATION_MEMBERS = (
    "src/stage6_common.py",
    "src/stage6_metrics.py",
    "src/stage6_notebooks02_07_runtime.py",
    "src/stage6_notebook01_materializer.py",
    "src/stage6_behavior_tournament.py",
    "src/stage6_expanded_layers.py",
    "src/stage6_expanded_feature_freeze.py",
    "src/stage6_expanded_tournament.py",
    "src/stage6_expanded_finalist.py",
    "src/stage6_expanded_adjudication.py",
    "src/stage6_expanded_diagnostics_adapter.py",
    "contracts/stage6_data_contract_v1.json",
    "contracts/stage6_split_contract_v1.json",
    "contracts/stage6_scientific_gate_v1.json",
    "contracts/stage6_worker_policy_v3.json",
    "contracts/stage6_expanded_predictive_layers_contract_v1.json",
)


def science_implementation_binding() -> dict[str, Any]:
    stage_root = Path(__file__).resolve().parents[1]
    members = []
    for relative_path in SCIENCE_IMPLEMENTATION_MEMBERS:
        path = stage_root / relative_path
        raw = path.read_bytes()
        members.append({
            "relative_path": relative_path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return seal_document({
        "schema_version": "stage6_expanded_science_implementation_binding_v1",
        "member_order": list(SCIENCE_IMPLEMENTATION_MEMBERS),
        "members": members,
    })


SCIENCE_IMPLEMENTATION_BINDING = science_implementation_binding()
SCIENCE_IMPLEMENTATION_DIGEST = SCIENCE_IMPLEMENTATION_BINDING["self_sha256"]
