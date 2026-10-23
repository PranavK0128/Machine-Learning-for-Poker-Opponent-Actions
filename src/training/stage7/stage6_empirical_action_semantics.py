"""Shared wager semantics for empirical replay and the Assistant worker.

An all-in is a funding state, not an aggression label. Unknown monetary context
invalidates the last-aggressor identity instead of assigning a guessed opponent.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def aggression_role(row: Mapping[str, Any]) -> bool | None:
    action = str(row.get("action_label", row.get("action_type_canonical", "")))
    if action in {"bet", "raise"}:
        return True
    if action != "all_in":
        return False
    if "aggression_role" in row:
        value = row["aggression_role"]
        return value if isinstance(value, bool) else None
    before = number(row.get("actor_committed_before_bb"))
    after = number(row.get("actor_committed_after_bb"))
    call = number(row.get("amount_to_call_bb"))
    if before is not None and after is not None and call is not None:
        increment = after - before
        if increment <= 0 or call < 0:
            return None
        return increment > call + 1e-9
    return None


def annotate_visible_prefix(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use exact raise-to amounts or complete postflop incremental commitments.

    Preflop blind contributions are absent from normalized voluntary-action rows,
    so an unresolved preflop all-in stays unknown. Do not infer a full raise or
    reopening from a positive wager increase: those are separate legal rules.
    """
    wagers: dict[str, float | None] = {}
    commitments: dict[tuple[str, str], float] = {}
    result = []
    for source in actions:
        row = dict(source)
        street, actor = str(row["street"]), str(row["actor_id"])
        key = street, actor
        previous = wagers.get(street, None if street == "preflop" else 0.0)
        committed = commitments.get(key, None if street == "preflop" else 0.0)
        total = number(row.get("raise_to_bb"))
        increment = number(row.get("amount_bb"))
        if total is None and committed is not None and increment is not None:
            total = committed + increment
        role = aggression_role(row)
        if row["action_label"] == "all_in":
            role = None if total is None or previous is None else total > previous + 1e-9
        row["aggression_role"] = role
        if role is True:
            wagers[street] = total if total is not None else None
        elif role is None:
            wagers[street] = None
        if total is not None and row["action_label"] not in {"fold", "check"}:
            commitments[key] = total
        result.append(row)
    return result


def last_aggressor(actions: list[Mapping[str, Any]]) -> str | None:
    actor = None
    for row in actions:
        role = aggression_role(row)
        if role is True:
            actor = str(row["actor_id"])
        elif role is None:
            actor = None
    return actor
