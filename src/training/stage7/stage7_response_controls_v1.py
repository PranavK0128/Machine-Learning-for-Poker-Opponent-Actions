"""Fixed training-cutoff stale features and descriptive opportunity support.

These pure helpers consume caller-authenticated source rows and feature arrays.
They neither fit models nor select thresholds from evaluation outcomes. Exhaust
the stale generator before accepting its complete-game inventory. Notebook05's
controller must still bind and publish their outputs before execution release.
"""
from __future__ import annotations

import math
import numpy as np
import stage7_hierarchical_response_v1 as h

SCHEMA = "stage7_response_controls_v1"
SUPPORT_BINS = (("0", 0, 1), ("1-4", 1, 5), ("5-9", 5, 10), ("10-24", 10, 25),
                ("25-49", 25, 50), ("50-99", 50, 100), ("100+", 100, None))
STALE_RULE = "freeze_population_player_session_counts_after_selected_training_games; no_calibration_or_evaluation_updates"


def _protocol(protocol, game_bounds, window):
    if window not in ("50", "75", "100"):
        raise ValueError("control_fixed_window_required")
    predecessor = dict(train_game_ids=protocol["train_game_ids"], calibration_game_ids=protocol["calibration_game_ids"],
                       evaluation_game_ids=protocol["development_evaluation_game_ids"],
                       diagnostic_game_ids=protocol["excluded_diagnostic_game_ids"], model_config=protocol["model_config"])
    expected = h.development_protocol(predecessor, game_bounds)
    if h.incumbent.canonical(protocol) != h.incumbent.canonical(expected):
        raise ValueError("control_frozen_chronological_protocol_required")
    return set(protocol["training_windows"][window]), set(protocol["development_evaluation_game_ids"])


def _row(row, games, bounds):
    clocks = h.validate_observation(row)
    if (row["game_id"] not in games
            or [row["game_started_utc"], row["game_completed_utc"]] != list(bounds[row["game_id"]])):
        raise ValueError("control_row_cohort_or_game_bounds_mismatch")
    return clocks


def stale_feature_stream(training_rows, evaluation_rows, protocol, game_bounds, window):
    """Use only completed selected-training evidence for all later predictions.

    Training input is ordered by (completion, row ID); evaluation input by
    (decision, row ID). Current public state, conditional action and elapsed age
    remain current. Counts and posteriors stay fixed; there is no stale refit or
    stale-temperature selection. The caller applies the unchanged session model.
    """
    training_games, evaluation_games = _protocol(protocol, game_bounds, window)
    history = h.History()
    actual_train, previous = set(), None
    for row in training_rows:
        clocks = _row(row, training_games, game_bounds)
        order = (clocks["hand_completed_utc"], row["row_id"])
        if previous is not None and order <= previous:
            raise ValueError("control_training_cursor_not_unique_chronological")
        previous = order
        history.observe(row)
        actual_train.add(row["game_id"])
    if actual_train != training_games:
        raise ValueError("control_incomplete_training_game_inventory")
    actual_evaluation, evaluation_ids, previous = set(), set(), None
    frozen_observations = history.observations
    for row in evaluation_rows:
        clocks = _row(row, evaluation_games, game_bounds)
        order = (clocks["decision_utc"], row["row_id"])
        if (previous is not None and order <= previous) or row["row_id"] in evaluation_ids or row["row_id"] in history.observed_ids:
            raise ValueError("control_evaluation_cursor_not_unique_chronological")
        previous = order
        evaluation_ids.add(row["row_id"])
        actual_evaluation.add(row["game_id"])
        features = {}
        support = {}
        for role in h.ROLES:
            if h.target(row, role) is None:
                continue
            features[role] = h.feature_vector(row, role, "session", history)
            summary = history.summary(row, role)
            support[role] = dict(population_opportunities=int(summary["counts"][1]),
                                 player_opportunities=int(summary["counts"][2]),
                                 session_opportunities=int(summary["counts"][3]),
                                 player_context_age_days=summary["age_days"],
                                 latest_history_completion=summary["latest_history_completion"])
        yield row, features, dict(schema_version=SCHEMA, rule=STALE_RULE, window=window,
                                 frozen_training_observations=frozen_observations, support=support)
    if actual_evaluation != evaluation_games:
        raise ValueError("control_incomplete_evaluation_game_inventory")


def support_groups(features, role):
    """Partition rows by exact context opportunity counts, not causal adaptation.

    Input is the unchanged full session feature schema, normally the fresh
    preparation restricted to the comparison's explicit common row set. Bins are
    disjoint empirical support slices, not exact-N truncation experiments or a
    trained drift detector. No outcome or predicted probability chooses a bin.
    """
    names = h.feature_names(role, "session")
    values = np.asarray(features)
    if values.dtype != np.dtype(np.float64) or values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError("control_exact_finite_feature_schema_required")
    result = {}
    for level in ("population", "player", "session"):
        logs = values[:, names.index(level + "_log_opportunities")]
        # Integer opportunities are exactly what the authenticated preparation
        # logged. Reject fabricated negative/fractional/overflowed supports.
        if np.any(logs < 0) or np.any(logs > math.log1p(2 ** 53 - 1)):
            raise ValueError("control_invalid_opportunity_count")
        counts = np.rint(np.expm1(logs))
        if np.any(np.abs(np.log1p(counts) - logs) > 2 * np.spacing(logs)):
            raise ValueError("control_noninteger_opportunity_count")
        counts = counts.astype(np.int64)
        result[level] = {label: np.flatnonzero((counts >= lower) & (True if upper is None else counts < upper)).tolist()
                         for label, lower, upper in SUPPORT_BINS}
        if sorted(index for indices in result[level].values() for index in indices) != list(range(len(values))):
            raise ValueError("control_support_partition_mismatch")
    return dict(schema_version=SCHEMA + "_support_groups", role=role, rows=len(values), groups=result,
                cutpoints=[0, 1, 5, 10, 25, 50, 100], interval_semantics="lower inclusive; upper exclusive; final unbounded",
                claim="descriptive observed-opportunity support slices; not causal or exact-N adaptation",
                evidence_class=h.EVIDENCE, authority=h.AUTHORITY)
