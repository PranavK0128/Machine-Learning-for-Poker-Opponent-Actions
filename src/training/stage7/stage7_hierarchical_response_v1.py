"""Development-only observed-response candidates with causal target statistics.

No fitting happens on import. Full-corpus work belongs in Colab. This module
does not load or alter the Assistant registry. Size means the coarse observed
size conditional on an observed aggressive action, never response to an
unchosen Hero size. Missing private holdings are not features or labels here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np
import stage6_empirical_range_learning_v2 as incumbent

SCHEMA = "stage7_hierarchical_response_v1"
ACTION = "observed_action_family"
SIZE = "conditional_aggressive_size"
ROLES = (ACTION, SIZE)
AGGRESSIVE = ("bet", "raise", "all_in")
LABELS = {ACTION: tuple(incumbent.ACTIONS), SIZE: ("small", "medium", "large")}
ARMS = ("public_only", "population", "player", "session")
STRENGTH = 100.0
TEMPERATURES = (.5, .75, 1.0, 1.25, 1.5, 2.0)
AUTHORITY = dict(incumbent.AUTHORITY)
EVIDENCE = "DEVELOPMENT_REUSE_ONLY"


def validate_observation(row):
    """Check the compact public contract without pretending to re-audit source."""
    for key in ("row_id", "game_id", "hand_id", "actor_id", "session_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("missing_exact_identity:" + key)
    clocks = {name: incumbent.utc(row[name]) for name in (
        "game_started_utc", "game_completed_utc", "decision_utc",
        "public_available_utc", "action_observed_utc", "hand_completed_utc")}
    if not (clocks["game_started_utc"] <= clocks["decision_utc"]
            <= clocks["action_observed_utc"] <= clocks["hand_completed_utc"]
            <= clocks["game_completed_utc"]
            and clocks["public_available_utc"] <= clocks["decision_utc"]):
        raise ValueError("invalid_observation_chronology")
    incumbent.public_features(row["public"])
    legal = row["legal_actions"]
    if (not isinstance(legal, list) or not legal or len(legal) != len(set(legal))
            or not set(legal) <= set(LABELS[ACTION]) or row["action"] not in legal):
        raise ValueError("invalid_observed_action_or_legal_mask")
    if row["action"] in AGGRESSIVE:
        size = row.get("observed_action_size_pot_fraction")
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0:
            raise ValueError("missing_or_invalid_observed_size")
    return clocks


def target(row, role):
    if role == ACTION:
        return LABELS[role].index(row["action"])
    if role != SIZE:
        raise ValueError("unknown_response_role")
    if row["action"] not in AGGRESSIVE:
        return None
    size = row.get("observed_action_size_pot_fraction")
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0:
        raise ValueError("missing_or_invalid_observed_size")
    return 0 if size < .5 else 1 if size <= 1 else 2


def context_key(row, role):
    """Only predecision public fields; observed action is allowed for SIZE only."""
    public = row["public"]
    depth = public["effective_stack_before_bb"]
    depth_band = "under150" if depth < 150 else "150_200" if depth < 200 else "200_250" if depth <= 250 else "over250"
    opponents = public["active_opponent_count_before"]
    formation = "zero" if opponents == 0 else "heads_up" if opponents == 1 else "multiway"
    legal = tuple(action for action in LABELS[ACTION] if action in row["legal_actions"])
    if role not in ROLES:
        raise ValueError("unknown_response_role")
    conditional = row["action"] if role == SIZE else ""
    if role == SIZE and conditional not in AGGRESSIVE:
        raise ValueError("conditional_size_requires_aggressive_action")
    return (public["street"], public["opportunity"], public["position"], depth_band, formation, legal, conditional)


def shrink(counts, parent, strength=STRENGTH):
    counts, parent = np.asarray(counts, dtype=float), np.asarray(parent, dtype=float)
    if (counts.shape != parent.shape or counts.ndim != 1 or len(counts) == 0
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(parent).all() or np.any(parent < 0)
            or not np.isclose(parent.sum(), 1) or not math.isfinite(strength) or strength <= 0):
        raise ValueError("invalid_hierarchical_counts")
    return (counts + strength * parent) / (counts.sum() + strength)


@dataclass
class Counts:
    values: np.ndarray
    last_completed: float | None = None

    def add(self, label, completed):
        self.values[label] += 1
        self.last_completed = completed if self.last_completed is None else max(completed, self.last_completed)


@dataclass
class History:
    """Statistics of completed earlier hands; no fitted model or private cards."""
    tables: dict = field(default_factory=dict)
    last_completed: float = -math.inf
    observations: int = 0
    hand_completions: dict = field(default_factory=dict)
    observed_ids: set = field(default_factory=set)
    observed_hands: set = field(default_factory=set)

    def bind_hand(self, row):
        completion = incumbent.utc(row["hand_completed_utc"])
        key = (row["game_id"], row["hand_id"])
        if self.hand_completions.setdefault(key, completion) != completion:
            raise ValueError("inconsistent_hand_completion")

    def _get(self, key, width):
        # A query never creates state, which also makes ablation order inert.
        return self.tables.get(key, Counts(np.zeros(width, dtype=float)))

    def observe(self, row):
        clocks = validate_observation(row)
        self.bind_hand(row)
        if row["row_id"] in self.observed_ids:
            raise ValueError("duplicate_observation_row_id")
        completed = clocks["hand_completed_utc"]
        if completed < self.last_completed:
            raise ValueError("history_observations_not_sorted_by_completion")
        self.last_completed = completed
        self.observed_ids.add(row["row_id"])
        self.observed_hands.add((row["game_id"], row["hand_id"]))
        for role in ROLES:
            label = target(row, role)
            if label is None:
                continue
            context = context_key(row, role)
            # Legal set and conditional action also define the broad fallback.
            broad = (context[-2], context[-1])
            keys = ((role, "global", broad), (role, "population", context),
                    (role, "player", context, row["actor_id"]),
                    (role, "session", context, row["actor_id"], row["game_id"], row["session_id"]))
            for key in keys:
                self.tables.setdefault(key, Counts(np.zeros(len(LABELS[role])))).add(label, completed)
        self.observations += 1

    def summary(self, row, role):
        self.bind_hand(row)
        if (row["game_id"], row["hand_id"]) in self.observed_hands:
            raise ValueError("current_hand_already_in_history")
        context = context_key(row, role)
        width = len(LABELS[role])
        broad = (context[-2], context[-1])
        states = [self._get(key, width) for key in (
            (role, "global", broad), (role, "population", context),
            (role, "player", context, row["actor_id"]),
            (role, "session", context, row["actor_id"], row["game_id"], row["session_id"]))]
        decision = incumbent.utc(row["decision_utc"])
        if any(s.last_completed is not None and s.last_completed >= decision for s in states):
            raise ValueError("history_not_strictly_before_decision")
        legal = np.asarray([label in row["legal_actions"] for label in LABELS[ACTION]]) if role == ACTION else np.ones(width, dtype=bool)
        base = legal.astype(float) / legal.sum()
        global_prior = shrink(states[0].values, base, float(legal.sum()))
        population = shrink(states[1].values, global_prior)
        player = shrink(states[2].values, population)
        session = shrink(states[3].values, player)
        # Total variation is a descriptive drift feature, not a hypothesis test.
        drift = float(np.abs(session - player).sum() / 2)
        last = states[2].last_completed
        age_days = (decision - last) / 86400 if last is not None else 0.0
        counts = [float(s.values.sum()) for s in states]
        return dict(population=population, player=player, session=session,
                    counts=counts, age_days=age_days, drift=drift,
                    unseen=counts[2] == 0, sparse=counts[2] < 25,
                    latest_history_completion=max((s.last_completed for s in states if s.last_completed is not None), default=None))


def feature_names(role, arm):
    if role not in ROLES or arm not in ARMS:
        raise ValueError("unknown_role_or_ablation")
    names = list(incumbent.PUBLIC_NAMES)
    if role == SIZE:
        names += ["conditional_action_" + action for action in AGGRESSIVE]
    for level in ARMS[1:ARMS.index(arm) + 1]:
        names += [level + "_rate_" + label for label in LABELS[role]]
        names += [level + "_log_opportunities"]
    if arm in ("player", "session"):
        names += ["player_context_unseen", "player_context_sparse", "player_context_age_days"]
    if arm == "session":
        names += ["session_lifetime_total_variation"]
    return names


def feature_vector(row, role, arm, history):
    feature_names(role, arm)  # Validate even public-only calls.
    values = incumbent.public_features(row["public"])
    if role == SIZE:
        if row["action"] not in AGGRESSIVE:
            raise ValueError("conditional_size_requires_aggressive_action")
        values += [float(row["action"] == action) for action in AGGRESSIVE]
    if arm == "public_only":
        return values
    summary = history.summary(row, role)
    for level in ARMS[1:ARMS.index(arm) + 1]:
        values += summary[level].tolist() + [math.log1p(summary["counts"][ARMS.index(level)])]
    if arm in ("player", "session"):
        values += [float(summary["unseen"]), float(summary["sparse"]), summary["age_days"]]
    if arm == "session":
        values += [summary["drift"]]
    return values


def causal_feature_stream(decisions, completed_observations, history=None):
    """Merge two authenticated source cursors without retaining all raw rows.

    Callers supply unique decisions ordered by (decision time, row ID) and
    unique observations ordered by (hand completion, row ID). Only strictly
    earlier hand completions are released, including during calibration and
    development evaluation. Model weights remain frozen during that replay.
    """
    state = history if history is not None else History()
    observations = iter(completed_observations)
    previous_decision = previous_observation = None
    decision_ids, observation_ids = set(), set(state.observed_ids)

    def acquire():
        nonlocal previous_observation
        observed = next(observations, None)
        if observed is None:
            return None
        observed_clocks = validate_observation(observed)
        observed_order = (observed_clocks["hand_completed_utc"], observed["row_id"])
        if previous_observation is not None and observed_order <= previous_observation:
            raise ValueError("observations_not_unique_chronological")
        if observed["row_id"] in observation_ids:
            raise ValueError("duplicate_observation_row_id")
        observation_ids.add(observed["row_id"])
        previous_observation = observed_order
        state.bind_hand(observed)
        return observed

    pending = acquire()
    for row in decisions:
        clocks = validate_observation(row)
        order = (clocks["decision_utc"], row["row_id"])
        if previous_decision is not None and order <= previous_decision:
            raise ValueError("decisions_not_unique_chronological")
        if row["row_id"] in decision_ids:
            raise ValueError("duplicate_decision_row_id")
        decision_ids.add(row["row_id"])
        state.bind_hand(row)
        previous_decision = order
        while pending is not None and incumbent.utc(pending["hand_completed_utc"]) < clocks["decision_utc"]:
            state.observe(pending)
            pending = acquire()
        features = {role: {arm: feature_vector(row, role, arm, state) for arm in ARMS}
                    for role in ROLES if role == ACTION or row["action"] in AGGRESSIVE}
        audit = state.summary(row, ACTION)
        yield row, features, {key: audit[key] for key in ("counts", "age_days", "drift", "unseen", "sparse", "latest_history_completion")}
    # Exhaustion is part of validation: never silently bless malformed trailing
    # rows, but do not commit them to the history used by this replay.
    while pending is not None:
        pending = acquire()


def normalize(probabilities, rows, role):
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (len(rows), len(LABELS[role])) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p.sum(axis=1) <= 0)):
        raise ValueError("invalid_probability_matrix")
    if role == ACTION:
        return incumbent.legal_probabilities(p, rows, LABELS[role])
    return p / p.sum(axis=1, keepdims=True)


def collapse_incumbent(probabilities, labels, rows, role):
    """Project the frozen joint target onto a compatible evaluation role."""
    p = np.asarray(probabilities, dtype=float)
    allowed = {a + "@none" for a in LABELS[ACTION] if a not in AGGRESSIVE}
    allowed |= {a + "@" + size for a in AGGRESSIVE for size in LABELS[SIZE]}
    if (role not in ROLES or len(labels) != len(set(labels)) or not set(labels) <= allowed
            or p.shape != (len(rows), len(labels)) or not np.isfinite(p).all() or np.any(p < 0)):
        raise ValueError("invalid_incumbent_joint_predictions")
    result = np.zeros((len(rows), len(LABELS[role])))
    if role == ACTION:
        for j, label in enumerate(labels):
            result[:, LABELS[ACTION].index(label.split("@")[0])] += p[:, j]
    else:
        for i, row in enumerate(rows):
            if row["action"] not in AGGRESSIVE:
                raise ValueError("conditional_size_requires_aggressive_action")
            for size in LABELS[SIZE]:
                label = row["action"] + "@" + size
                if label not in labels:
                    raise ValueError("incumbent_conditional_size_class_absent")
                result[i, LABELS[SIZE].index(size)] = p[i, labels.index(label)]
    return normalize(result, rows, role)


def batch_tree_prediction(model, features):
    """Vectorized portable JSON scorer; same exported semantics as incumbent."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["feature_names"]):
        raise ValueError("portable_feature_schema_mismatch")
    tree_count = model["trees"]["num_tree_per_iteration"]
    raw = np.zeros((len(x), tree_count))
    for tree in model["trees"]["tree_info"]:
        stack = [(tree["tree_structure"], np.arange(len(x)))]
        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if "leaf_value" in node:
                raw[indices, tree["tree_index"] % tree_count] += node["leaf_value"]
                continue
            if node["decision_type"] != "<=":
                raise ValueError("unsupported_tree_split")
            values = x[indices, node["split_feature"]]
            left = np.where(np.isnan(values), node["default_left"], values <= float(node["threshold"]))
            stack += [(node["left_child"], indices[left]), (node["right_child"], indices[~left])]
    if tree_count == 1:
        positive = 1 / (1 + np.exp(-np.clip(raw[:, 0], -700, 700)))
        values = np.column_stack((1 - positive, positive))
    else:
        values = np.exp(raw - raw.max(axis=1, keepdims=True))
        values /= values.sum(axis=1, keepdims=True)
    result = np.full((len(x), len(model["labels"])), 1e-12)
    result[:, model["fitted_class_indices"]] = values
    return incumbent.temper(result, model.get("temperature", 1.0))


def fit_candidate(train_x, train_rows, calibration_x, calibration_rows, role, arm, config):
    """Fixed family/configuration; scalar temperature uses calibration only."""
    width = len(feature_names(role, arm))
    if np.asarray(train_x).shape != (len(train_rows), width) or np.asarray(calibration_x).shape != (len(calibration_rows), width):
        raise ValueError("candidate_feature_schema_mismatch")
    if not train_rows or not calibration_rows:
        raise ValueError("training_and_calibration_required")
    row_ids, hand_completions, game_bounds = set(), {}, {}
    for group in (train_rows, calibration_rows):
        for row in group:
            clocks = validate_observation(row)
            if row["row_id"] in row_ids:
                raise ValueError("duplicate_fitting_row_id")
            row_ids.add(row["row_id"])
            hand = (row["game_id"], row["hand_id"])
            if hand_completions.setdefault(hand, clocks["hand_completed_utc"]) != clocks["hand_completed_utc"]:
                raise ValueError("inconsistent_hand_completion")
            bounds = (clocks["game_started_utc"], clocks["game_completed_utc"])
            if game_bounds.setdefault(row["game_id"], bounds) != bounds:
                raise ValueError("inconsistent_game_bounds")
    train_games = {r["game_id"] for r in train_rows}
    calibration_games = {r["game_id"] for r in calibration_rows}
    if train_games & calibration_games:
        raise ValueError("whole_game_cohort_overlap")
    if max(incumbent.utc(r["game_completed_utc"]) for r in train_rows) >= min(incumbent.utc(r["game_started_utc"]) for r in calibration_rows):
        raise ValueError("cohorts_not_chronological_whole_games")
    train_y = [target(r, role) for r in train_rows]
    calibration_y = [target(r, role) for r in calibration_rows]
    if None in train_y or None in calibration_y:
        raise ValueError("unsupported_role_target")
    if set(config) != {"iterations", "num_leaves", "learning_rate", "min_child_samples", "seed"}:
        raise ValueError("exact_model_configuration_required")
    model = fit_response_booster(train_x, train_y, role, config)
    if model is None:
        return None, dict(status="INSUFFICIENT_TARGET_SUPPORT", authority=AUTHORITY)
    raw = incumbent.full_prediction(model, calibration_x, LABELS[role])
    loss = lambda t: float(incumbent.loss_rows(normalize(incumbent.temper(raw, t), calibration_rows, role), calibration_y)[0].mean())
    temperature = min(TEMPERATURES, key=lambda t: (loss(t), abs(t - 1), t))
    exported = incumbent.booster_export(model, LABELS[role], feature_names(role, arm))
    exported.update(temperature=temperature, schema_version=SCHEMA, role=role, arm=arm,
                    evidence_class=EVIDENCE, authority=AUTHORITY,
                    temperature_selection="fixed_grid_calibration_whole_games_only",
                    trained_game_ids=sorted(train_games), calibration_game_ids=sorted(calibration_games))
    return exported, dict(status="FIT_DEVELOPMENT_CANDIDATE", training_rows=len(train_rows),
                          calibration_rows=len(calibration_rows), temperature=temperature,
                          calibration_log_loss=loss(temperature), authority=AUTHORITY)


def fit_response_booster(x, y, role, config):
    """Fit observed classes while retaining their fixed output-schema indices."""
    labels = LABELS[role]
    observed = np.unique(y)
    if not np.issubdtype(observed.dtype, np.integer) or any(i < 0 or i >= len(labels) for i in observed):
        raise ValueError("invalid_fixed_role_target_indices")
    # sklearn encodes y into its observed classes. LightGBM's num_class must
    # match that encoded space, including a two-class multiclass objective.
    # full_prediction/booster_export restore the original fixed label indices.
    return incumbent.fit_booster(x, y, [labels[int(i)] for i in observed], config)


def development_protocol(predecessor, game_bounds):
    """Keep the accepted partition; retire its already-inspected test claim."""
    cohorts = {name: list(predecessor[name + "_game_ids"]) for name in ("train", "calibration", "evaluation", "diagnostic")}
    flat = sum(cohorts.values(), [])
    if len(flat) != len(set(flat)) or set(flat) != set(game_bounds):
        raise ValueError("exact_whole_game_partition_required")
    for game, bounds in game_bounds.items():
        if len(bounds) != 2 or incumbent.utc(bounds[0]) > incumbent.utc(bounds[1]):
            raise ValueError("invalid_game_bounds:" + game)
    for left, right in (("train", "calibration"), ("calibration", "evaluation"), ("train", "evaluation")):
        if not cohorts[left] or not cohorts[right]:
            raise ValueError("nonempty_development_cohorts_required")
        if max(incumbent.utc(game_bounds[g][1]) for g in cohorts[left]) >= min(incumbent.utc(game_bounds[g][0]) for g in cohorts[right]):
            raise ValueError("cohorts_not_chronological_whole_games")
    ordered = sorted(cohorts["train"], key=lambda g: (incumbent.utc(game_bounds[g][1]), g))
    # Exact full-window incumbent is reused. Only shorter, distinct recipe
    # fits and separately versioned feature candidates justify new fitting.
    windows = {str(percent): ordered[:max(1, len(ordered) * percent // 100)] for percent in (50, 75, 100)}
    return dict(schema_version=SCHEMA + "_protocol", evidence_class=EVIDENCE,
                train_game_ids=cohorts["train"], calibration_game_ids=cohorts["calibration"],
                development_evaluation_game_ids=cohorts["evaluation"], excluded_diagnostic_game_ids=cohorts["diagnostic"],
                untouched_confirmation_game_ids=[], previously_inspected_game_ids=sorted(flat),
                training_windows=windows, full_window_same_recipe="REUSE_ACCEPTED_NOTEBOOK03_MODEL",
                model_config=dict(predecessor["model_config"]), arms=list(ARMS), roles=list(ROLES),
                prior_strength=STRENGTH, temperature_grid=list(TEMPERATURES),
                history_rule="strict_prior_hand_completion; prequential_public_updates; no_current_hand_or_future_target",
                history_scope="selected_training_window_then_calibration_then_development_evaluation; diagnostic_games_excluded",
                counterfactual_response_identified=False, authority=AUTHORITY)
