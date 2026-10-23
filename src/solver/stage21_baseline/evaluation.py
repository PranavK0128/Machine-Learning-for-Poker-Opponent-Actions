from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from .chance import legal_next_cards, sample_flop
from .common import (
    CROSSPLAY_FAMILYWISE_ALPHA,
    CROSSPLAY_LIMIT_EFFECTIVE_STACK_FRACTION_PER_HAND,
    CROSSPLAY_MAXIMUM_MISSING_POLICY_REACH_FRACTION,
    CROSSPLAY_PAIRING_UNIT,
    sha256_json,
    with_self_hash,
)
from .continuation import ContinuationAuthority
from .coverage import combo_feature_record
from .crossplay import build_crossplay_pair_evidence
from .game import (
    Action,
    Deal,
    HoldemState,
    advance_flop,
    advance_public_card,
    apply_action,
    information_set_key,
    legal_actions,
    strategy_action_labels,
    terminal_utility,
    terminal_utilities,
    strategy_information_set_key,
)
from .ranges import JointRangeSampler


OPPONENT_FAMILIES = (
    "balanced_control",
    "uniform_legal",
    "tight_passive",
    "loose_passive",
    "tight_aggressive",
    "loose_aggressive",
    "overfold",
    "underfold",
    "overcall",
    "underbluff",
    "overbluff",
    "small_size_heavy",
    "large_size_heavy",
    "jam_heavy",
)
SAFE_CONTROL_FAMILIES = (
    "uniform_legal",
    "fold_call_control",
    "low_variance_control",
)
EVALUATION_CONTRACT = {
    "schema_version": "stage21_evaluation_contract_v5",
    "both_player_roles": True,
    "opponent_families": list(OPPONENT_FAMILIES),
    "safe_controls": list(SAFE_CONTROL_FAMILIES),
    "same_panel_candidate_and_controls": True,
    "required_heldout_panel_count": 4,
    "required_panel_roles": [
        "confirmation_1",
        "confirmation_2",
        "confirmation_3",
        "final_holdout",
    ],
    "panel_sampling_namespaces_must_be_unique": True,
    "optimization_stream_identity_must_be_distinct": True,
    "full_two_player_candidate_profile_self_play": True,
    "maximum_missing_policy_reach_fraction": (
        CROSSPLAY_MAXIMUM_MISSING_POLICY_REACH_FRACTION
    ),
    "independent_confirmation_panels": True,
    "held_out_metrics": ["mean_bb_per_100", "standard_error", "cvar_05_bb_per_100"],
    "restricted_exact_requires_independent_best_response_and_nashconv": True,
    "restricted_exact_nashconv_limit_chips": 5.0,
    "approximate_roots_claim_exact_exploitability": False,
    "minimum_complete_members_per_root": 6,
    "per_seat_pooled_confidence_lower_bound_floor_bb_per_100": -5.0,
    "slice_noninferiority_effective_stack_fraction_per_hand": 0.05,
    "slice_familywise_alpha": 0.05,
    "maximum_preregistered_qualification_looks": 8,
    "qualification_look_alpha_spending": (
        "bonferroni_across_eight_tier16_spaced_looks"
    ),
    "slice_multiplicity_policy": "bonferroni_one_sided_normal_lower_bound",
    "paired_delta_cvar_effective_stack_fraction_per_hand": 1.0,
    "bounded_deviation_reference": "candidate_two_player_self_play_on_paired_deals",
    "bounded_deviation_noninferiority_effective_stack_fraction_per_hand": 0.05,
    "bounded_deviation_familywise_alpha": 0.05,
    "bounded_response_policy_class": "fourteen_state_aware_preregistered_families",
    "bounded_response_is_exact_best_response": False,
    "bounded_response_gate_required_for_approximate_roots": True,
    "root_crossplay_required_for_approximate_roots": True,
    "crossplay_max_absolute_symmetric_advantage_effective_stack_fraction_per_hand": (
        CROSSPLAY_LIMIT_EFFECTIVE_STACK_FRACTION_PER_HAND
    ),
    "crossplay_pairing_unit": CROSSPLAY_PAIRING_UNIT,
    "crossplay_familywise_alpha": CROSSPLAY_FAMILYWISE_ALPHA,
    "runtime_minimum_cell_support": 8,
    "runtime_rare_cell_minimum_support": 3,
    "runtime_minimum_sampled_marginal_ess": 256.0,
    "restricted_projected_runtime_minimum_sampled_marginal_ess": 2.0,
    "declared_preconditioning_marginal_ess_is_separate_diagnostic": True,
    "runtime_minimum_observed_to_expected_histogram_ess_ratio": 0.50,
    "released_minimum_feature_hand_samples": 3_584,
    "runtime_minimum_feature_samples_per_player": 1_792,
    "feature_sample_weighting": (
        "on_distribution_joint_range_draws_have_unit_monte_carlo_weight"
    ),
}
EVALUATION_CONTRACT_SHA256 = sha256_json(EVALUATION_CONTRACT)


def _normalize(weights: Sequence[float]) -> tuple[float, ...]:
    positive = [max(0.0, float(value)) for value in weights]
    total = math.fsum(positive)
    if total <= 0:
        return tuple(1.0 / len(positive) for _ in positive)
    return tuple(value / total for value in positive)


@lru_cache(maxsize=8_192)
def _family_policy_cached(
    family: str,
    state: HoldemState,
    own: tuple[str, str],
    actor: int,
    actions: tuple[Action, ...],
) -> tuple[float, ...]:
    """Return one distinct state- and own-hand-aware opponent policy.

    These are preregistered stress policies, not claims about real opponent
    populations or exact best responses.  They may use the acting player's own
    cards and all public state, but never the other player's private cards.
    """

    if actor not in {0, 1} or state.actor != actor:
        raise ValueError("family policy actor does not match the decision state")
    features = combo_feature_record(state.board, own)
    made_name = str(features["made"])
    made_order = {
        "preflop_unmade": 0,
        "high_card": 0,
        "pair": 1,
        "two_pair": 2,
        "trips": 3,
        "straight": 4,
        "flush": 5,
        "full_house": 6,
        "quads": 7,
        "straight_flush": 8,
    }
    strength = made_order.get(made_name, 0) / 8.0
    draw = str(features["draw"]) not in {
        "no_live_draw",
        "not_applicable_river",
        "no_suited_seed",
    }
    blocker = str(features["blocker"]) not in {
        "no_declared_blocker",
        "preflop_rank_blockers",
    }
    to_call = max(0, state.current_bet - state.street_contributions[actor])
    pressure = min(3.0, to_call / max(1, state.pot))
    prior_aggression = sum(row.kind in {"bet", "raise"} for row in state.history)
    position_bonus = 0.15 if actor == state.button_player else 0.0
    aggressive_amounts = [
        row.amount for row in actions if row.kind in {"bet", "raise"} and row.label != "jam"
    ]
    weights = []
    for action in actions:
        aggressive = action.kind in {"bet", "raise"}
        jam = action.label == "jam"
        fold = action.kind == "fold"
        passive = action.kind in {"check", "call"}
        if family == "uniform_legal":
            weight = 1.0
        elif family == "balanced_control":
            if fold:
                weight = 0.45 + 1.30 * (1.0 - strength) * min(1.5, pressure + 0.2)
            elif passive:
                weight = 1.15 + 0.70 * strength + 0.35 * int(draw)
            else:
                weight = 0.55 + 1.65 * strength + 0.45 * int(draw) + position_bonus
        elif family == "tight_passive":
            weight = (2.2 + pressure) if fold else (2.0 + 0.4 * strength if passive else 0.25)
        elif family == "loose_passive":
            weight = 3.0 if action.kind == "call" else (1.8 if action.kind == "check" else 0.45 + 0.2 * int(draw))
        elif family == "tight_aggressive":
            weight = 0.25 + 3.0 * strength + 0.4 * int(blocker) if aggressive else (1.3 if fold else 0.45)
        elif family == "loose_aggressive":
            weight = 2.5 + 0.5 * int(draw) + 0.15 * prior_aggression if aggressive else 0.55
        elif family == "overfold":
            weight = 4.5 + pressure if fold else 0.65 + 0.2 * strength
        elif family == "underfold":
            weight = 0.03 if fold else 1.4 + 0.3 * strength
        elif family == "overcall":
            weight = 4.5 + 0.4 * (1.0 - strength) if action.kind == "call" else 0.6
        elif family == "underbluff":
            weight = 0.15 + 1.6 * strength if aggressive else 1.9
        elif family == "overbluff":
            weight = 3.4 + 0.8 * (1.0 - strength) + 0.4 * int(blocker) if aggressive else 0.5
        elif family == "small_size_heavy":
            weight = (
                2.5
                if aggressive_amounts and aggressive and not jam and action.amount == min(aggressive_amounts)
                else 0.8
            )
        elif family == "large_size_heavy":
            weight = (
                2.5
                if aggressive_amounts and aggressive and not jam and action.amount == max(aggressive_amounts)
                else 0.8
            )
        elif family == "jam_heavy":
            weight = 4.2 + 0.8 * strength + 0.4 * int(draw) if jam else 0.65
        elif family == "fold_call_control":
            weight = 2.2 + 0.2 * (1.0 - pressure) if action.kind in {"fold", "call", "check"} else 0.08
        elif family == "low_variance_control":
            weight = 3.2 + 0.2 * strength if action.kind in {"check", "call"} else (1.1 + pressure if fold else 0.03)
        else:
            raise ValueError(f"unknown opponent/control family: {family}")
        weights.append(weight)
    return _normalize(weights)


def _family_policy(
    family: str,
    state: HoldemState,
    deal: Deal,
    actor: int,
    actions: Sequence[Action],
) -> tuple[float, ...]:
    if actor not in {0, 1}:
        raise ValueError("invalid family-policy actor")
    own = deal.hole0 if actor == 0 else deal.hole1
    return _family_policy_cached(family, state, own, actor, tuple(actions))


@dataclass(frozen=True)
class PolicyAccessor:
    policy: Mapping[str, Mapping[str, float]]
    missing_reach_limit: float = float(
        EVALUATION_CONTRACT["maximum_missing_policy_reach_fraction"]
    )
    strategy_context: Mapping[str, Any] | None = None
    exact_information_sets: bool = False
    strategy_context_key: str | None = None

    def probabilities(
        self, state: HoldemState, deal: Deal, actor: int, actions: Sequence[Action]
    ) -> tuple[tuple[float, ...], bool]:
        labels = (
            tuple(action.label or action.kind for action in actions)
            if self.exact_information_sets
            else strategy_action_labels(state, actions)
        )
        key = (
            information_set_key(state, deal, actor)
            if self.exact_information_sets
            else strategy_information_set_key(
                state,
                deal,
                actor,
                self.strategy_context,
                actions=actions,
                action_labels=labels,
                context_key=self.strategy_context_key,
                deal_already_validated=True,
            )
        )
        row = self.policy.get(key)
        if row is None:
            return _family_policy("uniform_legal", state, deal, actor, actions), True
        if set(row) != set(labels):
            raise RuntimeError("candidate policy row does not match its legal action signature")
        weights = [float(row.get(label, 0.0)) for label in labels]
        return _normalize(weights), False


@dataclass(frozen=True)
class FamilyAccessor:
    family: str

    def probabilities(
        self, state: HoldemState, deal: Deal, actor: int, actions: Sequence[Action]
    ) -> tuple[tuple[float, ...], bool]:
        return _family_policy(self.family, state, deal, actor, actions), False


def _select(probabilities: Sequence[float], uniform: float) -> int:
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if uniform <= running:
            return index
    return len(probabilities) - 1


def _play(
    initial_state: HoldemState,
    deal: Deal,
    *,
    candidate_player: int,
    candidate: PolicyAccessor | FamilyAccessor,
    opponent_family: str,
    public_chance_stream: random.Random,
    action_stream: random.Random,
    depth_limit: str | None,
    continuation: ContinuationAuthority | None,
    strategy_context: Mapping[str, Any] | None,
) -> tuple[float, int, int]:
    state = initial_state
    decisions = 0
    missing = 0
    while True:
        if depth_limit is not None and state.street == depth_limit:
            if continuation is None:
                raise RuntimeError("depth-limited evaluation is missing continuation authority")
            return (
                continuation.resolve_preregistered_mixture(
                    state, deal, context=strategy_context
                )[candidate_player],
                decisions,
                missing,
            )
        if state.terminal:
            return terminal_utility(state, deal, candidate_player), decisions, missing
        if state.actor is None or state.all_in_runout:
            if state.street == "preflop":
                state = advance_flop(state, sample_flop(deal, public_chance_stream))
            else:
                cards = legal_next_cards(state.board, deal)
                state = advance_public_card(
                    state, cards[public_chance_stream.randrange(len(cards))]
                )
            continue
        actions = legal_actions(state)
        decisions += 1
        if state.actor == candidate_player:
            probabilities, fallback = candidate.probabilities(
                state, deal, state.actor, actions
            )
            missing += int(fallback)
        else:
            probabilities = _family_policy(
                opponent_family, state, deal, state.actor, actions
            )
        index = _select(probabilities, action_stream.random())
        state = apply_action(
            state,
            actions[index],
            prevalidated_legal_actions=actions,
        )


def _summary(values_chips: Sequence[float], *, big_blind: int = 100) -> dict[str, float]:
    if not values_chips:
        raise ValueError("evaluation panel is empty")
    bb100 = [value / big_blind * 100.0 for value in values_chips]
    mean = math.fsum(bb100) / len(bb100)
    variance = (
        math.fsum((value - mean) ** 2 for value in bb100) / max(1, len(bb100) - 1)
    )
    standard_error = math.sqrt(variance / len(bb100))
    tail_count = max(1, math.ceil(0.05 * len(bb100)))
    return {
        "sample_count": float(len(bb100)),
        "mean_bb_per_100": mean,
        "standard_error_bb_per_100": standard_error,
        "cvar_05_bb_per_100": math.fsum(sorted(bb100)[:tail_count]) / tail_count,
        "minimum_bb_per_100": min(bb100),
        "maximum_bb_per_100": max(bb100),
        "ci95_lower_bb_per_100": mean - 1.96 * standard_error,
        "ci95_upper_bb_per_100": mean + 1.96 * standard_error,
    }


def _play_profile(
    initial_state: HoldemState,
    deal: Deal,
    *,
    profile: PolicyAccessor,
    public_chance_stream: random.Random,
    action_stream: random.Random,
    depth_limit: str | None,
    continuation: ContinuationAuthority | None,
    strategy_context: Mapping[str, Any] | None,
) -> tuple[tuple[float, float], int, int]:
    state = initial_state
    decisions = 0
    missing = 0
    while True:
        if depth_limit is not None and state.street == depth_limit:
            if continuation is None:
                raise RuntimeError("depth-limited profile evaluation lacks continuation authority")
            pair = continuation.resolve_preregistered_mixture(
                state, deal, context=strategy_context
            )
            return pair, decisions, missing
        if state.terminal:
            pair = terminal_utilities(state, deal)
            return pair, decisions, missing
        if state.actor is None or state.all_in_runout:
            if state.street == "preflop":
                state = advance_flop(state, sample_flop(deal, public_chance_stream))
            else:
                cards = legal_next_cards(state.board, deal)
                state = advance_public_card(
                    state, cards[public_chance_stream.randrange(len(cards))]
                )
            continue
        actions = legal_actions(state)
        probabilities, fallback = profile.probabilities(
            state, deal, state.actor, actions
        )
        decisions += 1
        missing += int(fallback)
        state = apply_action(
            state,
            actions[_select(probabilities, action_stream.random())],
            prevalidated_legal_actions=actions,
        )


def _derived_seed(*parts: Any) -> int:
    return int(sha256_json(list(parts))[:16], 16)


def _deal_identity(deal: Deal) -> str:
    return sha256_json(
        {"hole0": sorted(deal.hole0), "hole1": sorted(deal.hole1)}
    )


def _sampler_marginals(
    sampler: Any,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    if hasattr(sampler, "sampled_marginals"):
        rows = sampler.sampled_marginals()
        return dict(rows[0]), dict(rows[1])
    if hasattr(sampler, "first") and hasattr(sampler, "second"):
        # Lightweight test samplers may expose only one already-projected
        # marginal per player. JointRangeSampler itself always takes the exact
        # blocker-conditioned branch above.
        return (
            dict(zip(sampler.first.combos, sampler.first.probabilities)),
            dict(zip(sampler.second.combos, sampler.second.probabilities)),
        )
    if hasattr(sampler, "rows"):
        rows = (defaultdict(float), defaultdict(float))
        for projected in sampler.rows:
            rows[0][projected.deal.hole0] += float(projected.probability)
            rows[1][projected.deal.hole1] += float(projected.probability)
        return dict(rows[0]), dict(rows[1])
    raise ValueError("evaluation sampler does not expose an auditable support distribution")


def _declared_preconditioning_sampler_marginals(
    sampler: Any,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]] | None:
    if hasattr(sampler, "declared_preconditioning_marginals"):
        rows = sampler.declared_preconditioning_marginals()
        return dict(rows[0]), dict(rows[1])
    if hasattr(sampler, "rows"):
        # A ProjectedDealSampler already represents the actual projected joint
        # law; it has no independent pre-compatibility player marginals.
        return None
    if hasattr(sampler, "first") and hasattr(sampler, "second"):
        return (
            dict(zip(sampler.first.combos, sampler.first.probabilities)),
            dict(zip(sampler.second.combos, sampler.second.probabilities)),
        )
    return None


def _marginal_identity(marginal: Mapping[tuple[str, str], float]) -> str:
    return sha256_json(
        [
            {"combo": list(combo), "probability": probability}
            for combo, probability in sorted(marginal.items())
        ]
    )


def _runtime_feature_coverage(
    *,
    board: Sequence[str],
    sampler: Any,
    histograms: Mapping[str, Counter[str]],
    indicator_histograms: Mapping[str, Counter[str]],
    restricted_projected_game: bool = False,
) -> dict[str, Any]:
    marginals = _sampler_marginals(sampler)
    declared_marginals = _declared_preconditioning_sampler_marginals(sampler)
    minimum = int(EVALUATION_CONTRACT["runtime_minimum_cell_support"])
    rare_minimum = int(EVALUATION_CONTRACT["runtime_rare_cell_minimum_support"])
    minimum_marginal_ess = float(
        EVALUATION_CONTRACT[
            "restricted_projected_runtime_minimum_sampled_marginal_ess"
            if restricted_projected_game
            else "runtime_minimum_sampled_marginal_ess"
        ]
    )
    histogram_ess_ratio = float(
        EVALUATION_CONTRACT[
            "runtime_minimum_observed_to_expected_histogram_ess_ratio"
        ]
    )
    minimum_player_samples = int(
        EVALUATION_CONTRACT["runtime_minimum_feature_samples_per_player"]
    )
    player_rows = []
    for player in (0, 1):
        name = f"player_{player}"
        observed = histograms[name]
        observed_indicators = indicator_histograms[name]
        sample_count = sum(observed.values())
        expected_cells: defaultdict[str, float] = defaultdict(float)
        expected_indicators: defaultdict[str, float] = defaultdict(float)
        for combo, probability in marginals[player].items():
            record = combo_feature_record(board, combo)
            signature = (
                f"made={record['made']}|draw={record['draw']}|"
                f"blocker={record['blocker']}"
            )
            expected_cells[signature] += probability * sample_count
            for indicator, enabled in record["indicators"].items():
                if enabled:
                    expected_indicators[indicator] += probability * sample_count
        marginal_ess = 1.0 / math.fsum(
            float(probability) ** 2 for probability in marginals[player].values()
        )
        declared_marginal_ess = (
            None
            if declared_marginals is None
            else 1.0
            / math.fsum(
                float(probability) ** 2
                for probability in declared_marginals[player].values()
            )
        )
        expected_histogram_ess = 1.0 / math.fsum(
            (expected / max(1, sample_count)) ** 2
            for expected in expected_cells.values()
        )
        minimum_histogram_ess = max(
            1.0, expected_histogram_ess * histogram_ess_ratio
        )
        mandatory_cells = {
            cell: expected
            for cell, expected in expected_cells.items()
            if expected >= 2.0 * minimum
        }
        rare_cells = {
            cell: expected
            for cell, expected in expected_cells.items()
            if minimum <= expected < 2.0 * minimum
        }
        mandatory_indicators = {
            feature: expected
            for feature, expected in expected_indicators.items()
            if expected >= 2.0 * minimum
        }
        histogram_ess = (
            float(sample_count * sample_count)
            / math.fsum(float(count * count) for count in observed.values())
            if sample_count and observed
            else 0.0
        )
        missing_mandatory = sorted(
            cell for cell in mandatory_cells if observed.get(cell, 0) < minimum
        )
        missing_rare = sorted(
            cell for cell in rare_cells if observed.get(cell, 0) < rare_minimum
        )
        missing_indicators = sorted(
            feature
            for feature in mandatory_indicators
            if observed_indicators.get(feature, 0) < minimum
        )
        gate = (
            sample_count >= minimum_player_samples
            and marginal_ess >= minimum_marginal_ess
            and histogram_ess >= minimum_histogram_ess
            and not missing_mandatory
            and not missing_rare
            and not missing_indicators
        )
        player_rows.append(
            {
                "player": player,
                "weighted_hand_sample_count": sample_count,
                "minimum_weighted_hand_sample_count": minimum_player_samples,
                "distribution_aware_histogram_ess": histogram_ess,
                "expected_distribution_histogram_ess": expected_histogram_ess,
                "minimum_histogram_ess": minimum_histogram_ess,
                "sampled_blocker_conditioned_marginal_combo_ess": marginal_ess,
                "minimum_sampled_marginal_combo_ess": minimum_marginal_ess,
                "sampled_marginal_distribution_sha256": _marginal_identity(
                    marginals[player]
                ),
                "declared_preconditioning_marginal_combo_ess": (
                    declared_marginal_ess
                ),
                "declared_preconditioning_marginal_distribution_sha256": (
                    None
                    if declared_marginals is None
                    else _marginal_identity(declared_marginals[player])
                ),
                "declared_preconditioning_marginal_is_diagnostic_only": True,
                "mandatory_combination_expected_support": dict(
                    sorted(mandatory_cells.items())
                ),
                "rare_combination_expected_support": dict(sorted(rare_cells.items())),
                "mandatory_indicator_expected_support": dict(
                    sorted(mandatory_indicators.items())
                ),
                "observed_combination_histogram": dict(sorted(observed.items())),
                "observed_indicator_histogram": dict(
                    sorted(observed_indicators.items())
                ),
                "missing_mandatory_combination_cells": missing_mandatory,
                "missing_rare_combination_cells": missing_rare,
                "missing_mandatory_indicator_cells": missing_indicators,
                "gate_pass": gate,
            }
        )
    return {
        "schema_version": "stage21_runtime_feature_sample_coverage_v3",
        "sampling_weight_contract": (
            "on_distribution_joint_range_draws_have_unit_monte_carlo_weight"
        ),
        "expected_support_source": (
            "exact_blocker_conditioned_sampled_player_marginals"
            if declared_marginals is not None
            else "exact_projected_deal_marginals"
        ),
        "sampled_marginal_joint_identity_sha256": sha256_json(
            [_marginal_identity(row) for row in marginals]
        ),
        "declared_preconditioning_marginal_joint_identity_sha256": (
            None
            if declared_marginals is None
            else sha256_json(
                [_marginal_identity(row) for row in declared_marginals]
            )
        ),
        "restricted_projected_game": restricted_projected_game,
        "minimum_weighted_cell_support": minimum,
        "rare_cell_minimum_weighted_support": rare_minimum,
        "minimum_weighted_hand_samples_total": EVALUATION_CONTRACT[
            "released_minimum_feature_hand_samples"
        ],
        "player_rows": player_rows,
        "static_registry_coverage_is_not_substituted_for_runtime_evidence": True,
        "gate_pass": all(row["gate_pass"] for row in player_rows),
    }


def _multiplicity_adjusted_lower_bound(
    summary: Mapping[str, Any], comparisons: int, *, familywise_alpha: float
) -> float:
    alpha = float(familywise_alpha)
    z = NormalDist().inv_cdf(1.0 - alpha / max(1, comparisons))
    return float(summary["mean_bb_per_100"]) - z * float(
        summary["standard_error_bb_per_100"]
    )


def _slice_lower_bound(summary: Mapping[str, Any], comparisons: int) -> float:
    return _multiplicity_adjusted_lower_bound(
        summary,
        comparisons,
        familywise_alpha=float(EVALUATION_CONTRACT["slice_familywise_alpha"]),
    )


def evaluate_policy_member(
    *,
    root_id: str,
    initial_state: HoldemState,
    joint_sampler: JointRangeSampler,
    policy: Mapping[str, Mapping[str, float]],
    panel_seed: int,
    deals_per_family_role: int,
    panel_role: str = "final_holdout",
    panel_index: int = 0,
    sampling_namespace: str | None = None,
    depth_limit: str | None = None,
    continuation: ContinuationAuthority | None = None,
    strategy_context: Mapping[str, Any] | None = None,
    exact_information_sets: bool = False,
    policy_sha256: str | None = None,
    qualification_look_index: int = 1,
    qualification_look_count: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if deals_per_family_role <= 0:
        raise ValueError("evaluation panel size must be positive")
    if panel_role not in EVALUATION_CONTRACT["required_panel_roles"]:
        raise ValueError("unregistered evaluation panel role")
    qualification_look_index = int(qualification_look_index)
    qualification_look_count = int(qualification_look_count)
    if not 1 <= qualification_look_index <= qualification_look_count:
        raise ValueError("qualification look index/count is invalid")
    if qualification_look_count > int(
        EVALUATION_CONTRACT["maximum_preregistered_qualification_looks"]
    ):
        raise ValueError("qualification look count exceeds the preregistered maximum")
    slice_familywise_alpha = float(
        EVALUATION_CONTRACT["slice_familywise_alpha"]
    ) / qualification_look_count
    bounded_familywise_alpha = float(
        EVALUATION_CONTRACT["bounded_deviation_familywise_alpha"]
    ) / qualification_look_count
    namespace = sampling_namespace or sha256_json(
        [root_id, panel_seed, panel_role, panel_index]
    )
    candidate = PolicyAccessor(
        policy,
        strategy_context=strategy_context,
        exact_information_sets=exact_information_sets,
        strategy_context_key=(
            "none" if strategy_context is None else sha256_json(strategy_context)[:16]
        ),
    )
    control_accessors = {
        control: FamilyAccessor(control) for control in SAFE_CONTROL_FAMILIES
    }
    family_rows: list[dict[str, Any]] = []
    all_values: list[float] = []
    all_control_values = {control: [] for control in SAFE_CONTROL_FAMILIES}
    all_control_deltas = {control: [] for control in SAFE_CONTROL_FAMILIES}
    seat_control_deltas = {
        player: {control: [] for control in SAFE_CONTROL_FAMILIES}
        for player in (0, 1)
    }
    total_decisions = 0
    total_missing = 0
    feature_histograms = {"player_0": Counter(), "player_1": Counter()}
    indicator_histograms = {"player_0": Counter(), "player_1": Counter()}
    deal_ledger: list[dict[str, Any]] = []
    last_progress_monotonic = time.monotonic()

    def draw(label: str) -> tuple[Deal, int, int]:
        deal_seed = _derived_seed(namespace, label, "private_deal")
        chance_seed = _derived_seed(namespace, label, "public_chance")
        action_seed = _derived_seed(namespace, label, "actions")
        deal = joint_sampler.sample(random.Random(deal_seed))
        deal_ledger.append(
            {
                "draw_id": sha256_json([namespace, label]),
                "deal_seed": deal_seed,
                "public_chance_seed": chance_seed,
                "action_seed": action_seed,
                "deal_identity_sha256": _deal_identity(deal),
            }
        )
        return deal, chance_seed, action_seed

    def observe_deal_features(deal: Deal) -> None:
        for player, hole in enumerate((deal.hole0, deal.hole1)):
            record = combo_feature_record(initial_state.board, hole)
            signature = (
                f"made={record['made']}|draw={record['draw']}|"
                f"blocker={record['blocker']}"
            )
            name = f"player_{player}"
            feature_histograms[name][signature] += 1
            for indicator, enabled in record["indicators"].items():
                if enabled:
                    indicator_histograms[name][indicator] += 1

    for family_index, family in enumerate(OPPONENT_FAMILIES):
        for candidate_player in (0, 1):
            values: list[float] = []
            control_values = {control: [] for control in SAFE_CONTROL_FAMILIES}
            candidate_self_play_values: list[float] = []
            for deal_index in range(deals_per_family_role):
                label = f"family={family_index}|seat={candidate_player}|deal={deal_index}"
                deal, chance_seed, action_seed = draw(label)
                observe_deal_features(deal)
                value, decisions, missing = _play(
                    initial_state,
                    deal,
                    candidate_player=candidate_player,
                    candidate=candidate,
                    opponent_family=family,
                    public_chance_stream=random.Random(chance_seed),
                    action_stream=random.Random(action_seed),
                    depth_limit=depth_limit,
                    continuation=continuation,
                    strategy_context=strategy_context,
                )
                values.append(value)
                total_decisions += decisions
                total_missing += missing
                self_play_pair, self_play_decisions, self_play_missing = _play_profile(
                    initial_state,
                    deal,
                    profile=candidate,
                    public_chance_stream=random.Random(chance_seed),
                    action_stream=random.Random(action_seed),
                    depth_limit=depth_limit,
                    continuation=continuation,
                    strategy_context=strategy_context,
                )
                candidate_self_play_values.append(
                    self_play_pair[candidate_player]
                )
                total_decisions += self_play_decisions
                total_missing += self_play_missing
                for control in SAFE_CONTROL_FAMILIES:
                    control_value, _, _ = _play(
                        initial_state,
                        deal,
                        candidate_player=candidate_player,
                        candidate=control_accessors[control],
                        opponent_family=family,
                        public_chance_stream=random.Random(chance_seed),
                        action_stream=random.Random(action_seed),
                        depth_limit=depth_limit,
                        continuation=continuation,
                        strategy_context=strategy_context,
                    )
                    control_values[control].append(control_value)
                if progress_callback is not None:
                    now = time.monotonic()
                    if (
                        deal_index + 1 == deals_per_family_role
                        or now - last_progress_monotonic >= 15.0
                    ):
                        progress_callback(
                            {
                                "completed_work_units": (
                                    (family_index * 2 + candidate_player)
                                    * deals_per_family_role
                                    + deal_index
                                    + 1
                                ),
                                "total_work_units": (
                                    len(OPPONENT_FAMILIES)
                                    * 2
                                    * deals_per_family_role
                                ),
                                "opponent_family": family,
                                "candidate_player": candidate_player,
                                "intra_slice_heartbeat": (
                                    deal_index + 1 < deals_per_family_role
                                ),
                            }
                        )
                        last_progress_monotonic = now
            all_values.extend(values)
            paired = {
                control: [
                    candidate_value - control_value
                    for candidate_value, control_value in zip(
                        values, control_values[control]
                    )
                ]
                for control in SAFE_CONTROL_FAMILIES
            }
            for control in SAFE_CONTROL_FAMILIES:
                all_control_values[control].extend(control_values[control])
                all_control_deltas[control].extend(paired[control])
                seat_control_deltas[candidate_player][control].extend(paired[control])
            family_rows.append(
                {
                    "opponent_family": family,
                    "candidate_player": candidate_player,
                    "summary": _summary(values),
                    "safe_controls": {
                        control: _summary(control_values[control])
                        for control in SAFE_CONTROL_FAMILIES
                    },
                    "paired_delta_by_safe_control": {
                        control: _summary(paired[control])
                        for control in SAFE_CONTROL_FAMILIES
                    },
                    "paired_delta_vs_candidate_self_play": _summary(
                        [
                            candidate_value - self_play_value
                            for candidate_value, self_play_value in zip(
                                values, candidate_self_play_values
                            )
                        ]
                    ),
                }
            )
    missing_fraction = total_missing / max(1, total_decisions)
    paired_overall = {
        control: _summary(all_control_deltas[control])
        for control in SAFE_CONTROL_FAMILIES
    }
    comparison_count = len(family_rows) * len(SAFE_CONTROL_FAMILIES)
    effective_stack_bb = min(initial_state.starting_stacks) / 100.0
    margin = -100.0 * effective_stack_bb * float(
        EVALUATION_CONTRACT[
            "slice_noninferiority_effective_stack_fraction_per_hand"
        ]
    )
    cvar_floor = -100.0 * effective_stack_bb * float(
        EVALUATION_CONTRACT[
            "paired_delta_cvar_effective_stack_fraction_per_hand"
        ]
    )
    seat_confidence_floor = float(
        EVALUATION_CONTRACT[
            "per_seat_pooled_confidence_lower_bound_floor_bb_per_100"
        ]
    )
    slice_rows = []
    for family_row in family_rows:
        for control, summary in family_row["paired_delta_by_safe_control"].items():
            lower = _multiplicity_adjusted_lower_bound(
                summary,
                comparison_count,
                familywise_alpha=slice_familywise_alpha,
            )
            slice_rows.append(
                {
                    "candidate_player": family_row["candidate_player"],
                    "opponent_family": family_row["opponent_family"],
                    "safe_control": control,
                    "summary": summary,
                    "multiplicity_adjusted_lower_bb_per_100": lower,
                    "noninferiority_margin_bb_per_100": margin,
                    "paired_delta_cvar_05_bb_per_100": summary[
                        "cvar_05_bb_per_100"
                    ],
                    "paired_delta_cvar_floor_bb_per_100": cvar_floor,
                    "gate_pass": lower >= margin
                    and float(summary["cvar_05_bb_per_100"]) >= cvar_floor,
                }
            )
    seat_rows = []
    for player in (0, 1):
        for control in SAFE_CONTROL_FAMILIES:
            summary = _summary(seat_control_deltas[player][control])
            lower = _multiplicity_adjusted_lower_bound(
                summary,
                len(SAFE_CONTROL_FAMILIES) * 2,
                familywise_alpha=slice_familywise_alpha,
            )
            seat_rows.append(
                {
                    "candidate_player": player,
                    "safe_control": control,
                    "summary": summary,
                    "multiplicity_adjusted_lower_bb_per_100": lower,
                    "pooled_confidence_lower_bound_floor_bb_per_100": (
                        seat_confidence_floor
                    ),
                    "paired_delta_cvar_floor_bb_per_100": cvar_floor,
                    "gate_pass": lower >= seat_confidence_floor
                    and float(summary["cvar_05_bb_per_100"]) >= cvar_floor,
                }
            )
    safe_noninferiority = all(row["gate_pass"] for row in (*slice_rows, *seat_rows))
    bounded_margin = -100.0 * effective_stack_bb * float(
        EVALUATION_CONTRACT[
            "bounded_deviation_noninferiority_effective_stack_fraction_per_hand"
        ]
    )
    bounded_comparison_count = len(family_rows)
    bounded_deviation_rows = []
    for family_row in family_rows:
        summary = family_row["paired_delta_vs_candidate_self_play"]
        lower = _multiplicity_adjusted_lower_bound(
            summary,
            bounded_comparison_count,
            familywise_alpha=bounded_familywise_alpha,
        )
        bounded_deviation_rows.append(
            {
                "candidate_player": family_row["candidate_player"],
                "opponent_family": family_row["opponent_family"],
                "summary": summary,
                "multiplicity_adjusted_lower_bb_per_100": lower,
                "noninferiority_margin_bb_per_100": bounded_margin,
                "paired_delta_cvar_floor_bb_per_100": cvar_floor,
                "gate_pass": lower >= bounded_margin
                and float(summary["cvar_05_bb_per_100"]) >= cvar_floor,
            }
        )
    bounded_response = {
        str(candidate_player): min(
            (
                row
                for row in family_rows
                if row["candidate_player"] == candidate_player
            ),
            key=lambda row: row["paired_delta_vs_candidate_self_play"][
                "mean_bb_per_100"
            ],
        )
        for candidate_player in (0, 1)
    }
    bounded_response_gate = all(
        row["gate_pass"] for row in bounded_deviation_rows
    )

    profile_values: list[list[float]] = [[], []]
    profile_decisions = 0
    profile_missing = 0
    profile_zero_sum_error = 0.0
    for deal_index in range(deals_per_family_role):
        deal, chance_seed, action_seed = draw(f"self_profile|deal={deal_index}")
        observe_deal_features(deal)
        pair, decisions, missing = _play_profile(
            initial_state,
            deal,
            profile=candidate,
            public_chance_stream=random.Random(chance_seed),
            action_stream=random.Random(action_seed),
            depth_limit=depth_limit,
            continuation=continuation,
            strategy_context=strategy_context,
        )
        profile_values[0].append(pair[0])
        profile_values[1].append(pair[1])
        profile_decisions += decisions
        profile_missing += missing
        profile_zero_sum_error = max(profile_zero_sum_error, abs(sum(pair)))
    profile_missing_fraction = profile_missing / max(1, profile_decisions)
    feature_coverage = _runtime_feature_coverage(
        board=initial_state.board,
        sampler=joint_sampler,
        histograms=feature_histograms,
        indicator_histograms=indicator_histograms,
        restricted_projected_game=exact_information_sets,
    )
    policy_sha = sha256_json(policy) if policy_sha256 is None else policy_sha256
    panel_identity = sha256_json(
        {
            "root_id": root_id,
            "panel_role": panel_role,
            "panel_index": panel_index,
            "panel_seed": panel_seed,
            "sampling_namespace": namespace,
            "policy_sha256": policy_sha,
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
        }
    )
    return with_self_hash(
        {
            "schema_version": "stage21_both_role_heldout_evaluation_v5",
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
            "root_id": root_id,
            "policy_sha256": policy_sha,
            "panel_role": panel_role,
            "panel_index": panel_index,
            "panel_seed": panel_seed,
            "sampling_namespace": namespace,
            "panel_identity_sha256": panel_identity,
            "independent_confirmation_panel": True,
            "physical_deal_duplicates_are_valid_iid_draws_not_panel_overlap": True,
            "draw_identity_ledger_sha256": sha256_json(
                [row["draw_id"] for row in deal_ledger]
            ),
            "physical_deal_sequence_sha256": sha256_json(
                [row["deal_identity_sha256"] for row in deal_ledger]
            ),
            "rng_stream_ledger_sha256": sha256_json(
                [
                    {
                        key: row[key]
                        for key in (
                            "draw_id",
                            "deal_seed",
                            "public_chance_seed",
                            "action_seed",
                        )
                    }
                    for row in deal_ledger
                ]
            ),
            "draw_count": len(deal_ledger),
            "deals_per_family_role": deals_per_family_role,
            "qualification_look_index": qualification_look_index,
            "qualification_look_count": qualification_look_count,
            "slice_familywise_alpha_after_look_spending": (
                slice_familywise_alpha
            ),
            "bounded_deviation_familywise_alpha_after_look_spending": (
                bounded_familywise_alpha
            ),
            "effective_stack_bb": effective_stack_bb,
            "family_role_count": len(family_rows),
            "family_role_rows": family_rows,
            "overall_summary": _summary(all_values),
            "overall_safe_controls": {
                control: _summary(all_control_values[control])
                for control in SAFE_CONTROL_FAMILIES
            },
            "paired_delta_overall_by_safe_control": paired_overall,
            "per_seat_safe_control_rows": seat_rows,
            "per_family_seat_safe_control_rows": slice_rows,
            "per_family_seat_bounded_deviation_rows": bounded_deviation_rows,
            "bounded_deviation_comparison_count": bounded_comparison_count,
            "bounded_deviation_reference": EVALUATION_CONTRACT[
                "bounded_deviation_reference"
            ],
            "slice_multiplicity_policy": EVALUATION_CONTRACT[
                "slice_multiplicity_policy"
            ],
            "slice_comparison_count": comparison_count,
            "safe_control_noninferiority_pass": safe_noninferiority,
            "missing_policy_reach_fraction": missing_fraction,
            "missing_policy_reach_limit": candidate.missing_reach_limit,
            "policy_reach_complete": missing_fraction <= candidate.missing_reach_limit,
            "bounded_policy_class_approximate_best_response": {
                "policy_class": EVALUATION_CONTRACT[
                    "bounded_response_policy_class"
                ],
                "opponent_family_count": len(OPPONENT_FAMILIES),
                "by_candidate_player": bounded_response,
                "per_family_seat_rows": bounded_deviation_rows,
                "gate_pass": bounded_response_gate,
                "is_exact_best_response": False,
                "claim_scope": (
                    "minimum_over_preregistered_finite_state_aware_policy_class_only"
                ),
            },
            "finite_opponent_family_response_stress": bounded_response,
            "finite_opponent_family_response_is_exact_best_response": False,
            "made_hand_draw_blocker_sample_coverage": feature_coverage,
            "full_two_player_candidate_profile_self_play": {
                "player_0_summary": _summary(profile_values[0]),
                "player_1_summary": _summary(profile_values[1]),
                "per_sample_max_zero_sum_error_chips": profile_zero_sum_error,
                "policy_missing_reach_fraction": profile_missing_fraction,
                "policy_missing_reach_limit": candidate.missing_reach_limit,
                "policy_reach_complete": profile_missing_fraction
                <= candidate.missing_reach_limit,
            },
            "exact_best_response_claim": False,
            "exact_nashconv_claim": False,
        }
    )


def evaluate_policy_panels(
    *,
    root_id: str,
    initial_state: HoldemState,
    joint_sampler: JointRangeSampler,
    policy: Mapping[str, Mapping[str, float]],
    base_seed: int,
    panel_count: int,
    deals_per_family_role: int,
    final_deals_per_family_role: int,
    optimization_stream_identity_sha256: str,
    depth_limit: str | None = None,
    continuation: ContinuationAuthority | None = None,
    strategy_context: Mapping[str, Any] | None = None,
    exact_information_sets: bool = False,
    expected_policy_sha256: str | None = None,
    qualification_look_index: int = 1,
    qualification_look_count: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    roles = tuple(EVALUATION_CONTRACT["required_panel_roles"])
    if panel_count != len(roles):
        raise ValueError("released qualification requires exactly four held-out panels")
    qualification_look_index = int(qualification_look_index)
    qualification_look_count = int(qualification_look_count)
    if not 1 <= qualification_look_index <= qualification_look_count:
        raise ValueError("qualification look index/count is invalid")
    policy_sha = (
        sha256_json(policy)
        if expected_policy_sha256 is None
        else str(expected_policy_sha256)
    )
    if len(policy_sha) != 64:
        raise ValueError("solver/evaluation policy identity is malformed")
    panels = []
    panel_deal_counts = [
        final_deals_per_family_role if role == "final_holdout" else deals_per_family_role
        for role in roles
    ]
    panel_work_units = [
        len(OPPONENT_FAMILIES) * 2 * count for count in panel_deal_counts
    ]
    total_work_units = sum(panel_work_units)
    completed_before_panel = 0
    for panel_index, role in enumerate(roles):
        namespace = sha256_json(
            [
                root_id,
                base_seed,
                role,
                panel_index,
                qualification_look_index,
                qualification_look_count,
                "heldout_panel_namespace_v2",
            ]
        )
        panel_seed = _derived_seed(namespace, "panel_seed")
        def emit_member_progress(
            row: Mapping[str, Any],
            *,
            offset: int = completed_before_panel,
            current_panel: int = panel_index,
            current_role: str = role,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        **dict(row),
                        "panel_index": current_panel,
                        "panel_role": current_role,
                        "completed_evaluation_work_units": (
                            offset + int(row["completed_work_units"])
                        ),
                        "total_evaluation_work_units": total_work_units,
                    }
                )

        panels.append(
            evaluate_policy_member(
                root_id=root_id,
                initial_state=initial_state,
                joint_sampler=joint_sampler,
                policy=policy,
                panel_seed=panel_seed,
                deals_per_family_role=(
                    final_deals_per_family_role
                    if role == "final_holdout"
                    else deals_per_family_role
                ),
                panel_role=role,
                panel_index=panel_index,
                sampling_namespace=namespace,
                depth_limit=depth_limit,
                continuation=continuation,
                strategy_context=strategy_context,
                exact_information_sets=exact_information_sets,
                policy_sha256=policy_sha,
                qualification_look_index=qualification_look_index,
                qualification_look_count=qualification_look_count,
                progress_callback=emit_member_progress,
            )
        )
        completed_before_panel += panel_work_units[panel_index]
    panel_ids = [row["panel_identity_sha256"] for row in panels]
    namespaces = [row["sampling_namespace"] for row in panels]
    draw_ledgers = [row["draw_identity_ledger_sha256"] for row in panels]
    rng_ledgers = [row["rng_stream_ledger_sha256"] for row in panels]
    disjoint = all(
        len(values) == len(set(values))
        for values in (panel_ids, namespaces, draw_ledgers, rng_ledgers)
    ) and optimization_stream_identity_sha256 not in set(rng_ledgers)
    return with_self_hash(
        {
            "schema_version": "stage21_identity_bound_evaluation_panel_bundle_v2",
            "evaluation_contract_sha256": EVALUATION_CONTRACT_SHA256,
            "root_id": root_id,
            "policy_sha256": policy_sha,
            "optimization_stream_identity_sha256": (
                optimization_stream_identity_sha256
            ),
            "panel_count": len(panels),
            "qualification_look_index": qualification_look_index,
            "qualification_look_count": qualification_look_count,
            "required_panel_roles": list(roles),
            "panel_roles": [row["panel_role"] for row in panels],
            "panel_identity_sha256s": panel_ids,
            "sampling_namespaces": namespaces,
            "draw_identity_ledger_sha256s": draw_ledgers,
            "rng_stream_ledger_sha256s": rng_ledgers,
            "panel_identity_and_sampling_namespace_disjoint": disjoint,
            "independent_confirmation_panel": disjoint,
            "panels": panels,
        }
    )


def _paired_seat_balanced_advantages(
    *,
    first_seat0: Sequence[float],
    second_seat1: Sequence[float],
    first_seat1: Sequence[float],
    second_seat0: Sequence[float],
) -> list[float]:
    lengths = {
        len(first_seat0), len(second_seat1), len(first_seat1), len(second_seat0)
    }
    if len(lengths) != 1 or not first_seat0:
        raise ValueError("cross-play seat assignments must be nonempty paired deals")
    return [
        math.fsum(
            (
                first_seat0[index] - second_seat1[index],
                first_seat1[index] - second_seat0[index],
            )
        )
        / 2.0
        for index in range(len(first_seat0))
    ]


def evaluate_policy_crossplay(
    *,
    root_id: str,
    initial_state: HoldemState,
    joint_sampler: JointRangeSampler,
    first_policy: Mapping[str, Mapping[str, float]],
    second_policy: Mapping[str, Mapping[str, float]],
    first_member: Mapping[str, Any],
    second_member: Mapping[str, Any],
    panel_seed: int,
    deals_per_assignment: int,
    depth_limit: str | None = None,
    continuation: ContinuationAuthority | None = None,
    strategy_context: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if deals_per_assignment <= 0:
        raise ValueError("cross-play panel must be nonempty")
    strategy_context_key = (
        "none" if strategy_context is None else sha256_json(strategy_context)[:16]
    )
    accessors = (
        PolicyAccessor(
            first_policy,
            strategy_context=strategy_context,
            strategy_context_key=strategy_context_key,
        ),
        PolicyAccessor(
            second_policy,
            strategy_context=strategy_context,
            strategy_context_key=strategy_context_key,
        ),
    )
    observations = {
        deal_index: {"deal_index": deal_index, "assignments": []}
        for deal_index in range(deals_per_assignment)
    }
    progress_started = time.monotonic()
    last_progress = 0.0
    completed_assignments = 0
    total_assignments = 2 * deals_per_assignment

    def emit_progress(*, force: bool = False) -> None:
        nonlocal last_progress
        if progress_callback is None:
            return
        now = time.monotonic()
        if not force and now - last_progress < 60.0:
            return
        progress_callback(
            {
                "completed_crossplay_assignments": completed_assignments,
                "total_crossplay_assignments": total_assignments,
                "elapsed_seconds": now - progress_started,
            }
        )
        last_progress = now

    emit_progress(force=True)
    # Both seat assignments are contractually paired on the same private deal
    # and RNG stream identities. Materialize those immutable bindings once;
    # resampling them inside each assignment repeated hashes, range sampling,
    # and Random construction without adding independent evidence.
    paired_inputs = []
    for deal_index in range(deals_per_assignment):
        namespace = [root_id, panel_seed, deal_index, "crossplay_pair_v3"]
        deal_seed = _derived_seed(*namespace, "private_deal")
        chance_seed = _derived_seed(*namespace, "public_chance")
        action_seed = _derived_seed(*namespace, "actions")
        deal = joint_sampler.sample(random.Random(deal_seed))
        paired_inputs.append(
            (
                deal_index,
                sha256_json(namespace),
                deal_seed,
                chance_seed,
                action_seed,
                deal,
            )
        )
    for assignment in (0, 1):
        # assignment=0 puts first member in seat 0; assignment=1 puts it in seat 1.
        seat_profiles = accessors if assignment == 0 else accessors[::-1]
        for (
            deal_index,
            paired_deal_identity,
            deal_seed,
            chance_seed,
            action_seed,
            deal,
        ) in paired_inputs:
            # Reuse the exact private deal and base random-number streams for
            # both seat assignments.  The policies may visit different public
            # branches, but this paired construction removes avoidable deal
            # noise from the member/seat comparison.
            state = initial_state
            local_missing = [0, 0]
            local_decisions = [0, 0]
            chance_rng = random.Random(chance_seed)
            action_rng = random.Random(action_seed)
            while True:
                if depth_limit is not None and state.street == depth_limit:
                    if continuation is None:
                        raise RuntimeError("cross-play depth limit lacks continuation")
                    pair = continuation.resolve_preregistered_mixture(
                        state, deal, context=strategy_context
                    )
                    break
                if state.terminal:
                    pair = terminal_utilities(state, deal)
                    break
                if state.actor is None or state.all_in_runout:
                    if state.street == "preflop":
                        state = advance_flop(state, sample_flop(deal, chance_rng))
                    else:
                        cards = legal_next_cards(state.board, deal)
                        state = advance_public_card(
                            state, cards[chance_rng.randrange(len(cards))]
                        )
                    continue
                actor = state.actor
                actions = legal_actions(state)
                probabilities, fallback = seat_profiles[actor].probabilities(
                    state, deal, actor, actions
                )
                local_decisions[actor] += 1
                local_missing[actor] += int(fallback)
                state = apply_action(
                    state,
                    actions[_select(probabilities, action_rng.random())],
                    prevalidated_legal_actions=actions,
                )
            first_seat = assignment
            second_seat = 1 - assignment
            observations[deal_index]["assignments"].append(
                {
                    "assignment": assignment,
                    "first_member_seat": first_seat,
                    "second_member_seat": second_seat,
                    "paired_deal_identity_sha256": paired_deal_identity,
                    "physical_deal_identity_sha256": _deal_identity(deal),
                    "deal_seed": deal_seed,
                    "public_chance_seed": chance_seed,
                    "action_seed": action_seed,
                    "first_member_value_chips": pair[first_seat],
                    "second_member_value_chips": pair[second_seat],
                    "first_member_decision_count": local_decisions[first_seat],
                    "second_member_decision_count": local_decisions[second_seat],
                    "first_member_missing_policy_count": local_missing[first_seat],
                    "second_member_missing_policy_count": local_missing[second_seat],
                }
            )
            completed_assignments += 1
            emit_progress()
    emit_progress(force=True)
    return build_crossplay_pair_evidence(
        root_id=root_id,
        first_member=first_member,
        second_member=second_member,
        deals_per_assignment=deals_per_assignment,
        raw_observations=[observations[index] for index in sorted(observations)],
        pairing_unit=EVALUATION_CONTRACT["crossplay_pairing_unit"],
    )


def adjudicate_lane(
    *,
    root_kind: str,
    solver_result: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    exact_oracle_result: Mapping[str, Any] | None = None,
    required_terminal_iterations: int,
    continuation_diagnostics: Mapping[str, Any] | None = None,
    runtime_feature_board: Sequence[str] | None = None,
    runtime_feature_sampler: Any | None = None,
    runtime_feature_restricted_projected_game: bool = False,
) -> dict[str, Any]:
    panels = evaluation_result.get("panels", ())
    valid_panels = [row for row in panels if isinstance(row, Mapping)]
    expected_roles = list(EVALUATION_CONTRACT["required_panel_roles"])

    def every_panel(predicate) -> bool:
        return len(valid_panels) == len(expected_roles) and all(
            predicate(panel) for panel in valid_panels
        )

    def _expected_panel_thresholds(panel: Mapping[str, Any]) -> tuple[float, float, float]:
        effective_stack_bb = float(panel.get("effective_stack_bb", math.nan))
        slice_margin = -100.0 * effective_stack_bb * float(
            EVALUATION_CONTRACT[
                "slice_noninferiority_effective_stack_fraction_per_hand"
            ]
        )
        bounded_margin = -100.0 * effective_stack_bb * float(
            EVALUATION_CONTRACT[
                "bounded_deviation_noninferiority_effective_stack_fraction_per_hand"
            ]
        )
        cvar_floor = -100.0 * effective_stack_bb * float(
            EVALUATION_CONTRACT[
                "paired_delta_cvar_effective_stack_fraction_per_hand"
            ]
        )
        return slice_margin, bounded_margin, cvar_floor

    def valid_qualification_look_binding(panel: Mapping[str, Any]) -> bool:
        look_index = int(panel.get("qualification_look_index", -1))
        look_count = int(panel.get("qualification_look_count", -1))
        if not 1 <= look_index <= look_count <= int(
            EVALUATION_CONTRACT["maximum_preregistered_qualification_looks"]
        ):
            return False
        expected_slice_alpha = float(
            EVALUATION_CONTRACT["slice_familywise_alpha"]
        ) / look_count
        expected_bounded_alpha = float(
            EVALUATION_CONTRACT["bounded_deviation_familywise_alpha"]
        ) / look_count
        return math.isclose(
            float(
                panel.get(
                    "slice_familywise_alpha_after_look_spending", math.nan
                )
            ),
            expected_slice_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) and math.isclose(
            float(
                panel.get(
                    "bounded_deviation_familywise_alpha_after_look_spending",
                    math.nan,
                )
            ),
            expected_bounded_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )

    def valid_slice_row(
        row: Mapping[str, Any], panel: Mapping[str, Any]
    ) -> bool:
        summary = row.get("summary", {})
        expected_count = int(panel.get("deals_per_family_role", -1))
        expected_lower = _multiplicity_adjusted_lower_bound(
            summary,
            len(OPPONENT_FAMILIES) * 2 * len(SAFE_CONTROL_FAMILIES),
            familywise_alpha=float(
                panel.get(
                    "slice_familywise_alpha_after_look_spending", math.nan
                )
            ),
        )
        expected_margin, _, expected_cvar_floor = _expected_panel_thresholds(panel)
        values = (
            float(summary.get("mean_bb_per_100", math.nan)),
            float(summary.get("standard_error_bb_per_100", math.nan)),
            float(summary.get("cvar_05_bb_per_100", math.nan)),
            float(summary.get("sample_count", math.nan)),
            float(row.get("multiplicity_adjusted_lower_bb_per_100", math.nan)),
            float(row.get("noninferiority_margin_bb_per_100", math.nan)),
            float(row.get("paired_delta_cvar_05_bb_per_100", math.nan)),
            float(row.get("paired_delta_cvar_floor_bb_per_100", math.nan)),
        )
        return (
            all(math.isfinite(value) for value in values)
            and values[1] >= 0.0
            and expected_count > 0
            and values[3] == expected_count
            and math.isclose(values[4], expected_lower, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(values[5], expected_margin, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                values[7], expected_cvar_floor, rel_tol=0.0, abs_tol=1e-9
            )
            and math.isclose(values[2], values[6], rel_tol=0.0, abs_tol=1e-9)
            and values[4] >= values[5]
            and values[6] >= values[7]
            and row.get("gate_pass") is True
        )

    def valid_seat_row(
        row: Mapping[str, Any], panel: Mapping[str, Any]
    ) -> bool:
        summary = row.get("summary", {})
        expected_count = int(panel.get("deals_per_family_role", -1)) * len(
            OPPONENT_FAMILIES
        )
        expected_lower = _multiplicity_adjusted_lower_bound(
            summary,
            len(SAFE_CONTROL_FAMILIES) * 2,
            familywise_alpha=float(
                panel.get(
                    "slice_familywise_alpha_after_look_spending", math.nan
                )
            ),
        )
        _, _, expected_cvar_floor = _expected_panel_thresholds(panel)
        expected_confidence_floor = float(
            EVALUATION_CONTRACT[
                "per_seat_pooled_confidence_lower_bound_floor_bb_per_100"
            ]
        )
        values = (
            float(summary.get("mean_bb_per_100", math.nan)),
            float(summary.get("standard_error_bb_per_100", math.nan)),
            float(summary.get("cvar_05_bb_per_100", math.nan)),
            float(summary.get("sample_count", math.nan)),
            float(row.get("multiplicity_adjusted_lower_bb_per_100", math.nan)),
            float(
                row.get(
                    "pooled_confidence_lower_bound_floor_bb_per_100", math.nan
                )
            ),
            float(row.get("paired_delta_cvar_floor_bb_per_100", math.nan)),
        )
        return (
            all(math.isfinite(value) for value in values)
            and values[1] >= 0.0
            and expected_count > 0
            and values[3] == expected_count
            and math.isclose(values[4], expected_lower, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                values[5], expected_confidence_floor, rel_tol=0.0, abs_tol=1e-9
            )
            and math.isclose(
                values[6], expected_cvar_floor, rel_tol=0.0, abs_tol=1e-9
            )
            and values[4] >= values[5]
            and values[2] >= values[6]
            and row.get("gate_pass") is True
        )

    def valid_bounded_deviation_row(
        row: Mapping[str, Any], panel: Mapping[str, Any]
    ) -> bool:
        summary = row.get("summary", {})
        expected_count = int(panel.get("deals_per_family_role", -1))
        expected_lower = _multiplicity_adjusted_lower_bound(
            summary,
            len(OPPONENT_FAMILIES) * 2,
            familywise_alpha=float(
                panel.get(
                    "bounded_deviation_familywise_alpha_after_look_spending",
                    math.nan,
                )
            ),
        )
        _, expected_margin, expected_cvar_floor = _expected_panel_thresholds(panel)
        values = (
            float(summary.get("mean_bb_per_100", math.nan)),
            float(summary.get("standard_error_bb_per_100", math.nan)),
            float(summary.get("cvar_05_bb_per_100", math.nan)),
            float(summary.get("sample_count", math.nan)),
            float(row.get("multiplicity_adjusted_lower_bb_per_100", math.nan)),
            float(row.get("noninferiority_margin_bb_per_100", math.nan)),
            float(row.get("paired_delta_cvar_floor_bb_per_100", math.nan)),
        )
        return (
            all(math.isfinite(value) for value in values)
            and values[1] >= 0.0
            and expected_count > 0
            and values[3] == expected_count
            and math.isclose(values[4], expected_lower, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(values[5], expected_margin, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                values[6], expected_cvar_floor, rel_tol=0.0, abs_tol=1e-9
            )
            and values[4] >= values[5]
            and values[2] >= values[6]
            and row.get("gate_pass") is True
        )

    def valid_runtime_feature_coverage(coverage: Mapping[str, Any]) -> bool:
        rows = coverage.get("player_rows", ())
        if (
            runtime_feature_board is None
            or runtime_feature_sampler is None
            or coverage.get("gate_pass") is not True
            or len(rows) != 2
            or {row.get("player") for row in rows} != {0, 1}
        ):
            return False
        try:
            recomputed = _runtime_feature_coverage(
                board=runtime_feature_board,
                sampler=runtime_feature_sampler,
                histograms={
                    f"player_{int(row['player'])}": Counter(
                        {
                            str(key): int(value)
                            for key, value in row.get(
                                "observed_combination_histogram", {}
                            ).items()
                        }
                    )
                    for row in rows
                },
                indicator_histograms={
                    f"player_{int(row['player'])}": Counter(
                        {
                            str(key): int(value)
                            for key, value in row.get(
                                "observed_indicator_histogram", {}
                            ).items()
                        }
                    )
                    for row in rows
                },
                restricted_projected_game=(
                    runtime_feature_restricted_projected_game
                ),
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
        if dict(coverage) != recomputed:
            return False
        for row in rows:
            sample_count = int(row.get("weighted_hand_sample_count", -1))
            sample_minimum = int(row.get("minimum_weighted_hand_sample_count", -1))
            ess = float(row.get("distribution_aware_histogram_ess", math.nan))
            ess_minimum = float(row.get("minimum_histogram_ess", math.nan))
            marginal_ess = float(
                row.get(
                    "sampled_blocker_conditioned_marginal_combo_ess", math.nan
                )
            )
            marginal_ess_minimum = float(
                row.get("minimum_sampled_marginal_combo_ess", math.nan)
            )
            if (
                row.get("gate_pass") is not True
                or sample_count < sample_minimum
                or not math.isfinite(ess)
                or not math.isfinite(ess_minimum)
                or ess < ess_minimum
                or not math.isfinite(marginal_ess)
                or not math.isfinite(marginal_ess_minimum)
                or marginal_ess < marginal_ess_minimum
                or row.get("missing_mandatory_combination_cells")
                or row.get("missing_rare_combination_cells")
                or row.get("missing_mandatory_indicator_cells")
            ):
                return False
        return True

    def valid_panel_identity_bundle() -> bool:
        fields = (
            ("panel_identity_sha256s", "panel_identity_sha256"),
            ("sampling_namespaces", "sampling_namespace"),
            ("draw_identity_ledger_sha256s", "draw_identity_ledger_sha256"),
            ("rng_stream_ledger_sha256s", "rng_stream_ledger_sha256"),
        )
        for bundle_field, panel_field in fields:
            values = evaluation_result.get(bundle_field, ())
            if (
                len(values) != len(expected_roles)
                or len(set(values)) != len(expected_roles)
                or list(values) != [panel.get(panel_field) for panel in valid_panels]
            ):
                return False
        return (
            evaluation_result.get("panel_identity_and_sampling_namespace_disjoint")
            is True
            and evaluation_result.get("optimization_stream_identity_sha256")
            == solver_result.get("optimization_stream_identity_sha256")
            and solver_result.get("optimization_stream_identity_sha256")
            not in set(evaluation_result.get("rng_stream_ledger_sha256s", ()))
        )

    checks = {
        "fresh_regrets": solver_result.get("fresh_regrets") is True,
        "fresh_strategy_sum": solver_result.get("fresh_strategy_sum") is True,
        "on_policy_sampling": solver_result.get("on_policy_sampling") is True,
        "importance_ratio_one": solver_result.get("importance_ratio") == 1.0,
        "nonzero_solver_state": int(solver_result.get("node_count", 0)) > 0,
        "released_terminal_iterations_complete": int(
            solver_result.get("iterations", -1)
        )
        >= int(required_terminal_iterations),
        "four_identity_bound_heldout_panels": evaluation_result.get("panel_count")
        == len(expected_roles)
        and evaluation_result.get("panel_roles") == expected_roles,
        "preregistered_qualification_look_alpha_spending": (
            int(evaluation_result.get("qualification_look_index", -1)) >= 1
            and int(evaluation_result.get("qualification_look_count", -1)) >= int(
                evaluation_result.get("qualification_look_index", -1)
            )
            and every_panel(valid_qualification_look_binding)
            and every_panel(
                lambda panel: panel.get("qualification_look_index")
                == evaluation_result.get("qualification_look_index")
                and panel.get("qualification_look_count")
                == evaluation_result.get("qualification_look_count")
            )
        ),
        "optimization_and_heldout_streams_disjoint": valid_panel_identity_bundle(),
        "panel_policy_and_root_identity_bound": every_panel(
            lambda panel: panel.get("root_id") == solver_result.get("root_id")
            and panel.get("policy_sha256") == solver_result.get("policy_sha256")
            and panel.get("evaluation_contract_sha256")
            == EVALUATION_CONTRACT_SHA256
        ),
        "complete_both_role_opponent_panels": every_panel(
            lambda panel: panel.get("family_role_count")
            == len(OPPONENT_FAMILIES) * 2
        ),
        "policy_reach_complete": every_panel(
            lambda panel: panel.get("policy_reach_complete") is True
            and math.isfinite(
                float(panel.get("missing_policy_reach_fraction", math.nan))
            )
            and 0.0
            <= float(panel.get("missing_policy_reach_fraction", math.nan))
            <= float(EVALUATION_CONTRACT["maximum_missing_policy_reach_fraction"])
            and math.isclose(
                float(panel.get("missing_policy_reach_limit", math.nan)),
                float(EVALUATION_CONTRACT["maximum_missing_policy_reach_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "safe_controls_complete": every_panel(
            lambda panel: set(
                panel.get("paired_delta_overall_by_safe_control", {})
            )
            == set(SAFE_CONTROL_FAMILIES)
        ),
        "per_seat_per_family_tail_safe_control_noninferiority": every_panel(
            lambda panel: panel.get("safe_control_noninferiority_pass") is True
            and math.isfinite(float(panel.get("effective_stack_bb", math.nan)))
            and float(panel.get("effective_stack_bb", math.nan)) > 0.0
            and len(panel.get("per_seat_safe_control_rows", ()))
            == len(SAFE_CONTROL_FAMILIES) * 2
            and len(panel.get("per_family_seat_safe_control_rows", ()))
            == len(OPPONENT_FAMILIES) * 2 * len(SAFE_CONTROL_FAMILIES)
            and all(
                valid_seat_row(row, panel)
                for row in panel.get("per_seat_safe_control_rows", ())
            )
            and all(
                valid_slice_row(row, panel)
                for row in panel.get("per_family_seat_safe_control_rows", ())
            )
        ),
        "full_candidate_profile_zero_sum": every_panel(
            lambda panel: float(
                panel.get("full_two_player_candidate_profile_self_play", {}).get(
                    "per_sample_max_zero_sum_error_chips", math.inf
                )
            )
            <= 1e-9
        ),
        "full_candidate_profile_policy_reach_complete": every_panel(
            lambda panel: (
                panel.get("full_two_player_candidate_profile_self_play", {}).get(
                    "policy_reach_complete"
                )
                is True
                and math.isfinite(
                    float(
                        panel.get(
                            "full_two_player_candidate_profile_self_play", {}
                        ).get("policy_missing_reach_fraction", math.nan)
                    )
                )
                and 0.0
                <= float(
                    panel.get(
                        "full_two_player_candidate_profile_self_play", {}
                    ).get("policy_missing_reach_fraction", math.nan)
                )
                <= float(
                    EVALUATION_CONTRACT["maximum_missing_policy_reach_fraction"]
                )
                and math.isclose(
                    float(
                        panel.get(
                            "full_two_player_candidate_profile_self_play", {}
                        ).get("policy_missing_reach_limit", math.nan)
                    ),
                    float(
                        EVALUATION_CONTRACT[
                            "maximum_missing_policy_reach_fraction"
                        ]
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ),
        "finite_heldout_metrics": every_panel(
            lambda panel: all(
                math.isfinite(float(panel.get("overall_summary", {}).get(key, math.nan)))
                for key in (
                    "mean_bb_per_100",
                    "standard_error_bb_per_100",
                    "cvar_05_bb_per_100",
                )
            )
        ),
        "runtime_made_hand_draw_blocker_sample_coverage": every_panel(
            lambda panel: valid_runtime_feature_coverage(
                panel.get("made_hand_draw_blocker_sample_coverage", {})
            )
        ),
    }
    if root_kind == "approximate":
        checks["bounded_policy_class_approximate_best_response"] = every_panel(
            lambda panel: panel.get(
                "bounded_policy_class_approximate_best_response", {}
            ).get("gate_pass")
            is True
            and panel.get(
                "bounded_policy_class_approximate_best_response", {}
            ).get("is_exact_best_response")
            is False
            and panel.get("bounded_deviation_reference")
            == EVALUATION_CONTRACT["bounded_deviation_reference"]
            and panel.get("bounded_deviation_comparison_count")
            == len(OPPONENT_FAMILIES) * 2
            and len(panel.get("per_family_seat_bounded_deviation_rows", ()))
            == len(OPPONENT_FAMILIES) * 2
            and all(
                valid_bounded_deviation_row(row, panel)
                for row in panel.get(
                    "per_family_seat_bounded_deviation_rows", ()
                )
            )
        )
    if continuation_diagnostics is not None:
        checks["continuation_all_backoff_and_fallback_bounds"] = (
            continuation_diagnostics.get("within_all_backoff_bounds") is True
        )
    if root_kind == "restricted_exact":
        checks.update(
            {
                "independent_exact_oracle_present": exact_oracle_result is not None,
                "restricted_scope_bound": exact_oracle_result is not None
                and exact_oracle_result.get("claim_scope") == "restricted_projected_game_only",
                "nashconv_finite": exact_oracle_result is not None
                and math.isfinite(float(exact_oracle_result.get("nashconv", math.nan))),
                "nashconv_within_restricted_limit": exact_oracle_result is not None
                and float(exact_oracle_result.get("nashconv", math.inf))
                <= float(EVALUATION_CONTRACT["restricted_exact_nashconv_limit_chips"]),
                "restricted_profile_zero_sum": exact_oracle_result is not None
                and float(exact_oracle_result.get("zero_sum_error", math.inf)) <= 1e-9,
            }
        )
    accepted = all(checks.values())
    return with_self_hash(
        {
            "schema_version": "stage21_lane_adjudication_v5",
            "root_id": solver_result.get("root_id"),
            "algorithm": solver_result.get("algorithm"),
            "seed": solver_result.get("seed"),
            "checks": checks,
            "artifact_hash_valid": True,
            "independent_confirmation_panel": checks[
                "four_identity_bound_heldout_panels"
            ]
            and checks["optimization_and_heldout_streams_disjoint"],
            "qualified": accepted,
            "claim_scope": "restricted_projected_game_only"
            if root_kind == "restricted_exact"
            else "bounded_empirical_no_exact_exploitability_claim",
            "required_terminal_iterations": int(required_terminal_iterations),
        }
    )
