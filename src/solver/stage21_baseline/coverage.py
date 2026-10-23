from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .cards import (
    hand_features,
    legal_hole_combos,
    parse_card,
    preflop_class,
    straight_completion_ranks,
)
from .game import HoldemState, state_from_root
from .ranges import (
    JointRangeSampler,
    RangeDistribution,
    build_player_range,
    combo_range_features,
    unnormalized_combo_weight,
)


MADE_CATEGORY_NAMES = {
    0: "high_card",
    1: "pair",
    2: "two_pair",
    3: "trips",
    4: "straight",
    5: "flush",
    6: "full_house",
    7: "quads",
    8: "straight_flush",
}
MINIMUM_RANGE_DISTRIBUTION_ESS = 256.0
MINIMUM_REQUIRED_FEATURE_EFFECTIVE_SUPPORT = 8.0
MINIMUM_QUALIFIED_COMBINATION_EFFECTIVE_SUPPORT = 8.0


def effective_sample_size(probabilities: Sequence[float]) -> float:
    denominator = math.fsum(float(value) ** 2 for value in probabilities)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("invalid probability vector for ESS")
    return 1.0 / denominator


def _nut_flush_blocker(board: Sequence[str], hole: Sequence[str]) -> bool:
    if not board:
        return False
    suit_counts: Counter[int] = Counter(parse_card(card)[1] for card in board)
    maximum = max(suit_counts.values())
    candidate_suits = {suit for suit, count in suit_counts.items() if count == maximum}
    return maximum >= 2 and any(
        parse_card(card)[0] == 14 and parse_card(card)[1] in candidate_suits
        for card in hole
    )


def _straight_blocker(board: Sequence[str], hole: Sequence[str]) -> bool:
    completion = set(straight_completion_ranks(board)) if board else set()
    return bool(completion & {parse_card(card)[0] for card in hole})


@lru_cache(maxsize=65_536)
def _combo_feature_record_cached(
    board: tuple[str, ...], hole: tuple[str, ...]
) -> dict[str, Any]:
    hand_class = preflop_class(hole)
    if not board:
        pair = len(hand_class) == 2
        suited = hand_class.endswith("s")
        return {
            "made": "preflop_unmade",
            "draw": "suited_seed" if suited else "no_suited_seed",
            "blocker": "preflop_rank_blockers",
            "indicators": {
                "pocket_pair": pair,
                "suited": suited,
                "offsuit": hand_class.endswith("o"),
            },
        }

    features = hand_features(board, hole)
    category = int(features["made_category"])
    if len(board) == 5:
        draw = "not_applicable_river"
    elif features["flush_draw"] and (
        features["open_ended_straight_draw"] or features["gutshot"]
    ):
        draw = "combo_draw"
    elif features["flush_draw"]:
        draw = "flush_draw"
    elif features["open_ended_straight_draw"]:
        draw = "open_ended"
    elif features["gutshot"]:
        draw = "gutshot"
    else:
        draw = "no_live_draw"
    blockers = []
    if _nut_flush_blocker(board, hole):
        blockers.append("nut_flush_blocker")
    if _straight_blocker(board, hole):
        blockers.append("straight_completion_blocker")
    if int(features["overcards"]) > 0:
        blockers.append("overcard_blocker")
    blocker = "+".join(blockers) if blockers else "no_declared_blocker"
    return {
        "made": MADE_CATEGORY_NAMES[category],
        "draw": draw,
        "blocker": blocker,
        "indicators": {
            "live_gutshot": len(board) < 5 and bool(features["gutshot"]),
            "live_open_ended": len(board) < 5
            and bool(features["open_ended_straight_draw"]),
            "live_flush_draw": len(board) < 5 and bool(features["flush_draw"]),
            "live_combo_draw": len(board) < 5
            and bool(features["flush_draw"])
            and bool(
                features["open_ended_straight_draw"] or features["gutshot"]
            ),
            "made_two_pair_or_better": category >= 2,
            "made_straight_or_better": category >= 4,
            "made_full_house_or_better": category >= 6,
            "nut_flush_blocker": _nut_flush_blocker(board, hole),
            "straight_completion_blocker": _straight_blocker(board, hole),
            "overcard_blocker": int(features["overcards"]) > 0,
        },
    }


def combo_feature_record(board: Sequence[str], hole: Sequence[str]) -> dict[str, Any]:
    """Return an isolated copy of a cached deterministic card-feature record."""

    cached = _combo_feature_record_cached(tuple(board), tuple(hole))
    return {**cached, "indicators": dict(cached["indicators"])}


@lru_cache(maxsize=512)
def _representative_preflop_class_rows(
    board: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, str], int, tuple[int, ...]], ...]:
    groups: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for combo in legal_hole_combos(board):
        groups[preflop_class(combo)].append(combo)
    rows = []
    for hand_class, combos in sorted(groups.items()):
        representative = combos[len(combos) // 2]
        made_rank = combo_range_features(board, representative)[4]
        rows.append((hand_class, representative, len(combos), made_rank))
    if not rows:
        raise AssertionError("postflop legal class inventory is empty")
    return tuple(rows)


def _class_mass_profile(
    root: Mapping[str, Any], state: HoldemState, player: int
) -> tuple[dict[str, float], dict[str, tuple[int, ...]]]:
    raw = {}
    ranks = {}
    for hand_class, representative, multiplicity, made_rank in (
        _representative_preflop_class_rows(tuple(state.board))
    ):
        raw[hand_class] = (
            unnormalized_combo_weight(root, state, player, representative)
            * multiplicity
        )
        ranks[hand_class] = made_rank
    total = math.fsum(raw.values())
    return ({name: value / total for name, value in raw.items()}, ranks)


def weighted_nut_advantage_transition(
    root: Mapping[str, Any],
    child_state: HoldemState,
    parent_board: Sequence[str],
    *,
    elite_fraction: float = 0.05,
) -> dict[str, Any]:
    """Measure a transparent fixed-posterior elite-combo nut-advantage proxy.

    This is a runout classifier, not an equilibrium theorem.  The same causal
    action-history model is held fixed on parent and child; only card removal
    and board-relative hand strength change.  A reversal requires a strict sign
    crossing in player-0 minus player-1 top-five-percent combo mass.
    """

    parent_street = "flop" if len(parent_board) == 3 else "turn"
    parent_state = replace(
        child_state,
        street=parent_street,
        board=tuple(parent_board),
    )
    parent_profiles = [
        _class_mass_profile(root, parent_state, player) for player in (0, 1)
    ]
    child_profiles = [
        _class_mass_profile(root, child_state, player) for player in (0, 1)
    ]

    def elite_classes(
        ranks: Mapping[str, tuple[int, ...]]
    ) -> frozenset[str]:
        count = max(1, math.ceil(len(ranks) * elite_fraction))
        ordered = sorted(ranks, key=lambda name: (ranks[name], name), reverse=True)
        return frozenset(ordered[:count])

    parent_elite = elite_classes(parent_profiles[0][1])
    child_elite = elite_classes(child_profiles[0][1])
    parent_mass = [
        math.fsum(profile[name] for name in parent_elite)
        for profile, _ in parent_profiles
    ]
    child_mass = [
        math.fsum(profile[name] for name in child_elite)
        for profile, _ in child_profiles
    ]
    transition_rows = _representative_preflop_class_rows(tuple(child_state.board))
    straight_completion_mass = []
    flush_completion_mass = []
    for player in (0, 1):
        weighted_rows = []
        for _, representative, multiplicity, _ in transition_rows:
            child_weight = (
                unnormalized_combo_weight(root, child_state, player, representative)
                * multiplicity
            )
            parent_features = combo_range_features(parent_board, representative)
            child_features = combo_range_features(child_state.board, representative)
            weighted_rows.append(
                (
                    child_weight,
                    parent_features[0] < 4 <= child_features[0],
                    parent_features[0] < 5 <= child_features[0],
                )
            )
        total_weight = math.fsum(row[0] for row in weighted_rows)
        straight_completion_mass.append(
            math.fsum(weight for weight, straight, _ in weighted_rows if straight)
            / total_weight
        )
        flush_completion_mass.append(
            math.fsum(weight for weight, _, flush in weighted_rows if flush)
            / total_weight
        )
    parent_advantage = parent_mass[0] - parent_mass[1]
    child_advantage = child_mass[0] - child_mass[1]
    reversal = parent_advantage * child_advantage < 0.0
    return {
        "schema_version": "stage21_range_weighted_nut_advantage_transition_v1",
        "method": (
            "fixed_action_posterior_top_five_percent_preflop_class_mass_proxy"
        ),
        "class_representative_rule": (
            "median_lexicographic_legal_combo_with_exact_board-conditioned_"
            "physical_class_multiplicity"
        ),
        "elite_class_fraction": elite_fraction,
        "parent_elite_class_count": len(parent_elite),
        "child_elite_class_count": len(child_elite),
        "parent_player_mass": parent_mass,
        "child_player_mass": child_mass,
        "parent_player0_minus_player1": parent_advantage,
        "child_player0_minus_player1": child_advantage,
        "nut_advantage_reversal": reversal,
        "range_straight_completion_mass_by_player": straight_completion_mass,
        "range_flush_completion_mass_by_player": flush_completion_mass,
        "minimum_mass_for_range_transition_label": 0.02,
        "range_straight_completion": max(straight_completion_mass) >= 0.02,
        "range_flush_completion": max(flush_completion_mass) >= 0.02,
        "claim_boundary": (
            "transparent class-stratified representative-root runout proxy; not exact equity, "
            "exploitability, or a universal nut-advantage theorem"
        ),
    }


def _required_features(stage: str) -> tuple[str, ...]:
    if stage in {"flop", "turn"}:
        return (
            "live_gutshot",
            "live_open_ended",
            "live_flush_draw",
            "live_combo_draw",
            "made_two_pair_or_better",
            "made_straight_or_better",
            "nut_flush_blocker",
            "overcard_blocker",
        )
    if stage == "river":
        return (
            "made_two_pair_or_better",
            "made_straight_or_better",
            "made_full_house_or_better",
            "nut_flush_blocker",
            "overcard_blocker",
        )
    return ("pocket_pair", "suited", "offsuit")


def registry_range_feature_coverage(
    stage: str, roots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not roots:
        raise ValueError("coverage registry is empty")
    combination_mass: defaultdict[str, float] = defaultdict(float)
    combination_effective_support: defaultdict[str, float] = defaultdict(float)
    combination_combo_support: Counter[str] = Counter()
    feature_mass: defaultdict[str, float] = defaultdict(float)
    feature_effective_support: defaultdict[str, float] = defaultdict(float)
    feature_combo_support: Counter[str] = Counter()
    profile_rows = []

    for root in roots:
        state = state_from_root(root)
        distributions = [
            build_player_range(root, state, player) for player in (0, 1)
        ]
        sampler = JointRangeSampler(distributions[0], distributions[1], state.board)
        sampled_marginals = sampler.sampled_marginals()
        declared_marginals = sampler.declared_preconditioning_marginals()
        root_profile_rows = []
        for player in (0, 1):
            distribution = distributions[player]
            declared_ess = effective_sample_size(distribution.probabilities)
            sampled_ess = effective_sample_size(
                tuple(sampled_marginals[player].values())
            )
            profile_feature_mass: defaultdict[str, float] = defaultdict(float)
            for combo, probability in sampled_marginals[player].items():
                features = combo_feature_record(state.board, combo)
                signature = (
                    f"made={features['made']}|draw={features['draw']}|"
                    f"blocker={features['blocker']}"
                )
                combination_mass[signature] += probability
                combination_effective_support[signature] += probability * sampled_ess
                combination_combo_support[signature] += 1
                for name, enabled in features["indicators"].items():
                    if enabled:
                        feature_mass[name] += probability
                        feature_effective_support[name] += probability * sampled_ess
                        feature_combo_support[name] += 1
                        profile_feature_mass[name] += probability
            root_profile_rows.append(
                {
                    "root_id": root["root_id"],
                    "player": player,
                    "range_identity_sha256": distribution.identity_sha256,
                    "combo_count": len(distribution.combos),
                    "effective_sample_size": sampled_ess,
                    "sampled_blocker_conditioned_marginal_effective_sample_size": (
                        sampled_ess
                    ),
                    "declared_preconditioning_marginal_effective_sample_size": (
                        declared_ess
                    ),
                    "minimum_ess_pass": (
                        sampled_ess >= MINIMUM_RANGE_DISTRIBUTION_ESS
                    ),
                    "required_feature_probability_mass": {
                        name: profile_feature_mass.get(name, 0.0)
                        for name in _required_features(stage)
                    },
                    "declared_preconditioning_range_identity_sha256": (
                        distributions[player].identity_sha256
                    ),
                    "declared_preconditioning_marginal_joint_identity_sha256": (
                        sampler.declared_preconditioning_marginal_identity_sha256
                    ),
                    "sampled_blocker_conditioned_marginal_joint_identity_sha256": (
                        sampler.sampled_blocker_conditioned_marginal_identity_sha256
                    ),
                    "declared_vs_sampled_marginal_total_variation": 0.5
                    * math.fsum(
                        abs(
                            declared_marginals[player].get(combo, 0.0)
                            - sampled_marginals[player].get(combo, 0.0)
                        )
                        for combo in set(declared_marginals[player])
                        | set(sampled_marginals[player])
                    ),
                }
            )
        profile_rows.extend(root_profile_rows)

    profile_count = len(profile_rows)
    combination_histogram = {
        name: {
            "mean_weighted_probability": combination_mass[name] / profile_count,
            "weighted_effective_support": combination_effective_support[name],
            "combo_support_across_root_player_profiles": combination_combo_support[name],
            "qualified_cell": combination_effective_support[name]
            >= MINIMUM_QUALIFIED_COMBINATION_EFFECTIVE_SUPPORT,
        }
        for name in sorted(combination_mass)
    }
    required = _required_features(stage)
    feature_histogram = {
        name: {
            "mean_weighted_probability": feature_mass.get(name, 0.0) / profile_count,
            "weighted_effective_support": feature_effective_support.get(name, 0.0),
            "combo_support_across_root_player_profiles": feature_combo_support.get(name, 0),
            "minimum_weighted_effective_support": (
                MINIMUM_REQUIRED_FEATURE_EFFECTIVE_SUPPORT
            ),
            "gate_pass": feature_effective_support.get(name, 0.0)
            >= MINIMUM_REQUIRED_FEATURE_EFFECTIVE_SUPPORT,
        }
        for name in required
    }
    minimum_ess = min(float(row["effective_sample_size"]) for row in profile_rows)
    low_support_cells = [
        name
        for name, row in combination_histogram.items()
        if row["qualified_cell"] is not True
    ]
    return {
        "schema_version": "stage21_range_feature_coverage_v3",
        "stage": stage,
        "root_player_profile_count": profile_count,
        "range_probability_model": (
            "exact sampled blocker-conditioned player marginal derived from the "
            "normalized compatible two-player joint action-posterior law"
        ),
        "effective_sample_size_definition": "1/sum(p_combo^2)",
        "minimum_range_distribution_ess": MINIMUM_RANGE_DISTRIBUTION_ESS,
        "minimum_observed_range_distribution_ess": minimum_ess,
        "every_root_player_range_ess_gate_pass": all(
            row["minimum_ess_pass"] for row in profile_rows
        ),
        "declared_and_sampled_blocker_conditioned_marginal_identities_receipted": all(
            row.get("declared_preconditioning_marginal_joint_identity_sha256")
            and row.get("sampled_blocker_conditioned_marginal_joint_identity_sha256")
            for row in profile_rows
        ),
        "minimum_required_feature_effective_support": (
            MINIMUM_REQUIRED_FEATURE_EFFECTIVE_SUPPORT
        ),
        "required_feature_histogram": feature_histogram,
        "made_hand_draw_blocker_combination_histogram": combination_histogram,
        "minimum_qualified_combination_effective_support": (
            MINIMUM_QUALIFIED_COMBINATION_EFFECTIVE_SUPPORT
        ),
        "qualified_combination_cell_count": sum(
            row["qualified_cell"] is True
            for row in combination_histogram.values()
        ),
        "low_support_diagnostic_cells": low_support_cells,
        "low_support_cells_are_not_claimed_as_covered": True,
        "per_root_player_profiles": profile_rows,
        "coverage_gate_pass": (
            minimum_ess >= MINIMUM_RANGE_DISTRIBUTION_ESS
            and all(row["gate_pass"] for row in feature_histogram.values())
        ),
    }
