from __future__ import annotations

import bisect
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .cards import DECK, hand_features, legal_hole_combos, parse_card, preflop_class
from .common import sha256_json
from .game import Deal, HoldemState


SUPPORTED_RANGE_FAMILIES = frozenset(
    {
        "equilibrium_like",
        "balanced",
        "balanced_control",
        "linear",
        "value_heavy",
        "bluff_heavy",
        "capped",
        "uncapped",
        "polarized",
        "merged",
    }
)


@dataclass(frozen=True)
class RangeDistribution:
    player: int
    combos: tuple[tuple[str, str], ...]
    probabilities: tuple[float, ...]
    model_spec: Mapping[str, Any]
    identity_sha256: str

    def validate(self, board: Sequence[str]) -> None:
        if self.player not in {0, 1}:
            raise ValueError("invalid player")
        if len(self.combos) != len(self.probabilities) or not self.combos:
            raise ValueError("range vectors are empty or misaligned")
        if any(probability <= 0 or not math.isfinite(probability) for probability in self.probabilities):
            raise ValueError("range probabilities must be finite and positive")
        if abs(sum(self.probabilities) - 1.0) > 1e-12:
            raise ValueError("range probabilities do not sum to one")
        legal = set(legal_hole_combos(board))
        if any(combo not in legal for combo in self.combos):
            raise ValueError("range contains a board collision")
        expected = sha256_json(
            {
                "player": self.player,
                "combos": [list(combo) for combo in self.combos],
                "probabilities": list(self.probabilities),
                "model_spec": self.model_spec,
            }
        )
        if expected != self.identity_sha256:
            raise ValueError("range identity mismatch")


def _class_strength(hand_class: str) -> float:
    rank_order = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
    high = rank_order[hand_class[0]]
    low = rank_order[hand_class[1]]
    pair = len(hand_class) == 2
    suited = hand_class.endswith("s")
    gap = high - low
    score = (high + low) / 28.0
    if pair:
        score += 0.28 + high / 70.0
    if suited:
        score += 0.08
    if not pair:
        score += max(0.0, 0.08 - 0.018 * max(gap - 1, 0))
    return min(max(score, 0.01), 1.35)


def _family_multiplier(family: str, strength: float, made_category: int, draw: bool) -> float:
    normalized_preflop_strength = min(1.0, strength / 1.35)
    strong = min(1.0, 0.70 * made_category / 8.0 + 0.30 * normalized_preflop_strength)
    weak = 1.0 - min(1.0, strong)
    middle = 1.0 - abs(strong - 0.5) * 2.0
    if family in {"equilibrium_like", "balanced", "balanced_control"}:
        return 1.0
    if family == "linear":
        return 0.35 + 1.9 * strong
    if family == "value_heavy":
        return 0.30 + 2.4 * strong
    if family == "bluff_heavy":
        return 0.25 + 0.55 * strong + 2.10 * weak + (0.45 if draw else 0.0)
    if family == "capped":
        return 0.35 + 1.6 * middle + 0.25 * weak
    if family == "uncapped":
        return 0.55 + 1.5 * strong + (0.35 if draw else 0.0)
    if family == "polarized":
        return 0.35 + 1.5 * max(strong, weak) + (0.25 if draw else 0.0)
    if family == "merged":
        return 0.45 + 1.25 * (0.6 * strong + 0.4 * middle)
    raise ValueError(f"unsupported range family: {family}")


def _history_multiplier(
    root: Mapping[str, Any],
    state: HoldemState,
    player: int,
    made_category: int,
    draw: bool,
    strength: float,
) -> float:
    history = str(root.get("action_history_family", "fresh_street"))
    initiative = str(root.get("initiative_role", "none"))
    position = str(root.get("position", ""))
    normalized_preflop_strength = min(1.0, strength / 1.35)
    strong_score = min(
        1.0,
        0.70 * made_category / 8.0 + 0.30 * normalized_preflop_strength,
    )
    weak_score = 1.0 - strong_score
    middle_score = max(0.0, 1.0 - abs(strong_score - 0.5) * 2.0)
    multiplier = 1.0
    aggressive_words = ("bet", "raise", "barrel", "jam", "lead", "donk")
    passive_words = ("call", "check")
    role_text = f"{history} {initiative}".lower()
    player_is_hero = player == int(root.get("hero_player", 0))
    aggressors = [
        row.player for row in state.history if row.kind in {"bet", "raise"}
    ]
    callers = [row.player for row in state.history if row.kind == "call"]
    last_aggressor = aggressors[-1] if aggressors else None
    last_caller = callers[-1] if callers else None
    if player == last_aggressor:
        multiplier *= 0.55 + 1.35 * strong_score + (0.25 if draw else 0.0)
    elif player == last_caller:
        multiplier *= 0.60 + 0.95 * middle_score + (0.30 if draw else 0.0)
    if any(word in role_text for word in aggressive_words):
        if player_is_hero:
            multiplier *= 0.62 + 1.20 * strong_score + (0.30 if draw else 0.0)
        else:
            multiplier *= 0.78 + 0.65 * strong_score + (0.18 if draw else 0.0)
    elif any(word in role_text for word in passive_words):
        multiplier *= 0.82 + 0.60 * middle_score + (0.25 if draw else 0.0)

    position_text = position.lower()
    hero_is_ip = "hero_btn_sb_ip" in position_text or "sb_btn_vs_bb" in position_text
    player_is_ip = hero_is_ip if player_is_hero else not hero_is_ip
    if player_is_ip:
        multiplier *= 0.90 + 0.16 * weak_score + (0.10 if draw else 0.0)
    else:
        multiplier *= 0.88 + 0.24 * strong_score + (0.06 if draw else 0.0)
    return multiplier


def _range_family(root: Mapping[str, Any], player: int) -> str:
    contract = root.get("range_contract", {})
    explicit = contract.get(f"player_{player}_family")
    if explicit:
        return str(explicit)
    if player == int(root.get("hero_player", 0)):
        return str(root.get("range_family", "equilibrium_like"))
    return str(root.get("opponent_range_family", "equilibrium_like"))


@lru_cache(maxsize=16_384)
def _cached_hand_features(
    board: tuple[str, ...], combo: tuple[str, str]
) -> tuple[int, bool, float, str, tuple[int, ...]]:
    hand_class = preflop_class(combo)
    if board:
        features = hand_features(board, combo)
        made = int(features["made_category"])
        made_rank = tuple(int(value) for value in features["made_rank"])
        draw = bool(
            features["flush_draw"]
            or features["open_ended_straight_draw"]
            or features["gutshot"]
        )
    else:
        made = 0
        made_rank = ()
        draw = hand_class.endswith("s")
    return made, draw, _class_strength(hand_class), hand_class, made_rank


def combo_range_features(
    board: Sequence[str], combo: tuple[str, str]
) -> tuple[int, bool, float, str, tuple[int, ...]]:
    return _cached_hand_features(tuple(board), combo)


def unnormalized_combo_weight(
    root: Mapping[str, Any],
    state: HoldemState,
    player: int,
    combo: tuple[str, str],
) -> float:
    """Return the exact pre-normalization weight used by a player range."""

    family = _range_family(root, player)
    if family not in SUPPORTED_RANGE_FAMILIES:
        raise ValueError(f"unsupported declared range family: {family}")
    target_classes = tuple(
        str(value) for value in (root.get("preflop_hand_class_targets") or ())
    )
    if not target_classes and root.get("preflop_hand_class_target"):
        target_classes = (str(root["preflop_hand_class_target"]),)
    contract = root.get("range_contract", {})
    target_multiplier = float(contract.get("target_class_bias_multiplier", 1.25))
    made, draw, strength, hand_class, _ = _cached_hand_features(state.board, combo)
    weight = _family_multiplier(family, strength, made, draw)
    weight *= _history_multiplier(root, state, player, made, draw, strength)
    if hand_class in target_classes:
        weight *= target_multiplier
    if state.board:
        board_ranks = {parse_card(card)[0] for card in state.board}
        hole_ranks = {parse_card(card)[0] for card in combo}
        if max(hole_ranks) > max(board_ranks):
            weight *= 1.03
        suit_counts: dict[int, int] = {}
        for card in state.board:
            _, suit = parse_card(card)
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        dominant = max(suit_counts, key=suit_counts.get)
        if suit_counts[dominant] >= 3 and any(
            parse_card(card) == (14, dominant) for card in combo
        ):
            weight *= 1.08
    return max(weight, 1e-15)


def build_player_range(
    root: Mapping[str, Any], state: HoldemState, player: int
) -> RangeDistribution:
    if player not in {0, 1}:
        raise ValueError("invalid player")
    family = _range_family(root, player)
    if family not in SUPPORTED_RANGE_FAMILIES:
        raise ValueError(f"unsupported declared range family: {family}")
    target_classes = tuple(
        str(value) for value in (root.get("preflop_hand_class_targets") or ())
    )
    if not target_classes and root.get("preflop_hand_class_target"):
        target_classes = (str(root["preflop_hand_class_target"]),)
    if len(target_classes) != len(set(target_classes)):
        raise ValueError("duplicate preflop target classes")
    contract = root.get("range_contract", {})
    if contract.get("target_class_is_point_hand") is True:
        raise ValueError("point-hand target contracts are forbidden")
    combos = legal_hole_combos(state.board)
    raw: list[float] = []
    for combo in combos:
        raw.append(unnormalized_combo_weight(root, state, player, combo))
    total = math.fsum(raw)
    probabilities = tuple(value / total for value in raw)
    spec = {
        "schema_version": "stage21_player_specific_action_posterior_v3",
        "player": player,
        "family": family,
        "board_relative_made_draw_features": bool(state.board),
        "action_history_family": root.get("action_history_family"),
        "executable_action_history_conditioned": True,
        "initiative_role": root.get("initiative_role"),
        "position": root.get("position"),
        "position_conditioned_with_hand_varying_likelihood": True,
        "target_classes": list(target_classes),
        "target_class_count": len(target_classes),
        "target_class_weighting": contract.get("target_class_weighting"),
        "target_class_is_stratification_not_point_hand": True,
        "blocker_conditioned": True,
        "claim": "transparent_preregistered_posterior_model_not_learned_ground_truth",
    }
    identity = sha256_json(
        {
            "player": player,
            "combos": [list(combo) for combo in combos],
            "probabilities": list(probabilities),
            "model_spec": spec,
        }
    )
    distribution = RangeDistribution(player, combos, probabilities, spec, identity)
    distribution.validate(state.board)
    return distribution


class JointRangeSampler:
    def __init__(self, first: RangeDistribution, second: RangeDistribution, board: Sequence[str]):
        if first.player != 0 or second.player != 1:
            raise ValueError("joint sampler requires ordered player 0/player 1 ranges")
        first.validate(board)
        second.validate(board)
        if first.combos == second.combos and max(
            abs(first.probabilities[index] - second.probabilities[index])
            for index in range(len(first.combos))
        ) <= 1e-15:
            raise ValueError("player-specific ranges cannot be the same normalized distribution")
        self.first = first
        self.second = second
        self.board = tuple(board)
        first_card_mass: dict[str, float] = {}
        first_combo_mass: dict[tuple[str, str], float] = {}
        for combo, probability in zip(first.combos, first.probabilities):
            first_combo_mass[tuple(sorted(combo))] = probability
            for card in combo:
                first_card_mass[card] = first_card_mass.get(card, 0.0) + probability
        second_card_mass: dict[str, float] = {}
        second_combo_mass: dict[tuple[str, str], float] = {}
        for combo, probability in zip(second.combos, second.probabilities):
            second_combo_mass[tuple(sorted(combo))] = probability
            for card in combo:
                second_card_mass[card] = second_card_mass.get(card, 0.0) + probability
        first_adjusted = []
        conditional_mass = []
        for combo, probability in zip(first.combos, first.probabilities):
            illegal = (
                second_card_mass.get(combo[0], 0.0)
                + second_card_mass.get(combo[1], 0.0)
                - second_combo_mass.get(tuple(sorted(combo)), 0.0)
            )
            mass = max(0.0, 1.0 - illegal)
            conditional_mass.append(mass)
            first_adjusted.append(probability * mass)
        self.normalizer = math.fsum(first_adjusted)
        if self.normalizer <= 0:
            raise ValueError("joint range normalizer is zero")
        self._first_probabilities = tuple(value / self.normalizer for value in first_adjusted)
        second_adjusted = []
        for combo, probability in zip(second.combos, second.probabilities):
            illegal = (
                first_card_mass.get(combo[0], 0.0)
                + first_card_mass.get(combo[1], 0.0)
                - first_combo_mass.get(tuple(sorted(combo)), 0.0)
            )
            second_adjusted.append(probability * max(0.0, 1.0 - illegal))
        self._second_probabilities = tuple(
            value / self.normalizer for value in second_adjusted
        )
        if (
            abs(math.fsum(self._first_probabilities) - 1.0) > 1e-12
            or abs(math.fsum(self._second_probabilities) - 1.0) > 1e-12
        ):
            raise ValueError("blocker-conditioned joint marginals do not normalize")
        self.declared_preconditioning_marginal_identity_sha256 = sha256_json(
            {
                "player_0": {
                    "combos": [list(combo) for combo in first.combos],
                    "probabilities": list(first.probabilities),
                },
                "player_1": {
                    "combos": [list(combo) for combo in second.combos],
                    "probabilities": list(second.probabilities),
                },
            }
        )
        self.sampled_blocker_conditioned_marginal_identity_sha256 = sha256_json(
            {
                "joint_normalizer": self.normalizer,
                "player_0": {
                    "combos": [list(combo) for combo in first.combos],
                    "probabilities": list(self._first_probabilities),
                },
                "player_1": {
                    "combos": [list(combo) for combo in second.combos],
                    "probabilities": list(self._second_probabilities),
                },
            }
        )
        self._first_cdf = self._cdf(self._first_probabilities)
        self._second_cdf = self._cdf(second.probabilities)
        self._conditional_mass = tuple(conditional_mass)
        card_bits = {card: 1 << index for index, card in enumerate(DECK)}
        self._first_masks = tuple(
            card_bits[combo[0]] | card_bits[combo[1]] for combo in first.combos
        )
        self._second_masks = tuple(
            card_bits[combo[0]] | card_bits[combo[1]] for combo in second.combos
        )
        self._first_indices = {combo: index for index, combo in enumerate(first.combos)}
        self._second_indices = {
            combo: index for index, combo in enumerate(second.combos)
        }
        self._sample_calls = 0
        self._second_draw_attempts = 0
        self._rejection_count = 0
        self._sample_nanoseconds = 0

    def sampled_marginals(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        """Return the exact marginals of the compatibility-conditioned draw law."""

        return (
            dict(zip(self.first.combos, self._first_probabilities)),
            dict(zip(self.second.combos, self._second_probabilities)),
        )

    def declared_preconditioning_marginals(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        return (
            dict(zip(self.first.combos, self.first.probabilities)),
            dict(zip(self.second.combos, self.second.probabilities)),
        )

    @staticmethod
    def _cdf(probabilities: Sequence[float]) -> tuple[float, ...]:
        total = 0.0
        values = []
        for probability in probabilities:
            total += probability
            values.append(total)
        values[-1] = 1.0
        return tuple(values)

    def sample(self, rng: random.Random) -> Deal:
        started = time.perf_counter_ns()
        index0 = bisect.bisect_left(self._first_cdf, rng.random())
        index0 = min(index0, len(self.first.combos) - 1)
        hole0 = self.first.combos[index0]
        blocked_mask = self._first_masks[index0]
        # Rejection from player 1's marginal is exactly its distribution
        # conditioned on compatibility with the already sampled player-0 hand.
        # This replaces a full 1,081/1,128/1,225/1,326-combo scan per sample.
        attempts = 0
        while True:
            attempts += 1
            index1 = bisect.bisect_left(self._second_cdf, rng.random())
            index1 = min(index1, len(self.second.combos) - 1)
            if not (blocked_mask & self._second_masks[index1]):
                break
        self._sample_calls += 1
        self._second_draw_attempts += attempts
        self._rejection_count += attempts - 1
        self._sample_nanoseconds += time.perf_counter_ns() - started
        # Range construction already validated both supports against the board;
        # the bit-mask overlap gate above proves the only remaining invariant.
        return Deal(hole0, self.second.combos[index1])

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": "stage21_joint_range_sampler_diagnostics_v1",
            "sample_calls": self._sample_calls,
            "second_draw_attempts": self._second_draw_attempts,
            "rejection_count": self._rejection_count,
            "mean_second_draw_attempts": (
                self._second_draw_attempts / self._sample_calls
                if self._sample_calls
                else 0.0
            ),
            "sample_seconds": self._sample_nanoseconds / 1_000_000_000.0,
            "combo_overlap_uses_precomputed_bitmasks": True,
            "per_sample_board_revalidation_removed_after_constructor_proof": True,
        }

    def joint_probability(self, hole0: tuple[str, str], hole1: tuple[str, str]) -> float:
        try:
            i0 = self._first_indices[hole0]
            i1 = self._second_indices[hole1]
        except KeyError:
            return 0.0
        if self._first_masks[i0] & self._second_masks[i1]:
            return 0.0
        return self.first.probabilities[i0] * self.second.probabilities[i1] / self.normalizer

    def iter_joint(self):
        for first_index, (hole0, probability0) in enumerate(
            zip(self.first.combos, self.first.probabilities)
        ):
            blocked_mask = self._first_masks[first_index]
            for second_index, (hole1, probability1) in enumerate(
                zip(self.second.combos, self.second.probabilities)
            ):
                if not (blocked_mask & self._second_masks[second_index]):
                    yield Deal(hole0, hole1), probability0 * probability1 / self.normalizer


def build_joint_sampler(root: Mapping[str, Any], state: HoldemState) -> JointRangeSampler:
    return JointRangeSampler(
        build_player_range(root, state, 0),
        build_player_range(root, state, 1),
        state.board,
    )
