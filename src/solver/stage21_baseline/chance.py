from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .cards import DECK, canonical_flop_orbits, canonical_public_board, normalize_cards
from .game import Deal
from .ranges import JointRangeSampler


@lru_cache(maxsize=8_192)
def _legal_next_cards_cached(
    board: tuple[str, ...], deal: Deal
) -> tuple[str, ...]:
    deal.validate(board)
    blocked = set((*board, *deal.hole0, *deal.hole1))
    return tuple(card for card in DECK if card not in blocked)


def legal_next_cards(board: Sequence[str], deal: Deal) -> tuple[str, ...]:
    return _legal_next_cards_cached(tuple(board), deal)


@lru_cache(maxsize=2_048)
def _remaining_preflop_cards(deal: Deal) -> tuple[str, ...]:
    deal.validate(())
    blocked = set((*deal.hole0, *deal.hole1))
    return tuple(card for card in DECK if card not in blocked)


def per_deal_next_card_distribution(board: Sequence[str], deal: Deal) -> dict[str, float]:
    if len(board) not in {3, 4}:
        raise ValueError("next-card distribution requires a Flop or Turn board")
    cards = legal_next_cards(board, deal)
    expected = 52 - len(board) - 4
    if len(cards) != expected:
        raise AssertionError("next-card denominator mismatch")
    probability = 1.0 / len(cards)
    return {card: probability for card in cards}


def marginal_next_card_distribution(
    board: Sequence[str], sampler: JointRangeSampler
) -> dict[str, float]:
    """Exact range/blocker-conditioned public next-card distribution.

    The public candidate set contains 49 cards on the Flop and 48 on the Turn.
    Once both two-card private hands are fixed, exactly 45 or 44 candidates
    remain. Marginal probabilities therefore use P(card absent from both hands)
    divided by that per-deal legal-card denominator; they are never assumed
    uniform across the public candidate set.
    """

    public = normalize_cards(board)
    if len(public) not in {3, 4}:
        raise ValueError("marginal next-card distribution requires Flop or Turn")
    candidates = [card for card in DECK if card not in public]
    denominator = 52 - len(public) - 4
    blocked_mass = {card: 0.0 for card in candidates}
    joint_mass = 0.0
    for deal, probability in sampler.iter_joint():
        joint_mass += probability
        for card in (*deal.hole0, *deal.hole1):
            blocked_mass[card] += probability
    if abs(joint_mass - 1.0) > 1e-10:
        raise AssertionError("joint range distribution does not sum to one")
    result = {card: (1.0 - blocked_mass[card]) / denominator for card in candidates}
    if abs(math.fsum(result.values()) - 1.0) > 1e-10:
        raise AssertionError("marginal next-card probabilities do not sum to one")
    return result


def sample_next_card(board: Sequence[str], deal: Deal, rng: random.Random) -> tuple[str, float]:
    cards = legal_next_cards(board, deal)
    index = rng.randrange(len(cards))
    return cards[index], 1.0 / len(cards)


def legal_flops_for_deal(deal: Deal) -> tuple[tuple[str, str, str], ...]:
    remaining = _remaining_preflop_cards(deal)
    flops = tuple(itertools.combinations(remaining, 3))
    if len(flops) != 17_296:
        raise AssertionError("a fixed two-player deal must have C(48,3) legal flops")
    return flops


def sample_flop(deal: Deal, rng: random.Random) -> tuple[str, str, str]:
    """Sample one physical Flop uniformly without materializing C(48, 3)."""

    remaining = _remaining_preflop_cards(deal)
    selected = set(rng.sample(remaining, 3))
    # Return deck order so equivalent random permutations have one canonical
    # public representation. Every unordered combination has exactly 3!
    # generating permutations and therefore remains uniform.
    flop = tuple(card for card in DECK if card in selected)
    if len(flop) != 3:
        raise AssertionError("uniform Flop sampler did not return three cards")
    return flop  # type: ignore[return-value]


def canonical_flop_distribution_for_deal(deal: Deal) -> dict[tuple[str, ...], float]:
    counts: Counter[tuple[str, ...]] = Counter()
    for flop in legal_flops_for_deal(deal):
        counts[canonical_public_board(flop)] += 1
    total = math.comb(48, 3)
    if sum(counts.values()) != total:
        raise AssertionError("canonical blocker-conditioned flop partition is incomplete")
    return {key: value / total for key, value in sorted(counts.items())}


def physical_flop_orbit_receipt() -> dict[str, Any]:
    orbits = canonical_flop_orbits()
    return {
        "schema_version": "stage21_physical_flop_orbit_receipt_v2",
        "physical_flop_count": sum(orbits.values()),
        "suit_isomorphic_orbit_count": len(orbits),
        "multiplicity_min": min(orbits.values()),
        "multiplicity_max": max(orbits.values()),
        "every_physical_flop_mapped_exactly_once": sum(orbits.values()) == math.comb(52, 3),
    }


def enumerate_all_in_runouts(board: Sequence[str], deal: Deal) -> tuple[tuple[str, ...], ...]:
    public = normalize_cards(board)
    missing = 5 - len(public)
    if missing < 0 or missing > 2:
        raise ValueError("bounded all-in runout enumerator supports Flop through River")
    remaining = legal_next_cards(public, deal) if missing else ()
    if missing == 0:
        return (public,)
    if missing == 1:
        return tuple(public + (card,) for card in remaining)
    return tuple(public + tuple(cards) for cards in itertools.combinations(remaining, 2))
