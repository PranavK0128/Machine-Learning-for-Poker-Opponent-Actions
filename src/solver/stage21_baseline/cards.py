from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
RANK_TO_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_CHARS)}
VALUE_TO_RANK = {value: rank for rank, value in RANK_TO_VALUE.items()}
SUIT_TO_VALUE = {suit: index for index, suit in enumerate(SUIT_CHARS)}
DECK = tuple(f"{rank}{suit}" for rank in RANK_CHARS for suit in SUIT_CHARS)
SUIT_PERMUTATIONS = tuple(itertools.permutations(range(4)))


@lru_cache(maxsize=len(DECK))
def parse_card(card: str) -> tuple[int, int]:
    if not isinstance(card, str) or len(card) != 2:
        raise ValueError(f"invalid card: {card!r}")
    try:
        return RANK_TO_VALUE[card[0]], SUIT_TO_VALUE[card[1]]
    except KeyError as error:
        raise ValueError(f"invalid card: {card!r}") from error


def normalize_cards(cards: Iterable[str]) -> tuple[str, ...]:
    result = tuple(cards)
    for card in result:
        parse_card(card)
    if len(result) != len(set(result)):
        raise ValueError("duplicate cards")
    return result


@lru_cache(maxsize=len(DECK))
def card_sort_key(card: str) -> tuple[int, int]:
    rank, suit = parse_card(card)
    return (-rank, suit)


def canonical_public_board(board: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize a public board under all four-suit permutations."""

    board_tuple = normalize_cards(board)
    candidates: list[tuple[str, ...]] = []
    for permutation in SUIT_PERMUTATIONS:
        transformed = []
        for card in board_tuple:
            rank, suit = parse_card(card)
            transformed.append(f"{VALUE_TO_RANK[rank]}{SUIT_CHARS[permutation[suit]]}")
        candidates.append(tuple(sorted(transformed, key=card_sort_key)))
    return min(candidates)


def canonical_public_private_state(
    board: Sequence[str], hole0: Sequence[str], hole1: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    all_cards = normalize_cards((*board, *hole0, *hole1))
    del all_cards
    candidates = []
    for permutation in SUIT_PERMUTATIONS:
        groups = []
        for group in (board, hole0, hole1):
            transformed = []
            for card in group:
                rank, suit = parse_card(card)
                transformed.append(f"{VALUE_TO_RANK[rank]}{SUIT_CHARS[permutation[suit]]}")
            groups.append(tuple(sorted(transformed, key=card_sort_key)))
        candidates.append(tuple(groups))
    return min(candidates)


def preflop_class(hole: Sequence[str]) -> str:
    if len(hole) != 2:
        raise ValueError("a Hold'em hole hand has two cards")
    first, second = normalize_cards(hole)
    rank0, suit0 = parse_card(first)
    rank1, suit1 = parse_card(second)
    high, low = sorted((rank0, rank1), reverse=True)
    if high == low:
        return VALUE_TO_RANK[high] * 2
    suffix = "s" if suit0 == suit1 else "o"
    return f"{VALUE_TO_RANK[high]}{VALUE_TO_RANK[low]}{suffix}"


def all_preflop_classes() -> tuple[str, ...]:
    classes = {preflop_class(combo) for combo in itertools.combinations(DECK, 2)}
    if len(classes) != 169:
        raise AssertionError("preflop registry must contain 169 classes")
    return tuple(sorted(classes))


def legal_hole_combos(board: Sequence[str]) -> tuple[tuple[str, str], ...]:
    public = set(normalize_cards(board))
    remaining = [card for card in DECK if card not in public]
    return tuple(itertools.combinations(remaining, 2))


def _straight_high(ranks: Iterable[int]) -> int:
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if all(rank in unique for rank in range(high - 4, high + 1)):
            return high
    return 0


@lru_cache(maxsize=16_384)
def _five_card_rank_cached(cards: tuple[str, ...]) -> tuple[int, ...]:
    parsed = [parse_card(card) for card in cards]
    ranks = [rank for rank, _ in parsed]
    suits = [suit for _, suit in parsed]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight = _straight_high(ranks)
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    ordered = tuple(sorted(ranks, reverse=True))
    if flush:
        return (5, *ordered)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    pairs = sorted((rank for count, rank in groups if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(rank for rank in ranks if rank not in pairs)
        return (2, *pairs, kicker)
    if len(pairs) == 1:
        kickers = sorted((rank for rank in ranks if rank != pairs[0]), reverse=True)
        return (1, pairs[0], *kickers)
    return (0, *ordered)


def five_card_rank(cards: Sequence[str]) -> tuple[int, ...]:
    if len(cards) != 5:
        raise ValueError("five_card_rank requires exactly five cards")
    normalized = normalize_cards(cards)
    return _five_card_rank_cached(tuple(sorted(normalized)))


@lru_cache(maxsize=8_192)
def _best_hand_rank_cached(cards: tuple[str, ...]) -> tuple[int, ...]:
    return max(
        _five_card_rank_cached(tuple(sorted(combo)))
        for combo in itertools.combinations(cards, 5)
    )


def best_hand_rank(cards: Sequence[str]) -> tuple[int, ...]:
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError("best_hand_rank requires five to seven cards")
    normalized = normalize_cards(cards)
    return _best_hand_rank_cached(tuple(sorted(normalized)))


def board_features(board: Sequence[str]) -> dict[str, object]:
    cards = normalize_cards(board)
    if len(cards) not in {3, 4, 5}:
        raise ValueError("public postflop board must have three to five cards")
    parsed = [parse_card(card) for card in cards]
    rank_counts = Counter(rank for rank, _ in parsed)
    suit_counts = Counter(suit for _, suit in parsed)
    rank_pattern = tuple(sorted(rank_counts.values(), reverse=True))
    max_suit = max(suit_counts.values())
    straight_present = len(cards) == 5 and _straight_high(rank_counts) > 0
    windows = 0
    minimum_missing = 5
    rank_set = set(rank_counts)
    if 14 in rank_set:
        rank_set.add(1)
    for low in range(1, 11):
        present = len(set(range(low, low + 5)) & rank_set)
        if present >= 3:
            windows += 1
        minimum_missing = min(minimum_missing, 5 - present)
    if rank_pattern[0] >= 4:
        pairedness = "quads_on_board"
    elif rank_pattern[0] == 3:
        pairedness = "trips_on_board"
    elif rank_pattern.count(2) >= 2:
        pairedness = "double_paired"
    elif rank_pattern[0] == 2:
        pairedness = "paired"
    else:
        pairedness = "unpaired"
    suit_texture = {
        1: "rainbow",
        2: "two_tone",
        3: "monotone" if len(cards) == 3 else "three_flush",
        4: "four_flush",
        5: "five_flush",
    }[max_suit]
    return {
        "canonical_board": list(canonical_public_board(cards)),
        "pairedness": pairedness,
        "rank_count_pattern": list(rank_pattern),
        "suit_texture": suit_texture,
        "max_suit_count": max_suit,
        "straight_on_board": straight_present,
        "straight_window_count": windows,
        "straight_window_gap_min": minimum_missing,
        "highest_rank": max(rank_counts),
    }


def straight_completion_ranks(cards: Sequence[str]) -> tuple[int, ...]:
    """Return distinct ranks that complete at least one five-rank straight.

    Ace is reported as 14 for both Broadway and wheel completion.  This is a
    rank-structure helper: callers decide whether the structure is a live draw
    on the current street or only a River missed-draw diagnostic.
    """

    normalized = normalize_cards(cards)
    unique_ranks = {parse_card(card)[0] for card in normalized}
    if 14 in unique_ranks:
        unique_ranks.add(1)
    completion_ranks: set[int] = set()
    for low in range(1, 11):
        window = set(range(low, low + 5))
        missing = window - unique_ranks
        if len(missing) == 1:
            missing_rank = next(iter(missing))
            completion_ranks.add(14 if missing_rank == 1 else missing_rank)
    return tuple(sorted(completion_ranks))


def classify_public_transition(
    parent_board: Sequence[str], transition_card: str
) -> dict[str, object]:
    """Classify an exact Flop->Turn or Turn->River public-card transition.

    Several labels may be true at once.  ``blank`` is deliberately structural:
    an overcard or undercard can still be a blank when it does not pair the
    board, change straight/flush pressure, or create a counterfeit threat.
    ``nut_advantage_reversal`` is initialized as false here and is replaced by
    the fixed-posterior range calculation when an executable root is built.
    """

    parent = normalize_cards(parent_board)
    if len(parent) not in {3, 4}:
        raise ValueError("transition parent must be an exact Flop or Turn board")
    parse_card(transition_card)
    if transition_card in parent:
        raise ValueError("transition card collides with its parent board")
    child = (*parent, transition_card)
    parent_features = board_features(parent)
    child_features = board_features(child)
    transition_rank, transition_suit = parse_card(transition_card)
    parent_ranks = [parse_card(card)[0] for card in parent]
    parent_suit_count = sum(
        parse_card(card)[1] == transition_suit for card in parent
    )
    parent_completions = straight_completion_ranks(parent)
    child_completions = straight_completion_ranks(child)

    pairing = transition_rank in set(parent_ranks)
    overcard = transition_rank > max(parent_ranks)
    undercard = transition_rank < min(parent_ranks)
    flush_completion = parent_suit_count == 2
    new_flush_draw = parent_suit_count == 1
    straight_completion = (
        int(parent_features["straight_window_gap_min"]) == 1
        and int(child_features["straight_window_gap_min"]) == 0
    )
    new_oesd = len(child_completions) >= 2 and len(parent_completions) < 2
    new_gutshot = len(child_completions) == 1 and len(parent_completions) == 0
    parent_pairedness = str(parent_features["pairedness"])
    child_pairedness = str(child_features["pairedness"])
    counterfeit = pairing and (
        parent_pairedness == "unpaired"
        or child_pairedness in {"double_paired", "trips_on_board", "quads_on_board"}
    )
    structural_change = any(
        (
            pairing,
            flush_completion,
            new_flush_draw,
            straight_completion,
            new_oesd,
            new_gutshot,
            counterfeit,
        )
    )
    flags = {
        "blank": not structural_change,
        "overcard": overcard,
        "undercard": undercard,
        "pairing": pairing,
        "flush_completion": flush_completion,
        "new_flush_draw": new_flush_draw,
        "straight_completion": straight_completion,
        "new_gutshot": new_gutshot,
        "new_oesd": new_oesd,
        "counterfeit": counterfeit,
        "nut_advantage_reversal": False,
    }
    labels = [name for name, enabled in flags.items() if enabled]
    if not labels:
        labels = ["between_rank_nonblank"]
    return {
        "schema_version": "stage21_public_transition_classification_v2",
        "parent_board": list(parent),
        "transition_card": transition_card,
        "child_board": list(child),
        "parent_street": "flop" if len(parent) == 3 else "turn",
        "child_street": "turn" if len(child) == 4 else "river",
        "parent_board_classification": parent_features,
        "child_board_classification": child_features,
        "parent_straight_completion_ranks": list(parent_completions),
        "child_straight_completion_ranks": list(child_completions),
        "flags": flags,
        "labels": labels,
        "child_street_has_future_public_card": len(child) < 5,
        "river_draw_labels_are_structural_missed_draw_diagnostics": len(child) == 5,
        "definitions": {
            "blank": (
                "no pairing, flush-pressure change, straight-pressure change, "
                "straight completion, or counterfeit threat; an overcard or "
                "undercard may still be structurally blank"
            ),
            "overcard": "transition rank is strictly above every parent-board rank",
            "undercard": "transition rank is strictly below every parent-board rank",
            "pairing": "transition rank duplicates a parent-board rank",
            "flush_completion": (
                "structurally changes a two-card same-suit parent texture to a "
                "three-card same-suit child; executable roots may additionally "
                "promote this flag when range-weighted private combos newly make a flush"
            ),
            "new_flush_draw": (
                "changes a one-card same-suit parent texture to two cards of that "
                "suit; live on a Turn child and a missed/backdoor structural diagnostic "
                "on a River child"
            ),
            "straight_completion": (
                "closes a parent public-board five-rank gap; executable roots may "
                "add range-weighted private-combo straight completions"
            ),
            "new_gutshot": (
                "child rank structure has exactly one straight-completing rank and "
                "parent had none; live on Turn and diagnostic on River"
            ),
            "new_oesd": (
                "child rank structure has at least two distinct straight-completing "
                "ranks and parent had fewer than two; includes double-gutshot rank "
                "structures and is live on Turn but diagnostic on River"
            ),
            "counterfeit": (
                "pairing transition creates the first public pair or advances the "
                "board to double-paired, trips-on-board, or quads-on-board"
            ),
            "nut_advantage_reversal": (
                "false in the structural classifier; executable roots replace it "
                "with the documented fixed-posterior range proxy"
            ),
        },
    }


def hand_features(board: Sequence[str], hole: Sequence[str]) -> dict[str, object]:
    public = normalize_cards(board)
    private = normalize_cards(hole)
    if len(private) != 2:
        raise ValueError("hole hand must contain two cards")
    normalize_cards((*public, *private))
    parsed = [parse_card(card) for card in (*public, *private)]
    ranks = [rank for rank, _ in parsed]
    suits = [suit for _, suit in parsed]
    rank = best_hand_rank((*public, *private)) if len(public) >= 3 else None
    suit_counts = Counter(suits)
    flush_draw = len(public) < 5 and max(suit_counts.values()) == 4
    completion_ranks = straight_completion_ranks((*public, *private))
    # Two distinct completing ranks are an eight-out open-ended or double-
    # gutshot draw. One completing rank is a four-out gutshot, including the
    # A-2-3-4 wheel edge. Merely missing a window endpoint is not sufficient:
    # the neighboring window must supply the second real completion rank.
    open_ended = len(completion_ranks) >= 2
    gutshot = len(completion_ranks) == 1
    hole_ranks = sorted((parse_card(card)[0] for card in private), reverse=True)
    return {
        "made_category": None if rank is None else rank[0],
        "made_rank": None if rank is None else list(rank),
        "flush_draw": flush_draw,
        "open_ended_straight_draw": open_ended,
        "gutshot": gutshot,
        "straight_completion_ranks": list(completion_ranks),
        "pocket_pair": hole_ranks[0] == hole_ranks[1],
        "overcards": sum(value > max(parse_card(card)[0] for card in public) for value in hole_ranks)
        if public
        else 0,
        "preflop_class": preflop_class(private),
    }


@lru_cache(maxsize=1)
def canonical_flop_orbits() -> dict[tuple[str, ...], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    for flop in itertools.combinations(DECK, 3):
        counts[canonical_public_board(flop)] += 1
    if sum(counts.values()) != 22_100:
        raise AssertionError("physical flop partition is incomplete")
    return dict(sorted(counts.items()))
