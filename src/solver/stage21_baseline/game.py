from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .cards import best_hand_rank, board_features, hand_features, normalize_cards
from .common import STRATEGY_ABSTRACTION_VERSION, sha256_json


STREETS = ("preflop", "flop", "turn", "river")
BOARD_LENGTH = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
DEFAULT_BET_FRACTIONS = (0.33, 0.50, 0.75, 1.00, 1.50)


@dataclass(frozen=True)
class Deal:
    hole0: tuple[str, str]
    hole1: tuple[str, str]

    def validate(self, board: Sequence[str]) -> None:
        _validate_deal(self, tuple(board))


@lru_cache(maxsize=8_192)
def _validate_deal(deal: Deal, board: tuple[str, ...]) -> None:
    cards = normalize_cards((*board, *deal.hole0, *deal.hole1))
    if len(cards) != len(board) + 4:
        raise ValueError("invalid deal")


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0
    label: str = ""


@dataclass(frozen=True)
class ActionRecord:
    street: str
    player: int
    kind: str
    amount: int
    to_amount: int
    full_raise: bool
    reopened: bool


@dataclass(frozen=True)
class HoldemState:
    street: str
    board: tuple[str, ...]
    button_player: int
    actor: int | None
    starting_stacks: tuple[int, int]
    stacks: tuple[int, int]
    contributions: tuple[int, int]
    street_contributions: tuple[int, int]
    current_bet: int
    last_full_raise: int
    acted_since_full_raise: tuple[bool, bool]
    checks_in_row: int
    folded: tuple[bool, bool]
    terminal: bool
    all_in_runout: bool
    history: tuple[ActionRecord, ...]
    bet_fractions: tuple[float, ...] = DEFAULT_BET_FRACTIONS

    @lru_cache(maxsize=8_192)
    def validate(self) -> None:
        vectors = (
            self.starting_stacks,
            self.stacks,
            self.contributions,
            self.street_contributions,
            self.acted_since_full_raise,
            self.folded,
        )
        if any(len(values) != 2 for values in vectors):
            raise ValueError("Stage 21 baseline state vectors must be heads-up")
        if self.street not in STREETS:
            raise ValueError("invalid street")
        if len(self.board) != BOARD_LENGTH[self.street]:
            raise ValueError("board length does not match street")
        normalize_cards(self.board)
        if self.button_player not in {0, 1}:
            raise ValueError("button must be player 0 or 1")
        if self.actor not in {None, 0, 1}:
            raise ValueError("invalid actor")
        for index in range(2):
            if self.starting_stacks[index] <= 0:
                raise ValueError("starting stacks must be positive")
            if min(
                self.stacks[index],
                self.contributions[index],
                self.street_contributions[index],
            ) < 0:
                raise ValueError("negative chip state")
            if self.stacks[index] + self.contributions[index] != self.starting_stacks[index]:
                raise ValueError("stack plus contribution must conserve chips")
            if self.street_contributions[index] > self.contributions[index]:
                raise ValueError("street contribution exceeds total contribution")
        if self.current_bet != max(self.street_contributions):
            raise ValueError("current bet mismatch")
        if self.last_full_raise <= 0:
            raise ValueError("last full raise must be positive")
        if self.checks_in_row < 0 or self.checks_in_row > 2:
            raise ValueError("invalid consecutive-check count")
        if sum(bool(value) for value in self.folded) > 1:
            raise ValueError("both players cannot be folded")
        if self.terminal and self.actor is not None:
            raise ValueError("terminal state cannot have an actor")
        if not self.terminal and self.actor is not None and self.stacks[self.actor] <= 0:
            raise ValueError("nonterminal decision actor must have positive chips")
        if self.all_in_runout and (self.actor is not None or min(self.stacks) != 0):
            raise ValueError("all-in runout state must have no actor and an empty stack")
        if self.terminal and not any(self.folded) and self.street != "river":
            raise ValueError("non-fold terminal showdown requires a River board")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in self.bet_fractions
        ):
            raise ValueError("bet fractions must be positive finite numbers")
        _history_replay_contexts(self)

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    @property
    def to_call(self) -> int:
        if self.actor is None:
            return 0
        return min(
            self.current_bet - self.street_contributions[self.actor],
            self.stacks[self.actor],
        )


@dataclass(frozen=True)
class PotLayer:
    lower: int
    upper: int
    amount: int
    contributors: tuple[int, ...]
    eligible: tuple[int, ...]
    uncalled_return_to: int | None


@dataclass(frozen=True)
class Settlement:
    payouts: tuple[int, int]
    net: tuple[int, int]
    layers: tuple[PotLayer, ...]


def postflop_first_actor(button_player: int) -> int:
    return 1 - button_player


def preflop_first_actor(button_player: int) -> int:
    return button_player


def make_preflop_start(
    *,
    starting_stacks: tuple[int, int] = (10_000, 10_000),
    button_player: int = 0,
    small_blind: int = 50,
    big_blind: int = 100,
    bet_fractions: Sequence[float] = DEFAULT_BET_FRACTIONS,
) -> HoldemState:
    if len(starting_stacks) != 2 or any(
        not isinstance(stack, int) or isinstance(stack, bool) or stack <= 0
        for stack in starting_stacks
    ):
        raise ValueError("heads-up starting stacks must be two positive integers")
    if button_player not in {0, 1}:
        raise ValueError("button must be player 0 or 1")
    if small_blind <= 0 or big_blind <= small_blind:
        raise ValueError("invalid blinds")
    contributions = [0, 0]
    contributions[button_player] = min(small_blind, starting_stacks[button_player])
    bb_player = 1 - button_player
    contributions[bb_player] = min(big_blind, starting_stacks[bb_player])
    stacks = tuple(starting_stacks[i] - contributions[i] for i in range(2))
    first_actor = preflop_first_actor(button_player)
    first_actor_has_decision = stacks[first_actor] > 0 and (
        stacks[bb_player] > 0
        or contributions[first_actor] < contributions[bb_player]
    )
    blind_all_in_runout = min(stacks) == 0 and not first_actor_has_decision
    state = HoldemState(
        street="preflop",
        board=(),
        button_player=button_player,
        actor=first_actor if first_actor_has_decision else None,
        starting_stacks=starting_stacks,
        stacks=stacks,
        contributions=tuple(contributions),
        street_contributions=tuple(contributions),
        current_bet=max(contributions),
        last_full_raise=big_blind,
        acted_since_full_raise=(False, False),
        checks_in_row=0,
        folded=(False, False),
        terminal=False,
        all_in_runout=blind_all_in_runout,
        history=(),
        bet_fractions=tuple(float(value) for value in bet_fractions),
    )
    state.validate()
    return state


def make_street_start(
    *,
    street: str,
    board: Sequence[str],
    starting_stacks: tuple[int, int],
    contributions: tuple[int, int],
    button_player: int = 0,
    actor: int | None = None,
    forced_runout: bool = False,
    big_blind: int = 100,
    bet_fractions: Sequence[float] = DEFAULT_BET_FRACTIONS,
) -> HoldemState:
    if street == "preflop":
        raise ValueError("use make_preflop_start for a fresh preflop state")
    board_tuple = normalize_cards(board)
    if len(board_tuple) != BOARD_LENGTH.get(street, -1):
        raise ValueError("board length does not match requested street")
    stacks = tuple(starting_stacks[i] - contributions[i] for i in range(2))
    if actor is None and not forced_runout:
        actor = postflop_first_actor(button_player)
    all_in = forced_runout or min(stacks) == 0
    state = HoldemState(
        street=street,
        board=board_tuple,
        button_player=button_player,
        actor=None if all_in else actor,
        starting_stacks=starting_stacks,
        stacks=stacks,
        contributions=contributions,
        street_contributions=(0, 0),
        current_bet=0,
        last_full_raise=big_blind,
        acted_since_full_raise=(False, False),
        checks_in_row=0,
        folded=(False, False),
        terminal=street == "river" and all_in and len(board_tuple) == 5,
        all_in_runout=all_in and street != "river",
        history=(),
        bet_fractions=tuple(float(value) for value in bet_fractions),
    )
    state.validate()
    return state


def state_from_root(root: Mapping[str, Any]) -> HoldemState:
    snapshot = root["executable_state"]
    state = HoldemState(
        street=str(root["street"]),
        board=tuple(root.get("representative_board", ())),
        button_player=int(snapshot.get("button_player", 0)),
        actor=snapshot.get("actor"),
        starting_stacks=tuple(int(value) for value in snapshot["starting_stacks"]),
        stacks=tuple(int(value) for value in snapshot["stacks"]),
        contributions=tuple(int(value) for value in snapshot["contributions"]),
        street_contributions=tuple(int(value) for value in snapshot["street_contributions"]),
        current_bet=int(snapshot["current_bet"]),
        last_full_raise=int(snapshot["last_full_raise"]),
        acted_since_full_raise=tuple(bool(value) for value in snapshot["acted_since_full_raise"]),
        checks_in_row=int(snapshot["checks_in_row"]),
        folded=tuple(bool(value) for value in snapshot["folded"]),
        terminal=bool(snapshot["terminal"]),
        all_in_runout=bool(snapshot["all_in_runout"]),
        history=tuple(ActionRecord(**row) for row in snapshot.get("history", ())),
        bet_fractions=tuple(float(value) for value in snapshot["bet_fractions"]),
    )
    state.validate()
    return state


def state_snapshot(state: HoldemState) -> dict[str, Any]:
    state.validate()
    return {
        "button_player": state.button_player,
        "actor": state.actor,
        "starting_stacks": list(state.starting_stacks),
        "stacks": list(state.stacks),
        "contributions": list(state.contributions),
        "street_contributions": list(state.street_contributions),
        "current_bet": state.current_bet,
        "last_full_raise": state.last_full_raise,
        "acted_since_full_raise": list(state.acted_since_full_raise),
        "checks_in_row": state.checks_in_row,
        "folded": list(state.folded),
        "terminal": state.terminal,
        "all_in_runout": state.all_in_runout,
        "history": [record.__dict__ for record in state.history],
        "bet_fractions": list(state.bet_fractions),
    }


@lru_cache(maxsize=8_192)
def _history_replay_contexts(
    state: HoldemState,
) -> tuple[tuple[ActionRecord, int, int, int], ...]:
    """Replay the contribution ledger and return causal sizing context.

    Each tuple is ``(record, pot_before, to_call_before,
    aggressive_increment)``. Postflop street contributions always begin at
    zero; current-Preflop forced blinds are derived from the final ledger.
    This validates loaded/root snapshots as well as states produced live.
    """

    if not state.history:
        return ()
    if any(row.street not in STREETS or row.player not in {0, 1} for row in state.history):
        raise ValueError("action history contains an invalid street or player")
    if any(row.amount < 0 or row.to_amount < 0 for row in state.history):
        raise ValueError("action history contains negative chip movement")
    street_indices = [STREETS.index(row.street) for row in state.history]
    if street_indices != sorted(street_indices):
        raise ValueError("action history street order is not chronological")
    running_pot = state.pot - sum(row.amount for row in state.history)
    if running_pot < 0:
        raise ValueError("action history transfers exceed the current pot")

    groups: list[tuple[str, tuple[ActionRecord, ...]]] = []
    for street in STREETS:
        rows = tuple(row for row in state.history if row.street == street)
        if rows:
            groups.append((street, rows))
    contexts: list[tuple[ActionRecord, int, int, int]] = []
    for group_index, (street, rows) in enumerate(groups):
        transferred = [
            sum(row.amount for row in rows if row.player == player) for player in (0, 1)
        ]
        if street == state.street:
            street_contributions = [
                state.street_contributions[player] - transferred[player]
                for player in (0, 1)
            ]
            if min(street_contributions) < 0:
                raise ValueError("current street history exceeds the contribution ledger")
            if street != "preflop" and any(street_contributions):
                raise ValueError("postflop action history must begin from zero street commitments")
        elif street == "preflop":
            inferred: list[int | None] = [None, None]
            for player in (0, 1):
                first = next((row for row in rows if row.player == player), None)
                if first is not None:
                    inferred[player] = first.to_amount - first.amount
                    if inferred[player] < 0:
                        raise ValueError("Preflop action predates its forced contribution")
            missing = [player for player, value in enumerate(inferred) if value is None]
            if group_index == 0 and len(missing) == 1:
                inferred[missing[0]] = running_pot - int(inferred[1 - missing[0]])
            street_contributions = [int(value or 0) for value in inferred]
            if group_index == 0 and sum(street_contributions) != running_pot:
                raise ValueError("Preflop history does not reconcile to its forced pot")
        else:
            street_contributions = [0, 0]

        for row in rows:
            player = row.player
            current_bet = max(street_contributions)
            to_call = max(0, current_bet - street_contributions[player])
            if row.kind not in {"fold", "check", "call", "bet", "raise"}:
                raise ValueError("action history contains an unsupported action")
            if row.kind in {"fold", "check"} and row.amount != 0:
                raise ValueError("fold/check history cannot transfer chips")
            if row.kind == "fold" and to_call <= 0:
                raise ValueError("fold history requires a pending wager")
            if row.kind == "check" and to_call != 0:
                raise ValueError("check history cannot face a pending wager")
            if row.kind == "call" and (row.amount <= 0 or to_call <= 0 or row.amount > to_call):
                raise ValueError("call history has invalid chip movement")
            if row.kind == "bet" and (current_bet != 0 or row.amount <= 0):
                raise ValueError("bet history requires an unopened street")
            if row.kind == "raise" and (
                current_bet <= 0 or row.amount <= to_call or row.to_amount <= current_bet
            ):
                raise ValueError("raise history does not exceed the prior wager")
            expected_to = street_contributions[player] + row.amount
            if row.to_amount != expected_to:
                raise ValueError("action history transfer and to-amount disagree")
            if row.reopened is not row.full_raise:
                raise ValueError("action history reopening flag disagrees with full-raise status")
            aggressive_increment = (
                row.amount - to_call if row.kind in {"bet", "raise"} else 0
            )
            contexts.append((row, running_pot, to_call, aggressive_increment))
            street_contributions[player] = expected_to
            running_pot += row.amount

        if street == state.street and tuple(street_contributions) != state.street_contributions:
            raise ValueError("replayed current street contributions do not match state")
    if running_pot != state.pot:
        raise ValueError("replayed action history does not conserve the pot")
    return tuple(contexts)


@lru_cache(maxsize=8_192)
def legal_actions(state: HoldemState) -> tuple[Action, ...]:
    """Return the immutable legal action set for one immutable state.

    ``HoldemState`` is frozen, so identical states have identical validation
    and action results.  Solver and evaluation walks revisit the same public
    payoff states many times; retaining this bounded result avoids replaying
    the complete action history and rebuilding the same tuple at every visit.
    """

    state.validate()
    if state.terminal or state.actor is None or state.all_in_runout:
        return ()
    player = state.actor
    opponent = 1 - player
    to_call = state.current_bet - state.street_contributions[player]
    actions: list[Action] = []
    if to_call > 0:
        actions.append(Action("fold", 0, "fold"))
        call_amount = min(to_call, state.stacks[player])
        if call_amount > 0:
            actions.append(
                Action(
                    "call",
                    call_amount,
                    "call" if call_amount == to_call else "call_all_in",
                )
            )
    else:
        actions.append(Action("check", 0, "check"))

    can_reopen = not state.acted_since_full_raise[player]
    maximum_to = state.street_contributions[player] + state.stacks[player]
    opponent_maximum_to = state.street_contributions[opponent] + state.stacks[opponent]
    effective_maximum_to = min(maximum_to, opponent_maximum_to)
    minimum_to = state.current_bet + state.last_full_raise
    if state.current_bet == 0:
        minimum_to = max(state.last_full_raise, 1)
    if can_reopen and effective_maximum_to > state.current_bet:
        pot_after_call = state.pot + max(to_call, 0)
        targets = set()
        for fraction in state.bet_fractions:
            target = state.street_contributions[player] + max(to_call, 0) + round(pot_after_call * fraction)
            target = min(target, effective_maximum_to)
            if target >= minimum_to:
                targets.add(target)
        if effective_maximum_to >= minimum_to:
            targets.add(effective_maximum_to)
        elif maximum_to > state.current_bet:
            # A short all-in below the full-raise threshold is legal, but it
            # does not reopen action for a player who has already acted.
            targets.add(min(maximum_to, effective_maximum_to))
        for target in sorted(targets):
            if target <= state.current_bet:
                continue
            increment = target - state.street_contributions[player]
            kind = "bet" if state.current_bet == 0 else "raise"
            label = "jam" if target == effective_maximum_to else f"{kind}_to_{target}"
            actions.append(Action(kind, increment, label))
    return tuple(actions)


def _record(
    state: HoldemState,
    action: Action,
    *,
    to_amount: int,
    full_raise: bool,
    reopened: bool,
) -> ActionRecord:
    assert state.actor is not None
    return ActionRecord(
        street=state.street,
        player=state.actor,
        kind=action.kind,
        amount=action.amount,
        to_amount=to_amount,
        full_raise=full_raise,
        reopened=reopened,
    )


def apply_action(
    state: HoldemState,
    action: Action,
    *,
    prevalidated_legal_actions: Sequence[Action] | None = None,
) -> HoldemState:
    """Apply one action, optionally reusing the caller's exact legal set.

    Hot solver/evaluation paths necessarily obtain ``legal_actions(state)``
    immediately before choosing an action.  Passing that immutable tuple here
    avoids a redundant lookup/validation while membership is still checked and
    the newly created child state is still fully validated before return.
    """

    legal = (
        legal_actions(state)
        if prevalidated_legal_actions is None
        else tuple(prevalidated_legal_actions)
    )
    if action not in legal:
        raise ValueError(f"illegal action {action}; legal={legal}")
    assert state.actor is not None
    player = state.actor
    opponent = 1 - player
    stacks = list(state.stacks)
    contributions = list(state.contributions)
    street_contributions = list(state.street_contributions)
    acted = list(state.acted_since_full_raise)

    if action.kind == "fold":
        folded = list(state.folded)
        folded[player] = True
        record = _record(state, action, to_amount=street_contributions[player], full_raise=False, reopened=False)
        result = replace(
            state,
            actor=None,
            folded=tuple(folded),
            terminal=True,
            history=state.history + (record,),
        )
        result.validate()
        return result

    amount = action.amount
    stacks[player] -= amount
    contributions[player] += amount
    street_contributions[player] += amount
    previous_bet = state.current_bet
    new_bet = max(street_contributions)
    raise_size = new_bet - previous_bet
    aggressive = action.kind in {"bet", "raise"}
    full_raise = aggressive and raise_size >= state.last_full_raise
    reopened = full_raise
    if full_raise:
        acted = [False, False]
    acted[player] = True
    record = _record(state, action, to_amount=street_contributions[player], full_raise=full_raise, reopened=reopened)

    round_closed = False
    checks = state.checks_in_row
    if action.kind == "check":
        checks += 1
        round_closed = all(acted) and street_contributions[0] == street_contributions[1]
    elif action.kind == "call":
        checks = 0
        # A blind is a forced contribution, not an action. A preflop limp
        # therefore passes the option to the big blind; calls responding to an
        # actual bet/raise close because both players have acted. A short
        # all-in call also closes even though contributions remain unmatched.
        round_closed = all(acted) or stacks[player] == 0
    else:
        checks = 0

    both_all_in_or_matched = (
        street_contributions[0] == street_contributions[1]
        and (stacks[0] == 0 or stacks[1] == 0)
    )
    if both_all_in_or_matched:
        round_closed = True

    terminal = state.terminal
    all_in_runout = state.all_in_runout
    next_actor: int | None = opponent
    if round_closed:
        next_actor = None
        if state.street == "river":
            terminal = True
        elif stacks[0] == 0 or stacks[1] == 0:
            all_in_runout = True

    result = replace(
        state,
        actor=next_actor,
        stacks=tuple(stacks),
        contributions=tuple(contributions),
        street_contributions=tuple(street_contributions),
        current_bet=new_bet,
        last_full_raise=raise_size if full_raise else state.last_full_raise,
        acted_since_full_raise=tuple(acted),
        checks_in_row=checks,
        terminal=terminal,
        all_in_runout=all_in_runout,
        history=state.history + (record,),
    )
    result.validate()
    return result


@lru_cache(maxsize=8_192)
def advance_public_card(state: HoldemState, card: str) -> HoldemState:
    state.validate()
    if state.terminal or state.street == "river":
        raise ValueError("cannot advance a terminal/River state")
    if state.actor is not None and not state.all_in_runout:
        raise ValueError("betting round is not closed")
    normalize_cards((*state.board, card))
    next_street = STREETS[STREETS.index(state.street) + 1]
    board = state.board + (card,)
    if state.street == "preflop":
        raise ValueError("a preflop-to-flop transition requires three cards")
    terminal = next_street == "river" and state.all_in_runout
    actor = None if state.all_in_runout else postflop_first_actor(state.button_player)
    result = replace(
        state,
        street=next_street,
        board=board,
        actor=actor,
        street_contributions=(0, 0),
        current_bet=0,
        last_full_raise=100,
        acted_since_full_raise=(False, False),
        checks_in_row=0,
        terminal=terminal,
    )
    result.validate()
    return result


def advance_flop(state: HoldemState, flop: Sequence[str]) -> HoldemState:
    state.validate()
    if state.street != "preflop" or state.actor is not None and not state.all_in_runout:
        raise ValueError("preflop betting round is not closed")
    cards = normalize_cards(flop)
    if len(cards) != 3:
        raise ValueError("flop transition requires three cards")
    actor = None if state.all_in_runout else postflop_first_actor(state.button_player)
    result = replace(
        state,
        street="flop",
        board=cards,
        actor=actor,
        street_contributions=(0, 0),
        current_bet=0,
        last_full_raise=100,
        acted_since_full_raise=(False, False),
        checks_in_row=0,
        terminal=False,
    )
    result.validate()
    return result


def build_pot_layers(
    contributions: Sequence[int], folded: Sequence[bool]
) -> tuple[PotLayer, ...]:
    if len(contributions) != len(folded) or not contributions:
        raise ValueError("contribution/fold vectors must be non-empty and aligned")
    values = tuple(int(value) for value in contributions)
    if any(value < 0 for value in values):
        raise ValueError("contributions cannot be negative")
    levels = sorted(set(value for value in values if value > 0))
    lower = 0
    layers: list[PotLayer] = []
    for upper in levels:
        contributors = tuple(index for index, value in enumerate(values) if value >= upper)
        amount = (upper - lower) * len(contributors)
        eligible = tuple(index for index in contributors if not folded[index])
        uncalled = contributors[0] if len(contributors) == 1 else None
        layers.append(PotLayer(lower, upper, amount, contributors, eligible, uncalled))
        lower = upper
    if sum(layer.amount for layer in layers) != sum(values):
        raise AssertionError("pot layers do not conserve contributions")
    return tuple(layers)


def settle_contributions(
    contributions: Sequence[int],
    folded: Sequence[bool],
    ranks: Sequence[tuple[int, ...] | None],
    *,
    odd_chip_order: Sequence[int] = (0, 1),
) -> Settlement:
    if not (len(contributions) == len(folded) == len(ranks) == 2):
        raise ValueError("Stage 21 baseline settlement is heads-up")
    order = tuple(int(player) for player in odd_chip_order)
    if set(order) != {0, 1} or len(order) != 2:
        raise ValueError("odd-chip order must contain both players exactly once")
    layers = build_pot_layers(contributions, folded)
    payouts = [0, 0]
    for layer in layers:
        if layer.uncalled_return_to is not None:
            if folded[layer.uncalled_return_to]:
                raise ValueError("a folded player cannot own an unmatched excess contribution")
            payouts[layer.uncalled_return_to] += layer.amount
            continue
        eligible = layer.eligible
        if not eligible:
            raise ValueError("pot layer has no eligible player")
        if len(eligible) == 1:
            winners = eligible
        else:
            best = max(ranks[player] for player in eligible if ranks[player] is not None)
            winners = tuple(player for player in eligible if ranks[player] == best)
        share, remainder = divmod(layer.amount, len(winners))
        for player in winners:
            payouts[player] += share
        for player in order:
            if remainder <= 0:
                break
            if player in winners:
                payouts[player] += 1
                remainder -= 1
    net = tuple(payouts[index] - int(contributions[index]) for index in range(2))
    if sum(payouts) != sum(contributions) or sum(net) != 0:
        raise AssertionError("settlement does not conserve chips")
    return Settlement(tuple(payouts), net, layers)


@lru_cache(maxsize=8_192)
def terminal_settlement(state: HoldemState, deal: Deal) -> Settlement:
    state.validate()
    if not state.terminal:
        raise ValueError("terminal settlement requires terminal state")
    deal.validate(state.board)
    if any(state.folded):
        ranks: tuple[tuple[int, ...] | None, tuple[int, ...] | None] = (None, None)
    else:
        if len(state.board) != 5:
            raise ValueError("showdown requires a complete five-card board")
        ranks = (
            best_hand_rank((*state.board, *deal.hole0)),
            best_hand_rank((*state.board, *deal.hole1)),
        )
    return settle_contributions(
        state.contributions,
        state.folded,
        ranks,
        odd_chip_order=(1 - state.button_player, state.button_player),
    )


def terminal_utility(state: HoldemState, deal: Deal, player: int) -> float:
    if player not in {0, 1}:
        raise ValueError("invalid player")
    return float(terminal_settlement(state, deal).net[player])


def terminal_utilities(state: HoldemState, deal: Deal) -> tuple[float, float]:
    """Return both zero-sum utilities from one validated settlement pass."""

    settlement = terminal_settlement(state, deal)
    return float(settlement.net[0]), float(settlement.net[1])


@lru_cache(maxsize=8_192)
def _information_set_key_cached(
    state: HoldemState, own: tuple[str, str]
) -> str:
    public_history = "/".join(
        f"{row.street}:{row.player}:{row.kind}:{row.to_amount}" for row in state.history
    )
    return "|".join(
        (
            state.street,
            ",".join(state.board),
            ",".join(sorted(own)),
            str(state.actor),
            str(state.pot),
            str(state.current_bet),
            public_history,
        )
    )


def information_set_key(state: HoldemState, deal: Deal, player: int) -> str:
    """Return an exact information-set key without caching hidden opposition.

    The public API retains ``Deal`` for compatibility, but the memoized pure
    function is keyed only by the public state and the acting player's own
    cards—the exact inputs used to construct the value.
    """

    if player not in {0, 1}:
        raise ValueError("invalid player")
    own = deal.hole0 if player == 0 else deal.hole1
    return _information_set_key_cached(state, own)


def _ratio_bucket(value: float) -> str:
    if value <= 0:
        return "zero"
    if value < 0.25:
        return "lt_025"
    if value < 0.50:
        return "025_050"
    if value < 1.00:
        return "050_100"
    if value < 2.00:
        return "100_200"
    return "ge_200"


_STRATEGY_SIZE_ANCHORS = (0.25, 0.33, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00)


def _strategy_action_base_label(state: HoldemState, action: Action) -> str:
    if action.kind not in {"bet", "raise"} or action.label == "jam":
        return action.label or action.kind
    if state.actor is None:
        raise ValueError("strategy action label requires a decision state")
    to_call = max(0, state.current_bet - state.street_contributions[state.actor])
    pot_after_call = max(1, state.pot + to_call)
    aggressive_increment = max(0, action.amount - to_call)
    fraction = aggressive_increment / pot_after_call
    anchor = min(_STRATEGY_SIZE_ANCHORS, key=lambda value: (abs(value - fraction), value))
    return f"{action.kind}_{anchor:.2f}pot"


@lru_cache(maxsize=8_192)
def _strategy_action_labels_cached(
    state: HoldemState, candidates: tuple[Action, ...]
) -> tuple[str, ...]:
    bases = tuple(_strategy_action_base_label(state, action) for action in candidates)
    labels = list(bases)
    for base in dict.fromkeys(bases):
        indices = [index for index, candidate in enumerate(bases) if candidate == base]
        if len(indices) == 1:
            continue
        ordered = sorted(
            indices,
            key=lambda index: (
                candidates[index].amount,
                candidates[index].kind,
                candidates[index].label,
                index,
            ),
        )
        for slot, index in enumerate(ordered, start=1):
            labels[index] = f"{base}_slot{slot}of{len(ordered)}"
    result = tuple(labels)
    if len(result) != len(set(result)):
        raise AssertionError("strategy action labels are not unique")
    return result


def strategy_action_labels(
    state: HoldemState, actions: Sequence[Action] | None = None
) -> tuple[str, ...]:
    """Return unique pot-relative labels for one complete legal action set.

    Several executable integer targets can legitimately round to the same
    pot-fraction anchor. A policy is a mapping keyed by label, so silently
    reusing that label would merge distinct actions and lose probability mass.
    Colliding anchors receive stable amount-ordered slots while singleton
    anchors retain their portable coarse label.
    """

    candidates = tuple(legal_actions(state) if actions is None else actions)
    if not candidates:
        return ()
    return _strategy_action_labels_cached(state, candidates)


def strategy_action_label(state: HoldemState, action: Action) -> str:
    """Return one member of the state's unique strategy action label set."""

    actions = legal_actions(state)
    try:
        index = actions.index(action)
    except ValueError as error:
        raise ValueError("strategy action label requires a legal action") from error
    return strategy_action_labels(state, actions)[index]


@lru_cache(maxsize=8_192)
def _strategy_history_key(state: HoldemState) -> str:
    """Encode the complete public action history without payoff-state buckets.

    Public actions, exact chip transfers, and reopening status are observed by
    both players.  Truncating or ratio-bucketing this sequence can merge states
    in which a player took a different earlier action or faces different pot
    odds, invalidating perfect-recall reasoning.  The full causal sequence is
    retained here; private opponent cards remain excluded.
    """

    selected = _history_replay_contexts(state)
    rows = []
    for row, pot_before, to_call_before, aggressive_increment in selected:
        rows.append(
            ":".join(
                (
                    row.street,
                    str(row.player),
                    row.kind,
                    str(row.amount),
                    str(row.to_amount),
                    str(int(row.full_raise)),
                    str(pot_before),
                    str(to_call_before),
                    str(aggressive_increment),
                )
            )
        )
    return "/".join(rows)


def _context_key(context: Mapping[str, Any] | None) -> str:
    if context is None:
        return "none"
    return sha256_json(context)[:16]


@lru_cache(maxsize=4_096)
def _canonical_public_board_key_cached(board: tuple[str, ...]) -> str:
    cards = normalize_cards(board)
    if not cards:
        return "none"
    return ",".join((*sorted(cards[:3]), *cards[3:]))


def _canonical_public_board_key(board: Sequence[str]) -> str:
    """Canonicalize the public board without erasing Turn/River chronology.

    The three Flop cards arrive simultaneously and are therefore sorted.  Turn
    and River identities stay in chronological order because they are public
    perfect-recall observations, not an unordered five-card set.
    """

    return _canonical_public_board_key_cached(tuple(board))


@lru_cache(maxsize=8_192)
def _strategy_hand_key(board: tuple[str, ...], own: tuple[str, str]) -> str:
    hand = hand_features(board, own)
    return ":".join(
        (
            str(hand["made_category"]),
            str(int(bool(hand["flush_draw"]))),
            str(int(bool(hand["open_ended_straight_draw"]))),
            str(int(bool(hand["gutshot"]))),
            str(int(bool(hand["pocket_pair"]))),
            str(hand["overcards"]),
            str(hand["preflop_class"]),
        )
    )


@lru_cache(maxsize=4_096)
def _strategy_board_key(board_cards: tuple[str, ...]) -> str:
    if not board_cards:
        return "none"
    board = board_features(board_cards)
    return ":".join(
        (
            str(board["pairedness"]),
            str(board["suit_texture"]),
            str(board["straight_window_gap_min"]),
            str(board["straight_window_count"]),
            str(board["highest_rank"] // 3),
        )
    )


@lru_cache(maxsize=8_192)
def _strategy_information_set_key_cached(
    state: HoldemState,
    own: tuple[str, str],
    player: int,
    context_identity: str,
    labels: tuple[str, ...],
) -> str:
    return "|".join(
        (
            STRATEGY_ABSTRACTION_VERSION,
            state.street,
            "BTN_SB" if player == state.button_player else "BB",
            f"actor:{state.actor}",
            f"context:{context_identity}",
            "payoff:"
            + ":".join(
                str(value)
                for value in (
                    state.pot,
                    state.current_bet,
                    *state.starting_stacks,
                    *state.stacks,
                    *state.contributions,
                    *state.street_contributions,
                    state.last_full_raise,
                    *[int(value) for value in state.acted_since_full_raise],
                    state.checks_in_row,
                )
            ),
            f"public:{_canonical_public_board_key(state.board)}",
            f"board:{_strategy_board_key(state.board)}",
            f"hand:{_strategy_hand_key(state.board, own)}",
            f"own:{','.join(sorted(own))}",
            f"history:{_strategy_history_key(state)}",
            f"legal:{','.join(labels)}",
        )
    )


def strategy_information_set_key(
    state: HoldemState,
    deal: Deal,
    player: int,
    context: Mapping[str, Any] | None = None,
    *,
    actions: Sequence[Action] | None = None,
    action_labels: Sequence[str] | None = None,
    context_key: str | None = None,
    deal_already_validated: bool = False,
) -> str:
    """Perfect-recall own-card/public-state key for approximate street banks.

    The key includes the acting player's exact private cards, complete public
    action history, and exact payoff-relevant chip state.  It never includes
    the opponent's private cards.  Feature fields remain available for
    explicitly gated downstream backoff, but the solver's primary key no
    longer merges states with different pot odds or remembered actions.
    Restricted exact anchors continue to use :func:`information_set_key`.
    """

    if player not in {0, 1}:
        raise ValueError("invalid player")
    if not deal_already_validated:
        deal.validate(state.board)
    labels = (
        tuple(action_labels)
        if action_labels is not None
        else strategy_action_labels(state, actions)
    )
    if len(labels) != len(set(labels)):
        raise ValueError("strategy information set requires unique action labels")
    return _strategy_information_set_key_cached(
        state,
        deal.hole0 if player == 0 else deal.hole1,
        player,
        context_key if context_key is not None else _context_key(context),
        labels,
    )


@lru_cache(maxsize=65_536)
def strategy_backoff_keys(key: str) -> tuple[str, ...]:
    """Pre-register progressively coarser policy-bank lookups.

    The exact solve still uses the first key.  Downstream policy rollouts may
    use the later, explicitly ordered projections when an exact abstraction
    row was not visited.  The legal-action signature is retained at every
    level, so probabilities can never be applied to a different action set.
    """

    parts = key.split("|")
    if len(parts) != 12 or parts[0] != STRATEGY_ABSTRACTION_VERSION:
        raise ValueError("unsupported strategy abstraction key")
    (
        version,
        street,
        position,
        actor,
        context,
        payoff,
        public,
        board,
        hand,
        own,
        history,
        legal,
    ) = parts
    if not public.startswith("public:"):
        raise ValueError("malformed exact public-board key")
    board_values = board.removeprefix("board:").split(":")
    hand_values = hand.removeprefix("hand:").split(":")
    if (board_values != ["none"] and len(board_values) != 5) or len(hand_values) != 7:
        raise ValueError("malformed strategy feature key")
    coarse_board = (
        "board:none"
        if board_values == ["none"]
        else "board:" + ":".join(board_values[:2] + ["*", "*", "*"])
    )
    broad_board = "board:none" if board_values == ["none"] else "board:*:*:*:*:*"
    coarse_hand = "hand:" + ":".join(hand_values[:4] + ["*", "*", "*"])
    broad_hand = "hand:" + ":".join([hand_values[0], "*", "*", "*", "*", "*", "*"])
    candidates = (
        key,
        "|".join((version, street, position, actor, context, payoff, public, board, hand, own, "history:*", legal)),
        "|".join((version, street, position, actor, context, payoff, "public:*", coarse_board, coarse_hand, "own:*", "history:*", legal)),
        "|".join((version, street, position, actor, "context:*", payoff, "public:*", coarse_board, coarse_hand, "own:*", "history:*", legal)),
        "|".join((version, street, position, actor, "context:*", "payoff:*", "public:*", broad_board, broad_hand, "own:*", "history:*", legal)),
        "|".join((version, street, position, actor, "context:*", "payoff:*", "public:*", broad_board, "hand:*:*:*:*:*:*:*", "own:*", "history:*", legal)),
    )
    return tuple(dict.fromkeys(candidates))
