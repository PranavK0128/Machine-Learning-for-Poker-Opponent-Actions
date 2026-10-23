from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

from .chance import legal_next_cards
from .common import with_self_hash
from .exact import ProjectedDeal
from .game import (
    Action,
    Deal,
    HoldemState,
    advance_public_card,
    apply_action,
    information_set_key,
    legal_actions,
    terminal_utility,
)


def _policy_row(
    policy: Mapping[str, Mapping[str, float]],
    key: str,
    actions: Sequence[Action],
) -> tuple[float, ...]:
    row = policy.get(key)
    if row is None:
        raise ValueError("restricted profile policy is missing a reachable information set")
    labels = tuple(action.label or action.kind for action in actions)
    if set(row) != set(labels):
        raise ValueError("restricted profile policy action membership mismatch")
    values = [max(0.0, float(row[label])) for label in labels]
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("restricted profile policy has no legal positive mass")
    return tuple(value / total for value in values)


class IndependentRestrictedProfileOracle:
    """Belief-state best-response oracle independent of CFR update code."""

    def __init__(self, initial_state: HoldemState, deals: Sequence[ProjectedDeal]):
        if initial_state.street not in {"turn", "river"}:
            raise ValueError("restricted profile oracle supports Turn/River")
        self.initial_state = initial_state
        self.deals = tuple(deals)
        self._terminal_cache: dict[tuple[HoldemState, Deal, int], float] = {}
        self._action_cache: dict[HoldemState, tuple[Action, ...]] = {}
        self._action_child_cache: dict[tuple[HoldemState, Action], HoldemState] = {}
        self._public_child_cache: dict[tuple[HoldemState, str], HoldemState] = {}
        self._legal_cards_cache: dict[
            tuple[tuple[str, ...], Deal], tuple[str, ...]
        ] = {}
        self._information_set_cache: dict[
            tuple[HoldemState, tuple[str, str], int], str
        ] = {}
        self._policy_row_cache: dict[
            tuple[str, tuple[str, ...]], tuple[float, ...]
        ] = {}
        self._policy_object_id: int | None = None
        self._progress_callback: Callable[[Mapping[str, Any]], None] | None = None
        self._oracle_walk_calls = 0
        self._completed_oracle_passes = 0
        self._last_progress_monotonic = 0.0

    def _oracle_progress(self, *, force: bool = False) -> None:
        if not force:
            self._oracle_walk_calls += 1
        if self._progress_callback is None:
            return
        # Avoid placing a clock call on every recursive node while still
        # keeping long individual profile/best-response passes observable.
        if not force and self._oracle_walk_calls % 4096:
            return
        now = time.monotonic()
        if not force and now - self._last_progress_monotonic < 15.0:
            return
        self._progress_callback(
            {
                "completed_oracle_passes": self._completed_oracle_passes,
                "total_oracle_passes": 4,
                "oracle_walk_calls": self._oracle_walk_calls,
                "intra_pass_heartbeat": not force,
            }
        )
        self._last_progress_monotonic = now

    def _actions(self, state: HoldemState) -> tuple[Action, ...]:
        actions = self._action_cache.get(state)
        if actions is None:
            actions = legal_actions(state)
            self._action_cache[state] = actions
        return actions

    def _action_child(self, state: HoldemState, action: Action) -> HoldemState:
        key = (state, action)
        child = self._action_child_cache.get(key)
        if child is None:
            child = apply_action(
                state,
                action,
                prevalidated_legal_actions=self._actions(state),
            )
            self._action_child_cache[key] = child
        return child

    def _public_child(self, state: HoldemState, card: str) -> HoldemState:
        key = (state, card)
        child = self._public_child_cache.get(key)
        if child is None:
            child = advance_public_card(state, card)
            self._public_child_cache[key] = child
        return child

    def _legal_cards(self, board: tuple[str, ...], deal: Deal) -> tuple[str, ...]:
        key = (board, deal)
        cards = self._legal_cards_cache.get(key)
        if cards is None:
            cards = legal_next_cards(board, deal)
            self._legal_cards_cache[key] = cards
        return cards

    def _information_set_key(
        self, state: HoldemState, deal: Deal, actor: int
    ) -> str:
        own = deal.hole0 if actor == 0 else deal.hole1
        cache_key = (state, own, actor)
        key = self._information_set_cache.get(cache_key)
        if key is None:
            key = information_set_key(state, deal, actor)
            self._information_set_cache[cache_key] = key
        return key

    def _normalized_policy_row(
        self,
        policy: Mapping[str, Mapping[str, float]],
        key: str,
        actions: Sequence[Action],
    ) -> tuple[float, ...]:
        policy_id = id(policy)
        if self._policy_object_id != policy_id:
            self._policy_row_cache.clear()
            self._policy_object_id = policy_id
        labels = tuple(action.label or action.kind for action in actions)
        cache_key = (key, labels)
        probabilities = self._policy_row_cache.get(cache_key)
        if probabilities is None:
            probabilities = _policy_row(policy, key, actions)
            self._policy_row_cache[cache_key] = probabilities
        return probabilities

    def _terminal_value(
        self,
        state: HoldemState,
        player: int,
        own_combo: tuple[str, str],
        opponent_weights: Mapping[tuple[str, str], float],
    ) -> float:
        result = 0.0
        for opponent_combo, weight in opponent_weights.items():
            deal = (
                Deal(own_combo, opponent_combo)
                if player == 0
                else Deal(opponent_combo, own_combo)
            )
            key = (state, deal, player)
            value = self._terminal_cache.get(key)
            if value is None:
                value = terminal_utility(state, deal, player)
                self._terminal_cache[key] = value
            result += weight * value
        return result

    def _walk(
        self,
        state: HoldemState,
        *,
        player: int,
        own_combo: tuple[str, str],
        opponent_weights: Mapping[tuple[str, str], float],
        policy: Mapping[str, Mapping[str, float]],
        best_response: bool,
    ) -> float:
        self._oracle_progress()
        if state.terminal:
            return self._terminal_value(state, player, own_combo, opponent_weights)
        if state.actor is None or state.all_in_runout:
            if state.street == "preflop":
                raise ValueError("restricted Turn oracle cannot traverse a Flop")
            # Candidate membership depends on the hidden opponent combo. Build
            # each public branch by propagating only compatible beliefs with the
            # exact per-deal 1/(52-board-4) chance probability.
            result = 0.0
            for card in sorted({
                candidate
                for opponent_combo in opponent_weights
                for candidate in self._legal_cards(
                    state.board,
                    Deal(own_combo, opponent_combo)
                    if player == 0
                    else Deal(opponent_combo, own_combo),
                )
            }):
                child_weights = {}
                for opponent_combo, weight in opponent_weights.items():
                    deal = (
                        Deal(own_combo, opponent_combo)
                        if player == 0
                        else Deal(opponent_combo, own_combo)
                    )
                    cards = self._legal_cards(state.board, deal)
                    if card in cards:
                        child_weights[opponent_combo] = weight / len(cards)
                if child_weights:
                    result += self._walk(
                        self._public_child(state, card),
                        player=player,
                        own_combo=own_combo,
                        opponent_weights=child_weights,
                        policy=policy,
                        best_response=best_response,
                    )
            return result

        actor = state.actor
        actions = self._actions(state)
        if actor == player:
            representative_opponent = next(iter(opponent_weights))
            deal = (
                Deal(own_combo, representative_opponent)
                if player == 0
                else Deal(representative_opponent, own_combo)
            )
            key = self._information_set_key(state, deal, player)
            values = [
                self._walk(
                    self._action_child(state, action),
                    player=player,
                    own_combo=own_combo,
                    opponent_weights=opponent_weights,
                    policy=policy,
                    best_response=best_response,
                )
                for action in actions
            ]
            if best_response:
                return max(values)
            probabilities = self._normalized_policy_row(policy, key, actions)
            return math.fsum(probability * value for probability, value in zip(probabilities, values))

        total = 0.0
        for action_index, action in enumerate(actions):
            child_weights = {}
            for opponent_combo, weight in opponent_weights.items():
                deal = (
                    Deal(own_combo, opponent_combo)
                    if player == 0
                    else Deal(opponent_combo, own_combo)
                )
                key = self._information_set_key(state, deal, actor)
                probability = self._normalized_policy_row(
                    policy, key, actions
                )[action_index]
                if probability > 0:
                    child_weights[opponent_combo] = weight * probability
            if child_weights:
                total += self._walk(
                    self._action_child(state, action),
                    player=player,
                    own_combo=own_combo,
                    opponent_weights=child_weights,
                    policy=policy,
                    best_response=best_response,
                )
        return total

    def value(
        self,
        player: int,
        policy: Mapping[str, Mapping[str, float]],
        *,
        best_response: bool,
    ) -> float:
        beliefs: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for row in self.deals:
            own = row.deal.hole0 if player == 0 else row.deal.hole1
            opponent = row.deal.hole1 if player == 0 else row.deal.hole0
            beliefs.setdefault(own, {})[opponent] = (
                beliefs.setdefault(own, {}).get(opponent, 0.0) + row.probability
            )
        return math.fsum(
            self._walk(
                self.initial_state,
                player=player,
                own_combo=own,
                opponent_weights=opponents,
                policy=policy,
                best_response=best_response,
            )
            for own, opponents in sorted(beliefs.items())
        )

    def evaluate(
        self,
        policy: Mapping[str, Mapping[str, float]],
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._progress_callback = progress_callback
        self._oracle_walk_calls = 0
        self._completed_oracle_passes = 0
        self._last_progress_monotonic = time.monotonic()
        values = []
        best = []
        for player in (0, 1):
            values.append(self.value(player, policy, best_response=False))
            self._completed_oracle_passes = len(values)
            self._oracle_progress(force=True)
        for player in (0, 1):
            best.append(self.value(player, policy, best_response=True))
            self._completed_oracle_passes = 2 + len(best)
            self._oracle_progress(force=True)
        self._progress_callback = None
        nashconv = (best[0] - values[0]) + (best[1] - values[1])
        return with_self_hash(
            {
                "schema_version": "stage21_independent_restricted_profile_oracle_v2",
                "profile_values": values,
                "best_response_values": best,
                "nashconv": nashconv,
                "exploitability": nashconv / 2.0,
                "zero_sum_error": abs(sum(values)),
                "finite": all(math.isfinite(value) for value in (*values, *best, nashconv)),
                "claim_scope": "restricted_projected_game_only",
                "universal_equilibrium_claim": False,
                "oracle_independent_of_solver_update_code": True,
            }
        )
