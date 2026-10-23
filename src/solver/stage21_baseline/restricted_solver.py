from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .chance import legal_next_cards
from .common import (
    read_self_hashed_json_streamed_mapping,
    sha256_json,
    with_self_hash,
    write_self_hashed_json_stream,
)
from .exact import (
    EXACT_CONTRACT,
    EXACT_CONTRACT_SHA256,
    ProjectedDeal,
    projected_deal_distribution,
)
from .game import (
    Deal,
    HoldemState,
    advance_public_card,
    apply_action,
    information_set_key,
    legal_actions,
    terminal_utility,
)
from .policy_store import DiskBackedPolicy, DiskPolicyBuilder
from .ranges import RangeDistribution
from .solver import Node, regret_matching


class FullChanceRestrictedCFR:
    """CFR+/DCFR over an explicitly projected Turn/River game.

    Private chance is the predeclared top-mass projection and every legal public
    River card is enumerated. The result is exact only for that restricted game;
    the class deliberately refuses Preflop/Flop and universal-equilibrium claims.
    """

    def __init__(
        self,
        *,
        root_id: str,
        root_semantic_sha256: str,
        initial_state: HoldemState,
        first_range: RangeDistribution,
        second_range: RangeDistribution,
        algorithm: str,
        evaluation_seed: int,
        task_semantic_sha256: str,
        solver_continuation_semantic_sha256: str | None = None,
        combos_per_player: int = 4,
        checkpoint_io_lock_path: Path | None = None,
        checkpoint_spool_root: Path | None = None,
    ) -> None:
        if initial_state.street not in {"turn", "river"}:
            raise ValueError("restricted full-chance solver supports Turn/River only")
        if algorithm not in {"cfr_plus", "dcfr"}:
            raise ValueError("invalid exact algorithm")
        self.root_id = root_id
        self.root_semantic_sha256 = root_semantic_sha256
        self.initial_state = initial_state
        self.algorithm = algorithm
        self.evaluation_seed = int(evaluation_seed)
        if not isinstance(task_semantic_sha256, str) or len(task_semantic_sha256) != 64:
            raise ValueError("restricted solver requires a SHA-256 task identity")
        self.task_semantic_sha256 = task_semantic_sha256
        if solver_continuation_semantic_sha256 is None:
            solver_continuation_semantic_sha256 = task_semantic_sha256
        if (
            not isinstance(solver_continuation_semantic_sha256, str)
            or len(solver_continuation_semantic_sha256) != 64
        ):
            raise ValueError(
                "restricted solver requires a SHA-256 continuation identity"
            )
        self.solver_continuation_semantic_sha256 = (
            solver_continuation_semantic_sha256
        )
        self.combos_per_player = combos_per_player
        self.checkpoint_io_lock_path = checkpoint_io_lock_path
        self.checkpoint_spool_root = checkpoint_spool_root
        self.deals: tuple[ProjectedDeal, ...] = projected_deal_distribution(
            first_range, second_range, combos_per_player=combos_per_player
        )
        order_ids = tuple(
            int(value) for value in EXACT_CONTRACT["deterministic_numerical_order_ids"]
        )
        if self.evaluation_seed not in order_ids:
            raise ValueError("restricted solver requires a preregistered numerical order ID")
        self.numerical_order_rank = order_ids.index(self.evaluation_seed)
        planned_offsets = tuple(
            rank * len(self.deals) // len(order_ids) for rank in range(len(order_ids))
        )
        if len(set(planned_offsets)) != len(order_ids):
            raise ValueError("projected game is too small for three distinct numerical orders")
        self.deal_order_offset = planned_offsets[self.numerical_order_rank]
        self._ordered_deals = (
            self.deals[self.deal_order_offset :]
            + self.deals[: self.deal_order_offset]
        )
        self.optimization_stream_identity_sha256 = sha256_json(
            {
                "solver_continuation_semantic_sha256": (
                    self.solver_continuation_semantic_sha256
                ),
                "exact_contract_sha256": EXACT_CONTRACT_SHA256,
                "algorithm": self.algorithm,
                "numerical_order_id": self.evaluation_seed,
                "numerical_order_rank": self.numerical_order_rank,
                "deal_order_offset": self.deal_order_offset,
            }
        )
        self.iteration = 0
        self.nodes: dict[str, Node] = {}
        self._terminal_cache: dict[tuple[HoldemState, Deal, int], float] = {}
        self._action_cache: dict[HoldemState, tuple[Any, ...]] = {}
        self._action_child_cache: dict[tuple[HoldemState, Any], HoldemState] = {}
        self._public_child_cache: dict[tuple[HoldemState, str], HoldemState] = {}
        self._public_cards_cache: dict[tuple[tuple[str, ...], Deal], tuple[str, ...]] = {}
        self._information_set_cache: dict[
            tuple[HoldemState, tuple[str, str], int], str
        ] = {}

    def _node(self, key: str, actions) -> Node:
        labels = tuple(action.label or action.kind for action in actions)
        node = self.nodes.get(key)
        if node is None:
            node = Node(labels)
            self.nodes[key] = node
        elif node.actions != labels:
            raise RuntimeError("restricted exact legal-action drift")
        return node

    def _actions(self, state: HoldemState):
        actions = self._action_cache.get(state)
        if actions is None:
            actions = legal_actions(state)
            self._action_cache[state] = actions
        return actions

    def _action_child(self, state: HoldemState, action):
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

    def _public_cards(self, state: HoldemState, deal: Deal) -> tuple[str, ...]:
        key = (state.board, deal)
        cards = self._public_cards_cache.get(key)
        if cards is None:
            cards = legal_next_cards(state.board, deal)
            self._public_cards_cache[key] = cards
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

    def _public_child(self, state: HoldemState, card: str):
        key = (state, card)
        child = self._public_child_cache.get(key)
        if child is None:
            child = advance_public_card(state, card)
            self._public_child_cache[key] = child
        return child

    def _discount_dcfr(self) -> None:
        if self.algorithm != "dcfr":
            return
        t = float(self.iteration)
        positive = t**1.5 / (t**1.5 + 1.0)
        negative = 0.5
        average = (t / (t + 1.0)) ** 2.0
        for node in self.nodes.values():
            node.regrets = [
                value * (positive if value >= 0 else negative) for value in node.regrets
            ]
            node.strategy_sum = [value * average for value in node.strategy_sum]
            node.checkpoint_dirty = True

    def _walk(
        self,
        state: HoldemState,
        deal: Deal,
        *,
        update_player: int,
        chance_probability: float,
        reach: tuple[float, float],
        regret_delta: dict[str, list[float]],
        average_delta: dict[str, list[float]],
    ) -> float:
        if state.terminal:
            cache_key = (state, deal, update_player)
            value = self._terminal_cache.get(cache_key)
            if value is None:
                value = terminal_utility(state, deal, update_player)
                self._terminal_cache[cache_key] = value
            return value
        if state.actor is None or state.all_in_runout:
            if state.street == "preflop":
                raise ValueError("restricted exact solver does not traverse physical Flops")
            cards = self._public_cards(state, deal)
            probability = 1.0 / len(cards)
            return math.fsum(
                probability
                * self._walk(
                    self._public_child(state, card),
                    deal,
                    update_player=update_player,
                    chance_probability=chance_probability * probability,
                    reach=reach,
                    regret_delta=regret_delta,
                    average_delta=average_delta,
                )
                for card in cards
            )
        actor = state.actor
        actions = self._actions(state)
        key = self._information_set_key(state, deal, actor)
        node = self._node(key, actions)
        policy = node.policy()
        node.visits += 1
        node.checkpoint_dirty = True
        values = []
        for index, action in enumerate(actions):
            next_reach = list(reach)
            next_reach[actor] *= policy[index]
            values.append(
                self._walk(
                    self._action_child(state, action),
                    deal,
                    update_player=update_player,
                    chance_probability=chance_probability,
                    reach=tuple(next_reach),
                    regret_delta=regret_delta,
                    average_delta=average_delta,
                )
            )
        value = math.fsum(probability * item for probability, item in zip(policy, values))
        if actor == update_player:
            regrets = regret_delta.setdefault(key, [0.0] * len(actions))
            averages = average_delta.setdefault(key, [0.0] * len(actions))
            for index, action_value in enumerate(values):
                regrets[index] += chance_probability * reach[1 - actor] * (action_value - value)
                averages[index] += (
                    chance_probability
                    * reach[actor]
                    * (float(self.iteration) if self.algorithm == "cfr_plus" else 1.0)
                    * policy[index]
                )
        return value

    def step(self) -> None:
        self.iteration += 1
        regret_delta: dict[str, list[float]] = {}
        average_delta: dict[str, list[float]] = {}
        # This is a deterministic accumulation-order stability axis, not an
        # independent random solve. Batch application preserves the same full-
        # chance projected game for every order ID.
        for update_player in (0, 1):
            for row in self._ordered_deals:
                self._walk(
                    self.initial_state,
                    row.deal,
                    update_player=update_player,
                    chance_probability=row.probability,
                    reach=(1.0, 1.0),
                    regret_delta=regret_delta,
                    average_delta=average_delta,
                )
        for key, deltas in regret_delta.items():
            node = self.nodes[key]
            for index, delta in enumerate(deltas):
                node.regrets[index] += delta
                if self.algorithm == "cfr_plus":
                    node.regrets[index] = max(0.0, node.regrets[index])
            for index, delta in enumerate(average_delta[key]):
                node.strategy_sum[index] += delta
            node.checkpoint_dirty = True
        # Apply DCFR to the just-completed iteration. This preserves the
        # published t-based weighting instead of shifting every contribution
        # forward by one iteration.
        self._discount_dcfr()

    def run(
        self,
        target_iterations: int,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        progress_interval_seconds: float = 60.0,
        finalize_result: bool = True,
    ) -> dict[str, Any] | None:
        if target_iterations < self.iteration:
            raise ValueError("target exact iteration precedes checkpoint")
        last_progress = time.monotonic()
        while self.iteration < target_iterations:
            self.step()
            if progress_callback is not None:
                now = time.monotonic()
                if (
                    self.iteration == target_iterations
                    or now - last_progress >= progress_interval_seconds
                ):
                    progress_callback(
                        {
                            "event": "solver_progress",
                            "iteration": self.iteration,
                            "target_iteration": target_iterations,
                            "node_count": len(self.nodes),
                        }
                    )
                    last_progress = now
        return self.result() if finalize_result else None

    def policy(self) -> dict[str, dict[str, float]]:
        return {
            key: dict(zip(self.nodes[key].actions, self.nodes[key].average_policy()))
            for key in sorted(self.nodes)
        }

    def _result_from_policy(
        self,
        policy: Mapping[str, Mapping[str, float]],
        *,
        node_count: int,
        projected_deal_count: int,
        policy_sha256: str | None = None,
    ) -> dict[str, Any]:
        return with_self_hash(
            {
                "schema_version": "stage21_full_chance_restricted_result_v6",
                "exact_contract_sha256": EXACT_CONTRACT_SHA256,
                "root_id": self.root_id,
                "root_semantic_sha256": self.root_semantic_sha256,
                "algorithm": self.algorithm,
                "seed": self.evaluation_seed,
                "numerical_order_id": self.evaluation_seed,
                "task_semantic_sha256": self.task_semantic_sha256,
                "solver_continuation_semantic_sha256": (
                    self.solver_continuation_semantic_sha256
                ),
                "replication_axis": "numerical_accumulation_order_stability",
                "independent_solver_random_seed_claim": False,
                "deal_order_offset": self.deal_order_offset,
                "numerical_order_rank": self.numerical_order_rank,
                "optimization_stream_identity_sha256": (
                    self.optimization_stream_identity_sha256
                ),
                "iterations": self.iteration,
                "combos_per_player": self.combos_per_player,
                "projected_deal_count": projected_deal_count,
                "full_public_runout_enumeration": True,
                "node_count": node_count,
                "policy_sha256": (
                    sha256_json(policy) if policy_sha256 is None else policy_sha256
                ),
                "fresh_regrets": True,
                "fresh_strategy_sum": True,
                "on_policy_sampling": True,
                "importance_ratio": 1.0,
                "claim_scope": "restricted_projected_game_only",
                "universal_equilibrium_claim": False,
            }
        )

    def result(
        self,
        *,
        materialized_policy: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        policy = self.policy() if materialized_policy is None else materialized_policy
        return self._result_from_policy(
            policy,
            node_count=len(self.nodes),
            projected_deal_count=len(self.deals),
        )

    def finalize_policy_and_release(
        self,
        *,
        policy_store_path: Path | None = None,
    ) -> tuple[
        dict[str, Any], Mapping[str, Mapping[str, float]], dict[str, Any]
    ]:
        """Finalize a restricted policy while releasing consumed solver state."""

        node_count = len(self.nodes)
        deal_count = len(self.deals)
        vector_values = sum(
            len(node.regrets) + len(node.strategy_sum)
            for node in self.nodes.values()
        )
        cache_entries = {
            "terminal": len(self._terminal_cache),
            "actions": len(self._action_cache),
            "action_children": len(self._action_child_cache),
            "public_children": len(self._public_child_cache),
            "public_cards": len(self._public_cards_cache),
            "information_sets": len(self._information_set_cache),
        }

        # No cache participates in average-policy construction.  Releasing it
        # first avoids carrying the complete restricted-game cache into the
        # final policy allocation.
        self._terminal_cache.clear()
        self._action_cache.clear()
        self._action_child_cache.clear()
        self._public_child_cache.clear()
        self._public_cards_cache.clear()
        self._information_set_cache.clear()
        self._ordered_deals = ()

        builder = (
            DiskPolicyBuilder(policy_store_path, require_sorted_input=False)
            if policy_store_path is not None
            else None
        )
        policy: Mapping[str, Mapping[str, float]]
        materialized_policy: dict[str, dict[str, float]] | None = (
            {} if builder is None else None
        )
        try:
            while self.nodes:
                key, node = self.nodes.popitem()
                row = dict(zip(node.actions, node.average_policy()))
                if builder is None:
                    assert materialized_policy is not None
                    materialized_policy[key] = row
                else:
                    builder.add(key, row)
                del node
            if builder is None:
                assert materialized_policy is not None
                policy = materialized_policy
                policy_sha256 = sha256_json(policy)
            else:
                store = builder.finalize()
                policy_sha256 = str(store["policy_sha256"])
                policy = DiskBackedPolicy(
                    Path(str(store["path"])),
                    expected_policy_sha256=policy_sha256,
                    expected_row_count=int(store["row_count"]),
                )
        except Exception:
            if builder is not None:
                builder.abort()
            raise

        solver_result = self._result_from_policy(
            policy,
            node_count=node_count,
            projected_deal_count=deal_count,
            policy_sha256=policy_sha256,
        )
        self.initial_state = None
        self.deals = ()
        release = {
            "schema_version": "stage21_solver_training_state_release_v1",
            "released_node_count": node_count,
            "released_vector_value_count": vector_values,
            "released_projected_deal_reference_count": deal_count,
            "released_internal_cache_entries": cache_entries,
            "progressive_node_release_during_policy_materialization": True,
            "nodes_remaining_when_policy_completed": len(self.nodes),
            "full_policy_and_full_training_nodes_simultaneously_retained": False,
            "policy_and_result_retained_outside_solver": True,
            "final_policy_disk_backed": builder is not None,
            "complete_policy_materialized_in_python": builder is None,
        }
        return solver_result, policy, release

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "stage21_full_chance_restricted_checkpoint_v6",
            "exact_contract_sha256": EXACT_CONTRACT_SHA256,
            "root_id": self.root_id,
            "root_semantic_sha256": self.root_semantic_sha256,
            "algorithm": self.algorithm,
            "seed": self.evaluation_seed,
            "numerical_order_id": self.evaluation_seed,
            "originating_task_semantic_sha256": self.task_semantic_sha256,
            "solver_continuation_semantic_sha256": (
                self.solver_continuation_semantic_sha256
            ),
            "replication_axis": "numerical_accumulation_order_stability",
            "deal_order_offset": self.deal_order_offset,
            "numerical_order_rank": self.numerical_order_rank,
            "optimization_stream_identity_sha256": (
                self.optimization_stream_identity_sha256
            ),
            "combos_per_player": self.combos_per_player,
            "iteration": self.iteration,
        }

    def _checkpoint_body(self) -> dict[str, Any]:
        return {
            **self._checkpoint_metadata(),
            "nodes": {
                key: self.nodes[key].to_json() for key in sorted(self.nodes)
            },
        }

    def checkpoint_document(self) -> dict[str, Any]:
        return with_self_hash(self._checkpoint_body())

    def save_checkpoint(self, path: Path) -> dict[str, Any]:
        dirty_node_count = sum(node.checkpoint_dirty for node in self.nodes.values())
        receipt = write_self_hashed_json_stream(
            path,
            document=self._checkpoint_metadata(),
            streamed_mappings={
                "nodes": (
                    (key, self.nodes[key].to_json())
                    for key in sorted(self.nodes)
                )
            },
            binding_fields=(
                "schema_version",
                "exact_contract_sha256",
                "root_id",
                "root_semantic_sha256",
                "algorithm",
                "seed",
                "solver_continuation_semantic_sha256",
                "iteration",
            ),
            io_lock_path=self.checkpoint_io_lock_path,
            spool_root=self.checkpoint_spool_root,
        )
        for node in self.nodes.values():
            node.checkpoint_dirty = False
        return {
            **receipt,
            "schema_version": "stage21_checkpoint_publication_receipt_v2",
            "checkpoint_node_count": len(self.nodes),
            "checkpoint_dirty_node_count": dirty_node_count,
            "checkpoint_iteration": self.iteration,
            "lazy_dcfr_state_persisted_without_full_node_materialization": False,
        }

    def load_checkpoint(self, path: Path) -> None:
        self.nodes.clear()

        def load_node(key: str, row: Mapping[str, Any]) -> None:
            if key in self.nodes:
                raise ValueError("checkpoint contains a duplicate node key")
            self.nodes[key] = Node(
                actions=tuple(row["actions"]),
                regrets=[float(value) for value in row["regrets"]],
                strategy_sum=[float(value) for value in row["strategy_sum"]],
                visits=int(row["visits"]),
                checkpoint_dirty=False,
            )

        streamed = read_self_hashed_json_streamed_mapping(
            path,
            streamed_mapping_key="nodes",
            on_member=load_node,
        )
        document = streamed["document"]
        expected = {
            "exact_contract_sha256": EXACT_CONTRACT_SHA256,
            "root_id": self.root_id,
            "root_semantic_sha256": self.root_semantic_sha256,
            "algorithm": self.algorithm,
            "seed": self.evaluation_seed,
            "numerical_order_id": self.evaluation_seed,
            "solver_continuation_semantic_sha256": (
                self.solver_continuation_semantic_sha256
            ),
            "deal_order_offset": self.deal_order_offset,
            "numerical_order_rank": self.numerical_order_rank,
            "optimization_stream_identity_sha256": (
                self.optimization_stream_identity_sha256
            ),
            "combos_per_player": self.combos_per_player,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"restricted exact checkpoint binding mismatch: {key}")
        self.iteration = int(document["iteration"])

    def release_training_state(self) -> dict[str, Any]:
        deal_count = len(self.deals)
        node_count = len(self.nodes)
        cache_entries = {
            "terminal": len(self._terminal_cache),
            "actions": len(self._action_cache),
            "action_children": len(self._action_child_cache),
            "public_children": len(self._public_child_cache),
            "public_cards": len(self._public_cards_cache),
            "information_sets": len(self._information_set_cache),
        }
        self.nodes.clear()
        self._terminal_cache.clear()
        self._action_cache.clear()
        self._action_child_cache.clear()
        self._public_child_cache.clear()
        self._public_cards_cache.clear()
        self._information_set_cache.clear()
        self._ordered_deals = ()
        self.deals = ()
        return {
            "schema_version": "stage21_solver_training_state_release_v1",
            "released_node_count": node_count,
            "released_projected_deal_reference_count": deal_count,
            "released_internal_cache_entries": cache_entries,
            "policy_and_result_retained_outside_solver": True,
        }
