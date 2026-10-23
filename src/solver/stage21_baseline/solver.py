from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .chance import legal_next_cards, sample_flop
from .common import (
    STRATEGY_ABSTRACTION_VERSION,
    read_self_hashed_json_streamed_mapping,
    sha256_json,
    write_self_hashed_json_stream,
    with_self_hash,
)
from .continuation import ContinuationAuthority
from .policy_store import DiskBackedPolicy, DiskPolicyBuilder
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
    strategy_information_set_key,
)
from .ranges import JointRangeSampler


SOLVER_CONTRACT = {
    "schema_version": "stage21_external_sampling_solver_contract_v8",
    "algorithms": ["cfr_plus", "dcfr"],
    "sampling": "on_policy_external_sampling",
    "alternating_player_updates": True,
    "private_chance": "joint_player_specific_blocker_conditioned_posterior",
    "public_chance": "uniform_physical_cards_conditional_on_fixed_private_deal",
    "off_policy_importance_correction": "target_probability_over_proposal_probability",
    "current_proposal": "target_distribution_so_ratio_is_exactly_one",
    "cfr_plus_regrets": "nonnegative_after_each_update",
    "dcfr_parameters": {"alpha": 1.5, "beta": 0.0, "gamma": 2.0},
    "average_strategy_estimator": "two_player_external_sampling_opponent_turn_stochastically_weighted",
    "cfr_plus_average_strategy": "linear_iteration_weighted_stochastic_increment",
    "dcfr_average_strategy": "unit_stochastic_increment_then_gamma_discount",
    "average_strategy_updated_on_traverser_nodes": False,
    "dcfr_sparse_discounting": "exact_lazy_product_through_every_elapsed_iteration",
    "dcfr_positive_prefix_storage": "scalar_current_previous_with_per_node_anchor",
    "dcfr_touched_node_bookkeeping": "reused_node_reference_lists_with_epoch_markers",
    "depth_limited_leaf": "hand_specific_preregistered_multifamily_policy_rollout",
    "scalar_family_selection": False,
    "leaf_residual_recentering": False,
    "range_context_bound_to_strategy_and_continuation": True,
    "opponent_private_cards_in_strategy_key": False,
    "strategy_abstraction": STRATEGY_ABSTRACTION_VERSION,
    "strategy_abstraction_semantics": (
        "perfect_recall_own_card_exact_public_board_payoff_state"
    ),
    "named_rng_streams": ["private_deal", "public_chance", "action_selection"],
    "rng_streams_are_independently_seeded": True,
    "solver_continuation_and_package_identity_bound_to_checkpoint": True,
    "execution_budget_and_evaluation_sample_size_excluded_from_solver_continuation_identity": True,
}
SOLVER_CONTRACT_SHA256 = sha256_json(SOLVER_CONTRACT)


def regret_matching(regrets: Sequence[float]) -> tuple[float, ...]:
    positive = [max(0.0, float(value)) for value in regrets]
    total = math.fsum(positive)
    if total <= 0:
        return tuple(1.0 / len(positive) for _ in positive)
    return tuple(value / total for value in positive)


def _sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if threshold <= running:
            return index
    return len(probabilities) - 1


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


@dataclass
class Node:
    actions: tuple[str, ...]
    regrets: list[float] = field(default_factory=list)
    strategy_sum: list[float] = field(default_factory=list)
    visits: int = 0
    last_regret_discount_iteration: int = 0
    last_strategy_discount_iteration: int = 0
    regret_positive_log_prefix_anchor: float = 0.0
    checkpoint_dirty: bool = True
    regret_touch_epoch: int = -1
    average_touch_epoch: int = -1

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("solver node requires at least one action")
        if len(self.actions) != len(set(self.actions)):
            raise ValueError("solver node action labels must be unique")
        if not self.regrets:
            self.regrets = [0.0] * len(self.actions)
        if not self.strategy_sum:
            self.strategy_sum = [0.0] * len(self.actions)
        if len(self.regrets) != len(self.actions) or len(self.strategy_sum) != len(self.actions):
            raise ValueError("node vectors do not align")

    def policy(self) -> tuple[float, ...]:
        return regret_matching(self.regrets)

    def average_policy(self) -> tuple[float, ...]:
        total = math.fsum(self.strategy_sum)
        if total <= 0:
            return tuple(1.0 / len(self.actions) for _ in self.actions)
        return tuple(value / total for value in self.strategy_sum)

    def to_json(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "regrets": list(self.regrets),
            "strategy_sum": list(self.strategy_sum),
            "visits": self.visits,
            "last_regret_discount_iteration": self.last_regret_discount_iteration,
            "last_strategy_discount_iteration": self.last_strategy_discount_iteration,
            "regret_positive_log_prefix_anchor": (
                self.regret_positive_log_prefix_anchor
            ),
        }


class ExternalSamplingSolver:
    def __init__(
        self,
        *,
        root_id: str,
        root_semantic_sha256: str,
        initial_state: HoldemState,
        joint_sampler: JointRangeSampler,
        algorithm: str,
        seed: int,
        task_semantic_sha256: str,
        solver_continuation_semantic_sha256: str | None = None,
        depth_limit: str | None = None,
        continuation_authority: ContinuationAuthority | None = None,
        strategy_context: Mapping[str, Any] | None = None,
        checkpoint_io_lock_path: Path | None = None,
        checkpoint_spool_root: Path | None = None,
    ) -> None:
        if algorithm not in {"cfr_plus", "dcfr"}:
            raise ValueError("algorithm must be cfr_plus or dcfr")
        if depth_limit not in {None, "turn", "flop"}:
            raise ValueError("invalid depth limit")
        if (depth_limit is None) != (continuation_authority is None):
            raise ValueError("depth-limited solvers require exactly one continuation authority")
        initial_state.validate()
        self.root_id = root_id
        self.root_semantic_sha256 = root_semantic_sha256
        self.initial_state = initial_state
        self.joint_sampler = joint_sampler
        self.algorithm = algorithm
        self.seed = int(seed)
        if not isinstance(task_semantic_sha256, str) or len(task_semantic_sha256) != 64:
            raise ValueError("solver requires a SHA-256 task identity")
        self.task_semantic_sha256 = task_semantic_sha256
        if solver_continuation_semantic_sha256 is None:
            solver_continuation_semantic_sha256 = task_semantic_sha256
        if (
            not isinstance(solver_continuation_semantic_sha256, str)
            or len(solver_continuation_semantic_sha256) != 64
        ):
            raise ValueError("solver requires a SHA-256 continuation identity")
        self.solver_continuation_semantic_sha256 = (
            solver_continuation_semantic_sha256
        )
        self.depth_limit = depth_limit
        self.continuation_authority = continuation_authority
        self.strategy_context = dict(strategy_context) if strategy_context is not None else None
        self.strategy_context_sha256 = sha256_json(self.strategy_context)
        self.checkpoint_io_lock_path = checkpoint_io_lock_path
        self.checkpoint_spool_root = checkpoint_spool_root
        self.rng_stream_seeds = {
            name: int(
                sha256_json(
                    {
                        "root_id": self.root_id,
                        "algorithm": self.algorithm,
                        "seed": self.seed,
                        "solver_continuation_semantic_sha256": (
                            self.solver_continuation_semantic_sha256
                        ),
                        "stream": name,
                    }
                )[:16],
                16,
            )
            for name in ("private_deal", "public_chance", "action_selection")
        }
        self.deal_rng = random.Random(self.rng_stream_seeds["private_deal"])
        self.public_chance_rng = random.Random(
            self.rng_stream_seeds["public_chance"]
        )
        self.action_rng = random.Random(self.rng_stream_seeds["action_selection"])
        self.optimization_stream_identity_sha256 = sha256_json(
            {
                "solver_continuation_semantic_sha256": (
                    self.solver_continuation_semantic_sha256
                ),
                "rng_stream_seeds": self.rng_stream_seeds,
                "solver_contract_sha256": SOLVER_CONTRACT_SHA256,
            }
        )
        self.nodes: dict[str, Node] = {}
        self.iteration = 0
        self.sampled_utility_sum = [0.0, 0.0]
        self._dcfr_positive_log_prefix_iteration = 0
        self._dcfr_positive_log_prefix_value = 0.0
        self._dcfr_previous_positive_log_prefix_value = 0.0
        self._regret_touched_nodes: list[Node] = []
        self._average_touched_nodes: list[Node] = []
        self._touch_epoch = 0

    def _node(self, key: str, labels: Sequence[str]) -> Node:
        labels = tuple(labels)
        node = self.nodes.get(key)
        if node is None:
            node = Node(
                labels,
                last_regret_discount_iteration=max(0, self.iteration - 1),
                last_strategy_discount_iteration=max(0, self.iteration - 1),
                regret_positive_log_prefix_anchor=(
                    self._dcfr_previous_positive_log_prefix_value
                    if self.algorithm == "dcfr"
                    else 0.0
                ),
            )
            self.nodes[key] = node
        elif node.actions != labels:
            raise RuntimeError(f"information-set action mismatch for {key}")
        return node

    def _depth_value(self, state: HoldemState, deal: Deal, player: int) -> float | None:
        if self.depth_limit is None or state.street != self.depth_limit:
            return None
        assert self.continuation_authority is not None
        return self.continuation_authority.resolve_preregistered_mixture(
            state, deal, context=self.strategy_context
        )[player]

    def _ensure_dcfr_prefix(self, iteration: int) -> None:
        if self.algorithm != "dcfr":
            return
        if iteration < self._dcfr_positive_log_prefix_iteration - 1:
            raise RuntimeError("DCFR prefix request precedes retained scalar anchors")
        while self._dcfr_positive_log_prefix_iteration < iteration:
            t = float(self._dcfr_positive_log_prefix_iteration + 1)
            factor = t**1.5 / (t**1.5 + 1.0)
            self._dcfr_previous_positive_log_prefix_value = (
                self._dcfr_positive_log_prefix_value
            )
            self._dcfr_positive_log_prefix_value += math.log(factor)
            self._dcfr_positive_log_prefix_iteration += 1

    def _dcfr_prefix_value(self, iteration: int) -> float:
        if iteration == self._dcfr_positive_log_prefix_iteration:
            return self._dcfr_positive_log_prefix_value
        if iteration == self._dcfr_positive_log_prefix_iteration - 1:
            return self._dcfr_previous_positive_log_prefix_value
        if iteration == 0 and self._dcfr_positive_log_prefix_iteration <= 1:
            return 0.0
        raise RuntimeError("DCFR scalar prefix does not cover the requested iteration")

    def _discount_regrets_through(self, node: Node, iteration: int) -> None:
        if self.algorithm != "dcfr" or iteration <= node.last_regret_discount_iteration:
            return
        self._ensure_dcfr_prefix(iteration)
        positive = math.exp(
            self._dcfr_prefix_value(iteration)
            - node.regret_positive_log_prefix_anchor
        )
        negative = 0.5 ** (iteration - node.last_regret_discount_iteration)
        node.regrets = [
            value * (positive if value >= 0.0 else negative) for value in node.regrets
        ]
        node.last_regret_discount_iteration = iteration
        node.regret_positive_log_prefix_anchor = self._dcfr_prefix_value(iteration)
        node.checkpoint_dirty = True

    def _discount_strategy_through(self, node: Node, iteration: int) -> None:
        if self.algorithm != "dcfr" or iteration <= node.last_strategy_discount_iteration:
            return
        start = node.last_strategy_discount_iteration + 1
        factor = (float(start) / float(iteration + 1)) ** 2.0
        node.strategy_sum = [value * factor for value in node.strategy_sum]
        node.last_strategy_discount_iteration = iteration
        node.checkpoint_dirty = True

    def _materialize_dcfr_through(self, iteration: int) -> None:
        if self.algorithm != "dcfr":
            return
        for node in self.nodes.values():
            self._discount_regrets_through(node, iteration)
            self._discount_strategy_through(node, iteration)

    def _chance_transition(self, state: HoldemState, deal: Deal) -> HoldemState:
        if state.street == "preflop":
            return advance_flop(state, sample_flop(deal, self.public_chance_rng))
        cards = legal_next_cards(state.board, deal)
        return advance_public_card(
            state, cards[self.public_chance_rng.randrange(len(cards))]
        )

    def _traverse(
        self,
        state: HoldemState,
        deal: Deal,
        update_player: int,
        regret_touched: list[Node] | None,
        average_touched: list[Node] | None,
        touch_epoch: int,
    ) -> float:
        depth_value = self._depth_value(state, deal, update_player)
        if depth_value is not None:
            return depth_value
        if state.terminal:
            return terminal_utility(state, deal, update_player)
        if state.actor is None or state.all_in_runout:
            return self._traverse(
                self._chance_transition(state, deal),
                deal,
                update_player,
                regret_touched,
                average_touched,
                touch_epoch,
            )

        actions = legal_actions(state)
        if not actions:
            raise RuntimeError("nonterminal decision state has no legal actions")
        actor = state.actor
        labels = strategy_action_labels(state, actions)
        key = strategy_information_set_key(
            state,
            deal,
            actor,
            self.strategy_context,
            actions=actions,
            action_labels=labels,
            context_key=(
                "none"
                if self.strategy_context is None
                else self.strategy_context_sha256[:16]
            ),
            deal_already_validated=True,
        )
        node = self._node(key, labels)
        self._discount_regrets_through(node, self.iteration - 1)
        policy = node.policy()
        node.visits += 1

        if actor == update_player:
            action_values = []
            for action in actions:
                action_values.append(
                    self._traverse(
                        apply_action(
                            state,
                            action,
                            prevalidated_legal_actions=actions,
                        ),
                        deal,
                        update_player,
                        regret_touched,
                        average_touched,
                        touch_epoch,
                    )
                )
            node_value = math.fsum(
                probability * value for probability, value in zip(policy, action_values)
            )
            for index, value in enumerate(action_values):
                node.regrets[index] += value - node_value
                if self.algorithm == "cfr_plus":
                    node.regrets[index] = max(0.0, node.regrets[index])
            node.checkpoint_dirty = True
            if regret_touched is not None and node.regret_touch_epoch != touch_epoch:
                node.regret_touch_epoch = touch_epoch
                regret_touched.append(node)
            return node_value

        sampled_index = _sample_index(policy, self.action_rng)
        value = self._traverse(
            apply_action(
                state,
                actions[sampled_index],
                prevalidated_legal_actions=actions,
            ),
            deal,
            update_player,
            regret_touched,
            average_touched,
            touch_epoch,
        )
        self._discount_strategy_through(node, self.iteration - 1)
        average_weight = float(self.iteration) if self.algorithm == "cfr_plus" else 1.0
        for index, probability in enumerate(policy):
            node.strategy_sum[index] += average_weight * probability
        node.checkpoint_dirty = True
        if average_touched is not None and node.average_touch_epoch != touch_epoch:
            node.average_touch_epoch = touch_epoch
            average_touched.append(node)
        return value

    def run(
        self,
        target_iterations: int,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        progress_interval_seconds: float = 60.0,
        finalize_result: bool = True,
    ) -> dict[str, Any] | None:
        if target_iterations < self.iteration:
            raise ValueError("target iterations precede checkpoint")
        last_progress = time.monotonic()
        while self.iteration < target_iterations:
            self.iteration += 1
            if self.algorithm == "dcfr":
                self._ensure_dcfr_prefix(self.iteration)
            deal = self.joint_sampler.sample(self.deal_rng)
            for player in (0, 1):
                self._touch_epoch += 1
                self._regret_touched_nodes.clear()
                self._average_touched_nodes.clear()
                regret_touched = (
                    self._regret_touched_nodes
                    if self.algorithm == "dcfr"
                    else None
                )
                average_touched = (
                    self._average_touched_nodes
                    if self.algorithm == "dcfr"
                    else None
                )
                value = self._traverse(
                    self.initial_state,
                    deal,
                    player,
                    regret_touched,
                    average_touched,
                    self._touch_epoch,
                )
                self.sampled_utility_sum[player] += value
                if regret_touched is not None:
                    for node in regret_touched:
                        self._discount_regrets_through(node, self.iteration)
                if average_touched is not None:
                    for node in average_touched:
                        self._discount_strategy_through(node, self.iteration)
            if progress_callback is not None and (
                self.iteration == target_iterations or self.iteration % 256 == 0
            ):
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
        self._materialize_dcfr_through(self.iteration)
        return {
            key: dict(zip(self.nodes[key].actions, self.nodes[key].average_policy()))
            for key in sorted(self.nodes)
        }

    def _result_from_policy(
        self,
        policy: Mapping[str, Mapping[str, float]],
        *,
        node_count: int,
        nonzero_regret_node_count: int,
        policy_sha256: str | None = None,
    ) -> dict[str, Any]:
        body = {
                "schema_version": "stage21_external_sampling_result_v7",
                "solver_contract_sha256": SOLVER_CONTRACT_SHA256,
                "root_id": self.root_id,
                "root_semantic_sha256": self.root_semantic_sha256,
                "algorithm": self.algorithm,
                "seed": self.seed,
                "task_semantic_sha256": self.task_semantic_sha256,
                "solver_continuation_semantic_sha256": (
                    self.solver_continuation_semantic_sha256
                ),
                "rng_stream_seeds_sha256": sha256_json(self.rng_stream_seeds),
                "optimization_stream_identity_sha256": (
                    self.optimization_stream_identity_sha256
                ),
                "iterations": self.iteration,
                "depth_limit": self.depth_limit,
                "strategy_context_sha256": self.strategy_context_sha256,
                "node_count": node_count,
                "nonzero_regret_node_count": nonzero_regret_node_count,
                "sampled_mean_utility": [
                    value / self.iteration if self.iteration else 0.0
                    for value in self.sampled_utility_sum
                ],
                "on_policy_sampling": True,
                "importance_ratio": 1.0,
                "fresh_regrets": True,
                "fresh_strategy_sum": True,
                "policy_sha256": (
                    sha256_json(policy) if policy_sha256 is None else policy_sha256
                ),
            }
        return with_self_hash(body)

    def result(
        self,
        *,
        materialized_policy: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        policy = self.policy() if materialized_policy is None else materialized_policy
        return self._result_from_policy(
            policy,
            node_count=len(self.nodes),
            nonzero_regret_node_count=sum(
                any(abs(value) > 0 for value in node.regrets)
                for node in self.nodes.values()
            ),
        )

    def snapshot_policy_to_disk(
        self, policy_store_path: Path
    ) -> tuple[dict[str, Any], DiskBackedPolicy]:
        """Create a bounded disk policy for an interim qualification look.

        Solver nodes, RNG streams, and continuation state remain live so a
        failed look can continue solving without reconstruction.  The caller
        owns and must close/cleanup the returned policy before the next look.
        """

        self._materialize_dcfr_through(self.iteration)
        node_count = len(self.nodes)
        nonzero_regret_node_count = sum(
            any(abs(value) > 0 for value in node.regrets)
            for node in self.nodes.values()
        )
        builder = DiskPolicyBuilder(
            policy_store_path, require_sorted_input=False
        )
        try:
            for key, node in self.nodes.items():
                builder.add(
                    key, dict(zip(node.actions, node.average_policy()))
                )
            store = builder.finalize()
        except Exception:
            builder.abort()
            raise
        policy = DiskBackedPolicy(
            Path(str(store["path"])),
            expected_policy_sha256=str(store["policy_sha256"]),
            expected_row_count=int(store["row_count"]),
        )
        result = self._result_from_policy(
            policy,
            node_count=node_count,
            nonzero_regret_node_count=nonzero_regret_node_count,
            policy_sha256=str(store["policy_sha256"]),
        )
        return result, policy

    def finalize_policy_and_release(
        self,
        *,
        policy_store_path: Path | None = None,
    ) -> tuple[
        dict[str, Any], Mapping[str, Mapping[str, float]], dict[str, Any]
    ]:
        """Materialize the final policy while releasing each consumed node.

        This irreversible production path is called only after the final durable
        checkpoint.  ``policy()`` and ``result()`` intentionally retain their
        public behavior for callers that still need a live solver.
        """

        self._materialize_dcfr_through(self.iteration)
        node_count = len(self.nodes)
        nonzero_regret_node_count = sum(
            any(abs(value) > 0 for value in node.regrets)
            for node in self.nodes.values()
        )
        vector_values = sum(
            len(node.regrets) + len(node.strategy_sum)
            for node in self.nodes.values()
        )

        # These lists can retain Node objects independently of ``self.nodes``.
        # They are no longer needed once final discounting and checkpointing are
        # complete, so clear them before progressively popping the node map.
        self._regret_touched_nodes.clear()
        self._average_touched_nodes.clear()
        self.continuation_authority = None
        self.deal_rng = None
        self.public_chance_rng = None
        self.action_rng = None

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
            nonzero_regret_node_count=nonzero_regret_node_count,
            policy_sha256=policy_sha256,
        )
        self.initial_state = None
        self.joint_sampler = None
        self.strategy_context = None
        self._dcfr_positive_log_prefix_iteration = 0
        self._dcfr_positive_log_prefix_value = 0.0
        self._dcfr_previous_positive_log_prefix_value = 0.0
        release = {
            "schema_version": "stage21_solver_training_state_release_v1",
            "released_node_count": node_count,
            "released_vector_value_count": vector_values,
            "released_dcfr_prefix_entry_count": (
                2 if self.algorithm == "dcfr" else 0
            ),
            "released_rng_stream_count": 3,
            "progressive_node_release_during_policy_materialization": True,
            "nodes_remaining_when_policy_completed": len(self.nodes),
            "full_policy_and_full_training_nodes_simultaneously_retained": False,
            "policy_and_result_retained_outside_solver": True,
            "final_policy_disk_backed": builder is not None,
            "complete_policy_materialized_in_python": builder is None,
        }
        return solver_result, policy, release

    def _checkpoint_metadata(self) -> dict[str, Any]:
        metadata = {
            "schema_version": "stage21_external_sampling_checkpoint_v7",
            "solver_contract_sha256": SOLVER_CONTRACT_SHA256,
            "root_id": self.root_id,
            "root_semantic_sha256": self.root_semantic_sha256,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "originating_task_semantic_sha256": self.task_semantic_sha256,
            "solver_continuation_semantic_sha256": (
                self.solver_continuation_semantic_sha256
            ),
            "rng_stream_seeds": self.rng_stream_seeds,
            "optimization_stream_identity_sha256": (
                self.optimization_stream_identity_sha256
            ),
            "depth_limit": self.depth_limit,
            "strategy_context": self.strategy_context,
            "strategy_context_sha256": self.strategy_context_sha256,
            "iteration": self.iteration,
            "sampled_utility_sum": list(self.sampled_utility_sum),
            "dcfr_positive_log_prefix_state": {
                "iteration": self._dcfr_positive_log_prefix_iteration,
                "current_value": self._dcfr_positive_log_prefix_value,
                "previous_value": self._dcfr_previous_positive_log_prefix_value,
            },
            "rng_states": {
                "private_deal": self.deal_rng.getstate(),
                "public_chance": self.public_chance_rng.getstate(),
                "action_selection": self.action_rng.getstate(),
            },
        }
        return metadata

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
        metadata = self._checkpoint_metadata()
        dirty_node_count = sum(node.checkpoint_dirty for node in self.nodes.values())
        rss_before = None
        try:
            import psutil  # type: ignore

            rss_before = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        receipt = write_self_hashed_json_stream(
            path,
            document=metadata,
            streamed_mappings={
                "nodes": (
                    (key, self.nodes[key].to_json())
                    for key in sorted(self.nodes)
                )
            },
            binding_fields=(
                "schema_version",
                "solver_contract_sha256",
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
        rss_after = None
        try:
            import psutil  # type: ignore

            rss_after = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        return {
            **receipt,
            "schema_version": "stage21_checkpoint_publication_receipt_v2",
            "checkpoint_node_count": len(self.nodes),
            "checkpoint_dirty_node_count": dirty_node_count,
            "checkpoint_iteration": self.iteration,
            "lazy_dcfr_state_persisted_without_full_node_materialization": True,
            "checkpoint_rss_before_bytes": rss_before,
            "checkpoint_rss_after_bytes": rss_after,
        }

    def load_checkpoint(self, path: Path) -> None:
        self.nodes.clear()

        def load_node(key: str, value: Mapping[str, Any]) -> None:
            if key in self.nodes:
                raise ValueError("checkpoint contains a duplicate node key")
            self.nodes[key] = Node(
                actions=tuple(value["actions"]),
                regrets=[float(item) for item in value["regrets"]],
                strategy_sum=[float(item) for item in value["strategy_sum"]],
                visits=int(value["visits"]),
                last_regret_discount_iteration=int(
                    value["last_regret_discount_iteration"]
                ),
                last_strategy_discount_iteration=int(
                    value["last_strategy_discount_iteration"]
                ),
                regret_positive_log_prefix_anchor=float(
                    value["regret_positive_log_prefix_anchor"]
                ),
                checkpoint_dirty=False,
            )

        streamed = read_self_hashed_json_streamed_mapping(
            path,
            streamed_mapping_key="nodes",
            on_member=load_node,
        )
        document = streamed["document"]
        binding = {
            "solver_contract_sha256": SOLVER_CONTRACT_SHA256,
            "root_id": self.root_id,
            "root_semantic_sha256": self.root_semantic_sha256,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "solver_continuation_semantic_sha256": (
                self.solver_continuation_semantic_sha256
            ),
            "rng_stream_seeds": self.rng_stream_seeds,
            "optimization_stream_identity_sha256": (
                self.optimization_stream_identity_sha256
            ),
            "depth_limit": self.depth_limit,
            "strategy_context_sha256": self.strategy_context_sha256,
        }
        for key, value in binding.items():
            if document.get(key) != value:
                raise ValueError(f"stale checkpoint binding: {key}")
        self.iteration = int(document["iteration"])
        self.sampled_utility_sum = [float(value) for value in document["sampled_utility_sum"]]
        states = document["rng_states"]
        self.deal_rng.setstate(_nested_tuple(states["private_deal"]))
        self.public_chance_rng.setstate(_nested_tuple(states["public_chance"]))
        self.action_rng.setstate(_nested_tuple(states["action_selection"]))
        prefix = document.get("dcfr_positive_log_prefix_state")
        if not isinstance(prefix, Mapping):
            raise ValueError("checkpoint lacks scalar DCFR prefix state")
        self._dcfr_positive_log_prefix_iteration = int(prefix["iteration"])
        self._dcfr_positive_log_prefix_value = float(prefix["current_value"])
        self._dcfr_previous_positive_log_prefix_value = float(
            prefix["previous_value"]
        )
        expected_prefix_iteration = self.iteration if self.algorithm == "dcfr" else 0
        if self._dcfr_positive_log_prefix_iteration != expected_prefix_iteration:
            raise ValueError("checkpoint DCFR prefix iteration mismatch")
        self._regret_touched_nodes.clear()
        self._average_touched_nodes.clear()
        self._touch_epoch = 0

    def release_training_state(self) -> dict[str, Any]:
        """Release solver-only state after the immutable policy/result exist."""

        node_count = len(self.nodes)
        vector_values = sum(
            len(node.regrets) + len(node.strategy_sum) for node in self.nodes.values()
        )
        self.nodes.clear()
        self._regret_touched_nodes.clear()
        self._average_touched_nodes.clear()
        self.continuation_authority = None
        self.deal_rng = None
        self.public_chance_rng = None
        self.action_rng = None
        self._dcfr_positive_log_prefix_iteration = 0
        self._dcfr_positive_log_prefix_value = 0.0
        self._dcfr_previous_positive_log_prefix_value = 0.0
        return {
            "schema_version": "stage21_solver_training_state_release_v1",
            "released_node_count": node_count,
            "released_vector_value_count": vector_values,
            "released_dcfr_prefix_entry_count": (
                2 if self.algorithm == "dcfr" else 0
            ),
            "released_rng_stream_count": 3,
            "policy_and_result_retained_outside_solver": True,
        }


def bounded_resume_canary(
    solver: ExternalSamplingSolver,
    *,
    first_target: int,
    final_target: int,
    checkpoint: Path,
    reconstructed_solver: ExternalSamplingSolver,
) -> dict[str, Any]:
    solver.run(first_target)
    solver.save_checkpoint(checkpoint)
    reconstructed_solver.load_checkpoint(checkpoint)
    resumed = reconstructed_solver.run(final_target)
    return {
        "schema_version": "stage21_bounded_resume_canary_v2",
        "first_target": first_target,
        "final_target": final_target,
        "resumed_result": resumed,
        "checkpoint_readback_valid": True,
    }
