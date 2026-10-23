from __future__ import annotations

import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .chance import legal_flops_for_deal, legal_next_cards, sample_flop
from .common import (
    STRATEGY_ABSTRACTION_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    stream_file_identity,
    validate_self_hash,
    with_self_hash,
)
from .game import (
    Deal,
    HoldemState,
    advance_flop,
    advance_public_card,
    apply_action,
    legal_actions,
    strategy_action_labels,
    strategy_backoff_keys,
    strategy_information_set_key,
    terminal_utilities,
)


class ContinuationAuthorityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContinuationValue:
    key: str
    families: Mapping[str, tuple[float, float]]

    def validate(self, required_families: Sequence[str]) -> None:
        missing = sorted(set(required_families) - set(self.families))
        if missing:
            raise ContinuationAuthorityError(f"continuation row missing families: {missing}")
        for family, values in self.families.items():
            if len(values) != 2 or any(not math.isfinite(value) for value in values):
                raise ContinuationAuthorityError(f"invalid continuation values for {family}")
            if abs(values[0] + values[1]) > 1e-9:
                raise ContinuationAuthorityError(f"continuation family {family} is not zero sum")


def continuation_key(
    state: HoldemState,
    deal: Deal,
    context: Mapping[str, Any] | None = None,
) -> str:
    deal.validate(state.board)
    payload = {
        "street": state.street,
        "board": list(state.board),
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
        "history": [row.__dict__ for row in state.history],
        "bet_fractions": list(state.bet_fractions),
        "hole0": sorted(deal.hole0),
        "hole1": sorted(deal.hole1),
        "context": context,
    }
    return f"continuation_v3:{sha256_json(payload)}"


class ContinuationAuthority:
    def __init__(self, manifest: Mapping[str, Any], rows: Mapping[str, ContinuationValue]):
        self.manifest = dict(manifest)
        self.rows = dict(rows)
        self._shared_immutable_rows = False
        validate_self_hash(self.manifest)
        if self.manifest.get("status") != "ACCEPTED_SCIENCE_AUTHORITY":
            raise ContinuationAuthorityError("continuation authority is not accepted")
        if self.manifest.get("complete_member_authority") is not True:
            raise ContinuationAuthorityError("continuation authority membership is incomplete")
        required = tuple(self.manifest.get("required_continuation_families", ()))
        if len(required) < 2:
            raise ContinuationAuthorityError("multi-family continuation authority is required")
        for row in self.rows.values():
            row.validate(required)
        actual_hash = sha256_json(
            {
                key: {family: list(values) for family, values in sorted(row.families.items())}
                for key, row in sorted(self.rows.items())
            }
        )
        if actual_hash != self.manifest.get("continuation_rows_sha256"):
            raise ContinuationAuthorityError("continuation row identity mismatch")

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        rows_path: Path,
        *,
        _manifest_raw: bytes | None = None,
        _manifest: Mapping[str, Any] | None = None,
    ) -> "ContinuationAuthority":
        manifest_raw = (
            manifest_path.read_bytes() if _manifest_raw is None else _manifest_raw
        )
        manifest = json.loads(manifest_raw) if _manifest is None else dict(_manifest)
        rows_raw = rows_path.read_bytes()
        if sha256_bytes(rows_raw) != manifest.get("continuation_file_sha256"):
            raise ContinuationAuthorityError("continuation file byte hash mismatch")
        rows: dict[str, ContinuationValue] = {}
        for line in rows_raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["key"])
            if key in rows:
                raise ContinuationAuthorityError(f"duplicate continuation key: {key}")
            rows[key] = ContinuationValue(
                key,
                {
                    str(family): (float(values[0]), float(values[1]))
                    for family, values in record["families"].items()
                },
            )
        return cls(manifest, rows)

    def resolve_all_families(
        self,
        state: HoldemState,
        deal: Deal,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, tuple[float, float]]:
        key = continuation_key(state, deal, context)
        try:
            return self.rows[key].families
        except KeyError as error:
            raise ContinuationAuthorityError(f"missing exact continuation key: {key}") from error

    def resolve_preregistered_mixture(
        self,
        state: HoldemState,
        deal: Deal,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        families = self.resolve_all_families(state, deal, context=context)
        weights = self.manifest.get("preregistered_family_mixture", {})
        if set(weights) != set(self.manifest["required_continuation_families"]):
            raise ContinuationAuthorityError("family mixture membership mismatch")
        if any(float(value) < 0 for value in weights.values()) or abs(
            math.fsum(float(value) for value in weights.values()) - 1.0
        ) > 1e-12:
            raise ContinuationAuthorityError("family mixture is not a probability vector")
        result = tuple(
            math.fsum(float(weights[family]) * families[family][player] for family in weights)
            for player in range(2)
        )
        if abs(result[0] + result[1]) > 1e-9:
            raise ContinuationAuthorityError("mixture continuation value is not zero sum")
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "authority_kind": "exact_hand_specific_rows",
            "row_count": len(self.rows),
            "hard_fallback_fraction": 0.0,
            "within_fallback_bound": True,
            "within_all_backoff_bounds": True,
        }

    def release_runtime_state(self) -> dict[str, Any]:
        row_count = len(self.rows)
        shared_immutable_rows_retained = bool(self._shared_immutable_rows)
        if not shared_immutable_rows_retained:
            self.rows.clear()
        return {
            "schema_version": "stage21_continuation_cache_release_v1",
            "authority_kind": "exact_hand_specific_rows",
            "released_row_count": 0 if shared_immutable_rows_retained else row_count,
            "shared_immutable_row_count_retained": (
                row_count if shared_immutable_rows_retained else 0
            ),
            "released_recursively": True,
        }

    def spawn_runtime_view(self) -> "ContinuationAuthority":
        """Reuse immutable validated rows without sharing mutable task state."""

        instance = object.__new__(ContinuationAuthority)
        instance.manifest = dict(self.manifest)
        instance.rows = self.rows
        instance._shared_immutable_rows = True
        return instance


def _normalize(weights: Sequence[float]) -> tuple[float, ...]:
    positive = [max(0.0, float(value)) for value in weights]
    total = math.fsum(positive)
    if total <= 0:
        return tuple(1.0 / len(positive) for _ in positive)
    return tuple(value / total for value in positive)


def _low_variance_fallback(actions: Sequence[Any]) -> tuple[float, ...]:
    return _normalize(
        [
            3.0
            if action.kind in {"check", "call"}
            else (1.0 if action.kind == "fold" else 0.05)
            for action in actions
        ]
    )


def _select(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if threshold <= running:
            return index
    return len(probabilities) - 1


class PolicyRolloutContinuationAuthority:
    """Hand-specific, jointly zero-sum continuation from accepted policy banks.

    Each family uses preregistered common-random-number rollouts.  The same
    exact public/private leaf therefore returns the same value pair to both
    traversers.  No scalar family is selected and no residual is recentered.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        bank: Mapping[str, Any],
        *,
        downstream: "PolicyRolloutContinuationAuthority | None" = None,
        _allow_pending_calibration: bool = False,
    ) -> None:
        self.manifest = dict(manifest)
        self.bank = dict(bank)
        self.downstream = downstream
        validate_self_hash(self.manifest)
        validate_self_hash(self.bank)
        if self.manifest.get("status") != "ACCEPTED_SCIENCE_AUTHORITY":
            raise ContinuationAuthorityError("policy-rollout continuation is not accepted")
        if self.manifest.get("schema_version") != "stage21_empirical_policy_rollout_continuation_manifest_v6":
            raise ContinuationAuthorityError("unsupported policy-rollout manifest schema")
        if self.manifest.get("complete_member_authority") is not True:
            raise ContinuationAuthorityError("policy-rollout continuation membership is incomplete")
        if self.manifest.get("authority_kind") != "policy_rollout":
            raise ContinuationAuthorityError("unexpected continuation authority kind")
        if self.bank.get("schema_version") != "stage21_policy_bank_v6":
            raise ContinuationAuthorityError("unsupported policy-bank schema")
        if self.bank.get("strategy_abstraction_version") != STRATEGY_ABSTRACTION_VERSION:
            raise ContinuationAuthorityError(
                "policy bank does not bind the exact-board key schema"
            )
        calibration = self.manifest.get("calibration")
        if _allow_pending_calibration:
            if self.manifest.get("calibration_status") != "PENDING_BUILD_TIME_FAIL_CLOSED_CALIBRATION":
                raise ContinuationAuthorityError("calibration candidate has unexpected status")
        else:
            if self.manifest.get("calibration_status") != "PASS":
                raise ContinuationAuthorityError("policy-rollout continuation calibration did not pass")
            if not isinstance(calibration, Mapping):
                raise ContinuationAuthorityError("policy-rollout continuation lacks calibration receipt")
            validate_self_hash(calibration)
            if (
                calibration.get("gate_pass") is not True
                or calibration.get("policy_bank_self_sha256") != self.bank.get("self_sha256")
            ):
                raise ContinuationAuthorityError("policy-rollout continuation calibration is invalid")
        if (
            self.manifest.get("safe_solving_claim") is not False
            or self.manifest.get("exploitability_nonincrease_guarantee") is not False
            or self.manifest.get("claim_scope")
            != "bounded_empirical_zero_sum_policy_rollout_leaf_model"
        ):
            raise ContinuationAuthorityError(
                "policy rollout must retain its narrow non-safe-solving claim"
            )
        required = tuple(self.manifest.get("required_continuation_families", ()))
        if set(required) != set(self.bank.get("families", {})) or len(required) < 2:
            raise ContinuationAuthorityError("policy-bank family membership mismatch")
        if self.bank.get("street") != self.manifest.get("street"):
            raise ContinuationAuthorityError("policy-bank street mismatch")
        for family in required:
            member_rows = self.bank.get("family_members", {}).get(family, ())
            if len(member_rows) < 2:
                raise ContinuationAuthorityError(
                    f"policy-bank family {family} lacks two accepted members"
                )
            for key, row in self.bank["families"][family].items():
                parts = str(key).split("|")
                if (
                    len(parts) != 12
                    or parts[0] != STRATEGY_ABSTRACTION_VERSION
                    or not parts[6].startswith("public:")
                    or not parts[-1].startswith("legal:")
                ):
                    raise ContinuationAuthorityError("malformed policy-bank strategy key")
                labels = tuple(filter(None, parts[-1].removeprefix("legal:").split(",")))
                if not isinstance(row, Mapping) or set(row) != set(labels):
                    raise ContinuationAuthorityError("policy-bank row action mismatch")
                probabilities = [float(row[label]) for label in labels]
                if (
                    any(not math.isfinite(value) or value < 0 for value in probabilities)
                    or abs(math.fsum(probabilities) - 1.0) > 1e-9
                ):
                    raise ContinuationAuthorityError("invalid policy-bank probability row")
        weights = self.manifest.get("preregistered_family_mixture", {})
        if set(weights) != set(required) or any(float(value) < 0 for value in weights.values()):
            raise ContinuationAuthorityError("invalid preregistered family mixture")
        if abs(math.fsum(float(value) for value in weights.values()) - 1.0) > 1e-12:
            raise ContinuationAuthorityError("family mixture is not normalized")
        if int(self.manifest.get("rollouts_per_family", 0)) <= 0:
            raise ContinuationAuthorityError("rollout count must be positive")
        if self.manifest.get("scalar_family_selection_used") is not False:
            raise ContinuationAuthorityError("scalar family selection is forbidden")
        if self.manifest.get("aggregate_residual_recentering_used") is not False:
            raise ContinuationAuthorityError("continuation recentering is forbidden")
        expected_downstream = self.manifest.get("downstream")
        if (expected_downstream is None) != (downstream is None):
            raise ContinuationAuthorityError("downstream continuation binding mismatch")
        if downstream is not None and downstream.manifest.get("street") != self.manifest.get(
            "depth_limit"
        ):
            raise ContinuationAuthorityError("downstream street does not match depth limit")
        self._cache_limit = int(self.manifest.get("cache_entry_limit", 4096))
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._cache: OrderedDict[str, dict[str, tuple[float, float]]] = OrderedDict()
        self._row_probability_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]], tuple[float, ...]
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._rollout_calls = 0
        self._rollout_nanoseconds = 0
        self._action_queries = 0
        self._hard_fallbacks = 0
        self._backoff_hits = [0] * 6
        limits = self.manifest.get("backoff_fraction_limits_by_level")
        if not isinstance(limits, list) or len(limits) != 6 or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in limits
        ):
            raise ContinuationAuthorityError("all six backoff levels require bounds")
        self._backoff_limits = tuple(float(value) for value in limits)

    def spawn_runtime_view(self) -> "PolicyRolloutContinuationAuthority":
        """Create clean mutable counters/caches over validated immutable documents."""

        instance = object.__new__(PolicyRolloutContinuationAuthority)
        instance.manifest = dict(self.manifest)
        instance.bank = dict(self.bank)
        instance.downstream = (
            self.downstream.spawn_runtime_view()
            if self.downstream is not None
            else None
        )
        instance._cache_limit = self._cache_limit
        instance._reset_runtime_state()
        return instance

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        bank_path: Path,
        *,
        _visited: frozenset[Path] = frozenset(),
        _manifest_raw: bytes | None = None,
        _manifest: Mapping[str, Any] | None = None,
    ) -> "PolicyRolloutContinuationAuthority":
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest in _visited:
            raise ContinuationAuthorityError("continuation authority cycle detected")
        manifest_raw = (
            manifest_path.read_bytes() if _manifest_raw is None else _manifest_raw
        )
        manifest = json.loads(manifest_raw) if _manifest is None else dict(_manifest)
        del manifest_raw
        validate_self_hash(manifest)
        bank_identity = stream_file_identity(bank_path)
        if bank_identity["sha256"] != manifest.get("policy_bank_file_sha256"):
            raise ContinuationAuthorityError("policy-bank file byte hash mismatch")
        # Hash the exact raw file in a bounded-memory pass, then parse it from a
        # second local sequential stream.  json.load's decoded text is released
        # before this method recurses into a downstream bank, so raw bytes and
        # multiple parsed banks are never retained together by this loader.
        with bank_path.open("r", encoding="utf-8") as handle:
            bank = json.load(handle)
        if bank.get("self_sha256") != manifest.get("policy_bank_self_sha256"):
            raise ContinuationAuthorityError("policy-bank semantic identity mismatch")
        downstream = None
        binding = manifest.get("downstream")
        if binding is not None:
            base = manifest_path.parent
            downstream = cls.load(
                base / str(binding["manifest_relative_path"]),
                base / str(binding["policy_bank_relative_path"]),
                _visited=_visited | {resolved_manifest},
            )
            if downstream.manifest.get("self_sha256") != binding.get("manifest_self_sha256"):
                raise ContinuationAuthorityError("downstream manifest identity mismatch")
        return cls(manifest, bank, downstream=downstream)

    def _policy_probabilities(
        self,
        family: str,
        state: HoldemState,
        deal: Deal,
        context: Mapping[str, Any] | None,
        *,
        actions: Sequence[Any] | None = None,
        context_key: str | None = None,
    ) -> tuple[tuple[float, ...], int | None]:
        if state.actor is None:
            raise ContinuationAuthorityError("policy lookup requires an actor")
        action_set = legal_actions(state) if actions is None else tuple(actions)
        labels = strategy_action_labels(state, action_set)
        full_key = strategy_information_set_key(
            state,
            deal,
            state.actor,
            context,
            actions=action_set,
            action_labels=labels,
            context_key=context_key,
            deal_already_validated=True,
        )
        family_rows = self.bank["families"][family]
        self._action_queries += 1
        for level, key in enumerate(strategy_backoff_keys(full_key)):
            row = family_rows.get(key)
            if row is None:
                continue
            cache_key = (family, key, labels)
            probabilities = self._row_probability_cache.get(cache_key)
            if probabilities is None:
                if set(row) != set(labels):
                    raise ContinuationAuthorityError(
                        "policy row action membership mismatch"
                    )
                probabilities = _normalize([float(row[label]) for label in labels])
                self._row_probability_cache[cache_key] = probabilities
                if len(self._row_probability_cache) > self._cache_limit:
                    self._row_probability_cache.popitem(last=False)
            else:
                self._row_probability_cache.move_to_end(cache_key)
            self._backoff_hits[level] += 1
            return probabilities, level
        self._hard_fallbacks += 1
        return _low_variance_fallback(action_set), None

    def _rollout(
        self,
        state: HoldemState,
        deal: Deal,
        family: str,
        public_chance_rng: random.Random,
        action_rng: random.Random,
        context: Mapping[str, Any] | None,
        *,
        context_key: str | None = None,
    ) -> tuple[float, float]:
        current = state
        resolved_context_key = (
            "none" if context is None else sha256_json(context)[:16]
        ) if context_key is None else context_key
        for _ in range(256):
            if current.terminal:
                pair = terminal_utilities(current, deal)
                break
            if self.downstream is not None and current.street == self.downstream.manifest["street"]:
                pair = self.downstream.resolve_preregistered_mixture(
                    current, deal, context=context
                )
                break
            if current.all_in_runout:
                pair = self._exact_all_in_value(current, deal, context)
                break
            if current.actor is None or current.all_in_runout:
                if current.street == "preflop":
                    current = advance_flop(
                        current, sample_flop(deal, public_chance_rng)
                    )
                else:
                    cards = legal_next_cards(current.board, deal)
                    current = advance_public_card(
                        current,
                        cards[public_chance_rng.randrange(len(cards))],
                    )
                continue
            actions = legal_actions(current)
            probabilities, _ = self._policy_probabilities(
                family,
                current,
                deal,
                context,
                actions=actions,
                context_key=resolved_context_key,
            )
            current = apply_action(
                current,
                actions[_select(probabilities, action_rng)],
                prevalidated_legal_actions=actions,
            )
        else:
            raise ContinuationAuthorityError("policy rollout exceeded action/chance bound")
        if any(not math.isfinite(value) for value in pair) or abs(sum(pair)) > 1e-9:
            raise ContinuationAuthorityError("policy rollout is not finite and jointly zero sum")
        return pair

    def _exact_all_in_value(
        self,
        state: HoldemState,
        deal: Deal,
        context: Mapping[str, Any] | None,
    ) -> tuple[float, float]:
        if state.terminal:
            return terminal_utilities(state, deal)
        if self.downstream is not None and state.street == self.downstream.manifest["street"]:
            return self.downstream.resolve_preregistered_mixture(
                state, deal, context=context
            )
        if not state.all_in_runout:
            raise ContinuationAuthorityError("exact all-in enumerator received a decision state")
        children = (
            (advance_flop(state, flop) for flop in legal_flops_for_deal(deal))
            if state.street == "preflop"
            else (
                advance_public_card(state, card)
                for card in legal_next_cards(state.board, deal)
            )
        )
        player_values = ([], [])
        for child in children:
            pair = self._exact_all_in_value(child, deal, context)
            player_values[0].append(pair[0])
            player_values[1].append(pair[1])
        result = tuple(
            math.fsum(player_values[player]) / len(player_values[player])
            for player in range(2)
        )
        if abs(sum(result)) > 1e-9:
            raise ContinuationAuthorityError("exact all-in continuation is not zero sum")
        return result

    def resolve_all_families(
        self,
        state: HoldemState,
        deal: Deal,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, tuple[float, float]]:
        if state.street != self.manifest.get("street"):
            raise ContinuationAuthorityError("continuation called at the wrong street")
        key = continuation_key(state, deal, context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return cached
        self._cache_misses += 1
        rollout_started = time.perf_counter_ns()
        values: dict[str, tuple[float, float]] = {}
        count = int(self.manifest["rollouts_per_family"])
        context_key = "none" if context is None else sha256_json(context)[:16]
        rollout_seeds = []
        for rollout_index in range(count):
            seed_material = {
                "continuation_key": key,
                "rollout_index": rollout_index,
                "policy_bank_self_sha256": self.bank["self_sha256"],
                "rollout_contract": "paired_named_streams_v1",
            }
            common_seed = sha256_json(seed_material)
            rollout_seeds.append(
                (
                    int(sha256_json([common_seed, "public_chance"])[:16], 16),
                    int(sha256_json([common_seed, "action_selection"])[:16], 16),
                )
            )
        for family in self.manifest["required_continuation_families"]:
            samples = []
            for chance_seed, action_seed in rollout_seeds:
                self._rollout_calls += 1
                samples.append(
                    self._rollout(
                        state,
                        deal,
                        family,
                        random.Random(chance_seed),
                        random.Random(action_seed),
                        context,
                        context_key=context_key,
                    )
                )
            pair = tuple(math.fsum(row[player] for row in samples) / count for player in range(2))
            if abs(sum(pair)) > 1e-9:
                raise ContinuationAuthorityError("family rollout mean is not zero sum")
            values[family] = pair
        self._rollout_nanoseconds += time.perf_counter_ns() - rollout_started
        self._cache[key] = values
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
            self._cache_evictions += 1
        return values

    def exact_policy_expectation(
        self,
        state: HoldemState,
        deal: Deal,
        family: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Enumerate the exact expectation of this fixed empirical policy model.

        This is an estimator-calibration oracle, not an equilibrium or safe-
        solving certificate. It enumerates legal actions and public chance until
        terminal utility or the identity-bound downstream leaf model.
        """

        if family not in self.manifest["required_continuation_families"]:
            raise ContinuationAuthorityError("unknown calibration family")
        memo: dict[HoldemState, tuple[float, float]] = {}
        context_key = "none" if context is None else sha256_json(context)[:16]

        def walk(current: HoldemState) -> tuple[float, float]:
            cached = memo.get(current)
            if cached is not None:
                return cached
            if current.terminal:
                result = terminal_utilities(current, deal)
            elif (
                self.downstream is not None
                and current.street == self.downstream.manifest["street"]
            ):
                result = self.downstream.resolve_preregistered_mixture(
                    current, deal, context=context
                )
            elif current.all_in_runout:
                result = self._exact_all_in_value(current, deal, context)
            elif current.actor is None:
                children = (
                    (
                        advance_flop(current, flop)
                        for flop in legal_flops_for_deal(deal)
                    )
                    if current.street == "preflop"
                    else (
                        advance_public_card(current, card)
                        for card in legal_next_cards(current.board, deal)
                    )
                )
                player_values = ([], [])
                for child in children:
                    pair = walk(child)
                    player_values[0].append(pair[0])
                    player_values[1].append(pair[1])
                result = tuple(
                    math.fsum(player_values[player]) / len(player_values[player])
                    for player in (0, 1)
                )
            else:
                actions = legal_actions(current)
                probabilities, _ = self._policy_probabilities(
                    family,
                    current,
                    deal,
                    context,
                    actions=actions,
                    context_key=context_key,
                )
                pairs = [
                    walk(
                        apply_action(
                            current,
                            action,
                            prevalidated_legal_actions=actions,
                        )
                    )
                    for action in actions
                ]
                result = tuple(
                    math.fsum(
                        probability * pair[player]
                        for probability, pair in zip(probabilities, pairs)
                    )
                    for player in (0, 1)
                )
            if abs(sum(result)) > 1e-9 or any(
                not math.isfinite(value) for value in result
            ):
                raise ContinuationAuthorityError(
                    "exact fixed-policy calibration expectation is invalid"
                )
            memo[current] = result
            return result

        return walk(state)

    def calibration_rollout_mean(
        self,
        state: HoldemState,
        deal: Deal,
        family: str,
        *,
        count: int,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        if count <= 0:
            raise ContinuationAuthorityError("calibration rollout count must be positive")
        key = continuation_key(state, deal, context)
        context_key = "none" if context is None else sha256_json(context)[:16]
        samples = []
        for rollout_index in range(count):
            common = sha256_json(
                {
                    "continuation_key": key,
                    "rollout_index": rollout_index,
                    "policy_bank_self_sha256": self.bank["self_sha256"],
                    "rollout_contract": "paired_named_streams_v1",
                }
            )
            samples.append(
                self._rollout(
                    state,
                    deal,
                    family,
                    random.Random(
                        int(sha256_json([common, "public_chance"])[:16], 16)
                    ),
                    random.Random(
                        int(sha256_json([common, "action_selection"])[:16], 16)
                    ),
                    context,
                    context_key=context_key,
                )
            )
        return tuple(
            math.fsum(sample[player] for sample in samples) / count
            for player in (0, 1)
        )

    def resolve_preregistered_mixture(
        self,
        state: HoldemState,
        deal: Deal,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        families = self.resolve_all_families(state, deal, context=context)
        weights = self.manifest["preregistered_family_mixture"]
        pair = tuple(
            math.fsum(float(weights[family]) * values[player] for family, values in families.items())
            for player in range(2)
        )
        if abs(sum(pair)) > 1e-9:
            raise ContinuationAuthorityError("family rollout mixture is not zero sum")
        return pair

    def diagnostics(self) -> dict[str, Any]:
        fraction = self._hard_fallbacks / max(1, self._action_queries)
        limit = float(self.manifest.get("hard_fallback_fraction_limit", 0.05))
        backoff_fractions = [
            count / max(1, self._action_queries) for count in self._backoff_hits
        ]
        backoff_pass = [
            observed <= bound
            for observed, bound in zip(backoff_fractions, self._backoff_limits)
        ]
        downstream = self.downstream.diagnostics() if self.downstream is not None else None
        within = fraction <= limit and all(backoff_pass) and (
            downstream is None
            or downstream.get("within_all_backoff_bounds") is True
        )
        return {
            "authority_kind": "policy_rollout",
            "street": self.manifest.get("street"),
            "action_queries": self._action_queries,
            "hard_fallback_count": self._hard_fallbacks,
            "hard_fallback_fraction": fraction,
            "hard_fallback_fraction_limit": limit,
            "within_fallback_bound": within,
            "backoff_hits_by_level": list(self._backoff_hits),
            "backoff_fractions_by_level": backoff_fractions,
            "backoff_fraction_limits_by_level": list(self._backoff_limits),
            "backoff_level_gate_pass": backoff_pass,
            "within_all_backoff_bounds": within,
            "continuation_families_use_common_random_number_seeds": True,
            "public_chance_and_action_rng_streams_are_separate": True,
            "cache_entries": len(self._cache),
            "prevalidated_policy_row_cache_entries": len(
                self._row_probability_cache
            ),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_evictions": self._cache_evictions,
            "rollout_calls": self._rollout_calls,
            "rollout_seconds": self._rollout_nanoseconds / 1_000_000_000.0,
            "estimated_cache_bytes": sum(
                len(key) + 16 * sum(len(pair) for pair in values.values())
                for key, values in self._cache.items()
            ),
            "downstream": downstream,
        }

    def release_runtime_state(self) -> dict[str, Any]:
        entries = len(self._cache)
        row_entries = len(self._row_probability_cache)
        estimated_bytes = sum(
            len(key) + 16 * sum(len(pair) for pair in values.values())
            for key, values in self._cache.items()
        )
        self._cache.clear()
        self._row_probability_cache.clear()
        downstream = (
            self.downstream.release_runtime_state()
            if self.downstream is not None
            else None
        )
        return {
            "schema_version": "stage21_continuation_cache_release_v1",
            "authority_kind": "policy_rollout",
            "released_cache_entries": entries,
            "released_prevalidated_policy_row_cache_entries": row_entries,
            "estimated_released_cache_bytes": estimated_bytes,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_evictions": self._cache_evictions,
            "rollout_calls": self._rollout_calls,
            "rollout_seconds": self._rollout_nanoseconds / 1_000_000_000.0,
            "downstream": downstream,
            "released_recursively": True,
        }


def load_continuation_authority(
    manifest_path: Path,
    payload_path: Path,
) -> ContinuationAuthority | PolicyRolloutContinuationAuthority:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    kind = manifest.get("authority_kind", "exact_hand_specific_rows")
    if kind == "policy_rollout":
        return PolicyRolloutContinuationAuthority.load(
            manifest_path,
            payload_path,
            _manifest_raw=manifest_raw,
            _manifest=manifest,
        )
    if kind == "exact_hand_specific_rows":
        return ContinuationAuthority.load(
            manifest_path,
            payload_path,
            _manifest_raw=manifest_raw,
            _manifest=manifest,
        )
    raise ContinuationAuthorityError(f"unsupported continuation authority kind: {kind}")


def materialize_hand_specific_rows(
    samples: Iterable[Mapping[str, Any]], required_families: Sequence[str]
) -> tuple[dict[str, ContinuationValue], dict[str, Any]]:
    """Aggregate exact per-deal predecessor rollouts without scalar recentering."""

    accumulators: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    sample_count = 0
    for sample in samples:
        sample_count += 1
        key = str(sample["key"])
        family = str(sample["family"])
        if family not in required_families:
            raise ContinuationAuthorityError(f"unexpected continuation family: {family}")
        weight = float(sample["reach_weight"])
        values = (float(sample["value_player0"]), float(sample["value_player1"]))
        if weight <= 0 or not math.isfinite(weight):
            raise ContinuationAuthorityError("invalid reach weight")
        if any(not math.isfinite(value) for value in values) or abs(sum(values)) > 1e-9:
            raise ContinuationAuthorityError("per-deal continuation sample is not finite zero sum")
        accumulators.setdefault(key, {}).setdefault(family, []).append((weight, *values))
    rows: dict[str, ContinuationValue] = {}
    for key, family_rows in accumulators.items():
        if set(family_rows) != set(required_families):
            raise ContinuationAuthorityError(f"incomplete families for continuation key: {key}")
        values_by_family = {}
        for family, values in family_rows.items():
            denominator = math.fsum(row[0] for row in values)
            pair = tuple(
                math.fsum(row[0] * row[player + 1] for row in values) / denominator
                for player in range(2)
            )
            if abs(sum(pair)) > 1e-9:
                raise ContinuationAuthorityError("aggregated continuation is not zero sum")
            values_by_family[family] = pair
        row = ContinuationValue(key, values_by_family)
        row.validate(required_families)
        rows[key] = row
    receipt = with_self_hash(
        {
            "schema_version": "stage21_hand_specific_continuation_materialization_v2",
            "sample_count": sample_count,
            "row_count": len(rows),
            "required_families": list(required_families),
            "scalar_family_selection_used": False,
            "aggregate_residual_recentering_used": False,
            "per_deal_zero_sum_validated": True,
        }
    )
    return rows, receipt


def serialize_rows(rows: Mapping[str, ContinuationValue]) -> bytes:
    lines = []
    for key, row in sorted(rows.items()):
        lines.append(
            canonical_json_bytes(
                {
                    "key": key,
                    "families": {
                        family: list(values) for family, values in sorted(row.families.items())
                    },
                }
            )
        )
    return b"\n".join(lines) + (b"\n" if lines else b"")
