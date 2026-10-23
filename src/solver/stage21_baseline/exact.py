from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .common import sha256_json, with_self_hash
from .game import Deal
from .ranges import RangeDistribution


EXACT_CONTRACT = {
    "schema_version": "stage21_restricted_exact_contract_v4",
    "scope": "explicit_ordered_compatible_top_mass_private_range_projection_and_full_public_runout_enumeration",
    "universal_equilibrium_claim": False,
    "perfect_exploitation_claim": False,
    "projection_membership_predeclared": True,
    "both_algorithms": ["cfr_plus", "dcfr"],
    "cfr_plus_average_strategy": "linear_iteration_weighted_exact_reach",
    "dcfr_average_strategy": "unit_exact_reach_increment_then_gamma_discount",
    "dcfr_discount_timing": "after_each_completed_simultaneous_full_chance_iteration",
    "deterministic_numerical_order_ids": [1729, 3253, 4933],
    "order_id_role": "floating_point_accumulation_order_stability_only",
    "independent_solver_random_seed_claim": False,
}
EXACT_CONTRACT_SHA256 = sha256_json(EXACT_CONTRACT)


@dataclass(frozen=True)
class ProjectedDeal:
    deal: Deal
    probability: float


class ProjectedDealSampler:
    def __init__(self, rows: Sequence[ProjectedDeal]):
        if not rows:
            raise ValueError("projected deal sampler is empty")
        self.rows = tuple(rows)
        total = math.fsum(row.probability for row in self.rows)
        if abs(total - 1.0) > 1e-12:
            raise ValueError("projected deal sampler probabilities do not normalize")
        running = 0.0
        cdf = []
        for row in self.rows:
            if row.probability <= 0 or not math.isfinite(row.probability):
                raise ValueError("invalid projected deal probability")
            running += row.probability
            cdf.append(running)
        cdf[-1] = 1.0
        self._cdf = tuple(cdf)

    def sample(self, rng: random.Random) -> Deal:
        threshold = rng.random()
        for index, value in enumerate(self._cdf):
            if threshold <= value:
                return self.rows[index].deal
        return self.rows[-1].deal


def projected_deal_distribution(
    first: RangeDistribution,
    second: RangeDistribution,
    *,
    combos_per_player: int = 4,
) -> tuple[ProjectedDeal, ...]:
    if combos_per_player < 2 or combos_per_player > 4:
        raise ValueError("restricted exact projection supports two to four combos per player")
    first_ranked = sorted(
        zip(first.combos, first.probabilities), key=lambda row: (-row[1], row[0])
    )
    second_ranked = sorted(
        zip(second.combos, second.probabilities), key=lambda row: (-row[1], row[0])
    )
    # Selecting both top-K lists independently can choose the same physical
    # cards for both players and produce an empty "exact" game.  Player 0 is
    # the preregistered ordering anchor; player 1 receives the highest-mass K
    # combos that cover every selected player-0 combo. If the independent
    # top-K support is infeasible, the smallest deterministic top-mass
    # player-0 replacement is tried; no random repair is permitted.
    def select_second_support(first_rows):
        full_mask = (1 << combos_per_player) - 1
        # (count, covered-first-mask) -> (log-mass score, selected indices)
        dynamic = {(0, 0): (0.0, ())}
        for index, (combo, probability) in enumerate(second_ranked):
            mask = 0
            for first_index, (first_combo, _) in enumerate(first_rows):
                if set(combo).isdisjoint(first_combo):
                    mask |= 1 << first_index
            if mask == 0:
                continue
            prior = list(dynamic.items())
            for (count, covered), (score, indices) in prior:
                if count >= combos_per_player:
                    continue
                candidate_key = (count + 1, covered | mask)
                candidate = (score + math.log(probability), (*indices, index))
                existing = dynamic.get(candidate_key)
                if existing is None or candidate[0] > existing[0] + 1e-15 or (
                    abs(candidate[0] - existing[0]) <= 1e-15
                    and candidate[1] < existing[1]
                ):
                    dynamic[candidate_key] = candidate
        selected = dynamic.get((combos_per_player, full_mask))
        if selected is None:
            return None
        return [second_ranked[index] for index in selected[1]]

    first_rows = None
    second_rows = None
    candidate_pool = first_ranked[: min(len(first_ranked), max(12, combos_per_player + 2))]
    for candidate_indices in itertools.combinations(
        range(len(candidate_pool)), combos_per_player
    ):
        candidate_first = [candidate_pool[index] for index in candidate_indices]
        candidate_second = select_second_support(candidate_first)
        if candidate_second is not None:
            first_rows = candidate_first
            second_rows = candidate_second
            break
    if first_rows is None or second_rows is None:
        raise ValueError("restricted exact projection cannot select compatible supports")
    rows = []
    for (hole0, probability0), (hole1, probability1) in itertools.product(first_rows, second_rows):
        if set(hole0).isdisjoint(hole1):
            rows.append((Deal(hole0, hole1), probability0 * probability1))
    normalizer = math.fsum(probability for _, probability in rows)
    if normalizer <= 0:
        raise ValueError("restricted exact projection has no legal deals")
    result = tuple(ProjectedDeal(deal, probability / normalizer) for deal, probability in rows)
    if abs(math.fsum(row.probability for row in result) - 1.0) > 1e-12:
        raise AssertionError("restricted exact deal projection does not normalize")
    return result


def exact_scope_receipt(
    *,
    root_id: str,
    first: RangeDistribution,
    second: RangeDistribution,
    combos_per_player: int = 4,
) -> dict[str, Any]:
    deals = projected_deal_distribution(
        first, second, combos_per_player=combos_per_player
    )
    return with_self_hash(
        {
            "schema_version": "stage21_restricted_exact_scope_receipt_v2",
            "exact_contract_sha256": EXACT_CONTRACT_SHA256,
            "root_id": root_id,
            "combos_per_player": combos_per_player,
            "projection_selection": "deterministic_top_mass_compatible_bipartite_support",
            "legal_projected_deal_count": len(deals),
            "deal_distribution_sha256": sha256_json(
                [
                    {
                        "hole0": list(row.deal.hole0),
                        "hole1": list(row.deal.hole1),
                        "probability": row.probability,
                    }
                    for row in deals
                ]
            ),
            "full_public_runout_enumeration_required": True,
            "claim_scope": "restricted_projected_game_only",
            "universal_equilibrium_claim": False,
        }
    )
