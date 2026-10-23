from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from .common import (
    CROSSPLAY_DEALS_PER_ASSIGNMENT,
    sha256_json,
    validate_self_hash,
    with_self_hash,
)


MEMBER_IDENTITY_FIELDS = ("lane_id", "algorithm", "seed", "policy_sha256")


class CrossplayEvidenceError(ValueError):
    """Raised when archived cross-play evidence is not independently reproducible."""


_VALIDATED_RECEIPT_TOKEN = object()


class ValidatedCrossplayReceipt:
    """In-process proof that raw cross-play evidence was recomputed once."""

    __slots__ = (
        "_token",
        "root_id",
        "evidence_self_sha256",
        "member_identity_sha256",
        "gate_pass",
    )

    def __init__(
        self,
        token: object,
        *,
        root_id: str,
        evidence_self_sha256: str,
        member_identity_sha256: str,
        gate_pass: bool,
    ) -> None:
        if token is not _VALIDATED_RECEIPT_TOKEN:
            raise TypeError("validated cross-play receipts are factory-only")
        self._token = token
        self.root_id = root_id
        self.evidence_self_sha256 = evidence_self_sha256
        self.member_identity_sha256 = member_identity_sha256
        self.gate_pass = gate_pass

    def matches(
        self,
        *,
        root_id: str,
        evidence_self_sha256: str,
        members: Sequence[Mapping[str, Any]],
    ) -> bool:
        try:
            fingerprint = sha256_json(
                sorted(
                    (_canonical_member(member) for member in members),
                    key=lambda row: row["lane_id"],
                )
            )
        except (KeyError, TypeError, ValueError):
            return False
        return (
            self._token is _VALIDATED_RECEIPT_TOKEN
            and self.root_id == root_id
            and self.evidence_self_sha256 == evidence_self_sha256
            and self.member_identity_sha256 == fingerprint
        )

    def compact_document(self) -> dict[str, Any]:
        """Return the serializable binding after raw observations were recomputed."""

        return with_self_hash(
            {
                "schema_version": "stage21_compact_validated_crossplay_receipt_v1",
                "root_id": self.root_id,
                "evidence_self_sha256": self.evidence_self_sha256,
                "member_identity_sha256": self.member_identity_sha256,
                "gate_pass": self.gate_pass,
                "factory_token_verified_in_process": self._token
                is _VALIDATED_RECEIPT_TOKEN,
                "raw_observations_retained_in_aggregate_archive_only": True,
            }
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_member(member: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "lane_id": str(member.get("lane_id", "")),
        "algorithm": str(member.get("algorithm", "")),
        "seed": int(member.get("seed", -1)),
        "policy_sha256": member.get("policy_sha256"),
    }
    if not result["lane_id"]:
        raise CrossplayEvidenceError("cross-play member lacks a lane identity")
    if result["algorithm"] not in {"cfr_plus", "dcfr"}:
        raise CrossplayEvidenceError("cross-play member has an invalid algorithm")
    if result["seed"] < 0:
        raise CrossplayEvidenceError("cross-play member has an invalid seed")
    if not _is_sha256(result["policy_sha256"]):
        raise CrossplayEvidenceError("cross-play member has an invalid policy hash")
    return result


def _member_key(member: Mapping[str, Any]) -> tuple[str, str, int, str]:
    canonical = _canonical_member(member)
    return (
        canonical["lane_id"],
        canonical["algorithm"],
        canonical["seed"],
        canonical["policy_sha256"],
    )


def _summary(values_chips: Sequence[float], *, big_blind: int = 100) -> dict[str, float]:
    if not values_chips:
        raise CrossplayEvidenceError("cross-play value sequence is empty")
    values = [float(value) for value in values_chips]
    if any(not math.isfinite(value) for value in values):
        raise CrossplayEvidenceError("cross-play contains a non-finite utility")
    bb100 = [value / big_blind * 100.0 for value in values]
    mean = math.fsum(bb100) / len(bb100)
    variance = math.fsum((value - mean) ** 2 for value in bb100) / max(
        1, len(bb100) - 1
    )
    tail_count = max(1, math.ceil(0.05 * len(bb100)))
    standard_error = math.sqrt(variance / len(bb100))
    return {
        "sample_count": float(len(bb100)),
        "mean_bb_per_100": mean,
        "standard_error_bb_per_100": standard_error,
        "cvar_05_bb_per_100": math.fsum(sorted(bb100)[:tail_count]) / tail_count,
        "minimum_bb_per_100": min(bb100),
        "maximum_bb_per_100": max(bb100),
        "ci95_lower_bb_per_100": mean - 1.96 * standard_error,
        "ci95_upper_bb_per_100": mean + 1.96 * standard_error,
    }


def _canonical_assignment(
    row: Mapping[str, Any],
    *,
    expected_assignment: int,
) -> dict[str, Any]:
    assignment = int(row.get("assignment", -1))
    first_seat = int(row.get("first_member_seat", -1))
    second_seat = int(row.get("second_member_seat", -1))
    if assignment != expected_assignment:
        raise CrossplayEvidenceError("cross-play assignment identity is missing or duplicated")
    if first_seat != assignment or second_seat != 1 - assignment:
        raise CrossplayEvidenceError("cross-play member-to-seat binding is invalid")
    paired_identity = row.get("paired_deal_identity_sha256")
    physical_identity = row.get("physical_deal_identity_sha256")
    if not _is_sha256(paired_identity) or not _is_sha256(physical_identity):
        raise CrossplayEvidenceError("cross-play deal identity is invalid")
    deal_seed = int(row.get("deal_seed", -1))
    public_chance_seed = int(row.get("public_chance_seed", -1))
    action_seed = int(row.get("action_seed", -1))
    if min(deal_seed, public_chance_seed, action_seed) < 0:
        raise CrossplayEvidenceError("cross-play RNG seed is invalid")
    first_value = float(row.get("first_member_value_chips", math.nan))
    second_value = float(row.get("second_member_value_chips", math.nan))
    if not all(math.isfinite(value) for value in (first_value, second_value)):
        raise CrossplayEvidenceError("cross-play utility is non-finite")
    if abs(first_value + second_value) > 1e-9:
        raise CrossplayEvidenceError("cross-play assignment utility is not zero-sum")
    first_decisions = int(row.get("first_member_decision_count", -1))
    second_decisions = int(row.get("second_member_decision_count", -1))
    first_missing = int(row.get("first_member_missing_policy_count", -1))
    second_missing = int(row.get("second_member_missing_policy_count", -1))
    if (
        min(first_decisions, second_decisions, first_missing, second_missing) < 0
        or first_missing > first_decisions
        or second_missing > second_decisions
    ):
        raise CrossplayEvidenceError("cross-play policy-reach counts are invalid")
    return {
        "assignment": assignment,
        "first_member_seat": first_seat,
        "second_member_seat": second_seat,
        "paired_deal_identity_sha256": paired_identity,
        "physical_deal_identity_sha256": physical_identity,
        "deal_seed": deal_seed,
        "public_chance_seed": public_chance_seed,
        "action_seed": action_seed,
        "first_member_value_chips": first_value,
        "second_member_value_chips": second_value,
        "first_member_decision_count": first_decisions,
        "second_member_decision_count": second_decisions,
        "first_member_missing_policy_count": first_missing,
        "second_member_missing_policy_count": second_missing,
    }


def _canonical_observations(
    raw_observations: Sequence[Mapping[str, Any]],
    *,
    deals_per_assignment: int,
) -> list[dict[str, Any]]:
    if len(raw_observations) != deals_per_assignment:
        raise CrossplayEvidenceError("cross-play has the wrong paired-deal count")
    by_index: dict[int, Mapping[str, Any]] = {}
    for raw in raw_observations:
        deal_index = int(raw.get("deal_index", -1))
        if deal_index in by_index:
            raise CrossplayEvidenceError("cross-play has a duplicate deal index")
        by_index[deal_index] = raw
    if set(by_index) != set(range(deals_per_assignment)):
        raise CrossplayEvidenceError("cross-play deal indices are not complete")

    result: list[dict[str, Any]] = []
    paired_identities: set[str] = set()
    for deal_index in range(deals_per_assignment):
        raw = by_index[deal_index]
        assignment_rows = raw.get("assignments", ())
        if not isinstance(assignment_rows, Sequence) or len(assignment_rows) != 2:
            raise CrossplayEvidenceError("cross-play must contain both seat assignments")
        by_assignment: dict[int, Mapping[str, Any]] = {}
        for assignment_row in assignment_rows:
            if not isinstance(assignment_row, Mapping):
                raise CrossplayEvidenceError("cross-play assignment row is malformed")
            assignment = int(assignment_row.get("assignment", -1))
            if assignment in by_assignment:
                raise CrossplayEvidenceError("cross-play has a duplicate seat assignment")
            by_assignment[assignment] = assignment_row
        if set(by_assignment) != {0, 1}:
            raise CrossplayEvidenceError("cross-play seat assignments are incomplete")
        assignments = [
            _canonical_assignment(by_assignment[index], expected_assignment=index)
            for index in (0, 1)
        ]
        base_fields = (
            "paired_deal_identity_sha256",
            "physical_deal_identity_sha256",
            "deal_seed",
            "public_chance_seed",
            "action_seed",
        )
        if any(assignments[0][field] != assignments[1][field] for field in base_fields):
            raise CrossplayEvidenceError(
                "cross-play seat assignments do not share one base deal/RNG stream"
            )
        paired_identity = assignments[0]["paired_deal_identity_sha256"]
        if paired_identity in paired_identities:
            raise CrossplayEvidenceError("cross-play has a duplicate paired-deal identity")
        paired_identities.add(paired_identity)
        symmetric_advantage = math.fsum(
            assignments[index]["first_member_value_chips"]
            - assignments[index]["second_member_value_chips"]
            for index in (0, 1)
        ) / 2.0
        result.append(
            {
                "deal_index": deal_index,
                "assignments": assignments,
                "symmetric_advantage_chips": symmetric_advantage,
            }
        )
    return result


def build_crossplay_pair_evidence(
    *,
    root_id: str,
    first_member: Mapping[str, Any],
    second_member: Mapping[str, Any],
    deals_per_assignment: int,
    raw_observations: Sequence[Mapping[str, Any]],
    pairing_unit: str,
) -> dict[str, Any]:
    deals_per_assignment = int(deals_per_assignment)
    pairing_unit = str(pairing_unit)
    if deals_per_assignment <= 0:
        raise CrossplayEvidenceError("cross-play panel must be nonempty")
    first = _canonical_member(first_member)
    second = _canonical_member(second_member)
    if _member_key(first) == _member_key(second):
        raise CrossplayEvidenceError("cross-play pair repeats one member")
    observations = _canonical_observations(
        raw_observations, deals_per_assignment=deals_per_assignment
    )
    first_values: list[float] = []
    second_values: list[float] = []
    missing = [0, 0]
    decisions = [0, 0]
    base_stream_ledger = []
    assignment_stream_ledger = []
    for observation in observations:
        assignments = observation["assignments"]
        for assignment in assignments:
            first_values.append(float(assignment["first_member_value_chips"]))
            second_values.append(float(assignment["second_member_value_chips"]))
            decisions[0] += int(assignment["first_member_decision_count"])
            decisions[1] += int(assignment["second_member_decision_count"])
            missing[0] += int(assignment["first_member_missing_policy_count"])
            missing[1] += int(assignment["second_member_missing_policy_count"])
            assignment_stream_ledger.append(
                {
                    "deal_index": observation["deal_index"],
                    "assignment": assignment["assignment"],
                    **{
                        field: assignment[field]
                        for field in (
                            "paired_deal_identity_sha256",
                            "physical_deal_identity_sha256",
                            "deal_seed",
                            "public_chance_seed",
                            "action_seed",
                        )
                    },
                }
            )
        base_stream_ledger.append(
            {
                "deal_index": observation["deal_index"],
                **{
                    field: assignments[0][field]
                    for field in (
                        "paired_deal_identity_sha256",
                        "physical_deal_identity_sha256",
                        "deal_seed",
                        "public_chance_seed",
                        "action_seed",
                    )
                },
            }
        )
    return with_self_hash(
        {
            "schema_version": "stage21_member_crossplay_pair_v4",
            "root_id": str(root_id),
            "first_member": first,
            "second_member": second,
            "deals_per_assignment": deals_per_assignment,
            "paired_deal_count": len(observations),
            "raw_assignment_observation_count": len(observations) * 2,
            "first_member_summary": _summary(first_values),
            "second_member_summary": _summary(second_values),
            "symmetric_advantage_summary": _summary(
                [row["symmetric_advantage_chips"] for row in observations]
            ),
            "missing_policy_reach_fraction_by_member": [
                missing[index] / max(1, decisions[index]) for index in (0, 1)
            ],
            "missing_policy_count_by_member": missing,
            "decision_count_by_member": decisions,
            "raw_observations": observations,
            "raw_observation_ledger_sha256": sha256_json(observations),
            "deal_and_rng_ledger_sha256": sha256_json(assignment_stream_ledger),
            "paired_base_stream_ledger": base_stream_ledger,
            "paired_base_stream_ledger_sha256": sha256_json(base_stream_ledger),
            "seat_assignment_stream_pairing_gate_pass": True,
            "paired_common_random_numbers_across_seat_assignments": True,
            "symmetric_advantage_pairing_unit": pairing_unit,
            "exact_best_response_claim": False,
        }
    )


def recompute_crossplay_pair_evidence(pair: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_hash(pair)
    recomputed = build_crossplay_pair_evidence(
        root_id=str(pair.get("root_id", "")),
        first_member=pair.get("first_member", {}),
        second_member=pair.get("second_member", {}),
        deals_per_assignment=int(pair.get("deals_per_assignment", -1)),
        raw_observations=pair.get("raw_observations", ()),
        pairing_unit=str(pair.get("symmetric_advantage_pairing_unit", "")),
    )
    if dict(pair) != recomputed:
        raise CrossplayEvidenceError(
            "cross-play pair metrics or committed metadata are not reproducible"
        )
    return recomputed


def _bonferroni_two_sided_z(familywise_alpha: float, comparisons: int) -> float:
    if not 0.0 < familywise_alpha < 1.0 or comparisons <= 0:
        raise CrossplayEvidenceError("cross-play confidence inputs are invalid")
    return NormalDist().inv_cdf(
        1.0 - familywise_alpha / (2.0 * comparisons)
    )


def build_root_crossplay_evidence(
    *,
    root_id: str,
    pair_results: Sequence[Mapping[str, Any]],
    expected_members: Sequence[Mapping[str, Any]] | None,
    deals_per_assignment: int,
    familywise_alpha: float,
    effective_stack_bb: float,
    limit_effective_stack_fraction_per_hand: float,
    maximum_missing_policy_reach_fraction: float,
    pairing_unit: str,
) -> dict[str, Any]:
    deals_per_assignment = int(deals_per_assignment)
    familywise_alpha = float(familywise_alpha)
    effective_stack_bb = float(effective_stack_bb)
    limit_effective_stack_fraction_per_hand = float(
        limit_effective_stack_fraction_per_hand
    )
    maximum_missing_policy_reach_fraction = float(
        maximum_missing_policy_reach_fraction
    )
    pairing_unit = str(pairing_unit)
    canonical_pairs = [recompute_crossplay_pair_evidence(pair) for pair in pair_results]
    if len(canonical_pairs) != 15:
        raise CrossplayEvidenceError("cross-play must contain all 15 member pairs")
    if deals_per_assignment != CROSSPLAY_DEALS_PER_ASSIGNMENT:
        raise CrossplayEvidenceError(
            "released cross-play requires exactly "
            f"{CROSSPLAY_DEALS_PER_ASSIGNMENT} paired deals"
        )
    if not math.isfinite(effective_stack_bb) or effective_stack_bb <= 0.0:
        raise CrossplayEvidenceError("cross-play effective stack is invalid")
    if (
        not 0.0 < familywise_alpha < 1.0
        or not 0.0 < limit_effective_stack_fraction_per_hand <= 1.0
        or not 0.0 <= maximum_missing_policy_reach_fraction < 1.0
    ):
        raise CrossplayEvidenceError("cross-play gate constants are invalid")

    members: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    observed_pair_keys: set[
        tuple[tuple[str, str, int, str], tuple[str, str, int, str]]
    ] = set()
    for pair in canonical_pairs:
        if pair["root_id"] != str(root_id):
            raise CrossplayEvidenceError("cross-play pair is bound to another root")
        if pair["deals_per_assignment"] != deals_per_assignment:
            raise CrossplayEvidenceError("cross-play pair has the wrong deal count")
        if pair["symmetric_advantage_pairing_unit"] != pairing_unit:
            raise CrossplayEvidenceError("cross-play pair uses the wrong pairing unit")
        first = _canonical_member(pair["first_member"])
        second = _canonical_member(pair["second_member"])
        first_key = _member_key(first)
        second_key = _member_key(second)
        members[first_key] = first
        members[second_key] = second
        pair_key = tuple(sorted((first_key, second_key)))
        if pair_key in observed_pair_keys:
            raise CrossplayEvidenceError("cross-play contains a duplicate member pair")
        observed_pair_keys.add(pair_key)
    if len(members) != 6:
        raise CrossplayEvidenceError("cross-play must bind exactly six members")
    expected_pair_keys = {
        tuple(sorted(pair)) for pair in combinations(sorted(members), 2)
    }
    if observed_pair_keys != expected_pair_keys:
        raise CrossplayEvidenceError("cross-play member-pair coverage is incomplete")

    canonical_members = [members[key] for key in sorted(members)]
    if expected_members is not None:
        if len(expected_members) != 6:
            raise CrossplayEvidenceError("expected cross-play member authority is incomplete")
        expected_by_lane: dict[str, Mapping[str, Any]] = {}
        for expected in expected_members:
            lane_id = str(expected.get("lane_id", ""))
            if not lane_id or lane_id in expected_by_lane:
                raise CrossplayEvidenceError("expected member lane identities are invalid")
            expected_by_lane[lane_id] = expected
        if set(expected_by_lane) != {member["lane_id"] for member in canonical_members}:
            raise CrossplayEvidenceError("cross-play members do not match the bound six lanes")
        for member in canonical_members:
            expected = expected_by_lane[member["lane_id"]]
            if (
                str(expected.get("algorithm")) != member["algorithm"]
                or int(expected.get("seed", -1)) != member["seed"]
                or (
                    expected.get("policy_sha256") is not None
                    and expected.get("policy_sha256") != member["policy_sha256"]
                )
            ):
                raise CrossplayEvidenceError("cross-play member metadata is forged")

    canonical_pairs.sort(
        key=lambda pair: tuple(
            sorted((_member_key(pair["first_member"]), _member_key(pair["second_member"])))
        )
    )
    z = _bonferroni_two_sided_z(familywise_alpha, len(canonical_pairs))
    limit = 100.0 * effective_stack_bb * limit_effective_stack_fraction_per_hand
    pair_gates = []
    for pair in canonical_pairs:
        summary = pair["symmetric_advantage_summary"]
        bound = abs(float(summary["mean_bb_per_100"])) + z * float(
            summary["standard_error_bb_per_100"]
        )
        missing_gate = max(pair["missing_policy_reach_fraction_by_member"]) <= (
            maximum_missing_policy_reach_fraction
        )
        stream_gate = (
            pair["seat_assignment_stream_pairing_gate_pass"] is True
            and pair["paired_deal_count"] == deals_per_assignment
            and pair["raw_assignment_observation_count"] == 2 * deals_per_assignment
            and pair["symmetric_advantage_pairing_unit"] == pairing_unit
        )
        pair_gates.append(
            {
                "pair_self_sha256": pair["self_sha256"],
                "first_member_lane_id": pair["first_member"]["lane_id"],
                "second_member_lane_id": pair["second_member"]["lane_id"],
                "sample_count": int(summary["sample_count"]),
                "mean_bb_per_100": float(summary["mean_bb_per_100"]),
                "standard_error_bb_per_100": float(
                    summary["standard_error_bb_per_100"]
                ),
                "multiplicity_adjusted_absolute_upper_bb_per_100": bound,
                "limit_bb_per_100": limit,
                "effective_stack_bb": effective_stack_bb,
                "limit_effective_stack_fraction_per_hand": (
                    limit_effective_stack_fraction_per_hand
                ),
                "maximum_missing_policy_reach_fraction": (
                    maximum_missing_policy_reach_fraction
                ),
                "missing_reach_gate_pass": missing_gate,
                "paired_seat_assignment_streams_gate_pass": stream_gate,
                "gate_pass": bound <= limit and missing_gate and stream_gate,
            }
        )
    return with_self_hash(
        {
            "schema_version": "stage21_root_crossplay_evidence_v3",
            "root_id": str(root_id),
            "member_count": len(canonical_members),
            "member_bindings": canonical_members,
            "pair_count": len(canonical_pairs),
            "deals_per_assignment": deals_per_assignment,
            "raw_assignment_observation_count": (
                len(canonical_pairs) * deals_per_assignment * 2
            ),
            "all_unique_unordered_member_pairs_complete": True,
            "confidence_policy": "bonferroni_two_sided_normal_absolute_bound",
            "familywise_alpha": familywise_alpha,
            "effective_stack_bb": effective_stack_bb,
            "max_absolute_symmetric_advantage_effective_stack_fraction_per_hand": (
                limit_effective_stack_fraction_per_hand
            ),
            "maximum_missing_policy_reach_fraction": (
                maximum_missing_policy_reach_fraction
            ),
            "symmetric_advantage_pairing_unit": pairing_unit,
            "pair_results": canonical_pairs,
            "pair_gates": pair_gates,
            "gate_pass": all(row["gate_pass"] for row in pair_gates),
            "archive_gate_boolean_is_not_authoritative": True,
            "exact_best_response_claim": False,
        }
    )


def validate_and_recompute_root_crossplay_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_root_id: str,
    expected_members: Sequence[Mapping[str, Any]],
    deals_per_assignment: int,
    familywise_alpha: float,
    effective_stack_bb: float,
    limit_effective_stack_fraction_per_hand: float,
    maximum_missing_policy_reach_fraction: float,
    pairing_unit: str,
) -> dict[str, Any]:
    validate_self_hash(evidence)
    recomputed = build_root_crossplay_evidence(
        root_id=expected_root_id,
        pair_results=evidence.get("pair_results", ()),
        expected_members=expected_members,
        deals_per_assignment=deals_per_assignment,
        familywise_alpha=familywise_alpha,
        effective_stack_bb=effective_stack_bb,
        limit_effective_stack_fraction_per_hand=(
            limit_effective_stack_fraction_per_hand
        ),
        maximum_missing_policy_reach_fraction=(
            maximum_missing_policy_reach_fraction
        ),
        pairing_unit=pairing_unit,
    )
    if dict(evidence) != recomputed:
        raise CrossplayEvidenceError(
            "root cross-play metrics, limits, pair gates, or aggregate gate are forged"
        )
    return recomputed


def validate_root_crossplay_once(
    evidence: Mapping[str, Any],
    *,
    expected_root_id: str,
    expected_members: Sequence[Mapping[str, Any]],
    deals_per_assignment: int,
    familywise_alpha: float,
    effective_stack_bb: float,
    limit_effective_stack_fraction_per_hand: float,
    maximum_missing_policy_reach_fraction: float,
    pairing_unit: str,
) -> tuple[dict[str, Any], ValidatedCrossplayReceipt]:
    recomputed = validate_and_recompute_root_crossplay_evidence(
        evidence,
        expected_root_id=expected_root_id,
        expected_members=expected_members,
        deals_per_assignment=deals_per_assignment,
        familywise_alpha=familywise_alpha,
        effective_stack_bb=effective_stack_bb,
        limit_effective_stack_fraction_per_hand=(
            limit_effective_stack_fraction_per_hand
        ),
        maximum_missing_policy_reach_fraction=(
            maximum_missing_policy_reach_fraction
        ),
        pairing_unit=pairing_unit,
    )
    member_identity = sha256_json(
        sorted(
            (_canonical_member(member) for member in expected_members),
            key=lambda row: row["lane_id"],
        )
    )
    return recomputed, ValidatedCrossplayReceipt(
        _VALIDATED_RECEIPT_TOKEN,
        root_id=expected_root_id,
        evidence_self_sha256=recomputed["self_sha256"],
        member_identity_sha256=member_identity,
        gate_pass=recomputed["gate_pass"] is True,
    )
