from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cards import (
    all_preflop_classes,
    board_features,
    classify_public_transition,
    normalize_cards,
)
from .common import ALGORITHMS, REVISION, SCIENCE_SEEDS, STAGES, sha256_json, with_self_hash
from .coverage import weighted_nut_advantage_transition
from .game import (
    ActionRecord,
    HoldemState,
    make_street_start,
    state_snapshot,
)


STACK_CHIPS = {
    "short_10_25bb": 2_000,
    "intermediate_40_80bb": 6_000,
    "deep_120_200bb": 15_000,
}
POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB = (20, 60, 100, 150)
POSTFLOP_STACK_CONFIGURATIONS = (
    "equal_effective",
    "hero_covers_1_5x",
    "opponent_covers_1_5x",
)
TRANSITION_COVERAGE_LABELS = (
    "blank",
    "overcard",
    "undercard",
    "pairing",
    "flush_completion",
    "new_flush_draw",
    "straight_completion",
    "new_gutshot",
    "new_oesd",
    "counterfeit",
    "nut_advantage_reversal",
)
SPR_TARGETS = {
    "spr_lt_1": (0.30, 0.50, 0.75),
    "spr_1_to_3": (1.25, 2.00, 2.75),
    "spr_3_to_8": (3.75, 5.50, 7.25),
    "spr_gt_8": (9.00, 12.00, 16.00),
}
POT_LINE_INDEX = {
    "limped": 0,
    "single_raised": 0,
    "iso_raised": 1,
    "three_bet": 1,
    "four_bet": 2,
    "post_all_in_aftermath": 2,
}
BET_FRACTIONS = {
    "small_25_33pct": (0.25, 0.33, 0.75, 1.00, 1.50),
    "medium_40_60pct": (0.40, 0.50, 0.75, 1.00, 1.50),
    "large_66_85pct": (0.66, 0.75, 0.85, 1.00, 1.50),
    "pot_90_110pct": (0.75, 0.90, 1.00, 1.10, 1.50),
    "overbet_125_200pct": (0.75, 1.00, 1.25, 1.50, 2.00),
    "jam": (0.50, 0.75, 1.00, 1.50, 2.00),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hero_player(position: str) -> int:
    text = position.lower()
    if "hero_btn_sb" in text:
        return 0
    if "hero_bb" in text:
        return 1
    return 1 if "oop" in text else 0


def _base_contribution(source: Mapping[str, Any], start: int) -> int:
    band = str(source.get("spr_band", "spr_3_to_8"))
    targets = SPR_TARGETS.get(band, SPR_TARGETS["spr_3_to_8"])
    index = POT_LINE_INDEX.get(str(source.get("pot_line", "single_raised")), 1)
    target = targets[index]
    # target = remaining / pot = (start-c)/(2c)
    contribution = round(start / (2.0 * target + 1.0))
    return max(1, min(contribution, start - 1))


def _derived_spr(state: HoldemState) -> float:
    return min(state.stacks) / state.pot if state.pot else math.inf


def _derived_spr_band(state: HoldemState) -> str:
    spr = _derived_spr(state)
    if spr < 1.0:
        return "spr_lt_1"
    if spr < 3.0:
        return "spr_1_to_3"
    if spr <= 8.0:
        return "spr_3_to_8"
    return "spr_gt_8"


def _derived_stack_band(state: HoldemState) -> str:
    effective_starting_bb = min(state.starting_stacks) / 100.0
    if effective_starting_bb <= 25.0:
        return "short_10_25bb"
    if effective_starting_bb <= 80.0:
        return "intermediate_40_80bb"
    return "deep_120_200bb"


def _legal_family(state: HoldemState) -> str:
    if state.actor is None:
        return "forced_runout_no_decision"
    to_call = state.current_bet - state.street_contributions[state.actor]
    if to_call <= 0:
        return "checked_to_check_or_bet"
    opponent = 1 - state.actor
    if state.stacks[opponent] == 0:
        return "jam_or_fold"
    if state.street_contributions[state.actor] > 0:
        return "facing_raise_fold_call_reraise"
    return "facing_bet_fold_call_raise"


def _postflop_starting_stacks(
    source: Mapping[str, Any], hero: int
) -> tuple[int, int]:
    anchor_bb = int(source.get("stack_anchor_bb", 0))
    if anchor_bb:
        if anchor_bb not in POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB:
            raise ValueError(f"unsupported postflop stack anchor: {anchor_bb}bb")
        effective = anchor_bb * 100
    else:
        effective = STACK_CHIPS.get(str(source.get("stack_band")), 6_000)
    configuration = str(source.get("stack_configuration", "equal_effective"))
    if configuration not in POSTFLOP_STACK_CONFIGURATIONS:
        raise ValueError(f"unsupported postflop stack configuration: {configuration}")
    if configuration == "equal_effective":
        return effective, effective
    covering = round(effective * 1.5)
    hero_stack = covering if configuration == "hero_covers_1_5x" else effective
    opponent_stack = effective if configuration == "hero_covers_1_5x" else covering
    values = [0, 0]
    values[hero] = hero_stack
    values[1 - hero] = opponent_stack
    return tuple(values)  # type: ignore[return-value]


def _postflop_state(source: Mapping[str, Any], street: str) -> HoldemState:
    board = tuple(source["representative_board"])
    normalize_cards(board)
    hero = _hero_player(str(source.get("position", "hero_btn_sb_ip")))
    starting_stacks = _postflop_starting_stacks(source, hero)
    effective_start = min(starting_stacks)
    button = 0
    base = _base_contribution(source, effective_start)
    forced = source.get("decision_kind") == "forced_public_runout_evaluation"
    fractions = BET_FRACTIONS.get(
        str(source.get("action_size_family", "medium_40_60pct")),
        BET_FRACTIONS["medium_40_60pct"],
    )
    if source.get("exact_scope") is True:
        # Restricted exact anchors enumerate the full action/chance tree. Their
        # explicit canary abstraction is check/call/fold/effective-stack jam;
        # broader multi-size roots remain in the approximate partition.
        fractions = ()
    if forced:
        return make_street_start(
            street=street,
            board=board,
            starting_stacks=starting_stacks,
            contributions=(effective_start, effective_start),
            button_player=button,
            forced_runout=True,
            bet_fractions=fractions,
        )

    family = str(source.get("legal_action_family", "checked_to_check_or_bet"))
    state = make_street_start(
        street=street,
        board=board,
        starting_stacks=starting_stacks,
        contributions=(base, base),
        button_player=button,
        actor=hero,
        bet_fractions=fractions,
    )
    if family == "checked_to_check_or_bet":
        if hero == button:
            checker = 1 - hero
            record = ActionRecord(street, checker, "check", 0, 0, False, False)
            state = replace(
                state,
                actor=hero,
                checks_in_row=1,
                acted_since_full_raise=tuple(index == checker for index in range(2)),
                history=(record,),
            )
        state.validate()
        return state

    opponent = 1 - hero
    pot = state.pot
    fraction = fractions[1]
    bet = max(100, round(pot * fraction))
    bet = min(bet, max(1, state.stacks[opponent] // 2))
    total_contributions = list(state.contributions)
    remaining = list(state.stacks)
    street_contributions = [0, 0]
    history: list[ActionRecord] = []
    if family == "facing_bet_fold_call_raise":
        if hero != button:
            history.append(ActionRecord(street, hero, "check", 0, 0, False, False))
        street_contributions[opponent] = bet
        total_contributions[opponent] += bet
        remaining[opponent] -= bet
        history.append(ActionRecord(street, opponent, "bet", bet, bet, True, True))
        last_raise = bet
        acted = [False, False]
        acted[opponent] = True
    elif family == "facing_raise_fold_call_reraise":
        opening_bet = max(100, round(pot * fractions[0]))
        opening_bet = min(
            opening_bet,
            max(1, remaining[hero] // 3),
            max(1, remaining[opponent] // 3),
        )
        raise_to = max(opening_bet * 2, opening_bet + 100)
        raise_to = min(raise_to, remaining[opponent])
        if raise_to <= opening_bet:
            raise_to = min(remaining[opponent], opening_bet + 1)
        street_contributions[hero] = opening_bet
        street_contributions[opponent] = raise_to
        total_contributions[hero] += opening_bet
        total_contributions[opponent] += raise_to
        remaining[hero] -= opening_bet
        remaining[opponent] -= raise_to
        if hero == button:
            history.append(ActionRecord(street, opponent, "check", 0, 0, False, False))
        history.extend(
            (
                ActionRecord(street, hero, "bet", opening_bet, opening_bet, True, True),
                ActionRecord(
                    street,
                    opponent,
                    "raise",
                    raise_to,
                    raise_to,
                    raise_to - opening_bet >= opening_bet,
                    raise_to - opening_bet >= opening_bet,
                ),
            )
        )
        last_raise = max(1, raise_to - opening_bet)
        acted = [False, False]
        acted[opponent] = True
    elif family == "jam_or_fold":
        if hero != button:
            history.append(ActionRecord(street, hero, "check", 0, 0, False, False))
        jam = remaining[opponent]
        street_contributions[opponent] = jam
        total_contributions[opponent] += jam
        remaining[opponent] = 0
        history.append(ActionRecord(street, opponent, "bet", jam, jam, True, True))
        last_raise = max(100, jam)
        acted = [False, False]
        acted[opponent] = True
    else:
        raise ValueError(f"unknown historical legal-action family: {family}")

    state = HoldemState(
        street=street,
        board=board,
        button_player=button,
        actor=hero,
        starting_stacks=starting_stacks,
        stacks=tuple(remaining),
        contributions=tuple(total_contributions),
        street_contributions=tuple(street_contributions),
        current_bet=max(street_contributions),
        last_full_raise=last_raise,
        acted_since_full_raise=tuple(acted),
        checks_in_row=0,
        folded=(False, False),
        terminal=False,
        all_in_runout=False,
        history=tuple(history),
        bet_fractions=fractions,
    )
    state.validate()
    if _legal_family(state) != family:
        raise AssertionError(f"failed to reconstruct {family}: {_legal_family(state)}")
    return state


def _preflop_history(line_id: str, contributions: tuple[int, int]) -> tuple[ActionRecord, ...]:
    c0, c1 = contributions
    if line_id == "BB_VS_LIMP":
        return (ActionRecord("preflop", 0, "call", 50, 100, False, False),)
    if line_id == "BB_VS_SB_OPEN":
        return (ActionRecord("preflop", 0, "raise", c0 - 50, c0, True, True),)
    if line_id == "SB_VS_BB_LIMP_RAISE":
        return (
            ActionRecord("preflop", 0, "call", 50, 100, False, False),
            ActionRecord("preflop", 1, "raise", c1 - 100, c1, True, True),
        )
    if line_id == "SB_VS_BB_THREE_BET":
        return (
            ActionRecord("preflop", 0, "raise", 200, 250, True, True),
            ActionRecord("preflop", 1, "raise", c1 - 100, c1, True, True),
        )
    if line_id == "BB_VS_SB_FOUR_BET":
        return (
            ActionRecord("preflop", 0, "raise", 200, 250, True, True),
            ActionRecord("preflop", 1, "raise", 750, 850, True, True),
            ActionRecord("preflop", 0, "raise", c0 - 250, c0, True, True),
        )
    if line_id == "ALL_IN_CLOSED_TO_FLOP":
        return (
            ActionRecord("preflop", 0, "raise", c0 - 50, c0, True, True),
            ActionRecord("preflop", 1, "call", c1 - 100, c1, False, False),
        )
    raise ValueError(f"unknown corrected preflop line: {line_id}")


def _preflop_state(correction: Mapping[str, Any]) -> HoldemState:
    source = correction["proposed_executable_state"]
    starting = tuple(round(float(value) * 100) for value in source["starting_stacks_bb"])
    contributions = tuple(round(float(value) * 100) for value in source["contributions_bb"])
    stacks = tuple(starting[index] - contributions[index] for index in range(2))
    actor = source.get("actor")
    forced = bool(source["forced_runout"])
    current = max(contributions)
    minimum_raise_to = source.get("minimum_raise_to_bb")
    if minimum_raise_to is None:
        last_raise = 100
    else:
        last_raise = max(1, round(float(minimum_raise_to) * 100) - current)
    acted = [False, False]
    if actor is not None:
        acted[1 - int(actor)] = True
    state = HoldemState(
        street="preflop",
        board=(),
        button_player=int(source["button_player"]),
        actor=None if forced else int(actor),
        starting_stacks=starting,
        stacks=stacks,
        contributions=contributions,
        street_contributions=contributions,
        current_bet=current,
        last_full_raise=last_raise,
        acted_since_full_raise=tuple(acted),
        checks_in_row=0,
        folded=(False, False),
        terminal=False,
        all_in_runout=forced,
        history=_preflop_history(str(source["line_id"]), contributions),
        bet_fractions=(0.33, 0.50, 0.75, 1.00, 1.50),
    )
    state.validate()
    pending = 0 if state.actor is None else state.current_bet - state.street_contributions[state.actor]
    if pending != round(float(source["pending_call_bb"]) * 100):
        raise AssertionError("corrected preflop pending-call state does not replay")
    return state


def _kind(source: Mapping[str, Any]) -> str:
    if source.get("decision_kind") == "forced_public_runout_evaluation" or source.get(
        "source_decision_kind"
    ) == "forced_public_runout_evaluation":
        return "forced"
    if source.get("exact_scope") is True:
        return "restricted_exact"
    return "approximate"


def _causal_axes(stage: str) -> dict[str, str]:
    axes = {
        "board": "public_chance_payoff_ranges_and_blockers",
        "action_history_family": "player_specific_action_posterior_range",
        "action_size_family": "legal_bet_abstraction",
        "initiative_role": "player_specific_action_posterior_range",
        "legal_action_family": "replayed_executable_public_state",
        "position": "hero_role_actor_order_and_player_specific_range",
        "pot_line": "within_band_pot_and_spr_construction",
        "range_family": "player_specific_private_chance_distribution",
        "opponent_range_family": "player_specific_private_chance_distribution",
        "stack_band": "starting_stack_and_absolute_utility_scale",
        "stack_anchor_bb": "effective_starting_stack_and_spr_scale",
        "stack_configuration": "equal_or_explicit_covering_stack_ownership",
        "spr_band": "executable_effective_stack_to_pot_ratio",
        "opponent_style": "noncausal_source_stratification_full_14_family_panel_runs_for_every_root",
        "safe_exploit_control": "noncausal_source_stratification_all_3_safe_controls_run_for_every_root",
        "partition": "noncausal_split_metadata_excluded_from_strategic_coverage",
        "producer_token": "noncausal_transport_metadata_excluded_from_strategic_coverage",
        "historical_root_id": "noncausal_provenance_alias_excluded_from_strategic_coverage",
    }
    if stage in {"turn", "river"}:
        axes.update(
            {
                "parent_public_board": "exact_previous_street_public_state",
                "transition_card": "exact_public_chance_identity",
                "transition_classification": (
                    "board_structure_and_fixed_action_posterior_runout_effect"
                ),
            }
        )
    if stage == "preflop":
        axes["preflop_hand_class_targets"] = (
            "all_169_class_stratification_groups_not_point_hands"
        )
    return axes


def _normalized_root(
    *,
    stage: str,
    source: Mapping[str, Any],
    index: int,
    state: HoldemState,
    root_id: str,
    correction: Mapping[str, Any] | None = None,
    include_nut_profile: bool = True,
) -> dict[str, Any]:
    kind = _kind(correction or source)
    executable_stack_band = _derived_stack_band(state)
    executable_spr_band = _derived_spr_band(state)
    range_contract = (
        dict(correction["proposed_range_contract"])
        if correction is not None
        else {
            "player_0_family": str(source.get("range_family", "equilibrium_like")),
            "player_1_family": "equilibrium_like",
            "target_class_bias_multiplier": 1.25,
            "target_class_is_point_hand": False,
        }
    )
    if stage == "preflop":
        range_contract.update(
            {
                "target_class_grouping": "complete_169_class_covering_partition",
                "target_class_weighting": (
                    "per_physical_combo; class mass therefore preserves 6_pair_"
                    "4_suited_12_offsuit_multiplicity_before_posterior_factors"
                ),
                "all_1326_physical_combos_remain_positive_support": True,
            }
        )
    board = list(state.board)
    parent_board = source.get("parent_public_board")
    transition_card = source.get("transition_card")
    transition = None
    if stage in {"turn", "river"}:
        if not isinstance(parent_board, (list, tuple)) or not isinstance(
            transition_card, str
        ):
            raise ValueError(f"{stage} root lacks exact parent/transition lineage")
        transition = classify_public_transition(parent_board, transition_card)
        if transition["child_board"] != board:
            raise ValueError("transition lineage does not reconstruct final board")
    starting = tuple(state.starting_stacks)
    effective_stack_bb = min(starting) // 100
    hero_player = (
        _hero_player(str(source.get("position", "hero_btn_sb_ip")))
        if stage != "preflop"
        else 0
    )
    hero_range_family = str(
        source.get("range_family")
        or (correction or {}).get("source_range_family")
        or "equilibrium_like"
    )
    opponent_range_family = "equilibrium_like"
    # Seat ownership is resolved only after the executable hero seat is known.
    # Historical correction documents described the hero's range, not a fixed
    # player-0 range.  Binding those fields before resolving position reverses
    # the intended posterior on every OOP root.
    range_contract["player_0_family"] = (
        hero_range_family if hero_player == 0 else opponent_range_family
    )
    range_contract["player_1_family"] = (
        hero_range_family if hero_player == 1 else opponent_range_family
    )
    range_contract["hero_player"] = hero_player
    range_contract["hero_family"] = hero_range_family
    range_contract["opponent_family"] = opponent_range_family
    range_contract["seat_binding_rule"] = "hero_resolved_before_physical_player_binding"
    if starting[0] == starting[1]:
        stack_configuration = "equal_effective"
    elif starting[hero_player] > starting[1 - hero_player]:
        stack_configuration = "hero_covers_1_5x"
    else:
        stack_configuration = "opponent_covers_1_5x"
    target_classes = list(source.get("preflop_hand_class_targets") or ())
    if not target_classes:
        target = source.get("preflop_hand_class_target") or (
            correction.get("source_preflop_hand_class_target") if correction else None
        )
        target_classes = [] if target is None else [str(target)]
    root = {
        "schema_version": "stage21_repaired_executable_root_v3",
        "revision": REVISION,
        "stage": stage,
        "street": stage,
        "root_id": root_id,
        "root_index": index,
        "root_kind": kind,
        "historical_root_id": source.get("root_id") or source.get("source_root_id"),
        "historical_root_semantic_sha256": source.get("root_semantic_sha256")
        or source.get("source_root_semantic_sha256"),
        "historical_labels": {
            key: source.get(key)
            for key in (
                "action_history_family",
                "action_size_family",
                "initiative_role",
                "legal_action_family",
                "opponent_style",
                "partition",
                "position",
                "pot_line",
                "producer_token",
                "range_family",
                "safe_exploit_control",
                "spr_band",
                "stack_band",
            )
        },
        "representative_board": board,
        "board_classification": None if not board else board_features(board),
        "parent_public_board": None if parent_board is None else list(parent_board),
        "transition_card": transition_card,
        "transition_classification": transition,
        "hero_player": hero_player,
        "action_history_family": source.get("action_history_family")
        or f"corrected_{correction['proposed_executable_state']['line_id'].lower()}",
        "action_size_family": source.get("action_size_family") or "multi_size_abstraction",
        "initiative_role": source.get("initiative_role") or "preflop_line_actor",
        "position": (
            (
                "hero_btn_sb_ip"
                if _hero_player(str(source.get("position", "hero_btn_sb_ip"))) == 0
                else "hero_bb_oop"
            )
            if stage != "preflop"
            else "SB_BTN_vs_BB"
        ),
        "historical_position_label_repaired": (
            stage != "preflop"
            and source.get("position")
            not in {"hero_btn_sb_ip", "hero_bb_oop"}
        ),
        "stack_band": executable_stack_band,
        "stack_anchor_bb": effective_stack_bb,
        "covering_stack_bb": max(starting) / 100.0,
        "stack_configuration": stack_configuration,
        "spr_band": executable_spr_band,
        "historical_stack_band_label_repaired": (
            source.get("stack_band") != executable_stack_band
        ),
        "historical_spr_band_label_repaired": (
            source.get("spr_band") != executable_spr_band
        ),
        "pot_line": source.get("pot_line") or correction["proposed_executable_state"]["line_id"],
        "range_family": hero_range_family,
        "opponent_range_family": opponent_range_family,
        "opponent_style": source.get("opponent_style") or "balanced_control",
        "safe_exploit_control": source.get("safe_exploit_control") or "confidence_abstain",
        "partition": source.get("partition") or correction["source_partition"],
        "preflop_hand_class_target": target_classes[0] if target_classes else None,
        "preflop_hand_class_targets": target_classes,
        "range_contract": range_contract,
        "executable_state": state_snapshot(state),
        "derived_legal_action_family": _legal_family(state),
        "derived_spr": _derived_spr(state),
        "causal_axes": _causal_axes(stage),
        "solve_contract": {
            "algorithms": list(ALGORITHMS),
            "seeds": list(SCIENCE_SEEDS),
            "lanes_per_root": 6,
            "minimum_passing_seeds_by_algorithm": {algorithm: 2 for algorithm in ALGORITHMS},
            "replication_axis": (
                "deterministic_numerical_accumulation_order"
                if kind == "restricted_exact"
                else "independent_stochastic_solver_seed"
            ),
            "independent_random_seed_claim": kind != "restricted_exact",
            "third_seed_role": (
                "third_numerical_order_stability_check"
                if kind == "restricted_exact"
                else "preregistered_stability_and_reserve_replication"
            ),
            "fresh_regrets_required": True,
            "fresh_strategy_sum_required": True,
            "historical_checkpoint_resume_allowed": False,
        },
        "depth_limit": STAGES[stage].depth_limit,
        "continuation_authority_required": STAGES[stage].depth_limit is not None,
        "fresh_regrets_required": kind != "forced",
        "fresh_strategy_sum_required": kind != "forced",
        "historical_policy_reuse_allowed": False,
        "teacher_eligible_only_after_strict_accept": True,
        "perfect_exploitation_claim": False,
        "universal_equilibrium_claim": False,
    }
    if transition is not None and include_nut_profile:
        nut_profile = weighted_nut_advantage_transition(
            root, state, transition["parent_board"]
        )
        transition["range_weighted_nut_advantage"] = nut_profile
        transition["flags"]["nut_advantage_reversal"] = nut_profile[
            "nut_advantage_reversal"
        ]
        transition["flags"]["straight_completion"] = bool(
            transition["flags"]["straight_completion"]
            or nut_profile["range_straight_completion"]
        )
        transition["flags"]["flush_completion"] = bool(
            transition["flags"]["flush_completion"]
            or nut_profile["range_flush_completion"]
        )
        transition["flags"]["blank"] = not any(
            transition["flags"][label]
            for label in (
                "pairing",
                "flush_completion",
                "new_flush_draw",
                "straight_completion",
                "new_gutshot",
                "new_oesd",
                "counterfeit",
                "nut_advantage_reversal",
            )
        )
        transition["labels"] = [
            name for name, enabled in transition["flags"].items() if enabled
        ] or ["between_rank_nonblank"]
    body = dict(root)
    root["root_semantic_sha256"] = sha256_json(body)
    return root


def _attach_transition_nut_profile(root: dict[str, Any]) -> dict[str, Any]:
    transition = root.get("transition_classification")
    if transition is None:
        return root
    if "range_weighted_nut_advantage" not in transition:
        state = HoldemState(
            street=str(root["street"]),
            board=tuple(root["representative_board"]),
            button_player=int(root["executable_state"]["button_player"]),
            actor=root["executable_state"]["actor"],
            starting_stacks=tuple(root["executable_state"]["starting_stacks"]),
            stacks=tuple(root["executable_state"]["stacks"]),
            contributions=tuple(root["executable_state"]["contributions"]),
            street_contributions=tuple(
                root["executable_state"]["street_contributions"]
            ),
            current_bet=int(root["executable_state"]["current_bet"]),
            last_full_raise=int(root["executable_state"]["last_full_raise"]),
            acted_since_full_raise=tuple(
                root["executable_state"]["acted_since_full_raise"]
            ),
            checks_in_row=int(root["executable_state"]["checks_in_row"]),
            folded=tuple(root["executable_state"]["folded"]),
            terminal=bool(root["executable_state"]["terminal"]),
            all_in_runout=bool(root["executable_state"]["all_in_runout"]),
            history=tuple(
                ActionRecord(**row) for row in root["executable_state"]["history"]
            ),
            bet_fractions=tuple(root["executable_state"]["bet_fractions"]),
        )
        profile = weighted_nut_advantage_transition(
            root, state, transition["parent_board"]
        )
        transition["range_weighted_nut_advantage"] = profile
        transition["flags"]["nut_advantage_reversal"] = profile[
            "nut_advantage_reversal"
        ]
        transition["flags"]["straight_completion"] = bool(
            transition["flags"]["straight_completion"]
            or profile["range_straight_completion"]
        )
        transition["flags"]["flush_completion"] = bool(
            transition["flags"]["flush_completion"]
            or profile["range_flush_completion"]
        )
        transition["flags"]["blank"] = not any(
            transition["flags"][label]
            for label in (
                "pairing",
                "flush_completion",
                "new_flush_draw",
                "straight_completion",
                "new_gutshot",
                "new_oesd",
                "counterfeit",
                "nut_advantage_reversal",
            )
        )
        transition["labels"] = [
            name for name, enabled in transition["flags"].items() if enabled
        ] or ["between_rank_nonblank"]
    root.pop("root_semantic_sha256", None)
    root["root_semantic_sha256"] = sha256_json(root)
    return root


def _postflop_coverage_source(
    source: Mapping[str, Any], index: int
) -> dict[str, Any]:
    enriched = dict(source)
    enriched["stack_anchor_bb"] = POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB[
        index % len(POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB)
    ]
    enriched["stack_configuration"] = POSTFLOP_STACK_CONFIGURATIONS[
        (index // len(POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB))
        % len(POSTFLOP_STACK_CONFIGURATIONS)
    ]
    return enriched


def _transition_candidate_sources(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    final_board = list(normalize_cards(source["representative_board"]))
    candidates = []
    for transition_index, transition_card in enumerate(final_board):
        parent = [
            card for card_index, card in enumerate(final_board) if card_index != transition_index
        ]
        candidate = dict(source)
        candidate["parent_public_board"] = parent
        candidate["transition_card"] = transition_card
        candidate["representative_board"] = [*parent, transition_card]
        candidates.append(candidate)
    return candidates


def _transition_candidate_score(
    root: Mapping[str, Any],
    *,
    desired_label: str,
    observed: Mapping[str, int],
) -> tuple[float, float, float, str]:
    flags = root["transition_classification"]["flags"]
    desired = 1.0 if flags.get(desired_label) is True else 0.0
    rare_score = math.fsum(
        (3.0 if label in {"straight_completion", "nut_advantage_reversal"} else 1.0)
        / (1.0 + int(observed.get(label, 0)))
        for label in TRANSITION_COVERAGE_LABELS
        if flags.get(label) is True
    )
    enabled_count = sum(flags.get(label) is True for label in TRANSITION_COVERAGE_LABELS)
    # The final card identity is a stable tie-breaker. max(..., key=...) uses
    # the negated ordinal through lexical reversal only after scientific scores.
    return desired, rare_score, -float(enabled_count), str(root["transition_card"])


def build_postflop_registry(stage: str, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in {"river", "turn", "flop"}:
        raise ValueError("postflop registry builder supports River, Turn, and Flop")
    source_rows = list(source_manifest["roots"])
    roots = []
    transition_counts: Counter[str] = Counter()
    prefix = {"river": "RV", "turn": "TN", "flop": "FL"}[stage]
    for index, source in enumerate(source_rows, start=1):
        enriched = _postflop_coverage_source(source, index - 1)
        root_id = f"S{STAGES[stage].number}-{prefix}-{index:04d}"
        if stage == "flop":
            state = _postflop_state(enriched, stage)
            selected = _normalized_root(
                stage=stage,
                source=enriched,
                index=index - 1,
                state=state,
                root_id=root_id,
            )
        else:
            desired = TRANSITION_COVERAGE_LABELS[
                (index - 1) % len(TRANSITION_COVERAGE_LABELS)
            ]
            candidates = []
            for candidate_source in _transition_candidate_sources(enriched):
                state = _postflop_state(candidate_source, stage)
                candidates.append(
                    _normalized_root(
                        stage=stage,
                        source=candidate_source,
                        index=index - 1,
                        state=state,
                        root_id=root_id,
                        # The desired transition may itself be defined by the
                        # range-weighted nut profile.  Compute that profile for
                        # every candidate before scoring rather than attaching
                        # it only after a winner has already been selected.
                        include_nut_profile=True,
                    )
                )
            selected = max(
                candidates,
                key=lambda root: _transition_candidate_score(
                    root, desired_label=desired, observed=transition_counts
                ),
            )
            selected["transition_selection_receipt"] = with_self_hash(
                {
                    "schema_version": "stage21_transition_selection_receipt_v1",
                    "desired_transition_label": desired,
                    "selection_scored_all_candidate_range_weighted_profiles": True,
                    "candidate_count": len(candidates),
                    "selected_transition_card": selected["transition_card"],
                    "candidates": [
                        {
                            "transition_card": candidate["transition_card"],
                            "parent_public_board": candidate["parent_public_board"],
                            "enabled_transition_labels": candidate[
                                "transition_classification"
                            ]["labels"],
                            "range_weighted_nut_advantage": candidate[
                                "transition_classification"
                            ]["range_weighted_nut_advantage"],
                            "selection_score": list(
                                _transition_candidate_score(
                                    candidate,
                                    desired_label=desired,
                                    observed=transition_counts,
                                )
                            ),
                        }
                        for candidate in candidates
                    ],
                }
            )
            selected.pop("root_semantic_sha256", None)
            selected["root_semantic_sha256"] = sha256_json(selected)
            transition_counts.update(
                label
                for label, enabled in selected["transition_classification"][
                    "flags"
                ].items()
                if enabled
            )
        roots.append(selected)
    return build_registry_document(stage, roots, source_manifest.get("self_sha256"))


def _complete_preflop_target_groups(
    corrections: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    classes = set(all_preflop_classes())
    groups: list[list[str]] = [[] for _ in corrections]
    assigned: set[str] = set()
    for index, correction in enumerate(corrections):
        target = str(correction["source_preflop_hand_class_target"])
        if target in classes and target not in assigned:
            groups[index].append(target)
            assigned.add(target)
    for hand_class in sorted(classes - assigned):
        target_index = min(range(len(groups)), key=lambda value: (len(groups[value]), value))
        groups[target_index].append(hand_class)
    flattened = [hand_class for group in groups for hand_class in group]
    if set(flattened) != classes or len(flattened) != 169:
        raise AssertionError("preflop target groups do not cover all 169 classes exactly once")
    if not all(1 <= len(group) <= 2 for group in groups):
        raise AssertionError("preflop target groups must contain one or two classes")
    return groups


def build_preflop_registry(
    source_manifest: Mapping[str, Any], correction_proposal: Mapping[str, Any]
) -> dict[str, Any]:
    source_map = {row["root_id"]: row for row in source_manifest.get("roots", ())}
    corrections = list(correction_proposal["proposed_corrections"])
    target_groups = _complete_preflop_target_groups(corrections)
    roots = []
    for index, correction in enumerate(corrections, start=1):
        source = source_map.get(
            correction["source_root_id"],
            {
                "root_id": correction["source_root_id"],
                "root_semantic_sha256": correction["source_root_semantic_sha256"],
                "decision_kind": correction["source_decision_kind"],
                "partition": correction["source_partition"],
                "range_family": correction["source_range_family"],
                "preflop_hand_class_target": correction[
                    "source_preflop_hand_class_target"
                ],
            },
        )
        merged_source = dict(source)
        merged_source.update(
            {
                "source_root_id": correction["source_root_id"],
                "source_root_semantic_sha256": correction["source_root_semantic_sha256"],
                "source_decision_kind": correction["source_decision_kind"],
                "source_partition": correction["source_partition"],
                "source_range_family": correction["source_range_family"],
                "source_preflop_hand_class_target": correction[
                    "source_preflop_hand_class_target"
                ],
                "preflop_hand_class_targets": target_groups[index - 1],
                "preflop_hand_class_target": target_groups[index - 1][0],
            }
        )
        state = _preflop_state(correction)
        roots.append(
            _normalized_root(
                stage="preflop",
                source=merged_source,
                correction=correction,
                index=index - 1,
                state=state,
                root_id=f"S4-PF-{index:04d}",
            )
        )
    provenance = sha256_json(
        {
            "source_manifest": source_manifest.get("self_sha256"),
            "correction_proposal": correction_proposal.get("self_sha256"),
        }
    )
    return build_registry_document("preflop", roots, provenance)


def build_registry_document(
    stage: str, roots: Sequence[Mapping[str, Any]], provenance_sha256: str | None
) -> dict[str, Any]:
    definition = STAGES[stage]
    counts = {
        kind: sum(root["root_kind"] == kind for root in roots)
        for kind in ("approximate", "restricted_exact", "forced")
    }
    expected = {
        "approximate": definition.approximate_count,
        "restricted_exact": definition.restricted_exact_count,
        "forced": definition.forced_count,
    }
    if len(roots) != definition.root_count or counts != expected:
        raise ValueError(
            f"{stage} repaired partition mismatch: roots={len(roots)} counts={counts} expected={expected}"
        )
    ids = [str(root["root_id"]) for root in roots]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate repaired root IDs")
    document = {
        "schema_version": "stage21_repaired_root_registry_v3",
        "revision": REVISION,
        "stage": stage,
        "root_count": len(roots),
        "counts": counts,
        "source_provenance_sha256": provenance_sha256,
        "science_seeds": list(SCIENCE_SEEDS),
        "algorithms": list(ALGORITHMS),
        "roots_per_producer": 8,
        "lanes_per_root": 6,
        "lanes_per_producer": 48,
        "historical_scientifically_revalidated_roots": 0,
        "historical_scientifically_revalidated_policies": 0,
        "roots": list(roots),
    }
    document["semantic_sha256"] = sha256_json(document)
    return document


def validate_registry(registry: Mapping[str, Any]) -> None:
    stage = str(registry["stage"])
    definition = STAGES[stage]
    if registry.get("root_count") != definition.root_count:
        raise ValueError("root count mismatch")
    if registry.get("lanes_per_producer") != 48 or registry.get("science_seeds") != list(
        SCIENCE_SEEDS
    ):
        raise ValueError("lane/seed contract mismatch")
    body = dict(registry)
    semantic = body.pop("semantic_sha256", None)
    if semantic != sha256_json(body):
        raise ValueError("registry semantic hash mismatch")
    root_ids = [str(root["root_id"]) for root in registry["roots"]]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("duplicate root ID")
    observed_counts = {
        kind: sum(root["root_kind"] == kind for root in registry["roots"])
        for kind in ("approximate", "restricted_exact", "forced")
    }
    expected_counts = {
        "approximate": definition.approximate_count,
        "restricted_exact": definition.restricted_exact_count,
        "forced": definition.forced_count,
    }
    if registry.get("counts") != expected_counts or observed_counts != expected_counts:
        raise ValueError("root partition mismatch")
    for root in registry["roots"]:
        root_body = dict(root)
        root_hash = root_body.pop("root_semantic_sha256", None)
        if root_hash != sha256_json(root_body):
            raise ValueError(f"root semantic hash mismatch: {root['root_id']}")
        state = HoldemState(
            street=stage,
            board=tuple(root["representative_board"]),
            button_player=root["executable_state"]["button_player"],
            actor=root["executable_state"]["actor"],
            starting_stacks=tuple(root["executable_state"]["starting_stacks"]),
            stacks=tuple(root["executable_state"]["stacks"]),
            contributions=tuple(root["executable_state"]["contributions"]),
            street_contributions=tuple(root["executable_state"]["street_contributions"]),
            current_bet=root["executable_state"]["current_bet"],
            last_full_raise=root["executable_state"]["last_full_raise"],
            acted_since_full_raise=tuple(root["executable_state"]["acted_since_full_raise"]),
            checks_in_row=root["executable_state"]["checks_in_row"],
            folded=tuple(root["executable_state"]["folded"]),
            terminal=root["executable_state"]["terminal"],
            all_in_runout=root["executable_state"]["all_in_runout"],
            history=tuple(ActionRecord(**row) for row in root["executable_state"]["history"]),
            bet_fractions=tuple(root["executable_state"]["bet_fractions"]),
        )
        state.validate()
        if _legal_family(state) != root["derived_legal_action_family"]:
            raise ValueError(f"legal family replay mismatch: {root['root_id']}")
        expected_stack_band = _derived_stack_band(state)
        expected_spr_band = _derived_spr_band(state)
        if root.get("stack_band") != expected_stack_band:
            raise ValueError(f"executable stack-band mismatch: {root['root_id']}")
        if root.get("spr_band") != expected_spr_band:
            raise ValueError(f"executable SPR-band mismatch: {root['root_id']}")
        if not math.isclose(
            float(root.get("derived_spr", math.nan)),
            _derived_spr(state),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"derived SPR replay mismatch: {root['root_id']}")
        historical = root.get("historical_labels", {})
        if root.get("historical_stack_band_label_repaired") is not (
            historical.get("stack_band") != expected_stack_band
        ):
            raise ValueError(f"stack-band repair receipt mismatch: {root['root_id']}")
        if root.get("historical_spr_band_label_repaired") is not (
            historical.get("spr_band") != expected_spr_band
        ):
            raise ValueError(f"SPR-band repair receipt mismatch: {root['root_id']}")
        if stage in {"river", "turn", "flop"}:
            if state.button_player != 0 or root["position"] not in {
                "hero_btn_sb_ip",
                "hero_bb_oop",
            }:
                raise ValueError(f"postflop position mismatch: {root['root_id']}")
            expected_hero = 0 if root["position"] == "hero_btn_sb_ip" else 1
            if root["hero_player"] != expected_hero:
                raise ValueError(f"postflop hero position mismatch: {root['root_id']}")
            range_contract = root.get("range_contract", {})
            if (
                range_contract.get(f"player_{expected_hero}_family")
                != root.get("range_family")
                or range_contract.get(f"player_{1 - expected_hero}_family")
                != root.get("opponent_range_family")
                or range_contract.get("hero_player") != expected_hero
                or range_contract.get("seat_binding_rule")
                != "hero_resolved_before_physical_player_binding"
            ):
                raise ValueError(
                    f"hero/opponent range seat binding mismatch: {root['root_id']}"
                )
            effective_bb = min(state.starting_stacks) / 100.0
            if root.get("stack_anchor_bb") != effective_bb or effective_bb not in {
                float(value) for value in POSTFLOP_EFFECTIVE_STACK_ANCHORS_BB
            }:
                raise ValueError(f"postflop stack-anchor mismatch: {root['root_id']}")
            if state.starting_stacks[0] == state.starting_stacks[1]:
                expected_configuration = "equal_effective"
            elif state.starting_stacks[expected_hero] > state.starting_stacks[1 - expected_hero]:
                expected_configuration = "hero_covers_1_5x"
            else:
                expected_configuration = "opponent_covers_1_5x"
            if root.get("stack_configuration") != expected_configuration:
                raise ValueError(f"postflop covering-stack mismatch: {root['root_id']}")
            if not math.isclose(
                float(root.get("covering_stack_bb", math.nan)),
                max(state.starting_stacks) / 100.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"postflop covering-stack amount mismatch: {root['root_id']}")
            if stage in {"turn", "river"}:
                parent = root.get("parent_public_board")
                card = root.get("transition_card")
                if not isinstance(parent, list) or not isinstance(card, str):
                    raise ValueError(f"missing transition lineage: {root['root_id']}")
                if [*parent, card] != root["representative_board"]:
                    raise ValueError(f"transition lineage order mismatch: {root['root_id']}")
                expected_transition = classify_public_transition(parent, card)
                observed_transition = root.get("transition_classification", {})
                for key in (
                    "parent_board",
                    "transition_card",
                    "child_board",
                    "parent_street",
                    "child_street",
                    "parent_board_classification",
                    "child_board_classification",
                    "parent_straight_completion_ranks",
                    "child_straight_completion_ranks",
                    "blank_definition",
                ):
                    if observed_transition.get(key) != expected_transition.get(key):
                        raise ValueError(
                            f"transition classification mismatch {key}: {root['root_id']}"
                        )
                for label in TRANSITION_COVERAGE_LABELS:
                    if label in {"blank", "nut_advantage_reversal"}:
                        continue
                    expected_flag = expected_transition["flags"].get(label)
                    nut_profile = observed_transition.get(
                        "range_weighted_nut_advantage", {}
                    )
                    if label == "straight_completion":
                        expected_flag = bool(
                            expected_flag
                            or nut_profile.get("range_straight_completion") is True
                        )
                    if label == "flush_completion":
                        expected_flag = bool(
                            expected_flag
                            or nut_profile.get("range_flush_completion") is True
                        )
                    if observed_transition.get("flags", {}).get(label) is not expected_flag:
                        raise ValueError(
                            f"transition flag mismatch {label}: {root['root_id']}"
                        )
                expected_blank = not any(
                    observed_transition.get("flags", {}).get(label) is True
                    for label in (
                        "pairing",
                        "flush_completion",
                        "new_flush_draw",
                        "straight_completion",
                        "new_gutshot",
                        "new_oesd",
                        "counterfeit",
                        "nut_advantage_reversal",
                    )
                )
                if observed_transition.get("flags", {}).get("blank") is not expected_blank:
                    raise ValueError(f"transition blank mismatch: {root['root_id']}")
                if observed_transition.get("flags", {}).get(
                    "nut_advantage_reversal"
                ) is not nut_profile.get("nut_advantage_reversal"):
                    raise ValueError(
                        f"nut-advantage transition mismatch: {root['root_id']}"
                    )
            elif root.get("parent_public_board") is not None or root.get(
                "transition_card"
            ) is not None:
                raise ValueError(f"Flop root has false transition lineage: {root['root_id']}")
        if root["root_kind"] == "restricted_exact" and state.bet_fractions:
            raise ValueError(f"restricted exact root has broad bet sizes: {root['root_id']}")
        if root["root_kind"] == "forced" and not (
            state.all_in_runout or (stage == "river" and state.terminal)
        ):
            raise ValueError(f"forced root is not an all-in runout: {root['root_id']}")
    if stage == "preflop":
        target_rows = [
            str(hand_class)
            for root in registry["roots"]
            for hand_class in root.get("preflop_hand_class_targets", ())
        ]
        if len(target_rows) != 169 or set(target_rows) != set(all_preflop_classes()):
            raise ValueError("preflop roots do not explicitly cover all 169 classes")
        if any(count != 1 for count in Counter(target_rows).values()):
            raise ValueError("preflop target class assignment is not exactly once")
