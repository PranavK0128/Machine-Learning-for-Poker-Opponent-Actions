from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, Mapping

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    validate_document_self_hash,
)
from stage6_decision_row_builder import build_feature_role_manifest
from stage6_notebook01_common import recursive_stage6_code_identities
from stage6_notebook01_materializer import (
    LANE_COUNT,
    _lane_ids,
    _pin_inner_threads,
    materialize_lane_semantics,
    stable_lane_id,
)
from stage6_worker_policy import worker_resolution_report


REGISTRY_RELATIVE = Path("contracts/stage6_profile_metric_registry_hold_v1.json")
FEATURE_MANIFEST_RELATIVE = Path(
    "contracts/stage6_decision_feature_role_manifest_v1.json"
)
PARITY_CODE_RELATIVES = (
    Path("src/stage6_notebook01_parallel_parity.py"),
)


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage6ContractError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    with path.open("r", encoding="utf-8", errors="strict") as source:
        value = json.load(source, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise Stage6ContractError(f"JSON authority is not an object: {path}")
    return value


def _players_by_lane() -> dict[str, str]:
    players: dict[str, str] = {}
    candidate = 0
    while len(players) < LANE_COUNT:
        player = f"bounded-parity-player-{candidate:04d}"
        players.setdefault(stable_lane_id(player), player)
        candidate += 1
    return dict(sorted(players.items()))


def _decision_record(ordinal: int, player: str) -> dict[str, object]:
    alias = f"alias-{ordinal:02d}"
    return {
        "dataset_version": "action_state_v1_stage004_action_history",
        "hand_action_id": f"action-{ordinal:02d}",
        "hand_id": f"hand-{ordinal:02d}",
        "game_id": "bounded-parity-game",
        "street": "preflop",
        "board_texture_present": "false",
        "board_card_count": "",
        "board_high_card_value": "",
        "board_pair_level": "",
        "board_is_paired": "false",
        "board_is_two_tone": "false",
        "board_is_monotone": "false",
        "board_connectedness_score": "",
        "big_blind_raw": "100",
        "legal_mask_quality": "action_state_export_exact_v1",
        "can_fold": "false",
        "can_check": "true",
        "can_call": "false",
        "can_bet": "true",
        "can_raise": "false",
        "can_all_in": "true",
        "legal_min_bet_to_raw": "100",
        "legal_min_raise_to_raw": "",
        "legal_max_bet_raise_to_raw": "10000",
        "active_opponent_canonical_player_ids_before": '["opponent-player"]',
        "active_opponent_count_before": "1",
        "sequence_context_quality": "action_history_export_v1",
        "street_action_count_before": "0",
        "street_aggressive_action_count_before": "0",
        "street_call_count_before": "0",
        "street_check_count_before": "0",
        "street_fold_count_before": "0",
        "street_prior_actor_action_count": "0",
        "street_prior_actor_bet_raise_count": "0",
        "street_last_aggressor_is_actor": "false",
        "facing_bet_or_raise": "false",
        "check_raise_opportunity": "false",
        "facing_cbet": "false",
        "facing_donk_lead": "false",
        "missed_cbet_probe_opportunity": "false",
        "delayed_cbet_opportunity": "false",
        "turn_barrel_opportunity": "false",
        "river_barrel_opportunity": "false",
        "preflop_limper_count_before": "0",
        "preflop_caller_count_before": "0",
        "preflop_raise_count_before": "0",
        "preflop_prior_entry_count_before": "0",
        "preflop_facing_open_raise": "false",
        "preflop_facing_three_bet_or_more": "false",
        "preflop_squeeze_opportunity": "false",
        "preflop_limp_opportunity": "true",
        "preflop_actor_has_limped_before": "false",
        "preflop_actor_has_called_before": "false",
        "preflop_actor_has_raised_before": "false",
        "pokernow_player_id": alias,
        "seat": "1",
        "normalized_type": "check",
        "action_label": "check",
        "pot_before_raw": "150",
        "stack_before_raw": "10000",
        "amount_to_call_raw": "0",
        "street_contribution_before_raw": "0",
        "street_contribution_after_raw": "0",
        "is_fold": "false",
        "is_check": "true",
        "is_call": "false",
        "is_bet_or_raise": "false",
        "is_all_in": "false",
        "effective_stack_before_bb": "100",
        "spr_before": "66.666667",
        "active_player_count_before": "2",
        "players_with_chips_before": "2",
        "decision_utc": "2026-07-22T20:55:38.352Z",
        "actor_pokernow_player_id": alias,
        "canonical_group_player_id": player,
        "identity_grouping_status": "resolved_retrospective_group",
        "source_row_ordinal": ordinal,
        "event_index": 2,
        "decision_epoch_ms": 1_784_753_738_352,
    }


def _payloads() -> list[tuple[str, list[dict[str, object]]]]:
    return [
        (lane_id, [_decision_record(ordinal, player)])
        for ordinal, (lane_id, player) in enumerate(_players_by_lane().items())
    ]


def _run_payload(
    payload: tuple[str, list[dict[str, object]]],
    registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    _pin_inner_threads()
    lane_id, decisions = payload
    semantic = materialize_lane_semantics(
        lane_id,
        decisions,
        [],
        registry,
        feature_manifest,
    )
    return lane_id, semantic


def _result_root(rows: list[tuple[str, dict[str, Any]]]) -> str:
    payload = [
        {"lane_id": lane_id, "semantic": semantic}
        for lane_id, semantic in sorted(rows, key=lambda row: row[0])
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run_bounded_parallel_parity(
    stage6_root: Path,
    *,
    requested_workers: Any = "auto",
) -> dict[str, Any]:
    stage6_root = stage6_root.resolve()
    if "OneDrive" in str(stage6_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    registry = _strict_json_object(stage6_root / REGISTRY_RELATIVE)
    feature_manifest = _strict_json_object(
        stage6_root / FEATURE_MANIFEST_RELATIVE
    )
    validate_document_self_hash(registry)
    validate_document_self_hash(feature_manifest)
    if feature_manifest != build_feature_role_manifest():
        raise Stage6ContractError("feature-role manifest differs from code-pinned roles")

    payloads = _payloads()
    if [lane_id for lane_id, _ in payloads] != _lane_ids():
        raise Stage6ContractError("bounded parity fixture does not cover exact lane order")
    roots: dict[str, str] = {}
    resolution = worker_resolution_report(
        requested=requested_workers,
        unfinished_lane_count=len(payloads),
        lane_cap=LANE_COUNT,
    )
    worker_counts = sorted({1, max(1, int(resolution["live_workers"]))})
    context = multiprocessing.get_context("spawn")
    for workers in worker_counts:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            max_tasks_per_child=1,
        ) as executor:
            futures = [
                executor.submit(
                    _run_payload,
                    payload,
                    registry,
                    feature_manifest,
                )
                for payload in payloads
            ]
            results = [future.result() for future in futures]
        roots[str(workers)] = _result_root(results)
    if len(set(roots.values())) != 1:
        raise Stage6ContractError(
            "one-worker and resolved-parallel-worker semantic roots differ"
        )

    code_identities = recursive_stage6_code_identities(
        stage6_root, PARITY_CODE_RELATIVES
    )
    receipt: dict[str, Any] = {
        "schema_version": "stage6_notebook01_bounded_parallel_parity_v1",
        "status": "PASS_BOUNDED_ONE_WORKER_VS_RESOLVED_WORKER_PARITY",
        "bounded_fixture_only": True,
        "lane_count": LANE_COUNT,
        "requested_workers": requested_workers,
        "worker_resolution": resolution,
        "worker_counts_compared": worker_counts,
        "one_independent_lane_per_worker_process_lifetime": True,
        "inner_numeric_threads_per_worker": 1,
        "canonical_lane_order": _lane_ids(),
        "semantic_roots_by_worker_count": roots,
        "worker_count_invariant_semantic_root": roots["1"],
        "code_identities": code_identities,
        "runtime_authority": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "drive_publication_authority": False,
        "live_action_authority": False,
    }
    receipt["self_sha256"] = document_self_sha256(receipt)
    return receipt
