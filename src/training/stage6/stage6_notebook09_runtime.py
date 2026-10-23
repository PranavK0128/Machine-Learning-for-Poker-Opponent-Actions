"""One-time successor-lockbox adjudication and Stage 6 closeout."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_metrics import multiclass_metric_report, selective_risk_curve
from stage6_notebook08_runtime import SUCCESSOR_FINAL_PANEL, _successor_holdout_manifest_from_frame
from stage6_notebooks02_07_runtime import (
    ACTIONS,
    BEHAVIOR_FEATURE_COLUMNS,
    LANE_COUNT,
    LEGAL_COLUMNS,
    NOTEBOOK01_RUN_ID,
    NOTEBOOK01_SPLIT_ROOT,
    _lane_paths,
    _parallel_lanes,
    _predict_context_model,
    _project_legal,
    _street_mask,
    canonical_json_bytes,
    materialization_root,
    resolve_workers,
    seal_document,
    sha256_file,
    validate_document,
)


EXPECTED_NOTEBOOK08_NAME = "08_Stage6_Notebook08_6b8febd599d81830_eac5f0dd0ec43464.zip"
EXPECTED_NOTEBOOK08_BYTES = 136075
EXPECTED_NOTEBOOK08_SHA256 = "eac5f0dd0ec434645640066d233fe18c028ba08df0a80305f5d84a955465930f"
EXPECTED_CANDIDATE_FREEZE_SELF_SHA256 = "e963dadac88d60c72db1124f243d8512aa35ec5b9c5be79ce8decf8ded1ecd4c"
EXPECTED_CANDIDATE_PAYLOAD_ROOT = "b1051674a1a850da26994dc148a9a349c6973838625275a813e9665d65e0ce8c"
EXPECTED_SUCCESSOR_MANIFEST_SELF_SHA256 = "dfedc908c5fa57c21e42896ef472bb772c3d13098d4f7c0a4b4bf3bf114a22ad"
EXPECTED_SUCCESSOR_MAPPING_ROOT = "7e03ca3a25c7b6786f12d9af9d6d57a7f1577ea135eced313e2513356c259e8d"
REQUIRED_PANELS = (
    "future_known_player",
    "unseen_player_cold_start",
    "unseen_game",
    SUCCESSOR_FINAL_PANEL,
)
REQUIRED_BASELINES = (
    "class_frequency",
    "current_qualified_artifact",
    "deterministic_chart_or_heuristic",
    "legal_random_or_uniform_where_semantically_valid",
)
STANDARD_PROBABILITY_BASELINES = tuple(
    name for name in REQUIRED_BASELINES if name != "current_qualified_artifact"
)
INCUMBENT_ARTIFACT_FILES = {
    "feature_builder.py": "a8eb34f09baa22dd83b03f3a3dcf4cea1fd4e99ebf8c909da0149b0c2ce7a1cf",
    "preflop_global/feature_schema.json": "6e3a81c6bc733289395959972aeaae7b07aa2f7f0f084b4a74255d33ceebe88d",
    "preflop_global/label_schema.json": "40d4151737dcf65ca0a92c5f315a616bcea50250e85b984cb61f1588f64a5bcb",
    "preflop_global/lightgbm_baseline.joblib": "16144a3fd2af6103f5d650d5d80a3e234c075a483c165dd5d73f63db3d21ccc8",
    "action_state_global/feature_schema.json": "3c702cc4f3be8f8c9812bba2e514c98f182c12226b332f19e7f2be61f7a5a5ba",
    "action_state_global/label_schema.json": "3f7cb7f6fe602a95d4baa27050a84fe9acf39e4846e3b195c589a3f4189d0ef3",
    "action_state_global/lightgbm_baseline.joblib": "1007443ad893b8b1ef770d474a26385646fb4a7e6e9f5fc0f6e7d755eb13ab0d",
    "action_state_global/lightgbm_calibration.json": "7d6530ea373ea309a8b218cc2b30ad779207685b53c5baadfc26f0ecea27f3ed",
}
NOTEBOOK01_INPUT_NAME = "POKERNOW_ASSISTANT_STAGE6_QUALIFIED_LAYERS_INPUTS_R34.zip"
NOTEBOOK01_INPUT_BYTES = 519480798
NOTEBOOK01_INPUT_SHA256 = "89ddc45894ab129d52715724256a71a623b230e6411790c0278ef010bb5cd34d"
NOTEBOOK01_ACTION_MEMBER = (
    "Poker Assistant Model/Stage 6 - Qualified Layers Improvement/"
    "data/exports/ml/action-state-decision-rows.csv"
)
NOTEBOOK01_ACTION_MEMBER_BYTES = 2393137787
NOTEBOOK01_ACTION_MEMBER_SHA256 = "2136ffb385c814bd12810a170c2243dff7ce0f9593800475fe81d3cf594d8d6d"
PREFLOP_NATIVE_FEATURE_NAME = "stage6-incumbent-preflop-native-features.parquet"
PREFLOP_NATIVE_FEATURE_ROWS = 1285167
PREFLOP_SOURCE_BYTES = 310292577
PREFLOP_SOURCE_SHA256 = "f0b8e15c055939c272c16552c019afd0f577617ca6bc895b9f397a433bc8f0fd"
PREFLOP_NATIVE_REQUIRED_COLUMNS = (
    "decisionType", "stakeLabel", "position", "positionGroup", "primaryOpportunity",
    "facingSituation", "openerPosition", "threeBettorPosition", "seat", "stackBb",
    "rfiOpportunity", "stealOpportunity", "coldCallOpportunity", "threeBetOpportunity",
    "squeezeOpportunity", "foldToStealOpportunity", "facedOpenRaise", "facedThreeBet",
    "facedFourBet", "foldToFourBetOpportunity", "openRaiseAmountBb",
    "threeBetToAmountBb", "priorLimperCount", "priorCallerCount", "playersLeftToAct",
)
ROLLING_STREETS = ("preflop", "flop", "turn", "river")
ROLLING_PREFIXES = ("actor_roll", "active_opponent_roll")
ACTION_NATIVE_SOURCE_COLUMNS = (
    "hand_action_id", "hand_id", "game_id", "event_index", "street",
    "board_texture_present", "board_run", "board_card_count", "board_high_card_rank",
    "board_high_card_value", "board_has_ace", "board_has_king", "board_broadway_count",
    "board_low_card_count", "board_pair_level", "board_is_paired", "board_is_two_pair",
    "board_is_trips", "board_is_quads", "board_is_rainbow", "board_is_two_tone",
    "board_is_monotone", "board_has_flush_possible", "board_has_flush_draw_possible",
    "board_unique_rank_count", "board_max_rank_gap", "board_connectedness_score",
    "board_has_straight_possible", "board_has_open_ended_straight_draw_possible",
    "board_has_gutshot_straight_draw_possible", "hand_started_at", "big_blind_raw",
    "can_fold", "can_check", "can_call", "can_bet", "can_raise", "can_all_in",
    "call_amount_raw", "pot_after_call_raw", "min_raise_to_raw_v2", "max_raise_to_raw_v2",
    "legal_min_bet_to_raw", "legal_min_raise_to_raw", "legal_max_bet_raise_to_raw",
    "legal_bet_raise_min_delta_raw", "legal_bet_raise_max_delta_raw",
    "sequence_context_quality", "street_action_count_before",
    "street_aggressive_action_count_before", "street_call_count_before",
    "street_check_count_before", "street_fold_count_before", "street_prior_actor_action_count",
    "street_prior_actor_check_count", "street_prior_actor_call_count",
    "street_prior_actor_bet_raise_count", "street_last_aggressor_is_actor",
    "street_last_aggression_event_index", "facing_bet_or_raise", "check_raise_opportunity",
    "facing_cbet", "facing_donk_lead", "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity", "turn_barrel_opportunity", "river_barrel_opportunity",
    "preflop_limper_count_before", "preflop_caller_count_before",
    "preflop_raise_count_before", "preflop_prior_entry_count_before",
    "preflop_facing_open_raise", "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity", "preflop_limp_opportunity",
    "preflop_actor_has_limped_before", "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before", "seat", "pot_before_raw", "pot_before_amount",
    "pot_before_bb", "stack_before_raw", "stack_before_amount", "stack_before_bb",
    "amount_to_call_raw", "amount_to_call_amount", "amount_to_call_bb",
    "current_bet_before_raw", "current_bet_before_amount", "current_bet_before_bb",
    "minimum_raise_to_raw", "minimum_raise_to_amount", "minimum_raise_to_bb",
    "effective_stack_before_bb", "spr_before", "active_player_count_before",
    "players_with_chips_before", "street_contribution_before_raw",
    "hand_contribution_before_raw",
)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 6092026
RESULT_MEMBER_ORDER = [
    "candidate_manifest.json",
    "successor_lockbox_report.json",
    "controller_receipt.json",
]
_INCUMBENT_CACHE: dict[tuple[str, str], tuple[Any, Any, list[str], list[str], float]] = {}


def _payload_root(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        {"component": name, "self_sha256": payload["self_sha256"]}
        for name, payload in sorted(payloads.items())
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _read_notebook08_result(path: Path) -> dict[str, Any]:
    if (
        path.name != EXPECTED_NOTEBOOK08_NAME
        or path.stat().st_size != EXPECTED_NOTEBOOK08_BYTES
        or sha256_file(path) != EXPECTED_NOTEBOOK08_SHA256
    ):
        raise RuntimeError("Notebook 08 returned-result identity mismatch")
    expected = [
        "candidate_freeze.json",
        "validation_report.json",
        "controller_receipt.json",
        "RESULT_MANIFEST.json",
    ]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)):
            raise RuntimeError("Notebook 08 returned-result membership is invalid")
        raw: dict[str, bytes] = {}
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise RuntimeError("Notebook 08 returned-result has an unsafe member")
            raw[name] = archive.read(name)
    documents = {name: json.loads(value) for name, value in raw.items()}
    for document in documents.values():
        validate_document(document)
    manifest = documents["RESULT_MANIFEST.json"]
    if (
        manifest.get("schema_version") != "stage6_notebook08_result_manifest_v1"
        or manifest.get("notebook_token") != "08"
        or manifest.get("member_order") != expected[:-1]
        or [row.get("path") for row in manifest.get("members", [])] != expected[:-1]
    ):
        raise RuntimeError("Notebook 08 result manifest is invalid")
    for identity in manifest["members"]:
        payload = raw[identity["path"]]
        if len(payload) != identity["bytes"] or hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            raise RuntimeError("Notebook 08 result member identity mismatch")
    freeze = documents["candidate_freeze.json"]
    report = documents["validation_report.json"]
    controller = documents["controller_receipt.json"]
    successor = freeze.get("successor_holdout_manifest")
    components = freeze.get("selected_components")
    if not isinstance(successor, Mapping) or not isinstance(components, Mapping):
        raise RuntimeError("Notebook 08 successor freeze is incomplete")
    validate_document(successor)
    for component in components.values():
        validate_document(component)
    requirements = successor.get("notebook09_enforcement_requirements", {})
    if (
        controller.get("status") != "PASS_VALIDATION_ONLY_CANDIDATE_FREEZE"
        or controller.get("notebook01_run_identity") != NOTEBOOK01_RUN_ID
        or controller.get("notebook01_split_assignment_root_digest") != NOTEBOOK01_SPLIT_ROOT
        or controller.get("candidate_freeze_self_sha256") != freeze.get("self_sha256")
        or controller.get("validation_report_self_sha256") != report.get("self_sha256")
        or controller.get("private_lockbox_opened") is not False
        or freeze.get("self_sha256") != EXPECTED_CANDIDATE_FREEZE_SELF_SHA256
        or freeze.get("candidate_payload_root_digest") != EXPECTED_CANDIDATE_PAYLOAD_ROOT
        or freeze.get("candidate_frozen") is not True
        or freeze.get("private_lockbox_opened") is not False
        or freeze.get("test_selection_event_count") != 0
        or _payload_root(components) != EXPECTED_CANDIDATE_PAYLOAD_ROOT
        or successor.get("self_sha256") != EXPECTED_SUCCESSOR_MANIFEST_SELF_SHA256
        or successor.get("mapping_root_digest") != EXPECTED_SUCCESSOR_MAPPING_ROOT
        or successor.get("effective_required_panels") != sorted(REQUIRED_PANELS)
        or successor.get("consumed_original_private_lockbox_excluded") is not True
        or requirements.get("recompute_mapping_from_split_metadata_before_target_read") is not True
        or requirements.get("required_mapping_root_digest") != EXPECTED_SUCCESSOR_MAPPING_ROOT
        or requirements.get("reject_original_private_panel_before_target_read") is not True
        or requirements.get("rejected_original_private_panel") != "final_private_lockbox"
        or requirements.get("claim_successor_event_before_target_read") is not True
        or requirements.get("selection_or_tuning_after_claim_allowed") is not False
        or report.get("per_table_size_metrics_status") != "PENDING_SUCCESSOR_NOTEBOOK09_EVALUATION"
    ):
        raise RuntimeError("Notebook 08 candidate-freeze authority binding is invalid")
    if set(components) != {
        "preflop_behavior", "postflop_behavior", "conditional_size",
        "profile", "review", "reference",
    }:
        raise RuntimeError("Notebook 08 component inventory is invalid")
    return {"freeze": freeze, "report": report, "controller": controller}


def _read_event_anchor(path: Path, notebook08: Mapping[str, Any]) -> dict[str, Any]:
    anchor = json.loads(path.read_text(encoding="utf-8"))
    validate_document(anchor)
    successor = notebook08["freeze"]["successor_holdout_manifest"]
    notebook01_source = anchor.get("notebook01_native_action_source")
    preflop_source = anchor.get("incumbent_preflop_native_feature_source")
    if (
        anchor.get("schema_version") != "stage6_notebook09_successor_lockbox_event_anchor_v1"
        or anchor.get("event_scope") != "one_time_successor_lockbox_adjudication"
        or anchor.get("notebook08_result_name") != EXPECTED_NOTEBOOK08_NAME
        or anchor.get("notebook08_result_bytes") != EXPECTED_NOTEBOOK08_BYTES
        or anchor.get("notebook08_result_sha256") != EXPECTED_NOTEBOOK08_SHA256
        or anchor.get("candidate_freeze_self_sha256") != notebook08["freeze"]["self_sha256"]
        or anchor.get("candidate_payload_root_digest") != notebook08["freeze"]["candidate_payload_root_digest"]
        or anchor.get("successor_holdout_manifest_self_sha256") != successor["self_sha256"]
        or anchor.get("successor_mapping_root_digest") != successor["mapping_root_digest"]
        or anchor.get("required_panels") != list(REQUIRED_PANELS)
        or anchor.get("required_baselines") != list(REQUIRED_BASELINES)
        or anchor.get("current_qualified_artifact_files") != INCUMBENT_ARTIFACT_FILES
        or notebook01_source != {
            "name": NOTEBOOK01_INPUT_NAME,
            "bytes": NOTEBOOK01_INPUT_BYTES,
            "sha256": NOTEBOOK01_INPUT_SHA256,
            "action_member": NOTEBOOK01_ACTION_MEMBER,
            "action_member_bytes": NOTEBOOK01_ACTION_MEMBER_BYTES,
            "action_member_sha256": NOTEBOOK01_ACTION_MEMBER_SHA256,
            "read_timing": "after_successor_event_claim_only",
        }
        or not isinstance(preflop_source, Mapping)
        or preflop_source.get("name") != PREFLOP_NATIVE_FEATURE_NAME
        or preflop_source.get("row_count") != PREFLOP_NATIVE_FEATURE_ROWS
        or preflop_source.get("source_preflop_csv_bytes") != PREFLOP_SOURCE_BYTES
        or preflop_source.get("source_preflop_csv_sha256") != PREFLOP_SOURCE_SHA256
        or preflop_source.get("target_or_label_columns_included") is not False
        or not isinstance(preflop_source.get("bytes"), int)
        or not isinstance(preflop_source.get("sha256"), str)
        or anchor.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or anchor.get("bootstrap_confidence") != BOOTSTRAP_CONFIDENCE
        or anchor.get("bootstrap_seed") != BOOTSTRAP_SEED
        or anchor.get("multiple_comparison_method") != "holm_bonferroni"
        or anchor.get("independent_audit_status") != "PASS_P0_0_P1_0"
        or anchor.get("single_use_successor_lockbox_event") is not True
        or anchor.get("selection_or_tuning_after_open") is not False
    ):
        raise RuntimeError("Notebook 09 successor event anchor is invalid")
    return anchor


def _materialization_path(stage_drive_root: Path) -> Path:
    return stage_drive_root / "runtime_state" / "notebook01" / "materializations" / NOTEBOOK01_RUN_ID


def _recompute_successor_manifest_before_target_read(
    root: Path,
    frozen_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    columns = [
        "row_id", "game_id", "hand_id", "decision_utc", "panel", "role",
        "eligible_for_selection",
    ]
    frames = []
    for lane in range(LANE_COUNT):
        _, split_path, _ = _lane_paths(root, lane)
        frames.append(pd.read_parquet(split_path, columns=columns))
    recomputed = _successor_holdout_manifest_from_frame(
        pd.concat(frames, ignore_index=True),
        frozen_manifest["consumed_predecessor_notebook09"],
    )
    if recomputed != dict(frozen_manifest):
        raise RuntimeError("successor mapping changed before target read")
    return recomputed


def _load_successor_lane(
    root_text: str,
    lane: int,
    decision_columns: Sequence[str],
    first_successor_utc: str,
) -> pd.DataFrame:
    root = Path(root_text)
    decision_path, split_path, _ = _lane_paths(root, lane)
    split_columns = [
        "row_id", "game_id", "hand_id", "decision_utc", "panel", "role",
        "eligible_for_selection",
    ]
    splits = pd.read_parquet(split_path, columns=split_columns)
    if splits["row_id"].astype(str).duplicated().any():
        raise RuntimeError(f"lane-{lane:02d} has duplicate split identities")
    original_private = (
        splits["role"].astype(str).eq("test")
        & splits["panel"].astype(str).eq("final_private_lockbox")
    )
    selected = splits.loc[splits["role"].astype(str).eq("test") & ~original_private].copy()
    if not selected.empty and not selected["eligible_for_selection"].eq(False).all():
        raise RuntimeError("successor evaluation rows are selection eligible")
    if selected["panel"].astype(str).eq("final_private_lockbox").any():
        raise RuntimeError("consumed original private rows entered successor selection")
    selected["panel"] = np.where(
        selected["decision_utc"].astype(str).ge(first_successor_utc),
        SUCCESSOR_FINAL_PANEL,
        selected["panel"].astype(str),
    )
    columns = list(dict.fromkeys(("row_id", *decision_columns)))
    if selected.empty:
        return pd.DataFrame(columns=list(dict.fromkeys((*columns, *split_columns))))
    selected_ids = selected["row_id"].astype(str).tolist()
    decisions = pq.read_table(
        decision_path,
        columns=columns,
        filters=[("row_id", "in", selected_ids)],
    ).to_pandas()
    if decisions["row_id"].astype(str).duplicated().any():
        raise RuntimeError(f"lane-{lane:02d} has duplicate successor decision identities")
    joined = decisions.merge(selected, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(selected) or len(joined) != len(decisions):
        raise RuntimeError(f"lane-{lane:02d} successor decision/split join is not exact")
    return joined


def _hand_size_successor_lane(
    root_text: str,
    lane: int,
    first_successor_utc: str,
) -> dict[str, Any]:
    frame = _load_successor_lane(
        root_text, lane, ("active_player_count_before",), first_successor_utc
    )
    active = pd.to_numeric(frame["active_player_count_before"], errors="raise").to_numpy(dtype=int)
    if (active < 2).any():
        raise RuntimeError("successor table-size derivation received an invalid active-player count")
    return {
        "clusters": (frame["game_id"].astype(str) + "\x1f" + frame["hand_id"].astype(str)).tolist(),
        "active_players": active.tolist(),
    }


def _successor_hand_size_map(lanes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    clusters = sum((list(lane["clusters"]) for lane in lanes), [])
    active_players = sum((list(lane["active_players"]) for lane in lanes), [])
    if not clusters or len(clusters) != len(active_players):
        raise RuntimeError("successor per-hand table-size source is empty or misaligned")
    frame = pd.DataFrame({"cluster": clusters, "active_players": active_players})
    sizes = frame.groupby("cluster", sort=True)["active_players"].max().astype(int).to_dict()
    if any(value < 2 for value in sizes.values()):
        raise RuntimeError("successor per-hand table-size map is invalid")
    return sizes


def _validate_incumbent_artifacts(package_root: Path) -> Path:
    root = package_root / "current_qualified_artifact"
    for relative, expected_sha256 in INCUMBENT_ARTIFACT_FILES.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"current qualified artifact identity mismatch: {relative}")
    return root


def _validate_preflop_native_feature_source(
    package_root: Path,
    anchor: Mapping[str, Any],
) -> Path:
    identity = anchor["incumbent_preflop_native_feature_source"]
    path = package_root / "current_qualified_artifact" / PREFLOP_NATIVE_FEATURE_NAME
    if (
        not path.is_file()
        or path.stat().st_size != int(identity["bytes"])
        or sha256_file(path) != identity["sha256"]
    ):
        raise RuntimeError("incumbent preflop native feature source identity mismatch")
    header = pq.read_schema(path).names
    forbidden = {
        name for name in header
        if name.lower().startswith("label")
        or name in {"action_label", "action_to_bb", "is_all_in"}
    }
    if forbidden:
        raise RuntimeError(f"preflop native feature source contains targets: {sorted(forbidden)}")
    return path


def _validate_notebook01_action_source_after_claim(
    stage_drive_root: Path,
) -> Path:
    path = stage_drive_root / "inputs" / NOTEBOOK01_INPUT_NAME
    if (
        not path.is_file()
        or path.stat().st_size != NOTEBOOK01_INPUT_BYTES
        or sha256_file(path) != NOTEBOOK01_INPUT_SHA256
    ):
        raise RuntimeError("Notebook 01 R34 native action source package identity mismatch")
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(NOTEBOOK01_ACTION_MEMBER)
        except KeyError as error:
            raise RuntimeError("Notebook 01 R34 native action member is missing") from error
        if info.file_size != NOTEBOOK01_ACTION_MEMBER_BYTES:
            raise RuntimeError("Notebook 01 R34 native action member size mismatch")
    return path


def _empty_action_stats() -> dict[str, float]:
    stats = {
        "decisions": 0.0, "fold": 0.0, "call": 0.0, "check": 0.0,
        "bet_raise": 0.0, "facing_bet": 0.0, "facing_fold": 0.0,
        "facing_call": 0.0, "facing_raise": 0.0,
        "open_bet_opportunities": 0.0, "open_bets": 0.0,
        "preflop_decisions": 0.0, "preflop_vpip": 0.0, "preflop_pfr": 0.0,
        "postflop_decisions": 0.0, "postflop_aggressive": 0.0,
        "bet_size_pct_pot_sum": 0.0, "bet_size_pct_pot_count": 0.0,
        "all_in": 0.0, "spr_sum": 0.0, "spr_count": 0.0,
        "amount_to_call_pct_pot_sum": 0.0, "amount_to_call_pct_pot_count": 0.0,
        "stack_bb_sum": 0.0, "stack_bb_count": 0.0,
    }
    for street in ROLLING_STREETS:
        for suffix in ("decisions", "fold", "call", "check", "bet_raise"):
            stats[f"{street}_{suffix}"] = 0.0
    return stats


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _native_ratio_or_nan(numerator: Any, denominator: Any) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0:
        return float("nan")
    return float(numerator) / denominator_value


def _action_stat_feature_values(prefix: str, stats: Mapping[str, float]) -> dict[str, float]:
    decisions = float(stats.get("decisions", 0.0))
    facing = float(stats.get("facing_bet", 0.0))
    preflop = float(stats.get("preflop_decisions", 0.0))
    postflop = float(stats.get("postflop_decisions", 0.0))
    values = {
        f"{prefix}_decisions": decisions,
        f"{prefix}_fold_pct": _safe_ratio(float(stats.get("fold", 0.0)), decisions),
        f"{prefix}_call_pct": _safe_ratio(float(stats.get("call", 0.0)), decisions),
        f"{prefix}_check_pct": _safe_ratio(float(stats.get("check", 0.0)), decisions),
        f"{prefix}_bet_raise_pct": _safe_ratio(float(stats.get("bet_raise", 0.0)), decisions),
        f"{prefix}_aggression_pct": _safe_ratio(float(stats.get("bet_raise", 0.0)), decisions),
        f"{prefix}_facing_bet_decisions": facing,
        f"{prefix}_facing_bet_fold_pct": _safe_ratio(float(stats.get("facing_fold", 0.0)), facing),
        f"{prefix}_facing_bet_call_pct": _safe_ratio(float(stats.get("facing_call", 0.0)), facing),
        f"{prefix}_facing_bet_raise_pct": _safe_ratio(float(stats.get("facing_raise", 0.0)), facing),
        f"{prefix}_open_bet_pct": _safe_ratio(
            float(stats.get("open_bets", 0.0)),
            float(stats.get("open_bet_opportunities", 0.0)),
        ),
        f"{prefix}_preflop_decisions": preflop,
        f"{prefix}_preflop_vpip_pct": _safe_ratio(float(stats.get("preflop_vpip", 0.0)), preflop),
        f"{prefix}_preflop_pfr_pct": _safe_ratio(float(stats.get("preflop_pfr", 0.0)), preflop),
        f"{prefix}_postflop_decisions": postflop,
        f"{prefix}_postflop_aggression_pct": _safe_ratio(
            float(stats.get("postflop_aggressive", 0.0)), postflop
        ),
        f"{prefix}_avg_bet_size_pct_pot": _safe_ratio(
            float(stats.get("bet_size_pct_pot_sum", 0.0)),
            float(stats.get("bet_size_pct_pot_count", 0.0)),
        ),
        f"{prefix}_all_in_pct": _safe_ratio(float(stats.get("all_in", 0.0)), decisions),
        f"{prefix}_avg_spr": _safe_ratio(
            float(stats.get("spr_sum", 0.0)), float(stats.get("spr_count", 0.0))
        ),
        f"{prefix}_avg_amount_to_call_pct_pot": _safe_ratio(
            float(stats.get("amount_to_call_pct_pot_sum", 0.0)),
            float(stats.get("amount_to_call_pct_pot_count", 0.0)),
        ),
        f"{prefix}_avg_stack_bb": _safe_ratio(
            float(stats.get("stack_bb_sum", 0.0)), float(stats.get("stack_bb_count", 0.0))
        ),
    }
    for street in ROLLING_STREETS:
        street_decisions = float(stats.get(f"{street}_decisions", 0.0))
        values[f"{prefix}_{street}_decisions"] = street_decisions
        for action in ("fold", "call", "check", "bet_raise"):
            values[f"{prefix}_{street}_{action}_pct"] = _safe_ratio(
                float(stats.get(f"{street}_{action}", 0.0)), street_decisions
            )
    return values


def _coarse_action_label(value: Any) -> str:
    label = str(value).lower()
    return "bet_or_raise" if label in {"bet", "raise", "all_in", "bet_or_raise"} else label


def _update_action_stats(stats: dict[str, float], row: Mapping[str, Any]) -> None:
    action = _coarse_action_label(row.get("action_label"))
    street = str(row.get("street", "")).lower()
    amount_to_call = float(row.get("amount_to_call_bb") or 0.0)
    pot_before = float(row.get("pot_before_bb") or 0.0)
    action_delta = max(
        0.0,
        float(row.get("actor_committed_after_bb") or 0.0)
        - float(row.get("actor_committed_before_bb") or 0.0),
    )
    spr = float(row.get("spr_before") or 0.0)
    stack = float(row.get("actor_stack_before_bb") or 0.0)
    is_bet_raise = action == "bet_or_raise"
    stats["decisions"] += 1.0
    if action in {"fold", "call", "check"}:
        stats[action] += 1.0
    elif is_bet_raise:
        stats["bet_raise"] += 1.0
    if str(row.get("action_label", "")).lower() == "all_in":
        stats["all_in"] += 1.0
    if street in ROLLING_STREETS:
        stats[f"{street}_decisions"] += 1.0
        if action in {"fold", "call", "check"}:
            stats[f"{street}_{action}"] += 1.0
        elif is_bet_raise:
            stats[f"{street}_bet_raise"] += 1.0
    if amount_to_call > 0:
        stats["facing_bet"] += 1.0
        if action == "fold":
            stats["facing_fold"] += 1.0
        elif action == "call":
            stats["facing_call"] += 1.0
        elif is_bet_raise:
            stats["facing_raise"] += 1.0
    else:
        stats["open_bet_opportunities"] += 1.0
        if is_bet_raise:
            stats["open_bets"] += 1.0
    if street == "preflop":
        stats["preflop_decisions"] += 1.0
        if action in {"call", "bet_or_raise"}:
            stats["preflop_vpip"] += 1.0
        if is_bet_raise:
            stats["preflop_pfr"] += 1.0
    elif street in {"flop", "turn", "river"}:
        stats["postflop_decisions"] += 1.0
        if is_bet_raise:
            stats["postflop_aggressive"] += 1.0
    if is_bet_raise and pot_before > 0 and action_delta > 0:
        stats["bet_size_pct_pot_sum"] += (action_delta / pot_before) * 100.0
        stats["bet_size_pct_pot_count"] += 1.0
    if pot_before > 0 and amount_to_call > 0:
        stats["amount_to_call_pct_pot_sum"] += (amount_to_call / pot_before) * 100.0
        stats["amount_to_call_pct_pot_count"] += 1.0
    if spr > 0:
        stats["spr_sum"] += spr
        stats["spr_count"] += 1.0
    if stack > 0:
        stats["stack_bb_sum"] += stack
        stats["stack_bb_count"] += 1.0


def _parse_active_opponents(value: Any) -> list[str]:
    parsed = json.loads(str(value)) if not isinstance(value, list) else value
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise RuntimeError("active-opponent rolling identity list is invalid")
    return [item for item in parsed if item]


def _sum_action_stats(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    result = _empty_action_stats()
    for row in rows:
        for key in result:
            result[key] += float(row.get(key, 0.0))
    return result


def _subtract_action_stats(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: max(0.0, float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
        for key in _empty_action_stats()
    }


def _all_accepted_rolling_rows(root: Path) -> pd.DataFrame:
    columns = [
        "decision_id", "decision_utc", "decision_order", "game_id", "hand_id",
        "actor_id", "seat", "street", "active_opponent_canonical_player_ids_before",
        "action_label", "amount_to_call_bb", "pot_before_bb",
        "actor_committed_before_bb", "actor_committed_after_bb", "actor_stack_before_bb",
        "spr_before",
    ]
    frames = []
    for lane in range(LANE_COUNT):
        decision_path, _, _ = _lane_paths(root, lane)
        frame = pq.read_table(decision_path, columns=columns).to_pandas()
        frame["_original_index"] = np.arange(len(frame), dtype=np.int64) * LANE_COUNT + lane
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["decision_id"].astype(str).duplicated().any():
        raise RuntimeError("accepted rolling source has duplicate decision identities")
    return result


def _successor_identity_frame(
    root: Path,
    first_successor_utc: str,
) -> pd.DataFrame:
    columns = ("decision_id", "decision_order", "actor_id", "seat", "street")
    frames = [
        _load_successor_lane(str(root), lane, columns, first_successor_utc)
        for lane in range(LANE_COUNT)
    ]
    result = pd.concat(frames, ignore_index=True)
    if result.empty or result["decision_id"].astype(str).duplicated().any():
        raise RuntimeError("successor incumbent identity source is empty or duplicated")
    return result


def _read_action_native_source(
    notebook01_input: Path,
    successor_postflop_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_parts: list[pd.DataFrame] = []
    hand_start_parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(notebook01_input) as archive, archive.open(NOTEBOOK01_ACTION_MEMBER) as source:
        for index, chunk in enumerate(pd.read_csv(
            source,
            usecols=list(ACTION_NATIVE_SOURCE_COLUMNS),
            chunksize=100_000,
            low_memory=False,
        )):
            hand_start_parts.append(chunk[["hand_id", "hand_started_at"]].drop_duplicates())
            selected = chunk["hand_action_id"].astype(str).isin(successor_postflop_ids)
            if selected.any():
                native_parts.append(chunk.loc[selected].copy())
            if (index + 1) % 5 == 0:
                print(json.dumps({
                    "event": "stage6_incumbent_native_action_scan_progress",
                    "source_rows": (index + 1) * 100_000,
                    "matched_successor_rows": sum(len(part) for part in native_parts),
                }, sort_keys=True), flush=True)
    native = pd.concat(native_parts, ignore_index=True) if native_parts else pd.DataFrame()
    if (
        len(native) != len(successor_postflop_ids)
        or native["hand_action_id"].astype(str).duplicated().any()
        or set(native["hand_action_id"].astype(str)) != successor_postflop_ids
    ):
        raise RuntimeError("native action source does not exactly cover successor postflop decisions")
    starts = pd.concat(hand_start_parts, ignore_index=True).drop_duplicates()
    conflicts = starts.groupby("hand_id", sort=False)["hand_started_at"].nunique(dropna=False)
    if (conflicts != 1).any():
        raise RuntimeError("native action source has conflicting hand start times")
    return native, starts.drop_duplicates("hand_id")


def _rolling_successor_features(
    accepted: pd.DataFrame,
    hand_starts: pd.DataFrame,
    successor_postflop_ids: set[str],
) -> pd.DataFrame:
    ordered = accepted.merge(hand_starts, on="hand_id", how="left", validate="many_to_one")
    if ordered["hand_started_at"].isna().any():
        raise RuntimeError("rolling source is missing an exact native hand start time")
    ordered["_hand_started_at"] = pd.to_numeric(ordered["hand_started_at"], errors="raise")
    ordered = ordered.sort_values(
        ["_hand_started_at", "game_id", "hand_id", "decision_order", "_original_index"],
        kind="mergesort",
    )
    player_stats: dict[str, dict[str, float]] = {}
    game_stats: dict[str, dict[str, float]] = {}
    game_player_stats: dict[tuple[str, str], dict[str, float]] = {}
    output: list[dict[str, Any]] = []
    for ordinal, row_tuple in enumerate(ordered.itertuples(index=False), start=1):
        row = row_tuple._asdict()
        actor_id = str(row["actor_id"])
        game_id = str(row["game_id"])
        actor_stats = player_stats.setdefault(actor_id, _empty_action_stats())
        table_stats = game_stats.setdefault(game_id, _empty_action_stats())
        actor_table = game_player_stats.setdefault((game_id, actor_id), _empty_action_stats())
        opponent_pool = _subtract_action_stats(table_stats, actor_table)
        active_ids = [
            value for value in _parse_active_opponents(
                row["active_opponent_canonical_player_ids_before"]
            ) if value != actor_id
        ]
        active_stats = (
            _sum_action_stats([
                player_stats.get(value, _empty_action_stats()) for value in active_ids
            ])
            if active_ids else opponent_pool
        )
        decision_id = str(row["decision_id"])
        if decision_id in successor_postflop_ids:
            features: dict[str, Any] = {"decision_id": decision_id}
            features.update(_action_stat_feature_values("actor_roll", actor_stats))
            features.update(_action_stat_feature_values("active_opponent_roll", active_stats))
            output.append(features)
        _update_action_stats(actor_stats, row)
        _update_action_stats(table_stats, row)
        _update_action_stats(actor_table, row)
        if ordinal % 250_000 == 0:
            print(json.dumps({
                "event": "stage6_incumbent_rolling_progress",
                "processed_rows": ordinal,
                "total_rows": len(ordered),
                "captured_successor_rows": len(output),
            }, sort_keys=True), flush=True)
    result = pd.DataFrame(output)
    if (
        len(result) != len(successor_postflop_ids)
        or result["decision_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError("causal rolling reconstruction did not exactly cover successor postflop rows")
    return result


def _read_preflop_native_source(
    path: Path,
    successor_first: pd.DataFrame,
) -> pd.DataFrame:
    keys = set(zip(successor_first["hand_id"].astype(str), successor_first["seat"].astype(int)))
    parts: list[pd.DataFrame] = []
    row_count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        row_count += len(chunk)
        mask = [
            (str(hand_id), int(seat)) in keys
            for hand_id, seat in zip(chunk["hand_id"], chunk["seat"])
        ]
        if any(mask):
            parts.append(chunk.loc[mask].copy())
    if row_count != PREFLOP_NATIVE_FEATURE_ROWS:
        raise RuntimeError("preflop native feature source row count changed")
    selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if selected.empty or selected.duplicated(["hand_id", "seat"]).any():
        raise RuntimeError("preflop native feature source match is empty or duplicated")
    merged = successor_first.merge(selected, on=["hand_id", "seat"], how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("current qualified preflop artifact has no in-domain successor rows")
    if not merged["decisionType"].astype(str).eq("first_preflop_action").all():
        raise RuntimeError("preflop native feature source decision type changed")
    return merged


def _prepare_incumbent_native_context(
    stage_drive_root: Path,
    package_root: Path,
    root: Path,
    first_successor_utc: str,
    context_root: Path,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    notebook01_input = _validate_notebook01_action_source_after_claim(stage_drive_root)
    preflop_source = _validate_preflop_native_feature_source(package_root, anchor)
    successor = _successor_identity_frame(root, first_successor_utc)
    accepted = _all_accepted_rolling_rows(root)
    successor_ids = set(successor["decision_id"].astype(str))
    postflop_ids = set(
        successor.loc[~successor["street"].astype(str).eq("preflop"), "decision_id"].astype(str)
    )
    native_action, hand_starts = _read_action_native_source(notebook01_input, postflop_ids)
    rolling = _rolling_successor_features(accepted, hand_starts, postflop_ids)
    native_action = native_action.rename(columns={"hand_action_id": "decision_id"})
    action_context = native_action.merge(rolling, on="decision_id", how="inner", validate="one_to_one")
    preflop_all = accepted.loc[accepted["street"].astype(str).eq("preflop")].sort_values(
        ["hand_id", "seat", "decision_order", "_original_index"], kind="mergesort"
    )
    first_preflop = preflop_all.drop_duplicates(["hand_id", "seat"], keep="first")
    successor_first = first_preflop.loc[
        first_preflop["decision_id"].astype(str).isin(successor_ids),
        ["decision_id", "hand_id", "seat"],
    ].copy()
    preflop_context = _read_preflop_native_source(preflop_source, successor_first)
    context_root.mkdir(parents=True, exist_ok=True)
    preflop_path = context_root / "preflop-native.parquet"
    action_path = context_root / "action-native-rolling.parquet"
    preflop_context.to_parquet(preflop_path, index=False)
    action_context.to_parquet(action_path, index=False)
    return {
        "preflop_path": str(preflop_path),
        "postflop_path": str(action_path),
        "preflop_native_domain_rows": len(preflop_context),
        "preflop_successor_rows": int(successor["street"].astype(str).eq("preflop").sum()),
        "postflop_native_domain_rows": len(action_context),
        "postflop_successor_rows": len(postflop_ids),
        "rolling_source_rows": len(accepted),
        "native_action_source_package_sha256": NOTEBOOK01_INPUT_SHA256,
        "preflop_native_feature_source_sha256": sha256_file(preflop_source),
    }


def _load_incumbent_head(package_root_text: str, street_mode: str) -> tuple[Any, Any, list[str], list[str], float]:
    key = (package_root_text, street_mode)
    cached = _INCUMBENT_CACHE.get(key)
    if cached is not None:
        return cached
    package_root = Path(package_root_text)
    root = _validate_incumbent_artifacts(package_root)
    folder = "preflop_global" if street_mode == "preflop" else "action_state_global"
    head = "preflop.global" if street_mode == "preflop" else "action_state"
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("current qualified LightGBM baseline requires joblib") from error
    spec = importlib.util.spec_from_file_location(
        "stage6_current_qualified_feature_builder", root / "feature_builder.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("current qualified feature builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    schema_path = root / folder / "feature_schema.json"
    builder = module.StackHUDFeatureBuilder(schema_path)
    model = joblib.load(root / folder / "lightgbm_baseline.joblib")
    label_schema = json.loads((root / folder / "label_schema.json").read_text(encoding="utf-8"))
    if not isinstance(label_schema, Mapping) or not all(
        isinstance(label, str) and isinstance(index, int) for label, index in label_schema.items()
    ):
        raise RuntimeError(f"current qualified {head} label schema is invalid")
    labels_by_index = [label for label, _ in sorted(label_schema.items(), key=lambda item: item[1])]
    classes = list(getattr(model, "classes_", range(len(labels_by_index))))
    labels = [
        labels_by_index[int(value)] if isinstance(value, (int, np.integer)) else str(value)
        for value in classes
    ]
    temperature = 1.0
    calibration_path = root / folder / "lightgbm_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        temperature = float(calibration["temperature"])
    result = (model, builder, labels, list(builder.feature_order), temperature)
    _INCUMBENT_CACHE[key] = result
    return result


def _project_visible(probabilities: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, int, int]:
    if probabilities.shape != legal.shape or probabilities.ndim != 2:
        raise RuntimeError("visible projection inputs are misaligned")
    illegal = int(np.count_nonzero((probabilities > 1e-15) & ~legal))
    projected = np.where(legal, probabilities, 0.0)
    totals = projected.sum(axis=1)
    fallback = totals <= 0.0
    if fallback.any():
        legal_counts = legal[fallback].sum(axis=1)
        if (legal_counts <= 0).any():
            raise RuntimeError("current qualified comparison has an empty legal action family")
        projected[fallback] = legal[fallback].astype(float) / legal_counts[:, None]
        totals = projected.sum(axis=1)
    return projected / totals[:, None], illegal, int(fallback.sum())


def _incumbent_comparison(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    legal: np.ndarray,
    truth: Sequence[str],
    street_mode: str,
    package_root_text: str,
    hand_sizes: Mapping[str, int],
    context_root_text: str,
) -> dict[str, Any]:
    model, builder, source_labels, feature_order, temperature = _load_incumbent_head(
        package_root_text, street_mode
    )
    context_path = Path(context_root_text) / (
        "preflop-native.parquet" if street_mode == "preflop" else "action-native-rolling.parquet"
    )
    decision_ids = frame["decision_id"].astype(str).tolist()
    native = pq.read_table(
        context_path,
        filters=[("decision_id", "in", decision_ids)],
    ).to_pandas()
    native["decision_id"] = native["decision_id"].astype(str)
    if native["decision_id"].duplicated().any():
        raise RuntimeError("incumbent native context has duplicate decision identities")
    staged = frame.reset_index(drop=True).copy()
    staged["decision_id"] = staged["decision_id"].astype(str)
    staged["_row_position"] = np.arange(len(staged), dtype=int)
    native = native.rename(columns={
        name: f"native__{name}" for name in native.columns if name != "decision_id"
    })
    merged = staged.merge(native, on="decision_id", how="inner", validate="one_to_one")
    if street_mode != "preflop" and len(merged) != len(staged):
        raise RuntimeError("current qualified action artifact lacks exact native successor coverage")
    if merged.empty:
        raise RuntimeError("current qualified artifact comparison has no native in-domain rows")
    positions = merged["_row_position"].to_numpy(dtype=int)
    records = []
    for row in merged.to_dict(orient="records"):
        cluster = f"{row['game_id']}\x1f{row['hand_id']}"
        if cluster not in hand_sizes:
            raise RuntimeError("successor row is missing its all-decision per-hand table size")
        state = {
            name.removeprefix("native__"): value
            for name, value in row.items() if name.startswith("native__")
        }
        state.update({name: value for name, value in row.items() if not name.startswith("native__")})
        if street_mode == "preflop":
            missing_native = sorted(set(PREFLOP_NATIVE_REQUIRED_COLUMNS) - set(state))
            if missing_native:
                raise RuntimeError(
                    f"current qualified artifact native input is missing/defaulted: {missing_native}"
                )
            state["stack_before_bb"] = state["stackBb"]
        else:
            state["street_action_pressure"] = _native_ratio_or_nan(
                state["street_aggressive_action_count_before"],
                state["street_action_count_before"],
            )
            state["street_actor_aggression_share"] = _native_ratio_or_nan(
                state["street_prior_actor_bet_raise_count"],
                state["street_prior_actor_action_count"],
            )
            rolling_required = set().union(*(
                set(_action_stat_feature_values(prefix, _empty_action_stats()))
                for prefix in ROLLING_PREFIXES
            ))
            native_required = (
                set(ACTION_NATIVE_SOURCE_COLUMNS)
                | {"street_action_pressure", "street_actor_aggression_share"}
            ) - {"hand_action_id"}
            missing_native = sorted((native_required | rolling_required) - set(state))
            if missing_native:
                raise RuntimeError(
                    f"current qualified artifact native input is missing/defaulted: {missing_native}"
                )
            for prefix in ROLLING_PREFIXES:
                state[prefix.replace("_roll", "_rolling")] = {
                    name.removeprefix(f"{prefix}_"): value
                    for name, value in state.items() if name.startswith(f"{prefix}_")
                }
        state.update({
            "stack_before_bb": (
                state["stackBb"] if street_mode == "preflop"
                else row.get("actor_stack_before_bb")
            ),
            "effective_stack_bb": row.get("effective_stack_before_bb"),
            "min_raise_to_bb": (
                row.get("legal_min_raise_to_bb")
                if float(row.get("amount_to_call_bb") or 0.0) > 0.0
                else row.get("legal_min_bet_to_bb")
            ),
            "street_contribution_before_bb": row.get("actor_committed_before_bb"),
            "players_dealt_count": int(hand_sizes[cluster]),
        })
        built = (
            builder.build_preflop_features(state)
            if street_mode == "preflop"
            else builder.build_action_state_features(state)
        )
        if built.missing_features or built.defaulted_features:
            raise RuntimeError(
                "current qualified artifact native input is missing/defaulted: "
                f"missing={built.missing_features}, defaulted={built.defaulted_features}"
            )
        records.append(built.features)
    model_frame = pd.DataFrame(records, columns=feature_order)
    for column in builder.categorical_features:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str).astype("category")
    for column in builder.numeric_features:
        model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce")
    incumbent_raw = np.asarray(model.predict_proba(model_frame), dtype=float)
    if incumbent_raw.shape != (len(merged), len(source_labels)):
        raise RuntimeError("current qualified artifact probability width changed")
    incumbent_raw = _temperature_scale(incumbent_raw, temperature)
    if street_mode == "preflop":
        labels = ("call", "fold", "no_action", "raise")
        truth_map = {
            "call": "call", "fold": "fold", "check": "no_action",
            "bet": "raise", "raise": "raise", "all_in": "raise",
        }
        exact_groups = {
            "call": ("call",), "fold": ("fold",), "no_action": ("check",),
            "raise": ("bet", "raise", "all_in"),
        }
    else:
        labels = ("bet_or_raise", "call", "check", "fold")
        truth_map = {
            "call": "call", "fold": "fold", "check": "check",
            "bet": "bet_or_raise", "raise": "bet_or_raise", "all_in": "bet_or_raise",
        }
        exact_groups = {
            "bet_or_raise": ("bet", "raise", "all_in"),
            "call": ("call",), "check": ("check",), "fold": ("fold",),
        }
    if set(source_labels) != set(labels):
        raise RuntimeError("current qualified artifact label inventory changed")
    source_index = {label: index for index, label in enumerate(source_labels)}
    incumbent = np.column_stack([incumbent_raw[:, source_index[label]] for label in labels])
    candidate = candidate[positions]
    legal = legal[positions]
    truth = [truth[index] for index in positions]
    candidate_family = np.column_stack([
        candidate[:, [ACTIONS.index(action) for action in exact_groups[label]]].sum(axis=1)
        for label in labels
    ])
    legal_family = np.column_stack([
        legal[:, [ACTIONS.index(action) for action in exact_groups[label]]].any(axis=1)
        for label in labels
    ])
    incumbent, incumbent_illegal, incumbent_fallback = _project_visible(incumbent, legal_family)
    candidate_family, _, _ = _project_visible(candidate_family, legal_family)
    return {
        "positions": positions.tolist(),
        "labels": list(labels),
        "truth": [truth_map[value] for value in truth],
        "candidate": candidate_family,
        "incumbent": incumbent,
        "incumbent_illegal_preprojection": incumbent_illegal,
        "incumbent_zero_mass_fallback": incumbent_fallback,
        "feature_builder_sha256": INCUMBENT_ARTIFACT_FILES["feature_builder.py"],
        "model_sha256": INCUMBENT_ARTIFACT_FILES[f"{'preflop_global' if street_mode == 'preflop' else 'action_state_global'}/lightgbm_baseline.joblib"],
        "comparison_label_space": "incumbent_native_action_families_with_exact_visible_legal_projection",
        "native_domain": (
            "exact_first_preflop_action_rows_matched_to_deployed_native_feature_source"
            if street_mode == "preflop"
            else "all_successor_postflop_rows_with_exact_native_static_and_causal_rolling_inputs"
        ),
        "native_domain_rows": len(merged),
        "successor_scope_rows": len(staged),
        "missing_or_defaulted_required_input_count": 0,
    }


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature must be positive and finite")
    powered = np.power(probabilities, 1.0 / temperature)
    totals = powered.sum(axis=1)
    if (totals <= 0).any():
        raise RuntimeError("calibration received an empty legal probability row")
    return powered / totals[:, None]


def _legal_uniform(legal: np.ndarray) -> np.ndarray:
    counts = legal.sum(axis=1)
    if (counts <= 0).any():
        raise RuntimeError("visible legal action set is empty")
    return legal.astype(float) / counts[:, None]


def _deterministic_public_heuristic(frame: pd.DataFrame, legal: np.ndarray) -> np.ndarray:
    result = np.zeros((len(frame), len(ACTIONS)), dtype=float)
    to_call = pd.to_numeric(frame["amount_to_call_bb"], errors="raise").to_numpy(dtype=float)
    for row in range(len(frame)):
        preferences = ("call", "fold", "check") if to_call[row] > 0 else ("check", "call", "fold")
        choice = next((ACTIONS.index(action) for action in preferences if legal[row, ACTIONS.index(action)]), None)
        if choice is None:
            legal_choices = np.flatnonzero(legal[row])
            if len(legal_choices) == 0:
                raise RuntimeError("deterministic heuristic received no legal action")
            choice = int(legal_choices[0])
        result[row, choice] = 1.0
    return result


def _behavior_successor_lane(
    root_text: str,
    lane: int,
    street_mode: str,
    component: Mapping[str, Any],
    first_successor_utc: str,
    package_root_text: str,
    hand_sizes: Mapping[str, int],
    context_root_text: str,
) -> dict[str, Any]:
    model = component["source_candidate"]
    columns = (
        "decision_id", "decision_order", "actor_id", "seat", "action_label",
        "active_player_count_before", *BEHAVIOR_FEATURE_COLUMNS,
    )
    frame = _load_successor_lane(root_text, lane, columns, first_successor_utc)
    frame = frame.loc[_street_mask(frame, street_mode)].copy()
    truth = frame["action_label"].astype(str).tolist()
    if not truth:
        empty = np.empty((0, len(ACTIONS)))
        return {
            "truth": [], "candidate": empty,
            "baselines": {name: empty for name in STANDARD_PROBABILITY_BASELINES},
            "incumbent": None,
            "accepted": [], "panels": [], "streets": [], "clusters": [], "table_sizes": [],
            "support_buckets": [], "candidate_illegal_preprojection": 0,
            "candidate_zero_mass_fallback": 0,
        }
    if component.get("selected_family") != "trained_candidate":
        raise RuntimeError("successor adjudication requires the frozen trained behavior candidate")
    if model.get("schema_version") != "stage6_backoff_context_frequency_behavior_head_v1":
        raise RuntimeError("successor adjudication received an unsupported behavior model")
    legal = frame[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
    priors = np.asarray(model["class_priors"], dtype=float)
    raw, support_buckets = _predict_context_model(frame, model)
    candidate, illegal_pre, fallback_count = _project_legal(raw, legal, priors)
    candidate = _temperature_scale(candidate, float(component["selected_temperature"]))
    class_frequency, _, _ = _project_legal(
        np.broadcast_to(priors, candidate.shape).copy(), legal, priors
    )
    uniform = _legal_uniform(legal)
    heuristic = _deterministic_public_heuristic(frame, legal)
    baselines = {
        "class_frequency": class_frequency,
        "deterministic_chart_or_heuristic": heuristic,
        "legal_random_or_uniform_where_semantically_valid": uniform,
    }
    clusters = (frame["game_id"].astype(str) + "\x1f" + frame["hand_id"].astype(str)).tolist()
    table_sizes = []
    for cluster in clusters:
        if cluster not in hand_sizes:
            raise RuntimeError("successor behavior row is missing its all-decision per-hand table size")
        table_sizes.append(int(hand_sizes[cluster]))
    incumbent = _incumbent_comparison(
        frame, candidate, legal, truth, street_mode, package_root_text, hand_sizes,
        context_root_text,
    )
    confidence = candidate.max(axis=1)
    accepted = confidence >= float(component["confidence_threshold"])
    return {
        "truth": truth,
        "candidate": candidate,
        "baselines": baselines,
        "incumbent": incumbent,
        "accepted": accepted.tolist(),
        "panels": frame["panel"].astype(str).tolist(),
        "streets": frame["street"].astype(str).tolist(),
        "clusters": clusters,
        "table_sizes": table_sizes,
        "support_buckets": support_buckets,
        "candidate_illegal_preprojection": illegal_pre,
        "candidate_zero_mass_fallback": fallback_count,
    }


def _metric_set(truth: Sequence[str], matrices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: multiclass_metric_report(truth, matrix, ACTIONS) for name, matrix in matrices.items()}


def _behavior_report(
    lanes: Sequence[Mapping[str, Any]],
    component: Mapping[str, Any],
    *,
    include_inputs: bool = False,
) -> Any:
    truth = sum((list(lane["truth"]) for lane in lanes), [])
    if not truth:
        raise RuntimeError("successor behavior adjudication set is empty")
    candidate = np.concatenate([lane["candidate"] for lane in lanes], axis=0)
    baselines = {
        name: np.concatenate([lane["baselines"][name] for lane in lanes], axis=0)
        for name in STANDARD_PROBABILITY_BASELINES
    }
    incumbent_lanes = [lane["incumbent"] for lane in lanes if lane["incumbent"] is not None]
    if not incumbent_lanes:
        raise RuntimeError("current qualified artifact comparison is empty")
    incumbent_labels = list(incumbent_lanes[0]["labels"])
    if any(list(row["labels"]) != incumbent_labels for row in incumbent_lanes):
        raise RuntimeError("current qualified artifact label space changed between lanes")
    incumbent_truth = sum((list(row["truth"]) for row in incumbent_lanes), [])
    incumbent_candidate = np.concatenate([row["candidate"] for row in incumbent_lanes], axis=0)
    incumbent_probabilities = np.concatenate([row["incumbent"] for row in incumbent_lanes], axis=0)
    incumbent_positions: list[int] = []
    offset = 0
    for lane in lanes:
        row = lane["incumbent"]
        if row is not None:
            incumbent_positions.extend(offset + int(value) for value in row["positions"])
        offset += len(lane["truth"])
    incumbent_position_array = np.asarray(incumbent_positions, dtype=int)
    panels = np.asarray(sum((list(lane["panels"]) for lane in lanes), []), dtype=object)
    streets = np.asarray(sum((list(lane["streets"]) for lane in lanes), []), dtype=object)
    clusters = np.asarray(sum((list(lane["clusters"]) for lane in lanes), []), dtype=object)
    table_sizes = np.asarray(sum((list(lane["table_sizes"]) for lane in lanes), []), dtype=int)
    support = np.asarray(sum((list(lane["support_buckets"]) for lane in lanes), []), dtype=object)
    accepted = np.asarray(sum((list(lane["accepted"]) for lane in lanes), []), dtype=bool)
    if (
        not accepted.any()
        or not incumbent_truth
        or len(incumbent_truth) != len(incumbent_position_array)
        or len(set(incumbent_positions)) != len(incumbent_positions)
        or (incumbent_position_array >= len(truth)).any()
    ):
        raise RuntimeError("frozen behavior threshold abstained on every successor row")
    all_matrices = {"candidate": candidate, **baselines}
    truth_array = np.asarray(truth, dtype=object)
    incumbent_truth_array = np.asarray(incumbent_truth, dtype=object)

    def slice_metrics(mask: np.ndarray) -> dict[str, Any]:
        result = _metric_set(
            truth_array[mask].tolist(),
            {name: matrix[mask] for name, matrix in all_matrices.items()},
        )
        incumbent_mask = mask[incumbent_position_array]
        current = {
            "label_order": incumbent_labels,
            "native_domain_row_count": int(incumbent_mask.sum()),
            "scope_row_count": int(mask.sum()),
            "status": "IN_NATIVE_DOMAIN" if incumbent_mask.any() else "OUTSIDE_NATIVE_DOMAIN",
        }
        if incumbent_mask.any():
            current["candidate"] = multiclass_metric_report(
                incumbent_truth_array[incumbent_mask].tolist(),
                incumbent_candidate[incumbent_mask], incumbent_labels,
            )
            current["current_qualified_artifact"] = multiclass_metric_report(
                incumbent_truth_array[incumbent_mask].tolist(),
                incumbent_probabilities[incumbent_mask], incumbent_labels,
            )
        result["current_qualified_artifact_comparison"] = current
        return result

    per_panel = {}
    for panel in REQUIRED_PANELS:
        mask = panels == panel
        if not mask.any():
            raise RuntimeError(f"required successor panel is empty: {panel}")
        per_panel[panel] = slice_metrics(mask)
    per_street = {
        str(street): slice_metrics(streets == street)
        for street in sorted(set(streets))
    }
    per_table_size = {
        str(size): slice_metrics(table_sizes == size)
        for size in sorted(set(table_sizes))
    }
    per_support_bucket = {
        str(bucket): slice_metrics(support == bucket)
        for bucket in sorted(set(support))
    }
    predicted = np.asarray([ACTIONS[index] for index in candidate.argmax(axis=1)], dtype=object)
    report = seal_document({
        "schema_version": "stage6_behavior_successor_lockbox_report_v1",
        "street_scope": component["street_scope"],
        "frozen_candidate_self_sha256": component["self_sha256"],
        "selected_family": component["selected_family"],
        "selected_temperature": component["selected_temperature"],
        "confidence_threshold": component["confidence_threshold"],
        "row_count": len(truth),
        "overall_metrics": slice_metrics(np.ones(len(truth), dtype=bool)),
        "covered_candidate_metrics": multiclass_metric_report(
            truth_array[accepted].tolist(), candidate[accepted], ACTIONS
        ),
        "accepted_rows": int(accepted.sum()),
        "actual_coverage": float(accepted.mean()),
        "abstention_rate": float(1.0 - accepted.mean()),
        "selective_risk": float((predicted[accepted] != truth_array[accepted]).mean()),
        "selective_risk_by_coverage": selective_risk_curve(
            truth, candidate, ACTIONS, coverages=(0.25, 0.5, 0.75, 0.9, 1.0)
        ),
        "per_panel_metrics": per_panel,
        "per_street_metrics": per_street,
        "per_table_size_metrics": per_table_size,
        "per_support_bucket_metrics": per_support_bucket,
        "table_size_derivation": "max_active_player_count_before_across_all_successor_decisions_within_exact_game_id_plus_hand_id_before_street_filtering",
        "current_qualified_artifact_baseline": {
            "source": "exact_current_qualified_preflop_global_or_action_state_lightgbm_artifact",
            "feature_builder_sha256": incumbent_lanes[0]["feature_builder_sha256"],
            "model_sha256": incumbent_lanes[0]["model_sha256"],
            "comparison_label_space": incumbent_lanes[0]["comparison_label_space"],
            "proper_scoring_surrogate": None,
            "illegal_preprojection_count": int(sum(row["incumbent_illegal_preprojection"] for row in incumbent_lanes)),
            "zero_mass_fallback_count": int(sum(row["incumbent_zero_mass_fallback"] for row in incumbent_lanes)),
            "native_domain": incumbent_lanes[0]["native_domain"],
            "native_domain_row_count": len(incumbent_truth),
            "successor_scope_row_count": len(truth),
            "missing_or_defaulted_required_input_count": 0,
        },
        "illegal_preprojection_count": int(sum(int(lane["candidate_illegal_preprojection"]) for lane in lanes)),
        "illegal_postprojection_count": 0,
        "zero_mass_fallback_count": int(sum(int(lane["candidate_zero_mass_fallback"]) for lane in lanes)),
        "selection_or_tuning_event_count": 0,
        "successor_lockbox_opened": True,
        "promotion_authority": False,
    })
    if not include_inputs:
        return report
    truth_indices = np.asarray([ACTIONS.index(value) for value in truth], dtype=int)
    candidate_loss = -np.log(np.clip(
        candidate[np.arange(len(truth_indices)), truth_indices], 1e-15, 1.0
    ))
    differences = {}
    difference_clusters = {}
    for baseline_name, matrix in baselines.items():
        baseline_loss = -np.log(np.clip(
            matrix[np.arange(len(truth_indices)), truth_indices], 1e-15, 1.0
        ))
        differences[baseline_name] = candidate_loss - baseline_loss
        difference_clusters[baseline_name] = clusters
    incumbent_truth_indices = np.asarray(
        [incumbent_labels.index(value) for value in incumbent_truth], dtype=int
    )
    differences["current_qualified_artifact"] = -np.log(np.clip(
        incumbent_candidate[np.arange(len(incumbent_truth_indices)), incumbent_truth_indices],
        1e-15, 1.0,
    )) + np.log(np.clip(
        incumbent_probabilities[np.arange(len(incumbent_truth_indices)), incumbent_truth_indices],
        1e-15, 1.0,
    ))
    difference_clusters["current_qualified_artifact"] = clusters[incumbent_position_array]
    return report, {
        "truth": truth,
        "candidate": candidate,
        "baselines": baselines,
        "clusters": clusters,
        "differences": differences,
        "difference_clusters": difference_clusters,
    }


def _size_successor_lane(
    root_text: str,
    lane: int,
    component: Mapping[str, Any],
    first_successor_utc: str,
    hand_sizes: Mapping[str, int],
) -> dict[str, Any]:
    columns = (
        "street", "action_label", "action_to_bb", "active_player_count_before",
        "legal_min_bet_to_bb", "legal_min_raise_to_bb", "legal_max_bet_raise_to_bb",
    )
    frame = _load_successor_lane(root_text, lane, columns, first_successor_utc)
    model = component["source_candidate"]
    targets: list[float] = []
    candidate: list[float] = []
    baseline: list[float] = []
    panels: list[str] = []
    streets: list[str] = []
    clusters: list[str] = []
    table_sizes: list[int] = []
    abstained = 0
    for row in frame.itertuples(index=False):
        action = str(row.action_label)
        if action not in {"bet", "raise", "all_in"}:
            continue
        maximum = float(row.legal_max_bet_raise_to_bb)
        target = float(row.action_to_bb)
        if action == "all_in":
            minimum = maximum
            predicted = maximum
            midpoint = maximum
        else:
            minimum = float(row.legal_min_bet_to_bb if action == "bet" else row.legal_min_raise_to_bb)
            if not all(math.isfinite(value) for value in (minimum, maximum, target)) or maximum <= minimum:
                abstained += 1
                continue
            head = model["heads"].get(f"{row.street}:{action}")
            if not isinstance(head, Mapping):
                abstained += 1
                continue
            fraction = float(head["fraction_between_visible_min_and_max"])
            predicted = minimum + min(max(fraction, 0.0), 1.0) * (maximum - minimum)
            midpoint = minimum + 0.5 * (maximum - minimum)
        if not all(math.isfinite(value) for value in (minimum, maximum, target, predicted, midpoint)):
            abstained += 1
            continue
        if predicted < minimum - 1e-9 or predicted > maximum + 1e-9:
            raise RuntimeError("successor conditional size projection escaped visible bounds")
        cluster = f"{row.game_id}\x1f{row.hand_id}"
        if cluster not in hand_sizes:
            raise RuntimeError("successor size row is missing its all-decision per-hand table size")
        targets.append(target)
        candidate.append(predicted)
        baseline.append(midpoint)
        panels.append(str(row.panel))
        streets.append(str(row.street))
        clusters.append(cluster)
        table_sizes.append(int(hand_sizes[cluster]))
    return {
        "targets": targets, "candidate": candidate, "baseline": baseline,
        "panels": panels, "streets": streets, "clusters": clusters,
        "table_sizes": table_sizes, "abstained": abstained,
    }


def _regression_metrics(target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    return {
        "row_count": int(len(target)),
        "candidate_mae_bb": float(np.mean(np.abs(candidate - target))),
        "candidate_rmse_bb": float(np.sqrt(np.mean((candidate - target) ** 2))),
        "midpoint_baseline_mae_bb": float(np.mean(np.abs(baseline - target))),
        "midpoint_baseline_rmse_bb": float(np.sqrt(np.mean((baseline - target) ** 2))),
    }


def _size_report(
    lanes: Sequence[Mapping[str, Any]],
    component: Mapping[str, Any],
    *,
    include_inputs: bool = False,
) -> Any:
    target = np.asarray(sum((list(lane["targets"]) for lane in lanes), []), dtype=float)
    candidate = np.asarray(sum((list(lane["candidate"]) for lane in lanes), []), dtype=float)
    baseline = np.asarray(sum((list(lane["baseline"]) for lane in lanes), []), dtype=float)
    if len(target) == 0:
        raise RuntimeError("successor conditional-size adjudication set is empty")
    panels = np.asarray(sum((list(lane["panels"]) for lane in lanes), []), dtype=object)
    streets = np.asarray(sum((list(lane["streets"]) for lane in lanes), []), dtype=object)
    clusters = np.asarray(sum((list(lane["clusters"]) for lane in lanes), []), dtype=object)
    table_sizes = np.asarray(sum((list(lane["table_sizes"]) for lane in lanes), []), dtype=int)
    missing_panels = sorted(set(REQUIRED_PANELS) - set(panels.astype(str)))
    if missing_panels:
        raise RuntimeError(f"required successor conditional-size panels are empty: {missing_panels}")
    per_panel = {
        panel: _regression_metrics(target[panels == panel], candidate[panels == panel], baseline[panels == panel])
        for panel in REQUIRED_PANELS if (panels == panel).any()
    }
    per_street = {
        str(street): _regression_metrics(target[streets == street], candidate[streets == street], baseline[streets == street])
        for street in sorted(set(streets))
    }
    per_table_size = {
        str(size): _regression_metrics(target[table_sizes == size], candidate[table_sizes == size], baseline[table_sizes == size])
        for size in sorted(set(table_sizes))
    }
    report = seal_document({
        "schema_version": "stage6_conditional_size_successor_lockbox_report_v1",
        "frozen_candidate_self_sha256": component["self_sha256"],
        "overall_metrics": _regression_metrics(target, candidate, baseline),
        "per_panel_metrics": per_panel,
        "per_street_metrics": per_street,
        "per_table_size_metrics": per_table_size,
        "table_size_derivation": "max_active_player_count_before_across_all_successor_decisions_within_exact_game_id_plus_hand_id_before_action_filtering",
        "unknown_bound_abstention_count": int(sum(int(lane["abstained"]) for lane in lanes)),
        "illegal_postprojection_count": 0,
        "selection_or_tuning_event_count": 0,
        "successor_lockbox_opened": True,
        "promotion_authority": False,
    })
    if not include_inputs:
        return report
    return report, {
        "difference": np.abs(candidate - target) - np.abs(baseline - target),
        "clusters": clusters,
    }


def _paired_cluster_bootstrap(
    differences: Mapping[str, np.ndarray],
    clusters: Sequence[str],
    seed: int,
) -> dict[str, dict[str, Any]]:
    cluster_values, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    if len(cluster_values) < 2:
        raise RuntimeError("paired cluster bootstrap requires at least two hand clusters")
    counts = np.bincount(inverse).astype(float)
    names = list(differences)
    sums = np.column_stack([
        np.bincount(inverse, weights=np.asarray(differences[name], dtype=float), minlength=len(cluster_values))
        for name in names
    ])
    rng = np.random.default_rng(seed)
    samples = np.empty((BOOTSTRAP_REPLICATES, len(names)), dtype=float)
    batch_size = 25
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(cluster_values), size=(stop - start, len(cluster_values)))
        denominator = counts[indices].sum(axis=1)
        samples[start:stop] = sums[indices].sum(axis=1) / denominator[:, None]
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    result: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(names):
        values = samples[:, column]
        observed = np.asarray(differences[name], dtype=float)
        result[name] = {
            "paired_difference_direction": "candidate_minus_baseline_lower_is_better",
            "observed_mean_difference": float(observed.mean()),
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "confidence_interval_lower": float(np.quantile(values, alpha / 2.0)),
            "confidence_interval_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "one_sided_p_value": float((1 + np.count_nonzero(values >= 0.0)) / (BOOTSTRAP_REPLICATES + 1)),
            "cluster_count": int(len(cluster_values)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": seed,
        }
    return result


def _holm_bonferroni(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered = sorted(comparisons, key=lambda name: (float(comparisons[name]["one_sided_p_value"]), name))
    count = len(ordered)
    previous = 0.0
    adjusted: dict[str, dict[str, Any]] = {}
    for rank, name in enumerate(ordered):
        raw = float(comparisons[name]["one_sided_p_value"])
        value = max(previous, min(1.0, (count - rank) * raw))
        previous = value
        row = dict(comparisons[name])
        row["holm_adjusted_p_value"] = value
        row["passes_preregistered_superiority_rule"] = (
            row["observed_mean_difference"] < 0.0
            and row["confidence_interval_upper"] < 0.0
            and value < 0.05
        )
        adjusted[name] = row
    return dict(sorted(adjusted.items()))


def _event_paths(stage_drive_root: Path, anchor: Mapping[str, Any]) -> tuple[Path, Path]:
    root = stage_drive_root / "runtime_state" / "notebook09_successor_lockbox_events"
    root.mkdir(parents=True, exist_ok=True)
    event = anchor["self_sha256"]
    return root / f"{event}.claimed.json", root / f"{event}.completed.json"


def _write_exclusive_document(path: Path, document: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(document) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    if path.stat().st_size != len(raw) or sha256_file(path) != hashlib.sha256(raw).hexdigest():
        raise RuntimeError(f"Drive readback mismatch: {path.name}")


def _read_event_claim(path: Path, anchor: Mapping[str, Any], notebook08: Mapping[str, Any]) -> dict[str, Any]:
    claim = json.loads(path.read_text(encoding="utf-8"))
    validate_document(claim)
    successor = notebook08["freeze"]["successor_holdout_manifest"]
    if (
        claim.get("schema_version") != "stage6_notebook09_successor_lockbox_event_claim_v1"
        or claim.get("status") != "CLAIMED_BEFORE_SUCCESSOR_TARGET_READ"
        or claim.get("event_anchor_self_sha256") != anchor["self_sha256"]
        or claim.get("candidate_freeze_self_sha256") != notebook08["freeze"]["self_sha256"]
        or claim.get("successor_mapping_root_digest") != successor["mapping_root_digest"]
        or claim.get("selection_or_tuning_after_open") is not False
    ):
        raise RuntimeError("Notebook 09 existing successor-lockbox claim is invalid")
    return claim


def _completed_result(stage_drive_root: Path, completed_path: Path, anchor: Mapping[str, Any]) -> dict[str, Any] | None:
    if not completed_path.exists():
        return None
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    validate_document(completed)
    if (
        completed.get("schema_version") != "stage6_notebook09_successor_lockbox_event_completion_v1"
        or completed.get("event_anchor_self_sha256") != anchor["self_sha256"]
        or completed.get("status") not in {
            "PASS_STAGE6_COMPLETE_PROMOTION_AUTHORIZED",
            "PASS_STAGE6_COMPLETE_PROMOTION_HELD",
        }
    ):
        raise RuntimeError("Notebook 09 completion receipt is invalid")
    result = completed.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Notebook 09 completion result binding is missing")
    path = stage_drive_root / "returned_results" / str(result["name"])
    if not path.is_file() or path.stat().st_size != int(result["bytes"]) or sha256_file(path) != result["sha256"]:
        raise RuntimeError("Notebook 09 completed result readback mismatch")
    return {**dict(result), "status": "PASS", "resumed_from_completed_event": True}


def _fixed_zip_write(target: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def _publish(
    stage_drive_root: Path,
    candidate: Mapping[str, Any],
    report: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = [
        ("candidate_manifest.json", canonical_json_bytes(candidate) + b"\n"),
        ("successor_lockbox_report.json", canonical_json_bytes(report) + b"\n"),
        ("controller_receipt.json", canonical_json_bytes(controller) + b"\n"),
    ]
    manifest = seal_document({
        "schema_version": "stage6_notebook09_result_manifest_v1",
        "notebook_token": "09",
        "member_order": RESULT_MEMBER_ORDER,
        "members": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in payloads
        ],
    })
    payloads.append(("RESULT_MANIFEST.json", canonical_json_bytes(manifest) + b"\n"))
    temporary = Path(tempfile.gettempdir()) / f"stage6-notebook09-{controller['self_sha256'][:16]}.zip"
    _fixed_zip_write(temporary, payloads)
    result_sha = sha256_file(temporary)
    destination_root = stage_drive_root / "returned_results"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"09_Stage6_Notebook09_{controller['self_sha256'][:16]}_{result_sha[:16]}.zip"
    if destination.exists():
        if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
            raise RuntimeError("existing Notebook 09 result has different bytes")
    else:
        with temporary.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
    if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
        raise RuntimeError("Notebook 09 returned-result Drive readback mismatch")
    temporary.unlink(missing_ok=True)
    return {
        "name": destination.name,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": result_sha,
        "candidate_manifest_self_sha256": candidate["self_sha256"],
        "successor_lockbox_report_self_sha256": report["self_sha256"],
        "stage_complete": True,
        "promotion_authority": bool(candidate["promotion_authority"]),
    }


def run_notebook09(stage_drive_root: Path, package_root: Path, requested_workers: str = "auto") -> dict[str, Any]:
    if "OneDrive" in str(stage_drive_root) or "OneDrive" in str(package_root):
        raise RuntimeError("historical OneDrive checkout is forbidden")
    notebook08 = _read_notebook08_result(package_root / "results" / EXPECTED_NOTEBOOK08_NAME)
    anchor = _read_event_anchor(package_root / "event_anchor.json", notebook08)
    _validate_incumbent_artifacts(package_root)
    worker_report = resolve_workers(requested_workers)
    workers = int(worker_report["live_workers"])
    successor = notebook08["freeze"]["successor_holdout_manifest"]
    raw_root = _materialization_path(stage_drive_root)
    recomputed = _recompute_successor_manifest_before_target_read(raw_root, successor)
    print(json.dumps({
        "event": "stage6_successor_mapping_verified_before_target_read",
        "mapping_root_digest": recomputed["mapping_root_digest"],
        "source_test_rows": recomputed["source_test_row_count"],
        "successor_final_rows": recomputed["successor_final_row_count"],
    }, sort_keys=True), flush=True)

    claimed_path, completed_path = _event_paths(stage_drive_root, anchor)
    completed = _completed_result(stage_drive_root, completed_path, anchor)
    if completed is not None:
        print(json.dumps({"event": "stage6_notebook09_already_complete", **completed}, sort_keys=True), flush=True)
        return completed
    resumed_after_claimed_operational_failure = claimed_path.exists()
    if resumed_after_claimed_operational_failure:
        claim = _read_event_claim(claimed_path, anchor, notebook08)
    else:
        claim = seal_document({
            "schema_version": "stage6_notebook09_successor_lockbox_event_claim_v1",
            "status": "CLAIMED_BEFORE_SUCCESSOR_TARGET_READ",
            "event_anchor_self_sha256": anchor["self_sha256"],
            "candidate_freeze_self_sha256": notebook08["freeze"]["self_sha256"],
            "candidate_payload_root_digest": notebook08["freeze"]["candidate_payload_root_digest"],
            "successor_holdout_manifest_self_sha256": successor["self_sha256"],
            "successor_mapping_root_digest": successor["mapping_root_digest"],
            "selection_or_tuning_after_open": False,
        })
        _write_exclusive_document(claimed_path, claim)
    print(json.dumps({
        "event": "stage6_notebook09_start",
        "worker_resolution": worker_report,
        "successor_lockbox_event_claimed": True,
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
    }), flush=True)

    # The claim precedes this verifier because it hashes target-bearing Parquets.
    root = materialization_root(stage_drive_root)
    first_successor_utc = str(successor["first_successor_final_decision_utc"])
    hand_size_lanes = _parallel_lanes(
        _hand_size_successor_lane, root, workers, first_successor_utc,
    )
    hand_sizes = _successor_hand_size_map(hand_size_lanes)
    print(json.dumps({
        "event": "stage6_successor_all_decision_hand_size_map_complete",
        "hand_count": len(hand_sizes),
        "table_sizes": sorted(set(hand_sizes.values())),
    }, sort_keys=True), flush=True)
    components = notebook08["freeze"]["selected_components"]
    context_root = Path(tempfile.mkdtemp(prefix="stage6-notebook09-incumbent-native-"))
    incumbent_context = _prepare_incumbent_native_context(
        stage_drive_root, package_root, root, first_successor_utc, context_root, anchor
    )
    print(json.dumps({
        "event": "stage6_incumbent_native_context_complete",
        **incumbent_context,
    }, sort_keys=True), flush=True)
    preflop_lanes = _parallel_lanes(
        _behavior_successor_lane, root, workers, "preflop",
        components["preflop_behavior"], first_successor_utc, str(package_root), hand_sizes,
        str(context_root),
    )
    postflop_lanes = _parallel_lanes(
        _behavior_successor_lane, root, workers, "postflop",
        components["postflop_behavior"], first_successor_utc, str(package_root), hand_sizes,
        str(context_root),
    )
    size_lanes = _parallel_lanes(
        _size_successor_lane, root, workers,
        components["conditional_size"], first_successor_utc, hand_sizes,
    )
    preflop, preflop_inputs = _behavior_report(
        preflop_lanes, components["preflop_behavior"], include_inputs=True
    )
    postflop, postflop_inputs = _behavior_report(
        postflop_lanes, components["postflop_behavior"], include_inputs=True
    )
    size, size_inputs = _size_report(
        size_lanes, components["conditional_size"], include_inputs=True
    )
    shutil.rmtree(context_root)

    comparisons: dict[str, dict[str, Any]] = {}
    for scope_index, (scope, inputs) in enumerate((
        ("preflop", preflop_inputs), ("postflop", postflop_inputs),
    )):
        scoped = {}
        for baseline, difference in inputs["differences"].items():
            row = _paired_cluster_bootstrap(
                {baseline: difference},
                inputs["difference_clusters"][baseline],
                BOOTSTRAP_SEED + scope_index,
            )
            scoped[baseline] = row[baseline]
        comparisons.update({f"{scope}:{name}": value for name, value in scoped.items()})
        print(json.dumps({
            "event": "stage6_behavior_bootstrap_complete",
            "scope": scope,
            "comparisons": len(scoped),
            "rows": len(inputs["truth"]),
        }, sort_keys=True), flush=True)
    size_comparison = _paired_cluster_bootstrap(
        {"midpoint_baseline": size_inputs["difference"]},
        size_inputs["clusters"], BOOTSTRAP_SEED + 2,
    )
    comparisons["conditional_size:midpoint_baseline"] = size_comparison["midpoint_baseline"]
    adjusted = _holm_bonferroni(comparisons)

    behavior_gates = {
        scope: all(
            adjusted[f"{scope}:{baseline}"]["passes_preregistered_superiority_rule"]
            for baseline in REQUIRED_BASELINES
        ) and scope_report["illegal_postprojection_count"] == 0
        for scope, scope_report in (("preflop", preflop), ("postflop", postflop))
    }
    size_gate = (
        adjusted["conditional_size:midpoint_baseline"]["passes_preregistered_superiority_rule"]
        and size["illegal_postprojection_count"] == 0
    )
    profile_component = components["profile"]
    profile_gate = (
        profile_component.get("runtime_profile_lookup_enabled") is True
        and profile_component.get("runtime_profile_adjustments_enabled") is False
        and profile_component.get("incomplete_or_partial_roster") == "abstain"
        and "active_opponent_count_before == active_player_count_before - 1" in profile_component.get("lookup_guard", "")
    )
    review_component = components["review"]
    review_gate = (
        review_component.get("authority_scope") == "review_only"
        and review_component.get("recommendation_authority") is False
        and review_component.get("action_authority") is False
        and len(review_component.get("runtime_concepts_enabled", [])) == review_component.get("eligible_concept_count") == 6
    )
    reference_component = components["reference"]
    reference_gate = (
        reference_component.get("exact_reference_rows") == 440
        and reference_component.get("strict_river_lane_receipt_parity_count") == 268
        and reference_component.get("runtime_mode") == "abstain_until_structured_stage6_promotion_receipt"
    )
    component_gates = {
        "preflop_behavior": behavior_gates["preflop"],
        "postflop_behavior": behavior_gates["postflop"],
        "conditional_size": size_gate,
        "profile": profile_gate,
        "review": review_gate,
        "reference": reference_gate,
    }
    promotion_authority = all(component_gates.values())

    report = seal_document({
        "schema_version": "stage6_notebook09_successor_lockbox_adjudication_report_v1",
        "status": "PASS_ONE_TIME_SUCCESSOR_LOCKBOX_ADJUDICATION_COMPLETE",
        "event_anchor_self_sha256": anchor["self_sha256"],
        "candidate_freeze_self_sha256": notebook08["freeze"]["self_sha256"],
        "candidate_payload_root_digest": notebook08["freeze"]["candidate_payload_root_digest"],
        "successor_holdout_manifest_self_sha256": successor["self_sha256"],
        "successor_mapping_root_digest": successor["mapping_root_digest"],
        "preflop_behavior": preflop,
        "postflop_behavior": postflop,
        "conditional_size": size,
        "paired_cluster_bootstrap_holm": adjusted,
        "incumbent_native_context": incumbent_context,
        "component_qualification_gates": component_gates,
        "profile_runtime_lookup_enabled": profile_gate,
        "profile_runtime_adjustments_enabled": False,
        "review_runtime_concepts_enabled": review_component["runtime_concepts_enabled"] if review_gate else [],
        "review_authority_scope": "review_only",
        "reference_runtime_authority": reference_gate,
        "per_table_size_metrics_status": "PASS_COMPUTED_FROM_VALIDATED_PER_HAND_DERIVATION",
        "selection_or_tuning_event_count": 0,
        "successor_lockbox_adjudication_event_count": 1,
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
        "successor_lockbox_opened": True,
        "scientific_authority": promotion_authority,
        "promotion_authority": promotion_authority,
        "live_action_authority": False,
    })
    candidate = seal_document({
        "schema_version": "stage6_final_candidate_manifest_v2",
        "status": "STAGE6_COMPLETE_PROMOTION_AUTHORIZED" if promotion_authority else "STAGE6_COMPLETE_PROMOTION_HELD",
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "notebook08_result_sha256": EXPECTED_NOTEBOOK08_SHA256,
        "candidate_freeze_self_sha256": notebook08["freeze"]["self_sha256"],
        "candidate_payload_root_digest": notebook08["freeze"]["candidate_payload_root_digest"],
        "successor_lockbox_report_self_sha256": report["self_sha256"],
        "selected_component_self_sha256": {
            name: value["self_sha256"] for name, value in sorted(components.items())
        },
        "component_qualification_gates": component_gates,
        "qualified_component_authority": {
            "preflop_behavior": "read_only_prediction_layer",
            "postflop_behavior": "read_only_prediction_layer",
            "conditional_size": "read_only_visible_bounds_projection_layer",
            "profile": "exact_complete_roster_lookup_only_no_policy_adjustment",
            "review": "review_only_no_recommendation_or_action_authority",
            "reference": "exact_state_router_only",
        },
        "promotion_disposition": "PROMOTE_TO_READ_ONLY_ASSISTANT" if promotion_authority else "HOLD_FAILED_COMPONENTS",
        "promotion_hold_components": sorted(name for name, passed in component_gates.items() if not passed),
        "selection_or_tuning_after_successor_open": False,
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
        "successor_lockbox_opened": True,
        "stage_complete": True,
        "scientific_authority": promotion_authority,
        "promotion_authority": promotion_authority,
        "live_action_authority": False,
    })
    controller_status = (
        "PASS_STAGE6_COMPLETE_PROMOTION_AUTHORIZED"
        if promotion_authority else "PASS_STAGE6_COMPLETE_PROMOTION_HELD"
    )
    controller = seal_document({
        "schema_version": "stage6_notebook09_controller_receipt_v2",
        "status": controller_status,
        "notebook_token": "09",
        "event_anchor_self_sha256": anchor["self_sha256"],
        "event_claim_self_sha256": claim["self_sha256"],
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "logical_lane_count": LANE_COUNT,
        "worker_resolution": worker_report,
        "candidate_manifest_self_sha256": candidate["self_sha256"],
        "successor_lockbox_report_self_sha256": report["self_sha256"],
        "selection_or_tuning_event_count": 0,
        "successor_lockbox_adjudication_event_count": 1,
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
        "successor_lockbox_opened": True,
        "stage_complete": True,
        "scientific_authority": promotion_authority,
        "promotion_authority": promotion_authority,
        "live_action_authority": False,
    })
    result = _publish(stage_drive_root, candidate, report, controller)
    completion = seal_document({
        "schema_version": "stage6_notebook09_successor_lockbox_event_completion_v1",
        "status": controller_status,
        "event_anchor_self_sha256": anchor["self_sha256"],
        "event_claim_self_sha256": claim["self_sha256"],
        "result": result,
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
        "successor_lockbox_reopen_required": False,
        "promotion_authority": promotion_authority,
    })
    _write_exclusive_document(completed_path, completion)
    final = {
        "status": "PASS",
        **result,
        "event_completion_self_sha256": completion["self_sha256"],
        "resumed_after_claimed_operational_failure": resumed_after_claimed_operational_failure,
        "resumed_from_completed_event": False,
    }
    print(json.dumps({"event": "stage6_notebook09_complete", **final}, sort_keys=True, indent=2), flush=True)
    return final
