"""Validation-only calibration and candidate freeze for Stage 6 Notebook 08."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stage6_metrics import multiclass_metric_report, selective_risk_curve
from stage6_notebooks02_07_runtime import (
    ACTIONS,
    LANE_COUNT,
    NOTEBOOK01_RUN_ID,
    NOTEBOOK01_SPLIT_ROOT,
    _lane_paths,
    _behavior_validation_lane,
    _parallel_lanes,
    _size_report,
    _size_validation_lane,
    canonical_json_bytes,
    materialization_root,
    resolve_workers,
    seal_document,
    sha256_file,
    validate_document,
)


EXPECTED_RESULTS = {
    "02": ("02_Stage6_Notebook02_51da377e41e19349_592d6a0d3930774f.zip", "592d6a0d3930774fbbe91a30e7031c4fd1e7aaa584a04d28d6506702ed530c90"),
    "03": ("03_Stage6_Notebook03_7027c54a5ae160f2_8d34fa90a0a47e9c.zip", "8d34fa90a0a47e9c36c3ca2fab523d212db0ee05c13061d12f2257de50513f07"),
    "04": ("04_Stage6_Notebook04_7810a60e6a9091a9_1dd775d467a75012.zip", "1dd775d467a75012f3d831a321fd7ee8e5f101b633e3af340fd49203efcf3b5f"),
    "05": ("05_Stage6_Notebook05_df258aa4ed3a9411_8a65d4249d2d700b.zip", "8a65d4249d2d700bf890d27f44c6d4ddd3bf55b99037ffa744d67f0f09d7161f"),
    "06": ("06_Stage6_Notebook06_4bc5b687af43f7ae_a03abf4765516479.zip", "a03abf4765516479f2cfda18e2ccd32ac5877589e6cac3b616ae1a7aa3e67e1a"),
    "07": ("07_Stage6_Notebook07_290a497d0ee7dde6_486d0d992daf1a2e.zip", "486d0d992daf1a2e6641bd3f2b4e200b6b538f5d31bd5e029ff7b40597c846ae"),
}
TEMPERATURE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
RISK_COVERAGES = (0.25, 0.5, 0.75, 0.9, 1.0)
SELECTIVE_COVERAGE = 0.9
SUCCESSOR_FINAL_HAND_FRACTION = 0.25
SUCCESSOR_FINAL_PANEL = "successor_final_private_lockbox"
SCIENTIFIC_GATE_RAW_SHA256 = "a8dc51229e9eedbf73fe59eda1b30413d4bb379efabcc3995b0d87ccf3a1cca6"
CONSUMED_PREDECESSOR_RESULT_NAME = "09_Stage6_Notebook09_27e1f3d8d4239913_67a7fbdeaf8d37f8.zip"
CONSUMED_PREDECESSOR_RESULT_BYTES = 5052
CONSUMED_PREDECESSOR_RESULT_SHA256 = "67a7fbdeaf8d37f875bfcd033c18fa4c845445f31b895a650d806d29d4f256fd"
CONSUMED_PREDECESSOR_CONTROLLER_SELF_SHA256 = "27e1f3d8d4239913642c821e0b90ec00d816f98a9454b2bfbc0f96eb16a140e7"
CONSUMED_PREDECESSOR_EVENT_ANCHOR_SELF_SHA256 = "f8a3e1c2713bb6a2c07daf45b85a851267767f41d03bf6521ca4fb3e38635725"


def _read_consumed_predecessor(path: Path) -> dict[str, Any]:
    if (
        path.name != CONSUMED_PREDECESSOR_RESULT_NAME
        or path.stat().st_size != CONSUMED_PREDECESSOR_RESULT_BYTES
        or sha256_file(path) != CONSUMED_PREDECESSOR_RESULT_SHA256
    ):
        raise RuntimeError("consumed predecessor Notebook 09 identity mismatch")
    expected = [
        "candidate_manifest.json",
        "private_lockbox_report.json",
        "controller_receipt.json",
        "RESULT_MANIFEST.json",
    ]
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != expected:
            raise RuntimeError("consumed predecessor Notebook 09 membership mismatch")
        raw = {name: archive.read(name) for name in expected}
    documents = {name: json.loads(value) for name, value in raw.items()}
    for document in documents.values():
        validate_document(document)
    controller = documents["controller_receipt.json"]
    report = documents["private_lockbox_report.json"]
    if (
        controller.get("self_sha256") != CONSUMED_PREDECESSOR_CONTROLLER_SELF_SHA256
        or controller.get("event_anchor_self_sha256") != CONSUMED_PREDECESSOR_EVENT_ANCHOR_SELF_SHA256
        or controller.get("status") != "PASS_STAGE6_COMPLETE_NO_PROMOTION"
        or controller.get("notebook01_split_assignment_root_digest") != NOTEBOOK01_SPLIT_ROOT
        or controller.get("private_lockbox_opened") is not True
        or controller.get("private_lockbox_adjudication_event_count") != 1
        or controller.get("promotion_authority") is not False
        or report.get("private_lockbox_opened") is not True
        or report.get("private_lockbox_adjudication_event_count") != 1
    ):
        raise RuntimeError("consumed predecessor Notebook 09 authority binding mismatch")
    return {
        "result_name": path.name,
        "result_bytes": path.stat().st_size,
        "result_sha256": CONSUMED_PREDECESSOR_RESULT_SHA256,
        "controller_self_sha256": controller["self_sha256"],
        "event_anchor_self_sha256": controller["event_anchor_self_sha256"],
        "consumed_panel": "final_private_lockbox",
        "private_lockbox_opened": True,
        "promotion_authority": False,
    }


def _successor_holdout_manifest_from_frame(
    frame: pd.DataFrame,
    consumed_predecessor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "row_id", "game_id", "hand_id", "decision_utc", "panel", "role",
        "eligible_for_selection",
    }
    if set(frame.columns) != required:
        raise RuntimeError("successor holdout metadata columns are invalid")
    untouched = frame.loc[
        frame["role"].astype(str).eq("test")
        & ~frame["panel"].astype(str).eq("final_private_lockbox")
    ].copy()
    if untouched.empty or not untouched["eligible_for_selection"].eq(False).all():
        raise RuntimeError("untouched non-private test rows are unavailable")
    if untouched["row_id"].astype(str).duplicated().any():
        raise RuntimeError("successor holdout source row identities are not unique")
    source_panels = set(untouched["panel"].astype(str))
    expected_panels = {"future_known_player", "unseen_player_cold_start", "unseen_game"}
    if source_panels != expected_panels:
        raise RuntimeError("successor holdout source panels differ from the preregistered test panels")

    hand_rows = (
        untouched.groupby(["game_id", "hand_id"], sort=False, as_index=False)
        .agg(
            first_decision_utc=("decision_utc", "min"),
            last_decision_utc=("decision_utc", "max"),
            row_count=("row_id", "size"),
        )
        .sort_values(
            ["first_decision_utc", "game_id", "hand_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if len(hand_rows) < 8:
        raise RuntimeError("successor holdout has too few untouched hand groups")
    running_max = hand_rows["last_decision_utc"].astype(str).cummax()
    target_final_hands = max(1, int(math.ceil(SUCCESSOR_FINAL_HAND_FRACTION * len(hand_rows))))
    candidates: list[tuple[int, int]] = []
    for index in range(1, len(hand_rows)):
        if str(hand_rows.loc[index, "first_decision_utc"]) > str(running_max.iloc[index - 1]):
            candidates.append((abs((len(hand_rows) - index) - target_final_hands), index))
    if not candidates:
        raise RuntimeError("successor holdout has no strict chronological hand-interval gap")
    _, boundary = min(candidates)
    final_hands = {
        (str(row.game_id), str(row.hand_id))
        for row in hand_rows.iloc[boundary:].itertuples(index=False)
    }
    if not final_hands or len(final_hands) == len(hand_rows):
        raise RuntimeError("successor final lockbox boundary is degenerate")

    digest = hashlib.sha256()
    final_rows = 0
    source_panel_row_counts: dict[str, int] = {}
    retained_panel_row_counts: dict[str, int] = {}
    for row in untouched.sort_values("row_id", kind="mergesort").itertuples(index=False):
        original_panel = str(row.panel)
        source_panel_row_counts[original_panel] = source_panel_row_counts.get(original_panel, 0) + 1
        is_final = (str(row.game_id), str(row.hand_id)) in final_hands
        mapped_panel = SUCCESSOR_FINAL_PANEL if is_final else original_panel
        if is_final:
            final_rows += 1
        else:
            retained_panel_row_counts[original_panel] = retained_panel_row_counts.get(original_panel, 0) + 1
        digest.update(canonical_json_bytes({"row_id": str(row.row_id), "panel": mapped_panel}) + b"\n")
    if set(retained_panel_row_counts) != expected_panels or min(retained_panel_row_counts.values()) <= 0:
        raise RuntimeError("successor holdout removed an established evaluation panel")

    first_final = str(hand_rows.iloc[boundary]["first_decision_utc"])
    last_retained = str(running_max.iloc[boundary - 1])
    if first_final <= last_retained:
        raise RuntimeError("successor final lockbox is not strictly later than retained panels")
    predecessor = dict(consumed_predecessor or {
        "result_name": CONSUMED_PREDECESSOR_RESULT_NAME,
        "result_bytes": CONSUMED_PREDECESSOR_RESULT_BYTES,
        "result_sha256": CONSUMED_PREDECESSOR_RESULT_SHA256,
        "controller_self_sha256": CONSUMED_PREDECESSOR_CONTROLLER_SELF_SHA256,
        "event_anchor_self_sha256": CONSUMED_PREDECESSOR_EVENT_ANCHOR_SELF_SHA256,
        "consumed_panel": "final_private_lockbox",
        "private_lockbox_opened": True,
        "promotion_authority": False,
    })
    return seal_document({
        "schema_version": "stage6_successor_holdout_manifest_v1",
        "status": "PASS_UNTOUCHED_TEST_PANELS_REPARTITIONED_BEFORE_TARGET_READ",
        "source_notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "source_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "source_role": "test",
        "source_panels": sorted(expected_panels),
        "source_scientific_gate_schema_version": "stage6_scientific_gate_v1",
        "source_scientific_gate_raw_sha256": SCIENTIFIC_GATE_RAW_SHA256,
        "consumed_predecessor_notebook09": predecessor,
        "consumed_original_private_lockbox_excluded": True,
        "replacement_panel_registration": {
            "original_required_panel": "final_private_lockbox",
            "replacement_panel": SUCCESSOR_FINAL_PANEL,
            "reason": "original required panel was consumed by the exact bound predecessor no-promotion event",
        },
        "effective_required_panels": sorted((*expected_panels, SUCCESSOR_FINAL_PANEL)),
        "notebook09_enforcement_requirements": {
            "recompute_mapping_from_split_metadata_before_target_read": True,
            "required_mapping_root_digest": digest.hexdigest(),
            "reject_original_private_panel_before_target_read": True,
            "rejected_original_private_panel": "final_private_lockbox",
            "claim_successor_event_before_target_read": True,
            "selection_or_tuning_after_claim_allowed": False,
        },
        "source_test_row_count": int(len(untouched)),
        "source_test_hand_count": int(len(hand_rows)),
        "source_panel_row_counts": dict(sorted(source_panel_row_counts.items())),
        "retained_panel_row_counts": dict(sorted(retained_panel_row_counts.items())),
        "successor_final_panel": SUCCESSOR_FINAL_PANEL,
        "successor_final_hand_fraction_target": SUCCESSOR_FINAL_HAND_FRACTION,
        "successor_final_hand_count": len(final_hands),
        "successor_final_row_count": final_rows,
        "last_retained_decision_utc": last_retained,
        "first_successor_final_decision_utc": first_final,
        "strict_chronological_gap": True,
        "mapping_root_digest": digest.hexdigest(),
        "target_columns_read": [],
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def build_successor_holdout_manifest(root: Path, package_root: Path) -> dict[str, Any]:
    gate_path = package_root / "contracts" / "stage6_scientific_gate_v1.json"
    if not gate_path.is_file() or sha256_file(gate_path) != SCIENTIFIC_GATE_RAW_SHA256:
        raise RuntimeError("Stage 6 scientific gate identity mismatch")
    consumed_predecessor = _read_consumed_predecessor(
        package_root / "results" / "consumed_predecessor" / CONSUMED_PREDECESSOR_RESULT_NAME
    )
    frames: list[pd.DataFrame] = []
    columns = [
        "row_id", "game_id", "hand_id", "decision_utc", "panel", "role",
        "eligible_for_selection",
    ]
    for lane in range(LANE_COUNT):
        _, split_path, _ = _lane_paths(root, lane)
        frames.append(pd.read_parquet(split_path, columns=columns))
    return _successor_holdout_manifest_from_frame(
        pd.concat(frames, ignore_index=True),
        consumed_predecessor,
    )


def _read_result(path: Path, token: str) -> dict[str, Any]:
    expected_name, expected_sha = EXPECTED_RESULTS[token]
    if path.name != expected_name or sha256_file(path) != expected_sha:
        raise RuntimeError(f"Notebook {token} returned-result identity mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected_order = ["candidate.json", "validation_report.json", "controller_receipt.json", "RESULT_MANIFEST.json"]
        if names != expected_order or len(names) != len(set(names)):
            raise RuntimeError(f"Notebook {token} result membership is invalid")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise RuntimeError(f"Notebook {token} result has an unsafe member")
        raw = {name: archive.read(name) for name in names}
    documents = {name: json.loads(value) for name, value in raw.items()}
    for document in documents.values():
        validate_document(document)
    manifest = documents["RESULT_MANIFEST.json"]
    if (
        manifest.get("schema_version") != "stage6_notebook_result_manifest_v1"
        or manifest.get("notebook_token") != token
        or manifest.get("member_order") != expected_order[:-1]
        or [row.get("path") for row in manifest.get("members", [])] != expected_order[:-1]
    ):
        raise RuntimeError(f"Notebook {token} result manifest is invalid")
    for identity in manifest["members"]:
        payload = raw[identity["path"]]
        if len(payload) != identity["bytes"] or hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            raise RuntimeError(f"Notebook {token} result member mismatch")
    candidate = documents["candidate.json"]
    report = documents["validation_report.json"]
    controller = documents["controller_receipt.json"]
    if (
        controller.get("status") != "PASS"
        or controller.get("notebook_token") != token
        or controller.get("notebook01_run_identity") != NOTEBOOK01_RUN_ID
        or controller.get("notebook01_split_assignment_root_digest") != NOTEBOOK01_SPLIT_ROOT
        or controller.get("logical_lane_count") != LANE_COUNT
        or controller.get("candidate_self_sha256") != candidate.get("self_sha256")
        or controller.get("validation_report_self_sha256") != report.get("self_sha256")
        or controller.get("private_lockbox_opened") is not False
        or controller.get("promotion_authority") is not False
        or report.get("test_selection_event_count") != 0
        or report.get("private_lockbox_opened") is not False
    ):
        raise RuntimeError(f"Notebook {token} result authority binding is invalid")
    return {
        "candidate": candidate,
        "report": report,
        "controller": controller,
        "result_sha256": expected_sha,
        "result_name": expected_name,
    }


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature must be positive and finite")
    powered = np.power(probabilities, 1.0 / temperature)
    totals = powered.sum(axis=1)
    if (totals <= 0).any():
        raise RuntimeError("calibration received an empty legal probability row")
    return powered / totals[:, None]


def _tie_inclusive_threshold_report(
    truth: Sequence[str],
    probabilities: np.ndarray,
    target_coverage: float,
) -> dict[str, Any]:
    if not truth or probabilities.shape != (len(truth), len(ACTIONS)):
        raise RuntimeError("selective-threshold inputs are inconsistent")
    confidence = probabilities.max(axis=1)
    predicted = np.asarray([ACTIONS[index] for index in probabilities.argmax(axis=1)], dtype=object)
    truth_array = np.asarray(truth, dtype=object)
    requested_rows = max(1, int(math.ceil(target_coverage * len(truth))))
    order = np.lexsort((np.arange(len(truth)), -confidence))
    threshold = float(confidence[order[requested_rows - 1]])
    accepted = confidence >= threshold
    cutoff_ties = confidence == threshold
    accepted_rows = int(accepted.sum())
    if accepted_rows < requested_rows:
        raise RuntimeError("tie-inclusive threshold failed the requested coverage")
    return {
        "requested_coverage": target_coverage,
        "requested_rows": requested_rows,
        "confidence_threshold": threshold,
        "comparison": "greater_than_or_equal",
        "cutoff_tie_policy": "accept_all",
        "cutoff_tie_rows": int(cutoff_ties.sum()),
        "accepted_rows": accepted_rows,
        "actual_coverage": accepted_rows / len(truth),
        "selective_risk": float((predicted[accepted] != truth_array[accepted]).mean()),
    }


def _behavior_freeze(
    root: Path,
    workers: int,
    model: Mapping[str, Any],
    street_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lanes = _parallel_lanes(_behavior_validation_lane, root, workers, street_mode, model)
    truth = sum((list(lane["truth"]) for lane in lanes), [])
    panels = np.asarray(sum((list(lane["panels"]) for lane in lanes), []), dtype=object)
    candidate = np.concatenate([lane["candidate"] for lane in lanes], axis=0)
    baseline = np.concatenate([lane["baseline"] for lane in lanes], axis=0)
    if not truth:
        raise RuntimeError(f"{street_mode} validation replay is empty")
    alternatives: dict[str, dict[str, Any]] = {}
    matrices = {"trained_candidate": candidate, "class_frequency_fallback": baseline}
    selected_key: tuple[Any, ...] | None = None
    selected_name = ""
    selected_temperature = 0.0
    selected_matrix: np.ndarray | None = None
    for family in ("trained_candidate", "class_frequency_fallback"):
        for temperature in TEMPERATURE_GRID:
            calibrated = _temperature_scale(matrices[family], temperature)
            metrics = multiclass_metric_report(truth, calibrated, ACTIONS)
            name = f"{family}:temperature={temperature:g}"
            alternatives[name] = metrics
            key = (
                metrics["log_loss"],
                -metrics["macro_f1"],
                metrics["expected_calibration_error"],
                metrics["multiclass_brier_score"],
                0 if family == "class_frequency_fallback" else 1,
                temperature,
            )
            if selected_key is None or key < selected_key:
                selected_key = key
                selected_name = family
                selected_temperature = temperature
                selected_matrix = calibrated
    assert selected_matrix is not None
    selected_metrics = multiclass_metric_report(truth, selected_matrix, ACTIONS)
    risk = selective_risk_curve(truth, selected_matrix, ACTIONS, coverages=RISK_COVERAGES)
    deployed_threshold = _tie_inclusive_threshold_report(truth, selected_matrix, SELECTIVE_COVERAGE)
    truth_array = np.asarray(truth, dtype=object)
    per_panel = {
        str(panel): multiclass_metric_report(truth_array[panels == panel].tolist(), selected_matrix[panels == panel], ACTIONS)
        for panel in sorted(set(panels))
    }
    candidate = seal_document({
        "schema_version": "stage6_calibrated_legal_behavior_fallback_v1",
        "street_scope": street_mode,
        "selected_family": selected_name,
        "selected_temperature": selected_temperature,
        "class_priors": list(model["class_priors"]),
        "label_order": list(ACTIONS),
        "legal_projection": "visible_mask_then_renormalize_else_abstain",
        "confidence_threshold": deployed_threshold["confidence_threshold"],
        "confidence_comparison": deployed_threshold["comparison"],
        "cutoff_tie_policy": deployed_threshold["cutoff_tie_policy"],
        "validation_actual_coverage": deployed_threshold["actual_coverage"],
        "validation_selective_risk": deployed_threshold["selective_risk"],
        "preregistered_target_coverage": SELECTIVE_COVERAGE,
        "abstention_rule": "abstain_below_confidence_threshold_or_on_missing_legal_mask",
        "source_candidate_self_sha256": model["self_sha256"],
        "source_candidate": dict(model),
        "trained_candidate_selected": selected_name == "trained_candidate",
        "fit_partition": "validation_only",
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    report = seal_document({
        "schema_version": "stage6_calibrated_behavior_validation_report_v1",
        "street_scope": street_mode,
        "selection_partition": "validation_only",
        "temperature_grid": list(TEMPERATURE_GRID),
        "selection_order": "log_loss_then_macro_f1_then_ece_then_brier_then_fail_closed_fallback",
        "temperature_selection_evidence": alternatives,
        "selected_family": selected_name,
        "selected_temperature": selected_temperature,
        "selected_metrics": selected_metrics,
        "selected_risk_coverage": risk,
        "deployed_tie_inclusive_threshold": deployed_threshold,
        "per_panel": per_panel,
        "trained_candidate_rejected": selected_name != "trained_candidate",
        "illegal_postprojection_count": 0,
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    return candidate, report


def _payload_root(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        {"component": name, "self_sha256": payload["self_sha256"]}
        for name, payload in sorted(payloads.items())
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _fixed_zip_write(target: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def _publish(stage_drive_root: Path, freeze: Mapping[str, Any], report: Mapping[str, Any], worker_report: Mapping[str, Any]) -> dict[str, Any]:
    validate_document(freeze)
    validate_document(report)
    controller = seal_document({
        "schema_version": "stage6_notebook08_controller_receipt_v1",
        "status": "PASS_VALIDATION_ONLY_CANDIDATE_FREEZE",
        "notebook_token": "08",
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "logical_lane_count": LANE_COUNT,
        "worker_resolution": dict(worker_report),
        "candidate_freeze_self_sha256": freeze["self_sha256"],
        "validation_report_self_sha256": report["self_sha256"],
        "private_lockbox_opened": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "live_action_authority": False,
    })
    payloads = [
        ("candidate_freeze.json", canonical_json_bytes(freeze) + b"\n"),
        ("validation_report.json", canonical_json_bytes(report) + b"\n"),
        ("controller_receipt.json", canonical_json_bytes(controller) + b"\n"),
    ]
    manifest = seal_document({
        "schema_version": "stage6_notebook08_result_manifest_v1",
        "notebook_token": "08",
        "member_order": [name for name, _ in payloads],
        "members": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in payloads
        ],
    })
    payloads.append(("RESULT_MANIFEST.json", canonical_json_bytes(manifest) + b"\n"))
    temporary = Path(tempfile.gettempdir()) / f"stage6-notebook08-{controller['self_sha256'][:16]}.zip"
    _fixed_zip_write(temporary, payloads)
    result_sha = sha256_file(temporary)
    destination_root = stage_drive_root / "returned_results"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"08_Stage6_Notebook08_{controller['self_sha256'][:16]}_{result_sha[:16]}.zip"
    if destination.exists():
        if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
            raise RuntimeError("existing Notebook 08 returned result has different bytes")
    else:
        with temporary.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
    if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
        raise RuntimeError("Notebook 08 returned-result Drive readback mismatch")
    temporary.unlink(missing_ok=True)
    return {
        "status": "PASS",
        "name": destination.name,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": result_sha,
        "candidate_freeze_self_sha256": freeze["self_sha256"],
        "candidate_payload_root_digest": freeze["candidate_payload_root_digest"],
        "private_lockbox_opened": False,
        "promotion_authority": False,
    }


def run_notebook08(stage_drive_root: Path, package_root: Path, requested_workers: str = "auto") -> dict[str, Any]:
    if "OneDrive" in str(stage_drive_root) or "OneDrive" in str(package_root):
        raise RuntimeError("historical OneDrive checkout is forbidden")
    root = materialization_root(stage_drive_root)
    results = {
        token: _read_result(package_root / "results" / name, token)
        for token, (name, _) in EXPECTED_RESULTS.items()
    }
    worker_report = resolve_workers(requested_workers)
    workers = int(worker_report["live_workers"])
    print(json.dumps({"event": "stage6_notebook08_start", "worker_resolution": worker_report}), flush=True)

    # This reads only frozen split metadata.  The exact successor mapping is
    # sealed before any target column from the still-unopened test panels is
    # decoded by Notebook 09.  The already-consumed original private panel is
    # excluded permanently and can never enter this successor event.
    successor_holdout = build_successor_holdout_manifest(root, package_root)
    print(json.dumps({
        "event": "stage6_successor_holdout_precommitted",
        "manifest_self_sha256": successor_holdout["self_sha256"],
        "mapping_root_digest": successor_holdout["mapping_root_digest"],
        "source_test_rows": successor_holdout["source_test_row_count"],
        "successor_final_rows": successor_holdout["successor_final_row_count"],
    }, sort_keys=True), flush=True)

    preflop, preflop_report = _behavior_freeze(root, workers, results["02"]["candidate"], "preflop")
    postflop, postflop_report = _behavior_freeze(root, workers, results["03"]["candidate"], "postflop")
    size_lanes = _parallel_lanes(_size_validation_lane, root, workers, results["04"]["candidate"])
    size_replay = _size_report(size_lanes)
    if size_replay != results["04"]["report"]:
        raise RuntimeError("Notebook 04 size-head validation replay changed")
    if (
        size_replay["illegal_postprojection_count"] != 0
        or size_replay["overall"]["candidate_mae_bb"] >= size_replay["overall"]["midpoint_baseline_mae_bb"]
    ):
        raise RuntimeError("Notebook 04 size head does not pass validation selection")

    size = seal_document({
        "schema_version": "stage6_selected_conditional_size_head_v1",
        "source_candidate": results["04"]["candidate"],
        "source_candidate_self_sha256": results["04"]["candidate"]["self_sha256"],
        "selection_partition": "validation_only",
        "selection_reason": "candidate_mae_below_midpoint_with_zero_illegal_postprojection",
        "unknown_or_inconsistent_bounds": "abstain",
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    profile = seal_document({
        "schema_version": "stage6_profile_runtime_disposition_v1",
        "candidate_thresholds": results["05"]["candidate"]["candidate_thresholds"],
        "selected_minimum_support": results["05"]["candidate"]["selected_minimum_support"],
        "runtime_profile_lookup_enabled": True,
        "targeting_mode": "exact_active_opponent_set_before_decision",
        "exact_target_field": "active_opponent_canonical_player_ids_before",
        "lookup_guard": "active_opponent_count_before == active_player_count_before - 1 and exact_target_id_count == active_opponent_count_before",
        "incomplete_or_partial_roster": "abstain",
        "support_is_statistic_specific": True,
        "aggressor_specific_metrics_enabled": False,
        "aggressor_specific_abstention_required": True,
        "runtime_profile_adjustments_enabled": False,
        "reason": "exact active-opponent-set lookup is available; unvalidated policy deltas remain disabled and missing exact aggressor identity abstains",
        "fallback": "no_profile_policy_adjustment",
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    eligible_review_concepts = sorted(
        name
        for name, screen in results["06"]["candidate"]["screens"].items()
        if screen.get("rebuild_eligible") is True
    )
    review = seal_document({
        "schema_version": "stage6_review_runtime_disposition_v1",
        "eligible_concept_count": results["06"]["candidate"]["eligible_concept_count"],
        "current_review_model_bytes_disposition": results["06"]["candidate"]["current_review_model_bytes_disposition"],
        "runtime_concepts_enabled": eligible_review_concepts,
        "authority_scope": "review_only",
        "recommendation_authority": False,
        "action_authority": False,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    reference = seal_document({
        "schema_version": "stage6_reference_runtime_disposition_v1",
        "source_candidate_self_sha256": results["07"]["candidate"]["self_sha256"],
        "exact_reference_rows": results["07"]["report"]["exact_reference_match_count"],
        "strict_river_lane_receipt_parity_count": results["07"]["report"]["strict_river_lane_receipt_parity_count"],
        "runtime_mode": "abstain_until_structured_stage6_promotion_receipt",
        "runtime_authority": False,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    payloads = {
        "preflop_behavior": preflop,
        "postflop_behavior": postflop,
        "conditional_size": size,
        "profile": profile,
        "review": review,
        "reference": reference,
    }
    freeze = seal_document({
        "schema_version": "stage6_notebook08_candidate_freeze_v1",
        "status": "PASS_VALIDATION_ONLY_CANDIDATE_FREEZE_PRIVATE_LOCKBOX_SEALED",
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "source_result_sha256": {token: result["result_sha256"] for token, result in results.items()},
        "successor_holdout_manifest": successor_holdout,
        "successor_holdout_manifest_self_sha256": successor_holdout["self_sha256"],
        "selected_components": payloads,
        "candidate_payload_root_digest": _payload_root(payloads),
        "calibration_partition": "validation_only",
        "threshold_selection_partition": "validation_only",
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "candidate_frozen": True,
        "scientific_authority": False,
        "promotion_authority": False,
        "live_action_authority": False,
    })
    report = seal_document({
        "schema_version": "stage6_notebook08_validation_report_v1",
        "status": "PASS_CALIBRATION_SELECTIVE_ABSTENTION_AND_LEGAL_REPLAY",
        "preflop_behavior": preflop_report,
        "postflop_behavior": postflop_report,
        "conditional_size": size_replay,
        "successor_holdout_precommit": successor_holdout,
        "profile_runtime_lookup_enabled": True,
        "profile_runtime_adjustment_count": 0,
        "review_runtime_concept_count": len(eligible_review_concepts),
        "review_authority_scope": "review_only",
        "reference_runtime_abstention_required": True,
        "illegal_postprojection_count": 0,
        "unknown_legal_mask_or_size_bounds": "abstain",
        "per_table_size_metrics_status": "PENDING_SUCCESSOR_NOTEBOOK09_EVALUATION",
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    result = _publish(stage_drive_root, freeze, report, worker_report)
    print(json.dumps({"event": "stage6_notebook08_complete", **result}, sort_keys=True, indent=2), flush=True)
    return result
