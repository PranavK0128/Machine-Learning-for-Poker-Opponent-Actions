"""Aggregate honest Stage 6 retrospective nested out-of-fold predictions.

The 55 search notebooks each seal one layer/outer-fold prediction checkpoint.
This module validates those checkpoints, reconstructs only fold-local baselines
from the corresponding inner fit partition, and applies the already-frozen
Stage 6 qualification gates.  Outer-test results never select a candidate,
feature, support rule, hyperparameter, iteration, or temperature.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from stage6_retrospective_nested_search import (
    ACTIONS,
    ADDITIONAL_LAYERS,
    BEHAVIOR_LAYERS,
    EXPANDED_LAYERS,
    FOLD_RUN_ID,
    LAYER_FAMILIES,
    _fold,
    _bind_feature_release_v2,
    _load_dataset,
    _support_mask,
    canonical_json_bytes,
    load_fold_freeze,
    seal_document,
    validate_sealed_document,
)


ALL_LAYERS = (*EXPANDED_LAYERS, *BEHAVIOR_LAYERS, *ADDITIONAL_LAYERS)
HELD_PARTIAL_LAYER = "hero_opponent_interaction"
HELD_PARTIAL_FOLD_ID = "outer-fold-04"
HELD_PARTIAL_FOLD_INDEX = 4
EXPECTED_COMPLETE_FOLD_RESULTS = 54
EXPECTED_PARTIAL_FOLD_RESULTS = 1
EXPECTED_EVALUATED_LAYERS = 10
MINIMUM_GAME_CLUSTERS = 20
FAMILY_ALPHA = 0.05
REQUIRED_BEHAVIOR_BASELINES = (
    "class_frequency",
    "current_qualified_artifact",
    "deterministic_chart_or_heuristic",
    "legal_random_or_uniform_where_semantically_valid",
)
CONTRACT_NAME = "stage6_retrospective_nested_qualification_contract_v1.json"
RESULT_MANIFEST_NAME = "retrospective-nested-aggregate-manifest.json"
R1_PROJECTION_COUNTER_SWAP_IMPLEMENTATION_SHA256 = (
    "080c75bebfc12f4ad42feea00b1889cfed9968b05a09b79e5dc30bdf1dead983"
)
R2_CORRECTED_PROJECTION_COUNTER_IMPLEMENTATION_SHA256 = (
    "7340c6891cc6ff6f3f9e3ca90d0abc0a5ad7d8a4a3c9a12b590f38602e7c20fb"
)
R3_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256 = (
    "0c4780afbdb6543ffd13721daa259f8220dadbe10861de60f9685699b6786325"
)
R4_EXACT_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256 = (
    "0d60299b8c47a2bd73942d79affcd1b0899d8248fb54dead3d8819a9b9f0b634"
)
R5_RUNTIME_INDEPENDENT_RESUME_IMPLEMENTATION_SHA256 = (
    "836571872b3f3740fc615792b58b5f78141d94d65b5768aa6583460b4113847c"
)
R6_PARALLEL_EXPANDED_FINAL_IMPLEMENTATION_SHA256 = (
    "9ef408c9f2289989040abb94b5b2a29a0bab0e9638fdaa717c41c14062040dae"
)
APPROVED_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256 = (
    "01f0364b00538d841c19e1a62fea68769c6f57a5582b8dbea36adf75940d6996"
)
R1_LEGACY_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256 = (
    "b4bef16829c54ab5a6ec1d2752cff32f294fd2abf55e27e9bb24d7ca0d3eec9b"
)
COMPATIBLE_SEARCH_IMPLEMENTATION_SHA256 = frozenset({
    R1_PROJECTION_COUNTER_SWAP_IMPLEMENTATION_SHA256,
    R2_CORRECTED_PROJECTION_COUNTER_IMPLEMENTATION_SHA256,
    R3_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256,
    R4_EXACT_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256,
    R5_RUNTIME_INDEPENDENT_RESUME_IMPLEMENTATION_SHA256,
    R6_PARALLEL_EXPANDED_FINAL_IMPLEMENTATION_SHA256,
})
CORRECTED_PROJECTION_COUNTER_SEMANTICS = (
    "raw_illegal_argmax_diagnostic_separate_from_true_zero_mass_fallback"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_zip_bytes(members: Sequence[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return stream.getvalue()


def _load_contract(stage_drive_root: Path) -> dict[str, Any]:
    path = stage_drive_root / "contracts" / CONTRACT_NAME
    if not path.is_file():
        # Colab input packages keep immutable contracts below their extracted
        # source root; the notebook supplies this path through PYTHONPATH.
        path = Path(__file__).resolve().parents[1] / "contracts" / CONTRACT_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_sealed_document(document)
    if (
        document.get("status") != "USER_APPROVED_RETROSPECTIVE_NESTED_QUALIFICATION"
        or document.get("authority_class") != "retrospective_nested_out_of_fold"
        or document.get("fresh_or_prospective_claim") is not False
        or document.get("maximum_outer_fold_count") != 5
        or document.get("minimum_independent_game_clusters_per_claim")
        != MINIMUM_GAME_CLUSTERS
        or document.get("existing_effect_legality_support_and_multiplicity_gates_unchanged")
        is not True
        or document.get("action_timing_layer_included") is not False
    ):
        raise RuntimeError("retrospective nested qualification contract changed")
    return document


def _validate_probabilities(
    probabilities: np.ndarray, legal: np.ndarray, *, label: str
) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.shape != legal.shape
        or not np.isfinite(values).all()
        or (values < -1e-12).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=2e-6)
        or (values[~legal] > 2e-7).any()
    ):
        raise RuntimeError(f"{label} probabilities violate the visible legal simplex")


def _project_fold_priors(
    priors: np.ndarray, legal: np.ndarray
) -> tuple[np.ndarray, int, int]:
    from stage6_expanded_tournament import _project_legal

    raw = np.broadcast_to(np.asarray(priors, dtype=float), legal.shape).copy()
    projected, raw_illegal, fallback = _project_legal(raw, legal, priors)
    _validate_probabilities(projected, legal, label="fold-local class-frequency")
    return (
        np.asarray(projected, dtype=np.float64),
        int(fallback),
        int(raw_illegal),
    )


def _projection_counter_mode(
    implementation_sha256: str,
    counter_semantics_present: bool,
    counter_semantics: Any,
) -> str:
    if implementation_sha256 == R1_PROJECTION_COUNTER_SWAP_IMPLEMENTATION_SHA256:
        if counter_semantics_present:
            raise RuntimeError("R1 nested projection counter semantics changed")
        return "r1_swapped_diagnostic_counter"
    if implementation_sha256 in {
        R2_CORRECTED_PROJECTION_COUNTER_IMPLEMENTATION_SHA256,
        R3_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256,
        R4_EXACT_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256,
        R5_RUNTIME_INDEPENDENT_RESUME_IMPLEMENTATION_SHA256,
        R6_PARALLEL_EXPANDED_FINAL_IMPLEMENTATION_SHA256,
    }:
        if (
            not counter_semantics_present
            or counter_semantics != CORRECTED_PROJECTION_COUNTER_SEMANTICS
        ):
            raise RuntimeError("R2 nested projection counter semantics changed")
        return "r2_corrected_counter"
    raise RuntimeError("unapproved nested search implementation")


def _approved_search_implementation_sha256(
    document: Mapping[str, Any],
) -> str | None:
    """Return the exact approved search bytes for legacy or resume manifests."""
    run_binding = document.get("run_binding", {})
    legacy_members = run_binding.get("implementation_members", {})
    legacy_sha256 = legacy_members.get(
        "stage6_retrospective_nested_search.py"
    )
    legacy_additional_sha256_by_search = {
        R1_PROJECTION_COUNTER_SWAP_IMPLEMENTATION_SHA256: (
            R1_LEGACY_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256
        ),
        R3_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256: (
            APPROVED_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256
        ),
        R4_EXACT_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256: (
            APPROVED_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256
        ),
    }
    if (
        set(legacy_members) == {
            "stage6_retrospective_nested_search.py",
            "stage6_retrospective_nested_additional_source.py",
        }
        and legacy_sha256 in legacy_additional_sha256_by_search
        and legacy_members.get(
            "stage6_retrospective_nested_additional_source.py"
        )
        == legacy_additional_sha256_by_search[legacy_sha256]
    ):
        return str(legacy_sha256)

    execution_members = document.get("execution_implementation_members", {})
    execution_sha256 = execution_members.get(
        "stage6_retrospective_nested_search.py"
    )
    if (
        run_binding.get("schema")
        == "stage6_retrospective_nested_layer_fold_search_v2"
        and run_binding.get("science_compatibility_sha256")
        == R4_EXACT_INVALID_CANDIDATE_REJECTION_IMPLEMENTATION_SHA256
        and document.get("runtime_independent_checkpoint_resume") is True
        and execution_sha256 in {
            R5_RUNTIME_INDEPENDENT_RESUME_IMPLEMENTATION_SHA256,
            R6_PARALLEL_EXPANDED_FINAL_IMPLEMENTATION_SHA256,
        }
        and set(execution_members) == {
            "stage6_retrospective_nested_search.py",
            "stage6_retrospective_nested_additional_source.py",
        }
        and execution_members.get(
            "stage6_retrospective_nested_additional_source.py"
        )
        == APPROVED_ADDITIONAL_SOURCE_IMPLEMENTATION_SHA256
    ):
        return str(execution_sha256)
    return None


def _read_outer_checkpoint(
    path: Path, manifest: Mapping[str, Any], layer: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    identity = manifest["outer_checkpoint_file"]
    raw = path.read_bytes()
    if (
        path.name != identity.get("name")
        or len(raw) != int(identity.get("bytes", -1))
        or hashlib.sha256(raw).hexdigest() != identity.get("sha256")
    ):
        raise RuntimeError(f"{layer} outer checkpoint identity changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != [
            "outer-test-predictions.npz",
            "outer-test-report.json",
        ]:
            raise RuntimeError(f"{layer} outer checkpoint inventory changed")
        prediction_raw = archive.read("outer-test-predictions.npz")
        report = json.loads(archive.read("outer-test-report.json").decode("utf-8"))
    validate_sealed_document(report)
    prediction_identity = report.get("prediction_file", {})
    if (
        report.get("schema_version")
        != "stage6_retrospective_nested_outer_test_checkpoint_v1"
        or report.get("run_id") != manifest.get("run_id")
        or report.get("layer") != layer
        or report.get("selected_checkpoint_self_sha256")
        != manifest.get("selected_checkpoint_self_sha256")
        or report.get("selected_configuration_sha256")
        != manifest.get("selected_configuration_sha256")
        or report.get("selected_best_iteration")
        != manifest.get("selected_best_iteration")
        or report.get("qualification_authority") is not False
        or prediction_identity.get("name") != "outer-test-predictions.npz"
        or prediction_identity.get("bytes") != len(prediction_raw)
        or prediction_identity.get("sha256")
        != hashlib.sha256(prediction_raw).hexdigest()
    ):
        raise RuntimeError(f"{layer} outer checkpoint authority changed")
    required = [
        "row_ids",
        "game_ids",
        "hand_ids",
        "target",
        "legal",
        "support_mask",
        "candidate_probabilities",
    ]
    if LAYER_FAMILIES[layer] == "expanded":
        required.extend([
            "matched_control_probabilities",
            "served_candidate_probabilities",
        ])
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(prediction_raw), allow_pickle=False) as payload:
        if payload.files != required:
            raise RuntimeError(f"{layer} outer prediction inventory changed")
        for name in payload.files:
            arrays[name] = np.asarray(payload[name]).copy()
    row_count = len(arrays["target"])
    if any(len(values) != row_count for values in arrays.values()):
        raise RuntimeError(f"{layer} outer prediction row counts changed")
    receipt = {
        "name": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "report_self_sha256": report["self_sha256"],
        "prediction_bytes": len(prediction_raw),
        "prediction_sha256": hashlib.sha256(prediction_raw).hexdigest(),
    }
    return arrays, report, receipt


def _completed_fold_manifest(
    stage_drive_root: Path,
    layer: str,
    assignment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, list[str]]:
    family = LAYER_FAMILIES[layer]
    root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "searches"
        / str(assignment["fold_id"])
        / family
        / layer
    )
    candidates: list[tuple[str, dict[str, Any], Path]] = []
    for path in sorted(root.glob("*/nested-search-result-manifest.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_sealed_document(document)
        if (
            document.get("status") == "PASS_RETROSPECTIVE_NESTED_LAYER_FOLD_SEARCH"
            and document.get("fold_run_id") == FOLD_RUN_ID
            and document.get("fold_id") == assignment.get("fold_id")
            and document.get("fold_index") == assignment.get("outer_fold_index")
            and document.get("fold_assignment_self_sha256")
            == assignment.get("self_sha256")
            and document.get("layer") == layer
            and document.get("family") == family
            and document.get("candidate_count") == 256
            and document.get("candidate_evaluations_per_fold") == 384
            and document.get("fresh_or_prospective_claim") is False
            and document.get("action_timing_layer_included") is False
            and document.get("qualification_authority") is False
            and _approved_search_implementation_sha256(document) is not None
        ):
            candidates.append((str(document["run_id"]), document, path))
    if not candidates:
        raise RuntimeError(
            f"{layer} {assignment['fold_id']} has no completed nested search result"
        )
    candidates.sort(key=lambda row: row[0])
    selected = candidates[0]
    return selected[1], selected[2], [row[0] for row in candidates]


def _partial_fold_manifest(
    stage_drive_root: Path,
    layer: str,
    assignment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, list[str]]:
    """Load only the explicitly authorized N35 383/384 partial closeout."""
    import stage6_retrospective_nested_partial_finalize as partial_finalize

    if (
        layer != HELD_PARTIAL_LAYER
        or assignment.get("fold_id") != HELD_PARTIAL_FOLD_ID
        or assignment.get("outer_fold_index") != HELD_PARTIAL_FOLD_INDEX
    ):
        raise RuntimeError("partial nested result is not allowed for this layer/fold")
    family = LAYER_FAMILIES[layer]
    root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "searches"
        / str(assignment["fold_id"])
        / family
        / layer
    )
    candidates: list[tuple[str, dict[str, Any], Path]] = []
    expected_implementation_sha256 = _sha256_file(
        Path(partial_finalize.__file__).resolve()
    )
    for path in sorted(root.glob("*/nested-search-partial-result-manifest.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_sealed_document(document)
        if (
            document.get("schema_version")
            == "stage6_retrospective_nested_partial_fold_result_manifest_v1"
            and document.get("status")
            == "HELD_INCOMPLETE_RETROSPECTIVE_NESTED_LAYER_FOLD_SEARCH"
            and document.get("fold_run_id") == FOLD_RUN_ID
            and document.get("fold_id") == assignment.get("fold_id")
            and document.get("fold_index") == assignment.get("outer_fold_index")
            and document.get("fold_assignment_self_sha256")
            == assignment.get("self_sha256")
            and document.get("layer") == layer
            and document.get("family") == family
            and document.get("candidate_count") == 256
            and document.get("candidate_evaluations_per_fold") == 383
            and document.get("expected_candidate_evaluations_per_fold") == 384
            and document.get("completed_stage_candidate_counts") == [256, 96, 31]
            and document.get("expected_stage_candidate_counts") == [256, 96, 32]
            and document.get("missing_candidate_count") == 1
            and document.get("missing_candidate", {}).get("candidate_id")
            == "catboost-012-1eda5f47d8c2"
            and document.get("scores_fabricated_for_missing_candidate") is False
            and document.get("outer_test_checkpoint_created") is False
            and document.get("outer_test_scored") is False
            and document.get("fresh_or_prospective_claim") is False
            and document.get("action_timing_layer_included") is False
            and document.get("qualification_authority") is False
            and document.get("partial_finalizer_implementation_sha256")
            == expected_implementation_sha256
        ):
            candidates.append((str(document["run_id"]), document, path))
    if not candidates:
        raise RuntimeError(
            "hero_opponent_interaction outer-fold-04 has no valid partial closeout"
        )
    candidates.sort(key=lambda row: row[0])
    selected = candidates[0]
    return selected[1], selected[2], [row[0] for row in candidates]


def _held_partial_expanded_claim(
    stage_drive_root: Path,
    layer: str,
    assignments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate four complete folds plus the sole held N35 partial manifest."""
    if layer != HELD_PARTIAL_LAYER:
        raise RuntimeError("held partial claim is limited to hero interaction")
    receipts: list[dict[str, Any]] = []
    complete_count = 0
    partial_count = 0
    partial_manifest: dict[str, Any] | None = None
    for assignment in assignments["folds"]:
        if assignment.get("fold_id") == HELD_PARTIAL_FOLD_ID:
            manifest, path, alternatives = _partial_fold_manifest(
                stage_drive_root, layer, assignment
            )
            raw = path.read_bytes()
            partial_count += 1
            partial_manifest = manifest
            receipts.append({
                "fold_id": assignment["fold_id"],
                "fold_index": assignment["outer_fold_index"],
                "run_id": manifest["run_id"],
                "result_class": "partial_held_no_outer_prediction",
                "deterministic_completed_run_selection": (
                    "lexicographically_smallest_run_id"
                ),
                "eligible_run_ids": alternatives,
                "manifest_relative_path": path.relative_to(
                    stage_drive_root
                ).as_posix(),
                "manifest_bytes": len(raw),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "manifest_self_sha256": manifest["self_sha256"],
                "candidate_evaluations_per_fold": 383,
                "expected_candidate_evaluations_per_fold": 384,
                "outer_test_scored": False,
                "qualification_authority": False,
            })
            continue
        manifest, path, alternatives = _completed_fold_manifest(
            stage_drive_root, layer, assignment
        )
        checkpoint_path = path.parent / manifest["outer_checkpoint_file"]["name"]
        _, report, checkpoint_receipt = _read_outer_checkpoint(
            checkpoint_path, manifest, layer
        )
        raw = path.read_bytes()
        complete_count += 1
        receipts.append({
            "fold_id": assignment["fold_id"],
            "fold_index": assignment["outer_fold_index"],
            "run_id": manifest["run_id"],
            "result_class": "complete_outer_prediction_not_aggregated",
            "deterministic_completed_run_selection": (
                "lexicographically_smallest_run_id"
            ),
            "eligible_run_ids": alternatives,
            "manifest_relative_path": path.relative_to(stage_drive_root).as_posix(),
            "manifest_bytes": len(raw),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_self_sha256": manifest["self_sha256"],
            "outer_checkpoint": checkpoint_receipt,
            "prediction_report": report["prediction_report"],
            "candidate_evaluations_per_fold": 384,
            "outer_test_scored": True,
            "qualification_authority": False,
        })
    if complete_count != 4 or partial_count != 1 or partial_manifest is None:
        raise RuntimeError("hero interaction partial fold conservation failed")
    claim = {
        "layer": layer,
        "status": "HELD_INCOMPLETE_RETROSPECTIVE_NESTED_LAYER",
        "evaluation_complete": False,
        "completed_outer_fold_count": complete_count,
        "partial_outer_fold_count": partial_count,
        "expected_outer_fold_count": 5,
        "completed_candidate_evaluations_in_partial_fold": 383,
        "expected_candidate_evaluations_in_partial_fold": 384,
        "missing_candidate_id": partial_manifest["missing_candidate"][
            "candidate_id"
        ],
        "hold_reason": partial_manifest["abandonment_reason"],
        "scores_fabricated_for_missing_candidate": False,
        "outer_predictions_aggregated": False,
        "qualification_authority": False,
        "qualified": False,
        "inference": {
            "paired_sign_flip_two_sided_p_value": 1.0,
            "independent_cluster_gate_met": False,
            "supported_game_cluster_count": 0,
            "confidence_interval_95": [None, None],
            "mean_candidate_minus_control_log_loss": None,
            "placeholder_used_only_for_conservative_family_multiplicity": True,
        },
        "fold_receipts": receipts,
    }
    return claim, receipts


def _load_layer_oof(
    stage_drive_root: Path,
    layer: str,
    assignments: Mapping[str, Any],
) -> dict[str, Any]:
    base = _load_dataset(stage_drive_root, layer)
    chunks: dict[str, list[np.ndarray]] = {}
    fold_receipts: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    prior_fallback_count = 0
    prior_raw_illegal_argmax_count = 0
    candidate_fallback_count = 0
    candidate_served_illegal_count = 0
    candidate_raw_illegal_argmax_count = 0
    matched_control_fallback_count = 0
    matched_control_served_illegal_count = 0
    matched_control_raw_illegal_argmax_count = 0
    for assignment in assignments["folds"]:
        manifest, manifest_path, _ = _completed_fold_manifest(
            stage_drive_root, layer, assignment
        )
        checkpoint_path = manifest_path.parent / manifest["outer_checkpoint_file"]["name"]
        arrays, report, checkpoint_receipt = _read_outer_checkpoint(
            checkpoint_path, manifest, layer
        )
        folded = _fold(base, assignment)
        roles = folded["frame"]["nested_role"].astype(str).to_numpy()
        fit = np.flatnonzero(np.isin(roles, ["train", "validation"]))
        test = np.flatnonzero(roles == "test")
        expected_rows = folded["frame"].iloc[test]
        row_ids = arrays["row_ids"].astype(str)
        if (
            row_ids.tolist() != expected_rows["row_id"].astype(str).tolist()
            or arrays["game_ids"].astype(str).tolist()
            != expected_rows["game_id"].astype(str).tolist()
            or arrays["hand_ids"].astype(str).tolist()
            != expected_rows["hand_id"].astype(str).tolist()
            or not np.array_equal(
                arrays["target"].astype(np.int8),
                np.asarray(folded["target"], dtype=np.int8)[test],
            )
            or not np.array_equal(
                arrays["legal"].astype(bool),
                np.asarray(folded["legal"], dtype=bool)[test],
            )
            or not np.array_equal(
                arrays["support_mask"].astype(bool),
                _support_mask(folded, layer)[test],
            )
        ):
            raise RuntimeError(f"{layer} {assignment['fold_id']} outer rows changed")
        if seen_rows.intersection(row_ids.tolist()):
            raise RuntimeError(f"{layer} outer rows are reused across folds")
        seen_rows.update(row_ids.tolist())
        legal = arrays["legal"].astype(bool)
        target = arrays["target"].astype(np.int8)
        if not legal[np.arange(len(target)), target.astype(int)].all():
            raise RuntimeError(f"{layer} outer target is illegal")
        _validate_probabilities(
            arrays["candidate_probabilities"], legal, label=f"{layer} candidate"
        )
        counts = np.bincount(
            np.asarray(folded["target"], dtype=np.int8)[fit], minlength=len(ACTIONS)
        ).astype(float)
        priors = (counts + 1.0) / (counts.sum() + len(ACTIONS))
        fold_frequency, fallback, raw_illegal = _project_fold_priors(priors, legal)
        prior_fallback_count += fallback
        prior_raw_illegal_argmax_count += raw_illegal
        arrays["fold_class_frequency"] = fold_frequency
        prediction_report = report["prediction_report"]
        candidate_projection = prediction_report["candidate_projection_counts"]
        implementation_sha256 = _approved_search_implementation_sha256(manifest)
        if implementation_sha256 is None:
            raise RuntimeError("unapproved nested search implementation")
        counter_semantics = prediction_report.get("projection_counter_semantics")
        counter_mode = _projection_counter_mode(
            implementation_sha256,
            "projection_counter_semantics" in prediction_report,
            counter_semantics,
        )
        if counter_mode == "r2_corrected_counter":
            candidate_fallback_count += int(
                candidate_projection["legal_projection_fallback_count"]
            )
            candidate_raw_illegal_argmax_count += int(
                candidate_projection["raw_preprojection_illegal_argmax_count"]
            )
        else:
            # R1 passed the second return value from _project_legal through as
            # the fallback counter.  That value is the diagnostic raw illegal
            # argmax count.  A real fallback was the third return value and R1
            # raised before publishing, so every completed R1 checkpoint has
            # exactly zero true fallbacks.
            candidate_raw_illegal_argmax_count += int(
                candidate_projection["legal_projection_fallback_count"]
            )
        candidate_served_illegal_count += int(
            candidate_projection["served_illegal_probability_mass_count"]
        )
        if LAYER_FAMILIES[layer] == "expanded":
            _validate_probabilities(
                arrays["matched_control_probabilities"],
                legal,
                label=f"{layer} matched control",
            )
            _validate_probabilities(
                arrays["served_candidate_probabilities"],
                legal,
                label=f"{layer} served candidate",
            )
            support = arrays["support_mask"].astype(bool)
            if (
                not np.array_equal(
                    arrays["served_candidate_probabilities"][support],
                    arrays["candidate_probabilities"][support],
                )
                or not np.array_equal(
                    arrays["served_candidate_probabilities"][~support],
                    arrays["matched_control_probabilities"][~support],
                )
                or report["prediction_report"].get(
                    "below_support_served_exact_matched_control"
                )
                is not True
            ):
                raise RuntimeError(f"{layer} support fallback changed")
            matched_projection = prediction_report[
                "matched_control_projection_counts"
            ]
            if counter_mode == "r2_corrected_counter":
                matched_control_fallback_count += int(
                    matched_projection["legal_projection_fallback_count"]
                )
                matched_control_raw_illegal_argmax_count += int(
                    matched_projection[
                        "raw_preprojection_illegal_argmax_count"
                    ]
                )
            else:
                matched_control_raw_illegal_argmax_count += int(
                    matched_projection["legal_projection_fallback_count"]
                )
            matched_control_served_illegal_count += int(
                matched_projection["served_illegal_probability_mass_count"]
            )
        for name, values in arrays.items():
            chunks.setdefault(name, []).append(values)
        manifest_raw = manifest_path.read_bytes()
        fold_receipts.append({
            "fold_id": assignment["fold_id"],
            "fold_index": assignment["outer_fold_index"],
            "run_id": manifest["run_id"],
            "deterministic_completed_run_selection": "lexicographically_smallest_run_id",
            "selected_candidate_id": manifest["selected_candidate_id"],
            "selected_configuration_sha256": manifest[
                "selected_configuration_sha256"
            ],
            "selected_family": manifest["selected_family"],
            "selected_best_iteration": manifest["selected_best_iteration"],
            "manifest_relative_path": manifest_path.relative_to(
                stage_drive_root
            ).as_posix(),
            "manifest_bytes": len(manifest_raw),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_self_sha256": manifest["self_sha256"],
            "outer_checkpoint": checkpoint_receipt,
            "prediction_report": prediction_report,
            "fit_prior_target_access_roles": ["train", "validation"],
            "outer_test_used_for_prior_estimation": False,
            "outer_test_row_count": len(target),
            "outer_test_game_count": len(set(arrays["game_ids"].astype(str))),
        })
        print(json.dumps({
            "event": "stage6_retrospective_nested_aggregate_fold_loaded",
            "layer": layer,
            "fold_id": assignment["fold_id"],
            "completed": len(fold_receipts),
            "denominator": 5,
            "outer_rows": len(target),
        }, sort_keys=True), flush=True)
        del folded, arrays
        gc.collect()
    combined = {
        name: np.concatenate(values, axis=0) for name, values in chunks.items()
    }
    if len(set(combined["row_ids"].astype(str))) != len(combined["row_ids"]):
        raise RuntimeError(f"{layer} aggregate duplicates outer row identity")
    return {
        "arrays": combined,
        "fold_receipts": fold_receipts,
        "fold_prior_projection_fallback_count": prior_fallback_count,
        "fold_prior_raw_illegal_argmax_count": (
            prior_raw_illegal_argmax_count
        ),
        "candidate_projection_fallback_count": candidate_fallback_count,
        "candidate_served_illegal_probability_mass_count": (
            candidate_served_illegal_count
        ),
        "candidate_raw_illegal_argmax_count": (
            candidate_raw_illegal_argmax_count
        ),
        "matched_control_projection_fallback_count": (
            matched_control_fallback_count
        ),
        "matched_control_served_illegal_probability_mass_count": (
            matched_control_served_illegal_count
        ),
        "matched_control_raw_illegal_argmax_count": (
            matched_control_raw_illegal_argmax_count
        ),
    }


def _normalized_metric_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Remove only bounded float32 serialization drift before metric scoring."""
    values = np.asarray(probabilities, dtype=np.float64).copy()
    if (
        values.ndim != 2
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
    ):
        raise RuntimeError("metric probabilities must be a finite nonnegative matrix")
    row_sums = values.sum(axis=1)
    if (
        (row_sums <= 0.0).any()
        or not np.allclose(row_sums, 1.0, rtol=0.0, atol=2e-6)
    ):
        raise RuntimeError("metric probabilities exceed bounded serialization drift")
    values /= row_sums[:, None]
    residual = 1.0 - values.sum(axis=1)
    values[np.arange(len(values)), values.argmax(axis=1)] += residual
    return values


def _metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from stage6_metrics import multiclass_metric_report

    truth = [ACTIONS[int(value)] for value in np.asarray(target, dtype=int)]
    return multiclass_metric_report(
        truth, _normalized_metric_probabilities(probabilities), ACTIONS
    )


def _losses(target: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=int)
    normalized = _normalized_metric_probabilities(probabilities)
    return -np.log(np.clip(normalized[np.arange(len(y)), y], 1e-15, 1.0))


def _expanded_claim(layer: str, oof: Mapping[str, Any]) -> dict[str, Any]:
    from stage6_expanded_adjudication import _paired_cluster_inference

    arrays = oof["arrays"]
    support = arrays["support_mask"].astype(bool)
    positions = np.flatnonzero(support)
    if not len(positions):
        raise RuntimeError(f"{layer} has no supported outer rows")
    candidate = arrays["candidate_probabilities"].astype(float)[positions]
    control = arrays["matched_control_probabilities"].astype(float)[positions]
    target = arrays["target"].astype(int)[positions]
    candidate_metrics = _metrics(target, candidate)
    control_metrics = _metrics(target, control)
    inference = _paired_cluster_inference(
        _losses(target, candidate) - _losses(target, control),
        arrays["game_ids"].astype(str)[positions].tolist(),
        int(hashlib.sha256(f"retrospective-nested:{layer}".encode()).hexdigest()[:8], 16),
        MINIMUM_GAME_CLUSTERS,
    )
    return {
        "layer": layer,
        "oof_row_count": len(arrays["target"]),
        "supported_oof_row_count": len(positions),
        "supported_oof_game_cluster_count": len(
            set(arrays["game_ids"].astype(str)[positions])
        ),
        "selected_metrics": candidate_metrics,
        "matched_control_metrics": control_metrics,
        "inference": inference,
        "below_support_oof_row_count": int((~support).sum()),
        "below_support_served_exact_matched_control": bool(np.array_equal(
            arrays["served_candidate_probabilities"][~support],
            arrays["matched_control_probabilities"][~support],
        )),
        "served_illegal_probability_mass_count": int(
            oof["candidate_served_illegal_probability_mass_count"]
            + oof["matched_control_served_illegal_probability_mass_count"]
        ),
        "zero_legal_mass_fallback_count": int(
            oof["candidate_projection_fallback_count"]
            + oof["matched_control_projection_fallback_count"]
            + oof["fold_prior_projection_fallback_count"]
        ),
        "raw_preprojection_illegal_argmax_count": int(
            oof["candidate_raw_illegal_argmax_count"]
            + oof["matched_control_raw_illegal_argmax_count"]
            + oof["fold_prior_raw_illegal_argmax_count"]
        ),
        "raw_preprojection_illegal_argmax_is_diagnostic_only": True,
        "fold_receipts": oof["fold_receipts"],
    }


def _holm_from_inference(
    inference: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    from stage6_current_corpus_adjudication import holm_step_down

    return holm_step_down({
        key: float(value["paired_sign_flip_two_sided_p_value"])
        for key, value in inference.items()
    }, alpha=FAMILY_ALPHA)


def _behavior_frames(
    stage_drive_root: Path,
    row_ids: Sequence[str],
    all_outer_games: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from stage6_current_corpus_adjudication import _behavior_base_frame

    source = _load_dataset(stage_drive_root, "within_hand_line_surprise")[
        "feature_frame"
    ]
    source = _behavior_base_frame(source)
    indexed = source.copy()
    indexed["row_id"] = indexed["row_id"].astype(str)
    if indexed["row_id"].duplicated().any():
        raise RuntimeError("behavior aggregate source duplicates row_id")
    indexed = indexed.set_index("row_id", drop=False)
    missing = sorted(set(row_ids) - set(indexed.index))
    if missing:
        raise RuntimeError(f"behavior aggregate source is missing row ids: {missing[:3]}")
    ordered = indexed.loc[list(row_ids)].reset_index(drop=True)
    full_outer = source.loc[
        source["game_id"].astype(str).isin(all_outer_games)
    ].reset_index(drop=True)
    return ordered, full_outer


def _behavior_family(
    stage_drive_root: Path,
    loaded: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from stage6_current_corpus_adjudication import holm_step_down
    from stage6_notebook09_runtime import (
        _deterministic_public_heuristic,
        _incumbent_comparison,
        _legal_uniform,
    )
    from stage6_notebook11_runtime import (
        _extract_notebook09_package,
        _hand_sizes,
        _losses as native_losses,
        _map_six_action_baseline,
        _prepare_incumbent_context,
    )
    from stage6_notebooks02_07_runtime import materialization_root
    from stage6_metrics import multiclass_metric_report

    row_ids = [
        str(value)
        for layer in BEHAVIOR_LAYERS
        for value in loaded[layer]["arrays"]["row_ids"]
    ]
    all_games = {
        str(value)
        for layer in BEHAVIOR_LAYERS
        for value in loaded[layer]["arrays"]["game_ids"]
    }
    ordered_all, full_outer = _behavior_frames(
        stage_drive_root, row_ids, all_games
    )
    ordered_index = ordered_all.set_index(
        ordered_all["row_id"].astype(str), drop=False
    )
    local = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    token = hashlib.sha256("\n".join(sorted(all_games)).encode()).hexdigest()[:16]
    notebook09_root = local / f"stage6-nested-incumbent-{token}"
    context_root = local / f"stage6-nested-context-{token}"
    if notebook09_root.exists():
        shutil.rmtree(notebook09_root)
    if context_root.exists():
        shutil.rmtree(context_root)
    reports: dict[str, Any] = {}
    inference: dict[str, Any] = {}
    try:
        notebook09_anchor = _extract_notebook09_package(
            stage_drive_root, notebook09_root
        )
        _prepare_incumbent_context(
            stage_drive_root,
            materialization_root(stage_drive_root),
            full_outer,
            notebook09_root,
            notebook09_anchor,
            context_root,
        )
        hand_sizes = _hand_sizes(full_outer)
        for layer in BEHAVIOR_LAYERS:
            scope = layer.removesuffix("_behavior")
            arrays = loaded[layer]["arrays"]
            frame = ordered_index.loc[
                arrays["row_ids"].astype(str).tolist()
            ].reset_index(drop=True)
            candidate = arrays["candidate_probabilities"].astype(float)
            legal = arrays["legal"].astype(bool)
            truth_six = [ACTIONS[int(value)] for value in arrays["target"]]
            incumbent = _incumbent_comparison(
                frame,
                candidate,
                legal,
                truth_six,
                scope,
                str(notebook09_root),
                hand_sizes,
                str(context_root),
            )
            positions = np.asarray(incumbent["positions"], dtype=int)
            labels = list(incumbent["labels"])
            truth = list(incumbent["truth"])
            matrices: dict[str, np.ndarray] = {
                "candidate": np.asarray(incumbent["candidate"], dtype=float),
                "current_qualified_artifact": np.asarray(
                    incumbent["incumbent"], dtype=float
                ),
            }
            for name, six in (
                ("class_frequency", arrays["fold_class_frequency"]),
                (
                    "deterministic_chart_or_heuristic",
                    _deterministic_public_heuristic(frame, legal),
                ),
                (
                    "legal_random_or_uniform_where_semantically_valid",
                    _legal_uniform(legal),
                ),
            ):
                mapped_labels, mapped_truth, mapped = _map_six_action_baseline(
                    scope, six, legal, truth_six, positions
                )
                if mapped_labels != labels or mapped_truth != truth:
                    raise RuntimeError(f"{layer} behavior baseline mapping changed")
                matrices[name] = mapped
            matrices = {
                name: _normalized_metric_probabilities(values)
                for name, values in matrices.items()
            }
            metrics = {
                name: multiclass_metric_report(truth, values, labels)
                for name, values in matrices.items()
            }
            candidate_losses = native_losses(truth, matrices["candidate"], labels)
            clusters = frame.iloc[positions]["game_id"].astype(str).tolist()
            layer_inference: dict[str, Any] = {}
            from stage6_expanded_adjudication import _paired_cluster_inference

            for comparator in REQUIRED_BEHAVIOR_BASELINES:
                key = f"{layer}:{comparator}"
                comparison = _paired_cluster_inference(
                    candidate_losses
                    - native_losses(truth, matrices[comparator], labels),
                    clusters,
                    int(hashlib.sha256(
                        f"retrospective-nested:{key}".encode()
                    ).hexdigest()[:8], 16),
                    MINIMUM_GAME_CLUSTERS,
                )
                inference[key] = comparison
                layer_inference[comparator] = comparison
            sizes = np.asarray([
                hand_sizes[f"{game}\x1f{hand}"]
                for game, hand in zip(
                    frame.iloc[positions]["game_id"].astype(str),
                    frame.iloc[positions]["hand_id"].astype(str),
                )
            ])
            truth_array = np.asarray(truth, dtype=object)
            per_table_size = {}
            for size in sorted(set(sizes.tolist())):
                mask = sizes == size
                per_table_size[str(int(size))] = {
                    name: multiclass_metric_report(
                        truth_array[mask].tolist(), values[mask], labels
                    )
                    for name, values in matrices.items()
                }
            reports[layer] = {
                "scope": layer,
                "oof_scope_row_count": len(frame),
                "native_domain_row_count": len(truth),
                "native_domain_game_cluster_count": len(set(clusters)),
                "metrics": metrics,
                "per_table_size_metrics": per_table_size,
                "inference": layer_inference,
                "illegal_postprojection_count": int(
                    loaded[layer][
                        "candidate_served_illegal_probability_mass_count"
                    ]
                    + loaded[layer]["candidate_projection_fallback_count"]
                ),
                "incumbent_illegal_preprojection_count": int(
                    incumbent["incumbent_illegal_preprojection"]
                ),
                "incumbent_zero_mass_fallback_count": int(
                    incumbent["incumbent_zero_mass_fallback"]
                ),
                "missing_or_defaulted_required_incumbent_input_count": int(
                    incumbent["missing_or_defaulted_required_input_count"]
                ),
                "comparison_label_space": incumbent["comparison_label_space"],
                "fold_receipts": loaded[layer]["fold_receipts"],
            }
            print(json.dumps({
                "event": "stage6_retrospective_nested_aggregate_layer_complete",
                "layer": layer,
                "oof_rows": len(frame),
                "game_clusters": len(set(clusters)),
            }, sort_keys=True), flush=True)
    finally:
        shutil.rmtree(notebook09_root, ignore_errors=True)
        shutil.rmtree(context_root, ignore_errors=True)
        gc.collect()
    holm = holm_step_down({
        key: value["paired_sign_flip_two_sided_p_value"]
        for key, value in inference.items()
    })
    qualification: dict[str, bool] = {}
    for layer in BEHAVIOR_LAYERS:
        comparison_gates = {}
        for comparator in REQUIRED_BEHAVIOR_BASELINES:
            key = f"{layer}:{comparator}"
            estimate = inference[key]
            comparison_gates[comparator] = bool(
                estimate["independent_cluster_gate_met"]
                and estimate["mean_candidate_minus_control_log_loss"] < 0.0
                and estimate["confidence_interval_95"][1] < 0.0
                and holm[key]["reject_no_improvement_null"]
                and reports[layer]["metrics"]["candidate"][
                    "multiclass_brier_score"
                ]
                <= reports[layer]["metrics"][comparator][
                    "multiclass_brier_score"
                ]
            )
        reports[layer]["preregistered_comparison_gates"] = comparison_gates
        qualification[layer] = bool(
            all(comparison_gates.values())
            and reports[layer]["illegal_postprojection_count"] == 0
            and reports[layer][
                "missing_or_defaulted_required_incumbent_input_count"
            ]
            == 0
        )
    return reports, {"inference": inference, "holm": holm}, qualification


def _additional_claim(layer: str, oof: Mapping[str, Any]) -> dict[str, Any]:
    from stage6_additional_exploitation_execution import (
        _game_cluster_mean_differences,
        _paired_sign_flip_pvalue,
    )

    arrays = oof["arrays"]
    support = arrays["support_mask"].astype(bool)
    positions = np.flatnonzero(support)
    candidate = arrays["candidate_probabilities"].astype(float)[positions]
    baseline = arrays["fold_class_frequency"].astype(float)[positions]
    target = arrays["target"].astype(int)[positions]
    candidate_metrics = _metrics(target, candidate) if len(positions) else None
    baseline_metrics = _metrics(target, baseline) if len(positions) else None
    if len(positions):
        differences = _game_cluster_mean_differences(
            _losses(target, candidate) - _losses(target, baseline),
            arrays["game_ids"].astype(str)[positions],
        )
    else:
        differences = np.asarray([], dtype=float)
    return {
        "layer": layer,
        "row_count": len(arrays["target"]),
        "supported_row_count": len(positions),
        "prediction_support_coverage": float(
            len(positions) / max(1, len(arrays["target"]))
        ),
        "candidate_metrics": candidate_metrics,
        "frozen_baseline_metrics": baseline_metrics,
        "candidate_log_loss_minus_baseline": (
            float(candidate_metrics["log_loss"] - baseline_metrics["log_loss"])
            if candidate_metrics is not None else float("inf")
        ),
        "candidate_brier_minus_baseline": (
            float(
                candidate_metrics["multiclass_brier_score"]
                - baseline_metrics["multiclass_brier_score"]
            )
            if candidate_metrics is not None else float("inf")
        ),
        "paired_sign_flip_raw_pvalue": _paired_sign_flip_pvalue(differences),
        "paired_sign_flip_inference_unit": "game_id_cluster_mean",
        "supported_game_cluster_count": len(differences),
        "served_illegal_probability_mass_count": int(
            oof["candidate_served_illegal_probability_mass_count"]
        ),
        "legal_projection_fallback_count": int(
            oof["candidate_projection_fallback_count"]
            +
            oof["fold_prior_projection_fallback_count"]
        ),
        "raw_preprojection_illegal_argmax_count": int(
            oof["candidate_raw_illegal_argmax_count"]
            + oof["fold_prior_raw_illegal_argmax_count"]
        ),
        "raw_preprojection_illegal_argmax_is_diagnostic_only": True,
        "fold_receipts": oof["fold_receipts"],
    }


def _aggregate_run_id(
    contract: Mapping[str, Any], receipts: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str:
    binding = {
        "schema": "stage6_retrospective_nested_oof_aggregate_run_v1",
        "contract_self_sha256": contract["self_sha256"],
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "fold_manifest_self_sha256": {
            layer: [row["manifest_self_sha256"] for row in receipts[layer]]
            for layer in ALL_LAYERS
        },
    }
    return hashlib.sha256(canonical_json_bytes(binding)).hexdigest()


def run_retrospective_nested_oof_aggregate(
    stage_drive_root: str | Path,
    *,
    requested_workers: str | int = "auto",
    accelerator: str = "CPU",
) -> dict[str, Any]:
    """Aggregate 54 complete folds and conservatively hold the sole partial fold."""
    from stage6_additional_exploitation_execution import _holm_adjust
    from stage6_drive_fuse_publication import create_or_verify_bytes

    stage_drive_root = Path(stage_drive_root).resolve()
    _bind_feature_release_v2()
    accelerator = str(accelerator).strip().upper()
    if accelerator not in {"CPU", "GPU", "TPU"}:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    raw_workers = str(requested_workers).strip().lower()
    if raw_workers != "auto" and not (
        raw_workers.isascii() and raw_workers.isdecimal() and int(raw_workers) > 0
    ):
        raise ValueError("REQUESTED_WORKERS must be auto or a positive integer")
    started = time.monotonic()
    contract = _load_contract(stage_drive_root)
    fold_freeze = load_fold_freeze(stage_drive_root)
    assignments = fold_freeze["assignments"]
    loaded: dict[str, dict[str, Any]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    completed_layers = 0

    expanded_claims: dict[str, Any] = {}
    expanded_inference: dict[str, Any] = {}
    for layer in EXPANDED_LAYERS:
        if layer == HELD_PARTIAL_LAYER:
            claim, layer_receipts = _held_partial_expanded_claim(
                stage_drive_root, layer, assignments
            )
            expanded_claims[layer] = claim
            expanded_inference[layer] = claim["inference"]
            receipts[layer] = layer_receipts
            print(json.dumps({
                "event": "stage6_retrospective_nested_aggregate_layer_held",
                "phase": "expanded",
                "layer": layer,
                "completed_outer_folds": 4,
                "partial_outer_folds": 1,
                "reason": claim["hold_reason"],
                "scores_fabricated": False,
            }, sort_keys=True), flush=True)
            continue
        oof = _load_layer_oof(stage_drive_root, layer, assignments)
        claim = _expanded_claim(layer, oof)
        expanded_claims[layer] = claim
        expanded_inference[layer] = claim["inference"]
        receipts[layer] = claim["fold_receipts"]
        completed_layers += 1
        print(json.dumps({
            "event": "stage6_retrospective_nested_aggregate_progress",
            "phase": "expanded",
            "layer": layer,
            "completed": completed_layers,
            "denominator": len(ALL_LAYERS),
        }, sort_keys=True), flush=True)
        del oof
        gc.collect()
    expanded_holm = _holm_from_inference(expanded_inference)
    expanded_qualification = {}
    for layer, claim in expanded_claims.items():
        estimate = claim["inference"]
        if claim.get("evaluation_complete") is False:
            qualified = False
        else:
            qualified = bool(
                estimate["independent_cluster_gate_met"]
                and estimate["mean_candidate_minus_control_log_loss"] < 0.0
                and estimate["confidence_interval_95"][1] < 0.0
                and expanded_holm[layer]["reject_no_improvement_null"]
                and claim["selected_metrics"]["multiclass_brier_score"]
                <= claim["matched_control_metrics"]["multiclass_brier_score"]
                and claim["zero_legal_mass_fallback_count"] == 0
                and claim["served_illegal_probability_mass_count"] == 0
                and claim["below_support_served_exact_matched_control"]
            )
        claim["holm"] = expanded_holm[layer]
        claim["qualified"] = qualified
        expanded_qualification[layer] = qualified

    for layer in BEHAVIOR_LAYERS:
        loaded[layer] = _load_layer_oof(stage_drive_root, layer, assignments)
        receipts[layer] = loaded[layer]["fold_receipts"]
    behavior_reports, behavior_statistics, behavior_qualification = _behavior_family(
        stage_drive_root, loaded
    )
    completed_layers += len(BEHAVIOR_LAYERS)
    print(json.dumps({
        "event": "stage6_retrospective_nested_aggregate_progress",
        "phase": "behavior",
        "completed": completed_layers,
        "denominator": len(ALL_LAYERS),
    }, sort_keys=True), flush=True)
    loaded.clear()
    gc.collect()

    additional_results = []
    additional_pvalues = {}
    for layer in ADDITIONAL_LAYERS:
        oof = _load_layer_oof(stage_drive_root, layer, assignments)
        result = _additional_claim(layer, oof)
        receipts[layer] = result["fold_receipts"]
        additional_results.append(result)
        additional_pvalues[layer] = result["paired_sign_flip_raw_pvalue"]
        completed_layers += 1
        print(json.dumps({
            "event": "stage6_retrospective_nested_aggregate_progress",
            "phase": "additional",
            "layer": layer,
            "completed": completed_layers,
            "denominator": len(ALL_LAYERS),
        }, sort_keys=True), flush=True)
        del oof
        gc.collect()
    additional_holm = _holm_adjust(additional_pvalues)
    from stage6_additional_exploitation_execution import PREREGISTERED_FRESH_GATES

    gates = PREREGISTERED_FRESH_GATES
    additional_qualification = {}
    for row in additional_results:
        layer = row["layer"]
        row["paired_sign_flip_holm_pvalue"] = additional_holm[layer]
        row["gates"] = {
            "minimum_rows": row["row_count"]
            >= int(gates["minimum_fresh_rows_per_layer"]),
            "minimum_supported_rows": row["supported_row_count"]
            >= int(gates["minimum_supported_fresh_rows_per_layer"]),
            "prediction_support": row["prediction_support_coverage"]
            >= float(gates["minimum_prediction_support_coverage"]),
            "minimum_independent_game_clusters": row[
                "supported_game_cluster_count"
            ]
            >= MINIMUM_GAME_CLUSTERS,
            "log_loss": row["candidate_log_loss_minus_baseline"] < 0.0,
            "brier": row["candidate_brier_minus_baseline"] <= 0.0,
            "holm_significance": additional_holm[layer]
            <= float(gates["paired_sign_flip_holm_familywise_alpha"]),
            "legality": row["served_illegal_probability_mass_count"] == 0
            and row["legal_projection_fallback_count"] == 0,
            "coverage": float(gates["minimum_legal_coverage"]) == 1.0,
        }
        row["qualified"] = bool(all(row["gates"].values()))
        additional_qualification[layer] = row["qualified"]

    qualification = {
        **expanded_qualification,
        **behavior_qualification,
        **additional_qualification,
    }
    qualified_layers = [layer for layer in ALL_LAYERS if qualification[layer]]
    run_id = _aggregate_run_id(contract, receipts)
    manifest = seal_document({
        "schema_version": "stage6_retrospective_nested_oof_aggregate_manifest_v1",
        "status": (
            "PASS_ALL_RETROSPECTIVE_NESTED_LAYER_GATES"
            if len(qualified_layers) == len(ALL_LAYERS)
            else "HOLD_ONE_OR_MORE_RETROSPECTIVE_NESTED_LAYER_GATES"
        ),
        "run_id": run_id,
        "authority_class": "retrospective_nested_out_of_fold",
        "fresh_or_prospective_claim": False,
        "contract_self_sha256": contract["self_sha256"],
        "fold_run_id": FOLD_RUN_ID,
        "fold_assignment_manifest_self_sha256": fold_freeze["manifest"][
            "self_sha256"
        ],
        "fold_assignments_self_sha256": assignments["self_sha256"],
        "outer_fold_count": 5,
        "outer_test_reuse_count_per_game_per_layer": 1,
        "outer_test_used_for_candidate_feature_support_hyperparameter_iteration_or_temperature_selection": False,
        "fold_local_class_frequency_priors_use_roles": ["train", "validation"],
        "action_timing_layer_included": False,
        "cfr_or_dcfr_dependency": False,
        "expanded": {
            "claims": expanded_claims,
            "holm_family_size": 2,
            "holm": expanded_holm,
            "qualification": expanded_qualification,
        },
        "behavior": {
            "reports": behavior_reports,
            **behavior_statistics,
            "required_baselines": list(REQUIRED_BEHAVIOR_BASELINES),
            "holm_family_size": 8,
            "qualification": behavior_qualification,
            "current_qualified_artifact_role": "fixed_preexisting_comparator_only",
        },
        "additional": {
            "layers": additional_results,
            "holm_family_size": 7,
            "holm_adjusted_pvalues": additional_holm,
            "preregistered_gates": gates,
            "qualification": additional_qualification,
        },
        "fold_result_conservation": {
            "expected_fold_result_count": 55,
            "complete_fold_result_count": EXPECTED_COMPLETE_FOLD_RESULTS,
            "partial_held_fold_result_count": EXPECTED_PARTIAL_FOLD_RESULTS,
            "evaluated_layer_count": EXPECTED_EVALUATED_LAYERS,
            "held_incomplete_layer_count": 1,
            "held_incomplete_layer": HELD_PARTIAL_LAYER,
            "held_incomplete_fold_id": HELD_PARTIAL_FOLD_ID,
            "missing_candidate_scores_fabricated": False,
        },
        "candidate_layer_count": len(ALL_LAYERS),
        "evaluated_layer_count": EXPECTED_EVALUATED_LAYERS,
        "qualified_layer_count": len(qualified_layers),
        "qualified_layers": qualified_layers,
        "held_layers": [layer for layer in ALL_LAYERS if layer not in qualified_layers],
        "assistant_installation_allowed_for_qualified_layers_after_focused_independent_audit": True,
        "automatic_assistant_installation_performed": False,
        "direct_action_authority": False,
        "live_action_authority": False,
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_gpu_or_tpu_acceleration": False,
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
    })
    result_root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "aggregate"
        / run_id
    )
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    manifest_path = result_root / RESULT_MANIFEST_NAME
    create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=run_id)
    returned_raw = _fixed_zip_bytes([(RESULT_MANIFEST_NAME, manifest_raw)])
    returned_name = (
        "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_OOF_AGGREGATE_RESULT_"
        f"{run_id[:16]}.zip"
    )
    returned_path = stage_drive_root / "returned_results" / returned_name
    returned_path.parent.mkdir(parents=True, exist_ok=True)
    create_or_verify_bytes(returned_path, returned_raw, owner_run_id=run_id)
    print(json.dumps({
        "event": "stage6_retrospective_nested_oof_aggregate_complete",
        "status": manifest["status"],
        "qualified_layers": qualified_layers,
        "held_layers": manifest["held_layers"],
        "execution_telemetry": {
            "requested_workers": raw_workers,
            "accepted_accelerator": accelerator,
            "elapsed_seconds": time.monotonic() - started,
        },
        "returned_result": str(returned_path),
    }, sort_keys=True), flush=True)
    return {
        "status": manifest["status"],
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "qualified_layers": qualified_layers,
        "held_layers": manifest["held_layers"],
        "returned_result": {
            "path": str(returned_path),
            "name": returned_name,
            "bytes": len(returned_raw),
            "sha256": hashlib.sha256(returned_raw).hexdigest(),
        },
    }


__all__ = ["run_retrospective_nested_oof_aggregate"]
