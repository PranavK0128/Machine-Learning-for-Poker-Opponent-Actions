"""Seal the user-approved partial closeout of the final N35 search.

This module never invents a score for the abandoned candidate and never emits
an outer-test prediction.  It validates and packages the 383 durable candidate
evaluation checkpoints that were already completed, records the sole missing
candidate, and leaves the layer/fold scientifically held.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from stage6_retrospective_nested_search import (
    EXACT_CANDIDATE_EVALUATIONS_PER_FOLD,
    FOLD_RUN_ID,
    LAYER_FAMILIES,
    R2_RUNTIME_RESUME_STORAGE_RUN_IDS,
    _checkpoint_inventory,
    _fixed_zip_bytes,
    _rank,
    _registry,
    canonical_json_bytes,
    load_fold_freeze,
    seal_document,
    validate_sealed_document,
)


PARTIAL_LAYER = "hero_opponent_interaction"
PARTIAL_FOLD_INDEX = 4
PARTIAL_STAGE_COUNTS = (256, 96, 31)
EXPECTED_STAGE_COUNTS = (256, 96, 32)
EXPECTED_MISSING_CANDIDATE_ID = "catboost-012-1eda5f47d8c2"
PARTIAL_MANIFEST_NAME = "nested-search-partial-result-manifest.json"
ABANDONMENT_REASON = (
    "USER_APPROVED_ABANDONED_PROHIBITIVELY_SLOW_FINAL_CANDIDATE"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(work_root: Path, stage: int) -> list[Path]:
    root = work_root / "candidate_results" / f"stage-{stage}"
    return sorted(root.glob("*.json"), key=lambda path: path.name)


def _load_stage(
    work_root: Path,
    stage: int,
    registry_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = _checkpoint_files(work_root, stage)
    reports: list[dict[str, Any]] = []
    stage_bindings: set[str] = set()
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_sealed_document(document)
        candidate_id = str(document.get("candidate_id", ""))
        entry = registry_by_id.get(candidate_id)
        if (
            document.get("schema_version")
            != "stage6_retrospective_nested_candidate_checkpoint_v1"
            or document.get("stage") != stage
            or path.stem != candidate_id
            or entry is None
            or document.get("configuration_sha256")
            != entry.get("configuration_sha256")
            or document.get("qualification_authority") is not False
            or document.get("result", {}).get(
                "validation_temperature_selected"
            )
            is not False
            or not str(document.get("run_id", ""))
            or not str(document.get("stage_data_binding_self_sha256", ""))
        ):
            raise RuntimeError(f"partial-finalize checkpoint binding changed: {path}")
        stage_bindings.add(str(document["stage_data_binding_self_sha256"]))
        reports.append(document)
    if len(stage_bindings) != 1:
        raise RuntimeError(
            f"stage {stage} checkpoints do not share one frozen data binding"
        )
    if len(reports) != len({str(row["candidate_id"]) for row in reports}):
        raise RuntimeError(f"stage {stage} duplicates a candidate checkpoint")
    return reports, paths


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["candidate_id"]) for row in rows]


def finalize_partial_hero_opponent_interaction_fold_4(
    stage_drive_root: str | Path,
) -> dict[str, Any]:
    """Validate, seal, and package the 383 completed N35 evaluations."""
    from stage6_drive_fuse_publication import (
        create_or_verify_bytes,
        recover_claimed_bytes,
    )

    stage_drive_root = Path(stage_drive_root).resolve()
    fold_freeze = load_fold_freeze(stage_drive_root)
    assignment = fold_freeze["assignments"]["folds"][PARTIAL_FOLD_INDEX]
    validate_sealed_document(assignment)
    if (
        assignment.get("outer_fold_index") != PARTIAL_FOLD_INDEX
        or assignment.get("fold_id") != "outer-fold-04"
    ):
        raise RuntimeError("partial-finalize fold assignment changed")

    storage_run_id = R2_RUNTIME_RESUME_STORAGE_RUN_IDS[
        (str(assignment["fold_id"]), PARTIAL_LAYER)
    ]
    work_root = (
        stage_drive_root
        / "runtime_state"
        / "retrospective_nested_qualification"
        / "searches"
        / str(assignment["fold_id"])
        / LAYER_FAMILIES[PARTIAL_LAYER]
        / PARTIAL_LAYER
        / storage_run_id
    )
    completed_manifest = work_root / "nested-search-result-manifest.json"
    if completed_manifest.exists():
        raise RuntimeError(
            "N35 already has a completed result; partial finalization is not allowed"
        )

    registry = _registry(PARTIAL_LAYER)
    registry_by_id = {str(row["candidate_id"]): row for row in registry}
    stage_1, stage_1_paths = _load_stage(work_root, 1, registry_by_id)
    stage_2, stage_2_paths = _load_stage(work_root, 2, registry_by_id)
    stage_3, stage_3_paths = _load_stage(work_root, 3, registry_by_id)
    observed_counts = (len(stage_1), len(stage_2), len(stage_3))
    if observed_counts != PARTIAL_STAGE_COUNTS:
        raise RuntimeError(
            f"partial-finalize checkpoint counts changed: {observed_counts}"
        )
    if set(_ids(stage_1)) != set(registry_by_id):
        raise RuntimeError("stage 1 no longer covers the exact 256-candidate registry")

    expected_stage_2 = _ids(_rank(stage_1)[:EXPECTED_STAGE_COUNTS[1]])
    if set(_ids(stage_2)) != set(expected_stage_2):
        raise RuntimeError("stage 2 is not the exact frozen stage-1 survivor set")
    expected_stage_3 = _ids(_rank(stage_2)[:EXPECTED_STAGE_COUNTS[2]])
    missing_ids = sorted(set(expected_stage_3) - set(_ids(stage_3)))
    unexpected_ids = sorted(set(_ids(stage_3)) - set(expected_stage_3))
    if missing_ids != [EXPECTED_MISSING_CANDIDATE_ID] or unexpected_ids:
        raise RuntimeError(
            "partial-finalize missing finalist changed: "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )

    stage_reports = (stage_1, stage_2, stage_3)
    inventories = []
    for stage, reports in enumerate(stage_reports, start=1):
        inventory, root_sha256 = _checkpoint_inventory(reports)
        inventories.append({
            "stage": stage,
            "completed_candidate_count": len(reports),
            "expected_candidate_count": EXPECTED_STAGE_COUNTS[stage - 1],
            "checkpoint_inventory": inventory,
            "checkpoint_inventory_root_sha256": root_sha256,
        })

    missing_entry = registry_by_id[EXPECTED_MISSING_CANDIDATE_ID]
    completed_stage_3_ranking = _rank(stage_3)
    best_completed = completed_stage_3_ranking[0]
    partial_run_binding = {
        "schema": "stage6_retrospective_nested_partial_finalize_run_v1",
        "fold_run_id": FOLD_RUN_ID,
        "fold_assignment_self_sha256": assignment["self_sha256"],
        "layer": PARTIAL_LAYER,
        "storage_run_id": storage_run_id,
        "checkpoint_inventory_root_sha256": [
            row["checkpoint_inventory_root_sha256"] for row in inventories
        ],
        "missing_candidate_id": EXPECTED_MISSING_CANDIDATE_ID,
    }
    partial_run_id = hashlib.sha256(
        canonical_json_bytes(partial_run_binding)
    ).hexdigest()
    manifest = seal_document({
        "schema_version": (
            "stage6_retrospective_nested_partial_fold_result_manifest_v1"
        ),
        "status": "HELD_INCOMPLETE_RETROSPECTIVE_NESTED_LAYER_FOLD_SEARCH",
        "run_id": partial_run_id,
        "fold_run_id": FOLD_RUN_ID,
        "fold_id": assignment["fold_id"],
        "fold_index": PARTIAL_FOLD_INDEX,
        "fold_assignment_self_sha256": assignment["self_sha256"],
        "layer": PARTIAL_LAYER,
        "family": LAYER_FAMILIES[PARTIAL_LAYER],
        "candidate_count": len(registry),
        "candidate_evaluations_per_fold": sum(PARTIAL_STAGE_COUNTS),
        "expected_candidate_evaluations_per_fold": (
            EXACT_CANDIDATE_EVALUATIONS_PER_FOLD
        ),
        "completed_stage_candidate_counts": list(PARTIAL_STAGE_COUNTS),
        "expected_stage_candidate_counts": list(EXPECTED_STAGE_COUNTS),
        "checkpoint_inventories": inventories,
        "missing_candidate_count": 1,
        "missing_candidate": {
            "candidate_id": EXPECTED_MISSING_CANDIDATE_ID,
            "family": missing_entry["family"],
            "configuration_sha256": missing_entry["configuration_sha256"],
            "parameters": missing_entry["parameters"],
        },
        "abandonment_reason": ABANDONMENT_REASON,
        "abandonment_authority": "explicit_user_instruction_2026-08-28",
        "scores_fabricated_for_missing_candidate": False,
        "missing_candidate_rank_eligible": False,
        "outer_test_checkpoint_created": False,
        "outer_test_scored": False,
        "best_completed_finalist_descriptive_only": {
            "candidate_id": best_completed["candidate_id"],
            "family": best_completed["result"]["family"],
            "configuration_sha256": best_completed["configuration_sha256"],
            "parameters": best_completed["result"]["parameters"],
            "validation_metrics": best_completed["result"][
                "validation_metrics"
            ],
            "best_iteration": best_completed["result"]["best_iteration"],
            "selection_or_qualification_authority": False,
        },
        "completed_checkpoints_are_evaluation_records_not_model_binaries": True,
        "qualification_authority": False,
        "fresh_or_prospective_claim": False,
        "action_timing_layer_included": False,
        "direct_action_authority": False,
        "live_action_authority": False,
        "partial_finalizer_implementation_sha256": _sha256_file(
            Path(__file__).resolve()
        ),
    })
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    manifest_path = work_root / PARTIAL_MANIFEST_NAME
    recover_claimed_bytes(manifest_path)
    create_or_verify_bytes(manifest_path, manifest_raw, owner_run_id=partial_run_id)

    checkpoint_paths = (*stage_1_paths, *stage_2_paths, *stage_3_paths)
    returned_members = [(PARTIAL_MANIFEST_NAME, manifest_raw)]
    returned_members.extend(
        (
            path.relative_to(work_root).as_posix(),
            path.read_bytes(),
        )
        for path in checkpoint_paths
    )
    returned_raw = _fixed_zip_bytes(returned_members)
    returned_name = (
        "POKERNOW_ASSISTANT_STAGE6_RETROSPECTIVE_NESTED_PARTIAL_RESULT_"
        f"{PARTIAL_LAYER}_{assignment['fold_id']}_{partial_run_id[:12]}.zip"
    )
    returned_path = stage_drive_root / "returned_results" / returned_name
    returned_path.parent.mkdir(parents=True, exist_ok=True)
    recover_claimed_bytes(returned_path)
    create_or_verify_bytes(returned_path, returned_raw, owner_run_id=partial_run_id)
    return {
        "status": manifest["status"],
        "run_id": partial_run_id,
        "manifest_path": str(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "completed_candidate_evaluations": sum(PARTIAL_STAGE_COUNTS),
        "expected_candidate_evaluations": EXACT_CANDIDATE_EVALUATIONS_PER_FOLD,
        "missing_candidate_id": EXPECTED_MISSING_CANDIDATE_ID,
        "qualification_authority": False,
        "returned_result": {
            "path": str(returned_path),
            "name": returned_name,
            "bytes": len(returned_raw),
            "sha256": hashlib.sha256(returned_raw).hexdigest(),
        },
    }


__all__ = ["finalize_partial_hero_opponent_interaction_fold_4"]
