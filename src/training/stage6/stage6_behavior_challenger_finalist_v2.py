"""Development finalist, refit, calibration, and aggregate freeze for behavior v2."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from stage6_behavior_json_publication_v2 import (
    load_json_with_recovery,
    write_json_exclusive,
)
from stage6_drive_fuse_publication import create_or_verify_bytes, recover_claimed_bytes
from stage6_behavior_challenger_notebook_runtime_v2 import (
    ActiveWorkHeartbeat,
    HEARTBEAT_INTERVAL_SECONDS,
    PREFERRED_NOTEBOOK_HOST,
    PRIMARY_MANIFEST_NAME as SHARD_MANIFEST_NAME,
    _completed_manifest,
    load_behavior_dataset,
    notebook_shard_run_id,
    shard_work_root,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    FINALIST_COUNT_PER_SHARD,
    FINALIST_ITERATION_CAP,
    FINALIST_MINIMUM_IMPROVEMENT,
    FINALIST_PATIENCE,
    SCOPES,
    SOURCE_DEPENDENCY_DIGEST,
    TARGETED_REFINEMENT_MAX_CANDIDATES,
    candidate_registry,
    candidate_registry_shard,
    canonical_json_bytes,
    checkpoint_path as shard_checkpoint_path,
    feature_contract,
    load_candidate_checkpoint,
    registry_digest,
    resolve_candidate_workers,
    scope_data_root_binding,
    scope_predicate,
    search_run_id,
    shard_digest,
    stage_data_binding,
    targeted_refinement_spec,
    worker_policy_contract,
)
from stage6_behavior_tournament import feature_names
from stage6_expanded_feature_freeze_v2 import BASE_FEATURE_COLUMNS, FEATURE_RUN_ID
from stage6_expanded_finalist import (
    _calibrated_report,
    _fit_fixed_candidate,
    _full_selection_arrays,
)
from stage6_expanded_layers import (
    ACTIONS,
    SCIENCE_IMPLEMENTATION_DIGEST,
    seal_document,
    validate_document,
)
from stage6_expanded_tournament import (
    _apply_imputer,
    _evaluate_candidate,
    _impute_medians,
    _write_stage_arrays,
    runtime_environment,
)
from stage6_notebooks02_07_runtime import sha256_file


REQUIRED_BASELINES = (
    "class_frequency",
    "current_qualified_artifact",
    "deterministic_chart_or_heuristic",
    "legal_random_or_uniform_where_semantically_valid",
)
FRESH_CUTOFF_UTC = "2026-08-21T00:00:00.000Z"
MINIMUM_INDEPENDENT_GAME_CLUSTERS = 20
BOOTSTRAP_REPLICATES = 2_000
SIGN_FLIP_REPLICATES = 10_000
FAMILY_ALPHA = 0.05
MAX_CANDIDATE_ATTEMPTS = 2
FINALIST_STAGE = {
    "stage": 4,
    "iteration_cap": FINALIST_ITERATION_CAP,
    "patience": FINALIST_PATIENCE,
    "minimum_meaningful_improvement": FINALIST_MINIMUM_IMPROVEMENT,
}

# R10 shard runs finished before the R11 transport-only packaging repair.  R11
# added the five already-frozen R5 release files to the science ZIP so the
# finalist bootstrap could reproduce package_identity() in Colab.  That edit
# changed only notebook/package and post-shard finalist control code, but the
# original binding included both in each shard run identity even though shard
# training imports neither one.  Preserve the four completed, compute-identical
# shard results with an exact, fail-closed compatibility record instead of
# spending hours retraining them.
R10_SHARD_NONCOMPUTE_MEMBER_IDENTITIES = {
    "src/stage6_behavior_challenger_science_notebook_builder_v2.py": {
        "relative_path": (
            "src/stage6_behavior_challenger_science_notebook_builder_v2.py"
        ),
        "bytes": 19_159,
        "sha256": (
            "993911024e6ee6709d99f45fb543a796def4a74d4b3678f3e060e592cbebd7ed"
        ),
    },
    "src/stage6_behavior_challenger_finalist_v2.py": {
        "relative_path": "src/stage6_behavior_challenger_finalist_v2.py",
        "bytes": 49_918,
        "sha256": (
            "0411c072924c82f099f1602c4b0bc45685ba15308e9d18936857d8e9d524d7fb"
        ),
    },
}
R10_SHARD_IMPLEMENTATION_SELF_SHA256 = (
    "a973adb00667f1a63d20f8c34479a57bd697de762008e41d6495f24b25ee6049"
)
R10_SHARD_SOURCE_DEPENDENCY_DIGEST = (
    "423aae26fcc471f5cc6db5fab6444251d931bbc90b7294fe5ef10eb17a387719"
)
R10_COMPLETED_SHARDS = {
    ("preflop_behavior", 0): {
        "notebook_shard_run_id": (
            "c8a1b76a1064f01d7dc448eebf2c4599ebe45c9830faacda120b2dd805fd01a8"
        ),
        "scope_search_run_id": (
            "94a676190223c5e7c8099e0935d68685a862ecf26222565abc5bc8f12197e65d"
        ),
        "manifest_self_sha256": (
            "cd1e3b43e910574e817b94421c4bdd318ab7ab10d3a1d6ea42de91c2e3d82a5b"
        ),
    },
    ("preflop_behavior", 1): {
        "notebook_shard_run_id": (
            "f3277cb183843b03547baddf6fe51bef08a8a90b17124812e1999bc156509c28"
        ),
        "scope_search_run_id": (
            "b30dce516ac836587873411a446726fd89841a6ed00f24a629b764fb3eaca4bf"
        ),
        "manifest_self_sha256": (
            "fbeaf9e2d6b3a1372b3dc7a5660154673c6e45175d14102c0912d8d7bd5b6d44"
        ),
    },
    ("postflop_behavior", 0): {
        "notebook_shard_run_id": (
            "42caefbef3f0c23869b6ea68dfb120201d25c519ab9b2ff6d8622186ceda7170"
        ),
        "scope_search_run_id": (
            "e7acad94b800289aa0d6204265f819f33ccdcf15ec7c6b139d9c2500a8849f92"
        ),
        "manifest_self_sha256": (
            "2a1662c95ca61ea5eda0b777e4c0946da11edef03b979aa24232c897fc54436c"
        ),
    },
    ("postflop_behavior", 1): {
        "notebook_shard_run_id": (
            "0ba68c8a900e9b1ab82552957cc6dcaccf4d854961beba3998ac2e79cc24e648"
        ),
        "scope_search_run_id": (
            "0d9bbf1ae8094206e3525d6e05e5711c5a7e89701ff64dcbd16c6fbe89cdd4b5"
        ),
        "manifest_self_sha256": (
            "d6dac8e332e4fcf5cce2a418b9d05458a5c75c4778307ac5e3cf8c100c663d83"
        ),
    },
}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _science_package_binding(
    package_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if package_binding is None:
        from stage6_behavior_challenger_science_notebook_builder_v2 import (
            package_identity,
        )

        candidate = package_identity()["runtime_package_binding"]
    else:
        candidate = dict(package_binding)
    validate_document(candidate)
    if (
        candidate.get("schema_version")
        != "stage6_behavior_challenger_science_runtime_package_binding_v3"
        or candidate.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or candidate.get("feature_run_id") != FEATURE_RUN_ID
        or not candidate.get("package_manifest_self_sha256")
        or not candidate.get("science_precommit_self_sha256")
        or not candidate.get("package_sha256")
    ):
        raise RuntimeError("behavior science runtime package binding is invalid")
    return candidate


def aggregate_run_id(
    package_binding: Mapping[str, Any] | None = None,
) -> str:
    runtime_package = _science_package_binding(package_binding)
    return _digest({
        "schema": "stage6_behavior_challenger_aggregate_freeze_v3",
        "feature_run_id": FEATURE_RUN_ID,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "science_runtime_package_binding": runtime_package,
        "scope_shard_run_ids": {
            scope: [notebook_shard_run_id(scope, shard) for shard in (0, 1)]
            for scope in SCOPES
        },
        "registry_digests": {scope: registry_digest(scope) for scope in SCOPES},
        "finalist_count_per_shard": FINALIST_COUNT_PER_SHARD,
        "finalist_stage": FINALIST_STAGE,
        "targeted_refinement_max_candidates": TARGETED_REFINEMENT_MAX_CANDIDATES,
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "required_baselines": REQUIRED_BASELINES,
        "paired_cluster_identity": "game_id",
        "minimum_independent_game_clusters": MINIMUM_INDEPENDENT_GAME_CLUSTERS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "multiple_comparison_method": "holm_bonferroni",
        "family_alpha": FAMILY_ALPHA,
    })


def aggregate_work_root(stage_drive_root: Path) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "behavior_challenger_v2"
        / "aggregate_freezes"
        / aggregate_run_id()
    )


def _write_exact_json(path: Path, document: Mapping[str, Any]) -> None:
    write_json_exclusive(
        path,
        document,
        root=path.parent,
        publication_binding=seal_document({
            "schema_version": "stage6_behavior_finalist_json_binding_v3",
            "aggregate_run_id": aggregate_run_id(),
            "science_runtime_package_binding": _science_package_binding(),
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
            "feature_run_id": FEATURE_RUN_ID,
        }),
    )


def _load_exact_json(path: Path) -> dict[str, Any]:
    return load_json_with_recovery(
        path,
        root=path.parent,
        publication_binding=seal_document({
            "schema_version": "stage6_behavior_finalist_json_binding_v3",
            "aggregate_run_id": aggregate_run_id(),
            "science_runtime_package_binding": _science_package_binding(),
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
            "feature_run_id": FEATURE_RUN_ID,
        }),
    )


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _r10_completed_shard_root(
    stage_drive_root: Path,
    scope: str,
    shard: int,
) -> Path:
    release = R10_COMPLETED_SHARDS[(scope, shard)]
    return (
        stage_drive_root
        / "runtime_state"
        / "behavior_challenger_v2"
        / scope
        / f"shard-{shard}"
        / release["notebook_shard_run_id"]
    )


def _validate_r10_transport_only_binding(binding: Mapping[str, Any]) -> None:
    """Prove R10 differs only in code that does not execute inside a shard."""
    validate_document(binding)
    current_members = {
        str(row["relative_path"]): dict(row)
        for row in BEHAVIOR_IMPLEMENTATION_BINDING["members"]
    }
    legacy_members = {
        str(row["relative_path"]): dict(row)
        for row in binding.get("members", [])
        if isinstance(row, Mapping) and row.get("relative_path")
    }
    if (
        binding.get("schema_version")
        != "stage6_behavior_challenger_implementation_binding_v2"
        or binding.get("self_sha256")
        != R10_SHARD_IMPLEMENTATION_SELF_SHA256
        or binding.get("feature_run_id") != FEATURE_RUN_ID
        or binding.get("feature_release_binding")
        != BEHAVIOR_IMPLEMENTATION_BINDING["feature_release_binding"]
        or binding.get("member_order")
        != BEHAVIOR_IMPLEMENTATION_BINDING["member_order"]
        or set(legacy_members) != set(current_members)
        or any(
            legacy_members.get(relative_path) != identity
            for relative_path, identity
            in R10_SHARD_NONCOMPUTE_MEMBER_IDENTITIES.items()
        )
    ):
        raise RuntimeError("R10 behavior shard implementation binding is invalid")
    for relative_path, current in current_members.items():
        if relative_path in R10_SHARD_NONCOMPUTE_MEMBER_IDENTITIES:
            continue
        if legacy_members[relative_path] != current:
            raise RuntimeError(
                "R10 behavior shard differs outside post-shard control code"
            )


def _load_r10_completed_shard(
    stage_drive_root: Path,
    scope: str,
    shard: int,
) -> tuple[dict[str, Any], Path]:
    release = R10_COMPLETED_SHARDS[(scope, shard)]
    root = _r10_completed_shard_root(stage_drive_root, scope, shard)
    path = root / SHARD_MANIFEST_NAME
    recover_claimed_bytes(path)
    if not path.is_file():
        raise RuntimeError(f"{scope} shard {shard} must finish before finalists")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_document(manifest)
    terminal = manifest.get("terminal_ranked_results")
    summaries = manifest.get("halving_stage_summaries")
    environment = runtime_environment()
    if (
        manifest.get("schema_version")
        != "stage6_behavior_challenger_shard_manifest_v2"
        or manifest.get("status")
        != "PASS_DEVELOPMENT_SUCCESSIVE_HALVING_COMPLETE"
        or manifest.get("self_sha256") != release["manifest_self_sha256"]
        or manifest.get("notebook_shard_run_id")
        != release["notebook_shard_run_id"]
        or manifest.get("scope_search_run_id")
        != release["scope_search_run_id"]
        or manifest.get("scope") != scope
        or manifest.get("shard") != shard
        or manifest.get("registry_digest") != registry_digest(scope)
        or manifest.get("shard_digest") != shard_digest(scope, shard)
        or manifest.get("feature_contract") != feature_contract(scope)
        or manifest.get("behavior_implementation_self_sha256")
        != R10_SHARD_IMPLEMENTATION_SELF_SHA256
        or manifest.get("source_dependency_digest")
        != R10_SHARD_SOURCE_DEPENDENCY_DIGEST
        or manifest.get("feature_run_id") != FEATURE_RUN_ID
        or manifest.get("environment") != environment
        or manifest.get("environment_self_sha256")
        != environment.get("self_sha256")
        or manifest.get("base_candidate_count") != 128
        or not isinstance(summaries, list)
        or len(summaries) != 3
        or not isinstance(terminal, list)
        or len(terminal) != 16
        or manifest.get("fresh_targets_opened") is not False
        or manifest.get("qualification_authority") is not False
        or manifest.get("assistant_installation_allowed") is not False
    ):
        raise RuntimeError(f"{scope} shard {shard} R10 manifest is invalid")
    _validate_r10_transport_only_binding(
        manifest["behavior_implementation_binding"]
    )
    for expected_stage, identity in zip((1, 2, 3), summaries, strict=True):
        if not isinstance(identity, Mapping):
            raise RuntimeError("R10 behavior shard stage summary identity is invalid")
        summary_path = root / str(identity.get("name", ""))
        recover_claimed_bytes(summary_path)
        if (
            identity.get("stage") != expected_stage
            or identity.get("name") != f"stage-{expected_stage}-summary.json"
            or not summary_path.is_file()
            or summary_path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(summary_path) != identity.get("sha256")
        ):
            raise RuntimeError("R10 behavior shard stage summary bytes changed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_document(summary)
        if (
            summary.get("self_sha256") != identity.get("self_sha256")
            or summary.get("schema_version")
            != "stage6_behavior_challenger_halving_summary_v2"
            or summary.get("stage") != expected_stage
            or summary.get("scope") != scope
            or summary.get("shard") != shard
            or summary.get("scope_search_run_id")
            != release["scope_search_run_id"]
            or summary.get("notebook_shard_run_id")
            != release["notebook_shard_run_id"]
            or summary.get("environment_self_sha256")
            != environment.get("self_sha256")
            or summary.get("fresh_targets_opened") is not False
            or summary.get("qualification_authority") is not False
        ):
            raise RuntimeError("R10 behavior shard stage summary is invalid")
    if manifest.get("halving_stage_summary_self_sha256") != [
        row.get("self_sha256") for row in summaries
    ]:
        raise RuntimeError("R10 behavior terminal stage summary binding changed")
    return manifest, root


def _validate_r10_terminal_checkpoint(
    path: Path,
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    scope: str,
    shard: int,
    release: Mapping[str, Any],
) -> None:
    recover_claimed_bytes(path)
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("checkpoint_bytes", -1))
        or sha256_file(path) != row.get("checkpoint_sha256")
    ):
        raise RuntimeError("R10 terminal behavior checkpoint byte identity changed")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    validate_document(checkpoint)
    result = checkpoint.get("result")
    if (
        checkpoint.get("self_sha256") != row.get("checkpoint_self_sha256")
        or checkpoint.get("schema_version")
        != "stage6_behavior_challenger_candidate_checkpoint_v2"
        or checkpoint.get("status")
        != "PASS_DEVELOPMENT_CANDIDATE_STAGE_COMPLETE"
        or checkpoint.get("search_run_id") != release["scope_search_run_id"]
        or checkpoint.get("scope") != scope
        or checkpoint.get("shard") != shard
        or checkpoint.get("stage") != 3
        or checkpoint.get("candidate_id") != entry["candidate_id"]
        or checkpoint.get("family") != entry["family"]
        or checkpoint.get("configuration_sha256")
        != entry["configuration_sha256"]
        or checkpoint.get("registry_digest") != registry_digest(scope)
        or checkpoint.get("shard_digest") != shard_digest(scope, shard)
        or checkpoint.get("model_feature_contract_self_sha256")
        != feature_contract(scope)["self_sha256"]
        or checkpoint.get("behavior_implementation_self_sha256")
        != R10_SHARD_IMPLEMENTATION_SELF_SHA256
        or checkpoint.get("source_dependency_digest")
        != R10_SHARD_SOURCE_DEPENDENCY_DIGEST
        or checkpoint.get("feature_run_id") != FEATURE_RUN_ID
        or checkpoint.get("fresh_targets_opened") is not False
        or checkpoint.get("qualification_authority") is not False
        or checkpoint.get("assistant_installation_allowed") is not False
        or not isinstance(result, Mapping)
        or result.get("candidate_id") != entry["candidate_id"]
        or result.get("family") != entry["family"]
        or result.get("configuration_sha256") != entry["configuration_sha256"]
        or result.get("parameters") != entry["parameters"]
        or result.get("stage") != 3
        or result.get("candidate_internal_threads") != 1
    ):
        raise RuntimeError("R10 terminal behavior checkpoint is invalid")


def _load_shard_manifest(
    stage_drive_root: Path,
    scope: str,
    shard: int,
) -> tuple[dict[str, Any], Path]:
    root = shard_work_root(stage_drive_root, scope, shard)
    path = root / SHARD_MANIFEST_NAME
    legacy_r10 = not path.is_file()
    if legacy_r10:
        manifest, root = _load_r10_completed_shard(
            stage_drive_root, scope, shard
        )
    else:
        manifest = _completed_manifest(stage_drive_root, root, scope, shard)
        if manifest is None:
            raise RuntimeError(f"{scope} shard {shard} manifest disappeared")
    expected_notebook_shard_run_id = (
        R10_COMPLETED_SHARDS[(scope, shard)]["notebook_shard_run_id"]
        if legacy_r10
        else notebook_shard_run_id(scope, shard)
    )
    expected_scope_search_run_id = (
        R10_COMPLETED_SHARDS[(scope, shard)]["scope_search_run_id"]
        if legacy_r10
        else search_run_id(scope, shard)
    )
    terminal = manifest.get("terminal_ranked_results")
    if (
        manifest.get("schema_version")
        != "stage6_behavior_challenger_shard_manifest_v2"
        or manifest.get("status") != "PASS_DEVELOPMENT_SUCCESSIVE_HALVING_COMPLETE"
        or manifest.get("notebook_shard_run_id")
        != expected_notebook_shard_run_id
        or manifest.get("scope_search_run_id")
        != expected_scope_search_run_id
        or manifest.get("scope") != scope
        or manifest.get("shard") != shard
        or manifest.get("registry_digest") != registry_digest(scope)
        or manifest.get("shard_digest") != shard_digest(scope, shard)
        or manifest.get("base_candidate_count") != 128
        or not isinstance(terminal, list)
        or len(terminal) != 16
        or manifest.get("fresh_targets_opened") is not False
        or manifest.get("qualification_authority") is not False
    ):
        raise RuntimeError(f"{scope} shard {shard} manifest is invalid")
    expected = {
        row["candidate_id"]: row for row in candidate_registry_shard(scope, shard)
    }
    for rank, row in enumerate(terminal, start=1):
        entry = expected.get(str(row.get("candidate_id")))
        if (
            row.get("rank") != rank
            or entry is None
            or row.get("family") != entry["family"]
            or row.get("configuration_sha256") != entry["configuration_sha256"]
        ):
            raise RuntimeError("terminal behavior challenger ranking is not registry-bound")
        checkpoint_file = shard_checkpoint_path(
            root, scope, shard, 3, entry["candidate_id"]
        )
        if legacy_r10:
            _validate_r10_terminal_checkpoint(
                checkpoint_file,
                row,
                entry,
                scope,
                shard,
                R10_COMPLETED_SHARDS[(scope, shard)],
            )
            continue
        terminal_binding = manifest["halving_stage_data_bindings"][-1]
        checkpoint = load_candidate_checkpoint(
            checkpoint_file,
            scope,
            shard,
            3,
            entry["candidate_id"],
            stage_data_binding_self_sha256=terminal_binding["self_sha256"],
            feature_manifest_self_sha256=manifest[
                "feature_manifest_self_sha256"
            ],
            whole_hand_assignment_self_sha256=manifest[
                "whole_hand_temporal_assignment_binding"
            ]["self_sha256"],
            environment_self_sha256=manifest["environment"]["self_sha256"],
        )
        if (
            checkpoint is None
            or checkpoint["self_sha256"] != row.get("checkpoint_self_sha256")
            or checkpoint_file.stat().st_size != row.get("checkpoint_bytes")
            or sha256_file(checkpoint_file) != row.get("checkpoint_sha256")
        ):
            raise RuntimeError("terminal behavior checkpoint byte identity changed")
    return manifest, root


def _development_finalists(
    stage_drive_root: Path,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = {row["candidate_id"]: row for row in candidate_registry(scope)}
    manifests: list[dict[str, Any]] = []
    finalists: list[dict[str, Any]] = []
    for shard in (0, 1):
        manifest, _ = _load_shard_manifest(stage_drive_root, scope, shard)
        manifests.append(manifest)
        for row in manifest["terminal_ranked_results"][:FINALIST_COUNT_PER_SHARD]:
            finalists.append(copy.deepcopy(registry[str(row["candidate_id"])]))
    if len(finalists) != 2 * FINALIST_COUNT_PER_SHARD:
        raise RuntimeError("behavior finalist inventory must contain 12 per shard")
    if len({row["candidate_id"] for row in finalists}) != len(finalists):
        raise RuntimeError("behavior finalists overlap deterministic shards")
    if (
        manifests[0]["halving_stage_data_bindings"]
        != manifests[1]["halving_stage_data_bindings"]
    ):
        raise RuntimeError("behavior shards used different nested development rungs")
    if (
        manifests[0]["whole_hand_temporal_assignment_binding"]
        != manifests[1]["whole_hand_temporal_assignment_binding"]
    ):
        raise RuntimeError("behavior shards used different whole-hand assignments")
    return finalists, manifests


def refinement_registry(
    scope: str,
    winner: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    winner_ranking_self_sha256: str,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
) -> dict[str, Any]:
    """Seal the exact deterministic local neighborhood before calibration."""
    validate_document(specification)
    if specification.get("fresh_targets_opened") is not False:
        raise RuntimeError("targeted refinement cannot follow fresh target opening")
    roots = scope_data_root_binding(
        scope,
        feature_manifest_self_sha256=feature_manifest_self_sha256,
        stage_data_binding_self_sha256=stage_data_binding_self_sha256,
        whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
    )
    if (
        specification.get("winner_ranking_self_sha256")
        != winner_ranking_self_sha256
        or specification.get("scope_data_root_binding") != roots
        or specification.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or specification.get("feature_run_id") != FEATURE_RUN_ID
        or specification.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or specification.get("source_dependency_digest")
        != SOURCE_DEPENDENCY_DIGEST
    ):
        raise RuntimeError("targeted refinement specification binding changed")
    variants = copy.deepcopy(list(specification.get("refinement_candidates", [])))
    if (
        len(variants) != specification.get("refinement_candidate_count")
        or len(variants) > TARGETED_REFINEMENT_MAX_CANDIDATES
        or _digest(variants)
        != specification.get("refinement_candidate_registry_digest")
        or any(
            row.get("refinement_parent_candidate_id") != winner["candidate_id"]
            or row.get("family") != winner["family"]
            or row.get("candidate_internal_threads") != 1
            for row in variants
        )
    ):
        raise RuntimeError("sealed targeted refinement candidate registry changed")
    registry = seal_document({
        "schema_version": "stage6_behavior_targeted_refinement_registry_v2",
        "status": (
            "PASS_DEVELOPMENT_REFINEMENT_REGISTRY_FROZEN"
            if specification.get("enabled")
            else "PASS_DEVELOPMENT_REFINEMENT_NOT_REQUIRED"
        ),
        "scope": scope,
        "targeted_refinement_spec_self_sha256": specification["self_sha256"],
        "winner_candidate_id": winner["candidate_id"],
        "winner_configuration_sha256": winner["configuration_sha256"],
        "winner_ranking_self_sha256": winner_ranking_self_sha256,
        "scope_data_root_binding": roots,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "selection_target_roles": [
            "development_train",
            "development_validation",
        ],
        "retrospective_calibration_used": False,
        "candidate_count": len(variants),
        "maximum_candidate_count": TARGETED_REFINEMENT_MAX_CANDIDATES,
        "candidate_registry_digest": _digest(variants),
        "candidates": variants,
        "fresh_targets_opened": False,
        "qualification_authority": False,
    })
    return registry


def _checkpoint_path(root: Path, candidate_id: str) -> Path:
    return root / "candidate_results" / f"{candidate_id}.json"


def _load_finalist_checkpoint(
    path: Path,
    scope: str,
    finalist_run_id: str,
    entry: Mapping[str, Any],
    binding: str,
    feature_manifest: str,
    whole_hand_assignment: str,
    environment: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = _load_exact_json(path)
    if (
        document.get("schema_version")
        != "stage6_behavior_challenger_finalist_checkpoint_v2"
        or document.get("status") != "PASS_DEVELOPMENT_FINALIST_COMPLETE"
        or document.get("finalist_run_id") != finalist_run_id
        or document.get("scope") != scope
        or document.get("candidate_id") != entry["candidate_id"]
        or document.get("configuration_sha256") != entry["configuration_sha256"]
        or document.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or document.get("feature_run_id") != FEATURE_RUN_ID
        or document.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or document.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
        or document.get("scope_predicate_self_sha256")
        != scope_predicate(scope)["self_sha256"]
        or document.get("scope_data_root_binding")
        != scope_data_root_binding(
            scope,
            feature_manifest_self_sha256=feature_manifest,
            stage_data_binding_self_sha256=binding,
            whole_hand_assignment_self_sha256=whole_hand_assignment,
        )
        or document.get("feature_manifest_self_sha256") != feature_manifest
        or document.get("stage_data_binding_self_sha256") != binding
        or document.get("whole_hand_assignment_self_sha256")
        != whole_hand_assignment
        or document.get("environment_self_sha256") != environment
        or document.get("candidate_internal_threads") != 1
        or document.get("fresh_targets_opened") is not False
    ):
        raise RuntimeError("behavior finalist checkpoint identity mismatch")
    result = document.get("result")
    if (
        not isinstance(result, Mapping)
        or result.get("candidate_id") != entry["candidate_id"]
        or result.get("family") != entry["family"]
        or result.get("configuration_sha256") != entry["configuration_sha256"]
        or result.get("parameters") != entry["parameters"]
        or result.get("stage") != 4
        or result.get("candidate_internal_threads") != 1
    ):
        raise RuntimeError("behavior finalist nested result identity mismatch")
    return document


def _worker(job: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_candidate(job["entry"], job["paths"], job["stage"])


def _rank(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        reports,
        key=lambda row: (
            float(row["result"]["validation_metrics"]["log_loss"]),
            float(row["result"]["validation_metrics"]["multiclass_brier_score"]),
            str(row["candidate_id"]),
        ),
    )


def _winner_ranking_binding(
    scope: str,
    ranked: Sequence[Mapping[str, Any]],
    *,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
) -> dict[str, Any]:
    """Seal the complete base-finalist development ranking used to refine."""
    return seal_document({
        "schema_version": "stage6_behavior_base_finalist_ranking_v2",
        "scope": scope,
        "ranked_results": [
            {
                "rank": index + 1,
                "candidate_id": row["candidate_id"],
                "configuration_sha256": row["configuration_sha256"],
                "checkpoint_self_sha256": row["self_sha256"],
                "validation_log_loss": row["result"]["validation_metrics"][
                    "log_loss"
                ],
                "validation_brier": row["result"]["validation_metrics"][
                    "multiclass_brier_score"
                ],
            }
            for index, row in enumerate(ranked)
        ],
        "scope_data_root_binding": scope_data_root_binding(
            scope,
            feature_manifest_self_sha256=feature_manifest_self_sha256,
            stage_data_binding_self_sha256=stage_data_binding_self_sha256,
            whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
        ),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "selection_target_roles": [
            "development_train",
            "development_validation",
        ],
        "retrospective_calibration_used": False,
        "fresh_targets_opened": False,
    })


def _evaluate_entries(
    root: Path,
    scope: str,
    finalist_run_id: str,
    entries: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    binding: Mapping[str, Any],
    feature_manifest: str,
    whole_hand_assignment: str,
    environment: Mapping[str, Any],
    requested_workers: str,
    accelerator: str,
    heartbeat: ActiveWorkHeartbeat,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    unfinished = []
    by_id = {str(entry["candidate_id"]): entry for entry in entries}
    for entry in entries:
        checkpoint = _load_finalist_checkpoint(
            _checkpoint_path(root, str(entry["candidate_id"])),
            scope,
            finalist_run_id,
            entry,
            str(binding["self_sha256"]),
            feature_manifest,
            whole_hand_assignment,
            str(environment["self_sha256"]),
        )
        if checkpoint is None:
            unfinished.append(entry)
        else:
            reports.append(checkpoint)
    heartbeat.update(
        phase="development_finalist_selection",
        stage=4,
        denominator=len(entries),
        completed=len(reports),
        work_unit="candidates",
    )
    if unfinished:
        worker_report = resolve_candidate_workers(
            requested_workers,
            len(unfinished),
            arrays,
            unfinished,
            FINALIST_STAGE,
            accelerator=accelerator,
        )
        with tempfile.TemporaryDirectory(prefix=f"stage6-behavior-finalist-{scope}-") as tmp:
            paths = _write_stage_arrays(Path(tmp), arrays)
            pending: list[Mapping[str, Any]] = list(unfinished)
            attempts = {str(row["candidate_id"]): 0 for row in pending}
            while pending:
                retry = []
                with ProcessPoolExecutor(
                    max_workers=int(worker_report["live_candidate_workers"])
                ) as executor:
                    futures = {}
                    for entry in pending:
                        candidate_id = str(entry["candidate_id"])
                        attempts[candidate_id] += 1
                        futures[
                            executor.submit(_worker, {
                                "entry": entry,
                                "paths": paths,
                                "stage": FINALIST_STAGE,
                            })
                        ] = entry
                    for future in as_completed(futures):
                        entry = futures[future]
                        candidate_id = str(entry["candidate_id"])
                        try:
                            result = future.result()
                        except Exception as error:
                            if attempts[candidate_id] < MAX_CANDIDATE_ATTEMPTS:
                                retry.append(entry)
                                continue
                            raise RuntimeError(
                                f"finalist {candidate_id} failed twice"
                            ) from error
                        checkpoint = seal_document({
                            "schema_version": (
                                "stage6_behavior_challenger_finalist_checkpoint_v2"
                            ),
                            "status": "PASS_DEVELOPMENT_FINALIST_COMPLETE",
                            "finalist_run_id": finalist_run_id,
                            "scope": scope,
                            "candidate_id": candidate_id,
                            "family": entry["family"],
                            "parameters": entry["parameters"],
                            "configuration_sha256": entry["configuration_sha256"],
                            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
                            "feature_run_id": FEATURE_RUN_ID,
                            "science_implementation_self_sha256": (
                                SCIENCE_IMPLEMENTATION_DIGEST
                            ),
                            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
                            "scope_predicate_self_sha256": scope_predicate(scope)[
                                "self_sha256"
                            ],
                            "scope_data_root_binding": scope_data_root_binding(
                                scope,
                                feature_manifest_self_sha256=feature_manifest,
                                stage_data_binding_self_sha256=binding["self_sha256"],
                                whole_hand_assignment_self_sha256=(
                                    whole_hand_assignment
                                ),
                            ),
                            "feature_manifest_self_sha256": feature_manifest,
                            "stage_data_binding_self_sha256": binding["self_sha256"],
                            "whole_hand_assignment_self_sha256": (
                                whole_hand_assignment
                            ),
                            "environment_self_sha256": environment["self_sha256"],
                            "candidate_internal_threads": 1,
                            "result": result,
                            "fresh_targets_opened": False,
                            "qualification_authority": False,
                        })
                        _write_exact_json(
                            _checkpoint_path(root, candidate_id), checkpoint
                        )
                        readback = _load_finalist_checkpoint(
                            _checkpoint_path(root, candidate_id),
                            scope,
                            finalist_run_id,
                            by_id[candidate_id],
                            str(binding["self_sha256"]),
                            feature_manifest,
                            whole_hand_assignment,
                            str(environment["self_sha256"]),
                        )
                        if readback != checkpoint:
                            raise RuntimeError("finalist checkpoint readback changed")
                        reports.append(checkpoint)
                        heartbeat.update(
                            phase="development_finalist_selection",
                            stage=4,
                            denominator=len(entries),
                            completed=len(reports),
                            work_unit="candidates",
                        )
                pending = retry
    if len(reports) != len(entries):
        raise RuntimeError("behavior finalist inventory was not conserved")
    return reports


def _save_artifact(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    recover_claimed_bytes(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(dict(artifact), temporary, compress=3)
        create_or_verify_bytes(
            path,
            temporary.read_bytes(),
            owner_run_id=str(artifact["configuration_sha256"]),
        )
    except (FileExistsError, RuntimeError) as error:
        raise RuntimeError("conflicting behavior finalist artifact exists") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    identity = _file_identity(path)
    loaded = joblib.load(path)
    if (
        loaded.get("candidate_id") != artifact["candidate_id"]
        or loaded.get("configuration_sha256") != artifact["configuration_sha256"]
    ):
        raise RuntimeError("behavior finalist artifact readback changed")
    return identity


def _scope_finalist_run_id(scope: str, shard_manifests: Sequence[Mapping[str, Any]]) -> str:
    return _digest({
        "schema": "stage6_behavior_challenger_scope_finalist_v2",
        "aggregate_run_id": aggregate_run_id(),
        "scope": scope,
        "shard_manifest_self_sha256": [row["self_sha256"] for row in shard_manifests],
        "finalist_stage": FINALIST_STAGE,
        "targeted_refinement_max_candidates": TARGETED_REFINEMENT_MAX_CANDIDATES,
        "fresh_targets_opened": False,
    })


def run_scope_finalist(
    stage_drive_root: Path,
    scope: str,
    requested_workers: str = "auto",
    accelerator: str = "TPU",
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    finalists, shard_manifests = _development_finalists(stage_drive_root, scope)
    run_id = _scope_finalist_run_id(scope, shard_manifests)
    root = aggregate_work_root(stage_drive_root) / "scope_finalists" / scope / run_id
    manifest_path = root / "finalist-freeze-manifest.json"
    if manifest_path.is_file():
        existing = _load_exact_json(manifest_path)
        if (
            existing.get("schema_version")
            != "stage6_behavior_challenger_scope_finalist_manifest_v3"
            or existing.get("scope_finalist_run_id") != run_id
            or existing.get("feature_run_id") != FEATURE_RUN_ID
            or existing.get("behavior_implementation_self_sha256")
            != BEHAVIOR_IMPLEMENTATION_DIGEST
            or existing.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
            or existing.get("science_runtime_package_binding")
            != _science_package_binding()
            or existing.get("fresh_targets_opened") is not False
        ):
            raise RuntimeError("existing behavior scope finalist freeze is invalid")
        return {
            "status": "PASS",
            "scope": scope,
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": existing["self_sha256"],
            "selected_artifact": existing["selected_artifact"],
            "reused_completed_result": True,
        }

    dataset: dict[str, Any] | None = None
    heartbeat = ActiveWorkHeartbeat(
        scope=scope,
        shard=-1,
        interval_seconds=heartbeat_interval_seconds,
    )
    heartbeat.update(phase="loading_finalist_development_data", denominator=1, completed=0, work_unit="datasets")
    heartbeat.start()
    started = time.monotonic()
    try:
        environment = runtime_environment()
        dataset = load_behavior_dataset(stage_drive_root, scope)
        heartbeat.update(phase="loading_finalist_development_data", denominator=1, completed=1, work_unit="datasets")
        arrays = _full_selection_arrays(dataset)
        binding = stage_data_binding(
            arrays, dataset["whole_hand_temporal_assignment_binding"]
        )
        base_reports = _evaluate_entries(
            root,
            scope,
            run_id,
            finalists,
            arrays,
            binding,
            str(dataset["feature_manifest_self_sha256"]),
            str(dataset["whole_hand_temporal_assignment_binding"]["self_sha256"]),
            environment,
            requested_workers,
            accelerator,
            heartbeat,
        )
        ranked_base = _rank(base_reports)
        ranking_binding = _winner_ranking_binding(
            scope,
            ranked_base,
            stage_data_binding_self_sha256=str(binding["self_sha256"]),
            feature_manifest_self_sha256=str(
                dataset["feature_manifest_self_sha256"]
            ),
            whole_hand_assignment_self_sha256=str(
                dataset["whole_hand_temporal_assignment_binding"]["self_sha256"]
            ),
        )
        provisional = ranked_base[0]
        provisional_entry = next(
            row for row in finalists if row["candidate_id"] == provisional["candidate_id"]
        )
        refinement = targeted_refinement_spec(
            scope,
            {
                "rank": 1,
                "candidate_id": provisional_entry["candidate_id"],
                "configuration_sha256": provisional_entry["configuration_sha256"],
                "best_iteration": provisional["result"]["best_iteration"],
                "iteration_cap": FINALIST_ITERATION_CAP,
            },
            winner_ranking_self_sha256=ranking_binding["self_sha256"],
            stage_data_binding_self_sha256=str(binding["self_sha256"]),
            feature_manifest_self_sha256=str(
                dataset["feature_manifest_self_sha256"]
            ),
            whole_hand_assignment_self_sha256=str(
                dataset["whole_hand_temporal_assignment_binding"]["self_sha256"]
            ),
            fresh_targets_opened=False,
        )
        refinement_freeze = refinement_registry(
            scope,
            provisional_entry,
            refinement,
            winner_ranking_self_sha256=ranking_binding["self_sha256"],
            stage_data_binding_self_sha256=str(binding["self_sha256"]),
            feature_manifest_self_sha256=str(
                dataset["feature_manifest_self_sha256"]
            ),
            whole_hand_assignment_self_sha256=str(
                dataset["whole_hand_temporal_assignment_binding"]["self_sha256"]
            ),
        )
        _write_exact_json(
            root / "targeted-refinement-registry.json", refinement_freeze
        )
        refined_entries = refinement_freeze["candidates"]
        refinement_reports: list[dict[str, Any]] = []
        if refined_entries:
            refinement_reports = _evaluate_entries(
                root,
                scope,
                run_id,
                refined_entries,
                arrays,
                binding,
                str(dataset["feature_manifest_self_sha256"]),
                str(
                    dataset["whole_hand_temporal_assignment_binding"][
                        "self_sha256"
                    ]
                ),
                environment,
                requested_workers,
                accelerator,
                heartbeat,
            )
        ranked = _rank([*base_reports, *refinement_reports])
        winner_report = ranked[0]
        all_entries = {
            str(row["candidate_id"]): row for row in [*finalists, *refined_entries]
        }
        winner = all_entries[str(winner_report["candidate_id"])]

        frame = dataset["frame"]
        refit_indices = frame.index[frame["temporal_role"].isin(
            ("development_train", "development_validation")
        )].to_numpy(dtype=np.int64)
        calibration_indices = frame.index[
            frame["temporal_role"].eq("retrospective_calibration")
        ].to_numpy(dtype=np.int64)
        if not len(refit_indices) or not len(calibration_indices):
            raise RuntimeError("behavior refit or retrospective calibration slice is empty")
        matrix = np.asarray(dataset["matrix"], dtype=np.float32)
        target = np.asarray(dataset["target"], dtype=np.int8)
        legal = np.asarray(dataset["legal"], dtype=bool)[calibration_indices]
        medians = _impute_medians(matrix[refit_indices])
        refit_x = _apply_imputer(matrix[refit_indices], medians)
        calibration_x = _apply_imputer(matrix[calibration_indices], medians)
        class_counts = np.bincount(
            target[refit_indices], minlength=len(ACTIONS)
        ).astype(np.float64)
        priors = (class_counts + 1.0) / (class_counts.sum() + len(ACTIONS))
        iterations = max(1, min(
            FINALIST_ITERATION_CAP,
            int(winner_report["result"]["best_iteration"] or 1),
        ))
        heartbeat.update(phase="refit_and_freeze_calibration", denominator=1, completed=0, work_unit="models")
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=1):
            model = _fit_fixed_candidate(
                winner, refit_x, target[refit_indices], iterations
            )
            temperature, _, calibration_metrics = _calibrated_report(
                model,
                calibration_x,
                target[calibration_indices],
                legal,
                priors,
            )
            threadpools = [
                {
                    "internal_api": str(row.get("internal_api", "unknown")),
                    "prefix": str(row.get("prefix", "unknown")),
                    "num_threads": int(row.get("num_threads", 0)),
                }
                for row in threadpool_info()
            ]
        if any(row["num_threads"] > 1 for row in threadpools):
            raise RuntimeError("behavior final refit exceeded one native thread")
        street_scope = "preflop" if scope == "preflop_behavior" else "postflop"
        artifact = {
            "schema_version": "stage6_expanded_behavior_candidate_artifact_v1",
            "artifact_kind": "single_model",
            "candidate_id": winner["candidate_id"],
            "family": winner["family"],
            "parameters": dict(winner["parameters"]),
            "configuration_sha256": winner["configuration_sha256"],
            "street_scope": street_scope,
            "label_order": list(ACTIONS),
            "feature_names": feature_names(),
            "imputer_medians": medians.tolist(),
            "class_priors": priors.tolist(),
            "temperature": float(temperature),
            "model": model,
            "tournament_run_id": aggregate_run_id(),
            "science_runtime_package_binding": _science_package_binding(),
            "scope_finalist_run_id": run_id,
            "selected_iterations": iterations,
            "holdout_targets_used": 0,
            "fresh_targets_opened": False,
            "recommendation_authority": False,
            "action_authority": False,
            "live_action_authority": False,
        }
        artifact_path = root / "artifacts" / f"{scope}-selected.joblib"
        artifact_identity = _save_artifact(artifact_path, artifact)
        heartbeat.update(phase="refit_and_freeze_calibration", denominator=1, completed=1, work_unit="models")
        manifest = seal_document({
            "schema_version": "stage6_behavior_challenger_scope_finalist_manifest_v3",
            "status": "PASS_FINALIST_REFIT_AND_CALIBRATION_FROZEN",
            "scope_finalist_run_id": run_id,
            "aggregate_run_id": aggregate_run_id(),
            "science_runtime_package_binding": _science_package_binding(),
            "feature_run_id": FEATURE_RUN_ID,
            "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
            "scope": scope,
            "shard_manifest_self_sha256": [
                row["self_sha256"] for row in shard_manifests
            ],
            "base_finalist_count": len(finalists),
            "targeted_refinement_spec": refinement,
            "base_finalist_ranking_binding": ranking_binding,
            "targeted_refinement_registry": refinement_freeze,
            "targeted_refinement_candidate_count": len(refined_entries),
            "targeted_refinement_checkpoint_self_sha256": [
                row["self_sha256"] for row in refinement_reports
            ],
            "selection_stage": dict(FINALIST_STAGE),
            "stage_data_binding": binding,
            "environment": environment,
            "ranked_finalists": [
                {
                    "rank": index + 1,
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "configuration_sha256": row["configuration_sha256"],
                    "validation_log_loss": row["result"]["validation_metrics"][
                        "log_loss"
                    ],
                    "validation_brier": row["result"]["validation_metrics"][
                        "multiclass_brier_score"
                    ],
                    "best_iteration": row["result"]["best_iteration"],
                    "checkpoint_self_sha256": row["self_sha256"],
                }
                for index, row in enumerate(ranked)
            ],
            "selected_candidate_id": winner["candidate_id"],
            "selected_family": winner["family"],
            "selected_configuration_sha256": winner["configuration_sha256"],
            "selected_iterations": iterations,
            "selected_artifact": artifact_identity,
            "artifact_relative_path": artifact_path.relative_to(
                stage_drive_root
            ).as_posix(),
            "retrospective_calibration_row_count": len(calibration_indices),
            "retrospective_calibration_metrics": calibration_metrics,
            "selected_temperature": float(temperature),
            "calibration_frozen_after_refit": True,
            "selection_or_tuning_after_calibration_freeze_allowed": False,
            "refit_threadpool_receipt": threadpools,
            "candidate_internal_threads": 1,
            "feature_contract": feature_contract(scope),
            "fresh_targets_opened": False,
            "qualification_authority": False,
            "assistant_installation_allowed": False,
            "recommendation_authority": False,
            "action_authority": False,
            "live_action_authority": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_exact_json(manifest_path, manifest)
        return {
            "status": "PASS",
            "scope": scope,
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": manifest["self_sha256"],
            "selected_artifact": artifact_identity,
            "reused_completed_result": False,
        }
    finally:
        heartbeat.stop()
        if dataset is not None:
            dataset.clear()
        gc.collect()


def _materialization_inventory(stage_drive_root: Path) -> list[str]:
    root = stage_drive_root / "runtime_state" / "notebook01" / "materializations"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def run_finalist_aggregate(
    stage_drive_root: Path,
    requested_workers: str = "auto",
    accelerator: str = "TPU",
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    root = aggregate_work_root(stage_drive_root)
    manifest_path = root / "aggregate-freeze-manifest.json"
    custody_path = root / "fresh-custody-request.json"
    if manifest_path.is_file():
        manifest = _load_exact_json(manifest_path)
        custody = _load_exact_json(custody_path)
        if (
            manifest.get("schema_version")
            != "stage6_behavior_challenger_aggregate_freeze_manifest_v3"
            or manifest.get("aggregate_run_id") != aggregate_run_id()
            or manifest.get("feature_run_id") != FEATURE_RUN_ID
            or manifest.get("behavior_implementation_self_sha256")
            != BEHAVIOR_IMPLEMENTATION_DIGEST
            or manifest.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
            or manifest.get("behavior_implementation_binding")
            != BEHAVIOR_IMPLEMENTATION_BINDING
            or manifest.get("science_runtime_package_binding")
            != _science_package_binding()
            or manifest.get("fresh_targets_opened") is not False
            or custody.get("aggregate_freeze_manifest_self_sha256")
            != manifest["self_sha256"]
        ):
            raise RuntimeError("existing behavior aggregate freeze is invalid")
        for scope in SCOPES:
            scope_result = run_scope_finalist(
                stage_drive_root, scope, requested_workers, accelerator,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            )
            if (
                scope_result["manifest_self_sha256"]
                != manifest["scope_freezes"][scope]["finalist_manifest_self_sha256"]
            ):
                raise RuntimeError("existing behavior aggregate scope binding changed")
        return {
            "status": "PASS",
            "aggregate_run_id": aggregate_run_id(),
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": manifest["self_sha256"],
            "fresh_custody_request_path": str(custody_path),
            "fresh_custody_request_self_sha256": custody["self_sha256"],
            "reused_completed_result": True,
            "fresh_targets_opened": False,
        }

    worker_policy = worker_policy_contract(accelerator)
    scope_freezes = {}
    for scope in SCOPES:
        result = run_scope_finalist(
            stage_drive_root,
            scope,
            requested_workers,
            accelerator,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        manifest = _load_exact_json(Path(result["manifest_path"]))
        scope_freezes[scope] = {
            "scope_finalist_run_id": manifest["scope_finalist_run_id"],
            "finalist_manifest_self_sha256": manifest["self_sha256"],
            "selected_candidate_id": manifest["selected_candidate_id"],
            "selected_configuration_sha256": manifest[
                "selected_configuration_sha256"
            ],
            "selected_artifact": manifest["selected_artifact"],
            "artifact_relative_path": manifest["artifact_relative_path"],
            "selected_temperature": manifest["selected_temperature"],
            "calibration_frozen_after_refit": True,
        }
    preregistered_claims = {
        scope: {
            "candidate": scope_freezes[scope]["selected_candidate_id"],
            "required_comparators": list(REQUIRED_BASELINES),
            "identical_row_comparison_required": True,
            "primary_metric": "multiclass_log_loss",
            "direction": "candidate_minus_comparator_lower_is_better",
            "confidence_interval_rule": "upper_95_percent_bound_below_zero",
            "holm_family": "all_preflop_and_postflop_comparator_claims",
            "brier_noninferiority_required": True,
            "zero_illegal_postprojection_required": True,
            "zero_missing_incumbent_inputs_required": True,
        }
        for scope in SCOPES
    }
    freeze = seal_document({
        "schema_version": "stage6_behavior_challenger_aggregate_freeze_manifest_v3",
        "status": "PASS_CANDIDATES_AND_CLAIMS_FROZEN_AWAITING_FRESH_DATA",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "feature_run_id": FEATURE_RUN_ID,
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "scope_freezes": scope_freezes,
        "preregistered_claims": preregistered_claims,
        "fresh_source_requirements": {
            "new_notebook01_materialization_after_this_freeze_required": True,
            "producer_bound_fresh_custody_binding_required": True,
            "all_48_logical_lanes_required": True,
            "whole_hand_after_cutoff_required": True,
            "target_free_exact_incumbent_context_sidecars_required": True,
            "one_shot_adjudication": True,
        },
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "required_baselines": list(REQUIRED_BASELINES),
        "paired_cluster_identity": "game_id",
        "whole_hand_identity": "game_id_plus_hand_id",
        "minimum_independent_game_clusters_per_claim": (
            MINIMUM_INDEPENDENT_GAME_CLUSTERS
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "multiple_comparison_method": "holm_bonferroni",
        "claim_family": [
            f"{scope}:{baseline}"
            for scope in SCOPES
            for baseline in REQUIRED_BASELINES
        ],
        "family_alpha": FAMILY_ALPHA,
        "worker_policy": worker_policy,
        "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_tpu_acceleration": False,
        "selection_or_tuning_after_freeze_allowed": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    custody = seal_document({
        "schema_version": "stage6_behavior_challenger_fresh_custody_request_v3",
        "status": "FROZEN_AWAITING_ONE_NEW_NOTEBOOK01_MATERIALIZATION",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "preexisting_notebook01_materialization_run_ids": _materialization_inventory(
            stage_drive_root
        ),
        "one_shot_adjudication": True,
        "selection_or_tuning_after_fresh_open_allowed": False,
        "fresh_targets_opened": False,
    })
    _write_exact_json(custody_path, custody)
    _write_exact_json(manifest_path, freeze)
    return {
        "status": "PASS",
        "aggregate_run_id": aggregate_run_id(),
        "manifest_path": str(manifest_path),
        "manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_path": str(custody_path),
        "fresh_custody_request_self_sha256": custody["self_sha256"],
        "reused_completed_result": False,
        "fresh_targets_opened": False,
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "FAMILY_ALPHA",
    "FRESH_CUTOFF_UTC",
    "MINIMUM_INDEPENDENT_GAME_CLUSTERS",
    "REQUIRED_BASELINES",
    "SIGN_FLIP_REPLICATES",
    "aggregate_run_id",
    "aggregate_work_root",
    "refinement_registry",
    "run_finalist_aggregate",
    "run_scope_finalist",
]
