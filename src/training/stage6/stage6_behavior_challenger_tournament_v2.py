"""Frozen expanded search contract for the two failed Stage 6 behavior scopes.

This module is deliberately additive.  It does not open data, start a notebook,
or inspect a fresh/lockbox target.  It freezes the development search registry,
model runtime controls, and checkpoint identities that a later notebook adapter
can execute using the existing Stage 6 causal rows and legal projection.

The base search contains exactly 256 candidates for each behavior scope.  The
two deterministic shards use identical development data rungs and evaluate
128 -> 48 -> 16 candidates.  A small winner-neighborhood refinement is only a
development specification: it is disabled unless the selected winner is on a
searched hyperparameter edge or reaches an iterative cap, and it always fails
closed after any fresh target has been opened.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from stage6_behavior_tournament import feature_names as established_feature_names
from stage6_expanded_feature_freeze_v2 import (
    FEATURE_RUN_ID,
    R2_IMPLEMENTATION_DIGEST,
)
from stage6_expanded_finalist import (
    FINALIST_COUNT_PER_SHARD as EXPANDED_FINALIST_COUNT_PER_SHARD,
    FINALIST_ITERATION_CAP as EXPANDED_FINALIST_ITERATION_CAP,
    FINALIST_MINIMUM_IMPROVEMENT as EXPANDED_FINALIST_MINIMUM_IMPROVEMENT,
    FINALIST_PATIENCE as EXPANDED_FINALIST_PATIENCE,
)
from stage6_expanded_layers import (
    EXPANDED_LAYER_CONTRACT_DIGEST,
    SCIENCE_IMPLEMENTATION_MEMBERS,
    SCIENCE_IMPLEMENTATION_DIGEST,
    candidate_registry as expanded_candidate_registry,
    canonical_json_bytes,
    registry_digest as expanded_registry_digest,
    seal_document,
    validate_document,
)
from stage6_expanded_tournament import (
    HALVING_STAGES as EXPANDED_HALVING_STAGES,
    TEMPERATURE_GRID,
    _evaluate_candidate as evaluate_established_candidate,
    resolve_candidate_workers as resolve_established_candidate_workers,
    stage_data_binding as established_stage_data_binding,
)
from stage6_drive_fuse_publication import create_or_verify_bytes, recover_claimed_bytes


SCOPES = ("preflop_behavior", "postflop_behavior")
SHARDS = (0, 1)
FAMILIES = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "extra_trees",
    "random_forest",
    "logistic_regression",
    "gaussian_nb",
)
EXPECTED_FAMILY_COUNTS = {
    "lightgbm": 64,
    "xgboost": 64,
    "catboost": 48,
    "hist_gradient_boosting": 32,
    "extra_trees": 24,
    "random_forest": 8,
    "logistic_regression": 8,
    "gaussian_nb": 8,
}
CANDIDATE_COUNT_PER_SCOPE = 256
CANDIDATE_COUNT_PER_SHARD = 128
REGISTRY_SEED = 6_202_612
SOURCE_GRID_LAYER = "within_hand_line_surprise"
SOURCE_GRID_DIGEST = expanded_registry_digest(SOURCE_GRID_LAYER)

# The counts and resource/early-stopping settings intentionally match the
# already-approved expanded-layer implementation.  Stage 3 reruns the final 16
# at its largest development rung and leaves all 16 available to finalist work.
_RUNG_EVALUATED_COUNTS = (128, 48, 16)
HALVING_STAGES = tuple(
    {
        **dict(stage),
        "evaluated_candidate_count": _RUNG_EVALUATED_COUNTS[index],
    }
    for index, stage in enumerate(EXPANDED_HALVING_STAGES)
)
FINALIST_COUNT_PER_SHARD = EXPANDED_FINALIST_COUNT_PER_SHARD
FINALIST_ITERATION_CAP = EXPANDED_FINALIST_ITERATION_CAP
FINALIST_PATIENCE = EXPANDED_FINALIST_PATIENCE
FINALIST_MINIMUM_IMPROVEMENT = EXPANDED_FINALIST_MINIMUM_IMPROVEMENT
TARGETED_REFINEMENT_MAX_CANDIDATES = 16

FEATURE_RELEASE_BINDING = seal_document({
    "schema_version": "stage6_behavior_challenger_feature_release_binding_v1",
    "revision": "R5",
    "feature_run_id": FEATURE_RUN_ID,
    "r2_implementation_self_sha256": R2_IMPLEMENTATION_DIGEST,
    "feature_runtime_source": {
        "relative_path": "src/stage6_expanded_feature_freeze_v2.py",
        "bytes": 44_538,
        "sha256": "060bfa2818b2b48b4dfd76212a960c882bb0aab741d8ccd1284804aded2ec293",
    },
    "feature_release_builder": {
        "relative_path": "src/stage6_expanded_feature_freeze_notebook_builder_v2.py",
        "bytes": 14_419,
        "sha256": "e2c5a108e686fa1f06eebbf63382d5f9a805b80f2634b19e4eb8604cb8baf564",
    },
    "input_package": {
        "relative_path": "inputs/POKERNOW_ASSISTANT_STAGE6_EXPANDED_FEATURE_FREEZE_INPUTS_R5.zip",
        "bytes": 130_674,
        "sha256": "3c47557099eedee0e958e2b95edc2b131e91128db6a7e2820e2a3f6ff1ab240f",
        "package_manifest_self_sha256": "b6ed941173fe978bb300613c712ae8f896c8965f5f292717d7983a4102e77c56",
        "precommit_self_sha256": "254744762c121ebd3585fd793122d05d1ce6385a47c170253d35a7c4810c4658",
    },
    "notebook": {
        "relative_path": "notebooks/12_Stage6_Expanded_Layer_Causal_Feature_Freeze_R5.ipynb",
        "bytes": 8_625,
        "sha256": "5a11497abcbabefb077c101b73bafb329140a8fadb0009ea533d16a8f8b88eec",
    },
    "static_manifest": {
        "relative_path": "manifests/stage6_notebook12_static_manifest_r5.json",
        "bytes": 1_698,
        "sha256": "ba433360edb31335a2b54c12b7f3eeb8bcba7ef724861d4d0aa07a6d8bb2ab5c",
        "self_sha256": "bb55a5c6b56bc726e528a6e491d93abcc54f0cecb3cab8c4a6546ca24a6b0da5",
    },
    "fresh_targets_opened": False,
    "qualification_authority": False,
})

DEVELOPMENT_TARGET_ROLES = (
    "development_train",
    "development_validation",
    "retrospective_calibration",
)
SELECTION_TARGET_ROLES = (
    "development_train",
    "development_validation",
)
FRESH_OR_LOCKBOX_ROLE_TOKENS = (
    "fresh",
    "lockbox",
    "holdout",
    "private_test",
    "adjudication",
)

RAW_IDENTITY_FEATURE_NAMES = frozenset({
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "actor_id",
    "opponent_id",
    "player_id",
    "canonical_player_id",
    "actor_canonical_player_id",
    "opponent_canonical_player_id",
})
PROHIBITED_MODEL_FEATURE_TOKENS = (
    # Wall-clock/order fields are permitted for splitting or causal ordering,
    # never as a fitted behavior-model input.
    "decision_utc",
    "timestamp",
    "action_timing",
    "decision_duration",
    "decision_latency",
    "elapsed_seconds",
    "inter_action_time",
    "think_time",
    "time_to_act",
    # Private, future, outcome, and current-target information.
    "private_card",
    "hole_card",
    "hero_card",
    "pocket_card",
    "future_",
    "showdown",
    "hand_outcome",
    "financial_",
    "profit",
    "net_won",
    "pot_won",
    "winner",
    "action_label",
    "target_label",
)
BEHAVIOR_MODEL_FEATURE_NAMES = tuple(established_feature_names())
ONE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
EARLY_STOP_CAP_FAMILIES = frozenset({
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "logistic_regression",
})
SUPPORTED_NOTEBOOK_ACCELERATORS = ("CPU", "GPU", "TPU")
BEHAVIOR_IMPLEMENTATION_SOURCE_MEMBERS = (
    "src/stage6_drive_fuse_publication.py",
    "src/stage6_expanded_history_acceleration_v2.py",
    "src/stage6_expanded_feature_freeze_v2.py",
    "src/stage6_notebook08_runtime.py",
    "src/stage6_notebook09_runtime.py",
    "src/stage6_notebook11_runtime.py",
    "src/stage6_incumbent_preflop_feature_exporter.ts",
    "src/stage6_incumbent_preflop_feature_packer.py",
    "src/stage6_behavior_json_publication_v2.py",
    "src/stage6_behavior_challenger_tournament_v2.py",
    "src/stage6_behavior_challenger_notebook_runtime_v2.py",
    "src/stage6_behavior_challenger_notebook_builder_v2.py",
    "src/stage6_behavior_challenger_finalist_v2.py",
    "src/stage6_behavior_challenger_fresh_context_v2.py",
    "src/stage6_behavior_challenger_fresh_adjudication_v2.py",
    "src/stage6_behavior_challenger_science_notebook_builder_v2.py",
)
BEHAVIOR_IMPLEMENTATION_MEMBERS = tuple(dict.fromkeys((
    *SCIENCE_IMPLEMENTATION_MEMBERS,
    "contracts/stage6_decision_feature_role_manifest_v1.json",
    "contracts/stage6_notebook_dag_v2.json",
    *BEHAVIOR_IMPLEMENTATION_SOURCE_MEMBERS,
)))


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def behavior_implementation_binding() -> dict[str, Any]:
    """Bind every behavior source/runtime/builder/contract and feature run."""
    stage_root = Path(__file__).resolve().parents[1]
    members = []
    for relative_path in BEHAVIOR_IMPLEMENTATION_MEMBERS:
        raw = (stage_root / relative_path).read_bytes()
        members.append({
            "relative_path": relative_path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return seal_document({
        "schema_version": "stage6_behavior_challenger_implementation_binding_v2",
        "feature_run_id": FEATURE_RUN_ID,
        "feature_release_binding": FEATURE_RELEASE_BINDING,
        "member_order": list(BEHAVIOR_IMPLEMENTATION_MEMBERS),
        "members": members,
    })


def validate_feature_release_binding_at_build_time(
    stage_root: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the exact frozen R5 producer release before packaging."""
    root = Path(__file__).resolve().parents[1] if stage_root is None else Path(stage_root)
    for key in (
        "feature_runtime_source", "feature_release_builder", "input_package",
        "notebook", "static_manifest",
    ):
        identity = FEATURE_RELEASE_BINDING[key]
        path = root / str(identity["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
        ):
            raise RuntimeError(f"frozen R5 feature release identity changed: {key}")
    package_identity = FEATURE_RELEASE_BINDING["input_package"]
    package_path = root / str(package_identity["relative_path"])
    with zipfile.ZipFile(package_path) as archive:
        package_manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        precommit = json.loads(archive.read("r5_feature_freeze_precommit.json"))
    validate_document(package_manifest)
    validate_document(precommit)
    if (
        package_manifest.get("feature_run_id") != FEATURE_RUN_ID
        or package_manifest.get("r2_implementation_self_sha256")
        != R2_IMPLEMENTATION_DIGEST
        or package_manifest.get("self_sha256")
        != package_identity["package_manifest_self_sha256"]
        or package_manifest.get("precommit_self_sha256")
        != package_identity["precommit_self_sha256"]
        or precommit.get("self_sha256") != package_identity["precommit_self_sha256"]
    ):
        raise RuntimeError("frozen R5 feature package binding changed")
    static_identity = FEATURE_RELEASE_BINDING["static_manifest"]
    static = json.loads(
        (root / str(static_identity["relative_path"])).read_text(encoding="utf-8")
    )
    validate_document(static)
    if (
        static.get("self_sha256") != static_identity["self_sha256"]
        or static.get("feature_run_id") != FEATURE_RUN_ID
        or static.get("shared_package", {}).get("sha256")
        != package_identity["sha256"]
        or static.get("notebook", {}).get("sha256")
        != FEATURE_RELEASE_BINDING["notebook"]["sha256"]
    ):
        raise RuntimeError("frozen R5 feature static manifest binding changed")
    return copy.deepcopy(FEATURE_RELEASE_BINDING)


BEHAVIOR_IMPLEMENTATION_BINDING = behavior_implementation_binding()
BEHAVIOR_IMPLEMENTATION_DIGEST = BEHAVIOR_IMPLEMENTATION_BINDING["self_sha256"]


def _require_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"unknown behavior scope: {scope}")


def _require_shard(shard: int) -> None:
    if shard not in SHARDS:
        raise ValueError(f"unknown behavior challenger shard: {shard}")


def assert_safe_model_features(columns: Sequence[str]) -> None:
    """Reject raw identity, timing, private, future, and target features."""
    violations: set[str] = set()
    for value in columns:
        name = str(value).strip().lower()
        raw_identity = (
            name in RAW_IDENTITY_FEATURE_NAMES
            or name.endswith("_canonical_player_id")
            or name.endswith("_player_identity")
            or "raw_identity" in name
        )
        prohibited_information = any(
            token in name for token in PROHIBITED_MODEL_FEATURE_TOKENS
        )
        if raw_identity or prohibited_information:
            violations.add(str(value))
    if violations:
        raise RuntimeError(
            f"prohibited behavior challenger model features: {sorted(violations)}"
        )


assert_safe_model_features(BEHAVIOR_MODEL_FEATURE_NAMES)
MODEL_FEATURE_DIGEST = _digest(BEHAVIOR_MODEL_FEATURE_NAMES)
SOURCE_DEPENDENCY_DIGEST = _digest({
    "schema": "stage6_behavior_challenger_source_dependencies_v2",
    "behavior_implementation": BEHAVIOR_IMPLEMENTATION_DIGEST,
    "feature_run_id": FEATURE_RUN_ID,
    "established_science_implementation": SCIENCE_IMPLEMENTATION_DIGEST,
    "expanded_layer_contract": EXPANDED_LAYER_CONTRACT_DIGEST,
    "source_grid_digest": SOURCE_GRID_DIGEST,
    "model_feature_digest": MODEL_FEATURE_DIGEST,
})


def scope_predicate(scope: str) -> dict[str, Any]:
    """Freeze the exact street filter and development-selection roles."""
    _require_scope(scope)
    return seal_document({
        "schema_version": "stage6_behavior_challenger_scope_predicate_v2",
        "scope": scope,
        "street_predicate": (
            {"operator": "equals", "values": ["preflop"]}
            if scope == "preflop_behavior"
            else {"operator": "in", "values": ["flop", "turn", "river"]}
        ),
        "selection_target_roles": list(SELECTION_TARGET_ROLES),
        "retrospective_calibration_used_for_selection": False,
        "fresh_targets_opened": False,
    })


def scope_data_root_binding(
    scope: str,
    *,
    feature_manifest_self_sha256: str,
    stage_data_binding_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
) -> dict[str, Any]:
    """Bind a checkpoint to its exact scoped feature and whole-hand roots."""
    _require_scope(scope)
    return seal_document({
        "schema_version": "stage6_behavior_challenger_scope_data_roots_v2",
        "scope": scope,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "feature_manifest_self_sha256": str(feature_manifest_self_sha256),
        "stage_data_binding_self_sha256": str(stage_data_binding_self_sha256),
        "whole_hand_assignment_self_sha256": str(
            whole_hand_assignment_self_sha256
        ),
    })


def assert_development_only_roles(
    roles: Sequence[str],
    *,
    fresh_targets_opened: bool = False,
) -> None:
    """Fail closed unless every target role is an approved development role."""
    if fresh_targets_opened:
        raise RuntimeError("behavior challenger search cannot run after fresh targets open")
    normalized = {str(role).strip().lower() for role in roles}
    forbidden = sorted(
        role
        for role in normalized
        if any(token in role for token in FRESH_OR_LOCKBOX_ROLE_TOKENS)
    )
    unknown = sorted(normalized - set(DEVELOPMENT_TARGET_ROLES))
    if forbidden or unknown:
        raise RuntimeError(
            "behavior challenger target roles are not development-only: "
            f"forbidden={forbidden}, unknown={unknown}"
        )


@lru_cache(maxsize=len(SCOPES))
def _frozen_registry(scope: str) -> tuple[dict[str, Any], ...]:
    _require_scope(scope)
    source_rows = expanded_candidate_registry(SOURCE_GRID_LAYER)
    rows: list[dict[str, Any]] = []
    family_ordinals = {family: 0 for family in FAMILIES}
    for source in source_rows:
        family = str(source["family"])
        if family not in family_ordinals:
            raise RuntimeError(f"unexpected established model family: {family}")
        ordinal = family_ordinals[family]
        family_ordinals[family] += 1
        parameters = copy.deepcopy(source["parameters"])
        configuration = {
            "schema": "stage6_behavior_challenger_configuration_v2",
            "scope": scope,
            "family": family,
            "parameters": parameters,
            "registry_seed": REGISTRY_SEED,
            "model_feature_digest": MODEL_FEATURE_DIGEST,
            "candidate_internal_threads": 1,
        }
        configuration_sha256 = _digest(configuration)
        rows.append({
            "candidate_id": (
                f"{scope}--{family}-{ordinal:03d}-{configuration_sha256[:12]}"
            ),
            "scope": scope,
            "family": family,
            "parameters": parameters,
            "configuration_sha256": configuration_sha256,
            "candidate_internal_threads": 1,
        })
    observed_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in FAMILIES
    }
    if (
        len(rows) != CANDIDATE_COUNT_PER_SCOPE
        or len({row["candidate_id"] for row in rows}) != CANDIDATE_COUNT_PER_SCOPE
        or observed_counts != EXPECTED_FAMILY_COUNTS
    ):
        raise RuntimeError(
            "behavior challenger registry must freeze 256 distinct candidates "
            "across the established eight-family allocation"
        )
    return tuple(rows)


def candidate_registry(scope: str) -> list[dict[str, Any]]:
    """Return a defensive copy of the deterministic 256-candidate registry."""
    return copy.deepcopy(list(_frozen_registry(scope)))


def candidate_registry_shard(scope: str, shard: int) -> list[dict[str, Any]]:
    _require_shard(shard)
    rows = [
        row
        for index, row in enumerate(candidate_registry(scope))
        if index % len(SHARDS) == shard
    ]
    if len(rows) != CANDIDATE_COUNT_PER_SHARD:
        raise RuntimeError("behavior challenger shard must contain 128 candidates")
    return rows


def registry_digest(scope: str) -> str:
    return _digest(candidate_registry(scope))


def shard_digest(scope: str, shard: int) -> str:
    return _digest(candidate_registry_shard(scope, shard))


def feature_contract(scope: str) -> dict[str, Any]:
    _require_scope(scope)
    return seal_document({
        "schema_version": "stage6_behavior_challenger_feature_contract_v2",
        "scope": scope,
        "model_feature_names": list(BEHAVIOR_MODEL_FEATURE_NAMES),
        "model_feature_digest": MODEL_FEATURE_DIGEST,
        "raw_identity_model_features": False,
        "action_timing_model_features": False,
        "private_card_model_features": False,
        "future_or_outcome_model_features": False,
        "ordering_only_columns": [
            "game_id",
            "hand_id",
            "decision_id",
            "decision_utc",
            "decision_order",
        ],
        "target_only_columns": ["action_label"],
        "causal_data_source": "established_stage6_accepted_decision_rows",
        "visible_legality_columns_included": True,
    })


def search_run_id(scope: str, shard: int) -> str:
    _require_scope(scope)
    _require_shard(shard)
    return _digest({
        "schema": "stage6_behavior_challenger_search_run_v2",
        "scope": scope,
        "shard": shard,
        "registry_digest": registry_digest(scope),
        "shard_digest": shard_digest(scope, shard),
        "source_grid_digest": SOURCE_GRID_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "feature_contract": feature_contract(scope)["self_sha256"],
        "halving_stages": HALVING_STAGES,
        "temperature_grid": TEMPERATURE_GRID,
        "finalist_count_per_shard": FINALIST_COUNT_PER_SHARD,
        "finalist_iteration_cap": FINALIST_ITERATION_CAP,
        "finalist_patience": FINALIST_PATIENCE,
        "finalist_minimum_improvement": FINALIST_MINIMUM_IMPROVEMENT,
        "candidate_internal_threads": 1,
        "fresh_targets_opened": False,
    })


def search_plan(scope: str, accelerator: str = "TPU") -> dict[str, Any]:
    """Freeze the executable development search without touching any dataset."""
    _require_scope(scope)
    assert_development_only_roles(DEVELOPMENT_TARGET_ROLES)
    return seal_document({
        "schema_version": "stage6_behavior_challenger_search_plan_v2",
        "status": "FROZEN_DEVELOPMENT_SEARCH__EXECUTION_NOT_STARTED",
        "scope": scope,
        "candidate_count": CANDIDATE_COUNT_PER_SCOPE,
        "families": list(FAMILIES),
        "family_candidate_counts": dict(EXPECTED_FAMILY_COUNTS),
        "registry_digest": registry_digest(scope),
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "scope_predicate": scope_predicate(scope),
        "shards": [
            {
                "shard": shard,
                "candidate_count": CANDIDATE_COUNT_PER_SHARD,
                "shard_digest": shard_digest(scope, shard),
                "search_run_id": search_run_id(scope, shard),
            }
            for shard in SHARDS
        ],
        "halving_stages": [dict(stage) for stage in HALVING_STAGES],
        "finalist_count_per_shard": FINALIST_COUNT_PER_SHARD,
        "finalist_iteration_cap": FINALIST_ITERATION_CAP,
        "finalist_patience": FINALIST_PATIENCE,
        "finalist_minimum_improvement": FINALIST_MINIMUM_IMPROVEMENT,
        "calibration": {
            "method": "legal_projected_temperature_selection",
            "temperature_grid": list(TEMPERATURE_GRID),
            "role": "retrospective_calibration",
            "freeze_before_fresh_target_open": True,
        },
        "whole_hand_time_split": {
            "partition_unit": "whole_game_hand_by_max_decision_utc_later_role_wins",
            "identical_nested_rungs_across_shards": True,
            "selection_target_roles": [
                "development_train",
                "development_validation",
            ],
        },
        "model_feature_contract_self_sha256": feature_contract(scope)["self_sha256"],
        "visible_legality_projection_required": True,
        "auto_workers_required": True,
        "worker_policy": worker_policy_contract(accelerator),
        "candidate_internal_threads": 1,
        "checkpoint_after_every_candidate_stage": True,
        "lockbox_access_allowed": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "assistant_recommendation_authority": False,
        "assistant_live_action_authority": False,
        "cfr_or_dcfr_dependency": False,
    })


def worker_policy_contract(accelerator: str) -> dict[str, Any]:
    """Describe host-process model work on CPU, GPU, or TPU notebook hosts."""
    normalized = str(accelerator).strip().upper()
    if normalized not in SUPPORTED_NOTEBOOK_ACCELERATORS:
        raise ValueError(
            "notebook accelerator must be one of CPU, GPU, or TPU"
        )
    return seal_document({
        "schema_version": "stage6_behavior_challenger_worker_policy_v2",
        "notebook_accelerator": normalized,
        "preferred_model_heavy_notebook_accelerator": "v6e TPU",
        "supported_notebook_accelerators": list(SUPPORTED_NOTEBOOK_ACCELERATORS),
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_tpu_acceleration": False,
        "worker_request_values": "auto_or_positive_integer",
        "auto_worker_rule": (
            "maximum_detected_usable_logical_cpus_bounded_by_"
            "unfinished_candidates_and_available_ram"
        ),
        "candidate_internal_threads": 1,
        "logical_data_lane_count_changed": False,
        "confirmed_oom_recovery": (
            "resume_exact_checkpoints_with_a_smaller_positive_explicit_worker_count"
        ),
        "oom_recovery_may_change_model_configuration": False,
    })


def smaller_worker_request_after_confirmed_oom(
    previous_live_workers: int,
    *,
    oom_confirmed: bool,
) -> str:
    """Return a smaller explicit retry only after the caller confirms an OOM."""
    previous = int(previous_live_workers)
    if not oom_confirmed:
        raise RuntimeError("a smaller worker retry requires a confirmed OOM")
    if previous <= 1:
        raise RuntimeError("cannot reduce a one-worker candidate execution")
    return str(max(1, previous // 2))


def _registry_entry(scope: str, candidate_id: str) -> dict[str, Any]:
    for entry in candidate_registry(scope):
        if entry["candidate_id"] == candidate_id:
            return entry
    raise RuntimeError(f"candidate is not in the frozen {scope} registry: {candidate_id}")


def _stage(stage_number: int) -> dict[str, Any]:
    for stage in HALVING_STAGES:
        if int(stage["stage"]) == int(stage_number):
            return dict(stage)
    raise ValueError(f"unknown behavior challenger halving stage: {stage_number}")


def model_fit_spec(
    scope: str,
    candidate_id: str,
    stage_number: int,
) -> dict[str, Any]:
    """Return auditable per-family fit controls used by the shared executor."""
    entry = _registry_entry(scope, candidate_id)
    stage = _stage(stage_number)
    family = str(entry["family"])
    parameters = copy.deepcopy(entry["parameters"])
    iteration_cap = int(stage["iteration_cap"])
    patience = int(stage["patience"])
    tolerance = float(stage["minimum_meaningful_improvement"])
    constructor: dict[str, Any] = copy.deepcopy(parameters)
    fit_control: dict[str, Any] = {}
    native_thread_parameter: str | None = None

    if family == "lightgbm":
        constructor.update({
            "objective": "multiclass",
            "num_class": 6,
            "n_estimators": iteration_cap,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
        })
        fit_control = {
            "early_stopping_rounds": patience,
            "minimum_meaningful_improvement": tolerance,
        }
        native_thread_parameter = "n_jobs"
    elif family == "xgboost":
        constructor.update({
            "objective": "multi:softprob",
            "num_class": 6,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_estimators": iteration_cap,
            "n_jobs": 1,
        })
        fit_control = {
            "early_stopping_rounds": patience,
            "minimum_meaningful_improvement": tolerance,
        }
        native_thread_parameter = "n_jobs"
    elif family == "catboost":
        constructor.update({
            "loss_function": "MultiClass",
            "iterations": iteration_cap,
            "thread_count": 1,
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": patience,
        })
        fit_control = {"use_best_model": True}
        native_thread_parameter = "thread_count"
    elif family == "hist_gradient_boosting":
        constructor.update({
            "loss": "log_loss",
            "max_iter": iteration_cap,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": patience,
            "tol": tolerance,
        })
    elif family in {"extra_trees", "random_forest"}:
        estimator_cap = 96 if stage_number == 1 else (256 if stage_number == 2 else 768)
        constructor["n_estimators"] = min(
            int(constructor["n_estimators"]), estimator_cap
        )
        constructor.update({
            "n_jobs": 1,
            "bootstrap": family == "random_forest",
        })
        native_thread_parameter = "n_jobs"
    elif family == "logistic_regression":
        constructor.update({
            "solver": "saga",
            "penalty": "elasticnet",
            "max_iter": 100 if stage_number == 1 else (400 if stage_number == 2 else 1_200),
            "tol": tolerance,
            "n_jobs": 1,
        })
        native_thread_parameter = "n_jobs"
    elif family == "gaussian_nb":
        fit_control = {"preprocessor": "standard_scaler"}
    else:  # pragma: no cover - registry construction already rejects this.
        raise RuntimeError(f"unknown behavior challenger family: {family}")

    return {
        "scope": scope,
        "candidate_id": candidate_id,
        "configuration_sha256": entry["configuration_sha256"],
        "family": family,
        "stage": stage_number,
        "constructor_parameters": constructor,
        "fit_control": fit_control,
        "candidate_internal_threads": 1,
        "native_thread_parameter": native_thread_parameter,
        "threadpoolctl_limit": 1,
        "thread_environment": dict(ONE_THREAD_ENVIRONMENT),
    }


def resolve_candidate_workers(
    requested: str,
    unfinished_candidates: int,
    arrays: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]] | None = None,
    stage: Mapping[str, Any] | None = None,
    *,
    accelerator: str = "CPU",
) -> dict[str, Any]:
    """Apply the established CPU/RAM bound with a nonzero model estimate."""
    if int(unfinished_candidates) <= 0:
        raise RuntimeError("worker resolution requires unfinished candidates")
    if entries is None or stage is None:
        raise RuntimeError(
            "worker resolution requires exact unfinished entries and stage"
        )
    if len(entries) != int(unfinished_candidates) or not entries:
        raise RuntimeError(
            "unfinished entry inventory must exactly match unfinished candidates"
        )
    policy = worker_policy_contract(accelerator)
    report = resolve_established_candidate_workers(
        requested, unfinished_candidates, arrays, entries, stage
    )
    if report.get("candidate_internal_threads") != 1:
        raise RuntimeError("established worker policy no longer enforces one thread")
    if int(report.get("estimated_model_peak_bytes", 0)) <= 0:
        raise RuntimeError("RAM admission requires a positive model peak estimate")
    return {**report, "worker_policy": policy}


def stage_data_binding(
    arrays: Mapping[str, Any],
    whole_hand_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse the established exact array and whole-hand split identity."""
    return established_stage_data_binding(arrays, whole_hand_binding)


def evaluate_candidate(
    scope: str,
    entry: Mapping[str, Any],
    paths: Mapping[str, str],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one candidate through established legality and calibration."""
    expected = _registry_entry(scope, str(entry.get("candidate_id")))
    if dict(entry) != expected:
        raise RuntimeError("candidate entry does not match the frozen behavior registry")
    assert_development_only_roles(("development_train", "development_validation"))
    report = evaluate_established_candidate(entry, paths, stage)
    if (
        report.get("configuration_sha256") != expected["configuration_sha256"]
        or report.get("candidate_internal_threads") != 1
    ):
        raise RuntimeError("established candidate executor returned an unbound result")
    return report


def _numeric_parameter_edges(
    scope: str,
    winner: Mapping[str, Any],
) -> list[str]:
    family = str(winner["family"])
    peers = [row for row in candidate_registry(scope) if row["family"] == family]
    edges: list[str] = []
    for name, value in winner["parameters"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values = sorted({
            float(row["parameters"][name])
            for row in peers
            if isinstance(row["parameters"].get(name), (int, float))
            and not isinstance(row["parameters"].get(name), bool)
        })
        if len(values) > 1 and float(value) in {values[0], values[-1]}:
            edges.append(name)
    return sorted(edges)


def _bounded_refinement_value(
    value: int | float, candidate: float, name: str
) -> int | float:
    if name in {"subsample", "colsample_bytree", "max_features", "l1_ratio"}:
        candidate = min(1.0, max(0.01, candidate))
    elif name in {
        "learning_rate", "min_child_weight", "reg_alpha", "reg_lambda",
        "l2_leaf_reg", "random_strength", "bagging_temperature",
        "l2_regularization", "C", "var_smoothing",
    }:
        candidate = max(1e-12, candidate)
    elif name in {
        "num_leaves", "min_child_samples", "max_bin", "max_leaf_nodes",
        "min_samples_leaf", "n_estimators",
    }:
        candidate = max(2.0, candidate)
    elif name in {"depth", "max_depth"}:
        candidate = 4.0 if float(value) == -1 else max(2.0, candidate)
    return int(round(candidate)) if isinstance(value, int) else float(candidate)


def _exact_refinement_candidates(
    scope: str,
    winner: Mapping[str, Any],
    edge_parameters: Sequence[str],
    cap_winner: bool,
    *,
    winner_ranking_self_sha256: str,
    data_roots: Mapping[str, Any],
) -> list[dict[str, Any]]:
    names = list(edge_parameters)
    parameters = dict(winner["parameters"])
    if cap_winner:
        for fallback in (
            "learning_rate", "reg_lambda", "l2_leaf_reg",
            "l2_regularization", "C",
        ):
            if fallback in parameters and fallback not in names:
                names.append(fallback)
                break
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        value = parameters.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        magnitude = (
            max(1.0, abs(float(value)) * 0.25)
            if isinstance(value, int)
            else max(abs(float(value)) * 0.25, 1e-12)
        )
        for direction in (-1.0, 1.0):
            adjusted = _bounded_refinement_value(
                value, float(value) + direction * magnitude, name
            )
            if adjusted == value:
                continue
            refined = dict(parameters)
            refined[name] = adjusted
            payload = {
                "schema": "stage6_behavior_targeted_refinement_candidate_v2",
                "scope": scope,
                "parent_candidate_id": winner["candidate_id"],
                "parent_configuration_sha256": winner["configuration_sha256"],
                "family": winner["family"],
                "parameters": refined,
                "adjusted_parameter": name,
                "winner_ranking_self_sha256": winner_ranking_self_sha256,
                "scope_data_root_binding_self_sha256": data_roots["self_sha256"],
                "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
                "feature_run_id": FEATURE_RUN_ID,
                "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
                "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
                "fresh_targets_opened": False,
            }
            configuration = _digest(payload)
            if configuration in seen:
                continue
            seen.add(configuration)
            variants.append({
                "candidate_id": (
                    f"{scope}--refine-{winner['family']}-{configuration[:16]}"
                ),
                "scope": scope,
                "family": winner["family"],
                "parameters": refined,
                "configuration_sha256": configuration,
                "candidate_internal_threads": 1,
                "refinement_parent_candidate_id": winner["candidate_id"],
                "refinement_adjusted_parameter": name,
            })
            if len(variants) >= TARGETED_REFINEMENT_MAX_CANDIDATES:
                return variants
    return variants


def targeted_refinement_spec(
    scope: str,
    winner_report: Mapping[str, Any],
    *,
    winner_ranking_self_sha256: str,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
    fresh_targets_opened: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed development-only edge/cap refinement decision."""
    assert_development_only_roles(
        SELECTION_TARGET_ROLES,
        fresh_targets_opened=fresh_targets_opened,
    )
    if int(winner_report.get("rank", 0)) != 1:
        raise RuntimeError("targeted refinement requires the rank-one scope winner")
    winner = _registry_entry(scope, str(winner_report.get("candidate_id")))
    if winner_report.get("configuration_sha256") != winner["configuration_sha256"]:
        raise RuntimeError("targeted refinement winner configuration is not frozen")
    edge_parameters = _numeric_parameter_edges(scope, winner)
    best_iteration = int(winner_report.get("best_iteration", 0))
    iteration_cap = int(winner_report.get("iteration_cap", FINALIST_ITERATION_CAP))
    cap_winner = (
        winner["family"] in EARLY_STOP_CAP_FAMILIES
        and best_iteration > 0
        and best_iteration >= iteration_cap
    )
    enabled = bool(edge_parameters or cap_winner)
    data_roots = scope_data_root_binding(
        scope,
        feature_manifest_self_sha256=feature_manifest_self_sha256,
        stage_data_binding_self_sha256=stage_data_binding_self_sha256,
        whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
    )
    refinement_candidates = _exact_refinement_candidates(
        scope,
        winner,
        edge_parameters,
        cap_winner,
        winner_ranking_self_sha256=winner_ranking_self_sha256,
        data_roots=data_roots,
    ) if enabled else []
    return seal_document({
        "schema_version": "stage6_behavior_targeted_refinement_spec_v2",
        "status": (
            "ENABLED_DEVELOPMENT_ONLY_EDGE_OR_CAP_WINNER"
            if enabled
            else "DISABLED_WINNER_NOT_ON_EDGE_OR_CAP"
        ),
        "scope": scope,
        "winner_candidate_id": winner["candidate_id"],
        "winner_configuration_sha256": winner["configuration_sha256"],
        "winner_family": winner["family"],
        "edge_parameter_names": edge_parameters,
        "iteration_cap_reached": cap_winner,
        "enabled": enabled,
        "maximum_refinement_candidate_count": (
            TARGETED_REFINEMENT_MAX_CANDIDATES if enabled else 0
        ),
        "parameter_source": "deterministic_local_neighborhood_of_frozen_winner",
        "winner_ranking_self_sha256": str(winner_ranking_self_sha256),
        "scope_data_root_binding": data_roots,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "target_roles": list(SELECTION_TARGET_ROLES),
        "retrospective_calibration_used_for_refinement": False,
        "refinement_candidate_count": len(refinement_candidates),
        "refinement_candidate_registry_digest": _digest(refinement_candidates),
        "refinement_candidates": refinement_candidates,
        "must_freeze_before_fresh_target_open": True,
        "fresh_targets_opened": False,
        "lockbox_access_allowed": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
    })


def checkpoint_path(
    root: Path,
    scope: str,
    shard: int,
    stage_number: int,
    candidate_id: str,
) -> Path:
    _require_scope(scope)
    _require_shard(shard)
    _stage(stage_number)
    _registry_entry(scope, candidate_id)
    return (
        root
        / scope
        / f"shard-{shard}"
        / "candidate_results"
        / f"stage-{stage_number}"
        / f"{candidate_id}.json"
    )


def build_candidate_checkpoint(
    scope: str,
    shard: int,
    stage_number: int,
    candidate_id: str,
    *,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
    environment_self_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one candidate-stage result to every identity needed for resume."""
    entry = _registry_entry(scope, candidate_id)
    if entry not in candidate_registry_shard(scope, shard):
        raise RuntimeError("candidate does not belong to the requested shard")
    _stage(stage_number)
    assert_development_only_roles(("development_train", "development_validation"))
    return seal_document({
        "schema_version": "stage6_behavior_challenger_candidate_checkpoint_v2",
        "status": "PASS_DEVELOPMENT_CANDIDATE_STAGE_COMPLETE",
        "search_run_id": search_run_id(scope, shard),
        "scope": scope,
        "shard": shard,
        "stage": stage_number,
        "candidate_id": candidate_id,
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "registry_digest": registry_digest(scope),
        "shard_digest": shard_digest(scope, shard),
        "model_feature_contract_self_sha256": feature_contract(scope)["self_sha256"],
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "scope_data_root_binding": scope_data_root_binding(
            scope,
            feature_manifest_self_sha256=feature_manifest_self_sha256,
            stage_data_binding_self_sha256=stage_data_binding_self_sha256,
            whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
        ),
        "feature_manifest_self_sha256": feature_manifest_self_sha256,
        "stage_data_binding_self_sha256": stage_data_binding_self_sha256,
        "whole_hand_assignment_self_sha256": whole_hand_assignment_self_sha256,
        "environment_self_sha256": environment_self_sha256,
        "candidate_internal_threads": 1,
        "visible_legality_projection_applied": True,
        "calibration_frozen_on_development_data": True,
        "result": copy.deepcopy(dict(result)),
        "fresh_targets_opened": False,
        "lockbox_access_allowed": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
    })


def validate_candidate_checkpoint(
    document: Mapping[str, Any],
    scope: str,
    shard: int,
    stage_number: int,
    candidate_id: str,
    *,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
    environment_self_sha256: str,
) -> None:
    validate_document(document)
    _require_shard(shard)
    _stage(stage_number)
    entry = _registry_entry(scope, candidate_id)
    if entry not in candidate_registry_shard(scope, shard):
        raise RuntimeError("candidate does not belong to the expected checkpoint shard")
    expected = {
        "schema_version": "stage6_behavior_challenger_candidate_checkpoint_v2",
        "status": "PASS_DEVELOPMENT_CANDIDATE_STAGE_COMPLETE",
        "search_run_id": search_run_id(scope, shard),
        "scope": scope,
        "shard": shard,
        "stage": stage_number,
        "candidate_id": candidate_id,
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "registry_digest": registry_digest(scope),
        "shard_digest": shard_digest(scope, shard),
        "model_feature_contract_self_sha256": feature_contract(scope)["self_sha256"],
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "scope_predicate_self_sha256": scope_predicate(scope)["self_sha256"],
        "scope_data_root_binding": scope_data_root_binding(
            scope,
            feature_manifest_self_sha256=feature_manifest_self_sha256,
            stage_data_binding_self_sha256=stage_data_binding_self_sha256,
            whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
        ),
        "feature_manifest_self_sha256": feature_manifest_self_sha256,
        "stage_data_binding_self_sha256": stage_data_binding_self_sha256,
        "whole_hand_assignment_self_sha256": whole_hand_assignment_self_sha256,
        "environment_self_sha256": environment_self_sha256,
        "candidate_internal_threads": 1,
        "visible_legality_projection_applied": True,
        "calibration_frozen_on_development_data": True,
        "fresh_targets_opened": False,
        "lockbox_access_allowed": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
    }
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"behavior challenger checkpoint identity mismatch: {mismatches}")
    result = document.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("behavior challenger checkpoint result is not a mapping")
    nested_expected = {
        "candidate_id": candidate_id,
        "family": entry["family"],
        "configuration_sha256": entry["configuration_sha256"],
        "parameters": entry["parameters"],
        "stage": int(stage_number),
        "candidate_internal_threads": 1,
    }
    nested_mismatches = {
        key: (result.get(key), value)
        for key, value in nested_expected.items()
        if result.get(key) != value
    }
    if nested_mismatches:
        raise RuntimeError(
            "behavior challenger nested result identity mismatch: "
            f"{nested_mismatches}"
        )


def write_candidate_checkpoint(path: Path, document: Mapping[str, Any]) -> None:
    """Create exclusively, or accept only byte-identical sealed readback."""
    validate_document(document)
    raw = canonical_json_bytes(document) + b"\n"
    try:
        create_or_verify_bytes(
            path, raw, owner_run_id=str(document["self_sha256"])
        )
    except (FileExistsError, RuntimeError) as error:
        raise RuntimeError(
            "conflicting behavior challenger checkpoint exists"
        ) from error


def load_candidate_checkpoint(
    path: Path,
    scope: str,
    shard: int,
    stage_number: int,
    candidate_id: str,
    *,
    stage_data_binding_self_sha256: str,
    feature_manifest_self_sha256: str,
    whole_hand_assignment_self_sha256: str,
    environment_self_sha256: str,
) -> dict[str, Any] | None:
    recover_claimed_bytes(path)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_candidate_checkpoint(
        document,
        scope,
        shard,
        stage_number,
        candidate_id,
        stage_data_binding_self_sha256=stage_data_binding_self_sha256,
        feature_manifest_self_sha256=feature_manifest_self_sha256,
        whole_hand_assignment_self_sha256=whole_hand_assignment_self_sha256,
        environment_self_sha256=environment_self_sha256,
    )
    return document


__all__ = [
    "BEHAVIOR_IMPLEMENTATION_BINDING",
    "BEHAVIOR_IMPLEMENTATION_DIGEST",
    "BEHAVIOR_IMPLEMENTATION_MEMBERS",
    "BEHAVIOR_MODEL_FEATURE_NAMES",
    "CANDIDATE_COUNT_PER_SCOPE",
    "CANDIDATE_COUNT_PER_SHARD",
    "EXPECTED_FAMILY_COUNTS",
    "FAMILIES",
    "FEATURE_RELEASE_BINDING",
    "FINALIST_COUNT_PER_SHARD",
    "FINALIST_ITERATION_CAP",
    "FINALIST_MINIMUM_IMPROVEMENT",
    "FINALIST_PATIENCE",
    "HALVING_STAGES",
    "SCOPES",
    "SELECTION_TARGET_ROLES",
    "SOURCE_DEPENDENCY_DIGEST",
    "SHARDS",
    "assert_development_only_roles",
    "assert_safe_model_features",
    "build_candidate_checkpoint",
    "candidate_registry",
    "candidate_registry_shard",
    "checkpoint_path",
    "evaluate_candidate",
    "feature_contract",
    "load_candidate_checkpoint",
    "model_fit_spec",
    "registry_digest",
    "resolve_candidate_workers",
    "scope_data_root_binding",
    "scope_predicate",
    "search_plan",
    "search_run_id",
    "shard_digest",
    "stage_data_binding",
    "smaller_worker_request_after_confirmed_oom",
    "targeted_refinement_spec",
    "validate_candidate_checkpoint",
    "validate_feature_release_binding_at_build_time",
    "write_candidate_checkpoint",
    "worker_policy_contract",
]
