from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import heapq
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from stage6_causal_profile_snapshot import build_reference_causal_snapshots
from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    sha256_file,
    validate_document_self_hash,
)
from stage6_data_oracle import REQUIRED_COLUMNS as DECISION_OUTPUT_COLUMNS
from stage6_decision_row_builder import (
    ALLOWED_SOURCE_QUARANTINE_REASONS,
    ROW_ID_WIDTH,
    build_decision_row,
    build_feature_role_manifest,
    classify_source_quarantine_reason,
)
from stage6_source_quarantine_scan import (
    ACCEPTED_ROOT_ALGORITHM,
    QUARANTINE_ROOT_ALGORITHM,
    _update_accepted_root,
    _update_quarantine_root,
)
from stage6_joined_data_oracle import audit_joined_decision_profiles
from stage6_notebook01_common import (
    OwnedRegularFileAuthority,
    _attach_retry_owner,
    assert_active_run_root_identity,
    atomic_publish_owned_regular_file,
    create_exclusive_regular_file,
    create_exclusive_temporary_file,
    create_owned_staging_directory,
    ensure_durable_directory,
    exclusive_parent_run_lease,
    fsync_directory,
    fsync_regular_file,
    move_verified_regular_file,
    open_identity_bound_regular_file,
    publish_owned_staging_directory,
    recursive_stage6_code_identities,
    regular_file_identity,
    retain_owned_staging_directory,
    remove_verified_regular_file,
    remove_current_run_stale_staging_directories,
    remove_owned_staging_directory,
    verify_owned_staging_directory,
)
from stage6_split_assignment_builder import build_or_resume_split_assignments
from stage6_worker_policy import (
    CHECKPOINT_PATTERN,
    build_lane_plan,
    iter_unfinished_lane_requests,
    worker_resolution_report,
)


LANE_COUNT = 48
REQUESTED_WORKERS_DEFAULT = "auto"
BASE_SEED = 610_601
STAGE_ID = "stage6-notebook01-causal-materialization-runtime-neutral-v2"
EXPECTED_DECISION_ROWS = 2_334_700
EXPECTED_OBSERVATION_ROWS = 1_301_170
PARTITION_BUFFER_ROWS = 1_024
PROGRESS_INTERVAL_ROWS = 100_000
PROGRESS_HEARTBEAT_SECONDS = 30
MINIMUM_PARTITION_DISK_BYTES = 12 * 1024**3
MINIMUM_PARENT_RESERVE_BYTES = 2 * 1024**3
ESTIMATED_WORKER_RAM_BYTES = 768 * 1024**2
MAX_CHECKPOINT_JSON_BYTES = 8 * 1024**2

ACTION_RELATIVE = Path("data/exports/ml/action-state-decision-rows.csv")
IDENTITY_RELATIVE = Path(
    "data/exports/ml/stage6-action-decision-time-identity.csv"
)
OBSERVATION_RELATIVE = Path(
    "data/exports/ml/stage6-hand-profile-observations.csv"
)
REGISTRY_RELATIVE = Path("contracts/stage6_profile_metric_registry_hold_v1.json")
DATA_CONTRACT_RELATIVE = Path("contracts/stage6_data_contract_v1.json")
PROFILE_CONTRACT_RELATIVE = Path("contracts/stage6_player_profile_contract_v1.json")
FEATURE_MANIFEST_RELATIVE = Path(
    "contracts/stage6_decision_feature_role_manifest_v1.json"
)
QUARANTINE_CONTRACT_RELATIVE = Path(
    "contracts/stage6_source_quarantine_contract_r32.json"
)
QUARANTINE_EVIDENCE_RELATIVE = Path(
    "evidence/stage6_notebook01_source_quarantine_scan_2026-08-24_r32.json"
)
WORKER_CONTRACT_RELATIVE = Path("contracts/stage6_worker_policy_v3.json")
SPLIT_CONTRACT_RELATIVE = Path("contracts/stage6_split_contract_v1.json")

MATERIALIZER_CODE_RELATIVES = (
    Path("src/stage6_notebook01_materializer.py"),
    Path("src/stage6_notebook01_parallel_parity.py"),
    Path("src/stage6_notebook01_reference_validation.py"),
    Path("src/stage6_source_quarantine_scan.py"),
)

ACTION_PARTITION_FIELDS = (
    "dataset_version",
    "hand_action_id",
    "hand_id",
    "game_id",
    "street",
    "board_texture_present",
    "board_card_count",
    "board_high_card_value",
    "board_pair_level",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "board_connectedness_score",
    "big_blind_raw",
    "legal_mask_quality",
    "can_fold",
    "can_check",
    "can_call",
    "can_bet",
    "can_raise",
    "can_all_in",
    "legal_min_bet_to_raw",
    "legal_min_raise_to_raw",
    "legal_max_bet_raise_to_raw",
    "active_opponent_canonical_player_ids_before",
    "active_opponent_count_before",
    "sequence_context_quality",
    "street_action_count_before",
    "street_aggressive_action_count_before",
    "street_call_count_before",
    "street_check_count_before",
    "street_fold_count_before",
    "street_prior_actor_action_count",
    "street_prior_actor_bet_raise_count",
    "street_last_aggressor_is_actor",
    "facing_bet_or_raise",
    "check_raise_opportunity",
    "facing_cbet",
    "facing_donk_lead",
    "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity",
    "turn_barrel_opportunity",
    "river_barrel_opportunity",
    "preflop_limper_count_before",
    "preflop_caller_count_before",
    "preflop_raise_count_before",
    "preflop_prior_entry_count_before",
    "preflop_facing_open_raise",
    "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity",
    "preflop_limp_opportunity",
    "preflop_actor_has_limped_before",
    "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before",
    "pokernow_player_id",
    "seat",
    "normalized_type",
    "action_label",
    "pot_before_raw",
    "stack_before_raw",
    "amount_to_call_raw",
    "street_contribution_before_raw",
    "street_contribution_after_raw",
    "is_fold",
    "is_check",
    "is_call",
    "is_bet_or_raise",
    "is_all_in",
    "effective_stack_before_bb",
    "spr_before",
    "active_player_count_before",
    "players_with_chips_before",
)

IDENTITY_PARTITION_STRING_FIELDS = (
    "decision_utc",
    "actor_pokernow_player_id",
    "canonical_group_player_id",
    "identity_grouping_status",
)

PARTITION_DECISION_STRING_FIELDS = (
    *ACTION_PARTITION_FIELDS,
    *IDENTITY_PARTITION_STRING_FIELDS,
)
PARTITION_DECISION_INTEGER_FIELDS = (
    "source_row_ordinal",
    "event_index",
    "decision_epoch_ms",
)
OPTIONAL_DECISION_PARTITION_STRING_FIELDS = {
    "board_card_count",
    "board_high_card_value",
    "board_pair_level",
    "board_connectedness_score",
    "legal_min_bet_to_raw",
    "legal_min_raise_to_raw",
    "legal_max_bet_raise_to_raw",
    "effective_stack_before_bb",
    "spr_before",
}

OBSERVATION_STRING_FIELDS = (
    "hand_id",
    "game_id",
    "hand_completed_utc",
    "pokernow_player_id",
    "canonical_group_player_id",
)

DECISION_BOOLEAN_FIELDS = {
    "legal_fold",
    "legal_check",
    "legal_call",
    "legal_bet",
    "legal_raise",
    "legal_all_in",
    "short_all_in_call",
    "future_reveal_used",
    "post_decision_outcome_used",
    "private_card_features_used",
    "board_texture_present",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "street_last_aggressor_is_actor",
    "facing_bet_or_raise",
    "check_raise_opportunity",
    "facing_cbet",
    "facing_donk_lead",
    "missed_cbet_probe_opportunity",
    "delayed_cbet_opportunity",
    "turn_barrel_opportunity",
    "river_barrel_opportunity",
    "preflop_facing_open_raise",
    "preflop_facing_three_bet_or_more",
    "preflop_squeeze_opportunity",
    "preflop_limp_opportunity",
    "preflop_actor_has_limped_before",
    "preflop_actor_has_called_before",
    "preflop_actor_has_raised_before",
}
DECISION_INTEGER_FIELDS = {
    "decision_order",
    "board_card_count",
    "board_high_card_value",
    "active_opponent_count_before",
    "street_action_count_before",
    "street_aggressive_action_count_before",
    "street_call_count_before",
    "street_check_count_before",
    "street_fold_count_before",
    "street_prior_actor_action_count",
    "street_prior_actor_bet_raise_count",
    "preflop_limper_count_before",
    "preflop_caller_count_before",
    "preflop_raise_count_before",
    "preflop_prior_entry_count_before",
    "seat",
    "active_player_count_before",
    "players_with_chips_before",
}
DECISION_NULLABLE_INTEGER_FIELDS = {
    "board_card_count",
    "board_high_card_value",
}
DECISION_FLOAT_FIELDS = {
    "board_connectedness_score",
    "amount_to_call_bb",
    "pot_before_bb",
    "actor_committed_before_bb",
    "actor_committed_after_bb",
    "actor_stack_before_bb",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
    "action_to_bb",
    "effective_stack_before_bb",
    "spr_before",
}
DECISION_NULLABLE_FLOAT_FIELDS = {
    "board_connectedness_score",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
    "action_to_bb",
    "effective_stack_before_bb",
    "spr_before",
}

QUARANTINE_COLUMNS = (
    "source_row_ordinal",
    "decision_id",
    "game_id",
    "hand_id",
    "decision_order",
    "decision_utc",
    "canonical_player_id",
    "reason",
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


def _canonical_relative(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("/") or ".." in path.parts or ":" in value:
        raise Stage6ContractError(f"path is not canonical and relative: {value}")
    return value


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    assert_active_run_root_identity(path)
    ensure_durable_directory(path.parent)
    payload = json.dumps(
        dict(document), indent=2, sort_keys=True, ensure_ascii=True
    ) + "\n"
    assert_active_run_root_identity(path)
    descriptor, temporary_path, created = create_exclusive_temporary_file(
        path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    except BaseException as primary_error:
        try:
            remove_verified_regular_file(
                path.parent,
                temporary_path,
                expected_identity=(int(created.st_dev), int(created.st_ino)),
            )
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"failed JSON temporary cleanup also failed: {cleanup_error}"
            )
        raise

    def validate_json(reader: Any) -> None:
        if reader.read().decode("utf-8", errors="strict") != payload:
            raise Stage6ContractError(f"durable JSON readback mismatch: {path}")

    atomic_publish_owned_regular_file(
        temporary_path,
        path,
        expected_creation_identity=(int(created.st_dev), int(created.st_ino)),
        directory_fsync=fsync_directory,
        validate=validate_json,
    )


def _strict_json_object_with_identity(
    path: Path,
) -> tuple[dict[str, Any], tuple[int, int]]:
    before = os.lstat(path)
    if path.is_symlink() or not path.is_file():
        raise Stage6ContractError(f"JSON authority is not a regular file: {path}")
    document = _strict_json_object(path)
    after = os.lstat(path)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise Stage6ContractError(f"JSON authority changed while read: {path}")
    return document, (int(after.st_dev), int(after.st_ino))


def _run_lock_interleave_hook(_event: str, _path: Path) -> None:
    """Intentional no-op seam for deterministic run-lease regressions."""


@contextmanager
def _exclusive_run_lock(run_root: Path, *, run_identity: str):
    try:
        import psutil
    except ImportError as error:
        raise Stage6ContractError("psutil is required for run locking") from error
    import socket

    root = Path(os.path.abspath(run_root))
    lock_path = root / "materialization.lock"
    stale_root = root / "stale_locks"
    hostname = socket.gethostname()
    process = psutil.Process()
    token = uuid.uuid4().hex
    document: dict[str, Any] = {
        "schema_version": "stage6_notebook01_run_lock_v2",
        "run_identity": run_identity,
        "hostname": hostname,
        "pid": os.getpid(),
        "process_create_time": process.create_time(),
        "token": token,
        "authority": "OS_BACKED_RUN_ROOT_LEASE_PLUS_RETAINED_FILE_IDENTITY",
    }
    document["self_sha256"] = document_self_sha256(document)
    payload = json.dumps(document, sort_keys=True, ensure_ascii=True) + "\n"

    with exclusive_parent_run_lease(root) as run_authority:
        _run_lock_interleave_hook("after_parent_lease", root)
        run_authority.assert_path_identity()
        while os.path.lexists(lock_path):
            run_authority.assert_path_identity()
            existing: dict[str, Any] | None = None
            existing_identity: tuple[int, int] | None = None
            for _ in range(40):
                try:
                    candidate, identity = _strict_json_object_with_identity(lock_path)
                    validate_document_self_hash(candidate)
                    existing = candidate
                    existing_identity = identity
                    break
                except (FileNotFoundError, json.JSONDecodeError, Stage6ContractError):
                    time.sleep(0.05)
            run_authority.assert_path_identity()
            ensure_durable_directory(stale_root)
            if existing is None or existing_identity is None:
                invalid_stat = os.lstat(lock_path)
                invalid_stale_path = stale_root / (
                    f"materialization-invalid-{uuid.uuid4().hex}.lock"
                )
                _run_lock_interleave_hook("before_invalid_stale_move", lock_path)
                move_verified_regular_file(
                    root,
                    lock_path,
                    invalid_stale_path,
                    expected_identity=(
                        int(invalid_stat.st_dev),
                        int(invalid_stat.st_ino),
                    ),
                )
                continue
            active = existing.get("hostname") != hostname
            if not active:
                try:
                    existing_process = psutil.Process(int(existing["pid"]))
                    active = (
                        existing_process.is_running()
                        and abs(
                            existing_process.create_time()
                            - float(existing["process_create_time"])
                        )
                        < 1e-6
                    )
                except (KeyError, TypeError, ValueError, psutil.Error):
                    active = False
            if active:
                raise Stage6ContractError(
                    "an active Notebook 01 materialization already owns this run"
                )
            stale_path = stale_root / (
                f"materialization-{existing.get('token', 'unknown')}.lock"
            )
            _run_lock_interleave_hook("before_stale_move", lock_path)
            move_verified_regular_file(
                root,
                lock_path,
                stale_path,
                expected_identity=existing_identity,
            )

        _run_lock_interleave_hook("before_lock_creation", root)
        run_authority.assert_path_identity()
        ensure_durable_directory(stale_root)
        run_authority.assert_path_identity()
        descriptor, temporary_path, _ = create_exclusive_temporary_file(
            root, prefix=".materialization.lock.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        run_authority.assert_path_identity()
        temporary_stat = os.lstat(temporary_path)
        authority = OwnedRegularFileAuthority(
            root,
            temporary_path,
            expected_identity=(
                int(temporary_stat.st_dev),
                int(temporary_stat.st_ino),
            ),
        )
        with authority:
            try:
                authority.flush()
                authority.rename_to(lock_path)
                fsync_directory(root)
                if os.name == "nt":
                    authority.close()
                try:
                    observed, observed_identity = _strict_json_object_with_identity(
                        lock_path
                    )
                finally:
                    if os.name == "nt":
                        authority.reopen()
                if observed != document or observed_identity != authority.identity:
                    raise Stage6ContractError("run lock durable readback mismatch")
            except BaseException as primary_error:
                try:
                    authority.assert_path_identity()
                except BaseException:
                    pass
                else:
                    try:
                        authority.delete()
                        fsync_directory(root)
                    except BaseException as cleanup_error:
                        primary_error.add_note(
                            f"failed run-lock setup cleanup also failed: {cleanup_error}"
                        )
                raise

            body_error: BaseException | None = None
            try:
                _run_lock_interleave_hook("before_body", root)
                run_authority.assert_path_identity()
                yield document
                run_authority.assert_path_identity()
            except BaseException as error:
                body_error = error
                raise
            finally:
                completed = stale_root / f"materialization-completed-{token}.lock"
                try:
                    _run_lock_interleave_hook("before_cleanup", lock_path)
                    authority.rename_to(completed)
                    fsync_directory(root)
                    fsync_directory(stale_root)
                except BaseException as cleanup_error:
                    if body_error is None:
                        raise
                    body_error.add_note(
                        f"identity-bound run-lock cleanup also failed: {cleanup_error}"
                    )
                    _attach_retry_owner(body_error, authority)


def _file_identity(path: Path, *, relative_path: Path) -> dict[str, Any]:
    return regular_file_identity(
        path, relative_path=Path(_canonical_relative(relative_path))
    )


def _verify_identity_bound_parquet(
    run_root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
    expected_schema: Any | None = None,
) -> None:
    import pyarrow.parquet as pq

    with open_identity_bound_regular_file(
        run_root,
        path,
        expected_bytes=identity["bytes"],
        expected_sha256=identity["sha256"],
    ) as reader:
        parquet = pq.ParquetFile(reader)
        if parquet.metadata.num_rows != identity["row_count"]:
            raise Stage6ContractError(f"{label} row-count receipt mismatch")
        if expected_schema is not None and not parquet.schema_arrow.equals(
            expected_schema,
            check_metadata=False,
        ):
            raise Stage6ContractError(f"{label} schema receipt mismatch")


def _read_identity_bound_parquet_rows(
    run_root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    with open_identity_bound_regular_file(
        run_root,
        path,
        expected_bytes=identity["bytes"],
        expected_sha256=identity["sha256"],
    ) as reader:
        parquet = pq.ParquetFile(reader)
        if parquet.metadata.num_rows != identity["row_count"]:
            raise Stage6ContractError(f"{label} row-count receipt mismatch")
        return parquet.read(use_threads=False).to_pylist()


def _iter_identity_bound_parquet_rows(
    run_root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
    batch_size: int = 1_024,
) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    with open_identity_bound_regular_file(
        run_root,
        path,
        expected_bytes=identity["bytes"],
        expected_sha256=identity["sha256"],
    ) as reader:
        parquet = pq.ParquetFile(reader)
        if parquet.metadata.num_rows != identity["row_count"]:
            raise Stage6ContractError(f"{label} row-count receipt mismatch")
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            use_threads=False,
        ):
            yield from batch.to_pylist()


def _resource_snapshot(path: Path) -> dict[str, int]:
    try:
        import psutil
    except ImportError as error:
        raise Stage6ContractError("psutil is required for resource admission") from error
    usage = shutil.disk_usage(path)
    process = psutil.Process()
    return {
        "rss_bytes": process.memory_info().rss,
        "available_system_ram_bytes": psutil.virtual_memory().available,
        "available_disk_bytes": usage.free,
    }


def _progress(
    *,
    phase: str,
    completed: int,
    total: int,
    started: float,
    resource_root: Path,
    active: int = 1,
    queued: int | None = None,
    failed: int = 0,
    checkpoint_identity: str | None = None,
    cleanup_state: str = "not_started",
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / throughput if throughput > 0 else None
    resources = _resource_snapshot(resource_root)
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_materialization_progress",
                "current_phase": phase,
                "active": active,
                "queued": remaining if queued is None else queued,
                "completed": completed,
                "remaining": remaining,
                "failed": failed,
                "total": total,
                "throughput_work_units_per_second": throughput,
                "eta_seconds": eta,
                "checkpoint_identity": checkpoint_identity,
                "cleanup_state": cleanup_state,
                **resources,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def stable_lane_id(canonical_player_id: str, *, lane_count: int = LANE_COUNT) -> str:
    if (
        type(canonical_player_id) is not str
        or not canonical_player_id
        or canonical_player_id != canonical_player_id.strip()
    ):
        raise Stage6ContractError("canonical player id is not canonical text")
    if type(lane_count) is not int or lane_count <= 0:
        raise Stage6ContractError("lane_count must be a positive integer")
    digest = hashlib.sha256(canonical_player_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % lane_count
    return f"lane-{index:02d}"


def _lane_ids() -> list[str]:
    return [f"lane-{index:02d}" for index in range(LANE_COUNT)]


def _load_contracts(stage6_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "data_contract": DATA_CONTRACT_RELATIVE,
        "profile_contract": PROFILE_CONTRACT_RELATIVE,
        "registry": REGISTRY_RELATIVE,
        "feature_manifest": FEATURE_MANIFEST_RELATIVE,
        "quarantine_contract": QUARANTINE_CONTRACT_RELATIVE,
        "quarantine_evidence": QUARANTINE_EVIDENCE_RELATIVE,
        "worker_contract": WORKER_CONTRACT_RELATIVE,
        "split_contract": SPLIT_CONTRACT_RELATIVE,
    }
    loaded = {
        name: _strict_json_object(stage6_root / relative)
        for name, relative in paths.items()
    }
    for name in (
        "data_contract",
        "profile_contract",
        "registry",
        "feature_manifest",
        "quarantine_contract",
        "quarantine_evidence",
        "worker_contract",
        "split_contract",
    ):
        validate_document_self_hash(loaded[name])
    if tuple(loaded["data_contract"]["required_decision_columns"]) != tuple(
        DECISION_OUTPUT_COLUMNS
    ):
        raise Stage6ContractError("data contract differs from code-pinned decision schema")
    if loaded["feature_manifest"] != build_feature_role_manifest():
        raise Stage6ContractError("feature-role manifest differs from code-pinned roles")
    quarantine = loaded["quarantine_contract"]
    evidence = loaded["quarantine_evidence"]
    evidence_reason_counts = {
        reason: row["row_count"]
        for reason, row in evidence.get("quarantine_reasons", {}).items()
    }
    evidence_reason_roots = {
        reason: row["row_identity_root"]
        for reason, row in evidence.get("quarantine_reasons", {}).items()
    }
    if (
        quarantine.get("classifier_scan_evidence_relative_path")
        != QUARANTINE_EVIDENCE_RELATIVE.as_posix()
        or quarantine.get("classifier_scan_evidence_self_sha256")
        != evidence.get("self_sha256")
        or evidence.get("status")
        != "PASS_EXACT_PRODUCTION_CLASSIFIER_SOURCE_CONSERVATION"
        or evidence.get("source_work_unit_conservation_verified") is not True
        or evidence.get("reason_counts_sum_to_quarantine_count") is not True
        or quarantine.get("source_relative_path")
        != evidence.get("source_relative_path")
        or quarantine.get("source_bytes") != evidence.get("source_bytes")
        or quarantine.get("source_sha256") != evidence.get("source_sha256")
        or quarantine.get("source_row_count") != evidence.get("source_row_count")
        or quarantine.get("classifier_function")
        != evidence.get("classifier_function")
        or quarantine.get("classifier_relative_path")
        != evidence.get("classifier_relative_path")
        or quarantine.get("classifier_sha256")
        != evidence.get("classifier_sha256")
        or quarantine.get("accepted_decision_row_count")
        != evidence.get("accepted_decision_row_count")
        or quarantine.get("quarantined_decision_row_count")
        != evidence.get("quarantined_decision_row_count")
        or quarantine.get("accepted_row_identity_root_algorithm")
        != evidence.get("accepted_row_identity_root_algorithm")
        or quarantine.get("accepted_row_identity_root")
        != evidence.get("accepted_row_identity_root")
        or quarantine.get("quarantine_row_identity_root_algorithm")
        != evidence.get("quarantine_row_identity_root_algorithm")
        or quarantine.get("quarantine_row_identity_root")
        != evidence.get("quarantine_row_identity_root")
        or quarantine.get("quarantine_reason_counts") != evidence_reason_counts
        or quarantine.get("quarantine_reason_row_identity_roots")
        != evidence_reason_roots
        or set(evidence_reason_counts) != set(ALLOWED_SOURCE_QUARANTINE_REASONS)
        or evidence.get("accepted_decision_row_count", -1)
        + evidence.get("quarantined_decision_row_count", -1)
        != evidence.get("source_row_count")
        or sha256_file(stage6_root / evidence["classifier_relative_path"])
        != evidence.get("classifier_sha256")
        or sha256_file(stage6_root / evidence["scanner_relative_path"])
        != evidence.get("scanner_sha256")
    ):
        raise Stage6ContractError(
            "source quarantine contract/evidence production-classifier parity is invalid"
        )
    worker = loaded["worker_contract"]
    worker_resource = worker.get("resource_preflight", {})
    if (
        worker.get("schema_version") != "stage6_worker_policy_v3"
        or worker.get("accepted_runtime_types") != ["cpu", "gpu", "tpu"]
        or worker.get("runtime_type_restriction") is not None
        or worker.get("accelerator_required") is not False
        or worker.get("high_ram_runtime_class_required") is not False
        or worker.get("requested_workers_default") != REQUESTED_WORKERS_DEFAULT
        or worker.get("logical_lane_count") != LANE_COUNT
        or worker.get("inner_numeric_threads_per_worker") != 1
        or worker.get("live_worker_resolution_must_be_reported") is not True
        or worker.get("runtime_name_clipping_forbidden") is not True
        or worker.get("logical_work_units_must_not_be_reduced") is not True
        or worker.get("worker_request_is_restart_mutable_and_excluded_from_scientific_run_identity")
        is not True
        or worker.get("bounded_one_worker_vs_resolved_parallel_worker_parity_required")
        is not True
        or worker.get("exclusive_parent_run_lock_required") is not True
        or worker.get("same_host_stale_lock_recovery_required") is not True
        or worker_resource.get("estimated_peak_ram_bytes_per_worker")
        != ESTIMATED_WORKER_RAM_BYTES
        or worker_resource.get("minimum_parent_reserve_bytes")
        != MINIMUM_PARENT_RESERVE_BYTES
        or worker_resource.get("minimum_available_disk_bytes")
        != MINIMUM_PARTITION_DISK_BYTES
    ):
        raise Stage6ContractError("runtime-neutral 48-lane worker contract is invalid")
    return loaded


def _source_identities(stage6_root: Path) -> dict[str, dict[str, Any]]:
    relatives = (
        ACTION_RELATIVE,
        IDENTITY_RELATIVE,
        OBSERVATION_RELATIVE,
        DATA_CONTRACT_RELATIVE,
        PROFILE_CONTRACT_RELATIVE,
        REGISTRY_RELATIVE,
        FEATURE_MANIFEST_RELATIVE,
        QUARANTINE_CONTRACT_RELATIVE,
        QUARANTINE_EVIDENCE_RELATIVE,
        WORKER_CONTRACT_RELATIVE,
        SPLIT_CONTRACT_RELATIVE,
    )
    return {
        relative.as_posix(): _file_identity(
            stage6_root / relative, relative_path=relative
        )
        for relative in relatives
    }


def _code_identities(stage6_root: Path) -> dict[str, dict[str, Any]]:
    return recursive_stage6_code_identities(
        stage6_root, MATERIALIZER_CODE_RELATIVES
    )


def _run_binding(
    *,
    source_identities: Mapping[str, Any],
    code_identities: Mapping[str, Any],
    fresh_custody_binding: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if fresh_custody_binding is not None:
        expected_fields = {
            "schema_version",
            "aggregate_run_id",
            "aggregate_freeze_manifest_self_sha256",
            "fresh_custody_request_self_sha256",
        }
        if (
            set(fresh_custody_binding) != expected_fields
            or fresh_custody_binding.get("schema_version")
            != "stage6_notebook01_fresh_custody_binding_v1"
            or any(
                CHECKPOINT_PATTERN.fullmatch(str(fresh_custody_binding[field]))
                is None
                for field in expected_fields - {"schema_version"}
            )
        ):
            raise Stage6ContractError("Notebook 01 fresh custody binding is invalid")
    binding = {
        "schema_version": (
            "stage6_notebook01_materialization_run_binding_v3"
            if fresh_custody_binding is not None
            else "stage6_notebook01_materialization_run_binding_v2"
        ),
        "stage_id": STAGE_ID,
        "lane_count": LANE_COUNT,
        "worker_request_bound_to_scientific_identity": False,
        "base_seed": BASE_SEED,
        "partition_algorithm": "sha256_canonical_player_first_8_bytes_mod_48_v1",
        "source_identities": dict(source_identities),
        "code_identities": dict(code_identities),
    }
    if fresh_custody_binding is not None:
        binding["fresh_custody_binding"] = dict(fresh_custody_binding)
    return hashlib.sha256(canonical_json_bytes(binding)).hexdigest(), binding


def resolve_fresh_custody_binding(
    stage_drive_root: Path,
    requested_aggregate_run_id: str = "auto",
) -> dict[str, str]:
    stage_drive_root = stage_drive_root.resolve()
    if "OneDrive" in str(stage_drive_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    requested = str(requested_aggregate_run_id).strip()
    if requested != "auto" and CHECKPOINT_PATTERN.fullmatch(requested) is None:
        raise Stage6ContractError(
            "fresh custody aggregate run ID must be auto or 64 lowercase hex characters"
        )
    custody_sources = (
        (
            "expanded-layer",
            stage_drive_root / "runtime_state" / "expanded_layers" / "aggregate_freezes",
            "stage6_expanded_aggregate_freeze_manifest_v1",
            "PASS_CANDIDATES_FROZEN_AWAITING_FRESH_DATA",
            "stage6_expanded_fresh_custody_request_v1",
        ),
        (
            "behavior-challenger",
            stage_drive_root
            / "runtime_state"
            / "behavior_challenger_v2"
            / "aggregate_freezes",
            "stage6_behavior_challenger_aggregate_freeze_manifest_v3",
            "PASS_CANDIDATES_AND_CLAIMS_FROZEN_AWAITING_FRESH_DATA",
            "stage6_behavior_challenger_fresh_custody_request_v3",
        ),
    )
    candidates = []
    roots = []
    for (
        source_name,
        parent,
        freeze_schema,
        freeze_status,
        custody_schema,
    ) in custody_sources:
        if not parent.is_dir():
            continue
        selected_roots = (
            [parent / requested]
            if requested != "auto"
            else sorted(path for path in parent.iterdir() if path.is_dir())
        )
        roots.extend(
            (
                source_name,
                root,
                freeze_schema,
                freeze_status,
                custody_schema,
            )
            for root in selected_roots
        )
    for (
        source_name,
        root,
        freeze_schema,
        freeze_status,
        custody_schema,
    ) in roots:
        freeze_path = root / "aggregate-freeze-manifest.json"
        custody_path = root / "fresh-custody-request.json"
        if not freeze_path.is_file() or not custody_path.is_file():
            continue
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        custody = json.loads(custody_path.read_text(encoding="utf-8"))
        validate_document_self_hash(freeze)
        validate_document_self_hash(custody)
        if (
            freeze.get("schema_version") != freeze_schema
            or freeze.get("status") != freeze_status
            or freeze.get("aggregate_run_id") != root.name
            or freeze.get("fresh_targets_opened") is not False
            or custody.get("schema_version") != custody_schema
            or custody.get("status")
            != "FROZEN_AWAITING_ONE_NEW_NOTEBOOK01_MATERIALIZATION"
            or custody.get("aggregate_run_id") != root.name
            or custody.get("aggregate_freeze_manifest_self_sha256")
            != freeze["self_sha256"]
            or custody.get("one_shot_adjudication") is not True
            or custody.get("fresh_targets_opened") is not False
        ):
            raise Stage6ContractError(
                f"{source_name} fresh custody is invalid: {root.name}"
            )
        candidates.append({
            "schema_version": "stage6_notebook01_fresh_custody_binding_v1",
            "aggregate_run_id": root.name,
            "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
            "fresh_custody_request_self_sha256": custody["self_sha256"],
        })
    if len(candidates) != 1:
        raise Stage6ContractError(
            "fresh custody selection is missing or ambiguous across supported Stage 6 "
            "aggregate freezes; set the exact aggregate run ID"
        )
    return candidates[0]


def _partition_decision_schema():
    import pyarrow as pa

    return pa.schema(
        [pa.field(field, pa.string(), nullable=False) for field in PARTITION_DECISION_STRING_FIELDS]
        + [
            pa.field(field, pa.int64(), nullable=False)
            for field in PARTITION_DECISION_INTEGER_FIELDS
        ]
    )


def _observation_integer_fields(registry: Mapping[str, Any]) -> tuple[str, ...]:
    terms = sorted(
        {
            term
            for binding in registry["observation_bindings"]
            for field in ("numerator_terms", "opportunity_terms")
            for term in binding[field]
        }
    )
    presence = sorted(
        {
            binding["source_presence_field"]
            for binding in registry["observation_bindings"]
        }
    )
    return (
        "source_row_ordinal",
        "hand_completed_epoch_ms",
        "observed_hand",
        *presence,
        *terms,
    )


def _partition_observation_schema(registry: Mapping[str, Any]):
    import pyarrow as pa

    return pa.schema(
        [pa.field(field, pa.string(), nullable=False) for field in OBSERVATION_STRING_FIELDS]
        + [
            pa.field(field, pa.int64(), nullable=True)
            for field in _observation_integer_fields(registry)
        ]
    )


class _LaneParquetWriters:
    def __init__(
        self,
        root: Path,
        *,
        stem: str,
        schema: Any,
        buffer_rows: int = PARTITION_BUFFER_ROWS,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = root
        self.stem = stem
        self.schema = schema
        self.buffer_rows = buffer_rows
        self._pq = pq
        self._buffers: dict[str, list[dict[str, Any]]] = {
            lane_id: [] for lane_id in _lane_ids()
        }
        self._writers: dict[str, Any | None] = {
            lane_id: None for lane_id in _lane_ids()
        }
        self._handles: dict[str, Any | None] = {
            lane_id: None for lane_id in _lane_ids()
        }
        self._creation_identities: dict[str, tuple[int, int] | None] = {
            lane_id: None for lane_id in _lane_ids()
        }
        self.row_counts: Counter[str] = Counter()
        assert_active_run_root_identity(self.root)
        ensure_durable_directory(self.root)

    def _temporary_path(self, lane_id: str) -> Path:
        return self.root / f"{lane_id}-{self.stem}.parquet.tmp"

    def _final_path(self, lane_id: str) -> Path:
        return self.root / f"{lane_id}-{self.stem}.parquet"

    def append(self, lane_id: str, row: dict[str, Any]) -> None:
        if lane_id not in self._buffers:
            raise Stage6ContractError(f"unknown partition lane: {lane_id}")
        self._buffers[lane_id].append(row)
        self.row_counts[lane_id] += 1
        if len(self._buffers[lane_id]) >= self.buffer_rows:
            self._flush(lane_id)

    def _flush(self, lane_id: str) -> None:
        import pyarrow as pa

        rows = self._buffers[lane_id]
        if not rows:
            return
        writer = self._writers[lane_id]
        if writer is None:
            temporary = self._temporary_path(lane_id)
            assert_active_run_root_identity(temporary)
            descriptor, created = create_exclusive_regular_file(temporary)
            handle = os.fdopen(descriptor, "wb")
            try:
                writer = self._pq.ParquetWriter(
                    handle,
                    self.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            except BaseException:
                if not handle.closed:
                    handle.close()
                raise
            self._writers[lane_id] = writer
            self._handles[lane_id] = handle
            self._creation_identities[lane_id] = (
                int(created.st_dev),
                int(created.st_ino),
            )
        table = pa.Table.from_pylist(rows, schema=self.schema)
        writer.write_table(table, row_group_size=len(rows))
        assert_active_run_root_identity(self.root)
        rows.clear()

    def finalize(self) -> dict[str, dict[str, Any]]:
        import pyarrow as pa

        outputs: dict[str, dict[str, Any]] = {}
        for lane_id in _lane_ids():
            assert_active_run_root_identity(self.root)
            self._flush(lane_id)
            writer = self._writers[lane_id]
            temporary = self._temporary_path(lane_id)
            if writer is None:
                descriptor, created = create_exclusive_regular_file(temporary)
                with os.fdopen(descriptor, "wb") as handle:
                    self._pq.write_table(
                        pa.Table.from_pylist([], schema=self.schema),
                        handle,
                        compression="zstd",
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                self._creation_identities[lane_id] = (
                    int(created.st_dev),
                    int(created.st_ino),
                )
            else:
                writer.close()
                self._writers[lane_id] = None
                handle = self._handles[lane_id]
                if handle is not None and not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                self._handles[lane_id] = None
            assert_active_run_root_identity(self.root)
            fsync_regular_file(temporary)
            final = self._final_path(lane_id)
            expected_rows = self.row_counts[lane_id]

            def validate_partition(reader: Any) -> None:
                metadata_rows = self._pq.ParquetFile(reader).metadata.num_rows
                if metadata_rows != expected_rows:
                    raise Stage6ContractError(
                        f"partition row-count readback mismatch for {lane_id}/{self.stem}"
                    )

            creation_identity = self._creation_identities[lane_id]
            if creation_identity is None:
                raise Stage6ContractError(
                    f"partition creation identity is missing for {lane_id}/{self.stem}"
                )
            published_identity = atomic_publish_owned_regular_file(
                temporary,
                final,
                expected_creation_identity=creation_identity,
                directory_fsync=fsync_directory,
                validate=validate_partition,
            )
            relative = Path("partitions") / final.name
            identity = {
                "relative_path": relative.as_posix(),
                "bytes": published_identity["bytes"],
                "sha256": published_identity["sha256"],
            }
            identity["row_count"] = expected_rows
            outputs[lane_id] = identity
        return outputs

    def abort(self) -> None:
        for lane_id, writer in self._writers.items():
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                self._writers[lane_id] = None


def _integer(raw: Any, *, label: str) -> int:
    if isinstance(raw, bool):
        raise Stage6ContractError(f"{label} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise Stage6ContractError(f"{label} must be an integer") from error
    if str(value) != str(raw):
        raise Stage6ContractError(f"{label} is not canonical integer text")
    return value


def _select_decision_partition_row(
    action: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    physical_ordinal: int,
) -> dict[str, Any]:
    identity_ordinal = _integer(
        identity.get("source_row_ordinal"), label="identity.source_row_ordinal"
    )
    if identity_ordinal != physical_ordinal:
        raise Stage6ContractError(
            f"decision physical/source ordinal mismatch at {physical_ordinal}"
        )
    key_pairs = (
        ("hand_action_id", "hand_action_id"),
        ("hand_id", "hand_id"),
        ("game_id", "game_id"),
        ("event_index", "event_index"),
        ("pokernow_player_id", "actor_pokernow_player_id"),
    )
    for action_field, identity_field in key_pairs:
        if str(action.get(action_field)) != str(identity.get(identity_field)):
            raise Stage6ContractError(
                f"decision action/identity mismatch at {physical_ordinal}: "
                f"{action_field}/{identity_field}"
            )
    row = {field: str(action.get(field, "")) for field in ACTION_PARTITION_FIELDS}
    row.update(
        {
            field: str(identity.get(field, ""))
            for field in IDENTITY_PARTITION_STRING_FIELDS
        }
    )
    row.update(
        {
            "source_row_ordinal": identity_ordinal,
            "event_index": _integer(
                identity.get("event_index"), label="identity.event_index"
            ),
            "decision_epoch_ms": _integer(
                identity.get("decision_epoch_ms"),
                label="identity.decision_epoch_ms",
            ),
        }
    )
    required_string_fields = set(PARTITION_DECISION_STRING_FIELDS) - (
        OPTIONAL_DECISION_PARTITION_STRING_FIELDS
    )
    if any(row[field] == "" for field in required_string_fields):
        missing = [
            field for field in sorted(required_string_fields) if row[field] == ""
        ]
        raise Stage6ContractError(
            f"decision partition row {physical_ordinal} has empty fields: {missing}"
        )
    return row


def _select_observation_partition_row(
    raw: Mapping[str, Any],
    *,
    physical_ordinal: int,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    source_ordinal = _integer(
        raw.get("source_row_ordinal"), label="observation.source_row_ordinal"
    )
    if source_ordinal != physical_ordinal:
        raise Stage6ContractError(
            f"observation physical/source ordinal mismatch at {physical_ordinal}"
        )
    row: dict[str, Any] = {
        field: str(raw.get(field, "")) for field in OBSERVATION_STRING_FIELDS
    }
    if any(row[field] == "" for field in OBSERVATION_STRING_FIELDS):
        raise Stage6ContractError(
            f"observation partition row {physical_ordinal} has empty identity fields"
        )
    for field in _observation_integer_fields(registry):
        value = raw.get(field)
        row[field] = None if value in (None, "") else _integer(
            value, label=f"observation.{field}"
        )
    return row


def _verify_alias_owner(
    owners: dict[str, str], *, alias: str, canonical_player_id: str
) -> None:
    folded = alias.casefold()
    previous = owners.setdefault(folded, canonical_player_id)
    if previous != canonical_player_id:
        raise Stage6ContractError(
            f"PokerNow alias {alias!r} maps to multiple canonical players"
        )


def _verify_partition_manifest(
    run_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_run_identity: str,
    current_source_identities: Mapping[str, Any],
) -> dict[str, Any]:
    validate_document_self_hash(manifest)
    if (
        manifest.get("schema_version")
        != "stage6_notebook01_partition_manifest_v1"
        or manifest.get("status") != "PASS_DURABLE_PARTITION_READBACK"
        or manifest.get("run_identity") != expected_run_identity
        or manifest.get("source_identities") != current_source_identities
    ):
        raise Stage6ContractError("partition manifest binding is invalid")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or [row.get("lane_id") for row in lanes] != _lane_ids():
        raise Stage6ContractError("partition manifest lane order is invalid")
    decision_total = 0
    observation_total = 0
    for lane in lanes:
        for key in ("decision_partition", "observation_partition"):
            identity = lane[key]
            relative = Path(identity["relative_path"])
            path = run_root / relative
            _verify_identity_bound_parquet(
                run_root,
                path,
                identity,
                label=f"partition {relative.as_posix()}",
            )
        decision_total += lane["decision_partition"]["row_count"]
        observation_total += lane["observation_partition"]["row_count"]
    if decision_total != EXPECTED_DECISION_ROWS or observation_total != EXPECTED_OBSERVATION_ROWS:
        raise Stage6ContractError("partition work-unit totals are invalid")
    return dict(manifest)


def _create_task_owned_partition_staging(
    run_root: Path, *, run_identity: str
) -> tuple[Path, dict[str, Any]]:
    direct = run_root / "partitions"
    if direct.exists():
        observed = verify_owned_staging_directory(
            run_root,
            direct,
            kind="partitions",
        )
        if observed.get("run_identity") != run_identity:
            raise Stage6ContractError(
                "incomplete direct partition staging belongs to another run"
            )
        return direct, observed
    return create_owned_staging_directory(
        run_root,
        kind="partitions",
        run_identity=run_identity,
        destination_name="partitions",
    )


def _remove_task_owned_partition_staging(
    run_root: Path,
    staging: Path,
    *,
    ownership: Mapping[str, Any] | None = None,
) -> None:
    remove_owned_staging_directory(
        run_root,
        staging,
        kind="partitions",
        expected_receipt=ownership,
    )


def build_or_resume_partitions(
    stage6_root: Path,
    run_root: Path,
    *,
    run_identity: str,
    source_identities: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    partition_root = run_root / "partitions"
    manifest_path = partition_root / "partition-manifest.json"
    if manifest_path.is_file():
        return _verify_partition_manifest(
            run_root,
            _strict_json_object(manifest_path),
            expected_run_identity=run_identity,
            current_source_identities=source_identities,
        )
    remove_current_run_stale_staging_directories(
        run_root, kind="partitions", run_identity=run_identity
    )
    resources = _resource_snapshot(run_root)
    if resources["available_disk_bytes"] < MINIMUM_PARTITION_DISK_BYTES:
        raise Stage6ContractError(
            "insufficient disk for exact partitioning; select a runtime with more disk"
        )

    staging, staging_ownership = _create_task_owned_partition_staging(
        run_root, run_identity=run_identity
    )
    staging_authority = retain_owned_staging_directory(
        run_root,
        staging,
        kind="partitions",
        expected_receipt=staging_ownership,
    )
    decision_writers = _LaneParquetWriters(
        staging, stem="decisions", schema=_partition_decision_schema()
    )
    observation_writers = _LaneParquetWriters(
        staging,
        stem="observations",
        schema=_partition_observation_schema(registry),
    )
    started = time.perf_counter()
    aliases: dict[str, str] = {}
    decision_ids: set[str] = set()
    hand_state: dict[tuple[str, str], tuple[int, int]] = {}
    decision_count = 0
    observation_count = 0
    sentinel = object()
    try:
        with (stage6_root / ACTION_RELATIVE).open(
            "r", encoding="utf-8-sig", errors="strict", newline=""
        ) as action_source, (stage6_root / IDENTITY_RELATIVE).open(
            "r", encoding="utf-8-sig", errors="strict", newline=""
        ) as identity_source:
            action_reader = csv.DictReader(action_source)
            identity_reader = csv.DictReader(identity_source)
            for physical_ordinal, pair in enumerate(
                itertools.zip_longest(action_reader, identity_reader, fillvalue=sentinel)
            ):
                action, identity = pair
                if action is sentinel or identity is sentinel:
                    raise Stage6ContractError("action and identity source lengths differ")
                row = _select_decision_partition_row(
                    action, identity, physical_ordinal=physical_ordinal
                )
                decision_id = row["hand_action_id"]
                if decision_id in decision_ids:
                    raise Stage6ContractError(f"duplicate decision id: {decision_id}")
                decision_ids.add(decision_id)
                hand_key = (row["game_id"], row["hand_id"])
                previous = hand_state.get(hand_key)
                current = (row["event_index"], row["decision_epoch_ms"])
                if previous is not None and (
                    current[0] <= previous[0] or current[1] <= previous[1]
                ):
                    raise Stage6ContractError(
                        f"non-increasing decision order/time for hand {hand_key}"
                    )
                hand_state[hand_key] = current
                _verify_alias_owner(
                    aliases,
                    alias=row["actor_pokernow_player_id"],
                    canonical_player_id=row["canonical_group_player_id"],
                )
                lane_id = stable_lane_id(row["canonical_group_player_id"])
                decision_writers.append(lane_id, row)
                decision_count += 1
                if decision_count % PROGRESS_INTERVAL_ROWS == 0:
                    _progress(
                        phase="partition_decisions",
                        completed=decision_count,
                        total=EXPECTED_DECISION_ROWS,
                        started=started,
                        resource_root=run_root,
                    )
        if decision_count != EXPECTED_DECISION_ROWS:
            raise Stage6ContractError(
                f"decision source count changed: {decision_count}"
            )
        decision_outputs = decision_writers.finalize()

        observation_started = time.perf_counter()
        with (stage6_root / OBSERVATION_RELATIVE).open(
            "r", encoding="utf-8-sig", errors="strict", newline=""
        ) as observation_source:
            for physical_ordinal, raw in enumerate(csv.DictReader(observation_source)):
                row = _select_observation_partition_row(
                    raw,
                    physical_ordinal=physical_ordinal,
                    registry=registry,
                )
                _verify_alias_owner(
                    aliases,
                    alias=row["pokernow_player_id"],
                    canonical_player_id=row["canonical_group_player_id"],
                )
                lane_id = stable_lane_id(row["canonical_group_player_id"])
                observation_writers.append(lane_id, row)
                observation_count += 1
                if observation_count % PROGRESS_INTERVAL_ROWS == 0:
                    _progress(
                        phase="partition_observations",
                        completed=observation_count,
                        total=EXPECTED_OBSERVATION_ROWS,
                        started=observation_started,
                        resource_root=run_root,
                    )
        if observation_count != EXPECTED_OBSERVATION_ROWS:
            raise Stage6ContractError(
                f"observation source count changed: {observation_count}"
            )
        observation_outputs = observation_writers.finalize()
    except BaseException as primary_error:
        decision_writers.abort()
        observation_writers.abort()
        try:
            staging_authority.close()
        except BaseException as close_error:
            _attach_retry_owner(primary_error, staging_authority)
            primary_error.add_note(
                f"retained partition authority close also failed: {close_error}"
            )
        try:
            _remove_task_owned_partition_staging(
                run_root, staging, ownership=staging_ownership
            )
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"owned partition staging cleanup was held: {cleanup_error}"
            )
        raise
    finally:
        decision_ids.clear()
        hand_state.clear()
        gc.collect()

    try:
        refreshed_source_identities = _source_identities(stage6_root)
        if refreshed_source_identities != source_identities:
            raise Stage6ContractError("source bytes changed during partitioning")
        lanes = []
        for lane_id in _lane_ids():
            decision_identity = dict(decision_outputs[lane_id])
            observation_identity = dict(observation_outputs[lane_id])
            decision_identity["relative_path"] = (
                Path("partitions") / Path(decision_identity["relative_path"]).name
            ).as_posix()
            observation_identity["relative_path"] = (
                Path("partitions") / Path(observation_identity["relative_path"]).name
            ).as_posix()
            lanes.append(
                {
                    "lane_id": lane_id,
                    "decision_partition": decision_identity,
                    "observation_partition": observation_identity,
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": "stage6_notebook01_partition_manifest_v1",
            "status": "PASS_DURABLE_PARTITION_READBACK",
            "run_identity": run_identity,
            "partition_algorithm": "sha256_canonical_player_first_8_bytes_mod_48_v1",
            "lane_count": LANE_COUNT,
            "decision_source_row_count": decision_count,
            "observation_source_row_count": observation_count,
            "canonical_alias_count": len(aliases),
            "global_decision_identity_uniqueness_verified": True,
            "global_hand_order_and_timestamp_monotonicity_verified": True,
            "global_alias_owner_uniqueness_verified": True,
            "source_identities": dict(source_identities),
            "lanes": lanes,
        }
        manifest["self_sha256"] = document_self_sha256(manifest)
        _atomic_write_json(staging / "partition-manifest.json", manifest)
        publish_owned_staging_directory(
            run_root,
            staging,
            partition_root,
            kind="partitions",
            expected_receipt=staging_ownership,
            retained_authority=staging_authority,
        )
        return _verify_partition_manifest(
            run_root,
            _strict_json_object(manifest_path),
            expected_run_identity=run_identity,
            current_source_identities=source_identities,
        )
    finally:
        staging_authority.close()


def _decision_output_schema():
    import pyarrow as pa

    fields = []
    for column in DECISION_OUTPUT_COLUMNS:
        if column in DECISION_BOOLEAN_FIELDS:
            fields.append(pa.field(column, pa.bool_(), nullable=False))
        elif column in DECISION_INTEGER_FIELDS:
            fields.append(
                pa.field(column, pa.int64(), nullable=column in DECISION_NULLABLE_INTEGER_FIELDS)
            )
        elif column in DECISION_FLOAT_FIELDS:
            nullable = column in DECISION_NULLABLE_FLOAT_FIELDS
            fields.append(pa.field(column, pa.float64(), nullable=nullable))
        else:
            nullable = column in {
                "board_pair_level",
                "profile_source_max_event_utc",
                "private_card_visibility_utc",
            }
            fields.append(pa.field(column, pa.string(), nullable=nullable))
    return pa.schema(fields)


def _profile_output_schema(registry: Mapping[str, Any]):
    import pyarrow as pa

    fields = [
        pa.field("canonical_player_id", pa.string(), nullable=False),
        pa.field("profile_snapshot_id", pa.string(), nullable=False),
        pa.field("profile_snapshot_cutoff_utc", pa.string(), nullable=False),
        pa.field("profile_source_max_event_utc", pa.string(), nullable=True),
        pa.field("identity_resolution_status", pa.string(), nullable=False),
        pa.field("total_hands", pa.int64(), nullable=False),
        pa.field("games_played", pa.int64(), nullable=False),
    ]
    for metric in registry["metrics"]:
        fields.extend(
            [
                pa.field(metric["value_field"], pa.float64(), nullable=True),
                pa.field(metric["numerator_field"], pa.int64(), nullable=True),
                pa.field(
                    metric["opportunity_denominator_field"],
                    pa.int64(),
                    nullable=True,
                ),
                pa.field(
                    metric["uncertainty_lower_field"], pa.float64(), nullable=True
                ),
                pa.field(
                    metric["uncertainty_upper_field"], pa.float64(), nullable=True
                ),
            ]
        )
    fields.extend(
        [
            pa.field("authoritative_source_receipt_json", pa.string(), nullable=False),
            pa.field("alias_resolution_receipt_json", pa.string(), nullable=False),
        ]
    )
    return pa.schema(fields)


def _quarantine_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("source_row_ordinal", pa.int64(), nullable=False),
            pa.field("decision_id", pa.string(), nullable=False),
            pa.field("game_id", pa.string(), nullable=False),
            pa.field("hand_id", pa.string(), nullable=False),
            pa.field("decision_order", pa.int64(), nullable=False),
            pa.field("decision_utc", pa.string(), nullable=False),
            pa.field("canonical_player_id", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
        ]
    )


def _serialize_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(profile)
    source_receipt = row.pop("authoritative_source_receipt")
    alias_receipt = row.pop("alias_resolution_receipt")
    row["authoritative_source_receipt_json"] = canonical_json_bytes(
        source_receipt
    ).decode("ascii")
    row["alias_resolution_receipt_json"] = canonical_json_bytes(
        alias_receipt
    ).decode("ascii")
    return row


def deserialize_profile_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = dict(raw)
        row["authoritative_source_receipt"] = json.loads(
            row.pop("authoritative_source_receipt_json")
        )
        row["alias_resolution_receipt"] = json.loads(
            row.pop("alias_resolution_receipt_json")
        )
        output.append(row)
    return output


def _atomic_write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], schema: Any
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert_active_run_root_identity(path)
    ensure_durable_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        assert_active_run_root_identity(temporary)
        materialized_rows = rows if isinstance(rows, list) else list(rows)
        table = pa.Table.from_pylist(materialized_rows, schema=schema)
        descriptor, created = create_exclusive_regular_file(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            pq.write_table(
                table,
                handle,
                compression="zstd",
                use_dictionary=True,
                row_group_size=max(1, min(len(rows), 16_384)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        assert_active_run_root_identity(temporary)
        fsync_regular_file(temporary)
    except BaseException:
        raise

    expected_rows = len(rows)

    def validate_parquet(reader: Any) -> None:
        metadata_rows = pq.ParquetFile(reader).metadata.num_rows
        if metadata_rows != expected_rows:
            raise Stage6ContractError(
                f"Parquet row-count readback mismatch: {path}"
            )

    identity = atomic_publish_owned_regular_file(
        temporary,
        path,
        expected_creation_identity=(int(created.st_dev), int(created.st_ino)),
        directory_fsync=fsync_directory,
        validate=validate_parquet,
    )
    identity.pop("publication_status", None)
    identity["row_count"] = expected_rows
    return identity


def _pin_inner_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ARROW_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _release_worker_memory() -> bool:
    gc.collect()
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except (AttributeError, OSError):
        return False


def _completion_time_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def materialize_lane_semantics(
    lane_id: str,
    decision_records: Sequence[Mapping[str, Any]],
    observation_records: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one disjoint lane without depending on worker count or I/O order."""

    if lane_id not in _lane_ids():
        raise Stage6ContractError(f"unknown materialization lane: {lane_id}")
    for collection_name, records in (
        ("decision", decision_records),
        ("observation", observation_records),
    ):
        for record in records:
            canonical_player_id = record.get("canonical_group_player_id")
            observed_lane_id = stable_lane_id(canonical_player_id)
            if observed_lane_id != lane_id:
                raise Stage6ContractError(
                    f"{lane_id}: {collection_name} row for {canonical_player_id!r} "
                    f"belongs to {observed_lane_id}"
                )
    causal_decisions = [
        {
            "source_row_ordinal": row["source_row_ordinal"],
            "canonical_group_player_id": row["canonical_group_player_id"],
            "actor_pokernow_player_id": row["actor_pokernow_player_id"],
            "decision_utc": row["decision_utc"],
            "decision_epoch_ms": row["decision_epoch_ms"],
        }
        for row in decision_records
    ]
    causal = build_reference_causal_snapshots(
        causal_decisions, observation_records, registry
    )
    binding_by_ordinal = {
        row["source_row_ordinal"]: row for row in causal["decision_bindings"]
    }
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    quarantine_counts: Counter[str] = Counter()
    for record in decision_records:
        reason = classify_source_quarantine_reason(record)
        if reason is not None:
            quarantine_counts[reason] += 1
            quarantine.append(
                {
                    "source_row_ordinal": record["source_row_ordinal"],
                    "decision_id": record["hand_action_id"],
                    "game_id": record["game_id"],
                    "hand_id": record["hand_id"],
                    "decision_order": record["event_index"],
                    "decision_utc": record["decision_utc"],
                    "canonical_player_id": record["canonical_group_player_id"],
                    "reason": reason,
                }
            )
            continue
        ordinal = record["source_row_ordinal"]
        if ordinal not in binding_by_ordinal:
            raise Stage6ContractError(
                f"{lane_id}: causal profile binding is missing for ordinal {ordinal}"
            )
        accepted.append(
            build_decision_row(record, record, binding_by_ordinal[ordinal])
        )
    if len(accepted) + len(quarantine) != len(decision_records):
        raise Stage6ContractError(f"{lane_id}: decision work units were not conserved")
    joined_report = audit_joined_decision_profiles(
        accepted,
        causal["profile_snapshots"],
        registry["metrics"],
        feature_role_manifest=feature_manifest,
        expected_feature_role_manifest_sha256=feature_manifest["self_sha256"],
    )
    return {
        "accepted_decisions": accepted,
        "profile_snapshots": causal["profile_snapshots"],
        "quarantine": quarantine,
        "quarantine_reason_counts": dict(sorted(quarantine_counts.items())),
        "joined_oracle_report": joined_report,
    }


def _run_lane_worker(
    run_root_text: str,
    plan_row: Mapping[str, Any],
    registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _pin_inner_threads()
    import pyarrow as pa
    pa.set_cpu_count(1)
    try:
        pa.set_io_thread_count(1)
    except AttributeError:
        pass
    run_root = Path(run_root_text).resolve()
    if "OneDrive" in str(run_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    lane_id = str(plan_row["lane_id"])
    specification = plan_row["specification"]
    decision_partition = run_root / specification["decision_partition"][
        "relative_path"
    ]
    observation_partition = run_root / specification["observation_partition"][
        "relative_path"
    ]
    partition_records: dict[str, list[dict[str, Any]]] = {}
    for label, path, identity in (
        (
            "decision",
            decision_partition,
            specification["decision_partition"],
        ),
        (
            "observation",
            observation_partition,
            specification["observation_partition"],
        ),
    ):
        partition_records[label] = _read_identity_bound_parquet_rows(
            run_root,
            path,
            identity,
            label=f"{lane_id}: {label} partition",
        )
    decision_records = partition_records["decision"]
    observation_records = partition_records["observation"]
    del partition_records
    semantic = materialize_lane_semantics(
        lane_id,
        decision_records,
        observation_records,
        registry,
        feature_manifest,
    )
    accepted = semantic["accepted_decisions"]
    profile_snapshots = semantic["profile_snapshots"]
    quarantine = semantic["quarantine"]
    accepted_count = len(accepted)
    profile_snapshot_count = len(profile_snapshots)
    quarantine_count = len(quarantine)
    lane_root = run_root / "lanes" / lane_id
    outputs: dict[str, dict[str, Any]] = {}

    def write_output(
        key: str,
        rows: Sequence[Mapping[str, Any]],
        schema: Any,
        name: str,
    ) -> None:
        path = lane_root / name
        identity = _atomic_write_parquet(path, rows, schema)
        identity["relative_path"] = (
            Path("lanes") / lane_id / name
        ).as_posix()
        outputs[key] = identity

    write_output(
        "accepted_decisions",
        accepted,
        _decision_output_schema(),
        "accepted-decisions.parquet",
    )
    del accepted
    semantic["accepted_decisions"] = None
    gc.collect()

    for index, profile in enumerate(profile_snapshots):
        profile_snapshots[index] = _serialize_profile(profile)
    write_output(
        "profile_snapshots",
        profile_snapshots,
        _profile_output_schema(registry),
        "profile-snapshots.parquet",
    )
    del profile_snapshots
    semantic["profile_snapshots"] = None
    gc.collect()

    write_output(
        "quarantine",
        quarantine,
        _quarantine_schema(),
        "quarantine.parquet",
    )

    checkpoint: dict[str, Any] = {
        "schema_version": "stage6_notebook01_lane_checkpoint_v1",
        "status": "PASS_LANE_JOINED_ORACLE_AND_DURABLE_OUTPUT_READBACK",
        "stage_id": plan_row["stage_id"],
        "lane_id": lane_id,
        "seed": plan_row["seed"],
        "checkpoint_identity": plan_row["checkpoint_identity"],
        "decision_partition_row_count": len(decision_records),
        "observation_partition_row_count": len(observation_records),
        "accepted_decision_row_count": accepted_count,
        "quarantined_decision_row_count": quarantine_count,
        "quarantine_reason_counts": semantic["quarantine_reason_counts"],
        "profile_snapshot_row_count": profile_snapshot_count,
        "work_unit_conservation_verified": True,
        "joined_oracle_report": semantic["joined_oracle_report"],
        "outputs": outputs,
        "runtime_authority": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "live_action_authority": False,
    }
    checkpoint["self_sha256"] = document_self_sha256(checkpoint)
    checkpoint_relative = Path("checkpoints") / (
        f"{plan_row['checkpoint_identity']}.checkpoint"
    )
    checkpoint_path = run_root / checkpoint_relative
    _atomic_write_json(checkpoint_path, checkpoint)
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    completion: dict[str, Any] = {
        "schema_version": "stage6_lane_completion_receipt_v1",
        "status": "DURABLE_READBACK_VERIFIED",
        "stage_id": plan_row["stage_id"],
        "lane_id": lane_id,
        "seed": plan_row["seed"],
        "checkpoint_identity": plan_row["checkpoint_identity"],
        "checkpoint_relative_path": checkpoint_relative.as_posix(),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": observed_checkpoint_sha256,
        "durable_readback_verified": True,
        "completed_at_utc": _completion_time_utc(),
    }
    completion["self_sha256"] = document_self_sha256(completion)
    receipt_path = run_root / "completion_receipts" / f"{lane_id}.json"
    _atomic_write_json(receipt_path, completion)
    del decision_records, observation_records, quarantine, semantic
    malloc_trimmed = _release_worker_memory()
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_lane_cleanup",
                "lane_id": lane_id,
                "checkpoint_identity": plan_row["checkpoint_identity"],
                "malloc_trimmed": malloc_trimmed,
                **_resource_snapshot(run_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return completion


def _load_completion_receipts(run_root: Path) -> list[dict[str, Any]]:
    receipt_root = run_root / "completion_receipts"
    if not receipt_root.is_dir():
        return []
    return [
        _strict_json_object(path)
        for path in sorted(receipt_root.glob("*.json"), key=lambda item: item.name)
    ]


def _completion_receipt_for_plan(
    plan_row: Mapping[str, Any],
    completion_receipts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    matching = [
        receipt
        for receipt in completion_receipts
        if receipt.get("checkpoint_identity") == plan_row["checkpoint_identity"]
        and receipt.get("lane_id") == plan_row["lane_id"]
    ]
    if len(matching) != 1:
        raise Stage6ContractError(
            "lane plan does not have exactly one completion receipt"
        )
    return matching[0]


def _verify_lane_checkpoint(
    run_root: Path,
    plan_row: Mapping[str, Any],
    completion_receipt: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_relative = Path("checkpoints") / (
        f"{plan_row['checkpoint_identity']}.checkpoint"
    )
    expected_receipt_fields = {
        "schema_version",
        "status",
        "stage_id",
        "lane_id",
        "seed",
        "checkpoint_identity",
        "checkpoint_relative_path",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "durable_readback_verified",
        "completed_at_utc",
        "self_sha256",
    }
    if set(completion_receipt) != expected_receipt_fields:
        raise Stage6ContractError("lane completion receipt schema is invalid")
    validate_document_self_hash(completion_receipt)
    if (
        completion_receipt.get("schema_version")
        != "stage6_lane_completion_receipt_v1"
        or completion_receipt.get("status") != "DURABLE_READBACK_VERIFIED"
        or completion_receipt.get("durable_readback_verified") is not True
        or completion_receipt.get("stage_id") != plan_row["stage_id"]
        or completion_receipt.get("lane_id") != plan_row["lane_id"]
        or completion_receipt.get("seed") != plan_row["seed"]
        or completion_receipt.get("checkpoint_identity")
        != plan_row["checkpoint_identity"]
        or completion_receipt.get("checkpoint_relative_path")
        != checkpoint_relative.as_posix()
    ):
        raise Stage6ContractError(
            "lane completion receipt does not bind its frozen plan"
        )
    checkpoint_bytes = completion_receipt.get("checkpoint_bytes")
    if (
        type(checkpoint_bytes) is not int
        or checkpoint_bytes <= 0
        or checkpoint_bytes > MAX_CHECKPOINT_JSON_BYTES
    ):
        raise Stage6ContractError("lane checkpoint byte count is invalid")

    checkpoint_path = run_root / checkpoint_relative

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage6ContractError(
                    f"duplicate JSON key in receipt-bound checkpoint: {key}"
                )
            output[key] = value
        return output

    with open_identity_bound_regular_file(
        run_root,
        checkpoint_path,
        expected_bytes=checkpoint_bytes,
        expected_sha256=completion_receipt["checkpoint_sha256"],
    ) as reader:
        raw = reader.read(checkpoint_bytes + 1)
        if len(raw) != checkpoint_bytes:
            raise Stage6ContractError(
                "receipt-bound lane checkpoint byte count changed during read"
            )
        try:
            checkpoint = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=reject_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Stage6ContractError(
                "receipt-bound lane checkpoint is not strict JSON"
            ) from error
    if not isinstance(checkpoint, dict):
        raise Stage6ContractError("lane checkpoint JSON authority is not an object")
    validate_document_self_hash(checkpoint)
    expected_checkpoint_fields = {
        "schema_version",
        "status",
        "stage_id",
        "lane_id",
        "seed",
        "checkpoint_identity",
        "decision_partition_row_count",
        "observation_partition_row_count",
        "accepted_decision_row_count",
        "quarantined_decision_row_count",
        "quarantine_reason_counts",
        "profile_snapshot_row_count",
        "work_unit_conservation_verified",
        "joined_oracle_report",
        "outputs",
        "runtime_authority",
        "scientific_authority",
        "promotion_authority",
        "live_action_authority",
        "self_sha256",
    }
    if set(checkpoint) != expected_checkpoint_fields:
        raise Stage6ContractError("lane checkpoint schema is not exact")
    if (
        checkpoint.get("schema_version")
        != "stage6_notebook01_lane_checkpoint_v1"
        or checkpoint.get("status")
        != "PASS_LANE_JOINED_ORACLE_AND_DURABLE_OUTPUT_READBACK"
        or checkpoint.get("stage_id") != plan_row["stage_id"]
        or checkpoint.get("lane_id") != plan_row["lane_id"]
        or checkpoint.get("seed") != plan_row["seed"]
        or checkpoint.get("checkpoint_identity")
        != plan_row["checkpoint_identity"]
    ):
        raise Stage6ContractError("lane checkpoint does not bind its frozen plan")
    output_row_counts = {
        "accepted_decisions": checkpoint.get("accepted_decision_row_count"),
        "profile_snapshots": checkpoint.get("profile_snapshot_row_count"),
        "quarantine": checkpoint.get("quarantined_decision_row_count"),
    }
    outputs = checkpoint.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(output_row_counts):
        raise Stage6ContractError(
            "lane checkpoint does not bind all three required output artifacts"
        )
    if (
        checkpoint.get("work_unit_conservation_verified") is not True
        or checkpoint.get("accepted_decision_row_count", -1)
        + checkpoint.get("quarantined_decision_row_count", -1)
        != checkpoint.get("decision_partition_row_count")
    ):
        raise Stage6ContractError("lane checkpoint work-unit conservation is invalid")
    expected_schemas = {
        "accepted_decisions": _decision_output_schema(),
        "profile_snapshots": _profile_output_schema(registry),
        "quarantine": _quarantine_schema(),
    }
    for output_key, output in outputs.items():
        if (
            not isinstance(output, Mapping)
            or set(output) != {"relative_path", "bytes", "sha256", "row_count"}
            or output.get("row_count") != output_row_counts[output_key]
        ):
            raise Stage6ContractError(
                f"lane checkpoint {output_key} row-count binding is invalid"
            )
        relative = Path(output["relative_path"])
        path = run_root / relative
        _verify_identity_bound_parquet(
            run_root,
            path,
            output,
            label=f"lane output {relative.as_posix()}",
            expected_schema=expected_schemas[output_key],
        )
    return checkpoint


def _merge_checkpoint_output_rows(
    run_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    output_key: str,
    ordinal_field: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    def source_ordinal(row: Mapping[str, Any]) -> int:
        raw = row.get(ordinal_field)
        if ordinal_field == "row_id":
            prefix = "stage6-decision-"
            if (
                type(raw) is not str
                or not raw.startswith(prefix)
                or len(raw) != len(prefix) + ROW_ID_WIDTH
                or not raw[len(prefix) :].isdigit()
            ):
                raise Stage6ContractError(
                    f"{output_key} row has a noncanonical source ordinal"
                )
            return int(raw[len(prefix) :])
        if type(raw) is not int or raw < 0:
            raise Stage6ContractError(
                f"{output_key} row has an invalid source ordinal"
            )
        return raw

    iterators = []
    heap: list[tuple[int, int, dict[str, Any]]] = []
    for lane_index, checkpoint in enumerate(checkpoints):
        output = checkpoint["outputs"][output_key]
        relative = Path(output["relative_path"])
        iterator = iter(
            _iter_identity_bound_parquet_rows(
                run_root,
                run_root / relative,
                output,
                label=f"{output_key} merge input {relative.as_posix()}",
            )
        )
        iterators.append(iterator)
        first = next(iterator, None)
        if first is not None:
            heapq.heappush(
                heap, (source_ordinal(first), lane_index, first)
            )
    previous = -1
    while heap:
        ordinal, lane_index, row = heapq.heappop(heap)
        if ordinal <= previous:
            raise Stage6ContractError(
                f"{output_key} source ordinals are not globally unique and ordered"
            )
        previous = ordinal
        yield ordinal, row
        following = next(iterators[lane_index], None)
        if following is not None:
            heapq.heappush(
                heap,
                (source_ordinal(following), lane_index, following),
            )


def _recompute_materialized_source_identity_roots(
    run_root: Path, checkpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    accepted_digest = hashlib.sha256()
    quarantine_digest = hashlib.sha256()
    reason_digests = {
        reason: hashlib.sha256()
        for reason in sorted(ALLOWED_SOURCE_QUARANTINE_REASONS)
    }
    reason_counts: Counter[str] = Counter()
    accepted_count = 0
    quarantined_count = 0
    for ordinal, row in _merge_checkpoint_output_rows(
        run_root,
        checkpoints,
        output_key="accepted_decisions",
        ordinal_field="row_id",
    ):
        decision_id = row.get("decision_id")
        if type(decision_id) is not str or not decision_id:
            raise Stage6ContractError("accepted row has an invalid decision id")
        _update_accepted_root(accepted_digest, ordinal, decision_id)
        accepted_count += 1

    for ordinal, row in _merge_checkpoint_output_rows(
        run_root,
        checkpoints,
        output_key="quarantine",
        ordinal_field="source_row_ordinal",
    ):
        reason = row.get("reason")
        decision_id = row.get("decision_id")
        if reason not in reason_digests:
            raise Stage6ContractError(
                f"materialized quarantine has an unregistered reason: {reason!r}"
            )
        if type(decision_id) is not str or not decision_id:
            raise Stage6ContractError("quarantine row has an invalid decision id")
        reason_counts[reason] += 1
        _update_accepted_root(reason_digests[reason], ordinal, decision_id)
        _update_quarantine_root(quarantine_digest, ordinal, reason, decision_id)
        quarantined_count += 1
    return {
        "accepted_decision_row_count": accepted_count,
        "quarantined_decision_row_count": quarantined_count,
        "accepted_row_identity_root_algorithm": ACCEPTED_ROOT_ALGORITHM,
        "accepted_row_identity_root": accepted_digest.hexdigest(),
        "quarantine_row_identity_root_algorithm": QUARANTINE_ROOT_ALGORITHM,
        "quarantine_row_identity_root": quarantine_digest.hexdigest(),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "quarantine_reason_row_identity_roots": {
            reason: digest.hexdigest()
            for reason, digest in sorted(reason_digests.items())
        },
    }


def _build_plan(
    partition_manifest: Mapping[str, Any],
    *,
    run_identity: str,
) -> list[dict[str, Any]]:
    lanes = [
        {
            "lane_id": lane["lane_id"],
            "run_identity": run_identity,
            "decision_partition": lane["decision_partition"],
            "observation_partition": lane["observation_partition"],
            "causal_time_rule": "completed_hand_utc_strictly_before_decision_utc",
            "quarantine_policy": "exact_frozen_stage6_source_quarantine_contract_v1",
        }
        for lane in partition_manifest["lanes"]
    ]
    return build_lane_plan(stage_id=STAGE_ID, lanes=lanes, base_seed=BASE_SEED)


def _write_or_verify_plan(run_root: Path, plan: Sequence[Mapping[str, Any]]) -> None:
    document: dict[str, Any] = {
        "schema_version": "stage6_notebook01_lane_plan_v2",
        "status": "FROZEN_EXACT_48_LOGICAL_LANE_PLAN",
        "lane_count": LANE_COUNT,
        "live_worker_request_bound_to_plan": False,
        "plan": [dict(row) for row in plan],
    }
    document["self_sha256"] = document_self_sha256(document)
    path = run_root / "lane-plan.json"
    if path.is_file():
        if _strict_json_object(path) != document:
            raise Stage6ContractError("existing lane plan differs from current frozen plan")
    else:
        _atomic_write_json(path, document)


def _resource_admission(
    run_root: Path,
    *,
    worker_resolution: Mapping[str, Any],
) -> dict[str, Any]:
    resources = _resource_snapshot(run_root)
    live_workers = int(worker_resolution["live_workers"])
    required_ram = (
        MINIMUM_PARENT_RESERVE_BYTES + live_workers * ESTIMATED_WORKER_RAM_BYTES
    )
    required_disk = MINIMUM_PARTITION_DISK_BYTES
    report = {
        **resources,
        "worker_resolution": dict(worker_resolution),
        "live_workers": live_workers,
        "required_available_ram_bytes": required_ram,
        "required_available_disk_bytes": required_disk,
        "ram_admitted": resources["available_system_ram_bytes"] >= required_ram,
        "disk_admitted": resources["available_disk_bytes"] >= required_disk,
    }
    if not report["ram_admitted"] or not report["disk_admitted"]:
        raise Stage6ContractError(
            "selected runtime lacks RAM or disk for its resolved host-process "
            "parallelism; choose another CPU/GPU/TPU runtime or an explicit lower "
            "live-worker request without changing the 48 logical work units"
        )
    return report


def _execute_unfinished_lanes(
    run_root: Path,
    unfinished: Sequence[Mapping[str, Any]],
    *,
    requested_workers: Any,
    registry: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolution = worker_resolution_report(
        requested=requested_workers,
        unfinished_lane_count=len(unfinished),
        lane_cap=LANE_COUNT,
    )
    worker_count = int(resolution["live_workers"])
    if worker_count == 0:
        return {
            **_resource_snapshot(run_root),
            "worker_resolution": resolution,
            "live_workers": 0,
            "required_available_ram_bytes": 0,
            "required_available_disk_bytes": 0,
            "ram_admitted": True,
            "disk_admitted": True,
            "resume_without_new_workers": True,
        }
    admission = _resource_admission(
        run_root,
        worker_resolution=resolution,
    )
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    completed = 0
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        future_to_lane = {
            executor.submit(
                _run_lane_worker,
                str(run_root),
                dict(plan_row),
                dict(registry),
                dict(feature_manifest),
            ): str(plan_row["lane_id"])
            for plan_row in unfinished
        }
        pending = set(future_to_lane)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=PROGRESS_HEARTBEAT_SECONDS,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                lane_id = future_to_lane.pop(future)
                try:
                    completion = future.result()
                except BaseException:
                    failures += 1
                    for other in pending:
                        other.cancel()
                    raise
                completed += 1
                _progress(
                    phase="materialize_causal_lanes",
                    completed=completed,
                    total=len(unfinished),
                    started=started,
                    resource_root=run_root,
                    active=min(worker_count, len(pending)),
                    queued=max(len(pending) - worker_count, 0),
                    failed=failures,
                    checkpoint_identity=completion["checkpoint_identity"],
                    cleanup_state="completed_payload_released",
                )
                del completion
                gc.collect()
            if not done:
                _progress(
                    phase="materialize_causal_lanes",
                    completed=completed,
                    total=len(unfinished),
                    started=started,
                    resource_root=run_root,
                    active=min(worker_count, len(pending)),
                    queued=max(len(pending) - worker_count, 0),
                    failed=failures,
                    cleanup_state="waiting_for_lane_checkpoint",
                )
    return admission


def _aggregate(
    run_root: Path,
    plan: Sequence[Mapping[str, Any]],
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    run_identity: str,
    run_binding: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
    quarantine_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    resource_admission: Mapping[str, Any],
) -> dict[str, Any]:
    if len(checkpoints) != len(plan):
        raise Stage6ContractError("immutable checkpoint set differs from lane plan")
    for checkpoint in checkpoints:
        validate_document_self_hash(checkpoint)
    accepted_total = sum(row["accepted_decision_row_count"] for row in checkpoints)
    quarantined_total = sum(
        row["quarantined_decision_row_count"] for row in checkpoints
    )
    profile_total = sum(row["profile_snapshot_row_count"] for row in checkpoints)
    reason_counts: Counter[str] = Counter()
    for row in checkpoints:
        reason_counts.update(row["quarantine_reason_counts"])
    if accepted_total + quarantined_total != EXPECTED_DECISION_ROWS:
        raise Stage6ContractError("global decision work-unit conservation failed")
    if accepted_total != quarantine_contract["accepted_decision_row_count"]:
        raise Stage6ContractError("accepted decision count differs from frozen contract")
    if (
        split_manifest.get("status")
        != "PASS_SEALED_GROUP_TIME_SPLIT_ASSIGNMENT"
        or split_manifest.get("run_identity") != run_identity
        or split_manifest.get("accepted_decision_row_count") != accepted_total
        or split_manifest.get("work_unit_conservation_verified") is not True
        or split_manifest.get("private_lockbox_assignment_sealed") is not True
        or split_manifest.get("private_lockbox_opened") is not False
        or split_manifest.get("private_lockbox_eligible_for_selection") is not False
    ):
        raise Stage6ContractError("sealed split assignment manifest is invalid")
    if quarantined_total != quarantine_contract["quarantined_decision_row_count"]:
        raise Stage6ContractError("quarantine count differs from frozen contract")
    if dict(sorted(reason_counts.items())) != quarantine_contract[
        "quarantine_reason_counts"
    ]:
        raise Stage6ContractError("quarantine reasons differ from frozen contract")
    source_identity_roots = _recompute_materialized_source_identity_roots(
        run_root, checkpoints
    )
    expected_source_identity_roots = {
        "accepted_decision_row_count": quarantine_contract[
            "accepted_decision_row_count"
        ],
        "quarantined_decision_row_count": quarantine_contract[
            "quarantined_decision_row_count"
        ],
        "accepted_row_identity_root_algorithm": quarantine_contract[
            "accepted_row_identity_root_algorithm"
        ],
        "accepted_row_identity_root": quarantine_contract[
            "accepted_row_identity_root"
        ],
        "quarantine_row_identity_root_algorithm": quarantine_contract[
            "quarantine_row_identity_root_algorithm"
        ],
        "quarantine_row_identity_root": quarantine_contract[
            "quarantine_row_identity_root"
        ],
        "quarantine_reason_counts": quarantine_contract[
            "quarantine_reason_counts"
        ],
        "quarantine_reason_row_identity_roots": quarantine_contract[
            "quarantine_reason_row_identity_roots"
        ],
    }
    if source_identity_roots != expected_source_identity_roots:
        raise Stage6ContractError(
            "materialized accepted/quarantine row identities differ from frozen source scan"
        )
    semantic_lanes = [
        {
            "lane_id": row["lane_id"],
            "checkpoint_self_sha256": row["self_sha256"],
            "accepted_decision_row_count": row["accepted_decision_row_count"],
            "quarantined_decision_row_count": row[
                "quarantined_decision_row_count"
            ],
            "profile_snapshot_row_count": row["profile_snapshot_row_count"],
            "joined_oracle_report": row["joined_oracle_report"],
            "outputs": row["outputs"],
        }
        for row in checkpoints
    ]
    if [row["lane_id"] for row in semantic_lanes] != _lane_ids():
        raise Stage6ContractError("aggregation order is not canonical lane order")
    fixed_order_root = hashlib.sha256(
        canonical_json_bytes(semantic_lanes)
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": "stage6_notebook01_materialization_manifest_v1",
        "status": "PASS_FULL_CAUSAL_DECISION_PROFILE_MATERIALIZATION_AND_SEALED_SPLIT_LOCK",
        "run_identity": run_identity,
        "run_binding": dict(run_binding),
        "requested_workers_default": REQUESTED_WORKERS_DEFAULT,
        "lane_count": LANE_COUNT,
        "completed_lane_count": len(checkpoints),
        "worker_resolution_rule": (
            "min(requested_or_auto_host_cpu_capacity, "
            "unfinished_independent_lane_count, detected_usable_logical_cpus, 48)"
        ),
        "worker_request_bound_to_scientific_identity": False,
        "inner_numeric_threads_per_worker": 1,
        "partition_manifest_self_sha256": partition_manifest["self_sha256"],
        "resource_admission": dict(resource_admission),
        "source_decision_row_count": EXPECTED_DECISION_ROWS,
        "accepted_decision_row_count": accepted_total,
        "quarantined_decision_row_count": quarantined_total,
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "accepted_row_identity_root_algorithm": source_identity_roots[
            "accepted_row_identity_root_algorithm"
        ],
        "accepted_row_identity_root": source_identity_roots[
            "accepted_row_identity_root"
        ],
        "quarantine_row_identity_root_algorithm": source_identity_roots[
            "quarantine_row_identity_root_algorithm"
        ],
        "quarantine_row_identity_root": source_identity_roots[
            "quarantine_row_identity_root"
        ],
        "quarantine_reason_row_identity_roots": source_identity_roots[
            "quarantine_reason_row_identity_roots"
        ],
        "source_quarantine_identity_parity_verified": True,
        "profile_snapshot_row_count_across_lanes": profile_total,
        "work_unit_conservation_verified": True,
        "fixed_lane_order_aggregation": [row["lane_id"] for row in semantic_lanes],
        "fixed_order_semantic_root_digest": fixed_order_root,
        "split_assignment_manifest_self_sha256": split_manifest["self_sha256"],
        "split_assignment_root_digest": split_manifest[
            "split_assignment_root_digest"
        ],
        "split_assignment_status": split_manifest["status"],
        "split_assignment_row_count": split_manifest[
            "accepted_decision_row_count"
        ],
        "split_assignment_hand_count": split_manifest["accepted_hand_count"],
        "private_lockbox_assignment_sealed": True,
        "private_lockbox_opened": False,
        "private_lockbox_eligible_for_selection": False,
        "lanes": semantic_lanes,
        "runtime_authority": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "drive_publication_authority": False,
        "live_action_authority": False,
    }
    manifest["self_sha256"] = document_self_sha256(manifest)
    output = run_root / "stage6_notebook01_materialization_manifest.json"
    if output.is_file():
        observed = _strict_json_object(output)
        validate_document_self_hash(observed)
        stable_keys = set(manifest) - {"resource_admission", "self_sha256"}
        if (
            set(observed) != set(manifest)
            or any(observed.get(key) != manifest.get(key) for key in stable_keys)
        ):
            raise Stage6ContractError(
                "existing materialization manifest differs from recomputed durable outputs"
            )
        return observed
    _atomic_write_json(output, manifest)
    observed = _strict_json_object(output)
    if observed != manifest or validate_document_self_hash(observed) != manifest[
        "self_sha256"
    ]:
        raise Stage6ContractError("materialization manifest readback mismatch")
    return manifest


def _materialization_run_root(
    stage6_root: Path,
    *,
    run_identity: str,
    runtime_state_root: Path | None,
) -> Path:
    if (
        type(run_identity) is not str
        or CHECKPOINT_PATTERN.fullmatch(run_identity) is None
    ):
        raise Stage6ContractError("materialization run identity is invalid")
    if runtime_state_root is None:
        resolved_runtime_state_root = (
            stage6_root / "runtime_state" / "notebook01"
        ).resolve()
    else:
        resolved_runtime_state_root = Path(runtime_state_root).resolve()
    if "OneDrive" in str(resolved_runtime_state_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    return resolved_runtime_state_root / "materializations" / run_identity


def materialize(
    stage6_root: Path,
    *,
    requested_workers: Any = REQUESTED_WORKERS_DEFAULT,
    runtime_state_root: Path | None = None,
    fresh_custody_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    stage6_root = stage6_root.resolve()
    if "OneDrive" in str(stage6_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    contracts = _load_contracts(stage6_root)
    sources = _source_identities(stage6_root)
    codes = _code_identities(stage6_root)
    run_identity, run_binding = _run_binding(
        source_identities=sources,
        code_identities=codes,
        fresh_custody_binding=fresh_custody_binding,
    )
    run_root = _materialization_run_root(
        stage6_root,
        run_identity=run_identity,
        runtime_state_root=runtime_state_root,
    )
    ensure_durable_directory(run_root)
    with _exclusive_run_lock(run_root, run_identity=run_identity):
        assert_active_run_root_identity(run_root)
        partition_manifest = build_or_resume_partitions(
            stage6_root,
            run_root,
            run_identity=run_identity,
            source_identities=sources,
            registry=contracts["registry"],
        )
        assert_active_run_root_identity(run_root)
        plan = _build_plan(partition_manifest, run_identity=run_identity)
        _write_or_verify_plan(run_root, plan)
        assert_active_run_root_identity(run_root)
        existing_receipts = _load_completion_receipts(run_root)
        unfinished = list(
            iter_unfinished_lane_requests(
                plan,
                existing_receipts,
                checkpoint_root=run_root,
            )
        )
        assert_active_run_root_identity(run_root)
        resource_admission = _execute_unfinished_lanes(
            run_root,
            unfinished,
            requested_workers=requested_workers,
            registry=contracts["registry"],
            feature_manifest=contracts["feature_manifest"],
        )
        assert_active_run_root_identity(run_root)
        final_receipts = _load_completion_receipts(run_root)
        remaining = list(
            iter_unfinished_lane_requests(
                plan,
                final_receipts,
                checkpoint_root=run_root,
            )
        )
        if remaining:
            raise Stage6ContractError("materialization completed with unfinished lanes")
        assert_active_run_root_identity(run_root)
        lane_checkpoints = tuple(
            _verify_lane_checkpoint(
                run_root,
                plan_row,
                _completion_receipt_for_plan(plan_row, final_receipts),
                contracts["registry"],
            )
            for plan_row in plan
        )
        checkpoint_document_roots = tuple(
            checkpoint["self_sha256"] for checkpoint in lane_checkpoints
        )
        assert_active_run_root_identity(run_root)
        split_manifest = build_or_resume_split_assignments(
            run_root,
            lane_checkpoints,
            contracts["split_contract"],
            run_identity=run_identity,
        )
        for checkpoint in lane_checkpoints:
            validate_document_self_hash(checkpoint)
        if tuple(
            checkpoint["self_sha256"] for checkpoint in lane_checkpoints
        ) != checkpoint_document_roots:
            raise Stage6ContractError(
                "immutable checkpoint documents changed during split assignment"
            )
        assert_active_run_root_identity(run_root)
        manifest = _aggregate(
            run_root,
            plan,
            checkpoints=lane_checkpoints,
            run_identity=run_identity,
            run_binding=run_binding,
            partition_manifest=partition_manifest,
            quarantine_contract=contracts["quarantine_contract"],
            split_manifest=split_manifest,
            resource_admission=resource_admission,
        )
        assert_active_run_root_identity(run_root)
    return run_root, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage6-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--requested-workers",
        default=REQUESTED_WORKERS_DEFAULT,
        help=(
            "auto/0/blank uses available host CPU capacity; a positive integer "
            "sets a visible live-process cap without changing the 48 lanes"
        ),
    )
    args = parser.parse_args(argv)
    run_root, manifest = materialize(
        args.stage6_root,
        requested_workers=args.requested_workers,
    )
    print(
        json.dumps(
            {
                "event": "stage6_notebook01_materialization_complete",
                "run_root": str(run_root),
                "run_identity": manifest["run_identity"],
                "accepted_decision_row_count": manifest[
                    "accepted_decision_row_count"
                ],
                "quarantined_decision_row_count": manifest[
                    "quarantined_decision_row_count"
                ],
                "profile_snapshot_row_count_across_lanes": manifest[
                    "profile_snapshot_row_count_across_lanes"
                ],
                "fixed_order_semantic_root_digest": manifest[
                    "fixed_order_semantic_root_digest"
                ],
                "split_assignment_root_digest": manifest[
                    "split_assignment_root_digest"
                ],
                "private_lockbox_assignment_sealed": manifest[
                    "private_lockbox_assignment_sealed"
                ],
                "manifest_self_sha256": manifest["self_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
