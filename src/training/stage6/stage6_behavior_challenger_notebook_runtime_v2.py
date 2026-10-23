"""Colab runtime adapter for one Stage 6 behavior-challenger v2 shard.

The adapter consumes the already-frozen common/base feature arrays produced by
Notebook 12.  It never reads a fresh source or a fresh target.  Each invocation
runs one of the four independent scope/shard nodes, immediately checkpoints
every completed candidate-stage result, and publishes a compact primary
manifest for the existing deterministic result controller.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from stage6_behavior_json_publication_v2 import (
    load_json_with_recovery,
    write_json_exclusive,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    CANDIDATE_COUNT_PER_SHARD,
    FEATURE_RELEASE_BINDING,
    HALVING_STAGES,
    SCOPES,
    SOURCE_DEPENDENCY_DIGEST,
    assert_development_only_roles,
    assert_safe_model_features,
    build_candidate_checkpoint,
    candidate_registry_shard,
    checkpoint_path,
    evaluate_candidate,
    feature_contract,
    load_candidate_checkpoint,
    registry_digest,
    resolve_candidate_workers,
    search_run_id,
    shard_digest,
    stage_data_binding,
    write_candidate_checkpoint,
)
from stage6_expanded_feature_freeze_v2 import (
    BASE_FEATURE_COLUMNS,
    FEATURE_RUN_ID,
    _valid_completed_manifest_v2,
    expanded_feature_root,
)
from stage6_expanded_layers import (
    ACTIONS,
    LOGICAL_LANE_COUNT,
    canonical_json_bytes,
    seal_document,
    validate_document,
)
from stage6_expanded_tournament import (
    _stage_arrays,
    _verify_lane_identity,
    _write_stage_arrays,
    runtime_environment,
)
from stage6_notebooks02_07_runtime import sha256_file


HEARTBEAT_INTERVAL_SECONDS = 60.0
MAX_CANDIDATE_ATTEMPTS = 2
SUPPORTED_ACCELERATORS = ("CPU", "GPU", "TPU")
PREFERRED_NOTEBOOK_HOST = "v6e TPU"
PRIMARY_MANIFEST_NAME = "behavior-challenger-shard-manifest.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def notebook_shard_run_id(scope: str, shard: int) -> str:
    return _digest({
        "schema": "stage6_behavior_challenger_notebook_shard_v2",
        "feature_run_id": FEATURE_RUN_ID,
        "scope_search_run_id": search_run_id(scope, shard),
        "scope": scope,
        "shard": shard,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "fresh_targets_opened": False,
    })


def shard_work_root(stage_drive_root: Path, scope: str, shard: int) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "behavior_challenger_v2"
        / scope
        / f"shard-{shard}"
        / notebook_shard_run_id(scope, shard)
    )


def _available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _available_disk() -> int | None:
    try:
        target = Path("/content") if Path("/content").is_dir() else Path.cwd()
        return int(shutil.disk_usage(target).free)
    except Exception:
        return None


def _load_feature_manifest(stage_drive_root: Path) -> tuple[Path, dict[str, Any]]:
    """Authenticate the exact completed R5 feature freeze used by behavior-v2."""
    root = expanded_feature_root(stage_drive_root)
    path = root / "feature-freeze-manifest.json"
    if not path.is_file():
        raise RuntimeError("R5 Notebook 12 feature freeze must complete first")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    validate_document(candidate)
    package_binding = candidate.get("runtime_package_binding")
    if not isinstance(package_binding, Mapping):
        raise RuntimeError("R5 feature manifest lacks its runtime package binding")
    manifest = _valid_completed_manifest_v2(root, package_binding)
    expected_package = FEATURE_RELEASE_BINDING["input_package"]
    if (
        manifest is None
        or manifest.get("qualification_authority") is not False
        or package_binding.get("feature_run_id") != FEATURE_RUN_ID
        or package_binding.get("package_manifest_self_sha256")
        != expected_package["package_manifest_self_sha256"]
        or package_binding.get("precommit_self_sha256")
        != expected_package["precommit_self_sha256"]
    ):
        raise RuntimeError("R5 Notebook 12 feature freeze authority is invalid")
    return root, manifest


class ActiveWorkHeartbeat:
    """Emit phase progress independently while a long model fit is blocking."""

    def __init__(
        self,
        *,
        scope: str,
        shard: int,
        interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
        emit: Callable[[Mapping[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.scope = scope
        self.shard = int(shard)
        self.interval_seconds = float(interval_seconds)
        self.emit = emit or (
            lambda payload: print(json.dumps(payload, sort_keys=True), flush=True)
        )
        self.clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "initializing"
        self._stage: int | None = None
        self._denominator = 0
        self._completed = 0
        self._work_unit = "work_items"
        self._phase_started = self.clock()
        self._phase_completed_baseline = 0

    def update(
        self,
        *,
        phase: str,
        denominator: int,
        completed: int,
        stage: int | None = None,
        work_unit: str = "work_items",
    ) -> None:
        denominator = int(denominator)
        completed = int(completed)
        if denominator < 0 or completed < 0 or completed > denominator:
            raise ValueError("heartbeat completed/denominator is invalid")
        if not str(work_unit).strip():
            raise ValueError("heartbeat work_unit is required")
        with self._lock:
            phase_changed = str(phase) != self._phase or stage != self._stage
            counter_reset = completed < self._completed
            if phase_changed or counter_reset:
                self._phase_started = self.clock()
                self._phase_completed_baseline = completed
            self._phase = str(phase)
            self._stage = stage
            self._denominator = denominator
            self._completed = completed
            self._work_unit = str(work_unit)

    def payload(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            phase = self._phase
            stage = self._stage
            denominator = self._denominator
            completed = self._completed
            work_unit = self._work_unit
            started = self._phase_started
            completed_baseline = self._phase_completed_baseline
        elapsed = max(0.0, now - started)
        phase_completed = max(0, completed - completed_baseline)
        throughput = phase_completed / elapsed if elapsed > 0 and phase_completed > 0 else 0.0
        remaining = max(0, denominator - completed)
        eta = remaining / throughput if throughput > 0 else None
        return {
            "event": "stage6_behavior_challenger_active_heartbeat",
            "scope": self.scope,
            "shard": self.shard,
            "phase": phase,
            "stage": stage,
            "denominator": denominator,
            "completed": completed,
            "work_unit": work_unit,
            "phase_completed_baseline": completed_baseline,
            "phase_completed": phase_completed,
            "throughput_per_second": throughput,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "available_system_ram_bytes": _available_ram(),
            "available_disk_bytes": _available_disk(),
            "heartbeat_interval_seconds": self.interval_seconds,
            "fresh_targets_opened": False,
        }

    def emit_now(self) -> None:
        self.emit(self.payload())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.emit_now()

    def start(self) -> "ActiveWorkHeartbeat":
        if self._thread is not None:
            raise RuntimeError("heartbeat already started")
        with self._lock:
            self._phase_started = self.clock()
            self._phase_completed_baseline = self._completed
        self.emit_now()
        self._thread = threading.Thread(
            target=self._run,
            name=f"stage6-behavior-heartbeat-{self.scope}-{self.shard}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, emit_terminal: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, min(self.interval_seconds, 5.0)))
        if emit_terminal:
            self.emit_now()

    def __enter__(self) -> "ActiveWorkHeartbeat":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def _dataset_from_common_frame(
    common: pd.DataFrame,
    scope: str,
    whole_hand_binding: Mapping[str, Any],
    *,
    feature_manifest_self_sha256: str,
) -> dict[str, Any]:
    if scope not in SCOPES:
        raise ValueError(f"unknown behavior scope: {scope}")
    validate_document(whole_hand_binding)
    required = {
        "row_id",
        "game_id",
        "hand_id",
        "decision_utc",
        "street",
        "action_label",
        "temporal_role",
        *BASE_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(common.columns))
    if missing:
        raise RuntimeError(f"behavior common feature frame is missing columns: {missing}")
    if common["row_id"].astype(str).duplicated().any():
        raise RuntimeError("behavior common feature frame duplicates row_id")
    roles = common["temporal_role"].astype(str)
    assert_development_only_roles(sorted(roles.unique()))
    if scope == "preflop_behavior":
        mask = common["street"].astype(str).eq("preflop")
    else:
        mask = common["street"].astype(str).isin(("flop", "turn", "river"))
    frame = common.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{scope} common feature frame is empty")
    required_roles = {"development_train", "development_validation"}
    observed_roles = set(frame["temporal_role"].astype(str))
    if not required_roles.issubset(observed_roles):
        raise RuntimeError(f"{scope} is missing a development train/validation role")
    assert_safe_model_features(BASE_FEATURE_COLUMNS)
    matrix = frame[list(BASE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    labels = frame["action_label"].astype(str)
    if not labels.isin(ACTIONS).all():
        raise RuntimeError("behavior common target is outside the action registry")
    target = np.asarray([ACTIONS.index(label) for label in labels], dtype=np.int8)
    legal_columns = [f"base_legal_{action}" for action in ACTIONS]
    if not set(legal_columns).issubset(frame.columns):
        raise RuntimeError("behavior common feature frame lacks visible legality columns")
    legal = frame[legal_columns].to_numpy(dtype=bool)
    if not legal[np.arange(len(target)), target].all():
        raise RuntimeError("observed behavior target is illegal in its visible state")
    identity_frame = frame.loc[:, [
        "row_id",
        "game_id",
        "hand_id",
        "decision_utc",
        "temporal_role",
    ]].copy()
    return {
        "frame": identity_frame,
        "matrix": matrix,
        "target": target,
        "legal": legal,
        "feature_names": list(BASE_FEATURE_COLUMNS),
        "feature_manifest_self_sha256": feature_manifest_self_sha256,
        "whole_hand_temporal_assignment_binding": dict(whole_hand_binding),
    }


def load_behavior_dataset(stage_drive_root: Path, scope: str) -> dict[str, Any]:
    """Read only exact Notebook 12 common arrays for one behavior scope."""
    feature_root, manifest = _load_feature_manifest(stage_drive_root)
    whole_hand_binding = manifest.get("whole_hand_temporal_assignment_binding")
    if not isinstance(whole_hand_binding, Mapping):
        raise RuntimeError("feature manifest lacks the whole-hand temporal binding")
    validate_document(whole_hand_binding)
    frames: list[pd.DataFrame] = []
    for lane_number, lane in enumerate(manifest["lanes"]):
        if lane.get("lane_id") != f"lane-{lane_number:02d}":
            raise RuntimeError("behavior feature lane order changed")
        path = _verify_lane_identity(feature_root, lane["common"])
        frame = pd.read_parquet(path)
        if frame["temporal_role"].astype(str).str.contains(
            "fresh|lockbox|holdout|adjudication", case=False, regex=True
        ).any():
            raise RuntimeError("fresh target entered behavior challenger common arrays")
        frames.append(frame)
        print(json.dumps({
            "event": "stage6_behavior_common_lane_loaded",
            "scope": scope,
            "lane": lane_number,
            "completed": lane_number + 1,
            "denominator": LOGICAL_LANE_COUNT,
            "lane_rows": len(frame),
            "fresh_targets_opened": False,
        }, sort_keys=True), flush=True)
    common = pd.concat(frames, ignore_index=True)
    return _dataset_from_common_frame(
        common,
        scope,
        whole_hand_binding,
        feature_manifest_self_sha256=str(manifest["self_sha256"]),
    )


def _runtime_json_binding(scope: str, shard: int) -> dict[str, Any]:
    return seal_document({
        "schema_version": "stage6_behavior_shard_json_publication_binding_v2",
        "notebook_shard_run_id": notebook_shard_run_id(scope, shard),
        "scope_search_run_id": search_run_id(scope, shard),
        "scope": scope,
        "shard": shard,
        "feature_release_binding": FEATURE_RELEASE_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
    })


def _atomic_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    root: Path,
    scope: str,
    shard: int,
) -> None:
    write_json_exclusive(
        path,
        document,
        root=root,
        publication_binding=_runtime_json_binding(scope, shard),
    )


def _candidate_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_candidate(
        str(job["scope"]),
        job["entry"],
        job["paths"],
        job["stage"],
    )


def _rank_reports(reports: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        reports,
        key=lambda checkpoint: (
            float(checkpoint["result"]["validation_metrics"]["log_loss"]),
            float(
                checkpoint["result"]["validation_metrics"][
                    "multiclass_brier_score"
                ]
            ),
            str(checkpoint["candidate_id"]),
        ),
    )


def _completed_manifest(
    stage_drive_root: Path,
    work_root: Path,
    scope: str,
    shard: int,
) -> dict[str, Any] | None:
    path = work_root / PRIMARY_MANIFEST_NAME
    if not path.is_file():
        return None
    manifest = load_json_with_recovery(
        path,
        root=work_root,
        publication_binding=_runtime_json_binding(scope, shard),
    )
    feature_root, feature_manifest = _load_feature_manifest(stage_drive_root)
    del feature_root
    environment = runtime_environment()
    implementation = manifest.get("behavior_implementation_binding")
    whole_hand = manifest.get("whole_hand_temporal_assignment_binding")
    summaries = manifest.get("halving_stage_summaries")
    if (
        manifest.get("schema_version")
        != "stage6_behavior_challenger_shard_manifest_v2"
        or manifest.get("status") != "PASS_DEVELOPMENT_SUCCESSIVE_HALVING_COMPLETE"
        or manifest.get("notebook_shard_run_id") != notebook_shard_run_id(scope, shard)
        or manifest.get("scope_search_run_id") != search_run_id(scope, shard)
        or manifest.get("feature_run_id") != FEATURE_RUN_ID
        or manifest.get("feature_manifest_self_sha256")
        != feature_manifest.get("self_sha256")
        or manifest.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or manifest.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
        or manifest.get("environment_self_sha256") != environment.get("self_sha256")
        or not isinstance(implementation, Mapping)
        or implementation != BEHAVIOR_IMPLEMENTATION_BINDING
        or not isinstance(whole_hand, Mapping)
        or whole_hand != feature_manifest.get("whole_hand_temporal_assignment_binding")
        or not isinstance(summaries, list)
        or len(summaries) != len(HALVING_STAGES)
        or manifest.get("fresh_targets_opened") is not False
        or manifest.get("qualification_authority") is not False
    ):
        raise RuntimeError("completed behavior challenger shard manifest is invalid")
    validate_document(implementation)
    validate_document(whole_hand)
    last_summary: Mapping[str, Any] | None = None
    for expected_stage, identity in zip(HALVING_STAGES, summaries, strict=True):
        stage_number = int(expected_stage["stage"])
        summary_path = work_root / f"stage-{stage_number}-summary.json"
        summary = load_json_with_recovery(
            summary_path,
            root=work_root,
            publication_binding=_runtime_json_binding(scope, shard),
        )
        if (
            not isinstance(identity, Mapping)
            or identity.get("stage") != stage_number
            or identity.get("name") != summary_path.name
            or not summary_path.is_file()
            or summary_path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(summary_path) != identity.get("sha256")
        ):
            raise RuntimeError("completed behavior stage-summary identity changed")
        last_summary = summary
        binding = summary.get("stage_data_binding")
        if (
            summary.get("self_sha256") != identity.get("self_sha256")
            or summary.get("notebook_shard_run_id") != notebook_shard_run_id(scope, shard)
            or summary.get("scope_search_run_id") != search_run_id(scope, shard)
            or summary.get("stage") != stage_number
            or summary.get("environment_self_sha256") != environment["self_sha256"]
            or not isinstance(binding, Mapping)
        ):
            raise RuntimeError("completed behavior stage-summary authority changed")
        validate_document(binding)
        ranked = summary.get("ranked_results")
        if not isinstance(ranked, list) or len(ranked) != summary.get("evaluated_candidate_count"):
            raise RuntimeError("completed behavior checkpoint inventory changed")
        for row in ranked:
            candidate_id = str(row.get("candidate_id", ""))
            checkpoint_file = checkpoint_path(
                work_root, scope, shard, stage_number, candidate_id
            )
            if (
                not checkpoint_file.is_file()
                or checkpoint_file.stat().st_size != int(row.get("checkpoint_bytes", -1))
                or sha256_file(checkpoint_file) != row.get("checkpoint_sha256")
            ):
                raise RuntimeError("completed behavior checkpoint byte identity changed")
            checkpoint = load_candidate_checkpoint(
                checkpoint_file, scope, shard, stage_number, candidate_id,
                stage_data_binding_self_sha256=binding["self_sha256"],
                feature_manifest_self_sha256=feature_manifest["self_sha256"],
                whole_hand_assignment_self_sha256=whole_hand["self_sha256"],
                environment_self_sha256=environment["self_sha256"],
            )
            if checkpoint is None or checkpoint.get("self_sha256") != row.get("checkpoint_self_sha256"):
                raise RuntimeError("completed behavior checkpoint authority changed")
    if last_summary is None or manifest.get("terminal_survivor_ids") != last_summary.get("survivor_ids"):
        raise RuntimeError("completed behavior terminal survivors changed")
    return manifest


def run_behavior_challenger_shard(
    stage_drive_root: Path,
    scope: str,
    shard: int,
    requested_workers: str = "auto",
    accelerator: str = "TPU",
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run or resume one deterministic 128-candidate development shard."""
    accelerator = str(accelerator).strip().upper()
    if accelerator not in SUPPORTED_ACCELERATORS:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    work_root = shard_work_root(stage_drive_root, scope, shard)
    completed = _completed_manifest(stage_drive_root, work_root, scope, shard)
    if completed is not None:
        return {
            "status": "PASS",
            "scope": scope,
            "shard": shard,
            "run_identity": completed["notebook_shard_run_id"],
            "manifest_path": str(work_root / PRIMARY_MANIFEST_NAME),
            "manifest_self_sha256": completed["self_sha256"],
            "terminal_survivor_ids": completed["terminal_survivor_ids"],
            "reused_completed_result": True,
            "fresh_targets_opened": False,
            "qualification_authority": False,
        }

    started = time.monotonic()
    dataset: dict[str, Any] | None = None
    heartbeat = ActiveWorkHeartbeat(
        scope=scope,
        shard=shard,
        interval_seconds=heartbeat_interval_seconds,
    )
    heartbeat.update(phase="loading_development_features", denominator=1, completed=0, work_unit="datasets")
    heartbeat.start()
    try:
        heartbeat.update(phase="validating_runtime_environment", denominator=1, completed=0, work_unit="environments")
        environment = runtime_environment()
        heartbeat.update(phase="validating_runtime_environment", denominator=1, completed=1, work_unit="environments")
        heartbeat.update(phase="loading_development_features", denominator=1, completed=0, work_unit="datasets")
        dataset = load_behavior_dataset(stage_drive_root, scope)
        heartbeat.update(phase="loading_development_features", denominator=1, completed=1, work_unit="datasets")
        registry_rows = candidate_registry_shard(scope, shard)
        registry_by_id = {row["candidate_id"]: row for row in registry_rows}
        if len(registry_by_id) != CANDIDATE_COUNT_PER_SHARD:
            raise RuntimeError("behavior challenger shard registry changed")
        active_ids = list(registry_by_id)
        stage_summaries: list[dict[str, Any]] = []
        worker_reports: list[dict[str, Any]] = []
        whole_hand_binding = dataset["whole_hand_temporal_assignment_binding"]

        for stage in HALVING_STAGES:
            stage_number = int(stage["stage"])
            arrays = _stage_arrays(dataset, scope, shard, stage)
            binding = stage_data_binding(arrays, whole_hand_binding)
            reports: list[dict[str, Any]] = []
            unfinished: list[dict[str, Any]] = []
            for candidate_id in active_ids:
                entry = registry_by_id[candidate_id]
                checkpoint = load_candidate_checkpoint(
                    checkpoint_path(
                        work_root,
                        scope,
                        shard,
                        stage_number,
                        candidate_id,
                    ),
                    scope,
                    shard,
                    stage_number,
                    candidate_id,
                    stage_data_binding_self_sha256=binding["self_sha256"],
                    feature_manifest_self_sha256=dataset[
                        "feature_manifest_self_sha256"
                    ],
                    whole_hand_assignment_self_sha256=whole_hand_binding["self_sha256"],
                    environment_self_sha256=environment["self_sha256"],
                )
                if checkpoint is None:
                    unfinished.append(entry)
                else:
                    reports.append(checkpoint)

            heartbeat.update(
                phase=f"successive_halving_stage_{stage_number}",
                stage=stage_number,
                denominator=len(active_ids),
                completed=len(reports),
                work_unit="candidates",
            )
            if unfinished:
                worker_report = resolve_candidate_workers(
                    requested_workers,
                    len(unfinished),
                    arrays,
                    unfinished,
                    stage,
                    accelerator=accelerator,
                )
                worker_reports.append({"stage": stage_number, **worker_report})
                print(json.dumps({
                    "event": "stage6_behavior_challenger_stage_started",
                    "scope": scope,
                    "shard": shard,
                    "stage": stage_number,
                    "denominator": len(active_ids),
                    "completed": len(reports),
                    "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
                    "model_execution_device": "host_cpu_process",
                    "tree_training_uses_tpu_acceleration": False,
                    **worker_report,
                }, sort_keys=True), flush=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"stage6-behavior-v2-{scope}-{shard}-{stage_number}-"
                ) as temporary:
                    paths = _write_stage_arrays(Path(temporary), arrays)
                    pending: list[Mapping[str, Any]] = list(unfinished)
                    attempts = {str(row["candidate_id"]): 0 for row in pending}
                    while pending:
                        retry: list[Mapping[str, Any]] = []
                        with ProcessPoolExecutor(
                            max_workers=int(worker_report["live_candidate_workers"])
                        ) as executor:
                            futures = {}
                            for entry in pending:
                                candidate_id = str(entry["candidate_id"])
                                attempts[candidate_id] += 1
                                future = executor.submit(_candidate_worker, {
                                    "scope": scope,
                                    "entry": entry,
                                    "paths": paths,
                                    "stage": stage,
                                })
                                futures[future] = entry
                            for future in as_completed(futures):
                                entry = futures[future]
                                candidate_id = str(entry["candidate_id"])
                                try:
                                    result = future.result()
                                except Exception as error:
                                    print(json.dumps({
                                        "event": "stage6_behavior_challenger_candidate_failed",
                                        "scope": scope,
                                        "shard": shard,
                                        "stage": stage_number,
                                        "candidate_id": candidate_id,
                                        "attempt": attempts[candidate_id],
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                    }, sort_keys=True), flush=True)
                                    if attempts[candidate_id] < MAX_CANDIDATE_ATTEMPTS:
                                        retry.append(entry)
                                        continue
                                    raise RuntimeError(
                                        f"candidate {candidate_id} failed twice"
                                    ) from error
                                checkpoint = build_candidate_checkpoint(
                                    scope,
                                    shard,
                                    stage_number,
                                    candidate_id,
                                    stage_data_binding_self_sha256=binding["self_sha256"],
                                    feature_manifest_self_sha256=dataset[
                                        "feature_manifest_self_sha256"
                                    ],
                                    whole_hand_assignment_self_sha256=(
                                        whole_hand_binding["self_sha256"]
                                    ),
                                    environment_self_sha256=environment["self_sha256"],
                                    result=result,
                                )
                                destination = checkpoint_path(
                                    work_root,
                                    scope,
                                    shard,
                                    stage_number,
                                    candidate_id,
                                )
                                write_candidate_checkpoint(destination, checkpoint)
                                readback = load_candidate_checkpoint(
                                    destination,
                                    scope,
                                    shard,
                                    stage_number,
                                    candidate_id,
                                    stage_data_binding_self_sha256=binding["self_sha256"],
                                    feature_manifest_self_sha256=dataset[
                                        "feature_manifest_self_sha256"
                                    ],
                                    whole_hand_assignment_self_sha256=(
                                        whole_hand_binding["self_sha256"]
                                    ),
                                    environment_self_sha256=environment["self_sha256"],
                                )
                                if readback != checkpoint:
                                    raise RuntimeError("candidate checkpoint readback changed")
                                reports.append(checkpoint)
                                heartbeat.update(
                                    phase=f"successive_halving_stage_{stage_number}",
                                    stage=stage_number,
                                    denominator=len(active_ids),
                                    completed=len(reports),
                                    work_unit="candidates",
                                )
                        pending = retry

            if len(reports) != len(active_ids):
                raise RuntimeError("behavior challenger stage did not conserve candidates")
            ranked = _rank_reports(reports)
            survivor_count = min(int(stage["survivor_count"]), len(ranked))
            active_ids = [str(row["candidate_id"]) for row in ranked[:survivor_count]]
            stage_summary = seal_document({
                "schema_version": "stage6_behavior_challenger_halving_summary_v2",
                "notebook_shard_run_id": notebook_shard_run_id(scope, shard),
                "scope_search_run_id": search_run_id(scope, shard),
                "scope": scope,
                "shard": shard,
                "stage": stage_number,
                "stage_parameters": dict(stage),
                "stage_data_binding": binding,
                "environment_self_sha256": environment["self_sha256"],
                "evaluated_candidate_count": len(ranked),
                "survivor_count": survivor_count,
                "survivor_ids": active_ids,
                "ranked_results": [
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
                        "selected_temperature": row["result"]["selected_temperature"],
                        "best_iteration": row["result"]["best_iteration"],
                        "checkpoint_self_sha256": row["self_sha256"],
                        "checkpoint_bytes": checkpoint_path(
                            work_root,
                            scope,
                            shard,
                            stage_number,
                            str(row["candidate_id"]),
                        ).stat().st_size,
                        "checkpoint_sha256": sha256_file(checkpoint_path(
                            work_root,
                            scope,
                            shard,
                            stage_number,
                            str(row["candidate_id"]),
                        )),
                    }
                    for index, row in enumerate(ranked)
                ],
                "calibration": "legal_projected_temperature_selection",
                "candidate_internal_threads": 1,
                "fresh_targets_opened": False,
                "qualification_authority": False,
            })
            stage_summaries.append(stage_summary)
            _atomic_json(
                work_root / f"stage-{stage_number}-summary.json",
                stage_summary,
                root=work_root,
                scope=scope,
                shard=shard,
            )
            del arrays, reports, ranked
            gc.collect()

        manifest = seal_document({
            "schema_version": "stage6_behavior_challenger_shard_manifest_v2",
            "status": "PASS_DEVELOPMENT_SUCCESSIVE_HALVING_COMPLETE",
            "notebook_shard_run_id": notebook_shard_run_id(scope, shard),
            "scope_search_run_id": search_run_id(scope, shard),
            "feature_run_id": FEATURE_RUN_ID,
            "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
            "feature_manifest_self_sha256": dataset[
                "feature_manifest_self_sha256"
            ],
            "whole_hand_temporal_assignment_binding": whole_hand_binding,
            "environment_self_sha256": environment["self_sha256"],
            "scope": scope,
            "shard": shard,
            "registry_digest": registry_digest(scope),
            "shard_digest": shard_digest(scope, shard),
            "base_candidate_count": CANDIDATE_COUNT_PER_SHARD,
            "halving_stage_summary_self_sha256": [
                row["self_sha256"] for row in stage_summaries
            ],
            "halving_stage_summaries": [
                {
                    "stage": int(row["stage"]),
                    "name": f"stage-{int(row['stage'])}-summary.json",
                    "bytes": (work_root / f"stage-{int(row['stage'])}-summary.json").stat().st_size,
                    "sha256": sha256_file(work_root / f"stage-{int(row['stage'])}-summary.json"),
                    "self_sha256": row["self_sha256"],
                }
                for row in stage_summaries
            ],
            "halving_stage_data_bindings": [
                row["stage_data_binding"] for row in stage_summaries
            ],
            "terminal_survivor_ids": active_ids,
            "terminal_ranked_results": stage_summaries[-1]["ranked_results"],
            "feature_contract": feature_contract(scope),
            "worker_reports": worker_reports,
            "environment": environment,
            "notebook_accelerator": accelerator,
            "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
            "model_execution_device": "host_cpu_process",
            "tree_training_uses_tpu_acceleration": False,
            "requested_workers": requested_workers,
            "candidate_internal_threads": 1,
            "logical_data_lane_count": LOGICAL_LANE_COUNT,
            "checkpoint_after_every_candidate_stage": True,
            "calibration": "legal_projected_temperature_selection",
            "fresh_targets_opened": False,
            "lockbox_access_allowed": False,
            "qualification_authority": False,
            "assistant_installation_allowed": False,
            "recommendation_authority": False,
            "action_authority": False,
            "live_action_authority": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        work_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            work_root / PRIMARY_MANIFEST_NAME,
            manifest,
            root=work_root,
            scope=scope,
            shard=shard,
        )
        heartbeat.update(
            phase="shard_complete", denominator=len(active_ids), completed=len(active_ids), work_unit="candidates"
        )
        return {
            "status": "PASS",
            "scope": scope,
            "shard": shard,
            "run_identity": manifest["notebook_shard_run_id"],
            "manifest_path": str(work_root / PRIMARY_MANIFEST_NAME),
            "manifest_self_sha256": manifest["self_sha256"],
            "terminal_survivor_ids": manifest["terminal_survivor_ids"],
            "reused_completed_result": False,
            "fresh_targets_opened": False,
            "qualification_authority": False,
        }
    finally:
        heartbeat.stop()
        if dataset is not None:
            dataset.clear()
        gc.collect()


__all__ = [
    "ActiveWorkHeartbeat",
    "HEARTBEAT_INTERVAL_SECONDS",
    "PREFERRED_NOTEBOOK_HOST",
    "PRIMARY_MANIFEST_NAME",
    "SUPPORTED_ACCELERATORS",
    "_dataset_from_common_frame",
    "load_behavior_dataset",
    "notebook_shard_run_id",
    "run_behavior_challenger_shard",
    "shard_work_root",
]
