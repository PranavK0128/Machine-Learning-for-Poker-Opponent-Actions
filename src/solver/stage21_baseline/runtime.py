from __future__ import annotations

import concurrent.futures
import gc
import importlib
import inspect
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    ALGORITHMS,
    REQUESTED_WORKERS_DEFAULT,
    STALL_WARNING_SECONDS,
    WORK_LANES_PER_PRODUCER,
    atomic_write,
    canonical_json_bytes,
    stream_file_identity,
    write_self_hashed_json_stream,
    sha256_bytes,
    sha256_json,
    qualify_root,
    validate_self_hash,
    with_self_hash,
)
INNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

MAX_FAILURE_ERROR_MESSAGE_CHARS = 2_048

# Every per-lane JSON line is projected through this bounded field set.  The
# transient worker progress file may contain a few phase-specific counters, but
# active, completed, and failed operator records must expose the same stable
# observability contract without retaining a lane payload or traceback.
LANE_PROGRESS_DETAIL_FIELDS = (
    "lane_id",
    "root_id",
    "algorithm",
    "seed",
    "solver_continuation_semantic_sha256",
    "task_semantic_sha256",
    "status",
    "lane_phase",
    "event",
    "current_iteration",
    "target_iterations",
    "requested_target_iterations",
    "effective_target_iterations",
    "requested_target_tier",
    "effective_target_tier",
    "minimum_qualification_tier",
    "percent_complete",
    "iterations_per_second_since_resume",
    "eta_seconds",
    "last_checkpoint_iteration",
    "checkpoint_interval_iterations",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "node_count",
    "nonzero_regret_node_count",
    "qualification_look_index",
    "qualification_look_count",
    "qualification_look_percent_complete",
    "qualification_look_trigger",
    "qualification_look_qualified",
    "qualification_look_target_tier",
    "qualification_look_failed_checks",
    "completed_runouts",
    "total_runouts",
    "completed_oracle_passes",
    "total_oracle_passes",
    "completed_evaluation_work_units",
    "total_evaluation_work_units",
    "completed_aggregate_roots",
    "total_aggregate_roots",
    "qualified",
    "failed_checks",
    "qualification_failed_checks",
    "stopped_early_after_qualification",
    "progress_age_seconds",
    "stalled",
    "error_type",
    "error_message",
    "updated_unix_seconds",
)

# This is the single advertised contract for the bounded queue JSON emitted by
# ``emit_queue_progress``.  Notebook preflight cells import this tuple instead
# of maintaining a second, drift-prone copy.
QUEUE_PROGRESS_LOG_FIELDS = (
    "record_type",
    "active_lanes",
    "queued_lanes",
    "completed_lanes",
    "remaining_lanes",
    "failed_lanes",
    "qualified_completed_lanes",
    "not_qualified_completed_lanes",
    "completed_lanes_without_qualification_semantics",
    "fully_adjudicated_roots",
    "qualified_roots",
    "not_qualified_roots",
    "current_iteration_min",
    "current_iteration_max",
    "last_checkpoint_iteration_min",
    "last_checkpoint_iteration_max",
    "aggregate_percent_complete",
    "queue_eta_seconds",
    "stalled_lane_count",
    "stalled_lane_ids",
    "lane_detail_format",
    "active_lane_progress_one_record_per_line",
    "completed_lane_progress_one_record_per_line",
    "failed_lane_progress_one_record_per_line",
    "parent_rss_bytes",
    "available_system_ram_bytes",
    "parent_memory_release",
)


def _current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _release_process_memory() -> dict[str, Any]:
    """Release Python objects and trim glibc arenas when supported."""

    before = _current_rss_bytes()
    collected = gc.collect()
    malloc_trim_called = False
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            malloc_trim_called = bool(libc.malloc_trim(0))
        except (OSError, AttributeError):
            malloc_trim_called = False
    after = _current_rss_bytes()
    return {
        "gc_collected": collected,
        "malloc_trim_called": malloc_trim_called,
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "worker_or_parent_memory_released": True,
    }


def _progress_path(root: Path, lane_id: str) -> Path:
    safe = lane_id.replace("/", "_").replace("\\", "_")
    if safe != lane_id or ".." in Path(safe).parts:
        raise ValueError("unsafe progress lane ID")
    return root / f"{safe}.json"


def _read_progress(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _bounded_failure_message(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        message = "exception string conversion failed"
    return message[:MAX_FAILURE_ERROR_MESSAGE_CHARS]


def lane_progress_task_defaults(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit task-bound defaults for a bounded lane progress row."""

    requested_iterations = task.get("target_iterations")
    requested_tier = task.get("target_tier")
    minimum_tier = task.get("minimum_qualification_tier")
    return {
        "schema_version": "stage21_lane_progress_v2",
        "lane_id": str(task["lane_id"]),
        "root_id": str(task["root_id"]),
        "algorithm": task.get("algorithm"),
        "seed": task["seed"],
        "solver_continuation_semantic_sha256": task.get(
            "solver_continuation_semantic_sha256"
        ),
        "task_semantic_sha256": task.get("task_semantic_sha256"),
        "status": "NOT_STARTED",
        "lane_phase": "worker_startup",
        "event": "worker_admitted_waiting_for_first_progress",
        "current_iteration": None,
        "target_iterations": requested_iterations,
        "requested_target_iterations": requested_iterations,
        "effective_target_iterations": None,
        "requested_target_tier": requested_tier,
        "effective_target_tier": None,
        "minimum_qualification_tier": minimum_tier,
        "percent_complete": 0.0,
        "iterations_per_second_since_resume": None,
        "eta_seconds": None,
        "last_checkpoint_iteration": None,
        "checkpoint_interval_iterations": task.get(
            "checkpoint_interval_iterations"
        ),
        "checkpoint_sha256": None,
        "checkpoint_bytes": None,
        "node_count": None,
        "nonzero_regret_node_count": None,
        "qualification_look_index": None,
        "qualification_look_count": None,
        "qualification_look_percent_complete": None,
        "qualification_look_trigger": "NOT_STARTED",
        "qualification_look_qualified": None,
        "qualification_look_target_tier": None,
        "qualification_look_failed_checks": [],
        "completed_runouts": None,
        "total_runouts": None,
        "completed_oracle_passes": None,
        "total_oracle_passes": None,
        "completed_evaluation_work_units": None,
        "total_evaluation_work_units": None,
        "completed_aggregate_roots": None,
        "total_aggregate_roots": None,
        "qualified": None,
        "failed_checks": [],
        "qualification_failed_checks": [],
        "stopped_early_after_qualification": None,
        "progress_age_seconds": 0.0,
        "stalled": False,
        "error_type": None,
        "error_message": None,
        "updated_unix_seconds": None,
    }


def _lane_progress_detail_record(
    task: Mapping[str, Any],
    progress: Mapping[str, Any] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one active/completed/failed lane through the shared field set."""

    detail = lane_progress_task_defaults(task)
    for source in (progress, overrides):
        if not isinstance(source, Mapping):
            continue
        for field in LANE_PROGRESS_DETAIL_FIELDS:
            if field in source:
                detail[field] = source[field]

    # Task identity and requested bounds are authoritative even before a worker
    # has emitted its first progress heartbeat.
    defaults = lane_progress_task_defaults(task)
    for field in (
        "lane_id",
        "root_id",
        "algorithm",
        "seed",
        "solver_continuation_semantic_sha256",
        "task_semantic_sha256",
    ):
        detail[field] = defaults[field]
    for field in (
        "target_iterations",
        "requested_target_iterations",
        "requested_target_tier",
        "minimum_qualification_tier",
        "checkpoint_interval_iterations",
    ):
        if detail.get(field) is None:
            detail[field] = defaults[field]

    if detail.get("current_iteration") is None and detail.get(
        "effective_target_iterations"
    ) is not None:
        detail["current_iteration"] = detail["effective_target_iterations"]
    if detail.get("effective_target_iterations") is None and detail.get(
        "status"
    ) in {"COMPLETE", "QUALIFIED_COMPLETE", "NOT_QUALIFIED_COMPLETE"}:
        detail["effective_target_iterations"] = detail.get("current_iteration")

    for field in (
        "qualification_look_failed_checks",
        "failed_checks",
        "qualification_failed_checks",
    ):
        value = detail.get(field)
        detail[field] = list(value) if isinstance(value, (list, tuple)) else []
    if detail.get("error_message") is not None:
        detail["error_message"] = str(detail["error_message"])[
            :MAX_FAILURE_ERROR_MESSAGE_CHARS
        ]
    return {field: detail.get(field) for field in LANE_PROGRESS_DETAIL_FIELDS}


def _failed_lane_progress_record(
    task: Mapping[str, Any],
    progress: Mapping[str, Any] | None,
    error: BaseException,
) -> dict[str, Any]:
    row = progress if isinstance(progress, Mapping) else {}
    return _lane_progress_detail_record(
        task,
        row,
        overrides={
            "status": "FAILED",
            "event": "lane_failed",
            "lane_phase": row.get("lane_phase", "worker_exception"),
            "qualified": row.get("qualified"),
            "error_type": row.get("error_type", type(error).__name__),
            "error_message": row.get(
                "error_message", _bounded_failure_message(error)
            ),
            "progress_age_seconds": 0.0,
            "stalled": False,
        },
    )


def _write_final_progress(
    path: Path,
    task: Mapping[str, Any],
    *,
    qualified: bool | None,
    failed_checks: Sequence[str],
    current_iteration: int | None,
    requested_target_iterations: int | None,
    requested_target_tier: int | None,
    effective_target_tier: int | None,
    minimum_qualification_tier: int | None,
    stopped_early_after_qualification: bool | None,
) -> None:
    previous = _read_progress(path) or {}
    final_row = {
        **lane_progress_task_defaults(task),
        **previous,
        "schema_version": "stage21_lane_progress_v2",
        "lane_id": task["lane_id"],
        "root_id": task["root_id"],
        "algorithm": task.get("algorithm"),
        "seed": task["seed"],
        "solver_continuation_semantic_sha256": task.get(
            "solver_continuation_semantic_sha256"
        ),
        "task_semantic_sha256": task.get("task_semantic_sha256"),
        "event": "durable_lane_completion_readback_verified",
        "status": "QUALIFIED_COMPLETE"
        if qualified is True
        else "NOT_QUALIFIED_COMPLETE"
        if qualified is False
        else "COMPLETE",
        "lane_phase": "durable_completion",
        "qualified": qualified,
        "failed_checks": list(failed_checks),
        "qualification_failed_checks": list(failed_checks),
        "current_iteration": current_iteration,
        "effective_target_iterations": current_iteration,
        "target_iterations": requested_target_iterations,
        "requested_target_iterations": requested_target_iterations,
        "requested_target_tier": requested_target_tier,
        "effective_target_tier": effective_target_tier,
        "minimum_qualification_tier": minimum_qualification_tier,
        "stopped_early_after_qualification": stopped_early_after_qualification,
        "percent_complete": 100.0,
        "eta_seconds": 0.0,
        "last_checkpoint_iteration": current_iteration,
        "checkpoint_interval_iterations": task.get(
            "checkpoint_interval_iterations"
        ),
        "progress_age_seconds": 0.0,
        "stalled": False,
        "updated_unix_seconds": time.time(),
    }
    atomic_write(
        path,
        canonical_json_bytes(final_row),
    )


def _read_cgroup_cpu_limit(
    *,
    cgroup_file: Path = Path("/proc/self/cgroup"),
    v2_mount: Path = Path("/sys/fs/cgroup"),
    v1_mounts: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Return the tightest effective CPU quota across a cgroup hierarchy."""

    cgroup_rows: list[str] = []
    try:
        cgroup_rows = cgroup_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        pass

    def hierarchy(leaf: Path, mount: Path) -> list[Path]:
        leaf = leaf.resolve()
        mount = mount.resolve()
        if leaf != mount and mount not in leaf.parents:
            return [mount]
        rows = [leaf]
        while rows[-1] != mount:
            rows.append(rows[-1].parent)
        return rows

    v2_row = next((row for row in cgroup_rows if row.startswith("0::")), None)
    if v2_row is not None:
        relative = v2_row.split(":", 2)[2].lstrip("/")
        levels: list[dict[str, Any]] = []
        probe_incomplete = False
        for root in hierarchy(v2_mount / relative, v2_mount):
            candidate = root / "cpu.max"
            try:
                values = candidate.read_text(encoding="utf-8").strip().split()
                if len(values) != 2:
                    raise ValueError("invalid cgroup-v2 CPU quota")
                quota_text, period_text = values
                period = int(period_text)
                if period <= 0:
                    raise ValueError("invalid cgroup-v2 CPU period")
                quota = None if quota_text == "max" else int(quota_text)
                if quota is not None and quota <= 0:
                    raise ValueError("invalid cgroup-v2 CPU quota")
            except (OSError, ValueError):
                probe_incomplete = True
                continue
            levels.append(
                {
                    "path": str(candidate),
                    "quota_us": quota,
                    "period_us": period,
                    "cpu_limit": (
                        None
                        if quota is None
                        else max(1, math.floor(quota / period))
                    ),
                }
            )
        if probe_incomplete or not levels:
            return {
                "version": 2,
                "path": None,
                "quota_us": None,
                "period_us": None,
                "cpu_limit": 1,
                "probe_incomplete": True,
                "fail_closed_default_cpu_limit": 1,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
            }
        bounded = [row for row in levels if row["cpu_limit"] is not None]
        if bounded:
            selected = min(bounded, key=lambda row: int(row["cpu_limit"]))
            return {
                "version": 2,
                **selected,
                "probe_incomplete": False,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": True,
            }
        return {
            "version": 2,
            **levels[0],
            "probe_incomplete": False,
            "hierarchy_levels": levels,
            "hierarchy_level_count": len(levels),
            "selected_tightest_ancestor": False,
        }

    v1_relative = next(
        (
            row.split(":", 2)[2].lstrip("/")
            for row in cgroup_rows
            if "cpu" in row.split(":", 2)[1].split(",")
        ),
        "",
    )
    candidate_mounts = list(
        v1_mounts
        if v1_mounts is not None
        else (
            Path("/sys/fs/cgroup/cpu"),
            Path("/sys/fs/cgroup/cpu,cpuacct"),
        )
    )
    if any(
        "cpu" in row.split(":", 2)[1].split(",") for row in cgroup_rows
    ):
        for mount in candidate_mounts:
            leaf = mount / v1_relative
            if not leaf.exists() and not mount.exists():
                continue
            levels = []
            probe_incomplete = False
            for root in hierarchy(leaf, mount):
                quota_path = root / "cpu.cfs_quota_us"
                period_path = root / "cpu.cfs_period_us"
                try:
                    quota_raw = int(quota_path.read_text(encoding="utf-8").strip())
                    period = int(period_path.read_text(encoding="utf-8").strip())
                    if period <= 0 or quota_raw == 0:
                        raise ValueError("invalid cgroup-v1 CPU quota")
                    quota = None if quota_raw < 0 else quota_raw
                except (OSError, ValueError):
                    probe_incomplete = True
                    continue
                levels.append(
                    {
                        "path": str(root),
                        "quota_us": quota,
                        "period_us": period,
                        "cpu_limit": (
                            None
                            if quota is None
                            else max(1, math.floor(quota / period))
                        ),
                    }
                )
            if probe_incomplete or not levels:
                return {
                    "version": 1,
                    "path": None,
                    "quota_us": None,
                    "period_us": None,
                    "cpu_limit": 1,
                    "probe_incomplete": True,
                    "fail_closed_default_cpu_limit": 1,
                    "hierarchy_levels": levels,
                    "hierarchy_level_count": len(levels),
                    "selected_tightest_ancestor": False,
                }
            bounded = [row for row in levels if row["cpu_limit"] is not None]
            if bounded:
                selected = min(
                    bounded, key=lambda row: int(row["cpu_limit"])
                )
                return {
                    "version": 1,
                    **selected,
                    "probe_incomplete": False,
                    "hierarchy_levels": levels,
                    "hierarchy_level_count": len(levels),
                    "selected_tightest_ancestor": True,
                }
            return {
                "version": 1,
                **levels[0],
                "probe_incomplete": False,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
            }
    return {
        "version": None,
        "path": None,
        "quota_us": None,
        "period_us": None,
        "cpu_limit": None,
        "probe_incomplete": False,
    }


def _read_cgroup_memory_limit(
    *,
    cgroup_file: Path = Path("/proc/self/cgroup"),
    v2_mount: Path = Path("/sys/fs/cgroup"),
    v1_mount: Path = Path("/sys/fs/cgroup/memory"),
) -> dict[str, Any]:
    """Return the tightest remaining allowance across the cgroup hierarchy."""

    try:
        cgroup_rows = cgroup_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        cgroup_rows = []

    def hierarchy(leaf: Path, mount: Path) -> list[Path]:
        leaf = leaf.resolve()
        mount = mount.resolve()
        if leaf != mount and mount not in leaf.parents:
            return [mount]
        rows = [leaf]
        while rows[-1] != mount:
            rows.append(rows[-1].parent)
        return rows

    v2_row = next((row for row in cgroup_rows if row.startswith("0::")), None)
    if v2_row is not None:
        mount = v2_mount
        relative = v2_row.split(":", 2)[2].lstrip("/")
        levels: list[dict[str, Any]] = []
        observed_controller = False
        probe_incomplete = False
        for root in hierarchy(mount / relative, mount):
            limit_path = root / "memory.max"
            current_path = root / "memory.current"
            try:
                limit_text = limit_path.read_text(encoding="utf-8").strip()
                observed_controller = True
                limit = None if limit_text == "max" else int(limit_text)
                if limit is not None and limit <= 0:
                    raise ValueError("invalid cgroup-v2 memory limit")
            except (OSError, ValueError):
                probe_incomplete = True
                continue
            current = None
            if limit is not None:
                try:
                    current = int(
                        current_path.read_text(encoding="utf-8").strip()
                    )
                except (OSError, ValueError):
                    probe_incomplete = True
                    continue
            levels.append(
                {
                    "limit_path": str(limit_path),
                    "current_path": str(current_path),
                    "limit_bytes": limit,
                    "current_bytes": current,
                    "remaining_bytes": (
                        None if limit is None else max(0, limit - int(current))
                    ),
                }
            )
        if probe_incomplete or not observed_controller:
            return {
                "version": 2,
                "limit_path": None,
                "current_path": None,
                "limit_bytes": None,
                "current_bytes": None,
                "remaining_bytes": None,
                "bounded": any(
                    row["remaining_bytes"] is not None for row in levels
                ),
                "probe_incomplete": True,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
            }
        bounded_levels = [
            row for row in levels if row["remaining_bytes"] is not None
        ]
        if bounded_levels:
            selected = min(
                bounded_levels, key=lambda row: int(row["remaining_bytes"])
            )
            return {
                "version": 2,
                **selected,
                "bounded": True,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": True,
                "probe_incomplete": False,
            }
        if observed_controller:
            return {
                "version": 2,
                "limit_path": None,
                "current_path": None,
                "limit_bytes": None,
                "current_bytes": None,
                "remaining_bytes": None,
                "bounded": False,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
                "probe_incomplete": False,
            }

    v1_row = next(
        (
            row
            for row in cgroup_rows
            if "memory" in row.split(":", 2)[1].split(",")
        ),
        None,
    )
    if v1_row is not None:
        mount = v1_mount
        relative = v1_row.split(":", 2)[2].lstrip("/")
        levels = []
        observed_controller = False
        probe_incomplete = False
        for root in hierarchy(mount / relative, mount):
            limit_path = root / "memory.limit_in_bytes"
            current_path = root / "memory.usage_in_bytes"
            try:
                limit = int(limit_path.read_text(encoding="utf-8").strip())
                observed_controller = True
                effective_limit = (
                    None if limit <= 0 or limit >= (1 << 60) else limit
                )
            except (OSError, ValueError):
                probe_incomplete = True
                continue
            current = None
            if effective_limit is not None:
                try:
                    current = int(
                        current_path.read_text(encoding="utf-8").strip()
                    )
                except (OSError, ValueError):
                    probe_incomplete = True
                    continue
            levels.append(
                {
                    "limit_path": str(limit_path),
                    "current_path": str(current_path),
                    "limit_bytes": effective_limit,
                    "current_bytes": current,
                    "remaining_bytes": (
                        None
                        if effective_limit is None
                        else max(0, effective_limit - int(current))
                    ),
                }
            )
        if probe_incomplete or not observed_controller:
            return {
                "version": 1,
                "limit_path": None,
                "current_path": None,
                "limit_bytes": None,
                "current_bytes": None,
                "remaining_bytes": None,
                "bounded": any(
                    row["remaining_bytes"] is not None for row in levels
                ),
                "probe_incomplete": True,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
            }
        bounded_levels = [
            row for row in levels if row["remaining_bytes"] is not None
        ]
        if bounded_levels:
            selected = min(
                bounded_levels, key=lambda row: int(row["remaining_bytes"])
            )
            return {
                "version": 1,
                **selected,
                "bounded": True,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": True,
                "probe_incomplete": False,
            }
        if observed_controller:
            return {
                "version": 1,
                "limit_path": None,
                "current_path": None,
                "limit_bytes": None,
                "current_bytes": None,
                "remaining_bytes": None,
                "bounded": False,
                "hierarchy_levels": levels,
                "hierarchy_level_count": len(levels),
                "selected_tightest_ancestor": False,
                "probe_incomplete": False,
            }
    return {
        "version": None,
        "limit_path": None,
        "current_path": None,
        "limit_bytes": None,
        "current_bytes": None,
        "remaining_bytes": None,
        "bounded": False,
        "probe_incomplete": False,
    }


def cpu_capacity_snapshot() -> dict[str, Any]:
    os_count_raw = os.cpu_count()
    os_count = max(1, int(os_count_raw)) if os_count_raw is not None else 1
    affinity_count: int | None = None
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            observed = len(affinity(0))
            affinity_count = observed if observed > 0 else None
        except (OSError, TypeError, ValueError):
            pass
    cgroup = _read_cgroup_cpu_limit()
    caps = [os_count]
    if affinity_count is not None:
        caps.append(affinity_count)
    if cgroup["cpu_limit"] is not None:
        caps.append(int(cgroup["cpu_limit"]))
    return {
        "os_logical_cpu_count": os_count,
        "affinity_cpu_count": affinity_count,
        "cgroup": cgroup,
        "effective_logical_cpu_count": max(1, min(caps)),
    }


def detect_logical_cpu_count() -> int:
    """Return CPUs available after OS, affinity, and cgroup quota caps."""

    return int(cpu_capacity_snapshot()["effective_logical_cpu_count"])


def parse_requested_workers(value: Any) -> int:
    if value is None or (
        isinstance(value, str)
        and (not value.strip() or value.strip().lower() == "auto")
    ):
        return REQUESTED_WORKERS_DEFAULT
    try:
        workers = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "REQUESTED_WORKERS must be Auto/blank/0 or a positive integer"
        ) from error
    if workers < 0:
        raise ValueError(
            "REQUESTED_WORKERS must be Auto/blank/0 or a positive integer"
        )
    return workers


def resolve_worker_count(
    value: Any,
    *,
    unfinished_lane_count: int,
    logical_cpu_count: int | None = None,
    memory_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve workers from usable CPU capacity and unfinished logical lanes.

    RAM is observed for logs only. It never reduces Auto, rejects a positive
    operator override, or defers work.
    """

    unfinished = int(unfinished_lane_count)
    if unfinished < 0:
        raise ValueError("unfinished lane count cannot be negative")
    requested = parse_requested_workers(value)
    cpu_snapshot = (
        cpu_capacity_snapshot()
        if logical_cpu_count is None
        else {
            "os_logical_cpu_count": int(logical_cpu_count),
            "affinity_cpu_count": int(logical_cpu_count),
            "cgroup": {
                "version": None,
                "path": None,
                "quota_us": None,
                "period_us": None,
                "cpu_limit": None,
            },
            "effective_logical_cpu_count": int(logical_cpu_count),
            "injected_for_resolution": True,
        }
    )
    detected = int(cpu_snapshot["effective_logical_cpu_count"])
    if detected <= 0:
        raise ValueError("detected logical CPU count must be positive")
    logical_lane_limit = WORK_LANES_PER_PRODUCER
    runtime_limit = min(logical_lane_limit, detected)
    memory_evidence = dict(
        memory_snapshot if memory_snapshot is not None else _memory_snapshot()
    )
    available_raw = memory_evidence.get("available_bytes")
    available = None if available_raw is None else int(available_raw)
    if available is not None and available < 0:
        raise ValueError("available memory cannot be negative")
    if unfinished > 0 and requested > 0 and requested > runtime_limit:
        raise ValueError(
            "explicit REQUESTED_WORKERS exceeds the logical CPU/48-lane cap"
        )
    requested_target = runtime_limit if requested == 0 else requested
    resolved = min(requested_target, unfinished)
    clipped_by_cpu = requested == 0 and detected < logical_lane_limit
    clipped_by_lane_limit = requested == 0 and detected > logical_lane_limit
    clipped_by_unfinished = requested_target > unfinished
    clipping_reasons = []
    if clipped_by_cpu:
        clipping_reasons.append("detected_logical_cpu_count")
    if clipped_by_lane_limit:
        clipping_reasons.append("logical_lane_limit_48")
    if clipped_by_unfinished:
        clipping_reasons.append("unfinished_lane_count")
    return with_self_hash(
        {
            "schema_version": "stage21_worker_resolution_v5",
            "requested_workers": requested,
            "requested_workers_mode": "auto" if requested == 0 else "explicit",
            "detected_logical_cpu_count": detected,
            "cpu_capacity_snapshot": cpu_snapshot,
            "affinity_cpu_count": cpu_snapshot["affinity_cpu_count"],
            "cgroup_cpu_limit": cpu_snapshot["cgroup"]["cpu_limit"],
            "logical_lane_limit": logical_lane_limit,
            "runtime_worker_limit": runtime_limit,
            "ram_worker_admission_enabled": False,
            "available_memory_bytes_at_resolution": available,
            "memory_capacity_snapshot": memory_evidence,
            "unfinished_lane_count": unfinished,
            "resolved_live_worker_target": resolved,
            "worker_count_clipped_by_cpu": clipped_by_cpu,
            "worker_count_clipped_by_logical_lane_limit": clipped_by_lane_limit,
            "worker_count_clipped_by_unfinished_lane_count": clipped_by_unfinished,
            "clipping_reasons": clipping_reasons,
            "resolution_rule": (
                "auto=min(48,affinity_and_cgroup_cpu_cap,unfinished_lane_count);"
                " explicit=fail_if_above_cpu_or_48_lane_cap"
            ),
            "runtime_or_accelerator_name_used_for_cap": False,
        }
    )


def _memory_snapshot() -> dict[str, Any]:
    os_total: int | None = None
    os_available: int | None = None
    os_source: str | None = None
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        os_total = int(memory.total)
        os_available = int(memory.available)
        os_source = "psutil.virtual_memory"
    except Exception:
        if sys.platform != "win32" and hasattr(os, "sysconf"):
            try:
                page = int(os.sysconf("SC_PAGE_SIZE"))
                os_total = int(os.sysconf("SC_PHYS_PAGES")) * page
                os_available = int(os.sysconf("SC_AVPHYS_PAGES")) * page
                os_source = "os.sysconf"
            except (ValueError, OSError):
                pass
    cgroup = _read_cgroup_memory_limit()
    cgroup_remaining = cgroup.get("remaining_bytes")
    if cgroup.get("probe_incomplete") is True:
        effective_available = None
        source = "cgroup_probe_incomplete_fail_closed"
    else:
        candidates = [
            int(value)
            for value in (os_available, cgroup_remaining)
            if value is not None
        ]
        effective_available = min(candidates) if candidates else None
        source = None
    if source is None:
        if os_available is not None and cgroup_remaining is not None:
            source = "minimum_of_os_available_and_cgroup_remaining"
        elif cgroup_remaining is not None:
            source = "cgroup_remaining"
        elif os_available is not None:
            source = "os_available"
    return {
        "total_bytes": os_total,
        "available_bytes": effective_available,
        "os_available_bytes": os_available,
        "os_probe_source": os_source,
        "cgroup": cgroup,
        "effective_available_bytes_source": source,
    }


def controller_memory_observation(
    *,
    role: str,
    memory_snapshot: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Record controller RAM without gating or rationing controller work."""

    if not str(role):
        raise ValueError("controller role is required")
    memory = dict(memory_snapshot) if memory_snapshot is not None else _memory_snapshot()
    return with_self_hash(
        {
            "schema_version": "stage21_controller_memory_observation_v1",
            "role": str(role),
            "memory": memory,
            "ram_admission_enabled": False,
            "worker_count": 0,
            "worker_resolver_reused_for_controller": False,
            "status": "OBSERVED_NOT_GATED",
        }
    )


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return {"visible": bool(rows), "devices": rows}
    except Exception as error:
        return {"visible": False, "devices": [], "probe": type(error).__name__}


def resource_preflight(
    *,
    work_root: Path,
    requested_workers: int,
    estimated_required_disk_bytes: int,
    worker_resolution: Mapping[str, Any] | None = None,
    memory_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if requested_workers < 0:
        raise ValueError("planned live worker count cannot be negative")
    if estimated_required_disk_bytes < 0:
        raise ValueError("disk estimate must be nonnegative")
    work_root.mkdir(parents=True, exist_ok=True)
    memory = dict(memory_snapshot) if memory_snapshot is not None else _memory_snapshot()
    disk = shutil.disk_usage(work_root)
    disk_ok = estimated_required_disk_bytes <= int(disk.free * 0.90)
    detected_logical_cpu_count = (
        int(worker_resolution["detected_logical_cpu_count"])
        if worker_resolution is not None
        else detect_logical_cpu_count()
    )
    receipt = with_self_hash(
        {
            "schema_version": "stage21_disk_resource_preflight_and_ram_observation_v1",
            "planned_live_workers": requested_workers,
            "no_compute_required": requested_workers == 0,
            "logical_cpu_count": detected_logical_cpu_count,
            "cpu_capacity_snapshot": (
                worker_resolution.get("cpu_capacity_snapshot")
                if worker_resolution is not None
                else cpu_capacity_snapshot()
            ),
            "affinity_cpu_count": (
                worker_resolution.get("affinity_cpu_count")
                if worker_resolution is not None
                else None
            ),
            "cgroup_cpu_limit": (
                worker_resolution.get("cgroup_cpu_limit")
                if worker_resolution is not None
                else None
            ),
            "memory": memory,
            "ram_admission_enabled": False,
            "memory_is_observational_only": True,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "estimated_required_disk_bytes": estimated_required_disk_bytes,
            "memory_ok": True,
            "disk_ok": disk_ok,
            "gpu": _gpu_snapshot(),
            "worker_count_clipped_by_cpu": (
                worker_resolution.get("worker_count_clipped_by_cpu") is True
                if worker_resolution is not None
                else requested_workers > detected_logical_cpu_count
            ),
            "requested_workers": (
                worker_resolution.get("requested_workers")
                if worker_resolution is not None
                else requested_workers
            ),
            "requested_workers_mode": (
                worker_resolution.get("requested_workers_mode")
                if worker_resolution is not None
                else "explicit"
            ),
            "runtime_worker_limit": (
                worker_resolution.get("runtime_worker_limit")
                if worker_resolution is not None
                else min(WORK_LANES_PER_PRODUCER, detected_logical_cpu_count)
            ),
            "resolved_live_worker_target": (
                worker_resolution.get("resolved_live_worker_target")
                if worker_resolution is not None
                else requested_workers
            ),
            "unfinished_lane_count_at_resolution": (
                worker_resolution.get("unfinished_lane_count")
                if worker_resolution is not None
                else None
            ),
            "worker_count_clipping_reasons": (
                worker_resolution.get("clipping_reasons", [])
                if worker_resolution is not None
                else []
            ),
            "worker_resolution_rule": (
                worker_resolution.get("resolution_rule")
                if worker_resolution is not None
                else "pre-resolved explicit worker count"
            ),
            "worker_count_clipped_by_runtime_name": False,
            "gpu_claim": (
                "CFR recursion is CPU-bound; GPU visibility is recorded but does not "
                "change solver semantics or fabricate GPU acceleration"
            ),
        }
    )
    if not disk_ok:
        raise OSError(
            f"worker preflight failed: need {estimated_required_disk_bytes} disk bytes; "
            f"free={disk.free}"
        )
    return receipt


def _load_callable(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("worker callable must use module:function syntax")
    module = importlib.import_module(module_name)
    function = getattr(module, attribute)
    if not callable(function):
        raise TypeError(f"worker target is not callable: {path}")
    return function


def _lane_result_path(lane_root: Path, lane_id: str) -> Path:
    safe = lane_id.replace("/", "_").replace("\\", "_")
    if safe != lane_id or ".." in Path(safe).parts:
        raise ValueError("unsafe lane ID")
    return lane_root / f"{safe}.json"


def _lane_completion_receipt_path(lane_root: Path, lane_id: str) -> Path:
    safe = lane_id.replace("/", "_").replace("\\", "_")
    if safe != lane_id or ".." in Path(safe).parts:
        raise ValueError("unsafe lane ID")
    return lane_root / "completion_receipts" / f"{safe}.json"


def _validated_completed_lane_receipt(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    sidecar_path = _lane_completion_receipt_path(path.parent, str(expected["lane_id"]))
    if not path.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        validate_self_hash(sidecar)
        for key in (
            "stage",
            "token",
            "lane_id",
            "root_id",
            "algorithm",
            "seed",
            "solver_continuation_semantic_sha256",
            "task_semantic_sha256",
        ):
            if sidecar.get(key) != expected.get(key):
                return None
        if (
            sidecar.get("status") != "COMPLETE"
            or sidecar.get("lane_file_self_hash_verified_at_install") is not True
        ):
            return None
        identity = stream_file_identity(path)
        if (
            identity["size"] != int(sidecar["lane_file_size"])
            or identity["sha256"] != sidecar["lane_file_sha256"]
        ):
            return None
        return {
            "sidecar": sidecar,
            "sha256": identity["sha256"],
            "size": identity["size"],
            "self_sha256": sidecar["lane_file_self_sha256"],
        }
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def validate_completed_lane(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    receipt = _validated_completed_lane_receipt(path, expected)
    if receipt is None:
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_self_hash(document)
        return document
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _task_semantic_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transport locations while retaining every byte identity.

    A Drive mount may move without changing science. Paths therefore are not
    semantic, while all corresponding manifest/archive byte hashes are. This
    also keeps task identity reproducible across clean roots.
    """

    transport_keys = {
        "path",
        "staged_path",
        "source_path",
        "manifest_path",
        "payload_path",
        "continuation_manifest_path",
        "continuation_payload_path",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key not in transport_keys
                and not str(key).endswith("_path")
                and not str(key).startswith("local_")
                and key not in {
                    "task_semantic_sha256",
                    "solver_continuation_semantic_sha256",
                }
            }
        if isinstance(value, (list, tuple)):
            return [scrub(item) for item in value]
        return value

    return scrub(task)


def _solver_continuation_semantic_payload(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable solver-state identity for checkpoint continuation.

    A tier, terminal target, checkpoint cadence, or held-out sample-size change
    must invalidate a completed result but must not invalidate mathematically
    identical regrets, strategy sums, or RNG streams.  All game, root, solver,
    seed, range, continuation-authority, and package-source identities remain
    bound through the rest of the task payload and execution identity.
    """

    execution_only = {
        "target_tier",
        "minimum_qualification_tier",
        "target_iterations",
        "qualification_terminal_iterations",
        "checkpoint_interval_iterations",
        "evaluation_deals_per_family_role",
        "final_holdout_deals_per_family_role",
        "evaluation_panel_count",
    }
    return {
        key: value
        for key, value in _task_semantic_payload(task).items()
        if key not in execution_only
    }


def _checkpoint_iteration_hint(lane_root: Path, lane_id: str) -> int | None:
    """Read only the small canonical checkpoint header for queue ordering."""

    safe = lane_id.replace("/", "_").replace("\\", "_")
    if safe != lane_id or ".." in Path(safe).parts:
        return None
    path = lane_root / "checkpoints" / f"{safe}.json"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(256 * 1024)
        match = re.search(rb'"iteration":([0-9]+)', header)
        return None if match is None else int(match.group(1))
    except OSError:
        return None


def _lane_schedule_row(
    task: Mapping[str, Any], lane_root: Path
) -> dict[str, Any]:
    lane_id = str(task["lane_id"])
    checkpoint_iteration = _checkpoint_iteration_hint(lane_root, lane_id)
    target = max(0, int(task.get("target_iterations", 0)))
    remaining = max(0, target - int(checkpoint_iteration or 0))
    root = task.get("root")
    root_kind = str(root.get("root_kind", "")) if isinstance(root, Mapping) else ""
    if task.get("role") == "strict_result_member_and_crossplay_validation_shard":
        weights = {"approximate": 100, "restricted_exact": 20, "forced": 1}
        estimated_cost = sum(
            weights.get(str(binding.get("root", {}).get("root_kind", "")), 1)
            for binding in task.get("root_bindings", ())
        )
    elif root_kind == "restricted_exact":
        estimated_cost = remaining * 5_000
    elif root_kind == "forced":
        estimated_cost = {
            "preflop": 17_296,
            "flop": 45,
            "turn": 44,
            "river": 1_024,
        }.get(str(task.get("stage")), 1)
    else:
        estimated_cost = remaining
    return {
        "lane_id": lane_id,
        "checkpoint_iteration_hint": checkpoint_iteration,
        "resumed_checkpoint_present": checkpoint_iteration is not None,
        "estimated_remaining_cost_units": estimated_cost,
    }


def _invoke_one_lane(
    worker_path: str,
    task: Mapping[str, Any],
    lane_root_text: str,
    progress_path_text: str,
) -> dict[str, Any]:
    for key, value in INNER_THREAD_ENV.items():
        os.environ[key] = value
    lane_root = Path(lane_root_text)
    progress_path = Path(progress_path_text)
    output_path = _lane_result_path(lane_root, str(task["lane_id"]))
    sidecar_path = _lane_completion_receipt_path(lane_root, str(task["lane_id"]))
    payload: dict[str, Any] | None = None
    handle: dict[str, Any] | None = None
    worker = None
    raw_payload = None
    write_receipt = None
    adjudication = None
    required = None
    try:
        readback_receipt = _validated_completed_lane_receipt(output_path, task)
        reused = readback_receipt is not None
        if readback_receipt is None:
            worker = _load_callable(worker_path)
            parameter_count = len(inspect.signature(worker).parameters)
            raw_payload = (
                worker(task, lane_root, progress_path)
                if parameter_count >= 3
                else worker(task, lane_root)
            )
            payload = dict(raw_payload)
            required = {
                "lane_id": task["lane_id"],
                "root_id": task["root_id"],
                "algorithm": task.get("algorithm"),
                "seed": task["seed"],
                "solver_continuation_semantic_sha256": task[
                    "solver_continuation_semantic_sha256"
                ],
                "task_semantic_sha256": task["task_semantic_sha256"],
            }
            for key, value in required.items():
                if payload.get(key) != value:
                    raise RuntimeError(f"lane worker binding mismatch: {key}")
            payload["status"] = "COMPLETE"
            streamed_mappings = {}
            streamed_sequences = {}
            policy = payload.get("policy")
            if isinstance(policy, Mapping):
                policy_keys = (
                    iter(policy)
                    if getattr(policy, "canonical_key_order", False) is True
                    else iter(sorted(policy))
                )
                streamed_mappings["policy"] = (
                    (str(key), policy[key]) for key in policy_keys
                )
            for key, value in payload.items():
                if isinstance(value, (list, tuple)):
                    streamed_sequences[str(key)] = iter(value)
            small_document = {
                key: value
                for key, value in payload.items()
                if key not in streamed_mappings and key not in streamed_sequences
            }
            write_receipt = write_self_hashed_json_stream(
                output_path,
                document=small_document,
                streamed_mappings=streamed_mappings,
                streamed_sequences=streamed_sequences,
                binding_fields=(
                    "lane_id",
                    "root_id",
                    "algorithm",
                    "seed",
                    "solver_continuation_semantic_sha256",
                    "task_semantic_sha256",
                    "status",
                ),
                spool_root=Path(str(task["local_completion_spool_root"])),
            )
            adjudication = payload.get("adjudication", {})
            failed_checks = sorted(
                name
                for name, passed in adjudication.get("checks", {}).items()
                if passed is not True
            )
            qualified = payload.get("qualified")
            sidecar = with_self_hash(
                {
                    "schema_version": "stage21_lane_completion_receipt_v2",
                    "stage": task["stage"],
                    "token": task["token"],
                    "lane_id": task["lane_id"],
                    "root_id": task["root_id"],
                    "algorithm": task.get("algorithm"),
                    "seed": task["seed"],
                    "solver_continuation_semantic_sha256": task[
                        "solver_continuation_semantic_sha256"
                    ],
                    "task_semantic_sha256": task["task_semantic_sha256"],
                    "status": "COMPLETE",
                    "qualified": qualified if isinstance(qualified, bool) else None,
                    "qualification_failed_checks": failed_checks,
                    "requested_target_tier": payload.get(
                        "requested_target_tier"
                    ),
                    "effective_target_tier": payload.get("target_tier"),
                    "minimum_qualification_tier": payload.get(
                        "minimum_qualification_tier"
                    ),
                    "requested_target_iterations": payload.get(
                        "requested_target_iterations"
                    ),
                    "effective_target_iterations": payload.get(
                        "target_iterations"
                    ),
                    "stopped_early_after_qualification": payload.get(
                        "stopped_early_after_qualification"
                    ),
                    "artifact_hash_valid": payload.get("artifact_hash_valid") is True,
                    "independent_confirmation_panel": payload.get(
                        "independent_confirmation_panel"
                    )
                    is True,
                    "lane_file_size": write_receipt["size"],
                    "lane_file_sha256": write_receipt["sha256"],
                    "lane_file_self_sha256": write_receipt["self_sha256"],
                    "lane_file_self_hash_verified_at_install": (
                        write_receipt["independent_stream_self_hash_verified"] is True
                    ),
                    "lane_file_identity_verified_at_install": (
                        write_receipt["independent_stream_file_identity_verified"] is True
                    ),
                    "complete_policy_reparsed_for_handle": False,
                    "top_level_sequences_streamed_for_publication": sorted(
                        streamed_sequences
                    ),
                }
            )
            atomic_write(sidecar_path, canonical_json_bytes(sidecar))
            readback_receipt = _validated_completed_lane_receipt(output_path, task)
            if readback_receipt is None:
                raise RuntimeError("lane completion failed independent readback")
            if (
                readback_receipt["sha256"] != write_receipt["sha256"]
                or readback_receipt["size"] != write_receipt["size"]
            ):
                raise RuntimeError("lane completion write/readback receipt mismatch")
        sidecar = readback_receipt["sidecar"]
        failed_checks = list(sidecar.get("qualification_failed_checks", ()))
        qualified = sidecar.get("qualified")
        _write_final_progress(
            progress_path,
            task,
            qualified=qualified if isinstance(qualified, bool) else None,
            failed_checks=failed_checks,
            current_iteration=(
                int(sidecar["effective_target_iterations"])
                if sidecar.get("effective_target_iterations") is not None
                else None
            ),
            requested_target_iterations=(
                int(sidecar["requested_target_iterations"])
                if sidecar.get("requested_target_iterations") is not None
                else None
            ),
            requested_target_tier=(
                int(sidecar["requested_target_tier"])
                if sidecar.get("requested_target_tier") is not None
                else None
            ),
            effective_target_tier=(
                int(sidecar["effective_target_tier"])
                if sidecar.get("effective_target_tier") is not None
                else None
            ),
            minimum_qualification_tier=(
                int(sidecar["minimum_qualification_tier"])
                if sidecar.get("minimum_qualification_tier") is not None
                else None
            ),
            stopped_early_after_qualification=(
                sidecar.get("stopped_early_after_qualification")
                if isinstance(
                    sidecar.get("stopped_early_after_qualification"), bool
                )
                else None
            ),
        )
        handle = {
            "lane_id": task["lane_id"],
            "root_id": task["root_id"],
            "algorithm": task.get("algorithm"),
            "seed": task["seed"],
            "path": str(output_path),
            "sha256": readback_receipt["sha256"],
            "size": readback_receipt["size"],
            "reused": reused,
            "qualified": qualified if isinstance(qualified, bool) else None,
            "qualification_failed_checks": failed_checks,
            "requested_target_tier": sidecar.get("requested_target_tier"),
            "effective_target_tier": sidecar.get("effective_target_tier"),
            "minimum_qualification_tier": sidecar.get(
                "minimum_qualification_tier"
            ),
            "requested_target_iterations": sidecar.get(
                "requested_target_iterations"
            ),
            "effective_target_iterations": sidecar.get(
                "effective_target_iterations"
            ),
            "stopped_early_after_qualification": sidecar.get(
                "stopped_early_after_qualification"
            ),
            "artifact_hash_valid": sidecar.get("artifact_hash_valid") is True,
            "independent_confirmation_panel": sidecar.get(
                "independent_confirmation_panel"
            )
            is True,
            "durable_completion_readback_verified": True,
            "completion_receipt_self_sha256": sidecar["self_sha256"],
            "lane_document_self_sha256": sidecar["lane_file_self_sha256"],
            "coordinator_retains_lane_payload": False,
        }
    except BaseException as error:
        previous_progress = _read_progress(progress_path) or {}
        atomic_write(
            progress_path,
            canonical_json_bytes(
                {
                    **lane_progress_task_defaults(task),
                    **previous_progress,
                    "schema_version": "stage21_lane_progress_v2",
                    "lane_id": task["lane_id"],
                    "root_id": task["root_id"],
                    "algorithm": task.get("algorithm"),
                    "seed": task["seed"],
                    "solver_continuation_semantic_sha256": task.get(
                        "solver_continuation_semantic_sha256"
                    ),
                    "task_semantic_sha256": task.get("task_semantic_sha256"),
                    "event": "lane_failed",
                    "status": "FAILED",
                    "lane_phase": previous_progress.get(
                        "lane_phase", "worker_exception"
                    ),
                    "error_type": type(error).__name__,
                    "error_message": _bounded_failure_message(error),
                    "progress_age_seconds": 0.0,
                    "stalled": False,
                    "updated_unix_seconds": time.time(),
                }
            ),
        )
        raise
    finally:
        cleaned_policy_objects: set[int] = set()
        for candidate in (payload, raw_payload):
            if not isinstance(candidate, Mapping):
                continue
            policy_value = candidate.get("policy")
            cleanup = getattr(policy_value, "cleanup", None)
            if callable(cleanup) and id(policy_value) not in cleaned_policy_objects:
                cleanup()
                cleaned_policy_objects.add(id(policy_value))
        payload = None
        readback_receipt = None
        worker = None
        raw_payload = None
        write_receipt = None
        adjudication = None
        required = None
        policy = None
        small_document = None
        sidecar = None
        release = _release_process_memory()
    if handle is None:
        raise RuntimeError("lane did not produce a compact completion handle")
    handle["worker_memory_release"] = release
    return handle


def run_lane_queue(
    *,
    worker_path: str,
    tasks: Sequence[Mapping[str, Any]],
    lane_root: Path,
    requested_workers: Any = REQUESTED_WORKERS_DEFAULT,
    heartbeat_seconds: int = 60,
    stall_warning_seconds: int = STALL_WARNING_SECONDS,
    estimated_required_disk_bytes: int = 2 * 1024 * 1024 * 1024,
    local_scratch_root: Path | None = None,
) -> dict[str, Any]:
    if len(tasks) != WORK_LANES_PER_PRODUCER:
        raise ValueError("producer queue must contain exactly 48 lanes")
    lane_ids = [str(task["lane_id"]) for task in tasks]
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("duplicate lane IDs")
    bound_tasks = []
    for task in tasks:
        row = dict(task)
        semantic = row.pop("task_semantic_sha256", None)
        continuation_semantic = row.pop(
            "solver_continuation_semantic_sha256", None
        )
        computed_continuation = sha256_json(
            _solver_continuation_semantic_payload(row)
        )
        if (
            continuation_semantic is not None
            and continuation_semantic != computed_continuation
        ):
            raise ValueError(
                f"solver continuation semantic hash mismatch: {task['lane_id']}"
            )
        computed = sha256_json(_task_semantic_payload(row))
        if semantic is not None and semantic != computed:
            raise ValueError(f"task semantic hash mismatch: {task['lane_id']}")
        row["solver_continuation_semantic_sha256"] = computed_continuation
        row["task_semantic_sha256"] = computed
        bound_tasks.append(row)
    task_by_lane = {str(task["lane_id"]): task for task in bound_tasks}
    # Validate the operator input before inspecting or launching any lane.
    parse_requested_workers(requested_workers)
    lane_root.mkdir(parents=True, exist_ok=True)
    completed_handles: list[dict[str, Any]] = []
    unfinished: deque[dict[str, Any]] = deque()
    for task in bound_tasks:
        path = _lane_result_path(lane_root, str(task["lane_id"]))
        existing_receipt = _validated_completed_lane_receipt(path, task)
        if existing_receipt is None:
            unfinished.append(task)
        else:
            existing = existing_receipt["sidecar"]
            completed_handles.append(
                {
                    "lane_id": task["lane_id"],
                    "root_id": task["root_id"],
                    "algorithm": task.get("algorithm"),
                    "seed": task["seed"],
                    "path": str(path),
                    "sha256": existing_receipt["sha256"],
                    "size": existing_receipt["size"],
                    "reused": True,
                    "qualified": existing.get("qualified")
                    if isinstance(existing.get("qualified"), bool)
                    else None,
                    "qualification_failed_checks": list(
                        existing.get("qualification_failed_checks", ())
                    ),
                    "requested_target_tier": existing.get(
                        "requested_target_tier"
                    ),
                    "effective_target_tier": existing.get(
                        "effective_target_tier"
                    ),
                    "minimum_qualification_tier": existing.get(
                        "minimum_qualification_tier"
                    ),
                    "requested_target_iterations": existing.get(
                        "requested_target_iterations"
                    ),
                    "effective_target_iterations": existing.get(
                        "effective_target_iterations"
                    ),
                    "stopped_early_after_qualification": existing.get(
                        "stopped_early_after_qualification"
                    ),
                    "artifact_hash_valid": existing.get("artifact_hash_valid") is True,
                    "independent_confirmation_panel": existing.get(
                        "independent_confirmation_panel"
                    )
                    is True,
                    "durable_completion_readback_verified": True,
                    "completion_receipt_self_sha256": existing["self_sha256"],
                    "lane_document_self_sha256": existing[
                        "lane_file_self_sha256"
                    ],
                    "coordinator_retains_lane_payload": False,
                }
            )
    schedule_rows = [
        _lane_schedule_row(task, lane_root) for task in unfinished
    ]
    schedule_by_lane = {row["lane_id"]: row for row in schedule_rows}
    unfinished = deque(
        sorted(
            unfinished,
            key=lambda task: (
                -int(
                    schedule_by_lane[str(task["lane_id"])][
                        "resumed_checkpoint_present"
                    ]
                ),
                -int(
                    schedule_by_lane[str(task["lane_id"])][
                        "estimated_remaining_cost_units"
                    ]
                ),
                str(task["lane_id"]),
            ),
        )
    )
    scheduled_lane_order = [str(task["lane_id"]) for task in unfinished]
    reused_completion_release = _release_process_memory()
    memory_at_resolution = _memory_snapshot()
    worker_resolution = resolve_worker_count(
        requested_workers,
        unfinished_lane_count=len(unfinished),
        memory_snapshot=memory_at_resolution,
    )
    live_workers = int(worker_resolution["resolved_live_worker_target"])
    remaining_disk_estimate = math.ceil(
        estimated_required_disk_bytes * len(unfinished) / max(1, len(bound_tasks))
    )
    preflight = resource_preflight(
        work_root=lane_root,
        requested_workers=live_workers,
        estimated_required_disk_bytes=remaining_disk_estimate,
        worker_resolution=worker_resolution,
        memory_snapshot=memory_at_resolution,
    )
    admission_preflights = [preflight]
    initial_completed_count = len(completed_handles)
    heartbeat_interval = min(60.0, max(1.0, float(heartbeat_seconds)))
    stall_threshold = max(heartbeat_interval, float(stall_warning_seconds))
    queue_started = time.monotonic()
    parent_release: dict[str, Any] | None = None
    persistent_aggregate_workers = all(
        task.get("role")
        == "strict_result_member_and_crossplay_validation_shard"
        for task in bound_tasks
    )

    def qualification_summary() -> dict[str, Any]:
        lane_qualified = sum(row.get("qualified") is True for row in completed_handles)
        lane_not_qualified = sum(
            row.get("qualified") is False for row in completed_handles
        )
        root_rows: dict[str, list[dict[str, Any]]] = {}
        for row in completed_handles:
            if row.get("algorithm") not in ALGORITHMS:
                continue
            root_rows.setdefault(str(row["root_id"]), []).append(row)
        dispositions = [
            qualify_root(root_id, rows)
            for root_id, rows in sorted(root_rows.items())
            if len(rows) == 6
        ]
        return {
            "qualified_completed_lanes": lane_qualified,
            "not_qualified_completed_lanes": lane_not_qualified,
            "completed_lanes_without_qualification_semantics": (
                len(completed_handles) - lane_qualified - lane_not_qualified
            ),
            "fully_adjudicated_roots": len(dispositions),
            "qualified_roots": sum(row["accepted"] is True for row in dispositions),
            "not_qualified_roots": sum(
                row["accepted"] is not True for row in dispositions
            ),
        }

    telemetry_parent = (
        Path(tempfile.gettempdir()).resolve()
        if local_scratch_root is None
        else Path(local_scratch_root).resolve()
    )
    telemetry_parent.mkdir(parents=True, exist_ok=True)
    telemetry_root = Path(
        tempfile.mkdtemp(prefix="stage21_lane_progress_", dir=telemetry_parent)
    ).resolve()
    if telemetry_root.parent != telemetry_parent or not telemetry_root.name.startswith(
        "stage21_lane_progress_"
    ):
        raise RuntimeError("unsafe telemetry scratch root")
    reported_completed_lane_ids: set[str] = set()

    def emit_queue_progress(
        future_map: Mapping[concurrent.futures.Future[dict[str, Any]], Mapping[str, Any]],
        admitted_at: Mapping[str, float],
        *,
        event: str,
        failed_lanes: int = 0,
        failed_lane_progress: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        wall_now = time.time()
        progress_rows = []
        active_lane_progress = []
        stalled = []
        for task in future_map.values():
            lane_id = str(task["lane_id"])
            row = _read_progress(_progress_path(telemetry_root, lane_id)) or {}
            updated = float(row.get("updated_unix_seconds", admitted_at.get(lane_id, wall_now)))
            progress_age = max(0.0, wall_now - updated)
            is_stalled = progress_age >= stall_threshold
            if is_stalled:
                stalled.append(lane_id)
            if row:
                progress_rows.append(row)
            lane_detail = _lane_progress_detail_record(
                task,
                row,
                overrides={
                    "progress_age_seconds": progress_age,
                    "stalled": is_stalled,
                },
            )
            active_lane_progress.append(lane_detail)
        current_iterations = [
            int(row["current_iteration"])
            for row in progress_rows
            if row.get("current_iteration") is not None
        ]
        checkpoint_iterations = [
            int(row["last_checkpoint_iteration"])
            for row in progress_rows
            if row.get("last_checkpoint_iteration") is not None
        ]
        active_fraction = math.fsum(
            max(0.0, min(1.0, float(row.get("percent_complete", 0.0)) / 100.0))
            for row in progress_rows
        )
        total = len(bound_tasks)
        aggregate_percent = 100.0 * (
            len(completed_handles) + active_fraction
        ) / max(1, total)
        elapsed = max(1e-9, time.monotonic() - queue_started)
        completed_since_launch = max(
            0, len(completed_handles) - initial_completed_count
        )
        completion_rate = completed_since_launch / elapsed
        remaining = total - len(completed_handles)
        summary = qualification_summary()
        memory = _memory_snapshot()
        summary_record = {
            "record_type": "queue_summary",
            "phase": "lane_queue",
            "event": event,
            "active_lanes": len(future_map),
            "queued_lanes": len(unfinished),
            "completed_lanes": len(completed_handles),
            "remaining_lanes": remaining,
            "failed_lanes": failed_lanes,
            **summary,
            "current_iteration_min": min(current_iterations)
            if current_iterations
            else None,
            "current_iteration_max": max(current_iterations)
            if current_iterations
            else None,
            "last_checkpoint_iteration_min": min(checkpoint_iterations)
            if checkpoint_iterations
            else None,
            "last_checkpoint_iteration_max": max(checkpoint_iterations)
            if checkpoint_iterations
            else None,
            "aggregate_percent_complete": aggregate_percent,
            "queue_eta_seconds": remaining / completion_rate
            if completion_rate > 0
            else None,
            "stalled_lane_count": len(stalled),
            "stalled_lane_ids": sorted(stalled),
            "stall_warning_seconds": stall_threshold,
            "per_lane_progress_log_interval_seconds": heartbeat_interval,
            "queued_lane_ids": sorted(
                str(task["lane_id"]) for task in unfinished
            ),
            "parent_rss_bytes": _current_rss_bytes(),
            "available_system_ram_bytes": memory["available_bytes"],
            "parent_memory_release": parent_release,
            "lane_detail_format": "one_json_record_per_following_line",
        }
        print(json.dumps(summary_record, sort_keys=True), flush=True)
        for row in sorted(active_lane_progress, key=lambda item: item["lane_id"]):
            print(
                json.dumps(
                    {
                        "record_type": "active_lane_progress",
                        "phase": "lane_queue",
                        **row,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for row in sorted(
            completed_handles, key=lambda item: str(item["lane_id"])
        ):
            completed_lane_id = str(row["lane_id"])
            if completed_lane_id in reported_completed_lane_ids:
                continue
            completed_task = task_by_lane[completed_lane_id]
            final_progress = _read_progress(
                _progress_path(telemetry_root, completed_lane_id)
            ) or {}
            completion_status = (
                "QUALIFIED_COMPLETE"
                if row.get("qualified") is True
                else "NOT_QUALIFIED_COMPLETE"
                if row.get("qualified") is False
                else "COMPLETE"
            )
            completion_detail = _lane_progress_detail_record(
                completed_task,
                row,
                overrides={
                    **final_progress,
                    "status": completion_status,
                    "lane_phase": final_progress.get(
                        "lane_phase", "durable_completion"
                    ),
                    "event": final_progress.get(
                        "event", "durable_lane_completion_readback_verified"
                    ),
                    "percent_complete": 100.0,
                    "eta_seconds": 0.0,
                    "qualified": row.get("qualified"),
                    "failed_checks": list(
                        row.get("qualification_failed_checks", ())
                    ),
                    "qualification_failed_checks": list(
                        row.get("qualification_failed_checks", ())
                    ),
                    "progress_age_seconds": 0.0,
                    "stalled": False,
                },
            )
            print(
                json.dumps(
                    {
                        "record_type": "completed_lane_progress",
                        "phase": "lane_queue",
                        **completion_detail,
                        "reused_current_completion": row.get("reused") is True,
                        "durable_completion_readback_verified": row.get(
                            "durable_completion_readback_verified"
                        )
                        is True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            reported_completed_lane_ids.add(completed_lane_id)
        for row in failed_lane_progress:
            print(
                json.dumps(
                    {
                        "record_type": "failed_lane_progress",
                        "phase": "lane_queue",
                        **dict(row),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if unfinished:
        context = multiprocessing.get_context("spawn")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=live_workers,
                mp_context=context,
                # Solver lanes deliberately retain the one-task lifecycle so
                # a large node table can never bleed into the next lane.
                # Aggregate shards have already staged immutable inputs and
                # aggressively release payload/cache state, so reusing those
                # processes avoids dozens of redundant spawn/import tails.
                max_tasks_per_child=(None if persistent_aggregate_workers else 1),
            ) as executor:
                future_map: dict[
                    concurrent.futures.Future[dict[str, Any]], dict[str, Any]
                ] = {}
                admitted_at: dict[str, float] = {}

                def fill_workers(admission_limit: int | None = None) -> None:
                    admitted = 0
                    while (
                        unfinished
                        and len(future_map) < live_workers
                        and (admission_limit is None or admitted < admission_limit)
                    ):
                        task = unfinished.popleft()
                        lane_id = str(task["lane_id"])
                        progress = _progress_path(telemetry_root, lane_id)
                        future = executor.submit(
                            _invoke_one_lane,
                            worker_path,
                            task,
                            str(lane_root),
                            str(progress),
                        )
                        future_map[future] = task
                        admitted_at[lane_id] = time.time()
                        admitted += 1

                fill_workers()
                last_heartbeat = time.monotonic()
                emit_queue_progress(
                    future_map, admitted_at, event="queue_started_or_resumed"
                )
                while future_map:
                    done, _ = concurrent.futures.wait(
                        tuple(future_map),
                        timeout=min(5.0, heartbeat_interval),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    failed = 0
                    for future in done:
                        task = future_map.pop(future)
                        admitted_at.pop(str(task["lane_id"]), None)
                        try:
                            completed_handles.append(future.result())
                        except BaseException as error:
                            failed += 1
                            failed_progress = _failed_lane_progress_record(
                                task,
                                _read_progress(
                                    _progress_path(
                                        telemetry_root, str(task["lane_id"])
                                    )
                                ),
                                error,
                            )
                            for pending_future in future_map:
                                pending_future.cancel()
                            try:
                                emit_queue_progress(
                                    future_map,
                                    admitted_at,
                                    event="lane_failed_queue_stopped",
                                    failed_lanes=failed,
                                    failed_lane_progress=(failed_progress,),
                                )
                            except BaseException as telemetry_error:
                                # Failure logging must never replace the worker's
                                # original exception.  This bounded fallback has
                                # no dependency on the transient telemetry tree.
                                try:
                                    print(
                                        json.dumps(
                                            {
                                                "phase": "lane_queue",
                                                "event": (
                                                    "lane_failed_queue_progress_"
                                                    "emission_failed"
                                                ),
                                                "failed_lanes": failed,
                                                "failed_lane_progress": [
                                                    failed_progress
                                                ],
                                                "telemetry_error_type": type(
                                                    telemetry_error
                                                ).__name__,
                                                "telemetry_error_message": (
                                                    _bounded_failure_message(
                                                        telemetry_error
                                                    )
                                                ),
                                            },
                                            sort_keys=True,
                                        ),
                                        flush=True,
                                    )
                                except BaseException:
                                    pass
                            raise
                        finally:
                            task = None
                    if done:
                        parent_release = _release_process_memory()
                        replacement_count = min(
                            len(unfinished), max(0, live_workers - len(future_map))
                        )
                        if replacement_count:
                            boundary_memory = _memory_snapshot()
                            replacement_resolution = resolve_worker_count(
                                0
                                if parse_requested_workers(requested_workers) == 0
                                else replacement_count,
                                unfinished_lane_count=len(unfinished),
                                memory_snapshot=boundary_memory,
                            )
                            admitted_replacements = min(
                                replacement_count,
                                int(
                                    replacement_resolution[
                                        "resolved_live_worker_target"
                                    ]
                                ),
                            )
                            remaining_disk_estimate = math.ceil(
                                estimated_required_disk_bytes
                                * (len(unfinished) + len(future_map))
                                / max(1, len(bound_tasks))
                            )
                            boundary_preflight = (
                                resource_preflight(
                                    work_root=lane_root,
                                    requested_workers=admitted_replacements,
                                    estimated_required_disk_bytes=(
                                        remaining_disk_estimate
                                    ),
                                    worker_resolution=replacement_resolution,
                                    memory_snapshot=boundary_memory,
                                )
                                if admitted_replacements
                                else with_self_hash(
                                    {
                                        "schema_version": (
                                            "stage21_zero_replacement_deferred_v1"
                                        ),
                                        "planned_live_workers": 0,
                                        "memory": boundary_memory,
                                        "worker_resolution": (
                                            replacement_resolution
                                        ),
                                        "status": "NO_REPLACEMENT_CAPACITY",
                                    }
                                )
                            )
                            boundary_preflight = with_self_hash(
                                {
                                    **{
                                        key: value
                                        for key, value in boundary_preflight.items()
                                        if key != "self_sha256"
                                    },
                                    "schema_version": (
                                        "stage21_replacement_admission_preflight_v2"
                                    ),
                                    "active_workers_before_replacement": len(
                                        future_map
                                    ),
                                    "replacement_workers_requested": (
                                        replacement_count
                                    ),
                                    "replacement_workers_admitted": (
                                        admitted_replacements
                                    ),
                                    "queued_lanes_before_replacement": len(
                                        unfinished
                                    ),
                                    "parent_rss_bytes": _current_rss_bytes(),
                                    "completion_boundary_remeasured": True,
                                }
                            )
                            admission_preflights.append(boundary_preflight)
                            fill_workers(admitted_replacements)
                            if (
                                admitted_replacements == 0
                                and not future_map
                                and unfinished
                            ):
                                raise RuntimeError(
                                    "no replacement worker fits the CPU/lane capacity"
                                )
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_interval:
                        emit_queue_progress(
                            future_map,
                            admitted_at,
                            event="coalesced_progress_heartbeat",
                        )
                        last_heartbeat = now
                emit_queue_progress(
                    future_map,
                    admitted_at,
                    event="all_lane_completions_durable",
                )
        finally:
            if telemetry_root.parent != telemetry_parent or not telemetry_root.name.startswith(
                "stage21_lane_progress_"
            ):
                raise RuntimeError("unsafe telemetry cleanup target")
            shutil.rmtree(telemetry_root)
    else:
        emit_queue_progress(
            {}, {}, event="queue_already_complete_from_durable_readback"
        )
        if telemetry_root.parent != telemetry_parent or not telemetry_root.name.startswith(
            "stage21_lane_progress_"
        ):
            raise RuntimeError("unsafe telemetry cleanup target")
        shutil.rmtree(telemetry_root)
    completed_handles.sort(key=lambda row: row["lane_id"])
    if len(completed_handles) != len(bound_tasks):
        raise RuntimeError("lane queue did not complete every task")
    return with_self_hash(
        {
            "schema_version": "stage21_lane_queue_receipt_v5",
            "logical_lane_count": len(bound_tasks),
            "requested_workers": parse_requested_workers(requested_workers),
            "requested_workers_mode": worker_resolution[
                "requested_workers_mode"
            ],
            "detected_logical_cpu_count": worker_resolution[
                "detected_logical_cpu_count"
            ],
            "runtime_worker_limit": worker_resolution["runtime_worker_limit"],
            "worker_resolution_rule": worker_resolution["resolution_rule"],
            "worker_resolution": worker_resolution,
            "live_worker_target": live_workers,
            "initial_completed_lane_count": initial_completed_count,
            "unfinished_lane_count_at_launch": (
                len(bound_tasks) - initial_completed_count
            ),
            "one_task_per_process_lifetime": not persistent_aggregate_workers,
            "solver_lane_one_task_process_lifetime_retained": (
                not persistent_aggregate_workers
            ),
            "persistent_aggregate_worker_pool": persistent_aggregate_workers,
            "aggregate_process_startup_count_upper_bound": (
                live_workers if persistent_aggregate_workers else None
            ),
            "coordinator_consumes_worker_slot": False,
            "worker_count_clipped_by_cpu": worker_resolution[
                "worker_count_clipped_by_cpu"
            ],
            "worker_count_clipped_by_logical_lane_limit": worker_resolution[
                "worker_count_clipped_by_logical_lane_limit"
            ],
            "worker_count_clipped_by_unfinished_lane_count": worker_resolution[
                "worker_count_clipped_by_unfinished_lane_count"
            ],
            "cost_aware_schedule": {
                "schema_version": "stage21_cost_aware_lane_schedule_v1",
                "resume_checkpoint_lanes_admitted_first": True,
                "longest_estimated_remaining_cost_first_within_resume_class": True,
                "lane_order": scheduled_lane_order,
                "rows": sorted(schedule_rows, key=lambda row: row["lane_id"]),
            },
            "lane_requests_materialized_lazily": True,
            "all_solver_lane_states_loaded_in_parent": False,
            "completed_lane_payloads_released_after_durable_readback": True,
            "cross_revision_checkpoint_or_lane_reuse_enabled": False,
            "completed_workers_reaped_after_one_task": (
                not persistent_aggregate_workers
            ),
            "resume_resource_preflight_scales_to_unfinished_lanes": True,
            "replacement_admission_rechecks_disk_and_records_ram": True,
            "replacement_admission_uses_ram_to_limit_workers": False,
            "replacement_admission_preflight_count": max(
                0, len(admission_preflights) - 1
            ),
            "admission_preflights": admission_preflights,
            "reused_completed_payloads_released_after_scan": (
                reused_completion_release
            ),
            "parent_memory_release": parent_release,
            "progress_heartbeat_seconds": heartbeat_interval,
            "stall_warning_seconds": stall_threshold,
            "progress_log_fields": [
                "active_lanes",
                "queued_lanes",
                "completed_lanes",
                "remaining_lanes",
                "qualified_completed_lanes",
                "not_qualified_completed_lanes",
                "current_iteration_min",
                "current_iteration_max",
                "last_checkpoint_iteration_min",
                "last_checkpoint_iteration_max",
                "aggregate_percent_complete",
                "queue_eta_seconds",
                "stalled_lane_count",
            ],
            "qualification_summary": qualification_summary(),
            "preflight": preflight,
            "handles": completed_handles,
        }
    )
