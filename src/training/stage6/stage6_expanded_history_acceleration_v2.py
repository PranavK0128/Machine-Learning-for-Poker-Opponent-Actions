"""Additive R2 acceleration for the frozen expanded-layer history features.

R1 remains the scientific oracle.  This module changes only the execution
plan: completed-hand entity/dyad summaries are built once across all logical
lanes, rolling totals use groupwise cumulative-window differences, and only
those globally aggregated summaries may be partitioned across host workers.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from stage6_expanded_layers import (
    ACTIONS,
    ACTION_TO_INDEX,
    FEATURE_SOURCE_COLUMNS,
    _action_indicators,
    _aggregate_columns,
    _parse_exact_counterpart,
    _validate_source_frame,
    assert_no_prohibited_model_features,
    build_dyad_history_features,
    build_line_prefix_features,
    canonical_json_bytes,
    dyad_interaction_feature_columns,
    dyad_main_effect_feature_columns,
    dyad_numeric_feature_columns,
    entity_state_value_columns,
    seal_document,
)


HISTORY_WINDOWS_V2 = (16, 64)
HISTORY_PARTITION_COUNT_V2 = 32
HEARTBEAT_INTERVAL_SECONDS_V2 = 60.0
MINIMUM_WORKER_ESTIMATE_BYTES_V2 = 512 * 1024 * 1024
RAM_RESERVE_BYTES_V2 = 4 * 1024 * 1024 * 1024
SUPPORTED_ACCELERATORS_V2 = ("CPU", "GPU", "TPU")
PREFERRED_COLAB_HOST_V2 = "CPU High-RAM"


def _available_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception as exc:  # pragma: no cover - Colab/runtime defensive path
        raise RuntimeError("available RAM is required for worker admission") from exc


def _logical_cpu_count() -> int:
    detected = os.cpu_count()
    if detected is None or detected < 1:
        raise RuntimeError("logical CPU detection failed")
    return int(detected)


def resolve_history_workers_v2(
    requested_workers: str | int,
    *,
    unfinished_partitions: int,
    stage: str,
    estimated_worker_bytes: int,
    available_ram_bytes: int | None = None,
    logical_cpus: int | None = None,
    reserve_ram_bytes: int = RAM_RESERVE_BYTES_V2,
) -> dict[str, Any]:
    """Admit host workers using a nonzero, stage-specific RAM estimate."""
    if not stage or not str(stage).strip():
        raise ValueError("worker admission requires a stage")
    if int(unfinished_partitions) < 1:
        raise ValueError("worker admission requires unfinished partitions")
    if int(estimated_worker_bytes) < MINIMUM_WORKER_ESTIMATE_BYTES_V2:
        raise ValueError("estimated_worker_bytes is below the frozen safe floor")
    if int(reserve_ram_bytes) < 0:
        raise ValueError("reserve_ram_bytes must not be negative")

    cpus = _logical_cpu_count() if logical_cpus is None else int(logical_cpus)
    ram = _available_ram_bytes() if available_ram_bytes is None else int(available_ram_bytes)
    if cpus < 1 or ram < 1:
        raise RuntimeError("positive logical CPU and available RAM values are required")
    # Preserve the configured reserve on roomy Colab hosts, while allowing a
    # bounded one-worker path on smaller local audit machines.
    effective_reserve = min(int(reserve_ram_bytes), ram // 4)
    ram_slots = max(0, (ram - effective_reserve) // int(estimated_worker_bytes))
    safe_maximum = min(cpus, int(unfinished_partitions), int(ram_slots))
    if safe_maximum < 1:
        raise MemoryError("no history worker fits the explicit RAM admission estimate")

    if isinstance(requested_workers, str):
        normalized = requested_workers.strip().lower()
        if normalized == "auto":
            requested: str | int = "auto"
            admitted = safe_maximum
        else:
            try:
                parsed = int(normalized)
            except ValueError as exc:
                raise ValueError("requested_workers must be 'auto' or a positive integer") from exc
            if parsed < 1:
                raise ValueError("requested_workers must be 'auto' or a positive integer")
            requested = parsed
            admitted = min(parsed, safe_maximum)
    elif isinstance(requested_workers, (int, np.integer)):
        if int(requested_workers) < 1:
            raise ValueError("requested_workers must be 'auto' or a positive integer")
        requested = int(requested_workers)
        admitted = min(int(requested_workers), safe_maximum)
    else:
        raise ValueError("requested_workers must be 'auto' or a positive integer")

    return {
        "schema_version": "stage6_expanded_history_worker_admission_v2",
        "stage": str(stage),
        "requested_workers": requested,
        "admitted_workers": int(admitted),
        "safe_maximum_workers": int(safe_maximum),
        "logical_cpus": int(cpus),
        "unfinished_partitions": int(unfinished_partitions),
        "available_ram_bytes": int(ram),
        "configured_reserve_ram_bytes": int(reserve_ram_bytes),
        "effective_reserve_ram_bytes": int(effective_reserve),
        "estimated_worker_bytes": int(estimated_worker_bytes),
        "one_thread_per_worker": True,
        "training_execution": "host_cpu_process_work",
        "accelerator_used_for_tree_or_history_work": False,
    }


def smaller_worker_request_after_confirmed_oom_v2(
    current_workers: int,
    *,
    confirmed_oom: bool,
) -> int:
    """Permit a smaller retry only after the caller confirms a real OOM."""
    if not confirmed_oom:
        raise RuntimeError("worker reduction is allowed only after a confirmed OOM")
    if int(current_workers) < 2:
        raise RuntimeError("a one-worker job cannot be reduced further")
    return max(1, int(current_workers) // 2)


@dataclass
class FeatureHistoryHeartbeatV2:
    """Emit bounded, truthful progress while R2 work is active."""

    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS_V2
    disk_root: Path = field(default_factory=Path.cwd)
    emit: Callable[[Mapping[str, Any]], None] = field(
        default=lambda payload: print(json.dumps(payload, sort_keys=True), flush=True)
    )
    _last_emit: float = field(default_factory=lambda: -float("inf"), init=False)
    _phase_started: float = field(default_factory=time.monotonic, init=False)
    _current: dict[str, Any] = field(default_factory=dict, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def pump() -> None:
            while not self._stop.wait(float(self.interval_seconds)):
                with self._lock:
                    current = dict(self._current)
                if current:
                    self.update(force=True, **current)

        self._thread = threading.Thread(
            target=pump, name="stage6-expanded-history-heartbeat-v2", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(2.0, float(self.interval_seconds)))
        self._thread = None

    def start_phase(self, phase: str) -> None:
        self._phase_started = time.monotonic()
        self._last_emit = -float("inf")
        self.update(phase=phase, completed=0, denominator=1, checkpoint="phase-start", force=True)

    def update(
        self,
        *,
        phase: str,
        completed: int,
        denominator: int,
        checkpoint: str,
        force: bool = False,
    ) -> None:
        with self._lock:
            self._current = {
                "phase": str(phase),
                "completed": int(completed),
                "denominator": int(denominator),
                "checkpoint": str(checkpoint),
            }
        now = time.monotonic()
        if not force and now - self._last_emit < float(self.interval_seconds):
            return
        elapsed = max(0.0, now - self._phase_started)
        throughput = float(completed) / elapsed if elapsed > 0 else 0.0
        remaining = max(0, int(denominator) - int(completed))
        eta = remaining / throughput if throughput > 0 else None
        self.emit({
            "event": "stage6_expanded_history_v2_heartbeat",
            "phase": str(phase),
            "denominator": int(denominator),
            "completed": int(completed),
            "throughput_per_second": throughput,
            "eta_seconds": eta,
            "available_ram_bytes": _available_ram_bytes(),
            "available_disk_bytes": int(shutil.disk_usage(self.disk_root).free),
            "checkpoint": str(checkpoint),
            "elapsed_seconds": elapsed,
        })
        self._last_emit = now


def bottleneck_contract_v2() -> dict[str, Any]:
    """Machine-readable proof of the R1/R2 execution-plan difference."""
    per_state_python_rolling_calls = len(("decision_count", *_aggregate_columns())) * len(
        HISTORY_WINDOWS_V2
    )
    return seal_document({
        "schema_version": "stage6_expanded_history_bottleneck_contract_v2",
        "r1_entity_python_groupby_rolling_lambda_calls": per_state_python_rolling_calls,
        "r1_dyad_python_groupby_rolling_lambda_calls": per_state_python_rolling_calls,
        "r1_python_asof_group_loops": 3,
        "r2_python_groupby_rolling_lambda_calls": 0,
        "r2_global_completed_hand_aggregation_passes": 1,
        "r2_window_method": "vectorized_group_cumulative_window_difference",
        "r2_asof_method": "pandas_merge_asof_strict_backward",
        "raw_lane_partitioning_allowed": False,
        "post_aggregation_entity_dyad_partitioning_allowed": True,
    })


def build_line_prefix_features_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """R1-exact line features without a Python groupby transform lambda."""
    _validate_source_frame(frame)
    ordered = frame.loc[:, FEATURE_SOURCE_COLUMNS].copy()
    ordered["decision_order"] = pd.to_numeric(ordered["decision_order"], errors="raise")
    ordered = ordered.sort_values(
        ["game_id", "hand_id", "decision_order", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    hand_keys = [ordered["game_id"], ordered["hand_id"]]
    hand = ordered.groupby(["game_id", "hand_id"], sort=False, observed=True)
    if hand["decision_order"].diff().fillna(1).le(0).any():
        raise RuntimeError("within-hand decision order is not strictly increasing")

    output = pd.DataFrame({"row_id": ordered["row_id"].astype(str)})
    output["line_prefix_decision_count"] = hand.cumcount().astype(np.int32)
    aggressive = ordered["action_label"].isin(("bet", "raise", "all_in")).astype(np.int32)
    prior_aggressive = aggressive.groupby(hand_keys, sort=False).cumsum() - aggressive
    output["line_prefix_aggressive_count"] = prior_aggressive.astype(np.int32)
    denominator = output["line_prefix_decision_count"].replace(0, np.nan)
    output["line_prefix_aggressive_rate"] = (prior_aggressive / denominator).fillna(0.0).astype(np.float32)

    actor_keys = [ordered["game_id"], ordered["hand_id"], ordered["actor_id"]]
    actor_hand = ordered.groupby(["game_id", "hand_id", "actor_id"], sort=False, observed=True)
    output["line_actor_prior_decision_count"] = actor_hand.cumcount().astype(np.int32)
    actor_prior_aggressive = aggressive.groupby(actor_keys, sort=False).cumsum() - aggressive
    output["line_actor_prior_aggressive_count"] = actor_prior_aggressive.astype(np.int32)
    actor_denominator = output["line_actor_prior_decision_count"].replace(0, np.nan)
    output["line_actor_prior_aggressive_rate"] = (
        actor_prior_aggressive / actor_denominator
    ).fillna(0.0).astype(np.float32)

    street_hand = ordered.groupby(["game_id", "hand_id", "street"], sort=False, observed=True)
    output["line_street_prefix_decision_count"] = street_hand.cumcount().astype(np.int32)
    prior_street = hand["street"].shift(1)
    transition = (
        prior_street.notna() & prior_street.astype(str).ne(ordered["street"].astype(str))
    ).astype(np.int32)
    output["line_street_transition_count"] = transition.groupby(hand_keys, sort=False).cumsum().astype(np.int32)

    for lag in range(1, 7):
        previous_action = hand["action_label"].shift(lag)
        previous_actor = hand["actor_id"].shift(lag)
        previous_street = hand["street"].shift(lag)
        output[f"line_prev_action_{lag}_code"] = previous_action.map(ACTION_TO_INDEX).fillna(-1).astype(np.int8)
        output[f"line_prev_action_{lag}_same_actor"] = (
            previous_actor.notna() & previous_actor.astype(str).eq(ordered["actor_id"].astype(str))
        )
        output[f"line_prev_action_{lag}_same_street"] = (
            previous_street.notna() & previous_street.astype(str).eq(ordered["street"].astype(str))
        )

    previous_action = hand["action_label"].shift(1)
    shifted = previous_action.fillna("__NONE__")
    prior_shifted = shifted.groupby(hand_keys, sort=False).shift(1)
    boundary_flag = shifted.ne(prior_shifted)
    boundary = boundary_flag.astype(np.int64).groupby(hand_keys, sort=False).cumsum()
    output["line_previous_action_run_length"] = shifted.groupby(
        [ordered["game_id"], ordered["hand_id"], boundary], sort=False
    ).cumcount().add(1).where(previous_action.notna(), 0).astype(np.int16)

    output = output.set_index("row_id").loc[frame["row_id"].astype(str)].reset_index()
    assert_no_prohibited_model_features(output.columns)
    return output


def _stable_partition(keys: pd.DataFrame, partition_count: int) -> np.ndarray:
    joined = keys.astype(str).agg("\x1f".join, axis=1)
    return joined.map(
        lambda value: int.from_bytes(
            hashlib.sha256(value.encode("utf-8")).digest()[:8], "big"
        ) % int(partition_count)
    ).to_numpy(dtype=np.int16)


def _global_completed_hand_summaries_v2(
    working: pd.DataFrame,
    eligible: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate all lanes once; partitioning is forbidden before this point."""
    hand_times = working.groupby(["game_id", "hand_id"], sort=False, observed=True)[
        "decision_utc"
    ].agg(hand_start_utc="min", completed_utc="max").reset_index()
    indicated = _action_indicators(working)
    aggregates = _aggregate_columns()
    entity = indicated.groupby(
        ["actor_id", "game_id", "hand_id"], sort=False, observed=True
    ).agg(
        decision_count=("row_id", "size"),
        **{column: (column, "sum") for column in aggregates},
    ).reset_index()
    entity = entity.merge(
        hand_times.loc[:, ["game_id", "hand_id", "completed_utc"]],
        on=["game_id", "hand_id"], how="left", validate="many_to_one",
    )

    dyad_rows = indicated.loc[eligible].copy()
    if len(dyad_rows):
        dyad_rows["counterpart_id"] = working.loc[eligible, "counterpart_id"].astype(str).to_numpy()
        dyad = dyad_rows.groupby(
            ["actor_id", "counterpart_id", "game_id", "hand_id"],
            sort=False, observed=True,
        ).agg(
            decision_count=("row_id", "size"),
            **{column: (column, "sum") for column in aggregates},
        ).reset_index()
        dyad = dyad.merge(
            hand_times.loc[:, ["game_id", "hand_id", "completed_utc"]],
            on=["game_id", "hand_id"], how="left", validate="many_to_one",
        )
    else:
        dyad = pd.DataFrame(columns=[
            "actor_id", "counterpart_id", "game_id", "hand_id",
            "decision_count", *aggregates, "completed_utc",
        ])
    return hand_times, entity, dyad


def _group_cumulative_windows_v2(
    summary: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """One vector group-cumsum plus exact window differences for every value."""
    if summary.empty:
        empty = np.empty((0, len(value_columns)), dtype=np.float64)
        return empty, {window: empty.copy() for window in HISTORY_WINDOWS_V2}, np.empty(0, dtype=np.int32)
    grouped = summary.groupby(list(key_columns), sort=False, observed=True)
    cumulative = grouped[list(value_columns)].cumsum().to_numpy(dtype=np.float64)
    group_number = grouped.ngroup().to_numpy(dtype=np.int64)
    position = grouped.cumcount().to_numpy(dtype=np.int64)
    windows: dict[int, np.ndarray] = {}
    row_index = np.arange(len(summary), dtype=np.int64)
    for window in HISTORY_WINDOWS_V2:
        result = cumulative.copy()
        prior_index = row_index - int(window)
        subtract = (position >= int(window)) & (prior_index >= 0)
        result[subtract] -= cumulative[prior_index[subtract]]
        windows[int(window)] = result
    return cumulative, windows, position.astype(np.int32) + np.int32(1)


def _state_from_summary_partition_v2(
    summary: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    dyad: bool,
) -> pd.DataFrame:
    aggregates = _aggregate_columns()
    value_columns = ["decision_count", *aggregates]
    summary = summary.copy()
    for key in key_columns:
        summary[key] = summary[key].astype(str)
    summary["completed_ns"] = pd.to_datetime(
        summary["completed_utc"], utc=True, errors="raise"
    ).astype("int64")
    summary = summary.sort_values(
        [*key_columns, "completed_ns", "game_id", "hand_id"], kind="mergesort"
    ).reset_index(drop=True)
    cumulative, windows, hand_count = _group_cumulative_windows_v2(
        summary, key_columns, value_columns
    )
    column_index = {name: index for index, name in enumerate(value_columns)}
    data: dict[str, Any] = {
        key: summary[key].to_numpy(copy=False) for key in key_columns
    }
    data["completed_ns"] = summary["completed_ns"].to_numpy(dtype=np.int64)
    count_name = "dyad_prior_hand_count" if dyad else "prior_hand_count"
    data[count_name] = hand_count.astype(np.int32)

    def rate(matrix: np.ndarray, numerator: str, denominator: str) -> np.ndarray:
        numerator_values = matrix[:, column_index[numerator]]
        denominator_values = matrix[:, column_index[denominator]]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = numerator_values / np.where(denominator_values == 0, np.nan, denominator_values)
        return values.astype(np.float32)

    matrices = {"": cumulative, "16_": windows[16], "64_": windows[64]}
    for suffix, matrix in matrices.items():
        prefix = "dyad_" if dyad else ""
        data[f"{prefix}{suffix}aggression_rate"] = rate(matrix, "is_aggressive", "decision_count")
        for action in ACTIONS:
            data[f"{prefix}{suffix}{action}_rate"] = rate(matrix, f"action_{action}", "decision_count")
        for name in ("fold", "call", "aggressive_response"):
            data[f"{prefix}{suffix}facing_aggression_{name}_rate"] = rate(
                matrix, f"faced_{name}", "faced_aggression"
            )
    for street in ("preflop", "flop", "turn", "river"):
        prefix = "dyad_" if dyad else ""
        count_values = cumulative[:, column_index[f"street_{street}_decision"]].astype(np.float32)
        data[f"{prefix}{street}_prior_decision_count"] = count_values
        data[f"{prefix}{street}_aggression_rate"] = rate(
            cumulative, f"street_{street}_aggressive", f"street_{street}_decision"
        )
    state = pd.DataFrame(data)
    values = dyad_interaction_feature_columns() if dyad else entity_state_value_columns()
    return state.loc[:, [*key_columns, "completed_ns", *values]]


def _partitioned_state_v2(
    summary: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    dyad: bool,
    requested_workers: str | int,
    heartbeat: FeatureHistoryHeartbeatV2 | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    value_columns = dyad_interaction_feature_columns() if dyad else entity_state_value_columns()
    if summary.empty:
        return pd.DataFrame(columns=[*key_columns, "completed_ns", *value_columns]), {
            "stage": "dyad_state" if dyad else "entity_state",
            "partition_count": 0,
            "admitted_workers": 0,
        }
    partition_ids = _stable_partition(summary.loc[:, list(key_columns)], HISTORY_PARTITION_COUNT_V2)
    partitions = [
        summary.loc[partition_ids == partition].copy()
        for partition in range(HISTORY_PARTITION_COUNT_V2)
        if np.any(partition_ids == partition)
    ]
    estimated = max(
        MINIMUM_WORKER_ESTIMATE_BYTES_V2,
        int(max(part.memory_usage(index=True, deep=True).sum() for part in partitions) * 12),
    )
    stage = "dyad_state" if dyad else "entity_state"
    admission = resolve_history_workers_v2(
        requested_workers,
        unfinished_partitions=len(partitions),
        stage=stage,
        estimated_worker_bytes=estimated,
    )
    workers = int(admission["admitted_workers"])
    if heartbeat is not None:
        heartbeat.start_phase(stage)
    results: dict[int, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"stage6-{stage}-v2") as pool:
        futures = {
            pool.submit(
                _state_from_summary_partition_v2,
                partition,
                key_columns=key_columns,
                dyad=dyad,
            ): index
            for index, partition in enumerate(partitions)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            results[index] = future.result()
            if heartbeat is not None:
                heartbeat.update(
                    phase=stage,
                    completed=completed,
                    denominator=len(partitions),
                    checkpoint=f"partition-{index:02d}",
                    force=completed == len(partitions),
                )
    state = pd.concat([results[index] for index in sorted(results)], ignore_index=True)
    state = state.sort_values([*key_columns, "completed_ns"], kind="mergesort").reset_index(drop=True)
    admission["partition_count"] = len(partitions)
    admission["global_aggregation_preceded_partitioning"] = True
    admission["raw_lane_partitioning"] = False
    return state, admission


def _merge_asof_state_v2(
    query: pd.DataFrame,
    state: pd.DataFrame,
    *,
    query_keys: Sequence[str],
    state_keys: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=query.index, columns=list(value_columns), dtype=np.float32)
    if query.empty or state.empty:
        return result
    left = query.loc[:, [*query_keys, "hand_start_ns"]].copy()
    left["__position"] = np.arange(len(left), dtype=np.int64)
    rename = {state_key: query_key for query_key, state_key in zip(query_keys, state_keys, strict=True)}
    right = state.rename(columns=rename).loc[:, [*query_keys, "completed_ns", *value_columns]].copy()
    for key in query_keys:
        left[key] = left[key].astype(str)
        right[key] = right[key].astype(str)
    left = left.sort_values(["hand_start_ns", *query_keys, "__position"], kind="mergesort")
    right = right.sort_values(["completed_ns", *query_keys], kind="mergesort")
    matched = pd.merge_asof(
        left,
        right,
        left_on="hand_start_ns",
        right_on="completed_ns",
        by=list(query_keys),
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("__position", kind="mergesort")
    result.loc[:, list(value_columns)] = matched.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    return result


def build_dyad_history_features_v2(
    frame: pd.DataFrame,
    *,
    requested_workers: str | int = "auto",
    accelerator: str = "CPU",
    heartbeat: FeatureHistoryHeartbeatV2 | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build R1-exact history features from one global completed-hand graph."""
    _validate_source_frame(frame)
    accelerator_name = str(accelerator).strip().upper()
    if accelerator_name not in SUPPORTED_ACCELERATORS_V2:
        raise ValueError(f"accelerator must be one of {SUPPORTED_ACCELERATORS_V2}")
    working = frame.loc[:, FEATURE_SOURCE_COLUMNS].copy()
    working["counterpart_id"] = [
        _parse_exact_counterpart(value, count)
        for value, count in zip(
            working["active_opponent_canonical_player_ids_before"],
            working["active_opponent_count_before"],
            strict=True,
        )
    ]
    eligible = working["counterpart_id"].notna() & working["counterpart_id"].astype(str).ne(
        working["actor_id"].astype(str)
    )
    output = pd.DataFrame({
        "row_id": working["row_id"].astype(str),
        "dyad_eligible": eligible.astype(bool),
    })
    if heartbeat is not None:
        heartbeat.start_phase("global_completed_hand_aggregation")
    hand_times, entity_summary, dyad_summary = _global_completed_hand_summaries_v2(working, eligible)
    if heartbeat is not None:
        heartbeat.update(
            phase="global_completed_hand_aggregation", completed=1, denominator=1,
            checkpoint="global-summaries-sealed", force=True,
        )
    if not eligible.any():
        empty = pd.DataFrame(np.nan, index=output.index, columns=dyad_numeric_feature_columns(), dtype=np.float32)
        receipt = seal_document({
            "schema_version": "stage6_expanded_history_acceleration_receipt_v2",
            "row_count": len(frame), "eligible_row_count": 0,
            "global_entity_completed_hand_rows": len(entity_summary),
            "global_dyad_completed_hand_rows": 0,
            "raw_lane_partitioning": False, "fresh_targets_opened": False,
            "accelerator_environment": accelerator_name,
            "history_execution": "host_cpu_process_work",
            "tpu_acceleration_claimed": False,
        })
        return pd.concat([output, empty], axis=1), receipt

    rows = working.loc[eligible].copy().merge(
        hand_times.loc[:, ["game_id", "hand_id", "hand_start_utc"]],
        on=["game_id", "hand_id"], how="left", validate="many_to_one",
    )
    rows["counterpart_id"] = rows["counterpart_id"].astype(str)
    rows["actor_id"] = rows["actor_id"].astype(str)
    rows["hand_start_ns"] = pd.to_datetime(rows["hand_start_utc"], utc=True, errors="raise").astype("int64")

    entity_state, entity_workers = _partitioned_state_v2(
        entity_summary, key_columns=("actor_id",), dyad=False,
        requested_workers=requested_workers, heartbeat=heartbeat,
    )
    entity_values = entity_state_value_columns()
    actor_main = _merge_asof_state_v2(
        rows, entity_state, query_keys=("actor_id",), state_keys=("actor_id",),
        value_columns=entity_values,
    ).rename(columns={name: f"main_actor_{name}" for name in entity_values})
    counterpart_main = _merge_asof_state_v2(
        rows, entity_state, query_keys=("counterpart_id",), state_keys=("actor_id",),
        value_columns=entity_values,
    ).rename(columns={name: f"main_counterpart_{name}" for name in entity_values})

    dyad_state, dyad_workers = _partitioned_state_v2(
        dyad_summary, key_columns=("actor_id", "counterpart_id"), dyad=True,
        requested_workers=requested_workers, heartbeat=heartbeat,
    )
    dyad_values = dyad_interaction_feature_columns()
    interaction = _merge_asof_state_v2(
        rows, dyad_state,
        query_keys=("actor_id", "counterpart_id"),
        state_keys=("actor_id", "counterpart_id"),
        value_columns=dyad_values,
    )
    interaction["dyad_prior_hand_count"] = interaction["dyad_prior_hand_count"].fillna(0).astype(np.float32)
    merged = pd.concat(
        [rows.loc[:, ["row_id"]].reset_index(drop=True), actor_main, counterpart_main, interaction],
        axis=1,
    )
    output = output.merge(merged, on="row_id", how="left", validate="one_to_one", sort=False)
    assert_no_prohibited_model_features(
        [column for column in output.columns if column not in {"row_id", "dyad_eligible"}]
    )
    receipt = seal_document({
        "schema_version": "stage6_expanded_history_acceleration_receipt_v2",
        "row_count": len(frame),
        "eligible_row_count": int(eligible.sum()),
        "global_hand_count": len(hand_times),
        "global_entity_completed_hand_rows": len(entity_summary),
        "global_dyad_completed_hand_rows": len(dyad_summary),
        "entity_worker_admission": entity_workers,
        "dyad_worker_admission": dyad_workers,
        "partitioning_unit": "globally_preaggregated_completed_hand_entity_or_dyad",
        "raw_lane_partitioning": False,
        "strictly_prior_completed_hand_asof": True,
        "accelerator_environment": accelerator_name,
        "preferred_colab_host": PREFERRED_COLAB_HOST_V2,
        "history_execution": "host_cpu_process_work",
        "tpu_acceleration_claimed": False,
        "one_thread_per_worker": True,
        "fresh_targets_opened": False,
    })
    del entity_state, dyad_state, entity_summary, dyad_summary, rows
    gc.collect()
    return output, receipt


def assert_exact_history_frame_equivalence_v2(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
) -> None:
    """Require exact schema, order, dtype, null mask, and finite values."""
    if list(expected.columns) != list(observed.columns):
        raise AssertionError("feature columns or order changed")
    if len(expected) != len(observed):
        raise AssertionError("feature row count changed")
    for column in expected.columns:
        if expected[column].dtype != observed[column].dtype:
            raise AssertionError(
                f"feature dtype changed for {column}: {expected[column].dtype} != {observed[column].dtype}"
            )
        left = expected[column].to_numpy(copy=False)
        right = observed[column].to_numpy(copy=False)
        if left.dtype.kind in {"f", "c"}:
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise AssertionError(f"feature values changed for {column}")


def benchmark_history_builders_v2(
    frame: pd.DataFrame,
    *,
    repeats: int = 2,
    minimum_speedup: float = 1.25,
) -> dict[str, Any]:
    """Bounded old-vs-new benchmark; parity is checked before speed authority."""
    if len(frame) < 1 or len(frame) > 200_000:
        raise ValueError("benchmark frame must contain 1..200000 rows")
    if int(repeats) < 1 or int(repeats) > 5:
        raise ValueError("benchmark repeats must be in 1..5")
    if float(minimum_speedup) < 1.0:
        raise ValueError("minimum_speedup must be at least 1.0")
    expected = build_dyad_history_features(frame)
    observed, _ = build_dyad_history_features_v2(frame, requested_workers=1, accelerator="CPU")
    assert_exact_history_frame_equivalence_v2(expected, observed)

    old_times: list[float] = []
    new_times: list[float] = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        old = build_dyad_history_features(frame)
        old_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        new, _ = build_dyad_history_features_v2(frame, requested_workers=1, accelerator="CPU")
        new_times.append(time.perf_counter() - started)
        assert_exact_history_frame_equivalence_v2(old, new)
        del old, new
        gc.collect()
    old_median = float(np.median(np.asarray(old_times, dtype=np.float64)))
    new_median = float(np.median(np.asarray(new_times, dtype=np.float64)))
    speedup = old_median / new_median if new_median > 0 else float("inf")
    if speedup < float(minimum_speedup):
        raise RuntimeError(
            f"R2 bounded benchmark speedup {speedup:.3f} is below {minimum_speedup:.3f}"
        )
    return seal_document({
        "schema_version": "stage6_expanded_history_benchmark_receipt_v2",
        "row_count": len(frame),
        "repeats": int(repeats),
        "old_seconds": old_times,
        "new_seconds": new_times,
        "old_median_seconds": old_median,
        "new_median_seconds": new_median,
        "speedup": speedup,
        "minimum_speedup": float(minimum_speedup),
        "material_speedup_pass": True,
        "exact_frame_equivalence_pass": True,
        "benchmark_bound_rows": 200_000,
        "fresh_targets_opened": False,
    })


def _synthetic_row_v2(
    row_id: str,
    game_id: str,
    hand_id: str,
    order: int,
    actor: str,
    counterpart: str,
    action: str,
    utc: str,
    *,
    street: str = "preflop",
    facing: bool = False,
    lane: int = 0,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "game_id": game_id,
        "hand_id": hand_id,
        "decision_id": f"decision-{row_id}",
        "decision_utc": utc,
        "decision_order": order,
        "actor_id": actor,
        "street": street,
        "action_label": action,
        "active_opponent_canonical_player_ids_before": json.dumps([counterpart]),
        "active_opponent_count_before": 1,
        "facing_bet_or_raise": facing,
        "source_lane_id": f"lane-{lane:02d}",
    }


def adversarial_equivalence_frame_v2() -> pd.DataFrame:
    """Deterministic >64-hand, cross-lane, overlap and abstention oracle frame."""
    rows: list[dict[str, Any]] = []
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    actions = ("check", "bet", "call", "raise", "fold", "all_in")
    streets = ("preflop", "flop", "turn", "river")
    for hand in range(72):
        utc = (base + timedelta(minutes=hand)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        actor = f"P{hand % 4}"
        counterpart = f"P{(hand + 1) % 4}"
        for order in (1, 2):
            action = actions[(hand + order) % len(actions)]
            rows.append(_synthetic_row_v2(
                f"r-{hand:03d}-{order}", f"g-{hand % 3}", f"h-{hand:03d}", order,
                actor if order == 1 else counterpart,
                counterpart if order == 1 else actor,
                action, utc,
                street=streets[(hand + order) % len(streets)],
                facing=action in {"fold", "call", "raise", "all_in"},
                lane=(hand * 7 + order) % 48,
            ))
    rows.extend([
        _synthetic_row_v2("long-1", "g-long", "h-long", 1, "P0", "P1", "raise", "2026-07-03T00:00:00.000Z", lane=47),
        _synthetic_row_v2("long-2", "g-long", "h-long", 2, "P2", "P3", "call", "2026-07-03T00:10:00.000Z", lane=0),
        _synthetic_row_v2("overlap", "g-other", "h-overlap", 1, "P0", "P1", "check", "2026-07-03T00:05:00.000Z", lane=23),
        _synthetic_row_v2("after", "g-other", "h-after", 1, "P0", "P1", "fold", "2026-07-03T00:11:00.000Z", facing=True, lane=24),
    ])
    multiway = _synthetic_row_v2(
        "multiway", "g-other", "h-multiway", 1, "P0", "P1", "check",
        "2026-07-03T00:12:00.000Z", lane=25,
    )
    multiway["active_opponent_canonical_player_ids_before"] = json.dumps(["P1", "P2"])
    multiway["active_opponent_count_before"] = 2
    rows.append(multiway)
    return pd.DataFrame(rows[::-1])


def benchmark_cardinality_frame_v2(
    *, player_count: int = 80, hands_per_player: int = 20,
) -> pd.DataFrame:
    """Bounded many-group frame that isolates the repeated-groupby bottleneck."""
    rows: list[dict[str, Any]] = []
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    sequence = 0
    for player in range(int(player_count)):
        for hand in range(int(hands_per_player)):
            utc = (base + timedelta(minutes=sequence)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            sequence += 1
            rows.extend([
                _synthetic_row_v2(
                    f"b-{player}-{hand}-1", f"g-{player % 4}", f"h-{player}-{hand}", 1,
                    f"A{player}", f"B{player}", "raise" if hand % 3 == 0 else "check", utc,
                    lane=(player + hand) % 48,
                ),
                _synthetic_row_v2(
                    f"b-{player}-{hand}-2", f"g-{player % 4}", f"h-{player}-{hand}", 2,
                    f"B{player}", f"A{player}", "call", utc, street="flop", facing=True,
                    lane=(player + hand + 1) % 48,
                ),
            ])
    return pd.DataFrame(rows)


def build_preflight_equivalence_binding_v2(
    *, repeats: int = 1, minimum_speedup: float = 1.25,
) -> dict[str, Any]:
    """Freeze exact parity and material speedup before an operator restarts N12."""
    adversarial = adversarial_equivalence_frame_v2()
    expected_line = build_line_prefix_features(adversarial)
    observed_line = build_line_prefix_features_v2(adversarial)
    assert_exact_history_frame_equivalence_v2(expected_line, observed_line)
    expected_dyad = build_dyad_history_features(adversarial)
    observed_dyad, _ = build_dyad_history_features_v2(
        adversarial, requested_workers=1, accelerator="CPU"
    )
    assert_exact_history_frame_equivalence_v2(expected_dyad, observed_dyad)
    benchmark = benchmark_history_builders_v2(
        benchmark_cardinality_frame_v2(),
        repeats=repeats,
        minimum_speedup=minimum_speedup,
    )
    implementation = implementation_binding_v2()
    return seal_document({
        "schema_version": "stage6_expanded_history_preflight_equivalence_binding_v2",
        "r2_implementation_self_sha256": implementation["self_sha256"],
        "oracle": "frozen_r1_exact_frames_without_r1_implementation_hash",
        "adversarial_row_count": len(adversarial),
        "adversarial_source_sha256": hashlib.sha256(
            canonical_json_bytes(adversarial.astype(object).where(adversarial.notna(), None).to_dict("records"))
        ).hexdigest(),
        "line_exact_frame_equivalence_pass": True,
        "dyad_exact_frame_equivalence_pass": True,
        "benchmark_receipt": benchmark,
        "all_48_logical_lanes_present": len(set(adversarial["source_lane_id"].astype(str))) == 48,
        "whole_hand_temporal_semantics_changed": False,
        "action_timing_layer_included": False,
        "fresh_targets_opened": False,
        "operator_restart_allowed": True,
    })


def implementation_binding_v2() -> dict[str, Any]:
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    return seal_document({
        "schema_version": "stage6_expanded_history_implementation_binding_v2",
        "relative_path": "src/stage6_expanded_history_acceleration_v2.py",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "r1_oracle_module_mutated": False,
    })
