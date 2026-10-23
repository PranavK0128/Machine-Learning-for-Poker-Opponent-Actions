"""Guarded sequential auxiliary refits for four qualified non-CatBoost layers.

The supervisor protects the already-running primary installer.  The worker
publishes only complete, atomically staged layer directories and never writes
the combined install/evidence manifests or an active runtime pointer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


TARGET_LAYERS = (
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "opponent_bluff_likelihood",
    "supported_fold_response_bluff_opportunity",
)


def _event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, sort_keys=True), flush=True)


def _terminate(child: subprocess.Popen[Any], reason: str) -> int:
    _event("stage6_auxiliary_noncat_stop_requested", reason=reason, pid=child.pid)
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=30)
    _event(
        "stage6_auxiliary_noncat_stopped",
        reason=reason,
        pid=child.pid,
        returncode=child.returncode,
    )
    return 75


def supervise(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--workspace",
        str(args.workspace),
        "--stage6-root",
        str(args.stage6_root),
        "--assistant-root",
        str(args.assistant_root),
        "--materialization-root",
        str(args.materialization_root),
        "--aggregate-zip",
        str(args.aggregate_zip),
        "--additional-input-zip",
        str(args.additional_input_zip),
    ]
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    child = subprocess.Popen(command, env=environment)
    child_process = psutil.Process(child.pid)
    primary = psutil.Process(args.primary_pid)
    within_dir = (
        args.assistant_root
        / "runtime/ml/artifacts/stage6_retrospective_nested_v1"
        / "within_hand_line_surprise"
    )
    low_ram_samples = 0
    sample_seconds = 5.0
    progress_window_seconds = 30.0
    cpu_window_started = time.monotonic()
    primary_cpu_started = sum(primary.cpu_times()[:2])
    last_report = 0.0
    started = time.monotonic()
    _event(
        "stage6_auxiliary_noncat_supervisor_started",
        pid=child.pid,
        primary_pid=args.primary_pid,
        layers=list(TARGET_LAYERS),
        minimum_available_ram_bytes=args.minimum_available_ram_bytes,
    )
    while child.poll() is None:
        time.sleep(sample_seconds)
        if not psutil.pid_exists(args.primary_pid):
            return _terminate(child, "primary_installer_exited")
        if within_dir.exists():
            return _terminate(child, "primary_within_hand_layer_completed")
        available = int(psutil.virtual_memory().available)
        if available < args.minimum_available_ram_bytes:
            low_ram_samples += 1
        else:
            low_ram_samples = 0
        if low_ram_samples >= args.low_ram_samples_before_stop:
            return _terminate(child, "sustained_low_available_ram")

        now = time.monotonic()
        if now - cpu_window_started >= progress_window_seconds:
            current_cpu = sum(primary.cpu_times()[:2])
            cpu_delta = current_cpu - primary_cpu_started
            if (
                now - started >= args.primary_progress_grace_seconds
                and cpu_delta < args.minimum_primary_cpu_seconds_per_window
            ):
                return _terminate(child, "primary_refit_cpu_progress_degraded")
            cpu_window_started = now
            primary_cpu_started = current_cpu
        if now - last_report >= 30.0:
            _event(
                "stage6_auxiliary_noncat_supervisor_status",
                pid=child.pid,
                available_ram_bytes=available,
                low_ram_samples=low_ram_samples,
                primary_cpu_seconds=sum(primary.cpu_times()[:2]),
                child_private_bytes=(
                    child_process.memory_info().private
                    if hasattr(child_process.memory_info(), "private")
                    else child_process.memory_info().rss
                ),
            )
            last_report = now
    _event(
        "stage6_auxiliary_noncat_supervisor_complete",
        pid=child.pid,
        returncode=child.returncode,
    )
    return int(child.returncode or 0)


def worker(args: argparse.Namespace) -> int:
    import gc

    import numpy as np
    import pandas as pd

    from stage6_additional_exploitation_layers import _CausalHistoryIndex
    from stage6_notebooks02_07_runtime import LEGAL_COLUMNS
    from stage6_retrospective_nested_refit_install import (
        ADDITIONAL_BUILDERS,
        MODEL_LABELS,
        QUALIFICATION_INDEX_RELATIVE,
        VERSION_NAME,
        _additional_source,
        _assert_canonical,
        _existing_layer_row,
        _feature_names,
        _frozen_specs,
        _load_aggregate,
        _train_layer,
        _validate_qualification_index,
    )

    workspace = args.workspace.resolve()
    stage6_root = _assert_canonical(args.stage6_root, workspace)
    assistant_root = _assert_canonical(args.assistant_root, workspace)
    materialization_root = _assert_canonical(args.materialization_root, workspace)
    aggregate_zip = _assert_canonical(args.aggregate_zip, workspace)
    additional_zip = _assert_canonical(args.additional_input_zip, workspace)
    aggregate = _load_aggregate(aggregate_zip)
    _validate_qualification_index(
        stage6_root / QUALIFICATION_INDEX_RELATIVE, aggregate
    )
    specs = _frozen_specs(aggregate)
    candidate_root = assistant_root / "runtime/ml/artifacts" / VERSION_NAME
    if not candidate_root.is_dir():
        raise RuntimeError("primary versioned candidate root is missing")
    if (candidate_root / "install_manifest.json").exists():
        raise RuntimeError("primary versioned installation is already complete")

    pending = []
    for layer in TARGET_LAYERS:
        existing = _existing_layer_row(candidate_root / layer, specs[layer])
        if existing is None:
            pending.append(layer)
        else:
            _event("stage6_auxiliary_noncat_resumed", **existing)
    if not pending:
        _event("stage6_auxiliary_noncat_complete", layers=[])
        return 0

    _event("stage6_auxiliary_noncat_source_load_started", layers=pending)
    context, history, identity, source_binding = _additional_source(
        stage6_root, materialization_root, additional_zip
    )
    history_index = _CausalHistoryIndex(history)
    del history
    gc.collect()
    empty_history = pd.DataFrame()
    completed = []
    for layer in pending:
        destination = candidate_root / layer
        if destination.exists():
            raise RuntimeError(f"auxiliary destination appeared concurrently: {layer}")
        _event("stage6_auxiliary_noncat_refit_started", layer=layer)
        frame = ADDITIONAL_BUILDERS[layer](
            context, empty_history, _index=history_index
        )
        joined = identity.merge(frame, on="row_id", how="inner", validate="one_to_one")
        if len(joined) != len(identity) or len(joined) != len(frame):
            raise RuntimeError(f"auxiliary identity/feature join is incomplete: {layer}")
        names = list(_feature_names(frame))
        matrix = joined[names].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )
        if np.isinf(matrix).any():
            raise RuntimeError(f"auxiliary feature contains infinity: {layer}")
        target = np.asarray(
            [MODEL_LABELS.index(value) for value in joined["action_label"].astype(str)],
            dtype=np.int8,
        )
        legal = joined[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
        row = _train_layer(
            spec=specs[layer],
            matrix=matrix,
            target=target,
            legal=legal,
            roles=joined["temporal_role"].astype(str).to_numpy(),
            feature_names=names,
            feature_frame=frame,
            output_dir=destination,
            source_binding=source_binding,
        )
        completed.append(layer)
        _event("stage6_auxiliary_noncat_refit_complete", **row)
        del frame, joined, matrix, target, legal
        gc.collect()
    _event("stage6_auxiliary_noncat_complete", layers=completed)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage6-root", type=Path, required=True)
    parser.add_argument("--assistant-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--aggregate-zip", type=Path, required=True)
    parser.add_argument("--additional-input-zip", type=Path, required=True)
    parser.add_argument("--primary-pid", type=int, default=9236)
    parser.add_argument(
        "--minimum-available-ram-bytes", type=int, default=536_870_912
    )
    parser.add_argument("--low-ram-samples-before-stop", type=int, default=6)
    parser.add_argument("--primary-progress-grace-seconds", type=float, default=120.0)
    parser.add_argument(
        "--minimum-primary-cpu-seconds-per-window", type=float, default=10.0
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
