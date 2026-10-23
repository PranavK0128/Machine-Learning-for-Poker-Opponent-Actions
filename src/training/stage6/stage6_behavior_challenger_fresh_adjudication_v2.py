"""One-shot fresh adjudication for the frozen behavior challenger v2 winners.

This adapter opens targets only after an exclusive whole-hand row-panel claim.
It performs no fitting, calibration, selection, or refinement.  Qualification
remains held unless every preregistered comparison passes on the same rows.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_behavior_challenger_finalist_v2 import (
    FAMILY_ALPHA,
    FRESH_CUTOFF_UTC,
    MINIMUM_INDEPENDENT_GAME_CLUSTERS,
    REQUIRED_BASELINES,
    _science_package_binding,
    aggregate_run_id,
    aggregate_work_root,
)
from stage6_behavior_challenger_notebook_runtime_v2 import (
    ActiveWorkHeartbeat,
    HEARTBEAT_INTERVAL_SECONDS,
    PREFERRED_NOTEBOOK_HOST,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    SCOPES,
    SOURCE_DEPENDENCY_DIGEST,
    canonical_json_bytes,
    worker_policy_contract,
)
from stage6_behavior_json_publication_v2 import (
    load_json_with_recovery,
    recover_json,
    write_json_exclusive,
)
from stage6_expanded_feature_freeze_v2 import FEATURE_RUN_ID
from stage6_expanded_adjudication import (
    FRESH_PANEL_GENESIS_KEY,
    _load_fresh_source,
    _paired_cluster_inference,
    _validate_fresh_materialization,
)
from stage6_expanded_layers import seal_document, validate_document
from stage6_notebook11_runtime import (
    _extract_notebook09_package,
    _hand_sizes,
    _scope_adjudication,
)
from stage6_notebooks02_07_runtime import sha256_file


SUPPORTED_ACCELERATORS = ("CPU", "GPU", "TPU")
TARGET_FREE_CONTEXT_MANIFEST_NAME = (
    "stage6_behavior_v2_fresh_incumbent_context_manifest.json"
)
RESULT_MANIFEST_NAME = "fresh-adjudication-manifest.json"
RAM_RESERVE_BYTES = 4_000_000_000
MINIMUM_INFERENCE_WORKER_BYTES = 1_000_000_000


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _adjudication_json_binding(phase: str, run_identity: str) -> dict[str, Any]:
    return seal_document({
        "schema_version": "stage6_behavior_fresh_json_binding_v3",
        "phase": phase,
        "run_identity": run_identity,
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
    })


def _exact_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    phase: str,
    run_identity: str,
) -> None:
    write_json_exclusive(
        path,
        document,
        root=path.parent,
        publication_binding=_adjudication_json_binding(phase, run_identity),
    )


def _load_freeze(stage_drive_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = aggregate_work_root(stage_drive_root)
    freeze_path = root / "aggregate-freeze-manifest.json"
    custody_path = root / "fresh-custody-request.json"
    if not freeze_path.is_file() or not custody_path.is_file():
        raise RuntimeError("behavior finalist aggregate must freeze before fresh opening")
    finalist_binding = seal_document({
        "schema_version": "stage6_behavior_finalist_json_binding_v3",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
    })
    freeze = load_json_with_recovery(
        freeze_path, root=root, publication_binding=finalist_binding
    )
    custody = load_json_with_recovery(
        custody_path, root=root, publication_binding=finalist_binding
    )
    if (
        freeze.get("schema_version")
        != "stage6_behavior_challenger_aggregate_freeze_manifest_v3"
        or freeze.get("status")
        != "PASS_CANDIDATES_AND_CLAIMS_FROZEN_AWAITING_FRESH_DATA"
        or freeze.get("aggregate_run_id") != aggregate_run_id()
        or freeze.get("science_runtime_package_binding")
        != _science_package_binding()
        or freeze.get("behavior_implementation_binding")
        != BEHAVIOR_IMPLEMENTATION_BINDING
        or freeze.get("selection_or_tuning_after_freeze_allowed") is not False
        or freeze.get("fresh_targets_opened") is not False
        or custody.get("schema_version")
        != "stage6_behavior_challenger_fresh_custody_request_v3"
        or custody.get("aggregate_run_id") != aggregate_run_id()
        or custody.get("aggregate_freeze_manifest_self_sha256")
        != freeze["self_sha256"]
    ):
        raise RuntimeError("behavior aggregate freeze or custody identity is invalid")
    expected_claims = {
        f"{scope}:{baseline}"
        for scope in SCOPES
        for baseline in REQUIRED_BASELINES
    }
    if set(freeze.get("claim_family", [])) != expected_claims:
        raise RuntimeError("behavior aggregate preregistered claim family changed")
    return freeze, custody


def expected_fresh_custody_binding(
    freeze: Mapping[str, Any], custody: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "stage6_notebook01_fresh_custody_binding_v1",
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_self_sha256": custody["self_sha256"],
    }


def _fresh_run_id(
    freeze: Mapping[str, Any],
    custody: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    context_manifest: Mapping[str, Any],
    claim_self_sha256: str,
) -> str:
    return _digest({
        "schema": "stage6_behavior_challenger_fresh_adjudication_v3",
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_self_sha256": custody["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "incumbent_context_manifest_self_sha256": context_manifest["self_sha256"],
        "fresh_row_panel_claim_self_sha256": claim_self_sha256,
        "required_baselines": list(REQUIRED_BASELINES),
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "selection_or_tuning_after_open": False,
    })


def _context_file_identity(root: Path, row: Mapping[str, Any], name: str) -> Path:
    if row.get("name") != name or Path(str(row.get("name"))).name != name:
        raise RuntimeError("fresh incumbent context path is not canonical")
    path = root / name
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
        or pq.ParquetFile(path).metadata.num_rows != int(row.get("row_count", -1))
    ):
        raise RuntimeError("fresh incumbent context byte identity changed")
    prohibited = {
        "action_label",
        "target",
        "target_label",
        "outcome",
        "winner",
        "profit",
        "net_won",
    }
    schema_names = {value.lower() for value in pq.ParquetFile(path).schema.names}
    prohibited_tokens = (
        "action_label", "target_label", "hand_outcome", "showdown", "winner",
        "profit", "net_won", "pot_won", "private_card", "hole_card",
    )
    if schema_names & prohibited or any(
        token in column for column in schema_names for token in prohibited_tokens
    ):
        raise RuntimeError("fresh incumbent context contains a target/outcome column")
    return path


def validate_target_free_incumbent_context(
    root: Path,
    fresh_manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    stage_drive_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = root / TARGET_FREE_CONTEXT_MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError("fresh target-free incumbent context manifest is missing")
    from stage6_behavior_challenger_fresh_context_v2 import _context_json_binding

    document = load_json_with_recovery(
        path,
        root=root,
        publication_binding=_context_json_binding(
            "context", str(fresh_manifest["run_identity"])
        ),
    )
    files = document.get("files")
    if (
        document.get("schema_version")
        != "stage6_behavior_v2_fresh_incumbent_context_manifest_v2"
        or document.get("status")
        != "PASS_TARGET_FREE_EXACT_INCUMBENT_CONTEXT_FROZEN"
        or document.get("aggregate_run_id") != aggregate_run_id()
        or document.get("science_runtime_package_binding")
        != _science_package_binding()
        or document.get("behavior_implementation_binding")
        != BEHAVIOR_IMPLEMENTATION_BINDING
        or document.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or document.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
        or document.get("aggregate_freeze_manifest_self_sha256")
        != freeze["self_sha256"]
        or document.get("fresh_materialization_run_identity")
        != fresh_manifest["run_identity"]
        or document.get("fresh_materialization_manifest_self_sha256")
        != fresh_manifest["self_sha256"]
        or document.get("target_columns_included") is not False
        or document.get("built_before_any_fresh_target_read") is not True
        or not isinstance(document.get("target_free_native_source_manifest_self_sha256"), str)
        or not isinstance(document.get("target_free_native_source_producer_binding_self_sha256"), str)
        or not isinstance(files, list)
        or {row.get("name") for row in files}
        != {"preflop-native.parquet", "action-native-rolling.parquet"}
    ):
        raise RuntimeError("fresh target-free incumbent context authority is invalid")
    for row in files:
        _context_file_identity(root, row, str(row["name"]))
    if stage_drive_root is not None:
        from stage6_behavior_challenger_fresh_context_v2 import _load_source_bundle

        source_root = stage_drive_root / Path(
            str(document.get("target_free_native_source_root", ""))
        )
        source_manifest, _, _ = _load_source_bundle(
            stage_drive_root, source_root, fresh_manifest, freeze
        )
        if (
            source_manifest["self_sha256"]
            != document["target_free_native_source_manifest_self_sha256"]
            or source_manifest["producer_binding_self_sha256"]
            != document["target_free_native_source_producer_binding_self_sha256"]
        ):
            raise RuntimeError("fresh incumbent context producer/source binding changed")
    return document, root


def _metadata_fresh_panel(paths: Sequence[Path]) -> pd.DataFrame:
    """Read identity/time columns only; this is the pre-target-open operation."""
    frames = []
    for lane, path in enumerate(paths):
        frame = pq.read_table(
            path, columns=["row_id", "game_id", "hand_id", "decision_utc"]
        ).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
    identity = pd.concat(frames, ignore_index=True)
    for name in ("row_id", "game_id", "hand_id"):
        identity[name] = identity[name].astype(str)
    if identity["row_id"].duplicated().any():
        raise RuntimeError("fresh metadata panel duplicates row_id")
    identity["decision_ns"] = pd.to_datetime(
        identity["decision_utc"], utc=True, errors="raise"
    ).astype("int64")
    cutoff = pd.Timestamp(FRESH_CUTOFF_UTC).value
    hand_times = identity.groupby(["game_id", "hand_id"], sort=True)[
        "decision_ns"
    ].agg(["min", "max"])
    crossing = (hand_times["min"] < cutoff) & (hand_times["max"] >= cutoff)
    if crossing.any():
        raise RuntimeError("a whole hand crosses the behavior fresh cutoff")
    eligible = hand_times.index[hand_times["min"] >= cutoff]
    key = pd.MultiIndex.from_frame(identity[["game_id", "hand_id"]])
    selected = identity.loc[key.isin(eligible)].copy()
    if selected.empty:
        raise RuntimeError("new materialization has no whole-hand fresh rows")
    return selected


def _claim_panel(
    stage_drive_root: Path,
    freeze: Mapping[str, Any],
    custody: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    context_manifest: Mapping[str, Any],
    identity: pd.DataFrame,
) -> dict[str, Any]:
    if fresh_manifest["run_identity"] in set(
        custody.get("preexisting_notebook01_materialization_run_ids", [])
    ):
        raise RuntimeError("fresh materialization predates behavior aggregate freeze")
    current = {
        str(row.row_id): [
            str(row.game_id), str(row.hand_id), str(row.row_id), int(row.decision_ns)
        ]
        for row in identity.itertuples(index=False)
    }
    claims_root = (
        stage_drive_root / "runtime_state" / "expanded_layers"
        / "fresh_row_panel_claims"
    )
    previous_key = FRESH_PANEL_GENESIS_KEY
    previous_max_ns = int(pd.Timestamp(FRESH_CUTOFF_UTC).value) - 1
    consumed: set[str] = set()
    while True:
        chain_path = claims_root / f"{previous_key}.json"
        if not chain_path.is_file():
            break
        existing = load_json_with_recovery(
            chain_path,
            root=claims_root,
            publication_binding=_adjudication_json_binding(
                "global_claim", "global_fresh_claim_chain"
            ),
        )
        records = existing.get("row_identity_records")
        if not isinstance(records, list) or not records:
            raise RuntimeError("global fresh row-claim chain records are missing")
        normalized = [
            [str(row[0]), str(row[1]), str(row[2]), int(row[3])]
            for row in records
            if isinstance(row, list) and len(row) == 4
        ]
        row_ids = [row[2] for row in normalized]
        if (
            len(normalized) != len(records)
            or existing.get("schema_version")
            != "stage6_expanded_disjoint_fresh_row_panel_claim_v1"
            or existing.get("status") != "ATOMICALLY_CLAIMED_BEFORE_TARGET_OPEN"
            or existing.get("previous_claim_key") != previous_key
            or existing.get("target_columns_opened_before_claim") is not False
            or existing.get("row_count") != len(normalized)
            or existing.get("row_ids") != row_ids
            or len(set(row_ids)) != len(row_ids)
            or _digest(normalized) != existing.get("row_identity_root_sha256")
            or int(existing.get("panel_min_decision_ns")) <= previous_max_ns
            or int(existing.get("panel_max_decision_ns")) < int(existing.get("panel_min_decision_ns"))
            or consumed.intersection(row_ids)
        ):
            raise RuntimeError("global fresh row-claim chain is invalid")
        same_parent = (
            existing.get("aggregate_freeze_manifest_self_sha256") == freeze["self_sha256"]
            and existing.get("fresh_materialization_manifest_self_sha256")
            == fresh_manifest["self_sha256"]
        )
        if same_parent:
            if (
                existing.get("aggregate_run_id") != aggregate_run_id()
                or existing.get("science_runtime_package_binding")
                != _science_package_binding()
                or existing.get("behavior_implementation_self_sha256")
                != BEHAVIOR_IMPLEMENTATION_DIGEST
                or existing.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
                or existing.get("feature_run_id") != FEATURE_RUN_ID
                or existing.get("fresh_custody_request_self_sha256") != custody["self_sha256"]
                or existing.get("fresh_materialization_run_identity") != fresh_manifest["run_identity"]
                or existing.get("incumbent_context_manifest_self_sha256")
                != context_manifest["self_sha256"]
                or [current.get(row_id) for row_id in row_ids] != normalized
            ):
                raise RuntimeError("behavior fresh row-claim resume binding changed")
            return existing
        consumed.update(row_ids)
        previous_max_ns = int(existing["panel_max_decision_ns"])
        previous_key = str(existing["self_sha256"])

    hand_times = identity.groupby(["game_id", "hand_id"], sort=False)[
        "decision_ns"
    ].agg(["min", "max"])
    if ((hand_times["min"] <= previous_max_ns) & (hand_times["max"] > previous_max_ns)).any():
        raise RuntimeError("a hand crosses the next global fresh row-claim boundary")
    eligible_hands = hand_times.index[hand_times["min"] > previous_max_ns]
    keys = pd.MultiIndex.from_frame(identity[["game_id", "hand_id"]])
    panel = identity.loc[keys.isin(eligible_hands)].sort_values(
        ["decision_ns", "game_id", "hand_id", "row_id"], kind="mergesort"
    )
    if panel.empty:
        raise RuntimeError("no globally disjoint behavior fresh rows are available")
    records = [
        [str(row.game_id), str(row.hand_id), str(row.row_id), int(row.decision_ns)]
        for row in panel.itertuples(index=False)
    ]
    row_ids = [row[2] for row in records]
    if consumed.intersection(row_ids):
        raise RuntimeError("behavior fresh panel reuses a globally claimed row_id")
    claim = seal_document({
        "schema_version": "stage6_expanded_disjoint_fresh_row_panel_claim_v1",
        "status": "ATOMICALLY_CLAIMED_BEFORE_TARGET_OPEN",
        "previous_claim_key": previous_key,
        "aggregate_run_id": aggregate_run_id(),
        "science_runtime_package_binding": _science_package_binding(),
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
        "feature_run_id": FEATURE_RUN_ID,
        "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_self_sha256": custody["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "incumbent_context_manifest_self_sha256": context_manifest["self_sha256"],
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "panel_min_decision_ns": int(panel["decision_ns"].min()),
        "panel_max_decision_ns": int(panel["decision_ns"].max()),
        "row_count": len(records),
        "whole_hand_count": len({(row[0], row[1]) for row in records}),
        "game_count": len({row[0] for row in records}),
        "row_identity_root_sha256": _digest(records),
        "row_identity_records": records,
        "row_ids": row_ids,
        "claimed_row_ids": row_ids,
        "metadata_columns_read_before_claim": [
            "row_id",
            "game_id",
            "hand_id",
            "decision_utc",
        ],
        "fresh_target_columns_read_before_claim": [],
        "target_columns_opened_before_claim": False,
        "selection_or_tuning_after_claim_allowed": False,
        "one_shot_adjudication": True,
    })
    chain_path = claims_root / f"{previous_key}.json"
    _exact_json(
        chain_path,
        claim,
        phase="global_claim",
        run_identity="global_fresh_claim_chain",
    )
    claim_path = aggregate_work_root(stage_drive_root) / "fresh_adjudication" / "fresh-row-panel-claim.json"
    _exact_json(
        claim_path,
        claim,
        phase="local_claim",
        run_identity=str(fresh_manifest["run_identity"]),
    )
    return claim


def resolve_inference_workers(
    requested: str,
    unfinished_scopes: int,
    exact_artifact_bytes: Sequence[int],
    *,
    accelerator: str,
) -> dict[str, Any]:
    policy = worker_policy_contract(accelerator)
    if unfinished_scopes <= 0 or not exact_artifact_bytes:
        raise RuntimeError("inference workers require unfinished scopes and artifacts")
    raw = str(requested).strip()
    cpu = max(1, int(os.cpu_count() or 1))
    if raw.lower() == "auto":
        requested_capacity = cpu
        mode = "auto"
    elif raw.isascii() and raw.isdecimal() and int(raw) > 0:
        requested_capacity = int(raw)
        mode = "explicit"
    else:
        raise RuntimeError("REQUESTED_WORKERS must be auto or a positive integer")
    import psutil

    available = int(psutil.virtual_memory().available)
    peak = max(MINIMUM_INFERENCE_WORKER_BYTES, 4 * max(map(int, exact_artifact_bytes)))
    if peak > available - RAM_RESERVE_BYTES:
        raise RuntimeError("one adjudication scope cannot fit the RAM reserve")
    ram_capacity = max(1, (available - RAM_RESERVE_BYTES) // peak)
    live = min(requested_capacity, cpu, ram_capacity, unfinished_scopes)
    return {
        "requested_workers": raw,
        "requested_workers_mode": mode,
        "detected_usable_logical_cpus": cpu,
        "available_system_ram_bytes": available,
        "estimated_worker_peak_bytes": peak,
        "ram_bounded_worker_capacity": int(ram_capacity),
        "unfinished_scopes": int(unfinished_scopes),
        "live_scope_workers": int(live),
        "candidate_internal_threads": 1,
        "worker_policy": policy,
    }


def holm_bonferroni(
    inferences: Mapping[str, Mapping[str, Any]], alpha: float = FAMILY_ALPHA
) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        inferences,
        key=lambda key: (
            float(inferences[key]["paired_sign_flip_two_sided_p_value"]), key
        ),
    )
    output: dict[str, dict[str, Any]] = {}
    continue_rejecting = True
    for rank, key in enumerate(ordered, start=1):
        threshold = float(alpha) / (len(ordered) - rank + 1)
        p_value = float(inferences[key]["paired_sign_flip_two_sided_p_value"])
        reject = continue_rejecting and p_value <= threshold
        if not reject:
            continue_rejecting = False
        output[key] = {
            "holm_rank": rank,
            "raw_p_value": p_value,
            "holm_threshold": threshold,
            "reject_no_improvement_null": reject,
        }
    return output


def claim_passes(
    inference: Mapping[str, Any],
    holm: Mapping[str, Any],
    candidate_brier: float,
    comparator_brier: float,
) -> bool:
    return bool(
        inference.get("independent_cluster_gate_met")
        and float(inference["mean_candidate_minus_control_log_loss"]) < 0.0
        and float(inference["confidence_interval_95"][1]) < 0.0
        and holm.get("reject_no_improvement_null") is True
        and float(candidate_brier) <= float(comparator_brier)
    )


def _load_artifacts(
    stage_drive_root: Path, freeze: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts = {}
    identities = []
    for behavior_scope in SCOPES:
        frozen = freeze["scope_freezes"][behavior_scope]
        path = stage_drive_root / Path(frozen["artifact_relative_path"])
        identity = frozen["selected_artifact"]
        if (
            not path.is_file()
            or path.name != identity["name"]
            or path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != identity["sha256"]
        ):
            raise RuntimeError("frozen behavior artifact byte identity changed")
        artifact = joblib.load(path)
        expected_scope = "preflop" if behavior_scope == "preflop_behavior" else "postflop"
        if (
            artifact.get("candidate_id") != frozen["selected_candidate_id"]
            or artifact.get("configuration_sha256")
            != frozen["selected_configuration_sha256"]
            or artifact.get("tournament_run_id") != aggregate_run_id()
            or artifact.get("science_runtime_package_binding")
            != _science_package_binding()
            or artifact.get("street_scope") != expected_scope
            or artifact.get("holdout_targets_used") != 0
        ):
            raise RuntimeError("frozen behavior artifact authority changed")
        artifacts[expected_scope] = artifact
        identities.append({
            "scope": behavior_scope,
            "path": str(path),
            "artifact_relative_path": frozen["artifact_relative_path"],
            "candidate_id": frozen["selected_candidate_id"],
            "configuration_sha256": frozen["selected_configuration_sha256"],
            **identity,
        })
    return artifacts, identities


def _validate_completed_fresh_result(
    manifest_path: Path,
    *,
    root: Path,
    run_id: str,
    freeze: Mapping[str, Any],
    custody: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    context_manifest: Mapping[str, Any],
    claim: Mapping[str, Any],
    stage_drive_root: Path,
) -> dict[str, Any]:
    completed = load_json_with_recovery(
        manifest_path,
        root=root,
        publication_binding=_adjudication_json_binding("terminal", run_id),
    )
    reports = completed.get("scope_reports")
    inference = completed.get("paired_game_cluster_inference")
    holm = completed.get("holm_bonferroni")
    gates = completed.get("scope_qualification_gates")
    checkpoints = completed.get("scope_checkpoints")
    artifact_receipts = completed.get("selected_artifact_receipts")
    if (
        completed.get("schema_version")
        != "stage6_behavior_challenger_fresh_adjudication_manifest_v3"
        or completed.get("status") not in {
            "PASS_ALL_PREREGISTERED_BEHAVIOR_CLAIMS",
            "HOLD_ONE_OR_MORE_BEHAVIOR_CLAIMS_FAILED",
        }
        or completed.get("fresh_adjudication_run_id") != run_id
        or completed.get("aggregate_run_id") != aggregate_run_id()
        or completed.get("aggregate_freeze_manifest_self_sha256")
        != freeze["self_sha256"]
        or completed.get("fresh_custody_request_self_sha256") != custody["self_sha256"]
        or completed.get("fresh_materialization_run_identity")
        != fresh_manifest["run_identity"]
        or completed.get("fresh_materialization_manifest_self_sha256")
        != fresh_manifest["self_sha256"]
        or completed.get("incumbent_context_manifest_self_sha256")
        != context_manifest["self_sha256"]
        or completed.get("fresh_row_panel_claim_self_sha256") != claim["self_sha256"]
        or completed.get("fresh_row_count") != claim["row_count"]
        or completed.get("fresh_whole_hand_count") != claim["whole_hand_count"]
        or completed.get("fresh_game_count") != claim["game_count"]
        or completed.get("feature_run_id") != FEATURE_RUN_ID
        or completed.get("behavior_implementation_binding")
        != BEHAVIOR_IMPLEMENTATION_BINDING
        or completed.get("behavior_implementation_self_sha256")
        != BEHAVIOR_IMPLEMENTATION_DIGEST
        or completed.get("source_dependency_digest") != SOURCE_DEPENDENCY_DIGEST
        or completed.get("science_runtime_package_binding")
        != _science_package_binding()
        or completed.get("required_baselines") != list(REQUIRED_BASELINES)
        or completed.get("selection_or_tuning_after_fresh_open_count") != 0
        or completed.get("refinement_after_fresh_open_count") != 0
        or not isinstance(reports, Mapping)
        or set(reports) != {"preflop", "postflop"}
        or not isinstance(inference, Mapping)
        or set(inference) != {
            f"{scope}:{baseline}"
            for scope in SCOPES for baseline in REQUIRED_BASELINES
        }
        or not isinstance(holm, Mapping)
        or not isinstance(gates, Mapping)
        or not isinstance(checkpoints, Mapping)
        or set(checkpoints) != set(SCOPES)
        or not isinstance(artifact_receipts, Mapping)
        or set(artifact_receipts) != set(SCOPES)
    ):
        raise RuntimeError("completed fresh behavior result authority changed")
    if holm_bonferroni(inference) != holm:
        raise RuntimeError("completed fresh Holm correction changed")
    recomputed_gates: dict[str, bool] = {}
    for short_scope in ("preflop", "postflop"):
        behavior_scope = f"{short_scope}_behavior"
        report = reports[short_scope]
        comparison = {}
        for comparator in REQUIRED_BASELINES:
            key = f"{behavior_scope}:{comparator}"
            comparison[comparator] = claim_passes(
                inference[key], holm[key],
                report["metrics"]["candidate"]["multiclass_brier_score"],
                report["metrics"][comparator]["multiclass_brier_score"],
            )
        recomputed_gates[behavior_scope] = bool(
            all(comparison.values())
            and report["illegal_postprojection_count"] == 0
            and report["missing_or_defaulted_required_incumbent_input_count"] == 0
        )
        if report.get("preregistered_comparison_gates") != comparison:
            raise RuntimeError("completed fresh report gates changed")
        checkpoint_identity = checkpoints[behavior_scope]
        checkpoint_path = root / str(checkpoint_identity["name"])
        checkpoint = load_json_with_recovery(
            checkpoint_path,
            root=root,
            publication_binding=_adjudication_json_binding("scope", run_id),
        )
        artifact = artifact_receipts[behavior_scope]
        artifact_path = stage_drive_root / str(artifact["artifact_relative_path"])
        frozen = freeze["scope_freezes"][behavior_scope]
        if (
            checkpoint_path.stat().st_size != checkpoint_identity["bytes"]
            or sha256_file(checkpoint_path) != checkpoint_identity["sha256"]
            or checkpoint["self_sha256"] != checkpoint_identity["self_sha256"]
            or checkpoint.get("fresh_adjudication_run_id") != run_id
            or checkpoint.get("scope") != behavior_scope
            or checkpoint.get("science_runtime_package_binding")
            != _science_package_binding()
            or checkpoint.get("behavior_implementation_self_sha256")
            != BEHAVIOR_IMPLEMENTATION_DIGEST
            or checkpoint.get("selected_artifact_receipt") != artifact
            or checkpoint.get("scope_report") != report
            or checkpoint.get("claim_inferences")
            != {key: inference[key] for key in inference if key.startswith(f"{behavior_scope}:")}
            or checkpoint.get("holm_results")
            != {key: holm[key] for key in holm if key.startswith(f"{behavior_scope}:")}
            or checkpoint.get("scope_qualification_gate")
            != recomputed_gates[behavior_scope]
            or artifact.get("candidate_id") != frozen["selected_candidate_id"]
            or artifact.get("configuration_sha256")
            != frozen["selected_configuration_sha256"]
            or artifact.get("sha256") != frozen["selected_artifact"]["sha256"]
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact["bytes"]
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError("completed fresh scope checkpoint/artifact changed")
    if recomputed_gates != gates:
        raise RuntimeError("completed fresh scope qualification gates changed")
    all_pass = all(recomputed_gates.values())
    if (
        completed["status"]
        != ("PASS_ALL_PREREGISTERED_BEHAVIOR_CLAIMS" if all_pass else
            "HOLD_ONE_OR_MORE_BEHAVIOR_CLAIMS_FAILED")
        or completed.get("qualification_authority") is not all_pass
        or completed.get("assistant_installation_allowed") is not False
        or completed.get("recommendation_authority") is not False
        or completed.get("action_authority") is not False
        or completed.get("live_action_authority") is not False
    ):
        raise RuntimeError("completed fresh terminal authority changed")
    return completed


def run_fresh_adjudication(
    stage_drive_root: Path,
    fresh_materialization_root: Path,
    incumbent_context_root: Path,
    requested_workers: str = "auto",
    accelerator: str = "TPU",
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    accelerator = str(accelerator).strip().upper()
    if accelerator not in SUPPORTED_ACCELERATORS:
        raise ValueError("accelerator must be CPU, GPU, or TPU")
    freeze, custody = _load_freeze(stage_drive_root)
    from stage6_behavior_challenger_fresh_context_v2 import (
        validate_fresh_manifest_metadata_only,
    )

    fresh_manifest, paths = validate_fresh_manifest_metadata_only(
        stage_drive_root,
        fresh_materialization_root,
        expected_fresh_custody_binding(freeze, custody),
    )
    context_manifest, context_root = validate_target_free_incumbent_context(
        incumbent_context_root, fresh_manifest, freeze,
        stage_drive_root=stage_drive_root,
    )
    identity = _metadata_fresh_panel(paths)
    claim = _claim_panel(
        stage_drive_root, freeze, custody, fresh_manifest, context_manifest, identity
    )
    verified_manifest, verified_paths = _validate_fresh_materialization(
        stage_drive_root,
        fresh_materialization_root,
        expected_fresh_custody_binding(freeze, custody),
    )
    if verified_manifest != fresh_manifest or verified_paths != paths:
        raise RuntimeError("fresh materialization changed after metadata-only claim")
    run_id = _fresh_run_id(
        freeze, custody, fresh_manifest, context_manifest, claim["self_sha256"]
    )
    root = aggregate_work_root(stage_drive_root) / "fresh_adjudication" / run_id
    manifest_path = root / RESULT_MANIFEST_NAME
    recover_json(
        manifest_path,
        root=root,
        publication_binding=_adjudication_json_binding("terminal", run_id),
    )
    if manifest_path.is_file():
        completed = _validate_completed_fresh_result(
            manifest_path,
            root=root,
            run_id=run_id,
            freeze=freeze,
            custody=custody,
            fresh_manifest=fresh_manifest,
            context_manifest=context_manifest,
            claim=claim,
            stage_drive_root=stage_drive_root,
        )
        return {
            "status": completed["status"],
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": completed["self_sha256"],
            "reused_completed_result": True,
        }

    heartbeat = ActiveWorkHeartbeat(
        scope="behavior_fresh_adjudication",
        shard=-1,
        interval_seconds=heartbeat_interval_seconds,
    )
    heartbeat.update(phase="opening_claimed_fresh_targets", denominator=48, completed=0, work_unit="lanes")
    heartbeat.start()
    started = time.monotonic()
    artifacts = {}
    source: pd.DataFrame | None = None
    notebook09_root: Path | None = None
    try:
        artifacts, artifact_identities = _load_artifacts(stage_drive_root, freeze)
        worker_report = resolve_inference_workers(
            requested_workers,
            len(SCOPES),
            [int(row["bytes"]) for row in artifact_identities],
            accelerator=accelerator,
        )
        source = _load_fresh_source(paths)
        claimed = set(claim["claimed_row_ids"])
        source = source.loc[source["row_id"].astype(str).isin(claimed)].copy()
        if set(source["row_id"].astype(str)) != claimed or len(source) != claim["row_count"]:
            raise RuntimeError("opened fresh rows do not equal the preclaimed panel")
        heartbeat.update(phase="opening_claimed_fresh_targets", denominator=48, completed=48, work_unit="lanes")
        local_root = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        notebook09_root = local_root / f"stage6-behavior-v2-incumbent-{run_id}"
        _extract_notebook09_package(stage_drive_root, notebook09_root)
        hand_sizes = _hand_sizes(source)
        reports: dict[str, Any] = {}
        inference: dict[str, Any] = {}
        heartbeat.update(phase="fresh_identical_row_comparisons", denominator=2, completed=0, work_unit="scopes")

        def evaluate_scope(scope: str):
            mask = (
                source["street"].astype(str).eq("preflop")
                if scope == "preflop"
                else source["street"].astype(str).isin(("flop", "turn", "river"))
            )
            return _scope_adjudication(
                scope,
                source.loc[mask].copy(),
                artifacts[scope],
                notebook09_root,
                context_root,
                hand_sizes,
            )

        with ThreadPoolExecutor(
            max_workers=int(worker_report["live_scope_workers"])
        ) as executor:
            futures = {
                executor.submit(evaluate_scope, scope): scope
                for scope in ("preflop", "postflop")
            }
            completed_scopes = 0
            for future in as_completed(futures):
                scope = futures[future]
                report, differences, clusters = future.result()
                if set(differences) != set(REQUIRED_BASELINES):
                    raise RuntimeError("fresh behavior comparator inventory changed")
                reports[scope] = report
                for comparator in REQUIRED_BASELINES:
                    hand_clusters = clusters[comparator]
                    game_clusters = [
                        value.split("\x1f", 1)[0] for value in hand_clusters
                    ]
                    key = f"{scope}_behavior:{comparator}"
                    inference[key] = _paired_cluster_inference(
                        np.asarray(differences[comparator], dtype=np.float64),
                        game_clusters,
                        int(_digest({"run": run_id, "claim": key})[:8], 16),
                        MINIMUM_INDEPENDENT_GAME_CLUSTERS,
                    )
                    inference[key]["whole_hand_cluster_count"] = len(
                        set(hand_clusters)
                    )
                completed_scopes += 1
                heartbeat.update(
                    phase="fresh_identical_row_comparisons",
                    denominator=2,
                    completed=completed_scopes,
                    work_unit="scopes",
                )
        holm = holm_bonferroni(inference)
        scope_gates = {}
        for scope in ("preflop", "postflop"):
            report = reports[scope]
            comparison_gates = {}
            for comparator in REQUIRED_BASELINES:
                key = f"{scope}_behavior:{comparator}"
                comparison_gates[comparator] = claim_passes(
                    inference[key],
                    holm[key],
                    report["metrics"]["candidate"]["multiclass_brier_score"],
                    report["metrics"][comparator]["multiclass_brier_score"],
                )
            scope_gates[f"{scope}_behavior"] = bool(
                all(comparison_gates.values())
                and report["illegal_postprojection_count"] == 0
                and report["missing_or_defaulted_required_incumbent_input_count"] == 0
            )
            report["preregistered_comparison_gates"] = comparison_gates
        all_pass = all(scope_gates.values())
        selected_artifact_receipts = {
            str(row["scope"]): {
                key: value for key, value in row.items()
                if key not in {"scope", "path"}
            }
            for row in artifact_identities
        }
        scope_checkpoints: dict[str, dict[str, Any]] = {}
        for short_scope in ("preflop", "postflop"):
            behavior_scope = f"{short_scope}_behavior"
            scope_checkpoint = seal_document({
                "schema_version": (
                    "stage6_behavior_challenger_fresh_scope_checkpoint_v3"
                ),
                "status": "PASS_EXACT_FRESH_SCOPE_EVALUATION_COMPLETE",
                "fresh_adjudication_run_id": run_id,
                "aggregate_run_id": aggregate_run_id(),
                "science_runtime_package_binding": _science_package_binding(),
                "behavior_implementation_self_sha256": (
                    BEHAVIOR_IMPLEMENTATION_DIGEST
                ),
                "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
                "feature_run_id": FEATURE_RUN_ID,
                "fresh_row_panel_claim_self_sha256": claim["self_sha256"],
                "scope": behavior_scope,
                "candidate_id": artifacts[short_scope]["candidate_id"],
                "configuration_sha256": artifacts[short_scope][
                    "configuration_sha256"
                ],
                "selected_artifact_receipt": selected_artifact_receipts[
                    behavior_scope
                ],
                "identical_row_comparison": True,
                "scope_report": reports[short_scope],
                "claim_inferences": {
                    key: inference[key] for key in inference
                    if key.startswith(f"{behavior_scope}:")
                },
                "holm_results": {
                    key: holm[key] for key in holm
                    if key.startswith(f"{behavior_scope}:")
                },
                "scope_qualification_gate": scope_gates[behavior_scope],
                "selection_or_tuning_after_fresh_open_count": 0,
                "qualification_authority": False,
            })
            checkpoint_path = root / f"{short_scope}-scope-checkpoint.json"
            _exact_json(
                checkpoint_path,
                scope_checkpoint,
                phase="scope",
                run_identity=run_id,
            )
            scope_checkpoints[behavior_scope] = {
                "name": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
                "self_sha256": scope_checkpoint["self_sha256"],
            }
        if any(
            path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]
            for path, row in (
                (Path(item["path"]), item) for item in artifact_identities
            )
        ):
            raise RuntimeError("frozen candidate artifacts changed during adjudication")
        manifest = seal_document({
            "schema_version": "stage6_behavior_challenger_fresh_adjudication_manifest_v3",
            "status": (
                "PASS_ALL_PREREGISTERED_BEHAVIOR_CLAIMS"
                if all_pass
                else "HOLD_ONE_OR_MORE_BEHAVIOR_CLAIMS_FAILED"
            ),
            "fresh_adjudication_run_id": run_id,
            "aggregate_run_id": aggregate_run_id(),
            "science_runtime_package_binding": _science_package_binding(),
            "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "source_dependency_digest": SOURCE_DEPENDENCY_DIGEST,
            "feature_run_id": FEATURE_RUN_ID,
            "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
            "fresh_custody_request_self_sha256": custody["self_sha256"],
            "fresh_materialization_run_identity": fresh_manifest["run_identity"],
            "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
            "incumbent_context_manifest_self_sha256": context_manifest["self_sha256"],
            "fresh_row_panel_claim_self_sha256": claim["self_sha256"],
            "fresh_row_count": claim["row_count"],
            "fresh_whole_hand_count": claim["whole_hand_count"],
            "fresh_game_count": claim["game_count"],
            "scope_reports": reports,
            "paired_game_cluster_inference": inference,
            "holm_bonferroni": holm,
            "scope_qualification_gates": scope_gates,
            "scope_checkpoints": scope_checkpoints,
            "selected_artifact_receipts": selected_artifact_receipts,
            "required_baselines": list(REQUIRED_BASELINES),
            "identical_row_comparison_required": True,
            "whole_hand_clustering_reported": True,
            "game_clustering_used_for_inference": True,
            "legal_projection_required": True,
            "selection_or_tuning_after_fresh_open_count": 0,
            "refinement_after_fresh_open_count": 0,
            "worker_report": worker_report,
            "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
            "model_execution_device": "host_cpu_process",
            "tree_training_uses_tpu_acceleration": False,
            "qualification_authority": all_pass,
            "assistant_installation_allowed": False,
            "recommendation_authority": False,
            "action_authority": False,
            "live_action_authority": False,
            "independent_audit_required_before_any_installation": True,
            "elapsed_seconds": time.monotonic() - started,
        })
        _exact_json(
            manifest_path,
            manifest,
            phase="terminal",
            run_identity=run_id,
        )
        return {
            "status": manifest["status"],
            "manifest_path": str(manifest_path),
            "manifest_self_sha256": manifest["self_sha256"],
            "scope_qualification_gates": scope_gates,
            "reused_completed_result": False,
        }
    finally:
        heartbeat.stop()
        artifacts.clear()
        if source is not None:
            del source
        if notebook09_root is not None and notebook09_root.is_dir():
            shutil.rmtree(notebook09_root)
        gc.collect()


__all__ = [
    "TARGET_FREE_CONTEXT_MANIFEST_NAME",
    "claim_passes",
    "expected_fresh_custody_binding",
    "holm_bonferroni",
    "resolve_inference_workers",
    "run_fresh_adjudication",
    "validate_target_free_incumbent_context",
]
