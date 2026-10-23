"""Freeze expanded-layer finalists and adjudicate them once on fresh hands."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_behavior_tournament import MODEL_SOURCE_COLUMNS
from stage6_expanded_feature_freeze import (
    BASE_FEATURE_COLUMNS,
    FEATURE_RUN_ID,
    build_feature_frames,
)
from stage6_expanded_finalist import (
    DYAD_MINIMUM_HAND_SUPPORT,
    LINE_MINIMUM_PREFIX_ACTIONS,
    finalist_run_id,
    finalist_work_root,
)
from stage6_expanded_layers import (
    ACTIONS,
    DEVELOPMENT_CUTOFF_UTC,
    EXPANDED_LAYER_CONTRACT_DIGEST,
    FEATURE_SOURCE_COLUMNS,
    LAYERS,
    LOGICAL_LANE_COUNT,
    SCIENCE_IMPLEMENTATION_DIGEST,
    canonical_json_bytes,
    seal_document,
    validate_document,
)
from stage6_expanded_tournament import (
    _aligned_probabilities,
    _apply_imputer,
    _project_legal,
    _temperature_scale,
    expanded_model_matrix,
)
from stage6_metrics import multiclass_metric_report
from stage6_notebooks02_07_runtime import NOTEBOOK01_RUN_ID, sha256_file


BOOTSTRAP_REPLICATES = 2_000
SIGN_FLIP_REPLICATES = 10_000
FAMILY_ALPHA = 0.05
PAIRED_CLUSTER_IDENTITY = "game_id"
MINIMUM_INDEPENDENT_GAME_CLUSTERS = 20
FRESH_PANEL_GENESIS_KEY = hashlib.sha256(canonical_json_bytes({
    "schema": "stage6_expanded_fresh_panel_genesis_v1",
    "cutoff_utc_inclusive": DEVELOPMENT_CUTOFF_UTC,
})).hexdigest()


def aggregate_run_id() -> str:
    return hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_layer_aggregate_freeze_v1",
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation": SCIENCE_IMPLEMENTATION_DIGEST,
        "finalist_run_ids": {layer: finalist_run_id(layer) for layer in LAYERS},
        "fresh_cutoff_utc_inclusive": DEVELOPMENT_CUTOFF_UTC,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "paired_cluster_identity": PAIRED_CLUSTER_IDENTITY,
        "minimum_independent_game_clusters_per_claim": (
            MINIMUM_INDEPENDENT_GAME_CLUSTERS
        ),
        "multiple_comparison_method": "holm_bonferroni",
        "family_alpha": FAMILY_ALPHA,
    })).hexdigest()


def aggregate_work_root(stage_drive_root: Path) -> Path:
    return (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "aggregate_freezes"
        / aggregate_run_id()
    )


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    from stage6_expanded_feature_freeze import _atomic_json as write

    write(path, document)


def _load_finalist(stage_drive_root: Path, layer: str) -> tuple[dict[str, Any], Path]:
    root = finalist_work_root(stage_drive_root, layer)
    path = root / "finalist-manifest.json"
    if not path.is_file():
        raise RuntimeError(f"{layer} finalists must complete before aggregate freeze")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("schema_version") != "stage6_expanded_finalist_manifest_v1"
        or manifest.get("status") != "PASS_FINALIST_FROZEN_RETROSPECTIVE_ONLY"
        or manifest.get("finalist_run_id") != finalist_run_id(layer)
        or manifest.get("layer") != layer
        or manifest.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or manifest.get("fresh_targets_opened") is not False
        or manifest.get("qualification_authority") is not False
    ):
        raise RuntimeError(f"{layer} finalist manifest is invalid")
    identity = manifest["selected_artifact"]
    artifact = root / "artifacts" / identity["name"]
    if (
        not artifact.is_file()
        or artifact.stat().st_size != identity["bytes"]
        or sha256_file(artifact) != identity["sha256"]
    ):
        raise RuntimeError(f"{layer} finalist artifact identity changed")
    wrapper = joblib.load(artifact)
    if (
        wrapper.get("schema_version") != "stage6_expanded_predictive_layer_artifact_v1"
        or wrapper.get("layer") != layer
        or wrapper.get("finalist_run_id") != finalist_run_id(layer)
        or wrapper.get("recommendation_authority") is not False
        or wrapper.get("action_authority") is not False
        or wrapper.get("live_action_authority") is not False
        or wrapper.get("action_timing_layer_included") is not False
    ):
        raise RuntimeError(f"{layer} finalist artifact authority is invalid")
    return manifest, artifact


def _materialization_inventory(stage_drive_root: Path) -> list[str]:
    parent = stage_drive_root / "runtime_state" / "notebook01" / "materializations"
    if not parent.is_dir():
        return []
    return sorted(path.name for path in parent.iterdir() if path.is_dir())


def _load_custody_request(root: Path, aggregate_manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "fresh-custody-request.json"
    if not path.is_file():
        raise RuntimeError("fresh custody request is missing")
    request = json.loads(path.read_text(encoding="utf-8"))
    validate_document(request)
    if (
        request.get("schema_version") != "stage6_expanded_fresh_custody_request_v1"
        or request.get("aggregate_run_id") != aggregate_run_id()
        or request.get("aggregate_freeze_manifest_self_sha256")
        != aggregate_manifest["self_sha256"]
        or request.get("one_shot_adjudication") is not True
    ):
        raise RuntimeError("fresh custody request is not parent-bound")
    return request


def run_aggregate_freeze(stage_drive_root: Path) -> dict[str, Any]:
    root = aggregate_work_root(stage_drive_root)
    path = root / "aggregate-freeze-manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_document(existing)
        if (
            existing.get("schema_version") != "stage6_expanded_aggregate_freeze_manifest_v1"
            or existing.get("status") != "PASS_CANDIDATES_FROZEN_AWAITING_FRESH_DATA"
            or existing.get("aggregate_run_id") != aggregate_run_id()
            or existing.get("science_implementation_self_sha256")
            != SCIENCE_IMPLEMENTATION_DIGEST
            or existing.get("fresh_targets_opened") is not False
        ):
            raise RuntimeError("existing aggregate freeze is invalid")
        custody = _load_custody_request(root, existing)
        return {
            "status": "PASS",
            "aggregate_run_id": aggregate_run_id(),
            "manifest_path": str(path),
            "manifest_self_sha256": existing["self_sha256"],
            "fresh_custody_request_path": str(root / "fresh-custody-request.json"),
            "fresh_custody_request_self_sha256": custody["self_sha256"],
            "reused_completed_result": True,
            "fresh_targets_opened": False,
        }

    finalists = {}
    for layer in LAYERS:
        manifest, artifact = _load_finalist(stage_drive_root, layer)
        finalists[layer] = {
            "finalist_run_id": finalist_run_id(layer),
            "finalist_manifest_self_sha256": manifest["self_sha256"],
            "selected_candidate_id": manifest["selected_candidate_id"],
            "selected_artifact": manifest["selected_artifact"],
            "artifact_relative_path": artifact.relative_to(stage_drive_root).as_posix(),
        }
    freeze = seal_document({
        "schema_version": "stage6_expanded_aggregate_freeze_manifest_v1",
        "status": "PASS_CANDIDATES_FROZEN_AWAITING_FRESH_DATA",
        "aggregate_run_id": aggregate_run_id(),
        "feature_run_id": FEATURE_RUN_ID,
        "expanded_layer_contract_self_sha256": EXPANDED_LAYER_CONTRACT_DIGEST,
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "finalists": finalists,
        "fresh_cutoff_utc_inclusive": DEVELOPMENT_CUTOFF_UTC,
        "fresh_source_requirements": {
            "new_notebook01_materialization_run_required": True,
            "all_48_logical_lanes_required": True,
            "whole_hand_after_cutoff_required": True,
            "prior_adjudication_forbidden": True,
        },
        "paired_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_cluster_identity": PAIRED_CLUSTER_IDENTITY,
        "minimum_independent_game_clusters_per_claim": (
            MINIMUM_INDEPENDENT_GAME_CLUSTERS
        ),
        "paired_sign_flip_replicates": SIGN_FLIP_REPLICATES,
        "multiple_comparison_method": "holm_bonferroni",
        "claim_family": list(LAYERS),
        "family_alpha": FAMILY_ALPHA,
        "action_timing_layer_included": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    custody_path = root / "fresh-custody-request.json"
    if custody_path.is_file():
        custody = json.loads(custody_path.read_text(encoding="utf-8"))
        validate_document(custody)
        if (
            custody.get("schema_version") != "stage6_expanded_fresh_custody_request_v1"
            or custody.get("aggregate_run_id") != aggregate_run_id()
            or custody.get("aggregate_freeze_manifest_self_sha256") != freeze["self_sha256"]
        ):
            raise RuntimeError("owned pre-manifest custody request changed")
    else:
        custody = seal_document({
            "schema_version": "stage6_expanded_fresh_custody_request_v1",
            "status": "FROZEN_AWAITING_ONE_NEW_NOTEBOOK01_MATERIALIZATION",
            "aggregate_run_id": aggregate_run_id(),
            "aggregate_freeze_manifest_self_sha256": freeze["self_sha256"],
            "fresh_cutoff_utc_inclusive": DEVELOPMENT_CUTOFF_UTC,
            "preexisting_notebook01_materialization_run_ids": _materialization_inventory(
                stage_drive_root
            ),
            "one_shot_adjudication": True,
            "fresh_targets_opened": False,
        })
        _atomic_json(custody_path, custody)
    _atomic_json(path, freeze)
    return {
        "status": "PASS",
        "aggregate_run_id": aggregate_run_id(),
        "manifest_path": str(path),
        "manifest_self_sha256": freeze["self_sha256"],
        "fresh_custody_request_path": str(root / "fresh-custody-request.json"),
        "fresh_custody_request_self_sha256": custody["self_sha256"],
        "reused_completed_result": False,
        "fresh_targets_opened": False,
    }


def _validate_fresh_materialization(
    stage_drive_root: Path,
    root: Path,
    expected_fresh_custody_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = root / "stage6_notebook01_materialization_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("fresh Notebook 01 materialization manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("schema_version") != "stage6_notebook01_materialization_manifest_v1"
        or manifest.get("status")
        != "PASS_FULL_CAUSAL_DECISION_PROFILE_MATERIALIZATION_AND_SEALED_SPLIT_LOCK"
        or manifest.get("completed_lane_count") != LOGICAL_LANE_COUNT
        or manifest.get("private_lockbox_assignment_sealed") is not True
        or manifest.get("private_lockbox_opened") is not False
    ):
        raise RuntimeError("fresh Notebook 01 materialization authority is invalid")
    if manifest.get("run_identity") == NOTEBOOK01_RUN_ID:
        raise RuntimeError("development feature run cannot serve as fresh materialization")
    run_identity = str(manifest.get("run_identity", ""))
    if (
        len(run_identity) != 64
        or any(character not in "0123456789abcdef" for character in run_identity)
    ):
        raise RuntimeError("fresh materialization run identity is invalid")
    run_binding = manifest.get("run_binding")
    if (
        not isinstance(run_binding, Mapping)
        or run_binding.get("schema_version") not in {
            "stage6_notebook01_materialization_run_binding_v2",
            "stage6_notebook01_materialization_run_binding_v3",
        }
        or hashlib.sha256(canonical_json_bytes(run_binding)).hexdigest() != run_identity
    ):
        raise RuntimeError("fresh materialization run binding does not derive its identity")
    if expected_fresh_custody_binding is not None and (
        run_binding.get("schema_version")
        != "stage6_notebook01_materialization_run_binding_v3"
        or run_binding.get("fresh_custody_binding")
        != dict(expected_fresh_custody_binding)
    ):
        raise RuntimeError(
            "fresh materialization was not producer-bound to this exact aggregate custody"
        )
    canonical_root = (
        stage_drive_root
        / "runtime_state"
        / "notebook01"
        / "materializations"
        / run_identity
    )
    if root.resolve() != canonical_root.resolve() or root.name != run_identity:
        raise RuntimeError("fresh materialization is not the exact canonical run directory")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or [row.get("lane_id") for row in lanes] != [
        f"lane-{index:02d}" for index in range(LOGICAL_LANE_COUNT)
    ]:
        raise RuntimeError("fresh materialization lane inventory changed")
    paths = []
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        identity = lane.get("outputs", {}).get("accepted_decisions")
        if not isinstance(identity, Mapping):
            raise RuntimeError("fresh materialization accepted decision binding is missing")
        expected_relative_path = f"lanes/{lane_id}/accepted-decisions.parquet"
        if identity.get("relative_path") != expected_relative_path:
            raise RuntimeError("fresh accepted-decision path is not canonical")
        path = root / expected_relative_path
        if path.resolve().parent.parent.parent != root.resolve():
            raise RuntimeError("fresh accepted-decision path escapes the materialization root")
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != str(identity["sha256"])
            or pq.ParquetFile(path).metadata.num_rows != int(identity["row_count"])
        ):
            raise RuntimeError("fresh materialization accepted decision identity changed")
        paths.append(path)
    return manifest, paths


def _fresh_panel_identity(fresh_manifest: Mapping[str, Any]) -> str:
    accepted_root = str(fresh_manifest.get("accepted_row_identity_root", ""))
    if (
        len(accepted_root) != 64
        or any(character not in "0123456789abcdef" for character in accepted_root)
    ):
        raise RuntimeError("fresh materialization accepted-row identity root is invalid")
    lane_payloads = []
    for lane in fresh_manifest["lanes"]:
        identity = lane["outputs"]["accepted_decisions"]
        lane_payloads.append({
            "lane_id": lane["lane_id"],
            "row_count": int(identity["row_count"]),
            "sha256": str(identity["sha256"]),
        })
    return hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_fresh_panel_identity_v1",
        "accepted_row_identity_root": accepted_root,
        "accepted_decision_row_count": int(
            fresh_manifest["accepted_decision_row_count"]
        ),
        "lane_payloads": lane_payloads,
    })).hexdigest()


def _claim_one_shot_fresh_materialization(
    stage_drive_root: Path,
    aggregate_manifest: Mapping[str, Any],
    custody_request: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fresh_run_identity = str(fresh_manifest["run_identity"])
    fresh_panel_identity = _fresh_panel_identity(fresh_manifest)
    if fresh_run_identity in set(
        custody_request["preexisting_notebook01_materialization_run_ids"]
    ):
        raise RuntimeError(
            "fresh materialization existed before the aggregate custody request"
        )
    consumption = seal_document({
        "schema_version": "stage6_expanded_fresh_materialization_consumption_v1",
        "status": "CONSUMED_BY_ONE_EXPANDED_LAYER_AGGREGATE",
        "fresh_materialization_run_identity": fresh_run_identity,
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "fresh_panel_identity": fresh_panel_identity,
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_custody_request_self_sha256": custody_request["self_sha256"],
    })
    consumption_path = (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "fresh_consumption_ledger"
        / f"{fresh_panel_identity}.json"
    )
    lease = seal_document({
        "schema_version": "stage6_expanded_fresh_one_shot_lease_v1",
        "status": "CLAIMED_FOR_SINGLE_ADJUDICATION",
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_custody_request_self_sha256": custody_request["self_sha256"],
        "fresh_materialization_run_identity": fresh_run_identity,
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "fresh_panel_identity": fresh_panel_identity,
    })
    path = aggregate_work_root(stage_drive_root) / "fresh_adjudication" / "one-shot-lease.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_document(existing)
        if existing != lease:
            raise RuntimeError(
                "aggregate fresh adjudication is already leased to a different materialization"
            )
    if consumption_path.is_file():
        existing_consumption = json.loads(
            consumption_path.read_text(encoding="utf-8")
        )
        validate_document(existing_consumption)
        if existing_consumption != consumption:
            raise RuntimeError(
                "fresh materialization was already consumed by a different aggregate"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raw = canonical_json_bytes(lease) + b"\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_document(existing)
            if existing != lease:
                raise RuntimeError(
                    "aggregate fresh adjudication was concurrently leased to a different materialization"
                )
        else:
            with os.fdopen(descriptor, "wb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
    lease_readback = json.loads(path.read_text(encoding="utf-8"))
    validate_document(lease_readback)
    if lease_readback != lease:
        raise RuntimeError("fresh one-shot lease readback changed")

    consumption_path.parent.mkdir(parents=True, exist_ok=True)
    if not consumption_path.is_file():
        consumption_raw = canonical_json_bytes(consumption) + b"\n"
        try:
            descriptor = os.open(
                consumption_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            existing_consumption = json.loads(
                consumption_path.read_text(encoding="utf-8")
            )
            validate_document(existing_consumption)
            if existing_consumption != consumption:
                raise RuntimeError(
                    "fresh materialization was concurrently consumed by a different aggregate"
                )
        else:
            with os.fdopen(descriptor, "wb") as target:
                target.write(consumption_raw)
                target.flush()
                os.fsync(target.fileno())
    consumption_readback = json.loads(
        consumption_path.read_text(encoding="utf-8")
    )
    validate_document(consumption_readback)
    if consumption_readback != consumption:
        raise RuntimeError("fresh materialization consumption readback changed")
    return lease_readback, consumption_readback


def _plan_disjoint_fresh_rows(
    stage_drive_root: Path,
    paths: Sequence[Path],
    aggregate_manifest: Mapping[str, Any],
    custody_request: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str], Path, bool]:
    frames = []
    for lane, path in enumerate(paths):
        frame = pq.read_table(
            path,
            columns=["row_id", "game_id", "hand_id", "decision_utc"],
        ).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
    identity = pd.concat(frames, ignore_index=True)
    identity["row_id"] = identity["row_id"].astype(str)
    identity["game_id"] = identity["game_id"].astype(str)
    identity["hand_id"] = identity["hand_id"].astype(str)
    identity["decision_ns"] = pd.to_datetime(
        identity["decision_utc"], utc=True, errors="raise"
    ).astype("int64")
    if identity["row_id"].duplicated().any():
        raise RuntimeError("fresh preflight duplicates row_id")
    cutoff_ns = int(pd.Timestamp(DEVELOPMENT_CUTOFF_UTC).value)
    claims_root = (
        stage_drive_root
        / "runtime_state"
        / "expanded_layers"
        / "fresh_row_panel_claims"
    )
    previous_key = FRESH_PANEL_GENESIS_KEY
    previous_max_ns = cutoff_ns - 1
    consumed_row_ids: set[str] = set()
    current_identity_by_row = {
        str(row.row_id): [
            str(row.game_id),
            str(row.hand_id),
            str(row.row_id),
            int(row.decision_ns),
        ]
        for row in identity.itertuples(index=False)
    }
    while True:
        claim_path = claims_root / f"{previous_key}.json"
        if not claim_path.is_file():
            break
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        validate_document(existing)
        raw_records = existing.get("row_identity_records")
        if not isinstance(raw_records, list):
            raise RuntimeError("global fresh row-panel claim records are missing")
        records = []
        for record in raw_records:
            if not isinstance(record, list) or len(record) != 4:
                raise RuntimeError("global fresh row-panel claim record is invalid")
            records.append([
                str(record[0]),
                str(record[1]),
                str(record[2]),
                int(record[3]),
            ])
        row_ids = [record[2] for record in records]
        if (
            existing.get("schema_version") != "stage6_expanded_disjoint_fresh_row_panel_claim_v1"
            or existing.get("status") != "ATOMICALLY_CLAIMED_BEFORE_TARGET_OPEN"
            or existing.get("previous_claim_key") != previous_key
            or existing.get("target_columns_opened_before_claim") is not False
            or int(existing.get("row_count")) != len(records)
            or not records
            or len(set(row_ids)) != len(row_ids)
            or existing.get("row_ids") != row_ids
            or hashlib.sha256(canonical_json_bytes(records)).hexdigest()
            != existing.get("row_identity_root_sha256")
            or int(existing.get("panel_min_decision_ns"))
            != min(record[3] for record in records)
            or int(existing.get("panel_max_decision_ns"))
            != max(record[3] for record in records)
            or int(existing.get("panel_min_decision_ns")) <= previous_max_ns
            or int(existing.get("panel_max_decision_ns"))
            < int(existing.get("panel_min_decision_ns"))
        ):
            raise RuntimeError("global fresh row-panel claim chain is invalid")
        if consumed_row_ids.intersection(row_ids):
            raise RuntimeError("global fresh row-panel claim chain reuses row_id")
        same_aggregate = (
            existing.get("aggregate_freeze_manifest_self_sha256")
            == aggregate_manifest["self_sha256"]
        )
        same_materialization = (
            existing.get("fresh_materialization_manifest_self_sha256")
            == fresh_manifest["self_sha256"]
        )
        if same_aggregate and same_materialization:
            if (
                existing.get("aggregate_run_id") != aggregate_run_id()
                or existing.get("fresh_custody_request_self_sha256")
                != custody_request["self_sha256"]
                or existing.get("fresh_materialization_run_identity")
                != fresh_manifest["run_identity"]
            ):
                raise RuntimeError("fresh row-panel resume parent binding changed")
            current_records = [current_identity_by_row.get(row_id) for row_id in row_ids]
            if current_records != records:
                raise RuntimeError(
                    "claimed fresh row-panel identities changed in the materialization"
                )
            return existing, set(row_ids), claim_path, True
        consumed_row_ids.update(row_ids)
        previous_max_ns = int(existing["panel_max_decision_ns"])
        previous_key = str(existing["self_sha256"])

    hand_times = identity.groupby(
        ["game_id", "hand_id"], sort=False, observed=True
    )["decision_ns"].agg(["min", "max"])
    crossing = (hand_times["min"] <= previous_max_ns) & (
        hand_times["max"] > previous_max_ns
    )
    if crossing.any():
        raise RuntimeError("a hand crosses the next disjoint fresh panel boundary")
    eligible_hands = hand_times.index[hand_times["min"] > previous_max_ns]
    eligible_index = pd.MultiIndex.from_frame(
        identity.loc[:, ["game_id", "hand_id"]]
    ).isin(eligible_hands)
    panel = identity.loc[eligible_index].copy()
    if panel.empty:
        raise RuntimeError("no new disjoint fresh rows are available")
    panel = panel.sort_values(
        ["decision_ns", "game_id", "hand_id", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    records = [
        [str(row.game_id), str(row.hand_id), str(row.row_id), int(row.decision_ns)]
        for row in panel.itertuples(index=False)
    ]
    row_ids = [record[2] for record in records]
    if consumed_row_ids.intersection(row_ids):
        raise RuntimeError("new fresh row panel reuses a previously claimed row_id")
    claim = seal_document({
        "schema_version": "stage6_expanded_disjoint_fresh_row_panel_claim_v1",
        "status": "ATOMICALLY_CLAIMED_BEFORE_TARGET_OPEN",
        "previous_claim_key": previous_key,
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_custody_request_self_sha256": custody_request["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "panel_min_decision_ns": int(panel["decision_ns"].min()),
        "panel_max_decision_ns": int(panel["decision_ns"].max()),
        "row_count": len(row_ids),
        "row_identity_root_sha256": hashlib.sha256(
            canonical_json_bytes(records)
        ).hexdigest(),
        "row_identity_records": records,
        "row_ids": row_ids,
        "target_columns_opened_before_claim": False,
    })
    claim_path = claims_root / f"{previous_key}.json"
    return claim, set(row_ids), claim_path, False


def _commit_disjoint_fresh_row_claim(
    claim: Mapping[str, Any],
    claim_path: Path,
    *,
    already_committed: bool,
) -> dict[str, Any]:
    if already_committed:
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        validate_document(existing)
        if existing != claim:
            raise RuntimeError("existing fresh row-panel claim changed before use")
        return existing
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(claim) + b"\n"
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        validate_document(existing)
        if existing != claim:
            raise RuntimeError("another fresh row panel claimed this chain head")
    else:
        with os.fdopen(descriptor, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
    readback = json.loads(claim_path.read_text(encoding="utf-8"))
    validate_document(readback)
    if readback != claim:
        raise RuntimeError("fresh row-panel claim readback changed")
    return readback


def _load_fresh_source(paths: Sequence[Path]) -> pd.DataFrame:
    columns = list(dict.fromkeys((*FEATURE_SOURCE_COLUMNS, *MODEL_SOURCE_COLUMNS)))
    frames = []
    for lane, path in enumerate(paths):
        frame = pq.read_table(path, columns=columns).to_pandas()
        frame["source_lane_id"] = f"lane-{lane:02d}"
        frames.append(frame)
        print(json.dumps({
            "event": "stage6_expanded_fresh_lane_opened",
            "lane": lane,
            "completed": lane + 1,
            "total": LOGICAL_LANE_COUNT,
            "rows": len(frame),
        }, sort_keys=True), flush=True)
    source = pd.concat(frames, ignore_index=True)
    if source["row_id"].astype(str).duplicated().any():
        raise RuntimeError("fresh source duplicates row_id")
    hand_times = source.groupby(["game_id", "hand_id"], sort=False, observed=True)[
        "decision_utc"
    ].agg(["min", "max"])
    crossing = (hand_times["min"].astype(str) < DEVELOPMENT_CUTOFF_UTC) & (
        hand_times["max"].astype(str) >= DEVELOPMENT_CUTOFF_UTC
    )
    if crossing.any():
        raise RuntimeError("a hand crosses the fresh adjudication cutoff")
    if not source["decision_utc"].astype(str).ge(DEVELOPMENT_CUTOFF_UTC).any():
        raise RuntimeError("new materialization contains no fresh adjudication rows")
    return source


def _predict_artifact(
    artifact: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    selected_names, selected_x = expanded_model_matrix(feature_frame, layer)
    if selected_names != list(artifact["selected_feature_names"]):
        raise RuntimeError(f"{layer} selected feature schema changed")
    if layer == "hero_opponent_interaction":
        control_names, control_x = expanded_model_matrix(
            feature_frame, layer, include_dyad_interaction=False
        )
    else:
        control_names = list(BASE_FEATURE_COLUMNS)
        control_x = feature_frame[control_names].to_numpy(dtype=np.float32)
    if control_names != list(artifact["control_feature_names"]):
        raise RuntimeError(f"{layer} control feature schema changed")
    selected_x = _apply_imputer(
        selected_x, np.asarray(artifact["selected_imputer_medians"], dtype=np.float32)
    )
    control_x = _apply_imputer(
        control_x, np.asarray(artifact["control_imputer_medians"], dtype=np.float32)
    )
    legal_indices = [selected_names.index(f"base_legal_{action}") for action in ACTIONS]
    legal = selected_x[:, legal_indices].astype(bool)
    priors = np.asarray(artifact["class_priors"], dtype=np.float64)
    selected = _aligned_probabilities(artifact["selected_model"], selected_x)
    selected, _, invalid_selected = _project_legal(selected, legal, priors)
    selected = _temperature_scale(selected, float(artifact["selected_temperature"]))
    selected, _, invalid_selected_after = _project_legal(selected, legal, priors)
    control = _aligned_probabilities(artifact["control_model"], control_x)
    control, _, invalid_control = _project_legal(control, legal, priors)
    control = _temperature_scale(control, float(artifact["control_temperature"]))
    control, _, invalid_control_after = _project_legal(control, legal, priors)
    invalid = np.asarray([
        invalid_selected,
        invalid_selected_after,
        invalid_control,
        invalid_control_after,
    ], dtype=np.int64)
    return selected, control, invalid, selected_names, control_names


def _paired_cluster_inference(
    differences: np.ndarray,
    clusters: Sequence[str],
    seed: int,
    minimum_cluster_count: int = MINIMUM_INDEPENDENT_GAME_CLUSTERS,
) -> dict[str, Any]:
    frame = pd.DataFrame({"difference": differences, "cluster": list(clusters)})
    grouped = frame.groupby("cluster", sort=True, observed=True)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    if not len(sums):
        raise RuntimeError("paired cluster inference has no clusters")
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(sums), size=len(sums))
        bootstrap[index] = sums[draw].sum() / counts[draw].sum()
    null = np.empty(SIGN_FLIP_REPLICATES, dtype=np.float64)
    for index in range(SIGN_FLIP_REPLICATES):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(sums))
        null[index] = float(np.sum(sums * signs) / np.sum(counts))
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return {
        "row_count": len(differences),
        "cluster_count": len(sums),
        "cluster_identity": PAIRED_CLUSTER_IDENTITY,
        "minimum_independent_cluster_count": int(minimum_cluster_count),
        "independent_cluster_gate_met": len(sums) >= int(minimum_cluster_count),
        "mean_candidate_minus_control_log_loss": observed,
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "paired_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_sign_flip_two_sided_p_value": p_value,
        "paired_sign_flip_replicates": SIGN_FLIP_REPLICATES,
    }


def _fresh_stratum_report(
    target: np.ndarray,
    selected: np.ndarray,
    control: np.ndarray,
    served: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return {
            "row_count": 0,
            "selected_metrics": None,
            "matched_control_metrics": None,
            "served_metrics": None,
        }
    truth = [ACTIONS[int(value)] for value in target[positions]]
    return {
        "row_count": len(positions),
        "selected_metrics": multiclass_metric_report(truth, selected[positions], ACTIONS),
        "matched_control_metrics": multiclass_metric_report(truth, control[positions], ACTIONS),
        "served_metrics": multiclass_metric_report(truth, served[positions], ACTIONS),
    }


def _holm(results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        results,
        key=lambda layer: (
            float(results[layer]["inference"]["paired_sign_flip_two_sided_p_value"]),
            layer,
        ),
    )
    output = {}
    continue_rejecting = True
    family_size = len(ordered)
    for rank, layer in enumerate(ordered, start=1):
        threshold = FAMILY_ALPHA / (family_size - rank + 1)
        p_value = float(results[layer]["inference"]["paired_sign_flip_two_sided_p_value"])
        reject = continue_rejecting and p_value <= threshold
        if not reject:
            continue_rejecting = False
        output[layer] = {
            "holm_rank": rank,
            "raw_p_value": p_value,
            "holm_threshold": threshold,
            "reject_no_improvement_null": reject,
        }
    return output


def run_fresh_adjudication(
    stage_drive_root: Path,
    fresh_materialization_root: Path,
) -> dict[str, Any]:
    aggregate = run_aggregate_freeze(stage_drive_root)
    aggregate_manifest = json.loads(Path(aggregate["manifest_path"]).read_text(encoding="utf-8"))
    custody_request = _load_custody_request(
        aggregate_work_root(stage_drive_root), aggregate_manifest
    )
    expected_fresh_custody_binding = {
        "schema_version": "stage6_notebook01_fresh_custody_binding_v1",
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_custody_request_self_sha256": custody_request["self_sha256"],
    }
    fresh_manifest, paths = _validate_fresh_materialization(
        stage_drive_root,
        fresh_materialization_root,
        expected_fresh_custody_binding,
    )
    (
        planned_fresh_row_panel_claim,
        claimed_fresh_row_ids,
        fresh_row_panel_claim_path,
        fresh_row_panel_already_committed,
    ) = _plan_disjoint_fresh_rows(
            stage_drive_root,
            paths,
            aggregate_manifest,
            custody_request,
            fresh_manifest,
    )
    one_shot_lease, materialization_consumption = _claim_one_shot_fresh_materialization(
        stage_drive_root,
        aggregate_manifest,
        custody_request,
        fresh_manifest,
    )
    fresh_row_panel_claim = _commit_disjoint_fresh_row_claim(
        planned_fresh_row_panel_claim,
        fresh_row_panel_claim_path,
        already_committed=fresh_row_panel_already_committed,
    )
    fresh_adjudication_run_id = hashlib.sha256(canonical_json_bytes({
        "schema": "stage6_expanded_fresh_adjudication_v1",
        "aggregate_run_id": aggregate_run_id(),
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "fresh_custody_request_self_sha256": custody_request["self_sha256"],
        "fresh_one_shot_lease_self_sha256": one_shot_lease["self_sha256"],
        "fresh_materialization_consumption_self_sha256": materialization_consumption[
            "self_sha256"
        ],
        "fresh_row_panel_claim_self_sha256": fresh_row_panel_claim["self_sha256"],
        "fresh_row_panel_identity_root_sha256": fresh_row_panel_claim[
            "row_identity_root_sha256"
        ],
        "fresh_row_panel_row_count": fresh_row_panel_claim["row_count"],
    })).hexdigest()
    result_root = (
        aggregate_work_root(stage_drive_root)
        / "fresh_adjudication"
        / fresh_adjudication_run_id
    )
    completion_path = result_root / "fresh-adjudication-manifest.json"
    if completion_path.is_file():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        validate_document(existing)
        if (
            existing.get("schema_version") != "stage6_expanded_fresh_adjudication_manifest_v1"
            or existing.get("fresh_adjudication_run_id") != fresh_adjudication_run_id
            or existing.get("aggregate_run_id") != aggregate_run_id()
            or existing.get("science_implementation_self_sha256")
            != SCIENCE_IMPLEMENTATION_DIGEST
            or existing.get("aggregate_freeze_manifest_self_sha256")
            != aggregate_manifest["self_sha256"]
            or existing.get("fresh_materialization_run_identity")
            != fresh_manifest["run_identity"]
            or existing.get("fresh_materialization_manifest_self_sha256")
            != fresh_manifest["self_sha256"]
            or existing.get("fresh_row_panel_claim_self_sha256")
            != fresh_row_panel_claim["self_sha256"]
            or existing.get("fresh_row_panel_identity_root_sha256")
            != fresh_row_panel_claim["row_identity_root_sha256"]
            or existing.get("fresh_row_count") != fresh_row_panel_claim["row_count"]
        ):
            raise RuntimeError("completed fresh adjudication is not bound to this input")
        return {
            "status": existing["status"],
            "fresh_adjudication_run_id": fresh_adjudication_run_id,
            "manifest_path": str(completion_path),
            "manifest_self_sha256": existing["self_sha256"],
            "layer_qualification": existing["layer_qualification"],
            "reused_completed_result": True,
        }

    started = time.monotonic()
    source = _load_fresh_source(paths)
    common, line, dyad = build_feature_frames(
        source, allow_fresh_adjudication=True
    )
    common_row_ids = common["row_id"].astype(str)
    fresh_common = common.loc[common_row_ids.isin(claimed_fresh_row_ids)].copy()
    fresh_ids = set(fresh_common["row_id"].astype(str))
    if (
        fresh_common["row_id"].astype(str).duplicated().any()
        or fresh_ids != claimed_fresh_row_ids
        or len(fresh_common) != int(fresh_row_panel_claim["row_count"])
    ):
        raise RuntimeError("fresh adjudication rows do not exactly match the pre-target claim")
    claims: dict[str, dict[str, Any]] = {}
    for layer, derived in (
        ("hero_opponent_interaction", dyad),
        ("within_hand_line_surprise", line),
    ):
        frame = fresh_common.merge(
            derived.loc[derived["row_id"].astype(str).isin(fresh_ids)],
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
        if layer == "hero_opponent_interaction":
            frame = frame.loc[frame["dyad_eligible"].eq(True)].copy()
        artifact_reference = aggregate_manifest["finalists"][layer]
        artifact_path = stage_drive_root / artifact_reference["artifact_relative_path"]
        if (
            artifact_path.stat().st_size != artifact_reference["selected_artifact"]["bytes"]
            or sha256_file(artifact_path) != artifact_reference["selected_artifact"]["sha256"]
        ):
            raise RuntimeError(f"{layer} frozen artifact changed before adjudication")
        artifact = joblib.load(artifact_path)
        selected, control, invalid, _, _ = _predict_artifact(artifact, frame, layer)
        target = np.asarray([ACTIONS.index(value) for value in frame["action_label"].astype(str)], dtype=np.int8)
        if layer == "hero_opponent_interaction":
            claim_mask = frame["dyad_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64) >= DYAD_MINIMUM_HAND_SUPPORT
            unseen_mask = (
                frame["dyad_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64) == 0
            ) & (
                frame["main_actor_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64) > 0
            ) & (
                frame["main_counterpart_prior_hand_count"].fillna(0).to_numpy(dtype=np.float64) > 0
            )
        else:
            claim_mask = frame["line_prefix_decision_count"].to_numpy(dtype=np.int64) >= LINE_MINIMUM_PREFIX_ACTIONS
            unseen_mask = np.zeros(len(frame), dtype=bool)
        if not claim_mask.any():
            raise RuntimeError(f"{layer} fresh claim stratum is empty")
        abstain_mask = ~claim_mask
        served = selected.copy()
        served[abstain_mask] = control[abstain_mask]
        serving_exact_control = bool(np.array_equal(
            served[abstain_mask], control[abstain_mask]
        ))
        if not serving_exact_control:
            raise RuntimeError(f"{layer} abstention path did not return matched control")
        positions = np.flatnonzero(claim_mask)
        truth = [ACTIONS[int(value)] for value in target[positions]]
        selected_metrics = multiclass_metric_report(truth, selected[positions], ACTIONS)
        control_metrics = multiclass_metric_report(truth, control[positions], ACTIONS)
        selected_loss = -np.log(np.clip(
            selected[positions, target[positions]], 1e-15, 1.0
        ))
        control_loss = -np.log(np.clip(
            control[positions, target[positions]], 1e-15, 1.0
        ))
        clusters = frame.iloc[positions]["game_id"].astype(str).tolist()
        inference = _paired_cluster_inference(
            selected_loss - control_loss,
            clusters,
            int(hashlib.sha256(layer.encode("utf-8")).hexdigest()[:8], 16),
        )
        claims[layer] = {
            "claim_row_count": int(claim_mask.sum()),
            "selected_metrics": selected_metrics,
            "matched_control_metrics": control_metrics,
            "inference": inference,
            "illegal_postprojection_count": int(invalid.sum()),
            "below_support_or_prefix": {
                **_fresh_stratum_report(
                    target, selected, control, served, abstain_mask
                ),
                "abstention_count": int(abstain_mask.sum()),
                "served_probabilities_exactly_match_control": serving_exact_control,
            },
            "known_directed_dyad_row_count": int(claim_mask.sum()) if layer == "hero_opponent_interaction" else None,
            "unseen_directed_dyad_with_known_players_row_count": int(unseen_mask.sum()) if layer == "hero_opponent_interaction" else None,
            "qualification_strata": ({
                "known_directed_dyad": _fresh_stratum_report(
                    target, selected, control, served, claim_mask
                ),
                "unseen_directed_dyad_with_known_players": {
                    **_fresh_stratum_report(
                        target, selected, control, served, unseen_mask
                    ),
                    "abstention_count": int(unseen_mask.sum()),
                    "served_probabilities_exactly_match_control": bool(np.array_equal(
                        served[unseen_mask], control[unseen_mask]
                    )),
                },
            } if layer == "hero_opponent_interaction" else None),
        }

    holm = _holm(claims)
    qualification = {}
    for layer in LAYERS:
        claim = claims[layer]
        inference = claim["inference"]
        passed = (
            inference["independent_cluster_gate_met"]
            and inference["mean_candidate_minus_control_log_loss"] < 0
            and inference["confidence_interval_95"][1] < 0
            and holm[layer]["reject_no_improvement_null"]
            and claim["selected_metrics"]["multiclass_brier_score"]
            <= claim["matched_control_metrics"]["multiclass_brier_score"]
            and claim["illegal_postprojection_count"] == 0
            and claim["below_support_or_prefix"][
                "served_probabilities_exactly_match_control"
            ]
            and (
                layer != "hero_opponent_interaction"
                or claim["qualification_strata"][
                    "unseen_directed_dyad_with_known_players"
                ]["served_probabilities_exactly_match_control"]
            )
        )
        qualification[layer] = {
            "qualified": bool(passed),
            "holm": holm[layer],
            "reason": (
                "PASS_SIGNIFICANT_PAIRED_PROPER_SCORE_IMPROVEMENT"
                if passed
                else "HOLD_FRESH_GATE_NOT_MET"
            ),
        }
    manifest = seal_document({
        "schema_version": "stage6_expanded_fresh_adjudication_manifest_v1",
        "status": (
            "PASS_FRESH_ADJUDICATION_COMPLETE"
            if all(row["qualified"] for row in qualification.values())
            else "HOLD_FRESH_ADJUDICATION_COMPLETE"
        ),
        "fresh_adjudication_run_id": fresh_adjudication_run_id,
        "aggregate_run_id": aggregate_run_id(),
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "aggregate_freeze_manifest_self_sha256": aggregate_manifest["self_sha256"],
        "fresh_materialization_run_identity": fresh_manifest["run_identity"],
        "fresh_materialization_manifest_self_sha256": fresh_manifest["self_sha256"],
        "fresh_row_panel_claim_self_sha256": fresh_row_panel_claim["self_sha256"],
        "fresh_row_panel_identity_root_sha256": fresh_row_panel_claim[
            "row_identity_root_sha256"
        ],
        "fresh_row_panel_claim_status": fresh_row_panel_claim["status"],
        "fresh_cutoff_utc_inclusive": DEVELOPMENT_CUTOFF_UTC,
        "fresh_row_count": len(fresh_common),
        "claims": claims,
        "holm_bonferroni": holm,
        "layer_qualification": qualification,
        "action_timing_layer_included": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
        "assistant_installation_allowed_by_layer": {
            layer: bool(row["qualified"]) for layer, row in qualification.items()
        },
        "elapsed_seconds": time.monotonic() - started,
    })
    _atomic_json(completion_path, manifest)
    return {
        "status": manifest["status"],
        "fresh_adjudication_run_id": fresh_adjudication_run_id,
        "manifest_path": str(completion_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "layer_qualification": qualification,
        "reused_completed_result": False,
    }
