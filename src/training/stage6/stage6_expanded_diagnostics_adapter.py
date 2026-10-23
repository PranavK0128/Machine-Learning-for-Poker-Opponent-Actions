"""Read-only Assistant diagnostics adapter for freshly qualified Stage 6 layers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from stage6_expanded_adjudication import aggregate_run_id, aggregate_work_root, _predict_artifact
from stage6_expanded_finalist import DYAD_MINIMUM_HAND_SUPPORT, LINE_MINIMUM_PREFIX_ACTIONS
from stage6_expanded_layers import (
    ACTIONS,
    SCIENCE_IMPLEMENTATION_DIGEST,
    seal_document,
    validate_document,
)
from stage6_notebooks02_07_runtime import sha256_file


QUALIFIED_EXPANDED_DIAGNOSTICS_FOLDER = "Stage 6 Qualified Expanded Diagnostics"


def _load_qualified_artifact(
    stage_drive_root: Path,
    fresh_manifest_path: Path,
    layer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fresh = json.loads(fresh_manifest_path.read_text(encoding="utf-8"))
    validate_document(fresh)
    if (
        fresh.get("schema_version") != "stage6_expanded_fresh_adjudication_manifest_v1"
        or fresh.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or fresh.get("layer_qualification", {}).get(layer, {}).get("qualified") is not True
        or fresh.get("assistant_installation_allowed_by_layer", {}).get(layer) is not True
        or fresh.get("recommendation_authority") is not False
        or fresh.get("action_authority") is not False
        or fresh.get("live_action_authority") is not False
    ):
        raise RuntimeError(f"{layer} lacks fresh diagnostics qualification authority")
    aggregate_path = aggregate_work_root(stage_drive_root) / "aggregate-freeze-manifest.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_document(aggregate)
    if (
        aggregate.get("aggregate_run_id") != aggregate_run_id()
        or aggregate.get("self_sha256")
        != fresh.get("aggregate_freeze_manifest_self_sha256")
        or aggregate.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
    ):
        raise RuntimeError("qualified diagnostics aggregate binding changed")
    reference = aggregate["finalists"][layer]
    artifact_path = stage_drive_root / reference["artifact_relative_path"]
    identity = reference["selected_artifact"]
    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size != int(identity["bytes"])
        or sha256_file(artifact_path) != str(identity["sha256"])
    ):
        raise RuntimeError(f"{layer} qualified artifact bytes changed")
    artifact = joblib.load(artifact_path)
    if (
        artifact.get("schema_version") != "stage6_expanded_predictive_layer_artifact_v1"
        or artifact.get("layer") != layer
        or artifact.get("science_implementation_self_sha256")
        != SCIENCE_IMPLEMENTATION_DIGEST
        or artifact.get("recommendation_authority") is not False
        or artifact.get("action_authority") is not False
        or artifact.get("live_action_authority") is not False
    ):
        raise RuntimeError(f"{layer} artifact authority changed")
    return artifact, fresh


def diagnostic_rows(
    artifact: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    layer: str,
    *,
    observed_actions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected, control, invalid, _, _ = _predict_artifact(artifact, feature_frame, layer)
    if int(invalid.sum()):
        raise RuntimeError("qualified diagnostics produced invalid projected probabilities")
    if layer == "hero_opponent_interaction":
        support = feature_frame["dyad_prior_hand_count"].fillna(0).to_numpy(dtype=np.int64)
        enabled = support >= DYAD_MINIMUM_HAND_SUPPORT
    elif layer == "within_hand_line_surprise":
        support = feature_frame["line_prefix_decision_count"].fillna(0).to_numpy(dtype=np.int64)
        enabled = support >= LINE_MINIMUM_PREFIX_ACTIONS
    else:
        raise ValueError(f"unknown expanded diagnostics layer: {layer}")
    served = selected.copy()
    served[~enabled] = control[~enabled]
    if not np.array_equal(served[~enabled], control[~enabled]):
        raise RuntimeError("qualified diagnostics abstention did not return matched control")
    if observed_actions is not None and len(observed_actions) != len(feature_frame):
        raise ValueError("observed action count does not match diagnostics rows")
    output = []
    for index in range(len(feature_frame)):
        prediction_index = int(np.argmax(served[index]))
        surprise = None
        if observed_actions is not None:
            action = str(observed_actions[index])
            if action not in ACTIONS:
                raise ValueError(f"unknown observed action: {action}")
            surprise = -math.log(max(float(served[index, ACTIONS.index(action)]), 1e-15))
        output.append({
            "layer": layer,
            "diagnostic_prediction": ACTIONS[prediction_index],
            "diagnostic_confidence": float(served[index, prediction_index]),
            "diagnostic_surprise": surprise,
            "support": int(support[index]),
            "abstained": not bool(enabled[index]),
            "probabilities": {
                action: float(served[index, action_index])
                for action_index, action in enumerate(ACTIONS)
            },
            "recommendation_authority": False,
            "action_authority": False,
            "live_action_authority": False,
        })
    return output


def qualified_diagnostics_disposition(
    stage_drive_root: Path,
    fresh_manifest_path: Path,
) -> dict[str, Any]:
    layers = {}
    fresh_identity = {
        "bytes": fresh_manifest_path.stat().st_size,
        "sha256": sha256_file(fresh_manifest_path),
    }
    for layer in ("hero_opponent_interaction", "within_hand_line_surprise"):
        try:
            artifact, fresh = _load_qualified_artifact(
                stage_drive_root, fresh_manifest_path, layer
            )
        except RuntimeError:
            continue
        reference = json.loads(
            (aggregate_work_root(stage_drive_root) / "aggregate-freeze-manifest.json").read_text(
                encoding="utf-8"
            )
        )["finalists"][layer]
        layers[layer] = {
            "qualified": True,
            "artifact": reference["selected_artifact"],
            "artifact_relative_path": reference["artifact_relative_path"],
            "candidate_id": artifact["candidate_id"],
            "family": artifact["family"],
            "fresh_qualification": fresh["layer_qualification"][layer],
        }
    if not layers:
        raise RuntimeError("no expanded diagnostics layer is freshly qualified")
    return seal_document({
        "schema_version": "stage6_qualified_expanded_diagnostics_disposition_v1",
        "status": "QUALIFIED_DIAGNOSTICS_ONLY",
        "science_implementation_self_sha256": SCIENCE_IMPLEMENTATION_DIGEST,
        "fresh_adjudication_manifest": fresh_identity,
        "layers": layers,
        "authority": ["prediction", "confidence", "abstention", "review"],
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
        "action_timing_layer_included": False,
    })
