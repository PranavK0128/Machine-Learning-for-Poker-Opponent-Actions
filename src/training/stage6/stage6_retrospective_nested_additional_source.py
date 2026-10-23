"""Materialize leakage-safe raw additional-layer arrays for nested evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
LOGICAL_DATA_LANES = 48
SOURCE_MANIFEST_NAME = "RETROSPECTIVE_NESTED_ADDITIONAL_SOURCE_MANIFEST.json"
LAYERS = (
    "player_strategy_drift",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "showdown_aware_range_calibration",
    "joint_action_size_surprise",
    "opponent_bluff_likelihood",
    "supported_fold_response_bluff_opportunity",
)
ADDITIONAL_BASE_PACKAGE = {
    "name": "POKERNOW_ASSISTANT_STAGE6_ADDITIONAL_EXPLOITATION_RELEASE_R21.zip",
    "bytes": 79_218,
    "sha256": "c4e74aa7842d58fdd5b127be9dffff0c5b021a1b843ca7ef92cebd3dcf9c4888",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("self_sha256", None)
    document["self_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def validate_document(value: Mapping[str, Any]) -> None:
    document = dict(value)
    expected = str(document.pop("self_sha256", ""))
    observed = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    if not expected or expected != observed:
        raise RuntimeError("retrospective nested additional source seal changed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npy_bytes(values: np.ndarray) -> bytes:
    import io

    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _member_identity(path: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _feature_names(frame: pd.DataFrame) -> list[str]:
    from stage6_additional_exploitation_execution import _validate_feature_names

    numeric = [
        column
        for column in frame.columns
        if column != "row_id"
        and (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
        and not any(
            token in column.lower()
            for token in (
                "authority",
                "truth_claimed",
                "review_only",
                "direct_action",
                "causal_counterfactual",
                "selection_mechanism_modelled",
                "observed_association_only",
            )
        )
    ]
    return _validate_feature_names(numeric)


def write_retrospective_nested_additional_source_from_arrays(
    output_root: str | Path,
    layers: Mapping[str, Mapping[str, Any]],
    *,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write exact raw matrices; NaN is preserved for fold-local imputation."""
    from stage6_additional_exploitation_execution import (
        _create_or_verify,
        _strict_legal_mask,
    )

    if list(layers) != list(LAYERS):
        raise RuntimeError("nested additional source layer order changed")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    layer_documents = []
    for layer in LAYERS:
        value = layers[layer]
        names = list(value["feature_names"])
        x_raw = np.asarray(value["x_raw"], dtype=np.float64)
        y = np.asarray(value["y"], dtype=np.int8)
        legal = _strict_legal_mask(
            value["legal"], label=f"nested raw legal mask for {layer}"
        )
        row_ids = np.asarray(value["row_ids"], dtype="U128")
        n = len(row_ids)
        if (
            x_raw.shape != (n, len(names))
            or y.shape != (n,)
            or legal.shape != (n, len(ACTIONS))
            or len(set(row_ids.tolist())) != n
            or np.isinf(x_raw).any()
            or not np.isin(y, range(len(ACTIONS))).all()
            or not legal[np.arange(n), y.astype(int)].all()
        ):
            raise RuntimeError(f"invalid nested raw additional arrays: {layer}")
        lanes = np.asarray([
            int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16)
            % LOGICAL_DATA_LANES
            for value in row_ids
        ], dtype=np.uint8)
        members = {
            "x_raw.npy": _npy_bytes(x_raw),
            "y.npy": _npy_bytes(y),
            "legal.npy": _npy_bytes(legal),
            "row_ids.npy": _npy_bytes(row_ids),
            "lanes.npy": _npy_bytes(lanes),
        }
        identities = []
        for name, raw in members.items():
            relative = f"layers/{layer}/{name}"
            _create_or_verify(root / relative, raw)
            identities.append(_member_identity(relative, raw))
        layer_documents.append({
            "layer": layer,
            "row_count": n,
            "feature_count": len(names),
            "feature_names": names,
            "members": identities,
            "missing_value_count": int(np.isnan(x_raw).sum()),
            "raw_nan_preserved_for_fold_local_imputation": True,
        })
    manifest = seal_document({
        "schema_version": "stage6_retrospective_nested_additional_source_v1",
        "status": "PASS_RAW_CAUSAL_ADDITIONAL_SOURCE_READY",
        "source_binding": dict(source_binding),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "additional_base_package": ADDITIONAL_BASE_PACKAGE,
        "layers": layer_documents,
        "logical_data_lanes": LOGICAL_DATA_LANES,
        "imputation_scope": "none_at_materialization__inner_train_per_fold_only",
        "current_action_used_as_feature": False,
        "action_or_decision_timing_feature": False,
        "raw_player_identity_used_as_feature": False,
        "qualification_authority": False,
    })
    _create_or_verify(
        root / SOURCE_MANIFEST_NAME,
        canonical_json_bytes(manifest) + b"\n",
        owner_run_id=str(manifest["self_sha256"]),
    )
    return manifest


def materialize_retrospective_nested_additional_source(
    decisions: pd.DataFrame,
    history: pd.DataFrame,
    legal_rows: pd.DataFrame,
    output_root: str | Path,
    *,
    source_binding: Mapping[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Build the seven existing causal feature frames without global imputation."""
    from stage6_additional_exploitation_execution import _strict_legal_mask
    from stage6_additional_exploitation_layers import (
        build_additional_exploitation_feature_frames,
    )

    legal_columns = [f"legal_{action}" for action in ACTIONS]
    required_decision = {"row_id", "game_id", "hand_id"}
    if required_decision - set(decisions) or {"row_id", "action_label"} - set(history):
        raise RuntimeError("nested raw additional identity source is incomplete")
    if {"row_id", *legal_columns} - set(legal_rows):
        raise RuntimeError("nested raw additional legal source is incomplete")
    if (
        decisions["row_id"].duplicated().any()
        or history["row_id"].duplicated().any()
        or legal_rows["row_id"].duplicated().any()
    ):
        raise RuntimeError("nested raw additional source duplicates row identity")
    identity = decisions[["row_id", "game_id", "hand_id"]].merge(
        history[["row_id", "action_label"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    identity = identity.merge(
        legal_rows[["row_id", *legal_columns]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if identity.isna().any().any():
        raise RuntimeError("nested raw additional target/legal join is incomplete")
    feature_frames = build_additional_exploitation_feature_frames(decisions, history)
    if list(feature_frames) != list(LAYERS):
        raise RuntimeError("nested raw additional feature layer order changed")
    layers: dict[str, dict[str, Any]] = {}
    for completed, layer in enumerate(LAYERS, start=1):
        frame = feature_frames[layer]
        joined = identity.merge(frame, on="row_id", how="left", validate="one_to_one")
        names = list(_feature_names(frame))
        matrix = joined[names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        labels = joined["action_label"].map(ACTION_TO_INDEX)
        if labels.isna().any() or np.isinf(matrix).any():
            raise RuntimeError(f"nested raw additional values are invalid: {layer}")
        layers[layer] = {
            "x_raw": matrix,
            "y": labels.to_numpy(dtype=np.int8),
            "legal": _strict_legal_mask(
                joined[legal_columns].to_numpy(),
                label=f"nested raw materialized legal mask for {layer}",
            ),
            "row_ids": joined["row_id"].astype(str).to_numpy(),
            "feature_names": names,
        }
        if progress_callback is not None:
            progress_callback(completed, len(LAYERS), layer)
    return write_retrospective_nested_additional_source_from_arrays(
        output_root, layers, source_binding=source_binding
    )


def load_retrospective_nested_additional_layer(
    source_root: str | Path,
    layer: str,
    *,
    local_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate and load only one layer from the raw nested source."""
    from stage6_additional_exploitation_execution import (
        _load_npy,
        _stream_verified_local_cache,
        _strict_legal_mask,
    )

    if layer not in LAYERS:
        raise ValueError(f"unknown nested raw additional layer: {layer}")
    root = Path(source_root).resolve()
    manifest_path = root / SOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("status") != "PASS_RAW_CAUSAL_ADDITIONAL_SOURCE_READY"
        or manifest.get("logical_data_lanes") != LOGICAL_DATA_LANES
        or manifest.get("imputation_scope")
        != "none_at_materialization__inner_train_per_fold_only"
        or manifest.get("action_or_decision_timing_feature") is not False
        or [row.get("layer") for row in manifest.get("layers", [])]
        != list(LAYERS)
    ):
        raise RuntimeError("nested raw additional source authority changed")
    document = next(row for row in manifest["layers"] if row["layer"] == layer)
    cache_root = Path(local_cache_root).resolve() if local_cache_root else None
    paths: dict[str, Path] = {}
    for identity in document["members"]:
        source_path = root / str(identity["path"])
        if cache_root is None:
            if (
                not source_path.is_file()
                or source_path.stat().st_size != int(identity["bytes"])
                or _sha256_file(source_path) != str(identity["sha256"])
            ):
                raise RuntimeError(f"nested raw source member changed: {source_path}")
            authenticated = source_path
        else:
            authenticated = cache_root / str(identity["path"])
            _stream_verified_local_cache(
                source_path,
                authenticated,
                expected_bytes=int(identity["bytes"]),
                expected_sha256=str(identity["sha256"]),
                progress=None,
                completed_before=0,
                total_bytes=int(identity["bytes"]),
            )
        paths[authenticated.name] = authenticated
    required = {"x_raw.npy", "y.npy", "legal.npy", "row_ids.npy", "lanes.npy"}
    if set(paths) != required:
        raise RuntimeError("nested raw additional member inventory changed")
    arrays = {name[:-4]: _load_npy(path) for name, path in paths.items()}
    x_raw = np.asarray(arrays["x_raw"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.int8)
    legal = _strict_legal_mask(
        arrays["legal"], label=f"authenticated nested raw legal mask for {layer}"
    )
    row_ids = np.asarray(arrays["row_ids"]).astype(str)
    expected_lanes = np.asarray([
        int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
        % LOGICAL_DATA_LANES
        for value in row_ids
    ], dtype=np.uint8)
    n = int(document["row_count"])
    if (
        x_raw.shape != (n, int(document["feature_count"]))
        or y.shape != (n,)
        or legal.shape != (n, len(ACTIONS))
        or row_ids.shape != (n,)
        or not np.array_equal(np.asarray(arrays["lanes"]), expected_lanes)
        or np.isinf(x_raw).any()
        or not legal[np.arange(n), y.astype(int)].all()
    ):
        raise RuntimeError(f"nested raw additional layer values changed: {layer}")
    return {
        "manifest": manifest,
        "layer_document": document,
        "x_raw": x_raw,
        "y": y,
        "legal": legal,
        "row_ids": row_ids,
        "feature_names": list(document["feature_names"]),
    }


__all__ = [
    "LAYERS",
    "SOURCE_MANIFEST_NAME",
    "load_retrospective_nested_additional_layer",
    "materialize_retrospective_nested_additional_source",
    "write_retrospective_nested_additional_source_from_arrays",
]
