"""Compact, validation-only runtime for Stage 6 Notebooks 02 through 07.

The module consumes Notebook 01's sealed 48-lane materialization.  It never
opens test or private-lockbox labels.  Candidate selection is restricted to
the train and validation roles, and every result remains non-promotional.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage6_metrics import multiclass_metric_report


LANE_COUNT = 48
NOTEBOOK01_RUN_ID = "8a6ee5a801b7ae945ddcca7e122d33dd722d25ab5112b6de785965d62e4c6226"
NOTEBOOK01_SPLIT_ROOT = "7a3d1b0bc3d3b816dfd12e6d2b1ea05b960564c8d79444ef1f9294f92eddb8cf"
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
STREETS = ("preflop", "flop", "turn", "river")
LEGAL_COLUMNS = tuple(f"legal_{action}" for action in ACTIONS)
NUMERIC_FEATURES = (
    "amount_to_call_bb",
    "pot_before_bb",
    "actor_committed_before_bb",
    "actor_stack_before_bb",
    "legal_min_bet_to_bb",
    "legal_min_raise_to_bb",
    "legal_max_bet_raise_to_bb",
)
MODEL_FEATURE_COLUMNS = (
    "street",
    *LEGAL_COLUMNS,
    *NUMERIC_FEATURES,
    "short_all_in_call",
)
CONTEXT_FEATURE_COLUMNS = (
    "active_player_count_before",
    "players_with_chips_before",
    "effective_stack_before_bb",
    "spr_before",
    "board_card_count",
    "board_pair_level",
    "board_is_paired",
    "board_is_two_tone",
    "board_is_monotone",
    "board_connectedness_score",
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
)
BEHAVIOR_FEATURE_COLUMNS = tuple(dict.fromkeys((*MODEL_FEATURE_COLUMNS, *CONTEXT_FEATURE_COLUMNS)))
CONTEXT_LEVELS = ("specific", "reduced", "legal_street")
CONTEXT_MIN_SUPPORT = 100
CONTEXT_PRIOR_STRENGTH = 25.0
SPLIT_COLUMNS = ("row_id", "panel", "role", "eligible_for_selection")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_document(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def validate_document(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping) or not isinstance(document.get("self_sha256"), str):
        raise RuntimeError("self-hashed document is invalid")
    expected = str(document["self_sha256"])
    candidate = dict(document)
    candidate.pop("self_sha256", None)
    if hashlib.sha256(canonical_json_bytes(candidate)).hexdigest() != expected:
        raise RuntimeError("self-hashed document mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_identity_bound_parquet(
    root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected_keys = {"relative_path", "bytes", "sha256", "row_count"}
    observed_keys = set(identity) - {"lane_id"}
    if observed_keys != expected_keys:
        raise RuntimeError(f"{label} identity is invalid")
    relative = Path(str(identity["relative_path"]))
    expected_path = root / relative
    if path != expected_path or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} path binding is invalid")
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    if path.stat().st_size != int(identity["bytes"]):
        raise RuntimeError(f"{label} byte count mismatch")
    if sha256_file(path) != str(identity["sha256"]):
        raise RuntimeError(f"{label} SHA-256 mismatch")
    if pq.ParquetFile(path).metadata.num_rows != int(identity["row_count"]):
        raise RuntimeError(f"{label} row-count mismatch")


def _verify_materialized_payloads(root: Path, manifest: Mapping[str, Any]) -> None:
    lanes = manifest.get("lanes")
    expected_lane_ids = [f"lane-{lane:02d}" for lane in range(LANE_COUNT)]
    if not isinstance(lanes, list) or [row.get("lane_id") for row in lanes] != expected_lane_ids:
        raise RuntimeError("Notebook 01 materialization lane order is invalid")
    accepted_total = 0
    profile_total = 0
    for lane, lane_row in enumerate(lanes):
        lane_id = expected_lane_ids[lane]
        outputs = lane_row.get("outputs")
        if not isinstance(outputs, Mapping):
            raise RuntimeError(f"{lane_id} output bindings are invalid")
        for key, name in (
            ("accepted_decisions", "accepted-decisions.parquet"),
            ("profile_snapshots", "profile-snapshots.parquet"),
        ):
            identity = outputs.get(key)
            if not isinstance(identity, Mapping):
                raise RuntimeError(f"{lane_id} {key} binding is missing")
            expected_relative = (Path("lanes") / lane_id / name).as_posix()
            if identity.get("relative_path") != expected_relative:
                raise RuntimeError(f"{lane_id} {key} relative path is invalid")
            _verify_identity_bound_parquet(
                root,
                root / expected_relative,
                identity,
                label=f"{lane_id} {key}",
            )
        accepted_total += int(outputs["accepted_decisions"]["row_count"])
        profile_total += int(outputs["profile_snapshots"]["row_count"])
    if (
        accepted_total != manifest.get("accepted_decision_row_count")
        or profile_total != manifest.get("profile_snapshot_row_count_across_lanes")
    ):
        raise RuntimeError("Notebook 01 materialization row totals are invalid")

    split_path = root / "splits" / "split-assignment-manifest.json"
    if not split_path.is_file():
        raise RuntimeError(f"Notebook 01 split manifest is missing: {split_path}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_document(split_manifest)
    if (
        split_manifest.get("schema_version")
        != "stage6_notebook01_split_assignment_manifest_v1"
        or split_manifest.get("status") != "PASS_SEALED_GROUP_TIME_SPLIT_ASSIGNMENT"
        or split_manifest.get("run_identity") != NOTEBOOK01_RUN_ID
        or split_manifest.get("split_assignment_root_digest") != NOTEBOOK01_SPLIT_ROOT
        or split_manifest.get("private_lockbox_assignment_sealed") is not True
        or split_manifest.get("private_lockbox_opened") is not False
    ):
        raise RuntimeError("Notebook 01 split manifest authority is invalid")
    outputs = split_manifest.get("lane_outputs")
    if not isinstance(outputs, list) or [row.get("lane_id") for row in outputs] != expected_lane_ids:
        raise RuntimeError("Notebook 01 split lane order is invalid")
    split_total = 0
    for lane, identity in enumerate(outputs):
        lane_id = expected_lane_ids[lane]
        expected_relative = (Path("splits") / f"{lane_id}-split-assignments.parquet").as_posix()
        if identity.get("relative_path") != expected_relative:
            raise RuntimeError(f"{lane_id} split relative path is invalid")
        _verify_identity_bound_parquet(
            root,
            root / expected_relative,
            identity,
            label=f"{lane_id} split assignments",
        )
        split_total += int(identity["row_count"])
    if (
        split_total != split_manifest.get("accepted_decision_row_count")
        or split_total != manifest.get("split_assignment_row_count")
        or split_manifest.get("self_sha256")
        != manifest.get("split_assignment_manifest_self_sha256")
    ):
        raise RuntimeError("Notebook 01 split payload binding is invalid")


def resolve_workers(requested: str) -> dict[str, Any]:
    raw = str(requested).strip()
    usable = max(1, int(os.cpu_count() or 1))
    if raw.lower() == "auto" or raw in {"", "0"}:
        requested_capacity = usable
        mode = "auto"
    elif raw.isascii() and raw.isdecimal() and int(raw) > 0:
        requested_capacity = int(raw)
        mode = "explicit"
    else:
        raise RuntimeError("REQUESTED_WORKERS must be auto, blank, 0, or a positive integer")
    live = min(requested_capacity, usable, LANE_COUNT)
    return {
        "requested_workers": raw or "auto",
        "requested_workers_mode": mode,
        "detected_usable_logical_cpus": usable,
        "logical_lane_count": LANE_COUNT,
        "live_workers": live,
        "accelerator_device_cores_used_as_process_workers": False,
        "scientific_lane_count_changed": False,
    }


def materialization_root(stage_drive_root: Path) -> Path:
    root = (
        stage_drive_root
        / "runtime_state"
        / "notebook01"
        / "materializations"
        / NOTEBOOK01_RUN_ID
    )
    manifest_path = root / "stage6_notebook01_materialization_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Notebook 01 materialization is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_document(manifest)
    if (
        manifest.get("schema_version") != "stage6_notebook01_materialization_manifest_v1"
        or manifest.get("status")
        != "PASS_FULL_CAUSAL_DECISION_PROFILE_MATERIALIZATION_AND_SEALED_SPLIT_LOCK"
        or manifest.get("run_identity") != NOTEBOOK01_RUN_ID
        or manifest.get("completed_lane_count") != LANE_COUNT
        or manifest.get("split_assignment_root_digest") != NOTEBOOK01_SPLIT_ROOT
        or manifest.get("private_lockbox_assignment_sealed") is not True
        or manifest.get("private_lockbox_opened") is not False
    ):
        raise RuntimeError("Notebook 01 materialization authority is invalid")
    _verify_materialized_payloads(root, manifest)
    return root


def _lane_paths(root: Path, lane: int) -> tuple[Path, Path, Path]:
    lane_id = f"lane-{lane:02d}"
    return (
        root / "lanes" / lane_id / "accepted-decisions.parquet",
        root / "splits" / f"{lane_id}-split-assignments.parquet",
        root / "lanes" / lane_id / "profile-snapshots.parquet",
    )


def _load_role_lane(
    root_text: str,
    lane: int,
    decision_columns: Sequence[str],
    role: str,
) -> pd.DataFrame:
    root = Path(root_text)
    decision_path, split_path, _ = _lane_paths(root, lane)
    splits = pd.read_parquet(split_path, columns=list(SPLIT_COLUMNS))
    if splits["row_id"].duplicated().any():
        raise RuntimeError(f"lane-{lane:02d} has duplicate row identities")
    selected_splits = _selection_rows(splits, role)
    selected_ids = selected_splits["row_id"].astype(str).tolist()
    if not selected_ids:
        return selected_splits.iloc[0:0].copy()
    # Apply the role selection at the Parquet read boundary.  Target columns
    # from test/private-lockbox rows never enter the Python process.
    decisions = pq.read_table(
        decision_path,
        columns=list(dict.fromkeys(decision_columns)),
        filters=[("row_id", "in", selected_ids)],
    ).to_pandas()
    if decisions["row_id"].duplicated().any():
        raise RuntimeError(f"lane-{lane:02d} has duplicate selected decision identities")
    joined = decisions.merge(selected_splits, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(selected_splits) or len(joined) != len(decisions):
        raise RuntimeError(f"lane-{lane:02d} selected decision/split join is not exact")
    return joined


def _selection_rows(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    selected = frame.loc[frame["role"].astype(str).eq(role)].copy()
    if (selected["panel"].astype(str) == "final_private_lockbox").any():
        raise RuntimeError("private lockbox rows were exposed before Notebook 09")
    if role == "validation" and not selected["eligible_for_selection"].eq(True).all():
        raise RuntimeError("validation rows are not selection eligible")
    return selected


def _street_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    street = frame["street"].astype(str)
    if mode == "preflop":
        return street.eq("preflop")
    if mode == "postflop":
        return street.isin(("flop", "turn", "river"))
    raise RuntimeError(f"unknown street mode: {mode}")


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    street = frame["street"].astype(str)
    if not street.isin(STREETS).all():
        raise RuntimeError("unknown street value in accepted decisions")
    parts: list[np.ndarray] = [
        np.column_stack([(street == value).to_numpy(dtype=float) for value in STREETS])
    ]
    for column in LEGAL_COLUMNS + ("short_all_in_call",):
        values = frame[column]
        if not values.map(lambda value: type(value) in {bool, np.bool_}).all():
            raise RuntimeError(f"{column} is not strict boolean data")
        parts.append(values.to_numpy(dtype=float).reshape(-1, 1))
    for column in NUMERIC_FEATURES:
        raw = frame[column].to_numpy()
        if any(isinstance(value, (bool, np.bool_)) for value in raw if not pd.isna(value)):
            raise RuntimeError(f"{column} contains boolean numeric data")
        try:
            numeric = raw.astype(float)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{column} is not numeric") from error
        if np.isinf(numeric).any():
            raise RuntimeError(f"{column} contains infinity")
        parts.append(numeric.reshape(-1, 1))
    return np.column_stack(parts)


def _parallel_lanes(
    function: Callable[..., Any],
    root: Path,
    workers: int,
    *extra: Any,
) -> list[Any]:
    results: dict[int, Any] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(function, str(root), lane, *extra): lane
            for lane in range(LANE_COUNT)
        }
        for future in as_completed(pending):
            lane = pending[future]
            results[lane] = future.result()
            print(json.dumps({"event": "stage6_lane_complete", "lane": lane, "completed": len(results), "total": LANE_COUNT}), flush=True)
    return [results[lane] for lane in range(LANE_COUNT)]


def _behavior_train_lane(root: str, lane: int, street_mode: str) -> dict[str, Any]:
    columns = ("row_id", "action_label", *MODEL_FEATURE_COLUMNS)
    frame = _load_role_lane(root, lane, columns, "train")
    frame = frame.loc[_street_mask(frame, street_mode)]
    matrix = _feature_matrix(frame)
    labels = frame["action_label"].astype(str).to_numpy()
    width = matrix.shape[1]
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "global_count": np.isfinite(matrix).sum(axis=0),
        "global_sum": np.nansum(matrix, axis=0),
        "global_sumsq": np.nansum(matrix * matrix, axis=0),
        "classes": {},
    }
    for action in ACTIONS:
        values = matrix[labels == action]
        result["classes"][action] = {
            "rows": int(len(values)),
            "count": np.isfinite(values).sum(axis=0),
            "sum": np.nansum(values, axis=0),
            "sumsq": np.nansum(values * values, axis=0),
        }
    if width != 4 + len(LEGAL_COLUMNS) + 1 + len(NUMERIC_FEATURES):
        raise RuntimeError("behavior feature width changed")
    return result


def _fit_gaussian_model(lanes: Sequence[Mapping[str, Any]], street_mode: str) -> dict[str, Any]:
    total_rows = sum(int(lane["rows"]) for lane in lanes)
    if total_rows <= 0:
        raise RuntimeError("behavior training set is empty")
    global_count = np.sum([lane["global_count"] for lane in lanes], axis=0).astype(float)
    global_sum = np.sum([lane["global_sum"] for lane in lanes], axis=0).astype(float)
    global_sumsq = np.sum([lane["global_sumsq"] for lane in lanes], axis=0).astype(float)
    imputer = np.divide(global_sum, global_count, out=np.zeros_like(global_sum), where=global_count > 0)
    global_var = np.maximum(
        np.divide(global_sumsq, global_count, out=np.zeros_like(global_sum), where=global_count > 0) - imputer * imputer,
        0.0,
    )
    variance_floor = np.maximum(global_var * 1e-6, 1e-9)
    class_rows = np.asarray([sum(int(lane["classes"][a]["rows"]) for lane in lanes) for a in ACTIONS], dtype=float)
    smoothed_priors = (class_rows + 1.0) / (class_rows.sum() + len(ACTIONS))
    means = []
    variances = []
    for action, rows in zip(ACTIONS, class_rows):
        if rows == 0:
            mean = imputer.copy()
            variance = np.maximum(global_var, variance_floor)
        else:
            count = np.sum([lane["classes"][action]["count"] for lane in lanes], axis=0).astype(float)
            sums = np.sum([lane["classes"][action]["sum"] for lane in lanes], axis=0).astype(float)
            sumsq = np.sum([lane["classes"][action]["sumsq"] for lane in lanes], axis=0).astype(float)
            missing = rows - count
            sums = sums + missing * imputer
            sumsq = sumsq + missing * imputer * imputer
            mean = sums / rows
            variance = np.maximum(sumsq / rows - mean * mean, variance_floor)
        means.append(mean)
        variances.append(variance)
    return seal_document({
        "schema_version": "stage6_diagonal_gaussian_behavior_head_v1",
        "street_scope": street_mode,
        "label_order": list(ACTIONS),
        "source_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "derived_encoding": {"street_one_hot_order": list(STREETS)},
        "numeric_imputer_fit_partition": "train_only",
        "training_row_count": int(total_rows),
        "class_counts": {action: int(count) for action, count in zip(ACTIONS, class_rows)},
        "class_prior_smoothing": "add_one_over_registered_action_schema",
        "class_priors": smoothed_priors.tolist(),
        "imputer_means": imputer.tolist(),
        "class_means": np.asarray(means).tolist(),
        "class_variances": np.asarray(variances).tolist(),
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def _predict_gaussian(matrix: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    imputer = np.asarray(model["imputer_means"], dtype=float)
    values = np.where(np.isfinite(matrix), matrix, imputer)
    means = np.asarray(model["class_means"], dtype=float)
    variances = np.asarray(model["class_variances"], dtype=float)
    log_priors = np.log(np.asarray(model["class_priors"], dtype=float))
    log_scores = log_priors - 0.5 * (
        np.log(2 * math.pi * variances).sum(axis=1)[None, :]
        + (((values[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]).sum(axis=2)
    )
    log_scores -= log_scores.max(axis=1, keepdims=True)
    scores = np.exp(log_scores)
    return scores / scores.sum(axis=1, keepdims=True)


def _strict_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    invalid = raw.notna().to_numpy() & np.isnan(numeric)
    if invalid.any() or np.isinf(numeric).any():
        raise RuntimeError(f"{column} is not finite numeric data")
    return numeric


def _strict_boolean(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if not values.map(lambda value: type(value) in {bool, np.bool_}).all():
        raise RuntimeError(f"{column} is not strict boolean data")
    return values.to_numpy(dtype=bool)


def _strict_category(frame: pd.DataFrame, column: str, allowed: set[str]) -> np.ndarray:
    values = frame[column].astype("string")
    if values.isna().any() or not values.isin(allowed).all():
        raise RuntimeError(f"{column} is outside the registered categorical schema")
    return values.astype(str).to_numpy()


def _numeric_bucket(values: np.ndarray, boundaries: Sequence[float]) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    result[finite] = np.digitize(values[finite], np.asarray(boundaries, dtype=float), right=False)
    return result


def _count_bucket(values: np.ndarray, maximum: int) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    rounded = np.rint(values[finite])
    if not np.allclose(values[finite], rounded, rtol=0.0, atol=1e-9):
        raise RuntimeError("causal count feature contains a non-integral value")
    result[finite] = np.clip(rounded, 0, maximum).astype(np.int16)
    return result


def _context_keys(frame: pd.DataFrame, street_mode: str) -> dict[str, list[str]]:
    if street_mode not in {"preflop", "postflop"}:
        raise RuntimeError(f"unknown street mode: {street_mode}")
    street = frame["street"].astype(str).to_numpy()
    if not pd.Series(street).isin(STREETS).all():
        raise RuntimeError("unknown street value in behavior context")
    legal_arrays = [_strict_boolean(frame, column) for column in LEGAL_COLUMNS]
    legal_signatures = [
        "".join("1" if values[row] else "0" for values in legal_arrays)
        for row in range(len(frame))
    ]
    active = _count_bucket(_strict_numeric(frame, "active_player_count_before"), 10)
    amount_to_call = _strict_numeric(frame, "amount_to_call_bb")
    pot = _strict_numeric(frame, "pot_before_bb")
    call_to_pot = np.divide(
        amount_to_call,
        np.maximum(pot, 0.25),
        out=np.full(len(frame), np.nan, dtype=float),
        where=np.isfinite(amount_to_call) & np.isfinite(pot),
    )
    call_bucket = _numeric_bucket(call_to_pot, (0.0, 0.1, 0.25, 0.5, 1.0, 2.0))
    spr_bucket = _numeric_bucket(_strict_numeric(frame, "spr_before"), (0.0, 1.0, 2.0, 4.0, 8.0, 16.0))

    specific_parts: list[np.ndarray] = [street, np.asarray(legal_signatures), active, call_bucket, spr_bucket]
    if street_mode == "preflop":
        for column in (
            "preflop_limper_count_before",
            "preflop_caller_count_before",
            "preflop_raise_count_before",
            "preflop_prior_entry_count_before",
        ):
            specific_parts.append(_count_bucket(_strict_numeric(frame, column), 4))
        for column in (
            "preflop_facing_open_raise",
            "preflop_facing_three_bet_or_more",
            "preflop_squeeze_opportunity",
            "preflop_limp_opportunity",
            "preflop_actor_has_limped_before",
            "preflop_actor_has_called_before",
            "preflop_actor_has_raised_before",
        ):
            specific_parts.append(_strict_boolean(frame, column).astype(np.int8))
    else:
        for column, maximum in (
            ("board_card_count", 5),
            ("street_action_count_before", 6),
            ("street_aggressive_action_count_before", 4),
            ("street_call_count_before", 4),
            ("street_check_count_before", 4),
            ("street_prior_actor_bet_raise_count", 4),
        ):
            specific_parts.append(_count_bucket(_strict_numeric(frame, column), maximum))
        specific_parts.append(
            _strict_category(
                frame,
                "board_pair_level",
                {"unpaired", "one_pair", "two_pair", "trips", "quads", "full_house_possible"},
            )
        )
        specific_parts.append(
            _numeric_bucket(
                _strict_numeric(frame, "board_connectedness_score"),
                (0.0, 0.25, 0.5, 0.75, 1.0),
            )
        )
        for column in (
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
        ):
            specific_parts.append(_strict_boolean(frame, column).astype(np.int8))

    def encode(parts: Sequence[np.ndarray]) -> list[str]:
        return ["|".join(str(part[row]) for part in parts) for row in range(len(frame))]

    return {
        "specific": encode(specific_parts),
        "reduced": encode(specific_parts[:5]),
        "legal_street": encode(specific_parts[:2]),
    }


def _behavior_context_train_lane(root: str, lane: int, street_mode: str) -> dict[str, Any]:
    columns = ("row_id", "action_label", *BEHAVIOR_FEATURE_COLUMNS)
    frame = _load_role_lane(root, lane, columns, "train")
    frame = frame.loc[_street_mask(frame, street_mode)].copy()
    labels = frame["action_label"].astype(str)
    if not labels.isin(ACTIONS).all():
        raise RuntimeError("behavior training label is outside the registered action schema")
    keys = _context_keys(frame, street_mode)
    contexts: dict[str, dict[str, list[int]]] = {}
    for level in CONTEXT_LEVELS:
        grouped = (
            pd.DataFrame({"key": keys[level], "action": labels.to_numpy()})
            .groupby(["key", "action"], sort=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=ACTIONS, fill_value=0)
        )
        contexts[level] = {
            str(key): [int(value) for value in row]
            for key, row in zip(grouped.index.tolist(), grouped.to_numpy(dtype=np.int64))
        }
    return {
        "rows": int(len(frame)),
        "class_counts": [int((labels == action).sum()) for action in ACTIONS],
        "contexts": contexts,
    }


def _fit_context_model(lanes: Sequence[Mapping[str, Any]], street_mode: str) -> dict[str, Any]:
    class_counts = np.sum([lane["class_counts"] for lane in lanes], axis=0).astype(float)
    total_rows = int(class_counts.sum())
    if total_rows <= 0:
        raise RuntimeError("behavior context training set is empty")
    priors = (class_counts + 1.0) / (class_counts.sum() + len(ACTIONS))
    tables: dict[str, dict[str, Any]] = {}
    for level in CONTEXT_LEVELS:
        merged: dict[str, np.ndarray] = {}
        for lane in lanes:
            for key, counts in lane["contexts"][level].items():
                merged.setdefault(str(key), np.zeros(len(ACTIONS), dtype=np.int64))
                merged[str(key)] += np.asarray(counts, dtype=np.int64)
        table = {}
        for key in sorted(merged):
            counts = merged[key]
            support = int(counts.sum())
            if support < CONTEXT_MIN_SUPPORT:
                continue
            probabilities = (counts.astype(float) + CONTEXT_PRIOR_STRENGTH * priors) / (
                support + CONTEXT_PRIOR_STRENGTH
            )
            table[key] = {"support": support, "probabilities": probabilities.tolist()}
        tables[level] = table
    return seal_document({
        "schema_version": "stage6_backoff_context_frequency_behavior_head_v1",
        "street_scope": street_mode,
        "label_order": list(ACTIONS),
        "source_feature_columns": list(BEHAVIOR_FEATURE_COLUMNS),
        "context_levels_most_specific_first": list(CONTEXT_LEVELS),
        "minimum_context_support": CONTEXT_MIN_SUPPORT,
        "prior_strength": CONTEXT_PRIOR_STRENGTH,
        "training_row_count": total_rows,
        "class_counts": {action: int(count) for action, count in zip(ACTIONS, class_counts)},
        "class_priors": priors.tolist(),
        "context_tables": tables,
        "context_count_by_level": {level: len(tables[level]) for level in CONTEXT_LEVELS},
        "fit_partition": "train_only",
        "fixed_parameters_not_validation_tuned": True,
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def _predict_context_model(
    frame: pd.DataFrame,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    street_mode = str(model["street_scope"])
    keys = _context_keys(frame, street_mode)
    priors = np.asarray(model["class_priors"], dtype=float)
    probabilities = np.broadcast_to(priors, (len(frame), len(ACTIONS))).copy()
    selected_levels = ["global_prior"] * len(frame)
    unresolved = np.ones(len(frame), dtype=bool)
    tables = model["context_tables"]
    for level in model["context_levels_most_specific_first"]:
        table = tables[level]
        for row, key in enumerate(keys[level]):
            if unresolved[row] and key in table:
                probabilities[row] = np.asarray(table[key]["probabilities"], dtype=float)
                selected_levels[row] = str(level)
                unresolved[row] = False
    return probabilities, selected_levels


def _project_legal(probabilities: np.ndarray, legal: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, int, int]:
    raw_choice = probabilities.argmax(axis=1)
    illegal_pre = int((~legal[np.arange(len(legal)), raw_choice]).sum())
    projected = probabilities * legal
    totals = projected.sum(axis=1)
    fallback_count = 0
    zero = np.flatnonzero(totals <= 0)
    for row in zero:
        choices = np.flatnonzero(legal[row])
        if len(choices) == 0:
            raise RuntimeError("visible legal action set is empty")
        choice = choices[np.argmax(priors[choices])]
        projected[row, choice] = 1.0
        totals[row] = 1.0
        fallback_count += 1
    projected /= totals[:, None]
    if (projected[~legal] != 0).any():
        raise RuntimeError("legal projection emitted illegal probability mass")
    return projected, illegal_pre, fallback_count


def _behavior_validation_lane(root: str, lane: int, street_mode: str, model: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(model.get("schema_version", ""))
    feature_columns = (
        BEHAVIOR_FEATURE_COLUMNS
        if schema == "stage6_backoff_context_frequency_behavior_head_v1"
        else MODEL_FEATURE_COLUMNS
    )
    columns = ("row_id", "game_id", "hand_id", "action_label", *feature_columns)
    frame = _load_role_lane(root, lane, columns, "validation")
    frame = frame.loc[_street_mask(frame, street_mode)].copy()
    legal = frame[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
    priors = np.asarray(model["class_priors"], dtype=float)
    if schema == "stage6_backoff_context_frequency_behavior_head_v1":
        candidate_raw, selected_levels = _predict_context_model(frame, model)
    elif schema == "stage6_diagonal_gaussian_behavior_head_v1":
        candidate_raw = _predict_gaussian(_feature_matrix(frame), model)
        selected_levels = ["diagonal_gaussian"] * len(frame)
    else:
        raise RuntimeError("unknown behavior candidate schema")
    baseline_raw = np.broadcast_to(priors, candidate_raw.shape).copy()
    candidate, candidate_illegal, candidate_fallback = _project_legal(candidate_raw, legal, priors)
    baseline, baseline_illegal, baseline_fallback = _project_legal(baseline_raw, legal, priors)
    return {
        "truth": frame["action_label"].astype(str).tolist(),
        "candidate": candidate,
        "baseline": baseline,
        "panels": frame["panel"].astype(str).tolist(),
        "visible_players_with_chips_before": frame["players_with_chips_before"].tolist()
        if "players_with_chips_before" in frame.columns
        else [None] * len(frame),
        "selected_context_levels": selected_levels,
        "clusters": (frame["game_id"].astype(str) + "\x1f" + frame["hand_id"].astype(str)).tolist(),
        "candidate_illegal_preprojection": candidate_illegal,
        "baseline_illegal_preprojection": baseline_illegal,
        "candidate_zero_mass_fallback": candidate_fallback,
        "baseline_zero_mass_fallback": baseline_fallback,
    }


def _losses(truth: Sequence[str], probabilities: np.ndarray) -> np.ndarray:
    indices = np.asarray([ACTIONS.index(label) for label in truth], dtype=int)
    return -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-15, 1.0))


def _behavior_report(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = sum((list(lane["truth"]) for lane in lanes), [])
    if not truth:
        raise RuntimeError("behavior validation set is empty")
    candidate = np.concatenate([lane["candidate"] for lane in lanes], axis=0)
    baseline = np.concatenate([lane["baseline"] for lane in lanes], axis=0)
    panels = np.asarray(sum((list(lane["panels"]) for lane in lanes), []), dtype=object)
    roster_sizes = np.asarray(
        sum((list(lane["visible_players_with_chips_before"]) for lane in lanes), []),
        dtype=object,
    )
    candidate_report = multiclass_metric_report(truth, candidate, ACTIONS)
    baseline_report = multiclass_metric_report(truth, baseline, ACTIONS)
    panel_reports = {}
    truth_array = np.asarray(truth, dtype=object)
    for panel in sorted(set(panels)):
        mask = panels == panel
        panel_reports[str(panel)] = {
            "candidate": multiclass_metric_report(truth_array[mask].tolist(), candidate[mask], ACTIONS),
            "class_frequency_baseline": multiclass_metric_report(truth_array[mask].tolist(), baseline[mask], ACTIONS),
        }
    roster_reports = {}
    for size in sorted({int(value) for value in roster_sizes if value is not None and not pd.isna(value)}):
        mask = np.asarray([value is not None and not pd.isna(value) and int(value) == size for value in roster_sizes])
        roster_reports[str(size)] = {
            "candidate": multiclass_metric_report(truth_array[mask].tolist(), candidate[mask], ACTIONS),
            "class_frequency_baseline": multiclass_metric_report(truth_array[mask].tolist(), baseline[mask], ACTIONS),
        }
    context_level_counts: dict[str, int] = {}
    for level in sum((list(lane["selected_context_levels"]) for lane in lanes), []):
        context_level_counts[str(level)] = context_level_counts.get(str(level), 0) + 1
    return seal_document({
        "schema_version": "stage6_behavior_head_validation_report_v1",
        "selection_partition": "validation_only",
        "candidate": candidate_report,
        "class_frequency_baseline": baseline_report,
        "candidate_minus_baseline_log_loss": float(_losses(truth, candidate).mean() - _losses(truth, baseline).mean()),
        "candidate_illegal_preprojection_count": int(sum(int(lane["candidate_illegal_preprojection"]) for lane in lanes)),
        "candidate_illegal_postprojection_count": 0,
        "candidate_zero_legal_mass_fallback_count": int(sum(int(lane["candidate_zero_mass_fallback"]) for lane in lanes)),
        "baseline_illegal_preprojection_count": int(sum(int(lane["baseline_illegal_preprojection"]) for lane in lanes)),
        "baseline_illegal_postprojection_count": 0,
        "baseline_zero_legal_mass_fallback_count": int(sum(int(lane["baseline_zero_mass_fallback"]) for lane in lanes)),
        "per_panel": panel_reports,
        "per_visible_players_with_chips_before": roster_reports,
        "selected_context_level_counts": context_level_counts,
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def run_behavior_head(root: Path, workers: int, street_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    training = _parallel_lanes(_behavior_context_train_lane, root, workers, street_mode)
    model = _fit_context_model(training, street_mode)
    validation = _parallel_lanes(_behavior_validation_lane, root, workers, street_mode, model)
    report = _behavior_report(validation)
    return model, report


def _size_train_lane(root: str, lane: int) -> dict[str, Any]:
    columns = (
        "row_id", "street", "action_label", "action_to_bb",
        "legal_min_bet_to_bb", "legal_min_raise_to_bb", "legal_max_bet_raise_to_bb",
    )
    frame = _load_role_lane(root, lane, columns, "train")
    result: dict[str, Any] = {}
    for street in STREETS:
        for action in ("bet", "raise"):
            rows = frame.loc[frame["street"].astype(str).eq(street) & frame["action_label"].astype(str).eq(action)]
            minimum_column = "legal_min_bet_to_bb" if action == "bet" else "legal_min_raise_to_bb"
            minimum = rows[minimum_column].to_numpy(dtype=float)
            maximum = rows["legal_max_bet_raise_to_bb"].to_numpy(dtype=float)
            target = rows["action_to_bb"].to_numpy(dtype=float)
            valid = np.isfinite(minimum) & np.isfinite(maximum) & np.isfinite(target) & (maximum > minimum)
            fraction = (target[valid] - minimum[valid]) / (maximum[valid] - minimum[valid])
            if ((fraction < -1e-9) | (fraction > 1 + 1e-9)).any():
                raise RuntimeError("accepted sizing target is outside visible bounds")
            result[f"{street}:{action}"] = {"count": int(len(fraction)), "sum": float(np.clip(fraction, 0, 1).sum())}
    return result


def _fit_size_model(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heads = {}
    for street in STREETS:
        for action in ("bet", "raise"):
            key = f"{street}:{action}"
            count = sum(int(lane[key]["count"]) for lane in lanes)
            total = math.fsum(float(lane[key]["sum"]) for lane in lanes)
            heads[key] = {"training_rows": count, "fraction_between_visible_min_and_max": float(total / count) if count else 0.5}
    return seal_document({
        "schema_version": "stage6_conditional_visible_bound_size_head_v1",
        "heads": heads,
        "all_in_rule": "exact_visible_maximum",
        "unknown_or_inconsistent_bounds": "abstain",
        "fit_partition": "train_only",
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def _size_validation_lane(root: str, lane: int, model: Mapping[str, Any]) -> dict[str, Any]:
    columns = (
        "row_id", "street", "action_label", "action_to_bb",
        "legal_min_bet_to_bb", "legal_min_raise_to_bb", "legal_max_bet_raise_to_bb",
    )
    frame = _load_role_lane(root, lane, columns, "validation")
    targets: list[float] = []
    candidate: list[float] = []
    baseline: list[float] = []
    panels: list[str] = []
    abstained = 0
    for row in frame.itertuples(index=False):
        action = str(row.action_label)
        if action not in {"bet", "raise", "all_in"}:
            continue
        maximum = float(row.legal_max_bet_raise_to_bb)
        target = float(row.action_to_bb)
        if action == "all_in":
            minimum = maximum
            predicted = maximum
            midpoint = maximum
        else:
            minimum_raw = row.legal_min_bet_to_bb if action == "bet" else row.legal_min_raise_to_bb
            minimum = float(minimum_raw)
            if not all(math.isfinite(value) for value in (minimum, maximum, target)) or maximum <= minimum:
                abstained += 1
                continue
            fraction = float(model["heads"][f"{row.street}:{action}"]["fraction_between_visible_min_and_max"])
            predicted = minimum + min(max(fraction, 0.0), 1.0) * (maximum - minimum)
            midpoint = minimum + 0.5 * (maximum - minimum)
        if predicted < minimum - 1e-9 or predicted > maximum + 1e-9:
            raise RuntimeError("conditional size projection escaped visible bounds")
        targets.append(target)
        candidate.append(predicted)
        baseline.append(midpoint)
        panels.append(str(row.panel))
    return {"targets": targets, "candidate": candidate, "baseline": baseline, "panels": panels, "abstained": abstained}


def _size_report(lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = np.asarray(sum((list(lane["targets"]) for lane in lanes), []), dtype=float)
    candidate = np.asarray(sum((list(lane["candidate"]) for lane in lanes), []), dtype=float)
    baseline = np.asarray(sum((list(lane["baseline"]) for lane in lanes), []), dtype=float)
    panels = np.asarray(sum((list(lane["panels"]) for lane in lanes), []), dtype=object)
    if len(target) == 0:
        raise RuntimeError("size validation set is empty")
    def report(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "row_count": int(mask.sum()),
            "candidate_mae_bb": float(np.mean(np.abs(candidate[mask] - target[mask]))),
            "candidate_rmse_bb": float(np.sqrt(np.mean((candidate[mask] - target[mask]) ** 2))),
            "midpoint_baseline_mae_bb": float(np.mean(np.abs(baseline[mask] - target[mask]))),
            "midpoint_baseline_rmse_bb": float(np.sqrt(np.mean((baseline[mask] - target[mask]) ** 2))),
        }
    all_mask = np.ones(len(target), dtype=bool)
    return seal_document({
        "schema_version": "stage6_conditional_size_validation_report_v1",
        "selection_partition": "validation_only",
        "overall": report(all_mask),
        "per_panel": {str(panel): report(panels == panel) for panel in sorted(set(panels))},
        "unknown_bound_abstention_count": int(sum(int(lane["abstained"]) for lane in lanes)),
        "illegal_postprojection_count": 0,
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })


def run_size_head(root: Path, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    training = _parallel_lanes(_size_train_lane, root, workers)
    model = _fit_size_model(training)
    validation = _parallel_lanes(_size_validation_lane, root, workers, model)
    return model, _size_report(validation)


def _profile_lane(root: str, lane: int, metrics: Sequence[Mapping[str, Any]], thresholds: Sequence[int]) -> dict[str, Any]:
    decision_path, split_path, profile_path = _lane_paths(Path(root), lane)
    splits = pd.read_parquet(split_path, columns=list(SPLIT_COLUMNS))
    if splits["row_id"].duplicated().any():
        raise RuntimeError(f"lane-{lane:02d} has duplicate split identities")
    selected_splits = _selection_rows(splits, "validation")
    selected_ids = selected_splits["row_id"].astype(str).tolist()
    decisions = pq.read_table(
        decision_path,
        columns=["row_id", "profile_snapshot_id"],
        filters=[("row_id", "in", selected_ids)],
    ).to_pandas()
    joined = decisions.merge(selected_splits, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(selected_splits):
        raise RuntimeError("validation profile decision/split join is not exact")
    fields = ["profile_snapshot_id"]
    for metric in metrics:
        fields.extend([metric["opportunity_denominator_field"], metric["uncertainty_lower_field"], metric["uncertainty_upper_field"]])
    profiles = pd.read_parquet(profile_path, columns=list(dict.fromkeys(fields)))
    if profiles["profile_snapshot_id"].duplicated().any():
        raise RuntimeError("profile snapshot identity is not unique inside a lane")
    joined = joined.merge(profiles, on="profile_snapshot_id", how="left", validate="many_to_one")
    out: dict[str, Any] = {}
    for metric in metrics:
        denominator = joined[metric["opportunity_denominator_field"]].to_numpy(dtype=float)
        lower = joined[metric["uncertainty_lower_field"]].to_numpy(dtype=float)
        upper = joined[metric["uncertainty_upper_field"]].to_numpy(dtype=float)
        width = upper - lower
        finite = np.isfinite(denominator) & np.isfinite(width) & (denominator >= 0) & (width >= 0)
        candidates = {}
        for threshold in thresholds:
            mask = finite & (denominator >= threshold)
            candidates[str(threshold)] = {
                "eligible": int(mask.sum()),
                "width_sum": float(width[mask].sum()),
                "width_leq_0_25": int((width[mask] <= 0.25).sum()),
            }
        out[str(metric["metric_id"])] = {"observed": int(finite.sum()), "candidates": candidates}
    return out


def run_profile_support(root: Path, package_root: Path, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads((package_root / "contracts" / "stage6_profile_metric_registry_hold_v1.json").read_text(encoding="utf-8"))
    metrics = registry["metrics"]
    thresholds = (5, 10, 20, 40, 80, 160)
    lane_results = _parallel_lanes(_profile_lane, root, workers, metrics, thresholds)
    selected = {}
    audit = {}
    for metric in metrics:
        metric_id = str(metric["metric_id"])
        observed = sum(int(lane[metric_id]["observed"]) for lane in lane_results)
        rows = {}
        chosen = None
        for threshold in thresholds:
            key = str(threshold)
            eligible = sum(int(lane[metric_id]["candidates"][key]["eligible"]) for lane in lane_results)
            width_sum = math.fsum(float(lane[metric_id]["candidates"][key]["width_sum"]) for lane in lane_results)
            narrow = sum(int(lane[metric_id]["candidates"][key]["width_leq_0_25"]) for lane in lane_results)
            mean_width = float(width_sum / eligible) if eligible else None
            rows[key] = {"eligible": eligible, "mean_wilson_width": mean_width, "width_leq_0_25_rate": float(narrow / eligible) if eligible else None}
            if chosen is None and eligible >= 100 and mean_width is not None and mean_width <= 0.25:
                chosen = threshold
        selected[metric_id] = chosen if chosen is not None else 2147483647
        audit[metric_id] = {"validation_observed_snapshots": observed, "candidates": rows, "selected_minimum_support": selected[metric_id]}
    candidate = seal_document({
        "schema_version": "stage6_profile_support_threshold_candidate_v1",
        "selection_partition": "validation_only",
        "candidate_thresholds": list(thresholds),
        "selection_rule": "lowest_threshold_with_at_least_100_rows_and_mean_wilson_width_at_most_0_25_else_hold_sentinel",
        "selected_minimum_support": selected,
        "targeting_status": "ABSTAIN_MISSING_EXACT_VISIBLE_OPPONENT_OR_AGGRESSOR_BINDING",
        "profile_adjustment_runtime_authority": False,
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    report = seal_document({
        "schema_version": "stage6_profile_support_and_heuristic_audit_v1",
        "status": "PASS_VALIDATION_SUPPORT_SELECTION__HEURISTIC_REMAINS_BASELINE_ONLY",
        "metric_audit": audit,
        "exact_targeting_repair": "held_fail_closed_because_materialized_public_decision_schema_has_no_exact_visible_opponent_or_aggressor_identity",
        "historical_heuristic_vs_no_profile_control": "not_scientifically_evaluable_without_exact_target_binding_and_policy_context; no adjustment applied",
        "runtime_profile_adjustment_count": 0,
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    return candidate, report


REVIEW_TASKS = {
    "preflop_dynamic_isolation": [
        "preflop_limper_count_before", "preflop_facing_open_raise",
        "active_player_count_before", "action_label",
    ],
    "heads_up_postflop_baseline": ["active_player_count_before", "street"],
    "heads_up_3bet_pot_baseline": [
        "active_player_count_before", "preflop_raise_count_before", "street",
    ],
    "multiway_slowdown_baseline": ["active_player_count_before", "street"],
    "facing_overbet_bluffcatch_allin": [
        "facing_bet_or_raise", "amount_to_call_bb", "pot_before_bb",
    ],
    "exploit_overlay_candidates": [
        "active_opponent_canonical_player_ids_before",
        "active_opponent_count_before",
        "actor_profile_status",
    ],
}


def _schema_lane(root: str, lane: int) -> dict[str, Any]:
    decision_path, split_path, _ = _lane_paths(Path(root), lane)
    decision_file = pq.ParquetFile(decision_path)
    split_file = pq.ParquetFile(split_path)
    return {
        "decision_columns": list(decision_file.schema_arrow.names),
        "split_columns": list(split_file.schema_arrow.names),
        "decision_rows": int(decision_file.metadata.num_rows),
        "split_rows": int(split_file.metadata.num_rows),
    }


def run_review_screens(root: Path, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    lanes = _parallel_lanes(_schema_lane, root, workers)
    schemas = [set(lane["decision_columns"]) | set(lane["split_columns"]) for lane in lanes]
    common = set.intersection(*schemas)
    if any(int(lane["decision_rows"]) != int(lane["split_rows"]) for lane in lanes):
        raise RuntimeError("review screen found lane row-count mismatch")
    screens = {}
    for task, required in REVIEW_TASKS.items():
        missing = sorted(set(required) - common)
        screens[task] = {
            "required_causal_context_columns": required,
            "missing_columns": missing,
            "rebuild_eligible": not missing,
            "disposition": "REBUILD_SCREEN_ELIGIBLE" if not missing else "RETIRE_CURRENT_BYTES_NO_CAUSAL_REBUILD_INPUT",
        }
    candidate = seal_document({
        "schema_version": "stage6_review_concept_screen_candidate_v1",
        "screens": screens,
        "eligible_concept_count": sum(int(row["rebuild_eligible"]) for row in screens.values()),
        "current_review_model_bytes_disposition": "retire_current_bytes",
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    report = seal_document({
        "schema_version": "stage6_review_concept_screen_report_v1",
        "status": "PASS_SIX_CAUSAL_CONCEPT_SCREENS",
        "logical_lane_count": LANE_COUNT,
        "accepted_decision_rows_scanned": sum(int(lane["decision_rows"]) for lane in lanes),
        "common_materialized_columns": sorted(common),
        "screen_results": screens,
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    return candidate, report


def _reference_lane(package_root_text: str, lane: int) -> dict[str, Any]:
    from stage6_reference_router import ReferencePolicyIndex, ReferenceQuery, load_jsonl

    package_root = Path(package_root_text)
    rows = load_jsonl(package_root / "resources" / "stage050c_primary_reference_rows.jsonl")
    index = ReferencePolicyIndex(rows)
    selected = [row for position, row in enumerate(rows) if position % LANE_COUNT == lane]
    matches = 0
    runtime_abstentions = 0
    for row in selected:
        key = row["information_state_key"].split(":")
        parsed = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in key[2:]}
        query = ReferenceQuery(
            table_format=row["table_format"],
            street=row["street"],
            position=row["position"],
            hand_bucket=row["hand_bucket"],
            board_texture=row["board_texture"],
            history=row["history"],
            pot_bb=float(parsed["pot"].removesuffix("bb")),
            stack_bb=float(parsed["stack"].removesuffix("bb")),
            legal_actions=tuple(row["legal_actions"]),
        )
        evaluation = index.lookup(query, mode="evaluation")
        runtime = index.lookup(query, mode="runtime")
        if evaluation.get("status") != "MATCH" or evaluation.get("row_id") != row["row_id"] or evaluation.get("policy") != row["reference_policy_probabilities"]:
            raise RuntimeError("exact reference evaluation parity failed")
        if runtime.get("status") != "ABSTAIN":
            raise RuntimeError("unpromoted reference row did not abstain at runtime")
        matches += 1
        runtime_abstentions += 1
    return {"rows": len(selected), "exact_matches": matches, "runtime_abstentions": runtime_abstentions}


def run_reference_and_river(package_root: Path, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = package_root / "resources" / "stage050c_primary_reference_rows.jsonl"
    policy_path = package_root / "resources" / "stage021_archive_primary_policy.json"
    river_path = package_root / "resources" / "stage1_river_strict_2x2_substitution_manifest.json"
    expected = {
        reference_path: "9bed485ac4c0f9cc6990b4dbfc9837d214b1c7af82caf6bacbc1180b0e23bf8c",
        policy_path: "de0dcc2928dfb1e1665f394145ed0a8f101f75f67d7591936d4acb8784caf3e4",
        river_path: "be0cc7f367dea1c5651ce1b58e2501f416b56b0c192bd320da7854abeacb11d8",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"frozen reference resource mismatch: {path.name}")
    lanes = _parallel_lanes(_reference_lane, package_root, workers)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    river = json.loads(river_path.read_text(encoding="utf-8"))
    counts = river["counts"]
    roots = river["stage1_roots"]
    if len(roots) != 67 or counts["strict_stage1_selected_lane_count"] != 268 or counts["active_river_root_count"] != 80:
        raise RuntimeError("strict River root inventory changed")
    manifest_receipts = {
        receipt["lane_id"]: receipt["policy_sha256"]
        for root in roots for receipt in root["selected_lane_receipts"]
    }
    policy_receipts = {
        receipt["lane_id"]: receipt["policy_sha256"]
        for receipt in policy["source_inventory"]["river_stage1_strict_2x2"]["selected_lane_receipts"]
    }
    if manifest_receipts != policy_receipts or len(manifest_receipts) != 268:
        raise RuntimeError("compiled River policy lane receipts do not match the strict manifest")
    support = {
        root["root_id"]: min(root["selected_cfr_plus_lane_count"] / 2.0, root["selected_dcfr_lane_count"] / 2.0)
        for root in roots
    }
    candidate = seal_document({
        "schema_version": "stage6_support_scored_reference_router_candidate_v1",
        "exact_stage050c_reference_rows": 440,
        "strict_river_root_support": support,
        "strict_river_root_count": 67,
        "strict_river_lane_count": 268,
        "active_river_root_count": 80,
        "route_rule": "exact_state_match_and_support_score_at_least_1_else_abstain",
        "runtime_authority": False,
        "test_rows_used": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    report = seal_document({
        "schema_version": "stage6_reference_and_river_parity_report_v1",
        "status": "PASS_EXACT_REFERENCE_AND_STRICT_RIVER_PARITY",
        "logical_lane_count": LANE_COUNT,
        "reference_row_count": sum(int(lane["rows"]) for lane in lanes),
        "exact_reference_match_count": sum(int(lane["exact_matches"]) for lane in lanes),
        "unpromoted_runtime_abstention_count": sum(int(lane["runtime_abstentions"]) for lane in lanes),
        "strict_river_lane_receipt_parity_count": len(manifest_receipts),
        "source_hashes": {path.name: digest for path, digest in expected.items()},
        "test_selection_event_count": 0,
        "private_lockbox_opened": False,
        "promotion_authority": False,
    })
    return candidate, report


def _fixed_zip_write(target: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def publish_result(
    stage_drive_root: Path,
    token: str,
    candidate: Mapping[str, Any],
    report: Mapping[str, Any],
    worker_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_document(candidate)
    validate_document(report)
    controller = seal_document({
        "schema_version": "stage6_notebook02_07_controller_receipt_v1",
        "status": "PASS",
        "notebook_token": token,
        "notebook01_run_identity": NOTEBOOK01_RUN_ID,
        "notebook01_split_assignment_root_digest": NOTEBOOK01_SPLIT_ROOT,
        "logical_lane_count": LANE_COUNT,
        "worker_resolution": dict(worker_report),
        "candidate_self_sha256": candidate["self_sha256"],
        "validation_report_self_sha256": report["self_sha256"],
        "private_lockbox_opened": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "live_action_authority": False,
    })
    payloads = [
        ("candidate.json", canonical_json_bytes(candidate) + b"\n"),
        ("validation_report.json", canonical_json_bytes(report) + b"\n"),
        ("controller_receipt.json", canonical_json_bytes(controller) + b"\n"),
    ]
    manifest = seal_document({
        "schema_version": "stage6_notebook_result_manifest_v1",
        "notebook_token": token,
        "member_order": [name for name, _ in payloads],
        "members": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in payloads
        ],
    })
    payloads.append(("RESULT_MANIFEST.json", canonical_json_bytes(manifest) + b"\n"))
    temporary = Path(tempfile.gettempdir()) / f"stage6-notebook{token}-{controller['self_sha256'][:16]}.zip"
    _fixed_zip_write(temporary, payloads)
    result_sha = sha256_file(temporary)
    destination_root = stage_drive_root / "returned_results"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{token}_Stage6_Notebook{token}_{controller['self_sha256'][:16]}_{result_sha[:16]}.zip"
    if destination.exists():
        if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
            raise RuntimeError("existing returned result has different bytes")
    else:
        with temporary.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
    if destination.stat().st_size != temporary.stat().st_size or sha256_file(destination) != result_sha:
        raise RuntimeError("returned-result Drive readback mismatch")
    temporary.unlink(missing_ok=True)
    return {
        "status": "PASS",
        "path": str(destination),
        "name": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": result_sha,
        "controller_receipt_self_sha256": controller["self_sha256"],
        "result_manifest_self_sha256": manifest["self_sha256"],
        "private_lockbox_opened": False,
        "promotion_authority": False,
    }


def run_notebook(
    token: str,
    stage_drive_root: Path,
    package_root: Path,
    requested_workers: str = "auto",
) -> dict[str, Any]:
    if token not in {"02", "03", "04", "05", "06", "07"}:
        raise RuntimeError(f"unsupported Stage 6 notebook token: {token}")
    if "OneDrive" in str(stage_drive_root) or "OneDrive" in str(package_root):
        raise RuntimeError("historical OneDrive checkout is forbidden")
    root = materialization_root(stage_drive_root)
    worker_report = resolve_workers(requested_workers)
    workers = int(worker_report["live_workers"])
    print(json.dumps({"event": "stage6_notebook_start", "token": token, "worker_resolution": worker_report}), flush=True)
    if token == "02":
        candidate, report = run_behavior_head(root, workers, "preflop")
    elif token == "03":
        candidate, report = run_behavior_head(root, workers, "postflop")
    elif token == "04":
        candidate, report = run_size_head(root, workers)
    elif token == "05":
        candidate, report = run_profile_support(root, package_root, workers)
    elif token == "06":
        candidate, report = run_review_screens(root, workers)
    else:
        candidate, report = run_reference_and_river(package_root, workers)
    result = publish_result(stage_drive_root, token, candidate, report, worker_report)
    print(json.dumps({"event": "stage6_notebook_complete", "token": token, **result}, sort_keys=True, indent=2), flush=True)
    return result
