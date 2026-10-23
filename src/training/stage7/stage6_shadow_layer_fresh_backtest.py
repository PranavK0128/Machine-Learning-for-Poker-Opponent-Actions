"""Frozen-byte, leakage-safe evaluation of the installed Stage 6 layers.

The script scores the already installed v2 packages on decisions after the
2026-08-21 development cutoff.  It never fits, tunes, promotes, or grants an
action-changing recommendation authority.  Historical inputs are used only to
construct strictly prior completed-hand features.
"""

from __future__ import annotations

import hashlib
import argparse
import io
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd


STAGE6_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = STAGE6_ROOT.parents[1]
SRC_ROOT = STAGE6_ROOT / "src"
RUNTIME_ROOT = WORKSPACE / "Assistant Qualified/Assistant Code/runtime"
for module_root in (SRC_ROOT, RUNTIME_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from stage6_additional_exploitation_layers import (  # noqa: E402
    ACTIONS,
    build_additional_exploitation_feature_frames,
)
from stage6_behavior_tournament import (  # noqa: E402
    feature_matrix as behavior_feature_matrix,
    feature_names as behavior_feature_names,
)
from stage6_causal_profile_snapshot import build_reference_causal_snapshots  # noqa: E402
from stage6_decision_row_builder import (  # noqa: E402
    build_decision_row,
    classify_source_quarantine_reason,
)
from stage6_expanded_feature_freeze import BASE_FEATURE_COLUMNS  # noqa: E402
from stage6_expanded_layers import build_line_prefix_features  # noqa: E402
from stage6_expanded_tournament import expanded_model_matrix  # noqa: E402
from ml.inference.stage6_retrospective_nested import (  # noqa: E402
    SUPPORT_FEATURES,
    Stage6RetrospectiveNestedRuntime,
)


EVALUATION_ROOT = STAGE6_ROOT / "evidence/shadow_layer_evaluation_20260829"
FRESH_ACTIONS = EVALUATION_ROOT / "fresh-action-state-decision-rows.csv"
FRESH_IDENTITIES = EVALUATION_ROOT / "fresh-decision-time-identity.csv"
FRESH_OBSERVATIONS = EVALUATION_ROOT / "fresh-hand-profile-observations.csv"
HISTORICAL_ACTIONS = STAGE6_ROOT / "data/exports/ml/action-state-decision-rows.csv"
HISTORICAL_IDENTITIES = STAGE6_ROOT / "data/exports/ml/stage6-action-decision-time-identity.csv"
HISTORICAL_OBSERVATIONS = STAGE6_ROOT / "data/exports/ml/stage6-hand-profile-observations.csv"
LABEL_ZIP = STAGE6_ROOT / "inputs/POKERNOW_ASSISTANT_STAGE6_ADDITIONAL_EXPLOITATION_SOURCE_INPUTS_R1.zip"
ARTIFACT_ROOT = RUNTIME_ROOT / "ml/artifacts/stage6_retrospective_nested_v2"
INSTALL_MANIFEST = ARTIFACT_ROOT / "install_manifest.json"
EXPECTED_INSTALL_MANIFEST_SHA256 = "1f5c2aa62097cc815515d6c99e5128109586d9ccae35ed6488a7679b9668682c"
EXPECTED_INSTALL_MANIFEST_SELF_SHA256 = "86da8e38033e95a69a48201ba208ba5a31ec86f42658d48b28e1081ffb8a25a1"
REGISTRY_PATH = STAGE6_ROOT / "contracts/stage6_profile_metric_registry_hold_v1.json"
PREPARED_DECISIONS = EVALUATION_ROOT / "prepared-fresh-decisions.parquet"
PREPARED_HISTORY = EVALUATION_ROOT / "prepared-completed-history.parquet"
PREPARATION_RECEIPT = EVALUATION_ROOT / "stage6_shadow_backtest_preparation_receipt.json"
HERO_POKERNOW_ID = "auGRocdUlB"
DEVELOPMENT_CUTOFF = pd.Timestamp("2026-08-21T00:00:00Z")
FINAL_AUDIT_CUTOFF = pd.Timestamp("2026-08-27T00:00:00Z")
CHUNK_ROWS = 50_000
BOOTSTRAP_REPLICATES = 1_000
MINIMUM_GAME_CLUSTERS_FOR_NEW_PREDICTIVE_QUALIFICATION = 20


def configure_run(arguments: list[str] | None = None) -> argparse.Namespace:
    """Keep prior evidence immutable; new cumulative exports need a new run root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-input-root", type=Path, default=EVALUATION_ROOT,
                        help="Canonical all-player CSV export directory, not live-worker hero payloads")
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT,
                        help="Separate versioned output directory; existing results are never overwritten")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Authenticate and prepare causal rows without scoring or promoting")
    options = parser.parse_args(arguments)
    global FRESH_ACTIONS, FRESH_IDENTITIES, FRESH_OBSERVATIONS
    global PREPARED_DECISIONS, PREPARED_HISTORY, PREPARATION_RECEIPT
    options.fresh_input_root = options.fresh_input_root.resolve()
    options.evaluation_root = options.evaluation_root.resolve()
    FRESH_ACTIONS = options.fresh_input_root / "fresh-action-state-decision-rows.csv"
    FRESH_IDENTITIES = options.fresh_input_root / "fresh-decision-time-identity.csv"
    FRESH_OBSERVATIONS = options.fresh_input_root / "fresh-hand-profile-observations.csv"
    PREPARED_DECISIONS = options.evaluation_root / "prepared-fresh-decisions.parquet"
    PREPARED_HISTORY = options.evaluation_root / "prepared-completed-history.parquet"
    PREPARATION_RECEIPT = options.evaluation_root / "stage6_shadow_backtest_preparation_receipt.json"
    return options


def source_lineage() -> dict[str, Any]:
    paths = (FRESH_ACTIONS, FRESH_IDENTITIES, FRESH_OBSERVATIONS, HISTORICAL_ACTIONS,
             HISTORICAL_IDENTITIES, HISTORICAL_OBSERVATIONS, LABEL_ZIP, REGISTRY_PATH,
             Path(__file__), SRC_ROOT / "stage6_causal_profile_snapshot.py",
             SRC_ROOT / "stage6_decision_row_builder.py", SRC_ROOT / "stage6_additional_exploitation_layers.py",
             SRC_ROOT / "stage6_behavior_tournament.py", SRC_ROOT / "stage6_expanded_layers.py",
             SRC_ROOT / "stage6_expanded_feature_freeze.py", SRC_ROOT / "stage6_expanded_tournament.py")
    return {
        "schema_version": "stage6_shadow_backtest_source_lineage_v1",
        "sources": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)} for path in paths],
        "development_cutoff_utc": DEVELOPMENT_CUTOFF.isoformat(),
        "final_audit_cutoff_utc": FINAL_AUDIT_CUTOFF.isoformat(),
        "cohort": "canonical_all_player_decisions",
        "history_rule": "hand_completed_utc_strictly_before_each_decision_utc",
        "minimum_game_clusters": MINIMUM_GAME_CLUSTERS_FOR_NEW_PREDICTIVE_QUALIFICATION,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def validate_cache_lineage(preparation: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    validate_seal(preparation, label="shadow-backtest preparation receipt")
    if preparation.get("source_lineage") != expected:
        raise RuntimeError("preparation cache does not match current input/code/protocol; use a new evaluation root")

EXPECTED_LAYERS = (
    "within_hand_line_surprise",
    "postflop_behavior",
    "joint_action_size_surprise",
    "latent_player_archetype_shrinkage",
    "multiway_target_response",
    "opponent_bluff_likelihood",
    "player_strategy_drift",
    "showdown_aware_range_calibration",
    "supported_fold_response_bluff_opportunity",
)
MODEL_MEMBER_BY_LAYER = {
    layer: "lightgbm_baseline.joblib" if layer == "postflop_behavior" else "model.joblib"
    for layer in EXPECTED_LAYERS
}

ACTION_HISTORY_COLUMNS = (
    "hand_action_id",
    "hand_id",
    "game_id",
    "event_index",
    "street",
    "active_opponent_canonical_player_ids_before",
    "active_opponent_count_before",
    "facing_bet_or_raise",
    "action_label",
    "is_all_in",
    "big_blind_raw",
    "can_fold",
    "can_check",
    "amount_to_call_raw",
    "stack_before_raw",
    "legal_min_bet_to_raw",
    "legal_min_raise_to_raw",
    "pot_before_bb",
    "street_contribution_before_raw",
    "street_contribution_after_raw",
)
IDENTITY_HISTORY_COLUMNS = (
    "hand_action_id",
    "hand_id",
    "game_id",
    "event_index",
    "decision_utc",
    "actor_pokernow_player_id",
    "canonical_group_player_id",
    "identity_grouping_status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def progress(phase: str, **values: Any) -> None:
    print(json.dumps({"event": "stage6_shadow_backtest_progress", "phase": phase, **values}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def seal(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def validate_seal(document: Mapping[str, Any], *, label: str = "sealed document") -> None:
    candidate = dict(document)
    expected = candidate.pop("self_sha256", None)
    if expected != hashlib.sha256(canonical_bytes(candidate)).hexdigest():
        raise RuntimeError(f"{label} self seal mismatch")


def authenticate_frozen_installation() -> tuple[dict[str, Any], dict[str, bytes]]:
    """Authenticate every installed member before deserializing any estimator.

    Model bytes are retained from the authenticated read and later deserialized
    from memory.  This closes both incomplete-inventory and path re-read gaps.
    """

    manifest_raw = INSTALL_MANIFEST.read_bytes()
    manifest_raw_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_raw_sha256 != EXPECTED_INSTALL_MANIFEST_SHA256:
        raise RuntimeError("Stage 6 retrospective install manifest raw SHA-256 mismatch")
    manifest = json.loads(manifest_raw)
    validate_seal(manifest, label="Stage 6 retrospective install manifest")
    if manifest.get("self_sha256") != EXPECTED_INSTALL_MANIFEST_SELF_SHA256:
        raise RuntimeError("Stage 6 retrospective install manifest trusted self SHA-256 mismatch")
    expected_top = {
        "schema_version": "stage6_retrospective_nested_install_manifest_v2",
        "status": "PASS_VERSIONED_SHADOW_ARTIFACTS_PREPARED",
        "version": "stage6_retrospective_nested_v2",
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Stage 6 install manifest {key} mismatch")
    for key in (
        "automatic_pokernow_action",
        "direct_action_authority",
        "live_action_authority",
        "stage21_strategy_authority_changed",
    ):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Stage 6 install manifest authority boundary violated: {key}")
    if tuple(manifest.get("qualified_layers") or ()) != EXPECTED_LAYERS:
        raise RuntimeError("Stage 6 install manifest qualified-layer inventory/order mismatch")

    rows = manifest.get("layers")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_LAYERS):
        raise RuntimeError("Stage 6 install manifest must contain exactly nine layer rows")
    by_layer = {str(row.get("layer")): row for row in rows if isinstance(row, Mapping)}
    if len(by_layer) != len(rows) or tuple(by_layer) != EXPECTED_LAYERS:
        raise RuntimeError("Stage 6 install manifest layer rows are duplicated, missing, or reordered")

    authenticated_models: dict[str, bytes] = {}
    layer_inventory: dict[str, Any] = {}
    for layer in EXPECTED_LAYERS:
        row = by_layer[layer]
        if row.get("artifact_directory") != layer:
            raise RuntimeError(f"Stage 6 install manifest artifact directory mismatch: {layer}")
        layer_dir = ARTIFACT_ROOT / layer
        if not layer_dir.is_dir():
            raise RuntimeError(f"Stage 6 installed layer directory missing: {layer}")
        for member in layer_dir.rglob("*"):
            if member.is_symlink():
                raise RuntimeError(f"Stage 6 installed layer contains a symlink: {member}")

        declared = row.get("files")
        if not isinstance(declared, Mapping) or not declared:
            raise RuntimeError(f"Stage 6 install manifest has no member inventory: {layer}")
        expected_members: dict[str, Mapping[str, Any]] = {}
        for declared_name, identity in declared.items():
            if not isinstance(identity, Mapping):
                raise RuntimeError(f"Stage 6 install member identity is invalid: {layer}/{declared_name}")
            relative_path = str(identity.get("relative_path") or "")
            parsed = PurePosixPath(relative_path)
            if (
                not relative_path
                or parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in relative_path
                or str(declared_name) != relative_path
                or relative_path in expected_members
            ):
                raise RuntimeError(f"Stage 6 install member path is unsafe or duplicated: {layer}/{relative_path}")
            expected_members[relative_path] = identity

        actual_members = sorted(
            member.relative_to(layer_dir).as_posix()
            for member in layer_dir.rglob("*")
            if member.is_file()
        )
        if sorted(expected_members) != actual_members:
            raise RuntimeError(f"Stage 6 installed member inventory mismatch: {layer}")

        authenticated_member_identities: dict[str, Any] = {}
        for relative_path in actual_members:
            identity = expected_members[relative_path]
            member_bytes = (layer_dir / Path(*PurePosixPath(relative_path).parts)).read_bytes()
            observed_sha256 = hashlib.sha256(member_bytes).hexdigest()
            if len(member_bytes) != int(identity.get("bytes", -1)):
                raise RuntimeError(f"Stage 6 installed member byte count mismatch: {layer}/{relative_path}")
            if observed_sha256 != identity.get("sha256"):
                raise RuntimeError(f"Stage 6 installed member SHA-256 mismatch: {layer}/{relative_path}")
            authenticated_member_identities[relative_path] = {
                "bytes": len(member_bytes),
                "sha256": observed_sha256,
            }
            if relative_path == MODEL_MEMBER_BY_LAYER[layer]:
                authenticated_models[layer] = member_bytes
        if layer not in authenticated_models:
            raise RuntimeError(f"Stage 6 authenticated model member is missing: {layer}")
        layer_inventory[layer] = {
            "artifact_directory": layer,
            "member_count": len(authenticated_member_identities),
            "model_member": MODEL_MEMBER_BY_LAYER[layer],
            "members": authenticated_member_identities,
        }

    evidence = {
        "status": "PASS_EXACT_COMPLETE_INSTALL_INVENTORY_AUTHENTICATED_BEFORE_MODEL_DESERIALIZATION",
        "manifest_relative_path": INSTALL_MANIFEST.relative_to(WORKSPACE).as_posix(),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": manifest_raw_sha256,
        "manifest_self_sha256": manifest["self_sha256"],
        "authenticated_before_model_deserialization": True,
        "exact_layer_inventory_enforced": True,
        "exact_member_bytes_and_sha256_enforced": True,
        "deserialization_uses_authenticated_in_memory_model_bytes": True,
        "layer_inventory": layer_inventory,
    }
    return evidence, authenticated_models


def strict_bool(value: Any) -> bool:
    if type(value) in {bool, np.bool_}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RuntimeError(f"invalid strict boolean value: {value!r}")


def normalized_action(row: Mapping[str, Any]) -> str:
    if strict_bool(row["is_all_in"]):
        return "all_in"
    source = str(row["action_label"])
    if source == "bet_or_raise":
        return "bet" if float(row["amount_to_call_raw"]) == 0.0 else "raise"
    if source not in ACTIONS:
        raise RuntimeError(f"invalid source action: {source}")
    return source


def load_fresh_source() -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    actions = pd.read_csv(
        FRESH_ACTIONS, dtype=str, keep_default_na=False, low_memory=False
    )
    identities = pd.read_csv(
        FRESH_IDENTITIES, dtype=str, keep_default_na=False, low_memory=False
    )
    if len(actions) != len(identities):
        raise RuntimeError("fresh action/identity row counts differ")
    for column in ("hand_action_id", "hand_id", "game_id", "event_index"):
        if not actions[column].astype(str).equals(identities[column].astype(str)):
            raise RuntimeError(f"fresh action/identity order differs: {column}")
    for frame, label in ((actions, "actions"), (identities, "identities")):
        if frame["hand_action_id"].eq("").any() or frame["hand_action_id"].duplicated().any():
            raise RuntimeError(f"duplicate or empty fresh decision identity: {label}")
    if actions.duplicated(["game_id", "hand_id", "event_index"]).any():
        raise RuntimeError("duplicate game/hand/action occurrence in fresh source")
    merged = actions.copy()
    for column in identities.columns:
        if column not in merged.columns:
            merged[column] = identities[column].to_numpy()
    if not merged["identity_grouping_status"].eq("resolved_retrospective_group").all():
        raise RuntimeError("fresh identity source contains unresolved actors")
    hero_rows = merged.loc[merged["actor_pokernow_player_id"].eq(HERO_POKERNOW_ID), "canonical_group_player_id"]
    hero_ids = sorted(set(hero_rows.astype(str)))
    if len(hero_ids) != 1:
        raise RuntimeError(f"hero canonical identity is not unique: {hero_ids}")
    return merged, merged.to_dict("records"), hero_ids[0]


def load_profile_history(actor_ids: set[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str], dict[str, str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    required_terms = sorted({
        term
        for binding in registry["observation_bindings"]
        for field in ("numerator_terms", "opportunity_terms")
        for term in binding[field]
    })
    columns = [
        "source_row_ordinal", "hand_id", "game_id", "hand_completed_utc",
        "hand_completed_epoch_ms", "pokernow_player_id", "canonical_group_player_id",
        "identity_grouping_status", "observed_hand", "preflop_flags_present",
        "postflop_flags_present", *required_terms,
    ]
    causal_rows: list[dict[str, Any]] = []
    completion_by_hand: dict[tuple[str, str], str] = {}
    canonical_by_source: dict[str, str] = {}
    current_identity_rows = pd.read_csv(
        FRESH_OBSERVATIONS,
        usecols=["pokernow_player_id", "canonical_group_player_id"],
    ).drop_duplicates()
    if current_identity_rows["pokernow_player_id"].astype(str).duplicated().any():
        raise RuntimeError("fresh observation identity crosswalk is not one-to-one")
    current_canonical_by_source = {
        str(source_id): str(canonical_id)
        for source_id, canonical_id in current_identity_rows.itertuples(index=False, name=None)
    }
    identity_drift_rows = 0
    ordinal = 0
    for source in (HISTORICAL_OBSERVATIONS, FRESH_OBSERVATIONS):
        for chunk in pd.read_csv(source, usecols=columns, chunksize=CHUNK_ROWS, low_memory=False):
            if not chunk["identity_grouping_status"].eq("resolved_retrospective_group").all():
                raise RuntimeError(f"unresolved identity in observation source: {source}")
            remapped = chunk["pokernow_player_id"].astype(str).map(current_canonical_by_source)
            changed = remapped.notna() & remapped.ne(chunk["canonical_group_player_id"].astype(str))
            identity_drift_rows += int(changed.sum())
            chunk.loc[remapped.notna(), "canonical_group_player_id"] = remapped.loc[remapped.notna()]
            for game_id, hand_id, completion in chunk[["game_id", "hand_id", "hand_completed_utc"]].drop_duplicates().itertuples(index=False, name=None):
                key = (str(game_id), str(hand_id))
                observed = completion_by_hand.setdefault(key, str(completion))
                if observed != str(completion):
                    raise RuntimeError(f"inconsistent hand completion: {key}")
            for source_id, canonical_id in chunk[["pokernow_player_id", "canonical_group_player_id"]].drop_duplicates().itertuples(index=False, name=None):
                canonical_by_source[str(source_id)] = current_canonical_by_source.get(
                    str(source_id), str(canonical_id)
                )
            selected = chunk.loc[chunk["canonical_group_player_id"].astype(str).isin(actor_ids)].copy()
            for column in (
                "observed_hand",
                "preflop_flags_present",
                "postflop_flags_present",
                *required_terms,
            ):
                selected[column] = [
                    ""
                    if pd.isna(value)
                    else int(value)
                    if float(value).is_integer()
                    else value
                    for value in selected[column]
                ]
            selected["source_row_ordinal"] = np.arange(ordinal, ordinal + len(selected), dtype=np.int64)
            ordinal += len(selected)
            causal_rows.extend(selected.to_dict("records"))
        progress(
            "load_profile_history",
            source=source.name,
            selected_rows=len(causal_rows),
            completed_hands=len(completion_by_hand),
            retrospectively_remapped_identity_rows=identity_drift_rows,
        )
    return causal_rows, completion_by_hand, canonical_by_source


def build_fresh_decisions(fresh_records: list[dict[str, Any]], causal_observations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    causal_decisions = [{
        "source_row_ordinal": int(row["source_row_ordinal"]),
        "canonical_group_player_id": str(row["canonical_group_player_id"]),
        "actor_pokernow_player_id": str(row["actor_pokernow_player_id"]),
        "decision_utc": str(row["decision_utc"]),
        "decision_epoch_ms": int(row["decision_epoch_ms"]),
    } for row in fresh_records]
    snapshot = build_reference_causal_snapshots(causal_decisions, causal_observations, registry)
    binding_by_ordinal = {int(row["source_row_ordinal"]): row for row in snapshot["decision_bindings"]}
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in fresh_records:
        reason = classify_source_quarantine_reason(row)
        if reason is not None:
            quarantine.append({
                "source_row_ordinal": int(row["source_row_ordinal"]),
                "decision_id": str(row["hand_action_id"]),
                "reason": reason,
            })
            continue
        accepted.append(build_decision_row(row, row, binding_by_ordinal[int(row["source_row_ordinal"])]))
    frame = pd.DataFrame(accepted).sort_values(["decision_utc", "game_id", "hand_id", "decision_order", "row_id"], kind="mergesort").reset_index(drop=True)
    if pd.to_datetime(frame["decision_utc"], utc=True).min() < DEVELOPMENT_CUTOFF:
        raise RuntimeError("pre-cutoff decision entered fresh evaluation")
    progress("fresh_decisions", accepted=len(frame), quarantined=len(quarantine), profile_snapshots=len(snapshot["profile_snapshots"]))
    return frame, pd.DataFrame(quarantine)


def load_exact_labels(canonical_by_source: Mapping[str, str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    labels: dict[tuple[str, str, str], dict[str, Any]] = {}
    with zipfile.ZipFile(LABEL_ZIP) as archive:
        with archive.open("stage6_additional_revealed_hand_labels.jsonl") as handle:
            for raw in handle:
                row = json.loads(raw)
                if row.get("derivation_status") != "exact_completed_revealed_single_board_holdem_v1":
                    continue
                if row.get("support_status") != "supported":
                    continue
                if row.get("known_hole_card_count") != 2 or row.get("public_board_card_count") != 5:
                    continue
                canonical = canonical_by_source.get(str(row["pokernow_player_id"]))
                if canonical is None:
                    continue
                key = (str(row["game_id"]), str(row["hand_id"]), canonical)
                labels[key] = row
    progress("exact_labels", supported_player_hands=len(labels))
    return labels


def aligned_source_chunks(action_path: Path, identity_path: Path) -> Iterable[pd.DataFrame]:
    action_reader = pd.read_csv(action_path, usecols=list(ACTION_HISTORY_COLUMNS), chunksize=CHUNK_ROWS, low_memory=False)
    identity_reader = pd.read_csv(identity_path, usecols=list(IDENTITY_HISTORY_COLUMNS), chunksize=CHUNK_ROWS, low_memory=False)
    action_count = identity_count = 0
    for actions, identities in zip(action_reader, identity_reader, strict=True):
        if len(actions) != len(identities):
            raise RuntimeError(f"aligned source chunk differs: {action_path.name}")
        for column in ("hand_action_id", "hand_id", "game_id", "event_index"):
            if not actions[column].astype(str).equals(identities[column].astype(str)):
                raise RuntimeError(f"aligned source differs at {column}: {action_path.name}")
        if not identities["identity_grouping_status"].eq("resolved_retrospective_group").all():
            raise RuntimeError(f"unresolved action identity: {identity_path.name}")
        merged = actions.copy()
        for column in identities.columns:
            if column not in merged:
                merged[column] = identities[column].to_numpy()
        action_count += len(actions)
        identity_count += len(identities)
        yield merged
    if action_count != identity_count:
        raise RuntimeError("action/identity source conservation failed")


def build_completed_history(
    completion_by_hand: Mapping[tuple[str, str], str],
    labels: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[pd.DataFrame, Counter[str]]:
    output: list[dict[str, Any]] = []
    quarantine: Counter[str] = Counter()
    processed = 0
    for action_path, identity_path in (
        (HISTORICAL_ACTIONS, HISTORICAL_IDENTITIES),
        (FRESH_ACTIONS, FRESH_IDENTITIES),
    ):
        current_hand: tuple[str, str] | None = None
        last_aggressor: dict[str, str] = {}
        last_size: dict[str, float] = {}
        for chunk in aligned_source_chunks(action_path, identity_path):
            for row in chunk.to_dict("records"):
                processed += 1
                reason = classify_source_quarantine_reason(row)
                if reason is not None:
                    quarantine[reason] += 1
                    continue
                game_hand = (str(row["game_id"]), str(row["hand_id"]))
                if game_hand != current_hand:
                    current_hand = game_hand
                    last_aggressor = {}
                    last_size = {}
                street = str(row["street"])
                actor_id = str(row["canonical_group_player_id"])
                action = normalized_action(row)
                opponents = json.loads(str(row["active_opponent_canonical_player_ids_before"]))
                facing = strict_bool(row["facing_bet_or_raise"])
                prior_aggressor = last_aggressor.get(street)
                facing_size = (
                    last_size.get(street)
                    if facing and prior_aggressor is not None and prior_aggressor in opponents
                    else None
                )
                action_size = None
                if action in {"bet", "raise", "all_in"}:
                    pot = float(row["pot_before_bb"])
                    contribution = (
                        float(row["street_contribution_after_raw"])
                        - float(row["street_contribution_before_raw"])
                    ) / float(row["big_blind_raw"])
                    if pot <= 0.0 or contribution <= 0.0:
                        raise RuntimeError("aggressive history action lacks a positive exact size")
                    action_size = contribution / pot
                    last_aggressor[street] = actor_id
                    last_size[street] = action_size
                completion = completion_by_hand.get(game_hand)
                if completion is None:
                    raise RuntimeError(f"missing completion for {game_hand}")
                label = labels.get((game_hand[0], game_hand[1], actor_id))
                output.append({
                    "game_id": game_hand[0],
                    "hand_id": game_hand[1],
                    "decision_utc": str(row["decision_utc"]),
                    "hand_completed_utc": completion,
                    "actor_id": actor_id,
                    "street": street,
                    "action_label": action,
                    "facing_bet_or_raise": facing,
                    "active_opponent_count_before": int(row["active_opponent_count_before"]),
                    "action_size_pot_fraction": action_size,
                    "facing_aggression_size_pot_fraction": facing_size,
                    "showdown_revealed": label is not None,
                    "showdown_hand_category_ordinal": (
                        None
                        if label is None
                        else float(label["hand_strength"]["category_rank"]) / 8.0
                    ),
                    "bluff_shown_proxy": None if label is None else label.get("bluff_shown_proxy"),
                    "showdown_missed_draw": None if label is None else label.get("missed_draw"),
                    "showdown_bottom_pair": None if label is None else label.get("bottom_pair"),
                })
            if processed % 250_000 < CHUNK_ROWS:
                progress("completed_history", processed_decisions=processed, accepted_decisions=len(output), quarantined=sum(quarantine.values()))
        progress("completed_history_source", source=action_path.name, accepted_decisions=len(output))
    return pd.DataFrame(output), quarantine


def build_decision_context(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = decisions.sort_values(["game_id", "hand_id", "decision_order", "row_id"], kind="mergesort")
    for (_game_id, _hand_id), hand in ordered.groupby(["game_id", "hand_id"], sort=False, observed=True):
        last_aggressor: dict[str, str] = {}
        for row in hand.to_dict("records"):
            street = str(row["street"])
            opponents = json.loads(str(row["active_opponent_canonical_player_ids_before"]))
            facing = strict_bool(row["facing_bet_or_raise"])
            target = None
            if facing:
                candidate = last_aggressor.get(street)
                if candidate in opponents:
                    target = candidate
            elif len(opponents) == 1:
                target = opponents[0]
            pot = float(row["pot_before_bb"])
            committed = float(row["actor_committed_before_bb"])
            minimum = row["legal_min_raise_to_bb"] if facing else row["legal_min_bet_to_bb"]
            candidate_size = None if pd.isna(minimum) else max(0.0, float(minimum) - committed) / pot
            rows.append({
                "row_id": row["row_id"],
                "game_id": row["game_id"],
                "hand_id": row["hand_id"],
                "decision_utc": row["decision_utc"],
                "actor_id": row["actor_id"],
                "active_opponent_canonical_player_ids_before": row["active_opponent_canonical_player_ids_before"],
                "active_opponent_count_before": row["active_opponent_count_before"],
                "target_opponent_canonical_player_ids_before": None if target is None else json.dumps([target], separators=(",", ":")),
                "target_opponent_count_before": 0 if target is None else 1,
                "candidate_bet_size_pot_fraction": candidate_size,
            })
            if str(row["action_label"]) in {"bet", "raise", "all_in"}:
                last_aggressor[street] = str(row["actor_id"])
    return pd.DataFrame(rows).set_index("row_id").loc[decisions["row_id"].astype(str)].reset_index()


def calibrated_legal_probabilities(package: Mapping[str, Any], matrix: np.ndarray, legal: np.ndarray) -> np.ndarray:
    raw = Stage6RetrospectiveNestedRuntime._probabilities(package, matrix)
    temperature = float(package["temperature"])
    logits = np.log(np.clip(raw, 1e-15, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities[~legal] = 0.0
    totals = probabilities.sum(axis=1)
    missing = totals <= 0.0
    if missing.any():
        probabilities[missing] = np.asarray(package["class_priors_laplace_one"], dtype=float) * legal[missing]
        totals = probabilities.sum(axis=1)
    return probabilities / totals[:, None]


def legal_prior(package: Mapping[str, Any], legal: np.ndarray) -> np.ndarray:
    values = np.asarray(package["class_priors_laplace_one"], dtype=float)[None, :] * legal
    totals = values.sum(axis=1)
    if (totals <= 0.0).any():
        raise RuntimeError("fresh decision has no legal baseline action")
    return values / totals[:, None]


def cluster_ci(game_ids: np.ndarray, improvements: np.ndarray, *, seed: int) -> list[float]:
    groups = pd.DataFrame({"game": game_ids.astype(str), "value": improvements}).groupby("game", sort=True)["value"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(dtype=float)
    counts = groups["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(groups), size=len(groups))
        draws[index] = float(sums[sample].sum() / counts[sample].sum())
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def score_scope(
    *,
    package: Mapping[str, Any],
    probabilities: np.ndarray,
    legal: np.ndarray,
    targets: np.ndarray,
    game_ids: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {"supported_rows": 0, "status": "NO_SUPPORTED_ROWS"}
    predicted = probabilities[indices]
    observed = targets[indices]
    baseline = legal_prior(package, legal[indices])
    model_nll = -np.log(np.clip(predicted[np.arange(len(indices)), observed], 1e-15, 1.0))
    base_nll = -np.log(np.clip(baseline[np.arange(len(indices)), observed], 1e-15, 1.0))
    one_hot = np.eye(len(ACTIONS), dtype=float)[observed]
    model_brier = np.square(predicted - one_hot).sum(axis=1)
    base_brier = np.square(baseline - one_hot).sum(axis=1)
    nll_improvement = base_nll - model_nll
    brier_improvement = base_brier - model_brier
    return {
        "supported_rows": int(len(indices)),
        "game_clusters": int(len(set(game_ids[indices].astype(str)))),
        "accuracy": float((predicted.argmax(axis=1) == observed).mean()),
        "log_loss": float(model_nll.mean()),
        "frozen_prior_log_loss": float(base_nll.mean()),
        "log_loss_improvement": float(nll_improvement.mean()),
        "log_loss_improvement_cluster_bootstrap_95_ci": cluster_ci(game_ids[indices], nll_improvement, seed=seed),
        "multiclass_brier": float(model_brier.mean()),
        "frozen_prior_multiclass_brier": float(base_brier.mean()),
        "brier_improvement": float(brier_improvement.mean()),
        "brier_improvement_cluster_bootstrap_95_ci": cluster_ci(game_ids[indices], brier_improvement, seed=seed + 1),
    }


def evaluate_models(
    decisions: pd.DataFrame,
    context: pd.DataFrame,
    history: pd.DataFrame,
    hero_id: str,
    authenticated_model_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    target = np.asarray([ACTIONS.index(str(value)) for value in decisions["action_label"]], dtype=np.int8)
    legal = decisions[[f"legal_{action}" for action in ACTIONS]].to_numpy(dtype=bool)
    if not legal[np.arange(len(target)), target].all():
        raise RuntimeError("fresh observed action is illegal under the exact mask")
    decision_times = pd.to_datetime(decisions["decision_utc"], utc=True)
    scope_masks = {
        "all_fresh": np.ones(len(decisions), dtype=bool),
        "development_diagnostic_2026_08_21_through_26": (decision_times < FINAL_AUDIT_CUTOFF).to_numpy(),
        "untouched_final_audit_2026_08_27_and_later": (decision_times >= FINAL_AUDIT_CUTOFF).to_numpy(),
        "hero_alias_group_all_fresh": decisions["actor_id"].astype(str).eq(hero_id).to_numpy(),
    }
    behavior_names = behavior_feature_names()
    behavior_matrix = behavior_feature_matrix(decisions)
    common = decisions[["row_id"]].copy()
    for index, name in enumerate(BASE_FEATURE_COLUMNS):
        common[name] = behavior_matrix[:, index]
    line = build_line_prefix_features(decisions)
    line_frame = common.merge(line, on="row_id", how="inner", validate="one_to_one")
    line_names, line_matrix = expanded_model_matrix(line_frame, "within_hand_line_surprise")
    additional = build_additional_exploitation_feature_frames(context, history)

    result: dict[str, Any] = {}
    for layer in [
        "postflop_behavior", "within_hand_line_surprise", "joint_action_size_surprise",
        "latent_player_archetype_shrinkage", "multiway_target_response",
        "opponent_bluff_likelihood", "player_strategy_drift",
        "showdown_aware_range_calibration", "supported_fold_response_bluff_opportunity",
    ]:
        package = joblib.load(io.BytesIO(authenticated_model_bytes[layer]))
        names = list(package["feature_names"])
        if layer == "postflop_behavior":
            matrix = behavior_matrix
            observed_names = behavior_names
            support = decisions["street"].astype(str).isin(("flop", "turn", "river")).to_numpy()
        elif layer == "within_hand_line_surprise":
            matrix = line_matrix
            observed_names = line_names
            support = line_frame["line_prefix_decision_count"].to_numpy(dtype=float) >= 1.0
        else:
            frame = additional[layer].set_index("row_id").loc[decisions["row_id"].astype(str)].reset_index()
            observed_names = names
            matrix = frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
            support_columns = SUPPORT_FEATURES.get(layer, ())
            support = np.logical_or.reduce([frame[column].astype(bool).to_numpy() for column in support_columns])
        if list(observed_names) != names:
            raise RuntimeError(f"frozen feature order differs for {layer}")
        probabilities = calibrated_legal_probabilities(package, matrix, legal)
        scopes = {
            scope: score_scope(
                package=package,
                probabilities=probabilities,
                legal=legal,
                targets=target,
                game_ids=decisions["game_id"].astype(str).to_numpy(),
                mask=support & scope_mask,
                seed=6202608 + index,
            )
            for index, (scope, scope_mask) in enumerate(scope_masks.items())
        }
        final_metrics = scopes["untouched_final_audit_2026_08_27_and_later"]
        ci = final_metrics.get("log_loss_improvement_cluster_bootstrap_95_ci")
        positive_fresh_evidence = isinstance(ci, list) and ci[0] > 0.0
        final_game_clusters = int(final_metrics.get("game_clusters", 0))
        if positive_fresh_evidence and final_game_clusters >= MINIMUM_GAME_CLUSTERS_FOR_NEW_PREDICTIVE_QUALIFICATION:
            predictive_verdict = "SUPPORTED_NEW_INDEPENDENT_PREDICTIVE_QUALIFICATION"
        elif positive_fresh_evidence:
            predictive_verdict = "SUPPORTED_FRESH_PREDICTIVE_EVIDENCE__INSUFFICIENT_GAME_CLUSTERS_FOR_NEW_QUALIFICATION"
        else:
            predictive_verdict = "NOT_PROVEN_ON_UNTOUCHED_FINAL_AUDIT"
        result[layer] = {
            "model_sha256": hashlib.sha256(authenticated_model_bytes[layer]).hexdigest(),
            "family": package["family"],
            "feature_count": len(names),
            "fresh_rows": int(len(decisions)),
            "supported_rows": int(support.sum()),
            "coverage": float(support.mean()),
            "predictive_verdict": predictive_verdict,
            "recommendation_authority": False,
            "live_action_authority": False,
            "scopes": scopes,
        }
        progress("evaluate_layer", layer=layer, supported_rows=int(support.sum()), verdict=predictive_verdict)
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 6 Shadow-Layer Fresh Predictive Evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Fresh decisions: `{report['counts']['accepted_fresh_decisions']:,}`",
        f"- Date range: `{report['fresh_data']['minimum_decision_utc']}` through `{report['fresh_data']['maximum_decision_utc']}`",
        f"- Hero aliases: PeanutJr / Peanut / Nut, PokerNow id `{HERO_POKERNOW_ID}`",
        "- Frozen-model policy: no fitting, tuning, fusion selection, or promotion used fresh targets.",
        f"- New-qualification cluster threshold: at least `{report['scientific_boundary']['minimum_game_clusters_for_new_predictive_qualification']}` independent games; the smallest supported final-slice count is `{report['scientific_boundary']['minimum_observed_final_game_clusters']}`, so this is fresh evidence, not a new scientific qualification.",
        "- Authority: predictive diagnostics only; this backtest does not authorize a shadow layer to change the poker recommendation.",
        "",
        "| Layer | Fresh coverage | Untouched final supported rows | Log-loss improvement vs frozen prior | 95% game-cluster CI | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for layer, row in report["layers"].items():
        final = row["scopes"]["untouched_final_audit_2026_08_27_and_later"]
        improvement = final.get("log_loss_improvement")
        ci = final.get("log_loss_improvement_cluster_bootstrap_95_ci")
        lines.append(
            f"| `{layer}` | {row['coverage']:.2%} | {final.get('supported_rows', 0):,} | "
            f"{('n/a' if improvement is None else f'{improvement:.6f}')} | "
            f"{('n/a' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]')} | `{row['predictive_verdict']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A positive improvement means lower loss than the package's frozen class-prior baseline after the same exact legal-action projection. The confidence interval resamples whole games, not individual decisions.",
        "",
        "Observed-action prediction is not counterfactual poker EV. Fresh predictive evidence cannot be converted into an action weight or ordering without enough independent game clusters and a separately preregistered, leakage-safe fusion/strategy experiment. Until then Recommendation Testing must fall back to the incumbent recommendation while displaying these layer diagnostics.",
        "",
    ])
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> None:
    options = configure_run(arguments)
    started = time.perf_counter()
    for path in (FRESH_ACTIONS, FRESH_IDENTITIES, FRESH_OBSERVATIONS, HISTORICAL_ACTIONS, HISTORICAL_IDENTITIES, HISTORICAL_OBSERVATIONS, LABEL_ZIP, REGISTRY_PATH, INSTALL_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_json = options.evaluation_root / "stage6_shadow_layer_fresh_backtest_v2.json"
    output_md = options.evaluation_root / "STAGE6_SHADOW_LAYER_FRESH_BACKTEST_V2.md"
    if output_json.exists() or output_md.exists():
        raise FileExistsError("refusing to overwrite prior shadow-layer evaluation evidence")
    options.evaluation_root.mkdir(parents=True, exist_ok=True)
    lineage = source_lineage()

    artifact_authentication, authenticated_model_bytes = authenticate_frozen_installation()
    progress(
        "authenticate_frozen_installation",
        manifest_sha256=artifact_authentication["manifest_sha256"],
        layers=len(artifact_authentication["layer_inventory"]),
    )

    fresh, fresh_records, hero_id = load_fresh_source()
    if PREPARED_DECISIONS.is_file() and PREPARED_HISTORY.is_file() and PREPARATION_RECEIPT.is_file():
        preparation = json.loads(PREPARATION_RECEIPT.read_text(encoding="utf-8"))
        validate_cache_lineage(preparation, lineage)
        for path, key in (
            (PREPARED_DECISIONS, "prepared_fresh_decisions"),
            (PREPARED_HISTORY, "prepared_completed_history"),
        ):
            identity = preparation[key]
            if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
                raise RuntimeError(f"prepared shadow-backtest cache identity mismatch: {path.name}")
        decisions = pd.read_parquet(PREPARED_DECISIONS)
        history = pd.read_parquet(PREPARED_HISTORY)
        fresh_quarantine_count = int(preparation["counts"]["quarantined_fresh_decisions"])
        history_quarantine = Counter(preparation["history_quarantine_reason_counts"])
        causal_observation_count = int(preparation["counts"]["causal_profile_observation_rows"])
        exact_label_count = int(preparation["counts"]["exact_revealed_player_hand_labels"])
        progress("load_prepared_cache", decisions=len(decisions), history=len(history))
    elif any(path.exists() for path in (PREPARED_DECISIONS, PREPARED_HISTORY, PREPARATION_RECEIPT)):
        raise RuntimeError("incomplete shadow-backtest preparation cache; manual review required")
    else:
        actor_ids = set(fresh["canonical_group_player_id"].astype(str))
        causal_observations, completion_by_hand, canonical_by_source = load_profile_history(actor_ids)
        decisions, quarantine = build_fresh_decisions(fresh_records, causal_observations)
        labels = load_exact_labels(canonical_by_source)
        history, history_quarantine = build_completed_history(completion_by_hand, labels)
        decisions.to_parquet(PREPARED_DECISIONS, compression="zstd", index=False)
        history.to_parquet(PREPARED_HISTORY, compression="zstd", index=False)
        fresh_quarantine_count = int(len(quarantine))
        causal_observation_count = int(len(causal_observations))
        exact_label_count = int(len(labels))
        preparation = seal({
            "schema_version": "stage6_shadow_backtest_preparation_receipt_v2",
            "source_lineage": lineage,
            "status": "PASS_DURABLE_CAUSAL_REPLAY_PREPARATION",
            "created_at": utc_now(),
            "prepared_fresh_decisions": {
                "bytes": PREPARED_DECISIONS.stat().st_size,
                "sha256": sha256_file(PREPARED_DECISIONS),
                "rows": int(len(decisions)),
            },
            "prepared_completed_history": {
                "bytes": PREPARED_HISTORY.stat().st_size,
                "sha256": sha256_file(PREPARED_HISTORY),
                "rows": int(len(history)),
            },
            "counts": {
                "quarantined_fresh_decisions": fresh_quarantine_count,
                "causal_profile_observation_rows": causal_observation_count,
                "exact_revealed_player_hand_labels": exact_label_count,
            },
            "history_quarantine_reason_counts": dict(sorted(history_quarantine.items())),
            "current_identity_remap_applied": True,
            "direct_action_authority": False,
            "live_action_authority": False,
        })
        if source_lineage() != lineage:
            raise RuntimeError("input/code changed during preparation; no preparation receipt sealed")
        with PREPARATION_RECEIPT.open("xb") as handle:
            handle.write(canonical_bytes(preparation) + b"\n")
        progress("write_prepared_cache", decisions=len(decisions), history=len(history), receipt=preparation["self_sha256"])
    if options.prepare_only:
        progress("prepare_only_complete", promotion=False, receipt=str(PREPARATION_RECEIPT))
        return
    context = build_decision_context(decisions)
    layers = evaluate_models(decisions, context, history, hero_id, authenticated_model_bytes)

    new_independent_predictive_qualification = all(
        row["predictive_verdict"] == "SUPPORTED_NEW_INDEPENDENT_PREDICTIVE_QUALIFICATION"
        for row in layers.values()
    )
    fresh_predictive_evidence_supported = all(
        row["predictive_verdict"].startswith("SUPPORTED_")
        for row in layers.values()
    )
    minimum_observed_final_game_clusters = min(
        int(row["scopes"]["untouched_final_audit_2026_08_27_and_later"].get("game_clusters", 0))
        for row in layers.values()
    )
    if new_independent_predictive_qualification:
        report_status = "PASS_FROZEN_PREDICTIVE_EVIDENCE_COMPLETE__NEW_INDEPENDENT_PREDICTIVE_QUALIFICATION__NO_FUSION_OR_ACTION_AUTHORITY"
    elif fresh_predictive_evidence_supported:
        report_status = "PASS_FROZEN_PREDICTIVE_EVIDENCE_COMPLETE__INSUFFICIENT_GAME_CLUSTERS_FOR_NEW_QUALIFICATION__NO_FUSION_OR_ACTION_AUTHORITY"
    else:
        report_status = "COMPLETE_FROZEN_PREDICTIVE_EVALUATION__ONE_OR_MORE_LAYERS_NOT_PROVEN__NO_FUSION_OR_ACTION_AUTHORITY"

    decision_times = pd.to_datetime(decisions["decision_utc"], utc=True)
    report = seal({
        "schema_version": "stage6_shadow_layer_fresh_backtest_v2",
        "status": report_status,
        "created_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "fresh_data": {
            "cutoff_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "final_audit_cutoff_utc": FINAL_AUDIT_CUTOFF.isoformat(),
            "minimum_decision_utc": decision_times.min().isoformat(),
            "maximum_decision_utc": decision_times.max().isoformat(),
            "game_count": int(decisions["game_id"].nunique()),
            "hand_count": int(decisions["hand_id"].nunique()),
            "hero_pokernow_id": HERO_POKERNOW_ID,
            "hero_canonical_id": hero_id,
        },
        "counts": {
            "raw_fresh_decisions": int(len(fresh)),
            "accepted_fresh_decisions": int(len(decisions)),
            "quarantined_fresh_decisions": fresh_quarantine_count,
            "completed_history_decisions": int(len(history)),
            "completed_history_quarantined_decisions": int(sum(history_quarantine.values())),
            "causal_profile_observation_rows": causal_observation_count,
            "exact_revealed_player_hand_labels": exact_label_count,
        },
        "history_quarantine_reason_counts": dict(sorted(history_quarantine.items())),
        "input_identities": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (FRESH_ACTIONS, FRESH_IDENTITIES, FRESH_OBSERVATIONS, HISTORICAL_ACTIONS, HISTORICAL_IDENTITIES, HISTORICAL_OBSERVATIONS, LABEL_ZIP, REGISTRY_PATH)
        },
        "artifact_authentication": artifact_authentication,
        "source_lineage": lineage,
        "layers": layers,
        "scientific_boundary": {
            "frozen_installed_model_bytes": True,
            "complete_install_manifest_authenticated_before_model_deserialization": True,
            "fresh_targets_used_for_fit_or_tuning": False,
            "outer_or_fresh_result_based_retuning": False,
            "game_cluster_bootstrap": True,
            "minimum_game_clusters_for_new_predictive_qualification": MINIMUM_GAME_CLUSTERS_FOR_NEW_PREDICTIVE_QUALIFICATION,
            "minimum_observed_final_game_clusters": minimum_observed_final_game_clusters,
            "fresh_predictive_evidence_supported": fresh_predictive_evidence_supported,
            "new_independent_predictive_qualification": new_independent_predictive_qualification,
            "recommendation_fusion_qualified": False,
            "shadow_layer_action_weight": None,
            "direct_action_authority": False,
            "live_action_authority": False,
            "automatic_pokernow_action": False,
            "observed_action_prediction_proves_counterfactual_ev": False,
        },
    })
    if source_lineage() != lineage:
        raise RuntimeError("input/code changed during scoring; no report published")
    authenticate_frozen_installation()
    with output_json.open("xb") as handle:
        handle.write(canonical_bytes(report) + b"\n")
    with output_md.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(report))
    progress("complete", report=str(output_json), self_sha256=report["self_sha256"], elapsed_seconds=report["elapsed_seconds"])


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
