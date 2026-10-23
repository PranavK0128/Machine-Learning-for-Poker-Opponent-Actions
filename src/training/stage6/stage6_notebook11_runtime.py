"""Retrospective confirmation of Stage 6 behavior-model challengers.

The current August 16-20 panel was opened by the prior failed promotion run.
This notebook may report comparative evidence, but it cannot qualify or install
a behavior challenger. Promotion requires genuinely new, never-opened labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stage6_behavior_tournament import (
    CANDIDATE_REGISTRY_DIGEST,
    HOLDOUT_START_UTC,
    TOURNAMENT_RUN_ID,
    build_temporal_assignment,
    feature_matrix,
    load_holdout_frame,
    predict_artifact,
)
from stage6_metrics import multiclass_metric_report
from stage6_notebook09_runtime import (
    BOOTSTRAP_REPLICATES,
    INCUMBENT_ARTIFACT_FILES,
    NOTEBOOK01_INPUT_NAME,
    PREFLOP_NATIVE_FEATURE_NAME,
    REQUIRED_BASELINES,
    _all_accepted_rolling_rows,
    _deterministic_public_heuristic,
    _holm_bonferroni,
    _incumbent_comparison,
    _legal_uniform,
    _paired_cluster_bootstrap,
    _read_action_native_source,
    _read_preflop_native_source,
    _rolling_successor_features,
    _validate_incumbent_artifacts,
    _validate_notebook01_action_source_after_claim,
    _validate_preflop_native_feature_source,
)
from stage6_notebooks02_07_runtime import (
    ACTIONS,
    LEGAL_COLUMNS,
    LANE_COUNT,
    NOTEBOOK01_RUN_ID,
    canonical_json_bytes,
    materialization_root,
    resolve_workers,
    seal_document,
    sha256_file,
    validate_document,
)


NOTEBOOK09_PACKAGE_NAME = "POKERNOW_ASSISTANT_STAGE6_NOTEBOOK09_INPUTS_R3.zip"
NOTEBOOK09_PACKAGE_BYTES = 93_176_757
NOTEBOOK09_PACKAGE_SHA256 = "3ca44fdfd11da02ca3969cc1f5b21d07ec4554ce0909258637633ea5e74d1a61"
BOOTSTRAP_SEED = 610_611
RESULT_MEMBER_ORDER = (
    "expanded_behavior_retrospective_report.json",
    "expanded_behavior_candidate_manifest.json",
    "controller_receipt.json",
)


def _validate_zip_members(path: Path) -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            raise RuntimeError(f"invalid ZIP membership: {path.name}")
        raw = {}
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise RuntimeError(f"unsafe ZIP member: {name}")
            raw[name] = archive.read(name)
    return names, raw


def _read_notebook10_result(stage_drive_root: Path) -> dict[str, Any]:
    prefix = f"10_Stage6_Notebook10_{TOURNAMENT_RUN_ID[:16]}_"
    completion_path = (
        stage_drive_root
        / "runtime_state"
        / "notebook10"
        / TOURNAMENT_RUN_ID
        / "completion-receipt.json"
    )
    if not completion_path.is_file():
        raise RuntimeError("Notebook 11 requires the Notebook 10 completion receipt")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    validate_document(completion)
    result_identity = completion.get("returned_result")
    result_name = (
        result_identity.get("name") if isinstance(result_identity, dict) else None
    )
    if (
        completion.get("schema_version") != "stage6_notebook10_completion_receipt_v1"
        or completion.get("status") != "PASS_COMPLETE"
        or completion.get("tournament_run_id") != TOURNAMENT_RUN_ID
        or completion.get("holdout_targets_opened") is not False
        or completion.get("qualification_authority") is not False
        or completion.get("assistant_installation_allowed") is not False
        or not isinstance(result_name, str)
        or Path(result_name).name != result_name
        or not result_name.startswith(prefix)
    ):
        raise RuntimeError("Notebook 10 completion receipt is invalid")
    candidates = [stage_drive_root / "returned_results" / result_name]
    valid = []
    expected_order = [
        "temporal_assignment.json",
        "preflop_candidate_freeze.json",
        "postflop_candidate_freeze.json",
        "preflop_validation_report.json",
        "postflop_validation_report.json",
        "candidate_publication_manifest.json",
        "controller_receipt.json",
        "RESULT_MANIFEST.json",
    ]
    for path in candidates:
        if (
            not path.is_file()
            or path.stat().st_size != result_identity.get("bytes")
        ):
            continue
        digest = sha256_file(path)
        if (
            digest != result_identity.get("sha256")
            or path.stem.rsplit("_", 1)[-1] != digest[:32]
        ):
            continue
        names, raw = _validate_zip_members(path)
        if names != expected_order:
            continue
        documents = {name: json.loads(value) for name, value in raw.items()}
        try:
            for document in documents.values():
                validate_document(document)
        except Exception:
            continue
        result_manifest = documents["RESULT_MANIFEST.json"]
        if (
            result_manifest.get("schema_version") != "stage6_notebook10_result_manifest_v1"
            or result_manifest.get("tournament_run_id") != TOURNAMENT_RUN_ID
            or result_manifest.get("member_order") != expected_order[:-1]
            or result_manifest.get("holdout_targets_opened") is not False
            or result_manifest.get("self_sha256")
            != result_identity.get("result_manifest_self_sha256")
        ):
            continue
        if any(
            len(raw[row["path"]]) != row["bytes"]
            or hashlib.sha256(raw[row["path"]]).hexdigest() != row["sha256"]
            for row in result_manifest["members"]
        ):
            continue
        temporal = documents["temporal_assignment.json"]
        publication = documents["candidate_publication_manifest.json"]
        controller = documents["controller_receipt.json"]
        freezes = {
            "preflop": documents["preflop_candidate_freeze.json"],
            "postflop": documents["postflop_candidate_freeze.json"],
        }
        if (
            temporal.get("candidate_registry_digest") != CANDIDATE_REGISTRY_DIGEST
            or temporal.get("holdout_targets_opened") is not False
            or publication.get("tournament_run_id") != TOURNAMENT_RUN_ID
            or publication.get("candidate_registry_digest") != CANDIDATE_REGISTRY_DIGEST
            or publication.get("holdout_targets_opened") is not False
            or publication.get("self_sha256")
            != completion.get("candidate_publication_manifest_self_sha256")
            or publication.get("selected_artifacts")
            != completion.get("selected_artifacts")
            or controller.get("candidate_frozen") is not True
            or controller.get("holdout_targets_opened") is not False
            or any(freeze.get("candidate_frozen") is not True for freeze in freezes.values())
            or any(freeze.get("holdout_targets_opened") is not False for freeze in freezes.values())
            or publication.get("scope_freezes") != {
                scope: freeze["self_sha256"] for scope, freeze in freezes.items()
            }
            or any(
                publication.get("selected_artifacts", {}).get(scope, {}).get("sha256")
                != freeze.get("selected_artifact", {}).get("sha256")
                or publication.get("selected_artifacts", {}).get(scope, {}).get("bytes")
                != freeze.get("selected_artifact", {}).get("bytes")
                for scope, freeze in freezes.items()
            )
            or publication.get("scope_validation_reports") != {
                "preflop": documents["preflop_validation_report.json"]["self_sha256"],
                "postflop": documents["postflop_validation_report.json"]["self_sha256"],
            }
        ):
            continue
        valid.append({
            "path": path,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "documents": documents,
            "temporal": temporal,
            "publication": publication,
            "freezes": freezes,
        })
    if len(valid) != 1:
        raise RuntimeError(
            "Notebook 11 could not validate the exact Notebook 10 result bound "
            f"by its completion receipt; found {len(valid)}"
        )
    return valid[0]


def _extract_notebook09_package(stage_drive_root: Path, target: Path) -> dict[str, Any]:
    path = stage_drive_root / "inputs" / NOTEBOOK09_PACKAGE_NAME
    if (
        not path.is_file()
        or path.stat().st_size != NOTEBOOK09_PACKAGE_BYTES
        or sha256_file(path) != NOTEBOOK09_PACKAGE_SHA256
    ):
        raise RuntimeError("current-incumbent Notebook 09 package identity mismatch")
    names, raw = _validate_zip_members(path)
    if names[-1] != "PACKAGE_MANIFEST.json":
        raise RuntimeError("Notebook 09 package manifest is not last")
    manifest = json.loads(raw["PACKAGE_MANIFEST.json"])
    validate_document(manifest)
    if any(
        len(raw[row["path"]]) != row["bytes"]
        or hashlib.sha256(raw[row["path"]]).hexdigest() != row["sha256"]
        for row in manifest["members"]
    ):
        raise RuntimeError("Notebook 09 package member identity mismatch")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    _validate_incumbent_artifacts(target)
    return json.loads((target / "event_anchor.json").read_text(encoding="utf-8"))


def _claim_paths(stage_drive_root: Path) -> tuple[Path, Path]:
    root = stage_drive_root / "runtime_state" / "notebook11" / TOURNAMENT_RUN_ID
    root.mkdir(parents=True, exist_ok=True)
    return root / "holdout-claim.json", root / "completion-receipt.json"


def _claim_holdout(
    stage_drive_root: Path,
    notebook10: Mapping[str, Any],
    temporal_self_sha256: str,
) -> tuple[dict[str, Any], bool]:
    claim_path, completion_path = _claim_paths(stage_drive_root)
    claim = seal_document({
        "schema_version": "stage6_notebook11_temporal_holdout_claim_v1",
        "status": "CLAIMED_BEFORE_FIRST_HOLDOUT_TARGET_READ",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "notebook10_result_name": notebook10["path"].name,
        "notebook10_result_bytes": notebook10["bytes"],
        "notebook10_result_sha256": notebook10["sha256"],
        "temporal_assignment_self_sha256": temporal_self_sha256,
        "preflop_candidate_freeze_self_sha256": notebook10["freezes"]["preflop"]["self_sha256"],
        "postflop_candidate_freeze_self_sha256": notebook10["freezes"]["postflop"]["self_sha256"],
        "candidate_registry_digest": CANDIDATE_REGISTRY_DIGEST,
        "selection_or_tuning_after_claim_allowed": False,
        "holdout_start_utc_inclusive": HOLDOUT_START_UTC,
        "promotion_authority": False,
    })
    raw = canonical_json_bytes(claim) + b"\n"
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = json.loads(claim_path.read_text(encoding="utf-8"))
        validate_document(existing)
        if existing != claim:
            raise RuntimeError("Notebook 11 existing holdout claim conflicts with the frozen candidate")
        return existing, True
    with os.fdopen(descriptor, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    if completion_path.exists():
        raise RuntimeError("Notebook 11 completion exists without a prior visible claim")
    return claim, False


def _completed_result(
    stage_drive_root: Path,
    notebook10: Mapping[str, Any],
) -> dict[str, Any] | None:
    claim_path, completion_path = _claim_paths(stage_drive_root)
    if not completion_path.exists():
        return None
    if not claim_path.exists():
        raise RuntimeError("Notebook 11 completion exists without its holdout claim")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    validate_document(claim)
    validate_document(completion)
    result = completion.get("returned_result")
    if (
        claim.get("schema_version") != "stage6_notebook11_temporal_holdout_claim_v1"
        or claim.get("notebook10_result_sha256") != notebook10["sha256"]
        or completion.get("schema_version") != "stage6_notebook11_completion_receipt_v1"
        or completion.get("status") != "PASS_COMPLETE"
        or completion.get("tournament_run_id") != TOURNAMENT_RUN_ID
        or completion.get("holdout_claim_self_sha256") != claim["self_sha256"]
        or not isinstance(result, Mapping)
    ):
        raise RuntimeError("Notebook 11 completion receipt authority mismatch")
    path = stage_drive_root / "returned_results" / str(result.get("name", ""))
    if (
        not path.is_file()
        or path.stat().st_size != result.get("bytes")
        or sha256_file(path) != result.get("sha256")
    ):
        raise RuntimeError("Notebook 11 completed returned-result identity mismatch")
    return {
        "status": "PASS",
        "component_qualification_gates": completion["component_qualification_gates"],
        "retrospective_superiority_checks": completion[
            "retrospective_superiority_checks"
        ],
        "returned_result": dict(result),
        "returned_result_path": str(path),
        "assistant_installation_allowed_after_focused_independent_audit": False,
        "live_action_authority": False,
        "resumed_completed_event": True,
    }


def _load_selected_artifacts(
    stage_drive_root: Path,
    notebook10: Mapping[str, Any],
) -> dict[str, Any]:
    import joblib
    publication = notebook10["publication"]
    root = stage_drive_root / Path(publication["artifact_root_relative"])
    artifacts = {}
    for scope in ("preflop", "postflop"):
        identity = publication["selected_artifacts"][scope]
        path = root / identity["name"]
        if (
            not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or sha256_file(path) != identity["sha256"]
        ):
            raise RuntimeError(f"{scope} frozen candidate artifact identity mismatch")
        artifact = joblib.load(path)
        if (
            artifact.get("tournament_run_id") != TOURNAMENT_RUN_ID
            or artifact.get("street_scope") != scope
            or artifact.get("holdout_targets_used") != 0
        ):
            raise RuntimeError(f"{scope} frozen candidate artifact authority mismatch")
        artifacts[scope] = artifact
    return artifacts


def _prepare_incumbent_context(
    stage_drive_root: Path,
    root: Path,
    holdout: pd.DataFrame,
    notebook09_root: Path,
    notebook09_anchor: Mapping[str, Any],
    context_root: Path,
) -> dict[str, Any]:
    notebook01_input = _validate_notebook01_action_source_after_claim(stage_drive_root)
    preflop_source = _validate_preflop_native_feature_source(notebook09_root, notebook09_anchor)
    accepted = _all_accepted_rolling_rows(root)
    holdout_ids = set(holdout["decision_id"].astype(str))
    postflop_ids = set(
        holdout.loc[~holdout["street"].astype(str).eq("preflop"), "decision_id"].astype(str)
    )
    native_action, hand_starts = _read_action_native_source(notebook01_input, postflop_ids)
    rolling = _rolling_successor_features(accepted, hand_starts, postflop_ids)
    native_action = native_action.rename(columns={"hand_action_id": "decision_id"})
    action_context = native_action.merge(rolling, on="decision_id", how="inner", validate="one_to_one")
    preflop_all = accepted.loc[accepted["street"].astype(str).eq("preflop")].sort_values(
        ["hand_id", "seat", "decision_order", "_original_index"], kind="mergesort"
    )
    first_preflop = preflop_all.drop_duplicates(["hand_id", "seat"], keep="first")
    holdout_first = first_preflop.loc[
        first_preflop["decision_id"].astype(str).isin(holdout_ids),
        ["decision_id", "hand_id", "seat"],
    ].copy()
    preflop_context = _read_preflop_native_source(preflop_source, holdout_first)
    context_root.mkdir(parents=True, exist_ok=True)
    preflop_context.to_parquet(context_root / "preflop-native.parquet", index=False)
    action_context.to_parquet(context_root / "action-native-rolling.parquet", index=False)
    return {
        "preflop_native_domain_rows": len(preflop_context),
        "postflop_native_domain_rows": len(action_context),
        "rolling_source_rows": len(accepted),
        "notebook01_input_name": NOTEBOOK01_INPUT_NAME,
        "preflop_native_feature_name": PREFLOP_NATIVE_FEATURE_NAME,
    }


def _hand_sizes(frame: pd.DataFrame) -> dict[str, int]:
    grouped = frame.groupby(["game_id", "hand_id"], sort=False)["active_player_count_before"].max()
    result = {f"{game_id}\x1f{hand_id}": int(value) for (game_id, hand_id), value in grouped.items()}
    if any(value < 2 or value > 10 for value in result.values()):
        raise RuntimeError("temporal holdout table size is outside the qualified 2-10 range")
    return result


def _map_six_action_baseline(
    scope: str,
    matrix: np.ndarray,
    legal: np.ndarray,
    truth: Sequence[str],
    positions: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    if scope == "preflop":
        labels = ["call", "fold", "no_action", "raise"]
        truth_map = {"call": "call", "fold": "fold", "check": "no_action", "bet": "raise", "raise": "raise", "all_in": "raise"}
        groups = {"call": ("call",), "fold": ("fold",), "no_action": ("check",), "raise": ("bet", "raise", "all_in")}
    else:
        labels = ["bet_or_raise", "call", "check", "fold"]
        truth_map = {"call": "call", "fold": "fold", "check": "check", "bet": "bet_or_raise", "raise": "bet_or_raise", "all_in": "bet_or_raise"}
        groups = {"bet_or_raise": ("bet", "raise", "all_in"), "call": ("call",), "check": ("check",), "fold": ("fold",)}
    selected = matrix[positions]
    selected_legal = legal[positions]
    family = np.column_stack([
        selected[:, [ACTIONS.index(action) for action in groups[label]]].sum(axis=1)
        for label in labels
    ])
    family_legal = np.column_stack([
        selected_legal[:, [ACTIONS.index(action) for action in groups[label]]].any(axis=1)
        for label in labels
    ])
    family = np.where(family_legal, family, 0.0)
    totals = family.sum(axis=1)
    if (totals <= 0).any():
        raise RuntimeError("mapped baseline has empty legal probability mass")
    family /= totals[:, None]
    return labels, [truth_map[truth[index]] for index in positions], family


def _losses(truth: Sequence[str], probabilities: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    index = {label: position for position, label in enumerate(labels)}
    target = np.asarray([index[value] for value in truth], dtype=int)
    return -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-15, 1.0))


def _scope_adjudication(
    scope: str,
    frame: pd.DataFrame,
    artifact: Mapping[str, Any],
    notebook09_root: Path,
    context_root: Path,
    hand_sizes: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, list[str]]]:
    matrix = feature_matrix(frame)
    legal = frame[list(LEGAL_COLUMNS)].to_numpy(dtype=bool)
    truth_six = frame["action_label"].astype(str).tolist()
    raw_candidate = predict_artifact(artifact, frame, matrix, legal)
    priors = np.asarray(artifact["class_priors"], dtype=float)
    illegal_pre = int((~legal[np.arange(len(legal)), raw_candidate.argmax(axis=1)]).sum())
    candidate = np.where(legal, raw_candidate, 0.0)
    totals = candidate.sum(axis=1)
    if (totals <= 0).any():
        candidate[totals <= 0] = legal[totals <= 0].astype(float)
        totals = candidate.sum(axis=1)
    candidate /= totals[:, None]

    incumbent = _incumbent_comparison(
        frame, candidate, legal, truth_six, scope,
        str(notebook09_root), hand_sizes, str(context_root),
    )
    positions = np.asarray(incumbent["positions"], dtype=int)
    labels = list(incumbent["labels"])
    truth = list(incumbent["truth"])
    candidate_native = np.asarray(incumbent["candidate"], dtype=float)
    incumbent_native = np.asarray(incumbent["incumbent"], dtype=float)

    class_frequency = np.broadcast_to(priors, candidate.shape).copy()
    class_frequency = np.where(legal, class_frequency, 0.0)
    class_frequency /= class_frequency.sum(axis=1, keepdims=True)
    heuristic = _deterministic_public_heuristic(frame, legal)
    uniform = _legal_uniform(legal)
    weak = {}
    for name, six in (
        ("class_frequency", class_frequency),
        ("deterministic_chart_or_heuristic", heuristic),
        ("legal_random_or_uniform_where_semantically_valid", uniform),
    ):
        mapped_labels, mapped_truth, mapped = _map_six_action_baseline(
            scope, six, legal, truth_six, positions
        )
        if mapped_labels != labels or mapped_truth != truth:
            raise RuntimeError(f"{scope} baseline native mapping is misaligned")
        weak[name] = mapped
    matrices = {
        "candidate": candidate_native,
        "current_qualified_artifact": incumbent_native,
        **weak,
    }
    metrics = {
        name: multiclass_metric_report(truth, probability, labels)
        for name, probability in matrices.items()
    }
    clusters_all = (
        frame["game_id"].astype(str) + "\x1f" + frame["hand_id"].astype(str)
    ).to_numpy(dtype=object)
    clusters = clusters_all[positions].tolist()
    sizes_all = np.asarray([
        hand_sizes[value] for value in clusters_all
    ], dtype=int)
    sizes = sizes_all[positions]
    per_table_size = {}
    truth_array = np.asarray(truth, dtype=object)
    for size in sorted(set(sizes.tolist())):
        mask = sizes == size
        per_table_size[str(size)] = {
            name: multiclass_metric_report(
                truth_array[mask].tolist(), probability[mask], labels
            )
            for name, probability in matrices.items()
        }
    candidate_losses = _losses(truth, candidate_native, labels)
    differences = {
        name: candidate_losses - _losses(truth, probability, labels)
        for name, probability in matrices.items() if name != "candidate"
    }
    report = {
        "scope": scope,
        "candidate_id": artifact["candidate_id"],
        "candidate_family": artifact["family"],
        "native_domain_rows": len(truth),
        "all_holdout_scope_rows": len(frame),
        "metrics": metrics,
        "per_table_size_metrics": per_table_size,
        "illegal_preprojection_argmax_count": illegal_pre,
        "illegal_postprojection_count": 0,
        "incumbent_illegal_preprojection_count": incumbent["incumbent_illegal_preprojection"],
        "incumbent_zero_mass_fallback_count": incumbent["incumbent_zero_mass_fallback"],
        "comparison_label_space": incumbent["comparison_label_space"],
        "missing_or_defaulted_required_incumbent_input_count": incumbent["missing_or_defaulted_required_input_count"],
    }
    return report, differences, {name: clusters for name in differences}


def _fixed_zip_write(path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def _publish_result(
    stage_drive_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    members = [
        (name, canonical_json_bytes(documents[name]) + b"\n")
        for name in RESULT_MEMBER_ORDER
    ]
    manifest = seal_document({
        "schema_version": "stage6_notebook11_result_manifest_v1",
        "notebook_token": "11",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "member_order": list(RESULT_MEMBER_ORDER),
        "members": [
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in members
        ],
    })
    members.append(("RESULT_MANIFEST.json", canonical_json_bytes(manifest) + b"\n"))
    with tempfile.TemporaryDirectory(prefix="stage6-notebook11-result-") as temporary:
        source = Path(temporary) / "notebook11-result.zip"
        _fixed_zip_write(source, members)
        digest = sha256_file(source)
        name = f"11_Stage6_Notebook11_{TOURNAMENT_RUN_ID[:16]}_{digest[:32]}.zip"
        target_root = stage_drive_root / "returned_results"
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / name
        shutil.copyfile(source, target)
    if sha256_file(target) != digest:
        raise RuntimeError("Notebook 11 returned-result Drive copy changed bytes")
    return target, {
        "name": target.name,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "result_manifest_self_sha256": manifest["self_sha256"],
    }


def run_notebook11(
    stage_drive_root: Path,
    package_root: Path,
    requested_workers: str = "auto",
) -> dict[str, Any]:
    del package_root
    started = time.monotonic()
    worker_report = resolve_workers(requested_workers)
    notebook10 = _read_notebook10_result(stage_drive_root)
    metadata_root = (
        stage_drive_root
        / "runtime_state"
        / "notebook01"
        / "materializations"
        / NOTEBOOK01_RUN_ID
    )
    # Preclaim reproduction projects only row/game/hand/time metadata.  The
    # full accepted-Parquet byte verification (which hashes target-bearing
    # payloads) is intentionally deferred until after the event claim.
    hand_roles, recomputed_temporal = build_temporal_assignment(metadata_root)
    del hand_roles
    if recomputed_temporal["self_sha256"] != notebook10["temporal"]["self_sha256"]:
        raise RuntimeError("Notebook 10 temporal assignment did not reproduce before holdout target read")
    completed = _completed_result(stage_drive_root, notebook10)
    if completed is not None:
        return completed
    artifacts = _load_selected_artifacts(stage_drive_root, notebook10)
    claim, resumed = _claim_holdout(
        stage_drive_root, notebook10, recomputed_temporal["self_sha256"]
    )
    print(json.dumps({
        "event": "stage6_notebook11_holdout_claim_committed",
        "claim_self_sha256": claim["self_sha256"],
        "resumed": resumed,
        "selection_or_tuning_after_claim_allowed": False,
    }, sort_keys=True), flush=True)

    root = materialization_root(stage_drive_root)
    holdout, opened_temporal = load_holdout_frame(root)
    if opened_temporal["self_sha256"] != recomputed_temporal["self_sha256"]:
        raise RuntimeError("temporal assignment changed during holdout open")
    local_root = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    notebook09_root = local_root / f"stage6_notebook11_incumbent_{TOURNAMENT_RUN_ID}"
    notebook09_anchor = _extract_notebook09_package(stage_drive_root, notebook09_root)
    context_root = local_root / f"stage6_notebook11_context_{TOURNAMENT_RUN_ID}"
    if context_root.exists():
        shutil.rmtree(context_root)
    incumbent_context = _prepare_incumbent_context(
        stage_drive_root, root, holdout, notebook09_root, notebook09_anchor, context_root
    )
    hand_sizes = _hand_sizes(holdout)

    reports = {}
    all_differences = {}
    all_clusters = {}
    for scope in ("preflop", "postflop"):
        mask = (
            holdout["street"].astype(str).eq("preflop")
            if scope == "preflop"
            else holdout["street"].astype(str).isin(("flop", "turn", "river"))
        )
        report, differences, clusters = _scope_adjudication(
            scope, holdout.loc[mask].copy(), artifacts[scope],
            notebook09_root, context_root, hand_sizes,
        )
        reports[scope] = report
        for name, values in differences.items():
            key = f"{scope}:{name}"
            row = _paired_cluster_bootstrap(
                {name: values}, clusters[name], BOOTSTRAP_SEED + (0 if scope == "preflop" else 1)
            )[name]
            all_differences[key] = row
            all_clusters[key] = len(set(clusters[name]))
        print(json.dumps({
            "event": "stage6_notebook11_scope_adjudicated",
            "scope": scope,
            "native_domain_rows": report["native_domain_rows"],
            "comparisons": len(differences),
        }, sort_keys=True), flush=True)
    adjusted = _holm_bonferroni(all_differences)
    retrospective_checks = {
        scope: all(
            adjusted[f"{scope}:{baseline}"]["passes_preregistered_superiority_rule"]
            for baseline in REQUIRED_BASELINES
        ) and reports[scope]["illegal_postprojection_count"] == 0
        for scope in ("preflop", "postflop")
    }
    report_document = seal_document({
        "schema_version": "stage6_expanded_behavior_retrospective_confirmation_v1",
        "status": "PASS_RETROSPECTIVE_CONFIRMATION_NO_PROMOTION_AUTHORITY",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "notebook10_result_sha256": notebook10["sha256"],
        "temporal_assignment_self_sha256": recomputed_temporal["self_sha256"],
        "holdout_claim_self_sha256": claim["self_sha256"],
        "holdout_row_count": len(holdout),
        "holdout_hand_count": recomputed_temporal["role_hand_counts"]["tournament_final_holdout"],
        "scope_reports": reports,
        "paired_cluster_bootstrap_holm": adjusted,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_cluster_counts": all_clusters,
        "required_baselines": list(REQUIRED_BASELINES),
        "component_qualification_gates": {
            "preflop_behavior": False,
            "postflop_behavior": False,
        },
        "retrospective_superiority_checks": {
            "preflop_behavior": retrospective_checks["preflop"],
            "postflop_behavior": retrospective_checks["postflop"],
        },
        "incumbent_native_context": incumbent_context,
        "per_table_size_metrics_status": "PASS_COMPUTED",
        "selection_or_tuning_after_claim_count": 0,
        "holdout_open_event_count": 1,
        "resumed_after_claimed_operational_failure": resumed,
        "panel_scientific_authority": "retrospective_only_previously_opened_existing_corpus",
        "test_panel_reuse_after_failed_promotion": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "live_action_authority": False,
    })
    candidate_manifest = seal_document({
        "schema_version": "stage6_expanded_behavior_retrospective_candidate_manifest_v1",
        "status": "RETROSPECTIVE_ONLY_RETAIN_CURRENT_QUALIFIED_BEHAVIOR_INCUMBENTS",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "adjudication_report_self_sha256": report_document["self_sha256"],
        "component_qualification_gates": {
            "preflop_behavior": False,
            "postflop_behavior": False,
        },
        "retrospective_superiority_checks": {
            "preflop_behavior": retrospective_checks["preflop"],
            "postflop_behavior": retrospective_checks["postflop"],
        },
        "qualified_scopes": [],
        "retained_incumbent_scopes": ["preflop", "postflop"],
        "selected_artifacts": notebook10["publication"]["selected_artifacts"],
        "current_incumbent_artifact_files": INCUMBENT_ARTIFACT_FILES,
        "assistant_installation_allowed_after_focused_independent_audit": False,
        "automatic_installation_performed": False,
        "panel_scientific_authority": "retrospective_only_previously_opened_existing_corpus",
        "test_panel_reuse_after_failed_promotion": False,
        "qualification_authority": False,
        "live_action_authority": False,
    })
    controller = seal_document({
        "schema_version": "stage6_notebook11_controller_receipt_v1",
        "status": "PASS_EXPANDED_BEHAVIOR_RETROSPECTIVE_CONFIRMATION_COMPLETE",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "worker_report": worker_report,
        "logical_lane_count": LANE_COUNT,
        "elapsed_seconds": time.monotonic() - started,
        "holdout_claim_self_sha256": claim["self_sha256"],
        "adjudication_report_self_sha256": report_document["self_sha256"],
        "candidate_manifest_self_sha256": candidate_manifest["self_sha256"],
        "component_qualification_gates": candidate_manifest["component_qualification_gates"],
        "retrospective_superiority_checks": candidate_manifest[
            "retrospective_superiority_checks"
        ],
        "notebook10_result_sha256": notebook10["sha256"],
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "live_action_authority": False,
    })
    documents = {
        "expanded_behavior_retrospective_report.json": report_document,
        "expanded_behavior_candidate_manifest.json": candidate_manifest,
        "controller_receipt.json": controller,
    }
    result_path, result_identity = _publish_result(stage_drive_root, documents)
    _, completion_path = _claim_paths(stage_drive_root)
    completion = seal_document({
        "schema_version": "stage6_notebook11_completion_receipt_v1",
        "status": "PASS_COMPLETE",
        "tournament_run_id": TOURNAMENT_RUN_ID,
        "holdout_claim_self_sha256": claim["self_sha256"],
        "returned_result": result_identity,
        "candidate_manifest_self_sha256": candidate_manifest["self_sha256"],
        "component_qualification_gates": candidate_manifest["component_qualification_gates"],
        "retrospective_superiority_checks": candidate_manifest[
            "retrospective_superiority_checks"
        ],
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "live_action_authority": False,
    })
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")
    shutil.rmtree(notebook09_root, ignore_errors=True)
    shutil.rmtree(context_root, ignore_errors=True)
    return {
        "status": "PASS",
        "component_qualification_gates": candidate_manifest["component_qualification_gates"],
        "retrospective_superiority_checks": candidate_manifest[
            "retrospective_superiority_checks"
        ],
        "returned_result": result_identity,
        "returned_result_path": str(result_path),
        "assistant_installation_allowed_after_focused_independent_audit": False,
        "live_action_authority": False,
    }


__all__ = ["run_notebook11"]
