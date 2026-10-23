"""Pure common-support calculations for the fixed Notebook05 comparison family.

The controller must authenticate every contributing job and preparation before
calling this module. It does not load files, select a winner or publish results.
"""
from __future__ import annotations

import numpy as np
import stage7_hierarchical_runner_v1 as runner

h = runner.h
FRAME_NAMES = (*h.ARMS, "same_recipe")
JOINT_LABELS = sorted([action + "@" + size for action in h.AGGRESSIVE for size in h.LABELS[h.SIZE]]
                      + [action + "@none" for action in h.LABELS[h.ACTION] if action not in h.AGGRESSIVE])
PAIRS = tuple((arm, "same_recipe") for arm in h.ARMS) + (
    ("population", "public_only"), ("player", "population"), ("session", "player"))


def compare(window, role, frames, records, training_actors, seed):
    """Compare every declared arm on one identical, explicit supported row set."""
    if window not in ("50", "75", "100") or role not in h.ROLES or set(frames) != set(FRAME_NAMES):
        raise ValueError("exact_fixed_comparison_family_required")
    if not isinstance(training_actors, (set, frozenset)) or any(not isinstance(actor, str) or not actor for actor in training_actors):
        raise ValueError("role_specific_training_actors_required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative_fixed_seed_required")
    by_id = {}
    for row in records:
        h.validate_observation(row)
        if row["row_id"] in by_id or h.target(row, role) is None:
            raise ValueError("duplicate_or_incompatible_paired_record")
        by_id[row["row_id"]] = row
    indices = {}
    for name in FRAME_NAMES:
        frame = frames[name]
        if set(frame) != {"role", "labels", "probabilities", "target", "row_id", "game_id"}:
            raise ValueError("exact_prediction_frame_schema_required")
        expected_role = "joint" if name == "same_recipe" and window in ("50", "75") else role
        if frame["role"] != expected_role:
            raise ValueError("incompatible_frame_role")
        labels = frame["labels"]
        if not isinstance(labels, list) or len(labels) != len(set(labels)) or not labels:
            raise ValueError("unique_frame_labels_required")
        if frame["role"] == role and labels != list(h.LABELS[role]):
            raise ValueError("exact_role_label_schema_required")
        if frame["role"] == "joint" and labels != JOINT_LABELS:
            raise ValueError("exact_frozen_joint_label_schema_required")
        p, y, ids, games = [np.asarray(frame[key]) for key in ("probabilities", "target", "row_id", "game_id")]
        if (p.dtype != np.dtype(np.float64) or y.dtype != np.dtype(np.int16) or ids.dtype.kind != "U" or games.dtype.kind != "U"
                or p.shape != (len(ids), len(labels)) or y.shape != (len(ids),) or games.shape != (len(ids),)
                or ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or not set(ids.tolist()) <= set(by_id)
                or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
                or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=1e-12)):
            raise ValueError("invalid_prediction_frame_payload")
        for index, row_id in enumerate(ids):
            row = by_id[row_id]
            target_label = h.incumbent.size_label(row) if frame["role"] == "joint" else h.LABELS[role][h.target(row, role)]
            if (games[index] != row["game_id"] or not 0 <= y[index] < len(labels)
                    or labels[int(y[index])] != target_label):
                raise ValueError("paired_target_or_game_identity_mismatch")
        frame_rows = [by_id[row_id] for row_id in ids]
        if frame["role"] == "joint":
            # The exact frozen vocabulary/order was checked above; this
            # projection adds numeric and action-mask checks.
            h.collapse_incumbent(p, labels, frame_rows, h.ACTION)
            normalized = h.incumbent.legal_probabilities(p, frame_rows, labels)
        else:
            normalized = h.normalize(p, frame_rows, role)
        if not np.allclose(p, normalized, atol=1e-12, rtol=1e-12):
            raise ValueError("illegal_prediction_frame_mass")
        indices[name] = {row_id: index for index, row_id in enumerate(ids.tolist())}
    # All four feature arms must cover the same role inventory. Only the
    # explicitly support-limited joint comparator may reduce the common set.
    if any(set(indices[arm]) != set(by_id) for arm in h.ARMS):
        raise ValueError("feature_arm_role_inventory_mismatch")
    common_ids = sorted(set.intersection(*(set(indices[name]) for name in FRAME_NAMES)))
    rows = [by_id[row_id] for row_id in common_ids]
    probabilities = {}
    for name in FRAME_NAMES:
        frame = frames[name]
        selected = np.asarray(frame["probabilities"])[[indices[name][row_id] for row_id in common_ids]]
        probabilities[name] = (h.collapse_incumbent(selected, frame["labels"], rows, role)
                               if frame["role"] == "joint" else selected)
    targets = np.asarray([h.target(row, role) for row in rows], dtype=np.int16)
    comparisons = {}
    for candidate, baseline in PAIRS:
        result = runner.score(probabilities[candidate], targets, rows, probabilities[baseline], seed)
        if rows:
            result["diagnostic_slices"] = h.incumbent.diagnostic_slices(
                probabilities[candidate], targets, rows, probabilities[baseline], training_actors)
        runner.transport.verify_score_arithmetic(result, sorted({row["game_id"] for row in records}), seed)
        comparisons[candidate + "_vs_" + baseline] = result
    return dict(schema_version="stage7_paired_response_v1", window=window, role=role, evidence_class=h.EVIDENCE,
                authority=h.AUTHORITY, familywise_confirmation_claim=False, selected_winner=None,
                scope="fixed seven comparisons; descriptive whole-game intervals; no winner selection or averaging",
                role_rows=len(records), common_rows=len(common_ids), common_game_ids=sorted({row["game_id"] for row in rows}),
                common_row_ids=common_ids, excluded_role_row_ids=sorted(set(by_id) - set(common_ids)),
                frame_rows={name: len(indices[name]) for name in FRAME_NAMES}, comparisons=comparisons)
