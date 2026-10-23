"""Authenticated stale-response evaluation and supported paired score output.

Heavy calls are Colab-only. Reuse is externally rooted and never regenerates
features or reruns inference. Source/model bytes and complete producer jobs must
already be accepted; this job never fits a model or recalibrates temperature.
"""
from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import time
import uuid

import stage7_response_assembly_v1 as assembly
import stage7_response_controls_v1 as controls
import numpy as np

h, prep, joint, runner, auth, t = assembly.h, assembly.prep, assembly.joint, assembly.runner, assembly.auth, assembly.t
SCHEMA = "stage7_response_control_job_v1"
MEMBERS = ("comparison_report.json", "control_predictions.npz")
RECEIPT = "control_receipt.json"
ARRAY_FIELDS = ("stale_features", "fresh", "stale", "target", "ordinal", "row_id", "game_id")


def source_hashes():
    names = set(assembly._source_hashes(True)) | {Path(__file__).name, "stage7_response_assembly_v1.py",
             "stage7_paired_response_v1.py", "stage7_response_controls_v1.py"}
    return {name: t.sha(Path(__file__).with_name(name)) for name in sorted(names)}


def inventory(output, expected):
    actual = list(output.iterdir()) if output.exists() else []
    if {member.name for member in actual} != set(expected) or any(not member.is_file() for member in actual):
        raise ValueError("unmanifested_or_partial_control_output_requires_explicit_recovery")


def authenticated_array(path, expected):
    auth.validate_identity(expected)
    raw, _ = auth.authenticated_bytes(path, expected["sha256"], expected["bytes"])
    # A read-only mmap remains backed by a mutable path. Parse the authenticated
    # buffer instead, so later replacement cannot alter the consumed features.
    return np.load(BytesIO(raw), allow_pickle=False)


def _load_arrays(path, expected):
    auth.validate_identity(expected)
    raw, _ = auth.authenticated_bytes(path, expected["sha256"], expected["bytes"])
    with np.load(BytesIO(raw), allow_pickle=False) as stored:
        keys = {role + "/" + field for role in h.ROLES for field in ARRAY_FIELDS}
        if len(stored.files) != len(keys) or set(stored.files) != keys:
            raise ValueError("control_prediction_inventory_mismatch")
        return {name: stored[name] for name in stored.files}


def inputs(plan_path, expected_identity):
    comparison = assembly.assemble_window(plan_path, expected_identity)
    plan = joint.verified_json(plan_path, expected_identity)
    contract = assembly._contract(plan)
    directory = Path(plan["preparation"]["directory"])
    receipt = joint.verified_json(directory / "preparation_receipt.json", plan["preparation"]["identity"])
    data = {}
    # The assembler has verified exact preparation and completion inventories.
    # Load only the two session models/arrays needed for this fixed control.
    for role in h.ROLES:
        entry = plan["jobs"][role + "/session"]
        output = Path(entry["arguments"]["output"])
        completion = joint.verified_json(output / "job_receipt.json", entry["receipt_identity"])
        model = joint.verified_json(output / "model.json", completion["members"]["model.json"])
        expected = completion["members"]["predictions.npz"]
        auth.validate_identity(expected)
        raw, _ = auth.authenticated_bytes(output / "predictions.npz", expected["sha256"], expected["bytes"])
        with np.load(BytesIO(raw), allow_pickle=False) as stored:
            values = {name: stored[name] for name in stored.files}
        stem = directory / receipt["attempt"] / (role + "_evaluation")
        feature_path = Path(str(stem) + ".x.npy")
        fresh_features = authenticated_array(feature_path, receipt["members"][feature_path.relative_to(directory).as_posix()])
        rows = prep.metadata(directory, receipt, values["ordinal"])
        data[role] = dict(model=model, values=values, rows=rows, fresh_features=fresh_features,
                          actors=runner.training_actors_for_role(directory, receipt, role))
    return plan, contract, receipt, comparison, data


def generate(plan, contract, receipt, data):
    """Stream completed training history once, then infer with fixed models."""
    directory = Path(plan["preparation"]["directory"])
    database = directory / receipt["attempt"] / "events.sqlite"
    connection = prep.read_only(database, receipt["members"][receipt["attempt"] + "/events.sqlite"]["sha256"])
    arrays = {}
    offsets = {role: 0 for role in h.ROLES}
    for role in h.ROLES:
        values = data[role]["values"]
        arrays[role + "/stale_features"] = np.empty_like(data[role]["fresh_features"])
        arrays[role + "/fresh"] = values["candidate"]
        for field in ("target", "ordinal", "row_id", "game_id"):
            arrays[role + "/" + field] = values[field]
    last_progress = time.monotonic()
    try:
        def training_cursor():
            nonlocal last_progress
            for count, (value,) in enumerate(connection.execute(
                    "SELECT compact_json FROM events WHERE partition='train' ORDER BY completion,row_id"), 1):
                if time.monotonic() - last_progress >= 20:
                    print(h.incumbent.canonical(dict(phase="stale_training_history", window=plan["window"],
                        rows=count, total=receipt["row_counts"][h.ACTION]["train"])), flush=True)
                    last_progress = time.monotonic()
                yield auth.parse_json(value)
        training = training_cursor()
        evaluation = (row for _, row in joint.event_rows(directory, receipt, "evaluation"))
        for row, features, _ in controls.stale_feature_stream(training, evaluation, contract["protocol"], contract["game_bounds"], plan["window"]):
            for role, vector in features.items():
                index = offsets[role]
                if index >= len(data[role]["rows"]) or row["row_id"] != data[role]["rows"][index]["row_id"]:
                    raise ValueError("control_stale_stream_row_identity_mismatch")
                arrays[role + "/stale_features"][index] = vector
                offsets[role] += 1
            if time.monotonic() - last_progress >= 20:
                print(h.incumbent.canonical(dict(phase="stale_features", window=plan["window"], completed=offsets,
                                                 total={role: len(data[role]["rows"]) for role in h.ROLES})), flush=True)
                last_progress = time.monotonic()
    finally:
        connection.close()
    if any(offsets[role] != len(data[role]["rows"]) for role in h.ROLES):
        raise ValueError("control_stale_stream_denominator_mismatch")
    for role in h.ROLES:
        item = data[role]
        n = min(64, len(item["rows"]))
        fresh_parity = h.normalize(h.batch_tree_prediction(item["model"], item["fresh_features"][:n]), item["rows"][:n], role)
        if not np.allclose(fresh_parity, item["values"]["candidate"][:n], atol=1e-10, rtol=1e-10):
            raise ValueError("control_fixed_session_model_parity_failed")
        stale = np.empty_like(item["values"]["candidate"])
        for start in range(0, len(stale), 8192):
            stop = min(len(stale), start + 8192)
            stale[start:stop] = h.normalize(h.batch_tree_prediction(item["model"], arrays[role + "/stale_features"][start:stop]), item["rows"][start:stop], role)
            if time.monotonic() - last_progress >= 20:
                print(h.incumbent.canonical(dict(phase="stale_fixed_model_inference", window=plan["window"], role=role,
                                                 rows=stop, total=len(stale))), flush=True)
                last_progress = time.monotonic()
        arrays[role + "/stale"] = stale
    return arrays


def report(plan, comparison, data, arrays):
    seed = plan["jobs"][h.ACTION + "/session"]["arguments"]["protocol"]["model_config"]["seed"]
    result = dict(schema_version=SCHEMA + "_report", evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                  window=plan["window"], paired_comparisons=comparison, stale_rule=controls.STALE_RULE,
                  model_refitted=False, temperature_reselected=False, counterfactual_response_identified=False,
                  support_claim="descriptive support slices; not exact-N causal adaptation", roles={})
    for role in h.ROLES:
        item, values = data[role], data[role]["values"]
        current = {field: arrays[role + "/" + field] for field in ARRAY_FIELDS}
        for field in ("target", "ordinal", "row_id", "game_id"):
            if current[field].dtype != values[field].dtype or not np.array_equal(current[field], values[field]):
                raise ValueError("control_prediction_identity_mismatch")
        if current["fresh"].dtype != values["candidate"].dtype or not np.array_equal(current["fresh"], values["candidate"]):
            raise ValueError("control_fresh_candidate_changed")
        x, fresh_x, rows = current["stale_features"], item["fresh_features"], item["rows"]
        public_width = len(h.feature_names(role, "public_only"))
        if (x.dtype != np.dtype(np.float64) or x.shape != fresh_x.shape or not np.isfinite(x).all()
                or not np.array_equal(x[:, :public_width], fresh_x[:, :public_width])):
            raise ValueError("control_stale_feature_schema_or_public_boundary_changed")
        controls.support_groups(x, role)
        probabilities = current["stale"]
        if (probabilities.dtype != np.dtype(np.float64) or probabilities.shape != current["fresh"].shape
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=1e-12)
                or not np.allclose(probabilities, h.normalize(probabilities, rows, role), atol=1e-12, rtol=1e-12)):
            raise ValueError("control_invalid_or_illegal_stale_predictions")
        by_id = {row["row_id"]: index for index, row in enumerate(rows)}
        common = comparison["roles"][role]["common_row_ids"]
        indices = np.asarray([by_id[row_id] for row_id in common], dtype=np.int64)
        common_rows = [rows[index] for index in indices]
        groups = controls.support_groups(fresh_x[indices], role)
        def score(selected):
            absolute = indices[selected]
            selected_rows = [rows[index] for index in absolute]
            calculated = runner.score(current["fresh"][absolute], current["target"][absolute], selected_rows, probabilities[absolute], seed)
            if selected_rows:
                calculated["diagnostic_slices"] = h.incumbent.diagnostic_slices(
                    current["fresh"][absolute], current["target"][absolute], selected_rows, probabilities[absolute], item["actors"])
            t.verify_score_arithmetic(calculated, comparison["roles"][role]["common_game_ids"], seed)
            return calculated
        result["roles"][role] = dict(labels=list(h.LABELS[role]), feature_names=h.feature_names(role, "session"),
            role_rows=len(rows), common_row_ids=common, excluded_role_row_ids=comparison["roles"][role]["excluded_role_row_ids"],
            common_rows=len(common_rows), fixed_session_temperature=item["model"]["temperature"],
            fresh_vs_stale=score(np.arange(len(indices), dtype=np.int64)), support_groups=groups,
            support_scores={level: {label: score(np.asarray(selected, dtype=np.int64)) for label, selected in bins.items()}
                            for level, bins in groups["groups"].items()})
    return result


def run(plan_path, expected_plan_identity, output, expected_receipt_identity=None):
    output = Path(output)
    receipt_path = output / RECEIPT
    if expected_receipt_identity is not None and not receipt_path.is_file():
        raise ValueError("expected_control_receipt_missing")
    if receipt_path.exists() and expected_receipt_identity is None:
        raise ValueError("trusted_control_receipt_identity_required")
    inventory(output, (*MEMBERS, RECEIPT) if receipt_path.exists() else ())
    binding = dict(assembly_plan_identity=expected_plan_identity, source_hashes=source_hashes(), stale_rule=controls.STALE_RULE)
    closed = None
    if receipt_path.exists():
        closed = joint.verified_json(receipt_path, expected_receipt_identity)
        if (set(closed) != {"schema_version", "binding", "evidence_class", "authority", "members"}
                or closed["schema_version"] != SCHEMA + "_receipt" or not assembly._same(closed["binding"], binding)
                or closed["evidence_class"] != h.EVIDENCE or closed["authority"] != h.AUTHORITY
                or set(closed["members"]) != set(MEMBERS)):
            raise ValueError("control_receipt_contract_mismatch")
    plan, contract, preparation, comparison, data = inputs(plan_path, expected_plan_identity)
    if closed is not None:
        stored_report = joint.verified_json(output / MEMBERS[0], closed["members"][MEMBERS[0]])
        arrays = _load_arrays(output / MEMBERS[1], closed["members"][MEMBERS[1]])
        if not assembly._same(stored_report, report(plan, comparison, data, arrays)):
            raise ValueError("control_report_arithmetic_or_contract_mismatch")
        return dict(window=plan["window"], reused=True, receipt=expected_receipt_identity)
    arrays = generate(plan, contract, preparation, data)
    calculated = report(plan, comparison, data, arrays)
    output.mkdir(parents=True, exist_ok=True)
    partial = output / (MEMBERS[1] + ".partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.link(partial, output / MEMBERS[1])
    partial.unlink()
    t.publish_local_bytes(output / MEMBERS[0], (h.incumbent.canonical(calculated) + "\n").encode())
    members = {name: t.identity(output / name) for name in MEMBERS}
    verified = _load_arrays(output / MEMBERS[1], members[MEMBERS[1]])
    if not assembly._same(report(plan, comparison, data, verified), calculated):
        raise ValueError("control_first_publication_validation_failed")
    inventory(output, MEMBERS)
    t.publish_local_bytes(receipt_path, (h.incumbent.canonical(dict(schema_version=SCHEMA + "_receipt", binding=binding,
        evidence_class=h.EVIDENCE, authority=h.AUTHORITY, members=members)) + "\n").encode())
    return dict(window=plan["window"], reused=False, receipt=t.identity(receipt_path))
