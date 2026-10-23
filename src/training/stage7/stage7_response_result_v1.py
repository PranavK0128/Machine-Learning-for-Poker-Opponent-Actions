"""Portable Notebook05 evidence export and strict no-inference result import.

Export runs beside completed Colab work. Import authenticates an externally
retained envelope and the original execution plan before parsing; it checks
stored predictions and score arithmetic, never prepares data or replays trees.
The compact context is a new producer-attested derivative, not a claim that
local import reconstructs training history or establishes scientific validity.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import os
from pathlib import Path
import re

import numpy as np
import stage7_response_controller_v1 as controller

control, c = controller.control, controller.checkpoints
assembly, h, prep, runner, joint, auth, t = (controller.assembly, controller.h, controller.prep,
    controller.runner, controller.joint, controller.auth, controller.t)
paired = assembly.paired
SCHEMA = "stage7_response_result_v1"
MANIFEST = "result_manifest.json"


def code_identities():
    return dict(controller.code_identities(), **{Path(__file__).name: t.identity(__file__)})


def required_units():
    return ({prefix + w for w in ("50", "75", "100") for prefix in ("prepare_", "assembly_plan_", "controls_")}
            | {role + "_" + arm + "_" + w for w in ("50", "75", "100") for role in h.ROLES for arm in h.ARMS}
            | {"joint_50", "joint_75"})


def copy_pinned(source, destination, identity):
    """Copy exactly producer-pinned bytes; existing bytes must match."""
    auth.validate_identity(identity)
    destination = Path(destination)
    if destination.exists():
        t.verify(destination, identity)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed copy remains diagnostic evidence, not a completed member.
    partial = destination.with_name(destination.name + ".partial")
    with Path(source).open("rb") as src, partial.open("xb") as dst:
        c._stream(src, dst, identity)
        dst.flush(); os.fsync(dst.fileno())
    os.link(partial, destination)
    partial.unlink()


def write_json(path, value):
    t.publish_local_bytes(path, (h.incumbent.canonical(value) + "\n").encode())
    return t.identity(path)


def load_npz(path, identity, keys):
    raw, _ = auth.authenticated_bytes(path, identity["sha256"], identity["bytes"])
    with np.load(BytesIO(raw), allow_pickle=False) as values:
        if len(values.files) != len(keys) or set(values.files) != set(keys):
            raise ValueError("result_prediction_inventory_mismatch")
        return {name: values[name] for name in values.files}


def export(root, durable, plan_path, plan_identity, completion_index, output):
    """Create one exact compact evidence directory from completed source pins."""
    root, durable, output = Path(root).resolve(), Path(durable), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("result_export_existing_output_requires_external_recovery")
    plan, contract = controller.verify_plan(plan_path, plan_identity, root)
    binding = dict(schema_version=controller.SCHEMA, run_id=plan["run_id"], execution_plan_identity=plan_identity)
    journal = c.Journal(root / "work", durable, binding, completion_index)
    if set(journal.entries) != required_units():
        raise ValueError("result_requires_all_35_completions")
    if (journal.root / "controller.lock").exists():
        raise ValueError("result_cannot_export_active_controller")
    output.mkdir(parents=True, exist_ok=True)
    members = {}
    def include(source, name, expected):
        c.relative(name)
        if name in members and members[name] != expected:
            raise ValueError("result_member_pin_conflict")
        copy_pinned(source, output / name, expected)
        members[name] = expected
    include(plan_path, "execution_plan.json", plan_identity)
    include(root / plan["input_contract"]["path"], "input_contract.json", plan["input_contract"]["identity"])
    include(durable / completion_index["file_name"], "proofs/" + completion_index["file_name"], completion_index["identity"])
    for unit, entry in sorted(journal.entries.items()):
        proof = c.read_manifest(durable, entry["snapshot"], dict(controller=binding, unit=unit, receipt_identity=entry["receipt_identity"]))
        if proof["members"].get(entry["receipt_path"]) != entry["receipt_identity"]:
            raise ValueError("result_completion_receipt_not_bound")
        include(durable / entry["snapshot"]["file_name"], "proofs/" + entry["snapshot"]["file_name"], entry["snapshot"]["identity"])
        chosen = {entry["receipt_path"]: entry["receipt_identity"]} if unit.startswith("prepare_") else proof["members"]
        for name, expected in chosen.items():
            include(journal.root / name, "work/" + name, expected)
    for window in plan["windows"]:
        entry = journal.entries["assembly_plan_" + window]
        path = journal.root / entry["receipt_path"]
        assembly_plan, _, receipt, _, data = control.inputs(path, entry["receipt_identity"])
        # Reuse validation recalculates stored score arithmetic, not inference.
        controls_entry = journal.entries["controls_" + window]
        control.run(path, entry["receipt_identity"], journal.root / ("controls_" + window), controls_entry["receipt_identity"])
        directory = Path(assembly_plan["preparation"]["directory"])
        context = dict(schema_version=SCHEMA + "_context", window=window,
            preparation_identity=assembly_plan["preparation"]["identity"],
            records=list(joint.event_rows(directory, receipt, "evaluation")),
            training_actors={role: sorted(data[role]["actors"]) for role in h.ROLES},
            training_joint_counts=dict(joint.training_counts(directory, receipt)),
            calibration_joint_counts=dict(Counter(h.incumbent.size_label(row) for _, row in joint.event_rows(directory, receipt, "calibration"))))
        name = "contexts/" + window + ".json"
        (output / "contexts").mkdir(exist_ok=True)
        members[name] = write_json(output / name, context)
        for role in h.ROLES:
            name = receipt["attempt"] + "/" + role + "_evaluation.x.npy"
            include(directory / name, "contexts/" + window + "_" + role + ".npy", receipt["members"][name])
    manifest = dict(schema_version=SCHEMA, run_id=plan["run_id"], evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
        execution_plan_identity=plan_identity, input_contract_identity=plan["input_contract"]["identity"],
        code_files=code_identities(), completion_index=completion_index, original_root=str(root), members=members,
        context_claim="producer-attested evaluation metadata and training summaries; no local source reconstruction",
        model_replay_performed_by_importer=False, scientific_qualification=False)
    identity = write_json(output / MANIFEST, manifest)
    validate(output, identity, plan_identity)
    return identity


def _structure(output, manifest_identity, expected_plan_identity):
    output = Path(output)
    manifest = joint.verified_json(output / MANIFEST, manifest_identity)
    if (set(manifest) != {"schema_version", "run_id", "evidence_class", "authority", "execution_plan_identity",
            "input_contract_identity", "code_files", "completion_index", "original_root", "members", "context_claim",
            "model_replay_performed_by_importer", "scientific_qualification"}
            or manifest["schema_version"] != SCHEMA or manifest["run_id"] != "stage7_hierarchical_response_r1"
            or manifest["evidence_class"] != h.EVIDENCE or manifest["authority"] != h.AUTHORITY
            or manifest["execution_plan_identity"] != expected_plan_identity or manifest["code_files"] != code_identities()
            or manifest["model_replay_performed_by_importer"] is not False or manifest["scientific_qualification"] is not False
            or manifest["context_claim"] != "producer-attested evaluation metadata and training summaries; no local source reconstruction"):
        raise ValueError("result_manifest_contract_mismatch")
    members = manifest["members"]
    c.member_map(members)
    actual = [p for p in output.rglob("*") if p.is_file() or p.is_symlink()]
    if (any(p.is_symlink() for p in output.rglob("*")) or any(not p.is_file() for p in actual)
            or {p.relative_to(output).as_posix() for p in actual} != set(members) | {MANIFEST}):
        raise ValueError("result_exact_payload_inventory_required")
    for name, expected in members.items():
        t.verify(output / name, expected)
    if members.get("execution_plan.json") != expected_plan_identity or members.get("input_contract.json") != manifest["input_contract_identity"]:
        raise ValueError("result_input_pin_mismatch")
    plan = joint.verified_json(output / "execution_plan.json", expected_plan_identity)
    if (set(plan) != {"schema_version", "run_id", "input_contract", "source_database", "incumbent", "code_files", "windows"}
            or plan["schema_version"] != controller.SCHEMA + "_plan" or plan["run_id"] != manifest["run_id"]
            or plan["code_files"] != controller.code_identities() or plan["windows"] != ["50", "75", "100"]
            or plan["input_contract"]["identity"] != manifest["input_contract_identity"]):
        raise ValueError("result_fixed_execution_plan_required")
    for key in ("input_contract", "source_database", "incumbent"):
        if set(plan[key]) != {"path", "identity"}:
            raise ValueError("result_input_reference_schema_mismatch")
        c.relative(plan[key]["path"]); auth.validate_identity(plan[key]["identity"])
    contract = assembly._contract(dict(input_contract=dict(path=str(output / "input_contract.json"), identity=manifest["input_contract_identity"])))
    if (plan["source_database"]["identity"] != contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]
            or plan["incumbent"]["identity"] != {k: contract["accepted_sources"]["range_response_artifact.json"][k] for k in ("bytes", "sha256")}):
        raise ValueError("result_not_accepted_source_inputs")
    index_ref = manifest["completion_index"]
    if set(index_ref) != {"file_name", "identity"} or index_ref["file_name"] != "index_000035.json":
        raise ValueError("result_terminal_index_required")
    if members.get("proofs/" + index_ref["file_name"]) != index_ref["identity"]:
        raise ValueError("result_index_pin_mismatch")
    binding = dict(schema_version=controller.SCHEMA, run_id=plan["run_id"], execution_plan_identity=expected_plan_identity)
    journal = c.Journal(output / "work", output / "proofs", binding, index_ref)
    if set(journal.entries) != required_units():
        raise ValueError("result_requires_all_35_completions")
    expected = {"execution_plan.json", "input_contract.json", "proofs/" + index_ref["file_name"]}
    for unit, entry in journal.entries.items():
        if set(entry) != {"snapshot", "receipt_path", "receipt_identity"}:
            raise ValueError("result_completion_entry_schema_mismatch")
        receipt_path = ("prepared_" + unit[8:] + "/preparation_receipt.json" if unit.startswith("prepare_") else
                        unit + ".json" if unit.startswith("assembly_plan_") else
                        unit + "/" + (control.RECEIPT if unit.startswith("controls_") else "job_receipt.json"))
        if entry["receipt_path"] != receipt_path or entry["snapshot"]["file_name"] != unit + ".checkpoint.json":
            raise ValueError("result_canonical_completion_path_required")
        snapshot_name = "proofs/" + entry["snapshot"]["file_name"]
        if members.get(snapshot_name) != entry["snapshot"]["identity"]:
            raise ValueError("result_checkpoint_pin_mismatch")
        proof = c.read_manifest(output / "proofs", entry["snapshot"], dict(controller=binding, unit=unit, receipt_identity=entry["receipt_identity"]))
        if proof["members"].get(receipt_path) != entry["receipt_identity"]:
            raise ValueError("result_checkpoint_receipt_mismatch")
        expected.add(snapshot_name)
        chosen = {receipt_path: entry["receipt_identity"]} if unit.startswith("prepare_") else proof["members"]
        for name, identity in chosen.items():
            if members.get("work/" + name) != identity:
                raise ValueError("result_original_member_pin_mismatch")
            expected.add("work/" + name)
    for window in plan["windows"]:
        expected.add("contexts/" + window + ".json")
        expected.update("contexts/" + window + "_" + role + ".npy" for role in h.ROLES)
    if set(members) != expected:
        raise ValueError("result_semantic_member_inventory_mismatch")
    return manifest, plan, contract, journal


def _context(output, manifest, contract, journal, window):
    members = manifest["members"]
    name = "contexts/" + window + ".json"
    context = joint.verified_json(output / name, members[name])
    entry = journal.entries["prepare_" + window]
    preparation = joint.verified_json(output / "work" / entry["receipt_path"], entry["receipt_identity"])
    protocol = contract["protocol"]
    games = dict(train=protocol["training_windows"][window], calibration=protocol["calibration_game_ids"], evaluation=protocol["development_evaluation_game_ids"])
    binding = dict(window=window, game_ids=games, database_sha256=contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]["sha256"],
                   game_bounds_sha256=runner.digest(contract["game_bounds"]), protocol_sha256=runner.digest(protocol))
    if (set(context) != {"schema_version", "window", "preparation_identity", "records", "training_actors", "training_joint_counts", "calibration_joint_counts"}
            or context["schema_version"] != SCHEMA + "_context" or context["window"] != window
            or context["preparation_identity"] != entry["receipt_identity"] or preparation["game_ids"] != games
            or any(not assembly._same(preparation["binding"].get(k), v) for k, v in binding.items())
            or preparation["binding"].get("execution_plan_identity") != manifest["execution_plan_identity"]):
        raise ValueError("result_context_preparation_binding_mismatch")
    if (set(preparation) != {"schema_version", "binding", "evidence_class", "authority", "attempt", "source_preparation_recomputed",
            "source_compact_database_reused", "game_ids", "row_counts", "feature_names", "members"}
            or preparation["schema_version"] != h.SCHEMA + "_preparation" or preparation["evidence_class"] != h.EVIDENCE
            or preparation["authority"] != h.AUTHORITY or preparation["source_preparation_recomputed"] is not False
            or preparation["source_compact_database_reused"] is not True
            or not re.fullmatch(r"attempt_[a-f0-9]{32}", preparation["attempt"])
            or preparation["feature_names"] != {role: h.feature_names(role, "session") for role in h.ROLES}
            or not assembly._same(preparation["binding"], dict(binding, execution_plan_identity=manifest["execution_plan_identity"]))):
        raise ValueError("result_preparation_semantic_schema_mismatch")
    attempt = preparation["attempt"]
    expected_members = {attempt + "/events.sqlite"} | {attempt + "/" + role + "_" + part + "." + kind + ".npy"
        for role in h.ROLES for part in ("train", "calibration", "evaluation") for kind in ("x", "y", "ordinal")}
    c.member_map(preparation["members"])
    if set(preparation["members"]) != expected_members:
        raise ValueError("result_preparation_member_inventory_mismatch")
    if (set(preparation["row_counts"]) != set(h.ROLES) or any(set(v) != {"train", "calibration", "evaluation"}
            or any(type(n) is not int or n < 0 for n in v.values()) for v in preparation["row_counts"].values())):
        raise ValueError("result_preparation_count_schema_mismatch")
    records = context["records"]
    if (not isinstance(records, list) or any(not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int or pair[0] < 0 for pair in records)
            or len({pair[0] for pair in records}) != len(records)):
        raise ValueError("result_record_ordinal_inventory_mismatch")
    seen, order = set(), []
    for ordinal, row in records:
        h.validate_observation(row)
        if row["row_id"] in seen or row["game_id"] not in games["evaluation"]:
            raise ValueError("result_record_game_or_identity_mismatch")
        seen.add(row["row_id"])
        order.append((h.incumbent.utc(row["decision_utc"]), row["row_id"]))
    if (order != sorted(order) or len(records) != preparation["row_counts"][h.ACTION]["evaluation"]
            or {r["game_id"] for _, r in records} != set(games["evaluation"])):
        raise ValueError("result_record_order_or_denominator_mismatch")
    if set(context["training_actors"]) != set(h.ROLES):
        raise ValueError("result_role_actor_inventory_mismatch")
    for role, actors in context["training_actors"].items():
        if not isinstance(actors, list) or any(not isinstance(a, str) or not a for a in actors) or actors != sorted(set(actors)):
            raise ValueError("result_canonical_training_actors_required")
    for name, partition in (("training_joint_counts", "train"), ("calibration_joint_counts", "calibration")):
        counts = context[name]
        if (not isinstance(counts, dict) or not counts or not set(counts) <= set(paired.JOINT_LABELS)
                or any(type(v) is not int or v < 1 for v in counts.values())
                or sum(counts.values()) != preparation["row_counts"][h.ACTION][partition]
                or sum(v for k, v in counts.items() if k.split("@")[0] in h.AGGRESSIVE) != preparation["row_counts"][h.SIZE][partition]):
            raise ValueError("result_training_or_calibration_joint_denominator_mismatch")
    return context, preparation


def _job(output, manifest, plan, preparation, name, entry, rows, ordinals, actors, counts, window):
    arguments, is_joint = entry["arguments"], name == "same_recipe"
    role, arm = (joint.ROLE, None) if is_joint else tuple(name.split("/"))
    unit = "joint_" + window if is_joint else role + "_" + arm + "_" + window
    directory = output / "work" / unit
    closed = joint.verified_json(directory / "job_receipt.json", entry["receipt_identity"])
    expected_binding = dict(arguments, source_hashes=assembly._source_hashes(is_joint))
    if is_joint:
        expected_binding["implementation"] = joint.SCHEMA
    if (set(closed) != {"schema_version", "evidence_class", "authority", "binding", "native_portable_parity_rows", "members"}
            or closed["schema_version"] != h.SCHEMA + "_job" or closed["evidence_class"] != h.EVIDENCE
            or closed["authority"] != h.AUTHORITY or not assembly._same(closed["binding"], expected_binding)
            or set(closed["members"]) != {"model.json", "scores.json", "predictions.npz"}):
        raise ValueError("result_job_receipt_contract_mismatch")
    for member, identity in closed["members"].items():
        if manifest["members"].get("work/" + unit + "/" + member) != identity:
            raise ValueError("result_job_original_pin_mismatch")
    model = joint.verified_json(directory / "model.json", closed["members"]["model.json"])
    report = joint.verified_json(directory / "scores.json", closed["members"]["scores.json"])
    values = load_npz(directory / "predictions.npz", closed["members"]["predictions.npz"], assembly.ARRAY_NAMES)
    labels = paired.JOINT_LABELS if is_joint else list(h.LABELS[role])
    assembly._arrays(values, rows, ordinals, labels, "joint" if is_joint else role)
    if (type(closed["native_portable_parity_rows"]) is not int or closed["native_portable_parity_rows"] != min(64, len(rows))
            or model.get("schema_version") != (joint.SCHEMA if is_joint else h.SCHEMA)
            or report.get("schema_version") != (joint.SCHEMA if is_joint else h.SCHEMA) + "_scores"
            or any(v.get("evidence_class") != h.EVIDENCE or v.get("authority") != h.AUTHORITY or v.get("role") != role for v in (model, report))
            or model.get("trained_game_ids") != preparation["game_ids"]["train"]
            or model.get("calibration_game_ids") != preparation["game_ids"]["calibration"]
            or report.get("job_id") != unit or report.get("labels") != labels or report.get("counterfactual_response_identified") is not False
            or model.get("temperature") not in h.TEMPERATURES or report.get("temperature") != model["temperature"]):
        raise ValueError("result_job_semantic_mismatch")
    if is_joint:
        expected = joint.metadata(arguments, preparation, sorted(counts), counts, model["temperature"])
        if (set(model) != set(expected) | {"trees"} or not assembly._same({k: model[k] for k in expected}, expected)
                or report.get("window") != window or report.get("fitted_labels") != sorted(counts)
                or report.get("full_window_refitted") is not False or report.get("game_ids") != preparation["game_ids"]
                or report.get("evaluation_supported_rows") != len(rows) or report.get("training_rows") != sum(counts.values())):
            raise ValueError("result_joint_model_or_cohort_mismatch")
        report_keys = {"schema_version", "job_id", "role", "window", "evidence_class", "authority", "labels", "fitted_labels",
            "training_rows", "temperature", "game_ids", "calibration_supported_rows", "evaluation_supported_rows", "unsupported_target_counts",
            "support_policy", "counterfactual_response_identified", "full_window_refitted", "frozen_incumbent", "training_prior"}
        if (set(report) != report_keys
                or report["support_policy"] != "score only joint targets observed in this training window; retain all excluded counts"):
            raise ValueError("result_joint_report_schema_mismatch")
    elif (model.get("arm") != arm or report.get("arm") != arm or model.get("labels") != labels
            or model.get("feature_names") != h.feature_names(role, arm)
            or model.get("temperature_selection") != "fixed_grid_calibration_whole_games_only"
            or report.get("training_rows") != preparation["row_counts"][role]["train"]
            or report.get("calibration_games") != preparation["game_ids"]["calibration"]
            or report.get("evaluation_games") != preparation["game_ids"]["evaluation"]):
        raise ValueError("result_role_model_or_cohort_mismatch")
    if not is_joint:
        if (set(model) != {"labels", "feature_names", "fitted_class_indices", "trees", "inference", "schema_version", "role", "arm",
                "temperature", "temperature_selection", "evidence_class", "authority", "trained_game_ids", "calibration_game_ids"}
                or set(report) != {"schema_version", "job_id", "role", "arm", "evidence_class", "authority", "temperature", "training_rows",
                    "calibration_games", "evaluation_games", "labels", "counterfactual_response_identified", "frozen_incumbent", "training_prior"}):
            raise ValueError("result_role_model_or_report_schema_mismatch")
    for comparator in ("frozen_incumbent", "training_prior"):
        calculated = runner.score(values["candidate"], values["target"], rows, values[comparator], arguments["protocol"]["model_config"]["seed"])
        if rows:
            calculated["diagnostic_slices"] = h.incumbent.diagnostic_slices(values["candidate"], values["target"], rows, values[comparator], actors)
        if not assembly._same(calculated, report[comparator]):
            raise ValueError("result_job_score_arithmetic_mismatch")
    return model, report, values


def validate(output, manifest_identity, expected_plan_identity):
    """Read-only validation of portable evidence, with no model replay."""
    output = Path(output)
    manifest, plan, contract, journal = _structure(output, manifest_identity, expected_plan_identity)
    summaries = []
    for window in plan["windows"]:
        context, preparation = _context(output, manifest, contract, journal, window)
        assembly_entry = journal.entries["assembly_plan_" + window]
        original = joint.verified_json(output / "work" / assembly_entry["receipt_path"], assembly_entry["receipt_identity"])
        # Preserve the original plan bytes. Compare lexical producer paths;
        # never follow Colab paths or rewrite receipts to local paths.
        origin = manifest["original_root"].replace("\\", "/").rstrip("/")
        expected_jobs = {role + "/" + arm for role in h.ROLES for arm in h.ARMS} | ({"same_recipe"} if window != "100" else set())
        if (set(original) != {"schema_version", "window", "input_contract", "preparation", "jobs"}
                or original["schema_version"] != assembly.SCHEMA + "_plan" or original["window"] != window
                or set(original["jobs"]) != expected_jobs
                or original["input_contract"]["identity"] != manifest["input_contract_identity"]
                or original["input_contract"]["path"].replace("\\", "/") != origin + "/" + plan["input_contract"]["path"]
                or original["preparation"]["identity"] != context["preparation_identity"]
                or original["preparation"]["directory"].replace("\\", "/") != origin + "/work/prepared_" + window):
            raise ValueError("result_original_assembly_plan_mismatch")
        loaded = {}
        for name, entry in original["jobs"].items():
            is_joint = name == "same_recipe"
            role, arm = (h.ACTION, None) if is_joint else name.split("/")
            unit = "joint_" + window if is_joint else role + "_" + arm + "_" + window
            arguments = entry["arguments"]
            required_args = assembly.BASE_ARGUMENTS | ({"window", "input_contract_path", "input_contract_identity"} if is_joint else {"role", "arm"})
            if (set(entry) != {"arguments", "receipt_identity"} or set(arguments) != required_args
                    or entry["receipt_identity"] != journal.entries[unit]["receipt_identity"] or arguments["job_id"] != unit
                    or not assembly._same(arguments["protocol"], contract["protocol"])
                    or arguments["output"].replace("\\", "/") != origin + "/work/" + unit
                    or arguments["prepared_directory"].replace("\\", "/") != origin + "/work/prepared_" + window
                    or arguments["preparation_identity"] != context["preparation_identity"]
                    or not assembly._same(arguments["preparation_binding"], preparation["binding"])
                    or arguments["incumbent_identity"] != plan["incumbent"]["identity"]
                    or arguments["incumbent_path"].replace("\\", "/") != origin + "/" + plan["incumbent"]["path"]
                    or (is_joint and (arguments["window"] != window or arguments["input_contract_identity"] != manifest["input_contract_identity"]
                        or arguments["input_contract_path"].replace("\\", "/") != origin + "/" + plan["input_contract"]["path"]))
                    or (not is_joint and (arguments["role"] != role or arguments["arm"] != arm))):
                raise ValueError("result_original_job_binding_mismatch")
            selected = [(o, r) for o, r in context["records"] if h.incumbent.size_label(r) in context["training_joint_counts"]] if is_joint else [(o, r) for o, r in context["records"] if h.target(r, role) is not None]
            loaded[name] = _job(output, manifest, plan, preparation, name, entry, [r for _, r in selected], [o for o, _ in selected],
                set(context["training_actors"][role]), context["training_joint_counts"], window)
            if is_joint:
                report = loaded[name][1]
                fitted = set(context["training_joint_counts"])
                calibration = context["calibration_joint_counts"]
                excluded = dict(calibration={k: v for k, v in calibration.items() if k not in fitted},
                    evaluation=dict(Counter(h.incumbent.size_label(r) for _, r in context["records"] if h.incumbent.size_label(r) not in fitted)))
                if (not assembly._same(report["unsupported_target_counts"], excluded)
                        or report["calibration_supported_rows"] != sum(v for k, v in calibration.items() if k in fitted)):
                    raise ValueError("result_joint_unsupported_denominator_mismatch")
        roles, data = {}, {}
        for role in h.ROLES:
            rows = [r for _, r in context["records"] if h.target(r, role) is not None]
            if len(rows) != preparation["row_counts"][role]["evaluation"]:
                raise ValueError("result_role_denominator_mismatch")
            frames, reference = {}, None
            for arm in h.ARMS:
                values = loaded[role + "/" + arm][2]
                if reference is not None and any(not np.array_equal(reference[k], values[k]) for k in ("frozen_incumbent", "training_prior")):
                    raise ValueError("result_cross_arm_comparator_mismatch")
                reference = values if reference is None else reference
                frames[arm] = assembly._frame(values, list(h.LABELS[role]), role)
            if window == "100":
                frames["same_recipe"] = assembly._frame(reference, list(h.LABELS[role]), role, "frozen_incumbent")
            else:
                values = loaded["same_recipe"][2]
                ids = {row["row_id"] for row in rows}
                selected = np.asarray([i for i, row_id in enumerate(values["row_id"]) if row_id in ids], dtype=np.int64)
                frames["same_recipe"] = assembly._frame(values, paired.JOINT_LABELS, "joint", selected=selected)
            actors = set(context["training_actors"][role])
            roles[role] = paired.compare(window, role, frames, rows, actors, contract["protocol"]["model_config"]["seed"])
            name = "contexts/" + window + "_" + role + ".npy"
            original_name = preparation["attempt"] + "/" + role + "_evaluation.x.npy"
            if manifest["members"][name] != preparation["members"][original_name]:
                raise ValueError("result_fresh_feature_not_original_pin")
            features = control.authenticated_array(output / name, manifest["members"][name])
            model, _, values = loaded[role + "/session"]
            data[role] = dict(model=model, values=values, rows=rows, actors=actors, fresh_features=features)
        comparison = dict(schema_version=assembly.SCHEMA, window=window, evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
            plan_identity=assembly_entry["receipt_identity"], input_contract_identity=manifest["input_contract_identity"],
            preparation_identity=context["preparation_identity"], completion_identities={n: original["jobs"][n]["receipt_identity"] for n in sorted(expected_jobs)},
            model_fitting_performed=False, model_replay_performed=False, preparation_performed=False,
            full_window_same_recipe="REUSE_ACCEPTED_NOTEBOOK03_MODEL", roles=roles)
        unit = "controls_" + window
        entry = journal.entries[unit]
        closed = joint.verified_json(output / "work" / entry["receipt_path"], entry["receipt_identity"])
        if (set(closed) != {"schema_version", "binding", "evidence_class", "authority", "members"}
                or closed["schema_version"] != control.SCHEMA + "_receipt" or closed["evidence_class"] != h.EVIDENCE
                or closed["authority"] != h.AUTHORITY or set(closed["members"]) != set(control.MEMBERS)
                or not assembly._same(closed["binding"], dict(assembly_plan_identity=assembly_entry["receipt_identity"],
                    source_hashes=control.source_hashes(), stale_rule=control.controls.STALE_RULE))):
            raise ValueError("result_control_receipt_mismatch")
        for member, identity in closed["members"].items():
            if manifest["members"].get("work/" + unit + "/" + member) != identity:
                raise ValueError("result_control_original_pin_mismatch")
        directory = output / "work" / unit
        arrays = control._load_arrays(directory / control.MEMBERS[1], closed["members"][control.MEMBERS[1]])
        calculated = control.report(original, comparison, data, arrays)
        stored = joint.verified_json(directory / control.MEMBERS[0], closed["members"][control.MEMBERS[0]])
        if not assembly._same(calculated, stored):
            raise ValueError("result_comparison_control_or_support_arithmetic_mismatch")
        summaries.append(dict(window=window, roles={role: dict(role_rows=roles[role]["role_rows"], common_rows=roles[role]["common_rows"])
                                                   for role in h.ROLES}))
    return dict(schema_version=SCHEMA + "_validation", status="ACCEPT_FOR_DIAGNOSTIC_REVIEW", evidence_class=h.EVIDENCE,
        authority=h.AUTHORITY, manifest_identity=manifest_identity, execution_plan_identity=expected_plan_identity,
        windows=summaries, completed_units=35, model_fitting_performed=False, model_replay_performed=False,
        preparation_performed=False, scientific_qualification=False)


def publish(output, manifest_identity, plan_identity, durable):
    validate(output, manifest_identity, plan_identity)
    manifest = joint.verified_json(Path(output) / MANIFEST, manifest_identity)
    return c.publish(output, dict(manifest["members"], **{MANIFEST: manifest_identity}), durable, "notebook05_results",
        dict(schema_version=SCHEMA, execution_plan_identity=plan_identity, result_manifest_identity=manifest_identity))


def import_bundle(durable, envelope_reference, manifest_identity, plan_identity, output):
    """Fresh quarantine only; a validation failure never emits acceptance."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("result_import_requires_empty_quarantine")
    binding = dict(schema_version=SCHEMA, execution_plan_identity=plan_identity, result_manifest_identity=manifest_identity)
    envelope = c.read_manifest(durable, envelope_reference, binding)
    if envelope["members"].get(MANIFEST) != manifest_identity:
        raise ValueError("result_envelope_manifest_pin_mismatch")
    c.restore(output, durable, envelope_reference, binding)
    manifest = joint.verified_json(output / MANIFEST, manifest_identity)
    if envelope["members"] != dict(manifest["members"], **{MANIFEST: manifest_identity}):
        raise ValueError("result_envelope_exact_inventory_mismatch")
    return validate(output, manifest_identity, plan_identity)
