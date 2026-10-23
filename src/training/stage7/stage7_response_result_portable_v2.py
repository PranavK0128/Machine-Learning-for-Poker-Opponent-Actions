"""Local-only portable arithmetic validation of immutable Notebook05 v1 results.

The original producer/importer and all returned bytes remain unchanged. Only
two score-report comparisons in the copied v1 validation functions below use
the bounded scalar policy. Source, schema, lineage, counts, model metadata and
prediction identities retain their original exact checks. No inference occurs.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import struct

import numpy as np
import stage7_response_result_v1 as original

controller, control, c = original.controller, original.control, original.c
assembly, h, prep, runner, joint, auth, t = (
    original.assembly, original.h, original.prep, original.runner,
    original.joint, original.auth, original.t)
paired, SCHEMA = original.paired, original.SCHEMA
_structure, _context, load_npz = original._structure, original._context, original.load_npz
VALIDATOR_SCHEMA = "stage7_response_result_portable_v2"
ORIGINAL_IMPORTER = dict(
    bytes=34597, sha256="199db3d63a95c19320b47c3fb047fad6563088900c4afbac4c9fa763896abeb2")
# Recomputed metric leaves only. Fixed bootstrap quantiles, temperatures, count
# bins, integer denominators and every other scalar remain exact.
ARITHMETIC_FIELDS = frozenset({
    "log_loss", "brier", "accuracy", "expected_calibration_error_10_bins",
    "log_loss_improvement", "brier_improvement", "log_loss_sum", "brier_sum",
    "confidence_sum", "probability_sum",
})
MAX_ULP_STEPS = 512
RELATIVE_TOLERANCE = 1e-15
ABSOLUTE_TOLERANCE = 1e-15


def _ulp_steps(left, right):
    """Exact adjacent binary64 distance for finite numbers of the same sign."""
    a = struct.unpack(">Q", struct.pack(">d", left))[0]
    b = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(a - b)


def _score_same(left, right, path=()):
    """Exact structure/scalars; metric floats require BOTH tight error bounds."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _score_same(left[key], right[key], path + (key,)) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _score_same(a, b, path + (index,)) for index, (a, b) in enumerate(zip(left, right)))
    if type(left) is float:
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        if left.hex() == right.hex():
            return True
        return (bool(path) and path[-1] in ARITHMETIC_FIELDS
                and left != 0.0 and right != 0.0
                and math.copysign(1.0, left) == math.copysign(1.0, right)
                and _ulp_steps(left, right) <= MAX_ULP_STEPS
                and math.isclose(left, right, rel_tol=RELATIVE_TOLERANCE,
                                 abs_tol=ABSOLUTE_TOLERANCE))
    return left == right


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
        if not _score_same(calculated, report[comparator]):
            raise ValueError("result_job_score_arithmetic_mismatch")
    return model, report, values


def _validate_portable(output, manifest_identity, expected_plan_identity):
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
        if not _score_same(calculated, stored):
            raise ValueError("result_comparison_control_or_support_arithmetic_mismatch")
        summaries.append(dict(window=window, roles={role: dict(role_rows=roles[role]["role_rows"], common_rows=roles[role]["common_rows"])
                                                   for role in h.ROLES}))
    return dict(schema_version=SCHEMA + "_validation", status="ACCEPT_FOR_DIAGNOSTIC_REVIEW", evidence_class=h.EVIDENCE,
        authority=h.AUTHORITY, manifest_identity=manifest_identity, execution_plan_identity=expected_plan_identity,
        windows=summaries, completed_units=35, model_fitting_performed=False, model_replay_performed=False,
        preparation_performed=False, scientific_qualification=False)



def validate(output, manifest_identity, expected_plan_identity):
    """Authenticate unchanged v1 provenance, then validate stored arithmetic."""
    t.verify(original.__file__, ORIGINAL_IMPORTER)
    validated = _validate_portable(output, manifest_identity, expected_plan_identity)
    return dict(validated, schema_version=VALIDATOR_SCHEMA + "_validation",
        producer_result_schema=SCHEMA,
        arithmetic_policy=dict(name="finite_metric_binary64_conjunctive_bounds",
            maximum_adjacent_steps=MAX_ULP_STEPS, relative_tolerance=RELATIVE_TOLERANCE,
            absolute_tolerance=ABSOLUTE_TOLERANCE, both_ulp_and_isclose_required=True,
            integer_and_nonnumeric_fields_exact=True,
            fixed_numeric_parameters_exact=True, signed_zero_exact=True),
        validator_identity=t.identity(__file__), original_importer_identity=dict(ORIGINAL_IMPORTER))
