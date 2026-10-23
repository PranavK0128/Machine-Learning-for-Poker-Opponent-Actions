"""Read-only assembly of externally authenticated Notebook05 job completions.

The controller freezes the plan from its prior completion records and supplies
its independent identity. Never hash an untrusted replacement plan to authorize
it. Packaged source is authenticated before import. This module does no model
fitting, model replay, preparation, publication, or automatic checkpoint repair.
Full-corpus aggregation belongs in Colab; local tests use synthetic completions.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import stage7_joint_recipe_v1 as joint
import stage7_paired_response_v1 as paired
import numpy as np

runner, h, prep, auth, t = joint.runner, joint.h, joint.prep, joint.auth, joint.t
SCHEMA = "stage7_response_assembly_v1"
BASE_ARGUMENTS = frozenset(("job_id", "protocol", "output", "prepared_directory", "preparation_binding",
                            "preparation_identity", "incumbent_path", "incumbent_identity"))
ARRAY_NAMES = frozenset(("candidate", "frozen_incumbent", "training_prior", "target", "ordinal", "row_id", "game_id"))


def _same(left, right):
    return h.incumbent.canonical(left) == h.incumbent.canonical(right)


def _json(path, identity):
    return joint.verified_json(path, identity)


def _source_hashes(is_joint):
    names = ["stage7_hierarchical_runner_v1.py", "stage7_hierarchical_preparation_v1.py",
             "stage7_hierarchical_response_v1.py", "stage6_empirical_range_learning_v2.py",
             "stage6_empirical_range_learning_v2_release.py"]
    if is_joint:
        names += ["stage7_joint_recipe_v1.py", "stage7_hierarchical_input_freeze_v1.py"]
    return {name: t.sha(Path(__file__).with_name(name)) for name in names}


def _contract(plan):
    reference = plan["input_contract"]
    if set(reference) != {"path", "identity"}:
        raise ValueError("assembly_input_reference_schema_mismatch")
    contract = _json(reference["path"], reference["identity"])
    if (contract.get("schema_version") != h.SCHEMA + "_input_contract_v2"
            or contract.get("authority") != h.AUTHORITY or contract.get("evidence_class") != h.EVIDENCE
            or contract.get("execution_release") is not False):
        raise ValueError("assembly_input_contract_mismatch")
    protocol = contract["protocol"]
    predecessor = dict(train_game_ids=protocol["train_game_ids"], calibration_game_ids=protocol["calibration_game_ids"],
                       evaluation_game_ids=protocol["development_evaluation_game_ids"],
                       diagnostic_game_ids=protocol["excluded_diagnostic_game_ids"], model_config=protocol["model_config"])
    if not _same(protocol, h.development_protocol(predecessor, contract["game_bounds"])):
        raise ValueError("assembly_exact_chronological_protocol_required")
    if set(contract["accepted_sources"]) != joint.ACCEPTED_SOURCE_NAMES:
        raise ValueError("assembly_accepted_source_inventory_mismatch")
    for name, source in contract["accepted_sources"].items():
        if set(source) != {"path", "bytes", "sha256"} or source["path"] != "returned_results/03_range_r2/" + name:
            raise ValueError("assembly_accepted_source_provenance_mismatch")
        auth.validate_identity({key: source[key] for key in ("bytes", "sha256")})
    return contract


def _preparation(plan, contract):
    reference = plan["preparation"]
    if set(reference) != {"directory", "identity"}:
        raise ValueError("assembly_preparation_reference_schema_mismatch")
    directory = Path(reference["directory"])
    receipt = _json(directory / "preparation_receipt.json", reference["identity"])
    protocol, window = contract["protocol"], plan["window"]
    groups = dict(train=protocol["training_windows"][window], calibration=protocol["calibration_game_ids"],
                  evaluation=protocol["development_evaluation_game_ids"])
    database = contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]
    auth.validate_identity(database)
    binding = receipt["binding"]
    expected = dict(window=window, game_ids=groups, database_sha256=database["sha256"],
                    game_bounds_sha256=runner.digest(contract["game_bounds"]), protocol_sha256=runner.digest(protocol))
    if any(not _same(binding.get(key), value) for key, value in expected.items()):
        raise ValueError("assembly_preparation_not_frozen_source_window")
    prep.validate_receipt(directory, receipt, binding)
    return receipt


def _job(plan, contract, preparation, name):
    entry = plan["jobs"][name]
    if set(entry) != {"arguments", "receipt_identity"}:
        raise ValueError("assembly_external_completion_reference_required")
    auth.validate_identity(entry["receipt_identity"])
    arguments, is_joint = entry["arguments"], name == "same_recipe"
    extra = {"window", "input_contract_path", "input_contract_identity"} if is_joint else {"role", "arm"}
    if set(arguments) != BASE_ARGUMENTS | extra:
        raise ValueError("assembly_exact_job_arguments_required")
    for key in ("job_id", "output", "prepared_directory", "incumbent_path"):
        if not isinstance(arguments[key], str) or not arguments[key]:
            raise ValueError("assembly_nonempty_job_identity_required")
    accepted = {key: contract["accepted_sources"]["range_response_artifact.json"][key] for key in ("bytes", "sha256")}
    if (not _same(arguments["protocol"], contract["protocol"])
            or arguments["incumbent_identity"] != accepted
            or Path(arguments["prepared_directory"]).resolve() != Path(plan["preparation"]["directory"]).resolve()
            or arguments["preparation_identity"] != plan["preparation"]["identity"]
            or not _same(arguments["preparation_binding"], preparation["binding"])):
        raise ValueError("assembly_job_not_bound_to_frozen_inputs")
    if is_joint:
        if (arguments["window"] != plan["window"]
                or arguments["input_contract_identity"] != plan["input_contract"]["identity"]
                or Path(arguments["input_contract_path"]).resolve() != Path(plan["input_contract"]["path"]).resolve()):
            raise ValueError("assembly_joint_input_mismatch")
    elif name != arguments["role"] + "/" + arguments["arm"]:
        raise ValueError("assembly_role_arm_mismatch")
    binding = dict(arguments, source_hashes=_source_hashes(is_joint))
    if is_joint:
        binding["implementation"] = joint.SCHEMA
    output = Path(arguments["output"])
    receipt = _json(output / "job_receipt.json", entry["receipt_identity"])
    if (set(receipt) != {"schema_version", "evidence_class", "authority", "binding", "native_portable_parity_rows", "members"}
            or receipt["schema_version"] != h.SCHEMA + "_job" or receipt["evidence_class"] != h.EVIDENCE
            or receipt["authority"] != h.AUTHORITY or not _same(receipt["binding"], binding)
            or set(receipt["members"]) != {"model.json", "scores.json", "predictions.npz"}):
        raise ValueError("assembly_closed_job_binding_mismatch")
    runner.validate_output_inventory(output)
    # Parse the same buffers whose external receipt supplied the identities.
    model = _json(output / "model.json", receipt["members"]["model.json"])
    report = _json(output / "scores.json", receipt["members"]["scores.json"])
    expected = receipt["members"]["predictions.npz"]
    auth.validate_identity(expected)
    raw, _ = auth.authenticated_bytes(output / "predictions.npz", expected["sha256"], expected["bytes"])
    with np.load(BytesIO(raw), allow_pickle=False) as stored:
        if len(stored.files) != len(ARRAY_NAMES) or set(stored.files) != ARRAY_NAMES:
            raise ValueError("assembly_prediction_inventory_mismatch")
        values = {key: stored[key] for key in stored.files}
    if (type(receipt["native_portable_parity_rows"]) is not int
            or receipt["native_portable_parity_rows"] != min(64, len(values["target"]))):
        raise ValueError("assembly_parity_denominator_mismatch")
    if (model.get("schema_version") != (joint.SCHEMA if is_joint else h.SCHEMA)
            or report.get("schema_version") != (joint.SCHEMA if is_joint else h.SCHEMA) + "_scores"
            or model.get("authority") != h.AUTHORITY or model.get("evidence_class") != h.EVIDENCE
            or report.get("authority") != h.AUTHORITY or report.get("evidence_class") != h.EVIDENCE
            or report.get("job_id") != arguments["job_id"] or report.get("counterfactual_response_identified") is not False
            or model.get("trained_game_ids") != preparation["game_ids"]["train"]
            or model.get("calibration_game_ids") != preparation["game_ids"]["calibration"]):
        raise ValueError("assembly_job_semantic_mismatch")
    role, arm = (joint.ROLE, None) if is_joint else (arguments["role"], arguments["arm"])
    if (model.get("role") != role or report.get("role") != role
            or (not is_joint and (model.get("arm") != arm or report.get("arm") != arm
                                 or model.get("labels") != list(h.LABELS[role])
                                 or model.get("feature_names") != h.feature_names(role, arm)))):
        raise ValueError("assembly_model_role_or_feature_mismatch")
    # Tree payloads are authenticated, not inferred with or retained by the
    # aggregator. Only metadata and prediction arrays cross this boundary.
    return {key: value for key, value in model.items() if key != "trees"}, report, values


def _arrays(values, rows, ordinals, labels, role):
    expected_y = [labels.index(h.incumbent.size_label(row)) if role == "joint" else h.target(row, role) for row in rows]
    if (values["target"].dtype != np.dtype(np.int16) or values["ordinal"].dtype != np.dtype(np.int64)
            or values["row_id"].dtype.kind != "U" or values["game_id"].dtype.kind != "U"
            or not np.array_equal(values["target"], expected_y) or not np.array_equal(values["ordinal"], ordinals)
            or not np.array_equal(values["row_id"], [row["row_id"] for row in rows])
            or not np.array_equal(values["game_id"], [row["game_id"] for row in rows])):
        raise ValueError("assembly_prepared_prediction_identity_mismatch")
    for key in ("candidate", "frozen_incumbent", "training_prior"):
        probabilities = values[key]
        if (probabilities.dtype != np.dtype(np.float64) or probabilities.shape != (len(rows), len(labels))
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=1e-12)):
            raise ValueError("assembly_invalid_probability_payload")
        normalized = (h.incumbent.legal_probabilities(probabilities, rows, labels) if role == "joint"
                      else h.normalize(probabilities, rows, role))
        if not np.allclose(probabilities, normalized, atol=1e-12, rtol=1e-12):
            raise ValueError("assembly_illegal_probability_mass")


def _frame(values, labels, role, key="candidate", selected=None):
    selected = slice(None) if selected is None else selected
    return dict(role=role, labels=labels, probabilities=values[key][selected], target=values["target"][selected],
                row_id=values["row_id"][selected], game_id=values["game_id"][selected])


def assemble_window(plan_path, expected_identity):
    """Assemble one fixed window, authenticated from an external completion plan."""
    plan = _json(plan_path, expected_identity)
    if (set(plan) != {"schema_version", "window", "input_contract", "preparation", "jobs"}
            or plan["schema_version"] != SCHEMA + "_plan" or plan["window"] not in ("50", "75", "100")):
        raise ValueError("assembly_exact_plan_schema_required")
    required = {role + "/" + arm for role in h.ROLES for arm in h.ARMS}
    if plan["window"] != "100":
        required.add("same_recipe")
    if set(plan["jobs"]) != required:
        raise ValueError("assembly_complete_fixed_job_family_required")
    entries = list(plan["jobs"].values())
    if (len({entry["arguments"]["job_id"] for entry in entries}) != len(entries)
            or len({str(Path(entry["arguments"]["output"]).resolve()) for entry in entries}) != len(entries)):
        raise ValueError("assembly_duplicate_job_or_output")
    contract = _contract(plan)
    preparation = _preparation(plan, contract)
    # Verify completion seals before any arithmetic. There is no worker call,
    # missing-job fit fallback, or source/model deserialization for inference.
    loaded = {name: _job(plan, contract, preparation, name) for name in sorted(required)}
    directory = plan["preparation"]["directory"]
    records = list(joint.event_rows(directory, preparation, "evaluation"))
    joint_values = None
    if plan["window"] != "100":
        model, report, joint_values = loaded["same_recipe"]
        counts = joint.training_counts(directory, preparation)
        fitted = sorted(counts)
        selected = [(ordinal, row) for ordinal, row in records if h.incumbent.size_label(row) in fitted]
        if (report.get("labels") != paired.JOINT_LABELS or model.get("labels") != fitted
                or model.get("window") != plan["window"] or report.get("window") != plan["window"]
                or model.get("model_config") != contract["protocol"]["model_config"]
                or report.get("full_window_refitted") is not False
                or model.get("feature_names") != h.incumbent.PUBLIC_NAMES):
            raise ValueError("assembly_joint_training_or_label_contract_mismatch")
        _arrays(joint_values, [row for _, row in selected], [ordinal for ordinal, _ in selected], paired.JOINT_LABELS, "joint")
    results = {}
    for role in h.ROLES:
        selected = [(ordinal, row) for ordinal, row in records if h.target(row, role) is not None]
        rows, ordinals = [row for _, row in selected], [ordinal for ordinal, _ in selected]
        frames, reference = {}, None
        for arm in h.ARMS:
            _, _, values = loaded[role + "/" + arm]
            _arrays(values, rows, ordinals, list(h.LABELS[role]), role)
            if reference is not None and any(not np.array_equal(values[key], reference[key]) for key in ("frozen_incumbent", "training_prior")):
                raise ValueError("assembly_cross_arm_comparator_mismatch")
            reference = values if reference is None else reference
            frames[arm] = _frame(values, list(h.LABELS[role]), role)
        if joint_values is None:
            frames["same_recipe"] = _frame(reference, list(h.LABELS[role]), role, "frozen_incumbent")
        else:
            row_ids = {row["row_id"] for row in rows}
            indices = np.asarray([index for index, row_id in enumerate(joint_values["row_id"]) if row_id in row_ids], dtype=np.int64)
            frames["same_recipe"] = _frame(joint_values, paired.JOINT_LABELS, "joint", selected=indices)
        actors = runner.training_actors_for_role(directory, preparation, role)
        results[role] = paired.compare(plan["window"], role, frames, rows, actors, contract["protocol"]["model_config"]["seed"])
    return dict(schema_version=SCHEMA, window=plan["window"], evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                plan_identity=expected_identity, input_contract_identity=plan["input_contract"]["identity"],
                preparation_identity=plan["preparation"]["identity"],
                completion_identities={name: plan["jobs"][name]["receipt_identity"] for name in sorted(required)},
                model_fitting_performed=False, model_replay_performed=False, preparation_performed=False,
                full_window_same_recipe="REUSE_ACCEPTED_NOTEBOOK03_MODEL", roles=results)
