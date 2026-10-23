"""Short-window public joint-target refits; accepted full-window fit is reused.

Heavy calls belong in Colab. The notebook controller supplies authenticated
preparation and completion identities. No source preparation or registry write
occurs here. Unseen joint targets are explicitly counted, never silently scored.
"""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import uuid

import stage7_hierarchical_runner_v1 as runner  # Sets native inner threads to one.
import stage7_hierarchical_input_freeze_v1 as auth
import numpy as np

h, prep, t = runner.h, runner.prep, runner.transport
SCHEMA = "stage7_joint_recipe_v1"
ROLE = "observed_next_action_size"
SELECTION = "calibration_whole_games_only; fixed_grid_0.5_0.75_1_1.25_1.5_2"
ACCEPTED_SOURCE_NAMES = frozenset(("protocol.json", "range_adapter_receipt.json", "range_response_report.json",
                                    "range_response_artifact.json", "release_plan.json"))


def verified_json(path, identity):
    auth.validate_identity(identity)
    raw, _ = auth.authenticated_bytes(path, identity["sha256"], identity["bytes"])
    return auth.parse_json(raw)


def event_rows(directory, receipt, partition):
    path = Path(directory) / receipt["attempt"] / "events.sqlite"
    con = prep.read_only(path, receipt["members"][path.relative_to(directory).as_posix()]["sha256"])
    try:
        for ordinal, compact in con.execute(
                "SELECT ordinal,compact_json FROM events WHERE partition=? ORDER BY decision,row_id", (partition,)):
            row = auth.parse_json(compact)
            h.validate_observation(row)
            if row["game_id"] not in receipt["game_ids"][partition]:
                raise ValueError("joint_event_game_mismatch")
            yield ordinal, row
    finally:
        con.close()


def training_counts(directory, receipt):
    counts = Counter(h.incumbent.size_label(row) for _, row in event_rows(directory, receipt, "train"))
    if sum(counts.values()) != receipt["row_counts"][h.ACTION]["train"]:
        raise ValueError("joint_training_denominator_mismatch")
    return counts


def partition_data(directory, receipt, partition, labels, materialize_rows=True):
    x, action_y, ordinals = prep.matrices(directory, receipt, h.ACTION, partition, "public_only")
    indices = {label: index for index, label in enumerate(labels)}
    targets = np.full(len(ordinals), -1, dtype=np.int16)
    rows, unsupported = [], Counter()
    count = 0
    for index, (ordinal, row) in enumerate(event_rows(directory, receipt, partition)):
        if index >= len(ordinals) or ordinal != ordinals[index] or h.target(row, h.ACTION) != action_y[index]:
            raise ValueError("joint_prepared_row_order_or_target_mismatch")
        label = h.incumbent.size_label(row)
        targets[index] = indices.get(label, -1)
        if label not in indices:
            unsupported[label] += 1
        elif materialize_rows:
            rows.append(row)
        count += 1
    if count != len(ordinals):
        raise ValueError("joint_partition_denominator_mismatch")
    supported = targets >= 0
    return x[supported], targets[supported], ordinals[supported], rows, dict(sorted(unsupported.items()))


def inputs(arguments):
    directory, protocol, window = Path(arguments["prepared_directory"]), arguments["protocol"], arguments["window"]
    if window not in ("50", "75"):
        raise ValueError("full_window_joint_recipe_must_reuse_accepted_model")
    contract = verified_json(arguments["input_contract_path"], arguments["input_contract_identity"])
    if (contract.get("schema_version") != h.SCHEMA + "_input_contract_v2" or contract.get("authority") != h.AUTHORITY
            or contract.get("evidence_class") != h.EVIDENCE or contract.get("execution_release") is not False
            or h.incumbent.canonical(contract.get("protocol")) != h.incumbent.canonical(protocol)):
        raise ValueError("joint_protocol_input_contract_mismatch")
    sources = contract.get("accepted_sources", {})
    if set(sources) != ACCEPTED_SOURCE_NAMES:
        raise ValueError("joint_exact_accepted_source_inventory_required")
    for name, source in sources.items():
        if (set(source) != {"path", "bytes", "sha256"}
                or source["path"] != "returned_results/03_range_r2/" + name):
            raise ValueError("joint_accepted_source_schema_or_provenance_mismatch")
        auth.validate_identity({key: source[key] for key in ("bytes", "sha256")})
    accepted_incumbent = {key: sources["range_response_artifact.json"][key] for key in ("bytes", "sha256")}
    auth.validate_identity(arguments["incumbent_identity"])
    if arguments["incumbent_identity"] != accepted_incumbent:
        raise ValueError("joint_incumbent_not_accepted_input_contract_artifact")
    # Physical staging paths may differ between Windows and Colab; the exact
    # accepted bytes, not the submitted filename, identify the frozen model.
    accepted_database = contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]
    auth.validate_identity(accepted_database)
    if (protocol.get("schema_version") != h.SCHEMA + "_protocol" or protocol.get("authority") != h.AUTHORITY
            or protocol.get("evidence_class") != h.EVIDENCE or protocol.get("untouched_confirmation_game_ids") != []
            or protocol.get("full_window_same_recipe") != "REUSE_ACCEPTED_NOTEBOOK03_MODEL"):
        raise ValueError("joint_fixed_development_protocol_required")
    groups = dict(train=protocol["training_windows"][window], calibration=protocol["calibration_game_ids"],
                  evaluation=protocol["development_evaluation_game_ids"])
    predecessor = dict(train_game_ids=protocol["train_game_ids"], calibration_game_ids=protocol["calibration_game_ids"],
                       evaluation_game_ids=protocol["development_evaluation_game_ids"],
                       diagnostic_game_ids=protocol["excluded_diagnostic_game_ids"], model_config=protocol["model_config"])
    derived = h.development_protocol(predecessor, contract["game_bounds"])
    if h.incumbent.canonical(protocol) != h.incumbent.canonical(derived):
        raise ValueError("joint_exact_chronological_training_windows_required")
    if not set(groups["train"]) < set(protocol["training_windows"]["100"]):
        raise ValueError("joint_shorter_distinct_training_window_required")
    receipt = verified_json(directory / "preparation_receipt.json", arguments["preparation_identity"])
    prep.validate_receipt(directory, receipt, arguments["preparation_binding"])
    if (receipt["binding"]["window"] != window or receipt["binding"]["protocol_sha256"] != runner.digest(protocol)
            or receipt["binding"]["game_bounds_sha256"] != runner.digest(contract["game_bounds"])
            or receipt["binding"]["database_sha256"] != accepted_database["sha256"]
            or receipt["game_ids"] != groups):
        raise ValueError("joint_protocol_preparation_mismatch")
    config = protocol["model_config"]
    if (set(config) != {"iterations", "num_leaves", "learning_rate", "min_child_samples", "seed"}
            or not 1 <= config["iterations"] <= 1000 or not 2 <= config["num_leaves"] <= 127
            or not 0 < config["learning_rate"] <= 1 or config["min_child_samples"] < 1):
        raise ValueError("joint_bounded_frozen_configuration_required")
    artifact = verified_json(arguments["incumbent_path"], arguments["incumbent_identity"])
    if artifact["authority"] != h.AUTHORITY:
        raise ValueError("joint_incumbent_authority_mismatch")
    old = artifact["models"][ROLE]
    allowed = {a + "@none" for a in h.LABELS[h.ACTION] if a not in h.AGGRESSIVE}
    allowed |= {a + "@" + size for a in h.AGGRESSIVE for size in h.LABELS[h.SIZE]}
    if (old["feature_names"] != h.incumbent.PUBLIC_NAMES or len(old["labels"]) != len(set(old["labels"]))
            or not set(old["labels"]) <= allowed or not old["labels"]):
        raise ValueError("joint_incumbent_label_or_feature_schema_mismatch")
    counts = training_counts(directory, receipt)
    labels = sorted(counts)
    if len(labels) < 2 or not set(labels) <= set(old["labels"]):
        raise ValueError("joint_insufficient_or_incompatible_target_support")
    return receipt, old, labels, counts


def metadata(arguments, receipt, labels, counts, temperature):
    prior = np.array([counts[label] + 1 for label in labels], dtype=float)
    prior /= prior.sum()
    return dict(schema_version=SCHEMA, role=ROLE, window=arguments["window"],
                evidence_class=h.EVIDENCE, authority=h.AUTHORITY, feature_names=h.incumbent.PUBLIC_NAMES,
                labels=labels, fitted_class_indices=list(range(len(labels))),
                model_config=arguments["protocol"]["model_config"], trained_game_ids=receipt["game_ids"]["train"],
                calibration_game_ids=receipt["game_ids"]["calibration"], training_target_prior=prior.tolist(),
                temperature=temperature, temperature_selection=SELECTION,
                inference="LightGBM raw tree sum then logistic/softmax; legality normalization")


def evaluate(exported, arguments, receipt, old, labels, counts):
    expected_metadata = metadata(arguments, receipt, labels, counts, exported.get("temperature"))
    if (set(exported) != set(expected_metadata) | {"trees"}
            or h.incumbent.canonical({key: exported[key] for key in expected_metadata}) != h.incumbent.canonical(expected_metadata)
            or exported["temperature"] not in h.TEMPERATURES):
        raise ValueError("joint_exact_model_contract_mismatch")
    x, y, ordinals, rows, unsupported = partition_data(arguments["prepared_directory"], receipt, "evaluation", labels)
    # Align output columns to the full frozen schema for compatible paired
    # scores. A joint class absent in this window remains zero, not invented.
    candidate_local = h.incumbent.legal_probabilities(h.batch_tree_prediction(exported, x), rows, labels)
    candidate = np.zeros((len(rows), len(old["labels"])), dtype=float)
    prior = np.zeros_like(candidate)
    for index, label in enumerate(labels):
        candidate[:, old["labels"].index(label)] = candidate_local[:, index]
        prior[:, old["labels"].index(label)] = exported["training_target_prior"][index]
    prior = h.incumbent.legal_probabilities(prior, rows, old["labels"])
    frozen = h.incumbent.legal_probabilities(h.batch_tree_prediction(old, x), rows, old["labels"])
    targets = np.asarray([old["labels"].index(labels[int(value)]) for value in y], dtype=np.int16)
    _, _, _, calibration, calibration_unsupported = partition_data(arguments["prepared_directory"], receipt, "calibration", labels)
    report = dict(schema_version=SCHEMA + "_scores", job_id=arguments["job_id"], role=ROLE, window=arguments["window"],
                  evidence_class=h.EVIDENCE, authority=h.AUTHORITY, labels=list(old["labels"]), fitted_labels=labels,
                  training_rows=sum(counts.values()), temperature=exported["temperature"],
                  game_ids=receipt["game_ids"], calibration_supported_rows=len(calibration),
                  evaluation_supported_rows=len(rows), unsupported_target_counts=dict(calibration=calibration_unsupported, evaluation=unsupported),
                  support_policy="score only joint targets observed in this training window; retain all excluded counts",
                  counterfactual_response_identified=False, full_window_refitted=False,
                  frozen_incumbent=runner.score(candidate, targets, rows, frozen, arguments["protocol"]["model_config"]["seed"]),
                  training_prior=runner.score(candidate, targets, rows, prior, arguments["protocol"]["model_config"]["seed"]))
    actors = runner.training_actors_for_role(arguments["prepared_directory"], receipt, h.ACTION)
    if rows:
        for name, baseline in (("frozen_incumbent", frozen), ("training_prior", prior)):
            report[name]["diagnostic_slices"] = h.incumbent.diagnostic_slices(candidate, targets, rows, baseline, actors)
    arrays = dict(candidate=candidate, frozen_incumbent=frozen, training_prior=prior, target=targets, ordinal=ordinals,
                  row_id=np.asarray([row["row_id"] for row in rows], dtype=str), game_id=np.asarray([row["game_id"] for row in rows], dtype=str))
    return report, arrays, candidate_local


def validate_completed(output, arguments, receipt, old, labels, counts, members):
    model = verified_json(output / "model.json", members["model.json"])
    report = verified_json(output / "scores.json", members["scores.json"])
    expected_report, arrays, _ = evaluate(model, arguments, receipt, old, labels, counts)
    if h.incumbent.canonical(report) != h.incumbent.canonical(expected_report):
        raise ValueError("joint_score_or_semantic_contract_mismatch")
    for name in ("frozen_incumbent", "training_prior"):
        t.verify_score_arithmetic(report[name], receipt["game_ids"]["evaluation"], arguments["protocol"]["model_config"]["seed"])
    t.verify(output / "predictions.npz", members["predictions.npz"])
    with np.load(output / "predictions.npz", allow_pickle=False) as stored:
        if set(stored.files) != set(arrays) or any(stored[name].dtype != arrays[name].dtype
                or not np.array_equal(stored[name], arrays[name]) for name in arrays):
            raise ValueError("joint_prediction_payload_mismatch")
    return expected_report


def fit_joint_job(arguments):
    output = Path(arguments["output"])
    expected = arguments.get("expected_job_receipt_identity")
    if expected is not None and not (output / "job_receipt.json").is_file():
        raise ValueError("expected_job_receipt_missing")
    runner.validate_output_inventory(output)
    receipt, old, labels, counts = inputs(arguments)
    source_hashes = {name: t.sha(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "stage7_hierarchical_runner_v1.py", "stage7_hierarchical_input_freeze_v1.py",
        "stage7_hierarchical_preparation_v1.py", "stage7_hierarchical_response_v1.py",
        "stage6_empirical_range_learning_v2.py", "stage6_empirical_range_learning_v2_release.py")}
    binding = dict({key: value for key, value in arguments.items() if key != "expected_job_receipt_identity"},
                   implementation=SCHEMA, source_hashes=source_hashes)
    closed = runner.check_job_receipt(output, binding, expected)
    if closed is not None:
        report = validate_completed(output, arguments, receipt, old, labels, counts, closed["members"])
        if type(closed["native_portable_parity_rows"]) is not int or closed["native_portable_parity_rows"] != min(64, report["evaluation_supported_rows"]):
            raise ValueError("joint_parity_row_count_mismatch")
        return dict(job_id=arguments["job_id"], reused=True, receipt=t.identity(output / "job_receipt.json"))
    x, y, _, _, unsupported = partition_data(arguments["prepared_directory"], receipt, "train", labels, False)
    if unsupported:
        raise ValueError("joint_training_target_inventory_changed")
    # Labels are exactly the sorted distinct training targets, matching the
    # accepted recipe and avoiding a new class-space or family search.
    cx, cy, _, calibration, _ = partition_data(arguments["prepared_directory"], receipt, "calibration", labels)
    if not len(cy):
        raise ValueError("joint_calibration_has_no_supported_targets")
    model = h.incumbent.fit_booster(x, y, labels, arguments["protocol"]["model_config"])
    del x, y
    cp = h.incumbent.full_prediction(model, cx, labels)
    temperature = min(h.TEMPERATURES, key=lambda value: (
        float(h.incumbent.loss_rows(h.incumbent.legal_probabilities(h.incumbent.temper(cp, value), calibration, labels), cy)[0].mean()), abs(value - 1)))
    exported = h.incumbent.booster_export(model, labels, h.incumbent.PUBLIC_NAMES)
    exported.update(metadata(arguments, receipt, labels, counts, temperature))
    del cx, cy, cp, calibration
    report, arrays, portable = evaluate(exported, arguments, receipt, old, labels, counts)
    ex, _, _, evaluation, _ = partition_data(arguments["prepared_directory"], receipt, "evaluation", labels)
    parity_rows = min(64, len(evaluation))
    if parity_rows:
        native = h.incumbent.legal_probabilities(h.incumbent.temper(h.incumbent.full_prediction(model, ex[:parity_rows], labels), temperature), evaluation[:parity_rows], labels)
        if not np.allclose(native, portable[:parity_rows], atol=1e-10, rtol=1e-10):
            raise ValueError("joint_native_portable_parity_failed")
    del ex, model, evaluation
    output.mkdir(parents=True, exist_ok=True)
    t.publish_local_bytes(output / "model.json", (h.incumbent.canonical(exported) + "\n").encode())
    t.publish_local_bytes(output / "scores.json", (h.incumbent.canonical(report) + "\n").encode())
    partial = output / ("predictions.npz.partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    destination = output / "predictions.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(not np.array_equal(previous[name], arrays[name]) for name in arrays):
                raise ValueError("different_existing_joint_prediction_arrays")
    else:
        os.link(partial, destination)
    partial.unlink()
    members = {name: t.identity(output / name) for name in ("model.json", "scores.json", "predictions.npz")}
    validate_completed(output, arguments, receipt, old, labels, counts, members)
    runner.validate_output_inventory(output)
    closed = dict(schema_version=h.SCHEMA + "_job", evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                  binding=binding, native_portable_parity_rows=parity_rows, members=members)
    t.publish_local_bytes(output / "job_receipt.json", (h.incumbent.canonical(closed) + "\n").encode())
    return dict(job_id=arguments["job_id"], reused=False, receipt=t.identity(output / "job_receipt.json"))
