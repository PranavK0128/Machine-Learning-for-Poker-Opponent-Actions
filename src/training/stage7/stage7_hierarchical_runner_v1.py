"""Colab worker/controller for fixed chronological response comparisons.

The caller must authenticate packaged code before importing this module.
Local execution is reserved for bounded synthetic fixtures. Completed jobs are
never refitted: source, model, preparation, configuration and code hashes bind
each manifest-last job receipt. No Assistant registry is written.
"""
from __future__ import annotations

import os
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
import uuid

import numpy as np
import stage7_hierarchical_response_v1 as h
import stage7_hierarchical_preparation_v1 as prep
import stage6_empirical_range_learning_v2_release as transport


def digest(value):
    return hashlib.sha256(h.incumbent.canonical(value).encode()).hexdigest()


def effective_cpus():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota_capacity = None
    unified = Path("/sys/fs/cgroup/cpu.max")
    legacy_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    legacy_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if unified.exists():
        quota, period = unified.read_text().split()
        if quota != "max":
            quota_capacity = max(1, math.floor(int(quota) / int(period)))
    elif legacy_quota.exists() and legacy_period.exists():
        quota, period = int(legacy_quota.read_text()), int(legacy_period.read_text())
        if quota > 0:
            quota_capacity = max(1, math.floor(quota / period))
    return dict(logical_cpus=os.cpu_count(), affinity_cpus=affinity, quota_cpus=quota_capacity,
                effective_cpus=min(affinity, quota_capacity) if quota_capacity is not None else affinity,
                inner_threads=1, ram_policy="TELEMETRY_ONLY")


def frozen_model(path, expected):
    # Even JSON model parsing occurs after current-byte authentication.
    transport.verify(path, expected)
    artifact = transport.load(path)
    if artifact.get("authority") != h.AUTHORITY:
        raise ValueError("incumbent_authority_changed")
    model = artifact["models"]["observed_next_action_size"]
    if model["feature_names"] != h.incumbent.PUBLIC_NAMES:
        raise ValueError("incumbent_public_feature_schema_changed")
    return model


def score(probabilities, targets, rows, baseline, seed):
    result = h.incumbent.scores(probabilities, np.asarray(targets), rows, baseline, seed)
    result.update(evidence_class=h.EVIDENCE, familywise_confirmation_claim=False,
                  interval_scope="descriptive development; inherited two-role interval is not correction across windows or ablations")
    if not rows:
        return result
    result["diagnostic_slices"] = h.incumbent.diagnostic_slices(probabilities, np.asarray(targets), rows, baseline, set())
    # Classwise reliability is retained alongside top-label ECE, which can hide
    # an under-calibrated minority action in an imbalanced response task.
    classwise = []
    for label in range(probabilities.shape[1]):
        bins = []
        for index in range(10):
            values = probabilities[:, label]
            selected = (values >= index / 10) & (values < (index + 1) / 10 if index < 9 else values <= 1)
            bins.append(dict(bin=index, rows=int(selected.sum()), probability_sum=float(values[selected].sum()),
                             observed_count=int(np.sum(np.asarray(targets)[selected] == label))))
        classwise.append(bins)
    result["classwise_calibration_bins"] = classwise
    return result


def check_job_receipt(output, binding, expected_receipt_identity=None):
    output = Path(output)
    path = output / "job_receipt.json"
    if not path.exists():
        if expected_receipt_identity is not None:
            raise ValueError("expected_job_receipt_missing")
        return None
    if expected_receipt_identity is None:
        raise ValueError("trusted_job_receipt_identity_required")
    transport.verify(path, expected_receipt_identity)
    receipt = transport.load(path)
    if set(receipt) != {"schema_version", "evidence_class", "authority", "binding", "native_portable_parity_rows", "members"}:
        raise ValueError("job_receipt_schema_mismatch")
    if (receipt.get("schema_version") != h.SCHEMA + "_job" or receipt.get("binding") != binding
            or receipt.get("authority") != h.AUTHORITY or receipt.get("evidence_class") != h.EVIDENCE):
        raise ValueError("job_receipt_binding_or_authority_mismatch")
    if set(receipt["members"]) != {"model.json", "scores.json", "predictions.npz"}:
        raise ValueError("job_result_inventory_mismatch")
    for name, expected in receipt["members"].items():
        transport.verify(output / name, expected)
    validate_output_inventory(output)
    return receipt


def validate_output_inventory(output):
    output = Path(output)
    if not output.exists():
        return
    permitted = {"model.json", "scores.json", "predictions.npz", "job_receipt.json"}
    for member in output.iterdir():
        if member.name not in permitted and not any(re.fullmatch(re.escape(name) + r"\.partial\.[a-f0-9]{32}", member.name) for name in permitted):
            raise ValueError("unmanifested_job_output_member")
        if not member.is_file():
            raise ValueError("invalid_job_output_member")


def training_actors_for_role(directory, receipt, role):
    database = Path(directory) / receipt["attempt"] / "events.sqlite"
    con = prep.read_only(database, receipt["members"][database.relative_to(directory).as_posix()]["sha256"])
    try:
        return {row["actor_id"] for compact, in con.execute("SELECT compact_json FROM events WHERE partition='train'")
                if (row := json.loads(compact)) and (role == h.ACTION or row["action"] in h.AGGRESSIVE)}
    finally:
        con.close()


def validate_closed_job(output, arguments, preparation):
    """Check the semantics consumed on resume after external hash checks."""
    output = Path(output)
    role, arm, protocol = arguments["role"], arguments["arm"], arguments["protocol"]
    model, report = transport.load(output / "model.json"), transport.load(output / "scores.json")
    expected_model_keys = {"labels", "feature_names", "fitted_class_indices", "trees", "inference",
                           "schema_version", "role", "arm", "temperature", "temperature_selection",
                           "evidence_class", "authority", "trained_game_ids", "calibration_game_ids"}
    expected_report_keys = {"schema_version", "job_id", "role", "arm", "evidence_class", "authority", "temperature",
                            "training_rows", "calibration_games", "evaluation_games", "labels",
                            "counterfactual_response_identified", "frozen_incumbent", "training_prior"}
    if set(model) != expected_model_keys or set(report) != expected_report_keys:
        raise ValueError("closed_job_exact_schema_mismatch")
    labels = list(h.LABELS[role])
    for value, schema in ((model, h.SCHEMA), (report, h.SCHEMA + "_scores")):
        if (value.get("schema_version") != schema or value.get("role") != role or value.get("arm") != arm
                or value.get("authority") != h.AUTHORITY or value.get("evidence_class") != h.EVIDENCE
                or list(value.get("labels", [])) != labels):
            raise ValueError("closed_job_semantic_contract_mismatch")
    if (model["feature_names"] != h.feature_names(role, arm) or model["temperature"] not in h.TEMPERATURES
            or model["temperature_selection"] != "fixed_grid_calibration_whole_games_only"
            or report["temperature"] != model["temperature"] or report["job_id"] != arguments["job_id"]
            or model["trained_game_ids"] != preparation["game_ids"]["train"]
            or model["calibration_game_ids"] != preparation["game_ids"]["calibration"]
            or report["calibration_games"] != preparation["game_ids"]["calibration"]
            or report["evaluation_games"] != preparation["game_ids"]["evaluation"]
            or report["training_rows"] != preparation["row_counts"][role]["train"]
            or report.get("counterfactual_response_identified") is not False):
        raise ValueError("closed_job_role_or_cohort_mismatch")
    stem = Path(arguments["prepared_directory"]) / preparation["attempt"] / (role + "_evaluation")
    expected_y = np.load(str(stem) + ".y.npy", mmap_mode="r", allow_pickle=False)
    expected_ordinals = np.load(str(stem) + ".ordinal.npy", mmap_mode="r", allow_pickle=False)
    rows = prep.metadata(arguments["prepared_directory"], preparation, expected_ordinals)
    expected_keys = {"candidate", "frozen_incumbent", "training_prior", "target", "ordinal", "row_id", "game_id"}
    with np.load(output / "predictions.npz", allow_pickle=False) as stored:
        if set(stored.files) != expected_keys:
            raise ValueError("closed_job_prediction_inventory_mismatch")
        values = {name: stored[name] for name in stored.files}
    if (values["target"].dtype != np.dtype(np.int16) or values["ordinal"].dtype != np.dtype(np.int64)
            or not np.array_equal(values["target"], expected_y) or not np.array_equal(values["ordinal"], expected_ordinals)
            or values["row_id"].dtype.kind != "U" or values["game_id"].dtype.kind != "U"
            or not np.array_equal(values["row_id"], [r["row_id"] for r in rows])
            or not np.array_equal(values["game_id"], [r["game_id"] for r in rows])
            or len(set(values["row_id"].tolist())) != len(rows)):
        raise ValueError("closed_job_prediction_identity_mismatch")
    for name in ("candidate", "frozen_incumbent", "training_prior"):
        p = values[name]
        if (p.dtype != np.dtype(np.float64) or p.shape != (len(rows), len(labels))
                or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=1e-12)):
            raise ValueError("closed_job_probability_contract_mismatch")
        if not np.allclose(p, h.normalize(p, rows, role), atol=1e-12, rtol=1e-12):
            raise ValueError("closed_job_probability_legality_mismatch")
    actors = training_actors_for_role(arguments["prepared_directory"], preparation, role)
    for name in ("frozen_incumbent", "training_prior"):
        transport.verify_score_arithmetic(report[name], preparation["game_ids"]["evaluation"], protocol["model_config"]["seed"])
        recomputed = score(values["candidate"], values["target"], rows, values[name], protocol["model_config"]["seed"])
        recomputed["diagnostic_slices"] = h.incumbent.diagnostic_slices(values["candidate"], values["target"], rows, values[name], actors)
        if h.incumbent.canonical(report[name]) != h.incumbent.canonical(recomputed):
            raise ValueError("closed_job_scores_do_not_match_prediction_arrays")


def fit_job(arguments):
    directory, output = Path(arguments["prepared_directory"]), Path(arguments["output"])
    if arguments.get("expected_job_receipt_identity") is not None and not (output / "job_receipt.json").is_file():
        raise ValueError("expected_job_receipt_missing")
    validate_output_inventory(output)
    protocol, role, arm = arguments["protocol"], arguments["role"], arguments["arm"]
    if (role not in h.ROLES or arm not in h.ARMS or protocol.get("evidence_class") != h.EVIDENCE
            or protocol.get("authority") != h.AUTHORITY or protocol.get("untouched_confirmation_game_ids") != []):
        raise ValueError("fixed_development_job_required")
    source_hashes = {name: transport.sha(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "stage7_hierarchical_preparation_v1.py", "stage7_hierarchical_response_v1.py",
        "stage6_empirical_range_learning_v2.py", "stage6_empirical_range_learning_v2_release.py")}
    binding = dict({key: value for key, value in arguments.items() if key != "expected_job_receipt_identity"}, source_hashes=source_hashes)
    transport.verify(directory / "preparation_receipt.json", arguments["preparation_identity"])
    receipt = transport.load(directory / "preparation_receipt.json")
    prep.validate_receipt(directory, receipt, arguments["preparation_binding"])
    window = receipt["binding"]["window"]
    expected_groups = dict(train=protocol["training_windows"][window], calibration=protocol["calibration_game_ids"],
                           evaluation=protocol["development_evaluation_game_ids"])
    if (receipt["binding"]["protocol_sha256"] != digest(protocol) or receipt["game_ids"] != expected_groups
            or receipt["feature_names"] != {r: h.feature_names(r, "session") for r in h.ROLES}):
        raise ValueError("job_protocol_preparation_mismatch")
    closed = check_job_receipt(output, binding, arguments.get("expected_job_receipt_identity"))
    if closed is not None:
        transport.verify(arguments["incumbent_path"], arguments["incumbent_identity"])
        validate_closed_job(output, arguments, receipt)
        return dict(job_id=arguments["job_id"], reused=True, receipt=transport.identity(output / "job_receipt.json"))
    old = frozen_model(arguments["incumbent_path"], arguments["incumbent_identity"])
    config = protocol["model_config"]
    if (set(config) != {"iterations", "num_leaves", "learning_rate", "min_child_samples", "seed"}
            or not 1 <= config["iterations"] <= 1000 or not 2 <= config["num_leaves"] <= 127
            or not 0 < config["learning_rate"] <= 1 or config["min_child_samples"] < 1):
        raise ValueError("bounded_frozen_model_configuration_required")
    x, y, _ = prep.matrices(directory, receipt, role, "train", arm)
    if not len(y) or len(np.unique(y)) < 2:
        raise ValueError("insufficient_target_support_requires_explicit_hold")
    model = h.fit_response_booster(x, y, role, config)
    del x
    calibration_x, calibration_y, calibration_ordinals = prep.matrices(directory, receipt, role, "calibration", arm)
    calibration_rows = prep.metadata(directory, receipt, calibration_ordinals)
    if not calibration_rows:
        raise ValueError("calibration_required")
    raw = h.incumbent.full_prediction(model, calibration_x, h.LABELS[role])
    loss = lambda t: float(h.incumbent.loss_rows(h.normalize(h.incumbent.temper(raw, t), calibration_rows, role), calibration_y)[0].mean())
    temperature = min(h.TEMPERATURES, key=lambda t: (loss(t), abs(t - 1), t))
    exported = h.incumbent.booster_export(model, h.LABELS[role], h.feature_names(role, arm))
    exported.update(schema_version=h.SCHEMA, role=role, arm=arm, temperature=temperature,
                    temperature_selection="fixed_grid_calibration_whole_games_only",
                    evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                    trained_game_ids=receipt["game_ids"]["train"], calibration_game_ids=receipt["game_ids"]["calibration"])
    del calibration_x, raw, calibration_rows, calibration_y, calibration_ordinals
    evaluation_x, evaluation_y, evaluation_ordinals = prep.matrices(directory, receipt, role, "evaluation", arm)
    evaluation_rows = prep.metadata(directory, receipt, evaluation_ordinals)
    candidate = h.normalize(h.incumbent.temper(h.incumbent.full_prediction(model, evaluation_x, h.LABELS[role]), temperature), evaluation_rows, role)
    portable = h.normalize(h.batch_tree_prediction(exported, evaluation_x[:min(64, len(evaluation_x))]), evaluation_rows[:64], role)
    if not np.allclose(portable, candidate[:64], atol=1e-10, rtol=1e-10):
        raise ValueError("candidate_native_portable_parity_failed")
    # Public features are the leading exact incumbent columns in every arm.
    joint = h.batch_tree_prediction(old, evaluation_x[:, :len(h.incumbent.PUBLIC_NAMES)])
    frozen = h.collapse_incumbent(joint, old["labels"], evaluation_rows, role)
    prior = np.bincount(y, minlength=len(h.LABELS[role])).astype(float) + 1
    prior /= prior.sum()
    prior = h.normalize(np.tile(prior, (len(evaluation_rows), 1)), evaluation_rows, role)
    training_actors = training_actors_for_role(directory, receipt, role)
    report = dict(schema_version=h.SCHEMA + "_scores", job_id=arguments["job_id"], role=role, arm=arm,
                  evidence_class=h.EVIDENCE, authority=h.AUTHORITY, temperature=temperature,
                  training_rows=len(y), calibration_games=receipt["game_ids"]["calibration"],
                  evaluation_games=receipt["game_ids"]["evaluation"], labels=list(h.LABELS[role]),
                  counterfactual_response_identified=False,
                  frozen_incumbent=score(candidate, evaluation_y, evaluation_rows, frozen, config["seed"]),
                  training_prior=score(candidate, evaluation_y, evaluation_rows, prior, config["seed"]))
    for comparison, baseline in (("frozen_incumbent", frozen), ("training_prior", prior)):
        report[comparison]["diagnostic_slices"] = h.incumbent.diagnostic_slices(candidate, evaluation_y, evaluation_rows, baseline, training_actors)
    output.mkdir(parents=True, exist_ok=True)
    transport.publish_local_bytes(output / "model.json", (h.incumbent.canonical(exported) + "\n").encode())
    transport.publish_local_bytes(output / "scores.json", (h.incumbent.canonical(report) + "\n").encode())
    partial = output / ("predictions.npz.partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        np.savez_compressed(stream, candidate=candidate, frozen_incumbent=frozen, training_prior=prior,
                            target=evaluation_y, ordinal=evaluation_ordinals,
                            row_id=np.asarray([r["row_id"] for r in evaluation_rows]),
                            game_id=np.asarray([r["game_id"] for r in evaluation_rows]))
        stream.flush(); os.fsync(stream.fileno())
    # NPZ zip timestamps are not byte deterministic. Once the verified payload
    # is closed, preserve it and compare arrays on an interrupted retry.
    destination = output / "predictions.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as prior_file, np.load(partial, allow_pickle=False) as current:
            if set(prior_file.files) != set(current.files) or any(not np.array_equal(prior_file[k], current[k]) for k in current.files):
                raise ValueError("different_existing_prediction_arrays")
    else:
        os.link(partial, destination)
    partial.unlink()
    validate_closed_job(output, arguments, receipt)
    closed = dict(schema_version=h.SCHEMA + "_job", evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                  binding=binding, native_portable_parity_rows=len(portable),
                  members={name: transport.identity(output / name) for name in ("model.json", "scores.json", "predictions.npz")})
    validate_output_inventory(output)
    transport.publish_local_bytes(output / "job_receipt.json", (h.incumbent.canonical(closed) + "\n").encode())
    return dict(job_id=arguments["job_id"], reused=False, receipt=transport.identity(output / "job_receipt.json"))


def run_jobs(jobs, workers="auto"):
    """One fresh process per job; deterministic returned order, no RAM cap."""
    if len({job["job_id"] for job in jobs}) != len(jobs) or len({str(Path(job["output"]).resolve()) for job in jobs}) != len(jobs):
        raise ValueError("duplicate_job_or_output")
    capacity = effective_cpus()
    if workers != "auto" and (isinstance(workers, bool) or int(workers) < 1):
        raise ValueError("positive_worker_count_required")
    if not jobs:
        return []
    count = min(capacity["effective_cpus"], len(jobs), capacity["effective_cpus"] if workers == "auto" else int(workers))
    print(h.incumbent.canonical(dict(phase="hierarchical_workers", workers=count, jobs=len(jobs), **capacity)), flush=True)
    completed = {}
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        futures = {pool.submit(fit_job, job): job["job_id"] for job in jobs}
        while futures:
            done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                name = futures.pop(future); completed[name] = future.result()
                print(h.incumbent.canonical(dict(phase="hierarchical_job_complete", **completed[name])), flush=True)
            if not done:
                print(h.incumbent.canonical(dict(phase="hierarchical_jobs_running", pending=len(futures), complete=len(completed))), flush=True)
    return [completed[name] for name in sorted(completed)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--workers", default="auto")
    args = parser.parse_args()
    if transport.sha(args.jobs) != args.expected_sha256:
        raise ValueError("job_plan_hash_mismatch")
    jobs = transport.load(args.jobs)
    print(h.incumbent.canonical(dict(phase="hierarchical_jobs_complete", results=run_jobs(jobs, args.workers))), flush=True)


if __name__ == "__main__":
    main()
