"""Notebook05 staged preparation, process workers, checkpoints and comparisons.

The wrapper authenticates code/inputs before invoking this controller in Colab.
Cold resume requires an independently retained completion-index reference.
Local execution is reserved for the synthetic canary. No live registry is used.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import time

import stage7_response_control_job_v1 as control
import stage7_response_checkpoints_v1 as checkpoints

assembly, h, prep, runner, joint, auth, t = control.assembly, control.h, control.prep, control.runner, control.joint, control.auth, control.t
SCHEMA = "stage7_response_controller_v1"


def code_identities():
    names = set(control.source_hashes()) | {Path(__file__).name, "stage7_response_checkpoints_v1.py"}
    return {name: t.identity(Path(__file__).with_name(name)) for name in sorted(names)}


def verify_plan(plan_path, expected_identity, root):
    plan = joint.verified_json(plan_path, expected_identity)
    if (set(plan) != {"schema_version", "run_id", "input_contract", "source_database", "incumbent", "code_files", "windows"}
            or plan["schema_version"] != SCHEMA + "_plan" or plan["run_id"] != "stage7_hierarchical_response_r1"
            or plan["windows"] != ["50", "75", "100"] or plan["code_files"] != code_identities()):
        raise ValueError("controller_fixed_execution_plan_required")
    for name in ("input_contract", "source_database", "incumbent"):
        reference = plan[name]
        if set(reference) != {"path", "identity"}:
            raise ValueError("controller_exact_input_reference_required")
        checkpoints.relative(reference["path"]); auth.validate_identity(reference["identity"])
    source_reference = plan["input_contract"]
    contract = assembly._contract({"input_contract": dict(path=str(root / source_reference["path"]), identity=source_reference["identity"])})
    accepted_model = {key: contract["accepted_sources"]["range_response_artifact.json"][key] for key in ("bytes", "sha256")}
    database = contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]
    if plan["incumbent"]["identity"] != accepted_model or plan["source_database"]["identity"] != database:
        raise ValueError("controller_input_not_accepted_source")
    # The exact accepted source inputs are authenticated before work creation.
    t.verify(root / plan["source_database"]["path"], database)
    t.verify(root / plan["incumbent"]["path"], accepted_model)
    return plan, contract


@contextmanager
def controller_lock(root, binding):
    path = root / "controller.lock"
    root.mkdir(parents=True, exist_ok=True)
    # Never infer a stale lease from elapsed wall time. An interrupted owner's
    # exact lock is retained for operator process/liveness reconciliation.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(h.incumbent.canonical(dict(pid=os.getpid(), binding=binding)) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    try:
        yield
    finally:
        path.unlink()


def seal_unit(journal, unit, receipt_relative, receipt_identity, members):
    if unit in journal.entries:
        raise ValueError("completed_controller_unit_cannot_be_resealed")
    t.verify(journal.root / receipt_relative, receipt_identity)
    checkpoints.member_map(members)
    if members.get(receipt_relative) != receipt_identity:
        raise ValueError("controller_seal_receipt_inventory_mismatch")
    reference = checkpoints.publish(journal.root, members, journal.durable, unit,
        dict(controller=journal.binding, unit=unit, receipt_identity=receipt_identity))
    return journal.record(unit, reference, receipt_relative, receipt_identity)


def prepare_window(plan, contract, root, journal, window):
    unit, directory = "prepare_" + window, journal.root / ("prepared_" + window)
    receipt_relative = (directory / "preparation_receipt.json").relative_to(journal.root).as_posix()
    expected = journal.restore_unit(unit, receipt_relative) if unit in journal.entries else None
    binding = dict(execution_plan_identity=journal.binding["execution_plan_identity"])
    receipt = prep.prepare(root / plan["source_database"]["path"], plan["source_database"]["identity"]["sha256"],
        contract["protocol"], contract["game_bounds"], directory, binding, window=window, expected_receipt_identity=expected)
    identity = expected or t.identity(directory / "preparation_receipt.json")
    if unit not in journal.entries:
        if not assembly._same(joint.verified_json(directory / "preparation_receipt.json", identity), receipt):
            raise ValueError("new_preparation_return_does_not_match_receipt")
        members = {receipt_relative: identity, **{(directory / name).relative_to(journal.root).as_posix(): expected
                   for name, expected in receipt["members"].items()}}
        seal_unit(journal, unit, receipt_relative, identity, members)
    return dict(directory=str(directory), identity=identity), receipt


def job_arguments(plan, contract, root, preparation, receipt, window):
    jobs = {}
    for role in h.ROLES:
        for arm in h.ARMS:
            unit = role + "_" + arm + "_" + window
            jobs[role + "/" + arm] = dict(job_id=unit, protocol=contract["protocol"], role=role, arm=arm,
                output=str(root / "work" / unit), prepared_directory=preparation["directory"],
                preparation_binding=receipt["binding"], preparation_identity=preparation["identity"],
                incumbent_path=str(root / plan["incumbent"]["path"]), incumbent_identity=plan["incumbent"]["identity"])
    if window != "100":
        unit = "joint_" + window
        jobs["same_recipe"] = dict(job_id=unit, protocol=contract["protocol"], window=window, output=str(root / "work" / unit),
            prepared_directory=preparation["directory"], preparation_binding=receipt["binding"], preparation_identity=preparation["identity"],
            incumbent_path=str(root / plan["incumbent"]["path"]), incumbent_identity=plan["incumbent"]["identity"],
            input_contract_path=str(root / plan["input_contract"]["path"]), input_contract_identity=plan["input_contract"]["identity"])
    return jobs


def _worker(arguments, is_joint):
    return joint.fit_joint_job(arguments) if is_joint else runner.fit_job(arguments)


def run_window_jobs(jobs, journal, workers):
    completed, pending = {}, {}
    for name, arguments in jobs.items():
        unit, output = arguments["job_id"], Path(arguments["output"])
        relative = (output / "job_receipt.json").relative_to(journal.root).as_posix()
        if unit in journal.entries:
            expected = journal.restore_unit(unit, relative)
            binding = dict(arguments, source_hashes=assembly._source_hashes(name == "same_recipe"))
            if name == "same_recipe":
                binding["implementation"] = joint.SCHEMA
            runner.check_job_receipt(output, binding, expected)
            # Read-only structural/semantic assembly validates these completions
            # together. No model inference is rerun merely to resume the queue.
            completed[name] = dict(arguments=arguments, receipt_identity=expected)
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("orphan_worker_output_requires_explicit_recovery:" + unit)
            pending[name] = arguments
    if not pending:
        return completed
    capacity = runner.effective_cpus()
    count = min(len(pending), capacity["effective_cpus"], capacity["effective_cpus"] if workers == "auto" else workers)
    print(h.incumbent.canonical(dict(phase="response_worker_queue", pending=len(pending), workers=count, **capacity)), flush=True)
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        futures = {pool.submit(_worker, arguments, name == "same_recipe"): name for name, arguments in pending.items()}
        while futures:
            done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda value: futures[value]):
                name = futures.pop(future)
                result, arguments = future.result(), pending[name]
                if result["job_id"] != arguments["job_id"] or result["reused"] is not False:
                    raise ValueError("controller_worker_completion_mismatch")
                unit, output = arguments["job_id"], Path(arguments["output"])
                receipt_relative = (output / "job_receipt.json").relative_to(journal.root).as_posix()
                closed = joint.verified_json(output / "job_receipt.json", result["receipt"])
                members = {receipt_relative: result["receipt"], **{(output / member).relative_to(journal.root).as_posix(): expected
                           for member, expected in closed["members"].items()}}
                seal_unit(journal, unit, receipt_relative, result["receipt"], members)
                completed[name] = dict(arguments=arguments, receipt_identity=result["receipt"])
            if not done:
                print(h.incumbent.canonical(dict(phase="response_jobs_in_flight", pending=len(futures),
                                                 completed=len(completed), completion_index=journal.reference)), flush=True)
    return completed


def compare_window(plan, root, journal, window, preparation, jobs):
    unit = "assembly_plan_" + window
    value = dict(schema_version=assembly.SCHEMA + "_plan", window=window,
        input_contract=dict(path=str(root / plan["input_contract"]["path"]), identity=plan["input_contract"]["identity"]),
        preparation=preparation, jobs=jobs)
    path = journal.root / ("assembly_plan_" + window + ".json")
    if unit in journal.entries:
        identity = journal.restore_unit(unit, path.relative_to(journal.root).as_posix())
        if not assembly._same(joint.verified_json(path, identity), value):
            raise ValueError("restored_assembly_plan_does_not_match_completions")
    else:
        t.publish_local_bytes(path, (h.incumbent.canonical(value) + "\n").encode())
        identity = t.identity(path)
        seal_unit(journal, unit, path.relative_to(journal.root).as_posix(), identity, {path.relative_to(journal.root).as_posix(): identity})
    unit, output = "controls_" + window, journal.root / ("controls_" + window)
    receipt_relative = (output / control.RECEIPT).relative_to(journal.root).as_posix()
    expected = journal.restore_unit(unit, receipt_relative) if unit in journal.entries else None
    result = control.run(path, identity, output, expected)
    if unit not in journal.entries:
        closed = joint.verified_json(output / control.RECEIPT, result["receipt"])
        seal_unit(journal, unit, receipt_relative, result["receipt"],
                  {receipt_relative: result["receipt"], **{(output / name).relative_to(journal.root).as_posix(): expected
                   for name, expected in closed["members"].items()}})
    return dict(window=window, assembly_plan=dict(path=path.relative_to(journal.root).as_posix(), identity=identity),
                control_receipt=dict(path=receipt_relative, identity=result["receipt"]))


def run(plan_path, expected_plan_identity, root, durable, expected_index=None, workers="auto"):
    root, durable = Path(root).resolve(), Path(durable).resolve()
    if workers != "auto" and (type(workers) is not int or workers < 1):
        raise ValueError("controller_positive_worker_count_required")
    if root == durable or root in durable.parents or durable in root.parents:
        raise ValueError("independent_runtime_and_durable_directories_required")
    plan, contract = verify_plan(plan_path, expected_plan_identity, root)
    binding = dict(schema_version=SCHEMA, run_id=plan["run_id"], execution_plan_identity=expected_plan_identity)
    work = root / "work"
    journal = checkpoints.Journal(work, durable, binding, expected_index)
    if expected_index is None and work.exists() and any(work.iterdir()):
        raise ValueError("existing_work_requires_external_resume_or_explicit_recovery")
    allowed_units = {"prepare_" + w for w in plan["windows"]} | {"assembly_plan_" + w for w in plan["windows"]} | {"controls_" + w for w in plan["windows"]}
    allowed_units |= {role + "_" + arm + "_" + w for role in h.ROLES for arm in h.ARMS for w in plan["windows"]} | {"joint_50", "joint_75"}
    if not set(journal.entries) <= allowed_units:
        raise ValueError("controller_unknown_completed_unit")
    for index, window in enumerate(plan["windows"]):
        worker_units = {role + "_" + arm + "_" + window for role in h.ROLES for arm in h.ARMS}
        if window != "100":
            worker_units.add("joint_" + window)
        window_units = worker_units | {"prepare_" + window, "assembly_plan_" + window, "controls_" + window}
        present = set(journal.entries)
        if ((present & worker_units and "prepare_" + window not in present)
                or ("assembly_plan_" + window in present and not worker_units <= present)
                or ("controls_" + window in present and "assembly_plan_" + window not in present)
                or (index and present & window_units and "controls_" + plan["windows"][index - 1] not in present)):
            raise ValueError("controller_completion_dependency_hole")
    with controller_lock(work, binding):
        results = []
        for window in plan["windows"]:
            preparation, receipt = prepare_window(plan, contract, root, journal, window)
            jobs = job_arguments(plan, contract, root, preparation, receipt, window)
            completed = run_window_jobs(jobs, journal, workers)
            results.append(compare_window(plan, root, journal, window, preparation, completed))
        if set(journal.entries) != allowed_units:
            raise ValueError("controller_terminal_completion_inventory_mismatch")
        result = dict(schema_version=SCHEMA + "_result", run_id=plan["run_id"], evidence_class=h.EVIDENCE,
            authority=h.AUTHORITY, execution_plan_identity=expected_plan_identity, completion_index=journal.reference,
            windows=results, completed_units=len(journal.entries), full_window_refitted=False)
        print(h.incumbent.canonical(dict(phase="notebook05_computations_complete", **result)), flush=True)
        return result
