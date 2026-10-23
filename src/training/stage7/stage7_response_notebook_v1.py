"""Notebook05 member-addressable release builder and Colab-only entrypoint.

The builder does no source preparation or model fitting. A release authorization
is separately frozen by the owner after exact-byte independent reviews. The
notebook authenticates every packaged byte before importing this module.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import zipfile

import stage7_response_result_v1 as results

controller, c, h, joint, auth, t = results.controller, results.c, results.h, results.joint, results.auth, results.t
SCHEMA = "stage7_response_notebook_v1"
RUNTIME_NAME = "stage7_hierarchical_response_r1"
NOTEBOOK = "05_Stage7_Hierarchical_Response_R1.ipynb"
DRIVE_STAGE = "/content/drive/MyDrive/Poker Assistant Model/Stage 7 - Empirical Assistant Layers"
PACKAGES = {"numpy": "2.2.6", "scipy": "1.16.3", "scikit-learn": "1.7.2", "lightgbm": "4.6.0", "threadpoolctl": "3.6.0", "pytest": "8.4.2"}
CANARY_TESTS = ("test_stage7_hierarchical_response_v1.py", "test_stage7_joint_recipe_v1.py",
    "test_stage7_response_assembly_v1.py", "test_stage7_response_controller_v1.py", "test_stage7_response_result_v1.py")

# This exact standard-library bootstrap is embedded in the notebook. Small
# code ZIP bytes are authenticated and parsed from the same immutable buffer.
BOOTSTRAP = '''def authenticate_code(package, identity, members, root):
    import hashlib, io, os, pathlib, stat, zipfile, uuid
    raw = pathlib.Path(package).read_bytes()
    if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("notebook_code_package_identity_mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) != len(members) or {i.filename for i in infos} != set(members):
            raise ValueError("notebook_exact_code_inventory_required")
        for item in infos:
            name, expected = item.filename, members[item.filename]
            p = pathlib.PurePosixPath(name)
            if (not name or p.as_posix() != name or p.is_absolute() or ".." in p.parts or "\\\\" in name or ":" in name
                    or item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
                    or item.file_size != expected["bytes"]):
                raise ValueError("notebook_unsafe_code_member")
            data = archive.read(item)
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError("notebook_code_member_identity_mismatch")
            target = pathlib.Path(root) / name
            if any(parent.is_symlink() for parent in (target, *target.parents)):
                raise ValueError("notebook_symlink_destination_forbidden")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("notebook_conflicting_existing_code")
                continue
            partial = target.with_name(target.name + ".partial." + uuid.uuid4().hex)
            with partial.open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.link(partial, target)
            partial.unlink()
    return len(members)
'''

# Standard-library-only parent reporting works during dependency installation,
# silent hash/decompression phases, native fits and result validation alike.
# Printed child completion identities are observed, never discovered/self-hashed.
PROCESS_RUNNER = '''def execute_process(arguments, phase, cwd, interval_seconds=20):
    import json, pathlib, queue, subprocess, threading, time, os
    if not 0 < interval_seconds <= 30:
        raise ValueError("progress_interval_must_be_under_60_seconds")
    started = last_output = time.monotonic()
    messages = queue.Queue()
    last_progress, last_phase, checkpoint = {}, phase, None
    process = subprocess.Popen(arguments,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    def read_output():
        try:
            for line in process.stdout: messages.put(line)
        finally:
            messages.put(None)
    reader = threading.Thread(target=read_output,name="notebook05-output-reader",daemon=True)
    reader.start()
    def report(event):
        observed = {"phase":event,"operation":phase,"last_child_phase":last_phase,"pid":process.pid,
            "elapsed_seconds":round(time.monotonic()-started,3),"seconds_since_child_output":round(time.monotonic()-last_output,3),
            "last_observed_progress":last_progress,"latest_completion_index":checkpoint}
        # These are observed OS counters, not inferred scientific completion.
        io_path = pathlib.Path("/proc") / str(process.pid) / "io"
        try:
            counters = dict(line.split(":",1) for line in io_path.read_text().splitlines())
            observed["process_io_chars"] = {k:int(counters[k]) for k in ("rchar","wchar")}
        except (OSError, ValueError, KeyError):
            observed["process_io_chars"] = None
        print(json.dumps(observed,sort_keys=True),flush=True)
    report("subprocess_started")
    last_report = time.monotonic()
    try:
        while True:
            remaining = max(0.001, interval_seconds - (time.monotonic()-last_report))
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                print(line.rstrip(),flush=True)
                last_output = time.monotonic()
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    value = None
                if isinstance(value,dict) and isinstance(value.get("phase"),str):
                    last_phase = value["phase"]
                    last_progress = {k:value[k] for k in ("unit","window","rows","total","bytes","completed","completed_units","pending","workers") if k in value}
                    if "completion_index" in value:
                        checkpoint = value["completion_index"]
            if time.monotonic()-last_report >= interval_seconds:
                report("subprocess_progress")
                last_report = time.monotonic()
        code = process.wait()
        reader.join(timeout=1)
        process.stdout.close()
        report("subprocess_completed" if code == 0 else "subprocess_failed")
        if code != 0:
            raise RuntimeError("Execution failed with exit code %s; preserve all logs, partials and checkpoints" % code)
        return {"exit_code":code,"latest_completion_index":checkpoint,"last_child_phase":last_phase}
    except BaseException:
        if process.poll() is None:
            # Preserve exact process ownership on notebook interruption; never
            # launch a replacement while this PID/controller may still run.
            report("subprocess_observer_interrupted_child_status_requires_reconciliation")
        raise
'''


def source_identities():
    return dict(results.code_identities(), **{Path(__file__).name: t.identity(__file__)})


def authorization(path, identity, plan_identity):
    value = joint.verified_json(path, identity)
    if (set(value) != {"schema_version", "execution_release", "execution_plan_identity", "code_files", "audit_acceptances"}
            or value["schema_version"] != SCHEMA + "_authorization" or value["execution_release"] is not True
            or value["execution_plan_identity"] != plan_identity or value["code_files"] != source_identities()
            or not isinstance(value["audit_acceptances"], dict) or not value["audit_acceptances"]):
        raise ValueError("notebook_exact_reviewed_execution_authorization_required")
    # Acceptance identities are independently retained owner evidence. Parsing
    # a downloaded self-hashed report never grants execution authorization.
    for name, expected in value["audit_acceptances"].items():
        c.relative(name); auth.validate_identity(expected)
    return value


def zip_members(path, sources):
    """Preserve existing packages only when every decompressed member matches."""
    members = {name: expected for name, (_, expected) in sources.items()}
    c.member_map(members)
    if Path(path).exists():
        c.verify_archive(path, members)
        return t.identity(path), members
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, (source, expected) in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            with Path(source).open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                c._stream(src, dst, expected)
    c.verify_archive(path, members)
    return t.identity(path), members


def extract_pinned(path, transport, members, destination):
    """Authenticated large-file extraction, without loading the archive in RAM."""
    t.verify(path, transport)
    with zipfile.ZipFile(path) as archive:
        c.archive_inventory(archive, members)
        for name, expected in members.items():
            target = Path(destination) / name
            if any(parent.is_symlink() for parent in (target, *target.parents)):
                raise ValueError("notebook_symlink_input_destination_forbidden")
            if target.exists():
                t.verify(target, expected)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with archive.open(name) as src, partial.open("xb") as dst:
                c._stream(src, dst, expected)
                dst.flush(); os.fsync(dst.fileno())
            os.link(partial, target)
            partial.unlink()


def preflight(require_colab=True):
    import importlib.metadata
    if sys.version_info[:2] != (3, 12):
        raise ValueError("notebook_frozen_python_312_required")
    if require_colab and (not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG")):
        raise ValueError("notebook_heavy_execution_is_colab_only")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise ValueError("notebook_dependency_version_mismatch:" + h.incumbent.canonical(versions))
    capacity = controller.runner.effective_cpus()
    memory = Path("/proc/meminfo")
    ram_bytes = None
    if memory.exists():
        fields = dict(line.split(":", 1) for line in memory.read_text().splitlines())
        ram_bytes = int(fields["MemTotal"].split()[0]) * 1024
    value = dict(phase="runtime_preflight", python=sys.version, dependencies=versions,
        ram_bytes=ram_bytes, **capacity)
    print(h.incumbent.canonical(value), flush=True)
    return value


def verify_runtime(root, release_path, release_identity):
    release = joint.verified_json(release_path, release_identity)
    if (set(release) != {"schema_version", "run_id", "execution_plan_identity", "authorization_identity", "code_files",
            "incumbent_archive", "compact_checkpoint_relative", "packages", "canary_files"}
            or release["schema_version"] != SCHEMA + "_release" or release["run_id"] != RUNTIME_NAME
            or release["code_files"] != source_identities() or release["packages"] != PACKAGES
            or set(release["canary_files"]) != set(CANARY_TESTS)
            or release["compact_checkpoint_relative"] != "runtime_state/stage7_range_response_v2_r2/compact_rows.checkpoint.zip"):
        raise ValueError("notebook_fixed_runtime_release_required")
    authorization(Path(root) / "release_authorization.json", release["authorization_identity"], release["execution_plan_identity"])
    for name, expected in release["code_files"].items():
        t.verify(Path(root) / "src" / name, expected)
    for name, expected in release["canary_files"].items():
        t.verify(Path(root) / "tests" / name, expected)
    return release


def stage_inputs(root, stage, release):
    root, stage = Path(root), Path(stage)
    plan = joint.verified_json(root / "execution_plan.json", release["execution_plan_identity"])
    contract = results.assembly._contract(dict(input_contract=dict(path=str(root / plan["input_contract"]["path"]), identity=plan["input_contract"]["identity"])))
    compact = contract["compact_checkpoint"]["manifest"]
    # Manifest bytes were accepted independently in the input contract; the
    # actual accepted archive still needs two complete raw reads in Colab.
    archive = stage / release["compact_checkpoint_relative"]
    local = root / "transport/compact_rows.checkpoint.zip"
    for pass_number in (1, 2):
        t.verify(archive, compact["transport"])
        print(h.incumbent.canonical(dict(phase="accepted_compact_archive_readback", pass_number=pass_number, identity=compact["transport"])), flush=True)
    results.copy_pinned(archive, local, compact["transport"])
    extract_pinned(local, compact["transport"], compact["members"], root / "accepted")
    incumbent = release["incumbent_archive"]
    if (set(incumbent) != {"relative_path", "identity", "members"}
            or incumbent["relative_path"] != "inputs/stage7_05_incumbent_r1.zip"
            or incumbent["members"] != {"accepted/range_response_artifact.json": plan["incumbent"]["identity"]}):
        raise ValueError("notebook_incumbent_archive_contract_mismatch")
    source = stage / incumbent["relative_path"]
    for _ in (1, 2):
        t.verify(source, incumbent["identity"])
    local = root / "transport/incumbent.zip"
    results.copy_pinned(source, local, incumbent["identity"])
    extract_pinned(local, incumbent["identity"], incumbent["members"], root)
    controller.verify_plan(root / "execution_plan.json", release["execution_plan_identity"], root)
    print(h.incumbent.canonical(dict(phase="accepted_inputs_authenticated", source_preparation_recomputed=False,
        compact_archive_raw_readbacks=2, incumbent_archive_raw_readbacks=2)), flush=True)


def execute(root, stage, release_path, release_identity, resume_index=None, resume_result_manifest=None, resume_envelope=None):
    preflight()
    root, stage = Path(root).resolve(), Path(stage)
    release = verify_runtime(root, release_path, release_identity)
    output = root / "result_export"
    if resume_envelope is not None and resume_result_manifest is None:
        raise ValueError("notebook_envelope_resume_requires_external_result_manifest")
    if resume_result_manifest is not None:
        results.validate(output, resume_result_manifest, release["execution_plan_identity"])
        manifest = joint.verified_json(output / results.MANIFEST, resume_result_manifest)
        if resume_index != manifest["completion_index"]:
            raise ValueError("notebook_result_resume_completion_index_mismatch")
        manifest_identity, completion_index = resume_result_manifest, resume_index
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("notebook_existing_export_requires_external_manifest_or_explicit_recovery")
        stage_inputs(root, stage, release)
        durable = stage / "runtime_state" / RUNTIME_NAME
        completion = controller.run(root / "execution_plan.json", release["execution_plan_identity"], root, durable, resume_index)
        # Keep the trusted returned completion in this process through export.
        completion_index = completion["completion_index"]
        manifest_identity = results.export(root, durable, root / "execution_plan.json", release["execution_plan_identity"], completion_index, output)
        print(h.incumbent.canonical(dict(phase="notebook05_export_validated", result_manifest_identity=manifest_identity,
            completion_index=completion_index, execution_plan_identity=release["execution_plan_identity"])), flush=True)
    destination = stage / "returned_results" / "05_hierarchical_response_r1"
    if resume_envelope is None:
        envelope = results.publish(output, manifest_identity, release["execution_plan_identity"], destination)
    else:
        bound = c.read_manifest(destination, resume_envelope, dict(schema_version=results.SCHEMA,
            execution_plan_identity=release["execution_plan_identity"], result_manifest_identity=manifest_identity))
        manifest = joint.verified_json(output / results.MANIFEST, manifest_identity)
        if bound["members"] != dict(manifest["members"], **{results.MANIFEST: manifest_identity}):
            raise ValueError("notebook_result_envelope_member_mismatch")
        for _ in (1, 2):
            t.verify(destination / bound["file_name"], bound["transport"])
        c.verify_archive(destination / bound["file_name"], bound["members"])
        envelope = resume_envelope
    value = dict(phase="notebook05_result_published", result_directory=str(destination), envelope=envelope,
        result_manifest_identity=manifest_identity, execution_plan_identity=release["execution_plan_identity"],
        completion_index=completion_index, external_two_copy_readback_and_strict_import_required=True,
        scientific_qualification=False, authority=h.AUTHORITY)
    print(h.incumbent.canonical(value), flush=True)
    return value


def notebook(package, package_identity, members, release_identity):
    import nbformat
    cells = []
    def pair(title, prose, code):
        cells.append(nbformat.v4.new_markdown_cell("## " + title + "\n\n" + prose))
        cells.append(nbformat.v4.new_code_cell(code))
    cells.append(nbformat.v4.new_markdown_cell("# 05 — Chronological hierarchical observed response\n\n"
        "Execution companion for all 50%, 75%, and 100% training windows, action and conditional sizing roles, four fixed feature arms, "
        "two short-window joint refits, seven paired comparisons, and fixed-model stale-history/support controls. "
        "Accepted Notebook01–04 are reused, never rerun. Outputs are development diagnostics; no untouched future confirmation, "
        "counterfactual strategy EV, promotion, or automatic action is established.\n\n"
        "Run all cells in order in Colab Python 3.12. CPU High-RAM is recommended; authorized v6e CPU use also works. "
        "Actual CPU affinity/quota controls parallelism, and RAM is telemetry only. Results are not yet executed in this generated notebook."))
    pair("1. Connect the authorized runtime", "Mount Drive once. Preserve the independently retained completion index for a cold restart; do not discover and self-hash the newest file.",
        "import os, sys, pathlib, subprocess, json\n"
        "if sys.version_info[:2] != (3, 12): raise RuntimeError('Select Python 3.12')\n"
        "if not pathlib.Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'): raise RuntimeError('Colab required')\n"
        "for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'\n"
        "from google.colab import drive\nif not pathlib.Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')\n"
        f"STAGE=pathlib.Path({DRIVE_STAGE!r})\nROOT=pathlib.Path('/content')/{RUNTIME_NAME!r}\n"
        "RESUME_INDEX=None  # Owner supplies only an externally retained exact index reference after a cold restart.\n"
        "RESUME_RESULT_MANIFEST=None  # Exact printed manifest identity for publication-only recovery.\n"
        "RESUME_ENVELOPE=None  # Exact printed envelope reference for completed publication reuse.\n")
    pair("2. Authenticate the release bytes", "Authenticate the small code ZIP and every member before any packaged import. Original completed files are retained; conflicting bytes stop execution.",
        BOOTSTRAP + f"\nCODE_MEMBERS=json.loads({h.incumbent.canonical(members)!r})\n"
        f"authenticate_code(STAGE/'inputs'/{package!r}, {package_identity!r}, CODE_MEMBERS, ROOT)\n"
        f"RELEASE_ID={release_identity!r}\nprint({{'phase':'code_authenticated','members':len(CODE_MEMBERS)}})\n")
    pair("3. Install frozen dependencies", "Subsequent checks and computation run in fresh subprocesses so already-loaded notebook libraries cannot replace the declared versions.",
        PROCESS_RUNNER + f"\nrequirements={['%s==%s' % item for item in PACKAGES.items()]!r}\n"
        "execute_process([sys.executable,'-m','pip','install','--quiet',*requirements],'install_frozen_dependencies',ROOT)\n")
    pair("4. Check runtime and synthetic canary", "Verify reviewed execution authorization and actual hardware, then run the synthetic end-to-end export/import and no-refit cold-restore canary before accessing real data.",
        "ENTRY=ROOT/'src/stage7_response_notebook_v1.py'\n"
        "execute_process([sys.executable,'-B',str(ENTRY),'preflight','--root',str(ROOT),'--release-identity',json.dumps(RELEASE_ID)],'verify_runtime',ROOT)\n"
        "execute_process([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','tests/test_stage7_response_controller_v1.py','tests/test_stage7_response_result_v1.py'],'synthetic_canary',ROOT)\n")
    pair("5. Run and preserve all windows", "Authenticate the accepted compact checkpoint and incumbent with two complete raw reads, reuse exact completed units, run only unfinished jobs, export compact prediction evidence, and publish payload before the result envelope. Every completion prints its exact durable index reference for the owner to retain outside Colab.",
        "execute_process([sys.executable,'-B',str(ENTRY),'execute','--root',str(ROOT),'--stage',str(STAGE),"
        "'--release-identity',json.dumps(RELEASE_ID),'--resume-index',json.dumps(RESUME_INDEX),"
        "'--resume-result-manifest',json.dumps(RESUME_RESULT_MANIFEST),'--resume-envelope',json.dumps(RESUME_ENVELOPE)],'accepted_inputs_compute_export_publish',ROOT)\n")
    cells.append(nbformat.v4.new_markdown_cell("## 6. External acceptance remains required\n\n"
        "The owner retrieves two complete independent raw copies, verifies the exact envelope and each member, runs strict local import "
        "without refitting or model replay, and obtains independent result review. Preserve any failure. Do not relabel diagnostics as "
        "scientific qualification or enable learned strategy weights. The task heartbeat remains active for the wider nine-layer roadmap."))
    book = nbformat.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
        language_info=dict(name="python", version="3.12"), colab=dict(name=NOTEBOOK)))
    for index, cell in enumerate(book.cells):
        cell.id = "cell-%02d" % index
    nbformat.validate(book)
    return book


def execution_plan(stage, input_contract_identity):
    stage = Path(stage)
    contract_path = stage / "manifests/stage7_05_input_contract_r2.json"
    contract = results.assembly._contract(dict(input_contract=dict(path=str(contract_path), identity=input_contract_identity)))
    return dict(schema_version=controller.SCHEMA + "_plan", run_id=RUNTIME_NAME,
        input_contract=dict(path="input_contract.json", identity=input_contract_identity),
        source_database=dict(path="accepted/fit/prepared/rows.sqlite", identity=contract["compact_checkpoint"]["manifest"]["members"]["fit/prepared/rows.sqlite"]),
        incumbent=dict(path="accepted/range_response_artifact.json", identity={key: contract["accepted_sources"]["range_response_artifact.json"][key] for key in ("bytes", "sha256")}),
        code_files=controller.code_identities(), windows=["50", "75", "100"])


def build(stage, input_contract_identity, authorization_path, authorization_identity):
    """Packaging only; require exact owner authorization after source audits."""
    stage = Path(stage)
    contract_path = stage / "manifests/stage7_05_input_contract_r2.json"
    plan = execution_plan(stage, input_contract_identity)
    raw = (h.incumbent.canonical(plan) + "\n").encode()
    plan_identity = dict(bytes=len(raw), sha256=h.incumbent.hashlib.sha256(raw).hexdigest())
    authorization(authorization_path, authorization_identity, plan_identity)
    plan_path = stage / "manifests/stage7_05_execution_plan_r1.json"
    t.publish_local_bytes(plan_path, raw)
    incumbent = stage / "returned_results/03_range_r2/range_response_artifact.json"
    incumbent_path = stage / "inputs/stage7_05_incumbent_r1.zip"
    incumbent_identity, incumbent_members = zip_members(incumbent_path, {plan["incumbent"]["path"]: (incumbent, plan["incumbent"]["identity"])})
    canary = {name: t.identity(stage / "tests" / name) for name in CANARY_TESTS}
    release = dict(schema_version=SCHEMA + "_release", run_id=RUNTIME_NAME, execution_plan_identity=plan_identity,
        authorization_identity=authorization_identity, code_files=source_identities(), packages=PACKAGES, canary_files=canary,
        incumbent_archive=dict(relative_path="inputs/" + incumbent_path.name, identity=incumbent_identity, members=incumbent_members),
        compact_checkpoint_relative="runtime_state/stage7_range_response_v2_r2/compact_rows.checkpoint.zip")
    release_path = stage / "manifests/stage7_05_runtime_release_r1.json"
    release_identity = results.write_json(release_path, release)
    sources = {"src/" + name: (stage / "src" / name, identity) for name, identity in source_identities().items()}
    sources.update({"tests/" + name: (stage / "tests" / name, identity) for name, identity in canary.items()})
    sources.update({"execution_plan.json": (plan_path, plan_identity), "input_contract.json": (contract_path, input_contract_identity),
                    "release_authorization.json": (authorization_path, authorization_identity), "runtime_release.json": (release_path, release_identity)})
    package_path = stage / "inputs/stage7_05_code_r1.zip"
    package_identity, members = zip_members(package_path, sources)
    import nbformat
    book = notebook(package_path.name, package_identity, members, release_identity)
    notebook_path = stage / "notebooks" / NOTEBOOK
    t.publish_local_bytes(notebook_path, nbformat.writes(book).encode())
    release_record = dict(schema_version=SCHEMA + "_built", execution_release=True, execution_plan_identity=plan_identity,
        notebook=dict(path=notebook_path.relative_to(stage).as_posix(), identity=t.identity(notebook_path)),
        code_package=dict(path=package_path.relative_to(stage).as_posix(), identity=package_identity, members=members),
        incumbent_archive=release["incumbent_archive"], runtime_release_identity=release_identity,
        full_corpus_run=False, external_drive_readback_pending=True)
    results.write_json(stage / "manifests/stage7_05_built_release_r1.json", release_record)
    return release_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path(DRIVE_STAGE))
    parser.add_argument("--release-identity", required=True)
    parser.add_argument("--resume-index", default="null")
    parser.add_argument("--resume-result-manifest", default="null")
    parser.add_argument("--resume-envelope", default="null")
    args = parser.parse_args()
    identity = auth.parse_json(args.release_identity)
    if args.mode == "preflight":
        preflight()
        verify_runtime(args.root, args.root / "runtime_release.json", identity)
    else:
        execute(args.root, args.stage, args.root / "runtime_release.json", identity, auth.parse_json(args.resume_index),
                auth.parse_json(args.resume_result_manifest), auth.parse_json(args.resume_envelope))


if __name__ == "__main__":
    main()
