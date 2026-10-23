"""Build the additive Colab release for behavior-challenger v2 shard nodes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import nbformat

from stage6_behavior_challenger_notebook_runtime_v2 import (
    HEARTBEAT_INTERVAL_SECONDS,
    PREFERRED_NOTEBOOK_HOST,
    notebook_shard_run_id,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    FEATURE_RELEASE_BINDING,
    HALVING_STAGES,
    SCOPES,
    candidate_registry,
    registry_digest,
    search_plan,
    shard_digest,
    validate_feature_release_binding_at_build_time,
)
from stage6_expanded_layers import (
    PINNED_RUNTIME_DISTRIBUTIONS,
    SCIENCE_IMPLEMENTATION_MEMBERS,
    canonical_json_bytes,
    seal_document,
)


STAGE6_ROOT = Path(__file__).resolve().parents[1]
REVISION = "R9"
PACKAGE_NAME = "POKERNOW_ASSISTANT_STAGE6_BEHAVIOR_CHALLENGER_V2_INPUTS_R9.zip"
NOTEBOOK_SPECS = (
    {
        "token": "21",
        "name": "21_Stage6_Behavior_Challenger_V2_Preflop_Shard_0_R9.ipynb",
        "scope": "preflop_behavior",
        "shard": 0,
    },
    {
        "token": "22",
        "name": "22_Stage6_Behavior_Challenger_V2_Preflop_Shard_1_R9.ipynb",
        "scope": "preflop_behavior",
        "shard": 1,
    },
    {
        "token": "23",
        "name": "23_Stage6_Behavior_Challenger_V2_Postflop_Shard_0_R9.ipynb",
        "scope": "postflop_behavior",
        "shard": 0,
    },
    {
        "token": "24",
        "name": "24_Stage6_Behavior_Challenger_V2_Postflop_Shard_1_R9.ipynb",
        "scope": "postflop_behavior",
        "shard": 1,
    },
)
ADDITIVE_SOURCE_MEMBERS = (
    "contracts/stage6_decision_feature_role_manifest_v1.json",
    "contracts/stage6_notebook_dag_v2.json",
    "src/stage6_expanded_notebook_controller.py",
    "src/stage6_expanded_history_acceleration_v2.py",
    "src/stage6_expanded_feature_freeze_v2.py",
    "src/stage6_notebook08_runtime.py",
    "src/stage6_notebook09_runtime.py",
    "src/stage6_notebook11_runtime.py",
    "src/stage6_incumbent_preflop_feature_exporter.ts",
    "src/stage6_incumbent_preflop_feature_packer.py",
    "src/stage6_behavior_json_publication_v2.py",
    "src/stage6_drive_fuse_publication.py",
    "src/stage6_behavior_challenger_tournament_v2.py",
    "src/stage6_behavior_challenger_notebook_runtime_v2.py",
    "src/stage6_behavior_challenger_notebook_builder_v2.py",
    "src/stage6_behavior_challenger_finalist_v2.py",
    "src/stage6_behavior_challenger_fresh_context_v2.py",
    "src/stage6_behavior_challenger_fresh_adjudication_v2.py",
    "src/stage6_behavior_challenger_science_notebook_builder_v2.py",
)
PACKAGE_MEMBER_PATHS = tuple(dict.fromkeys((
    *SCIENCE_IMPLEMENTATION_MEMBERS,
    *ADDITIVE_SOURCE_MEMBERS,
)))


def _fixed_zip_bytes(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return buffer.getvalue()


def package_bytes() -> bytes:
    validate_feature_release_binding_at_build_time()
    members = [
        (relative_path, (STAGE6_ROOT / relative_path).read_bytes())
        for relative_path in PACKAGE_MEMBER_PATHS
    ]
    precommit = seal_document({
        "schema_version": "stage6_behavior_challenger_colab_precommit_v3",
        "status": "FROZEN_BEFORE_BEHAVIOR_DEVELOPMENT_TARGET_READ",
        "revision": REVISION,
        "feature_source": "notebook12_exact_common_base_feature_arrays",
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_release_binding": FEATURE_RELEASE_BINDING,
        "scopes": list(SCOPES),
        "candidate_registries": {
            scope: candidate_registry(scope) for scope in SCOPES
        },
        "candidate_registry_digests": {
            scope: registry_digest(scope) for scope in SCOPES
        },
        "shard_digests": {
            scope: [shard_digest(scope, shard) for shard in (0, 1)]
            for scope in SCOPES
        },
        "search_plans": {scope: search_plan(scope) for scope in SCOPES},
        "notebook_shard_run_ids": {
            scope: [notebook_shard_run_id(scope, shard) for shard in (0, 1)]
            for scope in SCOPES
        },
        "halving_stages": [dict(stage) for stage in HALVING_STAGES],
        "requested_workers_default": "auto",
        "candidate_internal_threads": 1,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
        "accepted_runtime_types": ["cpu", "gpu", "tpu"],
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_tpu_acceleration": False,
        "fresh_targets_opened": False,
        "lockbox_access_allowed": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    members.append((
        "behavior_challenger_v2_search_precommit.json",
        canonical_json_bytes(precommit) + b"\n",
    ))
    package_manifest = seal_document({
        "schema_version": "stage6_behavior_challenger_input_package_manifest_v3",
        "revision": REVISION,
        "search_precommit_self_sha256": precommit["self_sha256"],
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "member_order": [name for name, _ in members],
        "members": [
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in members
        ],
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    members.append((
        "PACKAGE_MANIFEST.json",
        canonical_json_bytes(package_manifest) + b"\n",
    ))
    return _fixed_zip_bytes(members)


def package_identity(raw: bytes | None = None) -> dict[str, Any]:
    value = package_bytes() if raw is None else raw
    with zipfile.ZipFile(io.BytesIO(value)) as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    return {
        "name": PACKAGE_NAME,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "package_manifest_self_sha256": manifest["self_sha256"],
        "search_precommit_self_sha256": manifest[
            "search_precommit_self_sha256"
        ],
    }


def _markdown(value: str) -> Any:
    return nbformat.v4.new_markdown_cell(value)


def _code(value: str) -> Any:
    return nbformat.v4.new_code_cell(value.strip() + "\n")


def _bootstrap_code(package: Mapping[str, Any]) -> str:
    pinned_versions = {
        distribution: version
        for distribution, _, version in PINNED_RUNTIME_DISTRIBUTIONS
    }
    pinned_modules = {
        distribution: module
        for distribution, module, _ in PINNED_RUNTIME_DISTRIBUTIONS
    }
    return f'''import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

PINNED_RUNTIME_DISTRIBUTIONS = {json.dumps(pinned_versions, sort_keys=True)}
PINNED_RUNTIME_MODULES = {json.dumps(pinned_modules, sort_keys=True)}
installed_before = {{
    distribution: (
        importlib.metadata.version(distribution)
        if importlib.util.find_spec(PINNED_RUNTIME_MODULES[distribution]) is not None
        else None
    )
    for distribution in PINNED_RUNTIME_DISTRIBUTIONS
}}
mismatched = {{
    distribution: {{"installed": installed_before[distribution], "required": required}}
    for distribution, required in PINNED_RUNTIME_DISTRIBUTIONS.items()
    if installed_before[distribution] != required
}}
preloaded_mismatched = [
    PINNED_RUNTIME_MODULES[distribution]
    for distribution in mismatched
    if PINNED_RUNTIME_MODULES[distribution] in sys.modules
]
if mismatched:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        *[
            f"{{distribution}}=={{version}}"
            for distribution, version in PINNED_RUNTIME_DISTRIBUTIONS.items()
        ],
    ])
installed_after = {{
    distribution: importlib.metadata.version(distribution)
    for distribution in PINNED_RUNTIME_DISTRIBUTIONS
}}
if installed_after != PINNED_RUNTIME_DISTRIBUTIONS:
    raise RuntimeError(f"PINNED_ENVIRONMENT_INSTALL_FAILED: {{installed_after}}")
if preloaded_mismatched:
    raise RuntimeError(
        "PINNED_ENVIRONMENT_RESTART_REQUIRED: restart this session and Run all; "
        + ",".join(preloaded_mismatched)
    )

try:
    from google.colab import drive
except ImportError as error:
    raise RuntimeError("This release notebook is intended for Google Colab") from error

drive.mount("/content/drive")
STAGE6_DRIVE_ROOT = Path("/content/drive/MyDrive/Poker Assistant Model/Stage 6 - Qualified Layers Improvement")
PACKAGE_PATH = STAGE6_DRIVE_ROOT / "inputs" / "{PACKAGE_NAME}"
EXPECTED_PACKAGE_SHA256 = "{package['sha256']}"
digest = hashlib.sha256()
with PACKAGE_PATH.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != EXPECTED_PACKAGE_SHA256:
    raise RuntimeError("behavior challenger package SHA-256 mismatch")

PACKAGE_ROOT = Path("/content/stage6_behavior_challenger_v2_r9")
if PACKAGE_ROOT.exists():
    shutil.rmtree(PACKAGE_ROOT)
PACKAGE_ROOT.mkdir(parents=True)
with zipfile.ZipFile(PACKAGE_PATH) as archive:
    names = archive.namelist()
    if not names or names[-1] != "PACKAGE_MANIFEST.json" or len(names) != len(set(names)):
        raise RuntimeError("invalid behavior challenger package membership")
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\\\" in name:
            raise RuntimeError(f"unsafe package member: {{name}}")
    archive.extractall(PACKAGE_ROOT)
package_manifest = json.loads(
    (PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
)
if package_manifest["member_order"] != names[:-1]:
    raise RuntimeError("behavior challenger package member order changed")
for member in package_manifest["members"]:
    member_path = PACKAGE_ROOT / member["path"]
    raw = member_path.read_bytes()
    if len(raw) != member["bytes"] or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise RuntimeError(f"package member identity mismatch: {{member['path']}}")

if str(PACKAGE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))
print(json.dumps({{
    "event": "stage6_behavior_challenger_release_ready",
    "package_sha256": EXPECTED_PACKAGE_SHA256,
    "fresh_targets_opened": False,
}}, sort_keys=True))'''


def build_notebook(spec: Mapping[str, Any], package: Mapping[str, Any]) -> Any:
    scope = str(spec["scope"])
    shard = int(spec["shard"])
    token = str(spec["token"])
    run_identity = notebook_shard_run_id(scope, shard)
    cells = [
        _markdown(f"""# Stage 6 — Behavior Challenger v2 · {scope} · shard {shard}

**Notebook {token} · additive behavior search {REVISION}**

{PREFERRED_NOTEBOOK_HOST} is the preferred model-heavy notebook host. CPU,
GPU, and TPU runtimes are accepted, but all eight model families train in
bounded host CPU processes with one thread per model. This notebook does not
claim TPU tree acceleration. It reads only the exact Notebook 12 development
common/base arrays; fresh, holdout, and lockbox targets remain closed."""),
        _markdown("## Operator settings"),
        _code('''REQUESTED_WORKERS = "auto"  # @param {type:"string"}
NOTEBOOK_ACCELERATOR = "TPU"  # @param ["CPU", "GPU", "TPU"]
import os
REQUESTED_WORKERS = os.environ.get(
    "STAGE6_REQUESTED_WORKERS", REQUESTED_WORKERS
).strip()
NOTEBOOK_ACCELERATOR = NOTEBOOK_ACCELERATOR.strip().upper()
print({
    "requested_workers": REQUESTED_WORKERS,
    "worker_rule": "max safe detected host logical CPUs bounded by unfinished candidates and RAM",
    "candidate_internal_threads": 1,
    "accepted_runtime_types": ["CPU", "GPU", "TPU"],
    "preferred_notebook_host": "v6e TPU",
    "model_execution_device": "host_cpu_process",
    "tree_training_uses_tpu_acceleration": False,
})'''),
        _markdown("## Mount Drive and verify the exact additive package"),
        _code(_bootstrap_code(package)),
        _markdown("## Run or resume this deterministic shard"),
        _code(f'''from stage6_behavior_challenger_notebook_runtime_v2 import run_behavior_challenger_shard
result = run_behavior_challenger_shard(
    STAGE6_DRIVE_ROOT,
    scope="{scope}",
    shard={shard},
    requested_workers=REQUESTED_WORKERS,
    accelerator=NOTEBOOK_ACCELERATOR,
    heartbeat_interval_seconds=60.0,
)
RUN_IDENTITY = "{run_identity}"
PRIMARY_MANIFEST_PATH = Path(result["manifest_path"])
print(result)'''),
        _markdown("## Publish and download the compact returned result"),
        _code(f'''from stage6_expanded_notebook_controller import publish_compact_result
returned_result = publish_compact_result(
    STAGE6_DRIVE_ROOT,
    token="{token}",
    run_identity=RUN_IDENTITY,
    result=result,
    primary_manifest_path=PRIMARY_MANIFEST_PATH,
)
print(returned_result)
try:
    from google.colab import files
    files.download(returned_result["path"])
except Exception as error:
    print({{
        "download_status": "manual_download_may_be_required",
        "detail": str(error),
        "path": returned_result["path"],
    }})'''),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "colab": {"name": spec["name"], "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "stage6": {
                "token": token,
                "revision": REVISION,
                "scope": scope,
                "shard": shard,
                "parallel_group": "behavior_challenger_v2_broad_shards",
                "feature_dependency": "notebook12_exact_common_base_feature_arrays",
                "runtime_preference": PREFERRED_NOTEBOOK_HOST,
                "accepted_runtime_types": ["cpu", "gpu", "tpu"],
                "accelerator_required": False,
                "model_execution_device": "host_cpu_process",
                "tree_training_uses_tpu_acceleration": False,
                "requested_workers_default": "auto",
                "worker_rule": (
                    "maximum_safe_detected_host_logical_cpus_bounded_by_"
                    "unfinished_candidates_and_ram"
                ),
                "candidate_internal_threads": 1,
                "candidate_count": 128,
                "halving_evaluated_counts": [128, 48, 16],
                "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                "fresh_targets_opened": False,
                "lockbox_access_allowed": False,
                "qualification_authority": False,
                "recommendation_authority": False,
                "action_authority": False,
                "live_action_authority": False,
                "input_package_sha256": package["sha256"],
                "run_identity": run_identity,
            },
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = hashlib.sha256(
            f"stage6-behavior-v2-r9:{token}:{index}:{cell.cell_type}:{cell.source}".encode(
                "utf-8"
            )
        ).hexdigest()[:8]
    nbformat.validate(notebook)
    return notebook


def build_notebooks(package: Mapping[str, Any] | None = None) -> list[Any]:
    identity = package_identity() if package is None else dict(package)
    return [build_notebook(spec, identity) for spec in NOTEBOOK_SPECS]


def _atomic_write(path: Path, raw: bytes) -> None:
    """Compatibility name: create exclusively or accept identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"conflicting behavior broad release exists: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())


def write_release(output_root: Path) -> dict[str, Any]:
    """Explicit operator action; tests and imports never mutate release folders."""
    raw_package = package_bytes()
    package = package_identity(raw_package)
    package_path = output_root / "inputs" / PACKAGE_NAME
    _atomic_write(package_path, raw_package)
    notebook_rows = []
    for spec in NOTEBOOK_SPECS:
        notebook = build_notebook(spec, package)
        raw = nbformat.writes(notebook).encode("utf-8")
        path = output_root / "notebooks" / str(spec["name"])
        _atomic_write(path, raw)
        notebook_identity = {
            "token": spec["token"],
            "name": path.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "scope": spec["scope"],
            "shard": spec["shard"],
        }
        static = seal_document({
            "schema_version": "stage6_behavior_challenger_notebook_static_manifest_v3",
            "revision": REVISION,
            "notebook": notebook_identity,
            "shared_package": package,
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "run_identity": notebook_shard_run_id(str(spec["scope"]), int(spec["shard"])),
            "fresh_targets_opened": False,
            "qualification_authority": False,
        })
        static_name = f"stage6_notebook{spec['token']}_behavior_v2_static_manifest_r9.json"
        static_path = output_root / "manifests" / static_name
        _atomic_write(
            static_path,
            json.dumps(static, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        notebook_identity["static_manifest"] = {
            "name": static_name,
            "bytes": static_path.stat().st_size,
            "sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
            "self_sha256": static["self_sha256"],
        }
        notebook_rows.append(notebook_identity)
    return {"package": package, "notebooks": notebook_rows}


__all__ = [
    "NOTEBOOK_SPECS",
    "PACKAGE_MEMBER_PATHS",
    "PACKAGE_NAME",
    "REVISION",
    "build_notebook",
    "build_notebooks",
    "package_bytes",
    "package_identity",
    "write_release",
]
