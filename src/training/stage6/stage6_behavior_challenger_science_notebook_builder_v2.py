"""Deterministic Colab builders for behavior-v2 finalist and fresh science."""

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

from stage6_behavior_challenger_finalist_v2 import (
    FRESH_CUTOFF_UTC,
    REQUIRED_BASELINES,
    aggregate_run_id,
)
from stage6_behavior_challenger_notebook_builder_v2 import (
    PACKAGE_NAME as BROAD_PACKAGE_NAME,
    _bootstrap_code as _broad_bootstrap_code,
    _code,
    _fixed_zip_bytes,
    _markdown,
)
from stage6_behavior_challenger_notebook_runtime_v2 import (
    HEARTBEAT_INTERVAL_SECONDS,
    PREFERRED_NOTEBOOK_HOST,
)
from stage6_behavior_challenger_tournament_v2 import (
    BEHAVIOR_IMPLEMENTATION_BINDING,
    BEHAVIOR_IMPLEMENTATION_DIGEST,
    FEATURE_RELEASE_BINDING,
    validate_feature_release_binding_at_build_time,
)
from stage6_expanded_layers import (
    SCIENCE_IMPLEMENTATION_MEMBERS,
    canonical_json_bytes,
    seal_document,
)


STAGE6_ROOT = Path(__file__).resolve().parents[1]
REVISION = "R14"
PACKAGE_NAME = "POKERNOW_ASSISTANT_STAGE6_BEHAVIOR_CHALLENGER_V2_SCIENCE_R14.zip"
NOTEBOOK_SPECS = (
    {
        "token": "25",
        "name": "25_Stage6_Behavior_Challenger_V2_Finalist_Aggregate_R14.ipynb",
        "phase": "finalist_aggregate",
    },
    {
        "token": "26",
        "name": "26_Stage6_Behavior_Challenger_V2_Fresh_Context_R14.ipynb",
        "phase": "fresh_context",
    },
    {
        "token": "27",
        "name": "27_Stage6_Behavior_Challenger_V2_Fresh_Adjudication_R14.ipynb",
        "phase": "fresh_adjudication",
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
FROZEN_FEATURE_RELEASE_MEMBER_PATHS = tuple(
    str(FEATURE_RELEASE_BINDING[key]["relative_path"])
    for key in (
        "feature_runtime_source",
        "feature_release_builder",
        "input_package",
        "notebook",
        "static_manifest",
    )
)
PACKAGE_MEMBER_PATHS = tuple(dict.fromkeys((
    *SCIENCE_IMPLEMENTATION_MEMBERS,
    *ADDITIVE_SOURCE_MEMBERS,
    *FROZEN_FEATURE_RELEASE_MEMBER_PATHS,
)))


def package_bytes() -> bytes:
    validate_feature_release_binding_at_build_time()
    members = [
        (relative, (STAGE6_ROOT / relative).read_bytes())
        for relative in PACKAGE_MEMBER_PATHS
    ]
    precommit = seal_document({
        "schema_version": "stage6_behavior_challenger_science_precommit_v3",
        "status": "FROZEN_BEFORE_FINALIST_CALIBRATION_OR_FRESH_TARGET_OPEN",
        "revision": REVISION,
        "behavior_implementation_binding": BEHAVIOR_IMPLEMENTATION_BINDING,
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_release_binding": FEATURE_RELEASE_BINDING,
        "scope_shard_manifest_count": 4,
        "scope_shard_inventory": [
            "preflop_behavior:shard-0",
            "preflop_behavior:shard-1",
            "postflop_behavior:shard-0",
            "postflop_behavior:shard-1",
        ],
        "finalist_count_per_shard": 12,
        "finalist_iteration_cap": 12_000,
        "finalist_patience": 250,
        "finalist_minimum_improvement": 1e-5,
        "targeted_refinement_candidate_cap": 16,
        "targeted_refinement_must_freeze_before_calibration_and_fresh": True,
        "fresh_cutoff_utc_inclusive": FRESH_CUTOFF_UTC,
        "preregistered_claim_count": 2 * len(REQUIRED_BASELINES),
        "required_baselines": list(REQUIRED_BASELINES),
        "requested_workers_default": "auto",
        "worker_values": "auto_or_positive_integer",
        "candidate_internal_threads": 1,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "accepted_runtime_types": ["cpu", "gpu", "tpu"],
        "preferred_notebook_host": PREFERRED_NOTEBOOK_HOST,
        "model_execution_device": "host_cpu_process",
        "tree_training_uses_tpu_acceleration": False,
        "new_notebook01_materialization_after_freeze_required": True,
        "fresh_context_phase": {
            "notebook_token": "26",
            "runs_before_target_open": True,
            "canonical_source_root": (
                "inputs/behavior_v2_fresh_native_context/<fresh_run_identity>"
            ),
            "required_label_free_source_files": [
                "preflop-native-source.parquet",
                "action-native-rolling-source.parquet",
            ],
            "sources_are_derived_by_authenticated_causal_producer": True,
            "manual_context_output_path_required": False,
        },
        "selection_or_tuning_after_fresh_open_allowed": False,
        "fresh_targets_opened": False,
        "qualification_authority": False,
        "assistant_installation_allowed": False,
    })
    members.append((
        "behavior_challenger_v2_science_precommit.json",
        canonical_json_bytes(precommit) + b"\n",
    ))
    manifest = seal_document({
        "schema_version": "stage6_behavior_challenger_science_package_manifest_v3",
        "revision": REVISION,
        "science_precommit_self_sha256": precommit["self_sha256"],
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
        "qualification_authority": False,
    })
    members.append(("PACKAGE_MANIFEST.json", canonical_json_bytes(manifest) + b"\n"))
    return _fixed_zip_bytes(members)


def package_identity(raw: bytes | None = None) -> dict[str, Any]:
    value = package_bytes() if raw is None else raw
    with zipfile.ZipFile(io.BytesIO(value)) as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        precommit = json.loads(archive.read(
            "behavior_challenger_v2_science_precommit.json"
        ))
    runtime_binding = seal_document({
        "schema_version": (
            "stage6_behavior_challenger_science_runtime_package_binding_v3"
        ),
        "package_name": PACKAGE_NAME,
        "package_bytes": len(value),
        "package_sha256": hashlib.sha256(value).hexdigest(),
        "package_manifest_self_sha256": manifest["self_sha256"],
        "science_precommit_self_sha256": precommit["self_sha256"],
        "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
        "feature_run_id": FEATURE_RELEASE_BINDING["feature_run_id"],
        "member_identity_root_sha256": hashlib.sha256(canonical_json_bytes(
            manifest["members"]
        )).hexdigest(),
    })
    return {
        "name": PACKAGE_NAME,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "package_manifest_self_sha256": manifest["self_sha256"],
        "science_precommit_self_sha256": manifest[
            "science_precommit_self_sha256"
        ],
        "runtime_package_binding": runtime_binding,
    }


def _bootstrap_code(package: Mapping[str, Any]) -> str:
    return (
        _broad_bootstrap_code(package)
        .replace(BROAD_PACKAGE_NAME, PACKAGE_NAME)
        .replace(
            "/content/stage6_behavior_challenger_v2_r9",
            "/content/stage6_behavior_challenger_v2_science_r14",
        )
        .replace(
            "stage6_behavior_challenger_release_ready",
            "stage6_behavior_challenger_science_release_ready",
        )
    )


def _operator_cell() -> Any:
    return _code('''REQUESTED_WORKERS = "auto"  # @param {type:"string"}
NOTEBOOK_ACCELERATOR = "TPU"  # @param ["CPU", "GPU", "TPU"]
import os
REQUESTED_WORKERS = os.environ.get(
    "STAGE6_REQUESTED_WORKERS", REQUESTED_WORKERS
).strip()
NOTEBOOK_ACCELERATOR = NOTEBOOK_ACCELERATOR.strip().upper()
print({
    "requested_workers": REQUESTED_WORKERS,
    "worker_rule": "maximum safe host logical CPUs bounded by unfinished work and RAM",
    "candidate_internal_threads": 1,
    "accepted_runtime_types": ["CPU", "GPU", "TPU"],
    "preferred_notebook_host": "v6e TPU",
    "model_execution_device": "host_cpu_process",
    "tree_training_uses_tpu_acceleration": False,
})''')


def build_notebook(spec: Mapping[str, Any], package: Mapping[str, Any]) -> Any:
    token = str(spec["token"])
    phase = str(spec["phase"])
    if phase == "finalist_aggregate":
        title = "Finalist, refit, calibration, and aggregate freeze"
        run_cell = _code('''from stage6_behavior_challenger_finalist_v2 import (
    aggregate_run_id,
    run_finalist_aggregate,
)
result = run_finalist_aggregate(
    STAGE6_DRIVE_ROOT,
    requested_workers=REQUESTED_WORKERS,
    accelerator=NOTEBOOK_ACCELERATOR,
    heartbeat_interval_seconds=60.0,
)
RUN_IDENTITY = aggregate_run_id()
PRIMARY_MANIFEST_PATH = Path(result["manifest_path"])
print(result)''')
        fresh_metadata = False
    elif phase == "fresh_context":
        title = "Target-free incumbent context freeze"
        run_cell = _code('''FRESH_MATERIALIZATION_RUN_ID = ""  # @param {type:"string"}
if not FRESH_MATERIALIZATION_RUN_ID.strip():
    raise RuntimeError(
        "Enter the new Notebook01 run identity produced after Notebook 25 freeze"
    )
fresh_root = (
    STAGE6_DRIVE_ROOT / "runtime_state" / "notebook01" / "materializations"
    / FRESH_MATERIALIZATION_RUN_ID.strip()
)
from stage6_behavior_challenger_fresh_context_v2 import (
    build_fresh_incumbent_context,
    canonical_source_root,
    produce_target_free_native_source_for_fresh_materialization,
)
from stage6_behavior_challenger_notebook_runtime_v2 import ActiveWorkHeartbeat
print({
    "canonical_label_free_source_root": str(
        canonical_source_root(STAGE6_DRIVE_ROOT, FRESH_MATERIALIZATION_RUN_ID.strip())
    ),
    "source_generation": "authenticated deterministic causal producer",
    "target_or_outcome_columns_allowed": False,
})
heartbeat = ActiveWorkHeartbeat(
    scope="behavior_fresh_context", shard=-1, interval_seconds=60.0
)
heartbeat.update(phase="building_target_free_incumbent_context", denominator=1, completed=0, work_unit="contexts")
heartbeat.start()
try:
    producer_result = produce_target_free_native_source_for_fresh_materialization(
        STAGE6_DRIVE_ROOT, fresh_root
    )
    result = build_fresh_incumbent_context(STAGE6_DRIVE_ROOT, fresh_root)
    heartbeat.update(
        phase="building_target_free_incumbent_context", denominator=1,
        completed=1, work_unit="contexts"
    )
finally:
    heartbeat.stop()
PRIMARY_MANIFEST_PATH = Path(result["manifest_path"])
RUN_IDENTITY = FRESH_MATERIALIZATION_RUN_ID.strip()
print(result)''')
        fresh_metadata = False
    else:
        title = "One-shot genuinely fresh adjudication"
        run_cell = _code('''FRESH_MATERIALIZATION_RUN_ID = ""  # @param {type:"string"}
if not FRESH_MATERIALIZATION_RUN_ID.strip():
    raise RuntimeError(
        "Enter the new Notebook01 run identity produced after Notebook 25 freeze"
    )
fresh_root = (
    STAGE6_DRIVE_ROOT / "runtime_state" / "notebook01" / "materializations"
    / FRESH_MATERIALIZATION_RUN_ID.strip()
)
from stage6_behavior_challenger_fresh_context_v2 import fresh_context_root
context_root = fresh_context_root(
    STAGE6_DRIVE_ROOT, FRESH_MATERIALIZATION_RUN_ID.strip()
)
from stage6_behavior_challenger_fresh_adjudication_v2 import run_fresh_adjudication
result = run_fresh_adjudication(
    STAGE6_DRIVE_ROOT,
    fresh_root,
    context_root,
    requested_workers=REQUESTED_WORKERS,
    accelerator=NOTEBOOK_ACCELERATOR,
    heartbeat_interval_seconds=60.0,
)
PRIMARY_MANIFEST_PATH = Path(result["manifest_path"])
RUN_IDENTITY = PRIMARY_MANIFEST_PATH.parent.name
print(result)''')
        fresh_metadata = True
    cells = [
        _markdown(f"""# Stage 6 — Behavior Challenger v2 · {title}

**Notebook {token} · additive behavior science {REVISION}**

{PREFERRED_NOTEBOOK_HOST} is preferred for model-heavy work. CPU, GPU, and TPU
hosts are accepted; model work remains one-thread host CPU process work and no
TPU tree-training acceleration is claimed. Authority remains held pending the
exact gates and a focused independent audit."""),
        _markdown("## Operator settings"),
        _operator_cell(),
        _markdown("## Mount Drive and verify the exact science package"),
        _code(_bootstrap_code(package)),
        _markdown(f"## Run or resume {title.lower()}"),
        run_cell,
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
    print({{"download_status": "manual_download_may_be_required", "detail": str(error)}})'''),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "colab": {"name": spec["name"], "provenance": []},
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3"},
            "stage6": {
                "token": token,
                "revision": REVISION,
                "phase": phase,
                "aggregate_run_id": aggregate_run_id(
                    package["runtime_package_binding"]
                ),
                "science_runtime_package_binding": package[
                    "runtime_package_binding"
                ],
                "runtime_preference": PREFERRED_NOTEBOOK_HOST,
                "accepted_runtime_types": ["cpu", "gpu", "tpu"],
                "model_execution_device": "host_cpu_process",
                "tree_training_uses_tpu_acceleration": False,
                "requested_workers_default": "auto",
                "candidate_internal_threads": 1,
                "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                "new_notebook01_materialization_after_freeze_required": (
                    phase != "finalist_aggregate"
                ),
                "fresh_target_opening_authorized_only_after_exclusive_claim": fresh_metadata,
                "target_free_context_producer_required_before_fresh": (
                    phase in {"fresh_context", "fresh_adjudication"}
                ),
                "canonical_context_source_root_derived_from_fresh_run_id": (
                    phase in {"fresh_context", "fresh_adjudication"}
                ),
                "manual_context_output_path_required": False,
                "selection_or_tuning_after_fresh_open_allowed": False,
                "qualification_authority": False,
                "assistant_installation_allowed": False,
                "input_package_sha256": package["sha256"],
            },
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = hashlib.sha256(
            f"stage6-behavior-science-r14:{token}:{index}:{cell.cell_type}:{cell.source}".encode(
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
            raise RuntimeError(f"conflicting behavior science release exists: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())


def write_release(output_root: Path) -> dict[str, Any]:
    """Explicit operator action only; imports/tests do not materialize releases."""
    raw_package = package_bytes()
    package = package_identity(raw_package)
    package_path = output_root / "inputs" / PACKAGE_NAME
    _atomic_write(package_path, raw_package)
    rows = []
    for spec in NOTEBOOK_SPECS:
        raw = nbformat.writes(build_notebook(spec, package)).encode("utf-8")
        path = output_root / "notebooks" / str(spec["name"])
        _atomic_write(path, raw)
        notebook_identity = {
            "token": spec["token"],
            "name": path.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "phase": spec["phase"],
        }
        static = seal_document({
            "schema_version": "stage6_behavior_challenger_science_notebook_static_manifest_v3",
            "revision": REVISION,
            "notebook": notebook_identity,
            "shared_package": package,
            "aggregate_run_id": aggregate_run_id(
                package["runtime_package_binding"]
            ),
            "science_runtime_package_binding": package[
                "runtime_package_binding"
            ],
            "behavior_implementation_self_sha256": BEHAVIOR_IMPLEMENTATION_DIGEST,
            "fresh_targets_opened_before_node": False,
            "qualification_authority": False,
        })
        static_name = f"stage6_notebook{spec['token']}_behavior_v2_science_static_manifest_r14.json"
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
        rows.append(notebook_identity)
    return {"package": package, "notebooks": rows}


__all__ = [
    "NOTEBOOK_SPECS",
    "FROZEN_FEATURE_RELEASE_MEMBER_PATHS",
    "PACKAGE_MEMBER_PATHS",
    "PACKAGE_NAME",
    "REVISION",
    "build_notebook",
    "build_notebooks",
    "package_bytes",
    "package_identity",
    "write_release",
]
