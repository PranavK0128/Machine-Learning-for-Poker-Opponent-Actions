"""Bind Notebook 13-19 to the exact R5 feature run and downstream science.

The historical adapter incorrectly replaced the downstream science identity
with only the feature-freeze implementation digest. This additive binding
keeps the R5 feature release as an authenticated parent and separately hashes
every source/contract that can affect the Notebook 13-19 results.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import stage6_expanded_feature_freeze_v2 as feature_freeze_v2
from stage6_expanded_layers import canonical_json_bytes, seal_document


R5_FEATURE_RUN_ID = (
    "6761da35e9e7d0024038429ad49990cfebffbc6800902a2591398e45f93deaf7"
)
if feature_freeze_v2.FEATURE_RUN_ID_V2 != R5_FEATURE_RUN_ID:
    raise RuntimeError("downstream adapter feature-run parent changed")

R5_FEATURE_RELEASE_BINDING = seal_document({
    "schema_version": "stage6_expanded_feature_release_binding_r5",
    "feature_run_id": R5_FEATURE_RUN_ID,
    "feature_implementation_self_sha256": (
        "73387f289e467eaecbc1d12d7abb38969f729362f214245d1e6c73908516a476"
    ),
    "equivalence_gate_spec_self_sha256": (
        "d4387d41961dff7b989167fa61b841179b82ba1a8ff96a8d628d0d94a362ade7"
    ),
    "input_package": {
        "name": "POKERNOW_ASSISTANT_STAGE6_EXPANDED_FEATURE_FREEZE_INPUTS_R5.zip",
        "bytes": 130674,
        "sha256": "3c47557099eedee0e958e2b95edc2b131e91128db6a7e2820e2a3f6ff1ab240f",
        "package_manifest_self_sha256": (
            "b6ed941173fe978bb300613c712ae8f896c8965f5f292717d7983a4102e77c56"
        ),
        "precommit_self_sha256": (
            "254744762c121ebd3585fd793122d05d1ce6385a47c170253d35a7c4810c4658"
        ),
    },
    "notebook": {
        "name": "12_Stage6_Expanded_Layer_Causal_Feature_Freeze_R5.ipynb",
        "bytes": 8625,
        "sha256": "5a11497abcbabefb077c101b73bafb329140a8fadb0009ea533d16a8f8b88eec",
    },
    "static_manifest": {
        "name": "stage6_notebook12_static_manifest_r5.json",
        "bytes": 1698,
        "sha256": "ba433360edb31335a2b54c12b7f3eeb8bcba7ef724861d4d0aa07a6d8bb2ab5c",
        "self_sha256": "bb55a5c6b56bc726e528a6e491d93abcc54f0cecb3cab8c4a6546ca24a6b0da5",
    },
})

DOWNSTREAM_OUTPUT_AFFECTING_MEMBERS_V2 = (
    "src/stage6_common.py",
    "src/stage6_metrics.py",
    "src/stage6_notebooks02_07_runtime.py",
    "src/stage6_behavior_tournament.py",
    "src/stage6_expanded_layers.py",
    "src/stage6_expanded_tournament.py",
    "src/stage6_expanded_finalist.py",
    "src/stage6_expanded_adjudication.py",
    "src/stage6_expanded_notebook_controller.py",
    "src/stage6_expanded_downstream_rebind_v2.py",
    "contracts/stage6_data_contract_v1.json",
    "contracts/stage6_split_contract_v1.json",
    "contracts/stage6_scientific_gate_v1.json",
    "contracts/stage6_worker_policy_v3.json",
    "contracts/stage6_expanded_predictive_layers_contract_v1.json",
)

DOWNSTREAM_CONSUMER_MODULES_V2 = (
    "stage6_expanded_tournament",
    "stage6_expanded_finalist",
    "stage6_expanded_adjudication",
    "stage6_expanded_notebook_controller",
)


def downstream_science_implementation_binding_v2(
    stage_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(stage_root) if stage_root is not None else Path(__file__).resolve().parents[1]
    members = []
    for relative_path in DOWNSTREAM_OUTPUT_AFFECTING_MEMBERS_V2:
        raw = (root / relative_path).read_bytes()
        members.append({
            "relative_path": relative_path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return seal_document({
        "schema_version": "stage6_expanded_downstream_science_binding_v2",
        "feature_release_binding": R5_FEATURE_RELEASE_BINDING,
        "feature_release_binding_self_sha256": R5_FEATURE_RELEASE_BINDING["self_sha256"],
        "member_order": list(DOWNSTREAM_OUTPUT_AFFECTING_MEMBERS_V2),
        "members": members,
        "logical_data_lane_count": 48,
        "candidate_internal_threads": 1,
        "model_grid_or_science_gate_changed": False,
        "fresh_targets_opened": False,
    })


DOWNSTREAM_SCIENCE_IMPLEMENTATION_BINDING_V2 = (
    downstream_science_implementation_binding_v2()
)
DOWNSTREAM_SCIENCE_IMPLEMENTATION_DIGEST_V2 = (
    DOWNSTREAM_SCIENCE_IMPLEMENTATION_BINDING_V2["self_sha256"]
)


def activate_downstream_rebind_v2() -> dict[str, Any]:
    existing = sys.modules.get("stage6_expanded_feature_freeze")
    if existing is not None and existing is not feature_freeze_v2:
        raise RuntimeError(
            "R5 feature rebind must activate before the historical N12 module is imported"
        )
    sys.modules["stage6_expanded_feature_freeze"] = feature_freeze_v2

    # Consumers import the digest by value, so rebind the source module before
    # importing any downstream consumer.
    layers = importlib.import_module("stage6_expanded_layers")
    setattr(
        layers,
        "SCIENCE_IMPLEMENTATION_BINDING",
        DOWNSTREAM_SCIENCE_IMPLEMENTATION_BINDING_V2,
    )
    setattr(
        layers,
        "SCIENCE_IMPLEMENTATION_DIGEST",
        DOWNSTREAM_SCIENCE_IMPLEMENTATION_DIGEST_V2,
    )

    rebound: list[str] = []
    for name in DOWNSTREAM_CONSUMER_MODULES_V2:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            # Compact feature-only packages may omit later consumers.
            continue
        if hasattr(module, "FEATURE_RUN_ID"):
            setattr(module, "FEATURE_RUN_ID", R5_FEATURE_RUN_ID)
        if hasattr(module, "SCIENCE_IMPLEMENTATION_BINDING"):
            setattr(
                module,
                "SCIENCE_IMPLEMENTATION_BINDING",
                DOWNSTREAM_SCIENCE_IMPLEMENTATION_BINDING_V2,
            )
        if hasattr(module, "SCIENCE_IMPLEMENTATION_DIGEST"):
            setattr(
                module,
                "SCIENCE_IMPLEMENTATION_DIGEST",
                DOWNSTREAM_SCIENCE_IMPLEMENTATION_DIGEST_V2,
            )
        rebound.append(name)
    return {
        "schema_version": "stage6_expanded_downstream_rebind_receipt_v2",
        "feature_run_id": R5_FEATURE_RUN_ID,
        "feature_release_binding_self_sha256": R5_FEATURE_RELEASE_BINDING["self_sha256"],
        "downstream_science_implementation_self_sha256": (
            DOWNSTREAM_SCIENCE_IMPLEMENTATION_DIGEST_V2
        ),
        "rebound_modules": rebound,
        "gate_logic_changed": False,
        "model_grid_changed": False,
        "historical_r1_source_bytes_mutated": False,
        "historical_r1_implementation_hash_emitted": False,
        "fresh_targets_opened": False,
    }
