from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import sha256_json, validate_self_hash, with_self_hash


LEGACY_COMPLETED_RESULT_CONTRACT_SCHEMA = (
    "stage21_bound_legacy_completed_result_contract_v1"
)
LEGACY_COMPLETED_RESULT_TOKENS = (
    "A01",
    "A02",
    "A03",
    "A05",
    "A08",
    "A11",
    "A14",
    "A15",
    "A16",
)


def normalized_root_science(root: Mapping[str, Any]) -> dict[str, Any]:
    """Return the revision-independent scientific root payload."""

    return {
        key: value
        for key, value in root.items()
        if key not in {"revision", "root_semantic_sha256"}
    }


def load_legacy_completed_result_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_self_hash(document)
    if document.get("schema_version") != LEGACY_COMPLETED_RESULT_CONTRACT_SCHEMA:
        raise ValueError("unsupported legacy completed-result contract")
    if document.get("stage") != "river":
        raise ValueError("legacy completed-result contract is not River scoped")
    if (
        document.get("qualification_threshold_change_allowed") is not False
        or document.get("checkpoint_recovery_allowed") is not False
        or document.get("runtime_state_recovery_allowed") is not False
        or document.get("legacy_runtime_tree_is_not_read") is not True
        or document.get("legacy_completed_result_archives_are_read_only") is not True
    ):
        raise ValueError("legacy completed-result contract enables forbidden recovery")
    return document


def validate_legacy_completed_result_contract_target(
    contract: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> None:
    """Bind the exact completed R9 result subset to the current River A17 DAG."""

    validate_self_hash(contract)
    if contract.get("schema_version") != LEGACY_COMPLETED_RESULT_CONTRACT_SCHEMA:
        raise ValueError("legacy completed-result contract schema mismatch")
    if execution.get("stage") != "river" or contract.get("stage") != "river":
        raise ValueError("legacy completed results may only be used by River")
    target = contract.get("target_package")
    source = contract.get("source_package")
    if not isinstance(target, Mapping) or not isinstance(source, Mapping):
        raise ValueError("legacy completed-result package bindings are absent")
    if (
        target.get("revision") != execution.get("revision")
        or target.get("execution_manifest_self_sha256")
        != execution.get("self_sha256")
        or target.get("root_registry_semantic_sha256")
        != registry.get("semantic_sha256")
        or target.get("package_source_semantic_sha256")
        != execution.get("source_semantic_hashes", {}).get(
            "package_source_semantic_sha256"
        )
        or target.get("qualification_contract_self_sha256")
        != qualification.get("self_sha256")
        or contract.get("shared_scientific_contracts", {}).get(
            "qualification_contract_self_sha256"
        )
        != qualification.get("self_sha256")
        or source.get("qualification_contract_self_sha256")
        != qualification.get("self_sha256")
    ):
        raise ValueError("legacy completed-result target package mismatch")

    producer_rows = [
        row
        for row in execution.get("notebooks", ())
        if row.get("kind") != "strict_aggregate"
    ]
    current_tokens = {str(row["token"]) for row in producer_rows}
    if set(map(str, contract.get("producer_tokens", ()))) != current_tokens:
        raise ValueError("legacy completed-result producer topology mismatch")
    current_lanes = {
        str(lane["lane_id"]): {
            "lane": lane,
            "token": str(row["token"]),
            "root_kind": row.get("root_kind"),
        }
        for row in producer_rows
        for lane in row.get("lanes", ())
    }
    lane_bindings = contract.get("lane_bindings")
    if (
        not isinstance(lane_bindings, Mapping)
        or set(lane_bindings) != set(current_lanes)
        or int(contract.get("lane_count", -1)) != len(current_lanes)
    ):
        raise ValueError("legacy completed-result lane coverage mismatch")
    for lane_id, current in current_lanes.items():
        lane = current["lane"]
        spec = lane_bindings[lane_id]
        if not isinstance(spec, Mapping):
            raise ValueError("legacy completed-result lane binding is malformed")
        for field in ("lane_id", "root_id", "algorithm", "seed", "role"):
            if spec.get(field) != lane.get(field):
                raise ValueError(
                    f"legacy completed-result lane differs for {lane_id}: {field}"
                )
        if spec.get("token") != current["token"]:
            raise ValueError(f"legacy completed-result lane differs for {lane_id}: token")
        if spec.get("root_kind") != current["root_kind"]:
            raise ValueError(
                f"legacy completed-result lane differs for {lane_id}: root_kind"
            )
        for digest_field in (
            "legacy_execution_identity_sha256",
            "legacy_root_semantic_sha256",
            "legacy_task_semantic_sha256",
        ):
            digest = spec.get(digest_field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("legacy completed-result lane digest is invalid")

    current_roots = {
        str(root["root_id"]): root for root in registry.get("roots", ())
    }
    root_bindings = contract.get("root_bindings")
    if (
        not isinstance(root_bindings, Mapping)
        or set(root_bindings) != set(current_roots)
        or int(contract.get("root_count", -1)) != len(current_roots)
    ):
        raise ValueError("legacy completed-result root coverage mismatch")
    for root_id, root in current_roots.items():
        spec = root_bindings[root_id]
        if (
            not isinstance(spec, Mapping)
            or spec.get("root_id") != root_id
            or spec.get("root_kind") != root.get("root_kind")
            or spec.get("target_root_semantic_sha256")
            != root.get("root_semantic_sha256")
            or spec.get("normalized_root_science_sha256")
            != sha256_json(normalized_root_science(root))
        ):
            raise ValueError(f"legacy completed-result root differs: {root_id}")

    expected_archive_names = {
        f"POKERNOW_ASSISTANT_STAGE1_{token}_RESULTS.zip"
        for token in LEGACY_COMPLETED_RESULT_TOKENS
    }
    archives = contract.get("known_complete_returned_archives")
    allowed_tokens = set(map(str, contract.get("allowed_legacy_result_tokens", ())))
    if (
        not isinstance(archives, Mapping)
        or set(archives) != expected_archive_names
        or allowed_tokens != set(LEGACY_COMPLETED_RESULT_TOKENS)
    ):
        raise ValueError("legacy completed-result archive set mismatch")
    for filename, binding in archives.items():
        if (
            not isinstance(binding, Mapping)
            or binding.get("filename") != filename
            or int(binding.get("size", 0)) <= 0
            or not isinstance(binding.get("sha256"), str)
            or len(str(binding["sha256"])) != 64
        ):
            raise ValueError(f"legacy completed-result archive identity is invalid: {filename}")


def _validate_nested_self_hash(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"legacy lane lacks {label}")
    validate_self_hash(value)
    return value


def validate_legacy_result_lane_document(
    document: Mapping[str, Any],
    *,
    current_lane: Mapping[str, Any],
    contract: Mapping[str, Any],
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one untouched R9 result lane for mixed River A17 aggregation."""

    lane_id = str(current_lane["lane_id"])
    spec = contract.get("lane_bindings", {}).get(lane_id)
    if not isinstance(spec, Mapping):
        raise ValueError(f"legacy result lane is absent from contract: {lane_id}")
    for field in ("lane_id", "root_id", "algorithm", "seed"):
        if document.get(field) != spec.get(field):
            raise ValueError(f"legacy result lane binding mismatch: {field}")
        if current_lane.get(field) != spec.get(field):
            raise ValueError(f"current lane binding mismatch: {field}")
    if (
        document.get("status") != "COMPLETE"
        or document.get("task_semantic_sha256")
        != spec["legacy_task_semantic_sha256"]
        or document.get("root_semantic_sha256")
        != spec["legacy_root_semantic_sha256"]
        or sha256_json(document.get("execution_identity"))
        != spec["legacy_execution_identity_sha256"]
    ):
        raise ValueError("legacy result lane identity is not authorized")

    policy = document.get("policy")
    if policy_sha256 is None and isinstance(policy, Mapping):
        policy_sha256 = sha256_json(policy)
    root_kind = str(spec["root_kind"])
    if root_kind == "forced":
        if document.get("role") != "forced_runout_evaluation":
            raise ValueError("legacy forced lane role mismatch")
    else:
        solver_result = _validate_nested_self_hash(
            document.get("solver_result"), "solver result"
        )
        evaluation = _validate_nested_self_hash(document.get("evaluation"), "evaluation")
        adjudication = _validate_nested_self_hash(
            document.get("adjudication"), "adjudication"
        )
        if (
            solver_result.get("root_id") != spec["root_id"]
            or solver_result.get("root_semantic_sha256")
            != spec["legacy_root_semantic_sha256"]
            or solver_result.get("task_semantic_sha256")
            != spec["legacy_task_semantic_sha256"]
            or solver_result.get("algorithm") != spec.get("algorithm")
            or solver_result.get("seed") != spec["seed"]
            or policy_sha256 is None
            or solver_result.get("policy_sha256") != policy_sha256
            or document.get("qualified") is not adjudication.get("qualified")
        ):
            raise ValueError("legacy result solver lane nested binding mismatch")
        if root_kind == "approximate" and solver_result.get(
            "solver_contract_sha256"
        ) != contract["source_package"]["solver_contract_sha256"]:
            raise ValueError("legacy result approximate solver contract mismatch")
        if evaluation.get("evaluation_contract_sha256") != contract[
            "shared_scientific_contracts"
        ]["evaluation_contract_sha256"]:
            raise ValueError("legacy result evaluation contract mismatch")
        for panel in evaluation.get("panels", ()):
            _validate_nested_self_hash(panel, "evaluation panel")

    if (
        document.get("artifact_hash_valid") is not True
        or document.get("independent_confirmation_panel") is not True
    ):
        raise ValueError("legacy result lane lacks strict validation flags")
    return with_self_hash(
        {
            "schema_version": "stage21_validated_legacy_result_lane_receipt_v1",
            "legacy_completed_result_contract_self_sha256": contract["self_sha256"],
            "lane_id": lane_id,
            "root_id": spec["root_id"],
            "algorithm": spec.get("algorithm"),
            "seed": spec["seed"],
            "legacy_task_semantic_sha256": spec["legacy_task_semantic_sha256"],
            "legacy_root_semantic_sha256": spec["legacy_root_semantic_sha256"],
            "policy_sha256": policy_sha256,
            "qualified": document.get("qualified"),
            "legacy_document_self_sha256": document["self_sha256"],
            "legacy_document_preserved_without_relabelling": True,
        }
    )


def validate_legacy_result_execution_identity_receipt(
    identity: Mapping[str, Any],
    *,
    token: str,
    contract: Mapping[str, Any],
    source_archive_identity: Mapping[str, Any],
    result_filename: str,
) -> dict[str, Any]:
    validate_self_hash(identity)
    source = contract["source_package"]
    if (
        identity.get("stage") != "river"
        or identity.get("token") != token
        or identity.get("execution_manifest_self_sha256")
        != source["execution_manifest_self_sha256"]
        or identity.get("root_registry_semantic_sha256")
        != source["root_registry_semantic_sha256"]
        or identity.get("package_source_semantic_sha256")
        != source["package_source_semantic_sha256"]
        or identity.get("coverage_contract_self_sha256")
        != source["coverage_contract_self_sha256"]
        or identity.get("dependency_contract_self_sha256")
        != source["dependency_contract_self_sha256"]
        or identity.get("continuation_content_identity") is not None
    ):
        raise ValueError("legacy result producer execution identity mismatch")
    expected_archive = contract.get("known_complete_returned_archives", {}).get(
        result_filename
    )
    if not isinstance(expected_archive, Mapping):
        raise ValueError("legacy result producer archive is not externally contracted")
    if (
        expected_archive.get("filename") != result_filename
        or int(expected_archive.get("size", -1))
        != int(source_archive_identity.get("size", -2))
        or expected_archive.get("sha256")
        != source_archive_identity.get("sha256")
        or token not in LEGACY_COMPLETED_RESULT_TOKENS
    ):
        raise ValueError("legacy result producer archive raw identity mismatch")
    return with_self_hash(
        {
            "schema_version": "stage21_validated_legacy_result_producer_receipt_v1",
            "token": token,
            "legacy_execution_identity_receipt_self_sha256": identity["self_sha256"],
            "legacy_completed_result_contract_self_sha256": contract["self_sha256"],
            "legacy_result_filename": result_filename,
            "legacy_result_size": int(source_archive_identity["size"]),
            "legacy_result_sha256": source_archive_identity["sha256"],
            "legacy_archive_externally_contracted": True,
            "legacy_receipt_preserved_without_relabelling": True,
        }
    )
