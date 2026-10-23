from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import (
    atomic_install_bytes,
    atomic_write,
    canonical_json_bytes,
    deterministic_zip_bytes,
    member_manifest,
    read_self_hashed_json_streamed_mapping,
    sha256_bytes,
    stage_file_once,
    stream_file_identity,
    validate_self_hash,
    with_self_hash,
)
from .policy_store import stage_self_hashed_policy_document


_INSTALLED_RESULT_RECEIPT_TOKEN = object()
_VALIDATED_PRODUCER_SET_TOKEN = object()
_CANONICAL_RESULT_RECEIPTS = {
    "receipts/dependency_read_receipt.json",
    "receipts/execution_identity_receipt.json",
    "receipts/lane_queue_receipt.json",
}
_CANONICAL_RESULT_RECEIPT_SCHEMAS = {
    "receipts/dependency_read_receipt.json": "stage21_dependency_read_receipt_v2",
    "receipts/execution_identity_receipt.json": "stage21_execution_identity_receipt_v3",
    "receipts/lane_queue_receipt.json": "stage21_lane_queue_receipt_v5",
}


def _validate_result_receipt_document(
    name: str,
    raw: bytes,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
) -> None:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"result archive receipt is not canonical JSON: {name}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"result archive receipt is not an object: {name}")
    validate_self_hash(document)
    if document.get("schema_version") != _CANONICAL_RESULT_RECEIPT_SCHEMAS[name]:
        raise ValueError(f"result archive receipt schema mismatch: {name}")
    if name == "receipts/dependency_read_receipt.json":
        if document.get("stage") != expected_stage:
            raise ValueError("dependency receipt stage mismatch")
    elif name == "receipts/execution_identity_receipt.json":
        if (
            document.get("stage") != expected_stage
            or document.get("token") != expected_token
        ):
            raise ValueError("execution identity receipt stage/token mismatch")
    else:
        handles = document.get("handles")
        if (
            document.get("logical_lane_count") != 48
            or not isinstance(handles, list)
            or {str(row.get("lane_id")) for row in handles if isinstance(row, Mapping)}
            != set(map(str, expected_lane_ids))
            or len(handles) != 48
        ):
            raise ValueError("lane queue receipt membership mismatch")


def _canonical_result_member_rows(
    manifest: Mapping[str, Any],
    names: Sequence[str],
    *,
    expected_lane_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    if (
        manifest.get("schema_version") != "stage21_result_archive_manifest_v2"
        or manifest.get("lane_count") != 48
    ):
        raise ValueError("result archive manifest schema/lane count is not canonical")
    members = manifest.get("members")
    if not isinstance(members, list) or not all(
        isinstance(row, Mapping) for row in members
    ):
        raise ValueError("result archive members must be a list of mappings")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in members:
        if set(row) != {"path", "size", "sha256"}:
            raise ValueError("result archive manifest row schema is not canonical")
        path = row.get("path")
        if not isinstance(path, str) or path in expected:
            raise ValueError("duplicate or invalid result archive manifest path")
        if (
            not isinstance(row.get("size"), int)
            or int(row["size"]) < 0
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
        ):
            raise ValueError("invalid result archive manifest member identity")
        expected[path] = row
    lane_paths = {
        path
        for path in expected
        if path.startswith("lanes/") and path.endswith(".json")
    }
    normalized_lane_ids = [str(lane_id) for lane_id in expected_lane_ids]
    if (
        len(normalized_lane_ids) != 48
        or len(normalized_lane_ids) != len(set(normalized_lane_ids))
        or any(
            not lane_id
            or "/" in lane_id
            or "\\" in lane_id
            for lane_id in normalized_lane_ids
        )
    ):
        raise ValueError("expected result lane identity set is not canonical")
    expected_lane_paths = {
        f"lanes/{lane_id}.json" for lane_id in normalized_lane_ids
    }
    if len(lane_paths) != 48 or any(
        len(Path(path).parts) != 2
        or not Path(path).name.removesuffix(".json")
        for path in lane_paths
    ):
        raise ValueError("result archive does not contain exactly 48 canonical lanes")
    if lane_paths != expected_lane_paths:
        raise ValueError("result archive lane membership differs from execution manifest")
    if set(expected) - lane_paths != _CANONICAL_RESULT_RECEIPTS:
        raise ValueError("result archive receipt membership is not canonical")
    if len(expected) != 51:
        raise ValueError("result archive manifest row count is not canonical")
    actual = set(names) - {"result_manifest.json"}
    if actual != set(expected):
        raise ValueError("result archive member manifest mismatch")
    return expected


def _read_member_once(
    archive: zipfile.ZipFile,
    name: str,
    *,
    capture: bool,
) -> tuple[int, str, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured = bytearray() if capture else None
    with archive.open(name, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            if captured is not None:
                captured.extend(block)
    return size, digest.hexdigest(), None if captured is None else bytes(captured)


def _extract_member_once(
    archive: zipfile.ZipFile,
    name: str,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary: Path | None = None
    try:
        with archive.open(name, "r") as source, tempfile.NamedTemporaryFile(
            dir=target.parent, delete=False
        ) as destination:
            temporary = Path(destination.name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
                digest.update(block)
                size += len(block)
            destination.flush()
            os.fsync(destination.fileno())
        if size != int(expected_size) or digest.hexdigest() != expected_sha256:
            raise ValueError(f"result archive member identity mismatch: {name}")
        temporary.replace(target)
        temporary = None
        if target.stat().st_size != size:
            raise IOError(f"staged lane member size changed: {name}")
        return {"path": str(target), "size": size, "sha256": digest.hexdigest()}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, init=False)
class InstalledResultValidationReceipt:
    """Trusted in-process wrapper for one completely validated installed ZIP."""

    path: Path
    document: Mapping[str, Any]

    def __init__(self, token: object, path: Path, document: Mapping[str, Any]) -> None:
        if token is not _INSTALLED_RESULT_RECEIPT_TOKEN:
            raise TypeError("installed-result receipts are factory-only")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "document", document)

    @classmethod
    def bind(
        cls,
        path: Path,
        document: Mapping[str, Any],
        *,
        expected_stage: str,
        expected_token: str,
    ) -> "InstalledResultValidationReceipt":
        receipt = dict(document)
        validate_self_hash(receipt)
        resolved_path = path.resolve()
        receipt_path = Path(str(receipt.get("path", ""))).resolve()
        if (
            receipt.get("schema_version")
            != "stage21_installed_result_validation_receipt_v1"
            or receipt.get("stage") != expected_stage
            or receipt.get("token") != expected_token
            or receipt.get("full_member_validation_pass_count") != 1
            or receipt_path != resolved_path
        ):
            raise ValueError("installed-result validation receipt binding mismatch")
        bound = cls(_INSTALLED_RESULT_RECEIPT_TOKEN, resolved_path, receipt)
        bound.lightweight_stat_check()
        return bound

    def lightweight_stat_check(self) -> dict[str, Any]:
        stat = self.path.stat()
        if (
            stat.st_size != int(self.document["size"])
            or stat.st_mtime_ns != int(self.document["installed_stat_mtime_ns"])
        ):
            raise IOError("installed result changed after complete validation")
        return {
            "path": str(self.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "bound_receipt_self_sha256": self.document["self_sha256"],
            "lightweight_existence_stat_check_only": True,
        }


@dataclass(frozen=True, init=False)
class ValidatedProducerArchiveSet:
    """Factory-bound coordinator staging consumed by strict authority."""

    stage: str
    aggregate_token: str
    document: Mapping[str, Any]
    producers: tuple[Mapping[str, Any], ...]

    def __init__(
        self,
        token: object,
        stage: str,
        aggregate_token: str,
        document: Mapping[str, Any],
        producers: tuple[Mapping[str, Any], ...],
    ) -> None:
        if token is not _VALIDATED_PRODUCER_SET_TOKEN:
            raise TypeError("validated producer sets are factory-only")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "aggregate_token", aggregate_token)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "producers", producers)

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        aggregate_token: str,
        staged_archives: Sequence[Mapping[str, Any]],
    ) -> "ValidatedProducerArchiveSet":
        producers = tuple(staged_archives)
        if not producers:
            raise ValueError("validated producer set cannot be empty")
        tokens = [str(row["receipt"]["token"]) for row in producers]
        if len(tokens) != len(set(tokens)):
            raise ValueError("validated producer set contains duplicate tokens")
        for row in producers:
            receipt = row.get("receipt")
            if not isinstance(receipt, Mapping):
                raise TypeError("validated producer set requires typed staging receipts")
            validate_self_hash(receipt)
            if (
                receipt.get("stage") != stage
                or receipt.get("staged_members_independently_read_back") is not True
            ):
                raise ValueError("validated producer staging receipt mismatch")
        document = with_self_hash(
            {
                "schema_version": "stage21_validated_producer_archive_set_v1",
                "stage": stage,
                "aggregate_token": aggregate_token,
                "producer_count": len(producers),
                "producer_receipt_self_sha256s": sorted(
                    str(row["receipt"]["self_sha256"]) for row in producers
                ),
                "source_archive_sha256s_by_token": {
                    str(row["receipt"]["token"]): str(
                        row["receipt"]["source_archive_sha256"]
                    )
                    for row in sorted(
                        producers, key=lambda item: str(item["receipt"]["token"])
                    )
                },
                "factory_validated": True,
                "authority_consumes_factory_validated_local_compressed_archives": True,
                "authority_materializes_one_lane_policy_store_at_a_time": True,
                "authority_reopens_mounted_source_members": False,
                "producer_zip_reopen_allowed": False,
            }
        )
        return cls(
            _VALIDATED_PRODUCER_SET_TOKEN,
            stage,
            aggregate_token,
            document,
            producers,
        )


def local_disk_preflight(
    local_root: Path,
    *,
    components: Mapping[str, int],
    safety_margin_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Fail before work when ephemeral scratch cannot hold bounded spools."""

    local_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(local_root)
    normalized = {str(key): max(0, int(value)) for key, value in components.items()}
    subtotal = sum(normalized.values())
    required = subtotal + max(int(safety_margin_bytes), subtotal // 4)
    if required > int(disk.free * 0.90):
        raise OSError(
            f"local scratch preflight failed: need {required} bytes; free={disk.free}"
        )
    return with_self_hash(
        {
            "schema_version": "stage21_local_scratch_disk_preflight_v1",
            "local_root": str(local_root.resolve()),
            "components": normalized,
            "component_subtotal_bytes": subtotal,
            "safety_margin_bytes": required - subtotal,
            "required_bytes": required,
            "disk_free_bytes": disk.free,
            "maximum_usable_fraction": 0.90,
            "status": "PASS",
        }
    )


def build_result_archive(
    *,
    token: str,
    stage: str,
    lane_handles: Sequence[Mapping[str, Any]],
    receipt_documents: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Reference-only in-memory builder used by bounded parity tests."""
    members: dict[str, bytes] = {}
    for handle in sorted(lane_handles, key=lambda row: row["lane_id"]):
        path = Path(str(handle["path"]))
        raw = path.read_bytes()
        if sha256_bytes(raw) != handle["sha256"]:
            raise ValueError(f"lane handle hash mismatch: {handle['lane_id']}")
        document = json.loads(raw)
        validate_self_hash(document)
        members[f"lanes/{handle['lane_id']}.json"] = raw
    for name, document in sorted(receipt_documents.items()):
        members[f"receipts/{name}.json"] = canonical_json_bytes(document)
    manifest = with_self_hash(
        {
            "schema_version": "stage21_result_archive_manifest_v2",
            "stage": stage,
            "token": token,
            "lane_count": len(lane_handles),
            "members": member_manifest(members),
            "science_claim": "UNADJUDICATED_MEMBER_RESULTS",
        }
    )
    members["result_manifest.json"] = canonical_json_bytes(manifest)
    return deterministic_zip_bytes(members)


def _read_validated_result_archive(
    archive: zipfile.ZipFile,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
    capture_members: Sequence[str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    requested = set(capture_members)
    names = archive.namelist()
    if len(names) != len(set(names)) or "result_manifest.json" not in names:
        raise ValueError("invalid result archive membership")
    if any(name.startswith("/") or "\\" in name or ".." in Path(name).parts for name in names):
        raise ValueError("unsafe result archive member")
    # ``ZipFile.read`` performs and enforces the stored CRC check. Every
    # member is read below for its manifest SHA-256, so a separate CRC pass
    # would decompress the complete archive twice.
    manifest = json.loads(archive.read("result_manifest.json"))
    validate_self_hash(manifest)
    if manifest.get("stage") != expected_stage or manifest.get("token") != expected_token:
        raise ValueError("result archive stage/token mismatch")
    expected = _canonical_result_member_rows(
        manifest, names, expected_lane_ids=expected_lane_ids
    )
    actual_names = set(names) - {"result_manifest.json"}
    if not requested <= actual_names:
        raise ValueError("requested capture member is absent from result archive")
    captured: dict[str, bytes] = {}
    for name in sorted(actual_names):
        row = expected[name]
        validate_receipt = name in _CANONICAL_RESULT_RECEIPTS
        size, sha256, member = _read_member_once(
            archive, name, capture=name in requested or validate_receipt
        )
        if size != int(row["size"]) or sha256 != row["sha256"]:
            raise ValueError(f"result archive member identity mismatch: {name}")
        if validate_receipt:
            assert member is not None
            _validate_result_receipt_document(
                name,
                member,
                expected_stage=expected_stage,
                expected_token=expected_token,
                expected_lane_ids=expected_lane_ids,
            )
        if name in requested and member is not None:
            captured[name] = member
    return manifest, captured


def read_validated_result_archive(
    raw: bytes,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
    capture_members: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate every member once and retain only explicitly requested bytes."""

    with zipfile.ZipFile(BytesIO(raw), "r") as archive:
        return _read_validated_result_archive(
            archive,
            expected_stage=expected_stage,
            expected_token=expected_token,
            expected_lane_ids=expected_lane_ids,
            capture_members=capture_members,
        )


def read_validated_result_archive_path(
    path: Path,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
    capture_members: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate an archive without materializing its compressed bytes in RAM."""

    with zipfile.ZipFile(path, "r") as archive:
        return _read_validated_result_archive(
            archive,
            expected_stage=expected_stage,
            expected_token=expected_token,
            expected_lane_ids=expected_lane_ids,
            capture_members=capture_members,
        )


def validate_result_archive(
    raw: bytes,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
) -> dict[str, Any]:
    manifest, _ = read_validated_result_archive(
        raw,
        expected_stage=expected_stage,
        expected_token=expected_token,
        expected_lane_ids=expected_lane_ids,
    )
    return manifest


def validate_result_archive_path(
    path: Path,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
) -> dict[str, Any]:
    manifest, _ = read_validated_result_archive_path(
        path,
        expected_stage=expected_stage,
        expected_token=expected_token,
        expected_lane_ids=expected_lane_ids,
    )
    return manifest


def stage_validated_result_archive(
    path: Path,
    *,
    expected_stage: str,
    expected_token: str,
    expected_lane_ids: Sequence[str],
    staging_root: Path,
    capture_members: Sequence[str] = (),
    expected_source_size: int | None = None,
    expected_source_sha256: str | None = None,
    materialize_policy_stores: bool = True,
) -> dict[str, Any]:
    """Copy a producer ZIP once, then validate only its local staged bytes.

    Aggregate coordination uses ``materialize_policy_stores=False`` so the
    compressed local archive is retained while expanded lane JSON and SQLite
    policies are created only for the root currently being evaluated.
    """

    staging_root.mkdir(parents=True, exist_ok=True)
    local_archive_path = staging_root / "source_archive.zip"
    staged_source = stage_file_once(
        path,
        local_archive_path,
        expected_size=expected_source_size,
        expected_sha256=expected_source_sha256,
    )
    source_identity = {
        "path": str(path.resolve()),
        "size": staged_source["size"],
        "sha256": staged_source["sha256"],
    }
    lane_handles: dict[str, dict[str, Any]] = {}
    requested_captures = set(capture_members)
    if not requested_captures <= _CANONICAL_RESULT_RECEIPTS:
        raise ValueError("only canonical result receipts may be captured")
    captured: dict[str, bytes] = {}
    with zipfile.ZipFile(local_archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "result_manifest.json" not in names:
            raise ValueError("invalid result archive membership")
        if any(
            name.startswith("/") or "\\" in name or ".." in Path(name).parts
            for name in names
        ):
            raise ValueError("unsafe result archive member")
        manifest = json.loads(archive.read("result_manifest.json"))
        validate_self_hash(manifest)
        if (
            manifest.get("stage") != expected_stage
            or manifest.get("token") != expected_token
        ):
            raise ValueError("result archive stage/token mismatch")
        expected = _canonical_result_member_rows(
            manifest, names, expected_lane_ids=expected_lane_ids
        )
        actual = set(names) - {"result_manifest.json"}
        if not requested_captures <= actual:
            raise ValueError("requested staged capture member is absent")
        for name in sorted(actual):
            row = expected[name]
            if name in requested_captures:
                size, sha256, raw = _read_member_once(
                    archive, name, capture=True
                )
                if (
                    size != int(row["size"])
                    or sha256 != row["sha256"]
                    or raw is None
                ):
                    raise ValueError(
                        f"result archive member identity mismatch: {name}"
                    )
                captured[name] = raw
            if not (name.startswith("lanes/") and name.endswith(".json")):
                if name in requested_captures:
                    raw = captured[name]
                    size = len(raw)
                    sha256 = sha256_bytes(raw)
                else:
                    size, sha256, raw = _read_member_once(
                        archive,
                        name,
                        capture=name in _CANONICAL_RESULT_RECEIPTS,
                    )
                if size != int(row["size"]) or sha256 != row["sha256"]:
                    raise ValueError(
                        f"result archive member identity mismatch: {name}"
                    )
                if name in _CANONICAL_RESULT_RECEIPTS:
                    assert raw is not None
                    _validate_result_receipt_document(
                        name,
                        raw,
                        expected_stage=expected_stage,
                        expected_token=expected_token,
                        expected_lane_ids=expected_lane_ids,
                    )
                continue
            lane_id = name.removeprefix("lanes/").removesuffix(".json")
            if not lane_id or lane_id.replace("/", "_").replace("\\", "_") != lane_id:
                raise ValueError("unsafe staged lane member identity")
            target = staging_root / "lane_validation_spool" / f"{lane_id}.json"
            staged_identity = _extract_member_once(
                archive,
                name,
                target,
                expected_size=int(row["size"]),
                expected_sha256=str(row["sha256"]),
            )
            if materialize_policy_stores:
                staged_document = stage_self_hashed_policy_document(
                    target,
                    policy_store_path=staging_root
                    / "policy_stores"
                    / f"{lane_id}.sqlite",
                    expected_size=int(row["size"]),
                    expected_sha256=str(row["sha256"]),
                )
                document = staged_document["document"]
                lane_handles[lane_id] = {
                    "lane_id": lane_id,
                    "staged_path": None,
                    "size": staged_identity["size"],
                    "sha256": row["sha256"],
                    "document_self_sha256": document["self_sha256"],
                    "document_metadata": document,
                    "canonical_self_hash_verified": True,
                    "policy_sha256": staged_document["policy_sha256"],
                    "policy_mapping_present": staged_document[
                        "policy_mapping_present"
                    ],
                    "policy_store": staged_document["policy_store"],
                    "staged_member_released_after_policy_spool": True,
                    "expanded_lane_json_retained_with_policy_store": False,
                }
            else:
                streamed = read_self_hashed_json_streamed_mapping(
                    target,
                    streamed_mapping_key="policy",
                    on_member=lambda _key, _value: None,
                )
                identity = streamed["file_identity"]
                if (
                    int(identity["size"]) != int(row["size"])
                    or identity["sha256"] != str(row["sha256"])
                ):
                    raise ValueError(
                        f"result archive lane changed during local validation: {name}"
                    )
                document = streamed["document"]
                policy_present = streamed.get("streamed_mapping_found") is True
                lane_handles[lane_id] = {
                    "lane_id": lane_id,
                    "archive_member": name,
                    "local_archive_path": str(local_archive_path.resolve()),
                    "local_archive_size": int(staged_source["size"]),
                    "local_archive_stat_mtime_ns": local_archive_path.stat().st_mtime_ns,
                    "source_archive_sha256": str(staged_source["sha256"]),
                    "size": int(row["size"]),
                    "sha256": str(row["sha256"]),
                    "document_self_sha256": document["self_sha256"],
                    "document_metadata": None if policy_present else document,
                    "document_binding": {
                        field: document.get(field)
                        for field in (
                            "lane_id",
                            "root_id",
                            "algorithm",
                            "seed",
                            "status",
                        )
                    },
                    "canonical_self_hash_verified": True,
                    "policy_sha256": streamed.get("streamed_mapping_sha256"),
                    "policy_mapping_present": policy_present,
                    "policy_row_count": int(streamed["streamed_member_count"]),
                    "policy_materialization_deferred": True,
                    "expanded_lane_json_retained_with_policy_store": False,
                    "coordinator_retains_complete_lane_metadata": (
                        not policy_present
                    ),
                }
            target.unlink()
    validation_spool = staging_root / "lane_validation_spool"
    if validation_spool.is_dir():
        validation_spool.rmdir()
    if materialize_policy_stores:
        local_archive_path.unlink()
    receipt = with_self_hash(
        {
            "schema_version": "stage21_validated_producer_staging_receipt_v1",
            "stage": expected_stage,
            "token": expected_token,
            "source_archive_size": source_identity["size"],
            "source_archive_sha256": source_identity["sha256"],
            "source_result_manifest_self_sha256": manifest["self_sha256"],
            "staged_lane_count": len(lane_handles),
            "staged_lane_handles": [
                {
                    "lane_id": handle["lane_id"],
                    "size": handle["size"],
                    "sha256": handle["sha256"],
                    "document_self_sha256": handle[
                        "document_self_sha256"
                    ],
                    "policy_sha256": handle["policy_sha256"],
                    "policy_row_count": int(
                        handle.get("policy_row_count")
                        if handle.get("policy_row_count") is not None
                        else handle["policy_store"]["row_count"]
                    ),
                }
                for handle in sorted(
                    lane_handles.values(), key=lambda row: row["lane_id"]
                )
            ],
            "source_archive_stream_hashed_once": True,
            "source_archive_parsed_only_from_local_staged_bytes": True,
            "source_members_decompressed_and_validated_once": True,
            "expanded_lane_json_released_after_validated_policy_spool": (
                materialize_policy_stores
            ),
            "expanded_lane_json_never_retained_by_coordinator": (
                not materialize_policy_stores
            ),
            "coordinator_retains_only_compressed_local_archive": (
                not materialize_policy_stores
            ),
            "policy_stores_materialized_on_demand_per_root": (
                not materialize_policy_stores
            ),
            "local_compressed_archive_retained": not materialize_policy_stores,
            "staged_members_independently_read_back": True,
            "staging_is_not_scientific_authority": True,
        }
    )
    return {
        "source_identity": source_identity,
        "manifest": manifest,
        "lane_handles": lane_handles,
        "captured_members": captured,
        "receipt": receipt,
        "staging_root": str(staging_root.resolve()),
        "local_archive_path": (
            str(local_archive_path.resolve())
            if not materialize_policy_stores
            else None
        ),
    }


def materialize_local_archive_lane_group(
    lane_handles: Sequence[Mapping[str, Any]],
    *,
    staging_root: Path,
) -> list[dict[str, Any]]:
    """Materialize one bounded root/lane group from a validated local ZIP."""

    handles = [dict(handle) for handle in lane_handles]
    if not handles:
        raise ValueError("cannot materialize an empty archive lane group")
    archive_paths = {str(handle.get("local_archive_path")) for handle in handles}
    if len(archive_paths) != 1:
        raise ValueError("one materialization group must bind one local archive")
    archive_path = Path(archive_paths.pop()).resolve()
    expected_sizes = {int(handle.get("local_archive_size", -1)) for handle in handles}
    expected_mtimes = {
        int(handle.get("local_archive_stat_mtime_ns", -1)) for handle in handles
    }
    if len(expected_sizes) != 1 or len(expected_mtimes) != 1:
        raise ValueError("local archive stat binding is inconsistent")
    stat = archive_path.stat()
    if stat.st_size != expected_sizes.pop() or stat.st_mtime_ns != expected_mtimes.pop():
        raise IOError("validated local archive changed before materialization")
    staging_root = staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=False)
    materialized: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for handle in handles:
                lane_id = str(handle["lane_id"])
                name = str(handle["archive_member"])
                target = staging_root / "lane_json" / f"{lane_id}.json"
                _extract_member_once(
                    archive,
                    name,
                    target,
                    expected_size=int(handle["size"]),
                    expected_sha256=str(handle["sha256"]),
                )
                staged = stage_self_hashed_policy_document(
                    target,
                    policy_store_path=staging_root
                    / "policy_stores"
                    / f"{lane_id}.sqlite",
                    expected_size=int(handle["size"]),
                    expected_sha256=str(handle["sha256"]),
                    expected_self_sha256=str(handle["document_self_sha256"]),
                )
                target.unlink()
                expected_document = handle.get("document_metadata")
                expected_binding = handle.get("document_binding")
                if (
                    (
                        isinstance(expected_document, Mapping)
                        and staged["document"] != expected_document
                    )
                    or (
                        not isinstance(expected_document, Mapping)
                        and (
                            not isinstance(expected_binding, Mapping)
                            or any(
                                staged["document"].get(field) != value
                                for field, value in expected_binding.items()
                            )
                        )
                    )
                    or staged["policy_sha256"] != handle.get("policy_sha256")
                    or int(staged["policy_store"]["row_count"])
                    != int(handle.get("policy_row_count", 0))
                ):
                    raise ValueError(
                        f"lane {lane_id} changed between coordinator validation and materialization"
                    )
                materialized.append(
                    {
                        **handle,
                        "document_metadata": staged["document"],
                        "policy_store": staged["policy_store"],
                        "policy_mapping_present": staged[
                            "policy_mapping_present"
                        ],
                        "policy_materialization_deferred": False,
                        "staged_member_released_after_policy_spool": True,
                    }
                )
        lane_json_root = staging_root / "lane_json"
        if lane_json_root.is_dir():
            lane_json_root.rmdir()
        return materialized
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def publish_result(path: Path, raw: bytes) -> dict[str, Any]:
    return atomic_write(path, raw)


def _deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    if name.startswith("/") or "\\" in name or ".." in Path(name).parts:
        raise ValueError(f"unsafe archive member: {name}")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "size": size, "sha256": digest.hexdigest()}


def lightweight_validate_installed_result(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_token: str,
) -> dict[str, Any]:
    bound = InstalledResultValidationReceipt.bind(
        path,
        receipt,
        expected_stage=expected_stage,
        expected_token=expected_token,
    )
    return bound.lightweight_stat_check()


def build_and_publish_result_archive(
    path: Path,
    *,
    token: str,
    stage: str,
    lane_handles: Sequence[Mapping[str, Any]],
    receipt_documents: Mapping[str, Mapping[str, Any]],
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a deterministic result ZIP while retaining at most one lane in RAM.

    Lane handles have already passed durable completion readback. Their compact
    size/hash receipts are used to construct the manifest, and each lane is
    independently rechecked as it is added to the archive. The final ZIP is
    atomically installed, stream-hashed, and then validated directly from disk.
    """

    lane_sources: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    member_rows: list[dict[str, Any]] = []
    for handle in sorted(lane_handles, key=lambda row: str(row["lane_id"])):
        lane_id = str(handle["lane_id"])
        name = f"lanes/{lane_id}.json"
        if name in lane_sources:
            raise ValueError(f"duplicate lane handle: {lane_id}")
        source = Path(str(handle["path"]))
        size = int(handle["size"])
        sha256 = str(handle["sha256"])
        if size < 0 or len(sha256) != 64:
            raise ValueError(f"invalid lane handle receipt: {lane_id}")
        lane_sources[name] = (source, handle)
        member_rows.append({"path": name, "size": size, "sha256": sha256})

    receipt_members = {
        f"receipts/{name}.json": canonical_json_bytes(document)
        for name, document in sorted(receipt_documents.items())
    }
    member_rows.extend(member_manifest(receipt_members))
    manifest = with_self_hash(
        {
            "schema_version": "stage21_result_archive_manifest_v2",
            "stage": stage,
            "token": token,
            "lane_count": len(lane_handles),
            "members": sorted(member_rows, key=lambda row: row["path"]),
            "science_claim": "UNADJUDICATED_MEMBER_RESULTS",
        }
    )
    fixed_members = {
        **receipt_members,
        "result_manifest.json": canonical_json_bytes(manifest),
    }
    if progress_callback is not None:
        progress_callback(
            {
                "event": "result_archive_publication_started",
                "completed_members": 0,
                "total_members": len(lane_sources) + len(fixed_members),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            ordered_names = sorted([*lane_sources, *fixed_members])
            for member_index, name in enumerate(ordered_names, start=1):
                if name in fixed_members:
                    raw = fixed_members[name]
                    archive.writestr(_deterministic_zip_info(name), raw)
                else:
                    source, lane_handle = lane_sources[name]
                    digest = hashlib.sha256()
                    copied = 0
                    with source.open("rb") as source_handle, archive.open(
                        _deterministic_zip_info(name), "w"
                    ) as member_handle:
                        for block in iter(
                            lambda: source_handle.read(1024 * 1024), b""
                        ):
                            member_handle.write(block)
                            digest.update(block)
                            copied += len(block)
                    if copied != int(lane_handle["size"]) or digest.hexdigest() != str(
                        lane_handle["sha256"]
                    ):
                        raise ValueError(f"lane handle identity mismatch: {lane_handle['lane_id']}")
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "result_archive_member_written",
                            "completed_members": member_index,
                            "total_members": len(ordered_names),
                        }
                    )
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
        readback = file_identity(path)
        validated_manifest, _ = read_validated_result_archive_path(
            path,
            expected_stage=stage,
            expected_token=token,
            expected_lane_ids=[
                str(handle["lane_id"]) for handle in lane_handles
            ],
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "result_archive_installed_full_readback_complete",
                    "completed_members": len(lane_sources) + len(fixed_members),
                    "total_members": len(lane_sources) + len(fixed_members),
                }
            )
        installed_stat = path.stat()
        return with_self_hash(
            {
                **readback,
                "schema_version": "stage21_installed_result_validation_receipt_v1",
                "stage": stage,
                "token": token,
                "lane_count": len(lane_handles),
                "result_manifest_self_sha256": validated_manifest["self_sha256"],
                "validated_members": [
                    *validated_manifest["members"],
                    {
                        "path": "result_manifest.json",
                        "size": len(fixed_members["result_manifest.json"]),
                        "sha256": sha256_bytes(
                            fixed_members["result_manifest.json"]
                        ),
                    },
                ],
                "installed_stat_mtime_ns": installed_stat.st_mtime_ns,
                "post_replace_stream_identity_pass_count": 1,
                "full_member_validation_pass_count": 1,
                "lane_payload_json_reparse_during_zip_write": False,
                "lane_payload_streamed_into_zip": True,
                "temporary_zip_pre_replace_full_hash_pass_count": 0,
            }
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
