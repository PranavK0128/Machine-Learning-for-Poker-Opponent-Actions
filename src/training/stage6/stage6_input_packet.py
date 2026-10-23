from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from stage6_common import Stage6ContractError, sha256_file


WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
    *(f"COM{index}" for index in "¹²³"),
    *(f"LPT{index}" for index in "¹²³"),
    "CONIN$",
    "CONOUT$",
}
WINDOWS_INVALID_FILENAME_CHARACTERS = set('<>:"|?*')
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
EOCD = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR = struct.Struct("<4sLQL")
ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
ZIP64_EXTRA_ID = 0x0001
UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF
ALLOWED_ZIP_FLAGS = 0x0800
ALLOWED_COMPRESSION_METHODS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


def _validate_windows_segment(part: str, *, raw_name: str) -> None:
    if part != part.strip() or part.endswith((".", " ")):
        raise Stage6ContractError(
            f"ZIP member has Windows-aliased surrounding whitespace/dot: {raw_name!r}"
        )
    if any(
        ord(character) < 32 or character in WINDOWS_INVALID_FILENAME_CHARACTERS
        for character in part
    ):
        raise Stage6ContractError(
            f"ZIP member has a Windows-invalid filename character: {raw_name!r}"
        )
    basename = part.split(".", 1)[0].rstrip(" .").upper()
    if basename in WINDOWS_RESERVED_BASENAMES:
        raise Stage6ContractError(
            f"ZIP member uses a reserved Windows device name: {raw_name!r}"
        )


def _normalized_member_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name:
        raise Stage6ContractError("ZIP contains an empty or NUL member name")
    if "\\" in raw_name:
        raise Stage6ContractError(
            f"ZIP member must use canonical POSIX separators: {raw_name!r}"
        )
    normalized = unicodedata.normalize("NFC", raw_name)
    if normalized != raw_name:
        raise Stage6ContractError(
            f"ZIP member name must already be NFC-normalized: {raw_name!r}"
        )
    member_body = normalized[:-1] if normalized.endswith("/") else normalized
    raw_parts = member_body.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise Stage6ContractError(f"ZIP member has unsafe path parts: {raw_name!r}")
    path = PurePosixPath(*raw_parts)
    parts = path.parts
    if path.is_absolute() or not parts:
        raise Stage6ContractError(f"ZIP member is absolute: {raw_name!r}")
    if any(":" in part for part in parts):
        raise Stage6ContractError(
            f"ZIP member has a drive-qualified or alternate-stream path: {raw_name!r}"
        )
    for part in parts:
        _validate_windows_segment(part, raw_name=raw_name)
    return PurePosixPath(*parts).as_posix()


def _windows_path_key(normalized: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in PurePosixPath(normalized).parts
    )


def _validated_zip_info_name(info: zipfile.ZipInfo) -> str:
    raw_name = getattr(info, "orig_filename", None)
    if type(raw_name) is not str or raw_name != info.filename:
        raise Stage6ContractError(
            "ZIP raw member name was normalized before validation: "
            f"raw={raw_name!r}, filename={info.filename!r}"
        )
    return _normalized_member_name(raw_name)


def _hash_open_file(source: Any) -> tuple[int, str]:
    source.seek(0)
    digest = hashlib.sha256()
    byte_count = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
        byte_count += len(block)
    source.seek(0)
    return byte_count, digest.hexdigest()


def _read_exact(source: BinaryIO, byte_count: int, *, label: str) -> bytes:
    raw = source.read(byte_count)
    if len(raw) != byte_count:
        raise Stage6ContractError(f"ZIP is truncated while reading {label}")
    return raw


def _parse_extra_fields(raw: bytes, *, label: str) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 4:
            raise Stage6ContractError(f"ZIP {label} extra field is truncated")
        field_id, field_size = struct.unpack_from("<HH", raw, offset)
        offset += 4
        end = offset + field_size
        if end > len(raw):
            raise Stage6ContractError(f"ZIP {label} extra payload is truncated")
        if field_id in fields:
            raise Stage6ContractError(f"ZIP {label} repeats extra field {field_id:#06x}")
        fields[field_id] = raw[offset:end]
        offset = end
    unsupported = sorted(set(fields) - {ZIP64_EXTRA_ID})
    if unsupported:
        raise Stage6ContractError(
            f"ZIP {label} contains unsupported extra fields: {unsupported}"
        )
    return fields


def _zip64_values(
    fields: dict[int, bytes],
    *,
    label: str,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int | None = None,
    disk_number: int | None = None,
) -> tuple[int, int, int | None, int | None]:
    needs = (
        uncompressed_size == UINT32_MAX,
        compressed_size == UINT32_MAX,
        local_offset == UINT32_MAX if local_offset is not None else False,
        disk_number == UINT16_MAX if disk_number is not None else False,
    )
    payload = fields.get(ZIP64_EXTRA_ID)
    if not any(needs):
        if payload is not None:
            raise Stage6ContractError(f"ZIP {label} has an unnecessary ZIP64 extra field")
        return uncompressed_size, compressed_size, local_offset, disk_number
    if payload is None:
        raise Stage6ContractError(f"ZIP {label} is missing its ZIP64 extra field")
    offset = 0

    def consume(width: int, field: str) -> int:
        nonlocal offset
        end = offset + width
        if end > len(payload):
            raise Stage6ContractError(f"ZIP {label} ZIP64 {field} is truncated")
        value = int.from_bytes(payload[offset:end], "little")
        offset = end
        return value

    if needs[0]:
        uncompressed_size = consume(8, "uncompressed size")
    if needs[1]:
        compressed_size = consume(8, "compressed size")
    if needs[2]:
        local_offset = consume(8, "local-header offset")
    if needs[3]:
        disk_number = consume(4, "disk number")
    if offset != len(payload):
        raise Stage6ContractError(f"ZIP {label} ZIP64 payload has trailing bytes")
    return uncompressed_size, compressed_size, local_offset, disk_number


def _decode_zip_name(raw: bytes, flags: int, *, label: str) -> str:
    encoding = "utf-8" if flags & 0x0800 else "cp437"
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise Stage6ContractError(f"ZIP {label} filename is not valid {encoding}") from error


def _validate_raw_zip_envelope(
    source: BinaryIO, archive_bytes: int
) -> list[dict[str, Any]]:
    if archive_bytes < EOCD.size:
        raise Stage6ContractError("ZIP is shorter than an end-of-central-directory record")
    source.seek(archive_bytes - EOCD.size)
    eocd_raw = _read_exact(source, EOCD.size, label="end-of-central-directory")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk_16,
        total_entries_16,
        central_size_32,
        central_offset_32,
        comment_size,
    ) = EOCD.unpack(eocd_raw)
    if signature != b"PK\x05\x06" or comment_size != 0:
        raise Stage6ContractError(
            "ZIP must end exactly at one comment-free end-of-central-directory record"
        )
    if disk_number != 0 or central_disk != 0:
        raise Stage6ContractError("multi-disk ZIP archives are forbidden")

    uses_zip64 = (
        entries_on_disk_16 == UINT16_MAX
        or total_entries_16 == UINT16_MAX
        or central_size_32 == UINT32_MAX
        or central_offset_32 == UINT32_MAX
    )
    eocd_offset = archive_bytes - EOCD.size
    if uses_zip64:
        locator_offset = eocd_offset - ZIP64_LOCATOR.size
        if locator_offset < 0:
            raise Stage6ContractError("ZIP64 locator is missing")
        source.seek(locator_offset)
        locator = ZIP64_LOCATOR.unpack(
            _read_exact(source, ZIP64_LOCATOR.size, label="ZIP64 locator")
        )
        if locator[0] != b"PK\x06\x07" or locator[1] != 0 or locator[3] != 1:
            raise Stage6ContractError("ZIP64 locator is invalid or multi-disk")
        zip64_offset = locator[2]
        if zip64_offset < 0 or zip64_offset + ZIP64_EOCD.size != locator_offset:
            raise Stage6ContractError("ZIP64 end record is not structurally contiguous")
        source.seek(zip64_offset)
        zip64 = ZIP64_EOCD.unpack(
            _read_exact(source, ZIP64_EOCD.size, label="ZIP64 end record")
        )
        (
            zip64_signature,
            zip64_record_size,
            _made_by,
            _needed,
            zip64_disk,
            zip64_central_disk,
            entries_on_disk,
            total_entries,
            central_size,
            central_offset,
        ) = zip64
        if (
            zip64_signature != b"PK\x06\x06"
            or zip64_record_size != 44
            or zip64_disk != 0
            or zip64_central_disk != 0
            or entries_on_disk != total_entries
        ):
            raise Stage6ContractError("ZIP64 end record is invalid or multi-disk")
        central_end = zip64_offset
    else:
        if entries_on_disk_16 != total_entries_16:
            raise Stage6ContractError("central-directory entry counts disagree")
        total_entries = total_entries_16
        central_size = central_size_32
        central_offset = central_offset_32
        central_end = eocd_offset
    if central_offset + central_size != central_end:
        raise Stage6ContractError(
            "ZIP central directory is not contiguous with its terminal records"
        )

    records: list[dict[str, Any]] = []
    source.seek(central_offset)
    for index in range(total_entries):
        raw_header = _read_exact(
            source, CENTRAL_HEADER.size, label=f"central header {index}"
        )
        values = CENTRAL_HEADER.unpack(raw_header)
        if values[0] != b"PK\x01\x02":
            raise Stage6ContractError("ZIP central directory contains a hidden gap or record")
        (
            _,
            made_by,
            needed,
            flags,
            method,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            entry_comment_size,
            entry_disk,
            internal_attributes,
            external_attributes,
            local_offset,
        ) = values
        if flags & ~ALLOWED_ZIP_FLAGS or method not in ALLOWED_COMPRESSION_METHODS:
            raise Stage6ContractError("ZIP uses unsupported flags or compression")
        if entry_comment_size != 0:
            raise Stage6ContractError("ZIP member comments are forbidden")
        raw_name = _read_exact(source, name_size, label=f"central name {index}")
        raw_extra = _read_exact(source, extra_size, label=f"central extra {index}")
        _read_exact(source, entry_comment_size, label=f"central comment {index}")
        extra_fields = _parse_extra_fields(raw_extra, label=f"central member {index}")
        (
            uncompressed_size,
            compressed_size,
            local_offset,
            entry_disk,
        ) = _zip64_values(
            extra_fields,
            label=f"central member {index}",
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_offset=local_offset,
            disk_number=entry_disk,
        )
        if entry_disk != 0:
            raise Stage6ContractError("ZIP member belongs to another disk")
        records.append(
            {
                "raw_name": raw_name,
                "name": _decode_zip_name(raw_name, flags, label=f"central member {index}"),
                "made_by": made_by,
                "needed": needed,
                "flags": flags,
                "method": method,
                "modified_time": modified_time,
                "modified_date": modified_date,
                "crc32": crc32,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "internal_attributes": internal_attributes,
                "external_attributes": external_attributes,
                "local_offset": local_offset,
            }
        )
    if source.tell() != central_offset + central_size:
        raise Stage6ContractError("ZIP central-directory size or entry count is invalid")

    ranges: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        local_offset = record["local_offset"]
        if type(local_offset) is not int or local_offset < 0 or local_offset >= central_offset:
            raise Stage6ContractError("ZIP local-header offset is invalid")
        source.seek(local_offset)
        local_raw = _read_exact(source, LOCAL_HEADER.size, label=f"local header {index}")
        local = LOCAL_HEADER.unpack(local_raw)
        if local[0] != b"PK\x03\x04":
            raise Stage6ContractError("ZIP local header signature is invalid")
        (
            _,
            local_needed,
            local_flags,
            local_method,
            local_time,
            local_date,
            local_crc,
            local_compressed,
            local_uncompressed,
            local_name_size,
            local_extra_size,
        ) = local
        local_name = _read_exact(source, local_name_size, label=f"local name {index}")
        local_extra_raw = _read_exact(
            source, local_extra_size, label=f"local extra {index}"
        )
        if (
            local_name != record["raw_name"]
            or local_flags != record["flags"]
            or local_method != record["method"]
            or local_time != record["modified_time"]
            or local_date != record["modified_date"]
            or local_crc != record["crc32"]
            or local_needed != record["needed"]
        ):
            raise Stage6ContractError("ZIP local and central headers disagree")
        local_extra = _parse_extra_fields(local_extra_raw, label=f"local member {index}")
        local_uncompressed, local_compressed, _, _ = _zip64_values(
            local_extra,
            label=f"local member {index}",
            uncompressed_size=local_uncompressed,
            compressed_size=local_compressed,
        )
        if (
            local_uncompressed != record["uncompressed_size"]
            or local_compressed != record["compressed_size"]
        ):
            raise Stage6ContractError("ZIP local and central sizes disagree")
        data_end = source.tell() + record["compressed_size"]
        if data_end > central_offset:
            raise Stage6ContractError("ZIP compressed member overlaps the central directory")
        ranges.append((local_offset, data_end))
    ordered_ranges = sorted(ranges)
    if records and (
        ordered_ranges[0][0] != 0
        or any(
            ordered_ranges[index][1] != ordered_ranges[index + 1][0]
            for index in range(len(ordered_ranges) - 1)
        )
        or ordered_ranges[-1][1] != central_offset
    ):
        raise Stage6ContractError("ZIP local records contain a prefix, gap, or hidden record")
    if not records and central_offset != 0:
        raise Stage6ContractError("empty ZIP contains bytes outside its directory")
    return records


def _audit_open_zip_packet(
    raw_archive: BinaryIO,
    *,
    archive_label: str,
    max_member_count: int = 100_000,
    max_member_uncompressed_bytes: int = 100 * 1024**3,
    max_total_uncompressed_bytes: int = 200 * 1024**3,
    max_compression_ratio: float = 1_000.0,
) -> dict[str, Any]:
    if not raw_archive.seekable() or not raw_archive.readable():
        raise Stage6ContractError("input ZIP stream must be readable and seekable")
    integer_limits = {
        "max_member_count": max_member_count,
        "max_member_uncompressed_bytes": max_member_uncompressed_bytes,
        "max_total_uncompressed_bytes": max_total_uncompressed_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in integer_limits.values()):
        raise Stage6ContractError("ZIP integer resource limits must be positive integers")
    if (
        isinstance(max_compression_ratio, bool)
        or not isinstance(max_compression_ratio, (int, float))
        or not math.isfinite(float(max_compression_ratio))
        or max_compression_ratio <= 0
    ):
        raise Stage6ContractError(
            "max_compression_ratio must be finite, numeric, and positive"
        )

    members: list[dict[str, Any]] = []
    seen_windows_keys: set[tuple[str, ...]] = set()
    file_windows_keys: set[tuple[str, ...]] = set()
    total_uncompressed = 0
    archive_bytes, archive_sha256 = _hash_open_file(raw_archive)
    raw_records = _validate_raw_zip_envelope(raw_archive, archive_bytes)
    if len(raw_records) > max_member_count:
        raise Stage6ContractError(
            f"ZIP member count {len(raw_records)} exceeds limit {max_member_count}"
        )
    raw_archive.seek(0)
    if not zipfile.is_zipfile(raw_archive):
        raise Stage6ContractError(f"input is not a valid ZIP: {archive_label}")
    raw_archive.seek(0)
    with zipfile.ZipFile(raw_archive) as archive:
        infos = archive.infolist()
        if len(infos) != len(raw_records):
            raise Stage6ContractError("ZIP parser and raw envelope entry counts disagree")
        for index, info in enumerate(infos):
            raw_record = raw_records[index]
            if (
                info.orig_filename != raw_record["name"]
                or info.header_offset != raw_record["local_offset"]
                or info.flag_bits != raw_record["flags"]
                or info.compress_type != raw_record["method"]
                or info.CRC != raw_record["crc32"]
                or info.compress_size != raw_record["compressed_size"]
                or info.file_size != raw_record["uncompressed_size"]
                or info.external_attr != raw_record["external_attributes"]
            ):
                raise Stage6ContractError(
                    "ZIP raw envelope and decoded member metadata disagree"
                )
            normalized = _validated_zip_info_name(info)
            collision_key = _windows_path_key(normalized)
            if collision_key in seen_windows_keys:
                raise Stage6ContractError(
                    f"ZIP has duplicate/colliding member name: {info.filename!r}"
                )
            for prefix_length in range(1, len(collision_key)):
                if collision_key[:prefix_length] in file_windows_keys:
                    raise Stage6ContractError(
                        "ZIP member descends from a path already declared as a file: "
                        f"{info.filename!r}"
                    )
            if not info.is_dir():
                for existing_key in seen_windows_keys:
                    if (
                        len(existing_key) > len(collision_key)
                        and existing_key[: len(collision_key)] == collision_key
                    ):
                        raise Stage6ContractError(
                            "ZIP file path is also a directory prefix: "
                            f"{info.filename!r}"
                        )
                file_windows_keys.add(collision_key)
            seen_windows_keys.add(collision_key)
            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise Stage6ContractError(
                    f"ZIP special filesystem member is forbidden: {normalized}"
                )
            if file_type == stat.S_IFDIR and not info.is_dir():
                raise Stage6ContractError(
                    f"ZIP member has inconsistent directory metadata: {normalized}"
                )
            if info.file_size > max_member_uncompressed_bytes:
                raise Stage6ContractError(f"ZIP member exceeds size limit: {normalized}")
            total_uncompressed += info.file_size
            if total_uncompressed > max_total_uncompressed_bytes:
                raise Stage6ContractError("ZIP total uncompressed size exceeds limit")
            if info.file_size > 0:
                if info.compress_size == 0:
                    raise Stage6ContractError(
                        f"ZIP member has an infinite compression ratio: {normalized}"
                    )
                ratio = info.file_size / info.compress_size
                if ratio > max_compression_ratio:
                    raise Stage6ContractError(
                        f"ZIP member compression ratio exceeds limit: {normalized}"
                    )
            else:
                ratio = 0.0

            digest = None
            if not info.is_dir():
                hasher = hashlib.sha256()
                observed_bytes = 0
                with archive.open(info, "r") as member_source:
                    for block in iter(lambda: member_source.read(1024 * 1024), b""):
                        observed_bytes += len(block)
                        if observed_bytes > info.file_size:
                            raise Stage6ContractError(
                                f"ZIP member expanded beyond declared size: {normalized}"
                            )
                        hasher.update(block)
                if observed_bytes != info.file_size:
                    raise Stage6ContractError(
                        f"ZIP member size readback mismatch: {normalized}"
                    )
                digest = hasher.hexdigest()
            members.append(
                {
                    "path": normalized,
                    "is_directory": info.is_dir(),
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "compression_ratio": ratio,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": digest,
                }
            )
    final_archive_bytes, final_archive_sha256 = _hash_open_file(raw_archive)
    if final_archive_bytes != archive_bytes or final_archive_sha256 != archive_sha256:
        raise Stage6ContractError("input ZIP changed while it was being audited")
    return {
        "schema_version": "stage6_input_zip_audit_v1",
        "status": "PASS",
        "archive": archive_label,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "member_count": len(members),
        "file_member_count": sum(not item["is_directory"] for item in members),
        "total_uncompressed_bytes": total_uncompressed,
        "members": sorted(members, key=lambda item: item["path"]),
        "raw_envelope_validated": True,
    }


def audit_zip_packet(
    zip_path: Path,
    *,
    max_member_count: int = 100_000,
    max_member_uncompressed_bytes: int = 100 * 1024**3,
    max_total_uncompressed_bytes: int = 200 * 1024**3,
    max_compression_ratio: float = 1_000.0,
) -> dict[str, Any]:
    if not zip_path.is_file():
        raise Stage6ContractError(f"input ZIP is missing: {zip_path}")
    with zip_path.open("rb") as raw_archive:
        return _audit_open_zip_packet(
            raw_archive,
            archive_label=str(zip_path.resolve()),
            max_member_count=max_member_count,
            max_member_uncompressed_bytes=max_member_uncompressed_bytes,
            max_total_uncompressed_bytes=max_total_uncompressed_bytes,
            max_compression_ratio=max_compression_ratio,
        )


def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _path_metadata(path: Path, *, label: str, require_directory: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise Stage6ContractError(f"{label} disappeared: {path}") from error
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    if (
        path.is_symlink()
        or is_junction
        or bool(reparse_attribute and file_attributes & reparse_attribute)
    ):
        raise Stage6ContractError(f"{label} reparse point is forbidden: {path}")
    expected_type = stat.S_IFDIR if require_directory else stat.S_IFREG
    if stat.S_IFMT(metadata.st_mode) != expected_type:
        raise Stage6ContractError(f"{label} has the wrong filesystem type: {path}")
    return metadata


def _assert_path_identity(
    path: Path,
    expected: tuple[int, int, int],
    *,
    label: str,
    require_directory: bool,
) -> None:
    observed = _identity(
        _path_metadata(path, label=label, require_directory=require_directory)
    )
    if observed != expected:
        raise Stage6ContractError(f"{label} filesystem identity changed: {path}")


def _capture_directory_chain(
    directory: Path, *, label: str
) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    """Capture every existing ancestor without resolving through reparse points."""

    absolute = Path(os.path.abspath(directory))
    if not os.path.lexists(absolute):
        raise Stage6ContractError(f"{label} must already exist: {absolute}")
    ordered = [*reversed(absolute.parents), absolute]
    chain: list[tuple[Path, tuple[int, int, int]]] = []
    for path in ordered:
        metadata = _path_metadata(
            path, label=f"{label} ancestor", require_directory=True
        )
        chain.append((path, _identity(metadata)))
    return tuple(chain)


def _assert_directory_chain(
    chain: tuple[tuple[Path, tuple[int, int, int]], ...], *, label: str
) -> None:
    if not chain:
        raise Stage6ContractError(f"{label} directory chain is empty")
    for path, expected in chain:
        _assert_path_identity(
            path,
            expected,
            label=f"{label} ancestor",
            require_directory=True,
        )


def _ensure_owned_directory(
    root: Path,
    root_identity: tuple[int, int, int],
    parts: tuple[str, ...],
    parent_chain: tuple[tuple[Path, tuple[int, int, int]], ...],
) -> Path:
    _assert_directory_chain(parent_chain, label="extraction destination")
    _assert_path_identity(
        root,
        root_identity,
        label="extraction staging root",
        require_directory=True,
    )
    current = root
    for part in parts:
        current = current / part
        if os.path.lexists(current):
            _path_metadata(
                current, label="extraction staging directory", require_directory=True
            )
        else:
            _assert_directory_chain(parent_chain, label="extraction destination")
            current.mkdir()
            _path_metadata(
                current, label="created extraction directory", require_directory=True
            )
        _assert_directory_chain(parent_chain, label="extraction destination")
        _assert_path_identity(
            root,
            root_identity,
            label="extraction staging root",
            require_directory=True,
        )
    return current


def _verify_extracted_tree(
    root: Path,
    root_identity: tuple[int, int, int],
    receipts: dict[str, dict[str, Any]],
) -> None:
    pending = [root]
    observed: dict[str, dict[str, Any]] = {}
    while pending:
        directory = pending.pop()
        _assert_path_identity(
            root,
            root_identity,
            label="extracted root",
            require_directory=True,
        )
        _path_metadata(
            directory, label="extracted directory", require_directory=True
        )
        for entry in os.scandir(directory):
            path = Path(entry.path)
            metadata = path.lstat()
            reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if (
                entry.is_symlink()
                or bool(
                    reparse_attribute
                    and getattr(metadata, "st_file_attributes", 0) & reparse_attribute
                )
                or bool(getattr(path, "is_junction", lambda: False)())
            ):
                raise Stage6ContractError(f"extracted reparse point is forbidden: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise Stage6ContractError(
                    f"extracted non-regular filesystem entry is forbidden: {path}"
                )
            relative = path.relative_to(root).as_posix()
            observed[relative] = {
                "bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
    expected_paths = {
        path for path, receipt in receipts.items() if not receipt["is_directory"]
    }
    if set(observed) != expected_paths:
        raise Stage6ContractError(
            "extracted file inventory does not exactly match the audited ZIP members"
        )
    for path, values in observed.items():
        receipt = receipts[path]
        if (
            values["bytes"] != receipt["uncompressed_bytes"]
            or values["sha256"] != receipt["sha256"]
        ):
            raise Stage6ContractError(f"extracted member readback mismatch: {path}")


def safe_extract_zip(
    zip_path: Path,
    target: Path,
    *,
    expected_archive_sha256: str,
    **audit_limits: Any,
) -> dict[str, Any]:
    resolved_target = Path(os.path.abspath(target))
    parent_chain = _capture_directory_chain(
        resolved_target.parent, label="extraction target parent"
    )
    _assert_directory_chain(parent_chain, label="extraction target parent")
    if os.path.lexists(resolved_target):
        raise Stage6ContractError(f"extraction target already exists: {resolved_target}")
    staging: Path | None = None
    staging_created = False
    target_published = False
    staging_identity: tuple[int, int, int] | None = None
    try:
        source_digest = hashlib.sha256()
        _assert_directory_chain(parent_chain, label="extraction destination")
        with Path(zip_path).open("rb") as source, tempfile.TemporaryFile(
            mode="w+b", dir=resolved_target.parent
        ) as captured:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                captured.write(block)
                source_digest.update(block)
            captured.flush()
            os.fsync(captured.fileno())
            if source_digest.hexdigest() != expected_archive_sha256:
                raise Stage6ContractError(
                    "input ZIP hash does not match the immutable packet manifest"
                )
            captured.seek(0)
            report = _audit_open_zip_packet(
                captured,
                archive_label=f"captured input ZIP for {resolved_target}",
                **audit_limits,
            )
            if report["archive_sha256"] != expected_archive_sha256:
                raise Stage6ContractError(
                    "captured input ZIP does not match the immutable packet manifest"
                )
            _assert_directory_chain(parent_chain, label="extraction destination")
            raw_staging = tempfile.mkdtemp(
                prefix=f".{resolved_target.name}.",
                suffix=".stage6_extracting",
                dir=resolved_target.parent,
            )
            staging = Path(raw_staging)
            staging_created = True
            _assert_directory_chain(parent_chain, label="extraction destination")
            staging_identity = _identity(
                _path_metadata(
                    staging,
                    label="created extraction staging",
                    require_directory=True,
                )
            )
            receipt_by_path = {item["path"]: item for item in report["members"]}
            extracted_paths: set[str] = set()
            captured.seek(0)
            with zipfile.ZipFile(captured) as archive:
                for info in archive.infolist():
                    normalized = _validated_zip_info_name(info)
                    parts = PurePosixPath(normalized).parts
                    if info.is_dir():
                        _ensure_owned_directory(
                            staging, staging_identity, parts, parent_chain
                        )
                        continue
                    parent = _ensure_owned_directory(
                        staging, staging_identity, parts[:-1], parent_chain
                    )
                    destination = parent / parts[-1]
                    if os.path.lexists(destination):
                        raise Stage6ContractError(
                            f"extraction destination unexpectedly exists: {normalized}"
                        )
                    _assert_directory_chain(parent_chain, label="extraction destination")
                    with archive.open(info, "r") as member_source, destination.open(
                        "xb"
                    ) as sink:
                        for block in iter(
                            lambda: member_source.read(1024 * 1024), b""
                        ):
                            sink.write(block)
                        sink.flush()
                        os.fsync(sink.fileno())
                    _assert_directory_chain(parent_chain, label="extraction destination")
                    extracted_paths.add(normalized)
                    if sha256_file(destination) != receipt_by_path[normalized]["sha256"]:
                        raise Stage6ContractError(
                            f"extracted member hash mismatch: {normalized}"
                        )
            expected_file_paths = {
                path
                for path, receipt_row in receipt_by_path.items()
                if not receipt_row["is_directory"]
            }
            if extracted_paths != expected_file_paths:
                raise Stage6ContractError(
                    "not every audited ZIP file member was extracted exactly once"
                )
            _verify_extracted_tree(staging, staging_identity, receipt_by_path)
            captured_bytes, captured_sha256 = _hash_open_file(captured)
            if (
                captured_bytes != report["archive_bytes"]
                or captured_sha256 != expected_archive_sha256
            ):
                raise Stage6ContractError("captured input ZIP changed during extraction")
        receipt = {
            "status": "PASS",
            "target": str(resolved_target),
            "archive_sha256": report["archive_sha256"],
            "member_count": report["member_count"],
            "all_member_hashes_read_back": True,
        }
        if os.path.lexists(resolved_target):
            raise Stage6ContractError(
                f"extraction target appeared before publication: {resolved_target}"
            )
        if staging is None or staging_identity is None:
            raise Stage6ContractError("extraction staging identity was not captured")
        _assert_path_identity(
            staging,
            staging_identity,
            label="extraction staging before publication",
            require_directory=True,
        )
        _assert_directory_chain(parent_chain, label="extraction destination")
        os.replace(staging, resolved_target)
        staging_created = False
        target_published = True
        _assert_path_identity(
            resolved_target,
            staging_identity,
            label="published extraction target",
            require_directory=True,
        )
        _assert_directory_chain(parent_chain, label="extraction destination")
        _verify_extracted_tree(resolved_target, staging_identity, receipt_by_path)
        target_published = False
        return receipt
    except BaseException as error:
        for preserved in (staging, resolved_target):
            if preserved is not None and os.path.lexists(preserved):
                error.add_note(
                    "Stage 6 automatic pathname cleanup is intentionally disabled; "
                    f"inspect and explicitly remove the preserved failed artifact: {preserved}"
                )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Stage 6 immutable input ZIP")
    parser.add_argument("--zip", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(audit_zip_packet(args.zip.resolve()), indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
