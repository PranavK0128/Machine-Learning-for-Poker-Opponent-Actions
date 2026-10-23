from __future__ import annotations

import ast
import errno
import hashlib
import json
import os
import re
import stat
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    validate_document_self_hash,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FILE_IDENTITY_CHUNK_BYTES = 8 * 1024**2
MAX_REGULAR_FILE_SNAPSHOT_BYTES = 64 * 1024**2
STAGING_OWNER_RECEIPT_NAME = ".stage6-staging-owner.json"
STAGING_KINDS = frozenset({"partitions", "splits"})
STAGING_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "staging_kind",
        "owner_token",
        "run_identity",
        "relative_path",
        "directory_device",
        "directory_inode",
        "self_sha256",
    }
)


def _is_reparse_point(stat_result: os.stat_result) -> bool:
    attributes = getattr(stat_result, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _stable_regular_file_stat(
    path: Path, before: os.stat_result, after: os.stat_result
) -> os.stat_result:
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise Stage6ContractError(f"file changed while it was read: {path}")
    current = os.lstat(path)
    if (
        not stat.S_ISREG(current.st_mode)
        or _is_reparse_point(current)
        or any(getattr(after, field) != getattr(current, field) for field in stable_fields)
    ):
        raise Stage6ContractError(f"file identity changed after readback: {path}")
    return after


def _regular_file_snapshot(
    path: Path, *, max_bytes: int = MAX_REGULAR_FILE_SNAPSHOT_BYTES
) -> tuple[bytes, os.stat_result]:
    candidate = Path(os.path.abspath(path))
    authority = _active_run_root_authority(candidate) if os.name != "nt" else None
    if authority is None and candidate.is_symlink():
        raise Stage6ContractError(f"required regular file is a link: {path}")
    try:
        if authority is not None:
            authority.assert_path_identity()
            descriptor = authority.open_regular_file(candidate, os.O_RDONLY)
            opened_handle = os.fdopen(descriptor, "rb")
        else:
            opened_handle = candidate.open("rb")
        with opened_handle as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or _is_reparse_point(before):
                raise Stage6ContractError(f"required regular file is invalid: {path}")
            if before.st_size > max_bytes:
                raise Stage6ContractError(
                    f"regular file snapshot exceeds the {max_bytes}-byte cap: {path}"
                )
            chunks: list[bytes] = []
            observed_bytes = 0
            while True:
                chunk = handle.read(min(FILE_IDENTITY_CHUNK_BYTES, max_bytes + 1))
                if not chunk:
                    break
                observed_bytes += len(chunk)
                if observed_bytes > max_bytes:
                    raise Stage6ContractError(
                        f"regular file snapshot exceeds the {max_bytes}-byte cap: {path}"
                    )
                chunks.append(chunk)
            after = os.fstat(handle.fileno())
    except FileNotFoundError as error:
        raise Stage6ContractError(f"required regular file is missing: {path}") from error
    payload = b"".join(chunks)
    if authority is not None:
        current = authority.stat_path(candidate)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise Stage6ContractError(f"file changed while it was read: {candidate}")
        authority.assert_path_identity()
    else:
        _stable_regular_file_stat(candidate, before, after)
    if len(payload) != after.st_size:
        raise Stage6ContractError(f"file byte count changed while it was read: {path}")
    return payload, after


def _stream_regular_file_identity(path: Path) -> tuple[int, str, os.stat_result]:
    """Hash one stable regular-file handle with bounded reads."""

    candidate = Path(os.path.abspath(path))
    authority = _active_run_root_authority(candidate) if os.name != "nt" else None
    if authority is None and candidate.is_symlink():
        raise Stage6ContractError(f"required regular file is a link: {path}")
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        if authority is not None:
            authority.assert_path_identity()
            descriptor = authority.open_regular_file(candidate, os.O_RDONLY)
            opened_handle = os.fdopen(descriptor, "rb")
        else:
            opened_handle = candidate.open("rb")
        with opened_handle as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or _is_reparse_point(before):
                raise Stage6ContractError(f"required regular file is invalid: {path}")
            while True:
                chunk = handle.read(FILE_IDENTITY_CHUNK_BYTES)
                if not chunk:
                    break
                observed_bytes += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except FileNotFoundError as error:
        raise Stage6ContractError(f"required regular file is missing: {path}") from error
    if authority is not None:
        current = authority.stat_path(candidate)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise Stage6ContractError(f"file changed while it was read: {candidate}")
        authority.assert_path_identity()
    else:
        _stable_regular_file_stat(candidate, before, after)
    if observed_bytes != after.st_size:
        raise Stage6ContractError(f"file byte count changed while it was read: {path}")
    return observed_bytes, digest.hexdigest(), after


def regular_file_identity(path: Path, *, relative_path: Path) -> dict[str, Any]:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Stage6ContractError(f"relative file identity path is invalid: {relative_path}")
    byte_count, digest, _ = _stream_regular_file_identity(path)
    return {
        "relative_path": relative.as_posix(),
        "bytes": byte_count,
        "sha256": digest,
    }


def recursive_stage6_code_identities(
    stage6_root: Path, entry_relatives: Iterable[Path]
) -> dict[str, dict[str, Any]]:
    """Return the exact recursive internal Python import closure for entry points."""

    root = stage6_root.resolve()
    source_root = root / "src"
    module_paths: dict[str, Path] = {}
    for path in sorted(source_root.glob("stage6_*.py"), key=lambda item: item.name):
        module = path.stem
        if module in module_paths:
            raise Stage6ContractError(f"duplicate Stage 6 module name: {module}")
        module_paths[module] = path

    pending: list[Path] = []
    for raw_relative in entry_relatives:
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or relative.parent != Path("src")
            or relative.suffix != ".py"
            or relative.stem not in module_paths
        ):
            raise Stage6ContractError(f"invalid Stage 6 code entry point: {relative}")
        pending.append(relative)
    if not pending:
        raise Stage6ContractError("Stage 6 code identity closure has no entry point")

    identities: dict[str, dict[str, Any]] = {}
    while pending:
        relative = pending.pop()
        key = relative.as_posix()
        if key in identities:
            continue
        path = root / relative
        payload, _ = _regular_file_snapshot(path)
        try:
            tree = ast.parse(payload.decode("utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise Stage6ContractError(f"Stage 6 code cannot be parsed: {relative}") from error
        identities[key] = {
            "relative_path": key,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".", 1)[0])
        for module in sorted(imported_modules, reverse=True):
            if not module.startswith("stage6_"):
                continue
            dependency = module_paths.get(module)
            if dependency is None:
                raise Stage6ContractError(
                    f"unresolved internal Stage 6 import {module!r} in {relative}"
                )
            pending.append(dependency.relative_to(root))
    return dict(sorted(identities.items()))


def _strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage6ContractError(f"duplicate JSON key in {label}: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Stage6ContractError(f"nonfinite JSON value in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage6ContractError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict):
        raise Stage6ContractError(f"JSON authority is not an object: {label}")
    return value


def _validate_staging_kind_and_identity(kind: str, run_identity: str) -> None:
    if kind not in STAGING_KINDS:
        raise Stage6ContractError(f"unsupported staging kind: {kind!r}")
    if type(run_identity) is not str or SHA256_PATTERN.fullmatch(run_identity) is None:
        raise Stage6ContractError("staging run identity is not a lowercase SHA-256")


def fsync_regular_file(path: Path) -> None:
    """Flush the exact regular file currently named by *path*."""

    candidate = Path(path)
    assert_active_run_root_identity(candidate)
    authority = _active_run_root_authority(candidate) if os.name != "nt" else None
    if authority is None and candidate.is_symlink():
        raise Stage6ContractError(f"durability target is a link: {candidate}")
    if authority is not None:
        descriptor = authority.open_regular_file(candidate, os.O_RDWR)
        opened_handle = os.fdopen(descriptor, "r+b")
    else:
        opened_handle = candidate.open("r+b")
    with opened_handle as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(before):
            raise Stage6ContractError(
                f"durability target is not a regular file: {candidate}"
            )
        os.fsync(handle.fileno())
        after = os.fstat(handle.fileno())
    current = (
        authority.stat_path(candidate) if authority is not None else os.lstat(candidate)
    )
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(after, field) != getattr(current, field)
        for field in fields
    ):
        raise Stage6ContractError(
            f"durability target identity changed while flushing: {candidate}"
        )
    if authority is not None:
        authority.assert_path_identity()


def _open_windows_directory_handle(path: Path, *, deny_delete: bool) -> int:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    invalid_handle = ctypes.c_void_p(-1).value
    share_mode = 0x00000001 | 0x00000002
    if not deny_delete:
        share_mode |= 0x00000004
    desired_access = 0x80000000 | 0x40000000
    if deny_delete:
        desired_access |= 0x00010000
    handle = create_file(
        str(Path(os.path.abspath(path))),
        desired_access,
        share_mode,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == invalid_handle:
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), path)
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number))


def _release_windows_mutex(handle: int) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    release_mutex = kernel32.ReleaseMutex
    release_mutex.argtypes = [ctypes.c_void_p]
    release_mutex.restype = ctypes.c_int
    if not release_mutex(ctypes.c_void_p(handle)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number))


def _flush_windows_handle(handle: int, path: Path) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = [ctypes.c_void_p]
    flush_file_buffers.restype = ctypes.c_int
    if not flush_file_buffers(ctypes.c_void_p(handle)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), path)


def _is_colab_drive_path(path: Path) -> bool:
    normalized = str(Path(os.path.abspath(path))).replace("\\", "/")
    return normalized == "/content/drive" or normalized.startswith("/content/drive/")


def _fsync_posix_directory_descriptor(descriptor: int, path: Path) -> None:
    """Flush a directory when supported; Colab Drive FUSE rejects this call."""

    try:
        os.fsync(descriptor)
    except OSError as error:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error.errno not in unsupported or not _is_colab_drive_path(path):
            raise


def fsync_directory(path: Path) -> None:
    """Flush retained directory metadata on Windows and POSIX runtimes."""

    candidate = Path(os.path.abspath(path))
    assert_active_run_root_identity(candidate)
    authority = _active_run_root_authority(candidate) if os.name != "nt" else None
    if authority is not None:
        before = authority.stat_path(candidate)
        if not stat.S_ISDIR(before.st_mode) or _is_reparse_point(before):
            raise Stage6ContractError(
                f"durability target is not a directory: {candidate}"
            )
        authority.fsync_directory(candidate)
        after = authority.stat_path(candidate)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise Stage6ContractError(
                f"durability directory identity changed while flushing: {candidate}"
            )
        authority.assert_path_identity()
        return
    before = os.lstat(candidate)
    if not stat.S_ISDIR(before.st_mode) or _is_reparse_point(before):
        raise Stage6ContractError(f"durability target is not a directory: {candidate}")
    if os.name == "nt":
        handle = _open_windows_directory_handle(candidate, deny_delete=False)
        try:
            _flush_windows_handle(handle, candidate)
        finally:
            _close_windows_handle(handle)
    else:
        descriptor = os.open(
            candidate,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            _fsync_posix_directory_descriptor(descriptor, candidate)
        finally:
            os.close(descriptor)
    after = os.lstat(candidate)
    if (
        before.st_dev,
        before.st_ino,
    ) != (
        after.st_dev,
        after.st_ino,
    ):
        raise Stage6ContractError(
            f"durability directory identity changed while flushing: {candidate}"
        )


def ensure_durable_directory(path: Path) -> None:
    """Create missing directory levels and durably retain every directory entry."""

    candidate = Path(os.path.abspath(path))
    assert_active_run_root_identity(candidate)
    missing: list[Path] = []
    cursor = candidate
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir() or cursor.is_symlink():
        raise Stage6ContractError(f"directory ancestor is unsafe: {cursor}")
    for directory in reversed(missing):
        assert_active_run_root_identity(directory)
        create_exclusive_directory(directory)
        fsync_directory(directory)
        fsync_directory(directory.parent)
    fsync_directory(candidate)


def _windows_rename_handle_no_replace(handle: int, destination: Path) -> None:
    import ctypes

    class FileRenameInfo(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", ctypes.c_int),
            ("root_directory", ctypes.c_void_p),
            ("file_name_length", ctypes.c_uint32),
            ("file_name", ctypes.c_wchar * 1),
        ]

    destination_raw = str(Path(os.path.abspath(destination))).encode("utf-16-le")
    buffer_size = FileRenameInfo.file_name.offset + len(destination_raw) + 2
    buffer = ctypes.create_string_buffer(buffer_size)
    information = FileRenameInfo.from_buffer(buffer)
    information.replace_if_exists = 0
    information.root_directory = None
    information.file_name_length = len(destination_raw)
    ctypes.memmove(
        ctypes.addressof(buffer) + FileRenameInfo.file_name.offset,
        destination_raw,
        len(destination_raw),
    )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    if not set_information(
        ctypes.c_void_p(handle), 3, buffer, buffer_size
    ):
        error_number = ctypes.get_last_error()
        if error_number in {80, 183}:
            raise FileExistsError(
                error_number, ctypes.FormatError(error_number), destination
            )
        raise OSError(error_number, ctypes.FormatError(error_number), destination)


def _windows_mark_directory_for_delete(handle: int, path: Path) -> None:
    import ctypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    information = FileDispositionInfo(1)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    if not set_information(
        ctypes.c_void_p(handle),
        4,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), path)


def _open_windows_regular_file_handle(path: Path, *, deny_delete: bool) -> int:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    invalid_handle = ctypes.c_void_p(-1).value
    share_mode = 0x00000001 | 0x00000002
    if not deny_delete:
        share_mode |= 0x00000004
    desired_access = 0x80000000 | 0x40000000
    if deny_delete:
        desired_access |= 0x00010000
    handle = create_file(
        str(Path(os.path.abspath(path))),
        desired_access,
        share_mode,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == invalid_handle:
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), path)
    return int(handle)


def _windows_mark_regular_file_for_delete(handle: int, path: Path) -> None:
    import ctypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    information = FileDispositionInfo(1)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    if not set_information(
        ctypes.c_void_p(handle),
        4,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), path)


def _posix_rename_no_replace(source: Path, destination: Path) -> None:
    """Publish without replacement, including on Colab Drive FUSE."""

    if os.name == "nt":
        raise Stage6ContractError("POSIX no-replace rename requested on Windows")
    source_parent = os.open(
        source.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    destination_parent = os.open(
        destination.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        _posix_rename_no_replace_at(
            source_parent,
            source.name,
            destination_parent,
            destination.name,
            destination_label=destination,
        )
    finally:
        os.close(destination_parent)
        os.close(source_parent)


class _PosixNoReplaceUnsupported(Stage6ContractError):
    """The mounted filesystem cannot perform an atomic no-replace rename."""


def _posix_rename_no_replace_fallback_at(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
    *,
    destination_label: Path,
) -> None:
    """Fail closed; callers must use exclusive final-name construction."""

    del source_parent, source_name, destination_parent, destination_name
    raise _PosixNoReplaceUnsupported(
        f"atomic POSIX no-replace rename is unsupported for {destination_label}"
    )


def _posix_no_replace_reservation_hook(
    _destination_parent: int,
    _destination_name: str,
    _reservation_identity: tuple[int, int],
    _reservation_is_directory: bool,
) -> None:
    """Intentional no-op seam for deterministic reservation-race regressions."""


def _posix_rename_no_replace_at(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
    *,
    destination_label: Path,
) -> None:
    """Rename relative to retained directory descriptors without replacement."""

    if os.name == "nt":
        raise Stage6ContractError("POSIX no-replace rename requested on Windows")
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _posix_rename_no_replace_fallback_at(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
            destination_label=destination_label,
        )
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent,
        os.fsencode(source_name),
        destination_parent,
        os.fsencode(destination_name),
        1,
    )
    if result:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination_label
            )
        if error_number in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            _posix_rename_no_replace_fallback_at(
                source_parent,
                source_name,
                destination_parent,
                destination_name,
                destination_label=destination_label,
            )
            return
        raise OSError(error_number, os.strerror(error_number), destination_label)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _owned_file_interleave_hook(_event: str, _path: Path) -> None:
    """Intentional no-op seam for deterministic file-identity regressions."""


class _CompositeCloseRetryOwner:
    """Retry all distinct cleanup owners without losing an earlier owner."""

    def __init__(self, *owners: Any) -> None:
        self._owners: list[Any] = []
        for owner in owners:
            self._add(owner)

    def _add(self, owner: Any) -> None:
        if owner is None or owner is self:
            return
        if isinstance(owner, _CompositeCloseRetryOwner):
            for nested_owner in tuple(owner._owners):
                self._add(nested_owner)
            return
        if not any(existing is owner for existing in self._owners):
            self._owners.append(owner)

    def close(self) -> None:
        if not self._owners:
            return
        pending = tuple(self._owners)
        self._owners.clear()
        primary_error: BaseException | None = None
        for owner in pending:
            try:
                owner.close()
            except BaseException as close_error:
                retry_owner = getattr(close_error, "stage6_retry_owner", owner)
                self._add(retry_owner)
                self._add(owner)
                if primary_error is None:
                    primary_error = close_error
                else:
                    primary_error.add_note(
                        f"additional retained cleanup owner close failed: {close_error}"
                    )
        if primary_error is not None:
            setattr(primary_error, "stage6_retry_owner", self)
            raise primary_error


def _attach_retry_owner(error: BaseException, owner: Any) -> Any:
    """Attach one retry owner while identity-deduplicating any existing owner."""

    existing = getattr(error, "stage6_retry_owner", None)
    if existing is None or existing is owner:
        merged = owner if existing is None else existing
    else:
        merged = _CompositeCloseRetryOwner(existing, owner)
    setattr(error, "stage6_retry_owner", merged)
    return merged


def _close_authority_preserving_primary(
    authority: Any, primary_error: BaseException, *, label: str
) -> None:
    """Never let constructor cleanup replace the error that required cleanup."""

    try:
        authority.close()
    except BaseException as close_error:
        _attach_retry_owner(primary_error, authority)
        primary_error.add_note(f"{label} close also failed: {close_error}")


class _DescriptorCloseRetryOwner:
    """Retain one descriptor when an injected/rare close failure is retryable."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def close(self) -> None:
        if self._descriptor < 0:
            return
        descriptor = self._descriptor
        os.close(descriptor)
        self._descriptor = -1


class _ParentLeaseRetryOwner:
    """Retain one POSIX parent lease until unlock and close both succeed."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor
        self._unlock_pending = False

    @property
    def descriptor(self) -> int:
        return self._descriptor

    def mark_acquired(self) -> None:
        if self._descriptor < 0:
            raise Stage6ContractError("cannot acquire a closed parent lease")
        self._unlock_pending = True

    def close(self) -> None:
        if self._descriptor < 0:
            return
        descriptor = self._descriptor
        try:
            if self._unlock_pending:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
                self._unlock_pending = False
            os.close(descriptor)
        except BaseException as error:
            _attach_retry_owner(error, self)
            raise
        self._descriptor = -1


class _WindowsMutexRetryOwner:
    """Retain one Windows parent mutex until release and close both succeed."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._release_pending = False

    @property
    def handle(self) -> int:
        return self._handle

    def mark_acquired(self) -> None:
        if self._handle < 0:
            raise Stage6ContractError("cannot acquire a closed parent mutex")
        self._release_pending = True

    def close(self) -> None:
        if self._handle < 0:
            return
        handle = self._handle
        try:
            if self._release_pending:
                _release_windows_mutex(handle)
                self._release_pending = False
            _close_windows_handle(handle)
        except BaseException as error:
            _attach_retry_owner(error, self)
            raise
        self._handle = -1


def _close_descriptor_preserving_primary(
    owner: _DescriptorCloseRetryOwner,
    primary_error: BaseException,
    *,
    label: str,
) -> None:
    try:
        owner.close()
    except BaseException as close_error:
        _attach_retry_owner(primary_error, owner)
        primary_error.add_note(f"{label} close also failed: {close_error}")


def _close_descriptor_with_retry_owner(owner: _DescriptorCloseRetryOwner) -> None:
    try:
        owner.close()
    except BaseException as close_error:
        _attach_retry_owner(close_error, owner)
        raise


class OwnedRegularFileAuthority:
    """Retain one exact regular file across validation, rename, and cleanup."""

    def __init__(
        self,
        allowed_root: Path,
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        self.root = Path(os.path.abspath(allowed_root))
        self.path = Path(os.path.abspath(path))
        self.original_path = self.path
        assert_active_run_root_identity(self.root)
        if not _path_is_within(Path(os.path.abspath(self.path.parent)), self.root):
            raise Stage6ContractError(f"owned file is outside its allowed root: {path}")
        self._run_authority = (
            _active_run_root_authority(self.path) if os.name != "nt" else None
        )
        before = (
            self._run_authority.stat_path(self.path)
            if self._run_authority is not None
            else os.lstat(self.path)
        )
        if not stat.S_ISREG(before.st_mode) or _is_reparse_point(before):
            raise Stage6ContractError(f"owned file is not a regular file: {path}")
        self.identity = (int(before.st_dev), int(before.st_ino))
        if expected_identity is not None and self.identity != expected_identity:
            raise Stage6ContractError("owned file identity changed before authority opened")
        self._windows_handle: int | None = None
        self._descriptor = -1
        self._delete_pending = False
        self.retained_evidence_path: Path | None = None
        try:
            if os.name == "nt":
                self._windows_handle = _open_windows_regular_file_handle(
                    self.path, deny_delete=True
                )
            elif self._run_authority is not None:
                self._descriptor = self._run_authority.open_regular_file(
                    self.path, os.O_RDONLY
                )
            else:
                self._descriptor = os.open(
                    self.path,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                os.set_inheritable(self._descriptor, False)
                opened = os.fstat(self._descriptor)
                if (opened.st_dev, opened.st_ino) != self.identity:
                    raise Stage6ContractError(
                        "retained regular-file descriptor has the wrong identity"
                    )
            _owned_file_interleave_hook("after_authority_open", self.path)
            after = (
                self._run_authority.stat_path(self.path)
                if self._run_authority is not None
                else os.lstat(self.path)
            )
            if (after.st_dev, after.st_ino) != self.identity:
                raise Stage6ContractError(
                    "regular-file identity changed while authority was opened"
                )
        except BaseException as primary_error:
            _close_authority_preserving_primary(
                self,
                primary_error,
                label="retained regular-file constructor authority",
            )
            raise

    def assert_path_identity(self) -> None:
        self._assert_object_identity(require_run_root_identity=True)

    def _assert_object_identity(self, *, require_run_root_identity: bool) -> None:
        if self._delete_pending:
            return
        current = (
            self._run_authority.stat_path(self.path)
            if self._run_authority is not None
            else os.lstat(self.path)
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or _is_reparse_point(current)
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise Stage6ContractError(
                "regular-file pathname no longer names the retained owned object"
            )
        if self._descriptor >= 0:
            opened = os.fstat(self._descriptor)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise Stage6ContractError("retained regular-file descriptor changed")
        if require_run_root_identity and self._run_authority is not None:
            self._run_authority.assert_path_identity()

    def _duplicate_read_descriptor(self) -> int:
        """Duplicate the retained authority without reopening its pathname."""

        self.assert_path_identity()
        descriptor = -1
        if self._windows_handle is not None:
            import ctypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            duplicate_handle = kernel32.DuplicateHandle
            duplicate_handle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_uint32,
            ]
            duplicate_handle.restype = ctypes.c_int
            current_process = kernel32.GetCurrentProcess()
            duplicate = ctypes.c_void_p()
            if not duplicate_handle(
                current_process,
                ctypes.c_void_p(self._windows_handle),
                current_process,
                ctypes.byref(duplicate),
                0,
                0,
                0x00000002,
            ):
                error_number = ctypes.get_last_error()
                raise OSError(
                    error_number, ctypes.FormatError(error_number), self.path
                )
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(duplicate.value), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
                duplicate = ctypes.c_void_p()
            finally:
                if duplicate.value:
                    _close_windows_handle(int(duplicate.value))
        elif self._descriptor >= 0:
            descriptor = os.dup(self._descriptor)
        else:
            raise Stage6ContractError("retained regular-file authority is closed")
        try:
            os.set_inheritable(descriptor, False)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _is_reparse_point(opened)
                or (opened.st_dev, opened.st_ino) != self.identity
            ):
                raise Stage6ContractError(
                    "duplicated regular-file descriptor has the wrong identity"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @contextmanager
    def open_binary_reader(self) -> Iterator[Any]:
        """Yield a reader for the retained object, never its mutable pathname."""

        descriptor = self._duplicate_read_descriptor()
        reader = os.fdopen(descriptor, "rb")
        primary_error: BaseException | None = None
        try:
            before = os.fstat(reader.fileno())
            reader.seek(0)
            yield reader
            after = os.fstat(reader.fileno())
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise Stage6ContractError(
                    "retained regular file changed during exact-object validation"
                )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                reader.close()
            except BaseException as close_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"retained validation reader close also failed: {close_error}"
                    )
                else:
                    raise
            try:
                self.assert_path_identity()
            except BaseException as identity_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "retained validation pathname identity also failed: "
                        f"{identity_error}"
                    )
                else:
                    raise

    def flush(self) -> None:
        self.assert_path_identity()
        if self._windows_handle is not None:
            _flush_windows_handle(self._windows_handle, self.path)
        elif self._descriptor >= 0:
            os.fsync(self._descriptor)
        else:
            raise Stage6ContractError("retained regular-file authority is closed")
        self.assert_path_identity()

    def snapshot(self) -> tuple[int, str, os.stat_result]:
        """Hash the retained file itself, never a later pathname occupant."""

        self.assert_path_identity()
        if self._windows_handle is not None:
            before = os.lstat(self.path)
            descriptor = self._duplicate_read_descriptor()
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                byte_count = 0
                while True:
                    chunk = os.read(descriptor, FILE_IDENTITY_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
            finally:
                os.close(descriptor)
            after = os.lstat(self.path)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise Stage6ContractError("retained regular file changed while hashed")
            if byte_count != before.st_size:
                raise Stage6ContractError("retained regular-file byte count changed")
            self.assert_path_identity()
            return byte_count, digest.hexdigest(), before
        if self._descriptor < 0:
            observed = _stream_regular_file_identity(self.path)
            self.assert_path_identity()
            return observed
        before = os.fstat(self._descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_reparse_point(before)
            or (before.st_dev, before.st_ino) != self.identity
        ):
            raise Stage6ContractError("retained regular-file descriptor is invalid")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(self._descriptor, FILE_IDENTITY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(self._descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise Stage6ContractError("retained regular file changed while hashed")
        if byte_count != before.st_size:
            raise Stage6ContractError("retained regular-file byte count changed")
        self.assert_path_identity()
        return byte_count, digest.hexdigest(), before

    def reopen(self) -> None:
        """Reacquire Windows delete authority after stable path-based validation."""

        if self._delete_pending:
            raise Stage6ContractError("cannot reopen a delete-pending file authority")
        if self._windows_handle is not None or self._descriptor >= 0:
            self.assert_path_identity()
            return
        before = os.lstat(self.path)
        if (before.st_dev, before.st_ino) != self.identity:
            raise Stage6ContractError(
                "regular-file identity changed before authority reacquisition"
            )
        if os.name == "nt":
            self._windows_handle = _open_windows_regular_file_handle(
                self.path, deny_delete=True
            )
        else:
            self._descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            os.set_inheritable(self._descriptor, False)
        try:
            self.assert_path_identity()
        except BaseException as primary_error:
            _close_authority_preserving_primary(
                self,
                primary_error,
                label="retained regular-file reacquisition authority",
            )
            raise
        if self._windows_handle is not None:
            _flush_windows_handle(self._windows_handle, self.path)
        elif self._descriptor >= 0:
            os.fsync(self._descriptor)
        else:
            raise Stage6ContractError("retained regular-file authority is closed")
        self.assert_path_identity()

    def _copy_to_exclusive_final_name(
        self, destination: Path, *, require_run_root_identity: bool
    ) -> None:
        """Publish through an O_EXCL final file when Drive FUSE cannot rename."""

        source = self.path
        expected_bytes, expected_sha256, _ = self.snapshot()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        destination_descriptor = -1
        destination_identity: tuple[int, int] | None = None
        old_descriptor_owner: _DescriptorCloseRetryOwner | None = None
        published = False
        try:
            if self._run_authority is not None:
                destination_descriptor = self._run_authority.open_regular_file(
                    destination, flags
                )
            else:
                destination_descriptor = os.open(
                    destination,
                    flags
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
                os.set_inheritable(destination_descriptor, False)
            created = os.fstat(destination_descriptor)
            destination_identity = (int(created.st_dev), int(created.st_ino))

            source_descriptor = self._duplicate_read_descriptor()
            copied_bytes = 0
            copied_digest = hashlib.sha256()
            try:
                os.lseek(source_descriptor, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(source_descriptor, FILE_IDENTITY_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied_digest.update(chunk)
                    copied_bytes += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_descriptor, view)
                        if written <= 0:
                            raise Stage6ContractError(
                                "exclusive final-name publication made no write progress"
                            )
                        view = view[written:]
            finally:
                os.close(source_descriptor)
            if (copied_bytes, copied_digest.hexdigest()) != (
                expected_bytes,
                expected_sha256,
            ):
                raise Stage6ContractError(
                    "exclusive final-name publication copied the wrong bytes"
                )
            os.fsync(destination_descriptor)
            destination_after = os.fstat(destination_descriptor)
            if (
                not stat.S_ISREG(destination_after.st_mode)
                or _is_reparse_point(destination_after)
                or (int(destination_after.st_dev), int(destination_after.st_ino))
                != destination_identity
                or int(destination_after.st_size) != expected_bytes
            ):
                raise Stage6ContractError(
                    "exclusive final-name publication descriptor changed"
                )
            os.lseek(destination_descriptor, 0, os.SEEK_SET)
            observed_digest = hashlib.sha256()
            observed_bytes = 0
            while True:
                chunk = os.read(destination_descriptor, FILE_IDENTITY_CHUNK_BYTES)
                if not chunk:
                    break
                observed_digest.update(chunk)
                observed_bytes += len(chunk)
            if (observed_bytes, observed_digest.hexdigest()) != (
                expected_bytes,
                expected_sha256,
            ):
                raise Stage6ContractError(
                    "exclusive final-name publication durable readback mismatch"
                )

            _owned_file_interleave_hook(
                "before_no_replace_copy_final_identity", destination
            )
            current = (
                self._run_authority.stat_path(destination)
                if self._run_authority is not None
                else os.lstat(destination)
            )
            if (int(current.st_dev), int(current.st_ino)) != destination_identity:
                raise Stage6ContractError(
                    "exclusive final-name destination changed before publication"
                )
            current_source = (
                self._run_authority.stat_path(source)
                if self._run_authority is not None
                else os.lstat(source)
            )
            if (int(current_source.st_dev), int(current_source.st_ino)) != self.identity:
                raise Stage6ContractError(
                    "exclusive final-name source changed before publication"
                )

            old_descriptor_owner = _DescriptorCloseRetryOwner(self._descriptor)
            self._descriptor = destination_descriptor
            destination_descriptor = -1
            self.path = destination
            self.identity = destination_identity
            published = True
            self._assert_object_identity(
                require_run_root_identity=require_run_root_identity
            )

            source_parent = None
            try:
                if self._run_authority is not None:
                    source_parent = self._run_authority.open_directory_descriptor(
                        source.parent
                    )
                    current_source = os.stat(
                        source.name, dir_fd=source_parent, follow_symlinks=False
                    )
                else:
                    current_source = os.lstat(source)
                if (
                    int(current_source.st_dev),
                    int(current_source.st_ino),
                ) == (int(created.st_dev), int(created.st_ino)):
                    raise Stage6ContractError(
                        "exclusive final-name source and destination identities alias"
                    )
                old_status = os.fstat(old_descriptor_owner._descriptor)
                if (
                    int(current_source.st_dev),
                    int(current_source.st_ino),
                ) == (int(old_status.st_dev), int(old_status.st_ino)):
                    _owned_file_interleave_hook(
                        "before_no_replace_copy_source_cleanup", source
                    )
                    if self._run_authority is not None:
                        rechecked = os.stat(
                            source.name,
                            dir_fd=source_parent,
                            follow_symlinks=False,
                        )
                        if (int(rechecked.st_dev), int(rechecked.st_ino)) == (
                            int(old_status.st_dev),
                            int(old_status.st_ino),
                        ):
                            os.unlink(source.name, dir_fd=source_parent)
                        else:
                            self.retained_evidence_path = source
                    else:
                        rechecked = os.lstat(source)
                        if (int(rechecked.st_dev), int(rechecked.st_ino)) == (
                            int(old_status.st_dev),
                            int(old_status.st_ino),
                        ):
                            os.unlink(source)
                        else:
                            self.retained_evidence_path = source
                else:
                    self.retained_evidence_path = source
            except FileNotFoundError:
                pass
            except OSError:
                self.retained_evidence_path = source
            finally:
                if source_parent is not None:
                    os.close(source_parent)
                _close_descriptor_with_retry_owner(old_descriptor_owner)
                old_descriptor_owner = None
        except BaseException as primary_error:
            if not published and destination_identity is not None:
                try:
                    current = (
                        self._run_authority.stat_path(destination)
                        if self._run_authority is not None
                        else os.lstat(destination)
                    )
                    if (int(current.st_dev), int(current.st_ino)) == destination_identity:
                        if self._run_authority is not None:
                            parent = self._run_authority.open_directory_descriptor(
                                destination.parent
                            )
                            try:
                                os.unlink(destination.name, dir_fd=parent)
                            finally:
                                os.close(parent)
                        else:
                            os.unlink(destination)
                except FileNotFoundError:
                    pass
            if old_descriptor_owner is not None:
                if (
                    getattr(primary_error, "stage6_retry_owner", None)
                    is not old_descriptor_owner
                ):
                    _close_descriptor_preserving_primary(
                        old_descriptor_owner,
                        primary_error,
                        label="pre-publication source descriptor",
                    )
                old_descriptor_owner = None
            raise
        finally:
            if destination_descriptor >= 0:
                os.close(destination_descriptor)

    def rename_to(
        self, destination: Path, *, require_run_root_identity: bool = True
    ) -> None:
        target = Path(os.path.abspath(destination))
        target_parent = Path(os.path.abspath(target.parent))
        if not _path_is_within(target_parent, self.root):
            raise Stage6ContractError(f"owned-file destination is outside root: {target}")
        target_exists = False
        if self._run_authority is not None:
            try:
                self._run_authority.stat_path(target)
            except FileNotFoundError:
                pass
            else:
                target_exists = True
        else:
            target_exists = os.path.lexists(target)
        if target_exists:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
        source = self.path
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )
        _owned_file_interleave_hook("before_no_replace_rename_source", source)
        _owned_file_interleave_hook("before_no_replace_rename_destination", target)
        if require_run_root_identity:
            assert_active_run_root_identity(source)
            assert_active_run_root_identity(target)
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )
        if os.name == "nt" and self._windows_handle is None:
            raise Stage6ContractError("Windows regular-file rename authority is closed")
        copied_to_final_name = False
        if self._windows_handle is not None:
            _windows_rename_handle_no_replace(self._windows_handle, target)
        elif self._run_authority is not None:
            try:
                self._run_authority.rename_no_replace(source, target)
            except _PosixNoReplaceUnsupported:
                self._copy_to_exclusive_final_name(
                    target,
                    require_run_root_identity=require_run_root_identity,
                )
                copied_to_final_name = True
        else:
            try:
                _posix_rename_no_replace(source, target)
            except _PosixNoReplaceUnsupported:
                self._copy_to_exclusive_final_name(
                    target,
                    require_run_root_identity=require_run_root_identity,
                )
                copied_to_final_name = True
        if copied_to_final_name:
            return
        self.path = target
        try:
            moved = (
                self._run_authority.stat_path(target)
                if self._run_authority is not None
                else os.lstat(target)
            )
            if (
                not stat.S_ISREG(moved.st_mode)
                or _is_reparse_point(moved)
                or (moved.st_dev, moved.st_ino) != self.identity
            ):
                raise Stage6ContractError(
                    "no-replace transaction moved a foreign regular file"
                )
        except BaseException as primary_error:
            target_still_exists = False
            if self._run_authority is not None:
                try:
                    self._run_authority.stat_path(target)
                except FileNotFoundError:
                    pass
                else:
                    target_still_exists = True
            else:
                target_still_exists = os.path.lexists(target)
            if target_still_exists:
                try:
                    if os.name == "nt":
                        if self._windows_handle is None:
                            raise Stage6ContractError(
                                "Windows regular-file restoration authority is closed"
                            )
                        _windows_rename_handle_no_replace(
                            self._windows_handle, source
                        )
                    elif self._run_authority is not None:
                        self._run_authority.rename_no_replace(target, source)
                    else:
                        _posix_rename_no_replace(target, source)
                    self.path = source
                except BaseException as restore_error:
                    primary_error.add_note(
                        f"foreign regular-file restoration also failed: {restore_error}"
                    )
            raise
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )

    def delete(self) -> None:
        """Remove the public name without a racy POSIX pathname deletion.

        Windows can mark the retained handle itself for deletion. POSIX has no
        unlink-by-descriptor primitive, so the exact object is atomically
        retired under an unguessable evidence name and deliberately retained.
        """

        self._assert_object_identity(require_run_root_identity=False)
        if self._windows_handle is not None:
            _windows_mark_regular_file_for_delete(self._windows_handle, self.path)
            self._delete_pending = True
            return
        retired = self.path.with_name(
            f".{self.path.name}.owned-delete-{uuid.uuid4().hex}"
        )
        self.rename_to(retired, require_run_root_identity=False)
        self.retained_evidence_path = retired
        _owned_file_interleave_hook("after_safe_retire", retired)
        self._assert_object_identity(require_run_root_identity=False)

    def close(self) -> None:
        if self._descriptor >= 0:
            descriptor = self._descriptor
            os.close(descriptor)
            self._descriptor = -1
        if self._windows_handle is not None:
            handle = self._windows_handle
            _close_windows_handle(handle)
            self._windows_handle = None

    def __enter__(self) -> "OwnedRegularFileAuthority":
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc_value: object,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            _attach_retry_owner(close_error, self)
            if isinstance(exc_value, BaseException):
                _attach_retry_owner(exc_value, self)
                exc_value.add_note(
                    f"retained regular-file authority close failed: {close_error}"
                )
            else:
                raise


@contextmanager
def open_identity_bound_regular_file(
    allowed_root: Path,
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Iterator[Any]:
    """Yield one exact receipt-bound regular file, never a later path occupant."""

    if type(expected_bytes) is not int or expected_bytes < 0:
        raise Stage6ContractError("expected regular-file byte count is invalid")
    if type(expected_sha256) is not str or SHA256_PATTERN.fullmatch(
        expected_sha256
    ) is None:
        raise Stage6ContractError("expected regular-file SHA-256 is invalid")
    candidate = Path(os.path.abspath(path))
    with OwnedRegularFileAuthority(allowed_root, candidate) as authority:
        observed_bytes, observed_sha256, _ = authority.snapshot()
        if (observed_bytes, observed_sha256) != (
            expected_bytes,
            expected_sha256,
        ):
            raise Stage6ContractError(
                f"regular-file identity changed from its frozen receipt: {candidate}"
            )
        _owned_file_interleave_hook("after_identity_bound_snapshot", candidate)
        authority.assert_path_identity()
        with authority.open_binary_reader() as reader:
            yield reader
        final_bytes, final_sha256, _ = authority.snapshot()
        if (final_bytes, final_sha256) != (expected_bytes, expected_sha256):
            raise Stage6ContractError(
                f"receipt-bound regular-file identity changed during consumption: "
                f"{candidate}"
            )
        authority.assert_path_identity()


def _delete_owned_file_preserving_primary(
    authority: OwnedRegularFileAuthority, primary_error: BaseException
) -> None:
    try:
        authority.delete()
    except BaseException as cleanup_error:
        primary_error.add_note(f"owned regular-file cleanup also failed: {cleanup_error}")


def atomic_publish_owned_regular_file(
    temporary: Path,
    destination: Path,
    *,
    expected_creation_identity: tuple[int, int],
    directory_fsync: Callable[[Path], None] = fsync_directory,
    validate: Callable[[Any], None] | None = None,
    accept_existing_identical: bool = True,
) -> dict[str, Any]:
    """Publish exact bytes and validate only identity-bound open objects."""

    temporary = Path(os.path.abspath(temporary))
    destination = Path(os.path.abspath(destination))
    assert_active_run_root_identity(temporary)
    assert_active_run_root_identity(destination)
    if temporary.parent != destination.parent:
        raise Stage6ContractError("owned regular-file publication must stay in one directory")
    status = "PUBLISHED_NEW"
    with OwnedRegularFileAuthority(
        destination.parent,
        temporary,
        expected_identity=expected_creation_identity,
    ) as authority:
        published = False
        preserve_temporary_on_error = False
        try:
            authority.flush()
            byte_count, digest, _ = authority.snapshot()
            _owned_file_interleave_hook("before_publish", destination)
            try:
                authority.rename_to(destination)
            except FileExistsError:
                if not accept_existing_identical:
                    raise
                preserve_temporary_on_error = True
                retry_identity = {
                    "bytes": byte_count,
                    "sha256": digest,
                    "publication_status": "EXISTING_IDENTICAL",
                }
                with OwnedRegularFileAuthority(
                    destination.parent, destination
                ) as existing_authority:
                    existing_authority.flush()
                    existing_bytes, existing_digest, _ = existing_authority.snapshot()
                    if (existing_bytes, existing_digest) != (byte_count, digest):
                        raise Stage6ContractError(
                            "existing destination differs from deterministic retry: "
                            f"{destination}"
                        )
                    _owned_file_interleave_hook(
                        "after_existing_identical_snapshot", destination
                    )
                    existing_authority.assert_path_identity()
                    directory_fsync(destination.parent)
                    existing_authority.assert_path_identity()
                    if validate is not None:
                        with existing_authority.open_binary_reader() as reader:
                            validate(reader)
                    observed_bytes, observed_digest, _ = existing_authority.snapshot()
                    if (observed_bytes, observed_digest) != (byte_count, digest):
                        raise Stage6ContractError(
                            "existing deterministic-retry destination changed during "
                            f"validation: {destination}"
                        )
                    existing_authority.assert_path_identity()
                # A close failure above must preserve the exact temporary for
                # evidence and deterministic retry. Retire it only after the
                # retained destination authority has closed successfully.
                authority.delete()
                return retry_identity
            published = True
            directory_fsync(destination.parent)
            authority.assert_path_identity()
            observed_bytes, observed_digest, _ = authority.snapshot()
            if (observed_bytes, observed_digest) != (byte_count, digest):
                raise Stage6ContractError(
                    f"published regular-file readback mismatch: {destination}"
                )
            if validate is not None:
                with authority.open_binary_reader() as reader:
                    validate(reader)
            observed_bytes, observed_digest, _ = authority.snapshot()
            if (observed_bytes, observed_digest) != (byte_count, digest):
                raise Stage6ContractError(
                    f"published regular-file changed during validation: {destination}"
                )
            authority.assert_path_identity()
        except BaseException as primary_error:
            if published:
                recovery = temporary
                if os.path.lexists(recovery):
                    recovery = destination.with_name(
                        f".{destination.name}.owned-recovery-{uuid.uuid4().hex}"
                    )
                try:
                    _owned_file_interleave_hook("before_rollback", destination)
                    authority.rename_to(
                        recovery, require_run_root_identity=False
                    )
                    directory_fsync(destination.parent)
                except BaseException as rollback_error:
                    primary_error.add_note(
                        f"identity-bound regular-file rollback also failed: {rollback_error}"
                    )
            if not preserve_temporary_on_error:
                try:
                    authority.assert_path_identity()
                except BaseException:
                    pass
                else:
                    _delete_owned_file_preserving_primary(authority, primary_error)
            raise
    return {
        "bytes": byte_count,
        "sha256": digest,
        "publication_status": status,
    }


def move_verified_regular_file(
    allowed_root: Path,
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Move only the exact validated regular file; never consume a replacement."""

    assert_active_run_root_identity(source)
    assert_active_run_root_identity(destination)
    with OwnedRegularFileAuthority(
        allowed_root, source, expected_identity=expected_identity
    ) as authority:
        authority.rename_to(destination)
        fsync_directory(source.parent)
        if destination.parent != source.parent:
            fsync_directory(destination.parent)


def remove_verified_regular_file(
    allowed_root: Path,
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only the exact regular file identified by the retained receipt."""

    assert_active_run_root_identity(path)
    with OwnedRegularFileAuthority(
        allowed_root, path, expected_identity=expected_identity
    ) as authority:
        authority.delete()
        fsync_directory(path.parent)


_PARENT_LEASE_KEYS: set[str] = set()
_PARENT_LEASE_GUARD = threading.Lock()
_ACTIVE_RUN_ROOT_AUTHORITIES: dict[str, "RetainedRunRootAuthority"] = {}
_ACTIVE_RUN_ROOT_GUARD = threading.Lock()


def _run_root_mutation_interleave_hook(_event: str, _path: Path) -> None:
    """Intentional no-op seam immediately before an anchored mutation syscall."""


class RetainedRunRootAuthority:
    """Retain the exact run-root object for the complete parent run."""

    def __init__(self, run_root: Path) -> None:
        self.root = Path(os.path.abspath(run_root))
        before = os.lstat(self.root)
        if not stat.S_ISDIR(before.st_mode) or _is_reparse_point(before):
            raise Stage6ContractError(f"run root is not a stable directory: {self.root}")
        self.identity = (int(before.st_dev), int(before.st_ino))
        self._windows_handle: int | None = None
        self._descriptor = -1
        self._descendant_directory_anchors: dict[
            str, tuple[Path, int, tuple[int, int]]
        ] = {}
        try:
            if os.name == "nt":
                self._windows_handle = _open_windows_directory_handle(
                    self.root, deny_delete=True
                )
            else:
                self._descriptor = os.open(
                    self.root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                os.set_inheritable(self._descriptor, False)
                opened = os.fstat(self._descriptor)
                if (opened.st_dev, opened.st_ino) != self.identity:
                    raise Stage6ContractError(
                        "retained run-root descriptor has the wrong identity"
                    )
            after = os.lstat(self.root)
            if (after.st_dev, after.st_ino) != self.identity:
                raise Stage6ContractError(
                    "run-root identity changed while authority was opened"
                )
        except BaseException as primary_error:
            _close_authority_preserving_primary(
                self,
                primary_error,
                label="retained run-root constructor authority",
            )
            raise

    def assert_path_identity(self) -> None:
        current = os.lstat(self.root)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_reparse_point(current)
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise Stage6ContractError(
                "run-root pathname no longer names the retained owned directory"
            )
        if self._descriptor >= 0:
            opened = os.fstat(self._descriptor)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise Stage6ContractError("retained run-root descriptor identity changed")
        elif os.name == "nt" and self._windows_handle is None:
            raise Stage6ContractError("retained Windows run-root authority is closed")

    def _assert_retained_identity(self) -> None:
        if os.name == "nt":
            if self._windows_handle is None:
                raise Stage6ContractError("retained Windows run-root authority is closed")
            return
        if self._descriptor < 0:
            raise Stage6ContractError("retained POSIX run-root authority is closed")
        opened = os.fstat(self._descriptor)
        if (opened.st_dev, opened.st_ino) != self.identity:
            raise Stage6ContractError("retained run-root descriptor identity changed")

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        candidate = Path(os.path.abspath(path))
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as error:
            raise Stage6ContractError(
                f"path is outside retained run-root authority: {candidate}"
            ) from error
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise Stage6ContractError(
                f"path is not canonical below retained run root: {candidate}"
            )
        return tuple(relative.parts)

    def _open_directory_descriptor_from_root(self, path: Path) -> int:
        """Open below the root descriptor without consulting descendant anchors."""

        self._assert_retained_identity()
        parts = self._relative_parts(path)
        descriptor = os.dup(self._descriptor)
        os.set_inheritable(descriptor, False)
        try:
            for part in parts:
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                os.set_inheritable(child, False)
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or _is_reparse_point(opened):
                    os.close(child)
                    raise Stage6ContractError(
                        f"retained descendant is not a stable directory: {path}"
                    )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _assert_descendant_anchor_path_identity(
        self, anchor_path: Path, descriptor: int, identity: tuple[int, int]
    ) -> None:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _is_reparse_point(opened)
            or (opened.st_dev, opened.st_ino) != identity
        ):
            raise Stage6ContractError("retained descendant-directory anchor changed")
        parts = self._relative_parts(anchor_path)
        if not parts:
            raise Stage6ContractError("run root cannot be a descendant anchor")
        parent = self._open_directory_descriptor_from_root(anchor_path.parent)
        try:
            current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_reparse_point(current)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise Stage6ContractError(
                "descendant-directory pathname no longer names its retained object"
            )

    def register_descendant_directory_anchor(
        self,
        path: Path,
        descriptor: int,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        """Take ownership of a just-created exact descendant directory descriptor."""

        if os.name == "nt":
            raise Stage6ContractError("descendant descriptor anchors are POSIX-only")
        candidate = Path(os.path.abspath(path))
        self._relative_parts(candidate)
        key = os.path.normcase(str(candidate))
        if key in self._descendant_directory_anchors:
            raise Stage6ContractError("descendant-directory anchor already exists")
        self._assert_descendant_anchor_path_identity(
            candidate, descriptor, expected_identity
        )
        self._descendant_directory_anchors[key] = (
            candidate,
            descriptor,
            expected_identity,
        )

    def ensure_descendant_directory_anchor(
        self, path: Path, *, expected_identity: tuple[int, int]
    ) -> None:
        """Retain a verified existing descendant before any resumed child write."""

        if os.name == "nt":
            raise Stage6ContractError("descendant descriptor anchors are POSIX-only")
        candidate = Path(os.path.abspath(path))
        self._relative_parts(candidate)
        key = os.path.normcase(str(candidate))
        existing = self._descendant_directory_anchors.get(key)
        if existing is not None:
            _, descriptor, identity = existing
            if identity != expected_identity:
                raise Stage6ContractError(
                    "descendant-directory anchor identity mismatch"
                )
            self._assert_descendant_anchor_path_identity(
                candidate, descriptor, expected_identity
            )
            return

        descriptor = self._open_directory_descriptor_from_root(candidate)
        try:
            opened = os.fstat(descriptor)
            if (int(opened.st_dev), int(opened.st_ino)) != expected_identity:
                raise Stage6ContractError(
                    "existing descendant directory has the wrong identity"
                )
            self.register_descendant_directory_anchor(
                candidate,
                descriptor,
                expected_identity=expected_identity,
            )
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def release_descendant_directory_anchor(
        self, path: Path, *, expected_identity: tuple[int, int]
    ) -> None:
        candidate = Path(os.path.abspath(path))
        key = os.path.normcase(str(candidate))
        entry = self._descendant_directory_anchors.get(key)
        if entry is None:
            return
        _, descriptor, identity = entry
        if identity != expected_identity:
            raise Stage6ContractError("descendant-directory anchor identity mismatch")
        os.close(descriptor)
        del self._descendant_directory_anchors[key]

    def open_directory_descriptor(self, path: Path) -> int:
        """Open an exact descendant through the deepest retained directory anchor."""

        if os.name == "nt":
            raise Stage6ContractError("descriptor-relative directory open is POSIX-only")
        candidate = Path(os.path.abspath(path))
        anchors = [
            entry
            for entry in self._descendant_directory_anchors.values()
            if candidate == entry[0] or _path_is_within(candidate, entry[0])
        ]
        if not anchors:
            return self._open_directory_descriptor_from_root(candidate)
        anchor_path, anchor_descriptor, identity = max(
            anchors, key=lambda entry: len(entry[0].parts)
        )
        self._assert_descendant_anchor_path_identity(
            anchor_path, anchor_descriptor, identity
        )
        descriptor = os.dup(anchor_descriptor)
        os.set_inheritable(descriptor, False)
        relative_parts = tuple(candidate.relative_to(anchor_path).parts)
        try:
            for part in relative_parts:
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                os.set_inheritable(child, False)
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or _is_reparse_point(opened):
                    os.close(child)
                    raise Stage6ContractError(
                        f"retained descendant is not a stable directory: {path}"
                    )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def stat_path(self, path: Path) -> os.stat_result:
        candidate = Path(os.path.abspath(path))
        parts = self._relative_parts(candidate)
        if not parts:
            self._assert_retained_identity()
            return os.fstat(self._descriptor)
        parent = self.open_directory_descriptor(candidate.parent)
        try:
            return os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)

    def open_regular_file(self, path: Path, flags: int, mode: int = 0o600) -> int:
        candidate = Path(os.path.abspath(path))
        parts = self._relative_parts(candidate)
        if not parts:
            raise Stage6ContractError("run root itself cannot be opened as a regular file")
        parent = self.open_directory_descriptor(candidate.parent)
        try:
            descriptor = os.open(
                parts[-1],
                flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
        os.set_inheritable(descriptor, False)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse_point(opened):
            os.close(descriptor)
            raise Stage6ContractError(f"retained descendant is not a regular file: {path}")
        return descriptor

    def mkdir(self, path: Path, *, mode: int = 0o700) -> os.stat_result:
        candidate = Path(os.path.abspath(path))
        parts = self._relative_parts(candidate)
        if not parts:
            raise Stage6ContractError("run root itself cannot be created as a descendant")
        parent = self.open_directory_descriptor(candidate.parent)
        try:
            os.mkdir(parts[-1], mode=mode, dir_fd=parent)
            created = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            _fsync_posix_directory_descriptor(parent, candidate.parent)
        finally:
            os.close(parent)
        if not stat.S_ISDIR(created.st_mode) or _is_reparse_point(created):
            raise Stage6ContractError(f"created descendant is not a directory: {path}")
        return created

    def mkdir_and_open(
        self, path: Path, *, mode: int = 0o700
    ) -> tuple[int, os.stat_result]:
        """Create and immediately retain the exact new descendant directory."""

        candidate = Path(os.path.abspath(path))
        parts = self._relative_parts(candidate)
        if not parts:
            raise Stage6ContractError("run root itself cannot be created as a descendant")
        parent = self.open_directory_descriptor(candidate.parent)
        descriptor = -1
        try:
            os.mkdir(parts[-1], mode=mode, dir_fd=parent)
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            os.set_inheritable(descriptor, False)
            created = os.fstat(descriptor)
            current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(created.st_mode)
                or _is_reparse_point(created)
                or (created.st_dev, created.st_ino)
                != (current.st_dev, current.st_ino)
            ):
                raise Stage6ContractError(
                    f"created descendant identity could not be retained: {path}"
                )
            _fsync_posix_directory_descriptor(parent, candidate.parent)
            return descriptor, created
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        finally:
            os.close(parent)

    def rename_no_replace(self, source: Path, destination: Path) -> None:
        source_candidate = Path(os.path.abspath(source))
        destination_candidate = Path(os.path.abspath(destination))
        source_parts = self._relative_parts(source_candidate)
        destination_parts = self._relative_parts(destination_candidate)
        if not source_parts or not destination_parts:
            raise Stage6ContractError("run root itself cannot be renamed")
        source_parent = self.open_directory_descriptor(source_candidate.parent)
        destination_parent = self.open_directory_descriptor(destination_candidate.parent)
        try:
            _posix_rename_no_replace_at(
                source_parent,
                source_parts[-1],
                destination_parent,
                destination_parts[-1],
                destination_label=destination_candidate,
            )
            _fsync_posix_directory_descriptor(source_parent, source_candidate.parent)
            if source_parent != destination_parent:
                _fsync_posix_directory_descriptor(
                    destination_parent, destination_candidate.parent
                )
        finally:
            os.close(destination_parent)
            os.close(source_parent)

    def fsync_directory(self, path: Path) -> None:
        descriptor = self.open_directory_descriptor(path)
        try:
            _fsync_posix_directory_descriptor(descriptor, path)
        finally:
            os.close(descriptor)

    def close(self) -> None:
        for key, (_, descriptor, _) in tuple(
            self._descendant_directory_anchors.items()
        ):
            os.close(descriptor)
            del self._descendant_directory_anchors[key]
        if self._descriptor >= 0:
            descriptor = self._descriptor
            os.close(descriptor)
            self._descriptor = -1
        if self._windows_handle is not None:
            handle = self._windows_handle
            _close_windows_handle(handle)
            self._windows_handle = None


def assert_active_run_root_identity(path: Path) -> None:
    """Fail closed before mutating any path under an active run root."""

    candidate = Path(os.path.abspath(path))
    with _ACTIVE_RUN_ROOT_GUARD:
        authorities = tuple(_ACTIVE_RUN_ROOT_AUTHORITIES.values())
    for authority in authorities:
        if candidate == authority.root or _path_is_within(candidate, authority.root):
            authority.assert_path_identity()


def _active_run_root_authority(path: Path) -> RetainedRunRootAuthority | None:
    candidate = Path(os.path.abspath(path))
    with _ACTIVE_RUN_ROOT_GUARD:
        authorities = tuple(_ACTIVE_RUN_ROOT_AUTHORITIES.values())
    matches = [
        authority
        for authority in authorities
        if candidate == authority.root or _path_is_within(candidate, authority.root)
    ]
    if not matches:
        return None
    return max(matches, key=lambda authority: len(authority.root.parts))


def create_exclusive_regular_file(path: Path) -> tuple[int, os.stat_result]:
    """Create one exact regular file through retained run-root authority."""

    candidate = Path(os.path.abspath(path))
    authority = _active_run_root_authority(candidate)
    if authority is not None and os.name != "nt":
        authority.assert_path_identity()
        _run_root_mutation_interleave_hook(
            "before_exclusive_regular_file_create", candidate
        )
        descriptor = authority.open_regular_file(
            candidate, os.O_RDWR | os.O_CREAT | os.O_EXCL
        )
        created = os.fstat(descriptor)
        try:
            authority.assert_path_identity()
        except BaseException as primary_error:
            os.close(descriptor)
            retained = candidate.parent / (
                f".{candidate.name}.owned-root-race-retained-{uuid.uuid4().hex}"
            )
            try:
                authority.rename_no_replace(candidate, retained)
                primary_error.add_note(
                    f"temporary file was retained through original run-root authority at {retained.name}"
                )
            except BaseException as retirement_error:
                primary_error.add_note(
                    f"temporary-file root-race retirement also failed: {retirement_error}"
                )
            raise
        return descriptor, created
    assert_active_run_root_identity(candidate)
    _run_root_mutation_interleave_hook(
        "before_exclusive_regular_file_create", candidate
    )
    descriptor = os.open(
        candidate,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.set_inheritable(descriptor, False)
    return descriptor, os.fstat(descriptor)


def create_exclusive_temporary_file(
    directory: Path, *, prefix: str, suffix: str = ".tmp"
) -> tuple[int, Path, os.stat_result]:
    """Allocate an unguessable exact file through retained run-root authority."""

    parent = Path(os.path.abspath(directory))
    if not prefix or any(separator in prefix for separator in ("/", "\\")):
        raise Stage6ContractError("temporary-file prefix is invalid")
    if any(separator in suffix for separator in ("/", "\\")):
        raise Stage6ContractError("temporary-file suffix is invalid")
    for _ in range(128):
        candidate = parent / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            descriptor, created = create_exclusive_regular_file(candidate)
        except FileExistsError:
            continue
        return descriptor, candidate, created
    raise Stage6ContractError("could not allocate an exclusive temporary file")


def _create_exclusive_directory_with_descriptor(
    path: Path,
) -> tuple[int | None, os.stat_result]:
    """Create a directory and retain its exact descriptor before returning."""

    candidate = Path(os.path.abspath(path))
    authority = _active_run_root_authority(candidate)
    if authority is not None and os.name != "nt":
        authority.assert_path_identity()
        _run_root_mutation_interleave_hook("before_exclusive_directory_create", candidate)
        descriptor, created = authority.mkdir_and_open(candidate)
        try:
            authority.assert_path_identity()
        except BaseException as primary_error:
            os.close(descriptor)
            retained = authority.root / f".owned-directory-root-race-retained-{uuid.uuid4().hex}"
            try:
                authority.rename_no_replace(candidate, retained)
                primary_error.add_note(
                    f"created directory was retained through original run-root authority at {retained.name}"
                )
            except BaseException as retirement_error:
                primary_error.add_note(
                    f"directory root-race retirement also failed: {retirement_error}"
                )
            raise
        return descriptor, created
    assert_active_run_root_identity(candidate)
    _run_root_mutation_interleave_hook("before_exclusive_directory_create", candidate)
    candidate.mkdir(parents=False, exist_ok=False)
    created = os.lstat(candidate)
    if os.name == "nt":
        return None, created
    descriptor = os.open(
        candidate,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    os.set_inheritable(descriptor, False)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (created.st_dev, created.st_ino):
        os.close(descriptor)
        raise Stage6ContractError("created directory identity changed before retention")
    return descriptor, opened


def create_exclusive_directory(path: Path) -> os.stat_result:
    """Create one directory entry through retained run-root authority when active."""

    descriptor, created = _create_exclusive_directory_with_descriptor(path)
    if descriptor is not None:
        os.close(descriptor)
    return created


def _register_active_run_root(authority: RetainedRunRootAuthority) -> str:
    key = os.path.normcase(str(authority.root))
    with _ACTIVE_RUN_ROOT_GUARD:
        if key in _ACTIVE_RUN_ROOT_AUTHORITIES:
            raise Stage6ContractError(
                "an active Notebook 01 materialization already owns this run"
            )
        _ACTIVE_RUN_ROOT_AUTHORITIES[key] = authority
    return key


def _unregister_active_run_root(key: str, authority: RetainedRunRootAuthority) -> None:
    with _ACTIVE_RUN_ROOT_GUARD:
        if _ACTIVE_RUN_ROOT_AUTHORITIES.get(key) is authority:
            del _ACTIVE_RUN_ROOT_AUTHORITIES[key]


@contextmanager
def exclusive_parent_run_lease(
    run_root: Path,
) -> Iterator[RetainedRunRootAuthority]:
    """Crash-releasing path lease plus retained exact run-root authority."""

    root = Path(os.path.abspath(run_root))
    key = os.path.normcase(str(root))
    with _PARENT_LEASE_GUARD:
        if key in _PARENT_LEASE_KEYS:
            raise Stage6ContractError(
                "an active Notebook 01 materialization already owns this run"
            )
        _PARENT_LEASE_KEYS.add(key)
    parent_lease_owner: _ParentLeaseRetryOwner | None = None
    windows_mutex_owner: _WindowsMutexRetryOwner | None = None
    authority: RetainedRunRootAuthority | None = None
    active_key: str | None = None
    body_error: BaseException | None = None
    try:
        if os.name == "nt":
            import ctypes

            name_digest = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()
            mutex_name = f"Local\\StackSocietyStage6Notebook01-{name_digest}"
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_mutex = kernel32.CreateMutexW
            create_mutex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
            create_mutex.restype = ctypes.c_void_p
            raw_handle = create_mutex(None, 0, mutex_name)
            if not raw_handle:
                error_number = ctypes.get_last_error()
                raise OSError(error_number, ctypes.FormatError(error_number), root)
            windows_mutex_owner = _WindowsMutexRetryOwner(int(raw_handle))
            wait = kernel32.WaitForSingleObject
            wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            wait.restype = ctypes.c_uint32
            wait_result = int(
                wait(ctypes.c_void_p(windows_mutex_owner.handle), 0)
            )
            if wait_result not in {0, 0x80}:
                raise Stage6ContractError(
                    "an active Notebook 01 materialization already owns this run"
                )
            windows_mutex_owner.mark_acquired()
        else:
            import fcntl

            parent = root.parent
            parent_before = os.lstat(parent)
            if not stat.S_ISDIR(parent_before.st_mode) or _is_reparse_point(
                parent_before
            ):
                raise Stage6ContractError(
                    f"run-root parent is not a stable directory: {parent}"
                )
            parent_descriptor = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            parent_lease_owner = _ParentLeaseRetryOwner(parent_descriptor)
            os.set_inheritable(parent_descriptor, False)
            try:
                fcntl.flock(parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Stage6ContractError(
                    "an active Notebook 01 materialization already owns this run"
                ) from error
            parent_lease_owner.mark_acquired()
            parent_after = os.fstat(parent_descriptor)
            parent_current = os.lstat(parent)
            if (
                (parent_after.st_dev, parent_after.st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
                or (parent_current.st_dev, parent_current.st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
            ):
                raise Stage6ContractError(
                    "run-root parent identity changed while lease was acquired"
                )
        authority = RetainedRunRootAuthority(root)
        authority.assert_path_identity()
        active_key = _register_active_run_root(authority)
        yield authority
    except BaseException as error:
        body_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        if authority is not None and active_key is not None:
            _unregister_active_run_root(active_key, authority)
        if authority is not None:
            try:
                authority.close()
            except BaseException as error:
                _attach_retry_owner(error, authority)
                cleanup_error = error
        if parent_lease_owner is not None:
            try:
                parent_lease_owner.close()
            except BaseException as error:
                retry_owner = getattr(error, "stage6_retry_owner", parent_lease_owner)
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    _attach_retry_owner(cleanup_error, retry_owner)
                    cleanup_error.add_note(
                        f"POSIX parent lease close also failed: {error}"
                    )
        if windows_mutex_owner is not None:
            try:
                windows_mutex_owner.close()
            except BaseException as error:
                retry_owner = getattr(error, "stage6_retry_owner", windows_mutex_owner)
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    _attach_retry_owner(cleanup_error, retry_owner)
                    cleanup_error.add_note(
                        f"Windows parent mutex close also failed: {error}"
                    )
        with _PARENT_LEASE_GUARD:
            _PARENT_LEASE_KEYS.discard(key)
        if cleanup_error is not None:
            if body_error is not None:
                retry_owner = getattr(cleanup_error, "stage6_retry_owner", None)
                if retry_owner is not None:
                    _attach_retry_owner(body_error, retry_owner)
                body_error.add_note(f"parent run lease close failed: {cleanup_error}")
            else:
                raise cleanup_error


def _owned_staging_interleave_hook(_event: str, _path: Path) -> None:
    """Intentional no-op seam for deterministic identity-race regressions."""


class _OwnedStagingAuthority:
    """Retain the exact owned directory across validation and mutation."""

    def __init__(
        self,
        run_root: Path,
        staging: Path,
        *,
        kind: str,
        expected_receipt: Mapping[str, Any],
    ) -> None:
        self.root = Path(os.path.abspath(run_root))
        self.path = Path(os.path.abspath(staging))
        self.original_path = self.path
        assert_active_run_root_identity(self.root)
        self._run_authority = (
            _active_run_root_authority(self.path) if os.name != "nt" else None
        )
        self.kind = kind
        self.receipt = dict(expected_receipt)
        self.identity = (
            int(self.receipt.get("directory_device", -1)),
            int(self.receipt.get("directory_inode", -1)),
        )
        self._windows_handle: int | None = None
        self._descriptor = -1
        self._delete_pending = False
        self.retained_evidence_path: Path | None = None
        self._retired_receipt_path: Path | None = None
        self._retired_receipt_identity: tuple[int, int] | None = None
        verify_owned_staging_directory(
            self.root,
            self.path,
            kind=kind,
            expected_receipt=self.receipt,
        )
        before = (
            self._run_authority.stat_path(self.path)
            if self._run_authority is not None
            else os.lstat(self.path)
        )
        try:
            if os.name == "nt":
                self._windows_handle = _open_windows_directory_handle(
                    self.path, deny_delete=True
                )
            elif self._run_authority is not None:
                self._descriptor = self._run_authority.open_directory_descriptor(
                    self.path
                )
            else:
                self._descriptor = os.open(
                    self.path,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                os.set_inheritable(self._descriptor, False)
            _owned_staging_interleave_hook("after_authority_open", self.path)
            after = (
                self._run_authority.stat_path(self.path)
                if self._run_authority is not None
                else os.lstat(self.path)
            )
            if self._descriptor >= 0:
                opened = os.fstat(self._descriptor)
                if (opened.st_dev, opened.st_ino) != self.identity:
                    raise Stage6ContractError(
                        "retained staging descriptor has the wrong identity"
                    )
            if (
                (before.st_dev, before.st_ino) != self.identity
                or (after.st_dev, after.st_ino) != self.identity
            ):
                raise Stage6ContractError(
                    "staging identity changed while retained authority was opened"
                )
            self.verify_receipt()
        except BaseException as primary_error:
            _close_authority_preserving_primary(
                self,
                primary_error,
                label="retained staging constructor authority",
            )
            raise

    def assert_path_identity(self) -> None:
        self._assert_object_identity(require_run_root_identity=True)

    def _assert_object_identity(self, *, require_run_root_identity: bool) -> None:
        if self._delete_pending:
            return
        current = (
            self._run_authority.stat_path(self.path)
            if self._run_authority is not None
            else os.lstat(self.path)
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_reparse_point(current)
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise Stage6ContractError(
                "staging pathname no longer names the retained owned object"
            )
        if self._descriptor >= 0:
            opened = os.fstat(self._descriptor)
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise Stage6ContractError("retained staging descriptor identity changed")
        if require_run_root_identity and self._run_authority is not None:
            self._run_authority.assert_path_identity()

    def verify_receipt(self) -> None:
        self.assert_path_identity()
        receipt_path = self.path / STAGING_OWNER_RECEIPT_NAME
        if self._descriptor >= 0:
            receipt_descriptor = os.open(
                STAGING_OWNER_RECEIPT_NAME,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._descriptor,
            )
            try:
                receipt_stat = os.fstat(receipt_descriptor)
                if not stat.S_ISREG(receipt_stat.st_mode) or _is_reparse_point(
                    receipt_stat
                ):
                    raise Stage6ContractError(
                        "retained staging ownership receipt is invalid"
                    )
                if receipt_stat.st_size > MAX_REGULAR_FILE_SNAPSHOT_BYTES:
                    raise Stage6ContractError(
                        "retained staging ownership receipt exceeds size cap"
                    )
                chunks: list[bytes] = []
                remaining = receipt_stat.st_size
                while remaining:
                    chunk = os.read(
                        receipt_descriptor,
                        min(FILE_IDENTITY_CHUNK_BYTES, remaining),
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if len(payload) != receipt_stat.st_size:
                    raise Stage6ContractError(
                        "retained staging ownership receipt changed while read"
                    )
            finally:
                os.close(receipt_descriptor)
        else:
            payload, _ = _regular_file_snapshot(receipt_path)
        observed = _strict_json_object_bytes(payload, label=str(receipt_path))
        validate_document_self_hash(observed)
        if observed != self.receipt:
            raise Stage6ContractError("retained staging ownership receipt changed")

    def flush(self) -> None:
        self.assert_path_identity()
        if self._windows_handle is not None:
            _flush_windows_handle(self._windows_handle, self.path)
        elif self._descriptor >= 0:
            os.fsync(self._descriptor)
        else:
            raise Stage6ContractError("retained staging authority is closed")
        self.assert_path_identity()

    def rename_to(
        self, destination: Path, *, require_run_root_identity: bool = True
    ) -> None:
        target = Path(os.path.abspath(destination))
        target_exists = False
        if self._run_authority is not None:
            try:
                self._run_authority.stat_path(target)
            except FileNotFoundError:
                pass
            else:
                target_exists = True
        else:
            target_exists = os.path.lexists(target)
        if target.parent != self.root or target_exists:
            raise Stage6ContractError(
                f"owned staging publish destination is invalid: {target}"
            )
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )
        _owned_staging_interleave_hook("before_no_replace_rename_source", self.path)
        _owned_staging_interleave_hook("before_no_replace_rename_destination", target)
        if require_run_root_identity:
            assert_active_run_root_identity(self.path)
            assert_active_run_root_identity(target)
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )
        if self._windows_handle is not None:
            _windows_rename_handle_no_replace(self._windows_handle, target)
            self.path = target
        else:
            source = self.path
            if self._run_authority is not None:
                self._run_authority.rename_no_replace(source, target)
            else:
                _posix_rename_no_replace(source, target)
            self.path = target
            try:
                moved = (
                    self._run_authority.stat_path(target)
                    if self._run_authority is not None
                    else os.lstat(target)
                )
                if (
                    not stat.S_ISDIR(moved.st_mode)
                    or _is_reparse_point(moved)
                    or (moved.st_dev, moved.st_ino) != self.identity
                ):
                    raise Stage6ContractError(
                        "no-replace transaction moved a foreign staging directory"
                    )
            except BaseException as primary_error:
                target_still_exists = False
                if self._run_authority is not None:
                    try:
                        self._run_authority.stat_path(target)
                    except FileNotFoundError:
                        pass
                    else:
                        target_still_exists = True
                else:
                    target_still_exists = os.path.lexists(target)
                if target_still_exists:
                    try:
                        if self._run_authority is not None:
                            self._run_authority.rename_no_replace(target, source)
                        else:
                            _posix_rename_no_replace(target, source)
                        self.path = source
                    except BaseException as restore_error:
                        primary_error.add_note(
                            f"foreign staging restoration also failed: {restore_error}"
                        )
                raise
        self._assert_object_identity(
            require_run_root_identity=require_run_root_identity
        )

    def delete(self) -> None:
        """Retire the exact staging tree without pathname-deleting descendants."""

        self.verify_receipt()
        retired = self.root / (
            f".owned-staging-retained-{self.kind}-{uuid.uuid4().hex}"
        )
        self.rename_to(retired, require_run_root_identity=False)
        self.retained_evidence_path = retired
        _owned_staging_interleave_hook("after_safe_retire", retired)
        self._assert_object_identity(require_run_root_identity=False)

    def remove_receipt(self) -> None:
        self.verify_receipt()
        receipt_path = self.path / STAGING_OWNER_RECEIPT_NAME
        _, receipt_stat = _regular_file_snapshot(receipt_path)
        expected_identity = (int(receipt_stat.st_dev), int(receipt_stat.st_ino))
        retained = self.root / (
            ".stage6-staging-owner-retained-"
            f"{self.receipt['owner_token']}-{uuid.uuid4().hex}.json"
        )
        with OwnedRegularFileAuthority(
            self.root,
            receipt_path,
            expected_identity=expected_identity,
        ) as authority:
            authority.rename_to(retained)
            retired_identity = authority.identity
        self._retired_receipt_path = retained
        self._retired_receipt_identity = retired_identity
        self.flush()

    def restore_receipt(self) -> None:
        receipt_path = self.path / STAGING_OWNER_RECEIPT_NAME
        if receipt_path.exists():
            self.verify_receipt()
            return
        if (
            self._retired_receipt_path is not None
            and self._retired_receipt_identity is not None
            and os.path.lexists(self._retired_receipt_path)
        ):
            move_verified_regular_file(
                self.root,
                self._retired_receipt_path,
                receipt_path,
                expected_identity=self._retired_receipt_identity,
            )
            self._retired_receipt_path = None
            self._retired_receipt_identity = None
            self.flush()
            self.verify_receipt()
            return
        payload = canonical_json_bytes(self.receipt) + b"\n"
        if self._descriptor >= 0:
            descriptor = os.open(
                STAGING_OWNER_RECEIPT_NAME,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with receipt_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        self.flush()

    def close(self) -> None:
        if self._descriptor >= 0:
            descriptor = self._descriptor
            os.close(descriptor)
            self._descriptor = -1
        if self._windows_handle is not None:
            handle = self._windows_handle
            _close_windows_handle(handle)
            self._windows_handle = None

    def __enter__(self) -> "_OwnedStagingAuthority":
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc_value: object,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            _attach_retry_owner(close_error, self)
            if isinstance(exc_value, BaseException):
                _attach_retry_owner(exc_value, self)
                exc_value.add_note(
                    f"retained staging authority close failed: {close_error}"
                )
            else:
                raise


def retain_owned_staging_directory(
    run_root: Path,
    staging: Path,
    *,
    kind: str,
    expected_receipt: Mapping[str, Any],
) -> _OwnedStagingAuthority:
    """Retain exact resumed staging through all child writes and final sealing."""

    authority = _OwnedStagingAuthority(
        run_root,
        staging,
        kind=kind,
        expected_receipt=expected_receipt,
    )
    try:
        if authority._run_authority is not None:
            authority._run_authority.ensure_descendant_directory_anchor(
                authority.path,
                expected_identity=authority.identity,
            )
        authority.assert_path_identity()
        authority.verify_receipt()
        return authority
    except BaseException as primary_error:
        _close_authority_preserving_primary(
            authority,
            primary_error,
            label="retained resumed-staging authority",
        )
        raise


def create_owned_staging_directory(
    run_root: Path,
    *,
    kind: str,
    run_identity: str,
    destination_name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    _validate_staging_kind_and_identity(kind, run_identity)
    root = Path(os.path.abspath(run_root))
    assert_active_run_root_identity(root)
    owner_token = uuid.uuid4().hex
    if destination_name is None:
        staging_name = f".{kind}-staging-{owner_token}"
    else:
        if destination_name != kind:
            raise Stage6ContractError(
                "direct owned-staging destination must equal its staging kind"
            )
        staging_name = destination_name
    staging = root / staging_name
    assert_active_run_root_identity(staging)
    staging_descriptor, directory_stat = _create_exclusive_directory_with_descriptor(
        staging
    )
    if not stat.S_ISDIR(directory_stat.st_mode) or _is_reparse_point(directory_stat):
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        raise Stage6ContractError(f"new staging directory is invalid: {staging}")
    directory_identity = (
        int(directory_stat.st_dev),
        int(directory_stat.st_ino),
    )
    run_authority = (
        _active_run_root_authority(staging) if os.name != "nt" else None
    )
    anchor_registered = False
    if run_authority is not None:
        if staging_descriptor is None:
            raise Stage6ContractError("POSIX staging descriptor was not retained")
        try:
            run_authority.register_descendant_directory_anchor(
                staging,
                staging_descriptor,
                expected_identity=directory_identity,
            )
        except BaseException:
            os.close(staging_descriptor)
            raise
        anchor_registered = True
        staging_descriptor = None
    receipt: dict[str, Any] = {
        "schema_version": "stage6_owned_staging_directory_v1",
        "status": "ACTIVE_TASK_OWNED_STAGING",
        "staging_kind": kind,
        "owner_token": owner_token,
        "run_identity": run_identity,
        "relative_path": staging.name,
        "directory_device": int(directory_stat.st_dev),
        "directory_inode": int(directory_stat.st_ino),
    }
    receipt["self_sha256"] = document_self_sha256(receipt)
    receipt_path = staging / STAGING_OWNER_RECEIPT_NAME
    payload = canonical_json_bytes(receipt) + b"\n"
    try:
        _owned_staging_interleave_hook(
            "after_directory_create_before_receipt", staging
        )
        if run_authority is not None:
            staging_descriptor = run_authority.open_directory_descriptor(staging)
        if staging_descriptor is not None:
            opened = os.fstat(staging_descriptor)
            current = os.lstat(staging)
            if (
                (opened.st_dev, opened.st_ino) != directory_identity
                or (current.st_dev, current.st_ino) != directory_identity
            ):
                raise Stage6ContractError(
                    "new staging identity changed before receipt creation"
                )
            receipt_descriptor = os.open(
                STAGING_OWNER_RECEIPT_NAME,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=staging_descriptor,
            )
            opened_receipt = os.fdopen(receipt_descriptor, "wb")
        else:
            opened_receipt = receipt_path.open("xb")
        with opened_receipt as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if staging_descriptor is not None:
            _fsync_posix_directory_descriptor(staging_descriptor, staging)
        else:
            fsync_directory(staging)
        fsync_directory(root)
        if staging_descriptor is not None:
            receipt_descriptor = os.open(
                STAGING_OWNER_RECEIPT_NAME,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=staging_descriptor,
            )
            try:
                receipt_stat = os.fstat(receipt_descriptor)
                if (
                    not stat.S_ISREG(receipt_stat.st_mode)
                    or _is_reparse_point(receipt_stat)
                    or receipt_stat.st_size > MAX_REGULAR_FILE_SNAPSHOT_BYTES
                ):
                    raise Stage6ContractError(
                        "staging ownership receipt durable readback is invalid"
                    )
                observed = b""
                while len(observed) < receipt_stat.st_size:
                    chunk = os.read(
                        receipt_descriptor,
                        min(
                            FILE_IDENTITY_CHUNK_BYTES,
                            receipt_stat.st_size - len(observed),
                        ),
                    )
                    if not chunk:
                        break
                    observed += chunk
            finally:
                os.close(receipt_descriptor)
        else:
            observed, _ = _regular_file_snapshot(receipt_path)
        if observed != payload:
            raise Stage6ContractError("staging ownership receipt durable readback mismatch")
        if anchor_registered and staging_descriptor is not None:
            os.close(staging_descriptor)
            staging_descriptor = None
    except BaseException as primary_error:
        retained = root / (
            f".owned-staging-constructor-retained-{kind}-{uuid.uuid4().hex}"
        )
        handle: int | None = None
        try:
            current = (
                run_authority.stat_path(staging)
                if run_authority is not None
                else os.lstat(staging)
            )
            if (current.st_dev, current.st_ino) != (
                directory_stat.st_dev,
                directory_stat.st_ino,
            ):
                raise Stage6ContractError(
                    "new staging identity changed before constructor retirement"
                )
            if os.name == "nt":
                assert_active_run_root_identity(staging)
                handle = _open_windows_directory_handle(staging, deny_delete=True)
                _windows_rename_handle_no_replace(handle, retained)
            elif run_authority is not None:
                run_authority.rename_no_replace(staging, retained)
            else:
                _posix_rename_no_replace(staging, retained)
            moved = (
                run_authority.stat_path(retained)
                if run_authority is not None
                else os.lstat(retained)
            )
            if (moved.st_dev, moved.st_ino) != (
                directory_stat.st_dev,
                directory_stat.st_ino,
            ):
                raise Stage6ContractError(
                    "constructor retirement moved a foreign staging directory"
                )
            primary_error.add_note(
                f"incomplete staging evidence was retained at {retained}"
            )
        except BaseException as retirement_error:
            primary_error.add_note(
                f"incomplete staging retirement also failed: {retirement_error}"
            )
        finally:
            if handle is not None:
                try:
                    _close_windows_handle(handle)
                except BaseException as close_error:
                    primary_error.add_note(
                        f"incomplete staging retirement handle close also failed: {close_error}"
                    )
            if staging_descriptor is not None:
                try:
                    os.close(staging_descriptor)
                    staging_descriptor = None
                except BaseException as close_error:
                    primary_error.add_note(
                        f"staging descriptor close also failed: {close_error}"
                    )
            if anchor_registered and run_authority is not None:
                try:
                    run_authority.release_descendant_directory_anchor(
                        staging, expected_identity=directory_identity
                    )
                except BaseException as close_error:
                    primary_error.add_note(
                        f"staging descriptor-anchor release also failed: {close_error}"
                    )
        raise
    if not anchor_registered and staging_descriptor is not None:
        os.close(staging_descriptor)
    return staging, receipt


def verify_owned_staging_directory(
    run_root: Path,
    staging: Path,
    *,
    kind: str,
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in STAGING_KINDS:
        raise Stage6ContractError(f"unsupported staging kind: {kind!r}")
    root = Path(os.path.abspath(run_root))
    assert_active_run_root_identity(root)
    candidate = Path(staging)
    name_match = re.fullmatch(rf"\.{re.escape(kind)}-staging-([0-9a-f]{{32}})", candidate.name)
    direct_final_name = candidate.name == kind
    if (
        (name_match is None and not direct_final_name)
        or Path(os.path.abspath(candidate.parent)) != root
    ):
        raise Stage6ContractError(f"staging path does not match exact owner grammar: {candidate}")
    try:
        directory_stat = os.lstat(candidate)
    except FileNotFoundError as error:
        raise Stage6ContractError(f"owned staging directory is missing: {candidate}") from error
    if (
        candidate.is_symlink()
        or not stat.S_ISDIR(directory_stat.st_mode)
        or _is_reparse_point(directory_stat)
        or Path(os.path.abspath(candidate.resolve().parent)) != root
    ):
        raise Stage6ContractError(f"owned staging directory identity is unsafe: {candidate}")
    receipt_path = candidate / STAGING_OWNER_RECEIPT_NAME
    payload, _ = _regular_file_snapshot(receipt_path)
    receipt = _strict_json_object_bytes(payload, label=str(receipt_path))
    if set(receipt) != STAGING_RECEIPT_FIELDS:
        raise Stage6ContractError("staging ownership receipt schema is invalid")
    validate_document_self_hash(receipt)
    if (
        receipt.get("schema_version") != "stage6_owned_staging_directory_v1"
        or receipt.get("status") != "ACTIVE_TASK_OWNED_STAGING"
        or receipt.get("staging_kind") != kind
        or type(receipt.get("owner_token")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", receipt["owner_token"]) is None
        or (
            name_match is not None
            and receipt.get("owner_token") != name_match.group(1)
        )
        or receipt.get("relative_path") != candidate.name
        or receipt.get("directory_device") != int(directory_stat.st_dev)
        or receipt.get("directory_inode") != int(directory_stat.st_ino)
        or type(receipt.get("run_identity")) is not str
        or SHA256_PATTERN.fullmatch(receipt["run_identity"]) is None
    ):
        raise Stage6ContractError("staging ownership receipt binding is invalid")
    if expected_receipt is not None and dict(expected_receipt) != receipt:
        raise Stage6ContractError("staging pathname no longer names the retained owned object")
    return receipt


def remove_owned_staging_directory(
    run_root: Path,
    staging: Path,
    *,
    kind: str,
    expected_receipt: Mapping[str, Any] | None = None,
) -> None:
    if expected_receipt is None:
        raise Stage6ContractError(
            "owned staging removal requires the expected run/owner receipt"
        )
    root = Path(os.path.abspath(run_root))
    assert_active_run_root_identity(root)
    authority = _OwnedStagingAuthority(
        root,
        staging,
        kind=kind,
        expected_receipt=expected_receipt,
    )
    with authority:
        _owned_staging_interleave_hook("before_delete", Path(staging))
        authority.verify_receipt()
        authority.delete()
    run_authority = (
        _active_run_root_authority(root) if os.name != "nt" else None
    )
    if run_authority is not None:
        run_authority.release_descendant_directory_anchor(
            staging,
            expected_identity=(
                int(expected_receipt["directory_device"]),
                int(expected_receipt["directory_inode"]),
            ),
        )
    if os.path.lexists(staging):
        raise Stage6ContractError("owned staging deletion did not remove its pathname")
    fsync_directory(root)


def remove_current_run_stale_staging_directories(
    run_root: Path, *, kind: str, run_identity: str
) -> dict[str, int]:
    """Delete only valid staging owned by this exact run; preserve every other run."""

    _validate_staging_kind_and_identity(kind, run_identity)
    root = Path(os.path.abspath(run_root))
    assert_active_run_root_identity(root)
    removed = 0
    preserved_other_run = 0
    for staging in sorted(root.glob(f".{kind}-staging-*")):
        observed = verify_owned_staging_directory(root, staging, kind=kind)
        if observed["run_identity"] != run_identity:
            preserved_other_run += 1
            continue
        remove_owned_staging_directory(
            root,
            staging,
            kind=kind,
            expected_receipt=observed,
        )
        removed += 1
    return {
        "removed_current_run_staging_count": removed,
        "preserved_other_run_staging_count": preserved_other_run,
    }


def publish_owned_staging_directory(
    run_root: Path,
    staging: Path,
    destination: Path,
    *,
    kind: str,
    expected_receipt: Mapping[str, Any],
    retained_authority: _OwnedStagingAuthority | None = None,
) -> None:
    root = Path(os.path.abspath(run_root))
    assert_active_run_root_identity(root)
    source = Path(os.path.abspath(staging))
    target = Path(os.path.abspath(destination))
    in_place = source == target
    if target.parent.resolve() != root or (os.path.lexists(target) and not in_place):
        raise Stage6ContractError(f"owned staging publish destination is invalid: {target}")
    if retained_authority is None:
        authority = retain_owned_staging_directory(
            root,
            staging,
            kind=kind,
            expected_receipt=expected_receipt,
        )
    else:
        authority = retained_authority
        if (
            authority.root != root
            or authority.path != source
            or authority.kind != kind
            or authority.receipt != dict(expected_receipt)
        ):
            raise Stage6ContractError(
                "retained staging publication authority binding is invalid"
            )
        authority.assert_path_identity()
        authority.verify_receipt()
    publication_succeeded = False
    with authority:
        published = False
        renamed = False
        receipt_removed = False
        try:
            authority.flush()
            _owned_staging_interleave_hook("before_publish", Path(staging))
            authority.verify_receipt()
            if not in_place:
                authority.rename_to(target)
                renamed = True
            published = True
            fsync_directory(root)
            authority.verify_receipt()
            authority.remove_receipt()
            receipt_removed = True
            fsync_directory(root)
            authority.assert_path_identity()
            publication_succeeded = True
        except BaseException as primary_error:
            if published and receipt_removed:
                try:
                    authority.restore_receipt()
                except BaseException as restore_error:
                    primary_error.add_note(
                        f"staging receipt restoration also failed: {restore_error}"
                    )
            if published and renamed:
                rollback = Path(staging)
                if os.path.lexists(rollback):
                    rollback = root / (
                        f".{kind}-recovery-{expected_receipt['owner_token']}"
                    )
                try:
                    authority.rename_to(rollback)
                    fsync_directory(root)
                except BaseException as rollback_error:
                    primary_error.add_note(
                        "identity-bound staging publication rollback also failed: "
                        f"{rollback_error}"
                    )
            raise
    if publication_succeeded:
        run_authority = (
            _active_run_root_authority(root) if os.name != "nt" else None
        )
        if run_authority is not None:
            run_authority.release_descendant_directory_anchor(
                staging,
                expected_identity=(
                    int(expected_receipt["directory_device"]),
                    int(expected_receipt["directory_inode"]),
                ),
            )
