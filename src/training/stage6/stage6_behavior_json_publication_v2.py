"""Crash-safe create-exclusive JSON publication for behavior-v2 runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from stage6_expanded_layers import canonical_json_bytes, seal_document, validate_document


def _active_claim(path: Path) -> Path:
    return path.with_name(f".{path.name}.stage6-behavior-json-claim")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("behavior JSON artifact is outside its bound root") from error


def _claim_document(
    path: Path,
    root: Path,
    raw: bytes,
    publication_binding: Mapping[str, Any],
) -> dict[str, Any]:
    document = json.loads(raw.decode("utf-8"))
    validate_document(document)
    validate_document(publication_binding)
    return seal_document({
        "schema_version": "stage6_behavior_json_publication_claim_v2",
        "publication_binding": dict(publication_binding),
        "artifact_relative_path": _relative(path, root),
        "active_claim_relative_path": _relative(_active_claim(path), root),
        "expected_bytes": len(raw),
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "document_self_sha256": document["self_sha256"],
        "authority": "same_binding_same_artifact_exact_bytes_recovery_only",
    })


def _read_claim(
    path: Path,
    root: Path,
    publication_binding: Mapping[str, Any],
    expected_raw: bytes | None = None,
) -> tuple[Path, bytes, dict[str, Any]] | None:
    directory = _active_claim(path)
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise RuntimeError("conflicting behavior JSON active-claim path")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != {"claim.json", "staged.json"}:
        raise RuntimeError("behavior JSON active-claim inventory is invalid")
    claim_raw = entries["claim.json"].read_bytes()
    staged_raw = entries["staged.json"].read_bytes()
    try:
        claim = json.loads(claim_raw.decode("utf-8"))
        staged = json.loads(staged_raw.decode("utf-8"))
        validate_document(claim)
        validate_document(staged)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError("behavior JSON active-claim bytes are invalid") from error
    if (
        canonical_json_bytes(claim) + b"\n" != claim_raw
        or canonical_json_bytes(staged) + b"\n" != staged_raw
        or claim != _claim_document(path, root, staged_raw, publication_binding)
        or (expected_raw is not None and staged_raw != expected_raw)
    ):
        raise RuntimeError("behavior JSON active-claim identity changed")
    return entries["staged.json"], staged_raw, claim


def _cleanup_prefix(path: Path, claim: Mapping[str, Any]) -> str:
    return f"{_active_claim(path).name}.cleanup-{claim['self_sha256'][:24]}-"


def _after_cleanup_event(event: str, tombstone: Path) -> None:
    """Crash-injection seam."""


def _cleanup_tombstone(
    tombstone: Path,
    claim_raw: bytes,
    staged_raw: bytes,
) -> bool:
    if not tombstone.exists():
        return True
    if not tombstone.is_dir():
        return False
    entries = {entry.name: entry for entry in tombstone.iterdir()}
    if not set(entries).issubset({"claim.json", "staged.json"}):
        return False
    expected = {"claim.json": claim_raw, "staged.json": staged_raw}
    if any(entry.read_bytes() != expected[name] for name, entry in entries.items()):
        return False
    for name in ("claim.json", "staged.json"):
        member = tombstone / name
        if member.exists():
            member.unlink()
            _after_cleanup_event(f"after_{name}_delete", tombstone)
    try:
        tombstone.rmdir()
    except OSError:
        return False
    return True


def _resume_cleanup(
    path: Path,
    root: Path,
    raw: bytes,
    publication_binding: Mapping[str, Any],
) -> None:
    claim = _claim_document(path, root, raw, publication_binding)
    claim_raw = canonical_json_bytes(claim) + b"\n"
    for tombstone in sorted(path.parent.glob(f"{_cleanup_prefix(path, claim)}*")):
        _cleanup_tombstone(tombstone, claim_raw, raw)


def _retire_claim(
    path: Path,
    root: Path,
    raw: bytes,
    publication_binding: Mapping[str, Any],
) -> None:
    observed = _read_claim(path, root, publication_binding, raw)
    if observed is None:
        _resume_cleanup(path, root, raw, publication_binding)
        return
    if not path.is_file() or path.read_bytes() != raw:
        raise RuntimeError("behavior JSON final changed before claim retirement")
    claim = observed[2]
    directory = _active_claim(path)
    tombstone = directory.with_name(
        _cleanup_prefix(path, claim) + uuid.uuid4().hex
    )
    directory.rename(tombstone)
    _after_cleanup_event("after_cleanup_rename", tombstone)
    _cleanup_tombstone(
        tombstone,
        canonical_json_bytes(claim) + b"\n",
        raw,
    )


def _ensure_claim(
    path: Path,
    root: Path,
    raw: bytes,
    publication_binding: Mapping[str, Any],
) -> Path:
    observed = _read_claim(path, root, publication_binding, raw)
    if observed is not None:
        return observed[0]
    claim = _claim_document(path, root, raw, publication_binding)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{path.name}.{claim['self_sha256'][:16]}.",
        suffix=".claim-staging",
        dir=path.parent,
    ))
    try:
        for name, payload in (
            ("staged.json", raw),
            ("claim.json", canonical_json_bytes(claim) + b"\n"),
        ):
            with (temporary / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            temporary.rename(_active_claim(path))
        except OSError:
            if not _active_claim(path).exists():
                raise
            for member in temporary.iterdir():
                member.unlink()
            temporary.rmdir()
        observed = _read_claim(path, root, publication_binding, raw)
        if observed is None:
            raise RuntimeError("behavior JSON active claim publication failed")
        return observed[0]
    finally:
        if temporary.exists():
            for member in temporary.iterdir():
                member.unlink()
            temporary.rmdir()


def _publish_staging(staging: Path, destination: Path) -> None:
    """Create final bytes exclusively; separated for crash injection."""
    with staging.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def recover_json(
    path: Path,
    *,
    root: Path,
    publication_binding: Mapping[str, Any],
) -> bool:
    observed = _read_claim(path, root, publication_binding)
    if observed is None:
        if path.is_file():
            try:
                _resume_cleanup(
                    path, root, path.read_bytes(), publication_binding
                )
            except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError):
                pass
        return False
    staging, raw, _ = observed
    if path.exists():
        current = path.read_bytes()
        if current == raw:
            _retire_claim(path, root, raw, publication_binding)
            return True
        if len(current) >= len(raw) or not raw.startswith(current):
            raise RuntimeError("foreign bytes block behavior JSON recovery")
        path.unlink()
    _publish_staging(staging, path)
    if path.read_bytes() != raw:
        raise RuntimeError("behavior JSON publication readback failed")
    _retire_claim(path, root, raw, publication_binding)
    return True


def write_json_exclusive(
    path: Path,
    document: Mapping[str, Any],
    *,
    root: Path,
    publication_binding: Mapping[str, Any],
) -> None:
    validate_document(document)
    validate_document(publication_binding)
    raw = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current == raw:
            _retire_claim(path, root, raw, publication_binding)
            return
        observed = _read_claim(path, root, publication_binding, raw)
        if observed is None or len(current) >= len(raw) or not raw.startswith(current):
            raise RuntimeError("conflicting behavior JSON final exists")
        path.unlink()
    staging = _ensure_claim(path, root, raw, publication_binding)
    _publish_staging(staging, path)
    if path.read_bytes() != raw:
        raise RuntimeError("behavior JSON publication readback failed")
    _retire_claim(path, root, raw, publication_binding)


def load_json_with_recovery(
    path: Path,
    *,
    root: Path,
    publication_binding: Mapping[str, Any],
) -> dict[str, Any]:
    recover_json(path, root=root, publication_binding=publication_binding)
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        validate_document(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError("behavior JSON terminal is invalid or unowned") from error
    if canonical_json_bytes(document) + b"\n" != raw:
        raise RuntimeError("behavior JSON terminal serialization changed")
    return document


__all__ = ["load_json_with_recovery", "recover_json", "write_json_exclusive"]
