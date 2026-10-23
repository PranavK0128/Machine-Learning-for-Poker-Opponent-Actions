from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping


class Stage6ContractError(ValueError):
    """Raised when a Stage 6 fail-closed contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def document_self_sha256(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("self_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def validate_document_self_hash(document: Mapping[str, Any]) -> str:
    claimed = document.get("self_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise Stage6ContractError("document self_sha256 is missing or not frozen")
    observed = document_self_sha256(document)
    if observed != claimed:
        raise Stage6ContractError(
            f"document self hash mismatch: expected {claimed}, observed {observed}"
        )
    return observed


def sha256_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Stage6ContractError(f"{label} must be a strict numeric value")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise Stage6ContractError(f"{label} must be finite and nonnegative")
    return numeric


def parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or not value or value != value.strip():
        raise Stage6ContractError(f"{label} must be canonical timezone-aware UTC text")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise Stage6ContractError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage6ContractError(f"{label} is timezone-naive")
    if parsed.utcoffset() != timedelta(0):
        raise Stage6ContractError(f"{label} is not expressed in UTC")
    return parsed.astimezone(timezone.utc)
