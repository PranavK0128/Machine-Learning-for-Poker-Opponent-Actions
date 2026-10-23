"""Drive-FUSE-safe create-exclusive publication for exact Stage 6 bytes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_line(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )


def _object_state(path: Path, expected: bytes, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is foreign: {path}")
    observed = path.read_bytes()
    if observed == expected:
        return "exact"
    if len(observed) < len(expected) and observed == expected[: len(observed)]:
        return "authentic_interrupted_prefix"
    return "foreign"


def _unlink_owned(
    path: Path,
    expected: bytes,
    *,
    label: str,
    allowed_states: set[str],
) -> None:
    state = _object_state(path, expected, label=label)
    if state not in allowed_states:
        raise RuntimeError(f"{label} bytes are foreign and were preserved: {path}")
    if _object_state(path, expected, label=label) != state:
        raise RuntimeError(f"{label} changed during authenticated cleanup: {path}")
    path.unlink()


def _owner(path: Path, raw: bytes, owner_run_id: str) -> dict[str, object]:
    return {
        "schema_version": "stage6_drive_fuse_publication_claim_v1",
        "owner_run_id": owner_run_id,
        "artifact_resolved_path_sha256": _sha256(
            str(path.resolve()).encode("utf-8")
        ),
        "artifact_name": path.name,
        "expected_bytes": len(raw),
        "expected_sha256": _sha256(raw),
        "namespace_primitive": "create_exclusive_directory_and_file",
    }


def _claim_path(path: Path, owner: dict[str, object]) -> Path:
    claim_id = _sha256(_canonical_line(owner))
    return path.with_name(f".{path.name}.stage6-publication-claim-{claim_id}")


def _ensure_claim(
    path: Path, raw: bytes, owner_run_id: str
) -> tuple[Path, dict[str, object], bytes]:
    owner = _owner(path, raw, owner_run_id)
    claim = _claim_path(path, owner)
    prefix = f".{path.name}.stage6-publication-claim-"
    foreign = sorted(candidate for candidate in path.parent.glob(f"{prefix}*") if candidate != claim)
    if foreign:
        raise FileExistsError(f"foreign publication claim exists: {foreign[0]}")
    try:
        claim.mkdir(exist_ok=False)
    except FileExistsError:
        if claim.is_symlink() or not claim.is_dir():
            raise RuntimeError(f"publication claim is foreign: {claim}")
    allowed = {"CLAIM.json", "ARTIFACT.stage6-staging"}
    unexpected = sorted(child.name for child in claim.iterdir() if child.name not in allowed)
    if unexpected:
        raise RuntimeError(f"publication claim contains foreign bytes: {unexpected}")
    receipt = claim / "CLAIM.json"
    receipt_raw = _canonical_line(owner)
    if receipt.exists():
        state = _object_state(receipt, receipt_raw, label="publication claim receipt")
        if state == "foreign":
            raise RuntimeError(
                f"publication claim receipt bytes are foreign and were preserved: {receipt}"
            )
        if state == "authentic_interrupted_prefix":
            _unlink_owned(
                receipt,
                receipt_raw,
                label="publication claim receipt",
                allowed_states={"authentic_interrupted_prefix"},
            )
    if not receipt.exists():
        with receipt.open("xb") as target:
            target.write(receipt_raw)
            target.flush()
            os.fsync(target.fileno())
    if receipt.read_bytes() != receipt_raw:
        raise RuntimeError(f"publication claim receipt readback failed: {receipt}")
    return claim, owner, receipt_raw


def _cleanup_claim(
    claim: Path, receipt_raw: bytes, raw: bytes
) -> None:
    staging = claim / "ARTIFACT.stage6-staging"
    receipt = claim / "CLAIM.json"
    if staging.exists():
        _unlink_owned(
            staging,
            raw,
            label="publication staging",
            allowed_states={"exact", "authentic_interrupted_prefix"},
        )
    if receipt.exists():
        _unlink_owned(
            receipt,
            receipt_raw,
            label="publication claim receipt",
            allowed_states={"exact"},
        )
    claim.rmdir()


def create_or_verify_bytes(
    path: Path, raw: bytes, *, owner_run_id: str
) -> str:
    """Create exact bytes without hard links/renames, or verify exact reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = _owner(path, raw, owner_run_id)
    claim = _claim_path(path, owner)
    if path.exists() and _object_state(path, raw, label="publication final") == "exact":
        if claim.exists():
            claim, _, receipt_raw = _ensure_claim(path, raw, owner_run_id)
            _cleanup_claim(claim, receipt_raw, raw)
        return "reused_exact"

    claim, _, receipt_raw = _ensure_claim(path, raw, owner_run_id)
    if path.exists():
        state = _object_state(path, raw, label="publication final")
        if state != "authentic_interrupted_prefix":
            raise RuntimeError(f"conflicting publication final exists: {path}")
        _unlink_owned(
            path,
            raw,
            label="publication final",
            allowed_states={"authentic_interrupted_prefix"},
        )

    staging = claim / "ARTIFACT.stage6-staging"
    if staging.exists():
        state = _object_state(staging, raw, label="publication staging")
        if state == "foreign":
            raise RuntimeError(
                f"publication staging bytes are foreign and were preserved: {staging}"
            )
        if state == "authentic_interrupted_prefix":
            _unlink_owned(
                staging,
                raw,
                label="publication staging",
                allowed_states={"authentic_interrupted_prefix"},
            )
    if not staging.exists():
        with staging.open("xb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
    if staging.read_bytes() != raw:
        raise RuntimeError(f"publication staging readback failed: {staging}")

    try:
        with staging.open("rb") as source, path.open("xb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    except FileExistsError:
        if _object_state(path, raw, label="publication final") != "exact":
            raise
    if _object_state(path, raw, label="publication final") != "exact":
        raise RuntimeError(f"publication final readback failed: {path}")
    _cleanup_claim(claim, receipt_raw, raw)
    return "created"


def recover_claimed_bytes(path: Path) -> str | None:
    """Finish one authenticated interrupted publication before a read/resume."""
    prefix = f".{path.name}.stage6-publication-claim-"
    claims = sorted(path.parent.glob(f"{prefix}*"))
    if not claims:
        return None
    if len(claims) != 1:
        raise FileExistsError(f"multiple publication claims exist: {claims}")
    claim = claims[0]
    if claim.is_symlink() or not claim.is_dir():
        raise RuntimeError(f"publication claim is foreign: {claim}")
    claim_id = claim.name[len(prefix) :]
    if (
        len(claim_id) != 64
        or any(character not in "0123456789abcdef" for character in claim_id)
    ):
        raise RuntimeError(f"publication claim identity is foreign: {claim}")
    allowed = {"CLAIM.json", "ARTIFACT.stage6-staging"}
    unexpected = sorted(child.name for child in claim.iterdir() if child.name not in allowed)
    if unexpected:
        raise RuntimeError(f"publication claim contains foreign bytes: {unexpected}")

    receipt = claim / "CLAIM.json"
    if not receipt.exists():
        # Cleanup removes authenticated payload bytes before the claim directory.
        # If a crash lands in that final window, only an empty, correctly-shaped
        # claim beside an already-published regular file can remain.  Removing the
        # empty namespace is safe; the caller still validates the final payload.
        if any(claim.iterdir()) or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"publication claim receipt is missing or foreign: {receipt}")
        claim.rmdir()
        return "reused_exact"
    if receipt.is_symlink() or not receipt.is_file():
        raise RuntimeError(f"publication claim receipt is missing or foreign: {receipt}")
    receipt_raw = receipt.read_bytes()
    try:
        owner = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"publication claim receipt is not canonical JSON: {receipt}") from error
    if not isinstance(owner, dict) or _canonical_line(owner) != receipt_raw:
        raise RuntimeError(f"publication claim receipt is not canonical: {receipt}")
    expected_owner = {
        "schema_version": "stage6_drive_fuse_publication_claim_v1",
        "owner_run_id": owner.get("owner_run_id"),
        "artifact_resolved_path_sha256": _sha256(str(path.resolve()).encode("utf-8")),
        "artifact_name": path.name,
        "expected_bytes": owner.get("expected_bytes"),
        "expected_sha256": owner.get("expected_sha256"),
        "namespace_primitive": "create_exclusive_directory_and_file",
    }
    if owner != expected_owner or not isinstance(owner.get("owner_run_id"), str):
        raise RuntimeError(f"publication claim receipt identity is foreign: {receipt}")
    if not isinstance(owner.get("expected_bytes"), int) or int(owner["expected_bytes"]) < 0:
        raise RuntimeError(f"publication claim expected byte count is invalid: {receipt}")
    expected_sha256 = owner.get("expected_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError(f"publication claim expected digest is invalid: {receipt}")
    if claim != _claim_path(path, owner):
        raise RuntimeError(f"publication claim directory identity is foreign: {claim}")

    staging = claim / "ARTIFACT.stage6-staging"
    if not staging.exists():
        # Drive FUSE can durably expose the authenticated claim receipt before
        # either payload namespace entry becomes visible.  Preserve that receipt
        # and let the caller recompute the deterministic artifact.  The normal
        # create path will reuse this claim only when the recomputed run, path,
        # byte count, and digest reproduce the exact recorded owner identity.
        if not path.exists() and not path.is_symlink():
            return "retry_owned_claim"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"publication staging is unavailable for recovery: {staging}")
        final_raw = path.read_bytes()
        if (
            len(final_raw) != owner["expected_bytes"]
            or _sha256(final_raw) != expected_sha256
        ):
            raise RuntimeError(f"publication staging is unavailable for recovery: {staging}")
        _cleanup_claim(claim, receipt_raw, final_raw)
        return "reused_exact"
    if staging.is_symlink() or not staging.is_file():
        raise RuntimeError(f"publication staging is unavailable for recovery: {staging}")
    raw = staging.read_bytes()
    if len(raw) != owner["expected_bytes"] or _sha256(raw) != expected_sha256:
        raise RuntimeError(f"publication staging bytes are foreign and were preserved: {staging}")

    if path.exists():
        state = _object_state(path, raw, label="publication final")
        if state == "exact":
            _cleanup_claim(claim, receipt_raw, raw)
            return "reused_exact"
        if state != "authentic_interrupted_prefix":
            raise RuntimeError(f"conflicting publication final exists: {path}")
        _unlink_owned(
            path,
            raw,
            label="publication final",
            allowed_states={"authentic_interrupted_prefix"},
        )
    try:
        with staging.open("rb") as source, path.open("xb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    except FileExistsError:
        if _object_state(path, raw, label="publication final") != "exact":
            raise
    if _object_state(path, raw, label="publication final") != "exact":
        raise RuntimeError(f"publication final recovery readback failed: {path}")
    _cleanup_claim(claim, receipt_raw, raw)
    return "recovered"


__all__ = ["create_or_verify_bytes", "recover_claimed_bytes"]
