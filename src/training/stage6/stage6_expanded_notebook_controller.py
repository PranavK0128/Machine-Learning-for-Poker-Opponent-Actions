"""Shared deterministic returned-result publisher for Stage 6 notebooks 12-20."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

from stage6_expanded_layers import canonical_json_bytes, seal_document, validate_document
from stage6_notebooks02_07_runtime import sha256_file


def _fixed_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)


def publish_compact_result(
    stage_drive_root: Path,
    *,
    token: str,
    run_identity: str,
    result: Mapping[str, Any],
    primary_manifest_path: Path,
) -> dict[str, Any]:
    if not primary_manifest_path.is_file():
        raise RuntimeError(f"Notebook {token} primary manifest is missing")
    primary_raw = primary_manifest_path.read_bytes()
    primary = __import__("json").loads(primary_raw)
    validate_document(primary)
    receipt = seal_document({
        "schema_version": "stage6_expanded_notebook_controller_receipt_v1",
        "notebook_token": token,
        "run_identity": run_identity,
        "primary_manifest_name": primary_manifest_path.name,
        "primary_manifest_bytes": len(primary_raw),
        "primary_manifest_sha256": hashlib.sha256(primary_raw).hexdigest(),
        "primary_manifest_self_sha256": primary["self_sha256"],
        "result": dict(result),
        "action_timing_layer_included": False,
        "qualification_authority": bool(
            primary.get("status") == "PASS_FRESH_ADJUDICATION_COMPLETE"
        ),
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    members = [
        ("primary_manifest.json", primary_raw),
        ("controller_receipt.json", canonical_json_bytes(receipt) + b"\n"),
    ]
    result_manifest = seal_document({
        "schema_version": "stage6_expanded_notebook_result_manifest_v1",
        "notebook_token": token,
        "run_identity": run_identity,
        "member_order": [name for name, _ in members],
        "members": [
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in members
        ],
        "action_timing_layer_included": False,
        "recommendation_authority": False,
        "action_authority": False,
        "live_action_authority": False,
    })
    members.append(("RESULT_MANIFEST.json", canonical_json_bytes(result_manifest) + b"\n"))
    with tempfile.TemporaryDirectory(prefix=f"stage6-notebook{token}-result-") as temporary:
        local = Path(temporary) / "result.zip"
        _fixed_zip(local, members)
        digest = sha256_file(local)
        name = f"{token}_Stage6_ExpandedLayers_{run_identity[:16]}_{digest[:32]}.zip"
        destination = stage_drive_root / "returned_results" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != local.stat().st_size or sha256_file(destination) != digest:
                raise RuntimeError("expanded notebook returned-result destination conflicts")
        else:
            shutil.copyfile(local, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError("expanded notebook returned-result publication changed bytes")
    return {
        "name": destination.name,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "result_manifest_self_sha256": result_manifest["self_sha256"],
    }

