from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from stage6_common import (
    Stage6ContractError,
    document_self_sha256,
    sha256_file,
)
from stage6_decision_row_builder import (
    ALLOWED_SOURCE_QUARANTINE_REASONS,
    classify_source_quarantine_reason,
)


ACTION_RELATIVE = Path("data/exports/ml/action-state-decision-rows.csv")
CLASSIFIER_RELATIVE = Path("src/stage6_decision_row_builder.py")
SCANNER_RELATIVE = Path("src/stage6_source_quarantine_scan.py")
EXPECTED_SOURCE_ROWS = 2_334_700
PROGRESS_INTERVAL_ROWS = 250_000
ROW_ID_FIELD = "hand_action_id"
ACCEPTED_ROOT_ALGORITHM = "sha256_uint64_source_ordinal_uint32_utf8_row_id_v1"
QUARANTINE_ROOT_ALGORITHM = (
    "sha256_uint64_source_ordinal_uint16_utf8_reason_uint32_utf8_row_id_v1"
)


class _HashingRawReader(io.RawIOBase):
    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        payload = self._source.read(len(buffer))
        count = len(payload)
        if count:
            buffer[:count] = payload
            self.digest.update(payload)
            self.bytes_read += count
        return count


def _update_accepted_root(digest: Any, ordinal: int, row_id: str) -> None:
    encoded = row_id.encode("utf-8")
    digest.update(ordinal.to_bytes(8, "big", signed=False))
    digest.update(len(encoded).to_bytes(4, "big", signed=False))
    digest.update(encoded)


def _update_quarantine_root(
    digest: Any, ordinal: int, reason: str, row_id: str
) -> None:
    reason_bytes = reason.encode("utf-8")
    row_id_bytes = row_id.encode("utf-8")
    digest.update(ordinal.to_bytes(8, "big", signed=False))
    digest.update(len(reason_bytes).to_bytes(2, "big", signed=False))
    digest.update(reason_bytes)
    digest.update(len(row_id_bytes).to_bytes(4, "big", signed=False))
    digest.update(row_id_bytes)


def _validate_regular_file_snapshot(
    path: Path, before: os.stat_result, after: os.stat_result
) -> os.stat_result:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if not stat.S_ISREG(before.st_mode) or any(
        getattr(before, field) != getattr(after, field) for field in fields
    ):
        raise Stage6ContractError(f"source changed during quarantine scan: {path}")
    current = os.lstat(path)
    if not stat.S_ISREG(current.st_mode) or any(
        getattr(after, field) != getattr(current, field) for field in fields
    ):
        raise Stage6ContractError(f"source pathname identity changed after scan: {path}")
    return current


def scan_source(stage6_root: Path) -> dict[str, Any]:
    root = stage6_root.resolve()
    if "OneDrive" in str(root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    source_path = root / ACTION_RELATIVE
    classifier_path = root / CLASSIFIER_RELATIVE
    scanner_path = root / SCANNER_RELATIVE
    before_path = os.lstat(source_path)
    accepted_digest = hashlib.sha256()
    combined_quarantine_digest = hashlib.sha256()
    reason_digests = {
        reason: hashlib.sha256() for reason in sorted(ALLOWED_SOURCE_QUARANTINE_REASONS)
    }
    reason_counts: Counter[str] = Counter()
    accepted_count = 0
    quarantined_count = 0
    row_count = 0
    started = time.perf_counter()

    with source_path.open("rb", buffering=0) as binary_source:
        before_handle = os.fstat(binary_source.fileno())
        raw_reader = _HashingRawReader(binary_source)
        with io.BufferedReader(raw_reader, buffer_size=1024 * 1024) as buffered:
            with io.TextIOWrapper(
                buffered,
                encoding="utf-8-sig",
                errors="strict",
                newline="",
            ) as text_source:
                reader = csv.DictReader(text_source)
                if not reader.fieldnames or ROW_ID_FIELD not in reader.fieldnames:
                    raise Stage6ContractError(
                        f"source is missing required row id field: {ROW_ID_FIELD}"
                    )
                for ordinal, row in enumerate(reader):
                    row_id = row.get(ROW_ID_FIELD)
                    if type(row_id) is not str or not row_id:
                        raise Stage6ContractError(
                            f"source row {ordinal} has an invalid {ROW_ID_FIELD}"
                        )
                    reason = classify_source_quarantine_reason(row)
                    if reason is None:
                        accepted_count += 1
                        _update_accepted_root(accepted_digest, ordinal, row_id)
                    else:
                        if reason not in reason_digests:
                            raise Stage6ContractError(
                                f"source row {ordinal} has an unregistered reason: {reason}"
                            )
                        quarantined_count += 1
                        reason_counts[reason] += 1
                        _update_accepted_root(reason_digests[reason], ordinal, row_id)
                        _update_quarantine_root(
                            combined_quarantine_digest, ordinal, reason, row_id
                        )
                    row_count += 1
                    if row_count % PROGRESS_INTERVAL_ROWS == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            json.dumps(
                                {
                                    "event": "stage6_source_quarantine_scan_progress",
                                    "completed": row_count,
                                    "total": EXPECTED_SOURCE_ROWS,
                                    "throughput_rows_per_second": row_count / elapsed,
                                    "eta_seconds": (EXPECTED_SOURCE_ROWS - row_count)
                                    / (row_count / elapsed),
                                },
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
        after_handle = os.fstat(binary_source.fileno())

    current = _validate_regular_file_snapshot(source_path, before_handle, after_handle)
    if any(
        getattr(before_path, field) != getattr(current, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    ):
        raise Stage6ContractError("source identity changed between preflight and scan")
    if raw_reader.bytes_read != current.st_size:
        raise Stage6ContractError("source byte count was not conserved during scan")
    if row_count != EXPECTED_SOURCE_ROWS:
        raise Stage6ContractError(
            f"source row count changed: expected {EXPECTED_SOURCE_ROWS}, observed {row_count}"
        )
    if accepted_count + quarantined_count != row_count:
        raise Stage6ContractError("accepted/quarantined source work units were not conserved")

    per_reason = {
        reason: {
            "row_count": reason_counts[reason],
            "row_identity_root": reason_digests[reason].hexdigest(),
        }
        for reason in sorted(reason_digests)
    }
    report: dict[str, Any] = {
        "schema_version": "stage6_source_quarantine_scan_v1",
        "status": "PASS_EXACT_PRODUCTION_CLASSIFIER_SOURCE_CONSERVATION",
        "source_relative_path": ACTION_RELATIVE.as_posix(),
        "source_bytes": raw_reader.bytes_read,
        "source_sha256": raw_reader.digest.hexdigest(),
        "source_row_count": row_count,
        "classifier_relative_path": CLASSIFIER_RELATIVE.as_posix(),
        "classifier_sha256": sha256_file(classifier_path),
        "classifier_function": "classify_source_quarantine_reason",
        "scanner_relative_path": SCANNER_RELATIVE.as_posix(),
        "scanner_sha256": sha256_file(scanner_path),
        "row_id_field": ROW_ID_FIELD,
        "accepted_decision_row_count": accepted_count,
        "quarantined_decision_row_count": quarantined_count,
        "accepted_row_identity_root_algorithm": ACCEPTED_ROOT_ALGORITHM,
        "accepted_row_identity_root": accepted_digest.hexdigest(),
        "quarantine_row_identity_root_algorithm": QUARANTINE_ROOT_ALGORITHM,
        "quarantine_row_identity_root": combined_quarantine_digest.hexdigest(),
        "quarantine_reasons": per_reason,
        "source_work_unit_conservation_verified": True,
        "reason_counts_sum_to_quarantine_count": sum(reason_counts.values())
        == quarantined_count,
        "runtime_authority": False,
        "scientific_authority": False,
        "promotion_authority": False,
        "drive_publication_authority": False,
        "live_action_authority": False,
    }
    report["self_sha256"] = document_self_sha256(report)
    return report


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the frozen Stage 6 decision source with the production quarantine classifier."
    )
    parser.add_argument("--stage6-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    print(json.dumps(scan_source(arguments.stage6_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
