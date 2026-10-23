from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .common import (
    canonical_json_bytes,
    read_self_hashed_json_streamed_mapping,
)


class DiskPolicyBuilder:
    """Write canonical policy rows to one bounded local SQLite mapping."""

    def __init__(
        self,
        path: Path,
        *,
        require_sorted_input: bool = True,
        insert_batch_rows: int = 2048,
    ) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            "CREATE TABLE policy (policy_key TEXT PRIMARY KEY, row_json TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE metadata (metadata_key TEXT PRIMARY KEY, metadata_value TEXT NOT NULL)"
        )
        self._connection.commit()
        self._connection.execute("BEGIN IMMEDIATE")
        self._digest = hashlib.sha256()
        self._digest.update(b"{")
        self._count = 0
        self._previous_key: str | None = None
        self._require_sorted_input = bool(require_sorted_input)
        self._insert_batch_rows = max(1, int(insert_batch_rows))
        self._pending_rows: list[tuple[str, str]] = []
        self._closed = False

    def _flush_pending(self) -> None:
        if not self._pending_rows:
            return
        self._connection.executemany(
            "INSERT INTO policy(policy_key, row_json) VALUES (?, ?)",
            self._pending_rows,
        )
        self._pending_rows.clear()

    def add(self, key: str, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("policy builder is closed")
        key = str(key)
        if (
            self._require_sorted_input
            and self._previous_key is not None
            and key <= self._previous_key
        ):
            raise ValueError("policy rows are not in strict canonical key order")
        if not isinstance(row, Mapping) or not row:
            raise ValueError("policy row is not a non-empty mapping")
        key_raw = canonical_json_bytes(key)
        row_raw = canonical_json_bytes(dict(row))
        if self._require_sorted_input:
            if self._count:
                self._digest.update(b",")
            self._digest.update(key_raw)
            self._digest.update(b":")
            self._digest.update(row_raw)
        self._pending_rows.append((key, row_raw.decode("ascii")))
        if len(self._pending_rows) >= self._insert_batch_rows:
            self._flush_pending()
        self._previous_key = key
        self._count += 1

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("policy builder was already closed")
        self._flush_pending()
        if not self._require_sorted_input:
            self._digest = hashlib.sha256()
            self._digest.update(b"{")
            for index, (key, row_json) in enumerate(
                self._connection.execute(
                    "SELECT policy_key, row_json FROM policy ORDER BY policy_key"
                )
            ):
                if index:
                    self._digest.update(b",")
                self._digest.update(canonical_json_bytes(str(key)))
                self._digest.update(b":")
                self._digest.update(str(row_json).encode("ascii"))
        self._digest.update(b"}")
        policy_sha256 = self._digest.hexdigest()
        self._connection.executemany(
            "INSERT INTO metadata(metadata_key, metadata_value) VALUES (?, ?)",
            (
                ("row_count", str(self._count)),
                ("policy_sha256", policy_sha256),
                ("schema_version", "stage21_disk_policy_store_v1"),
            ),
        )
        self._connection.commit()
        self._connection.close()
        self._closed = True
        return {
            "schema_version": "stage21_disk_policy_store_v1",
            "path": str(self.path),
            "row_count": self._count,
            "policy_sha256": policy_sha256,
            "sqlite_size": self.path.stat().st_size,
            "complete_policy_materialized_in_python": False,
            "input_rows_required_canonical_order": self._require_sorted_input,
            "canonical_hash_computed_from_disk_sorted_cursor": (
                not self._require_sorted_input
            ),
            "bounded_insert_batch_rows": self._insert_batch_rows,
            "one_sqlite_statement_per_policy_row": False,
        }

    def abort(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True
        self.path.unlink(missing_ok=True)


class DiskBackedPolicy(Mapping[str, Mapping[str, float]]):
    """Read-only policy mapping with a bounded decoded-row LRU cache."""

    canonical_key_order = True

    def __init__(
        self,
        path: Path,
        *,
        expected_policy_sha256: str | None = None,
        expected_row_count: int | None = None,
        cache_entries: int = 4096,
    ) -> None:
        self.path = path.resolve()
        uri = f"file:{self.path.as_posix()}?mode=ro&immutable=1"
        self._connection = sqlite3.connect(uri, uri=True)
        self._cache_limit = max(0, int(cache_entries))
        self._cache: OrderedDict[str, Mapping[str, float]] = OrderedDict()
        self._closed = False
        self._connection.execute("PRAGMA query_only=ON")
        self._connection.execute("PRAGMA cache_size=-16384")
        metadata = dict(
            self._connection.execute(
                "SELECT metadata_key, metadata_value FROM metadata"
            )
        )
        if metadata.get("schema_version") != "stage21_disk_policy_store_v1":
            self.close()
            raise ValueError("unsupported disk policy store")
        self.policy_sha256 = str(metadata["policy_sha256"])
        self.row_count = int(metadata["row_count"])
        if (
            expected_policy_sha256 is not None
            and self.policy_sha256 != str(expected_policy_sha256)
        ):
            self.close()
            raise ValueError("disk policy SHA-256 binding mismatch")
        if expected_row_count is not None and self.row_count != int(expected_row_count):
            self.close()
            raise ValueError("disk policy row-count binding mismatch")

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self) -> Iterator[str]:
        cursor = self._connection.execute(
            "SELECT policy_key FROM policy ORDER BY policy_key"
        )
        for (key,) in cursor:
            yield str(key)

    def __getitem__(self, key: str) -> Mapping[str, float]:
        key = str(key)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        row = self._connection.execute(
            "SELECT row_json FROM policy WHERE policy_key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        decoded = json.loads(str(row[0]))
        if not isinstance(decoded, Mapping):
            raise ValueError("disk policy row is not a mapping")
        value = {str(label): float(probability) for label, probability in decoded.items()}
        if self._cache_limit:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return value

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._cache.clear()
        self._connection.close()
        self._closed = True

    def cleanup(self) -> None:
        path = self.path
        self.close()
        path.unlink(missing_ok=True)
        Path(f"{path}-shm").unlink(missing_ok=True)
        Path(f"{path}-wal").unlink(missing_ok=True)

    def __enter__(self) -> "DiskBackedPolicy":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Aggregate validation deliberately keeps at most six policies open for
        # one root.  This is a final safety net for exceptional exits; normal
        # paths still close handles explicitly at the root boundary.
        try:
            self.close()
        except Exception:
            pass


def stage_self_hashed_policy_document(
    path: Path,
    *,
    policy_store_path: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_self_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one lane document and spool its policy without retaining it."""

    builder = DiskPolicyBuilder(policy_store_path)
    try:
        streamed = read_self_hashed_json_streamed_mapping(
            path,
            streamed_mapping_key="policy",
            on_member=builder.add,
        )
        store = builder.finalize()
    except Exception:
        builder.abort()
        raise
    identity = streamed["file_identity"]
    document = streamed["document"]
    if expected_size is not None and int(identity["size"]) != int(expected_size):
        Path(store["path"]).unlink(missing_ok=True)
        raise ValueError("lane document size binding mismatch")
    if expected_sha256 is not None and identity["sha256"] != str(expected_sha256):
        Path(store["path"]).unlink(missing_ok=True)
        raise ValueError("lane document raw SHA-256 binding mismatch")
    if (
        expected_self_sha256 is not None
        and document.get("self_sha256") != str(expected_self_sha256)
    ):
        Path(store["path"]).unlink(missing_ok=True)
        raise ValueError("lane document self-hash binding mismatch")
    streamed_policy_sha256 = streamed.get("streamed_mapping_sha256")
    if streamed.get("streamed_mapping_found") is True:
        if store["policy_sha256"] != streamed_policy_sha256:
            Path(store["path"]).unlink(missing_ok=True)
            raise AssertionError("policy spool identity changed during staging")
        policy_sha256: str | None = str(store["policy_sha256"])
    else:
        if int(store["row_count"]) != 0:
            raise AssertionError("missing policy produced stored rows")
        policy_sha256 = None
    return {
        "document": document,
        "file_identity": identity,
        "policy_store": store,
        "policy_sha256": policy_sha256,
        "policy_mapping_present": streamed.get("streamed_mapping_found") is True,
        "canonical_self_hash_verified": True,
        "complete_policy_materialized_in_python": False,
    }


def open_staged_policy(handle: Mapping[str, Any], *, cache_entries: int = 4096) -> DiskBackedPolicy:
    store = handle.get("policy_store")
    if not isinstance(store, Mapping):
        raise ValueError("lane handle lacks a disk policy store")
    return DiskBackedPolicy(
        Path(str(store["path"])),
        expected_policy_sha256=str(store["policy_sha256"]),
        expected_row_count=int(store["row_count"]),
        cache_entries=cache_entries,
    )
