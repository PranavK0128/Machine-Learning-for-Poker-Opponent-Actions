"""Authenticated, disk-backed derivative of the accepted Notebook03 checkpoint.

Source SQLite is opened immutable/read-only after hashing. New attempt folders
preserve interruptions. A receipt is published last; completed members are
reused by hash. There is no raw-source regeneration or model deserialization.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

import numpy as np
import stage7_hierarchical_response_v1 as h
import stage6_empirical_range_learning_v2_release as transport


def read_only(path, expected_sha256):
    path = Path(path).resolve()
    if transport.sha(path) != expected_sha256:
        raise ValueError("accepted_compact_database_hash_mismatch")
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("source_database_has_unsealed_sidecar")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def validate_receipt(directory, receipt, binding):
    if set(receipt) != {"schema_version", "binding", "evidence_class", "authority", "attempt",
                        "source_preparation_recomputed", "source_compact_database_reused", "game_ids",
                        "row_counts", "feature_names", "members"}:
        raise ValueError("preparation_receipt_schema_mismatch")
    if receipt.get("schema_version") != h.SCHEMA + "_preparation" or receipt.get("binding") != binding:
        raise ValueError("hierarchical_preparation_binding_mismatch")
    if receipt.get("authority") != h.AUTHORITY or receipt.get("evidence_class") != h.EVIDENCE:
        raise ValueError("preparation_authority_changed")
    attempt = receipt.get("attempt", "")
    if not isinstance(attempt, str) or not re.fullmatch(r"attempt_[a-f0-9]{32}", attempt):
        raise ValueError("invalid_preparation_attempt_name")
    expected_names = {role: h.feature_names(role, "session") for role in h.ROLES}
    if receipt.get("feature_names") != expected_names or receipt.get("game_ids") != binding.get("game_ids"):
        raise ValueError("preparation_semantic_schema_mismatch")
    expected_members = {attempt + "/events.sqlite"}
    expected_members.update(attempt + "/" + role + "_" + part + "." + kind + ".npy"
                            for role in h.ROLES for part in ("train", "calibration", "evaluation") for kind in ("x", "y", "ordinal"))
    if set(receipt.get("members", {})) != expected_members:
        raise ValueError("preparation_exact_member_inventory_mismatch")
    if receipt.get("source_preparation_recomputed") is not False or receipt.get("source_compact_database_reused") is not True:
        raise ValueError("preparation_reuse_claim_mismatch")
    for name, expected in receipt["members"].items():
        transport.safe_name(name)
        transport.verify(Path(directory) / name, expected)
    if {p.name for p in (Path(directory) / attempt).iterdir()} != {Path(n).name for n in expected_members}:
        raise ValueError("preparation_unmanifested_attempt_member")
    database = Path(directory) / attempt / "events.sqlite"
    con = read_only(database, receipt["members"][attempt + "/events.sqlite"]["sha256"])
    try:
        rows = {role: {part: 0 for part in ("train", "calibration", "evaluation")} for role in h.ROLES}
        actual_games = {part: set() for part in rows[h.ACTION]}
        for part, game, action, count in con.execute(
                "SELECT partition,json_extract(compact_json,'$.game_id'),json_extract(compact_json,'$.action'),COUNT(*) "
                "FROM events GROUP BY 1,2,3"):
            if part not in rows[h.ACTION] or action not in h.LABELS[h.ACTION]:
                raise ValueError("preparation_event_inventory_mismatch")
            actual_games[part].add(game)
            rows[h.ACTION][part] += count
            if action in h.AGGRESSIVE:
                rows[h.SIZE][part] += count
        if receipt.get("row_counts") != rows or any(actual_games[p] != set(receipt["game_ids"][p]) for p in actual_games):
            raise ValueError("preparation_game_or_row_counts_mismatch")
    finally:
        con.close()
    for role in h.ROLES:
        for part in rows[role]:
            stem = Path(directory) / attempt / (role + "_" + part)
            for kind, dtype in (("x", np.float64), ("y", np.int16), ("ordinal", np.int64)):
                array = np.load(str(stem) + "." + kind + ".npy", mmap_mode="r", allow_pickle=False)
                shape = (rows[role][part], len(expected_names[role])) if kind == "x" else (rows[role][part],)
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise ValueError("preparation_array_shape_or_dtype_mismatch")
    return receipt


def prepare(source_database, expected_sha256, protocol, game_bounds, directory, binding, window="100", expected_receipt_identity=None):
    """Stream one fixed training window and its later calibration/evaluation."""
    directory = Path(directory)
    receipt_path = directory / "preparation_receipt.json"
    if expected_receipt_identity is not None and not receipt_path.is_file():
        raise ValueError("expected_preparation_receipt_missing")
    if (protocol.get("schema_version") != h.SCHEMA + "_protocol"
            or protocol.get("evidence_class") != h.EVIDENCE
            or protocol.get("authority") != h.AUTHORITY
            or protocol.get("untouched_confirmation_game_ids") != []
            or window not in protocol["training_windows"]):
        raise ValueError("frozen_development_protocol_required")
    groups = {"train": protocol["training_windows"][window],
              "calibration": protocol["calibration_game_ids"],
              "evaluation": protocol["development_evaluation_game_ids"]}
    membership = {game: part for part, games in groups.items() for game in games}
    expected_binding = dict(binding, database_sha256=expected_sha256, window=window, game_ids=groups,
                            game_bounds_sha256=h.incumbent.hashlib.sha256(h.incumbent.canonical(game_bounds).encode()).hexdigest(),
                            protocol_sha256=h.incumbent.hashlib.sha256(h.incumbent.canonical(protocol).encode()).hexdigest())
    if receipt_path.exists():
        if expected_receipt_identity is None:
            raise ValueError("trusted_preparation_receipt_identity_required")
        # The controller obtains this identity from its own completion record
        # or an independently authenticated durable checkpoint manifest.
        transport.verify(receipt_path, expected_receipt_identity)
        return validate_receipt(directory, transport.load(receipt_path), expected_binding)
    if sum(map(len, groups.values())) != len(membership):
        raise ValueError("whole_game_cohort_overlap")
    for left, right in (("train", "calibration"), ("calibration", "evaluation")):
        if not groups[left] or not groups[right] or max(h.incumbent.utc(game_bounds[g][1]) for g in groups[left]) >= min(h.incumbent.utc(game_bounds[g][0]) for g in groups[right]):
            raise ValueError("cohorts_not_chronological_whole_games")
    # Verify the source before creating a derivative or issuing SQL against it.
    source = read_only(source_database, expected_sha256)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / ("attempt_" + uuid.uuid4().hex)
    attempt.mkdir()
    events_path = attempt / "events.sqlite"
    events = sqlite3.connect(events_path)
    counts = Counter(); actual_games = set(); last_progress = time.monotonic()

    def progress(phase, **values):
        nonlocal last_progress
        if time.monotonic() - last_progress >= 20:
            print(h.incumbent.canonical(dict(phase=phase, window=window, **values)), flush=True)
            last_progress = time.monotonic()

    try:
        events.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE events (
                ordinal INTEGER PRIMARY KEY, row_id TEXT UNIQUE NOT NULL,
                partition TEXT NOT NULL, decision REAL NOT NULL,
                completion REAL NOT NULL, compact_json TEXT NOT NULL);
        """)
        # Source identity includes its unique row_id index; test fixtures use
        # the same invariant. Both cursor keys use parsed UTC seconds.
        for ordinal, database_row_id, partition, compact in source.execute(
                "SELECT ordinal,row_id,partition,compact_json FROM rows ORDER BY ordinal"):
            row = json.loads(compact)
            game = row["game_id"]
            if game not in membership:
                continue
            if membership[game] != partition or database_row_id != row["row_id"]:
                raise ValueError("accepted_row_cohort_or_identity_mismatch")
            if [row["game_started_utc"], row["game_completed_utc"]] != list(game_bounds[game]):
                raise ValueError("accepted_row_game_bounds_mismatch")
            clocks = h.validate_observation(row)
            # Private reveal fields are intentionally absent from this derivative.
            row = {key: row[key] for key in ("row_id", "game_id", "hand_id", "actor_id", "session_id",
                "game_started_utc", "game_completed_utc", "decision_utc", "public_available_utc",
                "action_observed_utc", "hand_completed_utc", "action", "legal_actions", "observed_action_size_pot_fraction", "public")}
            row["public"] = {key: row["public"][key] for key in ("street", "board", "position", "opportunity", *h.incumbent.NUMERIC)}
            events.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (
                ordinal, row["row_id"], partition, clocks["decision_utc"], clocks["hand_completed_utc"],
                h.incumbent.canonical(row)))
            actual_games.add(game)
            counts[(h.ACTION, partition)] += 1
            if row["action"] in h.AGGRESSIVE:
                counts[(h.SIZE, partition)] += 1
            progress("index_authenticated_compact_rows", rows=sum(n for (r, _), n in counts.items() if r == h.ACTION))
        if actual_games != set(membership):
            raise ValueError("declared_game_missing_from_compact_source")
        events.executescript("""
            CREATE INDEX decision_order ON events(decision,row_id);
            CREATE INDEX completion_order ON events(completion,row_id);
        """)
        events.commit()
        arrays = {}; offsets = Counter()
        for role in h.ROLES:
            for partition in groups:
                n = counts[(role, partition)]
                stem = attempt / (role + "_" + partition)
                width = len(h.feature_names(role, "session"))
                arrays[(role, partition)] = {
                    "x": np.lib.format.open_memmap(str(stem) + ".x.npy", mode="w+", dtype=np.float64, shape=(n, width)),
                    "y": np.lib.format.open_memmap(str(stem) + ".y.npy", mode="w+", dtype=np.int16, shape=(n,)),
                    "ordinal": np.lib.format.open_memmap(str(stem) + ".ordinal.npy", mode="w+", dtype=np.int64, shape=(n,))}
        def cursor(order):
            for ordinal, partition, compact in events.execute("SELECT ordinal,partition,compact_json FROM events ORDER BY " + order):
                row = json.loads(compact)
                row["_ordinal"] = ordinal; row["_partition"] = partition
                yield row
        # Two independent ordered cursors over the same unique event inventory.
        emitted = 0
        for row, features, _audit in h.causal_feature_stream(cursor("decision,row_id"), cursor("completion,row_id")):
            for role, variants in features.items():
                key = (role, row["_partition"]); index = offsets[key]
                arrays[key]["x"][index] = variants["session"]
                arrays[key]["y"][index] = h.target(row, role)
                arrays[key]["ordinal"][index] = row["_ordinal"]
                offsets[key] += 1
            emitted += 1
            progress("causal_hierarchical_features", rows=emitted)
        if offsets != counts:
            raise ValueError("hierarchical_feature_denominator_mismatch")
        for group in arrays.values():
            for array in group.values():
                array.flush()
        arrays.clear()
        events.close(); source.close()
    except BaseException:
        events.close(); source.close()
        raise  # Preserve the attempt directory for diagnosis and safe retry.
    # The source must still match the accepted hash after both cursors close.
    # read_only repeats sidecar rejection as well as the independent full hash.
    final_source_check = read_only(source_database, expected_sha256)
    final_source_check.close()
    members = {p.relative_to(directory).as_posix(): transport.identity(p) for p in sorted(attempt.iterdir()) if p.is_file()}
    result = dict(schema_version=h.SCHEMA + "_preparation", binding=expected_binding,
                  evidence_class=h.EVIDENCE, authority=h.AUTHORITY, attempt=attempt.name,
                  source_preparation_recomputed=False, source_compact_database_reused=True,
                  game_ids=groups, row_counts={role: {part: counts[(role, part)] for part in groups} for role in h.ROLES},
                  feature_names={role: h.feature_names(role, "session") for role in h.ROLES},
                  members=members)
    transport.publish_local_bytes(receipt_path, (h.incumbent.canonical(result) + "\n").encode())
    print(h.incumbent.canonical(dict(phase="hierarchical_preparation_complete", window=window, rows=emitted)), flush=True)
    return validate_receipt(directory, result, expected_binding)


def matrices(directory, receipt, role, partition, arm):
    """Memory-map the complete feature schema and select a fixed ablation."""
    if role not in h.ROLES or partition not in receipt["game_ids"] or arm not in h.ARMS:
        raise ValueError("unknown_prepared_role_partition_or_arm")
    if receipt["feature_names"] != {r: h.feature_names(r, "session") for r in h.ROLES}:
        raise ValueError("preparation_semantic_schema_mismatch")
    if not re.fullmatch(r"attempt_[a-f0-9]{32}", receipt["attempt"]):
        raise ValueError("invalid_preparation_attempt_name")
    stem = Path(directory) / receipt["attempt"] / (role + "_" + partition)
    complete = np.load(str(stem) + ".x.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(str(stem) + ".y.npy", mmap_mode="r", allow_pickle=False)
    ordinal = np.load(str(stem) + ".ordinal.npy", mmap_mode="r", allow_pickle=False)
    names = receipt["feature_names"][role]
    indices = [names.index(name) for name in h.feature_names(role, arm)]
    # Only a fitting worker materializes the selected columns for its one job.
    return complete[:, indices], target, ordinal


def metadata(directory, receipt, ordinals):
    database = Path(directory) / receipt["attempt"] / "events.sqlite"
    relative = database.relative_to(directory).as_posix()
    con = read_only(database, receipt["members"][relative]["sha256"])
    try:
        result = []
        for ordinal in ordinals:
            found = con.execute("SELECT compact_json FROM events WHERE ordinal=?", (int(ordinal),)).fetchone()
            if found is None:
                raise ValueError("prepared_ordinal_missing")
            result.append(json.loads(found[0]))
        return result
    finally:
        con.close()
