from __future__ import annotations

import gc
import hashlib
import heapq
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    document_self_sha256,
    parse_utc_timestamp,
    validate_document_self_hash,
)
from stage6_notebook01_common import (
    STAGING_OWNER_RECEIPT_NAME,
    atomic_publish_owned_regular_file,
    assert_active_run_root_identity,
    create_exclusive_regular_file,
    create_exclusive_temporary_file,
    create_owned_staging_directory,
    ensure_durable_directory,
    fsync_directory,
    fsync_regular_file,
    open_identity_bound_regular_file,
    publish_owned_staging_directory,
    regular_file_identity,
    remove_current_run_stale_staging_directories,
    remove_owned_staging_directory,
    verify_owned_staging_directory,
)


ALGORITHM_VERSION = "stage6_group_time_panel_assignment_v1"
LOCKBOX_FRACTION = 0.10
PRELOCKBOX_TRAIN_FRACTION = 0.70
PRELOCKBOX_VALIDATION_FRACTION = 0.15
PRELOCKBOX_TEST_FRACTION = 0.15
UNSEEN_PLAYER_TRAIN_ACTOR_PERCENT = 20
UNSEEN_PLAYER_TRAIN_HAND_PERCENT = 20
UNSEEN_GAME_HAND_PERCENT = 10
LANE_IDS = tuple(f"lane-{index:02d}" for index in range(48))
ASSIGNMENT_COLUMNS = (
    "row_id",
    "game_id",
    "hand_id",
    "decision_id",
    "decision_utc",
    "decision_order",
    "actor_id",
    "panel",
    "role",
    "feature_cutoff_utc",
    "profile_snapshot_cutoff_utc",
    "future_reveal_used",
    "post_decision_outcome_used",
    "eligible_for_selection",
)
EXPECTED_PANEL_ROLES = {
    "future_known_player": {"train", "validation", "test"},
    "unseen_player_cold_start": {"train", "validation", "test"},
    "unseen_game": {"train", "validation", "test"},
    "final_private_lockbox": {"test"},
}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _epoch_millis(value: Any, *, label: str) -> int:
    parsed = parse_utc_timestamp(value, label=label)
    delta = parsed - EPOCH
    return (
        delta.days * 86_400_000
        + delta.seconds * 1_000
        + delta.microseconds // 1_000
    )


def _canonical_relative(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("/") or ".." in path.parts or ":" in value:
        raise Stage6ContractError(f"path is not canonical and relative: {value}")
    return value


def _file_identity(path: Path, *, relative_path: Path) -> dict[str, Any]:
    return regular_file_identity(
        path, relative_path=Path(_canonical_relative(relative_path))
    )


def _verify_identity_bound_parquet(
    run_root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
) -> None:
    import pyarrow.parquet as pq

    with open_identity_bound_regular_file(
        run_root,
        path,
        expected_bytes=identity["bytes"],
        expected_sha256=identity["sha256"],
    ) as reader:
        if pq.ParquetFile(reader).metadata.num_rows != identity["row_count"]:
            raise Stage6ContractError(f"{label} row-count receipt mismatch")


def _iter_identity_bound_parquet_rows(
    run_root: Path,
    path: Path,
    identity: Mapping[str, Any],
    *,
    label: str,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    with open_identity_bound_regular_file(
        run_root,
        path,
        expected_bytes=identity["bytes"],
        expected_sha256=identity["sha256"],
    ) as reader:
        parquet = pq.ParquetFile(reader)
        if parquet.metadata.num_rows != identity["row_count"]:
            raise Stage6ContractError(f"{label} row-count receipt mismatch")
        for batch in parquet.iter_batches(batch_size=16_384, use_threads=False):
            yield from batch.to_pylist()


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage6ContractError(f"duplicate JSON key in {path}: {key}")
            output[key] = value
        return output

    with path.open("r", encoding="utf-8", errors="strict") as source:
        value = json.load(source, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise Stage6ContractError(f"JSON authority is not an object: {path}")
    return value


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    ensure_durable_directory(path.parent)
    payload = json.dumps(
        dict(document), indent=2, sort_keys=True, ensure_ascii=True
    ) + "\n"
    descriptor, temporary, created = create_exclusive_temporary_file(
        path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())

    def validate_json(reader: Any) -> None:
        if reader.read().decode("utf-8", errors="strict") != payload:
            raise Stage6ContractError(f"durable JSON readback mismatch: {path}")

    atomic_publish_owned_regular_file(
        temporary,
        path,
        expected_creation_identity=(int(created.st_dev), int(created.st_ino)),
        directory_fsync=fsync_directory,
        validate=validate_json,
    )


def _assignment_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("row_id", pa.string(), nullable=False),
            pa.field("game_id", pa.string(), nullable=False),
            pa.field("hand_id", pa.string(), nullable=False),
            pa.field("decision_id", pa.string(), nullable=False),
            pa.field("decision_utc", pa.string(), nullable=False),
            pa.field("decision_order", pa.int64(), nullable=False),
            pa.field("actor_id", pa.string(), nullable=False),
            pa.field("panel", pa.string(), nullable=False),
            pa.field("role", pa.string(), nullable=False),
            pa.field("feature_cutoff_utc", pa.string(), nullable=False),
            pa.field(
                "profile_snapshot_cutoff_utc", pa.string(), nullable=False
            ),
            pa.field("future_reveal_used", pa.bool_(), nullable=False),
            pa.field("post_decision_outcome_used", pa.bool_(), nullable=False),
            pa.field("eligible_for_selection", pa.bool_(), nullable=False),
        ]
    )


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_durable_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    table = pa.Table.from_pylist(list(rows), schema=_assignment_schema())
    descriptor, created = create_exclusive_regular_file(temporary)
    with os.fdopen(descriptor, "wb") as handle:
        pq.write_table(
            table,
            handle,
            compression="zstd",
            use_dictionary=True,
            row_group_size=max(1, min(len(rows), 16_384)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    assert_active_run_root_identity(temporary)
    fsync_regular_file(temporary)
    expected_rows = len(rows)

    def validate_parquet(reader: Any) -> None:
        metadata_rows = pq.ParquetFile(reader).metadata.num_rows
        if metadata_rows != expected_rows:
            raise Stage6ContractError(f"split Parquet row-count mismatch: {path}")

    identity = atomic_publish_owned_regular_file(
        temporary,
        path,
        expected_creation_identity=(int(created.st_dev), int(created.st_ino)),
        directory_fsync=fsync_directory,
        validate=validate_parquet,
    )
    identity.pop("publication_status", None)
    identity["row_count"] = expected_rows
    return identity


def _hash_bucket(namespace: str, *values: str) -> int:
    payload = {"namespace": namespace, "values": list(values)}
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") % 100


def _fixed_role(namespace: str, value: str) -> str:
    bucket = _hash_bucket(namespace, value)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _nearest_gap(
    gaps: Sequence[int], *, target: int, minimum: int = 1, maximum: int
) -> int:
    eligible = [value for value in gaps if minimum <= value < maximum]
    if not eligible:
        raise Stage6ContractError(
            f"no strict global hand interval gap in [{minimum}, {maximum})"
        )
    return min(eligible, key=lambda value: (abs(value - target), value))


def _build_hand_roles(
    hand_metadata: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    ordered = sorted(
        hand_metadata.items(),
        key=lambda item: (
            item[1]["min_epoch_ms"],
            item[0][0],
            item[0][1],
        ),
    )
    if len(ordered) < 4:
        raise Stage6ContractError("not enough accepted hands for grouped panels")
    running_max: list[int] = []
    maximum = -1
    for _, metadata in ordered:
        maximum = max(maximum, metadata["max_epoch_ms"])
        running_max.append(maximum)
    gaps = [
        index + 1
        for index in range(len(ordered) - 1)
        if running_max[index] < ordered[index + 1][1]["min_epoch_ms"]
    ]
    lockbox_start = _nearest_gap(
        gaps,
        target=round(len(ordered) * (1.0 - LOCKBOX_FRACTION)),
        maximum=len(ordered),
    )
    train_end = _nearest_gap(
        gaps,
        target=round(lockbox_start * PRELOCKBOX_TRAIN_FRACTION),
        maximum=lockbox_start,
    )
    validation_end = _nearest_gap(
        gaps,
        target=round(
            lockbox_start
            * (PRELOCKBOX_TRAIN_FRACTION + PRELOCKBOX_VALIDATION_FRACTION)
        ),
        minimum=train_end + 1,
        maximum=lockbox_start,
    )
    roles: dict[tuple[str, str], str] = {}
    for index, (key, _) in enumerate(ordered):
        roles[key] = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
            if index < lockbox_start
            else "lockbox"
        )
    boundary = {
        "accepted_hand_count": len(ordered),
        "strict_gap_count": len(gaps),
        "train_end_hand_index": train_end,
        "validation_end_hand_index": validation_end,
        "lockbox_start_hand_index": lockbox_start,
        "train_max_decision_epoch_ms": running_max[train_end - 1],
        "validation_min_decision_epoch_ms": ordered[train_end][1][
            "min_epoch_ms"
        ],
        "validation_max_decision_epoch_ms": running_max[validation_end - 1],
        "test_min_decision_epoch_ms": ordered[validation_end][1][
            "min_epoch_ms"
        ],
        "prelockbox_max_decision_epoch_ms": running_max[lockbox_start - 1],
        "lockbox_min_decision_epoch_ms": ordered[lockbox_start][1][
            "min_epoch_ms"
        ],
    }
    return roles, boundary


def _input_lane_identities(
    run_root: Path, lane_checkpoints: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    observed_lane_ids = [str(row["lane_id"]) for row in lane_checkpoints]
    if observed_lane_ids != list(LANE_IDS):
        raise Stage6ContractError("split input lanes are not in exact lexical order")
    for checkpoint in lane_checkpoints:
        identity = dict(checkpoint["outputs"]["accepted_decisions"])
        relative = Path(identity["relative_path"])
        path = run_root / relative
        _verify_identity_bound_parquet(
            run_root,
            path,
            identity,
            label=f"accepted decision lane {relative.as_posix()}",
        )
        rows.append({"lane_id": checkpoint["lane_id"], "accepted_decisions": identity})
    return rows


def _streaming_assignment_root(
    run_root: Path, outputs: Sequence[Mapping[str, Any]]
) -> tuple[str, int]:
    iterators: list[Iterator[dict[str, Any]]] = []
    heap: list[tuple[str, int, dict[str, Any]]] = []
    for index, output in enumerate(outputs):
        relative = Path(output["relative_path"])
        iterator = _iter_identity_bound_parquet_rows(
            run_root,
            run_root / relative,
            output,
            label=f"split root input {relative.as_posix()}",
        )
        iterators.append(iterator)
        first = next(iterator, None)
        if first is not None:
            heapq.heappush(heap, (first["row_id"], index, first))
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    previous_row_id: str | None = None
    while heap:
        row_id, index, row = heapq.heappop(heap)
        if previous_row_id is not None and row_id <= previous_row_id:
            raise Stage6ContractError("split row IDs are not globally unique and ordered")
        if count:
            digest.update(b",")
        digest.update(canonical_json_bytes(row))
        count += 1
        previous_row_id = row_id
        following = next(iterators[index], None)
        if following is not None:
            heapq.heappush(heap, (following["row_id"], index, following))
    digest.update(b"]")
    return digest.hexdigest(), count


def _verify_existing_manifest(
    run_root: Path,
    manifest: Mapping[str, Any],
    *,
    run_identity: str,
    input_lanes: Sequence[Mapping[str, Any]],
    split_contract_self_sha256: str,
) -> dict[str, Any]:
    validate_document_self_hash(manifest)
    if (
        manifest.get("schema_version")
        != "stage6_notebook01_split_assignment_manifest_v1"
        or manifest.get("status")
        != "PASS_SEALED_GROUP_TIME_SPLIT_ASSIGNMENT"
        or manifest.get("run_identity") != run_identity
        or manifest.get("input_lanes") != list(input_lanes)
        or manifest.get("split_contract_self_sha256")
        != split_contract_self_sha256
    ):
        raise Stage6ContractError("existing split manifest binding is invalid")
    outputs = manifest.get("lane_outputs")
    if not isinstance(outputs, list) or [row.get("lane_id") for row in outputs] != list(
        LANE_IDS
    ):
        raise Stage6ContractError("existing split output lane order is invalid")
    for output in outputs:
        relative = Path(output["relative_path"])
        path = run_root / relative
        _verify_identity_bound_parquet(
            run_root,
            path,
            output,
            label=f"split output {relative.as_posix()}",
        )
    root, count = _streaming_assignment_root(run_root, outputs)
    if (
        root != manifest["split_assignment_root_digest"]
        or count != manifest["accepted_decision_row_count"]
    ):
        raise Stage6ContractError("split assignment root readback mismatch")
    return dict(manifest)


def build_or_resume_split_assignments(
    run_root: Path,
    lane_checkpoints: Sequence[Mapping[str, Any]],
    split_contract: Mapping[str, Any],
    *,
    run_identity: str,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    if "OneDrive" in str(run_root):
        raise Stage6ContractError("historical OneDrive checkout is forbidden")
    validate_document_self_hash(split_contract)
    algorithm = split_contract.get("assignment_algorithm", {})
    if (
        algorithm.get("version") != ALGORITHM_VERSION
        or algorithm.get("final_private_lockbox_fraction_of_hands")
        != LOCKBOX_FRACTION
        or algorithm.get("prelockbox_train_fraction")
        != PRELOCKBOX_TRAIN_FRACTION
        or algorithm.get("prelockbox_validation_fraction")
        != PRELOCKBOX_VALIDATION_FRACTION
        or algorithm.get("prelockbox_test_fraction")
        != PRELOCKBOX_TEST_FRACTION
        or algorithm.get("unseen_player_train_actor_bucket_percent")
        != UNSEEN_PLAYER_TRAIN_ACTOR_PERCENT
        or algorithm.get("unseen_player_train_hand_sample_percent")
        != UNSEEN_PLAYER_TRAIN_HAND_PERCENT
        or algorithm.get("unseen_game_hand_sample_percent")
        != UNSEEN_GAME_HAND_PERCENT
    ):
        raise Stage6ContractError("split assignment contract differs from code constants")
    input_lanes = _input_lane_identities(run_root, lane_checkpoints)
    split_root = run_root / "splits"
    manifest_path = split_root / "split-assignment-manifest.json"
    if manifest_path.is_file():
        existing = _verify_existing_manifest(
            run_root,
            _strict_json_object(manifest_path),
            run_identity=run_identity,
            input_lanes=input_lanes,
            split_contract_self_sha256=split_contract["self_sha256"],
        )
        receipt_path = split_root / STAGING_OWNER_RECEIPT_NAME
        if os.path.lexists(receipt_path):
            staging_ownership = verify_owned_staging_directory(
                run_root,
                split_root,
                kind="splits",
            )
            if staging_ownership.get("run_identity") != run_identity:
                raise Stage6ContractError(
                    "manifest-complete direct split staging belongs to another run"
                )
            publish_owned_staging_directory(
                run_root,
                split_root,
                split_root,
                kind="splits",
                expected_receipt=staging_ownership,
            )
            existing = _verify_existing_manifest(
                run_root,
                _strict_json_object(manifest_path),
                run_identity=run_identity,
                input_lanes=input_lanes,
                split_contract_self_sha256=split_contract["self_sha256"],
            )
        return existing

    remove_current_run_stale_staging_directories(
        run_root, kind="splits", run_identity=run_identity
    )
    direct = run_root / "splits"
    if direct.exists():
        staging_ownership = verify_owned_staging_directory(
            run_root,
            direct,
            kind="splits",
        )
        if staging_ownership.get("run_identity") != run_identity:
            raise Stage6ContractError(
                "incomplete direct split staging belongs to another run"
            )
        staging = direct
    else:
        staging, staging_ownership = create_owned_staging_directory(
            run_root,
            kind="splits",
            run_identity=run_identity,
            destination_name="splits",
        )
    hand_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    accepted_count = 0
    try:
        for lane in input_lanes:
            source = lane["accepted_decisions"]
            relative = Path(source["relative_path"])
            path = run_root / relative
            for row in _iter_identity_bound_parquet_rows(
                run_root,
                path,
                source,
                label=f"split metadata input {relative.as_posix()}",
            ):
                key = (row["game_id"], row["hand_id"])
                epoch_ms = _epoch_millis(
                    row["decision_utc"], label=f"{row['row_id']}.decision_utc"
                )
                metadata = hand_metadata.setdefault(
                    key,
                    {
                        "min_epoch_ms": epoch_ms,
                        "max_epoch_ms": epoch_ms,
                        "actors": set(),
                    },
                )
                metadata["min_epoch_ms"] = min(metadata["min_epoch_ms"], epoch_ms)
                metadata["max_epoch_ms"] = max(metadata["max_epoch_ms"], epoch_ms)
                metadata["actors"].add(row["actor_id"])
                accepted_count += 1
        if accepted_count != sum(
            row["accepted_decisions"]["row_count"] for row in input_lanes
        ):
            raise Stage6ContractError("accepted split input work units were not conserved")

        hand_roles, boundaries = _build_hand_roles(hand_metadata)
        actor_roles: dict[str, set[str]] = defaultdict(set)
        for key, metadata in hand_metadata.items():
            role = hand_roles[key]
            if role != "lockbox":
                for actor in metadata["actors"]:
                    actor_roles[actor].add(role)
        cold_actor_role: dict[str, str] = {}
        multi_eval_cold_actors: set[str] = set()
        for actor, roles in actor_roles.items():
            if "train" in roles:
                continue
            if roles == {"validation"}:
                cold_actor_role[actor] = "validation"
            elif roles == {"test"}:
                cold_actor_role[actor] = "test"
            elif roles == {"validation", "test"}:
                cold_actor_role[actor] = "validation"
                multi_eval_cold_actors.add(actor)
            else:
                raise Stage6ContractError(
                    f"cold-start actor has unsupported prelockbox roles: {actor}={roles}"
                )
        forced_test_games = {
            key[0]
            for key, metadata in hand_metadata.items()
            if hand_roles[key] == "test"
            and metadata["actors"].intersection(multi_eval_cold_actors)
        }
        game_fixed_role = {
            game_id: (
                "test"
                if game_id in forced_test_games
                else _fixed_role("stage6-unseen-game-role-v1", game_id)
            )
            for game_id in sorted({key[0] for key in hand_metadata})
        }

        preserved_future_train_row: dict[str, str] = {}
        for lane in input_lanes:
            source = lane["accepted_decisions"]
            relative = Path(source["relative_path"])
            path = run_root / relative
            for row in _iter_identity_bound_parquet_rows(
                run_root,
                path,
                source,
                label=f"split training-presence input {relative.as_posix()}",
            ):
                key = (row["game_id"], row["hand_id"])
                if hand_roles[key] == "train":
                    previous = preserved_future_train_row.get(row["actor_id"])
                    if previous is None or row["row_id"] < previous:
                        preserved_future_train_row[row["actor_id"]] = row["row_id"]

        panel_role_counts: Counter[tuple[str, str]] = Counter()
        unseen_actor_roles: dict[str, set[str]] = defaultdict(set)
        unseen_game_roles: dict[str, set[str]] = defaultdict(set)
        future_train_actors: set[str] = set()
        future_eval_actors: set[str] = set()
        lane_outputs: list[dict[str, Any]] = []
        assignment_count = 0
        for lane in input_lanes:
            lane_id = lane["lane_id"]
            source = lane["accepted_decisions"]
            relative = Path(source["relative_path"])
            path = run_root / relative
            assignments: list[dict[str, Any]] = []
            previous_row_id: str | None = None
            for row in _iter_identity_bound_parquet_rows(
                run_root,
                path,
                source,
                label=f"split assignment input {relative.as_posix()}",
            ):
                row_id = row["row_id"]
                if previous_row_id is not None and row_id <= previous_row_id:
                    raise Stage6ContractError(f"{lane_id}: accepted rows are not ordered")
                previous_row_id = row_id
                key = (row["game_id"], row["hand_id"])
                temporal_role = hand_roles[key]
                actor = row["actor_id"]
                game = row["game_id"]
                if temporal_role == "lockbox":
                    panel = "final_private_lockbox"
                    role = "test"
                elif actor in cold_actor_role:
                    role = temporal_role
                    if cold_actor_role[actor] == role:
                        panel = "unseen_player_cold_start"
                    else:
                        if role != "test" or game_fixed_role[game] != "test":
                            raise Stage6ContractError(
                                "multi-role cold-start actor cannot be routed without leakage"
                            )
                        panel = "unseen_game"
                else:
                    role = temporal_role
                    preserve = preserved_future_train_row.get(actor) == row_id
                    actor_train_sample = (
                        role == "train"
                        and not preserve
                        and _hash_bucket("stage6-unseen-player-train-actor-v1", actor)
                        < UNSEEN_PLAYER_TRAIN_ACTOR_PERCENT
                        and _hash_bucket(
                            "stage6-unseen-player-train-hand-v1", game, row["hand_id"]
                        )
                        < UNSEEN_PLAYER_TRAIN_HAND_PERCENT
                    )
                    game_sample = (
                        not preserve
                        and game_fixed_role[game] == role
                        and _hash_bucket(
                            "stage6-unseen-game-hand-v1", game, row["hand_id"]
                        )
                        < UNSEEN_GAME_HAND_PERCENT
                    )
                    if actor_train_sample:
                        panel = "unseen_player_cold_start"
                    elif game_sample:
                        panel = "unseen_game"
                    else:
                        panel = "future_known_player"

                eligible = role == "validation" and panel != "final_private_lockbox"
                assignment = {
                    "row_id": row_id,
                    "game_id": game,
                    "hand_id": row["hand_id"],
                    "decision_id": row["decision_id"],
                    "decision_utc": row["decision_utc"],
                    "decision_order": row["decision_order"],
                    "actor_id": actor,
                    "panel": panel,
                    "role": role,
                    "feature_cutoff_utc": row["feature_cutoff_utc"],
                    "profile_snapshot_cutoff_utc": row[
                        "profile_snapshot_cutoff_utc"
                    ],
                    "future_reveal_used": row["future_reveal_used"],
                    "post_decision_outcome_used": row[
                        "post_decision_outcome_used"
                    ],
                    "eligible_for_selection": eligible,
                }
                if assignment["future_reveal_used"] or assignment[
                    "post_decision_outcome_used"
                ]:
                    raise Stage6ContractError("split assignment contains a leakage flag")
                assignments.append(assignment)
                assignment_count += 1
                panel_role_counts[(panel, role)] += 1
                if panel == "unseen_player_cold_start":
                    unseen_actor_roles[actor].add(role)
                elif panel == "unseen_game":
                    unseen_game_roles[game].add(role)
                elif panel == "future_known_player":
                    if role == "train":
                        future_train_actors.add(actor)
                    elif role in {"validation", "test"}:
                        future_eval_actors.add(actor)

            output_path = staging / f"{lane_id}-split-assignments.parquet"
            identity = _write_parquet(output_path, assignments)
            identity["relative_path"] = (
                Path("splits") / output_path.name
            ).as_posix()
            identity["lane_id"] = lane_id
            lane_outputs.append(identity)
            del assignments
            gc.collect()

        if assignment_count != accepted_count:
            raise Stage6ContractError("split assignments do not conserve accepted rows")
        actor_role_violations = {
            actor: sorted(roles)
            for actor, roles in unseen_actor_roles.items()
            if len(roles) != 1
        }
        game_role_violations = {
            game: sorted(roles)
            for game, roles in unseen_game_roles.items()
            if len(roles) != 1
        }
        unseen_future_actors = sorted(future_eval_actors - future_train_actors)
        if actor_role_violations:
            raise Stage6ContractError(
                f"unseen-player actor roles overlap: {list(actor_role_violations.items())[:5]}"
            )
        if game_role_violations:
            raise Stage6ContractError(
                f"unseen-game roles overlap: {list(game_role_violations.items())[:5]}"
            )
        if unseen_future_actors:
            raise Stage6ContractError(
                f"future-known evaluation actors lack training rows: {unseen_future_actors[:5]}"
            )
        observed_panel_roles: dict[str, set[str]] = defaultdict(set)
        for (panel, role), count in panel_role_counts.items():
            if count:
                observed_panel_roles[panel].add(role)
        if dict(observed_panel_roles) != EXPECTED_PANEL_ROLES:
            raise Stage6ContractError(
                f"split panel/role coverage differs from contract: {dict(observed_panel_roles)}"
            )

        provisional_outputs = [
            {
                **row,
                "relative_path": (
                    Path(staging.name) / Path(row["relative_path"]).name
                ).as_posix(),
            }
            for row in lane_outputs
        ]
        root_digest, root_count = _streaming_assignment_root(
            run_root, provisional_outputs
        )
        if root_count != accepted_count:
            raise Stage6ContractError("split root does not conserve accepted rows")

        manifest: dict[str, Any] = {
            "schema_version": "stage6_notebook01_split_assignment_manifest_v1",
            "status": "PASS_SEALED_GROUP_TIME_SPLIT_ASSIGNMENT",
            "run_identity": run_identity,
            "algorithm_version": ALGORITHM_VERSION,
            "split_contract_self_sha256": split_contract["self_sha256"],
            "input_lanes": input_lanes,
            "accepted_decision_row_count": accepted_count,
            "accepted_hand_count": len(hand_metadata),
            "panel_role_counts": {
                f"{panel}:{role}": count
                for (panel, role), count in sorted(panel_role_counts.items())
            },
            "temporal_boundaries": boundaries,
            "cold_start_actor_count": len(cold_actor_role),
            "multi_eval_cold_start_actor_count": len(multi_eval_cold_actors),
            "forced_test_game_count": len(forced_test_games),
            "unseen_player_actor_role_overlap_count": 0,
            "unseen_game_role_overlap_count": 0,
            "future_known_eval_actor_without_train_count": 0,
            "future_reveal_violation_count": 0,
            "post_decision_outcome_violation_count": 0,
            "work_unit_conservation_verified": True,
            "private_lockbox_assignment_sealed": True,
            "private_lockbox_opened": False,
            "private_lockbox_eligible_for_selection": False,
            "split_assignment_root_order": "row_id_lexical_order",
            "split_assignment_root_digest": root_digest,
            "lane_outputs": lane_outputs,
            "scientific_authority": False,
            "promotion_authority": False,
            "drive_publication_authority": False,
            "live_action_authority": False,
        }
        manifest["self_sha256"] = document_self_sha256(manifest)
        _atomic_write_json(staging / "split-assignment-manifest.json", manifest)
        publish_owned_staging_directory(
            run_root,
            staging,
            split_root,
            kind="splits",
            expected_receipt=staging_ownership,
        )
        return _verify_existing_manifest(
            run_root,
            _strict_json_object(manifest_path),
            run_identity=run_identity,
            input_lanes=input_lanes,
            split_contract_self_sha256=split_contract["self_sha256"],
        )
    except BaseException as primary_error:
        try:
            remove_owned_staging_directory(
                run_root,
                staging,
                kind="splits",
                expected_receipt=staging_ownership,
            )
        except BaseException as cleanup_error:
            primary_error.add_note(f"owned split staging cleanup was held: {cleanup_error}")
        raise
