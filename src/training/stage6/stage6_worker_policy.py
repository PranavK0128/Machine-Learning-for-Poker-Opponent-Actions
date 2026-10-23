from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from stage6_common import (
    Stage6ContractError,
    canonical_json_bytes,
    parse_utc_timestamp,
    sha256_file,
    validate_document_self_hash,
)
from stage6_input_packet import _normalized_member_name


AUTO_WORKER_VALUES = (None, 0, "", "0", "auto")
CHECKPOINT_PATTERN = re.compile(r"[0-9a-f]{64}")
PLAN_FIELDS = {
    "stage_id",
    "lane_id",
    "seed",
    "specification",
    "checkpoint_identity",
}
COMPLETION_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "stage_id",
    "lane_id",
    "seed",
    "checkpoint_identity",
    "checkpoint_relative_path",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "durable_readback_verified",
    "completed_at_utc",
    "self_sha256",
}


def _canonical_id(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


def _requested_cap(requested: Any, *, logical_cpus: int) -> int:
    auto = (
        requested is None
        or (type(requested) is int and requested == 0)
        or (
            type(requested) is str
            and requested in {"", "0", "auto"}
        )
    )
    if auto:
        return logical_cpus
    if isinstance(requested, bool):
        raise Stage6ContractError("requested workers must not be boolean")
    if type(requested) is int:
        requested_cap = requested
    elif type(requested) is str and requested == requested.strip():
        if not requested.isascii() or not requested.isdecimal():
            raise Stage6ContractError(
                f"requested workers is invalid: {requested!r}"
            )
        requested_cap = int(requested)
        if str(requested_cap) != requested:
            raise Stage6ContractError(
                "explicit requested workers must use canonical decimal text"
            )
    else:
        raise Stage6ContractError(f"requested workers is invalid: {requested!r}")
    if requested_cap <= 0:
        raise Stage6ContractError(
            "explicit requested workers must be a positive integer"
        )
    return requested_cap


def usable_logical_cpu_count() -> int:
    """Return the host CPU capacity available to this process.

    Colab accelerator selection is deliberately irrelevant here: GPU and TPU
    device cores are not process-pool lanes.  Linux affinity is the closest
    observable bound for the host processes; other platforms fall back to the
    interpreter-visible logical CPU count.
    """

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except (OSError, NotImplementedError):
            affinity_count = 0
        if affinity_count > 0:
            return affinity_count
    return max(1, int(os.cpu_count() or 1))


def resolve_worker_count(
    *,
    requested: Any,
    unfinished_lane_count: int,
    logical_cpus: int | None = None,
    lane_cap: int = 32,
) -> int:
    if type(unfinished_lane_count) is not int or unfinished_lane_count < 0:
        raise Stage6ContractError("unfinished_lane_count must be a nonnegative integer")
    if type(lane_cap) is not int or lane_cap <= 0:
        raise Stage6ContractError("lane_cap must be a positive integer")
    if logical_cpus is None:
        logical_cpus = usable_logical_cpu_count()
    if type(logical_cpus) is not int or logical_cpus <= 0:
        raise Stage6ContractError("logical_cpus must be a positive integer")
    requested_cap = _requested_cap(requested, logical_cpus=logical_cpus)
    if unfinished_lane_count == 0:
        return 0
    return min(logical_cpus, unfinished_lane_count, lane_cap, requested_cap)


def worker_resolution_report(
    *,
    requested: Any,
    unfinished_lane_count: int,
    logical_cpus: int | None = None,
    lane_cap: int = 48,
) -> dict[str, Any]:
    """Resolve live concurrency and make every capacity bound visible."""

    if logical_cpus is None:
        logical_cpus = usable_logical_cpu_count()
    live_workers = resolve_worker_count(
        requested=requested,
        unfinished_lane_count=unfinished_lane_count,
        logical_cpus=logical_cpus,
        lane_cap=lane_cap,
    )
    auto = requested in AUTO_WORKER_VALUES
    requested_cap = _requested_cap(requested, logical_cpus=logical_cpus)
    bounds = {
        "requested_or_auto_host_cpu_capacity": requested_cap,
        "unfinished_lane_count": unfinished_lane_count,
        "detected_usable_logical_cpus": logical_cpus,
        "logical_lane_cap": lane_cap,
    }
    active_bounds = sorted(
        name for name, value in bounds.items() if value == live_workers
    ) if live_workers else ["no_unfinished_lanes"]
    return {
        "schema_version": "stage6_runtime_neutral_worker_resolution_v1",
        "requested_workers": requested,
        "requested_workers_mode": "auto" if auto else "explicit",
        "detected_usable_logical_cpus": logical_cpus,
        "unfinished_lane_count": unfinished_lane_count,
        "logical_lane_cap": lane_cap,
        "live_workers": live_workers,
        "resolution_rule": (
            "min(requested_or_auto_host_cpu_capacity, unfinished_lane_count, "
            "detected_usable_logical_cpus, logical_lane_cap)"
        ),
        "active_capacity_bounds": active_bounds,
        "runtime_name_used_for_resolution": False,
        "accelerator_device_cores_used_as_process_workers": False,
        "scientific_lane_count_changed": False,
    }


def resolve_exact_worker_count(
    *,
    requested: Any,
    unfinished_lane_count: int,
    contracted_workers: int = 48,
) -> int:
    """Resolve a frozen no-clipping worker request for independent work units.

    This successor path deliberately does not consult ``os.cpu_count()`` or a
    runtime label.  Resource admission is a separate fail-closed preflight;
    once an exact Stage 6 run is admitted, the coordinator releases up to the
    contracted number of disjoint lane processes and never silently changes
    the worker contract.
    """

    if type(unfinished_lane_count) is not int or unfinished_lane_count < 0:
        raise Stage6ContractError("unfinished_lane_count must be a nonnegative integer")
    if type(contracted_workers) is not int or contracted_workers <= 0:
        raise Stage6ContractError("contracted_workers must be a positive integer")
    if isinstance(requested, bool):
        raise Stage6ContractError("requested workers must not be boolean")
    if type(requested) is int:
        requested_workers = requested
    elif type(requested) is str and requested == requested.strip():
        if not requested.isascii() or not requested.isdecimal():
            raise Stage6ContractError(
                f"requested workers is invalid: {requested!r}"
            )
        requested_workers = int(requested)
        if str(requested_workers) != requested:
            raise Stage6ContractError(
                "explicit requested workers must use canonical decimal text"
            )
    else:
        raise Stage6ContractError(f"requested workers is invalid: {requested!r}")
    if requested_workers != contracted_workers:
        raise Stage6ContractError(
            "requested workers does not match the frozen exact-worker contract"
        )
    if unfinished_lane_count == 0:
        return 0
    return min(contracted_workers, unfinished_lane_count)


def deterministic_lane_seed(*, stage_id: str, lane_id: str, base_seed: int) -> int:
    stage_id = _canonical_id(stage_id, label="stage_id")
    lane_id = _canonical_id(lane_id, label="lane_id")
    if type(base_seed) is not int or base_seed < 0:
        raise Stage6ContractError("base_seed must be a nonnegative integer")
    payload = canonical_json_bytes(
        {"stage_id": stage_id, "lane_id": lane_id, "base_seed": base_seed}
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def lane_identity(
    *, stage_id: str, lane_id: str, seed: int, specification: Mapping[str, Any]
) -> str:
    stage_id = _canonical_id(stage_id, label="stage_id")
    lane_id = _canonical_id(lane_id, label="lane_id")
    if type(seed) is not int or seed < 0:
        raise Stage6ContractError("lane seed must be a nonnegative integer")
    if not isinstance(specification, Mapping):
        raise Stage6ContractError("lane specification must be a mapping")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "stage_id": stage_id,
                "lane_id": lane_id,
                "seed": seed,
                "specification": dict(specification),
            }
        )
    ).hexdigest()


def build_lane_plan(
    *,
    stage_id: str,
    lanes: Iterable[Mapping[str, Any]],
    base_seed: int,
) -> list[dict[str, Any]]:
    stage_id = _canonical_id(stage_id, label="stage_id")
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    casefold_bindings: dict[str, str] = {}
    for raw_lane in lanes:
        if not isinstance(raw_lane, Mapping):
            raise Stage6ContractError("lane specification row must be a mapping")
        lane = dict(raw_lane)
        raw_lane_id = lane.pop("lane_id", "")
        lane_id = _canonical_id(raw_lane_id, label="lane_id")
        existing = casefold_bindings.setdefault(lane_id.casefold(), lane_id)
        if existing != lane_id:
            raise Stage6ContractError(
                f"case-only lane aliases are forbidden: {existing!r} versus {lane_id!r}"
            )
        if not lane_id or lane_id in seen:
            raise Stage6ContractError(f"duplicate or empty lane_id: {lane_id!r}")
        seen.add(lane_id)
        seed = deterministic_lane_seed(
            stage_id=stage_id, lane_id=lane_id, base_seed=base_seed
        )
        plan.append(
            {
                "stage_id": stage_id,
                "lane_id": lane_id,
                "seed": seed,
                "specification": lane,
                "checkpoint_identity": lane_identity(
                    stage_id=stage_id,
                    lane_id=lane_id,
                    seed=seed,
                    specification=lane,
                ),
            }
        )
    return sorted(plan, key=lambda row: row["lane_id"])


def iter_unfinished_lane_requests(
    plan: Iterable[Mapping[str, Any]],
    completion_receipts: Iterable[Mapping[str, Any]],
    *,
    checkpoint_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    plan_identities: set[str] = set()
    lane_ids: set[str] = set()
    lane_casefold_bindings: dict[str, str] = {}
    plan_stage_ids: set[str] = set()
    for raw_lane in plan:
        if not isinstance(raw_lane, Mapping) or set(raw_lane) != PLAN_FIELDS:
            raise Stage6ContractError("lane plan row schema is invalid")
        lane = dict(raw_lane)
        stage_id = _canonical_id(lane["stage_id"], label="plan.stage_id")
        lane_id = _canonical_id(lane["lane_id"], label="plan.lane_id")
        plan_stage_ids.add(stage_id)
        existing_lane_id = lane_casefold_bindings.setdefault(
            lane_id.casefold(), lane_id
        )
        if existing_lane_id != lane_id:
            raise Stage6ContractError(
                "lane plan contains case-only aliases: "
                f"{existing_lane_id!r} versus {lane_id!r}"
            )
        seed = lane["seed"]
        specification = lane["specification"]
        expected_identity = lane_identity(
            stage_id=stage_id,
            lane_id=lane_id,
            seed=seed,
            specification=specification,
        )
        identity = lane["checkpoint_identity"]
        if type(identity) is not str or identity != expected_identity:
            raise Stage6ContractError("lane checkpoint_identity is missing or invalid")
        if identity in plan_identities or lane_id in lane_ids:
            raise Stage6ContractError("lane plan contains duplicate identities")
        plan_identities.add(identity)
        lane_ids.add(lane_id)
        validated.append(lane)
    if len(plan_stage_ids) > 1:
        raise Stage6ContractError("lane plan mixes multiple stage identities")
    if isinstance(completion_receipts, (str, bytes, Mapping)) or not isinstance(
        completion_receipts, Iterable
    ):
        raise Stage6ContractError("completion receipts must be an iterable of objects")
    materialized_receipts = list(completion_receipts)
    completed_identities: set[str] = set()
    completed_lane_ids: set[str] = set()
    completed_checkpoint_paths: set[str] = set()
    resolved_checkpoint_root = checkpoint_root.resolve() if checkpoint_root else None
    for receipt in materialized_receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != COMPLETION_RECEIPT_FIELDS:
            raise Stage6ContractError("lane completion receipt schema is invalid")
        if (
            receipt.get("schema_version") != "stage6_lane_completion_receipt_v1"
            or receipt.get("status") != "DURABLE_READBACK_VERIFIED"
            or receipt.get("durable_readback_verified") is not True
        ):
            raise Stage6ContractError("lane completion receipt status is invalid")
        stage_id = _canonical_id(receipt.get("stage_id"), label="completion.stage_id")
        lane_id = _canonical_id(receipt.get("lane_id"), label="completion.lane_id")
        seed = receipt.get("seed")
        if type(seed) is not int or seed < 0:
            raise Stage6ContractError("completion seed must be a nonnegative integer")
        identity = receipt.get("checkpoint_identity")
        checkpoint_sha256 = receipt.get("checkpoint_sha256")
        if (
            type(identity) is not str
            or CHECKPOINT_PATTERN.fullmatch(identity) is None
            or type(checkpoint_sha256) is not str
            or CHECKPOINT_PATTERN.fullmatch(checkpoint_sha256) is None
        ):
            raise Stage6ContractError("completion checkpoint hashes are invalid")
        byte_count = receipt.get("checkpoint_bytes")
        if type(byte_count) is not int or byte_count <= 0:
            raise Stage6ContractError("completion checkpoint byte count is invalid")
        relative_path = receipt.get("checkpoint_relative_path")
        if type(relative_path) is not str or _normalized_member_name(relative_path) != relative_path:
            raise Stage6ContractError("completion checkpoint path is not canonical")
        expected_relative_path = f"checkpoints/{identity}.checkpoint"
        if relative_path != expected_relative_path:
            raise Stage6ContractError(
                "completion checkpoint path does not bind checkpoint_identity"
            )
        parse_utc_timestamp(
            receipt.get("completed_at_utc"), label=f"{lane_id}.completed_at_utc"
        )
        validate_document_self_hash(receipt)
        if (
            identity in completed_identities
            or lane_id in completed_lane_ids
            or relative_path.casefold() in completed_checkpoint_paths
        ):
            raise Stage6ContractError("duplicate lane completion receipt")
        completed_identities.add(identity)
        completed_lane_ids.add(lane_id)
        completed_checkpoint_paths.add(relative_path.casefold())
        matching_plan = next(
            (
                lane
                for lane in validated
                if lane["checkpoint_identity"] == identity
            ),
            None,
        )
        if matching_plan is None:
            raise Stage6ContractError(
                "completed checkpoint is outside the frozen lane plan"
            )
        if (
            matching_plan["stage_id"] != stage_id
            or matching_plan["lane_id"] != lane_id
            or matching_plan["seed"] != seed
        ):
            raise Stage6ContractError("completion receipt does not bind its lane plan")
        if resolved_checkpoint_root is None or not resolved_checkpoint_root.is_dir():
            raise Stage6ContractError(
                "durable completion receipts require an existing checkpoint root"
            )
        checkpoint_path = (
            resolved_checkpoint_root / Path(*relative_path.split("/"))
        ).resolve()
        try:
            checkpoint_path.relative_to(resolved_checkpoint_root)
        except ValueError as error:
            raise Stage6ContractError("checkpoint path escapes its frozen root") from error
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != byte_count
            or sha256_file(checkpoint_path) != checkpoint_sha256
        ):
            raise Stage6ContractError(
                f"{lane_id}: durable checkpoint byte readback failed"
            )
    stale = completed_identities - plan_identities
    if stale:
        raise Stage6ContractError(
            f"completed checkpoints are outside the frozen lane plan: {sorted(stale)[:5]}"
        )
    for lane in sorted(validated, key=lambda row: row["lane_id"]):
        if lane["checkpoint_identity"] not in completed_identities:
            yield lane
