from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


REVISION = "STAGE21-BASELINE-TIER64-EVIDENCE-R22-R18-CANDIDATE"
SCIENCE_SEEDS = (1729, 3253, 4933)
ALGORITHMS = ("cfr_plus", "dcfr")
STRATEGY_ABSTRACTION_VERSION = "strategy_perfect_recall_v5"
ROOTS_PER_PRODUCER = 8
WORK_LANES_PER_ROOT = len(SCIENCE_SEEDS) * len(ALGORITHMS)
WORK_LANES_PER_PRODUCER = ROOTS_PER_PRODUCER * WORK_LANES_PER_ROOT
# Zero is the released notebook's explicit Auto sentinel.  Logical lane
# membership remains 48; simultaneous worker processes resolve at runtime.
REQUESTED_WORKERS_DEFAULT = 0
QUALIFYING_SEEDS_PER_ALGORITHM = 2
# Approximate solving is expressed in preregistered tier units.  Tier 64 is
# the released default for all four stages.  Keeping the stage-specific base
# explicit lets a later, preregistered higher tier reuse the same solver state
# without changing any poker, traversal, RNG, or qualification semantics.
APPROXIMATE_BASE_ITERATIONS_BY_STAGE = {
    "river": 60_000,
    "turn": 80_000,
    "flop": 15_000,
    "preflop": 15_000,
}
APPROXIMATE_DEFAULT_TIER = 64
APPROXIMATE_FIRST_QUALIFICATION_TIER = 16
APPROXIMATE_QUALIFICATION_TIER_STRIDE = 16
APPROXIMATE_MAXIMUM_QUALIFICATION_LOOKS = 8
APPROXIMATE_ALLOWED_TIERS = (
    16,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
)
APPROXIMATE_TERMINAL_ITERATIONS_BY_STAGE = {
    stage: base * APPROXIMATE_DEFAULT_TIER
    for stage, base in APPROXIMATE_BASE_ITERATIONS_BY_STAGE.items()
}
APPROXIMATE_CHECKPOINT_INTERVAL_BY_STAGE = {
    # Operator-selected durable cadence.  The runtime always adds an exact
    # final-target checkpoint when the tier target is not evenly divisible.
    "river": 250_000,
    "turn": 400_000,
    "flop": 40_000,
    "preflop": 40_000,
}
EVALUATION_DEALS_PER_FAMILY_ROLE = 1_024
EVALUATION_ALLOWED_DEALS_PER_FAMILY_ROLE = (1_024, 2_048, 3_072, 4_096)
RESTRICTED_EXACT_TERMINAL_ITERATIONS = 256
RESTRICTED_EXACT_CHECKPOINT_INTERVAL_ITERATIONS = 64
CHECKPOINT_GENERATIONS_RETAINED = 1
PROGRESS_HEARTBEAT_SECONDS = 60
STALL_WARNING_SECONDS = 180
CROSSPLAY_DEALS_PER_ASSIGNMENT = 1_024
CROSSPLAY_FAMILYWISE_ALPHA = 0.05


@dataclass(frozen=True)
class StreamedJSONMapping:
    """A canonical mapping whose members must be emitted incrementally."""

    items: Iterable[tuple[str, Any]]


@dataclass(frozen=True)
class StreamedJSONSequence:
    """A canonical JSON sequence whose items must be emitted incrementally."""

    items: Iterable[Any]


CROSSPLAY_LIMIT_EFFECTIVE_STACK_FRACTION_PER_HAND = 0.05
CROSSPLAY_MAXIMUM_MISSING_POLICY_REACH_FRACTION = 0.01
CROSSPLAY_PAIRING_UNIT = "same_deal_average_over_both_seat_assignments"


@dataclass(frozen=True)
class StageDefinition:
    number: int
    token: str
    street: str
    root_count: int
    approximate_count: int
    restricted_exact_count: int
    forced_count: int
    depth_limit: str | None
    predecessor: str | None

    def validate(self) -> None:
        if self.root_count != (
            self.approximate_count + self.restricted_exact_count + self.forced_count
        ):
            raise ValueError(f"{self.token}: root partition does not sum")
        if self.street in {"river", "turn"} and self.depth_limit is not None:
            raise ValueError(f"{self.token}: River and Turn cannot be depth limited")
        if self.street == "flop" and self.depth_limit != "turn":
            raise ValueError("Flop must terminate at accepted Turn continuation values")
        if self.street == "preflop" and self.depth_limit != "flop":
            raise ValueError("Preflop must terminate at accepted Flop continuation values")


STAGES = {
    "river": StageDefinition(1, "S1", "river", 128, 104, 16, 8, None, None),
    "turn": StageDefinition(2, "S2", "turn", 192, 176, 8, 8, None, "river"),
    "flop": StageDefinition(3, "S3", "flop", 192, 184, 0, 8, "turn", "turn"),
    "preflop": StageDefinition(4, "S4", "preflop", 128, 120, 0, 8, "flop", "flop"),
}
for _stage in STAGES.values():
    _stage.validate()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def with_self_hash(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("self_sha256", None)
    result["self_sha256"] = sha256_json(result)
    return result


def self_hashed_json_bytes(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Build a self-hashed document and canonical bytes in one value pass.

    ``canonical_json_bytes(with_self_hash(document))`` serializes every nested
    value twice: once to calculate the body hash and again to produce the file
    bytes. Checkpoints and lane results can contain very large policy/node
    mappings, so the duplicate traversal is material. This helper serializes
    each top-level value once, streams those exact fragments into the body
    hash, and reuses them when it inserts the ``self_sha256`` member.

    The returned bytes are byte-for-byte identical to the original two-pass
    expression. Non-string top-level keys use the conservative legacy path;
    Stage 21 contracts use string keys exclusively.
    """

    body = dict(document)
    body.pop("self_sha256", None)
    if not all(isinstance(key, str) for key in body):
        result = with_self_hash(body)
        return result, canonical_json_bytes(result)

    serialized_items = [
        (
            key,
            canonical_json_bytes(key) + b":" + canonical_json_bytes(value),
        )
        for key, value in sorted(body.items())
    ]
    digest = hashlib.sha256()
    digest.update(b"{")
    for index, (_, fragment) in enumerate(serialized_items):
        if index:
            digest.update(b",")
        digest.update(fragment)
    digest.update(b"}")
    self_sha256 = digest.hexdigest()

    result = dict(body)
    result["self_sha256"] = self_sha256
    self_fragment = (
        canonical_json_bytes("self_sha256")
        + b":"
        + canonical_json_bytes(self_sha256)
    )
    final_fragments = [
        fragment
        for _, fragment in sorted(
            [*serialized_items, ("self_sha256", self_fragment)],
            key=lambda item: item[0],
        )
    ]
    return result, b"{" + b",".join(final_fragments) + b"}"


def validate_self_hash(document: Mapping[str, Any]) -> None:
    expected = document.get("self_sha256")
    body = dict(document)
    body.pop("self_sha256", None)
    if not isinstance(expected, str) or expected != sha256_json(body):
        raise ValueError("self hash mismatch")


def release_runtime_caches() -> dict[str, Any]:
    """Release bounded pure-function caches before large result serialization.

    Each solver process serves exactly one lane.  The caches accelerate its
    solver/evaluation work but are no longer useful once the lane payload is
    complete; clearing them here prevents cached states from overlapping the
    peak memory required to serialize and atomically verify that payload.
    """

    from . import cards, chance, coverage, evaluation, game, ranges

    cached_functions = {
        "cards.five_card_rank": cards._five_card_rank_cached,
        "cards.best_hand_rank": cards._best_hand_rank_cached,
        "chance.legal_next_cards": chance._legal_next_cards_cached,
        "chance.preflop_remaining_cards": chance._remaining_preflop_cards,
        "coverage.combo_feature_record": coverage._combo_feature_record_cached,
        "evaluation.family_policy": evaluation._family_policy_cached,
        "game.deal_validation": game._validate_deal,
        "game.state_validation": game.HoldemState.validate,
        "game.history_replay": game._history_replay_contexts,
        "game.legal_actions": game.legal_actions,
        "game.public_transition": game.advance_public_card,
        "game.terminal_settlement": game.terminal_settlement,
        "game.exact_information_set": game._information_set_key_cached,
        "game.strategy_action_labels": game._strategy_action_labels_cached,
        "game.strategy_history": game._strategy_history_key,
        "game.public_board_key": game._canonical_public_board_key_cached,
        "game.strategy_hand_key": game._strategy_hand_key,
        "game.strategy_board_key": game._strategy_board_key,
        "game.strategy_information_set": game._strategy_information_set_key_cached,
        "game.strategy_backoff_keys": game.strategy_backoff_keys,
        "ranges.hand_features": ranges._cached_hand_features,
    }
    before = {}
    for name, function in cached_functions.items():
        info = function.cache_info()
        before[name] = {
            "hits": int(info.hits),
            "misses": int(info.misses),
            "maxsize": info.maxsize,
            "currsize": int(info.currsize),
        }
    for function in cached_functions.values():
        function.cache_clear()
    return {
        "schema_version": "stage21_runtime_cache_release_v2",
        "cache_count": len(cached_functions),
        "entries_released": sum(row["currsize"] for row in before.values()),
        "cache_info_before_release": before,
        "released_before_large_lane_result_serialization": True,
    }


def chunks(values: Sequence[str], size: int = ROOTS_PER_PRODUCER) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def scientific_lane_plan(root_ids: Sequence[str]) -> list[dict[str, Any]]:
    if len(root_ids) != ROOTS_PER_PRODUCER:
        raise ValueError("a scientific producer must contain exactly eight roots")
    rows: list[dict[str, Any]] = []
    for root_id in root_ids:
        for algorithm in ALGORITHMS:
            for seed in SCIENCE_SEEDS:
                rows.append(
                    {
                        "lane_id": f"{root_id}__{algorithm}__seed_{seed}",
                        "root_id": root_id,
                        "algorithm": algorithm,
                        "seed": seed,
                        "role": "qualification_candidate",
                    }
                )
    if len(rows) != WORK_LANES_PER_PRODUCER:
        raise AssertionError("scientific lane construction is not 48 lanes")
    return rows


def evaluation_lane_plan(root_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Build 48 independent evaluation shards for eight forced roots.

    Forced roots have no strategy decision, so labeling their work as CFR or
    DCFR would be scientifically false. Six pre-registered deal-panel shards
    per root retain the same 48-lane runtime shape without fabricating solver
    iterations.
    """

    if len(root_ids) != ROOTS_PER_PRODUCER:
        raise ValueError("a forced-root producer must contain exactly eight roots")
    rows = [
        {
            "lane_id": f"{root_id}__evaluation_panel_{panel:02d}",
            "root_id": root_id,
            "algorithm": None,
            "seed": SCIENCE_SEEDS[panel % len(SCIENCE_SEEDS)] + panel * 100_003,
            "panel_index": panel,
            "role": "forced_runout_evaluation",
        }
        for root_id in root_ids
        for panel in range(WORK_LANES_PER_ROOT)
    ]
    if len(rows) != WORK_LANES_PER_PRODUCER:
        raise AssertionError("evaluation lane construction is not 48 lanes")
    return rows


def lane_plan(root_ids: Sequence[str], *, forced: bool = False) -> list[dict[str, Any]]:
    return evaluation_lane_plan(root_ids) if forced else scientific_lane_plan(root_ids)


def qualify_root(
    root_id: str,
    lane_results: Sequence[Mapping[str, Any]],
    *,
    root_kind: str = "approximate",
    aggregate_root_evidence: Mapping[str, Any] | None = None,
    require_aggregate_evidence: bool = False,
    expected_effective_stack_bb: float | None = None,
    validated_crossplay_receipt: Any | None = None,
) -> dict[str, Any]:
    """Adjudicate one root with complete six-member authority.

    Every planned member must be present, hash-valid, and independently
    evaluated. Approximate roots require two stochastic seeds per algorithm.
    Restricted full-chance roots use the same numeric IDs only as deterministic
    accumulation-order stability checks and never call them independent solves.
    """

    relevant = [row for row in lane_results if row.get("root_id") == root_id]
    expected = {
        (algorithm, seed) for algorithm in ALGORITHMS for seed in SCIENCE_SEEDS
    }
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in relevant:
        key = (str(row.get("algorithm")), int(row.get("seed", -1)))
        if key in observed:
            duplicates.append(key)
        observed[key] = row
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    hash_invalid = sorted(
        key for key, row in observed.items() if row.get("artifact_hash_valid") is not True
    )
    panel_invalid = sorted(
        key
        for key, row in observed.items()
        if row.get("independent_confirmation_panel") is not True
    )
    pass_seeds = {
        algorithm: sorted(
            seed
            for (alg, seed), row in observed.items()
            if alg == algorithm
            and row.get("qualified") is True
            and row.get("artifact_hash_valid") is True
            and row.get("independent_confirmation_panel") is True
        )
        for algorithm in ALGORITHMS
    }
    complete = not (missing or unexpected or duplicates or hash_invalid or panel_invalid)
    member_gate = complete and all(
        len(pass_seeds[algorithm]) >= QUALIFYING_SEEDS_PER_ALGORITHM
        for algorithm in ALGORITHMS
    )
    crossplay_validation_error: str | None = None
    validated_crossplay_self_sha256: str | None = None
    if root_kind == "restricted_exact":
        aggregate_gate = (
            aggregate_root_evidence is not None
            and aggregate_root_evidence.get("root_kind") == "restricted_exact"
            and aggregate_root_evidence.get("numerical_order_stability", {}).get(
                "gate_pass"
            )
            is True
        )
        replication_semantics = (
            "deterministic_full_chance_numerical_order_stability_not_independent_seeds"
        )
    else:
        aggregate_gate = False
        if aggregate_root_evidence is not None:
            try:
                if aggregate_root_evidence.get("root_kind") != "approximate":
                    raise ValueError("aggregate evidence has the wrong root kind")
                crossplay = aggregate_root_evidence.get("crossplay", {})
                if validated_crossplay_receipt is not None:
                    from .crossplay import ValidatedCrossplayReceipt

                    if not isinstance(
                        validated_crossplay_receipt, ValidatedCrossplayReceipt
                    ) or not validated_crossplay_receipt.matches(
                        root_id=root_id,
                        evidence_self_sha256=str(crossplay.get("self_sha256", "")),
                        members=relevant,
                    ):
                        raise ValueError("validated cross-play receipt binding mismatch")
                    aggregate_gate = validated_crossplay_receipt.gate_pass is True
                    validated_crossplay_self_sha256 = (
                        validated_crossplay_receipt.evidence_self_sha256
                    )
                else:
                    validate_self_hash(aggregate_root_evidence)
                    if expected_effective_stack_bb is None:
                        raise ValueError("expected effective stack is required")
                    from .crossplay import validate_and_recompute_root_crossplay_evidence

                    validated = validate_and_recompute_root_crossplay_evidence(
                        crossplay,
                        expected_root_id=root_id,
                        expected_members=relevant,
                        deals_per_assignment=CROSSPLAY_DEALS_PER_ASSIGNMENT,
                        familywise_alpha=CROSSPLAY_FAMILYWISE_ALPHA,
                        effective_stack_bb=float(expected_effective_stack_bb),
                        limit_effective_stack_fraction_per_hand=(
                            CROSSPLAY_LIMIT_EFFECTIVE_STACK_FRACTION_PER_HAND
                        ),
                        maximum_missing_policy_reach_fraction=(
                            CROSSPLAY_MAXIMUM_MISSING_POLICY_REACH_FRACTION
                        ),
                        pairing_unit=CROSSPLAY_PAIRING_UNIT,
                    )
                    aggregate_gate = validated["gate_pass"] is True
                    validated_crossplay_self_sha256 = validated["self_sha256"]
            except (KeyError, TypeError, ValueError) as error:
                crossplay_validation_error = str(error)
        replication_semantics = "independent_stochastic_solver_seed"
    accepted = member_gate and (
        aggregate_gate if require_aggregate_evidence else True
    )
    return with_self_hash(
        {
            "schema_version": "stage21_root_qualification_v5",
            "root_id": root_id,
            "expected_member_count": WORK_LANES_PER_ROOT,
            "observed_member_count": len(relevant),
            "complete_member_authority": complete,
            "missing_members": [list(item) for item in missing],
            "unexpected_members": [list(item) for item in unexpected],
            "duplicate_members": [list(item) for item in sorted(set(duplicates))],
            "hash_invalid_members": [list(item) for item in hash_invalid],
            "panel_invalid_members": [list(item) for item in panel_invalid],
            "passing_seeds": pass_seeds,
            "passing_numerical_order_ids": pass_seeds
            if root_kind == "restricted_exact"
            else None,
            "minimum_passing_seeds_per_algorithm": QUALIFYING_SEEDS_PER_ALGORITHM,
            "replication_semantics": replication_semantics,
            "independent_random_seed_claim": root_kind != "restricted_exact",
            "aggregate_evidence_required": require_aggregate_evidence,
            "aggregate_evidence_gate_pass": aggregate_gate
            if require_aggregate_evidence
            else None,
            "crossplay_authority_recomputed": (
                validated_crossplay_self_sha256 is not None
                if root_kind == "approximate" and require_aggregate_evidence
                else None
            ),
            "validated_crossplay_self_sha256": validated_crossplay_self_sha256,
            "crossplay_validation_error": crossplay_validation_error,
            "aggregate_root_evidence_self_sha256": aggregate_root_evidence.get(
                "self_sha256"
            )
            if aggregate_root_evidence is not None
            else None,
            "accepted": accepted,
        }
    )


def deterministic_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            if name.startswith("/") or "\\" in name or ".." in Path(name).parts:
                raise ValueError(f"unsafe archive member: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, members[name])
    return output.getvalue()


def atomic_write(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    readback = path.read_bytes()
    if readback != raw:
        raise IOError(f"readback mismatch: {path}")
    return {"path": str(path), "size": len(raw), "sha256": sha256_bytes(raw)}


def atomic_install_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    """Fsync and atomically install bytes without a duplicate full readback.

    Callers that use this lower-level primitive must perform their own
    independent post-replace validation.  It exists for large lane payloads,
    where :func:`atomic_write` followed by a JSON validation would otherwise
    read the complete file twice.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
        return {"path": str(path), "size": len(raw), "sha256": sha256_bytes(raw)}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def stream_file_identity(path: Path, *, block_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a bounded-memory file identity."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "size": size, "sha256": digest.hexdigest()}


def stage_file_once(
    source: Path,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    block_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Copy one immutable input to local scratch while hashing that same pass.

    The destination is atomically installed only after the streamed size and
    SHA-256 match any external contract.  Callers must parse/decompress the
    installed local bytes, never reopen the mounted source.
    """

    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("single-pass staging source and destination are identical")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary: Path | None = None
    try:
        with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as target_handle:
            temporary = Path(target_handle.name)
            for block in iter(lambda: source_handle.read(block_size), b""):
                target_handle.write(block)
                digest.update(block)
                size += len(block)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        sha256 = digest.hexdigest()
        if expected_size is not None and size != int(expected_size):
            raise ValueError("staged source size does not match immutable contract")
        if expected_sha256 is not None and sha256 != str(expected_sha256):
            raise ValueError("staged source SHA-256 does not match immutable contract")
        temporary.replace(destination)
        temporary = None
        stat = destination.stat()
        if stat.st_size != size:
            raise IOError("staged local source size changed after atomic install")
        return {
            "source_path": str(source),
            "path": str(destination),
            "size": size,
            "sha256": sha256,
            "source_stream_read_count": 1,
            "source_hash_and_copy_same_pass": True,
            "parse_must_use_staged_path": True,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_ephemeral_local_path(
    path: Path,
    *,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    """Resolve a scratch path and reject mounted-Drive or durable descendants."""

    resolved = path.resolve()
    normalized = resolved.as_posix().lower().rstrip("/")
    if normalized == "/content/drive" or normalized.startswith("/content/drive/"):
        raise ValueError(f"ephemeral scratch cannot resolve under mounted Drive: {resolved}")
    for root in forbidden_roots:
        durable = root.resolve()
        if resolved == durable or durable in resolved.parents:
            raise ValueError(f"ephemeral scratch overlaps durable state: {resolved}")
    return resolved


class _IncrementalJSONReader:
    """Small canonical-JSON pull reader that never buffers a streamed mapping."""

    def __init__(self, path: Path, *, chunk_chars: int = 1024 * 1024) -> None:
        # Stage 21 documents are canonical JSON produced with ``ensure_ascii``.
        # Reading bytes here lets the parser bind the exact raw-file digest to
        # the very same pass that supplies parsed content.  A previous
        # hash-then-reopen implementation left a mounted-Drive TOCTOU window
        # and doubled large-file I/O.
        self._path = path
        self._handle = path.open("rb")
        self._chunk_chars = max(4096, int(chunk_chars))
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._decoder = json.JSONDecoder()
        self._raw_digest = hashlib.sha256()
        self._raw_size = 0

    def close(self) -> None:
        self._handle.close()

    def _compact(self) -> None:
        if self._position:
            self._buffer = self._buffer[self._position :]
            self._position = 0

    def _fill(self) -> bool:
        if self._eof:
            return False
        raw = self._handle.read(self._chunk_chars)
        if not raw:
            self._eof = True
            return False
        self._raw_digest.update(raw)
        self._raw_size += len(raw)
        try:
            chunk = raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("streamed JSON is not canonical ASCII JSON") from error
        self._buffer += chunk
        return True

    def file_identity(self) -> dict[str, Any]:
        if not self._eof:
            raise RuntimeError("streamed JSON identity requested before EOF")
        return {
            "path": str(self._path),
            "size": self._raw_size,
            "sha256": self._raw_digest.hexdigest(),
        }

    def peek(self) -> str:
        self._compact()
        while not self._buffer and self._fill():
            pass
        if not self._buffer:
            raise ValueError("unexpected end of streamed JSON")
        return self._buffer[0]

    def expect(self, character: str) -> None:
        if len(character) != 1 or self.peek() != character:
            raise ValueError(f"streamed JSON expected {character!r}")
        self._position = 1

    def value(self) -> Any:
        self._compact()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, 0)
            except json.JSONDecodeError:
                if not self._fill():
                    raise ValueError("invalid or truncated streamed JSON")
                continue
            self._position = end
            return value

    def require_eof(self) -> None:
        self._compact()
        while self._fill():
            pass
        if self._buffer:
            raise ValueError("trailing bytes after streamed JSON document")


def read_self_hashed_json_streamed_mapping(
    path: Path,
    *,
    streamed_mapping_key: str,
    on_member: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Validate canonical self-hash while materializing one mapping member at a time."""

    reader = _IncrementalJSONReader(path)
    body_digest = hashlib.sha256()
    body_digest.update(b"{")
    body_member_count = 0
    streamed_member_count = 0
    streamed_mapping_found = False
    streamed_mapping_digest = hashlib.sha256()
    metadata: dict[str, Any] = {}
    claimed_self_hash: str | None = None
    previous_top_key: str | None = None

    def emit_body_member(key: str, value_writer: Callable[[], None]) -> None:
        nonlocal body_member_count
        if body_member_count:
            body_digest.update(b",")
        body_digest.update(canonical_json_bytes(key))
        body_digest.update(b":")
        value_writer()
        body_member_count += 1

    try:
        reader.expect("{")
        first_top = True
        while reader.peek() != "}":
            if not first_top:
                reader.expect(",")
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError("streamed JSON top-level key is not a string")
            if previous_top_key is not None and key <= previous_top_key:
                raise ValueError("streamed JSON top-level keys are not canonical")
            previous_top_key = key
            reader.expect(":")
            if key == streamed_mapping_key:
                if streamed_mapping_found:
                    raise ValueError("streamed mapping appears more than once")
                streamed_mapping_found = True
                if claimed_self_hash is not None:
                    raise ValueError("streamed mapping appears after self hash unexpectedly")

                def emit_mapping() -> None:
                    nonlocal streamed_member_count
                    body_digest.update(b"{")
                    streamed_mapping_digest.update(b"{")
                    reader.expect("{")
                    first_member = True
                    previous_member_key: str | None = None
                    while reader.peek() != "}":
                        if not first_member:
                            reader.expect(",")
                            body_digest.update(b",")
                            streamed_mapping_digest.update(b",")
                        member_key = reader.value()
                        if not isinstance(member_key, str):
                            raise ValueError("streamed mapping key is not a string")
                        if (
                            previous_member_key is not None
                            and member_key <= previous_member_key
                        ):
                            raise ValueError("streamed mapping keys are not canonical")
                        previous_member_key = member_key
                        reader.expect(":")
                        member_value = reader.value()
                        if not isinstance(member_value, Mapping):
                            raise ValueError("streamed mapping member is not an object")
                        key_raw = canonical_json_bytes(member_key)
                        value_raw = canonical_json_bytes(member_value)
                        body_digest.update(key_raw)
                        body_digest.update(b":")
                        body_digest.update(value_raw)
                        streamed_mapping_digest.update(key_raw)
                        streamed_mapping_digest.update(b":")
                        streamed_mapping_digest.update(value_raw)
                        on_member(member_key, member_value)
                        streamed_member_count += 1
                        first_member = False
                    reader.expect("}")
                    body_digest.update(b"}")
                    streamed_mapping_digest.update(b"}")

                emit_body_member(key, emit_mapping)
            else:
                value = reader.value()
                if key == "self_sha256":
                    if not isinstance(value, str) or len(value) != 64:
                        raise ValueError("streamed JSON has an invalid self hash")
                    claimed_self_hash = value
                else:
                    metadata[key] = value
                    emit_body_member(
                        key,
                        lambda value=value: body_digest.update(canonical_json_bytes(value)),
                    )
            first_top = False
        reader.expect("}")
        reader.require_eof()
    finally:
        reader.close()
    body_digest.update(b"}")
    if claimed_self_hash != body_digest.hexdigest():
        raise ValueError("streamed JSON self hash mismatch")
    identity = reader.file_identity()
    if streamed_mapping_found:
        # Preserve absence exactly for forced/aggregate documents that do not
        # have this mapping at all.  Callers replace this placeholder with a
        # disk-backed mapping only when the original member was present.
        metadata[streamed_mapping_key] = None
    metadata["self_sha256"] = claimed_self_hash
    return {
        "document": metadata,
        "file_identity": identity,
        "streamed_mapping_key": streamed_mapping_key,
        "streamed_member_count": streamed_member_count,
        "streamed_mapping_found": streamed_mapping_found,
        "streamed_mapping_sha256": (
            streamed_mapping_digest.hexdigest()
            if streamed_mapping_found
            else None
        ),
        "canonical_self_hash_verified": True,
        "complete_document_materialized": False,
    }


@contextmanager
def checkpoint_io_admission(
    lock_path: Path,
    *,
    timeout_seconds: float = 1_800.0,
    stale_seconds: float = 3_600.0,
) -> Iterator[dict[str, Any]]:
    """Serialize large checkpoint publications across sibling lane processes.

    This sibling-process admission lock is deliberately local. Token-level
    durable ownership prevents a second Colab host from reaching checkpoint
    publication; this lock only bounds complete local checkpoint spools to one.
    """

    lock_path = require_ephemeral_local_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    attempts = 0
    owner_uuid = os.urandom(16).hex()
    backoff_seconds = 0.05
    while True:
        attempts += 1
        try:
            descriptor = os.open(
                str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            reclaim = False
            try:
                age = max(0.0, time.time() - lock_path.stat().st_mtime)
                owner = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(owner.get("pid", -1))
                alive = owner_pid > 0
                if alive:
                    try:
                        os.kill(owner_pid, 0)
                    except OSError:
                        alive = False
                reclaim = age >= stale_seconds and not alive
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A malformed owner record is ambiguous.  Never risk two
                # concurrent Drive publications merely because it is old;
                # wait and fail closed at the bounded timeout instead.
                reclaim = False
            if reclaim:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(
                    f"checkpoint I/O admission timed out: {lock_path}"
                )
            time.sleep(backoff_seconds)
            backoff_seconds = min(2.0, backoff_seconds * 1.7)
            continue
        else:
            try:
                payload = canonical_json_bytes(
                    {
                        "pid": os.getpid(),
                        "owner_uuid": owner_uuid,
                        "created_unix_seconds": time.time(),
                    }
                )
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if lock_path.read_bytes() != payload:
                raise IOError("local checkpoint admission lock readback mismatch")
            break
    receipt = {
        "queue_wait_seconds": time.monotonic() - started,
        "admission_attempts": attempts,
        "lock_path": str(lock_path),
        "owner_uuid": owner_uuid,
        "bounded_spool_admission_count": 1,
        "bounded_exponential_backoff": True,
    }
    try:
        yield receipt
    finally:
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            owner = None
        if owner is not None:
            if owner.get("owner_uuid") != owner_uuid:
                raise RuntimeError("checkpoint admission lock owner changed before release")
            lock_path.unlink()


def _copy_stream(
    source: Path,
    destination,
    *,
    digest: Any | None = None,
    block_size: int = 1024 * 1024,
) -> int:
    size = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            destination.write(block)
            if digest is not None:
                digest.update(block)
            size += len(block)
    return size


def write_self_hashed_json_stream(
    path: Path,
    *,
    document: Mapping[str, Any],
    streamed_mappings: Mapping[str, Iterable[tuple[str, Any]]],
    streamed_sequences: Mapping[str, Iterable[Any]] | None = None,
    binding_fields: Sequence[str] = (),
    io_lock_path: Path | None = None,
    spool_root: Path | None = None,
) -> dict[str, Any]:
    """Write a canonical self-hashed JSON object with bounded peak memory.

    Large top-level mappings and sequences are serialized one member at a time
    into local spools.  The body hash and final document reuse those spools, so
    no full node mapping or full checkpoint byte string is materialized.  After atomic
    replacement, one stream pass independently verifies both the file identity
    and the embedded self hash, plus exact small binding fragments.
    """

    if io_lock_path is not None:
        with checkpoint_io_admission(io_lock_path) as admission:
            result = write_self_hashed_json_stream(
                path,
                document=document,
                streamed_mappings=streamed_mappings,
                streamed_sequences=streamed_sequences,
                binding_fields=binding_fields,
                io_lock_path=None,
                spool_root=spool_root,
            )
            result["io_admission_queue_wait_seconds"] = admission[
                "queue_wait_seconds"
            ]
            result["io_admission_attempts"] = admission["admission_attempts"]
            result["checkpoint_spool_admission_limit"] = admission[
                "bounded_spool_admission_count"
            ]
            result["checkpoint_spool_admission_is_local"] = True
            return result

    resolved_spool_root = (
        require_ephemeral_local_path(spool_root)
        if spool_root is not None
        else Path(tempfile.gettempdir()).resolve()
    )
    resolved_spool_root.mkdir(parents=True, exist_ok=True)
    body = dict(document)
    body.pop("self_sha256", None)
    sequence_values = {} if streamed_sequences is None else dict(streamed_sequences)
    overlap = (set(body) & set(streamed_mappings)) | (
        set(body) & set(sequence_values)
    ) | (set(streamed_mappings) & set(sequence_values))
    if overlap:
        raise ValueError(f"streamed values duplicate document keys: {sorted(overlap)}")
    if not all(
        isinstance(key, str)
        for key in (*body.keys(), *streamed_mappings.keys(), *sequence_values.keys())
    ):
        raise TypeError("streamed self-hashed JSON requires string top-level keys")

    serialization_started = time.monotonic()
    spool_paths: dict[str, Path] = {}
    small_fragments: dict[str, bytes] = {}
    temporary: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for key, value in body.items():
            small_fragments[key] = canonical_json_bytes(key) + b":" + canonical_json_bytes(value)
        def write_value(output, value: Any) -> None:
            if isinstance(value, StreamedJSONSequence):
                output.write(b"[")
                for index, nested_value in enumerate(value.items):
                    if index:
                        output.write(b",")
                    write_value(output, nested_value)
                output.write(b"]")
                return
            if not isinstance(value, StreamedJSONMapping):
                output.write(canonical_json_bytes(value))
                return
            output.write(b"{")
            previous_nested: str | None = None
            nested_count = 0
            for nested_key, nested_value in value.items:
                nested_key = str(nested_key)
                if previous_nested is not None and nested_key <= previous_nested:
                    raise ValueError("nested streamed mapping is not strictly sorted")
                if nested_count:
                    output.write(b",")
                output.write(canonical_json_bytes(nested_key))
                output.write(b":")
                write_value(output, nested_value)
                previous_nested = nested_key
                nested_count += 1
            output.write(b"}")

        for top_key, items in streamed_mappings.items():
            with tempfile.NamedTemporaryFile(
                dir=resolved_spool_root, delete=False
            ) as spool:
                spool_path = Path(spool.name)
                spool.write(b"{")
                previous: str | None = None
                count = 0
                for member_key, member_value in items:
                    member_key = str(member_key)
                    if previous is not None and member_key <= previous:
                        raise ValueError(
                            f"streamed mapping {top_key} is not strictly sorted"
                        )
                    if count:
                        spool.write(b",")
                    spool.write(canonical_json_bytes(member_key))
                    spool.write(b":")
                    write_value(spool, member_value)
                    previous = member_key
                    count += 1
                spool.write(b"}")
                spool.flush()
            spool_paths[top_key] = spool_path
        for top_key, items in sequence_values.items():
            with tempfile.NamedTemporaryFile(
                dir=resolved_spool_root, delete=False
            ) as spool:
                spool_path = Path(spool.name)
                write_value(spool, StreamedJSONSequence(items))
                spool.flush()
            spool_paths[top_key] = spool_path

        top_keys = sorted([*small_fragments, *spool_paths])
        body_digest = hashlib.sha256()
        body_digest.update(b"{")
        for index, key in enumerate(top_keys):
            if index:
                body_digest.update(b",")
            prefix = canonical_json_bytes(key) + b":"
            if key in small_fragments:
                body_digest.update(small_fragments[key])
            else:
                body_digest.update(prefix)
                with spool_paths[key].open("rb") as spool:
                    for block in iter(lambda: spool.read(1024 * 1024), b""):
                        body_digest.update(block)
        body_digest.update(b"}")
        self_sha256 = body_digest.hexdigest()
        self_fragment = (
            canonical_json_bytes("self_sha256")
            + b":"
            + canonical_json_bytes(self_sha256)
        )
        final_keys = sorted([*top_keys, "self_sha256"])
        self_index = final_keys.index("self_sha256")
        if self_index == 0:
            raise ValueError("streamed document requires a key sorted before self_sha256")
        serialization_seconds = time.monotonic() - serialization_started

        def publish() -> dict[str, Any]:
            nonlocal temporary
            write_started = time.monotonic()
            output_digest = hashlib.sha256()
            field_offsets: dict[str, tuple[int, bytes]] = {}
            self_omit_start = -1
            self_omit_end = -1
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)

                def emit(raw: bytes) -> None:
                    output.write(raw)
                    output_digest.update(raw)

                emit(b"{")
                for index, key in enumerate(final_keys):
                    if index:
                        comma_offset = output.tell()
                        emit(b",")
                    else:
                        comma_offset = -1
                    fragment_offset = output.tell()
                    if key == "self_sha256":
                        self_omit_start = comma_offset
                        emit(self_fragment)
                        self_omit_end = output.tell()
                        fragment = self_fragment
                    elif key in small_fragments:
                        fragment = small_fragments[key]
                        emit(fragment)
                    else:
                        prefix = canonical_json_bytes(key) + b":"
                        emit(prefix)
                        _copy_stream(spool_paths[key], output, digest=output_digest)
                        fragment = b""
                    if key in binding_fields:
                        if not fragment:
                            raise ValueError("binding fields cannot be streamed values")
                        field_offsets[key] = (fragment_offset, fragment)
                emit(b"}")
                fsync_started = time.monotonic()
                output.flush()
                os.fsync(output.fileno())
                fsync_seconds = time.monotonic() - fsync_started
                written_size = output.tell()
            written_sha256 = output_digest.hexdigest()
            publication_started = time.monotonic()
            temporary.replace(path)
            temporary = None
            publication_seconds = time.monotonic() - publication_started
            write_seconds = time.monotonic() - write_started

            readback_started = time.monotonic()
            final_digest = hashlib.sha256()
            readback_body_digest = hashlib.sha256()
            position = 0
            with path.open("rb") as readback:
                for block in iter(lambda: readback.read(1024 * 1024), b""):
                    final_digest.update(block)
                    block_start = position
                    block_end = position + len(block)
                    if block_end <= self_omit_start or block_start >= self_omit_end:
                        readback_body_digest.update(block)
                    else:
                        if block_start < self_omit_start:
                            readback_body_digest.update(
                                block[: self_omit_start - block_start]
                            )
                        if block_end > self_omit_end:
                            readback_body_digest.update(
                                block[self_omit_end - block_start :]
                            )
                    position = block_end
                for field, (offset, expected_fragment) in field_offsets.items():
                    readback.seek(offset)
                    if readback.read(len(expected_fragment)) != expected_fragment:
                        raise IOError(f"streamed JSON binding readback mismatch: {field}")
            readback_seconds = time.monotonic() - readback_started
            if position != written_size or final_digest.hexdigest() != written_sha256:
                raise IOError(f"streamed JSON file identity mismatch: {path}")
            if readback_body_digest.hexdigest() != self_sha256:
                raise IOError(f"streamed JSON self hash readback mismatch: {path}")
            return {
                "path": str(path),
                "size": written_size,
                "sha256": written_sha256,
                "self_sha256": self_sha256,
                "serialization_seconds": serialization_seconds,
                "write_and_fsync_seconds": write_seconds,
                "fsync_seconds": fsync_seconds,
                "atomic_publication_seconds": publication_seconds,
                "independent_stream_readback_seconds": readback_seconds,
                "independent_stream_file_identity_verified": True,
                "independent_stream_self_hash_verified": True,
                "binding_fields_readback_verified": sorted(field_offsets),
                "full_checkpoint_bytes_materialized_in_memory": False,
                "full_document_bytes_materialized_in_memory": False,
            }

        result = publish()
        result["io_admission_queue_wait_seconds"] = 0.0
        result["io_admission_attempts"] = 0
        result["stream_spool_root"] = str(resolved_spool_root)
        return result
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        for spool_path in spool_paths.values():
            try:
                spool_path.unlink()
            except FileNotFoundError:
                pass


def member_manifest(members: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "size": len(raw), "sha256": sha256_bytes(raw)}
        for name, raw in sorted(members.items())
    ]


def require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValueError(f"{label} missing keys: {missing}")
