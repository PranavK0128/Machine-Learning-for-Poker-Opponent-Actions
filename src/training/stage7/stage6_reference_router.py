from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from stage6_common import Stage6ContractError, canonical_json_bytes


ABSTRACTION_ID = "stackhud_stage050b_postflop_cbet_raise_overbet_bucket_v1"
SCHEMA_ID = "stackhud_reference_row_v5_stage050c"
TABLE_FORMAT = "heads_up"
STREET = "flop_proxy"
POSITION_TO_ACTOR = {"IP_CBETTOR": "p0", "OOP_DEFENDER": "p1"}
REFERENCE_ROW_FIELDS = {
    "abstraction_id",
    "artifact_path",
    "automatic_pokernow_actions_allowed",
    "board_texture",
    "convergence_metrics",
    "created_at",
    "display_allowed",
    "game_name",
    "game_variant",
    "generator_algorithm",
    "generator_version",
    "hand_bucket",
    "history",
    "information_state_key",
    "legal_action_validation",
    "legal_actions",
    "license_id",
    "position",
    "primary_action",
    "primary_action_probability",
    "probability_sum_validation",
    "production_writes_allowed",
    "promotion_allowed",
    "reference_method",
    "reference_policy_probabilities",
    "row_id",
    "runtime_allowed",
    "schema_id",
    "source_category",
    "source_id",
    "source_label",
    "source_role",
    "stage",
    "street",
    "table_format",
    "target_validity_flags",
    "training_allowed",
    "valid_claim_boundary",
    "variant",
}
LEGAL_ACTION_VALIDATION_FIELDS = {
    "legal_actions",
    "nonzero_actions_subset_of_legal_actions",
}
PROBABILITY_SUM_VALIDATION_FIELDS = {"passed", "sum", "tolerance"}
TARGET_VALIDITY_FIELDS = {
    "observed_behavior_label",
    "pokernow_no_limit_gto_label_allowed",
    "solver_policy_kl_allowed_for_this_stage050b_abstraction",
    "uses_future_board_or_actions",
    "uses_hidden_opponent_cards",
    "uses_mucked_cards",
}

INFO_STATE_PATTERN = re.compile(
    r"^stage050b_postflop_cbet:(?P<actor>p[01]):"
    r"pos=(?P<position>[^:]+):"
    r"hand_bucket=(?P<hand_bucket>[^:]+):"
    r"board_texture=(?P<board_texture>[^:]+):"
    r"history=(?P<history>[^:]+):"
    r"pot=(?P<pot_bb>[0-9]+(?:\.[0-9]+)?)bb:"
    r"stack=(?P<stack_bb>[0-9]+(?:\.[0-9]+)?)bb$"
)


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != unicodedata.normalize("NFC", value)
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise Stage6ContractError(f"{label} must be canonical nonempty text")
    return value


@dataclass(frozen=True)
class ReferenceQuery:
    table_format: str
    street: str
    position: str
    hand_bucket: str
    board_texture: str
    history: str
    pot_bb: float
    stack_bb: float
    legal_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "table_format",
            "street",
            "position",
            "hand_bucket",
            "board_texture",
            "history",
        ):
            value = getattr(self, field)
            _canonical_text(value, label=f"reference query {field}")
        for field in ("pot_bb", "stack_bb"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise Stage6ContractError(f"reference query has invalid {field}")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise Stage6ContractError(f"reference query has invalid {field}")
        if (
            not isinstance(self.legal_actions, tuple)
            or not self.legal_actions
            or any(
                type(action) is not str
                or not action
                or action != action.strip()
                for action in self.legal_actions
            )
            or len(set(self.legal_actions)) != len(self.legal_actions)
        ):
            raise Stage6ContractError(
                "reference query legal_actions must be unique nonempty strings"
            )

    @property
    def key(self) -> tuple[str, str, str, str, float, float]:
        return (
            self.position,
            self.hand_bucket,
            self.board_texture,
            self.history,
            float(self.pot_bb),
            float(self.stack_bb),
        )


def load_jsonl_bytes(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage6ContractError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Stage6ContractError(f"{label}: reference JSONL is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    Stage6ContractError(
                        f"{label}:{line_number}: nonfinite JSON token {token}"
                    )
                ),
            )
        except json.JSONDecodeError as error:
            raise Stage6ContractError(
                f"{label}:{line_number}: invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise Stage6ContractError(
                f"{label}:{line_number}: reference row must be an object"
            )
        rows.append(value)
    if not rows:
        raise Stage6ContractError(f"reference JSONL has no rows: {label}")
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_bytes(path.read_bytes(), label=str(path))


def _finite_probability(value: Any, *, row_id: str, action: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise Stage6ContractError(
            f"{row_id}: probability for {action} must be strict numeric data"
        )
    probability = float(value)
    if not math.isfinite(probability) or probability < 0 or probability > 1:
        raise Stage6ContractError(
            f"{row_id}: probability for {action} must be finite in [0, 1]"
        )
    return probability


def validate_reference_row(
    row: Mapping[str, Any],
) -> tuple[tuple[str, str, str, str, float, float], dict[str, Any]]:
    if not isinstance(row, Mapping) or set(row) != REFERENCE_ROW_FIELDS:
        missing = sorted(REFERENCE_ROW_FIELDS - set(row)) if isinstance(row, Mapping) else []
        unknown = sorted(set(row) - REFERENCE_ROW_FIELDS) if isinstance(row, Mapping) else []
        raise Stage6ContractError(
            f"reference row exact schema mismatch: missing={missing}, unknown={unknown}"
        )
    row_id = _canonical_text(row.get("row_id"), label="reference row_id")
    if row.get("schema_id") != SCHEMA_ID:
        raise Stage6ContractError(f"{row_id}: unsupported schema_id")
    if row.get("abstraction_id") != ABSTRACTION_ID:
        raise Stage6ContractError(f"{row_id}: unsupported abstraction_id")
    if row.get("table_format") != TABLE_FORMAT or row.get("street") != STREET:
        raise Stage6ContractError(f"{row_id}: unsupported table format or street")

    info_state = _canonical_text(
        row.get("information_state_key"), label=f"{row_id}.information_state_key"
    )
    match = INFO_STATE_PATTERN.fullmatch(info_state)
    if match is None:
        raise Stage6ContractError(f"{row_id}: malformed information_state_key")
    parsed = match.groupdict()
    position = _canonical_text(row.get("position"), label=f"{row_id}.position")
    hand_bucket = _canonical_text(
        row.get("hand_bucket"), label=f"{row_id}.hand_bucket"
    )
    board_texture = _canonical_text(
        row.get("board_texture"), label=f"{row_id}.board_texture"
    )
    history = _canonical_text(row.get("history"), label=f"{row_id}.history")
    if position not in POSITION_TO_ACTOR:
        raise Stage6ContractError(f"{row_id}: unsupported position")
    expected_fields = {
        "actor": POSITION_TO_ACTOR[position],
        "position": position,
        "hand_bucket": hand_bucket,
        "board_texture": board_texture,
        "history": history,
    }
    for field, expected in expected_fields.items():
        if parsed[field] != expected:
            raise Stage6ContractError(
                f"{row_id}: information-state {field} disagrees with row"
            )
    pot_bb = float(parsed["pot_bb"])
    stack_bb = float(parsed["stack_bb"])
    if pot_bb <= 0 or stack_bb <= 0:
        raise Stage6ContractError(f"{row_id}: pot and stack must be positive")

    legal_raw = row.get("legal_actions")
    policy_raw = row.get("reference_policy_probabilities")
    if not isinstance(legal_raw, list) or not legal_raw:
        raise Stage6ContractError(f"{row_id}: legal_actions must be a nonempty list")
    if not isinstance(policy_raw, Mapping) or not policy_raw:
        raise Stage6ContractError(f"{row_id}: policy must be a nonempty object")
    if any(
        type(action) is not str or not action or action != action.strip()
        for action in legal_raw
    ):
        raise Stage6ContractError(
            f"{row_id}: legal_actions must contain nonempty strings"
        )
    legal_actions = list(legal_raw)
    if len(set(legal_actions)) != len(legal_actions):
        raise Stage6ContractError(f"{row_id}: legal_actions contains duplicates")
    if any(type(action) is not str or not action or action != action.strip() for action in policy_raw):
        raise Stage6ContractError(f"{row_id}: policy action names are not canonical")
    policy = {
        action: _finite_probability(value, row_id=row_id, action=action)
        for action, value in policy_raw.items()
    }
    if set(policy) != set(legal_actions):
        raise Stage6ContractError(
            f"{row_id}: policy action set must exactly equal legal action set"
        )
    probability_sum = math.fsum(policy.values())
    if abs(probability_sum - 1.0) > 1e-6:
        raise Stage6ContractError(
            f"{row_id}: policy probabilities sum to {probability_sum}"
        )

    validation = row.get("legal_action_validation")
    if not isinstance(validation, Mapping) or set(validation) != LEGAL_ACTION_VALIDATION_FIELDS:
        raise Stage6ContractError(f"{row_id}: missing legal_action_validation")
    if validation.get("nonzero_actions_subset_of_legal_actions") is not True:
        raise Stage6ContractError(f"{row_id}: source legal-action validation failed")
    validation_actions = validation.get("legal_actions")
    if (
        not isinstance(validation_actions, list)
        or any(type(action) is not str for action in validation_actions)
        or set(validation_actions) != set(legal_actions)
    ):
        raise Stage6ContractError(f"{row_id}: legal-action receipts disagree")

    sum_validation = row.get("probability_sum_validation")
    if (
        not isinstance(sum_validation, Mapping)
        or set(sum_validation) != PROBABILITY_SUM_VALIDATION_FIELDS
        or sum_validation.get("passed") is not True
    ):
        raise Stage6ContractError(f"{row_id}: source probability validation failed")
    recorded_sum = _finite_probability(
        sum_validation.get("sum"), row_id=row_id, action="recorded_probability_sum"
    )
    tolerance_raw = sum_validation.get("tolerance")
    if isinstance(tolerance_raw, bool) or not isinstance(tolerance_raw, Real):
        raise Stage6ContractError(
            f"{row_id}: probability validation tolerance is invalid"
        )
    recorded_tolerance = float(tolerance_raw)
    if (
        not math.isfinite(recorded_tolerance)
        or recorded_tolerance < 0
        or abs(recorded_sum - probability_sum) > recorded_tolerance
    ):
        raise Stage6ContractError(
            f"{row_id}: probability sum receipt disagrees with policy"
        )

    primary = _canonical_text(
        row.get("primary_action"), label=f"{row_id}.primary_action"
    )
    if primary not in policy:
        raise Stage6ContractError(f"{row_id}: primary action is not legal")
    primary_probability = _finite_probability(
        row.get("primary_action_probability"), row_id=row_id, action=primary
    )
    if abs(primary_probability - policy[primary]) > 1e-12:
        raise Stage6ContractError(f"{row_id}: primary probability disagrees with policy")
    if primary_probability + 1e-12 < max(policy.values()):
        raise Stage6ContractError(f"{row_id}: primary action is not a policy maximum")

    validity = row.get("target_validity_flags")
    if not isinstance(validity, Mapping) or set(validity) != TARGET_VALIDITY_FIELDS:
        raise Stage6ContractError(f"{row_id}: target validity flags are missing")
    for prohibited in (
        "uses_future_board_or_actions",
        "uses_hidden_opponent_cards",
        "uses_mucked_cards",
        "observed_behavior_label",
        "pokernow_no_limit_gto_label_allowed",
    ):
        if validity.get(prohibited) is not False:
            raise Stage6ContractError(f"{row_id}: prohibited flag {prohibited} is set")
    if validity.get("solver_policy_kl_allowed_for_this_stage050b_abstraction") is not True:
        raise Stage6ContractError(f"{row_id}: solver-policy comparison is not allowed")

    for field in (
        "display_allowed",
        "runtime_allowed",
        "training_allowed",
        "promotion_allowed",
        "production_writes_allowed",
        "automatic_pokernow_actions_allowed",
    ):
        if type(row.get(field)) is not bool:
            raise Stage6ContractError(f"{row_id}: {field} must be a strict boolean")
    _canonical_text(
        row.get("valid_claim_boundary"), label=f"{row_id}.valid_claim_boundary"
    )

    key = (position, hand_bucket, board_texture, history, pot_bb, stack_bb)
    validated = dict(row)
    validated["reference_policy_probabilities"] = policy
    validated["legal_actions"] = legal_actions
    validated["stage6_exact_key"] = list(key)
    return key, validated


class ReferencePolicyIndex:
    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        runtime_authority_sha256: str | None = None,
    ) -> None:
        self._rows: dict[tuple[str, str, str, str, float, float], dict[str, Any]] = {}
        self._row_ids: set[str] = set()
        for source in rows:
            key, row = validate_reference_row(source)
            row_id = str(row["row_id"])
            if key in self._rows:
                raise Stage6ContractError(f"duplicate exact reference key: {key}")
            if row_id in self._row_ids:
                raise Stage6ContractError(f"duplicate reference row_id: {row_id}")
            self._rows[key] = row
            self._row_ids.add(row_id)
        if not self._rows:
            raise Stage6ContractError("reference index has no rows")
        if runtime_authority_sha256 is not None:
            raise Stage6ContractError(
                "an opaque SHA-256 is not a validated Stage 6 promotion receipt"
            )
        self.runtime_authority_sha256 = None

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def lookup(self, query: ReferenceQuery, *, mode: str = "evaluation") -> dict[str, Any]:
        if mode not in {
            "evaluation",
            "display",
            "runtime",
            "training",
            "automatic_action",
        }:
            raise Stage6ContractError(f"unknown reference lookup mode: {mode}")
        if query.table_format != TABLE_FORMAT or query.street != STREET:
            return self._abstain(mode, "OUTSIDE_TABLE_OR_STREET_SCOPE")
        row = self._rows.get(query.key)
        if row is None:
            return self._abstain(mode, "NO_EXACT_ABSTRACTION_STATE")
        if set(query.legal_actions) != set(row["legal_actions"]):
            return self._abstain(mode, "LEGAL_ACTION_SET_MISMATCH", row=row)

        authorization_field = {
            "display": "display_allowed",
            "runtime": "runtime_allowed",
            "training": "training_allowed",
            "automatic_action": "automatic_pokernow_actions_allowed",
        }.get(mode)
        if authorization_field is not None:
            if row.get(authorization_field) is not True:
                return self._abstain(
                    mode,
                    f"SOURCE_NOT_{authorization_field.upper()}",
                    row=row,
                )
            if self.runtime_authority_sha256 is None:
                return self._abstain(mode, "MISSING_STAGE6_PROMOTION_RECEIPT", row=row)
        if mode == "automatic_action" and row.get("production_writes_allowed") is not True:
            return self._abstain(mode, "PRODUCTION_WRITES_NOT_AUTHORIZED", row=row)

        return {
            "status": "MATCH",
            "reason": "EXACT_REVIEW_ONLY_MATCH" if mode == "evaluation" else "EXACT_MATCH",
            "mode": mode,
            "matched": True,
            "exact_match": True,
            "fuzzy_fallback_used": False,
            "row_id": row["row_id"],
            "information_state_key": row["information_state_key"],
            "legal_actions": list(row["legal_actions"]),
            "policy": dict(row["reference_policy_probabilities"]),
            "primary_action": row["primary_action"],
            "primary_action_probability": row["primary_action_probability"],
            "runtime_authorized": mode != "evaluation",
            "runtime_authority_sha256": self.runtime_authority_sha256,
            "valid_claim_boundary": row["valid_claim_boundary"],
        }

    @staticmethod
    def _abstain(
        mode: str,
        reason: str,
        *,
        row: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "ABSTAIN",
            "reason": reason,
            "mode": mode,
            "matched": row is not None,
            "exact_match": row is not None,
            "fuzzy_fallback_used": False,
            "row_id": row.get("row_id") if row else None,
            "policy": None,
            "runtime_authorized": False,
        }


def audit_reference_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = ReferencePolicyIndex(rows)
    validated = [validate_reference_row(row)[1] for row in rows]
    return {
        "schema_version": "stage6_stage050c_reference_audit_v1",
        "status": "PASS",
        "row_count": index.row_count,
        "unique_information_state_count": len(
            {row["information_state_key"] for row in validated}
        ),
        "unique_row_id_count": len({row["row_id"] for row in validated}),
        "position_counts": dict(sorted(Counter(row["position"] for row in validated).items())),
        "hand_bucket_counts": dict(
            sorted(Counter(row["hand_bucket"] for row in validated).items())
        ),
        "board_texture_counts": dict(
            sorted(Counter(row["board_texture"] for row in validated).items())
        ),
        "history_counts": dict(
            sorted(Counter(row["history"] for row in validated).items())
        ),
        "authorization_counts": {
            field: int(sum(row.get(field) is True for row in validated))
            for field in (
                "display_allowed",
                "runtime_allowed",
                "training_allowed",
                "promotion_allowed",
                "production_writes_allowed",
                "automatic_pokernow_actions_allowed",
            )
        },
        "scope": {
            "table_format": TABLE_FORMAT,
            "street": STREET,
            "pot_bb": sorted({row["stage6_exact_key"][4] for row in validated}),
            "stack_bb": sorted({row["stage6_exact_key"][5] for row in validated}),
        },
        "validated_row_root_digest": hashlib.sha256(
            canonical_json_bytes(
                sorted(validated, key=lambda row: str(row["row_id"]))
            )
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and index Stage 050C reference policies without promotion"
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.rows.resolve()
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256) is None:
        raise Stage6ContractError("Stage 050C expected source SHA-256 is invalid")
    try:
        source_raw = path.read_bytes()
    except OSError as error:
        raise Stage6ContractError("Stage 050C source cannot be read") from error
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        source_sha256 != args.expected_source_sha256
    ):
        raise Stage6ContractError(
            "Stage 050C source does not match the frozen external receipt"
        )
    rows = load_jsonl_bytes(source_raw, label=str(path))
    report = audit_reference_rows(rows)
    report["source"] = {
        "path": str(path),
        "bytes": len(source_raw),
        "sha256": source_sha256,
        "parsed_and_reported_from_one_immutable_buffer": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
