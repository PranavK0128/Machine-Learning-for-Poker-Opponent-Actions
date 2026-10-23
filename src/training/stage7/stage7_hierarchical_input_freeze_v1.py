"""Freeze increment05 inputs from accepted receipts and two raw manifest fetches."""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
from pathlib import Path
import re

import stage7_hierarchical_response_v1 as h
import stage6_empirical_range_learning_v2_release as t

STAGE = Path(__file__).resolve().parents[1]
CLOSEOUT_SHA = "edea3feaf8dc15a27444b2c902777f3390130cade5c2f7f64335e0fa622d918c"
IMPORT_SHA = "99ee963e605f362dc67fe48ff3350f1dc665c5e8e381595a6c5ac0d7c9e86e75"
CAPTURE_PATHS = tuple(f"runtime_state/notebook05_checkpoint_manifest_pass{number}.json" for number in (1, 2))
CODE_NAMES = frozenset(("stage6_empirical_range_learning_v2.py", "stage6_empirical_range_learning_v2_adapter.py",
                        "stage6_empirical_range_learning_v2_streaming.py", "stage6_empirical_range_learning_v2_release.py"))


def identity_bytes(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def validate_identity(value):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or value["bytes"] <= 0
            or not isinstance(value["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", value["sha256"])):
        raise ValueError("exact_positive_byte_identity_required")


def authenticated_bytes(path, expected_sha256, expected_bytes=None):
    raw = Path(path).read_bytes()
    actual = identity_bytes(raw)
    if actual["sha256"] != expected_sha256 or (expected_bytes is not None and actual["bytes"] != expected_bytes):
        raise ValueError("byte_identity_mismatch:" + str(path))
    return raw, actual


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key:" + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json:" + value)))


def validate_checkpoint(manifest, plan, report, receipt):
    if (set(manifest) != {"binding", "file_name", "members", "raw_byte_readbacks", "transport"}
            or set(manifest["binding"]) != {"code_files", "plan_sha256"}
            or manifest["file_name"] != "compact_rows.checkpoint.zip"
            or type(manifest["raw_byte_readbacks"]) is not int or manifest["raw_byte_readbacks"] != 2):
        raise ValueError("checkpoint_exact_schema_mismatch")
    if manifest["binding"]["plan_sha256"] != receipt["members"]["release_plan.json"]["sha256"]:
        raise ValueError("checkpoint_plan_binding_mismatch")
    if (set(plan["code_files"]) != CODE_NAMES or set(manifest["binding"]["code_files"]) != CODE_NAMES
            or manifest["binding"]["code_files"] != plan["code_files"]):
        raise ValueError("checkpoint_exact_code_lineage_mismatch")
    if set(manifest["members"]) != {"fit/prepared/rows.sqlite", "fit/prepared/prepared_receipt.json"}:
        raise ValueError("checkpoint_exact_member_inventory_mismatch")
    for value in [manifest["transport"], *manifest["members"].values(), *manifest["binding"]["code_files"].values()]:
        validate_identity(value)
    if manifest["members"]["fit/prepared/rows.sqlite"]["sha256"] != report["compact_preparation"]["database_sha256"]:
        raise ValueError("checkpoint_database_accepted_result_mismatch")


def freeze(stage=STAGE, *, expected_capture_identities):
    stage = Path(stage)
    closeout = stage / "manifests/stage7_closeout_manifest_r1.json"
    receipt_path = stage / "manifests/stage7_03_r2_import_receipt.json"
    # Authenticate and parse the same bytes; never verify a path then reopen it.
    try:
        _, closeout_identity = authenticated_bytes(closeout, CLOSEOUT_SHA)
        receipt_raw, import_identity = authenticated_bytes(receipt_path, IMPORT_SHA)
    except ValueError as error:
        raise ValueError("accepted_predecessor_anchor_changed") from error
    receipt = parse_json(receipt_raw)
    if receipt["status"] != "ACCEPT_DIAGNOSTIC_ARTIFACT" or receipt["authority"] != h.AUTHORITY:
        raise ValueError("accepted_predecessor_disposition_changed")
    returned = stage / "returned_results/03_range_r2"
    sources = {}; parsed = {}
    for name in ("protocol.json", "range_adapter_receipt.json", "range_response_report.json", "range_response_artifact.json", "release_plan.json"):
        expected = receipt["members"][name]
        validate_identity(expected)
        raw, _ = authenticated_bytes(returned / name, expected["sha256"], expected["bytes"])
        if name != "range_response_artifact.json":
            parsed[name] = parse_json(raw)
        sources[name] = dict(path=(returned / name).relative_to(stage).as_posix(), **receipt["members"][name])
    protocol, adapter, report = [parsed[name] for name in ("protocol.json", "range_adapter_receipt.json", "range_response_report.json")]
    if set(expected_capture_identities) != set(CAPTURE_PATHS):
        raise ValueError("two_exact_externally_authenticated_capture_paths_required")
    rows = []
    raw_copies = []
    captures = []
    for number in (1, 2):
        relative = CAPTURE_PATHS[number - 1]
        expected = expected_capture_identities[relative]
        validate_identity(expected)
        capture_raw, capture_identity = authenticated_bytes(stage / relative, expected["sha256"], expected["bytes"])
        value = parse_json(capture_raw)
        if (set(value) != {"schema_version", "pass", "id", "title", "parent_ids", "modified_time", "file_size_bytes", "b64_string"}
                or value["schema_version"] != "stage7_checkpoint_manifest_raw_download_v1"
                or type(value["pass"]) is not int or value["pass"] != number or value["id"] != "11s29IEhuUDwbGmYz1uPYmOVyXgz6-KYK"
                or value["title"] != "compact_rows.checkpoint.json"
                or value["parent_ids"] != ["1EC8y290SD0sby3WIdbaJPb78kT-M_HtQ"]):
            raise ValueError("raw_checkpoint_manifest_identity_mismatch")
        t.utc(value["modified_time"])
        raw = base64.b64decode(value["b64_string"], validate=True)
        if type(value["file_size_bytes"]) is not int or len(raw) != value["file_size_bytes"] or not raw:
            raise ValueError("raw_checkpoint_manifest_size_mismatch")
        raw_copies.append(raw); rows.append(value)
        captures.append(dict(path=relative, **capture_identity,
                             metadata={key: item for key, item in value.items() if key != "b64_string"}))
    if raw_copies[0] != raw_copies[1] or rows[0]["modified_time"] != rows[1]["modified_time"]:
        raise ValueError("two_stable_raw_checkpoint_manifest_copies_required")
    manifest = parse_json(raw_copies[0])
    validate_checkpoint(manifest, parsed["release_plan.json"], report, receipt)
    for name, identity in manifest["binding"]["code_files"].items():
        t.safe_name(name); t.verify(stage / "src" / name, identity)
    games = set().union(*(set(protocol[p + "_game_ids"]) for p in ("train", "calibration", "evaluation", "diagnostic")))
    extras = set(adapter["game_bounds"]) - games
    if extras != set(protocol["zero_accepted_row_game_exclusions"]):
        raise ValueError("exact_zero_row_exclusion_required")
    bounds = {game: adapter["game_bounds"][game] for game in sorted(games)}
    successor = h.development_protocol(protocol, bounds)
    result = dict(schema_version=h.SCHEMA + "_input_contract_v2", evidence_class=h.EVIDENCE, authority=h.AUTHORITY,
                  accepted_closeout=dict(path=closeout.relative_to(stage).as_posix(), **closeout_identity),
                  accepted_import=dict(path=receipt_path.relative_to(stage).as_posix(), **import_identity),
                  accepted_sources=sources, source_database_rows=report["compact_preparation"]["rows"],
                  source_zero_row_exclusions=sorted(extras), game_bounds=bounds, protocol=successor,
                  checkpoint_manifest_captures=captures,
                  compact_checkpoint=dict(manifest_id=rows[0]["id"], bytes=len(raw_copies[0]),
                      sha256=hashlib.sha256(raw_copies[0]).hexdigest(), manifest=manifest,
                      zip_id="1UhBErOPnfkfnecsxligQe6UHbWcSanQW", zip_raw_readback_pending=True),
                  execution_release=False)
    destination = stage / "manifests/stage7_05_input_contract_r2.json"
    t.publish_local_bytes(destination, (h.incumbent.canonical(result) + "\n").encode())
    return dict(path=str(destination), **t.identity(destination), source_games=len(games),
                cohorts={key: len(successor[key]) for key in ("train_game_ids", "calibration_game_ids", "development_evaluation_game_ids", "excluded_diagnostic_game_ids")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=str(STAGE))
    parser.add_argument("--capture-identities", required=True)
    parser.add_argument("--expected-capture-identities-sha256", required=True)
    args = parser.parse_args()
    raw, _ = authenticated_bytes(args.capture_identities, args.expected_capture_identities_sha256)
    print(h.incumbent.canonical(freeze(args.stage, expected_capture_identities=parse_json(raw))))
