"""Exact CSV/Hand-sidecar adapter for the separate reveal-selected experiment.

Reads immutable exported bytes only. This module neither queries a database nor
fits a model. Full conversion belongs in Colab; local fixtures are <=1000 rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import time
import uuid

from stage6_empirical_range_learning_v2 import ACTIONS, AUTHORITY, POSITIONS, VERSION, canonical, cards, public_features, sha, write_once

SCHEMA = "stage6_empirical_hand_sidecar_v2"


def integer(value):
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError("exact_positive_integer_required")
    return int(value)


def number(value):
    if value in (None, "") or isinstance(value, bool):
        raise ValueError("numeric_source_missing")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric_source_nonfinite")
    return result


def timestamp(value):
    milliseconds = integer(value)
    if milliseconds < 1000000000000 or milliseconds > 9999999999999:
        raise ValueError("epoch_milliseconds_required")
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def truth(value):
    if value not in (True, False, "true", "false"):
        raise ValueError("explicit_boolean_required")
    return value in (True, "true")


def normalized_card(value):
    # Canonical parser stores standard rank/suit strings; accept its lowercase
    # ranks and 10 notation without inventing a card or a suit.
    if not isinstance(value, str):
        raise ValueError("missing_card")
    text = value.strip().replace("10", "T")
    result = text[:1].upper() + text[1:].lower()
    cards([result])
    return result


def payload(event):
    result = event["payload"]
    # HandRawEvent.payload is the exact PokerNow payload, never a guessed
    # normalized action. Reject competing wrapper shapes.
    if not isinstance(result, dict) or "type" not in result or "payload" in result:
        raise ValueError("raw_payload_shape_unsupported")
    if event.get("rawType") is not None and int(event["rawType"]) != int(result["type"]):
        raise ValueError("raw_type_mismatch")
    return result


def index_hands(hand_rows):
    hands, games = {}, defaultdict(list)
    for hand in hand_rows:
        if hand.get("schema_version") != SCHEMA:
            raise ValueError("hand_sidecar_schema_mismatch")
        key = hand["id"]
        if key in hands and canonical(hands[key]) != canonical(hand):
            raise ValueError("conflicting_hand_sidecar")
        if key in hands:
            continue
        if hand["game"]["id"] != hand["gameId"]:
            raise ValueError("hand_game_identity_mismatch")
        hands[key] = hand
        games[hand["gameId"]].append(hand)
    bounds = {}
    for game, rows in games.items():
        starts = [integer(h["startedAt"]) for h in rows]
        expected_first = {integer(h["game"]["firstHandStartedAt"]) for h in rows}
        expected_last = {integer(h["game"]["lastHandStartedAt"]) for h in rows}
        if expected_first != {min(starts)} or expected_last != {max(starts)}:
            raise ValueError("sidecar_does_not_cover_full_game_bounds")
        observed = [integer(e["at"]) for h in rows for e in h["rawEvents"] if e.get("at") is not None]
        if not observed:
            raise ValueError("game_has_no_event_chronology")
        bounds[game] = (timestamp(min(starts)), timestamp(max(observed)))
    return hands, bounds


def opportunity(row):
    if row["street"] != "preflop":
        return "postflop"
    raises = int(number(row["preflop_raise_count_before"]))
    if truth(row["preflop_squeeze_opportunity"]):
        return "squeeze"
    if raises == 0:
        return "isolation" if number(row["preflop_limper_count_before"]) > 0 else "unopened"
    if raises == 1:
        return "facing_open"
    if raises == 2:
        # Distinguishing a squeeze from another 3-bet requires the exact
        # previous entry sequence; UNKNOWN is preferable to relabeling it.
        return "defend_3bet" if truth(row["preflop_actor_has_raised_before"]) else "cold_4bet"
    return "facing_4bet" if raises == 3 else "facing_5bet"


def normalized_action(row):
    if truth(row["is_all_in"]):
        return "all_in"
    action = row["action_label"]
    if action == "bet_or_raise":
        return "raise" if number(row["current_bet_before_raw"]) > 0 else "bet"
    if action not in ACTIONS:
        raise ValueError("unsupported_raw_action_label")
    return action


def position_and_behind(hand, row, active_ids, current_street_rows):
    roster = hand["players"]
    seats = {p["pokernowPlayerId"]: integer(p["seat"]) for p in roster}
    if len(seats) != len(roster) or len(set(seats.values())) != len(roster):
        raise ValueError("ambiguous_hand_start_seats")
    actor = row["pokernow_player_id"]
    if actor not in seats or integer(row["seat"]) != seats[actor]:
        raise ValueError("actor_seat_identity_mismatch")
    dealer = integer(hand["dealerSeat"])
    order = sorted(seats, key=lambda player: (seats[player] <= dealer, seats[player]))
    if dealer not in seats.values():
        raise ValueError("dealer_not_in_dealt_roster")
    n = len(order)
    if n < 2 or n > 10:
        raise ValueError("unsupported_dealt_roster")
    # Explicit seat-order convention: 6-handed first-to-act is LJ; 9-handed
    # early positions are UTG/UTG1/UTG2 before LJ/HJ/CO. HU dealer is BTN/SB.
    if n == 2:
        mapping = {order[0]: "BB", order[1]: "BTN"}
        preflop_order = [order[1], order[0]]
    else:
        early = {3: [], 4: ["CO"], 5: ["HJ", "CO"], 6: ["LJ", "HJ", "CO"],
            7: ["UTG", "LJ", "HJ", "CO"], 8: ["UTG", "UTG1", "LJ", "HJ", "CO"],
            9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"],
            10: ["UNKNOWN", "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"]}[n]
        mapping = dict(zip(order, ["SB", "BB"] + early + ["BTN"]))
        preflop_order = order[2:] + order[:2]
    action_order = preflop_order if row["street"] == "preflop" else order
    # Players behind is positional and able-to-act, not a claim that Hero closes
    # the current betting action. The latter needs a pending-action ledger.
    stacks = {p["pokernowPlayerId"]: number(p["stackRaw"]) for p in roster}
    for previous in current_street_rows:
        stacks[previous["pokernow_player_id"]] = number(previous["stack_after_raw"])
    behind = sum(p in active_ids and stacks[p] > 0 for p in action_order[action_order.index(actor) + 1:])
    return mapping[actor], behind


def reveal_label(hand, actor, action_at, full_board):
    seat_matches = [p["seat"] for p in hand["players"] if p["pokernowPlayerId"] == actor]
    if len(seat_matches) != 1:
        raise ValueError("reveal_actor_identity_ambiguous")
    candidates = []
    for event in hand["rawEvents"]:
        raw = payload(event)
        if int(raw["type"]) != 12 or raw.get("seat") != seat_matches[0]:
            continue
        values = raw.get("cards")
        if not isinstance(values, list) or len(values) != 2 or any(v is None for v in values):
            continue
        hole = [normalized_card(v) for v in values]
        cards(full_board + hole)
        if event.get("at") is None:
            continue
        candidates.append((integer(event["at"]), hole, event["id"]))
    if not candidates:
        return dict(state="source_missing", hole_cards=None, reason="no_exact_two_card_reveal_event_with_time")
    identities = {tuple(sorted(hole)) for _, hole, _ in candidates}
    if len(identities) != 1:
        raise ValueError("conflicting_reveal_events")
    known = [h for h in hand["holeCards"] if h["pokernowPlayerId"] == actor and h["visibilitySource"] == "SHOWN_CARDS_EVENT"]
    known_cards = {normalized_card(h["card"]) for h in known}
    if known_cards != set(candidates[0][1]):
        raise ValueError("raw_reveal_hole_table_mismatch")
    at, hole, event_id = min(candidates)
    if at <= action_at:
        # Exposed holdings before the observed action define a different game
        # information set; exclude rather than train a hidden-range model on it.
        raise ValueError("actor_cards_exposed_before_action")
    return dict(state="revealed_after_decision", hole_cards=hole, available_utc=timestamp(at), source_event_id=event_id)


def convert_row(row, identity, hand, bounds, prior_rows):
    if hand["bombPot"] or hand["doubleBoard"] or hand.get("straddleSeat") is not None:
        raise ValueError("unsupported_bomb_double_board_or_straddle")
    if str(hand["gameType"]).lower() not in {"th", "holdem", "hold'em", "nlh", "texas_holdem"}:
        raise ValueError("unsupported_or_unknown_game_type")
    if identity["identity_grouping_status"] != "resolved_retrospective_group" or not identity["canonical_group_player_id"]:
        raise ValueError("unresolved_canonical_identity")
    for name in ["hand_action_id", "hand_id", "game_id", "event_index"]:
        if str(row[name]) != str(identity[name]):
            raise ValueError("action_identity_join_mismatch")
    if identity["actor_pokernow_player_id"] != row["pokernow_player_id"] or hand["id"] != row["hand_id"] or hand["gameId"] != row["game_id"]:
        raise ValueError("actor_hand_game_identity_mismatch")
    events = sorted(hand["rawEvents"], key=lambda e: integer(e["eventIndex"]))
    if len({e["eventIndex"] for e in events}) != len(events) or len({e["id"] for e in events}) != len(events):
        raise ValueError("duplicate_raw_event_identity")
    if any(e.get("at") is None for e in events):
        raise ValueError("incomplete_raw_event_chronology")
    ats = [integer(e["at"]) for e in events]
    if ats != sorted(ats):
        raise ValueError("raw_event_time_order_conflict")
    current = [e for e in events if integer(e["eventIndex"]) == integer(row["event_index"])]
    if len(current) != 1:
        raise ValueError("source_action_event_missing")
    current = current[0]
    at = integer(current["at"])
    if at != integer(identity["decision_epoch_ms"]) or timestamp(at) != identity["decision_utc"]:
        raise ValueError("action_time_sidecar_mismatch")
    prefix = [e for e in events if integer(e["eventIndex"]) < integer(row["event_index"])]
    # Exact previous public state timestamp; never invent time by subtracting 1ms.
    decision_at = max([integer(hand["startedAt"])] + [integer(e["at"]) for e in prefix])
    if decision_at >= at:
        raise ValueError("no_strictly_ordered_pre_action_timestamp")
    if not any(int(payload(e)["type"]) == 10 for e in events):
        raise ValueError("no_completed_pot_award_event")
    board, full_board = [], []
    for event in events:
        raw = payload(event)
        if int(raw["type"]) == 9:
            if raw.get("run", 1) != 1 or raw.get("turn") not in (1, 2, 3):
                raise ValueError("unsupported_board_event")
            dealt = [normalized_card(v) for v in raw["cards"]]
            full_board.extend(dealt)
            if integer(event["eventIndex"]) < integer(row["event_index"]):
                board.extend(dealt)
    cards(full_board)
    table_board = [normalized_card(v["card"]) for v in sorted(hand["boardCards"], key=lambda c: (int(c["run"]), ["flop", "turn", "river"].index(c["street"]), int(c["cardIndex"])))]
    if full_board != table_board:
        raise ValueError("board_event_table_mismatch")
    if len(board) != {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(row["street"]):
        raise ValueError("decision_board_street_mismatch")
    if row["street"] != "preflop" and integer(row["board_card_count"]) != len(board):
        raise ValueError("decision_board_count_mismatch")
    active = json.loads(row["active_player_ids_before"])
    if len(set(active)) != len(active):
        raise ValueError("duplicate_active_player")
    previous = [r for r in prior_rows if r["street"] == row["street"]]
    position, behind = position_and_behind(hand, row, active, prior_rows)
    bb = number(row["big_blind_raw"])
    pot = number(row["pot_before_bb"])
    if bb <= 0 or pot <= 0 or bb != number(hand["bigBlindRaw"]):
        raise ValueError("invalid_blind_or_pot")
    aggressive = lambda r: number(r["current_bet_after_raw"]) > number(r["current_bet_before_raw"])
    last_full_raise = bb
    for previous_row in previous:
        increase = number(previous_row["current_bet_after_raw"]) - number(previous_row["current_bet_before_raw"])
        if increase > 0:
            if truth(previous_row["is_all_in"]) and increase < last_full_raise:
                raise ValueError("source_legality_after_short_all_in_raise_unverified")
            last_full_raise = increase
    action = normalized_action(row)
    pub = dict(street=row["street"], board=board, position=position, opportunity=opportunity(row),
        pot_before_bb=pot, amount_to_call_bb=number(row["amount_to_call_bb"]),
        effective_stack_before_bb=number(row["effective_stack_before_bb"]), spr_before=number(row["spr_before"]),
        active_opponent_count_before=number(row["active_opponent_count_before"]), players_behind=behind,
        prior_raise_count=sum(aggressive(r) for r in previous),
        prior_call_count=sum(normalized_action(r) == "call" or (normalized_action(r) == "all_in" and not aggressive(r)) for r in previous),
        facing_size_pot_fraction=number(row["amount_to_call_bb"]) / pot,
        actor_committed_before_bb=number(row["street_contribution_before_raw"]) / bb,
        dealt_count_at_hand_start=len(hand["players"]))
    public_features(pub)
    if row["legal_mask_quality"] != "action_state_export_exact_v1":
        raise ValueError("nonexact_legal_mask")
    legal = [action for action in ACTIONS if truth(row["can_" + action])]
    if action not in legal:
        raise ValueError("observed_action_not_legal")
    return dict(row_id=row["hand_action_id"], game_id=row["game_id"], hand_id=row["hand_id"],
        actor_id=identity["canonical_group_player_id"], actor_pokernow_player_id=row["pokernow_player_id"],
        session_id="game:" + row["game_id"], game_started_utc=bounds[0], game_completed_utc=bounds[1],
        hand_started_utc=timestamp(hand["startedAt"]), decision_utc=timestamp(decision_at), public_available_utc=timestamp(decision_at),
        action_observed_utc=timestamp(at), hand_completed_utc=timestamp(max(ats)), public=pub,
        action=action, legal_actions=legal,
        observed_action_size_pot_fraction=number(row["action_contribution_delta_raw"]) / bb / pot if action in {"bet", "raise", "all_in"} else None,
        card_visibility=reveal_label(hand, row["pokernow_player_id"], at, full_board),
        source_event_ids=[e["id"] for e in prefix] or ["hand_start:" + hand["id"]],
        raw_dealt_count=len(hand["players"]), source_definition=dict(
            decision_time="last_observed_public_prefix_event_before_actual_action", position="dealt_seat_order_v2",
            players_behind="positional_players_with_chips_not_pending_action_ledger", size_target="incremental_contribution_divided_by_preaction_pot",
            session="whole_game_proxy_no_synthetic_session_boundaries"))


def convert(actions, identities, hand_rows):
    hands, bounds = index_hands(hand_rows)
    identity = {r["hand_action_id"]: r for r in identities}
    if len(identity) != len(identities) or {r["hand_action_id"] for r in actions} != set(identity):
        raise ValueError("action_identity_inventory_mismatch")
    seen, prior, accepted, quarantine = {}, defaultdict(list), [], []
    for row in sorted(actions, key=lambda r: (r["hand_id"], integer(r["event_index"]))):
        key = row["hand_action_id"]
        if key in seen:
            if seen[key] != canonical(row):
                raise ValueError("conflicting_action_duplicate")
            continue
        seen[key] = canonical(row)
        hand = hands.get(row["hand_id"])
        try:
            if hand is None:
                raise ValueError("hand_sidecar_missing")
            accepted.append(convert_row(row, identity[key], hand, bounds[row["game_id"]], prior[row["hand_id"]]))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            quarantine.append(dict(row_id=key, game_id=row["game_id"], hand_id=row["hand_id"], reason=str(error)))
        prior[row["hand_id"]].append(row)
    accepted.sort(key=lambda r: (r["decision_utc"], r["game_id"], r["hand_id"], r["row_id"]))
    return accepted, quarantine, dict(schema_version=VERSION + "_adapter", source_action_rows=len(seen), accepted_rows=len(accepted),
        quarantined_rows=len(quarantine), quarantine_reasons=dict(Counter(r["reason"] for r in quarantine)),
        accepted_games=sorted({r["game_id"] for r in accepted}), physical_reveal_label_rows=sum(r["card_visibility"]["state"] == "revealed_after_decision" for r in accepted),
        latent_rows=sum(r["card_visibility"]["state"] != "revealed_after_decision" for r in accepted), authority=AUTHORITY)


def sidecar_stream(path):
    previous = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            hand = json.loads(line)
            if hand.get("schema_version") != SCHEMA or not isinstance(hand.get("id"), str):
                raise ValueError("hand_sidecar_schema_mismatch")
            if previous is not None and hand["id"] <= previous:
                raise ValueError("sidecar_hand_order_or_duplicate_identity")
            previous = hand["id"]
            yield hand


def streaming_game_bounds(path):
    """Only one Hand plus a small game-bound table is resident."""
    games, game_identities = {}, {}
    hand_count = 0
    for hand in sidecar_stream(path):
        if hand["game"]["id"] != hand["gameId"]:
            raise ValueError("hand_game_identity_mismatch")
        public_id=hand["game"]["pokernowGameId"]
        if not public_id or game_identities.setdefault(public_id,hand["gameId"])!=hand["gameId"]:
            raise ValueError("conflicting_exact_game_identity")
        start = integer(hand["startedAt"])
        times = [integer(e["at"]) for e in hand["rawEvents"] if e.get("at") is not None]
        info = games.setdefault(hand["gameId"], dict(first=start, last=start, end=0,
            expected_first=integer(hand["game"]["firstHandStartedAt"]), expected_last=integer(hand["game"]["lastHandStartedAt"])))
        if info["expected_first"] != integer(hand["game"]["firstHandStartedAt"]) or info["expected_last"] != integer(hand["game"]["lastHandStartedAt"]):
            raise ValueError("inconsistent_game_source_bounds")
        info["first"] = min(info["first"], start); info["last"] = max(info["last"], start)
        info["end"] = max(info["end"], max(times, default=0))
        hand_count += 1
    for game, info in games.items():
        if info["first"] != info["expected_first"] or info["last"] != info["expected_last"] or not info["end"]:
            raise ValueError("sidecar_does_not_cover_full_game_bounds:" + game)
    return {game: (timestamp(info["first"]), timestamp(info["end"])) for game, info in games.items()}, hand_count, game_identities


def paired_action_stream(actions_path, identity_path):
    previous = None
    with Path(actions_path).open(encoding="utf-8-sig", newline="") as action_stream, Path(identity_path).open(encoding="utf-8-sig", newline="") as identity_stream:
        for action, identity in itertools.zip_longest(csv.DictReader(action_stream), csv.DictReader(identity_stream)):
            if action is None or identity is None:
                raise ValueError("action_identity_row_count_mismatch")
            key = (action["hand_id"], integer(action["event_index"]))
            if previous is not None and key <= previous:
                raise ValueError("action_hand_event_order_or_duplicate")
            previous = key
            if any(str(action[k]) != str(identity[k]) for k in ["hand_action_id", "hand_id", "game_id", "event_index"]) or action["pokernow_player_id"] != identity["actor_pokernow_player_id"]:
                raise ValueError("ordered_action_identity_join_mismatch")
            yield action, identity


def publish_partial(partial, destination):
    if destination.exists():
        if destination.stat().st_size != partial.stat().st_size or sha(destination) != sha(partial):
            raise ValueError("different_existing_conversion_member:" + destination.name)
        partial.unlink()  # exact owned temporary file, never a recursive delete
    else:
        os.rename(partial, destination)


def streaming_convert(actions_path, identity_path, hand_path, output, fixture=False):
    bounds, hand_count, game_identities = streaming_game_bounds(hand_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    rows_path, quarantine_path = output / "range_training_rows.jsonl", output / "range_quarantine_rows.jsonl"
    rows_partial, quarantine_partial = output / (rows_path.name + ".partial." + token), output / (quarantine_path.name + ".partial." + token)
    accepted = rejected = total = revealed = 0
    reasons, game_counts, player_identities = Counter(), Counter(), {}
    hand_iterator = iter(sidecar_stream(hand_path)); current_hand = next(hand_iterator, None)
    started = last_progress = time.monotonic()
    with rows_partial.open("x", encoding="utf-8", newline="\n") as good, quarantine_partial.open("x", encoding="utf-8", newline="\n") as bad:
        for hand_id, grouped in itertools.groupby(paired_action_stream(actions_path, identity_path), key=lambda pair: pair[0]["hand_id"]):
            while current_hand is not None and current_hand["id"] < hand_id:
                current_hand = next(hand_iterator, None)
            if current_hand is None or current_hand["id"] != hand_id:
                raise ValueError("ordered_hand_sidecar_missing:" + hand_id)
            prior = []
            for action, identity in grouped:
                total += 1
                if fixture and total > 1000:
                    raise ValueError("bounded_synthetic_fixture_only")
                try:
                    row = convert_row(action, identity, current_hand, bounds[action["game_id"]], prior)
                    if player_identities.setdefault(row['actor_pokernow_player_id'],row['actor_id'])!=row['actor_id']:
                        raise RuntimeError('conflicting_exact_player_identity')
                    good.write(canonical(row) + "\n")
                    accepted += 1; game_counts[row["game_id"]] += 1
                    revealed += row["card_visibility"]["state"] == "revealed_after_decision"
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    reason = str(error); reasons[reason] += 1; rejected += 1
                    bad.write(canonical(dict(row_id=action["hand_action_id"], game_id=action["game_id"], hand_id=hand_id, reason=reason)) + "\n")
                prior.append(action)
            if time.monotonic() - last_progress >= 20:
                elapsed = time.monotonic() - started
                print(canonical(dict(phase="streaming_range_adapter", rows=total, accepted=accepted, quarantined=rejected, rows_per_second=total/elapsed)), flush=True)
                last_progress = time.monotonic()
        # Exhaust the sidecar to verify its ordered suffix, even when those hands
        # have no qualifying action rows. No source row is dropped by sampling.
        for _ in hand_iterator:
            pass
    publish_partial(rows_partial, rows_path); publish_partial(quarantine_partial, quarantine_path)
    return dict(schema_version=VERSION + "_adapter", source_action_rows=total, sidecar_hands=hand_count, accepted_rows=accepted,
        quarantined_rows=rejected, quarantine_reasons=dict(sorted(reasons.items())), accepted_games=sorted(game_counts),
        accepted_rows_by_game=dict(sorted(game_counts.items())), physical_reveal_label_rows=revealed, latent_rows=accepted-revealed,
        game_bounds=bounds,exact_identities=dict(games=game_identities,players=player_identities),conversion="two_pass_sidecar_bounds_then_ordered_single_hand_merge", authority=AUTHORITY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if set(manifest) != ({"actions", "identities", "hands", "synthetic_fixture"} if args.fixture else {"actions", "identities", "hands"}):
        raise ValueError("exact_input_manifest_roles_required")
    for key in ["actions", "identities", "hands"]:
        path = getattr(args, key)
        if sha(path) != manifest[key]["sha256"] or path.stat().st_size != manifest[key]["bytes"]:
            raise ValueError("direct_input_identity_mismatch:" + key)
    if not args.fixture and (not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG")):
        raise ValueError("full_conversion_colab_only")
    if args.fixture and manifest.get("synthetic_fixture") is not True:
        raise ValueError("bounded_synthetic_fixture_only")
    receipt_path = args.output / "range_adapter_receipt.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous["lineage"]["input_manifest_sha256"] != sha(args.input_manifest) or previous["lineage"]["adapter_sha256"] != sha(Path(__file__)):
            raise ValueError("completed_conversion_lineage_mismatch")
        for name, identity in previous["members"].items():
            if Path(name).name != name or sha(args.output / name) != identity["sha256"] or (args.output / name).stat().st_size != identity["bytes"]:
                raise ValueError("completed_conversion_member_mismatch")
        print(canonical(dict(phase="completed_conversion_reused", source_action_rows=previous["source_action_rows"])), flush=True)
        return
    report = streaming_convert(args.actions, args.identities, args.hands, args.output, args.fixture)
    for key in ["actions", "identities", "hands"]:
        if sha(getattr(args, key)) != manifest[key]["sha256"]:
            raise ValueError("input_changed_during_conversion:" + key)
    report["lineage"] = dict(input_manifest_sha256=sha(args.input_manifest), adapter_sha256=sha(Path(__file__)), source_inputs=manifest, synthetic_fixture=args.fixture)
    exact=report.pop('exact_identities')
    write_once(args.output/'identity_mapping.json',dict(schema_version='stage6_empirical_exact_identity_map_v2',source_manifest_sha256=sha(args.input_manifest),**exact))
    report["members"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in [args.output / "range_training_rows.jsonl", args.output / "range_quarantine_rows.jsonl", args.output/'identity_mapping.json']}
    write_once(args.output / "range_adapter_receipt.json", report)
    print(canonical(report), flush=True)


if __name__ == "__main__":
    main()
