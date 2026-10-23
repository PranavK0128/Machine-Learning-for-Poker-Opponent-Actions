"""Separately versioned, assumption-dependent range and response experiments.

Full fits run in Colab. Missing private cards are never imputed or called weak.
Selected-reveal diagnostics cannot establish population ranges or strategy EV.
No original Stage 026A artifact, selected Stage 6 model or runtime is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import threading
import uuid

VERSION = "stage6_empirical_range_learning_v2"
DIAGNOSTIC_CONTROLS = dict(schema_version=VERSION+"_diagnostic_controls", stale_prior_cutoff_utc="2026-07-01T00:00:00Z", permutation_seed=20260904)
ACTIONS = ["fold", "check", "call", "bet", "raise", "all_in"]
RANKS = "23456789TJQKA"
DECK = tuple(r + s for r in RANKS for s in "cdhs")
POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB", "UNKNOWN"]
OPPORTUNITIES = ["unopened", "isolation", "facing_open", "squeeze", "defend_squeeze", "defend_3bet", "cold_4bet", "facing_4bet", "facing_5bet", "postflop", "UNKNOWN"]
NUMERIC = ["pot_before_bb", "amount_to_call_bb", "effective_stack_before_bb", "spr_before", "active_opponent_count_before", "players_behind", "prior_raise_count", "prior_call_count", "facing_size_pot_fraction", "actor_committed_before_bb", "dealt_count_at_hand_start"]
PUBLIC_NAMES = (["street_" + s for s in ["preflop", "flop", "turn", "river"]]
                + ["position_" + s for s in POSITIONS] + ["opportunity_" + s for s in OPPORTUNITIES]
                + NUMERIC + ["board_count", "board_high_rank", "board_distinct_ranks", "board_max_suit"])
HOLE_NAMES = ["hole_high_rank", "hole_low_rank", "hole_pair", "hole_suited", "made_category", "kicker_1", "kicker_2", "kicker_3", "kicker_4", "kicker_5", "hole_max_board_suit", "straight_completion_rank_count"]
REQUIRED = {"row_id", "game_id", "hand_id", "actor_id", "session_id", "game_started_utc", "game_completed_utc", "decision_utc", "action_observed_utc", "hand_completed_utc", "public_available_utc", "public", "action", "legal_actions", "card_visibility", "source_event_ids"}
AUTHORITY = dict(population_range_qualified=False, strategy_qualified=False,
                 solver_value=False, promotion=False, runtime_weight=0,
                 excluded_shadow_layers_reactivated=False, automatic_action=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("explicit_utc_Z_required")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def cards(values):
    if not isinstance(values, list) or any(c not in DECK for c in values) or len(set(values)) != len(values):
        raise ValueError("invalid_or_duplicate_cards")
    return values


def hand_class(hole):
    cards(hole)
    if len(hole) != 2:
        raise ValueError("two_hole_cards_required")
    high, low = sorted([RANKS.index(c[0]) for c in hole], reverse=True)
    return RANKS[high] + RANKS[low] + ("" if high == low else "s" if hole[0][1] == hole[1][1] else "o")


def five_score(hand):
    ranks = [RANKS.index(c[0]) + 2 for c in hand]
    groups = sorted(((n, r) for r, n in Counter(ranks).items()), reverse=True)
    unique = sorted(set(ranks), reverse=True)
    straight = (unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 5 if unique == [14, 5, 4, 3, 2] else 0)
    flush = len({c[1] for c in hand}) == 1
    if flush and straight:
        result = [8, straight]
    elif groups[0][0] == 4:
        result = [7] + [r for _, r in groups]
    elif [n for n, _ in groups] == [3, 2]:
        result = [6] + [r for _, r in groups]
    elif flush:
        result = [5] + unique
    elif straight:
        result = [4, straight]
    else:
        result = [{3: 3, 2: 2 if len(groups) > 1 and groups[1][0] == 2 else 1}.get(groups[0][0], 0)] + [r for _, r in groups]
    return tuple((result + [0] * 6)[:6])


def public_features(public):
    board = cards(public["board"])
    streets = ["preflop", "flop", "turn", "river"]
    if public["street"] not in streets or len(board) != [0, 3, 4, 5][streets.index(public["street"])]:
        raise ValueError("board_street_mismatch")
    if public["position"] not in POSITIONS or public["opportunity"] not in OPPORTUNITIES:
        raise ValueError("unknown_context_must_be_explicit_UNKNOWN")
    values = [float(public["street"] == s) for s in streets]
    values += [float(public["position"] == p) for p in POSITIONS]
    values += [float(public["opportunity"] == p) for p in OPPORTUNITIES]
    for key in NUMERIC:
        value = public[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("missing_or_invalid_public_numeric:" + key)
        values.append(float(value))
    values += [len(board), max([RANKS.index(c[0]) + 2 for c in board] or [0]), len({c[0] for c in board}), max(Counter(c[1] for c in board).values(), default=0)]
    return values


def combo_features(public, hole):
    board = public["board"]
    cards(board + hole)
    high, low = sorted([RANKS.index(c[0]) + 2 for c in hole], reverse=True)
    best = max((five_score(list(x)) for x in itertools.combinations(hole + board, 5)), default=(-1, 0, 0, 0, 0, 0))
    ranks = {RANKS.index(c[0]) + 2 for c in hole + board}
    if 14 in ranks:
        ranks.add(1)
    missing = set()
    made = False
    for start in range(1, 11):
        absent = set(range(start, start + 5)) - ranks
        if not absent:
            made = True
        elif len(absent) == 1:
            value = next(iter(absent))
            missing.add(14 if value == 1 else value)
    return public_features(public) + [high, low, float(high == low), float(hole[0][1] == hole[1][1]), *best,
        max((sum(c[1] == h[1] for c in board) for h in hole), default=0), 0 if made else len(missing)]


def known_label(row):
    visibility = row["card_visibility"]
    state = visibility.get("state")
    if state in {"latent", "mucked", "folded_unshown", "source_missing"}:
        if visibility.get("hole_cards") not in (None, []):
            raise ValueError("latent_row_cannot_contain_invented_cards")
        return None
    if state != "revealed_after_decision" or len(cards(visibility.get("hole_cards"))) != 2:
        raise ValueError("unsupported_card_label_authority")
    if not visibility.get("source_event_id") or utc(visibility["available_utc"]) <= utc(row["decision_utc"]) or utc(visibility["available_utc"]) > utc(row["game_completed_utc"]):
        raise ValueError("invalid_reveal_chronology")
    cards(row["public"]["board"] + visibility["hole_cards"])
    return visibility["hole_cards"]


def hand_actor_identity(row):
    """Identify a dealt holding separately from its shared profile owner."""
    if "actor_pokernow_player_id" in row:
        actor = row["actor_pokernow_player_id"]
        if not isinstance(actor, str) or not actor:
            raise ValueError("exact_dealt_actor_identity_required")
        return "pokernow:" + actor
    # Older bounded fixtures have one actor identity and no source-ID field.
    actor = row["actor_id"]
    if not isinstance(actor, str) or not actor:
        raise ValueError("exact_identity_required")
    return "legacy:" + actor


def validate_rows(rows, protocol):
    unique, hand_labels, game_bounds = {}, {}, {}
    partitions = {name: set(protocol.get(name + "_game_ids", [])) for name in ("train", "calibration", "evaluation", "diagnostic")}
    if any(len(partitions[n]) != len(protocol.get(n + "_game_ids", [])) for n in partitions):
        raise ValueError("duplicate_cohort_game")
    if any(partitions[a] & partitions[b] for a, b in itertools.combinations(partitions, 2)):
        raise ValueError("whole_game_cohort_overlap")
    if partitions["evaluation"] & set(protocol.get("previously_inspected_game_ids", [])):
        raise ValueError("evaluation_contains_previously_inspected_game")
    for row in rows:
        if REQUIRED - row.keys():
            raise ValueError("missing_input_fields:" + ",".join(sorted(REQUIRED - row.keys())))
        if not all(isinstance(row[k], str) and row[k] for k in ["row_id", "game_id", "hand_id", "actor_id", "session_id"]):
            raise ValueError("exact_identity_required")
        if row["game_id"] not in set.union(*partitions.values()):
            raise ValueError("unassigned_game")
        if row["row_id"] in unique:
            if unique[row["row_id"]] != canonical(row):
                raise ValueError("conflicting_duplicate_row")
            continue
        unique[row["row_id"]] = canonical(row)
        if not (utc(row["game_started_utc"]) <= utc(row["public_available_utc"]) <= utc(row["decision_utc"]) < utc(row["action_observed_utc"]) <= utc(row["hand_completed_utc"]) <= utc(row["game_completed_utc"])):
            raise ValueError("noncausal_public_or_action_times")
        if not row["source_event_ids"] or len(set(row["source_event_ids"])) != len(row["source_event_ids"]):
            raise ValueError("unique_public_source_events_required")
        bounds = (row["game_started_utc"], row["game_completed_utc"])
        if game_bounds.setdefault(row["game_id"], bounds) != bounds:
            raise ValueError("inconsistent_game_bounds")
        public_features(row["public"])
        legal = row["legal_actions"]
        if not legal or len(set(legal)) != len(legal) or any(a not in ACTIONS for a in legal) or row["action"] not in legal:
            raise ValueError("illegal_observed_action")
        label = known_label(row)
        key = (row["game_id"], row["hand_id"], hand_actor_identity(row))
        if label and hand_labels.setdefault(key, tuple(sorted(label))) != tuple(sorted(label)):
            raise ValueError("conflicting_private_hand_labels")
        size = row.get("observed_action_size_pot_fraction")
        if row["action"] in {"bet", "raise", "all_in"} and (not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0):
            raise ValueError("observed_action_size_required")
    present = set(game_bounds)
    if present != set.union(*partitions.values()):
        raise ValueError("declared_game_missing_from_input")
    for left, right in [("train", "calibration"), ("calibration", "evaluation"), ("train", "evaluation")]:
        if partitions[left] and partitions[right] and max(utc(game_bounds[g][1]) for g in partitions[left]) >= min(utc(game_bounds[g][0]) for g in partitions[right]):
            raise ValueError("cohorts_not_chronological_whole_games")
    if protocol.get("schema_version") != VERSION or protocol.get("selection_assumption") != "revealed_subset_only_unidentified_population":
        raise ValueError("explicit_selection_contract_required")
    return [json.loads(v) for v in unique.values()]


def support_disposition(rows):
    missing = Counter()
    revealed = latent = 0
    for row in rows:
        missing.update(REQUIRED - row.keys())
        state = row.get("card_visibility", {}).get("state")
        revealed += state == "revealed_after_decision"
        latent += state in {"latent", "mucked", "folded_unshown", "source_missing"}
    return dict(schema_version=VERSION, rows=len(rows), games=len({r.get("game_id") for r in rows if r.get("game_id")}),
        required_fields_missing=dict(sorted(missing.items())), revealed_label_rows=revealed, latent_rows=latent,
        disposition="INPUT_CONTRACT_INCOMPLETE" if missing else "REVEAL_SELECTED_EXPERIMENT_ONLY" if revealed else "PRIVATE_COMBO_LIKELIHOOD_VALID_NEGATIVE_NO_LABELS",
        next_action_model_support="REQUIRES_VALIDATED_PUBLIC_ACTION_CONTRACT" if missing else "OBSERVED_ACTION_ONLY",
        limitations=["Reveal selection depends on hidden cards and later play; population identification is unavailable.",
          "Absent exact private cards remain latent, even when an ordinal showdown category exists.",
          "Observed next actions do not identify responses to unchosen sizes or continuation EV."], authority=AUTHORITY)


def size_label(row):
    action = row["action"]
    if action not in {"bet", "raise", "all_in"}:
        return action + "@none"
    fraction = row["observed_action_size_pot_fraction"]
    return action + "@" + ("small" if fraction < .5 else "medium" if fraction <= 1 else "large")


def legal_probabilities(probabilities, rows, labels):
    import numpy as np
    result = np.asarray(probabilities, dtype=float).copy()
    for i, row in enumerate(rows):
        result[i, [j for j, label in enumerate(labels) if label.split("@")[0] not in row["legal_actions"]]] = 0
        if not result[i].sum() > 0:
            raise ValueError("model_without_legal_mass")
        result[i] /= result[i].sum()
    return result


def loss_rows(probabilities, targets):
    import numpy as np
    p = np.clip(probabilities, 1e-15, 1)
    y = np.asarray(targets, dtype=int)
    one_hot = np.eye(p.shape[1])[y]
    return -np.log(p[np.arange(len(y)), y]), np.sum((p - one_hot) ** 2, axis=1)


def scores(probabilities, targets, rows, baseline, seed=20260904):
    import numpy as np
    if not rows:
        return {"rows": 0, "games": 0, "disposition": "NO_SUPPORT"}
    ll, brier = loss_rows(probabilities, targets)
    base_ll, base_brier = loss_rows(baseline, targets)
    games = sorted({r["game_id"] for r in rows})
    by_game = {g: [] for g in games}
    for i, row in enumerate(rows):
        by_game[row["game_id"]].append(i)
    confidence = np.max(probabilities, axis=1)
    correct = np.argmax(probabilities, axis=1) == targets
    prior_confidence = np.max(baseline, axis=1)
    prior_correct = np.argmax(baseline, axis=1) == targets

    def bins(conf, hits, indices):
        result = []
        conf, hits = conf[indices], hits[indices]
        for i in range(10):
            mask = (conf >= i / 10) & (conf < (i + 1) / 10 if i < 9 else conf <= 1)
            result.append(dict(bin_index=i, count=int(mask.sum()), confidence_sum=float(conf[mask].sum()), correct_count=int(hits[mask].sum())))
        return result

    def sufficient(indices, losses, briers, conf, hits):
        return dict(log_loss_sum=float(losses[indices].sum()), brier_sum=float(briers[indices].sum()),
            correct_count=int(hits[indices].sum()), calibration_bins=bins(conf, hits, indices))

    paired = [dict(game_id=game, row_count=len(indices),
        candidate=sufficient(indices, ll, brier, confidence, correct),
        prior=sufficient(indices, base_ll, base_brier, prior_confidence, prior_correct)) for game, indices in by_game.items()]
    totals = np.asarray([[item["row_count"], item["prior"]["log_loss_sum"]-item["candidate"]["log_loss_sum"], item["prior"]["brier_sum"]-item["candidate"]["brier_sum"]] for item in paired])
    rng = np.random.default_rng(seed)
    boots = totals[rng.integers(0, len(games), size=(2000, len(games)))].sum(axis=1)
    lower = np.quantile(boots[:, 1:] / boots[:, :1], .025 / 2, axis=0, method="linear")
    pooled_bins = dict(candidate=bins(confidence, correct, list(range(len(rows)))), prior=bins(prior_confidence, prior_correct, list(range(len(rows)))))
    ece = lambda values: sum(abs(item["confidence_sum"] - item["correct_count"]) for item in values) / len(rows)
    return dict(rows=len(rows), games=len(games), log_loss=float(ll.mean()), brier=float(brier.mean()),
        accuracy=float(correct.mean()), expected_calibration_error_10_bins=ece(pooled_bins["candidate"]),
        log_loss_improvement=float((base_ll - ll).mean()), brier_improvement=float((base_brier - brier).mean()),
        whole_game_lower_bounds={"log_loss": float(lower[0]), "brier": float(lower[1])},
        positive_game_fraction={"log_loss": float(np.mean(totals[:, 1] > 0)), "brier": float(np.mean(totals[:, 2] > 0)), "both": float(np.mean((totals[:, 1] > 0) & (totals[:, 2] > 0)))},
        prior_metrics=dict(log_loss=float(base_ll.mean()), brier=float(base_brier.mean()), accuracy=float(prior_correct.mean()), expected_calibration_error_10_bins=ece(pooled_bins["prior"])),
        paired_game_statistics=paired, calibration_bins=pooled_bins,
        bootstrap=dict(seed=int(seed), rng="numpy.random.PCG64", replicates=2000, lower_quantile=.025 / 2, quantile_method="linear",
            role_bonferroni_count=2, resampling_unit="whole_game", sampling="uniform_with_replacement", estimator="pooled_rows", game_order=games),
        interval_scope="whole-game bootstrap; two-role Bonferroni; revealed-subset condition where applicable",
        predictive_support_floor_met=len(games) >= 20,
        population_range_qualified=False, strategy_qualified=False)


def diagnostic_slices(probabilities, targets, rows, baseline, training_actors):
    import numpy as np
    ll, brier = loss_rows(probabilities, targets)
    bll, bbrier = loss_rows(baseline, targets)
    selectors = {
        "heads_up": lambda r: r["public"]["active_opponent_count_before"] == 1,
        "multiway": lambda r: r["public"]["active_opponent_count_before"] >= 2,
        "150_200bb": lambda r: 150 <= r["public"]["effective_stack_before_bb"] < 200,
        "200_250bb": lambda r: 200 <= r["public"]["effective_stack_before_bb"] <= 250,
        "over250bb": lambda r: r["public"]["effective_stack_before_bb"] > 250,
        "squeeze": lambda r: r["public"]["opportunity"] == "squeeze",
        "known_opponent": lambda r: r["actor_id"] in training_actors,
        "unseen_opponent": lambda r: r["actor_id"] not in training_actors,
    }
    selectors.update({"street_" + street: lambda r, s=street: r["public"]["street"] == s for street in ["preflop", "flop", "turn", "river"]})
    selectors.update({"position_" + position: lambda r, p=position: r["public"]["position"] == p for position in POSITIONS})
    result = {}
    for name, select in selectors.items():
        mask = np.asarray([select(r) for r in rows])
        result[name] = dict(rows=int(mask.sum()), games=len({r["game_id"] for r, keep in zip(rows, mask) if keep}),
            log_loss=float(ll[mask].mean()) if mask.any() else None, brier=float(brier[mask].mean()) if mask.any() else None,
            log_loss_improvement=float((bll - ll)[mask].mean()) if mask.any() else None,
            brier_improvement=float((bbrier - brier)[mask].mean()) if mask.any() else None,
            disposition="DESCRIPTIVE_SLICE_NO_ADDITIONAL_QUALIFICATION" if mask.any() else "NO_SUPPORT")
    return result


def fit_booster(x, y, labels, config):
    import lightgbm as lgb
    import numpy as np
    if lgb.__version__ != "4.6.0":
        raise ValueError("frozen_lightgbm_4_6_0_required")
    if not len(x) or len(set(y)) < 2:
        return None
    # Existing winning family reused for a NEW compatible target; no model search.
    model = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels),
        n_estimators=config["iterations"], num_leaves=config["num_leaves"], learning_rate=config["learning_rate"],
        min_child_samples=config["min_child_samples"], n_jobs=1, random_state=config["seed"],
        deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
    return model


def booster_export(model, labels, feature_names):
    return dict(labels=labels, feature_names=feature_names, fitted_class_indices=[int(x) for x in model.classes_],
        trees=model.booster_.dump_model(), inference="LightGBM raw tree sum then logistic/softmax; legality normalization")


def full_prediction(model, x, labels):
    import numpy as np
    raw = model.predict_proba(np.asarray(x, dtype=float))
    result = np.full((len(x), len(labels)), 1e-12)
    result[:, model.classes_.astype(int)] = raw
    return result / result.sum(axis=1, keepdims=True)


def temper(probabilities, temperature):
    import numpy as np
    result = np.clip(probabilities, 1e-15, 1) ** (1 / temperature)
    return result / result.sum(axis=1, keepdims=True)


def counts_prior(rows):
    # One row per unique hand/actor/entry context, not one pseudo-hand per action.
    result, seen = {}, set()
    for row in sorted(rows, key=lambda r: (r["decision_utc"], r["row_id"])):
        hole = known_label(row)
        if hole is None:
            continue
        context = prior_key(row["public"])
        identity = (row["game_id"], row["hand_id"], hand_actor_identity(row), context)
        if identity in seen:
            continue
        seen.add(identity)
        actor = result.setdefault(context, {}).setdefault(row["actor_id"], {})
        session = actor.setdefault(row["session_id"], {"counts": {}, "completed_through_utc": row["game_completed_utc"], "events": []})
        cls = hand_class(hole)
        session["counts"][cls] = session["counts"].get(cls, 0) + 1
        session["completed_through_utc"] = max(session["completed_through_utc"], row["game_completed_utc"])
        session["events"].append(row["row_id"])
    for actors in result.values():
        for sessions in actors.values():
            for evidence in sessions.values():
                evidence["events"].sort()
    return result


def prior_key(public):
    depth = public["effective_stack_before_bb"]
    band = "under150" if depth < 150 else "150_200" if depth < 200 else "200_250" if depth <= 250 else "over250"
    return "|".join([public["street"], public["position"], public["opportunity"], band])


def observed_public_support(rows):
    return {"categories": {key: sorted({r["public"][key] for r in rows}) for key in ["street", "position", "opportunity"]},
        "numeric_ranges": {key: [min(r["public"][key] for r in rows), max(r["public"][key] for r in rows)] for key in NUMERIC} if rows else {}}


def diagnostic_controls(protocol):
    controls=protocol.get("diagnostic_controls",DIAGNOSTIC_CONTROLS if protocol.get("synthetic_fixture") is True else None)
    if controls!=DIAGNOSTIC_CONTROLS:
        raise ValueError("explicit_frozen_profile_diagnostic_controls_required")
    return controls


def diagnostic_prior_history(full_counts, stale_counts, eligible_games, controls):
    import numpy as np
    actors=sorted({actor for context in full_counts.values() for actor in context})
    shuffled=np.random.default_rng(controls["permutation_seed"]).permutation(actors).tolist()
    wrong={actor:actors[(i+1)%len(actors)] for i,actor in enumerate(actors)}
    permutation=dict(zip(actors,shuffled))
    return dict(schema_version=VERSION+"_prior_controls",configuration=controls,stale_eligible_train_game_ids=sorted(eligible_games),
        stale_prior_counts=stale_counts,stale_prior_counts_sha256=hashlib.sha256(canonical(stale_counts).encode()).hexdigest(),
        training_profile_actor_ids=actors,wrong_player_map=wrong,shuffled_player_map=permutation,
        shuffled_fixed_points=[actor for actor in actors if permutation[actor]==actor],
        wrong_method="sorted_training_profile_actor_ids_rotate_one",shuffle_method="one_numpy_PCG64_permutation_fixed_points_retained",
        selection_assumption="selected_reveals_only; exact earlier complete train games; diagnostic only")


def fit_experiment(rows, protocol, checkpoint_dir=None, checkpoint_identity=None):
    import numpy as np
    rows = validate_rows(rows, protocol)
    controls=diagnostic_controls(protocol)
    config = protocol["model_config"]
    if set(config) != {"iterations", "num_leaves", "learning_rate", "min_child_samples", "seed"}:
        raise ValueError("frozen_model_configuration_required")
    if not (1 <= config["iterations"] <= 1000 and 2 <= config["num_leaves"] <= 127 and 0 < config["learning_rate"] <= 1 and config["min_child_samples"] >= 1):
        raise ValueError("bounded_model_configuration_required")
    groups = {name: [r for r in rows if r["game_id"] in protocol[name + "_game_ids"]] for name in ["train", "calibration", "evaluation"]}
    if not groups["train"]:
        raise ValueError("training_games_required")
    report = dict(schema_version=VERSION, source_support=support_disposition(rows), roles={}, authority=AUTHORITY)
    artifact = dict(schema_version=VERSION, selection_assumption=protocol["selection_assumption"], authority=AUTHORITY,
        trained_through_utc=max(r["game_completed_utc"] for r in groups["train"]),
        available_after_utc=max(r["game_completed_utc"] for r in groups["train"] + groups["calibration"]),
        trained_game_ids=protocol["train_game_ids"], calibration_game_ids=protocol["calibration_game_ids"], model_family="LightGBM_4.6.0_no_family_search",
        public_features=PUBLIC_NAMES, models={}, prior_counts=counts_prior(groups["train"]))
    artifact["public_support"] = observed_public_support(groups["train"])
    stale_rows=[row for row in groups["train"] if utc(row["game_completed_utc"])<utc(controls["stale_prior_cutoff_utc"])]
    artifact["diagnostic_prior_history"]=diagnostic_prior_history(artifact["prior_counts"],counts_prior(stale_rows),{row["game_id"] for row in stale_rows},controls)
    for role in ["physical_combo_action_likelihood", "observed_next_action_size"]:
        checkpoint_path = Path(checkpoint_dir) / (role + ".json") if checkpoint_dir else None
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint["binding"] != checkpoint_identity or checkpoint["payload_sha256"] != hashlib.sha256(canonical(checkpoint["payload"]).encode()).hexdigest():
                raise ValueError("range_role_checkpoint_identity_mismatch")
            stored = checkpoint["payload"]
            report["roles"][role] = stored["report"]
            if stored["model"]:
                artifact["models"][role] = stored["model"]
            print(canonical({"phase": "role_checkpoint_reused", "role": role}), flush=True)
            continue
        combo = role == "physical_combo_action_likelihood"
        train = [r for r in groups["train"] if not combo or known_label(r) is not None]
        labels = ACTIONS if combo else sorted({size_label(r) for r in groups["train"]})
        target = lambda row: row["action"] if combo else size_label(row)
        feature = lambda row: combo_features(row["public"], known_label(row)) if combo else public_features(row["public"])
        x, y = [feature(r) for r in train], [labels.index(target(r)) for r in train]
        model = fit_booster(x, y, labels, config)
        if model is None:
            report["roles"][role] = {"disposition": "INSUFFICIENT_OBSERVED_TARGET_SUPPORT", "rows": len(train), "authority": AUTHORITY}
            if checkpoint_path:
                payload = {"model": None, "report": report["roles"][role]}
                write_once(checkpoint_path, dict(binding=checkpoint_identity, payload_sha256=hashlib.sha256(canonical(payload).encode()).hexdigest(), payload=payload))
            continue
        exported = booster_export(model, labels, PUBLIC_NAMES + HOLE_NAMES if combo else PUBLIC_NAMES)
        exported["public_support"] = observed_public_support(train)
        calibration = [r for r in groups["calibration"] if (not combo or known_label(r) is not None) and target(r) in labels]
        temperature = 1.0
        if calibration:
            raw = full_prediction(model, [feature(r) for r in calibration], labels)
            temperature = min([.5, .75, 1.0, 1.25, 1.5, 2.0], key=lambda t: (
                float(loss_rows(legal_probabilities(temper(raw, t), calibration, labels), [labels.index(target(r)) for r in calibration])[0].mean()), abs(t - 1)))
        exported["temperature"] = temperature
        exported["temperature_selection"] = "calibration_whole_games_only; fixed_grid_0.5_0.75_1_1.25_1.5_2"
        artifact["models"][role] = exported
        prior = np.bincount(y, minlength=len(labels)).astype(float) + 1
        prior /= prior.sum()
        artifact["models"][role]["training_target_prior"] = prior.tolist()
        role_report = {"training_rows": len(train), "training_games": len({r["game_id"] for r in train}),
            "target": "P(observed actor action | hypothetical physical holding, public context, selected reveal)" if combo else "P(observed next action and size bucket | public context)",
            "counterfactual_response_identified": False, "calibration": {}, "evaluation": {}, "unsupported_target_rows": {}}
        # Calibration is reported independently; no evaluation-driven refit/selection.
        for name in ["calibration", "evaluation"]:
            candidates = [r for r in groups[name] if not combo or known_label(r) is not None]
            selected = [r for r in candidates if target(r) in labels]
            role_report["unsupported_target_rows"][name] = len(candidates) - len(selected)
            if not selected:
                role_report[name] = {"rows": 0, "games": 0, "disposition": "NO_SUPPORT"}
                continue
            probs = legal_probabilities(temper(full_prediction(model, [feature(r) for r in selected], labels), temperature), selected, labels)
            base = legal_probabilities(np.tile(prior, (len(selected), 1)), selected, labels)
            role_report[name] = scores(probs, [labels.index(target(r)) for r in selected], selected, base, config["seed"])
            role_report[name]["diagnostic_slices"] = diagnostic_slices(probs, [labels.index(target(r)) for r in selected], selected, base, {r["actor_id"] for r in train})
        role_report["disposition"] = "ASSUMPTION_DEPENDENT_EXPERIMENT_RETAIN_SHADOW"
        report["roles"][role] = role_report
        if checkpoint_path:
            payload = {"model": artifact["models"][role], "report": role_report}
            write_once(checkpoint_path, dict(binding=checkpoint_identity, payload_sha256=hashlib.sha256(canonical(payload).encode()).hexdigest(), payload=payload))
        print(canonical({"phase": "role_complete", "role": role, "training_rows": len(train)}), flush=True)
    # A continuation action model is not an outcome/CFV model. The existing exact
    # ledger/terminalAward interface can consume independently justified leaves.
    artifact["continuation_value"] = {"status": "NOT_IDENTIFIED", "reason": "unchosen_action_outcomes_and_human_propensities_unavailable", "supports_unchosen_action_ev": False}
    range_budget = protocol.get("range_diagnostic_rows_per_game", 0)
    if not isinstance(range_budget, int) or not 0 <= range_budget <= 200:
        raise ValueError("invalid_preregistered_range_diagnostic_budget")
    report["range_diagnostics"] = evaluate_ranges(artifact, groups["evaluation"], range_budget)
    return artifact, report


def tree_prediction(model, features):
    """Portable native-JSON scorer; parity-tested against fitted LightGBM."""
    trees = model["trees"]
    count = trees["num_tree_per_iteration"]
    raw = [0.0] * count
    for tree in trees["tree_info"]:
        node = tree["tree_structure"]
        while "leaf_value" not in node:
            value = features[node["split_feature"]]
            if node["decision_type"] != "<=":
                raise ValueError("unsupported_tree_split")
            left = node["default_left"] if value is None or math.isnan(value) else value <= float(node["threshold"])
            node = node["left_child"] if left else node["right_child"]
        raw[tree["tree_index"] % count] += node["leaf_value"]
    if count == 1:
        positive = 1 / (1 + math.exp(-max(-700, min(700, raw[0]))))
        values = [1 - positive, positive]
    else:
        values = [math.exp(value - max(raw)) for value in raw]
        total = sum(values)
        values = [value / total for value in values]
    result = [1e-12] * len(model["labels"])
    for cls, value in zip(model["fitted_class_indices"], values):
        result[cls] = value
    result = [max(1e-15, value) ** (1 / model.get("temperature", 1.0)) for value in result]
    total = sum(result)
    return [value / total for value in result]


def range_prior(artifact, request):
    """Hierarchical revealed-sample prior; no current-hand evidence is persisted."""
    blocked = cards(request["known_cards"])
    public_features(request["public"])
    if any(c not in blocked for c in request["public"]["board"]):
        raise ValueError("public_board_must_be_blocked")
    if utc(request["decision_utc"]) <= utc(artifact["available_after_utc"]):
        raise ValueError("model_not_available_before_decision")
    if not (utc(artifact["available_after_utc"]) < utc(request["hand_started_utc"]) <= utc(request["decision_utc"])):
        raise ValueError("model_must_predate_current_hand")
    actor, session_id = request["opponent_id"], request["session_id"]
    if not actor or not session_id:
        raise ValueError("opponent_and_session_identity_required")
    strengths = request["prior_strengths"]
    if set(strengths) != {"population", "player", "session"} or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in strengths.values()):
        raise ValueError("positive_frozen_prior_strengths_required")
    partitions = {"population": Counter(), "player": Counter(), "session": Counter()}
    roles = request.get("prior_roles", ["population", "player", "session"])
    if not isinstance(roles, list) or len(set(roles)) != len(roles) or any(r not in partitions for r in roles):
        raise ValueError("invalid_prior_ablation_roles")
    sources = artifact["prior_counts"].get(prior_key(request["public"]), {})
    events = set()
    for player, sessions in sources.items():
        for sess, evidence in sessions.items():
            if utc(evidence["completed_through_utc"]) >= utc(request["decision_utc"]):
                raise ValueError("noncausal_prior_evidence")
            if events & set(evidence["events"]):
                raise ValueError("duplicate_prior_evidence")
            events.update(evidence["events"])
            role = "population" if player != actor else "session" if sess == session_id else "player"
            partitions[role].update(evidence["counts"])
    current_evidence = request.get("completed_reveal_evidence", [])
    if not isinstance(current_evidence, list) or len(current_evidence) > 10000:
        raise ValueError("bounded_completed_reveal_evidence_required")
    hand_contexts = set()
    for evidence in current_evidence:
        key = evidence["row_id"]
        if not key or key in events or not evidence["source_event_id"]:
            raise ValueError("duplicate_or_missing_completed_evidence_identity")
        events.add(key)
        if evidence["game_id"] in artifact["trained_game_ids"] + artifact["calibration_game_ids"] or evidence["hand_id"] == request["hand_id"]:
            raise ValueError("training_or_current_hand_evidence_reused")
        if not (utc(evidence["completed_utc"]) < utc(request["decision_utc"]) and utc(evidence["available_utc"]) < utc(request["decision_utc"])):
            raise ValueError("completed_reveal_evidence_not_predecision")
        public_features(evidence["public"])
        hole = cards(evidence["hole_cards"])
        if len(hole) != 2:
            raise ValueError("exact_completed_reveal_cards_required")
        cards(evidence["public"]["board"] + hole)
        context = prior_key(evidence["public"])
        hand_context = (evidence["game_id"], evidence["hand_id"], evidence["opponent_id"], context)
        if hand_context in hand_contexts:
            raise ValueError("same_completed_hand_context_counted_twice")
        hand_contexts.add(hand_context)
        if context != prior_key(request["public"]):
            continue
        role = "population" if evidence["opponent_id"] != actor else "session" if evidence["session_id"] == session_id else "player"
        partitions[role][hand_class(hole)] += 1
    combos = [list(c) for c in itertools.combinations([c for c in DECK if c not in blocked], 2)]
    if not combos:
        raise ValueError("empty_legal_range")
    multiplicities = Counter(hand_class(c) for c in combos)
    weights = [1 / len(combos)] * len(combos)
    prior_contribution = 1.0
    support = {}
    for role, counts in partitions.items():
        # Class labels distribute over surviving physical combos uniformly: suit
        # exchangeability is explicit; cards/blockers never silently disappear.
        usable = sum(v for k, v in counts.items() if k in multiplicities)
        support[role] = usable
        if usable and role in roles:
            blend = usable / (usable + strengths[role])
            weights = [(1 - blend) * w + blend * counts[hand_class(c)] / multiplicities[hand_class(c)] / usable for c, w in zip(combos, weights)]
            prior_contribution *= 1 - blend
    return combos, weights, support, prior_contribution


def infer_belief(artifact, request):
    if artifact.get("schema_version") != VERSION or artifact.get("authority") != AUTHORITY:
        raise ValueError("unrecognized_or_authority_modified_artifact")
    combos, weights, support, prior_contribution = range_prior(artifact, request)
    model = artifact["models"].get("physical_combo_action_likelihood")
    seen, previous, applied, abstentions = {}, -math.inf, [], []
    for event in request.get("actions", []):
        key = event["event_id"]
        if not key:
            raise ValueError("event_identity_required")
        if key in seen:
            if seen[key] != canonical(event):
                raise ValueError("conflicting_duplicate_action")
            continue
        seen[key] = canonical(event)
        when = utc(event["observed_utc"])
        if when < previous or when >= utc(request["decision_utc"]) or utc(event["public_available_utc"]) >= when:
            raise ValueError("noncausal_action_prefix")
        previous = when
        if event["hand_id"] != request["hand_id"] or event["opponent_id"] != request["opponent_id"]:
            raise ValueError("wrong_hand_or_opponent_prefix")
        if when < utc(request["hand_started_utc"]) or request["public"]["board"][:len(event["public"]["board"])] != event["public"]["board"]:
            raise ValueError("action_board_not_current_hand_prefix")
        if not model:
            abstentions.append({"event_id": key, "reason": "no_physical_combo_likelihood_model"})
            continue
        unsupported = public_support_reasons(artifact, event["public"], "physical_combo_action_likelihood")
        if unsupported:
            abstentions.append({"event_id": key, "reason": "out_of_training_public_support", "fields": unsupported})
            continue
        if event["action"] not in model["labels"]:
            raise ValueError("unmodeled_action")
        # Action likelihood is an assumed selected-reveal conditional model;
        # report sensitivity by running the same request with strength 0 and 1.
        strength = request["likelihood_strength"]
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("bounded_assumption_strength_required")
        if event["action"] not in event["legal_actions"] or any(a not in ACTIONS for a in event["legal_actions"]):
            raise ValueError("invalid_action_prefix_legality")
        probabilities = []
        for combo in combos:
            prediction = tree_prediction(model, combo_features(event["public"], combo))
            legal_mass = sum(p for action, p in zip(model["labels"], prediction) if action in event["legal_actions"])
            probabilities.append(prediction[model["labels"].index(event["action"])] / legal_mass)
        weights = [w * ((1 - strength) + strength * p) for w, p in zip(weights, probabilities)]
        total = sum(weights)
        if not total > 0:
            raise ValueError("empty_posterior")
        weights = [w / total for w in weights]
        applied.append(key)
    return dict(schema_version=VERSION + "_belief", rows=[dict(cards=c, weight=w) for c, w in zip(combos, weights)],
        support=support, prior_contribution=prior_contribution, effective_combo_count=1 / sum(w * w for w in weights),
        applied_event_ids=applied, hand_id=request["hand_id"], opponent_id=request["opponent_id"],
        abstentions=abstentions,
        uncertainty="selected_reveal_transport_and_suit_exchangeability_unidentified",
        freshness=dict(model_trained_through_utc=artifact["trained_through_utc"], model_available_after_utc=artifact["available_after_utc"], decision_utc=request["decision_utc"]),
        persistent_update=False, drift_state="not_estimated", authority=AUTHORITY)


def public_support_reasons(artifact, public, role=None):
    public_features(public)
    support = artifact.get("models", {}).get(role, {}).get("public_support") if role else artifact.get("public_support")
    if support is None:
        return ["missing_role_public_support"]
    reasons = [key for key, values in support["categories"].items() if public[key] not in values]
    reasons += [key for key, (low, high) in support["numeric_ranges"].items() if not low <= public[key] <= high]
    return reasons


def predict_observed_response(artifact, request):
    if artifact.get("schema_version") != VERSION or artifact.get("authority") != AUTHORITY:
        raise ValueError("unrecognized_or_authority_modified_artifact")
    if utc(request["decision_utc"]) <= utc(artifact["available_after_utc"]):
        raise ValueError("model_not_available_before_decision")
    reasons = public_support_reasons(artifact, request["public"], "observed_next_action_size")
    model = artifact["models"].get("observed_next_action_size")
    if reasons or not model:
        return dict(status="ABSTAIN", reason="outside_training_support" if reasons else "model_unavailable", fields=reasons, authority=AUTHORITY)
    legal = request["legal_actions"]
    if not legal or len(set(legal)) != len(legal) or any(a not in ACTIONS for a in legal):
        raise ValueError("invalid_response_legality")
    p = tree_prediction(model, public_features(request["public"]))
    pairs = [(label, value) for label, value in zip(model["labels"], p) if label.split("@")[0] in legal]
    total = sum(value for _, value in pairs)
    if not total:
        return dict(status="ABSTAIN", reason="no_supported_legal_response_class", authority=AUTHORITY)
    return dict(status="READY_DIAGNOSTIC", probabilities={label: value / total for label, value in pairs},
        target="observed_next_actor_action_and_size", counterfactual_intervention_identified=False,
        expected_continuation_ev=None, joint_opponent_dependence="Only the next actor is modeled; no independent-opponent product.", authority=AUTHORITY)


def evaluate_ranges(artifact, rows, per_game):
    """Fixed revealed sample and preregistered profile-integrity controls."""
    if not per_game or "physical_combo_action_likelihood" not in artifact["models"]:
        return dict(status="NOT_RUN", reason="preregistered_budget_or_exact_card_likelihood_missing", authority=AUTHORITY)
    controls=artifact["diagnostic_prior_history"]
    if controls["configuration"]!=DIAGNOSTIC_CONTROLS:
        raise ValueError("profile_control_configuration_changed")
    full_counts=artifact["prior_counts"]
    def permuted(mapping):
        return {context:{target:evidence[donor] for target,donor in mapping.items() if donor in evidence} for context,evidence in full_counts.items()}
    variants={"current":artifact,"stale":dict(artifact,prior_counts=controls["stale_prior_counts"]),
        "wrong":dict(artifact,prior_counts=permuted(controls["wrong_player_map"])),
        "shuffled":dict(artifact,prior_counts=permuted(controls["shuffled_player_map"]))}
    candidates = [r for r in rows if known_label(r) is not None]
    selected = []
    for game in sorted({r["game_id"] for r in candidates}):
        selected.extend(sorted([r for r in candidates if r["game_id"] == game], key=lambda r: hashlib.sha256(r["row_id"].encode()).hexdigest())[:per_game])
    scenarios = {"uniform_no_profile": ([],0,"current"), "population_only": (["population"],0,"current"),
        "current_hand_likelihood_off": (["population","player","session"],0,"current"),
        "player_and_action_assumption": (["population","player","session"],1,"current"),
        "stale_profile_and_action": (["population","player","session"],1,"stale"),
        "wrong_player_profile_and_action": (["population","player","session"],1,"wrong"),
        "shuffled_player_profile_and_action": (["population","player","session"],1,"shuffled")}
    evidence = []
    for row in selected:
        past = sorted([r for r in rows if r["hand_id"] == row["hand_id"] and r["game_id"] == row["game_id"] and hand_actor_identity(r) == hand_actor_identity(row) and utc(r["action_observed_utc"]) < utc(row["decision_utc"])], key=lambda r: (r["action_observed_utc"], r["row_id"]))
        request = dict(known_cards=row["public"]["board"], public=row["public"], opponent_id=row["actor_id"], session_id=row["session_id"],
            hand_id=row["hand_id"], hand_started_utc=row.get("hand_started_utc", row["game_started_utc"]), decision_utc=row["decision_utc"],
            prior_strengths=dict(population=100, player=100, session=100),
            actions=[dict(event_id=r["row_id"], opponent_id=r["actor_id"], hand_id=r["hand_id"], observed_utc=r["action_observed_utc"],
                public_available_utc=r["public_available_utc"], public=r["public"], action=r["action"], legal_actions=r["legal_actions"]) for r in past])
        observed = set(known_label(row));context=prior_key(row["public"]);actor=row["actor_id"]
        values = {}
        for name, (roles,strength,variant) in scenarios.items():
            held=None;donor=None
            if variant=="stale":
                if not controls["stale_eligible_train_game_ids"]:held="no_complete_train_games_before_frozen_stale_cutoff"
                elif not variants["stale"]["prior_counts"].get(context):held="no_stale_revealed_prior_support_in_context"
                elif canonical(variants["stale"]["prior_counts"].get(context,{}))==canonical(full_counts.get(context,{})):held="no_newer_training_prior_evidence_in_context"
            elif variant in {"wrong","shuffled"}:
                mapping=controls["wrong_player_map" if variant=="wrong" else "shuffled_player_map"]
                donor=mapping.get(actor)
                if donor is None:held="target_has_no_training_profile"
                elif donor==actor:held="fixed_point_retained_without_reshuffling"
                elif actor not in full_counts.get(context,{}) or donor not in full_counts.get(context,{}):held="no_matched_target_and_donor_profile_in_context"
            if held:
                values[name]=dict(status="HELD",hold_reason=held,profile_donor_id=donor,log_loss=None,brier=None,applied_actions=None,effective_combo_count=None)
                continue
            result = infer_belief(variants[variant],dict(request,prior_roles=roles,likelihood_strength=strength))
            probability = next(r["weight"] for r in result["rows"] if set(r["cards"]) == observed)
            values[name] = dict(status="SCORED",profile_donor_id=donor,log_loss=-math.log(max(1e-15, probability)), brier=1 + sum(r["weight"] ** 2 for r in result["rows"]) - 2 * probability,
                applied_actions=len(result["applied_event_ids"]), effective_combo_count=result["effective_combo_count"],support=result["support"])
        evidence.append(dict(row_id=row["row_id"], game_id=row["game_id"],opponent_id=actor,prior_context=context,scenarios=values))
    summaries={}
    for name in scenarios:
        scored=[item for item in evidence if item["scenarios"][name]["status"]=="SCORED"]
        held=[item for item in evidence if item["scenarios"][name]["status"]=="HELD"]
        summaries[name]=dict(sampled_rows=len(evidence),scored_rows=len(scored),held_rows=len(held),
            hold_reasons=dict(Counter(item["scenarios"][name]["hold_reason"] for item in held)),
            metrics_denominator="scored_rows_only; held rows retained in full fixed sample",**{metric:sum(item["scenarios"][name][metric] for item in scored)/len(scored) if scored else None for metric in ["log_loss","brier"]},
            matched_full_minus_control={metric:sum(item["scenarios"]["player_and_action_assumption"][metric]-item["scenarios"][name][metric] for item in scored)/len(scored) if scored else None for metric in ["log_loss","brier"]})
    sample=[dict(row_id=row["row_id"],game_id=row["game_id"]) for row in evidence]
    return dict(status="REVEAL_SELECTED_DIAGNOSTIC", eligible_revealed_rows=len(candidates), sampled_rows=len(evidence),
        sampled_games=len({r["game_id"] for r in evidence}), sample_rule="lowest_sha256_row_id_within_game_before_label_scoring",
        sample_membership_sha256=hashlib.sha256(canonical(sample).encode()).hexdigest(),
        per_game_budget=per_game, prior_strengths_assumed_not_optimized={"population": 100, "player": 100, "session": 100},
        diagnostic_controls=controls["configuration"],profile_control_provenance={key:controls[key] for key in controls if key!="stale_prior_counts"},
        control_definitions=dict(current_hand_likelihood_off="Remove current-hand action likelihood only; same trained models and prior evidence, no reduced-feature refit.",
            stale_profile_and_action="Use only whole train games completed before the fixed stale-history cutoff; original actions retained.",
            wrong_player_profile_and_action="Fixed sorted-ID rotation of prior ownership; actual actor, contexts and action prefix unchanged.",
            shuffled_player_profile_and_action="One fixed seeded training-player bijection; fixed points held and never reshuffled using outcomes."),
        results=evidence, summary=summaries,
        limitations=["Revealed subset only; no population range inference is validated.", "Fixed sensitivity ablations are assumptions, not learned strategy weights.",
            "No effect interval or multiple-comparison qualification is assigned to exploratory range diagnostics.","Held controls remain in the fixed denominator; conditional scored means are not full-sample performance estimates."], authority=AUTHORITY)


def write_once(path, value):
    data = (canonical(value) + "\n").encode()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("refuse_overwrite_different_artifact:" + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial=path.with_name(path.name+".partial."+uuid.uuid4().hex)
    with partial.open("xb") as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    if partial.read_bytes()!=data:
        raise ValueError("closed_partial_identity_mismatch")
    try:os.link(partial,path)
    except FileExistsError:
        if path.read_bytes()!=data:raise ValueError("refuse_overwrite_different_artifact:"+str(path))
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["support", "fit", "infer", "response"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="exercise the disk-backed full-corpus path on a bounded fixture")
    args = parser.parse_args()
    if args.mode in {"infer", "response"}:
        if not args.artifact or sha(args.artifact) != args.artifact_sha256:
            raise ValueError("authenticate_artifact_before_inference")
        predictor = infer_belief if args.mode == "infer" else predict_observed_response
        write_once(args.output, predictor(json.loads(args.artifact.read_text()), json.loads(args.input.read_text())))
        return
    if args.mode == "support":
        missing, games, revealed, latent, total = Counter(), set(), 0, 0, 0
        with args.input.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line); total += 1; missing.update(REQUIRED - row.keys())
                if row.get("game_id"): games.add(row["game_id"])
                state = row.get("card_visibility", {}).get("state")
                revealed += state == "revealed_after_decision"; latent += state in {"latent","mucked","folded_unshown","source_missing"}
        result = support_disposition([])
        result.update(rows=total,games=len(games),required_fields_missing=dict(missing),revealed_label_rows=revealed,latent_rows=latent,
            disposition="INPUT_CONTRACT_INCOMPLETE" if missing else "REVEAL_SELECTED_EXPERIMENT_ONLY" if revealed else "PRIVATE_COMBO_LIKELIHOOD_VALID_NEGATIVE_NO_LABELS")
        write_once(args.output, result)
        return
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha(args.input) != protocol["input_sha256"] or sha(Path(__file__)) != protocol["source_code_sha256"]:
        raise ValueError("direct_input_or_source_hash_mismatch")
    if args.fixture:
        rows = []
        with args.input.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip(): rows.append(json.loads(line))
                if len(rows) > 1000: raise ValueError("bounded_synthetic_fixture_only")
        if protocol.get("synthetic_fixture") is not True:
            raise ValueError("bounded_synthetic_fixture_only")
    elif not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG"):
        raise ValueError("full_training_colab_only")
    start = time.monotonic()
    print(canonical({"phase": "fit", "fixture_rows": len(rows) if args.fixture else None, "model_family": "LightGBM", "inner_threads": 1}), flush=True)
    binding = dict(input_sha256=sha(args.input), protocol_sha256=sha(args.protocol), source_code_sha256=sha(Path(__file__)), synthetic_fixture=args.fixture)
    if not args.fixture or args.streaming:
        streaming_source = Path(__file__).with_name("stage6_empirical_range_learning_v2_streaming.py")
        if protocol.get("streaming_code_sha256") != sha(streaming_source):
            raise ValueError("direct_streaming_code_identity_mismatch")
        binding["streaming_code_sha256"] = sha(streaming_source)
    completion = args.output / "completion_receipt.json"
    if completion.exists():
        previous = json.loads(completion.read_text(encoding="utf-8"))
        if previous.get("lineage") != binding or previous.get("authority") != AUTHORITY:
            raise ValueError("completed_experiment_identity_mismatch")
        for name, identity in previous["members"].items():
            if Path(name).name != name or sha(args.output / name) != identity["sha256"] or (args.output / name).stat().st_size != identity["bytes"]:
                raise ValueError("completed_experiment_member_mismatch")
        print(canonical({"phase": "completed_experiment_reused", "scientific_recomputation": False}), flush=True)
        return
    fit_started_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")
    stop = threading.Event()
    def progress():
        while not stop.wait(20):
            print(canonical({"phase": "running", "elapsed_seconds": round(time.monotonic() - start, 1), "completed_role_checkpoints": len(list((args.output / "checkpoints").glob("*.json")))}), flush=True)
    monitor = threading.Thread(target=progress, daemon=True); monitor.start()
    try:
        if not args.fixture or args.streaming:
            from stage6_empirical_range_learning_v2_streaming import fit_parallel
            artifact, report = fit_parallel(args.input, protocol, args.output, binding)
        else:
            artifact, report = fit_experiment(rows, protocol, args.output / "checkpoints", binding)
    finally:
        stop.set(); monitor.join(timeout=1)
    artifact["lineage"] = binding
    report["lineage"] = artifact["lineage"]
    write_once(args.output / "range_response_artifact.json", artifact)
    write_once(args.output / "range_response_report.json", report)
    write_once(args.output / "completion_receipt.json", dict(schema_version=VERSION, status="COMPLETE_EXPERIMENT", authority=AUTHORITY,
        members={p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in [args.output / "range_response_artifact.json", args.output / "range_response_report.json"]}, lineage=artifact["lineage"],
        execution_provenance=dict(fit_invocation_started_utc=fit_started_utc,fit_completed_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z"),
            available_after_semantics="source_and_calibration_cutoff",historical_deployment_evidence=False)))
    print(canonical({"phase": "complete", "rows": report["source_support"]["rows"], "seconds": time.monotonic() - start, "authority": AUTHORITY}), flush=True)


if __name__ == "__main__":
    main()
