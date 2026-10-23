"""Frozen-winner empirical evaluation; no fitting, solver or automatic promotion.

The same prepared rows feed every comparator. Persisted per-layer predictions
permit independent metric reproduction and future role-specific comparisons.
Full-corpus execution belongs in Colab; local validation uses bounded fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd

from stage6_empirical_action_semantics import aggression_role

VERSION = "stage6_empirical_evaluation_v2"
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
EXCLUDED = ("opponent_bluff_likelihood", "showdown_aware_range_calibration")


def sha(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace different evidence: {path}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(raw)
    partial.replace(path)


def progress(phase: str, **values: Any) -> None:
    print(json.dumps({"phase": phase, **values}), flush=True)


def decision_context(decisions: pd.DataFrame) -> pd.DataFrame:
    """Exact actor target with monetary all-in classification and unknown fallback."""
    output = []
    ordered = decisions.sort_values(["game_id", "hand_id", "decision_order", "row_id"], kind="stable")
    for _, hand in ordered.groupby(["game_id", "hand_id"], sort=False):
        aggressors: dict[str, str] = {}
        for row in hand.to_dict("records"):
            street = str(row["street"])
            opponents = json.loads(row["active_opponent_canonical_player_ids_before"])
            facing = bool(row["facing_bet_or_raise"])
            target = aggressors.get(street) if facing else opponents[0] if len(opponents) == 1 else None
            if target not in opponents:
                target = None
            pot, committed = float(row["pot_before_bb"]), float(row["actor_committed_before_bb"])
            minimum = row["legal_min_raise_to_bb"] if facing else row["legal_min_bet_to_bb"]
            size = None if pd.isna(minimum) or pot <= 0 else max(0.0, float(minimum) - committed) / pot
            output.append({key: row[key] for key in ("row_id", "game_id", "hand_id", "decision_utc", "actor_id",
                "active_opponent_canonical_player_ids_before", "active_opponent_count_before")} | {
                "target_opponent_canonical_player_ids_before": None if target is None else json.dumps([target]),
                "target_opponent_count_before": int(target is not None), "candidate_bet_size_pot_fraction": size})
            role = aggression_role(row)
            if role is True:
                aggressors[street] = str(row["actor_id"])
            elif role is None:
                aggressors.pop(street, None)
    return pd.DataFrame(output).set_index("row_id").loc[decisions["row_id"]].reset_index()


def whole_game_cohorts(decisions: pd.DataFrame, history: pd.DataFrame, protocol: dict) -> pd.Series:
    clocks = pd.concat([decisions[["game_id", "decision_utc"]], history[["game_id", "decision_utc"]]])
    clocks["decision_utc"] = pd.to_datetime(clocks["decision_utc"], utc=True, errors="raise")
    starts = clocks.groupby("game_id")["decision_utc"].min()
    for game,value in protocol.get('source_game_start_utc',{}).items():
        if game in starts:
            starts.loc[game]=min(starts.loc[game],pd.Timestamp(value))
    inspected = set(protocol["previously_inspected_game_ids"])
    training = set(protocol["training_game_ids"])
    development = set(protocol["development_game_ids"])
    final = set(protocol["confirmation_game_ids"])
    if final & (inspected | training | development):
        raise ValueError("Confirmation games overlap inspected, training or development data")
    if not set(decisions.game_id).issubset(inspected | training | development | final):
        raise ValueError("Every evaluated whole game needs an explicit cohort assignment")
    freeze = pd.Timestamp(protocol["candidate_frozen_utc"])
    if freeze.tzinfo is None:
        raise ValueError("Candidate freeze requires an explicit timezone")
    design=protocol.get('confirmation_design','prospective_after_candidate_freeze')
    if design=='prospective_after_candidate_freeze':
        boundary=freeze
    elif design=='locked_retrospective_unseen_games':
        if protocol.get('outcomes_inspected_before_protocol_freeze') is not False:
            raise ValueError('Retrospective holdout requires explicit uninspected outcomes')
        boundary=pd.Timestamp(protocol['model_training_cutoff_utc'])
        if boundary.tzinfo is None or boundary>freeze:raise ValueError('Invalid model training cutoff')
        if not final.issubset(protocol.get('source_game_start_utc',{})):
            raise ValueError('Retrospective holdout requires exact source game starts')
    else:raise ValueError('Unknown confirmation design')
    if any(starts[game] <= boundary for game in final if game in starts):
        raise ValueError("A confirmation game began before the candidate freeze or training boundary")
    return decisions.game_id.map(lambda game: "confirmation" if game in final else "training_overlap" if game in training else "historical_development")


def paired_metrics(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, games: np.ndarray,
                   *, alpha: float = .05, replicates: int = 2000, seed: int = 260904) -> dict:
    if not len(target):
        return {"status": "NO_SUPPORTED_ROWS", "rows": 0, "game_clusters": 0}
    for values in (prediction, baseline):
        if values.shape != (len(target), len(ACTIONS)) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid probability matrix")
        if not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Probabilities do not sum to one")
    onehot = np.eye(len(ACTIONS))[target]
    loss = -np.log(np.clip(prediction[np.arange(len(target)), target], 1e-15, 1))
    base_loss = -np.log(np.clip(baseline[np.arange(len(target)), target], 1e-15, 1))
    brier = ((prediction - onehot) ** 2).sum(axis=1)
    base_brier = ((baseline - onehot) ** 2).sum(axis=1)
    grouped = pd.DataFrame({"game": games, "ll": base_loss-loss, "bs": base_brier-brier}).groupby("game", sort=True).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = []
    sums = grouped[[('ll', 'sum'), ('bs', 'sum')]].to_numpy()
    counts = grouped[('ll', 'count')].to_numpy()
    for _ in range(replicates):
        selected = rng.integers(len(grouped), size=len(grouped))
        draws.append(sums[selected].sum(axis=0) / counts[selected].sum())
    intervals = np.quantile(draws, [alpha / 2, 1-alpha / 2], axis=0)
    confidence = prediction.max(axis=1)
    correct = prediction.argmax(axis=1) == target
    calibration = []
    for lower in np.arange(0, 1, .1):
        mask = (confidence >= lower) & (confidence < lower+.1 if lower < .9 else confidence <= 1)
        if mask.any():
            calibration.append({"lower": float(lower), "rows": int(mask.sum()),
                "confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    ece = sum(row['rows'] * abs(row['confidence']-row['accuracy']) for row in calibration) / len(target)
    return {"rows": len(target), "game_clusters": len(grouped), "log_loss": float(loss.mean()),
        "baseline_log_loss": float(base_loss.mean()), "log_loss_improvement": float((base_loss-loss).mean()),
        "brier": float(brier.mean()), "baseline_brier": float(base_brier.mean()),
        "brier_improvement": float((base_brier-brier).mean()), "accuracy": float(correct.mean()),
        "log_loss_improvement_interval": intervals[:, 0].tolist(), "brier_improvement_interval": intervals[:, 1].tolist(),
        "interval_alpha": alpha, "game_weighting": "row_weighted_paired_whole_game_bootstrap",
        "positive_game_fraction": float((grouped[('ll', 'sum')] > 0).mean()),
        "ece_10_bins": ece, "calibration_bins": calibration}


def layer_disposition(layer: str, metrics: dict, protocol: dict) -> dict:
    confirmation = metrics.get("confirmation", {}).get("frozen_prior", {})
    reasons = []
    if protocol.get('synthetic_canary') is True:
        reasons.append('bounded_canary_is_not_qualification_evidence')
    if confirmation.get("game_clusters", 0) < protocol["minimum_confirmation_games"]:
        reasons.append("insufficient_uninspected_confirmation_games")
    if confirmation.get("log_loss_improvement_interval", [-math.inf])[0] <= 0:
        reasons.append("no_positive_multiplicity_adjusted_log_loss_bound")
    if confirmation.get("brier_improvement_interval", [-math.inf])[0] <= 0:
        reasons.append("no_positive_multiplicity_adjusted_brier_bound")
    if confirmation.get("positive_game_fraction", 0) <= protocol["minimum_positive_game_fraction"]:
        reasons.append("improvement_not_consistent_across_games")
    prediction_pass = not reasons
    if layer in EXCLUDED:
        reasons.append("existing_operator_exclusion_retained")
    # This experiment measures observed actor choices, not the value of Hero's
    # unchosen actions. No action strategy is promoted from this report alone.
    return {"predictive_qualification": prediction_pass,
        "prediction_role": "observed_actor_action_distribution",
        "strategy_qualification": False, "main_recommendation_promotion": False,
        "disposition": "PREDICTION_QUALIFIED_STRATEGY_UNEVALUATED" if prediction_pass and layer not in EXCLUDED else "RETAIN_SHADOW",
        "reasons": reasons + ["no_identified_counterfactual_or_validated_strategy_comparison"]}


def score_layer(task: dict) -> dict:
    import joblib
    from threadpoolctl import threadpool_limits
    import stage6_shadow_layer_fresh_backtest as old
    layer, model_path, matrix_path = task["layer"], Path(task["model_path"]), Path(task["matrix_path"])
    raw = model_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != task["model_sha256"] or sha(matrix_path) != task["matrix_sha256"]:
        raise ValueError("Frozen model or feature matrix changed")
    with threadpool_limits(limits=1):
        package = joblib.load(io.BytesIO(raw))
        if list(package["feature_names"]) != task["feature_names"]:
            raise ValueError("Feature-order mismatch")
        with np.load(matrix_path, allow_pickle=False) as data:
            probabilities = old.calibrated_legal_probabilities(package, data["matrix"], data["legal"])
            prior = old.legal_prior(package, data["legal"])
            output = Path(task["output_path"])
            partial = output.with_suffix('.partial')
            with partial.open('wb') as handle:
                np.savez_compressed(handle, probabilities=probabilities, prior=prior, support=data["support"])
            partial.replace(output)
    return {"layer": layer, "prediction_sha256": sha(output), "rows": len(probabilities), "model_sha256": task["model_sha256"]}


def _run(snapshot_root: Path, output: Path, *, workers: int = 1) -> dict:
    import stage6_shadow_layer_fresh_backtest as old
    snapshot_root, output = Path(snapshot_root), Path(output)
    protocol_path = snapshot_root / 'protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    if not protocol.get('synthetic_canary') and not Path('/content').is_dir():
        raise ValueError('Full evaluation belongs in Colab; local execution requires an explicit bounded canary protocol')
    workspace = Path(__file__).resolve().parents[3]
    for relative, identity in protocol['code_files'].items():
        path = workspace/relative
        if path.is_symlink() or sha(path) != identity['sha256']:
            raise ValueError(f'Direct code identity changed: {relative}')
    for name, identity in protocol['files'].items():
        path = snapshot_root/name
        if path.is_symlink() or path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
            raise ValueError(f'Input authentication failed: {name}')
    authentication, model_bytes = old.authenticate_frozen_installation()
    if protocol['installed_manifest_sha256'] != authentication['manifest_sha256']:
        raise ValueError('Model selection identity mismatch')
    run_identity = hashlib.sha256(canonical({'protocol': sha(protocol_path), 'direct_code': protocol['code_files']})).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'run_identity.json', {'run_id': run_identity})
    completed = output/'report.json'
    if completed.exists():
        result = json.loads(completed.read_bytes())
        receipt = json.loads((output/'report_receipt.json').read_bytes())
        if result['run_id'] != run_identity or receipt['sha256'] != sha(completed):
            raise ValueError('Completed report authentication failed')
        return result
    decisions = pd.read_parquet(snapshot_root/'decisions.parquet')
    history = pd.read_parquet(snapshot_root/'history.parquet')
    if decisions.row_id.duplicated().any() or decisions[['game_id','hand_id','decision_id']].duplicated().any():
        raise ValueError('Duplicate decision identity')
    for column in ('future_reveal_used', 'post_decision_outcome_used', 'private_card_features_used'):
        if decisions[column].fillna(True).astype(bool).any():
            raise ValueError(f'Unexpected information boundary: {column}')
    cohorts = whole_game_cohorts(decisions, history, protocol)
    legal = decisions[[f'legal_{action}' for action in ACTIONS]].to_numpy(dtype=bool)
    target = np.array([ACTIONS.index(action) for action in decisions.action_label], dtype=np.int8)
    if not legal[np.arange(len(target)), target].all():
        raise ValueError('Observed target violates the exact legal mask')
    progress('prepare_features', rows=len(decisions), games=decisions.game_id.nunique(), run_id=run_identity)
    behavior = old.behavior_feature_matrix(decisions)
    common = decisions[['row_id']].copy()
    for index, name in enumerate(old.BASE_FEATURE_COLUMNS):
        common[name] = behavior[:, index]
    line = common.merge(old.build_line_prefix_features(decisions), on='row_id', validate='one_to_one')
    line_names, line_matrix = old.expanded_model_matrix(line, 'within_hand_line_surprise')
    additional = old.build_additional_exploitation_feature_frames(decision_context(decisions), history)
    tasks = []
    import joblib
    for layer in old.EXPECTED_LAYERS:
        package = joblib.load(io.BytesIO(model_bytes[layer]))
        names = list(package['feature_names'])
        if layer == 'postflop_behavior':
            matrix, observed_names = behavior, old.behavior_feature_names()
            support = decisions.street.isin(('flop','turn','river')).to_numpy()
        elif layer == 'within_hand_line_surprise':
            matrix, observed_names = line_matrix, line_names
            support = line.line_prefix_decision_count.to_numpy() >= 1
        else:
            frame = additional[layer].set_index('row_id').loc[decisions.row_id]
            matrix, observed_names = frame[names].to_numpy(dtype=np.float32), names
            support = np.logical_or.reduce([frame[column].astype(bool).to_numpy() for column in old.SUPPORT_FEATURES[layer]])
        # The frozen feature schemas use NaN for missing legal bounds/board
        # fields. Preserve that representation; do not invent zero-valued data.
        if names != list(observed_names) or np.isinf(matrix).any():
            raise ValueError(f'Feature parity failed: {layer}')
        matrix_path = output/f'{layer}.features.npz'
        if not matrix_path.exists():
            with matrix_path.with_suffix('.partial').open('wb') as handle:
                np.savez_compressed(handle, matrix=matrix, legal=legal, support=support)
            matrix_path.with_suffix('.partial').replace(matrix_path)
        else:
            with np.load(matrix_path, allow_pickle=False) as cached:
                if not (np.array_equal(cached['matrix'], matrix, equal_nan=True) and np.array_equal(cached['legal'], legal) and np.array_equal(cached['support'], support)):
                    raise ValueError('Resumed feature cache differs')
        tasks.append({'layer': layer, 'model_path': str(old.ARTIFACT_ROOT/layer/old.MODEL_MEMBER_BY_LAYER[layer]),
            'model_sha256': hashlib.sha256(model_bytes[layer]).hexdigest(), 'feature_names': names,
            'matrix_path': str(matrix_path), 'matrix_sha256': sha(matrix_path), 'output_path': str(output/f'{layer}.predictions.npz')})
    del model_bytes, additional, history
    pending = []
    for task in tasks:
        receipt = output/f"{task['layer']}.receipt.json"
        if receipt.exists():
            value = json.loads(receipt.read_bytes())
            if value.get('task') != task or value.get('prediction_sha256') != sha(Path(task['output_path'])):
                raise ValueError('Prediction checkpoint identity mismatch')
        else:
            pending.append(task)
    if workers < 1:
        raise ValueError('Positive explicit worker count required')
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(pending))), max_tasks_per_child=1) as pool:
        futures = {pool.submit(score_layer, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            value = future.result()
            write_json(output/f"{task['layer']}.receipt.json", value | {'task': task})
            elapsed = time.monotonic()-started
            progress('score_layer', layer=task['layer'], completed=index, total=len(pending),
                     elapsed_seconds=elapsed, eta_seconds=elapsed/index*(len(pending)-index))
    games = decisions.game_id.astype(str).to_numpy()
    slices = {cohort: cohorts.eq(cohort).to_numpy() for cohort in ('historical_development','training_overlap','confirmation')}
    scenarios={f'street:{street}': decisions.street.eq(street).to_numpy() for street in ('preflop','flop','turn','river')}
    scenarios.update({'active:multiway': decisions.active_opponent_count_before.ge(2).to_numpy(),
        'depth:150_200': decisions.effective_stack_before_bb.ge(150).to_numpy() & decisions.effective_stack_before_bb.lt(200).to_numpy(),
        'depth:200_250': decisions.effective_stack_before_bb.ge(200).to_numpy() & decisions.effective_stack_before_bb.le(250).to_numpy(),
        'depth:over250': decisions.effective_stack_before_bb.gt(250).to_numpy(),
        'opportunity:squeeze': decisions.preflop_squeeze_opportunity.astype(bool).to_numpy()})
    for name,mask in scenarios.items():
        slices['exploratory_pooled:'+name]=mask
        for cohort in ('historical_development','training_overlap','confirmation'):
            slices[cohort+':'+name]=mask & cohorts.eq(cohort).to_numpy()
    with np.load(output/'postflop_behavior.predictions.npz', allow_pickle=False) as data:
        postflop_prediction, postflop_support = data['probabilities'], data['support']
    results = {}
    for task in tasks:
        layer = task['layer']
        with np.load(task['output_path'], allow_pickle=False) as data:
            probabilities, prior, support = data['probabilities'], data['prior'], data['support']
        if (probabilities[~legal] != 0).any():
            raise ValueError('Model assigned probability to an illegal action')
        metrics = {}
        for name, scope in slices.items():
            mask = scope & support
            metrics[name] = {'frozen_prior': paired_metrics(target[mask], probabilities[mask], prior[mask], games[mask], alpha=protocol['family_alpha']/len(tasks))}
            paired = mask & postflop_support
            metrics[name]['selected_postflop_predictor'] = paired_metrics(target[paired], probabilities[paired], postflop_prediction[paired], games[paired], alpha=protocol['family_alpha']/len(tasks))
        results[layer] = {'model_sha256': task['model_sha256'], 'coverage': float(support.mean()), 'metrics': metrics,
            'qualification': layer_disposition(layer, metrics, protocol)}
        progress('metrics', layer=layer, disposition=results[layer]['qualification']['disposition'])
    report = {'schema_version': VERSION, 'run_id': run_identity, 'protocol_sha256': sha(protocol_path),
        'confirmation_design':protocol.get('confirmation_design','prospective_after_candidate_freeze'),
        'rows': len(decisions), 'game_clusters': int(decisions.game_id.nunique()), 'cohort_rows': cohorts.value_counts().to_dict(),
        'minimum_decision_utc': str(decisions.decision_utc.min()), 'maximum_decision_utc': str(decisions.decision_utc.max()),
        'authentication': authentication, 'layers': results, 'fit_performed': False,
        'claims': {'new_solver_science': False, 'counterfactual_EV_identified': False, 'automatic_pokernow_actions': False},
        'limitations': ['Human observed actions are prediction targets, not optimal Hero actions.',
            'Previously inspected data is historical development, not new confirmation.',
            'No physical cards, exact position or hand-total side-pot ledger in this prepared feature schema.',
            'The postflop comparator is a predictor, not the Stage21 recommendation policy.']}
    write_json(completed, report)
    write_json(output/'report_receipt.json', {'sha256': sha(completed), 'run_id': run_identity})
    return report


def run(snapshot_root: Path, output: Path, *, workers: int = 1) -> dict:
    done=threading.Event()
    started=time.monotonic()
    def monitor():
        while not done.wait(20):
            progress('execution_heartbeat',elapsed_seconds=round(time.monotonic()-started,1),
                     note='See last phase and completed-layer counters; no speculative ETA during feature preparation.')
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        return _run(snapshot_root,output,workers=workers)
    finally:
        done.set()
        thread.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run(args.snapshot_root, args.output, workers=args.workers)
