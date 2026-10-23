"""Colab causal preparation of exact completed exports for frozen-model replay.

The historical evaluator remains immutable. This successor rebuilds its history
with monetary all-in roles and explicit current identity remapping. No estimator
is fitted or scored here. Raw membership and prepared artifacts are independently
bound to receipts; restarting reuses a completed preparation only after hashing.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import time
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import stage6_shadow_layer_fresh_backtest as old
from stage6_empirical_action_semantics import aggression_role
from stage6_empirical_evaluation_v2 import canonical, sha, write_json, progress

WORKSPACE=Path(__file__).resolve().parents[3]


def verify_files(identities: dict, root: Path) -> None:
    for relative,expected in identities.items():
        path=root/relative
        if path.is_symlink() or path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError(f'Preparation input identity changed: {relative}')


def raw_role(row: dict) -> bool | None:
    bb=float(row['big_blind_raw'])
    return aggression_role({'action_label':old.normalized_action(row),
        'actor_committed_before_bb':float(row['street_contribution_before_raw'])/bb,
        'actor_committed_after_bb':float(row['street_contribution_after_raw'])/bb,
        'amount_to_call_bb':float(row['amount_to_call_raw'])/bb})


def history_records(rows, completion_by_hand, labels, canonical_by_source, seen, counts):
    """Source iterator; preserve all-in targets while correcting pressure roles."""
    current_hand=None;last_aggressor={};last_size={};last_event=-1
    for row in rows:
        key=str(row['hand_action_id'])
        actor=canonical_by_source.get(str(row['actor_pokernow_player_id']),str(row['canonical_group_player_id']))
        opponents=[canonical_by_source.get(str(source)) for source in json.loads(str(row['active_opponent_pokernow_player_ids_before']))]
        if any(value is None for value in opponents):raise ValueError('History opponent lacks an exact identity crosswalk')
        normalized=dict(row,canonical_group_player_id=actor,
                        active_opponent_canonical_player_ids_before=json.dumps(opponents))
        digest=hashlib.sha256(json.dumps(normalized,sort_keys=True,default=str).encode()).hexdigest()
        duplicate=key in seen
        if duplicate:
            if seen[key]!=digest:raise ValueError(f'Conflicting source decision: {key}')
            counts['identical_duplicate_actions']+=1
        seen[key]=digest
        game_hand=(str(row['game_id']),str(row['hand_id']))
        if game_hand!=current_hand:
            current_hand=game_hand;last_aggressor={};last_size={};last_event=-1
        event=int(row['event_index'])
        if duplicate and event==last_event:continue
        if event<=last_event:raise ValueError('History source is not ordered within the hand')
        last_event=event
        street=str(row['street']);action=old.normalized_action(row);facing=old.strict_bool(row['facing_bet_or_raise'])
        facing_size=last_size.get(street) if facing and last_aggressor.get(street) in opponents else None
        role=raw_role(row)
        action_size=None
        if action in {'bet','raise','all_in'}:
            pot=float(row['pot_before_bb'])
            contribution=(float(row['street_contribution_after_raw'])-float(row['street_contribution_before_raw']))/float(row['big_blind_raw'])
            if pot<=0 or contribution<=0:raise ValueError('Wager target has no exact positive contribution/pot')
            action_size=contribution/pot
        if role is True:last_aggressor[street]=actor;last_size[street]=action_size
        elif role is None:last_aggressor.pop(street,None);last_size.pop(street,None)
        # Replayed duplicate prefixes still reconstruct pressure for newly
        # exported later events in the same hand, but never add evidence twice.
        if duplicate:continue
        if action=='all_in' and role is False:counts['all_in_call_roles']+=1
        reason=old.classify_source_quarantine_reason(row)
        if reason is not None:
            counts['quarantine:'+reason]+=1
            continue
        completion=completion_by_hand.get(game_hand)
        if completion is None:raise ValueError(f'Missing hand completion: {game_hand}')
        if pd.Timestamp(row['decision_utc'])>=pd.Timestamp(completion):raise ValueError('Decision is not before hand completion')
        label=labels.get((*game_hand,actor))
        yield {'game_id':game_hand[0],'hand_id':game_hand[1],'decision_id':key,
            'decision_utc':str(row['decision_utc']),'hand_completed_utc':completion,'actor_id':actor,
            'street':street,'action_label':action,'aggression_role':role,'facing_bet_or_raise':facing,
            'active_opponent_count_before':int(row['active_opponent_count_before']),
            'action_size_pot_fraction':action_size,'facing_aggression_size_pot_fraction':facing_size,
            'showdown_revealed':label is not None,
            'showdown_hand_category_ordinal':None if label is None else float(label['hand_strength']['category_rank'])/8,
            'bluff_shown_proxy':None if label is None else label.get('bluff_shown_proxy'),
            'showdown_missed_draw':None if label is None else label.get('missed_draw'),
            'showdown_bottom_pair':None if label is None else label.get('bottom_pair')}


def build_history(completion_by_hand,labels,canonical_by_source):
    # These extra raw fields recover exact source identities, not name joins.
    old.ACTION_HISTORY_COLUMNS=tuple(dict.fromkeys((*old.ACTION_HISTORY_COLUMNS,'active_opponent_pokernow_player_ids_before')))
    output=[];seen={};counts=Counter()
    for action_path,identity_path in ((old.HISTORICAL_ACTIONS,old.HISTORICAL_IDENTITIES),(old.FRESH_ACTIONS,old.FRESH_IDENTITIES)):
        def rows():
            for chunk in old.aligned_source_chunks(action_path,identity_path):
                yield from chunk.to_dict('records')
                progress('history_source_chunk',source=action_path.name,unique_decisions=len(seen),accepted=len(output))
        output.extend(history_records(rows(),completion_by_hand,labels,canonical_by_source,seen,counts))
    return pd.DataFrame(output),dict(counts)


def deduplicate_observations(rows):
    by_key={};duplicates=0
    for row in rows:
        key=(str(row['game_id']),str(row['hand_id']),str(row['pokernow_player_id']))
        value={name:item for name,item in row.items() if name!='source_row_ordinal'}
        if key in by_key:
            if canonical(by_key[key])!=canonical(value):raise ValueError('Conflicting completed observation')
            duplicates+=1
        else:by_key[key]=value
    return [dict(row,source_row_ordinal=index) for index,row in enumerate(by_key.values())],duplicates


def load_decision_slice():
    """Retain the full corpus as history while materializing only replay dates."""
    selected=[];processed=0
    action_reader=pd.read_csv(old.FRESH_ACTIONS,dtype=str,keep_default_na=False,chunksize=old.CHUNK_ROWS)
    identity_reader=pd.read_csv(old.FRESH_IDENTITIES,dtype=str,keep_default_na=False,chunksize=old.CHUNK_ROWS)
    for actions,identities in zip(action_reader,identity_reader,strict=True):
        keys=['hand_action_id','hand_id','game_id','event_index']
        if not actions[keys].equals(identities[keys]):raise ValueError('Full action/identity source order differs')
        if not identities.identity_grouping_status.eq('resolved_retrospective_group').all():raise ValueError('Unresolved full-source identity')
        mask=pd.to_datetime(identities.decision_utc,utc=True)>=old.DEVELOPMENT_CUTOFF
        merged=actions.loc[mask].copy()
        for column in identities.columns:
            if column not in merged:merged[column]=identities.loc[mask,column].to_numpy()
        selected.append(merged);processed+=len(actions)
        progress('select_evaluation_dates',source_rows=processed,selected_rows=sum(len(frame) for frame in selected))
    fresh=pd.concat(selected,ignore_index=True)
    if fresh.hand_action_id.duplicated().any() or fresh[['game_id','hand_id','event_index']].duplicated().any():
        raise ValueError('Duplicate fresh source identity')
    hero=sorted(set(fresh.loc[fresh.actor_pokernow_player_id.eq(old.HERO_POKERNOW_ID),'canonical_group_player_id']))
    if len(hero)!=1:raise ValueError('Hero canonical identity is not unique')
    return fresh,fresh.to_dict('records'),hero[0]


def prepare(plan_path: Path,output: Path) -> dict:
    started=time.monotonic();plan=json.loads(plan_path.read_bytes())
    if not plan['evaluation_protocol'].get('synthetic_canary') and not Path('/content').is_dir():
        raise ValueError('Full causal preparation belongs in Colab; local execution requires an explicit bounded canary plan')
    verify_files(plan['source_files'],WORKSPACE);verify_files(plan['code_files'],WORKSPACE)
    preparation_id=hashlib.sha256(canonical(plan)).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    receipt_path=output/'preparation_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_bytes())
        if receipt['preparation_id']!=preparation_id:raise ValueError('Different resumed preparation')
        verify_files(receipt['files'],output)
        return receipt
    for name,relative in plan['source_roles'].items():setattr(old,name,WORKSPACE/relative)
    fresh,records,hero=load_decision_slice() if plan.get('full_completed_source') else old.load_fresh_source()
    observations,completions,crosswalk=old.load_profile_history(set(fresh.canonical_group_player_id.astype(str)))
    observations,duplicates=deduplicate_observations(observations)
    decisions,quarantine=old.build_fresh_decisions(records,observations)
    labels=old.load_exact_labels(crosswalk)
    del records,observations,fresh;gc.collect()
    history,history_counts=build_history(completions,labels,crosswalk)
    for name,frame in [('decisions.parquet',decisions),('history.parquet',history),('quarantine.parquet',quarantine)]:
        target=output/name;temporary=target.with_suffix('.partial')
        frame.to_parquet(temporary,index=False,compression='zstd')
        if target.exists():
            if sha(target)!=sha(temporary):raise ValueError('Partial preparation output differs')
            temporary.unlink()
        else:temporary.replace(target)
    cohort_plan=plan['evaluation_protocol']
    protocol={**cohort_plan,'files':{name:{'bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name in ('decisions.parquet','history.parquet')},
        'code_files':plan['evaluation_code_files'],'preparation_id':preparation_id}
    write_json(output/'protocol.json',protocol)
    receipt={'schema_version':'stage6_empirical_causal_preparation_v2','preparation_id':preparation_id,
        'status':'COMPLETE_CAUSAL_DATA_PREPARATION_NO_MODEL_SCORING','raw_source_plan_sha256':sha(plan_path),
        'decisions':len(decisions),'history_rows':len(history),'history_games':history.game_id.nunique(),
        'decision_games':decisions.game_id.nunique(),'quarantined_decisions':len(quarantine),'history_counts':history_counts,
        'identical_duplicate_observations':duplicates,'exact_historical_reveal_labels':len(labels),'hero_canonical_id':hero,
        'reveal_missingness':'No reveal label means unobserved, never weak-hand truth; newer reveals absent from the frozen R1 label source remain unobserved.',
        'elapsed_seconds':time.monotonic()-started,
        'files':{name:{'bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name in ('decisions.parquet','history.parquet','quarantine.parquet','protocol.json')}}
    # Validate source bytes once more before sealing the prepared outputs.
    verify_files(plan['source_files'],WORKSPACE)
    write_json(receipt_path,receipt);progress('preparation_complete',**receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    done=threading.Event();started=time.monotonic()
    def monitor():
        while not done.wait(20):progress('preparation_heartbeat',elapsed_seconds=round(time.monotonic()-started,1))
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:prepare(args.plan,args.output)
    finally:done.set();thread.join(timeout=1)
