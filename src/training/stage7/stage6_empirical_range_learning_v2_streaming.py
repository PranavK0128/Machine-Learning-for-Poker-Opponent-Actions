"""Disk-backed full-corpus preparation/fitting for empirical range v2.

Raw decision dictionaries and public prefixes are streamed one row at a time.
Training matrices are memory-mapped; only calibration/evaluation metadata and
the portable sufficient statistics are materialized. No RAM-based worker cap.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import stage6_empirical_range_learning_v2 as m


def blob(values):
    return struct.pack('<' + 'd' * len(values), *values)


def read_blob(value, width):
    return struct.unpack('<' + 'd' * width, value)


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=DELETE')
    connection.execute('PRAGMA synchronous=FULL')
    return connection


def compile_rows(input_path, protocol, directory, binding):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    db_path, receipt_path = directory / 'rows.sqlite', directory / 'prepared_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['binding'] != binding or receipt['database_sha256'] != m.sha(db_path):
            raise ValueError('compact_preparation_identity_mismatch')
        return db_path, receipt
    partial = directory / ('rows.sqlite.partial.' + uuid.uuid4().hex)
    con = connect(partial)
    con.executescript('''
      CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, row_id TEXT UNIQUE, digest TEXT, partition TEXT,
        game_id TEXT, hand_id TEXT, actor_id TEXT, hand_actor_id TEXT, action TEXT, size_label TEXT, revealed INTEGER,
        public_features BLOB, combo_features BLOB, compact_json TEXT);
      CREATE INDEX roles ON rows(partition,revealed,ordinal);
      CREATE INDEX hands ON rows(game_id,hand_id,hand_actor_id);
      CREATE TABLE observations (game_id TEXT,hand_id TEXT,actor_id TEXT,hand_actor_id TEXT,context TEXT,session_id TEXT,
        class TEXT,hole TEXT,row_id TEXT,completion TEXT,PRIMARY KEY(game_id,hand_id,hand_actor_id,context));
      CREATE TABLE hand_labels (game_id TEXT,hand_id TEXT,hand_actor_id TEXT,hole TEXT,PRIMARY KEY(game_id,hand_id,hand_actor_id));
    ''')
    partition = {}
    for name in ['train', 'calibration', 'evaluation', 'diagnostic']:
        for game in protocol.get(name + '_game_ids', []):
            if game in partition:
                raise ValueError('whole_game_cohort_overlap')
            partition[game] = name
    if set(protocol['evaluation_game_ids']) & set(protocol.get('previously_inspected_game_ids', [])):
        raise ValueError('evaluation_contains_previously_inspected_game')
    game_bounds, public_support, count = {}, {'categories': {}, 'numeric_ranges': {}}, 0
    private_support={'categories': {}, 'numeric_ranges': {}}
    source_support = Counter(); role_counts = Counter(); training_actors = set()
    last_progress = started = time.monotonic()
    try:
        with Path(input_path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                game = row.get('game_id')
                if game not in partition:
                    raise ValueError('unassigned_game')
                # Validate the exact full row before discarding provenance from
                # memory. Whole-game global membership/chronology is checked below.
                one = dict(protocol, train_game_ids=[game], calibration_game_ids=[], evaluation_game_ids=[], diagnostic_game_ids=[])
                m.validate_rows([row], one)
                digest = hashlib.sha256(m.canonical(row).encode()).hexdigest()
                existing = con.execute('SELECT digest FROM rows WHERE row_id=?', (row['row_id'],)).fetchone()
                if existing:
                    if existing[0] != digest:
                        raise ValueError('conflicting_duplicate_row')
                    continue
                bounds = (row['game_started_utc'], row['game_completed_utc'])
                if game_bounds.setdefault(game, bounds) != bounds:
                    raise ValueError('inconsistent_game_bounds')
                hole = m.known_label(row); role = partition[game]; hand_actor = m.hand_actor_identity(row)
                public_values = m.public_features(row['public'])
                combo_values = m.combo_features(row['public'], hole) if hole is not None else None
                compact = {k: row[k] for k in ['row_id','game_id','hand_id','actor_id','session_id','game_started_utc','game_completed_utc',
                    'decision_utc','action_observed_utc','public_available_utc','hand_completed_utc','public','action','legal_actions','card_visibility']}
                compact['observed_action_size_pot_fraction'] = row.get('observed_action_size_pot_fraction')
                compact['hand_started_utc'] = row.get('hand_started_utc', row['game_started_utc'])
                if 'actor_pokernow_player_id' in row:
                    compact['actor_pokernow_player_id'] = row['actor_pokernow_player_id']
                con.execute('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (count,row['row_id'],digest,role,game,row['hand_id'],row['actor_id'],hand_actor,row['action'],m.size_label(row),int(hole is not None),
                     blob(public_values),blob(combo_values) if combo_values is not None else None,m.canonical(compact)))
                count += 1; role_counts[role] += 1; source_support['revealed' if hole is not None else 'latent'] += 1
                if hole is not None:
                    physical = m.canonical(sorted(hole))
                    previous = con.execute('SELECT hole FROM hand_labels WHERE game_id=? AND hand_id=? AND hand_actor_id=?', (game,row['hand_id'],hand_actor)).fetchone()
                    if previous and previous[0] != physical:
                        raise ValueError('conflicting_private_hand_labels')
                    con.execute('INSERT OR IGNORE INTO hand_labels VALUES(?,?,?,?)', (game,row['hand_id'],hand_actor,physical))
                    if role == 'train':
                        con.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                            (game,row['hand_id'],row['actor_id'],hand_actor,m.prior_key(row['public']),row['session_id'],m.hand_class(hole),physical,row['row_id'],row['game_completed_utc']))
                if role == 'train':
                    training_actors.add(row['actor_id'])
                    for support in ([public_support,private_support] if hole is not None else [public_support]):
                        for key in ['street','position','opportunity']:
                            support['categories'].setdefault(key,set()).add(row['public'][key])
                        for key in m.NUMERIC:
                            value = row['public'][key]
                            limits = support['numeric_ranges'].setdefault(key,[value,value])
                            limits[0] = min(limits[0],value); limits[1] = max(limits[1],value)
                if count % 10000 == 0:
                    con.commit()
                if time.monotonic() - last_progress >= 20:
                    print(m.canonical(dict(phase='compact_range_preparation',rows=count,rows_per_second=count/(time.monotonic()-started))),flush=True)
                    last_progress = time.monotonic()
        if set(game_bounds) != set(partition):
            raise ValueError('declared_game_missing_from_input')
        for left,right in [('train','calibration'),('calibration','evaluation'),('train','evaluation')]:
            a,b=protocol[left+'_game_ids'],protocol[right+'_game_ids']
            if a and b and max(m.utc(game_bounds[g][1]) for g in a) >= min(m.utc(game_bounds[g][0]) for g in b):
                raise ValueError('cohorts_not_chronological_whole_games')
        con.commit(); con.close()
        if db_path.exists():
            if m.sha(db_path) != m.sha(partial):
                raise ValueError('different_existing_compact_database')
            partial.unlink()
        else:
            os.rename(partial,db_path)
    except BaseException:
        con.close()
        raise
    public_support['categories'] = {key:sorted(values) for key,values in public_support['categories'].items()}
    private_support['categories'] = {key:sorted(values) for key,values in private_support['categories'].items()}
    receipt = dict(schema_version=m.VERSION+'_compact',binding=binding,database_sha256=m.sha(db_path),rows=count,
        role_counts=dict(role_counts),source_support=dict(source_support),game_bounds=game_bounds,
        public_support=public_support,private_support=private_support,training_actors=sorted(training_actors),authority=m.AUTHORITY)
    m.write_once(receipt_path,receipt)
    return db_path,receipt


def prior_counts(con, eligible_games=None):
    result = {}
    if eligible_games is not None and not eligible_games:return result
    query='SELECT context,actor_id,session_id,class,row_id,completion FROM observations'
    if eligible_games is not None:query+=' WHERE game_id IN ('+','.join('?' for _ in eligible_games)+')'
    query+=' ORDER BY context,actor_id,session_id,row_id'
    for context,actor,session,cls,event,completion in con.execute(query,eligible_games or []):
        entry=result.setdefault(context,{}).setdefault(actor,{}).setdefault(session,dict(counts={},events=[],completed_through_utc=completion))
        entry['counts'][cls]=entry['counts'].get(cls,0)+1; entry['events'].append(event)
        entry['completed_through_utc']=max(entry['completed_through_utc'],completion)
    return result


def matrices(con, directory, role, partition, labels):
    combo=role=='physical_combo_action_likelihood'
    column='combo_features' if combo else 'public_features'
    target='action' if combo else 'size_label'
    extra=' AND revealed=1' if combo else ''
    count=con.execute('SELECT COUNT(*) FROM rows WHERE partition=?'+extra,(partition,)).fetchone()[0]
    width=len(m.PUBLIC_NAMES)+(len(m.HOLE_NAMES) if combo else 0)
    stem=Path(directory)/(role+'_'+partition)
    x=np.lib.format.open_memmap(str(stem)+'.x.npy',mode='w+',dtype=np.float64,shape=(count,width))
    y=np.lib.format.open_memmap(str(stem)+'.y.npy',mode='w+',dtype=np.int16,shape=(count,))
    indices=np.lib.format.open_memmap(str(stem)+'.ordinal.npy',mode='w+',dtype=np.int64,shape=(count,))
    valid=0; unsupported=0
    query='SELECT ordinal,'+column+','+target+' FROM rows WHERE partition=?'+extra+' ORDER BY ordinal'
    for ordinal,features,value in con.execute(query,(partition,)):
        if value not in labels:
            unsupported+=1;continue
        x[valid]=read_blob(features,width);y[valid]=labels.index(value);indices[valid]=ordinal;valid+=1
    x.flush();y.flush();indices.flush()
    return x[:valid],y[:valid],indices[:valid],unsupported


def metadata(con, ordinals):
    # Only held-out/calibration metadata is materialized, not the training corpus.
    return [json.loads(con.execute('SELECT compact_json FROM rows WHERE ordinal=?',(int(i),)).fetchone()[0]) for i in ordinals]


def fit_streaming(input_path, protocol, directory, binding, role_filter=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    db_path,prepared=compile_rows(input_path,protocol,directory/'prepared',binding)
    con=connect(db_path)
    config=protocol['model_config']
    controls=m.diagnostic_controls(protocol)
    if set(config)!={'iterations','num_leaves','learning_rate','min_child_samples','seed'} or not(1<=config['iterations']<=1000 and 2<=config['num_leaves']<=127 and 0<config['learning_rate']<=1 and config['min_child_samples']>=1):
        raise ValueError('bounded_frozen_model_configuration_required')
    train_games=protocol['train_game_ids']; calibration_games=protocol['calibration_game_ids']
    if not train_games:raise ValueError('training_games_required')
    artifact=dict(schema_version=m.VERSION,selection_assumption=protocol['selection_assumption'],authority=m.AUTHORITY,
        trained_through_utc=max(prepared['game_bounds'][g][1] for g in train_games),
        available_after_utc=max(prepared['game_bounds'][g][1] for g in train_games+calibration_games),
        trained_game_ids=train_games,calibration_game_ids=calibration_games,model_family='LightGBM_4.6.0_no_family_search',
        public_features=m.PUBLIC_NAMES,models={},prior_counts=prior_counts(con),public_support=prepared['public_support'])
    stale_games=sorted(game for game in train_games if m.utc(prepared['game_bounds'][game][1])<m.utc(controls['stale_prior_cutoff_utc']))
    artifact['diagnostic_prior_history']=m.diagnostic_prior_history(artifact['prior_counts'],prior_counts(con,stale_games),stale_games,controls)
    report=dict(schema_version=m.VERSION,source_support=dict(rows=prepared['rows'],games=len(prepared['game_bounds']),
        revealed_label_rows=prepared['source_support'].get('revealed',0),latent_rows=prepared['source_support'].get('latent',0),
        disposition='REVEAL_SELECTED_EXPERIMENT_ONLY',authority=m.AUTHORITY),roles={},authority=m.AUTHORITY)
    for role in ['physical_combo_action_likelihood','observed_next_action_size']:
        if role_filter is not None and role not in role_filter:continue
        checkpoint_path=directory/'checkpoints'/(role+'.json')
        if checkpoint_path.exists():
            checkpoint=json.loads(checkpoint_path.read_text())
            if checkpoint['binding']!=binding or checkpoint['payload_sha256']!=hashlib.sha256(m.canonical(checkpoint['payload']).encode()).hexdigest():
                raise ValueError('range_role_checkpoint_identity_mismatch')
            stored=checkpoint['payload'];report['roles'][role]=stored['report']
            if stored['model']:artifact['models'][role]=stored['model']
            continue
        combo=role=='physical_combo_action_likelihood'
        labels=m.ACTIONS if combo else [r[0] for r in con.execute("SELECT DISTINCT size_label FROM rows WHERE partition='train' ORDER BY size_label")]
        matrix_dir=directory/'matrices'/role;matrix_dir.mkdir(parents=True,exist_ok=True)
        x,y,train_indices,_=matrices(con,matrix_dir,role,'train',labels)
        model=m.fit_booster(x,y,labels,config)
        exported=None
        if model is None:
            role_report=dict(disposition='INSUFFICIENT_OBSERVED_TARGET_SUPPORT',rows=len(y),authority=m.AUTHORITY)
        else:
            exported=m.booster_export(model,labels,m.PUBLIC_NAMES+m.HOLE_NAMES if combo else m.PUBLIC_NAMES)
            exported['public_support']=prepared['private_support'] if combo else prepared['public_support']
            prior=np.bincount(y,minlength=len(labels)).astype(float)+1;prior/=prior.sum()
            exported['training_target_prior']=prior.tolist()
            temperature=1.0
            cx,cy,ci,cu=matrices(con,matrix_dir,role,'calibration',labels)
            calibration=metadata(con,ci)
            if len(cy):
                cp=m.full_prediction(model,cx,labels)
                temperature=min([.5,.75,1.0,1.25,1.5,2.0],key=lambda t:(float(m.loss_rows(m.legal_probabilities(m.temper(cp,t),calibration,labels),cy)[0].mean()),abs(t-1)))
            exported['temperature']=temperature
            exported['temperature_selection']='calibration_whole_games_only; fixed_grid_0.5_0.75_1_1.25_1.5_2'
            role_report=dict(training_rows=len(y),training_games=con.execute("SELECT COUNT(DISTINCT game_id) FROM rows WHERE partition='train'"+(' AND revealed=1' if combo else '')).fetchone()[0],
                target='P(observed actor action | hypothetical physical holding, public context, selected reveal)' if combo else 'P(observed next action and size bucket | public context)',
                counterfactual_response_identified=False,unsupported_target_rows={'calibration':cu},disposition='ASSUMPTION_DEPENDENT_EXPERIMENT_RETAIN_SHADOW')
            for name in ['calibration','evaluation']:
                if name=='calibration':ex,ey,ei,unsupported=cx,cy,ci,cu;rows=calibration
                else:ex,ey,ei,unsupported=matrices(con,matrix_dir,role,name,labels);rows=metadata(con,ei)
                role_report['unsupported_target_rows'][name]=unsupported
                if not len(ey):role_report[name]=dict(rows=0,games=0,disposition='NO_SUPPORT');continue
                probs=m.legal_probabilities(m.temper(m.full_prediction(model,ex,labels),temperature),rows,labels)
                base=m.legal_probabilities(np.tile(prior,(len(rows),1)),rows,labels)
                role_report[name]=m.scores(probs,ey,rows,base,config['seed'])
                role_actors={r[0] for r in con.execute("SELECT DISTINCT actor_id FROM rows WHERE partition='train'"+(' AND revealed=1' if combo else ''))}
                role_report[name]['diagnostic_slices']=m.diagnostic_slices(probs,ey,rows,base,role_actors)
                del probs,base
            artifact['models'][role]=exported
            del model,calibration,cx,cy,ci
        report['roles'][role]=role_report
        stored=dict(model=exported,report=role_report)
        m.write_once(checkpoint_path,dict(binding=binding,payload_sha256=hashlib.sha256(m.canonical(stored).encode()).hexdigest(),payload=stored))
        del x,y,train_indices
        print(m.canonical(dict(phase='compact_role_complete',role=role,training_rows=role_report.get('training_rows',role_report.get('rows',0)))),flush=True)
    artifact['continuation_value']=dict(status='NOT_IDENTIFIED',reason='unchosen_action_outcomes_and_human_propensities_unavailable',supports_unchosen_action_ev=False)
    budget=protocol.get('range_diagnostic_rows_per_game',0) if role_filter is None else 0
    if not isinstance(budget,int) or not 0<=budget<=200:raise ValueError('invalid_preregistered_range_diagnostic_budget')
    selected={}
    if budget:
        # Lowest immutable identity hashes are retained using O(games*budget)
        # memory. Outcomes never determine membership of the bounded sample.
        for row_id,game,hand,actor in con.execute("SELECT row_id,game_id,hand_id,hand_actor_id FROM rows WHERE partition='evaluation' AND revealed=1"):
            values=selected.setdefault(game,[]);values.append((hashlib.sha256(row_id.encode()).hexdigest(),hand,actor));values.sort()
            if len(values)>budget:values.pop()
    range_rows={}
    for game,values in selected.items():
        for _,hand,actor in values:
            for data, in con.execute('SELECT compact_json FROM rows WHERE game_id=? AND hand_id=? AND hand_actor_id=?',(game,hand,actor)):
                row=json.loads(data);range_rows[row['row_id']]=row
    report['range_diagnostics']=m.evaluate_ranges(artifact,list(range_rows.values()),budget)
    if budget and report['range_diagnostics']['status']=='REVEAL_SELECTED_DIAGNOSTIC':
        report['range_diagnostics']['eligible_revealed_rows']=con.execute("SELECT COUNT(*) FROM rows WHERE partition='evaluation' AND revealed=1").fetchone()[0]
    report['compact_preparation']=dict(rows=prepared['rows'],database_sha256=prepared['database_sha256'],training_storage='disk_backed_sqlite_and_float64_memory_maps')
    con.close()
    return artifact,report


def _fit_role(arguments):
    input_path,protocol,directory,binding,role=arguments
    fit_streaming(input_path,protocol,directory,binding,[role])
    return role


def fit_parallel(input_path,protocol,directory,binding):
    """Independent one-thread model roles with deterministic checkpoint reduction."""
    directory=Path(directory)
    compile_rows(input_path,protocol,directory/'prepared',binding)
    roles=[role for role in ['physical_combo_action_likelihood','observed_next_action_size'] if not (directory/'checkpoints'/(role+'.json')).exists()]
    capacity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    requested=protocol.get('workers','auto')
    if requested=='auto':workers=min(2,capacity,len(roles))
    elif isinstance(requested,int) and not isinstance(requested,bool) and 1<=requested<=min(2,capacity):workers=min(requested,len(roles))
    else:raise ValueError('invalid_role_worker_request')
    print(m.canonical(dict(phase='independent_range_roles',workers=workers,usable_logical_cpus=capacity,unfinished_roles=len(roles),inner_threads=1,ram_is_telemetry_only=True)),flush=True)
    tasks=[(str(input_path),protocol,str(directory),binding,role) for role in roles]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            for role in pool.map(_fit_role,tasks):print(m.canonical(dict(phase='role_process_reaped',role=role)),flush=True)
    else:
        for task in tasks:_fit_role(task)
    return fit_streaming(input_path,protocol,directory,binding)
