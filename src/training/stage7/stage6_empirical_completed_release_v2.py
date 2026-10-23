"""Freeze completed raw-source preparation and selected-winner Colab replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd
from stage6_empirical_evaluation_v2 import sha, canonical, write_json
from stage6_empirical_release_v2 import dependencies, identity, STAGE, ROOT, RUNTIME


def create_plan(*,canary=False,revision='r1',snapshot_manifest=None):
    if not revision.isalnum():raise ValueError('Alphanumeric revision required')
    destination=STAGE/f'inputs/empirical_v2_completed_{"canary_" if canary else ""}{revision}'
    if destination.exists():raise ValueError('Plan destination already exists')
    destination.mkdir(parents=True)
    snapshot_manifest=(Path(snapshot_manifest) if snapshot_manifest else STAGE/'manifests/stage6_empirical_v2_completed_snapshot.json').resolve()
    snapshot=json.loads(snapshot_manifest.read_bytes())
    source_root=Path(snapshot['files']['fresh-action-state-decision-rows.csv']['path']).parent
    game_path=source_root/'game-identities.json'
    games=json.loads(game_path.read_bytes())['games']
    names={row['pokernowGameId']:row['id'] for row in games}
    starts={row['id']:datetime.fromtimestamp(int(row['firstHandStartedAt'])/1000,timezone.utc).isoformat() for row in games if row['firstHandStartedAt'] is not None}
    previously_replayed={names[name] for name in ('pgl8cdDzWBGiScq2GsdVpPl6Q','pgl0pqNU8P-7SwPeHButJsg7H','pgl9JZecpbuv1adxf6MILRTDR')}
    exposure_path=STAGE/'manifests/stage6_empirical_v2_evaluation_exposures.json'
    exposures=json.loads(exposure_path.read_bytes())
    known=set(snapshot['previously_inspected_game_ids'])|previously_replayed|set(exposures['bounded_source_canary_game_ids'])
    cutoff='2026-08-21T00:00:00+00:00'
    evaluated={row['game_id'] for row in snapshot['games'] if pd.Timestamp(row['maximum_decision_utc'])>=pd.Timestamp(cutoff)}
    training={game for game in set(snapshot['previously_inspected_game_ids']) if game not in starts or pd.Timestamp(starts[game])<=pd.Timestamp(cutoff)}
    final={game for game in evaluated-known-training if game in starts and pd.Timestamp(starts[game])>pd.Timestamp(cutoff)}
    roles={
        'HISTORICAL_ACTIONS':STAGE/'data/exports/ml/action-state-decision-rows.csv',
        'HISTORICAL_IDENTITIES':STAGE/'data/exports/ml/stage6-action-decision-time-identity.csv',
        'HISTORICAL_OBSERVATIONS':STAGE/'data/exports/ml/stage6-hand-profile-observations.csv',
        'FRESH_ACTIONS':source_root/'fresh-action-state-decision-rows.csv',
        'FRESH_IDENTITIES':source_root/'fresh-decision-time-identity.csv',
        'FRESH_OBSERVATIONS':source_root/'fresh-hand-profile-observations.csv',
        'LABEL_ZIP':STAGE/'inputs/POKERNOW_ASSISTANT_STAGE6_ADDITIONAL_EXPLOITATION_SOURCE_INPUTS_R1.zip',
        'REGISTRY_PATH':STAGE/'contracts/stage6_profile_metric_registry_hold_v1.json'}
    if snapshot.get('full_completed_source'):
        from stage6_empirical_snapshot_v2 import require_full_history_coverage
        require_full_history_coverage(snapshot['older_prepared_hand_coverage'])
        # The completed full source already includes every older accepted hand.
        # Empty legacy partitions avoid duplicate observations or stale versions.
        for role,fresh_role in [('HISTORICAL_ACTIONS','FRESH_ACTIONS'),('HISTORICAL_IDENTITIES','FRESH_IDENTITIES'),('HISTORICAL_OBSERVATIONS','FRESH_OBSERVATIONS')]:
            target=destination/(role.lower()+'.csv')
            with roles[fresh_role].open('r',encoding='utf-8-sig') as source:
                target.write_text(source.readline(),encoding='utf-8')
            roles[role]=target
    if canary:
        actions=pd.read_csv(roles['FRESH_ACTIONS'],dtype=str,keep_default_na=False,nrows=2000)
        actions=actions[actions.game_id.isin(known)]
        selected=set(actions.hand_id.drop_duplicates().head(4))
        actions=actions[actions.hand_id.isin(selected)]
        ids=pd.read_csv(roles['FRESH_IDENTITIES'],dtype=str,keep_default_na=False,nrows=2000)
        ids=ids[ids.hand_id.isin(selected)]
        # This is bounded source extraction, not feature construction on the
        # corpus. Locate only selected completed-hand observations in chunks.
        observations=[]
        for chunk in pd.read_csv(roles['FRESH_OBSERVATIONS'],dtype=str,keep_default_na=False,chunksize=5000):
            observations.append(chunk[chunk.hand_id.isin(selected)])
            if set(pd.concat(observations).hand_id)==selected:break
        observations=pd.concat(observations)
        for name,frame in [('FRESH_ACTIONS',actions),('FRESH_IDENTITIES',ids),('FRESH_OBSERVATIONS',observations),
                           ('HISTORICAL_ACTIONS',actions.iloc[:0]),('HISTORICAL_IDENTITIES',ids.iloc[:0]),('HISTORICAL_OBSERVATIONS',observations.iloc[:0])]:
            path=destination/(name.lower()+'.csv');frame.to_csv(path,index=False);roles[name]=path
        label=destination/'empty_selected_labels.zip'
        with zipfile.ZipFile(label,'x') as out:out.writestr('stage6_additional_revealed_hand_labels.jsonl',b'')
        roles['LABEL_ZIP']=label
        evaluated=set(ids.game_id);known|=evaluated;final=set()
    code=dependencies(STAGE/'src/stage6_empirical_preparation_v2.py')
    eval_code=dependencies(STAGE/'src/stage6_empirical_evaluation_v2.py')
    protocol={'schema_version':'stage6_empirical_selected_model_protocol_v2','synthetic_canary':canary,
        'candidate_frozen_utc':datetime.now(timezone.utc).isoformat(),
        'confirmation_design':'locked_retrospective_unseen_games','model_training_cutoff_utc':cutoff,
        'outcomes_inspected_before_protocol_freeze':False,
        'source_disposition':'completed_validated_raw_exports_causal_reconstruction',
        'source_game_start_utc':{game:starts[game] for game in sorted(evaluated) if game in starts},
        'installed_manifest_sha256':'1f5c2aa62097cc815515d6c99e5128109586d9ccae35ed6488a7679b9668682c',
        'previously_inspected_game_ids':sorted(known),'training_game_ids':sorted(training),
        'development_game_ids':sorted(evaluated-final-training),'confirmation_game_ids':sorted(final),
        'minimum_confirmation_games':20,'minimum_positive_game_fraction':.5,'family_alpha':.05,
        'selection_policy':'reuse_exact_nine_frozen_winners_no_fit_no_temperature_tuning',
        'primary_endpoint':'locked_unseen_game_paired_log_loss_improvement_against_own_frozen_prior',
        'secondary_required_endpoint':'paired_brier_improvement',
        'multiplicity':'Bonferroni_nine_layers_two_required_endpoints_intersection',
        'strategy_promotion_from_prediction_alone':False}
    source_files={path.relative_to(ROOT).as_posix():identity(path) for path in roles.values()}
    if not canary:
        for expected in snapshot['files'].values():
            path=Path(expected['path'])
            if identity(path)!={key:expected[key] for key in ('bytes','sha256')}:raise ValueError('Completed raw source changed')
    plan={'schema_version':'stage6_empirical_causal_plan_v2','source_roles':{name:path.relative_to(ROOT).as_posix() for name,path in roles.items()},
        'source_files':source_files,'code_files':{path.relative_to(ROOT).as_posix():identity(path) for path in code},
        'evaluation_code_files':{path.relative_to(ROOT).as_posix():identity(path) for path in eval_code},
        'evaluation_protocol':protocol,'completed_snapshot_sha256':sha(snapshot_manifest),
        'completed_snapshot_path':snapshot_manifest.relative_to(ROOT).as_posix(),
        'full_completed_source':snapshot.get('full_completed_source',False),
        'game_identity_metadata':{'path':game_path.relative_to(ROOT).as_posix(),**identity(game_path)},
        'outcome_exposure_metadata':{'path':exposure_path.relative_to(ROOT).as_posix(),**identity(exposure_path)},
        'worker_policy':{'maximum':2,'inner_threads':1,'capacity':'CPU_affinity_and_cgroup_quota','ram_worker_cap':False}}
    if not canary and snapshot.get('full_completed_source'):
        hands=source_root/'complete-hand-sidecar.jsonl'
        receipt_path=hands.with_name(hands.name+'.receipt.json')
        receipt=json.loads(receipt_path.read_bytes())
        if receipt['status']!='COMPLETE_READ_ONLY_EXACT_HAND_SOURCE_EXPORT' or receipt['producer_receipt_sha256']!=snapshot['producer_receipt']['sha256']:
            raise ValueError('Full sidecar belongs to a different or incomplete producer snapshot')
        if identity(hands)!={key:receipt[key] for key in ('bytes','sha256')}:
            raise ValueError('Full hand sidecar changed')
        plan['shared_range_source']={role:{'path':path.relative_to(ROOT).as_posix(),**identity(path)}
            for role,path in [('hands',hands),('receipt',receipt_path),('membership',source_root/'sidecar-game-membership.json')]}
    write_json(destination/'plan.json',plan)
    return destination,plan


def package(destination,plan,revision):
    archive=STAGE/f'inputs/stage6_empirical_v2_completed_{revision}.zip'
    files={ROOT/name for name in set(plan['source_files'])|set(plan['code_files'])}
    files.add(destination/'plan.json')
    files.add(ROOT/plan['game_identity_metadata']['path'])
    files.add(ROOT/plan['outcome_exposure_metadata']['path'])
    files.add(ROOT/plan['completed_snapshot_path'])
    files.update(ROOT/item['path'] for item in plan.get('shared_range_source',{}).values())
    install=RUNTIME/'ml/artifacts/stage6_retrospective_nested_v2'
    files.add(install/'install_manifest.json')
    for layer in json.loads((install/'install_manifest.json').read_bytes())['layers']:
        files.update(install/layer['layer']/name for name in layer['files'])
    members={path.relative_to(ROOT).as_posix():identity(path) for path in sorted(files)}
    manifest={'schema_version':'stage6_empirical_completed_package_v2','members':members,
        'plan_relative_path':(destination/'plan.json').relative_to(ROOT).as_posix(),'plan_sha256':sha(destination/'plan.json')}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as bundle:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix();digest=hashlib.sha256();count=0
            with path.open('rb') as source,bundle.open(name,'w',force_zip64=True) as target:
                while raw:=source.read(8*1024*1024):target.write(raw);digest.update(raw);count+=len(raw)
            if {'bytes':count,'sha256':digest.hexdigest()}!=members[name]:raise ValueError('Source changed while packaging')
            print(json.dumps({'phase':'package','member':name,'bytes':count}),flush=True)
        bundle.writestr('package_manifest.json',canonical(manifest))
    return archive,manifest


def notebook(archive,manifest,revision):
    path=STAGE/f'notebooks/88_Stage6_Completed_Data_Selected_Models_V2_{revision}.ipynb'
    if path.exists():raise ValueError('Notebook already exists')
    cells=[]
    def md(value):cells.append(nbformat.v4.new_markdown_cell(value))
    def code(value):cells.append(nbformat.v4.new_code_cell(value))
    md('# 88 — Completed data and the nine selected models\n\nBuild strictly causal features from the completed producer exports and preserved historical games, then replay the exact installed winners. No model search or fitting. Whole-game cohorts are frozen before scoring; previously reviewed games remain development. The locked retrospective unseen cohort is distinct from future prospective confirmation. Prediction results alone do not establish Hero strategy EV.\n\nRun all cells in order. Use CPU High-RAM, Python 3.12. Preserve failed/checkpoint outputs; resume this exact package. Returned ZIPs require external byte readback and strict import.')
    code("""import os,sys,json,hashlib,pathlib,subprocess,zipfile,time,math,shutil,uuid
assert sys.version_info[:2]==(3,12),sys.version
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
host_gb=int(mem['MemTotal'].split()[0])*1024/1e9
assert host_gb>=48,f'CPU High-RAM required; observed {host_gb:.2f} GB'
capacity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
quota=pathlib.Path('/sys/fs/cgroup/cpu.max')
if quota.exists():
    q,period=quota.read_text().split()
    if q!='max':capacity=min(capacity,max(1,math.floor(int(q)/int(period))))
REQUESTED_WORKERS='Auto' # Positive custom count 1 or 2 is supported.
workers=min(2,capacity) if REQUESTED_WORKERS=='Auto' else min(int(REQUESTED_WORKERS),2,capacity)
assert workers>=1
from google.colab import drive
if not pathlib.Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
DRIVE_ROOT=pathlib.Path('/content/drive/MyDrive/Poker Assistant Model/Stage 6 - Qualified Layers Improvement')
print({'python':sys.version,'host_ram_gb':host_gb,'effective_cpus':capacity,'workers':workers,'inner_threads':1})
""")
    md('## Frozen dependencies\nInference and feature work run in fresh subprocesses after installation.')
    code("""requirements=['numpy==2.2.6','pandas==2.3.3','pyarrow==21.0.0','joblib==1.5.2','scikit-learn==1.7.2','lightgbm==4.6.0','catboost==1.2.10','xgboost==3.3.0','threadpoolctl==3.6.0']
subprocess.run([sys.executable,'-m','pip','install','--quiet',*requirements],check=True)
""")
    md('## Authenticate direct inputs\nZIP membership, safe paths, byte counts and SHA-256 are checked before importing packaged code or deserializing models. Copy once to local scratch; durable checkpoints stay in this notebook’s own directory.')
    code(f"""EXPECTED_PACKAGE_SHA256={sha(archive)!r}
SOURCE=DRIVE_ROOT/'inputs/{archive.name}'
def file_sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
assert file_sha(SOURCE)==EXPECTED_PACKAGE_SHA256,'Package mismatch'
WORKSPACE=pathlib.Path('/content/stage6_empirical_v2_'+EXPECTED_PACKAGE_SHA256[:12]);WORKSPACE.mkdir(exist_ok=True)
LOCAL=WORKSPACE/'input.zip'
if not LOCAL.exists():
    partial=LOCAL.with_name(LOCAL.name+'.partial-'+uuid.uuid4().hex)
    shutil.copyfile(SOURCE,partial)
    assert partial.stat().st_size==SOURCE.stat().st_size and file_sha(partial)==EXPECTED_PACKAGE_SHA256
    partial.replace(LOCAL)
assert file_sha(LOCAL)==EXPECTED_PACKAGE_SHA256
with zipfile.ZipFile(LOCAL) as bundle:
    names=bundle.namelist();assert len(names)==len(set(names))
    manifest=json.loads(bundle.read('package_manifest.json'))
    assert set(names)==set(manifest['members'])|{{'package_manifest.json'}}
    for name,identity in manifest['members'].items():
        relative=pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts and '\\\\' not in name and ':' not in name
        target=WORKSPACE/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            partial=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
            with bundle.open(name) as source,partial.open('xb') as output:shutil.copyfileobj(source,output,8*1024*1024)
            assert partial.stat().st_size==identity['bytes'] and file_sha(partial)==identity['sha256'],name
            partial.replace(target)
        assert target.stat().st_size==identity['bytes'] and file_sha(target)==identity['sha256'],name
PLAN=WORKSPACE/manifest['plan_relative_path'];assert file_sha(PLAN)==manifest['plan_sha256']
DURABLE=DRIVE_ROOT/'runtime_state'/('88_empirical_v2_'+manifest['plan_sha256'][:16])
PREPARED=DURABLE/'prepared';SCORES=DURABLE/'scores'
ENTRY=WORKSPACE/'Poker Assistant Model/Stage 6 - Qualified Layers Improvement/src'
def execute(command):
    process=subprocess.Popen(command,cwd=WORKSPACE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:print(line.rstrip(),flush=True)
    assert process.wait()==0,'Execution failed; preserve exact evidence and checkpoints'
print({{'authenticated_members':len(manifest['members']),'plan_sha256':manifest['plan_sha256'],'durable':str(DURABLE)}})
""")
    md('## Causal source preparation\nAll older games remain eligible history. Public observations enter features only after their hand completes and before the decision. Monetary all-in calls retain their true aggressor; exact identities are remapped without using names. Missing newer reveal labels remain unobserved. A completed preparation resumes by its receipt.')
    code("""execute([sys.executable,str(ENTRY/'stage6_empirical_preparation_v2.py'),'--plan',str(PLAN),'--output',str(PREPARED)])
preparation=json.loads((PREPARED/'preparation_receipt.json').read_text())
print({key:preparation[key] for key in ('decisions','history_rows','history_games','decision_games','quarantined_decisions')})
""")
    md('## Frozen model replay\nIndependent layer jobs have disjoint checkpoints, at most two processes and one inner thread. Completed layers resume without refitting or rescoring. The same causal rows feed each paired comparison.')
    code("""execute([sys.executable,str(ENTRY/'stage6_empirical_evaluation_v2.py'),'--snapshot-root',str(PREPARED),'--output',str(SCORES),'--workers',str(workers)])
report=json.loads((SCORES/'report.json').read_text())
print({'rows':report['rows'],'games':report['game_clusters'],'cohorts':report['cohort_rows'],'design':report['confirmation_design']})
for name,layer in report['layers'].items():print(name,layer['qualification'])
""")
    md('## Publish direct results\nPublish the complete prepared source, protocol, per-layer predictions and receipts, then the embedded manifest last. No automatic promotion. This exact result is a prerequisite for downstream empirical comparisons.')
    code("""RETURN=DRIVE_ROOT/'returned_results'/('88_Stage6_Completed_Replay_'+report['run_id'][:16]+'.zip')
members={}
for path in PREPARED.iterdir():
    if path.is_file() and path.suffix!='.partial':members['prepared/'+path.name]=path
for path in SCORES.iterdir():
    if path.is_file() and (path.name.endswith('.predictions.npz') or path.name.endswith('.receipt.json') or path.name in ('report.json','report_receipt.json','run_identity.json')):members['scores/'+path.name]=path
result_manifest={'schema_version':'stage6_empirical_completed_result_v2','run_id':report['run_id'],'package_sha256':EXPECTED_PACKAGE_SHA256,'plan_sha256':manifest['plan_sha256'],'members':{name:{'bytes':path.stat().st_size,'sha256':file_sha(path)} for name,path in sorted(members.items())}}
if not RETURN.exists():
    partial=RETURN.with_name(RETURN.name+'.partial-'+uuid.uuid4().hex)
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,path in sorted(members.items()):bundle.write(path,name)
        bundle.writestr('result_manifest.json',json.dumps(result_manifest,sort_keys=True,separators=(',',':')))
    partial.replace(RETURN)
with zipfile.ZipFile(RETURN) as bundle:
    assert len(bundle.namelist())==len(set(bundle.namelist()))
    assert json.loads(bundle.read('result_manifest.json'))==result_manifest
    for name,identity in result_manifest['members'].items():
        with bundle.open(name) as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==identity['sha256']
print({'result':str(RETURN),'bytes':RETURN.stat().st_size,'sha256':file_sha(RETURN),'status':'PUBLISHED_AWAITING_EXTERNAL_READBACK_AND_STRICT_IMPORT'})
""")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python','version':'3.12'}})
    nbformat.validate(nb)
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,str(path),'exec')
    nbformat.write(nb,path)
    return path


def build(canary=False,revision='r1',snapshot_manifest=None):
    destination,plan=create_plan(canary=canary,revision=revision,snapshot_manifest=snapshot_manifest)
    if canary:return {'plan':str(destination/'plan.json'),'plan_sha256':sha(destination/'plan.json')}
    archive,manifest=package(destination,plan,revision)
    note=notebook(archive,manifest,revision)
    release={'schema_version':'stage6_empirical_completed_release_v2','package':{'path':str(archive),**identity(archive)},
        'notebook':{'path':str(note),**identity(note)},'plan':{'path':str(destination/'plan.json'),**identity(destination/'plan.json')},
        'status':'LOCAL_FROZEN_PENDING_INDEPENDENT_REVIEW_AND_DRIVE_READBACK'}
    write_json(STAGE/f'manifests/stage6_empirical_v2_completed_release_{revision}.json',release)
    return release


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--canary',action='store_true');parser.add_argument('--revision',default='r1')
    parser.add_argument('--snapshot-manifest',type=Path)
    args=parser.parse_args();print(json.dumps(build(args.canary,args.revision,args.snapshot_manifest),indent=2))
