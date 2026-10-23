"""Build the minimal selected-model replay package and runnable Colab notebook."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd

from stage6_empirical_evaluation_v2 import canonical, sha, write_json

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parents[1]
OLD = STAGE/'evidence/shadow_layer_evaluation_20260829'
RUNTIME = ROOT/'Assistant Qualified/Assistant Code/runtime'


def dependencies(entry: Path) -> list[Path]:
    queue, visited = [entry], set()
    while queue:
        path=queue.pop()
        if path in visited:
            continue
        visited.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='SCIENCE_IMPLEMENTATION_MEMBERS' for target in node.targets):
                for relative in ast.literal_eval(node.value):
                    member=(STAGE/relative).resolve()
                    if not member.is_relative_to(STAGE) or not member.is_file():
                        raise ValueError('Missing/unsafe runtime source-binding dependency')
                    queue.append(member)
            if isinstance(node,ast.Import):
                modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    candidate=base/((node.module or '').replace('.','/')+'.py')
                    if candidate.is_file():queue.append(candidate)
                elif node.module:
                    modules=[node.module]
            for module in modules:
                for base in (STAGE/'src',RUNTIME):
                    candidate=base/(module.replace('.','/')+'.py')
                    if candidate.is_file():queue.append(candidate)
    return sorted(visited)


def identity(path: Path) -> dict:
    return {'bytes':path.stat().st_size,'sha256':sha(path)}


def prepare_snapshot(destination: Path, *, canary: bool=False) -> dict:
    receipt_path=OLD/'stage6_shadow_backtest_preparation_receipt.json'
    receipt=json.loads(receipt_path.read_bytes())
    from stage6_shadow_layer_fresh_backtest import validate_seal, authenticate_frozen_installation
    validate_seal(receipt)
    source_decisions=OLD/'prepared-fresh-decisions.parquet'
    source_history=OLD/'prepared-completed-history.parquet'
    for name,path in [('prepared_fresh_decisions',source_decisions),('prepared_completed_history',source_history)]:
        if identity(path) != {key:receipt[name][key] for key in ('bytes','sha256')}:
            raise ValueError('Prepared source identity mismatch')
    authentication,_=authenticate_frozen_installation()
    # Membership extraction is packaging metadata, not model fitting or scoring.
    history_keys=pd.read_parquet(source_history,columns=['game_id','decision_utc'])
    game_keys=pd.read_parquet(source_decisions,columns=['game_id','decision_utc'])
    training=sorted(history_keys.loc[pd.to_datetime(history_keys.decision_utc,utc=True)<pd.Timestamp('2026-08-21T00:00:00Z'),'game_id'].unique())
    all_games=sorted(history_keys.game_id.unique())
    decisions_games=sorted(game_keys.game_id.unique())
    destination.mkdir(parents=True,exist_ok=True)
    if canary:
        frame=pd.read_parquet(source_decisions)
        selected=frame.hand_id.drop_duplicates().head(4)
        frame=frame[frame.hand_id.isin(selected)].copy()
        frame.to_parquet(destination/'decisions.parquet',index=False)
        history=pd.read_parquet(source_history)
        # Bounded mechanics fixture only; explicitly not population evaluation.
        history=history.sort_values('decision_utc',kind='stable').tail(1000)
        history.to_parquet(destination/'history.parquet',index=False)
    else:
        for source,name in [(source_decisions,'decisions.parquet'),(source_history,'history.parquet')]:
            target=destination/name
            if target.exists() and identity(target)!=identity(source):
                raise ValueError('Destination contains a different snapshot')
            if not target.exists():shutil.copy2(source,target)
            if identity(target)!=identity(source):raise ValueError('Copy identity failed')
    files=dependencies(STAGE/'src/stage6_empirical_evaluation_v2.py')
    protocol={'schema_version':'stage6_empirical_selected_model_protocol_v2','synthetic_canary':canary,
        'candidate_frozen_utc':datetime.now(timezone.utc).isoformat(),
        'source_disposition':'authenticated_historical_prepared_snapshot_previously_inspected',
        'preparation_receipt_sha256':sha(receipt_path),'installed_manifest_sha256':authentication['manifest_sha256'],
        'previously_inspected_game_ids':all_games,'training_game_ids':training,
        'development_game_ids':sorted(set(decisions_games)-set(training)),'confirmation_game_ids':[],
        'minimum_confirmation_games':20,'minimum_positive_game_fraction':.5,'family_alpha':.05,
        'selection_policy':'reuse_exact_nine_frozen_winners_no_fit_no_temperature_tuning',
        'primary_endpoint':'confirmation_paired_log_loss_improvement_against_own_frozen_prior',
        'secondary_required_endpoint':'confirmation_paired_brier_improvement',
        'multiplicity':'Bonferroni_across_nine_layers_two_required_endpoints_intersection',
        'all_other_slices':'diagnostic_not_used_to_relabel_historical_data_as_confirmation',
        'strategy_promotion_from_prediction_alone':False,
        'files':{name:identity(destination/name) for name in ('decisions.parquet','history.parquet')},
        'code_files':{path.relative_to(ROOT).as_posix():identity(path) for path in files}}
    protocol_path=destination/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_bytes())
        protocol['candidate_frozen_utc']=previous['candidate_frozen_utc']
    write_json(protocol_path,protocol)
    return protocol


def build(*, canary: bool=False, revision: str='r1') -> dict:
    if not revision.isalnum():raise ValueError('Revision must be alphanumeric')
    suffix=f'canary_{revision}' if canary else 'historical'
    snapshot=STAGE/f'inputs/empirical_v2_{suffix}_snapshot'
    protocol=prepare_snapshot(snapshot,canary=canary)
    if canary:
        return {'snapshot':str(snapshot),'protocol_sha256':sha(snapshot/'protocol.json')}
    archive=STAGE/'inputs/stage6_empirical_v2_selected_replay.zip'
    static=STAGE/'manifests/stage6_empirical_v2_selected_replay.json'
    notebook=STAGE/'notebooks/88_Stage6_Selected_Models_Empirical_Replay_V2.ipynb'
    if any(path.exists() for path in (archive,static,notebook)):
        raise ValueError('Release already exists; preserve the frozen release')
    files={ROOT/relative for relative in protocol['code_files']}
    files.update(snapshot.iterdir())
    install_root=RUNTIME/'ml/artifacts/stage6_retrospective_nested_v2'
    install=json.loads((install_root/'install_manifest.json').read_bytes())
    files.add(install_root/'install_manifest.json')
    for layer in install['layers']:
        files.update(install_root/layer['layer']/name for name in layer['files'])
    members={path.relative_to(ROOT).as_posix():identity(path) for path in sorted(files)}
    manifest={'schema_version':'stage6_empirical_v2_replay_package','members':members,
        'snapshot_relative_path':snapshot.relative_to(ROOT).as_posix(),'protocol_sha256':sha(snapshot/'protocol.json')}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(files):
            raw=path.read_bytes();relative=path.relative_to(ROOT).as_posix()
            if hashlib.sha256(raw).hexdigest()!=members[relative]['sha256']:raise ValueError('Source changed during freeze')
            out.writestr(relative,raw)
        out.writestr('package_manifest.json',canonical(manifest))
    archive_sha=sha(archive)
    cells=[]
    def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):cells.append(nbformat.v4.new_code_cell(text))
    md('# 88 — Selected Stage 6 models: empirical replay\n\nRe-evaluate the nine selected models without retraining. This frozen source is historical development, not new untouched confirmation. The notebook produces per-layer predictions, paired losses, calibration and support slices. It cannot promote Hero strategy from prediction accuracy. CPU Python 3.12; no GPU work.\n\nRun every cell in order. A returned result is not accepted until independent raw-byte readback and strict import.')
    md('## 1. Runtime and durable storage\nMount once. Keep the exact closed checkpoints in this notebook-owned Drive directory. Reconnect only this owned runtime. Choose workers from actual CPU capacity; inner math threads stay one.')
    code("""import os, sys, json, hashlib, pathlib, subprocess, zipfile, time
assert sys.version_info[:2] == (3, 12), f'Python 3.12 required; observed {sys.version}'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from google.colab import drive
if not pathlib.Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
DRIVE_ROOT=pathlib.Path('/content/drive/MyDrive/Poker Assistant Model/Stage 6 - Qualified Layers Improvement')
REQUESTED_WORKERS=1
print({'python':sys.version,'cpus':os.cpu_count(),'requested_workers':REQUESTED_WORKERS})
""")
    md('## 2. Install the frozen inference environment\nNo search, fitting, calibration tuning or accelerator conversion is performed.')
    code("""requirements=['numpy==2.2.6','pandas==2.3.3','pyarrow==21.0.0','joblib==1.5.2','scikit-learn==1.7.2','lightgbm==4.6.0','catboost==1.2.10','xgboost==3.3.0','threadpoolctl==3.6.0']
subprocess.run([sys.executable,'-m','pip','install','--quiet',*requirements],check=True)
# Scoring runs in fresh subprocesses, so installing wheels cannot leave stale
# imported native libraries in the scientific process.
print('Frozen inference dependencies installed; no model training.')
""")
    md('## 3. Authenticate and stage exact inputs\nEvery archive path, member length and SHA-256 is checked before importing scientific code or loading model bytes.')
    code(f"""PACKAGE=DRIVE_ROOT/'inputs/{archive.name}'
EXPECTED_PACKAGE_SHA256={archive_sha!r}
def file_sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
assert file_sha(PACKAGE)==EXPECTED_PACKAGE_SHA256,'Package SHA mismatch'
WORKSPACE=pathlib.Path('/content/stage6_empirical_v2_'+EXPECTED_PACKAGE_SHA256[:12])
WORKSPACE.mkdir(exist_ok=True)
with zipfile.ZipFile(PACKAGE) as bundle:
    names=bundle.namelist()
    assert len(names)==len(set(names)),'Duplicate ZIP members'
    manifest=json.loads(bundle.read('package_manifest.json'))
    assert set(names)==set(manifest['members'])|{{'package_manifest.json'}},'Member inventory mismatch'
    for name,identity in manifest['members'].items():
        relative=pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts and '\\\\' not in name and ':' not in name
        raw=bundle.read(name)
        assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256'],name
        target=WORKSPACE/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==raw,'Different existing staged bytes'
        else:target.write_bytes(raw)
SNAPSHOT=WORKSPACE/manifest['snapshot_relative_path']
assert file_sha(SNAPSHOT/'protocol.json')==manifest['protocol_sha256']
OUTPUT=DRIVE_ROOT/'runtime_state'/('88_empirical_v2_'+manifest['protocol_sha256'][:16])
print({{'authenticated_members':len(manifest['members']),'snapshot':str(SNAPSHOT),'output':str(OUTPUT)}})
""")
    md('## 4. Execute frozen models and paired comparisons\nFeature preparation preserves causal history. Layer jobs own separate outputs; completed predictions resume by exact task/hash receipts. Whole games remain in one cohort. The old final cohort is explicitly historical here.')
    code("""ENTRY=WORKSPACE/'Poker Assistant Model/Stage 6 - Qualified Layers Improvement/src/stage6_empirical_evaluation_v2.py'
command=[sys.executable,str(ENTRY),'--snapshot-root',str(SNAPSHOT),'--output',str(OUTPUT),'--workers',str(REQUESTED_WORKERS)]
started=time.monotonic()
process=subprocess.Popen(command,cwd=WORKSPACE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
for line in process.stdout:print(line.rstrip(),flush=True)
assert process.wait()==0,'Evaluation failed; preserve checkpoints and exact error'
print({'elapsed_seconds':time.monotonic()-started})
""")
    md('## 5. Review and publish the direct result ZIP\nRead per-layer qualifications rather than an overall green label. Missing confirmation or strategy evidence remains an explicit non-qualification. Do not tune a candidate on these reported outcomes and call the same games untouched.')
    code("""report=json.loads((OUTPUT/'report.json').read_text())
for name,layer in report['layers'].items():
    metric=layer['metrics']['historical_development']['frozen_prior']
    print(name,{'coverage':layer['coverage'],'log_loss_gain':metric.get('log_loss_improvement'),'games':metric.get('game_clusters'),'qualification':layer['qualification']})
RETURN=DRIVE_ROOT/'returned_results'/('88_Stage6_Empirical_Replay_'+report['run_id'][:16]+'.zip')
members=[OUTPUT/'report.json',OUTPUT/'report_receipt.json',OUTPUT/'run_identity.json',SNAPSHOT/'protocol.json']
members+=sorted(OUTPUT.glob('*.predictions.npz'))+sorted(OUTPUT.glob('*.receipt.json'))
result_manifest={'schema_version':'stage6_empirical_v2_result','run_id':report['run_id'],'package_sha256':EXPECTED_PACKAGE_SHA256,'members':{p.name:{'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in members}}
if not RETURN.exists():
    partial=RETURN.with_suffix('.partial')
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in members:bundle.write(path,path.name)
        bundle.writestr('result_manifest.json',json.dumps(result_manifest,sort_keys=True,separators=(',',':')))
    partial.replace(RETURN)
with zipfile.ZipFile(RETURN) as bundle:
    assert json.loads(bundle.read('result_manifest.json'))==result_manifest
    for name,identity in result_manifest['members'].items():
        raw=bundle.read(name)
        assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256']
print({'result':str(RETURN),'bytes':RETURN.stat().st_size,'sha256':file_sha(RETURN),'status':'PUBLISHED_AWAITING_EXTERNAL_READBACK_AND_IMPORT'})
""")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python','version':'3.12'}})
    nbformat.validate(nb);nbformat.write(nb,notebook)
    result={'schema_version':'stage6_empirical_v2_notebook_release','package':{'path':str(archive),**identity(archive)},
        'notebook':{'path':str(notebook),**identity(notebook)},'protocol_sha256':sha(snapshot/'protocol.json'),
        'direct_code':protocol['code_files'],'status':'LOCAL_CANDIDATE_PENDING_AUDIT_AND_DRIVE_READBACK'}
    write_json(static,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--canary',action='store_true')
    parser.add_argument('--revision',default='r1')
    args=parser.parse_args()
    print(json.dumps(build(canary=args.canary,revision=args.revision),indent=2))
