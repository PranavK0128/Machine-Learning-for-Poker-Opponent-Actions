"""Strict no-overwrite landing of Notebook 88 results; never promotes a policy."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from stage6_empirical_evaluation_v2 import ACTIONS, canonical, sha, write_json, whole_game_cohorts, layer_disposition


def safe_inventory(bundle,manifest):
    names=bundle.namelist()
    if len(names)!=len(set(names)) or set(names)!=set(manifest['members'])|{'result_manifest.json'}:
        raise ValueError('Result inventory differs or has duplicates')
    for name,expected in manifest['members'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe result path')
        member=bundle.getinfo(name)
        if member.file_size!=expected['bytes'] or (member.external_attr>>16)&0o170000==0o120000:
            raise ValueError('Result member size/type differs')
        with bundle.open(name) as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected['sha256']:
                raise ValueError('Result member hash differs')


def inspect_result(archive:Path,release_path:Path) -> tuple[dict,dict]:
    release=json.loads(release_path.read_bytes())
    plan_path=Path(release['plan']['path'])
    if sha(plan_path)!=release['plan']['sha256']:raise ValueError('Local frozen plan changed')
    plan=json.loads(plan_path.read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('result_manifest.json'))
        if manifest.get('schema_version')!='stage6_empirical_completed_result_v2':raise ValueError('Wrong result schema')
        if manifest['package_sha256']!=release['package']['sha256'] or manifest['plan_sha256']!=sha(plan_path):
            raise ValueError('Result belongs to a different frozen release')
        safe_inventory(bundle,manifest)
        read=lambda name:json.loads(bundle.read(name))
        protocol=read('prepared/protocol.json');preparation=read('prepared/preparation_receipt.json')
        expected_protocol={**plan['evaluation_protocol'],'code_files':plan['evaluation_code_files'],
            'preparation_id':sha(plan_path),'files':{name:manifest['members']['prepared/'+name] for name in ('decisions.parquet','history.parquet')}}
        if protocol!=expected_protocol:raise ValueError('Protocol or prepared-source binding differs')
        if preparation['preparation_id']!=sha(plan_path) or preparation['raw_source_plan_sha256']!=sha(plan_path):
            raise ValueError('Preparation receipt belongs to different inputs')
        for name,expected in preparation['files'].items():
            if manifest['members'].get('prepared/'+name)!=expected:raise ValueError('Preparation receipt member differs')
        run_id=hashlib.sha256(canonical({'protocol':hashlib.sha256(bundle.read('prepared/protocol.json')).hexdigest(),'direct_code':protocol['code_files']})).hexdigest()
        report=read('scores/report.json');receipt=read('scores/report_receipt.json')
        if any(value!=run_id for value in (manifest['run_id'],report['run_id'],receipt['run_id'],read('scores/run_identity.json')['run_id'])):
            raise ValueError('Run identity differs')
        if receipt['sha256']!=hashlib.sha256(bundle.read('scores/report.json')).hexdigest():raise ValueError('Report receipt differs')
        if report['protocol_sha256']!=hashlib.sha256(bundle.read('prepared/protocol.json')).hexdigest() or report['authentication']['manifest_sha256']!=protocol['installed_manifest_sha256']:
            raise ValueError('Report model/protocol authentication differs')
        if report['fit_performed'] is not False or any(report['claims'].values()):raise ValueError('Unsupported inference/authority claim')
        decisions=pd.read_parquet(io.BytesIO(bundle.read('prepared/decisions.parquet')))
        history=pd.read_parquet(io.BytesIO(bundle.read('prepared/history.parquet')),columns=['game_id','decision_utc'])
        cohorts=whole_game_cohorts(decisions,history,protocol)
        if report['rows']!=len(decisions) or report['game_clusters']!=decisions.game_id.nunique() or report['cohort_rows']!=cohorts.value_counts().to_dict():
            raise ValueError('Report source/cohort counts differ')
        from stage6_shadow_layer_fresh_backtest import EXPECTED_LAYERS,MODEL_MEMBER_BY_LAYER,INSTALL_MANIFEST
        if sha(INSTALL_MANIFEST)!=protocol['installed_manifest_sha256']:raise ValueError('Frozen installed model manifest differs')
        installed={row['layer']:row for row in json.loads(INSTALL_MANIFEST.read_bytes())['layers']}
        if set(report['layers'])!=set(EXPECTED_LAYERS):raise ValueError('Wrong model-layer inventory')
        mandatory={'prepared/'+name for name in ('decisions.parquet','history.parquet','quarantine.parquet','protocol.json','preparation_receipt.json')}
        mandatory|={'scores/'+name for name in ('report.json','report_receipt.json','run_identity.json')}
        mandatory|={f'scores/{layer}.{suffix}' for layer in EXPECTED_LAYERS for suffix in ('predictions.npz','receipt.json')}
        if set(manifest['members'])!=mandatory:raise ValueError('Unexpected/missing result evidence')
        legal=decisions[[f'legal_{action}' for action in ACTIONS]].to_numpy(dtype=bool)
        target=np.array([ACTIONS.index(value) for value in decisions.action_label])
        if not legal[np.arange(len(target)),target].all():raise ValueError('Observed action is illegal')
        for layer,value in report['layers'].items():
            prediction_name=f'scores/{layer}.predictions.npz';layer_receipt=read(f'scores/{layer}.receipt.json')
            if layer_receipt['prediction_sha256']!=manifest['members'][prediction_name]['sha256']:
                raise ValueError('Prediction receipt differs')
            model_sha=installed[layer]['files'][MODEL_MEMBER_BY_LAYER[layer]]['sha256']
            if any(observed!=model_sha for observed in (value['model_sha256'],layer_receipt['model_sha256'],layer_receipt['task']['model_sha256'])):
                raise ValueError('Per-layer model identity differs')
            if value['qualification']!=layer_disposition(layer,value['metrics'],protocol):raise ValueError('Qualification gate was not reproduced')
            with np.load(io.BytesIO(bundle.read(prediction_name)),allow_pickle=False) as data:
                probability,prior,support=data['probabilities'],data['prior'],data['support']
                if support.shape!=(len(decisions),) or support.dtype!=np.bool_:raise ValueError('Support shape/type differs')
                if not math.isclose(value['coverage'],float(support.mean()),abs_tol=1e-12):raise ValueError('Coverage differs')
                for array in (probability,prior):
                    if array.shape!=legal.shape or not np.isfinite(array).all() or (array<0).any() or (array[~legal]!=0).any() or not np.allclose(array.sum(axis=1),1,atol=1e-6):
                        raise ValueError('Prediction probability/legal contract differs')
                # Arithmetic verification of existing outputs, not a new model
                # run or a fresh evaluation. Full bootstrap is retained for
                # independent scientific review from these exact predictions.
                for cohort in ('historical_development','training_overlap','confirmation'):
                    mask=support&cohorts.eq(cohort).to_numpy();metric=value['metrics'][cohort]['frozen_prior']
                    if metric['rows']!=int(mask.sum()):raise ValueError('Paired support count differs')
                    if not mask.any():continue
                    observed=target[mask];pred=probability[mask];base=prior[mask]
                    loss=-np.log(np.clip(pred[np.arange(len(observed)),observed],1e-15,1))
                    base_loss=-np.log(np.clip(base[np.arange(len(observed)),observed],1e-15,1))
                    if not math.isclose(metric['log_loss_improvement'],float((base_loss-loss).mean()),abs_tol=1e-10):
                        raise ValueError('Paired loss arithmetic differs')
        inspection={'schema_version':'stage6_empirical_import_receipt_v2','archive':{'bytes':archive.stat().st_size,'sha256':sha(archive)},
            'release_sha256':sha(release_path),'run_id':run_id,'rows':len(decisions),'game_clusters':decisions.game_id.nunique(),
            'status':'PASS_STRICT_TRANSPORT_SCHEMA_CAUSAL_MEMBERSHIP_AND_OUTPUT_ARITHMETIC',
            'independent_scientific_approval':False,'main_recommendation_promotion':False,
            'predictive_qualifications':{layer:value['qualification'] for layer,value in report['layers'].items()}}
        return inspection,manifest


def import_result(archive,release,destination):
    inspection,manifest=inspect_result(archive,release)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name,expected in manifest['members'].items():
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if sha(path)!=expected['sha256']:raise ValueError('Refusing to overwrite differing landed result')
            else:
                partial=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
                with bundle.open(name) as source,partial.open('xb') as target:
                    import shutil
                    shutil.copyfileobj(source,target,8*1024*1024)
                if partial.stat().st_size!=expected['bytes'] or sha(partial)!=expected['sha256']:
                    raise ValueError('Incomplete landed bytes; partial evidence preserved')
                partial.replace(path)
    write_json(destination/'import_receipt.json',inspection)
    return inspection


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--release',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(import_result(args.archive,args.release,args.destination),indent=2))
