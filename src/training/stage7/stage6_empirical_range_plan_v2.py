"""Freeze Notebook 90 memberships from authenticated source metadata only.

This is a chronology/inventory operation. It does not construct model features,
inspect outcome performance, fit, calibrate or score any real-data model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).resolve().parents[1]


def identity(path):
    with Path(path).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'bytes':Path(path).stat().st_size,'sha256':digest}


def instant(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Explicit source timezone required')
    return result


def cohort_template(bounds, inspected):
    train_end=instant('2026-08-01T00:00:00Z')
    evaluation_start=instant('2026-08-21T00:00:00Z')
    result={role+'_game_ids':[] for role in ('train','calibration','evaluation','diagnostic')}
    for game,(first,last) in sorted(bounds.items()):
        first,last=instant(first),instant(last)
        if last<first:raise ValueError('Invalid whole-game source bounds')
        if last<train_end:role='train'
        elif first>=train_end and last<evaluation_start:role='calibration'
        elif first>=evaluation_start and game not in inspected:role='evaluation'
        else:role='diagnostic'
        result[role+'_game_ids'].append(game)
    result.update(selection_assumption='revealed_subset_only_unidentified_population',
        previously_inspected_game_ids=sorted(inspected),workers='auto',
        model_config={'iterations':200,'num_leaves':31,'learning_rate':.05,'min_child_samples':20,'seed':20260904},
        range_diagnostic_rows_per_game=8,
        diagnostic_controls={'schema_version':'stage6_empirical_range_learning_v2_diagnostic_controls',
            'stale_prior_cutoff_utc':'2026-07-01T00:00:00Z','permutation_seed':20260904})
    return result


def build(completed_plan,archive,output):
    from stage6_empirical_range_learning_v2_adapter import streaming_game_bounds
    from stage6_empirical_range_learning_v2_release import validate_plan,publish_local_bytes
    completed_plan,archive,output=map(Path,(completed_plan,archive,output))
    if output.exists():raise FileExistsError('Preserve existing range plan')
    plan=json.loads(completed_plan.read_bytes())
    if not plan.get('full_completed_source'):raise ValueError('Full completed source is required for this plan')
    snapshot_path=ROOT/plan['completed_snapshot_path']
    if identity(snapshot_path)['sha256']!=plan['completed_snapshot_sha256']:raise ValueError('Snapshot changed')
    snapshot=json.loads(snapshot_path.read_bytes())
    source={role:ROOT/plan['source_roles'][name] for role,name in [('actions','FRESH_ACTIONS'),('identities','FRESH_IDENTITIES')]}
    source['hands']=ROOT/plan['shared_range_source']['hands']['path']
    expected={role:plan['source_files'][path.relative_to(ROOT).as_posix()] for role,path in source.items() if role!='hands'}
    expected['hands']={key:plan['shared_range_source']['hands'][key] for key in ('bytes','sha256')}
    for role,path in source.items():
        if identity(path)!=expected[role]:raise ValueError('Completed source changed: '+role)
    exposures_path=ROOT/plan['outcome_exposure_metadata']['path']
    if identity(exposures_path)!={key:plan['outcome_exposure_metadata'][key] for key in ('bytes','sha256')}:
        raise ValueError('Exposure inventory changed')
    inspected=set(plan['evaluation_protocol']['previously_inspected_game_ids'])
    # This shared helper reads only game/hand IDs and time bounds. No conversion,
    # private labels, public features or prediction outcomes are evaluated here.
    bounds,hands,public_ids=streaming_game_bounds(source['hands'])
    if identity(source['hands'])!=expected['hands']:raise ValueError('Sidecar changed during chronology inventory')
    with __import__('zipfile').ZipFile(archive) as bundle:
        package=json.loads(bundle.read('package_manifest.json'))
        if package['plan_sha256']!=identity(completed_plan)['sha256']:raise ValueError('Wrong shared archive plan')
        for role,path in source.items():
            member=path.relative_to(ROOT).as_posix()
            if package['members'].get(member)!=expected[role] or bundle.getinfo(member).file_size!=expected[role]['bytes']:
                raise ValueError('Shared archive source member mismatch')
    transport=identity(archive)
    value={'schema_version':'stage6_empirical_range_learning_v2_release_plan','release_id':'stage6_empirical_range_response_v2_r1',
        'drive_stage_path':'/content/drive/MyDrive/Poker Assistant Model/Stage 6 - Qualified Layers Improvement',
        'inputs':{role:{'file_name':archive.name,**transport,'member':path.relative_to(ROOT).as_posix(),
            'member_bytes':expected[role]['bytes'],'member_sha256':expected[role]['sha256']} for role,path in source.items()},
        'exposure_manifest_sha256':plan['outcome_exposure_metadata']['sha256'],
        'producer_receipt_sha256':snapshot['producer_receipt']['sha256'],
        'cohort_template':cohort_template(bounds,inspected),
        'chronology_inventory':{'source_sidecar':expected['hands'],'hands':hands,'game_bounds':bounds,
            'pokernow_to_canonical_games':public_ids,'model_features_or_outcomes_scored':False,
            'completed_snapshot_sha256':identity(snapshot_path)['sha256'],
            'created_utc':datetime.now(timezone.utc).isoformat()}}
    validate_plan(value)
    publish_local_bytes(output,(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode())
    return {'path':str(output),**identity(output),'source_hands':hands,
        'cohort_games':{role:len(value['cohort_template'][role+'_game_ids']) for role in ('train','calibration','evaluation','diagnostic')}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completed-plan',type=Path,required=True)
    parser.add_argument('--shared-package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();done=threading.Event();started=time.monotonic()
    def heartbeat():
        while not done.wait(20):print(json.dumps({'phase':'metadata_only_range_cohort_freeze','elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:print(json.dumps(build(args.completed_plan,args.shared_package,args.output),indent=2),flush=True)
    finally:done.set();thread.join()
