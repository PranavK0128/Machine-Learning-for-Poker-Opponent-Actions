"""Prepare read-only producer adapters without editing StackHUD-owned source.

Only path ownership changes in copied adapters; exporter query/row semantics
remain the existing producer implementations. Export commands are explicit,
no-overwrite, and bound to a completed processing/validation receipt.
"""
from __future__ import annotations
import hashlib
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).resolve().parents[1]
PRODUCER=Path('C:/Users/konda/Stack Society PokerNow Analyzer')


def identity(path):
    with Path(path).open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':digest}


def older_hand_coverage(old_hands, action_hands, identity_hands, observation_hands):
    """A replacement must preserve each accepted hand in every source role."""
    old_hands=set(old_hands)
    roles={'actions':set(action_hands),'identities':set(identity_hands),
           'observations':set(observation_hands)}
    covered=set.intersection(old_hands,*roles.values())
    missing=lambda values:[{'game_id':game,'hand_id':hand} for game,hand in sorted(values)]
    return {'known_hands':len(old_hands),'covered_hands':len(covered),
        'missing_hand_ids':missing(old_hands-covered),
        'source_roles':{role:{'covered_hands':len(old_hands&hands),
            'missing_hand_ids':missing(old_hands-hands)} for role,hands in roles.items()}}


def require_full_history_coverage(coverage):
    roles=coverage.get('source_roles',{})
    if set(roles)!={'actions','identities','observations'} or any(
        role.get('missing_hand_ids') or role.get('covered_hands')!=coverage.get('known_hands')
        for role in roles.values()) or coverage.get('missing_hand_ids') or coverage.get('covered_hands')!=coverage.get('known_hands'):
        raise ValueError('Full source is missing older usable hands in actions, identities or observations; explicit legacy union is required')


def prepare_adapters(revision='r2',snapshot_name='completed_snapshot_v2_20260904',full_source=False):
    receipt_path=PRODUCER/'data/manifests/deployment/smart-pipeline-resume.json'
    receipt=json.loads(receipt_path.read_bytes())
    if receipt['status']!='completed' or receipt['completedCommands']!=receipt['plannedCommands']:
        raise ValueError('Producer is not complete')
    reports=[]
    for name in ('db-validation-summary.json','stage-6-derived-validation.json','ledger-validation-summary.json',
                 'identity-review-validation-summary.json','action-state-validation-summary.json','hand-search-validation-summary.json'):
        path=PRODUCER/'data/manifests/validation'/name
        value=json.loads(path.read_bytes())
        summary=value.get('summary',value)
        if summary.get('errorCount') != 0:
            raise ValueError(f'Producer validation not accepted: {name}')
        reports.append(identity(path))
    if not revision.isalnum() or not snapshot_name.replace('_','').isalnum():raise ValueError('Unsafe snapshot/revision name')
    destination=STAGE/'inputs'/snapshot_name
    adapters=STAGE/f'runtime_state/empirical_v2_producer_adapters_{revision}'
    destination.mkdir(parents=True,exist_ok=True);adapters.mkdir(parents=True,exist_ok=True)
    files=[]
    for name in ('stage6_decision_time_identity_exporter.ts','stage6_hand_profile_observation_exporter.ts'):
        source=STAGE/'src'/name;raw=source.read_text()
        raw=raw.replace('"../../../scripts/export/exportFileWriter"',json.dumps((PRODUCER/'scripts/export/exportFileWriter.ts').as_uri()))
        raw=raw.replace('"../../../packages/db/src/index"',json.dumps((PRODUCER/'packages/db/src/index.ts').as_uri()))
        # Keep the explicit standalone allowlist; only DB/import ownership is
        # canonical. Caller runs from the actual standalone Stage 6 root.
        raw=raw.replace('"C:/Users/konda/Stack Society PokerNow Analyzer"','"C:/Users/konda/Poker Assistant"')
        target=adapters/name
        if target.exists() and target.read_text()!=raw:raise ValueError('Different existing adapter')
        if not target.exists():target.write_text(raw,encoding='utf-8')
        files.append({'source':identity(source),'adapter':identity(target),
                      'changes':'absolute_canonical_imports_and_standalone_output_cwd_only'})
    binding={'schema_version':'stage6_empirical_completed_producer_preparation_v2',
        'created_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt':identity(receipt_path),
        'producer':receipt,'validation_reports':reports,'adapters':files,
        'action_exporter':identity(PRODUCER/'scripts/export/db-export-action-state-decision-rows.ts'),
        'output_directory':str(destination),'started_at_min_ms':None if full_source else 1787270400000,
        'full_completed_source':full_source,
        'historical_inputs_preserved':True,'status':'ADAPTERS_PREPARED_EXPORTS_NOT_YET_COMPLETE'}
    path=STAGE/f'manifests/stage6_empirical_v2_producer_preparation_{revision}.json'
    if path.exists():raise ValueError('Preparation binding exists; preserve it')
    path.write_text(json.dumps(binding,indent=2)+'\n')
    return binding


def seal_exports(revision='r2'):
    """Inventory source identities/game membership before any outcome scoring."""
    import pandas as pd
    binding_path=STAGE/f'manifests/stage6_empirical_v2_producer_preparation_{revision}.json'
    binding=json.loads(binding_path.read_bytes())
    root=Path(binding['output_directory'])
    observed_receipt=identity(binding['producer_receipt']['path'])
    receipt_transition=None
    if observed_receipt != binding['producer_receipt']:
        current=json.loads(Path(observed_receipt['path']).read_bytes())
        export_finished=max(path.stat().st_mtime for path in root.glob('*.csv'))
        next_started=datetime.fromisoformat(current['createdAt'].replace('Z','+00:00')).timestamp()
        if current['processingRunId']==binding['producer']['processingRunId'] or next_started<=export_finished:
            raise ValueError('Producer receipt changed before the exports completed')
        receipt_transition={'new_run_receipt':observed_receipt,'new_run_started_utc':current['createdAt'],
            'exports_completed_utc':datetime.fromtimestamp(export_finished,timezone.utc).isoformat(),
            'classification':'NEXT_PRODUCER_RUN_STARTED_AFTER_ALL_EXPORTS_COMPLETED'}
    for expected in [binding['action_exporter'],*binding['validation_reports']]:
        if identity(expected['path']) != expected:
            raise ValueError(f"Producer changed during export: {expected['path']}")
    for pair in binding['adapters']:
        for expected in (pair['source'],pair['adapter']):
            if identity(expected['path']) != expected:raise ValueError('Exporter changed during export')
    keys=['hand_action_id','hand_id','game_id','event_index']
    actions=pd.read_csv(root/'fresh-action-state-decision-rows.csv',usecols=keys+['hand_started_at'],dtype=str)
    ids=pd.read_csv(root/'fresh-decision-time-identity.csv',dtype=str)
    if not actions[keys].equals(ids[keys]):raise ValueError('Export identity/order mismatch')
    if actions.hand_action_id.isna().any() or actions.hand_action_id.duplicated().any() or actions[keys[1:]].duplicated().any():
        raise ValueError('Duplicate/missing decision identity')
    if not ids.identity_grouping_status.eq('resolved_retrospective_group').all():raise ValueError('Unresolved identities')
    old=pd.read_parquet(STAGE/'evidence/shadow_layer_evaluation_20260829/prepared-completed-history.parquet',columns=['game_id','hand_id','decision_utc'])
    known=set(old.game_id)
    # This inventory does not inspect model outcomes. Known games retain their
    # prior exposure even when new hands from the same game are now available.
    game_rows=[]
    for game,frame in ids.groupby('game_id',sort=True):
        starts=actions.loc[frame.index,'hand_started_at']
        game_rows.append({'game_id':game,'decisions':len(frame),'hands':frame.hand_id.nunique(),
            'minimum_decision_utc':frame.decision_utc.min(),'maximum_decision_utc':frame.decision_utc.max(),
            'minimum_exported_hand_start':starts.min(),'previously_in_prepared_history':game in known})
    observations=pd.read_csv(root/'fresh-hand-profile-observations.csv',usecols=['game_id','hand_id','pokernow_player_id','identity_grouping_status'],dtype=str)
    if observations.duplicated(['game_id','hand_id','pokernow_player_id']).any():raise ValueError('Duplicate hand observations')
    if not observations.identity_grouping_status.eq('resolved_retrospective_group').all():raise ValueError('Unresolved observations')
    if set(zip(actions.game_id,actions.hand_id))-set(zip(observations.game_id,observations.hand_id)):
        raise ValueError('Action hands lack completed observation membership')
    old_hands=set(zip(old.game_id,old.hand_id))
    coverage=older_hand_coverage(old_hands,set(zip(actions.game_id,actions.hand_id)),
        set(zip(ids.game_id,ids.hand_id)),set(zip(observations.game_id,observations.hand_id)))
    files={path.name:identity(path) for path in sorted(root.glob('*.csv'))}
    result={'schema_version':'stage6_empirical_completed_snapshot_v2','created_utc':datetime.now(timezone.utc).isoformat(),
        'producer_preparation':identity(binding_path),'producer_receipt':binding['producer_receipt'],
        'subsequent_producer_run':receipt_transition,
        'files':files,'raw_decisions':len(ids),'hand_observations':len(observations),
        'full_completed_source':binding.get('full_completed_source',False),
        'older_prepared_hand_coverage':coverage,
        'game_count':len(game_rows),'hands':actions.hand_id.nunique(),'games':game_rows,
        'previously_inspected_game_ids':sorted(known),
        'new_game_ids':sorted(set(ids.game_id)-known),'model_outcomes_inspected':False,
        'status':'COMPLETE_VALIDATED_SOURCE_EXPORTED_COHORT_METADATA_FROZEN'}
    target=STAGE/('manifests/stage6_empirical_v2_completed_snapshot'+('' if revision=='r2' else '_'+revision)+'.json')
    if target.exists():raise ValueError('Completed snapshot already sealed')
    target.write_text(json.dumps(result,indent=2)+'\n')
    return {'manifest':str(target),'sha256':identity(target)['sha256'],'raw_decisions':len(ids),
            'games':len(game_rows),'new_games':len(result['new_game_ids']),'hands':result['hands']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal-exports',action='store_true')
    parser.add_argument('--revision',default='r2');parser.add_argument('--snapshot-name',default='completed_snapshot_v2_20260904')
    parser.add_argument('--full-source',action='store_true')
    args=parser.parse_args()
    print(json.dumps(seal_exports(args.revision) if args.seal_exports else prepare_adapters(args.revision,args.snapshot_name,args.full_source),indent=2))
