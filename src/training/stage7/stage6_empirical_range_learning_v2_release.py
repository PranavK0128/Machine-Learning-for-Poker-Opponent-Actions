"""Notebook 90 builder, durable checkpoint transport and strict result importer.

No release is built by importing this module. The parent supplies the final
completed-source plan and owns the immutable package/Drive/audit freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

SCHEMA='stage6_empirical_range_learning_v2'
AUTHORITY=dict(population_range_qualified=False,strategy_qualified=False,solver_value=False,promotion=False,runtime_weight=0,excluded_shadow_layers_reactivated=False,automatic_action=False)
SOURCE_NAMES=['stage6_empirical_range_learning_v2.py','stage6_empirical_range_learning_v2_adapter.py','stage6_empirical_range_learning_v2_streaming.py','stage6_empirical_range_learning_v2_release.py']
RESULT_NAMES={'range_response_artifact.json','range_response_report.json','completion_receipt.json','range_adapter_receipt.json','protocol.json','release_plan.json','return_manifest.json','identity_mapping.json'}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def identity(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite_json:'+s)))
def utc(value):
    if not isinstance(value,str) or not value.endswith('Z'):raise ValueError('explicit_utc_provenance_required')
    return datetime.fromisoformat(value[:-1]+'+00:00')
def safe_name(name):
    if not isinstance(name,str) or '\\' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or ':' in name:raise ValueError('unsafe_member_name')
    return name
def publish_local_bytes(path,data):
    """Closed local file publication without ever replacing an existing inode.

    A failed write/link leaves its uniquely named partial for inspection. Both
    supported execution destinations (local NTFS and Colab /content) support
    hard links. Drive transport continues to use its separate copy protocol.
    """
    path=Path(path);expected=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    if path.exists():verify(path,expected);return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial.'+uuid.uuid4().hex)
    with partial.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    verify(partial,expected)
    try:os.link(partial,path)
    except FileExistsError:verify(path,expected)
    partial.unlink()
def atomic_json(path,value):
    path=Path(path);data=(canonical(value)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('different_existing_json:'+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial.'+uuid.uuid4().hex)
    with partial.open('xb') as stream:stream.write(data)
    os.rename(partial,path)
def verify(path,expected):
    if identity(path)!=dict(bytes=expected['bytes'],sha256=expected['sha256']):raise ValueError('byte_identity_mismatch:'+str(path))


def validate_plan(plan):
    if plan.get('schema_version')!=SCHEMA+'_release_plan' or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,80}',plan.get('release_id','')):raise ValueError('invalid_release_plan_identity')
    if set(plan['inputs'])!={'actions','identities','hands'}:raise ValueError('three_exact_source_roles_required')
    for role,source in plan['inputs'].items():
        if Path(source['file_name']).name!=source['file_name'] or not re.fullmatch('[a-f0-9]{64}',source['sha256']) or source['bytes']<1:raise ValueError('invalid_source_transport_identity:'+role)
        if source.get('member') is not None:
            safe_name(source['member'])
            if source.get('member_bytes',0)<1 or not re.fullmatch('[a-f0-9]{64}',source.get('member_sha256','')):raise ValueError('exact_source_member_identity_required')
    if not isinstance(plan['drive_stage_path'],str) or not plan['drive_stage_path'].startswith('/content/drive/MyDrive/'):raise ValueError('explicit_mounted_stage_path_required')
    template=plan['cohort_template'];assigned=set()
    for role in ['train','calibration','evaluation','diagnostic']:
        values=template.get(role+'_game_ids',[])
        if len(values)!=len(set(values)) or assigned&set(values):raise ValueError('frozen_whole_game_membership_conflict')
        assigned.update(values)
    if set(template['evaluation_game_ids'])&set(template.get('previously_inspected_game_ids',[])):raise ValueError('previously_inspected_evaluation_game')
    if template['selection_assumption']!='revealed_subset_only_unidentified_population':raise ValueError('explicit_reveal_selection_limit_required')
    if template.get('diagnostic_controls')!=dict(schema_version=SCHEMA+'_diagnostic_controls',stale_prior_cutoff_utc='2026-07-01T00:00:00Z',permutation_seed=20260904):raise ValueError('explicit_frozen_profile_diagnostic_controls_required')
    if not re.fullmatch('[a-f0-9]{64}',plan.get('exposure_manifest_sha256','')) or not re.fullmatch('[a-f0-9]{64}',plan.get('producer_receipt_sha256','')):raise ValueError('producer_and_exposure_identity_required')
    return plan


def resolve_protocol(plan,adapter):
    accepted=set(adapter['accepted_games']);template=plan['cohort_template']
    planned=set().union(*(set(template.get(role+'_game_ids',[])) for role in ['train','calibration','evaluation','diagnostic']))
    if accepted-planned:raise ValueError('accepted_game_absent_from_frozen_cohorts')
    protocol=dict(template,schema_version=SCHEMA,input_sha256=adapter['members']['range_training_rows.jsonl']['sha256'],
        source_code_sha256=plan['code_files']['stage6_empirical_range_learning_v2.py']['sha256'],
        streaming_code_sha256=plan['code_files']['stage6_empirical_range_learning_v2_streaming.py']['sha256'])
    for role in ['train','calibration','evaluation','diagnostic']:
        protocol[role+'_game_ids']=[g for g in template.get(role+'_game_ids',[]) if g in accepted]
    protocol['zero_accepted_row_game_exclusions']=sorted(planned-accepted)
    protocol['zero_row_exclusion_rule']='Predeclared removal only; no reassignment of a game across cohorts.'
    return protocol


def verify_score_arithmetic(score,allowed_games,seed):
    """Recompute score arithmetic from paired game summaries, never refit."""
    import numpy as np
    def integer(value,minimum=0):
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError('invalid_score_count')
        return value
    def number(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('invalid_score_number')
        return float(value)
    def close(actual,expected):
        if not math.isclose(number(actual),float(expected),rel_tol=1e-10,abs_tol=1e-10):raise ValueError('score_arithmetic_mismatch')
    for claim in ['population_range_qualified','strategy_qualified']:
        if claim in score and score[claim] is not False:raise ValueError('score_scientific_claim_changed')
    count=integer(score['rows'])
    if not count:
        if score.get('games')!=0 or score.get('disposition')!='NO_SUPPORT':raise ValueError('invalid_empty_score')
        return
    integer(score['games'],1)
    paired=score['paired_game_statistics'];games=[item['game_id'] for item in paired]
    if not games or games!=sorted(set(games)) or not set(games)<=set(allowed_games) or score['games']!=len(games):raise ValueError('paired_score_game_membership_mismatch')
    metadata=dict(seed=seed,rng='numpy.random.PCG64',replicates=2000,lower_quantile=.025/2,quantile_method='linear',role_bonferroni_count=2,resampling_unit='whole_game',sampling='uniform_with_replacement',estimator='pooled_rows',game_order=games)
    if score['bootstrap']!=metadata:raise ValueError('score_bootstrap_protocol_changed')
    sums={role:dict(log_loss_sum=0.0,brier_sum=0.0,correct_count=0,calibration_bins=[dict(bin_index=i,count=0,confidence_sum=0.0,correct_count=0) for i in range(10)]) for role in ['candidate','prior']}
    totals=[];rows=0
    for item in paired:
        n=integer(item['row_count'],1);rows+=n
        for role in ['candidate','prior']:
            value=item[role];hits=integer(value['correct_count'])
            if hits>n or not 0<=number(value['log_loss_sum'])<=-math.log(1e-15)*n+1e-8 or not 0<=number(value['brier_sum'])<=2*n+1e-8:raise ValueError('impossible_game_score_sums')
            bins=value['calibration_bins']
            if len(bins)!=10:raise ValueError('ten_calibration_bins_required')
            bin_rows=bin_hits=0
            for i,entry in enumerate(bins):
                bcount=integer(entry['count']);correct=integer(entry['correct_count']);conf=number(entry['confidence_sum'])
                if entry['bin_index']!=i or correct>bcount or not i/10*bcount-1e-9<=conf<=(i+1)/10*bcount+1e-9:raise ValueError('invalid_calibration_bin_arithmetic')
                if not bcount and (conf!=0 or correct!=0):raise ValueError('nonzero_empty_calibration_bin')
                bin_rows+=bcount;bin_hits+=correct
                for key in ['count','confidence_sum','correct_count']:sums[role]['calibration_bins'][i][key]+=entry[key]
            if bin_rows!=n or bin_hits!=hits:raise ValueError('calibration_bin_denominator_mismatch')
            for key in ['log_loss_sum','brier_sum','correct_count']:sums[role][key]+=value[key]
        totals.append([n,item['prior']['log_loss_sum']-item['candidate']['log_loss_sum'],item['prior']['brier_sum']-item['candidate']['brier_sum']])
    if rows!=count:raise ValueError('paired_score_row_denominator_mismatch')
    for role in ['candidate','prior']:
        expected=sums[role];reported=score if role=='candidate' else score['prior_metrics']
        close(reported['log_loss'],expected['log_loss_sum']/count);close(reported['brier'],expected['brier_sum']/count)
        close(reported['accuracy'],expected['correct_count']/count)
        close(reported['expected_calibration_error_10_bins'],sum(abs(b['confidence_sum']-b['correct_count']) for b in expected['calibration_bins'])/count)
        reported_bins=score['calibration_bins'][role]
        if len(reported_bins)!=10:raise ValueError('ten_pooled_calibration_bins_required')
        for actual,wanted in zip(reported_bins,expected['calibration_bins']):
            for key in ['bin_index','count','correct_count']:
                if actual[key]!=wanted[key]:raise ValueError('pooled_calibration_bin_count_mismatch')
            close(actual['confidence_sum'],wanted['confidence_sum'])
    close(score['log_loss_improvement'],(sums['prior']['log_loss_sum']-sums['candidate']['log_loss_sum'])/count)
    close(score['brier_improvement'],(sums['prior']['brier_sum']-sums['candidate']['brier_sum'])/count)
    totals=np.asarray(totals,dtype=float);rng=np.random.default_rng(seed)
    boots=totals[rng.integers(0,len(games),size=(2000,len(games)))].sum(axis=1)
    lower=np.quantile(boots[:,1:]/boots[:,:1],.025/2,axis=0,method='linear')
    for i,key in enumerate(['log_loss','brier']):
        close(score['whole_game_lower_bounds'][key],lower[i]);close(score['positive_game_fraction'][key],np.mean(totals[:,i+1]>0))
    close(score['positive_game_fraction']['both'],np.mean((totals[:,1]>0)&(totals[:,2]>0)))
    if score['predictive_support_floor_met']!=(len(games)>=20) or score.get('population_range_qualified') is not False or score.get('strategy_qualified') is not False:raise ValueError('score_authority_or_support_floor_changed')


def verify_report_arithmetic(report,artifact,protocol):
    expected={'physical_combo_action_likelihood','observed_next_action_size'}
    if set(report.get('roles',{}))!=expected:raise ValueError('score_report_role_inventory_mismatch')
    if report.get('authority')!=AUTHORITY or report.get('source_support',{}).get('authority')!=AUTHORITY:raise ValueError('report_or_source_support_authority_changed')
    for role,value in report['roles'].items():
        if role not in artifact['models']:
            if value.get('disposition')!='INSUFFICIENT_OBSERVED_TARGET_SUPPORT' or value.get('authority')!=AUTHORITY or isinstance(value.get('rows'),bool) or not isinstance(value.get('rows'),int) or value['rows']<0:raise ValueError('unsupported_role_report_mismatch')
            continue
        if value.get('counterfactual_response_identified') is not False or value.get('disposition')!='ASSUMPTION_DEPENDENT_EXPERIMENT_RETAIN_SHADOW':raise ValueError('modeled_role_scientific_claim_changed')
        for partition in ['calibration','evaluation']:
            verify_score_arithmetic(value[partition],protocol[partition+'_game_ids'],protocol['model_config']['seed'])
            for sliced in value[partition].get('diagnostic_slices',{}).values():
                if sliced.get('disposition')!=('DESCRIPTIVE_SLICE_NO_ADDITIONAL_QUALIFICATION' if sliced['rows'] else 'NO_SUPPORT'):raise ValueError('diagnostic_slice_disposition_changed')


def verify_range_controls(report,artifact,protocol,adapter):
    """Check frozen profile-control provenance and returned summary arithmetic."""
    import numpy as np
    controls=protocol['diagnostic_controls'];history=artifact['diagnostic_prior_history'];full=artifact['prior_counts'];stale=history['stale_prior_counts']
    actors=sorted({actor for context in full.values() for actor in context})
    expected_stale=sorted(game for game in protocol['train_game_ids'] if utc(adapter['game_bounds'][game][1])<utc(controls['stale_prior_cutoff_utc']))
    wrong={actor:actors[(i+1)%len(actors)] for i,actor in enumerate(actors)}
    shuffled=dict(zip(actors,np.random.default_rng(controls['permutation_seed']).permutation(actors).tolist()))
    if history.get('schema_version')!=SCHEMA+'_prior_controls' or history.get('configuration')!=controls or history.get('stale_eligible_train_game_ids')!=expected_stale or history.get('training_profile_actor_ids')!=actors or history.get('wrong_player_map')!=wrong or history.get('shuffled_player_map')!=shuffled or history.get('shuffled_fixed_points')!=[a for a in actors if shuffled[a]==a]:raise ValueError('frozen_profile_control_provenance_mismatch')
    if history['stale_prior_counts_sha256']!=hashlib.sha256(canonical(stale).encode()).hexdigest():raise ValueError('stale_prior_count_identity_mismatch')
    if stale and not expected_stale:raise ValueError('stale_counts_without_eligible_games')
    for context,players in stale.items():
        for player,sessions in players.items():
            for session,value in sessions.items():
                parent=full.get(context,{}).get(player,{}).get(session)
                if parent is None or not set(value['events'])<=set(parent['events']) or utc(value['completed_through_utc'])>=utc(controls['stale_prior_cutoff_utc']):raise ValueError('noncausal_or_unmatched_stale_prior')
                if sum(value['counts'].values())!=len(value['events']) or any(isinstance(n,bool) or not isinstance(n,int) or n<0 or n>parent['counts'].get(key,0) for key,n in value['counts'].items()):raise ValueError('invalid_stale_prior_sufficient_statistics')
    diagnostic=report['range_diagnostics'];budget=protocol.get('range_diagnostic_rows_per_game',0)
    if diagnostic.get('authority')!=AUTHORITY:raise ValueError('range_diagnostic_authority_changed')
    if not budget or 'physical_combo_action_likelihood' not in artifact['models']:
        if diagnostic.get('status')!='NOT_RUN':raise ValueError('unexpected_unbudgeted_range_diagnostics')
        return
    if diagnostic.get('status')!='REVEAL_SELECTED_DIAGNOSTIC' or diagnostic.get('authority')!=AUTHORITY or diagnostic.get('diagnostic_controls')!=controls or diagnostic.get('profile_control_provenance')!={key:value for key,value in history.items() if key!='stale_prior_counts'}:raise ValueError('range_control_report_provenance_mismatch')
    results=diagnostic['results'];names={'uniform_no_profile','population_only','current_hand_likelihood_off','player_and_action_assumption','stale_profile_and_action','wrong_player_profile_and_action','shuffled_player_profile_and_action'}
    sample=[dict(row_id=row['row_id'],game_id=row['game_id']) for row in results]
    if len({row['row_id'] for row in results})!=len(results) or any(row['game_id'] not in protocol['evaluation_game_ids'] for row in results):raise ValueError('range_control_sample_identity_mismatch')
    expected_order=sorted(results,key=lambda row:(row['game_id'],hashlib.sha256(row['row_id'].encode()).hexdigest()))
    if results!=expected_order or diagnostic['sample_membership_sha256']!=hashlib.sha256(canonical(sample).encode()).hexdigest():raise ValueError('range_control_sample_order_or_hash_mismatch')
    game_counts={game:sum(row['game_id']==game for row in results) for game in {row['game_id'] for row in results}}
    if any(n>budget for n in game_counts.values()) or diagnostic['per_game_budget']!=budget or diagnostic['sampled_rows']!=len(results) or diagnostic['sampled_games']!=len(game_counts) or diagnostic['eligible_revealed_rows']<len(results):raise ValueError('range_control_sample_denominator_mismatch')
    for row in results:
        if set(row['scenarios'])!=names:raise ValueError('range_control_inventory_mismatch')
        for name,value in row['scenarios'].items():
            context=row['prior_context'];actor=row['opponent_id'];expected_hold=None
            if name=='stale_profile_and_action':
                if not expected_stale:expected_hold='no_complete_train_games_before_frozen_stale_cutoff'
                elif not stale.get(context):expected_hold='no_stale_revealed_prior_support_in_context'
                elif canonical(stale.get(context,{}))==canonical(full.get(context,{})):expected_hold='no_newer_training_prior_evidence_in_context'
            elif name in {'wrong_player_profile_and_action','shuffled_player_profile_and_action'}:
                donor=(wrong if name=='wrong_player_profile_and_action' else shuffled).get(actor)
                if donor is None:expected_hold='target_has_no_training_profile'
                elif donor==actor:expected_hold='fixed_point_retained_without_reshuffling'
                elif actor not in full.get(context,{}) or donor not in full.get(context,{}):expected_hold='no_matched_target_and_donor_profile_in_context'
            if expected_hold is not None and (value['status']!='HELD' or value.get('hold_reason')!=expected_hold) or expected_hold is None and value['status']!='SCORED':raise ValueError('profile_control_hold_rule_changed')
            if value['status']=='HELD':
                if name not in {'stale_profile_and_action','wrong_player_profile_and_action','shuffled_player_profile_and_action'} or any(value.get(metric) is not None for metric in ['log_loss','brier']):raise ValueError('invalid_held_control_metrics')
            elif value['status']=='SCORED':
                if any(isinstance(value[metric],bool) or not isinstance(value[metric],(int,float)) or not math.isfinite(value[metric]) or not 0<=value[metric]<=ceiling+1e-9 for metric,ceiling in [('log_loss',-math.log(1e-15)),('brier',2)]):raise ValueError('invalid_range_control_score')
            else:raise ValueError('unknown_range_control_disposition')
            if name in {'wrong_player_profile_and_action','shuffled_player_profile_and_action'}:
                expected=(wrong if name=='wrong_player_profile_and_action' else shuffled).get(row['opponent_id'])
                if value['profile_donor_id']!=expected or expected==row['opponent_id'] and value['status']!='HELD':raise ValueError('profile_control_permutation_changed')
    if set(diagnostic['summary'])!=names:raise ValueError('range_control_summary_inventory_mismatch')
    for name in names:
        scored=[row for row in results if row['scenarios'][name]['status']=='SCORED'];held=[row for row in results if row['scenarios'][name]['status']=='HELD'];summary=diagnostic['summary'][name]
        reasons={reason:sum(row['scenarios'][name]['hold_reason']==reason for row in held) for reason in {row['scenarios'][name]['hold_reason'] for row in held}}
        if (summary['sampled_rows'],summary['scored_rows'],summary['held_rows'])!=(len(results),len(scored),len(held)) or summary['hold_reasons']!=reasons:raise ValueError('range_control_summary_denominator_mismatch')
        for metric in ['log_loss','brier']:
            expected=sum(row['scenarios'][name][metric] for row in scored)/len(scored) if scored else None
            difference=sum(row['scenarios']['player_and_action_assumption'][metric]-row['scenarios'][name][metric] for row in scored)/len(scored) if scored else None
            for actual,wanted in [(summary[metric],expected),(summary['matched_full_minus_control'][metric],difference)]:
                if wanted is None and actual is not None or wanted is not None and (isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isclose(actual,wanted,rel_tol=1e-10,abs_tol=1e-10)):raise ValueError('range_control_summary_arithmetic_mismatch')


def safe_extract(path,directory,members):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(members):raise ValueError('archive_inventory_mismatch')
        for name,expected in members.items():
            safe_name(name)
            if archive.getinfo(name).file_size!=expected['bytes']:raise ValueError('archive_member_size_mismatch')
            destination=directory/name
            if destination.exists():verify(destination,expected);continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            partial=destination.with_name(destination.name+'.partial.'+uuid.uuid4().hex)
            with archive.open(name) as source,partial.open('xb') as target:shutil.copyfileobj(source,target,1024*1024)
            verify(partial,expected);os.rename(partial,destination)


def durable_copy(source,destination):
    source,destination=Path(source),Path(destination);expected=identity(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        partial=destination.with_name(destination.name+'.partial.'+uuid.uuid4().hex)
        with source.open('rb') as input_stream,partial.open('xb') as output_stream:shutil.copyfileobj(input_stream,output_stream,1024*1024)
        verify(partial,expected);os.rename(partial,destination)
    verify(destination,expected);verify(destination,expected)
    return dict(expected,raw_byte_readbacks=2)


def checkpoint(root,relative_files,drive_state,label,binding):
    root,drive_state=Path(root),Path(drive_state)
    manifest_path=drive_state/(label+'.checkpoint.json')
    if manifest_path.exists():
        previous=load(manifest_path)
        if previous['binding']!=binding:raise ValueError('checkpoint_binding_changed')
        verify(drive_state/previous['file_name'],previous['transport']);return
    paths=[root/safe_name(name) for name in relative_files]
    if any(not path.is_file() for path in paths):return
    members={name:identity(path) for name,path in zip(relative_files,paths)}
    local=root/(label+'.checkpoint.zip.partial.'+uuid.uuid4().hex)
    with zipfile.ZipFile(local,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,path in zip(relative_files,paths):archive.write(path,name)
    destination=drive_state/(label+'.checkpoint.zip')
    transport=durable_copy(local,destination)
    atomic_json(manifest_path,dict(binding=binding,file_name=destination.name,transport={k:transport[k] for k in ['bytes','sha256']},members=members,raw_byte_readbacks=2))
    local.unlink()


def restore_checkpoint(root,drive_state,label,binding):
    root,drive_state=Path(root),Path(drive_state);manifest_path=drive_state/(label+'.checkpoint.json')
    if not manifest_path.exists():return False
    manifest=load(manifest_path)
    if manifest['binding']!=binding:raise ValueError('checkpoint_binding_changed')
    source=drive_state/manifest['file_name'];verify(source,manifest['transport'])
    safe_extract(source,root,manifest['members']);return True


def execute_with_progress(command,publish=None):
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            print(line,end='',flush=True)
            if publish is not None:publish()
        status=process.wait()
    except BaseException:
        # Preserve the child and its computation on an interrupted display loop;
        # the notebook runner owns runtime/process intervention.
        raise
    if status:raise RuntimeError('child_failed:'+str(status))
    if publish is not None:publish()


def assemble_return_packet(result,sources,release_id,plan_sha256):
    """Publish seven closed members, verify them all, then seal the manifest."""
    result=Path(result);result.mkdir(parents=True,exist_ok=True)
    if set(sources)!=RESULT_NAMES-{'return_manifest.json'}:raise ValueError('result_assembly_inventory_mismatch')
    members={name:identity(source) for name,source in sources.items()}
    for name,source in sources.items():
        raw=Path(source).read_bytes()
        if dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())!=members[name]:raise ValueError('source_changed_during_result_assembly')
        publish_local_bytes(result/name,raw)
    # Abandoned UUID partials remain for inspection and never enter inventory.
    for name,source in sources.items():
        verify(source,members[name]);verify(result/name,members[name])
    manifest=dict(schema_version=SCHEMA+'_return',release_id=release_id,authority=AUTHORITY,plan_sha256=plan_sha256,members=members)
    publish_local_bytes(result/'return_manifest.json',(canonical(manifest)+'\n').encode())
    return manifest


def run_colab(root):
    if not Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'):raise ValueError('colab_execution_only')
    root=Path(root);plan=validate_plan(load(root/'release_plan.json'))
    if plan.get('authority')!=AUTHORITY or set(plan.get('code_files',{}))!=set(SOURCE_NAMES):raise ValueError('release_code_or_authority_contract_mismatch')
    for name,expected in plan['code_files'].items():verify(root/name,expected)
    stage=Path(plan['drive_stage_path']);work=root/'work';work.mkdir(exist_ok=True)
    durable=stage/'runtime_state'/plan['release_id'];binding=dict(plan_sha256=sha(root/'release_plan.json'),code_files=plan['code_files'])
    source_dir=work/'source';source_dir.mkdir(exist_ok=True)
    local_sources={};transport_cache={}
    for role,source in plan['inputs'].items():
        remote=stage/'inputs'/source['file_name']
        transport_identity={key:source[key] for key in ['bytes','sha256']}
        cache_key=source['file_name']
        if cache_key in transport_cache and transport_cache[cache_key]!=transport_identity:raise ValueError('conflicting_shared_transport_identity')
        if cache_key not in transport_cache:verify(remote,transport_identity);transport_cache[cache_key]=transport_identity
        destination=source_dir/(role+('.jsonl' if role=='hands' else '.csv'))
        expected=transport_identity if source.get('member') is None else dict(bytes=source['member_bytes'],sha256=source['member_sha256'])
        if destination.exists():verify(destination,expected)
        else:
            partial=destination.with_name(destination.name+'.partial.'+uuid.uuid4().hex)
            if source.get('member') is None:
                with remote.open('rb') as inp,partial.open('xb') as out:shutil.copyfileobj(inp,out,1024*1024)
            else:
                with zipfile.ZipFile(remote) as archive:
                    if archive.namelist().count(source['member'])!=1 or archive.getinfo(source['member']).file_size!=source['member_bytes']:raise ValueError('source_member_missing_duplicate_or_wrong_size')
                    with archive.open(source['member']) as inp,partial.open('xb') as out:shutil.copyfileobj(inp,out,1024*1024)
            verify(partial,expected);os.rename(partial,destination)
        local_sources[role]=expected
    atomic_json(work/'input_manifest.json',local_sources)
    restore_checkpoint(work,durable,'prepared_rows',binding)
    prepared=work/'prepared_rows'
    execute_with_progress([sys.executable,str(root/'stage6_empirical_range_learning_v2_adapter.py'),'--actions',str(source_dir/'actions.csv'),'--identities',str(source_dir/'identities.csv'),
        '--hands',str(source_dir/'hands.jsonl'),'--input-manifest',str(work/'input_manifest.json'),'--output',str(prepared)])
    checkpoint(work,['prepared_rows/range_training_rows.jsonl','prepared_rows/range_quarantine_rows.jsonl','prepared_rows/identity_mapping.json','prepared_rows/range_adapter_receipt.json'],durable,'prepared_rows',binding)
    protocol=resolve_protocol(plan,load(prepared/'range_adapter_receipt.json'));atomic_json(work/'protocol.json',protocol)
    fit=work/'fit'
    for label in ['compact_rows','physical_combo_action_likelihood','observed_next_action_size']:restore_checkpoint(work,durable,label,binding)
    published={label for label in ['compact_rows','physical_combo_action_likelihood','observed_next_action_size'] if (durable/(label+'.checkpoint.json')).exists()}
    def publish_fit():
        if 'compact_rows' not in published and (fit/'prepared/prepared_receipt.json').exists():
            receipt=load(fit/'prepared/prepared_receipt.json')
            if sha(fit/'prepared/rows.sqlite')==receipt['database_sha256']:
                checkpoint(work,['fit/prepared/rows.sqlite','fit/prepared/prepared_receipt.json'],durable,'compact_rows',binding)
                published.add('compact_rows')
        for role in ['physical_combo_action_likelihood','observed_next_action_size']:
            if role in published:continue
            path=fit/'checkpoints'/(role+'.json')
            if path.exists():
                try:closed=load(path)
                except (ValueError,json.JSONDecodeError):continue
                if closed['payload_sha256']==hashlib.sha256(canonical(closed['payload']).encode()).hexdigest():
                    checkpoint(work,['fit/checkpoints/'+role+'.json'],durable,role,binding)
                    published.add(role)
    execute_with_progress([sys.executable,str(root/'stage6_empirical_range_learning_v2.py'),'fit','--input',str(prepared/'range_training_rows.jsonl'),'--protocol',str(work/'protocol.json'),'--output',str(fit)],publish_fit)
    result=work/'result'
    sources={name:source for source,name in [(fit/'range_response_artifact.json','range_response_artifact.json'),(fit/'range_response_report.json','range_response_report.json'),
        (fit/'completion_receipt.json','completion_receipt.json'),(prepared/'range_adapter_receipt.json','range_adapter_receipt.json'),(prepared/'identity_mapping.json','identity_mapping.json'),(work/'protocol.json','protocol.json'),(root/'release_plan.json','release_plan.json')]}
    assemble_return_packet(result,sources,plan['release_id'],sha(root/'release_plan.json'))
    result_zip=work/('90_'+plan['release_id']+'_RESULTS.zip')
    if not result_zip.exists():
        temporary=result_zip.with_name(result_zip.name+'.partial.'+uuid.uuid4().hex)
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(RESULT_NAMES):archive.write(result/name,name)
        os.rename(temporary,result_zip)
    transport=durable_copy(result_zip,stage/'returned_results'/result_zip.name)
    atomic_json(stage/'returned_results'/(result_zip.name+'.receipt.json'),dict(release_id=plan['release_id'],file_name=result_zip.name,authority=AUTHORITY,transport=transport))
    print(canonical(dict(status='COMPLETE_EXPERIMENT',result_filename=result_zip.name,transport=transport,authority=AUTHORITY)),flush=True)


BOOK='03_Stage7_Empirical_Range_Response_V2_r1.ipynb'

def build(plan_path,output):
    plan=validate_plan(load(plan_path));output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('release_output_must_be_new_empty_directory')
    output.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).parent
    plan['code_files']={name:identity(source_root/name) for name in SOURCE_NAMES}
    plan['authority']=AUTHORITY
    atomic_json(output/'release_plan.json',plan)
    package=output/('stage6_empirical_range_learning_v2_'+plan['release_id']+'.zip')
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in SOURCE_NAMES:archive.write(source_root/name,name)
        archive.write(output/'release_plan.json','release_plan.json')
    package_identity=identity(package)
    code_members=dict(plan['code_files'],**{'release_plan.json':identity(output/'release_plan.json')})
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),colab=dict(name=BOOK)),cells=[])
    def cell(kind,text):notebook['cells'].append(dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),**(dict(execution_count=None,outputs=[]) if kind=='code' else {})))
    cell('markdown','# 90 â€” Empirical range and response experiment\n\nUse a **CPU, High-RAM** Colab runtime. This reuses the selected LightGBM family for new explicit targets; it does not rerun model-family search or solver work. Reveal-selected range assumptions and observed response predictions remain diagnostics with zero strategy authority. Whole-game membership and source hashes are frozen in this package. No PokerNow actions. Run top-to-bottom.\n')
    cell('code','import os, sys, subprocess, pathlib\nassert os.path.isdir("/content") and os.environ.get("COLAB_RELEASE_TAG"), "Colab runtime required"\nassert sys.version_info[:2] == (3, 12), "Select the frozen Python 3.12 runtime before mounting Drive"\nmeminfo = dict(line.split(":",1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())\nactual_ram_bytes = int(meminfo["MemTotal"].split()[0]) * 1024\nassert actual_ram_bytes >= 48 * 1000**3, "Select CPU High-RAM (at least 48 GB) before mounting Drive"\nassert os.environ.get("COLAB_GPU", "0") in ("", "0"), "Select the frozen CPU runtime"\nprint({"runtime_python":sys.version,"actual_ram_bytes":actual_ram_bytes,"actual_ram_gb":actual_ram_bytes/1e9,"actual_ram_gib":actual_ram_bytes/1024**3,"ram_role":"frozen runtime shape prerequisite; not a worker cap"})\nfrom google.colab import drive\ndrive.mount("/content/drive")\nsubprocess.check_call([sys.executable,"-m","pip","install","--quiet","numpy==2.2.6","scikit-learn==1.7.2","lightgbm==4.6.0","threadpoolctl==3.6.0"])\n')
    bootstrap=f'''import hashlib, json, pathlib, zipfile, shutil, uuid\nSTAGE = pathlib.Path({plan['drive_stage_path']!r})\nPACKAGE = STAGE / "inputs" / {package.name!r}\nROOT = pathlib.Path("/content") / {('range90_'+plan['release_id'])!r}\nROOT.mkdir(exist_ok=True)\ndef digest(path):\n    h=hashlib.sha256()\n    with path.open("rb") as stream:\n        for part in iter(lambda:stream.read(1048576),b""): h.update(part)\n    return h.hexdigest()\nassert PACKAGE.stat().st_size == {package_identity['bytes']} and digest(PACKAGE) == {package_identity['sha256']!r}\nMEMBERS = json.loads({canonical(code_members)!r})\nwith zipfile.ZipFile(PACKAGE) as archive:\n    assert len(archive.namelist())==len(set(archive.namelist())) and set(archive.namelist())==set(MEMBERS)\n    for name, expected in MEMBERS.items():\n        assert pathlib.PurePosixPath(name).name==name\n        raw=archive.read(name)\n        assert len(raw)==expected["bytes"] and hashlib.sha256(raw).hexdigest()==expected["sha256"]\n        target=ROOT/name\n        if target.exists(): assert target.read_bytes()==raw\n        else:\n            partial=target.with_name(target.name+".partial."+uuid.uuid4().hex)\n            with partial.open("xb") as stream:\n                stream.write(raw); stream.flush(); os.fsync(stream.fileno())\n            assert partial.stat().st_size==expected["bytes"] and digest(partial)==expected["sha256"]\n            try: os.link(partial,target)\n            except FileExistsError: assert target.read_bytes()==raw\n            partial.unlink()\nprint("Authenticated release and source code", ROOT)\n'''
    cell('code',bootstrap)
    cell('markdown','The next cell streams exact whole-game input conversion, preserves quarantines, resolves only predeclared zero-row exclusions, builds disk-backed features, fits/calibrates both roles and publishes closed checkpoints. Progress prints at least every 20 seconds during long fits. RAM is telemetry only. Completed preparations and role checkpoints resume by exact lineage; changed bytes require a new release.\n')
    cell('code','subprocess.check_call([sys.executable,str(ROOT/"stage6_empirical_range_learning_v2_release.py"),"run-colab","--root",str(ROOT)])\n')
    cell('markdown','The direct result ZIP and publication receipt are in Stage 6 `returned_results`. Two raw-byte Drive readbacks are required by the producer. The runner retrieves the exact Drive object and the implementation task performs strict import and independent review. A completed notebook does not promote any range, strategy or model.\n')
    for index,cell_doc in enumerate(notebook['cells']):
        cell_doc['id']=f'cell-{index:02}'
        if cell_doc['cell_type']=='markdown':
            cell_doc['source']=''.join(cell_doc['source']).replace('# 90 ', '# 03 — Stage 7 ').replace('Stage 6 `returned_results`','Stage 7 `returned_results`').splitlines(keepends=True)
    atomic_json(output/BOOK,notebook)
    release=dict(schema_version=SCHEMA+'_release_manifest',release_id=plan['release_id'],plan_sha256=sha(output/'release_plan.json'),authority=AUTHORITY,
        code_package=dict(file_name=package.name,**package_identity,members=code_members),notebook=dict(file_name=BOOK,**identity(output/BOOK)))
    atomic_json(output/'range_learning_v2_release_manifest.json',release)
    return release


def strict_import(result_zip,expected_sha,release_manifest,destination):
    result_zip=Path(result_zip);release=load(release_manifest);destination=Path(destination)
    if release.get('schema_version')!=SCHEMA+'_release_manifest' or release.get('authority')!=AUTHORITY:raise ValueError('local_release_contract_mismatch')
    if sha(result_zip)!=expected_sha or not re.fullmatch('[a-f0-9]{64}',expected_sha):raise ValueError('returned_transport_identity_mismatch')
    with zipfile.ZipFile(result_zip) as archive:
        if len(archive.namelist())!=len(RESULT_NAMES) or set(archive.namelist())!=RESULT_NAMES:raise ValueError('unexpected_return_members')
        if any(info.file_size>128*1024*1024 for info in archive.infolist()) or sum(info.file_size for info in archive.infolist())>384*1024*1024:raise ValueError('return_size_limit')
        values={name:archive.read(name) for name in RESULT_NAMES}
    objects={name:json.loads(value,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite_return_json'))) for name,value in values.items()}
    returned,plan,artifact,completion,adapter,protocol=[objects[n] for n in ['return_manifest.json','release_plan.json','range_response_artifact.json','completion_receipt.json','range_adapter_receipt.json','protocol.json']]
    if returned['schema_version']!=SCHEMA+'_return' or returned['release_id']!=release['release_id'] or returned['authority']!=AUTHORITY:raise ValueError('return_contract_mismatch')
    if hashlib.sha256(values['release_plan.json']).hexdigest()!=release['plan_sha256'] or returned['plan_sha256']!=release['plan_sha256']:raise ValueError('return_plan_identity_mismatch')
    if set(returned['members'])!=RESULT_NAMES-{'return_manifest.json'}:raise ValueError('return_manifest_inventory_mismatch')
    for name,expected in returned['members'].items():
        if dict(bytes=len(values[name]),sha256=hashlib.sha256(values[name]).hexdigest())!=expected:raise ValueError('return_member_identity_mismatch')
    validate_plan(plan)
    expected_inputs={role:({key:source[key] for key in ['bytes','sha256']} if source.get('member') is None else dict(bytes=source['member_bytes'],sha256=source['member_sha256'])) for role,source in plan['inputs'].items()}
    expected_input_manifest_hash=hashlib.sha256((canonical(expected_inputs)+'\n').encode()).hexdigest()
    adapter_lineage=adapter.get('lineage',{})
    if adapter_lineage.get('source_inputs')!=expected_inputs or adapter_lineage.get('input_manifest_sha256')!=expected_input_manifest_hash or adapter_lineage.get('adapter_sha256')!=plan['code_files']['stage6_empirical_range_learning_v2_adapter.py']['sha256'] or adapter_lineage.get('synthetic_fixture') is not False:
        raise ValueError('adapter_direct_source_lineage_mismatch')
    mapping=objects['identity_mapping.json']
    if mapping.get('schema_version')!='stage6_empirical_exact_identity_map_v2' or mapping.get('source_manifest_sha256')!=expected_input_manifest_hash or returned['members']['identity_mapping.json']!=adapter['members'].get('identity_mapping.json'):
        raise ValueError('exact_identity_mapping_lineage_mismatch')
    if not all(isinstance(k,str) and k and isinstance(v,str) and v for section in ['games','players'] for k,v in mapping.get(section,{}).items()):raise ValueError('invalid_exact_identity_mapping')
    if protocol!=resolve_protocol(plan,adapter):raise ValueError('cohort_or_protocol_changed_after_freeze')
    if artifact.get('lineage',{}).get('synthetic_fixture') is not False or objects['range_response_report.json'].get('authority')!=AUTHORITY or artifact.get('authority')!=AUTHORITY or completion.get('authority')!=AUTHORITY or adapter.get('authority')!=AUTHORITY:raise ValueError('synthetic_or_unauthorized_return')
    if artifact.get('schema_version')!=SCHEMA or objects['range_response_report.json'].get('schema_version')!=SCHEMA or adapter.get('schema_version')!=SCHEMA+'_adapter':raise ValueError('returned_artifact_schema_mismatch')
    if artifact.get('model_family')!='LightGBM_4.6.0_no_family_search' or set(artifact.get('models',{}))-{'physical_combo_action_likelihood','observed_next_action_size'}:raise ValueError('unexpected_model_role_or_family')
    if artifact.get('continuation_value',{}).get('supports_unchosen_action_ev') is not False or artifact.get('continuation_value',{}).get('status')!='NOT_IDENTIFIED':raise ValueError('unsupported_continuation_authority')
    expected_lineage=dict(input_sha256=protocol['input_sha256'],protocol_sha256=hashlib.sha256(values['protocol.json']).hexdigest(),source_code_sha256=protocol['source_code_sha256'],streaming_code_sha256=protocol['streaming_code_sha256'],synthetic_fixture=False)
    if artifact['lineage']!=expected_lineage or completion['lineage']!=expected_lineage or objects['range_response_report.json']['lineage']!=expected_lineage:raise ValueError('model_lineage_mismatch')
    if artifact['trained_game_ids']!=protocol['train_game_ids'] or artifact['calibration_game_ids']!=protocol['calibration_game_ids']:raise ValueError('artifact_fitting_membership_mismatch')
    if artifact['trained_through_utc']!=max(adapter['game_bounds'][g][1] for g in protocol['train_game_ids']) or artifact['available_after_utc']!=max(adapter['game_bounds'][g][1] for g in protocol['train_game_ids']+protocol['calibration_game_ids']):raise ValueError('artifact_availability_time_mismatch')
    if completion['schema_version']!=SCHEMA or completion['status']!='COMPLETE_EXPERIMENT' or set(completion['members'])!={'range_response_artifact.json','range_response_report.json'}:raise ValueError('completion_contract_mismatch')
    fit_provenance=completion.get('execution_provenance',{})
    if fit_provenance.get('available_after_semantics')!='source_and_calibration_cutoff' or fit_provenance.get('historical_deployment_evidence') is not False:
        raise ValueError('source_cutoff_or_execution_provenance_mismatch')
    if not utc(artifact['available_after_utc'])<=utc(fit_provenance['fit_invocation_started_utc'])<=utc(fit_provenance['fit_completed_utc']):raise ValueError('invalid_fit_wall_clock_provenance')
    for name,expected in completion['members'].items():
        if expected!=returned['members'][name]:raise ValueError('completion_member_identity_mismatch')
    verify_report_arithmetic(objects['range_response_report.json'],artifact,protocol)
    verify_range_controls(objects['range_response_report.json'],artifact,protocol,adapter)
    receipt=dict(schema_version=SCHEMA+'_strict_import',status='ACCEPT_DIAGNOSTIC_ARTIFACT',release_id=release['release_id'],authority=AUTHORITY,
        result_zip_sha256=expected_sha,release_manifest_sha256=sha(release_manifest),members={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in values.items()},scientific_recomputation=False,runtime_registry_written=False,paired_score_arithmetic_verified=True,score_arithmetic_does_not_authenticate_observational_truth=True)
    receipt_path=destination/'strict_import_receipt.json'
    imported_utc=load(receipt_path).get('execution_provenance',{}).get('imported_utc') if receipt_path.exists() else datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')
    if utc(imported_utc)<utc(fit_provenance['fit_completed_utc']):raise ValueError('import_precedes_fit_completion')
    receipt['execution_provenance']=dict(imported_utc=imported_utc,source_fit_completed_utc=fit_provenance['fit_completed_utc'],historical_deployment_evidence=False)
    permitted=RESULT_NAMES|{'strict_import_receipt.json'}
    if destination.exists():
        for path in destination.iterdir():
            if path.name in permitted:
                if not path.is_file():raise ValueError('invalid_import_destination_member')
            elif not any(re.fullmatch(re.escape(name)+r'\.partial\.[a-f0-9]{32}',path.name) for name in permitted):
                raise ValueError('unrelated_import_destination_member')
        for name,expected in receipt['members'].items():
            if (destination/name).exists():verify(destination/name,expected)
    reused=receipt_path.exists()
    if reused and load(receipt_path)!=receipt:raise ValueError('completed_import_receipt_identity_mismatch')
    if reused and any(not (destination/name).is_file() for name in RESULT_NAMES):raise ValueError('completed_import_member_missing')
    destination.mkdir(parents=True,exist_ok=True)
    for name in sorted(RESULT_NAMES):publish_local_bytes(destination/name,values[name])
    publish_local_bytes(receipt_path,(canonical(receipt)+'\n').encode())
    return dict(receipt,reused=reused)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['build','run-colab','import'])
    parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--root',type=Path)
    parser.add_argument('--result',type=Path);parser.add_argument('--result-sha256');parser.add_argument('--release-manifest',type=Path)
    args=parser.parse_args()
    if args.mode=='build':print(canonical(build(args.plan,args.output)),flush=True)
    elif args.mode=='run-colab':run_colab(args.root)
    else:print(canonical(strict_import(args.result,args.result_sha256,args.release_manifest,args.output)),flush=True)


if __name__=='__main__':main()
