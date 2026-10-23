"""Strict review-only import of a frozen capture comparison; never installs models."""
from __future__ import annotations
import argparse
import hashlib
import json
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import stage6_empirical_capture_v2 as capture

RETURNED_PARENT = '1zlvQjuq_MX_Omi1tg-q9C53kN7VDrB43'


def sealed(value):
    value = dict(value); expected = value.pop('self_sha256')
    if hashlib.sha256(capture.canonical(value)).hexdigest() != expected:
        raise ValueError('self_seal_mismatch')
    return value


def identity(path):
    p=Path(path);return {'bytes':p.stat().st_size,'sha256':capture.sha(p)}


def check_readback(receipt, result, *, expected_parent=RETURNED_PARENT):
    doc=json.loads(Path(receipt).read_text(encoding='utf-8'))
    if doc.get('schema_version')!='stage6_capture_drive_readback_v2' or not doc.get('file_id'):
        raise ValueError('external_two_pass_readback_required')
    passes=doc.get('passes',[])
    if len(passes)!=2 or [p.get('pass') for p in passes]!=[1,2]:raise ValueError('exactly_two_readback_passes_required')
    expected=identity(result);metadata=[];raw_paths=[]
    for entry in passes:
        meta=entry['metadata'];raw=Path(entry['local_raw_path']).resolve()
        if meta.get('id')!=doc['file_id'] or expected_parent not in meta.get('parents',[]) or int(meta.get('size',-1))!=expected['bytes']:
            raise ValueError('readback_metadata_identity_mismatch')
        if not meta.get('version') or not meta.get('modifiedTime') or not entry.get('downloaded_utc'):
            raise ValueError('readback_metadata_provenance_missing')
        if identity(raw)!=expected or {k:entry[k] for k in ('bytes','sha256')}!=expected:
            raise ValueError('readback_raw_bytes_mismatch')
        metadata.append({k:meta[k] for k in ('id','parents','size','version','modifiedTime')});raw_paths.append(raw)
    if metadata[0]!=metadata[1] or raw_paths[0]==raw_paths[1]:raise ValueError('independent_stable_readbacks_required')
    if passes[0]['downloaded_utc']==passes[1]['downloaded_utc']:raise ValueError('distinct_readback_observations_required')
    return {'file_id':doc['file_id'],'parent_id':expected_parent,'sha256':expected['sha256'],'bytes':expected['bytes'],'passes':2,'receipt_sha256':capture.sha(receipt)}


def authenticate_zip(archive, expected, manifest_name, expected_schema):
    if identity(archive)!=expected:raise ValueError('archive_identity_mismatch')
    bundle=zipfile.ZipFile(archive)
    names=bundle.namelist()
    if len(names)!=len(set(names)) or not names or names[-1]!=manifest_name:
        bundle.close();raise ValueError('archive_duplicate_or_not_manifest_last')
    manifest=json.loads(bundle.read(manifest_name))
    if manifest.get('schema_version')!=expected_schema or set(names)!=set(manifest.get('members',{}))|{manifest_name}:
        bundle.close();raise ValueError('archive_schema_or_inventory_mismatch')
    for name,item in manifest['members'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            bundle.close();raise ValueError('archive_unsafe_path')
        info=bundle.getinfo(name)
        if ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('archive_symlink')
        with bundle.open(name) as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if info.file_size!=item['bytes'] or actual!=item['sha256']:raise ValueError('archive_member_changed:'+name)
    return bundle,manifest


def run_identity(protocol):
    identities=protocol['code_files']+[protocol['session_parser']]
    identities += [f for g in protocol['games'] for f in g['files']]
    identities += [s['archive'] for s in protocol['snapshots'].values()]
    identities += [f for s in protocol['snapshots'].values() for f in s.get('external_members',[])]
    return hashlib.sha256(capture.canonical({'protocol_sha256':hashlib.sha256(capture.canonical(protocol)).hexdigest(),'inputs':identities})).hexdigest()


def jsonl(raw):
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]


def validate_result_members(bundle, manifest, protocol, package_sha256):
    expected_protocol=hashlib.sha256(capture.canonical(protocol)).hexdigest()
    if manifest.get('protocol_sha256')!=expected_protocol or manifest.get('package_sha256')!=package_sha256 or manifest.get('promotion_allowed') is not False:
        raise ValueError('result_frozen_identity_or_authority_mismatch')
    expected_run=run_identity(protocol)
    binding=json.loads(bundle.read('run_binding.json'))
    if binding!={'run_identity':expected_run,'protocol_sha256':expected_protocol}:raise ValueError('result_run_binding_mismatch')
    expected_names={'run_binding.json','capture_comparison_report.json'};phase_files={};output_files={}
    for phase in ('prepare',*capture.ROLES):
        name=phase+'_receipt.json';expected_names.add(name)
        receipt=sealed(json.loads(bundle.read(name)))
        if receipt.get('schema_version')!='stage6_empirical_capture_phase_v2' or receipt.get('phase')!=phase or receipt.get('run_identity')!=expected_run:
            raise ValueError('phase_binding_mismatch')
        files=receipt.get('files',[])
        required={'prepared_capture_records.jsonl','preparation.json'} if phase=='prepare' else {phase+'_replay.jsonl'}
        if {item['name'] for item in files}!=required or len(files)!=len(required):raise ValueError('phase_exact_outputs_required')
        for item in files:
            p=item['path']
            if not p.startswith('phases/'+phase+'/attempt_') or Path(p).name!=item['name']:raise ValueError('phase_output_path_mismatch')
            if {k:item[k] for k in ('bytes','sha256')}!=manifest['members'].get(p):raise ValueError('phase_member_identity_mismatch')
            expected_names.add(p);phase_files[item['name']]=p;output_files[p]={k:item[k] for k in ('bytes','sha256')}
    if set(manifest['members'])!=expected_names:raise ValueError('unexpected_unaccepted_result_members')
    records=jsonl(bundle.read(phase_files['prepared_capture_records.jsonl']))
    record_ids={r['decision_id'] for r in records}
    if len(record_ids)!=len(records):raise ValueError('duplicate_prepared_decision')
    cohorts={g['game_id']:g['cohort'] for g in protocol['games']}
    capture_inputs=capture.capture_input_owners(protocol)
    for r in records:
        if r.get('schema_version')!='stage6_empirical_capture_record_v2' or r.get('cohort')!=cohorts.get(r['game_id']):raise ValueError('record_cohort_or_schema_mismatch')
        if r['decision_id']!='|'.join([r['game_id'],r['hand_id'],r['action_order']]):raise ValueError('record_decision_identity_mismatch')
        dates=[datetime.fromisoformat(r[k].replace('Z','+00:00')) for k in ('capture_utc','action_utc','hand_completed_utc')]
        if not dates[0]<dates[1]<=dates[2]:raise ValueError('record_noncausal_timestamps')
        if capture_inputs.get(r['source_capture']['path'])!=r['game_id'] or r['capture'].get('pokernow_game_id')!=r['game_id']:raise ValueError('record_source_identity_mismatch')
        if r['outcome_identification'].get('counterfactual_ev_identified') is not False or r['outcome_identification'].get('human_behavior_propensity') is not None:raise ValueError('unsupported_counterfactual_claim')
        if not all(r['validation'].get(k) is True for k in ('exact_prefix','exact_board','complete_hand','next_action_exact_hero')):raise ValueError('record_validation_not_passed')
        for e in r.get('decision_public_ledger',{}).get('events',[]):
            if e.get('at') and datetime.fromisoformat(e['at'].replace('Z','+00:00'))>dates[0]:raise ValueError('future_ledger_event')
    rows={role:jsonl(bundle.read(phase_files[role+'_replay.jsonl'])) for role in capture.ROLES}
    by_id={r['decision_id']:r for r in records}
    for role,items in rows.items():
        if {r['decision_id'] for r in items}!=record_ids:raise ValueError('replay_not_exact_prepared_cohort')
        for item in items:
            if item.get('role')!=role:raise ValueError('replay_role_mismatch')
            record=by_id[item['decision_id']]
            if item.get('game_id')!=record['game_id']:raise ValueError('replay_game_mismatch')
            if item.get('status')=='REPLAYED':
                if any(item.get(k)!=record[k] for k in ('hand_id','cohort','observed_action')) or item.get('street')!=record['capture'].get('street'):
                    raise ValueError('replay_label_or_context_mismatch')
                if item.get('counterfactual_ev_identified') is not False:raise ValueError('replay_counterfactual_claim')
    report=sealed(json.loads(bundle.read('capture_comparison_report.json')))
    recomputed=capture.summarize(rows)
    for key,value in recomputed.items():
        if report.get(key)!=value:raise ValueError('recomputed_summary_mismatch:'+key)
    metadata={'schema_version','protocol_sha256','created_utc','snapshots','source_bytes_stable','workers','files'}
    if set(report)!=set(recomputed)|metadata:raise ValueError('unexpected_report_fields')
    if report['schema_version']!='stage6_empirical_capture_comparison_v2' or report['protocol_sha256']!=expected_protocol or report['snapshots']!=protocol['snapshots'] or report['source_bytes_stable'] is not True or report['files']!=output_files:
        raise ValueError('report_provenance_mismatch')
    workers=report['workers']
    if workers.get('ram_used_as_gate') is not False or not 1<=workers.get('resolved',0)<=2 or workers['resolved']>workers.get('cpu_cap',0):raise ValueError('worker_contract_mismatch')
    preparation=json.loads(bundle.read(phase_files['preparation.json']))
    if preparation.get('source_bytes_stable') is not True or preparation.get('decisions')!=len(records) or preparation.get('qualification') is not False or preparation.get('live_action_authority') is not False:
        raise ValueError('preparation_receipt_mismatch')
    if {g['game_id']:g['cohort'] for g in preparation['games']}!=cohorts:raise ValueError('preparation_games_mismatch')
    return {'disposition':recomputed['disposition'],'records':len(records),'roles':recomputed['roles'],
            'cohorts':recomputed['cohorts'],'qualification':recomputed['qualification'],'recomputed_summary_sha256':hashlib.sha256(capture.canonical(recomputed)).hexdigest()}


def import_result(release_path, release_sha256, package_path, result_path, result_sha256, readback_path, destination):
    if capture.sha(release_path)!=release_sha256:raise ValueError('release_sha256_mismatch')
    release=sealed(json.loads(Path(release_path).read_text(encoding='utf-8')))
    if release.get('schema_version')!='stage6_capture_release_v2' or release.get('promotion_allowed') is not False:raise ValueError('release_schema_or_authority_mismatch')
    transport=check_readback(readback_path,result_path)
    if transport['sha256']!=result_sha256:raise ValueError('explicit_result_sha256_mismatch')
    package,package_manifest=authenticate_zip(package_path,{k:release['package'][k] for k in ('bytes','sha256')},'package_manifest.json','stage6_capture_package_v2')
    try:
        if package_manifest!=release['package_manifest']:raise ValueError('package_manifest_differs_from_release')
        protocol_raw=package.read('protocol.json');protocol=json.loads(protocol_raw)
        if protocol!=release['protocol'] or hashlib.sha256(protocol_raw).hexdigest()!=release['protocol_sha256'] or protocol_raw!=capture.canonical(protocol):raise ValueError('protocol_mismatch_or_noncanonical')
        capture.validate_protocol(protocol)
        for path in (Path(capture.__file__).resolve(),Path(__file__).resolve()):
            name='src/'+path.name
            if identity(path)!=package_manifest['members'].get(name):raise ValueError('review_code_differs_from_frozen_release')
    finally:package.close()
    result,result_manifest=authenticate_zip(result_path,{'bytes':transport['bytes'],'sha256':result_sha256},'result_manifest.json','stage6_capture_result_bundle_v2')
    try:recomputed=validate_result_members(result,result_manifest,protocol,release['package']['sha256'])
    finally:result.close()
    destination=Path(destination)
    if destination.exists():raise FileExistsError('review_import_never_overwrites')
    destination.mkdir(parents=True)
    receipt={'schema_version':'stage6_capture_import_review_v2','accepted_for_review':True,'source_release_sha256':release_sha256,'result_sha256':result_sha256,
             'transport':transport,'recomputed':recomputed,'model_installation':False,'prediction_qualification':False,'strategy_qualification':False,'promotion_allowed':False,'live_action_authority':False}
    receipt['self_sha256']=hashlib.sha256(capture.canonical(receipt)).hexdigest();capture.write_json(destination/'import_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','package','result','readback-receipt','destination'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('release-sha256','result-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();stop=threading.Event();started=time.monotonic()
    def heartbeat():
        while not stop.wait(20):print(json.dumps({'stage':'capture_import_authentication_and_review','elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    t=threading.Thread(target=heartbeat,daemon=True);t.start()
    try:print(json.dumps(import_result(a.release,a.release_sha256,a.package,a.result,a.result_sha256,a.readback_receipt,a.destination),indent=2),flush=True)
    finally:stop.set();t.join()


if __name__=='__main__':main()
