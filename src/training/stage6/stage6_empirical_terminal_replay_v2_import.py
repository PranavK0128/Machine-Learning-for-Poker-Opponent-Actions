"""Strict Notebook 91 result review: exact bytes/receipts, recomputed summary, no install."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import threading
import time
import zipfile
import stage6_empirical_terminal_replay_v2 as core

RETURNED_PARENT='1zlvQjuq_MX_Omi1tg-q9C53kN7VDrB43'

def readback(receipt_path,result_path):
    doc=core.load(receipt_path);expected=core.identity(result_path)
    core.expect(doc.get('schema_version')==core.SCHEMA+'_drive_readback' and doc.get('file_id'),'exact_external_readback_required')
    passes=doc.get('passes',[]);core.expect(len(passes)==2 and [p.get('pass') for p in passes]==[1,2],'two_external_readbacks_required')
    metadata=[];paths=[];times=[]
    for p in passes:
        m=p['metadata'];core.expect(m.get('id')==doc['file_id'] and RETURNED_PARENT in m.get('parents',[]) and int(m.get('size',-1))==expected['bytes'],'readback_metadata_identity_mismatch')
        core.expect(m.get('version') and m.get('modifiedTime') and p.get('downloaded_utc'),'readback_provenance_missing')
        path=Path(p['local_raw_path']).resolve();core.verify(path,expected);core.expect({k:p[k] for k in ('bytes','sha256')}==expected,'readback_return_hash_mismatch')
        metadata.append({k:m[k] for k in ('id','parents','size','version','modifiedTime')});paths.append(path);times.append(p['downloaded_utc'])
    core.expect(metadata[0]==metadata[1] and paths[0]!=paths[1] and times[0]!=times[1],'independent_stable_readbacks_required')
    return dict(file_id=doc['file_id'],parent_id=RETURNED_PARENT,passes=2,receipt_sha256=core.sha(receipt_path),**expected)

def validate_result(result_root,result_manifest,upstream_records,binding_sha,release_sha,package_sha):
    root=Path(result_root);run=core.load(root/'run_binding.json');binding=core.load(root/'execution_binding.json')
    core.expect(core.sha(root/'execution_binding.json')==binding_sha,'returned_execution_binding_changed')
    core.validate_binding(binding,release_sha,package_sha)
    run_id=core.digest(dict(binding_sha256=binding_sha,release_sha256=release_sha))
    core.expect(run==dict(schema_version=core.SCHEMA+'_run',run_id=run_id,binding_sha256=binding_sha,release_sha256=release_sha,package_sha256=package_sha,authority=core.AUTHORITY),'run_binding_changed')
    core.expect(result_manifest.get('schema_version')==core.SCHEMA+'_result' and result_manifest.get('run_id')==run_id and result_manifest.get('authority')==core.AUTHORITY,'result_manifest_authority_or_run_changed')
    completion=core.unseal(core.load(root/'completion_receipt.json'))
    core.expect(completion.get('schema_version')==core.SCHEMA+'_completion' and completion.get('run_id')==run_id and completion.get('authority')==core.AUTHORITY,'completion_binding_mismatch')
    members=result_manifest['members'];core.expect(completion['files']=={k:v for k,v in members.items() if k!='completion_receipt.json'},'completion_exact_member_inventory_mismatch')
    expected={'run_binding.json','execution_binding.json','upstream_acceptance.json','terminal_report.json','completion_receipt.json'};rows=[]
    games=sorted({r['game_id'] for r in upstream_records})
    for game in games:
        import hashlib
        key=hashlib.sha256(game.encode()).hexdigest()[:20];receipt_name='game_'+key+'_receipt.json';expected.add(receipt_name)
        phase=core.unseal(core.load(root/receipt_name));core.expect(phase.get('schema_version')==core.SCHEMA+'_phase' and phase.get('run_id')==run_id and phase.get('game_id')==game,'phase_receipt_binding_mismatch')
        files=phase['files'];core.expect(len(files)==2 and {f['name'] for f in files}=={'terminal_rows.jsonl','game_receipt.json'},'phase_exact_files_required')
        for item in files:
            name=item['path'];core.safe_name(name);core.expect(name.startswith('phases/'+key+'/attempt_') and Path(name).name==item['name'] and {k:item[k] for k in ('bytes','sha256')}==members.get(name),'phase_member_binding_mismatch');expected.add(name)
            core.verify(root/name,item)
            if item['name']=='terminal_rows.jsonl':
                group=core.jsonl(root/name);core.expect(all(r.get('game_id')==game for r in group),'cross_game_phase_row');rows.extend(group)
            else:
                game_receipt=core.load(root/name);core.expect(game_receipt.get('schema_version')==core.SCHEMA+'_game' and game_receipt.get('game_id')==game and game_receipt.get('authority')==core.AUTHORITY and game_receipt.get('records')==sum(r['game_id']==game for r in upstream_records),'game_receipt_mismatch')
    core.expect(set(members)==expected,'unexpected_result_member_or_missing_phase')
    report=core.load(root/'terminal_report.json');recomputed=core.summarize(rows,upstream_records)
    core.expect(set(report)==set(recomputed)|{'resources','run_id'} and all(report.get(k)==v for k,v in recomputed.items()) and report['run_id']==run_id,'recomputed_terminal_report_mismatch')
    resources=report['resources'];core.expect(resources.get('inner_threads')==1 and resources.get('ram_used_as_worker_gate') is False and 1<=resources.get('workers',0)<=min(2,resources.get('cpu_capacity',0)),'worker_contract_mismatch')
    return recomputed

def strict_import(binding_path,binding_sha,release_path,release_sha,package_path,result_path,result_sha,readback_path,input_root,work,destination):
    core.expect(core.sha(binding_path)==binding_sha and core.sha(release_path)==release_sha and core.sha(result_path)==result_sha,'explicit_import_identity_mismatch')
    release=core.load(release_path);core.expect(release.get('schema_version')==core.SCHEMA+'_release' and release.get('authority')==core.AUTHORITY,'release_schema_or_authority_mismatch')
    core.verify(package_path,release['package']);binding=core.load(binding_path);core.validate_binding(binding,release_sha,release['package']['sha256'])
    for file in (Path(core.__file__),Path(__file__)):
        core.expect(core.identity(file)==release['members'].get('src/'+file.name),'review_code_differs_from_frozen_release')
    transport=readback(readback_path,result_path);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    core.extract_exact(package_path,work/'code91',release['members'],'package_manifest.json')
    upstream=core.authenticate_upstreams(binding,input_root,work)
    with zipfile.ZipFile(result_path) as z:manifest=json.loads(z.read('result_manifest.json'))
    result_root=core.extract_exact(result_path,work/'terminal91_result',manifest['members'],'result_manifest.json')
    core.expect((result_root/'execution_binding.json').read_bytes()==Path(binding_path).read_bytes(),'result_used_other_execution_binding')
    core.expect(core.load(result_root/'upstream_acceptance.json')==upstream['receipt'],'upstream_acceptance_changed')
    report=validate_result(result_root,manifest,core.jsonl(upstream['records']),binding_sha,release_sha,release['package']['sha256'])
    destination=Path(destination);core.expect(not destination.exists(),'strict_import_never_overwrites');destination.mkdir(parents=True)
    receipt=core.seal(dict(schema_version=core.SCHEMA+'_strict_import',status='ACCEPT_FOR_DIAGNOSTIC_REVIEW',binding_sha256=binding_sha,release_sha256=release_sha,result_sha256=result_sha,
        transport=transport,report_sha256=core.digest(report),supported=report['supported'],held=report['held'],records=report['records'],authority=core.AUTHORITY,model_installation=False,scientific_recomputation=False))
    core.write(destination/'recomputed_terminal_report.json',report);core.write(destination/'strict_import_receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','release','package','result','readback','input-root','work','destination'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('binding-sha256','release-sha256','result-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();stop=threading.Event();started=time.monotonic()
    def heartbeat():
        while not stop.wait(20):print(json.dumps(dict(phase='terminal_strict_import',elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    t=threading.Thread(target=heartbeat,daemon=True);t.start()
    try:print(json.dumps(strict_import(a.binding,a.binding_sha256,a.release,a.release_sha256,a.package,a.result,a.result_sha256,a.readback,a.input_root,a.work,a.destination),indent=2),flush=True)
    finally:stop.set();t.join()
