"""Notebook 91 immutable upstream binding, causal source audit and terminal replay."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

SCHEMA='stage6_empirical_terminal_replay_v2'
AUTHORITY=dict(strategy_weight=0,strategy_qualified=False,promotion_allowed=False,counterfactual_identification=False,solver_value=False,as_of_decision_deployment_test=False)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def load(path):return json.loads(Path(path).read_bytes(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite_json')))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def identity(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def expect(ok,reason):
    if not ok:raise ValueError(reason)
def safe_name(name):
    p=PurePosixPath(name)
    expect(isinstance(name,str) and name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and all(x!='.env' and not x.startswith('.env.') for x in p.parts),'unsafe_member_path')
    return name
def resolve(root,name):
    safe_name(name);root=Path(root).resolve();p=(root/name).resolve();expect(p.is_relative_to(root) and p!=root,'member_outside_root');return p
def verify(path,expected):
    expect(identity(path)=={k:expected[k] for k in ('bytes','sha256')},'raw_identity_mismatch:'+str(path));return Path(path)
def atomic_bytes(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():expect(path.read_bytes()==raw,'sealed_output_changed:'+str(path));return
    partial=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with partial.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if path.exists():expect(path.read_bytes()==raw,'concurrent_sealed_output_changed');return
    partial.rename(path)
def write(path,value):atomic_bytes(path,canonical(value)+b'\n')
def seal(value):return dict(value,self_sha256=digest(value))
def unseal(value):
    body={k:v for k,v in value.items() if k!='self_sha256'};expect(value.get('self_sha256')==digest(body),'receipt_self_hash_mismatch');return body
def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite_json'))) for line in f if line.strip()]

def extract_exact(archive,root,members,manifest_name=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();expected=set(members)|({manifest_name} if manifest_name else set())
        expect(len(names)==len(set(names)) and set(names)==expected,'archive_member_inventory_mismatch')
        if manifest_name:expect(names[-1]==manifest_name,'manifest_not_last')
        for name,item in members.items():
            safe_name(name);info=z.getinfo(name);expect(((info.external_attr>>16)&0o170000)!=0o120000,'archive_symlink')
            target=resolve(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as f:expect(info.file_size==item['bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==item['sha256'],'archive_member_changed:'+name)
            if target.exists():verify(target,item);continue
            partial=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
            with z.open(name) as source,partial.open('xb') as dest:shutil.copyfileobj(source,dest,8*1024*1024);dest.flush();os.fsync(dest.fileno())
            verify(partial,item);expect(not target.exists(),'concurrent_extraction');partial.rename(target)
        if manifest_name:atomic_bytes(root/manifest_name,z.read(manifest_name))
    return root

def cpu_capacity():
    cap=os.cpu_count() or 1
    if hasattr(os,'sched_getaffinity'):cap=min(cap,len(os.sched_getaffinity(0)))
    root=Path('/sys/fs/cgroup');quota=root/'cpu.max'
    if quota.exists():
        q,p=quota.read_text().split()[:2]
        if q!='max':cap=min(cap,max(1,math.floor(int(q)/int(p))))
    else:
        for part in (root/'cpu',root):
            q,p=part/'cpu.cfs_quota_us',part/'cpu.cfs_period_us'
            if q.exists() and p.exists() and int(q.read_text())>0:cap=min(cap,max(1,math.floor(int(q.read_text())/int(p.read_text()))));break
    return max(1,cap)

def worker_count(capacity,job_count):
    """One Node process per independent game, bounded by effective CPU capacity."""
    expect(type(capacity) is int and capacity>=1 and type(job_count) is int and job_count>=0,'invalid_worker_capacity_or_job_count')
    return max(1,min(capacity,job_count))

def preflight():
    expect(Path('/content').is_dir() and bool(os.environ.get('COLAB_RELEASE_TAG')),'full_replay_requires_colab')
    expect(sys.version_info[:2]==(3,12),'python_3_12_runtime_required')
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());ram=int(memory['MemTotal'].split()[0])*1024
    expect(ram/1e9>=48,'cpu_high_ram_runtime_at_least_48GB_required')
    expect(os.environ.get('COLAB_GPU','0') in ('','0'),'cpu_runtime_required')
    return dict(python=sys.version,actual_ram_bytes=ram,ram_used_as_worker_gate=False,cpu_capacity=cpu_capacity(),inner_threads=1)

def execute(command,cwd=None):
    """Reader thread prevents stdout backpressure; a separate heartbeat survives Node blocking."""
    process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    def read():
        for line in process.stdout:print(line.rstrip(),flush=True)
    reader=threading.Thread(target=read,daemon=True);reader.start();started=time.monotonic()
    while process.poll() is None:
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:print(json.dumps(dict(phase='terminal_subprocess_running',elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    reader.join();process.stdout.close();expect(process.returncode==0,'terminal_subprocess_failed')

def load_module(name,file):
    sys.path.insert(0,str(Path(file).parent));spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_carry_module(release):
    file=Path(__file__).with_name('stage7_terminal_carry_forward_r1.py')
    expect(identity(file)==release['members'].get('src/'+file.name),'executing_carry_code_differs_from_frozen_release')
    return load_module('stage7_terminal_carry_forward_r1',file)

def validate_binding(binding,release_sha,package_sha):
    expect(binding.get('schema_version')==SCHEMA+'_execution_binding' and binding.get('release_sha256')==release_sha and binding.get('package_sha256')==package_sha,'execution_release_binding_mismatch')
    expect(binding.get('experiment_type')=='RETROSPECTIVE_MODEL_SENSITIVITY' and binding.get('authority')==AUTHORITY,'execution_authority_mismatch')
    expect(set(binding['upstream'])=={'capture89','range90'},'exact_upstream_roles_required')
    required={'capture89':{'release','package','result','import_receipt'},'range90':{'release','code_package','result','import_receipt'}}
    for role,items in binding['upstream'].items():expect(set(items)==required[role],'upstream_member_roles_mismatch')
    settings=binding['settings'];expect(set(settings)=={'prior_strengths','likelihood_strength','samples','range_maximum_ms','terminal_maximum_ms'},'exact_settings_required')
    expect(set(settings['prior_strengths'])=={'population','player','session'} and all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in settings['prior_strengths'].values()),'frozen_prior_strengths_required')
    expect(0<=settings['likelihood_strength']<=1 and isinstance(settings['samples'],int) and 2<=settings['samples']<=1024 and all(0<settings[n]<=1000 for n in ('range_maximum_ms','terminal_maximum_ms')),'bounded_inference_settings_required')
    payoff=binding.get('payoff_contract');expect(payoff is None or set(payoff)=={'path','bytes','sha256'},'payoff_descriptor_invalid')
    expect(isinstance(binding.get('payoff_evidence',[]),list),'payoff_evidence_inventory_required')
    for items in binding['upstream'].values():
        for item in items.values():safe_name(item['path']);expect(set(item)=={'path','bytes','sha256'},'exact_input_identity_required')

def authenticate_upstreams(binding,input_root,work):
    """Revalidate accepted artifacts with their frozen upstream validators; never trust a result label alone."""
    paths={role:{name:verify(resolve(input_root,item['path']),item) for name,item in items.items()} for role,items in binding['upstream'].items()}
    c=paths['capture89'];release=unseal(load(c['release']));expect(release['schema_version']=='stage6_capture_release_v2','capture_release_schema')
    verify(c['package'],release['package']);manifest=release['package_manifest'];croot=extract_exact(c['package'],work/'capture89',manifest['members'],'package_manifest.json')
    expect(load(croot/'package_manifest.json')==manifest and sha(croot/'protocol.json')==release['protocol_sha256'],'capture_package_protocol_changed')
    capture_import=load_module('frozen_capture89_import',croot/'src/stage6_empirical_capture_import_v2.py')
    with zipfile.ZipFile(c['result']) as z:
        result_manifest=json.loads(z.read('result_manifest.json'))
        expect(z.namelist()[-1]=='result_manifest.json','capture_result_manifest_not_last')
    rroot=extract_exact(c['result'],work/'capture89_result',result_manifest['members'],'result_manifest.json')
    with zipfile.ZipFile(c['result']) as z:recomputed=capture_import.validate_result_members(z,result_manifest,release['protocol'],release['package']['sha256'])
    accepted=load(c['import_receipt']);unseal(accepted)
    expect(accepted.get('accepted_for_review') is True and accepted.get('source_release_sha256')==sha(c['release']) and accepted.get('result_sha256')==sha(c['result']) and accepted.get('recomputed')==recomputed,'capture_accepted_import_receipt_mismatch')
    expect(accepted.get('transport',{}).get('passes')==2,'capture_external_readbacks_not_accepted')
    prepare=load(rroot/'prepare_receipt.json');record_members=[f for f in prepare['files'] if f['name']=='prepared_capture_records.jsonl'];expect(len(record_members)==1,'prepared_records_member_not_exact')
    records=verify(resolve(rroot,record_members[0]['path']),record_members[0])
    r=paths['range90'];rrelease=load(r['release']);verify(r['code_package'],rrelease['code_package'])
    coderoot=extract_exact(r['code_package'],work/'range90_code',rrelease['code_package']['members'])
    rimport=load_module('frozen_range90_import',coderoot/'stage6_empirical_range_learning_v2_release.py')
    rr=rimport.strict_import(r['result'],sha(r['result']),r['release'],work/'range90_result')
    raccept=load(r['import_receipt']);expect(all(raccept.get(k)==rr.get(k) for k in ('schema_version','status','release_id','authority','result_zip_sha256','release_manifest_sha256','members')),'range_accepted_import_receipt_mismatch')
    expect(raccept.get('status')=='ACCEPT_DIAGNOSTIC_ARTIFACT','range_artifact_not_accepted')
    payoff=None;evidence=[]
    if binding.get('payoff_contract'):
        payoff=verify(resolve(input_root,binding['payoff_contract']['path']),binding['payoff_contract']);doc=load(payoff)
        expect(doc.get('schema_version')==SCHEMA+'_payoff_contracts' and doc.get('independent_source_binding') is True,'independent_payoff_contract_schema')
        expect(len({x['decision_id'] for x in doc.get('decisions',[])})==len(doc.get('decisions',[])) and len({x['game_id'] for x in doc.get('games',[])})==len(doc.get('games',[])),'duplicate_payoff_decision_or_game')
        evidence=[verify(resolve(input_root,e['path']),e) for e in binding.get('payoff_evidence',[])]
        hashes={sha(p) for p in evidence};rules=doc.get('decisions',[])+doc.get('games',[]);expect(hashes and rules and all(set(d.get('evidence_sha256',[]))<=hashes and d.get('evidence_sha256') for d in rules),'payoff_evidence_not_authenticated')
    upstream_receipt=seal(dict(schema_version=SCHEMA+'_upstream_acceptance',capture89_result_sha256=sha(c['result']),range90_result_sha256=sha(r['result']),
        capture89_import_sha256=sha(c['import_receipt']),range90_import_sha256=sha(r['import_receipt']),records=identity(records),authority=AUTHORITY))
    return dict(capture_root=croot,capture_result_root=rroot,capture_protocol=release['protocol'],records=records,range_root=work/'range90_result',payoff=payoff,receipt=upstream_receipt,paths=paths)

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def near(actual,expected,reason):
    expect(finite(actual) and finite(expected) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-7),reason)

def exact_authority(value):
    expect(isinstance(value,dict),'terminal_value_authority_mismatch')
    for key,expected in AUTHORITY.items():
        if key=='as_of_decision_deployment_test':continue
        expect(key in value and type(value[key]) is type(expected) and value[key]==expected,'terminal_value_authority_mismatch')

def request_bytes(request):
    """Authenticate original Node serialization; never round/normalize probabilities."""
    raw=request.get('canonical_input_json');ledger_raw=request.get('canonical_ledger_json')
    expect(isinstance(raw,str) and isinstance(ledger_raw,str),'canonical_request_bytes_required')
    expect(hashlib.sha256(raw.encode()).hexdigest()==request['request_sha256'],'terminal_request_digest_mismatch')
    expect(hashlib.sha256(ledger_raw.encode()).hexdigest()==request['reconstructed_ledger_sha256'],'terminal_ledger_digest_mismatch')
    def unique(pairs):
        result={}
        for key,value in pairs:
            expect(key not in result,'duplicate_canonical_request_key');result[key]=value
        return result
    decode=lambda text:json.loads(text,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite_request')))
    # Compare the two parsed values under the same encoder so true != 1 and
    # false != 0. Hash authentication above still uses the original Node bytes.
    expect(canonical(decode(raw))==canonical(request['input']),'terminal_request_bytes_differ_from_input')
    return decode(ledger_raw)

def validate_summary(value,bounds,samples):
    expect(type(value.get('samples')) is int and value['samples']==samples,'terminal_summary_sample_mismatch')
    expect(value.get('uncertainty_excludes_range_model_and_reveal_selection') is True,'terminal_uncertainty_contract_mismatch')
    reported=value.get('physical_value_bounds_bb');expect(isinstance(reported,list) and len(reported)==2,'terminal_physical_bounds_missing')
    for actual,wanted in zip(reported,bounds):near(actual,wanted,'terminal_physical_bounds_mismatch')
    mean=value.get('mean_bb');expect(finite(mean) and bounds[0]-1e-7<=mean<=bounds[1]+1e-7,'terminal_mean_outside_physical_bounds')
    se=value.get('monte_carlo_standard_error_bb');expect(finite(se) and 0<=se<=(bounds[1]-bounds[0])/(2*math.sqrt(samples-1))+1e-7,'terminal_standard_error_outside_bounds')
    radius=(bounds[1]-bounds[0])*math.sqrt(math.log(40)/(2*samples))
    ci=value.get('sampling_95_ci_hoeffding_bb');expect(isinstance(ci,list) and len(ci)==2,'terminal_interval_missing')
    for actual,wanted in zip(ci,[max(bounds[0],mean-radius),min(bounds[1],mean+radius)]):near(actual,wanted,'terminal_interval_arithmetic_mismatch')

def validate_terminal_result(row):
    """Check identities and deterministic accounting, without sampling or scoring."""
    result=row.get('result')
    if result is None:return
    exact_authority(result)
    expect(result.get('schema_version')=='stage6_empirical_terminal_value_v2' and result.get('decision_context_key')==row['decision_id'] and result.get('value_type')=='model_dependent_net_future_value_bb','terminal_result_contract_mismatch')
    expect(finite(result.get('latency_ms')) and result['latency_ms']>=0,'terminal_latency_invalid')
    completed=result.get('completed_samples');attempts=result.get('attempts')
    expect(type(completed) is int and type(attempts) is int and 0<=completed<=attempts,'terminal_sample_counts_invalid')
    if result.get('status')=='ABSTAIN':
        expect(row['disposition']=='HELD' and isinstance(result.get('reason'),str) and result['reason'] in row['reasons'] and result.get('action_values') is None and result.get('paired_differences') is None,'terminal_abstention_contract_mismatch')
        return
    expect(result.get('status')=='READY_MODEL_DEPENDENT_DIAGNOSTIC' and result.get('reason')=='proved_terminal_passive_action_subset','terminal_ready_contract_mismatch')
    request=row['request'];inp=request['input'];ledger=request_bytes(request)
    expect(result.get('input_sha256')==request['request_sha256'] and result.get('ledger_sha256')==request['reconstructed_ledger_sha256'] and inp.get('decision_context_key')==row['decision_id'],'terminal_request_digest_mismatch')
    samples=inp['sampler']['samples'];expect(type(samples) is int and 2<=samples<=1024 and completed==samples and samples<=attempts<=samples*64 and result.get('joint_card_rejections')==attempts-samples,'terminal_sample_counts_invalid')
    expect(result.get('seed')==inp['sampler']['seed'] and isinstance(result.get('scenario_sha256'),str) and len(result['scenario_sha256'])==64 and all(c in '0123456789abcdef' for c in result['scenario_sha256']),'terminal_sampler_identity_mismatch')
    expect(result.get('closure')==dict(kind=inp['closure']['kind'],reconstructed_pending_actor_ids=[inp['hero_id']],future_betting=False),'terminal_closure_mismatch')
    players={p['id']:dict(id=p['id'],total_bb=0,folded=False) for p in inp['ledger']['players']};street={};seen=set()
    for event in inp['ledger']['events']:
        if event['id'] in seen:continue
        seen.add(event['id']);actor=players[event['actor_id']];key=(event['street'],actor['id']);committed=street.get(key,0);action=event['action']
        if action=='fold':actor['folded']=True;continue
        if action=='check':continue
        if action=='return_uncalled':delta=-event['amount_bb']
        elif action=='raise_to':delta=event['raise_to_bb']-committed
        else:
            expect(action in ('post_blind','post_ante','bet','call','all_in_increment'),'terminal_ledger_action_invalid');delta=event['amount_bb']
        expect(finite(delta),'terminal_ledger_nonfinite');actor['total_bb']+=delta
        if action!='post_ante':street[key]=committed+delta
    expect(len(ledger['players'])==len(players),'terminal_ledger_roster_mismatch')
    for player in ledger['players']:
        expected=players[player['id']];near(player['total_bb'],expected['total_bb'],'terminal_ledger_total_mismatch');expect(player['folded'] is expected['folded'],'terminal_ledger_fold_mismatch')
    hero=inp['hero_id'];hero_current=next(p for p in inp['current_players'] if p['id']==hero)
    cost=min(max(0,inp['current_wager_bb']-hero_current['street_committed_bb']),hero_current['remaining_stack_bb'])
    passive=inp['legal']['call' if cost>1e-8 else 'check'];actions=[(passive['id'],passive['kind'],cost)]
    if inp['legal'].get('fold',{}).get('allowed') is True:actions.append((inp['legal']['fold']['id'],'fold',0))
    expect(len(set(inp['requested_action_ids']))==len(actions) and set(inp['requested_action_ids'])=={a[0] for a in actions},'terminal_requested_actions_mismatch')
    values=result.get('action_values');expect(isinstance(values,list) and [a.get('action_id') for a in values]==[a[0] for a in actions],'terminal_action_ids_mismatch')
    near(sum(p['total_bb'] for p in players.values())-inp['ledger'].get('rake_bb',0),inp['pot_bb'],'terminal_pot_ledger_mismatch')
    bounds=[]
    for value,(action_id,kind,investment) in zip(values,actions):
        expect(value.get('kind')==kind,'terminal_action_kind_mismatch')
        near(value.get('future_investment_bb'),investment,'terminal_future_cost_mismatch');near(value.get('sunk_hero_contribution_bb'),players[hero]['total_bb'],'terminal_sunk_cost_mismatch')
        contributions=[dict(p,total_bb=p['total_bb']+(investment if p['id']==hero else 0),folded=p['folded'] or (p['id']==hero and kind=='fold')) for p in players.values()]
        levels=sorted({p['total_bb'] for p in contributions if p['total_bb']>0});expected_pots=[];previous=0;rake=inp['rake']['total_bb_by_action'][action_id]
        expect(finite(rake) and rake>=0,'terminal_rake_invalid')
        for level in levels:
            participants=[p for p in contributions if p['total_bb']>=level];amount=(level-previous)*len(participants);previous=level
            eligible=[p['id'] for p in participants if not p['folded']];uncalled=len(participants)==1;deduction=0 if uncalled else min(amount,rake);rake-=deduction
            expected_pots.append(dict(amount_bb=amount,eligible_ids=eligible,contributors=[p['id'] for p in participants],uncalled_return=uncalled,pot_index=len(expected_pots),eligibility_group_key=canonical(sorted(eligible)).decode(),rake_bb=deduction,net_amount_bb=amount-deduction))
        near(rake,0,'terminal_rake_exceeds_pot')
        pots=value.get('pots');expect(isinstance(pots,list) and len(pots)==len(expected_pots),'terminal_pot_count_mismatch')
        award=0
        for pot,expected in zip(pots,expected_pots):
            expect(set(pot)==set(expected)|{'expected_hero_share','expected_hero_award_bb'},'terminal_pot_fields_mismatch')
            for key,wanted in expected.items():
                if key in ('amount_bb','rake_bb','net_amount_bb'):near(pot[key],wanted,'terminal_pot_arithmetic_mismatch')
                else:expect(type(pot[key]) is type(wanted) and pot[key]==wanted,'terminal_pot_identity_mismatch')
            share=pot['expected_hero_share'];expect(finite(share) and 0<=share<=1,'terminal_share_outside_bounds')
            if hero not in expected['eligible_ids']:near(share,0,'terminal_ineligible_hero_award')
            elif expected['eligible_ids']==[hero]:near(share,1,'terminal_uncontested_hero_award')
            near(pot['expected_hero_award_bb'],share*pot['net_amount_bb'],'terminal_pot_award_mismatch');award+=pot['expected_hero_award_bb']
        maximum=sum(p['net_amount_bb'] for p in expected_pots if hero in p['eligible_ids']);guaranteed=sum(p['net_amount_bb'] for p in expected_pots if p['uncalled_return'] and hero in p['eligible_ids'])
        bound=[guaranteed-investment,maximum-investment];bounds.append(bound);validate_summary(value,bound,samples)
        near(value.get('expected_terminal_award_bb'),award,'terminal_total_award_mismatch');near(value.get('mean_bb'),award-investment,'terminal_value_arithmetic_mismatch')
    expected_pairs=[(i,j) for i in range(len(actions)) for j in range(i+1,len(actions))];pairs=result.get('paired_differences')
    expect(isinstance(pairs,list) and len(pairs)==len(expected_pairs),'terminal_pair_count_mismatch')
    for pair,(i,j) in zip(pairs,expected_pairs):
        expect(pair.get('action_id')==actions[i][0] and pair.get('minus_action_id')==actions[j][0] and pair.get('common_joint_scenarios') is True,'terminal_pair_identity_mismatch')
        validate_summary(pair,[bounds[i][0]-bounds[j][1],bounds[i][1]-bounds[j][0]],samples)
        near(pair.get('mean_bb'),values[i]['mean_bb']-values[j]['mean_bb'],'terminal_pair_arithmetic_mismatch')

def publish_report(output,report,resources,run_id):
    """Reuse original execution telemetry when closed work resumes on another VM."""
    target=Path(output)/'terminal_report.json'
    if target.exists():
        previous=load(target)
        expect(set(previous)==set(report)|{'resources','run_id'} and previous['run_id']==run_id and all(previous[k]==v for k,v in report.items()),'sealed_terminal_report_changed')
        prior=previous['resources'];expect(prior.get('inner_threads')==1 and prior.get('ram_used_as_worker_gate') is False and type(prior.get('workers')) is int and type(prior.get('cpu_capacity')) is int and 1<=prior['workers']<=prior['cpu_capacity'],'sealed_terminal_resources_invalid')
        return
    write(target,dict(report,resources=resources,run_id=run_id))

def summarize(rows,records):
    expected={r['decision_id']:r for r in records};expect(len(expected)==len(records) and len(rows)==len(records) and {r['decision_id'] for r in rows}==set(expected),'terminal_denominator_not_exact_upstream')
    groups=defaultdict(lambda:dict(records=0,supported=0,held=0,reasons=Counter(),paired_differences=[]));reasons=Counter();supported=0
    for row in rows:
        rec=expected[row['decision_id']];expect(all(row.get(k)==rec[k] for k in ('game_id','hand_id','cohort')),'terminal_record_provenance_mismatch')
        expect(row.get('schema_version')==SCHEMA and row.get('authority')==AUTHORITY and row.get('experiment_type')=='RETROSPECTIVE_MODEL_SENSITIVITY','terminal_row_authority_mismatch')
        validate_terminal_result(row)
        group=groups[(row['cohort'],row['game_id'])];group['records']+=1
        if row['disposition']=='SUPPORTED_MODEL_DEPENDENT_TERMINAL_VALUE':
            expect(not row['reasons'] and row.get('request') and row.get('source_audit',{}).get('closure'),'supported_value_missing_closure_or_request')
            result=row.get('result');expect(result and result['status']=='READY_MODEL_DEPENDENT_DIAGNOSTIC' and all(result.get(k)==v for k,v in AUTHORITY.items() if k!='as_of_decision_deployment_test'),'terminal_value_authority_mismatch')
            request=row['request'];expect(request['request_sha256']==result['input_sha256'] and result['ledger_sha256']==request['reconstructed_ledger_sha256'] and result['decision_context_key']==row['decision_id'],'terminal_request_digest_mismatch')
            expect(result['completed_samples']==request['input']['sampler']['samples'] and len(result['action_values'])==len(request['input']['requested_action_ids']),'partial_or_wrong_action_value')
            supported+=1;group['supported']+=1;group['paired_differences'] += [p['mean_bb'] for p in result['paired_differences']]
        else:
            expect(row['disposition']=='HELD' and row['reasons'] and (row.get('result') is None or row['result']['status']=='ABSTAIN'),'invalid_held_disposition');group['held']+=1
        group['reasons'].update(row['reasons']);reasons.update(row['reasons'])
    by_game=[]
    for (cohort,game),g in sorted(groups.items()):
        values=g.pop('paired_differences');g['reasons']=dict(sorted(g['reasons'].items()));by_game.append(dict(cohort=cohort,game_id=game,**g,mean_supported_paired_difference_bb=sum(values)/len(values) if values else None))
    return dict(schema_version=SCHEMA+'_report',disposition='DIAGNOSTIC_SUPPORTED_SUBSET' if supported else 'HELD_NO_SUPPORTED_VALUE_SUBSET',records=len(records),supported=supported,held=len(records)-supported,
        source_audit_completed=sum(r.get('source_audit') is not None for r in rows),reasons=dict(sorted(reasons.items())),by_game=by_game,authority=AUTHORITY,
        interpretation='Model-dependent terminal subset. Missing values remain in the source denominator. No identified off-policy strategy return or promotion.')

def phase_game(game,config,output,run_id,node,tsx,script):
    key=hashlib.sha256(game.encode()).hexdigest()[:20];receipt_path=output/('game_'+key+'_receipt.json')
    if receipt_path.exists():
        receipt=unseal(load(receipt_path));expect(receipt['run_id']==run_id and receipt['game_id']==game,'resumed_phase_binding_changed')
        for item in receipt['files']:verify(resolve(output,item['path']),item)
        return receipt_path,receipt
    attempt=output/'phases'/key/('attempt_'+uuid.uuid4().hex);attempt.mkdir(parents=True);write(attempt/'config.json',config)
    execute([node,'--import',tsx,str(script),str(attempt/'config.json'),str(attempt)],cwd=script.parent)
    files=[dict(name=n,path=(attempt/n).relative_to(output).as_posix(),**identity(attempt/n)) for n in ('terminal_rows.jsonl','game_receipt.json')]
    receipt=dict(schema_version=SCHEMA+'_phase',run_id=run_id,game_id=game,files=files);write(receipt_path,seal(receipt));return receipt_path,receipt

def game_configs(upstream,work,code_root,binding):
    """Shared by execution and strict import; construct inputs from accepted upstreams."""
    work,code_root=Path(work),Path(code_root);croot=upstream['capture_root'];protocol=upstream['capture_protocol'];runtime=work/'capture_runtime'
    rroot=upstream['range_root'];artifact=rroot/'range_response_artifact.json';completion=load(rroot/'completion_receipt.json')
    common=dict(records_path=str(upstream['records']),native_root=str(code_root/'native'),parser_path=str(croot/protocol['session_parser']['path']),
        runtime_module=str(runtime/'Assistant Qualified/Assistant Code/scripts/ml/stage017-live-assistant-popup.mjs'),runtime_root=str(runtime),runtime_args=protocol['runtime_args'],hero_name=protocol['hero_name'],
        artifact_path=str(artifact),artifact_identity=identity(artifact),identity_path=str(rroot/'identity_mapping.json'),identity_sha256=sha(rroot/'identity_mapping.json'),
        range_protocol_path=str(rroot/'protocol.json'),range_adapter_path=str(rroot/'range_adapter_receipt.json'),payoff_path=str(upstream['payoff']) if upstream['payoff'] else None,
        actual_artifact_created_utc=completion.get('execution_provenance',{}).get('fit_completed_utc'),settings=binding['settings'])
    games={r['game_id'] for r in jsonl(upstream['records'])}
    return {game['game_id']:dict(common,game_id=game['game_id'],log_files=[dict(f,path=str(croot/f['path'])) for f in game['files'] if f['role'] in ('session_log','direct_log')]) for game in protocol['games'] if game['game_id'] in games}

def run(binding_path,binding_sha,release_path,release_sha,package_path,input_root,work,output,node='node',tsx='tsx',bounded_fixture=False):
    resources=preflight() if not bounded_fixture else dict(cpu_capacity=cpu_capacity(),inner_threads=1,ram_used_as_worker_gate=False)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
    expect(sha(binding_path)==binding_sha and sha(release_path)==release_sha,'execution_binding_or_release_sha_mismatch')
    release=load(release_path);expect(release['schema_version']==SCHEMA+'_release' and release['authority']==AUTHORITY,'terminal_release_schema')
    verify(package_path,release['package']);binding=load(binding_path);validate_binding(binding,release_sha,release['package']['sha256'])
    work,output=Path(work),Path(output);work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    code_root=extract_exact(package_path,work/'code91',release['members'],'package_manifest.json')
    expect(identity(Path(__file__))==release['members']['src/'+Path(__file__).name],'executing_controller_not_frozen')
    carry=load_carry_module(release)
    run_id=digest(dict(binding_sha256=binding_sha,release_sha256=release_sha));run_binding=dict(schema_version=SCHEMA+'_run',run_id=run_id,binding_sha256=binding_sha,release_sha256=release_sha,package_sha256=sha(package_path),authority=AUTHORITY)
    if not (output/'run_binding.json').exists():expect(all(p.is_file() and p.name.startswith('run_binding.json.partial-') for p in output.iterdir()),'unowned_output_directory')
    write(output/'run_binding.json',run_binding);atomic_bytes(output/'execution_binding.json',Path(binding_path).read_bytes())
    upstream=authenticate_upstreams(binding,input_root,work);write(output/'upstream_acceptance.json',upstream['receipt'])
    croot=upstream['capture_root'];protocol=upstream['capture_protocol'];snapshot=protocol['snapshots']['experimental']
    frozen_capture=load_module('frozen_capture91_controller',croot/'src/stage6_empirical_capture_v2.py')
    runtime=work/'capture_runtime';archive=croot/snapshot['archive']['path']
    if runtime.exists():frozen_capture.verify_extracted(archive,runtime)
    else:frozen_capture.safe_extract(archive,runtime)
    for item in snapshot['external_members']:frozen_capture.copy_authenticated(croot/item['path'],frozen_capture.resolve_member(runtime,item['runtime_path']),item)
    modules=code_root/'node_modules';expect(modules.is_dir(),'pinned_npm_dependencies_missing')
    if not (runtime/'node_modules').exists():(runtime/'node_modules').symlink_to(modules,target_is_directory=True)
    expect((runtime/'node_modules').resolve()==modules.resolve(),'runtime_node_dependencies_changed')
    records=jsonl(upstream['records']);game_ids=sorted({r['game_id'] for r in records})
    configs=game_configs(upstream,work,code_root,binding)
    carried=carry.authenticate(binding,input_root,work,release,configs,records,upstream['receipt'])
    resources['workers']=worker_count(resources['cpu_capacity'],len(game_ids)-len(carried))
    print(json.dumps(dict(phase='parallel_game_process_plan',games=len(game_ids),carried_games=len(carried),**resources)),flush=True)
    def one(game):
        if game in carried:return carry.publish(game,carried[game],output,run_id)
        return phase_game(game,configs[game],output,run_id,node,tsx,code_root/'src/stage6_empirical_terminal_replay_v2.mjs')
    with ThreadPoolExecutor(max_workers=resources['workers']) as pool:phases=list(pool.map(one,game_ids))
    rows=[];members={'run_binding.json','execution_binding.json','upstream_acceptance.json'}
    for receipt_path,receipt in phases:
        members.add(receipt_path.relative_to(output).as_posix())
        for item in receipt['files']:
            members.add(item['path'])
            if item['name']=='terminal_rows.jsonl':rows.extend(jsonl(output/item['path']))
    report=summarize(rows,records);publish_report(output,report,resources,run_id);members.add('terminal_report.json')
    # Authenticate exact inputs again after execution; no changed source is accepted.
    for items in binding['upstream'].values():
        for item in items.values():verify(resolve(input_root,item['path']),item)
    if binding.get('payoff_contract'):verify(resolve(input_root,binding['payoff_contract']['path']),binding['payoff_contract'])
    for item in binding.get('payoff_evidence',[]):verify(resolve(input_root,item['path']),item)
    for name,item in release['members'].items():verify(resolve(code_root,name),item)
    write(output/'completion_receipt.json',seal(dict(schema_version=SCHEMA+'_completion',run_id=run_id,authority=AUTHORITY,files={n:identity(output/n) for n in sorted(members)})))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','release','package','input-root','work','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--binding-sha256',required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--node',default='node');p.add_argument('--tsx',default='tsx');a=p.parse_args()
    stop=threading.Event();started=time.monotonic()
    def heartbeat():
        while not stop.wait(20):print(json.dumps(dict(phase='terminal_authentication_or_execution',elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    t=threading.Thread(target=heartbeat,daemon=True);t.start()
    try:print(json.dumps(run(a.binding,a.binding_sha256,a.release,a.release_sha256,a.package,a.input_root,a.work,a.output,a.node,a.tsx)),flush=True)
    finally:stop.set();t.join()
if __name__=='__main__':main()
