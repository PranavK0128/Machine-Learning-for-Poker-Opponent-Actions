"""Narrow, authenticated carry-forward of the two closed Notebook04 r4 phases."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile
import stage6_empirical_terminal_replay_v2 as core

PREDECESSOR={
    'release':dict(path='manifests/stage7_04_release_r4.json',bytes=2739,sha256='cc423cce617f0a5afad471b83e44c84a3dc02b1cea7663afd3d4188e7963d3ec'),
    'package':dict(path='inputs/stage7_terminal_replay_v2_code_r4.zip',bytes=61916,sha256='4acebd7a726a7521275746111aa8701e3470e20c0950272b1404e626f4614008'),
    'binding':dict(path='manifests/stage7_04_execution_binding_r4.json',bytes=2037,sha256='26f48e8a01bbc58a4d395e0c397ed750ba862517d2b60e1e0a1c3e29cfe54678'),
    'partial_archive':dict(path='runtime_state/notebook04_r4_partial_preserved_r1.zip',bytes=22881,sha256='1e5da95a7e8fc7c2a1e949acad474876683de1ae33bad7d38be2fabd678a6440'),
    'games':['pglEe0Gm8CFL8R-9OsxY1JZbZ','pglUh7wPbv768eg_RT448nMTx'],
}
PATH_KEYS={'records_path','native_root','parser_path','runtime_module','runtime_root','artifact_path','identity_path','range_protocol_path','range_adapter_path','payoff_path'}
SUCCESSOR_REPLAY_SHA256='ae0a156da1524f4476d710737c69eae00802052ebe210a2c8392cc5a3be6a500'
DISPLAY_SUCCESSOR_REPLAY_SHA256='7b0f104f89202ebb25a8896a86b07e9d5ea8cc876201804939903747f3f95755'
PREDECESSOR_R5={
    'release':dict(path='manifests/stage7_04_release_r5.json',bytes=3055,sha256='afbbc71ed6e3da84a970754e001df91618e1defa9a2dcbe69ff39552eebc53c7'),
    'package':dict(path='inputs/stage7_terminal_replay_v2_code_r5.zip',bytes=66131,sha256='70b22ba82446d42c70aba088225382f707eea6d22b276f6b7603b553a4dfeb5e'),
    'binding':dict(path='manifests/stage7_04_execution_binding_r5.json',bytes=2749,sha256='9c3e7d58e4b0ca010015f77ca62dbe17195deea594760c490cf7042e7cc5c026'),
    'result':dict(path='returned_results/91_terminal_v2_ad8de96cf4e5e747.zip',bytes=99083,sha256='5fed23ef6bbd554c3c1794668ac09d89a335472e0402350e3dda5fe44c37d5e3'),
    'import_receipt':dict(path='returned_results/04_terminal_r5/strict_import_receipt.json',bytes=1103,sha256='0f106bdf26148489dc738eeb171183d1e0cb06cb265b754f3565ba1b6a943b92'),
    'games':['pgl0pqNU8P-7SwPeHButJsg7H','pgl8cdDzWBGiScq2GsdVpPl6Q','pglEe0Gm8CFL8R-9OsxY1JZbZ','pglTOp5olR76I1nfds-TiWFbR','pglUh7wPbv768eg_RT448nMTx'],
}

def authenticate_r5(binding,input_root,work,release,configs,records,upstream_receipt):
    """Authenticate the complete r5 result and its nested r4 provenance, without replay."""
    spec=binding['predecessor'];core.expect(spec==PREDECESSOR_R5,'unsupported_predecessor_carry_forward')
    core.expect(release['members']['src/stage6_empirical_terminal_replay_v2.mjs']['sha256']==DISPLAY_SUCCESSOR_REPLAY_SHA256,'unreviewed_successor_replay_for_carry')
    files={k:core.verify(core.resolve(input_root,v['path']),v) for k,v in spec.items() if k!='games'}
    prior=core.load(files['binding']);old_release=core.load(files['release'])
    core.validate_binding(prior,spec['release']['sha256'],spec['package']['sha256'])
    keep=lambda b:{k:v for k,v in b.items() if k not in ('release_sha256','package_sha256','predecessor')}
    core.expect(core.canonical(keep(prior))==core.canonical(keep(binding)),'predecessor_scientific_or_upstream_binding_changed')
    core.verify(files['package'],old_release['package'])
    for name,item in old_release['members'].items():
        if name.startswith('native/') or name=='src/stage6_empirical_terminal_replay_v2.py':
            core.expect(release['members'].get(name)==item,'carried_native_or_controller_source_changed')
    oldcode=core.extract_exact(files['package'],Path(work)/'r5_predecessor_code',old_release['members'],'package_manifest.json')
    # Both modules are authenticated package members before deserialization/import.
    oldcarry=core.load_module('terminal_r5_frozen_carry',oldcode/'src/stage7_terminal_carry_forward_r1.py')
    oldimport=core.load_module('terminal_r5_frozen_import',oldcode/'src/stage6_empirical_terminal_replay_v2_import.py')
    capture_release=core.unseal(core.load(core.resolve(input_root,prior['upstream']['capture89']['release']['path'])))
    with zipfile.ZipFile(core.resolve(input_root,prior['upstream']['capture89']['result']['path'])) as z:
        manifest=json.loads(z.read('result_manifest.json'))
    names=[n for n in manifest['members'] if n.endswith('/prepared_capture_records.jsonl')]
    core.expect(len(names)==1,'exact_prepared_records_member_required')
    independent=core.game_configs(dict(capture_root=Path(work)/'capture89',capture_protocol=capture_release['protocol'],
        records=Path(work)/'capture89_result'/names[0],range_root=Path(work)/'range90_result',payoff=None),work,Path(work)/'code91',prior)
    core.expect(prior['payoff_contract'] is None and core.canonical(independent)==core.canonical(configs),'predecessor_config_not_exact_successor_input')
    nested=oldcarry.authenticate(prior,input_root,work,old_release,independent,records,upstream_receipt)
    with zipfile.ZipFile(files['result']) as z:manifest=json.loads(z.read('result_manifest.json'))
    root=core.extract_exact(files['result'],Path(work)/'r5_predecessor_result',manifest['members'],'result_manifest.json')
    core.expect(core.load(root/'upstream_acceptance.json')==upstream_receipt,'predecessor_upstream_acceptance_changed')
    oldimport.validate_result(root,manifest,records,spec['binding']['sha256'],spec['release']['sha256'],spec['package']['sha256'],nested,oldcarry)
    accepted=core.unseal(core.load(files['import_receipt']))
    core.expect(accepted.get('status')=='ACCEPT_FOR_DIAGNOSTIC_REVIEW' and accepted.get('result_sha256')==spec['result']['sha256'] and accepted.get('binding_sha256')==spec['binding']['sha256'] and accepted.get('release_sha256')==spec['release']['sha256'] and accepted.get('authority')==core.AUTHORITY,'predecessor_import_binding_changed')
    oldrun=core.digest(dict(binding_sha256=spec['binding']['sha256'],release_sha256=spec['release']['sha256']))
    result={}
    for game in spec['games']:
        key=hashlib.sha256(game.encode()).hexdigest()[:20];name='game_'+key+'_receipt.json';phase=core.unseal(core.load(root/name))
        provenance=dict(schema_version='stage7_terminal_display_carry_r1',run_id=oldrun,archive_sha256=spec['result']['sha256'],
            phase=dict(path=name,**core.identity(root/name)),files=phase['files'],import_receipt_sha256=spec['import_receipt']['sha256'],
            source_ingestion_telemetry='ORIGINAL_R5_GAME_RECEIPT_PRESERVED_INCLUDING_NESTED_R4_LEGACY_NOTICE',
            unchanged_scope='Five frozen games without observed display-stack-only failure; exact reviewed successor source and upstream/controller/native identities pinned')
        result[game]=dict(root=root,provenance=provenance,files=phase['files'])
    return result

def rebase_config(config,old_work,new_work):
    """Only explicit path fields may change; preserve every other input exactly."""
    def path(value):
        if value is None:return None
        p=PurePosixPath(value);root=PurePosixPath(old_work)
        core.expect(p.is_absolute() and p.is_relative_to(root) and '..' not in p.parts,'predecessor_config_path_outside_work')
        return str(Path(new_work).joinpath(*p.relative_to(root).parts))
    result={k:path(v) if k in PATH_KEYS else v for k,v in config.items()}
    result['log_files']=[dict(item,path=path(item['path'])) for item in config['log_files']]
    return result

def authenticate(binding,input_root,work,release,configs,records,upstream_receipt):
    if 'predecessor' not in binding:return {}
    if binding['predecessor']==PREDECESSOR_R5:return authenticate_r5(binding,input_root,work,release,configs,records,upstream_receipt)
    spec=binding['predecessor'];core.expect(spec==PREDECESSOR,'unsupported_predecessor_carry_forward')
    core.expect(release['members']['src/stage6_empirical_terminal_replay_v2.mjs']['sha256']==SUCCESSOR_REPLAY_SHA256,'unreviewed_successor_replay_for_carry')
    files={k:core.verify(core.resolve(input_root,v['path']),v) for k,v in spec.items() if k!='games'}
    prior=core.load(files['binding']);old_release=core.load(files['release'])
    core.validate_binding(prior,spec['release']['sha256'],spec['package']['sha256'])
    core.expect('predecessor' not in prior,'nested_predecessor_not_supported')
    keep=lambda b:{k:v for k,v in b.items() if k not in ('release_sha256','package_sha256','predecessor')}
    core.expect(core.canonical(keep(prior))==core.canonical(keep(binding)),'predecessor_scientific_or_upstream_binding_changed')
    core.verify(files['package'],old_release['package'])
    oldcode=core.extract_exact(files['package'],Path(work)/'predecessor_code',old_release['members'],'package_manifest.json')
    for name,item in old_release['members'].items():
        if name.startswith('native/'):core.expect(release['members'].get(name)==item,'carried_native_source_changed')
    oldrun=core.digest(dict(binding_sha256=spec['binding']['sha256'],release_sha256=spec['release']['sha256']))
    with zipfile.ZipFile(files['partial_archive']) as z:inventory=json.loads(z.read('partial_inventory.json'))
    core.expect(inventory['run_id']==oldrun,'partial_inventory_run_changed')
    root=core.extract_exact(files['partial_archive'],Path(work)/'predecessor_result',inventory['files'],'partial_inventory.json')
    core.expect((root/'execution_binding.json').read_bytes()==files['binding'].read_bytes(),'predecessor_returned_binding_changed')
    core.expect(core.load(root/'run_binding.json')==dict(schema_version=core.SCHEMA+'_run',run_id=oldrun,binding_sha256=spec['binding']['sha256'],release_sha256=spec['release']['sha256'],package_sha256=spec['package']['sha256'],authority=core.AUTHORITY),'predecessor_run_binding_changed')
    core.expect(core.load(root/'upstream_acceptance.json')==upstream_receipt,'predecessor_upstream_acceptance_changed')
    expected_receipts={'game_'+hashlib.sha256(g.encode()).hexdigest()[:20]+'_receipt.json' for g in spec['games']}
    core.expect({n for n in inventory['files'] if n.startswith('game_')}==expected_receipts,'predecessor_closed_game_inventory_changed')
    old_work='/content/terminal91_'+spec['package']['sha256'][:16]+'/execution_'+spec['binding']['sha256'][:16]
    result={}
    for game in spec['games']:
        key=hashlib.sha256(game.encode()).hexdigest()[:20];name='game_'+key+'_receipt.json'
        receipt=core.unseal(core.load(root/name))
        core.expect(set(receipt)=={'schema_version','run_id','game_id','files'} and receipt['schema_version']==core.SCHEMA+'_phase' and receipt['run_id']==oldrun and receipt['game_id']==game,'predecessor_phase_binding_changed')
        items=receipt['files'];core.expect(len(items)==2 and {i['name'] for i in items}=={'terminal_rows.jsonl','game_receipt.json'},'predecessor_exact_files_required')
        dirs=set()
        for item in items:
            p=PurePosixPath(item['path']);core.safe_name(item['path'])
            core.expect(item['path'].startswith('phases/'+key+'/attempt_') and p.name==item['name'],'predecessor_phase_path_changed')
            core.verify(core.resolve(root,item['path']),item);dirs.add(str(p.parent))
        core.expect(len(dirs)==1,'predecessor_phase_attempt_mismatch')
        config_path=next(iter(dirs))+'/config.json';config=core.load(core.resolve(root,config_path))
        core.expect(core.canonical(rebase_config(config,old_work,work))==core.canonical(configs[game]),'predecessor_config_not_exact_successor_input')
        game_receipt=core.load(root/next(i['path'] for i in items if i['name']=='game_receipt.json'))
        count=sum(r['game_id']==game for r in records)
        core.expect(game_receipt.get('schema_version')==core.SCHEMA+'_game' and game_receipt.get('game_id')==game and game_receipt.get('records')==count and game_receipt.get('authority')==core.AUTHORITY and type(game_receipt.get('source_log_events')) is int and game_receipt['source_log_events']>=0,'predecessor_game_receipt_changed')
        rows=core.jsonl(root/next(i['path'] for i in items if i['name']=='terminal_rows.jsonl'))
        core.summarize(rows,[r for r in records if r['game_id']==game])
        provenance=dict(schema_version='stage7_terminal_carry_forward_r1',run_id=oldrun,archive_sha256=spec['partial_archive']['sha256'],phase=dict(path=name,**core.identity(root/name)),config=dict(path=config_path,**core.identity(root/config_path)),files=items,
            source_ingestion_telemetry='LEGACY_EXACT_MERGER_COMPLETED_OBSERVATION_COUNTS_UNAVAILABLE')
        result[game]=dict(root=root,provenance=provenance,files=items)
    return result

def validate_phase(phase,carried,output):
    core.expect(set(phase)=={'schema_version','run_id','game_id','files','carried_from'} and phase['carried_from']==carried['provenance'],'carried_phase_provenance_changed')
    wanted={i['name']:i for i in carried['files']};items=phase['files']
    core.expect(len(items)==2 and {i['name'] for i in items}==set(wanted),'carried_phase_file_inventory_changed')
    for item in items:
        original=wanted[item['name']]
        core.expect({k:item[k] for k in ('bytes','sha256')}=={k:original[k] for k in ('bytes','sha256')},'carried_phase_bytes_changed')
        core.verify(core.resolve(output,item['path']),original)

def publish(game,carried,output,run_id):
    key=hashlib.sha256(game.encode()).hexdigest()[:20];target=Path(output)/('game_'+key+'_receipt.json')
    if target.exists():
        phase=core.unseal(core.load(target));core.expect(phase['run_id']==run_id and phase['game_id']==game,'resumed_phase_binding_changed')
        validate_phase(phase,carried,output);return target,phase
    folder=Path(output)/'phases'/key/('attempt_carried_'+carried['provenance']['run_id'][:16]);items=[]
    for original in carried['files']:
        source=core.verify(core.resolve(carried['root'],original['path']),original);dest=folder/original['name']
        core.atomic_bytes(dest,source.read_bytes());items.append(dict(original,path=dest.relative_to(output).as_posix()))
    phase=dict(schema_version=core.SCHEMA+'_phase',run_id=run_id,game_id=game,files=items,carried_from=carried['provenance'])
    validate_phase(phase,carried,output);core.write(target,core.seal(phase));return target,phase
