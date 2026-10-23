"""Authenticate, prepare and compare saved live captures on three frozen runtimes.

Full runs require Colab. This controller never trains, promotes or operates PokerNow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import math
import threading
import time
import zipfile
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'stage6_empirical_capture_protocol_v2'
ROLES = ('original', 'mechanics', 'experimental')
PREVIOUSLY_INSPECTED = {'pgl8cdDzWBGiScq2GsdVpPl6Q', 'pgl0pqNU8P-7SwPeHButJsg7H', 'pgl9JZecpbuv1adxf6MILRTDR'}
ACTIONS = ('fold', 'check', 'call', 'bet', 'raise', 'all_in')


def cpu_capacity(cgroup_root=Path('/sys/fs/cgroup')):
    """Actual affinity and cgroup CPU quota; memory is deliberately not an input."""
    capacity = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        capacity = min(capacity, len(os.sched_getaffinity(0)))
    root = Path(cgroup_root)
    v2 = root / 'cpu.max'
    if v2.is_file():
        quota, period = v2.read_text().split()[:2]
        if quota != 'max':
            capacity = min(capacity, max(1, math.floor(int(quota) / int(period))))
    else:
        for prefix in (root / 'cpu', root):
            quota, period = prefix / 'cpu.cfs_quota_us', prefix / 'cpu.cfs_period_us'
            if quota.is_file() and period.is_file() and int(quota.read_text()) > 0:
                capacity = min(capacity, max(1, math.floor(int(quota.read_text()) / int(period.read_text()))))
                break
    return max(1, min(48, capacity))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve existing sealed JSON: '+str(path))
    pending=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    if path.exists():raise FileExistsError('Sealed JSON appeared; preserve partial: '+str(path))
    pending.rename(path)


def copy_authenticated(source,target,expected):
    """Interrupted copies remain unaccepted unique partials and can be retried."""
    if target.exists():
        if target.stat().st_size!=expected['bytes'] or sha(target)!=expected['sha256']:
            raise ValueError('resumed_external_member_changed:'+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    pending=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
    shutil.copyfile(source,pending)
    if pending.stat().st_size!=expected['bytes'] or sha(pending)!=expected['sha256']:
        raise ValueError('copied_external_member_changed:'+str(target))
    if target.exists():raise FileExistsError('External member appeared; preserve partial')
    pending.rename(target)


def ensure_run_binding(output,run_identity,expected_sha):
    output.mkdir(parents=True,exist_ok=True)
    binding=output/'run_binding.json'
    expected={'run_identity':run_identity,'protocol_sha256':expected_sha}
    if binding.exists():
        if json.loads(binding.read_text())!=expected:raise ValueError('output_owned_by_different_frozen_run')
    else:
        if any(not path.is_file() or not path.name.startswith('run_binding.json.partial-') for path in output.iterdir()):
            raise ValueError('unowned_nonempty_output_directory')
        write_json(binding,expected)


def resolve_member(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError('member_outside_package:' + str(relative))
    return candidate


def authenticate(root, item):
    path = resolve_member(root, item['path'])
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError('identity_mismatch:' + item['path'])
    return path


def validate_protocol(protocol):
    if protocol.get('schema_version') != SCHEMA:
        raise ValueError('unsupported_capture_protocol')
    if set(protocol.get('snapshots', {})) != set(ROLES):
        raise ValueError('three_frozen_source_roles_required')
    games = protocol.get('games', [])
    if len({g['game_id'] for g in games}) != len(games):
        raise ValueError('duplicate_game_cluster')
    for game in games:
        if game['cohort'] not in ('previously_inspected_diagnostic', 'locked_available_diagnostic', 'held_unverifiable'):
            raise ValueError('invalid_or_unjustified_cohort')
        if game['game_id'] in PREVIOUSLY_INSPECTED and game['cohort'] != 'previously_inspected_diagnostic':
            raise ValueError('previously_inspected_game_cannot_be_confirmation')
        if not game.get('files'):
            raise ValueError('game_has_no_source_inventory')
        for item in game['files']:
            if item['role'] not in ('capture', 'session_log', 'direct_log', 'comparison_log'):
                raise ValueError('unsupported_capture_input_role')
    if not protocol.get('hero_pokernow_id'):
        raise ValueError('exact_hero_identity_required')
    # Name is a display setting, never an identity resolver.
    if not protocol.get('runtime_args') or 'maxStateAgeMs' not in protocol['runtime_args']:
        raise ValueError('frozen_runtime_settings_required')
    if not protocol.get('code_files') or not protocol.get('session_parser'):
        raise ValueError('exact_code_and_parser_inventory_required')
    capture_input_owners(protocol)


def capture_input_owners(protocol):
    owners={}
    for game in protocol['games']:
        for item in game['files']:
            if item['role']!='capture':continue
            name=item['path']
            if name in owners and owners[name]!=game['game_id']:
                raise ValueError('capture_path_has_ambiguous_game_ownership')
            owners[name]=game['game_id']
    return owners


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():raise FileExistsError('Preserve existing runtime tree')
    pending=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
    pending.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate_archive_member')
        for info in bundle.infolist():
            if '\\' in info.filename or info.filename.startswith('/') or ':' in info.filename:
                raise ValueError('nonportable_archive_member')
            resolve_member(pending, info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('archive_symlink_not_allowed')
        bundle.extractall(pending)
    verify_extracted(archive,pending)
    if destination.exists():raise FileExistsError('Runtime tree appeared; preserve partial')
    pending.rename(destination)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_extracted(archive, destination):
    """Authenticate resumed source bytes against the already authenticated archive."""
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            target = resolve_member(destination, info.filename)
            with bundle.open(info) as stream:
                expected = hashlib.file_digest(stream, 'sha256').hexdigest()
            if not target.is_file() or sha(target) != expected:
                raise ValueError('resumed_runtime_source_changed:' + info.filename)


def phase(output, name, config, expected_files, command_prefix, root, run_identity):
    receipt_path = output / (name + '_receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        seal = receipt.pop('self_sha256')
        if hashlib.sha256(canonical(receipt)).hexdigest() != seal or receipt['run_identity'] != run_identity:
            raise ValueError('phase_receipt_mismatch:' + name)
        return {item['name']: authenticate(output, item) for item in receipt['files']}
    phase_root = output / 'phases' / name
    phase_root.mkdir(parents=True, exist_ok=True)
    # Failed attempts remain intact. Only a sealed complete attempt can be reused.
    attempt = phase_root / ('attempt_%04d' % (1 + len(list(phase_root.iterdir()))))
    attempt.mkdir()
    config_path = attempt / 'config.json'
    write_json(config_path, config)
    subprocess.run([*command_prefix, 'prepare' if name == 'prepare' else 'replay', str(config_path), str(attempt)], check=True, cwd=root)
    files = []
    for filename in expected_files:
        p = attempt / filename
        files.append({'name': filename, 'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    receipt = {'schema_version': 'stage6_empirical_capture_phase_v2', 'run_identity': run_identity, 'phase': name, 'files': files}
    receipt['self_sha256'] = hashlib.sha256(canonical(receipt)).hexdigest()
    write_json(receipt_path, receipt)
    return {item['name']: output / item['path'] for item in files}


def adopt_frozen_preparation(root, output, protocol, run_identity):
    """Reuse the two exact accepted preparation outputs in a new run namespace."""
    reuse = protocol.get('preparation_reuse')
    if reuse is None:
        return
    source = authenticate(root, reuse['source_result'])
    files = reuse['files']
    if {item['name'] for item in files} != {'prepared_capture_records.jsonl', 'preparation.json'} or len(files) != 2:
        raise ValueError('reuse_exact_preparation_members_required')
    # All source identities are checked before creating an accepted successor receipt.
    with zipfile.ZipFile(source) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError('reuse_duplicate_archive_member')
        old_receipt = json.loads(bundle.read('prepare_receipt.json'))
        seal = old_receipt.pop('self_sha256')
        if hashlib.sha256(canonical(old_receipt)).hexdigest() != seal or old_receipt != reuse['source_phase']:
            raise ValueError('reuse_preparation_phase_changed')
        if old_receipt['files'] != files:
            raise ValueError('reuse_preparation_members_changed')
        for item in files:
            info = bundle.getinfo(item['path'])
            with bundle.open(info) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if info.file_size != item['bytes'] or actual != item['sha256']:
                raise ValueError('reuse_preparation_member_changed')
        receipt_path = output / 'prepare_receipt.json'
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text())
            expected_files = [{**item, 'path': 'phases/prepare/attempt_0001/' + item['name']} for item in files]
            if existing.get('files') != expected_files or existing.get('reused_from') != reuse['source_result']:
                raise ValueError('resumed_preparation_reuse_changed')
            return  # The normal phase() path verifies its seal, run ID and member bytes.
        destination = output / 'phases/prepare/attempt_0001'
        destination.mkdir(parents=True, exist_ok=True)
        accepted = []
        for item in files:
            target = destination / item['name']
            if not target.exists():
                pending = target.with_name(target.name + '.partial-' + uuid.uuid4().hex)
                with bundle.open(item['path']) as inp, pending.open('xb') as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
                    out.flush(); os.fsync(out.fileno())
                if pending.stat().st_size != item['bytes'] or sha(pending) != item['sha256']:
                    raise ValueError('copied_preparation_member_changed')
                if target.exists():
                    raise FileExistsError('Prepared target appeared; preserve partial')
                pending.rename(target)
            accepted.append({**item, 'path': target.relative_to(output).as_posix()})
            authenticate(output, accepted[-1])
    authenticate(root, reuse['source_result'])
    receipt = {'schema_version': 'stage6_empirical_capture_phase_v2', 'run_identity': run_identity,
               'phase': 'prepare', 'files': accepted, 'reused_from': reuse['source_result']}
    receipt['self_sha256'] = hashlib.sha256(canonical(receipt)).hexdigest()
    write_json(receipt_path, receipt)
    print(json.dumps({'stage': 'accepted_preparation_reused', 'members': len(accepted),
                      'source_result_sha256': reuse['source_result']['sha256']}), flush=True)


def action_family(row):
    action = str((row or {}).get('id', ''))
    if action.startswith('all_in') or action in ('allin', 'shove'):
        return 'all_in'
    return next((name for name in ACTIONS if action == name or action.startswith(name + '_')), 'unknown')


def finite_amount(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def recorded_all_in_bounds(row):
    """Use the exact public ledger and captured Hero, not recommendation sizing."""
    hero = (row.get('decision_context') or {}).get('hero') or {}
    actor = hero.get('visible_pokernow_player_id')
    ledger = row.get('decision_public_ledger') or {}
    if not actor or ledger.get('status') != 'READY_EXACT_PUBLIC_LEDGER':
        return None
    matches = [p for p in ledger.get('players', []) if p.get('player_id') == actor]
    if len(matches) != 1:
        return None
    player = matches[0]
    values = [finite_amount(player.get(key)) for key in
              ('remaining_bb', 'street_bb', 'start_raw', 'total_raw', 'street_raw')]
    stack = finite_amount(hero.get('stack_bb'))
    if stack is None or any(value is None for value in values):
        return None
    remaining, committed, start_raw, total_raw, street_raw = values
    if stack <= 0 or min(values) < 0 or abs(stack - remaining) > 1e-7:
        return None
    cap_bb = remaining + committed
    cap_raw = start_raw - total_raw + street_raw
    if cap_raw <= 0 or total_raw > start_raw + 1e-7 or street_raw > total_raw + 1e-7:
        return None
    return cap_bb, cap_raw


def recorded_option_is_legal(option, row):
    """An all-in label requires exact chips and an available action path."""
    if not isinstance(option, dict) or option.get('legal') is False:
        return False
    rec = row['recommendation']
    mask = rec.get('actual_legal_action_mask') or rec.get('legal_action_mask') or {}
    family = action_family(option)
    if family != 'all_in':
        return bool(mask.get('can_' + family, False))
    bounds = recorded_all_in_bounds(row)
    sizing = option.get('sizing') or {}
    amount_bb, amount_raw = (finite_amount(sizing.get(key)) for key in ('amount_bb', 'amount_raw'))
    if bounds is None or amount_bb is None or amount_raw is None:
        return False
    cap_bb, cap_raw = bounds
    if abs(amount_bb - cap_bb) > 1e-7 or abs(amount_raw - cap_raw) > 1e-7:
        return False
    if not any(mask.get(key, False) for key in ('can_all_in', 'can_bet', 'can_raise')):
        return False
    maximum = finite_amount(mask.get('max_raise_to_bb'))
    return maximum is None or amount_bb <= maximum + 1e-7


def family_distribution(recommendation):
    result = {a: 0.0 for a in ACTIONS}
    unknown = 0.0
    for row in recommendation.get('action_mix') or []:
        value = float(row.get('pct', 0)) / 100
        if not 0 <= value <= 1:
            raise ValueError('invalid_action_probability')
        family = action_family(row)
        if family == 'unknown':
            unknown += value
        else:
            result[family] += value
    total = sum(result.values()) + unknown
    if total and abs(total - 1) > 0.00001:
        raise ValueError('unnormalized_action_mix')
    return result, unknown


def clustered_interval(values_by_game, seed=6212026, draws=2000):
    """Row-weighted paired mean; resample complete game clusters, never rows."""
    groups = list(values_by_game.values())
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(map(sum, sample)) / sum(map(len, sample)))
    means.sort()
    return [means[int(.025 * draws)], means[min(draws - 1, int(.975 * draws))]]


def summarize(role_rows):
    index = {}
    for role in ROLES:
        rows = role_rows[role]
        if len({r['decision_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate_replayed_decision:' + role)
        index[role] = {r['decision_id']: r for r in rows}
    if len({frozenset(rows) for rows in index.values()}) != 1:
        raise ValueError('three_way_decision_ids_differ')
    report = {'roles': {}, 'paired_comparisons': {}, 'cohorts': {}, 'qualification': {
        'prediction_qualified': False, 'hero_strategy_qualified': False, 'promotion_allowed': False,
        'counterfactual_ev_identified': False, 'reason': 'Policy replay measures changed recommendations and observed action agreement. Human propensities, counterfactual outcomes and calibrated continuation models are absent.'}}
    for role, rows in role_rows.items():
        ready = [r for r in rows if r['status'] == 'REPLAYED']
        latencies = sorted(r['elapsed_ms'] for r in ready)
        actions, sources, reasons, legal_failures = Counter(), Counter(), Counter(), 0
        illegal_mix=[]
        for row in ready:
            rec = row['recommendation']; action = action_family(rec.get('primary_action'))
            actions[action] += 1; sources[str(rec.get('source'))] += 1; reasons[str(rec.get('reason'))] += 1
            if action != 'unknown' and not recorded_option_is_legal(rec.get('primary_action'), row):
                legal_failures += 1
            family_distribution(rec)
            illegal_mass=sum(float(option.get('pct',0))/100 for option in rec.get('action_mix') or []
                if float(option.get('pct',0))>0 and not recorded_option_is_legal(option,row))
            if illegal_mass>0:illegal_mix.append({'decision_id':row['decision_id'],'illegal_probability_mass':illegal_mass})
            if not row.get('main_left_equal') or row['main_recommendation'] != row['left_recommendation']:
                raise ValueError('main_left_parity_failed:' + row['decision_id'])
        report['roles'][role] = {'rows': len(rows), 'replayed': len(ready), 'runtime_errors': len(rows) - len(ready),
            'action_counts': dict(actions), 'source_counts': dict(sources), 'reason_counts': dict(reasons), 'illegal_primary_actions': legal_failures,
            'illegal_action_mix_decisions':len(illegal_mix),'illegal_action_mix':illegal_mix,
            'latency_ms': {'median': statistics.median(latencies) if latencies else None, 'p95': latencies[min(len(latencies)-1, int(.95*len(latencies)))] if latencies else None},
            'latency_scope': 'Offline sequential reconstruction and recommendation only; excludes live ingestion, asynchronous model inference, UI and network.',
            'main_left_equal': True, 'live_lifecycle_verified': False}
    for base, candidate in (('original', 'mechanics'), ('mechanics', 'experimental'), ('original', 'experimental')):
        differences = []; agreement_delta = defaultdict(list)
        for key, left in index[base].items():
            right = index[candidate][key]
            if left['status'] != 'REPLAYED' or right['status'] != 'REPLAYED':
                continue
            a, b = left['recommendation'], right['recommendation']
            ap, au = family_distribution(a); bp, bu = family_distribution(b)
            # Preserve full concrete-size distributions even when family weights agree.
            changes = {field: a.get(field) != b.get(field) for field in ('status', 'primary_action', 'action_mix', 'sizing', 'gate_checks', 'decision_context', 'stage021_primary_context')}
            differences.append({'decision_id': key, 'game_id': left['game_id'], 'changes': changes,
                'family_total_variation': .5 * (sum(abs(ap[x] - bp[x]) for x in ACTIONS) + abs(au - bu))})
            label = left.get('observed_action')
            agreement_delta[left['game_id']].append(float(action_family(b.get('primary_action')) == label) - float(action_family(a.get('primary_action')) == label))
        flat = [x for rows in agreement_delta.values() for x in rows]
        report['paired_comparisons'][base + '_to_' + candidate] = {'paired_rows': len(differences), 'game_clusters': len(agreement_delta),
            'changed_fields': {field: sum(r['changes'][field] for r in differences) for field in ('status', 'primary_action', 'action_mix', 'sizing', 'gate_checks', 'decision_context', 'stage021_primary_context')},
            'observed_action_agreement_delta': statistics.mean(flat) if flat else None,
            'agreement_delta_game_bootstrap_95_ci': clustered_interval(agreement_delta),
            'agreement_is_strategy_value': False, 'differences': differences}
    cohorts = defaultdict(set)
    for row in role_rows['original']:
        if row.get('cohort'):
            cohorts[row['cohort']].add(row['game_id'])
    report['cohorts'] = {name: sorted(games) for name, games in cohorts.items()}
    report['disposition'] = 'READY_DIAGNOSTIC_COMPARISON' if all(r['replayed'] and not r['runtime_errors'] and not r['illegal_primary_actions'] and not r['illegal_action_mix_decisions'] for r in report['roles'].values()) else 'HELD_INCOMPLETE_REPLAY'
    return report


def run(protocol_path, expected_sha, root, output, node='node', tsx_import='tsx'):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    if sha(protocol_path) != expected_sha:
        raise ValueError('protocol_sha256_mismatch')
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    validate_protocol(protocol)
    if not (os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU') or Path('/content').is_dir()) and not protocol.get('bounded_fixture'):
        raise RuntimeError('Full capture preparation and replay must run in Colab.')
    root, output = Path(root).resolve(), Path(output).resolve()
    identities = protocol['code_files'] + [protocol['session_parser']]
    identities += [f for g in protocol['games'] for f in g['files']]
    identities += [s['archive'] for s in protocol['snapshots'].values()]
    identities += [f for s in protocol['snapshots'].values() for f in s.get('external_members', [])]
    if protocol.get('preparation_reuse'):
        identities.append(protocol['preparation_reuse']['source_result'])
    for item in identities:
        authenticate(root, item)
    script = Path(__file__).with_suffix('.mjs').resolve()
    own_files = {Path(item['path']).name for item in protocol['code_files']}
    if Path(__file__).name not in own_files or script.name not in own_files:
        raise ValueError('controller_and_node_driver_must_be_bound')
    for own in (Path(__file__).resolve(), script):
        matches = [item for item in protocol['code_files'] if (root / item['path']).resolve() == own]
        if len(matches) != 1:
            raise ValueError('executing_code_not_in_frozen_inventory')
    run_identity = hashlib.sha256(canonical({'protocol_sha256': expected_sha, 'inputs': identities})).hexdigest()
    ensure_run_binding(output,run_identity,expected_sha)
    runtimes = {}
    for role, snapshot in protocol['snapshots'].items():
        # Runtime source trees and node_modules links live on the local disk;
        # Drive FUSE does not reliably support symlinks. Only accepted phase
        # results/receipts are durable in OUTPUT and survive a fresh Colab VM.
        runtime = root / '.capture_runtime_sources' / run_identity / role
        archive = authenticate(root, snapshot['archive'])
        if runtime.exists():
            verify_extracted(archive, runtime)
        else:
            safe_extract(archive, runtime)
        for member in snapshot.get('external_members', []):
            source = authenticate(root, member)
            target = resolve_member(runtime, member['runtime_path'])
            copy_authenticated(source,target,member)
        # Node resolves dependencies relative to the imported source. This link is
        # local to the new output tree; source snapshots and system installs stay intact.
        modules = root / 'node_modules'
        if not modules.is_dir():
            raise FileNotFoundError('Install pinned Node dependencies in package_root/node_modules before execution.')
        if not (runtime / 'node_modules').exists():
            (runtime / 'node_modules').symlink_to(modules, target_is_directory=True)
        elif (runtime / 'node_modules').resolve() != modules.resolve():
            raise ValueError('resumed_node_dependencies_changed')
        runtimes[role] = str(runtime)
    common = {key: protocol[key] for key in ('games', 'hero_name', 'hero_pokernow_id', 'runtime_args', 'session_parser')}
    common.update(package_root=str(root), preparation_runtime_root=runtimes['mechanics'], bounded_fixture=protocol.get('bounded_fixture', False))
    prefix = [node, '--import', tsx_import, str(script)]
    adopt_frozen_preparation(root, output, protocol, run_identity)
    preparation = phase(output, 'prepare', common, ['prepared_capture_records.jsonl', 'preparation.json'], prefix, root, run_identity)
    records = preparation['prepared_capture_records.jsonl']
    role_rows = {}
    result_files = [*preparation.values()]
    def replay_role(role):
        config = {**common, 'role': role, 'runtime_root': runtimes[role], 'records_path': str(records)}
        result = phase(output, role, config, [role + '_replay.jsonl'], prefix, root, run_identity)
        return role, result
    capacity = cpu_capacity()
    requested = protocol.get('workers', 'auto')
    if requested != 'auto' and (not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= capacity):
        raise ValueError('worker_count_outside_cpu_cap')
    workers = min(len(ROLES), min(2, capacity) if requested == 'auto' else requested)
    if workers > 2 and not protocol.get('accepted_higher_worker_parity_receipt'):
        raise ValueError('higher_workers_require_accepted_parity_receipt')
    # Independent complete runtime lanes, one process per role and disjoint receipts.
    # Deterministic reduction stays in ROLES order. RAM never chooses worker counts.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(replay_role, ROLES))
    for role, result in results:
        role_rows[role] = read_rows(result[role + '_replay.jsonl'])
        result_files.extend(result.values())
    # Two-pass input identity checks surround processing; immutable input receipts
    # remain independent from future Drive transport receipts.
    for item in identities:
        authenticate(root, item)
    report = summarize(role_rows)
    report.update(schema_version='stage6_empirical_capture_comparison_v2', protocol_sha256=expected_sha,
        created_utc=datetime.now(timezone.utc).isoformat(), snapshots=protocol['snapshots'], source_bytes_stable=True,
        workers={'requested': requested, 'resolved': workers, 'cpu_cap': capacity, 'ram_used_as_gate': False},
        files={p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in result_files})
    report['self_sha256'] = hashlib.sha256(canonical(report)).hexdigest()
    report_path = output / 'capture_comparison_report.json'
    if report_path.exists():
        existing = json.loads(report_path.read_text()); seal = existing.pop('self_sha256')
        if hashlib.sha256(canonical(existing)).hexdigest() != seal or existing['protocol_sha256'] != expected_sha or existing['files'] != report['files']:
            raise ValueError('resumed_final_report_mismatch')
    else:
        write_json(report_path, report)
    print(json.dumps({'stage': 'complete', 'disposition': report['disposition'], 'roles': report['roles']}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', default='node')
    parser.add_argument('--tsx-import', default='tsx')
    args = parser.parse_args()
    stop=threading.Event();started=time.monotonic()
    def heartbeat():
        while not stop.wait(20):
            print(json.dumps({'stage':'capture_controller_authentication_or_phase_execution','elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        run(args.protocol, args.protocol_sha256, args.package_root, args.output, args.node, args.tsx_import)
    finally:
        stop.set();thread.join()


if __name__ == '__main__':
    main()
