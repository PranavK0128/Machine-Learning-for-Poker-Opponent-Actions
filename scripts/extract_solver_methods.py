"""Summarize solver design and one hash-verified completed root from private receipts."""
from pathlib import Path
import argparse,collections,hashlib,json,zipfile

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    pub=Path(__file__).resolve().parents[1]
    manifest=json.loads((pub/'results/stage1_river_strict_2x2_substitution_manifest.json').read_text())
    contracts=pub/'src/solver/river_r22/contracts'
    registry=json.loads((contracts/'root_registry.json').read_text());execution=json.loads((contracts/'execution_manifest.json').read_text())
    selected_ids={r['root_id'] for r in manifest['stage1_roots']};selected=[r for r in registry['roots'] if r['root_id'] in selected_ids]
    assert len(selected)==67
    def inventory(rows):
        return {k:dict(collections.Counter(str(r[k]) for r in rows)) for k in ['root_kind','range_family','opponent_range_family','stack_anchor_bb','action_size_family']}
    result={'schema':'paper_solver_methods_v1','registry_semantic_sha256':registry['semantic_sha256'],
            'full_registry':inventory(registry['roots']),'selected_corrected_roots':inventory(selected),
            'planned_execution':{k:execution[k] for k in ['science_seeds','approximate_base_iterations_per_tier','allowed_approximate_tiers','default_approximate_tier','checkpoint_interval_iterations','checkpoint_generations_retained_per_lane','evaluation_deals_per_family_role','evaluation_panel_count','crossplay_deals_per_assignment','restricted_exact_combos_per_player','restricted_exact_terminal_iterations_per_lane']},
            'case_study':{'root_id':'S1-RV-0001','selection':'illustrative completed root, not a representative sample or total compute estimate','lanes':[]}}
    source=args.workspace/'Poker Assistant Model/Stage 1 - River CFR and DCFR/returned_results/POKERNOW_ASSISTANT_STAGE1_A01_RESULTS.zip'
    expected_archive=next(a for a in manifest['source_archives'] if a['token']=='A01')
    assert sha(source)==expected_archive['sha256']
    root=next(r for r in manifest['stage1_roots'] if r['root_id']=='S1-RV-0001')
    with zipfile.ZipFile(source) as z:
        for receipt in root['selected_lane_receipts']:
            raw=z.read(receipt['source_member']);assert hashlib.sha256(raw).hexdigest()==receipt['source_member_sha256']
            d=json.loads(raw);solver=d['solver_result'];checkpoint=d['checkpoint_lifecycle']
            result['case_study']['lanes'].append({'lane_id':d['lane_id'],'algorithm':d['algorithm'],'seed':d['seed'],
                'requested_iterations':d['requested_target_iterations'],'actual_iterations':solver['iterations'],'node_count':solver['node_count'],
                'qualified':d['qualified'],'stopped_early_after_qualification':d['stopped_early_after_qualification'],
                'checkpoint_interval_iterations':checkpoint['checkpoint_interval_iterations'],
                'solver_contract_sha256':solver['solver_contract_sha256'],'source_member':receipt['source_member'],
                'source_member_sha256':receipt['source_member_sha256']})
    result['case_study']['archive_sha256']=expected_archive['sha256']
    result['limitations']=['Root budgets are configured limits, not measured total compute.','Numeric labels in restricted-exact runs encode deterministic accumulation-order checks rather than independent random replications.','No verified project-wide wall-time, GPU-hour or monetary-cost total is available.']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'selected':result['selected_corrected_roots'],'case_study_lanes_verified':len(result['case_study']['lanes'])},indent=2))

if __name__=='__main__':main()
