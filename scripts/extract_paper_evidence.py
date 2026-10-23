"""Build compact, identifier-free paper data from preserved local results."""
from pathlib import Path
import json,csv,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
S7=ROOT/'Poker Assistant Model/Stage 7 - Empirical Assistant Layers'
S6=ROOT/'Poker Assistant Model/Stage 6 - Qualified Layers Improvement'
sources=[]
def read(p):
    b=p.read_bytes();sources.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});return json.loads(b)
def write(name,d): (OUT/name).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def main():
    d=read(S7/'returned_results/01_completed_r2/scores/report.json');layers=[]
    for name,v in d['layers'].items():
        m=v['metrics']['confirmation'];a=m['frozen_prior'];b=m['selected_postflop_predictor']
        ref=read(ROOT/f'Assistant Qualified/Assistant Code/runtime/ml/artifacts/stage6_retrospective_nested_v2/{name}/refit_receipt.json')
        layers.append({'layer':name,'family':ref['family'],'candidate_id':ref['candidate_id'],'iterations':ref['best_iteration'],'features':ref['feature_count'],'fit_rows':ref['fit_row_count'],'calibration_rows':ref['calibration_row_count'],'confirmation_rows':a['rows'],'games':a['game_clusters'],'accuracy':a['accuracy'],'log_loss':a['log_loss'],'prior_log_loss':a['baseline_log_loss'],'improvement_prior':a['log_loss_improvement'],'interval_prior':a['log_loss_improvement_interval'],'improvement_postflop':b['log_loss_improvement'],'interval_postflop':b['log_loss_improvement_interval'],'interval_alpha':a['interval_alpha'],'brier':a['brier'],'ece':a['ece_10_bins'],'disposition':v['qualification']['disposition']})
    write('layer_metrics.json',{'schema':'paper_layer_metrics_v1','evaluation':'locked retrospective unseen games','rows':d['rows'],'game_clusters':d['game_clusters'],'cohort_rows':d['cohort_rows'],'minimum_decision_utc':d['minimum_decision_utc'],'maximum_decision_utc':d['maximum_decision_utc'],'layers':layers,'interpretation':'Observed-actor prediction, not Hero policy value. Support differs by comparison.'})
    d=read(S7/'returned_results/03_range_r2/range_response_report.json');roles={}
    for k,v in d['roles'].items():
        e=v['evaluation'];roles[k]={**{k:v for k,v in e.items() if not isinstance(v,(dict,list))},'lower_bounds':e['whole_game_lower_bounds'],'training_rows':v['training_rows'],'training_games':v['training_games']}
    write('range_metrics.json',{'roles':roles,'range_ablation':d['range_diagnostics']['summary'],'sampled_rows':d['range_diagnostics']['sampled_rows'],'sampled_games':d['range_diagnostics']['sampled_games'],'source_support':{k:v for k,v in d['source_support'].items() if not isinstance(v,(dict,list))}})
    d=read(S7/'runtime_state/notebook05_held_diagnostic_summary_r1.json');accept=read(S7/'manifests/stage7_05_returned_result_audit_acceptance_r1.json');rows=[]
    for w in d['windows']:
        for role,r in w['roles'].items():
            for comparison,m in r['fixed_comparisons'].items():rows.append({'window':w['window'],'role':role,'comparison':comparison,'rows':m['rows'],'games':m['games'],'accuracy':m['accuracy'],'log_loss':m['log_loss'],'improvement':m['log_loss_improvement'],'lower_bound':m['whole_game_lower_bounds']['log_loss'],'brier_improvement':m['brier_improvement'],'brier_lower_bound':m['whole_game_lower_bounds']['brier']})
    write('history_ablations.json',{'status':'ACCEPT_FOR_DIAGNOSTIC_REVIEW','evidence_class':d['evidence_class'],'cohorts':d['protocol_cohort_counts'],'comparisons':rows,'note':'Later acceptance receipt supersedes pending labels in stored summary. Intervals are descriptive, not family-wide corrected across windows/ablations.'})
    d=read(S7/'returned_results/04_terminal_r7/recomputed_terminal_report.json');write('terminal_coverage.json',{k:d[k] for k in ['disposition','held','supported','records','reasons'] if k in d})
    d=read(ROOT/'research_organization/outcome_audit_summary.json');write('captured_outcome_coverage.json',{k:v for k,v in d.items() if k not in ['hero_id','identity_basis','unparsed_money_examples']})
    p=ROOT/'research_organization/captured_hand_outcomes_private.csv';b=p.read_bytes();sources.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    r=list(csv.DictReader(p.open(encoding='utf-8')));group=collections.defaultdict(list)
    for x in r:group[x['game_id']].append(x)
    summary=[]
    for i,(game,rr) in enumerate(sorted(group.items(),key=lambda kv:min(x['start_utc'] for x in kv[1]))):
        ok=[x for x in rr if x['eligible']=='True'];summary.append({'session_group':f'G{i+1:02d}','candidate_hands':len(rr),'reconciled_hands':len(ok),'net_bb_subset':sum(float(x['net_bb']) for x in ok),'positive_hands':sum(float(x['net_chips'])>0 for x in ok),'negative_hands':sum(float(x['net_chips'])<0 for x in ok),'zero_hands':sum(float(x['net_chips'])==0 for x in ok)})
    write('captured_game_aggregates.json',{'note':'Anonymous captured-hand groups; selected subset, not full-session results or treatment comparisons.','groups':summary})
    write('source_provenance.json',{'source_root':'original private workspace (not bundled)','sources':sources,'transform':'extract_paper_evidence.py; aggregate fields only; original files unchanged'})
    print('Wrote',len(list(OUT.glob('*.json'))),'compact evidence files')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();ROOT=args.workspace.resolve();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    S7=ROOT/'Poker Assistant Model/Stage 7 - Empirical Assistant Layers';S6=ROOT/'Poker Assistant Model/Stage 6 - Qualified Layers Improvement'
    main()
