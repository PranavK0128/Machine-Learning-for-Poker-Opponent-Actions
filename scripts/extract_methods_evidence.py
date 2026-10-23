"""Extract aggregate methods receipts from an authorized private workspace."""
from pathlib import Path
import argparse,hashlib,json,collections
import pyarrow.parquet as pq

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.workspace.resolve();sources=[]
    def read(p):
        sources.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
        return json.loads(p.read_text(encoding='utf-8'))
    materialization=root/'Poker Assistant Model/Stage 6 - Qualified Layers Improvement/runtime_state/notebook01/materializations/cbb5f9ca2799df0e54f7925197f2816bd34f0b3b25a5a5912665ca382b436dba'
    manifest=read(materialization/'stage6_notebook01_materialization_manifest.json')
    games=set();actors=set();hands={};minimum=None;maximum=None;rows=0
    for p in sorted((materialization/'lanes').glob('lane-*/accepted-decisions.parquet')):
        table=pq.read_table(p,columns=['game_id','hand_id','actor_id','decision_utc'])
        frame=table.to_pandas();rows+=len(frame)
        games.update(frame.game_id.astype(str));actors.update(frame.actor_id.astype(str))
        lo=str(frame.decision_utc.min());hi=str(frame.decision_utc.max())
        minimum=min(minimum,lo) if minimum else lo;maximum=max(maximum,hi) if maximum else hi
        for (g,h),r in frame.groupby(['game_id','hand_id']).decision_utc.agg(['max','size']).iterrows():
            old=hands.get((g,h));hands[(g,h)]=(max(old[0],str(r['max'])) if old else str(r['max']),int(r['size'])+(old[1] if old else 0))
    assert rows==manifest['accepted_decision_row_count']
    assert len(hands)==manifest['split_assignment_hand_count']
    roles=collections.Counter()
    for t,n in hands.values():
        role='development_train' if t<'2026-08-01T00:00:00.000Z' else 'development_validation' if t<'2026-08-16T00:00:00.000Z' else 'retrospective_calibration' if t<'2026-08-21T00:00:00.000Z' else 'fresh_adjudication'
        roles[role]+=n
    modelroot=root/'Assistant Qualified/Assistant Code/runtime/ml/artifacts/stage6_retrospective_nested_v2'
    models=[];additional=None
    for p in sorted(modelroot.glob('*/refit_receipt.json')):
        receipt=read(p);schema=read(p.parent/'feature_schema.json');support=read(p.parent/'support_contract.json');calibration=read(p.parent/'calibration.json')
        fields=['layer','family','candidate_id','parameters','best_iteration','feature_count','fit_roles','fit_row_count','calibration_role','calibration_row_count','class_counts_fit','class_priors_laplace_one','qualification_fold_temperature','refit_temperature','selection_fold','selection_rule','elapsed_seconds','candidate_internal_threads']
        model={k:receipt.get(k) for k in fields};model['feature_schema']=schema;model['support_contract']=support;model['calibration_artifact']=calibration
        report=receipt['calibration_report'];model['calibration_receipt_mode']='retrospective_calibration_grid' if 'temperature_grid' in report else 'frozen_inner_temperature_diagnostic'
        model['calibration_grid']=report.get('temperature_grid');model['receipt_temperature_matches_qualification']=receipt['refit_temperature']==receipt['qualification_fold_temperature']
        models.append(model)
        if 'materialization_counts' in receipt['source_binding']:
            if additional is not None:assert additional==receipt['source_binding']['materialization_counts']
            additional=receipt['source_binding']['materialization_counts']
    result={'schema':'paper_methods_evidence_v1','data':{
        'source_decision_rows':manifest['source_decision_row_count'],'accepted_decision_rows':rows,
        'quarantined_rows':manifest['quarantined_decision_row_count'],'quarantine_reasons':manifest['quarantine_reason_counts'],
        'accepted_game_count':len(games),'accepted_hand_count':len(hands),'accepted_actor_count':len(actors),
        'minimum_decision_utc':minimum,'maximum_decision_utc':maximum,'whole_hand_role_rows':dict(roles),
        'source_identities':manifest['run_binding']['source_identities'],'additional_materialization_counts':additional,
        'materialization_resource_observation':manifest['resource_admission'],
        'interpretation':'Decision rows repeat players and hands. Canonical identity groups were resolved retrospectively. Historical feature values use earlier completed hands.'},
        'models':models,'calibration_provenance':{'grid_receipts':sum(m['calibration_receipt_mode']=='retrospective_calibration_grid' for m in models),
        'frozen_diagnostic_receipts':sum(m['calibration_receipt_mode']=='frozen_inner_temperature_diagnostic' for m in models),
        'all_installed_temperatures_match_qualification':all(m['receipt_temperature_matches_qualification'] for m in models),
        'interpretation':'Retained receipts take precedence over later source revisions when describing executed work.'},'sources':sources}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result['data'].items() if k not in ['source_identities','additional_materialization_counts']},indent=2));print(result['calibration_provenance'])

if __name__=='__main__':main()
