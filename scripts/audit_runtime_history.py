"""Read-only version/role audit of all retained Assistant Testing comparison logs."""
from pathlib import Path
import json,hashlib,collections,argparse

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--audit-dir', type=Path, required=True, help='Private output directory outside this release')
    ap.add_argument('--aggregate-output', type=Path, required=True)
    args=ap.parse_args();ROOT=args.workspace.resolve();OUT=args.audit_dir.resolve()
    release=Path(__file__).resolve().parents[1]
    if OUT.is_relative_to(release):ap.error('--audit-dir must be outside the public release; it contains private examples')
    if not (ROOT/'data/exports/ml_runtime/stage016/live_chrome').is_dir():ap.error('Private capture root is missing')
    OUT.mkdir(parents=True,exist_ok=True);sessions=[];examples=[];errors=[];formula_max_error=0.;first_fusion=last_fusion=None;all_six_records=0
    for p in sorted((ROOT/'data/exports/ml_runtime/stage016/live_chrome').glob('*/assistant_testing_comparisons.jsonl'),key=lambda p:p.parent.name):
        counts=collections.defaultdict(collections.Counter);n=0;diffhands=set();primaryhands=set();fusedhands=set();layercounts=collections.Counter();diff=primary=fused=fused_diff=fused_primary=0;first=last=None
        for lineno,line in enumerate(p.open(encoding='utf-8'),1):
            try:d=json.loads(line)
            except Exception as e:errors.append({'path':str(p.relative_to(ROOT)),'line':lineno,'error':str(e)});continue
            n+=1;first=first or d.get('observed_at_utc');last=d.get('observed_at_utc');f=d.get('experimental_fusion') or {};t=d.get('testing') or {};i=d.get('incumbent') or {};h=(d.get('pokernow_game_id'),d.get('pokernow_hand_id'))
            for name,obj,fields in [('fusion',f,['status','reason','qualified_base_weight','shadow_weight','predictive_action_weight','range_strategy_weight','recommendation_fusion_qualified']),('incumbent',i,['source','status']),('testing',t,['source','status','recommendation_difference_reason']),('assessment',d.get('empirical_assessment') or {},['status','reason']),('inference',d.get('shadow_inference') or {},['status','reason'])]:
                for k in fields:
                    if k in obj:counts[name+'.'+k][str(obj[k])]+=1
            if t.get('recommendation_difference') is True:diff+=1;diffhands.add(h)
            if t.get('primary_recommendation_difference') is True:primary+=1;primaryhands.add(h)
            if t.get('source')=='stage6_assistant_testing_qualified_base_shadow_fusion_v2':
                fused+=1;fusedhands.add(h);fused_diff+=t.get('recommendation_difference') is True;fused_primary+=t.get('primary_recommendation_difference') is True
                first_fusion=first_fusion or d.get('observed_at_utc');last_fusion=d.get('observed_at_utc')
                all_six_records+=len(f.get('supported_layers',[]))==6
                for family,value in f['target_family_pct'].items():
                    reconstructed=0.25*f['incumbent_family_pct'][family]+25*f['postflop_behavior_represented_probability'][family]+50*f['shadow_mean_represented_probability'][family]
                    formula_max_error=max(formula_max_error,abs(value-reconstructed))
                for x in f.get('supported_layers',[]):layercounts[str(x)]+=1
                if len([x for x in examples if x['session']==p.parent.name])<2 and t.get('primary_recommendation_difference'):
                    examples.append({'session':p.parent.name,'line':lineno,'observed_at_utc':d.get('observed_at_utc'),'game':h[0],'hand':h[1],'state_sequence':d.get('state_sequence'),'incumbent':i,'testing':t,'fusion':f,'shadow_layers':d.get('shadow_layers'),'shadow_inference':d.get('shadow_inference')})
            counts['premium_squeeze.applied'][str((d.get('premium_squeeze_proposal') or {}).get('applied'))]+=1
        sessions.append({'session':p.parent.name,'date_utc':p.parent.name.split('-')[2],'source_path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size,'records':n,'first_observed_utc':first,'last_observed_utc':last,'counts':{k:dict(v) for k,v in counts.items()},'fusion_source_records':fused,'fusion_distribution_difference_records':fused_diff,'fusion_primary_difference_records':fused_primary,'fusion_distinct_game_hands':len(fusedhands),'supported_layer_record_counts':dict(layercounts),'distribution_difference_records':diff,'distribution_difference_distinct_game_hands':len(diffhands),'primary_difference_records':primary,'primary_difference_distinct_game_hands':len(primaryhands)})
    report={'sessions':sessions,'session_count':len(sessions),'total_records':sum(s['records'] for s in sessions),'errors':errors,'note':'Repeated telemetry records, not independent decisions. Hand keys can be missing; inspect examples before interpreting counts. No outcome/adherence attribution.'}
    (OUT/'all_sessions_private.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');(OUT/'historical_fusion_examples_private.json').write_text(json.dumps(examples,indent=2)+'\n',encoding='utf-8')
    legacy=[s for s in sessions if s['fusion_source_records']];layers=collections.Counter()
    for s in legacy:layers.update(s['supported_layer_record_counts'])
    public={'scope':'all retained primary live_chrome Assistant Testing comparison logs; immutable read-only snapshot','session_count':len(sessions),'total_records':report['total_records'],'parse_errors':len(errors),'first_fusion_observed_utc':first_fusion,'last_fusion_observed_utc':last_fusion,'legacy_fusion':{'source':'stage6_assistant_testing_qualified_base_shadow_fusion_v2','records':sum(s['fusion_source_records'] for s in legacy),'distribution_difference_records':sum(s['fusion_distribution_difference_records'] for s in legacy),'primary_action_difference_records':sum(s['fusion_primary_difference_records'] for s in legacy),'postflop_base_records':sum(s['fusion_source_records'] for s in legacy),'supported_shadow_layer_records':dict(layers),'all_six_supported_shadow_records':all_six_records,'nominal_target_weights':{'stage021_incumbent':0.25,'postflop_behavior':0.25,'supported_shadow_mean':0.5},'maximum_logged_target_reconstruction_error_percentage_points':formula_max_error,'excluded_layers':['opponent_bluff_likelihood','showdown_aware_range_calibration']},'sessions':[{'session_start_utc_token':s['date_utc'],'records':s['records'],'sha256':s['sha256'],'fusion_records':s['fusion_source_records'],'distribution_difference_records':s['distribution_difference_records'],'primary_difference_records':s['primary_difference_records'],'counts':s['counts']} for s in sessions],'interpretation':'Telemetry records can repeat decisions. Changes in displayed experimental advice do not establish adherence, better actions, or winnings. One legacy record lacks a hand identifier; no hand count is asserted.'}
    args.aggregate_output.parent.mkdir(parents=True,exist_ok=True)
    args.aggregate_output.write_text(json.dumps(public,indent=2)+'\n',encoding='utf-8')
    for s in sessions:print(s['date_utc'],s['records'],'fusion',s['fusion_source_records'],'mix differences',s['distribution_difference_records'],'primary differences',s['primary_difference_records'],'fused hand keys',s['fusion_distinct_game_hands'])
    print('TOTAL',report['total_records'],'ERRORS',len(errors))
if __name__=='__main__':main()
