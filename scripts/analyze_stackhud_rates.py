"""Compute anonymized ledger/hand-time rates from an authorized private DB export.

The export schema is documented in docs/WIN_RATE_METHOD.md. No connection is
made by this script. Aggregate results are descriptive, not a treatment effect.
"""
from pathlib import Path
from decimal import Decimal
import argparse, hashlib, json
import numpy as np


def union_ms(intervals):
    merged=[]
    for a,b in sorted(intervals):
        assert b>=a
        if merged and a<=merged[-1][1]: merged[-1][1]=max(b,merged[-1][1])
        else: merged.append([a,b])
    return sum(b-a for a,b in merged)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--time-correction',type=Path,help='Author-supplied aggregate hours; retained separately from reconstructed per-game clocks')
    args=ap.parse_args();d=json.loads(args.input.read_text(encoding='utf-8'))
    financial={r['pokernowGameId']:r for r in d['financial']}
    ledger={r['pokernowGameId']:r for r in d['ledger_player']}
    checks={r['pokernowGameId']:r for r in d['game_sequence_checks']}
    assert len(financial)==len(d['financial'])==len(d['games'])==len(ledger)
    rows=[];included_spans=[];all_spans=[]
    for i,g in enumerate(d['games'],1):
        k=g['pokernowGameId'];f=financial[k];q=checks[k]
        hs=sorted([h for h in d['hero_hand_times'] if h['pokernowGameId']==k],key=lambda h:int(h['startedAt']))
        assert len(hs)==len({h['pokernowHandId'] for h in hs})>0
        assert g['handCount']==g['actual_hand_rows']==g['unique_hands']==q['distinct_numbers']
        assert int(q['last_number'])-int(q['first_number'])+1==q['distinct_numbers'], 'Game has missing hand numbers'
        assert int(q['first_number'])==1, 'Game does not start at hand one'
        assert ledger[k]['netMatchesLedger']
        assert Decimal(str(ledger[k]['totalNetRaw']))==Decimal(str(f['rawFinancialResult']['totalNetRaw']))
        spans=[];previous=None
        for h in hs:
            a=int(h['startedAt']);b=int(h['last_event'])
            assert a<=b and h['hero_actions']>0 and h['stackRaw']>0
            assert h['next_start'] is None or b<int(h['next_start'])
            if previous and previous['next_start']==h['startedAt']:
                spans[-1][1]=b
            else: spans.append([a,b])
            previous=h
        reasons=[]
        hero_blinds=sorted({h['bigBlindRaw'] for h in hs})
        hero_cents={h['cents'] for h in hs}
        if len(hero_blinds)!=1 or len(hero_cents)!=1: reasons.append('variable_stakes_during_hero_participation')
        if f['confidence']!='ledger_verified' or not f['rawFinancialResult']['ledgerImportBalanced']: reasons.append('unbalanced_nit_suspect_ledger')
        bb=hero_blinds[0] if not 'variable_stakes_during_hero_participation' in reasons else None
        net_bb=float(Decimal(str(f['rawFinancialResult']['totalNetRaw']))/Decimal(bb)) if bb else None
        if bb: assert abs(net_bb-float(f['netProfitBb']))<1e-6
        active_ms=sum(b-a for a,b in spans)
        hand_ms=sum(int(h['last_event'])-int(h['startedAt']) for h in hs)
        envelope_ms=int(hs[-1]['last_event'])-int(hs[0]['startedAt'])
        row={'game':f'G{i:02d}','included':not reasons,'exclusion_reasons':reasons,
             'hero_hands':len(hs),'game_hands':g['actual_hand_rows'],'participation_episodes':len(spans),
             'active_table_hours':active_ms/3600000,'hand_event_hours':hand_ms/3600000,
             'first_to_last_hand_hours':envelope_ms/3600000,'net_bb':net_bb,
             'ledger_confidence':f['confidence'],'hand_numbers_contiguous_from_one':True}
        row['table_changed_blinds_outside_hero_participation']=len(g['observed_big_blinds'])>1 and len(hero_blinds)==1
        rows.append(row);all_spans.extend(spans)
        if not reasons: included_spans.extend(spans)
    included=[r for r in rows if r['included']]
    net=sum(Decimal(str(r['net_bb'])) for r in included)
    hours=sum(r['active_table_hours'] for r in included)
    handhours=sum(r['hand_event_hours'] for r in included)
    envelope=sum(r['first_to_last_hand_hours'] for r in included)
    rng=np.random.default_rng(261023);draws=rng.integers(0,len(included),size=(20000,len(included)))
    gains=np.array([r['net_bb'] for r in included]);times=np.array([r['active_table_hours'] for r in included])
    boot=gains[draws].sum(axis=1)/times[draws].sum(axis=1)
    interval=np.quantile(boot,[.025,.975]).tolist()
    output={'schema':'stackhud_hand_time_rate_v1','snapshot_utc':d['snapshot']['snapshot_utc'],
      'source_export_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
      'cohort_rule':'All distinct games in retained Assistant session inventory; identity bound to the user-confirmed exact PokerNow ID. This is not a continuous Assistant-exposure cohort.',
      'inventory_games':len(d['cohort']['game_ids']),'matched_games':len(rows),'unmatched_games':len(d['cohort']['game_ids'])-len(rows),
      'matched_hero_hands':sum(r['hero_hands'] for r in rows),'included_games':len(included),
      'included_hero_hands':sum(r['hero_hands'] for r in included),'net_bb':float(net),
      'active_table_hours':hours,'bb_per_active_table_hour':float(net)/hours,
      'union_active_hours':union_ms(included_spans)/3600000,
      'cross_table_overlap_hours':hours-union_ms(included_spans)/3600000,
      'sensitivity':{'hand_event_hours':handhours,'bb_per_hand_event_hour':float(net)/handhours,
                     'first_to_last_hand_hours':envelope,'bb_per_first_to_last_hour':float(net)/envelope,
                     'including_flagged_ledger_as_reported_bb_per_hour':sum(r['net_bb'] for r in rows)/sum(r['active_table_hours'] for r in rows)},
      'descriptive_game_bootstrap':{'draws':20000,'seed':261023,'percentiles':[.025,.975],'bb_per_hour_interval':interval,
                                    'unit':'whole game; ratio of resampled total BB to resampled active table hours'},
      'causal_assistant_lift':None,'continuous_assistant_exposure_bb_per_hour':None,
      'checks':{'unique_hero_hands':True,'positive_stack_and_recorded_hero_action_every_hand':True,
                'all_game_hand_numbers_contiguous_from_one':True,'game_import_counts_match':True,
                'hand_end_before_next_start':True,'exact_id_ledger_net_reconciles':True},
      'time_definition':'For each maximal run of consecutive table hands in which Hero appears, last recorded event of final hand minus start of first hand. Includes intervals between consecutive participated hands; excludes absent/sitting-out hands and time after final hand.',
      'limitations':['Convenience cohort of recorded games; runtime versions and human adherence vary.',
                     'Hand-based active table time is not login duration, attention time, or verified Assistant treatment exposure.',
                     'One inventory game has no matching imported game; one unbalanced nit-suspect ledger is excluded. Two games changed blinds only outside Hero participation and remain eligible.',
                     'Ledger profit is used at the entire game level; no selection by individual winning/losing hands.',
                     'Raw ledger session fields can include idle/spectator time or missing endpoints; DB timestamp fields also differ from raw CSV times. Neither is used for the denominator.',
                     f'Bootstrap is descriptive with only {len(included)} game clusters, changing stakes/opponents, and no random assignment.'],
      'games':rows}
    if args.time_correction:
        correction=json.loads(args.time_correction.read_text(encoding='utf-8'))
        corrected_hours=Decimal(str(correction['playing_hours']))
        assert corrected_hours>0 and correction['applies_to_included_games']==len(included)
        original={k:output.pop(k) for k in ['active_table_hours','bb_per_active_table_hour','union_active_hours','cross_table_overlap_hours','sensitivity','descriptive_game_bootstrap','time_definition']}
        original['status']='Superseded denominator diagnostic; does not reproduce the corrected aggregate playing time.'
        output.update(schema='stackhud_corrected_aggregate_rate_v2',playing_hours=float(corrected_hours),
                      bb_per_playing_hour=float(net/corrected_hours),time_correction=correction,
                      prior_hand_time_diagnostic=original,descriptive_game_bootstrap=None,
                      time_definition='Author-supplied corrected aggregate playing time for the existing eligible cohort; revised per-game allocations are unavailable.',
                      game_time_fields_status='Original diagnostic clocks only; not allocations of corrected playing_hours')
        output['limitations']=[
            'The 66.49-hour correction is supplied by the project author and applied to the existing 32-game profit cohort; its underlying calculation is unavailable for independent verification.',
            'Original per-game hand clocks are retained as a superseded diagnostic and are not rescaled to invent corrected allocations.',
            'No corrected hourly-rate confidence interval or timing sensitivity is reported without revised per-game exposure.',
            'One inventory game is unmatched and one unbalanced nit-suspect ledger is excluded; complete eligible game ledgers determine profit.',
            'Changing versions, opponents and stakes; no verified continuous Assistant exposure, adherence or causal comparison.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['games','limitations','time_definition','cohort_rule','checks']},indent=2))


if __name__=='__main__': main()
