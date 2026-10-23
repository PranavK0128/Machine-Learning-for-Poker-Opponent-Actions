"""Reconstruct complete ID-bound hands and independently reconcile next-hand stacks.

PokerNow calls/raises are street totals. Antes are separate contributions.
This audit produces captured-hand observations, not a causal Assistant treatment effect.
"""
from pathlib import Path
from decimal import Decimal as D
import sqlite3,re,json,csv,collections,datetime
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
HERO=None
NUM=r'([\d,]+(?:\.\d+)?)'
def val(s):return D(s.replace(',',''))
def pid(token):
    m=re.fullmatch(r'".* @ ([^" ]+)"',token.strip());return m[1] if m else None
def ts(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp() if s else None
def main():
    c=sqlite3.connect(OUT/'session_events.sqlite');rows=[];unknown=collections.Counter();games=[]
    for game, in c.execute('select distinct game from events order by game'):
        events=c.execute('select event_id,at,text from events where game=? order by cast(event_id as integer),at,text',(game,)).fetchall()
        # Use ID-bearing direct-log records only. Legacy DOM labels cannot authenticate Hero.
        events=[e for e in events if e[0].isdigit() and e[1]]
        hands=[];cur=None
        for eid,at,t in events:
            m=re.match(r'-- starting hand #(\d+) \(id: ([^)]+)\)',t)
            if m:
                if cur:hands.append(cur)
                cur={'game_id':game,'hand_id':m[2],'number':int(m[1]),'start':at,'end':None,'events':[]}
            if cur:cur['events'].append((eid,at,t))
            if cur and t.startswith('-- ending hand #'):cur['end']=at
        if cur:hands.append(cur)
        for idx,h in enumerate(hands):
            stacks={}; contrib=collections.defaultdict(lambda:D(0));street=collections.defaultdict(lambda:D(0));returns=collections.defaultdict(lambda:D(0));wins=collections.defaultdict(lambda:D(0));bounty=collections.defaultdict(lambda:D(0));adjust=collections.defaultdict(lambda:D(0));bb=None;flags=[];endseen=False; seen=set()
            for eid,at,t in h['events']:
                if (eid,t) in seen:continue
                seen.add((eid,t))
                if t.startswith('Player stacks:'):
                    for token,amt in re.findall(r'#\d+\s+("[^"]+")\s+\('+NUM+r'\)',t):
                        p=pid(token)
                        if p:stacks[p]=val(amt)
                    continue
                if t.startswith(('Flop:','Turn:','River:')):street.clear();continue
                if t.startswith('-- ending hand #'):endseen=True;continue
                m=re.match(r'("[^"]+") posts (?:a |an )?(small blind|big blind|ante|dead small blind|dead big blind|missed small blind|missed big blind|straddle) of '+NUM,t,re.I)
                if m:
                    p=pid(m[1]);amount=val(m[3]);kind=m[2].lower()
                    if not p:flags.append('unbound_actor');continue
                    contrib[p]+=amount
                    if kind=='big blind':bb=amount
                    if kind!='ante':street[p]+=amount
                    if 'dead' in kind or 'missed' in kind:flags.append('ambiguous_missed_blind')
                    continue
                m=re.match(r'("[^"]+") (bets|calls|raises to) '+NUM,t)
                if m:
                    p=pid(m[1]);amount=val(m[3])
                    if not p:flags.append('unbound_actor');continue
                    delta=amount-street[p]
                    if delta<0:flags.append('negative_street_delta')
                    contrib[p]+=delta;street[p]=amount;continue
                m=re.match(r'Uncalled bet of '+NUM+r' returned to ("[^"]+")',t)
                if m:
                    p=pid(m[2]);returns[p]+=val(m[1]);continue
                m=re.match(r'("[^"]+") collected '+NUM+r' from (?:main |side )?pot',t)
                if m:wins[pid(m[1])]+=val(m[2]);continue
                m=re.match(r'("[^"]+") paid '+NUM+r' for the 7-2 bounty to ("[^"]+")',t)
                if m:bounty[pid(m[1])]-=val(m[2]);bounty[pid(m[3])]+=val(m[2]);continue
                m=re.match(r'The admin updated the player ("[^"]+") stack from '+NUM+' to '+NUM,t)
                if m:adjust[pid(m[1])]+=val(m[3])-val(m[2]);continue
                if any(x in t for x in [' posts ',' calls ',' bets ',' raises ','collected ','Uncalled bet',' paid ']):
                    flags.append('unparsed_money_event');unknown[t]+=1
            if HERO not in stacks:continue
            cash=wins[HERO]+returns[HERO]-contrib[HERO]+bounty[HERO]
            residual=sum(contrib.values())-sum(returns.values())-sum(wins.values())
            if not h['end']:flags.append('missing_hand_end')
            if not bb or bb<=0:flags.append('missing_big_blind')
            if abs(residual)>D('0.02'):flags.append('table_cashflow_not_conserved')
            next_h=hands[idx+1] if idx+1<len(hands) else None;next_stack=None;stack_residual=None;duration=None
            if next_h and next_h['number']==h['number']+1:
                for _,_,t in next_h['events']:
                    if t.startswith('Player stacks:'):
                        for token,amt in re.findall(r'#\d+\s+("[^"]+")\s+\('+NUM+r'\)',t):
                            if pid(token)==HERO:next_stack=val(amt)
                        break
                if next_stack is not None:
                    stack_residual=next_stack-stacks[HERO]-adjust[HERO]-cash
                    if abs(stack_residual)>D('0.02'):flags.append('next_stack_mismatch')
                else:flags.append('missing_next_hero_stack')
                duration=ts(next_h['start'])-ts(h['start'])
                if duration<=0 or duration>1800:flags.append('duration_gap')
            else:flags.append('missing_consecutive_next_hand')
            rows.append({'game_id':game,'hand_id':h['hand_id'],'hand_number':h['number'],'start_utc':h['start'],'end_utc':h['end'],'next_start_utc':next_h['start'] if next_h else None,'hero_start':float(stacks[HERO]),'hero_next':float(next_stack) if next_stack is not None else None,'contributed':float(contrib[HERO]),'collected':float(wins[HERO]),'returned':float(returns[HERO]),'bounty_net':float(bounty[HERO]),'admin_adjustment':float(adjust[HERO]),'net_chips':float(cash),'big_blind':float(bb) if bb else None,'net_bb':float(cash/bb) if bb else None,'duration_seconds':duration,'table_residual':float(residual),'next_stack_residual':float(stack_residual) if stack_residual is not None else None,'eligible':not flags,'hold_reasons':'|'.join(sorted(set(flags)))})
    c.close()
    with (OUT/'captured_hand_outcomes_private.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    accepted=[r for r in rows if r['eligible']];reason=collections.Counter(x for r in rows for x in r['hold_reasons'].split('|') if x)
    summary={'hero_id':HERO,'identity_basis':'Exact ID supplied explicitly by the operator; name-only matches are excluded.','candidate_id_bound_hands':len(rows),'reconciled_hands':len(accepted),'reconciled_games':len(set(r['game_id'] for r in accepted)),'excluded_hands':len(rows)-len(accepted),'hold_reasons':dict(reason),'net_bb_reconciled_subset':sum(r['net_bb'] for r in accepted),'net_chips_reconciled_subset':sum(r['net_chips'] for r in accepted),'positive_hands':sum(r['net_chips']>0 for r in accepted),'negative_hands':sum(r['net_chips']<0 for r in accepted),'zero_hands':sum(r['net_chips']==0 for r in accepted),'observed_consecutive_hand_hours':sum(r['duration_seconds'] for r in accepted)/3600,'overall_bb_per_hour':None,'causal_assistant_lift':None,'limitations':['Selected reconciled subset; exclusions and missing exposure prevent a full-session win-rate estimate.','Hand interval includes between-hand time but excludes missing hands; not wall-clock Assistant exposure.','Recorded actions are human actions; recommendation adherence and baseline control are not established.','Bounties are included only when explicitly recorded; external payments are not observed.'],'unparsed_money_examples':unknown.most_common(12)}
    (OUT/'outcome_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--hero-id',required=True)
    ap.add_argument('--audit-dir',type=Path,required=True,help='Contains session_events.sqlite; private outputs will be written here')
    args=ap.parse_args();HERO=args.hero_id;OUT=args.audit_dir.resolve()
    if not (OUT/'session_events.sqlite').is_file():ap.error('session_events.sqlite is missing')
    main()
