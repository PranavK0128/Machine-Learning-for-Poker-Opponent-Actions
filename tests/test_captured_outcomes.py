"""Synthetic accounting checks; these are not measured gameplay results."""
from pathlib import Path
import importlib.util,sqlite3,tempfile,json,csv,contextlib,io,unittest
P=Path(__file__).resolve().parents[1]/'scripts/audit_captured_outcomes.py'
spec=importlib.util.spec_from_file_location('outcome_audit',P);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class Accounting(unittest.TestCase):
    def run_fixture(self,ambiguous=False):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);db=sqlite3.connect(p/'session_events.sqlite');db.execute('create table events(game text,event_id text,at text,text text)')
            texts=[
                '-- starting hand #1 (id: h1)', 'Player stacks: #1 "Hero @ HERO" (100) | #2 "Other @ OTHER" (100)',
                '"Hero @ HERO" posts a small blind of 1','"Other @ OTHER" posts a big blind of 2',
                '"Hero @ HERO" calls 2','Flop: [As, 7c, 2d]',
                '"Hero @ HERO" bets 4','"Other @ OTHER" raises to 10','"Hero @ HERO" calls 10',
                '"Hero @ HERO" collected 24 from pot','"Other @ OTHER" paid 2 for the 7-2 bounty to "Hero @ HERO"',
                '-- ending hand #1', '-- starting hand #2 (id: h2)',
                'Player stacks: #1 "Hero @ HERO" (114) | #2 "Other @ OTHER" (86)',
                '"Hero @ HERO" posts a big blind of 2','"Other @ OTHER" posts a small blind of 1',
                '"Other @ OTHER" folds','Uncalled bet of 1 returned to "Hero @ HERO"',
                '"Hero @ HERO" collected 2 from pot','The admin updated the player "Hero @ HERO" stack from 115 to 125',
                '-- ending hand #2','-- starting hand #3 (id: h3)',
                'Player stacks: #1 "Hero @ HERO" (125) | #2 "Other @ OTHER" (85)']
            if ambiguous:texts.insert(8,'"Other @ OTHER" posts a bet of 4 (bomb pot bet)')
            for i,line in enumerate(texts):db.execute('insert into events values (?,?,?,?)',('synthetic',str(i+1),f'2026-01-01T00:00:{i:02d}Z',line))
            db.commit();db.close();audit.OUT=p;audit.HERO='HERO'
            with contextlib.redirect_stdout(io.StringIO()):audit.main()
            summary=json.loads((p/'outcome_audit_summary.json').read_text())
            with (p/'captured_hand_outcomes_private.csv').open() as f:rows=list(csv.DictReader(f))
            return summary,rows
    def test_street_totals_returns_bounty_and_admin_adjustment(self):
        s,r=self.run_fixture();self.assertEqual(s['reconciled_hands'],2);self.assertEqual(s['excluded_hands'],1)
        self.assertEqual(float(r[0]['contributed']),12);self.assertEqual(float(r[0]['net_bb']),7)
        self.assertEqual(float(r[1]['net_bb']),.5);self.assertEqual(float(r[1]['admin_adjustment']),10)
        self.assertAlmostEqual(s['net_bb_reconciled_subset'],7.5);self.assertIsNone(s['overall_bb_per_hour'])
    def test_unparsed_money_is_held(self):
        s,r=self.run_fixture(True);self.assertEqual(r[0]['eligible'],'False');self.assertIn('unparsed_money_event',r[0]['hold_reasons'])
        self.assertEqual(s['reconciled_hands'],1)
if __name__=='__main__':unittest.main()
