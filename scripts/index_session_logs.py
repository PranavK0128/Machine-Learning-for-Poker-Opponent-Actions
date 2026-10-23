"""Deduplicate recorded PokerNow log events for an offline outcome audit. No network."""
from pathlib import Path
import json,sqlite3,hashlib,collections,csv,datetime
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def main():
    db=sqlite3.connect(OUT/'session_events.sqlite');db.executescript('CREATE TABLE IF NOT EXISTS events(game TEXT,event_id TEXT,at TEXT,text TEXT,PRIMARY KEY(game,event_id,text)); CREATE TABLE IF NOT EXISTS sessions(path TEXT PRIMARY KEY,game TEXT,hero_name TEXT,first_observed TEXT,last_observed TEXT,observations INTEGER,sha256 TEXT,bytes INTEGER);')
    fs=sorted((ROOT/'data/exports/ml_runtime/stage016/live_chrome').glob('*/session_log_updates.jsonl'))
    errors=[]
    for n,p in enumerate(fs):
        if db.execute('SELECT 1 FROM sessions WHERE path=?',(str(p.relative_to(ROOT)),)).fetchone():continue
        cfg=p.parent/'bridge_config.json';c=json.loads(cfg.read_text(encoding='utf-8')) if cfg.exists() else {};count=0;first=None;last=None;sha=hashlib.sha256();game=c.get('game_id');batch=[]
        with p.open('rb') as f:
            for line in f:
                sha.update(line)
                if not line.strip():continue
                try:d=json.loads(line)
                except Exception as e:errors.append({'path':str(p),'line':count+1,'error':str(e)});continue
                game=d.get('gameId',game);ts=d.get('observedAtUtc');first=first or ts;last=ts;count+=1
                for e in d.get('sessionLog',{}).get('entries',[]):
                    text=e.get('text','');at=e.get('at') or e.get('entryTime') or '';eid=str(e.get('created_at') or e.get('id') or at)
                    if text:batch.append((game,eid,at,text))
                if len(batch)>10000:db.executemany('INSERT OR IGNORE INTO events VALUES(?,?,?,?)',batch);batch=[]
        db.executemany('INSERT OR IGNORE INTO events VALUES(?,?,?,?)',batch)
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',(str(p.relative_to(ROOT)),game,c.get('hero_name'),first,last,count,sha.hexdigest(),p.stat().st_size));db.commit()
        if n%15==0:print(f'Indexed {n+1}/{len(fs)} logs',flush=True)
    summary={'sessions':db.execute('SELECT count(*) FROM sessions').fetchone()[0],'games':db.execute('SELECT count(DISTINCT game) FROM events').fetchone()[0],'unique_events':db.execute('SELECT count(*) FROM events').fetchone()[0],'errors':errors,'hero_config_names':db.execute('SELECT hero_name,count(*) FROM sessions GROUP BY hero_name').fetchall()}
    (OUT/'session_index_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
    with (OUT/'session_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        c=db.execute('SELECT * FROM sessions ORDER BY first_observed');w=csv.writer(f);w.writerow([v[0] for v in c.description]);w.writerows(c)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--audit-dir',type=Path,required=True)
    args=ap.parse_args();ROOT=args.workspace.resolve();OUT=args.audit_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'session_events.sqlite').exists():ap.error('Use a new audit directory; existing snapshots are not silently reused')
    main()
