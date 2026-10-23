"""One-time no-overwrite copy of the empirical program; preserve predecessors."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

STAGE=Path(__file__).resolve().parents[1]
PREVIOUS=STAGE.parent/'Stage 6 - Qualified Layers Improvement'

def identity(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def main():
    receipt=STAGE/'manifests/stage7_predecessor_copy.json'
    if receipt.exists():raise ValueError('Predecessor already imported; preserve successor edits')
    for name in ('src','tests','notebooks','inputs','manifests','runtime_state','returned_results','audits'):
        (STAGE/name).mkdir(parents=True,exist_ok=True)
    files=set(PREVIOUS.glob('src/stage6_empirical*'))|set(PREVIOUS.glob('tests/*empirical*.py'))
    queue=list(files)
    while queue:
        source=queue.pop()
        if source.suffix!='.py':continue
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8-sig'))):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                dependency=PREVIOUS/'src'/(module.replace('.','/')+'.py')
                if dependency.is_file() and dependency not in files:files.add(dependency);queue.append(dependency)
    files.update(PREVIOUS.glob('*EMPIRICAL*.md'))
    files.add(PREVIOUS/'stage6_empirical_terminal_replay_v2_HANDOFF.md')
    files.update(PREVIOUS.glob('audits/empirical_v2_independent/*.md'))
    for name in ('stage6_empirical_v2_completed_r3b.zip','stage6_frozen_capture_comparison_v2_r1.zip','empirical_v2_range_plan_r3b.json'):
        files.add(PREVIOUS/'inputs'/name)
    for name in ('stage6_empirical_v2_completed_release_r3b.json','stage6_frozen_capture_comparison_v2_r1.json','stage6_empirical_v2_completed_snapshot_r3.json','stage6_empirical_v2_evaluation_exposures.json'):
        files.add(PREVIOUS/'manifests'/name)
    files.add(PREVIOUS/'runtime_state/empirical_v2_execution_status.json')
    copied=[]
    for source in sorted(files):
        relative=source.relative_to(PREVIOUS);target=STAGE/relative;before=identity(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Destination already exists: '+str(target))
        shutil.copy2(source,target)
        if identity(target)!=before or identity(source)!=before:raise ValueError('Copy parity failed: '+str(source))
        copied.append(dict(path=relative.as_posix(),**before))
    receipt.write_text(json.dumps(dict(schema_version='stage7_predecessor_copy_v1',source=str(PREVIOUS),destination=str(STAGE),files=copied,parity='PASS',predecessors_preserved=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(copied=len(copied),receipt=str(receipt))))

if __name__=='__main__':main()
