"""Land bounded connector raw responses and verify exact upload identities."""
import base64
import json
import sys
from pathlib import Path
import stage6_empirical_terminal_replay_v2 as core

STAGE=Path(__file__).resolve().parents[1]

def main(payload_path):
    payload=core.load(payload_path);record=payload['record'];number=payload['pass']
    core.expect(type(number) is int and number in (1,2),'invalid_readback_pass')
    expected=core.identity(core.resolve(STAGE,record['folder']+'/'+record['name']))
    meta=payload['metadata'];fetched=payload['raw'];revision=payload['revisions']
    core.expect(meta['id']==record['id'] and meta['title']==record['name'] and record['parent'] in meta['parent_ids'] and int(meta['size'])==expected['bytes'],'live_upload_metadata_mismatch')
    core.expect(fetched['id']==record['id'] and fetched['file_name']==record['name'] and fetched['file_size_bytes']==expected['bytes'],'raw_download_identity_mismatch')
    core.expect(revision['fileId']==record['id'] and revision.get('currentRevisionId'),'current_revision_required')
    raw=base64.b64decode(fetched['b64_string'],validate=True)
    core.expect(len(raw)==expected['bytes'] and core.hashlib.sha256(raw).hexdigest()==expected['sha256'],'raw_download_bytes_mismatch')
    destination=STAGE/'runtime_state/drive_readbacks'/record['id']/('pass'+str(number))
    destination.mkdir(parents=True,exist_ok=True);target=destination/record['name'];core.atomic_bytes(target,raw)
    receipt=dict(schema_version='stage7_raw_upload_readback_pass_v1',file_id=record['id'],pass_number=number,
        downloaded_utc=payload['downloaded_utc'],local_raw_path=str(target),**expected,
        metadata=dict(id=meta['id'],name=meta['title'],parents=meta['parent_ids'],size=int(meta['size']),modifiedTime=meta['modified_time'],
            currentRevisionId=revision['currentRevisionId'],version_counter=None,version_counter_note='Drive connector does not expose files.version; actual currentRevisionId recorded without inventing a version counter'))
    core.write(destination/'receipt.json',receipt)
    if number==2:
        first=core.load(destination.parent/'pass1/receipt.json')
        core.expect(first['metadata']==receipt['metadata'] and first['downloaded_utc']!=receipt['downloaded_utc'] and first['local_raw_path']!=receipt['local_raw_path'],'two_independent_stable_downloads_required')
        core.verify(first['local_raw_path'],expected)
        core.write(destination.parent/'accepted.json',dict(schema_version='stage7_raw_upload_readback_v1',status='PASS_TWO_RAW_DOWNLOADS',file_id=record['id'],passes=[first,receipt]))
    print(json.dumps(dict(file=record['name'],pass_number=number,status='PASS',**expected)))

if __name__=='__main__':main(Path(sys.argv[1]))
