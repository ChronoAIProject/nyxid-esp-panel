"""Gated paired original/replacement batches; no TF, models or new RNG draws."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from schedule_hey_broad_v1 import sampler
from hey_batch_schedule_v1 import batch_digest
from common_carrier_slot_policy_v1 import map_end

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Batches:
    def __init__(self,target,condition):
        assert target in ('hey_willow','hey_nyx') and condition in ('control','command_context')
        # Fail before original sampler/feature reads if new material not admitted.
        reportpath=R/'4090-hey-common-carrier-feature-result-v1.json'
        auditpath=R/'4090-hey-common-carrier-feature-independent-audit-v1.json'
        report=json.loads(reportpath.read_text());audit=json.loads(auditpath.read_text())
        assert audit['all_passed'] and audit['all_rows_independently_recomputed'] and audit['original_result_sha256']==sha(reportpath)
        root=BASE/'hey-common-carrier-features-v1'
        assert sha(root/'features.json')==report['metadata_sha256'] and sha(root/'features.bin')==report['binary_sha256']
        rows=json.loads((root/'features.json').read_text());assert len(rows)==4860
        self.new_rows={(r['anchor_id'],r['carrier'],r['variant']):r for r in rows};assert len(self.new_rows)==4860
        self.newraw=np.memmap(root/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        assert len(self.newraw)==report['frames'] and self.newraw.nbytes==report['binary_bytes']
        for r in rows:assert r['label']==0 and r['split']=='training' and r['target_id']=='common_negative'
        decisionpath=R/'4090-hey-common-carrier-replacement-decisions-v1.json';decision=json.loads(decisionpath.read_text())
        policy=Path(__file__).with_name('common_carrier_slot_policy_v1.py');assert sha(policy)==decision['policy_sha256']
        entry=next(t for t in decision['targets'] if t['target']==target);self.expected_selected=entry['selected_slots']
        assert sha(entry['path'])==entry['sha256'];self.log=gzip.open(entry['path'],'rt')
        self.env=sampler(target);self.condition=condition;self.count=0;self.selected=0
        oldrows=json.loads((BASE/'hey-broad-training-features-v1'/target/'features.json').read_text())
        self.old_rows={(r['source_id'],r['variant']):r for r in oldrows if r['engine']=='kokoro' and r['target_id']=='common_negative'}
        assert len(self.old_rows)==1620
    def next(self):
        row=json.loads(next(self.log));self.count+=1;assert row['step']==self.count<=10000
        x,y,w=self.env['batch'](mine=False,augment=False);baseline=batch_digest(x,y,w);assert baseline==row['original_batch_sha256']
        original=x.copy();oldy=y.copy();oldw=w.copy();selected=[];seen=set()
        for choice in row['choices']:
            slot=choice['slot'];assert slot not in seen;seen.add(slot)
            old=self.old_rows[choice['anchor_id'],choice['variant']]
            assert old['frames']==choice['old_frames'] and old['lead_frame']==choice['old_lead']
            offset=old['offset'];end=choice['old_end'];olddata=self.env['raw'][offset+end-225:offset+end]
            assert np.array_equal(original[slot],np.clip(np.asarray(olddata,dtype=np.float32)/25.6,0,26)) and y[slot,0]==0 and w[slot]==20
            if not choice['replace']:continue
            new=self.new_rows[choice['anchor_id'],choice['carrier'],choice['variant']]
            assert new['native_order']==choice['native_order'] and new['conditioning_id']==old['conditioning_id'] and new['speed']==old['source_pair_key']['scale']
            mapped=map_end(end,old['frames'],old['lead_frame'],new['frames'],new['lead_frame'])
            at=new['offset'];replacement=np.clip(np.asarray(self.newraw[at+mapped-225:at+mapped],dtype=np.float32)/25.6,0,26)
            assert replacement.shape==(225,40)
            if self.condition=='command_context':x[slot]=replacement
            selected.append(dict(slot=slot,new_order=new['order'],new_end=mapped))
        mask=np.ones(128,dtype=bool);mask[[r['slot'] for r in selected]]=False
        assert np.array_equal(x[mask],original[mask]) and y.tobytes()==oldy.tobytes() and w.tobytes()==oldw.tobytes()
        if self.condition=='control':assert x.tobytes()==original.tobytes()
        self.selected+=len(selected)
        receipt=dict(step=self.count,original_batch_sha256=baseline,tensor_sha256=batch_digest(x,y,w),selected=selected,nonselected_rows_exact=True,labels_weights_exact=True)
        return (x,y,w),receipt
    def finish(self):
        assert self.count==10000 and self.selected==self.expected_selected==37114
        assert self.log.readline()=='';self.log.close()
        return dict(batches=self.count,selected=self.selected)

if __name__=='__main__':
    # Readiness check only: no constructor/no feature or model access.
    paths=[R/'4090-hey-common-carrier-feature-result-v1.json',R/'4090-hey-common-carrier-feature-independent-audit-v1.json']
    missing=[str(p) for p in paths if not p.exists()]
    print(json.dumps(dict(status='not_ready' if missing else 'ready',missing=missing)))
    raise SystemExit(2 if missing else 0)
