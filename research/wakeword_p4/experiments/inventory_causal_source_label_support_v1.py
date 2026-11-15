"""Training-only metadata census of lexical/source support; no model replay or fit."""
import collections
import hashlib
import json
from pathlib import Path

FEATURES=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json')
CARRIERS=Path('.local/wakeword/hey-common-carrier-features-v1/features.json')
DEST=Path('research/wakeword_p4/results/4090-causal-source-label-support-v1.json')

def main():
    broad=json.loads(FEATURES.read_text());carriers=json.loads(CARRIERS.read_text())
    assert all(r['split']=='training' for r in broad+carriers)
    positives=[r for r in broad if r['label']==1 and r['source_pair_key']['carrier']=='bare']
    assert len(positives)==6444
    nearwords={'Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.'}
    groups=collections.defaultdict(set)
    for r in broad:
        if r['source_pair_key']['carrier']=='bare' and r['text'] in nearwords|{'Hey Nicks.'}:
            key=(r['engine'],json.dumps(r['source_pair_key'],sort_keys=True),r['variant'])
            groups[key].add(r['text'])
    rows=[]
    for engine in sorted({r['engine'] for r in positives}):
        pos=[r for r in positives if r['engine']==engine]
        ids={r['conditioning_id'] for r in pos}
        available=[r for r in broad if r['engine']==engine and r['conditioning_id'] in ids and r['text'] in nearwords and r['label']==0]
        current=[r for r in carriers if r['engine']==engine and r['conditioning_id'] in ids]
        current_ids={r['conditioning_id'] for r in current}
        rows.append(dict(engine=engine,positive_rows=len(pos),exact_source_speed_draw_variant_five_phrase_groups=sum(k[0]==engine and len(v)==5 for k,v in groups.items()),positive_groups_without_all_four_nearwords=sum(k[0]==engine and 'Hey Nicks.' in v and len(v)!=5 for k,v in groups.items()),positive_conditioning_ids=len(ids),current_extra_carrier_negative_rows=len(current),current_extra_carrier_negative_conditioning_ids=len(current_ids),positive_conditioning_ids_without_current_same_source_extra_negatives=len(ids-current_ids),available_unused_broad_bare_nearword_rows=len(available),available_bare_negative_phrases=dict(collections.Counter(r['text'] for r in available)),available_bare_negative_conditioning_ids=len({r['conditioning_id'] for r in available})))
    out=dict(schema='causal-source-label-support-v1',inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [FEATURES,CARRIERS]],records=rows,scope='Actual positive-coverage sampler supplies8 broad barepositive slots. Current12 extraKokoro command-negative slots use CARRIERS. Other current negative pools are ARU/Arctic models, not these broad-source identifiers. Counts are metadata availability, not draw frequencies, actual model fit or proof of source shortcut.',limits='Prior teacher/historical training used broader packs. Unused means absent from this causal broad insertion, not never seen by any ancestor. Conditioning IDs are synthetic controls, not independent people. No training, inference, new audio, numerical comparison, threshold selection or final access.')
    with DEST.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
