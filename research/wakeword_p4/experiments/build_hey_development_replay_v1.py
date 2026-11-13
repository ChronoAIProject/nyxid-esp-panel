"""Admit complete development material and generated timing, without scoring."""
import json
from pathlib import Path
import soundfile as sf
from audit_hey_training_pcm_history_v1 import pcm
from generate_hey_development_native_v1 import sha
from synthetic_output_quality_v1 import audit_output
from synthetic_development_render_quality_v1 import audit_render

BASE=Path('.local/wakeword')
NATIVE=BASE/'hey-paired-development-native-v1'
RENDER=BASE/'hey-paired-development-rendered-v1'
TOKENS=BASE/'hey-paired-development-token-durations-v1.jsonl'
HISTORY=Path('research/wakeword_p4/results/4090-hey-development-pcm-history-v1.json')
DEST=BASE/'hey-development-replay-v1.json'


def main():
    history=json.loads(HISTORY.read_text())
    assert history['no_unexpected_exact_hash_matches'] and not history['matches']
    assert history['native_records_sha256']==sha(NATIVE/'records.jsonl')
    assert history['render_records_sha256']==sha(RENDER/'records.jsonl')
    ts=json.loads(TOKENS.with_suffix('.summary.json').read_text())
    assert ts['records']==120 and ts['records_sha256']==sha(TOKENS)
    tokens={r['id']:r for r in map(json.loads,TOKENS.read_text().splitlines())};assert len(tokens)==120
    sources={r['id']:r for r in map(json.loads,(NATIVE/'records.jsonl').read_text().splitlines())}
    assert set(sources)==set(tokens)
    for source in sources.values():
        assert source['role']=='development' and audit_output(NATIVE,source)['passed']
    summary=json.loads((RENDER/'summary.json').read_text())
    assert summary['pack_qualified'] and summary['records_sha256']==sha(RENDER/'records.jsonl')
    rows=[]
    for r in map(json.loads,(RENDER/'records.jsonl').read_text().splitlines()):
        source=sources[r['id']];t=tokens[r['id']]
        assert t['source_raw_sha256']==source['raw_sha256']
        assert t['source_waveform_sha256']==source['sha256']
        assert t['allocation_matches_saved_waveform_length'] and not t['waveform_generated']
        assert t['native_samples']==source['native_samples']
        path=RENDER/r['path'];assert sha(path)==r['sha256']
        dry,rate=sf.read(NATIVE/source['path'],dtype='int16');assert rate==16000
        values,rate=sf.read(path,dtype='int16');assert rate==16000
        assert audit_render(dry,values,r)['passed']
        prefix=r['lead_samples']+t['generated_prefix_end_16k_sample']
        assert r['lead_samples']<prefix<=r['native_generation_end']
        rows.append(dict(path=str(path),sha256=r['sha256'],pcm_sha256=pcm(path,r['samples']),samples=r['samples'],role='development-synthetic',split='validation',source_id=r['id'],target_id=r['target_id'],voice=r['voice'],literal=r['literal'],text=r['text'],carrier=r['carrier'],length_scale=r['length_scale'],variant=r['variant'],lead_samples=r['lead_samples'],generated_prefix_end_sample=prefix,full_rir_prefix_end_sample=prefix+r['rir_tail_samples'],generated_container_end_sample=r['native_generation_end'],full_rir_container_end_sample=r['keyword_end'],reference_kind='Internal VITS allocation and full-convolution support; not measured acoustic or phonetic boundary'))
    assert len(rows)==360
    counts={k:sum(r['target_id']==k for r in rows) for k in {r['target_id'] for r in rows}}
    assert counts==dict(hey_willow=36,hey_nyx=36,hi_esp=36,common_negative=252)
    result=dict(schema='hey-development-replay-manifest-v1',records=rows,counts=counts,history_receipt_sha256=sha(HISTORY),token_records_sha256=sha(TOKENS),render_records_sha256=sha(RENDER/'records.jsonl'),builder_sha256=sha(Path(__file__)),scoring_authorized=False,acceptance_accessed=False,limits='Mechanical/source/timing admission only. Hash history audit is bounded; shared TTS parent and unknown pretraining/partial overlap remain. No wake-model scores or threshold selection.')
    with DEST.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),manifest_sha256=sha(DEST),counts=counts,scoring_authorized=False)))


if __name__=='__main__':main()
