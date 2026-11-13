"""Normalize396 frozen development rows without changing PCM or boundaries."""
import hashlib
import json
from pathlib import Path

RESULTS=Path('research/wakeword_p4/results')
OUT=Path('.local/wakeword/jenny-development-replay-v1.json')
TARGETS={'Hey Willow':'hey_willow','Hey Nicks':'hey_nyx','Hi E S P':'hi_esp',
    'Hey':'hey','Hello':'hello','Hey William':'hey_william','Hey pillow':'hey_pillow',
    'Hey Nick':'hey_nick','Hey Next':'hey_next','Hey Mix':'hey_mix','Hey Muse':'hey_muse'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    render_report=RESULTS/'4090-jenny-development-render-v1.json';report=json.loads(render_report.read_text())
    audit_path=RESULTS/'4090-jenny-development-render-independent-audit-v1.json';audit=json.loads(audit_path.read_text())
    history_path=RESULTS/'4090-jenny-waveform-history-v1.json';history=json.loads(history_path.read_text())
    assert audit['original_report_sha256']==sha(render_report) and audit['independently_qualified']==396 and not audit['failures']
    assert history['no_exact_hash_matches'] and history['authenticated_new_waveforms']==528
    for path,h in history['files'].items():assert sha(path)==h
    records_path=Path('.local/wakeword/jenny-development-rendered-v1/records.jsonl')
    assert sha(records_path)==report['records_sha256']
    records=[]
    for r in map(json.loads,records_path.read_text().splitlines()):
        source=r['source'];assert source['role']=='development' and r['quality_pass'] and all(r['checks'].values())
        assert sha(r['path'])==r['sha256']
        phrase=source['target'];target=TARGETS[phrase]
        dryprefix=r['lead_samples']+source['generated_prefix_end'];drycontainer=r['lead_samples']+source['source_samples']
        assert dryprefix+r['rir_tail_samples']==r['generated_prefix_end_sample']
        assert drycontainer+r['rir_tail_samples']==r['generated_container_end_sample']
        assert 0<r['lead_samples']<dryprefix<=drycontainer<r['samples']
        row=dict(source_id=source['id'],target_id=target,voice='author-direct-jenny',carrier=source['pair_key']['carrier'],
            length_scale=source['pair_key']['scale'],variant=r['variant'],condition=['quiet','25dB_noise','15dB_noise_full_RIR'][r['variant']],
            text=source['text'],phrase=phrase,path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],
            role='new-development',lead_samples=r['lead_samples'],tail_samples=r['tail_samples'],
            generated_prefix_end_sample=dryprefix,full_rir_prefix_end_sample=dryprefix+r['rir_tail_samples'],
            generated_container_end_sample=drycontainer,full_rir_container_end_sample=drycontainer+r['rir_tail_samples'],
            whole_input_end_sample=r['samples'],render_order=r['order'],native_source_sha256=source['source_sha256'],
            native_source_pcm_sha256=source['source_pcm_sha256'],render_seed=r['seed'],gain_after_mix=r['gain_after_mix'])
        records.append(row)
    assert len(records)==396 and len({(r['source_id'],r['variant']) for r in records})==396
    counts={}
    for target in ('hey_willow','hey_nyx','hi_esp'):
        positive=[r for r in records if r['target_id']==target]
        assert len(positive)==36 and sum(r['carrier']=='bare' for r in positive)==9
        counts[target]=dict(positive=36,standalone=9,command=27,total_other=360)
        if target!='hi_esp':counts[target].update(common=324,cross_target=36)
    result=dict(schema='jenny-complete-development-replay-v1',records=records,role='new-development',standalone_carriers=['bare'],
        expected_counts=counts,source_history_limits=history['limits'],declared_voice_count=1,
        scope='One declared new synthetic source only; not multi-speaker generalization. Synthetic internal duration boundaries, not human phonetic truth. No scoring/threshold selection authorized by manifest.',
        files={str(p):sha(p) for p in [Path(__file__),records_path,render_report,audit_path,history_path]})
    with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    receipt=dict(schema='jenny-development-replay-manifest-receipt-v1',path=str(OUT),sha256=sha(OUT),records=396,
        expected_counts=counts,standalone_carriers=['bare'],files=result['files'],wake_scoring_authorized=False)
    with (RESULTS/'4090-jenny-development-replay-manifest-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(records=396,path=str(OUT),sha256=sha(OUT))))


if __name__=='__main__':main()
