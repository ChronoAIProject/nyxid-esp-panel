"""Verify full saved native returns, summarize and archive consumed reference evidence."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import zlib

ROOT=Path('research/wakeword_p4')
PLAN=ROOT/'results/4090-willow-c161-reference-plan-v1.json'
OUT=Path('.local/wakeword/willow-c161-reference-v1')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plan=json.loads(PLAN.read_text())
    assert len(list(OUT.glob('*/report.json')))==len(plan['records'])==72,'Collection incomplete'
    groups={}
    files={}
    prime_events=0
    for index,row in enumerate(plan['records']):
        folder=OUT/f'{index:03d}'
        r=json.loads((folder/'report.json').read_text())
        assert r['record']==row and r['record_index']==index
        assert r['model']=='wn9_heywillow_tts' and r['mode']=='DEFAULT'
        assert r['actual_float32_bits']==r['post_prime_getter_bits']=='3f24dd2f'
        assert r['total_prime_samples']==81600 and r['transition_reset'] is False
        assert r['source_input_samples']==row['samples'] and r['source_pcm_sha256']==row['pcm_sha256']
        assert r['input_samples']==row['samples']+49344
        assert sha(folder/'serial.log')==r['serial_sha256']
        assert sha(folder/'raw-returns.bin')==r['raw_returns_sha256']
        for name in ('qemu','elf','flash'):
            assert r[name+'_sha256']==plan['files'][plan['runtime'][name]]
        text=(folder/'serial.log').read_text()
        returns=[tuple(map(int,t)) for t in re.findall(r'HEY_REFERENCE_RETURN call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
        n=r['input_samples']
        assert len(returns)==r['input_calls']==(n+511)//512
        assert all(c==i and e==min(i*512,n) and v in (-1,0,1) for i,(c,e,v) in enumerate(returns,1))
        assert bytes(v+1 for c,e,v in returns)==(folder/'raw-returns.bin').read_bytes()
        events=[dict(call=c,sample_end=e,result=v,audio_seconds=e/16000) for c,e,v in returns if v==1]
        assert events==r['events']
        real=[dict(e,relative_sample=e['sample_end']-49344) for e in events if e['sample_end']>49344]
        assert real==r['real_events']
        prime=[list(map(int,t)) for t in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
        assert prime==r['prime_returns'] and [c for c,v in prime]==list(range(1,64))
        prime_events+=sum(v==1 for c,v in prime)+len(r['additional_priming_events'])
        assert re.search(r':\s+0x00000068\b',(folder/'stage-monitor.txt').read_text())
        # Recheck original PCM CRC without writing a duplicate input.
        import soundfile as sf
        pcm,sr=sf.read(row['path'],dtype='int16');assert sr==16000 and pcm.ndim==1
        raw=pcm.astype('<i2').tobytes()
        assert hashlib.sha256(raw).hexdigest()==row['pcm_sha256']
        crc=zlib.crc32(raw,zlib.crc32(bytes(49344*2)))
        assert int(r['input_crc32'],16)==crc
        key=row['target_id']+'/'+('command' if row['carrier'] else 'bare')
        g=groups.setdefault(key,dict(clips=0,activated_clips=0,events=0))
        g['clips']+=1;g['activated_clips']+=bool(real);g['events']+=len(real)
        for f in sorted(folder.iterdir()):
            if f.is_file():files[str(f.relative_to(OUT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    summary=dict(schema='willow-c161-reference-result-v1',plan_sha256=sha(PLAN),records=72,
                 groups=groups,priming_events=prime_events,files=files,
                 limits='Consumed synthetic reference-only diagnostic.72 clips, no background/FAR evidence. Official native512 versus C161480 event policy differs. Same total81600 zero warmup alone does not establish P4 parity, human performance or model superiority. No threshold selection or new training.')
    result=ROOT/'results/4090-willow-c161-reference-result-v1.json'
    result.write_text(json.dumps(summary,indent=2)+'\n')
    archive=ROOT/'results/4090-willow-c161-reference-evidence-v1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in sorted(files):tar.add(OUT/name,arcname=name)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames())==set(files)
        for member in tar:
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==files[member.name]['sha256']
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),result_sha256=sha(result),groups=groups)
    (ROOT/'results/4090-willow-c161-reference-publication-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
