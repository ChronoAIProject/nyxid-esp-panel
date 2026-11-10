"""Summarize audited negative development corpora without retuning thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
from scipy.stats import chi2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',nargs=2,type=Path,action='append',required=True,metavar=('AUDIT','HIESP_REPLAY'))
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    sources=[];models={};thresholds={};speakers=set();engine=None
    for audit_path,replay_path in a.corpus:
        audit=json.loads(audit_path.read_text());replay=json.loads(replay_path.read_text())
        if (audit['hiesp_report_sha256']!=sha(replay_path) or audit['independent_final_audio_accessed']
                or not (audit.get('exact_stream_pcm_verified') or audit.get('same_original_pcm_verified'))):
            raise ValueError('Missing same-PCM audit')
        streams=replay.get('reports',[replay]);events=0;seconds=0
        for r in streams:
            identity=r['record']['speaker']
            if identity in speakers:raise ValueError('Repeated speaker/session across corpora')
            speakers.add(identity)
            if (not r['development_negative_validation'] or r['record']['split']!='validation'
                    or r['firmware_stage']!=104 or not r['firmware_crc32_matched']):
                raise ValueError('Invalid/incomplete development replay')
            signature=tuple(r[k] for k in ('model','threshold_requested','threshold_getter_after_set','elf_sha256','flash_sha256','qemu_sha256'))
            if engine is not None and signature!=engine:raise ValueError('Mixed engine/operating points')
            engine=signature;events+=len(r['events']);seconds+=r.get('source_seconds',r['original_seconds'])
        h=audit['hiesp']
        if events!=h['events'] or abs(seconds/3600-h['source_hours'])>1e-12:raise ValueError('Audit/replay counts differ')
        current={r['model_sha256']:r for r in audit['nyx']}
        if models and current.keys()!=models.keys():raise ValueError('Different model coverage')
        for key,r in current.items():
            if key in thresholds and r['threshold']!=thresholds[key]:raise ValueError('Nyx threshold changed across corpora')
            if abs(r['source_hours']-h['source_hours'])>1e-12:raise ValueError('Unequal model exposure')
            thresholds[key]=r['threshold'];models.setdefault(key,[]).append(r)
        sources.append(dict(audit_sha256=sha(audit_path),replay_sha256=sha(replay_path),
                            name=audit_path.name,hiesp=h,nyx=audit['nyx']))
    hours=sum(s['hiesp']['source_hours'] for s in sources)
    def metrics(count):
        return dict(events=count,source_hours=hours,false_per_hour=count/hours,
                    false_per_hour_upper_95=float(.5*chi2.ppf(.95,2*(count+1))/hours))
    hcount=sum(s['hiesp']['events'] for s in sources)
    result=dict(role='Descriptive pooled existing negative development, not acceptance or threshold selection',
                source_sha256=sha(Path(__file__)),corpora=sources,speakers_or_sessions=len(speakers),
                hiesp=dict(threshold_requested=engine[1],threshold_getter=engine[2],**metrics(hcount)),
                nyx=[dict(model_sha256=k,threshold=thresholds[k],same_observed_pooled_event_count=sum(r['events'] for r in rows)==hcount,
                          **metrics(sum(r['events'] for r in rows))) for k,rows in models.items()],
                independent_final_audio_accessed=False,thresholds_reselected=False,
                limits='Descriptive pooling of previously reused development corpora; all per-corpus results retained. Different acoustic populations and short far-field exposure remain. Equal observed counts do not establish equal population false-wake rates. Nominal stationary Poisson bound is not independent acceptance. Does not replace the existing per-corpus zero-event candidate gate, match general FAR, or measure positive recall/latency.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='corpora'}))


if __name__ == '__main__':main()
