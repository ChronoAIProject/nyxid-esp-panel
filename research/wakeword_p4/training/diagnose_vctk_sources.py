"""Descriptive voice/speed/level diagnostics from existing development evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','v30','v35','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    assert sha(a.manifest)=='dcf1bbd2ee183bd1bdb893f4662e94d74cc6554b2cc15651834bdc5e59f030da'
    source={r['path']:r for r in map(json.loads,a.manifest.read_text().splitlines())}
    reports={name:json.loads(getattr(a,name).read_text()) for name in ('v30','v35')}
    joined=[]
    for name,report in reports.items():
        for item in report['records']:
            row=item['record'];s=source[row['path']]
            assert row['source_sha256']==s['sha256'] and row['split']=='validation'
            joined.append(dict(model=name,path=s['path'],voice=s['voice'],length_multiplier=s['length_multiplier'],
                target=s['target'],rms=s['rms'],samples=s['samples'],peak=s['peak'],
                any_event=bool(item['nyx']['event_seconds'])))
    def summarize(rows):
        target=[r for r in rows if r['target']=='hey_nyx'];negative=[r for r in rows if r['target'] in ('hey_nick','hey_next','hey_mix')]
        return dict(target_clips=len(target),target_any_event=sum(r['any_event'] for r in target),
            common_negative_clips=len(negative),common_false_clips=sum(r['any_event'] for r in negative))
    tables={}
    for model in reports:
        rows=[r for r in joined if r['model']==model];positive=[r for r in rows if r['target']=='hey_nyx']
        tables[model]=dict(by_voice={voice:summarize([r for r in rows if r['voice']==voice]) for voice in sorted({r['voice'] for r in rows})},
            by_length_multiplier={str(value):summarize([r for r in rows if r['length_multiplier']==value]) for value in (.9,1.,1.1)},
            target_level_by_event={str(hit):dict(clips=len(group),rms_median=float(np.median([r['rms'] for r in group])) if group else None,
                samples_median=float(np.median([r['samples'] for r in group])) if group else None)
                for hit in (False,True) for group in [[r for r in positive if r['any_event']==hit]]},
            whole_source_longer_than24160={str(long):summarize([r for r in rows if (r['samples']>24160)==long]) for long in (False,True)})
    report=dict(role='Post-evaluation descriptive development diagnosis only',manifest_sha256=sha(a.manifest),
        input_report_sha256={name:sha(getattr(a,name)) for name in reports},source_sha256=sha(Path(__file__)),
        historical_thresholds={name:reports[name].get('threshold',reports[name].get('nyx_threshold')) for name in reports},
        tables=tables,records=joined,final_data_used=False,new_wake_scores_computed=False,
        limits='Associations only; voice, length and RMS confounded. Full waveform duration includes generated silence and does not prove keyword span exceeds receptive field. Separate historical thresholds, not matched FAR. No sample filtering or relabeling.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(tables),flush=True)


if __name__=='__main__':main()
