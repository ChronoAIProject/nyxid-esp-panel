"""Choose full-source geometry from training-only saved timing, no model scores."""
import hashlib
import json
from pathlib import Path
import math

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=RESULTS/'4090-hey-broad-context-decision-v1.json';assert not out.exists()
    matrices={};required=150
    for engine in ['piper','kokoro','qwen']:
        p=BASE/f'hey-broad-{engine}-native-v1/records.jsonl'
        lines=p.read_bytes().splitlines(keepends=True);assert all(s.endswith(b'\n') for s in lines)
        rows=[json.loads(s) for s in lines]
        if engine=='qwen':
            # Exact original first144 supervised positive requests already complete;
            # later common-negative generation is still live and is not evaluated.
            assert len(rows)>=144
            rows=rows[:144];raw=b''.join(lines[:144]);assert all(r['request']['role']=='own-target-positive' for r in rows)
            assert [r['request'] for r in rows]==json.loads((RESULTS/'4090-hey-broad-qwen-standalone-material-plan-v1.json').read_text())['requests'][:144]
            ends=[r['generated_reference_end_16k_sample'] for r in rows]
        else:
            rp=RESULTS/f'4090-hey-broad-{engine}-generation-v1.json';d=json.loads(rp.read_text());assert sha(p)==d['records_sha256']
            raw=p.read_bytes();ends=[r['generated_prefix_end_16k_sample'] for r in rows]
        assert all(r['quality_pass'] for r in rows)
        maxsamples=max(ends)+960
        frames=math.ceil(maxsamples/160)+27
        required=max(required,frames)
        matrices[engine]=dict(positive_cases=len(rows),completed_record_prefix_sha256=hashlib.sha256(raw).hexdigest(),
            whole_source_prefix_plus_rir_max=maxsamples,frames_with_max_jitter=frames)
    selected=3*math.ceil(required/3);assert selected==225
    result=dict(schema='hey-broad-context-decision-v1',source_sha256=sha(Path(__file__)),training_only_geometry=matrices,
        context_frames=selected,rule='ceil to multiple3 of max(150, ceil((full original source-to-generated-prefix-end+960 RIR)/160)+27) across every training positive; no model response/heldout used.',
        protection='Original source sample0 through entire internal wake-prefix (Qwen entire bare waveform) plus complete RIR tail. Initial sentinel allocation remains protected for Kokoro; no alternative token-start rule adopted.',
        architecture='teacher60 convolution/BN transferred exactly, new wider classifier initialized once before both arms, exact initialization/schedule/export locks required.',
        optimization='Single120000 updates/arm, fixed final120000 checkpoint. Fresh independent development selects thresholds only; no old consumed-development checkpoint selection.',
        shape_probe_sha256=sha(RESULTS/'4090-hey-context-transfer-shape-probe-v1.json'),
        required_before_training=['Final full Qwen mechanical and independent artifact audits, all other source audits.',
            'Exact context225 initialization/target/sampler/optimizer/frontend/export lock, CPU actual batches, paired smoke and real waveform streaming export admission.'],
        limits='Geometry and CPU weight-shape feasibility only. Not same original context150 model/initializer; no training, export, recognition or promotion claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print('Selected context225 without dropping any generated prefix or original source samples')
if __name__=='__main__':main()
