"""Exercise invalid preregistrations without loading a final detector model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh audit output required')
    code = Path(__file__)
    evaluator = code.with_name('evaluate_independent_dialogue.py')
    manifest = json.loads(a.manifest.read_text())
    freeze = dict(final_role='independent-dialogue-final',nyx_model_sha256=sha(a.model),
                  pcm_manifest_sha256=sha(a.manifest),nyx_evaluator_source_sha256=sha(evaluator),
                  detector_source_sha256=sha(code.with_name('evaluate.py')),
                  frontend_source_sha256=sha(code.with_name('features.py')),
                  source_audit_sha256=manifest['source_audit_sha256'],
                  source_manifest_sha256=manifest['source_manifest_sha256'],
                  expected_streams=16,sessions=sorted(r['session'] for r in manifest['records']),
                  unique_participants=16,expected_source_hours=manifest['source_hours'],
                  confidence=.95,false_per_hour_upper_95_limit=.5,sliding_window=5,
                  nyx_threshold=.91796875,limits='INVALID guard fixture; never an acceptance preregistration')
    expected = {
        'changed-model':'Changed preregistered model, code, PCM population or source audit',
        'nonfinite-threshold':'Changed frozen test population, exposure or operating point',
        'missing-session':'Changed frozen test population, exposure or operating point',
        'changed-cached-frontend':'Changed preregistered model, code, PCM population or source audit',
        'escaped-input':'Changed or escaped final input',
    }
    results = []
    with tempfile.TemporaryDirectory(prefix='nyx-dialogue-invalid-') as temp:
        root = Path(temp)
        for case,message in expected.items():
            invalid = copy.deepcopy(freeze)
            use_manifest = a.manifest
            if case == 'changed-model':
                invalid['nyx_model_sha256'] = '0'*64
            elif case == 'nonfinite-threshold':
                invalid['nyx_threshold'] = float('nan')
            elif case == 'missing-session':
                invalid['sessions'] = invalid['sessions'][:-1]
            else:
                fixture = copy.deepcopy(manifest)
                if case == 'changed-cached-frontend':
                    fixture['frontend_source_sha256'] = '0'*64
                else:
                    row = fixture['records'][0]
                    row['path'] = str((a.manifest.parent/row['path']).resolve())
                use_manifest = root/(case+'-manifest.json')
                use_manifest.write_text(json.dumps(fixture))
                invalid['pcm_manifest_sha256'] = sha(use_manifest)
            prereg = root/(case+'.json')
            prereg.write_text(json.dumps(invalid))
            out = root/(case+'-output')
            proc = subprocess.run([sys.executable,str(evaluator),'--model',str(a.model),
                                   '--manifest',str(use_manifest),'--preregistration',str(prereg),
                                   '--output',str(out)],capture_output=True,text=True)
            if not proc.returncode or out.exists() or 'ValueError: '+message not in proc.stderr:
                raise RuntimeError(f'Guard failed or reached unexpected path: {case}: {proc.stderr}')
            results.append(dict(case=case,rejected_before_output_creation=True,
                                error=proc.stderr.splitlines()[-1]))
    result = dict(role='Invalid dialogue evaluator guard checks; no final model inference',
                  source_sha256=sha(code),evaluator_source_sha256=sha(evaluator),guards=results,
                  independent_final_scored=False,
                  limits='All ephemeral fixtures deliberately invalid and deleted. Expected rejection branches checked before detector construction/output. Not recognition quality evidence.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
