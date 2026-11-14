"""Run the existing-priming diagnostic over a fixed consumed-input subset."""
import argparse
import json
from pathlib import Path

from replay_panel_priming_mac_v1 import replay, sha


def run(plan_path, packet, models, grid, output):
    plan = json.loads(plan_path.read_text())
    if json.loads(grid.read_text())['thresholds'] != plan['thresholds']:
        raise ValueError('Use the original threshold grid')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    manifest = json.loads((models / 'manifest.json').read_text())
    for item in manifest['files']:
        if sha(models / item['name']) != item['sha256']:
            raise ValueError('Original model file differs from handoff')
    rows = plan['records']
    for row in rows:
        wav = packet / 'audio' / Path(row['path']).name
        if sha(wav) != row['sha256']:
            raise ValueError('Consumed WAV differs from source record')
    result = dict(schema='mac-existing-priming-subset-v1',
                  plan_sha256=sha(plan_path), script_sha256=sha(__file__),
                  clips=len(rows), background_hours=0, candidate_promoted=False,
                  limits='Consumed subset only; not fresh development, background FAR, P4 or acceptance.',
                  records=[])
    for model in plan['models']:
        for i, row in enumerate(rows):
            destination = output / model / f'{i:06d}'
            current = replay(models / f'hey_nyx-{model}.tflite',
                             packet / 'audio' / Path(row['path']).name,
                             plan['context_frames'], grid, destination)
            result['records'].append(dict(model=model, index=i, source=row,
                                          result=str(destination.relative_to(output) / 'result.json'),
                                          phases=current['phases']))
            (output / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
            if (i + 1) % 12 == 0:
                print(json.dumps(dict(model=model, complete=i + 1, total=len(rows))), flush=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(complete=True, records=len(result['records']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'packet', 'models', 'grid', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.packet, args.models, args.grid, args.output)
