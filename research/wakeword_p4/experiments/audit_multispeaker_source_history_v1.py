"""Bounded explicit-identity metadata census; never open audio or score models."""
import hashlib
import json
from pathlib import Path
import subprocess

R = Path('research/wakeword_p4/results')
PATTERN = r'(?i)\b(?:aru|arctic)\b|en_GB-aru|en_US-arctic|9e74d089a8563f8b2446426d01becb046cd3c3bfbafe1a20fd03a9a79bd82619|483303e294947a3ec2f910ea96093d876e1640f5772e9d89e511d6c82c667286'
GLOBS = ['*.json', '*.jsonl', '*.md', '!*/site-packages/*', '!*/pkgs/*',
         '!*/share/*', '!*/mfa-mamba-root/*', '!*/mfa-cpu-env/*']
ROOTS = ['.local/wakeword', 'research/wakeword_p4']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    common = [arg for glob in GLOBS for arg in ['-g', glob]]
    paths = sorted(subprocess.check_output(['rg', '--files', '--hidden', *common, *ROOTS], text=True).splitlines())
    inventory = [dict(path=p, bytes=Path(p).stat().st_size) for p in paths]
    found = subprocess.run(['rg', '-l', '--hidden', PATTERN, *common, *ROOTS], text=True, capture_output=True)
    assert found.returncode in (0, 1), found.stderr
    matches = []
    for name in sorted(found.stdout.splitlines()):
        raw = Path(name).read_bytes()
        matches.append(dict(path=name, bytes=len(raw), sha256=sha(raw)))
    out = Path('.local/wakeword/multispeaker-source-history-v1')
    out.mkdir(exist_ok=False)
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    report = dict(schema='multispeaker-source-history-v1',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  roots=ROOTS, globs=GLOBS, pattern=PATTERN,
                  metadata_files=len(paths), matches=matches,
                  inventory_path=str(out / 'inventory.json'),
                  inventory_sha256=sha((out / 'inventory.json').read_bytes()),
                  limits='Current accessible explicit-identity metadata only; not deleted history, other branches/hosts, implicit aliases, physical people or pretraining. Inventory binds file paths/sizes; only matched contents are hash-bound. Mentions are not actual usage or role assertions. Does not authorize synthesis without manual match review and material lock.',
                  detector_invocations=0, audio_opened=False)
    with (R / '4090-multispeaker-source-history-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(metadata_files=len(paths), matched_files=len(matches), paths=[m['path'] for m in matches])))


if __name__ == '__main__':
    main()
