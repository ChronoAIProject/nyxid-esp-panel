"""Download only two frozen development shards; never open acceptance audio."""
import hashlib
import json
from pathlib import Path
import urllib.request

PLAN = Path('research/wakeword_p4/results/4090-prospective-background-role-plan-v1.json')
EXPECTED_PLAN_SHA = '09d296b391ac157b3141dda0f36a9fc803852ff97fb8cd2f8f672fee5fe4d41d'

def main():
    raw = PLAN.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_PLAN_SHA
    plan = json.loads(raw)
    output = Path('.local/wakeword/prospective-background-development-source-v1')
    output.mkdir(exist_ok=False)
    (output / 'source-plan-original.json').write_bytes(raw)
    receipts = []
    try:
        for entry in plan['shards']:
            if entry['role'] != 'development': continue
            name = Path(entry['path']).name
            partial = output / (name + '.partial')
            digest = hashlib.sha256(); size = 0
            url = f"https://huggingface.co/datasets/{plan['repository']}/resolve/{plan['revision']}/{entry['path']}"
            print(json.dumps(dict(downloading=name, expected_size=entry['size'])), flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as dest:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block: break
                    dest.write(block); digest.update(block); size += len(block)
            assert size == entry['size'] and digest.hexdigest() == entry['sha256']
            partial.rename(output / name)
            receipts.append(dict(path=name, size=size, sha256=digest.hexdigest(), role='development'))
            (output / 'download-progress.json').write_text(json.dumps(receipts, indent=2)+'\n')
        report = dict(plan_sha256=EXPECTED_PLAN_SHA, files=receipts,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            acceptance_downloaded=False, audio_decoded=False, detector_scored=False,
            limits='Raw downloads only. No continuity/duration/identity/PCM-disjointness qualification yet.')
        (output / 'download.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__, message=str(error),
            completed_files=receipts, automatic_retry=False), indent=2)+'\n')
        raise

if __name__ == '__main__': main()
