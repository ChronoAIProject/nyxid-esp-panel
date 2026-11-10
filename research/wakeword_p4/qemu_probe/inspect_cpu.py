"""Read CPU/QOM properties without booting firmware or scoring audio."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--qemu', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('Use a fresh report path')
command = [str(a.qemu), '-machine', 'none', '-cpu', 'espressif', '-display', 'none',
           '-nodefaults', '-S', '-qmp', 'stdio']
requests = [dict(execute='qmp_capabilities'), dict(execute='query-cpu-definitions'),
            dict(execute='qom-list-properties', arguments={'typename': 'espressif-riscv-cpu'}),
            dict(execute='quit')]
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
try:
    output, error = process.communicate(''.join(json.dumps(r)+'\n' for r in requests), timeout=15)
except BaseException:
    process.kill(); process.communicate(); raise
responses = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
if process.returncode or any('error' in r for r in responses):
    raise RuntimeError(f'QMP query failed: {error} {responses}')
properties = next(r['return'] for r in responses if isinstance(r.get('return'), list)
                  and any(x.get('name') == 'zicsr' for x in r['return']))
result = dict(role='Read-only official QEMU CPU capability inspection; no inference',
              command=command, requests=requests, responses=responses, exit_code=process.returncode,
              stderr=error, qemu_sha256=hashlib.sha256(a.qemu.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              property_count=len(properties),
              pie_named_properties=[x for x in properties if 'pie' in x['name'].lower()],
              detect_completed=False,
              limits='No property with PIE in its name was found. This is not an exhaustive proof of unsupported instructions or future emulator capability. Prior actual WakeNet instruction fault remains the runtime evidence; no Hi ESP quality comparison.')
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('responses', 'requests')}, indent=2))
