"""Run before generation/training/export: refuse insufficient scratch space."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--expected-gib',type=float,required=True);p.add_argument('--reserve-gib',type=float,default=50);p.add_argument('--path',default='.local/wakeword');a=p.parse_args()
if a.expected_gib<0 or a.reserve_gib<0:p.error('Budgets must be nonnegative')
u=shutil.disk_usage(Path(a.path));required=(a.expected_gib*1.5+a.reserve_gib)*2**30
print(json.dumps(dict(path=a.path,free_gib=round(u.free/2**30,2),expected_gib=a.expected_gib,reserve_gib=a.reserve_gib,required_gib=required/2**30,passed=u.free>=required)))
raise SystemExit(0 if u.free>=required else 2)
