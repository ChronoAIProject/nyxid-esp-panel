"""Reuse frozen graph/conversion recipe for the grid-aligned final only."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');SCRIPT=Path('research/wakeword_p4/experiments/export_matchbox_long_recovered_v1.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source():
 s=SCRIPT.read_text();s=s.replace("parent=R/'4090-matchbox-long-training-plan-v1.json'","parent=R/'4090-matchbox-grid-aligned-training-plan-v1.json'")
 s=s.replace("subprocess.run([sys.executable,str(Path(__file__)),'pack','--arm',ARM],check=True)","pack()")
 s=s.replace("paths=[Path(__file__),","paths=[Path(__file__),Path('research/wakeword_p4/experiments/export_matchbox_grid_aligned_v1.py'),")
 s=s.replace("f'4090-matchbox-long-{ARM}-int8-original-v1.json'","'4090-matchbox-grid-aligned-int8-original-v1.json'")
 return s
def main(action):
 ns=dict(__name__='grid_export_adapter',__file__=str(SCRIPT));exec(compile(source(),str(SCRIPT),'exec'),ns)
 ns.update(ARM='pooled',TRAIN=Path('.local/wakeword/matchbox-grid-aligned-v1'),OUT=Path('.local/wakeword/matchbox-grid-aligned-int8-v1'),LOCK=R/'4090-matchbox-grid-aligned-int8-plan-v1.json',ORIGINAL=R/'4090-matchbox-grid-aligned-training-original-v1.json')
 return ns[action]()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','pack','run']);args=a.parse_args();raise SystemExit(main(args.action))
