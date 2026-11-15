"""Unchanged consumed-recording collector bound to the grid-aligned export."""
import argparse
from pathlib import Path
SCRIPT=Path('research/wakeword_p4/experiments/collect_matchbox_long_recovered_melo_v1.py')
def main(action):
 s=SCRIPT.read_text()
 old='BASE/f"matchbox-long-{ARM}-{\'resumed\' if ARM==\'control\' else \'recovery\'}-v1/frontend.pt"'
 assert s.count(old)==2;s=s.replace(old,"BASE/'matchbox-grid-aligned-v1/frontend.pt'")
 s=s.replace("f'4090-matchbox-long-{ARM}-int8-original-v1.json'","'4090-matchbox-grid-aligned-int8-original-v1.json'")
 s=s.replace("f'4090-matchbox-long-{ARM}-consumed-melo-original-v1.json'","'4090-matchbox-grid-aligned-consumed-melo-original-v1.json'")
 s=s.replace('paths=[Path(__file__),',"paths=[Path(__file__),Path('research/wakeword_p4/experiments/collect_matchbox_grid_aligned_melo_v1.py'),Path('research/wakeword_p4/experiments/matchbox_grid_alignment_protocol_v1.md'),")
 ns=dict(__name__='grid_aligned_melo_adapter',__file__=str(SCRIPT));exec(compile(s,str(SCRIPT),'exec'),ns)
 ns.update(ARM='pooled',LOCK=Path('research/wakeword_p4/results/4090-matchbox-grid-aligned-consumed-melo-plan-v1.json'),OUT=Path('.local/wakeword/matchbox-grid-aligned-consumed-melo-v1'));ns[action]()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();main(args.action)
