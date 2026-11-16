import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');files=[R/f'4090-causal-{k}-all-training-wave-diagnostic-v1.json' for k in ['unit','position','contextual']]
out={'schema':'causal-candidate-selection-v1','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*files]},'candidates':{}}
for name,p in zip(['frame','position','contextual'],files):
 r=json.load(p.open());out['candidates'][name]={'target_hits':r['target_hits'],'target_count':r['target_count'],'negative_hits':r['negative_hits'],'negative_count':r['negative_count'],'qualification':False}
print(json.dumps(out,indent=2));(R/'4090-causal-candidate-selection-v1.json').write_text(json.dumps(out,indent=2)+'\n')
