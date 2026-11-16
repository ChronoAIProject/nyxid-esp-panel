"""Unified original wholewave synthetic training index, no feature/PCM copies."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 bare=R/'4090-generator-bare-support-v1.json.gz';commands=R/'4090-matchbox-joined-command-support-v1.json';intervalpath=R/'4090-joined-generator-token-intervals-v1.json.gz';admission=R/'4090-joined-command-pcm-admission-v1.json';records=[]
 with gzip.open(bare,'rt') as f:b=json.load(f)
 for x in b['records']:
  r=x['record'];records.append(dict(path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],lead=r['source_lead'],rir_tail=960 if r['variant']==2 else 0,conditioning=r['conditioning'],speed=r['scale'],variant=r['variant'],carrier='bare',class_id=r['class_id'],intervals=x['intervals'],role='training'))
 with gzip.open(intervalpath,'rt') as f:natives={r['path']:r for r in json.load(f)['records']}
 lookup=json.load(open(R/'4090-generator-token-target-census-v1.json'));nativepaths={r['path']:r['native_path'] for r in lookup['records']}
 for g in json.load(commands.open())['groups']:
  for label,r in g['classes'].items():records.append(dict(path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],lead=r['source_lead'],rir_tail=960 if g['variant']==2 else 0,conditioning=g['conditioning'],speed=g['speed'],variant=g['variant'],carrier=g['carrier'],class_id=int(label),intervals=natives[nativepaths[r['path']]]['intervals'],role='training'))
 assert len(records)==4320 and len({r['path'] for r in records})==4320
 vocab=[0,*sorted({t['token_id']+1 for r in records for t in r['intervals']})];assert len(vocab)==34
 cells=collections.defaultdict(dict)
 for i,r in enumerate(records):
  assert sha(r['path'])==r['sha256'];key=(r['carrier'],r['conditioning'],r['speed'],r['variant']);assert r['class_id'] not in cells[key];cells[key][r['class_id']]=i
 assert len(cells)==720 and all(set(c)==set(range(1,7)) for c in cells.values())
 data=dict(records=records,dense_to_original_label=vocab,cells=[dict(carrier=k[0],conditioning=k[1],speed=k[2],variant=k[3],indices=[g[i] for i in range(1,7)]) for k,g in sorted(cells.items())]);outpath=R/'4090-causal-unit-training-index-v1.json.gz'
 with outpath.open('xb') as f:f.write(gzip.compress((json.dumps(data,separators=(',',':'))+'\n').encode(),mtime=0))
 out=dict(schema='causal-unit-training-index-v1',files={str(p):sha(p) for p in [Path(__file__),bare,commands,intervalpath,admission,outpath]},records=len(records),matched_cells=len(cells),classes=6,unit_classes=len(vocab),carriers=dict(collections.Counter(r['carrier'] for r in records)),pcm_copied=False,policy='All4320existingKokorotrainingrenders, nofilter/crop ornewdata. NativePCM/admission inherited fromboundfullaudits, renderedSHA rechecked. Explicitdenseunitvocab34withoriginalpaddingtoken0mappedlabel1, unallocated0distinct.720cells=20conditioning*3speed*3variant*4carrier eachsixclass. Singleenginefamily, unknownphysical/pretrainingoverlap; notsourcegeneralizationproof. Nooptimization/heldout/qualification.')
 with (R/'4090-causal-unit-training-index-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
