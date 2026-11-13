"""Audit an immutable completed prefix of the live official grid, no metrics."""
import argparse,json,hashlib,zlib
from pathlib import Path
import soundfile as sf
from calibrate_development import audit_stream
from run_development_grid import PLAN,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    p=json.loads(PLAN.read_text());helper=Path(__file__).with_name('calibrate_development.py')
    assert sha(helper)==next(v for k,v in p['files'].items() if Path(k).resolve()==helper.resolve())
    log=Path('.local/wakeword/hey-official-development-sequence-v1.log')
    completed=[]
    for line in log.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):continue
        row=json.loads(line)
        if 'completed' in row:completed.append(row)
    assert completed and [r['completed'] for r in completed]==list(range(1,len(completed)+1))
    runtime={key:sha(path) for key,path in p['runtime'].items()}
    for key,path in p['runtime'].items():assert runtime[key]==p['files'][path]
    crc_cache={};records=[]
    for item in completed:
        model=item['model'];config=item['configuration'];index=item['source']
        ordinal=p['models'].index(model)*13*372+config*372+index+1
        assert item['completed']==ordinal and item['total']==p['streams']
        row=p['records'][index]
        if index not in crc_cache:
            assert sha(row['path'])==row['sha256'];crc=0;h=hashlib.sha256();count=0
            with sf.SoundFile(row['path']) as f:
                assert f.channels==1 and f.samplerate==16000 and f.subtype=='PCM_16'
                while True:
                    values=f.read(1048576,dtype='int16')
                    if not len(values):break
                    raw=values.astype('<i2').tobytes();crc=zlib.crc32(raw,crc);h.update(raw);count+=len(values)
            assert count==row['samples'] and h.hexdigest()==row['pcm_sha256'];crc_cache[index]=crc
        r,events,audit=audit_stream(Path(p['output']),f'{model}-{config:02}-{index:03}',row,model,p['threshold_requests'][config],runtime,crc_cache[index])
        records.append(dict(ordinal=ordinal,model=model,configuration=config,source_index=index,phase=row['phase'],input_samples=r['input_samples'],input_calls=r['input_calls'],**audit))
    result=dict(schema='hey-official-development-snapshot-audit-v1',auditor_sha256=sha(__file__),helper_sha256=sha(helper),execution_sha256=sha(PLAN),completed_prefix=len(completed),planned_streams=p['streams'],audited_calls=sum(r['input_calls'] for r in records),records=records,recognition_metrics_computed=False,far_computed=False,acceptance_accessed=False,limits='Completed immutable stream prefix only; not complete configurations, operating-point selection or official recognition/FAR conclusion. Original live task is untouched; no reruns.')
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
