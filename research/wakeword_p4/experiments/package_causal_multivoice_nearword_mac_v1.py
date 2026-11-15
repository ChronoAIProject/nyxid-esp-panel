"""Deliver original multivoice-nearword bytes and a simple experimental Mac profile."""
import gzip,hashlib,io,json,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/causal-multivoice-nearword-mac-handoff-v1')
MODEL=Path('.local/wakeword/causal-multivoice-nearword-int8-v1/int8.tflite');EXPECTED='95aba77a68610be66c0032500be3dd31962b8ec02a07e7398c26e38759ada5b3'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
 data=MODEL.read_bytes();assert len(data)==60952 and digest(data)==EXPECTED;original=json.loads((R/'4090-causal-multivoice-nearword-int8-original-v1.json').read_text());assert original['model']['sha256']==EXPECTED
 manifest=dict(schema='causal-multivoice-nearword-mac-handoff-v1',source_commit='1b781102',results_commit='d1084ec2',model=dict(file='hey_nyx.tflite',bytes=len(data),sha256=EXPECTED,original_path=str(MODEL),export_report=str(R/'4090-causal-multivoice-nearword-int8-original-v1.json')),profile=dict(context=309,threshold=.91796875,smoothing_steps=5,stride=3,input_dtype='int8',input_shape=[1,3,40],output_dtype='uint8',output_shape=[1,1],output_scale=1/256,output_zero_point=0),deployment=dict(existing_prime_samples=81600,frontend='existing unchanged frontend',event_policy='existing smoothing/rearm/cooldown, continuous through priming; ignore only application priming notifications'),status='Experimental candidate for actual P4 functionality/stability evaluation; v30 not replaced',observed_consumed_behavior=dict(target_any='33/36',commands='25/27',light='7/9',common_false_clips='1/288; one Hey next light variant',jenny_target='33/36',jenny_commands='24/27',jenny_common_false_clips='48/324; prior55/324, v30 12/324',tradeoff='Both batches lose one target hit versus prior9307 candidate; modest confusion improvement only',background_events=0,background_hours=10.483944826388889),limits='Original export bytes; no re-export, PCM selftest or numerical admission requirement. Generated-end early bare events are not by themselves failure or phonetic truth. Known nearword errors and misses retained; independent acceptance and officialHiESP same-protocol comparison remain incomplete. No audio or credentials included.')
 raw=(json.dumps(manifest,indent=2)+'\n').encode();OUT.mkdir(exist_ok=False);archive=OUT/'gpu-causal-multivoice-nearword-mac-v1.tar.gz'
 with archive.open('xb') as target,gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
  for name,body in [('hey_nyx.tflite',data),('manifest.json',raw)]:
   info=tarfile.TarInfo(name);info.size=len(body);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(body))
 (R/'4090-causal-multivoice-nearword-mac-manifest-v1.json').write_bytes(raw)
 result=dict(schema='causal-multivoice-nearword-mac-package-v1',archive_path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),members=[dict(name='hey_nyx.tflite',bytes=len(data),sha256=EXPECTED),dict(name='manifest.json',bytes=len(raw),sha256=digest(raw))],download_url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+archive.name,anonymous_download_verified=False)
 (R/'4090-causal-multivoice-nearword-mac-package-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
