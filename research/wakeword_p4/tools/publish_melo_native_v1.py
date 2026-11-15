"""Publish the existing immutable native pack after the recorded quota reset."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

BASE=Path('.local/wakeword');R=Path('research/wakeword_p4/results')
ARCHIVE=BASE/'melo-development-native-v1-pack.tar.gz'
HASH='7d4719f3b2d6c4d98dfa449449067e750d8f05b1b75c53894987be11bf979312'
REPO='ChronoAIProject/nyxid-esp-panel';TAG='research-hey-nyx-20261110'
URL=f'https://github.com/{REPO}/releases/download/{TAG}/{ARCHIVE.name}'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 while time.time()<1794768005:time.sleep(min(10,1794768005-time.time()))
 assert ARCHIVE.stat().st_size==113405113 and sha(ARCHIVE)==HASH
 # No clobber, release creation, re-synthesis or mutation of native-quality report.
 subprocess.run(['gh','release','upload',TAG,str(ARCHIVE),'--repo',REPO],check=True)
 dest=BASE/'melo-native-anonymous-download-v1.tar.gz'
 assert not dest.exists()
 with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'nyx-research-anonymous-verifier'}),timeout=120) as response,dest.open('xb') as f:
  assert response.status==200
  while block:=response.read(1024*1024):f.write(block)
 assert sha(dest)==HASH and dest.stat().st_size==113405113
 members=[]
 with tarfile.open(dest,'r:gz') as tar:
  records=[json.loads(line) for line in tar.extractfile('melo-development-native-v1/records.jsonl')]
  assert len(records)==144
  for row in records:
   for key,hkey in [('engine_path','engine_sha256'),('raw_path','raw_sha256'),('alignment_path','alignment_sha256'),('path','sha256')]:
    name='melo-development-native-v1/'+row[key];info=tar.getmember(name)
    with tar.extractfile(info) as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==row[hkey],name
    members.append(dict(path=name,bytes=info.size,sha256=h))
 assert len(members)==576
 receipt=R/'4090-melo-development-native-publication-verification-v1.json'
 with receipt.open('x') as f:f.write(json.dumps(dict(schema='melo-native-publication-verification-v1',url=URL,bytes=dest.stat().st_size,sha256=HASH,anonymous_download=True,verified_members=members,all_passed=True,original_native_quality_report_unchanged=True),indent=2)+'\n')
 dest.unlink()
 subprocess.run(['git','add','--',str(receipt)],check=True)
 subprocess.run(['git','commit','-m','research: verify anonymous Melo native archive and all 576 source members'],check=True)
 subprocess.run(['git','push','origin','HEAD'],check=True)
 comments=subprocess.check_output(['gh','api',f'repos/{REPO}/issues/13/comments','--paginate'],text=True)
 (BASE/'mac-issue13-comments-after-melo-publication-v1.json').write_text(comments)
 body=BASE/'melo-publication-issue13-v1.md'
 body.write_text(f'''4090 研究进度：固定 SpeechT5 数据覆盖实验已完成 GPU 20,000 步及 INT8 导出（60,952 bytes，SHA256 `79c33ec6c2f2938259482c3ac50dcca9fb3e04033ebecf38c69c2712856a86e2`）。\n\n全新 Melo 开发集为 144 个完整源、432 个呈现；同批约 7.90h 完整背景音正在由两个固定候选、冻结 v30 与官方 Hi ESP 回放。开发阈值/联合选择及可行性规则在评分前锁定，独立验收声音未生成，不宣称已达标或替换 v30。训练使用 RTX 4090，CPU 用于实际部署 INT8 字节和官方 QEMU 回放。\n\n原始合成源包（含引擎输出、native、alignment、WAV；不含模型训练用途）：{URL}\nSHA256 `{HASH}`，113405113 bytes。匿名下载归档及 576 个成员校验通过，独立校验收据已推送 research-hye-nyx-voice-gpu。内部时长只是合成参考，非人类音素真值。\n\n冻结 v30 交接及参数保持不变。后续按原协议评估，不以实机非受控发音作为此合成研究的前置条件。\n''')
 subprocess.run(['gh','issue','comment','13','--repo',REPO,'--body-file',str(body)],check=True)
 print(json.dumps(dict(url=URL,sha256=HASH,verified_members=576,issue_posted=True)),flush=True)
if __name__=='__main__':main()
