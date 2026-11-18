"""Audit the frozen WAV transcription, P4 object state, and first-command sample layout."""
from pathlib import Path
import shlex
import subprocess
import tempfile

from faster_whisper import WhisperModel
import numpy as np

from check_c161_coded_cue_v5 import HEADER
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import COEFF, STEM, WAV
from select_c161_synthesized_cue_v2 import load_wav


def compile_for_p4(directory):
    db=Path(".local/wakeword/c161-vad-build-v1/app/build/compile_commands.json")
    entry=next(row for row in read(db) if row["file"].endswith(".cpp"))
    cmd=shlex.split(entry["command"])
    src=directory/"matcher.cpp"
    obj=directory/"matcher.o"
    src.write_text('#include "c161_coded_cue_v5.h"\nextern "C" {\nnyx::CodedCueV5 cue;\nbool check_pcm(short x) { return cue.update(x); }\n}\n')
    cmd[cmd.index("-o")+1]=str(obj)
    cmd[cmd.index("-c")+1]=str(src)
    if "-MF" in cmd:
        cmd[cmd.index("-MF")+1]=str(directory/"matcher.d")
    cmd.extend(["-I",str(HEADER.parent.resolve())])
    subprocess.run(cmd,cwd=entry["directory"],check=True)
    tool=Path(cmd[0])
    out=subprocess.check_output([str(tool.with_name("riscv32-esp-elf-nm")),"-S","-C",str(obj)],text=True)
    symbol=next(s for s in out.splitlines() if s.endswith(" cue"))
    return dict(compiler=str(tool),compiler_sha256=sha(tool),object_sha256=sha(obj),source=src.read_text(),
                p4_state_bytes=int(symbol.split()[1],16),symbols=out.splitlines(),build_rule=entry)


def main():
    spec=read(RESULTS/f"{STEM}-source.json")
    assert sha(COEFF)==spec["coefficients_sha256"]
    cue=load_wav(dict(path=WAV,sha256=spec["wav_sha256"]))
    model=Path(".local/wakeword/whisper-small-en")
    plan=read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")
    assert all(sha(p)==plan["files"][str(p)] for p in model.iterdir() if p.is_file())
    asr=WhisperModel(str(model),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    seg,_=asr.transcribe(cue.astype(np.float32)/32768,language="en",temperature=0,beam_size=3,
                         condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
    words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability) for s in seg for w in s.words or []]
    commands=read(RESULTS/f"{STEM}-commands.json")
    with tempfile.TemporaryDirectory() as temp:
        p4=compile_for_p4(Path(temp))
    assert p4["p4_state_bytes"]==commands["state_bytes"]
    witnesses=[]
    for r in commands["records"]:
        if "wav" in r:
            assert sha(RESULTS/r["wav"])==r["wav_sha256"]
            witnesses.append(dict(wav=r["wav"],sha256=r["wav_sha256"],channel=r["source"]["channel"],
                notification=r["events"],cue_start=r["source"]["start"],command_start=r["source"]["end"]))
    save(RESULTS/f"{STEM}-audit.json",dict(cue_sha256=sha(WAV),header_sha256=sha(HEADER),coefficients_sha256=sha(COEFF),
         p4=p4,host_state_bytes=commands["state_bytes"],whisper_model={p.name:sha(p) for p in model.iterdir() if p.is_file()},
         transcript="".join(w["word"] for w in words).strip(),words=words,witnesses=witnesses))


if __name__=="__main__":
    main()
