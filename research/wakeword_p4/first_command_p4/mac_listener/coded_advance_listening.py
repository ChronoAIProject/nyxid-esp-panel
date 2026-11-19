"""Transcribe isolated source WAVs with Whisper and greedy wav2vec2, and archive all variants."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import coded_advance as advance
import check_c161_pronunciation_ctc_v1 as ctc
from run_c161_first_command_v5 import transcripts

ch=advance.challenge
STEM=advance.STEM+"-isolated"


def main():
    files=json.loads((ctc.MODEL/"source.json").read_bytes())["files"]
    assert files["model.safetensors"]["sha256"]==ctc.read("ctc-training-audit-plan")["expected_safetensors_sha256"]
    for name,meta in files.items():
        with (ctc.MODEL/name).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==meta["sha256"]
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    processor=Wav2Vec2Processor.from_pretrained(ctc.MODEL,local_files_only=True)
    net=Wav2Vec2ForCTC.from_pretrained(ctc.MODEL,local_files_only=True).cuda().eval()
    reader,identity=ch.load_reader()
    bank,_=ch.dry_bank()
    rows=[]
    pack=ch.RESULTS/f"{STEM}.zip"
    with torch.inference_mode(), ZipFile(pack,"x",ZIP_DEFLATED) as z:
        for voice in ch.VOICES:
            original=bank[voice,"Hey Nyx."]
            tail,early=advance.cue_pair(original)
            for label,pcm in (("ordinary",original[:len(early)]),("tail",tail),("advance",early)):
                inputs=processor(pcm.astype(np.float32)/32768,sampling_rate=16000,return_tensors="pt")
                ids=net(input_values=inputs.input_values.cuda()).logits.argmax(-1)
                rows.append(dict(voice=voice,mode=label,whisper=transcripts(reader,pcm),
                    wav2vec2=processor.batch_decode(ids.cpu())[0],
                    **advance.uplink.write_member(z,f"{voice}-{label}.wav",pcm)))
        z.writestr("manifest.json",json.dumps(dict(rate=16000,channels=1,format="PCM16",records=rows),indent=2)+"\n")
    ch.save(ch.RESULTS/f"{STEM}.json",dict(records=rows,pack_sha256=ch.sha(pack),whisper_files=identity,
        wav2vec2_files=files,gpu=torch.cuda.get_device_name(),source_sha256=ch.sha(ch.DRY),code_sha256=ch.sha(Path(__file__)),
        policy="Original, tail-coded and earlier-coded isolated synthetic cues share the final active frame. CUDA Whisper small.en without prompt and greedy wav2vec2-base-960h CTC without language model. Listening quality awaits physical playback."))


if __name__=="__main__":
    main()
