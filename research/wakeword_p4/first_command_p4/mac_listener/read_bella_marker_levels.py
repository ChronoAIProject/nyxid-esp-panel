"""Transcribe the three isolated af_bella addresses with the same complete-PCM readers."""
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

import check_bella_marker_only as cue

P = cue.P
OUT = P.RESULTS / "4090-c161-bella-marker-level-readings.json"


def main():
    parent = P.read(cue.OUT)
    proc, model, qid = cue.prior.load()
    assert qid == parent["qwen"]
    turbo = cue.prior.reader(parent["turbo"])
    rows = []
    for label, path in (("original",cue.overlap.board.bella.CUE), ("enhanced",cue.overlap.peak.CUE),
                        ("marker_only",cue.CUE)):
        pcm,sr = sf.read(path,dtype="int16")
        assert sr==16000 and len(pcm)==23200
        a = cue.prior.evaluate(proc,model,pcm,"Hey Nix")
        b = cue.prior.recognition(turbo,pcm,"Hey Nix")
        rows.append(dict(version=label,file=path.name,sha256=P.sha(path),pcm_sha256=P.digest(pcm),
                         samples=len(pcm),qwen=a["text"],turbo=b["text"]))
    known=parent["cue"]["dry_readings"]
    assert (rows[-1]["qwen"],rows[-1]["turbo"])==(known["qwen"]["text"],known["turbo"]["text"])
    with TemporaryDirectory(dir=cue.WORK) as temp:
        target=Path(temp)/OUT.name
        P.save(target,dict(records=rows,source_sha256=P.sha(cue.OUT),script_sha256=P.sha(Path(__file__)),
               qwen=qid,turbo=parent["turbo"],policy="Complete 16 kHz dry isolated handoff WAVs at the encoded gain. "
               "Report both unprompted English transcripts verbatim. No denoising, cropping or spelling substitution."))
        target.replace(OUT)
    print(rows,flush=True)


if __name__=="__main__":
    main()
