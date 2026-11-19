"""Apply the published DNS3 GTCRN to complete receiver-side Opus, including the quiet controls."""
from collections import defaultdict
from hashlib import sha256
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import einops
import numpy as np
import soundfile as sf
import torch

import check_bella_product_history as history
import check_bella_product_interference as muted

P, prior = history.P, history.prior
OUT = P.RESULTS / "4090-c161-bella-gtcrn.json"
WORK = Path(".local/wakeword/c161-gtcrn")
REV = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
URL = "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/" + REV + "/"
ASSETS = {
    "gtcrn.py": "20bf0610702506fa45cd543454b84bbeb862d73c7c20a3f8d820fdc04da6099b",
    "infer.py": "1061bb6a9a6f1a27423f7603871e683aaf0cd3ffd40a75efa9f9ba0965330d7b",
    "checkpoints/model_trained_on_dns3.tar": "a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79",
    "test_wavs/mix.wav": "8d47e1d03eeb457c2549be79c8ec33a349ccd79f21c3add05f946f8f760c5a99",
    "test_wavs/enh.wav": "fcc240b64e728243b35e018eebf3f4e2ae889f0013d588f4c27448b6f0bef4dc",
}


def load():
    for name, digest in ASSETS.items():
        path = WORK/name
        if not path.exists():
            with urlopen(URL+name,timeout=30) as src:
                data = src.read()
            assert sha256(data).hexdigest() == digest
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
        assert P.sha(path) == digest
    spec = util.spec_from_file_location("pinned_gtcrn",WORK/"gtcrn.py")
    net = util.module_from_spec(spec)
    spec.loader.exec_module(net)
    model = net.GTCRN().eval()
    model.load_state_dict(torch.load(WORK/"checkpoints/model_trained_on_dns3.tar",weights_only=True,map_location="cpu")["model"])
    return model


def enhance(model, x, *, keep_length):
    device = next(model.parameters()).device
    signal = torch.from_numpy(x).to(device)
    window = torch.hann_window(512,device=device).sqrt()
    spec = torch.stft(signal,512,256,512,window,return_complex=True)
    result = model(torch.view_as_real(spec)[None])[0].contiguous()
    # torch 2.x represents ISTFT's complex input explicitly. Asking for the original length
    # also reconstructs the final partial hop from the centered STFT, without cropping the call.
    output = torch.istft(torch.view_as_complex(result),512,256,512,window,length=len(x) if keep_length else None)
    y = output.cpu().numpy()
    assert np.isfinite(y).all() and (not keep_length or len(y)==len(x))
    return y


def control(model):
    mixture, sr = sf.read(WORK/"test_wavs/mix.wav",dtype="float32")
    example, sr2 = sf.read(WORK/"test_wavs/enh.wav",dtype="float32")
    assert sr == sr2 == 16000
    cpu = enhance(model,mixture,keep_length=False)
    gpu = enhance(model.cuda(),mixture,keep_length=False)
    np.testing.assert_allclose(cpu,gpu,rtol=1e-4,atol=3e-5)
    assert len(cpu)==len(example)
    rms = lambda x: float(np.sqrt(np.mean(x.astype(float)**2)))
    return dict(input_samples=len(mixture),output_samples=len(gpu),cpu_gpu_rms=rms(cpu-gpu),
                cpu_gpu_max=float(np.max(np.abs(cpu-gpu))),demo_rms_difference=rms(gpu-example),
                demo_snr_db=float(20*np.log10(rms(example)/rms(gpu-example))),
                demo_correlation=float(np.corrcoef(gpu,example)[0,1]))


def recordings():
    for row, packets in history.captures():
        yield row["target"]+".panel1_music1", row, packets
    stored = P.read(muted.OUT)
    path = muted.OUT.with_suffix(".zip")
    assert P.sha(path) == stored["archive_sha256"]
    with ZipFile(path) as archive:
        for r in stored["records"]:
            if r["condition"] == "panel1_music1":
                continue
            i = r["index"]
            raw = archive.read(f'{i:02}/{r["condition"]}/opus.log')
            assert sha256(raw).hexdigest() == r["opus_uart_sha256"]
            text = raw.decode().replace("\r\n","\n")
            packets = prior.packets(text,i,r["samples"],0)
            assert len(packets) == r["packets"]
            row = dict(file=r["file"],command=r["command"],opened=r["opened"],db=r["db"],decoded_sha256=r["decoded_sha256"],
                       full_qwen=r["qwen"]["complete"],full_turbo=r["turbo"]["complete_suffix"])
            yield "nyx."+r["condition"],row,packets


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = load()
    witness = control(model)
    processor, qwen, identity = prior.load()
    versions = P.read(history.reference.OUT)
    assert identity == versions["qwen"] and "4090" in identity["device"]
    turbo = prior.reader(versions["turbo"])
    decoder = prior.decoder_api()
    rows, summary = [], {}
    groups = defaultdict(list)
    for name,row,packets in recordings():
        row = dict(row,group=name)
        if packets is not None:
            pcm = prior.decode(decoder,packets)
            assert P.digest(pcm) == row["decoded_sha256"]
            y = enhance(model,pcm.astype(np.float32)/32768,keep_length=True)
            rounded = np.rint(y*32768)
            cleaned = rounded.clip(-32768,32767).astype("<i2")
            q = prior.evaluate(processor,qwen,cleaned,row["command"])
            assert q["pcm_sha256"] == P.digest(cleaned)
            row.update(samples=len(cleaned),gtcrn_sha256=q["pcm_sha256"],clipped=int(np.count_nonzero(cleaned!=rounded)),
                       qwen=q,turbo=prior.recognition(turbo,cleaned,row["command"]))
        rows.append(row)
        groups[name].append(row)
    for name, cases in groups.items():
        assert len(cases)==24
        opened = [r for r in cases if r["opened"] is not None]
        counts = dict(cases=len(cases),opened=len(opened))
        for reader, key in (("qwen","complete"),("turbo","complete_suffix")):
            pairs = [(r["full_"+reader],r[reader][key]) for r in opened]
            counts[reader] = dict(original=sum(a for a,_ in pairs),enhanced=sum(b for _,b in pairs),
                                  gained=pairs.count((False,True)),lost=pairs.count((True,False)))
        summary[name] = counts
    assert len(rows)==120 and sum(v["opened"] for v in summary.values())==97
    with TemporaryDirectory(dir=WORK) as tmp:
        path = Path(tmp)/OUT.name
        P.save(path,dict(records=rows,summary=summary,control=witness,assets=ASSETS,source_url=URL,torch=torch.__version__,
               einops=einops.__version__,device=identity["device"],qwen=identity,turbo=versions["turbo"],
               parameters=sum(p.numel() for p in model.parameters()),sources={p.name:P.sha(p) for p in (
                   history.reference.OUT,history.comparison.OUT,muted.OUT,Path(history.__file__),Path(__file__))},
               policy="Official GTCRN DNS3 weights in FP32. One forward pass over each original complete, decoded "
                      "16 kHz FIFO on the host GPU. Square-root Hann STFT (512 samples, hop 256) following infer.py, "
                      "explicit original sample count for ISTFT. No prompt, crop, remixed original signal, or threshold "
                      "selection. Four fixed Nyx interference cells and the original Willow full-mixture comparator. "
                      "The three Nyx source-mute cells keep the original full-mixture open."))
        path.replace(OUT)
    print(summary)


if __name__ == "__main__":
    main()
