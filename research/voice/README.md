# Hye Nyx voice research (simulation)

This directory is the board-free research track for Hye Nyx voice. It trains a
small speaker/style embedding model on **synthetic log-mel features**. The
dataset contains no recordings, credentials, or device-specific assumptions;
the same checkpoint can later be evaluated against an exported fixture from
the ESP voice pipeline.

## 2026-Q4 goals

1. Establish a reproducible GPU baseline on the RTX 4090: 10,000 synthetic
   utterances, 20 speakers, 40 epochs, and a held-out speaker classification
   accuracy of at least 90%.
2. Keep the validation/train accuracy gap within 5 percentage points and export a TorchScript
   encoder under 10 MiB, so it is practical to distill or port later.
3. Produce a dated JSON metrics file and a confusion matrix for every run;
   reload the export and check its logits against the original model.
4. Compare synthetic-only results with a small, consented fixture set before
   making any claim about real-world voice quality. Hardware validation is a
   separate gate and is intentionally not required for this phase.

## Quick start (4090)

System Python installation on this host is complete: PyTorch 2.15.0+cu132,
NumPy 2.4.6, Python 3.11. Exact installed package versions are recorded in
`environment.lock`. Run directly with `python3` on this host; the isolated
environment example below is for other machines.

```bash
cd research/voice
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python train.py --device cuda --epochs 40 --output runs/baseline
```

The script requires CUDA when requested, uses the default CUDA device and
mixed precision, and controls batch memory through batch size. For a CPU smoke run:

```bash
python train.py --device cpu --epochs 1 --samples 128 --output runs/smoke
```

All inputs are generated deterministically from `--seed`; no microphone,
network, board, or NyxID service is contacted.

## Scope and previous research

These inputs are random arrays with class-specific patterns, not synthesized
speech waveforms or measured log-mel features. High classification accuracy
only verifies optimization/export plumbing. This is not a wake-word detector,
TTS model, or evidence that Hey Nyx recognition quality improved.

The actual preceding wake-word experiments are on
`origin/research/macos-custom-wakeword` under `research/wakeword_p4`.
They use microWakeWord/TensorFlow and AI-generated speech waveforms for
**Hey Nyx** (pronounced **Hey Nicks**). Their selected quantized model was
rejected: 62.11% synthetic regression recall and 0/50 Mac regression recall.
Those experiments require a separate waveform corpus and feature extraction;
installing PyTorch does not reproduce them. Subsequent recognition work must
retain source/voice-disjoint splits, quantized streaming evaluation and
continuous-negative exposure, and report synthetic results separately from
real-user validation. No board is needed for synthesis or model training.

## Executed GPU baseline

On 2026-11-10, 40 epochs on RTX 4090 completed with 8,000 training and 2,000
validation examples. Both accuracies were 100%; the exported model was
244,869 bytes and passed logits comparison after reload. See
`results/4090-baseline.json` for the configuration, source hash, full history
and confusion matrix. Model weights are local at
`runs/4090-baseline/encoder.ts` and are excluded from git. This deliberately
easy random-feature task is a training-system check only.

