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
2. Keep validation loss within 15% of training loss and export a TorchScript
   encoder under 10 MiB, so it is practical to distill or port later.
3. Produce a dated JSON metrics file and a confusion matrix for every run.
4. Compare synthetic-only results with a small, consented fixture set before
   making any claim about real-world voice quality. Hardware validation is a
   separate gate and is intentionally not required for this phase.

## Quick start (4090)

```bash
cd research/voice
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python train.py --device cuda --epochs 40 --output runs/baseline
```

The script checks CUDA, selects the 4090 when available, uses mixed precision,
and caps memory growth through the configured batch size. For a CPU smoke run:

```bash
python train.py --device cpu --epochs 1 --samples 128 --output runs/smoke
```

All inputs are generated deterministically from `--seed`; no microphone,
network, board, or NyxID service is contacted.

