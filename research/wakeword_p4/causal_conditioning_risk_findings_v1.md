# Paired conditioning risk experiment: rejected

Both original RTX4090 arms completed5000steps; original session76101 exited0. Full training audit passed: identical initial model, all5000batch descriptors identical and reconstructed from locked sampler, original final artifact hashes intact, risk multipliers reconstructed from detached training losses, final risk state identical. Fixed eta0.1 and gradient clip5 retained, no restart/checkpoint/threshold/coefficient rescue.

| Fixed final checkpoint | Training target | Training negative clips | Reused development target | Reused development negative clips |
| --- | --- | --- | --- | --- |
| Uniform balanced | 439/576 | 46/2880 | 54/144 | 27/720 |
| Cycle group risk | 430/576 | 49/2880 | 51/144 | 21/720 |

All four full-wave replays exited0 in original session81189. Exact raw float argmax matcher, unchanged frontend and37frame chunk/full check. These are recording activation counts, not event-rate estimates, confidence-policy evaluation or trained INT8 qualification. The risk arm reduces development negative activation by6 but loses3target hits; both retain poor target coverage. The fixed intervention does not justify progression to export or sealed acceptance. No claim that group robustness generally cannot work; this recipe fails to resolve the observed conditioning transfer problem.

Uniform final checkpoint SHA256 bbaf016f8b0bb868dc7f9bba32ae41acff93e6d8b2d7affa2f4f2c9993837ddb; risk final e3545ee87daa50a9653c199f6862a26f56f5a6b2b74fae636e774ace3d0fd5d2. Both original training reports and all4complete waveform diagnostic JSONs published losslessly with bytes/SHA256 in4090-conditioning-paired-evidence-v1.json. No new PCM/features cache. Development remains reused, not fresh or independent. No protected final data accessed, no FAR or official matched INT8 claim from this experiment. Frozenv30 retained; larger research goal remains incomplete.
