# Paired context training continuation

Training supervisor: `.local/wakeword/context-pair-supervisor-v1/state.json`. Verify live `/proc/PID/stat` starttime against the recorded identity, then inspect each arm's progress and terminal exit code. Do not restart on observation timeout or missing progress alone. Order is silence then speech, each original-init 20000 updates. Training and export use `/usr/bin/python3 research/wakeword_p4/training/gpu_python.py`; constrain OpenBLAS1/OMP2/TF intra2/inter1.

After both child exits are zero, verify each training report's plan hash, final-weight hash, 20000 optimizer iterations, and all training checks. Commit and push both original reports with exactly named paths; retain logs/failures. Do not treat losses as recognition or select between arms before fixed-profile evaluation.

Final-only export entry point:

```
/usr/bin/python3 research/wakeword_p4/experiments/export_chatterbox_context_pair_v1.py plan silence
/usr/bin/python3 research/wakeword_p4/experiments/export_chatterbox_context_pair_v1.py plan speech
```

Commit/push both generated `4090-context-{silence,speech}-int8-plan-v1.json` before either run. Then execute `run silence` and `run speech` sequentially using the GPU runtime wrapper. Only final weights and the unchanged old calibration representative batch are admitted. No retry/alternate checkpoint/threshold rescue. If a stage fails, retain original evidence and diagnose before any further action. Export reports must pass static/runtime ABI and hash checks; this is not recognition qualification.

After both final exports pass, create and push locks for `replay_context_pair_v1.py plan builtin`, `plan daily`, `plan continuous`, and `replay_context_pair_melo_v1.py plan`. Execute each corresponding `run` sequentially. These are single-thread CPU INT8 deployment-byte diagnostics; no training takes place in these replays. The first three reuse 36 builtin utterances, 96 daily-language utterances, and both existing continuous-session gap schedules. Melo replays all474 consumed rows, including432synthetic and42complete reused background inputs. Do not label retained original role strings as fresh data, or add reused background duration as new exposure.

Then run `replay_context_pair_melo_v1.py score`: it authenticates saved C161 ledgers and raw traces, preserves all events and compares each arm at the original threshold. Report British (EN-BR) and US bare/command groups separately and all paired gains/losses. Also summarize both arms' builtin/daily/continuous results with all prime/gap/tail events. Do not reduce success to continuous-session gains while ignoring British bare coverage or common-language conflicts. Continue Legacy/Jenny/Joe/Mac consumed comparisons as needed; no protected acceptance access.

Archive original raw UINT8 traces with a member size/hash manifest, verify extracted bytes and independently recompute events before pushing results/Issue18. C161 remains baseline pending broader evidence and Mac/P4 feedback. First-command PCM retention, device resources, and Hi ESP/Hey Willow reference comparison remain separate incomplete requirements; host container counts do not prove them.
