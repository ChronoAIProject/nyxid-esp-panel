# Remaining false wakes span quiet, bare and connected commands

Saved actual Jenny app events from95ab/5557/c161, fixedprofile only. No model calls, new audio, threshold selection, training or final access. All144 Nick/Next/Mix/Muse source×variant clips aligned exactly by sourceID/variant, carrier and speed; remaining144 ordinary common clips are silent. This is consumed development, not independent acceptance.

c16144falseclips represent19native source groups with correlated variants. Carrier counts bare7/light11/time11/timer15; condition counts quiet14/25dBnoise17/15dBnoise+fullRIR13; speeds.9:11/1.0:21/1.1:12. Thus a noise-only, RIR-only, command-only or single-speed explanation cannot account for every failure. These strata have different denominators and shared native ancestry; raw totals are not causal source/noise effects.

First-event generated-coordinate bins:7beforeprefix,30betweenprefix/container,7at-or-aftercontainer. Timing does not establish which phonemes or command words cause an event. Boundaries remain synthesis references, early events are not automatic failure, and tail-gating based on known evaluation-container ends would not be a deployable streaming solution.

Compared95ab48→c16144 hides10removed/6new/38retained falseclips. Compared5557 66→44 gives23removed/1new/43retained. The6new-versus95ab cases are preserved individually; a net summary alone overstates stability of improvement. Existing source-balanced lexical supervision helps some sources, not uniformly.

Targets: c161bare9/9,light9/9,time6/9,timer9/9;95abidentical. Allthree c161 misses are timecarrier (.9variant2;1.0variant1/2). Positive-only5557time8/9 lost only1.0variant1. This localizes the observed sensitivity regression to one generated command carrier, not a general decline of all bare words or all commands. It does not establish a fix, justify selecting these clips into training, or prove human pronunciation quality.

Next implementation must address lexical discrimination while preserving this connected-command sensitivity; do not launch another ratio sweep, noise-only rescue, or evaluation-informed endpoint gate. Prior bare-only, command-end, aligned-prefix, multivoice negative, source-balance, ranking and temporal-loss results remain constraints. These observations narrow exclusive explanations; they do not yet support a distinct training intervention. Frozenv30 retained, no new fitting/job launched.

Full counts, inputSHA and paired changes: `results/4090-causal-nearword-event-context-v1.json`; executable saved-event reduction: `analyze_causal_nearword_event_context_v1.py`.
