# Matched paired-carrier experiment: not a replacement

The prelocked paired-carrier intervention improved Joe coverage but regressed US/BR target coverage and ordinary-name interference. Retain C161 and the existing Mac handoff; do not promote this candidate or sweep its threshold, quota, seed or checkpoint against these consumed cohorts.

Both original Binary40Span arms trained 20,000 steps on RTX 4090 with the same initial tensors, seed, batch size and objective. Treatment replaced four broad bare positives with Kokoro command positives and four bare negatives with matched Nick commands per batch, preserving the remaining 120 rows. This jointly changes engine/carrier/negative allocation and stochastic provenance; it does not isolate pairing causality. No new synthesis or holdout access occurred.

Independent training review reconstructed all 20,000 batches (2,560,000 descriptor rows per arm), verified control identity against the original experiment, initial/final finite tensors and terminal exit receipts. Both final-only INT8 exports used their own sampler and training-state representatives. Training review publication: 1b1ee811. Complete consumed evidence and independent review: 051eab37.

| Consumed target cohort | C161 | Control | Paired |
| --- | ---: | ---: | ---: |
| Builtin | See original baseline evidence | 8/12 | 8/12 |
| Legacy | 35/36 | 36/36 | 36/36 |
| Jenny | 33/36 | 33/36 | 33/36 |
| Joe | 0/72 | 17/72 | 31/72 |
| Melo US | 33/36 | 33/36 | 31/36 |
| Melo BR | 21/36 | 20/36 | 18/36 |
| Mac TTS | 1/4 historical matched-prime diagnostic | 2/4 | 2/4 |

Whole physical context target hits (12 each): C161/control/paired = builtin 5/8/9, US 2/11/9, BR 0/6/3. Paired US also triggers in one ordinary-name container (control zero). Container ownership is temporal attribution, not phonetic causality or proof of first-command retention.

Daily ordinary-name hits rise 1/24 to 2/24: existing “Hey Nick, what time is it?” remains and “Hey Nick, can you help me?” is added. Other daily groups remain zero. These common-name conflicts matter independently of rare constructed nearwords; this conclusion does not require zero constructed-nearword errors.

Reused background control/paired events: Legacy 0/0 over 10.483944826h; Jenny 0/0 over 9.614419583h; Melo 1/0 over 7.902862222h. These are previously consumed identical streams, never additional independent FAR exposure and never pooled across arms or historical runs. Removing one background event does not compensate for reduced normal coverage by itself.

Both exports are 64,800 bytes. Control SHA256 `454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c` reproduces original B40 bytes. Paired SHA256 `ca42573ac4797d8784aa752c9f3aff5cbadd0b1287395b114b62910c95c14a77`. Exact compressed model bytes and original export reports are preserved in results.

The independent consumed reviewer verified 16 complete cohorts, 3,574 archive members, source identities and six physical-context traces using a separate integer event decoder. All 60 subprocess stages terminated with exit 0; see the continuation receipt. Fixed event profile: threshold 235/256, smoothing 5, refractory 67, original rearm policy; continuous 81,600-sample zero PCM prime without boundary reset.

Next work should explain training-support gains versus accent/context regressions before another intervention. Neither training reconstruction nor these consumed synthetic diagnostics establishes human generalization, P4 resources, real-time safety, microphone stability or production readiness. Mac/P4 integration remains tracked in Issue #18, using the existing C161/B40/history handoff rather than substituting this candidate.

## Saved-outcome transition audit

`paired_carrier_transition_audit_v1.py` binds the original comparison and independently reviewed context files by SHA256 and records every changed outcome without inference. Joe's net +14 is 22 gained and 8 lost; gains/losses occur at all three length scales and all three rendered variants. These variants are correlated and do not establish independent generalization. Jenny's unchanged 33/36 hides two gained and two lost. Melo US+BR together have four gained and eight lost.

Whole-context builtin net +1 is two gained and one lost; US has zero gained and two lost; BR net -3 is two gained and five lost. US adds the ordinary-name container “Hey Max, can you carry the other bag upstairs?” (text index 46), also seen in the earlier multiscale experiment. The repeated sentence identity is evidence of a shared consumed failure case, not evidence that the interventions have the same cause. “Hey Nicks, is there any rain expected this afternoon?” (index 2) is lost in all three context voices. Those observations motivate a bounded float-versus-INT8 diagnostic before attributing the losses to either quantization or training; they do not justify another quota/threshold sweep.
