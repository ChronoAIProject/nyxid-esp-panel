# Prospective FP32 candidate route within the existing wake goal

Status: implementation-format clarification before new phase-final recognition
results. No candidate, operating point, acceptance access or deployment is
authorized by a passing standalone numeric probe. Current GPU formal96271,
its fixed finals, own135 complete witnesses per model and all372/138-point
diagnosis remain unchanged.

The active user goal requires mature-reference wake recognition, matched error
and latency evaluation, P4 resource measurements, first-utterance preservation
and long-call/reconnect stability. It does not specify INT8 as the only allowed
implementation format. The existing main research README likewise describes
quantized compatibility results and full-app requirements, not a user-imposed
precision restriction. A float path must therefore remain a possible complete
implementation if it satisfies those same requirements. This does not turn
FP32 diagnostics into acceptance or waive any requirement for an INT8 export.

Actual prior evidence: `0eabda4` executed four original trained raw models on
six original complete feature streams, twice fresh, for 8,336 P4 Invokes.
Worst Invoke was 5,206 us against the original 30,000-us limit; TF/LiteRT errors
were below 1e-4. Main arena use was 51,800 bytes plus 528 variable bytes;
reservations were still 256 KiB plus 64 KiB. This proves neither new-weight
qualification nor full-panel headroom. It is sufficient to avoid ruling out
FP32 solely because a separate INT8 conversion failed.

If a new raw fixed-final model passes its own complete numeric/phase checks
and the unchanged combined reused-diagnostic requirements:

1. Preserve every original point, failed model and failed INT8 result. Freeze
   any prospective FP32 development protocol before scoring, with the same
   matched sources/environments/FAR target and untouched acceptance separation.
   Do not select a diagnostic threshold or infer mature parity from a weak
   single-source reference. Follow the predeclared independent evaluation;
   timing support from synthesis remains distinct from phonetic latency.
2. Bind the exact candidate weights, float frontend scaling, full startup/tail,
   stream/reset state, binary64 event arithmetic and a complete model-bound
   PCM self-test. Its own host and P4 outputs must meet the original float/phase
   1e-4 numeric requirements. An old model's probe cannot qualify new weights.
3. Build an opt-in typed FP32 full-panel profile, using the standard float FC
   operator and float32 input/output paths. Keep existing INT8/NYXKWS02 guards
   intact. Do not reinterpret UINT8 scores, merely remove type checks, or load
   unrelated profiles. Actual complete app builds must fit the unchanged OTA
   layout and pass actual allocation, stack, CPU/deadline and stability checks.
   The old standalone reservations are not a claimed full-app allocation.
4. Validate the actual existing single I2S→AEC→AGC→ordered-FIFO microphone path,
   continuous original-speed input, buffered first utterance and transition to
   live audio, full GUI/camera/Wi-Fi/AEC load, long conversations/reconnects,
   and the user's matched wake-quality/latency requirements. Do not add a mic
   reader, duplicate preroll, accelerate audio or hide echo with transcript
   filtering. Use development hardware only; current restrictions on Mac audio
   I/O and the dining-room production board remain in force.

If these requirements fail, FP32 has not qualified and the existing Hi ESP
default remains. If an INT8 export is pursued, its original quantized numeric,
five-score, startup/phase and recognition gates all still apply; this route
does not relax them or describe rejected quantization as fixed. Choosing a
fully validated implementation format is separate from reducing the requested
product outcome. No raw candidate currently meets all these requirements.
