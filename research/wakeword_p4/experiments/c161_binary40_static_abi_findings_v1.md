# Original parallel models: static ABI and integration payloads

Read both exact frozen TFLite FlatBuffers without invoking, exporting or modifying models. Original SHA/size authentication passed. Separate BUILTIN_REF1thread interpreter allocation/metadata review cross-checks65C161 non-resource tensors and79Binary40 tensors, plus every B40 cache signature/index/quantization pair. Host allocation is not P4 allocation.

| Item | Logical bytes | Meaning |
|---|---:|---|
| C161 six internal INT8 state values |6020|READ_VARIABLE value shapes, not resource allocator overhead|
| B40 seven INT8 cache inputs |6544|retained state payload|
| B40 seven INT8 cache outputs |6544|output payload before feedback; actual aliasing runtime-dependent|
| Distinct C161+B40 state values |12564|6020+6544; excludes simultaneous cache outputs and all overhead|
| Exact original model files |125752|flash/filesystem payload, not RAM/arena|

C161 state shapes: [1,4,1,30], [1,2,1,40], [1,8,1,60], [1,12,1,60], [1,20,1,60], [1,57,1,60]. Binary40 cache0..6 shapes: [1,2,1,40], [1,4,1,64], [1,8,1,64], [1,16,1,64], [1,32,1,64], [1,40,1,64], [1,1,1,64]. B40 cache byte counts80,256,512,1024,2048,2560,64 sum6544.

All seven original trained Binary40 cache input/output scales and zero-points match exactly, independently agreeing with the original export report. For these exact model bytes, cache feedback can copy bytes with no requantization; real-zero cache initialization is signed-128, never raw0. The generic adapter retains requantization support for other models. Keep cache tensor mappings from the signature, not presumed flat tensor order; JSON census records all indices. Feed back only after invoke and preserve all outputs required by the next call. Do not claim in-place aliasing safe without runtime-specific proof. Cache transfer payload is6544bytes/native30ms call, not a measured execution cost.

C161 main graph requires CALL_ONCE v1, VAR_HANDLE v1, READ_VARIABLE v1, ASSIGN_VARIABLE v1, EXPAND_DIMS v1, CONCATENATION v2, STRIDED_SLICE v2, CONV_2D v3, DEPTHWISE_CONV_2D v3, RESHAPE v1, FULLY_CONNECTED v4, LOGISTIC v2, QUANTIZE v1. It has a separate variable initialization subgraph. P4 integration must support its resource lifecycle and only full-stream reset.

Binary40 requires CONCATENATION v2, STRIDED_SLICE v2, CONV_2D v3, DEPTHWISE_CONV_2D v3, RESHAPE v1, ADD v2 and LOGISTIC v2; no resource-variable operators. Static opcode presence/version does not prove TFLM/ESP-NN kernel availability, selected implementations or model numerical equivalence. C161 probability is uint8; Binary40 is signed int8 with+128 translation; retain both original quantizations and event policy.

Mac next: report actual resolver/kernel support and model load/init errors before recognition claims; measure both arenas and resource allocations with both interpreters resident, whole-panel internal/PSRAM peaks and frontend/audio/queue/task budgets, per-call scheduling, and first-command/stability evidence. Summing listed tensor byte counts does not estimate arena because constants, liveness, aliasing, alignment and scratch differ. No device resource or speed conclusion follows. Original research handoff remains unchanged and C161 stays baseline. Nick/Melo conflicts remain disclosed.
