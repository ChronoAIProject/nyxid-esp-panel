# Binary40 training tensor cache-start isolation

Fixed first128 original training rows, final20000 weights, RTX4090. Process session18292 exited0. Exact cold probabilities match the previously saved training-fit trace. All checkpoint tensors and original batch feature/mask/descriptor identities verified. No optimizer update.

Both zero-cache and silence-feature-cache arms retain32/32 positive mask-contained fivecall crossings and0/96 negative event rows. Only event timing on rows70/122 changes. Maximum probability difference in positive supervised cells is0.0006603003, negative supervised cells0.0016505613. This provides no evidence of a large cache-start fit failure in these128 training examples; it does not exclude one elsewhere or explain crossvoice development failures.

The prefix is the original frontend on81600zeroPCM:508 frames, of which507 form complete calls. The remaining one frame is deliberately not combined with the unchanged original tensor. Both arms restart the event decoder at the tensor boundary. This isolates neural cache initialization; it is NOT continuous physical-waveform/front-end priming, does not test deployment arming carry and is not independent validation. Training/calibration overlap and all earlier fit limitations remain. Do not use this narrow result to launch a priming-training rescue or assert deployment equivalence.

Original report and complete cold/warm probabilities, original weights and prefix are retained in results/4090-binary40-cache-start-original-v1.json and 4090-binary40-cache-start-traces-v1.npz. Prelock3f326396. C161 baseline and Binary40 candidate unchanged.
