# Correction to old recognition float-path terminology

This additive correction supersedes the shorthand “old float” in the same-recipe inventory and prior Issue discussion, without changing original evidence. Mac identified the issue in comment 6874357727. GPU checked the actual collector and helper SHA against the original regression plan: both exact.

The collector passes `bounded(model, tf)` to streaming conversion. Its Dense RELU6 uses W*3/8 and b*3/8+3, then Rescaling(8/3,-8), then sigmoid. In ideal arithmetic this is sigmoid(clip(raw logit,-8,8)); actual affine arithmetic remains part of the measured implementation. Therefore old Willow float-control any-event 7–8/36 describes **bounded-float**, not the original raw Dense sigmoid FP32 route. The condition name describes training, not inference output semantics.

No old bounded arrays or event decisions establish the prospective raw FP32 route's recognition quality. A small ideal probability clipping bound cannot transfer threshold, rearming, or stateful event decisions without verification. No raw FP32 run was performed here, and raw-model qualification remains unproven. Original integer rejections, original metrics, all source locks and frozen v30 remain unchanged.
