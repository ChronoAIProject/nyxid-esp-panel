# Prospective bounded-head converter feasibility

Natural training waveforms already reproduce extreme logits. Sentinel deletion
is not a justified remedy. Before any new trained-model export, test a synthetic
scalar graph only: sigmoid((8/3)*ReLU6((3/8)*z+3)-8), exactly
sigmoid(clip(z,-8,8)) in real arithmetic. Absolute float probability perturbation
is bounded by sigmoid(-8) (<0.000336). A fused FC+RELU6 could constrain the
FC output quantization before subsequent affine reconstruction; an unfused
post-FC clip would not establish this property.

One fixed toy conversion, no sweep or candidate export. Representative scalar
values span [-3000,400] plus dense [-8,8]. Independently check float algebra
at 1e-6 and inspect FlatBuffer FC fused activation and every tensor scale.
Record negative results without changing tolerances. No audio, trained weights,
new development, acceptance or optimizer access. This only decides whether this
converter can express the intended topology; it does not admit any trained model.
Original models and unchanged 270-case numerical gates remain authoritative.
