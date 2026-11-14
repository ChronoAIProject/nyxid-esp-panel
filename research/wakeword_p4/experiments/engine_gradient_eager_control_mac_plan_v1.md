# Prospective eager execution control after real graph mismatch

The original `bdca179` Mac graph replay actually terminated exit 1 after all
100 original cases. Same-instance initial arrays, CPU repeats, reversal/None
mechanics and unchanged variables pass, but all 100 cross-platform comparisons
fail. Worst domain logit absolute difference is 0.003215312957763672; worst
supervised gradient absolute difference is 0.39274515211582184. Saved analytic
domain bias errors are <=5.282386034144437e-8. All original results are retained;
none is reclassified as a pass.

The GPU source computes eager tapes while the first Mac helper uses tf.function.
Before attributing the difference to GPU precision, control this execution-mode
confound. The only intervention is selecting eager execution in the same helper.
Keep the exact original H5/NPZ/materials, CPU/runtime/threads, all 100 cases in
order, two fresh repetitions, zero updates and every original tolerance. Save in
a new directory. Continue all numeric failures; structural failures retain their
original evidence. Do not modify GPU jobs or infer a unique TF32/Grappler/BN
cause from this control alone.

This is a numerical diagnostic, not recognition, model selection, a new training
condition, or a relaxation of any existing qualification gate. The graph run
remains a failed run even if this separate control passes.
