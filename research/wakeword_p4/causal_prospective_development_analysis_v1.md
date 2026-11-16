# Prospective development failure decomposition

The conditioning-disjoint contextual candidate reaches only47/144 target recordings (32.6%) and29/720 negatives (4.0%) on the reserved development partition. Target hits are36 records per carrier before any carrier-specific filtering; negative events are bare4/180, light9/180, time7/180, timer9/180. The failure is therefore spread across all four carriers, with no carrier meeting a viable transfer gate.

This confirms a broad conditioning-transfer failure rather than a single carrier bug. The candidate is rejected before INT8 export or threshold selection. Training-set performance cannot be used to rescue it, and the development partition will not be reused as training. A future architecture must address voice/conditioning invariance at the representation or data-design level with a newly preregistered split. Frozen v30 remains unchanged.
