# Prospective conditioning-disjoint development replay

A new contextual candidate was trained for5000 steps on3456 records from16 conditioning names. The four reserved development conditioning names (`af_alloy`, `af_kore`, `am_echo`, `am_santa`) were excluded from every optimization batch. Full raw-float exact matcher replay on the864 development records gives47/144 target hits and29/720 negative hits. This is a poor cross-conditioning result; no threshold was locked and no INT8 export was started.

The development split was opened only after training finished. It remains separate from the candidate's optimization sources, but it shares the same synthetic TTS/rendering family and is not protected final acceptance. The low result confirms that the earlier all-conditioning training counts overstated transfer performance. Frozen v30 remains unchanged.
