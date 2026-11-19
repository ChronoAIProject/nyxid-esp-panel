#pragma once

#include <stdint.h>
#include "rnnoise.h"

// One cold boot per stream. FIR histories are zero-initialized in BSS.
void rnnoise16_process(DenoiseState *state, const int16_t input[160], int16_t output[160]);
