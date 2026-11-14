// Native execution witness for the actual production arithmetic header.
// This wrapper supplies no TFLM preparation or interpreter implementation.
#include "complete_fc_math.h"

extern "C" void complete_fc_math_host_v1(
    const int8_t *input, const int8_t *weights, const int32_t *bias,
    int8_t *output, int batches, int depth, int input_zero, int output_zero,
    double effective_scale, int activation_min, int activation_max) {
    nyx_complete_fc::scalar_fc(input, weights, bias, output, batches, depth,
        input_zero, output_zero, effective_scale, activation_min, activation_max);
}
