// Scalar int8 FC arithmetic matching LiteRT 2.3 BUILTIN_REF's scale overload.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nyx_complete_fc {
inline void scalar_fc(const int8_t *input, const int8_t *weights,
                      const int32_t *bias, int8_t *output, int batches,
                      int depth, int input_zero, int output_zero,
                      double effective_scale, int activation_min,
                      int activation_max) {
    for(int batch=0;batch<batches;++batch) {
        int64_t accumulator=bias?bias[0]:0;
        for(int at=0;at<depth;++at)
            accumulator+=int64_t(weights[at])*
                (int(input[batch*depth+at])-input_zero);
        double value=std::round(double(accumulator)*effective_scale)+output_zero;
        value=std::clamp(value,double(activation_min),double(activation_max));
        output[batch]=static_cast<int8_t>(value);
    }
}
} // namespace nyx_complete_fc
