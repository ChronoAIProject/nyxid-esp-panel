// Compile the pinned TFLM implementation under a distinct symbol for this probe's FC kernel.
#define TFLITE_SINGLE_ROUNDING 1
#define MultiplyByQuantizedMultiplier NyxFcSingleRoundMultiplier
#include "tensorflow/lite/kernels/internal/common.cc"
