// The pinned upstream TFLM registration supports per-channel INT8 dense weights.
#define Register_FULLY_CONNECTED Register_OnsetPerChannelFC
#define RegisterInference_FULLY_CONNECTED RegisterInference_OnsetPerChannelFC
#include "tensorflow/lite/micro/kernels/fully_connected.cc"
