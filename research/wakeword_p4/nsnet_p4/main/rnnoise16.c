#include <math.h>
#include <string.h>
#include "rnnoise16.h"

// scipy.signal.firwin(31, 1./3, window="hamming"). Each causal FIR has
// 15 samples of group delay at 48 kHz. The downsampler emits phase zero.
static const float taps[31] = {
    1.0372527798e-18f,    1.7693000184e-03f,    2.5341776172e-03f,    -2.1763142873e-18f,
    -5.8063744221e-03f,    -8.5270213218e-03f,    5.1584160293e-18f,    1.6913653039e-02f,
    2.3108742276e-02f,    -8.8444964982e-18f,    -4.2360041405e-02f,    -5.8300075400e-02f,
    1.1826598240e-17f,    1.3206305561e-01f,    2.7230021999e-01f,    3.3260872801e-01f,
    2.7230021999e-01f,    1.3206305561e-01f,    1.1826598240e-17f,    -5.8300075400e-02f,
    -4.2360041405e-02f,    -8.8444964982e-18f,    2.3108742276e-02f,    1.6913653039e-02f,
    5.1584160293e-18f,    -8.5270213218e-03f,    -5.8063744221e-03f,    -2.1763142873e-18f,
    2.5341776172e-03f,    1.7693000184e-03f,    1.0372527798e-18f,
};

static float up_history[10], down_history[30];

void rnnoise16_process(DenoiseState *state, const int16_t input[160], int16_t output[160])
{
    float source[170], high[510];
    memcpy(source, up_history, sizeof(up_history));
    memcpy(high, down_history, sizeof(down_history));
    for (int i = 0; i < 160; ++i) source[10 + i] = input[i];
    memcpy(up_history, source + 160, sizeof(up_history));
    for (int i = 0; i < 480; ++i) {
        float value = 0;
        for (int k = i % 3; k < 31; k += 3) {
            value += taps[k] * source[10 + (i - k) / 3];
        }
        high[30 + i] = 3 * value;
    }
    rnnoise_process_frame(state, high + 30, high + 30);
    for (int i = 0; i < 160; ++i) {
        float value = 0;
        for (int k = 0; k < 31; ++k) value += taps[k] * high[30 + 3 * i - k];
        output[i] = (int16_t)lrintf(fminf(32767.f, fmaxf(-32768.f, value)));
    }
    memcpy(down_history, high + 480, sizeof(down_history));
}
