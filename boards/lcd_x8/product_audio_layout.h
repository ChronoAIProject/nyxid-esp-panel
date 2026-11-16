#pragma once

#include <stddef.h>
#include <stdint.h>

/* LCD-X8 ES7210 DMA order: MIC1, MIC3 (ES8311 echo reference), MIC2, MIC4.
 * Expose the same signed 16-bit microphone/reference frames as the legacy BSP.
 * Source and destination may be identical; each input frame yields one frame.
 */
static inline void product_lcd_x8_normalize_capture(int16_t *out, const int16_t *tdm, size_t frames)
{
    for (size_t i = 0; i < frames; ++i) {
        out[2 * i] = tdm[4 * i];
        out[2 * i + 1] = tdm[4 * i + 1];
    }
}
