#pragma once
#include "esp_err.h"
#include "camera_profile.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
esp_err_t media_sys_prepare_preview_capture(void);
esp_err_t media_sys_capture_close(void);
int media_sys_preview_start(void);
bool media_sys_preview_frame_available(void);
esp_err_t media_sys_preview_snapshot_encode(uint8_t *dst, size_t cap, size_t *out_len);
esp_err_t media_sys_preview_copy_rgb565(uint8_t *dst, size_t cap, uint32_t last_seq,
    uint16_t *out_w, uint16_t *out_h, size_t *out_len, uint32_t *out_seq);
// Aspect-preserving reduced frame for QR decoding, never changes capture format.
esp_err_t media_sys_preview_copy_qr_rgb565(uint8_t *dst, size_t cap,
    uint16_t *out_w, uint16_t *out_h, size_t *out_len, uint32_t *out_seq);
#ifdef __cplusplus
}
#endif
