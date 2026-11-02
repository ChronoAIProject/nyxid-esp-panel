#pragma once

#include "settings.h"
#include "media_sys.h"
#include "network.h"
#include "sys_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// WebRTC
int start_webrtc(const char *session_id);
int start_webrtc_with_video_config(const char *session_id, uint8_t fps, uint32_t bitrate);
esp_err_t media_sys_prepare_publish_capture(uint8_t fps);
esp_err_t media_sys_prepare_preview_capture(void);
void media_sys_capture_close(void);
int media_sys_webrtc_capture_start(void);
int media_sys_webrtc_capture_stop(void);
int media_sys_preview_start(void);
int media_sys_preview_stop(void);
bool media_sys_preview_frame_available(void);
esp_err_t media_sys_snapshot_grab(uint32_t timeout_ms, size_t *out_len);
esp_err_t media_sys_preview_snapshot_encode(uint32_t timeout_ms, size_t *out_len);
esp_err_t media_sys_preview_copy_rgb565(uint8_t *dst, size_t cap, uint32_t last_seq,
                                        uint16_t *out_w, uint16_t *out_h, size_t *out_len,
                                        uint32_t *out_seq);
esp_err_t media_sys_preview_copy_latest_rgb565(uint8_t *dst, size_t cap, uint16_t *out_w,
                                               uint16_t *out_h, size_t *out_len,
                                               uint32_t *out_seq);
void query_webrtc(void);
int stop_webrtc(void);

// WHIP signaling
#include "esp_peer_signaling.h"
const esp_peer_signaling_impl_t *esp_signaling_get_whip_impl(void);

#ifdef __cplusplus
}
#endif
