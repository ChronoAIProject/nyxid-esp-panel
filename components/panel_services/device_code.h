#pragma once

#include "sdkconfig.h"
#include "esp_err.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char device_code[96];
    char user_code[16];
    char verification_uri[192];
    uint32_t expires_in;
    uint32_t poll_interval;
} device_code_request_result_t;

typedef enum {
    DEVICE_CODE_POLL_PENDING = 0,
    DEVICE_CODE_POLL_APPROVED,
    DEVICE_CODE_POLL_EXPIRED,
} device_code_poll_status_t;

typedef struct {
    device_code_poll_status_t status;
    char api_key[128];
    char node_id[96];
    char refresh_token[192];
} device_code_poll_result_t;

esp_err_t device_code_start_if_needed(void);
void device_code_cancel(void);
bool device_code_in_progress(void);
void device_code_get_user_code(char out[16]);
void device_code_get_verification_uri(char *out, size_t cap);
uint32_t device_code_seconds_until_rotate(void);
uint32_t device_code_last_poll_age_seconds(void);
uint32_t device_code_poll_attempts(void);
esp_err_t device_code_sign(const uint8_t *msg, size_t len, uint8_t out[64]);

#ifdef CONFIG_HA_CAM_DEVICE_CODE
esp_err_t device_code_request(void);
esp_err_t device_code_poll(void);
esp_err_t device_code_get_credentials(char *api_key_out, char *node_id_out, size_t cap);
esp_err_t device_code_get_public_key_hex(char out[65]);
esp_err_t device_code_client_request(const char *device_pubkey_hex,
                                     const char *hw_id,
                                     const char *suggested_label,
                                     device_code_request_result_t *out);
esp_err_t device_code_client_poll(const char *device_code,
                                  int64_t ts,
                                  const char *ts_sig_hex,
                                  device_code_poll_result_t *out);
#endif

#ifdef __cplusplus
}
#endif
