#pragma once

#include "esp_err.h"

#include <stddef.h>
#include <stdint.h>

#define SNAPSHOT_MAX_BYTES (80u * 1024u)

void snapshot_init(void);
esp_err_t snapshot_store_jpeg(const uint8_t *data, size_t len);
esp_err_t snapshot_copy_latest(uint8_t *dst, size_t cap, size_t *out_len);
