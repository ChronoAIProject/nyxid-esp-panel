#pragma once

#pragma once

#include "esp_err.h"

#include <stddef.h>
#include <stdint.h>

void qr_scanner_start(void);
void qr_scanner_poll(void);
esp_err_t qr_scanner_process_jpeg(const uint8_t *jpeg, size_t jpeg_len);
