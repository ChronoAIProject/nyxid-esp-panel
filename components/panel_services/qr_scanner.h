#pragma once

#include "esp_err.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    QR_SCAN_SCANNING,
    QR_SCAN_CONFIRM,
    QR_SCAN_CONNECTING,
    QR_SCAN_CONNECTED,
    QR_SCAN_ERROR,
} qr_scan_state_t;
typedef struct {
    qr_scan_state_t state;
    char ssid[33];
    char message[160];
} qr_scan_snapshot_t;

void qr_scanner_begin(void);
void qr_scanner_cancel(void);
void qr_scanner_confirm(void);
void qr_scanner_snapshot(qr_scan_snapshot_t *snapshot);

void qr_scanner_start(void);
void qr_scanner_poll(void);
esp_err_t qr_scanner_process_jpeg(const uint8_t *jpeg, size_t jpeg_len);
