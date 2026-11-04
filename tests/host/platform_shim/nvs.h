#pragma once
#include "esp_err.h"
#include <stddef.h>
#include <stdint.h>
typedef unsigned nvs_handle_t;
#define NVS_READONLY 0
#define NVS_READWRITE 1
esp_err_t nvs_open(const char *, int, nvs_handle_t *);
void nvs_close(nvs_handle_t);
esp_err_t nvs_commit(nvs_handle_t);
esp_err_t nvs_erase_all(nvs_handle_t);
esp_err_t nvs_set_blob(nvs_handle_t, const char *, const void *, size_t);
esp_err_t nvs_get_blob(nvs_handle_t, const char *, void *, size_t *);
esp_err_t nvs_set_str(nvs_handle_t, const char *, const char *);
esp_err_t nvs_get_str(nvs_handle_t, const char *, char *, size_t *);
#define NVS_NUMBER(bits) \
esp_err_t nvs_set_u##bits(nvs_handle_t, const char *, uint##bits##_t); \
esp_err_t nvs_get_u##bits(nvs_handle_t, const char *, uint##bits##_t *);
NVS_NUMBER(8)
NVS_NUMBER(16)
NVS_NUMBER(32)
