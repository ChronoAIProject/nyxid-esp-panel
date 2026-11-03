#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char device_code[128];
    char user_code[16];
    char verification_uri[192];
    char verification_uri_complete[384];
    uint32_t expires_in;
    uint32_t interval;
} device_login_request_t;

typedef enum {
    LOGIN_POLL_APPROVED,
    LOGIN_POLL_PENDING,
    LOGIN_POLL_SLOW_DOWN,
    LOGIN_POLL_RATE_LIMITED,
    LOGIN_POLL_DENIED,
    LOGIN_POLL_EXPIRED,
    LOGIN_POLL_DELIVERED,
    LOGIN_POLL_ERROR,
} device_login_poll_t;

bool device_login_parse_request(const char *json, device_login_request_t *out);
device_login_poll_t device_login_parse_poll(int status, const char *json);
uint32_t device_login_backoff(uint32_t interval);
