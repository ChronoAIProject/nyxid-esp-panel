#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#endif

#define MAILBOX_NOWAIT  0u
#define MAILBOX_FOREVER UINT32_MAX

typedef struct {
#ifdef ESP_PLATFORM
    QueueHandle_t queue;
    StaticQueue_t queue_storage;
#endif
    void *slots;
    size_t slot_size;
    size_t capacity;
#ifndef ESP_PLATFORM
    size_t head;
    size_t tail;
    size_t count;
#endif
} mailbox_t;

void mailbox_init(mailbox_t *mb, void *slots, size_t slot_size, size_t capacity);
bool mailbox_post(mailbox_t *mb, const void *item, uint32_t timeout_ms);
bool mailbox_recv(mailbox_t *mb, void *out, uint32_t timeout_ms);
size_t mailbox_count(const mailbox_t *mb);
void mailbox_clear(mailbox_t *mb);
