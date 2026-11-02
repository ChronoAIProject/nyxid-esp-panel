#include "mailbox.h"

#include <string.h>

void mailbox_init(mailbox_t *mb, void *slots, size_t slot_size, size_t capacity)
{
    if (!mb) {
        return;
    }
    mb->slots = slots;
    mb->slot_size = slot_size;
    mb->capacity = capacity;
#ifdef ESP_PLATFORM
    mb->queue = NULL;
    if (slots && slot_size > 0 && capacity > 0) {
        mb->queue = xQueueCreateStatic((UBaseType_t)capacity, (UBaseType_t)slot_size,
                                       (uint8_t *)slots, &mb->queue_storage);
    }
#else
    mb->head = 0;
    mb->tail = 0;
    mb->count = 0;
#endif
}

bool mailbox_post(mailbox_t *mb, const void *item, uint32_t timeout_ms)
{
    if (!mb || !mb->slots || !item || mb->slot_size == 0 || mb->capacity == 0) {
        return false;
    }

#ifdef ESP_PLATFORM
    if (!mb->queue) {
        return false;
    }
    TickType_t ticks = timeout_ms == MAILBOX_FOREVER ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    if (xQueueSendToBack(mb->queue, item, ticks) == pdTRUE) {
        return true;
    }
    if (timeout_ms != MAILBOX_NOWAIT) {
        return false;
    }
    uint8_t dropped[mb->slot_size];
    (void)xQueueReceive(mb->queue, dropped, 0);
    return xQueueSendToBack(mb->queue, item, 0) == pdTRUE;
#else
    (void)timeout_ms;
    if (mb->count == mb->capacity) {
        mb->head = (mb->head + 1) % mb->capacity;
        mb->count--;
    }

    uint8_t *dst = (uint8_t *)mb->slots + (mb->tail * mb->slot_size);
    memcpy(dst, item, mb->slot_size);
    mb->tail = (mb->tail + 1) % mb->capacity;
    mb->count++;
    return true;
#endif
}

bool mailbox_recv(mailbox_t *mb, void *out, uint32_t timeout_ms)
{
    if (!mb || !mb->slots || !out || mb->slot_size == 0 || mb->capacity == 0) {
        return false;
    }

#ifdef ESP_PLATFORM
    if (!mb->queue) {
        return false;
    }
    TickType_t ticks = timeout_ms == MAILBOX_FOREVER ? portMAX_DELAY : pdMS_TO_TICKS(timeout_ms);
    return xQueueReceive(mb->queue, out, ticks) == pdTRUE;
#else
    (void)timeout_ms;
    if (mb->count == 0) {
        return false;
    }
    uint8_t *src = (uint8_t *)mb->slots + (mb->head * mb->slot_size);
    memcpy(out, src, mb->slot_size);
    mb->head = (mb->head + 1) % mb->capacity;
    mb->count--;
    return true;
#endif
}

size_t mailbox_count(const mailbox_t *mb)
{
#ifdef ESP_PLATFORM
    return (mb && mb->queue) ? (size_t)uxQueueMessagesWaiting(mb->queue) : 0u;
#else
    return mb ? mb->count : 0;
#endif
}

void mailbox_clear(mailbox_t *mb)
{
    if (!mb) {
        return;
    }
#ifdef ESP_PLATFORM
    if (mb->queue) {
        (void)xQueueReset(mb->queue);
    }
#else
    mb->head = 0;
    mb->tail = 0;
    mb->count = 0;
#endif
}
