// Research-only observers: no PCM read, DMA configuration change or queue owner.
#include <atomic>
#include <cstdint>
#include "driver/i2s_common.h"
#include "esp_attr.h"

namespace {
static_assert(std::atomic<uint32_t>::is_always_lock_free);
DRAM_ATTR std::atomic<uint32_t> received{0}, overflows{0};
bool IRAM_ATTR on_received(i2s_chan_handle_t, i2s_event_data_t *event, void *) {
    received.fetch_add(event->size, std::memory_order_relaxed);
    return false;
}
bool IRAM_ATTR on_overflow(i2s_chan_handle_t, i2s_event_data_t *, void *) {
    overflows.fetch_add(1, std::memory_order_relaxed);
    return false;
}
}

extern "C" esp_err_t __real_i2s_new_channel(const i2s_chan_config_t *,
                                           i2s_chan_handle_t *, i2s_chan_handle_t *);
extern "C" esp_err_t __wrap_i2s_new_channel(const i2s_chan_config_t *config,
                                           i2s_chan_handle_t *tx, i2s_chan_handle_t *rx) {
    esp_err_t result = __real_i2s_new_channel(config, tx, rx);
    if (result != ESP_OK || !rx || !*rx) return result;
    i2s_event_callbacks_t callbacks{};
    callbacks.on_recv = on_received;
    callbacks.on_recv_q_ovf = on_overflow;
    result = i2s_channel_register_event_callback(*rx, &callbacks, nullptr);
    if (result != ESP_OK) {
        i2s_del_channel(*rx); *rx = nullptr;
        if (tx && *tx) { i2s_del_channel(*tx); *tx = nullptr; }
    }
    return result;
}

extern "C" void nyx_wake_capture_counters(uint32_t *bytes, uint32_t *lost) {
    *bytes = received.load(std::memory_order_relaxed);
    *lost = overflows.load(std::memory_order_relaxed);
}
