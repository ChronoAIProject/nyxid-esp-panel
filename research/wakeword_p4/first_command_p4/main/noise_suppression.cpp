// QEMU research fixture. Four little-endian uint32 fields per record followed by PCM16.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_ns.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {
constexpr int frame = 160, rate = 16000;
constexpr uint32_t flash_end = 4U << 20;
uint32_t cursor = 2U << 20;

void read_flash(void* dest, size_t bytes) {
    if (bytes > flash_end - cursor) abort();
    ESP_ERROR_CHECK(esp_flash_read(nullptr, dest, cursor, bytes));
    cursor += bytes;
}

void emit(int at, const int16_t* pcm) {
    constexpr char hex[] = "0123456789abcdef";
    char line[4*frame + 1];
    for (int j = 0; j < frame; ++j) {
        const uint16_t v = static_cast<uint16_t>(pcm[j]);
        line[4*j] = hex[(v >> 4) & 15];
        line[4*j+1] = hex[v & 15];
        line[4*j+2] = hex[v >> 12];
        line[4*j+3] = hex[(v >> 8) & 15];
    }
    line[4*frame] = 0;
    std::printf("PCM %d %s\n", at, line);
}

void worker(void*) {
    uint32_t count;
    read_flash(&count, sizeof(count));
    if (count == 0 || count > 8) abort();
    for (uint32_t row = 0; row < count; ++row) {
        uint32_t spec[4];
        read_flash(spec, sizeof(spec));
        const int id = spec[0], length = spec[1], begin = spec[2], mode = spec[3];
        if (length <= 0 || length > 240000 || begin < 0 || begin > length || mode != 1) abort();
        const auto free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        ns_handle_t handle = ns_pro_create(10, mode, rate);
        if (!handle) abort();
        const auto free_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        std::printf("BEGIN %d %d %d %d %u\n", id, length, begin, mode, unsigned(free_before - free_after));
        for (int at = 0; at < length + 1600; at += frame) {
            if (at % (128*frame) == 0) vTaskDelay(1);
            std::array<int16_t, frame> in{}, out{};
            if (at < length) read_flash(in.data(), std::min(frame, length-at) * sizeof(int16_t));
            ns_process(handle, in.data(), out.data());
            if (at + frame > begin) emit(at, out.data());
        }
        ns_destroy(handle);
        std::printf("END %d\n", id);
    }
    std::puts("NS PASS");
    vTaskDelete(nullptr);
}
}
extern "C" void app_main() {
    if (xTaskCreate(worker, "noise_suppression", 32*1024, nullptr, 3, nullptr) != pdPASS) abort();
}
