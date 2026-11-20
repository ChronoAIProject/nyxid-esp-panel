// Run consecutive listening epochs on one boot with diagnostic M/R resident in PSRAM.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "fixed_marker/board_marker.h"

enum { clips = 5, cycles = 64, block_size = 256 };
static const uint32_t external_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;

static void sample_heap(const char *stage, unsigned epoch)
{
    printf("REARM heap %u %s %u %u\n", epoch, stage,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_free_size(external_caps));
}

static void listen(void *arg)
{
    unsigned sizes[clips];
    int16_t *audio[clips];
    unsigned offset = 2u << 20;
    ESP_ERROR_CHECK(esp_flash_read(NULL, sizes, offset, sizeof(sizes)));
    offset += sizeof(sizes);
    for (unsigned i = 0; i < clips; ++i) {
        if (!sizes[i] || sizes[i] % block_size || sizes[i] > ((4u << 20) - offset) / 4) abort();
        unsigned bytes = 4 * sizes[i];
        audio[i] = heap_caps_aligned_alloc(8, bytes, external_caps);
        if (!audio[i]) abort();
        ESP_ERROR_CHECK(esp_flash_read(NULL, audio[i], offset, bytes));
        offset += bytes;
    }
    vTaskDelay(2);
    puts("REARM resident");
    for (unsigned epoch = 0; epoch < cycles; ++epoch) {
        sample_heap("before", epoch);
        void *storage = heap_caps_aligned_alloc(8, board_marker_bytes(), external_caps);
        if (!storage) abort();
        board_marker_t *marker = board_marker_begin(storage);
        sample_heap("ready", epoch);
        unsigned clock = 0, hit_clip = clips;
        uint64_t cue = 0;
        // Each listener hears one whole negative then the address without a reset between clips.
        for (unsigned lane = 0; lane < 2 && !cue; ++lane) {
            unsigned clip = lane ? clips - 1 : epoch % (clips - 1);
            for (unsigned at = 0; at < sizes[clip]; at += block_size) {
                UBaseType_t saved = portSET_INTERRUPT_MASK_FROM_ISR();
                asm volatile("fmv.w.x ft0, zero" ::: "ft0");
                cue = board_marker_feed_fixed(marker, audio[clip] + 2 * at, block_size, clock);
                portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
                clock += block_size;
                if (cue) {
                    hit_clip = clip;
                    break;
                }
                if (clock % (32 * block_size) == 0) vTaskDelay(1);
            }
        }
        printf("REARM decision %u %u %" PRIu64 " %u\n", epoch, hit_clip, cue, clock);
        heap_caps_free(storage);
        sample_heap("released", epoch);
    }
    for (unsigned i = 0; i < clips; ++i) heap_caps_free(audio[i]);
    puts("BOARD PASS");
    vTaskDelete(NULL);
}

void __wrap_app_main(void)
{
    const uint32_t caps[] = {MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_32BIT, 0, 0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps, 0x48000000, 0x49000000));
    if (xTaskCreate(listen, "rearm", 48 * 1024, NULL, 3, NULL) != pdPASS) abort();
}

void app_main(void) __attribute__((alias("__wrap_app_main")));
