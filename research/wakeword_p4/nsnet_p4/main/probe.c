#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#ifdef NYX_RNNOISE
#include "rnnoise16.h"
#else
#include "esp_nsn_iface.h"
#endif
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#ifndef NYX_RNNOISE
#include "model_path.h"

extern const uint8_t nsnet_pack[] asm("_binary_nsnet_pack_start");
extern const esp_nsn_iface_t esp_nsnet2_quantized;
#endif

static void emit(int index, const int16_t *signal, int count) {
    static const char hex[] = "0123456789abcdef";
    char line[4 * 512 + 1];
    for (int i = 0; i < count; ++i) {
        uint16_t x = (uint16_t)signal[i];
        line[4*i] = hex[(x >> 4) & 15];
        line[4*i+1] = hex[x & 15];
        line[4*i+2] = hex[x >> 12];
        line[4*i+3] = hex[(x >> 8) & 15];
    }
    line[4*count] = 0;
    printf("PCM %d %s\n", index, line);
}

static void measure(void *arg) {
    uint32_t header[3];
    uint32_t address = 2U << 20;
    ESP_ERROR_CHECK(esp_flash_read(NULL, header, address, sizeof(header)));
    if (header[0] != 0x324e534e || header[2] == 0 || header[2] > 240000) abort();
    address += sizeof(header);
#ifndef NYX_RNNOISE
    srmodel_list_t *models = srmodel_load(nsnet_pack);
    if (models == NULL || models->num != 1 || strcmp(models->model_name[0], "nsnet2")) abort();
#endif
    uint32_t internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint32_t external = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#ifdef NYX_RNNOISE
    DenoiseState *state = rnnoise_create(NULL);
    if (!state || rnnoise_get_frame_size() != 480) abort();
    int hop = 160, rate = 16000;
#else
    const esp_nsn_iface_t *api = &esp_nsnet2_quantized;
    // Enable the task FPU before the QEMU ROM parses the model metadata.
    __asm__ volatile("fmv.w.x ft0, zero" ::: "ft0");
    esp_nsn_data_t *state = api->create(models->model_name[0]);
    if (!state) abort();
    int hop = api->get_samp_chunksize(state);
    int rate = api->get_samp_rate(state);
    if (rate != 16000 || hop <= 0 || hop > 512) abort();
#endif
    printf("READY %"PRIu32" %"PRIu32" %d %d %"PRId32" %"PRId32"\n", header[1], header[2], hop, rate,
           (int32_t)(internal - heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           (int32_t)(external - heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
    int16_t input[512], output[512];
    for (uint32_t at = 0; at < header[2]; at += hop) {
        int n = header[2]-at < hop ? header[2]-at : hop;
        memset(input, 0, sizeof(input));
        ESP_ERROR_CHECK(esp_flash_read(NULL, input, address+2*at, 2*n));
        int64_t start = esp_timer_get_time();
#ifdef NYX_RNNOISE
        // Keep QEMU's lazy-FPU exception path out of ROM CSR instructions
        // across an RTOS tick while processing this isolated test frame.
        UBaseType_t mask = portSET_INTERRUPT_MASK_FROM_ISR();
        __asm__ volatile("fmv.w.x ft0, zero" ::: "ft0");
        rnnoise16_process(state, input, output);
        portCLEAR_INTERRUPT_MASK_FROM_ISR(mask);
        int status = 0;
#else
        int status = api->process(state, input, output);
#endif
        int64_t elapsed = esp_timer_get_time() - start;
        printf("FRAME %"PRIu32" %d %"PRId64"\n", at, status, elapsed);
        emit(at, output, hop);
        if ((at/hop)%64 == 0) vTaskDelay(1);
    }
    printf("ACTIVE %zu %zu %u\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)uxTaskGetStackHighWaterMark(NULL));
#ifdef NYX_RNNOISE
    rnnoise_destroy(state);
#else
    api->destroy(state);
#endif
    printf("FINISH %"PRIu32" %zu %zu\n", header[1],
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT), heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    puts("NSNET PASS");
    vTaskDelete(NULL);
}

void app_main(void) {
#if !defined(NYX_RNNOISE) && !defined(NYX_NSNET_BOARD)
    // QEMU maps this 16 MiB window as RAM; the physical board uses IDF PSRAM initialization.
    const uint32_t caps[] = {MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_32BIT, 0, 0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps, 0x48000000, 0x49000000));
#endif
    if (xTaskCreate(measure, "nsnet", 65536, NULL, 3, NULL) != pdPASS) abort();
}
