#include <stdint.h>
#include <stdio.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "speex/speex_echo.h"

#ifdef NYX_SPEEX_FIXED_CUE
int cue_step(int16_t pcm);
#endif

enum { offset = 2 << 20, frame = 160 };
static const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

static void replay(void *arg)
{
    (void)arg;
    uint32_t hdr[2];
    ESP_ERROR_CHECK(esp_flash_read(NULL, hdr, offset, sizeof(hdr)));
    if (hdr[0] != 0x35434541 || hdr[1] != 160000) abort();
    uint32_t start = offset + sizeof(hdr);
    unsigned before = heap_caps_get_free_size(caps);
    SpeexEchoState *state = speex_echo_state_init(frame, 2048);
    int rate = 16000;
    if (!state || speex_echo_ctl(state, SPEEX_ECHO_SET_SAMPLING_RATE, &rate)) abort();
    unsigned ready = heap_caps_get_free_size(caps);
    int16_t mic[frame], ref[frame], pcm[frame];
    char hex[frame * 4 + 1];
    static const char digits[] = "0123456789abcdef";
#ifdef NYX_SPEEX_FIXED_CUE
    for (int i = 0; i < 81600; ++i) cue_step(0);
#endif
    for (unsigned at = 0; at < hdr[1]; at += frame) {
        ESP_ERROR_CHECK(esp_flash_read(NULL, mic, start + 2*at, sizeof(mic)));
        ESP_ERROR_CHECK(esp_flash_read(NULL, ref, start + 2*(hdr[1]+at), sizeof(ref)));
        int64_t begin = esp_timer_get_time();
        speex_echo_cancellation(state, mic, ref, pcm);
#ifdef NYX_SPEEX_FIXED_CUE
        unsigned hits[frame], count = 0;
        for (unsigned i = 0; i < frame; ++i)
            if (cue_step(pcm[i])) hits[count++] = at+i+1;
#endif
        unsigned cost = esp_timer_get_time() - begin;
#ifdef NYX_SPEEX_FIXED_CUE
        for (unsigned i = 0; i < count; ++i) printf("SPEEX_CUE %u\n", hits[i]);
#endif
        const uint8_t *bytes = (const uint8_t *)pcm;
        for (size_t i = 0; i < sizeof(pcm); ++i) {
            hex[2*i] = digits[bytes[i] >> 4];
            hex[2*i+1] = digits[bytes[i] & 15];
        }
        hex[sizeof(hex)-1] = 0;
        printf("SPEEX_PCM %u %u %s\n", at, cost, hex);
        if ((at/frame) % 32 == 0) vTaskDelay(1);
    }
    speex_echo_state_destroy(state);
    printf("SPEEX_HEAP before=%u ready=%u after=%u stack=%u\n", before, ready,
           (unsigned)heap_caps_get_free_size(caps), (unsigned)uxTaskGetStackHighWaterMark(NULL));
    puts("SPEEX PASS");
    vTaskDelete(NULL);
}

void app_main(void)
{
    if (xTaskCreate(replay, "speex", 32*1024, NULL, 3, NULL) != pdPASS) abort();
}
