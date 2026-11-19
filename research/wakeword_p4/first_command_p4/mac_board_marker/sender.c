// P4 sender replay. The board mic_task owns all pop, encode, PTS and send decisions.
// The fixture supplies archived processed samples, a logical read clock and a local peer sink.
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "esp_opus_enc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "panel_voice_audio.h"
#include "sender_constants.h"

typedef struct {
    panel_voice_pcm_t capture;
    atomic_bool running, connected, provider_ready, controls, muted;
    bool wake_initiated, capture_started;
    unsigned replayed_samples;
    SemaphoreHandle_t peer_lock;
    EventGroupHandle_t done;
    void *peer;
} voice_call_t;

typedef struct { uint8_t *data; int size; unsigned pts; } esp_peer_audio_frame_t;
static voice_call_t call;
static struct { unsigned sent_packets, buffered_ms, replayed_ms; } status;
static SemaphoreHandle_t wake_lock, state_lock;
static atomic_bool microphone_muted;
static unsigned marker_epoch, marker_pcm_end;
static unsigned opened, first, samples, padded, elapsed_ms, connect_ms, ready_ms, encoded, peak;
static int16_t *pcm;

static void advance(unsigned ms)
{
    elapsed_ms += ms;
    if (elapsed_ms > 20000) abort();
    call.connected = elapsed_ms >= connect_ms;
    call.provider_ready = elapsed_ms >= ready_ms;
    const unsigned read_end = (opened + 16 * elapsed_ms) / 256 * 256;
    const unsigned produced = read_end / 160 * 160;
    const unsigned end = produced < first + padded ? produced : first + padded;
    if (!panel_voice_pcm_gate(&call.capture, &call.capture_started, call.provider_ready && call.connected, false)) abort();
    if (end > marker_pcm_end) {
        if (!panel_voice_pcm_push(&call.capture, pcm + (marker_pcm_end - first), end - marker_pcm_end)) abort();
        marker_pcm_end = end;
    }
    if (call.capture.count > peak) peak = call.capture.count;
    if (call.replayed_samples == padded) call.running = false;
}

static void packet(const char *kind, unsigned pts, const uint8_t *data, size_t size)
{
    static const char hex[] = "0123456789abcdef";
    char line[2*1275+1];
    if (!size || size > 1275) abort();
    for (size_t i=0; i<size; ++i) {
        line[2*i] = hex[data[i] >> 4];
        line[2*i+1] = hex[data[i] & 15];
    }
    line[2*size] = 0;
    printf("%s %u %s\n", kind, pts, line);
}

static esp_audio_err_t checked_encode(void *enc, esp_audio_enc_in_frame_t *in, esp_audio_enc_out_frame_t *out)
{
    static const int16_t zeros[FRAME_SAMPLES];
    if (in->len != sizeof(zeros)) abort();
    const int16_t *expected = zeros;
    if (call.provider_ready && call.connected) {
        if (call.replayed_samples + FRAME_SAMPLES > padded) abort();
        expected = pcm + call.replayed_samples;
    }
    if (memcmp(in->buffer, expected, sizeof(zeros))) abort();
    esp_audio_err_t rc = esp_opus_enc_process(enc, in, out);
    if (rc == ESP_AUDIO_ERR_OK && !call.connected)
        packet("UNSENT", encoded * 20, out->buffer, out->encoded_bytes);
    if (++encoded % 16 == 0) vTaskDelay(1); // Yield host UART time outside the logical scheduler.
    return rc;
}

static int esp_peer_send_audio(void *peer, esp_peer_audio_frame_t *frame)
{
    if (frame->pts != (encoded-1)*20) abort();
    packet("SENT", frame->pts, frame->data, frame->size);
    return 0;
}

static void state(const char *level, const char *message)
{
    printf("ERROR %s %s\n", level, message);
    abort();
}

// Single-threaded deterministic logical time. No peer/network work or codec cost advances it.
#undef ESP_LOGI
#define ESP_LOGI(tag, fmt, ...) printf(fmt "\n", __VA_ARGS__)
#define esp_timer_get_time() ((int64_t)elapsed_ms * 1000)
#define esp_opus_enc_process checked_encode
#undef taskYIELD
#define taskYIELD() advance(0)
#define vTaskDelay(ticks) advance((ticks) * portTICK_PERIOD_MS)
#define vTaskDelete(task) ((void)0)
#include "mic_wake.inc"
#undef vTaskDelete
#undef vTaskDelay

static void replay(void *unused)
{
    uint32_t header[4];
    const unsigned offset = 2u << 20;
    ESP_ERROR_CHECK(esp_flash_read(NULL, header, offset, sizeof(header)));
    unsigned index = header[0];
    opened = header[1];
    first = header[2];
    samples = header[3];
    if (opened % 256 || first + WAKE_LOOKBACK_SAMPLES != opened/160*160 ||
        samples < WAKE_LOOKBACK_SAMPLES || samples > 160000) abort();
    padded = (samples + FRAME_SAMPLES-1) / FRAME_SAMPLES * FRAME_SAMPLES;
    const unsigned caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    pcm = heap_caps_calloc(padded, sizeof(*pcm), caps);
    call.capture.samples = heap_caps_malloc(WAKE_BUFFER_SAMPLES * sizeof(*pcm), caps);
    if (!pcm || !call.capture.samples) abort();
    ESP_ERROR_CHECK(esp_flash_read(NULL, pcm, offset+sizeof(header), samples*sizeof(*pcm)));
    call.capture.capacity = WAKE_BUFFER_SAMPLES;
    call.wake_initiated = call.controls = true;
    wake_lock = xSemaphoreCreateMutex();
    state_lock = xSemaphoreCreateMutex();
    call.peer_lock = xSemaphoreCreateMutex();
    call.done = xEventGroupCreate();
    if (!wake_lock || !state_lock || !call.peer_lock || !call.done) abort();
    for (unsigned lane=0; lane<2; ++lane) {
        connect_ms = lane ? 500 : 0;
        ready_ms = lane ? 1500 : 0;
        marker_epoch = lane+1;
        encoded = elapsed_ms = status.sent_packets = call.replayed_samples = peak = 0;
        marker_pcm_end = first;
        call.running = true;
        call.capture_started = false;
        advance(0);
        printf("BEGIN %u %u %u %u %u\n", index, lane, opened, first, samples);
        mic_task(&call);
        if (call.capture.count || call.replayed_samples != padded || !(xEventGroupClearBits(call.done,DONE_MIC)&DONE_MIC)) abort();
        printf("END %u %u %u %u %u\n", lane, encoded, status.sent_packets, call.replayed_samples, peak);
    }
    puts("SENDER PASS");
    vTaskDelete(NULL);
}

void __wrap_app_main(void)
{
    const uint32_t caps[] = {MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT|MALLOC_CAP_32BIT,0,0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps,0x48000000,0x49000000));
    if (xTaskCreate(replay, "sender", 48*1024, NULL, 3, NULL) != pdPASS) abort();
}
