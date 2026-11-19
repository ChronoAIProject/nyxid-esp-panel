#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_opus_enc.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
extern "C" {
#include "panel_voice_audio.h"
}

// QEMU research fixture. The flash source is paced at the nominal I2S block
// period. A separate task uses the production mic_task's frame and burst sizes.
class CaptureOpus {
    static constexpr int frame = 320;
    struct Packet {
        uint16_t length;
        uint8_t data[1275];  // One Opus frame, RFC 6716 section 3.2.1.
    };
    panel_voice_pcm_t& fifo;
    SemaphoreHandle_t lock = xSemaphoreCreateMutex(), finished = xSemaphoreCreateBinary();
    TaskHandle_t producer = xTaskGetCurrentTaskHandle();
    esp_timer_handle_t timer{};
    Packet* packets{};
    int capacity, count = 0, peak = 0, pending = 0, backlog_ticks = 0;
    int64_t opened = 0, caught_up = -1;
    int64_t total_encode_us = 0;
    int peak_encode_us = 0, stack_free = 0;
    size_t heap_before = 0, heap_ready = 0;
    bool ending = false;
    static constexpr uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

    void encode() {
        esp_opus_enc_config_t cfg = ESP_OPUS_ENC_CONFIG_DEFAULT();
        cfg.sample_rate = 16000;
        cfg.channel = 1;
        cfg.bitrate = 24000;
        cfg.complexity = 0;
        cfg.enable_vbr = true;
        void* encoder = nullptr;
        if (esp_opus_enc_open(&cfg, sizeof(cfg), &encoder) != ESP_AUDIO_ERR_OK) abort();
        heap_ready = heap_caps_get_free_size(caps);
        int16_t pcm[frame];
        uint8_t bytes[2048];
        bool done = false;
        while (!done) {
            size_t remaining = 0;
            for (int burst = 0; burst < 4; ++burst) {
                xSemaphoreTake(lock, portMAX_DELAY);
                const size_t n = ending ? std::min<size_t>(frame, fifo.count) : frame;
                const bool have = n && panel_voice_pcm_pop(&fifo, pcm, n);
                remaining = fifo.count;
                done = ending && !remaining;
                xSemaphoreGive(lock);
                if (!have) break;
                if (n < frame) std::fill(pcm+n, pcm+frame, 0);
                esp_audio_enc_in_frame_t in{};
                esp_audio_enc_out_frame_t out{};
                in.buffer = reinterpret_cast<uint8_t*>(pcm);
                in.len = sizeof(pcm);
                out.buffer = bytes;
                out.len = sizeof(bytes);
                int64_t begin = esp_timer_get_time();
                if (esp_opus_enc_process(encoder, &in, &out) != ESP_AUDIO_ERR_OK) abort();
                int cost = esp_timer_get_time()-begin;
                total_encode_us += cost;
                peak_encode_us = std::max(peak_encode_us, cost);
                if (count >= capacity || !out.encoded_bytes || out.encoded_bytes > sizeof(Packet::data)) abort();
                packets[count].length = out.encoded_bytes;
                std::memcpy(packets[count++].data, bytes, out.encoded_bytes);
                xSemaphoreTake(lock, portMAX_DELAY);
                if (caught_up < 0 && fifo.count < frame) caught_up = esp_timer_get_time()-opened;
                xSemaphoreGive(lock);
                if (done) break;
            }
            if (!done) {
                if (remaining >= frame) taskYIELD();
                else vTaskDelay(pdMS_TO_TICKS(10));
            }
        }
        stack_free = uxTaskGetStackHighWaterMark(nullptr);
        esp_opus_enc_close(encoder);
        xSemaphoreGive(finished);
        vTaskDelete(nullptr);
    }

public:
    CaptureOpus(panel_voice_pcm_t& call, int samples, int period_us) : fifo(call), capacity((samples+319)/320) {
        packets = static_cast<Packet*>(heap_caps_calloc(capacity, sizeof(Packet), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (!packets || !lock || !finished) abort();
        esp_timer_create_args_t args{};
        args.callback = [](void* task){ xTaskNotifyGive(static_cast<TaskHandle_t>(task)); };
        args.arg = producer;
        args.name = "sample_clock";
        ESP_ERROR_CHECK(esp_timer_create(&args, &timer));
        ESP_ERROR_CHECK(esp_timer_start_periodic(timer, period_us));
    }

    void next_block() {
        const uint32_t queued = ulTaskNotifyTake(pdFALSE, portMAX_DELAY);
        if (queued > 1) ++backlog_ticks;
        pending = std::max(pending, int(queued)-1);
    }

    void start() {
        peak = fifo.count;
        heap_before = heap_caps_get_free_size(caps);
        opened = esp_timer_get_time();
        if (xTaskCreate([](void* self){static_cast<CaptureOpus*>(self)->encode();},
                        "voice_mic", 48*1024, this, 4, nullptr) != pdPASS) abort();
    }

    void push(const int16_t* data, int n) {
        xSemaphoreTake(lock, portMAX_DELAY);
        if (!panel_voice_pcm_push(&fifo, data, n)) abort();
        peak = std::max(peak, int(fifo.count));
        xSemaphoreGive(lock);
    }

    void finish(int expected) {
        ESP_ERROR_CHECK(esp_timer_stop(timer));
        ESP_ERROR_CHECK(esp_timer_delete(timer));
        if (opened) {
            xSemaphoreTake(lock, portMAX_DELAY);
            ending = true;
            xSemaphoreGive(lock);
            if (xSemaphoreTake(finished, pdMS_TO_TICKS(10000)) != pdTRUE) abort();
            if (fifo.count || count != (expected+319)/320) abort();
        }
        std::printf("MIC_QUEUE %d %d %lld %d %d %d %lld %d %u %u %d\n",count,peak,
                    (long long)caught_up,pending,backlog_ticks,peak_encode_us,(long long)total_encode_us,
                    stack_free,unsigned(heap_before),unsigned(heap_ready),int(fifo.count));
        static constexpr char hex[] = "0123456789abcdef";
        char line[2*sizeof(Packet::data)+1];
        for (int i=0;i<count;++i) {
            const Packet& pkt = packets[i];
            for (int j=0;j<pkt.length;++j) {
                line[2*j]=hex[pkt.data[j]>>4];
                line[2*j+1]=hex[pkt.data[j]&15];
            }
            line[2*pkt.length]=0;
            std::printf("MIC_PACKET %d %d %s\n",i*20,i*frame,line);
        }
        free(packets);
        vSemaphoreDelete(lock);
        vSemaphoreDelete(finished);
    }
};
