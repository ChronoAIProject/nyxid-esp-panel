#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "c161_coded_cue_v5.h"
#include "esp_opus_enc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" {
#include "panel_voice_audio.h"
}

namespace {
extern const uint8_t digital[] asm("_binary_4090_c161_coded_cue_v5_command_digital_wav_start");
extern const uint8_t near_wave[] asm("_binary_4090_c161_coded_cue_v5_command_near_wav_start");
extern const uint8_t far_wave[] asm("_binary_4090_c161_coded_cue_v5_command_far_wav_start");

constexpr int rate = 16000, capture_step = 160, frame = 320, prime = 81600;
static int16_t lookback_memory[2 * rate], capture_memory[4 * rate];

struct Fixture {
    const char* name;
    const uint8_t* wav;
    int event;
};

void packet(void* encoder, int pts, int source, int16_t* pcm) {
    uint8_t bytes[2048];
    esp_audio_enc_in_frame_t input{};
    input.buffer = reinterpret_cast<uint8_t*>(pcm);
    input.len = frame * sizeof(*pcm);
    esp_audio_enc_out_frame_t output{};
    output.buffer = bytes;
    output.len = sizeof(bytes);
    if (esp_opus_enc_process(encoder, &input, &output) != ESP_AUDIO_ERR_OK ||
            !output.encoded_bytes || output.encoded_bytes > sizeof(bytes)) abort();
    static const char hex[] = "0123456789abcdef";
    char encoded[2 * sizeof(bytes) + 1];
    for (size_t i = 0; i < output.encoded_bytes; ++i) {
        encoded[2*i] = hex[bytes[i] >> 4];
        encoded[2*i+1] = hex[bytes[i] & 15];
    }
    encoded[2 * output.encoded_bytes] = 0;
    std::printf("PACKET %d %d %s\n", pts, source, encoded);
}

void run(const Fixture& fixture, int delay_ms) {
    uint32_t size;
    std::memcpy(&size, fixture.wav + 40, 4);
    if (std::memcmp(fixture.wav + 36, "data", 4) || size % 2) abort();
    const int length = size / 2;
    panel_voice_pcm_t tail{lookback_memory, 2 * rate, 0, 0}, fifo{capture_memory, 4 * rate, 0, 0};
    nyx::CodedCueV5 detector;
    esp_opus_enc_config_t cfg = ESP_OPUS_ENC_CONFIG_DEFAULT();
    cfg.sample_rate = rate;
    cfg.channel = 1;
    cfg.bitrate = 24000;
    cfg.complexity = 0;
    cfg.enable_vbr = true;
    void* encoder = nullptr;
    int detected = -1, handoff = -1, first = -1, cursor = -1, pts = 0, peak = 0;
    bool started = false;
    std::printf("BEGIN %s %d\n", fixture.name, delay_ms);
    for (int at = -prime; at < length + 4*rate && cursor < length; at += capture_step) {
        int16_t capture[capture_step]{};
        if (at >= 0 && at < length) {
            const int n = std::min(capture_step, length-at);
            std::memcpy(capture, fixture.wav + 44 + 2*at, n*sizeof(int16_t));
        }
        const int end = at + capture_step;
        if (!encoder) {
            panel_voice_pcm_preroll_push(&tail, capture, capture_step);
            for (int i = 0; i < capture_step; ++i) {
                if (detector.update(capture[i])) detected = at + i + 1;
            }
            if (detected < 0) continue;
            if (detected != fixture.event) abort();
            handoff = end;
            first = cursor = end - int(tail.count);
            const size_t part = std::min(tail.count, tail.capacity-tail.head);
            if (!panel_voice_pcm_push(&fifo,tail.samples+tail.head,part) ||
                    !panel_voice_pcm_push(&fifo,tail.samples,tail.count-part)) abort();
            panel_voice_pcm_clear(&tail);
            if (esp_opus_enc_open(&cfg,sizeof(cfg),&encoder) != ESP_AUDIO_ERR_OK) abort();
        } else {
            const bool ready = end >= handoff + delay_ms*(rate/1000);
            if (!panel_voice_pcm_gate(&fifo,&started,ready,false) ||
                    !panel_voice_pcm_push(&fifo,capture,capture_step)) abort();
        }
        peak = std::max(peak, int(fifo.count));
        if ((end - handoff) % frame != 0) continue;
        const bool ready = end >= handoff + delay_ms*(rate/1000);
        panel_voice_pcm_gate(&fifo,&started,ready,false);
        for (int burst = 0; burst < (ready ? 4 : 1) && cursor < length; ++burst) {
            int16_t buffer[frame]{};
            if (ready && !panel_voice_pcm_pop(&fifo,buffer,frame)) break;
            if (ready) {
                for (int i = 0; i < frame; ++i) {
                    int16_t expected = 0;
                    if (cursor+i < length)
                        std::memcpy(&expected,fixture.wav+44+2*(cursor+i),2);
                    if (buffer[i] != expected) abort();
                }
            }
            packet(encoder,pts,ready ? cursor : -1,buffer);
            if (ready) cursor += frame;
            pts += 20;
        }
    }
    if (!encoder || cursor < length || cursor >= length+frame || peak >= int(fifo.capacity)) abort();
    esp_opus_enc_close(encoder);
    std::printf("END %s %d %d %d %d %d %d %d\n",fixture.name,delay_ms,detected,handoff,first,cursor,length,peak);
}

void worker(void*) {
    const Fixture fixtures[] = {{"digital",digital,102788},{"near",near_wave,102704},{"far",far_wave,102932}};
    for (const auto& fixture : fixtures) {
        run(fixture,0);
        run(fixture,1500);
    }
    std::puts("FIRST_COMMAND PASS");
    vTaskDelete(nullptr);
}
}  // namespace

extern "C" void app_main() {
    if (xTaskCreate(worker,"codec_replay",48*1024,nullptr,3,nullptr) != pdPASS) abort();
}
