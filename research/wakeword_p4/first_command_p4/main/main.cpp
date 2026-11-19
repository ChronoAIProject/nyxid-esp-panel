#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "c161_coded_cue_v5.h"
#include "esp_opus_enc.h"
#include "esp_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" {
#include "panel_voice_audio.h"
}

namespace {
extern const uint8_t digital[] asm("_binary_4090_c161_coded_cue_v5_command_digital_wav_start");
extern const uint8_t near_wave[] asm("_binary_4090_c161_coded_cue_v5_command_near_wav_start");
extern const uint8_t far_wave[] asm("_binary_4090_c161_coded_cue_v5_command_far_wav_start");

constexpr int rate = 16000, agc_frame = 160, frame = 320;
#ifdef NYX_AGC_REPLAY
#include "agc_replay.h"
#else
constexpr int capture_step = 160, prime = 81600;
constexpr int events[] = {102788, 102704, 102932};
#endif
static_assert(capture_step >= agc_frame && capture_step <= frame);
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

esp_opus_enc_config_t encoder_config() {
    esp_opus_enc_config_t cfg = ESP_OPUS_ENC_CONFIG_DEFAULT();
    cfg.sample_rate = rate;
    cfg.channel = 1;
    cfg.bitrate = 24000;
    cfg.complexity = 0;
    cfg.enable_vbr = true;
    return cfg;
}

void run(const Fixture& fixture, int delay_ms) {
    uint32_t size;
    std::memcpy(&size, fixture.wav + 40, 4);
    if (std::memcmp(fixture.wav + 36, "data", 4) || size % 2) abort();
    const int length = size / 2;
    panel_voice_pcm_t tail{lookback_memory, 2 * rate, 0, 0}, fifo{capture_memory, 4 * rate, 0, 0};
    nyx::CodedCueV5 detector;
    auto cfg = encoder_config();
    void* encoder = nullptr;
    int detected = -1, handoff = -1, first = -1, cursor = -1, pts = 0, peak = 0;
    int call_clock = -1, next_send = -1;
    bool started = false;
    std::printf("BEGIN %s %d\n", fixture.name, delay_ms);
    for (int clock = -prime + capture_step; clock < length + 4*rate && cursor < length; clock += capture_step) {
        // AEC consumes capture_step samples per read. Only completed AGC frames reach the FIFO.
        const int end = ((clock + prime) / agc_frame) * agc_frame - prime;
        const int at = ((clock - capture_step + prime) / agc_frame) * agc_frame - prime;
        const int produced = end - at;
        int16_t capture[frame]{};
        const int start = std::max(0, at);
        if (start < std::min(length, end)) {
            const int n = std::min(length, end) - start;
            std::memcpy(capture + (start-at), fixture.wav + 44 + 2*start, n*sizeof(int16_t));
        }
        if (!encoder) {
            panel_voice_pcm_preroll_push(&tail, capture, produced);
            for (int i = 0; i < produced; ++i) {
                if (detector.update(capture[i])) detected = at + i + 1;
            }
            if (detected < 0) continue;
            if (detected != fixture.event) abort();
            handoff = end;
            call_clock = next_send = clock;
            first = cursor = end - int(tail.count);
            const size_t part = std::min(tail.count, tail.capacity-tail.head);
            if (!panel_voice_pcm_push(&fifo,tail.samples+tail.head,part) ||
                    !panel_voice_pcm_push(&fifo,tail.samples,tail.count-part)) abort();
            panel_voice_pcm_clear(&tail);
            if (esp_opus_enc_open(&cfg,sizeof(cfg),&encoder) != ESP_AUDIO_ERR_OK) abort();
        } else {
            const bool ready = clock >= call_clock + delay_ms*(rate/1000);
            if (!panel_voice_pcm_gate(&fifo,&started,ready,false) ||
                    !panel_voice_pcm_push(&fifo,capture,produced)) abort();
        }
        peak = std::max(peak, int(fifo.count));
        if (clock < next_send) continue;
        const bool ready = next_send >= call_clock + delay_ms*(rate/1000);
        next_send += frame;
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
#ifdef NYX_AGC_REPLAY
    std::printf("CLOCK %s %d %d %d %d\n", fixture.name, delay_ms, call_clock, capture_step, prime);
#endif
    std::printf("END %s %d %d %d %d %d %d %d\n",fixture.name,delay_ms,detected,handoff,first,cursor,length,peak);
}

// This optional flash payload contains two complete, sample-aligned call captures.
bool captured_pair() {
    constexpr uint32_t offset = 2U << 20;
    uint32_t header[3];
    ESP_ERROR_CHECK(esp_flash_read(nullptr, header, offset, sizeof(header)));
    if (header[0] != 0x3552504f) return false;
    const unsigned samples = header[2];
    if (!samples || samples > ((4U<<20)-offset-sizeof(header))/4) abort();
    auto cfg = encoder_config();
    for (unsigned lane=0; lane<2; ++lane) {
        std::printf("UPLINK %lu %u %u\n",(unsigned long)header[1],lane,samples);
        void* encoder = nullptr;
        if (esp_opus_enc_open(&cfg,sizeof(cfg),&encoder) != ESP_AUDIO_ERR_OK) abort();
        unsigned frames = 0;
        for (unsigned start=0; start<samples; start+=frame) {
            alignas(16) int16_t pcm[frame]{};
            unsigned n = std::min(unsigned(frame),samples-start);
            ESP_ERROR_CHECK(esp_flash_read(nullptr,pcm,offset+sizeof(header)+2*(lane*samples+start),2*n));
            packet(encoder,frames*20,start,pcm);
            ++frames;
        }
        esp_opus_enc_close(encoder);
        std::printf("UPLINK_END %u %u\n",lane,frames);
    }
    return true;
}

void worker(void*) {
    if (captured_pair()) {
        std::puts("UPLINK PASS");
        vTaskDelete(nullptr);
        return;
    }
    const Fixture fixtures[] = {{"digital",digital,events[0]},{"near",near_wave,events[1]},{"far",far_wave,events[2]}};
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
