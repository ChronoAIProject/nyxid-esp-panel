// Opt-in original C161/B40/history adapter. Capture, AEC/AGC and FIFO stay in
// the application; this adapter consumes the single native 16 kHz stream.
#include "panel_wake_runtime.h"
#include "parallel_detector.h"
#include <new>

extern const uint8_t c161_bytes[] asm("_binary_c161_bin_start");
extern const uint8_t binary40_bytes[] asm("_binary_binary40_bin_start");
extern const uint8_t history_bytes[] asm("_binary_history_expert_bin_start");

namespace {
using panel_wake_parallel::ParallelDetector;
using panel_wake_parallel::chunk_samples;
using panel_wake_parallel::tag;

void log_heap(const char* phase) {
    constexpr unsigned caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    ESP_LOGI(tag, "resources phase=%s internal_free=%u internal_largest=%u internal_global_min=%u",
        phase, unsigned(heap_caps_get_free_size(caps)),
        unsigned(heap_caps_get_largest_free_block(caps)),
        unsigned(heap_caps_get_minimum_free_size(caps)));
}

model_iface_data_t* create(const void*, det_mode_t) {
    log_heap("before_create");
    auto* detector = new(std::nothrow)ParallelDetector;
    if (!detector) return nullptr;
    if (!detector->initialize(c161_bytes, binary40_bytes, history_bytes)) {
        ESP_LOGE(tag, "Parallel initialization failed");
        log_heap("failed_before_cleanup");
        delete detector;
        log_heap("failed_after_cleanup");
        return nullptr;
    }
    log_heap("after_create");
    return reinterpret_cast<model_iface_data_t*>(detector);
}
void destroy(model_iface_data_t* data) {
    delete reinterpret_cast<ParallelDetector*>(data);
    log_heap("after_destroy");
}
wakenet_state_t detect(model_iface_data_t* data, int16_t* pcm) {
    return static_cast<wakenet_state_t>(reinterpret_cast<ParallelDetector*>(data)->detect(pcm));
}
int chunk(model_iface_data_t*) { return chunk_samples; }
int rate(model_iface_data_t*) { return 16000; }
int channels(model_iface_data_t*) { return 1; }
} // namespace

extern "C" const esp_wn_iface_t panel_wake_runtime_iface = {
    .create=create, .get_start_point=nullptr, .get_samp_chunksize=chunk,
    .get_channel_num=channels, .get_samp_rate=rate, .get_word_num=nullptr,
    .get_word_name=nullptr, .set_det_threshold=nullptr, .reset_det_threshold=nullptr,
    .get_det_threshold=nullptr, .get_triggered_channel=nullptr, .get_vol_gain=nullptr,
    .detect=detect, .detect_mfcc=nullptr, .get_mfcc_data=nullptr, .clean=nullptr, .destroy=destroy
};
extern "C" int panel_wake_runtime_prime_samples(void) { return 81600; }
