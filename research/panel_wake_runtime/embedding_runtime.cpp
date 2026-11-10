// Experimental detector with an optional ordered consumer; no audio/network/NVS owner.
#include "panel_wake_runtime.h"
#include "embedding_config.h"
#include "embedding_frontend.h"
#include "embedding_fc.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
extern "C" {
#include "zlib.h"
}

extern const uint8_t backbone_data[] asm("_binary_embedding_backbone_bin_start");
extern const uint8_t backbone_end[] asm("_binary_embedding_backbone_bin_end");
extern const uint8_t dense_data[] asm("_binary_embedding_dense_bin_start");
extern const uint8_t dense_end[] asm("_binary_embedding_dense_bin_end");
extern const uint8_t temporal_data[] asm("_binary_embedding_temporal_bin_start");
extern const uint8_t temporal_end[] asm("_binary_embedding_temporal_bin_end");
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS
extern "C" void nyx_wake_capture_counters(uint32_t *, uint32_t *);
#endif

namespace {
constexpr char tag[] = "panel.wake.embedding";
constexpr size_t block_samples = 1280;
using Resolver = tflite::MicroMutableOpResolver<24>;

struct SchedulerProfile {
#if configUSE_TRACE_FACILITY && configGENERATE_RUN_TIME_STATS
    TaskStatus_t tasks[128]{};
    struct Previous { UBaseType_t number; configRUN_TIME_COUNTER_TYPE runtime; } previous[128]{};
    unsigned previous_count = 0;
    int64_t previous_time = 0;
    void print(unsigned steps) {
        configRUN_TIME_COUNTER_TYPE total = 0;
        UBaseType_t count = uxTaskGetSystemState(tasks, 128, &total);
        int64_t now = esp_timer_get_time();
        if (!count) { ESP_LOGW(tag, "scheduler snapshot exceeded fixed capacity"); return; }
        if (previous_time) {
            for (unsigned i = 0; i < count; ++i) for (unsigned j = 0; j < previous_count; ++j) {
                if (tasks[i].xTaskNumber != previous[j].number) continue;
                auto delta = tasks[i].ulRunTimeCounter - previous[j].runtime;
                if (uint64_t(delta) > uint64_t(now - previous_time) / 100)
                    ESP_LOGI(tag, "scheduler step=%u task=%s core=%d priority=%u cpu_us=%u wall_us=%u",
                        steps, tasks[i].pcTaskName, int(tasks[i].xCoreID), unsigned(tasks[i].uxCurrentPriority),
                        unsigned(delta), unsigned(now - previous_time));
            }
        }
        for (unsigned i = 0; i < count; ++i) previous[i] = {tasks[i].xTaskNumber, tasks[i].ulRunTimeCounter};
        previous_count = count; previous_time = now;
    }
#else
    void print(unsigned) {}
#endif
};

// Fixed storage: diagnostic attribution only, never an audio queue or clock.
struct OperatorProfile : tflite::MicroProfilerInterface {
    const char *tags[32]{};
    uint64_t totals[32]{};
    const char *events[16]{};
    int64_t starts[16]{};
    unsigned depth = 0;
    uint32_t BeginEvent(const char *name) override {
        if (depth == 16) abort();
        unsigned at = depth++;
        events[at] = name; starts[at] = esp_timer_get_time();
        return at;
    }
    void EndEvent(uint32_t at) override {
        if (!depth || at != depth - 1) abort();
        --depth;
        unsigned slot = 0;
        while (slot < 31 && tags[slot] && strcmp(tags[slot], events[at])) ++slot;
        tags[slot] = events[at];
        totals[slot] += esp_timer_get_time() - starts[at];
    }
    void print(unsigned steps) const {
        for (unsigned n = 0; n < 32; ++n) if (tags[n])
            ESP_LOGI(tag, "operator step=%u name=%s mean_us=%u", steps, tags[n], unsigned(totals[n] / steps));
    }
};

void *lut_init(TfLiteContext *, const char *bytes, size_t length) {
    return length == 256 ? const_cast<char *>(bytes) : nullptr;
}
TfLiteStatus lut_prepare(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE(context, node->user_data != nullptr);
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
    TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
    const auto *input = tflite::micro::GetEvalInput(context, node, 0);
    const auto *output = tflite::micro::GetEvalOutput(context, node, 0);
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, tflite::micro::GetTensorShape(input).FlatSize(),
        tflite::micro::GetTensorShape(output).FlatSize());
    return kTfLiteOk;
}
TfLiteStatus lut_invoke(TfLiteContext *context, TfLiteNode *node) {
    const auto *input = tflite::micro::GetEvalInput(context, node, 0);
    auto *output = tflite::micro::GetEvalOutput(context, node, 0);
    const auto *table = static_cast<const int8_t *>(node->user_data);
    const int count = tflite::micro::GetTensorShape(input).FlatSize();
    for (int n = 0; n < count; ++n) output->data.int8[n] = table[int(input->data.int8[n]) + 128];
    return kTfLiteOk;
}

struct ModelSlot {
    uint8_t *model = nullptr, *arena = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr, *output = nullptr;
    ~ModelSlot() { delete interpreter; free(arena); free(model); }

    bool initialize(const uint8_t *packed, size_t bytes, size_t arena_size,
                    Resolver &ops, tflite::MicroResourceVariables *variables = nullptr,
                    uint32_t arena_caps = MALLOC_CAP_SPIRAM,
                    tflite::MicroProfilerInterface *profile = nullptr) {
        if (bytes < 5) return false;
        uint32_t size;
        memcpy(&size, packed, sizeof(size));
        if (size < 1024 || size > 512 * 1024) return false;
        model = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, arena_size, arena_caps | MALLOC_CAP_8BIT));
        if (!model || !arena) return false;
        uLongf expanded = size;
        if (uncompress(model, &expanded, packed + 4, bytes - 4) != Z_OK || expanded != size) return false;
        const auto *graph = tflite::GetModel(model);
        if (graph->version() != TFLITE_SCHEMA_VERSION) return false;
        interpreter = new(std::nothrow) tflite::MicroInterpreter(graph, ops, arena, arena_size, variables, profile);
        if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) return false;
        input = interpreter->input(0); output = interpreter->output(0);
        return input && output && input->type == kTfLiteInt8 && output->type == kTfLiteInt8;
    }
};

struct Detector {
    Resolver ops;
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE
    OperatorProfile profile;
    SchedulerProfile scheduler;
#endif
    // Destroy slots before the resource-variable backing storage.
    uint8_t *variable_arena = nullptr;
    ModelSlot slots[3];
    NativeScratch scratch{};
    int16_t pcm[1760]{};
    int8_t history[16 * 96]{};
    int score_history[3]{};
    unsigned steps = 0, score_count = 0, score_at = 0, late_count = 0;
    int64_t last_event = -1000;
    bool armed = true, failed = false;
    uint64_t inference_us = 0;
    unsigned maximum_us = 0;
    unsigned recent_us[100]{};
    uint64_t stage_us[4]{};

    ~Detector() {
        // ModelSlot destructors normally run after this body. Explicitly
        // release interpreters before freeing storage used by their variables.
        for (auto &slot : slots) { delete slot.interpreter; slot.interpreter = nullptr; }
        free(variable_arena);
    }

    bool initialize() {
        if (ops.AddAssignVariable() != kTfLiteOk || ops.AddCallOnce() != kTfLiteOk ||
            ops.AddConcatenation() != kTfLiteOk || ops.AddReadVariable() != kTfLiteOk ||
            ops.AddStridedSlice() != kTfLiteOk || ops.AddVarHandle() != kTfLiteOk ||
            ops.AddPad() != kTfLiteOk || ops.AddConv2D() != kTfLiteOk ||
            ops.AddMaxPool2D() != kTfLiteOk || ops.AddReshape() != kTfLiteOk ||
            ops.AddFullyConnected(nyx_head_probe::reference_fc_registration()) != kTfLiteOk ||
            ops.AddMean() != kTfLiteOk || ops.AddDequantize() != kTfLiteOk ||
            ops.AddNeg() != kTfLiteOk || ops.AddQuantize() != kTfLiteOk ||
            ops.AddSquaredDifference() != kTfLiteOk || ops.AddAdd() != kTfLiteOk ||
            ops.AddRsqrt() != kTfLiteOk || ops.AddMul() != kTfLiteOk ||
            ops.AddLogistic() != kTfLiteOk || ops.AddReduceMax() != kTfLiteOk) return false;
        auto lut = tflite::micro::RegisterOp(lut_init, lut_prepare, lut_invoke);
        if (ops.AddCustom("NYX_INT8_LUT", &lut) != kTfLiteOk) return false;
        variable_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, 16 * 1024, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (!variable_arena) return false;
        auto *allocator = tflite::MicroAllocator::Create(variable_arena, 16 * 1024);
        auto *variables = allocator ? tflite::MicroResourceVariables::Create(allocator, 20) : nullptr;
        tflite::MicroProfilerInterface *operator_profile = nullptr;
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE
        operator_profile = &profile;
#endif
        if (!variables || !slots[0].initialize(backbone_data, backbone_end - backbone_data, 96 * 1024, ops, variables, MALLOC_CAP_INTERNAL, operator_profile) ||
            !slots[1].initialize(dense_data, dense_end - dense_data, 16 * 1024, ops) ||
            !slots[2].initialize(temporal_data, temporal_end - temporal_data, 24 * 1024, ops)) return false;
        auto &backbone = slots[0];
        if (backbone.input->bytes != 256 || backbone.output->bytes != 96 ||
            backbone.input->params.scale != input_scale || backbone.input->params.zero_point != input_zero) return false;
        for (int n = 1; n < 3; ++n) {
            const auto &s = slots[n];
            if (s.input->bytes != sizeof(history) || s.output->bytes != 1 ||
                s.input->params.scale != 1.f / 128 || s.input->params.zero_point != 0 ||
                s.output->params.scale != 1.f / 256 || s.output->params.zero_point != -128) return false;
        }
        ESP_LOGI(tag, "bundle=%s threshold=%.6f arena_used=%u/%u/%u psram_reserved=40960 internal_reserved=114688 core=%d",
            embedding_bundle_sha256, embedding_threshold,
            unsigned(slots[0].interpreter->arena_used_bytes()),
            unsigned(slots[1].interpreter->arena_used_bytes()),
            unsigned(slots[2].interpreter->arena_used_bytes()), int(xPortGetCoreID()));
        return true;
    }

    int detect(const int16_t *samples) {
        if (failed) return -1;
        int64_t start = esp_timer_get_time();
        memcpy(pcm + 480, samples, block_samples * sizeof(int16_t));
        native_frontend(pcm, scratch);
        for (unsigned n = 0; n < 256; ++n) slots[0].input->data.int8[n] = native_quantize(scratch.features[n]);
        memmove(pcm, pcm + block_samples, 480 * sizeof(int16_t));
        int64_t stage = esp_timer_get_time();
        stage_us[0] += stage - start;
        if (slots[0].interpreter->Invoke() != kTfLiteOk) { failed = true; return -1; }
        int64_t end = esp_timer_get_time();
        stage_us[1] += end - stage;
        memmove(history, history + 96, sizeof(history) - 96);
        memcpy(history + sizeof(history) - 96, slots[0].output->data.int8, 96);
        ++steps;
        int numerator = 0;
        if (steps >= 25) {
            for (int n = 1; n < 3; ++n) {
                memcpy(slots[n].input->data.int8, history, sizeof(history));
                stage = esp_timer_get_time();
                if (slots[n].interpreter->Invoke() != kTfLiteOk) { failed = true; return -1; }
                stage_us[n + 1] += esp_timer_get_time() - stage;
                numerator += int(slots[n].output->data.int8[0]) + 128;
            }
            score_history[score_at] = numerator; score_at = (score_at + 1) % 3;
            score_count = std::min(score_count + 1, 3u);
        }
        unsigned elapsed = unsigned(esp_timer_get_time() - start);
        inference_us += elapsed; maximum_us = std::max(maximum_us, elapsed);
        late_count += elapsed > 80000;
        recent_us[(steps - 1) % 100] = elapsed;
        if (steps % 100 == 0) {
            unsigned sorted[100];
            memcpy(sorted, recent_us, sizeof(sorted));
            std::sort(sorted, sorted + 100);
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS
            uint32_t received, overflows;
            nyx_wake_capture_counters(&received, &overflows);
            ESP_LOGI(tag, "capture step=%u rx_bytes=%u rx_overflows=%u", steps, received, overflows);
#endif
            ESP_LOGI(tag, "step=%u mean_us=%u p50_us=%u p95_us=%u max_us=%u over80ms=%u",
                steps, unsigned(inference_us / steps), sorted[50], sorted[95], maximum_us, late_count);
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE
            ESP_LOGI(tag, "stages step=%u frontend_us=%u backbone_us=%u dense_us=%u temporal_us=%u core=%d",
                steps, unsigned(stage_us[0] / steps), unsigned(stage_us[1] / steps),
                unsigned(stage_us[2] / steps), unsigned(stage_us[3] / steps), int(xPortGetCoreID()));
            profile.print(steps);
            scheduler.print(steps);
#endif
        }
        if (score_count != 3) return 0;
        double score = double(score_history[0] + score_history[1] + score_history[2]) / 1536;
        if (score < embedding_threshold * .5 && int64_t(steps) - last_event >= 25) armed = true;
        if (armed && score >= embedding_threshold && int64_t(steps) - last_event >= 25) {
            armed = false; last_event = steps;
            ESP_LOGI(tag, "activation score=%.6f step=%u inference_us=%u", score, steps, elapsed);
            return 1;
        }
        return 0;
    }
};

#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC
// One ordered consumer, fed exclusively by the existing I2S owner. No audio
// device, resampler, second recording stream or network client lives here.
struct AsyncDetector {
    struct Block { int16_t samples[block_samples]; int64_t queued_us; unsigned sequence; uint32_t crc; };
    static constexpr unsigned capacity = 4;
    Detector *detector = nullptr;
    uint8_t *storage = nullptr;
    StaticQueue_t queue_control{};
    QueueHandle_t queue = nullptr;
    SemaphoreHandle_t stopped = nullptr;
    TaskHandle_t task = nullptr;
    std::atomic<bool> stop{false}, failed{false};
    std::atomic<int> event{0};
    std::atomic<unsigned> highwater{0};
    unsigned prime_steps = 0, submitted = 0, consumed = 0;
    uint64_t total_delay_us = 0;
    unsigned max_delay_us = 0;

    ~AsyncDetector() {
        if (task) { stop.store(true); xSemaphoreTake(stopped, portMAX_DELAY); }
        if (queue) vQueueDelete(queue);
        if (stopped) vSemaphoreDelete(stopped);
        free(storage);
        if (detector) { detector->~Detector(); free(detector); }
    }
    bool initialize(Detector *value) {
        detector = value;
        storage = static_cast<uint8_t *>(heap_caps_malloc(capacity * sizeof(Block), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        stopped = xSemaphoreCreateBinary();
        if (storage) queue = xQueueCreateStatic(capacity, sizeof(Block), storage, &queue_control);
        return queue && stopped;
    }
    static void run(void *argument) {
        auto *self = static_cast<AsyncDetector *>(argument);
        Block block;
        while (!self->stop.load()) {
            if (xQueueReceive(self->queue, &block, pdMS_TO_TICKS(20)) != pdTRUE) continue;
            if (self->stop.load()) break;
            if (block.sequence != self->consumed + 1 || block.crc !=
                    crc32(0, reinterpret_cast<const Bytef *>(block.samples), sizeof(block.samples))) {
                ESP_LOGE(tag, "ordered detector sequence/PCM integrity failure");
                self->failed.store(true); break;
            }
            unsigned delay = unsigned(esp_timer_get_time() - block.queued_us);
            self->total_delay_us += delay;
            self->max_delay_us = std::max(self->max_delay_us, delay);
            int result = self->detector->detect(block.samples);
            if (result < 0) { self->failed.store(true); break; }
            if (result > 0) self->event.store(1);
            if (++self->consumed % 100 == 0)
                ESP_LOGI(tag, "queue consumed=%u highwater=%u max_delay_us=%u mean_delay_us=%u",
                    self->consumed, self->highwater.load(), self->max_delay_us,
                    unsigned(self->total_delay_us / self->consumed));
        }
        xSemaphoreGive(self->stopped);
        vTaskDelete(nullptr);
    }
    int detect(const int16_t *samples) {
        // The caller's existing 38400sample priming loop remains synchronous.
        // Start the sole consumer only after all30zero blocks have completed.
        if (prime_steps < 30) {
            int result = detector->detect(samples);
            if (result < 0) return result;
            if (++prime_steps == 30 && xTaskCreatePinnedToCore(run, "wake_model", 16384,
                    this, 4, &task, 1) != pdPASS) {
                task = nullptr; failed.store(true); return -1;
            }
            return result;
        }
        if (failed.load()) return -1;
        Block block;
        memcpy(block.samples, samples, sizeof(block.samples));
        block.queued_us = esp_timer_get_time();
        block.sequence = ++submitted;
        block.crc = crc32(0, reinterpret_cast<const Bytef *>(block.samples), sizeof(block.samples));
        if (xQueueSend(queue, &block, 0) != pdTRUE) {
            // No overwrites, skipped frames or hidden backlog. Fail the
            // experimental detector if its consumer cannot keep up.
            ESP_LOGE(tag, "ordered detector queue overflow");
            failed.store(true); stop.store(true); return -1;
        }
        highwater.store(std::max(highwater.load(), unsigned(uxQueueMessagesWaiting(queue))));
        return event.exchange(0);
    }
};
#endif

model_iface_data_t *create(const void *, det_mode_t) {
    void *memory = heap_caps_malloc(sizeof(Detector), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!memory) return nullptr;
    auto *detector = new(memory) Detector;
    if (!detector->initialize()) { detector->~Detector(); free(memory); return nullptr; }
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC
    // Queue control and synchronization atomics stay in internal memory;
    // only the bounded PCM storage and model objects use PSRAM.
    void *wrapper_memory = heap_caps_malloc(sizeof(AsyncDetector), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!wrapper_memory) { detector->~Detector(); free(memory); return nullptr; }
    auto *wrapper = new(wrapper_memory) AsyncDetector;
    if (!wrapper->initialize(detector)) { wrapper->~AsyncDetector(); free(wrapper_memory); return nullptr; }
    return reinterpret_cast<model_iface_data_t *>(wrapper);
#else
    return reinterpret_cast<model_iface_data_t *>(detector);
#endif
}
void destroy(model_iface_data_t *model) {
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC
    auto *detector = reinterpret_cast<AsyncDetector *>(model);
    if (detector) { detector->~AsyncDetector(); free(detector); }
#else
    auto *detector = reinterpret_cast<Detector *>(model);
    if (detector) { detector->~Detector(); free(detector); }
#endif
}
wakenet_state_t detect(model_iface_data_t *model, int16_t *samples) {
#ifdef NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC
    return static_cast<wakenet_state_t>(reinterpret_cast<AsyncDetector *>(model)->detect(samples));
#else
    return static_cast<wakenet_state_t>(reinterpret_cast<Detector *>(model)->detect(samples));
#endif
}
int chunk(model_iface_data_t *) { return block_samples; }
int rate(model_iface_data_t *) { return 16000; }
int channels(model_iface_data_t *) { return 1; }
} // namespace

extern "C" const esp_wn_iface_t panel_wake_runtime_iface = {
    .create=create, .get_start_point=nullptr, .get_samp_chunksize=chunk,
    .get_channel_num=channels, .get_samp_rate=rate, .get_word_num=nullptr,
    .get_word_name=nullptr, .set_det_threshold=nullptr, .reset_det_threshold=nullptr,
    .get_det_threshold=nullptr, .get_triggered_channel=nullptr, .get_vol_gain=nullptr,
    .detect=detect, .detect_mfcc=nullptr, .get_mfcc_data=nullptr, .clean=nullptr, .destroy=destroy
};
extern "C" int panel_wake_runtime_prime_samples(void) { return 38400; }
