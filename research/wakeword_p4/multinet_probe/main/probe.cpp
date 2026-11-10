// Standalone synthetic PCM diagnostic: no microphone, speakers, network or NVS.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_mn_models.h"
#include "esp_mn_speech_commands.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model_path.h"
#include "zlib.h"
#include "fixture_config.h"
#include "readonly_model_files.h"

extern const uint8_t packed_model[] asm("_binary_model_bin_start");
extern const uint8_t packed_model_end[] asm("_binary_model_bin_end");
extern const uint8_t packed_fixtures[] asm("_binary_fixtures_bin_start");
extern const uint8_t packed_fixtures_end[] asm("_binary_fixtures_bin_end");

static uint32_t read32(const uint8_t *p) {
    uint32_t n;
    memcpy(&n, p, sizeof(n));
    return n;
}

static uint8_t *expand(const uint8_t *begin, const uint8_t *end, size_t *size) {
    if (end - begin < 5) return nullptr;
    *size = read32(begin);
    if (!*size || *size > 16 * 1024 * 1024) return nullptr;
    auto *result = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, *size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!result) return nullptr;
    uLongf expanded = *size;
    if (uncompress(result, &expanded, begin + 4, end - begin - 4) != Z_OK || expanded != *size) {
        free(result);
        return nullptr;
    }
    return result;
}

static void json_string(const char *value, size_t capacity) {
    putchar('"');
    for (size_t n = 0; n < capacity && value[n]; ++n) {
        const unsigned char ch = value[n];
        if (ch == '"' || ch == '\\') printf("\\%c", ch);
        else if (ch < 32 || ch >= 127) printf("\\u%04x", unsigned(ch));
        else putchar(ch);
    }
    putchar('"');
}

static bool print_result(const char *marker, unsigned arm, unsigned clip,
                         unsigned end_sample, const esp_mn_results_t *result) {
    if (!result || result->num < 0 || result->num > ESP_MN_RESULT_MAX_NUM) return false;
    printf("%s {\"arm\":%u,\"clip\":%u,\"end_sample\":%u,\"state\":%d,\"num\":%d,\"commands\":[",
           marker, arm, clip, end_sample, int(result->state), result->num);
    for (int n = 0; n < result->num; ++n) {
        if (n) putchar(',');
        printf("{\"id\":%d,\"phrase_id\":%d,\"probability\":%.8f}",
               result->command_id[n], result->phrase_id[n], result->prob[n]);
    }
    printf("],\"string\":");
    json_string(result->string, sizeof(result->string));
    printf(",\"raw_string\":");
    json_string(result->raw_string, sizeof(result->raw_string));
    printf("}\n");
    fflush(stdout);
    return true;
}

static bool run() {
    size_t model_size, fixtures_size;
    auto *model_root = expand(packed_model, packed_model_end, &model_size);
    auto *fixtures = expand(packed_fixtures, packed_fixtures_end, &fixtures_size);
    if (!model_root || !fixtures || fixtures_size < 4 || read32(fixtures) != fixture_count) return false;
    srmodel_list_t *models = srmodel_load(model_root);
    if (!models || models->num != 1 || strcmp(models->model_name[0], "mn7_en")) return false;
    if (!model_files::install(models)) return false;
    auto *iface = esp_mn_handle_from_name(models->model_name[0]);
    if (!iface) return false;
    printf("MN_STAGE models_loaded model=%s\n", models->model_name[0]);
    fflush(stdout);
    const auto internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const auto psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    const int64_t create_start = esp_timer_get_time();
    auto *detector = iface->create(models->model_name[0], detector_timeout_ms);
    if (!detector) return false;
    const int64_t create_us = esp_timer_get_time() - create_start;
    const int chunk = iface->get_samp_chunksize(detector);
    if (chunk != 512 || iface->get_samp_rate(detector) != 16000) return false;
    iface->set_det_threshold(detector, detector_threshold);
    printf("MN_READY {\"chunk\":%d,\"rate\":16000,\"create_us\":%lld,"
        "\"internal_delta\":%lld,\"psram_delta\":%lld,\"model_expanded_bytes\":%u,"
        "\"fixtures_expanded_bytes\":%u}\n", chunk, (long long)create_us,
        (long long)internal_before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (long long)psram_before - heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT),
        unsigned(model_size), unsigned(fixtures_size));
    auto *buffer = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16, chunk * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    auto *timings = static_cast<int64_t *>(heap_caps_malloc(20000 * sizeof(int64_t), MALLOC_CAP_SPIRAM));
    if (!buffer || !timings) return false;
    unsigned calls = 0;
    bool passed = true;
    for (unsigned arm = 0; arm < command_arm_count && passed; ++arm) {
        // A zero-count first arm deliberately preserves the official grammar
        // loaded during create. Later zero-count arms would retain a replacement.
        if (!command_counts[arm] && arm != 0) return false;
        if (command_counts[arm]) {
            // alloc frees the previous global command list. destroy owns the
            // last list; explicitly freeing it too caused the old cleanup warning.
            ESP_ERROR_CHECK(esp_mn_commands_alloc(iface, detector));
            for (unsigned n = 0; n < command_counts[arm]; ++n) {
#ifdef MN_EXPLICIT_COMMAND_PHONEMES
                const char *phonemes = command_phonemes[arm][n];
                ESP_ERROR_CHECK(phonemes
                    ? esp_mn_commands_phoneme_add(n + 1, command_strings[arm][n], phonemes)
                    : esp_mn_commands_add(n + 1, command_strings[arm][n]));
#else
                ESP_ERROR_CHECK(esp_mn_commands_add(n + 1, command_strings[arm][n]));
#endif
            }
            auto *errors = esp_mn_commands_update();
            if (errors && errors->num) return false;
        }
        printf("MN_ARM {\"arm\":%u,\"stock_grammar\":%s}\n", arm, command_counts[arm] ? "false" : "true");
        iface->print_active_speech_commands(detector);
        const uint8_t *cursor = fixtures + 4;
        for (unsigned clip = 0; clip < fixture_count; ++clip) {
            if (cursor + 4 > fixtures + fixtures_size) return false;
            const uint32_t samples = read32(cursor);
            cursor += 4;
            if (!samples || size_t(fixtures + fixtures_size - cursor) < size_t(samples) * 2) return false;
            iface->clean(detector);
            unsigned target_events = 0, other_events = 0, timeouts = 0;
            int first_target_end = -1;
            float first_target_probability = -1;
            uLong input_crc = crc32(0, Z_NULL, 0);
            for (uint32_t at = 0; at < samples; at += chunk) {
                memset(buffer, 0, chunk * sizeof(int16_t));
                memcpy(buffer, cursor + at * 2, std::min(uint32_t(chunk), samples - at) * 2);
                if (calls >= 20000) return false;
                input_crc = crc32(input_crc, reinterpret_cast<const Bytef *>(buffer), chunk * sizeof(int16_t));
                const int64_t start = esp_timer_get_time();
                auto state = iface->detect(detector, buffer);
                timings[calls++] = esp_timer_get_time() - start;
                if (state == ESP_MN_STATE_DETECTED) {
                    auto *result = iface->get_results(detector);
                    if (!result || result->num < 1 || result->num > ESP_MN_RESULT_MAX_NUM) return false;
                    if (!print_result("MN_EVENT", arm, clip, at + chunk, result)) return false;
                    if (result->command_id[0] == 1) {
                        ++target_events;
                        if (first_target_end < 0) {
                            first_target_end = at + chunk;
                            first_target_probability = result->prob[0];
                        }
                    } else ++other_events;
                    iface->clean(detector);
                } else if (state == ESP_MN_STATE_TIMEOUT) {
                    ++timeouts;
                    iface->clean(detector);
                } else if (state != ESP_MN_STATE_DETECTING) return false;
                if (calls % 16 == 0) vTaskDelay(1);
            }
            if (!print_result("MN_DECODER", arm, clip, ((samples + chunk - 1) / chunk) * chunk,
                              iface->get_results(detector))) return false;
            cursor += samples * 2;
            printf("MN_CLIP {\"arm\":%u,\"clip\":%u,\"samples\":%lu,\"target_events\":%u,"
                "\"other_events\":%u,\"timeouts\":%u,\"first_target_end_sample\":%d,"
                "\"first_target_probability\":%.8f,\"input_crc32\":%lu}\n", arm, clip, (unsigned long)samples,
                target_events, other_events, timeouts, first_target_end, first_target_probability,
                (unsigned long)input_crc);
            fflush(stdout);
        }
        passed = cursor == fixtures + fixtures_size;
    }
    std::sort(timings, timings + calls);
    printf("MN_TIMING {\"calls\":%u,\"p50_us\":%lld,\"p95_us\":%lld,\"max_us\":%lld}\n",
        calls, (long long)(calls ? timings[calls / 2] : 0),
        (long long)(calls ? timings[calls * 95 / 100] : 0), (long long)(calls ? timings[calls - 1] : 0));
    free(timings);
    free(buffer);
    iface->destroy(detector);
    ESP_ERROR_CHECK(esp_vfs_unregister("/mnprobe"));
    srmodel_host_deinit(models);
    free(model_root);
    free(fixtures);
    return passed;
}

extern "C" void app_main() {
    const bool ok = run();
    printf("MN_DONE {\"completed\":%s,\"cases\":%u}\n", ok ? "true" : "false", fixture_count * command_arm_count);
    fflush(stdout);
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}
