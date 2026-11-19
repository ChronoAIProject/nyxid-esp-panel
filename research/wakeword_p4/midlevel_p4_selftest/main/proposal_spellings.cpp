#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern const uint8_t model_bytes[] asm("_binary_4090_c161_proposal_letters_student_tflite_start");

namespace {
constexpr unsigned frames = 71;
constexpr unsigned alphabet = 32;
constexpr unsigned max_letters = 9;
constexpr unsigned max_words = 31;
constexpr unsigned input_bytes = 144 * 40;

struct Header {
    uint32_t magic, count, target_count, word_count, separator, max_gap;
    double old_threshold, gap_threshold;
};
struct Word {
    uint8_t length;
    uint8_t symbols[max_letters];
};
static_assert(sizeof(Header) == 40 && sizeof(Word) == 10);
struct Path {
    int value;
    int gap;
};
struct Winner {
    unsigned index;
    Path path;
    float score;
};

alignas(16) uint8_t arena[72 * 1024];
int16_t deficits[frames][alphabet];
Word words[max_words];
int8_t reference[frames][alphabet];
tflite::MicroMutableOpResolver<3> resolver;

Path match(const Word& word) {
    // Alternate blank and character states. Descending updates leave predecessor
    // states on the previous frame. Resolve ties in stay, advance, skip order.
    Path state[2 * max_letters];
    std::fill_n(state, 2 * word.length, Path{-65536, 0});
    Path best{-65536, 0};
    for (unsigned t = 0; t < frames; ++t) {
        for (int i = 2 * word.length - 1; i >= 1; --i) {
            const int symbol = (i % 2) ? word.symbols[i / 2] : 0;
            Path source = state[i];
            if (state[i - 1].value > source.value) source = state[i - 1];
            if (symbol && i >= 3 && symbol != word.symbols[i / 2 - 1] &&
                state[i - 2].value > source.value) source = state[i - 2];
            if (i == 1 && source.value < 0) source = {0, 0};
            state[i] = {source.value + deficits[t][symbol], source.gap + (i == 6)};
        }
        const auto last = state[2 * word.length - 1];
        if (last.value > best.value) best = last;
    }
    return best;
}

void decode(const int8_t* codes, const Header& header, Winner& target, Winner& rival) {
    for (unsigned t = 0; t < frames; ++t) {
        const auto* row = codes + t * alphabet;
        const int peak = *std::max_element(row, row + alphabet);
        for (unsigned c = 0; c < alphabet; ++c) deficits[t][c] = row[c] - peak;
        deficits[t][0] = std::max(deficits[t][0], deficits[t][header.separator]);
    }
    for (unsigned i = 0; i < header.word_count; ++i) {
        const Path path = match(words[i]);
        const float value = static_cast<float>(path.value) / words[i].length;
        auto& winner = (i < header.target_count) ? target : rival;
        if (value > winner.score) winner = {i, path, value};
    }
}

bool replay() {
    uint32_t offset = 2 << 20;
    auto read = [&offset](void* dst, size_t size) {
        const bool ok = esp_flash_read(nullptr, dst, offset, size) == ESP_OK;
        offset += size;
        return ok;
    };
    Header h;
    if (!read(&h, sizeof(h)) || h.magic != 0x53504c31 || !h.count || h.count > 200 ||
        !h.target_count || h.target_count >= h.word_count || h.word_count > max_words ||
        h.separator >= alphabet || !read(words, h.word_count * sizeof(Word))) return false;
    for (unsigned i = 0; i < h.word_count; ++i) {
        const auto& w = words[i];
        if (!w.length || w.length > max_letters) return false;
        for (unsigned c = 0; c < w.length; ++c)
            if (!w.symbols[c] || w.symbols[c] >= alphabet) return false;
    }
    resolver.AddPad();
    resolver.AddConv2D();
    resolver.AddReshape();
    const auto start = esp_timer_get_time();
    tflite::MicroInterpreter vm(tflite::GetModel(model_bytes), resolver, arena, sizeof(arena));
    if (vm.AllocateTensors() != kTfLiteOk) return false;
    auto* in = vm.input(0);
    auto* out = vm.output(0);
    if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8 ||
        in->bytes != input_bytes || out->bytes != (frames + 1) * alphabet) return false;
    std::printf("SPELL_READY %u %lld\n", static_cast<unsigned>(vm.arena_used_bytes()), esp_timer_get_time()-start);
    for (unsigned i = 0; i < h.count; ++i) {
        if (!read(in->data.int8, input_bytes) || !read(reference, sizeof(reference))) return false;
        const auto begin = esp_timer_get_time();
        if (vm.Invoke() != kTfLiteOk) return false;
        const auto after_invoke = esp_timer_get_time();
        Winner target{0, {}, -65536}, rival{0, {}, -65536};
        decode(out->data.int8, h, target, rival);
        const double score = 2.0 * target.score - rival.score;
        const bool old = score >= h.old_threshold;
        const bool gap = score >= h.gap_threshold && target.path.gap <= static_cast<int>(h.max_gap);
        const auto after_decode = esp_timer_get_time();
        unsigned mismatches = 0;
        for (unsigned j = 0; j < sizeof(reference); ++j)
            mismatches += out->data.int8[j] != reinterpret_cast<const int8_t*>(reference)[j];
        std::printf("SPELL_ROW %u %u %u %d %u %d %d %d %d %lld %lld\n", i, mismatches,
                    target.index, target.path.value, rival.index, rival.path.value, target.path.gap,
                    old, gap, after_invoke-begin, after_decode-after_invoke);
        vTaskDelay(1);
    }
    std::printf("SPELL_END %u %u\n", static_cast<unsigned>(h.count),
                static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)));
    return true;
}
}

extern "C" void app_main() {
    if (!replay()) std::puts("SPELL_FAIL");
    std::puts("SPELL_DONE");
}
