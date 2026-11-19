// Host replay of the board's C feeder. Replay each C161 decision from the
// original UART while checking the full PCM window supplied to the detector.
#include "fixed_marker/board_marker.h"
#include "product_audio_layout.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MODEL_CHUNK = 480 };
typedef struct {
    const int16_t *mic;
    const uint8_t *hits;
    size_t offset;
} model_iface_data_t;
typedef struct {
    int (*detect)(model_iface_data_t *, int16_t *);
} esp_wn_iface_t;
static atomic_uint wake_processed_frames;
#include "panel_feeders.inc"

static int detect(model_iface_data_t *state, int16_t *pcm)
{
    if (memcmp(pcm, state->mic + state->offset, MODEL_CHUNK * sizeof(*pcm))) abort();
    state->offset += MODEL_CHUNK;
    return state->hits[state->offset - 1];
}

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    const size_t count = strtoul(argv[1], NULL, 10);
    const size_t read_size = strtoul(argv[2], NULL, 10);
    const size_t tx = strtoul(argv[3], NULL, 10);
    const bool x8 = atoi(argv[4]) == 4;
    // read_size=0 calls the synchronous receiver directly, including optional TX.
    int16_t *mic = calloc(count, sizeof(*mic));
    int16_t *ref = calloc(count, sizeof(*ref));
    uint8_t *hits = calloc(count, 1);
    if (!mic || !ref || !hits || fread(mic, sizeof(*mic), count, stdin) != count ||
        fread(ref, sizeof(*ref), count, stdin) != count || fread(hits, 1, count, stdin) != count) return 2;
    void *storage = malloc(board_marker_bytes());
    if (!storage) return 2;
    memset(storage, 0xa5, board_marker_bytes());
    board_marker_t *marker = board_marker_begin(storage);
    if (!read_size) {
        for (size_t i = 0; i < count; ++i) {
            unsigned route = board_marker_step(marker, mic[i], ref[i], hits[i], i + 1 == tx);
            if (route & BOARD_MARKER_C161) printf("M %zu\n", i + 1);
            if (route & BOARD_MARKER_CUE) printf("C %zu\n", i + 1);
        }
    } else {
        const esp_wn_iface_t wn = {.detect = detect};
        model_iface_data_t model = {.mic = mic, .hits = hits};
        int16_t scratch[MODEL_CHUNK];
        size_t used = 0;
        bool dirty = false, opened = false;
        int16_t *mr = calloc(4 * read_size, sizeof(*mr));
        if (!mr || tx) return 2;
        for (size_t i = 0; i < count; i += read_size) {
            size_t n = count - i < read_size ? count - i : read_size;
            size_t stride = x8 ? 4 : 2;
            for (size_t j = 0; j < n; ++j) {
                mr[stride*j] = mic[i+j];
                mr[stride*j+1] = ref[i+j];
                if (x8) {
                    mr[stride*j+2] = 30111;
                    mr[stride*j+3] = -28111;
                }
            }
            if (x8) product_lcd_x8_normalize_capture(mr, mr, n);
            int16_t lane[1024];
            if (n > 1024) return 2;
            for (size_t j = 0; j < n; ++j) lane[j] = mr[2*j];
            marker_decision_t result = {0};
            int hit = feed_marked_read(&wn, &model, scratch, &used, MODEL_CHUNK, lane,
                                       mr, n, &dirty, marker, i, &result);
            if (hit < 0) return 2;
            if (result.model_end) printf("M %llu\n", (unsigned long long)result.model_end);
            if (result.cue_end) printf("C %llu\n", (unsigned long long)result.cue_end);
            if (hit && !opened) {
                opened = true;
                printf("O %zu\n", i+n);
            }
        }
        if (model.offset + used != count || wake_processed_frames != count / MODEL_CHUNK) return 2;
        free(mr);
    }
    free(storage);
    free(mic);
    free(ref);
    free(hits);
    return 0;
}
