#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "pocketsphinx.h"

enum { block = 480, capacity = 28800, max_matches = 64 };
static const uint32_t external = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
static const uint32_t internal = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

static void report_heap(const char *stage)
{
    printf("LEX_HEAP stage=%s ext=%u int=%u min_ext=%u min_int=%u\n", stage,
           (unsigned)heap_caps_get_free_size(external), (unsigned)heap_caps_get_free_size(internal),
           (unsigned)heap_caps_get_minimum_free_size(external), (unsigned)heap_caps_get_minimum_free_size(internal));
}

struct match { int start, end, returned; };

static int run_case(ps_decoder_t *ps, const char *cmn, int idx, const int16_t *pcm, int start, int end, int proposal)
{
    struct match seen[max_matches];
    int count = 0, confirmed = -1;
    int64_t before = esp_timer_get_time();
    if (ps_reinit_feat(ps, NULL) < 0 || ps_set_cmn(ps, cmn) < 0 || ps_start_utt(ps) < 0) return -1;
    int64_t compute = esp_timer_get_time() - before;
    int64_t finish = compute, history = -1, confirm_us = -1, max_chunk = 0;
    for (int offset = 0; offset < end - start; offset += block) {
        int n = end - start - offset;
        if (n > block) n = block;
        int delivered = start + offset + n;
        int previous = count;
        before = esp_timer_get_time();
        if (ps_process_raw(ps, pcm + offset, n, FALSE, FALSE) < 0) return -1;
        for (ps_seg_t *s = ps_seg_iter(ps); s; s = ps_seg_next(s)) {
            int first, last;
            ps_seg_frames(s, &first, &last);
            int begin = start + first * 160, stop = start + (last + 1) * 160;
            int j = 0;
            while (j < count && (seen[j].start != begin || seen[j].end != stop)) ++j;
            if (j < count) continue;
            if (count == max_matches || strncmp(ps_seg_word(s), "hey nicks", 9)) return -1;
            seen[count++] = (struct match){begin, stop, delivered};
        }
        int64_t cost = esp_timer_get_time() - before;
        if (cost > max_chunk) max_chunk = cost;
        compute += cost;
        // The proposal starts a worker. Historical chunks are already available;
        // each future chunk becomes readable at its own final sample.
        int64_t available = (int64_t)(delivered - proposal) * 1000000 / 16000;
        if (available > finish) finish = available;
        finish += cost;
        if (delivered <= proposal) history = compute;
        for (int j = previous; j < count && confirmed < 0; ++j) {
            if (seen[j].end >= proposal - 7200 && seen[j].end <= end) {
                confirmed = delivered < proposal ? proposal : delivered;
                confirm_us = finish;
            }
        }
    }
    if (ps_end_utt(ps) < 0) return -1;
    for (int i = 0; i < count; ++i)
        printf("LEX_MATCH id=%d start=%d end=%d returned=%d\n", idx, seen[i].start, seen[i].end, seen[i].returned);
    printf("LEX_CASE id=%d count=%d confirmed=%d history_us=%lld compute_us=%lld confirm_us=%lld max_chunk_us=%lld\n",
           idx, count, confirmed, (long long)history, (long long)compute, (long long)confirm_us, (long long)max_chunk);
    return 0;
}

static int replay(void)
{
#ifdef CONFIG_NYX_LEXICAL_QEMU_RAM
    // The emulator exposes RAM directly. Opt in only in the QEMU configuration.
    const uint32_t caps[] = {MALLOC_CAP_SPIRAM | MALLOC_CAP_DEFAULT | MALLOC_CAP_8BIT | MALLOC_CAP_32BIT, 0, 0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps, 0x48000000, 0x49000000));
    heap_caps_malloc_extmem_enable(0);
#endif
    esp_vfs_fat_mount_config_t mount = {.max_files = 8};
    ESP_ERROR_CHECK(esp_vfs_fat_spiflash_mount_ro("/vol", "storage", &mount));
    FILE *input = fopen("/vol/cases.bin", "rb");
    int16_t *pcm = malloc(capacity * sizeof(*pcm));
    if (!input || !pcm) return -1;
    report_heap("before");
    ps_config_t *config = ps_config_init(NULL);
    if (!config) return -1;
    ps_config_set_str(config, "hmm", "/vol/am");
    ps_config_set_str(config, "dict", "/vol/two.dict");
    ps_config_set_str(config, "keyphrase", "hey nicks");
    ps_config_set_str(config, "loglevel", "WARN");
    ps_config_set_float(config, "kws_threshold", 1e-15);
    ps_config_set_bool(config, "mmap", FALSE);
    int64_t before = esp_timer_get_time();
    ps_decoder_t *ps = ps_init(config);
    if (!ps) return -1;
    printf("LEX_INIT us=%lld\n", (long long)(esp_timer_get_time() - before));
    const char *initial = ps_get_cmn(ps, FALSE);
    if (!initial) return -1;
    char *cmn = strdup(initial);
    if (!cmn) return -1;
    report_heap("ready");
    uint32_t hdr[3];
    int idx = 0;
    while (fread(hdr, 1, sizeof(hdr), input) == sizeof(hdr)) {
        if (hdr[1] < hdr[2] || hdr[2] < hdr[0] || hdr[1] - hdr[0] > capacity) return -1;
        int samples = hdr[1] - hdr[0];
        if (fread(pcm, sizeof(*pcm), samples, input) != samples) return -1;
        if (run_case(ps, cmn, idx, pcm, hdr[0], hdr[1], hdr[2]) < 0) return -1;
        ++idx;
        vTaskDelay(1);
    }
    if (ferror(input) || idx != 62) return -1;
    report_heap("decoded");
    ps_free(ps);
    ps_config_free(config);
    free(cmn);
    fclose(input);
    free(pcm);
    return idx;
}

static void worker(void *arg)
{
    (void)arg;
    int count = replay();
    printf("LEX_STACK remaining=%u\n", (unsigned)uxTaskGetStackHighWaterMark(NULL));
    printf("LEX_DONE cases=%d\n", count);
    vTaskDelete(NULL);
}

void app_main(void)
{
    if (xTaskCreatePinnedToCore(worker, "lexical", 128 * 1024, NULL, 3, NULL, 0) != pdPASS)
        puts("LEX_DONE cases=-1");
}
