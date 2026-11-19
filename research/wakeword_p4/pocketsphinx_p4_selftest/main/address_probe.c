#include "address_probe.h"

#include <string.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const unsigned char address_start[] asm("_binary_address_sendump_start");
extern const unsigned char address_end[] asm("_binary_address_sendump_end");

FILE *address_sendump_open(const char *path)
{
    if (strcmp(path, "/app/address-sendump")) return fopen(path, "rb");
    size_t size = (size_t)(address_end - address_start);
    printf("FSG_SENDUMP bytes=%u\n", (unsigned)size);
    return fmemopen((void *)address_start, size, "rb");
}

int address_cases(ps_decoder_t *decoder, const char *cmn, FILE *input, int16_t *pcm, size_t capacity)
{
    int index = 0;
    uint32_t samples;
    size_t count;
    while ((count = fread(&samples, 1, sizeof(samples), input)) == sizeof(samples)) {
        if (!samples || samples > capacity || fread(pcm, sizeof(*pcm), samples, input) != samples)
            return -1;
        int64_t before = esp_timer_get_time();
        if (ps_reinit_feat(decoder, NULL) < 0 || ps_set_cmn(decoder, cmn) < 0 ||
            ps_start_utt(decoder) < 0 || ps_process_raw(decoder, pcm, samples, FALSE, TRUE) < 0 ||
            ps_end_utt(decoder) < 0)
            return -1;
        const char *text = ps_get_hyp(decoder, NULL);
        int64_t elapsed = esp_timer_get_time() - before;
        printf("FSG_CASE id=%d samples=%u us=%lld text=%s\n", index, (unsigned)samples,
               (long long)elapsed, text ? text : "");
        for (ps_seg_t *seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)) {
            int start, end;
            ps_seg_frames(seg, &start, &end);
            printf("FSG_SEG id=%d begin=%d end=%d word=%s\n", index, start, end, ps_seg_word(seg));
        }
        ++index;
        vTaskDelay(1);
    }
    return (count || ferror(input)) ? -1 : index;
}
