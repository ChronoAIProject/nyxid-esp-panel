/* QEMU-only host-file transport. Official model and detect arithmetic unchanged. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"
#include "esp_rom_crc.h"

volatile unsigned nyx_wakenet_stage;
/* QEMU Xtensa simcall ABI, NOT the incompatible OpenOCD semihost ABI. */
static int host_call(int call, uintptr_t x, uintptr_t y, uintptr_t z) {
    register int a2 asm("a2") = call;
    register uintptr_t a3 asm("a3") = x;
    register uintptr_t a4 asm("a4") = y;
    register uintptr_t a5 asm("a5") = z;
    asm volatile("simcall" : "+r"(a2), "+r"(a3) : "r"(a4), "r"(a5) : "memory");
    return a2;
}
static int read_exact(int fd, void *buffer, unsigned bytes) {
    unsigned at = 0;
    while (at < bytes) {
        int got = host_call(3, fd, (uintptr_t)((uint8_t *)buffer+at), bytes-at);
        if (got <= 0) return 0;
        at += got;
    }
    return 1;
}
void app_main(void) {
    nyx_wakenet_stage = 1;
    int fd = host_call(5, (uintptr_t)"input.pcm", 0, 0);
    if (fd < 0) { printf("NYX_PCM_ERROR open\n"); return; }
    struct { uint32_t magic, samples; float threshold; uint32_t reserved; } header;
    if (!read_exact(fd, &header, sizeof(header)) || header.magic != 0x4e595852 ||
        header.samples == 0 || header.reserved || !isfinite(header.threshold) ||
        header.threshold < .4f || header.threshold > .9999f) {
        printf("NYX_PCM_ERROR header\n"); return;
    }
    srmodel_list_t *models = esp_srmodel_init("model");
    if (!models) { printf("NYX_PCM_ERROR model partition\n"); return; }
    char *name = esp_srmodel_filter(models, "wn9_", "hiesp");
    const esp_wn_iface_t *iface = name ? esp_wn_handle_from_name(name) : NULL;
    if (!iface) { printf("NYX_PCM_ERROR interface\n"); return; }
    model_iface_data_t *data = iface->create(name, DET_MODE_95);
    if (!data) { printf("NYX_PCM_ERROR create\n"); return; }
    int chunk = iface->get_samp_chunksize(data);
    int set_result = iface->set_det_threshold(data, header.threshold, 1);
    printf("NYX_PCM_PARAMETERS chunk=%d rate=%d channels=%d set_result=%d actual=%.8f\n",
           chunk, iface->get_samp_rate(data), iface->get_channel_num(data), set_result,
           (double)iface->get_det_threshold(data, 1));
    if (chunk != 512 || iface->get_samp_rate(data) != 16000 || iface->get_channel_num(data) != 1 ||
        set_result != 1 || !isfinite(iface->get_det_threshold(data, 1)) ||
        iface->get_det_threshold(data, 1) <= 0 || iface->get_det_threshold(data, 1) >= 1) {
        printf("NYX_PCM_ERROR model parameters\n"); return;
    }
    printf("NYX_PCM_BEGIN model=%s samples=%lu chunk=%d threshold=%.8f\n",
           name, (unsigned long)header.samples, chunk, (double)header.threshold);
    int16_t *pcm = calloc(chunk, sizeof(int16_t));
    if (!pcm) { printf("NYX_PCM_ERROR allocation\n"); return; }
    uint32_t at = 0, crc = 0, calls = 0, events = 0, verified = 0;
    nyx_wakenet_stage = 3;
    while (at < header.samples) {
        uint32_t count = header.samples-at;
        if (count > chunk) count = chunk;
        memset(pcm, 0, chunk*sizeof(int16_t));
        if (!read_exact(fd, pcm, count*sizeof(int16_t))) {
            printf("NYX_PCM_ERROR truncated sample=%lu\n", (unsigned long)at); return;
        }
        crc = esp_rom_crc32_le(crc, (const uint8_t *)pcm, count*sizeof(int16_t));
        int result = iface->detect(data, pcm);
        if (result < WAKENET_CHANNEL_VERIFIED || result > WAKENET_DETECTED) { printf("NYX_PCM_ERROR detect=%d\n", result); return; }
        calls++;
        if (result == WAKENET_CHANNEL_VERIFIED) verified++;
        if (result == WAKENET_DETECTED) {
            events++;
            printf("NYX_PCM_EVENT call=%lu sample_end=%lu result=%d\n",
                   (unsigned long)calls, (unsigned long)(at+chunk), result);
        }
        at += count;
        if (!(calls % 1875)) printf("NYX_PCM_PROGRESS samples=%lu calls=%lu events=%lu\n",
            (unsigned long)at, (unsigned long)calls, (unsigned long)events);
        if (!(calls % 32)) vTaskDelay(1);
    }
    uint8_t extra;
    if (host_call(3, fd, (uintptr_t)&extra, 1) != 0) {
        printf("NYX_PCM_ERROR trailing bytes\n"); return;
    }
    if (host_call(6, fd, 0, 0) != 0) { printf("NYX_PCM_ERROR close\n"); return; }
    free(pcm); iface->destroy(data); esp_srmodel_deinit(models);
    nyx_wakenet_stage = 104;
    printf("NYX_PCM_DONE samples=%lu calls=%lu events=%lu verified=%lu crc32=%08lx\n",
           (unsigned long)at, (unsigned long)calls, (unsigned long)events, (unsigned long)verified, (unsigned long)crc);
}
