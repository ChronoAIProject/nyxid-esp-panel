/* Runtime compatibility only: zero PCM is not recognition-quality evidence. */
#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "esp_err.h"

volatile unsigned nyx_wakenet_stage;
void app_main(void) {
    nyx_wakenet_stage = 1;
    printf("NYX_WN_BEGIN\n");
    srmodel_list_t *models = esp_srmodel_init("model");
    if (!models) { printf("NYX_WN_ERROR model partition\n"); return; }
    nyx_wakenet_stage = 2;
    char *name = esp_srmodel_filter(models, "wn9_", "hiesp");
    if (!name) { printf("NYX_WN_ERROR missing Hi ESP\n"); return; }
    const esp_wn_iface_t *iface = esp_wn_handle_from_name(name);
    if (!iface) { printf("NYX_WN_ERROR missing interface\n"); return; }
    printf("NYX_WN_MODEL %s\n", name);
    model_iface_data_t *data = iface->create(name, DET_MODE_95);
    if (!data) { printf("NYX_WN_ERROR create\n"); return; }
    nyx_wakenet_stage = 3;
    int chunk = iface->get_samp_chunksize(data);
    int channels = iface->get_channel_num(data);
    int rate = iface->get_samp_rate(data);
    printf("NYX_WN_CREATED chunk=%d channels=%d rate=%d threshold=%f\n",
           chunk, channels, rate, iface->get_det_threshold(data, 1));
    if (chunk <= 0 || chunk > 16000 || channels != 1 || rate != 16000) return;
    int16_t *pcm = calloc(chunk, sizeof(int16_t));
    if (!pcm) { printf("NYX_WN_ERROR PCM allocation\n"); return; }
    int64_t start = esp_timer_get_time();
    for (int i = 0; i < 100; i++) {
        int result = iface->detect(data, pcm);
        printf("NYX_WN_ZERO_CHUNK i=%d result=%d\n", i, result);
        nyx_wakenet_stage = 4 + i;
        vTaskDelay(1);
    }
    printf("NYX_WN_ZERO_DONE elapsed_us=%lld\n", (long long)(esp_timer_get_time()-start));
    free(pcm);
    iface->destroy(data);
    esp_srmodel_deinit(models);
    nyx_wakenet_stage = 104;
}
