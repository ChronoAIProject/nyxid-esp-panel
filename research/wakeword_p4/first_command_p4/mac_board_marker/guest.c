// Digital fixture for the actual C161 detector, raw M/R feeder and board receiver.
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "fixed_marker/board_marker.h"
#include "panel_wake_runtime.h"

static atomic_uint wake_processed_frames;
#include "panel_feeders.inc"

static void replay(void *unused)
{
    uint32_t count;
    ESP_ERROR_CHECK(esp_flash_read(NULL, &count, 2u<<20, sizeof(count)));
    if (!count || count % 256 || count > 160000) abort();
    const uint32_t caps=MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT;
    void *storage=heap_caps_aligned_alloc(8,board_marker_bytes(),caps);
    if (!storage) abort();
    board_marker_t *marker=board_marker_begin(storage);
    const esp_wn_iface_t *wn=&panel_wake_runtime_iface;
    UBaseType_t saved=portSET_INTERRUPT_MASK_FROM_ISR();
    asm volatile("fmv.w.x ft0, zero" ::: "ft0");
    model_iface_data_t *model=wn->create("hey_nyx",DET_MODE_95);
    portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
    if (!model || wn->get_samp_chunksize(model)!=480) abort();
    int16_t scratch[480]={0}, mr[512], mic[256];
    int primed=0;
    while (primed<panel_wake_runtime_prime_samples()) {
        saved=portSET_INTERRUPT_MASK_FROM_ISR();
        asm volatile("fmv.w.x ft0, zero" ::: "ft0");
        int result=wn->detect(model,scratch);
        portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
        if (result<0) abort();
        primed+=480;
    }
    size_t used=0;
    bool dirty=false;
    printf("BOARD begin prime=%d bytes=%u count=%lu\n",primed,(unsigned)board_marker_bytes(),(unsigned long)count);
    for (unsigned at=0;at<count;at+=256) {
        ESP_ERROR_CHECK(esp_flash_read(NULL,mr,(2u<<20)+4+4*at,sizeof(mr)));
        for (unsigned i=0;i<256;i++) mic[i]=mr[2*i];
        marker_decision_t decision={0};
        saved=portSET_INTERRUPT_MASK_FROM_ISR();
        asm volatile("fmv.w.x ft0, zero" ::: "ft0");
        int result=feed_marked_read(wn,model,scratch,&used,480,mic,mr,256,&dirty,marker,at,&decision);
        portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
        if (result<0) abort();
        if (result>0) printf("BOARD decision %llu %llu %u\n",(unsigned long long)decision.model_end,
                             (unsigned long long)decision.cue_end,at+256);
        if ((at / 256 + 1) % 32 == 0) vTaskDelay(1);
    }
    wn->destroy(model);
    heap_caps_free(marker);
    puts("BOARD PASS");
    vTaskDelete(NULL);
}

void __wrap_app_main(void)
{
    const uint32_t caps[]={MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT|MALLOC_CAP_32BIT,0,0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps,0x48000000,0x49000000));
    if (xTaskCreate(replay,"board_replay",48*1024,NULL,3,NULL)!=pdPASS) abort();
}
