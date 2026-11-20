// Digital listener heap and timing. Both modes replay the same interleaved M/R.
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "fixed_marker/board_marker.h"
#ifndef NYX_PANEL_FIXED_SOUND
#include "panel_wake_runtime.h"
static atomic_uint wake_processed_frames;
#include "panel_feeders.inc"
#endif

#ifdef NYX_PROFILE_RESIDENT_MONO
#include "esp_rom_crc.h"
// The QEMU loader places a count, CRC32 and mono PCM here, outside the fixture heap.
static const uint32_t *const recording = (const uint32_t *)0x48400000;
#endif

static void memory(const char *stage, unsigned epoch)
{
    printf("PROFILE heap %u %s %u %u\n",epoch,stage,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
}

static void run_epoch(unsigned count, unsigned epoch)
{
    int16_t mr[512];
    memory("before",epoch);
    int64_t start=esp_timer_get_time();
    void *storage=heap_caps_aligned_alloc(8,board_marker_bytes(),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    if (!storage) abort();
    board_marker_t *marker=board_marker_begin(storage);
#ifndef NYX_PANEL_FIXED_SOUND
    int16_t scratch[480]={0},mic[256];
    const esp_wn_iface_t *wn=&panel_wake_runtime_iface;
    UBaseType_t saved=portSET_INTERRUPT_MASK_FROM_ISR();
    asm volatile("fmv.w.x ft0, zero" ::: "ft0");
    model_iface_data_t *model=wn->create("hey_nyx",DET_MODE_95);
    portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
    if(!model || wn->get_samp_chunksize(model)!=480)abort();
    size_t used=0;
    bool dirty=false;
    for(int n=0;n<panel_wake_runtime_prime_samples();n+=480) {
        saved=portSET_INTERRUPT_MASK_FROM_ISR();
        asm volatile("fmv.w.x ft0, zero" ::: "ft0");
        int hit=wn->detect(model,scratch);
        portCLEAR_INTERRUPT_MASK_FROM_ISR(saved);
        if(hit<0)abort();
    }
#endif
    int64_t ready_us=esp_timer_get_time()-start;
    memory("ready",epoch);
    unsigned max_us=0,over_16ms=0;
    uint64_t total_us=0;
    for(unsigned at=0;at<count;at+=256) {
#ifdef NYX_PROFILE_RESIDENT_MONO
        const int16_t *pcm = (const int16_t *)(recording + 2) + at;
        for (unsigned j = 0; j < 256; ++j) {
            mr[2*j] = pcm[j];
            mr[2*j+1] = 0;
        }
#else
        ESP_ERROR_CHECK(esp_flash_read(NULL,mr,(2u<<20)+8+4*at,sizeof(mr)));
#endif
        // Guard the QEMU floating point context for both paths.
        UBaseType_t mask=portSET_INTERRUPT_MASK_FROM_ISR();
        asm volatile("fmv.w.x ft0, zero" ::: "ft0");
        start=esp_timer_get_time();
#ifdef NYX_PANEL_FIXED_SOUND
        uint64_t cue=board_marker_feed_fixed(marker,mr,256,at);
        uint64_t detected_model=0;
#else
        for(unsigned j=0;j<256;++j)mic[j]=mr[j*2];
        marker_decision_t decision={0};
        int hit=feed_marked_read(wn,model,scratch,&used,480,mic,mr,256,&dirty,marker,at,&decision);
        if(hit<0)abort();
        uint64_t cue=decision.cue_end,detected_model=decision.model_end;
#endif
        unsigned elapsed=(unsigned)(esp_timer_get_time()-start);
        portCLEAR_INTERRUPT_MASK_FROM_ISR(mask);
        total_us+=elapsed;
        if(elapsed>max_us)max_us=elapsed;
        if(elapsed>=16000)++over_16ms;
        if(cue || detected_model)
            printf("PROFILE event %u %llu %llu %u\n",epoch,(unsigned long long)detected_model,
                   (unsigned long long)cue,at+256);
        if(at/256%32==31)vTaskDelay(1);
    }
#ifndef NYX_PANEL_FIXED_SOUND
    wn->destroy(model);
#endif
    heap_caps_free(marker);
    memory("released",epoch);
    printf("PROFILE time %u %u %u %lld %llu %u %u\n",epoch,count,(unsigned)board_marker_bytes(),
           (long long)ready_us,(unsigned long long)total_us,max_us,over_16ms);
}

static void replay(void *arg)
{
#ifdef NYX_PROFILE_RESIDENT_MONO
    unsigned count = recording[0];
    if (!count || count > 6*1024*1024-256) abort();
    uint32_t crc = esp_rom_crc32_le(0,(const uint8_t *)(recording+2),count*2);
    printf("PROFILE input %u %08lx\n",count,(unsigned long)crc);
    if (crc != recording[1]) abort();
    run_epoch((count+255u)&~255u,0);
#else
    uint32_t header[2];
    ESP_ERROR_CHECK(esp_flash_read(NULL,header,2u<<20,sizeof(header)));
    if(!header[0] || header[0]%256 || header[0]>160000 || !header[1] || header[1]>2)abort();
    for(unsigned epoch=0;epoch<header[1];++epoch) run_epoch(header[0],epoch);
#endif
    puts("BOARD PASS");
    vTaskDelete(NULL);
}

void __wrap_app_main(void)
{
    const uint32_t caps[]={MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT|MALLOC_CAP_32BIT,0,0};
#ifdef NYX_PROFILE_RESIDENT_MONO
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps,0x48000000,(intptr_t)recording));
#else
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps,0x48000000,0x49000000));
#endif
    if(xTaskCreate(replay,"profile",48*1024,NULL,3,NULL)!=pdPASS)abort();
}

// IDF also roots the unwrapped app_main. Bind it to this same fixture so the
// simulator's original listener and its embedded model can be garbage-collected.
void app_main(void) __attribute__((alias("__wrap_app_main")));
