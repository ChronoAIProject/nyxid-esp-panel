// Standalone offline PCM benchmark. No microphone, Wi-Fi, NVS or flash writes.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"
#include "zlib.h"

#if BASELINE_THRESHOLD_CONTROL
#include <float.h>
#include <math.h>
#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "Threshold control requires IEEE binary32 float"
#endif
_Static_assert(sizeof(float)==sizeof(uint32_t), "Threshold control needs 32-bit float");

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return bits;
}

typedef struct {
    uint32_t sample_end;
    int32_t result;
} baseline_event_t;
#endif

extern const uint8_t packed_start[] asm(BASELINE_PACKED_START);
extern const uint8_t packed_end[] asm(BASELINE_PACKED_END);
extern const esp_wn_iface_t esp_sr_wakenet9_quantized;

#if BASELINE_REPORT_RESOURCES
typedef struct {
    size_t internal_free, psram_free, internal_largest, psram_largest;
} heap_snapshot_t;

static heap_snapshot_t heap_snapshot(void) {
    return (heap_snapshot_t){
        .internal_free=heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
        .psram_free=heap_caps_get_free_size(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT),
        .internal_largest=heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
        .psram_largest=heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT)};
}

static void print_heap(const char *key, heap_snapshot_t value) {
    printf(",\"%s\":{\"internal_free\":%u,\"psram_free\":%u,\"internal_largest\":%u,\"psram_largest\":%u}",
        key,(unsigned)value.internal_free,(unsigned)value.psram_free,
        (unsigned)value.internal_largest,(unsigned)value.psram_largest);
}
#endif

static bool read_exact(void *buffer, size_t bytes) {
    uint8_t *out=buffer;
    while(bytes) {
        int got=uart_read_bytes(UART_NUM_0,out,bytes,pdMS_TO_TICKS(15000));
        if(got<=0)return false;
        out+=got;bytes-=got;
    }
    return true;
}

void app_main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    uint32_t size=(uint32_t)packed_start[0] | (uint32_t)packed_start[1]<<8 |
        (uint32_t)packed_start[2]<<16 | (uint32_t)packed_start[3]<<24;
    if(size<1024 || size>512*1024)abort();
#if BASELINE_REPORT_RESOURCES
    heap_snapshot_t heap_before_model=heap_snapshot();
#endif
    void *root=heap_caps_malloc(size,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    uLongf expanded=size;
    if(!root || uncompress(root,&expanded,packed_start+4,packed_end-packed_start-4)!=Z_OK || expanded!=size)abort();
    srmodel_list_t *models=srmodel_load(root);
    if(!models || models->num!=1)abort();
    const char *name=esp_srmodel_filter(models,ESP_WN_PREFIX,BASELINE_MODEL_NAME);
    if(!name || strcmp(name,BASELINE_MODEL_NAME)!=0)abort();
#if BASELINE_USE_MODEL_FACTORY
    // Preserve the official model name; metadata labels do not select an ABI.
    const esp_wn_iface_t *wn=esp_wn_handle_from_name(name);
#else
    // Keep the original Hi ESP interface and lifecycle for historical replay.
    const esp_wn_iface_t *wn=&esp_sr_wakenet9_quantized;
#endif
    if(!wn)abort();
#if BASELINE_REPORT_RESOURCES
    heap_snapshot_t heap_after_model=heap_snapshot();
#endif
    uart_config_t config={.baud_rate=115200,.data_bits=UART_DATA_8_BITS,
        .parity=UART_PARITY_DISABLE,.stop_bits=UART_STOP_BITS_1,
        .flow_ctrl=UART_HW_FLOWCTRL_DISABLE,.source_clk=UART_SCLK_DEFAULT};
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0,&config));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0,64*1024,0,0,NULL,0));
    printf("WAKE_BENCH_READY {\"engine\":\"%s\",\"mode\":\"DET_MODE_95\",\"rate\":16000",name);
#if BASELINE_THRESHOLD_CONTROL
    printf(",\"control_protocol\":\"official-threshold-v1\"");
#endif
#if BASELINE_REPORT_RESOURCES
    printf(",\"resource_probe\":true,\"expanded_model_bytes\":%" PRIu32,size);
    print_heap("heap_before_model",heap_before_model);
    print_heap("heap_after_model",heap_after_model);
#endif
    printf("}\n");
    char command[96];
    for(;;) {
        size_t used=0;
        while(used<sizeof(command)-1) {
            char c;
            if(!read_exact(&c,1))continue;
            if(c=='\n')break;
            command[used++]=c;
        }
        command[used]=0;
        unsigned id=0,samples=0;
        unsigned baud=0;
        if(sscanf(command,"BAUD %u",&baud)==1 && (baud==115200 || baud==921600 || baud==1500000 || baud==3000000)) {
            printf("WAKE_BENCH_BAUD {\"baud\":%u}\n",baud);
            ESP_ERROR_CHECK(uart_wait_tx_done(UART_NUM_0,pdMS_TO_TICKS(1000)));
            vTaskDelay(pdMS_TO_TICKS(100));
            ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0,baud));
            printf("WAKE_BENCH_READY {\"engine\":\"%s\",\"mode\":\"DET_MODE_95\",\"rate\":16000",name);
#if BASELINE_THRESHOLD_CONTROL
            printf(",\"control_protocol\":\"official-threshold-v1\"");
#endif
#if BASELINE_REPORT_RESOURCES
            printf(",\"resource_probe\":true,\"expanded_model_bytes\":%" PRIu32,size);
            print_heap("heap_before_model",heap_before_model);
            print_heap("heap_after_model",heap_after_model);
#endif
            printf("}\n");
            continue;
        }
#if BASELINE_THRESHOLD_CONTROL
        int consumed=0;
        bool setter_called=false;
        uint32_t requested_bits=0;
        float requested_threshold=0;
        bool valid=sscanf(command,"CLIP %u %u DEFAULT %n",&id,&samples,&consumed)==2
            && consumed>0 && command[consumed]==0;
        if(!valid) {
            consumed=0;
            valid=sscanf(command,"CLIP %u %u SET %8" SCNx32 " %n",&id,&samples,&requested_bits,&consumed)==3
                && consumed>0 && command[consumed]==0;
            memcpy(&requested_threshold,&requested_bits,sizeof(requested_threshold));
            valid=valid && isfinite(requested_threshold) && requested_threshold>=.4f && requested_threshold<=.9999f;
            setter_called=true;
        }
        if(!valid || !samples || samples>16000*3600) {
#else
        if(sscanf(command,"CLIP %u %u",&id,&samples)!=2 || !samples || samples>16000*3600) {
#endif
            printf("WAKE_BENCH_ERROR {\"reason\":\"invalid_command\",\"bytes\":[");
            for(size_t i=0;i<used && i<32;++i)printf("%s%u",i?",":"",(unsigned)(uint8_t)command[i]);
            printf("]}\n");continue;
        }
        // WakeNet9 clean() is unsafe before lazy queues initialize. Match the
        // production recreate/prime lifecycle, without resetting PCM mid-clip.
#if BASELINE_REPORT_RESOURCES
        heap_snapshot_t heap_before_create=heap_snapshot();
        int64_t create_start=esp_timer_get_time();
#endif
        model_iface_data_t *model=wn->create(name,DET_MODE_95);
#if BASELINE_REPORT_RESOURCES
        int64_t create_us=esp_timer_get_time()-create_start;
        heap_snapshot_t heap_after_create=heap_snapshot();
#endif
        if(!model || wn->get_samp_rate(model)!=16000 || wn->get_channel_num(model)!=1)abort();
        int chunk=wn->get_samp_chunksize(model);
        if(chunk<1 || chunk>2048)abort();
#if BASELINE_THRESHOLD_CONTROL
        float default_getter=wn->get_det_threshold(model,1);
        int setter_result=setter_called ? wn->set_det_threshold(model,requested_threshold,1) : 0;
        float getter_after_set=wn->get_det_threshold(model,1);
        if((setter_called && setter_result!=1) || !isfinite(default_getter)
                || !isfinite(getter_after_set) || default_getter<=0 || default_getter>=1
                || getter_after_set<=0 || getter_after_set>=1) {
            printf("WAKE_BENCH_ERROR {\"reason\":\"threshold_api_failed\",\"setter_result\":%d}\n",setter_result);
            abort();
        }
        unsigned prime_calls=(32000u+(unsigned)chunk-1)/(unsigned)chunk;
        int8_t *prime_returns=calloc(prime_calls,sizeof(*prime_returns));
        size_t event_capacity=((size_t)samples+(unsigned)chunk-1)/(unsigned)chunk;
        baseline_event_t *full_events=heap_caps_malloc(event_capacity*sizeof(*full_events),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        if(!prime_returns || !full_events)abort();
#endif
        int16_t *pcm=heap_caps_aligned_calloc(16,chunk,sizeof(int16_t),MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
        if(!pcm)abort();
#if BASELINE_REPORT_RESOURCES
        int64_t prime_start=esp_timer_get_time();
#endif
#if BASELINE_THRESHOLD_CONTROL
        for(unsigned call=0;call<prime_calls;++call) {
            int result=wn->detect(model,pcm);
            if(result < WAKENET_CHANNEL_VERIFIED || result > WAKENET_DETECTED)abort();
            prime_returns[call]=(int8_t)result;
        }
#else
        for(int at=0;at<32000;at+=chunk)(void)wn->detect(model,pcm);
#endif
#if BASELINE_REPORT_RESOURCES
        int64_t prime_us=esp_timer_get_time()-prime_start;
        heap_snapshot_t heap_after_prime=heap_snapshot();
#endif
#if BASELINE_THRESHOLD_CONTROL
        float getter_after_prime=wn->get_det_threshold(model,1);
        if(!isfinite(getter_after_prime) || getter_after_prime<=0 || getter_after_prime>=1)abort();
        printf("WAKE_BENCH_CLIP_READY {\"id\":%u,\"chunk\":%d,\"threshold\":%.9g",id,chunk,(double)getter_after_prime);
        printf(",\"control\":{\"schema\":\"official-threshold-v1\",\"setter_called\":%s,\"setter_result\":",setter_called?"true":"false");
        if(setter_called)printf("%d",setter_result);else printf("null");
        printf(",\"requested_f32_hex\":");
        if(setter_called)printf("\"%08" PRIx32 "\"",requested_bits);else printf("null");
        printf(",\"default_getter\":%.9g,\"default_getter_f32_hex\":\"%08" PRIx32 "\",\"getter_after_set\":%.9g,\"getter_after_set_f32_hex\":\"%08" PRIx32 "\",\"getter_after_prime_f32_hex\":\"%08" PRIx32 "\",\"prime_returns\":[",
            (double)default_getter,float_bits(default_getter),(double)getter_after_set,float_bits(getter_after_set),float_bits(getter_after_prime));
        for(unsigned i=0;i<prime_calls;++i)printf("%s%d",i?",":"",(int)prime_returns[i]);
        printf("]}");
        free(prime_returns);
#else
        printf("WAKE_BENCH_CLIP_READY {\"id\":%u,\"chunk\":%d,\"threshold\":%.6f",id,chunk,wn->get_det_threshold(model,1));
#endif
#if BASELINE_REPORT_RESOURCES
        printf(",\"create_us\":%lld,\"prime_us\":%lld",(long long)create_us,(long long)prime_us);
        print_heap("heap_before_create",heap_before_create);
        print_heap("heap_after_create",heap_after_create);
        print_heap("heap_after_prime",heap_after_prime);
#endif
        printf("}\n");
#if BASELINE_THRESHOLD_CONTROL
        unsigned count=0,calls=0,no_detect_calls=0,channel_verified_calls=0;
#else
        unsigned count=0,offsets[64]={0},calls=0;
#endif
        int64_t total_us=0,max_us=0;
        uint32_t crc=crc32(0L,Z_NULL,0);
        bool complete=true;
        for(unsigned at=0;at<samples;at+=chunk) {
            unsigned take=samples-at<(unsigned)chunk?samples-at:(unsigned)chunk;
            memset(pcm,0,chunk*sizeof(int16_t));
            if(!read_exact(pcm,take*sizeof(int16_t))){complete=false;break;}
            crc=crc32(crc,(const Bytef *)pcm,take*sizeof(int16_t));
            int64_t start=esp_timer_get_time();
            int result=wn->detect(model,pcm);
            int64_t elapsed=esp_timer_get_time()-start;
            total_us+=elapsed;if(elapsed>max_us)max_us=elapsed;++calls;
#if BASELINE_THRESHOLD_CONTROL
            if(result==WAKENET_NO_DETECT)++no_detect_calls;
            else if(result==WAKENET_CHANNEL_VERIFIED)++channel_verified_calls;
            else if(result==WAKENET_DETECTED) {
                if(count>=event_capacity)abort();
                full_events[count++]=(baseline_event_t){.sample_end=at+take,.result=result};
            } else {
                printf("WAKE_BENCH_ERROR {\"reason\":\"unexpected_detect_result\",\"result\":%d}\n",result);
                abort();
            }
#else
            if(result>0) {
                if(count<64)offsets[count]=at+take;
                ++count;
            }
#endif
        }
#if BASELINE_THRESHOLD_CONTROL
        // Emit after consuming all PCM: the host can read every event without
        // overflowing UART while it is still writing a long input stream.
        for(unsigned i=0;i<count;++i) {
            uint32_t call=(full_events[i].sample_end+(uint32_t)chunk-1)/(uint32_t)chunk;
            printf("WAKE_BENCH_EVENT {\"id\":%u,\"call\":%" PRIu32 ",\"sample_end\":%" PRIu32 ",\"chunk_end\":%" PRIu32 ",\"result\":%" PRId32 "}\n",
                id,call,full_events[i].sample_end,call*(uint32_t)chunk,full_events[i].result);
        }
#endif
        printf("WAKE_BENCH_RESULT {\"id\":%u,\"complete\":%s,\"samples\":%u,\"crc32\":%" PRIu32 ",\"events\":%u,\"calls\":%u,\"total_us\":%lld,\"max_us\":%lld,\"offsets\":[",
            id,complete?"true":"false",samples,crc,count,calls,(long long)total_us,(long long)max_us);
#if BASELINE_THRESHOLD_CONTROL
        for(unsigned i=0;i<count && i<64;++i)printf("%s%" PRIu32,i?",":"",full_events[i].sample_end);
        printf("],\"events_emitted\":%u,\"offsets_complete\":%s,\"no_detect_calls\":%u,\"channel_verified_calls\":%u}\n",
            count,count<=64?"true":"false",no_detect_calls,channel_verified_calls);
        free(full_events);
#else
        for(unsigned i=0;i<count && i<64;++i)printf("%s%u",i?",":"",offsets[i]);
        printf("]}\n");
#endif
        free(pcm);wn->destroy(model);
        if(!complete)abort(); // Do not reinterpret a truncated payload as a new command.
    }
}
