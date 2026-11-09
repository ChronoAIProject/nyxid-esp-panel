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

extern const uint8_t packed_start[] asm("_binary_baseline_hi_esp_bin_start");
extern const uint8_t packed_end[] asm("_binary_baseline_hi_esp_bin_end");
extern const esp_wn_iface_t esp_sr_wakenet9_quantized;

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
    void *root=heap_caps_malloc(size,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    uLongf expanded=size;
    if(!root || uncompress(root,&expanded,packed_start+4,packed_end-packed_start-4)!=Z_OK || expanded!=size)abort();
    srmodel_list_t *models=srmodel_load(root);
    const char *name=esp_srmodel_filter(models,ESP_WN_PREFIX,"hiesp");
    if(!name)abort();
    const esp_wn_iface_t *wn=&esp_sr_wakenet9_quantized;
    uart_config_t config={.baud_rate=115200,.data_bits=UART_DATA_8_BITS,
        .parity=UART_PARITY_DISABLE,.stop_bits=UART_STOP_BITS_1,
        .flow_ctrl=UART_HW_FLOWCTRL_DISABLE,.source_clk=UART_SCLK_DEFAULT};
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0,&config));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0,64*1024,0,0,NULL,0));
    printf("WAKE_BENCH_READY {\"engine\":\"wn9_hiesp\",\"mode\":\"DET_MODE_95\",\"rate\":16000}\n");
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
        if(sscanf(command,"BAUD %u",&baud)==1 && (baud==115200 || baud==921600 || baud==1500000)) {
            printf("WAKE_BENCH_BAUD {\"baud\":%u}\n",baud);
            ESP_ERROR_CHECK(uart_wait_tx_done(UART_NUM_0,pdMS_TO_TICKS(1000)));
            vTaskDelay(pdMS_TO_TICKS(100));
            ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0,baud));
            printf("WAKE_BENCH_READY {\"engine\":\"wn9_hiesp\",\"mode\":\"DET_MODE_95\",\"rate\":16000}\n");
            continue;
        }
        if(sscanf(command,"CLIP %u %u",&id,&samples)!=2 || !samples || samples>16000*3600) {
            printf("WAKE_BENCH_ERROR {\"reason\":\"invalid_command\",\"bytes\":[");
            for(size_t i=0;i<used && i<32;++i)printf("%s%u",i?",":"",(unsigned)(uint8_t)command[i]);
            printf("]}\n");continue;
        }
        // WakeNet9 clean() is unsafe before lazy queues initialize. Match the
        // production recreate/prime lifecycle, without resetting PCM mid-clip.
        model_iface_data_t *model=wn->create(name,DET_MODE_95);
        if(!model || wn->get_samp_rate(model)!=16000 || wn->get_channel_num(model)!=1)abort();
        int chunk=wn->get_samp_chunksize(model);
        if(chunk<1 || chunk>2048)abort();
        int16_t *pcm=heap_caps_aligned_calloc(16,chunk,sizeof(int16_t),MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
        if(!pcm)abort();
        for(int at=0;at<32000;at+=chunk)(void)wn->detect(model,pcm);
        printf("WAKE_BENCH_CLIP_READY {\"id\":%u,\"chunk\":%d,\"threshold\":%.6f}\n",id,chunk,wn->get_det_threshold(model,1));
        unsigned count=0,offsets[64]={0},calls=0;
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
            if(result>0) {
                if(count<64)offsets[count]=at+take;
                ++count;
            }
        }
        printf("WAKE_BENCH_RESULT {\"id\":%u,\"complete\":%s,\"samples\":%u,\"crc32\":%" PRIu32 ",\"events\":%u,\"calls\":%u,\"total_us\":%lld,\"max_us\":%lld,\"offsets\":[",
            id,complete?"true":"false",samples,crc,count,calls,(long long)total_us,(long long)max_us);
        for(unsigned i=0;i<count && i<64;++i)printf("%s%u",i?",":"",offsets[i]);
        printf("]}\n");
        free(pcm);wn->destroy(model);
        if(!complete)abort(); // Do not reinterpret a truncated payload as a new command.
    }
}
