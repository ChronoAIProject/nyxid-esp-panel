// Separate Muse initialization/zero-input resource witness. No audio I/O or NVS.
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_wn_models.h"
#include "model_path.h"
#include "zlib.h"

#define MODEL "wn10_himuse"
#define PRIME_CALLS 63
#define MAX_WORDS 8
extern const uint8_t packed_start[] asm("_binary_muse_model_bin_start");
extern const uint8_t packed_end[] asm("_binary_muse_model_bin_end");

typedef struct {size_t internal_free, psram_free, internal_largest, psram_largest;} heap_t;
static heap_t heap(void) {
    return (heap_t){heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT),
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
        heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT)};
}
static void print_heap(const char *key, heap_t value) {
    printf(",\"%s\":{\"internal_free\":%u,\"psram_free\":%u,\"internal_largest\":%u,\"psram_largest\":%u}",
        key,(unsigned)value.internal_free,(unsigned)value.psram_free,
        (unsigned)value.internal_largest,(unsigned)value.psram_largest);
}
static uint32_t bits(float value) {uint32_t out;memcpy(&out,&value,sizeof(out));return out;}
static void quoted(const char *value) {
    putchar('"');
    for (const unsigned char *p=(const unsigned char *)value;*p;p++) {
        if (*p=='"' || *p=='\\') putchar('\\');
        if (*p<32 || *p>=127) printf("\\u%04x",*p);else putchar(*p);
    }
    putchar('"');
}
static void returns(const char *key, const int *values, int count) {
    printf(",\"%s\":[",key);
    for(int i=0;i<count;i++)printf("%s%d",i?",":"",values[i]);
    putchar(']');
}

void app_main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    uint32_t size=(uint32_t)packed_start[0]|(uint32_t)packed_start[1]<<8|
        (uint32_t)packed_start[2]<<16|(uint32_t)packed_start[3]<<24;
    if(size<1024 || size>1024*1024)abort();
    heap_t before_load=heap();
    void *root=heap_caps_malloc(size,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    uLongf expanded=size;
    if(!root || uncompress(root,&expanded,packed_start+4,packed_end-packed_start-4)!=Z_OK || expanded!=size)abort();
    srmodel_list_t *models=srmodel_load(root);
    if(!models || models->num!=1)abort();
    const char *name=esp_srmodel_filter(models,ESP_WN_PREFIX,MODEL);
    if(!name || strcmp(name,MODEL))abort();
    const esp_wn_iface_t *wn=esp_wn_handle_from_name(name);
    if(!wn)abort();
    printf("MUSE_REFERENCE_READY {\"schema\":\"muse-zero-v3\",\"engine\":\"%s\",\"mode\":\"DET_MODE_95\",\"expanded_model_bytes\":%" PRIu32,name,size);
    print_heap("heap_before_load",before_load);print_heap("heap_after_load",heap());printf("}\n");
    int expected_chunk=0;
    for(int id=0;id<5;id++) {
        heap_t before_create=heap();int64_t start=esp_timer_get_time();
        model_iface_data_t *model=wn->create(name,DET_MODE_95);
        int64_t create_us=esp_timer_get_time()-start;
        if(!model)abort();
        heap_t after_create=heap();
        int chunk=wn->get_samp_chunksize(model),rate=wn->get_samp_rate(model);
        int channels=wn->get_channel_num(model),words=wn->get_word_num(model);
        if(chunk<=1 || chunk>4096 || rate!=16000 || channels!=1 || words<1 || words>MAX_WORDS)abort();
        if(id && chunk!=expected_chunk)abort();
        expected_chunk=chunk;
        int samples=id==0?1:id==1?chunk-1:id==2?chunk+1:id==3?16000:16001;
        if(samples<1)abort();
        int calls=(samples+chunk-1)/chunk;
        int16_t *zero=heap_caps_calloc(chunk,sizeof(int16_t),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        int *raw=heap_caps_malloc(calls*sizeof(int),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
        if(!zero || !raw)abort();
        uint32_t before_bits[MAX_WORDS],after_bits[MAX_WORDS];int prime[PRIME_CALLS];
        for(int w=0;w<words;w++) {
            float value=wn->get_det_threshold(model,w+1);
            if(!isfinite(value) || value<=0 || value>=1)abort();
            before_bits[w]=bits(value);
        }
        start=esp_timer_get_time();
        for(int i=0;i<PRIME_CALLS;i++)prime[i]=(int)wn->detect(model,zero);
        int64_t prime_us=esp_timer_get_time()-start;
        heap_t after_prime=heap();
        for(int w=0;w<words;w++)after_bits[w]=bits(wn->get_det_threshold(model,w+1));
        uint32_t crc=crc32(0,NULL,0);int64_t total_us=0,max_us=0;
        for(int i=0;i<calls;i++) {
            int count=samples-i*chunk;if(count>chunk)count=chunk;
            crc=crc32(crc,(const uint8_t *)zero,count*sizeof(int16_t));
            start=esp_timer_get_time();raw[i]=(int)wn->detect(model,zero);
            int64_t elapsed=esp_timer_get_time()-start;total_us+=elapsed;if(elapsed>max_us)max_us=elapsed;
        }
        printf("MUSE_REFERENCE_CASE {\"id\":%d,\"samples\":%d,\"chunk\":%d,\"channels\":%d,\"rate\":%d,\"calls\":%d,\"crc32\":%" PRIu32 ",\"create_us\":%lld,\"prime_us\":%lld,\"detect_total_us\":%lld,\"detect_max_us\":%lld,\"words\":[",
            id,samples,chunk,channels,rate,calls,crc,(long long)create_us,(long long)prime_us,(long long)total_us,(long long)max_us);
        for(int w=0;w<words;w++) {
            const char *word_name=wn->get_word_name(model,w+1);
            if(!word_name)abort();
            printf("%s{\"index\":%d,\"name\":",w?",":"",w+1);quoted(word_name);
            printf(",\"default_bits\":\"%08" PRIx32 "\",\"after_prime_bits\":\"%08" PRIx32 "\"}",before_bits[w],after_bits[w]);
        }
        putchar(']');returns("prime_returns",prime,PRIME_CALLS);returns("input_returns",raw,calls);
        print_heap("heap_before_create",before_create);print_heap("heap_after_create",after_create);print_heap("heap_after_prime",after_prime);
        printf(",\"probe_buffer_bytes\":%u",(unsigned)(chunk*sizeof(int16_t)+calls*sizeof(int)));
        free(raw);free(zero);wn->destroy(model);print_heap("heap_after_destroy",heap());printf("}\n");
    }
    printf("MUSE_REFERENCE_DONE {\"schema\":\"muse-zero-v3\",\"cases\":5}\n");
}
