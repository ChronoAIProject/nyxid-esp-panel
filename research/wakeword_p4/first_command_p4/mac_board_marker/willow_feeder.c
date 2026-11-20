// Exercise the panel's exact warmup and feeding functions with archived
// WakeNet returns. Each callback also compares all 512 input samples.
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WAKENET_CHANNEL_VERIFIED=-1, WAKENET_DETECTED=1 };
typedef struct {
    int16_t *expected;
    uint8_t *returns;
    size_t call, count;
} model_iface_data_t;
typedef struct { int (*detect)(model_iface_data_t *, int16_t *); } esp_wn_iface_t;
static atomic_uint wake_processed_frames;
#include "willow_functions.inc"

static int detect(model_iface_data_t *m, int16_t *pcm)
{
    assert(m->call < m->count);
    assert(!memcmp(pcm, m->expected + WILLOW_CHUNK*m->call, WILLOW_CHUNK*sizeof(*pcm)));
    return (int)m->returns[m->call++] - 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    const size_t samples=strtoul(argv[1], NULL, 10), read_size=strtoul(argv[2], NULL, 10);
    const size_t total=WILLOW_PRIME_SAMPLES+samples, calls=total/WILLOW_CHUNK;
    assert(total%WILLOW_CHUNK == 0 && read_size > 0);
    int16_t *expected=calloc(total,sizeof(*expected));
    uint8_t *returns=malloc(calls);
    assert(expected && returns);
    assert(fread(expected+WILLOW_PRIME_SAMPLES,sizeof(*expected),samples,stdin) == samples);
    assert(fread(returns,1,calls,stdin) == calls);
    model_iface_data_t model={.expected=expected,.returns=returns,.count=calls};
    const esp_wn_iface_t wn={.detect=detect};
    int16_t scratch[WILLOW_CHUNK];
    bool dirty=false;
    assert(prime_wake_model(&wn,&model,scratch,WILLOW_CHUNK));
    size_t used=WILLOW_PRIME_SAMPLES%WILLOW_CHUNK;
    memset(scratch,0,used*sizeof(*scratch));
    for(size_t begin=0;begin<samples;begin+=read_size) {
        size_t count=samples-begin < read_size ? samples-begin : read_size;
        int first=feed_wake_model(&wn,&model,scratch,&used,WILLOW_CHUNK,
                                  expected+WILLOW_PRIME_SAMPLES+begin,count,&dirty);
        assert(first>=0);
        if(first)printf("%zu %zu\n",begin+(size_t)first, begin+count);
    }
    assert(dirty && used==0 && model.call==calls);
    free(returns);
    free(expected);
}
