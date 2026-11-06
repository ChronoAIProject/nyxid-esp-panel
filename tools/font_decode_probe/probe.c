/* SPDX-License-Identifier: Apache-2.0 */
// Exercise the real pinned font decoder, independently of panel/DSI hardware.
// --concurrent deliberately reproduces unsupported shared-RLE-state access.
#include "lvgl.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

LV_FONT_DECLARE(lv_font_noto_sans_sc_med_20);
enum { GLYPHS = 70, ITERATIONS = 3000 };
static lv_font_glyph_dsc_t glyphs[GLYPHS];
static unsigned char expected[GLYPHS][4096];
static size_t sizes[GLYPHS];
static atomic_int mismatches, ready;
static atomic_bool start;
static pthread_mutex_t decode_lock = PTHREAD_MUTEX_INITIALIZER;
static bool serialize = true;

static void *worker(void *arg)
{
    unsigned offset = (unsigned)(size_t)arg;
    lv_draw_buf_t *buf = lv_draw_buf_create(64, 64, LV_COLOR_FORMAT_A8, LV_STRIDE_AUTO);
    atomic_fetch_add(&ready, 1);
    while(!atomic_load(&start)) sched_yield();
    for(unsigned j = 0; j < ITERATIONS; j++) {
        unsigned i = (j + offset) % GLYPHS;
        // Each thread owns its descriptor and output. Only the font decoder's
        // internal state is shared, matching two software render workers.
        lv_font_glyph_dsc_t glyph = glyphs[i];
        if(serialize) pthread_mutex_lock(&decode_lock);
        lv_font_get_glyph_bitmap(&glyph, buf);
        if(serialize) pthread_mutex_unlock(&decode_lock);
        if(memcmp(expected[i], buf->data, sizes[i])) atomic_fetch_add(&mismatches, 1);
    }
    lv_draw_buf_destroy(buf);
    return NULL;
}

int main(int argc, char **argv)
{
    if(argc == 2 && !strcmp(argv[1], "--concurrent")) serialize = false;
    else if(argc != 1) {
        fprintf(stderr, "Usage: %s [--concurrent]\n", argv[0]);
        return 2;
    }
    const uint32_t cjk[] = {0x56fe, 0x7247, 0x5168, 0x5c4f, 0x6587, 0x5b57, 0x63cf, 0x8fb9,
                            0x6b22, 0x8fce, 0x56de, 0x5bb6, 0x6d4b, 0x8bd5, 0x76f8, 0x673a,
                            0x55ef, 0x5443, 0x5662, 0x6b38, 0x8bf6, 0x55e8};
    lv_init();
    lv_draw_buf_t *buf = lv_draw_buf_create(64, 64, LV_COLOR_FORMAT_A8, LV_STRIDE_AUTO);
    for(unsigned i = 0; i < GLYPHS; i++) {
        uint32_t codepoint = i < 48 ? 32 + i : cjk[i - 48];
        if(!lv_font_get_glyph_dsc(&lv_font_noto_sans_sc_med_20, &glyphs[i], codepoint, 0)) return 2;
        sizes[i] = lv_draw_buf_width_to_stride(glyphs[i].box_w, LV_COLOR_FORMAT_A8) * glyphs[i].box_h;
        lv_font_get_glyph_bitmap(&glyphs[i], buf);
        memcpy(expected[i], buf->data, sizes[i]);
    }
    lv_draw_buf_destroy(buf);
    pthread_t threads[2];
    for(unsigned i = 0; i < 2; i++)
        if(pthread_create(&threads[i], NULL, worker, (void *)(size_t)(i * 17))) return 2;
    while(atomic_load(&ready) != 2) sched_yield();
    atomic_store(&start, true);
    for(unsigned i = 0; i < 2; i++) pthread_join(threads[i], NULL);
    printf("serialized=%d mismatches=%d / %d glyph decodes\n",
           serialize, atomic_load(&mismatches), 2 * ITERATIONS);
    return atomic_load(&mismatches) ? 1 : 0;
}
