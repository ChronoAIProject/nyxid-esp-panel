#include "unity.h"
#include "panel_voice_audio.h"
#include <stdlib.h>
#include <limits.h>
void setUp(void) {}
void tearDown(void) {}
void test_preroll_preserves_speech_at_detection_boundary_including_wrap(void)
{
    int16_t tail[500],recording[2000],input[160],output[160];
    panel_voice_pcm_t lookback={.samples=tail,.capacity=500};
    panel_voice_pcm_t call={.samples=recording,.capacity=2000};
    for(int start=0;start<1600;start+=160) {
        for(int i=0;i<160;++i)input[i]=(int16_t)(start+i);
        panel_voice_pcm_preroll_push(&lookback,input,160);
    }
    TEST_ASSERT_EQUAL_UINT(500,lookback.count);
    size_t first=lookback.count<lookback.capacity-lookback.head?lookback.count:lookback.capacity-lookback.head;
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&call,lookback.samples+lookback.head,first));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&call,lookback.samples,lookback.count-first));
    for(int i=0;i<160;++i)input[i]=(int16_t)(1600+i);
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&call,input,160));
    for(int expected=1100;expected<1760;) {
        size_t n=1760-expected<160?(size_t)(1760-expected):160;
        TEST_ASSERT_TRUE(panel_voice_pcm_pop(&call,output,n));
        for(size_t i=0;i<n;++i)TEST_ASSERT_EQUAL_INT(expected+(int)i,output[i]);
        expected+=(int)n;
    }
    TEST_ASSERT_EQUAL_UINT(0,call.count);
}
void test_connecting_audio_and_live_audio_are_one_lossless_ordered_stream(void)
{
    // Different AEC/Opus block sizes, a connection delay, and 4x replay while
    // capture continues. The queue wraps repeatedly after catching up.
    int16_t memory[16000*3],in[160],out[320];
    panel_voice_pcm_t fifo={.samples=memory,.capacity=16000*3};
    size_t captured=0,sent=0;
    for(unsigned tick=0;tick<3000;++tick) {
        for(unsigned i=0;i<160;++i)in[i]=(int16_t)((captured+i)%32749);
        TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,160));captured+=160;
        if(tick<150 || tick%2)continue;
        for(unsigned frame=0;frame<4 && fifo.count>=320;++frame) {
            TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
            for(unsigned i=0;i<320;++i)TEST_ASSERT_EQUAL_INT((sent+i)%32749,out[i]);
            sent+=320;
        }
    }
    while(fifo.count>=320) {
        TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
        for(unsigned i=0;i<320;++i)TEST_ASSERT_EQUAL_INT((sent+i)%32749,out[i]);
        sent+=320;
    }
    TEST_ASSERT_EQUAL_UINT(captured,sent);
    TEST_ASSERT_EQUAL_UINT(0,fifo.count);
}
void test_recording_overflow_never_discards_old_speech_and_clear_removes_it(void)
{
    int16_t memory[640],in[320],out[320];
    panel_voice_pcm_t fifo={.samples=memory,.capacity=640};
    for(unsigned i=0;i<320;++i)in[i]=(int16_t)(i+1);
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_FALSE(panel_voice_pcm_push(&fifo,in,1));
    TEST_ASSERT_EQUAL_UINT(640,fifo.count);
    TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
    for(unsigned i=0;i<320;++i)TEST_ASSERT_EQUAL_INT(in[i],out[i]);
    TEST_ASSERT_FALSE(panel_voice_pcm_pop(&fifo,out,321));
    TEST_ASSERT_EQUAL_UINT(320,fifo.count);
    panel_voice_pcm_clear(&fifo);
    TEST_ASSERT_FALSE(panel_voice_pcm_pop(&fifo,out,1));
    for(unsigned i=0;i<640;++i)TEST_ASSERT_EQUAL_INT(0,memory[i]);
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
    for(unsigned i=0;i<320;++i)TEST_ASSERT_EQUAL_INT(in[i],out[i]);
}
void test_reconnect_discards_pending_and_gap_audio_but_preserves_first_connection(void)
{
    int16_t memory[640],first[320],fresh[320],out[320];
    panel_voice_pcm_t fifo={.samples=memory,.capacity=640};
    bool started=false;
    for(unsigned i=0;i<320;++i){first[i]=(int16_t)(i+1);fresh[i]=(int16_t)(i+1001);}
    // Readiness can fluctuate before the first connection, with either task
    // observing it first. The entire initial request remains ordered.
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,false,false));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,first,320));
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,false,false));
    TEST_ASSERT_EQUAL_UINT(320,fifo.count);
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,true,false));
    TEST_ASSERT_TRUE(started);
    TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
    TEST_ASSERT_EQUAL_INT16_ARRAY(first,out,320);
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,first,320));
    // A later control interruption must clear even previously queued speech.
    // Repeated gap capture must not refill the queue or overflow a long call.
    for(unsigned tick=0;tick<2000;++tick) {
        TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,false,false));
        TEST_ASSERT_EQUAL_UINT(0,fifo.count);
    }
    for(unsigned i=0;i<640;++i)TEST_ASSERT_EQUAL_INT(0,memory[i]);
    TEST_ASSERT_TRUE(started);
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,true,false));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,fresh,320));
    TEST_ASSERT_TRUE(panel_voice_pcm_pop(&fifo,out,320));
    TEST_ASSERT_EQUAL_INT16_ARRAY(fresh,out,320);
    // Repeated reconnects never return to initial-recording behavior.
    TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,false,false));
}
void test_mute_clears_initial_and_live_recordings_without_resetting_connection_history(void)
{
    int16_t memory[320],in[320]={1};
    panel_voice_pcm_t fifo={.samples=memory,.capacity=320};
    bool started=false;
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,false,true));
    TEST_ASSERT_EQUAL_UINT(0,fifo.count);
    TEST_ASSERT_FALSE(started);
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,false,false));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,true,true));
    TEST_ASSERT_TRUE(started);
    TEST_ASSERT_EQUAL_UINT(0,fifo.count);
    TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,false,false));
    TEST_ASSERT_TRUE(panel_voice_pcm_gate(&fifo,&started,true,false));
    TEST_ASSERT_TRUE(panel_voice_pcm_push(&fifo,in,320));
    TEST_ASSERT_FALSE(panel_voice_pcm_gate(&fifo,&started,true,true));
    TEST_ASSERT_EQUAL_UINT(0,fifo.count);
}
void test_playback_gain_is_symmetric_monotonic_and_bounded_at_full_scale(void)
{
    TEST_ASSERT_EQUAL_INT(0,panel_voice_amplify_sample(0,4));
    TEST_ASSERT_EQUAL_INT(4000,panel_voice_amplify_sample(1000,4));
    TEST_ASSERT_EQUAL_INT(-4000,panel_voice_amplify_sample(-1000,4));
    TEST_ASSERT_EQUAL_INT(0,panel_voice_amplify_sample(32767,0));
    int previous=-32768;
    for(int sample=-32768;sample<=32767;sample++) {
        int out=panel_voice_amplify_sample((int16_t)sample,8);
        TEST_ASSERT_TRUE(out>=previous && out>=-32767 && out<=32767);previous=out;
    }
    TEST_ASSERT_EQUAL_INT(panel_voice_amplify_sample(32767,8),panel_voice_amplify_sample(32767,UINT_MAX));
    TEST_ASSERT_TRUE(panel_voice_amplify_sample(32767,4)<32767);
    TEST_ASSERT_TRUE(panel_voice_amplify_sample(-32768,4)>-32768);
}

void test_ringback_has_bounded_volume_silent_gap_and_continuous_chunks(void)
{
    int16_t *whole=malloc(48000*sizeof(int16_t)),*chunks=malloc(48000*sizeof(int16_t));
    TEST_ASSERT_NOT_NULL(whole);TEST_ASSERT_NOT_NULL(chunks);
    uint32_t position=0;panel_voice_ringback(&position,whole,48000);
    TEST_ASSERT_EQUAL_UINT(0,position);
    unsigned audible=0;
    for(size_t i=0;i<48000;++i) {
        TEST_ASSERT_TRUE(abs(whole[i])<=3600);
        if(i>=16000)TEST_ASSERT_EQUAL_INT(0,whole[i]);
        else if(abs(whole[i])>100)++audible;
    }
    TEST_ASSERT_TRUE(audible>12000);
    TEST_ASSERT_EQUAL_INT(0,whole[0]);
    TEST_ASSERT_TRUE(abs(whole[15999])<50);
    for(size_t at=0;at<48000;) {
        size_t n=48000-at<317?48000-at:317;
        panel_voice_ringback(&position,chunks+at,n);at+=n;
    }
    for(size_t i=0;i<48000;++i)TEST_ASSERT_EQUAL_INT(whole[i],chunks[i]);
    free(whole);free(chunks);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_preroll_preserves_speech_at_detection_boundary_including_wrap);
    RUN_TEST(test_connecting_audio_and_live_audio_are_one_lossless_ordered_stream);
    RUN_TEST(test_recording_overflow_never_discards_old_speech_and_clear_removes_it);
    RUN_TEST(test_reconnect_discards_pending_and_gap_audio_but_preserves_first_connection);
    RUN_TEST(test_mute_clears_initial_and_live_recordings_without_resetting_connection_history);
    RUN_TEST(test_playback_gain_is_symmetric_monotonic_and_bounded_at_full_scale);
    RUN_TEST(test_ringback_has_bounded_volume_silent_gap_and_continuous_chunks);
    return UNITY_END();
}
