#include "unity.h"
#include "panel_voice_audio.h"
#include <stdlib.h>
#include <limits.h>
void setUp(void) {}
void tearDown(void) {}
void test_reference_preserves_pcm_across_different_codec_and_aec_frame_sizes(void)
{
    panel_voice_reference_t ref={0};int16_t input[320],output[256];
    for(int i=0;i<320;i++)input[i]=(int16_t)(i-160);
    panel_voice_reference_push(&ref,input,320);
    panel_voice_reference_read(&ref,output,256);
    for(int i=0;i<256;i++)TEST_ASSERT_EQUAL_INT(input[i],output[i]);
    panel_voice_reference_push(&ref,input,320);
    panel_voice_reference_read(&ref,output,256);
    for(int i=0;i<64;i++)TEST_ASSERT_EQUAL_INT(input[i+256],output[i]);
    for(int i=64;i<256;i++)TEST_ASSERT_EQUAL_INT(input[i-64],output[i]);
    panel_voice_reference_read(&ref,output,256);
    for(int i=0;i<128;i++)TEST_ASSERT_EQUAL_INT(input[i+192],output[i]);
    for(int i=128;i<256;i++)TEST_ASSERT_EQUAL_INT(0,output[i]);
    TEST_ASSERT_EQUAL_INT(0,ref.count);
}
void test_reference_overflow_keeps_recent_playback_and_underflow_is_silence(void)
{
    panel_voice_reference_t ref={0};int16_t in[2049],out[2048];
    for(int i=0;i<2049;i++)in[i]=(int16_t)i;
    panel_voice_reference_push(&ref,in,2048);
    panel_voice_reference_push(&ref,in+2048,1);
    panel_voice_reference_read(&ref,out,2048);
    for(int i=0;i<2048;i++)TEST_ASSERT_EQUAL_INT(i+1,out[i]);
    panel_voice_reference_push(&ref,in,2049);
    panel_voice_reference_read(&ref,out,2048);
    for(int i=0;i<2048;i++)TEST_ASSERT_EQUAL_INT(i+1,out[i]);
    panel_voice_reference_read(&ref,out,2048);
    for(int i=0;i<2048;i++)TEST_ASSERT_EQUAL_INT(0,out[i]);
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

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_reference_preserves_pcm_across_different_codec_and_aec_frame_sizes);
    RUN_TEST(test_reference_overflow_keeps_recent_playback_and_underflow_is_silence);
    RUN_TEST(test_playback_gain_is_symmetric_monotonic_and_bounded_at_full_scale);
    return UNITY_END();
}
