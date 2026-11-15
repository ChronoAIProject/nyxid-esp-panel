#include "product_audio_layout.h"
#include "unity.h"
#include <limits.h>

void setUp(void) {}
void tearDown(void) {}

static void test_mic_and_reference_keep_signed_samples_and_frame_order(void)
{
    const int16_t raw[] = {1, -2, 111, 222, INT16_MIN, INT16_MAX, 333, 444, -5, 6, 555, 666};
    const int16_t expected[] = {1, -2, INT16_MIN, INT16_MAX, -5, 6};
    int16_t out[8] = {0, 0, 0, 0, 0, 0, 777, 888};
    product_lcd_x8_normalize_capture(out, raw, 3);
    TEST_ASSERT_EQUAL_MEMORY(expected, out, sizeof(expected));
    TEST_ASSERT_EQUAL_INT(777, out[6]);
    TEST_ASSERT_EQUAL_INT(888, out[7]);
}

static void test_chunk_boundaries_do_not_drop_or_repeat_frames(void)
{
    int16_t raw[4 * 261], out[2 * 261];
    for (size_t i = 0; i < 261; ++i) {
        raw[4 * i] = (int16_t)i;
        raw[4 * i + 1] = -(int16_t)i;
        raw[4 * i + 2] = 1234;
        raw[4 * i + 3] = 5678;
    }
    for (size_t done = 0; done < 261;) {
        size_t frames = 261 - done;
        if (frames > 128) frames = 128;
        product_lcd_x8_normalize_capture(out + 2 * done, raw + 4 * done, frames);
        done += frames;
    }
    for (size_t i = 0; i < 261; ++i) {
        TEST_ASSERT_EQUAL_INT(i, out[2 * i]);
        TEST_ASSERT_EQUAL_INT(-(int)i, out[2 * i + 1]);
    }
}

static void test_zero_and_in_place_conversion(void)
{
    product_lcd_x8_normalize_capture(NULL, NULL, 0);
    int16_t raw[] = {1, 2, 30, 40, 5, 6, 70, 80};
    const int16_t expected[] = {1, 2, 5, 6};
    product_lcd_x8_normalize_capture(raw, raw, 2);
    TEST_ASSERT_EQUAL_MEMORY(expected, raw, sizeof(expected));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_mic_and_reference_keep_signed_samples_and_frame_order);
    RUN_TEST(test_chunk_boundaries_do_not_drop_or_repeat_frames);
    RUN_TEST(test_zero_and_in_place_conversion);
    return UNITY_END();
}
