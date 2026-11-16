#include "panel_board.h"
#include "unity.h"
#include <stdint.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static void image(uint8_t out[PANEL_BOARD_IMAGE_HEADER_SIZE])
{
    memset(out, 0, PANEL_BOARD_IMAGE_HEADER_SIZE);
    out[0] = 0xe9;
    out[12] = 0x12;
    out[15] = PANEL_BOARD_MIN_REVISION & 0xff;
    out[16] = PANEL_BOARD_MIN_REVISION >> 8;
    out[17] = PANEL_BOARD_MAX_REVISION & 0xff;
    out[18] = PANEL_BOARD_MAX_REVISION >> 8;
    memcpy(out + PANEL_BOARD_IMAGE_OFFSET, &panel_board_image_desc, sizeof(panel_board_image_desc));
}

static void test_exact_identity_and_every_truncated_prefix(void)
{
    uint8_t bytes[PANEL_BOARD_IMAGE_HEADER_SIZE];
    image(bytes);
    TEST_ASSERT_TRUE(panel_board_image_matches(bytes, sizeof(bytes)));
    TEST_ASSERT_FALSE(panel_board_image_matches(NULL, sizeof(bytes)));
    for (size_t n = 0; n < sizeof(bytes); ++n)
        TEST_ASSERT_FALSE(panel_board_image_matches(bytes, n));
    bytes[0] = 0;
    TEST_ASSERT_FALSE(panel_board_image_matches(bytes, sizeof(bytes)));
    for (size_t i = 12; i <= 18; ++i) {
        if (i == 14) continue; // Legacy minimum-revision byte is unused by IDF.
        image(bytes);
        bytes[i] ^= 1;
        TEST_ASSERT_FALSE(panel_board_image_matches(bytes, sizeof(bytes)));
    }
}

static void test_other_board_profile_and_noncanonical_padding_are_rejected(void)
{
    uint8_t bytes[PANEL_BOARD_IMAGE_HEADER_SIZE];
    // This exercises the production C verifier, including terminators/padding.
    // Matching project names, versions and a valid image magic are insufficient.
    for (size_t n = PANEL_BOARD_IMAGE_OFFSET; n < sizeof(bytes); ++n) {
        image(bytes);
        bytes[n] ^= 1;
        TEST_ASSERT_FALSE(panel_board_image_matches(bytes, sizeof(bytes)));
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_exact_identity_and_every_truncated_prefix);
    RUN_TEST(test_other_board_profile_and_noncanonical_padding_are_rejected);
    return UNITY_END();
}
