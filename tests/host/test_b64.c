#include "b64.h"
#include "unity.h"

#include <stdint.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static void assert_roundtrip(const uint8_t *input, size_t len, const char *expected)
{
    char enc[128];
    uint8_t dec[128];
    size_t enc_len = 0;
    size_t dec_len = 0;

    TEST_ASSERT_EQUAL_INT(0, b64_encode(input, len, enc, sizeof(enc), &enc_len));
    TEST_ASSERT_EQUAL_STRING(expected, enc);
    TEST_ASSERT_EQUAL_UINT(strlen(expected), enc_len);

    TEST_ASSERT_EQUAL_INT(0, b64_decode(enc, enc_len, dec, sizeof(dec), &dec_len));
    TEST_ASSERT_EQUAL_UINT(len, dec_len);
    if (len > 0) {
        TEST_ASSERT_EQUAL_UINT8_ARRAY(input, dec, len);
    }
}

void test_vectors_from_rfc4648(void)
{
    assert_roundtrip((const uint8_t *)"", 0, "");
    assert_roundtrip((const uint8_t *)"f", 1, "Zg==");
    assert_roundtrip((const uint8_t *)"fo", 2, "Zm8=");
    assert_roundtrip((const uint8_t *)"foo", 3, "Zm9v");
    assert_roundtrip((const uint8_t *)"foob", 4, "Zm9vYg==");
    assert_roundtrip((const uint8_t *)"fooba", 5, "Zm9vYmE=");
    assert_roundtrip((const uint8_t *)"foobar", 6, "Zm9vYmFy");
}

void test_binary_roundtrip(void)
{
    const uint8_t bytes[] = {0x00, 0xff, 0x10, 0x80, 0x7f};
    assert_roundtrip(bytes, sizeof(bytes), "AP8QgH8=");
}

void test_decode_rejects_invalid_input(void)
{
    uint8_t out[16];
    size_t out_len = 0;

    TEST_ASSERT_EQUAL_INT(-1, b64_decode("abc", 3, out, sizeof(out), &out_len));
    TEST_ASSERT_EQUAL_INT(-1, b64_decode("ab?=", 4, out, sizeof(out), &out_len));
    TEST_ASSERT_EQUAL_INT(-1, b64_decode("ab=c", 4, out, sizeof(out), &out_len));
    TEST_ASSERT_EQUAL_INT(-1, b64_decode("ab==aaaa", 8, out, sizeof(out), &out_len));
}

void test_capacity_checks(void)
{
    char enc[4];
    uint8_t dec[2];
    size_t out_len = 0;

    TEST_ASSERT_EQUAL_INT(-1, b64_encode((const uint8_t *)"foo", 3, enc, sizeof(enc), &out_len));
    TEST_ASSERT_EQUAL_INT(-1, b64_decode("Zm9v", 4, dec, sizeof(dec), &out_len));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_vectors_from_rfc4648);
    RUN_TEST(test_binary_roundtrip);
    RUN_TEST(test_decode_rejects_invalid_input);
    RUN_TEST(test_capacity_checks);
    return UNITY_END();
}
