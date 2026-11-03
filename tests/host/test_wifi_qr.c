#include "unity.h"
#include "wifi_qr.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

void test_wifi_escapes_unicode_and_hidden_network(void)
{
    wifi_qr_t wifi;
    TEST_ASSERT_TRUE(wifi_qr_parse("WIFI:T:WPA;S:客厅\\;WiFi;P:pass\\:word\\\\123;H:true;;", &wifi));
    TEST_ASSERT_EQUAL_STRING("客厅;WiFi", wifi.ssid);
    TEST_ASSERT_EQUAL_STRING("pass:word\\123", wifi.password);
    TEST_ASSERT_TRUE(wifi.hidden);
    TEST_ASSERT_TRUE(wifi_qr_parse("WIFI:T:nopass;S:Guest;;", &wifi));
    TEST_ASSERT_EQUAL_STRING("", wifi.password);
}

void test_wifi_invalid_and_ambiguous_codes_are_rejected(void)
{
    wifi_qr_t wifi;
    const char *invalid[] = {
        "WIFI:T:WPA;S:Home;P:short;;", "WIFI:T:WPA;S:Home;S:Other;P:12345678;;",
        "WIFI:T:WPA2-EAP;S:Office;P:12345678;;", "WIFI:T:nopass;S:Guest;P:secret;;",
        "WIFI:T:WPA;S:Home;P:12345678\\", "WIFI:T:WPA;S:Home;P:12345678",
        "WIFI:T:WPA;S:;P:12345678;;", "WIFI:T:WPA;S:Home;P:12345678;H:maybe;;",
        "nyxprov://bootstrap?token=nyx_obt_test&id=placeholder",
        "WIFI:T:WPA;S:123456789012345678901234567890123;P:12345678;;"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
        TEST_ASSERT_FALSE(wifi_qr_parse(invalid[i], &wifi));
    TEST_ASSERT_TRUE(wifi_qr_parse("WIFI:T:WPA;S:12345678901234567890123456789012;P:12345678;;", &wifi));
    TEST_ASSERT_EQUAL_UINT(32, strlen(wifi.ssid));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_wifi_escapes_unicode_and_hidden_network);
    RUN_TEST(test_wifi_invalid_and_ambiguous_codes_are_rejected);
    return UNITY_END();
}
