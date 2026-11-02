#include "unity.h"

#include "status_bar.h"

void setUp(void) {}
void tearDown(void) {}

static void test_symbols_for_link_and_ip(void)
{
    net_manager_snapshot_t net = {
        .wifi_link = true,
        .wifi_ip = true,
        .eth_link = true,
        .eth_ip = false,
    };
    status_bar_text_t out = {0};

    status_bar_format(&net, NYX_NODE_CONNECTING, "cam-one", &out);

    TEST_ASSERT_EQUAL_STRING("◐", out.eth);
    TEST_ASSERT_EQUAL_STRING("●", out.wifi);
    TEST_ASSERT_EQUAL_STRING("◐", out.nyx);
    TEST_ASSERT_EQUAL_STRING("cam-one", out.footer);
}

static void test_empty_state_is_open_circle(void)
{
    status_bar_text_t out = {0};

    status_bar_format(NULL, NYX_NODE_DOWN, NULL, &out);

    TEST_ASSERT_EQUAL_STRING("○", out.eth);
    TEST_ASSERT_EQUAL_STRING("○", out.wifi);
    TEST_ASSERT_EQUAL_STRING("○", out.nyx);
    TEST_ASSERT_EQUAL_STRING("nyx: unregistered", out.footer);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_symbols_for_link_and_ip);
    RUN_TEST(test_empty_state_is_open_circle);
    return UNITY_END();
}
