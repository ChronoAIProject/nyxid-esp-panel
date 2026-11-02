#include "unity.h"
#include "nyx_link_sm.h"

void setUp(void) {}
void tearDown(void) {}

void test_initial_state_is_disconnected(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_DISCONNECTED, nyx_link_sm_state(&sm));
}

void test_netif_up_connects(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_CONNECTING, nyx_link_sm_state(&sm));
}

void test_ack_reaches_connected(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ACK);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_CONNECTED, nyx_link_sm_state(&sm));
}

void test_wss_error_reconnects_with_backoff(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ACK);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ERR);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_RECONNECTING, nyx_link_sm_state(&sm));
    TEST_ASSERT_EQUAL_UINT32(1000, nyx_link_sm_backoff_ms(&sm));
}

void test_iface_change_forces_reconnect(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ACK);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_IFACE_CHANGED);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_RECONNECTING, nyx_link_sm_state(&sm));
}

void test_backoff_caps_at_sixty_seconds(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    for (int i = 0; i < 10; i++) {
        nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ERR);
    }
    TEST_ASSERT_EQUAL_UINT32(60000, nyx_link_sm_backoff_ms(&sm));
}

void test_success_resets_backoff(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    for (int i = 0; i < 4; i++) {
        nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ERR);
    }
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ACK);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ERR);
    TEST_ASSERT_EQUAL_UINT32(1000, nyx_link_sm_backoff_ms(&sm));
}

void test_netif_down_disconnects(void)
{
    nyx_link_sm_t sm;
    nyx_link_sm_init(&sm);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_UP);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_WSS_ACK);
    nyx_link_sm_apply(&sm, NYX_LINK_EVT_NETIF_DOWN);
    TEST_ASSERT_EQUAL_INT(NYX_LINK_DISCONNECTED, nyx_link_sm_state(&sm));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_initial_state_is_disconnected);
    RUN_TEST(test_netif_up_connects);
    RUN_TEST(test_ack_reaches_connected);
    RUN_TEST(test_wss_error_reconnects_with_backoff);
    RUN_TEST(test_iface_change_forces_reconnect);
    RUN_TEST(test_backoff_caps_at_sixty_seconds);
    RUN_TEST(test_success_resets_backoff);
    RUN_TEST(test_netif_down_disconnects);
    return UNITY_END();
}
