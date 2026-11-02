#include "unity.h"
#include "stream_sm.h"

void setUp(void) {}
void tearDown(void) {}

void test_initial_state_idle(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    TEST_ASSERT_EQUAL_INT(STREAM_IDLE, stream_sm_state(&sm));
}

void test_start_without_target_or_net_stays_idle(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_START);
    TEST_ASSERT_EQUAL_INT(STREAM_IDLE, stream_sm_state(&sm));
}

void test_start_with_target_and_net_enters_starting(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    TEST_ASSERT_EQUAL_INT(STREAM_STARTING, stream_sm_state(&sm));
}

void test_whip_ok_enters_publishing(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_OK);
    TEST_ASSERT_EQUAL_INT(STREAM_PUBLISHING, stream_sm_state(&sm));
}

void test_whip_error_fails(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_OK);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_ERR);
    TEST_ASSERT_EQUAL_INT(STREAM_FAILED, stream_sm_state(&sm));
}

void test_retry_disabled_while_net_down(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_ERR);
    TEST_ASSERT_TRUE(stream_sm_should_retry(&sm));

    stream_sm_apply(&sm, STREAM_EVT_NET_DOWN);
    TEST_ASSERT_FALSE(stream_sm_should_retry(&sm));

    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    TEST_ASSERT_EQUAL_INT(STREAM_IDLE, stream_sm_state(&sm));
}

void test_retry_timer_restarts_when_ready(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_ERR);
    stream_sm_apply(&sm, STREAM_EVT_RETRY_TIMER);
    TEST_ASSERT_EQUAL_INT(STREAM_STARTING, stream_sm_state(&sm));
}

void test_stop_returns_to_idle(void)
{
    stream_sm_t sm;
    stream_sm_init(&sm);
    stream_sm_apply(&sm, STREAM_EVT_NET_UP);
    stream_sm_apply(&sm, STREAM_EVT_TARGET_SET);
    stream_sm_apply(&sm, STREAM_EVT_START);
    stream_sm_apply(&sm, STREAM_EVT_WHIP_OK);
    stream_sm_apply(&sm, STREAM_EVT_STOP);
    TEST_ASSERT_EQUAL_INT(STREAM_IDLE, stream_sm_state(&sm));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_initial_state_idle);
    RUN_TEST(test_start_without_target_or_net_stays_idle);
    RUN_TEST(test_start_with_target_and_net_enters_starting);
    RUN_TEST(test_whip_ok_enters_publishing);
    RUN_TEST(test_whip_error_fails);
    RUN_TEST(test_retry_disabled_while_net_down);
    RUN_TEST(test_retry_timer_restarts_when_ready);
    RUN_TEST(test_stop_returns_to_idle);
    return UNITY_END();
}
