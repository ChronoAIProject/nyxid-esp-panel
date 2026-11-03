#include "unity.h"
#include "device_login_proto.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static const char *request = "{\"device_code\":\"nyx_adc_private\",\"user_code\":\"ABCD-EFGH\","
    "\"verification_uri\":\"https://nyxid.ai/login/device\","
    "\"verification_uri_complete\":\"https://nyxid.ai/login/device?user_code=ABCD-EFGH\","
    "\"expires_in\":600,\"interval\":5}";

void test_real_account_login_request_and_trusted_link(void)
{
    device_login_request_t parsed;
    TEST_ASSERT_TRUE(device_login_parse_request(request, &parsed));
    TEST_ASSERT_EQUAL_STRING("ABCD-EFGH", parsed.user_code);
    TEST_ASSERT_EQUAL_UINT(5, parsed.interval);
    TEST_ASSERT_EQUAL_UINT(600, parsed.expires_in);
    char changed[1024]; strcpy(changed, request);
    char *origin = strstr(changed, "nyxid.ai"); origin[0] = 'x';
    TEST_ASSERT_FALSE(device_login_parse_request(changed, &parsed));
    TEST_ASSERT_EQUAL_STRING("", parsed.device_code);
    TEST_ASSERT_FALSE(device_login_parse_request("{}", &parsed));
    strcpy(changed, request);
    *strstr(changed, "600") = '-';
    TEST_ASSERT_FALSE(device_login_parse_request(changed, &parsed));
}

void test_pending_slow_down_denied_expiry_and_one_time_delivery(void)
{
    TEST_ASSERT_EQUAL(LOGIN_POLL_PENDING, device_login_parse_poll(400, "{\"error_code\":11202}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_SLOW_DOWN, device_login_parse_poll(429, "{\"error_code\":11203}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_RATE_LIMITED, device_login_parse_poll(429, "{\"error_code\":11206}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_DENIED, device_login_parse_poll(403, "{\"error_code\":11204}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_EXPIRED, device_login_parse_poll(410, "{\"error_code\":11201}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_DELIVERED, device_login_parse_poll(410, "{\"error_code\":11205}"));
    TEST_ASSERT_EQUAL_UINT(10, device_login_backoff(5));
    TEST_ASSERT_EQUAL_UINT(15, device_login_backoff(10));
    TEST_ASSERT_EQUAL_UINT(300, device_login_backoff(300));
    TEST_ASSERT_EQUAL(LOGIN_POLL_ERROR, device_login_parse_poll(200, "{\"error_code\":11202}"));
}

void test_client_keys_cannot_be_accepted_as_node_account_authorization(void)
{
    TEST_ASSERT_EQUAL(LOGIN_POLL_APPROVED, device_login_parse_poll(200,
        "{\"auth_kind\":\"account_session\",\"access_token\":\"temporary-account\"}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_ERROR, device_login_parse_poll(200,
        "{\"auth_kind\":\"agent_key\",\"access_token\":\"wrong-kind\",\"api_key\":\"nyxid_ag_example\"}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_ERROR, device_login_parse_poll(200,
        "{\"api_key\":\"nyxid_ag_example\",\"node_id\":\"ownership-placeholder\"}"));
    TEST_ASSERT_EQUAL(LOGIN_POLL_ERROR, device_login_parse_poll(200, "not json"));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_real_account_login_request_and_trusted_link);
    RUN_TEST(test_pending_slow_down_denied_expiry_and_one_time_delivery);
    RUN_TEST(test_client_keys_cannot_be_accepted_as_node_account_authorization);
    return UNITY_END();
}
