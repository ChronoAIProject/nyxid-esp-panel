#include "nyx_proto.h"
#include "unity.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

void test_parse_auth_ok(void)
{
    const char *txt = "{\"type\":\"auth_ok\",\"node_id\":\"node-1\",\"capabilities\":{\"proxy_binary_chunks\":true}}";
    nyx_auth_result_t auth = {0};

    TEST_ASSERT_EQUAL(NYX_MSG_AUTH_OK, nyx_proto_parse_text(txt, NULL, &auth));
    TEST_ASSERT_EQUAL_STRING("node-1", auth.node_id);
    TEST_ASSERT_NULL(auth.auth_token);

    nyx_proto_free_auth_result(&auth);
}

void test_parse_register_ok(void)
{
    const char *txt = "{\"type\":\"register_ok\",\"node_id\":\"node-1\",\"auth_token\":\"tok\",\"signing_secret\":\"sig\"}";
    nyx_auth_result_t auth = {0};

    TEST_ASSERT_EQUAL(NYX_MSG_REGISTER_OK, nyx_proto_parse_text(txt, NULL, &auth));
    TEST_ASSERT_EQUAL_STRING("node-1", auth.node_id);
    TEST_ASSERT_EQUAL_STRING("tok", auth.auth_token);
    TEST_ASSERT_EQUAL_STRING("sig", auth.signing_secret);

    nyx_proto_free_auth_result(&auth);
}

void test_parse_auth_error(void)
{
    const char *txt = "{\"type\":\"auth_error\",\"message\":\"bad token\"}";
    nyx_auth_result_t auth = {0};

    TEST_ASSERT_EQUAL(NYX_MSG_AUTH_ERROR, nyx_proto_parse_text(txt, NULL, &auth));
    TEST_ASSERT_EQUAL_STRING("bad token", auth.message);

    nyx_proto_free_auth_result(&auth);
}

void test_parse_heartbeat_ping(void)
{
    TEST_ASSERT_EQUAL(NYX_MSG_HEARTBEAT_PING,
                      nyx_proto_parse_text("{\"type\":\"heartbeat_ping\",\"timestamp\":\"now\"}", NULL, NULL));
}

void test_parse_proxy_request_with_body_headers_and_hmac_fields(void)
{
    const char *txt =
        "{\"type\":\"proxy_request\","
        "\"request_id\":\"r1\","
        "\"service_id\":\"svc1\","
        "\"service_slug\":\"cam-a1\","
        "\"base_url\":\"http://node.local\","
        "\"method\":\"POST\","
        "\"path\":\"/lcd/text\","
        "\"query\":\"lang=zh\","
        "\"headers\":{\"content-type\":\"application/json\",\"x-test\":\"1\"},"
        "\"body\":\"eyJ0ZXh0Ijoi5L2g5aW9In0=\","
        "\"timestamp\":\"2026-05-13T00:00:00Z\","
        "\"nonce\":\"n1\","
        "\"signature\":\"s1\"}";
    nyx_proxy_request_t req = {0};

    TEST_ASSERT_EQUAL(NYX_MSG_PROXY_REQUEST, nyx_proto_parse_text(txt, &req, NULL));
    TEST_ASSERT_EQUAL_STRING("r1", req.request_id);
    TEST_ASSERT_EQUAL_STRING("svc1", req.service_id);
    TEST_ASSERT_EQUAL_STRING("cam-a1", req.service_slug);
    TEST_ASSERT_EQUAL_STRING("POST", req.method);
    TEST_ASSERT_EQUAL_STRING("/lcd/text", req.path);
    TEST_ASSERT_EQUAL_STRING("lang=zh", req.query);
    TEST_ASSERT_EQUAL_STRING("2026-05-13T00:00:00Z", req.timestamp);
    TEST_ASSERT_EQUAL_STRING("n1", req.nonce);
    TEST_ASSERT_EQUAL_STRING("s1", req.signature);
    TEST_ASSERT_EQUAL_UINT(2, req.header_count);
    TEST_ASSERT_EQUAL_STRING("content-type", req.headers[0].name);
    TEST_ASSERT_EQUAL_STRING("application/json", req.headers[0].value);
    TEST_ASSERT_EQUAL_STRING("{\"text\":\"你好\"}", (const char *)req.body);
    TEST_ASSERT_EQUAL_UINT(strlen("{\"text\":\"你好\"}"), req.body_len);

    nyx_proto_free_request(&req);
}

void test_build_auth_and_register_frames(void)
{
    char buf[256];

    TEST_ASSERT_GREATER_THAN_UINT(0, nyx_proto_build_auth(buf, sizeof(buf), "node-1", "tok"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"type\":\"auth\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"node_id\":\"node-1\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"token\":\"tok\""));

    TEST_ASSERT_GREATER_THAN_UINT(0, nyx_proto_build_register_token(buf, sizeof(buf), "regtok"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"type\":\"register\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"token\":\"regtok\""));
}

void test_build_proxy_response_base64_body(void)
{
    char buf[256];

    TEST_ASSERT_GREATER_THAN_UINT(0, nyx_proto_build_proxy_response(
                                         buf, sizeof(buf), "r1", 200, "application/json",
                                         "{\"ok\":true}", strlen("{\"ok\":true}")));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"type\":\"proxy_response\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"request_id\":\"r1\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"status\":200"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"content-type\":\"application/json\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"body\":\"eyJvayI6dHJ1ZX0=\""));
}

void test_build_pong(void)
{
    char buf[128];

    TEST_ASSERT_GREATER_THAN_UINT(0, nyx_proto_build_pong(buf, sizeof(buf), "now"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"type\":\"heartbeat_pong\""));
    TEST_ASSERT_NOT_NULL(strstr(buf, "\"timestamp\":\"now\""));
}

void test_small_buffer_fails(void)
{
    char buf[8];
    TEST_ASSERT_EQUAL_UINT(0, nyx_proto_build_auth(buf, sizeof(buf), "node-1", "tok"));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_parse_auth_ok);
    RUN_TEST(test_parse_register_ok);
    RUN_TEST(test_parse_auth_error);
    RUN_TEST(test_parse_heartbeat_ping);
    RUN_TEST(test_parse_proxy_request_with_body_headers_and_hmac_fields);
    RUN_TEST(test_build_auth_and_register_frames);
    RUN_TEST(test_build_proxy_response_base64_body);
    RUN_TEST(test_build_pong);
    RUN_TEST(test_small_buffer_fails);
    return UNITY_END();
}
