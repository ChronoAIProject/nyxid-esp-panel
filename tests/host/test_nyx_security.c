#include "unity.h"
#include "nyx_security.h"
#include "security_fixtures.h"
#include <time.h>
void setUp(void) {}
void tearDown(void) {}
time_t time(time_t *t) { if(t) *t=1793650000;return 1793650000; }
static const char *secret="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
static void test_python_proxy_signature_and_replay(void) {
    cJSON *p=cJSON_Parse(proxy_fixture);
    TEST_ASSERT_TRUE(nyx_security_verify_proxy(p,secret));
    TEST_ASSERT_FALSE(nyx_security_verify_proxy(p,secret));
    cJSON_Delete(p);
}
static void test_python_upload_signature_and_replay(void) {
    cJSON *u=cJSON_Parse(upload_fixture);
    TEST_ASSERT_TRUE(nyx_security_verify_upload(u,secret,"node-test"));
    TEST_ASSERT_FALSE(nyx_security_verify_upload(u,secret,"node-test"));
    cJSON_Delete(u);
}
static void test_tampering_wrong_key_and_identity(void) {
    cJSON *u=cJSON_Parse(upload_fixture);
    TEST_ASSERT_FALSE(nyx_security_verify_upload(u,"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","node-test"));
    TEST_ASSERT_FALSE(nyx_security_verify_upload(u,secret,"other-node"));
    cJSON_ReplaceItemInObject(cJSON_GetObjectItem(u,"parameters"),"path",cJSON_CreateString("/system/reboot"));
    TEST_ASSERT_FALSE(nyx_security_verify_upload(u,secret,"node-test"));
    cJSON_Delete(u);
}
static void test_malformed_upload_timestamp(void) {
    const char *values[]={"null","\"1793650000\"","1.5","1e100","-1"};
    for (size_t i=0;i<sizeof(values)/sizeof(values[0]);++i) {
        cJSON *u=cJSON_Parse(upload_fixture);
        cJSON_ReplaceItemInObject(u,"timestamp",cJSON_Parse(values[i]));
        TEST_ASSERT_FALSE(nyx_security_verify_upload(u,secret,"node-test"));
        cJSON_Delete(u);
    }
}
int main(void) { UNITY_BEGIN(); RUN_TEST(test_python_proxy_signature_and_replay);
    RUN_TEST(test_python_upload_signature_and_replay); RUN_TEST(test_tampering_wrong_key_and_identity);
    RUN_TEST(test_malformed_upload_timestamp);return UNITY_END(); }
