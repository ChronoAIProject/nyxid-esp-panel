#include "unity.h"
#include "panel_voice_proto.h"
#include <string.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}
void test_bot_discovery_preserves_dynamic_identity_and_rejects_unsafe_paths(void)
{
    panel_voice_bot_t bots[PANEL_VOICE_MAX_BOTS]; size_t count;
    cJSON *root = cJSON_Parse("{\"agents\":[{\"agent_id\":\"11111111-2222-3333-4444-555555555555\","
        "\"conversation_id\":\"nyxa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"name\":\"any-bot\",\"display_name\":\"My Bot\"}]}");
    TEST_ASSERT_TRUE(panel_voice_parse_agents(root, bots, &count));
    TEST_ASSERT_EQUAL_UINT(1, count);
    TEST_ASSERT_EQUAL_STRING("My Bot", bots[0].name);
    cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(root, "agents"), 0),
        "conversation_id", cJSON_CreateString("../../another/conversation"));
    TEST_ASSERT_FALSE(panel_voice_parse_agents(root, bots, &count));
    cJSON_Delete(root);
}
void test_voice_uses_saved_billing_binding_and_requires_available_supported_provider(void)
{
    cJSON *discovery = cJSON_Parse("{\"preferences\":{\"service_id\":\"11111111-2222-3333-4444-555555555555\","
        "\"connection_id\":null,\"key_source\":\"platform\",\"model\":\"chosen-live-model\",\"voice\":\"chosen-voice\",\"input_mode\":\"automatic\"}}");
    cJSON *options = cJSON_Parse("{\"options\":[{\"service_id\":\"11111111-2222-3333-4444-555555555555\","
        "\"connection_id\":null,\"key_source\":\"platform\",\"model\":\"chosen-live-model\",\"available\":true,"
        "\"voice\":{\"protocol\":\"openai_live\",\"voices\":[{\"id\":\"chosen-voice\"}]}}]}");
    cJSON *p = panel_voice_preferences(discovery, options, PANEL_VOICE_INPUT_HOLD);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL_STRING("chosen-live-model", cJSON_GetObjectItemCaseSensitive(p, "model")->valuestring);
    TEST_ASSERT_EQUAL_STRING("push_to_talk", cJSON_GetObjectItemCaseSensitive(p, "input_mode")->valuestring);
    cJSON_Delete(p);
    p = panel_voice_preferences(discovery, options, PANEL_VOICE_INPUT_AUTOMATIC);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL_STRING("automatic", cJSON_GetObjectItemCaseSensitive(p, "input_mode")->valuestring);
    TEST_ASSERT_EQUAL_STRING("chosen-live-model", cJSON_GetObjectItemCaseSensitive(p, "model")->valuestring);
    cJSON_Delete(p);
    TEST_ASSERT_NULL(panel_voice_preferences(discovery, options, (panel_voice_input_mode_t)99));
    cJSON *option = cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(options, "options"), 0);
    cJSON_ReplaceItemInObjectCaseSensitive(option, "available", cJSON_CreateBool(false));
    TEST_ASSERT_NULL(panel_voice_preferences(discovery, options, PANEL_VOICE_INPUT_HOLD));
    cJSON_ReplaceItemInObjectCaseSensitive(option, "available", cJSON_CreateBool(true));
    cJSON_ReplaceItemInObjectCaseSensitive(option, "key_source", cJSON_CreateString("own"));
    TEST_ASSERT_NULL(panel_voice_preferences(discovery, options, PANEL_VOICE_INPUT_HOLD));
    cJSON_ReplaceItemInObjectCaseSensitive(option, "key_source", cJSON_CreateString("platform"));
    cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(option, "voice"), "protocol", cJSON_CreateString("xai_realtime"));
    TEST_ASSERT_NULL(panel_voice_preferences(discovery, options, PANEL_VOICE_INPUT_HOLD));
    cJSON_Delete(discovery); cJSON_Delete(options);
}
void test_voice_errors_explain_provider_credits_without_reflecting_remote_text(void)
{
    cJSON *error = cJSON_Parse("{\"message\":\"secret provider response\",\"details\":{"
        "\"stage\":\"provider_create\",\"provider_status\":429,"
        "\"provider_code\":\"credit_balance_exhausted\"}}");
    TEST_ASSERT_NOT_NULL(strstr(panel_voice_error_message(error, 503), "no credits"));
    cJSON *details = cJSON_GetObjectItemCaseSensitive(error, "details");
    cJSON_ReplaceItemInObjectCaseSensitive(details, "provider_code", cJSON_CreateString("secret provider response"));
    TEST_ASSERT_NOT_NULL(strstr(panel_voice_error_message(error, 503), "rate limit"));
    TEST_ASSERT_NULL(strstr(panel_voice_error_message(error, 503), "secret"));
    TEST_ASSERT_NOT_NULL(strstr(panel_voice_error_message(error, 401), "Sign in again"));
    TEST_ASSERT_NOT_NULL(strstr(panel_voice_error_message(NULL, 403), "Bot's voice"));
    TEST_ASSERT_NULL(panel_voice_error_message(NULL, 500));
    cJSON_Delete(error);
}
void test_initial_sdp_offer_negotiates_dtls_without_changing_credentials(void)
{
    const char *offer="v=0\r\na=ice-pwd:unchanged\r\na=fingerprint:sha-256 unchanged\r\n"
        "a=setup:passive\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\na=setup:active\r\n";
    char *out=panel_voice_prepare_offer(offer,strlen(offer));
    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_EQUAL_STRING("v=0\r\na=ice-pwd:unchanged\r\na=fingerprint:sha-256 unchanged\r\n"
        "a=setup:actpass\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\na=setup:actpass\r\n",out);
    char *again=panel_voice_prepare_offer(out,strlen(out));
    TEST_ASSERT_EQUAL_STRING(out,again);free(again);free(out);
    TEST_ASSERT_NULL(panel_voice_prepare_offer("a=setup:holdconn",16));
    TEST_ASSERT_NULL(panel_voice_prepare_offer("v=0",3));
    TEST_ASSERT_NULL(panel_voice_prepare_offer("a=setup:active\0ignored",22));
    TEST_ASSERT_NULL(panel_voice_prepare_offer("",0));
    TEST_ASSERT_NULL(panel_voice_prepare_offer("x",64*1024+1));
}

void test_provider_readiness_accepts_large_session_without_trusting_other_events(void)
{
    char *event=malloc(9000); TEST_ASSERT_NOT_NULL(event);
    const char *head="{\"type\":\"session.started\",\"session\":{\"instructions\":\"";
    size_t n=strlen(head);memcpy(event,head,n);memset(event+n,'a',8192);n+=8192;
    memcpy(event+n,"\"}}",3);n+=3;
    TEST_ASSERT_TRUE(panel_voice_provider_started(event,n));
    TEST_ASSERT_FALSE(panel_voice_provider_started(event,n-1));
    free(event);
    const char *other="{\"type\":\"response.done\",\"text\":\"session.started\"}";
    TEST_ASSERT_FALSE(panel_voice_provider_started(other,strlen(other)));
    TEST_ASSERT_FALSE(panel_voice_provider_started(NULL,10));
    TEST_ASSERT_FALSE(panel_voice_provider_started("x",96*1024+1));
}

void test_normal_server_end_requires_closed_state_and_a_known_reason(void)
{
    cJSON *row=cJSON_Parse("{\"state\":\"closed\",\"end_reason\":\"idle\"}");
    TEST_ASSERT_TRUE(panel_voice_normal_end(row));
    cJSON_ReplaceItemInObjectCaseSensitive(row,"state",cJSON_CreateString("active"));
    TEST_ASSERT_FALSE(panel_voice_normal_end(row));
    cJSON_ReplaceItemInObjectCaseSensitive(row,"state",cJSON_CreateString("closed"));
    cJSON_ReplaceItemInObjectCaseSensitive(row,"end_reason",cJSON_CreateString("voice_connection_lost"));
    TEST_ASSERT_FALSE(panel_voice_normal_end(row));
    cJSON_ReplaceItemInObjectCaseSensitive(row,"end_reason",cJSON_CreateString("time_limit"));
    TEST_ASSERT_TRUE(panel_voice_normal_end(row));
    TEST_ASSERT_FALSE(panel_voice_normal_end(NULL));
    cJSON_Delete(row);
}


void test_input_modes_keep_authority_mute_and_hold_boundaries(void)
{
    TEST_ASSERT_TRUE(panel_voice_capture_allowed(PANEL_VOICE_INPUT_AUTOMATIC, false, false, true, true, true, false));
    TEST_ASSERT_FALSE(panel_voice_capture_allowed(PANEL_VOICE_INPUT_HOLD, false, false, true, true, true, false));
    TEST_ASSERT_TRUE(panel_voice_capture_allowed(PANEL_VOICE_INPUT_HOLD, true, false, true, true, true, false));
    for (int mode=0; mode<=1; ++mode) {
        TEST_ASSERT_FALSE(panel_voice_capture_allowed(mode, true, true, true, true, true, false));
        TEST_ASSERT_FALSE(panel_voice_capture_allowed(mode, true, false, false, true, true, false));
        TEST_ASSERT_FALSE(panel_voice_capture_allowed(mode, true, false, true, false, true, false));
        TEST_ASSERT_FALSE(panel_voice_capture_allowed(mode, true, false, true, true, false, false));
        TEST_ASSERT_FALSE(panel_voice_capture_allowed(mode, true, false, true, true, true, true));
    }
    TEST_ASSERT_FALSE(panel_voice_capture_allowed(99, true, false, true, true, true, false));
}
void test_caption_roles_survive_interleaved_and_reordered_fragments(void)
{
    cJSON *snapshot=cJSON_Parse("{\"captions\":["
        "{\"speaker\":\"user\",\"text\":\"你好\",\"start_ms\":100},"
        "{\"speaker\":\"assistant\",\"text\":\"您好\",\"start_ms\":200},"
        "{\"speaker\":\"user\",\"text\":\"打开相机\",\"start_ms\":300},"
        "{\"speaker\":\"user\",\"text\":\"stale draft\",\"start_ms\":100},"
        "{\"speaker\":\"system\",\"text\":\"untrusted role\"}]}");
    panel_voice_captions_t captions;
    TEST_ASSERT_TRUE(panel_voice_parse_captions(snapshot,&captions));
    TEST_ASSERT_EQUAL_STRING("打开相机",captions.user);
    TEST_ASSERT_EQUAL_STRING("您好",captions.bot);
    TEST_ASSERT_NULL(strstr(captions.user,"stale"));
    TEST_ASSERT_NULL(strstr(captions.bot,"untrusted"));
    cJSON_Delete(snapshot);
    snapshot=cJSON_Parse("{\"captions\":[{\"speaker\":\"assistant\",\"text\":\"一句回复\"}]}");
    TEST_ASSERT_TRUE(panel_voice_parse_captions(snapshot,&captions));
    TEST_ASSERT_EQUAL_STRING("一句回复",captions.bot);
    TEST_ASSERT_EQUAL_STRING("",captions.user);
    cJSON_Delete(snapshot);
    TEST_ASSERT_FALSE(panel_voice_parse_captions(NULL,&captions));
    TEST_ASSERT_EQUAL_STRING("",captions.user);
    TEST_ASSERT_EQUAL_STRING("",captions.bot);
}
void test_caption_bounds_preserve_complete_chinese_characters(void)
{
    cJSON *snapshot=cJSON_CreateObject(),*rows=cJSON_AddArrayToObject(snapshot,"captions"),*row=cJSON_CreateObject();
    cJSON_AddItemToArray(rows,row);
    cJSON_AddStringToObject(row,"speaker","user");
    char long_text[604];
    for (int i=0;i<200;++i) memcpy(long_text+3*i,"你",3);
    long_text[600]=0;
    cJSON_AddStringToObject(row,"text",long_text);
    struct { panel_voice_captions_t captions; unsigned char guard; } out;
    memset(&out,0x55,sizeof(out));
    TEST_ASSERT_TRUE(panel_voice_parse_captions(snapshot,&out.captions));
    TEST_ASSERT_EQUAL_UINT(510,strlen(out.captions.user));
    TEST_ASSERT_EQUAL_STRING("…",out.captions.user+507);
    TEST_ASSERT_EQUAL_STRING("",out.captions.bot);
    TEST_ASSERT_EQUAL_HEX8(0x55,out.guard);
    TEST_ASSERT_FALSE(panel_voice_parse_captions(snapshot,NULL));
    cJSON_Delete(snapshot);
}


void test_remote_requests_validate_full_payload_and_utf8_boundaries(void)
{
    const char *id="11111111-2222-3333-4444-555555555555";
    char json[1200];panel_voice_remote_t out;
    snprintf(json,sizeof(json),"{\"request_id\":\"%s\"}",id);
    TEST_ASSERT_TRUE(panel_voice_parse_remote(json,strlen(json),true,&out));
    TEST_ASSERT_FALSE(out.microphone_muted);
    TEST_ASSERT_FALSE(panel_voice_parse_remote(json,strlen(json),false,&out));
    snprintf(json,sizeof(json),"{\"request_id\":\"%s\",\"text\":\"你好\",\"microphone_muted\":true}",id);
    TEST_ASSERT_TRUE(panel_voice_parse_remote(json,strlen(json),true,&out));
    TEST_ASSERT_TRUE(out.microphone_muted);TEST_ASSERT_EQUAL_STRING("你好",out.text);
    TEST_ASSERT_FALSE(panel_voice_parse_remote(json,strlen(json),false,&out));
    const char *bad[]={
        "{\"text\":\"hi\"}",
        "{\"request_id\":\"bad\"}",
        "{\"request_id\":\"%s\",\"text\":\"\"}",
        "{\"request_id\":\"%s\",\"text\":\"   \"}",
        "{\"request_id\":\"%s\",\"text\":\"　\"}",
        "{\"request_id\":\"%s\",\"text\":\"a\\n\"}",
        "{\"request_id\":\"%s\",\"text\":\"a\\u0000b\"}",
        "{\"request_id\":\"%s\",\"text\":\"\xc0\xaf\"}",
        "{\"request_id\":\"%s\",\"text\":\"\xed\xa0\x80\"}",
        "{\"request_id\":\"%s\",\"text\":\"\xf4\x90\x80\x80\"}",
        "{\"request_id\":\"%s\",\"text\":\"\xe4\"}",
        "{\"request_id\":\"%s\",\"text\":\"a\",\"text\":\"b\"}",
        "{\"request_id\":\"%s\",\"bot\":\"other\"}",
        "{\"request_id\":\"%s\",\"microphone_muted\":1}",
        "{\"request_id\":\"%s\"} {}",
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);++i){
        snprintf(json,sizeof(json),bad[i],id);
        TEST_ASSERT_FALSE(panel_voice_parse_remote(json,strlen(json),true,&out));
    }
    cJSON *j=cJSON_CreateObject();cJSON_AddStringToObject(j,"request_id",id);
    char text[304];for(int i=0;i<101;++i)memcpy(text+3*i,"你",3);text[303]=0;
    cJSON_AddStringToObject(j,"text",text);
    char *encoded=cJSON_PrintUnformatted(j);
    TEST_ASSERT_FALSE(panel_voice_parse_remote(encoded,strlen(encoded),false,&out));free(encoded);
    text[300]=0;cJSON_ReplaceItemInObjectCaseSensitive(j,"text",cJSON_CreateString(text));
    encoded=cJSON_PrintUnformatted(j);
    TEST_ASSERT_TRUE(panel_voice_parse_remote(encoded,strlen(encoded),false,&out));free(encoded);cJSON_Delete(j);
    TEST_ASSERT_FALSE(panel_voice_parse_remote("x",1025,true,&out));
    const char embedded[]="{}\0ignored";
    TEST_ASSERT_FALSE(panel_voice_parse_remote(embedded,sizeof(embedded)-1,true,&out));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_remote_requests_validate_full_payload_and_utf8_boundaries);
    RUN_TEST(test_bot_discovery_preserves_dynamic_identity_and_rejects_unsafe_paths);
    RUN_TEST(test_voice_uses_saved_billing_binding_and_requires_available_supported_provider);
    RUN_TEST(test_voice_errors_explain_provider_credits_without_reflecting_remote_text);
    RUN_TEST(test_initial_sdp_offer_negotiates_dtls_without_changing_credentials);
    RUN_TEST(test_provider_readiness_accepts_large_session_without_trusting_other_events);
    RUN_TEST(test_normal_server_end_requires_closed_state_and_a_known_reason);
    RUN_TEST(test_input_modes_keep_authority_mute_and_hold_boundaries);
    RUN_TEST(test_caption_roles_survive_interleaved_and_reordered_fragments);
    RUN_TEST(test_caption_bounds_preserve_complete_chinese_characters);
    return UNITY_END();
}
