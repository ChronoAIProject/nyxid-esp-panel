#include "unity.h"
#include "panel_voice_proto.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void setUp(void) {}
void tearDown(void) {}
static const char *voice_service="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
static const char *other_service="bbbbbbbb-cccc-dddd-eeee-ffffffffffff";
static const char *bot_id="11111111-2222-3333-4444-555555555555";
static const char *key_id="99999999-8888-7777-6666-555555555555";

static cJSON *voice_inventory_row(const char *id, const char *slug, bool active)
{
    cJSON *row=cJSON_CreateObject();
    cJSON_AddStringToObject(row,"id",id);cJSON_AddStringToObject(row,"catalog_service_slug",slug);
    cJSON_AddBoolToObject(row,"is_active",active);
    char endpoint[256];snprintf(endpoint,sizeof(endpoint),"https://metadata.example/api/v1/proxy/nyxbot-voice/%s",id);
    cJSON_AddStringToObject(row,"endpoint_url",endpoint);
    // This public inventory field describes personal/org provenance, not the Bot binding.
    cJSON_AddStringToObject(row,"source","personal");
    return row;
}
void test_voice_inventory_requires_active_catalog_and_exact_service_path(void)
{
    panel_voice_service_t services[PANEL_VOICE_MAX_BOTS];size_t count;
    cJSON *root=cJSON_CreateObject(), *rows=cJSON_AddArrayToObject(root,"keys");
    cJSON_AddItemToArray(rows,voice_inventory_row(other_service,"other-service",true));
    cJSON_AddItemToArray(rows,voice_inventory_row(other_service,"nyxbot-voice",false));
    cJSON *row=voice_inventory_row(voice_service,"nyxbot-voice",true);cJSON_AddItemToArray(rows,row);
    TEST_ASSERT_TRUE(panel_voice_parse_services(root,services,&count));
    TEST_ASSERT_EQUAL_UINT(1,count);TEST_ASSERT_EQUAL_STRING(voice_service,services[0].id);
    const char *bad[]={
        "https://metadata.example/api/v1/assistant/nyxagent",
        "https://metadata.example/api/v1/proxy/nyxbot-voice/bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
        "https://metadata.example/api/v1/proxy/nyxbot-voice/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee?next=bad",
        "https://metadata.example/api/v1/proxy/nyxbot-voice/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/extra",
        "https://user:secret@metadata.example/api/v1/proxy/nyxbot-voice/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "/api/v1/proxy/nyxbot-voice/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    };
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);++i){
        cJSON_ReplaceItemInObjectCaseSensitive(row,"endpoint_url",cJSON_CreateString(bad[i]));
        TEST_ASSERT_FALSE(panel_voice_parse_services(root,services,&count));
    }
    cJSON_DeleteItemFromArray(rows,2);cJSON_AddItemToArray(rows,voice_inventory_row(voice_service,"nyxbot-voice",true));
    cJSON_AddItemToArray(rows,voice_inventory_row(voice_service,"nyxbot-voice",true));
    TEST_ASSERT_FALSE(panel_voice_parse_services(root,services,&count));
    cJSON_DeleteItemFromArray(rows,3);row=cJSON_GetArrayItem(rows,2);
    cJSON_ReplaceItemInObjectCaseSensitive(row,"id",cJSON_CreateString("../../other"));
    TEST_ASSERT_FALSE(panel_voice_parse_services(root,services,&count));
    cJSON_Delete(root);
    root=cJSON_Parse("{\"keys\":[]}");
    TEST_ASSERT_TRUE(panel_voice_parse_services(root,services,&count));TEST_ASSERT_EQUAL_UINT(0,count);cJSON_Delete(root);
    TEST_ASSERT_FALSE(panel_voice_parse_services(NULL,services,&count));
    root=cJSON_CreateObject();rows=cJSON_AddArrayToObject(root,"keys");
    for(unsigned i=0;i<=PANEL_VOICE_MAX_BOTS;++i){
        char id[40];snprintf(id,sizeof(id),"%08x-bbbb-cccc-dddd-eeeeeeeeeeee",i);
        cJSON_AddItemToArray(rows,voice_inventory_row(id,"nyxbot-voice",true));
    }
    TEST_ASSERT_FALSE(panel_voice_parse_services(root,services,&count));cJSON_Delete(root);
}
void test_bot_discovery_requires_one_bot_bound_to_the_discovered_service(void)
{
    panel_voice_bot_t bot;
    cJSON *root=cJSON_Parse("{\"agents\":[{\"service_id\":\"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\","
        "\"agent_id\":\"11111111-2222-3333-4444-555555555555\","
        "\"conversation_id\":\"nyxa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"name\":\"dynamic-bot\",\"display_name\":\"My\\nBot\"}]}");
    TEST_ASSERT_TRUE(panel_voice_parse_service_bot(root,voice_service,&bot));
    TEST_ASSERT_EQUAL_STRING("My Bot",bot.name);TEST_ASSERT_EQUAL_STRING(bot_id,bot.agent_id);
    TEST_ASSERT_EQUAL_STRING(voice_service,bot.service_id);
    TEST_ASSERT_FALSE(panel_voice_parse_service_bot(root,other_service,&bot));
    cJSON *rows=cJSON_GetObjectItemCaseSensitive(root,"agents"), *row=cJSON_GetArrayItem(rows,0);
    cJSON_DeleteItemFromObjectCaseSensitive(row,"service_id");
    TEST_ASSERT_FALSE(panel_voice_parse_service_bot(root,voice_service,&bot));
    cJSON_AddStringToObject(row,"service_id",voice_service);
    cJSON_ReplaceItemInObjectCaseSensitive(row,"conversation_id",cJSON_CreateString("../../another/conversation"));
    TEST_ASSERT_FALSE(panel_voice_parse_service_bot(root,voice_service,&bot));
    cJSON_ReplaceItemInObjectCaseSensitive(row,"conversation_id",cJSON_CreateString("nyxa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    cJSON_AddItemToArray(rows,cJSON_Duplicate(row,true));
    TEST_ASSERT_FALSE(panel_voice_parse_service_bot(root,voice_service,&bot));
    cJSON_DeleteItemFromArray(rows,1);cJSON_DeleteItemFromArray(rows,0);
    TEST_ASSERT_FALSE(panel_voice_parse_service_bot(root,voice_service,&bot));cJSON_Delete(root);
}
void test_voice_selection_migrates_only_unique_current_key_binding(void)
{
    panel_voice_bot_t bots[2]={0}, saved={0}, result={0};
    strcpy(bots[0].service_id,voice_service);strcpy(bots[0].agent_id,bot_id);
    strcpy(bots[0].conversation_id,"nyxa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");strcpy(bots[0].name,"Current name");
    saved=bots[0];saved.service_id[0]=0;strcpy(saved.name,"Old name");
    TEST_ASSERT_TRUE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots,1,&result));
    TEST_ASSERT_EQUAL_STRING(voice_service,result.service_id);TEST_ASSERT_EQUAL_STRING("Current name",result.name);
    TEST_ASSERT_FALSE(panel_voice_reconcile_bot(&saved,key_id,bot_id,bots,1,&result));
    bots[1]=bots[0];strcpy(bots[1].service_id,other_service);
    TEST_ASSERT_FALSE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots,2,&result));
    saved=bots[0];
    TEST_ASSERT_FALSE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots+1,1,&result));
    TEST_ASSERT_TRUE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots,1,&result));
    strcpy(saved.conversation_id,"nyxa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    TEST_ASSERT_FALSE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots,1,&result));
    saved=bots[0];memset(saved.service_id,'x',sizeof(saved.service_id));
    TEST_ASSERT_FALSE(panel_voice_bot_valid(&saved,true));
    TEST_ASSERT_FALSE(panel_voice_reconcile_bot(&saved,key_id,key_id,bots,1,&result));
}
void test_voice_urls_pin_configured_origin_service_and_bounded_operation(void)
{
    char url[512],operation[160],expected[512];
    TEST_ASSERT_TRUE(panel_voice_keys_url(url,sizeof(url),"https://nyxid.example/"));
    TEST_ASSERT_EQUAL_STRING("https://nyxid.example/api/v1/keys",url);
    TEST_ASSERT_TRUE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example",voice_service,"/voice/agents",false));
    TEST_ASSERT_EQUAL_STRING("https://nyxid.example/api/v1/proxy/nyxbot-voice/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/voice/agents",url);
    const char *ends[]={"","/control","/announcements","/stream"};
    for(size_t i=0;i<sizeof(ends)/sizeof(*ends);++i){
        snprintf(operation,sizeof(operation),"/voice/sessions/%s%s",bot_id,ends[i]);
        TEST_ASSERT_TRUE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example/",voice_service,operation,true));
        snprintf(expected,sizeof(expected),"wss://nyxid.example/api/v1/proxy/nyxbot-voice/%s%s",voice_service,operation);
        TEST_ASSERT_EQUAL_STRING(expected,url);TEST_ASSERT_NULL(strstr(url,"assistant/nyxagent"));
    }
    TEST_ASSERT_TRUE(panel_voice_service_url(url,sizeof(url),"http://192.168.1.2:3001",voice_service,operation,true));
    TEST_ASSERT_NOT_NULL(strstr(url,"ws://192.168.1.2:3001/api/v1/proxy/nyxbot-voice/"));
    TEST_ASSERT_TRUE(panel_voice_service_url(url,sizeof(url),"http://192.168.1.2:3001",voice_service,"/voice/options?conversation_id=nyxa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",false));
    TEST_ASSERT_TRUE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example",voice_service,"/voice/sessions",false));
    const char *bad[]={"/voice/sessions/../../other/stream","/voice/agents?next=bad","/voice/options?conversation_id=../../bad",
        "/voice/sessions/11111111-2222-3333-4444-555555555555/stream#bad","https://other.example/voice/agents"};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);++i)
        TEST_ASSERT_FALSE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example",voice_service,bad[i],false));
    TEST_ASSERT_FALSE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example", "../bad","/voice/agents",false));
    TEST_ASSERT_FALSE(panel_voice_service_url(url,sizeof(url),"https://name@nyxid.example",voice_service,"/voice/agents",false));
    TEST_ASSERT_FALSE(panel_voice_service_url(url,sizeof(url),"https://nyxid.example/api/v1",voice_service,"/voice/agents",false));
    TEST_ASSERT_FALSE(panel_voice_service_url(url,10,"https://nyxid.example",voice_service,"/voice/agents",false));
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
    TEST_ASSERT_NOT_NULL(strstr(panel_voice_error_message(NULL, 404), "Refresh Bots"));
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
    RUN_TEST(test_voice_inventory_requires_active_catalog_and_exact_service_path);
    RUN_TEST(test_bot_discovery_requires_one_bot_bound_to_the_discovered_service);
    RUN_TEST(test_voice_selection_migrates_only_unique_current_key_binding);
    RUN_TEST(test_voice_urls_pin_configured_origin_service_and_bounded_operation);
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
