#include "panel_voice_proto.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

const char *panel_voice_input_mode_name(panel_voice_input_mode_t mode)
{
    switch (mode) {
    case PANEL_VOICE_INPUT_HOLD: return "push_to_talk";
    case PANEL_VOICE_INPUT_AUTOMATIC: return "automatic";
    default: return NULL;
    }
}
bool panel_voice_capture_allowed(panel_voice_input_mode_t mode, bool held,
    bool local_muted, bool controls, bool connected, bool ready, bool server_muted)
{
    return panel_voice_input_mode_name(mode) && !local_muted && controls && connected && ready &&
        !server_muted && (mode == PANEL_VOICE_INPUT_AUTOMATIC || held);
}

static bool valid_speech_text(const unsigned char *p)
{
    bool nonspace=false;
    while (*p) {
        unsigned cp=*p++, extra=0;
        if(cp>=0xc2 && cp<=0xdf){cp&=31;extra=1;}
        else if(cp>=0xe0 && cp<=0xef){cp&=15;extra=2;}
        else if(cp>=0xf0 && cp<=0xf4){cp&=7;extra=3;}
        else if(cp>=0x80)return false;
        for(unsigned i=0;i<extra;++i){if((*p&0xc0)!=0x80)return false;cp=(cp<<6)|(*p++&63);}
        if((extra==1 && cp<0x80) || (extra==2 && cp<0x800) || (extra==3 && cp<0x10000) ||
            cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff) || cp<32 || (cp>=127 && cp<=159))return false;
        if(cp!=32 && cp!=0xa0 && cp!=0x3000 && !(cp>=0x2000 && cp<=0x200a))nonspace=true;
    }
    return nonspace;
}

bool panel_voice_parse_remote(const void *json, size_t size, bool open, panel_voice_remote_t *out)
{
    if (!json || !out || !size || size>1024 || memchr(json,0,size))return false;
    memset(out,0,sizeof(*out));
    const char *end=NULL;
    cJSON *root=cJSON_ParseWithLengthOpts(json,size,&end,false);
    bool ok=cJSON_IsObject(root);unsigned fields=0;
    if (end)while((size_t)(end-(const char *)json)<size && (*end==' ' || *end=='\n' || *end=='\r' || *end=='\t'))++end;
    if(!end || (size_t)(end-(const char *)json)!=size)ok=false;
    const cJSON *entry;
    cJSON_ArrayForEach(entry,root){
        unsigned field=entry->string && !strcmp(entry->string,"request_id")?1:
            entry->string && !strcmp(entry->string,"text")?2:
            open && entry->string && !strcmp(entry->string,"microphone_muted")?4:0;
        if(!field || (fields&field)){ok=false;break;}fields|=field;
        if(field==1){
            if(!cJSON_IsString(entry) || !panel_voice_uuid(entry->valuestring)){ok=false;break;}
            strcpy(out->request_id,entry->valuestring);
        }else if(field==2){
            if(!cJSON_IsString(entry) || !*entry->valuestring || strlen(entry->valuestring)>300){ok=false;break;}
            if(!valid_speech_text((const unsigned char *)entry->valuestring))ok=false;
            strcpy(out->text,entry->valuestring);
        }else{
            if(!cJSON_IsBool(entry))ok=false;
            out->microphone_muted=cJSON_IsTrue(entry);
        }
    }
    // cJSON's C strings cannot represent embedded JSON NULs safely.
    const char *raw=json;
    for(size_t i=0;i+6<=size;++i)if(!memcmp(raw+i,"\\u0000",6))ok=false;
    ok=ok && (fields&1) && (open || (fields&2));
    cJSON_Delete(root);return ok;
}

bool panel_voice_parse_greeting(const void *json, size_t size,
    const panel_voice_greeting_t *current, panel_voice_greeting_t *out)
{
    if (!json || !current || !out || !size || size>1024 || memchr(json,0,size)) return false;
    const char *raw=json,*end=NULL;
    cJSON *root=cJSON_ParseWithLengthOpts(raw,size,&end,false);
    if(end)while((size_t)(end-raw)<size && strchr(" \t\r\n",*end))++end;
    const cJSON *value=cJSON_GetObjectItemCaseSensitive(root,"enabled");
    bool ok=cJSON_IsObject(root) && cJSON_GetArraySize(root)==1 && cJSON_IsBool(value) &&
        end && (size_t)(end-raw)==size;
    for(size_t i=0;i+6<=size;++i)if(!memcmp(raw+i,"\\u0000",6))ok=false;
    if(ok){*out=*current;out->enabled=cJSON_IsTrue(value);}
    cJSON_Delete(root);return ok;
}

bool panel_voice_uuid(const char *value)
{
    if (!value || strlen(value) != 36) return false;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!strchr("0123456789abcdefABCDEF", value[i])) return false;
    }
    return true;
}
bool panel_voice_conversation(const char *value)
{
    if (!value || strlen(value) != 37 || strncmp(value, "nyxa-", 5)) return false;
    for (const char *p = value + 5; *p; ++p)
        if (!strchr("0123456789abcdef", *p)) return false;
    return true;
}
static const char *string(const cJSON *root, const char *field)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, field);
    return cJSON_IsString(item) ? item->valuestring : NULL;
}
const char *panel_voice_error_message(const cJSON *root, int http_status)
{
    if (http_status == 401) return "Key expired or revoked. Sign in again in Settings";
    if (http_status == 403) return "Allow this Bot's voice connection on the panel Key in NyxID";
    if (http_status == 404) return "Bot voice connection is unavailable. Refresh Bots in Settings";
    const cJSON *details = cJSON_GetObjectItemCaseSensitive(root, "details");
    const char *stage = string(details, "stage"), *code = string(details, "provider_code");
    const char *type = string(details, "provider_type");
    if (stage && !strcmp(stage, "provider_create")) {
        if ((code && (!strcmp(code, "credit_balance_exhausted") || !strcmp(code, "insufficient_quota"))) ||
            (type && !strcmp(type, "insufficient_quota")))
            return "Voice provider has no credits. Update the provider connection in NyxID";
        const cJSON *status = cJSON_GetObjectItemCaseSensitive(details, "provider_status");
        if (cJSON_IsNumber(status) && status->valueint == 429)
            return "Voice provider rate limit reached. Try again later";
        return "Voice provider rejected the session. Check voice settings in NyxID";
    }
    if (stage && !strcmp(stage, "credential"))
        return "Voice provider is unavailable or not authorized for this Bot in NyxID";
    if (http_status == 409) return "End the current voice call in NyxID, then try again";
    return NULL;
}
// Only an origin is accepted; callers cannot redirect a stored Agent Key with a path.
static size_t origin_length(const char *url)
{
    if (!url) return 0;
    size_t scheme = !strncmp(url,"https://",8) ? 8 : !strncmp(url,"http://",7) ? 7 : 0;
    if (!scheme) return 0;
    const char *p=url+scheme;
    for (; *p && *p!='/'; ++p)
        if ((unsigned char)*p<=32 || (unsigned char)*p>=127 || strchr("@?#\\",*p)) return 0;
    if (p==url+scheme || (*p && strcmp(p,"/"))) return 0;
    return (size_t)(p-url);
}
static bool formatted(int size, size_t cap) { return size>=0 && (size_t)size<cap; }
bool panel_voice_keys_url(char *out, size_t cap, const char *base)
{
    size_t n=origin_length(base);
    if (!out || !cap || !n || n>400) return false;
    return formatted(snprintf(out,cap,"%.*s/api/v1/keys",(int)n,base),cap);
}
static bool voice_operation(const char *operation)
{
    if (!operation) return false;
    if (!strcmp(operation,"") || !strcmp(operation,"/voice/agents") ||
        !strcmp(operation,"/voice/options") || !strcmp(operation,"/voice/sessions")) return true;
    const char *query="/voice/options?conversation_id=";
    if (!strncmp(operation,query,strlen(query))) return panel_voice_conversation(operation+strlen(query));
    const char *prefix="/voice/sessions/";
    if (strncmp(operation,prefix,strlen(prefix))) return false;
    const char *session=operation+strlen(prefix);
    if (strlen(session)<36) return false;
    char id[37];memcpy(id,session,36);id[36]=0;
    return panel_voice_uuid(id) && (!strcmp(session+36,"") || !strcmp(session+36,"/stream") ||
        !strcmp(session+36,"/control") || !strcmp(session+36,"/announcements"));
}
bool panel_voice_service_url(char *out, size_t cap, const char *base,
    const char *service_id, const char *operation, bool websocket)
{
    size_t n=origin_length(base);
    if (!out || !cap || !n || n>400 || !panel_voice_uuid(service_id) || !voice_operation(operation)) return false;
    bool secure=!strncmp(base,"https://",8);
    size_t skip=websocket?(secure?5:4):0;
    return formatted(snprintf(out,cap,"%s%.*s/api/v1/proxy/nyxbot-voice/%s%s",
        websocket?(secure?"wss":"ws"):"",(int)(n-skip),base+skip,service_id,operation),cap);
}
static bool service_endpoint(const char *endpoint, const char *id)
{
    if (!endpoint) return false;
    const char *path=strstr(endpoint,"://");
    if (!path) return false;
    path=strchr(path+3,'/');
    if (!path) return false;
    size_t n=(size_t)(path-endpoint);
    if (n>=400) return false;
    char origin[400];memcpy(origin,endpoint,n);origin[n]=0;
    if (!origin_length(origin)) return false;
    char expected[96];snprintf(expected,sizeof(expected),"/api/v1/proxy/nyxbot-voice/%s",id);
    return !strcmp(path,expected);
}
bool panel_voice_parse_services(const cJSON *root, panel_voice_service_t *services, size_t *count)
{
    if (!services || !count) return false;
    *count = 0;
    const cJSON *rows=cJSON_GetObjectItemCaseSensitive(root,"keys"), *row;
    if (!cJSON_IsArray(rows)) return false;
    cJSON_ArrayForEach(row,rows) {
        const char *slug=string(row,"catalog_service_slug");
        if (!slug || strcmp(slug,"nyxbot-voice") || !cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(row,"is_active"))) continue;
        const char *id=string(row,"id");
        if (!panel_voice_uuid(id) || !service_endpoint(string(row,"endpoint_url"),id) ||
            *count==PANEL_VOICE_MAX_BOTS) return false;
        for (size_t i=0;i<*count;++i) if (!strcmp(services[i].id,id)) return false;
        snprintf(services[(*count)++].id,sizeof(services[0].id),"%s",id);
    }
    return true;
}
bool panel_voice_bot_valid(const panel_voice_bot_t *bot, bool legacy)
{
    return bot && memchr(bot->service_id,0,sizeof(bot->service_id)) &&
        memchr(bot->agent_id,0,sizeof(bot->agent_id)) &&
        memchr(bot->conversation_id,0,sizeof(bot->conversation_id)) &&
        memchr(bot->name,0,sizeof(bot->name)) &&
        ((legacy && !bot->service_id[0]) || panel_voice_uuid(bot->service_id)) &&
        panel_voice_uuid(bot->agent_id) && panel_voice_conversation(bot->conversation_id);
}
bool panel_voice_parse_service_bot(const cJSON *root, const char *service_id, panel_voice_bot_t *bot)
{
    if (!bot || !panel_voice_uuid(service_id)) return false;
    const cJSON *rows = cJSON_GetObjectItemCaseSensitive(root, "agents");
    if (!cJSON_IsArray(rows) || cJSON_GetArraySize(rows)!=1) return false;
    const cJSON *row=cJSON_GetArrayItem(rows,0);
    const char *service=string(row,"service_id"), *id=string(row,"agent_id"), *thread=string(row,"conversation_id");
    const char *name=string(row,"display_name");
    if (!name || !*name) name=string(row,"name");
    if (!service || strcmp(service,service_id) || !panel_voice_uuid(id) || !panel_voice_conversation(thread) ||
        !name || !*name || strlen(name)>=sizeof(bot->name)) return false;
    memset(bot,0,sizeof(*bot));
    strcpy(bot->service_id,service);strcpy(bot->agent_id,id);strcpy(bot->conversation_id,thread);strcpy(bot->name,name);
    for (char *p=bot->name;*p;++p) if (*p=='\n' || *p=='\r') *p=' ';
    return true;
}
bool panel_voice_reconcile_bot(const panel_voice_bot_t *saved, const char *saved_key,
    const char *current_key, const panel_voice_bot_t *bots, size_t count, panel_voice_bot_t *out)
{
    if (!out || !bots || count>PANEL_VOICE_MAX_BOTS || !panel_voice_bot_valid(saved,true) ||
        !panel_voice_uuid(saved_key) || !panel_voice_uuid(current_key) || strcmp(saved_key,current_key)) return false;
    const panel_voice_bot_t *match=NULL;
    for (size_t i=0;i<count;++i) {
        if (!panel_voice_bot_valid(&bots[i],false) || strcmp(saved->agent_id,bots[i].agent_id) ||
            strcmp(saved->conversation_id,bots[i].conversation_id) ||
            (saved->service_id[0] && strcmp(saved->service_id,bots[i].service_id))) continue;
        if (match) return false;
        match=&bots[i];
    }
    if (!match) return false;
    *out=*match;
    return true;
}
static bool same(const cJSON *a, const cJSON *b, const char *field)
{
    const cJSON *av = cJSON_GetObjectItemCaseSensitive(a, field), *bv = cJSON_GetObjectItemCaseSensitive(b, field);
    if ((!av || cJSON_IsNull(av)) && (!bv || cJSON_IsNull(bv))) return true;
    return cJSON_IsString(av) && cJSON_IsString(bv) && !strcmp(av->valuestring, bv->valuestring);
}
cJSON *panel_voice_preferences(const cJSON *discovery, const cJSON *options,
    panel_voice_input_mode_t mode)
{
    const char *input_mode = panel_voice_input_mode_name(mode);
    if (!input_mode) return NULL;
    const cJSON *saved = cJSON_GetObjectItemCaseSensitive(discovery, "preferences");
    const cJSON *choices = cJSON_GetObjectItemCaseSensitive(options, "options"), *option;
    if (!cJSON_IsObject(saved) || !cJSON_IsArray(choices)) return NULL;
    const char *service = string(saved, "service_id"), *model = string(saved, "model");
    const char *source = string(saved, "key_source"), *connection = string(saved, "connection_id");
    if (!panel_voice_uuid(service) || !model || !*model || strlen(model) > 200 || !source ||
        (strcmp(source, "platform") && strcmp(source, "own")) ||
        (!strcmp(source, "own") && !panel_voice_uuid(connection)) ||
        (!strcmp(source, "platform") && connection)) return NULL;
    cJSON_ArrayForEach(option, choices) {
        const cJSON *metadata = cJSON_GetObjectItemCaseSensitive(option, "voice");
        const char *protocol = string(metadata, "protocol");
        if (!cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(option, "available")) ||
            !protocol || strcmp(protocol, "openai_live") ||
            !same(saved, option, "service_id") || !same(saved, option, "connection_id") ||
            !same(saved, option, "key_source") || !same(saved, option, "model")) continue;
        const char *voice = string(saved, "voice");
        bool valid_voice = !voice;
        const cJSON *voices = cJSON_GetObjectItemCaseSensitive(metadata, "voices"), *entry;
        cJSON_ArrayForEach(entry, voices) {
            const char *id = string(entry, "id");
            if (id && voice && !strcmp(id, voice)) valid_voice = true;
        }
        if (!valid_voice) return NULL;
        cJSON *result = cJSON_Duplicate(saved, true);
        if (result) {
            cJSON_DeleteItemFromObjectCaseSensitive(result, "input_mode");
            cJSON_AddStringToObject(result, "input_mode", input_mode);
            cJSON_DeleteItemFromObjectCaseSensitive(result, "notify_on_completion");
            cJSON_AddBoolToObject(result, "notify_on_completion", false);
        }
        return result;
    }
    return NULL;
}

static size_t utf8_prefix(const char *text, size_t limit)
{
    size_t n = strlen(text);
    if (n <= limit) return n;
    n = limit;
    while (n && ((unsigned char)text[n] & 0xc0) == 0x80) --n;
    return n;
}

void panel_voice_parse_description(const cJSON *root, char out[PANEL_VOICE_DESCRIPTION_BYTES])
{
    out[0]=0;
    const cJSON *rows=cJSON_GetObjectItemCaseSensitive(root,"agents");
    if (!cJSON_IsArray(rows) || cJSON_GetArraySize(rows)!=1) return;
    const char *text=string(cJSON_GetArrayItem(rows,0),"description");
    if (!text) return;
    size_t n=utf8_prefix(text,PANEL_VOICE_DESCRIPTION_BYTES-1);
    memcpy(out,text,n);out[n]=0;
    for (size_t i=0;i<n;++i) if ((unsigned char)out[i]<32 || out[i]==127) out[i]=' ';
}
bool panel_voice_parse_captions(const cJSON *snapshot, panel_voice_captions_t *out)
{
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    const cJSON *rows = cJSON_GetObjectItemCaseSensitive(snapshot, "captions"), *row;
    if (!cJSON_IsArray(rows) || cJSON_GetArraySize(rows) > 64) return false;
    const char *texts[2] = {NULL, NULL};
    double starts[2] = {-1, -1};
    cJSON_ArrayForEach(row, rows) {
        const char *speaker = string(row, "speaker"), *text = string(row, "text");
        if (!speaker || !text || !*text || strlen(text) > 4096) continue;
        int index = !strcmp(speaker, "user") ? 0 : !strcmp(speaker, "assistant") ? 1 : -1;
        if (index < 0) continue;
        const cJSON *start = cJSON_GetObjectItemCaseSensitive(row, "start_ms");
        double time = cJSON_IsNumber(start) ? start->valuedouble : starts[index];
        if (time < starts[index]) continue;
        starts[index] = time; texts[index] = text;
    }
    if (!texts[0] && !texts[1]) return false;
    char *destinations[2] = {out->user, out->bot};
    for (int i = 0; i < 2; ++i) {
        if (!texts[i]) continue;
        while (*texts[i] == ' ' || *texts[i] == '\n' || *texts[i] == '\r' || *texts[i] == '\t') ++texts[i];
        size_t n = utf8_prefix(texts[i], 509);
        memcpy(destinations[i], texts[i], n);
        if (texts[i][n]) memcpy(destinations[i] + n, "…", 3);
    }
    return true;
}

// RFC 5763 initial offers use actpass. The peer library emits passive; some
// endpoints return a passive answer as well, leaving contradictory DTLS roles.
// Normalize setup lines only, retaining all ICE credentials/fingerprints verbatim.
char *panel_voice_prepare_offer(const char *sdp, size_t size)
{
    if (!sdp || !size || size > 64*1024 || memchr(sdp, 0, size)) return NULL;
    size_t output_size = size, setups = 0;
    for (size_t i=0; i<size;) {
        size_t end=i; while(end<size && sdp[end]!='\r' && sdp[end]!='\n') ++end;
        size_t len=end-i;
        if (len>=8 && !memcmp(sdp+i,"a=setup:",8)) {
            if (len==14 && !memcmp(sdp+i+8,"active",6)) ++output_size;
            else if (len!=15 || (memcmp(sdp+i+8,"passive",7) && memcmp(sdp+i+8,"actpass",7))) return NULL;
            ++setups;
        }
        i=end; while(i<size && (sdp[i]=='\r' || sdp[i]=='\n')) ++i;
    }
    if (!setups || output_size>64*1024) return NULL;
    char *out=malloc(output_size+1); if(!out) return NULL;
    size_t used=0;
    for (size_t i=0; i<size;) {
        size_t end=i; while(end<size && sdp[end]!='\r' && sdp[end]!='\n') ++end;
        size_t len=end-i;
        if (len>=8 && !memcmp(sdp+i,"a=setup:",8)) {
            memcpy(out+used,"a=setup:actpass",15); used+=15;
        } else {memcpy(out+used,sdp+i,len);used+=len;}
        i=end; while(i<size && (sdp[i]=='\r' || sdp[i]=='\n')) out[used++]=sdp[i++];
    }
    out[used]=0; return out;
}

bool panel_voice_provider_started(const char *json, size_t size)
{
    if (!json || !size || size>96*1024 || memchr(json,0,size)) return false;
    cJSON *event=cJSON_ParseWithLength(json,size);
    const cJSON *type=cJSON_GetObjectItemCaseSensitive(event,"type");
    bool started=cJSON_IsString(type) && !strcmp(type->valuestring,"session.started");
    cJSON_Delete(event);return started;
}

panel_voice_snapshot_result_t panel_voice_snapshot_result(const cJSON *root,
    const char *session_id)
{
    if (!cJSON_IsObject(root) || !session_id) return PANEL_VOICE_SNAPSHOT_INVALID;
    const cJSON *type=cJSON_GetObjectItemCaseSensitive(root,"type");
    if (!cJSON_IsString(type) || strcmp(type->valuestring,"snapshot"))
        return PANEL_VOICE_SNAPSHOT_INVALID;
    const cJSON *session=cJSON_GetObjectItemCaseSensitive(root,"session");
    if (!cJSON_IsObject(session)) return PANEL_VOICE_SNAPSHOT_INVALID;
    const cJSON *id=cJSON_GetObjectItemCaseSensitive(session,"id");
    const cJSON *state=cJSON_GetObjectItemCaseSensitive(session,"state");
    if (!cJSON_IsString(id) ||
        strcmp(id->valuestring,session_id) || !cJSON_IsString(state))
        return PANEL_VOICE_SNAPSHOT_INVALID;
    if (!strcmp(state->valuestring,"closed") || !strcmp(state->valuestring,"failed") ||
        !strcmp(state->valuestring,"closing")) return PANEL_VOICE_SNAPSHOT_TERMINAL;
    if (strcmp(state->valuestring,"starting") && strcmp(state->valuestring,"active"))
        return PANEL_VOICE_SNAPSHOT_INVALID;
    const char *numbers[]={"generation","control_revision"};
    for (size_t i=0;i<2;i++) {
        const cJSON *n=cJSON_GetObjectItemCaseSensitive(session,numbers[i]);
        if (!cJSON_IsNumber(n) || !(n->valuedouble>=(i?0:1) && n->valuedouble<=UINT_MAX) ||
            n->valuedouble!=(double)(unsigned)n->valuedouble)
            return PANEL_VOICE_SNAPSHOT_INVALID;
    }
    if (!cJSON_IsBool(cJSON_GetObjectItemCaseSensitive(session,"muted")) ||
        !cJSON_IsBool(cJSON_GetObjectItemCaseSensitive(session,"input_muted")))
        return PANEL_VOICE_SNAPSHOT_INVALID;
    return PANEL_VOICE_SNAPSHOT_LIVE;
}

bool panel_voice_normal_end(const cJSON *session)
{
    const char *state=string(session,"state"),*reason=string(session,"end_reason");
    return state && !strcmp(state,"closed") && reason &&
        (!strcmp(reason,"idle") || !strcmp(reason,"time_limit") || !strcmp(reason,"user_ended"));
}
