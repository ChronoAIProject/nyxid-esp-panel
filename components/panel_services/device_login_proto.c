#include "device_login_proto.h"
#include "cJSON.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

static bool copy(cJSON *root, const char *key, char *out, size_t cap)
{
    const cJSON *value = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsString(value) || !value->valuestring || !value->valuestring[0] ||
        strlen(value->valuestring) >= cap) return false;
    strcpy(out, value->valuestring);
    return true;
}

bool device_login_parse_request(const char *json, device_login_request_t *out)
{
    return device_login_parse_request_at(json, "https://nyxid.ai/login/device", out);
}

bool device_login_parse_request_at(const char *json, const char *verification_uri,
                                  device_login_request_t *out)
{
    if (!json || !verification_uri || !verification_uri[0] || !out) return false;
    memset(out, 0, sizeof(*out));
    cJSON *root = cJSON_Parse(json);
    const cJSON *expires = cJSON_GetObjectItemCaseSensitive(root, "expires_in");
    const cJSON *interval = cJSON_GetObjectItemCaseSensitive(root, "interval");
    bool ok = cJSON_IsObject(root) && copy(root, "device_code", out->device_code, sizeof(out->device_code)) &&
        copy(root, "user_code", out->user_code, sizeof(out->user_code)) &&
        copy(root, "verification_uri", out->verification_uri, sizeof(out->verification_uri)) &&
        copy(root, "verification_uri_complete", out->verification_uri_complete, sizeof(out->verification_uri_complete)) &&
        cJSON_IsNumber(expires) && expires->valuedouble >= 1 && expires->valuedouble <= 3600 &&
        expires->valuedouble == floor(expires->valuedouble) &&
        cJSON_IsNumber(interval) && interval->valuedouble >= 1 && interval->valuedouble <= 300 &&
        interval->valuedouble == floor(interval->valuedouble);
    char expected[sizeof(out->verification_uri_complete)];
    for (const char *p=out->user_code;*p;p++)
        if (!((*p>='A' && *p<='Z') || (*p>='0' && *p<='9') || *p=='-')) ok=false;
    snprintf(expected,sizeof(expected),"%s?user_code=%s",verification_uri,out->user_code);
    ok = ok && !strcmp(out->verification_uri, verification_uri) &&
        !strcmp(out->verification_uri_complete, expected);
    if (ok) {
        out->expires_in = (uint32_t)expires->valuedouble;
        out->interval = (uint32_t)interval->valuedouble;
    } else memset(out, 0, sizeof(*out));
    cJSON_Delete(root);
    return ok;
}

device_login_poll_t device_login_parse_poll(int status, const char *json)
{
    cJSON *root = json ? cJSON_Parse(json) : NULL;
    device_login_poll_t result = LOGIN_POLL_ERROR;
    const cJSON *kind = cJSON_GetObjectItemCaseSensitive(root, "auth_kind");
    const cJSON *token = cJSON_GetObjectItemCaseSensitive(root, "access_token");
    if (status == 200 && cJSON_IsString(kind) && !strcmp(kind->valuestring, "account_session") &&
        cJSON_IsString(token) && token->valuestring[0]) result = LOGIN_POLL_APPROVED;
    const cJSON *code = cJSON_GetObjectItemCaseSensitive(root, "error_code");
    if (status != 200 && cJSON_IsNumber(code)) {
        switch (code->valueint) {
        case 11202: if (status == 400) result = LOGIN_POLL_PENDING; break;
        case 11203: if (status == 429) result = LOGIN_POLL_SLOW_DOWN; break;
        case 11206: if (status == 429) result = LOGIN_POLL_RATE_LIMITED; break;
        case 11204: if (status == 403) result = LOGIN_POLL_DENIED; break;
        case 11201: if (status == 410) result = LOGIN_POLL_EXPIRED; break;
        case 11205: if (status == 410) result = LOGIN_POLL_DELIVERED; break;
        }
    }
    cJSON_Delete(root);
    return result;
}

uint32_t device_login_backoff(uint32_t interval)
{
    return interval >= 295 ? 300 : interval + 5;
}

bool device_agent_parse_request(const char *json, device_login_request_t *out)
{
    return device_agent_parse_request_at(json, "https://nyxid.ai/login/agent-key", out);
}

bool device_agent_parse_request_at(const char *json, const char *verification_uri,
                                   device_login_request_t *out)
{
    if (!json || !verification_uri || !verification_uri[0] || !out) return false;
    memset(out,0,sizeof(*out));
    cJSON *root=cJSON_Parse(json);
    const cJSON *expires=cJSON_GetObjectItemCaseSensitive(root,"expires_in");
    const cJSON *interval=cJSON_GetObjectItemCaseSensitive(root,"interval");
    bool ok=copy(root,"device_code",out->device_code,sizeof(out->device_code)) &&
        copy(root,"user_code",out->user_code,sizeof(out->user_code)) &&
        copy(root,"verification_uri",out->verification_uri,sizeof(out->verification_uri)) &&
        !strcmp(out->verification_uri,verification_uri) &&
        cJSON_IsNumber(expires) && expires->valuedouble>=1 && expires->valuedouble<=3600 && expires->valuedouble==floor(expires->valuedouble) &&
        cJSON_IsNumber(interval) && interval->valuedouble>=1 && interval->valuedouble<=300 && interval->valuedouble==floor(interval->valuedouble);
    if (ok) {
        // The Agent Key API returns the bare verification URI, not a complete one.
        for (const char *p=out->user_code;*p;p++) if (!((*p>='A' && *p<='Z') || (*p>='0' && *p<='9') || *p=='-')) ok=false;
        if (ok) {
            snprintf(out->verification_uri_complete,sizeof(out->verification_uri_complete),"%s?user_code=%s",out->verification_uri,out->user_code);
            out->expires_in=(uint32_t)expires->valuedouble;out->interval=(uint32_t)interval->valuedouble;
        }
    }
    if (!ok) memset(out,0,sizeof(*out));
    cJSON_Delete(root);return ok;
}
device_login_poll_t device_agent_parse_poll(int status, const char *json)
{
    cJSON *root=json?cJSON_Parse(json):NULL;
    const cJSON *secret=cJSON_GetObjectItemCaseSensitive(root,"credential");
    const cJSON *id=cJSON_GetObjectItemCaseSensitive(root,"credential_id");
    device_login_poll_t result=LOGIN_POLL_ERROR;
    if (status==200 && cJSON_IsString(secret) && strlen(secret->valuestring)==73 &&
        !strncmp(secret->valuestring,"nyxid_ag_",9) && cJSON_IsString(id) && strlen(id->valuestring)==36) {
        result=LOGIN_POLL_APPROVED;
        for (const char *p=secret->valuestring+9;*p;p++) if (!strchr("0123456789abcdef",*p)) result=LOGIN_POLL_ERROR;
    }
    const cJSON *code=cJSON_GetObjectItemCaseSensitive(root,"error_code");
    if (status!=200 && cJSON_IsNumber(code)) switch(code->valueint) {
        case 11902:if(status==400)result=LOGIN_POLL_PENDING;break;
        case 11903:if(status==429)result=LOGIN_POLL_SLOW_DOWN;break;
        case 11906:if(status==429)result=LOGIN_POLL_RATE_LIMITED;break;
        case 11904:if(status==403)result=LOGIN_POLL_DENIED;break;
        case 11901:if(status==410)result=LOGIN_POLL_EXPIRED;break;
        case 11905:if(status==410)result=LOGIN_POLL_DELIVERED;break;
    }
    // Clear the parser's transient credential copy before freeing it.
    if (cJSON_IsString(secret)) {volatile char *p=secret->valuestring;while(*p)*p++=0;}
    cJSON_Delete(root);return result;
}
