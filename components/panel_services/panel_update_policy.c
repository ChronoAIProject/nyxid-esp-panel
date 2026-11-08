#include "panel_update_policy.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool version_parts(const char *text, unsigned parts[3])
{
    if (!text || !*text || strlen(text)>=32) return false;
    for (unsigned i=0;i<3;++i) {
        if (*text<'0' || *text>'9' || (*text=='0' && text[1]>='0' && text[1]<='9')) return false;
        char *end;
        unsigned long value=strtoul(text,&end,10);
        if (value>65535 || (i<2 ? *end!='.' : *end!='\0')) return false;
        parts[i]=(unsigned)value;
        text=i<2 ? end+1 : end;
    }
    return true;
}
bool panel_update_version_valid(const char *version)
{
    unsigned parts[3];
    return version_parts(version,parts);
}
bool panel_update_version_newer(const char *candidate, const char *running)
{
    unsigned a[3],b[3];
    if (!version_parts(candidate,a) || !version_parts(running,b)) return false;
    for (unsigned i=0;i<3;++i) if (a[i]!=b[i]) return a[i]>b[i];
    return false;
}
bool panel_update_download_url_allowed(const char *url)
{
    if (!url || strlen(url)>2048) return false;
    // GitHub may redirect release downloads to either of these asset hosts.
    const char *prefixes[]={PANEL_RELEASE_ROOT,"https://release-assets.githubusercontent.com/",
                            "https://objects.githubusercontent.com/"};
    for (size_t i=0;i<sizeof(prefixes)/sizeof(*prefixes);++i)
        if (!strncmp(url,prefixes[i],strlen(prefixes[i]))) return true;
    return false;
}
static const char *string(cJSON *root,const char *name)
{
    cJSON *value=cJSON_GetObjectItemCaseSensitive(root,name);
    return cJSON_IsString(value) ? value->valuestring : NULL;
}
static int hex(char c)
{
    return c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : -1;
}
bool panel_update_parse_release(const void *json,size_t length,const char *profile,panel_release_t *out)
{
    if (!json || !length || length>4096 || !profile || !out) return false;
    const char *end=NULL;
    cJSON *root=cJSON_ParseWithLengthOpts(json,length,&end,false);
    if (!root) return false;
    const char *limit=(const char *)json+length;
    while(end<limit && (*end==' ' || *end=='\n' || *end=='\r' || *end=='\t')) ++end;
    cJSON *schema=cJSON_GetObjectItemCaseSensitive(root,"schema");
    cJSON *size=cJSON_GetObjectItemCaseSensitive(root,"size");
    const char *project=string(root,"project"),*hardware=string(root,"profile");
    const char *version=string(root,"version"),*url=string(root,"url"),*sha=string(root,"sha256");
    bool valid=end==limit && cJSON_IsObject(root) && cJSON_IsNumber(schema) && schema->valuedouble==1 &&
        project && !strcmp(project,PANEL_RELEASE_PROJECT) && hardware && !strcmp(hardware,profile) &&
        panel_update_version_valid(version) && cJSON_IsNumber(size) && size->valuedouble>=1024 &&
        size->valuedouble<=PANEL_RELEASE_SLOT_SIZE && size->valuedouble==(double)(size_t)size->valuedouble &&
        sha && strlen(sha)==64 && url;
    char expected[256];
    if (valid) {
        int n=snprintf(expected,sizeof(expected),PANEL_RELEASE_ROOT "download/v%s/nyxid-esp-panel-%s.bin",version,profile);
        valid=n>0 && n<(int)sizeof(expected) && !strcmp(expected,url);
    }
    panel_release_t result={0};
    for (size_t i=0;valid && i<32;++i) {
        int a=hex(sha[i*2]),b=hex(sha[i*2+1]);
        if (a<0 || b<0) valid=false;
        else result.sha256[i]=(uint8_t)((a<<4)|b);
    }
    if (valid) {
        snprintf(result.version,sizeof(result.version),"%s",version);
        snprintf(result.url,sizeof(result.url),"%s",url);
        result.size=(size_t)size->valuedouble;
        *out=result;
    }
    cJSON_Delete(root);
    return valid;
}
