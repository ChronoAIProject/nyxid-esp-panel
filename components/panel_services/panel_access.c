#include "panel_access.h"
#include <stdio.h>
#include <string.h>

bool panel_management_slug(const char *daily, char *out, size_t cap)
{
    if (!daily || !*daily || !out || strlen(daily)>73) return false;
    int n=snprintf(out,cap,"%s-admin",daily);
    return n>0 && (size_t)n<cap;
}
bool panel_service_can_call(const char *daily, const char *slug, const char *method, const char *path)
{
    if (!daily || !*daily || !slug || !*slug || !method || !path) return false;
    char admin[80];
    if (panel_management_slug(daily,admin,sizeof(admin)) && !strcmp(slug,admin)) return true;
    if (strcmp(daily,slug)) return false;
    static const struct {const char *method,*path;} daily_routes[]={
        {"GET","/status"},{"GET","/openapi.json"},{"GET","/camera"},
        {"GET","/camera/snapshot.jpg"},{"GET","/snapshot.jpg"},
        {"POST","/camera/capture"},
        {"GET","/lcd/snapshot.jpg"},{"GET","/lcd/screen"},
        {"POST","/lcd/image"},{"POST","/lcd/text"},{"POST","/lcd/clear"},
        {"GET","/lcd/default-image"},{"POST","/lcd/default-image"},{"DELETE","/lcd/default-image"},
        {"GET","/lcd/brightness"},{"POST","/lcd/brightness"},
        {"GET","/config/audio"},{"POST","/config/audio"},{"POST","/voice/input-mode"},
        {"GET","/voice/greeting"},{"POST","/voice/greeting"},
        {"GET","/voice"},{"GET","/voice/agents"},{"POST","/voice/agents/refresh"},
        {"POST","/voice/open"},{"POST","/voice/start"},{"POST","/voice/stop"},
        {"POST","/voice/say"},{"POST","/voice/hold"},{"POST","/voice/microphone"},
        {"GET","/stream/status"},{"POST","/stream/stop"},
    };
    for(size_t i=0;i<sizeof(daily_routes)/sizeof(*daily_routes);++i)
        if(!strcmp(method,daily_routes[i].method) && !strcmp(path,daily_routes[i].path))return true;
    return false;
}
