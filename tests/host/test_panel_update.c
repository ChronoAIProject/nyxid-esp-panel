#include "unity.h"
#include "panel_update_policy.h"
#include <stdio.h>
#include <string.h>
void setUp(void) {}
void tearDown(void) {}
static char manifest[2048];
static void release(const char *profile,const char *version,const char *url,const char *hash)
{
    snprintf(manifest,sizeof(manifest),"{\"schema\":1,\"project\":\"nyxid-esp-panel\",\"profile\":\"%s\",\"version\":\"%s\",\"url\":\"%s\",\"size\":123456,\"sha256\":\"%s\"}",profile,version,url,hash);
}
static const char *url=PANEL_RELEASE_ROOT "download/v0.6.24/nyxid-esp-panel-p4-rev1-16mb.bin";
static const char *hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
static void versions_are_numeric_and_stable(void)
{
    TEST_ASSERT_TRUE(panel_update_version_newer("0.6.24","0.6.23"));
    TEST_ASSERT_TRUE(panel_update_version_newer("0.10.0","0.9.99"));
    TEST_ASSERT_FALSE(panel_update_version_newer("0.6.24","0.6.24"));
    TEST_ASSERT_FALSE(panel_update_version_newer("0.6.23","0.6.24"));
    const char *bad[]={"v0.6.24","0.6.24-beta","0.6.24+dev","01.2.3","1.2","1.2.3.4","1.65536.0","1.-1.0","1.2.3 ",""};
    for (size_t i=0;i<sizeof(bad)/sizeof(*bad);++i) TEST_ASSERT_FALSE(panel_update_version_valid(bad[i]));
    TEST_ASSERT_FALSE(panel_update_version_valid(NULL));
}
static void release_requires_exact_hardware_and_asset_identity(void)
{
    panel_release_t parsed={0};
    release("p4-rev1-16mb","0.6.24",url,hash);
    TEST_ASSERT_TRUE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    TEST_ASSERT_EQUAL_STRING("0.6.24",parsed.version);
    TEST_ASSERT_EQUAL_STRING(url,parsed.url);
    TEST_ASSERT_EQUAL_UINT(123456,parsed.size);
    TEST_ASSERT_EQUAL_UINT8(0x01,parsed.sha256[0]);
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev3-16mb",&parsed));
    release("p4-rev1-16mb","0.6.25",url,hash);
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24","https://evil.example/firmware.bin",hash);
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24",url,"invalid");
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
}
static void malformed_or_unbounded_metadata_is_rejected(void)
{
    panel_release_t parsed={0};
    const char *bad[]={"{}","[]","null","{\"schema\":1}","{", "true"};
    for (size_t i=0;i<sizeof(bad)/sizeof(*bad);++i)
        TEST_ASSERT_FALSE(panel_update_parse_release(bad[i],strlen(bad[i]),"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24",url,hash);
    strcat(manifest,"garbage");
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,4097,"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24",url,hash);
    char *size=strstr(manifest,"123456");memcpy(size,"999999",6);
    TEST_ASSERT_TRUE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24",url,hash);
    char *schema=strstr(manifest,"\"schema\":1");schema[9]='2';
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
    release("p4-rev1-16mb","0.6.24",url,hash);
    size=strstr(manifest,"123456");
    memcpy(size,"000000",6);
    TEST_ASSERT_FALSE(panel_update_parse_release(manifest,strlen(manifest),"p4-rev1-16mb",&parsed));
}
static void redirects_cannot_downgrade_tls_or_leave_github(void)
{
    TEST_ASSERT_TRUE(panel_update_download_url_allowed(url));
    TEST_ASSERT_TRUE(panel_update_download_url_allowed("https://release-assets.githubusercontent.com/path?signature=x"));
    TEST_ASSERT_TRUE(panel_update_download_url_allowed("https://github.com:443/ChronoAIProject/nyxid-esp-panel/releases/download/v0.6.24/update-p4-rev1-16mb.json"));
    TEST_ASSERT_TRUE(panel_update_download_url_allowed("https://release-assets.githubusercontent.com:443/path"));
    TEST_ASSERT_TRUE(panel_update_download_url_allowed("https://objects.githubusercontent.com:443/path"));
    const char *bad[]={"http://github.com/file","https://github.com.evil.example/","https://github.com@evil.example/","https://release-assets.githubusercontent.com.evil.example/","https://evil.example/","https://github.com/other/repo/","https://github.com:443/other/repo/","https://release-assets.githubusercontent.com:444/path","https://release-assets.githubusercontent.com:443@evil.example/"};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);++i) TEST_ASSERT_FALSE(panel_update_download_url_allowed(bad[i]));
}
int main(void)
{
    UNITY_BEGIN();RUN_TEST(versions_are_numeric_and_stable);RUN_TEST(release_requires_exact_hardware_and_asset_identity);
    RUN_TEST(malformed_or_unbounded_metadata_is_rejected);RUN_TEST(redirects_cannot_downgrade_tls_or_leave_github);return UNITY_END();
}
