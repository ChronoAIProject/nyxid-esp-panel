#include "unity.h"
#include "panel_access.h"
#include <string.h>
void setUp(void){}
void tearDown(void){}
static void test_daily_and_management_are_separate_services(void){
    char slug[80];TEST_ASSERT_TRUE(panel_management_slug("my-board",slug,sizeof(slug)));
    TEST_ASSERT_EQUAL_STRING("my-board-admin",slug);
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","POST","/voice/open"));
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","POST","/camera/capture"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board","other-board","POST","/camera/capture"));
    const char *denied[]={"/config/wifi","/config/lcd","/voice/config","/config/device","/config/lan_server","/system/reboot","/system/factory_reset","/system/ota","/system/update","/nyx/login"};
    for(size_t i=0;i<sizeof(denied)/sizeof(*denied);++i){
        TEST_ASSERT_FALSE(panel_service_can_call("my-board","my-board","POST",denied[i]));
        TEST_ASSERT_TRUE(panel_service_can_call("my-board",slug,"POST",denied[i]));
    }
    const char *daily[]={"/config/audio","/lcd/brightness","/voice/input-mode","/voice/greeting","/lcd/default-image"};
    for(size_t i=0;i<sizeof(daily)/sizeof(*daily);++i){
        TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","POST",daily[i]));
        TEST_ASSERT_TRUE(panel_service_can_call("my-board",slug,"POST",daily[i]));
        TEST_ASSERT_FALSE(panel_service_can_call("my-board","other-board","POST",daily[i]));
        TEST_ASSERT_FALSE(panel_service_can_call("my-board","my-board","PUT",daily[i]));
    }
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","GET","/lcd/brightness"));
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","GET","/config/audio"));
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","GET","/voice/greeting"));
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","GET","/lcd/default-image"));
    TEST_ASSERT_TRUE(panel_service_can_call("my-board","my-board","DELETE","/lcd/default-image"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board","my-board","GET","/voice/config"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board","other-board-admin","POST","/system/reboot"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board",NULL,"GET","/status"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board","my-board","POST","/voice/../config/wifi"));
    TEST_ASSERT_FALSE(panel_service_can_call("my-board","my-board","POST","/voice/open/extra"));
}
int main(void){UNITY_BEGIN();RUN_TEST(test_daily_and_management_are_separate_services);return UNITY_END();}
