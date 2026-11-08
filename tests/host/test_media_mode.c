#include "unity.h"
#include "media_mode.h"
#include "camera_capture.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int starts, closes, encodes;
static bool ota_busy;
bool panel_ota_busy(void) {return ota_busy;}
static esp_err_t start_error, encode_error, close_error;
static bool block_encode, in_encode, release_encode;
static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;

SemaphoreHandle_t xSemaphoreCreateMutex(void) {
    SemaphoreHandle_t m = malloc(sizeof(*m));
    pthread_mutex_init(m, NULL);
    return m;
}
int xSemaphoreTake(SemaphoreHandle_t m, unsigned timeout) {
    return (timeout == 0 ? pthread_mutex_trylock(m) : pthread_mutex_lock(m)) == 0;
}
void xSemaphoreGive(SemaphoreHandle_t m) { pthread_mutex_unlock(m); }
int64_t esp_timer_get_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}
esp_err_t media_sys_prepare_preview_capture(void) { ++starts; return start_error; }
int media_sys_preview_start(void) { return ESP_OK; }
esp_err_t media_sys_capture_close(void) { ++closes; return close_error; }
esp_err_t media_sys_preview_snapshot_encode(uint8_t *dst, size_t cap, size_t *len) {
    ++encodes;
    if (block_encode) {
        pthread_mutex_lock(&gate);
        in_encode = true;
        pthread_cond_broadcast(&changed);
        while (!release_encode) pthread_cond_wait(&changed, &gate);
        pthread_mutex_unlock(&gate);
    }
    memcpy(dst, "jpeg", 4); *len = 4;
    return encode_error;
}
void setUp(void) {
    ota_busy=false;
    start_error = encode_error = close_error = ESP_OK;
    block_encode = in_encode = release_encode = false;
    media_mode_init();
    media_mode_request_idle("test setup");
    starts = closes = encodes = 0;
}
void tearDown(void) { close_error = ESP_OK; media_mode_release_direct_preview(); media_mode_request_idle("test cleanup"); }
static void test_idle_capture_releases_and_lan_cannot_start(void) {
    uint8_t buf[16]; size_t len = 99;
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, media_mode_snapshot(buf, sizeof(buf), &len, false));
    TEST_ASSERT_EQUAL(0, starts); TEST_ASSERT_EQUAL(0, len);
    TEST_ASSERT_EQUAL(ESP_OK, media_mode_snapshot(buf, sizeof(buf), &len, true));
    TEST_ASSERT_EQUAL(1, starts); TEST_ASSERT_EQUAL(1, closes); TEST_ASSERT_EQUAL(4, len);
    TEST_ASSERT_EQUAL(MEDIA_MODE_IDLE, media_mode_current());
}
static void test_snapshot_preserves_native_preview(void) {
    uint8_t buf[16]; size_t len;
    TEST_ASSERT_EQUAL(ESP_OK, media_mode_request_preview("native"));
    TEST_ASSERT_EQUAL(ESP_OK, media_mode_snapshot(buf, sizeof(buf), &len, false));
    TEST_ASSERT_EQUAL(MEDIA_MODE_PREVIEW, media_mode_current());
    TEST_ASSERT_EQUAL(1, starts); TEST_ASSERT_EQUAL(0, closes);
    media_mode_request_idle("back"); TEST_ASSERT_EQUAL(1, closes);
}
static void test_failed_capture_never_returns_previous_image(void) {
    uint8_t buf[16]; size_t len = 99;
    encode_error = ESP_FAIL;
    TEST_ASSERT_EQUAL(ESP_FAIL, media_mode_snapshot(buf, sizeof(buf), &len, true));
    TEST_ASSERT_EQUAL(0, len); TEST_ASSERT_EQUAL(1, closes);
    TEST_ASSERT_EQUAL(MEDIA_MODE_IDLE, media_mode_current());
    start_error = ESP_FAIL;
    TEST_ASSERT_EQUAL(ESP_FAIL, media_mode_snapshot(buf, sizeof(buf), &len, true));
    TEST_ASSERT_EQUAL(1, encodes); TEST_ASSERT_EQUAL(0, len);
}
static void test_cleanup_failure_is_visible_and_recoverable(void) {
    uint8_t buf[16]; size_t len = 99;
    close_error = ESP_FAIL;
    TEST_ASSERT_EQUAL(ESP_FAIL, media_mode_snapshot(buf, sizeof(buf), &len, true));
    TEST_ASSERT_EQUAL(0, len); TEST_ASSERT_EQUAL(MEDIA_MODE_ERROR, media_mode_current());
    close_error = ESP_OK;
    TEST_ASSERT_EQUAL(ESP_OK, media_mode_request_idle("retry"));
    TEST_ASSERT_EQUAL(MEDIA_MODE_IDLE, media_mode_current());
}
static void *capture_thread(void *arg) {
    uint8_t buf[16]; size_t len;
    esp_err_t *err = arg;
    *err = media_mode_snapshot(buf, sizeof(buf), &len, true);
    return NULL;
}
static void *preview_thread(void *arg) {
    esp_err_t *err = arg;
    *err = media_mode_request_preview("concurrent native open");
    return NULL;
}
static void test_concurrent_capture_rejected_and_preview_serialized(void) {
    block_encode = true;
    pthread_t capture, preview;
    esp_err_t capture_err = ESP_FAIL, preview_err = ESP_FAIL;
    pthread_create(&capture, NULL, capture_thread, &capture_err);
    pthread_mutex_lock(&gate);
    while (!in_encode) pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    TEST_ASSERT_EQUAL(MEDIA_MODE_CAPTURE, media_mode_current());
    uint8_t buf[16]; size_t len = 99;
    TEST_ASSERT_EQUAL(ESP_ERR_TIMEOUT, media_mode_snapshot(buf, sizeof(buf), &len, true));
    TEST_ASSERT_EQUAL(0, len);
    pthread_create(&preview, NULL, preview_thread, &preview_err);
    pthread_mutex_lock(&gate); release_encode = true;
    pthread_cond_broadcast(&changed); pthread_mutex_unlock(&gate);
    pthread_join(capture, NULL); pthread_join(preview, NULL);
    TEST_ASSERT_EQUAL(ESP_OK, capture_err); TEST_ASSERT_EQUAL(ESP_OK, preview_err);
    TEST_ASSERT_EQUAL(2, starts); TEST_ASSERT_EQUAL(1, closes);
    TEST_ASSERT_EQUAL(MEDIA_MODE_PREVIEW, media_mode_current());
}
static void test_update_blocks_new_camera_work(void) {
    uint8_t buf[16];size_t len=99;
    ota_busy=true;
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_claim_direct_preview());
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_request_preview("native"));
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_snapshot(buf,sizeof(buf),&len,true));
    TEST_ASSERT_EQUAL(0,len);TEST_ASSERT_EQUAL(0,starts);
    TEST_ASSERT_EQUAL(MEDIA_MODE_IDLE,media_mode_current());
    ota_busy=false;
    TEST_ASSERT_EQUAL(ESP_OK,media_mode_snapshot(buf,sizeof(buf),&len,true));
    TEST_ASSERT_EQUAL(1,starts);
}
static void test_direct_preview_owns_camera_until_cleanup(void) {
    uint8_t buf[16]; size_t len=99;
    TEST_ASSERT_EQUAL(ESP_OK,media_mode_claim_direct_preview());
    TEST_ASSERT_EQUAL(MEDIA_MODE_PREVIEW,media_mode_current());
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_claim_direct_preview());
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_request_idle("shared close"));
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_request_preview("shared preview"));
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_snapshot(buf,sizeof(buf),&len,true));
    TEST_ASSERT_EQUAL(0,len); TEST_ASSERT_EQUAL(0,starts); TEST_ASSERT_EQUAL(0,closes);
    media_mode_release_direct_preview();
    TEST_ASSERT_EQUAL(MEDIA_MODE_IDLE,media_mode_current());
    TEST_ASSERT_EQUAL(ESP_OK,media_mode_snapshot(buf,sizeof(buf),&len,true));
    TEST_ASSERT_EQUAL(ESP_OK,media_mode_request_preview("shared preview"));
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE,media_mode_claim_direct_preview());
    media_mode_release_direct_preview();
    TEST_ASSERT_EQUAL(MEDIA_MODE_PREVIEW,media_mode_current());
}
int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_direct_preview_owns_camera_until_cleanup);
    RUN_TEST(test_update_blocks_new_camera_work);
    RUN_TEST(test_idle_capture_releases_and_lan_cannot_start);
    RUN_TEST(test_snapshot_preserves_native_preview);
    RUN_TEST(test_failed_capture_never_returns_previous_image);
    RUN_TEST(test_cleanup_failure_is_visible_and_recoverable);
    RUN_TEST(test_concurrent_capture_rejected_and_preview_serialized);
    return UNITY_END();
}
