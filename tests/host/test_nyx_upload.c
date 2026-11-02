#include "unity.h"
#include "nyx_upload.h"
#include <string.h>
void setUp(void) {}
void tearDown(void) {}
static const char *id = "01234567-89ab-cdef-0123-456789abcdef";
static uint8_t buffer[64], frame[100];
static nyx_upload_t upload;
static size_t make_frame(uint64_t seq, const char *payload, bool end) {
    memset(frame, 0, sizeof(frame));
    memcpy(frame, "NYXM", 4); frame[4] = 8; frame[5] = end;
    memcpy(frame + 6, upload.id, 16);
    for (int i = 29; i >= 22; --i) { frame[i] = seq & 255; seq >>= 8; }
    size_t len = strlen(payload); memcpy(frame + 30, payload, len);
    return 30 + len;
}
static void start(bool exact, size_t expected) {
    memset(&upload, 0, sizeof(upload));
    TEST_ASSERT_TRUE(nyx_upload_open(&upload, id, buffer, sizeof(buffer), exact, expected));
}
static void test_chunks_and_end(void) {
    start(true, 6);
    TEST_ASSERT_EQUAL(NYX_UPLOAD_MORE, nyx_upload_frame(&upload, frame, make_frame(0,"abc",false)));
    TEST_ASSERT_EQUAL(NYX_UPLOAD_MORE, nyx_upload_frame(&upload, frame, make_frame(1,"def",false)));
    TEST_ASSERT_EQUAL(NYX_UPLOAD_DONE, nyx_upload_frame(&upload, frame, make_frame(2,"",true)));
    TEST_ASSERT_EQUAL_UINT(6, upload.used); TEST_ASSERT_EQUAL_MEMORY("abcdef",buffer,6);
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID, nyx_upload_frame(&upload,frame,30));
}
static void test_truncation_and_order(void) {
    start(true, 4);
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID, nyx_upload_frame(&upload,frame,make_frame(0,"abc",true)));
    start(false, 0);
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID, nyx_upload_frame(&upload,frame,make_frame(1,"abc",false)));
    start(false, 0);
    TEST_ASSERT_EQUAL(NYX_UPLOAD_MORE, nyx_upload_frame(&upload,frame,make_frame(0,"abc",false)));
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID, nyx_upload_frame(&upload,frame,make_frame(0,"abc",false)));
}
static void test_overflow_cancel_and_identity(void) {
    start(false, 0);
    size_t n=make_frame(0,"",false); (void)n;
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID, nyx_upload_frame(&upload,frame,95));
    start(false, 0);
    make_frame(0,"abc",false); frame[4]=9;
    TEST_ASSERT_EQUAL(NYX_UPLOAD_ABORTED,nyx_upload_frame(&upload,frame,33));
    start(false, 0);
    make_frame(0,"abc",true); frame[6]^=1;
    TEST_ASSERT_EQUAL(NYX_UPLOAD_INVALID,nyx_upload_frame(&upload,frame,33));
    start(false, 0);
    TEST_ASSERT_FALSE(nyx_upload_open(&upload,id,buffer,64,false,0));
    nyx_upload_cancel(&upload); TEST_ASSERT_FALSE(upload.active);
    TEST_ASSERT_FALSE(nyx_upload_open(&upload,id,buffer,64,true,65));
}
int main(void) {
    UNITY_BEGIN(); RUN_TEST(test_chunks_and_end); RUN_TEST(test_truncation_and_order);
    RUN_TEST(test_overflow_cancel_and_identity); return UNITY_END();
}
