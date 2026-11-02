#include "unity.h"
#include "nyx_ws_rx.h"
#include <stdlib.h>
#include <string.h>
void setUp(void) {}
void tearDown(void) {}
static void test_transport_chunks_and_binary_continuation(void) {
    nyx_ws_rx_t rx={0}; uint8_t *out=NULL, op=0; size_t n=0;
    TEST_ASSERT_EQUAL(0,nyx_ws_rx_feed(&rx,2,false,4,0,"ab",2,&out,&n,&op));
    TEST_ASSERT_EQUAL(0,nyx_ws_rx_feed(&rx,2,false,4,2,"cd",2,&out,&n,&op));
    TEST_ASSERT_EQUAL(1,nyx_ws_rx_feed(&rx,0,true,2,0,"ef",2,&out,&n,&op));
    TEST_ASSERT_EQUAL_UINT(2,op); TEST_ASSERT_EQUAL_UINT(6,n);
    TEST_ASSERT_EQUAL_MEMORY("abcdef",out,6); free(out);
}
static void test_gap_oversize_and_disconnect(void) {
    nyx_ws_rx_t rx={0}; uint8_t *out=NULL, op=0; size_t n=0;
    TEST_ASSERT_EQUAL(0,nyx_ws_rx_feed(&rx,1,true,4,0,"ab",2,&out,&n,&op));
    TEST_ASSERT_EQUAL(-1,nyx_ws_rx_feed(&rx,1,true,4,3,"d",1,&out,&n,&op));
    TEST_ASSERT_NULL(rx.data);
    TEST_ASSERT_EQUAL(-1,nyx_ws_rx_feed(&rx,1,true,NYX_WS_MAX_MESSAGE+1,0,"",0,&out,&n,&op));
    TEST_ASSERT_EQUAL(-1,nyx_ws_rx_feed(&rx,0,true,1,0,"a",1,&out,&n,&op));
    TEST_ASSERT_EQUAL(0,nyx_ws_rx_feed(&rx,1,false,1,0,"a",1,&out,&n,&op));
    nyx_ws_rx_reset(&rx); TEST_ASSERT_NULL(rx.data);
    TEST_ASSERT_EQUAL(1,nyx_ws_rx_feed(&rx,1,true,2,0,"ok",2,&out,&n,&op));
    TEST_ASSERT_EQUAL_STRING("ok",out); free(out);
}
int main(void) { UNITY_BEGIN(); RUN_TEST(test_transport_chunks_and_binary_continuation);
    RUN_TEST(test_gap_oversize_and_disconnect); return UNITY_END(); }
