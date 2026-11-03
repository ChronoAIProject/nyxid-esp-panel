#include "unity.h"
#include "panel_scanout_gate.h"

void setUp(void) {}
void tearDown(void) {}

static void test_old_dma_selection_cannot_release_new_frame(void)
{
    int buffers[2];
    panel_scanout_gate_t gate = {0};
    TEST_ASSERT_FALSE(panel_scanout_complete(&gate, &buffers[0]));
    panel_scanout_submit(&gate, &buffers[1]);
    // DMA can sample the old index before the task submits the next one, then
    // finish its ISR afterward. Even repeated stale events must not wake LVGL.
    for (unsigned i = 0; i < 20; ++i)
        TEST_ASSERT_FALSE(panel_scanout_complete(&gate, &buffers[0]));
    TEST_ASSERT_TRUE(panel_scanout_complete(&gate, &buffers[1]));
    TEST_ASSERT_FALSE(panel_scanout_complete(&gate, &buffers[1]));
    panel_scanout_submit(&gate, &buffers[0]);
    TEST_ASSERT_FALSE(panel_scanout_complete(&gate, &buffers[1]));
    TEST_ASSERT_TRUE(panel_scanout_complete(&gate, &buffers[0]));
}

static void test_completion_before_submit_is_recovered_by_next_dma(void)
{
    int buffers[2];
    panel_scanout_gate_t gate = {0};
    // A switch before the task arms the gate is conservative, not a lost wake:
    // IDF invokes the callback at every real DMA completion, including repeats.
    TEST_ASSERT_FALSE(panel_scanout_complete(&gate, &buffers[1]));
    panel_scanout_submit(&gate, &buffers[1]);
    TEST_ASSERT_FALSE(panel_scanout_complete(&gate, NULL));
    TEST_ASSERT_TRUE(panel_scanout_complete(&gate, &buffers[1]));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_old_dma_selection_cannot_release_new_frame);
    RUN_TEST(test_completion_before_submit_is_recovered_by_next_dma);
    return UNITY_END();
}
