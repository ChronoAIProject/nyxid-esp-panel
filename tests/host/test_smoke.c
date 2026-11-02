#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_harness_alive(void)
{
    TEST_ASSERT_EQUAL_INT(2, 1 + 1);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_harness_alive);
    return UNITY_END();
}
