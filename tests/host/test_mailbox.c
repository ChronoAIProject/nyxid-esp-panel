#include "unity.h"
#include "mailbox.h"

void setUp(void) {}
void tearDown(void) {}

void test_fifo_ordering(void)
{
    mailbox_t mb;
    int slots[4];
    mailbox_init(&mb, slots, sizeof(slots[0]), 4);

    int a = 10;
    int b = 20;
    int c = 30;
    TEST_ASSERT_TRUE(mailbox_post(&mb, &a, MAILBOX_NOWAIT));
    TEST_ASSERT_TRUE(mailbox_post(&mb, &b, MAILBOX_NOWAIT));
    TEST_ASSERT_TRUE(mailbox_post(&mb, &c, MAILBOX_NOWAIT));

    int out = 0;
    TEST_ASSERT_TRUE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
    TEST_ASSERT_EQUAL_INT(10, out);
    TEST_ASSERT_TRUE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
    TEST_ASSERT_EQUAL_INT(20, out);
    TEST_ASSERT_TRUE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
    TEST_ASSERT_EQUAL_INT(30, out);
    TEST_ASSERT_FALSE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
}

void test_full_drops_oldest(void)
{
    mailbox_t mb;
    int slots[2];
    mailbox_init(&mb, slots, sizeof(slots[0]), 2);

    int v = 1;
    TEST_ASSERT_TRUE(mailbox_post(&mb, &v, MAILBOX_NOWAIT));
    v = 2;
    TEST_ASSERT_TRUE(mailbox_post(&mb, &v, MAILBOX_NOWAIT));
    v = 3;
    TEST_ASSERT_TRUE(mailbox_post(&mb, &v, MAILBOX_NOWAIT));

    int out = 0;
    TEST_ASSERT_TRUE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
    TEST_ASSERT_EQUAL_INT(2, out);
    TEST_ASSERT_TRUE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
    TEST_ASSERT_EQUAL_INT(3, out);
    TEST_ASSERT_FALSE(mailbox_recv(&mb, &out, MAILBOX_NOWAIT));
}

void test_count_and_clear(void)
{
    mailbox_t mb;
    int slots[4];
    mailbox_init(&mb, slots, sizeof(slots[0]), 4);
    int v = 0;
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_TRUE(mailbox_post(&mb, &v, MAILBOX_NOWAIT));
    }
    TEST_ASSERT_EQUAL_size_t(3, mailbox_count(&mb));
    mailbox_clear(&mb);
    TEST_ASSERT_EQUAL_size_t(0, mailbox_count(&mb));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_fifo_ordering);
    RUN_TEST(test_full_drops_oldest);
    RUN_TEST(test_count_and_clear);
    return UNITY_END();
}
