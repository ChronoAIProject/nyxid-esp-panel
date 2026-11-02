#include "unity.h"
#include "lcd_stack.h"

void setUp(void) {}
void tearDown(void) {}

void test_empty_has_no_top(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t top;
    TEST_ASSERT_FALSE(lcd_stack_top(&s, &top));
}

void test_push_sets_top_and_painted_time(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t item = { .kind = LCD_KIND_TEXT, .duration_ms = 5000, .opaque_id = 7 };
    TEST_ASSERT_TRUE(lcd_stack_push(&s, &item, 1000));

    lcd_stack_item_t top;
    TEST_ASSERT_TRUE(lcd_stack_top(&s, &top));
    TEST_ASSERT_EQUAL_INT(LCD_KIND_TEXT, top.kind);
    TEST_ASSERT_EQUAL_UINT32(1000, top.painted_at_ms);
    TEST_ASSERT_EQUAL_UINT32(7, top.opaque_id);
}

void test_ttl_expires_top(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t a = { .kind = LCD_KIND_TEXT, .duration_ms = 1000, .opaque_id = 1 };
    lcd_stack_push(&s, &a, 0);
    lcd_stack_tick(&s, 999);
    lcd_stack_item_t top;
    TEST_ASSERT_TRUE(lcd_stack_top(&s, &top));
    lcd_stack_tick(&s, 1000);
    TEST_ASSERT_FALSE(lcd_stack_top(&s, &top));
}

void test_zero_duration_persists(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t a = { .kind = LCD_KIND_IMAGE, .duration_ms = 0, .opaque_id = 1 };
    lcd_stack_push(&s, &a, 0);
    lcd_stack_tick(&s, 1000000);
    lcd_stack_item_t top;
    TEST_ASSERT_TRUE(lcd_stack_top(&s, &top));
    TEST_ASSERT_EQUAL_UINT32(1, top.opaque_id);
}

void test_expired_top_reveals_previous_item(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t a = { .kind = LCD_KIND_TEXT, .duration_ms = 0, .opaque_id = 1 };
    lcd_stack_item_t b = { .kind = LCD_KIND_IMAGE, .duration_ms = 3000, .opaque_id = 2 };
    lcd_stack_push(&s, &a, 0);
    lcd_stack_push(&s, &b, 100);
    TEST_ASSERT_EQUAL_size_t(2, lcd_stack_depth(&s));

    lcd_stack_tick(&s, 3100);
    lcd_stack_item_t top;
    TEST_ASSERT_TRUE(lcd_stack_top(&s, &top));
    TEST_ASSERT_EQUAL_UINT32(1, top.opaque_id);
    TEST_ASSERT_EQUAL_size_t(1, lcd_stack_depth(&s));
}

void test_overflow_drops_oldest(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    for (uint32_t i = 0; i < LCD_STACK_MAX_DEPTH + 1u; i++) {
        lcd_stack_item_t item = { .kind = LCD_KIND_TEXT, .opaque_id = i };
        TEST_ASSERT_TRUE(lcd_stack_push(&s, &item, i));
    }
    TEST_ASSERT_EQUAL_size_t(LCD_STACK_MAX_DEPTH, lcd_stack_depth(&s));
    TEST_ASSERT_EQUAL_UINT32(1, s.items[0].opaque_id);
}

void test_clear_removes_all_items(void)
{
    lcd_stack_t s;
    lcd_stack_init(&s);
    lcd_stack_item_t a = { .kind = LCD_KIND_TEXT };
    lcd_stack_push(&s, &a, 0);
    lcd_stack_clear(&s);
    lcd_stack_item_t top;
    TEST_ASSERT_FALSE(lcd_stack_top(&s, &top));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_empty_has_no_top);
    RUN_TEST(test_push_sets_top_and_painted_time);
    RUN_TEST(test_ttl_expires_top);
    RUN_TEST(test_zero_duration_persists);
    RUN_TEST(test_expired_top_reveals_previous_item);
    RUN_TEST(test_overflow_drops_oldest);
    RUN_TEST(test_clear_removes_all_items);
    return UNITY_END();
}
