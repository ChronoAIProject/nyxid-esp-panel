#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LCD_STACK_MAX_DEPTH 8u

typedef enum {
    LCD_KIND_NONE = 0,
    LCD_KIND_IMAGE,
    LCD_KIND_TEXT,
} lcd_stack_kind_t;

typedef struct {
    lcd_stack_kind_t kind;
    uint32_t duration_ms;
    uint32_t painted_at_ms;
    uint32_t opaque_id;
} lcd_stack_item_t;

typedef struct {
    lcd_stack_item_t items[LCD_STACK_MAX_DEPTH];
    size_t depth;
} lcd_stack_t;

void lcd_stack_init(lcd_stack_t *s);
bool lcd_stack_push(lcd_stack_t *s, const lcd_stack_item_t *item, uint32_t now_ms);
bool lcd_stack_top(const lcd_stack_t *s, lcd_stack_item_t *out);
void lcd_stack_tick(lcd_stack_t *s, uint32_t now_ms);
void lcd_stack_clear(lcd_stack_t *s);
size_t lcd_stack_depth(const lcd_stack_t *s);
