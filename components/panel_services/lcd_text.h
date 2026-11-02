#pragma once

#include "esp_err.h"
#include "lvgl.h"

#include <stdint.h>

typedef enum {
    LCD_TEXT_SIZE_SMALL = 0,
    LCD_TEXT_SIZE_MEDIUM,
    LCD_TEXT_SIZE_LARGE,
} lcd_text_size_t;

typedef enum {
    LCD_TEXT_ALIGN_LEFT = 0,
    LCD_TEXT_ALIGN_CENTER,
    LCD_TEXT_ALIGN_RIGHT,
} lcd_text_align_t;

typedef struct {
    const char *text;
    lcd_text_size_t size;
    uint32_t color;
    uint32_t bg;
    lcd_text_align_t align;
} lcd_text_style_t;

const lv_font_t *lcd_text_font(lcd_text_size_t size);
void lcd_text_apply(lv_obj_t *label, const lcd_text_style_t *style);
lv_text_align_t lcd_text_lv_align(lcd_text_align_t align);

