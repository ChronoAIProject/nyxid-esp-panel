#include "lcd_text.h"

#include <stddef.h>

LV_FONT_DECLARE(lv_font_noto_sans_sc_med_20);

const lv_font_t *lcd_text_font(lcd_text_size_t size)
{
    switch (size) {
    case LCD_TEXT_SIZE_LARGE:
    case LCD_TEXT_SIZE_MEDIUM:
        return &lv_font_noto_sans_sc_med_20;
    case LCD_TEXT_SIZE_SMALL:
    default:
        return LV_FONT_DEFAULT;
    }
}

lv_text_align_t lcd_text_lv_align(lcd_text_align_t align)
{
    switch (align) {
    case LCD_TEXT_ALIGN_LEFT:
        return LV_TEXT_ALIGN_LEFT;
    case LCD_TEXT_ALIGN_RIGHT:
        return LV_TEXT_ALIGN_RIGHT;
    case LCD_TEXT_ALIGN_CENTER:
    default:
        return LV_TEXT_ALIGN_CENTER;
    }
}

void lcd_text_apply(lv_obj_t *label, const lcd_text_style_t *style)
{
    if (!label || !style) {
        return;
    }
    lv_obj_set_style_text_font(label, lcd_text_font(style->size), 0);
    lv_obj_set_style_text_color(label, lv_color_hex(style->color), 0);
    lv_obj_set_style_text_align(label, lcd_text_lv_align(style->align), 0);
    lv_label_set_long_mode(label, LV_LABEL_LONG_MODE_WRAP);
    lv_label_set_text(label, style->text ? style->text : "");
}
