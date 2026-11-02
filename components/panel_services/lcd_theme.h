#pragma once

#include "lvgl.h"

#include <stdbool.h>

#define LCD_THEME_BLACK        0x000000u
#define LCD_THEME_GREEN        0x00ff41u
#define LCD_THEME_CYAN         0x00ffffu
#define LCD_THEME_DIM_GREEN    0x2a8c3eu
#define LCD_THEME_RED          0xff0040u
#define LCD_THEME_AMBER        0xffb300u

const lv_font_t *lcd_theme_font(void);
void lcd_theme_apply_screen(lv_obj_t *obj);
void lcd_theme_apply_panel(lv_obj_t *obj);
void lcd_theme_apply_label(lv_obj_t *obj, uint32_t color);
void lcd_theme_apply_button(lv_obj_t *obj);
void lcd_theme_apply_textarea(lv_obj_t *obj, bool password);
lv_obj_t *lcd_theme_button(lv_obj_t *parent, const char *text);
lv_obj_t *lcd_theme_label(lv_obj_t *parent, const char *text, uint32_t color);
