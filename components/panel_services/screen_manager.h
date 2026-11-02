#pragma once

#include "lvgl.h"

#include <stdbool.h>
#include <stddef.h>

typedef void (*screen_manager_destroy_cb_t)(lv_obj_t *screen, void *user);
typedef void (*screen_manager_change_cb_t)(void *user);

void screen_manager_init(lv_display_t *display);
void screen_manager_set_change_cb(screen_manager_change_cb_t cb, void *user);
bool screen_manager_push(lv_obj_t *screen, screen_manager_destroy_cb_t destroy_cb, void *user);
bool screen_manager_replace(lv_obj_t *screen, screen_manager_destroy_cb_t destroy_cb, void *user);
bool screen_manager_pop(void);
void screen_manager_clear(void);
lv_obj_t *screen_manager_current(void);
size_t screen_manager_depth(void);
