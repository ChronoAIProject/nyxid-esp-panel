#include "lcd_stack.h"

#include "esp_log.h"

static const char *TAG = "haCam.lcdstack";

static const char *kind_name(lcd_stack_kind_t kind)
{
    switch (kind) {
    case LCD_KIND_IMAGE:
        return "image";
    case LCD_KIND_TEXT:
        return "text";
    case LCD_KIND_NONE:
    default:
        return "none";
    }
}

void lcd_stack_init(lcd_stack_t *s)
{
    if (!s) {
        return;
    }
    s->depth = 0;
}

bool lcd_stack_push(lcd_stack_t *s, const lcd_stack_item_t *item, uint32_t now_ms)
{
    if (!s || !item || item->kind == LCD_KIND_NONE) {
        return false;
    }

    if (s->depth == LCD_STACK_MAX_DEPTH) {
        ESP_LOGI(TAG, "push overflow depth=%u dropping_oldest kind=%s reason=max_depth",
                 (unsigned)s->depth, kind_name(s->items[0].kind));
        for (size_t i = 1; i < s->depth; i++) {
            s->items[i - 1] = s->items[i];
        }
        s->depth--;
    }

    lcd_stack_item_t copy = *item;
    copy.painted_at_ms = now_ms;
    s->items[s->depth++] = copy;
    ESP_LOGI(TAG, "push kind=%s duration_ms=%lu id=%lu depth=%u reason=request",
             kind_name(copy.kind), (unsigned long)copy.duration_ms,
             (unsigned long)copy.opaque_id, (unsigned)s->depth);
    return true;
}

bool lcd_stack_top(const lcd_stack_t *s, lcd_stack_item_t *out)
{
    if (!s || !out || s->depth == 0) {
        return false;
    }
    *out = s->items[s->depth - 1];
    return true;
}

void lcd_stack_tick(lcd_stack_t *s, uint32_t now_ms)
{
    if (!s) {
        return;
    }

    while (s->depth > 0) {
        lcd_stack_item_t *top = &s->items[s->depth - 1];
        if (top->duration_ms == 0) {
            break;
        }
        if ((uint32_t)(now_ms - top->painted_at_ms) < top->duration_ms) {
            break;
        }
        ESP_LOGI(TAG, "pop kind=%s id=%lu depth=%u reason=expired",
                 kind_name(top->kind), (unsigned long)top->opaque_id, (unsigned)s->depth);
        s->depth--;
    }
}

void lcd_stack_clear(lcd_stack_t *s)
{
    if (s) {
        ESP_LOGI(TAG, "clear depth=%u reason=request", (unsigned)s->depth);
        s->depth = 0;
    }
}

size_t lcd_stack_depth(const lcd_stack_t *s)
{
    return s ? s->depth : 0;
}
