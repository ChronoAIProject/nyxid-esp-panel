#pragma once
#include "panel_board_config.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// IDF's custom app descriptor follows its 256-byte standard app descriptor.
#define PANEL_BOARD_IMAGE_OFFSET 288u
#define PANEL_BOARD_IMAGE_HEADER_SIZE 400u
typedef struct {
    char magic[16];
    char profile[48];
    char board[48];
} panel_board_image_desc_t;
extern const panel_board_image_desc_t panel_board_image_desc;

// Validates the board descriptor before any OTA erase/write is started.
bool panel_board_image_matches(const void *image, size_t size);

#ifdef __cplusplus
}
#endif
