#include "panel_board.h"
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(panel_board_image_desc_t) == 112, "Board image descriptor layout changed");
_Static_assert(sizeof(PANEL_BOARD_PROFILE) <= 48 && sizeof(PANEL_BOARD_ID) <= 48,
               "Board identity exceeds image descriptor capacity");
#ifdef ESP_PLATFORM
__attribute__((section(".rodata_custom_desc"), used, aligned(4)))
#endif
const panel_board_image_desc_t panel_board_image_desc = {
        .magic = "NYX-BOARD-V1", .profile = PANEL_BOARD_PROFILE, .board = PANEL_BOARD_ID,
    };

bool panel_board_image_matches(const void *image, size_t size)
{
    if (!image || size < PANEL_BOARD_IMAGE_HEADER_SIZE || *(const uint8_t *)image != 0xe9) return false;
    const uint8_t *bytes = image;
    const unsigned minimum = bytes[15] | ((unsigned)bytes[16] << 8);
    const unsigned maximum = bytes[17] | ((unsigned)bytes[18] << 8);
    if (bytes[12] != 0x12 || bytes[13] || minimum != PANEL_BOARD_MIN_REVISION ||
        maximum != PANEL_BOARD_MAX_REVISION) return false;
    panel_board_image_desc_t candidate;
    memcpy(&candidate, (const uint8_t *)image + PANEL_BOARD_IMAGE_OFFSET, sizeof(candidate));
    return !memcmp(&candidate, &panel_board_image_desc, sizeof(candidate));
}
