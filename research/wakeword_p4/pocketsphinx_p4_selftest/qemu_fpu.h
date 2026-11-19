#include "sdkconfig.h"
// Scoped to vectors.S in the QEMU build. It enables IDF's existing mtval opcode checks.
#undef CONFIG_ESP32P4_SELECTS_REV_LESS_V3
#define CONFIG_ESP32P4_SELECTS_REV_LESS_V3 1
