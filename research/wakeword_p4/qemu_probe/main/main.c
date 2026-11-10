/* Emulator boot/timer probe only: no microphone, wake inference, or board I/O. */
#include <stdio.h>
#include "esp_chip_info.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* Observable through QEMU memory inspection even if UART emulation is silent. */
volatile struct {
    uint32_t boot_magic;
    uint32_t tick_magic;
    int64_t elapsed_us;
    uint32_t revision;
} nyx_qemu_probe;

void app_main(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    nyx_qemu_probe.revision = chip.revision;
    nyx_qemu_probe.boot_magic = 0x4e595842;
    printf("NYX_QEMU_BOOT_OK idf=%s cores=%u revision=%u\n",
           esp_get_idf_version(), (unsigned)chip.cores, (unsigned)chip.revision);
    const int64_t start = esp_timer_get_time();
    vTaskDelay(pdMS_TO_TICKS(100));
    const int64_t elapsed = esp_timer_get_time() - start;
    if (elapsed > 0) {
        nyx_qemu_probe.elapsed_us = elapsed;
        nyx_qemu_probe.tick_magic = 0x4e595854;
        printf("NYX_QEMU_TICK_OK elapsed_us=%lld\n", (long long)elapsed);
    }
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
