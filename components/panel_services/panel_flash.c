#include "panel_flash.h"

#include "esp_attr.h"
#include "esp_flash.h"
#include "esp_private/spi_flash_os.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

// On the acceptance P4, status reads overlapping OTA erase/write cause an HP
// watchdog reset. Serialize every application flash caller, including native
// Settings/NVS. Keep the mutex in internal RAM and preserve the no-OS path.
static DRAM_ATTR StaticSemaphore_t lock_storage;
static DRAM_ATTR SemaphoreHandle_t flash_lock;

esp_err_t panel_flash_init(void)
{
    if (!flash_lock) flash_lock = xSemaphoreCreateRecursiveMutexStatic(&lock_storage);
    return flash_lock ? ESP_OK : ESP_ERR_NO_MEM;
}

void panel_flash_lock(void)
{
    if (flash_lock) xSemaphoreTakeRecursive(flash_lock, portMAX_DELAY);
}

void panel_flash_unlock(void)
{
    if (flash_lock) xSemaphoreGiveRecursive(flash_lock);
}

static bool use_lock(void)
{
    return flash_lock && spi_flash_guard_get() != &g_flash_guard_no_os_ops &&
           xTaskGetSchedulerState() == taskSCHEDULER_RUNNING;
}

#define FLASH_CALL(call) do {                     \
    bool locked = use_lock();                    \
    if (locked) panel_flash_lock();              \
    esp_err_t result = (call);                   \
    if (locked) panel_flash_unlock();            \
    return result;                              \
} while (0)

extern esp_err_t __real_esp_flash_read(esp_flash_t *, void *, uint32_t, uint32_t);
extern esp_err_t __real_esp_flash_write(esp_flash_t *, const void *, uint32_t, uint32_t);
extern esp_err_t __real_esp_flash_read_encrypted(esp_flash_t *, uint32_t, void *, uint32_t);
extern esp_err_t __real_esp_flash_write_encrypted(esp_flash_t *, uint32_t, const void *, uint32_t);
extern esp_err_t __real_esp_flash_erase_region(esp_flash_t *, uint32_t, uint32_t);
extern esp_err_t __real_esp_flash_erase_chip(esp_flash_t *);

esp_err_t __wrap_esp_flash_read(esp_flash_t *chip, void *buffer, uint32_t address, uint32_t size)
{
    FLASH_CALL(__real_esp_flash_read(chip, buffer, address, size));
}
esp_err_t __wrap_esp_flash_write(esp_flash_t *chip, const void *buffer, uint32_t address, uint32_t size)
{
    FLASH_CALL(__real_esp_flash_write(chip, buffer, address, size));
}
esp_err_t __wrap_esp_flash_read_encrypted(esp_flash_t *chip, uint32_t address, void *buffer, uint32_t size)
{
    FLASH_CALL(__real_esp_flash_read_encrypted(chip, address, buffer, size));
}
esp_err_t __wrap_esp_flash_write_encrypted(esp_flash_t *chip, uint32_t address, const void *buffer, uint32_t size)
{
    FLASH_CALL(__real_esp_flash_write_encrypted(chip, address, buffer, size));
}
esp_err_t __wrap_esp_flash_erase_region(esp_flash_t *chip, uint32_t address, uint32_t size)
{
    FLASH_CALL(__real_esp_flash_erase_region(chip, address, size));
}
esp_err_t __wrap_esp_flash_erase_chip(esp_flash_t *chip)
{
    FLASH_CALL(__real_esp_flash_erase_chip(chip));
}
