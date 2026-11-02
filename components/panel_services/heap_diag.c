#include "heap_diag.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

void heap_diag_log(const char *tag, const char *event)
{
    size_t free_8bit = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t free_spiram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_dma = heap_caps_get_free_size(MALLOC_CAP_DMA);
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t min_free_8bit = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t min_free_spiram = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    size_t min_free_dma = heap_caps_get_minimum_free_size(MALLOC_CAP_DMA);
    size_t min_free_internal = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    size_t largest_block_8bit = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    size_t largest_block_spiram = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    size_t largest_block_dma = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
    size_t largest_block_internal = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

    ESP_LOGI(tag ? tag : "heap",
             "heap event=%s free 8bit=%u/%u largest=%u | spiram=%u/%u largest=%u | dma=%u/%u largest=%u | internal=%u/%u largest=%u",
             event ? event : "-",
             (unsigned)free_8bit, (unsigned)min_free_8bit, (unsigned)largest_block_8bit,
             (unsigned)free_spiram, (unsigned)min_free_spiram, (unsigned)largest_block_spiram,
             (unsigned)free_dma, (unsigned)min_free_dma, (unsigned)largest_block_dma,
             (unsigned)free_internal, (unsigned)min_free_internal, (unsigned)largest_block_internal);
}
