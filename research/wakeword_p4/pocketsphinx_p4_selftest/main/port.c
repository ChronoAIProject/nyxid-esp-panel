#include <stddef.h>
#include "util/mmio.h"

#ifdef NYX_LEXICAL_FLASH_MAP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_partition.h"

struct mmio_file_s {
    const void *ptr;
    esp_partition_mmap_handle_t handle;
};

static const struct {
    const char *name;
    size_t offset, length;
} files[] = {
#include "map_offsets.inc"
};

mmio_file_t *mmio_file_read(const char *name)
{
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, "storage");
    if (!partition) return NULL;
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
        if (strcmp(name, files[i].name)) continue;
        mmio_file_t *file = malloc(sizeof(*file));
        if (!file) return NULL;
        if (esp_partition_mmap(partition, files[i].offset, files[i].length,
                               ESP_PARTITION_MMAP_DATA, &file->ptr, &file->handle) != ESP_OK) {
            free(file);
            return NULL;
        }
        printf("LEX_MAP file=%s offset=%u length=%u\n", name,
               (unsigned)files[i].offset, (unsigned)files[i].length);
        return file;
    }
    return NULL;
}

void *mmio_file_ptr(mmio_file_t *file) { return (void *)file->ptr; }

void mmio_file_unmap(mmio_file_t *file)
{
    esp_partition_munmap(file->handle);
    free(file);
}
#else
mmio_file_t *mmio_file_read(const char *name) { (void)name; return NULL; }
void *mmio_file_ptr(mmio_file_t *file) { (void)file; return NULL; }
void mmio_file_unmap(mmio_file_t *file) { (void)file; }
#endif
