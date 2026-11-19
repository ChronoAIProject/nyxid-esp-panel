#include <stdint.h>
#include <string.h>
#include "miniz.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "packed_gaussian.h"

FILE *packed_gaussian_open(const char *name, void **owner)
{
    enum { header = 72, groups = 6, levels = 256, values = 42 * 3 * 128 * 13,
           tables = groups * levels * 4, packed_size = header + tables + values + 4,
           expanded_size = header + values * 4 + 4 };
    FILE *source = fopen(name, "rb");
    if (!source) return NULL;
    uint8_t *compressed = NULL, *packed = NULL, *expanded = NULL;
    FILE *result = NULL;
    if (fseek(source, 0, SEEK_END) != 0) goto done;
    long length = ftell(source);
    if (length <= 0 || length > packed_size || fseek(source, 0, SEEK_SET) != 0) goto done;
    compressed = malloc(length);
    packed = malloc(packed_size);
    if (!compressed || !packed) goto done;
    for (size_t at = 0; at < (size_t)length;) {
        size_t chunk = (size_t)length - at;
        if (chunk > 16384) chunk = 16384;
        if (fread(compressed + at, 1, chunk, source) != chunk) goto done;
        at += chunk;
        vTaskDelay(1);
    }
    if (tinfl_decompress_mem_to_mem(packed, packed_size, compressed, length,
                                    TINFL_FLAG_PARSE_ZLIB_HEADER) != packed_size) goto done;
    expanded = malloc(expanded_size);
    if (!expanded) goto done;
    memcpy(expanded, packed, header);
    const uint8_t *codes = packed + header + tables;
    for (size_t i = 0; i < values; ++i) {
        size_t group = 2 * ((i / (128 * 13)) % 3) + (i % 13 != 0);
        size_t index = group * levels + codes[i];
        memcpy(expanded + header + i * 4, packed + header + index * 4, 4);
    }
    memcpy(expanded + expanded_size - 4, packed + packed_size - 4, 4);
    result = fmemopen(expanded, expanded_size, "rb");
    if (result) {
        vTaskDelay(1);
        *owner = expanded;
        expanded = NULL;
        printf("LEX_UNPACK file=%s bytes=%ld expanded=%d\n", name, length, expanded_size);
    }
done:
    fclose(source);
    free(compressed);
    free(packed);
    free(expanded);
    return result;
}
