#include "b64.h"

#include <stdbool.h>

static const char k_b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t b64_encoded_len(size_t input_len)
{
    return ((input_len + 2u) / 3u) * 4u;
}

size_t b64_decoded_max_len(size_t input_len)
{
    return (input_len / 4u) * 3u;
}

int b64_encode(const uint8_t *input, size_t input_len, char *out, size_t out_cap,
               size_t *out_len)
{
    size_t needed = b64_encoded_len(input_len);
    if (!out_len || (!input && input_len != 0u) || !out || out_cap < needed + 1u) {
        return -1;
    }

    size_t i = 0u;
    size_t j = 0u;
    while (i + 2u < input_len) {
        uint32_t n = ((uint32_t)input[i] << 16) |
                     ((uint32_t)input[i + 1u] << 8) |
                     (uint32_t)input[i + 2u];
        out[j++] = k_b64_table[(n >> 18) & 0x3fu];
        out[j++] = k_b64_table[(n >> 12) & 0x3fu];
        out[j++] = k_b64_table[(n >> 6) & 0x3fu];
        out[j++] = k_b64_table[n & 0x3fu];
        i += 3u;
    }

    if (i < input_len) {
        uint32_t n = (uint32_t)input[i] << 16;
        out[j++] = k_b64_table[(n >> 18) & 0x3fu];
        if (i + 1u < input_len) {
            n |= (uint32_t)input[i + 1u] << 8;
            out[j++] = k_b64_table[(n >> 12) & 0x3fu];
            out[j++] = k_b64_table[(n >> 6) & 0x3fu];
            out[j++] = '=';
        } else {
            out[j++] = k_b64_table[(n >> 12) & 0x3fu];
            out[j++] = '=';
            out[j++] = '=';
        }
    }

    out[j] = '\0';
    *out_len = j;
    return 0;
}

static int b64_value(char c)
{
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

int b64_decode(const char *input, size_t input_len, uint8_t *out, size_t out_cap,
               size_t *out_len)
{
    if (!out_len || (!input && input_len != 0u) || !out) {
        return -1;
    }
    if ((input_len % 4u) != 0u) {
        return -1;
    }

    size_t padding = 0u;
    if (input_len > 0u && input[input_len - 1u] == '=') {
        padding++;
    }
    if (input_len > 1u && input[input_len - 2u] == '=') {
        padding++;
    }

    size_t needed = b64_decoded_max_len(input_len) - padding;
    if (out_cap < needed) {
        return -1;
    }

    size_t j = 0u;
    for (size_t i = 0u; i < input_len; i += 4u) {
        int vals[4];
        bool pad2 = input[i + 2u] == '=';
        bool pad3 = input[i + 3u] == '=';

        if ((pad2 || pad3) && i + 4u != input_len) {
            return -1;
        }
        if (pad2 && !pad3) {
            return -1;
        }

        vals[0] = b64_value(input[i]);
        vals[1] = b64_value(input[i + 1u]);
        vals[2] = pad2 ? 0 : b64_value(input[i + 2u]);
        vals[3] = pad3 ? 0 : b64_value(input[i + 3u]);
        if (vals[0] < 0 || vals[1] < 0 || vals[2] < 0 || vals[3] < 0) {
            return -1;
        }

        uint32_t n = ((uint32_t)vals[0] << 18) |
                     ((uint32_t)vals[1] << 12) |
                     ((uint32_t)vals[2] << 6) |
                     (uint32_t)vals[3];

        out[j++] = (uint8_t)((n >> 16) & 0xffu);
        if (!pad2) {
            out[j++] = (uint8_t)((n >> 8) & 0xffu);
        }
        if (!pad3) {
            out[j++] = (uint8_t)(n & 0xffu);
        }
    }

    *out_len = j;
    return 0;
}
