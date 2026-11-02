#pragma once

#include <stddef.h>
#include <stdint.h>

size_t b64_encoded_len(size_t input_len);
size_t b64_decoded_max_len(size_t input_len);

int b64_encode(const uint8_t *input, size_t input_len, char *out, size_t out_cap,
               size_t *out_len);
int b64_decode(const char *input, size_t input_len, uint8_t *out, size_t out_cap,
               size_t *out_len);
