#pragma once
#include <stdio.h>
#include <stdint.h>
#include "pocketsphinx.h"

FILE *address_sendump_open(const char *path);
int address_cases(ps_decoder_t *decoder, const char *cmn, FILE *input, int16_t *pcm, size_t capacity);
