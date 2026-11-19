#pragma once
#include <stdio.h>
#include <stdlib.h>
FILE *packed_gaussian_open(const char *name, void **owner);
static inline void packed_gaussian_close(FILE *fp, void *owner)
{
    fclose(fp);
    free(owner);
}
