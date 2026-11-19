#include <stddef.h>
#include "util/mmio.h"

/* No POSIX mmap on this read-only FatFs volume. PocketSphinx uses its normal file reader. */
mmio_file_t *mmio_file_read(const char *name) { (void)name; return NULL; }
void *mmio_file_ptr(mmio_file_t *file) { (void)file; return NULL; }
void mmio_file_unmap(mmio_file_t *file) { (void)file; }
