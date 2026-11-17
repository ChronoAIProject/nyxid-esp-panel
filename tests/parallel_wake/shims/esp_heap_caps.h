#pragma once
#include <cstddef>
constexpr unsigned MALLOC_CAP_INTERNAL=1, MALLOC_CAP_8BIT=2;
void* heap_caps_aligned_alloc(std::size_t alignment, std::size_t size, unsigned caps);
void* heap_caps_malloc(std::size_t size, unsigned caps);
void heap_caps_free(void* data);
