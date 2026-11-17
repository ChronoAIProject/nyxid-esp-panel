#pragma once
#include <cstdio>
#define ESP_LOGI(tag, format, ...) std::printf("%s " format "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, format, ...) std::fprintf(stderr, "%s " format "\n", tag, ##__VA_ARGS__)
