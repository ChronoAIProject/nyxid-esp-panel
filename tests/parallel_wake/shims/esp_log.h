#pragma once
#include <cstdio>
#define ESP_LOGI(tag, format, ...) std::printf("%s " format "\n", tag, ##__VA_ARGS__)
