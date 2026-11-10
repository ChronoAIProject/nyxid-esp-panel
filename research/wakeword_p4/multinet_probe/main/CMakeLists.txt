idf_component_register(SRCS "probe.cpp" INCLUDE_DIRS "."
    REQUIRES espressif__esp-sr espressif__zlib esp_timer vfs
    EMBED_FILES "model.bin" "fixtures.bin" "commands_en.txt")
