set(model "${CMAKE_CURRENT_LIST_DIR}/../../results/4090-c161-dual-readout-v1.tflite")
file(SHA256 "${model}" digest)
if(NOT digest STREQUAL "a7776065ae71cbbccdfb2db8a76c5333805987353d475360e7fe5c7b69f47987")
    message(FATAL_ERROR "Unexpected auxiliary C161 model")
endif()
idf_component_register(SRCS "main.cpp" INCLUDE_DIRS "." REQUIRES esp-tflite-micro esp_timer
                       EMBED_FILES "${model}" "fixture.bin")
