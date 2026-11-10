idf_component_register(SRCS "probe.cpp" INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro esp_timer
    EMBED_FILES "head.tflite" "inputs.bin" "expected.bin")
if(NYX_HEAD_PROBE_TRACE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_HEAD_PROBE_TRACE=1)
endif()
