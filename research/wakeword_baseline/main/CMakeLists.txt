idf_component_register(SRCS "baseline.c" INCLUDE_DIRS "."
    REQUIRES esp-sr espressif__zlib esp_driver_uart esp_timer)
idf_component_get_property(sr_dir espressif__esp-sr COMPONENT_DIR)
set(wake_source "${sr_dir}/model/wakenet_model/wn9_hiesp")
set(wake_binary "${CMAKE_CURRENT_BINARY_DIR}/baseline-hi-esp.bin")
add_custom_command(OUTPUT "${wake_binary}"
    COMMAND ${PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/embed_wake_model.py" "${wake_source}" "${wake_binary}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/embed_wake_model.py"
        "${wake_source}/_MODEL_INFO_" "${wake_source}/wn9_index" "${wake_source}/wn9_data"
    VERBATIM)
add_custom_target(baseline_model DEPENDS "${wake_binary}")
add_dependencies(${COMPONENT_LIB} baseline_model)
target_add_binary_data(${COMPONENT_LIB} "${wake_binary}" BINARY)
