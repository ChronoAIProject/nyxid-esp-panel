idf_component_register(SRCS "baseline.c" INCLUDE_DIRS "."
    REQUIRES esp-sr espressif__zlib esp_driver_uart esp_timer)
idf_component_get_property(sr_dir espressif__esp-sr COMPONENT_DIR)
set(BASELINE_WAKE_MODEL "wn9_hiesp" CACHE STRING "Pinned offline reference model")
set_property(CACHE BASELINE_WAKE_MODEL PROPERTY STRINGS wn9_hiesp wn9_heywillow_tts)
if(BASELINE_WAKE_MODEL STREQUAL "wn9_hiesp")
    set(wake_basename "baseline-hi-esp")
    set(wake_factory 0)
elseif(BASELINE_WAKE_MODEL STREQUAL "wn9_heywillow_tts")
    set(wake_basename "baseline-hey-willow")
    set(wake_factory 1)
else()
    message(FATAL_ERROR "Unsupported BASELINE_WAKE_MODEL: ${BASELINE_WAKE_MODEL}")
endif()
set(wake_source "${sr_dir}/model/wakenet_model/${BASELINE_WAKE_MODEL}")
set(wake_binary "${CMAKE_CURRENT_BINARY_DIR}/${wake_basename}.bin")
string(REPLACE "-" "_" wake_symbol "${wake_basename}")
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    BASELINE_MODEL_NAME="${BASELINE_WAKE_MODEL}"
    BASELINE_USE_MODEL_FACTORY=${wake_factory}
    BASELINE_PACKED_START="_binary_${wake_symbol}_bin_start"
    BASELINE_PACKED_END="_binary_${wake_symbol}_bin_end")
add_custom_command(OUTPUT "${wake_binary}"
    COMMAND ${PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/embed_wake_model.py" "${wake_source}" "${wake_binary}"
        --model-name "${BASELINE_WAKE_MODEL}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/embed_wake_model.py"
        "${wake_source}/_MODEL_INFO_" "${wake_source}/wn9_index" "${wake_source}/wn9_data"
    VERBATIM)
add_custom_target(baseline_model DEPENDS "${wake_binary}")
add_dependencies(${COMPONENT_LIB} baseline_model)
target_add_binary_data(${COMPONENT_LIB} "${wake_binary}" BINARY)
