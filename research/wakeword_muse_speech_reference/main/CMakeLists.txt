# Compile the unchanged, already exercised complete-PCM UART detector path.
# The separate project changes only factory/model binding and pins the Muse ABI.
idf_component_register(
    SRCS "../../wakeword_baseline/main/baseline.c" "../../wakeword_muse_reference/main/layout.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-sr espressif__zlib esp_driver_uart esp_timer)
idf_component_get_property(sr_dir espressif__esp-sr COMPONENT_DIR)
set(model_dir "${sr_dir}/model/wakenet_model/wn10_himuse")
set(model_binary "${CMAKE_CURRENT_BINARY_DIR}/muse-model.bin")
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    BASELINE_MODEL_NAME="wn10_himuse"
    BASELINE_REPORT_RESOURCES=1
    BASELINE_THRESHOLD_CONTROL=1
    BASELINE_LONG_STREAMS=0
    BASELINE_USE_MODEL_FACTORY=1
    BASELINE_PACKED_START="_binary_muse_model_bin_start"
    BASELINE_PACKED_END="_binary_muse_model_bin_end")
add_custom_command(OUTPUT "${model_binary}"
    COMMAND ${PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/../../wakeword_muse_reference/pack_model.py"
        "${model_dir}" "${model_binary}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../wakeword_muse_reference/pack_model.py"
        "${sr_dir}/model/movemodel.py" "${sr_dir}/model/pack_model.py"
        "${model_dir}/_MODEL_INFO_p1" "${model_dir}/_MODEL_INFO_p2"
        "${model_dir}/wn10_data_p1" "${model_dir}/wn10_data_p2"
    VERBATIM)
add_custom_target(muse_model DEPENDS "${model_binary}")
add_dependencies(${COMPONENT_LIB} muse_model)
target_add_binary_data(${COMPONENT_LIB} "${model_binary}" BINARY)
