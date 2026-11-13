idf_component_register(SRCS "muse.c" INCLUDE_DIRS "."
    REQUIRES esp-sr espressif__zlib esp_timer)
idf_component_get_property(sr_dir espressif__esp-sr COMPONENT_DIR)
set(model_dir "${sr_dir}/model/wakenet_model/wn10_himuse")
set(model_binary "${CMAKE_CURRENT_BINARY_DIR}/muse-model.bin")
add_custom_command(OUTPUT "${model_binary}"
    COMMAND ${PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/../pack_model.py" "${model_dir}" "${model_binary}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../pack_model.py"
        "${model_dir}/_MODEL_INFO_p1" "${model_dir}/_MODEL_INFO_p2"
        "${model_dir}/wn10_data_p1" "${model_dir}/wn10_data_p2"
    VERBATIM)
add_custom_target(muse_model DEPENDS "${model_binary}")
add_dependencies(${COMPONENT_LIB} muse_model)
target_add_binary_data(${COMPONENT_LIB} "${model_binary}" BINARY)
