idf_component_register(SRCS "panel_wake_runtime.cpp" INCLUDE_DIRS "."
    REQUIRES esp-sr esp-tflite-micro esp-micro-speech-features esp_timer)
if(CMAKE_BUILD_EARLY_EXPANSION)
    return()
endif()

set(_model_dir "${CMAKE_CURRENT_LIST_DIR}/../../models/hey-nyx-c161")
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process(COMMAND ${PYTHON} "${CMAKE_CURRENT_LIST_DIR}/prepare_streaming_runtime.py"
    --profile "${_model_dir}/profile.json" --model "${_model_dir}/model.tflite"
    --output "${_generated}" COMMAND_ERROR_IS_FATAL ANY)
target_compile_definitions(${COMPONENT_LIB} PRIVATE PANEL_WAKE_STREAMING_PROFILE=1)
target_include_directories(${COMPONENT_LIB} PRIVATE "${_generated}")
target_add_binary_data(${COMPONENT_LIB} "${_generated}/hey_nyx_model.bin" BINARY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_model_dir}/model.tflite" "${_model_dir}/profile.json"
    "${CMAKE_CURRENT_LIST_DIR}/prepare_streaming_runtime.py")
