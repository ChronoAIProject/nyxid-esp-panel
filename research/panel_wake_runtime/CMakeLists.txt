set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_generated}")
idf_component_register(SRCS "panel_wake_runtime.cpp" INCLUDE_DIRS "."
    PRIV_INCLUDE_DIRS "${_generated}"
    REQUIRES esp-sr esp-tflite-micro esp-micro-speech-features esp_timer)
set(_selftest_args)
if(NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST)
    list(APPEND _selftest_args --selftest "${NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST}")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PANEL_WAKE_SELFTEST=1)
endif()
execute_process(
    COMMAND ${PYTHON} "${CMAKE_CURRENT_LIST_DIR}/prepare_runtime.py"
        --model "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}"
        --calibration "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
        --output "${_generated}"
        ${_selftest_args}
    RESULT_VARIABLE _prepare_result
    COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(_model_directory "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}" DIRECTORY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}"
    "${_model_directory}/training.json"
    "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
    "${CMAKE_CURRENT_LIST_DIR}/prepare_runtime.py")
target_add_binary_data(${COMPONENT_LIB} "${_generated}/hey_nyx_model.bin" BINARY)
if(NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST}")
    target_add_binary_data(${COMPONENT_LIB} "${_generated}/wake_selftest.bin" BINARY)
endif()
