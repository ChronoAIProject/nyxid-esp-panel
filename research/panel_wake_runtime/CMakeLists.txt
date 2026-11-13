if(CMAKE_BUILD_EARLY_EXPANSION)
    # IDF's dependency scan does not receive the model-path cache selector.
    # Declare dependencies for both opt-in adapters in that early pass.
    idf_component_register(REQUIRES esp-sr esp-tflite-micro esp-micro-speech-features
        esp_timer espressif__zlib PRIV_REQUIRES esp_driver_i2s)
    return()
endif()
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_generated}")
set(NYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE "" CACHE FILEPATH "Frozen225-frame development profile with complete NYXKWS02 self-test")
set(NYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE "" CACHE PATH "Private frozen embedding/equal-mean bundle for resource diagnostics")
if(NYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE OR NYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE)
        message(FATAL_ERROR "A teacher60 candidate profile cannot configure an embedding bundle")
    endif()
    option(NYX_PANEL_EXPERIMENTAL_WAKE_IRAM "Measure hot detector code in internal RAM" OFF)
    set(_wake_linker_fragments)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_IRAM)
        list(APPEND _wake_linker_fragments "embedding_iram.lf")
    endif()
    idf_component_register(SRCS "embedding_runtime.cpp" INCLUDE_DIRS "."
        PRIV_INCLUDE_DIRS "${_generated}"
        REQUIRES esp-sr esp-tflite-micro esp_timer espressif__zlib
        PRIV_REQUIRES esp_driver_i2s
        LDFRAGMENTS ${_wake_linker_fragments})
    target_link_libraries(${COMPONENT_LIB} PRIVATE zlibstatic)
    option(NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC "Measure a bounded detector consumer on core1" OFF)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_PANEL_EXPERIMENTAL_WAKE_ASYNC=1)
    endif()
    option(NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE "Enable verbose operator/scheduler attribution (perturbs synchronous capture)" OFF)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE=1)
    endif()
    option(NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS "Count hardware RX queue loss without reading PCM" OFF)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS)
        if(NOT NYX_PANEL_RESEARCH_WAKE_TIMING)
            target_sources(${COMPONENT_LIB} PRIVATE "embedding_capture.cpp")
            target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,--wrap=i2s_new_channel")
        endif()
        target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS=1)
    endif()
    set(NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP "" CACHE STRING "Explicit development LAN receiver for private input/numeric tracing; empty disables")
    set(NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT "30282" CACHE STRING "Development diagnostic UDP receiver port")
    if(NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP)
        if(NOT NYX_PANEL_NODE_DEV_MODE)
            message(FATAL_ERROR "Wake input tracing requires the isolated node development profile")
        endif()
        if(NOT NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP MATCHES "^(10\\.[0-9]+\\.[0-9]+\\.[0-9]+|192\\.168\\.[0-9]+\\.[0-9]+|172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]+\\.[0-9]+)$")
            message(FATAL_ERROR "Wake trace receiver must be an explicit private LAN IPv4 address")
        endif()
        string(REPLACE "." ";" _trace_octets "${NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP}")
        list(LENGTH _trace_octets _trace_octet_count)
        if(NOT _trace_octet_count EQUAL 4)
            message(FATAL_ERROR "Wake trace receiver must be an explicit IPv4 address")
        endif()
        foreach(_octet IN LISTS _trace_octets)
            if(NOT "${_octet}" MATCHES "^[0-9]+$" OR _octet GREATER 255)
                message(FATAL_ERROR "Invalid wake trace IPv4 octet")
            endif()
        endforeach()
        if(NOT "${NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT}" MATCHES "^[0-9]+$" OR
                NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT LESS 1024 OR
                NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT GREATER 65535)
            message(FATAL_ERROR "Wake trace port must be 1024..65535")
        endif()
        idf_component_optional_requires(PRIVATE lwip)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE
            NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP="${NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP}"
            NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT=${NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT})
    endif()
    execute_process(COMMAND ${PYTHON} "${CMAKE_CURRENT_LIST_DIR}/prepare_embedding_runtime.py"
        --bundle "${NYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE}"
        --model "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}"
        --calibration "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
        --output "${_generated}"
        COMMAND_ERROR_IS_FATAL ANY)
    foreach(_name backbone dense temporal)
        target_add_binary_data(${COMPONENT_LIB} "${_generated}/embedding_${_name}.bin" BINARY)
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${NYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE}/manifest.json"
        "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
        "${CMAKE_CURRENT_LIST_DIR}/prepare_embedding_runtime.py")
    return()
endif()
idf_component_register(SRCS "panel_wake_runtime.cpp" INCLUDE_DIRS "."
    PRIV_INCLUDE_DIRS "${_generated}"
    REQUIRES esp-sr esp-tflite-micro esp-micro-speech-features esp_timer)
set(_selftest_args)
set(_candidate_profile_args)
set(_prepare_script "${CMAKE_CURRENT_LIST_DIR}/prepare_runtime.py")
if(NYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE)
    if(NYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE OR NOT NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST)
        message(FATAL_ERROR "Complete profile requires its NYXKWS02 self-test and cannot use the legacy GPU profile")
    endif()
    get_filename_component(_complete_profile "${NYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    set(_prepare_script "${CMAKE_CURRENT_LIST_DIR}/prepare_complete_runtime.py")
    list(APPEND _candidate_profile_args --profile "${_complete_profile}")
    file(READ "${_complete_profile}" _complete_profile_json)
    get_filename_component(_complete_profile_dir "${_complete_profile}" DIRECTORY)
    foreach(_artifact model training completion calibration execution selftest)
        string(JSON _artifact_path GET "${_complete_profile_json}" artifacts "${_artifact}" path)
        get_filename_component(_artifact_absolute "${_artifact_path}" ABSOLUTE BASE_DIR "${_complete_profile_dir}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_artifact_absolute}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_complete_profile}" "${_prepare_script}")
elseif(NYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE)
    list(APPEND _candidate_profile_args --gpu-profile "${NYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${NYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE}")
endif()
if(NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST)
    list(APPEND _selftest_args --selftest "${NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST}")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PANEL_WAKE_SELFTEST=1)
endif()
execute_process(
    COMMAND ${PYTHON} "${_prepare_script}"
        --model "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}"
        --calibration "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
        --output "${_generated}"
        ${_selftest_args}
        ${_candidate_profile_args}
    RESULT_VARIABLE _prepare_result
    COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(_model_directory "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}" DIRECTORY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${NYX_PANEL_EXPERIMENTAL_WAKE_MODEL}"
    "${_model_directory}/training.json"
    "${NYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION}"
    "${CMAKE_CURRENT_LIST_DIR}/prepare_runtime.py"
    "${CMAKE_CURRENT_LIST_DIR}/plans/gpu-v30-runtime-v1.json")
target_add_binary_data(${COMPONENT_LIB} "${_generated}/hey_nyx_model.bin" BINARY)
if(NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST}")
    target_add_binary_data(${COMPONENT_LIB} "${_generated}/wake_selftest.bin" BINARY)
endif()
