cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(nyx_midlevel_selftest)

set(NYX_REFERENCE_OPS "" CACHE STRING "esp-nn kernels to replay with their TFLM reference branches")
if(NYX_REFERENCE_OPS)
    idf_component_get_property(micro_dir espressif__esp-tflite-micro COMPONENT_DIR)
    foreach(op IN LISTS NYX_REFERENCE_OPS)
        if(NOT op MATCHES "^(fully_connected|conv|depthwise_conv)$")
            message(FATAL_ERROR "Unknown midlevel probe kernel ${op}")
        endif()
        set_property(SOURCE "${micro_dir}/tensorflow/lite/micro/kernels/esp_nn/${op}.cc"
                     DIRECTORY "${micro_dir}" APPEND PROPERTY COMPILE_OPTIONS -UESP_NN)
    endforeach()
endif()

option(NYX_TRACE_TENSORS "Preserve intermediate tensors and trace score mismatches" OFF)
if(NYX_TRACE_TENSORS)
    idf_component_get_property(probe_main main COMPONENT_LIB)
    target_compile_definitions(${probe_main} PRIVATE NYX_TRACE_TENSORS=1)
endif()

option(NYX_FC_SINGLE_ROUNDING "Single-round the reference fully connected requantization" ON)
if(NYX_FC_SINGLE_ROUNDING)
    idf_component_get_property(micro_dir espressif__esp-tflite-micro COMPONENT_DIR)
    set_property(SOURCE "${micro_dir}/tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc"
                 DIRECTORY "${micro_dir}" APPEND PROPERTY COMPILE_OPTIONS
                 -UESP_NN -DMultiplyByQuantizedMultiplier=NyxFcSingleRoundMultiplier)
    idf_component_get_property(micro_lib espressif__esp-tflite-micro COMPONENT_LIB)
    target_sources(${micro_lib} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/main/fc_rounding.cpp")
endif()
