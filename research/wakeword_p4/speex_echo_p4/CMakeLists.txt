cmake_minimum_required(VERSION 3.16)
set(COMPONENTS main)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(nyx_speex_echo)

if(NYX_SPEEX_FLOAT AND NYX_QEMU_FPU)
    idf_component_get_property(riscv_dir riscv COMPONENT_DIR)
    set_property(SOURCE "${riscv_dir}/vectors.S" DIRECTORY "${riscv_dir}"
                 APPEND PROPERTY COMPILE_OPTIONS -include "${CMAKE_CURRENT_SOURCE_DIR}/../pocketsphinx_p4_selftest/qemu_fpu.h")
endif()
