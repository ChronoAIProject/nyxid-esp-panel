cmake_minimum_required(VERSION 3.16)
set(COMPONENTS main)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(nyx_lexical_probe)

if(CONFIG_NYX_LEXICAL_QEMU_RAM)
    # The emulated ECO5 core omits the FPU reason bit on a rounding-CSR read.
    # Use IDF's instruction decoder for lazy-FPU recovery in this vector object.
    idf_component_get_property(riscv_dir riscv COMPONENT_DIR)
    set_property(SOURCE "${riscv_dir}/vectors.S" DIRECTORY "${riscv_dir}"
                 APPEND PROPERTY COMPILE_OPTIONS -include "${CMAKE_CURRENT_SOURCE_DIR}/qemu_fpu.h")
endif()
