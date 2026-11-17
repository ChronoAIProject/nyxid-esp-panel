# Select a build-local BSP. Never let one profile modify another's managed BSP.
idf_component_get_property(_panel_bsp_dir ${PANEL_BOARD_BSP_COMPONENT} COMPONENT_DIR)
idf_component_get_property(_panel_bsp_lib ${PANEL_BOARD_BSP_COMPONENT} COMPONENT_LIB)
set(_panel_bsp_root "${CMAKE_BINARY_DIR}/board_bsp")
set(_panel_bsp_copy "${_panel_bsp_root}/waveshare__esp32_p4_platform")
set(_panel_board_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/waveshare_lcd_x8_board.patch")
foreach(_panel_file "esp32_p4_platform.c" "include/bsp/esp32_p4_platform.h")
    configure_file("${_panel_bsp_dir}/${_panel_file}" "${_panel_bsp_copy}/${_panel_file}" COPYONLY)
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_panel_board_patch}")
# Older builds patched managed_components in place. Normalize only our copy.
execute_process(COMMAND git apply --reverse --check "${_panel_board_patch}"
    WORKING_DIRECTORY "${_panel_bsp_root}" RESULT_VARIABLE _panel_bsp_patched
    OUTPUT_QUIET ERROR_QUIET)
if(_panel_bsp_patched EQUAL 0)
    execute_process(COMMAND git apply --reverse "${_panel_board_patch}"
        WORKING_DIRECTORY "${_panel_bsp_root}" COMMAND_ERROR_IS_FATAL ANY)
endif()
if(PANEL_BOARD_BSP_PATCH)
    set(_panel_board_patch "${CMAKE_CURRENT_SOURCE_DIR}/${PANEL_BOARD_BSP_PATCH}")
    execute_process(COMMAND git apply --check "${_panel_board_patch}"
        WORKING_DIRECTORY "${_panel_bsp_root}" COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND git apply "${_panel_board_patch}"
        WORKING_DIRECTORY "${_panel_bsp_root}" COMMAND_ERROR_IS_FATAL ANY)
endif()
get_target_property(_panel_bsp_sources ${_panel_bsp_lib} SOURCES)
set(_panel_bsp_replacements 0)
foreach(_panel_source IN LISTS _panel_bsp_sources)
    if(_panel_source MATCHES "(^|/)esp32_p4_platform[.]c$")
        list(REMOVE_ITEM _panel_bsp_sources "${_panel_source}")
        math(EXPR _panel_bsp_replacements "${_panel_bsp_replacements}+1")
    endif()
endforeach()
if(NOT _panel_bsp_replacements EQUAL 1)
    message(FATAL_ERROR "Expected exactly one pinned BSP implementation")
endif()
list(APPEND _panel_bsp_sources "${_panel_bsp_copy}/esp32_p4_platform.c")
set_property(TARGET ${_panel_bsp_lib} PROPERTY SOURCES "${_panel_bsp_sources}")
target_include_directories(${_panel_bsp_lib} BEFORE PUBLIC "${_panel_bsp_copy}/include")
target_include_directories(${_panel_bsp_lib} PRIVATE "${_panel_bsp_dir}/priv_include")
# These boards use the same DPI/LVGL DMA ownership backport. Other BSP
# adapters can choose their own display transport without this assumption.
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/panel_lcd_compat.cmake")
