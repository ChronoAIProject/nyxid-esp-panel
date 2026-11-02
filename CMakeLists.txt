# For more information about build system see
# https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html
# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)
if(NOT DEFINED SDKCONFIG_DEFAULTS)
    set(SDKCONFIG_DEFAULTS "sdkconfig.defaults;sdkconfig.rev1_3.defaults;sdkconfig.board-16mb.defaults")
endif()
set(EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/examples/ethernet/basic/components/ethernet_init")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PANEL_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
set(PROJECT_VER "${PANEL_VERSION}")
set(EXCLUDE_COMPONENTS brookesia_app_squareline_demo)
set(ENV{NYX_PANEL_PROJECT_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
project(nyxid-esp-panel)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_managed_component_patches.cmake")
# Component Manager resolves environment variables in local lock paths. Keep
# those paths portable while retaining every registry version/hash in the lock.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.lock")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.lock" _panel_lock)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "$NYX_PANEL_PROJECT_DIR/" _panel_lock "${_panel_lock}")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.lock" "${_panel_lock}")
endif()
