cmake_minimum_required(VERSION 3.16)
project(haCam_host_tests C)

set(CMAKE_C_STANDARD 11)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter)
include_directories(unity ../../main)
include_directories(../../managed_components/espressif__cjson/cJSON)

enable_testing()

add_library(unity STATIC unity/unity.c)
target_compile_definitions(unity PUBLIC UNITY_INCLUDE_CONFIG_H)
target_include_directories(unity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(haCam_add_test name)
    add_executable(${name} ${name}.c ${ARGN})
    target_link_libraries(${name} unity)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

haCam_add_test(test_smoke)
haCam_add_test(test_mailbox ../../main/mailbox.c)
haCam_add_test(test_nyx_link_sm ../../main/nyx_link_sm.c)
haCam_add_test(test_stream_sm ../../main/stream_sm.c)
haCam_add_test(test_lcd_stack ../../main/lcd_stack.c)
haCam_add_test(test_b64 ../../main/b64.c)
haCam_add_test(test_nyx_proto
    ../../main/nyx_proto.c
    ../../main/cjson_tiny.c
    ../../main/b64.c
    ../../managed_components/espressif__cjson/cJSON/cJSON.c)
haCam_add_test(test_status_bar ../../main/status_bar.c)
