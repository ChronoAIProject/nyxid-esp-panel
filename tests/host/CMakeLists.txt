cmake_minimum_required(VERSION 3.16)
project(haCam_host_tests C)

set(CMAKE_C_STANDARD 11)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter)
include_directories(unity ../../components/panel_services)
include_directories(third_party/cjson)

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
haCam_add_test(test_mailbox ../../components/panel_services/mailbox.c)
haCam_add_test(test_nyx_link_sm ../../components/panel_services/nyx_link_sm.c)
haCam_add_test(test_stream_sm ../../components/panel_services/stream_sm.c)
haCam_add_test(test_lcd_stack ../../components/panel_services/lcd_stack.c)
haCam_add_test(test_b64 ../../components/panel_services/b64.c)
haCam_add_test(test_nyx_proto
    ../../components/panel_services/nyx_proto.c
    ../../components/panel_services/cjson_tiny.c
    ../../components/panel_services/b64.c
    third_party/cjson/cJSON.c)
haCam_add_test(test_status_bar ../../components/panel_services/status_bar.c)
haCam_add_test(test_nyx_upload ../../components/panel_services/nyx_upload.c)
haCam_add_test(test_nyx_ws_rx ../../components/panel_services/nyx_ws_rx.c)
haCam_add_test(test_device_login ../../components/panel_services/device_login_proto.c third_party/cjson/cJSON.c)
target_link_libraries(test_device_login m)
haCam_add_test(test_wifi_qr ../../components/panel_services/wifi_qr.c)
haCam_add_test(test_nyx_creds ../../components/panel_services/nyx_creds.c ../../components/panel_services/nvs_config.c)
target_include_directories(test_nyx_creds PRIVATE platform_shim)
find_package(OpenSSL REQUIRED)
haCam_add_test(test_nyx_security ../../components/panel_services/nyx_security.c third_party/cjson/cJSON.c)
target_include_directories(test_nyx_security PRIVATE crypto_shim)
target_link_libraries(test_nyx_security OpenSSL::Crypto)
target_compile_options(test_nyx_security PRIVATE -Wno-deprecated-declarations)
