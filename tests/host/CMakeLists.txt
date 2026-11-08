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
haCam_add_test(test_scanout_gate)
target_include_directories(test_scanout_gate PRIVATE ../../compat/include)
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
haCam_add_test(test_panel_voice_audio ../../components/panel_services/panel_voice_audio.c)
target_link_libraries(test_panel_voice_audio m)
haCam_add_test(test_panel_access ../../components/panel_services/panel_access.c)
haCam_add_test(test_panel_voice ../../components/panel_services/panel_voice_proto.c third_party/cjson/cJSON.c)
haCam_add_test(test_wifi_qr ../../components/panel_services/wifi_qr.c)
haCam_add_test(test_nyx_creds ../../components/panel_services/nyx_creds.c ../../components/panel_services/nvs_config.c)
target_include_directories(test_nyx_creds PRIVATE platform_shim)
add_executable(test_nyx_creds_dev test_nyx_creds.c
    ../../components/panel_services/nyx_creds.c ../../components/panel_services/nvs_config.c)
target_include_directories(test_nyx_creds_dev PRIVATE platform_shim)
target_compile_definitions(test_nyx_creds_dev PRIVATE NYX_NODE_NVS_NAMESPACE="nn_test"
    NYX_AGENT_NVS_NAMESPACE="na_test" NYX_NODE_ALLOW_BAKED_CREDENTIALS=0)
target_link_libraries(test_nyx_creds_dev unity)
add_test(NAME test_nyx_creds_dev COMMAND test_nyx_creds_dev)
find_package(OpenSSL REQUIRED)
haCam_add_test(test_nyx_security ../../components/panel_services/nyx_security.c third_party/cjson/cJSON.c)
target_include_directories(test_nyx_security PRIVATE crypto_shim)
target_link_libraries(test_nyx_security OpenSSL::Crypto)
target_compile_options(test_nyx_security PRIVATE -Wno-deprecated-declarations)

find_package(Threads REQUIRED)
haCam_add_test(test_media_mode ../../components/panel_services/media_mode.c)
target_include_directories(test_media_mode PRIVATE camera_shim platform_shim)
target_link_libraries(test_media_mode Threads::Threads)
set_tests_properties(test_media_mode PROPERTIES TIMEOUT 20)

haCam_add_test(test_panel_update ../../components/panel_services/panel_update_policy.c third_party/cjson/cJSON.c)
