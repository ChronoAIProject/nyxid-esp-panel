set(panel_sources
    b64.c
    cjson_tiny.c
    devices.c
    device_code.c
    device_login_proto.c
    heap_diag.c
    lcd_background.c
    lcd_image.c
    lcd_stack.c
    lcd_text.c
    local_api.c
    local_http_server.c
    mailbox.c
    media_mode.c
    media_sys.c
    net_manager.c
    nvs_config.c
    nyx_creds.c
    nyx_link_sm.c
    nyx_node.c
    nyx_proto.c
    nyx_security.c
    nyx_upload.c
    nyx_ws_rx.c
    panel_ota.c
    panel_update.c
    panel_update_policy.c
    panel_flash.c
    panel_runtime.c
    panel_access.c
    panel_voice.c
    panel_voice_audio.c
    panel_voice_proto.c
    provisioning.c
    qr_scanner.c
    service_registrar.c
    status_bar.c
    wifi_qr.c
)
idf_component_register(
    SRCS ${panel_sources} "fonts/lv_font_noto_sans_sc_med_20.c" "fonts/lv_font_noto_sans_sc_speech_20.c"
    INCLUDE_DIRS "."
    REQUIRES panel_common console esp_app_format
        esp_https_ota esp_http_client esp_http_server app_update
        esp_timer esp_websocket_client esp_new_jpeg esp_driver_jpeg esp_video
        json lvgl nvs_flash spi_flash quirc
        waveshare__esp32_p4_platform product_support esp_peer media_lib_sal esp_audio_codec esp-sr)

set(NYX_PANEL_AGENT_DEV_API_URL "" CACHE STRING "LAN NyxID API origin for Agent Key development")
set(NYX_PANEL_AGENT_DEV_LOGIN_URL "" CACHE STRING "LAN Agent Key verification page")
option(NYX_PANEL_NODE_DEV_MODE "Also use the isolated LAN NyxID node and services" OFF)
if(NYX_PANEL_NODE_DEV_MODE AND NOT NYX_PANEL_AGENT_DEV_API_URL)
    message(FATAL_ERROR "Node development mode requires the LAN Agent API/login profile")
endif()
if(NYX_PANEL_AGENT_DEV_API_URL OR NYX_PANEL_AGENT_DEV_LOGIN_URL)
    if(NOT NYX_PANEL_AGENT_DEV_API_URL MATCHES "^http://(10\\.[0-9]+\\.[0-9]+\\.[0-9]+|192\\.168\\.[0-9]+\\.[0-9]+|172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]+\\.[0-9]+)(:[0-9]+)?$")
        message(FATAL_ERROR "Agent development API must be an explicit private LAN HTTP origin")
    endif()
    set(_agent_api_host "${CMAKE_MATCH_1}")
    if(NOT NYX_PANEL_AGENT_DEV_LOGIN_URL MATCHES "^http://([^/:]+)(:[0-9]+)?/login/agent-key$")
        message(FATAL_ERROR "Agent development login must name /login/agent-key")
    endif()
    if(NOT "${CMAKE_MATCH_1}" STREQUAL "${_agent_api_host}")
        message(FATAL_ERROR "Agent development API and login must use the same LAN host")
    endif()
    string(SHA256 _agent_profile "${NYX_PANEL_AGENT_DEV_API_URL}|${NYX_PANEL_AGENT_DEV_LOGIN_URL}")
    string(SUBSTRING "${_agent_profile}" 0 10 _agent_profile)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        NYX_PANEL_UPDATE_DISABLED=1
        NYX_AGENT_BASE_URL="${NYX_PANEL_AGENT_DEV_API_URL}"
        NYX_AGENT_VERIFICATION_URI="${NYX_PANEL_AGENT_DEV_LOGIN_URL}"
        NYX_AGENT_NVS_NAMESPACE="na_${_agent_profile}"
        NYX_VOICE_NVS_NAMESPACE="pv_${_agent_profile}")
    if(NYX_PANEL_NODE_DEV_MODE)
        string(REGEX REPLACE "^http:" "ws:" _node_ws "${NYX_PANEL_AGENT_DEV_API_URL}")
        string(REGEX REPLACE "/login/agent-key$" "/login/device" _node_login "${NYX_PANEL_AGENT_DEV_LOGIN_URL}")
        target_compile_definitions(${COMPONENT_LIB} PRIVATE
            HA_CAM_NYX_BASE_URL="${NYX_PANEL_AGENT_DEV_API_URL}"
            HA_CAM_NYX_WSS_URL="${_node_ws}/api/v1/nodes/ws"
            NYX_NODE_VERIFICATION_URI="${_node_login}"
            NYX_NODE_NVS_NAMESPACE="nn_${_agent_profile}"
            NYX_NODE_SETUP_NVS_NAMESPACE="ns_${_agent_profile}"
            NYX_NODE_ALLOW_BAKED_CREDENTIALS=0)
    endif()
endif()

foreach(flash_api esp_flash_read esp_flash_write esp_flash_read_encrypted
                  esp_flash_write_encrypted esp_flash_erase_region esp_flash_erase_chip)
    target_link_libraries(${COMPONENT_LIB} INTERFACE "-Wl,--wrap=${flash_api}")
endforeach()
