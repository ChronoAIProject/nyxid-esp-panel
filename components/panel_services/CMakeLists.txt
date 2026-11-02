set(panel_sources
    b64.c
    cjson_tiny.c
    devices.c
    heap_diag.c
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
    panel_runtime.c
    provisioning.c
    qr_scanner.c
    service_registrar.c
    snapshot.c
    status_bar.c
    stream_publisher.c
    stream_sm.c
    webrtc.c
    whip_signaling.c
)
if(CONFIG_HA_CAM_DEVICE_CODE)
    list(APPEND panel_sources "device_code.c")
    if(CONFIG_HA_CAM_DEVICE_CODE_MOCK)
        list(APPEND panel_sources "device_code_mock.c")
    endif()
endif()
idf_component_register(
    SRCS ${panel_sources} "fonts/lv_font_noto_sans_sc_med_20.c"
    INCLUDE_DIRS "."
    REQUIRES av_render panel_common console esp_app_format esp_capture
        esp_https_ota esp_http_client esp_http_server app_update esp_peer
        esp_timer esp_websocket_client esp_new_jpeg esp_driver_jpeg esp_video
        esp_webrtc json lvgl media_lib_sal nvs_flash quirc
        waveshare__esp32_p4_platform product_support)
