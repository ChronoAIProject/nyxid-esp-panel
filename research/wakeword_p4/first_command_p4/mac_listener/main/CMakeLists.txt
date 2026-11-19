set(mac "$ENV{NYX_MAC_WORK}/src/components/panel_wake_runtime")
set(panel "${CMAKE_CURRENT_LIST_DIR}/../../../../../components/panel_services")
set(sr "$ENV{NYX_SR_COMPONENT}")
idf_component_register(SRCS "listener.cpp" "${mac}/panel_wake_runtime.cpp" "${panel}/panel_voice_audio.c"
                       "../../main/qemu_reference_fft.c"
                       INCLUDE_DIRS "${mac}" "$ENV{NYX_MAC_WORK}/model" "${panel}" "../../../portable"
                                    "${sr}/include/esp32p4"
                       REQUIRES features esp-tflite-micro espressif__dl_fft spi_flash esp_timer
                       EMBED_FILES "$ENV{NYX_MAC_WORK}/model/hey_nyx_model.bin")
target_compile_definitions(${COMPONENT_LIB} PRIVATE PANEL_WAKE_STREAMING_PROFILE=1)
target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,-u,__wrap_dl_fft2r_fc32_arp4_" "-Wl,-u,__wrap_dl_ifft2r_fc32_arp4_"
    "-Wl,--wrap=dl_fft2r_fc32_arp4_" "-Wl,--wrap=dl_ifft2r_fc32_arp4_")
idf_component_get_property(fft espressif__dl_fft COMPONENT_LIB)
target_link_libraries(${COMPONENT_LIB} PRIVATE "-Wl,--start-group" "${sr}/lib/esp32p4/libesp_audio_processor.a"
    "${sr}/lib/esp32p4/libdl_lib.a" "$<TARGET_FILE:${fft}>" "-Wl,--end-group")
# QEMU's lazy-FPU rounding CSR workaround masks ticks during the imported runtime.
# UART logging takes a newlib mutex, so disable progress printf inside those calls.
set_source_files_properties("${mac}/panel_wake_runtime.cpp" PROPERTIES COMPILE_DEFINITIONS "LOG_LOCAL_LEVEL=0")
