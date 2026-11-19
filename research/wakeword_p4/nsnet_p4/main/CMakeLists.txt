set(NYX_RNNOISE_SRC "$ENV{NYX_RNNOISE_SRC}")
if(NYX_RNNOISE_SRC)
    set(rn "${NYX_RNNOISE_SRC}/src")
    idf_component_register(SRCS "probe.c" "rnnoise16.c"
        "${rn}/denoise.c" "${rn}/pitch.c" "${rn}/kiss_fft.c" "${rn}/rnn.c"
        "${rn}/rnn_data.c" "${rn}/celt_lpc.c" "${rn}/rnn_reader.c"
        INCLUDE_DIRS "${NYX_RNNOISE_SRC}/include" "${rn}"
        PRIV_REQUIRES spi_flash esp_timer)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_RNNOISE RNN_BUILD)
    target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-error=maybe-uninitialized)
    return()
endif()
set(sr "$ENV{NYX_SR_COMPONENT}")
idf_component_register(SRCS "probe.c" "${sr}/src/model_path.c"
    INCLUDE_DIRS "${sr}/src/include" "${sr}/include/esp32p4"
    PRIV_REQUIRES spi_flash spiffs esp_timer espressif__dl_fft espressif__cjson)
target_add_binary_data(${COMPONENT_LIB} "$ENV{NYX_NSNET_PACK}" BINARY RENAME_TO nsnet_pack)
idf_component_get_property(fft espressif__dl_fft COMPONENT_LIB)
idf_component_get_property(cjson espressif__cjson COMPONENT_LIB)
target_link_libraries(${COMPONENT_LIB} PRIVATE "-Wl,--start-group"
    "${sr}/lib/esp32p4/libhufzip.a" "${sr}/lib/esp32p4/libnsnet.a" "${sr}/lib/esp32p4/libdl_lib.a"
    "${sr}/lib/esp32p4/libc_speech_features.a" "${sr}/lib/esp32p4/libesp_audio_processor.a"
    "$<TARGET_FILE:${fft}>" "$<TARGET_FILE:${cjson}>" "-Wl,--end-group")
if(NYX_NSNET_ANSI_FFT)
    target_sources(${COMPONENT_LIB} PRIVATE "../../first_command_p4/main/qemu_reference_fft.c")
    target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,-u,__wrap_dl_fft2r_fc32_arp4_" "-Wl,-u,__wrap_dl_ifft2r_fc32_arp4_"
        "-Wl,--wrap=dl_fft2r_fc32_arp4_" "-Wl,--wrap=dl_ifft2r_fc32_arp4_")
endif()
if(NYX_NSNET_BOARD)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NYX_NSNET_BOARD)
endif()
