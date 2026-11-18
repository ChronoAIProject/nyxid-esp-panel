set(results "${CMAKE_CURRENT_LIST_DIR}/../../results")
set(panel "${CMAKE_CURRENT_LIST_DIR}/../../../../components/panel_services")
set(waves "")
foreach(channel IN ITEMS digital near far)
    list(APPEND waves "${results}/4090-c161-coded-cue-v5-command-${channel}.wav")
endforeach()
idf_component_register(SRCS "main.cpp" "${panel}/panel_voice_audio.c"
    INCLUDE_DIRS "../../portable" "${panel}" PRIV_REQUIRES esp_audio_codec
    EMBED_FILES ${waves})
