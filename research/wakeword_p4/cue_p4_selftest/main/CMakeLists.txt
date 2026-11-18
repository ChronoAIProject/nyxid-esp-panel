set(cue "${CMAKE_CURRENT_LIST_DIR}/../../results/hye-nyx-c161-synthetic-cue-v3.wav")
file(SHA256 "${cue}" cue_hash)
if(NOT cue_hash STREQUAL "ed23ed4da338f70602976b20d161485bea9094f431de7ed0931252175585bdeb")
    message(FATAL_ERROR "Unexpected v3 PCM fixture")
endif()
idf_component_register(SRCS "main.cpp" INCLUDE_DIRS "../../portable" PRIV_REQUIRES esp_timer heap
                       EMBED_FILES "${cue}")
