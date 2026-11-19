if(NYX_ONSET_EVIDENCE)
    set(model "${CMAKE_CURRENT_LIST_DIR}/../../results/4090-c161-onset-evidence-posterior-int8.tflite")
    idf_component_register(SRCS "onset_evidence.cpp" INCLUDE_DIRS "." REQUIRES esp-tflite-micro esp_timer spi_flash
                           EMBED_FILES "${model}")
    return()
endif()

set(model "${CMAKE_CURRENT_LIST_DIR}/../../results/4090-c161-midlevel-single-v1.tflite")
set(fixture "${CMAKE_CURRENT_LIST_DIR}/../../../../.local/wakeword/c161-midlevel-p4/midlevel.bin")
file(SHA256 "${model}" model_sha)
file(SHA256 "${fixture}" fixture_sha)
if(NOT model_sha STREQUAL "217f1a10a31eef92a8ca0baf95f0c41f0bbf72ae2335071c96423d3a1d4cb331" OR
   NOT fixture_sha STREQUAL "9730756e917a32074e30507f7857ce96fe6758ed543853a895acc3873db4df55")
    message(FATAL_ERROR "Unexpected midlevel model or forward fixture")
endif()
idf_component_register(SRCS "main.cpp" INCLUDE_DIRS "." REQUIRES esp-tflite-micro esp_timer spi_flash
                       EMBED_FILES "${model}" "${fixture}")
