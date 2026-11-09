idf_component_register(SRCS "probe.cpp" INCLUDE_DIRS "../generated"
    REQUIRES esp-tflite-micro esp-micro-speech-features esp_timer
    EMBED_FILES "../generated/okay_nabu.tflite" "../generated/smoke.tflite")
