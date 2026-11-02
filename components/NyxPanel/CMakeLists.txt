idf_component_register(SRCS "NyxPanel.cpp" INCLUDE_DIRS "."
    REQUIRES brookesia_core panel_services Settings Camera product_support esp_driver_jpeg)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-missing-field-initializers)
