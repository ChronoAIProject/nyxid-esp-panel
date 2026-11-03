idf_component_register(SRCS "NyxPanel.cpp" "assets/nyx_panel_icon.c" INCLUDE_DIRS "."
    REQUIRES brookesia_core panel_services panel_common Settings Camera product_support esp_driver_jpeg
    EMBED_FILES "assets/nyxbot-default.jpg")
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-missing-field-initializers)
target_include_directories(${COMPONENT_LIB} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../compat/include")
