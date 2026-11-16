idf_component_register(SRCS "panel_board.c"
    INCLUDE_DIRS "include" "${PANEL_BOARD_GENERATED_DIR}")
# Keep the IDF custom descriptor even when only compile-time metadata is used.
target_link_libraries(${COMPONENT_LIB} INTERFACE "-u panel_board_image_desc")
