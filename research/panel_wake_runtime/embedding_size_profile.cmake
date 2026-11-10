# Optional CMAKE_PROJECT_nyxid-esp-panel_INCLUDE for the private -Os experiment.
# Keep the measured scalar frontend/FC adapter at its original optimization.
# The calculator also retains its release flags: GCC -Os diagnoses its existing
# bounded history display as format-truncation. No warnings are suppressed.
# ESP-IDF creates component targets after CMake's project include returns.
# A matched Hi ESP control has no experimental-runtime component.
function(nyx_research_restore_component_optimization)
    if(TARGET __idf_panel_wake_runtime)
        target_compile_options(__idf_panel_wake_runtime PRIVATE -O2)
    endif()
    target_compile_options(__idf_brookesia_app_calculator PRIVATE -O2)
endfunction()
cmake_language(DEFER CALL nyx_research_restore_component_optimization)
