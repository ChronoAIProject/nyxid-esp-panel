# Wake selection is independent of hardware and server environment.
set(NYX_PANEL_WAKE_ENGINE "hey-nyx" CACHE STRING "Wake detector: hey-nyx or hi-esp")
set_property(CACHE NYX_PANEL_WAKE_ENGINE PROPERTY STRINGS hey-nyx hi-esp)
if(NOT NYX_PANEL_WAKE_ENGINE STREQUAL "hey-nyx" AND
        NOT NYX_PANEL_WAKE_ENGINE STREQUAL "hi-esp")
    message(FATAL_ERROR "Unknown NYX_PANEL_WAKE_ENGINE: ${NYX_PANEL_WAKE_ENGINE}")
endif()
