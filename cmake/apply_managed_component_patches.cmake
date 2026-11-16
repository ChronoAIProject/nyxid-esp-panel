set(_ha_cam_managed_component_patches
    "webrtc_h264_dma_psram.patch"
    "esp_capture_select_enabled_paths.patch"
    "gmf_video_force_sw_mjpeg.patch"
    "esp_video_codec_sw_mjpeg_rgb565.patch"
    "esp_lvgl_adapter_dma_scanout.patch"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/managed_components")
    foreach(_ha_cam_patch_name IN LISTS _ha_cam_managed_component_patches)
        set(_ha_cam_patch_file "${CMAKE_CURRENT_LIST_DIR}/../patches/${_ha_cam_patch_name}")
        string(REPLACE "." "_" _ha_cam_patch_stamp_name "${_ha_cam_patch_name}")
        set(_ha_cam_patch_stamp "${CMAKE_BINARY_DIR}/.ha_cam_${_ha_cam_patch_stamp_name}_patched")

        if(EXISTS "${_ha_cam_patch_file}")
            execute_process(
                COMMAND git apply --check --directory managed_components "${_ha_cam_patch_file}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                RESULT_VARIABLE _ha_cam_patch_check
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_ha_cam_patch_check EQUAL 0)
                execute_process(
                    COMMAND git apply --directory managed_components "${_ha_cam_patch_file}"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    RESULT_VARIABLE _ha_cam_patch_apply
                )
                if(NOT _ha_cam_patch_apply EQUAL 0)
                    message(FATAL_ERROR "Failed to apply ${_ha_cam_patch_file}")
                endif()
                file(WRITE "${_ha_cam_patch_stamp}" "applied\n")
            else()
                execute_process(
                    COMMAND git apply --reverse --check --directory managed_components "${_ha_cam_patch_file}"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    RESULT_VARIABLE _ha_cam_patch_reverse_check
                    OUTPUT_QUIET
                    ERROR_QUIET
                )
                if(_ha_cam_patch_reverse_check EQUAL 0)
                    file(WRITE "${_ha_cam_patch_stamp}" "already-present\n")
                else()
                    message(FATAL_ERROR "Managed component patch ${_ha_cam_patch_name} could not be applied and is not already present")
                endif()
            endif()
        endif()
    endforeach()
endif()
