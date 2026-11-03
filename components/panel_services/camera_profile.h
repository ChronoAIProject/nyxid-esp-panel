#pragma once

// Match the vendor Camera app on the 800x1280 OV5647 board. The sensor runs
// at its native 50 FPS only while a capture or native preview owns the camera.
#define PANEL_CAMERA_WIDTH 800u
#define PANEL_CAMERA_HEIGHT 1280u
#define PANEL_CAMERA_FRAME_BYTES (PANEL_CAMERA_WIDTH * PANEL_CAMERA_HEIGHT * 2u)
#define PANEL_CAMERA_BUFFER_COUNT 2u
#define PANEL_CAMERA_TASK_STACK 8192u
#define PANEL_CAMERA_TASK_PRIORITY 8u
#define PANEL_CAMERA_TASK_CORE 0
// Cold captures need a short exposure/white-balance settling period. Existing
// preview sessions already satisfy it, so their snapshots remain immediate.
#define PANEL_CAMERA_WARMUP_US 1000000LL
#define PANEL_CAMERA_QR_WIDTH 200u
#define PANEL_CAMERA_QR_HEIGHT 320u
#define PANEL_CAMERA_QR_BYTES (PANEL_CAMERA_QR_WIDTH * PANEL_CAMERA_QR_HEIGHT * 2u)
