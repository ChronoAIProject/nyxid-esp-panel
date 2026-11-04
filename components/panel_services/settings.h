#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// NyxID server
#ifndef HA_CAM_NYX_BASE_URL
#define HA_CAM_NYX_BASE_URL    "https://nyx-api.chrono-ai.fun"
#endif
#ifndef HA_CAM_NYX_WSS_URL
#define HA_CAM_NYX_WSS_URL     "wss://nyx-api.chrono-ai.fun/api/v1/nodes/ws"
#endif
#define HA_CAM_DEFAULT_LABEL   "esp32-cam"

// Outbound Agent Key login/voice can target a development NyxID independently
// of the production control node. Development builds provide origin-specific
// NVS namespaces so credentials never cross between servers.
#ifndef NYX_AGENT_BASE_URL
#define NYX_AGENT_BASE_URL HA_CAM_NYX_BASE_URL
#endif
#ifndef NYX_AGENT_VERIFICATION_URI
#define NYX_AGENT_VERIFICATION_URI "https://nyxid.ai/login/agent-key"
#endif
#ifndef NYX_AGENT_NVS_NAMESPACE
#define NYX_AGENT_NVS_NAMESPACE "nyx_agent"
#endif
#ifndef NYX_VOICE_NVS_NAMESPACE
#define NYX_VOICE_NVS_NAMESPACE "panel_voice"
#endif
#ifndef NYX_NODE_NVS_NAMESPACE
#define NYX_NODE_NVS_NAMESPACE "nyx"
#endif
#ifndef NYX_NODE_SETUP_NVS_NAMESPACE
#define NYX_NODE_SETUP_NVS_NAMESPACE "nyx_setup"
#endif
#ifndef NYX_NODE_VERIFICATION_URI
#define NYX_NODE_VERIFICATION_URI "https://nyxid.ai/login/device"
#endif
#ifndef NYX_NODE_ALLOW_BAKED_CREDENTIALS
#define NYX_NODE_ALLOW_BAKED_CREDENTIALS 1
#endif

// Defensive HMAC support for node -> NyxID frames. The server currently signs
// proxy_request frames to nodes but does not document verification for auth or
// proxy_response frames from nodes, so outbound signing stays off by default.
#ifndef HA_CAM_HMAC_SIGN_FRAMES
#define HA_CAM_HMAC_SIGN_FRAMES 0
#endif

// Hidden endpoint used for hardware-independent QR scanner self-tests.
#ifndef HA_CAM_DEV_QR_INJECT
#define HA_CAM_DEV_QR_INJECT 1
#endif

// Legacy local WHIP dev endpoint; Step 6 replaces this with runtime config.
#define HA_CAM_WHIP_SERVER_IP    "10.0.4.184"
#define HA_CAM_WHIP_SERVER_PORT  5050

// Video settings. MQTT is disabled by design, so WHIP owns the freed DMA budget.
#define HA_CAM_VIDEO_WIDTH   640
#define HA_CAM_VIDEO_HEIGHT  480
#define HA_CAM_VIDEO_FPS     15
#define HA_CAM_VIDEO_BITRATE 800000u
#define VIDEO_WIDTH   640
#define VIDEO_HEIGHT  480
#define VIDEO_FPS     15
#define VIDEO_BITRATE 800000u

// Call button GPIO
#define CALL_BUTTON_GPIO  35

// Data channel (not used for now)
#define DATA_CHANNEL_ENABLED (false)

#ifdef __cplusplus
}
#endif
