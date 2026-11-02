/* Media system for VoicePresence ESP32-P4 client
 *
 * Uses single I2S controller for full-duplex. Speaker codec device is
 * pre-opened at 16kHz before mic starts, so that av_render's later
 * esp_codec_dev_open is a no-op (device already open at correct rate)
 * and doesn't reconfigure I2S (which would kill mic DMA).
 */

#include "esp_video_init.h"
#include "av_render.h"
#include "av_render_default.h"
#include "common.h"
#include "esp_log.h"
#include "settings.h"
#include "media_lib_os.h"
#include "esp_audio_enc_default.h"
#include "esp_video_enc_default.h"
#include "esp_audio_dec_default.h"
#include "esp_video_dec_default.h"
#include "esp_capture_defaults.h"
#include "esp_capture_sink.h"
#include "esp_codec_dev.h"
#include "esp_timer.h"
#include "esp_video_ioctl.h"
#include "bsp/esp-bsp.h"
#include "driver/jpeg_encode.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "heap_diag.h"
#include "snapshot.h"

#include "linux/videodev2.h"
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#define TAG  "MEDIA_SYS"
#define PREVIEW_FPS 30u
#define PREVIEW_MIN_FPS 15u
#define PREVIEW_WIDTH 320
#define PREVIEW_HEIGHT 240
#define PREVIEW_FRAME_BYTES ((size_t)PREVIEW_WIDTH * (size_t)PREVIEW_HEIGHT * 2u)
#define PREVIEW_V4L2_BUF_COUNT 2u

typedef struct {
    esp_capture_sink_handle_t   capture_handle;
    esp_capture_video_src_if_t *vid_src;
    esp_capture_audio_src_if_t *aud_src;
} capture_system_t;

typedef struct {
    audio_render_handle_t  audio_render;
    av_render_handle_t     player;
} player_system_t;

static capture_system_t capture_sys;
static player_system_t  player_sys;
static esp_codec_dev_handle_t spk_handle_global = NULL;
static esp_codec_dev_handle_t mic_handle_global = NULL;
static esp_capture_sink_handle_t webrtc_sink;
static esp_capture_sink_handle_t snapshot_sink;
static SemaphoreHandle_t snapshot_poll_lock;
static SemaphoreHandle_t preview_frame_lock;
static TaskHandle_t preview_pump_task;
static uint8_t *preview_frame_buf;
static uint8_t *preview_jpeg_buf;
static size_t preview_jpeg_cap;
static jpeg_encoder_handle_t preview_jpeg_encoder;
static size_t preview_frame_len;
static uint32_t preview_frame_seq;
static uint16_t preview_frame_w;
static uint16_t preview_frame_h;
static bool preview_pump_running;
static bool webrtc_capture_started;
static bool preview_capture_started;
static bool camera_initialized;

static void preview_pump_task_fn(void *arg);

static esp_err_t ensure_camera_initialized(void)
{
    if (!camera_initialized) {
        i2c_master_bus_handle_t i2c_handle = bsp_i2c_get_handle();
        ESP_LOGI(TAG, "Camera init: I2C handle=%p", i2c_handle);

        esp_video_init_csi_config_t csi_config = {
            .sccb_config = {
                .init_sccb = false,
                .i2c_handle = i2c_handle,
                .freq = 100000,
            },
            .reset_pin = -1,
            .pwdn_pin = -1,
        };
        esp_video_init_config_t cam_config = {
            .csi = &csi_config,
        };
        int ret = esp_video_init(&cam_config);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Camera init FAILED: 0x%x", ret);
            return ESP_FAIL;
        }
        camera_initialized = true;
        ESP_LOGI(TAG, "Camera init OK");
    }
    return ESP_OK;
}

static esp_capture_video_src_if_t *create_video_source(uint8_t buf_count)
{
    if (ensure_camera_initialized() != ESP_OK) {
        return NULL;
    }

    esp_capture_video_v4l2_src_cfg_t v4l2_cfg = {
        .dev_name = "/dev/video0",
        .buf_count = buf_count,
    };
    esp_capture_video_src_if_t *src = esp_capture_new_video_v4l2_src(&v4l2_cfg);
    ESP_LOGI(TAG, "Video V4L2 source: %p", src);
    return src;
}

static int build_player_system(void)
{
    ESP_LOGI(TAG, "=== Building player system ===");

    spk_handle_global = bsp_audio_codec_speaker_init();
    if (spk_handle_global == NULL) {
        ESP_LOGE(TAG, "Speaker codec init FAILED");
        return -1;
    }
    ESP_LOGI(TAG, "Speaker codec init OK: handle=%p", spk_handle_global);

    // Pre-open speaker codec at 16kHz (Opus sample rate) BEFORE anything else.
    esp_codec_dev_sample_info_t spk_fs = {
        .sample_rate = 16000,
        .channel = 1,
        .bits_per_sample = 16,
    };
    esp_err_t err = esp_codec_dev_open(spk_handle_global, &spk_fs);
    ESP_LOGI(TAG, "Speaker pre-open(16kHz/1ch/16bit): %s", esp_err_to_name(err));

    err = esp_codec_dev_set_out_vol(spk_handle_global, 100);
    ESP_LOGI(TAG, "Speaker volume set to 80: %s", esp_err_to_name(err));

    i2s_render_cfg_t i2s_cfg = {
        .play_handle = spk_handle_global,
    };
    player_sys.audio_render = av_render_alloc_i2s_render(&i2s_cfg);
    if (player_sys.audio_render == NULL) {
        ESP_LOGE(TAG, "I2S audio render alloc FAILED");
        return -1;
    }
    ESP_LOGI(TAG, "I2S audio render: %p", player_sys.audio_render);

    av_render_cfg_t render_cfg = {
        .audio_render = player_sys.audio_render,
        .audio_raw_fifo_size = 8 * 4096,
        .audio_render_fifo_size = 100 * 1024,
        .allow_drop_data = false,
    };
    player_sys.player = av_render_open(&render_cfg);
    if (player_sys.player == NULL) {
        ESP_LOGE(TAG, "av_render_open FAILED");
        return -1;
    }
    ESP_LOGI(TAG, "av_render player: %p (raw_fifo=%d, render_fifo=%d)",
             player_sys.player, 8 * 4096, 100 * 1024);

    ESP_LOGI(TAG, "=== Player system ready ===");
    return 0;
}

static esp_err_t build_capture_system(bool include_audio)
{
    ESP_LOGI(TAG, "=== Building capture system ===");
    media_sys_capture_close();

    capture_sys.vid_src = create_video_source(2);
    if (!capture_sys.vid_src) {
        return ESP_FAIL;
    }

    if (include_audio) {
        if (!mic_handle_global) {
            mic_handle_global = bsp_audio_codec_microphone_init();
        }
        if (mic_handle_global == NULL) {
            ESP_LOGW(TAG, "Microphone codec init FAILED");
        } else {
            ESP_LOGI(TAG, "Microphone codec init OK: handle=%p", mic_handle_global);
        }

        esp_capture_audio_dev_src_cfg_t codec_cfg = {
            .record_handle = mic_handle_global,
        };
        capture_sys.aud_src = esp_capture_new_audio_dev_src(&codec_cfg);
        ESP_LOGI(TAG, "Audio dev source: %p", capture_sys.aud_src);
    }

    esp_capture_cfg_t cfg = {
        .sync_mode = include_audio ? ESP_CAPTURE_SYNC_MODE_AUDIO : ESP_CAPTURE_SYNC_MODE_NONE,
        .audio_src = capture_sys.aud_src,
        .video_src = capture_sys.vid_src,
    };
    int ret = esp_capture_open(&cfg, &capture_sys.capture_handle);
    ESP_LOGI(TAG, "Capture system open: ret=%d handle=%p", ret, capture_sys.capture_handle);

    ESP_LOGI(TAG, "=== Capture system ready ===");
    return ret == ESP_CAPTURE_ERR_OK ? ESP_OK : ESP_FAIL;
}

void media_sys_capture_close(void)
{
    webrtc_capture_started = false;
    preview_capture_started = false;
    preview_pump_running = false;
    webrtc_sink = NULL;
    snapshot_sink = NULL;

    if (capture_sys.capture_handle) {
        heap_diag_log(TAG, "capture_close_before");
        esp_capture_close(capture_sys.capture_handle);
        capture_sys.capture_handle = NULL;
        heap_diag_log(TAG, "capture_close_after");
    }

    for (int i = 0; preview_pump_task && i < 50; i++) {
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    if (preview_pump_task) {
        ESP_LOGW(TAG, "preview pump did not stop before capture close cleanup");
        return;
    }

    if (capture_sys.vid_src) {
        free(capture_sys.vid_src);
        capture_sys.vid_src = NULL;
    }
    if (capture_sys.aud_src) {
        free(capture_sys.aud_src);
        capture_sys.aud_src = NULL;
    }
    if (preview_frame_buf) {
        heap_caps_free(preview_frame_buf);
        preview_frame_buf = NULL;
    }
    if (preview_jpeg_encoder) {
        (void)jpeg_del_encoder_engine(preview_jpeg_encoder);
        preview_jpeg_encoder = NULL;
    }
    if (preview_jpeg_buf) {
        free(preview_jpeg_buf);
        preview_jpeg_buf = NULL;
        preview_jpeg_cap = 0u;
    }
    preview_frame_len = 0u;
    preview_frame_seq = 0u;
    preview_frame_w = 0u;
    preview_frame_h = 0u;
}

esp_err_t media_sys_prepare_preview_capture(void)
{
    media_sys_capture_close();
    esp_err_t err = ensure_camera_initialized();
    if (err != ESP_OK) {
        return err;
    }
    snapshot_init();
    if (!snapshot_poll_lock) {
        snapshot_poll_lock = xSemaphoreCreateMutex();
    }
    if (!preview_frame_lock) {
        preview_frame_lock = xSemaphoreCreateMutex();
    }
    if (!preview_frame_buf) {
        preview_frame_buf = heap_caps_aligned_alloc(64, PREVIEW_FRAME_BYTES,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!preview_frame_buf) {
            preview_frame_buf = heap_caps_aligned_alloc(64, PREVIEW_FRAME_BYTES, MALLOC_CAP_8BIT);
        }
    }
    if (!preview_frame_lock || !preview_frame_buf) {
        ESP_LOGW(TAG, "preview frame buffer setup failed");
        media_sys_capture_close();
        return ESP_ERR_NO_MEM;
    }
    if (!preview_jpeg_buf) {
        jpeg_encode_memory_alloc_cfg_t mem_cfg = {
            .buffer_direction = JPEG_ENC_ALLOC_OUTPUT_BUFFER,
        };
        preview_jpeg_buf = jpeg_alloc_encoder_mem(SNAPSHOT_MAX_BYTES, &mem_cfg, &preview_jpeg_cap);
    }
    if (!preview_jpeg_encoder) {
        jpeg_encode_engine_cfg_t eng = {
            .timeout_ms = 1000,
        };
        err = jpeg_new_encoder_engine(&eng, &preview_jpeg_encoder);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "preview jpeg encoder setup failed: %s", esp_err_to_name(err));
        }
    }
    ESP_LOGI(TAG, "preview capture prepared index=0 video=RGB565/%dx%d/%u",
             PREVIEW_WIDTH, PREVIEW_HEIGHT, (unsigned)PREVIEW_FPS);
    return ESP_OK;
}

esp_err_t media_sys_prepare_publish_capture(uint8_t fps)
{
    if (fps == 0u) {
        fps = VIDEO_FPS;
    }
    esp_err_t err = build_capture_system(false);
    if (err != ESP_OK) {
        return err;
    }
    esp_capture_sink_cfg_t sink_cfg = {
        .audio_info = { .format_id = ESP_CAPTURE_FMT_ID_NONE },
        .video_info = {
            .format_id = ESP_CAPTURE_FMT_ID_H264,
            .width = VIDEO_WIDTH,
            .height = VIDEO_HEIGHT,
            .fps = fps,
        },
    };
    int ret = esp_capture_sink_setup(capture_sys.capture_handle, 0, &sink_cfg, &webrtc_sink);
    if (ret != ESP_CAPTURE_ERR_OK) {
        ESP_LOGW(TAG, "webrtc sink setup failed: %d", ret);
        media_sys_capture_close();
        return ESP_FAIL;
    } else {
        ESP_LOGI(TAG, "webrtc sink prepared index=0 audio=none video=H264/%dx%d/%u",
                 VIDEO_WIDTH, VIDEO_HEIGHT, (unsigned)fps);
    }
    return ESP_OK;
}

int media_sys_webrtc_capture_start(void)
{
    if (!capture_sys.capture_handle || !webrtc_sink) {
        ESP_LOGW(TAG, "webrtc capture start skipped handle=%p sink=%p",
                 capture_sys.capture_handle, webrtc_sink);
        return -1;
    }
    if (webrtc_capture_started) {
        return 0;
    }

    heap_diag_log(TAG, "webrtc_capture_start_before");
    int ret = esp_capture_sink_enable(webrtc_sink, ESP_CAPTURE_RUN_MODE_ALWAYS);
    if (ret != ESP_CAPTURE_ERR_OK) {
        ESP_LOGW(TAG, "webrtc sink enable failed: %d", ret);
        heap_diag_log(TAG, "webrtc_capture_start_enable_failed");
        return ret;
    }
    webrtc_capture_started = true;

    ret = esp_capture_start(capture_sys.capture_handle);
    if (ret != ESP_CAPTURE_ERR_OK) {
        ESP_LOGW(TAG, "webrtc capture start failed: %d", ret);
        esp_capture_sink_enable(webrtc_sink, ESP_CAPTURE_RUN_MODE_DISABLE);
        webrtc_capture_started = false;
        heap_diag_log(TAG, "webrtc_capture_start_failed");
        return ret;
    }
    ESP_LOGI(TAG, "webrtc capture started sink=%p", webrtc_sink);
    heap_diag_log(TAG, "webrtc_capture_start_after");
    return 0;
}

int media_sys_webrtc_capture_stop(void)
{
    int ret = ESP_CAPTURE_ERR_OK;

    if (webrtc_sink && webrtc_capture_started) {
        heap_diag_log(TAG, "webrtc_capture_stop_before");
        ret = esp_capture_sink_enable(webrtc_sink, ESP_CAPTURE_RUN_MODE_DISABLE);
        if (ret != ESP_CAPTURE_ERR_OK) {
            ESP_LOGW(TAG, "webrtc sink disable failed: %d", ret);
        }
        webrtc_capture_started = false;
    }

    if (capture_sys.capture_handle) {
        int stop_ret = esp_capture_stop(capture_sys.capture_handle);
        if (stop_ret != ESP_CAPTURE_ERR_OK) {
            ESP_LOGW(TAG, "webrtc capture stop failed: %d", stop_ret);
            if (ret == ESP_CAPTURE_ERR_OK) {
                ret = stop_ret;
            }
        }
    }

    if (ret == ESP_CAPTURE_ERR_OK) {
        ESP_LOGI(TAG, "webrtc capture stopped");
    }
    heap_diag_log(TAG, "webrtc_capture_stop_after");
    return ret;
}

int media_sys_preview_start(void)
{
    if (preview_capture_started) {
        return 0;
    }
    heap_diag_log(TAG, "preview_capture_start_before");
    preview_capture_started = true;
    preview_pump_running = true;
    if (!preview_pump_task) {
        BaseType_t ok = xTaskCreate(preview_pump_task_fn, "preview_pump", 4096, NULL, 8,
                                    &preview_pump_task);
        if (ok != pdPASS) {
            preview_pump_task = NULL;
            preview_pump_running = false;
            ESP_LOGW(TAG, "preview pump task create failed");
            return ESP_ERR_NO_MEM;
        }
    }
    ESP_LOGI(TAG, "preview capture started direct_v4l2=1");
    heap_diag_log(TAG, "preview_capture_start_after");
    return 0;
}

int media_sys_preview_stop(void)
{
    int ret = ESP_CAPTURE_ERR_OK;
    preview_pump_running = false;
    if (snapshot_sink && preview_capture_started) {
        heap_diag_log(TAG, "preview_capture_stop_before");
        ret = esp_capture_sink_enable(snapshot_sink, ESP_CAPTURE_RUN_MODE_DISABLE);
        if (ret != ESP_CAPTURE_ERR_OK) {
            ESP_LOGW(TAG, "preview sink disable failed: %d", ret);
        }
        preview_capture_started = false;
    }
    preview_capture_started = false;
    if (capture_sys.capture_handle) {
        int stop_ret = esp_capture_stop(capture_sys.capture_handle);
        if (stop_ret != ESP_CAPTURE_ERR_OK) {
            ESP_LOGW(TAG, "preview capture stop failed: %d", stop_ret);
            if (ret == ESP_CAPTURE_ERR_OK) {
                ret = stop_ret;
            }
        }
    }
    heap_diag_log(TAG, "preview_capture_stop_after");
    return ret;
}

bool media_sys_preview_frame_available(void)
{
    return preview_capture_started;
}

esp_err_t media_sys_snapshot_grab(uint32_t timeout_ms, size_t *out_len)
{
    if (out_len) {
        *out_len = 0u;
    }
    if (!snapshot_sink || !preview_capture_started) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!snapshot_poll_lock || xSemaphoreTake(snapshot_poll_lock, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    bool log_heap = timeout_ms > 0u;
    if (log_heap) {
        heap_diag_log(TAG, "snapshot_grab_before");
    }
    esp_err_t err = ESP_ERR_TIMEOUT;
    int64_t deadline = esp_timer_get_time() + ((int64_t)timeout_ms * 1000LL);
    do {
        esp_capture_stream_frame_t frame = {
            .stream_type = ESP_CAPTURE_STREAM_TYPE_VIDEO,
        };
        int ret = esp_capture_sink_acquire_frame(snapshot_sink, &frame, true);
        if (ret == ESP_CAPTURE_ERR_OK) {
            err = snapshot_store_jpeg(frame.data, (size_t)frame.size);
            if (err == ESP_OK && out_len) {
                *out_len = (size_t)frame.size;
            }
            esp_capture_sink_release_frame(snapshot_sink, &frame);
            if (log_heap) {
                heap_diag_log(TAG, "snapshot_grab_after");
            }
            break;
        }
        if (timeout_ms == 0u) {
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    } while (esp_timer_get_time() < deadline);
    xSemaphoreGive(snapshot_poll_lock);
    if (err != ESP_OK && log_heap) {
        heap_diag_log(TAG, "snapshot_grab_failed");
    }
    return err;
}

esp_err_t media_sys_preview_snapshot_encode(uint32_t timeout_ms, size_t *out_len)
{
    if (out_len) {
        *out_len = 0u;
    }
    if (!preview_capture_started) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!snapshot_poll_lock || xSemaphoreTake(snapshot_poll_lock, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    if (!preview_frame_lock || !preview_frame_buf || !preview_jpeg_buf || !preview_jpeg_encoder) {
        xSemaphoreGive(snapshot_poll_lock);
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t err = ESP_ERR_NOT_FOUND;
    if (xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(timeout_ms)) == pdTRUE) {
        if (preview_frame_len == PREVIEW_FRAME_BYTES) {
            jpeg_encode_cfg_t enc_cfg = {
                .height = PREVIEW_HEIGHT,
                .width = PREVIEW_WIDTH,
                .src_type = JPEG_ENCODE_IN_FORMAT_RGB565,
                .sub_sample = JPEG_DOWN_SAMPLING_YUV420,
                .image_quality = 50,
            };
            uint32_t produced = 0u;
            err = jpeg_encoder_process(preview_jpeg_encoder, &enc_cfg,
                                       preview_frame_buf, PREVIEW_FRAME_BYTES,
                                       preview_jpeg_buf, (uint32_t)preview_jpeg_cap, &produced);
            if (err == ESP_OK && produced > 0u && produced <= SNAPSHOT_MAX_BYTES) {
                err = snapshot_store_jpeg(preview_jpeg_buf, produced);
                if (err == ESP_OK && out_len) {
                    *out_len = produced;
                }
            }
        }
        xSemaphoreGive(preview_frame_lock);
    } else {
        err = ESP_ERR_TIMEOUT;
    }
    xSemaphoreGive(snapshot_poll_lock);
    return err;
}

static void preview_pump_task_fn(void *arg)
{
    (void)arg;
    typedef struct {
        uint8_t *data;
        size_t len;
    } preview_v4l2_buf_t;

    int fd = -1;
    bool streaming = false;
    preview_v4l2_buf_t bufs[PREVIEW_V4L2_BUF_COUNT] = {0};
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    uint32_t actual_w = PREVIEW_WIDTH;
    uint32_t actual_h = PREVIEW_HEIGHT;

    do {
        fd = open("/dev/video0", O_RDONLY);
        if (fd < 0) {
            ESP_LOGW(TAG, "preview direct open failed");
            break;
        }

        struct timeval timeout = {
            .tv_sec = 0,
            .tv_usec = 100000,
        };
        (void)ioctl(fd, VIDIOC_S_DQBUF_TIMEOUT, &timeout);

        struct v4l2_format format = {
            .type = type,
        };
        if (ioctl(fd, VIDIOC_G_FMT, &format) != 0) {
            ESP_LOGW(TAG, "preview direct get format failed");
            break;
        }
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
        if (ioctl(fd, VIDIOC_S_FMT, &format) != 0) {
            ESP_LOGW(TAG, "preview direct set format failed width=%lu height=%lu",
                     (unsigned long)format.fmt.pix.width,
                     (unsigned long)format.fmt.pix.height);
            break;
        }
        actual_w = format.fmt.pix.width;
        actual_h = format.fmt.pix.height;

        struct v4l2_streamparm sparm = {
            .type = type,
        };
        uint32_t requested_fps = PREVIEW_FPS;
        sparm.type = type;
        sparm.parm.capture.capability = V4L2_CAP_TIMEPERFRAME;
        sparm.parm.capture.timeperframe.numerator = 1;
        sparm.parm.capture.timeperframe.denominator = requested_fps;
        if (ioctl(fd, VIDIOC_S_PARM, &sparm) != 0) {
            ESP_LOGW(TAG, "preview direct set fps=%lu failed",
                     (unsigned long)requested_fps);
        }
        ESP_LOGI(TAG, "preview direct format=%lux%lu fourcc=%c%c%c%c",
                 (unsigned long)actual_w, (unsigned long)actual_h,
                 (char)(format.fmt.pix.pixelformat & 0xff),
                 (char)((format.fmt.pix.pixelformat >> 8) & 0xff),
                 (char)((format.fmt.pix.pixelformat >> 16) & 0xff),
                 (char)((format.fmt.pix.pixelformat >> 24) & 0xff));

        struct v4l2_requestbuffers req = {
            .count = PREVIEW_V4L2_BUF_COUNT,
            .type = type,
            .memory = V4L2_MEMORY_MMAP,
        };
        if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0 || req.count == 0) {
            ESP_LOGW(TAG, "preview direct reqbufs failed");
            break;
        }

        uint32_t buf_count = req.count < PREVIEW_V4L2_BUF_COUNT ? req.count : PREVIEW_V4L2_BUF_COUNT;
        for (uint32_t i = 0; i < buf_count; ++i) {
            struct v4l2_buffer buf = {
                .type = type,
                .memory = V4L2_MEMORY_MMAP,
                .index = i,
            };
            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
                ESP_LOGW(TAG, "preview direct querybuf failed");
                break;
            }
            bufs[i].data = (uint8_t *)mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
            bufs[i].len = buf.length;
            if (!bufs[i].data || ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
                ESP_LOGW(TAG, "preview direct qbuf failed");
                break;
            }
        }

        if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
            ESP_LOGW(TAG, "preview direct streamon failed");
            break;
        }
        streaming = true;
    } while (0);

    uint32_t fps_window_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
    uint32_t fps_frames = 0u;
    while (preview_pump_running && preview_capture_started && fd >= 0 && streaming) {
        struct v4l2_buffer buf = {
            .type = type,
            .memory = V4L2_MEMORY_MMAP,
        };
        if (ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            continue;
        }
        if (buf.index < PREVIEW_V4L2_BUF_COUNT && bufs[buf.index].data &&
            (buf.flags & V4L2_BUF_FLAG_DONE) && buf.bytesused >= (actual_w * actual_h * 2u) &&
            preview_frame_lock && xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(20)) == pdTRUE) {
            const uint16_t *src = (const uint16_t *)bufs[buf.index].data;
            uint16_t *dst = (uint16_t *)preview_frame_buf;
            if (actual_w == PREVIEW_WIDTH && actual_h == PREVIEW_HEIGHT) {
                memcpy(dst, src, PREVIEW_FRAME_BYTES);
            } else {
                for (uint32_t y = 0; y < PREVIEW_HEIGHT; ++y) {
                    uint32_t sy = (y * actual_h) / PREVIEW_HEIGHT;
                    const uint16_t *src_row = src + ((size_t)sy * actual_w);
                    uint16_t *dst_row = dst + ((size_t)y * PREVIEW_WIDTH);
                    for (uint32_t x = 0; x < PREVIEW_WIDTH; ++x) {
                        uint32_t sx = (x * actual_w) / PREVIEW_WIDTH;
                        dst_row[x] = src_row[sx];
                    }
                }
            }
            preview_frame_len = PREVIEW_FRAME_BYTES;
            preview_frame_w = PREVIEW_WIDTH;
            preview_frame_h = PREVIEW_HEIGHT;
            preview_frame_seq++;
            xSemaphoreGive(preview_frame_lock);
            fps_frames++;
        }
        (void)ioctl(fd, VIDIOC_QBUF, &buf);
        uint32_t ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
        if ((uint32_t)(ms - fps_window_ms) >= 1000u) {
            ESP_LOGI(TAG, "preview_pump: captured %lu frames in %lu ms dma_largest=%lu",
                     (unsigned long)fps_frames,
                     (unsigned long)(ms - fps_window_ms),
                     (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL));
            fps_frames = 0u;
            fps_window_ms = ms;
        }
    }

    if (fd >= 0) {
        if (streaming) {
            (void)ioctl(fd, VIDIOC_STREAMOFF, &type);
        }
        for (uint32_t i = 0; i < PREVIEW_V4L2_BUF_COUNT; ++i) {
            if (bufs[i].data && bufs[i].len) {
                munmap(bufs[i].data, bufs[i].len);
            }
        }
        close(fd);
    }

    if (preview_frame_lock && xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(20)) == pdTRUE) {
        preview_frame_len = 0u;
        preview_frame_w = 0u;
        preview_frame_h = 0u;
        xSemaphoreGive(preview_frame_lock);
    }
    preview_pump_task = NULL;
    vTaskDelete(NULL);
}

esp_err_t media_sys_preview_copy_rgb565(uint8_t *dst, size_t cap, uint32_t last_seq,
                                        uint16_t *out_w, uint16_t *out_h, size_t *out_len,
                                        uint32_t *out_seq)
{
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (out_len) *out_len = 0u;
    if (out_seq) *out_seq = last_seq;
    if (!dst || cap == 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!preview_capture_started) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!preview_frame_lock || xSemaphoreTake(preview_frame_lock, 0) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    esp_err_t err = ESP_ERR_TIMEOUT;
    if (preview_frame_seq == last_seq) {
        err = ESP_ERR_NOT_FOUND;
    } else if (preview_frame_len == 0u || preview_frame_w == 0u || preview_frame_h == 0u) {
        err = ESP_ERR_INVALID_SIZE;
    } else if (preview_frame_len > cap) {
        err = ESP_ERR_NO_MEM;
    } else {
        memcpy(dst, preview_frame_buf, preview_frame_len);
        if (out_w) *out_w = preview_frame_w;
        if (out_h) *out_h = preview_frame_h;
        if (out_len) *out_len = preview_frame_len;
        if (out_seq) *out_seq = preview_frame_seq;
        err = ESP_OK;
    }
    xSemaphoreGive(preview_frame_lock);
    return err;
}

esp_err_t media_sys_preview_copy_latest_rgb565(uint8_t *dst, size_t cap, uint16_t *out_w,
                                               uint16_t *out_h, size_t *out_len,
                                               uint32_t *out_seq)
{
    return media_sys_preview_copy_rgb565(dst, cap, 0u, out_w, out_h, out_len, out_seq);
}

int media_sys_buildup(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Media system buildup starting");
    ESP_LOGI(TAG, "Free heap: %lu, free PSRAM: %lu",
             (unsigned long)esp_get_free_heap_size(),
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    esp_video_enc_register_default();
    esp_audio_enc_register_default();
    esp_video_dec_register_default();
    esp_audio_dec_register_default();
    ESP_LOGI(TAG, "Codecs registered");

    // Player FIRST — pre-opens speaker codec, configuring I2S TX at 8kHz.
    // Then capture — mic uses already-configured I2S RX.
    int ret = build_player_system();
    if (ret != 0) {
        ESP_LOGE(TAG, "Player system build FAILED: %d", ret);
    }

    ESP_LOGI(TAG, "Media system buildup complete");
    ESP_LOGI(TAG, "Free heap: %lu, free PSRAM: %lu",
             (unsigned long)esp_get_free_heap_size(),
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "========================================");
    return 0;
}

int play_music(const uint8_t *data, int size, int duration)
{
    return 0;
}

int stop_music(void)
{
    return 0;
}

int test_capture_to_player(void)
{
    return 0;
}

int media_sys_get_provider(esp_webrtc_media_provider_t *provide)
{
    provide->capture = capture_sys.capture_handle;
    provide->player = player_sys.player;
    ESP_LOGD(TAG, "Media provider: capture=%p player=%p", provide->capture, provide->player);
    return 0;
}
