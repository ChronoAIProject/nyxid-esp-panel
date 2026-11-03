/* Shared CSI camera: native preview/QR and bounded on-demand JPEG capture. */
#include "esp_video_init.h"
#include "common.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_video_ioctl.h"
#include "bsp/esp-bsp.h"
#include "driver/jpeg_encode.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "snapshot.h"
#include "linux/videodev2.h"
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#define TAG "MEDIA_SYS"
#define PREVIEW_WIDTH 320
#define PREVIEW_HEIGHT 240
#define PREVIEW_FRAME_BYTES ((size_t)PREVIEW_WIDTH * PREVIEW_HEIGHT * 2u)
#define PREVIEW_V4L2_BUF_COUNT 2u

static SemaphoreHandle_t preview_frame_lock;
static uint8_t *preview_frame_buf;
static uint8_t *preview_jpeg_buf;
static size_t preview_jpeg_cap;
static jpeg_encoder_handle_t preview_jpeg_encoder;
static size_t preview_frame_len;
static uint32_t preview_frame_seq;
static uint16_t preview_frame_w;
static uint16_t preview_frame_h;
static atomic_bool preview_pump_running;
static atomic_bool preview_pump_alive;
static atomic_bool preview_capture_started;
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
        int ret = esp_video_init_with_flags(&cam_config, ESP_VIDEO_INIT_FLAGS_MIPI_CSI | ESP_VIDEO_INIT_FLAGS_ISP);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Camera init FAILED: 0x%x", ret);
            return ESP_FAIL;
        }
        camera_initialized = true;
        ESP_LOGI(TAG, "Camera init OK");
    }
    return ESP_OK;
}

esp_err_t media_sys_capture_close(void)
{
    preview_capture_started = false;
    preview_pump_running = false;
    // DQBUF has a 100 ms timeout. Never free memory while the worker owns it.
    for (unsigned i = 0; atomic_load(&preview_pump_alive) && i < 150; ++i)
        vTaskDelay(pdMS_TO_TICKS(10));
    if (atomic_load(&preview_pump_alive)) return ESP_ERR_TIMEOUT;
    if (preview_frame_lock && xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(1000)) != pdTRUE)
        return ESP_ERR_TIMEOUT;
    heap_caps_free(preview_frame_buf);
    preview_frame_buf = NULL;
    preview_frame_len = 0;
    preview_frame_seq = 0;
    preview_frame_w = preview_frame_h = 0;
    if (preview_frame_lock) xSemaphoreGive(preview_frame_lock);
    if (preview_jpeg_encoder) {
        esp_err_t err = jpeg_del_encoder_engine(preview_jpeg_encoder);
        if (err != ESP_OK) return err;
        preview_jpeg_encoder = NULL;
    }
    free(preview_jpeg_buf);
    preview_jpeg_buf = NULL;
    preview_jpeg_cap = 0;
    if (camera_initialized) {
        esp_err_t err = esp_video_deinit_with_flags(ESP_VIDEO_INIT_FLAGS_MIPI_CSI | ESP_VIDEO_INIT_FLAGS_ISP);
        if (err != ESP_OK) return err;
        camera_initialized = false;
    }
    ESP_LOGI(TAG, "Camera idle: CSI, frame buffers and JPEG encoder released");
    return ESP_OK;
}

esp_err_t media_sys_prepare_preview_capture(void)
{
    esp_err_t err = media_sys_capture_close();
    if (err != ESP_OK) return err;
    err = ensure_camera_initialized();
    if (err != ESP_OK) return err;
    if (!preview_frame_lock) preview_frame_lock = xSemaphoreCreateMutex();
    preview_frame_buf = heap_caps_aligned_alloc(64, PREVIEW_FRAME_BYTES,
                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!preview_frame_lock || !preview_frame_buf) return ESP_ERR_NO_MEM;
    // JPEG resources are allocated only by a snapshot request, never by preview.
    return ESP_OK;
}

int media_sys_preview_start(void)
{
    if (preview_capture_started) return ESP_OK;
    if (!preview_frame_buf || !preview_frame_lock) return ESP_ERR_INVALID_STATE;
    preview_capture_started = true;
    preview_pump_running = true;
    atomic_store(&preview_pump_alive, true);
    if (xTaskCreate(preview_pump_task_fn, "preview_pump", 4096, NULL, 8, NULL) != pdPASS) {
        preview_capture_started = false;
        preview_pump_running = false;
        atomic_store(&preview_pump_alive, false);
        return ESP_ERR_NO_MEM;
    }
    for (unsigned i = 0; i < 80; ++i) {
        bool have_frame = false;
        if (xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(20)) == pdTRUE) {
            have_frame = preview_frame_len > 0;
            xSemaphoreGive(preview_frame_lock);
        }
        if (have_frame) return ESP_OK;
        if (!atomic_load(&preview_pump_alive)) break;
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    return ESP_ERR_NOT_FOUND;
}

bool media_sys_preview_frame_available(void)
{
    return preview_capture_started && atomic_load(&preview_pump_alive);
}

esp_err_t media_sys_preview_snapshot_encode(uint8_t *dst, size_t cap, size_t *out_len)
{
    *out_len = 0;
    if (!preview_capture_started || !preview_frame_lock || preview_jpeg_encoder)
        return ESP_ERR_INVALID_STATE;
    // Require a frame produced after this request; a stopped camera cannot
    // accidentally return the JPEG from a previous request.
    uint32_t initial_seq = 0;
    if (xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(200)) != pdTRUE) return ESP_ERR_TIMEOUT;
    initial_seq = preview_frame_seq;
    xSemaphoreGive(preview_frame_lock);
    jpeg_encode_memory_alloc_cfg_t mem_cfg = { .buffer_direction = JPEG_ENC_ALLOC_OUTPUT_BUFFER };
    preview_jpeg_buf = jpeg_alloc_encoder_mem(SNAPSHOT_MAX_BYTES, &mem_cfg, &preview_jpeg_cap);
    jpeg_encode_engine_cfg_t eng = { .timeout_ms = 1000 };
    esp_err_t err = preview_jpeg_buf ? jpeg_new_encoder_engine(&eng, &preview_jpeg_encoder) : ESP_ERR_NO_MEM;
    int64_t deadline = esp_timer_get_time() + 1500000;
    while (err == ESP_OK) {
        if (xSemaphoreTake(preview_frame_lock, pdMS_TO_TICKS(100)) != pdTRUE) {
            err = ESP_ERR_TIMEOUT;
            break;
        }
        if (preview_frame_len == PREVIEW_FRAME_BYTES && preview_frame_seq != initial_seq && preview_frame_seq >= 3) {
            jpeg_encode_cfg_t cfg = {
                .height = PREVIEW_HEIGHT, .width = PREVIEW_WIDTH,
                .src_type = JPEG_ENCODE_IN_FORMAT_RGB565,
                .sub_sample = JPEG_DOWN_SAMPLING_YUV420, .image_quality = 50,
            };
            uint32_t produced = 0;
            err = jpeg_encoder_process(preview_jpeg_encoder, &cfg, preview_frame_buf,
                                        PREVIEW_FRAME_BYTES, preview_jpeg_buf, preview_jpeg_cap, &produced);
            if (err == ESP_OK) {
                if (!produced || produced > cap) err = ESP_ERR_INVALID_SIZE;
                else { memcpy(dst, preview_jpeg_buf, produced); *out_len = produced; }
            }
            xSemaphoreGive(preview_frame_lock);
            break;
        }
        xSemaphoreGive(preview_frame_lock);
        if (esp_timer_get_time() >= deadline || !atomic_load(&preview_pump_alive)) {
            err = ESP_ERR_TIMEOUT;
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    if (preview_jpeg_encoder) {
        esp_err_t close_err = jpeg_del_encoder_engine(preview_jpeg_encoder);
        if (close_err == ESP_OK) preview_jpeg_encoder = NULL;
        else err = close_err;
    }
    free(preview_jpeg_buf);
    preview_jpeg_buf = NULL;
    preview_jpeg_cap = 0;
    if (err != ESP_OK) *out_len = 0;
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
        // esp_video 2.1 G_FMT copies a default descriptor with type zero.
        // Restore the requested stream before passing it back to S_FMT.
        format.type = type;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
        if (ioctl(fd, VIDIOC_S_FMT, &format) != 0) {
            ESP_LOGW(TAG, "preview direct set format failed width=%lu height=%lu errno=%d",
                     (unsigned long)format.fmt.pix.width,
                     (unsigned long)format.fmt.pix.height, errno);
            break;
        }
        actual_w = format.fmt.pix.width;
        actual_h = format.fmt.pix.height;

        // Keep the sensor's native frame interval. OV5647's 800x800 mode
        // has a fixed rate and rejects arbitrary VIDIOC_S_PARM requests.
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
        uint32_t queued = 0;
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
            if (!bufs[i].data || bufs[i].data == MAP_FAILED || ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
                ESP_LOGW(TAG, "preview direct qbuf failed");
                break;
            }
            ++queued;
        }
        if (queued != buf_count) break;

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
            if (bufs[i].data && bufs[i].data != MAP_FAILED && bufs[i].len) {
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
    atomic_store(&preview_pump_alive, false);
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
    if (!preview_frame_buf || !preview_capture_started) {
        err = ESP_ERR_INVALID_STATE;
    } else if (preview_frame_seq == last_seq) {
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
