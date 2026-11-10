#pragma once
// Explicit LAN-only research instrumentation. No microphone/audio owner.
#include <cstdint>
#include <cstring>
#include "lwip/sockets.h"
#include "lwip/inet.h"

class InputTrace {
    struct __attribute__((packed)) Header {
        uint32_t magic, instance, step, part, flags, pcm_crc;
        uint64_t started_us;
        int32_t dense, temporal;
        uint32_t numerator;
    };
    static_assert(sizeof(Header) == 44);
    int socket_ = -1;
    sockaddr_in destination_{};
    uint32_t instance_ = 0, sent_ = 0, dropped_ = 0;

    void send(Header header, unsigned part, const void *a, size_t a_bytes,
              const void *b = nullptr, size_t b_bytes = 0) {
        uint8_t packet[1452];
        if (sizeof(header) + a_bytes + b_bytes > sizeof(packet)) abort();
        header.part = part;
        memcpy(packet, &header, sizeof(header));
        memcpy(packet + sizeof(header), a, a_bytes);
        if (b_bytes) memcpy(packet + sizeof(header) + a_bytes, b, b_bytes);
        size_t bytes = sizeof(header) + a_bytes + b_bytes;
        int result = sendto(socket_, packet, bytes, MSG_DONTWAIT,
                            reinterpret_cast<sockaddr *>(&destination_), sizeof(destination_));
        if (result == int(bytes)) ++sent_; else ++dropped_;
    }

public:
    ~InputTrace() { if (socket_ >= 0) close(socket_); }
    bool initialize() {
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        if (socket_ < 0) return false;
        int nonblocking = 1;
        if (ioctl(socket_, FIONBIO, &nonblocking) < 0) return false;
        destination_.sin_family = AF_INET;
        destination_.sin_port = htons(NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT);
        destination_.sin_addr.s_addr = inet_addr(NYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP);
        instance_ = uint32_t(esp_timer_get_time());
        return destination_.sin_addr.s_addr != INADDR_NONE;
    }
    void write(unsigned step, int64_t started, const int16_t *pcm,
               const int8_t *features, const int8_t *embedding,
               const int *heads, unsigned numerator, bool event) {
        // Endian is explicitly little in the host receiver. Each UDP payload
        // fits one standard1500byte IPv4 packet, without fragmentation.
        Header header{0x3154584e, instance_, step, 0,
                      (step <= 30 ? 1u : 0u) | (event ? 2u : 0u),
                      uint32_t(crc32(0, reinterpret_cast<const Bytef *>(pcm), 2560)),
                      uint64_t(started), heads[0], heads[1], numerator};
        send(header, 0, pcm, 1280, features, 128);
        send(header, 1, pcm + 640, 1280, features + 128, 128);
        send(header, 2, embedding, 96);
        if (step % 100 == 0)
            ESP_LOGI("panel.wake.trace", "instance=%u step=%u packets=%u dropped=%u stack_free=%u",
                     unsigned(instance_), step, unsigned(sent_), unsigned(dropped_),
                     unsigned(uxTaskGetStackHighWaterMark(nullptr)));
    }
};
