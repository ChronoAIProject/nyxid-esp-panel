# Display compatibility

The reference is the phone-style `firmware/brookesia` in Waveshare's
ESP32-P4-Platform, commit `6f78fa810ec505048c1941e3d09cefc9f29129bd`.
Its native Camera uses 800×1280 RGB565, two camera buffers, 400 kHz SCCB and
OV5647's native 50 FPS mode. v0.3.1 restores that capture profile, including
the modern sensor component's explicit default-format choice. The preview
keeps the single shared owner used by remote photos and Wi-Fi QR scanning.

The panel uses the same BSP and JD9365 driver as that Brookesia checkout:
800×1280 RGB565, 80 MHz DPI, two DSI lanes at 1500 Mbps, and the vendor's
horizontal porches/pulse 20/20/40 and vertical 10/4/30. These panel initialization
commands and electrical/timing settings are unchanged by the compatibility fix.

The reference app selects triple partial refresh. Both projects resolve
esp_lvgl_adapter 0.6.4; on IDF 5.5 this adapter falls back from the unavailable
`on_frame_buf_complete` callback to `on_refresh_done`. The latter is a periodic
DSI bridge VSYNC interrupt. The IDF driver actually selects the next framebuffer
in a separate DMA-completion ISR. VSYNC does not establish retirement of the
previous DMA descriptor, so the fallback can release buffers prematurely.
This is a synchronization defect; the user's photo alone does not prove that
it accounts for every observed pixel artifact.

v0.3.1 uses full-frame double buffering and a project-local compatibility
extension reporting the framebuffer actually selected in the DMA ISR, after
retiring the previous transfer. A locked submission gate matches that pointer
before waking LVGL. Idle/repeated interrupts and an ISR that sampled the old
index before submission cannot release the wait. Completion before the task
arms its gate is recovered by the next actual DMA completion of the same
framebuffer. This uses actual transfer events, with no fixed VSYNC delay.

`cmake/panel_lcd_compat.cmake` copies only the pinned SDK's DPI source into the
build directory, applies `patches/esp_idf_5_5_dpi_scanout.patch`, and replaces
that source in the esp_lcd target. The SDK checkout and public headers stay
untouched. The managed adapter patch registers the extension and removes the
VSYNC fallback. Patch/source mismatches fail configuration. This backport is
restricted to IDF 5.5 and unrotated full-frame double buffering; changing SDK or
refresh mode requires reviewing the compatibility code. It does not support
the reference Camera's independent dummy-draw framebuffer writer.

Host tests check stale DMA selection, repeated callbacks, alternating buffers
and completion before gate arming. Hardware acceptance exercises actual display
flushes and CSI activity. A `/lcd/snapshot.jpg` read captures LVGL's logical
image and cannot establish that the physical scanout is artifact-free. Physical
follow-up and a long soak remain necessary to assess intermittent corruption.

## JPEG snapshot DMA buffers

This is separate from LCD scanout synchronization. A repeated static-screen
JPEG on the company board decoded with gray corruption despite an intact JPEG
start/end and unchanged length. The differing bytes included zeroed runs at
64-byte cache-line boundaries. IDF 5.5's JPEG output allocator uses aligned
`calloc`; its dirty zero cache lines must be retired before DMA writes into
that memory. The encoder driver synchronizes the input before DMA and the
output after completion, but does not perform that initial output handoff.

Both LCD and camera encoding now clean/invalidate their aligned output allocation
before calling the hardware encoder. This keeps later cache eviction from
overwriting compressed bytes. Repeated on-board snapshots are needed to verify
this fix; it does not by itself establish physical LCD correctness.
