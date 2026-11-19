# P4 上的 SpeexDSP 回声抵消和固定唤醒音

独立 ESP-IDF 5.5.4 应用使用 SpeexDSP 1.2.1，同步输入麦克风与扬声器参考，每次处理 16 kHz 单声道的 160 样本，滤波器长度 2048 样本。每个 10 秒输入独立启动 ESP32-P4 QEMU，输出完整 PCM、每帧处理用时和内部堆读数。输入来自两组 Bot 与外部合成声音，以及两份实测房间响应。[24 个普通语音用例和原始配置](../experiments/c161_lexical_echo.md)

| 24 个模拟输入，各 1,000 帧 | 定点、KISS FFT | 浮点、SmallFT |
| --- | ---: | ---: |
| 初始化滤波器的内部堆增量 | 52,356 B | 73,928 B |
| 每个 10 ms 块的计算时间中位数 | 0.295 ms | 0.703 ms |
| 最慢计算块 | 3.219 ms | 4.153 ms |
| 单独设备示范，误唤 /4 | 1 | 0 |
| 相同麦克风波形作为外部播放，唤醒 /4 | 2 | 2 |
| 外部独讲，完整命令 /4 | 4 | 4 |
| Bot 普通回复期间外部插话，完整命令 /4 | 0 | 0 |
| 双方同时念唤醒词、后接不同命令，外部完整命令 /4 | 2 | 4 |

普通语音确认使用主机的冻结 C161、双速提议和 Q8 确认器，处理实际 P4 输出 PCM。完整命令计入唤醒及 4090 Whisper 两秒回看转写。定点版的 3,840,000 个输出样本逐个等于同源码主机构建。浮点版有 77,740 个样本不同，最大绝对差为 46 个 PCM16 单位。两种实现均将全部输出和 UART 存档。[定点测量](../results/4090-c161-speex-p4.json) · [定点 UART](../results/4090-c161-speex-p4-uart.zip) · [浮点测量](../results/4090-c161-speex-p4-float.json) · [浮点 UART](../results/4090-c161-speex-p4-float-uart.zip)

## 已交付 v5 固定音

两个房间各有四个 10 秒用例。面板命令为 “Play the evening news.”，外部命令为 “Turn off the desk lamp.”，双方使用相同的 v5 唤醒文件。对 8 个双通道输入，P4 同时运行浮点 SpeexDSP 和原 v5 匹配器。滤波器输出与前一次纯滤波器回放逐样本相同，设备上三个通知的样本位置也均与从同一 PCM 得到的主机结果吻合。

| 场景 | P4 通知 | 两秒回看后正确转写外部命令 |
| --- | ---: | ---: |
| 仅面板播放 v5 | 0/2 | 无外部命令 |
| 仅面板普通回复 | 0/2 | 无外部命令 |
| 面板普通回复期间外部播放 v5 | 2/2 | 2/2 |
| 双方同时播放相同 v5，后接不同命令 | 1/2 | 1/2 |

同步播放时客厅在样本 97,584 通知，厨房漏检。两组 Bot 普通回复期间的外部播放均在样本 97,592 通知。单独滤波器的首次固定音重放有一帧耗时 12.346 ms（8,000 帧），与另一个模拟回放同时运行。滤波器和匹配器联合重放时 8,000 帧中位 0.758 ms、最大 4.467 ms。计时范围从 DSP 调用到匹配器返回。实际连续采集、扬声器音量、整机调度和最低内部堆由 Mac 板测核对。[首句、最慢帧和原始输出](../results/4090-c161-speex-fixed-playback.json) · [联合运行 UART](../results/4090-c161-speex-fixed-playback-device-uart.zip)

Mac 回放包内 8 个 WAV 均为 16 kHz PCM16，左声道为模拟麦克风，右声道为同步扬声器参考。注入自测应按 160 样本一块成对送入。该双声道文件用于数字输入验证。实机声学播放可使用此前交付的原始单声道 v5 文件。每个 WAV 的 SHA256 和预期通知采样点均在设备报告中。归档 SHA256 为 `ea3ad30ab891e89e0ac767f9f9e1da3e8fb9318204c37abd9ccaee21a615e9e3`。[双通道回放包](../results/4090-c161-speex-fixed-playback-device-stereo.zip) · [设备报告](../results/4090-c161-speex-fixed-playback-device.json)

## 复现

从 Debian 下载上游 speexdsp_1.2.1.orig.tar.gz，SHA256 为 `8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d`。下列命令在仓库根目录运行，使用已经设置的 ESP-IDF 5.5.4 环境。使用相同配置的 PC 定点或浮点动态库由回放脚本直接从校验过的上游源码构建。[发行源码](https://deb.debian.org/debian/pool/main/s/speexdsp/speexdsp_1.2.1.orig.tar.gz)

```sh
work="$PWD/.local/wakeword/c161-first-command-v5/lexical-echo"
app="$PWD/research/wakeword_p4/speex_echo_p4"
mkdir -p "$work"
curl -fsSL https://deb.debian.org/debian/pool/main/s/speexdsp/speexdsp_1.2.1.orig.tar.gz -o "$work/speexdsp-1.2.1.tar.gz"
printf '%s  %s\n' 8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d "$work/speexdsp-1.2.1.tar.gz" | sha256sum -c -
tar -xzf "$work/speexdsp-1.2.1.tar.gz" -C "$work"
idf.py -C "$app" -B "$work/build-cue" -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/sdkconfig" \
  -DSDKCONFIG_DEFAULTS="$app/sdkconfig.defaults" -DNYX_SPEEX_FLOAT=ON -DNYX_QEMU_FPU=ON -DNYX_SPEEX_FIXED_CUE=ON reconfigure
cmake --build "$work/build-cue" --parallel 2
```

定点回放使用默认选项，目录 build。纯浮点回放启用 NYX_SPEEX_FLOAT 和 NYX_QEMU_FPU，目录 build-float。浮点模拟使用先前自测中的 IDF 浮点异常指令识别配置。可在现有研究环境分别运行 replay_c161_speex_p4.py、其 --float 选项、check_c161_speex_fixed_playback.py 和 verify_c161_speex_cue.py。
