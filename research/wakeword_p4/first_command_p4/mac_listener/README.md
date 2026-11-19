# Mac 研究监听器在 P4 上的固定音与首句回放

将 Mac 研究分支的单 C161 运行时代码、原模型和 profile 直接编入 ESP32-P4 独立回放程序。麦克风原始 PCM 同时送入该运行时和固定 v5 匹配器。另一路运行 ESP-SR 的 AEC 与 AGC，零参考对应面板扬声器静默，处理后的 PCM 进入面板两秒回看缓存。在第一条通知所属的 256 样本读块末，将回看按面板代码的环形顺序转入 60 秒通话 FIFO，再陆续写入余下的处理后音频。每次触发都在 P4 将 FIFO 与同期连续处理后的 PCM 逐样本核对，随后把整个 FIFO 输出到 UART。原 C161 预热 81,600 样本，实时音频边界连续；AEC 从新建状态起步。两个检测器在通知后继续观察，供逐路核对。[Mac 运行时与实际监听循环](https://github.com/ChronoAIProject/nyxid-esp-panel/tree/0d8e1935702a561b535a8c1a0921fd8034943d2c/components/panel_wake_runtime) · [模拟回放](main/listener.cpp)

分别启动 96 次，使用此前冻结的两组家庭房间混音。厨房为空调及 “Turn off the desk lamp”，客厅为音乐及 “Play the evening news”。每种条件有四种既有合成命令声音和三个命令信噪比。固定 v5、普通合成 Nyx、字面 Hey Willow、静音四个唤醒段在每个条件内共享命令与背景。左侧原始麦克风数字注入，扬声器参考为零。英文 Whisper small.en 在 RTX 4090 上转写从通知回看起点至输入结束的整份录音，以完整命令逐词匹配转写末尾计分。[源混音和原始官方 Willow 对照](../../experiments/c161_v5_household_rooms.md)

| 固定 v5 的命令信噪比 | 唤醒 /8 | 实际处理后 FIFO 首句完整 /8 | 同一区间原始麦克风首句完整 /8 |
| --- | ---: | ---: | ---: |
| 0 dB | 8 | 1 | 4 |
| 6 dB | 8 | 6 | 7 |
| 12 dB | 8 | 6 | 8 |

原 C161 单独听固定音唤醒 11/24，合并 v5 后达到 24/24。普通合成 Nyx 唤醒 6/24，处理后完整 5/24，同区间原始麦克风完整 6/24。两路合并在 Willow 和静音各 24 条上都无通知。96 段里，固件中 C161 和 v5 各自的全部通知位置均与历史主机记录逐样本相同。此前官方 Willow 在它自己的 24 条上唤醒 21 条。两间房、两个文本和三个重复电平的结果用于本组配对分析。

30 次打开 FIFO 均保留完整的命令开始区间，最小提前量 2.07 秒。只改变首句来源、保持相同录音起止位置后，原始麦克风在固定音组恢复六条完整命令，普通 Nyx 组恢复一条。真实面板播放时需要同步参考和回声处理；此组覆盖面板扬声器静默时的背景音乐和空调。独立模拟应用 435,056 B，在 41,040 个 16 ms 块上，测量到的处理均值为 0.716 ms、最大 11.350 ms。真实 I2S、全应用并发、实际丢帧及整机最低堆仍依赖板测。[逐条通知和转写](../../results/4090-c161-mac-listener-household.json) · [96 份 UART 及触发后的 FIFO 样本](../../results/4090-c161-mac-listener-household-uart.zip)

## 复现

从仓库根目录运行，Mac 源提交须已获取。使用 ESP-IDF 5.5.4、ESP-SR 2.6.1、esp-tflite-micro 1.3.3~1、esp-nn 1.1.2 与 esp-micro-speech-features 1.2.3，前端使用 Mac 提交中的分配检查补丁。QEMU 使用 ANSI 神经网络内核和参考 FFT，并在 DSP 运算时屏蔽 tick 以避开模拟器浮点舍入 CSR 的延迟上下文切换问题；C161 内部进度日志在此模拟构建关闭。源 WAV 的生成和路径见上方配对实验。

```sh
app="$PWD/research/wakeword_p4/first_command_p4/mac_listener"
export NYX_MAC_WORK="$PWD/.local/wakeword/c161-mac-listener"
export NYX_SR_COMPONENT="$PWD/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
fft="$PWD/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__dl_fft"
bash "$app/prepare.sh"
idf.py -C "$app" -B "$NYX_MAC_WORK/build" -DIDF_TARGET=esp32p4 \
  "-DEXTRA_COMPONENT_DIRS=$fft;$NYX_MAC_WORK/features" "-DSDKCONFIG=$NYX_MAC_WORK/sdkconfig" \
  "-DSDKCONFIG_DEFAULTS=$PWD/research/wakeword_p4/midlevel_p4_selftest/sdkconfig.defaults" reconfigure
cmake --build "$NYX_MAC_WORK/build" --parallel 2
source .local/wakeword/c161-onset-names/env
taskset -c 2,3 /usr/bin/python3 "$app/replay.py" run
taskset -c 2,3 /usr/bin/python3 "$app/replay.py" score
```
