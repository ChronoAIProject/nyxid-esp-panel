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

## 首句经 P4 上行 Opus 后的比较

将上面全部 30 次实际通知的处理后队列和同区间原始麦克风录音各自送入独立的 Espressif Opus 2.4.1 编码器，每次在 P4 QEMU 读取完整 PCM。采用面板设置，16 kHz 单声道、20 ms 帧、24 kbit/s、VBR、复杂度 0。主机 libopus 1.3.1 按 PTS 顺序解码 16,940 个包，用同一 4090 Whisper 对整份录音计分。两路各有 8,470 个包，同一对录音使用相同起点、终点和包边界。无外部 flash 载荷时，另复验原六次回放的 1,553 个包，包体、源采样位置和 PTS 全部复现。

| 固定 v5，面板扬声器静默 | 0 dB，完整 /8 | 6 dB，完整 /8 | 12 dB，完整 /8 | 总完整 /24 |
| --- | ---: | ---: | ---: | ---: |
| AEC/AGC 队列经 P4 上行 Opus | 2 | 5 | 7 | 14 |
| 同区间原始麦克风经 P4 上行 Opus | 4 | 7 | 7 | 18 |

四个新增完整首句来自 0 dB 和 6 dB 各两条。逐句配对中，原始麦克风路径保留了处理后路径的全部 14 条成功。普通 Nyx 按 24 个原始输入计，唤醒 6 条，两路各完整 5 条和 6 条。模拟比较覆盖无面板播放、零参考、无丢包的录音。实际播放回声和板端采集状态需要单独记录。[上行包与逐句比较](../../results/4090-c161-mac-listener-household-uplink.json) · [原始 P4 编码包及 UART](../../results/4090-c161-mac-listener-household-uplink-uart.zip)

按上级首次命令项目说明，在 `.local/wakeword/c161-first-command-v5/build` 重建 Opus 应用；在研究 Python 环境运行 `taskset -c 2,3 /usr/bin/python3 research/wakeword_p4/first_command_p4/mac_listener/uplink.py`。脚本直接从上一节已归档的 UART 还原通话队列，并逐条检查其 SHA256。

两个评分程序将复核报告和串口归档默认写入本地 `.local/wakeword/c161-mac-listener/recheck`；用 `--out` 可指定新的复核目录。重读归档回放已复现 96 条监听结果、30 组成对编码结果及两份 UART 归档的 SHA256。

## 面板开始或停止播报时的首句

先在 `3527edf2` 固定切换规则，再用 RTX 4090 合成两种声音的两条命令和一路面板播报。将每条命令放入厨房空调和客厅音乐，背景信噪比 6 dB。外部声音和面板回声分别用同类房间的两条独立 MIT 响应模拟。面板连续播放时，命令有效帧内外部语音与回声为 0 dB。每对房间、声音各观察全程安静、命令中途开始播报、中途停播、仅命令中段播报四种时序。后面三种共享同一潜在播报和增益。16 个条件各独立启动 P4 QEMU，固定 v5 在全部条件均通知。最迟开始的录音也早于命令干声起点 2.225 秒。

切换只读取同步播放参考的当前和历史采样。PCM16 绝对幅值达到 64 时，录音在 10 ms 内转向 AEC/AGC，最后一个有效样本后保持 500 ms，再用 10 ms 淡回原麦。原麦用 1,440 样本环形延迟对齐。实际进入两秒回看和通话 FIFO 的全部 1,908,480 个选路样本（含旧场景触发）与主机独立计算逐一相等。四种录音截取完全相同的时间窗，分开经 P4 Opus 编码。共解码 23,868 个包。英文 Whisper small.en 在 RTX 4090 上读取整份解码音频，以命令完整出现在转写末尾计分。表内每格分母 4，总计分母 16。`20%`、`20 percent` 和 `twenty percent` 在此表计为相同，逐条报告另存此前的逐词计分。

| 面板播报 | 原麦 | 延迟 90 ms 的原麦 | 始终 AEC/AGC | 随参考切换 |
| --- | ---: | ---: | ---: | ---: |
| 全程安静 | 4 | 4 | 3 | 4 |
| 中途开始 | 1 | 2 | 2 | 2 |
| 中途停止 | 1 | 1 | 1 | 1 |
| 仅命令中段 | 1 | 2 | 2 | 2 |
| 总完整 | 7 | 9 | 8 | 9 |
| 原逐词计分总完整 | 5 | 7 | 7 | 7 |

切换与延迟对齐原麦的总数相同，但停播时厨房的 “Move the reading light…” 仅经处理或切换才完整，客厅的同一句经处理或切换则被转写为 “remove”。另用六条归档双声道回放核对，面板连续回复期间两次外部命令，处理与切换均完整 2/2，原麦两路均 0/2。面板自播固定音两次，参考匹配都否决了 v5，C161 两次均无独立通知。单播普通回复两次也未通知。在四条原单声道回归中，旧协议的两路通知、录音范围和触发后样本哈希复现。这里的两间房、两句命令和固定开发声音构成配对诊断，实际 I2S 参考、扬声器延迟和整机行为依赖 Mac 板测。[输入清单](../../results/4090-c161-mac-playback-transition-inputs.json) · [双声道 FLAC](../../results/4090-c161-mac-playback-transition-stereo.zip) · [逐句转写](../../results/4090-c161-mac-playback-transition.json) · [P4 UART 与编码包](../../results/4090-c161-mac-playback-transition-uart.zip) · [配对统计](../../results/4090-c161-mac-playback-transition-summary.json)

沿用上方 Mac 运行时构建和 P4 Opus 固件。在研究 Python 环境运行下列命令即可从已冻结的输入重测，输出在本地 `.local/wakeword/c161-mac-listener/transition`。用 `transition_inputs.py --out <新的目录>` 可另行合成输入。

```sh
app=research/wakeword_p4/first_command_p4/mac_listener
src=research/wakeword_p4/results
taskset -c 2,3 /usr/bin/python3 "$app/transition.py" score --inputs "$src" --out .local/wakeword/c161-mac-listener/transition/recheck
taskset -c 2,3 /usr/bin/python3 "$app/transition_summary.py" --inputs "$src" --output .local/wakeword/c161-mac-listener/transition/recheck
```
