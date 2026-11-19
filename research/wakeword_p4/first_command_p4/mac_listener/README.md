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

## 唤醒后取消或停播面板音频

对此前 12 个带播报的条件，使用首次 P4 通知所在读块的实际交接时钟控制面板播放。模拟立即或延后 500 ms 开始 10 ms 淡出，到录音结束前均抑制面板输出。分别重新生成同步参考与回声，保留停播前的原始麦克风样本和停播后的房间衰减。用量化后的安静麦克风、参考和原房间响应复算旧完整混音，与原 PCM 的最大偏差小于 1 个 16 位量化单位。独立核对两种停播条件下的通知源、首条采样时钟和录音窗均与各自原记录一致。这是在模拟声场中用实测通知驱动播放的配对试验，实际 DAC 停播延迟需在板端测量。

| 首句完整 /12，数字写法等价 | 按原时序播报 | 通知后立即淡出 | 通知后 500 ms 淡出 |
| --- | ---: | ---: | ---: |
| 原麦 | 3 | 10 | 11 |
| 延迟 90 ms 的原麦 | 5 | 12 | 12 |
| 始终 AEC/AGC | 5 | 9 | 9 |
| 随参考切换 | 5 | 11 | 12 |

其中 8 个条件在唤醒时尚未开始播报。两种停播时序都完整取消了其后排定的播报，所得双声道逐字节等于原安静条件，复用相应的四条安静 P4 录音和转写。其余 4 个条件在通知前已经开始出声，形成 8 条新双声道输入。全部重新经 P4 监听器与 Opus，主机核对了 884,160 个选路样本和 11,056 个编码包。在这 4 个已出声条件中，延迟 90 ms 的原麦从 1/4 提高至两种停播时序均 4/4；按参考切换由 1/4 提高至 3/4、4/4。立即停播时失去的一句是 “set the hallway lights **at** 20%”，目标为 “**to** 20%”。两条命令、两种声音各自绑定，足以核对这次差异，500 ms 的时序优势还需更多独立语料和板测。[逐条停播对照](../../results/4090-c161-mac-playback-transition-stop-comparison.json) · [停播时钟和输入来源](../../results/4090-c161-mac-playback-transition-stop-inputs.json) · [8 条新双声道](../../results/4090-c161-mac-playback-transition-stop-stereo.zip) · [逐句转写](../../results/4090-c161-mac-playback-transition-stop.json) · [P4 上行包](../../results/4090-c161-mac-playback-transition-stop-uart.zip)

Mac 的实机播放包有三条 16 kHz 单声道 PCM16 WAV。两条外部音轨各自完整包含冻结 v5 和合成命令，第三条是面板播报。按包内清单的采样位置同步，从独立扬声器播放外部轨，在 P4 播放面板轨并回采实际参考。清单给出原始时序，停播对照使用板上实际通知。记录通知点、首次与最后一个有效参考采样点、原麦及编码入口，另记两端播放增益、摆位和采样时钟。归档 SHA256 为 `2f526e4a05d7cf4976dd1f522b835893df73002e0dcc33cf1d4dae25cbaee927`。[Mac 实机播放包](../../results/4090-c161-mac-playback-transition-board-wavs.zip)

在相同研究环境中，从仓库根目录运行 `stop_playback.py prepare`、`stop_playback.py score`、`stop_playback.py compare` 可重建干预输入、上行回放及对照表，脚本位于本目录。评分也可用 `--inputs research/wakeword_p4/results --out <新的目录>` 直接消费归档记录。用 `board_playback.py <输出.zip>` 可从冻结生成器重新导出三条音轨，各个干声的 PCM 均校验回原计划。

## 两路检测同时核对自播固定音

保留厨房和客厅的全部 24 个固定音左声道，分别把相同的 v5 归为外部播放（面板参考为零）或面板自播（同步参考包含该固定音）。后续命令与背景保持一致。原监听器只在 v5 匹配器上核对播放参考。24 次自播的 v5 全部被拦截，另一路 C161 仍单独触发了 11 次。零参考时同一组原麦共有 24 次外部唤醒。[逐路原始通知](../../results/4090-c161-mac-self-cue.json)

研究固件提供双路检查的可选回放协议。参考活动时，每路阳性最多等一个 C161 步幅（480 个样本，30 ms）。这一窗口内若播放参考识别出 v5，丢弃待发通知；参考匹配后的 3,488 样本内也拦截两路。参考活动以最近 500 ms 内是否有绝对幅度至少 64 的 PCM16 样本判定。播放静默时立刻放行。模型、平滑、阈值、预热和两秒回看沿用原 C161 配置。[研究固件的逐采样实现](main/reference_wake.h)

| P4 QEMU 条件 | 唤醒 / 输入数 |
| --- | ---: |
| 面板自播固定 v5 | 0/24 |
| 相同麦克风音频，固定 v5 来自外部 | 24/24 |
| 四个自播样本的参考各提前或落后 16 ms | 0/8 |
| 外部普通合成 Nyx，安静播放参考 | 6/24，与原 C161 逐次相同 |
| 取上述实际命中的六句加入 440 Hz 纯电参考诊断 | 6/6，C161 放行延后 30 ms |
| 面板与外部在同一时刻播放固定 v5 | 0/2 |
| 先面板自播，外部在 3 秒后播放固定 v5 | 2/2，两次仅外部获准 |

同刻同音的外部请求也被否决。两路保护依赖板端同步参考能识别出固定 v5。90 次各自冷启动的正式回放逐样本检查了两路允许通知的时钟，另行复验原单声道协议的四条匹配用例。可在同一构建环境运行本目录的 `self_cue.py` 和 `self_guard.py`。[90 次逐路结果](../../results/4090-c161-mac-self-cue-guard.json) · [90 份 P4 事件日志](../../results/4090-c161-mac-self-cue-guard-events.zip)

## 用数字播放参考提前识别面板自己的固定音

在 MIT 加噪浴室输入中，面板自播 v5 时，C161 比播放参考的完整三段匹配早 924 个样本。既有 480 样本等待窗在第三段到来前已放行。增加一条仅作用于扬声器参考的检查：引用原有的两段同极性匹配器，在第二段匹配时检查该段内 32 ms 的数字波形。命中后，两条麦克风通知从该时刻起可被拦截 1,944 个后续样本；完整三段匹配继续覆盖余下的回声时段。例中第二段在 30,204 样本被识别，早于 31,200 样本的 C161 阳性和 32,124 样本的第三段。[实现](main/listener.cpp) · [提前窗口](main/reference_prefix.h) · [数字波形检查](main/reference_signature.h)

数字波形取原 v5 第 14,800 样本起每隔四个样本的符号，共 128 位。两段匹配通知到来时，仅在预期位置前后各 48 个样本内搜索，接受同极性或整体反相时最多 12 个符号差异。原始 WAV 为 0 个差异，编码振幅加倍的 WAV 为 5 个。输入为数字扬声器参考。麦克风接收的固定音继续用原三段模板检测。[逐输入符号差异](../../results/4090-c161-mac-reference-prefix-qualification.json)

| P4 QEMU 逐次冷启动 | 原三段参考保护 | 提前匹配并检查数字波形 |
| --- | ---: | ---: |
| 三个困难房间，外部原 v5 | 4/6 | 4/6 |
| 同一组麦克风，面板自播原 v5 | 1/6 误开 | 0/6 误开 |
| 外部编码振幅加倍版 | 6/6 | 6/6 |
| 同一组麦克风，面板自播加倍版 | 1/6 误开 | 0/6 误开 |

270 个已用房间的全量结果另用主机测得的两路麦克风事件和实际数字参考事件计算。原 v5 安静、加噪自播分别从 2/270、246/270 误放行降至 0/270、0/270；增强版分别从 0/270、233/270 降至 0/270、0/270。所有房间在此项计算中使用同一电气参考增益和播放时钟，表中三个困难房间单独做了 P4 模拟。在这 24 次模拟中，新固件处理了 6,576 个 16 ms 读块，最慢 5.360 ms，超时为零，研究应用最低内部空闲堆为 381,132 B。整机资源和调度由实机采集。更换实机参考的延迟、混音方式或传输格式会改变结果。[主机逐房间时钟](../../results/4090-c161-mac-reference-prefix-signature-host.json) · [24 次 P4 事件](../../results/4090-c161-mac-reference-prefix-signature-rooms.json)

三批已消费背景的旧两段匹配器共有 12 个事件。复读这 12 个原参考片段并重现两段匹配时钟，数字检查得到 38–53 个符号差异，全部超过准入上限。计入的连续背景总量仍为 28.001 小时。另把其中一处真实谈话匹配人为安排在外部 C161 真阳性前 30 ms。单靠两段声学匹配会拦下这条命令，加入数字检查即可放行；距离改为 200 ms 时两种方法均放行。该诊断的参考是电气注入，麦克风复用同一条外部命令。[两段匹配对照](../../results/4090-c161-mac-reference-prefix-collisions.json) · [加数字检查的 P4 对照](../../results/4090-c161-mac-reference-prefix-signature-collisions.json)

数字直送及反相衰减、2% 增益参考均能提前确认两份 WAV。另以 20 ms 帧和八种 Opus 配置压缩后解码至参考端。在单位增益下，原 v5 的提前检查通过 3/8，增强版 6/8，完整三段参考两版都通过 8/8。两处同时播放相同固定音时，参考保护会抑制外部请求。板上需记录实际参考的来源和采样时钟。[输入传输与增益](../../results/4090-c161-mac-reference-prefix-qualification.json) · [90 次历史场景复验](../../results/4090-c161-mac-reference-prefix-signature-regressions.json)

复验沿用本目录的 ESP-IDF 构建步骤，将构建目录设为 `.local/wakeword/c161-mac-listener/build-reference-prefix`。在研究 Python 环境运行 `prefix_guard.py rooms --signature`、`regressions --signature`、`collisions --signature` 和 `host --signature`，可用 `--out <新文件>` 指定输出。运行 `prefix_guard.py qualification` 可核对背景与传输格式。回放协议字为 `0x4452594e`，双声道依次放原麦克风和同步扬声器参考的连续 PCM16。
