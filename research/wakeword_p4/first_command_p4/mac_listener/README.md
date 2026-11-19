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

## 外部设备连续播放固定音和合成命令时的电平

四种既有合成声音的远场命令在唤醒后只剩 −6.36 至 −0.46 dB 命令信噪比。试验使用同一份增强版 v5，只处理紧接的数字命令。对照均沿用同一房间响应、接收增益、噪声波形和首音素位置。发送端给命令最多 12 dB 增益，PCM 样本峰值上限为 −1 dBFS。线性版本每句使用一个增益，限幅版本在短峰值附近收低增益，预先处理已生成的整句，输出样本数和采样时钟与原件一致。[发送端实现](../../experiments/check_c161_command_playback_level.py)

| 同一组 12 条命令 | 按首音素截取，原始 / 线性 / 限幅全句 | 从两秒回看起点转写到录音末尾，原始 / 线性 / 限幅完整命令 |
| --- | --- | --- |
| 数字直送 | 11 / 11 / 11 | 12 / 12 / 12 |
| 近场 | 8 / 10 / 7 | 11 / 10 / 12 |
| 远场 | 0 / 1 / 2 | 0 / 1 / 3 |

36 个目标的合并通知时钟逐条复现。将相同处理单独用于已经混入环境噪声的远场麦克风命令段，完整转写仍为 0/12。72 条相同前文和命令的 Willow 与静音替换对照，在发送端限幅后均零通知。[原始波形身份、通知与逐词转写](../../results/4090-c161-command-playback-level.json)

在 4090 上另外合成四种已用训练声音的两条新命令，每条分别通过 MIT 训练分区厨房空调和客厅音乐。用原命令有效帧将两组基准都配到 0 dB，环境和公共接收增益随后固定。增强固定音各条件均唤醒 16/16。按整份主机原麦回看录音的转写末尾是否包含完整命令计分，原音厨房、客厅分别为 5/8、3/8，发送端限幅分别为 7/8、7/8。16 个逐句配对保留全部原有成功。限幅后命令信噪比为 9.03–11.18 dB。另做等能量对照，按同一批房间的原命令有效帧将限幅后的命令送入增益调回约 0 dB，厨房、客厅分别为 7/8、5/8。此对照用到了已知模拟房间响应。替换为字面普通合成 “Hey Nyx” 时，处理前后均唤醒且首句完整 1/16；相同噪声与命令的静音唤醒词对照均零通知。[原始干声](../../results/4090-c161-command-playback-level-new-dry.zip) · [新房间逐句比较](../../results/4090-c161-command-playback-level-new-rooms.json) · [等能量对照](../../results/4090-c161-command-playback-level-new-rooms-matched.json)

旧远场原始与限幅输入各 12 条，全部经过带数字参考检查的 P4 QEMU 监听器。面板播放参考为零，从两秒回看开始，将实际通话队列和同期原麦分别送入 Espressif Opus 编码器。两路各解码 5,408 个 20 ms 包。两路的整段录音均从原始 0/12 提至限幅后 2/12。通话队列恢复 am_eric 的 “Lower the fan speed.” 和 am_fenrir 的 “Start the kitchen timer.”；后者转写开头还有一句多余的话，符合本表既定的命令末尾匹配口径。af_aoede 的同一句风扇命令在两组都失败。一次监听器回放在通过标记前重启，原失败串口未保留；相同 PCM 冷启动复验通过。另有一个 16 ms 读块耗时 18.457 ms。板上首词、延迟、丢帧和稳定性待 Mac 回采。[P4 开麦和逐包结果](../../results/4090-c161-command-playback-level-p4.json) · [串口事件及原始编码包](../../results/4090-c161-command-playback-level-p4-packets.zip)

供 Mac A/B 的包中有两种声音各两条 16 kHz 单声道 PCM16，顺序为同一增强版 v5 后直接接合成命令。用同一外部扬声器、音量和摆位播放，记录面板原麦、通知点与实际 Opus 解码音。清单给出每条 WAV 和 PCM 的 SHA256、命令起点及对应远场模拟输入身份。归档 SHA256 为 `f0677ef6d325b198aedb6af9a6caf5e38e781ab6d64e76b5cd52cf75cd6adf54`。[四条实机播放 WAV](../../results/4090-c161-command-playback-level-board.zip)

## 四声音编码唤醒后的首句与 P4 上行

四声音、两条命令、两个房间的 16 组原始编码音，在同声音 Willow 对照的统一命令时钟上均命中。每条合成命令另生成一个最多增加 12 dB、峰值限制在 −1 dBFS 的版本。两组采用相同的唤醒音、房间、背景、命令起点和接收增益。发送端处理前后的麦克风前缀逐样本相等，32 次模拟开麦均早于命令文件起点 273–712 ms，无接收样本削波。面板播放参考为静音。[逐句原始、限幅、时钟、识别和 P4 资源](../../results/4090-c161-coded-command-uplink.json)

| 16 个配对条件，转写末尾包含完整命令 | 原始命令 | 提高电平并限幅 |
| --- | ---: | ---: |
| 已知命令起点截取的主机麦克风 | 9 | 16 |
| 从两秒回看起点转写主机麦克风 | 11 | 12 |
| P4 通话队列经 Opus，解码整段两秒回看 | 8 | 15 |
| 同一批 P4 通话队列 Opus 包，解码后从开麦前 100 ms 转写 | 7 | 15 |
| 同期原麦经 P4 Opus，解码整段两秒回看 | 9 | 15 |

原两秒回看五条漏转写中，am_onyx 的厨房命令 “Raise the bedroom blinds halfway” 仅将末词拆为 “half way”；客厅同句还将 “blinds” 读成了 “blind”。客厅的另三条漏转写涉及 am_onyx 的日历命令，以及 bm_lewis 的两句命令。评分沿用原词序和分词口径。提高命令电平后，P4 通话队列原有八次成功均保留，并找回七条，仅 am_onyx 在客厅的 “Read tomorrow's first calendar entry” 仍将首词读成 “Breathe”。本组 32 次监听模拟共 15,670 个 16 ms 块，有一次超过周期，耗时 27.956 ms。最低内部空闲堆为 381,132 B。两路各解码 10,866 个 20 ms 包。[监听事件与实际 P4 编码包](../../results/4090-c161-coded-command-uplink-uart.zip)

Mac 可用同一外部扬声器、相同音量和摆位，配对播放每种声音每条命令的 original 和 limited 文件。16 条 WAV 都是 16 kHz 单声道 PCM16；清单给出唤醒音哈希、发送端命令起点、整份 WAV 和 PCM 哈希。归档 SHA256 为 `a6ee56d7d2390fd676040eeb1ce6834abdf0e4c386768d2893bb5d735df678a7`。[四声音、两命令的实机播放包](../../results/4090-c161-coded-command-uplink-playback.zip)

## 低码率的固定音自播

使用上面的 ESP-IDF 构建步骤，将构建目录设为 `.local/wakeword/c161-mac-listener/build-reference-third`。在研究 Python 环境运行 `third_reference.py` 可复现浴室与淋浴间的双路回放、参考时钟偏移和普通讲话碰撞。协议字 `0x5452594e` 在原双路协议的同一采样点读取麦克风与播放参考，对未经签名确认的前两段匹配等候第三段。Mac 可优先对照加噪浴室的 12 kbps 自播和普通讲话参考同时收到外部唤醒这两个情形。[固定 WAV、模拟结果与实机取样要求](../../experiments/c161_synthesized_cue_matchers_v1.md#低码率自播的修复与时钟偏差)

## LCD-X8 待机参考与实机回采

Mac 提交 `0d8e1935` 的 LCD-X8 使用 ES7210 采集四槽 TDM，同帧依次为 MIC1、MIC3、MIC2、MIC4。板级适配器输出 MIC1 麦克风与 MIC3 扬声器线路回采，MIC3 的 ADC 增益为 0 dB。待机循环在通话指针为空时把送入 AEC 的参考置零。实机集成研究匹配器时，其输入可直接取归一化采集帧的第二槽 `capture[2*i+1]`。其他板型须按各自的音频布局取样。[板级格式](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/0d8e1935702a561b535a8c1a0921fd8034943d2c/boards/lcd_x8/product_audio_layout.h#L6) · [ADC 配置](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/0d8e1935702a561b535a8c1a0921fd8034943d2c/boards/lcd_x8/product_backend.c#L54) · [待机取样](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/0d8e1935702a561b535a8c1a0921fd8034943d2c/components/panel_services/panel_voice.c#L1093)

在主机编译上述 Mac 提交的板级格式函数，按最多 128 帧一块解包四槽合成输入，再送入同一版 P4 QEMU 研究监听器。四个条件为两个已用加噪房间，各自播放原始 PCM 和 12 kbps Opus 唤醒音，随后紧接相同命令。模拟待机置零时，自播误开 4/4；输入同步理想参考或四种线路处理后的参考时，每组自播均误开 0/4。所有组外部播放都开麦。理想参考和低底噪线路均 4/4 在命令前交付，上行 Opus 完整命令 4/4。给线路加标准差 24 个 PCM16 单位的合成底噪时，两项均降至 2/4。现有算法仅需一个样本幅度达到 64 就将参考视为活动 500 ms，导致这组静音底噪期间几乎全程选择 AEC/AGC。三种高底噪设置的外部输入完全相同。48 次回放共 18,240 个读块，最慢 7.992 ms。这些衰减、滤波及底噪参数均为人工设置，板上频响与底噪仍待测量。[实验与参数](lcd_x8_reference.py) · [输入身份、参考标记和逐句结果](../../results/4090-c161-lcd-x8-reference.json) · [UART 与 Opus 包](../../results/4090-c161-lcd-x8-reference.zip)

对 Mac 实采的 16 kHz PCM16 WAV，运行 `python3 read_mr_capture.py capture.wav`。接受已归一化的双声道 M/R 或上述 LCD-X8 四槽格式。若开头恰有一秒已知静音，可加 `--idle-frames 16000`。工具需要 Python 3 与 C++17 编译器，使用仓库内与固件相同的固定音匹配器，输出每路幅度、静音段 RMS、参考两段与三段事件的采样时钟及 PCM 哈希。两路各有一个三段匹配时，额外给出参考减麦克风的标记时钟差；C161 回退开麦需结合板上 UART 时钟判读。用同一批 12 kbps 加噪浴室数据复算双声道及四槽输入，四个对照的全部固定音时钟与归档逐一相等，已知 ±256 样本偏移均能还原。[回采分析工具](read_mr_capture.py)

### 底噪下的播放活动判定

研究协议 `0x4152594e` 以连续 160 样本为一帧，参考去直流后的 RMS 达到 64 个 PCM16 单位时，才更新现有的 500 ms 播放保持计时。在板侧用整数方差实现。原参考逐样本直送 AEC 和两段、三段固定音检测；播放活动只控制麦克风通知的短暂等候，以及通话队列在原麦和 AEC/AGC 之间的选择。通过与独立主机参考逐样本核对，验证了 P4 实际送入 FIFO 的混音。Mac 侧回采工具也可输出已知静音期间的活动帧数。[能量判定](main/reference_activity.h) · [监听器入口](main/listener.cpp)

同一批加噪浴室和淋浴间重放 48 次，逐条核对了四槽输入、解包后的两路 PCM 和原始检测事件。在底噪 RMS 为 24.16 的外部播放中，命令前交付由 2/4 恢复至 4/4，解码上行的完整命令由 2/4 恢复至 4/4。原先的两处差异分别是浴室转写出 “closing upstairs curtains” 和淋浴间漏掉 “curtains” 的复数结尾。四种模拟线路设置各自的面板自播误开均为 0/4，待机参考置零仍误开 4/4。安静线路底噪 RMS 为 3.04，外部播放两版均完整 4/4。能量版使用 P4 上行编码器、RTX 4090 Whisper，48 次共 18,240 个读块，最慢 16.062 ms，有一块超过 16 ms，模拟器最低内部空闲堆 381,132 B。按前述 ESP-IDF 构建步骤将构建目录设为 `build-reference-energy`，运行 `lcd_x8_reference.py --energy` 可复验。[逐条输入、开麦时钟及转写](../../results/4090-c161-lcd-x8-reference-energy.json) · [原始 UART 与编码包](../../results/4090-c161-lcd-x8-reference-energy.zip)

### 普通语音候选首次误唤的来源

以采样时钟关联音乐配对前级的提议、教师复核与最终通知。此前 66 段、28.001 小时连续背景的 22 次误唤分布在 16 段中，全由辅助模型先触发；同样的输入上原 C161 为零。例如 `vox-test45-22` 的首次通知在样本 1,667,520，附近转写为 “I'm very happy”，`vox-test45-32` 在 2,172,480，附近为 “promise, I promise”。上下文只定位误唤附近的语音。带段界的 108 段普通谈话、96 条独立日常句和 48 条普通 Bot 回复均无通知。8 段 Bot 示范有 4 段误唤，均来自 bm_fable。数字、反射两版的 “To wake me, say Hey Nicks” 分别在 765,120 和 765,600 样本首次通知，后一句 “Hey Nicks, turn on the kitchen lights” 两版均在 800,640。表中分别保留 C161 命中标志、辅助提议采样点、五类复核输出与逐流 PCM 身份。[首条通知与全部触发段](../../results/4090-c161-music-proposal-first-notifications.json)

### 能量判定与播报重叠时的首句

同一 P4 固件切换峰值与能量两种协议，重放此前锁定的 22 条双路输入。两版开麦时钟逐一相同。面板连续播报期间的外部请求和完整首句均为 2/2，面板自播误开 0/2，普通 Bot 单独播放误开 0/2。另 16 个厨房和客厅条件均开麦，上行 Opus 完整命令由峰值版的 7/16 变为能量版的 6/16。唯一变化是客厅音乐、面板在命令中途短播时，Whisper 将 “reading light” 转写成 “breathing light”。此组能量版 11,326 个读块均在 16 ms 内，最慢 5.845 ms。能量判定可作为 X8 两版实播对照，静音底噪与真实面板出声时都需要采样。[配对开麦、上行转写和资源](../../results/4090-c161-reference-energy-playback.json) · [原始编码包](../../results/4090-c161-reference-energy-playback.zip)

首次测试的第 37 号编码器串口在一个包中间插入了任务看门狗输出，校验程序拒收。已保留该日志，完整重放 44 次，并在写入归档前验证两路每个包的 PTS、样本起点与总数。第一次中断前已完成的 29 份上行记录，全部包体在复验中逐字节相等。[首次异常串口](../../results/4090-c161-reference-energy-playback-first-opus-watchdog.log.gz) · [复验程序](energy_playback.py)

### 已缓冲帧内的首次播放峰值

客厅短播的 “reading light” 回归发生在实际开麦之后。参考的首个幅度峰值在样本 64,520，10 ms 帧结束才在 64,639 报告能量达标，相差 119 样本（7.44 ms）。最后一次能量脉冲也比最后一个峰值晚 40 样本。两套方法的原麦／处理后混音权重仅在 [64,520, 64,798) 与 [94,520, 94,719) 两段有差异。单独恢复这两段共同的选路时序后，整个 FIFO 和解码上行均逐字节复现峰值版，并恢复 “Move the reading light closer to the window”。这项试验定位到两处切换边界的共同影响。[时钟、区间与上行哈希](../../results/4090-c161-framed-capture-transitions.json)

研究协议 `0x4652594e` 在 AGC 已收齐的 160 个样本上计算参考的去直流 RMS。RMS 达到 64 时，录音切换沿用帧内实际达到幅度 64 的各个采样点。待机唤醒判断继续在帧末取得能量脉冲。22 条输入的开麦时钟全部与能量版相同。16 条厨房、客厅输入的完整命令从 6/16 恢复至 7/16，与旧峰值版逐条一致。另两次连续播报中的外部命令均完整；面板自播与普通 Bot 单播各误开 0/2。[研究实现](main/listener.cpp) · [串口与 Opus 包](../../results/4090-c161-framed-capture-transitions.zip)

再回放已交接单音的 48 条浴室和淋浴间输入。在 RMS 24.16 的模拟线路底噪下，四条外部命令仍全部在首词前开麦且上行完整。四种模拟线路的面板自播各误开 0/4，待机参考清零的自播误开 4/4。两批实测共 29,566 个 16 ms 块，最慢分别为 7.848 和 5.417 ms，均无超时。复现使用构建目录 `build-reference-framed`，在研究环境运行 `framed_capture.py transitions` 和 `framed_capture.py rooms`。[困难房间报告](../../results/4090-c161-framed-capture-rooms.json) · [UART 和原始编码包](../../results/4090-c161-framed-capture-rooms.zip)
