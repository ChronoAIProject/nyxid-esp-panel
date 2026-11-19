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

### 八条交接音打断面板播报

使用交给 Mac 的八条 WAV，四个合成声音分别在同一个固定唤醒音之后直接接两种命令。逐个校验 WAV 的哈希和解码样本，在已用厨房空调和客厅音乐中生成 16 个工况。面板从样本零开始播放同一条合成播报，使用各房间此前冻结的面板回声增益。模拟 MIC3 取带通、反相、0.25 倍增益的播报信号，加标准差为 24 个 PCM16 单位的底噪。第一次完整播放，第二次在第一次 P4 实际开麦时钟开始 10 ms 淡出并保留房间余响；逐对校验开麦前麦克风与参考输入完全相同。开麦前 20 ms 的数字播报 RMS 为 287.24 个 PCM16 单位。同一播报相位用于所有配对。[复现程序](handoff_barge_in.py)

| P4 QEMU 与 RTX 4090 Whisper | 面板继续播报 | 开麦即淡出 |
| --- | ---: | ---: |
| 命令首帧前开麦 | 16/16 | 16/16 |
| 厨房，上行转写末尾包含完整命令 | 3/8 | 8/8 |
| 客厅，上行转写末尾包含完整命令 | 3/8 | 7/8 |
| 整段转写与命令逐词相同 | 5/16 | 14/16 |

两组均在样本 57,088 开麦，比命令首帧 57,600 早 32 ms，全部由固定音匹配器触发。淡出后保留原有六条完整命令，新增九条。客厅 bf_lily 的 “Close the upstairs curtains” 仍被转成 “Cloned to the upstairs curtains”。同一声音的另一句多出前缀 “Either”，其余单词完整。32 次回放共 13,892 个 16 ms 块，最慢 5.828 ms，均无超时；最低内部空闲堆 381,132 B，麦克风无削波。每组实收 4,308 个 20 ms Opus 包。板上从开麦事件到扬声器实际降音的延迟仍待实测。[逐条开麦、转写与时钟](../../results/4090-c161-handoff-barge-in.json) · [监听 UART 与原始 Opus 包](../../results/4090-c161-handoff-barge-in.zip)

### 相同面板播报下的 Hey Willow 与普通 Hey Nyx

四种合成声音各说两条命令，在厨房和客厅各重放一次。分别接上同声音的 “Hey Willow”、普通 “Hey Nyx”、交接用固定音，或把唤醒词换成静音。每个配对使用相同的命令、房间、背景片段、面板数字播报及回声增益，命令首帧固定在样本 57,600。官方 `wn9_heywillow_tts` 在经典 ESP32 QEMU 上以 DEFAULT 阈值处理原麦输入。P4 QEMU 监听器读取同一原麦和模拟 MIC3 参考。两者均连续预热 81,600 个样本。所有输入均无削波。[输入身份与运行时](../../results/4090-c161-handoff-willow-plan.json)

官方 Willow 在自己的 16 条语音上均无通知；此前逐条核对的同一组无面板播报的麦克风输入有 5/16 次通知，其中 2/16 在命令首帧前。C161 在普通 “Hey Nyx” 上，两种播放条件均为 0/16。持续播报时，固定音在 P4 匹配器上提前开麦 16/16；静音唤醒词接命令两套系统均为 0/16。三种有声唤醒词在原输入的相对声级不同。固定音和普通词分别对应设备播放与语音识别两种使用方式。[逐条回放时钟](../../results/4090-c161-handoff-willow.json) · [官方原始返回和模拟串口](../../results/4090-c161-handoff-willow.zip)

另做接收声级诊断。将各路麦克风减去配对的静音唤醒词输入，得到该词及其房间余响。在各自的唤醒词时间窗内，把词与其余声音的 RMS 比值配到相同场景的固定音水平。仅缩放该词及余响，32 条均无削波，实得比值与目标的最大误差小于 0.00002 dB。这一计算用到了模拟房间中可单独取出的唤醒词和干扰。普通 “Hey Nyx” 增加 1.18–6.56 dB，C161 仍为 0/16。官方 “Hey Willow” 增加 2.01–7.54 dB 后有 4/16 次通知，其中 3/16 早于命令。四条都属于 bm_fable，三次提前 60 ms，厨房的花园灯命令晚 36 ms。两者互测对方的普通词各 0/16。板上说话人泛化仍需独立证据。[声级、逐词通知和采样时钟](../../results/4090-c161-handoff-willow-level.json) · [官方逐块返回与 UART](../../results/4090-c161-handoff-willow-level.zip)

### 开麦到淡出的延迟与上行转写起点

沿用八条交接 WAV 和两个房间，增加两种扬声器响应时序。开麦后 20 或 80 ms 开始 10 ms 淡出，分别在命令首帧前 12 ms 和之后 48 ms 开始。每个工况逐样本校验淡出前的麦克风及参考。保留 P4 编码的全部 Opus 包，另按此前四声音实验的做法，仅把 Whisper 的输入起点移到开麦前 100 ms。该起点由 P4 通知和录音队列的时钟计算。[淡出程序](handoff_fade_delay.py) · [解码与截取程序](handoff_command_window.py)

| 面板播报 | 两秒回看，末尾含完整命令 | 两秒回看，整段逐词相同 | 开麦前 100 ms 起，末尾含完整命令 | 开麦前 100 ms 起，整段逐词相同 |
| --- | ---: | ---: | ---: | ---: |
| 继续播报 | 6/16 | 5/16 | 6/16 | 5/16 |
| 开麦即淡出 | 15/16 | 14/16 | 15/16 | 14/16 |
| 延迟 20 ms 淡出 | 16/16 | 13/16 | 15/16 | 14/16 |
| 延迟 80 ms 淡出 | 15/16 | 12/16 | 16/16 | 15/16 |

开麦即淡出时，截短转写恢复客厅 bf_lily 的 “Close the upstairs curtains”，同时把 af_bella 的 “garden lights” 转成 “garden lines”。20 ms 组的两秒回看在两个房间各完整 8/8，有三条在命令前多出单词。80 ms 组的客厅 af_bella 窗帘命令用两秒回看时漏识，截短后整句恢复。四行都使用同一播报相位。[即时与持续两组的逐句结果](../../results/4090-c161-handoff-barge-in-window.json) · [两个延迟的逐句结果](../../results/4090-c161-handoff-fade-delay-window.json)

两个新增延迟的 32 次 QEMU 监听回放均提前 32 ms 开麦，无麦克风削波。共 13,892 个 16 ms 块，最慢 11.330 ms，零超时，最低内部空闲堆 381,132 B。第 20 号首次 Opus 编码的串口插入任务看门狗输出，校验拒收，已保留失败串口。相同 PCM 重新编码通过，两组分别验证并解码 4,308 个 20 ms 包。实机需记录开麦到扬声器线路开始下降的延迟及实收命令。[P4 时钟与转写](../../results/4090-c161-handoff-fade-delay.json) · [完整监听串口、编码包和首次异常](../../results/4090-c161-handoff-fade-delay.zip)

### 播报起点对固定音与首句的影响

使用同一段 7.3 秒的面板播报，每隔 14,600 样本取一个循环起点。每个起点都回放此前的八条交接 WAV 和两个房间，另以静音替换固定音，保留完整命令、背景和面板回声。冻结的 C161 与固定音检测器在主机按流式协议运行。128 组外部固定音均由匹配器在命令前开麦，其中 112 组提前 32 ms，16 组提前 64 ms。静音唤醒词对照误开 0/128。在起点 73,000 的同一条厨房输入中，有、无固定音的麦克风各削波 3 个样本。八个起点复用了同一批背景。[逐组输入、时钟及削波](../../results/4090-c161-handoff-panel-phase-grid.json)

按固定音三段编码的发送时刻，事先用面板数字播报在这三个窗口内的合并 RMS 选取最响的起点。原起点为 690，最响的起点为 87,600 样本，RMS 为 4,608 个 PCM16 单位。实际 P4 QEMU 双路回放时，固定音 16/16 在命令前 32 ms 开麦，静音替换 0/16。持续播报时的 Opus 上行转写末尾包含完整命令 8/16，整段逐词相同 7/16。开麦即做 10 ms 淡出并保留余响，两项均为 16/16。厨房完整数从 5/8 到 8/8，客厅从 3/8 到 8/8。48 次监听共 20,838 个 16 ms 块，最慢 13.359 ms，零超时，最低内部空闲堆 381,132 B；两个有声组各解码 4,308 个 20 ms 包。[P4 逐次结果](../../results/4090-c161-handoff-panel-phase.json) · [UART 与原始 Opus 包](../../results/4090-c161-handoff-panel-phase.zip)

Mac 可循环播放板用素材中的 `panel.wav`。若面板从文件开头播放，在第 7.975 秒开始由外部设备播放交接的固定音接命令，可复现上述数字源的相对位置。固定音长 1.10 秒，此时面板落在第二遍播报中。两次使用相同起播顺序、音量和摆位，分别保留持续播报及开麦后淡出的 M/R 与实际 Opus。[面板播报文件包](../../results/4090-c161-mac-playback-transition-board-wavs.zip) · [外部设备播放文件包](../../results/4090-c161-single-cue-advance-commands.zip)

### 从成对实采参考读取降音时钟

在本目录运行 `python3 measure_playback_drop.py continuous.wav fade.wav --control-open 57088 --trial-open 57088`。两个数字须分别替换为本次 UART 的 OPEN 换算到对应 WAV 后的采样位置，从零计。工具仅依赖 Python 3 标准库，读取 16 kHz PCM16 双声道的 R 或 LCD-X8 四槽的 MIC3，以 OPEN 前两个互不重叠的有声窗口估计两次播报之间的偏移、增益与极性。OPEN 前需有 800 ms 录音。对齐后的持续播报用于排除原声停顿。在试验录音自身的 OPEN 时钟上读取每个 10 ms 帧的交流 RMS，连续三帧有足够的对照信号且实采低至少 12 dB 时，报告首个低电平帧的起点 `first_low_frame_ms`。该值包含淡出和等候可测语音的时间；物理 DAC 开始改变的采样点须结合原始波形确认。[回采工具](measure_playback_drop.py)

以八个播报起点、独立线路噪声、反相与电平变化制作双声道及四槽合成 WAV。24 次已知淡出中，18 次可读出持续下降，另 3 次因对照信号太弱、3 次因开麦前对齐不足而无法判定。8 个持续播放对照均没有报告下降，其中一个无法对齐。最响起点的三组在开麦后 0、20、80 ms 开始 10 ms 淡出，首个低电平帧依次在 10、30、90 ms。另三组不同播报位置、空参考和周期纯音均报告无法对齐。[已知时钟、逐帧电平及所有未判定样本](../../results/4090-c161-playback-drop-reader.json)

### 固定音接命令经过 AAC、SBC 和重采样

复用八条交接 WAV，在外部播放端对“固定音接完整命令”整段处理，再送入原厨房和客厅的房间响应与噪声。面板没有播报。两条压缩路径分别是 44.1 kHz 双声道 AAC 128 kb/s 和 48 kHz 双声道 SBC 328 kb/s，两声道输入相同。另有原始 PCM 和仅做对应采样率往返的对照。独立校准噪声测得编解码固定延迟为 372、24 个 16 kHz 样本，回放前统一对齐。下表的提前量从麦克风样本上的命令起点计算。主机对每条交接音另做静音替换，只保留原有命令，并让整段通过同一编码器。[输入哈希、固定延迟及逐条回放](../../results/4090-c161-handoff-playback-codecs.json)

| 外部播放 | 命令前开麦 /16 | 静音替换后误开 /16 | P4 QEMU 上行末尾含完整命令 /16 | 两秒回看整段逐词相同 /16 |
| --- | ---: | ---: | ---: | ---: |
| 原始 16 kHz PCM | 16 | 0 | 16（既有回放） | 4 |
| 44.1 kHz PCM 往返 | 16 | 0 |  |  |
| 44.1 kHz AAC | 16 | 0 | 16 | 4 |
| 48 kHz PCM 往返 | 16 | 0 |  |  |
| 48 kHz SBC | 16 | 0 | 16 | 3 |

全部 80 条有固定音的主机回放均提前 64 ms 开麦。AAC 和 SBC 的 32 条正例另用现有 P4 QEMU 监听固件和 Opus 编码器冷启动复核，通知时刻全部与主机一致。解码收到的完整上行包后，从开麦前 100 ms 开始转写，两种编码分别为 16/16 整段逐词相同。实验采用软件编解码，没有蓝牙空口传输；真实 Mac 扬声器、P4 麦克风与上行的联合结果仍待板端采集。[解码与回放程序](handoff_playback_codecs.py) · [监听串口与全部上行包](../../results/4090-c161-handoff-playback-codecs.zip) · [交接用八条播放 WAV](../../results/4090-c161-single-cue-advance-commands.zip)

### 测量固定音播放与回采的时钟

解压并单次连续播放 63 秒的 `hye-nyx-v5-clock-63s.wav`，同时保留完整的 16 kHz PCM16 双通道 M/R 或 LCD-X8 四槽录音和板上 UART。音频内含七次原字节固定音，首末间隔 60 秒。录音可提前开始。自播时让面板连续播放整份 WAV，麦克风和电气播放参考均从同一份录音取出。外部设备播放时，麦克风一路即可估计相对播放端的速率。压缩包 SHA256 为 `f36d1b6d12040b7c625d4cdd80e96b98429fa595555a68f6a6f98239fcfd0df6`，解压后 WAV 为 `e2e0ad2c49ffa85de45fa19263ac98dc4ef59d2e5944da3ef2bfb6784bbe7e5e`。[播放包](../../results/4090-c161-cue-clock-reader.zip)

在本目录运行以下命令。工具调用现有 C++17 固定音匹配器，逐样本读取录音。也可以用 `python3 measure_cue_clock.py generate playback.wav` 从仓库的单次固定音重建播放文件。

```bash
python3 measure_cue_clock.py measure capture.wav > capture-clock.json
```

`mic.excess_capture_ppm` 和 `reference.excess_capture_ppm` 的正值表示相同源间隔在录音中占了更多采样点。若两路都能测量，`ref_minus_mic_samples` 列出七次的相对时钟差，正值表示参考较晚。固定差值可包含声传播、处理和缓冲延迟，线性变化表示观测期间相对位置在移动。两路共用 ADC 时，单路相对播放端的速率只能描述端到端比值。某一路标记数量与播放计划不符或最大拟合残差超过 16 样本时，该路报告无法估计，保留原始事件供比对 UART。报告中的 ±100 ppm 模拟自测覆盖双通道及四槽格式。[读取程序](measure_cue_clock.py) · [已知时钟和异常的合成验证](../../results/4090-c161-cue-clock-reader.json)

### 检查编码前后能否找到同一条命令的开头、中段和结尾

在本目录运行下面的命令。第一个 WAV 可用交接包内已知命令起点的干声，或同一次实采的原麦。第二个 WAV 用实际接收的解码上行。两个文件均为 16 kHz PCM16，工具需要 Python 3、NumPy 和 SciPy。第一路取第一个 WAV 的通道 0，可直接读取原麦在首槽的双声道或 LCD-X8 四槽。第二个 WAV 默认取通道 0，其他通道用 `--channel` 指定。把示例的 17,600 替换为第一个 WAV 中命令起点的样本数；该值适用于交接的八条“固定 v5 加紧接命令”干声。连续录音可用 `--command-end` 指定命令末尾的排他边界。

```bash
python3 measure_command_capture.py playback.wav uplink-decoded.wav --command-at 17600
```

工具将命令降到 4 kHz，用三个 160 ms 的波形窗口独立寻找开头、中段和尾部，同时检查是否有另一处相关性相近的位置。报告中的位置均已换算为 16 kHz 样本。三个窗口都匹配且相对时间一致时，状态为 `three_windows_matched`。该状态只覆盖取样的三个窗口。两个窗口一致且推算的活跃开头或结尾超出录音边界超过 20 ms 时，报告相应的边界缺失。其余情况保留各窗口的位置与相关系数，标为无法判定。

对 30 条直连 P4 上行，干声对照找到三个窗口 29 次。对 16 条厨房和客厅回放，干声对照有 15 条无法判定，改用同一输入的原麦作参考后，16/16 均找到三个窗口。对 46 条上行分别截去前 200 ms、截去后 200 ms、或只将前 200 ms 置零，共 138 个对照均未报三段齐全。截断开头正确报出 25/46，截断尾部正确报出 41/46，其余截断样本均为无法判定。对两段各重复播放两次及两段静音对照也返回无法判定。实机可同时保存原麦、解码 WAV 和 OPEN 时钟，核对板上通话链路。[测量程序](measure_command_capture.py) · [逐窗口实测与合成故障](../../results/4090-c161-command-capture-windows.json)

### 固定音与 Bot 同时说不同命令

播放包内有 12 条外部 WAV、两条面板 WAV 和配对清单。Mac 可直接使用单声道 16 kHz PCM16 源。每对从同一时钟的零点起播，面板使用清单中的 `panel`，外部扬声器使用 `external`。外部轨在第 5 秒开始固定 v5，随后说 “can you unlock the side door?”；面板同时示范 “Hey Nicks, turn on the kitchen lights.”。文件名中的 −6、0、+6 dB 指模拟麦克风在唤醒音区间的相对电平，实机值随扬声器音量、摆位和房间而变。播放包 SHA256 为 `a86cb485789a859accbf916f34b000b28e24fcc0e689f86e141dc1f62c2879af`。[独立面板与外部播放轨、manifest](../../results/4090-c161-fixed-cue-over-spoken-bot-playback.zip)

每组可分别播外部音、只让面板示范、两路同时播放。对同一配对保持摆位和音量，比较面板持续播报，以及在该次实板 OPEN 后请求面板淡出。外部命令需播完。从同次 UART、原麦 M、回采参考 R 和 Opus 解码 WAV 可分别读出开麦时钟、面板实际降音时钟、上行中是否保留了外部命令。清单内的 `qemu_open` 是模拟结果，实机淡出由本次 OPEN 驱动。上面两项时钟及捕获工具也可用于这组文件。[回放及完整转写](../../results/4090-c161-fixed-cue-playback-stop.json)

### 同一监听任务中运行 SpeexDSP 和固定 v5

在上述监听器的构建命令中，设置环境变量 `NYX_SPEEXDSP_SOURCE` 指向解压后的 SpeexDSP 1.2.1 源目录，使用独立构建目录 `$NYX_MAC_WORK/build-speex-capture`，增加 `-DNYX_SPEEX_CAPTURE=ON`。此配置保留 ESP-SR AEC、AGC、原麦 C161 和固定音匹配器，在同一任务的 160 样本帧内执行浮点 SpeexDSP，以其输出写入两秒回看和通话 FIFO。使用签名播放参考协议 0x4452594e。源归档 SHA256 为 `8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d`。从仓库根目录进入上文使用的研究 Python 环境后，运行本目录的 `check_joint_speex_capture.py`。[监听代码](main/listener.cpp) · [验证脚本](check_joint_speex_capture.py)

对 12 条独播、12 条双讲及两条不同的纯 Bot 音频，联合固件的每个 C161、v5、保护后通知和 OPEN 与原监听器一致。双讲时全部提前 36 ms 开麦，独播和双讲的各 12 份实际 FIFO 经 P4 Opus 后，Whisper small.en 的转写末尾均含完整门锁命令。两条纯 Bot 未开麦。全程 14,308 个包已重新解码核对。双讲时联合固件与前述独立 Speex 固件的 12 个 FIFO 均有字节差异，此处使用联合固件自身编码输出评分。[逐条结果](../../results/4090-c161-joint-speex-capture.json) · [UART 事件和完整编码包](../../results/4090-c161-joint-speex-capture.zip)

联合固件为 457,424 B，原监听固件为 438,256 B。每次初始化 Speex 增用 73,832 B 内部堆。26 次回放的最低内部空闲堆为 307,300 B；各次 16 ms 块的平均处理耗时为 1.48–1.75 ms。16,250 个块中有一次耗时 34.725 ms。同一输入再次独立启动三次，OPEN 和 FIFO 哈希均复现，峰值为 6.989、7.287、6.932 ms，三次均在预算内。模拟计时覆盖监听、回声处理和队列，Opus 由独立模拟应用编码。物理整机仍需测量持续采音、播放参考和通话并发时的峰值与丢帧。

### 联合监听器在厨房和客厅的两条命令

用已交接的四种合成声音各说关窗帘、关花园灯两条命令，分别放入厨房和客厅的房间响应与背景。在每个房间条件下分别让面板持续播报，或从该次实际 OPEN 开始用 10 ms 淡出。面板参考通道沿用已有带限、反相、增益和线路噪声的输入。32 次均逐样本哈希核对原始双路输入；原版和联合版的 C161、固定 v5、参考保护和实际 OPEN 事件一致，全部比标注命令起点早 32 ms。[回放程序](check_joint_speex_rooms.py) · [逐句结果](../../results/4090-c161-joint-speex-rooms.json)

| 完整的两秒回看经 Opus 解码，转写以外部命令全部单词结尾 | 面板持续播报，small.en /16 | 持续播报，large-v3-turbo /16 | OPEN 后淡出，small.en /16 | OPEN 后淡出，large-v3-turbo /16 |
| --- | ---: | ---: | ---: | ---: |
| 原选中上行 | 6 | 9 | 15 | 16 |
| 同次原麦上行 | 11 | 11 | 16 | 16 |
| 联合监听器的 SpeexDSP 上行 | 16 | 15 | 15 | 15 |

若从 OPEN 前 100 ms 开始转写，持续播报时 SpeexDSP 两个读者分别为 15/16、16/16；淡出时也分别为 15/16、16/16。新回放的 17,232 个 Opus 包复解后 PCM 哈希全部一致，其中原麦音轨的每个包还与旧运行逐字节相同。13,892 个模拟监听块有一个超过 16 ms。这里复用了相同的八段外部音，实机的声传播路径和任务调度仍需单独测量。[事件与编码包](../../results/4090-c161-joint-speex-rooms.zip)

### 在同一个 P4 模拟固件内并发采音和编码

以上一节的 Speex 构建为基础，使用独立构建目录 `$NYX_MAC_WORK/build-speex-opus`，增加 `-DNYX_CAPTURE_OPUS=ON`，将上行实验使用的 `esp_audio_codec` 2.4.1 加入 `EXTRA_COMPONENT_DIRS`。从仓库根目录运行 `check_concurrent_speex_opus.py`。模拟监听任务优先级 3，每 16 ms 的定时器通知提供 256 个采样。固定音触发后创建优先级 4、48 KiB 栈的独立编码任务，沿用产品的 320 样本帧和每批最多四帧的调度。两秒历史和实时音频进入同一 FIFO，编码包先存到模拟外部 RAM，结束后通过 UART 导出。此处收包端从 OPEN 起就绪。[模拟固件](main/capture_opus.h) · [复现脚本](check_concurrent_speex_opus.py)

12 条固定音和 Bot 双讲回放均提前 36 ms 开麦，完整门锁命令经 Opus 解码后转写正确 12/12；两个不同的纯面板示范均未开麦。全部 3,564 个编码包与先前独立编码器输出逐字节相同，归档串口复解的 PCM 哈希一致。FIFO 峰值为 32,000 样本，从 OPEN 到编码器追至不足一帧为 26.910–35.275 ms。联合固件 627,472 B，含两路监听和 Speex 的上版为 457,424 B。双讲各次最低内部空闲堆均为 236,192 B，编码任务栈最小余量为 25,068 B。14 次共 8,750 个采音块，有 12 个处理耗时超过 16 ms，最慢 37.502 ms；有一个读块开始时还积压了一个定时器通知。实板的 I2S、UI、网络和服务端就绪等待需由同次板测日志确认。[逐次资源和首句](../../results/4090-c161-concurrent-speex-opus.json) · [带 PTS 的原始 Opus 包和调度计数](../../results/4090-c161-concurrent-speex-opus.zip)

### 同优先级下的实际开麦时刻与通话建立等待

沿用上述并发固件构建方式，运行 `check_speex_opus_setup.py`。四个已有固定 v5 双讲输入均取 0 dB，监听与编码任务采用产品中的相同优先级 4。在模拟器启动周期采音定时器时记录微秒时钟，以同一时钟记录交付 OPEN、创建编码任务和首个编码包完成。两种协议为 `0x4f52594e` 和 `0x5752594e`，分别在 OPEN 后立即允许启动编码任务、等足 1.5 秒后在下一个采音块允许启动。等待期间采音写入同一 FIFO。

四对输入的 OPEN 样本时钟都比命令起点早 36 ms，客机微秒时钟的提前量为 33.400–33.848 ms。立即启动时首个编码包比命令开头早 25.032–25.660 ms，FIFO 最多 32,000 样本。模拟 1.5 秒建连等待时，首个包在 OPEN 后 1.511–1.512 秒完成，FIFO 增至 56,000 样本，OPEN 后 1.548–1.551 秒追至不足一帧。两种配置各 1,188 个包都与原独立编码实验的同源输出逐字节相同，命令转写各 4/4 完整。共 5,000 个块，均在 16 ms 预算内，最低内部空闲堆为 236,192 B。这里测到的是本地编码包完成时钟，实板网络发送与服务端收到首包的时钟仍由 Mac 回传。[逐条计时与开麦后积压](../../results/4090-c161-speex-opus-setup.json) · [实际编码包和计时串口](../../results/4090-c161-speex-opus-setup.zip)

### 同声线 Nyx、Nick 和 Willow 的连说播放包

包内 36 个 16 kHz 单声道 PCM16 WAV，覆盖四个声音、三条命令和三个称呼。每个文件从头到尾一次播放，按同一声线和命令比较三种称呼，保持外部扬声器、摆位及音量相同。每组三条的命令都从样本 48,000（3 秒）开始，此后 PCM 逐字节相同。门锁命令的 Nyx 文件以 001、028、055、082 开头，冻结 C161 单独开麦 2/4；加整数提前复核的研究路线开麦 4/4、Opus 首句完整 4/4，其中 1 条在命令开始前开麦。另两条命令也保留了 Hey Nick 的误开样本。清单中模拟时钟以 16 kHz 样本计。实板可按文件名回传 M/R、OPEN、首包及解码后的首句。固定 v5 的板测包见上文。[下载 36 条 WAV 和 manifest](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-connected-names-mac-playback-v1.zip) · [逐文件清单和校验和](../../results/c161-connected-names-mac-playback-v1.json)

已从 Release 匿名重新下载，校验 2,091,542 B 归档及全部 36 条 WAV、PCM、称呼区间和命令区间。归档 SHA256 为 `bfaad614a4fdef31f01b7edc9c6a86da03a8b38d1e8b6e1e5da1019010053932`。
