# 固定 v5 唤醒后的首句编码

ESP32-P4 QEMU 运行 v5 匹配器、面板的 `panel_voice_audio.c` 队列和生产依赖中的 Espressif Opus 编码器 2.4.1。三个已冻结的首句 WAV 分别在即时就绪和延迟 1500 ms 就绪下运行。以 160 样本作为输入块，保留 32000 个样本回看，在检测块末按生产代码的顺序转移环形缓存。每个 Opus 帧取 320 样本，16 kHz 单声道，24 kbit/s，复杂度 0，VBR。设定传输已经连接；服务等待期间送静音包，就绪后以每 20 ms 最多四包的模拟调度读取积压语音。实际研究固件据队列余量调度，在库中允许 60 秒缓存；本测试使用 4 秒容量，最大占用 3.5 秒。

| 合成输入 | 通知样本 | 编码器收到的首个 WAV 样本 | 命令首音素 | 首音素在 320 样本编码帧里的位置 |
| --- | ---: | ---: | ---: | ---: |
| 数字 | 102788 | 70880 | 106400 | 0 |
| 近场 | 102704 | 70720 | 106315 | 75 |
| 远场 | 102932 | 71040 | 106539 | 299 |

六次回放从回看起点到最后一帧，每次进入编码器的全部样本均与原 WAV 或末尾补零逐样本相同。延迟就绪增加 75 个先行静音包，首条命令仍落入同一个源音频帧。在主机用 libopus 1.3.1 解码 P4 输出包，命令段与输入互相关给出的编解码延时为 103–104 样本。4090 上 Whisper small.en 对数字与近场的原始命令和解码命令均转写出 “lower the fan speed.”；远场原始和解码命令均未转写出该句。三种输入代表一条合成命令的三个处理条件。测试从 WAV 开始，直接将样本馈入检测与录音缓存；麦克风、AEC/AGC 和网络仍需整机验证。[六次逐包和转写报告](../results/4090-c161-first-command-opus-v5.json) · [原始 UART 和 Opus 包](../results/4090-c161-first-command-opus-v5-uart.zip)

项目使用 ESP-IDF 5.5.4。选择 esp32p4、4 MB flash、Single factory app (large) 分区，并在配置时用 `EXTRA_COMPONENT_DIRS` 指向按根目录 dependencies.lock 解包的 esp_audio_codec 2.4.1。命令行可用 `idf.py -B build -DIDF_TARGET=esp32p4 -DEXTRA_COMPONENT_DIRS=/absolute/path/to/esp_audio_codec build`。研究脚本从仓库根目录运行，在 `.local/wakeword/c161-first-command-v5/build` 读取编译结果并启动 QEMU，随后从保存的串口包解码和转写。[模拟器与评分程序](../experiments/run_c161_first_command_v5.py)

## 远场唤醒后命令的信噪比

四种合成声音各读三条命令。原渲染按固定唤醒音的电平设定 7 dB 噪声条件。逐条分离命令与噪声并排除末尾填充后，12 条命令实际信噪比为 −6.36 至 −0.46 dB。保留同一房间响应、语速、音量和噪声波形，只调节噪声幅度。原条件的 12 条 PCM 均复现已冻结的 SHA256。将完整音频送入 v5 匹配器，将命令段送入 4090 上的 Whisper small.en，英语、无提示词。首词和全句评分均忽略大小写与标点。

| 远场背景噪声 | 唤醒段恰好一次通知，段外零次 | 首词正确 | 全句正确 |
| --- | ---: | ---: | ---: |
| 原始 | 12/12 | 0/12 | 0/12 |
| 降低 6 dB | 12/12 | 4/12 | 2/12 |
| 降低 12 dB | 12/12 | 7/12 | 4/12 |
| 关闭 | 12/12 | 7/12 | 6/12 |

在与上方 P4 远场回放相同的 “lower the fan speed.” 案例中，背景降低 12 dB 后转写为 “lower the band speed.”。该音频可供 Mac 检查首词，也能暴露命令中的错词。独立命令数为 12，各行重复使用这 12 条开发输入。[48 条逐项报告](../results/4090-c161-v5-far-command-noise.json) · [远场降低 12 dB 的 WAV](../results/4090-c161-v5-far-command-minus12.wav) · [复现程序](../experiments/check_c161_v5_far_command_noise.py)

## P4 回声消除和自动增益后的首句

将相同的三份 WAV 作为 16 kHz 麦克风输入，参考声道置零，采用面板的 ESP-SR 2.6.1 配置。AEC 使用单麦、单参考、8 个滤波分区、VOIP 高性能模式和强非线性抑制，每次处理 256 样本。AEC 输出按连续时钟分为 160 样本一帧，送入 AGC 模式 2（15 dB、限幅、目标 3 dB），再送入 v5 检测器。两次独立启动 QEMU，在每个条件下均得到逐样本相同的 AEC、AGC 输出和相同的通知位置。

| 同一条合成命令 | 输入 WAV 的通知样本 | 处理后通知样本 | AEC 后 Whisper 转写 | AEC + AGC 后 Whisper 转写 |
| --- | ---: | ---: | --- | --- |
| 数字 | 102788 | 104228 | lower the fan speed. | lower the fan speed. |
| 近场 | 102704 | 104144 | lower the fan speed. | lower the fan speed. |
| 远场 | 102932 | 104372 | 空 | We'll be right back. |

三条处理后音频均恰好通知一次，相对输入晚 1440 样本（90 ms）。用每条命令开始的 24000 个输入样本与 AEC 或 AGC 输出做归一化互相关，六组延迟估计也均为 1440 样本。表中转写在 4090 上运行，按各自延迟对齐并截取与原输入等长的命令段。原始远场输入转写为空，AGC 输出产生了一句与命令无关的文字。Mac 可用前两条检查首句，用第三条检查低信噪比时的错误转写。[每条的 PCM 哈希、转写和来源](../results/4090-c161-first-command-frontend-v5.json) · [带样本时钟的原始串口记录](../results/4090-c161-first-command-frontend-v5-uart.zip)

P4 QEMU 运行按根目录依赖锁校验的 ESP-SR 库。原生 dl_fft 0.8.0 在首个数字输入的静音预热阶段遇到模拟器缺少的 `esp.lp.setup`，指令编码为 `0x026fc02b`。完成回放的构建将正、逆浮点基 2 FFT 各切换到该组件的 ANSI 实现，并为仿真外部 RAM 注册 16 MiB 地址空间。测试在 DSP 调用期间屏蔽时钟中断，以避开模拟器恢复浮点状态时的异常。90 ms 是样本时钟上的音频处理延迟。实机优化 FFT 的输出、真实扬声器参考、I2S 采集、墙钟耗时和整机余量需由 Mac 在板上测量。

复现使用上面的 ESP-IDF 和分区配置。把 `NYX_SR_COMPONENT` 指向解包的 `espressif__esp-sr`，把 `EXTRA_COMPONENT_DIRS` 指向同级 `espressif__dl_fft`。在 `.local/wakeword/c161-first-command-v5/frontend-native` 构建原生 FFT，在同级 `frontend` 构建时额外传入 `-DNYX_QEMU_REFERENCE_FFT=ON`。从仓库根目录运行 `python3 research/wakeword_p4/experiments/run_c161_frontend_v5.py native boot repeat score`。评分程序会验证串口时钟连续、两轮输出相同、模型和库文件哈希，随后写出新的研究报告。[回放和评分程序](../experiments/run_c161_frontend_v5.py)

## 处理后的音频通过面板队列和 Opus

从上面保存的串口记录恢复三条 AGC 输出，逐条核对 PCM 哈希，然后在第二个 P4 QEMU 进程中将同一信号送入 v5、生产 PCM 队列和 Espressif Opus 2.4.1。保留 AEC 的 256 样本输入时钟、81664 样本预热和 AGC 的 160 样本成帧相位。每次模拟采集回调实际交付 160 或 320 个处理后的样本。发送端按 20 ms 节拍在首个可运行的采集回调读出队列，按既有回放协议测试服务立即就绪和延迟 1500 ms 两种条件。两段 QEMU 回放通过归档的 PCM 连接，真实的 I2S、网络和线程抢占留待整机测量。

| 输入 | 处理后命令起点（样本） | 首音素在 320 样本帧中的位置 | 解码后两种就绪条件的 Whisper 命令段转写 |
| --- | ---: | ---: | --- |
| 数字 | 107840 | 224 | lower the fan speed. |
| 近场 | 107755 | 139 | lower the fan speed. |
| 远场 | 107979 | 43 | 空 |

三个通知都发生在截止样本 104448 的同一采集回调。此时 AGC 已交付到 104416（不含），回看缓存从样本 72416 起送入编码器。六次在编码前逐样本检查全部有效载荷及末尾补零。服务延迟增加 75 个起始静音包，积压峰值为 56000 样本（3.5 秒），三条命令首音素均保留在相同源帧。数字和近场的编解码附加延迟均为 103 样本，远场为 104 样本。远场 AGC 命令段的错误转写在编码后变为空。要求核对首词可懂度时，数字和近场都有前后转写可对照。[六次逐包与转写报告](../results/4090-c161-first-command-after-agc-opus-v5.json) · [原始 Opus 包和源时钟](../results/4090-c161-first-command-after-agc-opus-v5-uart.zip)

从仓库根目录运行 `run_c161_after_agc_opus_v5.py prepare` 生成本地输入，再按本项目的 Opus 配置构建，设 `NYX_AGC_FIXTURES` 为 `.local/wakeword/c161-first-command-v5/after-agc-inputs` 的绝对路径，构建目录为同级 `after-agc-build`。随后运行该脚本的 `boot score`。基础 Opus 配置也重新构建并回放，六条原始输入的包字节和时间戳全部复现历史归档。[复现程序](../experiments/run_c161_after_agc_opus_v5.py)

## 相同输入上的单麦降噪

在已消费的四种合成声音、每种三句命令上，逐条校验 36 个完整输入的 PCM 哈希。两个降噪器都从录音第一样本处理到末尾，只评估送往语音识别的经处理信号；原 v5 检测仍使用原路径。ESP-SR 2.6.1 的 `ns_pro_create(10, 1, 16000)` 在原生 ESP32-P4 QEMU 上运行，10 ms 帧、中等抑制、单通道。36 次创建时可用堆均减少 40952 B；每条命令互相关估计延迟均为 96 样本（6 ms）。主机对照使用 DeepFilterNet v0.5.6 的发布可执行文件及其内置模型，逐条在单个 CPU 核心运行，补偿模型延迟。两者均用 4090 上同一 Whisper small.en、无提示词，识别等长的命令段。

| 同一组 12 条 | 原始首词 / 全句 | ESP-SR 首词 / 全句 | DeepFilterNet 首词 / 全句 |
| --- | ---: | ---: | ---: |
| 数字 | 11 / 11 | 11 / 11 | 11 / 11 |
| 近场 | 8 / 8 | 10 / 10 | 11 / 11 |
| 远场，命令 −6.36 至 −0.46 dB | 0 / 0 | 0 / 0 | 1 / 0 |

近场 ESP-SR 恢复了 af_aoede 的 “read” 和 am_eric 的 “start”。远场 DeepFilterNet 唯一首词正确的结果为 “Start the cheat time.”，目标是 “start the kitchen timer.”。这三个输入条件各含相同的 12 个命令实例、三种文本；单独的降噪实验从原始 WAV 开始。P4 实机整机的 AEC、AGC、Opus 和新增降噪一起运行的峰值内存与音质尚待测量。[P4 逐项报告](../results/4090-c161-v5-command-ns.json) · [P4 原始 UART](../results/4090-c161-v5-command-ns-uart.zip) · [DeepFilterNet 逐项报告](../results/4090-c161-v5-command-deepfilter.json) · [同一远场样本的降噪 WAV](../results/4090-c161-v5-command-deepfilter-far.wav)

切分边界也影响 Whisper。bf_lily 的数字样本按标注切到第一个音素时输出 “my shopping list.”；给相同 PCM 前置 200 ms 静音，得到 “Read my shopping list.”。12 条数字源命令前置等长静音后均全句正确。该诊断只改变送入识别器的起始上下文。

复现 P4 实验时沿用前端项目 ESP-SR 组件路径，用 `-DNYX_QEMU_NS=ON` 配置 `.local/wakeword/c161-first-command-v5/ns-build`，使用默认原生 FFT。每轮将三条录音写入研究 flash 镜像 2 MiB 以后的地址。主机实验按脚本中的 SHA256 校验发布文件，16 kHz 输入上采样到 48 kHz、增强后再返回 16 kHz。[P4 回放程序](../experiments/check_c161_v5_command_ns.py) · [发布模型对照](../experiments/check_c161_v5_command_deepfilter.py) · [DeepFilterNet v0.5.6](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6)

## 面板播报时的外部固定音和自播回声

将两路同步 16 kHz PCM 送入同版 P4 ESP-SR AEC、AGC 与冻结 v5 匹配器。左声道是模拟麦克风，右声道是面板扬声器的数字参考。对每段左声道保持所有样本完全相同，分别送入全零和实际合成播放参考。扬声器泄漏使用已登记的厨房与客厅实测脉冲响应。两种房间复用同一条外部 v5 和 “lower the fan speed.” 命令，以及同一个 Bot 声音。外部设备直达音与面板播报在有效命令帧中等功率。自播组从同一段 Bot 前文切换到面板自己播放 v5 和命令。第三组仅播放 Bot 普通回复。

| 两间房合计 | 参考置零时 v5 通知 | 送入播放参考时 v5 通知 | 置零时外部命令全句正确 | 送入参考时外部命令全句正确 |
| --- | ---: | ---: | ---: | ---: |
| 外部设备播放 v5 和命令，面板同时播报 | 2/2 | 2/2 | 0/2 | 2/2 |
| 仅面板自己播放 v5 和命令 | 2/2 | 2/2 | — | — |
| 仅面板普通播报 | 0/2 | 0/2 | — | — |

外部设备的 v5 均在输出样本 104228 通知。Whisper small.en 在参考置零时转写出面板的 kitchen timer 播报，使用播放参考后，两间房均转写出外部命令 “lower the fan speed.”。面板自己播放 v5 时，对唤醒音起点至录音末尾的 AEC 输出计算均方根幅度，与同一麦克风输入的零参考回放相比降低 42.92 和 42.49 dB。下游 AGC 后的 v5 匹配器仍在样本 103812 和 103836 通知，命令段的转写均为空。音量下降后依然有足以匹配固定波形的残余。麦克风实际输入的回放级别、参考通道串扰和整机调度由板侧测量。[逐条事件和转写](../results/4090-c161-v5-playback-reference.json) · [12 份同步双声道 WAV](../results/4090-c161-v5-playback-reference-stereo.zip) · [P4 QEMU 原始输出](../results/4090-c161-v5-playback-reference-uart.zip)

复现使用上方参考 FFT 前端配置，将构建目录设为 `.local/wakeword/c161-first-command-v5/echo-build`。从仓库根目录运行 `check_c161_v5_playback_reference.py replay score`，脚本按每次一个 QEMU 进程回放。无外部载荷的同一 ELF 复现了原三条零参考录音的全部 AEC 和 AGC 输出样本与通知时钟。12 段命令转写使用 4090、英语、无提示词，并按三条历史对照的 1440 样本延迟截取等长命令段。[回放与生成脚本](../experiments/check_c161_v5_playback_reference.py)

### 用播放参考识别面板自己的 v5

研究用的可选保护只作用于固定音通知。在 16 kHz 同步采集的右声道上运行第二个 v5 匹配器。参考声道匹配到 v5 后的 3488 样本（218 ms，含边界）内抑制麦克风固定音通知。窗口取此前测得的 1440 样本前端延迟，加 8 个 256 样本的 AEC 分区。这是本次仿真采用的容差。P4 对象符号大小为原麦克风匹配器 3448 B、增量保护状态 3456 B。逐样本调用 `allow(reference, mic_hit)`，其中 reference 是处理后麦克风输出所在样本序号对应的原始同步参考。在实际采集回调中需跨块保存参考样本，普通语音 C161 单独仲裁。[研究头文件](../portable/c161_coded_cue_reference_v5.h)

新 ELF 在 12 段既有输入复现全部原始 AEC 和 AGC 样本、所有麦克风通知时钟。在厨房和客厅回放中送入合成的同步播放参考后，两次面板自播通知均被抑制，两次外部唤醒都在原样本 104228 放行。参考置零时面板自播仍为 2/2。四条新增回放使用另一种 Bot 声音、另两份实测办公室和客厅响应以及 “Play the evening news.”。面板先播 v5，3 秒后外部扬声器再播时，两间房均压下第一次、放行第二次，两次外部通知都在样本 147024。两台扬声器同时播放同一固定音时，两间房都没有最终通知。办公室原麦克风分支有一次通知被抑制，客厅在保护之前就没有麦克风通知。两份相同标记重叠时，本试验无法区分外部唤醒意图。[16 次逐路事件与实测时间](../results/4090-c161-v5-playback-reference-guard.json) · [原始串口输出](../results/4090-c161-v5-playback-reference-guard-uart.zip) · [生成和独立核对程序](../experiments/check_c161_v5_reference_guard.py)

Mac 可先使用上面的双声道 WAV 做数字注入，按 WAV 第一帧作为样本零。左声道为麦克风，右声道为播放参考。再用相同输入做实体播放，记录真实同步参考、麦克风、v5 原始通知、保护后的通知和送往编码器的第一条命令；输入房间响应、左右声道同步、I2S 级别、循环运行成本和整体稳定性需要实机确认。

## 播放前经 Opus 压缩的固定 v5

把原 v5 WAV 单独送入 libopus 1.3.1，再把解码后的唤醒音接回原命令。16 kHz 单声道、每帧 20 ms，使用编码器报告的前瞻长度对齐。VOIP 和 AUDIO 两种模式分别测试 12、16、24、48 kbit/s。模拟房间、命令声音、噪声波形及增益均沿用原配对条件。原 WAV 的 36 段输入和通知逐项复现。八个压缩配置也分别在数字、近场、远场各 12/12 次于唤醒音内通知，每条恰好一次。各配置在两批 Piper 的 80 条非固定音语句上均无 v5 模板匹配；这些是重复处理同一批语句。原 WAV 及八个压缩版本的 RTX 4090 Whisper 转写均为 “Hey, Nix”，标点有别。[原 v5 WAV](../results/hye-nyx-c161-synthetic-cue-v5.wav) · [逐条比较](../results/4090-c161-v5-playback-opus.json) · [解码后的 WAV 和有长度前缀的原始 Opus 包](../results/4090-c161-v5-playback-opus-cues.zip)

取数字 VOIP 12、近场 AUDIO 12 和远场 AUDIO 16 kbit/s 三例，在 ESP32-P4 QEMU 运行面板采集队列与 Espressif Opus 2.4.1 上行编码器。远场取主机所有条件中最迟的一次通知。每个样本分别运行服务即时就绪和延迟 1.5 s 两次。六次通知均逐样本匹配主机，编码前 PCM 均通过逐样本比较，排队峰值分别为 32,000 和 56,000 样本。

| 输入 | 通知样本 | 命令开始样本 | 提前量 | 上行解码后的命令转写 |
| --- | ---: | ---: | ---: | --- |
| 数字 | 102788 | 106400 | 225.75 ms | lower the fan speed |
| 近场 | 102704 | 106315 | 225.69 ms | lower the fan speed |
| 远场 | 117584 | 120957 | 210.81 ms | 未正确转写 start the kitchen timer |

远场输入在上行编码前也未正确转写。三例均为模拟麦克风 PCM 的数字注入。实际扬声器、麦克风、音量、丢帧和整机内存需由 Mac 的实机记录确认。[P4 输入包](../results/4090-c161-v5-playback-opus-p4-wav.zip) · [文件 SHA256 与预期时钟](../results/4090-c161-v5-playback-opus-p4-inputs.json) · [P4 结果](../results/4090-c161-v5-playback-opus-p4.json) · [UART 和上行 Opus 包](../results/4090-c161-v5-playback-opus-p4-uart.zip)

复现时先运行 check_c161_v5_playback_opus.py，再运行 replay_c161_v5_playback_opus_p4.py prepare。沿用本项目的 ESP-IDF 5.5.4 和 esp_audio_codec 2.4.1 配置，指定 NYX_AGC_FIXTURES 为 .local/wakeword/c161-first-command-v5/playback-opus/fixtures 的绝对路径，并在同级 build 目录编译，再运行 replay_c161_v5_playback_opus_p4.py run。
