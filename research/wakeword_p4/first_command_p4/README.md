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
