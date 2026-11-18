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
