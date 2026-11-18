# 辅助路双输出 INT8 的 P4 数值自测

在已有原主路和辅助路的配置中，用 65096 字节的双输出模型替换辅助路使用的模型。辅助前端、卷积和状态变量按同一音频时钟更新一次。每次 Invoke 后，输出索引 0 给原辅助分数，索引 1 给对话前文训练后的辅助分数。两者都是 UINT8，量化单位 1/256，原主路继续运行冻结 C161。双输出模型 SHA256 为 `a7776065ae71cbbccdfb2db8a76c5333805987353d475360e7fe5c7b69f47987`。[双输出模型](../results/4090-c161-dual-readout-v1.tflite) · [935 条主机流逐步验证](../results/4090-c161-dual-readout-v1.json)

按前文切换的研究配置使用 WebRTC VAD 模式 0，每 160 样本更新。出现连续 30 个有声块后，连续第 20 个无声块的末端开启 48000 样本窗口。以关闭时刻为开区间端点，窗口内取输出 1，其他时刻取输出 0。逐步选择之后再送入原辅助路五步平滑器，与主路共享原 67 步通知冷却。对比时可用真实回放的开窗样本和逐步输出定位第一个分歧。[开窗与仲裁参考](../experiments/check_c161_speech_history_gate_v1.py) · [逐流窗口和通知](../results/4090-c161-speech-history-gate-v1.json)

自测内嵌 bf_lily 的第一条新 Nyx 会话所产生的 489 个辅助输入块，包含预热。每块携带 120 个 INT8 输入字节和两份单输出模型各自的期望分数，45 个时间步两路期望值不同。原件 fixture.bin 的 SHA256 为 `ec3cd22ea028a45c833d104e352715646492288c2921458369a8743c3bfde014`，在 Linux 训练环境中运行 make_fixture.py 可从原始 PCM 与冻结分数重建。

在已初始化的 ESP-IDF 5.5.4 环境从此目录构建默认硬件配置。

```sh
idf.py -B build-p4 -D IDF_TARGET=esp32p4 build
```

P4 QEMU 的 ANSI C 对照使用独立配置。

```sh
idf.py -B build-ansi -D IDF_TARGET=esp32p4 -D SDKCONFIG=sdkconfig-ansi -D SDKCONFIG_DEFAULTS=sdkconfig.qemu.defaults build
python3 run_qemu.py build-ansi /path/to/qemu-system-riscv32 report.json uart.log
```

硬件优化构建已通过编译。当前 QEMU 在 esp_nn_set_conv_scratch_buf_esp32p4 的 P4 向量扩展指令处报 Illegal instruction。改用同组件的 ANSI C 内核后串口返回 `NYX_DUAL_PASS calls=489 distinct=45 errors=0,0`。在这份独立程序中，分配给张量的 65536 B 中实际使用 28744 B，变量区预留 16384 B。模拟器中 489 次单辅助解释器调用最慢 3186 µs。Mac 实机需分别测默认优化内核的数值、整机 30 ms 调度、内部堆、丢帧和采集至编码器的首样本。[默认优化构建 UART](../results/4090-c161-dual-readout-p4-uart-v1.log) · [ANSI C 数值见证](../results/4090-c161-dual-readout-p4-qemu-ansi-v1.json) · [ANSI C UART](../results/4090-c161-dual-readout-p4-uart-ansi-v1.log)
