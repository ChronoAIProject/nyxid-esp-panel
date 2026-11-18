# ESP32-P4 双速声学提议自测

在 ESP-IDF 5.5.4、esp-tflite-micro 1.3.3~1 和 esp-nn 1.1.2 上，用两个有独立状态的 TFLM 解释器运行相同的 77496 字节中间层模型。按 30 ms 时钟交付原生前端三个连续特征帧；另一路把每对特征帧按 1、2 次依次复制，每凑满三个送入扩展解释器一次。两路各自在物理时钟上对最新 UINT8 分数做五步平滑。QEMU 使用 esp-nn ANSI C 内核。

测试覆盖四个声音的十二条原速连说请求。脚本从已发布的完整 WAV 重建原生输入，每个语流包含 81600 样本预热；两路期望分数来自此前独立保存的主机轨迹。每个时钟核对两个分数及两个提议位。UART 分别报告两路张量和变量分配器空间，以及模拟器执行每个物理输入块的耗时。测量范围从已量化的输入特征开始，到两路提议位产生为止。完整面板另有麦克风前端、原 C161、短语确认和音频采集任务。

在仓库根目录使用研究 Python 环境和已初始化的 ESP-IDF 环境重建；构建目录位于已有磁盘预算下的本地研究目录。

```sh
/usr/bin/python3 research/wakeword_p4/midlevel_p4_selftest/make_fixture.py
probe_build="$PWD/.local/wakeword/c161-midlevel-p4/build"
idf.py -C research/wakeword_p4/midlevel_p4_selftest -B "$probe_build" -DIDF_TARGET=esp32p4 \
  -DSDKCONFIG="$PWD/.local/wakeword/c161-midlevel-p4/sdkconfig" reconfigure
cmake --build "$probe_build" --parallel 2
/usr/bin/python3 research/wakeword_p4/midlevel_p4_selftest/run_qemu.py
```

## 数值与资源

首次使用 esp-nn ANSI C，全连接层沿用发行版的重新量化。5710 个物理时钟上两路合计 11420 个 UINT8 分数，其中五个与主机相差 1 或 10。五个时刻的提议位均一致。分别仅将全连接、普通卷积、深度卷积切到组件内的参考分支，均出现相同的五处差异。保留中间张量后，五处的 bounded_teacher 全连接输入（张量 63）逐字节相同，输出（张量 72）均比主机高 1。最后一个时刻的另一个全连接输出（张量 71）也有差异。对张量 72 的实际累加值 1100、1420、1228、1036，单次舍入重现主机输出，先做高位乘法舍入再做右移舍入则重现模拟器输出。[原配置的五处差异](../results/4090-c161-midlevel-p4-qemu-ansi.json) · [逐时刻中间张量比对](../results/4090-c161-midlevel-p4-qemu-ansi-trace-comparison.json)

自测默认仅为全连接参考内核编译同一 TFLM 组件的单次舍入函数，使用独立符号。模型 SHA256 为 217f1a10a31eef92a8ca0baf95f0c41f0bbf72ae2335071c96423d3a1d4cb331。在相同的 12 条输入上，原速实际调用 5710 次，扩展调用 8563 次，两路在物理时钟读取的 11420 个分数和全部提议位逐一匹配主机。在这份独立程序里，两路各使用张量分配器 25468 B，单独的变量分配器各使用 608 B。静态数组实际每路各预留 65536 B 和 16384 B，模拟器最低内部堆为 430756 B，最慢物理块为 4282 µs。每块时间只覆盖这两路推理、输入重排和提议判定。Mac 上的实际采集、整机资源与 PocketSphinx 确认开销仍待测量。[通过报告](../results/4090-c161-midlevel-p4-qemu-ansi-fc-single-round.json) · [原始 UART](../results/4090-c161-midlevel-p4-qemu-ansi-fc-single-round-uart.log)

重放结果写入 .local/wakeword/c161-midlevel-p4/results。对照配置设 NYX_FC_SINGLE_ROUNDING=OFF。可指定单个 NYX_REFERENCE_OPS 为 fully_connected、conv 或 depthwise_conv。在原始对照配置另开 NYX_TRACE_TENSORS=ON 可以复算中间张量，用同目录的 compare_tensors.py 与主机比较。包含分数差异的对照运行会保存证据并以非零状态退出。
