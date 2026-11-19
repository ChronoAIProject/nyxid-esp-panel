# P4 单麦降噪实验

探针接收预先合成的整段 16 kHz 单声道 PCM。每个输入冷启动一次。闪存偏移 2 MiB 处放小端 uint32 魔数 `0x324e534e`、编号、样本数，随后紧接 PCM16LE。前端从样本零连续运行，最后一帧补零。串口给出每帧起始样本、返回值、客机耗时以及小端输出样本。两种构建共用串口协议，`NSNET PASS` 表示指定输入的全部帧已完成。调用方必须校验帧数和样本偏移。

### ESP-SR NSNet2

工具链为 ESP-IDF 5.5.4，组件为 ESP-SR 2.6.1，其同一 managed_components 目录含 espressif__dl_fft 和 espressif__cjson。先激活 IDF 环境，再从仓库根目录运行下列命令。模型包来自该组件的 nsnet2 目录与 pack_model.py。

```bash
export IDF_COMPONENT_MANAGER=0
export NYX_SR_COMPONENT="$PWD/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
export NYX_NSNET_WORK="$PWD/.local/wakeword/c161-nsnet2"
bash research/wakeword_p4/nsnet_p4/build.sh
```

默认配置将 QEMU 的外部 RAM 地址登记给堆分配器，并链接原厂 P4 算子。加上 `NYX_NSNET_ANSI_FFT=ON` 可复现使用组件 ANSI FFT 后的下一处停止位置。实板诊断构建请使用单独的新目录，令 `NYX_NSNET_BOARD=ON` 并保留原厂 FFT。该构建使用 IDF 的 PSRAM 初始化。此处实测编译的配置为 P4 revision 3.1 起、200 MHz PSRAM、4 MiB 闪存布局，具体板型需匹配其芯片、内存和分区参数。这是独立闪存输入探针。完整面板固件和音频驱动的测试由 Mac 执行。

### 上游 RNNoise

使用 Xiph RNNoise v0.1.1 自带权重和标量 C 算子。仓库内两级因果 FIR 做 16→48→16 kHz 转换。每级滤波器的群延迟为 48 kHz 的 15 样本。测试保留 RNNoise 窗口本身的时延。模拟器按 160 样本处理一个 10 ms 帧，在一次运算期间屏蔽中断，以避免 QEMU 在任务切换后触发 ROM 浮点 CSR 异常。串口传输在计时区间外。此构建使用内部 RAM。

```bash
git clone --branch v0.1.1 --depth 1 https://github.com/xiph/rnnoise.git .local/wakeword/c161-rnnoise-src
export NYX_RNNOISE_SRC="$PWD/.local/wakeword/c161-rnnoise-src"
export NYX_RNNOISE_WORK="$PWD/.local/wakeword/c161-rnnoise"
bash research/wakeword_p4/nsnet_p4/build-rnnoise.sh
```

在既有 4090 研究环境，从仓库根目录运行下列命令。Python 使用原 C161、房间重建素材、已建 P4 Opus 固件和已有的 Whisper 权重。模拟器与编译的 CPU 亲和性沿用研究机的两个物理核心。

```bash
source .local/wakeword/c161-onset-names/env
runner=research/wakeword_p4/first_command_p4/mac_listener
nice -n 10 taskset -c 2,4 /usr/bin/python3 "$runner/check_room_rnnoise.py" collect
nice -n 10 taskset -c 2,4 /usr/bin/python3 "$runner/check_room_rnnoise.py" score
nice -n 10 taskset -c 2,4 /usr/bin/python3 "$runner/check_v5_rnnoise_uplink.py"
```

逐条输入哈希、完整 UART 与 Opus 包随结果入库。采音按 256 样本分块。开麦按所需前端帧最后一个样本到齐的块边界评分。普通称呼另给出串行叠加逐帧客机计算耗时后的通知时刻。最低堆和栈高水位覆盖这个独立程序，整机实时性需在音频、网络和编码同时运行时测量。

仓库含原始评分报告。评分命令在报告已存在时核对其归档 SHA256。重新计算可设置 `NYX_NOISE_RESULTS` 指向空目录，按输入哈希与固件 ELF 哈希复用本机完整串口记录。
