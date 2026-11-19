# P4 上的 PocketSphinx 短语确认

在 ESP-IDF 5.5.4 和 ESP32-P4 QEMU 独立应用中运行 PocketSphinx 5.0.4 原生 C 解码。使用已标定的 hey nicks、1e-15 阈值、25 字节两词发音、发行版英语声学参数和 30 ms 轮询。输入为上一轮双速中间层的实际提议。每次重建声学前端均值，并从提议前 1.5 秒回放，最多再送入 300 ms。每个候选独立计时。

62 个窗口包括旧原速语流 19 个、新对话 15 个和完整背景先前产生的全部 28 个候选。这些窗口可能属于同一句。相同输入下，P4 的短语段起止、首次返回时钟与最终确认在 62/62 个窗口逐项匹配主机。前两组分别确认 17 和 10 个候选，背景零确认。声学提议器和录音链路在另一个项目中测试。[逐个窗口来源和预期时钟](../results/4090-c161-lexical-p4-inputs.json) · [全部实测和二进制哈希](../results/4090-c161-lexical-p4-read.json) · [原始 UART](../results/4090-c161-lexical-p4-read-uart.log)

模拟器从只读 FatFs 加载模型。一次初始化为 2.86 秒，解码器就绪时相对初始化前增加 7940912 B 外部堆。128 KiB 工作任务栈实测高水位剩余 92420 B。逐个候选回放历史音频的中位计算时间为 357 ms，所有 30 ms 输入块的最大计算时间为 12.50 ms。将每个音频块末尾的可用时刻与实际计算时间合并，27 次成功确认的预计耗时为提议后 306–362 ms，中位 332 ms。队列竞争、采集任务、P4 实机时间和整机最低堆另需实测。

从仓库根目录运行 prepare.py，使用已固定 SHA256 的 5.0.4 源码包、英语模型及已有消费语料。脚本构造确定的原始 PCM 和外部存储卷，输出在 .local/wakeword/c161-lexical-p4。QEMU 构建将 CONFIG_NYX_LEXICAL_QEMU_RAM 设为 y，以 16 MiB 模拟 RAM 注册外部堆，并为 IDF 的浮点异常向量启用现有指令识别分支。该选项默认为关闭。模拟器首次读取 ROM 浮点舍入寄存器时未设置 lazy-FPU 原因位。任务堆栈从 32 KiB 增至 128 KiB 后完成初始化。移植使用目标上宽度为 32 的 int 定义两个上游类型，省去上游 CPU 进程计时，解码计时来自 esp_timer。读取模式使用上游文件加载分支。

```sh
app="$PWD/research/wakeword_p4/pocketsphinx_p4_selftest"
work="$PWD/.local/wakeword/c161-lexical-p4"
/usr/bin/python3 "$app/prepare.py"
printf '%s\n' 'CONFIG_NYX_LEXICAL_QEMU_RAM=y' > "$work/qemu.defaults"
idf.py -C "$app" -B "$work/build" -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/sdkconfig" \
  "-DSDKCONFIG_DEFAULTS=$app/sdkconfig.defaults;$work/qemu.defaults" reconfigure
cmake --build "$work/build" --parallel 2
/usr/bin/python3 "$app/run_qemu.py"
```

## 闪存映射与首句

映射构建从实际生成的 FatFs 镜像中定位两个完整且唯一连续的文件，结合准备清单核对 SHA256。解码器通过 esp_partition_mmap 映射 mdef（2,959,176 B）和 sendump（1,969,024 B），退出时释放映射。映射偏移相对本次 storage 分区。运行脚本按 UART 报出的实际偏移再校验镜像内字节，逐项核对 62 个窗口的匹配段、首次返回时钟及确认决定。[映射测量](../results/4090-c161-lexical-p4-mapped.json) · [原始 UART](../results/4090-c161-lexical-p4-mapped-uart.log)

| 独立 P4 QEMU 应用 | FatFs 读取 | 只读映射 |
| --- | ---: | ---: |
| 就绪时外部堆增量 | 7,940,912 B | 2,927,432 B |
| 初始化 | 2.860 s | 2.145 s |
| 单个窗口确认中位数 | 332 ms | 348 ms |
| 最慢 30 ms 输入块 | 12.496 ms | 15.773 ms |

每种实现各测一次。映射版节省 63.1% 外部堆；就绪时还占用 160 B 内部堆增量。模型和两词词典共 6,607,989 B。研究自测的 FatFs 镜像另含测试 PCM，面板现有 2 MiB SPIFFS 分区容不下这套完整模型。[面板分区表](../../../partitions.csv)

首句回放使用上述映射版每个窗口的启动、每个 30 ms 块和收尾耗时，按音频实际到达时刻排队。假定一个专用任务按先来先服务完整处理所有提议，原 C161 可独立通知，所有通知按 67 步冷却合并，录音从下一个 160 样本边界取数。62 个窗口中 13 个需要排队，最长等候 439 ms。27 个确认窗口从各自提议算起为 318–798 ms，中位数 357 ms，其中包含同一句的重复候选。实机采集、任务争用、通信就绪和 Opus 编码另需测量。

| 20 个合成目标 | 唤醒 | 即时起录后转写完整命令 | 回看两秒后转写完整命令 |
| --- | ---: | ---: | ---: |
| 旧四声音连说 | 11/12 | 0/12 | 11/12 |
| 四段新对话 | 6/8 | 1/8 | 6/8 |

分母包括全部目标。RTX 4090 运行无提示 Whisper small.en，比较转写末尾与命令全部单词。两秒回看也保留了 17 次唤醒的首个命令音素。旧语流 af_aoede 的一条请求没有提议。新对话 af_jessica 的两条请求都有提议，短语搜索均未确认。同批 66 段完整背景此前产生的 28 个候选仍为零确认。普通语音的实机基准为冻结 C161，固定音用 v5。[逐句、转写与队列记录](../results/4090-c161-lexical-p4-fifo-first-command.json)

复用上面的 app、work 和 SDK 配置，在仓库根目录运行：

```sh
idf.py -C "$app" -B "$work/build-mapped" -DNYX_LEXICAL_FLASH_MAP=ON -DIDF_TARGET=esp32p4 \
  -DSDKCONFIG="$work/sdkconfig" "-DSDKCONFIG_DEFAULTS=$app/sdkconfig.defaults;$work/qemu.defaults" reconfigure
cmake --build "$work/build-mapped" --parallel 2
/usr/bin/python3 "$app/run_qemu.py" --mapped
```

逐块测量写入本地 build-mapped/result.json 和 uart.log。首句脚本从已提交的映射版测量开始，生成文件时使用独占创建以保存原始报告。[首句脚本](../experiments/replay_c161_p4_first_command.py)

## 两词专用声学文件

PocketSphinx 的短语搜索还会用全部 42 个基础音素作为竞争项。保留这些音素以及 hey、nicks 所需的 6 个三音素上下文，再从混合权重的每一行取出实际使用的 144 个状态列。means、variances 和转移矩阵逐字节复用原文件。专用模型仅适用于这份两词词典及搜索配置。模型与词典由 6,607,989 B 降至 1,739,289 B。主机重放 625 段已消费语流中的全部 220 个提议，每次短语的分数、起止、首次返回及最终通知均与完整模型一致。[专用文件及原状态编号](../results/4090-c161-lexical-p4-compact-inputs.json) · [跨语料比较](../results/4090-c161-lexical-p4-compact-transfer.json)

同一 62 窗口在独立 P4 模拟器中均逐项匹配完整模型的短语时钟和确认结论，其中 28 个背景候选全部拒绝。初始化 2.382 s，就绪时外部堆增量 2,767,308 B，最慢 30 ms 音频块耗时 14.058 ms。按单工作任务排队重放，成功窗口提议至确认中位数 365 ms，最大 868 ms，含同一句产生的重复确认。RTX 4090 的首句完整转写在即时起录为 0/12 和 1/8，两秒回看为 11/12 和 6/8。[逐块用时](../results/4090-c161-lexical-p4-compact.json) · [原始 UART](../results/4090-c161-lexical-p4-compact-uart.log) · [首句转写](../results/4090-c161-lexical-p4-compact-fifo-first-command.json)

面板现有三份音乐文件共 1,446,087 B。专用声学文件及词典与音乐合计 3,185,376 B，超过现有 2 MiB SPIFFS 容量 1,088,224 B，另需文件系统开销。自测用独立研究分区，发布固件整机仍需单独验证。[现有面板数据](../../../spiffs/music) · [现有分区](../../../partitions.csv)

在上文准备步骤后，用 compact.py 生成专用模型。复用了已核对发行 wheel 中的原生格式转换函数。主机回归使用同一版本的 Python 扩展。

```sh
/usr/bin/python3 "$app/compact.py"
idf.py -C "$app" -B "$work/build-compact" -DNYX_LEXICAL_FLASH_MAP=ON -DNYX_LEXICAL_COMPACT=ON \
  -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/sdkconfig" "-DSDKCONFIG_DEFAULTS=$app/sdkconfig.defaults;$work/qemu.defaults" reconfigure
cmake --build "$work/build-compact" --parallel 2
/usr/bin/python3 "$app/run_qemu.py" --compact
```

## 保留现有音乐的 2 MiB SPIFFS 实测

RTX 4090 对专用模型的均值和方差分别拟合标量码本。每张表按 3 个特征流、每流首系数与其余系数分为 6 组，保留精确零值。在 8 位版本上，两张表再用 zlib 压缩，设备初始化时通过 P4 ROM 解压并恢复浮点输入，上游读表函数核对恢复值的 S3 校验和。声学文件与两词词典最终为 **455,703 B**。三份现有 MP3 共 1,446,087 B。按面板页大小和元数据配置，在 0xe00000 生成并挂载实际 2,097,152 B SPIFFS 镜像，逐份读回音乐并核对长度及 CRC32。测试 PCM 单独存放在研究 FatFs 分区。[权重、文件身份及参考窗口](../results/4090-c161-lexical-p4-q8-inputs.json) · [设备读回和逐块计时](../results/4090-c161-lexical-p4-q8.json)

| 已消费的 625 段语流、220 个提议 | 7 位 | 8 位 |
| --- | ---: | ---: |
| 匹配分数或匹配段有变化的语流 | 52 | 51 |
| 最终通知有变化的语流 | 5 | 0 |

7 位涉及连续对话、Joe 和家庭场景。8 位在 P4 重放 62 个窗口，短语段与首次返回时钟逐项匹配其自身主机参考；相对未量化的完整模型，58/62 个窗口的所有匹配段及返回时钟相同，62/62 的确认决定与确认音频时钟相同。28 个已知背景候选均被拒绝。两种位宽使用相同提议、阈值和已消费语料。[7 位逐项差异](../results/4090-c161-lexical-p4-q7-transfer.json) · [8 位逐项差异](../results/4090-c161-lexical-p4-q8-transfer.json)

首次模拟启动时，SPIFFS 连续读取和解码器初始化触发两次 CPU0 空闲任务看门狗。改为分块读取并显式让出处理器后，启动及 62 个窗口重放均通过，未再触发看门狗。该次解码器初始化 6.501 s，就绪外部堆增量 2,825,128 B，初始化前至最低外部余量之差为 2,979,968 B。27 个成功窗口的单任务 FIFO 确认中位数 336 ms，最大 747 ms，包含同一句的重复提议。旧组 11/12 和新对话 6/8 唤醒。RTX 4090 对两秒回看录音完整转写这 17 句，即时起录为 0/12、1/8。实际发布固件的 OTA 空间、全机调度、首句和长期资源需整机验证。[初次启动 UART](../results/4090-c161-lexical-p4-q8-watchdog-uart.log) · [通过 UART](../results/4090-c161-lexical-p4-q8-uart.log) · [逐句转写](../results/4090-c161-lexical-p4-q8-fifo-first-command.json)

Mac 研究包共 445,116 B，含声学文件、两词词典、上游许可和逐文件清单。归档 SHA256 为 `39aff8920e9849650399079fbc81804f211025a77b7c68dd4ec796e794bc8366`。均值和方差文件使用本工程的设备读表适配器。[模型归档](../results/4090-c161-lexical-p4-q8-model.tar.gz)

```sh
/usr/bin/python3 "$app/pack_gaussians.py" --bits 8
PYTHONPATH=research/wakeword_p4/experiments:.local/wakeword/c161-pocket-reference/site \
  /usr/bin/python3 "$app/prepare_packed.py"
sed 's/partitions.csv/partitions-packed.csv/g' "$work/sdkconfig" > "$work/q8-sdkconfig"
idf.py -C "$app" -B "$work/build-q8" -DNYX_LEXICAL_Q8=ON -DIDF_TARGET=esp32p4 \
  -DSDKCONFIG="$work/q8-sdkconfig" "-DSDKCONFIG_DEFAULTS=$app/sdkconfig.defaults;$work/qemu.defaults" reconfigure
cmake --build "$work/build-q8" --parallel 2
/usr/bin/python3 "$app/run_qemu.py" --q8
```
