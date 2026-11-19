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

## 在训练语音上适配短语声学模型

训练基于已经交给 Mac 的 Q8 模型。PocketSphinx 5.0.4 在每条有词尾标记的训练前缀上对齐 “hey nicks”，按线上 480 样本一块读取特征。第一组在 RTX 4090 上用最大后验估计更新六个音素、18 个上下文状态的高斯混合权重。231 条正训练来源中，182 条进入训练，共 12,246 帧，3 个前缀重复，45 条对齐失败，1 条音素不完整。第二组从该结果出发，用 182 个正前缀和 194 个姓名称呼负前缀进行 100 步判别训练，仅更新第二个词的 12 个状态。两个模型均只替换 55,936 B 的 am/sendump，使用原词典、Q8 高斯参数、42 个基础音素竞争项及 1e-15 阈值。模型与词典合计仍为 455,703 B。[正例适配训练记录](../results/4090-c161-q8-keyword-map-fit.json) · [姓名对比训练记录](../results/4090-c161-q8-name-contrast-fit.json)

| 已消费的开发语料，双速提议加确认 | 原 Q8 | 正例适配 | 再加姓名对比 |
| --- | ---: | ---: | ---: |
| 既有连续对话的目标命令唤醒 | 20/36 | 36/36 | 22/36 |
| 同批对话的日常语句误唤 | 0/108 | 1/108 | 0/108 |
| 独立日常短句误唤 | 0/96 | 2/96 | 0/96 |
| 普通 Bot 回复误唤 | 0/48 | 0/48 | 0/48 |
| Bot 主动示范唤醒词时通知 | 4/8 | 4/8 | 4/8 |
| 四声音安静时的完整命令，半量回声抑制 | 6/8 | 8/8 | 7/8 |
| 四声音在 Bot 播报时的完整命令，半量回声抑制 | 3/8 | 7/8 | 7/8 |
| 同批四声音的干扰语流有通知，半量回声抑制 | 0/32 | 1/32 | 0/32 |

正例适配在安静的 “Hey Max” 上误唤一次；独立日常句的两次是 “Hey Nick, can you help me?” 和 “Hey Chris, how was your day?”。姓名对比版本的四声音安静与插话条件均仍漏 af_jessica 的亮度命令。该版本只开原速提议，在这批四声音上同样各完整 7/8；在更早的两声音、两房间插话组，原速提议仍为 0/4，双速提议半量混合为 2/4。两种模型均核对 625 条开发语流及 66 段已有背景的提议。约 28 小时背景均无通知。[正例适配逐项结果](../results/4090-c161-q8-keyword-map-evaluation.json) · [姓名对比逐项结果](../results/4090-c161-q8-name-contrast-evaluation.json)

两张权重表分别放入含三份现有音乐的 2 MiB SPIFFS，并使用同一 Q8 ELF 启动 P4 QEMU。两次各有 62/62 个窗口与主机的短语返回时钟和确认结果吻合，28 个背景窗口均无通知。就绪外部堆增量均为 2,825,128 B。正例适配及姓名对比的初始化分别为 6.08 和 7.04 秒。按单个解码器排队计入实际块耗时，两秒回看后，旧四声音分别完整 11/12、11/12，新对话分别 8/8、7/8。两种实现成功窗口从提议到通知的延迟中位数分别为 345 和 350 ms，最长 751 和 787 ms。真实设备调度与首句仍待整机回报。[正例适配 P4 排队及录音](../results/4090-c161-q8-keyword-map-p4-fifo-first-command.json) · [姓名对比 P4 排队及录音](../results/4090-c161-q8-name-contrast-p4-fifo-first-command.json)

Mac 可从原 Q8 模型文件生成研究测试卷。在单独的测试副本中替换 am/sendump。正例适配文件 SHA256 为 `884010cf5ff555a90a110d06809c8be8c33631453fc98814a8a36a3da6679021`，姓名对比文件为 `314e8b0713609218136916902810463a8a1048a0bfa7b34b93d0db492d8efe14`。常规实机基准仍为原 C161。这两张表的训练只使用既有训练来源；这些测试结果来自合成开发语料。[正例适配权重](../results/4090-c161-q8-keyword-map.sendump) · [姓名对比权重](../results/4090-c161-q8-name-contrast.sendump)

## 换用 Piper 后的发音与连接命令

冻结上述两份权重后，在 RTX 4090 上用 Piper 的两个 OpenSLR83 英国开发声音生成了四段安静对话。每个声音分别说同样的查列车和卧室音箱命令，依次测试三种目标拼写、Hey Willow、Nick／Max／Chris 三个姓名和日常句。顺序按已存种子打乱。四段共 36 句，音量取合成器原值，重采样后削波为零。原始采样率浮点音频、实际音素和四段完整 16 kHz PCM 均可回放。姓名和日常句合计 20 条。[输入及时间轴](../results/4090-c161-q8-piper-spellings-plan.json) · [原音频归档](../results/4090-c161-q8-piper-spellings-pcm.zip)

| 安静条件，每格为唤醒数 / 4 | C161 | 原 Q8 双速 | 正例适配双速 | 姓名对比双速 |
| --- | ---: | ---: | ---: | ---: |
| Hye Nyx，首词 /haɪ/ | 0/4 | 0/4 | 0/4 | 0/4 |
| Hey Nyx，首词 /heɪ/ | 1/4 | 3/4 | 3/4 | 3/4 |
| Hey Nicks，首词 /heɪ/ | 1/4 | 3/4 | 4/4 | 4/4 |

Piper 对 Hey Nyx 和 Hey Nicks 给出相同音素，但两种拼写分别合成，均保留了各自的波形。两份适配权重在 /heɪ/ 组均唤醒 7/8，两秒回看后的固定 Whisper 转写均完整 6/8。原 Q8 双速为 6/8 唤醒、5/8 完整，C161 为 2/8、1/8。三种确认器仅接原速提议时均为 3/8、2/8。两份适配的姓名及日常句均误唤 0/20，对 Hey Willow 也均为 0/4。四次 /haɪ/ 及漏掉的一次 /heɪ/ 均未产生任何速度的前级提议。两种权重在已有提议的其余七次 /heɪ/ 上均有通知。语句外全配置零通知。[逐句提议、短语匹配和录音转写](../results/4090-c161-q8-piper-spellings-evaluation.json)

官方 Hey Willow DEFAULT 在同一批 PCM 上经经典 ESP32 QEMU 完整消费四段，自己的命令为 0/4，交叉及日常语句也没有通知。此前 Kokoro 的四声音亮度命令在安静和播报时各 4/4。两批合成器、声音与载句均有变化，四条样本不足以比较总体水平。开发声音来自 Google OpenSLR83、Demirsahin 等（LREC 2020），通过 rhasspy/piper-voices 的英国南部女声和北部男声模型生成，源数据采用 CC BY-SA 4.0。[官方逐块返回及串口记录](../results/4090-c161-q8-piper-spellings-willow.zip) · [声音来源和版本](../results/4090-piper-slr83-development-source-v1.json) · [同条件 Kokoro 结果](../speex_echo_p4/README.md)

在同一批输入上另做了语音起点诊断。用 30 ms WebRTC VAD 帧在至少 450 ms 安静后启动一个短语搜索窗口，取前 300 ms 和后 1.5 s。给姓名对比模型加上 /haɪ/ 所需的 6 个上下文状态，沿用原 144 个状态的权重和相同阈值。把这一路的实际音频返回时刻与原双速链路合并。36 句 Piper 产生 36 个窗口，共送入解码器 65.88 秒音频。下表统计各语句有无通知，宿主回放没有计入 P4 任务排队。

| 同一语流的通知数 / 语句数 | 姓名对比双速 | 再从语音起点查 /heɪ/ | 再从语音起点查 /heɪ/ 和 /haɪ/ |
| --- | ---: | ---: | ---: |
| Piper 首词 /heɪ/ | 7/8 | 8/8 | 8/8 |
| Piper 首词 /haɪ/ | 0/4 | 0/4 | 3/4 |
| Piper 称呼和日常句误唤 | 0/20 | 2/20 | 2/20 |
| 既有连续命令 | 22/36 | 22/36 | 22/36 |
| 同批常见语句误唤 | 0/108 | 0/108 | 0/108 |
| 独立日常短句误唤 | 0/96 | 1/96 | 2/96 |
| 普通 Bot 回复误唤 | 0/48 | 2/48 | 2/48 |
| Bot 示范唤醒词时通知 | 4/8 | 4/8 | 4/8 |

两种额外搜索均误唤了英国南部合成女声的 Hey Nick 和 Hey Max。双发音搜索也误唤 “Do not mix those colors.”、“This is a nice mix of songs.”，以及 Bot 播报 “Nick sent you a message about dinner.” 的数字和反射两个版本。此次额外路径只作主机诊断，原确认器的 28 小时背景结果只适用于原提议策略。[逐帧候选、匹配音素短语及各句结果](../results/4090-c161-q8-vad-pronunciations.json)

### 用训练声音复核额外的短语匹配

另在 RTX 4090 上用 12 个已登记的 Kokoro 训练声音生成 384 句。两种首词配 Nyx 为正例，Nick、Mick、Max、Chris 等称呼和普通词为负例，每种称呼都用同一对载句。用合成时的音素边界构造两种音量和有起止扰动的 9,216 个训练窗口。在原有 20 频带前端上训练 13,097 参数的小卷积模型 500 步，固定以 logit ≥ 0 通过。只检查语音起点额外产生的短语匹配。原姓名对比双速的通知仍直接通过。[训练来源与词边界](../results/4090-c161-vad-word-check-sources.json) · [训练记录](../results/4090-c161-vad-word-check-training.json)

模型冻结后，再用同两个 Piper 声音生成 44 条新语句，包含两条新命令、Hi Nick、Hey Nick、Hey Mick、Hi Max、Hi Chris、Hi Nikki、Hey Willow 和日常句。加入复核后的 /haɪ/ 及 /heɪ/ 均唤醒 4/4，两秒回看后的完整命令分别为 3/4、4/4。原姓名对比双速合计 5/8 唤醒、3/8 完整。新复核模型在 32 条称呼及日常干扰中误唤 8 条，原姓名对比双速为 2 条。从语音起点直接搜两种发音、跳过神经网络复核时为 9 条。三个配置对同批 4 条 Hey Willow 均无通知。

复核后仍误唤 Hey Nick 3/4、Hi Nick 2/4、Hey Mick 2/4，以及 1/8 日常句。此前那批 Piper 的两个姓名误唤和既有 96 条日常短句的两次误唤均消失，Bot 普通回复仍有 2/48。两批均只作开发对照，新增路径当前只在主机验证；普通语音的实机验证使用冻结 C161，固定音使用 v5。[新音频与输入清单](../results/4090-c161-vad-word-check-fresh-plan.json) · [逐匹配得分、误唤及首句转写](../results/4090-c161-vad-word-check-evaluation.json)

### Melo 连说、姓名与命令首音素

RTX 4090 上用 Melo 的 EN-US、EN-BR 各生成同两条命令，分别用逗号连接和直接连说。每对输入复用语句顺序和逐句随机种子，保留合成的实际音素、时长、原速 16 kHz 音频和逐句哈希。共 96 句，其中 32 条 Nyx 目标、40 条常见姓名、16 条日常句和 8 条 Hey Willow。日常句在逗号配对间重复。Nyx 主机回放在 2 秒回看录音上用同一 Whisper 检查完整命令。官方 Hey Willow DEFAULT 在经典 ESP32 QEMU 完整消费相同的八段语流。[输入和音素](../results/4090-c161-melo-address-grammar-plan.json) · [八段 WAV](../results/4090-c161-melo-address-grammar-wav.zip) · [Willow 报告](../results/4090-c161-melo-address-grammar-willow.json) · [串口及逐块返回](../results/4090-c161-melo-address-grammar-willow.zip)

在短语搜索产生候选时，用完整英文 PocketSphinx 声学模型对已到达的词段同时识别 hey/hi nicks、Nick、Mick、Max 等。下表每格为唤醒数、首句完整数，分母为该行语句数。辅助路径保留原 C161 的通知，只对语音起点额外产生的候选启用姓名语法。首词分组取自 Melo 的 hh ey 或 hh ay。[逐句事件和转写](../results/4090-c161-melo-address-grammar.json)

| Melo 目标语句 | 句数 | C161 唤醒、首句完整 | 加辅助路径后唤醒、首句完整 |
| --- | ---: | ---: | ---: |
| /heɪ/，带逗号 | 8 | 4、3 | 5、4 |
| /heɪ/，直接连说 | 8 | 1、0 | 4、3 |
| /haɪ/，带逗号 | 8 | 4、3 | 6、4 |
| /haɪ/，直接连说 | 8 | 5、2 | 5、2 |

两种发音合计，辅助路径为 20/32 唤醒、13/32 首句完整，C161 为 14/32、8/32。EN-BR 接 open 的八条目标两条路径均漏。对各自字面称呼的八条命令，辅助路径 Hey Nyx 为 5/8、3/8，官方 Hey Willow 为 2/8、1/8。双方均未对彼此的八条命令唤醒。此前两批 Piper，辅助路径分别为 11/12、10/12 和 7/8、5/8；姓名及日常句误唤分别为 0/20、5/32。已有连续命令唤醒 22/36，同批日常句 0/108、独立日常句 0/96、普通 Bot 回复 0/48，Bot 主动示范唤醒词时通知 4/8。将语法扩大到全部辅助候选后，连续命令唤醒降至 12/36。[Piper 和 Bot 明细](../results/4090-c161-address-grammar.json)

Melo 的 40 条姓名中辅助路径误唤 4 条，均为不带逗号接 switch。其余 16 条日常句均无误唤。用合成器的音素对齐估计，在原采样位置遮盖命令首音素，并另做等长后移 250 ms 和整条命令遮盖。逐句从语流开头重放，72 条原样对照均复现四条路径的原始通知采样点。遮住 /s/ 后 EN-US 的 Hey Mick 误唤消失，后移对照仍触发；另外三次 Nick 在原 C161 就触发，分别遮住 /s/ 或整条命令仍会触发。20 条原本命中的目标在三种干预下都保留。整条命令静音还使原先未命中的 3 条目标、2 条姓名产生通知。遮盖会改变停顿和检测行为，从这次结果只能把 Hey Mick 的一次误唤定位到该 /s/ 时段。此姓名语法只做过主机回放。[遮盖代码](../experiments/check_c161_melo_command_boundary.py) · [72 条逐句结果](../results/4090-c161-melo-address-grammar-boundary.json)

### 连接命令的配对训练

RTX 4090 使用原速语流中实际检测到的语音起点训练四类卷积分类器，分别识别 Nyx、Nick、Mick 和其他。12 个 Kokoro 训练声音各合成 40 句，三种称呼在每个声音、命令、语速及标点条件下配对，另有 171 条已登记的 Chatterbox 日常训练负例。共 651 条合成语句、1,302 个音量窗口。每次在检测到讲话 1.32 秒后，对过去 144 帧的 40 维原生声学特征作一次判断。训练 650 步，54,388 个参数。开发回放前冻结权重，固定 Nyx 后验概率 ≥ 0.5，再与原 C161 通知按同一冷却时间合并。[训练语句与音素](../results/4090-c161-onset-names-sources.json) · [训练结果](../results/4090-c161-onset-names-fit.json) · [权重](../results/4090-c161-onset-names-weights.npz)

冻结后，在已有 Melo EN-US、EN-BR 声音上另合成 save 和 turn 开头的两条命令。32 条目标、40 条姓名、16 条日常语句播放和 8 条 Hey Willow。下表的完整首句率以全部目标为分母。两秒回看采用同一 Whisper small.en。[新输入、实际音素](../results/4090-c161-onset-names-melo-plan.json) · [原始语流 WAV](../results/4090-c161-onset-names-melo-wav.zip)

| 新命令，每条路径消费相同 PCM | 唤醒 /32 | 完整命令 /32 | 姓名误唤 /40 | 日常句误唤 /16 |
| --- | ---: | ---: | ---: | ---: |
| C161 | 12 | 9 | 1 | 0 |
| C161 加新训练的分类器 | 14 | 11 | 3 | 4 |
| C161 加此前的短语搜索及姓名语法 | 18 | 16 | 2 | 0 |

只看无逗号直接连说，三条路径对于 /heɪ/ 分别唤醒 0、1、3 条，首句分别完整 0、1、3 条，分母均为 8。对 /haɪ/ 各唤醒 5/8，首句分别完整 3、3、4 条。新分类器的 4 次日常误唤来自同一 EN-US 声音的两份波形，各在标点配对中播放两次，原句为 “Nick saved the last photo.” 和 “Can you see my shopping list?”。另在此前 96 条日常短句中误唤 “Nick, please turn on the light.”。普通 Bot 回复 0/48，Bot 示范唤醒词时 4/8；较早的 EN-BR 接 open 的八次漏唤仍全部漏掉。两批 Piper 合并该分类器后，分别唤醒 5/12、3/8，首句完整 4/12、1/8。[分类器逐窗口概率和转写](../results/4090-c161-onset-names-evaluation.json) · [原姓名语法在新载句上的复测](../results/4090-c161-onset-names-melo-grammar.json)

官方 Hey Willow DEFAULT 在新载句八条自己的命令上唤醒 2 条、首句完整 2 条，均为 EN-US 带逗号，其他语句无事件。相同八种条件下，字面 Hey Nyx 的 C161 唤醒且完整 1 条，加新分类器 2 条，加此前姓名语法 4 条。以上都是合成开发对照。新增分类器的普通日常语句误唤及召回结果不足以支持实机替换 C161。[经典 ESP32 QEMU 报告](../results/4090-c161-onset-names-melo-willow.json) · [原始 UART 和逐块返回](../results/4090-c161-onset-names-melo-willow.zip)

### 姓名语法的 P4 词段回放

以编译出的有限状态语法图枚举每个词可能的左右音素，沿空转移传递上下文，再用 PocketSphinx 原生回退查询选取状态。保留 42 个基础音素、61 个三音素和 270 个共享状态。127 段已消费开发语流提供了 117 个不同的候选窗口。缩小模型后使用原浮点高斯表时，117 个窗口的文字和词边界全部与完整英文模型吻合，66 个分数有差异。换为既有 Q8 高斯表后，1 个窗口从 hey nicks 变成 hey mix，7 个窗口的分词时间有变化。该词段所属的连续命令仍会唤醒，主机通知晚 120 ms。[状态、文件及语法图哈希](../results/4090-c161-address-grammar-graph-inputs.json) · [逐窗口及完整语流比较](../results/4090-c161-address-grammar-graph-windows.json)

声学文件及词典、语法合计 506,138 B。整套文件与原有三首音乐放入面板配置的 2 MiB SPIFFS 时，实际制卷报空间不足。自测把 104,320 B 的 am/sendump 放入应用映像，其余 401,818 B 和现有音乐放入 SPIFFS。P4 QEMU 启动后核对三首音乐的长度及 CRC，117 个词段的文字与逐词起止帧全部与独立 Q8 主机一致。自测应用 678,192 B，语法初始化 5.57 s，就绪占用外部堆 3,050,712 B。启动前外部堆至回放最低余量之差为 3,237,196 B。各词段从复位特征到识别完成耗时中位数 174.78 ms，范围 96.37–245.64 ms。首次将双核模拟器固定在宿主单个逻辑 CPU 上时触发空闲任务看门狗，给同一映像三个宿主逻辑 CPU 后完整回放通过。[设备测量](../results/4090-c161-address-grammar-p4.json) · [两次原始 UART](../results/4090-c161-address-grammar-p4-uart.zip)

按每个候选实际可用的样本时钟，把 125 次语音起点短语匹配连同重复匹配送入单工作任务队列，逐项计入该词段对应的模拟设备耗时。供给候选至结束的中位数为 189.25 ms，最长 673.63 ms。下表采用两秒回看并在 RTX 4090 上重新转写。队列只计姓名语法这一步，前端、前级提议、其他面板任务和网络耗时仍需整机测量。[队列、通知和逐句转写](../results/4090-c161-address-grammar-p4-fifo.json)

| 合成载句 | 目标数 | 唤醒 | 主机原语法首句完整 | 计入 P4 语法排队后首句完整 |
| --- | ---: | ---: | ---: | ---: |
| 首批 Piper | 12 | 11 | 10 | 10 |
| 次批 Piper | 8 | 7 | 5 | 4 |
| Melo，switch / open | 32 | 20 | 13 | 13 |
| Melo，save / turn | 32 | 18 | 16 | 15 |

研究包 489,617 B，SHA256 为 `6e16686d21664914155d09221ae025586b07ed0bd94a0e81717eabe71143edaf`，内含声学文件、词典、语法、许可和逐文件清单。普通语音整机使用的冻结 C161 和固定播放音 v5 均已有独立交接。[姓名语法研究包](../results/4090-c161-address-grammar-graph-q8.tar.gz)

在已准备好上述 Q8 自测环境的研究工作树，从仓库根目录生成独立探针。主机生成脚本依赖本页前文的已锁定词典、语流、模型和开发输入。

```sh
export PYTHONPATH="research/wakeword_p4/experiments:.local/wakeword/c161-pocket-reference/site"
python3 "$app/compact_addresses.py" --graph
python3 "$app/check_address_windows.py"
cp "$work/q8-sdkconfig" "$work/addresses-graph/sdkconfig"
idf.py -C "$app" -B "$work/build-addresses" -DNYX_LEXICAL_Q8=ON -DNYX_LEXICAL_ADDRESSES=ON \
  -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/addresses-graph/sdkconfig" \
  "-DSDKCONFIG_DEFAULTS=$app/sdkconfig.defaults;$work/qemu.defaults" reconfigure
cmake --build "$work/build-addresses" --parallel 2
python3 "$app/run_address_qemu.py"
```

### 从语音起点同时查称呼和命令首词

直接以 WebRTC VAD 的讲话起点启动完整英文 PocketSphinx，在最多 2.10 秒内持续比较 Nyx、Nick、Mick、next、Willow 和常用语句。每 30 ms 更新，比较连续三个返回保持同一称呼、或保持同一称呼及命令首词的两种通知条件。后一条件用预先列出的 36 个命令首词。每条辅助方案都与冻结 C161 合并，使用同一冷却与两秒回看。这是主机音频时钟实验。[规则和词典](../experiments/check_c161_address_verb.py) · [已消费数据的候选变化、通知及首句](../results/4090-c161-address-verb.json)

在 EN-BR 的八条 “open the garage window” 目标中，VAD 八次均启动。原短语搜索八次均未匹配；其中三条带逗号语句有前级双速提议。直接解码也八次均未唤醒，搜索窗口末尾把五个第二词读作 next、三个读作 nick。固定 Whisper 对原始完整输入的八条中有六条转写以 “Hey Nix” 或 “Hi Nix” 开头。后一项使用了完整输入。[真实候选与遮盖对照](../results/4090-c161-melo-address-grammar-boundary.json) · [原始音素与语流](../results/4090-c161-melo-address-grammar-plan.json)

冻结规则后，RTX 4090 使用相同两种 Melo 声音生成 “open the kitchen blinds” 和 “unlock the garden gate”，每种称呼配带逗号及直接连说。open 在词典内，unlock 在词典外。32 条目标的唤醒数、首句完整数，原 C161 为 9、8，原姓名语法为 17、15，新语法确认称呼为 18、17，要求同时出现命令首词为 18、16。各方案对这批 40 条姓名、16 条日常句及 8 条 Hey Willow 均无通知。要求首词的辅助路在目标中独立命中 11 条，其中四条说 open 却匹配 stop、close 或 call，七条说 unlock 时匹配其他词。实际首词未被正确识别，不能用这一步解释为命令语义得到确认。

无逗号 /heɪ/ 与 /haɪ/ 各有八条，要求首词的合并路径分别唤醒 4、4 条，首句完整 4、2 条。字面 Hey Nyx 为 4/8 唤醒且完整。官方 Hey Willow DEFAULT 在自己的八条命令中为 2/8 唤醒且完整，均为 EN-US 带逗号。此前的 108 条连续普通句在新首词路径误唤 1 条，96 条独立日常句及普通 Bot 回复 0/48 均无误唤，主动示范 4/8。[新音频与音素](../results/4090-c161-address-verb-challenge-plan.json) · [逐句解码与首句转写](../results/4090-c161-address-verb-challenge.json) · [原姓名语法对照](../results/4090-c161-address-verb-challenge-grammar.json) · [官方 Willow](../results/4090-c161-address-verb-challenge-willow.json) · [原始串口与逐块返回](../results/4090-c161-address-verb-challenge-willow.zip)

### 用预训练语音特征区分 Nyx、Nick、Next 和 Mick

在 RTX 4090 上对齐三种分类器的输入时刻。真实 VAD 起点后 1.32 秒取已收到的 1.44 秒音频，固定以 Nyx 概率 ≥ 0.5 通知。训练仅使用 4,320 条既有 Piper、576 条 Kokoro 和 171 条 Chatterbox 合成训练语句，合计 5,067 个不同波形窗口。五类为 Nyx、Nick、Next、Mick、其他。第一种用冻结 wav2vec2-base-960h 的第六层在六个时间区间的特征训练分类器。另外两种小卷积网络均使用原生 40 维前端、相同初始化、相同训练批次和 1,000 次更新，其中一种额外学习预训练模型逐帧的字符概率。[训练来源、预训练文件身份](../results/4090-c161-onset-letters-sources.json) · [训练记录和权重哈希](../results/4090-c161-onset-letters-fit.json)

冻结权重后，继续使用 EN-US 和 EN-BR 合成两条新命令。104 句中有 32 个目标、48 个近音称呼（包括 16 次 Hey Next 或 Hi Next）、16 个日常句和 8 个 Hey Willow。下表均在同一完整语流上保留 C161 的通知并使用两秒回看。普通语音的实际设备仍使用 C161。

| 新载句 dim / move | 唤醒 /32 | 完整首句 /32 | 称呼误唤 /48 | 日常句误唤 /16 |
| --- | ---: | ---: | ---: | ---: |
| C161 | 10 | 4 | 5 | 0 |
| 加此前词段姓名语法，主机音频时钟 | 17 | 6 | 8 | 0 |
| 加预训练语音特征分类器，RTX 4090 | 27 | 14 | 6 | 0 |
| 加小卷积分类器 | 12 | 5 | 5 | 0 |
| 加逐帧字符监督的小卷积分类器 | 10 | 4 | 5 | 0 |

预训练路径补回旧 EN-BR 接 open 的 4/8 次唤醒，完整命令 2/8；新 EN-BR 接 move 则由 0/8 提到 6/8，完整 4/8。新载句直接连说 /heɪ/ 为 8/8 唤醒、4/8 完整，/haɪ/ 为 6/8、2/8。新组相对 C161 新增的一个误唤为 Hey Next；此前 save / turn 组另有日常句误唤 1/16。已有连续对话目标 21/36，普通句 0/108，独立日常句 0/96，普通 Bot 回复 0/48，Bot 明确示范 4/8。Piper 两批分别唤醒 7/12、5/8，首句完整 6/12、3/8。该路径依赖主机预训练声学编码器；两种小模型均未复现其跨口音增益。

字面 Hey Nyx 在八条自己的命令中，预训练路径唤醒 7 条，首句完整 4 条。官方 Hey Willow DEFAULT 在相同八种条件下唤醒 2 条、完整 2 条，均为 EN-US 带逗号。八次官方回放全部消费对应 PCM。[原始语流及音素](../results/4090-c161-onset-letters-challenge-plan.json) · [五类概率、通知和转写](../results/4090-c161-onset-letters-evaluation.json) · [官方 Willow](../results/4090-c161-onset-letters-challenge-willow.json) · [UART 与逐块返回](../results/4090-c161-onset-letters-challenge-willow.zip)

### 首句录音起点与命令首音

固定上表的 C161 加预训练分类器通知，在同样的四批 Melo 开发音频上检查 128 条目标。105 条唤醒后的标准两秒回看都包含合成器标注的命令首音，录音起点至少提前 1.14 秒。对照中，整句使用原 WAV 的语句边界，其他三种录音统一结束于语句尾后 400 ms。报告按原协议判断 Whisper 转写结尾能否逐词匹配整个命令。

| 载句，每组 32 条目标 | 已唤醒 | 已唤醒的完整原句转写正确 | 标准回看正确 | 按标注提前 120 ms 裁切正确 | 通知时按 CTC 字符时间裁切正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| switch / open | 27 | 16 | 17 | 23 | 19 |
| save / turn | 28 | 25 | 22 | 28 | 23 |
| open / unlock | 23 | 20 | 19 | 17 | 17 |
| dim / move | 27 | 14 | 14 | 13 | 9 |

标注来自 Melo 的 VITS 音素时长，仅用于离线定位。CTC 方案只读通知时已经到达的 1.44 秒音频，按识别到的称呼和下一词的时间戳裁切。105 次命中中 42 次匹配称呼，其余 63 次采用标准回看。该方案在四组合计恢复 4 句、损失 8 句，全部保留首音。119 次通知（含干扰）的额外主机处理时间中位数 4.98 ms、最大 179.96 ms，无额外音频等待。dim / move 的误转写包括 move 变为 moved，meeting 变为 meaning，dim 变为 turn。实机验证继续保留标准回看，并记录首音进入录音和完整命令转写两项。[逐句音素边界、录音范围和转写](../results/4090-c161-command-start.json) · [复现实验](../experiments/check_c161_command_start.py)

### 用五类称呼概率蒸馏端侧分类器

在 RTX 4090 上复用 5,067 个训练窗口和原小卷积架构。第一种小模型学习大模型的五类概率，第二种另学习六个时间区间对每一类分数的贡献。两个模型的初始化、1,000 批数据和唤醒阈值一致。冻结后合成 “close the bedroom door” 和 “read my next appointment”，使用原 EN-US、EN-BR 开发声线，共 104 句。[权重与训练记录](../results/4090-c161-onset-evidence-fit.json) · [新语流及音素](../results/4090-c161-onset-evidence-plan.json)

| close / read 新载句 | 唤醒 /32 | 完整首句 /32 | 近音称呼误唤 /48 | 日常句误唤 /16 |
| --- | ---: | ---: | ---: | ---: |
| 冻结 C161 | 14 | 10 | 6 | 0 |
| C161 加词段姓名语法 | 22 | 14 | 9 | 2 |
| C161 加预训练第六层分类器 | 26 | 20 | 8 | 0 |
| C161 加原小卷积 | 17 | 12 | 8 | 0 |
| C161 加五类概率蒸馏 | 17 | 12 | 6 | 0 |
| C161 加五类概率及六个时间区间蒸馏 | 17 | 12 | 7 | 0 |

新组五类概率蒸馏的六次近音误唤与 C161 相同，Hi Next 四次、Hi Nikki 两次。无逗号 /heɪ/ 的八条中，蒸馏小模型唤醒 4 条、首句完整 3 条，C161 为 1、1。无逗号 /haɪ/ 两者均为 3、2。EN-BR 接 read 时小模型与 C161 均为 2/8 唤醒和 2/8 完整，预训练大模型为 7/8 和 7/8。此前 96 条日常句，原小卷积误唤 3 条、五类概率蒸馏 0 条、再加时间区间蒸馏 1 条。三种小模型均在 108 条连续普通语句和 48 条普通 Bot 回复中零误唤。[所有通知与逐句转写](../results/4090-c161-onset-evidence-evaluation.json)

字面 Hey Nyx 的八条中，C161 为 1/8 唤醒和 0/8 完整，五类概率蒸馏为 3/8 和 1/8，预训练大模型为 5/8 和 4/8。官方 Hey Willow 在自己的八条命令中为 2/8 和 1/8。两条命中均为 EN-US 带逗号。[官方识别与首句](../results/4090-c161-onset-evidence-willow.json) · [八份完整串口和逐块返回](../results/4090-c161-onset-evidence-willow.zip)
