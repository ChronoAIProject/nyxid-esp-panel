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
