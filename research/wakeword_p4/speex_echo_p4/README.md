# P4 上的 SpeexDSP 回声抵消和固定唤醒音

独立 ESP-IDF 5.5.4 应用使用 SpeexDSP 1.2.1，同步输入麦克风与扬声器参考，每次处理 16 kHz 单声道的 160 样本，滤波器长度 2048 样本。每个 10 秒输入独立启动 ESP32-P4 QEMU，输出完整 PCM、每帧处理用时和内部堆读数。输入来自两组 Bot 与外部合成声音，以及两份实测房间响应。[24 个普通语音用例和原始配置](../experiments/c161_lexical_echo.md)

| 24 个模拟输入，各 1,000 帧 | 定点、KISS FFT | 浮点、SmallFT |
| --- | ---: | ---: |
| 初始化滤波器的内部堆增量 | 52,356 B | 73,928 B |
| 每个 10 ms 块的计算时间中位数 | 0.295 ms | 0.703 ms |
| 最慢计算块 | 3.219 ms | 4.153 ms |
| 单独设备示范，误唤 /4 | 1 | 0 |
| 相同麦克风波形作为外部播放，唤醒 /4 | 2 | 2 |
| 外部独讲，完整命令 /4 | 4 | 4 |
| Bot 普通回复期间外部插话，完整命令 /4 | 0 | 0 |
| 双方同时念唤醒词、后接不同命令，外部完整命令 /4 | 2 | 4 |

普通语音确认使用主机的冻结 C161、双速提议和 Q8 确认器，处理实际 P4 输出 PCM。完整命令计入唤醒及 4090 Whisper 两秒回看转写。定点版的 3,840,000 个输出样本逐个等于同源码主机构建。浮点版有 77,740 个样本不同，最大绝对差为 46 个 PCM16 单位。两种实现均将全部输出和 UART 存档。[定点测量](../results/4090-c161-speex-p4.json) · [定点 UART](../results/4090-c161-speex-p4-uart.zip) · [浮点测量](../results/4090-c161-speex-p4-float.json) · [浮点 UART](../results/4090-c161-speex-p4-float-uart.zip)

## 普通语音在面板播报时插话

在上述 24 个普通语音回放中，对浮点滤波器连接上游 SpeexDSP 1.2.1 的残余回声后处理，参数使用默认值，噪声抑制 −15 dB、回声抑制 −40 dB、近端说话时 −15 dB，AGC 关闭。其重叠相加缓冲为一块 160 个样本。四条外部独讲的输出相对原麦克风，互相关峰都在 159 个样本。两个滤波器均在 P4 QEMU 执行。识别和 2 秒回看用主机读取实际设备输出，保留原 C161、双速提议和 Q8 确认的阈值。[后处理、通知和逐句转写](../results/4090-c161-speex-residual.json) · [24 份原始 UART](../results/4090-c161-speex-residual-uart.zip)

| 每组四次 | 只做线性抵消 | 加默认后处理 | 两种 PCM 对齐后各占一半 |
| --- | ---: | ---: | ---: |
| 面板普通回复中插话，完整外部命令 | 0 | 2 | 2 |
| 双方同时说唤醒词，后接不同命令，完整外部命令 | 4 | 0 | 4 |
| 外部独讲，完整命令 | 4 | 4 | 4 |
| 单独面板示范，误唤 | 0 | 0 | 0 |
| 单独面板普通回复，误唤 | 0 | 0 | 0 |
| 同一面板示范波形作为外部输入，唤醒 | 2 | 2 | 2 |

默认后处理恢复了 af_aoede 在两个房间的插话，半量混合恢复的是 bf_lily。半量使用一块延迟对齐线性输出后逐样本取平均；同一条识别链处理混音。若让原线性流与全量后处理各自识别，按实际音频可用时刻和原冷却窗口合并通知，插话仍只保留 af_aoede 的 2/4，双方同时念词保留 4/4。此项主机合并没有计入两条分类器及短语确认的设备排队耗时。结果涉及两个声音、每声同一句在两间房重放。[混合、双路通知与命令](../results/4090-c161-speex-residual-mix.json)

按合成输入已知的请求开始位置离线截取，原线性输出中的四条外部命令均可完整转写。实际监听在这四条均未提出唤醒候选。再分别交换“外部独讲”和“面板播报中插话”的请求前历史与请求后 PCM，保持原先监听状态连续：换成安静历史后 3/4 可唤醒，只换成干净请求后 2/4 可唤醒，两个声音对历史与请求内残余回声的敏感性不同。该切换点来自合成标注，不能用来计算实时命中。[交叉拼接及四条已知区间转写](../results/4090-c161-speex-history-splice.json)

带后处理的 P4 初始化内部堆增量为 98,088 B，相对单独线性抵消增加 24,160 B；24,000 个块中位 1.524 ms，首条自播输入的第一个块耗时 11.503 ms，其余均在 10 ms 内。该计时只覆盖音频处理。复现时在纯浮点配置上增加 `-DNYX_SPEEX_PREPROCESS=ON`，构建目录 build-post，先运行 check_c161_speex_residual.py，再运行 check_c161_speex_residual_mix.py。check_c161_speex_history_splice.py 可直接读取原线性回放归档。

## 已交付 v5 固定音

两个房间各有四个 10 秒用例。面板命令为 “Play the evening news.”，外部命令为 “Turn off the desk lamp.”，双方使用相同的 v5 唤醒文件。对 8 个双通道输入，P4 同时运行浮点 SpeexDSP 和原 v5 匹配器。滤波器输出与前一次纯滤波器回放逐样本相同，设备上三个通知的样本位置也均与从同一 PCM 得到的主机结果吻合。

| 场景 | P4 通知 | 两秒回看后正确转写外部命令 |
| --- | ---: | ---: |
| 仅面板播放 v5 | 0/2 | 无外部命令 |
| 仅面板普通回复 | 0/2 | 无外部命令 |
| 面板普通回复期间外部播放 v5 | 2/2 | 2/2 |
| 双方同时播放相同 v5，后接不同命令 | 1/2 | 1/2 |

同步播放时客厅在样本 97,584 通知，厨房漏检。两组 Bot 普通回复期间的外部播放均在样本 97,592 通知。单独滤波器的首次固定音重放有一帧耗时 12.346 ms（8,000 帧），与另一个模拟回放同时运行。滤波器和匹配器联合重放时 8,000 帧中位 0.758 ms、最大 4.467 ms。计时范围从 DSP 调用到匹配器返回。实际连续采集、扬声器音量、整机调度和最低内部堆由 Mac 板测核对。[首句、最慢帧和原始输出](../results/4090-c161-speex-fixed-playback.json) · [联合运行 UART](../results/4090-c161-speex-fixed-playback-device-uart.zip)

Mac 回放包内 8 个 WAV 均为 16 kHz PCM16，左声道为模拟麦克风，右声道为同步扬声器参考。注入自测应按 160 样本一块成对送入。该双声道文件用于数字输入验证。实机声学播放可使用此前交付的原始单声道 v5 文件。每个 WAV 的 SHA256 和预期通知采样点均在设备报告中。归档 SHA256 为 `ea3ad30ab891e89e0ac767f9f9e1da3e8fb9318204c37abd9ccaee21a615e9e3`。[双通道回放包](../results/4090-c161-speex-fixed-playback-device-stereo.zip) · [设备报告](../results/4090-c161-speex-fixed-playback-device.json)

## 本机正在播放同一个 v5 时

播放调度器知道本机 v5 的起止采样点。在覆盖该音频的处理块中暂停 SpeexDSP 从后台向前台复制系数，避免把与本机完全同步的外部 v5 一并抵消。滤波器继续处理麦克风与参考的原速样本。厨房和客厅的原 8 个回放中，同时播放的两句外部命令均完整，此前厨房漏检。单独面板自播 0/2，外部覆盖普通播报 2/2 条完整命令。没有本机 v5 的四个对照，处理后 PCM 与原实现逐样本相同。[逐次结果](../results/4090-c161-speex-fixed-playback-hold.json) · [UART](../results/4090-c161-speex-fixed-playback-hold-uart.zip)

卧室和办公室分别使用一对独立的实测脉冲响应，代表同类型房间的面板路径和外部路径。每个房间复用同一条合成命令，外部相对开始时间为 −100、0、+150 ms，电平为 0 或 −6 dB。每条流独立启动，上电时匹配器按原协议输入数字零做初始化，回声滤波器从空状态开始。表中“完整”要求目标窗内发生通知，且 RTX 4090 上无提示 Whisper 从两秒回看中逐词转写出外部命令。目标窗为预期通知前后各 100 ms，同时播放时可能覆盖本机声音。[22 条冻结输入](../results/4090-c161-speex-hold-rooms-plan.json) · [44 次 P4 回放](../results/4090-c161-speex-hold-rooms.json) · [UART](../results/4090-c161-speex-hold-rooms-uart.zip)

| 场景 | 标准 SpeexDSP | 暂停前台更新 | 暂停更新并按适应状态屏蔽 |
| --- | ---: | ---: | ---: |
| 先播五秒，双方播放，目标窗通知 / 完整，各 12 次 | 8 / 8 | 12 / 11 | 12 / 11 |
| 先播五秒，仅面板 v5，误唤 | 0/2 | 0/2 | 0/2 |
| 冷启动，仅面板 v5，误唤 | 0/2 | 2/2 | 0/2 |
| 冷启动，双方同步，目标窗通知 / 完整，各 2 次 | 0 / 0 | 2 / 1 | 0 / 0 |

预播五秒后，卧室与办公室的目标窗通知分别从 4/6 到 6/6，完整命令分别从 4/6 到 5/6 和 6/6。卧室的等强同步例将 “desk lamp” 转写为 “desk lamps”。两处冷启动仅面板自播均误唤，所以单凭暂停更新后同步例的通知不能区分来源。两处外部独播均保留完整命令，仅面板普通回复均静默。

回退配置在本机播放 v5 的第一个块处理前读取 SpeexDSP 1.2.1 的 `adapted` 位，并在本次示范期间锁定该决定。尚未适应时，沿用 v5 参考匹配器，仅在参考命中及其后 3,488 个样本内屏蔽麦克风通知。运行时只使用本机调度、同步参考与滤波器状态。原 22 条的滤波输出与只暂停更新的版本逐样本一致。增加的对照中，静音等待五秒仍触发冷启动回退。冷启动外部单独播放为 2/2 条完整命令，面板先自播、外部晚三秒也为 2/2。稳态路径在本机 v5 开始时将麦克风里的本机回声放大两倍、参考维持原幅度，两个房间均误唤。[34 条回放](../results/4090-c161-speex-cold-guard.json) · [输入](../results/4090-c161-speex-cold-guard-plan.json) · [UART](../results/4090-c161-speex-cold-guard-uart.zip)

另用相同麦克风 PCM 对比播放器音量处理前后的参考。在本机 v5 开始时将播放器幅度加倍，再与各自房间响应卷积。取增益前参考时，仅面板自播误唤 2/2，取增益后参考时仍有办公室 1/2 误唤。双方同步时两种输入均有 2/2 次通知，外部完整命令分别为 0/2、1/2。另一个冷启动对照在原 v5 前播半秒已有普通播报，与半秒静音逐项比较。两处的滤波器都尚未适应，外部与面板 v5 同步时均被屏蔽。外部从开机时先播，两种条件各有 2/2 次通知、1/2 条完整命令，办公室把 “desk” 听成 “disk”。[20 条回放](../results/4090-c161-speex-player-gain.json) · [输入](../results/4090-c161-speex-player-gain-plan.json) · [UART](../results/4090-c161-speex-player-gain-uart.zip)

| 模拟计算范围 | 块数 | 计算中位数 | 最慢块 | 超过 10 ms |
| --- | ---: | ---: | ---: | ---: |
| 原厨房、客厅暂停更新 | 8,000 | 0.718 ms | 3.276 ms | 0 |
| 卧室、办公室标准版 | 22,000 | 0.731 ms | 5.585 ms | 0 |
| 卧室、办公室暂停更新 | 22,000 | 0.729 ms | 3.762 ms | 0 |
| 加冷启动屏蔽 | 34,000 | 0.712 ms | 3.994 ms | 0 |
| 半秒预播和音量切换 | 20,000 | 0.731 ms | 4.115 ms | 0 |

Mac 数字注入包内有 11 个双通道 WAV 和 manifest，列出左右声道顺序、每个文件 SHA256、实际本机播放区间、P4 放行与屏蔽的采样点、首句转写。按 16 kHz 每块 160 个样本送入，同一条输入的左右声道共用采样时钟。归档 SHA256 为 `b2a588e057c25b07bc990bbc85cb0e4a7f01a054f35efca69c0c8535ae317abe`。主线 7955dc2b 的监听循环在没有活动通话时把参考清零，活动通话使用同步 I2S 右声道。实机复核时需记录播放状态、实际参考、首句、丢帧与最低内部堆，才能将该分路与整机行为对应起来。[数字注入包](../results/4090-c161-speex-player-gain-stereo.zip) · [主线监听循环](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/7955dc2b/components/panel_services/panel_voice.c#L1095)

## 复现

从 Debian 下载上游 speexdsp_1.2.1.orig.tar.gz，SHA256 为 `8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d`。下列命令在仓库根目录运行，使用已经设置的 ESP-IDF 5.5.4 环境。使用相同配置的 PC 定点或浮点动态库由回放脚本直接从校验过的上游源码构建。[发行源码](https://deb.debian.org/debian/pool/main/s/speexdsp/speexdsp_1.2.1.orig.tar.gz)

```sh
work="$PWD/.local/wakeword/c161-first-command-v5/lexical-echo"
app="$PWD/research/wakeword_p4/speex_echo_p4"
mkdir -p "$work"
curl -fsSL https://deb.debian.org/debian/pool/main/s/speexdsp/speexdsp_1.2.1.orig.tar.gz -o "$work/speexdsp-1.2.1.tar.gz"
printf '%s  %s\n' 8c777343e4a6399569c72abc38a95b24db56882c83dbdb6c6424a5f4aeb54d3d "$work/speexdsp-1.2.1.tar.gz" | sha256sum -c -
tar -xzf "$work/speexdsp-1.2.1.tar.gz" -C "$work"
idf.py -C "$app" -B "$work/build-cue" -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/sdkconfig" \
  -DSDKCONFIG_DEFAULTS="$app/sdkconfig.defaults" -DNYX_SPEEX_FLOAT=ON -DNYX_QEMU_FPU=ON -DNYX_SPEEX_FIXED_CUE=ON reconfigure
cmake --build "$work/build-cue" --parallel 2
```

定点回放使用默认选项，目录 build。纯浮点回放启用 NYX_SPEEX_FLOAT 和 NYX_QEMU_FPU，目录 build-float。浮点模拟使用先前自测中的 IDF 浮点异常指令识别配置。可在现有研究环境分别运行 replay_c161_speex_p4.py、其 --float 选项、check_c161_speex_fixed_playback.py 和 verify_c161_speex_cue.py。

在上面的浮点联合配置基础上，build-hold 增加 `-DNYX_SPEEX_HOLD=ON`，build-cold-guard 再增加 `-DNYX_SPEEX_COLD_GUARD=ON`。这两个回放固件在输入 PCM 后读取两个小端 uint32，分别为本机 v5 的起止样本，无本机播放时均为零。使用同一研究环境依次运行 check_c161_speex_demo_hold.py、check_c161_speex_hold_rooms.py、check_c161_speex_cold_guard.py、check_c161_speex_player_gain.py，最后用 package_c161_speex_player_replay.py 可重建数字注入包。生成测量的脚本要求输出路径空闲，以保留原始 UART 和报告。
