# PocketSphinx 英语短语搜索对照

使用 PocketSphinx 5.0.4 随包的英语声学模型、CMU 发音词典和 `hey nicks` 单短语搜索。字典给出 HH EY 和 N IH K S。采用发布默认的阈值 1e-30、100 Hz 特征、10 帧关键词返回延迟，每收到 480 个 16 kHz 样本读取一次新结果。各录音独立重置特征、均值和搜索。只计连续送入时已返回的事件，保留原始返回样本、段起止和分数。程序在主机单进程 CPU 上运行。另一路保留 C161 原通知，在挖掘版词分类器的每一个实际候选点回看 1.5 秒、最多追加 300 ms，允许关键词结束于候选点前至多 450 ms，按实际可用时钟合并通知。[发布包与身份](https://pypi.org/project/pocketsphinx/5.0.4/) · [上游参数](https://pocketsphinx.readthedocs.io/en/latest/config_params.html)

| 路径 | 连说命中 /36 | 连续普通句误触 /108 | 独立日常句误触 /96 | 四声音原速 /12 | Joe 目标 /72 | Joe 近音误触 /216 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 C161 | 7 | 0 | 0 | 1 | 0 | 0 |
| PocketSphinx 连续监听 | 29 | 35 | 26 | 12 | 34 | 14 |
| C161 加词分类器提议、PocketSphinx 确认 | 30 | 2 | 1 | 6 | 21 | 5 |

连续监听在 af_aoede、am_eric、bf_lily、am_fenrir 的 12 条原速命令上全部命中，返回时钟在首个命令音素前 30 ms 至后 130 ms。96 条日常句中也会被 “Turn on the light.”、“I will see you next week.” 等唤醒。Bot 的 48 次普通回复有 21 次通知。加提议和确认后，连续普通句剩下 “Hey Nick, is there any rain expected this afternoon?” 和 “Hey Max, can you carry the other bag upstairs?”，独立日常句剩下 “Hey Nick, can you help me?”。普通 Bot 回复零通知，自播示范唤醒词 4/8。被前级漏掉的 af_aoede 和 am_fenrir 语音无法进入确认。Mac 历史目标录音在主机上分别为 4/4 和 2/4。[逐项结果](../results/4090-c161-pocketsphinx-v1.json)

在已有 66 段完整背景（约 28.001 小时）上，前级词分类器有 203 次提议，短语搜索确认其中 2 次，发生于 qajyo 和 thnuq。两处此前分别转写到 “so many bags” 和 “our first speaker”。这里评估的是提议后短语确认的完整链路。独立连续监听本轮只回放了标明的语音和家庭场景。原 C161 单路在同批长背景零通知。PocketSphinx 的英语声学目录约 6.4 MiB，另有约 3.1 MiB 字典，P4 移植、持续实时成本和实机资源尚待单独验证。[203 次提议的确认结果](../results/4090-c161-pocketsphinx-background-v1.json)

下轮原生短语引擎的灵敏度仅用现有 918 条训练源标定。比较 `1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30` 六个阈值，按每源是否正确分为目标、以 Hey 开头的负例、其他普通语句负例。给目标召回率一半权重，两类负例的拒绝率各四分之一；同分取更高的普通语句拒绝率，再取更高的目标召回率，再按上列顺序。用所选单一设置重放所有前述语音和 203 个完整背景提议，原单 C161 的工作点保持固定。这是本轮采用的研究选择规则，Joe 仍为已消费开发集。

## 训练源标定与主机资源

根据上面的规则，训练源选择了 1e-15。目标正确按首次标注词尾后 300 ms 内获得正确返回计入，负例检查整条语音。训练内三组正确数依次为 144/231、268/354、304/333。固定该设置后独立连续监听在既有四声音原速仍为 12/12，三段连续对话为 23/36、普通句误触 14/108，独立日常句误触 10/96。家庭噪声中的普通唤醒在 0、6、12 dB 为 0/8、1/8、0/8。小分类器提议后再确认则在三段连续对话中命中 20/36，保留原 C161 的七条，连续普通句 0/108、独立日常句 0/96。原速四声音仅 6/12、Joe 12/72。完整 66 段既有背景的 203 个候选中仍确认一次，在 qajyo 的普通广播语音处。仅在旧短句集上消除误触会高估这项改动。[六档训练内标定](../results/4090-c161-pocketsphinx-calibration-v1.json) · [完整语音](../results/4090-c161-pocketsphinx-calibrated-v1.json) · [完整背景候选](../results/4090-c161-pocketsphinx-calibrated-background-v1.json)

从发行字典提取原始两行发音后为 25 字节。24 段正负语流中，使用两词字典的原生解码器有无模型文件内存映射，逐条返回的短语、起止帧、首次报告时钟和分数均与发布字典一致。独立 Linux 主机进程中，从解码器初始化前到回放结束的 Private_Dirty 增量，完整字典为 24960 KiB，两词字典配合映射为 3244 KiB，关闭映射为 8104 KiB。这里是两个时点的内存增量，硬件、分配器和闪存路径会影响 P4 结果。两词字典仍需英语声学参数。[逐进程内存、CPU 和返回值](../results/4090-c161-pocketsphinx-dictionary-v1.json)

下一组在全部设置冻结后生成四段新连续对话，声音固定为 af_nova、af_jessica、am_santa、bm_lewis，均来自已有训练声音清单。每种声音使用相同的两个新载句，开头各用 Hey Nicks、Hey Willow、Hey Nick 和 Hey Max，另有两句日常英语作为对照。原速生成完整句，按固定随机顺序接成自然间隔的连续流。使用原 C161、官方 wn9_heywillow_tts 默认配置、PocketSphinx 默认及训练源标定配置，按各自适用的目标和返回样本比较。

## 四段新对话与原速首句

四个训练清单内声音各合成一段十句连续对话。两句新命令为 “set a reminder for my bus” 和 “raise the screen brightness”。每个地址词各有八句，另外八句为日常英语。按通知时钟落在各句的左闭右开区间计数，同一句最多计一次。所有路径在句间空隙均为零通知。

| 监听路径 | Hey Nicks /8 | Hey Willow /8 | Hey Nick /8 | Hey Max /8 | 日常句 /8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 C161 | 2 | 0 | 0 | 0 | 0 |
| 官方 wn9_heywillow_tts，默认阈值 | 0 | 8 | 0 | 0 | 0 |
| PocketSphinx 默认，直接监听 | 7 | 0 | 5 | 0 | 7 |
| PocketSphinx 标定，直接监听 | 7 | 0 | 3 | 0 | 3 |
| 原 C161 加词分类器提议、默认短语确认 | 6 | 0 | 0 | 0 | 0 |
| 原 C161 加词分类器提议、标定短语确认 | 5 | 0 | 0 | 0 | 0 |

标定直听误触包含 “The next bus leaves in five minutes.” 一次和 “Please mix the two colors before lunch.” 两次。标定组合在 af_nova、af_jessica、am_santa、bm_lewis 各命中 2、0、2、1 句。四段新流的句子和顺序相同，只提供这两个载句上的配对证据。[原速 WAV](../results/4090-c161-pocketsphinx-willow-wav.zip) · [句子和采样区间](../results/4090-c161-pocketsphinx-willow-plan.json) · [官方原始返回](../results/4090-c161-pocketsphinx-willow-official-v1.zip) · [原 C161 分数](../results/4090-c161-pocketsphinx-willow-scores-v1.npz) · [双方通知](../results/4090-c161-pocketsphinx-willow-v1.json) · [短语匹配和逐次提议](../results/4090-c161-pocketsphinx-willow-lexical.json)

旧四声音每个声音依次说 lower、start、read 三条命令。从目标起点到首个命令音素后 300 ms，取前级词分类器连续五步 UINT8 输出和的最高值。通知门槛为 1175，理论最高 1275。af_aoede 三句依次为 235、160、88，am_fenrir 为 19、80、0，六个完整目标区间均零提议。am_eric、bf_lily 每句均达 1275，每句均有提议。这个差异在短语确认之前已经出现。[前级实际提议](../results/4090-c161-room-word-negatives-hard-replay-v1.json) · [逐步分数](../results/4090-c161-room-word-negatives-hard-replay-scores-v1.npz)

以两条标定路径的首次通知为准，在下一个 160 样本交付边界起录，对同一份单声道 PCM 比较即时起录与回看 32000 样本。物理首音素按 Kokoro 时长标注判定。完整命令要求 RTX 4090 上无提示 Whisper small.en 的转写以全部预期命令词结尾，允许此前出现唤醒词。漏唤醒计在总请求分母中。此处回放到标注句尾，未覆盖实机端点检测。

| 输入与路径 | 已通知 / 请求 | 即时保留首音素 | 即时完整命令 | 回看保留首音素 | 回看完整命令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧四声音，直接 | 12/12 | 3/12 | 6/12 | 12/12 | 12/12 |
| 旧四声音，组合 | 6/12 | 3/12 | 3/12 | 6/12 | 6/12 |
| 新对话，直接 | 7/8 | 5/8 | 5/8 | 7/8 | 7/8 |
| 新对话，组合 | 5/8 | 4/8 | 4/8 | 5/8 | 5/8 |

新对话即时起录时，Whisper 在两条直接监听的录音中把 “set a reminder for my bus” 转写为 “a reminder for my bus”。在已有 Bot 回放中，标定的直接监听对普通回复有 9/48 条误触、对明确示范唤醒词有 6/8 条触发；标定组合依次为 0/48 和 4/8。标定组合在已有 28.001 小时背景的 203 次前级提议中确认一次，出现在 qajyo 广播的 “so many bags”。本轮 Mac 继续用原 C161 和固定 v5 做实机验证。[原速捕获和逐词转写](../results/4090-c161-pocketsphinx-first-command.json) · [Bot 及普通对话回放](../results/4090-c161-pocketsphinx-calibrated-v1.json)
