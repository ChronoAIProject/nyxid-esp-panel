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
