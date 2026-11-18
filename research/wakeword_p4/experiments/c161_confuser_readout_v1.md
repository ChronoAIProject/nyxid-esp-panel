# 在单输出 C161 内增加干扰词判别

以 8912650a 有声前文模型为父模型，在同一份 3480 维隐层特征上增加一个带 ReLU 的全连接输出，用来预测应从原唤醒 logit 扣除的非负量。在原 logistic 之前合并两路数值，对外仍是原单输入、单 UINT8 输出。使用已注册的 FullyConnected、Concatenation 和原 Logistic；原网络的状态与特征计算只进行一次。

4090 生成六个训练身份、两档语速的成组新句，同一命令分别由 Hey Nicks、Nick、Nicole 引入，另配日常数词、其他常用词句。将这些原音加入前轮已许可的 Piper、Kokoro、Chatterbox 训练原音。用静音和对话前文、两种前端取完整序列。在父模型分数最高的五步附近训练干扰词判别，使用固定的最终参数导出原生 INT8 执行图。观测完整连续命令逐项得失、两段广播、日常姓名与数词、Joe、Mac 以及四声音 Willow 对照，并核对导出的分数上界和事件时钟。已消费的开发原音继续只用于比较。

## 首次训练与单输出执行结果

六个既有训练声音合成 180 条完整语句，全部无削波。加入已有训练来源后，在 4090 对干扰词输出训练 640 步。导出的 INT8 长 65384 B，含一个输入、一个 UINT8 输出。两路分数在 911 次旧回放、四声音会话及两个完整广播中均低于或等于有声父模型。911 次回放的主路、辅助路及合并事件和时刻全都等于父模型。六组数据中，父模型原始分数达到 235 的帧没有一帧被扣分。“six cups” 触发时两路五步总分均为 1235，独立 BR Nicole 为 1195、1184，BR Nick 的辅助分数为 1203，三处均原样通过。另两个广播沿用父模型的误触。四声音结果也回到父模型的单路 5/12、双路 7/12。[生成来源](../results/4090-c161-confuser-readout-sources-v1.json) · [640 步训练](../results/4090-c161-confuser-readout-training-v1.json) · [逐句与分数验证](../results/4090-c161-confuser-readout-audit-v1.json) · [广播和四声音验证](../results/4090-c161-confuser-readout-transfer-v1.json)

后续控制实验复用同一图和同一训练原音。直接监督 ReLU 之前的干扰词分数，让高风险负例有梯度，即使起始干扰分数为负也能参与更新。保留父模型连续五步均值达到 0.70 的负例窗口，并在正例教师峰值处施加不扣分的约束。只用训练材料确定窗口与标准化参数，最终量化后重放相同的完整流。

## 直接拟合干扰分数后的实测

第二版在 RTX 4090 上用 552 个正例窗口、219 个高风险负例窗口拟合量化前的有符号分数。导出沿用父层权重尺度，117 个权重超出可表示范围并被截断。单输入单输出 INT8 模型为 65384 B，SHA256 为 `fde2dd183dd48efbd823ad80c730484bfa79f09896563154e0c6fa5ec5d6b48f`。阈值 235/256，五步平滑，67 步冷却。六组旧回放中父模型原始分数达到 235 的 3197 个通道帧里有 1924 个被扣分。[训练记录](../results/4090-c161-signed-confuser-training-v1.json) · [模型](../results/4090-c161-signed-confuser-single-v1.tflite)

| 36 条连续命令 | 原 C161 | 有声前文父模型 | 第二版 | 相对原 C161 新命中 | 相对原 C161 漏掉 |
| --- | ---: | ---: | ---: | --- | --- |
| 面板单主路 | 7 | 29 | 19 | 内置 2/7/8/11，US 1/3/7/9/10/11，BR 1/6 | 无 |
| 主路与辅助路合并 | 21 | 32 | 27 | 内置 7/8，BR 1/2/7/10/11 | US 6，leave the hallway light on until I get upstairs |

相对有声前文父模型，单路丢失内置 3/5、US 2/5/6、BR 0/3/7/9/11，双路丢失内置 3、US 4/6、BR 0/3。三段会话中全部 108 条普通语句的单路与合并通知均为零。独立日常 96 句也为零，包括 “We need six cups of water.”。独立 BR Nicole、BR 场景 0 的 Nick 和其余共享前文均为零。完整广播 kajfh 中的 “Mr. Kevin Strand” 及 ofbxh 中的 “our best policemen” 在第二版单路、辅助路与合并后均为零。[连续会话每条命令、普通语句和样本时钟](../results/4090-c161-signed-confuser-audit-v1.json) · [独立日常句](../results/4090-c161-signed-confuser-daily-v1.json) · [匹配前文](../results/4090-c161-signed-confuser-willow-v1.json)

旧四声音场景中，第二版单路 4/12，双路 6/12。原 C161 为 1/12、3/12，首版与有声前文父模型为 5/12、7/12，官方 Hey Willow 为 9/12。第二版四次单路通知在合成命令首音素后 60–170 ms，面板录音需要相应的前置缓冲。Mac 现有 4 条目标中命中 2 条；Joe 的 72 条仍全部漏掉。BR 连续命令单路仅 2/12。下一组比较使用冻结模型、新编的三组对话及四个既有诊断声音，同一场景的两个唤醒词共享同一份前文。66 个已消费背景按原 PCM 身份逐一清点。[旧四声音与两段广播的逐次通知](../results/4090-c161-signed-confuser-transfer-v1.json)

## 新场景、完整背景和原模型的命中保留

在模型冻结后用 4090 生成四个声音各三组新请求。每组的 Nyx 和 Willow 共用完全相同的聊天、姓名或数词前文和命令文本，所有合成片段无削波。在 12 个对应目标段中，C161 单路 1 次、双路 2 次，第二版分别 5 次、7 次，官方 Hey Willow DEFAULT 命中 12 次。第二版单路命中 am_eric 的场景 1/2 和 bf_lily 的全部三场，af_aoede、am_fenrir 均漏。命中的通知比命令首音素晚 70–180 ms。三条路径（原 C161、第二版与官方）在共享前文和对方目标段均为零次通知。旧、新四声音来自同一合成器，不能由此估算真人唤醒率。[新场景逐句时钟](../results/4090-c161-signed-forward-candidate-v1.json) · [官方返回和 UART](../results/4090-c161-signed-forward-official-v1.zip) · [冻结输入](../results/4090-c161-signed-forward-inputs-v1.zip)

66 个已消费背景按 PCM SHA256 去重，共 28.0012 小时。两段已完整运行的广播沿用逐帧分数，其余 64 段从 PCM 运行同一模型。三批共 12/12/42 段，主路、辅助路和合并通知均为零。[按文件的事件、时钟及来源](../results/4090-c161-signed-confuser-background-v1.json)

另外重放 1188 条既有的单独语句。对每条核对主路量化输入摘要、时钟及原 C161 的事件，再使用原阈值统计同一输入上的得失。下表的四组正例各含 9 个裸词、27 个命令变体。

| 声音 | 原单路 /36 | 第二版 /36 | 原先命中后漏掉 | 新增命中 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 35 | 22 | 14 | 1 |
| Jenny | 33 | 8 | 25 | 0 |
| Melo US | 33 | 14 | 21 | 2 |
| Melo BR | 21 | 5 | 17 | 1 |

Jenny 的 44 次构造近音误触减少到 19 次，同时丢失 25 条原本识别的唤醒。Legacy 中新出现 1 条交叉触发，输入为 “Hi E S P, turn on the light.”，样本 41760。原 C161 继续用于面板验证。历史报告的目标事件说明负例改善伴随明显的跨声音召回损失。保留两份训练模型及完整分数供复核。[Legacy](../results/4090-c161-signed-confuser-isolated-primed-v1.json) · [Jenny](../results/4090-c161-signed-confuser-isolated-jenny-v1.json) · [Melo](../results/4090-c161-signed-confuser-isolated-melo-v1.json)

同一批 1188 条孤立语句又用扣分前的有声父模型回放。四组各 36 条正常目标，父模型在 Legacy、Jenny、Melo US、Melo BR 上分别命中 36、33、35、31，加入第二版扣分后分别为 22、8、14、5。Jenny 的 25 条损失全部发生在加入扣分之后。[父模型 Legacy](../results/4090-c161-confuser-parent-isolated-primed-v1.json) · [Jenny](../results/4090-c161-confuser-parent-isolated-jenny-v1.json) · [Melo](../results/4090-c161-confuser-parent-isolated-melo-v1.json)

下一次训练使用同一单输出图，加入此前已许可但未用于干扰词拟合的 59 条 Chatterbox 正例。在正例的教师五步峰值窗口内保护高响应的各帧，并给干扰词权重使用独立的 INT8 尺度。在训练文件中按原音 SHA256 分出校准子集，用这部分教师已能命中的窗口确定扣分偏置，随后在四个旧声音和两套命令场景比较。训练和偏置选择均不使用回放结果或验收留出。

## 增加正例保护后的 4090 结果

新加入 59 条训练正例后，用 2697 个正例帧及 208 个高风险负例窗口拟合。按训练原音摘要预留的 104 个父模型命中窗口在校准计算中全数保留。新图仍为单输入、单 UINT8 输出，65384 B，SHA256 为 `6b816b3ef262a6543ec22de11c364b49caa8667d0ccdec6159f02f1b11845831`。[训练与模型身份](../results/4090-c161-protected-confuser-training-v1.json)

| 目标集 | 原 C161 单路 | 前一份扣分模型单路 | 新模型单路 |
| --- | ---: | ---: | ---: |
| 连续命令 /36 | 7 | 19 | 29 |
| Legacy /36 | 35 | 22 | 35 |
| Jenny /36 | 33 | 8 | 30 |
| Melo US /36 | 33 | 14 | 31 |
| Melo BR /36 | 21 | 5 | 19 |
| 最新四声音请求 /12 | 1 | 5 | 6 |

新模型的单、双路连续命令逐句均与有声前文父模型相同，双路 32/36。原单路 7 条和原双路 21 条全部保留。相对原单路新增的 22 条为内置 2/3/5/7/8/11，US 1/2/3/5/6/7/9/10/11，BR 0/1/3/6/7/9/11。独立语句与原 C161 匹配时，Legacy 得一失一，Jenny 失三，US 得二失四，BR 得五失七。Joe 仍为 0/72，Mac 为 2/4。最新四声音请求中，am_eric、bf_lily 的六条均命中，af_aoede、am_fenrir 的六条单路全漏。新双路 8/12，官方 Willow 对应请求为 12/12。本次重放复用已发布场景。[每条连续语句](../results/4090-c161-protected-confuser-audit-v1.json) · [四声音每条通知](../results/4090-c161-protected-confuser-forward-v1.json)

BR 场景 0 的 Nick 再次在样本 64800 误触合并路径，独立 Nicole 和 “We need six cups of water.” 在样本 12480 误触主路，广播 kajfh 的 Kevin Strand 在样本 13367040 误触主路与合并路径。另一个完整广播 ofbxh 在本模型上零通知。这份模型有已复现的日常词误唤醒，本轮只针对这两个广播做完整重放。66 段背景全部零事件的结果属于上一份扣分模型。面板普通语音基准继续使用原 C161。[独立日常句](../results/4090-c161-protected-confuser-daily-v1.json) · [两个完整广播、旧四声音](../results/4090-c161-protected-confuser-transfer-v1.json)
