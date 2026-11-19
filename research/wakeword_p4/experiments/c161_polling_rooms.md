# 普通 Nyx 在背景中的唤醒与首句

RTX 4090 从相同的五类后验蒸馏权重开始，分别微调 450 步，两次使用相同的小批量和标签。每次使用原有 5,067 个训练窗和新增的 6,615 个窗。新增 1,323 条训练来源覆盖 Piper、Kokoro 和 Chatterbox。每条有三个相差 120 ms 的发音窗、一个发音前背景窗和一个结尾空窗。房间训练组对其中 849 条加入训练分区房间响应和背景，保留其余干声；配对训练组使用全部干声。两份模型用相同的 200 个校准窗转为全 INT8。[来源及房间身份](../results/4090-c161-polling-rooms-sources.json) · [训练与原权重哈希](../results/4090-c161-polling-rooms-fit.json) · [导出、校准和哈希](../results/4090-c161-polling-rooms-exports.json)

两组模型冻结后，Melo 的 EN-US 和 EN-BR 各说 “pause the kitchen speakers” 和 “call the front desk”。每组包含有逗号与直接连说、四种 Nyx 拼写、Hey Willow、六种近似称呼和两条日常语句。每条连续语流配相同的干声与 6 dB 房间条件，分别使用厨房空调和客厅音乐。C161 保持 81,600 样本连续预热、五步平滑和 0.91796875 阈值。官方 Hey Willow 使用 DEFAULT 和匹配预热。在完全相同的语流上，各模型按各自目标打分。两秒回看用 RTX 4090 Whisper small.en 转写，正确需转写末尾包含完整命令。通知采用输入样本可用时钟。[16 段配对语流及音素](../results/4090-c161-polling-rooms-challenge-plan.json) · [音频](../results/4090-c161-polling-rooms-challenge-wav.zip) · [官方 Willow 原始返回](../results/4090-c161-polling-rooms-challenge-willow.zip)

| 额外路径，各自与原 C161 合并 | 干声唤醒、首句完整 /32 | 房间唤醒、首句完整 /32 | 干声、房间近音误唤 /48 | 房间日常误唤 /16 |
| --- | ---: | ---: | ---: | ---: |
| 仅 C161 | 11、8 | 7、4 | 8、0 | 0 |
| 原五类模型，每 120 ms 检查 | 18、14 | 7、4 | 8、0 | 0 |
| 干声微调，每 120 ms 检查 | 15、12 | 9、5 | 9、0 | 0 |
| 房间微调，按原语音活动起点检查 | 17、14 | 9、5 | 10、0 | 0 |
| 房间微调，每 120 ms 检查 | 21、15 | 15、10 | 13、15 | 3 |
| 单个小词模型提议后再作五类复核 | 18、15 | 8、5 | 11、0 | 0 |
| 两个小词模型均有足够分数，再作五类复核 | 12、9 | 7、4 | 8、0 | 0 |

定时检查要求连续两次 Nyx 后验至少 0.5。单个小词模型为此前用困难负例训练的 INT8，每个实际提议后 360 ms 取五类窗。双模型检查进一步要求在该提议时，干声与困难负例两个小模型逐步取较小值，连续五步之和达到 1175。两种提议规则是看过家庭回放或连续背景后的开发诊断。按提议检查的干声新增误唤涉及 Hi Next、Hi Nick、Hey Mick。官方 Willow 在自身八条目标上，干声与房间均 2/8 唤醒且首句完整；此前家庭组的固定 Willow 播放为 21/24。两种 Willow 来源和分母分别保留。[各声学条件和各路线的实际 INT8 与逐句转写](../results/4090-c161-polling-rooms-fresh.json) · [单模型提议](../results/4090-c161-polling-rooms-proposals-fresh.json) · [双模型复核](../results/4090-c161-polling-rooms-proposals-agreement.json) · [本轮 Willow 转写](../results/4090-c161-polling-rooms-challenge-willow.json)

## 家庭回放与日常误唤

原有 24 段普通 Nyx，每段开头有两秒实录背景。房间训练模型定时检查时，目标区间内累计有 12 段含通知，但第一条通知在唤醒词区间的只有 7/24，另有 9/24 提早被背景触发。单模型提议加复核的第一条正确通知是 9/24，双模型提议加复核是 6/24，C161 是 6/24。将唤醒音换成静音并保留后续命令，以完整录音计，三种研究配置分别有 11/24、3/24、1/24 误唤。单模型提议加复核的三条均是 “Play the evening news” 所在的客厅音乐条件。[家庭完整时钟](../results/4090-c161-polling-rooms-house.json) · [按提议复核时钟](../results/4090-c161-polling-rooms-proposals-house.json)

在三条已消费的连续对话中，36 条目标上原 C161 命中 7 条，定时检查 29 条、单模型提议加复核 20 条、双模型提议加复核 11 条。在同期 108 条普通对话中，误唤分别为 0、4、0、0；在另 96 条日常短句中分别为 0、1、0、0。在 48 条普通 Bot 回复中两种提议加复核均无通知，但 Bot 主动示范唤醒词时，单模型提议加复核仍触发 2/8。[连续对话](../results/4090-c161-polling-rooms-proposals-context.json) · [独立日常短句](../results/4090-c161-polling-rooms-proposals-daily.json) · [Bot 播报](../results/4090-c161-polling-rooms-proposals-bot.json)

66 条既有连续背景共 1,612,870,654 个 16 kHz 样本，28.001 小时。沿用逐样本身份已核对的原始小词模型和 C161 记录，再对全部 203 个小词提议中的 202 个完整后续窗复核，最后一个不完整窗不发通知。单模型提议加五类复核仍误唤 11 次，约 0.39 次/小时，实际转写包含 “so many bags” 及 “I think it's a good idea to have this conversation with you”。双模型提议加五类复核在同一组回放为零，原 C161 在该组也为零。这 66 条录音已用于方案诊断，不能作为独立验收时长。[逐条复核与八秒上下文](../results/4090-c161-polling-rooms-proposals-background.json)

## P4 数值核对

针对家庭、两种新命令、独立日常、连续对话和 Bot 共 131 个实际提议窗，在 ESP32-P4 QEMU 使用支持逐通道量化的全连接算子执行新房间 INT8。655 个输出码与主机逐项相等，张量区实用 65,164 B，单窗 Invoke 中位 24.392 ms，最大 26.911 ms。该计时只覆盖五类复核算子，整机实时调度、功耗和板端麦克风性能待实测。本轮普通语音实机基准继续使用冻结 C161；固定播放使用 v5。[房间 INT8](../results/4090-c161-polling-rooms-room.tflite) · [P4 时钟、样本哈希和输出码](../results/4090-c161-polling-rooms-proposals-p4.json) · [UART](../results/4090-c161-polling-rooms-proposals-p4-uart.log)
