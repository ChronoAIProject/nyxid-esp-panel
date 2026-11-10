> 当前评测与标注复核见 [BEST_PRACTICES_EVALUATION.md](BEST_PRACTICES_EVALUATION.md)。下文保留历史状态：v25 已完成且失败，Hi ESP 配对评分已完成但端点只属代理；最新结论以 GPU_RESEARCH.md 及新复核为准。

> 最新状态以本节为准，正文后续保留历史研究记录。
>
> 2026-11-10 再次在线核查 microWakeWord、openWakeWord、Howl 和乐鑫官方资料，均 HTTP 200；来源哈希见 `results/4090-best-practices-v25-sources.json`。
>
> **无板、纯 TTS 训练已有成熟路线。** 乐鑫明确支持无需真人录音的纯 TTS 定制；microWakeWord 同时明确提醒，自定义可用模型仍需要实验。官方商业训练配方并未完整开源。
>
> 当前应坚持：说话人隔离；多样化正样本及噪声/RIR增强；训练专用困难负例挖掘；先约束连续音频事件级误触再优化召回；导出 INT8 后用真实流式状态复核。openWakeWord 的约30,000小时负样本和预训练 embedding 是其实现条件，不是本项目必须照搬的门槛，其完整模型也不是 MCU 部署替代品。
>
> 本地已追加100.59088小时训练专用英文，v24仍出现饱和负例，延长训练并未解决。v25正在固定架构/初始化下试验训练池挖掘的困难负例，结果未出，不预判成功。
>
> **旧文“Hi ESP 无法无板 detect”已被后续实验替代：** classic ESP32 QEMU 已能运行未修改的官方 wn9_hiesp 并回放真实 PCM。33条验证说话人流全部完成，源音频5.121184826小时（不计插入静音），0事件，单侧95%上界0.584968591/h；阈值请求0.63、读回0.62496001。详见 `results/4090-hiesp-libridev-validation-v1.json`。这是朗读验证，不是自然对话、独立验收或匹配误触工作点的完整对比。
>
> 配对开发集72条已生成并做严格ASR审计。Hi ESP文本严格匹配0/12包含“High ESP”等同音拼写，不能解释为全部发音错误；不按识别结果过滤样本。该集尚未用于唤醒评分，合成结束点也不是音素结束真值。独立最终集继续冻结。
>
> 下一步以相同PCM、各自原生前端比较两引擎，在开发集固定工作点后比较召回和延迟。Nyx尚未证明达到Hi ESP，目标保持未完成。

# Hey Nyx：对照官方训练方法的复核

2026-11-10。依据一手文档；来源 URL 和下载内容哈希见
`results/4090-best-practices-sources.json`。以下区分上游方法与本项目已经实测的结果。

## 判断

纯 TTS、无需开发板训练自定义唤醒词是已有可行路线。我们已经证明
4090 能完成这条计算流程，尚未证明模型具备产品质量。当前结果不理想，
不应解释为任务本身必须接板，也不能通过不断换 TTS 或延长训练解决一切。
应先对齐成熟流程的数据覆盖、声学增强、标定与评测方法。

microWakeWord 的官方 README 明确提醒：训练可用模型仍需大量实验，
入门 notebook 通常不能直接产出可用模型。因此“有成熟方案”成立，
“合成若干语音后一次训练就有商业模型的效果”不成立。

## 一手依据与本地差距

| 方面 | 官方方法或要求 | 当前研究状态与动作 |
| --- | --- | --- |
| 无板、纯合成 | [乐鑫定制说明](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/wake_word_engine/ESP_Wake_Words_Customization.html)明确支持纯 TTS、无需真人录音训练 | 可继续全模拟训练；真实麦克风、远场和部署质量仍未验证 |
| 正样本 | [openWakeWord](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#training-new-models)建议至少数千条，性能受规模影响；多说话人、随机采样和语音变化是合成方法要点 | Piper 已有 6,000 条训练正样本，但跨来源泛化差；不能把同一声音的速度变化计为新说话人。新增 Qwen 仅有 30 条 ASR 一致正样本，不能据此认定多样性问题已解决 |
| 负样本 | openWakeWord 发布模型使用约 30,000 小时语音、噪声和音乐；其自动 notebook 示例用约 2,000 小时预计算训练特征和约 11 小时验证特征 | 后续已通过 MUSAN 噪声/音乐扩至约 18.7485 原始小时；仍远少于上游规模，电视、远场多人交谈和中文覆盖不足。优先改善类型覆盖，不能仅靠重复小语料 |
| 架构差别 | openWakeWord 的核心是大规模预训练、冻结的语音 embedding；microWakeWord 面向微控制器、直接训练小型流式网络 | 不能把 openWakeWord 的数据规模或效果直接套到 25,537 参数的 MixedNet。可增加桌面参考模型诊断数据问题，但不能把它当作 MCU 模型性能或资源占用的证明 |
| 房间与通道 | [microWakeWord notebook](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/notebooks/basic_training_notebook.ipynb)使用真实 RIR、背景音、增益、音高、EQ、失真和带阻；示例背景 SNR 为 -5 到 10 dB | v4–v6 已加入实测 MIT RIR 和 -5 到 10 dB SNR；v6 仍未通过开发回归，证明增加此增强还不足以解决泛化问题 |
| 特征增强 | microWakeWord 支持 SpecAugment；入门 notebook 默认时频掩码为零 | 当前没有时频掩码。应加入可配置实验，而非声称上游默认开启或把它当作唯一原因 |
| 模型选择 | microWakeWord 支持先约束环境误触率，再优化召回；可分别设置采样权重和分类惩罚 | 已有按来源统计、事件验证与 INT8 校准。后续用固定验证集比较受控实验；候选没有改善时不消耗独立最终集 |
| 最终评测 | [openWakeWord 评测说明](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#performance--evaluation)使用噪声/混响正样本和约 5.5 小时 Dinner Party 远场对话；microWakeWord 使用 DiPCo 流式测试 | 本地开发回归只有 1.358 小时朗读/环境负音频；零事件仍只有 2.206/h 的单侧 95% 上界。独立 10.745 小时负样本保留未评分，但仍主要是朗读语音，不能代表全场景 |
| 平滑和 VAD | [ESPHome 文档](https://esphome.io/components/micro_wake_word/)提供滑窗、阈值与可选 VAD；小窗口降低延迟，但可能增加误触 | 当前固定五分数平均。可做验证集消融；VAD 能抑制非语音误触，不能解决 Hey Next 等近音词，也不能掩盖唤醒召回差 |

乐鑫真人语料路线另有至少 20,000 条合格语料、500 余人、不同距离和速度等要求，
官方说明优化通常需两到三周。这些是其商业定制流程要求，不能当作
本项目纯 TTS 路线必须照搬的硬门槛，也不是其公开完整训练配方。

## 发现的兼容性风险

[官方预生成特征数据卡](https://huggingface.co/datasets/kahrendt/microwakeword/blob/0da95f94302ca2f4aae3b18fc6560fa6d2bba3d1/README.md)
写明 `window_step=20` ms、40 通道、uint16，转换系数为 0.0390625。
本项目是 10 ms 步长。两者不能只因为都有 40 通道就直接混训，
也不能靠简单插帧恢复原始前端状态。旧文件和新版 notebook 的组合
需逐一核对。优先取原始音频，用同一个 10 ms MicroFrontend 重新提取，
或先查明具体归档的实际配置；目前没有导入这些不明时间基准的特征。

## 当前具体进展与修正后的顺序

截至 2026-11-10 的更新，早期“尚未训练/接入”的待办已被后续实验替代：

1. 真实 MIT RIR、宽 SNR、语音/音乐/噪声已接入；v6–v9 均完成训练与
   INT8 验证。v6 来源平均召回 69.9479%，v7 67.8646%，v8 63.59375%，
   v9 63.6458%；增加训练数据、时长采样或困难负样本均未证明超过 v6。
2. Qwen 使用更强 small.en ASR 复核，扩充为 64 条正样本和 278 条负样本；
   其扩充候选 v8 未改善。ASR 一致性仅作训练质量审计，不是发音真值。
3. 官方 AISHELL-1 的 28 个说话人已按训练/验证/开发回归分开。
   冻结 v6 阈值 0.8，在 6.913985 小时中文验证音频中出现 **1 次**误触，
   单侧 95% 上界 0.686126/h，尚未满足 0.5/h。中文数据尚未用于训练。
4. 后续优先补充真实对话、中文与电视等训练/开发负音频，并统一前端；
   持续将来源泛化和近音词错误单独报告。可选 SpecAugment 仍需受控实验。
5. 新发现的官方 P4 QEMU 正在做固件启动验证，为无板 Hi ESP 实测寻找路径。
   运行时启动、固件启动和 WakeNet 推理是不同验收层次，逐项报告。
6. 只对更有希望的候选做开发回归；独立最终集继续冻结。
   相同音频、匹配误触条件的 Hi ESP 比较仍未完成。

官方资料支持方法选择，不代替本地实验。当前没有任何新证据表明 Hey Nyx
已达标；改进的依据应是更接近部署分布的音频上的实际流式结果。

## 后续实测更新：预训练特征参考

v6 的验证集来源平均召回为 69.948%，开发回归只有 61.120%，且连续
负音频误触率的单侧 95% 上界为 3.4929/h，候选仍被拒绝。
这进一步支持先诊断表示和跨来源泛化问题，而非将训练完成视为成功。

已经按上述建议准备官方 openWakeWord 冻结预训练特征的桌面参考。
94,418 条训练/验证序列来自真实波形，完整缓存和逐流一致性核对已通过；
相关分类头、上游学习率与挖掘实验已完成，但均未达到验收要求。由于网络、前端和帧率均不同，
它用于诊断，不能据此证明 MCU INT8 效果或 Hi ESP 等效性。
详见 `GPU_RESEARCH.md` 和 `results/4090-embedding-cache-v1.json`。

## 本次在线复核

再次读取乐鑫定制说明、microWakeWord 与 openWakeWord 官方 README，
原文内容哈希及访问结果见 `results/4090-best-practices-refresh.json`。
ESPHome 页面本次直接请求返回 HTTP 403，没有将其计作成功复核。

结论仍是：无板纯合成训练有成熟方法，但公开的小模型训练不是一键得到
商业质量。microWakeWord 当前 README 仍明确警告入门 notebook 通常不能
直接产出可用模型。openWakeWord 的纯合成效果还依赖大量预训练的冻结
语音表示，不能只复制其分类头训练时长来推断本项目效果。


## 最新官方复核与决策（2026-11-10）

本轮重新在线读取 microWakeWord README、官方 notebook、openWakeWord README
和乐鑫定制说明，四项均 HTTP 200。URL、内容 SHA256 与原文摘录见
`results/4090-best-practices-transfer-refresh.json`。

成熟的是无板训练与推理流程；自定义短词在多说话人、真实远场和低误触条件下
达到产品质量，仍需数据与评估工程。当前首要问题不在 GPU 是否能训练。

- **数据规模与覆盖**：openWakeWord 官方公开约 30,000 小时负样本，
  本地基础训练约 18.7485 小时。规模不是通用硬门槛，但差距不能忽略。
  后续真实中文和 DiPCo 追加训练也已实测，当前配置未改善，不能声称已解决覆盖问题。
- **预训练的作用**：openWakeWord 使用冻结的共享特征提取器；这不是从零训练
  MCU 小网络的直接对照。MCU 预训练迁移值得受控探索，但并非上游保证有效的配方。
- **增强须用实验证明**：v10 时频遮挡验证改善但开发回归下降，v12 仅频率遮挡
  也未改善。不能将“支持 SpecAugment”解读为必然提高本项目效果。
- **评测以流式事件为准**：优先约束连续环境误触，再比较召回与延迟；
  不以窗口峰值替代事件命中。独立最终集继续冻结。
- **下一项受控对照**：同一 150 帧、30/60 通道网络，比较从零训练与官方
  Hey Jarvis 权重迁移；分类头重置、BN 策略及数据/训练预算保持一致。
  这是计划，尚无训练结果。官方模型重建已通过 16 条探针检查，详见研究日志。

目前仍无证据证明 Hey Nyx 达到 Hi ESP。P4 QEMU 已启动真实固件，但该版本
无法执行 WakeNet 所用的 PIE 指令，尚未完成 detect()，因此没有可比的 Hi ESP 分数。


## 后续实测：标定网格和容量

v21对照发现，原阈值网格从0.8直接跳0.9会遗漏工作点；统一固定1/256步长
且同时复核v19/v21后，原11点指标均精确重现，两者验证来源平均分别81.61%和78.57%。
这只是开发标定，不能把更多候选阈值得到的高分当作独立改进。
原窄模型有饱和负例，任何仍能检测正例的单阈值都无法完全消除该误触；
扩大到70,801参数后虽能在高阈值下消除既有验证误触，但召回降至58.57%，
且远场曝光不足，仍没有达标模型。优先保持跨语料验证、校准规则与评测窗口明确，
拒绝以某一来源零事件或更大模型代替最终验收。详见GPU_RESEARCH.md。


---

# 本轮复核与当前实验调整

核查日期：2026-11-10。仅引用本次实际获取的官方文档；页面版本可变，获取内容 SHA256 见 `results/4090-best-practices-refresh-sources.json`。

## 结论

无需开发板、使用 TTS 正样本训练唤醒词是已有路线；但开源自定义词训练并非稳定的一键商品化流程。microWakeWord 官方明确表示训练好模型仍很困难，需要实验。Espressif 官方也提供纯 TTS 定制，不能再把真人正样本视为启动训练的必需条件。模型离线质量和设备声学整机质量应分别验证。

## 已验证的实践与本研究差距

| 方面 | 官方实践 | 本研究决定 |
| --- | --- | --- |
| MCU 路线 | microWakeWord 使用流式 MixConv、16kHz/40维特征、INT8；openWakeWord 官方提示其完整模型不适合低功耗 MCU | 保留 microWakeWord 为部署主线；大预训练 embedding 可做离线对照，不能直接宣称可部署或等价替换 |
| 正样本 | microWakeWord 使用 Piper；openWakeWord 推荐至少数千生成样本；Espressif 支持纯 TTS | 继续多声线/多生成器，控制原始说话人身份隔离；增加生成数量不等于增加独立真人 |
| 负样本 | openWakeWord 自述发布模型使用约30000小时语音/噪声/音乐；microWakeWord 列出 VOICES/Common Voice/FSD50K/FMA/WHAM! | 当前英文真人训练约3.225小时，覆盖是明显短板；优先已审计无 Piper 身份交叉的 train-clean-100，再扩自然对话/远场。30000小时是其实现规模，不是通用硬门槛 |
| 增强 | 噪声混合、RIR混响、SpecAugment；openWakeWord 评测示例使用5–10dB SNR | 核查现有增强分布，再补目标场景；避免仅重复同一 TTS 的高度相关变体 |
| 难负例 | microWakeWord 使用 adversarial phrase samples，并允许独立采样/惩罚权重 | 从训练专用池挖掘混淆发音；验证误报只做诊断，不回流训练；现有 v20 退化证明不能盲目堆近音词 |
| 选模 | microWakeWord 先满足 ambient false accepts/h，再优化识别指标 | 保留验证集事件级选模；量化后重新校准。停止仅凭窗口分类准确率选模 |
| 流式一致性 | microWakeWord 整窗训练后转流式；其训练期100ms采样只近似真实误报率 | 保留实际流式 INT8 全序列、前端状态、重置和相位核验，最终以连续负音频事件为准 |
| 评测 | openWakeWord 强调真实环境召回、数小时连续音频误报；microWakeWord 使用 VOiCES/DiPCo | 固定验证阈值；最终集继续封存，扩展独立远场/电视/对话评测。0事件时单侧95%上界为2.996/T，达到0.5/h需至少约5.992小时独立负音频；并不代表场景覆盖充分 |
| 后处理 | 多帧平滑；openWakeWord 可加 VAD、噪声抑制、指定说话人 verifier | VAD只可减少非语音误报，不能解决人声近音误报；指定说话人 verifier 会改变用户范围，不能作为通用提升。现有15帧平滑已降低召回，不继续以平滑掩盖模型问题 |
| Hi ESP 对标 | WakeNet9 为嵌入式膨胀卷积并使用多帧平均；官方定制训练非公开一键训练器 | 不同模型前端不能混用。当前 QEMU 在 PIE 指令处失败，尚无 detect 结果，不能宣称达到 Hi ESP |

## 下一轮顺序

1. 结束 v22 BN affine 消融并做相同密集阈值 INT8 验证；不佳则如实拒绝，不触碰最终集。
2. 完成100小时档案官方校验，审计实际身份、转写和解码，再仅追加 training 特征；保持验证/最终音频与特征不变。
3. 保持架构和评测固定，验证扩大负音频覆盖的效果，再考虑扩大模型；增加训练专用远场与自然对话。
4. 候选满足开发门槛后才冻结并做最终独立评测；Hi ESP 对比仍单列未完成。

## 原始来源

- microWakeWord：https://github.com/kahrendt/microWakeWord
- microWakeWord 数据来源：https://github.com/kahrendt/microWakeWord/blob/main/documentation/data_sources.md
- openWakeWord 训练、规模与评测：https://github.com/dscripka/openWakeWord#training-new-models
- openWakeWord 自动训练 notebook：https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
- WakeNet 架构：https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/wake_word_engine/README.html
- Espressif 纯 TTS 定制与语料要求：https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/wake_word_engine/ESP_Wake_Words_Customization.html

Espressif 的客户自带语料要求（至少20000条、超过500人等）属于该定制选项，不是纯TTS方法的前提。本次未联系销售或采购。
