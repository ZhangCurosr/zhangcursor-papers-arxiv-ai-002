---
title: "HiveTraceGuard-Pro-A-Compact-Generative-Guardrail-for-Prompt"
source: https://arxiv.org/pdf/2609.01046v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:24:45"
---

# 论文速读：HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation

## 一句话总结
HiveTraceGuard-Pro 是一个 0.6B 生成式安全护栏，基于 Qwen3-0.6B 经 LoRA 微调，对俄语和英语的请求/响应进行二元安全分类（safe/unsafe）；在 35 个护栏的横评中 aggregate key 达 0.7432 位列第 3，同时以 14.3ms 中位延迟刷新最低记录，并在俄语提示词注入 recall 上达到 0.999。

## 研究问题与动机
- 现有护栏论文的训练与评测几乎全部聚焦英语，缺乏对俄语提示词注入及俄语表面混淆（西里尔转写、键盘布局替换、符号插入等）的系统性证据。
- 生产部署需要独立于主模型的护栏以支持策略热更新，但 8B/12B 护栏在相同跑批下耗时 63–243ms，难以满足低延迟、低内存约束。
- 现有评测难以精确衡量过阻塞（over-blocking）：缺少在相同话题域内配对良性样本的设计，导致假阳性与假阴性无法在同域对照。
- 已有报告未提供对同一模型在干净输入与混淆输入上误差分布的细粒度拆解，难以指导实际部署的阈值选择。

## 核心贡献（创新点）
- **紧凑生成式二元护栏**：0.6B 模型仅输出 safe/unsafe 一个 token，决策规则为两 logit 的 argmax，区别于多数输出多类别标签或多步推理链的护栏。
- **配对式混淆语料构建**：有害样本尽可能与同域良性样本配对，并对双侧标签施加 8 种混淆变换，使过阻塞可在相同话题下直接度量。
- **预留确认集与重叠审计**：在开发前锁定 hivetrace/insecure-prompts 并在模型冻结后单次评估；同时完成训练-基准重叠扫描（PI-RU 27.1%、Aegis-2.0 21.3% 等），对受污染列给出重评分。
- **统一大横评框架**：一次跑批评分 35 个护栏在 19 个基准组（44 个数据集）上的表现，以几何平均 aggregate key 排序，并提供每模型逐变换误差拆解。
- **完整推理契约开源**：权重以 Apache-2.0 发布于 Hugging Face，论文附录给出 chat template、verdict token id、截断行为与可复现代码片段。

## 方法详解
- **基座与微调**：Qwen3-0.6B + LoRA（rank=64, α=32, 7 个目标模块），Unsloth + FlashAttention-2 + sequence packing，LR=2e-4，effective batch=32，2 epochs，max_seq_len=2048，单卡 A100-80GB 约 5.5h。
- **生成式打分与损失**：因果 LM head 直接生成 verdict token（safe=id 18675, unsafe=id 38157, EOS=id 151645）；仅在 completion 阶段的 verdict token 与 EOS 上计算 loss（train_on_responses_only）。
- **决策规则**：取最后位置 safe/unsafe 两 logit 的较大者，等价于固定 0.5 边界的二分类 softmax；全局最优阈值 sweep 得 τ=0.532，对 macro-F1 提升不足 0.001。
- **训练语料**：共 464,092 行，俄语 59.7%、安全样本 58.5%；含 14 个显式危害类别（35,013 行）+1 个隐藏策略类别（2,948 行）；增强块 52,493 行（11.3%），其中混淆 38,729 行、红队 13,764 行。
- **八种训练混淆变换**：random_case、random_insert_symbols、random_replace_symbols、zero_width_insert、leet、transliterate_to_latin、wrong_layout、phonetic_substitution，均作用于安全与有害双侧。
- **评测保留变换**（仅用于评估，未参与训练）：to_lower、to_upper、informal_rewrite。
- **推理序列化**：请求侧使用自然对话路径（最终 user turn 为目标）；响应侧采用 legacy standalone-reply（将回复作为第二个 user turn 传入，目标行仍为 Target: last user message），与论文声明的自然 assistant-role 路径不同。
- **截断与长度**：tokenizer 不自动截断，RoPE 支持至 40,960 位置；80,000 filler tokens 探测中有害请求被判为 safe，超长行为未定义。

## 实验与结果
- **横评规模**：35 个护栏 × 19 个基准组（16 公开 + 3 内部），以几何平均 aggregate key 排序；十五强对比见表 2。
- **综合得分**：aggregate key=0.7432，排名第 3（落后 YuFeng-XGuard-Reason-8B 的 0.7641 和 SingGuard-2B 的 0.7552）；仅算 16 个公开组时为 0.7153，有四款护栏领先。
- **俄语鲁棒性**：clean Russian robustness combined-F1=0.88（15 模型对比最高），FPR=0.02、FNR=0.05；augmented 侧 harm-F1=0.852。
- **俄语提示词注入**：recall=0.999（4,417 个攻击中仅漏 4 个），但 27.1% 与训练集重叠，泛化性存疑。
- **英语提示词注入**：recall=0.88（vs YuFeng-XGuard-Reason-8B 的 0.90）；Aegis-2.0 请求 combined-F1=0.82†（21.3% 训练重叠）。
- **延迟**：p50=14.30ms，15 模型中最低，为 Qwen3Guard-Gen-8B（152.71ms）的约 1/10.7；响应端 p99=37.62ms。
- **整体误差**：跨全套 FPR=0.268、FNR=0.156，过阻塞问题显著；reserved set 响应端 FPR=0.563。
- **类别细目**：16 个 RU-categories 中 25 个 cell 的 unsafe-class F1≥0.90；最低为 Military conflict（0.875）和 Copyright infringement（0.885）。
- **混淆拆解**：wrong_layout 变换 FPR 最高（0.262），held-out informal_rewrite FNR 最高（0.307）；六轮增强实验中改进一侧必恶化另一侧，未解耦。
- **Ablation**：同类 corpus 上用 classification head 得 macro-F1=0.822，低于生成式的 0.898；weight-averaged model soup 无收益；32B 教师蒸馏因在 0.6B 失败 cell 上表现更差被放弃。

## 相关工作脉络
- **Qwen3Guard [1] / Llama Guard [10] / ShieldGemma [11]**：多类别固定策略护栏；本文定位更紧凑（0.6B）、更侧重俄语文本混淆与低延迟场景。
- **WildGuard [12] / PolyGuard [13]**：前者面向对抗性提示与越狱，后者覆盖 17 种语言；两者均未报告俄语表面混淆的细粒度结果，本文填补该空白。
- **YuFeng-XGuard [2] / SingGuard [3] / Shieldstral [4]**：强调策略可解释或运行时策略适应；本文舍弃多类别输出换取推理开销的大幅下降。
- **OpenGuardrails [5]**：15B judge 平台化方案；本文证明 0.6B 生成式二元分类可在多数俄语指标上逼近更大模型。
- **现有工作共性缺口**：缺乏对同一模型在干净/混淆输入上 FPR/FNR 的成对拆解，以及开发前预留确认集+重叠审计的评测纪律，本文提供了可复用的方法论模板。

## 局限性与未来方向
- **响应侧评估路径非自然**：所有响应结果使用 standalone-reply 序列化，未全面评测 assistant-role 分支；仅在 Aegis-2.0 上做了一次敏感性检查（macro-F1 从 0.807→0.799）。
- **训练-评测重叠**：PI-RU（27.1%）、S-Eval attack（10.0%）、Aegis-2.0（21.3%）、ToxicChat（3.2%）均有残留重叠，部分高性能结果可能高估泛化。
- **混淆 trade-off 未解**：recall 与 over-blocking 在八种变换和六轮增强中始终耦合，0.6B 容量限制与数据配比限制的归因尚不明确。
- **超长上下文未验证**：超出 40,960 位置行为未定义，80k filler 探测显示可被绕过。
- **语言覆盖仅限俄/英**：其他语言部署需重新评估；单一全局阈值对英文排序质量也有影响（Aegis-2.0 AUROC=0.888）。
- **单次跑批、无置信区间**：aggregate key 排序中微小差距（如 0.7432 vs 0.7420）无法排除随机性。

## 研究启发与可借鉴点
- **配对式 over-blocking 度量**：在同域内构造 safe↔harm 配对并对双侧施加相同变换，是量化假阳性来源的有效范式，可迁移至任何多语言安全评测。
- **预留确认集 + 冻结后单次评估**：开发全程不查询 hivetrace/insecure-prompts 的做法能有效避免乐观偏差，适合安全模型的最终验证流程。
- **生成式二元 token 决策的简洁性**：只用两个固定 token 做 argmax，省去多分类头与校准步骤，对边缘/低延迟部署有直接参考价值。
- **双侧混淆增强**：对安全与有害样本施加相同扰动而非仅增强有害面，有助于模型学习语义不变性而非表面特征，该方法可迁移到其他鲁棒分类任务。
- **完整推理契约披露**：公开 template、token id、截断语义与复现代码，使外部复现成为可能，为安全模型评测树立了可复用规范。

## 关键术语表
**Guardrail（护栏）**：独立于主 LLM 的安全分类模块，在请求发出前或响应交付前进行内容审核。
**Prompt Injection（提示词注入）**：通过指令替换或角色操控使模型忽略系统提示词的攻防手段。
**Jailbreak（越狱）**：保持恶意意图不变，以角色扮演/假设情境等语义变换规避安全策略的攻击。
**Surface Obfuscation（表面混淆）**：不改语义仅改变文本表层形式的攻击，含转写、键盘布局替换、符号插入等八类变换。
**Combined-F1**：在 pooled safe 与 harmful slices 上按 unsafe-class F1 公式计算的 F1，不同于仅看 unsafe 类的 harm-F1。
**Aggregate Key**：19 个基准组分数的几何平均，用于对护栏做总体排序。
**Legacy Standalone-Reply**：将助手回复单独作为 user turn 传入的评估序列化方式，与非自然但被广泛使用的 response 评测路径对应。
**Over-blocking（过阻塞）**：护栏将良性内容错误判定为有害的
