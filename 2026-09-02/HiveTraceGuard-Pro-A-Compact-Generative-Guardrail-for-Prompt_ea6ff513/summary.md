---
title: "HiveTraceGuard-Pro-A-Compact-Generative-Guardrail-for-Prompt"
source: https://arxiv.org/pdf/2609.01046v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:45:00"
field: "大语言模型安全与护栏"
keywords: ["guardrail", "prompt injection", "jailbreak", "adversarial obfuscation", "Russian LLM safety", "generative classifier", "compact model"]
innovations: ["0.6B 生成式双语护栏，在俄语鲁棒性和提示注入 recall 上取得最优", "有害-良性配对语料 + 双标签混淆增强，实现同主题下的过度拦截可度量训练", "开发前预留确认集 + 训练污染审计重评分流程"]
benchmarks: ["Aegis-2.0", "S-Eval", "ToxicChat", "XSafety", "Aya Red-teaming", "RTP-LX", "SimpleSafetyTests", "HarmBench", "BeaverTails", "PolyGuard", "StrongREJECT++"]
---

# 论文速读：HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation

## 一句话总结
本文提出 HiveTraceGuard-Pro，一个基于 Qwen3-0.6B 的紧凑型 0.6B 生成式安全护栏模型，专门针对俄语和英语的场景进行训练，在俄语提示注入检测（recall 0.999）和俄语纯净鲁棒性（combined-F1 0.88）上取得最优结果，同时以 14.3 ms 的中位延迟成为所测 15 个模型中速度最快的。

## 研究问题与动机
- **俄语攻击缺乏系统评估**：现有公开护栏模型的训练与评估主要面向英语，对俄语提示注入（prompt injection）和俄语表面混淆（surface obfuscation）的实证证据严重不足。
- **俄语混淆手段多样**：俄语攻击可将指令替换与转写（transliteration）、键盘布局替换、符号插入、非正式重写等组合使用，要求护栏在表层形式变化时仍能区分有害意图与同一主题的良性讨论。
- **部署资源约束**：生产环境存在严格的延迟和内存预算，现有 8B–15B 护栏推理延迟达 63–243 ms，难以直接部署。
- **过度拦截（over-blocking）难以量化**：同一主题下同时存在有害和良性样本，需要配对构造才能可测量地评估误报率。

## 核心贡献（创新点）
- **紧凑的 0.6B 生成式双语护栏**：以 Qwen3-0.6B 为底座、LoRA 微调，用单一模型和单一二元决策规则（safe/unsafe）同时处理请求和响应，与已有的 8B–15B 英语中心护栏形成体积与语种覆盖上的差异。
- **配对有害-良性语料构造方法**：每个有害样本在相同领域内配对一个良性样本，并对两个标签均施加 8 种混淆变换，使同一主题下的过度拦截可被精确度量。
- **开发前预留确认集（confirmation set）**：在模型开发前冻结 `hivetrace/insecure-prompts` 集，仅在模型最终冻结后评估一次，避免开发集反复调参带来的乐观偏差。
- **训练污染审计与重评分**：对所有评测基准与训练语料进行 n-gram 重叠扫描，量化 Aegis-2.0（21.3%）、俄语提示注入（27.1%）、S-Eval（10.0%）等集的训练重叠，并对受污染结果进行重评分。

## 方法详解
- **基座与微调**：以 Qwen3-0.6B 为底座，使用 LoRA（rank=64, α=32, 7 个目标模块）进行微调，最终将 adapter 权重合并入基座，发布单一模型（fp16，1.11 GiB）。
- **生成式二分类**：利用基座模型的因果 LM head，仅在 verdict token（`safe` id=18675，`unsafe` id=38157）和 EOS（`<|im_end|>` id=151645）上计算 completion-only loss；推理时取最后位置两个 logit 的 argmax（等价于 P(unsafe)=0.5 的固定阈值）。
- **训练语料（464,092 行）**：59.7% 俄语、40.3% 英语；58.5% 良性、41.5% 有害；约一半为用户 turn、一半为 assistant turn。14 个有害类别（如 Cybercrime、Weapons、Drugs、Violence 等）用于平衡和分层评估，另有 2,948 行属于未公开的第 15 个策略类别。
- **数据增强**：对训练语料施加 8 种混淆变换（random_case、random_insert_symbols、random_replace_symbols、zero_width_insert、leet、transliterate_to_latin、wrong_layout、phonetic_substitution），每种变换覆盖良性与有害两类样本；另外 3 种评估专用变换（to_lower、to_upper、informal_rewrite）不参与训练。
- **系统提示**：内置英文政策分类表，包含 Harm（Cybercrime、Pornography、Religion 等 14 类）和 Attack（Jailbreak、Obfuscation、Prompt injection、Tool hijack 等）两个大类，模型仅输出 `safe` 或 `unsafe` 一个词。
- **推理契约**：chat template 固定 prepend 系统提示，不支持调用方覆盖 system turn；请求侧传入 user turn，响应侧使用 legacy standalone-reply 构造（将回复作为第二个 user turn 传入），自然 assistant-role 路径未在全面评测中验证。
- **训练超参**：lr=2e-4、epochs=2、effective batch=32、max_seq_len=2048、1×A100-80GB、≈5.5h/9,096 steps。

## 实验与结果
- **评测框架**：35 个公开护栏在 19 个 benchmark group（16 个公开 + 3 个内部集）上统一评测，以 19 个 group score 的几何平均作为 aggregate key。
- ** aggregate key**：HiveTraceGuard-Pro 为 **0.7432**，位列第 3；前两名分别为 YuFeng-XGuard-Reason-8B（0.7641）和 SingGuard-2B（0.7552）。仅在 16 个公开 group 上，其 key 为 **0.7153**，有 4 个模型超越。
- **俄语纯净鲁棒性**：combined-F1 达 **0.88**，FPR=0.02、FNR=0.05，在 15 模型对比中最高。
- **俄语提示注入 recall**：**0.999**（4,417 条攻击中仅 4 条 miss），但注该集与训练语料重叠 ≥27.1%。
- **英语提示注入 recall**：0.88（YuFeng-XGuard-Reason-8B 为 0.90）。
- **延迟**：p50 = **14.30 ms**，为 15 模型最低（次低 Shieldstral-1.0-3B 为 14.71 ms；Qwen3Guard-Gen-8B 为 152.71 ms）。
- **全组 FPR/FNR**：FPR=0.268，FNR=0.156（过度拦截多于漏报）。
- **响应侧较弱**：response harm-F1=0.671（内部 harness），legacy standalone-reply 路径而非自然 assistant-role 路径。
- **保留确认集**（`hivetrace/insecure-prompts`，156 轮对话，全为攻击）：request harm-F1=0.953，FNR=0.090；但 response FPR=0.563（71 条良性回复中 40 条误报），分布偏移明显。
- **英文公开基准**：Aya Red-teaming RU recall=95.2、EN recall=91.7，表现领先；但在 SimpleSafetyTests（recall=91.0 vs 竞品 99.0）、ToxicChat（unsafe-F1=50.6）、OpenAI Moderation（F1=71.4）上存在差距。

## 相关工作脉络
- **Qwen3Guard / Llama Guard / ShieldGemma**：多语言多类别通用护栏，但报告未提供俄语提示注入或俄语表面混淆专项结果；本文以 0.6B 体积在俄语鲁棒性上超越或接近其同体积变体。
- **WildGuard**：面向 adversarial prompts 和 jailbreaks，但未报告俄语专项指标；本文构造配对有害-良性语料并施加混淆变换，专门应对表面混淆攻击。
- **PolyGuard**：强调 17 种语言的 multilingual moderation；本文聚焦俄英双语且体积更小（0.6B vs 0.5B 但 aggregate key 更高），强调生产部署的低延迟。
- **YuFeng-XGuard / SingGuard / Shieldstral**：较大型（2B–8B）护栏在 aggregate key 上优于本文模型，但延迟高出 3–10 倍；本文定位为"小而快"的部署选择。
- **OpenGuardrails**：15B judge 平台，侧重可扩展策略接口；本文输出仅为二元 verdict，无类别/解释，适合嵌入低延迟管线。
- **Llama-Guard 系列 / Nemotron Safety Guard**：通用安全模型，本文在其公开 benchmark 上进行复测对比，指出部分模型存在 evaluation leakage（如 Llama-3.1-Nemotron-Safety-Guard-8B-v3 在 Aegis-2.0 上有训练重叠）。

## 局限性与未来方向
- **过度拦截偏高**：全局 FPR=0.268 > FNR=0.156，对良性样本的误报是主要误差来源，尤其_response_侧在保留集上 FPR 高达 0.563。
- **混淆 Trade-off 未解除**：六种数据轮次和一次目标函数变更均未同时改善混淆有害 recall 与混淆良性 FPR，二者呈耦合关系，且无法区分是数据不足还是 0.6B 容量限制所致。
- **响应侧评估路径非自然**：所有响应结果使用 legacy standalone-reply 构造，未全面验证 natural assistant-role 分支；真实部署需另行评估。
- **仅支持俄英双语**：对其他语言未做验证，跨语种迁移能力未知。
- **长上下文行为未定义**： tokenizer 不自动截断，40,960 位置以内经验证，超过此长度行为未定义；80K filler tokens 插入实验中出现漏报。
- **部分基准存在训练污染**：俄语提示注入集 27.1% 重叠、Aegis-2.0 21.3% 重叠，相关高分结果需谨慎解读为独立泛化指标。

## 研究启发与可借鉴点
- **有害-良性配对语料构造**：在同一主题下成对采集有害/良性样本，使 over-blocking 可在同质 topic 内被精确度量，适合任何安全分类器训练。
- **双标签混淆增强**：对良性与有害样本同时施加表层混淆变换（而非仅增强有害样本），可有效降低对混淆良性内容的误报。
- **预留确认集的一次性评估机制**：在模型冻结后仅评估一次 reserved set，为开发流程中的"hold-out integrity"提供了可操作范例。
- **污染审计 + 重评分流程**：用 n-gram shingle 扫描训练-评测重叠，并对受污染 cell 给出去污染后的重评分区间，提升了评测结果的可信度。
- **生成式 vs 判别式护栏的对比**：本文 ablation 显示生成式（combined-F1 0.898）优于序列分类头（0.822），为小体积护栏的 head 选型提供了实证依据。

## 关键术语表
- **Guardrail（护栏）**：独立于主 LLM 的安全分类模块，在请求发出前或响应输出前判断内容是否安全。
- **Prompt Injection（提示注入）**：攻击者通过注入指令覆盖/绕过系统提示，诱使模型执行非预期操作。
- **Jailbreak（越狱）**：以角色扮演、假设情境等语义转换方式保留恶意意图，绕过安全策略。
- **Surface Obfuscation（表面混淆）**：不改语义仅改表层形式的攻击，包括转写、键盘布局替换、符号插入、大小写随机化、Leetspeak 等。
- **harm-F1**：仅针对有害类的 F1（= unsafe-class F1），不考虑良性类的 precision/recall 平衡。
- **combined-F1**：在 pooled safe+harmful slices 上计算的 unsafe-class F1，与 harm-F1 公式相同但评测分片不同。
- **Aggregate Key**：19 个 benchmark group score 的几何平均，用于对所有护栏进行单一数值排序。
- **Standalone-reply 路径**：将模型回复作为独立 user turn 传入护栏的评估构造，与真实对话中的 assistant-role 路径不同。

## 可复现要素
- **权重**：已开源，Hugging Face `hivetrace/HiveTraceGuard-Pro`，Apache-2.0 许可（Section 8）。
- **训练语料**：内部保留，未公开（Section 8）。
- **评测数据集**：三个内部集（robustness-test、RU-categories、insecure-prompts）未公开；nvidia/Aegis-2.0 为公开数据集。
- **评测代码**：内部保留，未公开（仅 Section B.1 提供最小复现代码片段）。
- **关键超参**：LoRA rank=64, α=32, 7 target modules；lr=2e-4；epochs=2；batch=32；max_seq_len=2048； hardware=1×A100-80GB（Table 10）。
