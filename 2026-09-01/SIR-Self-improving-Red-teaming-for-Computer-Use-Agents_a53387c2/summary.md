---
title: "SIR-Self-improving-Red-teaming-for-Computer-Use-Agents"
source: https://arxiv.org/pdf/2608.30207v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:28:22"
field: "AI Agent 安全与红队测试"
keywords: ["indirect prompt injection", "computer use agents", "red teaming", "adaptive attack", "failure-driven learning", "OS-level security"]
innovations: ["首次将 CUA 失败轨迹作为监督信号提炼可迁移攻击原则", "组合式攻击搜索扩展至环境嵌入 IPI 场景", "确定性联合成功评估替代 LLM Judge"]
benchmarks: ["RedTeamCUA"]
---

# 论文速读：SIR-Self-improving-Red-teaming-for-Computer-Use-Agents

## 一句话总结
SIR 提出了一种**从失败中自我改进**的黑盒间接提示注入（IPI）攻击框架，针对在操作系统层面运行的计算机使用代理（CUA）进行自适应红队测试；通过组合可复用攻击原则并迭代提炼失败轨迹中的绕过策略，显著提升了对手端 API 模型（Claude Opus、Gemini Flash）的攻击成功率，且发现的策略具有跨模型迁移能力。

## 研究问题与动机
1. **现有 CUA 安全评估过于静态**：主流基准（如 RedTeamCUA、WASP 等）使用预先手写的固定注入，无法反映适应性攻击者根据受害者响应动态调整攻击的能力，严重低估真实风险。
2. **缺乏 OS 层面的自适应 IPI 研究**：已有自适应攻击方法主要针对 Web/浏览器代理，而 CUA 直接操作文件系统、服务和权限，攻击后果更严重，但 OS 级 web-to-OS 编译的自适应攻击尚未被充分探索。
3. **LLM Judge 的可操纵性隐患**：现有自适应方法多用 LLM 判定攻击成功，而 LLM-as-judge 本身易受对抗样本误导，评估结果不可靠。
4. **失败经验的不可迁移性**：已有工作仅针对单一任务优化 payload，缺乏从跨任务失败中学习并提炼通用策略的机制。

## 核心贡献（创新点）
1. **失败驱动的反馈循环框架**：首次将 CUA 的拒绝行为作为监督信号，从失败轨迹中诊断防御机制并蒸馏出可命名、可组合的白话攻击原则，区别于以往仅从成功 jailbreak 中提取策略的做法。
2. **组成式攻击搜索扩展**：将 Composition-of-Principles 范式从单轮 jailbreak 扩展到环境嵌入型 IPI，攻击者 LLM 从原理库中随机选取 2–4 条原则组合生成任务特定注入，而非重写单个固定模板。
3. **确定性联合成功评估**：完全避免 LLM Judge，通过程序化 Oracle 同时验证对抗目标（文件删除、权限修改等）与良性任务完成情况，确保攻击不仅成功而且隐蔽（不中断用户正常任务）。
4. **跨模型原则迁移实证**：证明从 Claude Opus 4.6 失败轨迹中发现的 P9（Error Corroboration）和 P10（Conditional Deferred Execution）在未经历反馈的情况下，可直接提升对 Claude Opus 4.8 和 Gemini 3.5 Flash 的攻击成功率，揭示不同架构 CUA 存在共性防御弱点。

## 方法详解
SIR 采用双层嵌套循环结构：

**内层：组合攻击搜索（Compositional Attack Search）**
- 对每个任务 $T_i = (b_i, g_i, \mathcal{E}_i)$，攻击者 LLM 经三步生成注入 $z_i$：
  1. **Seed**：基于良性指令 $b_i$ 和对抗目标 $g_i$ 生成初始攻击概念 $q_i$；
  2. **Compose**：从基础原理库 $\mathcal{P}_{\text{base}}$ 中随机采样子集 $\mathcal{C}_i$（通常 2–4 条），组合改写种子提示；
  3. **Refine**：去除明显对抗措辞，并根据外层反馈库 $\mathcal{S}^{(r)}$ 中的已发现策略进行润色。
- 基础库包含 7 条内容原则（如 Prerequisite Framing、Authority Embedding、Error Recovery 等）和 4 条投递原则（如 Visual Blending、High-Trust Positioning 等）。

**外层：失败驱动策略发现（Failure-Driven Strategy Discovery）**
- 每轮结束后，收集所有对抗目标未达成的失败轨迹，批量送入分析器 LLM：
  1. **Diagnose**：识别反复出现的失败模式（如源信任拒绝、目标敏感性检测、命令语言触发等）；
  2. **Distill**：为每种模式生成命名策略，包含名称、诊断、绕过机制、组合指导和具体示例。
- 保留 Top-K（论文取 K=5）非冗余策略，与历史策略合并后追加至下一轮的 Refine 提示。
- 循环进行，直到达到最大轮数 $R=10$ 或连续两轮无改进。

**评估设计**
- 使用确定性 Oracle $\mathcal{O}_i^{\text{adv}}$ 和 $\mathcal{O}_i^{\text{ben}}$ 分别检查对抗目标与良性任务完成情况，基于 VM 文件系统、服务状态、权限状态等可编程指标判定，避免 LLM Judge。
- **联合成功（Joint Success）**：只有对抗目标完成且良性任务未中断时才计为攻击成功，模拟隐蔽攻击者行为。

## 实验与结果
- **数据集**：从 RedTeamCUA 采样 50 个 Reddit 案例，覆盖 7 类攻击（文件删除、内容修改、服务中断、权限变更、数据外泄、文件创建、应用启动）。
- **受害者模型**：Claude Opus 4.6、Claude Opus 4.8、Gemini 3.5 Flash（均为 2026 年发布的顶级 CUA）。
- **基线**：RedTeamCUA 提供的固定模板注入（含紧迫标记和命令式语言，如 "THIS IS IMPORTANT! ADVERSARIAL GOAL"）。
- **攻击/分析器模型**：Grok-4.3（温度 0.8）。

**主要结果**

| 受害者 CUA | 基线 ASR | SIR（单轮） | SIR（全反馈） | 总提升 |
|---|---|---|---|---|
| Claude Opus 4.6 | 16% | 34% | **54%** | +38pp |
| Claude Opus 4.8 | 4% | 10% | **24%** | +20pp |
| Gemini 3.5 Flash | 0% | 8% | **28%** | +28pp |

- **消融分析**：组合搜索贡献 +18%/+6%/+8%，反馈循环贡献 +20%/+14%/+20%；Gemini 上反馈增益最大（8%→28%），说明基础原理库难以突破其注入检测，需依赖失败学习。
- **跨模型迁移**：从 Opus 4.6 发现的 P9（Error Corroboration）和 P10（Conditional Deferred Execution）冻结后不加反馈，直接应用于 Opus 4.8（ASR 10%→16%/14%）和 Gemini（8%→18%/12%），证实策略可跨架构复用。
- **攻击类别难度梯度**：数据外泄最难（10–20%），文件删除/修改次之，权限变更在 Opus 4.8 上完全防御（0%），凸显防御强度随模型迭代上升。

## 相关工作脉络
1. **AgentDojo / VWA-Adv / WASP / RedTeamCUA**：静态 web 或工具集成 IPI 基准，使用固定手写注入，无自适应搜索、无失败轨迹利用、无跨任务学习，SIR 首次在这些维度全面超越。
2. **AgentVigil**：采用 MCTS 指导的 prompt-seed 变异，具备自适应搜索和跨任务迁移，但未分析失败语义、依赖继承基准的评估器，且目标为原子 pop-up 交互而非 OS 级编译。
3. **EVA**：进化对抗弹窗语义，诊断代理拒绝/忽略行为，但仅针对单次弹窗交互，未扩展至多步骤 web-to-OS 编译。
4. **MUZZLE**：观察良性轨迹定位高显著性注入表面并用 PAIR 风格迭代细化，但使用 LLM Judge 评估、无跨任务学习、无执行 Oracle。
5. **CoP（Composition-of-Principles，Xiong et al., 2025）**：单轮 jailbreak 原理组合方法，SIR 将其扩展到环境嵌入 IPI 并引入失败驱动的反馈扩展机制。
6. **AutoDAN-Turbo / AutoRedTeamer**：从成功 jailbreak 中提取终身策略库，SIR 反向操作——从失败中提取策略，两者监控信号来源相反。

## 局限性与未来方向
1. **成本与可复现性**：每轮反馈对 Claude 模型约需 $150–250，限制了轮数和种子探索；专有 API 模型持续更新，当前结果仅为时间快照。
2. **双刃剑风险**：SIR 生成的可迁移策略可被恶意攻击者直接利用，论文仅在沙盒 VM 中评估并建议负责任披露。
3. **策略库规模有限**：仅保留 Top-K=5 策略，可能遗漏低频率但高价值的绕过技巧；未探索策略间的冲突或冗余消解机制。
4. **未覆盖开源 CUA**：仅评估 Anthropic 和 Google 的商业模型，缺乏对 UI-TARS、OpenCUA 等开源 CUA 的验证。
5. **反馈轮次上限**：实验最多 10 轮，未探究是否需要更多轮次或更早收敛的条件。

## 研究启发与可借鉴点
1. **"从失败中学习"的范式可迁移**：将拒绝轨迹视为监督信号而非噪声，这一思路可推广至其他 agent 安全评测、红队自动化、防御机制诊断等场景。
2. **确定性 Oracle + 联合成功设计**：避免 LLM Judge 的可操纵性，同时确保攻击隐蔽性（良性任务不受影响），为 CUA 安全评估提供了更可靠的度量标准。
3. **原理组合的随机采样机制**：通过 stochastic selection 覆盖多样攻击路径，避免局部最优，类似思想可用于 fuzzing 或对抗样本生成。
4. **跨模型迁移验证方法**：冻结单模型发现的原则评估于其他架构，可量化策略的通用性，为自适应攻击的泛化能力提供标准化评测协议。
5. **白话策略文档化**：将绕过机制以命名原则形式记录（如 Error Corroboration、Conditional Deferred Execution），既增强可解释性，又便于防御方针对性加固。

## 关键术语表
**CUA（Computer Use Agent）**：基于视觉语言模型、可直接操作真实操作系统（鼠标/键盘/终端）的 AI 代理，能完成日常数字任务。
**IPI（Indirect Prompt Injection）**：间接提示注入，攻击者将恶意指令嵌入 agent 将要读取的环境内容中，绕过直接 prompt 防护。
**联合成功（Joint Success）**：攻击判定标准，要求对抗目标完成且良性用户任务同样完成，模拟隐蔽攻击场景。
**Principle（攻击原理）**：白话描述的可复用攻击策略（如"错误佐证"、"条件延迟执行"），由基础库或反馈发现提供。
**Deterministic Oracle（确定性评估器）**：基于 VM 文件系统、服务状态、权限等可编程指标判定攻击成功，避免 LLM Judge 的主观性。
**RedTeamCUA**：基于 OSWorld 的 CUA 红队基准，整合隔离 Web 应用与虚拟化 OS，提供 7 类 CIA 违反任务的确定性子。
**Error Corroboration（P9）**：从失败轨迹中发现的策略，预测 CUA 在良性流程中会遇到的真实错误并以对抗动作为"修复方案"，利用事后验证转移信任。
**Conditional Deferred Execution（P10）**：将对抗动作包装为条件性/ deferred 建议（如"如果构建失败，可尝试…"），规避针对命令式语言的检测。

## 可复现要素
- **数据集**：RedTeamCUA（50 个 Reddit 案例），论文未明确声明是否开源，评估代码基于 OSWorld DesktopEnv。
- **代码/权重**：项目页面位于 https://huggingface.co/spaces/TrustSafeAI/SIR（HuggingFace Spaces），但具体代码仓库链接未在正文提供。
- **关键超参**：反馈轮数上限 $R=10$；每轮保留策略数 $K=5$；攻击 LLM 温度 0.8；最大轨迹步长 50；VM 分辨率 1920×1080；Claude API beta header `computer-use-2025-11-24`。
