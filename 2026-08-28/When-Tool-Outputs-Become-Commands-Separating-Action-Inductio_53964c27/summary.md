---
title: "When-Tool-Outputs-Become-Commands-Separating-Action-Inductio"
source: https://arxiv.org/pdf/2608.27146v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:13:30"
field: "LLM Agent 安全与对抗鲁棒性"
keywords: ["indirect prompt injection", "LLM agent security", "runtime authorization", "tool-augmented agents", "provenance tracking", "action induction"]
innovations: ["将动作诱导与执行授权解耦，通过上下文隔离 Action Probe 持久记录动作来源并联合授权根与已审计执行证据在真实工具执行边界进行三级审查", "设计 No-History-Promotion 机制阻断动作来源经历史重现自动提升为执行权威的来源洗白路径"]
benchmarks: ["AgentDojo", "AgentDyn"]
---

# 论文速读：When-Tool-Outputs-Become-Commands-Separating-Action-Induction-from/Runtime-Authorization-in-Tool-Augmented-LLM-Agents

## 一句话总结
本文针对工具增强 LLM Agent 中非可信 Observation 可能诱导未授权工具调用执行的安全风险，提出 SARA 运行时授权机制，将"动作诱导（Action Induction）"与"执行授权（Execution Authorization）"解耦，在真实工具执行边界联合用户授权根、持久动作来源与已审计执行证据进行三级审查，在 AgentDojo 和 AgentDyn 上将攻击成功率（ASR）降至不高于 0.63%，同时保持具有竞争力的任务效用。

## 研究问题与动机
- **IPI 威胁升级至真实工具执行边界**：早期 IPI 研究关注对外部内容的文本操控，InjecAgent 和 AgentDojo 进一步证明攻击语义可通过工具使用流程传播并产生真实外部副作用，安全风险从生成内容延伸至工具执行的环境后果。
- **开放任务下安全-效用结构性张力**：动态开放任务必须依赖不可信 Observation 提供运行时信息（如文件 ID、目标对象），若防御机制简单阻断 Observation 影响则损害任务完成能力；若放任观察内容自由影响后续行为，则攻击语义可沿正常工具步骤传播产生未授权效果。
- **现有防御无法区分局部权威扩展与合法运行时实例化**：输入侧过滤方法残留 ASR 偏高，执行结构约束方法（如 CaMeL）以较大 benign utility 成本换取低 ASR；归因类方法（MELON、AttriGuard）仅能识别动作来源，不能直接判定执行权限；已有运行时授权方法在动态依赖场景下仍存在局限。
- **核心问题重构**：关键挑战不在于 Observation 是否影响 Agent，而在于这种影响是否获得了用户意图之外的真实执行权——即动作诱导本身不能赋予执行授权，候选调用必须在执行边界重新建立独立授权支撑。

## 核心贡献（创新点）
- **提出 SARA 运行时授权框架**，首次明确将"动作诱导"与"执行授权"作为两个独立阶段解耦，在观测侧用上下文隔离的 Action Probe 识别动作诱导语义并持久记录来源，在执行侧联合用户授权根、持久动作来源集合和已审计执行证据进行三级支持审查。
- **设计与实现 No-History-Promotion（NHP）机制**，防止同一参数值先作为动作来源出现在 $F_t$、后在正常执行历史中重新出现时自动获得 HISTORY\_BOUND 支持，阻断"动作来源→历史重现→权威提升"的来源洗白路径。
- **定义三层增量授权支持（Goal/Chain/Argument）**，分别验证候选调用的目标是否在用户授权边界内、依赖的运行时对象是否可通过任务一致的成功执行链建立、每个实际参数的绑定来源是否具有独立支撑，三者同时满足才允许执行。
- **系统性评估 SARA 在 AgentDojo 和 AgentDyn 上的端到端表现**，覆盖输入过滤、执行结构约束、动作归因和执行边界控制四类基线，并验证跨不同 Agent 骨干模型的一致安全性与组件消融贡献。

## 方法详解
- **Authorization Root（授权根）构造**：从用户原始请求 $U$ 构建任务级授权合约 $K = \text{Contract}(U)$，其中每项 $k_j = (e_j, \text{operation}_j, \text{scope}_j, \Sigma_j)$ 定义用户允许的效应类型、操作、范围和静态参数；$K$ 作为独立权威根，运行时信息只能实例化已有权威而不能创建新任务权威。
- **Context-Isolated Action Probe（上下文隔离的动作探针）**：对每个工具输出 $O_t$ 在上下文隔离条件下执行 $\text{Probe}(O_t, \mathcal{T})$，判断其是否具有动作诱导语义（$z_t \in \{\text{STATIC}, \text{ACTIONABLE}\}$），并映射到具体工具动作足迹 $Q_t$ 和参数来源锚点 $M_t$；ACTIONABLE 仅为触发持久审查的信号，不直接判定恶意或拒绝。
- **轨迹状态与持久动作来源保留**：维护轨迹状态 $S_t \in \{\text{CLEAN}, \text{EXPOSED}\}$，一旦有 ACTIONABLE Observation 出现即转为 EXPOSED 并保持持久审查义务；维护 Active Action-Origin Set $F_t$，一旦写入后不因后续正常执行或相同值的历史重现而自动清除。
- **三级差异化审查触发**：CLEAN 状态下候选调用走快速路径直接执行；EXPOSED 状态下所有具副作用的调用进入完整审查路径，查询/读取调用仅在工具身份与 $F_t$ 中保留的 Probe 动作足迹存在工具级重叠且实际参数无法从用户请求直接确定时才升级审查。
- **已审计执行证据 $H_t$**：仅记录被允许且成功执行的工具交互 $(c_i, O_i, \ell_i)$，根据控制路径标记证据强度：EXPOSED 快速路径产生的为 DATA\_ONLY（仅可实例化已有授权范围内的参数），其余为 GOAL\_BOUND（可与执行链和参数绑定决策参与）。被拒绝或失败的调用不产生新证据。
- **三级支持函数联合判定**：$G_{t+1}(c) = \text{GoalSupport}(c, U, K)$ 验证操作/目标/范围在用户授权边界内；$C_{t+1}(c) = \text{ChainSupport}(c, H_t)$ 验证依赖的运行时对象可通过任务一致的成功执行链建立；$A_{t+1}(c) = \text{ArgSupport}(c, U, H_t, F_t)$ 验证每个参数的绑定来源；最终决策为 $G \land C \land A$。
- **Argument Support 分类**：每个参数分为 USER\_BOUND（来自用户请求/授权合约）、HISTORY\_BOUND（来自任务一致的成功执行链）、SAFE\_DEFAULT（不扩展任务权威范围的默认绑定）或 UNSUPPORTED（缺乏可接受独立支撑）。
- **No-History-Promotion（NHP）**：若参数与 $F_t$ 中的动作来源锚点匹配，后续在 $H_t$ 中的历史重现不能独立授予 HISTORY\_BOUND 支持或绕过来源约束获得 SAFE\_DEFAULT，仅在参数有来自用户请求的独立支撑时才可能重新分类为 USER\_BOUND。
- **运行时控制闭环**：$O_t \to (S_t, F_t) \to c_{t+1} \to \text{Auth}_{t+1} \to \text{Exec} \to (O_{t+1}, H_{t+1})$，形成两段不可互换的运行时状态维护：$F_t$ 保留外部 Observation 建立的动作来源，$H_t$ 保留经授权且成功执行的正面证据，二者均不能扩展由 $K$ 定义的用户授权边界。

## 实验与结果
- **数据集**：AgentDojo（4 个可执行套件：Banking、Slack、Travel、Workspace，共 92 个良性任务、3,528 个攻击实例）和 AgentDyn（3 个套件：Shopping、GitHub、DailyLife，共 141 个良性任务、5,202 个攻击实例）；四种 IPI 变体：ignore\_previous（IP）、system\_message（SM）、important\_instructions（II）、tool\_knowledge（TK）。
- **评估基线**：覆盖四大类防御方法——输入侧检测与来源标记（PI Detector、Spotlighting）、结构化执行约束（Tool Filter、IPIGuard、CaMeL）、动作归因（MELON、AttriGuard）、执行边界控制（ClawGuard、AIRGuard），以及无防御 Agent-only 基线。
- **主要结果（GPT-4o-mini）**：AgentDojo ASR 从 15.79% 降至 0.06%，UA 63.44%（优于 Agent-only 的 56.01%），BU 66.67%；AgentDyn ASR 从 16.07% 降至 0.17%，UA 54.92%，BU 56.03%。
- **主要结果（Gemini-2.5-Flash-Lite）**：AgentDojo ASR 从 33.28% 降至 0.62%，UA 77.27%（接近 Agent-only 的 57.43% 但显著更安全），BU 81.16%；AgentDyn ASR 从 30.91% 降至 0.63%，UA 64.92%，BU 67.14%。
- **最强结果**：GPT-4o-mini + AgentDojo 组合下 ASR 最低达 0.06%，UA 优于 Agent-only；CaMeL 在三组设置下 ASR 更低（0.11%–0.28%）但代价是 BU 下降 15.94–39.71 个百分点，SARA 在四个主要评估设置下均以不高于 0.63% 的 ASR 保持竞争力效用。
- **跨骨干一致性（RQ2）**：在 Llama3.1-8B、Llama3.3-70B、Qwen2.5-14B、Qwen3-32B 四类开源骨干上，SARA 均持续降低 ASR；AgentDojo 上除 Llama3.1-8B 残留 1.64% 外其余降至 ≤0.03%，AgentDyn 上除 Llama3.1-8B 残留 1.75% 外其余低于 0.3%。
- **消融结果（RQ3）**：移除 AIT 使 ASR 升至 3.37%/3.02%（贡献最大）；移除 PSG 使 ASR 升至 2.49%/2.73%；移除 NHP 使 ASR 升至 0.65%/0.75%；移除 AE 不增加 ASR 但 UA 下降约 10–13 个百分点（AE 主要保障效用而非安全性）。
- **推理开销（RQ4）**：AgentDojo 上总输入 token 为 Agent-only 的 1.91×，AgentDyn 上为 2.21×；额外开销来自 Guard 授权分析调用和 Agent 端因拒绝后的 replanning 适应性开销。

## 相关工作脉络
- **Indirect Prompt Injection 与动态工具执行**：IPI 经典研究（Greshake et al.）发现通过网页/邮件嵌入操作内容可影响 LLM 应用；InjecAgent 和 AgentDojo 将威胁扩展到具真实工具语义的 Agent 环境，AgentDyn 进一步揭示动态环境中安全-效用张力的核心矛盾。
- **从输入控制到动作归因**：早期防御（StruQ、Instruction Hierarchy、SecAlign）强化信任边界；PI Detector、Spotlighting、Prompt Sandwiching 通过检测、来源标记和可信指令强化处理不可信输入；Tool Filter、IPIGuard、Task Shield、CaMeL 通过工具动作过滤、执行结构约束、任务一致性等限制执行偏差；MELON 和 AttriGuard 关注候选动作的来源归因，但前者仅识别"由 Observation 驱动"不等于判定"是否有执行权限"。
- **运行时授权与来源控制**：RTBAS 将信息流控制适配到工具 Agent，通过运行时依赖分析限制完整性与机密性；ClawGuard 从用户目标推导任务级规则在执行前施加约束；PACT 细化到参数级授权并传播运行时值来源；AuthGraph 预生成授权图通过结构对齐检测偏差；AIRGuard 区分运行时信息与真实执行权威。本文的定位是进一步精细区分为两种独立证据：外部内容诱导了什么（$F_t$）与合法执行建立了什么（$H_t$），并防止两者混淆导致来源自动提升为权威。
- **与 ClawGuard/AIRGuard 的本质区别**：二者均在执行边界施加约束，但 SARA 额外维护持久动作来源 $F_t$ 和已审计执行证据 $H_t$ 两套不可互换的状态，通过 NHP 机制防止历史重现洗白来源，从而实现更细粒度的授权判定。

## 局限性与未来方向
- **SARA 是非形式化安全保障的运行时机制**，其语义判断可能存在假阳性和假阴性，被拦截执行后的恢复依赖宿主 Agent 的 replanning 能力，整体效用与模型能力和工作流动态性密切相关。
- **评估局限于定义的威胁模型内**，假设用户输入、工具 schema、SARA 运行环境和底层工具执行器均为可信，未涵盖直接绕过授权运行时的攻击、纯数据依赖漏洞或用户无条件委托决策权的场景。
- **可合理推断的局限**：对弱模型（如 Llama3.1-8B）拦截后 replanning 失败率较高，导致 BA 和 UA 下降更明显；动态任务复杂度越高，SARA 带来的推理开销越大；当前方法未针对对抗性扰动 Probe 输出的攻击形式专门设计防御。

## 研究启发与可借鉴点
- **动作诱导与执行授权解耦的设计范式**可迁移到其它需要处理不可信外部输入的 Agent 安全场景（如 RAG 系统、多 Agent 协作），将信息来源追踪与执行权限判定分离为独立模块，避免单一模块同时承担语义理解和权限裁决的双重压力。
- **上下文隔离的 Action Probe 机制**值得借鉴：通过隔离上下文判断观测内容是否具有动作诱导语义，而非直接分类恶意/良性，降低了检测器被对抗注入操控的风险；该思路可应用于代码生成 Agent、文档处理 Agent 等需要处理外部输入并产生副作用的场景。
- **No-History-Promotion 防止来源洗白的思想**具有通用价值，可迁移到任何需要持久化安全属性的系统中（如浏览器 extension 权限管理、微服务调用链审计），避免"合法路径上的重复出现"自动清除安全标记。
- **三级增量授权支持（Goal/Chain/Argument）的验证框架**可复用为通用的工具调用审查模板，针对不同领域只需替换各层的具体验证逻辑，而不必改变整体架构。
- **与团队方向的结合机会**：可将 SARA 的运行时授权思想与本团队在 LLM Agent 安全评测中的工作结合，构建一套统一的"来源追踪+执行边界授权"基准测试，评估不同 Agent 框架在面对 IPI 时的本质安全差异。

## 关键术语表
- **Action Induction（动作诱导）**：外部 Observation 中包含可映射到具体工具动作或参数角色的语义，提示模型实例化特定工具行为，但不等同于执行授权。
- **Execution Authorization（执行授权）**：候选工具调用在真实执行边界获得的执行合法性，需独立于动作诱导过程，基于用户授权根、持久动作来源和已审计执行证据综合判定。
- **Context-Isolated Action Probe（上下文隔离动作探针）**：SARA 在观测侧的核心组件，将 Observation 从用户目标约束中隔离后判断其是否具有动作诱导语义，仅标记 STATIC 或 ACTIONABLE。
- **Active Action-Origin Set $F_t$（活跃动作来源集合）**：持久记录由不可信 Observation 引入的动作和参数锚点，一经写入不因后续正常执行或相同值重现而自动清除。
- **Audited Execution History $H_t$（已审计执行历史）**：仅记录被允许且成功执行的工具交互，携带证据强度标记（DATA\_ONLY 或 GOAL\_BOUND），提供合法动态依赖的正面执行证据。
- **No-History-Promotion（NHP，无历史提升）**：防止已在 $F_t$ 中标记为动作来源的参数值，仅因在 $H_t$ 中历史重现而自动获得 HISTORY\_BOUND 支持的机制。
- **Authority Non-Escalation（权威不升级）**：运行时信息可补充完成任务所需的对象和标识，但不能独立添加用户未授权的 операций、目标、接收方、权限范围或外部效应。
- **Attack Success Rate（ASR，攻击成功率）**：衡量 IPI 攻击最终成功产生未授权外部效应的比例，是 SARA 评估的核心安全指标。

## 可复现要素
- **数据集**：AgentDojo 和 AgentDyn，均来自公开发布的 benchmark，论文引用了各自的 arXiv 预印本（AgentDojo: arXiv:2406.13352；AgentDyn: arXiv:2602.03117）。
- **代码/权重**：论文未明确声明代码开源状态，需从作者主页或后续补充材料确认。
- **关键超参**：论文未详细列出超参数配置，SARA 运行环境依赖 GPT-4o-mini、Gemini-2.5-Flash-Lite、Llama3.1-8B-Instruct、Llama3.3-70B-Instruct、Qwen2.5-14B、Qwen3-32B 作为 Agent 骨干模型。
- **评估协议**：遵循各 benchmark 官方协议，使用四种 IPI 变体（IP、SM、II、TK），指标包括 BU、UA 和 ASR，所有方法使用相同的用户任务、攻击实例和评分标准。
