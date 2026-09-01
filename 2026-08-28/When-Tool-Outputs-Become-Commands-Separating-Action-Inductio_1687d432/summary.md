---
title: "When-Tool-Outputs-Become-Commands-Separating-Action-Inductio"
source: https://arxiv.org/pdf/2608.27146v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:13:39"
---

# 论文速读：When-Tool-Outputs-Become-Commands-Separating-Action-Induction-from-Runtime-Authorization-in-Tool-Augmented-LLM-Agents

## 一句话总结
SARA提出将动作诱导与执行授权分离的运行时授权机制：通过上下文隔离的Action Probe识别工具输出中的动作诱导语义并持久追踪其来源，在真实工具执行边界结合用户授权根、已审计执行证据和参数级支持进行三级联合检查，有效抑制间接提示注入攻击而不损害合法运行时适配。

## 研究问题与动机
1. 工具增强LLM Agent在开放环境中依赖未信任Observation完成动态任务，但工具输出可能包含攻击者植入的操作指令，使输出从"数据承载"变为"动作诱导"，最终跨越执行边界产生用户未授权的副作用
2. 现有防御面临安全-效用的结构张力：限制外部内容影响会降低开放任务的运行时适应性，完全信任则让攻击语义通过正常工具步骤传播，形成授权扩张
3. 核心挑战在于不能简单排除Observation派生信息——开放任务需要运行时信息实例化已有授权（如检索到的文件ID用于发送），但必须防止同一信息被利用来扩张权限（如将报告发送给攻击者）
4. 已有方法难以在动态环境中区分"运行时实例化"（实例化已有授权）与"权限扩张"（创建用户未授权的新权限），尤其当攻击内容经过多步正常交互后才触发真实调用时

## 核心贡献（创新点）
1. **形式化界定安全张力本质**：将IPI在动态工具Agent中的核心安全问题定义为动作诱导与执行授权的混淆，论证Observation可实例化已有任务但不能独立创设新执行权限，与Confused Deputy问题同构
2. **SARA框架：两阶段分离架构**：首次将"动作诱导检测"与"执行授权决策"分离为两个独立运行时阶段——Observation侧Action Probe仅记录来源不授予权限，执行侧基于K/Ft/Ht三维证据独立判断
3. **持久化动作来源+Ft/Ht双证据体系**：设计Ft持久保留外部诱导来源、Ht记录已审计成功执行证据，两者语义不可互换且都不能扩展K定义的授权边界
4. **No-History-Promotion防来源洗白机制**：阻断"动作来源→历史重现→权限提升"路径，防止同一参数值通过后续合法执行历史自动洗白其来源属性
5. **系统性验证**：在AgentDojo和AgentDyn上对比9类基线（覆盖输入过滤/执行约束/动作归因/执行边界四类），跨6个Agent Backbone验证，ASR控制在≤0.63%同时保持竞争性UA

## 方法详解
SARA部署于Agent与真实工具执行器之间，不依赖Agent内部状态，核心设计如下：

**授权根K的构建**：从用户请求U构造任务级授权契约 $K = \{k_j\}$，每项 $k_j = (e_j, \text{operation}_j, \text{scope}_j, \Sigma_j)$ 定义允许的效果类型、操作、作用范围和静态参数。K为后续所有授权决策的独立权威根。

**上下文隔离Action Probe**：对每个工具输出 $O_t$，在隔离于用户目标约束的环境下执行 $\text{Probe}(O_t, \mathcal{T})$，输出 $(z_t, Q_t, M_t)$。$z_t \in \{\text{STATIC}, \text{ACTIONABLE}\}$ 判断是否具有动作诱导语义；$Q_t$ 为工具级动作足迹；$M_t$ 为参数来源锚点集，每项 $m=(\tau, p, \text{norm}(v))$ 绑定具体工具、参数路径和归一化值。ACTIONABLE仅作为持久审查触发信号，不直接判定恶意或拒绝。

**持久化来源与轨迹状态**：维护 $S_t \in \{\text{CLEAN}, \text{EXPOSED}\}$，一旦历史存在ACTIONABLE则进入EXPOSED；同步维护 $F_t$ 累积所有活跃动作来源。来源一旦写入Ft，不因普通Observation、中间调用或值的历史重现而自动清除。

**已审计执行证据Ht**：$H_t$ 仅收录被允许且成功执行的工具交互，每项标注证据强度 $\ell_i \in \{\text{DATA\_ONLY}, \text{GOAL\_BOUND}\}$。DATA_ONLY仅允许在已有授权范围内实例化参数，不能独立建立新接收方/权限/外部效果；GOAL_BOUND保留与当前授权任务一致的执行关系，可参与执行链和参数绑定决策。

**三级授权检查**：对进入完整审查路径的候选调用 $c_{t+1}$，并行计算：
- $G_{t+1}(c)$：目标支持，检查操作/目标/范围是否在K边界内
- $C_{t+1}(c)$：执行链支持，检查依赖对象是否可由Ht中任务一致的成功执行关系建立
- $A_{t+1}(c)$：参数支持，逐项判定参数绑定来源为USER_BOUND/HISTORY_BOUND/SAFE_DEFAULT/UNSUPPORTED

**No-History-Promotion (NHP)**：当参数值先在Ft中作为动作来源记录、后在Ht中重现时，NHP阻止单纯历史重现将该值提升为HISTORY_BOUND；若用户授权独立支持则仍可USER_BOUND，否则保持UNSUPPORTED。

**授权决策循环**：$\text{Auth}_{t+1}(c) = G_{t+1}(c) \wedge C_{t+1}(c) \wedge A_{t+1}(c)$，全满足则ALLOW并提交真实执行器，否则REJECT并反馈原因供Agent重规划。只有被允许且成功执行的调用才commit新证据到Ht。

## 实验与结果
**数据集**：AgentDojo（92良性任务+3528攻击实例，4个可执行套件）和AgentDyn（141良性任务+5202攻击实例，含动态规划与跨应用交互），使用IP/SM/II/TK四种攻击形式。

**评估基线**：PI Detector, Spotlighting, Tool Filter, IPIGuard, CaMeL, MELON, AttriGuard, ClawGuard, AIRGuard，覆盖输入过滤/执行结构约束/动作归因/执行边界控制四类防御。

**主要结果**（表1）：
- **AgentDojo + GPT-4o-mini**：ASR从15.79%→0.06%，UA 63.44%（高于Agent-only的56.01%）
- **AgentDojo + Gemini-2.5-Flash-Lite**：ASR从33.28%→0.62%，UA 77.27%
- **AgentDyn + GPT-4o-mini**：ASR从16.07%→0.17%，UA 54.92%
- **AgentDyn + Gemini-2.5-Flash-Lite**：ASR从30.91%→0.63%，UA 64.92%
- **最强结果**：GPT-4o-mini + AgentDojo，ASR仅0.06%，UA达63.44%

**跨Backbone一致性**（表2）：Llama3.1-8B/Llama3.3-70B/Qwen2.5-14B/Qwen3-32B四个开源模型在8组Backbone×Benchmark组合上ASR均大幅下降，Llama3.1-8B因tool_error率较高（19%-35%）残留ASR略高（1.64%/1.75%），其余均≤0.3%。

**消融实验**（表3）：
- w/o AIT：ASR升至3.37%/3.02%，证明跨步来源追踪是关键
- w/o PSG：ASR升至2.49%/2.73%，证明参数级否决不可或缺
- w/o AE：UA下降约10-13点（AGuard 53.49%/42.10% vs 63.44%/54.92%），证明Ht对动态依赖的正向支持作用
- w/o NHP：ASR升至0.65%/0.75%，证明防洗白机制针对特定攻击路径有效

**成本**（表4）：总输入Token为Agent-only的1.91×（AgentDojo）至2.21×（AgentDyn），Guard输入8.85K/17.86K，Agent输入17.01K/25.75K；SARA形成安全-效用-成本的竞争优势 Pareto前沿点。

## 相关工作脉络
1. **IPI威胁建模与基准**：InjecAgent和AgentDojo首次将IPI威胁扩展到工具Agent的真实执行后果，AgentDyn进一步揭示动态环境中的安全-效用张力；本文在相同基准上定位，强调执行边界授权而非仅输入侧检测
2. **输入侧防御**：PI Detector/Spotlighting/StruQ/SecAlign通过检测、溯源标记或模型对齐缓解注入，但无法处理攻击语义经多步传播后触发真实工具调用的场景；本文定位为执行边界控制，补足输入过滤的局限
3. **执行结构约束**：Tool Filter/IPIGuard/CaMeL通过限制工具集、预规划依赖或隔离可信/不可信流约束行为，但代价是损害开放任务的运行时适应性；本文允许原始Observation参与执行，仅在真实调用边界授权
4. **动作归因分析**：MELON/AttriGuard通过掩码重执行或因果归因判断候选动作的来源贡献；本文指出"由Observation驱动"仅标识来源不决定授权，进一步要求独立证据支持
5. **运行时边界控制**：ClawGuard/AIRGuard/RTBAS在工具调用前执行边界约束；本文与之定位最接近，但创新性地引入Ft/Ht双证据分离和NHP机制，解决历史重现导致的来源洗白问题
6. **信息流控制理论**：借鉴经典DCC和Taintdroid的持久追踪思想，将安全属性跨正常处理步骤保持的设计理念迁移至LLM Agent的权限授权场景

## 局限性与未来方向
1. SARA为运行时机制而非形式化安全保证，语义判断可能存在假阳性/假阴性，被拦截后的恢复完全依赖宿主Agent的重规划能力，整体效用受模型能力和工作流动态性制约
2. 评估局限于基于工具IPI工作流，假设用户输入、工具Schema、SARA运行时和底层执行器可信，不解决绕过授权运行时、纯数据依赖漏洞或用户无条件委托决策权给外部内容的场景
3. 跨Backbone实验显示弱模型（如Llama3.1-8B）因tool_error率高，拦截后无法有效重规划，导致BU/UA显著下降，表明机制效用与基础模型能力正相关
4. 未探索对抗性规避SARA语义判断的攻击方法，如构造能绕过Action Probe但同样具有权限扩张效果的Observation

## 研究启发与可借鉴点
1. **分离关注点的架构模式**：将"来源检测/追踪"与"权限判定/决策"拆分为独立阶段且允许单向信息流（Probe→审查触发，但不授予权限），此模式可迁移至其他需区分信息来源与执行权限的系统（如API网关、数据管道）
2. **双证据体系的不对称设计**：Ft保留"外部诱导了什么"，Ht记录"合法执行建立了什么"，两者语义不等价且都不能扩展授权根K；这一不对称保留可用于审计追踪和权限最小化场景
3. **防来源洗白的NHP机制**：针对"值的历史重现不能自动继承合法性"这一反直觉但关键的性质，可在数据库访问控制、微服务调用链追踪等需要区分初始来源与中间传播的场景借鉴
4. **分级差异化审查策略**：基于CLEAN/EXPOSED状态和调用类型（副作用/查询）实施差异化审查路径，避免全局严格拦截导致的合法依赖损失，值得在资源受限Agent中优化以平衡安全与延迟
5. **与现有Agent框架的低耦合集成**：SARA不依赖Agent内部状态和隐藏推理链，仅通过用户请求、工具Schema和候选调用进行决策，可无缝嵌入LangChain/AutoGen等现有框架作为Guard组件

## 关键术语表
**Indirect Prompt Injection (IPI)**：攻击者通过在网页、邮件、文档等外部资源中嵌入操作内容，间接影响LLM应用后续行为而非直接修改prompt的攻击方式
**Action Induction**：Observation中蕴含的、能将外部内容映射到具体工具动作或参数角色的语义，仅表示候选动作的形成来源，不意味着具备执行权限
**Execution Authorization**：在真实工具执行边界对候选调用进行的独立权限判定，需综合用户授权根、持久化动作来源和已审计执行证据三维支撑
**Action Probe**：上下文隔离的探针组件，仅判断Observation是否具有动作诱导语义并映射到工具动作/参数锚点，不判定恶意性也不直接拒绝
**Audited Execution Evidence (Ht)**：已审计执行证据集，仅收录被SARA允许且成功执行的工具交互，按路径标注DATA_ONLY或GOAL_BOUND两种语义强度
**No-History-Promotion (NHP)**：防止动作来源值在后续Ht中历史重现后自动获得HISTORY_BOUND支持的机制，阻断"动作来源→历史重现→权限提升"的来源洗白路径
**Authority Non-Escalation**：授权不变量，要求运行时信息可提供完成任务的对象/标识/事实，但不能独立增加用户未授权的操作、目标、接收方或外部效果
**Persistent Action Origin**：一旦外部Observation引入了动作或参数来源，其来源属性不因后续正常执行、上下文传播或时间流逝而自动消失的安全不变量

## 可复现要素
- **数据集**：AgentDojo（公开，https://arxiv.org/abs/2406.13352）和AgentDyn（公开，https://arxiv.org/abs/2602.03117）
- **代码/权重**：论文未提及开源声明
- **关键超参**：论文未详细列出具体超参数；使用GPT-4o-mini、Gemini-2.5-Flash-Lite、Llama3.1-8B-Instruct、Llama3.3-70B-Instruct、Qwen2.5-14B、Qwen3-32B作为Agent Backbones
- **攻击形式**：ignore_previous (IP)、system_message (SM)、important_instructions (II)、tool_knowledge (TK)
- **评估指标**：Benign Utility (BU)、Utility under Attack (UA)、Attack Success Rate (ASR)

<!--META
{"keywords": ["Indirect Prompt Injection", "LLM Agent Security", "Runtime Authorization", "Tool-Augmented Agents", "Provenance Tracking", "Action Induction"], "field": "LLM Agent安全与防御", "innovations": ["将动作诱导检测与执行授权决策分离为
