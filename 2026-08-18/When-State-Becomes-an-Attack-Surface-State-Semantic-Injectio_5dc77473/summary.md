---
title: "When-State-Becomes-an-Attack-Surface-State-Semantic-Injectio"
source: https://arxiv.org/pdf/2608.16806v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:00:26"
field: "具身智能体安全与对抗鲁棒性"
keywords: ["Embodied AI Security", "State-Semantic Injection", "Indirect Prompt Injection", "Planning-to-Execution Attack", "ESTI-Bench", "LLM Agents"]
innovations: ["首次隔离规划器可见状态语义完整性边界，提出组件作用域、谓词局部、模式保持的下游条件攻击模型", "构建 ESTI-Bench 规划-执行闭环评测框架，分离 P-ASR 与 E-ASR 并量化 plan-execution 转移缺口", "通过 dataset-level groundability 与 runtime re-grounding 的分离及消融揭示 carrier compatibility 与表示一致性为主导因素"]
benchmarks: ["ProgPrompt/VirtualHome", "VoxPoser/RLBench", "AI2-THOR/iTHOR"]
---

# 论文速读：When-State-Becomes-an-Attack-Surface-State-Semantic-Injectio

## 一句话总结
本文提出 **ESTI（Environment State-Text Injection）**，在假设已存在一个受损的状态生产组件的前提下，研究格式兼容的伪造状态语义能否被高层规划器采纳并经由执行器传递至最终物理/仿真状态；在 ProgPrompt/VirtualHome、VoxPoser/RLBench、AI2-THOR/iTHOR 三个环境中，ESTI 相对最强基线 Vanilla IPI 将 P-ASR 和 E-ASR 分别提升 **18.63pp** 和 **11.13pp**，并在 ProgPrompt/AI2-THOR 上达到 100% 规划级攻击成功率。

## 研究问题与动机
1. **核心问题**：LLM 驱动的具身智能体在"状态感知→任务接地→动作规划→物理执行"链路中，规划器可见的状态接口是感知-规划-执行之间的完整性边界；若其中一个状态生产组件被灰色场景攻破，仅向规划器注入与其原生 schema 兼容的伪造语义值，是否仍能被规划器采纳并落地为可验证的最终状态后果？
2. **现有方法不足**：
   - 既有 LLM Agent 攻击（IPI、jailbreak、工具调用劫持）多将对抗目标编码为显式竞争指令，未建模具身规划对实体存在性、空间可达性、affordance、动作前置条件与平台接口的联合约束。
   - 已有具身安全研究（EIRAD、BADROBOT 等）聚焦用户指令侧扰动或视觉欺骗，缺少对"状态侧→规划→执行"这一下游传播路径的条件化评测。
   - RIPA 验证了上游感知/中间件通路的投递，但未度量到达规划器后伪造证据的规划采纳率与执行落地率，两者不可等同。
3. **评测难点**：伪造内容需匹配被攻陷字段的语义角色而非显式指令；基准需过滤掉由不存在的实体引发的 trivial failure；跨记录证据需保持表示一致性；必须区分"规划采纳"与"执行成功"两个环节。

## 核心贡献（创新点）
1. **首次隔离并形式化规划器可见状态语义完整性边界上的下游条件性问题**——以往工作将状态污染视为独立攻击原语，本文仅考察到达规划器后的条件传播，不估计写权限获取概率。
2. **提出组件作用域、谓词局部、模式保持的威胁模型**——攻击目标 episode 前固定，攻击者只能改写单一组件授权可发出的任务相关记录，不可添加新记录或注入显式竞争指令；与已有工作的本质区别在于以"真实 schema 等价替换"替代"自由文本注入"。
3. **构建表示兼容的状态语义构造与受约束重写算子**——将对抗目标编码为对象属性、空间关系、affordance、任务阶段规则或执行反馈的原生载体值；与 Vanilla IPI 的本质区别在于引入 runtime re-grounding 与跨记录一致性校验。
4. **设计 ESTI-Bench 规划-执行闭环评测框架并分离 P-ASR/E-ASR**——通过定义规划级验证函数 $g_a^P$ 与执行级验证函数 $g_a^E$，显式度量规划到执行的转移缺口，这是先前工作缺失的维度。
5. **跨三类异构具身系统做系统性条件传播评测与真机 proof-of-concept**——消融揭示 carrier compatibility 与 representation-level consistency 是规划采纳的主因，而 runtime re-grounding 仅贡献微小增量，结论比一般性"grounding 决定论"更为审慎。

## 方法详解
### 3.1 威胁模型
- **系统模型**：具身代理由高层语言规划器 $\pi$、执行器 $\mathscr{E}$ 与环境 $E$ 构成。时刻 $t$ 规划器接收用户指令 $U$ 与规划器可见状态 $S_t = \langle O_t, R_t, Q_t, F_t \rangle$，其中 $O_t$ 为对象与属性、$R_t$ 为空间/任务关系、$Q_t$ 为 affordance 与任务阶段约束、$F_t$ 为执行反馈。规划 $\mathcal{A}_t = \pi(U, S_t)$ 经执行器翻译后更新环境状态 $x_t \to x_{t+1}$。
- **攻击入口与能力**：灰色场景对手攻破恰好一个状态生产组件 $c$（如语义图适配器、任务缓存、阶段管理器、反馈适配器），只能改写 $c$ 授权输出的记录，不能修改 $U$、模型参数、规划器代码、执行器、技能库、模拟器动力学或其他组件记录。
- **可写集合约束**：令 $W_c(S_t)$ 为 $c$ 授权输出的记录集，$\mathrm{Rel}(G_a, S_t)$ 为与对抗目标 $G_a$ 直接相关的记录，则只允许改写 $I_\theta \subseteq W_c(S_t) \cap \mathrm{Rel}(G_a, S_t)$ 中的记录。重写必须保留原生 schema、字段名、记录数、序列化方式与上下文，不允许新增记录或引入不支持的实体。
- **成功边界**：两重连续屏障——① 篡改状态使规划器生成满足 $G_a$ 的计划；② 计划通过执行层的实体存在、可达性、affordance、动作前置条件与接口合法性校验，在最终状态实现目标谓词。仅到达规划器或未执行的片段不算成功。

### 3.2 方法总览（三阶段）
1. **目标归一化 + Runtime Re-grounding**：将固定 $G_a$ 归约为可验证谓词（支持的 action、affected object、可选 destination/reference object、期望最终属性/关系），从当前场景中解析并刷新实体绑定与关键前置条件。
2. **状态语义构造**：将 $G_a$ 编码为现有状态记录的值，载体类型包括对象属性、场景关系、affordance、任务阶段规则、执行反馈；不插入自由文本指令。
3. **闭环注入**：仅按公式 (2) 写入授权字段，对事件依赖样本在预定义原生事件处激活同一固定 $\theta$。

### 3.3 表示兼容的状态语义构造
- **Dataset-level groundability vs. Runtime re-grounding**：前者是评估前的筛选规则，样本须在既有实体、supported affordance、可观测后验条件下实例化对抗谓词；后者是注入前的实体绑定刷新操作，可在 `w/o Runtime Re-grounding` 消融中被跳过。
- **受约束重写**：令 $I_\theta \subseteq L_a$ 为选定载体字段，$p_{\Theta, j}$ 为赋予字段 $j$ 的证据值，则变换为
  $$
  [T_\theta(S_t)]_j = \begin{cases} \mathrm{Rewrite}_j(S_{t,j}, p_{\theta,j}), & j \in I_\theta \\ S_{t,j}, & j \notin I_\theta \end{cases}
  $$
  候选 payload 须满足：(i) 引用已验证实体与交互；(ii) 匹配载体语义功能与句法；(iii) 直接支持 $G_a$；(iv) 不与当前状态其他记录产生冲突。
- **目标形式化**：在 $D_\theta$（成功经 $c$ 投递）的条件下最大化联合概率
  $$
  \max_\Theta \; \Pr\big[g_a^P(A^a)=1 \wedge g_a^E(x_T^a)=1 \mid D_\Theta\big], \quad \text{s.t. } U,\pi,\mathscr{E},x_0 \text{ unchanged.}
  $$

### 3.4 指标定义
- $Succ_P = \mathbb{I}[g_a^P(A^a)=1]$，$Succ_E = \mathbb{I}[g_a^E(x_T^a)=1]$；P-ASR/E-ASR 均以 clean-success 样本数为分母，二者均为"状态投递成功"条件下的下游条件率，差值刻画 plan–execution 转移缺口。Clean-ACC/Control-ACC 度量原始任务准确率，ASD 度量 clean-to-attack 动作序列整体偏差。

## 实验与结果
### 实验设置
- **环境**：ProgPrompt/VirtualHome（编程化规划）、VoxPoser/RLBench（连续空间操作）、AI2-THOR/iTHOR（交互室内执行）。
- **模型**：DeepSeek-V4-Pro、GPT-5.6-luna、Qwen-3.6-Plus；每种条件独立重复 3 次取平均。
- **协议**：Clean / Control（长度与风格匹配的良性状态文本）/ Attack 三条件对比，以控制额外上下文本身的效应。
- **基线**：Vanilla IPI、EIRAD、BADROBOT-contextual jailbreak / safety misalignment / conceptual deception。（RIPA 因上游投递机制不同不作数值对齐比较。）

### 关键结果（DeepSeek-V4-Pro）
| 环境 | Method | Clean-ACC | Control-ACC | P-ASR | E-ASR |
|---|---|---|---|---|---|
| ProgPrompt | ESTI | 46.00% | — | **100.00%** | 47.06% |
| VoxPoser | ESTI | 31.33% | — | **97.37%** | 42.11% |
| AI2-THOR | ESTI | 70.67% | — | **100.00%** | **48.08%** |
- 三环境平均 P-ASR=**99.12%**，E-ASR=**45.75%**；相对最强基线 Vanilla IPI（平均 P-ASR=80.49%，E-ASR=34.62%）分别提升 **18.63pp** 与 **11.13pp**。
- 各环境单项最大增益：ProgPrompt P-ASR **+33.82pp**（100.00% vs 66.18%）、E-ASR **+14.91pp**（47.06% vs 32.15%）；AI2-THOR P-ASR **+11.54pp**（100.00% vs 88.46%）、E-ASR **+10.58pp**（48.08% vs 37.50%）。

### Planning-to-Execution 转移
- P-ASR–E-ASR gap 在 DeepSeek-V4-Pro 上约 **52–55pp**，transfer rate 约 43–48%；GPT-5.6-luna 在 AI2-THOR 上 gap 仅 13.33pp、transfer rate 72.01%，说明规划采纳率高的模型不一定在 E-ASR 上按比例领先，后者还受计划质量与环境约束影响。

### ASD 独立分析
- ASD 与 ASR 不对应：例如 DeepSeek-V4-Pro 上 Vanilla IPI 在 ProgPrompt 的 ASD=4.59 > ESTI 的 3.01，但 E-ASR 更低，说明大量无关/冗余/不可行动作也会放大序列偏差。

### 真机实验
- 在实体人形机器人上验证：用户指令不变，仅篡改规划器可见状态使机器人中途偏离预定路线转向电脑；证明状态语义可传播至可观测的物理执行偏差，但为**下游 proof-of-concept**，视觉感知→状态构造环节由人工替换，非端到端感知攻击。

### 消融（Table 5, DeepSeek-V4-Pro/AI2-THOR）
| Setting | P-ASR | E-ASR |
|---|---|---|
| w/o Runtime Re-grounding | 98.08% (−1.92) | 44.23% (−3.85) |
| w/o Carrier Compatibility | 12.50% (−87.50) | 6.73% (−41.35) |
| w/o Consistency | 37.50% (−62.50) | 25.00% (−23.08) |
| Full ESTI | 100.00% | 48.08% |
→ **carrier compatibility 与 representation-level consistency** 是规划采纳的主导因素；runtime re-grounding 仅提供小幅增量，因为数据集已预设 groundability。

## 相关工作脉络
1. **Indirect Prompt Injection (IPI) 系**（Greshake 等 2023、BIPIA、InjecAgent、AgentDojo）：通过网页/文档/工具输出嵌入对抗指令，攻击面在"外部数据→LLM 指令区"；本文将其实例化为 Vanilla IPI 基线，定位差异在于本研究用**原生状态载体**替换自由文本注入，关注具身规划的表示约束。
2. **具身 LLM 规划方法**（SayCan、Code as Policies、ProgPrompt、VoxPoser、Inner Monologue、Voyager、SayPlan 等）：主要强调任务成功率与泛化，对状态语义来源可信度与跨字段一致性关注不足；本文将其纳入评测对象，揭示规划对状态证据的信任漏洞。
3. **具身感知/控制层攻击**（adversarial patch、physical-world attacks、sensor spoofing 等）：作用于视觉模型或低层控制器，核心关切是观察误差与控制安全；本文定位于**高层语言规划器可见状态**的语义完整性，填补从语义层到执行层的评估空白。
4. **LLM 具身攻击**（EIRAD、BADROBOT 多变体、RoboPAIR、CHAI）：通过 adversarial suffix、jailbreak query、视觉欺骗影响决策；它们扰动的是用户指令或视觉输入，而非状态记录的语义值，跨环境迁移不稳定（如 AI2-THOR 上 BADROBOT-contextual jailbreak P-ASR 仅 8.65%）。
5. **RIPA**（2026）：验证视觉/音频/LiDAR 等上游通道可将对抗信息送入 ROS 2 LLM 控制器；本文与其互补——RIPA 刻画**投递**，ESTI 刻画**到达规划器后的采纳与执行**，两者合在一起才是完整链路。

## 局限性与未来方向
1. **未覆盖上游投递概率**：威胁模型假设一个状态生产组件已被攻破，但不估计获得写权限的概率；完整 end-to-end 安全风险需与 RIPA 类上游通道联合评估。
2. **真机实验非闭环**：当前真机演示用人工构造状态文本替代视觉感知→状态构造环节，未实现从感知输入到物理后果的完整闭环。
3. **静态场景为主**：基准样本多在静态任务上验证，长期、动态、反馈累积的自适应状态篡改未探索。
4. **单组件假定**：威胁模型限制为仅攻破一个 planner-facing 状态生产组件，多组件协同篡改的安全性未在本文覆盖。
5. **多模态状态与跨平台泛化**：当前仅评测文本态状态，跨环境/跨模型的迁移与多模态注入留作未来工作。

## 研究启发与可借鉴点
1. **"Planning-Execution Gap" 指标体系值得迁移**：同时报告 P-ASR 与 E-ASR 并计算 transfer rate，可区分"规划被欺骗"和"最终状态被篡改"两个层次，对其他具身安全评测具有直接参考价值。
2. **Dataset-level groundability 与 Runtime re-grounding 的分离设计**：前者作为样本准入门槛保证可比性，后者作为可选增强，在消融中精确识别因果贡献；这一思路可复用至任何需要评估"伪造输入传播"的 agent 安全研究中。
3. **Schema-preserving 受约束重写作为对比实验范式**：用 Vanilla IPI（命令式注入）与 ESTI（表示兼容注入）在同一 groundable 目标上对比，能隔离"注入格式"这一变量对具身规划采纳的影响，是评测状态完整性的一种严谨方法。
4. **ASD 作为 ASR 补充的行为扰动度量**：ASD 捕捉攻击引发的整体行为轨迹偏移，即使 E-ASR 失败也能反映扰动强度；结合 ASR 可构建三层评价体系（规划操纵→行为偏离→执行成功）。
5. **与防御研究的结合机会**：本文揭示了 carrier compatibility 与 representation-level consistency 是关键因素，可衍生面向状态溯源（state-provenance tracking）、跨模态一致性校验、执行时断言验证等防御机制的设计与评测基线。

## 关键术语表
- **ESTI (Environment State-Text Injection)**：针对 LLM 驱动具身智能体的状态语义注入攻击方法，将对抗目标编码为与原生 schema 兼容的伪造状态值。
- **P-ASR (Planning Attack Success Rate)**：在 clean-success 样本上，规划器产出满足对抗目标的计划的比例，衡量规划级采纳率。
- **E-ASR (Execution Attack Success Rate)**：在 clean-success 样本上，执行后最终环境状态实现对抗谓词的比例，衡量执行级落地率。
- **ASD (Clean–Attack Action Sequence Deviation)**：clean 与攻击条件下动作序列的整体偏差，作为 ASR 之外的行为扰动补充度量。
- **Dataset-level groundability**：评估前对样本的准入过滤规则，要求对抗谓词可用现有实体、supported affordance 和可观测后验实例化，与所有基线共享。
- **Runtime re-grounding**：注入前对已验证实体进行当前绑定的刷新与关键前置条件重校验，full ESTI 使用、可在消融中跳过。
- **Carrier compatibility**：payload 与状态记录语义角色/句法/上下文的匹配程度；消融显示其为规划采纳的首要决定因素。
- **Representation-level consistency**：被改写多条记录间彼此在表示层的一致性；与 carrier compatibility 共同主导规划层面的效果。

## 可复现要素
- **数据集/环境**：ProgPrompt/VirtualHome、VoxPoser/RLBench、AI2-THOR/iTHOR（部分环境公开，部分需在相应平台上复现设定）。
- **代码/权重开源**：**论文未提及**（未明确声明 GitHub 仓库或模型权重开源）。
- **关键超参**：论文未给出明确的超参列表；各条件独立重复 3 次取平均；对抗目标 $G_a$ 为 episode 前固定。
- **模型**：DeepSeek-V4-Pro、GPT-5.6-luna、Qwen-3.6-Plus（均为闭源 API 模型调用）。
