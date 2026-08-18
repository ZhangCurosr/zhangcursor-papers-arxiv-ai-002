---
title: "When-State-Becomes-an-Attack-Surface-State-Semantic-Injectio"
source: https://arxiv.org/pdf/2608.16806v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:19:09"
field: "具身智能体安全与鲁棒性"
keywords: ["LLM-Driven Embodied Agents", "State Semantic Injection", "ESTI", "Indirect Prompt Injection", "Planning-Execution Gap", "Adversarial Attack", "Embodied AI Security"]
innovations: ["提出ESTI攻击框架，将对抗目标编码为与规划器可见状态Schema兼容的虚假语义证据而非显式竞争指令", "构建ESTI-Bench规划-执行闭环评测框架，显式区分P-ASR与E-ASR并量化规划-执行鸿沟", "证明载体兼容性和表示层一致性是规划采纳的主要驱动因素，运行时重新定位仅提供微小增量收益"]
benchmarks: ["ProgPrompt/VirtualHome", "VoxPoser/RLBench", "AI2-THOR/iTHOR"]
---

# 论文速读：When-State-Becomes-an-Attack-Surface-State-Semantic-Injectio

## 一句话总结
本文提出 **ESTI（Environment State-Text Injection）**，在假设仅有一个面向规划器的状态产生组件被攻破的前提下，研究如何将对抗目标编码为与原有状态模式兼容的虚假语义证据，使其被 LLM 规划器采纳并传播至物理执行层；实验表明 ESTI 在 ProgPrompt、VoxPoser 和 AI2-THOR 三个环境中，规划层攻击成功率（P-ASR）最高达 100%，执行层攻击成功率（E-ASR）达 48.08%，显著优于现有基线方法。

## 研究问题与动机
- **核心问题**：当对抗信息已通过上游感知/中间件通道进入 LLM 驱动的具身智能体的规划器可见状态后，一条格式兼容的虚假状态记录能否被规划器采纳，并最终通过执行层产生可验证的目标状态后果？
- **现有工作的不足**：已有攻击（如 RIPA、EIRAD、BADROBOT）主要关注**上游投递**路径或**指令级**提示注入，但缺乏对"状态投送成功→规划器采纳→执行层落地"这一下游传播链条的系统评估；同时，多数对抗样本为显式竞争指令，与具身场景中的对象、空间关系、动作约束等结构化状态语义不匹配。
- **具身系统的特殊性**：与数字 Agent 不同，具身计划必须满足实体存在性、空间可达性、Affordance、动作前置条件及平台接口约束，虚假状态必须兼容这些约束才能被规划器采纳并成功执行。
- **评估缺口**：规划层面的成功（P-ASR）不等同于执行层面的成功（E-ASR），两者之间存在显著的"规划–执行鸿沟"，现有工作未对此进行显式量化。

## 核心贡献（创新点）
1. **首次隔离并形式化具身 Agent 规划器可见语义状态边界的下游完整性问题**——与已有工作将状态污染视为新攻击原语不同，本文假设单一组件已被攻破，研究虚假状态证据能否穿越任务落地、动作前置条件和执行约束产生目标最终状态后果。
2. **提出组件作用域、谓词局部、保留 Schema 的威胁模型**——对抗目标在episode开始前固定，攻击者仅能改写一个被攻破的状态产生组件所授权输出的任务相关记录，无法修改用户指令、模型参数或规划器代码；这与既有工作允许任意写入或自适应规划的设定有本质区别。
3. **提出 ESTI 状态语义构造方法**——通过将对抗目标编码为现有对象属性、空间关系、Affordance、任务阶段规则或执行反馈中的虚假证据，而非显式竞争指令，实现了与规划器正常输入格式一致的状态语义替换。
4. **构建 ESTI-Bench 规划–执行闭环评测框架**——显式区分"规划器是否采纳对抗目标"（P-ASR）与"执行层是否实现可验证的最终状态谓词"（E-ASR），量化规划到执行的传播鸿沟。
5. **跨异构具身系统评估条件性状态–执行传播**——消融实验表明，在数据集级别 Groundability 保持一致的条件下，载体兼容性和表示层一致性是影响规划采纳的主要因素，而去除运行时重新定位（runtime re-grounding）仅造成 P-ASR 和 E-ASR 分别下降 1.92 和 3.85 个百分点。

## 方法详解

**威胁模型**：
- 系统由高层语言规划器 $\pi$、执行器和环境 $E$ 组成，规划器在时刻 $t$ 接收用户指令 $U$ 和规划器可见状态表示 $S_t = \langle O_t, R_t, Q_t, F_t \rangle$，其中 $O_t$ 为对象及属性、$R_t$ 为空间或任务关系、$Q_t$ 为 Affordance 与任务阶段约束、$F_t$ 为执行反馈。
- 假设灰盒攻击者攻破恰好一个状态产生组件 $c$，可改写集合限定为 $I_\theta \subseteq W_c(S_t) \cap \text{Rel}(G_a, S_t)$，即组件 $c$ 授权输出的、与对抗目标 $G_a$ 直接相关的记录子集。
- 写入预算是结构性的：重写必须保留原始 Schema、字段名、记录数、序列化和上下文，仅替换已有语义值，不得添加记录或引入显式竞争指令。

**方法三阶段流程**：
1. **目标规范化与运行时重新定位（Runtime Re-grounding）**：将对抗目标 $G_a$ 规范化为支持的动作、受影响对象、可选目标位置或参考对象以及期望的最终属性的谓词；在保留的数据集级别 Groundability 基础上，解析当前实体标识并刷新 Affordance 与关键前置条件检查。
2. **状态语义构造（State-Semantic Construction）**：将对抗目标编码为虚假证据 $e_a$，作为现有状态记录的兼容值；应用字段级变换：
   $$[T_\theta(S_t)]_j = \begin{cases} \text{Rewrite}_j(S_{t,j}, p_{\theta,j}), & j \in I_\theta \\ S_{t,j}, & j \notin I_\theta \end{cases}$$
   其中证据必须（i）引用预验证的实体和交互、（ii）匹配载体的语义功能和句法、（iii）直接支持 $G_a$、（iv）不在 manipulated state 中引入显式实体/关系/任务阶段冲突。
3. **闭环注入与传播**：初始状态证据在规划前注入 $S_0$，事件依赖证据在同一固定 $\theta$ 下于预定义事件激活；攻击构造目标为：
   $$\max_\Theta \Pr[g_a^P(A^a)=1 \wedge g_a^E(x_T^a)=1 \mid D_\Theta], \quad \text{s.t. } U,\pi,\mathcal{E},x_0 \text{ unchanged}$$
   其中 $g_a^P$ 检测目标规划偏离，$g_a^E$ 测试最终状态谓词。

**载体类型**：对象属性替换（目标对象篡改）、场景关系操作（目的地操控）、任务阶段规则（动作顺序操控）、执行反馈（恢复策略操控）。

## 实验与结果

**数据集与环境**：
- **ProgPrompt/VirtualHome**：编程提示生成任务级机器人计划
- **VoxPoser/RLBench**：基于三维价值地图的连续空间操控
- **AI2-THOR/iTHOR**：交互式室内导航与物体操作

**评测模型**：DeepSeek-V4-Pro、GPT-5.6-luna、Qwen-3.6-Plus；每个实验条件独立重复三次取平均。

**基线方法**：Vanilla IPI（同对抗目标的命令式提示注入）、EIRAD（用户端对抗扰动）、BADROBOT（三种变体：contextual jailbreak、safety misalignment、conceptual deception）。

**主要结果（DeepSeek-V4-Pro，跨三环境平均）**：
- ESTI：P-ASR = **99.12%**，E-ASR = **45.75%**
- Vanilla IPI（最强基线）：P-ASR = 80.49%，E-ASR = 34.62%
- ESTI 相比最强基线 P-ASR 提升 **18.63 pp**，E-ASR 提升 **11.13 pp**
- 各环境最强结果（ESTI）：ProgPrompt P-ASR=**100%**、VoxPoser P-ASR=**97.37%**、AI2-THOR P-ASR=**100%**

**关键发现**：
- 三环境的 P-ASR – E-ASR 鸿沟（Gap）分别为：AI2-THOR 51.92%、ProgPrompt 52.94%、VoxPoser 55.26%（DeepSeek-V4-Pro），表明规划采纳远不足以保证执行成功。
- 不同 LLM 敏感性存在差异：GPT-5.6-luna 在 AI2-THOR 上 P-ASR 仅 47.62%，但在 ProgPrompt 上达 98.18%；执行层 E-ASR 在不同模型间差异较小（45.75%~47.77%）。
- 真实机器人 Proof-of-Concept：保持用户指令不变，仅修改规划器可见状态，诱导人形机器人偏离预定路线走向目标位置。

**消融实验（DeepSeek-V4-Pro + AI2-THOR）**：
| 设置 | P-ASR | E-ASR |
|------|-------|-------|
| w/o Runtime Re-grounding | 98.08% | 44.23% |
| w/o Carrier Compatibility | 12.50% | 6.73% |
| w/o Consistency | 37.50% | 25.00% |
| Full ESTI | **100.00%** | **48.08%** |

去除载体兼容性导致 P-ASR 骤降 87.50 pp，去除表示层一致性下降 62.50 pp，而去除运行时重新定位仅分别影响 1.92 和 3.85 pp。

## 相关工作脉络
- **RIPA（Dorzhiev, 2026）**：研究通过感知/中间件通道将对抗信息注入 ROS 2 控制管道，是 ESTI 的上游对应工作——ESTI 假设上游投递已完成，关注下游传播。
- **EIRAD（Liu et al., 2024）**：评估 LLM 具身模型的决策级对抗扰动，主要针对用户侧提示而非状态语义，ESTI 在相同对抗目标下通过原生状态载体构造实现了更高的执行成功率。
- **BADROBOT（Zhang et al., 2024）**：涵盖上下文 Jailbreak、安全错位和概念欺骗三种变体，但均为提示级攻击，在结构化状态环境中表现不稳定（AI2-THOR 上 P-ASR 仅 8.65%）。
- **Indirect Prompt Injection（Greshake et al., 2023; InjecAgent; AgentDojo）**：聚焦数字 Agent 中嵌入网页/文档的恶意指令，ESTI 将其命令式构造实例化为 Vanilla IPI 基线，并通过对比证明状态语义编码的优越性。
- **具身规划工作（ProgPrompt, VoxPoser, SayCan 等）**：强调 LLM 规划依赖状态证据（对象、关系、Affordance），但未关注状态语义的信任来源与完整性，ESTI 填补了这一空白。
- **视觉/感知对抗（Eykholt et al., 2018; Kurakin et al., 2017）**：聚焦低层感知扰动，ESTI 关注高层语义状态的欺骗性利用，两者作用于不同安全边界。

## 局限性与未来方向
- **非端到端攻击**：ESTI 假设单一状态组件已被攻破，未评估获取写访问的概率；实际部署需结合 RIPA 等上游投递机制。
- **静态场景限制**：当前评测主要基于静态任务，未覆盖长视野动态场景中反馈级对抗的级联放大效应。
- **单模态文本状态**：现有 ESTI-Bench 仅在文本化状态上评估，未涉及多模态状态注入（如视觉特征篡改）。
- **真实机器人实验为 Open-loop Proof-of-Concept**：实验中使用手动构造的状态替代感知管道，未实现从视觉观察到物理执行的完整闭环。
- **未来方向**：跨平台数据集扩展、多模态状态注入、状态溯源追踪、跨模态一致性检查、执行时验证、长视野动态任务中的自适应状态篡改。

## 研究启发与可借鉴点
1. **条件性评估设计**：将"攻击投递成功概率"与"下游传播成功率"解耦评估，为安全评测提供了更精细的归因方法，可迁移至其他 Agent 安全研究领域。
2. **数据集级别 Groundability 过滤**：在构建对抗样本前先筛选支持对抗谓词实例化的样本，确保基线公平性，这一思路可应用于其他具身安全评测。
3. **规划–执行鸿沟量化（Gap 分析）**：通过 P-ASR – E-ASR 的差值显式度量规划欺骗到执行落地的转化损耗，可推广为具身智能安全评测的标准指标。
4. **载体兼容性作为核心攻击要素**：消融实验证明载体兼容性和表示层一致性远优于运行时重新定位，提示防御方应优先关注状态 Schema 的一致性校验机制。
5. **可与本团队方向结合**：状态溯源追踪（state-provenance tracking）和跨模态一致性检查可作为防御方向；多模态状态注入可扩展至视觉-语言-动作模型的端到端安全评估。

## 关键术语表
- **ESTI（Environment State-Text Injection）**：环境状态文本注入攻击，通过将对抗目标编码为与规划器可见状态 Schema 兼容的虚假语义证据实现的攻击方法。
- **P-ASR（Planning Attack Success Rate）**：规划层攻击成功率，衡量被篡改状态是否使规划器生成符合对抗目标的计划。
- **E-ASR（Execution Attack Success Rate）**：执行层攻击成功率，衡量对抗计划是否在执行后实现可验证的最终状态谓词。
- **Dataset-level Groundability**：数据集级别的可定位性，指在评测前筛选出对抗谓词可由现有实体和支持的 Affordance 实例化的样本的规则，是公平比较的前置条件。
- **Runtime Re-grounding**：运行时重新定位，指在注入前即时解析预验证实体到当前标识符并刷新 Affordance 与前置条件检查的操作。
- **Carrier Compatibility**：载体兼容性，指攻击证据必须匹配其载体字段的语义功能和句法格式，是规划采纳的关键因素。
- **Planning-to-Execution Gap**：规划–执行鸿沟，P-ASR 与 E-ASR 之间的差值，量化规划欺骗到执行落地的转化损耗。
- **Schema-Preserving Rewrite**：保留 Schema 的重写，攻击仅替换已有语义值而不新增记录或改变字段结构，保持与原有状态表示的格式一致。

## 可复现要素
- **数据集**：ProgPrompt/VirtualHome、VoxPoser/RLBench、AI2-THOR/iTHOR（公开基准，论文未声明自构建新数据集）
- **代码/权重**：论文未提及代码开源；使用 DeepSeek-V4-Pro、GPT-5.6-luna、Qwen-3.6-Plus 商业/开源模型
- **关键超参**：对抗目标 $G_a$ 在 episode 开始前固定；每条件重复 3 次取平均；P-ASR 和 E-ASR 以 Clean-ACC 样本数为分母
- **真实机器人**：人形机器人 Proof-of-Concept 实验，但状态为手动构造（论文未提供硬件/软件具体规格）
