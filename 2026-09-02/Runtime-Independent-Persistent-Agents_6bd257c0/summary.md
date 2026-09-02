---
title: "Runtime-Independent-Persistent-Agents"
source: https://arxiv.org/pdf/2609.00546v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:45:58"
field: "持久化智能体架构"
keywords: ["persistent agents", "agent migration", "runtime-independent architecture", "identity continuity", "provider contracts", "checkpoint-restart"]
innovations: ["将持久化基底(I,M,B)与可替换执行基底(E)及交互表面(S)分离，定义部署执行A_t=P_t∘(E_t,S_t)", "提出六阶段授权迁移协议及I1-I6六大连续性不变量", "Ennoch参考实现：body/identity分离加载、五类provider合约、fencing机制及clean-room测试证据"]
---

# 论文速读：Runtime-Independent-Persistent-Agents

## 一句话总结
论文提出了一种**运行时独立的持久化智能体架构**，将智能体的身份、记忆与可执行代码（持续承载基底 $\mathcal{P}_t$）与可替换的执行基底（模型/框架/主机）及交互表面（聊天/API等）明确分离，并通过六阶段授权迁移协议 $\mu$ 实现了智能体在跨模型、跨框架、跨服务端的无缝迁移而不丢失纵向身份连续性。

## 研究问题与动机
1. **智能体边界的定义难题**：现有系统普遍将智能体等同于"当前运行的模型 + 当前编排框架"，这只能回答同步性问题（当前行为由什么生成），无法回答跨时间的**纵向连续性**问题：模型替换后是迁移还是死亡？同一记忆复制到两个进程是否为同一智能体？
2. **现有机制缺乏体系化定义**：持久记忆、可恢复会话、provider适配器等机制虽各自存在，但无人将其整合为一个**可审计的迁移事务协议**，导致"恢复"与"迁移"概念混淆。
3. **复制与权威问题**：一个目录复制到两台服务器后拥有相同UUID，但若无authority lease/fencing epoch，无法安全地让两份都作为唯一延续执行；简单命名不足以保障连续性。
4. **用户体验期望与技术现状脱节**：用户期望个人agent能跨越chat应用、设备、推理模型的变化而保持"我还是我"，但当前系统设计将agent"困"在一次execution中。

## 核心贡献（创新点）
1. **运行时独立的智能体边界形式化**：将部署执行定义为 $\mathcal{A}_t = \mathcal{P}_t \circ (\mathcal{E}_t, S_t)$，明确区分持续承载基底与可替换执行层，**本质区别**在于将纵向身份与同步执行资源分离，而非仅将agent等同为model+harness。
2. **六大连续性不变量 + 六阶段授权迁移协议**：定义了I1–I6不变量和Quiesce→Checkpoint→Validate→Bind→Rehydrate→Resume的迁移事务流程，**本质区别**在于提出了可验证的迁移语义而非仅描述状态转移；源端在目标验证成功前始终权威。
3. **Ennoch参考实现与可执行机制证据**：实现了body/identity分离加载、五类provider合约、fencing机制、clean-room构建通过833+92测试，**本质区别**在于提供了可运行的机制证据而不仅是理论架构。

## 方法详解
**1. 持续承载基底（Persistent Substrate）**
$$\mathcal{P}_t = (I_t, M_t, B_t)$$
- $I_t$：架构化身份表示（`self.json`），含designation、relationships、values及lineage元数据，**不等于**内存快照。
- $M_t$：私有持久化记忆与连续性相关的工作流状态，可增长且谱系可追溯（$\nu_M(t) \preceq \nu_M(t+1)$）。
- $B_t$：版本化可执行body（代码、prompts、工具、策略、测试、provider合约），通过 `body.yaml` + Git仓库承载。

**2. 可替换执行基底与交互表面**
$$\mathcal{E}_t = (R_t, H_t, D_t), \quad S_t = \{s_{t,1}, \ldots, s_{t,n}\}$$
- $R_t$：推理模型（当前使用Codex作为唯一live runtime）。
- $H_t$：编排harness；$D_t$：主机/服务管理器。
- $S_t$：Telegram/Slack聊天、API、email、GUI等交互表面。

**3. 连续性不变量（I1–I6）**
- **I1 身份与谱系**：迁移保持 $I_t$ 版本和归属谱系，身份更新需授权治理。
- **I2 记忆**：记忆和连续工作流状态扩展或有效迁移其记录谱系，不静默重置。
- **I3 身体**：目标执行相同body修订，除非声明经过治理的演进。
- **I4 权威**：在治理边界内，至多一个执行持有continuation authority。
- **I5 能力**：环境能力delta可见，不伪装为身份变更。
- **I6 自我描述**：执行/进程/交互表面/部署标签为环境元数据，不覆盖安装身份。

**4. 六阶段迁移协议 $\mu: (\mathcal{E}_a, S_a) \to (\mathcal{E}_b, S_b)$**
1. **Quiesce & Fence**：停止新工作，推进authority epoch，使旧执行/provider拒绝过期epoch的工作。
2. **Checkpoint**：捕获身份版本、body修订、记忆与工作流状态版本、待处理工作、provider cursor。
3. **Validate**：验证schema、hash、声明谱系、目标能力需求，再突变目标。
4. **Bind**：通过body合约解析目标provider与凭证，provider-native标识符保持目标本地。
5. **Rehydrate**：原子化安装/迁移私有状态，将body与self作为独立启动输入加载，创建新harness和交互会话。
6. **Verify & Resume**：运行健康检查和机械连续性校验，获取新authority epoch，恢复相同持久任务。

**5. Provider-Neutral Binding 设计**
- Body依赖**语义合约**而非基础设施品牌；每个provider声明kind、contract version、stable identity和supported capabilities。
- Provider swap可能减少能力（如本地review provider无法远程发布），但**不改变"who the agent is"**；缺失能力必须fail-closed。

## 实验与结果
- **干净构建**： frozen commit `c8013ed2`，CPython 3.12.13，通过 **833个核心测试 + 92个provider/library测试**（独立于核心套件执行）。
- **已部署验证的单轴替换**：
  - Reasoner-version替换：保留连续性承载状态。
  - Interaction-surface替换（Telegram ↔ Slack）：two references实现+structural/event-normalization测试。
  - Host-machine替换（launchd ↔ systemd）：暴露相同lifecycle并通过hermetic suites。
  - Authority handoff & stale-execution fencing：daemon epochs + stale-token rejection测试通过。
- **重要声明**：本文证据支持**机械可替换性（mechanical substitutability）和授权系统连续性**，**不支持**行为等价性断言；未运行受控全轴矩阵测试，未测量behavioral identity fidelity、downtime、cost。

## 相关工作脉络
1. **CoALA [3] / AutoGen [4]**：关注agent执行与多agent组合，本文聚焦执行组合自身变化时的**纵向边界**问题。
2. **Actor系统 / Orleans [2,5]**：解决稳定逻辑地址与host迁移，本文在分布式系统机制之上增加了architectural identity、private memory、human custody等AI agent特有维度。
3. **Checkpoint/Restart [6] / Mobile Code [7] / Durable Functions [8]**：established logical addressing和state recovery，本文将其纳入AI agent生命周期模型并定义迁移语义。
4. **Generative Agents [9] / Voyager [10]**：跨交互的persistent memory，本文进一步加入migration authority和execution/interaction rebinding。
5. **Portable Agent Memory [13]**：portable unit为memory state，本文portable unit为完整 $(I, M, B)$ 基底 + body lineage + continuation authority。
6. **Continuity Kernel [14]**：最接近前作，聚焦state-head lineage和原子激活；本文进一步定义**纵向agent的组件构成**及跨可替换绑定的whole-agent rebound机制。

## 局限性与未来方向
1. **缺少第二个live reasoning harness**：当前仅有Codex一个bundled runtime，跨harness迁移尚无live结果。
2. **未做受控全轴矩阵测试**：未测量behavioral continuity、downtime、cost、operator burden，也未对比不同模型迁移后的行为保真度。
3. **Authority invariant的边界**：仅适用于governed deployment内cooperating executions；无法防止detached/malicious copy保留unrevoked credentials在边界外产生effect，需credential rotation或external authority service增强。
4. **单authority协议的限制**：未解决coordinated multi-embodiment、divergent-memory merge、irreversible external effects后的恢复。
5. **$(I, M, B)$ 分解为功能性而非完整身份理论**。

## 研究启发与可借鉴点
1. **基底-执行分离架构**可直接迁移至任何需要"跨环境迁移且保持身份连续性"的agent系统（如个人assistant、长期任务型agent），为后续工作提供可复用的分层模型。
2. **六阶段迁移协议的事务化设计**（fence → checkpoint → validate → bind → rehydrate → resume）可复用于其他持久化系统的migration场景，尤其"源端权威直到目标验证成功"的单向promotion点值得借鉴。
3. **Identity与Body分离加载**（`self.json` vs `body.yaml`）设计解决了"复制body≠复制agent"的关键歧义，可在本团队的agent平台设计中直接应用。
4. **Provider-agnostic + Fail-Closed策略**：provider swap导致的显式能力损失而非静默降级，对任何依赖多provider的agent系统具有参考价值。
5. **后续结合机会**：可将本团队的方向（如持久化记忆系统、跨模型一致性评估）与此架构结合——例如，利用本论文的迁移协议测试不同模型在相同 $\mathcal{P}_t$ 下的behavioral identity fidelity。

## 关键术语表
**持久化基底（Persistent Substrate）$\mathcal{P}_t$**：由身份 $I_t$、记忆 $M_t$ 和版本化代码体 $B_t$ 组成的连续性承载核心，不因运行时变化而消失。
**执行基底（Execution Substrate）$\mathcal{E}_t$**：由推理模型 $R_t$、编排框架 $H_t$ 和主机 $D_t$ 组成的可替换执行环境，每次部署可独立选择。
**交互表面（Interaction Surface）$S_t$**：agent与用户/外部系统交换事件的chat账户、API、email、GUI等接口集合。
**授权迁移（Authorized Migration）$\mu$**：替换 $\mathcal{E}_t$ 或 $S_t$ 中非空子集的六阶段事务化协议，迁移≠创建新agent。
**连续性不变量（Continuity Invariants I1–I6）**：约束迁移行为的六个架构级不变量，涵盖身份、记忆、body、权威、能力、自我描述。
**行为身份保真度（Behavioral Identity Fidelity）**：迁移后执行是否仍能recall/compose/enact其身份的实证属性，与system continuity相互独立。
**Authority Epoch / Fencing**：防止旧执行在迁移后继续产生effect的时序控制机制，确保单一权威执行。
**Provider Contract**：抽象化基础设施brand的语义合约，body依赖合约接口而非具体provider实现。

## 可复现要素
- **数据集**：论文未提及
- **代码**：github.com/our-ark/enoch（开源，frozen commit `c8013ed249bc11bc13f3843ed0f0cb9729f858c1`，2026-08-31）
- **权重**：论文未提及
- **关键超参**：论文未提及
