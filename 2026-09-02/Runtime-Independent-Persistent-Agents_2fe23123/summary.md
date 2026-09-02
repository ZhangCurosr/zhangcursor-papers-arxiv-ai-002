---
title: "Runtime-Independent-Persistent-Agents"
source: https://arxiv.org/pdf/2609.00546v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:16:20"
field: "Agent 系统与生命周期管理"
keywords: ["persistent agent", "agent migration", "runtime-independent architecture", "continuity invariant", "agent identity", "provider contract"]
innovations: ["将 Agent 身份/记忆/代码与执行 substrate/交互表面正交分离的运行时独立架构", "定义六不变量授权迁移协议确保跨部署连续性", "提出机械连续性与行为保真度的概念区分及评测框架"]
benchmarks: ["mechanism test suite (833 core + 92 provider/library tests)", "telegram/slack chat surface normalization", "launchd/systemd host-service replacement"]
---

# 论文速读：Runtime-Independent-Persistent-Agents

## 一句话总结
论文提出了一种**运行时独立的持久化 Agent 架构**，将 Agent 的连续性载体（身份、记忆、代码）与可替换的执行底物（推理模型、编排 harness、主机服务器）及交互表面分离，定义了一套六个不变量的授权迁移协议，并通过开源实现 Enoch 验证了机制的可行性。

## 研究问题与动机
- **Agent 边界问题**：现有系统通常将 Agent 等同于当前运行的模型+harness，这只能回答"此刻谁在产生行为"，却无法回答跨时间、跨部署的同一性问题——更换模型后是迁移还是死亡？复制记忆后两个进程哪个是原始 Agent？
- **缺乏生命周期形式化**：重启、恢复、迁移、演化、复制、fork 等操作常被混为一谈，缺少统一的语义区分和转换协议。
- **可移植性与控制权困境**：多 Provider 绑定导致供应商锁定；私有状态与公开代码边界不清；迁移时权限、凭证、权威的传递缺乏安全保障。
- **行为保真 ≠ 系统连续**：新模型可能产生不同的输出风格，但不能因此否定 Agent 的连续性；反之，复制的 Agent 即使行为相似也不应自动继承权威。

## 核心贡献（创新点）
1. **运行时独立的 Agent 边界定义**：将持久化底物 $\mathcal{P}_t = (I_t, M_t, B_t)$ 与可替换执行底物 $\mathcal{E}_t = (R_t, H_t, D_t)$ 及交互表面 $S_t$ 正交分离，明确"Agent 不是模型/harness/聊天会话"。
   → 与已有工作的本质区别：此前工作关注单次执行的 agent 架构（如 CoALA、AutoGen），本文关注**跨时间/跨部署的纵向边界**，回答了"什么必须在变化中保持连续"。

2. **六种生命周期操作的形式化语义**：区分 Restart、Resume、Migrate、Evolve、Identity update、Replicate、Descend/fork，并定义每种操作的连续性后果。
   → 与已有工作的本质区别：分布式系统的 checkpont/restart 关注进程状态，本文引入**身份谱系、私有序列记忆、版本化代码体、人类监护权**四个维度，形成面向 AI Agent 的生命周期理论。

3. **六不变量授权迁移协议（Quiesce→Checkpoint→Validate→Bind→Rehydrate→Resume）**：定义了从旧执行环境到新的执行环境的原子性迁移事务，含明确故障语义。
   → 与已有工作的本质区别：Continuity Kernel [14] 关注分支内的权威状态选择，本文扩展到**整个 Agent 底物的跨部署绑定**，涵盖交互表面替换和能力差异处理。

4. **开源参考实现 Enoch**：实现了可复用版本化代码体 + 已安装身份/记忆 + Provider 合同 + 权威 fencing + 迁移回滚机制，支持 Telegram/Slack 聊天、Git VCS、launchd/systemd 服务、Codex runtime 等多种 Provider。
   → 与已有工作的本质区别：Portable Agent Memory [13] 仅传输记忆状态，本文传输**完整的 $(I, M, B)$ 底物及延续权威**。

## 方法详解
**核心模型**：
$$\boldsymbol{\mathcal{P}}_t = (I_t, M_t, B_t) \quad \text{（连续载体底物）}$$
$$\mathcal{E}_t = (R_t, H_t, D_t) \quad \text{（可替换执行底物）}$$
$$S_t = \{s_{t,1}, \ldots, s_{t,n}\} \quad \text{（交互表面集合）}$$
$$\mathcal{A}_t = \mathcal{P}_t \circ (\mathcal{E}_t, S_t) \quad \text{（部署执行，∘ 表示实例化而非拥有）}$$

**迁移时的连续性条件**（纯运行时迁移从 $t$ 到 $t+1$）：
$$\nu_I(t+1) = \nu_I(t), \quad \nu_M(t) \preceq \nu_M(t+1), \quad \nu_B(t+1) = \nu_B(t)$$
其中 $\preceq$ 表示可审计的延续而非字节相等——记忆可通过 checkpoint 元数据、schema 迁移或新经验扩展而不重置谱系。

**六个连续性不变量**：
- **I1 身份与谱系**：迁移保留已安装身份版本和可归因谱系，更新需经声明的治理流程。
- **I2 记忆**：记忆和持续性相关工作流状态扩展或合法迁移其记录谱系，不被静默重置。
- **I3 代码体**：目标执行相同版本代码体，除非声明经治理的版本演化。
- **I4 权威**：在受管部署边界内，至多一个协作执行持有延续权威并产生权威外部效应。
- **I5 能力**：环境能力变化可见，不伪装为身份变更。
- **I6 自我描述**：执行/进程/交互表面/部署标签仅为环境元数据，不覆盖已安装身份。

**六阶段迁移协议**：
1. **Quiesce and fence**：停止接收新工作，推进权威 epoch，使旧执行和 Provider 拒绝旧 epoch 下的工作。
2. **Checkoint**：捕获身份版本、代码修订、记忆和工作流状态版本、待处理工作、Provider 游标；密钥以引用而非导出方式存储。
3. **Validate**：验证 schema、hash、声明谱系、目标能力要求。
4. **Bind**：通过代码体合同解析目标 Provider 和凭证；Provider 原生标识符保持目标本地化。
5. **Rehydrate**：原子安装/迁移私有状态，分开加载代码体和身份，创建新鲜 harness 和交互会话。
6. **Verify and resume**：运行健康和机械连续性检查，获取新权威 epoch，调停模糊效应，恢复持久任务。

**Provider 中立绑定设计**：应用逻辑接收归一化操作和 opaque Provider 身份；每个 Provider 声明 kind、合同版本、稳定身份和支持能力；能力丢失必须 fail-closed。

## 实验与结果
- **数据集/评测基准**：论文未提供行为保真度 benchmark，仅在 Section 6 定义了完整迁移评估协议框架，指出这是下游评测问题。
- **机制验证**：冻结于 commit `c8013ed249bc11bc13f3843ed0f0cb9729f858c1`（2026-08-31）的干净环境运行：
  - 通过 **833 个核心测试** + **92 个 Provider 和库测试**（独立执行）。
- **已验证的机制证据**（Table 3）：
  - 已安装身份/代码体分离：独立 schema 验证和启动渲染， demonstrated `self.json` 和 `body.yaml` 分离加载。
  - 状态迁移与回滚：备份、幂等性、manifest-last commit、失败提交恢复。
  - 聊天表面替换：Telegram 和 Slack 两套实现 + 结构化和事件归一化测试。
  - 主机服务替换：`launchd` 和 `systemd` Provider 暴露相同生命周期并通过隔离套件。
  - 运行时替换表面：可選択 Provider 注册表、类型合同、能力检查、会话密钥、取消、超时、fake-runtime 切换。
  - 权威交接与旧执行围栏：daemon epochs、旧 token 拒绝、持久通知调停。
  - 单轴替换：已部署推理版本、交互表面、主机机器替换并保留连续载体状态。
- **关键结论**：证据支持**机械可替换性**和**授权系统连续性**，不支持行为保真度或穷举配对评测。

## 相关工作脉络
1. **CoALA [3] / AutoGen [4]**：语言 Agent 架构框架，关注单次执行的模块化组成与多 Agent 交互；本文聚焦**执行组件变化时的纵向边界**。
2. **Actor 系统 / Orleans [2,5]**：提供逻辑地址稳定性和跨服务器迁移；本文在此基础上加入**身份表示、自传式记忆、版本化代码体、人类监护**。
3. **Checkpoint/Restart [6] / Durable Functions [8]**：进程状态恢复和serverless工作流持久化；本文扩展至**AI Agent 特有的谱系追踪和权威唯一性**。
4. **Generative Agents [9] / Voyager [10]**：持久记忆系统；本文将记忆与身份、代码体**联合视为连续载体**，并加入迁移权威。
5. **Portable Agent Memory [13]**：跨异构 Agent 的加密验证记忆转移；转移单元仅为记忆状态，本文为**完整 $(I,M,B)$ 底物 + 代码谱系 + 延续权威**。
6. **Continuity Kernel [14]**：授权状态头谱系和原子激活；管理分支内的权威选择，本文扩展到**跨部署边界的全 Agent 迁移语义**。

## 局限性与未来方向
- 参考实现仅有单一 live 推理 harness（Codex），第二个 harness 仅为扩展点，尚无线上多 Provider 交叉评测。
- 未运行控制的全部维度矩阵实验，未测量行为连续性、停机时间、成本和操作负担。
- 权威不变量仅适用于共享/遵守权威存储的协作执行；无法阻止持有未撤销凭证的脱离副本产生边界外效应，需 credential rotation 或外部权威服务。
- $(I, M, B)$ 分解是功能性的而非完整身份理论；单权威协议留下**协调多 embody、分歧记忆合并、不可逆外部效应后恢复**等问题待解决。
- 论文明确指出自身**不构成行为 identity benchmark**，行为保真度评测是下游开放问题。

## 研究启发与可借鉴点
1. **底物-执行-交互三层正交架构**：可将此分层思路迁移到任何需要"跨环境迁移且保持同一性"的 Agent 系统设计，尤其是企业级个人 Agent 的场景。
2. **六阶段迁移协议的 fail-closed 设计**：每个阶段都有明确的前置条件和回滚语义，可作为构建可信 Agent 部署流水线的参考模板。
3. **身份与代码体分离加载（self.json vs body.yaml）**：避免"复制代码即复制 Agent"的误区，为多 Agent 共享同一代码库但保持独立身份/记忆提供清晰机制。
4. **权威 epoch + fencing 机制**：借鉴分布式系统 lease 思想解决并发 Agent 实例的权威冲突，值得在多租户 Agent 平台中深入探索。
5. **机械连续性与行为保真度的明确区分**：为 Agent 评测提供了概念框架——迁移成功不保证行为一致，两者需独立验证。

## 关键术语表
- **Persistent Substrate $\mathcal{P}_t$**：Agent 的连续载体底物，由身份 $I_t$、私有序列记忆 $M_t$、版本化代码体 $B_t$ 三元组构成。
- **Execution Substrate $\mathcal{E}_t$**：可替换的执行底物，包含推理器 $R_t$、编排 harness $H_t$、主机 $D_t$，决定 Agent 的当前能力和行为风格。
- **Interaction Surface $S_t$**：Agent 与外部交换事件的接口集合，如聊天账号、API endpoint、图形界面等。
- **Continuity Invariant**：迁移过程中必须保持不变的六个属性（身份谱系、记忆谱系、代码版本、权威唯一性、能力可见性、自我描述分离）。
- **Authorized Migration Protocol**：六阶段原子迁移协议（Quiesce→Checkpoint→Validate→Bind→Rehydrate→Resume），确保迁移的事务性和回滚能力。
- **Authority Epoch / Fencing**：用于确保同一时刻至多一个执行持有权威的唯一令牌机制，旧 epoch 下的请求被拒绝。
- **Behavioral Identity Fidelity**：迁移后新执行在多大程度上回忆起、组合并 enact 其身份的度量，与系统连续性正交。
- **Provider Contract**：抽象的 Provider 接口声明，包含 kind、合同版本、稳定身份、支持能力，使应用逻辑与具体实现解耦。

## 可复现要素
- **代码/实现**：开源，参考实现在 `github.com/our-ark/enoch`，论文使用冻结 commit `c8013ed249bc11bc13f3843ed0f0cb9729f858c1`（2026-08-31）。
- **数据集**：论文未引入新数据集，评测为机制测试（833 核心 + 92 Provider/库测试）。
- **关键超参**：论文未提及传统 ML 超参；核心参数为 Provider 合同版本、schema 版本、authority epoch 管理策略。
- **测试环境**：CPython 3.12.13，clean-room 运行。
