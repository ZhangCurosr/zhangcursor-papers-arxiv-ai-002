---
title: "Terminal-Universe-Turning-Agent-Trajectories-into-Scalable-T"
source: https://arxiv.org/pdf/2609.04148v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:55:46"
field: "终端Agent训练数据构建"
keywords: ["terminal agent", "trajectory reconstruction", "environment synthesis", "SFT data scaling", "cross-workspace task", "multi-round agent"]
innovations: ["将Agent轨迹逆向还原为可复用可执行终端环境，通过确定性重放+智能体补全两阶段重建工作区", "提出广度（跨工作区依赖挖掘）与深度（多轮User Agent迭代）双轴任务扩展框架", "引入红测驱动的智能体验证器自动筛选高质量SFT数据"]
benchmarks: ["Terminal-Bench 2.1", "EvoCode-Bench v2"]
---

# 论文速读：Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

## 一句话总结
本文提出 Terminal-Universe，一种将 Agent 执行轨迹"逆向还原"为可重复利用的可执行终端环境的方法，并通过确定性重放 + 智能体补全重建工作区，在此基础上沿广度（跨工作区）和深度（多轮交互）两个轴扩展任务，最终用 32K 条过滤后轨迹微调 Qwen3.5-27B，在 Terminal-Bench 2.1 和 EvoCode-Bench v2 MT@4 上分别提升 11.9 分和 13.8 分。

## 研究问题与动机
- **可执行环境稀缺**：终端 Agent 轨迹大量积累，但可供后训练使用的真实、可执行工作区极少，二者价值不对等。
- **轨迹 vs 环境的本质差异**：轨迹是单一冻结示范，质量受限于产生它的策略模型，无法复用验证；环境则可被反复重询、生成多条可验证任务，并提供执行反馈，是更有价值的训练资源。
- **已有环境构建路线的局限**：仓库历史型方法（SWE-Gym、R2E-Gym）依赖人工或高成本仓库维护；扰动型方法（SWE-smith、CLI-Gym）只能构造修复类任务；从零合成的方法（Endless Term.、TMax）生成环境真实性不足、规模受限。
- **轨迹中的工具调用已暴露环境结构**：Read/Write/Edit 等操作足以重建工作区，但从未被系统性地用作环境来源——这是本文的核心观察。

## 核心贡献（创新点）
1. **轨迹即环境的范式反转**：将 Agent 轨迹视为可重建环境的观测记录，通过"确定性重放 + 智能体补全"两阶段恢复原始工作区，无需原始仓库或从零构建。
2. **四象限重询机制**：提出 Intent Recovery（意图还原）、Single-WS（单工作区新任务合成）、Cross-WS（跨工作区依赖挖掘）和 Multi-Round（多轮用户交互扩展）四种互补的任务生成方式。
3. **双轴扩展策略**：广度上通过 TF-IDF + LLM 判定建立工作区间方向性依赖关系并构造跨仓库任务；深度上引入 User Agent 维持需求追踪器并生成 feature extension/revision/conflict 三类多轮请求。
4. **验证驱动的自动筛选**：每项任务由容器内智能体编写可执行 pytest 验证器，仅保留全部测试通过的轨迹，保证数据质量。
5. **显著的效果验证**：在 32K 条 SFT 数据上微调 Qwen3.5-27B，Terminal-Bench 2.1 达 58.1%（+11.9）、EvoCode-Bench v2 MT@4 达 20.1（+13.8），消融证明重建环境重解远优于直接模仿原始轨迹。

## 方法详解
### 环境重建（§3.1）
- **Stage 1 确定性重放**：按时间顺序处理轨迹中的 Read/Write/Edit 操作，对每个被访问路径取 agent 首次修改前的最早版本作为 $\widehat{E}_0$；agent 新建和修改的文件单独保留用于后续验证，$\widehat{E}_0$ 仅为部分工作区。
- **Stage 2 智能体补全**：Completion Agent 在不移植解决方案的前提下，创建缺失文件、补全残缺文件、恢复依赖（包安装、配置文件等），使任务可求解。
- **Stage 3 环境筛选**：只读 Shell + 文件工具的 Agentic Judge 评估工作区是否具备足够项目上下文（源码/配置/数据/结构），仅保留 sufficient 环境。

### 四类重询机制（§3.2）
- **Intent Recovery**：从多轮对话中合并出单一自洽任务描述，只保留用户提出的要求，用 Agent 行为和文件证据辅助理解。
- **Single-WS**：离线生成器在每个工作区综合下合成 5 个候选任务，随机选 1 个进行 Teacher Rollout 并验证。
- **Cross-WS（广度）**：对每个工作区做 TF-IDF 向量检索找近邻，LLM Judge 判定方向性依赖关系（目标缺某能力、参考实现中有），每对分配一个任务，目标可写、参考只读挂载，要求solver自行发现至少 3 个隐藏在参考仓库中的具体事实。
- **Multi-Round（深度）**：初始查询完成后引入 User Agent，维护显式 Requirement Tracker（active/satisfied/updated），每轮由自动化 Verifier 编写验收测试，User Agent 解读测试结果并以自然语言反馈给 Coding Agent，共 3 种风格：feature extension（62.7%）、feature revision（29.3%）、feature conflict（8.0%），最多 6 轮。

### 验证与过滤（§3.3）
- 每个 Single-WS/Cross-WS 任务由容器内 Verifier Agent 编写 pytest 套件（含红测保证当前状态不满足新能力、白测保证现有行为不变）。
- Teacher（Qwen3.7-Max）在 Claude Code scaffold 中 rollout 解，temperature=1.0、top-p=0.95、256k 上下文，最多 500 turn、4h 超时。
- 多轮场景只保留至少 2 轮通过且含至少 1 次失败修复的会话，截去尾部连续失败轮。

## 实验与结果
- **数据来源**：6 个公开 corpus（SWE-rebench、SWE-smith、CoderForge、SWE-Gym、LFM2-Terminal、LiteCoder-Terminal），共 359,593 条轨迹；严格排除 Terminal-Bench 派生源防泄漏。
- **重建统计**：产出 68,263 个重建环境；智能体补全后文件数均值从 2.9 增至 22.4；经充分性筛选得 37,273 个 task-sufficient 环境。
- **SFT 语料**：32K 条（Single-WS 25,386、Cross-WS 3,512、Multi-Round 3,079），约 1.42B tokens。
- **Terminal-Bench 2.1**：Qwen3.5-27B base 46.2 → 58.1（**+11.9**，Terminus2-XML）；Claude Code scaffold 下 47.8 → 58.2（+10.4）。
- **EvoCode-Bench v2 MT@4**：base 6.3 → 20.1（**+13.8**）；Case Score 67.8 → 76.1。
- **消融关键结论**：
  - 重解 vs 原始轨迹 SFT：52.1 vs 36.7（+15.4 绝对提升）
  - 补全必要性：Replay+Completion 52.9 vs Replay-only 48.7（+4.2）
  - 验证器筛选：Cross-WS 有过滤 55.4 vs 无过滤 53.2
  - 预算分配：扩展环境数最有效（53.2→56.0），扩展查询/解几乎无效（53.8/53.9）

## 相关工作脉络
- **SWE-Gym / R2E-Gym**：从仓库历史 issue/commit 重建环境，需要原始仓库；本文从轨迹反推，无需仓库。
- **SWE-smith / CLI-Gym**：在健康工作区注入故障构造任务，任务类型限于修复；本文可生成任意类型任务。
- **Endless Term. / TMax / CLI-Universe**：从零合成环境+任务，真实性取决于生成器；本文环境来自真实轨迹还原，具有真实代码上下文。
- **SkillSynth / Terminal-World**：基于技能图/技能 taxonomy 组织生成；本文基于依赖关系挖掘与多轮交互扩展，覆盖更广的实操场景。
- **SWE-INTERACT / EvoCode-Bench**：多轮交互评测基准；本文的 Multi-Round 深度扩展与之对齐，但以 Agent 生成驱动迭代而非静态会话回放。
- **RST / CalibForge**：递归演化或对抗校准提升任务质量；本文通过双轴扩展和验证器筛选实现质量与规模的双重提升。

## 局限性与未来方向
- 使用标准 ubuntu:24.04 容器而非仓库专用镜像，需要特殊依赖/复杂编译的场景保真度受限。
- 工作区的领域/语言/工具链分布受限于源轨迹覆盖范围，难以覆盖未采样的工作负载。
- 任务、解、验证器均由单一 Teacher 生成，其能力瓶颈可能限制任务多样性，且 Teacher 错误可能绕过测试；未来可引入多 Teacher 与独立验证器模型。
- 复杂度与源轨迹质量正相关，未来随 Agent 能力提升可自然扩展。
- 仅验证了 SWE→Terminal 的跨域迁移，反向迁移留作未来工作。

## 研究启发与可借鉴点
1. **"轨迹→环境"的反向映射思路可迁移**：任何记录完整工具调用历史的 Agent 交互数据（如 GUI Agent、Web Agent）均可尝试类似重建思路，将"观察"转化为"可操作的实验环境"。
2. **双轴扩展设计（广度+深度）值得复用**：广度用依赖关系挖掘连接多工作区，深度用 User Agent 模拟迭代反馈，两者正交可叠加，适用于其他需要持续迭代的 Agent 训练场景。
3. **红测驱动（Red Check）的验证器设计**：要求验证器在未修改工作区上至少一条新能力测试失败，可有效排除"假任务"（即任务在当前工作区已满足），保证任务的真实挑战性。
4. **预算分配实验的启示**：扩展环境数 > 扩展查询/解，说明训练数据的边际价值主要来自新颖的执行上下文而非重复上下文下的多种解法，可为后续数据构建的投入优先级提供指导。
5. **SWE 到 Terminal 的跨域泛化验证**：证明重建管道生成的数据可跨越数据分布迁移学习，暗示该框架可用于更多细分子领域的场景扩充。

## 关键术语表
- **Deterministic Replay（确定性重放）**：按时间顺序处理轨迹中的文件读写操作，恢复 agent 首次修改前的每个文件版本，重建部分工作区。
- **Agentic Completion（智能体补全）**：用 LLM Agent 在部分工作区基础上创建缺失文件、补全残片、安装依赖，使任务可执行但不泄露解法。
- **Intent Recovery（意图还原）**：从多轮对话轨迹中提取自洽的原始用户任务描述，过滤掉非用户提出的约束。
- **Cross-WS（跨工作区合成）**：基于 TF-IDF + LLM 判定找到工作区间方向性依赖关系，构造一个需读取只读参考仓库并修改可写目标仓库的复合任务。
- **Multi-Round Depth Expansion（多轮深度扩展）**：在初始任务完成后由 User Agent 根据实际通过/失败情况生成延续请求，形成多轮迭代会话。
- **Red Check（红测）**：验证器必须在初始（未修改）工作区上至少一条新能力测试失败，证明任务确实存在能力缺口。
- **MT@4（Multi-Turn@4）**：EvoCode-Bench v2 的多轮指标，以 fail-stop 计分，仅统计首次失败前连续通过的轮次。
- **Workspace Sufficiency（工作区充分性）**：Agentic Judge 判断工作区是否包含足够的项目上下文（源码/配置/数据/结构），使一个强 Agent 能着手求解任务。

## 可复现要素
- **数据集**：37,273 个 task-sufficient 环境，32K SFT 演示数据；论文未明确公开数据集链接（文末未提供开源声明）。
- **代码/权重**：论文未提及开源；模型命名为 Terminal-Universe-27B，基于 Qwen3.5-27B 微调。
- **关键超参**：学习率 $7 \times 10^{-6}$，global batch size 256，序列长度 256k，训练 2 epochs；Teacher rollout temperature=1.0，top-p=0.95，256k 上下文，turn 上限 65,536 tokens，最大 500 turn，4h 超时。
- **评估配置**：Terminus2-XML parser，6 次独立运行取均值（Terminal-Bench），4 次运行（EvoCode-Bench）。
