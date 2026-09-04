---
title: "Terminal-Universe-Turning-Agent-Trajectories-into-Scalable-T"
source: https://arxiv.org/pdf/2609.04148v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:08:51"
field: "终端Agent训练数据构建"
keywords: ["terminal agent", "environment reconstruction", "trajectory-to-environment", "task synthesis", "SFT data scaling", "multi-round agent"]
innovations: ["从agent轨迹反向重建可执行工作空间环境", "跨工作区依赖挖掘与多轮用户仿真的双轴任务扩展", "agent-authored verifiable pytest suite闭环筛选"]
benchmarks: ["Terminal-Bench 2.1", "EvoCode-Bench v2"]
---

# 论文速读：Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

## 一句话总结
论文提出 Terminal-Universe 框架，利用现有 agent 轨迹中记录的工具调用历史反推并重建可执行的工作空间环境，进而沿广度（跨仓库依赖）和深度（多轮交互）两个轴合成新任务，最终生成 37.3k 个任务充足的环境用于 SFT，使 Qwen3.5-27B 在 Terminal-Bench 2.1 上提升 11.9 分。

## 研究问题与动机
- **环境稀缺而轨迹过剩**：终端 agent 积累了大量交互轨迹，但可用于 post-training 的真实可执行环境极少；轨迹是单一冻结示范，而环境可反复查询、验证、扩展。
- **已有环境构建方法的不足**：基于 git 历史的恢复仅能生成修复类任务；扰动方法受限于注入bug的种类；从零合成的环境缺乏真实项目上下文，规模与真实性受限。
- **轨迹本身携带环境信息**：轨迹中记录的文件读写操作暴露了工作区的结构与内容，使得从轨迹反向重建环境成为可能，而现有工作尚未充分利用这一资源。
- **需要可验证、可扩展的训练数据**：高质量 post-training 需要大量带 verifiable 任务的真实环境，人工构建难以满足规模需求。

## 核心贡献（创新点）
1. **轨迹到环境的反向映射**：将 agent 轨迹重构为可复用执行环境，通过确定性回放 + 智能体补全恢复完整工作区，无需原始仓库或从零生成，与已有从环境生成轨迹的范式相反。
2. **双轴任务扩展机制**：提出广度扩展（跨工作区依赖挖掘）与深度扩展（用户智能体驱动的多轮交互）两种合成路径，模拟真实软件工程场景。
3. **Agent-authored verifier 闭环**：每个新任务由容器内专用 agent 编写 pytest 套件验证，仅保留所有测试通过的轨迹，确保训练数据质量。
4. **大规模实证验证**：生成 37.3k 任务充足环境、32.0k SFT 样本，Qwen3.5-27B SFT 后在 Terminal-Bench 2.1 上 +11.9、EvoCode-Bench v2 MT@4 上 +13.8，且消融证明重解优于直接模仿原始轨迹。

## 方法详解
**环境重建（三阶段）**
- **确定性回放**：按时间顺序处理轨迹中的 read/write/edit 操作，恢复每个被访问路径的"智能体首次修改前"的文件状态，排除智能体新建文件与后续变更，得到部分工作区 $\widehat{E}_0$。
- **智能体补全**：completion agent 在不泄露解决方案的前提下，为缺失的项目文件、配置、依赖创建合理内容，使任务可解；补全后工作区记为 $\widehat{E}$。
- **充分性过滤**：只读 inspect 工作区，判断其源码/配置/数据/结构是否足以支撑任务，保留 task-sufficient 环境。

**四种 Re-querying 机制**
- **Intent Recovery**：从多轮对话中提取用户意图，规范化为自包含任务指令。
- **Single-WS 合成**：对单个工作区探索并生成 5 个候选任务，随机选取 1 个。
- **Cross-WS 广度扩展**：对每个工作区生成技术画像（domain/subdomain/frameworks/core_capabilities 等），通过 TF-IDF 检索候选对，LLM judge 识别"目标缺少参考已实现能力"的依赖边，每对生成 1 个跨仓库任务，要求 solver 自主阅读并适配参考实现。
- **Multi-Round 深度扩展**：初始任务完成后引入 user agent，维护显式需求跟踪器；每轮自动生成 acceptance test 后再请求 solver 行动；失败时以自然语言反馈，成功时延续新需求；最多 6 轮，三种交互风格（feature extension / revision / conflict）。

**验证与筛选**
- Verifier agent 在容器内基于任务规范和初始工作区编写独立 pytest 套件，要求红色检查（未修改工作区上新功能测试必失败、保留测试必通过）。
- Teacher（Qwen3.7-Max）在 Claude Code scaffold 内 rollout，temperature=1.0、top-p=0.95、256k 上下文、单回合上限 65,536 token、最多 500 回合、4h wall-clock。
- Single-WS/Cross-WS 全部测试通过才保留；Multi-Round 保留含至少两个通过回合的会话，截断尾部连续失败回合。

## 实验与结果
- **数据集与来源**：从 SWE-rebench、SWE-smith、CoderForge、SWE-Gym、LFM2-Terminal、LiteCoder-Terminal 共 359,593 条轨迹中重建，经去污与仓库级去重后获得 68,263 个重建环境，其中 37,273 个被判定为任务充分。
- **SFT 语料构成**：31,977 条经过 verifier 筛选的轨迹（25,386 Single-WS + 3,512 Cross-WS + 3,079 Multi-Round），总计约 1.42B tokens。
- **评估基线**：Terminal-Bench 2.0/2.1（单轮）、EvoCode-Bench v2（多轮）。
- **主要结果**：
  - Terminal-Bench 2.1（Terminus2-XML）：Qwen3.5-27B 基础 46.2 → 训练后 58.1，**提升 +11.9**；Claude Code scaffold 下 47.8 → 58.2（+10.4）。
  - EvoCode-Bench v2：MT@4 从 6.3 → 20.1（+13.8），Case score 从 67.8 → 76.1。
  - 在 Terminal-Bench 2.1 所有 task synthesis 方法中取得最高 58.1（TaskSynth 类别最强）。
- **消融结论**：重解（52.1）远优于直接模仿源轨迹（36.7）；补全比仅回放高 4.2 分；verifier 过滤对更难任务（Cross-WS）效果显著；环境扩展比查询/解扩展更有效。

## 相关工作脉络
- **SWE-Gym / R2E-Gym**：从仓库 git 历史恢复环境，侧重 issue 与 commit 关联；本文从轨迹反向重建，不依赖原始仓库版本控制。
- **SWE-smith / CLI-Gym**：在健康仓库上注入 bug 进行扰动；任务类型局限于修复，覆盖受限于注入策略；本文支持任意类型任务并在多轴上扩展。
- **Endless Term. / TMax / CLI-Universe / SkillSynth**：从任务条件或技能图从零合成环境；环境真实性依赖生成器；本文环境源自真实轨迹暴露的内容，真实性更高。
- **RST / CalibForge / FACET**：递归扩展或对抗校准任务；本文强调从轨迹反推环境后再重新 querying，视角不同。
- **SWE-INTERACT / SWE-Together / EvoCode-Bench**：多轮交互评估框架；本文方法可与这些 benchmark 天然结合，提供训练数据生成路径。

## 局限性与未来方向
- **统一 Ubuntu 容器的保真度限制**：未针对每个仓库定制镜像，特殊系统依赖或复杂编译场景可能降级；可结合 SWE-Factory/RepoLaunch 改善。
- **分布受限于源轨迹覆盖**：语言/领域/工具链分布由收集到的轨迹决定，扩展到其他领域需补充数据源。
- **单 teacher 的能力瓶颈**：任务、解、verifier 均由单一 teacher 生成，能力缺口可能导致任务覆盖不足或错误未被发现；未来可用多 teacher 及独立 verifier 模型。
- **轨迹复杂性决定上限**：更丰富的轨迹（多文件操作、长工具链）产生更高质量环境，随 agent 能力提升数据供给会自然增长。

## 研究启发与可借鉴点
- **"轨迹作为环境探针"的反向视角**：将轨迹视为环境的观测证据而非最终产物，这一范式可直接迁移到代码 agent、GUI agent、多模态 agent 的训练数据构建。
- **双轴扩展设计（广度 + 深度）**：跨工作区依赖挖掘与多轮用户仿真两种正交扩展轴，可作为通用 framework 模板用于其他 agent 场景的任务合成。
- **Agentic verifier 作为质量门控**：在容器内由 agent 编写并迭代调试 pytest 套件，确保任务可验证且保留测试通过，可迁移到任何需要自动验证的训练数据生成流程。
- **环境优先于轨迹的训练直觉**：消融表明重解优于模仿原始轨迹，提示后续工作应优先考虑构建可执行环境而非单纯积累 trajectory 数据。
- **TF-IDF + LLM judge 的组合检索策略**：用结构化画像做粗筛、LLM 做细判，兼顾效率与语义匹配，可用于跨仓库/跨领域知识复用任务构建。

## 关键术语表
- **Terminal-Universe**：将 agent 轨迹重构为可执行工作空间并沿多轴合成任务的框架。
- **Deterministic replay**：按时间顺序回放轨迹中的文件读写操作，恢复智能体修改前的文件状态。
- **Agentic completion**：由 completion agent 补全轨迹未覆盖的缺失文件与依赖，使环境可运行但不出示解。
- **Intent Recovery**：从源对话中提取用户原始意图，规范化为自包含任务指令。
- **Single-WS / Cross-WS / Multi-Round**：三种任务合成变体，分别在单工作区内、跨多个工作区、以及多轮用户交互中生成新任务。
- **Feature extension / revision / conflict**：多轮对话中 user agent 的三种交互风格，分别对应新增需求、修复缺陷、替换需求。
- **Verfier agent**：在目标容器内根据任务规范自动生成 pytest 套件的专用 agent，执行红色检查确保测试有效性。
- **MT@4 / Case score**：EvoCode-Bench v2 的两项多轮评估指标，前者衡量连续通过轮数，后者衡量验证器用例通过率。

## 可复现要素
- **数据集来源**：SWE-rebench、SWE-smith、CoderForge、SWE-Gym、LFM2-Terminal、LiteCoder-Terminal（均公开）；已排除 Terminal-Bench 派生数据。
- **代码/权重开源**：论文未明确声明代码仓库链接；模型权重通常随 Qwen 系列开源政策发布。
- **关键超参**：SFT 学习率 $7 \times 10^{-6}$、全局 batch size 256、序列长度 256k、训练 2 epochs；rollout temperature 1.0、top-p 0.95、256k 上下文、单回合上限 65,536 token、最多 500 回合、4h wall-clock。
- **评估配置**：Terminus2-XML parser；Claude Code scaffold 禁用交互/网页检索工具；每容器 12 CPU / 32 GiB；Terminal-Bench 6 次独立运行取均值。
