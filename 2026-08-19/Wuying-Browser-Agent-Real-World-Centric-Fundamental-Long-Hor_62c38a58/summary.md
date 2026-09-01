---
title: "Wuying-Browser-Agent-Real-World-Centric-Fundamental-Long-Hor"
source: https://arxiv.org/pdf/2608.17319v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:06:17"
field: "浏览器智能体与在线强化学习"
keywords: ["Browser Agent", "Long-Horizon", "GRPO", "Curriculum SFT", "Reward Shaping", "Web Benchmark", "Open-Source Agent"]
innovations: ["RUIC-SFT课程化监督融入错误恢复与复杂UI专项数据", "DAO-GRPO通过PBRS与分歧感知步级Credit提升长程在线RL", "BrowserBench填补中英双语长程真实网页评测空白"]
benchmarks: ["WebVoyager", "Online-Mind2Web", "BrowserBench", "Tau2-Bench", "BFCL-v4", "Claw-Eval"]
---

# 论文速读：Wuying-Browser-Agent-Real-World-Centric-Fundamental-Long-Horizon-Browser-Agents

## 一句话总结
本文针对真实部署中长程浏览器 Agent 的训练-部署鸿沟，提出端到端统一框架 Wuying-Browser-Agent，通过结构化 Browser Harness、面向错误恢复的 RUIC-SFT 课程训练与 DAO-GRPO 在线 RL，显著提升了长步数、多分支、中英文混合场景下的浏览器智能体性能。

## 研究问题与动机
1. **训练数据偏向成功轨迹，缺乏错误恢复监督**：现有 Agent 训练集以成功示范为主，当 Agent 在长程真实交互中误操作时，几乎没有学习如何识别错误并返回可行路径的示例，导致基座 SFT 模型的步骤恢复成功率仅 8.5%。
2. **长轨迹 credit assignment 稀疏且难以定位关键分支步**：真实浏览器任务动辄数十步，多个 rollout 共享长前缀，仅在少数分支决策步出现分歧；若对整条轨迹均匀优化，学习信号被稀释，无法精准强化决定性步骤。
3. **评估基准以英文短程任务为主，双语长程真实网页覆盖不足**：现有 Online 基准（WebVoyager、Online-Mind2Web 等）任务步数多在 8-15 步，中文网页覆盖率接近为零，难以暴露长程失败模式。
4. **浏览器 context 非追加式，动态状态变化带来 train-test mismatch**：浏览器交互在导航后 DOM snapshot 会失效或替换，若直接在完整 transcript 上优化而非按实际决策 context 重建，策略会被过时的无关信息干扰。

## 核心贡献（创新点）
1. **端到端共设计管道（Execution / Supervision / Optimization / Evaluation）**：将执行子层、监督数据构造、在线 RL 与评估基准围绕同一长程部署场景统一设计，而非仅靠模型规模扩展；区别于以往单点改进方法。
2. **结构化 Browser Harness 层 + 统一工具空间**：构建含 24 种原子操作的 structured tool space，并通过 task-level state management 实现 decision-oriented 的上下文重建，在 SFT/RL/评测中共享相同接口；与通用 GUI agent 相比更强调浏览器原生状态管理。
3. **RUIC-SFT 课程化监督初始化**：将 UI-specialized 交互数据与 reflection-rich 错误恢复数据按三阶段渐进曲线混入，早期稳定基础操作、中期强化复杂控件交互、后期引入自我纠正；与简单固定比例或顺序训练相比显著提升鲁棒性与效率。
4. **DAO-GRPO 在线 RL 优化框架**：结合 Potential-Based Reward Shaping（PBRS）与 LLM-based Divergence Estimator 在 step 层面重分配 credit，并在每个 response 下按真实决策 context 进行优化；弥补了 GRPO 类方法在稀疏终态奖励下对关键分支步无法聚焦的缺陷。
5. **BrowserBench 双语长程真实网页基准**：350 条中英双语任务、跨 254 个真实网站、平均 37.9 步，填补了现有 benchmark 在长程 + 中文覆盖上的空白；与 WebVoyager / Online-Mind2Web 等短程英文基准形成互补。

## 方法详解
### 1. 问题建模
浏览器 Agent 执行建模为部分可观测序贯决策：
- 观察 $o_t = \{S_t, V_t\}$（结构化 DOM 状态 + 可选截图）。
- 决策 context $c_t = \mathcal{R}(g, o_t, \{a_u, e_u\}_{u<t})$，由 harness 按当前任务状态重建，保留交互历史、拒绝冗余。
- 动作空间为 24 种原子操作（导航、交互、提取、文件、流程控制）。

### 2. Browser Harness 层
- **Structured Tool Space**：24 种确定性 schema 验证动作。
- **Execution Feedback**：无效调用以结构化错误返回，供策略自我纠正。
- **Task-Level State Management**：跨步骤维护任务状态与中间产物；所有 SFT/RL/Eval 使用同一 harness，保证接口一致。

### 3. RUIC-SFT 课程监督
数据集：
- $D_g$：通用浏览器任务示范。
- $D_u$：复杂 UI 控件（日期选择器、级联下拉、富文本编辑器等）专项数据。
- $D_r$：reflection 数据（包含错误步、自然语言归因、纠正策略）。

三阶段课程混合（图 4）：
- Phase 1（$D_g$ 主导）稳定基础操作。
- Phase 2 逐步引入 $D_u$ 强化复杂控件交互。
- Phase 3 加入 $D_r$ 培养自我纠正，同时降低 $D_u$ 比例。
- 损失仍为 next-token prediction，但采样分布随训练进程动态变化。

### 4. DAO-GRPO 在线 RL
三大组件协同：
- **PBRS 密集进度监督**：$r_{i,t}^{shape} = \gamma \Phi(s_{i,t+1}) - \Phi(s_{i,t})$，以任务条件进展估计 $\Phi$ 填充稀疏终态奖励；$\Phi(s_T)=0$ 不改变终态目标。
- **Divergence-Aware Step Credit Assignment**：LLM-based 估计器在多轨迹组内定位首次关键分歧步 $t^*$，计算分支重要性 $c^*$，并按公式 (7) 分配权重：
  - 共享前缀步 $\alpha_{shared}$。
  - 分歧步 $w = \alpha_{div} \cdot c^*$。
  - 分歧后正向分支衰减因子 $\delta^+$，负向分支 $\delta^-$，且 $\delta^+ > \delta^-$。
- **Response-Level 训练**：每个 response 在实时决策 context $c_{i,t}$ 下重新播放而非使用完整 transcript 累积；保留 GRPO 组相对优势 $A_i$ 的同时，避免上下文失真。

## 实验与结果
### 数据集与基准
- **WebVoyager**（643 任务，15 站点，英文，中位数 8-15 步）
- **Online-Mind2Web**（300 任务，136 站点，英文，平均 8.4 步）
- **BrowserBench**（350 任务，254 站点，中英双语，平均 37.9 步，最大 100 步）
- **通用智能体基准**：Tau2-Bench、Claw-Eval、BFCL-v4

### 主要结果（27B 规模）
| 模型 | WebVoyager | Online-Mind2Web | BrowserBench | Avg |
|---|---|---|---|---|
| **Wuying-Browser-Agent-27B** | **80.6%** | **66.7%** | **65.1%** | **70.8%** |
| Qwen3.8-Max | 77.8% | 68.9% | 64.3% | 70.3% |
| Qwen3.5-397B-A17B | 70.8% | 54.4% | 44.3% | 56.5% |
| GPT-5.5 | 85.1% | 74.7% | 67.4% | 75.7% |
| GPT-5 | 80.1% | 67.1% | 61.7% | 69.6% |

- 9B 模型 SFT→RL 提升：45.6% → 50.8%（+5.2%）。
- 27B 模型 SFT→RL 提升：62.7% → 70.8%（+8.1%）。
- 通用能力迁移：27B 模型在 Tau2-Bench / Claw-Eval / BFCL-v4 上平均 73.8，超越 Qwen3.5-27B（67.2）和 Qwen-UI-Agent-27B（72.4）。

### 消融关键
- RUIC-SFT 课程 vs. 固定混合：SR 38.0% vs. 36.6%，且平均步数 22.4 vs. 29.3。
- DAO-GRPO 全量 vs. 单组件累加：Full 42.9% > +PBRS+Div. Credit 41.9% > +PBRS 40.7% > Vanilla GRPO 39.4%。
- Hard 任务：从 18.1% 提升至 30.5%（+12.4%），长轨迹（>50 步）从 13.3% 提升至 26.7%（+13.4%）。

## 相关工作脉络
1. **WebVoyager / Online-Mind2Web**：主流英文短程在线基准，侧重一般 web 导航；本文指出其长度不足以反映真实部署，故构建 BrowserBench 补足。
2. **OpenWebRL / AgentRL / WebAgent-R1**：基于 GRPO 的多轮在线 RL 方法；本文在此基础上引入 PBRS 与 divergence-aware 步级 credit，适配浏览器动态 context。
3. **UI-TARS / FARA / MolmoWeb**：GUI/Browser 专用模型；本文强调训练数据中缺失的恢复与复杂 UI 监督，提出 RUIC-SFT 补位。
4. **OpenCUA / GUI-Owl / UI-Venus / EvoCUA**：通用计算机使用 Agent；本文聚焦浏览器长程交互的独特挑战（非追加 context、分支决策），并提出可比通用 Agent 的能力。
5. **WebArena / Mind2Web 2**：离线 / 半静态环境基准；本文采用真实在线网站与 goal-only 协议，更贴近部署。
6. **WebGym / PAE / OpenWebVoyager**：探索型与技能发现工作；本文与之互补——以 supervised + 在线 RL 双阶段闭环为主，并辅以自增强数据飞轮。

## 局限性与未来方向
1. **RL 阶段单次 rollout 上限 50 步**（评测为 100 步），训练时长截断可能损失部分极长任务的优化信号。
2. **LLM-based 分歧估计器存在噪声**：虽对 ±2 步扰动稳健，但完全随机定位时性能低于均匀 credit；精确度仍待提高。
3. **镜像阶段 26.7% 的零方差 group 持续存在**，依赖 dynamic sampling 丢弃，意味着有效 batch 始终小于名义 batch。
4. **训练数据中 reflection 与 UI 专项数据规模有限**（约 3K 初始轨迹 → 420 条 reflection、130 条 UI 场景），未来需扩展更多真实失败案例。
5. **未系统比较与闭源商业 Agent 在同等 harness 下的差距**，GPT-5.5 等仍领先 5 个百分点，真实开源差距仍需缩小。

## 研究启发与可借鉴点
1. **长程任务训练必须引入失败 / 恢复数据**：纯成功轨迹不足以训练鲁棒性，应构建 error-aware 数据集，并用课程方式分阶段引入，避免早期过度保守。
2. **PBRS + 分支定位可用于稀疏反馈的交互 Agent 训练**：不仅限于浏览器，亦可迁移到工具调用、GUI 控制等多步决策场景。
3. **Context 重建比 transcript 累积更贴合真实推理**：任何需要动态状态的非追加式交互系统均可借鉴"按实际决策信息集优化"的思路。
4. **Benchmark 难度应由实证可解性标定**：BrowserBench 采用多参考 Agent 标定 easy/medium/hard，兼顾可重复性与判据客观性。
5. **自增强数据飞轮设计**：统一 harness 使得 SFT/RL/Eval 产出同构数据，可直接用于下一轮数据收集与 preference 训练，适合持续演进的 web 环境。

## 关键术语表
**RUIC-SFT**：Reflection and UI-specialized Curriculum SFT，面向浏览器 Agent 的课程化监督微调，分三阶段整合通用、UI专项与错误恢复数据。
**DAO-GRPO**：Divergence-Aware Online GRPO，面向长程浏览器任务的在线 RL 框架，通过 PBRS 和分歧感知步级 credit 提升优化效率。
**PBRS（Potential-Based Reward Shaping）**：基于势函数的奖励塑形方法，在保持最优策略不变的前提下提供密集中间监督信号。
**BrowserBench**：作者提出的中英双语长程真实网页评测基准，350 任务、跨 254 网站、平均 37.9 步。
**Divergence Estimator**：基于 LLM 的分歧步估计器，在多轨迹组内定位首次语义分歧步以重分配 credit。
**Decision Context Reconstruction**：Harness 在每步根据当前任务指令、观察与历史重建的决策信息集，避免累积冗余上下文。
**Pass@1 Success Rate**：单次 rollout 成功求解率，用于衡量 Agent 在真实部署约束下的可靠性。
**Self-Reinforcing Data Flywheel**：基于 harness 的统一数据回路，将 SFT/RL/Eval 产生的轨迹自动分流并反哺下一轮训练。

## 可复现要素
- **数据集**：BrowserBench 350 任务由作者构建，论文未声明是否公开；原有 WebVoyager、Online-Mind2Web 等基准公开。
- **代码/权重**：Wuying-Browser-Agent 系列（4B / 9B / 27B）作为开源模型发布；具体仓库信息论文未明确。
- **关键超参**：SFT：LoRA rank=16, α=64, peak lr=5e-5, 2 epochs；RL：LoRA rank=8, α=16, trajectory 上限 50 步（评测 100 步）；PBRS 势函数由 LLM-based progress estimator 实现；分阶段混合曲线由验证集调定。
- **训练环境**：AgentBay 云沙箱，异步并行 rollout；多 GPU 集群启用 sequence parallelism。
