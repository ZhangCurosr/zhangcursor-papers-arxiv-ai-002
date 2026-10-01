---
title: "TANGO-Humanoid-Navigation-in-Cluttered-Environments-with-a-W"
source: https://arxiv.org/pdf/2609.09158v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:32:03"
field: "人形机器人全身导航"
keywords: ["Vision-Language Navigation", "Humanoid Robotics", "Whole-Body Control", "Vision-Language-Action Model", "Simulation-to-Real Transfer", "Cluttered Environment Traversal"]
innovations: ["首个全身VLA导航框架直接预测29-DoF关节空间动作", "PET管道自动生成碰撞自由全身遍历数据", "Flow-matching动作专家配合RTC实现连续实时导航"]
benchmarks: ["VLNVerse (423 seen + 825 unseen)", "增强VLNVerse杂乱场景", "Unitree G1真实世界零样本部署"]
---

# 论文速读：TANGO-Humanoid-Navigation-in-Cluttered-Environments-with-a-W

## 一句话总结
TANGO 是首个面向人形机器人的全身视觉-语言导航框架，给定自然语言指令和 RGB 观测，直接预测 29-DoF 关节空间动作，在杂乱的室内环境中实现了最先进的零样本 sim-to-real 导航性能。

## 研究问题与动机
- **人形全身导航的三维几何耦合问题**：与人形机器人不同，传统轮式/移动平台将导航建模为二维路径规划问题，但人形穿越杂乱环境需要连续协调手臂放置、躯干调整和步态调制以避免碰撞（如跨栏、侧步、蹲下）。
- **现有 VLN 方法的低维动作空间不足**：现有 Vision-Language Navigation（VLN）方法通常预测 2D 航点或离散动作，无法显式表达导航决策与全身可行性之间的关系，难以处理受约束场景。
- **人形基础模型的导航-控制解耦局限**：GR00T-N1.6、Ψ₀、WholeBodyVLA 等人形基础模型将导航表达为高层运动命令并委托给下游控制器，阻止了在导航期间对全身穿越可行性的显式推理。
- **碰撞感知遍历方法的泛化瓶颈**：HumanoidPF 等 RL 方法在特定遍历场景中有效，但依赖任务特定的先验或训练分布，难以扩展到多样化的长视野语言引导导航。

## 核心贡献（创新点）
1. **首个全身 VLA 导航框架**：提出 TANGO，直接预测 29-DoF 关节空间动作而非 2D 航点，避免了分离的导航和控制模块。
2. **Plan-Edit-Track (PET) 数据合成管道**：开发了可扩展的仿真管道，通过全局路径规划、运动编辑和 RL 跟踪自动生成碰撞自由的全身遍历轨迹，提供了动态可行的监督信号。
3. **Flow-matching 动作专家与系统融合架构**：将 Qwen2.5VL-7B 骨干网络与多模态扩散 Transformer (MM-DiT) 动作专家结合，通过 real-time chunking (RTC) 对齐离线训练与在线流式执行。
4. **零样本真实世界部署验证**：在 Unitree G1 人形机器人上实现零样本部署，无需任何真实世界导航数据训练，展示了鲁棒的语言引导穿越能力。
5. **统一的仿真-真机评估基准**：建立了 VLNVerse 基准（423 seen + 825 unseen），并引入碰撞率 (CR) 指标量化杂乱环境中的导航安全性。

## 方法详解
**Problem Formulation**：给定自然语言指令 $\ell$、当前观测 $\mathbf{o}_t$（含前后相机的 RGB 序列 $\mathbf{I}_{1:t}^{\mathrm{fr,dn}}$）和全身关节角 proprioceptive 状态 $\mathbf{q}_t$，模型预测一个 action chunk $\mathbf{A}_t = \{\mathbf{a}_1, \cdots, \mathbf{a}_H\}$，其中 $\mathbf{a}_i = \{\mathbf{q}_{\mathrm{d},i}, \mathbf{r}_{\mathrm{b},i\}$ 表示期望的全身关节角和基座 6D 旋转。

**PET 数据生成管道**：
- **Plan 阶段**：使用 A* 路径规划器生成碰撞感知平面参考路径，代价函数加入障碍物感知项 $c(\mathbf{x}) = c_{\text{step}} + \lambda \exp(-\Phi_{\text{2D}}(\mathbf{x})/d_0)$ 偏向安全区域；heading 调整模块检测窄通道并插入 90° 偏航变化诱导侧走；SONIC motion planner 产生自然步行姿态。
- **Edit 阶段**：应用 HumanoidPF 势场引导的软力编辑全身运动，包含：① Crouch controller 处理 squat 障碍物；② gait-adaptation 模块重定向脚部着陆位置越过地面障碍；③ 通过 SoftMimic-style pseudo-forces 将采样的排斥力应用到关键身体 link（shoulders, elbows, torso 等）。
- **Track 阶段**：使用 SONIC tracker 验证物理可行性，但训练监督信号来自 edit 阶段的碰撞自由参考运动而非 tracked 轨迹，以保持类人质量。

**TANGO 架构**（三系统分层）：
- **System-2 (VL 感知)**：Qwen2.5VL-7B，从 InternVLA-N1 权重 warmstart；使用 Budget-Aware Token Sampling (BATS) 管理长视频历史，采样概率 $P(t) = (1-\epsilon)e^{k(t-T)/T} + \epsilon$；spatial-grid pooling 分配更细网格给近期观测。
- **System-1 (Action Expert)**：基于 flow-matching 的 MM-DiT；预测 stabilized training target $\tilde{\mathbf{a}}_i = \{\mathbf{q}_{\mathrm{d},i}, \tilde{\mathbf{r}}_{\mathrm{b},i}, \Delta x_i, \Delta y_i, \Delta \psi_i\}$，其中 base yaw 相对于 chunk 首帧参数化，辅助 delta 显式编码 chunk 级平面位移；使用 RTC  conditioning 随机化的已 commit prefix 以 inpaint 剩余 horizon。
- **System-0 (Tracker)**：预训练的 SONIC tracker 以 ~200Hz 执行高频关节命令。

**联合训练目标**：$\mathcal{L} = \mathcal{L}_{\text{CE}} + w_{\text{FM}} \cdot \mathcal{L}_{\text{FM}}$，其中 $\mathcal{L}_{\text{CE}}$ 为 VideoQA 交叉熵损失，$\mathcal{L}_{\text{FM}}$ 为 flow-matching 损失，$w_{\text{FM}}=20$。全参数微调 1 个 epoch，学习率 $1\times10^{-5}$，使用 16×8 A100 GPUs（共 896 GPU-hours）。

**部署架构**：云端服务器端运行 VLA 模型（RTX PRO 6000，每 0.5s 推理），机载 Jetson Orin NX 运行 SONIC tracker（50Hz 重采样，~200Hz 控制闭环），端到端延迟约 20ms。

## 实验与结果
**数据集**：增强的 VLNVerse（205 scenes）和 SAGE-3D（373 scenes），共 64,633 条 trajectory，生成耗时 211 GPU-hours（RTX PRO 6000）。

**VLNVerse 基准（Table 1）**：
| 方法 | Seen SR↑ | Seen NE↓ | Seen SPL↑ | Unseen SR↑ | Unseen NE↓ |
|------|---------|---------|----------|-----------|-----------|
| InternVLA-N1 | 51.56 | 3.91 | 34.37 | 45.56 | 4.09 |
| **TANGO (Ours)** | **54.69** | **3.90** | **40.18** | **52.89** | **3.72** |

TANGO 在 Seen/Unseen 上均实现最高 SR 和最低 NE。

**杂乱环境穿越（Table 2，增强 Val Unseen）**：
| 方法 | SR↑ | SPL↑ | CR↓ |
|------|-----|------|-----|
| InternVLA-N1 + HumanoidPF | 41.88 | 29.49 | 15.81% |
| **TANGO (Ours)** | **43.75** | **31.83** | **9.90%** |

TANGO 碰撞率降低 37.4%（从 15.81%→9.90%），同时 SR 提升 1.87pp，SPL 提升 2.34pp，且仅使用 RGB 输入（HumanoidPF baseline 额外使用 LiDAR）。

**真实世界实验（Table 3，15 trials/method/setting）**：
| 方法 | Short-horizon SR↑/Coll.↓ | Long-horizon SR↑/Coll.↓ | Cluttered 3D SR↑/Coll.↓ |
|------|------------------------|------------------------|------------------------|
| InternVLA-N1 | 11/15 / 1.40 | 6/15 / 3.47 | 6/15 / 1.93 |
| **TANGO** | **12/15** / **0.40** | **8/15** / **1.07** | **10/15** / **0.73** |

**消融实验（Table 4-5）**：
- 移除 RTC：SR 从 43.75% 暴跌至 10.94%（-32.81pp），CR 从 9.90% 升至 14.60%。
- 移除 Motion Editing：SR 降至 36.25%，CR 升至 20.60%。
- Planar 变体 (Ours-2D) + 物理执行：SR 仅 26.67%，SPL 仅 8.34，远低于 TANGO 的 52.89%/40.18%，验证全身动作表示的价值。

## 相关工作脉络
1. **VLN 大模型**（Vint [14], LM-nav [15], NavGPT [22,23], InternVLA-N1 [10], Uni-NaVid [47]）：将导航建模为纯二维轨迹规划，忽略真实物理部署中的 embodiment gap；TANGO 通过全身动作预测弥合此差距。
2. **人形穿梭/跑酷**（Humanoid Parkour [21], HumanoidPF [12]）：聚焦短视野场景交互，HumanoidPF 虽实现高成功率但在长视野和复杂障碍物组合上难以扩展；TANGO 结合语义理解与物理感知实现长视野导航。
3. **全身 VLA 系统**（GR00T-N1.6 [18], Ψ₀ [17], WholeBodyVLA [19], LeVERB [34], HumanoidVLA [20], PhysiFlow [35]）：多数采用解耦设计（上层运动+底层追踪器），限制全身协调；TANGO 端到端预测可执行全身动作，复用预训练通用追踪器避免控制器共训练。
4. **碰撞感知 VLN**（Span-nav [32], MM-Nnav [13]）：受限于 2D 问题建模，主要考虑绕行行为；TANGO 学习丰富的人形全身运动以适应复杂障碍物。
5. **训练时 Action Conditioning**（RTC [29]）：TANGO 借鉴此技术对齐离线训练与在线流式执行，通过随机化已 commit prefix 实现 inpainting 剩余 horizon。

## 局限性与未来方向
- **低层追踪器能力瓶颈**：当前 SONIC tracker 限制了更复杂环境的部署，如上下楼梯（walking up stairs）仍不可行。
- **纯 RGB 输入的感知局限**：仅依赖 RGB 图像可能限制模型在视觉歧义或低光环境中的场景理解，深度相机和 LiDAR 有望提供帮助。
- **真实世界数据缺失**：目前仅在仿真中训练，真实世界的 sim-to-real gap 仍有挑战。
- **未来方向**：扩展到更通用的 locomanipulation foundation model，探索多传感器融合（RGB-D, LiDAR），以及支持更多样化的人形任务（如上下楼梯、跨越沟壑）。

## 研究启发与可借鉴点
1. **PET 数据合成管道的可迁移性**：Plan-Edit-Track 三步法（路径规划→运动编辑→可行性过滤）可推广到其他 embodied 任务的仿真数据生成，特别是需要碰撞感知全身运动的场景。
2. **Flow-matching action expert 的设计**：MM-DiT 配合 stabilized target（base yaw 相对首帧 + 辅助 delta）和 RTC 的训练技巧，可有效解决 VLA 系统中动作预测的连续性和时序对齐问题。
3. **Loss 加权策略**：运动条件 per-dimension loss weight（如 squat 对应 hip/spine/knee，stride 对应 spine/shoulder）可指导模型关注行为相关的身体部位，值得迁移到其它全身控制任务。
4. **端到端 vs 模块化权衡**：消融证明全身动作预测（29-DoF）显著优于平面变体（Ours-2D + 物理执行 SR 52.89% vs 26.67%），提示 embodied navigation 中显式建模全身几何的重要性。
5. **云端-边缘部署架构**：VLA 模型运行在服务器端（每 0.5s 推理），tracker 运行在机载端（50Hz 重采样），该设计平衡了计算密集型推理与实时控制的延迟需求，可借鉴于其他大型 VLA 系统的真实部署。

## 关键术语表
- **Vision-Language Navigation (VLN)**：给定自然语言指令和视觉观测，导航智能体在环境中到达目标位置的 task。
- **Vision-Language-Action (VLA) Model**：同时处理视觉、语言和动作模态的大型多模态模型，直接输出机器人可执行的底层动作。
- **Whole-Body Control (WBC)**：人形机器人中协调全身关节（包括手臂、躯干、腿部）以实现特定运动目标的低层控制策略。
- **Plan-Edit-Track (PET)**：TANGO 提出的自动化数据生成管道，依次执行路径规划、运动编辑和物理可行性跟踪三个阶段。
- **Real-Time Chunking (RTC)**：一种训练技术，通过 conditioning 模型于随机化的已 commit action prefix 来 inpaint 剩余 horizon，对齐离线训练与在线流式执行。
- **Collision Rate (CR)**：评估期间至少发生一次碰撞的 episode 百分比，用于量化杂乱环境中的导航安全性。
- **Flow Matching**：一种生成模型训练技术，通过学习向量场将噪声分布逐步映射到数据分布，TANGO 使用其作为动作 expert 的 backbone。
- **Budget-Aware Token Sampling (BATS)**：按时间概率分布 $P(t)=(1-\epsilon)e^{k(t-T)/T}+\epsilon$ 采样历史帧的视觉 token，以在有限记忆容量内管理长视频历史。

## 可复现要素
- **数据集**：增强的 VLNVerse 和 SAGE-3D 场景（论文声明将开源 data pipeline 和生成数据集）；训练数据 64,633 trajectories。
- **代码/权重**：论文声明将 open-source data pipeline, generated dataset, VLA framework, model checkpoint, and deployment system（website: tango-vla.github.io）。
- **关键超参**：学习率 $1\times10^{-5}$，训练 1 epoch，$w_{\text{FM}}=20$，action horizon $H=15$（30Hz 下 0.5s），VLA 推理间隔 0.5s，tracker 执行频率 50Hz（重采样后），控制闭环 ~200Hz。
- **硬件配置**：训练使用 16×8 NVIDIA A100 GPUs（共 896 GPU-hours）；部署服务器端 RTX PRO 6000，机载端 Jetson Orin NX；机器人 Unitree G1。
- **模型骨干**：Qwen2.5VL-7B（从 InternVLA-N1 权重 warmstart）；动作专家为 flow-based MM-DiT；底层追踪器为 SONIC。
