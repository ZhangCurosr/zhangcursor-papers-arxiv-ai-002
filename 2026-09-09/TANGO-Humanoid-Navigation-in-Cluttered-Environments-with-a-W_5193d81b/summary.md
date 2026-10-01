---
title: "TANGO-Humanoid-Navigation-in-Cluttered-Environments-with-a-W"
source: https://arxiv.org/pdf/2609.09158v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:31:57"
field: "人形机器人全身视觉-语言导航"
keywords: ["Vision-Language Navigation", "Whole-Body Control", "Humanoid Robot", "Vision-Language-Action Model", "Cluttered Environment", "Simulation-to-Real Transfer", "Flow-Matching"]
innovations: ["首个全身 VLA 导航框架，直接预测 29-DoF 关节空间动作实现语言条件全身导航", "PET 管道自动化合成碰撞-free 全身导航数据（64,633 条轨迹，211 GPU-hours）", "Triple-system 架构（VL 骨干 + flow-matching 动作专家 + SONIC tracker）配合 RTC 实现零样本 sim-to-real"]
benchmarks: ["VLNVerse", "Augmented VLNVerse", "Real-world Unitree G1"]
---

# 论文速读：TANGO-Humanoid-Navigation-in-Cluttered-Environments-with-a-W

## 一句话总结
TANGO 是首个面向杂乱室内环境的全身视觉-语言导航（Whole-Body VLN）框架，输入自然语言指令与自视角 RGB 图像，直接预测 29-DoF 关节空间动作，结合 PET（Plan-Edit-Track）仿真数据生成管道，在仿真和真实 Unitree G1 上均实现零样本迁移的健壮碰撞避免导航。

## 研究问题与动机
- **机器人身体构型是导航问题的内在部分**：与传统移动底盘将导航建模为 2D 路径规划不同，人形机器人在杂乱环境中的可通行性取决于手臂放置、躯干调整、步态调制等全身几何自适应，平面动作空间无法显式表达导航决策与全身可行性的耦合关系。
- **现有 VLN 方法忽略物理 embodiment gap**：主流 VLN 仅输出 2D 航点或离散动作，部署到真实机器人时因缺乏低层物理执行模块而失效；端到端全身控制模型（如 GR00T、Ψ₀）将导航委托给高层 locomotion command，无法在导航时显式推理全身可通行性。
- **已有碰撞感知 traversing 方法难以扩展到长视距语言导航**：HumanoidPF 等方法依赖任务先验或特定训练分布，可扩展性受限，无法联合处理语言目标与全身碰撞规避。
- **大规模高质量全身导航数据集稀缺**：人体动作捕捉+重定向 pipeline 存在运动退化和 embodiment 失配；需要可扩展的自动仿真合成方案。

## 核心贡献（创新点）
1. **首个全身 VLA 导航框架**：TANGO 直接预测 29-DoF 关节空间动作，消除了传统 VLN 中导航与控制的模块分离，使语言条件导航与全身几何自适应端到端联合训练。
2. **Plan-Edit-Track（PET）自动数据生成管道**：通过 A* 带障碍物感知代价的路径规划 → 基于势场的全身运动编辑（跨步/侧移/蹲下三类行为）→ SONIC 跟踪器可行性过滤，以 211 GPU-hours 低成本合成 64,633 条物理可执行轨迹。
3. **三重系统架构（System-2/1/0）**：VL 感知骨干（Qwen2.5VL-7B + InternVLA-N1 预热）+ 基于 flow-matching 的 MM-DiT 动作专家（配合训练时实时 chunking RTC）+ 现成 SONIC 低层跟踪器，联合训练并支持零样本 sim-to-real 迁移。
4. **稳定化动作表征设计**：将 base pose 参数化为相对第一帧的 yaw 角，并显式编码 chunk 级平面位移/航向变化辅助量，显著提升回归稳定性；同时通过 BATS 时序 token 采样管理长历史视频输入。
5. **严格的物理可执行性保障**：PET 中编辑后的参考运动由 SONIC tracker 作为可行性过滤器，但**不使用跟踪结果作为监督信号**（避免退化类人结构），而是用碰撞-free 的规划/编辑阶段参考运动作为训练监督，保证数据质量与执行可行性并存。

## 方法详解
### 问题形式化
给定语言指令 ℓ、当前观测 $\mathbf{o}_t$（含前后向 RGB 序列 $\mathbf{I}_{1:t}^{\mathrm{fr,dn}}$ 和 proprioceptive 状态 $\mathbf{q}_t$），模型预测动作 chunk $\mathbf{A}_t = \{\mathbf{a}_1, \cdots, \mathbf{a}_H\}$，其中 $\mathbf{a}_i = \{\mathbf{q}_{\mathrm{d},i}, \mathbf{r}_{\mathrm{b},i}\}$，$\mathbf{q}_{\mathrm{d},i} \in \mathbb{R}^{29}$ 为期望全身关节角，$\mathbf{r}_{\mathrm{b},i} \in \mathbb{R}^6$ 为 base 6D 旋转表示，随后流式传输到低层 motion tracker。

### PET 数据生成管道
- **Plan 阶段**：在增强场景上用 A* 规划带障碍物感知代价的 2D 参考路径，代价函数 $c(\mathbf{x}) = c_{\mathrm{step}} + \lambda \exp(-\Phi_{\mathrm{2D}}(\mathbf{x})/d_0)$，偏向安全区域；检测窄通道并插入 90° 转向指令侧行走；将路径转换为 SONIC 运动规划器的速度命令，生成自然步行参考运动。
- **Edit 阶段**：对参考运动施加全身障碍物规避编辑。构建 3D SDF $\Phi(\mathbf{p})$，限制在路径走廊 $w_{\mathrm{corr}} \approx 0.5\mathrm{m}$ 内计算；通过 SoftMimic 风格伪力将势场排斥力施加到 shoulders/elbows/wrists/torso 等关键连杆，转化为有界软位移目标；针对三类障碍物分别设计：squat 障碍物仅保留力的垂直分量引导下蹲，stride 障碍物通过 gait-adaptation module 重新定位 foot landing position 并调整 swing-foot clearance，sidle 障碍物通过水平分量推离躯干和手臂。
- **Track 阶段**：SONIC 跟踪器验证动力学可行性和碰撞自由，失败或碰撞轨迹被丢弃；训练监督使用编辑阶段的参考运动而非跟踪结果。

### TANGO 架构
- **System-2（VL 感知）**：基于 Qwen2.5VL-7B，以 InternVLA-N1 权重 warmstart；前后摄像头垂直堆叠为单帧，应用 BATS（Budget-Aware Token Sampling）处理长历史视频，采样概率 $P(t) = (1-\epsilon)e^{k(t-T)/T} + \epsilon$，经 Grid Pool 空间降维后送入 VLM 产出 latent context token $z$。
- **System-1（动作专家）**：MM-DiT 架构，基于 flow-matching 预测 horizon $H$ 的动作序列；训练时应用 RTC（Real-Time Chunking），对随机化的已 committed prefix 进行 inpainting；稳定化目标 $\tilde{\mathbf{a}}_i = \{\mathbf{q}_{d,i}, \tilde{\mathbf{r}}_{b,i}, \Delta x_i, \Delta y_i, \Delta \psi_i\}$ 辅助回归。
- **System-0（低层执行）**：现成 SONIC tracker，以约 200Hz 闭环控制。
- **联合训练目标**：$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + w_{\mathrm{FM}} \cdot \mathcal{L}_{\mathrm{FM}}$，其中 $w_{\mathrm{FM}}=20$，全参数微调 1 epoch，学习率 $1\times10^{-5}$，使用 16 节点 × 8×A100 GPU 约 7 小时（共 896 A100 GPU-hours）。

### 部署架构
- **仿真**：IsaacSim 数字孪生 + MuJoCo 物理模拟，通过前向运动学（FK）求解相机位姿获取观测。
- **真实**：server-client 云边架构，VLA 运行在 RTX PRO 6000 服务器（每 0.5s 推理一次，horizon=15@30Hz），SONIC tracker 运行在机载 Jetson Orin NX（约 200Hz）；前后向 RealSense D455/D435i RGB 输入，约 20ms 延迟，预测动作 chunk 重采样至 50Hz 后执行，实现零样本 sim-to-real。

## 实验与结果
### 数据集
- 原始 VLNVerse（263 场景）和 SAGE-3D（1,000 场景）经 Gemini 2.5 Flash 过滤后保留 205 + 373 = **578 个源场景**，增强后合成 **64,633 条轨迹**；障碍物类别占比：stride 44%、sidle 15%、squat 41%。
- VLNVerse 训练集 3,963 条轨迹用于微调；验证集 VLNVerse-seen 423 条、unseen 825 条。

### 基线与评估指标
- 基线包括 CMA、Seq2Seq、RDP、HNR、InternVLA-N1、Uni-NaVid（均在 teleportation 设置下评估）； cluttered 场景对比 InternVLA-N1 + Unitree WBC 和 InternVLA-N1 + HumanoidPF。
- 指标：SR（成功率）、SPL（路径长度加权成功率）、NE（导航误差）、OSR（Oracle 成功率）、CR（碰撞率）。

### 主要结果
- **VLNVerse 基准**（Table 1）：TANGO 在 VLNVerse-seen 上 SR=**54.69%**（最高），NE=**3.90**（最低）；unseen 上 SR=**52.89%**（最高），NE=**3.72**（最低），SPL 与最强基线 RDP 相当（40.18 vs 42.72）。
- **杂乱环境穿越**（Table 2）：TANGO 在增强 VLNVerse-unseen 上 SR=**43.75%**，SPL=**31.83**，CR=**9.90%**；相比最强基线 InternVLA-N1+HumanoidPF（SR=41.88%，CR=15.81%），SR 提升 **+1.87pp**，CR 从 15.81% 降至 **9.90%**（-5.91pp），且仅使用 RGB 输入（对手基线额外使用了 LiDAR）。
- **真实世界实验**（Table 3）：TANGO 在短视距 2D（12/15 vs 11/15）、长视距 2D（8/15 vs 6/15）、杂乱 3D（10/15 vs 6/15）三种设置下均取得最高 SR 和最低平均碰撞数（0.40/1.07/0.73 vs 1.40/3.47/1.93），实现零样本迁移。
- **消融实验**（Tables 4-5）：去除 RTC 导致 SR 从 43.75% 暴跌至 10.94%（**-32.81pp**）；去除 Motion Editing 导致 SR 降至 36.25%，CR 升至 20.60%；平面变体 Ours-2D + 物理执行 SR 仅 26.67%，验证了全身动作表征的核心价值。

## 相关工作脉络
1. **VLN 大模型**（InternVLA-N1、NaviD、Omnivla 等）：将 VLN 建模为纯平面轨迹规划，缺乏低层物理执行模块；TANGO 在其基础上引入全身动作空间和物理感知，弥补了 embodiment gap。
2. **碰撞感知人形穿越**（HumanoidPF）：基于 RL 实现室内无碰撞穿越，但依赖任务特定先验，难以扩展到长视距语言导航；TANGO 通过 PET 管道可规模化生成高质量数据，不依赖特定任务分布。
3. **人形全身控制大模型**（GR00T-N1.6、Ψ₀、WholeBodyVLA）：采用解耦设计（上层预测上半身运动 + 下发高层 locomotion command），限制了全身协调；TANGO 端到端预测完整 29-DoF 动作，不依赖控制器共训练。
4. **非解耦 VLA 方法**（LeVERB、HumanoidVLA、PhysiFlow）：学习潜在运动表示并由专用 controller 解码；TANGO 直接预测可执行动作，依赖预训练通用 tracker，避免 controller co-training。
5. **训练时实时 chunking（RTC）**（Black et al.）：TANGO 将此技术应用于全身 VLA，对齐离线训练与在线流式执行，是首次将此技巧用于人形全身导航场景。
6. **动作表征与 flow-matching VLA**（MM-DiT）：TANGO 将 flow-matching 动作专家集成到 VLN 框架，并设计稳定化训练目标（相对 base yaw + 辅助 delta），不同于传统离散/连续动作输出。

## 局限性与未来方向
- **低层 tracker 能力成为复杂环境的瓶颈**：论文明确指出当前 tracker 在处理如上下楼梯等更复杂场景时受限，限制了部署范围。
- **纯 RGB 输入的限制**：缺少深度相机和 LiDAR 在视觉模糊、低光照或杂乱场景中的补充感知能力，可能影响对复杂 3D 结构的理解。
- **未来方向**：将 TANGO 框架扩展到更通用的 loco-manipulation foundation model，支持需要协调运用整个人体的挑战性任务；探索多模态感知融合（RGB-D、LiDAR）；提升 tracker 以适应更多地形。

## 研究启发与可借鉴点
1. **PET 数据生成管道的"编辑+过滤"范式值得借鉴**：先通过规划生成基线运动，再用物理势场进行语义驱动的编辑（侧移/跨步/下蹲），最后用 tracker 做可行性过滤而非监督信号——这一"参考质量优先于执行拟合"的思路可迁移到其他需要高质量仿真数据的具身任务。
2. **稳定化动作表征设计**：将 base pose 参数化为相对第一帧的 yaw 角并显式编码 chunk 级位移辅助量，是提升 VLA 回归稳定性的有效技巧，可推广到其他全身控制任务。
3. **RTC + flow-matching 的组合策略**：训练时随机 committed prefix inpainting 与 flow-matching 动作生成结合，既保证连续性又支持实时 chunking 执行，为 VLA 的在线部署提供了可复用的训练-部署对齐方案。
4. **BATS 时序 token 采样 + Grid Pool 空间降维**：管理长视距历史观测的资源效率策略，可适用于任何需要长时间上下文的人形导航/操作任务。
5. **云边分离部署架构**：计算密集型 VLA 与高频 WBC 模块在 server-client 架构下解耦，通过标准 IP 网络连接，约 20ms 延迟下实现 0.5s 推理周期 + 200Hz 控制循环，为同类系统的工程部署提供了参考范式。

## 关键术语表
**TANGO（Traversability-Aware Vision-Language Navigation）**：面向杂乱环境的全身 VLA 导航框架，直接预测 29-DoF 关节空间动作。
**PET（Plan-Edit-Track）**：自动化仿真数据生成管道，包含 A* 路径规划、势场运动编辑和 SONIC 跟踪器可行性验证三个阶段。
**MM-DiT（Multi-Modal Diffusion Transformer）**：TANGO 的 System-1 动作专家，基于 flow-matching 生成动作 chunk。
**RTC（Real-Time Chunking）**：训练时将动作序列随机划分为已 commitment prefix 和待 inpainting suffix，对齐离线训练与在线流式执行。
**BATS（Budget-Aware Token Sampling）**：按指数衰减概率函数采样历史帧 token，在固定预算下管理长视频历史。
**CR（Collision Rate）**：评估 episode 级碰撞安全性的指标，定义为至少发生一次碰撞的 episode 占比。
**SONIC**：高保真人形全身运动跟踪器（Supersizing Motion Tracking for Natural Humanoid Control），作为 TANGO 的 System-0 低层执行模块。
**embodiment gap**：纯软件 VLN 策略与真实物理机器人执行之间的性能差距，源于物理约束和 sensorimotor 失配。

## 可复现要素
- **数据集**：64,633 条仿真轨迹，基于 VLNVerse 和 SAGE-3D 增强场景生成；论文声明将开源数据集和生成管道。
- **代码/权重**：论文声明将开源 data pipeline、生成数据集、VLA 框架、模型 checkpoint 和部署系统（网站：tango-vla.github.io）。
- **关键超参**：学习率 $1\times10^{-5}$，flow-matching 损失权重 $w_{\mathrm{FM}}=20$，训练 1 epoch，horizon $H=15$，动作频率 30Hz，tracker 频率约 200Hz，VLA 推理周期 0.5s。
- **训练硬件**：16 节点 × 8×NVIDIA A100 GPU（共 896 A100 GPU-hours）；数据生成使用 RTX PRO 6000（211 GPU-hours）。
- **真实机器人**：Unitree G1，RealSense D455（前向）+ D435i（向下）RGB 相机，Jetson Orin NX 机载 + RTX PRO 6000 服务器。
