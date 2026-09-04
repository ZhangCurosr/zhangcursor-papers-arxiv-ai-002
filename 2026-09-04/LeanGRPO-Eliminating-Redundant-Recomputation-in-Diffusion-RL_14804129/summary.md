---
title: "LeanGRPO-Eliminating-Redundant-Recomputation-in-Diffusion-RL"
source: https://arxiv.org/pdf/2609.03528v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:03:27"
field: "生成模型高效训练"
keywords: ["Diffusion Reinforcement Learning", "Gradient Recomputation", "Systems Optimization", "Policy Gradient", "Data Parallelism", "Training Efficiency"]
innovations: ["提出共享Prompt并行布局以降低单卡显存压力", "设计Retain调度策略直接复用rollout计算图", "设计Reweight调度策略通过临时梯度与延迟校正避免重计算"]
benchmarks: ["HPSv2", "PickScore", "FLUX.1-dev", "Wan2.1", "SD3.5-Medium"]
---

# 论文速读：LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL

## 一句话总结
本文提出 LeanGRPO 框架，通过重构数据并行布局并引入两种无需重计算的训练调度策略（Retain 与 Reweight），消除了轨迹对数概率扩散强化学习（Diffusion RL）在更新阶段对已选时间步的冗余重计算，在保持原有优化目标不变的前提下实现了最高 1.83 倍的端到端加速。

## 研究问题与动机
1.  **核心问题**：在 on-policy 设置下（rollout 与 update 使用相同后端且策略参数不变），现有轨迹对数概率扩散 RL 方法（如 DanceGRPO、FlowGRPO）在 rollout 后，为重建可微分路径而在 update 阶段重新计算已选时间步，这在数学上是冗余的。
2.  **现有方案不足**：最直接的方案是在 rollout 阶段开启梯度追踪，但这样需要在获得终端优势（terminal advantage）前保留大量计算图与激活值，导致 GPU 显存开销随所选时间步数量线性增长，难以实用。
3.  **研究缺口**：尽管该冗余重计算问题普遍存在于主流开源实现中（论文审计了 32 个公开仓库），但现有算法改进工作（聚焦奖励分配、时间步选择、采样效率）和分布式系统优化工作均保留了这种两阶段执行范式，未能从执行框架层面消除它。

## 核心贡献（创新点）
1.  **识别并形式化冗余重计算问题**：明确指出在 on-policy 单步更新设定下，update 阶段的重新前向传播是数学冗余的，这是现有扩散 RL 训练流程中的一个主要系统瓶颈。
2.  **提出共享 Prompt 并行布局**：改变了传统将不同 prompt 分配给不同 GPU 并本地生成多个样本的数据并行方式，改为所有 GPU 处理同一 prompt 但独立生成不同样本，从而将显存压力（计算图/梯度/激活值）分散到多个 rank，避免单卡累积多个样本状态。
3.  **设计两种互补的重计算-free 训练调度**：
     *   **LeanGRPO-Retain**：直接保留 rollout 阶段的计算图和保存的激活值，待终端优势可用后直接复用进行反向传播，消除更新阶段重计算。
     *   **LeanGRPO-Reweight**：在 rollout 期间对每个选定时间步使用临时优势（值为 1）立即进行反向传播以消耗计算图、释放激活值，累积局部梯度后延迟梯度同步，待轨迹完成后用真实优势校正梯度，再执行分布式同步与更新。
4.  **广泛的实验验证**：将 LeanGRPO 集成到 DanceGRPO、FlowGRPO、FlowGRPO-Fast、MixGRPO-Flash 等多种算法及 FLUX.1-dev、SD3.5、Wan 等不同规模模型 backbone 上，证明了其加速效果和奖励等效性，并提供了调度策略选择指导。

## 方法详解
**核心思想**：在 rollout 阶段就启用梯度追踪，并通过新的数据布局和调度策略管理由此带来的显存/计算开销。

1.  **共享 Prompt 并行布局 (Shared-Prompt Parallel Layout)**：
    *   传统布局：$R$ 个 rank，每个 rank 负责 $B$ 个 prompt，每个 prompt 本地生成 $M$ 个样本。显存压力随 $M$ 和 $B$ 累积。
    *   LeanGRPO 布局：所有 $R$ 个 rank 处理**同一个** prompt，各自独立生成不同的样本（总共 $M$ 个）。若 $M \ge R$，则每个 rank 生成 $Q=M/R$ 个样本；若 $M < R$，则将 ranks 分组。每个 rank 只需保留 $Q$ 个（或 $M$ 个）样本的状态，显存压力降为原来的 $1/R$。所有 rank 完成生成后，all-gather 奖励并计算组内优势，然后立即开始训练，避免多 prompt group 的显存累积。该布局在梯度上是等价的（附录 A 证明）。

2.  **LeanGRPO-Retain (基于计算图保留)**：
    *   在 rollout 阶段，对所有选定时间步 $t \in \mathcal{S}$ 的前向传播开启梯度追踪，得到可微的对数概率 $\ell_{b,m,t}^{\text{grad-roll}}$ 及其对应的计算图和保存的激活值。
    *   整个轨迹采样完成后，收集奖励并计算终端优势 $A_{b,m}$。
    *   直接利用保留的计算图和优势 $A_{b,m}$ 进行反向传播，计算梯度：$G_{b,m}^{\text{retain}} = -A_{b,m} \sum_{t \in \mathcal{S}} \nabla_\theta \ell_{b,m,t}^{\text{grad-roll}}$。这与原始 Native 方法在精确算术下等价，且完全省去了 update 阶段的重新前向传播。
    *   **代价**：激活内存随选定时间步数量 $|\mathcal{S}|$ 近似线性增长。

3.  **LeanGRPO-Reweight (基于临时反向传播与延迟校正)**：
    *   在 rollout 阶段，对每个选定时间步 $t$ 同样开启梯度追踪，但**不保留**计算图。
    *   立即使用一个临时优势（$A=1$）对该时间步的反向传播：$g_{i,t}^{\text{prov}} = -\nabla_\theta \ell_{i,t}^{\text{grad-roll}}$。这一步消耗了计算图并释放了激活值。然后将 latent $z_{t-1}$ detach 以继续 rollout。
    *   对同一个轨迹 $i$ 的所有选定时间步，累积临时梯度：$G_i^{\text{prov}} = \sum_{t \in \mathcal{S}} g_{i,t}^{\text{prov}}$。
    *   整个轨迹采样完成、优势 $A_i$ 已知后，在本地进行校正：$G_i^{\text{corr}} = A_i G_i^{\text{prov}}$。
    *   **关键分布式细节**：不同样本（即使来自同一 prompt）的优势 $A_{i}$ 通常不同。如果在校正前就进行分布式同步（如 FSDP 的 reduce-scatter），会混合不同样本的临时梯度，导致无法再用一个简单的标量优势恢复正确的加权梯度和（附录 B、C 证明）。因此，**必须在本地完成优势校正后，再执行一次全局的 reduce-scatter** 进行梯度同步。
    *   **代价**：需要为每个样本保留一个完整的临时梯度，直到优势校正完成。内存开销与 $|\mathcal{S}|$ 无关。

## 实验与结果
*   **实验设置**：硬件为单节点 8x NVIDIA RTX A6000 (48GB)。模型包括 FLUX.1-dev (12B), SD3.5-Medium-2.5B, Wan2.1-1.3B, Wan2.1-14B, Wan2.2-TI2V-5B。算法包括 DanceGRPO, FlowGRPO, FlowGRPO-Fast, MixGRPO-Flash。评估指标为端到端每步训练时间、GPU 显存使用量、以及 HPSv2/PickScore 奖励收敛曲线。
*   **主要结果**：
    1.  **端到端加速**：LeanGRPO 在不同设置下均实现加速。**最高加速达 1.83×**（FlowGRPO + SD3.5-Medium + BF16 全参微调）。BF16 全参微调下 Reweight 加速 1.44×-1.81×，Retain 加速 1.18×-1.29×。BF16 LoRA 下 Reweight 1.26×-1.43×，Retain 1.27×-1.42×。加速效果随选定时间步比例增加而提升。
    2.  **时间分解**：以 FLUX.1-dev + DanceGRPO 为例，Native 方法重计算耗时 402.45 秒。Reweight 通过合并同步操作，将总训练时间从 1022.64 秒降至原始 Native 的约 1/1.81。
    3.  **消融实验**：共享 prompt 布局本身加速不明显；消除重计算是主要提速来源；合并同步（coalesced sync）在 full fine-tuning 下贡献额外加速（如从 1.29× 提升至 1.81×），在 LoRA 下贡献较小。
    4.  **训练收敛与奖励等效性**：LeanGRPO 两种变体在 HPSv2 和 PickScore 奖励上均保持了与 Native 相当的奖励提升趋势。在相同时间内，LeanGRPO 更快达到目标奖励（例如 Wan2.1-1.3B 上 Reweight 快 1.34×，FLUX.1-dev 上两者均快 1.46×）。控制实验显示，Retain 在固定轨迹下梯度与 Native 比特级一致，Reweight 的梯度余弦相似度高达 0.999802，相对 $\ell_2$ 误差仅 1.99%。
    5.  **GPU 显存**：Retain 的显存随 $|\mathcal{S}|$ 线性增长，在视频生成等大 tensor 场景下更易 OOM。Reweight 的显存稳定，增加一个完整梯度，更适合大分辨率/视频任务或 LoRA 训练。
*   **最强结果**：FlowGRPO + SD3.5-Medium 配置下，LeanGRPO-Reweight 获得 **1.83×** 的端到端加速。

## 相关工作脉络
1.  **DDPO (Black et al., 2024), DPOK (Fan et al., 2023)**：早期将 RL 应用于扩散模型对齐的工作，采用两阶段 rollout/recompute 范式，存在论文所指的冗余重计算问题。LeanGRPO 与其优化目标正交，可直接加速这类方法。
2.  **FlowGRPO (Liu et al., 2025b), DanceGRPO (Xue et al., 2025b)**：近期主流的基于轨迹对数概率的扩散 RL 方法，广泛使用并依赖 update 阶段的重计算。LeanGRPO 直接针对并优化此类方法的核心执行瓶颈。
3.  **VeRL-Omni (Huang et al., 2026), BiDiRL (Tan et al., 2026)**：分布式 RL 系统优化工作，侧重于资源放置、异步执行和数据移动。LeanGRPO 专注于消除单阶段内的冗余计算，两者可结合使用。
4.  **DiffusionNFT (Zheng et al., 2026a), V-GRPO (Tang et al., 2026)** 等：这些方法优化的是前向匹配、ELBO 或有限差分等目标，**不包含**论文所定义的“选定转换重计算”这一特定模式，因此不在 LeanGRPO 的优化范围内。
5.  **PPO/GRPO (Schulman et al., 2017; Shao et al., 2024)**：LeanGRPO 的梯度校正和 clipped loss 设计在数学上继承自 GRPO 目标，但将其应用于扩散模型的轨迹对数概率设定，并解决了由此产生的特定系统冗余问题。

## 局限性与未来方向
1.  **当前局限**：
    *   **调度策略需手动选择**：Retain 和 Reweight 适用于不同规模模型和输入尺寸，当前实现需用户根据硬件和工作负载手动选择。
    *   **适用范围限定**：主要针对 on-policy、单步更新（每次 rollout 后仅做一次参数更新）的场景。对于多 epoch 重用同一批 rollout 样本或涉及 off-policy 更新的变体，LeanGRPO 仅能在第一次更新时消除重计算，后续更新仍需重计算（或需要额外的算法保护机制，如 Flow-DPPO 中的约束）。
    *   **后端一致性要求**：要求 rollout 和训练更新使用相同的策略执行后端（graph-colocation）。若使用分离的后端（如 rollout 用 vLLM，训练用 FSDP/Diffusers），则无法直接复用计算图。
2.  **未来方向**：
    *   开发自动化的调度策略选择器，根据硬件配置和工作负载特征自动选择 Retain 或 Reweight。
    *   探索如何将 LeanGRPO 的思想扩展到非 on-policy 或多 epoch 更新的设置中。
    *   结合其他算法层面的优化（如动态 timestep 选择、稀疏奖励建模）。

## 研究启发与可借鉴点
1.  **审视“理所当然”的计算开销**：在复杂 ML 流水线中，仔细分析数据流和依赖关系，可能发现大量在特定条件下（如 on-policy）数学上冗余的计算。本文从基础算法逻辑出发识别冗余，而非依赖性能剖析工具，是一种有价值的研究思路。
2.  **系统设计中的权衡抽象**：提出了“保留计算图/激活值”vs “保留完整梯度”的二元权衡，并设计了两种对应的调度策略。这种将系统开销（显存 vs 计算）抽象为可选择的策略模块，并使其对上层算法透明的设计模式，可借鉴到其他需要管理计算图生命周期的场景（如长序列训练、在线学习）。
3.  **数据并行布局的重构**：共享 Prompt 并行布局通过改变样本到 rank 的映射关系，以通信（all-gather 奖励）为代价显著降低了单卡显存压力，同时保持了梯度等价性。这种“牺牲一点通信换取更大计算/显存效率”的思路，在分布式训练资源受限时很有参考价值。
4.  **延迟同步用于局部校正**：在 Reweight 中，为了支持不同样本拥有不同优势系数的梯度校正，**必须延迟跨设备的梯度同步**直到本地校正完成。这打破了一些分布式训练框架中“先聚合后处理”的常规直觉，揭示了在引入样本级加权时，同步时机需要谨慎设计。
5.  **与团队方向结合的机会**：本团队若关注扩散模型/流匹配模型的 RL 对齐训练，LeanGRPO 提供了一个即插即用的系统优化层。可以将 LeanGRPO 与我们可能正在研究的算法改进（如新的奖励函数、 timestep 调度策略）结合，在不损失算法收益的前提下获得显著的纯系统加速。

## 关键术语表
*   **Trajectory-Logprob Diffusion RL**：将扩散/流匹配模型的去噪过程视为一个多步策略轨迹，并通过优化采样转换的对数概率与终端奖励的乘积来进行强化学习对齐的方法范式。
*   **Recompute (重计算)**：在扩散 RL 的 update 阶段，为了获得可微分的对数概率以计算策略梯度，而对 rollout 阶段已采样过的状态/时间步再次执行模型前向传播。
*   **Computation Graph Retention (计算图保留)**：LeanGRPO-Retain 策略的核心，在 rollout 阶段保留选定时间步的前向计算图及中间激活值，待优势可用后直接用于反向传播，避免重新执行前向计算。
*   **Provisional Gradient (临时梯度)**：LeanGRPO-Reweight 策略中，使用一个占位优势（如 1）对单个选定时间步立即执行反向传播所得到的梯度，它包含了策略梯度的方向信息但缺少正确的优势标量缩放。
*   **Advantage Correction (优势校正)**：LeanGRPO-Reweight 策略的关键步骤，在获取真实的终端优势后，在本地将累积的临时梯度乘以对应的优势值，以恢复正确的策略梯度。
*   **Shared-Prompt Parallel Layout (共享 Prompt 并行布局)**：一种数据并行策略变体，所有 GPU rank 处理同一个文本 prompt，并各自独立生成多个样本，以分散显存压力并促进及时的优势计算。
*   **Coalesced Synchronization (合并同步)**：LeanGRPO-Reweight 将多个小批量或逐时间步的分布式梯度同步操作（如 reduce-scatter）合并为一次在优势校正后的大规模同步，以减少通信开销。
*   **Graph-Colocation (图共存)**：指 rollout 和 update 阶段的梯度计算发生在同一个进程/后端中，能够共享 autograd 状态。这是 LeanGRPO 能够直接复用 rollout 计算图的必要系统条件。

## 可复现要素
*   **数据集**：论文使用了 HPSv2 (Wu et al., 2023) 和 PickScore (Kirstain et al., 2023) 作为奖励模型和评估基准。训练 prompt 数据集未明确提及，通常此类工作使用公开文本到图像/视频数据集（如 LAION 子集或内部数据集）进行采样。**论文未明确声明**。
*   **代码**：**已开源**。GitHub 仓库：`github.com/coderwayne3025/LeanGRPO`
*   **权重**：实验使用了开源模型 FLUX.1-dev, SD3.5-Medium, Wan2.1, Wan2.2，其预训练权重可从官方渠道获取。**论文未提供自定义训练的模型权重**。
*   **关键超参**：硬件为 8x NVIDIA RTX A6000 (48GB)。优化器为 AdamW。参数精度测试了 BF16 全参微调、BF16 LoRA、FP32 LoRA。使用了 FSDP2 进行模型分片。梯度累积和具体超参数（如学习率、clip 范围 ε、LoRA rank/alpha）详见论文第 5.1 节及附录 E 的实验协议描述。
