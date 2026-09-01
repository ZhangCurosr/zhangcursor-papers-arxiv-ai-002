---
title: "Successive-Capacity-Growth-Task-Complexity-Driven-Width-and"
source: https://arxiv.org/pdf/2608.27367v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:01:37"
field: "世界模型与表征学习"
keywords: ["JEPA", "world models", "adaptive architecture", "function-preserving expansion", "SIGReg", "Vision Transformer", "successive learning"]
innovations: ["提出SCG方法，从最小ViT编码器起步通过任务驱动测试-验证-回滚机制逐步扩展宽度或深度", "实现比特级函数保持扩展（fp_ratio=1.0, abs_diff=0.0），新容量立即可用", "揭示过度配置惩罚现象，SCG以11%参数量超越固定大模型"]
benchmarks: ["Two-Room (2D)", "Push-T (5D)", "30-Object Dynamics (60D)"]
---

# 论文速读：Successive-Capacity-Growth-Task-Complexity-Driven-Width-and-Depth-Expansion-for-Vision-Transformer-Encoders-in-JEPA-World-Models

## 一句话总结
提出 Successive Capacity Growth (SCG)，让 JEPA 世界模型的 ViT 编码器从最小结构（283K 参数）出发，通过任务驱动的"测试-验证-回滚"机制逐步扩展宽度或深度，以比特级精度保持函数不变，在合成环境中达到接近固定大模型的性能，同时实现高达 56 倍更高的参数效率。

## 研究问题与动机
1. **固定容量编码器的容量错配**：现有 JEPA 世界模型（如 LeWM）使用预分配的固定大小 ViT-Tiny 编码器（~5M 参数），对简单任务过度配置（造成冗余计算），对复杂任务又容量不足。
2. **严重冗余的注意力头**：LeWM 的 3 个注意力头 pairwise cosine similarity ≈ 0.0001，本质上是同一处理路径的三份复制；层 3–12 的 residual ratios 从 0.9 递减至 0.15，大部分深度被浪费。
3. **缺乏任务驱动的自适应触发机制**：Net2Net、bert2BERT 等工作虽提出函数 preserving expansion，但依赖预定义增长计划，缺少基于预测容量的在线触发决策。
4. **JEPA 训练动态对架构修改的安全性**：JEPA 无 stop-gradient、无 EMA，编码器在训练中可被修改而不破坏预测器，但如何在修改的同时保证语义维度不坍缩且保持独立，尚需 SIGReg 等正则化的协同保证。

## 核心贡献（创新点）
1. **SCG 方法**：从 1 head / 2 layer / 283K 参数的最小编码器起步，通过任务无关的"测试-验证-回滚"机制按需扩展宽度（低阶语义维度）或深度（高阶语义抽象），与预分配最大容量的范式形成根本对比。
2. **比特级函数保持证明**：宽度和深度扩展均实现 fp_ratio = 1.0、abs_diff = 0.0，新容量在扩展后立即可用，训练损失继续下降，零假阳性扩展。
3. **SIGReg 驱动的语义正交性保障**：扩展新增维度由 Sketched Isotropic Gaussian Regularizer 保证统计独立性，防止多任务表征坍缩到同一子空间。
4. **容量瓶颈分解的实证验证**：2D 导航任务触发宽度扩展（表征瓶颈），60D 多物体动力学任务触发深度扩展（计算瓶颈），5D 任务正确收敛不扩展，三者形成对照。
5. **揭示"过度配置惩罚"现象**：在 Two-Room 任务上，SCG 以仅 11% 参数量超越 Fixed Large（0.000176 vs 0.000362），说明过度配置会引入冗余参数捕捉噪声，反而损害预测性能。

## 方法详解

**整体框架（Algorithm 1）**：每个 effective epoch 结束时检测预测损失 $\mathcal{L}_{\text{pred}}$ 是否 plateau（最近 2 个 epoch 的中位数相对前 2 个 epoch 中位数改善 ≤ 2%），若 plateau 则依次尝试宽度扩展和深度扩展，各训练 2 epochs 后判断是否改善 > 2%；保留有效扩展，失败则回滚并保存完整优化器状态（含 Adam 动量），最终收敛则停止所有扩展。

**函数保持宽度扩展**：
- Patch embedding / CLS token / positional embeddings：拼接已有前 64 维度的副本
- QKV 投影：新头的 QKV = head 0 的 QKV 副本
- **Output projection：新头的行初始化为零**（扩展后新头贡献为零，精确保持函数）
- MLP：新 input/output 维度复制前 64 维
- Projector：输入权重补零，新维度被忽略
- 扩展后 $d_{\text{model}}$ 增加 64

**函数保持深度扩展**：
- 新增 Transformer block 的 Output projection：weights = 0, bias = 0 → attention output = 0
- MLP W2：weights = 0, bias = 0 → MLP output = 0
- LayerNorm：weight = 1, bias = 0
- 新 block 成为恒等映射 $g(y)=y$，插入计算图后 $f \circ g \circ h = f \circ h$，函数精确保持

**损失函数**：
$$\mathcal{L}_{\text{total}} = \frac{1}{3}\sum_{i=0}^{2}\text{MSE}(\hat{z}_{i+1}, z_{i+1}) + \lambda \cdot \mathcal{L}_{\text{SIGReg}}(Z)$$
其中 $\lambda = 0.1$，目标侧**不使用** stop-gradient，梯度双向流动，SIGReg 负责防坍缩。

**触发机制关键设计**：
- 仅用 $\mathcal{L}_{\text{pred}}$ 而非 $\mathcal{L}_{\text{total}}$ 检测 plateau（消融证明 total loss 受 SIGReg 波动干扰会产生假阳性）
- Cooldown = 2 epochs 确保优化器在新参数上适应后再检测下一轮 plateau
- Test epochs 不计入 effective epoch 预算，保证自适应与固定配置的公平比较

**理论保证**：
- 容量瓶颈分解：宽度扩展解决表征瓶颈（$d_{\text{model}}$ 不足），深度扩展解决计算瓶颈（层次抽象能力不足）
- 优化轨迹安全：回滚恢复完整参数空间与优化器状态，轨迹严格连续

## 实验与结果

**数据集/环境**：三个合成环境，逐步递增复杂度——
- Two-Room（2D 状态）：2D 网格导航，5 离散动作
- Push-T（5D 状态）：T 形方块推动，2D 连续动作
- 30-Object Dynamics（60D 状态）：30 个对象动力学，2D 全局力

每环境 200 episodes × 200 steps，30% 数据，帧跳 5，4-frame 子轨迹。

**模型配置**：
- Fixed Small：1 head / 2 layer / $d_{\text{model}}=64$ / 283K 参数
- Fixed Large：3 heads / 12 layers / $d_{\text{model}}=192$ / 5.7M 参数
- SCG：从 Fixed Small 起步自适应增长

**主要结果**：

| 任务 | SCG 相对 Fixed Small 提升 | SCG 参数量 | Fixed Large 参数量 |
|---|---|---|---|
| Two-Room (2D) | **+49.0%** | 283K→647K | 5.7M |
| Push-T (5D) | +6.4% | 283K（不扩展） | 5.7M |
| 30-Object (60D) | **+20.3%** | 333K | 5.7M |

**关键数字**：
- SCG 在 30-Object 上用 50K 额外参数获得 0.0017 的损失降低，而 Fixed Large 需 5.4M 额外参数获 0.0033 降低 → **56× 参数效率**
- Two-Room 任务 SCG seed 42 以仅 11% 参数量超越 Fixed Large（0.000176 vs 0.000362）
- 函数保持：fp_ratio = 1.0，abs_diff = 0.0，CLS diff = 0.0，Block diffs = [0.0, 0.0]，零假阳性扩展（0/9 runs）
- 新头 cosine similarity 从 1.0 下降至 0.91–0.99，新层 residual ratio 从 0.0 升至 0.22

## 相关工作脉络

1. **LeWM (Maes et al., 2026)**：JEPA 世界模型的基线框架，使用固定 ViT-Tiny 编码器 + SIGReg；本文在其上引入自适应编码器，保持训练目标不变。
2. **Net2Net (Chen et al., 2016)**：提出通过权重复制实现宽度扩展、恒等初始化实现深度扩展的理论基础；本文将其适配到 ViT 编码器，并加入零初始化 output projection 的创新。
3. **bert2BERT (Chen et al., 2021)**：通过函数保持变换扩展 BERT，但依赖预定义增长计划；本文的触发机制是任务驱动的在线测试-验证，无需预定义计划。
4. **LiGO (Wang et al., 2023)**：通过学习线性映射将小模型扩展为大模型，但需要单独的 growth phase；本文在正常训练中生长，无额外阶段。
5. **GESMundo & Maile (2023)**：提出 6 种可组合的函数保持变换，但缺少应用时机策略；本文填补了这一空白——用预测损失 plateau 作为触发信号。
6. **Recursive ViT (Zhang et al., 2026)**：根据图像内容动态调整深度和宽度以实现资源效率；本文通过函数保持扩展而非调整已有架构，且触发信号来自在线预测损失而非输入内容。

## 局限性与未来方向

**局限性**：
1. 所有实验均在合成环境上进行，尚未在真实数据（更高视觉复杂度）上验证扩展机制。
2. 未评估下游规划任务成功率（Cross-Entropy Method planning），仅评估预测损失。
3. 仅在 ≤5.7M 参数规模验证，扩展至 ViT-Base（86M）未知。
4. 30-Object 任务 seed 42 在 50 epoch 内未 plateau，无法触发扩展；需更长训练或更低阈值。

**未来方向**（作者提出）：
1. 在真实数据上验证
2. 集成完整 LeWM 规划流水线评估下游任务
3. 引入剪枝机制在任务简化时移除多余 head/layer
4. 结合动态深度宽度调整机制进一步节省推理资源
5. 扩展至更大架构（ViT-Base），冗余节省更显著

## 研究启发与可借鉴点

1. **测试-验证-回滚的架构搜索范式**：无需强化学习或 NAS 搜索，通过简单的损失 plateau 检测 + 函数保持试验实现自适应架构增长，可迁移至其他 JEPA 或 transformer 编码器的容量自适应场景。
2. **零初始化 output projection 的技巧**：新 attention head 的 output projection 初始化为零，使得新头"沉默"但 QKV 已具备有意义初始化，这一设计比纯恒等初始化更能保证新参数快速学习，值得在 width expansion 任务中借鉴。
3. **过配置惩罚的发现**：对于简单任务，"更大模型不一定更好"——冗余参数会捕捉噪声并增加优化难度；本团队在资源受限场景下可考虑从最小容量起步而非预设大模型。
4. **SIGReg 与函数保持扩展的协同**：SIGReg 确保新维度统计独立，是函数保持扩展得以成功的关键组件；在自建 JEPA 系统时，若引入容量增长，需搭配同等性质的正则化机制防止表征坍缩。
5. **Test epoch 不计入有效训练预算的设计**：保证了自适应模型与固定模型在相同有效训练量下的公平对比，这一实验设计严谨性值得在类似的动态架构工作中复现。

## 关键术语表

**JEPA (Joint-Embedding Predictive Architecture)**：LeCun 提出的世界模型架构，通过联合嵌入预测未来表征，无需生成像素级观测，具有稳定的训练动态。

**SCG (Successive Capacity Growth)**：本文提出的方法，让编码器从最小容量起步，根据任务需求逐步扩展宽度或深度。

**SIGReg (Sketched Isotropic Gaussian Regularizer)**：基于 Epps-Pulley 正态性检验的特征函数匹配正则化器，确保嵌入维度统计独立、防止表征坍缩。

**函数保持扩展 (Function-Preserving Expansion)**：扩展后模型输出不变的操作，分为宽度扩展（权重复制 + 零输出投影）和深度扩展（恒等初始化 block）。

**Capacity Bottleneck Decomposition**：将预测损失 plateau 归因于两类瓶颈——表征瓶颈（维度不足，需宽度扩展）和计算瓶颈（层次不足，需深度扩展）。

**Effective Epoch**：实际计入模型训练进度的 epoch 数，test epoch（用于扩展试验）不计入，确保与固定配置公平比较。

**Over-provisioning Penalty**：模型容量远超任务需求时，冗余参数捕捉噪声、优化器难以在高维冗余损失面上有效导航，导致性能反而下降的现象。

## 可复现要素

- **数据集**：合成环境（Two-Room / Push-T / 30-Object Dynamics），非公开基准，代码仓库提供数据生成脚本
- **代码/权重**：代码已开源，地址 https://github.com/121-labs/ViT-Expansion-in-JEPA-WM
- **关键超参**：lr=3e-4（500-step warmup）、batch_size=128、weight_decay=0.05、gradient_clipping max_norm=1.0、50 effective epochs、plateau threshold=2%、cooldown=2 epochs、test window=2 epochs、SIGReg λ=0.1 / sketch_dim=64 / num_knots=17、patch_size=14、image_size=224×224、d_head=64、MLP ratio=4、projector dim=192
- **硬件**：NVIDIA RTX A6000 (48GB VRAM)，PyTorch 2.4.1，CUDA 12.4
- **随机种子**：3072, 42, 123；数据生成种子=42
