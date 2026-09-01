---
title: "Successive-Capacity-Growth-Task-Complexity-Driven-Width-and"
source: https://arxiv.org/pdf/2608.27367v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:01:44"
field: "表征学习与世界模型"
keywords: ["JEPA", "world models", "adaptive architecture", "function-preserving expansion", "SIGReg", "ViT", "successive capacity growth"]
innovations: ["从283K参数最小ViT出发，通过无任务依赖的测试-验证-回滚机制动态增长宽度或深度", "提出容量瓶颈分解理论，将预测损失平台归因于表征瓶颈或计算瓶颈并自动选择扩展类型", "揭示过配惩罚现象：过度分配容量会损害泛化性能"]
benchmarks: ["Two-Room 2D navigation", "Push-T 5D block pushing", "30-Object Dynamics 60D multi-object tracking"]
---

# 论文速读：Successive-Capacity-Growth-Task-Complexity-Driven-Width-and-Depth-Expansion-for-Vision-Transformer-Encoders-in-JEPA-World-Models

## 一句话总结
论文提出**Successive Capacity Growth (SCG)**，一种JEPA世界模型编码器的动态架构增长方法：从最小ViT（283K参数）出发，通过无任务依赖的测试-验证-回滚机制，根据预测损失平台自动选择宽度或深度扩展，无需预设容量即可匹配任务复杂度。

## 研究问题与动机
- JEPA世界模型（如LeWM）普遍采用固定大小ViT编码器，简单任务严重过配（浪费算力），复杂任务可能欠配（表现不足）。
- 作者分析发现LeWM的ViT-Tiny存在**结构性冗余**：3个attention头两两余弦相似度≈0.0001（近乎完全重复），残差分析显示layer 3-12贡献递减（残差比从0.9降至0.15）。
- 现有动态架构方法（如Net2Net、bert2BERT、LiGO）要么依赖预定义增长调度，要么需要单独的"生长阶段"训练，无法与JEPA在线训练无缝结合。
- 核心问题：如何让编码器**在训练过程中按需自动增长**，而非预先分配最大容量？

## 核心贡献（创新点）
- **提出SCG方法**：从283K参数最小ViT出发，宽度扩展（新增低层语义维度）和深度扩展（新增高阶语义抽象）均由任务驱动的测试-验证机制触发，无需任何超参数调优。
- **比特精确的函数保持**：宽度/深度扩展均实现fp_ratio=1.0、abs_diff=0.0，新容量立即可用，Optimizer状态完整保留，失败时零成本回滚。
- **自然触发正确的扩展类型**：2D任务触发宽度扩展（49%损失改进），60D任务触发深度扩展（20.3%改进），5D任务正确收敛不扩展，证明机制能自动区分表征瓶颈与计算瓶颈。
- **揭示"过配惩罚"现象**：SCG在Two-Room任务上以11%参数量超越Fixed Large模型，证明过度分配容量不仅浪费算力，还可能因冗余参数捕获噪声而损害泛化。

## 方法详解
**SCG流程**：从1头2层（d_model=64，283K参数）开始，每 epoch 末监测预测损失 L_pred；若最近2 epoch中位数相比前2 epoch改善≤2%（检测平台），则触发测试-验证：
1. **尝试宽度扩展**：复制现有头的QKV，新头输出投影初始化为零（不影响输出），MLP/Projector相应扩展 → d_model+64 → 训练2 epoch → 若L_pred降低>2%则保留，否则回滚。
2. **尝试深度扩展**：新增transformer block初始化为恒等映射（LayerNorm weight=1/bias=0，输出投影和MLP W2均为零）→ 训练2 epoch → 同样按2%阈值判断。
3. **冷却机制**：成功扩展后cooldown=2 epoch，确保优化器适应新参数后再检测下一平台。
4. **终止条件**：两种扩展均失败则标记converged，不再尝试。

**关键设计细节**：
- **使用L_pred而非总损失**：SIGReg项独立波动，用总损失会导致误触发（消融实验证实：用L_pred消除全部0假阳性）。
- **测试epoch不计入有效epoch**：所有模型获得相同50 effective epochs，保证公平比较。
- **SIGReg的作用**：通过复现特征函数匹配的Epps-Pulley正态性检验，确保新增语义维度与已有维度统计独立，防止坍缩。

## 实验与结果
**环境**：Three合成环境（Two-Room 2D、Push-T 5D、30-Object Dynamics 60D），每任务3 seeds（3072/42/123），50 effective epochs。

**主要结果**：
- **Two-Room (2D)**：SCG宽度扩展（seed 42：64→128 dim）达到L_pred=0.000176，**超越Fixed Small 49%**，甚至击败Fixed Large（0.000362），仅用11%参数。
- **Push-T (5D)**：SCG正确收敛不扩展，L_pred=0.003534 vs Fixed Small 0.003775（6.4%改进），证明机制不会无谓增长。
- **30-Object (60D)**：SCG深度扩展（2→3 layers）达到L_pred=0.006666，**超越Fixed Small 20.3%**，接近Fixed Large的0.005080。
- **参数效率**：SCG在30-Object任务上用50K额外参数换取0.0017损失下降，而Fixed Large需5.4M额外参数换0.0033下降 → **56×更高参数效率**。
- **函数保持验证**：6次验证实验（2任务×3 seeds），宽/深度扩展fp_ratio均为1.0，abs_diff=0.0，CLS diff=0.0，Block diffs=[0.0, 0.0]。
- **消融**：用总损失做触发会导致假阳性扩展；Effective rank≈3无法区分2D/60D任务（网络总是压缩表征）。

## 相关工作脉络
- **LeWM / JEPA世界模型**：本文基于LeWM的稳定训练框架（无EMA、无stop-gradient），但首次引入自适应编码器；JEPA核心是仅用L_pred+SIGReg两个损失项从像素端到端训练。
- **Net2Net (Chen et al., 2016)**：理论奠基者，提出宽度扩展（权重复制）和深度扩展（恒等初始化）；本文创新在于将其应用于ViT编码器并设计zero-initialized输出投影。
- **bert2BERT (Chen et al., 2021)**：预训练语言模型的增长方法，但依赖预定义增长调度；SCG无预定义调度，由在线损失信号驱动。
- **LiGO (Wang et al., 2023)**：通过learned linear mappings学习最优增长模式，但需单独的growth phase训练；SCG在正常训练中无缝生长。
- **Composable Function-Preserving Transformations (Gesmundo & Maile, 2023)**：提出6种可组合变换但无触发策略；SCG填补了"何时扩展"的决策空白。
- **SIGReg (Balestriero & LeCun, 2025)**：JEPA的核心反坍缩正则化；本文证明其在架构增长时仍能保持新增维度的统计独立性。

## 局限性与未来方向
- 实验仅限合成环境，未验证于真实视觉数据的更高复杂度场景。
- 仅评估预测损失，未评估下游规划成功率（world model的最终目标）。
- 单GPU规模限制（最大5.7M参数），ViT-Base（86M）级扩展未测试。
- 30-Object任务中seed 42在50 epoch内未出现平台，可能需要更长训练或更低阈值。
- 未来方向：真实数据验证、完整LeWM规划pipeline集成、逆剪枝机制（任务简化时移除冗余头/层）、与动态推理扩展机制结合。

## 研究启发与可借鉴点
- **测试-验证-回滚范式**：可用于其他需要动态调整架构的场景（如NAS、持续学习），零风险探索新容量。
- **容量瓶颈分解理论**：将预测损失平台归因于"表征瓶颈"（需更宽）vs"计算瓶颈"（需更深），为架构诊断提供明确方向。
- **SIGReg在增长过程中的作用**：证明复现正则化可与动态架构生长兼容，可推广至其他自监督表征学习框架。
- **过配惩罚的发现**：挑战"越大越好"的惯例，为小样本/低资源场景下的紧凑模型设计提供实证支撑。
- **与本团队结合点**：若本团队做世界模型/表征学习，可直接借鉴SCG的扩展策略；若做NAS，可借鉴其"无预定义调度、在线触发"的思路。

## 关键术语表
- **JEPA (Joint-Embedding Predictive Architecture)**：仅用预测损失+正则化项从原始像素端到端训练的自监督框架，核心思想是预测嵌入空间中的慢变特征而非直接预测像素。
- **SIGReg (Sketched Isotropic Gaussian Regularizer)**：通过复现特征函数匹配进行Epps-Pulley正态性检验的反坍缩正则化器，确保各语义维度独立且对齐预测目标。
- **函数保持扩展 (Function-Preserving Expansion)**：扩展后网络输出严格不变的架构修改操作（新头/新层初始化为零或恒等映射），保证回滚零成本。
- **容量瓶颈分解**：预测损失平台源于"表征瓶颈"（d_model不足以容纳独立因素，需宽度扩展）或"计算瓶颈"（浅层不足以建模高阶交互，需深度扩展）。
- **测试与验证机制 (Test-and-Verify)**：触发扩展的无任务依赖决策逻辑，基于L_pred的2%阈值判断平台/改进，失败则回滚。
- **有效epoch (Effective Epoch)**：不计入测试/回滚epoch的训练轮次，所有配置获得相同数量以保证公平比较。

## 可复现要素
- **代码**：已开源 https://github.com/121-labs/ViT-Expansion-in-JEPA-WM
- **数据集**：三个合成环境（Two-Room、Push-T、30-Object Dynamics），由论文生成脚本产生，非公开标准数据集
- **关键超参**：AdamW lr=3e-4，warmup 500 steps，weight_decay=0.05，batch_size=128，gradient_clipping max_norm=1.0，SIGReg λ=0.1，sketch_dim=64，num_knots=17，plateau_threshold=2%，test_window=2 epochs，cooldown=2 epochs，effective_epochs=50，max_heads=12，max_layers=12，max_params=15M
- **随机种子**：3072, 42, 123；数据生成seed=42
- **硬件**：NVIDIA RTX A6000 (48GB VRAM)，PyTorch 2.4.1，CUDA 12.4
