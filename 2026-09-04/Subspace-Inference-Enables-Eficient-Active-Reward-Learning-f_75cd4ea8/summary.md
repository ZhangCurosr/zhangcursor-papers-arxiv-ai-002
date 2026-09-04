---
title: "Subspace-Inference-Enables-Eficient-Active-Reward-Learning-f"
source: https://arxiv.org/pdf/2609.04066v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:54:49"
field: "强化学习从人类反馈与主动奖励学习"
keywords: ["主动学习", "强化学习从人类反馈", "贝叶斯深度学习", "扩展卡尔曼滤波", "子空间推断", "奖励模型", "不确定性量化"]
innovations: ["首次将子空间滤波应用于神经网络奖励模型的主动偏好学习", "通过EKF在低维子空间中高效维护参数后验，支持任意数量采样计算InfoGain", "序列式单次查询更新避免重复训练历史数据，实现5-40倍加速"]
benchmarks: ["D4RL", "V-D4RL", "SOAR机器人数据集"]
---

# 论文速读：Subspace-Inference-Enables-Eficient-Active-Reward-Learning-f

## 一句话总结
论文提出 PreferenceEKF，一种将主动偏好学习建模为序列贝叶斯滤波问题的样本高效方法，通过在低维子空间中使用扩展卡尔曼滤波器（EKF）追踪神经网络奖励模型的不确定性，实现了比现有贝叶斯深度学习方法更优的样本效率、运行时间和校准性能。

## 研究问题与动机
- RLHF 中偏好反馈每次仅提供最多 1 比特信息，导致样本效率低下，需要主动学习策略选择最有价值的查询。
- 现有主动学习方法依赖不确定性量化，但贝叶斯方法难以扩展到大型神经网络参数空间，而集成方法和 Dropout 各有缺陷（计算成本高或后验近似质量差）。
- 神经网络参数空间维度极高，完整后验推断的计算复杂度和内存开销不可接受（协方差矩阵缩放为 $O(|\pmb{\theta}|^2)$）。
- 已有的低维奖励模型（如线性模型、高斯过程）虽能高效进行主动学习，但表达能力有限；神经网络奖励模型缺乏可扩展的贝叶斯推理框架。

## 核心贡献（创新点）
- **首次将子空间滤波应用于神经网络奖励模型训练**：将 EKF 与子空间推断结合，在低维子空间中维护参数后验分布，避免了全参数空间的计算瓶颈。
- **序列式后验更新机制**：PreferenceEKF 仅对最新查询-标签对进行贝叶斯更新，无需像基线方法那样重复训练所有历史数据，实现计算和内存效率的双重提升。
- **高效采样支持信息增益计算**：子空间后验的高斯结构允许任意数量模型采样，使得 InfoGain 等采样型采集函数可伸缩到神经网络奖励模型。
- **系统性实验验证**：在 D4RL 和 V-D4RL 基准上与 DeepEnsemble、Dropout、Laplace、LLMCMC 四种贝叶斯深度学习方法对比，在样本效率、运行时间、可扩展性和校准方面均取得优势。
- **扩展至真实机器人数据和像素输入**：证明了方法在稀疏偏好反馈（SOAR 机器人数据集）和基于图像的奖励模型（V-D4RL）中的适用性。

## 方法详解
- **问题建模**：将主动偏好学习形式化为序列贝叶斯滤波问题，参数后验 $b^i = p(\pmb{\theta} | \mathcal{D}_{1:i})$ 通过 Bayes 规则递归更新，动态模型设为恒等函数 $g(\pmb{\theta}) = \pmb{\theta}$，测量模型使用 Bradley-Terry  likelihood $p_\theta(y|\tau_a, \tau_b) = \sigma(r_\theta(\tau_a) - r_\theta(\tau_b))$。
- **EKF 更新**：对非线性 BT 似然进行一阶泰勒展开线性化，测量 Jacobian $\mathbf{H}_i = \sigma'(z)(\nabla_\theta r_\theta(\tau_a) - \nabla_\theta r_\theta(\tau_b))$，其中 $\sigma'$ 作为加权系数在模型不确定时最大化；假设加性高斯噪声，后验保持高斯形式 $\mathcal{N}(\mu_i, \Sigma_i)$。
- **子空间构造**：在初始 warm-up 数据集上运行 SGD 收集迭代轨迹 $\pmb{\theta}_{1:w}$，通过 SVD 提取前 $|z|=200$ 个主成分得到投影矩阵 $\mathbf{A} \in \mathbb{R}^{|\pmb{\theta}|\times|z|}$，或使用随机投影替代；初始参数偏移 $\pmb{\theta}_* = \pmb{\theta}_w$。
- **子空间推断**：在 $|z| \ll |\pmb{\theta}|$ 的子空间中维护后验 $b^i = \mathcal{N}(\pmb{\mu}_i', \pmb{\Sigma}_i')$，通过仿射映射 $\pmb{\theta}(z) = \mathbf{A}z + \pmb{\theta}_*$ 投影回全空间进行前向传播和预测分布计算。
- **主动学习循环**：基于子空间后验采样 $M$ 个模型，计算 InfoGain 采集函数 $Q_i^* = \arg\max_{Q_i} \frac{1}{M}\sum_{y_i}\sum_{\pmb{\theta}} P(y_i|Q_i,\pmb{\theta})\log_2\frac{M\cdot P(y_i|Q_i,\pmb{\theta})}{\sum_{\pmb{\theta}'} P(y_i|Q_i,\pmb{\theta}')} $，获取标注后更新后验。
- **关键超参**：子空间维度 $|z|=200$，查询预算 $B=60$，轨迹段长度 50，测量噪声协方差 $\mathbf{V}=0.07\cdot\mathbf{I}$，动力学噪声 $\mathbf{U}=0.0001\cdot\mathbf{I}$，先验噪声 $\mathbf{W}=0.07\cdot\mathbf{I}$，迭代 EKF 线性化次数 $n_{\text{linearize}}=5$。

## 实验与结果
- **数据集与基准**：D4RL（12 个 MuJoCo/Adroit/Maze2D 任务）和 V-D4RL（基于图像的视觉变体），使用合成噪声最优标签和真实机器人 SOAR 数据集。
- **评估指标**：测试集 BT log-likelihood（样本效率）、训练运行时间、Expected Calibration Error (ECE)、Brier score、离线 RL 策略 rollout return。
- **主要结果**：
  - **样本效率**：PreferenceEKF 在 12 个任务上聚合结果显示，active 和 random 变体均优于或持平于所有贝叶斯深度学习基线（DeepEnsemble、Dropout、Laplace、LLMCMC）。
  - **运行时间**：PreferenceEKF 训练速度比 DeepEnsemble 快约 5 倍，比 LLMCMC 快 40 倍以上（active: 12.1 vs 97.7 vs 780.2 分钟）。
  - **可扩展性**：随模型样本数 $M$ 和神经网络规模增加，PreferenceEKF 的运行时增长最平缓，同时保持高 log-likelihood。
  - **校准**：PreferenceEKF 具有最低的 ECE，Brier score 仅次于 active DeepEnsemble。
  - **离线 RL**：使用 PreferenceEKF 学习的奖励模型训练的 IQL 策略性能与基线方法相当，接近 ground truth reward 训练的策略。
  - **消融实验**：SVD 方法在小子空间维度表现更好，随机投影在维度增大时可达到或超越 SVD；无初始数据集时随机投影仍可保证良好性能。
  - **统计显著性**：active PreferenceEKF 显著优于 active DeepEnsemble (p<0.001, Cohen's d=2.26)、Dropout (p<0.001, d=5.23)、Laplace (p<0.001, d=5.17)，与 LLMCMC 性能相当但速度更快。
- **最强结果**：在相同 $M=5$ 模型样本数下 PreferenceEKF 仍最优（-0.220 vs DeepEnsemble -0.518 vs Dropout -0.774），证明优势不源于采样数量。

## 相关工作脉络
- **Bıyık et al. (2020, 2022, 2024)**：线性/高斯过程奖励模型的主动偏好学习 InfoGain 采集函数，本文将其推广到高维神经网络奖励模型。
- **Christiano et al. (2017)**：DeepRLHF 使用 DeepEnsemble 进行偏好学习，计算成本高但被广泛采用，PreferenceEKF 提供更低成本的替代方案。
- **Lee et al. (2021b) PEBBLE**：结合重标和预训练的反馈高效交互 RL，属于非贝叶斯方法，本文聚焦贝叶斯不确定性量化路径。
- **Izmailov et al. (2020)**：提出子空间推断用于贝叶斯深度学习，本文首次将其应用于奖励模型和主动学习场景。
- **Duran-Martin et al. (2022)**：将贝叶斯滤波与子空间方法结合用于神经网络带猪，本文延续此思路并适配到偏好学习设定。
- **Osband et al. (2022, 2023)**：Epistemic Neural Networks 关注联合预测不确定性，本文聚焦参数边际不确定性，两者互补。

## 局限性与未来方向
- **单注释者假设**：EKF 的高斯单峰假设限制了多模态偏好分布的学习，Crowdsourced 人类标注实验显示所有方法性能受限。
- **观测维度扩展挑战**：EKF 更新步随观测空间维度立方增长，直接处理原始像素不可行，需依赖预训练嵌入。
- **基础模型规模适应性未知**：方法在 MLP 尺度有效，但未验证在 foundation model-scale 奖励模型上的适用性。
- **未来方向**：扩展至多注释者/多模态偏好分布（如粒子滤波器）、结合 epistemic neural networks 的联合预测不确定性、探索奖励模型不确定性在 RL 探索和防止 reward hacking 中的作用、应用于机器人学习微调场景。

## 研究启发与可借鉴点
- **子空间+滤波的通用范式**：SVD/随机投影构造子空间 + EKF 序列更新的框架可迁移到其他需要高效贝叶斯推断的神经网络场景。
- **迭代线性化的实践价值**：使用 n_linearize=5 次迭代 EKF 在轻微额外计算下显著提升 log-likelihood，值得在其他滤波训练中借鉴。
- **噪声超参敏感性分析**：详细 ablation 展示了 W、V、U 三个噪声协方差对后验更新强度的影响规律（过大过小分别导致过拟合/欠拟合），为调参提供指导。
- **无初始数据集的灵活性**：随机投影子空间构造可完全消除对 warm-up 数据的依赖，扩展了方法在 cold-start 场景的适用性。
- **离线 RL 评估的参考价值**：展示奖励模型质量与策略性能的非单调关系（log-likelihood 差异大但策略性能接近），提醒研究者多维度评估。

## 关键术语表
- **PreferenceEKF**：本文提出的方法名，结合子空间推断与扩展卡尔曼滤波器的主动偏好学习奖励建模方法。
- **Bradley-Terry (BT) 模型**：用于 pairwise preference 学习的概率模型，通过 sigmoid 函数将轨迹回报差转化为偏好概率。
- **InfoGain 采集函数**：基于互信息的主动学习查询选择策略，最大化查询标签与模型参数之间的信息增益。
- **子空间推断**：在低维子空间（而非全参数空间）中执行贝叶斯推断，利用神经网络过参数化特性降低计算复杂度。
- **扩展卡尔曼滤波器 (EKF)**：处理非线性系统的序列贝叶斯滤波算法，通过一阶泰勒展开线性化实现闭环后验更新。
- **Expected Calibration Error (ECE)**：衡量模型预测概率校准程度的指标，将预测置信度区间分桶后比较平均置信度与准确率的偏差。
- **离线强化学习 (Offline RL)**：从固定数据集训练策略而非在线交互的方法，本文使用 Implicit Q-Learning (IQL) 作为离线 RL 算法。
- **后验采样**：从参数后验分布 $p(\pmb{\theta}|\mathcal{D})$ 中抽取多个模型样本，用于估计不确定性或计算采集函数。

## 可复现要素
- **代码开源**：JAX 框架实现已发布在 https://github.com/yutaizhou/bnn_pref。
- **依赖库**：Dynamax（EKF 实现）、Laplax（Laplace 近似）、BlackJAX（MCMC）、Unifloral（IQL 实现）、SciPy（统计检验）。
- **实验配置**：8× NVIDIA RTX A6000 GPU（SLURM sharding），MLP 奖励模型（2 层隐藏层，每层 64 单元），子空间维度 $|z|=200$，查询预算 $B=60$，轨迹段长度 50，初始化 warm-up 数据集 8 对查询。
- **数据集公开性**：D4RL 公开基准、V-D4RL 公开基准、SOAR 机器人数据集（Zhou et al., 2024）。
- **关键超参**：温度参数 $\beta=7$，测量噪声 $\mathbf{V}=0.07\cdot\mathbf{I}$，动力学噪声 $\mathbf{U}=0.0001\cdot\mathbf{I}$，先验噪声 $\mathbf{W}=0.07\cdot\mathbf{I}$，迭代线性化次数 $n_{\text{linearize}}=5$，模型样本数 $M=100$（PreferenceEKF/Dropout）或 $M=5$（DeepEnsemble）。
