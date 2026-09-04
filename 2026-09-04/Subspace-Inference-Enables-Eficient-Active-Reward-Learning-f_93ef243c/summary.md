---
title: "Subspace-Inference-Enables-Eficient-Active-Reward-Learning-f"
source: https://arxiv.org/pdf/2609.04066v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:07:51"
field: "Active Reward Learning from Preferences"
keywords: ["Active Preference Learning", "Bayesian Deep Learning", "Extended Kalman Filter", "Subspace Inference", "Reward Model", "RLHF", "Uncertainty Quantification"]
innovations: ["将主动偏好奖励学习框架化为序列贝叶斯滤波问题，利用EKF在低维子空间中高效推断神经网络后验", "首次将子空间滤波用于神经网络上网络奖励模型训练，支持任意数量采样以计算InfoGain等获取函数", "在D4RL/V-D4RL基准上实现了样本效率、运行时间、可扩展性和校准全面优于主流贝叶斯深度学习方法"]
benchmarks: ["D4RL", "V-D4RL", "SOAR"]
---

# 论文速读：Subspace-Inference-Enables-Eficient-Active-Reward-Learning-from-Preferences

## 一句话总结
本文提出 **PreferenceEKF**，将主动偏好奖励学习框架化为**序列贝叶斯滤波问题**，通过在低维参数子空间中使用**扩展卡尔曼滤波（EKF）**高效推断神经网络奖励模型的后验分布，实现样本高效、可扩展的主动偏好学习。

## 研究问题与动机
1. **RLHF样本效率低**：偏好反馈每次只提供1比特信息，让注释者回答数千个问题不可扩展。
2. **主动学习需要不确定性量化**：计算获取函数（acquisition function）需要对模型参数后验进行概率建模，但深度神经网络的高维参数空间使精确推断计算代价高昂。
3. **现有方法的局限**：Ensemble需训练多个独立模型，计算密集；Dropout虽轻量但后验近似质量差，学术界对其有效性存在争议。
4. **贝叶斯深度学习的可扩展性挑战**：传统贝叶斯方法（如MCMC、变分推断）难以直接扩展到大规模神经网络，需要更高效的近似推断策略。

## 核心贡献（创新点）
1. **首个将子空间滤波用于神经网络上网络奖励模型训练的框架（PreferenceEKF）**，相比基于Ensemble/Dropout的方法在样本效率和运行时显著更优。
2. **将偏好奖励学习重新表述为序列贝叶斯滤波问题**，利用EKF在低维子空间中进行递归后验更新，避免了重复训练已见数据导致的灾难性遗忘问题。
3. **支持从后验中采样任意数量模型以计算InfoGain等获取函数**，突破了InfoGain此前仅适用于低维线性/GP奖励模型的限制。
4. **在D4RL和V-D4RL基准上系统性验证**：相比四种主流贝叶斯深度学习方法（DeepEnsemble、Dropout、Laplace、LLMCMC），在样本效率、校准质量、运行时间可扩展性上均取得更优或相当的表现。

## 方法详解
**1. EKF 训练神经网络**
- 将神经网络参数 $\theta$ 视为隐藏状态，动态模型设为恒等映射 $g(\theta) = \theta$，测量模型使用 Bradley-Terry (BT) 似然 $h(\theta, Q) = p_\theta(\tau_a \succ \tau_b)$。
- 假设加性高斯噪声：动态噪声 $p(\theta_i|\theta_{i-1}) = \mathcal{N}(\theta_i | \theta_{i-1}, \mathbf{U})$，测量噪声 $p(y_i|\theta_i) = \mathcal{N}(y_i | h(\theta_i, Q_i), \mathbf{V})$。
- 后验保持高斯形式 $b^i = \mathcal{N}(\mu_i, \Sigma_i)$，通过预测步和更新步递归计算：
  - 预测：$\mu_{i|i-1} = \mu_{i-1}$，$\Sigma_{i|i-1} = \Sigma_{i-1} + \mathbf{U}$
  - 更新：利用BT似然的 Jacobian $\mathbf{H}_i = \sigma(z)(1-\sigma(z))(\nabla_\theta r_\theta(\tau_a) - \nabla_\theta r_\theta(\tau_b))$ 计算卡尔曼增益 $\mathbf{K}_i = \Sigma_{i|i-1}\mathbf{H}_i^\top (\mathbf{H}_i \Sigma_{i|i-1} \mathbf{H}_i^\top + \mathbf{V})^{-1}$

**2. 子空间推断（Subspace Inference）**
- 全参数空间协方差矩阵规模 $O(|\theta|^2)$ 不可行，将参数映射到 $|z| \ll |\theta|$ 的低维子空间：$\theta(z) = \mathbf{A}z + \theta_*$
- $\theta_*$ 由小规模预训练集上的SGD初始化；$\mathbf{A} \in \mathbb{R}^{|\theta| \times |z|}$ 通过对SGD迭代轨迹做SVD获得（主成分保留）。
- 后验表示为 $b^i(z) = \mathcal{N}(\mu'_i, \Sigma'_i)$，从子空间采样后通过 $\theta(z)$ 投影回全空间进行前向传播。

**3. 基于子空间推断的主动学习**
- 使用 InfoGain 获取函数：$Q_i^* = \arg\max_{Q_i} I(\theta; y_i | Q_i, b^{i-1})$
- 通过从 $b^{i-1}$ 采样 $M$ 个 $\theta$ 样本近似计算，PreferenceEKF 可高效采样任意数量模型而无需重新训练。

## 实验与结果
- **数据集**：D4RL（12个MuJoCo/Adroit/Maze2D任务）、V-D4RL（像素版）、SOAR真实机器人数据。
- **基线**：DeepEnsemble、Dropout、Laplace Approximation、Last-Layer MCMC (LLMCMC)。
- **评估指标**：测试集BT对数似然、训练运行时、校准误差（ECE/Brier score）、离线RL策略性能。
- **关键结果**：
  - **样本效率**：PreferenceEKF在12个D4RL任务上聚合的测试对数似显著优于所有基线（bootstrap检验 $p < 0.001$）。
  - **运行时**：约5× 快于 DeepEnsemble，40× 快于 LLMCMC（主动学习下 12.1min vs 97.7min vs 780.2min）。
  - **可扩展性**：随采样数 $M$ 和模型规模增大，PreferenceEKF运行时增长最平缓，且对数似然保持稳定。
  - **校准**：ECE最低（优于所有基线），Brier score第二低。
  - **离线RL**：使用PreferenceEKF学习到的奖励模型训练的IQL策略，平均性能接近 ground-truth 奖励下训练的GT策略。
  - **消融**：SVD子空间在较小维度更优，随机投影在大维度时可匹敌SVD；无初始数据集时随机投影仍可正常工作。

## 相关工作脉络
1. **Bıyık et al. (2020)** 提出InfoGain获取函数用于主动偏好学习，但仅限于低维线性/GP奖励模型；本文将其扩展到神经网络。
2. **Duran-Martin et al. (2022)** 首次在深度网络上应用子空间贝叶斯滤波；本文将其应用于奖励学习场景。
3. **Christiano et al. (2017)** 使用Ensemble进行主动偏好学习（DeepRLHF）；本文避免了训练多个独立模型的成本。
4. **Izmailov et al. (2020)** 提出通过SGD迭代SVD构建子空间；本文沿用并验证了该技术在贝叶斯推断中的有效性。
5. **Laplace Approximation (Daxberger et al., 2024)** 在单个最优参数附近做高斯近似；本文顺序更新后验，更适合流式数据。

## 局限性与未来方向
1. **单模态假设**：EKF维持高斯后验，无法捕捉多注释者带来的多模态偏好分布。
2. **可扩展性未知**：未验证在foundation model规模奖励模型上的有效性。
3. **未来方向**：探索粒子滤波处理多模态后验；结合epistemic neural networks研究联合预测不确定性；研究奖励后验对RL策略探索的辅助作用。

## 研究启发与可借鉴点
1. **顺序贝叶斯滤波 + 子空间推断的组合策略**可迁移到任意需要在线更新高维模型不确定性的场景（如在线强化学习、持续学习）。
2. **InfoGain在神经网络上的成功应用**验证了信息论获取函数在高维模型中的可行性，前提是拥有高效的采样机制。
3. **无需存储全部历史数据**：PreferenceEKF仅用最新一对(query, label)更新后验，相比基线需重新训练所有数据，大幅降低存储与计算成本。
4. **无初始数据集时的随机投影子空间**为冷启动场景提供了可行方案，解耦了对预训练数据的依赖。
5. **EKF迭代线性化（iterated EKF）** 的引入提升了精度，且对线性化步数超参相对鲁棒（$n_{linearize}=5$），可作为设计参考。

## 关键术语表
**PreferenceEKF**：本文提出的方法，在低维参数子空间中利用扩展卡尔曼滤波进行序列贝叶斯推断，实现高效的主动偏好奖励学习。
**Bradley-Terry (BT) 模型**：用于成对偏好比较的概率模型，轨迹a优于轨迹b的概率为 $\frac{\exp(\beta R(\tau_a))}{\exp(\beta R(\tau_a)) + \exp(\beta R(\tau_b))}$。
**InfoGain**：基于互信息的获取函数，选择能使模型参数后验熵减少最多的查询，最大化每轮反馈的信息量。
**子空间推断（Subspace Inference）**：通过对高维参数空间进行低秩投影，在低维子空间中进行贝叶斯推断以大幅降低计算复杂度。
**扩展卡尔曼滤波（EKF）**：处理非线性动态/测量系统的顺序贝叶斯滤波算法，通过一阶泰勒展开线性化后应用标准卡尔曼滤波公式。
**主动偏好学习**：迭代地选择最有信息量的成对偏好查询，以最少的人类反馈学会注释者的奖励函数。

## 可复现要素
- **代码**：开源，JAX实现，见 https://github.com/yutaizhou/bnn_pref
- **数据集**：D4RL（公开）、V-D4RL（公开）、SOAR（公开）
- **关键超参**：子空间维度 $|z|=200$（像素任务500）、查询预算 $B=60$、轨迹段长度50、采样数 $M=100$（DeepEnsemble $M=5$）、测量噪声 $\mathbf{V}=0.07\mathbf{I}$、动态噪声 $\mathbf{U}=0.0001\mathbf{I}$、先验噪声 $\mathbf{W}=0.07\mathbf{I}$、迭代线性化步数 $n_{linearize}=5$、温度参数 $\beta=7$。
