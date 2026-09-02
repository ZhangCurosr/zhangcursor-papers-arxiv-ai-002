---
title: "Residual-Sparsification-via-Output-Importance-for-Compressin"
source: https://arxiv.org/pdf/2609.00575v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:16:01"
field: "MoE LLM 压缩"
keywords: ["Mixture-of-Experts", "Model Compression", "Residual Sparsification", "Large Language Models", "GPU Memory Optimization"]
innovations: ["提出基于输出重要性的残差稀疏化目标，替代矩阵级误差最小化", "设计 hidden dimension 输出影响的可近似评估公式（Eq. 10）", "引入跨专家全局池化选择维度，避免高频 expert 过压缩"]
benchmarks: ["lm-eval-harness ARC/WinoGrande/HellaSwag/PIQA/OpenBookQA/MMLU", "WikiText Perplexity", "IFEval Instruction Following"]
---

# 论文速读：Residual-Sparsification-via-Output-Importance-for-Compressin

## 一句话总结
本文提出 PARSE，一种基于"输出重要性"的残差稀疏化方法，将 MoE-LLM 压缩目标从最小化单个投影矩阵误差转变为最小化专家最终输出误差，在相同显存节省下显著缩小与未压缩模型的精度差距。

## 研究问题与动机
- **MoE-LLM 显存瓶颈**：MoE 架构推理时需同时加载所有专家参数，例如 Mixtral-8x7B 需 97 GB 参数、峰值显存达 113 GB，严重依赖多 GPU 部署。
- **现有残差稀疏化方法的精度错位**：ResMoE、D2MoE 等方法仅最小化各投影矩阵（$W_u, W_g, W_d$）的压缩误差，但论文理论分析证明这种"矩阵级误差最小化"与"专家输出误差最小化"不对齐，因 hidden representation 的误差会经由非线性激活（SwiGLU）和矩阵交互被放大。
- **实证验证错位**：在 Qwen 上以 70% 压缩率测试 ResMoE 发现，即使 $\Delta W$ 被压缩到很窄范围，$\Delta h(x)$ 和 $\Delta E(x)$ 仍存在 1.3× 和 1.44× 的平均放大。
- **缺少按输出影响筛选维度的机制**：现有方法无差别地独立压缩每个残差矩阵，未考虑不同 hidden dimension 对最终输出的差异化贡献。

## 核心贡献（创新点）
- **揭示矩阵级误差最小化与输出级误差最小化的不对齐**：通过理论推导（Eq. 6）证明现有方法最小化 $\Delta W_u, \Delta W_g, \Delta W_d$ 并不足以最小化最终专家输出误差 $\Delta E(x)$。
- **提出 PARSER，以输出重要性替代矩阵误差作为压缩目标**：首次将 hidden dimension 对专家输出的实际贡献纳入残差稀疏化选择标准，而非仅依赖 Frobenius norm 最小化。
- **设计可高效近似输出的重要性评估公式**：将全维度反向传播简化为仅需计算第 $j$ 维参数与 base-only 贡献之差（Eq. 10），避免对每个维度做完整前向传播。
- **全局池化（global pooling）跨专家联合选择待压缩维度**：不按每个专家独立压缩，而是将所有专家的 $H(h_j)$ 分数汇总后选取最低的 $K$ 个维度，避免高频路由专家的维度被过度压缩。

## 方法详解
**分解与残差定义**：每个专家的三个投影矩阵（$W_u, W_g, W_d$）分解为共享 base 矩阵 $B_p$ 和 per-expert 残差 $R_p^{(i)} = W_p^{(i)} - B_p$，base 矩阵通过 Wasserstein barycenter 计算并保持原样。

**输出重要性定义（Eq. 7）**：
$$H(h_j) = \mathbb{E}_x \| E(x) - \hat{E}_{h_j}(x) \|_2^2$$
其中 $\hat{E}_{h_j}(x)$ 是将第 $j$ 个 hidden dimension 压缩后（用 base 替代）产生的专家输出。$H(h_j)$ 衡量移除该维度对最终输出的影响。

**可计算近似（Eq. 10）**：
$$H(h_j) \approx \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \| W_{d,j} h_j(x) - B_{d,j} h_j^b(x) \|_2^2$$
利用 $(E(x) - \hat{E}_{h_j}(x))$ 仅与第 $j$ 维有关，将复杂度从全维度反向传播降至单次逐维度计算。

**压缩流程（Algorithm 1）**：
1. 给定目标压缩比 $\rho$，计算待压缩维度总数 $K = \rho \cdot N \cdot H$（$N$ 专家数，$H$ 隐藏维度数）。
2. 对校准数据集 $\mathcal{D}$（约 1M tokens）计算所有维度的 $H(h_j)$。
3. 将所有专家的同层 $H(h_j)$ 合并 pooling。
4. 选取全局最小的 $K$ 个维度，移除对应 $R_u, R_g$ 的第 $j$ 行和 $R_d$ 的第 $j$ 列。
5. 保留至少 4 个 residual 维度以避免某专家被完全移除。

**推理重建**：压缩后，$\hat{W}_p^{(i)} = B_p + \hat{R}_p^{(i)}$，其中未压缩维度的 residual 正常参与计算，被移除维度的输出仅由 base 贡献。

## 实验与结果
**数据集与模型**：Qwen1.5-MoE-A2.7B、DeepSeek-V2-Lite，以及 OLMoE-1B-7B-0125 和 Moonlight-16B-A3B。评测用 lm-eval-harness 七项任务（ARC-Easy/Challenge、WinoGrande、HellaSwag、OpenBookQA、PIQA、MMLU），外加 WikiText 和 IFEval。

**主要结果（$\rho=90\%$，Table 1）**：
- **Qwen AVG**：无压缩 59.12% → PARSER 49.24%（差距 9.88pp）；最佳基线 D2MoE 45.22%（差距 13.9pp）。PARSER 将精度差距缩小 1.41×。
- **DeepSeek AVG**：无压缩 59.64% → PARSER 47.16%（差距 12.48pp）；D2MoE 41.66%（差距 17.98pp）。PARSER 将精度差距缩小 1.44×。
- **峰值显存**：Qwen 无压缩 65.85 GB → PARSER 52.19 GB（↓30.8%）；DeepSeek 无压缩 86.65 GB → PARSER 70.26 GB（↓18.9%），与最佳基线持平。

**额外模型（Table 8）**：
- OLMoE：PARSER 36.73% vs D2MoE 34.46%（+2.27pp）。
- Moonlight：PARSER 46.15% vs HC-SMoE 39.72%（+6.43pp）。

**不同压缩比（Table 9-10）**：在 $\rho=70\%$ 和 $80\%$ 下 PARSER 持续最优，平均精度差距缩小 1.44×。

**WikiText perplexity（Table 11）**：$\rho=90\%$ 时 Qwen 上 PARSER 20.71 vs D2MoE 23.97；DeepSeek 上 19.85 vs 31.24，优势显著。

**IFEval（Table 12）**：$\rho=90\%$ 时 Qwen 上 PARSER 18.85% vs D2MoE 14.97%；DeepSeek 上 21.81% vs 19.41%。

**消融（Table 3-4）**：
- 输出重要性 vs 矩阵误差最小化：Qwen 上提升 1.09×；vs Wanda 提升 1.05×。
- Global pooling vs Local selection：Qwen +0.96pp，DeepSeek +2.04pp。
- Global pooling vs Routing-aware global pooling：持续提升 0.6-0.64pp。

**延迟（Table 5）**：压缩时间 PARSER 比 ResMoE 慢约 1.17×（一次性成本）；推理吞吐 PARSER 最高，优于 ResMoE 1.05×、优于 D2MoE 2.03×。

## 相关工作脉络
- **ResMoE（Ai et al., 2025）**：残差稀疏化 SOTA，采用 TSVD 或 unstructured pruning 压缩残差，目标为 Frobenius norm 最小化，PARSER 在其基础上以输出重要性替代矩阵误差目标。
- **D2MoE（Gu et al., 2025）**：delta decompression 方法，同样基于残差稀疏化框架，PARSER 通过引入输出重要性在相同压缩比下获得更高精度。
- **MoE-I²（Yang et al., 2024）**：expert pruning + intra-expert low-rank decomposition，压缩粒度更粗（整 expert），PARSER 在细粒度维度选择上更优。
- **HC-SMoE（Chen et al., 2025）**：expert merging via hierarchical clustering，将相似专家合并，属于 coarse-grained 压缩；PARSER 属于 fine-grained residual 压缩。
- **Wanda（Sun et al., 2023）**：基于 activation-norm 的参数重要性剪枝，最初针对 dense LLM，本文将其适配为 MoE residual 压缩的消融对比基线。
- **SparseGPT（Frantar & Alistarh, 2023）**：one-shot 结构化剪枝，使用校准数据估计 Hessian 曲率；PARSER 与之共用 calibration dataset 思路，但目标从单层参数重要性扩展到跨专家跨维度的输出影响评估。

## 局限性与未来方向
- **忽略维度间交互**：$H(h_j)$ 是逐个维度独立评估的（diagonal approximation），未捕捉被移除维度间的误差交互项（$\langle \delta_j(x), \delta_k(x)\rangle$），可能在高压缩比下产生累积偏差。
- **依赖校准数据集 $\mathcal{D}$**：虽实验证明对 seed、source、size 变化鲁棒（最大 $\sigma=2.03$pp），但在域外任务或领域特定的 MoE 上可能需要定制化 $\mathcal{D}$。
- **仅针对推理压缩**：PARSER 设计目标是零重训的一次性压缩，未涉及训练中压缩或 fine-tuning 适配的动态压缩场景。
- **未覆盖强逻辑/安全敏感领域**：医学推理、复杂代码生成等对精度均匀性要求更高的场景可能需要额外的 domain-aware 目标设计。

## 研究启发与可借鉴点
- **目标函数设计的对齐原则**：将压缩目标从"局部重构误差"调整为"下游任务输出误差"是通用的性能提升策略，可迁移至其他模型压缩场景（如 dense LLM 剪枝、量化）。
- **可近似的全维度重要性评估公式**：从 Eq. 10 可见，将期望操作替换为校准集均值、利用结构分解规避反向传播，可在保持精度的同时大幅降低计算开销，此模式可复用于其他稀疏化方法。
- **跨专家全局池化 vs 逐专家局部选择**：不同 expert 对最终输出的贡献不均匀，全局 pooling 避免了对高频 expert 的过压缩，这一思想可推广至分层路由或多塔模型的资源分配。
- **保留最小维度保护机制**：防止任何 expert 被完全移除（保留至少 4 个 residual 维度）是一种简单有效的安全边际设计，适合其他结构化压缩方法。
- **校准数据集选择的鲁棒性分析范式**：通过 seed/source/size 三维交叉验证评估重要性估计的稳定性，为后续工作提供可靠的评估框架。

## 关键术语表
- **Residual Sparsification（残差稀疏化）**：将每个专家的投影矩阵分解为共享 base 和 per-expert 残差，仅压缩残差部分以节省显存的技术。
- **Output Importance（输出重要性）**：衡量单个 hidden dimension 被移除后对专家最终输出误差的影响大小，本文提出的核心评分指标。
- **Global Pooling（全局池化）**：将同层所有专家的维度重要性分数汇总后统一选择待压缩维度，而非每个专家独立选择。
- **Wasserstein Barycenter**：用于从多个专家投影矩阵中计算共享 base 矩阵的几何平均方法，保留跨专家共性信息。
- **TSVD / Unstructured Pruning（UP）**：两种残差压缩技术，TSVD 通过截断 SVD 降维，UP 通过置零低重要性参数构建稀疏矩阵。
- **Calibration Dataset（校准数据集 $\mathcal{D}$）**：用于估计 hidden dimension 重要性的一批代表性输入样本，不用于训练，仅用于离线重要性评估。
- **MoE Layer（混合专家层）**：包含多个 expert 和 router 的神经网络层，每个 token 仅激活 k 个 expert 进行计算。

## 可复现要素
- **数据集**：Qwen1.5-MoE-A2.7B、DeepSeek-V2-Lite、OLMoE-1B-7B-0125、Moonlight-16B-A3B（均公开）；评测用 lm-eval-harness 七任务（公开）；校准数据集 Dolly-15K（公开）。
- **代码**：论文声明开源，GitHub: https://github.com/OSSS-KU/PARSER。
- **关键超参**：压缩比 $\rho$（主实验 90%，遍历 10%-90%）；校准数据集大小 512 条序列、每条 2048 tokens；base 矩阵构建使用 Wasserstein barycenter；每专家至少保留 4 个 residual 维度。
- **硬件**：单 NVIDIA B200 GPU，Ubuntu 22.04，PyTorch 2.9.0，CUDA 13.0，bfloat16 精度。
- **压缩层范围**：仅对最后 2/3 的 MoE 层施加压缩（与 ResMoE 一致）。
