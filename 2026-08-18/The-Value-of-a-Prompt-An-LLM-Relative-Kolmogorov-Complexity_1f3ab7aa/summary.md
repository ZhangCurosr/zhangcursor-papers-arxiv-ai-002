---
title: "The-Value-of-a-Prompt-An-LLM-Relative-Kolmogorov-Complexity"
source: https://arxiv.org/pdf/2608.16438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:54:39"
field: "LLM 评估与提示工程"
keywords: ["prompt evaluation", "algorithmic information theory", "Kolmogorov complexity", "LLM reasoning", "prompt value", "thinking LLM", "reproduction cost"]
innovations: ["将 LLM 采样随机性等价于 program tape，使 Kolmogorov 复杂度在固定模型下可高效估计", "引入 thinking-path Levin 复杂度与 pKt，统一度量 prompt 的概率增益与计算成本节约", "证明 2^pKt 等于典型重放 token 花费，赋予 prompt value 精确的经济解释"]
benchmarks: ["GSM8K"]
---

# 论文速读：The-Value-of-a-Prompt-An-LLM-Relative-Kolmogorov-Complexity

## 一句话总结
本文提出了一种基于 **LLM 相对 Kolmogorov 复杂度** 的 prompt 价值度量框架（$\widetilde{\mathrm{Val}}_M^\kappa$），通过计算 prompt 与 artifact 之间的算法互信息（pKt 复杂度），量化 prompt 对 LLM 生成目标结果的贡献——既捕获生成概率的提升，也计入思考计算的节省。

## 研究问题与动机
1. **核心问题**：在 LLM  increasingly 参与创作的背景下，如何度量人类提供的 prompt / hint / 部分解对最终 artifact（证明、程序、设计等）的真实价值？
2. **现有方法的不足**：
   - 简单按 token 计数无法捕捉价值（短 hint 可能决定性地改变输出，长 prompt 可能冗余甚至有害）。
   - 已有 "author-contribution" 分数（如 Xie et al. [XQY⁺26]）仅基于条件自信息差，**没有显式地对 thinking / 计算成本进行计费**。
   - 经典 Kolmogorov 复杂度不可计算，且忽略时间资源，不适合度量"帮助模型减少思考"这一关键场景。
3. **需要兼顾的两件事**：prompt 使 artifact 更容易被采样的概率增益 + prompt 缩短所需思考步数带来的计算成本节约。

## 核心贡献（创新点）
1. **提出 LLM 相对 Kolmogorov 复杂度**：将通用图灵机替换为 LLM 本身，把模型的采样随机性视为"程序"，使原先不可计算的算法信息论在固定模型下可高效估计。
2. **引入实现在思考路径上的 Levin 复杂度 $\widetilde{Kt}_M^\kappa$ 与概率版 $\widetilde{\mathrm{pKt}}_M^\kappa$**：首次在 prompt-value 的定义中同时纳入描述长度与 thinking token 成本（通过对截断索引 $t$ 取 min），使"prompt 减少思考时间"被定量记账。
3. **给出重放实验的经济解释**：证明 $2^{\widetilde{\mathrm{pKt}}}$ 恰好等于"在中位 thinking 路径上重放生成 artifact 的典型 token 花费"，prompt value 即为有 / 无 prompt 时的典型花费比率。
4. **提供高效估计协议与统计保证**：仅需 $k = O(\zeta^{-2}\log(1/\eta))$ 次独立 rollout，经验分位数即能以高概率落在真实分位数 $\pm\zeta$ 区间内。
5. **GSM8K 小规模实验揭示非思考度量的误判风险**：在 12 道题上演示，纯概率层面的 prompt value 可能在 5/6 个案例中对正确首步给出负值，但计入 thinking 成本后转为正值。

## 方法详解
1. **非思考 LLM 的 prompt value（热身）**：
   - 定义 LLM 相对先验复杂度 $\widetilde{K}_M(z|y) = -\log_2 P_M(z|y)$。
   - Prompt value 为 $\widetilde{\mathrm{Val}}_M(p;z) = \widetilde{K}_M(z) - \widetilde{K}_M(z|p) = \log_2 \frac{P_M(z|p)}{P_M(z)}$，即逐 token log-likelihood ratio 之和。
   - 定理 2.6 证明其与程序长度版 $K_M$ 至多相差 2 bit。
2. **思考 LLM 的建模**：两阶段自回归——先生成 thinking 字符串 $H^y$（含 EOT 停止标记），再生成 artifact。
3. **实现在思考路径上的 Levin 复杂度**（式 4）：
   $$\widetilde{Kt}_M^\kappa(z|y; H^y) = \min_{t \in \mathbb{N}_0} \Big\{ \underbrace{-\log_2 P_M(z|y H_{\le t}^y \mathrm{EOT})}_{\text{description length}} + \underbrace{\log_2 \kappa(t)}_{\text{log running time}} \Big\}$$
   其中 $\kappa(t)$ 为 token-equivalent cost（默认 $c_{\text{pre}}=1, c_{\text{dec}}=32$ 等）。
4. **概率 Levin 复杂度**（式 5）：对 thinking 随机性取中位数（或 $\delta$-分位数）：
   $$\widetilde{\mathrm{pKt}}_{M,\delta}^\kappa(z|y) = \mathrm{med}_\delta\!\big[\widetilde{Kt}_M^\kappa(z|y; H^y)\big]$$
5. **Prompt value 最终定义**（式 6）：
   $$\widetilde{\mathrm{Val}}_{M,\delta}^\kappa(p;z) = \widetilde{\mathrm{pKt}}_{M,\delta}^\kappa(z) - \widetilde{\mathrm{pKt}}_{M,\delta}^\kappa(z|p)$$
   即算法互信息；b bits 意味着无 prompt 时典型重放花费是有 prompt 时的 $2^b$ 倍。
6. **估计协议**（Section 4）：每侧 $k$ 次独立 rollout → 每条路径枚举 $t=0,\dots,S$ → 取经验 $\delta$-分位数 → 两者相减；Chernoff 界保证误差。
7. **验证型 artifact 特例**（Section 3.5）：若存在判定器 $\mathcal{V}$，则 $\widetilde{\mathrm{Val}} = \log_2(\tau_\epsilon / \tau_p)$，其中 $\tau$ 为 acceptance 的最小成本的中位数。

## 实验与结果
- **数据集**：GSM8K 训练集抽样 12 题（参考解含 ≥3 步换行步骤）。
- **模型**：deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B，FP16，temperature 0.6，top-p=1。
- **设置**：每题每条件 64 次独立 rollout；prompt 为首步参考解（去 calculator 标注）。
- **成本参数**：默认 $c_{\text{pre}}=1, c_{\text{dec}}=32$；敏感性扫描 $c_{\text{dec}} \in [8, 256]$。
- **主要发现**：
  - **Thinking reverses non-thinking verdict**：6 个正例中 5 个在 $t=0$ 处概率反降，但凭借更快达到高 prob 而整体价值为正。
  - **Cost convention 差异显著**：generated-thought 值多为正，prefix-prefill 值接近零——说明部分增益来自"加速原本需 sequentially 解码的思考"而非纯粹概率提升。
  - **Correct ≠ valuable**：另一半 6 题中效果从有害到接近零不等，说明"正确但非最优"的提示可能无价值。
  - **Distribution-dependent**：同一 prompt 的 value 随 $\delta$ 变化而穿越零，中位数不能代表全分布。

## 相关工作脉络
1. **Kolmogorov / Levin Kt 复杂度**：经典算法信息论（Kolmogorov, Chaitin, Solomonof）与 Levin 的时间-描述长度复合度量是本工作的理论起点；资源限制并不能使复杂度可高效计算（Allender et al. [ABK⁺06]）。
2. **Probabilistic Kolmogorov complexity**（Goldberg et al. [GKLO22]）：以随机带 $r$ 为条件的 $pK_\delta^t$ 定义与本文结构同源，但参考机是通用 TM；本文将其替换为 LLM 并引入 cost 函数。
3. **Xie et al. [XQY⁺26]**：提出 AI 辅助生成中 human contribution 的归一化分数；本文的 non-thinking 版本是其未归一化分子，且本文**首次显式计入 thinking 计算成本**。
4. **Sorensen et al. [SRR⁺22]**：用 Shannon MI 挑选 prompt template；目标是模板排序而非度量特定 prompt 对特定 artifact 的价值。
5. **PMI / 信息论传统**：$\log P(z|p)/P(z)$ 的形式即点态互信息（Fano 1961; Church & Hanks 1990），本文将其推广到 thinking + cost 场景。
6. **Halpern & Pass 的计算信息价值**：value-of-computation 视角（HP11, HP15）与零知识模拟范式 [GMR89] 提供哲学基础——信息价值在于"无此信息时能多高效生成"。
7. **LLM watermarking**（Kuditipudi et al. [KTHL24], Christ et al. [CGZ24]）：共享"随机数 × 自回归 ⇒ 输出"的同构思想，但本文是价值度量而非水印。

## 局限性与未来方向
1. **Artifact 声明的主观性**：可构造"随机后缀 + 注入后缀的 prompt"骗取高 value；需 canonicalization 或 verifier 来约束。
2. **单轮 prompt 局限**：未处理多轮对话中 adaptive human policy 的价值（作者明确列为未来工作）。
3. **语义重随机化未验证**：提出用另一 LLM 改写保留语义但变换表层，以排除 inert padding，但实验部分留白。
4. **Cost 函数约定的人工性**：$\kappa^{\text{gen}}$ vs $\kappa^{\text{pre}}$ 等是 stylized 近似，真实 inference 的 KV-cache / prefill 并行度更复杂。
5. **实验规模小**：仅 12 题、单模型、单任务（GSM8K），未见跨模型 / 跨任务的泛化验证。

## 研究启发与可借鉴点
1. **"把模型自身当作图灵机"**：固定模型后，将采样随机性离散化为 dyadic program，使不可计算的 Kolmogorov 复杂度转化为可在多项式时间内估计的对数似然差——这一技巧可迁移至其他需度量"模型依赖信息量"的场景。
2. **中位数 vs 期望的统计选择**：用分位数而非期望来汇总 thinking 路径的复杂度，避免被极少数的"一语惊醒"路径主导——对任意重尾分布的生成过程均有参考价值。
3. **minimize over truncation index 的 trick**：式 (4) 中对 $t$ 取 min，天然编码了"概率提升 ↔ 计算节省"的权衡，可借鉴到其他需联合优化 success rate 与 latency 的 prompt 评估任务。
4. **重放实验解释**：用几何分布期望给出 $2^{\widetilde{Kt}} = \mathrm{median}(\mathrm{TokenCost})$ 的精确等式，使抽象的"bits of value"获得可操作的 token-cost 语义——可用于设计 prompt ROI 指标。
5. **verified artifact 特例的简化**：Section 3.5 的 acceptance-cost 形式 $\log_2(\tau_\epsilon/\tau_p)$ 极为简洁，可推广至代码生成（编译 acceptance）、数学证明（checker acceptance）等场景。

## 关键术语表
- **LLM-relative Kolmogorov complexity** $K_M$：以 LLM 采样随机性的最短 dyadic program 长度替代通用 TM 下的最短程序长度。
- **A priori complexity** $\widetilde{K}_M$：$-\log_2 P_M(z|y)$，与 $K_M$ 相差至多 2 bit，可直接从 log-prob 求和得到。
- **Realized-thought Levin complexity** $\widetilde{Kt}_M^\kappa$：沿固定 thinking 路径，最小化"条件自信息 + log 成本"的组合。
- **Probabilistic Levin complexity** $\widetilde{\mathrm{pKt}}_{M,\delta}^\kappa$：对 thinking 随机性取 $\delta$-分位数，得到不依赖于单次随机 rollout 的复杂度。
- **Prompt value** $\widetilde{\mathrm{Val}}_{M,\delta}^\kappa(p;z)$：有 / 无 prompt 的 pKt 复杂度之差，单位为 bits；$b$ bit 表示无 prompt 时典型重放花费是有 prompt 时的 $2^b$ 倍。
- **Token-equivalent cost function** $\kappa(t)$：将 thinking 步数映射为 token 等价花费，区分 generated-thought 与 prefix-prefill 两种约定。
- **Reproduction experiment**：固定 thinking 路径后重复采样 output stage 直至命中 artifact，其期望总花费的 median 等于 $2^{\widetilde{\mathrm{pKt}}}$。
- **Canonical / verified artifact**：通过判定器 $\mathcal{V}$ 声明一类可接受 artifact，使 prompt value 退化为 acceptance cost 中位数的对数比。

## 可复现要素
- **数据集**：GSM8K（公开），作者从训练集无放回抽样 100 题后保留前 12 题。
- **代码**：作者附 online notebook（文中脚注 9），可复现抽样、rollout、评分、绘图全流程。
- **模型**：deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B（HuggingFace 公开）。
- **关键超参**：temperature=0.6，top-p=1（作者刻意不设 nucleus truncation），每条件 64 rollouts；成本 $c_{\text{pre}}=1, c_{\text{dec}}=32$，敏感性网格 $c_{\text{dec}} \in [8,256]$。
- **模型精度**：FP16 加载，logits 转 FP32 再计算概率。
