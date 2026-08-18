---
title: "The-Value-of-a-Prompt-An-LLM-Relative-Kolmogorov-Complexity"
source: https://arxiv.org/pdf/2608.16438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:12:22"
field: "大语言模型分析与评估"
keywords: ["LLM prompt value", "Kolmogorov complexity", "algorithmic information theory", "thinking models", "token cost estimation", "GSM8K"]
innovations: ["将LLM相对Kolmogorov/Levin复杂度引入prompt价值度量，统一建模概率提升与思考时间节省", "建立pKt复杂度与典型token再现成本的严格指数对应关系", "提出基于rollout采样的高效分位数估计协议并提供高概率置信界"]
benchmarks: ["GSM8K"]
---

# 论文速读：The-Value-of-a-Prompt-An-LLM-Relative-Kolmogorov-Complexity

## 一句话总结
本文提出一种基于LLM相对Kolmogorov复杂度的提示(prompt)价值度量方法，通过将思考路径视为随机磁带并对计算时间收费，统一建模提示对产物生成概率的提升与所需思考时间的节省，并证明该度量与典型token再现成本的对数比直接对应。

## 研究问题与动机
- **核心问题**：当LLM生成一个有价值产物z（如证明、程序、设计）时，如何衡量提供的提示p对该产物的贡献价值？
- **现有不足**：简单测量提示长度/token数无法捕捉价值（短提示可能关键，长提示可能无效甚至有害）。
- **传统Kolmogorov复杂度局限**：不可计算，且忽略了计算复杂度，无法捕获"计算收益"。
- **纯概率度量的缺陷**：仅比较有/无提示的输出概率会遗漏提示减少思考时间的核心优势，在思考型LLM中可能给出错误结论。

## 核心贡献（创新点）
- **提出LLM相对Kolmogorov复杂度**：将LLM本身的采样过程作为参考机，用最短程序长度（决定输出的随机数前缀）定义复杂度，区别于传统通用图灵机设定。
- **引入概率Levin-Kt复杂度(pKt)**：将思考型LLM的实现思考路径H视为程序的随机磁带，结合先验复杂度与log运行时间，并对思考token收取token等价成本κ(t)。
- **建立pKt与经济再现成本的严格联系**：证明2^pKt等于在典型思考路径下重现产物所需的median token成本，prompt value即为无提示与有提示median成本的比值的对数。
- **给出高效可估计的采样协议**：通过多项式次rollout的empirical δ-quantile估计prompt value，并提供Hoeffding型高概率置信界。
- **实验揭示非思考度量的失效**：在GSM8K上演示纯概率度量在某些情况下给出相反结论，而纳入思考成本后得到合理排序。

## 方法详解
- **非思考LLM的prompt value**：定义LLM相对K复杂度K_M(z|y)为能强制LLM输出z的最短二进制程序长度；先验复杂度~K_M(z|y) = -log₂P_M(z|y)。两者相差<2 bits。Prompt value定义为~Val_M(p;z) = ~K_M(z) - ~K_M(z|p)，等价于log(P_M(z|p)/P_M(z))，形式上与点态互信息(PMI)一致。
- **思考LLM建模**：LLM分两阶段运行：首先生成思考token序列H（以EOT终止），然后在上下文yH EOT下生成输出。
- **实现的Levin复杂度**：对固定思考路径H，定义~Kt_M^κ(z|y;H) = min_{t∈ℕ₀}{~K_M(z|yH_{≤t}EOT) + log₂κ(t)}，其中κ(t)为声明的token等价成本函数（如生成式成本κ_gen或前缀预填充成本κ_pre）。
- **概率Levin复杂度(pKt)**：对随机rollout的H取中位数(或δ-分位数)：~pKt_{M,δ}^κ(z|y) = med_δ[~Kt_M^κ(z|y;H^y)]。
- **Prompt value**：~Val_{M,δ}^κ(p;z) = ~pKt_{M,δ}^κ(z) - ~pKt_{M,δ}^κ(z|p)。b bits的value意味着无提示的典型token成本是有提示的2ᵇ倍。
- **估计协议**：对每个条件做k次独立rollout，对每条路径计算所有截断t的最优Kt值，取empirical δ-quantile，两者之差即为估计值。Hoeffding不等式保证以1-4exp(-2kζ²)概率落在真实区间内。
- **成本函数**：κ_pre(t) = c_pre(|y|+t+1)+c_dec|z EOS|，κ_gen(t) = c_pre|y|+c_dec(t+1+|z EOS|)，其中c_dec/c_pre可显著大于1。

## 实验与结果
- **数据集**：GSM8K（小学数学应用题），选取12个参考解≥3步的问题。
- **模型**：DeepSeek-R1-Distill-Qwen-1.5B，FP16加载，logits转FP32，temperature=0.6，top-p=1，64次rollouts。
- **提示设计**：参考答案的第一步（去除calculator标注），作为partial computation插入。
- **评估设置**：比较c_dec∈[8,256]网格下的generated-thought与prefix-prefill两种成本约定，报告δ∈{0.2,0.5,0.8}处的prompt value。
- **关键结果**：
  - 5/6正value案例中，t=0时加入正确第一步反而降低gold artifact概率，但纳入思考后总体value为正——纯概率度量会给出相反定性判断。
  - 12题中仅6题在所有分位数下positive；正确步骤不等于有价值提示。
  - 同一提示在不同δ处value符号可能交叉，说明prompt value具有rollout分布依赖性。
  - 在prefix-prefill成本约定下部分case value趋近于零，表明提示主要起加速作用而非 steering。

## 相关工作脉络
- **算法信息论基础**：Kolmogorov复杂度[ Kol65, Cha66 ]与Levin Kt复杂度[ Lev73 ]，本文将其从通用图灵机适配到固定LLM。
- **概率Kolmogorov复杂度**：Goldberg等[pK_δ^t]将随机磁带上的复杂度取分位数，本文在LLM setting下的类比，但参考机为LLM而非UTM。
- **Xie et al. [XQY+26] 人贡献度量**：其"author-contribution"分数的分子本论文给出算法信息论基础，但本文额外显式计费思考/计算。
- **Halpern & Pass计算信息价值理论**：本文沿袭"信息价值在于节省计算"视角，但将其具体化到固定LLM的prompt评估。
- **PMI与prompt scoring**：无思考情形下本文score形式上等价于PMI；Sorensen等用Shannon MI选择prompt模板，本文度量单个prompt对单个artifact的价值。
- **LLM watermarking**：利用autoregressive采样与随机数耦合生成文本的技术，与本文的dyadic-program构造共享同一事实但目的不同。

## 局限性与未来方向
- **Artifact声明的主观性**：需预先指定产物z，否则可构造虚假高value（如z为随机串，p给出z）。论文建议采用canonical representation或verifier，但未实现。
- **语义重随机化未实证**：为消除恶意padding，提议用独立LLM重写z保持语义但变化表面形式，作为future work。
- **实验规模极小**：仅12个GSM8K样本，不足以代表泛化性能，论文明确定位为illustration。
- **仅处理单轮提示**：多轮对话的序贯求和不能刻画human adaptivity与formulation computation成本。
- **成本函数需外部声明**：κ的选择影响结果，论文提供两种约定但未讨论最优选择标准。

## 研究启发与可借鉴点
- **思考路径作为随机麻带的建模思路**：将LLM的thinking tokens视为程序的外部随机输入，分离"随机性"与"描述长度"，可用于分析CoT/ToT等推理过程的效率。
- **中位数/分位数聚合策略**：避免rare highly-successful rollout污染估计，这一稳健统计选择可迁移至其他LLM不确定性度量。
- **理论度量与实际cost的严格对应**：pKt与median token reproduction cost的指数关系为后续工作提供了可验证的经济解释框架。
- **与人类贡献评估的结合点**：本文框架可直接用于量化prompt engineer、human annotator、co-pilot在AI-assisted generation中的实际价值。
- **扩展至verifier-based artifact**：当存在可验证产物类时，prompt value退化为acceptance cost的对数比，为程序合成/定理证明场景提供简洁评估工具。

## 关键术语表
- **LLM-relative Kolmogorov复杂度**：以固定LLM及其采样过程为参考机的算法复杂度，用强制输出所需最短二进制随机数前缀长度度量。
- **先验复杂度(~K_M)**：产物在给定上下文的负对数概率(-log₂P)，与程序复杂度相差<2 bits。
- **Levin-Kt复杂度**：描述长度与log运行时间之和的最小值，平衡输出难度与计算代价。
- **实现的Levin复杂度(~Kt_M^κ)**：针对一条具体思考路径，在所有截断点取min{先验复杂度+log成本}。
- **概率Levin复杂度(pKt)**：对随机思考路径的~Kt值取δ-分位数（通常中位数），汇总随机性。
- **Prompt value**：无提示与有提示的pKt之差，b bits意味着无提示的典型token成本是有提示的2ᵇ倍。
- **Token等价成本函数(κ)**：将思考token数与解码token数按prefill/decode速率折算为统一成本单位。
- **Reproduction cost**：固定思考路径下独立重试直至成功重现产物的期望token开销。

## 可复现要素
- **数据集**：GSM8K（公开，可通过HuggingFace获取）
- **代码**：论文声明配套online notebook（链接在原文脚注9），包含完整reproduction code与plots
- **模型权重**：DeepSeek-R1-Distill-Qwen-1.5B（开源）
- **关键超参**：temperature=0.6，top-p=1，rollout数k=64，c_pre=1，c_dec=32（默认），成本敏感网格c_dec∈[8,256]
- **估算精度**：通过k次rollout，置信水平由Hoeffding界控制，误差参数ζ可调
