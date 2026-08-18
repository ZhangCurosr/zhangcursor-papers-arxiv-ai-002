---
title: "Tracing-Provenance-and-Detecting-Tampering-with-Complementar"
source: https://arxiv.org/pdf/2608.12713v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:25:57"
field: "可信赖大语言模型"
keywords: ["LLM watermarking", "provenance tracing", "tamper evidence", "generative watermark", "piggyback spoofing", "unbiased reweighting"]
innovations: ["通过鲁棒与脆弱双信号共嵌入同时实现溯源与篡改检测，解决单信号方案的内在冲突", "定义基于归一化文本的篡改证据概念，实现三元决策（完整/篡改/无水印）", "提出基于周期轮次分配的无偏 co-embedding 机制，以碰撞熵预算为旋钮调节信号强度比"]
benchmarks: ["C4 realnewslike", "LFQA"]
---

# 论文速读：Tracing-Provenance-and-Detecting-Tampering-with-Complementary-LLM-Watermarks

## 一句话总结
本文提出 Cocktail，一种面向大语言模型的生成式水印方法，通过在每 token 中共嵌入鲁棒信号与脆弱信号，同时实现溯源（provenance）与篡改检测（tamper evidence）。该方法有效抵御了 piggyback spoofing 攻击，在两项任务上均显著优于现有单信号水印方案。

## 研究问题与动机
1. **现有水印的溯源鲁棒性与篡改检测敏感性存在根本冲突**：为抵御编辑攻击，现有方案（如 KGW、Unigram、SIR）将信号设计为对编辑鲁棒，但这恰恰为 piggyback spoofing 攻击打开了后门——攻击者可篡改关键内容而保留高水印强度。
2. **piggyback spoofing 攻击威胁实际部署安全**：攻击者仅需修改少量词语即可改变文本语义或立场（如情感翻转），同时维持溯源归属，使水印沦为虚假内容的"背书"。
3. **既有最接近方案 Bileve 存在严重缺陷**：Bileve 对 token ID 而非读者可见文本签名，重分词导致 91.1% 的误报率；其 PPL 高达 62.2（是 Cocktail 的六倍）；检测需排列组合检验，耗时 144 秒。
4. **单信号方案无法兼顾两项任务**：信号越鲁棒，篡改敏感度越低；反之亦然。需从根本上解耦两项需求。

## 核心贡献（创新点）
1. **首次通过双信号共嵌入同时实现溯源与篡改检测**：在同一文本中分别嵌入鲁棒信号（短窗口）和脆弱信号（长窗口），将二元决策扩展为三元决策（完整 / 篡改 / 无水印）。与已有工作的本质区别在于不再追求单一信号的"全能"，而是通过互补解耦实现两项独立目标。
2. **定义基于归一化文本的篡改证据概念**：将水印锚定于读者可见的标准化文本而非中间 token ID，使篡改检测具有明确语义。与 Bileve 等签名方案的本质区别在于：本文检测的是"内容与原文本是否一致"，而非"token 序列是否精确匹配"。
3. **提出无偏 tournament 轮次周期分配机制**：通过多轮无偏重加权将两信号共嵌入每个 token，并以周期性模式控制信号强度比。与 BiMark/ENS 等独立无偏步骤的组合方案不同，本文的轮次分配将碰撞熵预算转化为可调的信号强度旋钮。
4. **实验验证超过 66 个百分点的篡改检测优势**：Cocktail 最强单信号基线（SynthID）的篡改检测率最高仅 23.1%，而 Cocktail 最低达到 89.5%，溯源性能保持竞争力且 PPL 无显著下降。

## 方法详解
Cocktail 由三个核心机制构成：

**1. 互补双信号设计**
- 鲁棒信号 $g_r$：采用短窗口 $h_r = 1$（前一个 token + 自身 token），对编辑不敏感，服务于溯源。
- 脆弱信号 $g_f$：采用长窗口 $n_f$（全归一化前缀 + 自身 token），对编辑极度敏感，服务于篡改检测。
- 两个信号使用独立密钥 $k_r \neq k_f$，各自生成独立的 green-red list：
$$g_{*}^{(t)} = \mathrm{PRNG}(k_{*}, s_{*}(t)) \in \{0,1\}^{|\mathcal{V}|}, \quad * \in \{r,f\}$$
- Seed 基于归一化文本：$s_r(t) = H(\mathcal{N}(x_{\max(1,t-h_r):t-1}))$，$s_f(t) = H(\mathrm{suffix}_{n_f}(\mathcal{N}(x_{1:t-1})))$。归一化操作包括：不可见字符清理、Unicode NFKC 标准化、同形字折叠（homoglyph folding）、大小写折叠、空白符压缩。

**2. 无偏 co-embedding（向量式 tournament 重加权）**
- 每步生成时依次应用 $d$ 轮重加权，每轮分配给鲁棒或脆弱信号：
$$\hat{p}^{(i)}(x) = (1 + g^{(i)}[x] - q^{(i)})\hat{p}^{(i-1)}(x), \quad q^{(i)} = \sum_{x \in \mathcal{V}} g^{(i)}[x]\hat{p}^{(i-1)}(x)$$
- 由于 $g^{(i)}[x] \sim \mathrm{Bernoulli}(0.5)$，每轮期望调整因子为 1，保持分布无偏。
- 通过周期性模式 $\pi$ 控制轮次分配：$\pi_i = \mathrm{FRAGILE}$ if $i \bmod(r+1) = 0$ else $\mathrm{ROBUST}$，对应强度比 $d_r:d_f \in \{1:1, 2:1, 4:1\}$。

**3. 二维评分空间与三元决策**
- 对检测文本计算两个 z-score：
$$z_{*} = \frac{\sum_{t=1}^{T}\sum_{i:\pi_i=*} g^{(i)}[x_t] - \frac{1}{2}Td_*}{\sqrt{Td_*/4}}, \quad * \in \{r,f\}$$
- 决策规则（先判溯源，再判篡改）：
$$\mathrm{Det}(x) = \begin{cases} \mathrm{Intact} & z_r > \tau_r \land z_f > \tau_f \\ \mathrm{Tampered} & z_r > \tau_r \land z_f \leq \tau_f \\ \mathrm{No\text{-}Watermark} & z_r \leq \tau_r \end{cases}$$
- 阈值 $\tau_r, \tau_f$ 均在负类尾部分布上校准至 1% FPR。

## 实验与结果
- **模型与数据集**：Llama-3.2-1B、Gemma-3-4B；C4 realnewslike、LFQA。每模型-数据集对生成 500 篇文本（每篇最多 512 token）。
- **基线方法**：KGW、Unigram、SynthID、SIR、Bileve。
- **攻击类型**：Paraphrase (Dipper)、Round-trip Translation (opus-mt)、Token Substitution、Sentiment Flip (gpt-oss:20b)、Homoglyph Substitution。
- **评估指标**：TPR@1%FPR（溯源 / 篡改检测）、PPL（生成质量）。
- **核心结果**：
  - **溯源性能**：Cocktail 4:1 在无攻击下 99.7–100.0%，Paraphrase 下最高 92.4%（C4/Llama），RT-Translation 下最高 97.4%，与最强单信号基线相当甚至更优。
  - **篡改检测性能**：Cocktail 所有变体在 1% FPR 下均达到 89.5–100.0%，而最强单信号基线最高仅 23.1%（Cocktail 超越幅度超过 66 个百分点）。
  - **质量保持**：PPL 在 7.5–10.6 范围内，与基线相当。
  - **轮次比调节**：从 1:1 到 4:1，溯源上升而篡改检测略降，但两者始终远超所有基线。
- **消融实验关键发现**：
  1. 归一化必须在 seeding 阶段进行，否则 homoglyph 攻击可将溯源降至 33.6%。
  2. Co-embedding 优于"每 token 只分配一个信号"的方案：后者在 T=50 时溯源从 98.3% 降至 86.0%。
  3. 脆弱窗口必须远长于鲁棒窗口，否则两信号退化至同一维度，无法分离篡改文本。

## 相关工作脉络
1. **KGW (Kirchenbauer et al., 2023)**：首个体量绿红列表水印，seed 基于前一 token。本文定位：KGW 是鲁棒端单信号代表，脆弱性不足是其天然缺陷。
2. **Unigram (Zhao et al., 2024)**：全局固定划分，最大化编辑鲁棒性。本文定位：鲁棒性最强但篡改检测几乎为零（0–17.2%），凸显单信号方案的局限性。
3. **SynthID (Dathathri et al., 2024)**：无偏 tournament 重加权的开创性工作。本文定位：Cocktail 直接继承其无偏重加权机制，但通过双信号扩展为三元决策。
4. **SIR (Liu et al., 2024)**：语义嵌入 seed，对 paraphrasing 鲁棒。本文定位：语义鲁棒性最高，但篡改检测为 0%，说明语义锚定牺牲了内容敏感性。
5. **Bileve (Zhou et al., 2024)**：最接近的双信号方案（粗粒度统计信号 + 精细数字签名）。本文定位：Bileve 签名基于 token ID 而非内容，存在 91.1% 误报率和 6 倍 PPL 代价，本文从根本上避免了这一问题。
6. **Dipper (Krishna et al., 2023)**：基于 paraphrasing 的检测器。本文定位：Dipper 是被动检测器（无生成水印），本文属于主动生成水印，两者解决不同层面的问题。

## 局限性与未来方向
- **尾部编辑的检测局限**：仅当编辑影响前缀 seed 时脆弱信号才响应，若编辑仅发生在文本末尾，脆弱信号可能失效。作者指出沿尾部分段读取 $z_f$ 是自然扩展方向。
- **鲁棒信号对 stealing 攻击的暴露**：短窗口鲁棒信号与现有鲁棒方案共享"易于通过查询推断 green list"的弱点；脆弱信号因 full-prefix seed 难以伪造，可作为补充防护。
- **未覆盖多模态场景**：作者明确提及"双互补信号配方可推广至其他需要溯源与篡改检测的生成模态"，但当前实验仅限文本。
- **轮次比非精确线性控制**：round ratio 是 strength ratio 的单调代理而非精确映射，因 collision entropy 消耗呈非线性。

## 研究启发与可借鉴点
1. **无偏重加权的可扩展性**：Vectorized tournament 的无偏性质可推广至多信号共嵌入场景，为未来多目标水印设计提供通用框架。
2. **三元决策空间的构建范式**：将两项冲突需求解耦为二维评分空间并通过优先级规则分类，这一范式可迁移至其他需要多状态判定的可信赖 AI 任务（如内容完整性验证）。
3. **归一化锚定的安全语义**：以读者可见的归一化文本而非内部 token ID 为锚点，确保水印语义与实际内容一致，这一设计原则对防伪造攻击具有普遍参考价值。
4. **碰撞熵预算的显式建模**：将多轮重加权的熵消耗显式建模为可分配预算，使得信号强度比可通过周期分配旋钮调节，为资源受限场景下的水印调优提供定量工具。
5. **与团队方向的结合机会**：若团队关注 AI 生成内容的版权追溯，Cocktail 的三元决策可直接用于内容发布审核流水线（区分"原生成 / 编辑后 / 非生成"）；若关注对抗防御，其归一化设计可增强现有水印对隐式攻击的鲁棒性。

## 关键术语表
**Piggyback Spoofing**：攻击者篡改水印文本的关键内容，利用信号对编辑的鲁棒性保留溯源归属，使水印沦为虚假内容的背书。

**Provenance（溯源）**：确认文本是否来源于特定水印 LLM 的能力，关注归属而非内容完整性。

**Tamper Evidence（篡改证据）**：检测已生成文本是否被后续编辑篡改的能力，关注内容保真度。

**Green-Red List**：将词表划分为绿红两组的 pseudorandom 标记，绿色 token 概率被提升、红色 token 概率被抑制，形成水印信号。

**Unbiased Reweighting（无偏重加权）**：调整采样分布但不改变期望分布的重加权操作，保证水印不降低生成质量。

**Collision Entropy（碰撞熵）**：衡量分布不确定性的 Renyi 熵，在 watermarking 中作为可分配的信号强度预算。

**Homoglyph Substitution（同形字替换）**：将字符替换为视觉上相似但编码不同的字符（如拉丁 a 替换为西里尔 а），属于不可见攻击。

**Three-State Decision（三元决策）**：将检测结果分类为 Intact / Tampered / No-Watermark 三种状态的判定规则。

## 可复现要素
- **数据集**：C4 realnewslike subset（公开）、LFQA（公开）。
- **模型**：Llama-3.2-1B、Gemma-3-4B（均为开源模型）。
- **代码**：论文未明确声明代码开源，但提供了完整 Algorithm 1/2、超参表（Table III）及归一化操作细节（Appendix B），复现可行性高。
- **关键超参**：$h_r=1$，$n_f=$ 全前缀，$d=30$，轮次比 $\{1:1, 2:1, 4:1\}$，temperature=1.0，top-k=100，阈值校准至 1% FPR。
- **硬件**：NVIDIA RTX 4090 Laptop GPU 16GB VRAM，Intel Core i9-14900HX CPU，64GB RAM。
