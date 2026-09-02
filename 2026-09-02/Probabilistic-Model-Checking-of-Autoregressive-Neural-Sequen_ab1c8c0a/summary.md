---
title: "Probabilistic-Model-Checking-of-Autoregressive-Neural-Sequen"
source: https://arxiv.org/pdf/2609.00838v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:13:38"
field: "神经语言模型形式化验证"
keywords: ["probabilistic model checking", "DTMC abstraction", "PCTL", "autoregressive transformers", "neural network verification", "population coverage", "CEGAR"]
innovations: ["将自回归多步生成过程抽象为可信 DTMC 并提供下近似证书", "在 PRISM 上实现逐输入 PCTL 验证并聚合为人口覆盖率曲线", "CEGAR 自适应精炼与最大似然反例提取的联合管线"]
benchmarks: ["CAPP 工艺规划（7,840 零件，1%-70% 训练比例）", "SMILES 分子生成（gpt2_zinc_87m，200 提示）"]
---

# 论文速读：Probabilistic-Model-Checking-of-Autoregressive-Neural-Sequen

## 一句话总结
论文提出了一套面向自回归Transformer的**概率模型检测流水线**，通过逐 token 展开生成过程提取离散时间马尔可夫链（DTMC），利用 PRISM 进行 PCTL 形式化验证，并将逐输入的 verdict 聚合为面向全输入空间的保守覆盖率曲线；两个案例（CAPP 工艺规划、SMILES 分子生成）证明该方法可暴露测试集准确率无法反映的隐藏概率质量与领域约束违反风险。

## 研究问题与动机
- **测试集准确率的盲区**：部署自回归模型时，准确率只打分单次贪心轨迹与参考序列的匹配度，无法回答模型把多少概率质量分配到"采样解码才会走到但贪心永远选不到"的可行替代路径上。
- **领域约束缺失覆盖**：许多领域约束（如制造阶段顺序、分子化学价态）不在训练信号中直接出现，准确率既无法报告失败原因，也难以衡量这些约束在整个输入人口上的满足比例。
- **决策温度与解码策略缺乏形式化边界**：贪心与采样在不同温度 T 下产生的成功/违规概率差异无法被现有指标刻画，部署工程师难以确定"安全操作包络"。
- **现有验证工具的语义错位**：确定性 DNN 验证器（Reluplex/Marabou 等）只处理单次前向传播，统计模型检查（UPPAAL SMC/Plasma）给出估计而非精确界；缺乏一种能把自回归多步生成过程抽象为可精确验证的概率模型，并输出人口级保证的方法。

## 核心贡献（创新点）
1. **一套面向自回归 Transformer 的概率模型检测流水线**：将模型视作 SUT，依次完成 DTMC 抽象 → PCTL 验证 → 人口覆盖率聚合；与以往仅验证单一前向传递或策略 MDP 的工作不同，本文验证的是**整个多步生成联合分布**。
2. **DTMC 的形而上确界定理（Theorem 1）**：证明提取的 DTMC 是 SUT 的**可信下近似**，每条显式路径保留原始概率，被剪入吸收汇的质量上界给出严格不等式；与仅做数值估计的 SMC 方法相比，这里给出的是**可证书的下界/上界区间**。
3. **CEGAR 自适应精炼循环**：按影响分数 $Pr[\text{reach } s] \cdot p_{\text{prune}}(s,t)$ 对剪枝转换排序，逐轮重新展开 top-K 并 splice 子树，直到 $P_D(\text{low\_prob}) \le \varepsilon_{\text{target}}$；这是将经典 CEGAR 思想首次适配到 DTMC 抽象 + PRISM 场景。
4. **最大似然反例提取算法**：在 DTMC 上以 $-\log p$ 为边权跑 Dijkstra，返回最可能违反规格的 token 序列及其形式概率；相比纯数值 verdict，这给出了**可直接供工程师审阅的可解释反例**。
5. **双案例验证与可移植性论证**：在 53 token 词汇的 CAPP 与 ≈2700 token 词汇的 SMILES 分子生成上均跑通 PRISM，且 SMILES 仅靠更换外部化学有效性预言机即可工作，证明方法对词汇规模和预言机类型**不敏感**。

## 方法详解
- **DTMC 抽象**：给定输入，按 BFS 展开每个非终止状态 s；设阈值 $\tau$（默认 0.5%），条件概率 $\ge \tau$ 的 token 获得显式后继，其余概率质量（未重缩放）路由到吸收态 `low_prob`；累计路径概率低于 $\rho$（默认 $10^{-4}$）同样路由到 `low_prob`；达到最大深度 $d_{\max}$（默认 20）则路由到 `truncated`；结构语法规则违反路由到 `invalid`；所有吸收汇保证终止。
- **关键公式**：对任意只在 `success` 终端成立的可达性质 $\phi$，Theorem 1 给出
  $$P_D(\phi) \le P_M(\phi) \le P_D(\phi) + P_D(\text{low\_prob}) + P_D(\text{invalid}) + P_D(\text{truncated})$$
  由于语法的前缀闭合性，`invalid` 路径不可到达 $\phi$-满足终端，Corollary 1 简化上界为
  $$P_M(\phi) \le P_D(\phi) + P_D(\text{low\_prob}) + P_D(\text{truncated})$$
  四汇质量之和为 1（Corollary 2）。
- **PCTL 导出与查询**：将四个吸收汇及 `critical`（top-1/top-2 概率 gap $<10\%$）编码为 PRISM 布尔标号；通用查询包括 $P{=?}[\mathsf{F}\ \text{success/low\_prob/invalid/truncated/critical}]$，领域查询例如 CAPP 的 `ordered`/`misordered`（要求 $P\to S^*\to F^*$），SMILES 用 `valid_smiles` 替换。
- **人口覆盖率**：对阈值 $\theta$ 统计 $\hat{\mu}(\theta)=\frac{1}{N}\sum_i \mathbf{1}[P_D(\phi_i)\ge\theta]$， sweeping $\theta\in[0,1]$ 得覆盖曲线；Lemma 1 保证 $\hat{\mu}(\theta)\le\hat{\mu}_M(\theta)$，即报告值**保守低估**真实人口覆盖。
- **CEGAR 循环**：每轮按 $Pr[\text{reach } s]\cdot p_{\text{prune}}(s,t)$ 排序剪枝对，重新展开 top-K，splicing 新子树后重跑 PRISM；目标停机条件为 $P_D(\text{low\_prob})\le\varepsilon_{\text{target}}$。
- **反例提取**：在 DTMC 上用 Dijkstra（边权 $-\log p$）找最大联合概率路径，解码回 token 序列；给出具体违反轨迹及其形式概率。

## 实验与结果
- **数据集**：CAPP（7,840 种完全可枚举的零件描述，8 个训练比例 1%–70%，固定 1,176 测试集）；SMILES（200 个 ZINC 短前缀提示，87M 参数 GPT-2，≈2,700 BPE token）。
- **主要数字（CAPP）**：从 15% 训练起测试集序列准确率达 100%，但 $P(\text{success})$ 仅从 0.907（15%）上升到 0.974（70%）；70% 模型的 cert 区间为 $[0.974, 1.000]$。结构化无效 $P(\text{invalid})=0$（仅 1% 训练时为 0.009）。人口覆盖率 $\hat{\mu}(0.90)$ 在 30% 训练时达到 0.971，50% 起满额 1.0。
- **解码温度包络**：30% 与 70% 模型在 $T\le 1.0$ 保持 $\hat{\mu}\ge 0.97$，$T=1.3$ 骤降；1% 模型即使在 $T=0.5$ 也仅达 0.789，推荐在数据饥饿场景使用 sharpened sampler（$T<1.0$）+ verifier 的 best-of-N 解码。
- **Best-of-N**：1% 模型在 $T=0.5$、N=10 时 pass@10(ordered)=1.00，greedy 仅为 0.47；distinct valid plans 为 1.87/输入。
- **CEGAR 效果**：5 轮后低训练比例模型的中位数 $P_D(\text{low\_prob})$ 改善有限（1% 仅从 0.929 降到 0.919），但中高位训练比例改善 25–35%；说明粗数据饥饿下的剩余质量是"不可约分散"，CEGAR 难以回收。
- **主要数字（SMILES）**：DTMC 平均 13,150 状态（最大 37,397），约为 70% CAPP 的 800 倍；200 次 PRISM 调用零求解失败。在 90 个至少产出一条化学合法终态的提示中，$P(\text{valid\_smiles})=0.084$，$P(\text{success})=0.248$，结构完整但化学无效的"语义空白质量"占比约 66%。
- **最强结果**：CAPP 70% 训练配置下 $P(\text{success})=0.974$（cert 下界）、ordering 覆盖率 $\hat{\mu}=1.0$；SMILES 在 800× 更大 DTMC 上仍 0% 求解失败，验证了端口可移植性。

## 相关工作脉络
- **MOSAIC / Gross et al. (MDP 抽象)**：面向 RL 策略或 LLM 有界输出的 MDP/interval-MDP 抽象与 PCTL 验证；本文目标改为**自回归生成联合分布的 DTMC**，并新增人口聚合层，与策略在环境随机性下的验证形成语义差异。
- **UPPAAL SMC / Plasma（统计模型检查）**：通过采样估计概率；本文在 DTMC 上用 PRISM 做**精确反向归纳**，聚合层只做 per-input verdict 汇总，精度与可证书属性优于 SMC。
- **Reluplex / Marabou / ERAN / α-β-CROWN（确定性 DNN 验证）**：针对单次前向传播的 feed-forward 网络；本文扩展至**多步解码生成过程**，必须使用概率形式化而非单纯 SMT/区间传播。
- **Weiss et al.（L*-style 从 RNN 提取有限自动机）**：依赖等价查询；本文以**直接前向展开 + 阈值剪枝**构建概率代理，显式用 Theorem 1 界定抽象误差，不依赖交互式查询。
- **Conformal prediction（模型无关人口级保证）**：不访问模型内部；本文直接读取 token 概率并允许嵌入领域专属 PCTL 规格（如阶段顺序），提供更强的结构语义。
- **DeepXplore / DeepGauge（覆盖导向测试）**：面向分类器激活空间；本文的覆盖曲线是对应概念的形式化版本，但带**可信下界保证**，可用于部署决策。

## 局限性与未来方向
- **深度/长度限制**：$d_{\max}=20$ 对长序列生成不足以覆盖，截断项会显著加宽 cert 区间；SMILES 的 $P(\text{truncated})$ 已在平均 0.018 级别。
- **非 GPT 架构与多样化采样**：目前仅在 GPT-2 风格 decoder 上验证，未覆盖 encoder-decoder、混合解码（nucleus/beam/sampling 组合）或长程 horizon 场景。
- **输入分布代表性假设**：CAPP 输入空间完全可枚举，覆盖率等于枚举比例；SMILES 的 200 提示并非来自任何部署分布的随机样本，其人口覆盖结论仅对"提示集合"成立。
- **计算可扩展性**：200 个 SMILES 提示已耗时明显（单提示平均抽取 485s、PRISM 43s），全量 ZINC 尺度不可行；需依赖稀疏采样或近似。
- **CAPP 领域规格偏简化**：仅编码 $P\to S^*\to F^*$ 阶段顺序，未包含资源冲突、刀具切换成本、批量相关工艺选择等 richer 制造约束。

## 研究启发与可借鉴点
- **外部可执行预言机即插即用**：SMILES 案例中仅用一个 RDKit `MolFromSmiles` 检查即可替换领域标签，无需改动抽取/导出/聚合代码；后续团队可为任何"终态 pass/fail 检查器"快速构造 PCTL 规格。
- **保守性作为部署信任基础**：Theorem 1 + Lemma 1 保证所有报告值均为下界，"低估"意味着部署决策不会因抽象误差而过度乐观；这对工业部署的合规报告很有价值。
- **CEGAR 自适应 refine 策略**：按 $Pr[\text{reach } s]\cdot p_{\text{prune}}(s,t)$ 排序剪枝对，比均匀展开更高效；该启发可迁移到其他需精细化概率抽象的场景（如 RNN/Transformer 策略验证）。
- **best-of-N 闭式评分**：通过 $\mathbb{E}[1-(1-P_x(\phi))^N]$ 可在不做额外前向传播的情况下估算 verifier+采样器的成功率，辅助解码器选型与温度选择。
- **关键状态诊断指标**：top-1/top-2 gap $<10\%$ 的 critical 标号可帮助快速识别模型处于"决策模糊"区域的输入与位置，作为早停/主动学习的触发信号。

## 关键术语表
- **DTMC（离散时间马尔可夫链）**：由状态集合、初始状态、转移概率矩阵与标号函数组成，本文将其用作自回归模型逐 token 生成的概率代理。
- **PCTL（概率计算树逻辑）**：扩展 CTL 引入概率算子 $\mathsf{P}_{\bowtie p}[\mathsf{F}\ \psi]$，用于表达"从某状态出发最终到达标号状态的概率满足阈值"这类性质。
- **CEGAR（反例引导的抽象细化）**：先用粗粒抽象验证，若发现反例则定位关键剪枝点并重新展开，迭代收敛到满足精度目标的形式。
- **覆盖曲线 $\hat{\mu}(\theta)$**：对全输入集合统计满足 $P_D(\phi)\ge\theta$ 的比例， sweeping $\theta$ 得到的一条表征人口级可靠性的曲线。
- **SUT（System Under Test）**：本文中被验证的自回归神经网络序列生成模型。
- **吸收汇（success/low_prob/invalid/truncated）**：DTMC 终止状态；success 对应语法完备且满足域属性的终态，其余三个分别对应低概率被剪枝、结构语法违规、达到最大深度的路径。
- **预言机（Oracle）**：黑盒判断器，对终态序列返回 pass/fail；文中 CAPP 用阶段顺序谓词，SMILES 用 RDKit 化学有效性检查。
- **pass@N**：对单输入重复采样 N 次并保留 verifier 接受的首个结果的通过率，本文用闭式 $\mathbb{E}[1-(1-P_x(\phi))^N]$ 估算。

## 可复现要素
- **数据集**：CAPP 来自 Stathatos et al. [27] 开源数据（7,840 零件 × 8 训练比例）；SMILES 使用 ZINC 数据库上的 gpt2_zinc_87m 预训练模型。
- **代码/权重**：论文声明 SUT 作为黑盒使用、未改动架构与训练；PRISM 4.10 开源。具体抽取/CEGAR/聚合代码在 arXiv 版本未明确给出仓库链接，标注"论文未提及开源仓库"；外部工具 PRISM 与 RDKit 开源可复。
- **关键超参**：$\tau=0.5\%$（单 token 阈值）、$\rho=10^{-4}$（累积下限）、$d_{\max}=20$（最大深度）、`critical` gap 阈值 10%、CEGAR top-K=5、$\varepsilon_{\text{target}}=10^{-3}$。
- **硬件**：Apple M2 Max（32 GB RAM），PRISM 显式状态引擎 4 路并行。
