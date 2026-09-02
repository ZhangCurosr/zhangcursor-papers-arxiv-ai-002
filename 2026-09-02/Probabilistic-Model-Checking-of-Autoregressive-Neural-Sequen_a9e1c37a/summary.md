---
title: "Probabilistic-Model-Checking-of-Autoregressive-Neural-Sequen"
source: https://arxiv.org/pdf/2609.00838v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:27:30"
field: "神经模型形式化验证"
keywords: ["probabilistic model checking", "DTMC abstraction", "PCTL verification", "autoregressive transformers", "population coverage", "CEGAR refinement", "neural sequence verification"]
innovations: ["自回归模型的DTMC概率抽象与PCTL验证流水线", "保守soundness定理与CEGAR自适应区间收紧", "最大似然反例提取与best-of-N闭式评估"]
benchmarks: ["CAPP工艺规划（53-token GPT-2）", "SMILES分子生成（2700-token GPT-2）"]
---

# 论文速读：Probabilistic-Model-Checking-of-Autoregressive-Neural-Sequen

## 一句话总结
本文提出了一种将自回归Transformer模型抽象为离散时间马尔可夫链(DTMC)的概率模型检测方法，通过PRISM模型检查器验证PCTL规范，实现对模型输出概率分布的**有界验证**与**人口级覆盖率**量化；在CAPP工艺规划与SMILES分子生成两个案例中，揭示了准确率指标下不可见的风险分布与领域约束违反模式。

---

## 研究问题与动机
1. **准确率的盲区**：测试集准确率仅反映greedy解码在单一轨迹上的表现，无法量化模型在采样解码下实际分布的概率质量；
2. **领域约束不可见**：许多部署约束（如制造顺序、化学有效性）不在训练信号中，准确率无从报告；
3. **全量输入覆盖率未知**：模型对多少比例的输入满足特定安全/有效性需求，缺乏整体性保证；
4. **工程部署缺口**：现有验证工具（Reluplex、Marabou等）仅针对单前向传播，不覆盖多步生成过程的全分布分析。

---

## 核心贡献（创新点）
1. **DTMC抽象+PCTL验证+人口覆盖率三位一体的流水线**：将自回归模型展开为DTMC，使用PRISM进行精确的PCTL可达性验证，再将逐输入判定聚合为覆盖率曲线——这是首次将概率模型检测系统应用于自回归Transformer的全生成过程；
2. **Soundness定理与CEGAR自适应收紧**：证明DTMC是对SUT的**保守下界近似**，并通过CEGAR循环自适应重新展开被剪枝的转移以收紧验证区间，而非依赖模拟估计；
3. **最大似然反例提取算法**：在DTMC上使用Dijkstra（以−log p为边权）提取最有可能违反规范的序列轨迹，为部署工程师提供可解释的调试工件；
4. **跨域可移植性验证**：同一流水线无需代码修改即适用于词汇量差异50倍（53 vs. ≈2,700）的两个案例，证明其domain-agnostic特性。

---

## 方法详解
### 1. DTMC提取（Sec. 3.3）
- 对给定输入，按广度优先展开token生成树；
- **阈值剪枝**：条件概率 < τ 的分支被路由到吸收态 `low_prob`；累积概率 < ρ 的路径同样剪枝；
- **深度限制** $d_{max}$ 保证终止；未达终止的路径路由到 `truncated` 吸收态；
- **结构有效性剪枝**：按领域语法检查前缀合法性（CAPP案例禁止<EOS>前填充等），违规前缀路由到 `invalid`；
- 四个吸收态：`success`, `low_prob`, `invalid`, `truncated`，满足质量守恒（Corollary 2）。

### 2. PCTL规范验证（Sec. 3.4）
- 导出DTMC至PRISM建模语言，将吸收态与关键状态标记为布尔命题；
- 典型查询：$P_{=?}[F \; \text{label}]$（最终达到某标签的精确概率）；
- 通用标签：success / low_prob / invalid / truncated / critical；
- 领域标签：CAPP用`ordered`/`misordered`（编码主工序→次工序→精加工顺序）；SMILES用`valid_smiles`（RDKit外部化学验证）。

### 3. 人口覆盖率（Sec. 3.5）
- 固定阈值 θ，统计满足 $P_D(\phi_i) \geq \theta$ 的输入比例 $\hat{\mu}(\theta)$；
- 扫描 θ ∈ [0,1] 得到覆盖率曲线；
- 由Lemma 1，曲线是真实SUT覆盖率的保守下界。

### 4. Soundness定理（Sec. 3.6）
**定理1**：对任意仅在success终端成立的可达性质 $\phi$：
$$
P_D(\phi) \leq P_M(\phi) \leq P_D(\phi) + P_D(\text{LOW\_PROB}) + P_D(\text{TRUNCATED})
$$
（`invalid`项因前缀闭包性质可排除于上界外）

### 5. CEGAR循环（Sec. 3.7）
- 每轮按影响度 $Pr[\text{reach } s] \cdot p_{prune}(s,t)$ 排序被剪枝边；
- 重新展开top-K边，拼接子树，重跑PRISM；
- 终止条件：$P_D(\text{LOW\_PROB}) \leq \varepsilon_{target}$ 或预算耗尽。

### 6. 反例提取（Sec. 3.8）
- 对违规属性，以 $- \log p$ 为边权运行Dijkstra，得到最大似然违规轨迹。

---

## 实验与结果
### 案例1：CAPP（工艺规划）
- **SUT**：Stathatos et al.的4层GPT-2，53-token词汇表，6特征离散零件描述→工序序列；
- **8个训练分数**（1%–70%），1,176–6,586测试样本；
- **准确率指标**：从15%起序列准确率达100%；
- **验证发现**：
  - 70%训练模型：$P(\text{success}) \in [0.974, 1.000]$（Theorem 1），0.026概率质量在`low_prob`吸收态；
  - 1%训练模型：$P(\text{success}) = 0.128$，但greedy准确率仍达47.8%；
  - 从30%训练起，$P(\text{ordered}) \geq 0.90$ 的覆盖率达到 $\hat{\mu}=0.971$；
  - 温度扫描：$T \leq 1.0$ 时覆盖稳定（$\hat{\mu} \geq 0.97$），$T=1.3$ 骤降；
  - Best-of-N分析：1%模型在 $T=0.5$ 时pass@10(ORDERED)=1.00，而greedy仅为0.47；
  - 反例：最可能违规轨迹为「首操作直接发出精加工(token)而无主工序」，联合概率 $3.5 \times 10^{-3}$。

### 案例2：SMILES分子生成
- **SUT**：gpt2_zinc_87m，87M参数，≈2,700 BPE词表（50×CAPP）；
- **200个prompt**，DTMC平均13,150状态（最大37,397）；
- **领域规范**：RDKit化学有效性oracle（`MolFromSmiles`）；
- **关键发现**：
  - 90个至少产生一个结构完整终端的prompt中，$P(\text{valid\_smiles})=0.084$ vs $P(\text{success})=0.248$，**66%概率质量对应无效分子**；
  - 110个prompt从未生成化学有效SMILES；
  - 证明了「结构完整性≠化学有效性」的gap，而这是准确率完全不可见的。

### 性能
- CAPP：所有37,554个DTMC、262,878次PRISM检查零失败；
- SMILES：200次检查零失败，平均提取485s + 验证43s/prompt（Apple M2 Max）。

---

## 相关工作脉络
1. **MOSAIC/Gross et al.**：将RL策略抽象为MDP并用PRISM验证PCTL——本文与之不同，目标自回归生成的**联合分布DTMC**而非环境随机性下的策略；
2. **UPPAAL SMC/Plasma**：基于模拟的统计验证——本文提供**精确的后向归纳**可达概率，而非采样估计；
3. **Reluplex/Marabou/ERAN/α-β-CROWN**：确定性前馈网络验证器——本文处理**多步解码的生成过程**，需要概率形式化；
4. **Conformal Prediction**：模型无关的人口级保证——本文访问模型token概率，支持**领域特定PCTL规范**；
5. **DeepXplore/DeepGauge**：激活空间覆盖率测试——本文覆盖率基于**概率边界**而非激活覆盖，且有形式化soundness保证。

---

## 局限性与未来方向
- **当前仅支持GPT架构**：未扩展至Long-context（$d \gg 20$）、非Transformer架构或多样化采样策略；
- **SMILES案例输入仅为200个prompt**：不具备随机代表性，人口覆盖率结论受限；
- **词汇量可扩展性挑战**：SMILES词汇量50倍增长导致状态空间扩大~800×，全量ZINC扫描仍不可行；
- **领域规范依赖人工设计**：ordering规范仅编码了相位顺序，未涵盖资源竞争、换刀成本等更丰富约束；
- **未来方向**：更长历史生成、更丰富的属性类、超越温度缩放的采样策略。

---

## 研究启发与可借鉴点
1. **将准确率指标的"噪声"显式建模为概率质量分布**：本文证明即使是100%准确率的模型，仍存在显著的sub-threshold概率分散——这对模型选择与部署决策极具参考价值；
2. **CEGAR循环适配DTMC场景**：将抽象细化思想应用于概率模型检查中的阈值剪枝策略，是一种可迁移的工程技巧；
3. **Best-of-N的闭式评估**：无需额外模型调用即可在DTMC上传播终端概率计算pass@N，为解码策略选择提供形式化依据；
4. **外部oracle无缝集成**：RDKit作为black-box pass/fail预言机直接作为PCTL标签——证明pipeline对领域规范的**即插即用**能力；
5. **覆盖率曲线的保守性语义**：定理保证低估而非高估，使报告结果可直接作为安全部署的决策依据。

---

## 关键术语表
- **DTMC（Discrete-Time Markov Chain）**：离散时间马尔可夫链，本文用于抽象自回归模型的token-by-token生成过程；
- **PCTL（Probabilistic Computation Tree Logic）**：概率计算树逻辑，用于表达带有概率量词的时序属性，如 $P_{\geq \theta}[F \; \phi]$；
- **CEGAR（Counterexample-Guided Abstraction Refinement）**：反例引导的抽象细化，本文用于自适应重新展开被剪枝的高影响转移以收紧验证区间；
- **SUT（System Under Test）**：被测系统，本文中指自回归Transformer模型本身；
- **Coverage Curve $\hat{\mu}(\theta)$**：覆盖率曲线，横轴为概率阈值θ，纵轴为满足该阈值的输入比例，刻画属性在输入空间的满足强度；
- **Sound Under-approximation**：保守下界近似，指DTMC的可达概率永远不超过真实模型但提供安全下界；
- **Critical State**：关键状态，指top-1/top-2概率差距<10%的节点，标记为诊断性PRISM标签；
- **Pass@N**：从N次采样中至少有一次通过验证的概率，本文在DTMC上闭式计算而不需额外模型运行。

---

## 可复现要素
- **数据集**：CAPP使用Stathatos et al.开源的8个训练分数checkpoint（1%–70%）；SMILES使用ZINC数据库的200个prefix字符串；
- **代码**：论文未明确声明开源仓库，但提到PRISM 4.10 + Apple M2 Max环境；
- **关键超参**：阈值 τ = 0.5%（可扫），累积底限 ρ = 10⁻⁴，最大深度 $d_{max}$ = 20，临界状态概率差距阈值 10%；
- **CEGAR参数**：top-K = 5，ε_target = 10⁻³；
- **温度扫描**：T ∈ {0.5, 0.7, 1.0, 1.3, 2.0}；
- **PRISM配置**：显式状态引擎，4-way并行。

---
