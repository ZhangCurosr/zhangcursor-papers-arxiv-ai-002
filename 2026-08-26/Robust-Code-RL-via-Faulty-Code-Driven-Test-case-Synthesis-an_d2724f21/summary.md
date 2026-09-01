---
title: "Robust-Code-RL-via-Faulty-Code-Driven-Test-case-Synthesis-an"
source: https://arxiv.org/pdf/2608.24135v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:02:12"
field: "大模型代码生成与强化学习"
keywords: ["RLVR", "code generation", "test case synthesis", "dense reward", "faulty code", "LLM"]
innovations: ["有缺陷代码驱动的定向测试用例合成流程（含近正确代码筛选与行为聚类选择）", "基于通过率的步进式密集奖励函数以缓解合成测试的噪声与假阴性", "提出 TSP 指标量化测试集诊断多样性并建立其与 RL 训练效果的关联"]
benchmarks: ["LiveCodeBench", "CodeForces"]
---

# 论文速读：Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping

## 一句话总结
本文提出 **RobustTests** 框架，通过"有缺陷代码驱动"的测试用例合成策略配合基于通过率的分层密集奖励函数，解决 RLVR 代码生成中测试覆盖率不足导致的奖励欺骗和策略退化问题；在 Qwen3-32B 上实验，较基线方法在 LiveCodeBench 取得 **+3% 绝对提升**。

## 研究问题与动机
- **RLVR 对测试覆盖率高度敏感**：现有 RL 微调依赖可验证奖励（exec 正确性判断），但测试用例覆盖不足时，错误代码可能通过少量测试（false positive），导致 reward hacking 和策略退化。
- **直接 LLM 合成测试存在幻觉与边界遗漏**：Naive LLM Generation 等方法缺乏对边界条件的覆盖，且容易生成违反问题约束的幻觉用例。
- **Generator Program 范式依赖人工标注，扩展性受限**：HardTests 等虽降低幻觉率，但需人工设计生成程序。
- **Validator 残存幻觉引入 false negative**：CodeContests+ 等引入 validator agent，但自动化 validator 无法保证完备性；二元稀疏奖励（0/1）下，被错误判为失败的"正确代码"会产生误导性梯度信号。
- **根本瓶颈**：单纯提升数据质量已触及天花板，需同时优化数据多样性 + 设计对噪声容忍的训练机制。

## 核心贡献（创新点）
1. **有缺陷代码驱动的定向测试合成流程**：先生成"近正确"故障代码池，再用其作为负样本引导 LLM 生成能触发特定逻辑缺陷的判别性测试用例；与直接生成或纯 generator 方法的本质区别在于以"故障语义边界"而非问题描述本身为驱动信号。
2. **三步过滤 + 行为聚类选优管道**：输入 validator → LLM 指令合规校验 → K-means 聚类 + round-robin 选代表性 medoids；区别于 CodeContests+ 仅做输入层面验证，本文确保测试用例具备跨多种故障模式的诊断覆盖。
3. **基于通过率的步进式密集奖励函数**：取代 0/1 稀疏奖励，引入 $r(s) = \frac{1}{10}\cdot \frac{|\text{pass}(s,T_{\text{synth.}})|}{|T_{\text{synth.}}|}$ 机制，允许模型从"部分正确"信号中学习；与 DeepSeek-R1 等沿用二元奖励的方法在噪声鲁棒性上有本质差异。
4. **提出 TSP（Test case Space Polarization）指标**：以香农熵形式量化测试集对不同故障代码的诊断覆盖多样性，可作为 RL 训练前低成本预估数据集潜力的代理指标。

## 方法详解
**整体框架**（图2）分三阶段：自动化测试用例生成 → 验证与选择 → 密集奖励设计。

### 3.1 有缺陷代码驱动的测试用例生成
1. **Faulty Code Generation**：对每道题 $P_i$，用 LLM 采样生成随机故障代码 $f \sim \text{LLM}(\cdot|P_i)$；对每个候选用原始测试集 $T_{\text{base}}$ 执行，得到二进执行向量 $V_i^f \in \{0,1\}^{|T_{\text{base}}|}$；仅保留满足 $0 < \frac{\|V_i^f\|_1}{|T_{\text{base}}|} < 1$ 的"部分通过/部分失败"代码（过滤掉完全正确和完全错误）。按执行向量去重，构建故障代码池 $F_{\text{base}}$。
2. **Faulty-Code-Driven Test Case Generation**：将一批 $f \in F_{\text{base}}$ 作为负样本嵌入 prompt，让 LLM 生成同时满足（i）符合输入规格；（ii）至少能触发一个故障代码的逻辑错误的测试用例 $t_i$。与 $T_{\text{base}}$ 合并为候选池 $T_{\text{cand}}$。

### 3.2 测试用例验证与选择（三步管道）
1. **Input Validation**：为每道题 $P_i$ 部署可执行 validator $A_i$，输出合法集合 $T_{\text{valid}} = \{t \in T_{\text{cand}} | A_i(t) = \text{True}\}$，剔除语义非法用例。
2. **LLM Instruction-compliance Validation**：用参考代码重新计算 ground-truth 输出，验证输入-输出对齐；并检查每个用例是否能对至少一个 $f \in F_{\text{base}}$ 触发失败，得到 $T_{\text{final}}$。
3. **Diversity-Driven Selection**：对每个 $t \in T_{\text{final}}$ 构建二进制执行向量 $V_i^t \in \{0,1\}^{|F_{\text{base}}|}$，用 K-means 聚类，以 Euclidean distance 为度量子优化目标 $\min \sum_k \sum_{V_i^t \in C_k} \|V_i^t - \pmb{\mu}_k\|_2^2$；对每个簇 round-robin 选取 medoids，得到最终合成测试集 $T_{\text{synth.}}$。

### 3.3 密集奖励函数设计
定义通过集合 $\text{pass}(s, T_{\text{synth.}}) = \{t \in T_{\text{synth.}} | \text{exe}(s,t)=\text{pass}\}$，奖励函数：
- 通过率=1 → 奖励 +1.1（鼓励完美解）
- 通过率=0 → 惩罚 -0.1（抑制全错解）
- 其他 → $r(s) = \frac{1}{10} \cdot \frac{|\text{pass}(s, T_{\text{synth.}})|}{|T_{\text{synth.}}|}$（渐进反馈）

该设计使部分正确的解答仍获得正向梯度，缓解 validator 残存幻觉导致的 false negative 影响。

### TSP 指标
$$TSP = \frac{1}{M} \cdot \sum_{s \in S} \left(-\frac{n_s}{N} \cdot \log_2 \frac{n_s}{N}\right)$$
其中 $M=|F_{\text{base}}|$，$N=|T_{\text{synth.}}|$，$n_s$ 为测试用例 $s$ 的覆盖频率。

## 实验与结果
- **数据集**：CodeContests+（11,636 题），经 Qwen3-32B 筛选出 pass@10 在 0.2~0.9 区间的中等难度子集，约 **3,300 题**（$\text{CodeContests}_{\text{train}}^+$）。合成测试集每题约 200 条高多样性用例（原文 4.1 节；正文摘要称约 40 条用于最终评估对比）。
- **基线**：Naive LLM Generation、HardTests、CodeContests+、CodeContests-O（均为 40 条/题上限，使用 0/1 稀疏奖励）。
- **评估基准**：LiveCodeBench（2024.08–2025.01）、CodeForces（Score / Rating / Percentile）。
- **训练配置**：Qwen3-32B + GRPO，$\beta_{\text{clip}}=0.2$，Adam（$\eta=5\times10^{-6}$，$\beta_1{=}0.9$，$\beta_2{=}0.95$），batch=256，10-step warmup；每 prompt 采样 8 次，$T{=}1.0$，$\lambda_{\text{KL}}{=}0.0$。
- **主结果**（Table 1）：
  - **LiveCodeBench**：RobustTests **68.39**（最高），较 CodeContests+（65.41）/HardTests（65.25）等基线均高出约 **3% 绝对**；CodeForces Score 38.50 / Rating 85.99 / Percentile 94.67，全面领先。
  - TSP 提升：RobustTests 相对原始 CodeContests+ 测试集提升 **7%**。
- **消融**（Table 2-3）：
  - 去除"测试用例合成"模块 → LiveCodeBench 从 68.39 降至 66.91（-1.48）；
  - 去除"多样性驱动选择" → 降至 66.82（-1.57）；
  - 去除两模块 → 降至 65.41（相当于 CodeContests+ 水平）；
  - 密集奖励 vs 稀疏奖励：在 RobustTests* 上带来 **+0.55**（67.84→68.39）的 LiveCodeBench 增益；在更噪声的无 validator 变体上增益更大（+1.19）。
  - 奖励尺度 0.05–0.20 范围内稳定，说明超参鲁棒。

## 相关工作脉络
1. **Naive LLM Generation（Li & Yuan, 2024）**：直接由 LLM 基于题目描述生成测试用例。本文与之区别在于：不依赖题目文本描述，而是利用"故障代码"作为定向引导，显著降低边界遗漏与幻觉率。
2. **HardTests（He et al., 2025）**：LLM 生成 generator 程序自动合成输入。本文差异在于引入故障语义维度与 validator 双阶段过滤，并进一步用聚类保证诊断多样性。
3. **CodeContests+（Wang et al., 2025）**：generator-validator 多智能体框架。本文承认其减少幻觉的贡献，但指出 validator 残余幻觉导致的 false negative 问题，并通过密集奖励与多样性选择加以缓解。
4. **CodeContests-O（Cai et al., 2026）**：闭环反馈迭代生成。本文定位差异在于：不只追求测试保真度，更强调测试的诊断覆盖多样性（TSP 可度量）及训练阶段对残留噪声的鲁棒性。
5. **DeepSeek-R1（Guo et al., 2025）**：GRPO SOTA，但在合成测试场景下二元奖励易受噪声干扰。本文将其扩展为密集奖励机制以适配噪声数据。
6. **CodeRL（Le et al., 2022）、AlphaCode（Li et al., 2022）**：早期编译/单元测试反馈的 RL 代码生成工作。本文继承"可执行反馈"思想，重点解决合成测试场景下的覆盖与噪声问题。

## 局限性与未来方向
- **依赖 ground-truth 参考解**：方法适用于有标准答案的编程题（如竞赛编程），在缺少 reference implementation 的真实工程场景中受限。
- **领域泛化未充分验证**：目前仅在 LiveCodeBench 与 CodeForces 评测，未扩展至更广泛的软件工程任务（如 bug fix、代码重构等）。
- **故障代码生成依赖 LLM 采样多样性**：故障语义覆盖可能受限于采样次数与 prompt 设计，极端复杂缺陷模式可能存在盲区。
- **Future direction（论文自述）**：（1）将合成策略推广至无参考解场景（如 mutation testing）；（2）验证 dense reward 在科学发现等其他可验证任务中的迁移能力。

## 研究启发与可借鉴点
1. **"近正确"代码作为负样本驱动测试合成的思路可迁移**：在数学推理、程序合成、自动测试生成等领域，均可借鉴"先构造部分正确的反例，再针对性生成覆盖其语义边界的验证用例"的范式。
2. **TSP（香农熵形式）作为数据质量代理指标的简洁性与可计算性**：可直接复用于任何"数据-标签"映射任务（如代码生成、程序综合）的数据集预筛，在大规模 RL 训练前低成本评估数据潜力。
3. **密集奖励函数的阶梯设计（满分+1.1、全错-0.1、部分通过线性缩放）对噪声数据的普适价值**：可推广至任何具有部分可验证信号的训练场景（如多步推理、工具调用规划），避免二元信号导致的梯度爆炸/消失。
4. **执行向量 + K-means 聚类选 medoid 的多样性选择策略**：可复用于其他需要平衡"诊断覆盖"与"数据规模预算"的场景，例如安全红队测试用例生成、异常检测数据集构建。
5. **与团队方向结合机会**：若团队涉及测试生成、代码合成 RL、或大模型代码能力提升，可将本框架与自有基座模型（如 Qwen3、DeepSeek-R1 系列）结合，重点测试其在更长上下文 / 更复杂算法题上的边界表现。

## 关键术语表
- **RLVR（Reinforcement Learning from Verifiable Rewards）**：利用可执行/可验证的客观信号（如测试通过与否）替代人类偏好反馈进行强化学习微调的方法。
- **Reward Hacking（奖励欺骗）**：模型利用奖励函数的漏洞获取高分，但实际能力并未提升的现象，常见于稀疏测试场景下的 false positive。
- **TSP（Test case Space Polarization）**：基于香农熵定义的测试集多样性指标，衡量测试用例在不同故障模式上的诊断覆盖均匀程度。
- **Dense Reward（密集奖励）**：相对于 0/1 稀疏奖励，提供连续/分级反馈信号（如基于通过率的阶梯值），缓解噪声测试导致的梯度失真。
- **Faulty-code-driven Synthesis（有缺陷代码驱动合成）**：以"部分通过/部分失败"的故障代码作为负例引导，定向生成能触发特定逻辑错误的判别性测试用例的策略。
- **Pass@K**：对同一问题采样 K 次，至少有一次通过全部测试的概率；本文以 pass@10 做题目难度筛选。
- **GRPO（Group Relative Policy Optimization）**：DeepSeek-R1 提出的 RL 算法，对组内样本计算相对优势并进行策略更新，本文作为训练优化器。
- **Validator Agent**：由 LLM 自动生成、用于静态/动态验证测试用例是否符合题目隐含约束的程序（如本论文的 testlib-based C++ validator）。

## 可复现要素
- **数据集**：CodeContests+（https://github.com/Windy Bay 等）；论文未明确声明自行合成的 RobustTests 测试集是否开源，仅说明在论文 Appendix 中提供代码示例。
- **代码/权重**：模型为 Qwen3-32B（开源权重）；框架代码未在论文中公开链接，附录提及 prompt 模板见 Appendix F。
- **关键超参**：$\beta_{\text{clip}}=0.2$、$\eta=5\times10^{-6}$、batch size=256、warmup=10 step、$n_{\text{sample}}=8$、$T{=}1.0$、$\lambda_{\text{KL}}{=}0.0$、$\beta_1{=}0.9$、$\beta_2{=}0.95$、max input=2048 tokens、max output=38,912 tokens。
