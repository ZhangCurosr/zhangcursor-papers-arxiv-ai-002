---
title: "Sampling-Luck-Masquerades-as-Allocation-Gain-Auditing-Test-T"
source: https://arxiv.org/pdf/2608.13087v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:24:42"
field: "NCO 测试时预算分配与评估审计"
keywords: ["neural combinatorial optimization", "test-time compute", "resource allocation", "selection bias", "optimizer's curse", "pre-registration", "evaluation methodology", "best-of-k sampling"]
innovations: ["首次量化 NCO 测试时分配价值的 in-sample 选择偏差，证明分布内增益为零", "提出 per-instance null 噪声底限与 d_split out-of-sample 校正流程", "在分布偏移混合负载上测得 11–12% 真实分配增益（信号免费）与 3–5% 计费保留"]
---

# 论文速读：Sampling-Luck-Masquerades-as-Allocation-Gain-Auditing-Test-Time-Budget-Allocation-for-Neural-Combinatorial-Optimization

## 一句话总结
本文首次系统审计了神经组合优化（NCO）求解器在测试时预算分配中的测量有效性：在分布内工作量上，传统"同批样本决定并评估分配"的做法会制造约 2.2–2.6% 的虚假增益（选择偏差/优化者诅咒），真实增益为零；但在分布偏移（均匀+聚类混合）下，基于留样统计的贪心分配可提升 best-of-k 约 11–12%，且该信号不只是分布标签的代理（剩余独立增益 4.2pp）。

## 研究问题与动机
- **Q1（价值）**：在固定总采样预算 S 下，按实例自适应分配相对均匀分配能带来多少收益？（此前无人测量）
- **Q2（测量可信度）**：自然的测量流程（同批样本决定分配 + 同批样本评估）是否会产生选择偏差？这正是 optimizer's curse / winner's curse / datasnooping 的经典场景。
- 现有方法不足：测试时计算分配文献（主要在 LLM 领域，如 Snell 等 2025、Damani 等 2025）多报告自适应分配增益却未声明决策样本与评估样本是否分离；NCO 领域更无此类审计标准。

## 核心贡献（创新点）
1. **首次测量 NCO 测试时采样的分配价值**：分布内无显著增益，分布偏移下 11–12%（信号免费），带计费探针下仍保留 3–5%；增益幅度在三求解器间与其分布偏移失效程度同序。
2. **首次量化该场景下的 in-sample 选择偏差**：构造 per-instance 零假设（真实增益=0），发现常规 in-sample 流程会伪造 2.2–2.6% 增益，且噪声底限在测试范围内对 K（每实例样本数）和 N（实例数）近似不变——多采集数据无法消除偏差。
3. **提出针对性校正并双向验证**：out-of-sample（d_split）移除分布内幻影增益的同时保留分布偏移的真实增益；同一校正对 POMO（负对照）预期归零也生效。
4. **完整预注册记录与全部物料开源**：门控、端点、修订及方向全部锁定后看数，附每处修订及其对判决的方向影响（包括对自身叙事不利的修订）。

## 方法详解
- **问题建模**：minimize Σᵢ fᵢ(kᵢ)，Σᵢ kᵢ=S，fᵢ 为 best-of-k 期望代价，凸且边际递减，贪心边际分配最优。
- **两条轴**：Axis A（随机解码 rollout，温度采样，AM 的标准模式，无上限）；Axis B（POMO 的标准 n×8 确定性池，TSP-100 下 |pool|=800，有上限 K_pool_i）。
- **离线回放（offline replay）**：每实例存储全部 K 个采样代价，任意策略均可由随机化排序后取前 kᵢ 来评估，无需再次推理；代价归一化为 gap-to-reference（分母用 LKH-3 确定性优 Tour，避免与分散度相关而 inflate 增益）。
- **两个估计量**：
  - d_in：分配与评估在同批存储数组上（有偏向上）；
  - d_split：数组对半切，一半定分配、另一半评估（out-of-sample 估计，但决策样本不计入 S，见局限 4）。
- **正则化 oracle**：对估计的边际曲线取最大凸下包络（greatest convex minorant），避免违反非增边际导致选取非法前缀。
- **实例级零假设（per-instance null）**：从一个实例的存储数组有放回重采样产生 N 个"伪实例"，其真实分配增益=0 by construction；对多种源实例重复得 per-source p95，以其中位数作为噪声底限（主）、p90（保守）；弃用 max（重尾发散）与 pooled null（混合集会上通胀）。
- **预注册确认实验**：新实例种子 + 新解码种子，其他冻结（N=100，K=1000，S/N=100，50:50 混合）；主端点 AM/Axis A，复制端点 SymNCO/Axis A，负对照 POMO/Axis A；2% 阈值延续前期 "proceed" 准则。
- **标签基线对比**：冻结分布标签的固定比例分配（AM 20:1，SymNCO 12:1）作 baseline，证明残差 4.2pp 不是标签的代理。
- **计费探针策略（探索性）**：每实例抽 m=20 rollout 计入同一预算 S，以变异系数驱动分配，不依赖参考 Tour；保留 3.4%（AM）/4.6%（SymNCO）于 50:50 组成。

## 实验与结果
- **数据集/设置**：TSP-100，50 均匀 + 50 聚类（4 高斯团，σ=0.06）；三求解器 POMO、AM、SymNCO；两轴 A/B。
- **分布内审计（Table I，Axis B，N=50，K=800，S/N=100）**：
  - d_in：POMO 2.227[1.63,2.82]、AM 2.567[1.96,3.10]、SymNCO 2.206[1.61,2.72]（均 CI 不含 0）；
  - d_split：0.457[-0.44,1.34]、0.015[-1.08,1.06]、−0.512[-1.79,0.38]（全含 0）；
  - 噪声底限（med/p90）：POMO 2.08/4.21、AM 3.44/5.49、SymNCO 3.55/5.79。
  - → 读第一列每行都像"2% 级增益"；读第二列什么都不剩。
- **底限对 K、N 不变（Tables II–III, Fig.2）**：K=400/600/800 与 N=10/20/30/50 下底限均平直（Monte-Carlo 误差内），说明"多采集无法 outrun 偏差"。
- **预注册确认实验（Table IV）**：
  - 主端点 AM/Axis A/d_split = 11.549[7.40,19.73]，CI 下界 ≥2% → pass；
  - 残差 > 标签基线 = 4.204[1.86,7.70] → pass；
  - 复制 SymNCO = 12.017[5.18,19.98] → pass；
  - 负对照 POMO = −0.290[−0.72,0.24] → interval 含 0，符合预期。
- **强度对应（Table V）**：AM 偏移 34.8× → 增益 11.5%；SymNCO 27.4× → 12.0%；POMO 2.4× → −0.3%（增益序与失效序一致）。
- **计费探针（Table VI）**：AM 3.394[0.61,7.56]、SymNCO 4.588[1.62,10.54]、POMO −0.391；信号成本使增益缩减约 3×，但仍保留显著正向。
- **组成扫掠（Table VII，探索性）**：增益随 OOD 份额非单调，10% 时 d_split=11.74、charged=11.00；50% 时 d_split=12.76、charged=3.05——注册 50:50 并非最有利的组成。

## 相关工作脉络
- **Optimizer's curse / winner's curse**（Smith & Winkler 2006；Andrews et al. 2024）：本文 d_in 即该结构在 N 实例预算分配上的应用；但本文强调极值缩放的经典结论不适用于此处底限（后者是 N 实例平均且对 N 不变）。
- **Reality Check / SPA / Deflated Sharpe / SIREN**（White 2000；Hansen 2005；Bailey & López de Prado 2014；Xu et al. 2026）：本文实例级零假设为该类校正的"分配版"特化，针对最小顺序统计量的噪声来源。
- **LLM 测试时计算分配**（Damani et al. 2025；Snell et al. 2025；Brown et al. 2024）：先例在 LLM 设定建立机制；本文贡献不在机制本身而在审计——连续最小顺序统计目标 vs. 验证器 Bernoulli 成功——并给出 NCO 领域专属校正与报告清单。
- **NCO 求解器选择**（Gao et al. 2025）：邻接问题——本文固定单 checkpoint 而非跨求解器选择。
- **NCO 评估实践批判**（Liu et al. 2023）与 **ImageNet 审计**（Recht et al. 2019）：共同动机支撑本报告清单。
- **Snell et al. 2025**：唯一被指出的"已做分离"例外（两折 CV 用于策略选择），但其难度分箱仍用 oracle 信息。

## 局限性与未来方向
1. 仅三个分布偏移点（2.4×/27.4×/34.8×），声称"排序"而非剂量-响应关系。
2. 注册组成 50:50 并非最优；探索性扫掠暗示内部峰值，但未估计位置与机制。
3. 单一问题类 TSP-100；CVRP-100 因参考求解器种子间波动过大被 gate out；三求解器不同解码分布部分替代多样性但不等同。
4. 主端点 d_split 不收费（决策用 500 样本 vs. 评估 100），估的是 headroom 而非可实现固定预算策略；计费变体仅探索性报告。
5. 预注册终止规则 keyed on d_in，无法 crediting d_split 的正向证据，导致四格门控均返回"信号缺席"——作为已知缺陷记录而非事后修补。
6. 底限因求解器×配置而异（表 II–III 中 AM 约 3.0、SymNCO 约 3.7），不可从本文直接借用，需按自身 solver/config 重算。

## 研究启发与可借鉴点
- **将"同批样本决定+评估"作为默认审计靶标**：任何依赖 best-of-k 汇报的方法（NCO、RL、LLM test-time scaling）均应报告 out-of-sample 估计或实例级零假设校准值；本文的 d_split / per-instance null 可直接复用为方法论 checklist。
- **凸正则化（GCM）处理估计边际**：对非增、非凸的估计边际曲线取最大凸下包络是通用技巧，避免非法前缀选择；适用于任何"边际收益分配"场景。
- **噪声底限对规模不变这一反直觉结论**：多采集 K 或 N 并不消除选择偏差，只有校正才有效——这在实验设计上意味着报告更大 in-sample 数字不能提升可信度。
- **标签残差分解**：证明分配信号不是分布标签的代理（4.2pp 残差），方法可迁移至任何"自适应策略 vs. 规则基线"的剥离分析。
- **预注册 + 全修订方向披露**：每个 amendment 记录其对判决方向的影响（包括对自身叙事不利的），可作为可重复研究流程模板。

## 关键术语表
- **d_in / d_split**：分别指 in-sample 与 out-of-sample（对半切）的相对平均 gap 改善估计；前者上偏、后者下偏。
- **Per-instance null**：对每个真实实例用有放回重采样生成等价"噪声副本"，真实增益=0 by construction，用于标定噪声底限。
- **Noise floor（med/p90）**：per-source p95 的中位数（主）或 p90（保守），作为 in-sample 增益的统计底限；弃用 max（重尾发散）与 pooled null（混合集上通胀）。
- **Greatest convex minorant (GCM) 正则化**：对估计的边际改善曲线取最大凸下包络，强制满足非增边际性质以保证贪心分配可行性。
- **Axis A / Axis B**：Axis A 为随机解码 rollout（AM 标准模式，无上限）；Axis B 为有限确定性池（POMO 标准 n×8，上界 |pool|_i）。
- **Label baseline**：冻结分布标签的固定比例分配（如 AM 20:1）；用于证明贪婪分配残差不是标签的简单代理。
- **Charged probe policy**：部署时可执行的预算收费策略——每实例抽 m=20 rollout 计入同一预算，以变异系数驱动剩余分配，不依赖参考 Tour。
- **Optimizer's curse / Winner's curse**：选择最大噪声估计并以其估值报告的结果系统性上偏；本文 d_in 即其在 N 实例分配场景的应用。

## 可复现要素
- **数据集**：TSP-100（50 均匀 + 50 聚类，4 高斯团 σ=0.06）；LKH-3 参考 Tour；论文未公开通用基准但给出全部成本数组 SHA-256 校验。
- **代码/权重**：开源。仓库 https://github.com/nepersoned/best-of-k-allocation 含全部 cost 数组（{expand,confirm}_{POMO,AM,SYMNCO}_axis{A,B}.npz）、三求解器 legacy checkpoint（含 rl4co 键重映射说明）、预注册记录（含每处修订与方向）、6 个结果溯源 JSON 文件、独立脚本（audit / confirm / 两处探索变体）。
- **关键超参**：N=50/100，K=400/600/800/1000，S/N=100，k_i ≤ K/2 且 S/N×4 ≤ K，Axis A 温度采样，Axis B 随机化排序后无放回取前 k_i，凸正则化，gap 归一化分母为 LKH-3 优 Tour，bootstrap CI 使用实例级重采样。
- **种子**：实例 20260812/20260813、解码 11/77、排序 22、bootstrap 33 均分离报告；重采 POMO 与原始数组 bit-for-bit 一致。
