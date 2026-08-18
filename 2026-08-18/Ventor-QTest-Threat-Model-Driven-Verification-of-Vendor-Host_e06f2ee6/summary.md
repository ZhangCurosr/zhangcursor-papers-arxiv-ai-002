---
title: "Ventor-QTest-Threat-Model-Driven-Verification-of-Vendor-Host"
source: https://arxiv.org/pdf/2608.16391v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:55:41"
field: "大语言模型部署验证与安全性"
keywords: ["LLM API 审计", "黑盒验证", "保真度损失", "coarsened KL", "分布式服务稳定性", "vendor-hosted model", "长程 agent 任务"]
innovations: ["复合黑盒审计框架同时报告均值 AFL 与尾部 EFL 两个互补统计量", "纯文本 Dirichlet 后验 coarsened-KL 估计器无需目标 logprob", "揭示 EFL 与长周期 Terminal-Bench 任务通过率的关联性"]
benchmarks: ["GPQA-Diamond", "Terminal-Bench 2.1"]
---

# 论文速读：Ventor-QTest-Threat-Model-Driven-Verification-of-Vendor-Host

## 一句话总结
Ventor-QTest 是一个针对第三方托管 LLM API 的黑盒审计框架，通过重复请求重构条件输出分布（AFL）和长序列中心惊异尾分布（EFL）两个互补指标，检测托管端点与可信参考部署之间的行为偏差；实验表明 EFL 与 Terminal-Bench 长周期任务通过率下降显著相关，提示极端保真度损失对长程代理任务可靠性影响更大。

## 研究问题与动机
- **托管模型身份声明缺乏可信保障**：第三方 API 所宣称的模型名称仅为声明而非密码学远程证明（remote attestation），服务端可能以旧版 checkpoint、廉价替代模型、量化部署或不同解码栈响应请求，但客户端仅能观察到文本回复与 provider 控制的元数据。
- **现有黑盒审计方法侧重归属/指纹，无法量化分布偏离**：如 LLMmap、FLIPS 等方法主要解决实例身份识别或 tamper 检测，不报告相对于可信参考的分布偏离量；KVV 等能力验证则关注功能行为，未分离持续均值偏差与间歇性尾部偏差。
- **短文本探针难以捕捉间歇性路由偏离**：异构副本池或时变路由器可能在大多数窗口表现正常但偶尔选择显著不同的配置，导致平均偏差适中而极端偏差被掩盖；现有方法未显式建模 serving-window 的随机过程维度。
- **量化（如 FP8/FP4）会改变可观测行为但不一定恶意**：部署验证研究表明量化可改变输出分布，但现有方法难以仅从文本区分有意替换与低精度部署。

## 核心贡献（创新点）
1. **将托管推理形式化为 route-level 随机过程并区分四种偏离类型**：现有工作（如 Gao et al.、KBF）侧重单次检验或固定探针，本文显式建模路由在不同 audit window 间的随机性，区分 persistent、intermittent、substitution 和 adaptive-routing 偏离。
2. **提出纯文本 AFL 估计器，无需目标 API 提供 logprob**：与 RUT、LLMmap 等需要 logprob 或大量参考生成的方法不同，本文仅用目标文本计数 + 参考概率向量即可估计 coarsened-KL，并在三个支持 logprob 的路由上与 logprob-derived 比较器呈现 $r = 0.971$ 线性一致。
3. **引入 EFL 长序列探针保留极端偏离信息**：不同于将偏差压缩为均值或二元标签的基线方法，EFL 以独立运行的 centered-surprisal 经验分布报告上尾统计（中位数、SD、0.95 分位数、最大值），两个指标具不同单位不合并为标量。
4. **发现 EFL 与长周期任务性能的关联模式**：AFL/EFL 与 GPQA-Diamond 准确率无显著关联，但 EFL 突出的 DigitalOcean 路由在 Terminal-Bench 中随任务暴露度增加通过率从 82.6% 降至 13.6%，揭示了极端保真度损失对长程任务可靠性的潜在威胁。

## 方法详解
**整体架构**：Ventor-QTest 为复合黑盒审计，包含 AFL（均值分量）和 EFL（尾部分量）两个互补组件，输出 $\mathcal{V}_r = (S_r, \widehat{F}_r^D)$。

**AFL 组件（重复请求均值探针）**：
- 对每个冻结约束上下文 $x_j$，预定义全映射 $g_j$ 将所有可能返回字符串分配到有限字母表 $\mathcal{A}_j$（如精确数字标签 + OTHER 桶）。
- 可信参考端点 $P$ 提供类别概率 $\pi_{ji} = \Pr_P[g_j(Y)=i \mid x_j]$；低质量类别（$M\pi_{ji} < c=1$）合并至 OTHER。
- 向目标路由 $Q_r$ 发送相同上下文 $M$ 次，类别计数服从 $\text{Multinomial}(M, \theta_{jr})$。
- 采用参考为中心的 Dirichlet 后验：$\theta_{jr} \sim \text{Dirichlet}(\tau \pi_j)$，$\tau=1$，后验期望粗 KL 为：
$$\widehat{K}_{jr}^{\text{post}} = \sum_i \frac{a_{jri}}{A_0}\left[\psi(a_{jri}+1) - \psi(A_0+1) - \log \pi_{ji}\right]$$
- 减去零假设下的有限样本基线 $b_j(P,M)$（通过 20,000 次参数化零假设抽样估计），得到无偏估计：
$$\widehat{K}_{jr}^{\text{bc}} = \widehat{K}_{jr}^{\text{post}} - b_j(P,M), \quad S_r = \frac{1}{J}\sum_{j=1}^J \widehat{K}_{jr}^{\text{bc}}$$
- 负值保留避免裁剪偏置；跨路由 Holm 校正控制族系误差。

**EFL 组件（长序列尾部探针）**：
- 独立生成 $b$ 轮长度为 $T$ 的序列，在每个位置 $t$ 用参考端点计算条件分布 $p_{rbt}(\cdot) = P(\cdot \mid Y_{rb,<t})$。
- 中心化参考惊异：$e_{rbt} = -\log p_{rbt}(Y_{rbt}) - H(p_{rbt})$，其中 $H(p) = -\sum_y p(y)\log p(y)$。
- 完整序列偏差：$D_{r,b}(T) = \left|\frac{1}{T}\sum_{t=1}^T e_{rbt}\right|$；若目标遵循参考分布，$e_{rbt}$ 构成 martingale-difference 序列。
- 跨 $B$ 次独立运行保留经验分布 $\widehat{F}_r^D$，报告中位数、SD、经验 0.95 分位数和最大值作为 EFL 摘要。
- 统计单位是完整序列而非单个 token，置信区间通过非参数 bootstrap 对独立运行聚类重采样（20,000 次）。

**关键设计原则**：AFL 和 EFL 具有不同单位，联合报告不合并为加权标量；参考端点需支持 next-token logprob，目标端点仅需普通文本生成；类别映射完全由参考概率决定，目标响应不影响 support。

## 实验与结果
- **数据集/路由**：DeepSeek V4 Flash 0731 的 7 个路由快照（Official self-check、Aliyun 0731、Ark 0731、Baidu Qianfan 0731、StreamLake、DeepInfra FP8、DigitalOcean），均基于 OpenRouter 或官方 API。
- **AFL 实验**：12 个冻结上下文，每上下文每路由 $M=50$ 次请求，temperature=1，单 token 输出限制。主要结果（Table 2）：
  - Official 控制：$S_r = -0.0007$，95% CrI $[-0.0144, 0.0193]$，Holm $p=0.515$（不拒绝零假设）。
  - 第六方路由全部拒绝（$p=0.00035$），其中 Aliyun 0731 最高 $S_r=0.5704$，DeepInfra FP8 最低 $S_r=0.1185$。
  - 与 logprob-derived 比较器的线性描述一致性：Pearson $r=0.971$，Spearman $\rho=0.657$；按路由平均后 Pearson $r=0.989$。
- **EFL 实验**：20 次独立运行 × 500 位置，7 路由配对（Table 3）：
  - DigitalOcean 具有最大中位数 $D_{500}=0.00876$、SD=0.01676、$q_{0.95}=0.05414$、Max=0.05960。
  - StreamLake 中位数最小（0.00409）但 SD（0.01390）和 $q_{0.95}$（0.04817）极高，说明 EFL 与 AFL 不完全对齐。
- **下游关联**：
  - GPQA-Diamond（198 题）：AFL/EFL 与准确率无显著关联（Pearson $r=0.347$, $p=0.446$）。
  - Terminal-Bench 2.1（89 任务，Terminus-2 agent）：$S_r$ 与通过率无关联（Pearson $r=0.081$, $p=0.873$）；但 DigitalOcean 在高暴露四分位（Q4）通过率降至 13.6% vs Q1 的 82.6%，与对照路由差距达 -20.6 个百分点（bootstrap 95% CI $[-36.0, -4.5]$，$p=0.0059$）。

## 相关工作脉络
- **Gao et al. [17]**：两样本模型相等性检验，检测商业 Llama 端点分布不一致；定位差异——本文从文本计数估计 coarsened-KL，不需要目标 logits 也不需要本地 replica。
- **RUT [58]**：将目标响应映射到随机排名并检验排名均匀性；定位差异——本文报告参考相对的具体偏离量（KL 统计量）而非二值判定。
- **LLMmap [39] / FLIPS [42]**：基于行为指纹的实例归属/身份识别；定位差异——本文不做分类器训练，报告声明有限结果空间上的参考相对效应，不解决归属问题。
- **KVV [34,36]**：托管部署的能力验证（工具调用、多模态、长输出推理、agent 编码）；定位差异——本文在能力基准测试前分离持续性均值偏差与间歇性尾部偏差，提供分布层面的前置审计。
- **One Token Is Enough [4] / IRIS [56]**（并发工作）：单 token 指纹和文本审计 fractional routing dilution；定位差异——本文同时报告 AFL 和 EFL 两个互补指标，而非仅做模型归属或路由分数估计。
- **You've Changed [14] / B3IT [6]**：API 变更跟踪，对比生成文本的语言特征分布或边界输入；定位差异——本文基于预声明的分类输出空间和 Dirichlet 后验估计，而非特征分布比较。

## 局限性与未来方向
- **测量范围有限**：AFL 仅在声明的 outcome map 和观察窗口内报告 coarsened KL；EFL 是 centered-surprisal 偏差而非 KL。当前样本支持观察分布比较，但不能精确估计稀有事件概率或 $X_r(m)$（跨窗口的极端偏差潜变量）。
- **假设脆弱性**： multinomial 校准将 cell 内请求视为近似独立，参考向量视为固定；相关路由、overdispersion、参考不确定性或参考漂移会使区间和 null $p$ 值偏乐观。
- **自适应审计防御威胁**：provider 可识别审计探针并选择性路由（adaptive selective routing），本文的上下文仅在声明的审计分布下有效，不将 prompt 保密视为对抗识别的防护。
- **下游关联非因果**：EFL 与 Terminal-Bench 通过率下降的共现模式尚为探索性解释；probe 和 benchmark 窗口未时间对齐，无法排除 timeout、rate limit、tool semantics 等混杂因素。
- **未来方向**：需要更多独立窗口以估计稳定的跨窗口上尾；需时间对齐的 probe-benchmark 联合收集以确认因果贡献；可扩展至工具调用、多模态预处理等未覆盖的 API 语义。

## 研究启发与可借鉴点
1. **均值-尾部双层审计范式**：将分布偏离拆分为 within-window 均值（AFL）和 run-level 上尾（EFL）两个独立统计量，避免单一指标压缩带来的信息损失；这一思路可迁移到任何需要对服务分布稳定性进行双重保证的场景（如 fine-tuning 后部署验证）。
2. **参考为中心的稳定 Dirichlet 后验估计**：使用 $\text{Dirichlet}(\tau\pi_j)$ 先验（$\tau=1$）稳定有限样本 KL 估计，同时减去参数化零假设下的基线偏置，该方法在少样本分类场景中具有通用价值，可降低对目标侧 logprob 的依赖。
3. **长序列中心化惊异的非参数 bootstrap 聚类**：将完整序列作为统计单位，以 run 为聚类单元进行 bootstrap，避免了 autoregressive 历史导致的 token 级独立性假设失效；适用于任何自回归模型的服务稳定性评估。
4. **EFL 与长程任务性能的关联发现**：揭示极端保真度损失对长 horizon 任务的敏感性，为后续研究提供了可复现的实验协议——可结合不同任务复杂度分层分析 EFL 与各类 agent benchmark 的关联模式。
5. **纯文本分类空间重构**：通过预声明的全映射 $g_j$ 将所有响应（包括异常/多 token/缺失）归类，确保每个请求都贡献似然，这种方法论避免了对齐过滤，可作为其他黑盒分布估计任务的通用设计。

## 关键术语表
**Route-level stochastic process**：将托管模型服务建模为跨 audit window 的随机过程 $\{Q_{r,b}\}_{b\in\mathcal{B}}$，而非静态分布，以显式表达间歇性偏离。

**AFL (Average Fidelity Loss)**：within-window 均值保真度损失，基于参考为中心 Dirichlet 后验的重建 coarsened-KL 统计量，经零偏置校正，单位为 KL 散度。

**EFL (Extreme Fidelity Loss)**：极端保真度损失，以独立运行的 centered-surprisal 偏差的经验分布上尾摘要表示（中位数、SD、$q_{0.95}$、最大值），不是 KL 估计量。

**Coarsened KL**：通过预声明的分类映射 $g_j$ 将连续输出空间粗化后的 KL 散度，由数据处理的单调性保证其为真实 KL 的上界下估计。

**Centered-surprisal**：$e_{rbt} = -\log p(Y_{rbt}) - H(p)$，即实际产生 token 的负对数概率减去参考条件分布的熵，期望为零构成 martingale-difference。

**Adaptive selective routing**：provider 根据请求分类动态选择路由目标，对审计探针返回忠实模型而对其余流量返回替代模型的行为策略。

**Termimal-Bench**：基于终端 CLI 环境的长周期 agent 任务基准（89 个任务），用于评估多步工具调用和执行能力。

**Holm correction**：用于多重假设检验的序贯 Bonferroni 方法，控制族系误差率（FWER），本文跨匹配路由族使用。

## 可复现要素
- **数据集**：GPQA-Diamond（198 题）、Terminal-Bench 2.1（89 任务）、12 个冻结上下文；GitHub 开源实现 + JSON/CSV 数据制品（不含 authorization headers 和 API keys）。
- **代码/权重**：代码已开源（https://github.com/Tencent/AI-Infra-Guard/tree/main/services/api_checker/ventor_qtest）；权重不涉及（黑盒方法）。
- **关键超参**：$M=50$（重复次数）、$J=12$（上下文数）、$T=500$（序列长度）、$c=1$（低质量类别合并阈值）、$\tau=1$（Dirichlet 先验强度）、temperature=1、Holm 校正、20,000 次参数化零假设抽样、20 次独立长序列运行。
- **环境**：Python 3.11.6, NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.11.1。
