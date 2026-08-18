---
title: "Ventor-QTest-Threat-Model-Driven-Verification-of-Vendor-Host"
source: https://arxiv.org/pdf/2608.16391v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:14:10"
field: "LLM 系统安全与可靠性"
keywords: ["LLM API 审计", "黑盒验证", "部署可信性", "分布偏离度量", "长序列保真度", "托管模型路由"]
innovations: ["将托管推理形式化为路由级随机过程并分离持久均值偏差与间歇极端偏差", "仅从文本重建共粗化KL分布的AFL估计器（零偏校正Dirichlet后验）", "保留完整经验上尾分布的EFL长序列探针（非坍缩报告）"]
benchmarks: ["GPQA-Diamond", "Terminal-Bench 2.1", "DeepSeek V4 Flash 0731 七路由快照"]
---

# 论文速读：Ventor-QTest-Threat-Model-Driven-Verification-of-Vendor-Host

## 一句话总结
本文针对第三方托管 LLM API 的部署可信性问题，提出 Ventor-QTest——一种无需目标 API 概率信息的黑盒审计框架，通过重复请求估计平均保真度损失（AFL）和长序列探测极端保真度损失（EFL），实现对托管模型分布偏离的统计性度量，并在 DeepSeek V4 Flash 七个路由快照上验证了其有效性。

## 研究问题与动机
- **部署验证缺口**：开源权重模型通过第三方 API 部署时，客户端只能看到响应文本和提供方控制的元数据，无法确认实际部署的是否为声称的模型版本、精度或解码配置（如 FP8 等量化版本），服务路径可能使用更便宜的替代模型或不同推理栈。
- **现有方法的局限**：已有黑盒方法（如 Gao et al. 的两样本模型等同检验、RUT 的秩均匀性测试、LLMmap/FLIPS 的指纹方法）主要解决模型归属识别问题，缺乏对服务分布偏离的系统性统计度量；KVV 等能力验证则侧重于功能层面的评测，未分离均值偏差与极端偏差。
- **自适应路由威胁**：若服务商能识别审计请求并选择性路由（将审计流量导向忠实模型、正常流量导向廉价模型），固定探测集将被完全绕过；因此需要形式化的威胁模型来界定审计的有效边界。
- **长期任务敏感性的未知性**：短文本分布度量能否预测长上下文 Agent 任务的表现，缺乏系统实验证据， motivates 联合报告 AFL 与 EFL。

## 核心贡献（创新点）
1. **将托管推理形式化为路由级随机过程**：区分持续偏差、间歇性偏差、替换和自适应路由四类偏离模式，引入交叉窗口均值 A_r 和极端尾部分布 X_r(m) 两个潜变量，使路由级声称为严格的过程建模对象——与既有工作仅做静态两点检验的本质区别在于显式刻画了服务窗口间的时序异质性。
2. **提出仅依赖文本的 AFL 估计器**：对固定上下文重复 M 次请求，通过预定义有限结果映射重建目标输出分布，结合参考端点的 logprob 计算零偏校正的 Dirichlet 后验共粗化 KL 统计量——与 logprob 辅助对比器的本质区别是 AFL 完全不依赖目标端点暴露的概率信息。
3. **引入独立长序列 EFL 探针**：在 N 次独立运行中记录每步中心化的参考 surprisal 偏离，保留经验上尾行为而不压缩为二元标签——与已有方法的本质区别是不做均值坍缩，完整报告经验分布的均值、SD、0.95 分位数与最大值。
4. **联合评估 AFL/EFL 与下游任务的关系**：在 GPQA-Diamond 和 Terminal-Bench 上发现 AFL/EFL 与前者无明显关联，但显著 EFL 与 Terminal-Bench 通过率随任务暴露增加而下降高度吻合——与既有工作的本质区别在于首次将分布偏离度量与长视界 Agent 任务的暴露敏感性联系起来。

## 方法详解
**威胁模型与假设**：
- 声称为 C = (m, ν, q, φ, E)，但审计仅验证文本生成投影。
- 路由 r 建模为随机过程 {Q_{r,b}}_{b∈B}，其中 b 索引独立审计窗口。
- 核心假设：参考端点真实且不与被测方串通（A1）；参考与目标使用相同声明的采样配置（A2）；参考支持 next-token logprob（A3）；窗口内分布近似平稳（A4a）；同上下文多次请求近似独立（A4b）。

**AFL 组件（重复请求）**：
- 对 J 个冻结上下文 x_j，预定义全映射 g_j: 将每返回文本映射到有限字母表 A_j。
- 参考端点给出类别概率 π_{ji} = Pr_P[g_j(Y)=i | x_j]，低质量类别（M·π_{ji} < c=1）合并到 OTHER。
- 目标端点发送 R 次（M=50），计数 (N_{jr1},...,N_{jr|A_j|}) ~ Multinomial(M, θ_{jr})。
- 采用 Dirichlet(τπ_j) 后验（τ=1），计算后验期望共粗化 KL：
  - K̂^{post}_{jr} = Σ_i (a_{jri}/A_0)[ψ(a_{jri}+1) - ψ(A_0+1) - log π_{ji}]
  - 减去零偏基线 b_j(P,M)（通过 20,000 次参数零抽取得到），得到偏校正估计 K̂^{bc}_{jr}
  - AFL = S_r = (1/J) Σ_j K̂^{bc}_{jr}
- 辅助对比器：对暴露 top logprob 的路由，按同一映射聚合后计算 coarsened-KL 作为一致性校验。

**EFL 组件（长序列探针）**：
- 在 N=20 次独立运行中，生成长度 T=500 的序列 Y_{rb,1:T}。
- 参考端点提供条件分布 p_{rbt}(·)=P(·|Y_{rb,<t})，计算中心化 surprisal：
  - e_{rbt} = -log p_{rbt}(Y_{rb,t}) - H(p_{rbt})
- 完整序列偏离：D_{r,b}(T) = |(1/T) Σ_{t=1}^T e_{rbt}|
- EFL 报告 D_{r,b}(500) 的经验分布的 median、SD、q_{0.95}、Maximum，不做阈值截断。

**统计推断**：
- 联合零假设 H_{0,r}: θ_{jr}=π_j（对所有 j），通过 20,000 次独立多项式抽样模拟联合零分布，单侧 p 值经 Holm 校正控制族误差率。
- 不假设 KL 与任务成功率之间存在单调映射。

## 实验与结果
- **数据集**：DeepSeek V4 Flash 0731 版本；12 个冻结上下文（digit 类别）；7 个路由快照（Official self-check、Aliyun 0731、Ark 0731、Baidu Qianfan 0731、StreamLake、DeepInfra FP8、DigitalOcean）。
- **评估基线**：官方 self-check 作为 centered at zero 控制；三个 logprob-capable 路由（Official、StreamLake、DigitalOcean）用于与 logprob 对比器的对比。
- **AFL vs logprob 对比器**：Pearson r=0.971，Spearman ρ=0.657；按路由平均后 Pearson r=0.989、完美秩一致（描述性结论）。
- **AFL 审计结果（Table 2）**：官方控制在零假设内（S_r=-0.0007，Holm p=0.515）；其余 6 个第三方路由均拒绝联合零假设（Holm p=0.00035）；最高偏离为 Aliyun（S_r=0.5704），最低为 DeepInfra FP8（S_r=0.1185）。
- **EFL 结果（Table 3）**：DigitalOcean 的 median（0.00876）、SD（0.01676）、q_{0.95}（0.05414）、Maximum（0.05960）均为最大；StreamLake 虽 median 最小（0.00409）但 SD 和 max 极高（SD=0.01390, max=0.04969），体现复合报告的必要性。
- **GPQA-Diamond**：AFL/EFL 与准确率无显著关联（median D_{500} vs accuracy: Pearson r=0.347, p=0.446）。
- **Terminal-Bench**：AFL 与通过率无关联（Pearson r=0.081, p=0.873）；但 DigitalOcean 在最高暴露四分位通过率降至 13.6%（vs 最低暴露 82.6%，差距 -20.6pp，p=0.0059），失败任务中 84% 为 AgentTimeoutError。

## 相关工作脉络
1. **Gao et al. [17]**：两样本模型等同检验，报告商业 Llama 端点的分布不一致；Ventor-QTest 与之区别在于仅从文本重建分布，无需目标 logits，且报告均值+上尾而非单一距离统计量。
2. **RUT [58]**：基于用户相似 prompt 的秩均匀性测试；Ventor-QTest 不依赖 rank-based 方法，而是直接估计 coarsened-KL 并报告经验上尾。
3. **Cai et al. [5]**：形式化对抗性模型替换，评估文本/logprob 审计器；Ventor-QTest 与之一致之处在于强调文本只审计的可行性，区别在于不训练分类器、直接估计参考相对效应。
4. **KBF [16] / IRIS [56]**：知识边界指纹和整流审计，处理混合/稀释路由；Ventor-QTest 与之定位为互补——前者测分布偏离、后者测能力正确性，且 Ventor-QTest 不假设路由分数可被精确识别。
5. **LLMmap [39] / FLIPS [42] / ESF [2]**：指纹/归属类方法，针对版本识别或篡改检测；Ventor-QTest 不训练分类器，不解决归属问题，而是报告参考相对偏离。
6. **KVV [34, 36]**：能力级验证框架，覆盖工具调用、多模态、长输出、Agent coding；Ventor-QTest 在其前增加一道分布一致性门禁，分离持久均值偏差与间歇极端偏差。
7. **"One Token Is Enough" [4]**：同期工作（2026-07），基于单 token 分布指纹验证；Ventor-QTest 与之区别在于报告 AFL+EFL 两个可交互解释的统计量，而非单一归属判定。

## 局限性与未来方向
- **测量范围有限**：AFL 仅在声明的共粗化映射和观测窗口内有效；EFL 测量的是中心化 surprisal 偏离而非 KL；不能精确估计稀有事件概率或潜变量 X_r(m)。
- **校准乐观风险**：多项式校准假设同单元格请求近似独立，缓存亲和、批处理、相关路由等会破坏此假设，导致区间和 p 值偏乐观。
- **下游因果推断受限**：实验窗口与 benchmark 窗口未对齐，无法排除超时、速率限制、工具语义等混杂因素；DigitalOcean 的 EFL-通过率负关联可能是探索性观察而非因果结论。
- **自适应路由防御的固有局限**：若服务商能完美区分审计流量并选择性路由，任何固定探测集均可被绕过；prompt 秘密性不构成对识别任务族提供商的保护。
- **未验证的扩展**：未覆盖 tool call 格式化、多模态预处理、Agent-环境交互等 API 语义维度；多窗口交叉均值 A_r 和 excess maximum X_r(m) 仅在理论层面提出，未在实验中直接估计。

## 研究启发与可借鉴点
1. **复合统计量设计范式**：将 A_r（均值偏差）与 X_r(m)（极端尾）作为正交可报告量而非加权合并，这一设计避免了对不同物理含义指标的任意归一化，可直接迁移至其他服务级分布审计场景（如向量检索 API、多模态模型 API）。
2. **Dirichlet 后验零偏校正策略**：利用 Dirichlet(τπ) 先验稳定稀疏类别的 KL 估计，再减去参数零分布下的期望基线 b_j(P,M)，该思路适用于任何小样本分类分布对比任务。
3. **经验上尾报告的完整化**：EFL 保留所有 20 次运行的中心 surprisal 分布（median/SD/q_{0.95}/max）而非二分判定，为后续研究者提供了细粒度诊断信号，建议在本团队的服务质量监控中也采用类似的非坍缩报告策略。
4. **AFL-EFL 与下游任务暴露敏感性的关联分析框架**：将分布度量按任务暴露程度分箱并比较 gap，该探索性分析范式可用于系统性研究分布保真度与长程 Agent 任务鲁棒性的关系。
5. **参考端点的双重角色**：参考端点既提供 logprob 用于 AFL 基线计算，又用于长序列每个位置的 rescoring；这种设计确保比较在同一概率测度下进行，可借鉴于任何需要参考分布对比的场景。

## 关键术语表
**AFL (Average Fidelity Loss)**：平均保真度损失，即零偏校正的窗口内共粗化 KL 统计量均值，衡量目标路由相对于参考的持久性分布偏差。

**EFL (Extreme Fidelity Loss)**：极端保真度损失，指在多次独立长序列运行中观测到的中心化 surprisal 偏离的经验上尾行为（含 median/SD/q_{0.95}/max），捕捉间歇性严重偏差。

**共粗化 KL (Coarsened KL)**：通过预定义的全映射 g_j 将连续/高维输出空间坍缩到有限字母表后计算的 KL 散度，受数据加工不等式约束，提供保守下界。

**路由（Route）**：托管 API 的实际服务路径，可能因负载均衡、副本切换、推理内核更换而在不同审计窗口间表现不同，建模为随机过程 {Q_{r,b}}。

**中心 surprisal (Centered Surprisal)**：e = -log p(y) - H(p)，即单个 token 的实际负对数概率减去参考分布熵，零期望条件于参考分布，构成鞅差结构。

**Huber 阈值 (Pooling threshold c)**：将参考端点预期质量低于 M·π_{ji}<c 的类别合并到 OTHER 的最小阈值，本文取 c=1 为主协议，同时报告 c=2,5 的敏感性分析。

**Terminal-Bench**：面向真实命令行环境的 Agent 基准，89 个任务，使用 Terminus-2 agent，用于评估长视界多步任务的执行正确性。

**GPQA-Diamond**：研究生级别 Google-proof 问答基准，198 道高质量题目，用于评估通用知识准确性。

## 可复现要素
- **数据集**：GPQA-Diamond（HuggingFace 公开）；Terminal-Bench 2.1（GitHub 公开）；DeepSeek V4 Flash 0731 的 12 个冻结上下文和类别映射（论文附录 + GitHub 释放 artifact）。
- **代码开源**：✅ 已开源，地址 https://github.com/Tencent/AI-Infra-Guard/tree/main/services/api_checker/ventor_qtest；实验数据（JSON/CSV）也在仓库中，包含 4,200 次匹配路由响应、12 个参考分布、完整 198 项 GPQA 结果和 89 项 Terminal-Bench 结果。
- **关键超参**：M=50（每上下文重复次数）；τ=1（Dirichlet 先验强度）；c=1（pooling 阈值）；J=12（冻结上下文数）；R=7（路由数）；N=20（独立运行数）；T=500（序列长度）；20,000 次参数零抽取；analysis seed=20260814；温度=1，单次 token 输出限制。
- **环境**：Python 3.11.6, NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.11.1。
