---
title: "The-Constitutional-Coverage-Trilemma-in-AI-Governance"
source: https://arxiv.org/pdf/2609.01275v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:52:43"
field: "AI 对齐与治理"
keywords: ["constitutional AI", "preference alignment", "value pluralism", "LLM audit", "menu regret", "sparse basis design", "regulatory governance"]
innovations: ["在单一五值单纯形框架下联合审计人类需求与前沿LLM供给并发现2%覆盖缺口", "形式化预算多元三难（BCP/PE/GMRP不可能性）并经验验证绑定", "发现跨厂商自主权单调漂移集中于低风险场景的staked-conditioned分解"]
benchmarks: ["AI Jamm配对交易问卷（20-item, 10 pairs × 2 reversed orderings）", "23-archetype frontier菜单（Claude/GPT/Gemini/Llama/Grok/DeepSeek 六族多版本）", "Vertex menu枚举（2^5-1=31个子集）"]
---

# 论文速读：The Constitutional Coverage Trilemma in AI Governance

## 一句话总结
本文首次将前沿 LLM 的"宪法型"供应与人类偏好需求置于同一五值单纯形框架下进行联合审计与配对交易研究，发现当前 23 个前沿架构仅覆盖人类需求凸包的约 2%，且随版本迭代自主权权重在 5/6 模型族中单调下降；最优稀疏顶点菜单 `{HON, AUT}` 将平均后悔降低 47%（以 21 个架构为代价），并严格形式化了"预算多元困境三难"。

## 研究问题与动机
- **匹配瓶颈被误诊**：主流观点认为 AI 治理只是路由问题（足够多的提供者 + 偏好提取就能匹配），本文指出真正的瓶颈是"宪法型是否覆盖需求"——即使偏好提取完美，若菜单缺少对应类型也无意义。
- **宪法型 homeless 概念化**：定义并量化"宪法型无家可归"（constitutional homelessness）——即用户的优势价值在现有菜单中不被任何架构作为首要优先级。
- **漂移方向恶化最弱势用户福利**：前沿供给并非静态——随版本迭代，已严重不足的自主权权重进一步下降，机械性地抬高劣势群体的后悔下界（Corollary 3）。
- **现有对齐框架不足**：HHH（helpfulness/honesty/safety）仅覆盖三个价值，无法表征治理中最频繁出现的自主权（AUT）与公平（EQT）冲突，需要扩展为五值单纯形。

## 核心贡献（创新点）
1. **联合审计框架**：在同一 21-paraphrase × 10-scenario × 2-ordering 仪器上同时测量 1,649 名美国参与者的宪法需求与 23 个前沿 LLM 架构的宪法供给，解决模型间差异约 5σ。
2. **覆盖度的 regret 下界定理**：Theorem 1 证明每用户后悔满足 `M_i^A ≥ (1 - β_{r_i^†}(A)) · m_i`，首次将供应侧覆盖系数 β_r 与需求侧优势 margin m_i 结合给出后悔的理论下界。
3. **预算多元困境三难形式化**：Theorem 2 证明在小菜单下，个性化效率（PE）、群体后悔均等（GMRP）、有界菜单规模（BCP）三者不可兼得；经验上 |A|=3 时最差组后悔无法低于 0.09。
4. **稀疏顶点充分性**：Theorem 3–4 证明线性福利下最优稀疏菜单必由单纯形顶点构成（内点架构是福利冗余的），且通过次模贪心可实现 (1-1/e) 近似，给出 `{e_HON, e_AUT}` 两顶点战胜 23 顶点前沿 47% 的定量结果。
5. **漂移的 stakes-conditioned 分解**：揭示自主权下降集中于"低风险"场景（辞职、讽刺），而非"高风险"场景，反转了老版本的前沿模型所展现的风险敏感差异，且跨六家共享独立架构的训练管道的厂商一致性（Spearman 趋势在 5/6 族中为负，order-permutation p=0.013）。

## 方法详解
- **五值单纯形模型**：每个用户 i 的偏好 π_i ∈ Δ^5（SAF, HLP, HON, AUT, EQT 和为 1），菜单 A ⊂ Δ^5，用户福利 U_i(α) = ⟨π_i, α⟩，理想福利 U_i^† = max_k π_i^k（达到于顶点 e_r），菜单后悔 M_i^A = U_i^† - max_{α∈A}⟨π_i, α⟩，优势 margin m_i = π_i^{r_i^†} - max_{k≠r_i^†} π_i^k。
- **覆盖系数**：β_r(A) = max_{α∈A} α_r，衡量菜单对价值 r 的最大覆盖度；β-严格多数无家可归指 β_r ≤ 1/2，argmax-主导无家可归指没有 α 满足 argmax_k α_k = r。
- **人类研究**：1,789 名美国成年人经 Prolific 招募，按政治身份分层；完成 20 项配对交易问卷（10 对五值组合 × 2 次反向排列）；以一致选择分布构造 π̂_i（一致率均值 0.793），排除后 N=1,649；分歧行为与犹豫时间强相关（Spearman ρ=0.84）。
- **前沿审计协议**：27 个前沿 LLM（Claude/GPT/Gemini/Llama/Grok/DeepSeek 六族多版本），每场景 21 种同义改写（Mistral Large 生成，排除出测试池），每改写双向位置控制各 1 次，共 420 次试验/模型；一致性阈值 ≥0.70 保留 23 个架构。
- **漂移检验**：对每族计算端点 Δ_r = α_r^newest - α_r^oldest，Spearman ρ 趋势检验，order-permutation test（permuting whole profiles within families）p=0.013；stakes 分解分离高风险（无人机优化、化疗替代）与低风险（辞职、讽刺）场景。
- **稀疏菜单设计**：枚举 2^5-1=31 个顶点子集；mean-greedy 依次添加 AUT→HON→SAF，worst-greedy 依次添加 AUT→EQT→HON；三顶点 {e_SAF, e_HON, e_AUT} 的 per-type 后悔如表 3。

## 实验与结果
- **人类需求分布**：SAF 32.6%，HON 22.6%，AUT 19.2%，HLP 18.1%，EQT 7.6%；主成分仅解释 35% 方差，需求高度异质。
- **前沿供给覆盖系数**：β=(0.394, 0.257, 0.333, 0.161, 0.381)，全部低于严格多数阈值 1/2；零个架构以 HLP 或 AUT 为首要优先。
- **凸包覆盖率**：23 个架构的凸包占人类需求凸包的 ~2%（保守噪声匹配估计）；0.10%（全精度）；预算匹配下 [1.1%, 4.2%]。
- **平均后悔**：23 架构前沿 E[M_i^A]=0.140（CI [0.135, 0.144]），只捕获 63% 理想福利；两顶点 {e_HON, e_AUT} 降至 0.074，改善 47%（CI [43%, 52%]）。
- **贪婪修复**：三顶点 mean-greedy 添加降低平均后悔 81%（至 ~0.027），worst-greedy 添加降低最差组后悔 64%（至 ~0.129）。
- **漂移方向**：Δ̄=(+0.052, +0.002, -0.043, -0.042, +0.031)，AUT 是唯一在 5/6 族中下降的价值；高风险 AUT 近零且未变，低风险 AUT 下降显著（Δ≈-0.22）。
- **三难绑定实例**：{e_SAF, e_HON, e_AUT} 三顶点 mean-opt 平均后悔 0.032 但 worst-group 0.215；worst-opt {e_HON, e_AUT, e_EQT} 平均 0.045/worst 0.093；|A|=3 时 worst-group 无法低于 0.09。

## 相关工作脉络
- **Constitutional AI [1]**：Bai et al. 2022 的 harmlessness from AI feedback 框架，本文将其扩展为五值覆盖审计，强调每个部署模型作为"宪法机构"做权衡的本质。
- **Pluralistic alignment [3,4]**：Sorensen et al. 2024、Conitzer et al. 2024 倡导多元对齐，本文比其更进一步——指出关键不在偏好提取而在供应覆盖，路由本身无法弥补类型缺失。
- **LLM 偏好模拟人类 [11,12,13]**：Argyle et al.、Aher et al.、Horton 探索 LLM 模拟人类样本；本文反其道：将 LLM 作为"响应者"（characterized, not used as surrogate）进行审计，避免污染风险。
- **政治意识形态 LLM 对齐 [6,7,8]**：Santurkar et al.、Durmus et al.、Hartmann et al. 发现模型偏向中间/进步立场；本文的 drift 分析揭示更精细的模式——跨厂商收敛到 SAF+EQT↑/HON+AUT↓ 的"可辩护区域"，而非随机偏移。
- **社会选择与公平不可能性 [14-20]**：Dwork et al.、Hardt et al.、Kleinberg et al. 的不可能性定理启发了本文 Theorem 2（BCP-PE-GMRP 三难）的形式化，类比至宪法型覆盖语境。
- **子模贪心 [23,24]**：Nemhauser-Wolsey-Fisher 的 (1-1/e) 近似理论支撑 Theorem 5/Corollary 5 的稀疏菜单设计，本文首次将该框架用于 AI 治理菜单选择。

## 局限性与未来方向
- **五值框架非穷尽**：未建模文化适当性、环境成本、代际影响等价值，扩展可能扩大差距（尤其文化适当性维度）。
- **线性福利是最有利假设**：凹聚合会使后悔下界更紧、最优小菜单退化为内点（Appendix C 以匹配距离重新度量，稀疏菜单优势仍存但顶点充分性失效）。
- **样本与场景局限**：仅美国 Prolific 用户、10 个设计师选场景；跨国样本可能重构需求几何； Appendix U 显示 tech-literacy、age、gender 均有梯度但政治身份仅 small demand-shift（Cramér's V=0.053）。
- **漂移因果机制未明**：观测性证据显示版本序有意义，但未识别驱动因素（benchmark 组成？监管压力？声誉风险？）；需要 release-cadence 审计来跟踪。
- **默认作为供应单位**：未审计 steer 变体；steer 可扩展可达凸包，是未来 work（reachable-hull audit）。
- **温度 0 默认查询**：更高温度可能改变 profile 估计；协议在 Appendix J 显示 21-variant mean 的 SE=0.035，模型间差异分辨至 ~5σ。

## 研究启发与可借鉴点
1. **同一仪器双端测量**：将人类研究与模型审计放在完全相同的配对交易仪器上（相同场景×相同改写×相同位置控制），使供需直接可比——这是该论文方法论的核心创新，可迁移至任何"偏好 vs. 系统输出"比较研究。
2. **Paraphrase 控制协议**：用 21 种同义改写消除模型对 position/content 的敏感性，以一致阈值过滤噪声响应——为大规模 LLM 审计提供了稳健的 protocol 设计范式。
3. **Stakes-conditioned decomposition**：将 AUT 轴拆分为高风险（医疗/危险请求）与低风险（日常决策）场景，揭示"安全训练正常工作"与"低 stakes 下的 agency 剥夺"的根本差异——这一分解方法可用于所有含 value tension 的场景审计。
4. **稀疏顶点设计 + 次模贪心**：Theorem 3-5 表明最优菜单必为顶点集，且贪心有 (1-1/e) 近似保证——可将此框架应用于任何多价值 menu selection 问题（如多 agent system 配置、policy 切换逻辑）。
5. **三难形式化的政策含义**：Theorem 2 证明小菜单下无法同时实现效率、均等、规模有界，且经验上已在 frontier 菜单中绑定——为 AI 治理监管（披露义务、release-cadence 审计）提供了精确的量化基准（420-trial 审计仅需 $0.42–$2.10/model）。

## 关键术语表
- **Constitutional coverage（宪法覆盖）**：菜单中对某价值 r 赋予最大权重的架构的权重值 β_r(A)=max_{α∈A} α_r，衡量供应侧对需求侧某一价值的覆盖能力。
- **Constitutional homelessness（宪法无家可归）**：用户的首要价值 r 在菜单 A 中既无 β-严格多数覆盖（β_r ≤ 1/2），也无 argmax 主导架构（无 α 满足 argmax_k α_k=r）的状态。
- **Menu regret（菜单后悔）**：用户在实际菜单 A 下最优匹配能达到的最高福利与理想顶点福利之间的差额 M_i^A = U_i^† - U_i^A。
- **Budgeted-pluralism trilemma（预算多元困境）**：在小菜单规模约束下，个性化效率（PE）、群体后悔均等（GMRP）、有界菜单规模（BCP）三者不可兼得的形式化不可能性定理。
- **Vertex menu（顶点菜单）**：由单纯形极端点 {e_1,...,e_K} 中若干顶点构成的菜单，线性福利下最优稀疏菜单必为顶点菜单（Theorem 3）。
- **Stakes-conditioned decomposition（风险情境分解）**：按场景对风险严重程度的高低，分离同一价值轴（如 AUT）上的不同行为模式——高风险下的 autonomy 退让 vs. 低风险下的 autonomy 保留。
- **Order-permutation test（顺序置换检验）**：将每个家族内部版本序列随机置换、保留家族内整体 profile 结构，检验版本序是否显著预测宪法位姿的统计检验。
- **Paraphrase-controlled audit（改写控制审计）**：对同一场景生成 21 种语义等价改写，分别查询模型以消除 position bias 和内容噪声的审计协议。

## 可复现要素
- **数据集**：人类配对交易数据 1,649 份（Prolific 美国成年人）；附录 E–G 提供完整问卷与招募细节；跨国扩展计划已在准备中（Appendix U 讨论人口梯度）。
- **代码/权重**：论文未提供开源代码或模型权重；审计协议描述完整（Appendix I–J），可在 API 级别复现。
- **关键超参**：paraphrase 数=21，concordance 阈值=0.70（敏感性检验 0.50/0.70/0.85 结论不变），温度=0（不支持者 0.2+3 samples），bootstrap B=1,000，order-permutation Monte Carlo B=2×10^5。
- **审计成本**：420-trial/模型，当前 API 价格每模型 $0.42–$2.10，六族审计约 $8.50 + 45 分钟。
