---
title: "Towards-a-Densing-Law-for-User-Representation-Learning-at-Bi"
source: https://arxiv.org/pdf/2608.23392v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:09:20"
field: "用户表征学习与行为建模"
keywords: ["用户表征学习", "行为分词", "缩放定律", "残差量化", "致密化", "RQ-VAE", "ALGN", "工业推荐"]
innovations: ["提出行为致密化定律，刻画数据规模与最小充分分词容量的幂律关系", "设计ALGN自适应变长分词，基于残差范数+编码不确定性实现实例级容量分配", "揭示十亿级工业场景下原始行为数据缩放的饱和墙及损失-质量解耦现象"]
benchmarks: ["Alipay PayBill生产数据", "50个下游分类数据集", "22个文本检索数据集", "22个U2U检索数据集"]
---

# 论文速读：Towards-a-Densing-Law-for-User-Representation-Learning-at-Bi

## 一句话总结
本文揭示了十亿级工业用户表征学习中**原始行为数据缩放的饱和瓶颈**（Raw Behavioral Scaling Wall），并提出了**行为致密化定律（Behavioral Densing Law）**来刻画数据规模与最小充分分词容量之间的定量关系；在此基础上设计了自适应变长分词方法 **ALGN**，在固定 token 预算下突破了原始缩放瓶颈，并在支付宝十亿级真实业务数据上验证了有效性。

## 研究问题与动机
- **原始数据缩放存在瓶颈**：在用户数量（N）、行为序列时长（D）和模型参数（P）三个维度上，单纯增加原始行为数据量会导致下游性能增益急剧递减，出现"缩放墙"。
- **预训练损失与下游质量解耦（Loss-Quality Dissociation）**：当模型从 0.2B 扩展到 0.4B 时，预训练对比损失持续下降，但下游分类准确度几乎不再提升——更大模型只是在拟合冗余行为细节，而非学习可泛化的用户特征。
- **缺乏分词配置随数据规模缩放的定量分析**：现有分词方法（如 RQ-VAE、VQ-VAE）通常使用固定配置，未研究"在不同数据规模下应如何配置最小充分分词容量"这一关键问题。
- **行为信息的异构性被忽视**：用户历史中存在大量重复/可预测的日常事件，也有少量富含信息的多样化行为，均匀分配 token 容量是次优的。

## 核心贡献（创新点）
1. **发现并形式化了"原始行为缩放墙"现象**：在支付宝十亿级生产数据上，系统性地量化了 N、D、P 三个维度的饱和阈值（N≈0.03B、D≈60天、P≈0.2B），揭示了信息密度而非模型参数量才是工业用户表征学习的关键瓶颈。
2. **提出"行为致密化定律"（Behavioral Densing Law）**：将致密化问题形式化为性能-成本的 Pareto 优化，推导出最小充分分词容量随数据规模呈幂律增长的定量公式 $\ln C^*(N,D) = \beta + \alpha_N \ln(N/N_0) + \alpha_D \ln(D/D_0)$，为配置选择提供理论指导。
3. **设计了自适应变长分词方法 ALGN**：基于残差范数（剩余信息量）和编码不确定性（语义模糊度）两个信号，动态决定每个行为段是否需要继续细化残差量化，实现了从"尺度级容量选择"到"样本级容量分配"的跨越。
4. **系统性验证了致密化定律的泛化性**：在 PayBill、SPM、Miniprogram 三种数据源以及 RQ-VAE、VQ-VAE、SARQ 三种分词方法上均验证了定律的有效性，并发现了斜率与源内多样性 $\mathcal{U}_d$ 近似呈平方比例关系（$\alpha_i \propto \mathcal{U}_d^2$）。

## 方法详解
- **预训练框架**：采用行为-文本对比学习。将用户行为序列分为过去段（由 Transformer encoder $f_\theta$ 编码）和未来段（由 LoRA 微调的 LLM embedding 模型 $g_\phi$ 编码生成文本描述），通过 Info-NCE 对比损失对齐：
  $$\mathcal{L}_{CP} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\text{sim}(\mathbf{e}_i^b, \mathbf{e}_i^q)/\tau)}{\sum_{j=1}^{B}\exp(\text{sim}(\mathbf{e}_i^b, \mathbf{e}_j^q)/\tau)}$$

- **RQ-VAE 残差量化分词**：将多源行为嵌入序列通过局部聚合压缩为固定长度 $H$ 的潜向量，再通过 $M$ 级残差量化生成离散 token 序列 $\mathbf{t}_{n,h}=(k_{n,h}^{(1)}, \ldots, k_{n,h}^{(M)})$。损失函数包含重构损失和 commitment loss：
  $$\mathcal{L}_{RQ} = \mathcal{L}_{rec}(\mathbf{Z}_n, \hat{\mathbf{Z}}_n) + \sum_{m=1}^{M}\|\text{sg}[\mathbf{r}^{(m)}]-\hat{\mathbf{r}}^{(m)}\|_2^2 + \beta\sum_{m=1}^{M}\|\mathbf{r}^{(m)}-\text{sg}[\hat{\mathbf{r}}^{(m)}]\|_2^2$$

- **行为致密化定律推导**：定义行为密度 $\rho^*(\mathbf{s})=\widehat{C}_{raw}(\mathbf{s})/C^*(\mathbf{s})$，通过 Pareto 优化 $\phi^*=\arg\max_\phi[U(\phi,\mathbf{s})-\lambda C(\phi)]$，在效用前沿 $U(C,\mathbf{s})=U_\infty(\mathbf{s})-a(\mathbf{s})C^{-b}$ 的局部近似下，解得最优容量满足幂律关系 $\ln C^*(\mathbf{s})=\beta+\sum_i\alpha_i\ln(s_i/s_{i,0})$。

- **ALGN 自适应变长分词**：在每个量化层级 $l$，通过残差范数 $R_l=\|r_{l-1}-e_{l,m}\|_2$ 和编码不确定性 $E_l=-\sum_{k=1}^l\log p(m_k|m_{<k})$ 两个信号，计算继续量化的概率：
  $$g_l=\sigma(\text{sp}(w_R)R_l+\text{sp}(w_E)E_l-\text{sp}(w_C)c_l+b)$$
  实际激活长度 $L_{act}=\min\{l:g_l\leq\theta\}$，训练损失加入 KL 正则项 $\mathcal{L}=\mathcal{L}_{rec}^{act}+\lambda\text{KL}(P_{len}\|Q)$，其中 $Q$ 为几何分布先验。

## 实验与结果
- **数据集**：支付宝（Alipay）PayBill 十亿级生产数据，用户数 N=100M–2B，序列长度 D=30–270 天，行为文本 token 约 2200/user。下游评测覆盖 50 个分类数据集、22 个文本检索数据集、22 个 U2U 检索数据集（每集 0.5M 用户）。
- **缩放墙发现**：
  - 用户数 $N>0.03\text{B}$ 后性能饱和；时间窗口 $D>60$ 天后增益急剧下降；模型参数 $P>0.2\text{B}$ 时预训练损失继续下降但下游准确度几乎不变。
- **分词 vs 原始数据**：在匹配的数据和计算预算下，RQ-VAE 分词在所有缩放维度上均优于原始序列，且随着 $N$ 和 $D$ 增大优势扩大，在 $N\approx1.2\times10^7$ 和 $D\approx64$ 天后开始超越原始基线。
- **致密化定律验证**：在三种数据源（PayBill $\mathcal{U}_d=0.5513$、SPM $\mathcal{U}_d=0.5891$、Miniprogram $\mathcal{U}_d=0.5292$）和三种分词方法（RQ-VAE、VQ-VAE、SARQ）上均验证了 $\ln C^*$ 与 $\ln s$ 的线性关系；斜率比约 $1:1.15:0.92$，与 $\mathcal{U}_d$ 平方比 $1:1.07^2:0.96^2$ 一致。
- **ALGN 最强结果**（PayBill，180天，0.1B 用户）：AUC=76.43% / KS=43.31% / Acc=83.52%，相比最佳基线 SARQ 分别提升 1.07%/2.03%/0.36%，同时节省 13.23% SID 容量。

## 相关工作脉络
- **用户数据缩放研究**（Ardalani et al., 2022; Shin et al., 2023; Guo et al., 2024; Shen et al., 2025; Zhang et al., 2024a）：关注原始行为数据、模型容量与下游效用的关系，但未涉及分词或容量配置问题；本文与之定位不同——本文揭示原始缩放墙并提出分词破局。
- **行为分词研究**（Feng et al., 2026; Liu et al., 2024a; Zhu et al., 2024; He et al., 2025）：使用 VQ-VAE/RQ-VAE 将行为映射为离散 token，但均采用固定配置，未研究容量随数据规模的scaling规律；本文填补了这一空白。
- **残差量化分词**（van den Oord et al., 2017; Lee et al., 2022; Rajput et al., 2023; He et al., 2025）：TIGER、U²QT 等工作，均使用静态分词配置；本文首次将分词容量配置与数据规模建立定量关系。
- **自适应量化**（Huijben et al., 2024; Seo and Kang, 2024; Chae et al., 2025; Kusupati et al., 2022）：允许不同输入激活不同量化层级，但假设最大可用容量已给定；本文进一步从定律层面指导"应给定多少容量"，再叠加 ALGN 实现实例级自适应分配。

## 局限性与未来方向
- **单一模态局限**：实验仅针对 PayBill 文本行为数据，尚未验证视频消费等其他行为模态下的定律适用性（论文自述）。
- **单平台数据**：基于支付宝内部数据，需在公开 benchmark 上进一步验证鲁棒性（论文自述）。
- **未来方向**：探索多模态致密化估计、更深层的理论分析、将密度优化作为用户建模的一等设计目标。

## 研究启发与可借鉴点
- **"缩放墙"诊断方法论**：系统地在 N、D、P 三个轴上扫描并绘制缩放曲面，观察 Loss-Quality 解耦现象，这一诊断框架可直接迁移到其他推荐/表征学习场景。
- **致密化定律的配方可复用**：$\ln C^*=\beta+\alpha_N\ln(N/N_0)+\alpha_D\ln(D/D_0)$ 的形式简洁且物理意义清晰，对于任何需要配置分词/压缩容量的场景均可借鉴其推导思路（Pareto 优化 + 幂律拟合）。
- **ALGN 的双信号门控机制**（残差范数 + 编码不确定性）：这种基于"边际效用 vs 边际成本"的自适应容量分配思想，可迁移到 NLP 中的 adaptive computation time、多模态中的 token dropping 等场景。
- **跨数据源迁移实验设计**：用 Bill 源训练的压缩配置直接迁移到 SPM 和 Miniprogram，保留 95%+ AUC，说明致密化定律具有较强的跨域泛化能力，验证设计值得借鉴。
- **与团队方向的结合机会**：若团队涉及用户画像压缩、行为序列建模、或大模型 in-context learning 中的 token 效率优化，可将 ALGN 的变长分配思想引入，或在团队已有的分词 pipeline 中嵌入致密化定律的配置搜索。

## 关键术语表
- **Raw Behavioral Scaling Wall（原始行为缩放墙）**：指增量扩大用户数量、行为序列长度或模型容量时，下游性能收益急剧递减直至饱和的现象，伴随预训练损失与下游质量的解耦。
- **Behavioral Densing（行为致密化）**：将冗长的原始行为历史转换为紧凑表征的过程，在保留任务相关区分信息的同时抑制冗余，使每个有效输入单元携带更多下游相关信息。
- **Behavioral Densing Law（行为致密化定律）**：刻画最小充分分词容量 $C^*$ 随数据规模 $s$（用户数、时间窗口等）呈幂律增长的定量规律：$\ln C^*=\beta+\sum\alpha_i\ln(s_i/s_{i,0})$。
- **Loss-Quality Dissociation（损失-质量解耦）**：预训练优化目标（如对比损失）持续下降，但下游任务准确度不再提升的现象，揭示"更好拟合≠更好表征"的核心问题。
- **RQ-VAE（Residual Quantization VAE）**：通过多级码本的残差量化将连续嵌入映射为离散 token 的方法，早期码本捕获宏观语义，后期码本细化残差差异。
- **ALGN（Adaptive Length Gated Network）**：本文提出的自适应变长分词网络，基于残差范数和编码不确定性动态决策是否继续量化，实现实例级的容量分配。
- **$\mathcal{U}_d$（源内多样性度量）**：基于 LLM embedding 的 mean k-NN cosine distance，用于量化行为数据的内部多样性，与致密化定律的斜率系数呈近似平方关系。
- **SID（Semantic ID）**：语义标识符，即分词器输出的离散 code 序列，代表一段行为历史的压缩表征。

## 可复现要素
- **数据集**：支付宝（Alipay）PayBill、SPM、Miniprogram 生产数据；用户数 100M–2B，序列长度 30–270 天；**未公开**（内部数据）。
- **代码/权重**：论文未明确提及开源声明，**未提及**。
- **关键超参**：Transformer encoder 0.05B–0.4B；LoRA rank=16；RQ-VAE 超参 $K$（码本大小）、$M$（残差层数）、$d$（嵌入维度）、$H$（token 长度）；ALGN 几何分布先验 $\gamma=0.3$；温度参数 $\tau$ 可学习；commitment 系数 $\beta$。
