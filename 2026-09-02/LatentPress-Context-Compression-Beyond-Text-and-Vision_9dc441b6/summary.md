---
title: "LatentPress-Context-Compression-Beyond-Text-and-Vision"
source: https://arxiv.org/pdf/2609.01507v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:51:53"
---

# 论文速读：LatentPress-Context-Compression-Beyond-Text-and-Vision

## 一句话总结
提出 LatentPress，将对话历史与长文档压缩为连续软令牌（soft tokens），直接注入冻结的语言模型输入嵌入层进行推理，全程无需文本重构。在 4–16× 压缩率下实现等价或更优的问答准确率，写入仅需 43 ms/对话，读取比原始上下文或缓存 OCR 快 5–9×，且仅需训练约 0.1% 的轻量适配器。

## 研究问题与动机
- 长对话助手与长文档阅读场景不断累积历史，但下游决策通常只依赖其中极小片段；现有系统仍以人类可读文本或需解码的图像作为机器接口的默认形式，造成冗余计算与存储。
- 传统连续向量压缩方法（如 Gist、AutoCompressor、ICAE）通常需要微调或 LoRA 适配整个 LLM-scale reader/encoder，训练开销大；视觉压缩路线（如 DeepSeek-OCR）需先渲染为图像再经自回归 OCR 解码为文本，引入额外延迟与精度衰减。
- 语言模型本身并不需要上下文恢复为人类可读文本，能否直接在表示层将上下文写入紧凑的连续向量，由冻结的 decoder 通过输入嵌入接口读取？
- 如何在严格冻结下游 decoder 的前提下，设计轻量级 WRITE/READ 接口，在压缩率、准确率、写入成本与读取延迟之间取得实用平衡？

## 核心贡献（创新点）
- **直接读取的软令牌接口（Direct-read soft-token interface）**：上下文经 writer 映射为连续向量后直接拼接入冻结 decoder 的 `inputs_embeds`，推理时完全跳过文本重构阶段。与 ICAE/xRAG 等需要自解码或仅压缩单段检索结果的方法在接口形态与训练规模上本质不同。
- **读者匹配的极轻量 writer**：仅训练一个复用 decoder 底部两层（冻结）的小线性适配器（4.2M–26.2M 参数，约 0.1% 模型规模），相比同类方法需全量微调或 LoRA 适配整个 LLM 的方案，参数量与计算开销骤降且不影响主模型权重。
- **结构化非对称压缩速率策略**：针对对话历史设计基于角色（user 轮次 $k=1$ 无损保留，assistant 轮次 $k \in \{8,16,32\}$ 池化）的变速率压缩；针对长文档采用均匀池化。无需学习分配策略即可显著优于固定均匀压缩，揭示角色信息分布对压缩表示的关键作用。
- **Reconstruction + Forward-KL 双监督损失**：结合 teacher-forced 交叉熵重建损失与完整上下文分布下的向前 KL 蒸馏损失（$\lambda=1.0$），使压缩表示既能恢复内容又能保留 decoder 的原始推理分布，有效缓解纯重建导致的分布偏移。
- **系统在对话记忆与长文档 QA 上的广泛验证**：在 LongMemEval 与 LongBench-QA 上系统评估了准确率、跨模型泛化、跨域/域内迁移、写入/读取延迟及端到端作业时间，证明软令牌可作为文本/视觉之外的通用机器面向上下文接口。

## 方法详解
- **WRITE/READ 分离抽象**：上下文 $x = (x_1, \dots, x_T)$ 经小型可训练 writer $\text{WRITE}_\phi$ 映射为短序列连续向量 $m$，随后与问题嵌入拼接为 $[m; \operatorname{emb}(q)]$ 输入冻结 decoder $f_\theta$ 直接解码输出 $y$，全程无文本重构。
- **Writer 架构**：deep-copy 目标 frozen decoder 底部 $L=2$ 个 transformer 层作为 encoder（梯度不回流至 reader），顶部接一个 $d \times d$ 线性适配器（初始化为恒等矩阵，使训练初期贴近原始 token embedding）。仅训练该 head，所有 decoder 权重与借用层均冻结。
- **压缩速率设置 $\pi$**：对话场景采用 role-based 规则 $k_{\text{user}}=1, k_{\text{assistant}} \in \{8,16,32\}$，user 轮次 bypass writer 保留原始 embedding，assistant 轮次均值池化；文档场景采用单一均匀因子 $k \in \{4,8,16\}$。整体压缩比由角色分布与 token 长度自然涌现，非硬性预设。
- **训练目标**：$\mathcal{L}(\phi) = \mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{fkl}}$。其中 $\mathcal{L}_{\text{rec}} =
