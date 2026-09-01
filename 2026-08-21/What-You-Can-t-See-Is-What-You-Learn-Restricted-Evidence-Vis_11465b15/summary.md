---
title: "What-You-Can-t-See-Is-What-You-Learn-Restricted-Evidence-Vis"
source: https://arxiv.org/pdf/2608.20054v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:08:38"
field: "多智能体语言模型组合泛化与可解释性"
keywords: ["组合泛化", "多智能体LLM", "证据可见性", "因果审计", "潜通信", "模块化神经网络"]
innovations: ["首次在封闭组合任务上通过严格配对实验证明限制证据可见性可因果提升组合泛化能力", "提出值索引包移植方法，建立 learned relay 中中间表征的可互换性证据", "公开完整预注册失败案例与审计协议，建立多细胞社会因果隔离实验新范式"]
benchmarks: ["Z_17仿射变换组合任务", "depth-2/depth-3 held-out程序评估"]
---

# 论文速读：What-You-Can-t-See-Is-What-You-Learn-Restricted-Evidence-Vis

## 一句话总结
本文通过严格配对的因果实验证明：在共享语言模型基因组的多细胞社会中，限制各细胞对证据的可见性（restricted evidence visibility）能显著提高组合泛化能力，促使模型学会可复用、值索引的中间通信接口；该效应源于抑制了整程序查找捷径，而非简单增加通信压力。

## 研究问题与动机
1. **核心科学问题**：在多模块神经网络系统中，改变每个模块的证据可见性（evidence visibility）是否会改变梯度训练所发现的解类型与泛化性质？
2. **现有方法的盲区**：先前多智能体 LLM 系统研究缺乏对 latent inter-cell relay 的因果审计，无法区分"通道必要"与"语义可复用"；可见性遮蔽研究（如 CoFlow）未评估组合泛化。
3. **整程序查找捷径**：当可训练模块能访问完整程序时，整体查找（whole-program lookup）是可用的解，这会抑制可复用本地变换+通信的学习；限制可见性可因果性地切断该捷径。
4. **泛化稳定性缺口**：模块化系统对组合泛化的学习不稳定，且缺乏关于"信息可见性"如何影响解空间 basin 分布的系统性对照实验。

## 核心贡献（创新点）
1. **首次因果隔离的证据可见性配对实验**：10对社会共享完全相同的初始化字节、训练数据顺序、token布局、位置几何与参数，仅注意力掩码不同；首次在封闭组合任务上证明可见性对泛化的因果效应。（与前人工作的本质区别：之前研究多改变智能体数量或带宽，本文保持四细胞固定仅改可见性）
2. **大型且持久的组合泛化提升**：9/10对中限制性社会在深度2和深度3均超越全局社会≥20个百分点，中位数配对优势为0.7648（深度2）和0.6050（深度3）；在未见复合函数的程序上中位数优势仍达0.558。（与前人工作的本质区别：先前工作未报告碰撞分层后的泛化优势）
3. **值索引包移植的因果机制审计**：通过同值/反事实值包移植证明，限制性社会学会可互换的中间值表征（移植准确率0.94–1.00），而唯一高性能全局社会（init 204/order 954）的包不可互换（0.12–0.25）。（与前人工作的本质区别：先前 latent channel 审计多用破坏性消融，本文用语义索引的交换干预建立因果等价性）
4. **完整的预注册报告与开源**：报告两次预注册失败（完整联合门槛不通过），公开19/20个checkpoint及全部评估/审计脚本；建立严格的配对实验复现标准。（与前人工作的本质区别：强调失败案例与协议透明，而非仅报告成功）

## 方法详解
**社会架构**：4个细胞 + 1个读出口共享一个冻结的 Qwen2.5-0.5B-Instruct 骨干和一个 rank-8 LoRA（应用于注意力投影）；无细胞专属参数或身份。训练参数约4.32M。
- **细胞执行流程**：Cell 1 从起始值槽发布包 P₁（无入站邮件）；Cells 2–4 各消费前驱包一次并应用其私有操作或结构转发。
- **包读写投影**：两个共享无偏投影实现包的读取（将 incoming 归一化的 2×896 包映射为两个 pseudo-token embedding）和写入（将对应最终 hidden state 映射为包残差）；包更新规则为：
  $$P_{\text{out}} = \text{RMSNorm}(P_{\text{in}} + 0.05 \Delta P)$$
- **读出口**：`h_final = h_base + β W_mouth(P_4)`，其中 β=0.003 为校准后冻结的标量；冻结 LM head 产生输出。
- **标签 logit 残差化**：`z̃_j = z_j(h_final) - z_j(h_base)`（adapter 在 base 路径上关闭），用于任务评估；普通语言保留性单独在全词表 logit 上评估。

**任务与课程**：
- 在 Z_17 上的有序自然语言函数组合任务；12 个仿射双射构成算子集，程序长度0–3，答案编码为17个 tokenizer 验证的单 token 标签。
- 课程概率：恒等 episode 0.10，单操作 episode 0.25，两操作和三操作各 0.325。
- 评估使用 held-out 有序程序、held-out 表达法与复合函数分层。

**实验干预（唯一差异）**：
- 两种 arm 均接收相同的固定四槽 token 序列（相同 padding、位置、包槽、Transformer 调用）；受限臂每个细胞的注意力掩码仅暴露其分配的 evidence 槽 + 共享问题（境外槽直接注意力影响力为零），全局臂可 attend 全部四槽。
- 10对匹配社会（5种初始化 × 2种数据顺序），训练 20,000 updates；预注册门限在训练前固定。

**统计与分析**：
- 因每初始化复用两种数据顺序，推断不确定性按初始化分层聚类（5个 strata）；报告配对差、分层均值差、cluster-bootstrap 95%区间与 exact sign-flip 检验。
- 碰撞分层分析（map-novel vs map-redundant）为 post hoc 描述性检验。

## 实验与结果
**数据集与基准**：
- 任务：Z_17 仿射变换组合任务，12个算子，depth-2 评估集 12 programs（816 episodes/checkpoint），depth-3 评估集 60 programs（4,080 episodes/checkpoint）。
- 基线：10对 global 社会；3个 staged centralized-scan 模型；3个 atom-trained flat one-call comparators。
- 机会准确率：1/17 ≈ 0.0588。

**主要结果**（Table 2 & Table 3）：
- 9/10对中，限制性社会在 depth-2 和 depth-3 均超越全局社会 ≥0.20。
- 中位数配对优势：Δ₂=0.7648，Δ₃=0.6050；cluster-bootstrap 95%区间：[0.628, 0.784]（深度2）和 [0.397, 0.669]（深度3）；exact sign-flip p=0.0625（每深度）。
- 限制性社会 depth-2 中位数准确率达 0.8339，depth-3 为 0.6988（**略低于预注册 0.70 门槛**）。
- 全局社会 9/10 呈现"记忆-无泛化" profile：primitive 训练集完全拟合，depth-3 训练 bank 准确率 0.41–1.00，但 held-out 组合仅 0.05–0.11。
- **唯一例外**：global init 204/order 954 达到 depth-3 准确率 0.843，但同值包移植仅 0.12–0.25（非可互换）。

**通信必要性**：所有10个限制性社会切断全部包后准确率降至 chance（1/17=0.0588）；唯一高性能全局社会同样在删除包后崩溃。

**碰撞分层分析**（Table 4，9对完整）：
- 在复合仿射映射从未出现在训练中的程序上，中位数深度-3 优势仍为 +0.558；全局社会保持近 chance（0.049–0.094）。
- 说明"熟悉函数查找"无法解释可见性效应。

**包移植审计**（Table 5，6个受限社会 + 1个全局）：
- 受限社会：同值移植准确率 0.94–1.00；反事实值移植将输出导向数学预测答案（0.74–1.00）。
- 唯一高性能全局社会（204/954）：同值移植仅 0.12–0.25，证明其包表征强 episode/context 依赖。

**最强结果与提升幅度**：
- 限制性社会最佳 depth-3 准确率 0.9338（init 201/order 951），对应全局仅 0.0650，提升 0.8688。
- 整体中位数提升 0.6050（depth-3）。

## 相关工作脉络
1. **CoFlow (Zou et al., 2026)**：支持完整队友可见性与注意力掩码代理本地模式；本文定位差异——CoFlow 评估离线 MARL，未评估组合泛化与 causal packet audit。
2. **Emergent compositional communication (Kaszyński, 2026)**：研究离散组合通信，改变智能体数量；本文定位差异——保持四细胞架构、包带宽、初始化、训练流完全固定，仅改证据可见性。
3. **Modular systematic generalization (Lake & Baroni, 2018; Bahdanau et al., 2019; D'Amario et al., 2021)**：证明名义模块化不保证组合性；本文定位差异——聚焦"信息可见性"作为诱导偏置对解类型分布的影响。
4. **Masking as inductive bias (Peng et al., 2026a; Ma et al., 2026a)**：AC-VLA 的感知捷径遮蔽、Transformer masking priors；本文定位差异——上述为单策略遮蔽，本文研究多细胞 latent relay 中的证据可见性因果效应。
5. **Latent multi-agent LLM communication (Zhang & Emu, 2026; Cheng et al., 2026; Peng et al., 2026b)**：StateBridge 训练无关的隐状态对齐；本文定位差异——关注 learned protocol 的形成而非 cross-agent 兼容性。
6. **Causal abstraction (Geiger et al., 2021, 2024, 2025) 与 Hidden APIs (Ma et al., 2026b)**：前者对齐高维因果变量与神经状态，后者识别 monolithic LM 中的可复用接口；本文定位差异——在显式通信边界应用同标准，检验 "running Z_17 value" 这一已知中间变量的可互换性。

## 局限性与未来方向
1. **预注册失败**：完整电池门槛（深度3准确率≥0.70）以0.6988略败；早期 qualification 队列 0/10 通过完整门，表明系统在普通语言保留性上有约50–61个百分点的 top-1 回归。
2. **任务/架构范围有限**：仅测试单一 architecture family、templated synthetic task、封闭 paired task world 与固定20,000步预算；未在更多任务/更多成功全局模型上验证值索引接口的频率。
3. **非最小信道**：每个包含两个896维连续向量，信道并非信息论最小；仅6/10受限社会完成包审计。
4. **二次控制基线未配对**：staged centralized 模型未与四细胞社会配对且非 compute-matched。
5. **未来方向**：跨任务验证受限可见性对值索引接口形成概率的影响；探索最小信道与动态路由；研究自由形式语言推理与长上下文场景。

## 研究启发与可借鉴点
1. **值索引包移植作为机制审计工具**：通过构造同值/反事实值 donor packet 进行跨 episode 交换，可因果地鉴定 learned relay 的语义可互换性；可迁移至其他 multi-agent latent communication 研究。
2. **严格配对实验设计的通用范式**：仅改一个变量（注意力掩码），其余（初始化、数据顺序、token布局、位置几何、参数、计算）完全一致——该范式可应用于其他"信息结构"干预研究。
3. **受限可见性作为归纳偏置**：切断整程序查找捷径可促使模型学习可复用本地变换+通信的组合策略；可迁移至 modular LLM、agent society、neural module network 的组合泛化研究。
4. **预注册失败的正向价值**：完整报告失败案例与异常事件（provider 中断、checkpoint 恢复）增强可重复性；可借鉴其 incident ledger 与 bit-exact reproduction gate 机制。
5. **碰撞分层评估**：通过 stratify 复合函数是否出现在训练中来排除"函数查找"解释；该方法可迁移至其他组合泛化 benchmark 的评估设计。

## 关键术语表
**Restricted evidence visibility**：限制每个细胞只能 attend 其分配的 evidence 槽，境外信息只能通过 learned packet channel 到达——一种稳定的语义所有权边界。
**Compositional generalization**：对未见过算子序列与表达法的组合任务进行测试，衡量模型能否超越记忆而进行系统泛化。
**Value-indexed packet**：包中编码的 running 中间值（如 Z_17 累加值），不同 episode 的同值包可互换并维持下游行为。
**Causal interchange intervention**：将 donor episode 的包 transplant 到 recipient episode 的指定接口，测试因果等价性（同值 vs 反事实值）。
**Whole-program lookup**：当模块可访问完整程序时，模型可能直接记忆输出映射而非学习可复用变换；本文限制可见性切断此捷径。
**Shared-genome society**：多细胞系统共享同一冻结 backbone + 单一 LoRA，无细胞专属参数或身份；细胞角色由位置（stage）决定。
**Residualized logits**：`z_j(h_final) - z_j(h_base)`，去除 base 路径贡献后提取 packet-induced 的任务信号。
**Collision-stratified analysis**：按复合仿射映射是否在训练中出现分层评估，排除函数级查找策略的解释。

## 可复现要素
- **数据集**：合成 Z_17 仿射变换组合任务，operator/split/phrasing 三个 seed 预生成并哈希（operator seed 6011, split hash 94e506d408b6def1, grammar hash def14eb4182e2949）；任务 world 在训练前封闭。
- **代码开源**：是，GitHub https://github.com/tokenosopher/populus-evidence-partitioning
- **权重开源**：是，Hugging Face https://huggingface.co/tokenosopher/populus-evidence-partitioning-checkpoints（19/20 checkpoints，init 202/order 952 因 provider 实例丢失不可恢复）
- **关键超参**：Qwen2.5-0.5B-Instruct backbone + rank-8 LoRA；包维度 2×896；包更新步长 0.05；mouth 缩放 β=0.003；训练 20,000 updates；梯度累积 2×16；辅助分类器权重 anneal 至 update 10,000；确定性设置（TF32 禁用）；cross-entropy 损失。
