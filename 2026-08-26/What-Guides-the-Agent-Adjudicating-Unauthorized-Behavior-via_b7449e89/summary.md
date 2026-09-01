---
title: "What-Guides-the-Agent-Adjudicating-Unauthorized-Behavior-via"
source: https://arxiv.org/pdf/2608.24022v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:18:08"
field: "LLM Agent安全与可解释性"
keywords: ["LLM Agent安全", "提示注入防御", "注意力归因", "行为指导指令定位", "运行时监控"]
innovations: ["将行为指导指令定位建模为注意力矩阵上的1-D目标检测任务", "提出sink-aware正则化抑制注意力噪声", "定位与裁决解耦支持策略适配的运行时防御框架"]
benchmarks: ["MCPTox", "InjecAgent"]
---

# 论文速读：What-Guides-the-Agent-Adjudicating-Unauthorized-Behavior-via

## 一句话总结
本文提出 AttnLocate，一种运行时监控框架，通过将行为指导指令定位建模为注意力矩阵上的目标检测任务，实现对外部注入内容的细粒度定位与基于来源权限的动态裁决，有效防御间接提示注入和工具投毒攻击。

## 研究问题与动机
- **现有防御方法的粒度不足**：静态扫描和架构隔离依赖检测或过滤明显恶意内容，无法识别伪装成正常内容但实际被模型解释为行为指导指令的注入片段。
- **归因方法的粒度过粗**：已有归因工作（如 MindGuard、TracLLM）在段落或元数据字段级别估算影响力，将短恶意指令与其良性载体混淆，无法精确定位功能上被解释为指令的上下文片段。
- **缺乏动态裁决能力**：现有方法难以根据可配置的安全策略对不同来源的外部上下文进行差异化的权限评估与未授权行为裁决。
- **注意力信号的可利用性**：研究表明，真正影响决策的 token 在注意力矩阵中表现出区别于一般上下文显著性或注意力 Sink 的独特激活模式，为细粒度定位提供内部信号基础。

## 核心贡献（创新点）
- **将未授权行为检测重新 formulation 为动态指令定位问题**：与已有方法通过粗粒度单元（段落/字段）估算影响力不同，本文直接在 token 级别定位真正引导工具调用决策的上下文片段。
- **提出注意力聚合-检测-裁决的三阶段框架**：通过多头多层注意力聚合构建 token 级特征空间，结合 1-D U-Net 与 anchor-free 检测头实现变长 span 边界预测，最后基于来源权限进行策略适配的裁决，而非依赖攻击特定的词法模式。
- **设计 sink-aware 正则化机制抑制注意力噪声**：通过为高注意力背景位置分配更高训练权重，显式训练模型区分注意力 Sink 的伪激活与真实的行为指导指令，显著降低误报率。
- **实现跨模型零样本迁移与无重训策略适配**：AttnLocate 在不同 LLM 架构间保持有效的定位与裁决性能，且支持通过调整权限策略适应不同部署场景而无需重新训练。

## 方法详解
**注意力聚合模块（Attention Aggregation）**
- **Head 聚合**：对每层 $l$ 的所有头 $h$ 的注意力矩阵取平均，得到 $\bar{A}^{(l)} = \frac{1}{H}\sum_{h=1}^{H} A^{(l,h)}$，整合不同头捕获的互补依赖模式。
- **Gaussian 层加权**：使用以 $\mu \approx 2L/3$ 为中心、$\sigma = L/6$ 的正态分布权重聚合各层矩阵，赋予中层与高层任务相关注意力更高权重。
- **决策条件切片**：仅提取与最终决策 $y$ 及外部上下文 $C_{external}$ 相关的注意力子矩阵 $A_y = A[P(y), P(C_{external})]$，支持 CoT 范式。
- **特征提取**：对每个上下文位置 $j$，计算 $A_y[:,j]$ 的均值、最大值、标准差及学习型注意力池化操作，得到 token 级特征向量序列 $Z$。

**指令定位模块（Instruction Localization）**
- **1-D U-Net 骨干**：Encoder 通过步长为 2 的卷积逐步扩大通道数、降低时间分辨率，捕获远距离上下文依赖；Decoder 通过跳跃连接融合粗粒度特征并恢复 token 级分辨率以精确还原边界。
- **Anchor-free 检测头**：基于 FCOS 设计，每个位置 $i$ 预测对象性 logit $o_i$、左右边界偏移 $(l_i, r_i)$ 及 centerness 分数 $c_i$，样本级头输出标量 logit $s$ 判断是否存在行为指导 span。推理时当 $\sigma(s) < \delta$ 时返回空集。
- **Sink-aware 正则化**：定义 sink 集合 $S_{sink} = \{i | b_i = 0, \bar{a}_i > \beta/T_x\}$，为高注意力背景位置分配权重 $\alpha_{sink} = 3$，结合 focal modulation 抑制注意力 Sink 导致的误报。
- **训练目标**：$\mathcal{I} = \mathcal{I}_{focal} + \lambda_{giou}\mathcal{I}_{giou} + \lambda_{cls}\mathcal{I}_{bce}$，其中 $\lambda_{cls}=1, \lambda_{giou}=2$。

**未授权行为裁决模块（Unauthorized Behavior Adjudication）**
- **溯源解析**：利用运行时上下文的结构化分隔符与溯源标注构建映射表 $\Gamma_t$，将定位到的 span 关联到其来源提供者 $\rho(\hat{\tau}_t)$。
- **权限仲裁器**：可配置策略 $\Pi$ 为每个提供者分配权限级别，裁决规则为：当 $\hat{\tau}_t \neq \emptyset$ 且 $\Pi(\rho(\hat{\tau}_t)) < R_y$（执行决策所需的最小权限）时判定为恶意。

## 实验与结果
- **数据集**：MCPTox（工具投毒攻击）、InjecAgent（间接提示注入攻击）。
- **模型覆盖**：6 个 LLM 家族（Qwen、DeepSeek、Phi、LLaMA、Mistral、Gemma），10 个 agent 配置，涵盖标准与 thinking 模式。
- **定位性能**：平均 mIoU 为 0.743，Hit@0.5 与 Hit@0.7 表现优异；thinking 模式下 Qwen3 提升约 5.4%。
- **裁决性能**：平均 AUROC 为 0.956，在 FPR=0.067 时 TPR 达 0.934；Gemma2-9B 在 InjecAgent 上 AUROC 最高达 0.989，TPR 为 0.984。
- **跨模型泛化**：在 Qwen3-8B† 上训练后零样本迁移至 DeepSeek-R1-Qwen-14B† 与 Phi-4†，AUROC 分别达 0.921 与 0.859，Hit@0.5 保持在 0.90 附近。
- **对比基线**：显著超越静态扫描（LLM-Guard、LLM Detector）、行为审计（MCIP）及归因方法（MindGuard、TracLLM），在 Phi-4† 上达到最高 TPR（0.947）与最低 FPR（0.070）。
- **消融实验**：Gaussian 加权中心 $\mu=2L/3$ 最优；sink-aware 正则化将 FPR 从 0.417 降至 0.056；权限策略调整（Principal-only → Tool-authorized → Result-authorized）保持 TPR>0.909。

## 相关工作脉络
- **MindGuard**（Wang et al. 2026b）：将工具调用归因于元数据字段，但粒度停留在字段级别，无法定位字段内真正起指导作用的子 span；AttnLocate 通过 token 级定位实现更细粒度的溯源。
- **TracLLM**（Wang et al. 2025c）：通过扰动估计长上下文影响力，计算成本高且精度受限于预设粒度；AttnLocate 利用注意力聚合直接建模依赖关系，效率更高且支持变长 span 检测。
- **AttnTrace**（Wang et al. 2026a）：通过注意力聚合与上下文采样支持提示注入归因，但未解决注意力 Sink 噪声与边界不确定的问题；AttnLocate 引入 sink-aware 正则化与 anchor-free 检测头针对性改进。
- **MCIP**（Jing et al. 2025）：行为审计方法通过上下文完整性约束限制不安全动作，但依赖显式策略且无法定位具体指导内容；AttnLocate 的裁决机制与定位解耦，支持动态策略适配。
- **IsolateGPT**（Wu et al. 2025）与**Defeating Prompt Injections by Design**（Debenedetti et al. 2025）：架构级隔离方法通过系统层分离控制流与数据流；AttnLocate 作为运行时监测框架无需修改 Agent 架构，兼容现有系统。
- **ContextCite**（Cohen-Wang et al. 2024）：通过扰动归因模型生成到上下文，但未针对 Agent 的工具调用决策设计；AttnLocate 专门面向工具调用场景，结合权限策略实现行为裁决。

## 局限性与未来方向
- **白盒访问依赖**：需要获取底层模型的注意力矩阵，限制了在闭源 API 场景下的直接部署。
- **长上下文性能衰减**：随着输入长度增加（>2048 token），mIoU 从 0.81 降至 0.70，TPR 同步下降，显著性信号被稀释。
- **跨模型迁移性能下降**：零样本迁移时 AUROC 较同模型训练下降约 0.1，需进一步探索跨架构对齐策略。
- **未覆盖的防御场景**：主要针对间接提示注入与工具投毒，对其他攻击类型（如数据投毒、模型窃取）的适用性未验证。
- **未来方向**：探索自监督或弱监督的注意力特征学习以减少标注依赖；设计上下文长度无关的归一化机制；扩展至多模态 Agent 与分布式 Agent 系统。

## 研究启发与可借鉴点
- **注意力矩阵的目标检测建模**：将行为指导指令定位转化为 1-D 目标检测问题，利用 anchor-free 头处理变长 span 的思路可迁移至其他细粒度归因任务（如思维链步骤溯源、知识注入检测）。
- **Sink-aware 正则化机制**：通过高注意力背景加权抑制噪声的思路可推广至任意基于注意力的归因方法，提升鲁棒性。
- **定位-裁决解耦设计**：将"识别指导内容"与"评估来源权限"分离，使定位模块通用化、裁决模块策略化，这种模块化设计支持灵活适配不同安全需求。
- **Gaussian 层加权策略**：以 2L/3 为中心的加权方案可有效整合多层注意力信号，该设计可复用于其他需要跨层聚合的模型解释任务。
- **跨模型零样本迁移可行性**：证明注意力模式在不同架构间具有可比性，为构建通用注意力分析模块提供了理论依据。

## 关键术语表
**Behavior-Guiding Instruction**：外部上下文中被 LLM 实际解释为行为指导的片段，即使表面看起来是良性的，也能影响模型的工具调用决策。
**Attention Sink**：注意力矩阵中某些位置因位置偏差或频繁出现而获得异常高注意力权重，但与决策无功能关联的背景噪声。
**Authority Arbiter**：基于可配置策略 $\Pi$ 的裁决组件，根据来源提供者的权限级别与决策所需最小权限的比较结果判定行为是否未授权。
**mIoU (mean Intersection-over-Union)**：定位精度指标，衡量预测 span 与真实 span 的重叠程度，值越高表示边界定位越精确。
**Sink-Aware Regularization**：为高注意力背景位置分配更大训练权重的正则化技术，显式训练模型区分注意力 Sink 与真实行为指导 span。
**Authority Policy**：配置不同来源提供者的权限级别的安全策略，如 Principal-only（仅系统与用户）、Tool-authorized（含工具指令）、Result-authorized（含结果来源）等层级。
**Anchor-Free Detection**：不依赖预设锚框的目标检测设计，直接预测边界偏移与对象性分数，更适合变长 span 检测任务。
**Think Mode**：LLM 启用扩展推理/思维链的模式，实验表明 AttnLocate 在该模式下面板定位性能提升约 5.4%。

## 可复现要素
- **数据集**：MCPTox（https://github.com/...，论文引用 Wang et al. 2026c）、InjecAgent（https://github.com/...，论文引用 Zhan et al. 2024）；数据集未开源声明。
- **代码**：论文未提及代码是否开源。
- **权重**：论文未提及预训练权重是否提供。
- **关键超参**：$\mu = 2L/3$，$\sigma = L/6$，$\beta = 10$，$\alpha_{sink} = 3$，$\lambda_{cls} = 1$，$\lambda_{giou} = 2$，检测头阈值 $\delta$（未明确数值）。
- **模型架构**：深度 4 的 1-D U-Net，基于 FCOS 的 anchor-free 检测头。
