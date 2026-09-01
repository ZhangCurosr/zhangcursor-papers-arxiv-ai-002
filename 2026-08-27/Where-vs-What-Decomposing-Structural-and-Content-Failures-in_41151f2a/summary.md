---
title: "Where-vs-What-Decomposing-Structural-and-Content-Failures-in"
source: https://arxiv.org/pdf/2608.25358v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:00:33"
---

# 论文速读：Where-vs-What-Decomposing-Structural-and-Content-Failures-in

## 一句话总结
论文提出 **Structure-Content Decomposition (SCD)** 评估框架，将大模型生成结构化输出（JSON/表格）的失败模式解耦为“结构错位”与“内容错误”，系统揭示复杂度上升时结构保真度优先劣化的“剪刀差”现象，并据此设计 **SA-RLVR** 将分解指标转化为 GRPO 可验证奖励，使 JSON 值的放置准确率（VPA）从 0.26 跃升至 0.63。

## 研究问题与动机
- **现有评估的单一性缺陷**：主流基准（JSONSchemaBench、BFCL、SWE-bench 等）将结构化输出失败视为整体信号，无法区分“值正确但放错位置”与“值本身错误”两类本质不同的失败模式。
- **失败模式的隐性代价**：下游执行系统对结构路径极度敏感，即使所有目标值均出现在输出中，错位也会导致语义无效甚至执行崩溃。
- **底层机制不明**：随着递归深度或网格复杂度增加，模型是真正理解拓扑结构，还是依赖语义启发式？现有工作缺乏定量隔离分析。
- **优化信号缺失**：现有 RLVR 使用整体通过/失败奖励，缺乏针对结构寻址的细粒度、可程序化验证的梯度压力。

## 核心贡献（创新点）
1. **提出 SCD 三分层评估框架**，将格式有效性、Schema 合规率（SCR）与值放置准确率（VPA）严格解耦，首次定量揭示被整体指标掩盖的“右值左位”失败模式。
2. **发现“结构优先劣化”（Scissors Pattern）**，在 JSON 树结构与表格网格结构两种拓扑、六个 7B~前沿规模模型上一致验证：复杂度上升时 VP 维持高位而 VPA 骤降，错位率（DR）最高达 74%。
3. **提出 SA-RLVR 训练范式**，将 SCD 指标直接转化为 GRPO 的确定性奖励；匹配 SFT 基线仅提升 VPA 至 0.28，而 SA-RLVR 跃升至 0.63，且具备强跨 Schema 泛化能力。
4. **提供机制性消融证据**，证明模型依赖语义提示与路径歧义消解作为寻址捷径，而非拓扑理解；去除这些线索会显著推高 DR。

## 方法详解
- **SCD 三层级定义**：
  - Level 1：格式有效性 $V(y) \in \{0,1\}$，不可解析则跳过后续层级。
  - Level 2：Schema 合规率 $\mathrm{SCR} = |R \cap A| / |R|$，$R$ 为 Schema 要求路径集，$A$ 为模型输出实际叶子路径集。
  - Level 3：值放置准确率 $\mathrm{VPA} = \frac{1}{|\mathcal{P}|} \sum_{(p_i, v_i) \in \mathcal{P}} \mathcal{H}[y(p_i) = v_i]$，严格核对指定路径处的植入值。
- **诊断派生指标**：
  - 值存在率 $\mathrm{VP}$：植入值是否出现在输出任意叶子/单元格（忽略位置）。
  - 结构合规缺口 $\mathrm{SCG} = \mathrm{VP} - \mathrm{VPA}$。
  - 错位率 $\mathrm{DR} = 1 - \mathrm{VPA}/\mathrm{VP}$，直接量化“右值左位”比例。
- **双拓扑实例化**：
  - **JSON 域**：Schema-guided generation，使用递归 `TreeNode` 模板，固定拓扑仅变化递归深度（S/M/L 对应 depth 2/3/4），70% 植入值指向深层路径。
  - **表格域**：Row-level modification，给定 HTML 表格与目标字段，模型需定位并输出修改后的完整行，测试二维网格寻址。
- **SA-RLVR 奖励与训练**：
  - 奖励函数：$r(y) = 1.0 \cdot \mathrm{VPA}(y) + 0.3 \cdot \mathrm{SCR}(y)$，排除 VP 避免奖励错误放置行为。
  - 算法：GRPO，每 prompt 采样 $K=10$，组归一化优势 + KL 惩罚（$\beta=0.04$）。
  - 配置：Qwen2.5-7B-Instruct + LoRA（rank=64, alpha=128, ~40M 可训练参数），学习率 $1\times10^{-5}$，500 steps，4×A6000，混合 3400 条提示（JSON 合成+真实Schema+表格）。

## 实验与结果
- **数据集与基线**：自构造合成基准（JSON 1500 任务/模型，表格 4490 实例/600 核心任务），评估 6 个模型（GPT-4o、DeepSeek-V3/V4-Flash、Qwen3-8B、Qwen2.5-14B/7B），temperature=0.1。
- **核心现象**：“剪刀差”跨领域一致。JSON L 级复杂度下，DeepSeek-V4-Flash 错位率 DR=35.4%，Qwen2.5-7B 高达 73.8%；表格 L 级强模型 DR≈17-28%，弱模型 VP 与 VPA 双降（Qwen2.5-7B DR=98%）。
- **机制消融**：去除字段语义（改为 left/right）使 DR 上升 7.5pp；引入重复字段名使 DR 上升 8.2pp，证实模型依赖语义捷径。表格行距离消融显示 DR 从 35.8%（近行）升至 44.6%（远行）。
- **SA-RLVR 提升**：JSON-ID VPA 0.264→0.629（+138%），VP 0.310→0.869；OOD-Eco 达 0.858（+247%），OOD-JSB 达 0.968。表格 FmtOK 26.5%→85.5%，但坐标级 VPA 仍仅 ~0.06-0.09。匹配 SFT 基线 VPA 仅 0.281。
- **最强结果**：SA-RLVR 在 JSON 跨 Schema 泛化上表现最优（OOD-JSB VPA=0.968）；复合奖励（VPA+0.3*SCR）在 placements 与 schema 间取得最佳平衡。

## 相关工作脉络
1. **结构化输出基准**：JSONSchemaBench、BFCL、API-Bank 等报告整体合规或 exact-match，本文指出其将结构/内容失败混为一谈，SCD 填补细粒度诊断空白。
2. **位置/结构推理研究**：Liu et al. (2024) 发现长上下文检索的位置敏感性；本文将其延伸至输出生成端，首次定量隔离结构劣化曲线。
3. **约束解码方法**：Outlines、SGLang、Guidance 仅保证 Level 1 格式有效，本文证明其无法覆盖 Level 2-3 寻址，需显式优化。
4. **可验证奖励 RL**：DeepSeek-R1/GRPO 在数学与代码成功，但结构化输出领域此前缺乏细粒度可微/可计算奖励；SA-RLVR 首次实现结构寻址的在线强化。
5. **偏好优化方法**：DPO/ORPO 依赖人工或模型偏好标签，缺乏结构特异性梯度；本文提供程序化、确定性的逐层级反馈。

## 局限性与未来方向
- 实验基于受控合成数据，缺乏真实场景中的噪声指令、模糊 Schema、语义等价输出及下游执行标准。
- SCD 假设目标值与预期位置已知，难以直接应用于开放式生成或多结构等价任务。
- 训练规模有限（仅 7B 模型、500 steps、JSON 主导），表格跨拓扑迁移效果有限，学习曲线未 plateau。
- 未验证更大模型规模、多种子设置及真实 Agent/Tool-calling 部署，需后续大规模实验验证。

## 研究启发与可借鉴点
1. **评估解耦范式可迁移**：将“值正确性”与“位置/坐标正确性”分离的评估思路，可直接复用到代码生成、Agent Tool Calling、表格问答等坐标敏感任务。
2. **连续部分积分奖励设计**：用 VPA/SCR 替代二元 EM 作为 RL 奖励，避免稀疏反馈；该“partial credit”机制可推广至任何可程序化验证的 RLHF/RLVR 场景。
3. **机制归因消融设计**：通过控制语义线索与路径歧义，定量证明模型依赖启发式而非拓扑理解，为 mechanistic interpretability 提供可操作的实验模板。
4. **严格对照基线**：Best-of-K SFT 作为 RL 的对照，隔离了“数据质量”与“优化范式”的差异，实验设计严谨，值得在强化学习论文中复用。
5. **混合拓扑训练策略**：虽表格 VPA 提升有限，但格式合规显著改善，提示在资源受限时将多拓扑数据混合可作为低成本提升结构化稳定性的实用技巧。

## 关键术语表
- **SCD (Structure-Content Decomposition)**：将结构化输出质量解耦为格式有效性、Schema 合规率与值放置准确率的三层评估框架。
- **VPA (Value Placement Accuracy)**：衡量植入值是否严格出现在指定结构路径或网格坐标的准确率。
- **VP (Value Presence)**：衡量植入值是否出现在输出任意位置（忽略结构坐标）的准确率。
- **DR (Displacement Rate)**：错位率，$1 - \mathrm{VPA}/\mathrm{VP}$，直接量化“右值左位”失败比例。
- **SA-RLVR**：Structure-Aware Reinforcement Learning with Verifiable Rewards，基于 SCD 指标构建奖励的 GRPO 在线强化学习方法。
- **剪刀差 (Scissors Pattern)**：复杂度上升时 VP 维持高位而 VPA 急剧下降的现象，揭示结构寻址是独立于内容生成的瓶颈。
- **Semantic Shortcuts**：模型依赖字段名称语义或位置近似进行寻址的启发式策略，而非真正的拓扑结构理解。
- **SCR (Schema Compliance Rate)**：模型输出实际路径与 Schema 要求路径的交集比例，衡量结构完整性。

## 可复现要素
- **数据集**：完全自构造合成数据（JSON 基于递归 `TreeNode` 模板，表格基于 53 个真实模板程序化生成）；论文未声明开源，但附录 A/B 提供了完整的任务构造与解析器伪代码。
- **代码/权重**：论文未公开代码仓库与训练权重，仅附录提供评估逻辑 sketch 与 prompt 模板。
- **关键超参**：LoRA rank=64, alpha=128, 可训练参数≈40M；GRPO
