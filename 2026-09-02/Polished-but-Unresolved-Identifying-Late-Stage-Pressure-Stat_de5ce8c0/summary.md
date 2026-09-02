---
title: "Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat"
source: https://arxiv.org/pdf/2609.00823v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:26:59"
field: "Agent 可靠性与表征工程"
keywords: ["late-stage pressure", "activation engineering", "long-horizon agents", "tool-use", "probe", "representation learning"]
innovations: ["提出 late-stage pressure state 并证明其在 hidden space 线性可分（AUROC=0.916）", "构建两级 PSPR 干预机制（activation relief + explicit organization）在线缓解过早闭合", "识别 constraint clarity 与 action mapping 为压力缓解的关键因素"]
benchmarks: ["DeepPlanning-Travel", "DeepPlanning-Shop", "TravelPlanner"]
---

# 论文速读：Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat

## 一句话总结
本文识别并建模了长 horizon tool-use agents 中的**晚期压力状态**（late-stage pressure state）——agent 在关键约束未解决时仍倾向于提交外观完整的最终答案；通过线性 probe 可检测该状态，并提出轻量化插件 PSPR 在提交前实时缓解压力，在多个长 horizon 规划基准上稳定提升最终答案质量。

## 研究问题与动机
- **核心现象**：长 horizon tool-use agents 常出现"看似完整但关键约束未满足"的提交（polished but unresolved），用户仅看到最终输出而难以察觉轨迹层面的缺陷。
- **现有工作不足**：已有研究（Ko et al., 2026; Wang et al., 2026b; Yu et al., 2026）从行为层面诊断过早闭合、证据衰减等问题，但未回答：在提交附近是否存在一个可测量、行为有意义的**内部状态**，驱动 agent 在约束未解时仍偏向闭合。
- **检测缺口**：缺乏对晚期压力状态的隐层表示识别与干预手段，无法在推理时在线感知并缓解该风险。
- **应用需求**：长 horizon 任务（旅行规划、购物规划等）要求 agent 持续跟踪多个交互约束并在充分验证后才提交， premature closure 会严重影响实用性。

## 核心贡献（创新点）
1. **识别并操作化 late-stage pressure state**：首次证明在 hidden space 中，agent 在"完整但约束未满足"提交边界处存在线性可分的内部压力状态（AUROC=0.916），而非仅是行为标签。
2. **揭示压力缓解的两个关键因素**：通过受控上下文操纵发现，constraint clarity（明确当前验证状态）与 action mapping（将未解约束映射为下一步动作）可显著降低压力并提升定向修复率（Combined: pressure 0.13, repair 0.85）。
3. **提出 PSPR 轻量化干预框架**：基于 probe 的在线压力感知 + 两级干预策略（中等压力用 activation relief direction，高压力用 explicit state organization），无需额外训练即插即用。
4. **跨模型/跨基准的普适性验证**：在 Qwen3-14B/32B、OLMo-3.1-32B 及 DeepPlanning-Travel、DeepPlanning-Shop、TravelPlanner 三个基准上，PSPR 稳定提升 CS/PS/CP 指标。

## 方法详解
- **Probe 构建**：在 action boundary（prefill 结束后、下一 action 开始前）取第一个 token 的 hidden state $h_T$，构造三分类对比：
  - PresC（正例）：$S_{sat} \leq 0.3$ 且 $DPS \geq 4$（约束未满足但提交外观完整）
  - HealC（负例1）：$S_{sat} \geq 0.85$ 且 $DPS \geq 4$（完整且约束满足，排除单纯"提交形式"信号）
  - ProdC（负例2）：后续两步继续解决未解约束（排除"困难度/未完成"混淆）
  - 训练线性分类器，AUROC=0.916，PR-AUC=0.921。
- **Activation Direction 构造**：采用 contrastive activation addition（CAA），在选定层 $\ell$ 对前 $K$ 个生成 token 的 hidden state 取均值：
  - $v_{\text{PresC-HealC}} = \mu(\text{PresC}) - \mu(\text{HealC})$
  - $v_{\text{PresC-ProdC}}$ 进一步 residualize 掉通用 commitment 方向：$v^{\perp \text{commit}} = v - \text{Proj}_{v_{\text{commit}}}(v)$
- **PSPR 两级干预机制**：
  - 阈值 $a=0.4, b=0.65$，压力分数 $s_t = \text{Probe}(h_{t,1}^{(\ell)})$
  - 中等压力（$a \leq s_t < b$）：施加 relief direction $v_{\text{relief}} = \mu_{LP} - \mu_{HP}$（HP 来自原始 PresC 节点，LP 来自 oracle 增强后的 ProdC 节点）
  - 高压力（$s_t \geq b$）：执行 explicit state organization，要求模型自行总结约束状态并规划下一步，插入 prefill 后继续生成
- **干预窗口**：在选定层对当前 action 的前 $K=5$（干预实验）或 $K=10$（PSPR）token 的 hidden state 加减方向向量，幅度 $|\alpha|=1$。

## 实验与结果
- **数据集**：DeepPlanning-Travel（120 个旅行规划任务，含 location/budget/user 硬约束）；泛化验证使用 DeepPlanning-Shop（100 实例）和 TravelPlanner（120 实例）。
- **基线方法**：CoT、ReAct、Reflexion₁（一次反思重试）。
- **主要结果**（Table 4，pass@3）：
  - Qwen3-14B + ReAct：CP 从 26.1→29.1（+3.0），CS 从 41.5→45.6（+4.1）
  - Qwen3-32B + ReAct：CP 从 29.3→33.2（+3.9），CS 从 45.3→50.5（+5.2）
  - Qwen3-32B + CoT：CP 从 25.2→28.1（+2.9）
- **泛化结果**（Table 5）：DeepPlanning-Shop 的 Acc 从 19.0→23.0（+4.0）；TravelPlanner 的 DR/CS/HC 全提升。
- **消融**（Table 6）：Relief direction（CP +0.5）、Explicit organization（CP +2.0），两者组合最优；Random/Periodic trigger 均低于 PSPR，说明压力信号驱动的时序干预关键。
- **成本**（Table 8）：平均 turns 从 4.62→7.94，tool calls 从 7.57→8.95，压力分数显著下降（Relief: $\Delta p = -0.31$；Explicit: $\Delta p = -0.57$）。
- **最强结果**：Qwen3-32B + ReAct + PSPR 在 DeepPlanning-Travel 上 CP=33.2，较 ReAct 基线提升 13.3%。

## 相关工作脉络
- **Long-Horizon Tool-Use Failures**（Ma et al., 2024; Xie et al., 2024; Garikaparthi et al., 2026）：关注 agent 在长轨迹中偏离任务、证据积累不足、资源分配失当等行为层面诊断；本文聚焦"表面完整但约束未满足"这一特定失败模式，并从隐层表示角度解释。
- **Late-Stage Constraint Failure**（Ko et al., 2026; Wang et al., 2026b; Yu et al., 2026; Fang et al., 2026）：分别从 illusory completion、早期错误放大、证据衰减、推理预算控制等角度分析；本文补充了"压力状态"这一表示层机制，并与行为诊断形成互补。
- **Activation Engineering**（Rimsky et al., 2024; Højer et al., 2025; Stolfo et al., 2025）：CAA 及后续工作用于指令遵循、诚实性引导；本文首次将其应用于长 horizon tool-use 中的"过早闭合"压力状态识别与干预。
- **Semantic Entropy Probes / Layer-wise Representation**（Kossen et al., 2024; Skean et al., 2025; Jin et al., 2025）：探索 LLM 内部表示的可读性；本文在其基础上针对特定行为状态（late-stage pressure）设计 probe 并验证行为因果性。
- **Function Vectors / Concept Erasure**（Todd et al., 2024; Belrose et al., 2023）：线性概念擦除与功能向量；本文借鉴 residualize 技术分离压力方向与通用 commitment 方向。

## 局限性与未来方向
- **适用范围受限**：主要验证于文本型、结构化、可验证的长 horizon tool-use 场景；扩展至开放式、动态变化或多模态 agent 环境尚待研究。
- **非因果唯一解释**：作者明确承认 late-stage pressure 并非 premature closure 的唯一或充分原因，早期规划错误、证据缺失、上下文丢失、工具错误等同样重要。
- **需要隐层访问**：PSPR 依赖对 open-weight 模型 hidden states 的访问，不适用于纯 API 黑盒场景。
- **离线标注成本**：probe 与 activation direction 的构建需额外 rollout 与标注（Gemini/GPT 辅助），引入一定开销。
- **阈值需经验设定**：$a=0.4, b=0.65$ 基于少量预实验固定，未做系统性扫描。

## 研究启发与可借鉴点
1. **Probe 驱动的在线监控范式**：将离线训练的线性 probe 作为在线控制信号，实现"感知-干预"闭环，可迁移至其他 agent 失败模式（如 hallucination、inconsistent planning）的实时检测。
2. **双阶段干预设计**：轻度用 activation steering、重度用显式 prompt organization 的分层策略，兼顾效率与效果，适合资源受限的部署场景。
3. **负例构造的对比学习思路**：HealC 排除"提交形式"混淆、ProdC 排除"困难度"混淆，这种多对照 design 值得在其他 representation learning 工作中借鉴。
4. **与主流 agent 框架的兼容性**：PSPR 作为 plugin 可叠加于 CoT/ReAct/Reflexion，说明表示层干预可与推理/工具使用层方法正交互补。
5. **Attention 分析验证干预有效性**：通过计算 appended span 的 attention mass ratio（1.6×），证实模型确实使用了补充信息，为后续介入式设计提供验证范式。

## 关键术语表
**Late-stage pressure state**：agent 在长 horizon 任务末期出现的内部状态，偏向提前闭合提交而非继续验证，即使关键约束仍未满足。
**PresC (Pressure-driven commit)**：正例标签，指约束满足度低（$S_{sat} \leq 0.3$）但交付外观完整（$DPS \geq 4$）的提交边界。
**DPS (Delivery Polish Score)**：由 Gemini 3.1 Pro Preview 评估的最终答案"完整性/可用性"评分（1-5 分），仅基于文本外观不校验内容。
**$S_{sat}$ (Boundary-level satisfaction score)**：基于 query、hard constraints 和当前可用证据计算的边界级约束满足度（0-1），用于区分"真满足"与"假闭合"。
**Activation intervention / CAA**：在选定层的 hidden state 上加减 contrastive direction（正例均值减负例均值），以线性方式引导模型行为。
**PSPR (Probe-Sensed Pressure Relief)**：本文提出的轻量化 plugin，通过 probe 在线监测压力分数，在中等/高压力下分别施加 activation relief 或显式状态组织干预。
**Valid $p_{next}$**：干预后下一个 action boundary 处的 probe 压力分数（排除立即终止的轨迹），用于评估干预的延迟效果。
**Pass@3**：每个 task 采样 3 次轨迹，取最优官方评分；用于缓解长 horizon 任务中单轮不稳定导致的评估噪声。

## 可复现要素
- **数据集**：DeepPlanning-Travel（公开）、DeepPlanning-Shop（公开）、TravelPlanner（公开）；论文声明为 permissive license。
- **代码/权重**：论文未提供开源代码仓库链接；使用 Qwen3-14B/32B、OLMo-3.1-32B-Instruct 等开源 backbone。
- **关键超参**：$K=5$（干预实验）/$K=10$（PSPR）、$|\alpha|=1$、$a=0.4$、$b=0.65$、temperature=0.6、layer=26（Qwen3-14B）/layer=46（Qwen3-32B）。
- **标注工具**：Gemini 3.1 Pro Preview（DPS、$S_{sat}$、ProdC 标签）；人工审计 50 例/类，LLM 审计 100 例/类。
- **训练数据量**：Probe 训练集 300 PresC + 110 HealC + 135 ProdC；测试集 80 PresC + 40 HealC + 40 ProdC。
