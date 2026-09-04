---
title: "Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica"
source: https://arxiv.org/pdf/2609.03450v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-04 15:32:38"
field: "LLM指令跟随与验证行为"
keywords: ["plan-pointer", "budgeted verification", "model specificity", "instruction following", "reproducible experiment"]
innovations: ["首次系统量化多模型指针跟随行为的模型特异性", "证明单字符ACTIVE PLAN ID指针可达到93-96%自然语言效力", "建立预注册+SHA256冻结文本的大规模对照实验框架"]
benchmarks: ["V73命中", "Share效应", "Study B'复制验证"]
---

# 论文速读：Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica

## 一句话总结
本文对多个 LLM 在预算化验证任务中的**信用指针（plan-pointer）与记录指令跟随行为**进行大规模对照实验，揭示了模型特异性远超指令形式差异、且 GPT-5.6 Sol 在多场景中表现异常稳健的核心发现。

## 研究问题与动机
- 不同 LLM 对同一意图的**验证指令引用（id / criterion / composite）**遵循比例差异极大，无法归纳为统一的"模型属性"，需系统量化。
- 验证指令是否等同于"可随意设置的系统侧开关"？即同一 referent 用不同编码方式会产生不同分配，指令形式本身是否具有稳定可迁移的效力。
- **指针（pointer）能否替代自然语言句子**驱动计划分配？一个字符的 `ACTIVE PLAN ID` 能否达到自然语言指令的大部分效果（共享效应≈0.93–0.96）。
- 现有研究缺乏跨模型、跨措辞变体、跨场景的系统性对照，难以区分"模型能力"与"指令形式"对决策的相对贡献。

## 核心贡献（创新点）
1. **首次系统性量化多模型指针跟随行为的模型特异性**：揭示了相同意图引用在不同模型间差异可达数十个百分点，且方向不一致，打破"统一模型属性"假设。
2. **证明指针效力接近自然语言句子**：`ACTIVE PLAN ID` 单字符指针能驱动约 93–96% 的自然语言计划分配效果，为低成本指令压缩提供实证基础。
3. **建立预注册的大规模对照实验框架（Study B′复制验证）**：1,200 episodes、6 模型、percentile block bootstrap，share=0.957，被判定为 POINTER-STRONG 且 REPLICATED。
4. **揭示后缀效应（suffix effects）存在显著模型差异**：Opus 5 / Fable 系列在多数 P 条件下显示 Exceeds Margin 负向后缀效应，而 Sonnet 5 / GPT-5.6 Sol 基本落在 Within Margin。
5. **提出精确冻结文本与 SHA256 前缀控制方法**：五个 Criterion Wording 变体通过冻结字节级文本确保实验可复现性。

## 方法详解
- **实验设计**：多模型（Opus 5、Fable 5、Fable 5.1、Sonnet 5、Haiku 4.5、GPT-5.6 Sol/Terra/Luna）在 Budgeted Verification 任务上测试多种 arm、措辞变体及跨场景迁移。
- **统计方法**：以 **Newcombe (1998) method 10 配对得分 95% CI** 为主，辅以 Wilson 分数区间和百分位 bootstrap（B=4,000），设置 δ=10 的 Margin 阈值判断 Exceeds/Within/Undetermined。
- **关键指标 V₇₃**：byte-controlled arms 命中计数（40/40 格式），用于衡量指针/指令遵循率。
- **Study H2 后缀效应设计**：在文本末尾追加锁定后缀 `' (memory 73)'`，共 5 个精确冻结的 Criterion Wording（P0–P4），SHA256 前缀严格锁定（如 P0: `ca394238`，P1: `b12a4173` 等）。
- **共享效应（share）计算**：指针跟随与自然语言句子跟随的共享比例，>0.90 阈值判定为 POINTER-STRONG。
- **Study I 设计**：160 个注册 cells，72 个 contrasts，n=25/cell，含 valid / superseded 两个 world，测量 Y₁（遵循档案记录做决策）及 V₇₃ turn-1 命中。

## 实验与结果
- **Study G（V₇₃ 命中，byte-controlled arms）**：
  - **GPT-5.6 Sol 全面强势**：crit、both_IL、both_fresh、both_inherited 均为 **40/40 [91.2, 100.0]%**。
  - **Opus 5**：crit = 40/40；both_IL = 仅 3/40 [2.6, 19.9]%；crit_pad = 40/40；crit_other = **0/40**；both_fresh = 13/40；both_inherited = 0/40。
  - **Fable 5 / Fable 5.1**：除 crit（Fable 5 = 40/40；Fable 5.1 = 14/40）外，几乎所有 other/IL 条件为 **0/40**。
- **Study B′（预注册复制验证）**：
  - Δ_plan = **+81.7 Exceeds Margin**，Δ_bridge = **+85.3 Exceeds Margin**，share = **0.957 Exceeds Margin（POINTER-STRONG；REPLICATED）**。
- **Study H2 后缀效应**：
  - Opus 5 / Fable 5 / Fable 5.1 在 P0–P4 多数条件显示显著负向后缀效应（Exceeds Margin）；Sonnet 5 与 GPT-5.6 Sol 几乎全部 Within Margin。
- **最强结果**：GPT-5.6 Sol 在所有测量条件下表现最稳定，多数 contrast 为 +0.0 Within Margin；share=0.957 达到 POINTER-STRONG 阈值。

## 相关工作脉络
1. **前作指针跟随研究**：已有工作关注 natural language instruction following，但本文首次在 byte-controlled 层面量化 pointer vs. natural language 共享效应。
2. **预算化验证任务基线**：此前工作缺乏跨模型系统对比，本文填补该空白，建立标准化的对照实验框架。
3. **Newcombe 配对区间方法**：本文采用该方法替代传统 t 检验，更适合二值命中数据的配对比较。
4. **预注册实验设计**：Study B′ 复制验证遵循预注册科学规范，区别于事后分析型论文。
5. **SHA256 文本锁定技术**：精确冻结文本与前缀哈希确保跨实验室复现，相关方法在 NLP 实验控制中较新。
6. **模型特异性差异研究**：本文揭示了指针效力模型特异性远超指令形式差异，与此前"统一指令响应"假设形成对立。

## 局限性与未来方向
- 实验主要聚焦 6 个模型，未覆盖更多架构（如小参数模型、多模态模型），结论外推性有限。
- 后缀效应研究仅针对单一后缀 `' (memory 73)'`，不同后缀形式的通用性未验证。
- Study I 中 Opus 5 在 valid world 下 crit−none = 0.0（不显著），superseded world 下 +100.0，valid/superseded 差异机制需进一步解释。
- 共享效应≈0.93–0.96 虽高但非 1.0，指针 vs. 自然语言剩余差异的理论解释不足。
- 未来可扩展至更多模型族、更多后缀变体，并探索指针压缩的理论边界。

## 研究启发与可借鉴点
1. **精确文本冻结方法可直接复用**：SHA256 前缀锁定 + artifacts/FIELD BLOCKS.json 的管理方式适用于任何需要严格控制的 LLM 实验。
2. **Newcombe method 10 配对 CI 值得在本团队研究中采用**：相比传统方法更适合二值命中数据的精度比较。
3. **预注册+复制验证框架可借鉴**：Study B′ 的 reproducible 声明方式提升了结论可信度，可作为后续研究的模板。
4. **指针压缩思路可迁移到 prompt engineering**：单字符 `ACTIVE PLAN ID` 接近自然语言效果，为低成本指令设计提供方向。
5. **模型特异性量化方法可扩展**：同一意图引用跨模型差异的量化框架可用于评测其他 LLM 行为一致性。

## 关键术语表
- **V₇₃**：byte-controlled arms 的命中指标，衡量指针/指令遵循率（40/40 格式）。
- **POINTER-STRONG**：共享效应（share）>0.90 的判定标准，表示指针效力接近自然语言句子。
- **Exceeds Margin / Within Margin / Undetermined**：基于 δ=10 阈值的三种统计判定等级。
- **SHARE（共享效应）**：指针跟随与自然语言句子跟随的共享比例，量化两者效力重叠度。
- **Criterion Wording**：精确冻结的准则文本变体（P0–P4），通过 SHA256 前缀锁定确保复现。
- **Newcombe (1998) method 10**：配对二值数据的 95% 置信区间估计方法，本文主要统计工具。
- **Suffix Effect**：在文本末尾追加锁定后缀产生的效应变化，本文通过 Study H2 量化。
- **Credit/Pointer-Following**：模型遵循计划指针或信用记录的决策行为。

## 可复现要素
- **数据集**：Budgeted Verification 任务，1,200 episodes（Study B′）、160 个注册 cells（Study I）、80 episodes/condition（Study G′）；论文声明实验数据与 artifacts/FIELD BLOCKS.json 公开。
- **代码/权重**：论文未明确提及代码开源；实验配置与 SHA256 前缀可在附属材料中获取。
- **关键超参**：δ=10（Margin 阈值）、B=4,000（bootstrap 次数）、n=25–40/cell、n=40（Study G/H2）。
- **冻结文本**：五个 Criterion Wording（P0–P4）的字节级锁定文本与 SHA256 前缀已在论文附录 Table 33 中公开。
