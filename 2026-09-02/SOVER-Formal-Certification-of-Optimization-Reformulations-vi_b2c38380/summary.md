---
title: "SOVER-Formal-Certification-of-Optimization-Reformulations-vi"
source: https://arxiv.org/pdf/2609.00728v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:16:32"
---

# 论文速读：SOVER-Formal-Certification-of-Optimization-Reformulations-vi

## 一句话总结
提出 SOVER，一种将 LLM 语义映射与 SMT 定理证明解耦的优化重构等价性验证框架；通过符号化检查可行域交叉一致性与目标全局保序性，在混合整数线性与连续非线性问题上实现可形式化证明的高精度等价性认证。

## 研究问题与动机
- LLM 辅助建模生成的公式常含变量歧义、系数错误或结构缺陷，仅靠求解器输出最优值对比易受非活跃约束、数值截断与目标缩放干扰，产生假阳性/假阴性。
- 异构表述（如 LP 经非线性变量代换退化为 NLP）与符号参数场景下，传统字符串匹配或图同构方法无法识别语义等价。
- 现有 EquivaMap 等方法仅做单次映射+轻量可行性检验，缺乏对目标弱序保持性的严格逻辑证伪，且在非线性连续域缺乏形式化保证。
- 缺少面向连续非线性重构的公开评测基准，难以量化模型对微小语义偏移（方向/系数/范围失配）的鉴别力。

## 核心贡献（创新点）
- **LLM+SMT 分层验证架构**：LLM 负责提取变量/参数对应关系并经由纠错提示精炼，Z3/dReal 负责严格逻辑证伪，实现语义猜测与形式化认证的职责分离。
- **超越最优值匹配的等价定义**：以域交叉可行性与全局目标保序性为判据，形式化证明两公式在映射下具有相同 argmin 解集，可容忍目标正比例缩放与单调变换。
- **非线性连续问题的 $\epsilon$-argmin 容差认证**：将 dReal 的 $\delta$-完备性引入优化验证，通过 margin-separated 查询与 $\epsilon$-最优邻域包含性证明（Proposition 2）处理超越函数与非凸边界。
- **NLEQUIV-150 开源基准**：发布 150 对应用驱动的非线性重构对（100 等价+50 硬负样本），覆盖 8 类 mismatch 机制，填补该方向评测空白。
- **求解器高效性设计**：仅需求解可信源问题，验证通过 SMT 查询完成；Z3 阶段平均仅 0.03 s，验证成本极低且不依赖独立求解对比。

## 方法详解
- **映射生成与纠错**：基于 EquivaMap 策略提取决策变量映射，叠加一次 targeted refinement prompt 强制修正多对一冲突；参数映射默认 1:1 且常数因子固定为 1.0，避免缩放漂移。
- **符号化预处理**：解析 `gurobipy` 字符串剥离 API 语法，提取约束谓词 $C(\mathbf{x})$ 与目标 $O(\mathbf{x})$；索引变量按固定代理维度展开，辅助/slack 变量经存在量词投影消去。
- **两阶段 SMT 验证**：
  1. 构造 $\neg(C_A(\mathbf{x}) \leftrightarrow \tilde{C}_B(\mathbf{x}))$ 查询，SAT 返回 `Feasibility Mismatch`，UNSAT 则通过。
  2. 引入源变量阴影副本 $\mathbf{y}$，构造 $\neg[(O_A(\mathbf{x}) \le O_A(\mathbf{y})) \leftrightarrow (\tilde{O}_B(\mathbf{x}) \le \tilde{O}_B(\mathbf{y}))]$ 全局序反例查询；UNSAT 结合可行性结果即证 $\arg\min$ 等价（Proposition 1）。
- **非线性扩展**：使用 dReal 配置 $\delta=10^{-5}$、分离边距 $10^{-3}$，仅将 UNSAT 视为有效证书，$\delta$-SAT 作为候选反例；验证精确最优解映射后落入对方 $\epsilon$-最优邻域（$\epsilon=10^{-3}$）（Proposition 2）。
- **工程流水线**：四阶段（解析去噪→符号声明与代换→可行性查询→序保持查询），Z3 实例硬超时 10,000 ms，输出结构化 CSV 记录状态、公式、反例与诊断元数据。

## 实验与结果
- **数据集**：EquivaFormulation（2178 对 MILP 变体）、NLEQUIV-150（100 等价+50 硬负非线性对）、FormulationBench 子集（116 对跨 20 基础问题）。
- **基线**：EquivaMap、Naive LLM Prompting、Gemini-CoT、WLT、Canonical Accuracy。
- **主要结果**：
  - EquivaFormulation：SOVER 2173/2178（99.77%），目标缩放（\_i）242/243（Canon. 0/243），线性代换（\_h）与目标→约束替换（\_f）均达 243/243，非活跃约束移除（\_l）241/243。
  - NLEQUIV-150：SOVER 149/150（99.33%），唯一失败为 LLM 映射不完整；50 个硬负样本全部正确拒绝。
  - FormulationBench：111/116（95.69%）。
  - EquivaMap 复现得分 1935/2178（88.84%），误差集中于需精确映射的场景。
- **性能分析**：SOVER 端到端平均 7.07 s/对（EquivaMap 2.60 s），但 Z3 验证阶段仅占 ~0.4%（0.03 s），主要开销为 LLM 映射；验证阶段无需独立求解目标问题，鲁棒性与低成本兼得。

## 相关工作脉络
- **EquivaMap (Zhai et al., 2025)**：本文直接延伸对象；EquivaMap 仅做单次映射+轻量检验，SOVER 引入二次纠错与严格可行性/序保持双重证明，并拓展至非线性 $\epsilon$-argmin 域。
- **WLT / Canonical Matching**：结构同构与字符级匹配对目标缩放、变量重命名、约束松弛等语义保真变换完全失效，本文证明符号逻辑验证的必要性与泛化优势。
- **OptiMUS / FormulationBench 生态**：聚焦 LLM 生成 MILP
