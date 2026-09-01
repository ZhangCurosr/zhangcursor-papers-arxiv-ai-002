---
title: "Task-Adaptive-Rubrics-for-GUI-Reward-Modeling"
source: https://arxiv.org/pdf/2608.24174v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:10:41"
field: "GUI Agent Reward Modeling"
keywords: ["GUI agents", "reward modeling", "outcome reward", "rubric-based verification", "task-adaptive criteria", "test-time scaling"]
innovations: ["提出粗到细两阶段任务自适应 rubric 框架，弥补静态模板与隐式推理之间的空白", "构建离线固定的 8 类别 GUI 任务 Rubric Bank，实现跨实例可复现的结构化验证边界", "细准则以紧凑软提示注入，最多 2 条并支持弃权机制，避免引入未声明约束"]
benchmarks: ["OGRBench", "MobileWorld", "AndroidWorld"]
---

# 论文速读：Task-Adaptive-Rubrics-for-GUI-Reward-Modeling

## 一句话总结
论文提出 **ADAPTRUBRIC**，一种"粗到细"两阶段评分准则构建框架，通过类别级粗准则检索与实例级细准则生成相结合的方式，为 GUI 结果奖励模型构造任务自适应的显式评判标准；在离线评测和在线强化学习中均显著优于现有奖励验证器（F1 +3.6、任务成功率 +4.23pp）。

## 研究问题与动机
1. **GUI 结果奖励建模的核心瓶颈**：判定轨迹是否成功之前，必须先确立"成功标准"（success criteria），但现有方法对"如何针对每个任务实例构造标准"缺乏明确规范。
2. **静态模板方法的不足**：如 ZeroGUI 等采用固定 rubric 结构，验证稳定但过于泛化，无法适配当前指令的目标对象、数值、操作范围、输出格式等具体约束。
3. **隐式推理方法的不足**：如 OS-Themis 等依赖模型推理过程中隐式生成判断边界，缺乏跨任务类别的明确验证边界，容易遗漏指令细节或过度引入未声明要求。
4. **核心诉求**：奖励验证需要在不同任务类别之间保持可迁移的验证边界，同时在当前指令实例上实现具体约束的自适应提取。

## 核心贡献（创新点）
1. **形式化任务自适应准则构建**：将 GUI 结果奖励建模定义为"在赋予奖励前先由指令推导显式评判准则 R"的问题，区别于仅关注证据收集/分解的既有工作。
2. **粗到细两阶段 rubric 框架（ADAPTRUBRIC）**：类别级粗准则提供结构化验证边界，实例级细准则由指令落地为紧凑的针对性检查项，两者以 $\mathcal{R} = R_c \oplus R_f$ 融合为最终准则。
3. **离线构建的类别级 Rubric Bank**：从开发集轨迹中经 LLM 归纳出 8 类 GUI 任务族的标准 rubric（含验证步骤、常见陷阱、特殊规则、输出格式），确保可复现性。
4. **统一的跨平台评测与在线 RL 迁移验证**：在 OGRBench（5 平台）上达到 86.7% Acc / 86.6 F1（超基线均值 F1 +3.6），并在 MobileWorld 上使用 GRPO 训练 MAI-UI-8B 获得 +4.23pp 任务成功率绝对增益。
5. **奖励引导的测试时扩展（Test-Time Scaling）评估**：在 113 个 AndroidWorld 任务上，ADAPTRUBRIC 在 EarlyStop@7 上提升 +11.88pp、BestOfN@8 上提升 +13.28pp，同时维持较低的误报率（11.17%）。

## 方法详解
**整体流程（Figure 2）**：输入用户指令 $\mathcal{I}$ 和轨迹 $\tau$ → 粗准则检索 → 细准则生成 → 准则融合 → 轨迹上下文选择 → VLM 验证器输出二元奖励 $\hat{r} \in \{0,1\}$。

1. **类别级粗准则检索（Section 4.1）**：
   - Rubric Bank 构建：从 AndroidWorld / MobileWorld 的成功/失败轨迹中，使用 Claude Opus 4.6 归纳重复出现的成功标准和失败模式，分组为 8 类任务族（`info_query`, `create_modify`, `delete_cleanup`, `communication`, `transfer`, `state_navigation`, `composite_workflow`, `general`）。
   - 每个类别条目 $E_c = (m_c, S_c)$，其中 $S_c = \{S_c^{\text{step}}, S_c^{\text{pitfall}}, S_c^{\text{rule}}, S_c^{\text{format}}\}$ 对应验证步骤、常见陷阱、特殊规则和输出格式。
   - 推理时，任务路由器 $c = g_\phi(\mathcal{I})$ 预测类别，通过 top-1 Lookup 检索 $R_c = \text{Render}(S_c)$；未识别则回退至 `general`。

2. **实例级细准则生成（Section 4.2）**：
   - 生成器 $R_f = h_\psi(\mathcal{I}, c, R_c)$，约束 $|R_f| \leq 2$。
   - 三个约束：① **指令接地**——每条细准则必须对应指令中的显式短语/值/约束；② **紧凑性**——不超过两项，避免冗余；③ **弃权机制**——可返回空集 $R_f = \varnothing$（NO_HINT）。
   - 细准则不作为硬性通过/失败项，而是以"软注意力提示"注入最终 prompt。

3. **准则融合与奖励验证（Section 4.3）**：
   - 融合：$\mathcal{R} = \Phi(R_c, R_f) = R_c \oplus R_f$，$R_f$ 以分离的任务特定块追加于 $R_c$ 之后。
   - 轨迹上下文选择：$\tau' = \sigma(\tau, c)$，保留初始/最终状态及高信号帧（文本输入、提交、状态变更操作），上限 $M$ 帧。
   - 最终验证器：$\hat{r} = f_\theta(\mathcal{I}, \tau', \mathcal{R})$，保持验证器架构不变，仅改变输入的准则和上下文。

## 实验与结果
- **离线评测**：OGRBench（1,409 轨迹，5 平台：Ubuntu/Android/Windows/macOS/Web），700 正/709 负，统一 10 截图预算。
  - 最强结果：**Qwen3.6-27B**  backbone 下 **Acc=92.4, F1=88.7**；整体均值 **Acc=86.7, F1=86.6**（超 ZeroGUI+OS-Themis 均值 F1 **+3.6pp**）；主要收益来自 Recall（80.2%→86.5%），Precision 保持 86.8%。
- **在线 RL**：ClawGUI + GRPO，MobileWorld 环境，MAI-UI-8B 策略，2 轮训练。ADAPTRUBRIC 作为奖励器取得 **SR=23.93%**，较无奖励器（19.70%）提升 **+4.23pp**，所有对比奖励器中最佳。
- **测试时扩展**：AndroidWorld 113 任务 × 10 候选轨迹（8×8B + 2×235B 混合池）。ADAPTRUBRIC：ES@7 **+11.88pp**，BoN@8 **+13.28pp**；Acc=88.14, F1=88.41, FPR=11.17%，接近最保守基线 OS-Themis。
- **效率**：相较 OS-Themis 和 ZeroGUI 显著减少 LLM Calls 和 Token 消耗（Table 5），质量-成本 trade-off 最优。
- **消融**：去细准则 −2.0 F1，去粗准则 −2.0 F1，两者均去 −5.9 F1，验证两阶段互补性。

## 相关工作脉络
1. **ZeroGUI**（Yang et al., 2025）：基于终端状态+投票的静态模板验证；本质是通用 rubric 不变，本文通过类别路由+实例细化实现自适应。
2. **OS-Themis**（Li et al., 2026）：多步里程碑分解的隐式验证，验证边界依赖模型推理；本文主张在证据评估**之前**先显式构造任务自适应准则。
3. **DigiRL / DistRL / AndroidGen / WebRL**：分别侧重程序化评估、分布式 RL、Android 特定、Web 特定场景；本文方法为跨平台通用框架，不受平台约束。
4. **CarmO**（Gupta et al., 2025）与 **Auto-Rubric**（Xie et al., 2026）：前者动态生成标准，后者从隐式权重提取显式 rubric；本文的类别粗准则预构建方案更强调可复现性和稳定比较。
5. **静态模板 vs 隐式推理的二分**：本文定位明确填补两者之间的空白——结构化分类边界 + 轻量级实例自适应，避免过度依赖单一流派。

## 局限性与未来方向
1. **覆盖边界**：评估虽涵盖移动端/桌面/Web，但未穷尽所有应用类型、界面设计风格和用户指令表达模式。
2. **粗准则库固定**：离线构建且评估期间保持静态，可能无法捕捉新兴应用领域或特定工作流。
3. **细准则的生成质量依赖指令明确度**：对于指令模糊或缺乏显式约束的任务，细准则可能为 $\varnothing$，此时退化为纯粗准则。
4. **未来方向**：可扩展细准则的动态更新机制、探索更多平台/应用领域的 rubric bank、或将粗准则的 LLM 归纳过程自动化以支持在线增量构建。

## 研究启发与可借鉴点
1. **粗到细的两阶段准则构建范式**：可迁移至其他需要"结构化边界+实例适应性"的评估任务（如代码生成、长程推理、多模态 QA）。
2. **Rubric Bank 的冷启动构建流程**：从成功/失败轨迹对出发，经 LLM 归纳并人工/自动校验 disagreement cases，这一流程可直接复用为其他领域的评估标准构建模板。
3. **软注意力 vs 硬要求的细准则设计**：细准则以提示词形式补充而非替代主 rubric，有效避免"增加未声明约束"的过拟合风险，对 reward modeling 的 prompt 设计有参考意义。
4. **异构候选池（Heterogeneous Pool）用于测试时扩展评估**：混用小/大模型轨迹以扩大"混合任务"比例（从 18.6% 提升至 46.0%），使 reward verifier 判别力更充分暴露，实验设计值得借鉴。
5. **与 RLTrajectory 选择的结合**：ADAPTRUBRIC 已验证对 GRPO 训练的增益，未来可与过程奖励（PRM）结合形成 OR+PR 联合框架。

## 关键术语表
**Outcome Reward Modeling (ORM)**：基于任务最终执行轨迹判定成功/失败的奖励建模方式，区别于过程奖励（PRM）。
**Task-Adaptive Criterion**：根据用户指令和任务类别动态构造的显式评判标准，而非固定模板或隐式推理。
**Coarse Rubric（粗准则）**：来自预构建 Rubric Bank 的类别级别结构化验证指南，提供跨实例可迁移的验证边界。
**Fine Rubric（细准则）**：由当前指令实例生成的少量紧凑检查项，补充粗准则的实例特定约束。
**OGRBench**：OmniGUIRewardBench，跨平台 GUI 结果奖励判别基准，含 1,409 条来自 5 平台的轨迹。
**Test-Time Scaling（测试时扩展）**：在同一任务的多个候选轨迹上，利用 reward verifier 选择最优轨迹以在推理阶段提升成功率。
**GRPO**：Group Relative Policy Optimization，用于在线 RL 训练 GUI agent 的策略优化算法。
**ABSTAIN / NO_HINT**：细准则生成器在无需额外指令特定线索时返回空集的机制，避免引入无关约束。

## 可复现要素
- **数据集**：OGRBench（Li et al., 2026），含 Ubuntu（OSWorld）、Mobile（AndroidWorld）、Windows、macOS（macOSArena）、Web（WebArena-Lite-v2）五平台共 1,409 条轨迹；MobileWorld 用于在线 RL。论文未声明 OGRBench 是否二次开源，但引用的原始 benchmark 均有公开链接。
- **代码/权重**：论文未明确声明开源（截至论文版本）；使用了 Qwen3-VL 系列（开源）和 Gemini 3 Flash（闭源）作为 judge backbone。
- **关键超参**：截图预算=10 帧；Coarse Rubric Bank 8 类别；Fine Rubric 最多 2 项；在线 RL 最大 episode=50 步，epochs=2，rollouts/task=4，batch size=8，学习率=1×10⁻⁵，KL coefficient=0.01，temperature（rollout）=0.7，validation=0.4，max prompt=28,000 tokens。详细见 Appendix C Table 7。
- **Rubric 构建工具**：Claude Opus 4.6（用于离线 bank 构建）；Qwen3.5-122B-A22B（轨迹生成与 routing/generation）。
