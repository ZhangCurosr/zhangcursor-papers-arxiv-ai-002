---
title: "WHAT-DO-COMPLIANCE-DETECTORS-READ"
source: https://arxiv.org/pdf/2608.16852v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:58:29"
field: "LLM 合规与安全评估"
keywords: ["compliance monitoring", "activation probing", "guard models", "rule blindness", "representation engineering", "LLM safety"]
innovations: ["提出免训练激活读出 ICS，仅用 10 对样本校准并通过单点积评分", "构造无混淆交叉规则-情境基准，证明仅链式推理能克服规则盲点", "系统审计 7 类 guard/ probe + 13 个基准，揭示规则盲点为领域共性问题"]
benchmarks: ["OmniCompliance", "TRIDENT", "IFEval", "FlexBench", "DynaBench", "SafePyramid", "AIReg-Bench", "CompliBench", "ToxicChat", "OpenAI Moderation", "Aegis", "WildGuard", "PKU-SafeRLHF", "BeaverTails"]
---

# 论文速读：WHAT-DO-COMPLIANCE-DETECTORS-READ

## 一句话总结
本文提出免训练的**内部合规评分（Internal Compliance Score, ICS）**作为激活探针，并系统性审计了现有守门模型与激活探针，发现所有被测试的检测器均存在**规则盲点（rule blindness）**——删除、置换或替换合规规则不会改变检测 verdict，说明探测器读的是通用违规信号而非规则与情境的组合；仅在链式推理下才能克服该缺陷。

## 研究问题与动机
1. **监管合规监控的可用性存疑**：部署的 LLM 需要在数据保护、医疗、金融、平台政策等 20 个监管域中对输出做规则合规检查，但当前检测器（guard model / activation probe）的判定是否真正依赖成文规则而非场景表层特征，尚无实证。
2. **现有方法可能只是在读"违规表象"**：删除、打乱、替换规则后，所有 tested guard 与 activation probe 的检测准确率几乎不变，说明它们捕获的是一个通用的"违规感"，并未真正将规则与情境组合（composition）来做裁决。
3. **缺乏能检验"规则-情境组合"能力的基准**：多数公开合规基准存在**词汇退化（lexical degeneracy）**——标签已编码在文本表层，bag-of-words 即可高准确率预测，导致任何探针结果都不可解释。
4. **需要一种低成本、可复用的审计工具**：guard model Fine-tune 成本高、冻结不可迁移；需要一种一次校准、零额外训练参数、可直接读监控模型激活的探测器来完成大规模审计。

## 核心贡献（创新点）
1. **提出 ICS（Internal Compliance Score）**：基于 10 对标注样本的均值差方向 + 单点积评分的免训练激活读出，与代表工程中的 refusal direction / mass-mean probing 一脉相承，但面向成文规则而非安全标签。与已有工作的本质区别在于：ICS 无需梯度、无需调参、可直接在微调后的模型上重新校准，而其他 guard/probe 是冻结网络。
2. **建立"规则盲点"现象的系统性证据**：通过对 7 种部署 guard、2 种激活探针、1 个 8B zero-shot judge、13 个外部基准的全套反事实干预（删除/置换/替换规则），证明所有高效检测器都无法区分"场景违规"和"给定规则违规"，定位了当前合规监测领域的共同缺陷。
3. **构造无混淆交叉规则-情境基准（crossed-rule benchmark）**：200 个模板 × 4 种 cell，使规则文本与情境文本各自单独无法预测标签，从设计上排除表层捷径，并证明只有 step-by-step reasoning 能突破 chance，而其他 detector 全部失败。
4. **系统审计 13 个公开合规基准的词汇退化性**：发现 7 个公开基准中有 4 个完全退化（TF-IDF floor ≥ 探针得分），仅 2 个可支持对探针的裁决，指出"任何带 narrated verdict 的基准都不可信"这一评估规范。
5. **ICS 的应用价值与边界同时揭示**：ICS 可提升 IFEval 机械验证通过率 +5.2pp、在监管 advisory 响应上 +11.5pp，但同时展示白盒 GCG 攻击可将其降至随机基线，明确了适用场景（非对抗 ranking）。

## 方法详解
1. **ICS 的构建（离线校准）**：
   - 给定 decoder transformer $M$，层 $\ell$ 残差流最后 token 激活 $a_\ell(x) \in \mathbb{R}^d$。
   - 用 $n=10$ 对合规/违规样本 $(x_i^+, x_i^-)$ 拟合**合规方向**：
     $$\hat{d}_\ell = \frac{\bar{a}_\ell^+ - \bar{a}_\ell^-}{\|\bar{a}_\ell^+ - \bar{a}_\ell^-\| + \varepsilon}, \quad \bar{a}_\ell^\pm = \frac{1}{n}\sum_i a_\ell(x_i^\pm)$$
   - 方向归一化后，**锚定层** $\ell^* = \arg\max_\ell \text{AUROC}_{\text{VAL-LAYER}}$；**阈值** $\tau$ 在 VAL-THRESH 上确定（中点阈值近似 LDA）。
   - 总成本：2n 次前向传播校准，**每次评分仅需 1 次前向 + 1 次点积**。

2. **评分与判定**：
   - $\text{ICS}_\ell(x) = \langle a_\ell(x), \hat{d}_\ell \rangle$，更高值表示更合规。
   - 报告余弦版本以剥离 norm 混杂；原始 AUROC 为主指标，FPR@95%TPR 为操作点。
   - 统计规则：paired bootstrap (B=2000) + BH 校正 + Clopper–Pearson 单侧置信界；所有数字对比**预算匹配随机方向 null**（非 0.5 随机基线）。

3. **分布外泛化协议**：
   - **Calibrate-once（LODO）**：从 11 个分布池中提取方向，每次持留出 1 个分布作为测试集，评估跨分布迁移。
   - **Calibrate-on-X / test-on-Y 矩阵**：完整 11×11 迁移矩阵，揭示"同家族内可迁移，跨家族坍缩"。

4. **反事实规则干预**：
   - 删除规则 / 打乱规则顺序 / 替换为其他域规则 → AUROC 不变（Δ ≈ +0.003，0/20 域显著下降）。
   - **正向控制**：仅保留规则无情境 → AUROC 坍缩至 0.56（19/20 域显著下降），证明情境携带标签信号。

5. **交叉规则-情境基准（Crossed Benchmark）**：
   - 8 个监管域 × 25 模板 = 200 模板，每模板提供 2 条规则 + 2 个情境，形成 2×2 正交设计。
   - 标签由数值阈值机械计算，**规则文本单独、情境文本单独、二者拼接**的 TF-IDF 均得 AUROC = 0.500。
   - 所有 fast detector 在 cross set 上均在 chance 附近；仅 chain-of-thought judge 达到 0.849 AUROC。

## 实验与结果
- **主要数据集**：Outcome-ablated OmniCompliance（20 域，移除 OUTCOME 字段后 1B 模型 AUROC 0.952）。
- **检测器对比（Table 3，原位）**：
  - ICS AUROC **0.952**，F1（10-pair recal）0.895；MLP probe（同激活）0.966（略高）。
  - 最佳 guard：LPG-4B AUROC 0.943，Qwen3Guard-Gen 0.908，WildGuard 0.875，Llama Guard 3 0.749。
  - ICS 优势来自**排序能力（AUROC）**而非阈值访问。
- **Calibrate-once 泛化（LODO，Table 2）**：
  - 池化 11 分布训练方向， leave-one-out 均值 AUROC = **0.728**，预算匹配 null = 0.557，清除 null。
  - TF-IDF 池化均值恰好 0.728（持平），但在 outcome-ablated 合规任务上 ICS 超越 TF-IDF +0.26。
  - 跨家族迁移坍缩：Compliance→Safety 0.552，Safety→Compliance 0.476。
- **基准退化审计（Table 16）**：
  - 7 个合规基准中 4 个退化（DynaBench Lex. Floor 0.982、AIReg-Bench 0.957、SafePyramid 0.90+、OmniCompliance 0.896）；仅 FlexBench 和 IFEval 可用。
  - 7 个安全基准中 2 个退化（ToxicChat、PKU-SafeRLHF）。
- **预注册闸门（Pre-registered Gate，Table 9）**：
  - ICS 在 20 域中 11/20 超越最大 trivial floor（Δ > 0，BH 校正 + bootstrap CI 不含 0）；阈值恰为 12/20，差 1 域未达预注册线。
- **候选选择应用（Table 7 / Table 25）**：
  - IFEval 机械验证：ICS-guided best-of-5 较 random **+5.2pp**（CI [+2.4, +7.8]），较 first-pick +6.1pp，捕获 22.6% oracle 空档。
  - 监管 advisory（LLM judge）：**+11.5pp**（CI [+8.8, +14.2]），捕获 43% oracle 空档。
- **对抗鲁棒性（Table 27）**：
  - 白盒 GCG 攻击（12 token，40 步）使 ICS-guided 机械通过率从 0.70 **坍缩至 0.00**（p=1/2000）。
  - 非自适应 fixed suffix 攻击仅降 1.1pp，内容选择在此设定下保留 79% 空档。
- **模型尺度稳健性（Table 31）**：
  - 1B→72B 跨 4 架构族，margin over TF-IDF floor 稳定在 +0.053～+0.068，无显著尺度趋势。
- **LPG 策略守卫复现（§B.5）**：
  - LPG-4B 正确引用管辖条款位置 91–95% 时间，但将违规条款替换为等效宽松条款后仅 **7.1%** 案例翻转 verdict，揭示 citation ≠ adjudication。

## 相关工作脉络
1. **Representation Engineering / Refusal Direction**（Zou et al., 2023；Arditi et al., 2024）：单方向拒绝表征的开山工作，ICS 沿袭其理念但面向成文规则而非安全标签。
2. **SIREN / GLiGuard**（Jiao et al., 2026；另）：基于内部表征的安全探针，本文与其同属 activation probe 家族但任务域为合规而非安全。
3. **Latent Policy Guard (LPG)**（Li et al., 2026）：唯一输入规则的 guard model，拥有 custom-policy channel；本文证明其仍受 rule blindness 困扰。
4. **Rachmil et al. (Rachmil whitening, 2025)**：最接近的并行工作——基于 activation-space whitening 的免训练合规检测；本文与之差异在于估计器（均值差 vs. 白化 OOD）与读出位置（last-token vs. full-context），且本文的 rule-blindness 结论同样适用于其 detector class。
5. **Gavel / Legilimens**（Rozenfeld et al., 2026；Wu et al., 2024）：另一类基于规则的 guard；本文指出跨 rule-family 盲点是系统级问题而非单一架构缺陷。
6. **McCoy et al. (Right for the wrong reasons, 2019) / shortcut learning 文献**：本文的基准退化诊断（lexical degeneracy）与该传统一脉相承，将"捷径学习"问题推广至规则条件化合规场景。

## 局限性与未来方向
1. **需要第一方激活访问**：ICS 依赖获取监控模型的中间层激活，适用于自监控场景，不适用于第三方审计。
2. **跨家族泛化差**：合规方向无法迁移到安全数据（0.476），反之亦然，必须按目标家族重新校准。
3. **仅捕获宽泛合规风险方向，非规则特异性裁决**：反事实干预证明 ICS 读的是违规 register 而非规则-情境组合。
4. **序列化敏感**：当文本以规则结尾而非情境时 AUROC 从 0.952 降至 0.668，要求 scored text 必须以 case 结尾。
5. **对抗脆弱性**：白盒 GCG 攻击可完全破坏 ICS-guided selection，仅适用于非对抗 ranking 场景。
6. **阈值不可迁移**：同一方向在不同域的 FPR@95%TPR 从 27.7% 激增至 p95 的 77–94%，阈值必须逐域重校准。
7. **未来方向**：① 探索能让 detector 真正组合规则-情境的架构（如显式推理模块）；② 将 crossed-rule benchmark 推广为合规 detector 的标准测试套件；③ 研究对抗鲁棒的 activation readout。

## 研究启发与可借鉴点
1. **交叉规则-情境设计范式**：用 2×2 正交设计构造"无捷径"基准——每个规则文本和情境文本各出现两次且与两种标签配对，确保任何一方单独信息量为 0。该设计可迁移至任何"条件化决策"任务（如政策遵循、法律推理）的评估。
2. **基准退化审计框架**：先评估 lexical floor（policy-blind bag-of-words），再判断基准是否可裁决探针。建议在后续研究中将"floor > probe"作为基准弃用的硬标准，避免虚高结果。
3. **预算匹配随机方向 null**：层选择本身会引入 selection inflation（random direction AUROC 可达 0.71），必须用同 pipeline 的随机方向作为 null 而非 chance 0.5。这对所有 activation probe 工作都是必要基线。
4. **ICS 的"零参数 + 可重校准"思路**：在微调后的模型上仅用 10 对样本重新校准方向，AUROC 波动 < 0.007。这一范式可用于低资源域适配场景，替代昂贵的 guard model 重新训练。
5. **Citation ≠ Adjudication 的分离检验**：LPG 复现显示模型可以准确引用规则位置（91–95%）却不被规则内容影响 verdict（仅 7.1% 翻转）。建议在 future work 中将"引用准确率"与"verdict 翻转率"分开报告，避免被 citation 指标掩盖规则盲点。

## 关键术语表
**Rule Blindness（规则盲点）**：检测器无视给定规则的内容，仅凭场景表层特征做出 verdict，删除/替换规则不改变检测结果。
**Internal Compliance Score (ICS)**：基于 10 对标注样本的均值差方向 + 单点积的免训练激活读出，用于直接评分监控模型的合规风险。
**Crossed Rule-Scenario Benchmark（交叉规则-情境基准）**：2×2 正交设计基准，每条规则与每个情境各配对一次违规一次合规，确保规则/情境单独无法预测标签。
**Lexical Degeneracy（词汇退化）**：基准标签可从文本表层词汇直接推导（如 narrated verdict），导致 bag-of-words 即可高准确率预测，使探针结果不可解释。
**Budget-Matched Selection Null（预算匹配选择 null）**：将随机方向经相同层搜索流程后得到的 AUROC 分布作为 null，用于校正层选择引入的 inflated 指标。
**Calibrate-once（一次性校准）**：从一个分布池训练方向，测试时持留整个分布，模拟部署场景下 guard 遇到未训练分布的情况。
**Activation Probe（激活探针）**：读取模型中间层激活以推断某属性（如安全/合规）的线性或非线性感知器。
**Guard Model（守门模型）**：专门训练用于检测有害/违规输出的分类模型，常作为 LLM 部署的安全层。

## 可复现要素
- **数据集**：OmniCompliance（20 域，已摘除 Outcome 字段）；TRIDENT（finance/law/med）；ToxicChat、OpenAI Mod、Aegis 1.0/2.0、WildGuard、PKU-SafeRLHF、BeaverTails；DynaBench、SafePyramid、AIReg-Bench、CompliBench、FlexBench、IFEval。**论文未明确说明 OmniCompliance 是否完全公开**，但称已发布 counterfactual protocol 与 evaluation code。
- **代码/权重**：论文声明 "Total compute is released with the evaluation protocol"、"Provenance-stamped JSONs are released with the code"，evaluation protocol 及交叉基准模板已开源。
- **关键超参**：校准对数 n=10 pairs/domain（总计 ~100–160 pooled）；数据划分 40/20/20/20（TRAIN / VAL-LAYER / VAL-THRESH / TEST）；seed=42；bfloat16 提取（72B 臂 int8）；512-token 截断；anchor layer 通过 VAL-LAYER 上 AUROC 选取。
- **模型清单**：Llama-3.2-1B/3B-Instruct、Llama-3.1-8B-Instruct、Gemma-2-9B-it、Mistral-7B-Instruct-v0.3、Qwen2.5-7B/ Qwen3-8B/ Qwen3.5-4B/9B/27B、Llama-3.3-70B、Qwen2.5-72B（int8）。
- **评估工具**：GPU 单卡完成全部检测 arm；GCG 攻击为唯一多步优化组件。
