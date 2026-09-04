---
title: "think-Doesn-t-Stop-Reasoning-Analysis-of-Spurious-CoT-Termin"
source: https://arxiv.org/pdf/2609.03633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:27:54"
field: "大语言模型推理效率与可控性"
keywords: ["CoT early exit", "spurious CoT termination", "attention intervention", "large reasoning models", "chain-of-thought", "EoT regeneration", "inference-time control"]
innovations: ["首次系统定义并量化虚假CoT终止现象，揭示EoT注入控制推理状态的形式-语义鸿沟", "提出EAB注意力偏置方法建立EoT注意力与状态转换成功率的因果关系", "提出一套可复用的状态转换质量评估指标体系（ERR/pre-post分割/Wait分布/boxed模式）"]
benchmarks: ["GSM8K", "MATH-500", "AMC 2023", "AIME 2024", "GPQA-Diamond"]
---

) to force reasoning-to-answering transition, the model often continues generating reasoning-like content in the answering phase, sometimes even regenerating another EoT. This is called "spurious CoT termination."

3. **Key findings**:
- EoT regeneration rate (ERR) varies by model and method
- The pre-regeneration segment shows reasoning-like behavior (Wait tokens, boxed-before-EoT patterns)
- Pre-regen length scales with reasoning tokens saved by early exit
- Hypothesis: insufficient attention to the injected EoT causes this phenomenon
- Solution: Exit-token Attention Biasing (EAB) - adding constant α to pre-softmax attention scores for the EoT key
- EAB reduces ERR and answering-phase length while mostly preserving accuracy

4. **Methods compared**: DEER and DynaSoR as early-exit baselines

5. **Models**: DeepSeek-R1-Distill-Qwen 1.5B/14B, Qwen3-14B, QwQ-32B

6. **Benchmarks**: GSM8K, MATH-500, AMC 2023, AIME 2024, GPQA-Diamond

Now I'll draft the structured Chinese note covering the summary, research problem and motivation, core contributions, detailed methodology, experimental results, related work, limitations, and research insights.
</think>

# 论文速读：</think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination

## 一句话总结
本文研究了CoT早期退出方法中注入结束思考Token（EoT）后，大推理模型仍会在作答阶段继续类推理行为（甚至再次生成EoT）的"虚假CoT终止"现象，并提出Exit-token Attention Biasing（EAB）通过增强对注入EoT的注意力来缓解该问题。

## 研究问题与动机
- **核心问题**：训练无关的CoT早期退出方法通过在推理中途注入`</think>`（EoT）强制触发"推理→作答"状态转换，但注入的EoT并不能始终保证模型真正停止推理。
- **现有方法不足**：早期退出方法（如DEER、DynaSoR）依赖外部匹配显式think-block格式来引导模型行为，但格式层面的Token插入本身并不足以激活模型内部的推理终止机制。
- **规模效应复杂**：不同同规模模型（如R1-Distill-14B vs Qwen3-14B）的EoT再生率差异显著，说明仅靠模型规模不足以解释该现象，训练方式同样重要。
- **效率损失严重**：EoT再生会导致作答阶段长度膨胀数倍甚至十余倍，严重削弱早期退出的效率收益。

## 核心贡献（创新点）
1. **首次系统定义并量化"虚假CoT终止"现象**：发现并形式化EoT注入后模型仍在作答阶段延续类推理行为的现象（以ERR、pre-regen长度、Wait标记、boxed-before-EoT模式为指标）。
2. **提出因果探针机制——EAB**：通过在推理阶段生成时对注入EoT的注意力logit施加常数偏置α，建立注意力强度与终止行为之间的因果关系。
3. **揭示EoT注入控制推理状态的局限性**：证明外部强制插入结构分隔符并不能保证有效的推理→作答状态转换，强调"格式合规≠语义生效"。
4. **跨模型/方法/数据集的广泛验证**：在4个LRM（1.5B–32B）、5个推理基准、2种早期退出方法上系统刻画了现象规律和干预效果。

## 方法详解
- **现象刻画**：
  - **EoT再生率（ERR）**：$ERR = \frac{N_{regen}}{N} \times 100\%$，衡量作答阶段重新生成EoT的样本比例。
  - **作答阶段分割**：将作答阶段在最后一次再生的EoT处切开，定义为pre-regen（注入EoT→再生EoT）和post-regen（再生EoT→序列结束）两段，分析其长度与行为特征。
  - **压缩率**：早期退出节省的推理token比例，ERR随压缩率升高而显著上升（90-100%压缩区间ERR约40%）。

- **Exit-token Attention Biasing（EAB）**：
  - 设exit token位于位置I，对回答阶段所有查询位置$i > I$，对所有层$l$和头$h$，修改预softmax注意力logit：
  - $\tilde{s}_{i,I}^{(l,h)} = s_{i,I}^{(l,h)} + \alpha$，其余key位置保持不变。
  - $\alpha > 0$增强对注入EoT的注意力，$\alpha < 0$则抑制。
  - 仅影响回答阶段生成，不改变推理阶段已生成的token。

- **控制基线方法**：
  - **Double-EoT**：在注入EoT后追加第二个EoT，测试额外EoT key的效果。
  - **Block-EoT**：将回答阶段的</think> logit设为−∞，强制禁止EoT再生。
  - **提示级干预**：Ans-Prefix（在EoT前添加过渡短语）、Post-Box/Post-Ans-Box（在EoT后插入boxed模板）。

## 实验与结果
- **模型**：DeepSeek-R1-Distill-Qwen 1.5B/14B、Qwen3-14B、QwQ-32B。
- **基准**：GSM8K、MATH-500、AMC 2023、AIME 2024、GPQA-Diamond。
- **早期退出方法**：DEER（基于Wait标记处置信度探测）、DynaSoR（固定间隔探测+一致性判定）。

| 模型 | 方法 | MATH | AMC | AIME | GSM8K | GPQA | 平均 |
|------|------|------|-----|------|-------|------|------|
| R1-Distill-14B | DEER | 19.8% | 22.5% | 26.7% | 3.3% | 9.1% | 16.3% |
| R1-Distill-14B | DynaSoR | 35.8% | 40.0% | 40.0% | 8.0% | 18.2% | 28.4% |
| QwQ-32B | DEER | 40.4% | 57.5% | 50.0% | 9.6% | 12.1% | 33.9% |

- **主要结果**：
  - **R1-Distill-14B + DEER + EAB(α=4)**：ERR从16.3%降至3.8%（↓12.5pp），平均作答长度从1,252降至667（近减半），准确率从73.3%升至75.1%。
  - **R1-Distill-1.5B**：EAB同样显著缩短作答长度（DEER从1,059→557，DynaSoR从1,127→386），并提升准确率。
  - **QwQ-32B效果有限**：No-CoT基线ERR超过60%，EAB对其效果较弱，表明QwQ-32B的EoT再生更多源于习得性生成模式而非注意力不足。
  - **Token特异性**：EAB效果仅针对注入的EoT token本身，对SoT或Offset-1/Offset-10位置无效；Offset-1甚至呈现相反趋势。
  - **选择性干预**：EAB主要针对存在虚假终止的样本产生显著效果，对无虚假终止样本几乎无影响。
  - **Double-EoT**效果与正α EAB相近；**Block-EoT**虽消除再生但增加作答长度并降低准确率。

## 相关工作脉络
1. **高效CoT推理（训练无关早期退出）**：DEER（Yang et al., 2026）和DynaSoR（Fu et al., 2026）与本文直接相关，二者均利用EoT强制状态转换，本文揭示了其潜在缺陷。
2. **自反思Token抑制**：Wang et al. (2025a)、Huang et al. (2026a) 研究抑制Wait等自反思标记以减少冗余CoT，本文从EoT注入视角补充了另一维度的理解。
3. **基于注意力的分析/干预**：Zhang et al. (2024)、Nguyen et al. (2026) 等在推理时修改注意力分布以引导模型行为；本文将注意力干预专门用于EoT Token，建立了因果联系。
4. **前置工作对EoT再生的观察**：Zhu et al. (2025)、Zhang et al. (2025d) 也曾注意到LRM可能在thinking阶段被外部终止后恢复推理，但其设定为prompt-level pre-filling跳过整个推理阶段，本文聚焦于动态中途注入场景并提供细粒度行为刻画和因果机制分析。
5. **CoT内部机制研究**：Choi et al. (2025)的Think Clearly（冗余Token剪枝）、Bogdan et al. (2025)的Thought Anchors从不同视角分析推理步骤的重要性，本文从状态转换信号角度切入。

## 局限性与未来方向
- **因果解释为间接证据**：受限于CoT行为的复杂性和训练数据不可见，无法直接验证exit token后的内部状态，结论主要依赖行为观察和注意力干预实验。
- **EAB尚未成为竞争性高效推理方法**：需要解决自适应偏置强度/时机的判定标准，以及兼容高效推理后端（如支持任意注意力偏置的解码引擎）的工程挑战。
- **未充分考虑EoT周围上下文**：当前分析将EoT视为独立转换信号，但boxed-before-EoT模式和Ans-Prefix的有效性暗示周围token和表达推理完成的模式也参与状态转换。
- **QwQ-32B等模型的习得性再生模式**：部分模型的EoT再生源于训练数据中的生成习惯，注意力干预对此类模式效果有限，需探索其他干预手段。

## 研究启发与可借鉴点
1. **注意力干预作为诊断探针的可复用范式**：EAB的方法（对特定key位置施加logit偏置）可迁移至其他需要验证"Token是否真正被模型利用"的场景，如系统Token、分隔符Token、工具调用标记的有效性检验。
2. **虚假状态转换的评估指标体系**：ERR、pre/post-regen长度比、Wait标记分布、boxed-before-EoT密度等指标形成了一套可复用的"状态转换质量"评估框架，适用于任何引入显式阶段分隔符的推理模型研究。
3. **与团队方向的结合机会**：若团队研究推理效率/早期退出/工具调用，可借鉴EAB思路检验自定义分隔符或信号Token的实际效力；也可考虑将EAB思想扩展到非EoT类的状态标记（如工具调用结束符）。
4. **控制基线设计的参考价值**：Double-EoT、Block-EoT、Ans-Prefix等简单控制基线的对照设计，为区分"格式效应"和"语义效应"提供了可借鉴的实验范式。
5. **负向发现的理论价值**：本文揭示了"格式匹配≠功能生效"这一反直觉结论，提醒后续研究在设计基于显式分隔符的推理控制方案时需关注模型内部是否真正利用该信号。

## 关键术语表
- **Spurious CoT Termination（虚假CoT终止）**：指在CoT早期退出中注入EoT后，模型并未真正进入作答阶段，而是在作答阶段继续产生类推理行为（包括自我修正标记和再次生成EoT）的现象。
- **EoT（End-of-Think Token）**：即`</think>`，标记推理阶段结束、作答阶段开始的特殊Token，被DEER/DynaSoR等早期退出方法用于强制状态转换。
- **ERR（EoT Regeneration Rate）**：EoT再生率，即作答阶段重新生成EoT的样本占总样本的比例，是衡量虚假终止严重程度的核心指标。
- **EAB（Exit-token Attention Biasing）**：出口Token注意力偏置，通过在推理阶段对各query位置到注入EoT key的预softmax注意力logit施加常数偏置α来实现的推理时干预方法。
- **Pre-regen / Post-regen**：将作答阶段在最后一次再生的EoT处分割为两段，pre-regen为注入EoT到再生EoT之间的段落，post-regen为再生EoT之后的剩余段落。
- **DEER（Dynamic Early Exit in Reasoning）**：一种基于在Wait标记处探测中间答案置信度的CoT早期退出方法，当置信度超过阈值时注入EoT触发作答。
- **DynaSoR**：另一种CoT早期退出方法，以固定token间隔探测中间答案，当连续三次预测一致时触发退出。
- **boxed-before-EoT模式**：模型在再生EoT前频繁生成`boxed{}`最终答案标记的模式，表明pre-regen段实质上是"伪作答"中的类推理延续。

## 可复现要素
- **数据集**：GSM8K（MIT）、MATH-500（公开于Hugging Face）、AMC 2023（Apache 2.0）、AIME 2024（MAA竞赛题）、GPQA-Diamond（CC BY 4.0）——均为公开数据集。
- **代码**：论文声明代码已开源，链接为 https://github.com/Seunghee-Koh/Spurious-CoT-Termination。
- **模型**：DeepSeek-R1-Distill-Qwen 1.5B/14B（MIT License）、Qwen3-14B（Apache 2.0）、QwQ-32B——均为公开可下载模型。
- **关键超参**：EAB的α值默认设为2（R1-Distill-14B用α=4）；推理阶段预算占总token budget 16,384的60%（R1-Distill/QwQ）或80%（Qwen3）；DEER的置信度阈值λ=0.95；DynaSoR固定探测间隔256 token；最多10次（DEER）/20次（DynaSoR）探测。
- **推理后端**：推理阶段使用vLLM，注意力干预阶段使用HuggingFace Transformers + Flash Attention 2 + xformers。
