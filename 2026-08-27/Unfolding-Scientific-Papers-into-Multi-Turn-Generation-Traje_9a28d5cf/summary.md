---
title: "Unfolding-Scientific-Papers-into-Multi-Turn-Generation-Traje"
source: https://arxiv.org/pdf/2608.25826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:18:26"
field: "合成数据与预训练"
keywords: ["synthetic data", "continued pre-training", "scientific paper", "instruction synthesis", "writing benchmark", "reverse construction"]
innovations: ["将科学论文展开为多轮生成轨迹（写作请求+全局计划+节级deliberation），恢复文档级写作过程用于CPT", "统一逆构造范式同时产出CPT语料、SFT数据集与带rubric/checklist的学术写作基准PAW-Bench"]
benchmarks: ["PAW-Bench", "WritingBench", "HelloBench", "LongBench-Write", "Qasper", "QASA", "LongBench v2", "MMLU-Pro", "GPQA-Diamond", "MATH500", "AIME 2025", "LiveCodeBench v6"]
---

# 论文速读：Unfolding-Scientific-Papers-into-Multi-Turn-Generation-Traje

## 一句话总结
该论文提出了一种将科学论文"展开"为多轮生成轨迹的流水线，通过反向推导还原整篇论文的写作过程（写作请求→全局计划→各节撰写前 deliberation），生成约两倍于原文的持续预训练语料，并同时产出SFT数据集与学术写作基准PAW-Bench；实验表明该方法在提升写作能力的同时不损害推理能力，并改善长文档阅读表现。

## 研究问题与动机
- **现有合成数据方法的局限**：当前主流方法仅在短网络文本（1-2K token）上进行局部思考重建，按固定块机械分割，输出为扁平注释，无法恢复文档级全局规划。
- **科学论文的结构优势**：论文由专家撰写，具有清晰的修辞结构（引言、方法、实验、结论），每节承担明确功能，适合按论文自身结构分解并恢复"文档级写作过程"。
- **写作过程的真实性**：作者确实经历过全局规划与逐节撰写决策，pipeline只需反向重建这一真实发生的过程，而非凭空编造。
- **数据效率与长上下文需求**：科学论文通常高度压缩，大量判断未落纸面；将论文展开为28-29K token的中位数轨迹，可弥补原始30B token短文档对长上下文训练的不足。

## 核心贡献（创新点）
- **可扩展的论文展开流水线**：将1.8M篇arXiv论文展开为多轮生成轨迹，30B token论文文本扩展至57-60B token CPT语料；与已有方法本质区别在于按论文修辞结构分解并恢复文档级规划，而非机械分段。
- **统一的反向构造范式延伸至SFT与评估**：同一"固定真实文本→反向推导生成过程"逻辑可生成200K样本SFT数据集与2,940题PAW-Bench基准；与已有工作区别在于评测任务自带rubric与可机器验证的checklist，且所有任务锚定于未见论文。
- **写作增益在双阶段训练中持续存在**：CPT后接相同SFT配方下，Our-Data模型仍领先无CPT基线1.6-2.4分；与已有方法区别在于证明了写作过程数据在CPT阶段的独立价值，而非仅靠SFT合成。
- **生成器规模影响小但语料复杂度有区分度**：4B/9B/27B生成器输出性能接近（平均差异0.86），最小生成器已足够；与已有预期不同，较强生成器并非必要条件，反而最小生成器的"粗糙性"带来更高训练难度与更好下游收益。

## 方法详解
**统一构造原则**：固定真实文本为最终产物，由LLM反向推导能产生该产物的潜在上下文。

**CPT轨迹结构**（论文公式化定义）：
- 序列：$q, p, c_1, s_1, c_2, s_2, \ldots, c_n, s_n, c_a, a$
  - $q$：重构的写作请求
  - $p$：全文全局计划（含各节outline）
  - $c_i$：第$i$节撰写前的prospective deliberation
  - $s_i$：论文原文各节（verbatim保留）
  - $c_a$：摘要撰写前的retrospective deliberation
  - $a$：论文原文摘要（verbatim保留）

**四阶段生成流程**：
1. **节摘要**：逐节生成覆盖关键主张、方法、结论的摘要，用于后续步骤降低输入长度。
2. **写作请求推导**：基于标题与所有节摘要，以用户口吻推断会要求写出该论文的作者-facing请求。
3. **全局计划与各节deliberation**：以作者身份（研究已完成但尚未撰写）推导章节顺序与各节功能；每节前结合前后节摘要与该节原文，撰写prospective rationale。
4. **摘要deliberation**：基于所有节摘要与真实摘要，推导如何压缩核心贡献与关键结果。

**Prompt约束**：
- 生成器以作者视角推理，讨论修辞目的、科学内容与过渡决策。
- 所有数值、引用、主张必须可从论文上下文恢复。
- 全局计划须覆盖所有保留章节。
- 节deliberation不得直接引用原文，须用自己的话解释。

**SFT数据构造**（Section 3.3）：
- 29种任务类型，分三族：Rewriting（56%）、QA/Extraction（18%）、Reorganization（26%）。
- 逆构造：固定passage为答案事实核心→推导user request（含长度/格式约束）→生成prospective deliberation。
- 34%样本无论文附件，训练纯写作能力；10K样本替换DeepWriting中对应位置形成Our-Mixed SFT。

**PAW-Bench**（Section 3.4）：
- 735篇held-out论文（2026年2-6月），每篇4题共2,940题。
- 双评分：weighted rubric（4-6维，LLM打分）+ checklist（3-6项，58.9%可代码验证）。
- 半数为prompt-only任务，四分之一附excerpt，四分之一附全文。

**生产规模**：
- 生成器：Qwen3.5-4B/9B/27B，temperature=1.0，8,192 token/调用预算。
- 每篇论文$n$节需$2n+3$次模型调用。
- 中位数轨迹长度：4B→29.3K，9B→29.4K，27B→28.3K token。

## 实验与结果
**训练设置**：
- 基座：Qwen2.5-7B，CPT学习率2e-5（cosine decay至10%），1 epoch，64K截断，~4M token/step，weight decay 0.01，gradient clipping 1.0。
- 混合配方：30B轨迹 + 20B FineWeb-Edu = 50B token CPT。
- 对照组：纯FineWeb-Edu CPT（50B）、原始论文纯文本CPT（30B）+ FineWeb-Edu（20B）。

**主要写作结果**（Table 2，GPT-5.5 judged）：
| 模型 | WritingBench Full | PAW-Bench Rubric | HelloBench Avg | LongBench-Write | Overall Avg |
|------|-------------------|------------------|----------------|-----------------|-------------|
| Qwen2.5-7B-Instruct（基线） | 47.3 | 59.09 | 43.30 | 54.10 | 50.95 |
| Plain-Paper CPT + DeepWriting SFT | 49.2 | 57.51 | 42.17 | 59.65 | 52.12 |
| **Our-Data-4B CPT + DeepWriting SFT** | **52.5** | **59.47** | **46.15** | 59.27 | **54.34** |
| Our-Data-9B CPT + DeepWriting SFT | 51.1 | 59.34 | 43.05 | 60.42 | 53.48 |
| Our-Data-27B CPT + DeepWriting SFT | 51.0 | 58.88 | 44.40 | 60.07 | 53.59 |
| Our-Data-27B CPT + **Our-Mixed SFT** | 51.2 | **64.13** | 44.73 | 57.47 | **54.38** |

- **最强结果**：Our-Data-4B CPT + DeepWriting SFT，Overall Avg 54.34，较基线提升+2.44（+4.7%）。
- **最大单项提升**：PAW-Bench rubric从59.09升至59.47（+0.38）至64.13（+8.04）；WritingBench Academic & Engineering提升+2.4至+3.9。
- **双阶段叠加**：Our-Mixed SFT单独使用PAW-Bench从55.42升至63.73（+8.31），再加轨迹CPT进一步升至64.13。

**推理能力保持**（Table 3，OpenThoughts SFT后）：
- All Our-Data CPT模型Avg在45.66-46.82，高于FineWeb-Edu CPT的44.30，接近direct OpenThoughts SFT的46.32。
- **无推理损伤**：CPT写作数据未损害MMLU-Pro/GPQA-D/MATH500/AIME/LCB v6表现。

**论文理解与长上下文**（Table 4，SmolTalk2 SFT后）：
- Qasper：Our-Data-27B达43.01，较基线39.82提升+3.19，Plain-Paper仅35.89（-3.93）。
- LongBench v2 ≤64K：Our-Data-27B达40.22，较基线36.96提升+3.26；Plain-Paper仅38.04（-0.92）。
- 约94%原始论文≤32K token，约1/3轨迹进入32-64K范围，解释了长上下文增益来源。

**训练动力学**（Section 4.5）：
- 4B语料CPT loss最高（1.453 > 原始论文1.422），但下游写作收益最大。
- 27B语料loss最低（1.374）， uniform prose牺牲了开放写作的多样性。
- 验证"harder corpus → better downstream"规律在新轴（生成器规模）上成立。

## 相关工作脉络
- **Ishibashi et al. (2025) / Ruan et al. (2025) / Wang et al. (2025)**：从短网络文本（1-2K token）反向提取latent thoughts作CPT；本文扩展至论文级别，恢复文档级规划与修辞结构。
- **Kim et al. (2026) Megadocs**：等分切分文档插入推理，视为"字符流无结构"处理；本文按论文自身章节结构分解，保留section-level rhetorical role。
- **Li et al. (2024) Self-Instruct / Patel et al. (2026) FineInstructions**：以真实文本为answer推导instruction；本文在同一范式下进一步推导"写作前的prospective deliberation"，而非仅推导prompt。
- **Zeng et al. (2026)**：在代码领域从GitHub仓库逆向工程开发轨迹；本文将同一思路迁移至自然语言学术写作领域。
- **Wu et al. (2025) WritingBench / Que et al. (2024) HelloBench**：现有写作基准多为通用生成任务或每prompt动态生成rubric；本文PAW-Bench每个任务锚定于具体论文并附机器可验证checklist。
- **Liang et al. (2024) / Liang et al. (2025)**：在单轮生成中整合planning；本文在CPT阶段以多轮trajectory形式显式建模写作决策链。

## 局限性与未来方向
- **扩展性未验证**：CPT实验仅覆盖单一基座（Qwen2.5-7B）与单一混合比例（30B轨迹+20B通用），大模型及其他配比未知。
- **评测依赖LLM Judge**：写作指标基于GPT-5.5判定，可能存在模型偏好与人类偏好的gap，需大规模人工评分验证。
- **长上下文/理解评测规模有限**：Qasper、QASA、LongBench v2数据集较小，增益作为corroborating evidence而非main result。
- **轨迹为post-hoc重建**：生成的writing request与deliberation无需匹配作者真实思考过程，存在" plausible但不真实"的风险。
- **未来方向**：方法可推广至其他具有清晰结构的长文档（技术报告、书籍章节、法律文件等）。

## 研究启发与可借鉴点
- **结构感知的文档分解**：按目标文档的自然修辞结构（而非等长分块）进行token-level分解，可使合成推理更符合人类认知流程，值得迁移至技术报告、专利、法律文档等领域。
- **逆构造范式的统一性**：同一"固定真实文本→反向推导上下文"逻辑可并行产出CPT语料、SFT数据、evaluation benchmark，减少数据管线复杂度的设计值得借鉴。
- **弱生成器≠差数据**：4B生成器产出的"粗糙"轨迹因更高training loss带来更好下游收益，提示合成预训练数据选择不应仅看generator scale，可考虑引入多样性/复杂度权衡。
- **Checklist与Rubric双轨评分**：PAW-Bench将58.9%检查项设为code-verifiable deterministic check，降低LLM judge噪声，对构建高可信写作评测有参考价值。
- **CPT与SFT增益可叠加**：本文证明写作轨迹CPT的价值无法被后续SFT完全覆盖，提示团队在训练recipe中应考虑mid-training阶段的专用数据设计而非仅依赖SFT。

## 关键术语表
**CPT（Continued Pre-Training）**：在已有预训练模型基础上继续使用新语料进行预训练，以适配特定领域或能力。
**Deliberation**：指模型在生成正式内容前进行的prospective reasoning过程，涵盖写作意图、章节规划、论证策略等latent thinking。
**PAW-Bench（Paper-Anchored Writing Benchmark）**：论文构建的学术写作评测基准，2,940题均锚定于未见论文，评分含rubric（LLM打分）与checklist（部分机器可验证）。
**Reverse Construction**：固定真实文本为最终输出，由LLM反向推导能产生该输出的请求、计划与思考过程的數據合成方法。
**Multi-Turn Generation Trajectory**：将单篇论文展开为多轮交互序列（请求→计划→各节 deliberation→原文节→摘要 deliberation→原文摘要），用于CPT训练。
**FineWeb-Edu**：Hugging Face出品的教育类网络文本高质量语料库，本文作为通用文本补充以缓解domain shift。
**SFT（Supervised Fine-Tuning）**：在指令-响应对上进行有监督微调，使模型学会遵循用户指令完成任务。
**LLM Judge**：使用大型语言模型作为裁判对生成结果进行自动评分，本文统一使用GPT-5.5（medium reasoning effort）。

## 可复现要素
- **数据集**：arXiv论文源（2006年至2026年1月用于CPT/SFT，2026年2-6月用于PAW-Bench held-out）；CPT语料1.8M论文/约60B token，SFT语料约200K样本，PAW-Bench 2,940题。**论文未明确声明开源状态**。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：CPT学习率2e-5（cosine decay至2e-6），1 epoch，64K截断，~4M token/step，weight decay 0.01，gradient clipping 1.0，seed 42；生成器temperature=1.0，top-p=0.95，top-k=20，presence penalty=1.5，8,192 token/调用预算。
- **基座模型**：Qwen2.5-7B（CPT初始化），Qwen3.5-4B/9B/27B与Qwen3.6-27B（数据生成）。
- **评测judge**：GPT-5.5（medium reasoning effort）统一用于所有写作benchmark评分。
