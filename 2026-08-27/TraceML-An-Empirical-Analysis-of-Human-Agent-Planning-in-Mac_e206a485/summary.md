---
title: "TraceML-An-Empirical-Analysis-of-Human-Agent-Planning-in-Mac"
source: https://arxiv.org/pdf/2608.26086v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:16:43"
field: "LLM Agent for ML Development"
keywords: ["LLM Agent", "Machine Learning Development", "Human-Agent Comparison", "Process-level Benchmark", "Kaggle", "Trajectory Analysis", "Planning Prompt"]
innovations: ["首个统一人类/agent版本级ML开发轨迹数据集（4,465条人类+207条agent），覆盖134项Kaggle竞赛", "提出跨源版本映射schema与四维transition标注体系（Action/Intent/Magnitude/Score-Effect）并用轻量labeler实现自动化", "基于过程诊断提炼规划harness，证明部分human-agent行为差距可通过instruction弥合"]
benchmarks: ["MLE-bench", "MLAgentBench", "RE-Bench", "HCAST"]
---

# 论文速读：TraceML: An Empirical Analysis of Human-Agent Planning in Mac

## 一句话总结
本文构建了**TraceML**——首个统一版本级的人类/智能体轨迹数据集与标注schema，用于对ML开发过程进行细粒度行为诊断；通过对比Kaggle人类专家与Codex、MLEvolve两个agent scaffold在134项竞赛上的演化路径，揭示了智能体在探索、验证、模型切换与集成等关键行为上与人类的系统性差距，并验证了基于人类最佳实践蒸馏出的规划prompt可在一定程度上弥合部分差距。

## 研究问题与动机
- **现有基准只能记录结果，无法解释差距成因**：Outcome-based benchmarks（如MLE-bench）只关注最终提交分，却忽略了中间开发过程的序列信息，导致"得分低"的原因不可见。
- **人类与智能体的开发过程缺乏可比表示**：人类在Kaggle留下的是数周的notebook保存历史，而agent留下的是命令行工作目录或搜索树日志，两者形态不同，难以并排分析。
- **智能体在竞赛中持续落后于强人类对手，且增加运行时间收益递减**：已有工作（Chan et al., 2024; Wijk et al., 2025）记录了这一现象，但未阐明其根源。
- **过程级数据可以揭示行为差异**：需要一个能在同一schema下容纳人类与agent轨迹的框架，以便逐版本比对编辑行为、意图和分数变化。

## 核心贡献（创新点）
1. **TraceML数据集（4,465条人类轨迹+207条agent轨迹）**：首次覆盖134个Kaggle竞赛的全量版本级轨迹，并在7个竞赛上提供了人类与agent的配对子集，为过程分析提供大规模实证基础。
2. **统一的版本级schema与标注体系**：将Kaggle notebook历史、CLI git commit和树搜索日志映射到同一表示，每个版本携带分数和时间戳，每个transition携带Action/Intent/Magnitude/Score-Effect四维标签。
3. **系统性的"人类vs智能体"过程诊断证据**：发现agent的两个关键差距——Codex陷入提交端微调的单循环（很少pivot），MLEvolve陷入原地突变模型的多pivot循环（pivot无收益），两者都缺乏"回到早期被放弃方向"的能力。
4. **可操作的规划harness与干预实验**：基于上述诊断提炼出约1000 token的规划skill prompt，包含防循环约束、人类优先实践和定期自检机制，在7项竞赛中的5项提升了分数，验证了部分差距可通过指令缩小。

## 方法详解
- **轨迹重建管线**：人类侧从Meta Kaggle数据库提取公开notebook版本，经哈希去重、谱系重建（DAG结构）、过滤（删除截止日期后编辑、浅链路、纯重提交的score-fishing）后规范化为版本序列；agent侧，Codex通过sidecar Git commit追踪，MLEvolve读取其搜索日志并沿root-to-leaf路径线性化为轨迹，统一用MLE-bench的留外验证器重评每个版本分数。
- **版本状态标注**：将每个版本分配至8个粗粒度ML流水线阶段（data/feature/augmentation/model/training/ensemble/validation/inference/infra/housekeeping）之一，每个阶段下有136个细粒度tag。
- **Transition标注四维度**：Action（编辑操作的多标签分类）、Intent（目的推断，6类：optimization/debugging/exploration/housekeeping/ensemble/validation）、Magnitude（编辑规模：micro/minor/major/critical）、Score-Effect（分数改善/持平/退化）。
- **自动化标注pipeline**：手工标注151,088个版本不可行，采用teacher-student方案：gpt-5.4-mini作为teacher在精选trace上生成约束标签，训练两个Qwen3-1.7B学生模型（一个负责版本状态、一个负责transition action）。可靠性验证：coarse stateκ=0.872，coarse action Jaccard=0.875，magnitudeκ=0.921；intent标注在三分类坍缩下保持结论不变。
- **规划harness设计**：约1000 token的prompt，含五部分——（1）时间预算节奏建议（前25%做baseline+K-fold+首轮集成）；（2）反循环约束（禁止单次holdout、禁止连续同类编辑、禁止首版大改）；（3）核心实践要求（K-fold CV、early ensemble、OOF缓存、多模型家族混合）；（4）任务类型条件分支（NLP/CV/tabular/time-series各有差异化建议）；（5）每30–60分钟强制自检7问，运行时周期性注入。

## 实验与结果
- **数据集规模**：134项竞赛共4,465条人类轨迹（149,483个snapshot）、7项竞赛上的207条agent轨迹（1,605个snapshot），其中430条人类轨迹与agent配对于12小时agent预算内。
- **Action Profile分析（图3）**：PCA投影显示Codex接近顶级人类，但MLEvolve与所有人类族群的距离达0.09–0.12 bits（JSD）；细粒度层面两者分别陷入独立于人类的窄带——Codex以5倍于人类的速率做ensemble re-weighting和post-processing微调，MLEvolve专注于原地模型层修改和epoch调整。
- **Pivot率（§4.2）**：人类pivot率为25%，Codex仅9%，MLEvolve高达58%但几乎无净收益（平均每步+0.089 vs −0.008）；控制代码状态后，Codex的out-pivoted比例仍为3:1。
- **Revisit能力（§4.3）**：顶级人类有9.1%的可召回版本回到了更早的工作线（78.5%的召回带来了更好分数），Codex在658个eligible版本中仅1次，MLEvolve在344个版本中0次；Agent能从头等角度恢复分数（Codex恢复89%的挫折），但缺乏"回到被放弃方向"的记忆机制。
- **集成行为质量（§4.4）**：78%的Codex集成编辑只是在re-weight一个从未增长的成员集，而顶级人类将最大比例的精力用于添加新成员；添加/改变成员的人类编辑使下一步改善概率+6.4分，仅re-weight则降低5.8分。
- **Harness实验（§5.2）**：7项竞赛中5项分数提升、2项在噪声范围内、无一退步；Codex重加权行为下降约5倍，早期微编辑从零升至超过顶级人类水平；Ablation证明增益来自prompt内容而非注入频率（Abl-B在hms上降至1.377 vs baseline 1.050）。

## 相关工作脉络
- **MLE-bench**（Chan et al., 2024）：75项竞赛的agent评估基准，仅提供最终分，不记录中间版本或人类轨迹，是TraceML直接对照的对象。
- **MLAgentBench**（Huang et al., 2024）：13项受限时实验工作流基准，同样缺少人类轨迹和过程级对齐。
- **AIRA**（Toledo et al., 2025）：基于MLE-bench的分析框架，补充了搜索算子和验证反馈的观测，但仍以agent-to-agent比较为主，未引入人类过程参照。
- **RE-Bench**（Wijk et al., 2025）：最接近TraceML的工作，在7个定制环境中配对时间预算下的人类和agent尝试，揭示了最终分无法捕捉的scaling模式；但TraceML覆盖134项真实竞赛，保留每个版本的完整代码和task-aligned人类轨迹，粒度更细。
- **HCAST**（Rein et al., 2025）：human-calibrated自主软件工程任务，在189个任务上记录人类轨迹，但聚焦通用编程而非ML工程开发，且不带评分的中间版本序列。
- **Tree of Thoughts / Reflexion**等规划框架：在经典规划域验证了多步推理能力，但未在真实ML开发场景下验证过程行为；TraceML为这类方法提供了第一个"process-level"的实证参照系。

## 局限性与未来方向
- **人类轨迹仅反映公开notebook保存版本**：参与者在本地私下的实验、fork复用和赛后清理版本未被完整捕获，人类侧存在幸存者偏差。
- **人类与agent的设置并非完全受控**：人类跨更长日历时间、使用不同工具和协作方式工作，无法严格匹配12小时agent预算；人类作为"参考分布"而非"控制组"解读。
- **Intent标注是推断而非观测**：基于diff和代码结构推断意图，最依赖intent的结论（如"agent极少diagnose"）在intent三分类坍缩下稳健，但仍需谨慎。
- **轨迹单元非独立**：fork谱系和代码共享导致样本相关性，所有统计区间按竞赛、谱系簇、agent run三层聚类，MLEvolve共享节点还需联合重采样。
- **未来方向**：构建agent自身的历史检索机制（memory）、开发读取当前run状态的控制器而非单纯增强基础模型、扩展至更多agent scaffold和非Kaggle平台、用更大的gold标注集和匹配预算重跑缩小统计噪声。

## 研究启发与可借鉴点
- **"过程级基准"范式可直接迁移**：不仅适用于ML开发，也可推广到软件工程、科学发现等需要长程迭代的工作；对任何"agent vs human"对比场景，提取version-level schema+transition标签是一个可复用的方法论模板。
- **诊断→干预的闭环设计值得借鉴**：先用大数据发现具体行为差距（pivot率、revisit率、ensemble质量），再针对每个差距设计可操作的prompt约束，最后用ablation分离"内容效应"和"调度效应"——这一流程对后续agent系统设计有通用参考价值。
- **Qwen3-1.7B轻量labeler的方案可复用于新领域**：用强teacher蒸馏到小student做大规模自动化标注，配合κ/Jaccard/行为一致性三重验证，在保持可靠性的同时大幅降低标注成本。
- **19个trajectory feature构建的"validation-and-ensembling discipline index"**（Spearman ρ与最终排名关联）可作为一个通用的过程健康度指标，用于快速评估新agent scaffold的行为质量。
- **跨scaffold提取工具包**（已支持Codex CLI、MLEvolve、AIDE、Claude Code、Gemini CLI五个框架）的模式——将"任何agent run在数分钟内转为TraceML轨迹+human cohort对照报告"——为未来评测新模型提供了标准化接口。

## 关键术语表
- **TraceML**：论文提出的版本级轨迹数据集与schema，将人类Kaggle notebook历史和agent运行记录统一为有序版本序列，每个版本带分数、时间戳和多维行为标签。
- **Agent scaffold**：指支撑agent完成ML开发的框架/架构；本文研究两种——单工作目录loop的Codex CLI和进化搜索型的MLEvolve。
- **Pivot**：agent在开发过程中改变"骨干"（backbone）/表征/目标/验证方案的编辑事件，是衡量探索灵活性的关键行为指标。
- **Solution revisit**：agent在偏离早期工作线后又回到该早期状态的编辑行为（通过版本状态签名Jaccard相似度≥0.6且中间曾降至<0.5判定），反映"记忆与回溯"能力。
- **Setback recovery**：agent从低于自身最佳分的状态重新爬升回分数的能力，仅依赖分数信号而非状态签名判定。
- **Out-of-fold (OOF) prediction**：在K-fold交叉验证中，对每个fold用其余fold训练得到的模型做出的预测，缓存后可用于后续stacking/ensembling，是人类专家的核心实践之一。
- **Planning harness**：从人类最佳实践中蒸馏出的规划prompt模块（约1000 token），含反循环约束、人类优先实践和定期自检，作为干预手段注入agent运行。
- **Validation-and-ensembling discipline index**：由fold_averaging、holdout_split、pos_first_ens、mode.ens、oof_prediction、change_weights六个feature构成的综合指标，与最终Kaggle排名呈最强负相关。

## 可复现要素
- **数据集**：TraceML已公开于HuggingFace（https://huggingface.co/datasets/jerryyan/TraceML），包含4,465条人类轨迹的脚本提取版和207条agent轨迹。
- **代码与模型**：提取管线、标注代码、labeler权重（Qwen3-1.7B，继承Apache 2.0）及干预harness均已开源，许可为CC BY 4.0。
- **关键超参**：agent运行预算12小时；Codex使用codex-cli 0.146.0（gpt-5.4-mini后端）；MLEvolve使用Du et al. (2026)的进化搜索配置；规划prompt约1000 token，每30分钟重新注入一次。
- **评估器**：使用MLE-bench的留外验证器对所有agent版本重评分数（非仅最终提交）。
- **未明确提及**：GPU具体型号、训练labeler的epoch/学习率/batch size等细节在论文正文未给出，见附录。
