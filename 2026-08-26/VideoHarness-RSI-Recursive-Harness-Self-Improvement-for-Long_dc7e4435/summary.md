---
title: "VideoHarness-RSI-Recursive-Harness-Self-Improvement-for-Long"
source: https://arxiv.org/pdf/2608.24302v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:17:08"
field: "长视频理解"
keywords: ["long-video understanding", "context construction", "frozen VLM", "recursive self-improvement", "automated harness search", "Write-Read-Pack"]
innovations: ["将可执行上下文构造程序作为外层搜索目标，冻结 VLM 实现递归自改进", "提供 propose-execute-evaluate-retains 基线及完整审计档案", "提出 Write-Read-Pack 分解框架分析上下文构造机制"]
benchmarks: ["LVBench", "Video-MME", "MLVU"]
---

# 论文速读：VideoHarness-RSI-Recursive-Harness-Self-Improvement-for-Long

## 一句话总结
论文提出了 VIDEOHARNESS-RSI，通过在冻结的视觉-语言模型（VLM）周围递归搜索可执行的上下文构造程序（harness），以可控方式研究"仅改进上下文构建本身"能带来多少收益。从均匀采样和较强 hand-crafted 基线出发，搜索均能发现可复用的上下文组织策略，并在 LVBench、Video-MME 和 MLVU 上取得提升。

## 研究问题与动机
- **长视频理解的核心瓶颈是上下文构建**：相关证据稀疏、时距远且被大量无关帧包围，直接全部输入不现实，性能不仅取决于 VLM 能力，也取决于系统暴露给模型的上下文。
- **现有方法难以隔离贡献**：当前系统通常同时改变证据表示、检索机制、工具链、推理工作流甚至底层模型，无法回答"仅改进可执行上下文构造程序"能带来多大提升。
- **自改进的可执行 harness 是否可行**：在冻结 VLM 和固定接口的前提下，能否通过外层提议者基于 prior programs、评估结果和执行轨迹生成候选并递归提升？
- **如何建立可控基线以研究搜索机制**：需要一个可复现的设置，分离模型能力、上下文构建能力和搜索能力，便于分析策略发现、转移性和失败模式。

## 核心贡献（创新点）
1. **将视频 harness RSI 形式化为可控自动设计场景**：可执行上下文构造被优化，而 VLM 及其回答接口固定不变；与已有工作的本质区别在于把上下文构建作为独立的搜索目标，而非模型训练或推理流程的一部分。
2. **提供 propose–execute–evaluate–retain 基线及完整审计档案**：包含候选代码、谱系、执行轨迹、逐题输出和评分；与已有工作不同，本文强调可复现性与透明度，而非仅提供一个黑盒系统。
3. **系统研究递归选择的上下文构造器的有效性、内部机制、直接复用和搜索–评估失败模式**：发现搜索早期集中提升、随后饱和，并揭示了"容量≠选择质量"的规律；与已有工作不同，本文关注搜索本身的边界与可靠性。
4. **建立 Write–Read–Pack 分析分解框架**：将 harness 分解为构建地址表示、检索问题条件证据、排序打包三个阶段，为结构化对比不同方法提供了统一视角。

## 方法详解
- **上下文构造的可执行程序定义**：harness $H$ 将视频 $V$ 和问题 $q$ 映射到有界多模态上下文 $C_H = H(V, q; K)$，冻结 VLM $M$ 在此基础上回答 $\hat{y} = M(C_H, q)$。优化目标是开发集上的准确率，$M$ 的参数和解码配置完全固定。
- **Write–Read–Pack 分解**：$H(V, q; K) = \text{Pack}_H(\text{Read}_H(\text{Write}_H(V), q), K)$。WRITE 构建可寻址表示（如视频流、标题列表或嵌入索引），READ 检索问题条件证据，PACK 将有限上下文排序格式化后送入 VLM。这是对单个构造器的分析分解，不代表三个独立优化目标。
- **递归搜索更新规则**：外层提议者基于当前前沿 $F_t$ 和历史档案 $\mathcal{A}_t$ 生成候选 $\{H_{t,j}\}$，经烟测和端到端评估后，仅当某候选的开发准确率严格高于当前前沿时才更新 $F_{t+1} = H_{t,j^*}$，否则保留原前沿。留出集不参与任何提议或选择。
- **成本与帕累托报告**：报告每个候选的平均输入代价 $c(H) = T_{\text{visual}}(H) + T_{\text{text}}(H)$（每问 token 数），但不用于搜索决策，仅作为辅助分析维度。
- **搜索合约约束**：可变对象限于可执行上下文构造代码，内层 VLM、回答接口、任务指标和评估器可用数据在搜索协议内保持固定。

## 实验与结果
- **数据集与拆分**：LVBench 本地可用 83 视频、1,232 问答对；Random(42).shuffle 后取前 350 题为开发集，剩余 882 题为留出验证集。跨基准直接使用 Video-MME（2,700 QA）和 MLVU（2,174 QA）。
- **冻结模型与协议**：主实验冻结 Qwen3-VL-8B-Instruct，温度=0、思考关闭；最终视觉上下文上限 $K=40$；2-fps 视频流；语义图像检索使用 CLIP ViT-B/32。
- **对照基线**：UNIFORM-40、CLIP-KNN、CAPTION-KNN、AKS、WORLDMM-STYLE/MAXBUDGET、HOMER-STYLE、VIDEOSEAL-MATCHED，以及 GEMINI-2.5-FLASH 和 GPT-5 在均匀采样下的参考。
- **最强结果**：从 AKS 出发的 TIMESTAMPED-AKS 在留出集上达到 **50.3%**（开发集 52.0%），优于 UNIFORM-40 的 36.3%、CLIP-KNN 的 41.7%、AKS 的 46.5%，并显著超过大部分 literature-inspired 对照。
- **从均匀采样出发的结果**：EMBEDNAVIGATE-HYBRID 在留出集上达到 45.4%，输入代价较高（16,898 tokens/问）。
- **跨基准直接复用**：不加搜索或适配，Hybrid 在 Video-MME 上从 59.9% 提升至 61.5%，在 MLVU 上从 63.2% 提升至 65.9%。
- **搜索轨迹**：两代接受更新（CAPTIONNAVIGATE → EMBEDNAVIGATE-HYBRID），后续提案饱和；高容量实验暴露开发–测试 gap：更大可寻址上下文不一定带来更好的留出选择。
- **机制诊断**：Hybrid 通过扩展证据可见性（172 vs 143）和移除 navigator 的格式失败带来提升；导航对 temporal/reasoning 类问题帮助大，视觉检索对 entity/key-information 类更有帮助。

## 相关工作脉络
- **长视频上下文构建（Shen et al., 2024; Wang et al., 2025; Fu et al., 2024; Zhou et al., 2025）**：设计或学习特定压缩/检索机制；本文定位不同——不提出新机制，而是让可执行构造程序本身成为外层搜索对象。
- **Agentic 视频理解（VideoAgent, Deep Video Discovery, WorldMM, Homer, VideoSEAL）**：使用 agent 主动获取证据或维护多模态记忆；本文窄化焦点，仅隔离最终上下文的可执行构造程序，保持回答模型固定。
- **自动化 agent/harness 优化（ADAS, AFlow, Meta-Harness）**：从执行反馈优化 agent 设计或模型 harness；本文不提出通用新搜索算法，而是提供长视频领域的受控实例化。
- **程序合成与 LLM 引导搜索（Gulwani et al., 2017; Koza, 1992; Ellis et al., 2021; Romera-Paredes et al., 2024）**：经典程序合成背景；本文将其应用于特定冻结 VLM 场景下的上下文构造。
- **MetaVideoAgent（Cui et al., 2026）**：研究视频 agent pipeline 的诊断驱动演化；本文与之不同，聚焦可执行 context constructor 而非多模块 pipeline 的演化。

## 局限性与未来方向
- **实验设置单一**：仅使用一个冻结 VLM、一个数据 seed、有限搜索轨迹，未建立跨模型行为或搜索方差结论。
- **LVBench 局限性**：使用本地可用子集（非完整源），且为题目级拆分而非视频级拆分。
- **提议者潜在污染**：专有 LLM 提议者可能包含基准级先验知识。
- **单视频上下文构建**：当前程序为 per-video 上下文，未探索跨问题的可变记忆。
- **高容量下的开发–测试 gap**：更大上下文暴露选择问题，提示单纯增加容量不可靠。
- **未来方向**：跨模型泛化搜索、跨问题动态记忆、更鲁棒的成本评估（含离线索引与 API 调用）、更大规模的搜索轨迹统计。

## 研究启发与可借鉴点
1. **可控消融设计**：冻结下游模型、固定接口、仅搜索可执行构造程序，是隔离"上下文构建能力"与"模型能力"的优雅范式，可直接迁移到其他模态或任务。
2. **审计档案的重要性**：保留候选代码、谱系、执行轨迹和逐题输出的完整归档，为可复现性和失败模式分析提供了基础设施，值得团队借鉴。
3. **开发集与留出集的严格分离**：搜索前沿更新仅基于开发集点估计，留出集完全不可见，避免了隐式过拟合；这种分离策略可推广到任何自动化程序搜索场景。
4. **Write–Read–Pack 分解作为分析工具**：将复杂 harness 统一分解为三个分析阶段，便于对比不同方法在哪个环节产生改进，可直接用于团队内部的方法分类与诊断。
5. **帕累托成本报告而非搜索目标**：输入代价作为报告维度而非优化目标，保持了搜索纯度；这一做法适用于任何存在多目标权衡的自动化设计场景。

## 关键术语表
- **Harness（上下文构造程序）**：将视频和问题映射到有界多模态上下文的可执行代码，决定冻结 VLM 能看到什么。
- **Recursive Self-Improvement（RSI）**：在此指外层提议者通过 propose–execute–evaluate–retain 循环迭代改进 harness 代码，而非改进 VLM 自身参数。
- **Write–Read–Pack 分解**：分析框架，WRITE 构建地址表示，READ 检索问题条件证据，PACK 排序打包后送入 VLM。
- **Development/Held-out 分离**：开发集仅用于搜索与前沿更新，留出集完全不可见，用于最终评估与转移测试。
- **Pareto 成本报告**：以平均视觉+文本 token 数为输入代价维度绘制帕累托前沿，但不影响搜索决策。
- **Proposer（提议者）**：使用 Claude Opus 4.6 + Claude Code 的外层 LLM，负责生成候选 harness 突变。
- **Smoke test（烟测）**：候选程序在正式评估前的快速功能性检查。
- **Search–evaluation gap（搜索–评估 gap）**：开发集上表现好的候选在高容量实验中可能在测试集上不如基线，提示过拟合风险。

## 可复现要素
- **数据集**：LVBench（本地 83 视频、1,232 QA）、Video-MME、MLVU；论文公开了 seed-42 索引及 83 可访问/20 不可访问视频 ID。
- **代码/权重**：论文未明确声明开源仓库，但提供了 reproduction package 包含固定 seed 索引、config 文件、per-question 预测与审计档案。
- **关键超参**：冻结 Qwen3-VL-8B-Instruct；temperature=0、thinking disabled；K=40 视觉观察上限；2-fps 视频流；CLIP ViT-B/32 作为语义评分器；每代提议 3 个候选、运行 5 代；严格点估计前沿更新。
- **提议者成本**：17,514 输入 tokens + 114,738 输出 tokens，约 $7.79 API 费用，35.8 分钟总时长。
