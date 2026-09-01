---
title: "VGI-BENCH-Probing-Visual-Intelligence-in-Video-Generation-Mo"
source: https://arxiv.org/pdf/2608.19583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 05:57:30"
---

# 论文速读：VGI-BENCH: Probing Visual Intelligence in Video Generation Models

## 一句话总结
提出了 VGI-BENCH，一个面向视频生成模型视觉推理能力的诊断性评测基准，通过写实风格输入、过程敏感的任务设计与难度可行性校准，系统揭示了当前模型在物理交互、规则遵循与时序状态跟踪方面的局限，并深入剖析了扩散去噪过程中“自我修正”能力缺失的内在机制。

## 研究问题与动机
- **视觉分布失配干扰推理评估**：现有基准多采用线稿或抽象示意图输入以提升可控性，但与视频生成模型的自然图像先验存在显著域偏差，导致评估结果混淆了“视觉风格不匹配”与“真实推理缺陷”。
- **缺乏对视觉滚动推演（Visual Rollout）的强制要求**：多数视觉推理/VQA任务仅需从输入帧直接作答，未要求模型通过显式的帧序列演化过程求解，无法检验模型是否真正具备“想象式推理”能力。
- **任务难度缺乏可行性校准**：部分基准包含超长时序或重度依赖非视觉专业知识（如医学）的任务，失败信号缺乏诊断价值；评测应贴近当前模型的能力边界，保持“有挑战但部分可行”的梯度。

## 核心贡献（创新点）
- **VGI-BENCH 基准构建**：提出包含 27 个任务、810 个实例的两级分类体系（4 大领域 + 7 项技能标签），采用写实输入与过程敏感设计，并通过预生成筛选与人工审核将难度校准至当前 SOTA 模型能力边界；与既往基准相比，从根本上解耦了视觉域失配对推理评估的干扰。
- **双指标过程感知评估框架**：设计全局 Completeness 与局部 Rubric Score 相乘的最终得分，强制模型既要在目标达成上取得实质性进展，又必须在中间轨迹上遵守显式规则，有效拦截“仅靠最终状态侥幸得分”的评估漏洞。
- **多粒度推理行为诊断**：从输出失效模式、输入条件敏感性、合成数据微调迁移边界、以及去噪轨迹动力学四个互补视角进行系统剖析，首次量化揭示当前视频模型在后期去噪步主要“精炼早期假设”而非“纠正推理错误”的内在机制。

## 方法详解
- **两级分类体系**：领域（Domain）互斥划分为 Visual Organization、Physical Manipulation、Structured Puzzles、Spatiotemporal Dynamics；技能标签（Skill Tags）为非互斥的 7 类：Spatial, Temporal, Planning, Attribute Grounding, Physics, Topology, Affordance。
- **任务构造与质量控制**：每个任务含输入图（真实拍摄或 GPT-Image-2 / Nano Banana Pro 生成后经人工 Loop 校验）、结构化 Prompt（明确目标、允许/禁止动作、背景与相机控制）及参考答案；难度分 Easy / Mid / Hard 三档。通过预生成筛选（至少一个 SOTA 模型解出且至少一个模型失败）与人工审核确保可行性。
- **评估指标设计**：
  - **Completeness (Comp.)**：VLM-Judge 基于 2fps 均匀采样帧与任务特定的三级标准（<complete> / <partial> / <failed>，映射为 1 / 0.5 / 0）评估全局进度。
  - **Rubric Score (Rub.)**：针对每项任务的细粒度检查清单，采用自适应粗到细采样（先 4fps 全片扫描，标记可疑区间后以 8fps 重采），配合 10 帧滑动焦点窗口（边帧重叠）进行局部判定；每项违规次数 $x$ 按逆衰减惩罚计分 $s_i = 1/(x+1)$，Rub 为所有 item 分数的均值。
  - **Final Score** = Comp. × Rub.，乘法聚合避免“静态结果好但过程违规”或“过程合规但无进展”的极端情况主导评分。
- **自动评测器**：基于 Gemini-3-Flash 的 VLM-Judge，含专门的粗/精两轮提示词与无图像的最终聚合 Polish 调用，经人工偏好标注验证（AUC 0.803，Pairwise Acc 73.2%）。

## 实验与结果
- **评测范围**：覆盖 Seedance 2.0、Sora2、Veo3.1、Kling3.0、Wan2.7、Gen4.5 等商用模型，以及 MiniMax-H3、HunyuanVideo-1.5、Wan2.2 等开源模型；另适配子集评估 Nano Banana Pro、Qwen-Image-3-Pro 等图像生成模型。
- **核心成绩**：最强模型 **Seedance 2.0** 总分 **51.0%**，商用模型整体显著优于开源；Structured Puzzles 领域最困难（Sdce2.0 仅 44.6%），Physical Manipulation 相对较好（56.0%）。严格成功率（Strict S.R.）极低，人类天花板约 **97.1%**。
- **失败模式**：主要集中在物理坍缩（如物体穿透/悬空违反重力）、规则违反（如汉诺塔同时移动多个盘、绕开路径穿越墙壁）、对象/状态不一致（中途消失、重复生成、状态回退）。
- **输入敏感性**：Oracle Prompting（提供完整逐步解）对闭源模型有小幅提升，但对开源模型（如 HY1.5）几乎无效；线稿输入导致模型排名显著变化，开源模型对视觉风格更敏感。
- **合成数据微调迁移**（VBVR-Wan2.2 vs Wan2.2-I2V）：按与训练分布的结构重叠度分组，Overlap 任务提升最大（如 MAZE +40.5），Non-overlap 提升有限甚至下降；能力提升集中在 Planning / Spatial 等训练分布覆盖的技能，Physics / Topology 等长尾技能改善不明显。
- **去噪轨迹动力学**：解码中间步骤发现，自修正（Wrong → Correct）比例始终低于 1%，且在步数 4 后完全消失；状态变更多为 Wrong → Wrong'（中期高达 24.8%），表明后期去噪主要“锁定并精修早期假设”，而非纠偏。

## 相关工作脉络
- **TiVi-Bench / V-ReasonBench / VBVR-Bench**：侧重从零样本视觉推理评测，但多依赖脚本生成/抽象输入，且缺乏过程敏感性与难度可行性校准；本文在其基础上引入写实输入与过程评估，提供更贴近真实先验的诊断信号。
- **WorldSimBench / PhysGenBench**：聚焦物理世界模拟与运动合理性，偏向底层动力学评估；本文任务更强调显式规则约束与多步状态跟踪的认知层面推理，评估维度从“物理逼真”延伸至“逻辑合规”。
