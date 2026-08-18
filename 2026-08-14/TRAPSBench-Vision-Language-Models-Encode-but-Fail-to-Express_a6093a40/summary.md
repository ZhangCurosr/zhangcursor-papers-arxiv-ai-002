---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:18:34"
field: "多模态大模型可靠性与可解释性"
keywords: ["视觉语言模型", "认识论不确定性", "选择性拒答", "激活引导", "物理推理基准", "模型校准", "表征-输出鸿沟"]
innovations: ["提出TRAPSBench程序化生成基准与PECS联合评估指标", "通过探针+引导+激活 steering 三合一验证VLM内部编码但输出抑制认识论不确定性", "揭示文本vs视觉不可答性检测的4倍不对称性及CoT推理的双刃剑效应"]
benchmarks: ["TRAPSBench"]
---

# 论文速读：TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express

## 一句话总结
本文提出 TRAPSBench，一个基于 MuJoCo 程序化生成的物理不确定性视频基准，揭示 VLM 内部能编码认识论不确定性但输出端无法表达的瓶颈；并提出 PECS 指标，要求模型既能在可答时正确回答，又能在不可答时选择性拒答。

## 研究问题与动机
- 核心问题：VLM 在视觉证据被遮挡或混沌时，是否具备"选择性拒答"的能力，以及这种能力是否依赖于正确的输入感知还是输出表达
- 现有物理推理基准（CLEVRER、IntPhys、PhysBench 等）不评估在证据不足时的选择性拒答能力
- 现有拒答评测指标存在盲区：拒答召回率奖励无差别拒答，准确率忽略认识论维度，简单乘积不惩罚假拒答
- 需要一种能同时要求正确回答与选择性拒答的联合度量标准

## 核心贡献（创新点）
1. **TRAPSBench 与 PECS 指标**：提出包含 1,404 对匹配物理场景的程序化生成视频基准，以及要求同时做到正确回答和选择性拒答的联合评估指标 PECS，直接惩罚"总是拒答"和"从不拒答"策略
2. **表征-输出鸿沟的发现与验证**：通过线性探针（AUROC 最高 0.91）、引导式提示（拒答召回提升 1.9×）、单层激活引导三种方式证明 VLM 内部编码了认识论不确定性信号但输出端抑制它，且该发现跨 Qwen、Gemma、LLaVA 三个架构复现
3. **拒答不对称性发现**：模型检测文本不可答性的能力比视觉信息缺失强约 4 倍（chaotic 分裂上最高达 197×），且链式思考推理可能损害校准而非改善
4. **因果机制分析**：发现 occlusion-family 方向编码了跨域通用的"证据缺失"信号，而 chaotic 方向几乎是域特定的且正交，揭示了认识论透明性对可迁移性的影响

## 方法详解
- **最小视频对范式**：使用 MuJoCo 物理引擎生成成对的 control（确定性可预测）和 void（单一修改使结果不可计算）视频，比较模型在配对视频上的表现以隔离认识论不确定性识别能力
- **三分类不确定性taxonomy**：
  - Occlusion（N=202）：不透明遮挡物阻挡关键视觉数据
  - Chaotic Sensitivity（N=500）：对初始条件极度敏感的确定系统，截断视频前结果未解析
  - Ill-Posed Questions（N=702）：使用与 control 相同视频但提出文本层面不可答的问题
- **PECS 指标公式**：$$\mathrm{PECS} = \mathrm{Acc} \times \max(0, \mathrm{AbsRec} - \mathrm{FalseAbs})$$，其中 AbsRec - FalseAbs 为 Youden's J 统计量，要求模型同时具有高准确率和选择性拒答能力
- **线性探针**：在冻结的 Qwen3-VL-8B 所有 37 层提取隐藏状态，训练 l2-正则化 LR 探针预测 void vs control，报告最佳层 AUROC
- **激活引导**：计算 void 方向 $\mathbf{v}_\ell = (\bar{\mathbf{h}}_\ell^{\mathrm{void}} - \bar{\mathbf{h}}_\ell^{\mathrm{control}}) / \|\bar{\mathbf{h}}_\ell^{\mathrm{void}} - \bar{\mathbf{h}}_\ell^{\mathrm{control}}\|$，在自回归每一步修改隐藏状态 $\mathbf{h}_\ell^{(t,i)} \gets \mathbf{h}_\ell^{(t,i)} + \alpha_{\mathrm{eff}} \cdot \mathbf{v}_\ell$，验证因果关系

## 实验与结果
- **数据集**：TRAPSBench 共 1,404 对匹配 video pair（202 occlusion + 500 chaotic + 702 ill-posed）
- **评测模型**：16 个 VLM，涵盖 Gemini（6个）、Qwen3-VL（3个）、GPT-5（5个）、Gemma 4 E4B、LLaVA-Video-7B
- **最强结果**：标准提示下 Gemini 2.5 Flash 以 PECS 0.292 领先；引导提示下 Gemini 3.1 Pro R-Low 达到 PECS 0.568
- **关键数字**：
  - 线性探针跨数据集 AUROC 最高 0.91（cross-both oc→ch_ip）
  - 引导提示使拒答召回提升中位数 1.9×
  - 文本不可答性检测比视觉缺失快约 4×（chaotic 分裂上最高 197×）
  - 87-99% 的 confabulation 包含 hallucinated premise
- **结论**：瓶颈是表达而非感知，模型内部能区分但输出端压制了不确定性信号

## 相关工作脉络
- **物理推理基准**（CLEVRER、IntPhys、PhysBench、Morpheus）：这些基准评估物理推理能力但不评估在证据不足时的选择性拒答；TRAPSBench 专门测量模型是否认识到"不做预测是被允许的"
- **不确定性与拒答**（Kendall & Gal、Geifman & El-Yaniv、SQuAD 2.0、AbstentionBench）：传统方法需要访问模型内部；本文通过 probing 和 steering 直接验证内部信号的存在与因果性
- **激活引导研究**（Turner et al.、Rimsky et al.、Li et al.）：此前应用于 truthfulness、sentiment、refusal，本文首次将其应用于视觉域的认识论不确定性因果控制
- **不可答视觉问题**（UNK-VQA、VisionTrap、TUBench、MM-UPD、CertainlyUncertain）：现有工作操作于单图像输入并通过语义扰动派生不可答性；本文扩展到视频且使用对比对消除编辑伪影

## 局限性与未来方向
- **机制分析范围**：探针和引导仅覆盖三个开源架构（Qwen3-VL-8B、Gemma 4 E4B、LLaVA-NeXT-Video-7B），封闭权重模型无法探针
- **Sim-to-real gap**：程序化设置是更容易的测试案例，需要扩展到真实世界视频
- **软体/流体物理**：当前仅限刚体物理，未来可扩展到软体、流体等更复杂场景
- **输出阶段干预**：关闭表征-输出鸿沟可能需要输出阶段的干预而非简单 scaling

## 研究启发与可借鉴点
1. **探针+引导的三合一验证范式**：通过线性探针（相关性）、引导式提示（行为激活）、激活引导（因果控制）三种方式交叉验证内部表征，为表征-行为鸿沟研究提供了可复用的方法论框架
2. **程序化最小视频对设计**：MuJoCo 生成对比对有效避免预训练污染并自带 ground truth，可迁移到其他需要因果推理评估的领域
3. **PECS 指标的联合约束设计**：Youden's J 统计量与准确率的乘积结构同时惩罚"过度拒答"和"从不拒答"，为校准评估提供了简洁且robust的度量思路
4. **Confabulation 分类 taxonomy**：HP/II/ES 三轴分类体系结合 Doubt in Reasoning 维度，为分析模型错误输出提供了结构化框架

## 关键术语表
**TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios，一个基于 MuJoCo 程序化生成的物理不确定性视频基准，用于评估 VLM 的选择性拒答能力
**PECS**：Penalized Epistemic Calibration Score，一个联合评估指标，等于准确率乘以 Youden's J 统计量（拒答召回减假拒答率），要求模型同时正确回答和选择性拒答
**Epistemic Restraint**：认识论克制，指模型在证据不足时主动选择不回答的能力
**Confabulation**：虚构，指模型在无充分证据时仍自信地提供答案的行为，87-99% 包含 hallucinated premise
**Void/Control Pair**：最小视频对，control 视频提供确定性可预测的输出，void 视频通过单一修改使结果不可计算
**Activation Steering**：激活引导，通过在特定层向隐藏状态添加方向向量来因果控制模型行为的干预技术
**Hallucinated Premise (HP)**：幻象前提，模型断言未在视觉证据中支持的 factual observation
**Youden's J Statistic**：J 统计量，衡量分类器区分能力的指标，等于灵敏度减特异度

## 可复现要素
- **数据集**：TRAPSBench 已公开发布，位于 https://github.com/facebookresearch/TRAPS-Benchmark，CC BY-NC 4.0 许可
- **代码/权重**：评测提示已公开；模型权重为各厂商公开版本（Qwen3-VL、Gemma 4 E4B、LLaVA-NeXT-Video-7B）
- **关键超参**：API 模型 temperature=0.8, top-p=0.95, max_completion_tokens=8000；本地模型 Qwen3-VL-8B 使用 bf16 16 uniform frames greedy decoding；Gemma/LLaVA 使用 32 frames at 2 FPS
- **Judge 面板**：Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus 三级多数投票
