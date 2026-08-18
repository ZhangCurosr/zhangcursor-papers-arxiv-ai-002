---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:18:47"
---

# 论文速读：TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express

## 一句话总结
本文提出过程序化生成的视频基准 TRAPSBench 与联合评价指标 PECS，系统验证了 VLM 在内部表征层能够区分“可答”与“不可答”的物理场景，但其自回归输出路径会抑制该信号，导致自发认知克制（abstention）能力极差；同时发现模型检测文本不可答性的能力约为检测视觉信息缺失的 4 倍，且链式思考（CoT）在某些架构下反而会放大虚构而非纠正校准。

## 研究问题与动机
1. **部署可靠性需求**：自主智能体常面对感官证据不足的输入（遮挡、混沌轨迹、问题模糊），此时“知道何时不回答”与“正确回答”同等重要，但现有物理推理基准均未系统评估证据不足时的选择性 abstention。
2. **评测指标缺陷**：传统 abstention recall 会奖励无差别拒答，accuracy 忽略认知维度，简单乘积无法惩罚 false abstention，缺乏能同时要求“该答时答对、该拒时拒掉”的鲁棒联合指标。
3. **表征-输出鸿沟存疑**：机制可解释性研究已证明 LLM 内部编码了真假、拒绝等信号，但在视觉-语言多模态领域，VLM 是否同样隐式编码了“视觉证据不足以支撑预测”的认识论不确定性，以及该信号为何无法流向输出，尚属空白。
4. **推理策略的双刃剑效应**：Chain-of-Thought 通常被认为提升可靠性，但其对认知不确定性的表达是促进还是抑制，缺乏跨架构的系统对比。

## 核心贡献（创新点）
1. **TRAPSBench 与 PECS 指标**：发布包含 1,404 对匹配可控/混沌物理视频的基准，并设计 Penalized Epistemic Calibration Score（PECS），通过 Youden's J 统计量双重惩罚“永远拒答”与“从不拒答”的退化策略，实现纯文本 judge 下的黑盒评测。
2. **证实表征-输出瓶颈而非感知瓶颈**：通过三种互补实验（引导提示激活、线性探针跨域迁移 AUROC 最高达 0.91、单层激活转向因果干预）一致证明 VLM 内部已编码答案可答性信号，但自回归解码路径将其压制，且该结论在 Qwen3-VL、Gemma、LLaVA 三个无共享训练链的开源家族中完全复现。
3. **揭示失效模式不对称性**：量化了文本不可答性（ill-posed question）比视觉信息缺失（visual void）更易被检测（平均约 4 倍差距），并发现 CoT 推理并非单调有益：Gemini Flash 能将内部怀疑转化为拒答，而 Qwen3-VL Think 反而覆盖自身怀疑继续虚构。
4. **提供因果机械解释与方向几何图谱**：通过 cross-dataset 激活转向发现，遮挡类（occlusion-family）方向编码了跨领域通用的“证据缺失”信号，而混沌类（chaotic-family）方向近乎正交且仅领域特定，从几何层面解释了为何某些不确定性更容易迁移与表达。

## 方法详解
- **最小视频对范式（Minimal Video Pair）**：基于 MuJoCo 刚性体物理引擎程序化生成配对视频。Control 视频包含完整可确定性预测的信息；Void 视频仅引入单一修改使结局无法从视觉证据推导，分为三类：① Occlusion（不透明遮挡物阻断关键区域，N=202）；② Chaotic Sensitivity（系统对初值极度敏感，视频在结果解析前截断，N=500）；③ Ill-Posed Questions（控制视频复用，但问题含文本级虚假前提或不可观测细节，N=702）。
- **PECS 评价指标**：`PECS = Acc × max(0, AbsRec − FalseAbs)`。Acc 为控制视频准确率；`AbsRec − FalseAbs` 即 Youden's J 统计量，衡量模型在可答/不可答样本间的辨别力。J≤0 时 PECS 强制归零，彻底封死“全拒答”“全作答”“随机”“完美但永不拒答”等退化策略。
- **纯文本 Judge 流水线**：由 Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus 组成三人评审团，仅接收模型生成的文本与 MuJoCo 确定性 ground truth，多数票投票。Abstention Detection 改编自 AbstentionBench 协议，精确识别“我无法判断/I don't know/没有足够证据”等语义；Correctness Evaluation 对数值答案允许 ±0.5 容差，分类答案要求精确字符串匹配。
- **线性探针（Linear Probing）**：冻结模型，在 37 层每层提取 hidden states，训练 l2 正则化逻辑回归探针预测 void/control 标签。训练集与测试集来自不同物理子场景，报告最佳层 AUROC，验证内部信号是否具备跨域泛化性。
- **激活转向（Activation Steering）**：在选定层 l 计算 void 方向向量 `v_l = (h_void − h_control) / ||h_void − h_control||`，在自回归生成每一步对 hidden state 施加扰动 `h → h ± α·v_l`（control 加 +α 诱导拒答，void 加 −α 压制拒答）。扫描 α∈{0,2,5,10} 观察拒答率的因果变化。

## 实验与结果
- **数据集与基线**：1,404 对视频，评估 16 个 VLM（覆盖 Gemini、Qwen3-VL、GPT-5、Gemma 4 E4B、LLaVA-Video-7B 五个家族），三种提示 regime（Standard / Guided / JSON），每模型 3 次独立推理，总计约 19.4 万评测对、38.7 万次调用。
- **主要结果**：
  - Standard  regime 下最优 PECS 为 **0.292**（Gemini 2.5 Flash，Acc 63.4%，AbsRec 49.4%）；Guided regime 下最优达 **0.568**（Gemini 3.1 Pro R-Low）。
  - 引导提示使主流视频原生模型的 AbsRec 中位数提升 **1.9×**（1.4–2.8×），且未显著损害控制准确率（多数模型 ±2pp），证明latent 克制能力普遍存在。
  - 跨架构复现模型（Gemma 4 E4B、LLaVA-Video-7B）在 Guided 下 PECS 仅达 0.215 与 0.101，仍未饱和。
- **视觉 vs 文本不对称**：所有 16 个模型在文本不可答性上检测更敏锐，混沌分割下差距达 3–25×（图像输入模型 GPT-5 系列最高 197×），中位数约 **4×**。
- **推理策略影响**：Gemini Flash vs NT 显示 CoT 提升 AbsRec 4–13pp；Qwen3-VL Think vs Instruct 呈现相反趋势，Think 版本在混沌不可答集上 AbsRec 反而更低（71.7% vs 82.3%），且其 reasoning trace 中 24% 表达了怀疑但最终输出仍覆盖怀疑进行虚构。
- **探针与转向**：跨数据集 LR 探针 AUROC 最高 **0.91**，即便将训练/测试集严格限制为模型已自信虚构的 void 样本（Cf→Cf），跨域可解码性仍保持（四类转移均值差异 ≤0.03）。激活转向证实遮挡类方向跨域强迁移（α=10 时 control 拒答率 75% vs 15%），混沌类方向近乎正交且仅领域内有效。

## 相关工作脉络
1. **物理推理基准（CLEVRER、IntPhys、PhysBench、Morphheus）**：聚焦模型能否预测物理结果，预设答案必然存在，未引入选择性拒答评测维度；TRAPSBench 测量的是“模型是否认识到预测不被许可”，填补这一空白。
2. **不确定性估计与选择性预测（Kendall & Gal、Geifman & El-Yaniv、AbstentionBench）**：多依赖模型内部概率/ logits 访问或仅在纯文本任务（SQuAD 2.0）验证；本文首次将 abstention 评测扩展至视频物理场景，并设计无需 logits 的纯文本联合指标 PECS。
3. **视觉不可答问题基准（UNK-VQA、VisionTrap、TUBench、MM-UPD、CertainlyUncertain）**：基于单张图像的语义扰动或 inpainting 构造不可答样本，易引入编辑伪影；TRAPSBench 基于物理动力学生成对比对，天然无 artifact，且支持黑盒 API 评测。
4. **LLM 元认知与机制可解释性（Burns et al. 真值表征、Arditi et al. 拒绝方向、Rimsky et al. 激活转向）**：已有工作证明 LLM 内部编码真值/拒绝信号且可通过定向干预因果调控；本文将该范式首次移植至视觉-语言多模态的认知不确定性，并揭示表征存在但输出通路被门控的分离现象。
5. **最小视频对方法（Krojer et al. 2025）**：利用最小对消除 shortcut reasoning，但未测试 abstention；本文继承该对比范式，将其目标从“防作弊”升级为“测认知边界”。

## 局限性与未来方向
- **机械分析范围受限**：探针与转向实验仅覆盖三个无共享训练链的开源家族（Qwen3-VL-8B、Gemma 4 E4B、LLaVA-Video-7B），闭源模型仅能做行为验证；提示门控强度与方向几何具有家族特异性，结论不可直接外推至所有架构。
- **合成场景简化**：MuJoCo 刚性体动力学是理想化易例，未涵盖软体、流体、真实世界复杂观测噪声；sim-to-real 鸿沟尚未验证。
- **因果声明边界**：激活转向属干预充分性证明，尚未进行完整中介分析（如 control/void 配对间的 activation patching），方向向量与“认知不确定性”概念的等价性依赖收敛证据而非直接标签监督。
- **未来方向**：扩展至软体/流体物理、真实世界视频；针对 void 视频进行 fine-tuning 以打通表征-输出链路；探索 generative rollout 下的不确定性建模；开发输出阶段干预（output-stage interventions）而非单纯扩大模型规模。

## 研究启发与可借鉴点
1. **程序化最小对（Minimal Pairs）+ 纯文本 Judge 的评测范式**：利用模拟器 ground truth 自动生成 matched control/void 对，彻底规避数据污染与标注噪声；Judge 仅看文本不仅符合黑盒部署场景，也避免了 judge-VLM 视觉感知混淆 target-VLM 输出评分的问题，值得在其它因果评测中复用。
2. **PECS 联合指标的设计思路**：将任务能力（Acc）与认识论辨别力（Youden's J）相乘并 clamp 于零，一举封死“全答”“全拒”“随机”“完美但永不拒答”四类退化策略，可作为通用“可靠回答能力”基准的参考模板。
3. **Cf→Cf 探针验证协议**：将训练/测试集严格限定为“模型已自信犯错”的样本子集，可排除“探针仅学会识别模型已表现出的行为”这一行为学混淆，是验证 latent 表征真实性的强有力控制实验，适用于其他表示-行为分离研究。
4. **激活转向作为表示-输出鸿沟的诊断工具**：通过单层方向注入/抽取直接验证表征是否因果驱动某行为，比相关性探针更进一步；本文揭示“标准提示下 +α 转向几乎无效（0→1%），但 −α 转向有效（10→4%）”的单向门控现象，为后续输出层干预（如 post-hoc routing、decode-time intervention）提供了明确的理论靶点。
5. **Confabulation Taxonomy（HP/II/ES/Doubt）**：将错误响应拆解为幻觉前提、无效推理、认知投降、推理潜伏怀疑四个独立轴，尤其揭示“Think 版本怀疑率最高（24%）但转化拒答率最低”的反直觉现象，为 CoT 训练目标的奖励设计优化提供了可直接照抄的标注框架。

## 关键术语表
- **TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios 的缩写，本文发布的过程化生成视频基准，用于评估 VLM 在物理证据不足时的选择性拒答能力。
- **PECS（Penalized Epistemic Calibration Score）**：结合准确率与 Youden's J 统计量的乘积指标，强制要求模型在可答时答对、在不可答时拒答，同时对 false abstention 施加惩罚。
- **Epistemic Restraint（认知克制/认识论克制）**：模型在感知到证据不足以支撑确定性结论时，主动选择拒答或表达不确定性的能力。
- **Minimal Video Pair Paradigm（最小视频对范式）**：通过单一变量修改（遮挡、截断、问题篡改）构造 control/void 配对视频，隔离模型对“信息缺失”的识别能力。
- **Visual vs. Textual Unanswerability Gap**：模型检测文本级不可能性（如虚假前提）的能力显著强于检测视觉信息缺失（如混沌截断）的现象，本文量化为约 4 倍中位差距。
- **Activation Steering（激活转向）**：在自回归生成的每一步向特定层的 hidden state 注入/抽取对比方向向量，以因果方式操控模型行为（如诱导或压制拒答）。
- **Confabulation / Hallucinated Premise（虚构/幻觉前提）**：模型在未观测到的视频内容上编造具体事实陈述（如虚构碰撞时间戳、物体位置），而非单纯推理错误，本文指出这是 VLM 拒答失败的主要形态（87–99%）。
- **Prompt Gating（提示门控）**：指某些模型在标准提示下存在单向输出约束，即使内部表征已编码不确定信号，自回归解码路径也会压制“拒答”动作的释放；引导提示可部分或完全打开该门。

## 可复现要素
- **数据集**：TRAPSBench 已开源，地址 `https://github.com/facebookresearch/TRAPS-Benchmark`，协议 CC BY-NC 4.0；包含 1,404 对 MuJoCo 刚性体视频、配套问题与 ground truth。
- **代码/权重**：
