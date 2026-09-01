---
title: "SonarLLM-A-Native-Sonar-Optical-Multimodal-Large-Language-Mo"
source: https://arxiv.org/pdf/2608.24325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:03:48"
field: "水下多模态感知与语言理解"
keywords: ["Multimodal Large Language Model", "Underwater Perception", "Sonar-Optical Fusion", "Reliability-Aware Fusion", "Cross-Modal Alignment", "Heterogeneous Sensor Fusion"]
innovations: ["声纳原生编码器 PSVT（多尺度 Stem + 范围-方位位置编码）", "模态特定物理感知增强（Optical-VFE / Acoustic-VFE）", "可靠性感知分层融合 AGFM（全局门控 + 均值保持 token 调制 + DeepStack 注入）"]
benchmarks: ["SonarBench", "RGBS50", "UMOD", "UATD", "SCTD"]
---

# 论文速读：SonarLLM-A-Native-Sonar-Optical-Multimodal-Large-Language-Mo

## 一句话总结
论文提出 SonarLLM，一个将成像声纳作为**原生感知模态**的声纳-光学多模态大语言模型，通过模态特定的物理感知特征增强和可靠性感知分层融合，实现水下异构传感的互补理解；同时提出 SonarBench 配对基准，在固定场景与声纳观测的前提下逐步退化光学质量，精准隔离单模态能力与跨模态互补收益。

## 研究问题与动机
1. **现有 MLLM 对声纳不兼容**：基于自然光图像的编码器无法适配声纳的距离-方位几何结构及斑点噪声、混响、声学阴影等声学伪影。
2. **光学退化不可逆、声纳几何稳定但噪声不同**：浊度增加会迅速抹除颜色、纹理与边界，而声纳保留结构信息却受声呐噪声和距离衰减影响，两者可靠性高度依赖观测环境。
3. **固定融合策略脆弱**：已有声纳-光学系统多为任务级检测/跟踪专用，缺乏开放语言推理能力；简单拼接异构输入容易受不可靠传感器主导。
4. **缺乏可控评估协议**：既有水下评测资源按模态或任务割裂，无法分离"更强单模态建模"与"真正互补证据"的贡献。

## 核心贡献（创新点）
1. **声纳原生编码器 PSVT**：从 Qwen3-VL 视觉塔迁移预训练先验，引入多尺度 Sonar Stem 与范围-方位位置编码，使声纳成为独立感知模态而非辅助 RGB 图像。
2. **模态特定的物理感知特征增强（VFE）**：Optical-VFE 使用可学习查询 + 冻结 DINOv2-L 结构参考修正散射退化；Acoustic-VFE 用可学习声纳查询 + token 范围参数化补偿混响与距离衰减，二者在特征空间而非原始图像空间进行校正。
3. **可靠性感知分层融合 AGFM + DeepStack**：AGFM 先估计 token 质量并聚合为模态级可靠性权重，再通过均值保持 token 调制重加权后注入 LLM 的第 8/16/24 层，避免异构表征过早压缩。
4. **四阶段渐进式训练**：Stage I MAE 声纳域适应 → Stage II 声学语义分类 → Stage III 配对跨模态对齐 + 门控监督 → Stage IV LoRA 指令微调，降低异构优化目标间的干扰。
5. **SonarBench 配对基准**：固定场景和声纳观测、仅渐变光学退化（清晰/浑浊/重浊），精确测量跨模态互补增益随光学可靠性下降的变化曲线。

## 方法详解
**整体架构**：保留 Qwen3-VL-8B 的光学编码器 Eo，新增独立声纳编码器 Es（PSVT）。每路编码器输出最终特征 Fm 和 3 个中间特征 {Fm(k)} (k=1,2,3 对应 block 8/16/24)。最终特征经 VFE 增强后由 AGFM0 重加权；中间特征绕过 VFE 经 AGFMk 重加权后通过 DeepStack 注入语言模型对应层。

**声纳原生表示（PSVT）**：
- 多尺度 Sonar Stem：`ΔIs = Conv1×1[f7(f5(f3(Is)))]`，`Ĩs = Is + γΔIs`，零初始化系数 γ 渐进引入声纳结构。
- 范围-方位位置编码：`zij^s = zij^base + λp[φr(ri) + φθ(θj)]`，显式适配距离-方位几何。

**模态特定增强**：
- Optical-VFE：可学习查询 eb 通过 MHA 估计散射相关特征 C_o，再与冻结 DINOv2-L 的结构参考 Do 融合：`F̄o = LN(Fo^c + Ψo(Do))`。
- Acoustic-VFE：可学习查询 er 估计混响分量 Cs，并通过 token 范围参数化补偿距离衰减：`Ḟs[i] = LN(G(ri)·Fs^c[i])`，其中 `G(ri) = exp[2log(ri/rmin) + 2αc(ri−rmin)]`。

**AGFM 可靠性感知融合**：
- Token 质量估计与模态级聚合：`q̄m(k) = Nm^−1 Σ qm,j(k)`，`g^(k) = softmax(q̄^(k)/τk)`。
- 均值保持 token 调制：`ρm,i(k) = um,i(k)/ūm(k)`，`f̃m,i(k) = gm(k)·ρm,i(k)·fm,i(k)`。
- 最终特征直接作为多模态上下文；中间特征经模态特定 DeepStack merger 注入 LLM 层 8/16/24。

**渐进训练四阶段**：
- **Stage I**：无标签声纳 MAE（掩码比 0.60，40K 图像），仅训 PSVT + decoder。
- **Stage II**：23 类声纳分类 CE，仅训 PSVT/Stem + 分类器。
- **Stage III**：配对数据 + 随机光学退化（p=0.7，η~U(0.1,1.0)），损失 `L_align = L_NCE + 0.5L_hier + 0.2L_gate + 0.1L_ent`，门控目标 πs(η) = clamp(0.5+0.4η, 0,1)，显式将退化强度映射为声纳权重监督信号。
- **Stage IV**：635K 指令数据（533K 声纳 + 102K 光学），LoRA r=128 α=256，lr=1e-4。

## 实验与结果
**数据集与基准**：SonarBench 含 25 个子集（4 声纳/12 光学/9 融合），每子集 150 QA，来自 held-out RGBS50 和 UMOD 序列，训练-评测无序列重叠。

**基线**：Qwen3-VL-8B、InternVL3.5-8B、MiniCPM-V 4.5、Qwen3.6-35B-A3B、Qwen3.8-27B、OceanGPT-o-7B、NAUTILUS-7B、Qwen3-VL-8B+LoRA（同源对照组）。

**主要结果**：
- **声纳-only 宏观准确率 72.0%**，超最强基线（Qwen3.8-27B: 29.1%）达 **+34.4 个百分点**，跨越识别(64.7%)、计数(84.7%)、VQA(66.5%) 三项稳定领先；声纳-only 描述 GOOD 率 33.3% vs 基线最佳 12.0%。
- **融合输入宏观准确率 68.7%**，超最佳基线 **+25.1 点**。
- **光学输入**达到 50.9%，在全部识别与 VQA 子集上排名第一；光学计数在清晰/重浊条件下稍弱。
- **受控退化互补增益**（融合-光学）：识别从清晰 +6.0 增至重浊 **+36.0 点**；计数从清晰 +6.0 增至重浊 **+36.0 点**；VQA 从 +7.1 增至 +14.0 点。融合性能本身仅波动 −2.6 ~ +0.6 点。
- **AGFM 内部响应**：声纳权重从 0.491 单调升至 0.795，Spearman ρ=0.749，95.3% 场景满足非递减。
- **对照组证明**：Qwen3-VL-8B+LoRA 与 SonarLLM 同 Stage-IV 数据与 LoRA，但声纳/光学/融合分别低 34.4/14.0/36.5 点；10,000 次 bootstrap 95% CI 下限均为正。
- **消融**：移除 Polar PE 降 13.5/10.1 点（融合/声纳宏观），移除 Sonar Stem 降 8.2/7.5 点；移除 Dual VFE 使融合/光学/声纳分别降 2.3/5.3/4.4 点；移除 DeepStack 降 6.2 点声纳宏观；AGFM vs 等权融合在重浊下差异 2.7 点。
- **效率**：参数量 10.3B（vs 8.77B 骨干），峰值显存 21.6GB（+22.0%），prefill 137.5ms（+78.1%），解码吞吐 21.1 tok/s（−45.1%）。

## 相关工作脉络
1. **MarineGPT / NAUTILUS / AquaticCLIP / OceanGPT**：水下视觉-语言模型，聚焦光学模态，未处理声纳原生表示或异质跨模态融合。
2. **RGBS50 / UMOD / SOVIS**：声纳-光学配对数据集，但面向任务级检测/跟踪，未连接开放语言推理。
3. **LLaVA / BLIP-2 / Flamingo / Qwen3-VL(DeepStack)**：通用 MLLM 多模态融合机制，假设输入视觉同质，未显式建模不同成像几何与环境依赖可靠性。
4. **UATD / SCTD**：声纳感知数据集，但缺乏配对光学退化控制与语言评测。
5. **NautData / UWBench / UIEB / EUVP**：水下视觉语言/图像增强基准，均以光学为中心，无法隔离声纳互补贡献。
6. **Gated Multimodal Units (Arevalo et al.) / Resilient Sensor Fusion (Park et al.)**：可靠性感知融合理论，但缺乏声纳几何适配与水下配对评估验证。

## 局限性与未来方向
1. **受控协议不足以覆盖自然浊度分布**：仅模拟了亮度衰减、白噪声、彩色眩光和高斯模糊，未涉及非均匀散射、动态悬浮颗粒、声纳特有故障（如多径、旁瓣干扰）及时间错位。
2. **AGFM 并非每问最优模态选择器**：融合计数/VQA 在部分场景不优于单模态，说明声纳并非总是有益的，需任务感知路由。
3. **未引入时序建模**：当前为单帧处理，实际水下感知常依赖视频时序一致性。
4. **作者自述未来方向**：扩展至真实配对数据、任务感知路由、时序声纳-光学推理。

## 研究启发与可借鉴点
1. **配对基准设计范式**："固定场景+固定声纳观测，仅渐变单一模态退化"的可控协议，可直接迁移至其他异构传感器（如红外-可见光、雷达-激光）的互补性量化评测。
2. **模态特定物理感知增强（VFE）思路**：用可学习查询提取退化分量、再用冻结预训练结构参考作先验校正，适用于任何具有明确成像退化模型的模态（如低光照、运动模糊、热噪声）。
3. **可靠性门控 + 均值保持 token 调制**：AGFM 将全局模态权重与局部 token 重分布解耦，避免异构序列过早压缩，可复用于任意双模态/多模态 LLM 融合。
4. **四阶段渐进训练**：域适应 → 语义分类 → 跨模态对齐+门控监督 → 指令微调，降低异构目标干扰；可推广至新传感器类型（如事件相机、LiDAR）接入 MLLM 的训练协议设计。
5. **与团队方向结合机会**：若团队关注水下/水下机器人感知，可直接复用 SonarBench 协议扩展至 sonar-video 或 sonar-infrared 多模态；若关注异构融合，AGFM 的可靠性感知分层注入可作为通用插件。

## 关键术语表
**MLLM（Multimodal Large Language Model）**：融合视觉/语言等多模态输入的大语言模型，支持开放-ended 理解与生成。
**SonarLLM**：本文提出的声纳-光学原生多模态 LLM，将成像声纳视为独立感知模态而非辅助 RGB 图像。
**PSVT（Polar-aware Sonar Vision Transformer）**：从 Qwen3-VL 视觉塔迁移初始化的声纳专用编码器，含多尺度 Stem 与范围-方位位置编码。
**VFE（Visual Feature Enhancement）**：模态特定的特征增强模块，Optical-VFE 修正散射退化，Acoustic-VFE 补偿混响与距离衰减。
**AGFM（Adaptive Gated Fusion Module）**：可靠性感知融合模块，估计 token 质量并输出模态级权重，对 token 做均值保持调制后注入 LLM 多层。
**DeepStack**：源自 Qwen3-VL 的分层视觉-语言特征注入机制，将中间层视觉特征重加权后分别送入 LLM 的第 8/16/24 层。
**SonarBench**：本文提出的配对水下感知基准，覆盖识别/计数/VQA/描述四项任务，支持声纳-only/光学-only/融合三种输入，在清晰/浑浊/重浊三档光学退化下评估。
**InfoNCE**：对比学习损失，用于声纳-光学配对特征的全局对齐。

## 可复现要素
- **数据集**：训练使用 RGBS50、UATD、SCTD、DeeperSense、FLC+FLS、OceanGym、OceanInstruct、OceanPile 的声纳/配对数据；评测使用 held-out RGBS50 和 UMOD 序列。论文未明确声明公开状态，"论文未提及"开源承诺。
- **代码/权重**：论文未明确声明开源，SonarLLM 基于 Qwen3-VL-8B（开源）构建，PSVT/AGFM/VFE 权重未声明开源，**代码未明确提及**。
- **关键超参**：输入分辨率 448×448；Stage I 掩码比 0.60；Stage II lr=5e-5；Stage III 光学退化概率 0.7、η~U(0.1,1.0)、lr=1e-4；Stage IV LoRA r=128 α=256 lr=1e-4；总参数 10.3B（骨干 8.77B + 新增 1.53B）。
