---
title: "LLaDA-Image-Building-Strong-Image-Generators-with-Fully-Open"
source: https://arxiv.org/pdf/2609.03796v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-05 02:01:41"
field: "开源图像生成模型"
keywords: ["image generation", "diffusion transformer", "open-source model", "text-to-image", "image editing", "model distillation"]
innovations: ["纯图像预训练渐进路线建立视觉先验，无需大规模配对数据", "单流 DiT 统一架构同时支持文生图与图像编辑", "TwinFlow 蒸馏将多步模型压缩至 2-4 步快速推理"]
benchmarks: ["Qwen-Image-Bench", "LongText-Bench", "CVTG-2K", "GenEval", "DPG-Bench", "GEdit-Bench"]
---

# 论文速读：LLaDA-Image-Building-Strong-Image-Generators-with-Fully-Open

## 一句话总结
本文提出 LLaDA-Image，一个从纯无标签图像数据开始的 6B 参数 Diffusion Transformer，通过两阶段预训练建立强大视觉先验，再逐步加入语言对齐与指令微调，最终开源一个在质量、美学、对齐度上均达开源 SOTA 的文生图与图像编辑统一模型，并支持 Turbo 蒸馏至 2–4 步推理。

## 研究问题与动机
- **开源模型依赖大规模配对图文数据**，难以在不使用大量 caption 的情况下达到高保真生成；如何仅凭图像预训练就能建立强大的视觉先验？
- **编辑能力与生成能力通常由独立分支/模型支撑**，导致任务特定的检查点负担重、部署复杂；能否用单一架构统一文生图与编辑？
- **大规模扩散模型训练稳定性与扩展性不足**，尤其是高分辨率、长序列、长尾分布下的训练收敛困难；如何设计更稳定的训练范式？
- **现有开源方案在文字渲染、文化概念理解、长文本生成上仍有明显短板**；如何在保持高生成质量的同时提升细粒度可控性？

## 核心贡献（创新点）
1. **纯图像驱动的渐进式预训练路线**：从 98% 真实图像的 220M 样本开始，先做图像-only 预训练与中期训练，再引入少量 SFT，无需大规模配对数据即可获得高保真生成先验。
2. **单流 DiT 统一架构**：将 dLLM-based VLM 理解模块、Residual Query Adapter 连接器与 Single-stream DiT 联合训练，语义条件与视觉 token 通过 joint self-attention 交互，同时支持文生图与参考图像编辑。
3. **训练稳定性与可扩展性优化**：全程使用参数无关 RMSNorm、Muon optimizer、logit-normal timestep sampling 与 aspect-ratio-bucketed 可变分辨率训练，配合检查点合并抑制采样噪声波动。
4. **Turbo 蒸馏与快速推理**：基于 TwinFlow（DMD2 + self-adversarial flow training）将多步模型蒸馏为 LLaDA-Image Turbo，仅用 2–4 采样步即可在开源基准上保持竞争力。

## 方法详解
- **模型架构**：6B 参数 Single-stream DiT，冻结基于 LLaDA2.0-Mini dLLM 的视觉-语言理解模块（含 SigLIP-VQ 视觉编码器）。
- **三大组件**：
  - **dLLM-based VLM**：处理文本指令，提供语义条件。
  - **Understanding-to-Generation Connector**：由 Residual Query Adapter (RQA) 与轻量 Transformer Connector 组成，将 VLM 特征映射到 DiT 条件空间。
  - **Single-stream DiT**：所有 Transformer 块联合处理语义条件与视觉 token，通过 joint self-attention 实现语义-像素交互。
- **编辑通路设计**：编辑指令走 RQA-VLM-Connector 路径；参考图像绕过 VLM，通过 SigLIP-VQ 提取语义特征（经 DiT 专用 branch $b_\omega$）与 FLUX.2 VAE 提取像素级 latent，与噪声 latent 拼接后输入 DiT。
- **训练策略**：
  - 分辨率渐进：Image-only Pre-training $256^2$ → Mid-training $512^2$（aspect-ratio bucketed）→ SFT Alignment $512^2 \rightarrow 1024^2$（progressive）→ Refinement & Editing $1024^2$。
  - 全程参数无关 RMSNorm、Muon optimizer。
  - logit-normal timestep sampling（$P_{\text{mean}}=0.8, P_{\text{std}}=0.8$）加强对高噪声阶段监督。
  - **Block diffusion CoT SFT**：block size $b=32$，masking ratio $\rho=\cos(r\pi/2), r\sim\mathcal{U}(0,1)$，每 batch 消费两次互补 mask。
- **蒸馏方案**：使用 TwinFlow 将多步模型蒸馏为 LLaDA-Image Turbo，采样步数压缩至 2–4。

## 实验与结果
- **数据集与基线**：训练数据 220M generation-training samples（98% 真实图像）；对比基线包括 GPT-Image 2、Nano-Banana 2.0/Pro、Seedream 5.0/4.5、FLUX.2 Max/Pro、Qwen-Image 2512 等闭源与开源模型。
- **Qwen-Image-Bench**：LLaDA-Image 英文轨 **53.53**，中文轨 **53.38**，均超越次优开源模型 Z-Image Turbo，且在 Quality、Aesthetics、Alignment 三项开源第一；Turbo 版本英文轨 **51.00**，中文轨 **50.98**（4 步采样）。
- **文字渲染**：LongText-Bench 英文 **0.923**、中文 **0.913**；CVTG-2K 平均词准确率 **0.875**（排名第二），NED **0.945**，CLIPScore **0.818**。
- **诊断基准**：GenEval Overall **0.85**，Single Object **1.00**，Counting **0.53**（短板）；DPG-Bench Overall **87.48**，四维均衡。
- **编辑能力**：GEdit-Bench-EN Overall **7.336**，CN **7.294**；语义一致性（G_SC）强于感知质量（G_PQ）。
- **定性展示**：12 张“疑似真实照片”全部为模型生成；中英文文本渲染与复杂风格生成具备竞争力。

## 相关工作脉络
- **闭源 SOTA 基线**：GPT-Image 2、Nano-Banana 2.0/Pro、Seedream 5.0/4.5、FLUX.2 Max/Pro、Imagen 4.0 等，本文定位在开源阵营内实现质量与对齐的领先。
- **开源对照模型**：Z-Image Turbo、SenseNova U1.5 Preview、Qwen-Image 2512、LongCat-Image、HunyuanImage 3.0、FLUX.1 [Dev] 等，本文强调从纯图像预训练起步而不依赖大规模配对数据。
- **技术前作**：IOMM（Sun et al., 2026c）、DMD2（Yin et al., 2024; Liu et al., 2026）、TwinFlow（Cheng et al., 2026）、FLUX.2 VAE、SigLIP-VQ，本文将这些组件整合为统一训练流程。
- **架构参考**：LLaDA 2.0 Uni、BLIP3-o、Janus Pro 等 VLM-DiT 融合思路，本文通过 RQA connector 与单流 joint attention 实现更紧密的语义-像素交互。
- **训练优化**：Muon optimizer、RMSNorm、logit-normal timestep sampling、block diffusion 等来自扩散模型可扩展性研究，本文将其系统化应用于 6B 规模图像生成。
- **蒸馏加速**：TwinFlow 基于 DMD2 的 self-adversarial flow training，本文用于将多步模型压缩至 Turbo 版本。

## 局限性与未来方向
- **世界知识与文化概念覆盖有限**：专业主题、长尾文化概念的生成能力仍有提升空间。
- **长文本与多区域渲染稳定性不足**：当前在长序列、多区域文字渲染上弱于专用最强模型。
- **分辨率上限为 $1024^2$**：原生 2K 生成尚未评估，高分辨率可扩展性待验证。
- **编辑感知质量（G_PQ）落后于语义一致性**：与专用编辑模型存在差距，需改进感知保真度。
- **未来方向**：强化 grounding 与排版、扩展知识密集概念、原生 2K 训练课程、闭环 agentic 生成（规划-评估-编辑）。

## 研究启发与可借鉴点
- **纯图像预训练路线可复用于其他多模态生成任务**：证明无需大量配对数据即可建立强大视觉先验，为低资源语言/领域扩展提供新思路。
- **单流 joint attention 架构值得迁移**：语义条件与视觉 token 在统一 Transformer 块中交互，可简化多模态模型的设计复杂度。
- **训练稳定性组合技术（RMSNorm + Muon + logit-normal sampling + block diffusion）可系统化集成**：为大规模扩散模型训练提供稳定范式。
- **TwinFlow 蒸馏到 2–4 步的方案可移植到其他 DiT 模型**：实现开源模型的高质量快速推理。
- **数据质量审核 pipeline（隐私/水印/文字转写/描述忠实度）可复用**：为大规模图像数据集的自动化清洗提供参考框架。

## 关键术语表
- **dLLM-based VLM**：基于 LLaDA2.0-Mini 骨干的冻结视觉-语言理解模块，提供语义条件。
- **Single-stream DiT**：所有 Transformer 块联合处理语义条件与视觉 token 的单流扩散 Transformer。
- **Residual Query Adapter (RQA)**：轻量适配器，将 VLM 特征映射到 DiT 条件空间。
- **TwinFlow**：基于 DMD2 与 self-adversarial flow training 的多步到少步蒸馏方法。
- **Block diffusion CoT SFT**：按 block 随机掩码的 chain-of-thought 指令微调，提升理解能力。
- **Logit-normal timestep sampling**：按 logit-normal 分布采样噪声级别，强化高噪声阶段监督。
- **Aspect-ratio-bucketed 训练**：将训练样本按宽高比分桶，保证各 rank 处理相似 token 数。
- **Muon optimizer**：参数无关优化器，提升大规模训练的可扩展性与稳定性。

## 可复现要素
- **数据集**：220M generation-training samples，其中 98% 为真实图像；具体数据源论文未完全公开，但提供了过滤标准（像素数 > $1024^2$、file-size/pixel ≥ 0.15 bytes、ArtiMuse 分数 ≥ 60、DeQA-Score ≥ 4.0）。
- **代码/权重**：模型架构、训练 recipe、Turbo 蒸馏方案均已开源；checkpoint merging 策略也公开。
- **关键超参**：训练分辨率演进（$256^2 \rightarrow 512^2 \rightarrow 1024^2$）、batch size（4,608–6,400）、学习率（$4\times10^{-4} \rightarrow 5\times10^{-5}$）、Muon optimizer、RMSNorm、logit-normal sampling 参数（$P_{\text{mean}}=0.8, P_{\text{std}}=0.8$）、block diffusion masking ratio 公式等均详细给出。
- **蒸馏配置**：TwinFlow 蒸馏至 2–4 步，LR $5\times10^{-6}$，batch 256，EMA ratio 0.995。
