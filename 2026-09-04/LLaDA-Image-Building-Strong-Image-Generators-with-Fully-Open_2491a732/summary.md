---
title: "LLaDA-Image-Building-Strong-Image-Generators-with-Fully-Open"
source: https://arxiv.org/pdf/2609.03796v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-04 12:05:15"
---

# 论文速读：LLaDA-Image-Building-Strong-Image-Generators-with-Fully-Open

## 一句话总结
LLaDA-Image 构建了一个纯开源的 6B Diffusion Transformer 图像生成器，通过“单模态视觉预训练→图文对齐→精修与蒸馏”的分阶段真实数据主导课程，在中等数据预算下实现了高稳定性训练与 2–4 步 Turbo 推理，并在多项开源基准上达成领先。

## 研究问题与动机
- 现有开源图像生成系统难以在数据预算、大规模训练稳定性与部署效率三者之间取得平衡，往往依赖闭源体系独占的海量私有数据与配方。
- 多模态理解（VLM）与图像生成（DiT）之间缺乏统一、轻量且可复用的条件注入机制，导致图文对齐成本高、图像编辑能力受限。
- 百亿参数级 DiT 从零训练时，常规优化器与归一化方案易出现数值梯度异常与长程不稳定，制约开源模型的规模化复现。
- 开源基座普遍缺乏对复杂排版、多区域文字渲染与高精度风格控制的支持，限制了在创意设计与工业场景中的直接落地。

## 核心贡献（创新点）
1. 提出分阶段训练配方（视觉先验→图文对齐→精修蒸馏），以 98% 真实图像为主的中低数据预算打破开源模型依赖海量合成数据的惯性，实现高质量可复用生成器。
2. 设计 RQA（Residual Query Adapter）与轻量 Transformer Connector 的统一条件注入模块，使冻结的 dLLM-VLM 骨干能高效暴露生成友好隐状态，单 Checkpoint 同时支持 T2I 与图像编辑。
3. 引入参数无关 RMSNorm 与 Muon 优化器组合，显著提升 6B DiT 从零训练的长程稳定性与收敛效率，降低对显存与算力的高强度依赖。
4. 发布完整的 6B 模型权重、推理代码及 Base/Turbo 四格式（FP16/FP8）Checkpoint，并结合 TwinFlow 蒸馏将推理步数压缩至 2–4 步，兼顾生成质量与部署吞吐。

## 方法详解
- **统一架构（单 Checkpoint 支持 T2I + 编辑）**：骨干由冻结的 SigLIP-VQ 视觉编码器 + LLaDA 2.0 Mini 语言模型构成 dLLM-based VLM；RQA 在 VLM 前附加可学习查询 token 并 cross-attend 到输入序列，使 VLM 暴露利于生成的隐藏状态；轻量 Transformer Connector 将 VLM hidden states 投影至 DiT 条件空间。
- **单流 DiT 核心**：6B 参数 Single-stream Diffusion Transformer，语义条件与视觉 token 在同一 self-attention 层交互；全部归一化层替换为参数无关 RMSNorm（Recipe #1），配合 Muon 优化器保障长程稳定。
- **编辑双路条件机制**：参考图不走 VLM，分为两路输入 DiT：
  - **语义支**：SigLIP-VQ → DiT 专用 embedder + 2 层 Transformer → 与 text-condition 拼接（对应原文 Eq.6/7）。
  - **像素支**：FLUX.2 VAE 编码干净参考图 latent，与加噪目标 latent 直接 concat 输入 DiT（对应原文 Eq.8），完整保留参考信息。
- **分阶段训练管线**：
  1. **CoT SFT**（理解骨干）：~2.6M packed seq（16384 tok），512²，AdamW，lr 1e-5→1e-6，Gen:Und:Text=9:9:2。
  2. **Image-only Pre-training**：256²，~24576 样本，batch 4608，lr 4e-4，Muon，EMA 关闭。
  3. **Image-only Mid-training**：512²（aspect-ratio bucketed），~6400 样本，batch 2048，lr 2e-4，引入可变分辨率策略。
  4. **SFT 512² 对齐**：~2880 样本，batch 2880，lr 5e-5，EMA 0.9995。
  5. **SFT 512²→1024² 过渡**：1024² bucketed，lr 3e-5。
  6. **Refine + Editing**：T2I:I2I=1:1 混合精修。
- **Turbo 蒸馏**：采用 TwinFlow 蒸馏管线，将多步去噪轨迹压缩至 2–4 步推理，在保持分布保真度的同时大幅提升生成吞吐量。

## 实验与结果
- **数据配置**：训练集包含 220M 生成样本（98% 为真实图像），>90% 支持纯图像训练；配对 SFT 中真实图像占比仍 >70%。
- **Qwen-Image-Bench（T2I）**：LLaDA-Image EN 50.98，LLaDA-Image Turbo CN 50.81，位列**开源模型第一**；GPT-Image 2 以 EN 65.23 / CN 64.69 遥遥领先。
- **LongText-Bench（长文本渲染）**：LLaDA-Image EN 0.923 / ZH 0.913，双语差距小；Qwen-Image 2512（EN 0.956 / ZH 0.965）等专用强模型仍占优。
- **CVTG-2K（复杂视觉文本）**：平均词准确率 0.875（仅次于 SenseNova U1.5 Preview 0.887），NED 0.945，CLIPScore 0.818；多区域稳定性强（2 区域 0.892 → 5 区域 0.857，降幅较小）。
- **GenEval（对象诊断）**：Single Object
