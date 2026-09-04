---
title: "Why-Gated-DeltaNet-Survives-4-Bit-Quantization-NVFP4-W4A4-f"
source: https://arxiv.org/pdf/2609.04098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:27:32"
---

# 论文速读：Why-Gated-DeltaNet-Survives-4-Bit-Quantization-NVFP4-W4A4-f

## 一句话总结
本文针对 Qwen3.8-27B 混合架构 LLM，首次实现了全 496 个线性层（含 GDN 门控投影）的 NVFP4 W4A4 后训练量化，证明其任务精度与 BF16 相当且预填充更快；并通过分层机制研究揭示 GDN 的 delta 递推结构与门控非线性参数化天然“保护”了量化噪声，推翻了“递归误差会随上下文累积”的行业直觉。

## 研究问题与动机
- **核心问题**：混合 LLM 中正快速用线性注意力（如 GDN）替代 softmax 注意力，但现有 4-bit 方案（含模型官方 FP8 版及社区 Unsloth/RadixArk）均将 GDN 块（尤其是控制衰减 $\alpha_t$ 和写入强度 $\beta_t$ 的门控投影 $a, b$）保留在 8/16-bit。
- **直觉假设**：递归状态 $S_t$ 跨越数万 token 传播，单步量化误差必然累积，门控误差累积最快，因此必须保护。
- **现有方法不足**：缺乏对线性注意力模块真实量化鲁棒性的系统性验证；服务栈测量存在隐蔽偏差（如 fused GEMM 尺度失配、thinking 模型原始补全评测失效），导致评估结论不可靠。
- **研究动机**：验证“全 4-bit 量化 GDN 是否可行”，从架构动力学角度给出因果解释，并建立正确的 hybrid 模型 NVFP4 部署与评测规范。

## 核心贡献（创新点）
1. **首个真正全 W4A4-GDN 的混合 LLM 量化方案 Minima**：对 Qwen3.8-27B 全部 496 个线性层实施 NVFP4 W4A4，在固定服务栈下 5 任务平均仅比 BF16 低 0.52，且显存最小（17.5 GiB）、预填充最快。*与已有工作的本质区别在于打破“必须保护门控”的行业惯例，仅凭 PTQ 校准即达到 BF16 级别任务精度，无需量化感知训练（QAT）。*
2. **四层机制链条解释 GDN 抗量化的内在原因**：NVFP4 块缩放局部化残差流异常值；门控非线性将 ~11% GEMM 误差压缩为 ~2% 输出误差；delta 规则使状态误差在 32K 内达到平稳 plateau 并以远快于衰减门时域的速度主动 overwrite 旧噪声；端到端量化代价随上下文位置增加而衰减。*与以往“经验性精度保护”不同，本文从算子自身的门控与校正结构给出可复现的物理解释。*
3. **服务栈测量偏差修复与 KV cache 协同优化**：发现并修复 per-module 校准与 fused-GEMM 服务之间的全局尺度失配（该失配会伪装出更优的长上下文 PPL）；提出 FP8 KV cache 配合 per-layer 校准尺度的部署建议，以零任务代价消除量化模型的长上下文 PPL 惩罚（恢复 83%）。*弥补了现有 hybrid 模型 NVFP4 部署中的隐蔽工程漏洞。*

## 方法详解
- **Minima 量化流程**：基于 llm-compressor 对 Qwen3.8-27B 的 48 个 GDN 层、16 个注意力层、64 个 MLP 层共 496 个线性投影应用 NVFP4 W4A4；仅保留 lm_head、embeddings、convolutions、norms 为 BF16。校准数据集为冻结的 128-sample × 32K-token 文本集。
- **尺度 Harmonization**：vLLM 将 GDN 投影融合为 fused GEMM（如 in_proj_qkv+z、in_proj_b+a），但 llm-compressor 为各模块独立分配全局 FP32 尺度；直接部署会导致 $a/b$ 门控权重尺度错位（最大差 2.75×）。修复方法：重写每个融合组的共享全局尺度，并将比例折叠进 per-block E4M3 尺度，重舍入误差 ≤ 6.2%。
- **GDN 递推与误差动力学**：状态更新 $S_t = \alpha_t S_{t-1} + \beta_t k
