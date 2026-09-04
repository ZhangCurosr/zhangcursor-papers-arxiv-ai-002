---
title: "Why-Gated-DeltaNet-Survives-4-Bit-Quantization-NVFP4-W4A4-f"
source: https://arxiv.org/pdf/2609.04098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:33:56"
field: "大模型低比特量化与高效推理"
keywords: ["W4A4量化", "Gated DeltaNet", "混合LLM", "NVFP4", "长上下文推理", "post-training quantization", "linear attention"]
innovations: ["首次证明GDN gate投影可安全量化至4比特且匹配BF16精度", "发现并修复fused-GEMM与per-module校准间的global-scale失配", "提出calibrated FP8 KV scales免费恢复83%长上下文PPL惩罚"]
benchmarks: ["MMLU-Pro", "GSM8K", "AIME'25", "GPQA-Diamond", "LiveCodeBench v6", "RULER @32K/64K", "WikiText-2 PPL @4K/32K"]
---

# 论文速读：Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM

## 一句话总结
本文首次证明混合大语言模型中曾被认为对量化敏感的Gated DeltaNet（GDN）线性注意力层，可被完整地以NVFP4 W4A4（全4比特权重+激活）进行后训练量化，且在Qwen3.8-27B（27B参数、48层GDN + 16层softmax注意力）上匹配BF16精度，同时显存缩小至17.5 GiB（BF16的35%），预填充速度提升14–19%。

## 研究问题与动机
- **社区共识认为GDN是量化"禁区"**：现有所有公开Qwen3.8-27B的4比特构建（包括作者自身的FP8发布版）均将GDN块的gate投影（控制衰减$\alpha_t$和写入强度$\beta_t$的$a$、$b$）保持在8比特或16比特，理由是"递归状态跨越数万个token传递，逐步量化误差会累积"。
- **直觉方向可能与架构相反**：论文检验"误差会沿上下文累积"这一假设，发现GDN的gate投影反而是整个块中对量化最不敏感的组件，原因深植于其参数化设计与delta规则的擦除机制。
- **实测pipeline存在多处静默错误**：per-module校准与fused-GEMM serving之间的全局scale失配会伪造"更好的长上下文PPL"；多模态serving路径与thinking模型的raw-completion harness均会导致指标失效，这些此前未被识别。
- **长上下文KV cache的代价被低估**：W4A4模型在FP8 KV cache下的PPL惩罚是BF16模型的3倍，但通过简单的per-layer calibrated scales可恢复83%的损失而不影响吞吐。

## 核心贡献（创新点）
1. **首个真正的W4A4-GDN全量化模型Minima**：将Qwen3.8-27B全部496个线性层（含GDN gate投影）量化至NVFP4 W4A4，在6项准确率套件中匹配BF16（5-task平均−0.52，在seed噪声内），是目前最小（17.5 GiB）且prefill最快（TTFT@32K: 4.03s vs BF16的6.90s，+14–19%）的方案。
2. **四段式机制解释为何GDN能扛住4比特**：从激活统计→单投影敏感度→递归误差传播→上下文位置分解，逐层论证block scaling定位异常值、gate非线性压缩噪声、delta规则主动擦除状态误差、端到端误差随上下文衰减——将"它恰好能工作"转变为架构层面的因果解释。
3. **修复服务栈中的全局scale失配**：发现llm-compressor的per-module校准与vLLM的fused GEMM在GDN相邻模块间存在1.82×–2.75× scale差异，会导致gate被错误缩放；提出checkpoint-side的scale harmonization修复（折叠至E4M3 local scale），使kernel误差从0.35/0.57降至0.002。
4. **KV cache校准方案Minima+scales**：为16个attention层添加32个FP8 per-tensor KV scale，在保持吞吐不变（±0.4%）的情况下将32K PPL惩罚从+0.41降至+0.07，恢复83%的量化损失。

## 方法详解
**量化格式**：NVFP4采用E2M1 4-bit值 + 每16元素block的E4M3 scale + 每tensor一个FP32全局scale。W4A4意味着权重与激活均在同一粒度下量化，GEMM运行于原生4-bit tensor core。

**GDN层结构**：每层5个线性投影——`in_proj_qkv`（输出经深度因果卷积+SiLU后拆分为$q_t, k_t, v_t$）、`in_proj_z`（output gate）、`in_proj_a`（decay gate $a_t$）和`in_proj_b`（write gate $b_t$）。Gate以log-space参数化：
$$g_t = -\exp(A_{\log})\ \mathrm{softplus}(a_t + \mathrm{dt\_bias}),\quad \alpha_t = e^{g_t}\in(0,1),\quad \beta_t = \sigma(b_t)\in(0,1)$$
递归更新：$S_t = \alpha_t S_{t-1} + \beta_t k_t(v_t - S_{t-1}^\top k_t)^\top$，输出$o_t = S_t^\top(q_t/\sqrt{K})$。

**Four-part机制**：
- (i) **Block scaling定位异常值**：GDN输入与attention共享同一残差流，max/RMS达63.5、kurtosis ~1560、10.6%的block为"1-hot"，但NVFP4的16元block scaling将每个异常值的影响局限在自身block内，激活误差在所有层角色间均匀（7.5–9.2%）。
- (ii) **Gate是非最敏感的**：单投影敏感度测试（96次replay）显示量化$a$仅使输出$y$变化2.1%、量化$b$仅2.6%，而softplus/exponential与sigmoid压缩了~11%的GEMM误差至~2%输出误差；真正携带误差的是out（12.7%）、qkv（10.4%）、z（9.9%）三个普通GEMM。
- (iii) **Delta规则有界并主动擦除**：32K token lockstep FP32递归显示全Minima的relS稳定在12.6% plateau（token 256: 12.96% → token 32768: 12.31%），不累积；单个1%状态impulse在80–2900步内衰减至1/e，远快于decay-gate隐含的44K–62K horizon，因为每次write沿当前key方向覆盖状态。
- (iv) **端到端误差随上下文衰减**：按位置分解NLL，权重量化gap在窗口前半为+0.081 nats、后半仅+0.011 nats，最后2K token Minima反而优于BF16（−0.053）。

**Serving修复**：对48层中每对fused模块（qkv+z、b+a）重写至共享全局scale并折叠至94个local E4M3 scale，worst ratio 2.81×，重舍入误差≤6.2%。

## 实验与结果
**模型与设置**：Qwen3.8-27B（48 GDN + 16 attention，hidden size 5120，K=V=128）；vLLM 0.27.1，TP=1，RTX PRO 6000（96GB，SM120）；FP8 KV cache，GPU利用率0.85，32K生成上限；128-sample × 32K-token校准集。

**评估套件**：WikiText-2 PPL@4K/32K、MMLU-Pro、GSM8K、AIME'25（pass@1 over 4 seeds）、GPQA-Diamond、LiveCodeBench v6、RULER（NIAH single/multikey @32K/64K）。所有任务均通过per-sample validity gate。

**主要结果（Table 1）**：

| 指标 | BF16 | Minima | Unsloth | RadixArk |
|------|------|--------|---------|----------|
| PPL@4K/32K | 6.95 / 10.35 | 7.67 / 10.84 | 7.16 / 9.91 | 7.35 / 9.95 |
| MMLU-Pro | 80.4 | 79.7 | 78.9 | 79.1 |
| GSM8K | 95.5 | 95.5 | 95.4 | 95.7 |
| AIME'25 | 86.7 | **86.7** | 87.5 | 84.2 |
| GPQA-D | 86.5 | 85.1 | 85.0 | 85.4 |
| LiveCodeBench | 79.0 | 78.5 | 79.9 | 79.6 |
| 5-task avg | 85.62 | **85.10** | 85.34 | 84.80 |
| 权重VRAM | 50.13 GiB | **17.53 GiB** | 20.23 GiB | 18.83 GiB |
| Decode tok/s@32 | 621 | **1,154** | 1,132 | 1,174 |
| TTFT@32K | 6.90s | **4.03s** | 4.49s | 4.39s |
| RULER@32K/64K | 100 | **100** | 100 | 100 |

- Minima与BF16在所有任务上均在seed噪声内（5-task平均差−0.52 < 一个AIME题3.3分），是**唯一一个GDN块全部4比特**的方案，VRAM仅为BF16的35%。
- **最强结果**：AIME'25精确匹配BF16（26/30 on all 4 seeds），RULER 64K retrieval满分100，PPL@32K gap仅+0.49且随位置递减。
- **Decode略低于RadixArk**：Minima在decode阶段落后2–4%，归因于小batch NVFP4激活量化overhead（kernel artifact，非recipe本质问题）。
- **Minima+scales**：PPL@32K从10.84降至10.50（恢复83% penalty），吞吐不变（±0.4%）。

## 相关工作脉络
1. **Gated DeltaNet [16]**：结合Mamba2-style gating与delta rule的线性注意力算子；本文论证其量化鲁棒性基于算子自身门控+校正结构，而非特定训练选择。
2. **NVFP4 [11]**：NVIDIA硬件原生4-bit微格式（E2M1+block scaling）；本文是其在recurrent-state mixer上的首次大规模应用。
3. **QUASAR [1]**（同期）：同样对Qwen3.8-27B全部496个投影做4比特量化，但通过量化感知训练（QAT）+蒸馏实现；本文证明仅靠校准PTQ即达BF16级准确率，且给出机制解释。
4. **GPTQ [4] / AWQ [9] / SmoothQuant [14]**：经典W4/W8后训练量化方法，主要针对softmax attention与MLP；本文首次系统研究recurrent-state mixer的全W4A4量化。
5. **KVQuant [6] / KIVI [10]**：KV cache后训练量化；本文揭示W4A4模型在FP8 KV下的惩罚是BF16的3倍，并提出free的calibrated scales方案。
6. **RULER [7]**：长上下文基准（NIAH @32K/64K）；Minima在此基准上保持100分，验证64K retrieval无误。
7. **Microscaling formats [13]**：block-scaled微格式理论框架；本文利用其"异常值仅影响15个邻居"性质解释GDN输入的均匀量化误差。

## 局限性与未来方向
- **单一模型族与格式**：仅在Qwen3.8-27B + NVFP4上验证，其他混合架构（如不同GDN变体、SSM混合比例）的泛化性未知；线性参数化的decay gate（非log-space）可能不具备同等鲁棒性。
- **无128K+长上下文压力测试**：机制研究直接回答了累积问题（32K内state error flat、gap随位置衰减、64K retrieval满分），但更长的外推仍需验证。
- **QUASAR未纳入基准**：因其发布于测量周期之后且属于不同recipe类别（QAT vs PTQ），无法直接比较。
- **Decode overhead**：Minima decode落后RadixArk 2–4%，需kernel优化消除小batch激活量化overhead。
- **未覆盖embedding/lm_head**：未来工作（组内）已将二者也纳入NVFP4，并探索3-bit MLP codebooks，加载仅12.3 GiB。

## 研究启发与可借鉴点
1. **架构保护即精度保护**：gate投影的log-space softplus/exponential + sigmoid参数化天然压缩量化噪声，提示我们在设计新架构时应将"量化鲁棒性"纳入门控与非线性的设计约束，而非事后补救。
2. **Fused-GEMM与per-module校准的scale失配风险**：任何将多个calibrated模块fused为单一GEMM的serving kernel都可能引入此类问题；建议在编译期验证fused-group内各模块全局scale的一致性。
3. **按位置分解NLL作为量化诊断工具**：将per-token cost分解为"weight cost（随上下文衰减）"和"KV cost（随位置上升）"两类签名，可快速定位误差来源是权重还是cache，本文的这一诊断范式值得推广。
4. **Calibrated KV scales为free优化**：32个per-tensor FP8 scale在serving时无额外compute成本，却恢复83%的长上下文PPL惩罚，建议作为所有W4A4 hybrid模型的标配部署步骤。
5. **Thinking模型的harness陷阱**：raw-completion路径在开启thinking的模型上会产生严重偏差（±40–60分），后续评测必须使用chat-template + thinking disabled + per-sample validity gate。

## 关键术语表
- **Gated DeltaNet (GDN)**：结合delta rule与Mamba2-style gating的线性注意力算子，状态矩阵固定大小、逐token递归更新，用于替代部分softmax attention层以降低长上下文计算成本。
- **NVFP4**：NVIDIA硬件原生4-bit浮点格式，E2M1值 + 每16元素block的E4M3 scale，原生支持于SM120及以上tensor core。
- **W4A4**：权重与激活均采用4-bit量化的推理模式，需GEMM全程在4-bit tensor core上执行。
- **Delta rule recurrence**：$S_t = \alpha_t S_{t-1} + \beta_t k_t(v_t - S_{t-1}^\top k_t)^\top$，状态按当前key方向被overwrite而非accumulate，这是误差有界的根本原因。
- **Per-module calibration vs fused-GEMM scaling**：校准阶段按独立模块设全局scale，但serving时相邻模块被fused为单一GEMM并取最大scale，导致内部scale被错误应用。
- **Context inversion**：BF16基座模型在32K request内对同一token的NLL高于孤立4K window（6.95→10.35），是模型固有属性而非量化引入。
- **1-hot blocks**：max/RMS > 3的16元block，单个值占block能量>56%，是block scaling下主要误差来源。
- **Log-space gate parameterization**：$a_t$经softplus+exp得到$1-\alpha_t$，使量化噪声在进入recurrence控制信号前被非线性压缩。

## 可复现要素
- **模型**：Qwen3.8-27B（HuggingFace: Qwen/Qwen3.8-27B）；已开源。
- **量化检查点**：Minima与Minima+scales已发布至 https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4。
- **校准数据集**：128样本 × 32K-token的frozen set，具体来源论文未明述（likely WikiText-2或自建）。
- **关键超参**：NVFP4 block size=16；GPU: 单卡RTX PRO 6000 (96GB, SM120)；vLLM 0.27.1, TP=1；GPU利用率0.85；生成上限32K tokens。
- **代码**：使用llm-compressor进行NVFP4 W4A4校准与量化；服务栈为vLLM；机制研究的lockstep recurrence用纯PyTorch FP32实现。
- **复现注意**：必须执行fused-scale harmonization修复（94个scale set），否则gate将被错误缩放导致指标失真。
