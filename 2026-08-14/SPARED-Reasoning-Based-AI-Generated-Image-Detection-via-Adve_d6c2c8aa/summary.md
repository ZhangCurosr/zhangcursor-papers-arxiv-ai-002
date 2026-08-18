---
title: "SPARED-Reasoning-Based-AI-Generated-Image-Detection-via-Adve"
source: https://arxiv.org/pdf/2608.12876v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:25:35"
---

# 论文速读：SPARED-Reasoning-Based-AI-Generated-Image-Detection-via-Adve

## 一句话总结
论文提出 SPARED，一种基于异构对抗强化学习的 AI 生成图像检测框架，通过扩散图像编辑器与推理 MLLM 的交替自进化训练，在仅以最终判罚正确性为奖励的设计下，使检测精度与可解释推理同步单调提升，并从数据构建层根除 provenance 数据偏差捷径。

## 研究问题与动机
- **核心问题**：部署级检测器需同时输出真伪判罚与自然语言解释，且必须持续泛化以适应不断演进的生成器族，而非过拟合静态数据集。
- **现有方法不足 1（数据捷径）**：传统指纹/频域检测器依赖特定生成架构的低层伪迹，面对新一代生成器性能骤降；基于预训练模型微调的检测器常因真实/伪造图来源不同而学到分辨率、语义分布等数据集风格捷径。
- **现有方法不足 2（解释模板化）**：在固定解释语料上监督微调的 MLLM 易产出因果浅层的模板化说辞；静态训练语料使决策边界成为 stationary target，新生成器可轻易调参绕过。
- **现有方法不足 3（对抗退化）**：自演化或单一模型自我改进缺乏外部强对抗源；若奖励仅以“欺骗检测器”为目标，攻击者会退化为跳过编辑或直接生成流形外噪声。

## 核心贡献（创新点）
- **解耦的生成式对抗强化学习循环**：引入扩散图像编辑器作为攻击者而非固定伪造操作策略，通过在训练数据层面构造源图与其编辑伪图的严格同源配对，从根源上关闭 provenance 捷径。与既有方法通过事后对齐或 debiasing 算法修正数据分布不同，本工作将去偏机制内嵌于数据生成管线本身。
- **PaCo 门控异构对抗架构**：将视觉连续像素空间的扩散编辑器与离散 token 自回归 MLLM 解耦训练，并以 PaCo 指令忠实度评分作为奖励门控（≥0.7），防止攻击者以“无编辑纯欺骗”换取奖励。与 LLM 安全领域共享参数或梯度耦合的自博弈框架不同，本架构彻底切断双方参数交互，仅通过交换数据与二值奖励信号迭代。
- **判罚唯一奖励驱动的可解释性涌现**：防御者仅以最终 verdict 正确性为 GRPO 奖励信号，解释生成交为准确率提升的副产品，实现精度与解释质量的同步单调上升。与直接对推理文本施加格式/内容奖励的 SFT 或 RLHF 范式不同，本设计迫使模型将解释视为定位判罚依据的工具而非独立优化目标。

## 方法详解
- **防御者（Reasoning Defender）**：基于 Qwen3.5-9B 骨干，先经 LoRA SFT 学习 `<reasoning>` 与 `<answer>` 格式及基础伪迹词汇，合并 adapter 后开展全参数 GRPO。对输入图像 $x$ 采样 $G$ 个响应，奖励函数为 `r_D(x, y) = H[ŷ = y]`，仅当 `<answer>` 标签与真实标签一致时给 1，否则 0；GRPO 在组内归一化优势，格式、长度、推理内容均不进入奖励计算。
- **攻击者（PaCo-Gated Diffusion Attacker）**：基于 Qwen-Image-Edit-2511 的 LoRA 适配器，使用 DiffusionNFT 在线优化。给定源图 $x_{\mathrm{src}}$ 与编辑指令 $c$，生成 $x_{\mathrm{edit}} = \pi_A(x_{\mathrm{src}}, c)$。奖励由两项冻结裁判共同决定：指令遵循分 $r_{\mathrm{PaCo}}(x_{\mathrm{src}}, c, x_{\mathrm{edit}}) \in [0,1]$ 与逆向检测分 $r_{\mathrm{det}} = \mathbb{1}[\neg(\pi_D^{(t)}(x_{\mathrm{src}})=\mathrm{real} \wedge \pi_D^{(t)}(x_{\mathrm{edit}})=\mathrm{fake})]$。最终奖励门控为 `r_A = r_det` 当且仅当 `r_PaCo ≥ 0.7`，否则为 0。
- **数据构建与训练调度**：源图集合从 ImgEdit、pico-banana-400k、MagicBrush 混合语料中固定采样一次，经感知哈希与所有评测基准交叉去重，杜绝训练源泄露至测试集。调度共 5 轮：SFT 初始化后，Iter1 防御
