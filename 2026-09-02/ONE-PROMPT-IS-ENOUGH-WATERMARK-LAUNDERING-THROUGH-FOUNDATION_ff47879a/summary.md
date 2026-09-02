---
title: "ONE-PROMPT-IS-ENOUGH-WATERMARK-LAUNDERING-THROUGH-FOUNDATION"
source: https://arxiv.org/pdf/2609.01249v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:11:52"
field: "数字水印鲁棒性与取证安全"
keywords: ["invisible watermarking", "watermark laundering", "foundation image models", "black-box attacks", "provenance security", "robustness evaluation"]
innovations: ["定义水印洗钱并建立 payload-fidelity 联合评估剖面", "揭示单提示黑盒基础模型重建可破坏隐写 payload 而不损害语义保真", "提示词消融证明攻击不依赖显式移除措辞，源于重建通道本身的信息瓶颈"]
benchmarks: ["MS-COCO 100 images", "DwtDct", "DwtDctSvd", "RivaGAN", "GPT Image 1/1.5/2", "Nano Banana/Pro/2"]
---

# 论文速读：ONE-PROMPT-IS-ENOUGH-WATERMARK-LAUNDERING-THROUGH-FOUNDATION

## 一句话总结
本文提出"水印洗钱"（watermark laundering）概念，发现通过单个自然语言提示词调用公共基础图像模型，即可在保持图像视觉/语义保真度的同时，不可靠化隐藏水印 payload；OpenAI 系列模型整体攻击能力最强，而 Google Nano Banana 2 在高保真重建下仍无法完全消除 DwtDct 水印。

## 研究问题与动机
- **现有评估盲区**：隐写水印的鲁棒性通常仅针对压缩、模糊、裁剪等固定扰动或专用去除模型进行评估，未考虑公共基础图像编辑模型通过提示词即可完成重建的攻击路径。
- **单提示黑盒威胁**：攻击者仅需提交一张带水印图像和一条自然语言提示（如"恢复/重建图像"），无需访问受害者解码器、嵌入密钥或模型梯度，即可获得视觉可用但 payload 被破坏的输出。
- **保真-破坏并存**：传统破坏性攻击往往同时损害图像可用性，而基础模型重建能在保持语义一致性的前提下破坏低频/高频载体信息，形成"高保真+强破坏"的新型洗钱 regime。
- **安全策略错配**：内容审核系统可能允许"图像修复"类指令，却拒绝显式"移除水印"请求；但消融实验表明，即使不含显式攻击措辞，普通重建提示同样能破坏 payload。

## 核心贡献（创新点）
- **定义水印洗钱形式化框架**：提出 payload–fidelity 联合评估剖面（BER + PSNR/SSIM/CLIP-Sim/NIQE 等多维指标），将"内容可用但 payload 失效"定义为洗钱成功，区别于传统仅关注 BER 或仅关注质量的单目标评估。
- **揭示基础模型重建作为缺失鲁棒性条件**：证明六款公共基础图像编辑模型（GPT Image 1/1.5/2、Nano Banana/Pro/2）均可作为无知识黑盒攻击接口，无需微调或优化循环，仅需单次 API 调用即可显著破坏三种代表性水印方案。
- **Prompt 无关性实证**：提示词消融表明，移除"隐藏信息移除"显式指令、颜色/亮度约束或几何保持条款，BER 几乎不变（ΔBER ≤ 0.0023），说明 payload 破坏源于重建通道本身的信息瓶颈，而非特定攻击措辞。
- **高保真洗钱的发现**：Nano Banana 2 在均值 PSNR=30.27、SSIM=0.879 的高保真条件下，仍使 DwtDct 水印 BER 达到 0.4110，证明 transform-domain 水印可在极高视觉质量下持续脆弱。

## 方法详解
- **威胁模型**：受害者水印方案 $W_V = (\text{Embed}_{W_V}, \text{Decode}_{W_V})$，在干净图像 $x$ 上嵌入 $k$-bit payload $b$ 得到 $x_V = \text{Embed}(x, b)$；攻击者调用黑盒基础模型编辑器 $y = F(x_V, p)$，其中 $p$ 为单条结构化提示词，无解码器反馈。
- **联合评估剖面**：$\mathcal{P}_D(W_V, F, p) = (\overline{\text{BER}}_D, \overline{M}_{\text{ref},D}, \overline{M}_{\text{clean},D}, \overline{M}_{\text{sem},D}, \overline{M}_{\text{nr},D})$，BER 接近 0.5 表示强破坏，保真度指标单独报告以捕捉 tradeoff。
- **重建瓶颈理论**：设 $Z = E_F(x_V)$ 为中间表示，由数据加工不等式 $I(b; y|p) \leq I(b; Z)$，若 $Z$ 对 payload $b$ 的信息量小（语义充分但取证不足），则最优恢复 BER 下界趋于随机解码。
- **高频率残差（HFR）分析**：$\text{HFR}(x_V, y) = \|P_H(y) - P_H(x_V)\|_2 / (\|P_H(x_V)\|_2 + \eta)$，度量 DCT 高频带变化幅度；全样本 Pearson $r=0.3158$、Spearman $\rho=0.3607$，HFR 是破坏因素之一但非唯一解释。
- **评估协议**：100 张 MS-COCO 图像（40 纹理/30 平坦/30 含文本），3×6=18 组配置，每组 N=100，共 1,800 次 API 调用；使用结构化提示 $P_{\text{struct}}$（含画布/几何/亮度/颜色/外观/隐藏信号六模块约束）。

## 实验与结果
- **数据集与模型**：100 张 MS-COCO 图像；6 个基础图像编辑模型（GPT Image 1/1.5/2、Nano Banana/Pro/2）；3 种水印方案（DwtDct、DwtDctSvd、RivaGAN）。
- **最强攻击结果**：GPT Image 1 对 DwtDct 的 BER=0.4944（最接近随机）；GPT Image 2 对 DwtDctSvd 的 BER=0.4517、对 RivaGAN 的 BER=0.3620，同时在 RivaGAN 上获得最佳 NIQE/BRISQUE/CLIP-IQA 质量分。
- **高保真 regime**：Nano Banana 2 对 DwtDct 达到 PSNR=30.27、SSIM=0.879，BER 仍为 0.4110，验证高保真下 payload 持续脆弱。
- **版本演进趋势**：OpenAI 模型从 Image 1→2，BER 从 0.4808→0.4134（远离随机），PSNR 提升；Google 模型从 Nano Banana→2，BER 从 0.3282→0.2667（改善恢复），但 DwtDct 例外（稳定在 ~0.41）。
- **对比传统攻击**：高斯噪声对 DwtDct 造成 BER=0.6225（超随机阈值），但 PSNR-clean 仅 20.45；BM3D 去噪 BER=0.4097 且保真度高（PSNR-clean=36.79），但基础模型在 DwtDctSvd/RivaGAN 上 BER 更接近 0.5。
- **提示词消融**：移除任意模块（可见内容保持/联合亮色约束/外观几何/显式隐藏信息语言）在 Nano Banana 2 上 ΔBER ≤ 0.0023；最小提示词仍达 BER=0.4167（Nano Banana 2）和 0.4215（GPT Image 2）。
- **HFR 关联**：全样本 Pearson $r=0.3158$，DwtDct 子集 $r=0.3361$，RivaGAN 子集 $r=0.4662$；GPT 模型内相关性近零，Nano Banana Pro 内 $r=0.3585$。

## 相关工作脉络
- **隐写鲁棒性基准**：HiDDeN、StegaStamp、RivaGAN、MBRS 等方案通常以压缩/滤波/几何变换为鲁棒性测试基准，本文指出其未覆盖基础模型提示重建这一新型攻击接口。
- **生成式去除攻击**：Zhao et al. (NeurIPS 2024) 和 Liu et al. (ICLR 2025) 使用本地可控扩散模型进行再生攻击，需访问模型权重或梯度；本文仅用公开黑盒 API，无需内部访问。
- **指令引导编辑**：SDEdit、Prompt-to-Prompt、InstructPix2Pix 等工作展示重建可保留场景内容但重写像素级信息，本文将其转化为取证 payload 破坏的新机制。
- **信息论水印分析**：Moulin & O'Sullivan (2003) 奠定容量-失真理论，本文将其扩展至"语义充分但取证不足"的瓶颈抽象，形式化说明公开内容保留与 payload 丢失可解耦。
- **C2PA 溯源标准**： Coalition for Content Provenance and Authenticity 规范定义 Content Credentials 元数据；本文审计发现所有输出均含 provider 标记，但受害者 payload 仍可被破坏，揭示元数据合规≠取证安全。
- **生成水印（不在评估范围）**：Stable Signature、Tree-Ring、Gaussian Shading 等工作将水印嵌入模型内部或潜在空间，本文明确声明未覆盖此类方案。

## 局限性与未来方向
- **模型与方案覆盖有限**：仅评估 6 个基础图像模型和 3 种水印方案，未涵盖 Stable Signature、Tree-Ring、Gaussian Shading 等生成水印，以及 Midjourney、Stable Diffusion 等开源平台。
- **单次提示单一 protocol**：未探索多轮对话、对抗性提示搜索、detector feedback 循环或跨模型组合攻击，实际攻击能力可能被低估。
- **缺乏统计显著性检验**：未报告置信区间或 effect size 检验，描述性差异（如消融实验中 SSIM/LPIPS 微小变化）无法判定统计显著性。
- **内部机制黑盒**：依赖行为观察而非模型内部因果验证，瓶颈假设（$I(b;Z|C)$ 小）未被直接测量。
- **政策合规边界**：未测试是否违反各 provider 的服务条款，仅声明研究目的为安全评估而非 policy bypass。

## 研究启发与可借鉴点
- **联合评估剖面设计**：BER–fidelity 双维度报告取代单一聚合分数，可有效捕捉"高保真+强破坏"这种传统指标下被掩盖的 regime，可迁移至任何对抗鲁棒性研究。
- **提示词无关性测试范式**：通过模块化提示消融验证攻击是否依赖特定措辞，区分" lexical filter 防御"与"结构性缺陷"，适用于所有基于 prompt 的安全评估。
- **HFR 作为诊断工具**：高频残差变化与 decoder error 的正相关（$r≈0.3-0.47$）可作为快速诊断工具，辅助判断水印方案对重建攻击的脆弱性来源。
- **版本演化追踪方法**：同系列模型不同版本的 BER–fidelity 轨迹（OpenAI 破坏增强 vs Google 保真提升）揭示厂商安全权衡差异，可推广至其他 AI 安全基准的纵向研究。
- **C2PA 兼容性与取证分离评估**：本文审计 provider 标记但独立评估受害者 payload，揭示"元数据合规≠payload 保留"，为 C2PA 类溯源系统的安全评估提供方法论。

## 关键术语表
**Watermark Laundering（水印洗钱）**：通过基础模型重建，在保持图像视觉/语义可用性的同时使隐藏水印 payload 无法可靠解码的失败模式。

**Payload–Fidelity Profile（Payload–保真度剖面）**：联合报告 BER（payload 破坏程度）与多类保真度指标（PSNR/SSIM/CLIP-Sim/NIQE）的评估框架，避免单目标聚合掩盖 tradeoff。

**Reconstruction Bottleneck（重建瓶颈）**：基础模型的中间表示 $Z$ 对可见内容 $C$ 信息充足但对 payload $b$ 信息不足，由数据加工不等式导致下游恢复 BER 趋于随机。

**High-Frequency Residual（HFR，高频率残差）**：输入与输出在 DCT 高频带的像素差异Norm比值，用于量化重建对高频载体的扰动幅度。

**Black-box Prompt Attack（黑盒提示攻击）**：攻击者仅通过自然语言提示调用公开 API，无解码器反馈、无模型权重、无梯度访问的单次重建攻击范式。

**C2PA / Content Credentials**：Coalition for Content Provenance and Authenticity 制定的溯源元数据标准，本文发现 provider 标记在清洗后仍可存在，但不等同于受害者 payload 的保留。

## 可复现要素
- **数据集**：MS-COCO（公开），100 张分层采样图像（40 纹理/30 平坦/30 含文本）。
- **代码/权重**：论文未提及开源代码或模型权重；基础模型为闭源 API（GPT Image 系列、Nano Banana 系列）。
- **关键超参**：图像分辨率统一 1024×1024；payload 长度未明确；每组 N=100 次调用；单一结构化提示 $P_{\text{struct}}$（含六模块约束）。
- **评估指标**：BER（raw）、PSNR、SSIM、LPIPS、CLIP-Sim、NIQE、BRISQUE、CLIP-IQA、HFR。
