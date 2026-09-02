---
title: "Jailbreaking-Text-to-Image-Models-Through-Cracks-Navigating"
source: https://arxiv.org/pdf/2609.01168v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:14:36"
field: "AI安全与对抗攻击"
keywords: ["jailbreak", "text-to-image models", "heterogeneous safety filters", "multi-agent debate", "detection surface", "adversarial prompt generation"]
innovations: ["提出Detection Surface几何框架形式化异构安全栈的联合决策边界", "设计三智能体辩论框架实现探索-诊断-仲裁解耦的自适应越狱搜索"]
benchmarks: ["NSFW-200", "I2P", "UnsafeDiff", "SD 1.4", "SDXL", "DALL·E 3", "Midjourney"]
---

# 论文速读：Jailbreaking-Text-to-Image-Models-Through-Cracks-Navigating

## 一句话总结
本文提出CRACK框架，通过引入"检测曲面"(Detection Surface)的几何分析形式化异构安全栈的联合决策边界，设计了三智能体辩论机制（攻击、防御、裁判）结合强化学习，实现了对异构多层T2I安全防护的自适应越狱搜索，在SD 1.4组合防御下ASR高达99.63%。

## 研究问题与动机
- 现有T2I越狱方法无法有效应对现代安全栈的异构性：component-specific attacks仅针对单一过滤器优化，黑盒query方法仅获得聚合反馈（成功/失败），无法定位具体失效层级。
- 跨层冲突问题：绕过一层过滤器的突变可能增加对另一层的暴露，现有方法缺乏层感知的搜索方向指导。
- 检测区域的稀疏性与非凸性导致盲目搜索效率低下，需要自适应的层间导航机制。
- 现有多智能体辩论方法主要针对文本LLM设计，未考虑T2I系统中prompt-image对的模态复杂性与随机性。

## 核心贡献（创新点）
- 提出Detection Surface统一几何框架：首次形式化刻画异构T2I安全栈的联合决策边界，揭示跨层冲突、稀疏性与非凸性三大结构性质。
- 设计CRACK多智能体辩论框架：通过Attack/Defense/Judge三智能体协作实现探索、诊断、仲裁的解耦，这是首个针对T2I异构安全栈的多智能体越狱框架。
- 引入层感知强化学习策略优化：在策略层面而非词元层面进行mutation选择，结合bypass奖励与judge连续评分的复合reward，适应稀疏非凸的逃避区域。
- 建立全面的实验评估体系：在SD 1.4、SDXL、DALL·E 3、Midjourney及多种异构安全配置下验证，证明检测曲面性质的实证有效性。

## 方法详解
- **Detection Surface形式化**：将离散prompt空间映射到连续表示空间X，定义每层过滤器Sk的safe region Bk = {x | fk(x) < 0}，检测曲面D = ∪∂Bk，逃避区域E* = (∩Bk) ∩ F(xh, δ)。
- **三智能体协作机制**：
  - Attack Agent：根据 Defense Agent 的层诊断反馈Rt，从策略库M中选择突变策略at ~ πφ(·|pt, Rt)，包含隐喻替换、思维链扩展等5种互补策略。
  - Defense Agent：构建三层代理检测管道——Tier 1关键词匹配(文本边界)、Tier 2 LLM语义风险评估(语义边界)、Tier 3 CLIPScore跨模态验证(跨模态边界)，输出结构化风险报告Rt = (r1t, r2t, r3t)。
  - Judge Agent：计算复合得分s = α·Srisk + β·σ(ΔCLIP)，其中ΔCLIP = CLIP(pt, cunsafe) - CLIP(pt+1, cunsafe)衡量跨模态边界进展。
- **强化学习策略优化**：采用Actor-Critic框架，状态st包含prompt与层检测状态，动作at为策略选择，reward由bypass奖励(+b/-b)与judge奖励s组成，最小化L(φ) = -E[Σlog πφ(at|st)·Ât] + E[Ât²]。
- **辩论协议**：每轮迭代执行三步——Attack生成突变prompt → Defense评估生成Rt+1 → Judge仲裁计算st+1，直至进入E*或达到最大轮数Tdebate。

## 实验与结果
- 数据集：NSFW-200（成人内容）、I2P（真实用户提示）、UnsafeDiff（暴力/ gore/歧视三类）。
- 目标模型：Stable Diffusion 1.4、SDXL、DALL·E 3、Midjourney。
- 安全配置：text-m、text-c、image-c、image-clip-c、text-image-c及DALL·E 3内置防护。
- 核心结果：SD 1.4 + text-image-c配置下ASR-Q16达83.01%(I2P)/78.69%(UnsafeDiff)，ASR-NudeNet达66.35%(NSFW-200)；SD 1.4 + text-m配置下ASR-Q16达99.30%/98.36%；DALL·E 3配置下ASR-Q16达43.57%/49.02%，约为最强baseline的2倍。
- 效率优势：平均仅需6次查询，远低于SneakyPrompt(39.96)和JailFuzzer(29.55)；单prompt生成耗时123.01秒，为最快迭代方法。
- 语义保真：CLIPScore最高达0.39(UnsafeDiff)，PPL最低29.78，显著优于baseline。

## 相关工作脉络
- SneakyPrompt [8]：基于RL的词元级替换攻击，仅获二元反馈，无法定位跨层冲突。
- JailFuzzer [17]：模糊测试与LLM Agent结合，反馈粒度为二进制成功信号，缺乏层感知。
- DACA [20]：LLM分解不安全请求，但未建模跨层交互与自适应搜索。
- PGJ [16]：基于感知信号的越狱方法，仅针对单一层优化。
- Ring-A-Bell [9]：白盒攻击，利用概念擦除模型的漏洞，不适用于黑盒商业API。
- MMA-Diffusion [19]：多模态对抗攻击，需要梯度访问，与本文黑盒设定不同。

## 局限性与未来方向
- 防御代理仅覆盖三种检测维度（词汇/语义/跨模态），未穷尽所有可能的安全机制实现。
- 对未知或自适应商业防御的泛化能力有限，需扩展代理模型。
- 实验主要集中在特定T2I模型，对更多新兴模型（如Flux、SD3）的验证不足。
- 未来方向：扩展检测曲面对未知/自适应防御的泛化，将形式化方法应用于其他组合安全流水线。

## 研究启发与可借鉴点
- **检测曲面分析框架**：可用于其他多组件安全系统的几何分析，揭示层间冲突的结构性原因。
- **三智能体解耦设计**：探索-诊断-仲裁的分离思路可迁移至红队测试、自动对抗生成等任务。
- **策略级RL优化**：在离散动作空间中学习策略选择而非词元替换，适用于资源受限的黑盒攻击场景。
- **复合reward设计**：结合二分bypass信号与连续进展评分，可在稀疏奖励环境下提供更密集的学习信号。

## 关键术语表
**Detection Surface**：异构安全栈各过滤器决策边界的并集，形式化刻画多约束下的prompt空间几何结构。
**跨层冲突**：绕过一层过滤器的突变可能增加对另一层的暴露，体现为梯度方向不一致。
**逃避区域E\***：同时满足所有过滤器安全条件且保持语义忠实度的prompt表示子集。
**Defense Agent**：提供分层诊断反馈的代理，近似真实安全栈的三层检测管道。
**Judge Agent**：仲裁跨层冲突的代理，输出复合reward信号指导策略优化。
**Query Budget**：攻击者可使用的模型查询次数上限，限制搜索空间规模。
**Semantic Fidelity**：生成图像与原始有害意图的语义相似性，由CLIPScore衡量。

## 可复现要素
- 数据集：NSFW-200、I2P、UnsafeDiff（公开可用）
- 代码：论文声明代码在补充材料中提供，未提及正式开源仓库
- 模型权重：使用DeepSeek V3作为LLM backbone
- 关键超参：α=0.3, β=0.7, σ(·)缩放因子10, bypass奖励b=10, 学习率3×10⁻⁴, 最大辩论轮数3轮
