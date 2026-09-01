---
title: "Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I"
source: https://arxiv.org/pdf/2608.26480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:14:15"
---

# 论文速读：Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I

## 一句话总结
本文在零训练、零基准特定调优的严格对照下，评估了基于共享文件系统账本的 Manager-Worker 脚手架对 LLM 代码生成能力的真实增益；结果表明该脚手架能显著提升多数模型（如 Qwen3.8-27B +23.4、Kimi-K3 +42）的准确率，虽将 Token 开销增至约三倍，但相比直接升级更大模型仍具有更高的成本效益，且其收益高度依赖模型规模与推理模式的启用状态。

## 研究问题与动机
- 多智能体 LLM 系统虽常被报道优于单模型基线，但现有比较严重混杂：Pipeline 往往同时改变 Token 预算、工具调用、Prompt 设计与检索模块，导致 aggregate gain 无法归因于具体机制。
- 需要在一个严格控制变量的实验设置下，固定底层模型与问题集，仅引入 Manager-Worker 协作结构，量化脚手架本身的边际收益。
- 验证无需训练的固定 Prompt 编排策略（zero-shot self-orchestration）能否在竞争性编程等复杂任务上匹敌或超越昂贵的训练型编排器（如 Fugu、Conductor）。
- 建立“准确性提升 vs Token 成本”的量化权衡视角，为工业界在 API 费用与模型性能之间做架构选型提供实证依据。

## 核心贡献（创新点）
1. **提出零样本账本式 Manager-Worker 编排框架**：通过共享文件系统（`task.md`, `plan.md`, `tasks.json`, `notes.md`, `solution.py`）实现跨调用的状态外化与动态任务调度，无需任何微调或任务特定演示。
   *本质区别*：与 Fugu/Conductor 等学习编排策略的模型不同，本文的编排器是完全固定的 Prompt，剥离了“学习能力”对增益的贡献，纯粹衡量结构化协作的零样本价值。
2. **构建了严格控制变量的对比实验协议**：在 LiveCodeBench-100 上对 9 款模型进行配对测试，通过固定后端（pinned backend）与五次独立重复跑消除 OpenRouter 路由噪声，配合配对符号翻转置换检验给出严格的显著性结论。
   *本质区别*：不同于多数论文混合多模型、多预算、多工具的综合评测，本文严格保持模型、问题集、温度一致，唯一变量是脚手架结构，使因果推断更为可靠。
3. **量化了脚手架的成本-收益 frontier 并给出工程化 Trade-off 结论**：证明管理器约使 Token 账单翻三倍，但用更低成本即可逼近甚至达到顶级闭源模型单调用的性能（如 GPT-5.6-Terra 配脚手架仅 $11.71 即接近 Fable 5 的 87.4%）。
