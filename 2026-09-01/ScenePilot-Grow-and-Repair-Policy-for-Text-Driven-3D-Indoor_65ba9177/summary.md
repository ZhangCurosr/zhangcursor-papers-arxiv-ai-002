---
title: "ScenePilot-Grow-and-Repair-Policy-for-Text-Driven-3D-Indoor"
source: https://arxiv.org/pdf/2608.30307v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:31:10"
---

# 论文速读：ScenePilot: Grow-and-Repair Policy for Text-Driven 3D Indoor Scene Generation

## 一句话总结
本文提出 ScenePilot，一种检索增强的“生长-修复”策略框架，将文本驱动的3D室内场景生成从单步预测重构为功能分组序列插入+循环修正的过程感知范式；通过分级检索布局先验引导分组规划，并在每组插入后执行轻量级局部修复、完成后执行全局协调，显著提升了物理合法性、功能连贯性与可控性。

## 研究问题与动机
- **单步/后处理范式的固有缺陷**：现有方法多直接预测最终布局或依赖昂贵的后验优化，忽略场景构建的过程依赖性，早期锚点对象错位会导致后续摆放产生连锁结构性错误。
- **短提示词先验缺失**：用户提示通常过于简略，缺乏对象关系、功能分区与房间级布局规律，从零推理极易不稳定。
- **过程监督数据空白**：标准3D场景数据集仅提供干净最终布局，缺少“破损中间状态→可执行修复轨迹”的细粒度配对数据，限制了修复策略的显式训练。
- **后验修正的误差固化与高开销**：基于VLM的后期精修只能作用于已传播的错误，计算成本高且难以恢复被早期错误绑定的全局结构。

## 核心贡献（创新点）
1. **过程感知的生长-修复生成框架**：将中间场景状态显式化为规划、观察与修正的一等公民目标，打破“生成→后处理”的两阶段割裂。
2. **分层检索增强规划（HRAP）**：构建锚点中心化的离线空间先验库，在房间级、功能组级、锚点级三级检索布局规律，以软提示形式指导分组计划与主从关系构建。
3. **强化多模态修复（RMR）策略**：设计受限的 `move–rotate–scale` 动作空间，融合渲染视图、结构化JSON与检索上下文进行局部/全局修正，并引入确定性质量评分 $Q$ 进行接受门控。
4. **SceneReverse-17k 过程监督数据集**：通过对高质量 3D-FRONT 场景施加位置/旋转/尺度扰动构造可逆退化轨迹，利用时间反转序列作为可执行修复目标，填补过程级修复监督的数据空白。

## 方法详解
- **整体流程**：输入提示 $x$ 与房间边界 $B$，HRAP 检索先验 $\mathcal{P}(x)$ 并输出有序功能分组计划 $\mathcal{G}=\{g_m\}_{m=1}^M$；基座文本驱动生成器 $G$ 依次插入分组得到 $S^{(m)}$；RMR 在每组插入后执行最多 $K_{\mathrm{local}}$ 轮局部修复；全部完成后执行全局修复 $\Pi_{\mathrm{global}}$ 产出 $S^{\mathrm{final}}$。
- **HRAP 先验构建**：从结构化 JSON 场景挖掘以床、沙发、餐桌、书桌等为主的锚点对象及其附属成员，构建锚点中心功能组 $(a, \mathcal{N}_a)$；在锚点局部坐标系下统计相对偏移、平面距离、朝向、支撑关系，并提取组级组成签名 $\boldsymbol{\sigma}(r,a)=\{(c_k,n_k)\}$；聚合为 overview、signature、member-prior 三类自然语言文档，经 Qwen3-Embedding-8B 编码为 4096 维向量后用 FAISS 索引。
- **RMR 观测与动作**：修复步 $t$ 观测 $o_t = (I_t^{\mathrm{top}}, I_t^{\mathrm{diag}}, I_t^{\mathrm{ann}}, J_t, H_t)$；策略 $\pi_\theta$（基于 Qwen3-VL-8B-Instruct）输出 JSON 动作列表，涵盖 `move$(i,\Delta x,\Delta y,\Delta z)$`、`rotate$(i,\Delta\theta)$`、`scale$(i,\Delta s_x,\Delta s_y,\Delta s_z)$`，非法索引或物理不可行动作由执行器丢弃。
- **质量评分与接受准则**：$Q(S) = -\lambda_{\mathrm{pbl}}\mathrm{PBL}(S) - \lambda_{\mathrm{rel}}\mathrm{REL}(S) - \lambda_{\mathrm{func}}\mathrm{FUNC}(S)$，PBL 度量越界与网格碰撞，REL 度量高层关系违规，FUNC 度量可达性与通行；局部搜索候选集 $\mathcal{C}^{(m,k)}$ 仅当 $Q(\bar{S}) > Q(\tilde{S}) + \epsilon_Q$ 时接受，防止有害漂移。
- **训练流程**：Stage I 在 SceneReverse-17k 上以交叉熵进行监督模仿（SFT），学习 JSON 格式、对象索引与粗粒度空间修正；Stage II 采用 GRPO 优化，采样 $N_{\mathrm{cand}}$ 个候选动作并计算模块化奖励 $R=\lambda_1 R_{\mathrm{format}}+\lambda_2 R_{\mathrm{apply}}+\lambda_3 R_{\mathrm{phys}}+\lambda_4 R_{\mathrm{vlm}}$（实验取值 $0.15,0.15,0.5,0.2$），通过组内相对优势
