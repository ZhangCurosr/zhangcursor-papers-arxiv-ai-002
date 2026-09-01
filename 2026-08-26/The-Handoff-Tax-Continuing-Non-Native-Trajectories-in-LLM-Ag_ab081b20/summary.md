---
title: "The-Handoff-Tax-Continuing-Non-Native-Trajectories-in-LLM-Ag"
source: https://arxiv.org/pdf/2608.24358v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:11:29"
---

# 论文速读：The-Handoff-Tax-Continuing-Non-Native-Trajectories-in-LLM-Ag

## 一句话总结
本文系统量化了长周期 Coding Agent 在中途切换低/高能力模型（LC↔HC）时面临的成本-质量权衡，提出“交接税”（Handoff Tax）概念：原始轨迹直传的升级（Escalation）仅能回收不足一半的质量优势却伴随数倍成本溢价，而降级（Downshift）可保留显著的性价比；同时揭示交接接口需依方向调整，削减发送方轨迹有助于升级，但丢弃前驱轨迹会严重损害降级质量。

## 研究问题与动机
- 现代 Coding Agent 单次任务常涉及数十至数百次模型调用，用户需在未知任务难度前预先选择模型，并在运行中面临“卡顿时升级”或“硬推理完成后降级省钱”的经济-技术权衡。
- 中途切换要求接收模型延续一条非自身生成的轨迹（包含发送方的假设、工具调用习惯、死胡同甚至错误），其对端到端质量与货币成本的具体影响缺乏系统实证。
- 现有模型路由/级联研究多聚焦“选哪个模型行动”，却将历史上下文视为自然继承的透明管道，未将“切换时机、方向与交接信息界面”作为联合控制变量进行隔离分析。
- 尽管主流 Coding Agent 产品（如 Kiro、Codex、Claude Code）已暴露 `/model` 切换命令，但底层交接代价的黑盒状态阻碍了工程实践的最优决策。

## 核心贡献（创新点）
- **首次系统量化中途模型能力交接代价**：覆盖 2 种模型族 × 2 个方向 × 7 个切换点 × 4 种接口，完成 58,000 次 Agent 运行、200 万次 API 调用与 360 亿 token 处理，填补长周期 Agent 切换成本的实证空白。
- **提出并实证“交接税”（Handoff Tax）现象**：证明 Raw 升级无法有效利用 HC 能力，对 Claude 对甚至出现“放弃 LC 轨迹直接重开 HC 更便宜更准”的反直觉结论（Raw 成本 $1.61 vs Abort+fresh $0.90）。
- **揭示交接接口的方向性对偶规律**：升级时应做减法（`Compact_pre` 或 `Traj-drop` 削减 LC 轨迹可提升质量与成本效率），降级时应做加法（保留 HC 轨迹对维持 LC 接收方质量至关重要，丢弃导致质量断崖）。
- **建立任务信息动态（Information Dynamics）与交接价值的关联**：通过 LiC 与 BrowseComp 扩展实验，证明需求是否晚期揭示、证据是否渐进积累会根本性改变最优切换方向，将交接设计从“纯模型能力问题”提升至“任务状态演化问题”。

## 方法详解
- **实验脚手架**：基于 `mini-swe-agent` 框架，统一 Bash 工具、提示词与 Docker 环境；LC/HC 对分别为 Claude Haiku 4.5 / Opus 4.7 与 GPT-5.6 Luna / Sol。
- **四种交接界面**：所有策略均保留前驱写入磁盘的工作树（$\mathcal{W}_K$），仅差异于传递给后缀模型的轨迹信息（$\mathcal{T}_{1:K}$）：
  - `Raw`：完整会话历史（系统提示、任务、推理、工具调用与观测）原文直传。
  - `Compact_pre`：前驱模型撰写纯文本续接摘要，仅摘要+W_K 传给后缀。
  - `Compact_suf`：后缀模型自行阅读前驱轨迹并生成摘要，再从摘要继续。
  - `Traj-drop`：不传递任何轨迹，仅注入固定续接消息（告知文件变更仍在、请继续解题）。
- **切换点校准**：为避免易题早结束造成的偏差，按任务难度桶（Easy/Medium/Hard）内前驱模型单模型终止步数分布的分位数 {5,10,15,25,35,45,50} 设定切换阈值 K。
- **公平评估子集**：仅在四种策略均触发切换的任务交集上进行对比，避免“部分任务提前完成导致基准污染”。
- **归一化指标**：
  - 质量回收率 `QRec(m,K) = 100·(R_m(K)−R_LC(K)) / (R_HC(K)−R_LC(K))`
  - 成本节省保留率 `CSRet(m,K) = 100·(C_HC(K)−C_m(K)) / (C_HC(K)−C_LC(K))`
  - QRec/CSRet 以单模型 LC/HC 为锚点，越接近理想角（QRec=100, CSRet=100）越优，负值表示劣于 HC-only。
- **成本分解机制**：将交接额外成本拆解为“接收方单步成本”与“接收方步数”。Raw 升级的税主要源于 HC 单步调用成本膨胀（Claude 2.2×, GPT 1.6×）；Traj-drop 降级的税主要源于 LC 被迫重复探索步数增加（Claude 1.6×, GPT 2.0×）。

## 实验与结果
- **主数据集与规模**：SWE-bench Verified（500 真实 GitHub issue）；辅实验 LiC（535 渐进需求对话）与 BrowseComp（200 需网页搜索的问题）。
- **基线对照**：LC-only、HC-only、Abort+HC fresh（支付 LC 前缀后丢弃轨迹重开 HC）、LC-full+HC-full（完整跑 LC 后重开 HC）。
- **Escalation (LC→HC) 核心结果**：
  - Raw 仅回收 QRec=47%（Claude）/ 36%（GPT），成本飙升至 LC 的 ~4-6 倍。Claude 对 Raw 成本 $1.61 远超 HC-only $0.72，Abort+fresh ($0.90) 严格占优。
  - `Compact_pre` 在 Claude 上实现 QRec=60%、CSRet=−11%（接近 HC-only 成本）；`Traj-drop` 提升至 QRec=64%（Claude）/ 84%（GPT），但成本仍高于 HC-only。
  - 难度分层显示：Hard 任务上 Reduced-context 接口可优于 HC-only（保留 12-42% 成本优势，回收 65-74% 质量），但 Raw 始终不具经济性。
- **Downshift (HC→LC) 核心结果**：
  - Raw 提供 favorable 中间点：Claude 保留 80% 成本优势且回收 50% 质量；GPT 保留 14% 成本优势但回收 79% 质量。
  - `Traj-drop` 在两种模型族上均成为最差策略（QRec 仅 28%/53%），证明 HC 轨迹对 LC 接收方的推理锚定作用不可替代。
- **跨域扩展结果**：
  - LiC（晚期需求揭示）：升级 QRec=86%、CSRet=36%，降级 QRec=31%、CSRet=53%，晚期任务倾向强后缀。
  - BrowseComp（渐进搜索）：升级 QRec=95.8% 但 CSRet=−30.0%（仍劣于 Abort+fresh），降级 QRec=56.7%、CSRet=76.8% 表现稳健。
- **总体结论**：交接不具备通用最优解；Raw 直传在升级侧严重失效，降级侧表现良好；接口选择必须与切换方向与任务信息动态联合优化。

## 相关工作脉络
- **模型路由与级联（Cascades/Routers）**（Chen et al., 2023; Yue et al., 2024; Hu et al., 2024; Aggarwal et al., 2025; Valkanas et al., 2025 等）：聚焦请求/对话级“选模型”决策，默认上下文透明传递，未建模中途接管非原生轨迹的隐性成本。
- **SWE-Router**（Son et al., 2026）：利用 LC 部分轨迹决定续跑或重开，但明确不转移轨迹；本文进一步探究“转移何种形式的轨迹”对质量的边际影响。
- **Directional Drift 研究**（Khraishi et al., 2026）：仅考察最终 turn 的模型切换漂移，缺乏多接口、多时机与成本维度的系统扫描。
- **Handoff Debt**（KC and Budathoki, 2026）：研究中断任务的后继“再发现成本”，模型作为无
