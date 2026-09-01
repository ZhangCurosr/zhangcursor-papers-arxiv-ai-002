---
title: "The-Empire-Long-Divided-Must-Unite-Architectural-Convergence"
source: https://arxiv.org/pdf/2608.23953v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:11:31"
---

# 论文速读：The-Empire-Long-Divided-Must-Unite-Architectural-Convergence

## 一句话总结
本文对三个哲学立场截然对立（开箱即用/极简主义/全插件化）的开源 coding-agent harness 进行源码级纵向对比，发现它们在“长程自主运行”的共同选择压力下已收敛于一种包含五个核心元素的中间架构形态；而“外部可验证记录”是三方均未触及、且属于预测性空白的关键边界维度。

## 研究问题与动机
- 在同等前沿模型中，harness 已成为长程任务性能的首要约束（binding constraint），但现有研究多聚焦模型本身或工具接口，缺乏对 harness 层架构演进轨迹的系统刻画。
- 既有源码级分类（prior taxonomy）仅做静态快照且排除 deepagents，无法回答架构“正走向何处”。
- 既往工作将收敛仅视为附带观察，未将其作为核心论题，也未检验是否存在尚未收敛的负载维度。
- 需要厘清三方独立演化是否真的走向同一形态，以及其背后的驱动机制与失效边界是什么。

## 核心贡献（创新点）
1. 首次对 deepagents、pi、dsh 进行源码级架构对比，填补 harness 通用层的研究空白。相较于 prior taxonomy 排除 deepagents 且仅做静态快照，本文纳入完整三层并以 commit archaeology 追踪演化轨迹。
2. 提出并实证五元素“收敛中间形态（convergent middle form）”。相较于将收敛仅作为附带观察的既往文献，本文将其作为核心论题，并分解为平行发现、扩散与字面复用三种机制。
3. 归纳跨哲学复现的四条收敛故障线（sync/async drift、normalisation trust gap、string-matching semantics、silent vs loud failure）。相较于仅关注正确架构设计的文献，本文从人工复现缺陷出发刻画 seam 层面的同构性风险。
4. 识别“外部可验证记录”为尚未收敛的预测性边界。相较于协议层自上而下规定可审计性（如 Autogenesis），本文自底向上证明当前所有生产 harness 均在此维度缺失，并将其定位为下一轮竞争轴。

## 方法详解
- **研究设计**：解释性、理论构建的多案例研究，采用最大变异案例选择逻辑；deepagents 与 pi 作为 grounding cases 归纳五元素模型，dsh 作为 held-out check 进行确认性检验。
- **数据源**：四路证据交叉验证——（i）固定 commit 的源码与 file:line 锚定；（ii）commit/PR/issue 考古用于轨迹分析；（iii）沙箱内手工复现两个插件与两个缺陷；（iv）上游仓库官方确认（已提交 issue 与 PR）。
- **分析流程**：预设九维对比表（Table 2）避免事后拟合；使用并行 AI agent 辅助源码扫描，所有主张均经人工逐行核验；阅读深度不对称（deepagents/pi 逐行，dsh 一轮深度+插件实验）并公开声明。
- **收敛判定**：五元素需在两个 grounding cases 中出现线级证据，并在第三个 case 中得到确认（允许部分独立）；机制分解为 parallel discovery / diffusion / literal reuse，不假设完全独立发明。
- **边界度量**：外部可验证性按“可验证阶梯（verifiability ladder）”评级，从“无持久记录”到“运行时可重建”再到“第三方无需信任运行时即可核验”，三方均停留在第二阶。

## 实验与结果
- 本文为架构解构与缺陷复现研究，非传统基准评测实验。动机引用 prior work [4]：单次 harness 改动可使 Terminal-Bench 2 pass@1 提升数点、SWE-bench Verified 提升最多 15 点（模型固定）。
- **核心架构结果**：五元素在三案中全部复现，但实现形态各异（类层次/数据目录/运行时解析器）；会话记录强度呈明确阶梯（deepagents view < pi append-only < dsh event-sourced with runtime assertion）。
- **缺陷复现结果**（Table 3）：4 例复现缺陷分别落在 sync/async 漂移、路径标准化信任缺口、字符串匹配语义、静默降级四个 seam；其中 3 例造成不可逆损失（递归删除绕过 deny-rule、历史被 compaction 覆盖），仅 1 例可恢复（crash）。
- **最强/最弱定位**：dsh 在会话记录与沙箱策略上最强，但在外部可验证性上仍为 ×；pi 循环最精简（796 行 agent-loop，无 try/catch）；deepagents 通过委托 LangGraph 实现循环商品化。

## 相关工作脉络
- [5] B. Rombaut 源码级 scaffold 分类（13 个）：方法论继承（固定 commit、线级证据），但本文聚焦 general harness 层而非 coding-agent 应用，纳入 deepagents/pi/dsh，且做纵向轨迹而非静态快照。
- [6] 命令式 harness 12 个月纵向研究：量化发布速度 vs 质量，本文则量化演化所产生的架构形态与收敛机制。
- [7][8] 综述与阅读清单：建立词汇表但未深入实现，本文以源码阅读补全该缺口并检验收敛假说。
- [9] Autogenesis 协议：自顶向下规定可审计 lineage，本文自底向上记录实际形态，并指出协议核心诉求是三方均未实现的维度。
- [12][13] MCP/A2A：标准化调用与通信接口，属于协议层规范，本文关注 harness 内部的执行循环、状态管理与扩展 seam。
- [4] “harness 是 binding constraint” 的形式化论证：为本文研究 harness 架构本身提供首要动机与性能上下文。

## 局限性与未来方向
- N=3 且为单一时间点快照，外推到其他 agent 形态与未来版本需谨慎；最大变异选择部分缓解。
- 三方非完全独立（dsh 直接依赖 pi 的 @earendil-works/pi-ai 包），部分收敛属 diffusion/reuse 而非独立发明，作者已透明处理。
- 单分析师+AI 辅助读取，存在 instrument 与 survivorship bias；dsh 阅读深度低于前两者。
- 五元素构念在读取中归纳（非 a priori），存在 construct validity 风险。
- 未来方向：将 Table 2 维度转化为盲测 detector 进行大规模独立复现；探索恢复语义、quirk 覆盖与外部可验证记录等未收敛维度的工程突破。

## 研究启发与可借鉴点
1. **“更薄在 coaching，更厚在 durability”**：模型能力越强，harness 越应在持久化、恢复、上下文披露等基础设施上加厚，而非一味追求 minimalism。
2. **可逆性优先原则**：所有自动化破坏性操作（压缩、驱逐、删除）必须附带日志与回滚能力，不可逆边界是缺陷与不可恢复损失的高发区。
3. **模型 quirks 应数据化而非条件化**：将 provider 异常、token 限制、格式偏移统一收敛为配置文件或数据目录，便于扩展与维护，避免散落 conditionals。
4. **显式 seam 优于隐式堆叠**：无论采用 middleware/事件总线/service-locator 哪种组合风格，都必须暴露正交扩展点，避免单体耦合导致故障线扩散。
5. **同步/异步孪生必须共享 guarded core**：手工维护两套分支极易产生 trust gap 与静默退化，应通过编译器校验或单一起源派生规避。

## 关键术语表
- **Agent harness**：包裹 LLM 的运行时基础设施层，负责构建上下文、编排工具、执行业务循环与持久化状态，是长程自主 agent 的首要性能约束。
- **Convergent middle form**：三种对立哲学 harness 在相同选择压力下独立/半独立趋同的五元素架构形态。
-
