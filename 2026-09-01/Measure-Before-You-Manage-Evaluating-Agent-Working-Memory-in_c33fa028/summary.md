---
title: "Measure-Before-You-Manage-Evaluating-Agent-Working-Memory-in"
source: https://arxiv.org/pdf/2608.31057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:15:53"
field: "AI Agent 系统与记忆管理"
keywords: ["agent working memory", "semantic heterogeneity", "memory management", "coding agent", "object-aware compression", "evaluation framework", "SWE-bench"]
innovations: ["揭示编码Agent工作记忆的语义异构性并通过类型化对象会计量化", "提出四级评估框架区分存储状态、交付上下文、管理工作与任务结果", "展示校准增益在保留任务上不转移，名义预算不等价于实际交付上下文"]
benchmarks: ["SWE-bench Lite", "Terminal-Bench"]
---

# 论文速读：Measure-Before-You-Manage-Evaluating-Agent-Working-Memory-in

## 一句话总结
本文通过对55条编码Agent轨迹的类型化对象分析，揭示Agent工作记忆中不同语义对象（工具输出、代码工件、指令、Agent状态）在大小、保留时间和压缩行为上存在显著异构性，并以此为基础评测了两种语义感知记忆管理策略（对象感知压缩OA与基于检索的策略GA），最终提出一个区分"存储状态—交付上下文—管理工作—任务结果"的四层评估框架，强调不能仅凭名义token预算评判记忆管理策略的有效性。

## 研究问题与动机
1. **工作记忆语义异构性未被量化**：编码Agent轨迹中的工具输出、代码工件、指令和Agent状态具有不同的语义角色、生命周期和表示形式，但现有记忆管理机制往往将其视为同质token池。
2. **校准增益的泛化性问题**：在开发任务上调优的记忆策略，其在保留任务上的表现是否稳定？等价的token预算是否等同于等价的实际交付上下文与管理成本？
3. **评估框架的缺失**：如何系统地区分存储状态、交付上下文、管理开销与任务结果，以避免将表面增益误认为策略有效性？

## 核心贡献（创新点）
1. **类型化对象层面的工作记忆表征**：首次对55条编码Agent轨迹中的工作记忆对象进行类型、大小、保留时长和压缩行为的多维统计，揭示工具输出占内容体积55.5%但仅占保留加权成本40.2%，而代码工件体积占28.3%、保留成本占38.9%，说明体积视角会低估工件的贡献。
2. **两种语义感知策略的案例研究**：提出并评测对象感知压缩（OA）策略和基于检索的策略（GA），前者基于类型权重、访问次数和版本陈旧性打分；后者融合近期性、语义相关性和重要性评分，展示异构记忆如何具体影响管理决策。
3. **四级评估框架（Stored State → Delivered Context → Management Work → Task/Process Outcome）**：提出评估记忆管理策略必须区分的四个层次，强调相同名义预算下不同策略的交付上下文和管理开销并不对等，校准显著的结果未必在保留任务上复现。

## 方法详解
### 对象类型与工作记忆模型
工作记忆对象分为四类：
- **Instructions（指令）**：受保护的系统/任务文本
- **Artifacts（工件）**：源代码视图、文件版本
- **Tool outputs（工具输出）**：执行反馈、错误信息
- **Agent state（Agent状态）**：模型生成的文本，非验证性推理

每个对象包含ID、类型/子类型、大小$ s_o $、创建步骤$ c_o $、首次驱逐步骤$ e_o $、表示形式（raw/compressed/summary/pointer）。对象驱逐不等同于永久删除，pointer形式可通过`recall_object`恢复。

### 内容会计量
$$ r _ { o } = \operatorname * { m a x } ( 0 , \operatorname * { m i n } ( e _ { o } , T ) - c _ { o } ) $$
$$ V _ { c } = \sum _ { o \in c } s _ { o } , \qquad R _ { c } = \sum _ { o \in c } s _ { o } r _ { o } $$
其中$ V_c $为内容体积，$ R_c $为保留加权成本。两者均为会计量，非账单或KV缓存度量。

### 对象感知压缩（OA）策略
评分公式：
$$ u _ { o } = \operatorname * { m a x } \bigl ( 10 ^ { - 6 } , b _ { t } e ^ { - d _ { t } a _ { o } } m _ { o } [ 1 + 0 . 35 \operatorname * { m i n } ( k _ { o } , 5 ) ] ( 1 - p _ { t } z _ { o } ) q _ { o } \bigr ) $$
$$ S _ { o } = u _ { o } / \operatorname * { m a x } ( s _ { o } ^ { \mathrm { rendered } } , 1 ) ^ { 0.5} $$
结合类型权重$ b_t $、衰减$ d_t $、子类型乘数$ m_o $、年龄$ a_o $、访问计数$ k_o $、陈旧标志$ z_o $和版本惩罚$ q_o $（旧版本artifact得分×0.25）。最多四轮降级：raw→compressed→summary→pointer。指令和新创建对象受保护。

**关键缺陷**：自动prompt包含更新访问时钟（包括pointer），导致LRU/OA的年龄/重用信号不纯净。

### 基于检索的策略（GA）
$$ S _ { o } ^ { \mathrm { GA } } = 0.5 \cdot 0.99^{s - \ell_o} + 3.0 \cos(\widetilde{e_o, e_q}) + 2.0 \widetilde{I_o} $$
三个分量均min-max归一化后加0.5基准：近期性（基于步骤计数器，非自动包含）、语义相关性（BAAI/bge-small-en-v1.5 embedding，query为issue+最新Agent状态）、重要性（单独模型请求评分1–10，缓存于run内）。

### 评估指标
主要过程指标：**重复调用数**——同一工具名+排序JSON参数与先前调用匹配的计为一次重复调用。注意：编辑后的合法重测也算重复，该指标度量流程规律性而非修复成功。

## 实验与结果
### 数据集
- **SWE-bench Lite**：55条完整轨迹，8个仓库（SymPy 31、pytest 8、seaborn 4、pylint 4、Sphinx 4、Django 2、Flask 1、requests 1）
- **Terminal-Bench探测**：18条轨迹，用于描述性边界比较
- **校准集**：15个SymPy任务
- **保留集**：8个任务（seaborn 2、pylint 2、pytest 2、requests 1、Sphinx 1）
- **检索跟进**：24个SymPy开发任务，8个完成全部六臂

### 基线
FIFO、LRU（含access clock缺陷）、Uniform Compression（UC，3步年龄阈值+FIFO）、Full-context无限制参考、LRU-Demand（LRU-D，仅READ/RETRIEVE/UPDATE更新时钟）

### 主要结果
**OA校准 vs 保留对比**：
| 对比 | n/m | Mean ∆ | 95% CI | Exact p | Holm p |
|------|-----|--------|--------|---------|--------|
| OA–FIFO（校准） | 15/12 | −1.633 | [−2.667, −0.700] | 0.0049 | 0.0146 |
| OA–LRU（校准） | 15/10 | −1.533 | [−2.633, −0.533] | 0.0215 | 0.0312 |
| OA–FIFO（保留） | 8/2 | −0.500 | [−1.250, 0.000] | 0.5000 | 0.5000 |
| OA–UC（保留） | 8/5 | −1.000 | [−1.750, −0.375] | 0.0625 | 0.1875 |

**GA检索对比（8任务完整六臂）**：
| 对比 | Mean ∆ | Exact p | Holm p |
|------|--------|---------|--------|
| GA–FIFO | −0.375 | 0.7500 | 1.0000 |
| GA–LRU-D | −0.125 | 0.3594 | 1.0000 |
| GA–UC | +0.125 | 1.0000 | 1.0000 |
| GA–OA | +0.500 | 0.2188 | 0.8750 |

**关键发现**：
- OA在校准集上显著优于FIFO（p=0.0049），但在保留集上不再显著
- 不同策略即使共享名义预算，实际交付上下文差异可达18.7%
- GA引入285次importance调用和OA引入169次summary调用，管理开销不可忽略
- 服务层重放显示：无限制臂在某任务达到37,883 tokens（超过32K限制6步），而受控臂均≤16,643 tokens

## 相关工作脉络
1. **MemGPT**（Packer et al., 2023）：管理文档分析与多会话对话间的记忆层级迁移，侧重跨会话记忆管理而非任务本地工作记忆。
2. **Generative Agents**（Park et al., 2023）：社会模拟中结合检索、反思与规划，本文仅借用其近期/相关/重要三组件，不涉及反思和社会环境。
3. **SWE-agent**（Yang et al., 2024）：展示Agent-计算机接口对软件工程行为的影响，侧重接口设计而非记忆管理评估框架。
4. **Lindenbauer et al. (2025)**：在SWE-bench Verified上比较观察遮罩与LLM摘要，本文与其不同在于聚焦任务本地工作记忆的语义异构性建模。
5. **Chintalapati et al. (2026)**：比较科学发现任务的记忆浓缩策略质量与token成本，本文补充了编码Agent场景下的类型化分析。
6. **Omri et al. (2026)**：刻画Agent记忆系统的构建/检索/生成成本，本文进一步区分存储状态、交付上下文、管理工作与任务结果四层。

## 局限性与未来方向
1. **样本规模小且仓库聚类**：55条轨迹来自8个仓库，保留集仅8任务且每臂仅运行一次，统计效力有限。
2. **修复成功评估缺失**：后续24个任务中的48个完整task-arm单元格缺乏有效的正式修复评估，Agent结束信号无法替代。
3. **OA生命周期信号缺陷**：自动prompt包含更新访问时钟（混淆渲染与需求），版本事件可能是渲染事件而非语义陈旧（附录F.1的案例：读取未修改源文件产生5次工具输出失效）。
4. **模型版本未固定**：使用的claude-opus-4.8别名不锁定服务修订版，辅助请求温度未记录。
5. **完整超参数搜索记录缺失**：OA的校准设计选择有存档默认值，但无完整搜索账本。

未来方向：改进生命周期信号（区分渲染与语义陈旧、隔离需求时钟）、在不同工作负载上验证框架、探索自动化语义感知策略。

## 研究启发与可借鉴点
1. **类型化对象会计方法**：将工作记忆按类型拆分统计体积与保留加权成本，可迁移至任何Agent系统，揭示"体积主导"与"保留成本主导"的排名反转现象。
2. **四级评估框架**：Stored State → Delivered Context → Management Work → Task/Process Outcome的分层评估思路，适用于所有记忆管理策略对比实验，避免单一token预算指标的误导性。
3. **校准+保留双阶段验证范式**：在开发任务上校准策略后，必须在保留任务上验证，且报告 nominal budget vs actual delivered context 的差异。
4. **重复调用作为过程指标**：在不依赖修复成功（可能难以评估）时，重复调用数可作为流程规律性的轻量指标，但需注意合法重测的歧义。
5. **与团队方向结合机会**：可借鉴其语义感知压缩思路，结合团队现有的工具调用优化或上下文压缩工作，设计类型感知的memory eviction策略。

## 关键术语表
**Agent working memory**：Agent在执行任务过程中临时存储和操作的类型化对象集合，包括指令、工件、工具输出和Agent状态。

**Semantic heterogeneity**：不同语义类型的记忆对象在大小、保留时长、表示形式和压缩行为上存在显著差异，不能视为同质token池。

**Object-aware compression (OA)**：基于类型权重、访问时间、访问计数和版本陈旧性打分的手写启发式压缩策略，支持raw→compressed→summary→pointer四级降级。

**Retrieval-based memory management (GA)**：融合近期性、语义相关性（embedding）和重要性评分的检索策略，源自Generative Agents但适配编码Agent场景。

**Retention-weighted cost**：$ R_c = \sum s_o r_o $，将对象大小乘以其在上下文中的保留步数，反映真实保留开销而非单次体积。

**Repeated call metric**：同一工具名+排序JSON参数与先前调用匹配计为一次重复调用，度量流程规律性而非修复成功率。

**Delivered context vs nominal budget**：名义token预算不等于实际交付上下文，不同策略共享预算时可能因表示形式差异导致实际上下文量相差达18.7%。

**Four-level evaluation framework**：将记忆管理评估分为存储状态、交付上下文、管理工作、任务/流程结果四个独立层次，避免单一指标误导结论。

## 可复现要素
- **数据集**：SWE-bench Lite（55条轨迹）和Terminal-Bench（18条轨迹），归档在working archive中
- **代码**：未公开；附录F提供审计脚本路径（analysis/build_appendix_evidence.py）和结果定位符
- **权重/模型**：claude-opus-4.8别名，服务修订版未锁定；embedding使用BAAI/bge-small-en-v1.5（revision 5c38ec7c405e）
- **关键超参**：OA类型权重$ b_t $（instruction 1.0/agent state 0.8/artifact 0.6/tool output 0.5）、衰减$ d_t $、陈旧惩罚$ p_t $、子类型乘数已存档；GA系数0.5/3.0/2.0；budget计算$ B_t = \max(\lfloor 0.15 P_t \rfloor, \lfloor F_t / 0.8 \rfloor + 1) $
- **环境**：SymPy/pytest/seaborn等仓库的本地Docker-free评估器；服务重放使用NVIDIA GB10（32,768 token限制，Qwen2.5-Coder-32B）
