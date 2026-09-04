---
title: "Speak-for-Me-Giving-LLMs-the-Situational-Awareness-to-Partic"
source: https://arxiv.org/pdf/2609.03923v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:54:14"
field: "多参与者对话代理"
keywords: ["meeting delegation", "LLM agent", "situational awareness", "multi-party dialogue", "state tracking", "floor-taking decision"]
innovations: ["CAPA架构：perceive-act-recalibrate循环结合显式会议状态追踪", "双层Controller-Generator分解实现SILENT作为一等公民选择", "双裁判Re校准机制将feedback注入状态而非输出重写"]
benchmarks: ["AMI Meeting Corpus", "ICSI Meeting Corpus"]
---

# 论文速读：Speak-for-Me-Giving-LLMs-the-Situational-Awareness-to-Participate-in-a-Meeting

## 一句话总结
本文针对LLM会议代理在在线多轮对话中"不知何时该发言"的核心痛点，提出CAPA架构——通过显式维护会议状态（参与者立场、议题覆盖、发言权等）实现持续的情境感知，在AMI语料库上将沉默率从51.4%降至2.5%，同时保持0.6%的低幻觉率。

## 研究问题与动机
1. **核心问题**：LLM作为缺席参会者的代理时，在何时发言（floor-taking decision）这一二元决策上严重失败，51.4%的有效发言机会被错失
2. **现有方法不足**：纯prompt-based方法存在两大缺陷——(a) cue政策过度依赖显式点名，遗漏隐式交接；(b) content政策将话题相关性与命题覆盖混淆，导致冗余或幻觉
3. **根本原因**：缺少显式结构化的会议状态追踪，标准proactive agent无法建模参与者立场、未决议题、发言权等隐式信息
4. **动机**：现实中缺席参会者需实时干预影响决策，而不仅仅是事后总结；直接扩展上下文窗口无法解决长对话中的推理缺陷

## 核心贡献（创新点）
1. **CAPA架构**：提出perceive–act–recalibrate循环的固定权重架构，首次将会议状态显式建模为可更新的 schema-constrained 结构，与纯prompt方法的本质区别在于将决策从"即时上下文阅读"转为"持续状态维护"
2. **双层Controller–Generator分解**：先在战略层commit到离散命题z再执行生成，使SILENT成为对具体命题的一等公民选择，与monolithic generation的根本区别是分离了战略推理与表面实现
3. **双裁判Re校准机制**：通过Environment Judge和Delegate Judge分别评估预测误差和行动误差，将反馈注入状态而非输出重写，解决了output-refinement无法处理silent failure的问题
4. **episode-level评估协议**：基于participant-owned idea units设计评分方案，同步评估whether/when/what三个维度，与常规turn-by-turn文本重叠指标的本质区别是锚定实质性贡献机会

## 方法详解
**整体框架**：CAPA（Collaborative Agent Predictive Architecture）采用因果回放协议，每轮仅观测当前时刻前的对话前缀，更新会议状态m，决定是否干预

**四个核心模块**：
1. **Perceiver**：schema-constrained LLM调用，将新观察到的话语o_t与先验状态m_{t-1}映射为更新状态m_t，提取五个字段：active topic、decisions/open questions、prior coverage、participant stances、floor
2. **Controller**：读取m_t和participant context c_{u*}，选择a_t ∈ {SILENT, SPEAK}；SPEAK时先commit到离散命题z_t，委托三个helper（candidate curator、coverage evaluator、speaker scorer）辅助决策
3. **Generator**：以参与者风格实现z_t为自然语言输出，遵循plan-then-realize范式
4. **Recalibrator**：利用Predictor的预言与两个judge的评估，融合为结构化状态更新

**Shared Memory四区设计**：
- Long-term memory：稳定内部画像p_{u*}（角色、专业知识、沟通风格）
- Session memory：会议特定briefing b_{u*}（候选贡献点）
- Working memory：动态外部状态m_t
- Episodic memory：先前行动的内部痕迹

**关键约束**（六条Desiderata）：严格因果性、参与者锚定、生成前承诺、沉默敏感度、显式状态维护、状态导向修正

## 实验与结果
**数据集**：AMI Meeting Corpus（scenario portion），137场会议，四参与者（项目经理、工业设计师、UI专家、营销专家）设计遥控器的场景对话

**评估协议**：
- 每场会议前60%构建 participant profile，后40%为eval testbed
- 基于idea units构建bounded episode，cue→anchor→k-turn evaluation window
- 四种outcome：strict hit (Δ=0)、loose hit (0≤Δ≤k)、uncredited attempt、not-attempted

**主要结果**（Table 2）：
| System | Strict recall ↑ | Loose recall ↑ | Not attempted ↓ | Decision F1↑ | Hallucination ↓ |
|--------|-----------------|----------------|-----------------|--------------|-----------------|
| Transcript-only | 10.7% | 26.1% | **51.4%** | 38.1% | 1.5% |
| Reflexion | 10.3% | 24.3% | 49.5% | 47.6% | 0.3% |
| **CAPA** | **25.1%** | **52.2%** | **2.5%** | **63.0%** | **0.6%** |

**关键提升**：
- 沉默率从51.4%降至2.5%（-48.9pp）
- credited recovery翻倍（26.1%→52.2%）
- Decision F1提升24.9pp（38.1%→63.0%）
- 幻觉率控制在0.6%，冗余0.0%
- 93.8%的credited match为anchor-aligned或pre-anchor，中位提前1.0 turn

**ICSI跨语料验证**：10场真实研究组会议（平均6参与者），silence 1.2%，loose recall 73.6%，hallucination 1.8%

## 相关工作脉络
1. **Meeting agents（Hu et al., 2025）**：最接近的benchmark，仅记录prompt-only delegation的failure modes，无架构支持；CAPA提供显式状态管理作为load-bearing component
2. **POMDP dialogue management（Williams & Young, 2007; Young et al., 2013）**：传统slot-filling任务导向对话，非proactive multi-party intervention；CAPA借鉴belief-policy分解但实例化为schema-constrained LLM模块
3. **Inference-time correction（Reflexion, Self-Refine等）**：仅target linguistic output，无法处理silent failure；CAPA将feedback注入explicit meeting state
4. **Sequential decision under partial observability（Kaelbling et al., 1998）**：经典belief-over-latent-state框架；CAPA采用此范式但避免learned rewards/transition dynamics（因缺乏可靠meeting simulator）
5. **MeetMap（Chen et al., 2025）**：实时协作对话映射系统，侧重facilitation而非participant-specific delegation
6. **MUCA（Mao et al., 2024）**：multi-user chat assistant，偏向group facilitation而非absent stakeholder representation

## 局限性与未来方向
1. **语料局限**：主要评估基于AMI scenario portion（脚本化四人会议）；ELITR、ICSI等语料需适配；议会/正式机构数据集的rigid turn-taking结构与本文设定不兼容
2. **骨干模型依赖**：性能显著依赖底层LLM能力；虽然GPT-4o/Gemini表现稳定，但保守型模型（如Qwen）still有17.8%沉默率
3. **coverage–restraint trade-off**：off-topic率从基线随沉默下降而上升5.2%，fixed-weight LLM的参与策略调优仍是open design question
4. **实时部署挑战**：论文标注响应路径mean 21.14s（含throttle 10.01s），provider-only约11s；live meeting需更低延迟
5. **伦理风险**：live impersonation需explicit recurring consent；存在false pretenses impersonation、speech laundering、asymmetric advantage等misuse路径
6. **未来方向**：跨因果信息边界领域泛化（如customer-support handoff）

## 研究启发与可借鉴点
1. **显式状态 vs 上下文扩展**：证明raw-context scaling无法解决long-range opportunity recognition；结构化state tracking是multiparty delegation的必需组件，可直接迁移至其他实时交互场景
2. **双层决策分解**：Controller–Generator的plan-then-realize分离使SILENT成为一等公民选择，这一架构模式适用于任何需要"是否干预"决策的agent系统
3. **双裁判Re校准**：将error source分解为perception error vs action error，分别用Environment Judge和Delegate Judge评估后注入状态，为fixed-weight agent的self-correction提供新范式
4. **episode-level评估协议**：锚定idea units的whether/when/what三维评分框架，比turn-by-turn重叠指标更能捕捉intervention quality，可复用于对话agent评估
5. **schema-constrained decoding**：所有模块输出structured JSON，确保跨模块一致性且可审计，这一工程实践值得agent系统采纳

## 关键术语表
**CAPA（Collaborative Agent Predictive Architecture）**：本文提出的perceive–act–recalibrate循环架构，通过显式会议状态实现多参与者会议代理
**Idea Unit**：participant-owned proposition-level claim/proposal/decision-relevant fact，作为episode构建的基本单位
**State Maintenance**：显式维护包含active topic、decisions、open questions、participant stances、prior coverage、floor六个字段的会议状态m_t
**Cue Policy**：检测发言时机（显式点名vs隐式交接vsrole-specific机会）的策略，基线方法在此失败
**Content Policy**：选择发言内容的策略，基线方法将topical relevance与proposition coverage混淆
**Recalibration**：基于双裁判反馈更新会议状态的过程，将correction从output refinement转向state maintenance
**Decision F1**：评估代理是否在anchor turn同步干预的floor-taking alignment指标
**Strict/Loose Hit**：严格匹配（Δ=0）与宽松匹配（0≤Δ≤k）的credited recovery分类

## 可复现要素
- **数据集**：AMI Meeting Corpus（scenario portion），CC BY 4.0许可，137场会议公开可用
- **代码开源**：是，GitHub https://github.com/FKIRSTE/emnlp2026-meeting-delegation，MIT许可，包含evaluation protocol、prompt templates、judges、analysis scripts
- **关键超参**：
  - 默认上下文窗口N=20 preceding utterances
  - 评估窗口k=5
  - Temperature: Generator T=0.3，其余模块T=0.1
  - top-p=1.0, frequency penalty=0.0, presence penalty=0.0
- **骨干模型**：GPT-4o（2024-11-20 snapshot），额外测试Gemini-2.5-Pro、Llama-3.3-70B、Qwen3.6-27B
