---
title: "Towards-Generalizable-Visually-Grounded-Exploration-of-House"
source: https://arxiv.org/pdf/2609.00845v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:53:33"
field: "具身智能与视觉语言模型"
keywords: ["visually grounded exploration", "generalizable manipulation", "state machine", "VLM benchmark", "embodied AI", "fine-grained visual grounding"]
innovations: ["提出VGEBench基准评估VLM在无手册条件下操作家庭设备的泛化视觉grounding能力", "设计Logic-Driven State Machine框架将设备交互逻辑形式化为确定性多轮状态转移图", "揭示细粒度视觉定位是当前VLM在家庭设备探索中的核心瓶颈"]
benchmarks: ["VGEBench"]
---

# 论文速读：Towards Generalizable Visually Grounded Exploration of Household Devices

## 一句话总结
本文提出 VGEBench，一个面向家庭设备的泛化视觉 grounding 探索基准，通过 Logic-Driven State Machine 框架模拟多轮假设-交互-修正过程，揭示了当前 VLM 在将抽象语义知识转化为精确物理操作方面的严重不足。

## 研究问题与动机
- **核心问题**：现有具身探索范式高度依赖人工标注轨迹的模仿学习，缺乏从抽象世界知识到细粒度视觉 affordance 的泛化能力，即"Generalizable Visually Grounded Exploration"。
- **现有方法不足**：
  - 传统机器人学习基准（如 BEHAVIOR-1k、RoboCasa）依赖特定环境训练的 IL/RL，无法泛化到未见设备。
  - LLM-based 工具学习主要在结构化 API 场景下工作，依赖显式文档，绕过了物理世界中"从抽象常识到具体交互接口"的核心挑战。
  - 现有具身视觉基准（如 OpenEQA、PhysToolBench）侧重被动观察或静态功能推理，缺乏主动状态改变操作和反馈驱动修正机制。
- **人类能力的启示**：人类遇到陌生设备时，会根据 affordance 感知形成假设、进行交互、并根据物理反馈修正理解，这一"假设-交互-修正"循环是实现泛化操作的关键。

## 核心贡献（创新点）
- **提出 VGEBench 基准**：构建含 968 个设备、26 个类别、7,888 个多视角图像和 14,953 个任务的交互基准，覆盖单轮与多轮指令场景。
- **设计 Logic-Driven State Machine 框架**：引入 Universal Category State Machine (UCSM) 和 Specific State Machine (SSM)，将设备交互逻辑形式化为确定性状态转移图，支持可验证的多轮交互反馈。
- **建立多维度评估协议**：从任务性能（SR、SSR）、效率（SPL、State-F1）、视觉定位（EIR、TIR、VSPS、GVPE）、探索效率（EER）四个维度全面评估 VLM。
- **揭示当前 VLM 的核心瓶颈**：实验表明细粒度视觉定位（而非导航）是主要瓶颈；最强模型 Gemini-3-Flash 仅达 54.27% SR，且 PSR（一次性最优路径成功率）仅 12.94%，说明现有模型缺乏零样本操作理解能力。
- **提供人机对比基线**：在相同条件下测试人类受试者，发现人类 SR 为 74.50%，进一步验证了当前 VLM 与人类之间的显著差距。

## 方法详解
- **数据构建流程**：从 Sketchfab 和 3D Warehouse 获取高质量 3D 模型，人工验证并采集 8 个视角图像（6 正交 + 2 轴测），共 7,888 视图。
- **UCSM 与 SSM 设计**：
  - 为每个设备类别设计 Universal Category State Machine (UCSM)，包含该类设备所有潜在功能和状态。
  - 根据具体 3D 模型的可见交互组件，裁剪得到 Specific State Machine (SSM)，确保每个设备有独立的逻辑图。
  - 组件名称仅用于内部状态跟踪，不对模型暴露，防止文本捷径。
- **交互机制**：
  - 动作空间：Press、Rotate、Push、Pull、Grasp、Timed_wait、Switch_view。
  - 分层验证协议：(1) 有效性检查（坐标是否在图像范围内、是否为有效视图类型）；(2) 几何与语义匹配（坐标是否在 SSM 定义的 bbox 内、动作类型是否匹配、参数是否满足转换条件）。
  - 反馈机制：成功转换 / 操作错误（正确组件但错误动作/参数）/ 无效坐标（null effect）。
- **任务评估规则**：
  - 任务由有序子目标序列 G = [g1, g2, ..., gn] 定义，完成当前子目标后才可见下一个目标。
  - 双预算约束：全局交互预算 B_global = λ × ΣC_opt（λ=5），局部预算 B_local 随子目标复杂度动态重置。
  - 完成标准基于状态序列而非轨迹，接受所有合法路径。
- **Dashboard View**：针对大型设备，引入模拟"靠近观察"的特写视图，通过主动 Switch_view 或被动空间交互解锁。

## 实验与结果
- **数据集**：VGEBench，含 968 设备、26 类别、7,888 视图、234 类组件类型、3,712 唯一组件、7,264 bbox、6,352 状态、15,861 转移边、14,953 任务（4,948 单轮 + 10,005 多轮）。
- **评估模型**：GPT-5-mini、Gemini-3-Flash、Doubao-1.5-Thinking-Vision-Pro、Qwen3-VL-8B-Instruct、InternVL3.5-8B、MiMo-Embodied-7B。
- **主要结果**：
  - 最强模型 Gemini-3-Flash：SR=54.27%，SSR=62.86%，SPL=39.34，State-F1=78.67，EER=64.36。
  - 第二名 Doubao-1.5：SR=16.68%，SSR=24.30%，大幅落后。
  - 其他模型 SR 均低于 15%。
  - PSR（一次性最优路径成功率）最高仅 12.94%（Gemini-3-Flash），说明现有模型几乎无法零样本理解设备操作逻辑。
- **人机对比**：Gemini-3-Flash SR=56.38%，人类 SR=74.50%，差距显著。
- **瓶颈分析**：
  - 细粒度视觉定位是主要瓶颈：GPT-5-mini 坐标误差比 Gemini-3-Flash 差 63.9%，平均错误距离差距更达 77.3%。
  - 视觉扰动敏感性：高斯模糊对模型影响最大（σ=5 时 Gemini-3-Flash SR 从 55.04% 降至 38.53%）。
  - 历史长度敏感性：Gemini-3-Flash 从完整历史中获益显著（SR 从 22.61% 提升至 55.04%）。
- **错误分布**：细粒度视觉定位错误占比 66.1%（无效坐标 35.5% + 组件选择错误 30.6%），远高于执行错误（17.2%）和导航错误（16.7%）。

## 相关工作脉络
- **工具学习与 API 交互**：Toolformer、ToolLLM、BFCL 等聚焦结构化 API 场景，依赖显式文档，不涉及物理世界的视觉探索和反馈修正。
- **GUI 代理**：OSWorld、AndroidWorld 操作 2D 屏幕，感知简化为 accessibility tree，缺乏 3D 物理世界的多视角主动感知。
- **细粒度视觉定位**：RefCOCO、Ferret 等评估指代表达理解，但未在交互约束下建模可操作的组件定位。
- **具身探索基准**：ALFWorld、BEHAVIOR-1k、RoboCasa 依赖特定模拟器训练，泛化能力受限；OpenEQA、PEAP 侧重被动观察，缺乏主动状态改变操作。
- **物理工具理解**：PhysToolBench、PhyBlock 关注静态物理推理，未建模多轮反馈驱动的交互循环。
- **定位差异**：VGEBench 聚焦无手册、无训练的自由探索，强调"假设-交互-修正"循环和细粒度视觉 grounding，填补了抽象知识到物理执行之间的空白。

## 局限性与未来方向
- **计算成本**：多轮交互评估需要大量推理时间和计算资源。
- **视觉编码需求**：高分辨率视觉观察对当前 VLM 的视觉编码效率和上下文窗口提出高要求。
- **抽象化局限**：使用渲染视图、离散原子动作和固定视角切换，未完全涵盖 sim-to-real 挑战（如连续机器人控制、自由相机导航）。
- **环境简化**：设备为中心的渲染图像不包含复杂背景、自然光照或真实遮挡，鲁棒性测试仅覆盖压缩、模糊和随机遮挡，无法完全替代真实观测。
- **未来方向**：探索更真实的 3D 仿真环境、引入连续动作空间、扩展至更多设备和场景、结合具身预训练提升细粒度视觉定位能力。

## 研究启发与可借鉴点
- **Logic-Driven State Machine 框架可迁移**：将设备交互逻辑形式化为确定性状态转移图的方法，可扩展到其他具身操作场景（如工业设备、厨房工具），为评估提供可验证的 ground truth。
- **双预算机制设计值得借鉴**：全局预算控制总资源消耗，局部预算动态重置避免局部最优，可有效防止交互循环，适用于其他长 horizon 具身任务。
- **Dashboard View 机制**：针对大型设备的特写视图解锁策略，平衡了全局视野和精细操作的需求，可作为通用的细粒度交互辅助手段。
- **多维度评估协议**：从任务性能、效率、视觉定位、探索效率四个维度的综合评估框架，为具身视觉 grounding 研究提供了全面的评测标准。
- **人机对比基线价值**：在相同接口和预算下测试人类，为模型能力提供了直观的参照系，建议后续研究沿用此方法。
- **错误分类体系**：将错误细分为细粒度视觉定位、执行错误、导航错误三类，有助于针对性改进模型能力。

## 关键术语表
- **Generalizable Visually Grounded Exploration**：泛化视觉 grounding 探索，指 VLM 在无手册、无训练条件下，通过主动视觉感知和反馈驱动修正来操作未知设备的能力。
- **Logic-Driven State Machine**：逻辑驱动的状态机，由 UCSM（类别级模板）实例化为 SSM（设备级具体图），形式化定义设备状态和交互转移逻辑。
- **UCSM (Universal Category State Machine)**：通用类别状态机，为每类设备定义包含所有潜在功能和状态的状态转移模板。
- **SSM (Specific State Machine)**：具体状态机，根据每个 3D 模型的可见组件裁剪 UCSM 得到的实例化逻辑图。
- **Affordance**：可供性，指物体表面上对行为的潜在提示或可能性（如旋钮暗示旋转操作）。
- **Hypothesis-Interaction-Refinement**：假设-交互-修正循环，人类探索陌生设备时的核心认知过程：形成假设→执行交互→根据反馈修正。
- **EIR (Effective Interaction Rate)**：有效交互率，衡量坐标落在有效交互组件 bbox 内的动作比例。
- **TIR (Target Interaction Rate)**：目标交互率，衡量坐标落在能缩短到目标最短路径的组件 bbox 内的比例。

## 可复现要素
- **数据集**：VGEBench（论文提供了完整的数据统计和构建流程，但代码/数据是否公开需进一步确认；论文未明确声明开源）。
- **代码/权重**：论文未提及代码开源或预训练权重。
- **关键超参**：冗余因子 λ=5；坐标系为 1000×1000 归一化坐标；动作空间包含 Press/Rotate/Push/Pull/Grasp/Timed_wait/Switch_view。
- **评估框架**：基于 ReAct 推理框架，暴露完整对话历史。
- **模型列表**：GPT-5-mini、Gemini-3-Flash、Doubao-1.5-Thinking-Vision-Pro、Qwen3-VL-8B-Instruct、InternVL3.5-8B、MiMo-Embodied-7B。
