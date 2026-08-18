---
title: "Motion-as-Prompt-Enhancing-Motion-Reasoning-in-Multimodal-La"
source: https://arxiv.org/pdf/2608.11655v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:32:59"
field: "多模态视频推理"
keywords: ["Motion Reasoning", "Multimodal LLM", "Visual Prompting", "Keyframe Sampling", "Trajectory Tracking", "Video Understanding"]
innovations: ["跨帧轨迹标记恢复稀疏采样丢失的运动信息", "运动能量驱动的关键帧选择机制", "训练-free的视觉提示增强方案"]
benchmarks: ["CLEVRER", "Something-Something-v2", "TempCompass"]
---

# 论文速读：Motion-as-Prompt: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting

## 一句话总结
论文提出 **Motion-as-Prompt (MaP)**，一种训练-free、模型无关的跨帧视觉提示框架，通过从全帧率视频中恢复稠密点轨迹并标记到稀疏采样帧上，使冻结的 MLLM 能够直接感知帧间运动信息，从而增强运动推理能力。在 CLEVRER 和 SSv2 基准上分别取得 GPT-5.5 的 +4.2% 和 +8.9% 提升。

---

## 研究问题与动机

1. **帧间运动丢失问题**：MLLMs 通常通过稀疏均匀采样（如 1 FPS）控制视觉 token 和注意力成本，但会丢失关键帧间的运动过渡信息（加速、转向、碰撞等），导致运动推理能力受限。

2. **现有方法的不足**：
   - Keyframe sampling 方法（AKS、FOCUS 等）基于语义相关性选择帧，可能遗漏运动密集的短暂事件；
   - Visual prompting 方法（SoM、GoM 等）主要标注单帧内对象/空间关系，无法揭示跨帧运动证据；
   - 运动感知方法需要修改模型架构或微调，缺乏通用性。

3. **核心洞察**：丢弃的运动信息可以从原始视频恢复并重新编码到稀疏视觉输入中，通过跨帧轨迹标记让冻结 MLLM 直接感知位移、方向变化和交互。

---

## 核心贡献（创新点）

1. **识别输入侧瓶颈并提出跨帧运动恢复提示**：首次明确将帧间运动丢失定位为稀疏采样 MLLMs 的输入端瓶颈，并引入跨帧运动恢复机制，从原始视频恢复丢弃的运动信息并在稀疏输入中直接可视化。

2. **提出 MaP 框架实现运动引导采样与轨迹标记**：结合运动引导的关键帧选择和帧间轨迹标记，构建训练-free、模型无关的视觉提示框架，无需修改 MLLM 架构。

3. **实验验证有效性**：在 CLEVRER 和 SSv2 基准上持续提升 MLLM 表现，GPT-5.5 分别获得 +4.2% 和 +8.9% 增益；消融表明轨迹标记是主要改进来源，且在大帧预算下更有效。

---

## 方法详解

### 整体流程
给定全帧率视频 V（N 帧，t 秒）和文本提示 T，MaP 在保持总帧预算 $B = F \cdot t$ 不变的前提下，输出运动感知的帧集合 $S'$ 和叠加在帧上的跨帧轨迹标记 $\mathcal{M}$，供冻结 MLLM 推理。

### 3.2 运动信号提取

**稠密跟踪**：使用冻结的点跟踪器 CoTTracer3 在全帧率视频上跟踪 $G \times G$ 查询点网格，定期重置网格以捕获中途进入的物体。

**相机补偿查询点速度**：
- 原始帧间位移包含查询点运动和相机自运动
- 使用全局相似变换 $S_\theta(\cdot)$ 拟合相机运动参数 $\theta = (a, b, t_x, t_y)$：
$$\theta_\ell^* = \arg\min_\theta \sum_{i \in \mathcal{V}_\ell} \|S_\theta(\mathbf{p}_\ell^i) - \mathbf{p}_{\ell+1}^i\|^2$$
- 物体速度为去除拟合相机流后的残差：
$$\mathbf{v}_\ell^i = (\mathbf{p}_{\ell+1}^i - \mathbf{p}_\ell^i) - (S_{\theta_\ell^*}(\mathbf{p}_\ell^i) - \mathbf{p}_\ell^i), \quad i \in \mathcal{V}_\ell$$

**运动能量**：从三个互补维度量化每帧运动强度：
$$\text{spd}_\ell = \frac{1}{n_\ell}\sum o_\ell^i \frac{\|\mathbf{v}_\ell^i\|}{d}, \quad \text{acc}_\ell = \frac{1}{n_\ell}\sum o_\ell^i \frac{\|\mathbf{v}_\ell^i - \mathbf{v}_{\ell-1}^i\|}{d}$$
$$\text{cur}_\ell = \frac{1}{n_\ell}\sum o_\ell^i \arccos\frac{\langle\mathbf{v}_{\ell-1}^i, \mathbf{v}_\ell^i\rangle}{\|\mathbf{v}_{\ell-1}^i\|\|\mathbf{v}_\ell^i\|} \cdot \frac{\min(\|\mathbf{v}_{\ell-1}^i\|, \|\mathbf{v}_\ell^i\|)}{d}$$
最终归一化并加权求和得到运动能量 $M(\ell) \in [0,1]$。

### 3.3 运动引导采样

给定运动能量序列 $M(\cdot)$ 和帧预算 $B < N$，通过三阶段机制选择索引集 $S'$：

1. **覆盖锚点**：均匀采样 $n_a = \max(2, \lceil\alpha B\rceil)$ 帧作为锚点，保证全局时间覆盖（默认 $\alpha = 1/4$）。

2. **运动峰值+非极大值抑制**：剩余帧按运动能量降序选择，施加最小时间间隔 $r = \max(1, \lfloor N/(2B)\rfloor)$ 的 NMS 约束，避免峰值过度集中。

3. **重要性采样回退**：若 NMS 返回帧数不足，按运动能量降序填充剩余槽位。

### 3.4 帧间轨迹标记

**移动点筛选**：对每个采样帧 $\ell$ 的查询点，若在后续 $\Delta$ 帧内最大位移超过 $\tau = 0.03$ 倍帧宽 $W$，则保留该点，每帧最多保留 $K$ 个点控制标注密度。

**跨帧轨迹渲染**：在采样帧 $s_k$ 上，绘制自上一采样帧 $s_{k-1}$ 以来累积的轨迹段 $\gamma_k^i = (\mathbf{p}_\ell^i)_{\ell=s_{k-1}}^{s_k}$，以单色折线形式呈现，末端带圆形标记，形成 Set-of-Marks 风格的视觉符号。

---

## 实验与结果

**数据集**：
- CLEVRER：合成场景中的运动密集推理，含 OE、MD、MC、MA、CI 五个子任务
- Something-Something-v2 (SSv2)：真实视频中细粒度人机交互理解
- TempCompass：非运动导向的视频推理基准，检验方法通用性

**基线方法**：
- 帧选择：AKS、FOCUS（语义相关）
- 像素级视觉提示：SoM、GoM

**主要结果（Table 1）**：

| MLLM | Method | CLEVRER Avg. | SSv2 |
|------|--------|-------------|------|
| Qwen3-VL-2B | uniform | 57.6 | 51.8 |
| Qwen3-VL-2B | **MaP** | **60.4** | **53.2** |
| GPT-5.5 | uniform | 74.9 | 71.1 |
| GPT-5.5 | **MaP** | **79.1** (+4.2%) | **80.0** (+8.9%) |

- MaP 在 CLEVRER 和 SSv2 上均取得最佳性能
- 相比最强基线 FOCUS，MaP 在 GPT-5.5 上分别提升 +1.2% 和 +2.4%
- SSv2 上小模型提升有限（+1.4%），归因于任务偏向动作语义识别

**泛化性（Table 2）**：TempCompass 上 MaP 保持或轻微提升非运动推理性能（Qwen3-VL-2B: 67.6% vs 67.3%，GPT-5.5: 88.6% vs 88.3%）。

**帧预算分析**：轨迹标记的收益随采样率增加而增大（0.5→1→2 FPS），表明高保真轨迹表征优于单纯补充稀疏性。

**消融（Figure 5）**：
- 轨迹标记：GPT-5.5 +1.3%，Qwen3-VL-2B +1.8%
- 运动引导采样：均 +1.0%
- 轨迹标记贡献更大，为主要改进来源

**预处理开销（Table 3）**：MaP 平均处理时间 783ms (CLEVRER)、609ms (SSv2)、1920ms (TempCompass)，GPU 显存平均仅 0.35GB，显著低于 SoM 和 GoM。

---

## 相关工作脉络

1. **Keyframe Sampling (AKS, FOCUS, VideoTree, MDP3)**：现有工作将稀疏采样视为证据选择问题，基于语义相关性、视觉相似度或 token 冗余性决策，但忽略帧间运动证据的连续性。MaP 定位为运动感知采样，强调保留动态过渡而非静态语义。

2. **Visual Prompting (SoM, GoM, ViKey, STOP)**：现有像素级提示主要在单帧内标注对象区域、身份或空间关系（intra-frame），无法揭示跨帧运动。MaP 补充此空白，提供 cross-frame 运动证据。

3. **Motion-aware Video Understanding (VideoExpert, Flow4Agent, DynImg, MotionSight)**：这些方法通过专用运动表征、架构修改或微调增强运动理解。MaP 与之本质区别在于训练-free、无需模型访问，通过输入侧预处理实现增强。

4. **Token Compression (LongVU, Dynamic-VLM, FastVID)**：压缩冗余帧/空间 token 以支持更长输入。MaP 不改变 token 总量，而是重新分配帧预算至运动信息密集型片段。

5. **Temporal Grounding (TimeChat, VTimeLLM, Seq2Time, Time-R1)**：关注时间定位任务，需模型修改或特定训练。MaP 针对通用运动推理，无训练负担。

---

## 局限性与未来方向

1. **点跟踪器依赖**：MaP 依赖冻结点跟踪器（CoTTracer3）恢复稠密轨迹，在遮挡严重、纹理缺失场景下跟踪质量可能下降，影响轨迹标记准确性。

2. **小模型利用受限**：SSv2 上 Qwen3-VL-2B 仅提升 +1.4%，弱于语义选择方法，表明小模型解析轨迹叠加的视觉/推理能力有限，未充分释放运动 cues 价值。

3. **帧预算敏感**：轨迹标记效能在低帧率（0.5 FPS）下有限——部分视频仅 2 帧导致轨迹退化为粗线段，且叠加标记可能遮蔽原始视觉证据。

4. **轨迹渲染策略**：当前仅渲染末尾采样帧上的轨迹段，未考虑起始帧或双向渲染，可能丢失部分运动上下文。

5. **未来方向**：探索自适应轨迹密度控制、与语义提示融合、扩展至长视频场景、研究更鲁棒的跟踪器集成。

---

## 研究启发与可借鉴点

1. **跨帧运动恢复范式**：将"丢弃信息可从原始输入恢复"的思路推广至其他丢失信号（如光照变化、遮挡恢复），可作为通用的输入侧增强策略。

2. **Set-of-Marks 风格轨迹可视化**：用轻量 polyline + 端点标记渲染运动轨迹，避免复杂 annotate 干扰 MLLM 感知，设计简洁有效，可直接迁移至其他视觉提示任务。

3. **运动能量三重度量（速度/加速度/曲率）**：综合描述运动动态特征，相比单一光流或帧差更丰富，可用于视频分帧、关键事件检测等下游任务。

4. **锚点+峰值NMS的采样策略**：兼顾全局覆盖与局部密集，避免语义选择器忽视均匀分布的时序证据，可推广至其他需要时空平衡的帧选择任务。

5. **训练-free 增强思路**：不修改模型架构、不微调参数，仅通过输入预处理提升性能，降低部署门槛，适合快速验证新能力。

---

## 关键术语表

**MaP (Motion-as-Prompt)**：论文提出的跨帧视觉提示框架，通过运动引导采样和帧间轨迹标记增强 MLLM 运动推理，无需训练或架构修改。

**Inter-frame motion loss**：稀疏均匀采样策略导致的帧间运动信息丢失问题，包括加速、转向、碰撞等关键过渡事件的不可见性。

**Motion energy $M(\ell)$**：综合速度、加速度、曲率三个维度的帧级运动强度度量，用于指导关键帧选择和轨迹渲染决策。

**Camera-compensated velocity**：去除全局相似变换拟合的相机自运动后，查询点的物体真实运动速度。

**Motion-guided sampling**：基于运动能量序列的选择性采样机制，通过锚点覆盖+峰值NMS+回退策略平衡全局与局部运动信息。

**Inter-frame trajectory marking**：将相邻采样帧间的点轨迹段渲染为可视化标记叠加在后一帧上，使 MLLM 可直接感知跨帧运动。

**Set-of-Marks (SoM)**：基础视觉提示方法，通过在帧上叠加分割区域和数字标识符支持视觉 grounding。

**Frame budget $B$**：允许处理的视频帧总数约束，$B = F \cdot t$，其中 $F$ 为采样率，$t$ 为视频时长。

---

## 可复现要素

- **数据集**：CLEVRER、Something-Something-v2、TempCompass（均为公开数据集）
- **代码**：项目页面 https://github.com/SunVictor23/MaP（论文声明开源）
- **模型权重**：Qwen3-VL-2B-Instruct（本地权重）、GPT-5.5（OpenAI API）；点跟踪器 CoTTracer3（98MB）
- **关键超参**：网格大小 $G \times G = 10 \times 10$，锚点比例 $\alpha = 1/4$，NMS 半径 $r = \max(1, \lfloor N/(2B)\rfloor)$，位移阈值 $\tau = 0.03$，采样率默认 1 FPS
- **预处理开销**：平均 GPU 显存 0.35GB，峰值 3.41GB

---
