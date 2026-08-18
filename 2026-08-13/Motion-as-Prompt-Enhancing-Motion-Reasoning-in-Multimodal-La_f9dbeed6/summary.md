---
title: "Motion-as-Prompt-Enhancing-Motion-Reasoning-in-Multimodal-La"
source: https://arxiv.org/pdf/2608.11655v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:12:20"
field: "多模态视频理解"
keywords: ["视频推理", "多模态大语言模型", "运动理解", "视觉提示", "关键帧采样"]
innovations: ["首次将帧间运动丢失定义为稀疏采样 MLLM 的输入侧瓶颈，提出跨帧运动恢复提示", "设计免训练的运动引导跨帧视觉提示框架 MaP，无需修改模型即可增强运动推理", "提出运动能量度量与自适应采样结合轨迹标记的完整方案"]
benchmarks: ["CLEVRER", "Something-Something-v2", "TempCompass"]
---

# 论文速读：Motion-as-Prompt: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting

## 一句话总结
论文提出了 **Motion-as-Prompt (MaP)**，一种免训练、模型无关的跨帧视觉提示框架，通过从完整帧率视频中恢复密集点轨迹、选择运动 informative 关键帧，并将帧间轨迹直接渲染到采样帧上，使冻结的 MLLM 能够直接感知被稀疏采样丢弃的物体位移、方向变化和动态交互信息。在 CLEVRER 和 SSv2 基准上，MaP 分别使 GPT-5.5 的运动推理准确率提升 **4.2%** 和 **8.9%**，且不损害非运动相关的视频推理能力。

## 研究问题与动机
1. **核心问题**：现有 MLLM 对视频采用稀疏均匀采样以控制视觉 token 和注意力成本，导致**帧间运动丢失（inter-frame motion loss）**——即采样帧之间的碰撞、加速、方向变化等关键过渡事件被遗漏。
2. **现有方法不足**：
   - 运动感知视频模型通常需要额外的微调或修改模型架构；
   - 现有视觉提示方法（如 SoM、GoM）主要为**帧内标注**，描述物体身份/空间关系，无法揭示跨帧运动证据；
   - 语义关键帧选择方法（如 AKS、FOCUS）倾向于保留语义显著帧，可能错过短寿命的动态事件（如碰撞瞬间）。
3. **关键观察**：被丢弃的运动可以从原始视频中恢复，并以显式视觉提示的形式重新编码到稀疏输入中，无需修改模型或训练。
4. **研究动机**：为运动中心视频推理提供简单高效的输入侧解决方案，同时保持方法对任意冻结 MLLM 的通用性。

## 核心贡献（创新点）
1. **首次将"帧间运动丢失"定义为输入侧瓶颈**：指出稀疏采样 MLLM 的核心限制在于跨帧运动证据的缺失，而非模型能力不足，并提出跨帧运动恢复提示这一新视角。
2. **提出 MaP 免训练框架**：结合运动引导采样与帧间轨迹标记，无需任何模型训练或架构修改即可增强运动推理，与需要微调的方法（如 VideoExp ert、Flow4Agent）形成本质区别。
3. **设计运动能量度量与自适应采样**：通过速度、加速度、曲率三个互补维度构建帧级运动能量 $M(\ell)$，并结合锚点覆盖 + 非极大值抑制（NMS）选择运动 informative 帧，区别于纯语义相关的采样策略。
4. **跨帧轨迹显式标记机制**：将相邻采样帧之间积累的轨迹段渲染为 Set-of-Marks 风格的视觉符号，使原本隐藏的位移和方向变化变得可见，与仅做帧内标注的 SoM/GoM 形成互补。
5. **系统验证与深入分析**：在 CLEVRER、SSv2、TempCompass 三个基准上验证，并提供帧预算敏感性、组件消融、预处理开销等全面分析。

## 方法详解

### 整体流程
给定全帧率视频 $V$（$N$ 帧）和文本提示 $T$，MaP 在保持总帧预算 $B = F \cdot t$ 不变的前提下，输出增广帧序列 $\text{MaP}(V)$，包含：(i) 运动感知的帧选择集合 $S'$（$|S'|=B$）；(ii) 叠加在这些帧上的帧间运动标记集合 $\mathcal{M}$。

### 1. 运动信号提取（Motion Signal Extraction）

**密集追踪**：使用冻结的点跟踪器 $\mathcal{T}$（CoTracker3），在全帧率视频上追踪 $G \times G$ 查询点网格（默认 $10 \times 10$），每隔固定间隔（如 1 秒）重新初始化网格以捕获新出现的物体，得到点轨迹 $\mathbf{p}_\ell^i \in \mathbb{R}^2$ 和可见性标志 $o_\ell^i \in \{0,1\}$。

**相机补偿查询点速度**：原始帧间位移 $\mathbf{p}_{\ell+1}^i - \mathbf{p}_\ell^i$ 包含查询点的物体运动和相机自运动。通过全局相似变换模型分离二者：
$$\mathbf{p} \mapsto s\mathbf{R}\mathbf{p} + \mathbf{b}$$
参数化为 $\pmb{\theta} = (a, b, t_x, t_y)$（其中 $a = s\cos\phi, b = s\sin\phi$），在相邻帧均可见的点集 $\mathcal{V}_\ell$ 上拟合：
$$\pmb{\theta}_\ell^\star = \arg\min_\pmb{\theta} \sum_{i \in \mathcal{V}_\ell} \|\mathbf{S}_\pmb{\theta}(\mathbf{p}_\ell^i) - \mathbf{p}_{\ell+1}^i\|^2$$
物体速度为去除拟合相机流后的残差：
$$\mathbf{v}_\ell^i = (\mathbf{p}_{\ell+1}^i - \mathbf{p}_\ell^i) - (\mathbf{S}_{\pmb{\theta}_\ell^\star}(\mathbf{p}_\ell^i) - \mathbf{p}_\ell^i), \quad i \in \mathcal{V}_\ell$$

**运动能量**：从三个互补维度量化帧级运动强度：
- **速度**：$\mathsf{spd}_\ell = \frac{1}{n_\ell}\sum_i o_\ell^i \frac{\|\mathbf{v}_\ell^i\|}{d}$
- **加速度**：$\mathsf{acc}_\ell = \frac{1}{n_\ell}\sum_i o_\ell^i \frac{\|\mathbf{v}_\ell^i - \mathbf{v}_{\ell-1}^i\|}{d}$
- **曲率**：$\mathsf{cur}_\ell = \frac{1}{n_\ell}\sum_i o_\ell^i \arccos\frac{\langle \mathbf{v}_{\ell-1}^i, \mathbf{v}_\ell^i \rangle}{\|\mathbf{v}_{\ell-1}^i\|\|\mathbf{v}_\ell^i\|} \cdot \frac{\min(\|\mathbf{v}_{\ell-1}^i\|, \|\mathbf{v}_\ell^i\|)}{d}$

其中 $d = \sqrt{H^2 + W^2}$ 为帧对角线，$n_\ell$ 为可见点数。各描述子经 95 分位数归一化后加权求和，并用长度为 $w=3$ 的移动平均平滑，最终归一化到 $[0,1]$：
$$M(\ell) = \Phi_w(\widehat{\mathsf{spd}}_\ell + \widehat{\mathsf{acc}}_\ell + \widehat{\mathsf{cur}}_\ell) \in [0,1]$$

### 2. 运动引导采样（Motion-Guided Sampling）

在帧预算 $B < N$ 约束下，通过三阶段机制选择索引集合 $S' \subset \{1,\ldots,N\}$：

**阶段一：锚点覆盖**：均匀采样 $n_a = \max(2, \lceil\alpha B\rceil)$ 帧作为锚点（默认 $\alpha = 1/4$），保证全局时间覆盖：
$$\mathcal{A} = \{\text{round}(\frac{k(N-1)}{n_a-1}) : k = 0,\ldots,n_a-1\}$$

**阶段二：运动峰值 + NMS**：剩余 $B - |\mathcal{A}|$ 帧通过贪心非极大值抑制分配给高能量运动峰值，最小时间间隔 $r = \max(1, \lfloor N/(2B)\rfloor)$ 防止峰值过度聚集。

**阶段三：重要性采样回退**：若 NMS 返回不足 $B$ 帧，按运动能量降序填充剩余槽位。

### 3. 帧间轨迹标记（Inter-Frame Trajectory Marking）

**运动点筛选**：对采样帧 $\ell$ 上的每个网格点 $i$，测量其后续 $\Delta$ 帧的最大位移，若超过帧宽 $\tau = 0.03$ 倍则保留，最多保留 $K$ 个点控制标注密度：
$$\max_{m=1,\ldots,\Delta}\|\mathbf{p}_{\ell+m}^i - \mathbf{p}_\ell^i\| > \tau W$$

**轨迹渲染**：对于采样索引 $S' = \{s_1 < \cdots < s_B\}$，在第 $s_k$ 帧上，为每个保留的可见点 $i$ 渲染自上一采样帧 $s_{k-1}$ 以来累积的轨迹段 $\gamma_k^i = (\mathbf{p}_\ell^i)_{\ell=s_{k-1}}^{s_k}$，以单色折线绘制，末端带圆形标记。

## 实验与结果

### 数据集与评估
- **CLEVRER**：合成场景中的运动密集型推理，含 5 个子任务（OE、MD、MC、MA、CI）
- **SSv2**（Something-Something-v2）：真实世界细粒度人机交互视频，采用四选一多项选择题，答案选项中的物体引用替换为"something"以避免物体识别捷径
- **TempCompass**：非运动相关的视频推理基准，用于验证方法泛化性

### 基线方法
- 关键帧选择：AKS、FOCUS（语义相关选择）
- 像素级视觉提示：SoM（叠加分割掩码和索引）、GoM（渲染场景图）

### 主要结果
| 模型 | 方法 | CLEVRER Avg. | SSv2 |
|------|------|--------------|------|
| Qwen3-VL-2B | Uniform | 57.6 | 51.8 |
| Qwen3-VL-2B | **MaP** | **60.4** (+2.8) | **53.2** (+1.4) |
| GPT-5.5 | Uniform | 74.9 | 71.1 |
| GPT-5.5 | **MaP** | **79.1** (+4.2) | **80.0** (+8.9) |

- MaP 在两个基准上均取得最佳平均准确率
- GPT-5.5 的提升幅度显著大于 Qwen3-VL-2B，表明更强模型能更有效地利用恢复的运动线索
- 语义关键帧选择方法（AKS、FOCUS）在 CLEVRER 上反而显著降级（GPT-5.5 从 74.9 降至 61.4/67.9），因连续过渡事件的帧级语义变化细微
- SoM/GoM 因密集静态标注干扰推理，普遍降低性能
- TempCompass 上 MaP 无性能退化（Qwen3-VL-2B: 67.6 vs 67.3；GPT-5.5: 88.6 vs 88.3）

### 帧预算分析
轨迹标记的收益随帧预算增加而增大（0.5 → 1 → 2 FPS），表明标记有效性取决于采样帧间运动表示的保真度，而非单纯由稀疏程度决定。

### 预处理开销
MaP 平均处理时间：CLEVRER 783ms、SSv2 609ms、TempCompass 1920ms，显著低于 SoM（43203ms）和 GoM（2381ms~47583ms），GPU 显存开销平均仅 0.35 GB。

## 相关工作脉络

1. **关键帧采样方法**（AKS、FOCUS、VideoTree、MDP3）：这些方法将稀疏采样视为证据选择问题，基于语义相关性、视觉相似度或 token 冗余决策，可能保留描述"视频里有什么"的帧但丢弃"物体如何运动"的证据。MaP 从运动密度角度重新定义采样目标，两者解决不同层面的问题。

2. **时空建模与视觉提示**（SoM、GoM、ViKey、STOP）：现有像素级视觉提示主要在单帧内编码对象身份/区域/空间关系，属于帧内语义标注。MaP 补充了这一缺失维度，通过跨帧轨迹标记揭示运动证据，与 SoM/GoM 形成互补而非替代关系。

3. **运动感知视频理解**（VideoExp ert、Flow4Agent、DynImG、MotionSight）：这些方法通过引入专用运动表征、架构组件或任务特定微调来增强运动理解，需要修改模型或额外训练。MaP 的核心区别在于**免训练、即插即用**，且直接从全帧率视频重建显式点轨迹。

4. **视频压缩与 Token 管理**（LongVU、Dynamic-VLM、FastVID）：这些方法通过压缩冗余帧或空间 token 支持更长输入，但仍解决计算效率问题。MaP 聚焦于**信息完整性**，确保关键运动证据不被丢弃。

5. **时序定位方法**（TimeChat、VTimeLLM、Seq2Time、Time-R1）：这些工作强调时间戳绑定、时序边界感知或序列知识迁移，主要通过训练增强时序 grounding。MaP 从输入侧增强运动可见性，无需修改模型。

## 局限性与未来方向

1. **低帧率下的轨迹保真度限制**：在 0.5 FPS 等极低帧预算下，采样帧间轨迹退化为粗 stroke，标记效果下降，需适度帧预算配合。

2. **点跟踪器的适用边界**：依赖 CoTracker3 等冻结点跟踪器，在遮挡严重、快速运动模糊或纹理缺失场景中可能丢失轨迹，影响运动能量估计准确性。

3. **视觉密度的权衡**：轨迹标记在视觉密集场景中可能遮挡原始视觉证据，对模型感知能力提出更高要求（如 SSv2 上小模型提升有限）。

4. **通用运动模型的探索**：当前仅使用点轨迹，未来可探索是否融合光流、对象级轨迹或物理属性等更丰富的运动表征。

5. **应用场景拓展**：目前验证集中在合成场景和人工动作视频，在机器人操作、自动驾驶等真实交互场景中的泛化性有待进一步验证。

## 研究启发与可借鉴点

1. **输入侧增强的新思路**：MaP 证明了在输入预处理阶段修复信息丢失（而非修改模型）是可行的低开销方案，这一思路可迁移到其他因采样/压缩导致信息损失的任务（如长视频理解、低帧率监控视频分析）。

2. **运动能量的多维度度量**：速度、加速度、曲率的组合度量具有良好的可解释性和计算效率，可作为运动显著性检测的通用组件，用于视频摘要、事件检测等下游任务。

3. **轨迹标记的视觉设计**：将跨帧轨迹渲染为 Set-of-Marks 风格符号，既保留了原始图像的完整性，又提供了明确的运动线索，这种"透明标注"设计可与现有视觉提示方法结合使用。

4. **免训练框架的灵活性**：MaP 对任意冻结 MLLM 有效且无需微调，这一特性使其易于集成到现有 pipelines 中，可作为即插即用模块应用于工业场景。

5. **帧预算与标记效果的协同分析**：研究发现标记收益随帧预算增加而非减少，这一反直觉结论提示研究者应重新审视稀疏采样的最优策略，避免过度追求极稀疏导致信息无法恢复。

## 关键术语表
**Motion-as-Prompt (MaP)**：一种免训练的跨帧视觉提示框架，通过恢复和标记帧间轨迹增强 MLLM 的运动推理能力。
**Inter-frame motion loss**：稀疏均匀采样导致的帧间关键运动过渡（如碰撞、加速）信息丢失问题。
**Motion energy**：由速度、加速度、曲率三个维度加权合成的帧级运动强度度量 $M(\ell) \in [0,1]$。
**Motion-guided sampling**：基于运动能量选择关键帧的策略，结合锚点覆盖、NMS 峰值选择和重要性回退三阶段机制。
**Trajectory marking**：将相邻采样帧间累积的点轨迹渲染为可视化标记叠加在目标帧上的技术。
**Camera-compensated velocity**：通过拟合全局相似变换分离相机自运动后得到的物体真实速度。
**Set-of-Marks (SoM)**：在视频帧上叠加分割掩码和数字标识符的视觉提示方法，用于空间 grounding。
**Frame budget**：MLLM 处理视频时允许的最大帧数限制，用于控制视觉 token 和注意力计算成本。

## 可复现要素
- **数据集**：CLEVRER、SSv2（Something-Something-v2）、TempCompass 均为公开基准
- **代码开源**：项目页面 https://github.com/SunVictor23/MaP，论文未明确说明代码实现语言
- **权重**：使用冻结的 MLLM（Qwen3-VL-2B-Instruct 本地权重、GPT-5.5 远程 API）、CoTracker3 点跟踪器（98 MB）
- **关键超参**：网格大小 $10 \times 10$、锚点比例 $\alpha = 1/4$、NMS 半径 $r = \max(1, \lfloor N/(2B)\rfloor)$、位移阈值 $\tau = 0.03$、采样率 1 FPS
