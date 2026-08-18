---
title: "Trajectory-Level-Automatic-Curriculum-Learning-for-Legged-Lo"
source: https://arxiv.org/pdf/2608.16164v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:13:34"
field: "足式机器人运动策略与课程学习"
keywords: ["curriculum learning", "legged locomotion", "automatic curriculum", "trajectory generation", "CVAE", "MCMC sampling", "unstructured terrain", "sim-to-real"]
innovations: ["在原始非结构化地形上以 CVAE 编码子轨迹上下文并训练 policy-conditioned 难度评估器", "用 MH-MCMC 在地图上的 waypoint 空间中搜索匹配当前策略能力带的轨迹 curriculum", "闭环 curriculum 通过 evaluator 重估实现目标难度带下的地形漂移，避免手工模板与预设轨迹"]
benchmarks: ["Map A（自建非结构化地形）", "Map HCL（Extreme Parkour 参数化模板集）", "跨方向 climbing robustness 评估", "sim-to-real Unitree Go2 部署"]
---

# 论文速读：Trajectory-Level-Automatic-Curriculum-Learning-for-Legged-Lo

## 一句话总结
本文提出 TACL（Trajectory-Level Automatic Curriculum Learning）框架，直接在原始非结构化地形图上自动为足式机器人学习轨迹跟踪策略生成可自适应的 curriculum；通过 CVAE 编码子轨迹上下文、训练 policy-conditioned 难度评估器，并用 MH-MCMC 搜索"当前能力刚好可学"的 waypoint 轨迹，形成闭环，在非结构化地形上显著提升成功率与跨方向泛化性。

## 研究问题与动机
- 非结构化地形缺乏显式的"任务难度排序"，难以像结构化地形那样直接构造 curriculum。
- 现有手工 curriculum（HCL）依赖参数化地形模板与人工预设 trajectory，易使策略过拟合到固定感知模式与固定轨迹几何，泛化受限。
- 现有自动 curriculum（ACL）仍多在参数化模板空间内采样，未能在原始高度图（unstructured map）上直接生成 trajectory-level curriculum，导致地形信息丢失与任务分布收窄。
- 核心难点在于：无需人工参数化地形的前提下，如何从当前 policy rollout 中学习"轨迹段在给定上下文中的实际难度"，并据此在地图上搜索可学习的候选轨迹。

## 核心贡献（创新点）
1. 提出 TACL 闭环框架，直接在非结构化地形图上手写式模板/预设轨迹地生成轨迹跟随 curriculum。与 HCL/ACL 的本质区别在于：输入是原始地图，输出是 waypoint trajectory，而不是参数化地形族内的曲线或参数步长。
2. 设计基于 transition context 的 CVAE 编码与 policy-conditioned 难度评估器，以当前 policy rollout 的成败为监督信号预测子轨迹失败概率；与仅用地形几何或 TD-error/surprise 等信号相比，评估器度量的是"对当前策略的实际难度"而非"内在地形难度"。
3. 将下一步 curriculum 的生成建模为含 hard/soft 能量的优化问题，用 MH-MCMC 在地图上行进方向与几何空间中搜索满足目标难度带的路径；与固定进度或对抗式环境生成不同，该方法通过持续重估 evaluator 使同一难度目标自动"漂移"到更难的现实地形区。
4. 在 Map A（非结构化）与 Map HCL（Extreme Parkour 模板集）上进行系统对比：TACL 相比无 curriculum 随机采样提升约 56.3%（轨迹级成功率），相比 HCL 在最难地形上提升约 18.5%，在多样逼近方向上最高提升约 39.74%；并提供 sim-to-real 部署验证。

## 方法详解
- **问题定义**：给定地形图 $\mathcal{M}$，在第 $k$ 次 curriculum 迭代为策略 $\pi_{\theta_k}$ 提供轨迹集合 $S_k=\{\mathcal{T}_j\}$；每条轨迹 $\mathcal{T}=\{p_0,\dots,p_N\}$ 由 waypoint 组成，相邻 waypoint 构成子轨迹 $\tau_i=(p_i,p_{i+1})$。
- **Transition context 表示**：每个子轨迹用 $\xi_i=(H_p,M_p,H_c,M_c,\delta_i)$ 描述——$(H_c,M_c)$ 为当前段的“高度 patch + 有效长度 mask"，$(H_p,M_p)$ 为前一段的同类型描述，$\delta_i$ 为前后段相对偏航角；patch 按 waypoint 方向裁剪并以起点附近中心线高程为基准去均值，mask 标记实际过渡长度覆盖区域；首段前向上下文置为平地区/零 mask/零角。
- ** rollout 与标签**：在当前 policy 下对可行轨迹 rollout，对子轨迹 $\tau_i$ 计算预期用时 $T_i=\|p_{i+1}-p_i\|_2/v+\epsilon$，成功（$y_i=0$）为在 $T_i$ 内到达，失败（$y_i=1$）为超时/坠落/其他终止；成功与失败样本被放入等长队列以平衡评估器训练数据。
- **CVAE 编码**：对 $(x_i=(H_c,M_c), c_i=(H_p,M_p,\delta_i))$ 训练条件变分自编码器，损失
  $\mathcal{L}_{\text{cvae}}=\lambda_H\|M_c\odot(H_c-H_c')\|_2^2+\lambda_M \text{BCE}(M_c,M_c')+ \lambda_{\text{KL}} D_{\text{KL}}(q_\phi(z_i|x_i,c_i)\|p(z_i))$，得到紧凑 latent code $z_i$。
- **难度评估器**：$\mathcal{D}_k(z_i)\in[0,1]$ 预测当前 policy 失败概率，用 BCE 训练；衡量的是政策相关难度而非绝对地形难度。
- **评价触发**：当政策在当代 curriculum 上达到平均 waypoint 完成率阈值 $\kappa_{\text{wp}}$，或达到更新周期 $K_{\text{upd}}$ 时触发评估器重训与新轨迹采样。
- **轨迹生成（sampler）**：候选轨迹能量
  $E(\mathcal{T})=E_{\text{hard}}(\mathcal{T})+\sum_{i=0}^{N-1}|d_i-d^{\text{tar}}|$, 其中 $d_i=\mathcal{D}_k(z_i)$；$E_{\text{hard}}$ 惩罚越界/非法/过长段；$d^{\text{tar}}$ 从截断高斯 $[\alpha,\beta]$ 中采样以聚焦“具挑战但可学”的难度带。
- **MH-MCMC 优化**：以衰减噪声扰动 waypoint，按 Metropolis-Hastings 接受率
  $A(\mathcal{T}\to\mathcal{T}')=\min(1,\exp(-(E(\mathcal{T}')-E(\mathcal{T}))/T_j))$ 接受/拒绝，温度与噪声尺度均随迭代衰减；由于对称高斯 proposal 抵消，接受比中 proposal 项消去。
- **闭环机制**：随着 policy 提升，原困难区经 evaluator 更新后 $d_i$ 下降，相同 $d^{\text{tar}}$ 将 sampler 推向真正更难的新区域，形成渐进 curriculum。

## 实验与结果
- **地形与设置**：Map A 为非结构化训练图（圆形平台、矩形块，障碍物高 0.05–0.6m）；Map HCL 来自 Extreme Parkour，含 steps/gaps/parkour/hurdles 四类障碍物各 10 级难度，每级 8 个实例、固定 8-waypoint 轨迹。策略基于 Extreme Parkour teacher-stage 的不对称 actor-critic。
- **基线**：RandST（随机可行轨迹）、HCL-E（Extreme Parkour 手工 curriculum）。
- **Map A（Q1）**：TACL vs RandST，16k epochs；在 climb-then-descend 三段轨迹上，RandST 可靠爬升上限约 0.3m，TACL 4k 即覆盖全高并达非零成功、8k 低于 0.65m 超 50%、16k 全部高度 >70%；全高平均成功率从 39.57% 提升到 94.54%（+55.97%，约 56.3%）。
- **Map HCL（Q2）**：50k epochs；HCL-E 在 steps/gaps/parkour/hurdles 最终成功率分别为 82/30/62/89（早期 5k 为 94/78/81/93，显式过拟合退化）；TACL 从低起点逐步上升到 98/96/47/96；四类平均从 65.75% 提升到 84.25%（+18.5%）。Parkour 因需要特定技巧、自动采样探索不足仍偏低（4%→47%）。
- **方向泛化（Q2延伸）**：在最高难度 0.5m step 地形上，以多样逼近方向评估，TACL 整体成功率从 HCL-E 的 58.68% 提升至 98.42%（约 +39.74%）。
- **CVAE 必要性（Q3/消融）**：去掉 CVAE 直接 CNN 融合 raw descriptor：evaluator 精度从 92.5% 降至 67.3%，采样平均能量更低但轨迹匹配能力差，最终成功率从 94.6% 降至 47.4%；作者解释 raw height patch 空间不光滑，微小几何扰动引起像素级大变化，CVAE 提供平滑 latent 空间关键。
- **Sim-to-real**：沿用 Extreme Parkour 两步蒸馏（stage-1 height-map teacher → stage-2 depth-image student，部署于 Unitree Go2 + Intel RealSense D435i），证明 curriculum 模块可与标准部署管线解耦复用。

## 相关工作脉络
- **HCL（Extreme Parkour 等）**：通过参数化地形族（steps/gaps/hurdles/ramps）与手工 waypoint 轨迹按启发式难度序列推进；本文定位差异在于不再依赖这些模板与预设轨迹，直接在原始地形上生成轨迹。
- **HACL**：利用历史 rollout 训练 reward predictor 辅助采样；本文与之不同在于以 sub-trajectory 层面的 policy-conditioned 成败而非 cumulative reward 为核心信号，并在轨迹几何空间中做 MCMC 搜索而非环境参数采样。
- **Unsupervised/Adversarial environment design（POET、GLAD 等）**：通过对抗或演化生成更难环境；本文不使用 adversarial/evolution 机制，而是通过学习"对当前策略实际难度"的可解释 evaluator 来维持 curriculum 的稳定收敛。
- **TD-error/surprise-based 方法**：以模型预测误差为优先信号；作者指出这类信号会混入不可约噪声，易将策略困在无产出的困难区；本文使用明确的成功/超时标签替代纯预测误差。
- **Learning-progress 导向方法**：关注性能提升最快区域；本文与之互补——在 trajectory 级别度量"当前可达性"并把目标难度带维持在 $[\alpha,\beta]$，从而避免过早收敛到过于简单的平坦区。
- **地形参数化自动 curriculum（scaling rough terrain、Allsteps 等）**：调整地形采样概率/离散 stepping-stone/潜变量 curriculum 范围；本文与之的关键区分是不进入参数化模板空间，直接在高度图上以 waypoint 优化生成 curriculum。

## 局限性与未来方向
- 高度结构化/需特定技巧的动作（如 parkour 类复杂机动）在当前通用轨迹采样下探索不足，成功率提升有限；作者建议引入弱先验或 demonstration-guided proposals 以提升此类行为的采样效率。
- MH-MCMC 在多维 waypoint 空间中搜索，理论上在高维长轨迹场景可能采样效率受限（论文未给出复杂度分析）。
- CVAE 编码依赖 rollout 数据构建，若实际部署中出现地形分布外移（out-of-distribution），evaluator 准确性可能下降；论文未讨论在线或持续校准策略。
- 难度目标带 $[\alpha,\beta]$ 与 MCMC 超参（温度衰减、proposal 方差等）依赖经验设定，跨地形迁移时可能需再调参。
- 当前仅验证到 Unitree Go2 平台与特定蒸馏管线，未展示对其他机型或不同传感器配置的普适性。

## 研究启发与可借鉴点
1. **"policy-conditioned 难度评估器"**：用当前策略 rollout 的成败信号（而非 TD-error 或 reward predictor）构造难度目标，概念简洁且与 curriculum 语义一致；可迁移到任何需要"当前能力对齐"的课程生成场景（手/足体操作、导航、无人机避障等）。
2. **CVAE 作为地形 patch 的平滑表征**：将非光滑的高度 patch + mask 压入低维 latent，显著提升 downstream 分类器的稳定性；可作为通用"地形感知→latent"的预训练工具，复用到 perception-based locomotion 或多模态 mapping。
3. **energy-based MCMC trajectory search**：将 curriculum 生成写成 $E_{\text{hard}}+E_{\text{soft}}$ 的形式，并用 MH-MCMC 在几何空间搜索；这一 formulation 可与约束优化、diffusion-based trajectory generation 结合，扩展为更 expressive 的 sampling backbone。
4. **evaluator 驱动的 curriculum 漂移机制**：保持目标难度带 $d^{\text{tar}}$ 固定而让 evaluator 随策略更新重估，使同一段"目标难度"自动映射到更难的真实地形；该设计避免了显式难度调度器，可推广到任何需要"自适应 ramp-up"的 RL 任务。
5. **可与本团队方向结合**：若团队在做多模态感知 locomotion 或 sim-to-real，可将本文的 teacher→student 蒸馏流程与本方法解耦复用；亦可在本文 evaluator 基础上叠加弱先验（如从人类演示或 imitation 数据初始化 proposal），缓解 parkour 等稀有技巧的发现瓶颈。

## 关键术语表
- **TACL**：Trajectory-Level Automatic Curriculum Learning，本文提出的面向足式机器人的闭环自动 curriculum 框架。
- **Handcrafted Curriculum Learning (HCL)**：基于人工设定的地形族与难度序列进行训练的 curriculum 方法。
- **Automatic Curriculum Learning (ACL)**：利用策略性能/学习信号自动选择或生成训练任务的 curriculum 方法。
- **Transition context ($\xi$)**：描述子轨迹及其前驱段几何与转向关系的特征向量，含前后 height patch、mask 与相对偏航角。
- **CVAE**：Conditional Variational Autoencoder，本文用于将高维 transition descriptor 压缩到平滑 latent space 的条件生成模型。
- **Difficulty evaluator ($\mathcal{D}_k$)**：以当前策略 rollout 成败为监督、预测子轨迹在当前策略下失败概率的分类器。
- **MH-MCMC**：Metropolis-Hastings Markov Chain Monte Carlo，本文用于在 waypoint 空间中优化轨迹能量以生成 curriculum。
- **Curriculum drift**：保持目标难度带不变、由 evaluator 随策略提升而重估地形难度，从而推动采样区域向更难真实地形自然迁移的现象。

## 可复现要素
- **数据集/地形**：Map A（作者自建非结构化图，0.05–0.6m 障碍）与 Map HCL（源自 Extreme Parkour codebase）；Extreme Parkour 代码公开，Map HCL 可通过其仓库复现；Map A 细节见附录。
- **代码/权重开源**：论文未明确声明 TACL 代码开源；策略基础来自 Extreme Parkour [3]（ICRA 2024）。
- **仿真平台**：Isaac Gym [7]，并行仿真。
- **关键超参**：目标难度带 $[\alpha,\beta]$、MH-MCMC 迭代数 $J$ 与温度/噪声衰减 schedule、CVAE 损失权重 $\lambda_H,\lambda_M,\lambda_{\text{KL}}$、评估器触发阈值 $\kappa_{\text{wp}}$ 与周期 $K_{\text{upd}}$；论文注明在 Appendix 给出，正文未披露具体数值。
- **实机**：Unitree Go2 + Intel RealSense D435i；两阶段蒸馏流程沿 Extreme Parkour 管线。
- **评测指标**：轨迹级成功率（所有 waypoint 均到达方为成功）、evaluator 分类精度、采样轨迹平均能量；跨方向鲁棒性以同一障碍物高度下的平均成功率汇总。
