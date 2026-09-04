---
title: "LeanGRPO-Eliminating-Redundant-Recomputation-in-Diffusion-RL"
source: https://arxiv.org/pdf/2609.03528v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:51:27"
field: "扩散模型强化学习系统优化"
keywords: ["diffusion RL", "GRPO", "trajectory log-prob", "recompute-free training", "shared-prompt parallelism", "provisional gradient correction"]
innovations: ["识别并消除on-policy扩散RL中update阶段数学冗余的重计算", "提出Shared-prompt并行布局使每卡显存负担降至1/R", "两种互补免重算调度：Retain（图保留）与Reweight（暂态backward+延迟修正）"]
benchmarks: ["HPSv2", "PickScore", "FLUX.1-dev", "Wan2.1-1.3B/14B", "SD3.5-Medium", "Wan2.2-TI2V-5B"]
---

# 论文速读：LeanGRPO-Eliminating-Redundant-Recomputation-in-Diffusion-RL

## 一句话总结
本文指出扩散强化学习（Diffusion RL）中 update 阶段的"重计算"在 on-policy 设定下数学冗余，提出 **LeanGRPO** 框架，通过重构数据并行布局与两种无需重计算的训练调度，在不改变优化目标的前提下实现最高 **1.83×** 端到端加速。

## 研究问题与动机
- 现有扩散 RL（FlowGRPO、DanceGRPO 等）采用"两阶段"流程：rollout 无梯度采样 → 更新阶段对选中 timestep 重新前向以重建可微路径；但当策略参数不变且后端相同时，这两步计算的 log-probability 值相同，update 重算本质冗余。
- 直观解法是在 rollout 时开启梯度追踪直接复用计算图，但终端 advantage 仅在 rollout 结束后才可得，期间需保留所有选中 timestep 的 computation graph 与 saved activations，GPU 显存随 |S| 线性增长，极易 OOM。
- 公开实现的代码审计（2026-08-24 snapshot，32 个方法）显示 **71.9%（23/32）** 存在目标重计算，其中 **95.7%（22/23）** 满足 graph-colocated 必要条件，说明该问题是系统性设计盲点而非个例。
- 现有改进（奖励分配、timestep 选择、采样效率、分布式编排）均保留两阶段执行范式，与本文正交，有显著的系统层优化空间。

## 核心贡献（创新点）
1. **首次形式化识别并量化 trajectory-logprob 扩散 RL 的 update 重计算瓶颈**：证明 on-policy + 同后端下 ρ=1、clipping 失效，重算是为丢弃的梯度路径"买单"，而非优化需要。
2. **Shared-Prompt 数据并行布局重构**：将所有 GPU 处理同一 prompt、跨卡并行生成 M 个样本，取代"每卡各自 prompt 本地多样本"的原布局，使每卡激活/梯度负担降至 1/R；与原生 DP 梯度等价（Appendix A 严格证明）。
3. **LeanGRPO-Retain：梯度追踪 rollout + 计算图保留至 advantage 可用后直接 backward**，零重计算，激活显存随 |S| 线性增长，适合小 timestep fraction / 低分辨率。
4. **LeanGRPO-Reweight：rollout 中对每个选中 timestep 做 provisional backward（advantage=1），立即释放激活；trajectory 完成后用真实 Aᵢ 修正累积梯度，再单次 reduce-scatter**，显存不随 |S| 增长，适合大分辨率视频 / LoRA；配合 coalesced sync 额外削减同步开销。
5. **开源代码与可复现实验**：github.com/coderwayne3025/LeanGRPO，覆盖 FLUX.1-dev / Wan / SD3.5 等多尺度 backbone 与 DanceGRPO / FlowGRPO / MixGRPO-Flash 等多种算法。

## 方法详解
### 3  Preliminaries — 轨迹 log-prob 扩散 RL 目标
- Rollout 记录不带梯度的 per-step transition log-prob：ℓⁱ,ᵗʳᵒˡˡ = log pθ_roll(zⁱₜ₋₁ | zⁱₜ, cᵢ)。
- 更新阶段重算得可微 log-prob：ℓⁱ,ᵗᵘᵖᵈ = log pθ_upd(zⁱₜ₋₁ | zⁱₜ, cᵢ)。
- GRPO ratio：ρᵢ,ₜ = exp(ℓᵢ,ₜᵘᵖᵈ − ℓᵢ,ₜʳᵒˡˡ)；clipped loss：ℒ_clip = ΣᵢΣₜ∈ₛ max(−Aᵢρᵢ,ₜ, −Aᵢ clip(ρᵢ,ₜ))。
- On-policy（θ_upd = θ_roll）+ 同后端 ⇒ ρᵢ,ₜ = 1（落入 clipping 区间内，clipping  inactive）⇒ 梯度简化为 G_native = −ΣᵢΣₜ∈ₛ Aᵢ ∇θ ℓᵢ,ₜᵘᵖᵈ。

### 4.1 Shared-Prompt Gradient-Enabled Rollout
- 设 R 卡、每 prompt M 个样本，Q = M/R（整除假设）。每卡并行生成同一 prompt 的 Q 个样本，而不是各自独立 M 个。
- 若 Q < 1（即 R > M），把 R 卡分成 P = R/M 组，每组 M 卡处理同一 prompt；不同组并行处理不同 prompt。
- 梯度等价性（Appendix A）：重索引 (b,m) ↦ (b,q,r) 为双射，Σ_b Σ_m Σ_t = Σ_b Σ_q Σ_r Σ_t，全局梯度项完全一致。

### 4.2 LeanGRPO-Retain
- Rollout 时对 t ∈ S 启用梯度追踪，保留 (ℓᵍʳᵃᵈ⁻ʳᵒˡˡᵢ,ₜ, graph, activations)。
- Advantage Aᵢ 可得后直接 backward：G_retain = −Aᵢ Σₜ∈ₛ ∇θ ℓᵍʳᵃᵈ⁻ʳᵒˡˡᵢ,ₜ = G_native（精确算术下等价）。
- 代价：saved activations 常驻显存，约随 |S| 线性增长。

### 4.3 LeanGRPO-Reweight（Algorithm 1）
- 对每个选中 timestep：做 `provisional backward`，loss = −ℓᵍʳᵃᵈ⁻ʳᵒˡˡ，得 gᵖʳᵒᵥ = −∇θ ℓᵍʳᵃᵈ⁻ʳᵒˡˡ，detach zₜ₋₁ 后继续 rollout；立即释放该 timestep 的 activations。
- 所有选中 timestep 共享同一 Aᵢ，provisional 梯度先累加：Gᵖʳᵒᵥᵢ = Σₜ∈ₛ gᵢ,ₜᵖʳᵒᵥ。
- 轨迹完成后 local correction：G_corrᵢ = Aᵢ · Gᵖʳᵒᵥᵢ = G_nativeᵢ（Appendix B.2 定理）。
- **关键分布式约束**（Appendix B.3 / Appendix C）：必须先 correction 再 reduce-scatter。若先同步，混合同一 prompt 下不同样本的 Gᵐⁱˣ = Σₘ Gᵖʳᵒᵥᵦ,ₘ 后，因各 Aᵦ,ₘ 不同，无法通过任何公共标量/逐元素向量恢复 Σₘ Aᵦ,ₘ Gᵖʳᵒᵥᵦ,ₘ（非识别性证明）。
- 因此 Reweight 在 provisional backward 阶段禁用同步，仅在 correction 后执行一次 reduce-scatter，并与 coalesced sync 结合削减开销。

## 实验与结果
**硬件**：单节点 8× NVIDIA RTX A6000（48 GB）；强缩放 4–32 GPU，FSDP2 节点内 sharding + 节点间 DP。  
**模型**：FLUX.1-dev (12B)、SD3.5-Medium (2.5B)、Wan2.1-1.3B/14B、Wan2.2-TI2V-5B。  
**算法**：DanceGRPO、FlowGRPO、FlowGRPO-Fast、MixGRPO-Flash。  
**精度/微调**：BF16 full fine-tune、BF16/FP32 LoRA。

**主要结果（端到端每步加速，取三跑均值）**：
| 配置 | Reweight | Retain |
|---|---|---|
| BF16 full-ft (DanceGRPO+FLUX) | **1.44×–1.81×** | 1.18×–1.29× |
| BF16 LoRA | 1.26×–1.43× | 1.27×–1.42× |
| Wan2.1-1.3B → 14B 跨尺度 | 1.24×–1.49× | 1.22×–1.46× |
| FlowGRPO-Fast / MixGRPO-Flash / FlowGRPO+SD3.5 | 1.25×–1.67× / 1.22×–1.47× / **1.43×–1.83×** | 1.19×–1.30× / 1.15×–1.29× / 1.12×–1.21× |
| 4→32 GPU 强缩放 | 稳定 | 稳定 |

**Time breakdown（Fig.4，BF16 full-ft，FLUX.1-dev）**：Native 重计算耗 402.45s；Retain 消除该开销（999.16 vs 989.44s）；Reweight 延迟同步仅 30.33s，总步时 535.15s → **1.81×**。BF16 LoRA 场景同步开销本身很小，两 variant 步时接近（956.36 vs 960.69s）。

**消融（Table 1）**：
- Shared-prompt layout 单独几乎无加速（+all-gather 略慢）；
- Coalesced sync 单独 → 1.33×；
- Retain（仅去重算）→ 1.29×；
- Reweight（去重算 + coalesced sync）→ **1.81×**。LoRA 下 coalesced sync 贡献几乎为零，提速主要来自去重算。

**收敛（Fig.5）**：两种 variant 与 Native 保留相近奖励上升趋势；按"达到目标奖励耗时"计：
- Wan2.1-1.3B：Native 13.00h → Retain 11.45h（1.14×）→ Reweight **9.68h（1.34×）**。
- FLUX.1-dev：Native 25.45h → 两者均 ~17.39h（**1.46×**）。

**GPU 显存（Fig.6）**：
- 图像生成（微 batch≥2）：Reweight 存 2 个完整梯度易 OOM；Retain 显存随 |S| 线性增长但低于 optimizer 峰值。
- 视频生成（微 batch=1，张量大）：Retain 在第 6 个选中 timestep 即 OOM；Reweight 稳定（仅多 1 个完整梯度/卡），且 1.22× 提速非靠多占显存（Native 同等峰值仅 1.03×）。

**数值对齐（Appendix E）**：
- Retain 在严格 deterministic 设置下与 Native **bitwise identical**（cos=1.0，RelL2=0%）。
- Reweight：梯度 cosine=0.999802，RelL2=1.99%（固定轨迹）；端到端真实 reward 路径下梯度 cosine≈0.9989–0.9997，Δθ cosine≈0.9975–0.9993，主要由 advantage 缩放顺序改变有限精度运算次序引起，非算法误差。

## 相关工作脉络
1. **DDPO / DPOK / FlowGRPO / DanceGRPO**：两阶段 rollout+recompute 范式代表；本文定位其为"数学冗余"的实现基线，22/32 审计目标均含该模式。
2. **VeRL-Omni / BiDiRL**：优化资源编排与异步执行；关注任务级调度，不触及单步内的策略重评估，正交。
3. **算法层改进**（奖励分配 Huang et al. 2026、timestep 选择 He et al. 2026、采样效率 Li et al. 2026a）：改善采样分布与 loss 设计，保留两阶段执行；本文改动"何时物化可微转移信息"，不影响上述改进。
4. **DiffusionNFT / V-GRPO / FDFO / TDM-R1 等 9 类方法**：优化 forward-process matching / ELBO / 偏好匹配 / 蒸馏目标，根本不涉及 sampled reverse transition log-prob 重建，故无目标重计算可供 LeanGRPO 消除。
5. **FSDP / activation checkpointing**：内部反向时的块级重算（与本文 "update-stage transition re-forward" 本质不同，后者是 rollout 后对同一 (zₜ, zₜ₋₁, c) 的第二次模型求值）；本文兼容激活检查点，但不依赖它。

## 局限性与未来方向
- **后端一致性假设**：要求 rollout 与 update 使用同一策略后端且 θ 在 optimizer step 前不变；跨后端 port（如 VeRL-Omni 用 vLLM-Omni 做 rollout、FSDP+Diffusers 做训练）需工程改造才能适用。
- **多 epoch / sample replay**：同一 rollout 数据的第 2 次及以上更新必须使用当前策略重算；LeanGRPO 仅对首次 on-policy 更新免重算，后续仍走 native recompute。
- **Reweight 的数值偏移**：provisional backward 将 advantage 缩放移到 backward 之后，BF16 下引入 ~2–5% RelL2；在极小梯度首轮 AdamW 更新会因坐标归一化放大符号扰动（附录 E 有定量分析）。
- **显存-计算权衡仍需人工选择**：当前通过超参手动切换 Retain/Reweight；未提供基于 workload 特征的自动调度器。
- **未来方向**：构建 workload 画像（分辨率、|S|、Q、是否 LoRA、是否 grad-checkpoint）自动推荐 schedule；扩展到 off-policy / multi-agent diffusion RL；与 reward modeling、timestep selection 算法层联合优化。

## 研究启发与可借鉴点
1. **"数学冗余重算"的系统层诊断框架可迁移**：凡是"rollout 无梯度 + update 同参数重前向"的 trajectory-based RL（如 RLHF 中的 trajectory sampling + policy gradient），均可套用本审计思路排查冗余。
2. **Provisional backward + delayed correction 的梯度编排技巧**：适用于"权重未知标量在 forward 后才可得"的一般策略梯度场景（如元学习、双层优化），核心是证明 local scaling 与 distributed reduction 的可交换条件。
3. **Shared-prompt 并行布局的梯度等价证明范式**（Appendix A 的双射重索引）可为分布式采样-训练解耦提供可复用的正确性论证模板。
4. **Coalesced sync 与去重算的解耦分析**（消融 Table 1）展示了如何将系统优化拆成独立正交组件并量化各自贡献，值得在训练系统论文中效仿。
5. **与本团队结合机会**：将 LeanGRPO-Reweight 应用于流匹配视频生成（Wan/TI2V 系列）可显著降低 OOM 风险；其 provisionally detached 设计可与 gradient checkpointing 叠加进一步压显存。

## 关键术语表
- **Trajectory-logprob Diffusion RL**：把去噪过程视作多步策略 π_θ，通过所选 timestep 的转移对数概率 log p_θ(zₜ₋₁|zₜ,c) 优化终端奖励的 RL 范式（DDPO/FlowGRPO/DanceGRPO 所属类别）。
- **Update-stage recomputation**：rollout 结束后对选中 timestep 再次前向以重建可微 log-prob 的计算，本文核心消除对象。
- **LeanGRPO-Retain**：rollout 开梯度追踪并保留选中 timestep 的 computation graph 与 saved activations，待 Aᵢ 可得后直接 backward，零重算但显存随 |S| 线性增长。
- **LeanGRPO-Reweight**：rollout 中每个选中 timestep 做 provisional backward（advantage=1）后立即释放激活，轨迹完成后用真实 Aᵢ 本地修正累积梯度再单次 reduce-scatter，显存稳定。
- **Shared-prompt parallelism**：将 M 个样本的生成均匀分配到 R 卡，每卡处理同一 prompt 的 Q=M/R 个样本，替代原生"每卡独立 M 个样本"布局，单卡负担降至 1/R。
- **Coalesced synchronization**：将每个选中 timestep 独立的 reduce-scatter 合并为 prompt-group 边界处的一次集中同步，主要贡献 Reweight 额外提速。
- **Provisional gradient / advantage correction**：Reweight 先用假想 advantage=1 得到 gᵖʳᵒᵥ = −∇θ ℓᵍʳᵃᵈ⁻ʳᵒˡˡ，事后乘 Aᵢ 得 G_corr = Aᵢ·Gᵖʳᵒᵥ，精确等价于 native gradient。
- **Graph-colocated**：rollout forward 与 update backward 在单一进程中可共享 autograd state，是 LeanGRPO 适用的必要系统条件（非充分）。

## 可复现要素
- **数据集**：HPSv2（Human Preference Score v2，Wu et al. 2023）、PickScore（Kirstain et al. 2023），均为公开benchmark。
- **模型权重**：FLUX.1-dev、Wan2.1、Wan2.2、SD3.5-Medium 均为开源/公开权重。
- **代码**：github.com/coderwayne3025/LeanGRPO（论文声明开源）。
- **关键超参**：BF16/FP32 混合精度、LoRA rank=64 / alpha=128、FSDP2 sharding、max grad norm=1.0（FLUX）/ 0.1（Wan）、3 次独立运行取均值。
- **随机种子**：model seed=42、SDE-noise seed=123456、deterministic PyTorch kernels + FlashAttention backward（严格复现实验）。
- **硬件**：8× A6000 48GB / 4–32× A6000 跨节点（强缩放）。
