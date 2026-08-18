---
title: "SIGMA-Lane-Scale-pyramId-Gated-MAmba-for-Temporally-Consiste"
source: https://arxiv.org/pdf/2608.16338v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:11:28"
---

# 论文速读：SIGMA-Lane: Scale-pyramid Gated MAmba for Temporally Consistent Video Lane Detection

## 一句话总结
针对视频车道检测中大车连续遮挡导致递归状态被噪声“污染”的问题，本文提出 SIGMA-Lane，通过在 SSM 的写入路径与残差融合路径引入障碍物先验双门控，并结合坐标一致对齐、历史结构检索与几何感知冷启动，在保持流式推理效率的同时显著提升了重遮挡场景下的时序稳定性与检测精度。

## 研究问题与动机
- **核心问题**：视频车道检测需跨帧稳定预测，但大型车辆遮挡会切断可见车道线索；在流式递归模型中，被遮挡帧的噪声特征一旦写入隐藏状态，便会沿时间累积并持续影响后续帧，产生“状态污染（state contamination）”。
- **现有方法不足（1）**：如 OMR 等遮挡感知方法仅将障碍物 mask 作为辅助输入送入 ConvLSTM 细化模块，并未直接约束遮挡特征写入递归状态的路径，属于间接保护，污染仍会通过 $\bar{B}_t x_t$ 项注入记忆。
- **现有方法不足（2）**：若特征与 mask 独立进行形变对齐，会导致坐标不一致，使门控信号偏离实际被遮挡区域，削弱遮挡抑制效果。
- **动机**：将遮挡退化形式化为 SSM 递归更新中的状态污染失效模式，主张将障碍物先验直接作用于 SSM 的写入项与残差融合项，并将时序滤波与空间结构恢复解耦，同时解决污染注入与拓扑丢失两个子问题。

## 核心贡献（创新点）
1. **状态污染视角与 SSM 双门控机制**：首次将遮挡噪声注入 SSM 隐藏状态定义为“状态污染”，并在写入路径（Input Gate）与残差融合路径（Output Gate）同步施加障碍物 mask 门控，与仅将 mask 作辅助输入的 ConvLSTM/Attention 方法形成本质区别。
2. **坐标一致性仿射对齐设计**：提出 Lane-Guided Affine Warp，对特征、历史车道 mask 与障碍物 mask 施加完全相同的仿射变换，确保门控信号与被调制区域在像素坐标严格对齐，避免独立对齐导致的信息错位。
3. **解耦的时序滤波与空间恢复双路径**：Mamba2 驱动的双门控模块负责抗污染的时序状态传播，SSR 交叉注意力模块负责从对齐历史中召回缺失车道结构，两者职责分离，兼顾流式效率与重遮挡拓扑重建。
4. **几何感知冷启动初始化**：设计可学习 Start Token（叠加 2D 位置编码投影），在视频流首帧观测写入状态前预先更新 SSM 隐藏状态，避免零初始化下首帧遮挡主导整个记忆库。

## 方法详解
- **基础框架**：继承 OMR 的 ResNet18 编码器与 eigenlane 解码器，替换其聚合模块。使用 Mamba2 作为时序核心，维护长度 $T=8$ 的 Video Memory Bank（历史特征 $\mathbf{F}$、障碍物 mask $\mathbf{M}$、上一帧车道 mask $L_{t-1}$）。
- **SSM-Consistent Dual-Gating**：
  - **Input Gate**：$x'_t = x_t \odot (1 - m_t)$，抑制高遮挡 token 的写入幅度，使 $\bar{B}_t x_t \to (1-m_t)\bar{B}_t x_t$。论文给出线性化误差递推证明：在 $\|\bar{A}_t\|\le\alpha<1$ 的合同性假设下，输入门控将遮挡噪声累积项缩放至 $\gamma \ll 1$，阻断污染向 $h_t$ 传播。
  - **Output Gate**：计算全局平均遮挡 $\bar{m} = \text{GAP}(M)$，经 MLP+Sigmoid 得残差保留比 $r$。融合公式为 $X_{\text{time}} = (1-M)\odot(X_{\text{in}}+X_{\text{out}}) + M\odot(r\odot X_{\text{in}}+X_{\text{out}})$，遮挡严重时降低当前帧特征混入比例，保留已传播的历史输出。
  - **Scale-pyramid**：并行低分辨率分支捕捉大范围时序上下文，通过 $\tilde{x}_t = x_{\text{out}} + \alpha_{\text{low}} \cdot \text{Upsample}(\text{Mamba}(\text{Pool}_s(X)))$ 与高分支残差融合。
- **Lane-Guided Affine Warp**：利用车道 mask 引导的全局平均池化估计相邻帧仿射残差 $\Delta\theta$，令 $\theta = I_{2\times3} + \Delta\theta$，同步 warp 特征、车道 mask 与障碍物 mask，保障三者在同一坐标系下进行后续门控与检索。最终 MLP 层零初始化使变换从恒等映射起步。
- **Structural Spatial Retrieval (SSR)**：Query 为当前帧特征 $x_t$，Key/Value 为对齐历史特征 $\tilde{x}_{t-1}$ 叠加车道 mask embedding $e(\tilde{\ell}_{t-1})$。通过多头交叉注意力计算软空间对应，从历史中召回被遮挡区域的局部车道结构，弥补双门控在弱历史状态下的恢复盲区。
- **Geometry-Aware State
