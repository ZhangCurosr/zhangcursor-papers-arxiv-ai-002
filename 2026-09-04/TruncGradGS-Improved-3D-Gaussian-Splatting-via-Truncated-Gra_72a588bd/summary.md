---
title: "TruncGradGS-Improved-3D-Gaussian-Splatting-via-Truncated-Gra"
source: https://arxiv.org/pdf/2609.03534v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:26:28"
field: "3D Gaussian Splatting 优化与动态重建"
keywords: ["3D Gaussian Splatting", "gradient vanishing", "truncated gradient", "novel view synthesis", "dynamic scene reconstruction", "rasterization optimization"]
innovations: ["提出分段截断梯度场，用线性近似替代高斯导数尾部以缓解梯度消失", "仅对死亡Gaussian应用截断梯度并配合半径扩展与延迟剪枝，保证优化稳定性", "引入包含6个复杂动态场景的合成基准数据集，填补工业级动态评估空白"]
benchmarks: ["Mip-NeRF360", "Neural 3D Video (N3DV)", "Self-built dynamic synthetic dataset"]
---

# 论文速读：TruncGradGS-Improved-3D-Gaussian-Splatting-via-Truncated-Gra

## 一句话总结
本文提出分段截断梯度（piecewise truncated gradient）优化方法，解决3D Gaussian Splatting中因高斯导数尾部趋零导致的梯度消失问题；通过在低密度区域用线性近似替代真实梯度，有效扩展Gaussian原语的有效支持区域，显著提升静态与动态场景的重建质量与初始化鲁棒性。

## 研究问题与动机
- **梯度消失瓶颈**：标准3DGS中，远离Gaussian中心的像素因高斯分布指数衰减，其空间梯度 $\frac{\partial L}{\partial \mu^{2D}}$ 趋近于零，导致Gaussian无法向远处相关像素区域移动以改善重建。
- **Tile-based Rasterization局限**：图像被划分为16×16像素块，仅参与当前tile的Gaussian接收梯度；超出tile边界或低贡献的Gaussian被丢弃，无法获得远距离像素的反传梯度。
- **初始化敏感性**：随机初始化或稀疏SfM点云下，大量Gaussian初始位置与目标场景偏差大，传统梯度难以驱动其长距离迁移，重建质量骤降。
- **动态场景时序发散**：动态3DGS中相邻帧位移小，但远帧间累积位移大，Vanishing gradient问题被放大，导致浮游物（floaters）、模糊或缺失结构。

## 核心贡献（创新点）
1. **形式化分析2D高斯偏导数的梯度消失根源**：推导 $\frac{\partial G^{2D}}{\partial \Delta_x}$ 与 $\frac{\partial G^{2D}}{\partial \Delta_y}$ 的显式表达，揭示Gaussian函数项 $G(\Delta)$ 在尾部指数归零导致链式法则失效，为优化设计提供理论依据。
2. **提出分段截断梯度场（piecewise truncated gradient）**：在Mahalanobis距离超过阈值 $\tau$ 的区域，用连续线性函数替代真实高斯导数，有效扩展单Gaussian的有效优化覆盖范围，同时保持梯度连续性。
3. **设计三路梯度更新路径与Dead Gaussian筛选策略**：区分正常路径（$\sigma\alpha \geq 1/255$）、截断路径（$\sigma\alpha < 1/255$ 且 $d(p)^2 > -2\log\tau$）与跳过路径；仅对低不透明度（$\sigma_i < \tau_{\text{pruning}}$）的"死亡Gaussian"应用截断梯度，避免破坏已收敛的高质Gaussian。
4. **引入首个综合性动态高斯拼接合成基准数据集**：包含alley、windy tree、water cup、neon city、bouncy balls、underwater共6个场景，每个300帧、多相机环视、涵盖流体/颗粒/焦散/大气效应等复杂动态，填补工业级动态评估空白。

## 方法详解
- **梯度分析**：2D高斯密度 $G^{2D}(\Delta) = \exp\{-\frac{1}{2}\Delta^T \Sigma'^{-1} \Delta\}$，其对均值偏导为 $\frac{\partial G^{2D}}{\partial \Delta_x} = -G(\Delta)(\Delta_x a + \Delta_y c)$，当 $G(\Delta)\to 0$ 时梯度消失。
- **截断梯度公式**：定义Mahalanobis距离平方 $d(p)^2 = -2\log G$，设密度阈值 $\tau$（与rasterizer的alpha丢弃阈值一致）。截断梯度为：
  $$\widetilde{\nabla G} = \begin{cases} \frac{\partial G}{\partial \Delta_x}, & d(p)^2 < -2\log\tau \\ \min(m x + b,\, \frac{\partial G}{\partial \Delta_x}), & \frac{\partial G}{\partial \Delta_x}(b) < 0 \\ \max(-m x + b,\, \frac{\partial G}{\partial \Delta_x}), & \frac{\partial G}{\partial \Delta_x}(b) > 0 \end{cases}$$
  其中边界点 $x_b$ 由 $G^{2D}(x_b)=\tau$ 解得，斜率 $m$ 为可调超参，偏置 $b$ 保证连续性。
- **三路径机制**：
  - **Normal path**：$\sigma\alpha \geq 1/255$，使用真实梯度。
  - **Truncated path**：$\sigma\alpha < 1/255$ 且 $d(p)^2 > -2\log\tau$，使用线性近似梯度。
  - **Skip path**：$\sigma\alpha < 1/255$ 且 $d(p)^2 \leq -2\log\tau$，无梯度。
  - 额外过滤：仅当 $\frac{\partial L}{\partial \epsilon_i} < 0$（梯度降低loss）时应用截断路径，避免Gaussian相互排斥。
- **工程实现**：对死亡Gaussian扩展tile覆盖半径（radius padding），延迟剪枝至训练末尾，动态场景通过帧差生成运动mask忽略静态像素贡献，训练开销1–10×基线，推理速度不变。

## 实验与结果
- **静态数据集**：Mip-NeRF360、自建数据集静态子集；评估PSNR/SSIM/LPIPS，测试随机初始化与COLMAP初始化。
  - **随机初始化**：3DGS+Ours在Our dataset上LPIPS 0.1974 vs 基线0.2106，PSNR 25.16 vs 24.74；#GS从1.126M降至0.799M。Mip-NeRF360上PSNR提升1.26 dB。
  - **COLMAP初始化**：3DGS+Ours在Our dataset上PSNR 30.86 vs 30.07，SSIM 0.9097 vs 0.9058。
- **动态数据集**：Neural 3D Video（N3DV）与自研动态benchmark。
  - **N3DV**：4DGS+Ours PSNR 31.70 vs 30.22，#GS从2.806M降至1.733M；CEM-4DGS+Ours PSNR 32.29 vs 32.07。
  - **自研动态数据集**：4DGS(θ)+Ours PSNR 27.56 vs 26.02，LPIPS 0.1496 vs 0.2288，显著优于基线。
- **消融实验**：去除截断梯度导致最大性能下降（Alley PSNR 28.82→21.87）；不筛选Dead Gaussian或负alpha会引发发散；完整方法综合最优。

## 相关工作脉络
- **PAPR [ZPML23]**：同样针对点渲染的梯度消失问题，但聚焦stochastic point propagation；本文将其思想推广至Gaussian splatting与tile rasterization，提出专用分段截断梯度。
- **RAIN-GS [JHA*24] / Librated-GS [PZZ*25]**：解决初始化依赖SfM的问题（SLV初始化、monocular depth alignment）；本文从梯度场层面增强优化鲁棒性，与初始化策略正交互补。
- **Taming 3DGS [M*24] / ConeGS [BEG*26]**：改进densification/pruning策略；本文不修改密度控制逻辑，仅改变梯度传播方式，可无缝叠加于上述方法。
- **4DGS [YYPZ24b] / CEM-4DGS [KPLL25] / 4D-Scaffold [CCK*26]**：主流动态3DGS基线；本文方法作为plug-and-play模块直接集成，在N3DV与自研benchmark上均有稳定提升。
- **Pixel-GS [ZHL*24]**：引入pixel-aware梯度控制；本文方法从Gaussian原语层面扩展支持区域，视角不同，可相互补充。

## 局限性与未来方向
- **训练耗时增加**：截断梯度使更多Gaussian-pixel参与反向传播，训练速度降低1–10×，在大规模场景下可能成为瓶颈。
- **超参数敏感性**：斜率 $m$ 与阈值 $\tau$ 需经验设定，未做系统调优；不同场景可能需要差异化配置。
- **未联合优化densification/pruning**：除延迟剪枝外，未重新调参自适应密度控制模块，与定制化 densification 策略联合优化留有提升空间。
- **动态场景mask依赖**：运动mask通过帧差生成，对快速闪烁或光照剧变场景可能失效。

## 研究启发与可借鉴点
- **梯度场工程改造的普适性**：修改rasterization/backprop的梯度计算逻辑而非仅调整模型结构，是一种轻量且通用的优化增强范式，可迁移至其他基于可微渲染的场景。
- **Dead Gaussian筛选策略**：仅对低贡献原语应用激进优化、保护已收敛主体，这种"保守增强"思想可降低训练不稳定风险，适用于多数迭代优化流程。
- **阈值驱动的连续化设计**：用Mahalanobis距离划分"真实梯度区"与"线性近似区"并保证连续性，避免梯度突变导致的训练震荡，该思路可扩展至其他基于距离的soft selection场景。
- **合成基准的构建价值**：针对评测瓶颈（动态场景缺乏复杂非刚性运动与长时序一致性）构建高难度合成benchmark，可有效推动领域发展，值得借鉴。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：用一组3D椭圆高斯原语参数化辐射场，通过可微rasterization与alpha blending渲染图像的方法。
- **Vanishing Gradient（梯度消失）**：在Gaussian尾部，偏导数因指数项 $G(\Delta)$ 趋零而失去优化信号，导致原语无法向远处像素迁移。
- **Piecewise Truncated Gradient（分段截断梯度）**：在Mahalanobis距离超过阈值后，用连续线性函数替代高斯导数尾部，扩展有效梯度支持区域。
- **Mahalanobis Distance（马氏距离）**：衡量像素相对于Gaussian分布协方差的标准化距离，用于定义截断边界 $d(p)^2 = -2\log G$。
- **Adaptive Density Control (ADC)**：3DGS中通过克隆、分裂与剪枝动态调整高斯原语数量与位置的机制。
- **Dead Gaussian**：不透明度 $\sigma_i$ 低于剪枝阈值的Gaussian，对其应用截断梯度以避免过早淘汰。
- **Radius Padding（半径扩展）**：增大死亡Gaussian的tile覆盖半径，使其参与更远像素的截断梯度计算。
- **Delayed Pruning（延迟剪枝）**：将Gaussian剪枝操作推迟至训练末尾，允许低贡献原语在后期继续接收长距离梯度。

## 可复现要素
- **数据集**：Mip-NeRF360（公开）、Neural 3D Video（公开）、自研动态数据集（论文声明公开，作者承诺代码与数据publication后开源）。
- **代码/权重**：论文未提供，但声明"will make our code and data available upon publication"。
- **关键超参**：斜率 $m$、密度阈值 $\tau$（设为alpha-blending discard阈值，即1/255）、剪枝阈值 $\tau_{\text{pruning}}$、tile大小16×16；具体数值未在正文详述，见supplementary material。
