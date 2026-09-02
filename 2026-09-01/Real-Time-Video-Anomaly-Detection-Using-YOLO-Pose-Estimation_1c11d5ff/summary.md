---
title: "Real-Time-Video-Anomaly-Detection-Using-YOLO-Pose-Estimation"
source: https://arxiv.org/pdf/2608.31074v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:27:49"
---

# 论文速读：Real-Time Video Anomaly Detection Using YOLO Pose Estimation and CLIP-Based Semantic Scoring

## 一句话总结
本文提出一种轻量级两阶段实时视频异常检测框架，利用 YOLO v11n-pose 一次性完成人物检测与17关键点姿态回归，再通过 CLIP ViT-B/32 对裁剪后的人物区域计算与固定文本提示的余弦相似度进行语义打分；该设计去除了光流、独立姿态估计器与密度估算模块，在单张 NVIDIA Titan XP 上实现约 51 FPS 的端到端吞吐，并在三个基准数据集上取得 70.26%–89.26% 的帧级 AUROC。

## 研究问题与动机
- **人工监控疲劳瓶颈**：CCTV 操作员持续盯屏数分钟后注意力显著下降，亟需自动化系统辅助识别跌倒、冲突、异常姿态等事件。
- **现有pipeline过于繁重**：代表性多特征基线（结合 AlphaPose、FlowNet2 与 CLIP 特征并叠加 GMM/kNN 密度估算）吞吐量仅约 15 FPS，难以满足实时部署需求。
- **目标场景与硬件约束明确**：聚焦固定室内监控视角下的人物级异常，要求系统能在单张桌面级 GPU 上稳定维持 >30 FPS。
- **视觉-语言模型带来简化机会**：CLIP 的零样本文本对齐能力与 YOLO v11n-pose 的单次前向推理能力，为去除中间组件提供了可行的架构重构路径。

## 核心贡献（创新点）
- **架构极简替代方案**：用直接 CLIP 语义余弦相似度取代多特征拼接与密度估算，在维持相当准确率的同时实现 3.36× 吞吐量提升。
- **零样本提示分类机制**：仅凭 4 条预定义自然语言提示即可识别多种异常类型，无需任何异常类别的监督标注即可完成打分。
- **端到端实时验证与落地**：在 CUHK Avenue、ShanghaiTech Campus 及自建 CU Indoor 数据集上全面评测，并在朱拉隆功大学真实电梯区域 5 路 CCTV 流上实现稳定在线运行。

## 方法详解
- **Stage 1：人物检测与姿态提取**
  - 输入帧 $I_t$ 送入 YOLO v11n-pose，单次前向推理输出所有人物边界框 $b_i=(x_i,y_i,w_i,h_i)$、置信度 $c_i$ 及 17 个 COCO 拓扑关键点 $\mathbf{k}_i$（含可见性标志 $v_j^k$）。
  - 按 $p_i = I_t[y_i-\delta : y_i+h_i+\delta,\ x_i-\delta : x_i+w_i+\delta]$ 裁剪人物区域，padding 参数 $\delta=10$ 像素经实测可保留足够上下文且不过度引入背景干扰。
- **Stage 2：CLIP 语义异常打分**
  - 将 $p_i$ 缩放至 $224 \times 224$ 后输入 CLIP ViT-B/32 视觉编码器，得到 $512$ 维归一化图像嵌入 $\mathbf{f}_i^{\mathrm{img}}$。
  - 初始化时预编码 4 条固定文本提示（“A person lying on the floor” / “falling down” / “fighting with another person” / “sitting on the floor”）得到 $\mathbf{f}_j^{\mathrm{txt}}$ 并缓存，全程无需微调。
  - 单人异常分：$s_i^t = \max_{j \in \{1,\dots,M\}} \mathbf{f}_i^{\mathrm{img}} \cdot \mathbf{f}_j^{\mathrm{txt}}$（余弦相似度等价于归一化内积）。
  - 帧级异常分：$S_t = \max_{i \in \{1,\dots,N_t\}} s_i^t$，取帧内最高分作为该帧异常强度。
- **时序平滑与决策**
  - 对原始 $S_t$ 施加一维高斯核平滑（$\sigma=5$，半窗宽 $w=15$）抑制瞬时波动。
  - 阈值 $\theta$ 在验证集上以 $[0.1, 0.9]$ 网格搜索确定，最优值 $\theta^*=0.7$，满足 $\hat{S}_t \geq \theta$ 即判为异常帧。
  - 文中明确说明：“zero-shot”仅指 CLIP 打分阶段无需异常标注，YOLO 检测器仍为有监督微调。

## 实验与结果
- **数据集**：CUHK Avenue（37 视频/30,652 帧）、ShanghaiTech Campus（437 视频/317,398 帧）、CU Indoor Anomaly（40 视频/17,798 帧，自建室内4类异常数据集，33/7 划分）。
- **评估基线**：重新实现的 Multi-feat. baseline（YOLO 检测 + GMM/kNN 密度估算，融合 AlphaPose/FlowNet2/CLIP 特征）。
- **主要 AUROC 结果**：CUHK Avenue 89.26%（与基线持平）、ShanghaiTech 70.26%（持平）、CU Indoor 84.13%（超越基线约 2.01 个百分点）。
- **检测性能**：YOLO v11n-pose 在 CU Indoor 测试集上整体 Precision 91%、Recall 97%、mAP@.5 为 92%；正常姿态 Precision 达 95%，异常姿态 Recall 仍保持 96%。
- **吞吐对比**：端到端 51.39 FPS，较基线 15.32 FPS 提速 3.36×；各组件独立测得 YOLO 推理 87.76 FPS、CLIP 打分 124.04 FPS；实际流水线低于组件峰值主要受 CPU-GPU 传输、动态 batch、视频解码与 GUI 渲染开销影响。
- **最强结果**：CU Indoor 数据集 AUROC 84.13%，为本文在目标场景下的最优指标。

## 相关工作脉络
- **重建/预测类方法（Conv-AE [3]、DSTN [10]、MPED-RNN [11]）**：依赖正常序列建模或光流/骨架轨迹时序预测；本文转向视觉-语言语义对齐，摆脱了对显式时序建模与大量正常数据的强依赖。
- **多特征密度估算基线（Attr-Based [6]）**：拼接姿态、光流与深度特征后用 GMM/kNN 评分；本文证明直接 CLIP 余弦相似度可在相近精度下大幅简化 pipeline，并将吞吐提升 3.36 倍。
- **VadCLIP [9]**：同样使用 CLIP，但依赖可学习提示词与弱监督训练，面向离线分析；本文采用固定提示词实现零样本在线实时推理，定位更偏工程落地。
- **语言引导决策覆盖（Moriyama et al. [12]）**：侧重 CLIP 重训练-free 的决策改写
