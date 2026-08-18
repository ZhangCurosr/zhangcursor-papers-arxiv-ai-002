---
title: "TokenSTFormer-A-Tokenized-Spatial-temporal-Attention-Model-f"
source: https://arxiv.org/pdf/2608.16122v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:12:46"
field: "医疗 AI 与步态分析"
keywords: ["Adolescent Idiopathic Scoliosis", "Gait Analysis", "Spatial-Temporal Tokenization", "Vision Transformer", "Mobile Health Screening"]
innovations: ["提出 ScoliGait 配对数据集与运动学知识图谱", "设计空间-时间标记化模块强化特征表示", "引入双流 CLS 对齐损失提升分类鲁棒性"]
benchmarks: ["ScoliGait", "Transformer encoder baseline"]
---

# 论文速读：TokenSTFormer-A-Tokenized-Spatial-temporal-Attention-Model-for-Holistic-Motion-Analysis-in-Adolescent-Idiopathic-Scoliosis-Screening

## 一句话总结
本文提出了 ScoliGait 数据集（1,516 段步态视频配 X 光标注）和 TokenSTFormer 模型，通过将运动学知识图谱进行空间-时间标记化（STT），实现了基于步态视频的青少年特发性脊柱侧弯（AIS）无创筛查，准确率 0.787 显著优于基线 Vision Transformer。

## 研究问题与动机
1. **现有筛查方法局限**：Adams’前屈试验与 Scoliometer 测量依赖主观判断与专业设备，在大规模筛查中可拓展性差，且受体型影响。
2. **静态图像信息不足**：单张背部照片等方法无法获取生物力学动态信息，难以反映步态运动的真实运动学特征。
3. **已有视频/骨骼方法预处理复杂**：如 GaitEdge、SkeletonGait 等方法依赖合成轮廓图或骨架图，复杂预处理管线限制实际部署。
4. **缺乏配对数据集**：现有工作缺少步态视频与 X 光 Cobb 角（CA）金标准配对的公开数据，阻碍数据驱动方法发展。

## 核心贡献（创新点）
1. **构建 ScoliGait 数据集**：首次收录 1,516 段步态视频与对应 X 光图像及 CA 标注，提供 AIS 筛查的高质量配对数据；与以往仅使用静态图像或合成图的工作本质不同，提供了真实临床金标准标签。
2. **设计去标识化运动学知识图谱**：将步态视频转化为 238 维特征向量（运动空间 140 维 + 自骨架空间 32 维 + 信号相关性 66 维）；区别于直接处理原始像素或简单骨架图的方法，该方法通过先验知识函数显式建模运动学特征。
3. **提出 TokenSTFormer 与空间-时间标记化（STT）模块**：通过 2D 卷积核分别对运动学知识图谱的行/列进行划分，生成独立的时空 Token；与标准 ViT 将 patch 随机拼接不同，STT 保持空间与时间语义的结构化分离与对齐。
4. **引入辅助 CLS Token 对齐损失**：最小化时间分支与空间分支 CLS Token 的 MSE，促进双流表征的一致性；该辅助损失在同类医疗视频分类工作中较少见。

## 方法详解
1. **运动学知识图谱构建**：使用 YOLOv8 提取 2D 关节坐标，经先验函数变换得到 t 周期、v 变量的矩阵 M(t,v)，共 238 个特征，数值独立归一化。
2. **Spatial-Temporal Tokenization (STT)**：对运动学知识图谱分别用列向与行向 2D 卷积核生成空间 token \(z_{spatial}^{(v,d)}\) 与时间 token \(z_{temporal}^{(t,d)}\)，分别经 LayerNorm 后拼接：
   \[
   z_{input}^{(t+v,d)} = concat(LN(z_{temporal}), LN(Dense(z_{spatial})))
   \]
   空间分支额外接 Dense 层增强特征表达。
3. **网络架构**：基于 Vision Transformer，含 5 层 MSA-MLP 残差块，MHA 维度 6×256，MLP 维度 384，Dropout=0.1，集成 LayerScale 与 Stochastic Depth。
4. **损失函数**：主损失为二分类交叉熵（BCM），辅助损失为两个分支 CLS Token 的 MSE：
   \[
   Loss = Loss_{BCM} + Loss_{CLS}, \quad Loss_{CLS} = MSE(CLS_{temp}, CLS_{spt})
   \]

## 实验与结果
- **数据集**：ScoliGait，758 名参与者，1,516 个非重叠视频片段（30Hz，1080p，5 秒），按 2.2:1 比例分层采样。
- **划分**：训练 1,216 / 验证 150 / 测试 150。
- **基线**：配置相同的 vanilla Vision Transformer encoder（patch size 6×6）。
- **主要结果**：
  | 方法 | Accuracy | Sensitivity | Specificity | PV+ | PV− |
  |---|---|---|---|---|---|
  | Transformer encoder | 0.740 | 0.796 | 0.617 | 0.820 | 0.580 |
  | **TokenSTFormer** | **0.787** | **0.845** | **0.660** | **0.845** | **0.660** |
- **提升幅度**：相较于 Transformer encoder，准确率 +4.7%，敏感性 +4.9%，特异性 +4.3%，PV+ 与 PV− 均提升约 2.5–8.0 个百分点。
- **消融**：移除 LayerNorm 导致 Accuracy 降至 0.720；使用单一位置编码使 Accuracy 降至 0.687，证明 STT 中 LayerNorm 与独立位置编码的关键作用。

## 相关工作脉络
1. **GaitEdge [10]**：利用合成步态轮廓图进行分类，预处理复杂；本文直接使用手机视频提取运动学特征，无需合成图像。
2. **SkeletonGait [11]**：基于骨架图进行步态识别；本文的 238 维知识图谱在骨架基础上进一步融合了运动空间与时域相关特征。
3. **静态照片分类方法 [8]**：依赖单帧背部图像，无法捕捉动态生物力学信息；本文利用全周期步态视频，提供更丰富的运动学信号。
4. **传统筛查工具（Scoliometer 等）**：文献报告敏感性仅 0.37–0.51，特异性 0.84–0.96；本文方法在敏感性上大幅提升，接近临床可用水平。
5. **iTransformer [16]**：时间序列预测的逆注意力结构；本文受其启发在空间标记后引入 Dense 层增强跨变量特征交互。

## 局限性与未来方向
1. **样本规模有限**：数据集仅 758 名参与者，跨地域、跨人群泛化能力待验证。
2. **未与其他深度学习基线全面对比**：仅与 vanilla ViT 对比，未与近期医疗视频分类或时序 Transformer 方法进行公平比较。
3. **实时部署与移动端效率未充分评估**：系统虽标榜移动端可部署，但模型推理延迟、能耗等工程指标未报告。
4. **特征工程依赖先验知识**：238 维特征由人工设计定义，可能存在信息瓶颈，未来可探索端到端学习从原始视频直接提取特征。

## 研究启发与可借鉴点
1. **结构化标记化思想**：将表格/矩阵型数据按语义行/列划分 token，适用于任何具有时空双模态结构的中继特征，可迁移至步态分析、视频动作识别等任务。
2. **双流对齐辅助损失**：使用 MSE 对齐不同分支 CLS token 可促进表征一致性，适合多视角/多模态分类任务，减少单分支过拟合风险。
3. **移动设备友好型采集流程**：手机拍摄 + 固定距离与高度设置，便于低成本大规模筛查，为其他移动医疗 AI 应用提供采集范式。
4. **去标识化知识图谱构建**：通过先验函数将原始坐标转化为运动学特征，兼顾隐私保护与可解释性，可借鉴至其他需脱敏的生物医学信号处理。

## 关键术语表
**Adolescent Idiopathic Scoliosis (AIS)**：青少年特发性脊柱侧弯，青春期常见的脊柱侧向弯曲畸形。
**Cobb Angle (CA)**：通过 X 光片测量脊柱冠状面弯曲角度的金标准指标，>10° 诊断为侧弯。
**Kinematic Knowledge Map**：由 238 个运动学特征构成的矩阵，包含运动空间、自骨架空间与信号相关性三类特征。
**Spatial-Temporal Tokenization (STT)**：将运动学知识图谱按列/行分别卷积，分离为空间与时间 Token 的模块。
**ScoliGait dataset**：首个包含 1,516 段步态视频与对应 X 光 Cobb 角标注的公开数据集。
**Vision Transformer (ViT)**：将图像分割为 patch 并通过自注意力进行全局建模的标准 Transformer 架构。
**Positive Predictive Value (PV+)**：预测阳性样本中真正阳性的比例，反映阳性预测准确度。
**Negative Predictive Value (PV−)**：预测阴性样本中真正阴性的比例，反映阴性预测准确度。

## 可复现要素
- **数据集**：ScoliGait，论文未声明是否公开，需联系作者获取。
- **代码/权重**：论文未提及开源仓库或模型权重。
- **关键超参**：MLP 维度 384，注意力头数 6，层数 5，MHA 维度 1536，Dropout 0.1，学习率 2e−5，Warmup ratio 0.1，Cosine 调度，Batch size 64。
