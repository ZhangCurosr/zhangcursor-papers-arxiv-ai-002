---
title: "Some-Emotions-Run-Deeper-Layer-wise-Probing-and-Causal-Inter"
source: https://arxiv.org/pdf/2609.01279v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:48:09"
field: "语言模型内部表征可解释性"
keywords: ["emotion representation", "layer-wise probing", "causal intervention", "early-exit", "large language models", "affective computing"]
innovations: ["系统揭示情感深度随文本源风格系统性迁移", "区分在线/离线干预证明情感层带因果效力", "探针选中层带支撑高效Early-exit情感分类"]
benchmarks: ["Emotion/CARER", "GoEmotions", "ISEAR"]
---

# 论文速读：Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models

## 一句话总结
本文系统研究了情感信息在解码器式大语言模型内部的层级分布规律，发现其并非固定于某一层级，而是随文本来源（从显式社交媒体帖子到隐性自传叙事）系统性迁移；并通过在线/离线干预与Early-exit实验，证实探针选中的层级带不仅可诊断、且具有因果效力，可支撑高效轻量级情感分类。

## 研究问题与动机
1. **情感信息是否集中在固定层级？** 现有研究多基于单一语料验证情感可解译性，未考察不同文本风格（显式表情标记 vs. 隐性情境推理）下情感表征深度的差异。
2. **可解译性是否等于因果相关性？** 探针分析仅证明信息可从表征中读出，但未验证该层级带是否在模型前向计算中起因果作用。
3. **情感层级带能否支持高效推理？** 若情感集中在特定宽层带，能否通过Early-exit截断Transformer计算，在保留精度的同时降低推理开销？
4. **跨数据集/跨情绪的情感表征是否共享？** 不同情绪类别与不同文本领域下选中的层带是否存在重叠，反映共享的情感计算机制？

## 核心贡献（创新点）
1. **首次系统刻画情感深度随文本源的风格依赖性**：对比三种英文情感语料（Emotion/CARER、GoEmotions、ISEAR），证明情感最佳探测层从输入邻近层（相对深度0.066）延伸至模型半深度以上（0.590），且排序在长度匹配后依然稳定。
2. **区分"可解译性"与"因果相关性"**：通过在线前向干预（α缩放）与离线特征缩放对比，证明探针选中层带的因果效力主要来自前向计算中的参与，而非最终缓存特征向量。
3. **提出基于情感层带的高效Early-exit方案**：轻量多分类Logistic回归头在探针选中层带上运行，平均较全深度exit提升6.9个百分点准确率，且增益非单调依赖于层带宽度（宽度1增益最大+7.8点）。
4. **揭示跨数据集/跨情绪的表征共享与分离**：own-band与transfer-band干预效果相近（差值−1.2点，不显著），但early-exit transfer增益较弱，表明情感敏感区域具有部分共享性但读取几何存在领域特异性。

## 方法详解
1. **层级探测框架**：对每个数据集、模型、目标情绪（fear/joy/anger/sadness）和Transformer层ℓ，冻结模型参数，使用concat(mean, max, min)池化拼接层内所有token表征，训练One-vs-Rest Logistic Regression探针（C=1.0, max_iter=5000），以验证集F1最大化确定最佳相对深度(ℓ+1)/L。
2. **在线前向干预**：在模型前向传播时，对选定层带施加乘法钩子（multiplicative hook），缩放因子α∈{0.0, 0.5, 1.5}，α=0完全消除、α=0.5衰减、α=1.5放大；比较own-band、同宽unrelated-band、跨数据集transfer-band的准确率下降幅度，检验因果特异性。
3. **离线特征缩放**：对已缓存的冻结表征直接施加相同缩放操作，测试静态表征层面的有用性，与在线干预形成对照。
4. **Early-exit分类**：在选定层带末尾截断前向传播，拼接带内各层concat(mean,max,min)池化输出，训练独立4类Multinomial Logistic Regression头；与全深度final w层、random band、low-sensitivity band对比。
5. **统计检验**：所有结论基于768配对设置（8模型×2数据集×4情绪×4宽度×3 α值），采用bootstrap置信区间、符号检验、Wilcoxon符号秩检验及BH-FDR校正，确保结果稳健。

## 实验与结果
**数据集**：Emotion/CARER（Twitter，6,000样本）、ISEAR（自传叙事，4,044样本）、GoEmotions（Reddit，3,069样本），均截取四情绪交集，按长度匹配排除短文本。

**模型**：8个1B–9B开源解码器LLM（Llama-3.2-1B/3B、Llama-3.1-8B；Qwen-3.5-2B/4B/9B；Granite-4.1-3B/8B）。

**核心结果**：
- **探测深度排序**：Emotion（均值相对深度0.066，中位数0.036）< GoEmotions（0.219）< ISEAR（0.590/0.598），长度匹配后维持0.131/0.288/0.642。
- **在线干预特异性**：own-band较unrelated-band更破坏性−6.1点（q<0.001）；α=0时宽度1–4特异性达−8.2/−13.4/−7.4/−10.8点；Granite家族特异性不显著（+1.2点，n.s.）。
- **离线干预较弱**：own-band较unrelated-band仅−0.8点（n.s.），表明因果证据主要来自在线前向参与。
- **Early-exit性能**：selected exit较full-depth +6.9点（q<0.001），较random +3.7点、low-sensitivity +8.2点、transfer +6.6点；较frozen RoBERTa-base +15.8点、DeBERTa-v3-base +21.5点。
- **宽度效应**：宽度1增益最大（+7.8点），宽度3最小（+5.9点），非单调性反映情感信号集中vs.容量权衡。
- **跨情绪脆弱性**：offline α=0时，fear最脆弱（−16.8点）、joy最稳健（−13.1点）、sadness作为source最破坏性（−16.5点）。

## 相关工作脉络
1. **Di Palma et al. (2025) LLaMAs have feelings too**：首次探测Llama系列情感表征，发现情感集中在early-to-mid层；本文扩展至三大家族/三语料/干预验证，揭示深度风格依赖性而非固定位置。
2. **Hewitt & Liang (2019) Probe design**：主张探针应尽可能简单（线性）以反映表征质量；本文遵循此原则，使用Logistic Regression+确定性池化，避免探针容量混淆。
3. **Tenney et al. (2019) BERTology**：发现BERT中句法/语义/事实信息分层编码；本文将此"深度结构"视角引入情感分析，证明情感深度随输入风格迁移。
4. **Geva et al. (2023) Factual recall in LLMs**：发现事实记忆在decoder-only模型中峰值较深；本文对照发现情感深度范围更广（从极浅到极深），反映情感表达多样性和事实检索的结构性差异。
5. **Sajjad et al. (2023) Layer dropping**：证明随机丢弃层会损害下游任务；本文精细定位情感敏感层带，实现因果特异性扰动（own vs. unrelated）而非全局消融。
6. **Xin et al. (2020) DeeBERT / Early exiting**：提出动态Early-exit加速推理；本文将其应用于情感任务，证明探针选中层带可作为高效读取点，且较传统固定exit更优。

## 局限性与未来方向
1. **数据集与语言范围受限**：仅三个英文语料、四种基础情绪，未覆盖surprise/disgust等 finer-grained标签，跨文化/跨语言generalizability存疑。
2. **模型规模与类型**：仅限1B–9B开源指令微调模型，未评估closed-weight模型、base checkpoint或更大规模（如65B+）模型。
3. **方法论保守性**：冻结参数+确定性池化+线性探针设计有意排除微调干扰，但可能低估任务适配后的表征重组能力；learnable attention pooling虽性能更高但训练不稳定。
4. **干预粒度粗**：当前使用连续层带（width 1–4）整带缩放，未尝试activation patching、direction-level editing、nullspace projection等更精细的因果定位技术。
5. **统计校正边界**：采用BH-FDR within comparison family，未施加更严格的family-wise correction；ISEAR数据来自HuggingFace社区上传，长期可复现性依赖数据源稳定性。

## 研究启发与可借鉴点
1. **风格依赖的深度定位范式**：将"文本源风格→情感深度"纳入表征分析框架，可迁移至偏见、价值观、常识等其他高阶语义成分的层级定位研究。
2. **在线vs.离线干预的对照设计**：两者分离可精确区分"计算参与"与"表征可用性"，为后续因果机制研究提供标准化评估协议。
3. **Early-exit作为高效情感classifier**：探针选中层带+轻量头方案在工业部署（实时情感分析、边缘设备）中有直接应用价值；非单调宽度效应提示需实证调参而非默认加宽。
4. **跨数据集转移的弱对比实验**：own vs. transfer band对干预vs. early-exit的不同表现，揭示了"因果相关≠读取最优"的深刻洞见，可引导未来研究设计更细粒度的 transferability benchmark。
5. **情绪脆弱性谱系**：fear最脆弱、joy最稳健的offline cross-emotion pattern，或反映训练数据分布偏差（负向情绪样本更少/更模糊），可作为公平性审计的切入点。

## 关键术语表
**Layer-wise probing**：使用线性分类器逐层检测Transformer中间表征中特定信息（如情感）的可解译性，以定位信息编码深度。
**Online forward intervention**：在模型前向传播过程中对选定层带施加乘法缩放扰动，测试该层级带对下游预测的因果贡献。
**Offline feature scaling**：对已缓存的冻结表征直接缩放，检验静态表征的有用性，不与在线干预构成因果链。
**Early-exit classification**：在选定层带末尾截断Transformer，用轻量分类头直接输出预测，以实现推理加速与精度权衡。
**One-vs-Rest (OvR) probe**：为每个目标情绪单独训练二分类探针，优于多类联合探针更能捕捉单情绪特异性层分布。
**Band width**：层带的连续层数（w∈{1,2,3,4}），影响信号捕获与off-peak噪声的平衡。
**BH-FDR correction**：Benjamini-Hochberg False Discovery Rate校正，控制多重比较下的假阳性率，提升统计结论可靠性。
**Normalized depth**：(ℓ+1)/L，将不同深度模型的层索引映射至[0,1]区间，便于跨模型比较。

## 可复现要素
- **数据集**：Emotion/CARER、GoEmotions、ISEAR均为HuggingFace公开数据集，训练/验证/测试划分使用seed=42的stratified split（Emotion/ISEAR为80/10/10，GoEmotions为70/15/15）。
- **代码**：论文未明确声明GitHub仓库，但附录提供CSV master文件（per-setting paired correctness contrasts）；probe配置（concat mean+max+min pooling + StandardScaler + LogisticRegression C=1.0 max_iter=5000 random_state=42）完整描述。
- **模型权重**：8个开源模型（Llama-3.2-1B/3B-Instruct、Llama-3.1-8B-Instruct、Qwen-3.5-2B/4B/9B、Granite-4.1-3B/8B）可从HuggingFace获取。
- **关键超参**：α∈{0.0, 0.5, 1.5}缩放因子；band width∈{1,2,3,4}；池化concat(mean,max,min)；探针Logistic Regression。
