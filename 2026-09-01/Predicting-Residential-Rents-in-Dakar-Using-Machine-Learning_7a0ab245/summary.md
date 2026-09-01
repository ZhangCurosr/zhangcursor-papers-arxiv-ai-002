---
title: "Predicting-Residential-Rents-in-Dakar-Using-Machine-Learning"
source: https://arxiv.org/pdf/2608.30865v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:43:01"
field: "住房价格/租金的机器学习建模与可解释性"
keywords: ["租金预测", "XGBoost", "LightGBM", "SHAP", "分位数回归", "目标编码", "达喀尔房产市场", "可解释机器学习"]
innovations: ["构建达喀尔长租房源标准化数据集并公开可申请获取", "同步开展Optuna调参、Gain/SHAP双重视角可解释性与分位数不确定性量化"]
benchmarks: ["80/20单划分测试R²=0.847/MAE=210,902/ RMSE=324,195 FCFA; 5折CV R²=0.852±0.011; 分位数区间名义80%但实际覆盖率64.2%"]
---

# 论文速读：Predicting-Residential-Rents-in-Dakar-Using-Machine-Learning

## 一句话总结
本研究基于2024-2025年网络房源爬虫数据，构建1,507条达喀尔住宅租金记录，系统比较线性回归、随机森林、XGBoost与LightGBM并借助Optuna调参，最终获得XGBoost最优解释方差比（单划分R²=0.847）；同时结合SHAP可解释性与分位数回归提供不确定性区间。

## 研究问题与动机
- 达喀尔租房市场信息碎片化：平台价格多为挂牌价而非成交实价，造成信息不对称，难以客观评估房屋租金价值。
- 传统线性/享乐定价方法难以刻画复杂非线性与特征交互，限制了多源房源变量对租金变化的解释力。
- 非洲城市长租市场的可解释机器学习研究较少，尤缺面向达喀尔的定价基准与不确定性评估。
- 缺少覆盖多区位、带标准化清洗流程的公开数据集，制约后续政策与市场比较研究。

## 核心贡献（创新点）
- 构建达喀尔长租房源结构化数据集并给出七步清洗与标准化流程，补齐当地数据基础。与已有非洲房产研究相比，本文更聚焦于“规范化+可复现”的长期租赁语料。
- 以Optuna对XGBoost/LightGBM进行贝叶斯超参搜索，并在同一 pipeline 内同步完成目标编码、特征工程与评估，提升表格数据上的可比性基线。与基线相比，核心区别是优化策略与评估一致性的系统化。
- 同时使用XGBoost gain与SHAP值进行特征重要性分析，揭示“位置”变量在不同度量下排名差异的方法论规律。与仅依赖单一重要性指标的先前工作相比，本文强调度量选择对结论的影响。
- 引入分位数回归为点预测提供80%名义覆盖区间，并评估其实际覆盖率与校准问题。与多数仅报告点估计的工作相比，本文额外给出不确定性量化与校准诊断。

## 方法详解
- 数据源与样本：2024年1月至2025年6月爬取达喀尔住宅出租房源，初采1,654条、36个变量；经过滤无效价格、住宅用途、剔除短租、标准化130+地名至24个区、处理异常值与缺失后保留1,507条、33个变量。
- 目标变换：租金右偏（偏度1.55），采用y=log(1+price)训练，预测反变换为price=exp(ŷ)−1，并以FCFA为原始尺度报告指标。
- 特征工程：新增luxury_score（泳池/海景/阁楼/高档次/电梯/车库/空调七项加权）、kw_score（18个溢价关键词加权）、nbre_chambre²（捕捉非线性）、ratio_sdb_ch（卫浴/卧室比）。
- 预处理器（ColumnTransformer四组）：数值块（中位数填充+StandardScaler）、类别块（type_appartement，最常见值填充+One-Hot）、位置块（KFold Target Encoding，训练_fold内中位数编码以避免泄漏）、布尔块（23项设施，最常见值填充）。
- KFold Target Encoding：enc(neighbourhood_k)=median{y_i: neighbourhood_i=k, i∈train}；测试集中未见类别回退至训练集全局中位数。
- 模型与调参：比较Linear Regression、Random Forest（300棵、max_depth=20、min_samples_leaf=2）、XGBoost baseline（1000树、lr=0.03、max_depth=6）、Optuna优化的XGBoost与LightGBM（8-9维超参：树数、学习率、深度、采样率、L1/L2正则、叶节点最小权重）。
- 评估协议：80/20单次划分，并辅以5折交叉验证；指标为MAE、RMSE、R²，均在反变换后的FCFA尺度计算。
- 可解释性：XGBoost gain（节点分裂带来的损失平均下降占比）与TreeSHAP（按观测分解至基值ŷ₀≈13.44，对应约686,870 FCFA）；并对比两者在location_encoded上的排序差异。
- 不确定性量化：用reg:quantileerror目标训练q10/q50/q90三个XGBoost模型，构造名义80%预测区间，检验 Empirical Coverage 与区间宽度。

## 实验与结果
- 最强模型：XGBoost tuned（Optuna）在80/20测试集取得R²=0.847、MAE=210,902 FCFA、RMSE=324,195 FCFA。
- 紧邻次优：LightGBM tuned 取得R²=0.842、MAE=215,259、RMSE=329,978。
- 相对线性回归（R²=0.679、MAE=257,147、RMSE=470,262）提升显著；XGBoost tuned 较线性回归R²提升约16.8个百分点。
- 5折交叉验证（log尺度）均值：XGBoost tuned R²=0.852±0.011，LightGBM tuned R²=0.851±0.013。
- FCFA尺度的5折再次校验显示更保守表现：R²=0.782±0.034（区间0.750–0.848），RMSE=369,744±26,230 FCFA，提示单次80/20划分偏高估风险。
- 可解释性主发现：gain口径前几位为luxury_score(23.7%)、nbre_chambre(14.4%)、nbre_chambre²(12.3%)、vue_mer(7.7%)、kw_score(7.3%)；SHAP口径前六位为luxury_score(0.163)、location_encoded(0.155)、kw_score(0.132)、surface(0.107)、nbre_chambre(0.075)、vue_mer(0.072)。Top6特征的value与SHAP相关系数均>0.89，其中vue_mer达0.993、kw_score达0.952。
- 区间校准：q10–q90名义80%区间在测试集上Empirical Coverage仅64.2%，中位区间宽度376,376 FCFA，呈现明显欠覆盖/偏窄。

## 相关工作脉络
- Hedonic定价基准（Rosen, 1974）：本文以该框架解释租金由区位/面积/卧室/设施等特征隐式定价。
- 非洲城市房产ML研究：Lagos（Abidoye, 2018）、Kampala（Irumba, 2015）、冈比亚大 Banjul 短租（Samb等, 2025）；本文填补达喀尔长租+可解释+不确定性的系统分析空白。
- 树模型在房价预测中的优势（Chen&Guestrin, 2016; Ke等, 2017）：本文验证非线性方法与特征交互对R²的关键增益。
- SHAP可解释框架（Lundberg&Lee, 2017; Lundberg等, 2020）：提供局部可加分解并与gain对比，避免单一重要性度量的误导性。
- 高基数目标的鲁棒编码（Pargent等, 2022）：本文采用KFold Target Encoding以减少位置变量带来的数据泄漏。
- 空间/出行可达性对房价影响（Chica-Olmo等, 2020）：作为对照，本文缺少精确地理坐标与到心/交通/绿地的距离，是精度差距的来源之一。

## 局限性与未来方向
- 数据选择偏差：主要来自在线平台，偏向上档房源；非公开渠道/非上市房源缺失，难以代表全市市场。
- 价格口径为挂牌价而非成交价，存在谈判降价导致的系统性偏高偏差。
- 无精确经纬度与到中心/交通/服务设施的距离等时空特征，限制区位效应的精细刻画。
- 样本量偏小（1,507）且24个区位分布不均，低样本区泛化能力受限。
- 分位数回归区间未校准（覆盖率64.2%<目标80%），不宜直接用于运营决策。
- 未来方向：持续采集更大样本、纳入成交实价、补充GIS/时间序列特征、采用Conformal Prediction等方法对区间重校准，并探索全国租赁交易注册库建设。

## 研究启发与可借鉴点
- 将“KFold Target Encoding”作为高基数地理类别的标准做法，可与本团队涉及地区编码、门店/城市层级特征的工作复用，避免泄漏与过拟合。
- 对明显右偏的价格型目标先做log(1+y)变换、在反变换尺度报告业务指标（MAE/RMSE），是可推广的稳健实践。
- 同一模型同时报告Gain与SHAP重要性，并重点解释二者分歧来源（如连续型编码变量在树分裂中“分散贡献”），可作为可解释性分析模板。
- 在提供点预测的同时给出分位数区间并报告Empirical Coverage vs Nominal Coverage，有助于后续工作建立“不确定性可用性”基线。
- 与Truong等（2020）对比显示：缺失微观GIS变量会形成约3-4个百分点R²差距；本团队若引入距离类/POI类特征，有望在同类任务中获得进一步跃升。

## 关键术语表
- **Hedonic pricing**：把商品价格拆解为各属性隐含价值之和的定价理论框架。
- **KFold Target Encoding**：在每折训练集内部计算类别目标统计量进行编码，防止测试信息泄漏。
- **XGBoost gain**：特征在所有分裂节点上带来损失下降的平均贡献占比。
- **SHAP（TreeSHAP）**：基于Shapley值的加法归因方法，可为每条样本提供局部贡献分解。
- **Quantile regression**：直接对目标分位数建模，用于生成预测区间与不确定性估计。
- **Empirical coverage**：真实标签落入预测区间的实际比例，用于检验区间校准程度。
- **Luxury score / kw score**：分别基于硬件高档属性与文本关键词权重的复合信号。

## 可复现要素
- 数据集：dataset.csv，CC BY 4.0，作者处可申请获取（论文声明）。
- 代码/笔记本：完整pipeline Jupyter notebook可向作者申请获取。
- 关键超参：
  - RF：n_estimators=300，max_depth=20，min_samples_leaf=2
  - XGBoost baseline：n_estimators=1000，learning_rate=0.03，max_depth=6
  - Optuna调参范围涵盖：树数量、学习率、最大深度、观测/特征采样率、L1/L2正则、叶节点最小权重
- 软件环境：Python 3.12.3；pandas 3.0.3；numpy 2.4.6；scikit-learn 1.9.0；XGBoost 3.3.0；LightGBM 4.6.0；Optuna 4.9.0；SHAP 0.52.0
- 评估设置：80/20单划分+5折CV；目标变换log(1+price)，指标在FCFA反变换尺度计算
