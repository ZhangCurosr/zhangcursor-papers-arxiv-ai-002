---
title: "Predicting-Residential-Rents-in-Dakar-Using-Machine-Learning"
source: https://arxiv.org/pdf/2608.30865v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:43:05"
field: "面向新兴市场的房地产价格预测与可解释AI"
keywords: ["rent prediction", "XGBoost", "SHAP interpretability", "quantile regression", "African real estate", "target encoding", "hedonic pricing", "Dakar housing market"]
innovations: ["构建达喀尔首个系统化住宅租金ML数据集并完整公开预处理流程", "系统对比XGBoost Gain与SHAP重要性，揭示编码后连续变量（location）被Gain低估的方法学洞见", "在非洲城市租金场景下引入分位数回归量化预测不确定性，识别校准缺口"]
benchmarks: ["Linear Regression baseline", "Random Forest", "XGBoost tuned (Optuna)", "LightGBM tuned (Optuna)", "5-fold cross-validation R²"]
---

# 论文速读：Predicting-Residential-Rents-in-Dakar-Using-Machine-Learning

## 一句话总结
本文首次构建了达喀尔（Senegal）住宅租赁市场的系统性机器学习预测框架，使用 XGBoost/LightGBM 在 1,507 条房源数据上达到 R²=0.847，并结合 SHAP 可解释性与分位数回归不确定性量化，填补了西非城市租金研究的空白。

## 研究问题与动机
1. 达喀尔租赁市场高度活跃（54.4% 家庭为租客），但缺乏系统性的量化租金研究，市场价格机制不透明。
2. 房产平台价格多为挂牌价（asking price）而非实际成交价，存在信息不对称，难以客观评估租金价值。
3. 非洲房地产市场的机器学习研究稀少，现有 hedonic 模型研究集中在拉各斯、坎帕拉等地，达喀尔长期缺位。
4. 尚无研究将租金预测、模型优化、SHAP 可解释性与不确定性量化（预测区间）在达喀尔语境下进行系统性整合。

## 核心贡献（创新点）
1. **构建达喀尔首个系统化租赁数据集**：通过自动化爬虫收集 1,654 条原始房源，经清洗得到 1,507 条标准化数据（24 个 locality、33 个变量），相比以往非洲研究样本量适中且覆盖多元社区。
2. **Optuna 贝叶斯优化的 XGBoost/LightGBM 预测框架**：在房价预测中超越线性回归与随机森林，R² 提升至 0.847（单切分），较线性回归提升 16.8 个 R² 点，凸显非线性关系对西非租金定价的关键作用。
3. **揭示 Gain vs. SHAP 的重要性分歧**：首次系统比较 XGBoost native gain 与 SHAP 在房产定价中的差异，发现 location 在 SHAP 中排名第 2（0.155），而在 gain 中仅排第 6（3.4%），证明单一重要性度量可能低估编码后连续变量的实际影响力。
4. **引入分位数回归量化预测不确定性**：训练 q10/q50/q90 三个 XGBoost 模型生成 80% 名义覆盖率的预测区间，实证覆盖率 64.2%，识别出校准不足的问题并为后续 conformal prediction 奠定基础。

## 方法详解
- **数据预处理**：七步流水线——本地化过滤、价格有效性检验、住宅用途筛选、类型归一化为 F2/F3/F4/F5/F6/Studio 六类、排除短租、130+ 地名标准化为 24 个地理类别、异常值/缺失值/冗余变量处理。
- **特征工程**：构造 4 个衍生变量：① `luxury_score`（泳池、海景、顶层、高档、电梯、车库、空调 7 项加权求和）；② `nbre_chambre²` 捕捉卧室数非线性效应；③ `ratio_sdb_ch` 卫浴/卧室比作为舒适度代理；④ `kw_score` 从描述中提取 18 个高档关键词加权得分。
- **编码策略**：33 个特征分四块输入 ColumnTransformer——数值块（中位数填充+StandardScaler）、类别块（One-Hot）、位置块（KFold Target Encoding）、布尔块（众数填充）。
- **KFold Target Encoding 防泄漏**：对 location 的编码仅在训练集 folds 内计算中位数，测试集出现的新 locality 使用整体训练集中位数，公式：$\operatorname{enc}(\text{neighborhood}_k) = \operatorname{median}\{y_i : \text{neighborhood}_i=k, i \in \text{train}\}$。
- **目标变换**：租金呈强右偏（skewness=1.55），应用 $y = \log(1+\text{price})$ 变换，预测后逆变换 $\hat{\text{price}} = \exp(\hat{y}) - 1$。
- **模型对比**：Linear Regression（基准）、Random Forest（300棵、depth=20）、XGBoost baseline（1000棵树、lr=0.03、max_depth=6）、XGBoost tuned、LightGBM tuned。
- **超参优化**：Optuna + TPE 算法搜索 8–9 个超参（树数、学习率、深度、采样率、L1/L2正则、最小叶子权重）。
- **可解释性**：① XGBoost gain importance（分裂损失平均下降占比）；② TreeSHAP（局部加性分解，base value=13.44，对应约 686,870 FCFA）。
- **分位数回归**：XGBoost `reg:quantileerror` 目标，分别训练 q0.10/q0.50/q0.90 模型，q10–q90 区间名义覆盖 80%。

## 实验与结果
- **数据集规模**：最终 1,507 条记录、33 个变量，覆盖 24 个 locality；中位租金 700,000 FCFA，均值 950,411 FCFA，标准差 796,534 FCFA；F4 户型占 53.4%，F3 占 23.6%；前五大社区（Almadies 19.5%、Ouakam 11.9%、Mermoz 10.4%、Point E 7.2%、Fann 6.6%）占总量 55.6%。
- **单切分 80/20 结果**：

| 模型 | MAE (FCFA) | RMSE (FCFA) | R² |
|---|---|---|---|
| Linear Regression | 257,147 | 470,262 | 0.679 |
| Random Forest | 222,201 | 363,525 | 0.808 |
| XGBoost baseline | 221,405 | 350,685 | 0.821 |
| **XGBoost tuned** | **210,902** | **324,195** | **0.847** |
| LightGBM tuned | 215,259 | 329,978 | 0.842 |

- **5 折交叉验证（log 尺度）**：XGBoost tuned 平均 R²=0.852（Std=0.011），LightGBM tuned 平均 R²=0.851（Std=0.013）。
- **5 折交叉验证（FCFA 逆变换尺度）**：XGBoost tuned 平均 R²=0.782（Std=0.034），平均 RMSE=369,744 FCFA（Std=26,230）。
- **最强结果**：XGBoost tuned 在单切分达到 R²=0.847，较线性回归提升 24.7% 的方差解释量。
- **特征重要性**：Gain 排名 TOP7：luxury_score（23.7%）、nbre_chambre（14.4%）、nbre_chambre²（12.3%）、vue_mer（7.7%）、kw_score（7.3%）、location_encoded（3.4%）、parking（2.9%）。SHAP 排名 TOP6：luxury_score（0.163）、location_encoded（0.155）、kw_score（0.132）、surface（0.107）、nbre_chambre（0.075）、vue_mer（0.072）。
- **预测区间**：q10–q90 名义覆盖 80%，实证覆盖率仅 64.2%，区间中位宽度 376,376 FCFA，未校准。

## 相关工作脉络
1. **Rosen (1974) Hedonic Pricing Theory**：租金特征定价理论奠基，本文在其框架下验证 Dakar 市场的价格形成机制。
2. **Abidoye & Chan (2018) 拉各斯房地产 hedonic 估值**：非洲最早的系统性实证研究之一，本文在其方法论基础上延伸至西非法语区并引入 ML 模型。
3. **Irumba (2015) 坎帕拉土地产权与房价研究**：关注土地制度对价格的影响，本文则聚焦市场化租赁信息的 ML 挖掘。
4. **Samb et al. (2025) 大贾富鲁短租价格建模**：与本文最相近的 West African 参考工作，同样使用树模型+SHAP，但本文扩展至长租市场并增加了分位数回归不确定性量化。
5. **Truong et al. (2020) 澳大利亚住房价格预测**：R²=0.88，拥有精细地理变量（学校/交通/绿地距离），本文对比指出缺少精确坐标是 3–4 个 R² 点差距的主因。
6. **Pargent et al. (2022) Regularized Target Encoding**：高基数特征编码的理论基础，本文采用其 KFold 变体（KFoldTargetEncoder）实现防泄漏编码。
7. **Lundberg & Lee (2017) SHAP**：Cooperative game theory 解释框架，本文应用 TreeSHAP 实现高效局部加性分解，并与 native gain 形成方法学对照。

## 局限性与未来方向
- **样本选择偏差**：数据来自在线房产平台，高估高档社区（Almadies/Ouakam 等）的代表性，大量非正式渠道租赁被遗漏。
- **挂牌价而非成交价**：Dakar 市场普遍存在谈判，实际成交价可能低于挂牌价，模型预测的是挂牌价水平而非实付租金。
- **地理覆盖不均**：24 个 locality 中前五大社区占 55.6%，偏远地区样本不足，模型泛化能力受限。
- **缺少精确地理坐标**：仅有 locality 级标签，无法纳入距离城市中心/交通/海滩的服务半径变量。
- **预测区间未校准**：64.2% 实证覆盖率远低于 80% 名义目标，分位数模型输出偏窄，需 conformal prediction 等后处理校准。
- **未来方向**：扩充多源数据（房产中介数据库、政府登记）、加入时空特征、追踪实际成交价、采用 conformal prediction 校准区间、建立国家租赁交易注册系统。

## 研究启发与可借鉴点
1. **KFold Target Encoding 的标准实践**：对高基数类别变量（如 24 个 locality），在 scikit-learn 管道内以 transformer 形式实现 KFold 目标编码，避免 naive encoding 导致的数据泄漏，可直接复用于任何含地理类别特征的任务。
2. **Gain vs. SHAP 的互补诊断**：本文揭示同一变量在不同重要性度量下排序可能逆转（location_encoded），提示团队在进行树模型可解释性分析时应同时报告两种度量，避免单一指标误导决策。
3. **对数变换+逆变换评估的统一范式**：目标变量强右偏时先做 $\log(1+y)$ 训练，评估时再逆变换到原始尺度计算 MAE/RMSE/R²，保证业务可解释性与模型训练稳定性的兼顾。
4. **分位数回归作为不确定性基线**：用 `reg:quantileerror` 快速获得预测区间（无需 ensemble 或 bootstrap），尽管校准有待改进，但可作为后续引入 conformal prediction 的第一步。
5. **综合 feature engineering 思路**：基于业务域知识构造复合得分（luxury_score、kw_score）和交互项（ratio_sdb_ch），而非依赖原始特征，这对数据质量有限的非洲市场尤为关键，可迁移至其他新兴市场定价研究。

## 关键术语表
- **Hedonic Pricing Theory**：将商品价格分解为其各项特征隐含价值的定价理论，广泛应用于房地产经济学。
- **KFold Target Encoding**：在 K 折交叉验证框架内仅用训练 fold 数据计算类别编码，防止测试集信息泄漏到训练过程。
- **XGBoost Gain Importance**：基于决策树分裂时损失函数下降量的平均值来衡量特征重要性，是 XGBoost 原生指标。
- **SHAP (SHapley Additive exPlanations)**：基于合作博弈论 Shapley 值的方法，为每个特征的每个预测提供局部加性贡献分解。
- **TreeSHAP**：针对树模型的 SHAP 高效精确计算算法，时间复杂度优于暴力枚举。
- **Quantile Regression**：直接建模目标变量条件分位数（而非均值）的回归方法，可输出预测区间。
- **Conformal Prediction**：一种后处理的预测区间校准方法，在有限交换性假设下提供覆盖率保证。
- **FCFA (West African CFA franc)**：西非经济货币联盟法定货币，1 EUR ≈ 655.957 FCFA，论文中租金金额单位。

## 可复现要素
- **数据集**：dataset.csv，CC BY 4.0 许可，可联系作者获取。
- **代码**：完整 Jupyter notebook 可联系作者获取（论文未上传公开仓库）。
- **关键超参**：XGBoost tuned / LightGBM tuned 经 Optuna（TPE 算法）搜索 8–9 个超参；具体最优值论文未列出表格。
- **随机种子**：论文未提及。
- **环境**：Python 3.12.3, pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, XGBoost 3.3.0, LightGBM 4.6.0, Optuna 4.9.0, SHAP 0.52.0。
