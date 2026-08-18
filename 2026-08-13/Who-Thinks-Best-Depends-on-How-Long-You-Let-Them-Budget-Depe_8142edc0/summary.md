---
title: "Who-Thinks-Best-Depends-on-How-Long-You-Let-Them-Budget-Depe"
source: https://arxiv.org/pdf/2608.12150v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:41:02"
field: "大语言模型评估与方法论"
keywords: ["LLM evaluation", "test-time compute scaling", "model routing", "overthinking", "budget-dependent ranking", "model complementarity"]
innovations: ["首次系统揭示LLM模型排名随token budget变化的反转现象", "提出item-level行为分类学量化overthinking的模型特异性", "设计budget-aware routing捕获14.1% oracle gap"]
benchmarks: ["GSM8K", "MATH-500", "GPQA-Diamond"]
---

# 论文速读：Who-Thinks-Best-Depends-on-How-Long-You-Let-Them-Budget-Dep

## 一句话总结
本文系统研究大语言模型推理评估中**token生成预算对模型排名的影响**，发现模型性能排名随预算变化而显著反转（3-19%题目呈现"过度思考"非单调行为），并据此提出预算感知的模型路由方法，在跨域测试中捕获14.1%的oracle差距。

## 研究问题与动机
- **核心问题**：现有LLM评测假设模型排名在推理条件下稳定不变，但这一假设在变长生成场景下是否成立？
- **现有方法不足**：
  1. Leaderboard和benchmark比较忽略token budget变化对排名的影响，导致部署场景与实际评估脱节
  2. 现有test-time compute scaling研究仅关注单一模型的scaling曲线，未比较多模型交叉情况
  3. Overthinking文献仅定性描述额外推理损害性能，未量化其在item-level的系统性分布及模型特异性
  4. 现有routing系统 treats routing as budget-agnostic，未将token budget作为显式路由信号

## 核心贡献（创新点）
1. **首次系统性揭示budget-dependent ranking现象**：通过7个budget级别×4模型×3基准的56,476次推理，证明模型排名随预算显著反转（所有基准均观测到，McNemar p<0.01）
   - *与已有工作的本质区别*：不同于Test-time compute scaling研究仅关注单模型scaling，本文揭示多模型scaling曲线会交叉且排名反转

2. **Item-level行为分类学**：将模型-题目对按budget变化下的正确率轨迹分为四类（always-correct/monotone-increasing/non-monotone/always-wrong），发现non-monotone（过度思考）行为占3-25.8%且高度模型特异（cross-model overlap仅6-14%）
   - *与已有工作的本质区别*：overthinking被证实为模型特异性现象而非题目固有属性，无法通过删除"问题题目"缓解

3. **Oracle gap dynamics量化**：揭示模型互补性在受限budget下最显著（GPQA上oracle超最佳单模型+27.8 pp），且互补性随budget增加收敛但不消失（Jaccard相似度从0.048升至0.741）
   - *与已有工作的本质区别*：首次系统量化多模型互补性与budget的动态关系，而非静态ensemble效果

4. **Budget-aware routing proof-of-concept**：设计XGBoost路由器，以budget为显式特征路由至最优模型，跨域评估达22.9%准确率，捕获14.1% oracle差距
   - *与已有工作的本质区别*：首次将token budget作为显式路由信号，SHAP分析证实budget特征重要性为文本特征的6.1倍

## 方法详解
**评估框架**：
- **模型**：4个open-weight推理模型（LLaMA-3 8B, Qwen-3 32B, LLaMA-3.3 70B, GPT-OSS 20B），greedy decoding (T=0)
- **基准**：3个推理benchmark（GSM8K 1,319题/MATH-500 500题/GPQA-Diamond 198题）
- **预算**：7个token budget级别b∈{64, 128, 256, 512, 1024, 2048, 4096}
- **评分**：regex解析最终答案，exact match二元正确性

**三层截断控制分析**：
1. **All-items**：标准评分（截断=错误）
2. **Stop-only**：仅保留完成生成的题目（每模型独立子集）
3. **Common non-truncated**：四模型均完成的配对比较（N<30视为不可靠）

**行为分类定义**：
对模型m和题目i，观察7预算下的正确率轨迹$\mathbf{c}_{m,i}\in\{0,1\}^7$，分为：
- Always-correct：全预算正确
- Monotone-increasing：0→1且不回退
- Non-monotone：存在1→0回退（overthinking）
- Always-wrong：全预算错误

**路由方法**：
- 为每个模型训练XGBoost二分类器$f_m(x,b):\{0,1\}$预测正确概率
- 特征：$\log_2(b)$ + 表面文本统计（字符/词数、特殊字符数、LaTeX存在、词熵、最大数值）+ 20维PCA降维的sentence embedding（all-MiniLM-L6-v2）
- 推理时：$m^* = \arg\max_m f_m(x,b)$

## 实验与结果
**数据集**：GSM8K（小学奥数）、MATH-500（竞赛级数学）、GPQA-Diamond（研究生级科学，IRT easiness均值0.172）

**基线方法**：Random、Largest-Always（GPT-OSS 20B）、Best-Overall、Best-Per-Budget、Oracle

**主要结果**：
1. **排名反转**：所有基准均观测到best model随budget变化
   - GSM8K：b=256时LLaMA-3.3 70B领先(62.4%)→b=4096时GPT-OSS 20B领先(94.9%, p<0.001)
   - GPQA：LLaMA-3 8B在b=256-512领先→b=1024被LLaMA-3.3 70B超越→b=4096被GPT-OSS 20B追平

2. **非单调行为率**：3.6%（GPT-OSS 20B/GSM8K）至25.8%（LLaMA-3 8B/GPQA）；filter截断后仍保持19.1%（LLaMA-3 8B/GPQA）

3. **Oracle gap**：GSM8K峰值+16.9 pp（b=256）；MATH-500单调增至+12.8 pp（b=4096）；GPQA达+27.8 pp（b=4096）

4. **路由效果**（跨域：train GSM8K+MATH-500, test GPQA）：
   - Router-Scoring准确率22.9%，vs Best-Per-Budget的20.3%，提升+2.67 pp（95% CI [0.94, 4.40]）
   - 捕获14.1% oracle gap（39.2%-20.3%=18.9pp差距）
   - 判别子集（520题）上提升+7.12 pp

5. **Budget特征重要性**：SHAP值2.21，为第二特征(LaTeX存在)的6.1倍

6. **消融**：Within-domain budget特征贡献+1.6至+5.7 pp；Cross-domain去除budget反而提升1.2 pp（domain-specific性）

## 相关工作脉络
1. **Test-time compute scaling [3,4]**：研究单模型compute分配对性能的影响；本文扩展至多模型同时评估，揭示scaling曲线交叉与排名反转
2. **Overthinking文献 [7,8]**：定性观察o1-like模型过度推理失败；本文item-level量化，证明overthinking为模型特异性而非题目固有属性
3. **Model routing [9,10,11]**：基于输入特征选优模型，忽略budget；本文首次将token budget作为显式路由信号
4. **Evaluation methodology [12,13,14]**：关注contamination/saturation/prompt sensitivity；本文揭示budget sensitivity为新的脆弱性维度
5. **Chain-of-thought [6]**：隐式增加token budget；本文显式控制budget，量化其对各模型的系统性影响差异
6. **Budget-forcing机制 [5]**：RL控制reasoning length；本文揭示不同模型对budget的敏感性差异，为force机制设计提供依据

## 局限性与未来方向
- **模型覆盖有限**：仅4个模型，未包含dedicated reasoning models（o1/DeepSeek-R1/QwQ等），因其dual-stream架构改变max_tokens语义
- **预算粒度粗糙**：7个对数间距budget，更细粒度可能揭示额外结构
- **路由特征局限**：仅surface-level text stats和静态embedding，未利用logit entropy/hidden-state等model-internal signals
- **任务类型局限**：仅closed-form reasoning benchmarks，未测试open-ended任务
- **路由性能瓶颈**：19 pp差距于oracle，说明大部分互补性未被利用
- **未来方向**：扩展至reasoning models、探索budget-forcing技术替代截断、开发domain-agnostic的budget-generalizable路由

## 研究启发与可借鉴点
1. **评测协议设计**：benchmark应报告多budget级别下的accuracy，而非单一数字；建议 adoption of budget-conditioned evaluation protocols
2. **方法复用**：三层截断控制分析（all-items/stop-only/common non-truncated）可迁移至任何变长生成评估场景
3. **实验设计借鉴**：item-level trajectory分析（7预算正确率序列）可揭示模型能力边界；non-monotone行为检测需filter truncation artifacts
4. **创新机会**：
   - 将budget-aware routing与ensemble方法结合，逼近oracle gap
   - 探索reasoning models的budget semantics重新定义（需区分internal thinking tokens与visible output tokens）
   - 开发domain-transferable budget features，缓解跨域路由性能下降问题
5. **计算效率洞察**：moderate budget（b=512-2048）下model complementarity最大，可作为deployment的sweet spot

## 关键术语表
- **Token generation budget (max_tokens)**：模型推理时允许生成的最大token数，决定输出长度上限
- **Non-monotone behavior (Overthinking)**：模型在更高budget下正确率下降的现象，即过度推理导致初始正确答案被覆盖
- **Oracle gap**：理想per-item最优选择集成与单模型最佳表现的准确率差值，衡量model complementarity
- **Budget-conditioned evaluation**：在多个token budget级别下分别报告benchmark性能，而非单一数值
- **Three-tier truncation analysis**：三层控制（all-items/stop-only/common non-truncated）分离genuine reasoning effects与truncation artifacts
- **Cross-model overlap**：多个模型均标记为non-monotone的题目比例，衡量overthinking的model-specific程度
- **Jaccard similarity**：两模型正确题目集合的交集/并集，量化model agreement随budget的变化
- **Budget-aware routing**：以token budget为显式特征的模型选择系统，根据题目和budget预测最优模型

## 可复现要素
- **数据集**：GSM8K、MATH-500、GPQA-Diamond（均为公开benchmark）
- **代码/权重**：论文未提及开源代码；模型为open-weight（Meta LLaMA、Qwen、GPT-OSS via Together.ai API）
- **关键超参**：T=0（greedy decoding）；budget∈{64,128,256,512,1024,2048,4096}；XGBoost分类器；20维PCA embeddings（all-MiniLM-L6-v2）
- **硬件**：云API endpoints；路由实验用single CPU
