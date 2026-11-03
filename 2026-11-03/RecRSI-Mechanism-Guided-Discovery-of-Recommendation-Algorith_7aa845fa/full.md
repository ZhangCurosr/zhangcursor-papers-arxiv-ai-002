# RecRSI: Mechanism-Guided Discovery of Recommendation Algorithms

Tingran Gan dynamisg@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Tao Li   
litao118@meituan.com   
Meituan   
Beijing, China   
Weiwen Liu<sup>†</sup>   
wwliu@sjtu.edu.cn   
Shanghai Jiao Tong University   
Shanghai, China

Zhouziyue Lu zhouziyue\_lu@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Yang Gao   
gaoyang94@meituan.com   
Meituan   
Beijing, China

Jianghao Lin<sup>†</sup> linjianghao@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Yong Yu<sup>†</sup> yyu@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Dian Jin 2430XH10019@m.fudan.edu.cn Fudan University Shanghai, China

Ming Liu   
liuming66@meituan.com   
Meituan   
Beijing, China

Weinan Zhang<sup>†</sup> wnzhang@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

## Abstract

Improving recommendation algorithms for the Web requires coordinating interacting design choices and learning from experiments. We present RecRSI, an LLM-based research framework that uses mechanism hypotheses to organize algorithm discovery. Each hypothesis links a proposed mechanism change to a reference design, a rationale, and an expected observation. MechSpace derives implementable primitives and composition rules from the task source; Research Line uses this space and prior evidence to develop and test hypotheses; Mechanism Feedback carries outcomes and their design context into subsequent research. This organization makes each experiment inform which mechanisms to develop, revise, or combine next. Across four initial recommenders and four datasets, RecRSI yields average relative improvements of 8.48% in NDCG@5 and 6.88% in HR@5 over the initial algorithms, and 2.60% and 2.20%, respectively, over the strongest competing result for each setting and metric. Component and process studies link these gains to coordinated mechanism choices and evidence-guided follow-up. Our code is publicly available at https://github.com/DynamisG210/RecRSI.

## CCS Concepts

• Information systems → Recommender systems.

## Keywords

Recommender Systems, Automated Algorithm Discovery, Research Agents, Mechanism Hypotheses

## 1 Introduction

Recommendation algorithms shape how users discover content and products across the Web, from personalized feeds to online marketplaces. Developing these ranking algorithms across diferent interaction settings requires coordinating representations, scoring functions, learning objectives, and optimization procedures, whose efectiveness depends on the task and on one another [18]. With limited training and evaluation budgets, automated research must identify worthwhile changes and use their outcomes to guide further discovery.

Large language model (LLM) agents increasingly automate algorithm design and experimentation. Program-evolution systems search over evaluated candidates [10, 21], while research agents develop and test hypotheses [6, 16, 31]. In recommendation, Self-EvolveRec couples user simulation with model diagnosis, while RecHarness combines direction routing with accumulated experimental skills [9, 12]. Sustained algorithm discovery poses two challenges for these systems.

Interdependent mechanisms obscure which changes to test. Studies of recommendation architectures and interaction search show that useful component choices depend on their task and surrounding design [4, 23, 28]. A scoring change, for example, afects both prediction and the scores consumed by the training objective. Research must decide which mechanisms to change together and how to implement their interaction. Distinct code edits may revisit similar mechanisms; a promising combination may span several components. Repeatedly reconstructing these relationships during program search makes targeted experimentation dificult within a limited budget.

Context-dependent outcomes are dificult to reuse. Reflection-based agents reuse feedback across trials [27, 33], and

![](images/aaaa249219b1da7e3082fd8e028c3db9ef50bcabc9b082055d413813733fb92a.jpg)  
Figure 1: Research organization and algorithm discovery. (a) Candidate-level evolution organizes iteration around programs; RecRSI uses mechanism hypotheses to connect proposed changes, their rationale, and experimental evidence for the next design. Both workflows generate, evaluate, and refine algorithms. (b) Mean best-so-far validation NDCG@5 gain over the initial recommender on the same eight tasks as Figure 3(a,b). Search repeats are averaged within each task, then tasks receive equal weight.

CausalEvolve emphasizes organizing past experience to guide further evolution [1]. In recommendation, an evaluation measures a complete algorithm under a particular mechanism combination. Its value for the next experiment depends on the tested hypothesis, reference design, and unchanged components. A failed alternative may challenge one explanation while leaving an earlier design worth pursuing. Research must connect the observation to that design context to decide what to refine, replace, or recombine.

A mechanism is a design with a specific computational or learning role, such as sequence aggregation, matching geometry, or a ranking objective. A mechanism hypothesis links a proposed change to a reference design, a rationale, and an expected observation. It makes an experiment both a concrete algorithmic intervention and a research question whose outcome can inform the next design.

We introduce RecRSI, a framework that uses mechanism hypotheses as the organizing unit of recommendation algorithm discovery. MechSpace derives task-specific primitives and composition rules, making proposed interventions implementable. Research Line develops and tests hypotheses through complementary roles for composition, refinement, explanation testing, and exploration. Mechanism Feedback interprets outcomes in the context of the tested hypotheses and designs to inform the next research decision. A new hypothesis can build on a successful earlier design and use an unsuccessful experiment to choose its next intervention. Section 3.4 demonstrates this process in a real research run. Figure 1 links this hypothesis-centered organization to algorithm discovery progress across eight tasks. The name reflects a recursive self-improvement perspective [36]: evaluated recommender designs and their evidence support successive algorithm improvements.

Experiments cover four initial recommenders and four datasets. Across the 16 model–dataset settings, RecRSI improves NDCG@5 and HR@5 by an average of 8.48% and 6.88% over the initial algorithms, and 2.60% and 2.20% over the strongest competing result for each setting and metric. Component and process studies trace these gains to coordinated mechanism choices, history-informed follow-up, and discoveries beyond the initial search horizon. Our contributions are:

• A mechanism-guided discovery framework. RecRSI makes mechanism hypotheses the research unit connecting algorithm design, experimentation, and follow-up.

• A method for cumulative mechanism research. Taskderived interventions and hypothesis-linked feedback turn earlier successes and failures into concrete follow-up experiments.

• Evidence of discovery quality and research value. Evaluation across 16 settings establishes ranking gains; component, history, and continuation studies connect them to the research decisions and mechanism combinations that produce them.

## 2 Related Work

## 2.1 Automated Algorithm Discovery and Program Evolution

Automatic algorithm design searches over executable computations, from AutoML-Zero’s mathematical operations to FunSearch’s LLM-generated programs [22, 24]. Evolution of Heuristics (EoH) co-evolves ideas and code [14]; AlphaEvolve, OpenEvolve, and ShinkaEvolve develop populations of evaluated programs with complementary selection and diversity mechanisms [10, 21, 26]. The Darwin Gödel Machine extends this loop to the coding agent’s own implementation [36].

Self-Refine iterates on generated outputs, Reflexion stores verbal feedback for subsequent trials, and ReEvo uses short- and longterm reflections in evolution [17, 27, 33]. DeepEvolve incorporates literature-informed hypotheses; CausalEvolve organizes outcomeand procedure-level factors in a causal scratchpad and uses unexpected outcomes to generate follow-up hypotheses [1, 15]. The AI Scientist automates research from ideas through experiments and manuscripts; its successor and AIDE organize experimentation through tree search [6, 16, 31]. The autoresearch software provides a compact code–train–evaluate loop with a fixed training-time budget [8]. Broader reviews examine memory, skills, protocols, and harnesses as externalized agent capabilities [38], and analyze agentic data generation through accuracy, complexity, and diversity [35].

AgentNAS builds interchangeable module slots; FairNAD structures literature-derived design attributes [5, 25]. MechSpace derives task-specific mechanism specifications and composition rules, connecting a proposed research question to an implementable intervention.

## 2.2 Automated Research for Recommender Systems

RecBole provides unified implementations and protocols [37]; systematic studies underline the importance of tuning and comparable evaluations [2, 23]. AutoFIS selects feature interactions, and AutoCTR searches their architectural combinations [13, 28].

LLMs support multiple stages of recommendation, from feature modeling and ranking to pipeline control [11]. Self-EvolveRec couples simulated-user critiques with model diagnostics [9]. RecHarness combines bandit routing with accumulated Experiment Skills, grouped trials, validation-based reward updates, and structural jumps followed by local retuning [12]. EvoRec distills methodologies from experiment memory; AgentX-Model coordinates reproduction, follow-up, composition, and diagnosis [20, 32]. RecRSI organizes recommendation research around a mechanism hypothesis specifying an implementable intervention, its reference design, a rationale, and an expected observation. This hypothesis guides candidate construction and links the resulting evidence to the mechanisms changed and held fixed, informing which designs to refine or recombine next.

## 3 Methodology

RecRSI carries a mechanism hypothesis through formulation, implementation, evaluation, and follow-up. Together, its three stages

connect a research question to an executable intervention and turn the outcome into evidence for the next decision (Figure 2).

## 3.1 Task and Mechanism Hypotheses

Given an initial recommendation algorithm $A _ { 0 } ,$ a task environment $\mathcal { T } ,$ , and a budget �, RecRSI searches for an improved algorithm implementation. Researchers supply the initial source, data splits, editable interfaces, and a fixed training–evaluation protocol Π:

$$
\theta _ { A } = \mathrm { T r a i n } _ { \Pi } ( A ; D _ { \mathrm { t r } } ) , \qquad q ( A ) = \mathrm { E v a l } _ { \Pi } ( A , \theta _ { A } ; D _ { \mathrm { v a l } } ) .\tag{1}
$$

The inner process learns parameters; the outer process changes the algorithm under limits on experimental opportunities and resources. RecRSI returns the highest-scoring evaluated algorithm, including $A _ { 0 }$ among the candidates.

A composition � describes the mechanisms used together and their relationships. A mechanism hypothesis is $h _ { t } = ( C _ { t } ^ { \mathrm { r e f } } , C _ { t } , \rho _ { t } ,$ $\epsilon _ { t } ) \mathrm { : }$ : the reference design, proposed composition, research rationale, and expected observation. The transition from $C _ { t } ^ { \mathrm { r e f } }$ to $C _ { t }$ specifies which mechanisms stay unchanged and which are added, replaced, or removed; $\rho _ { t }$ and $\epsilon _ { t }$ state why this intervention merits an experiment and what it is expected to reveal. The hypothesis guides implementation of candidate $A _ { t } ; \widehat { C } _ { t }$ denotes the composition described in the implemented candidate’s record.

The research state $\mathcal { R } _ { t } ~ = ~ ( \mathcal { H } _ { t } , \mathcal { U } _ { t } , A _ { t } ^ { \star } )$ contains hypothesis– experiment records, open questions, and the current best algorithm. Records store both the construction references and the observations motivating follow-up hypotheses.

## 3.2 MechSpace: A Design Space for Mechanism Hypotheses

MechSpace converts the initial source and task interfaces into a mechanism space $S _ { \mathcal { T } } = ( \mathcal { P _ { T } } , \Gamma _ { \mathcal { T } } )$ of primitives and composition rules. Each primitive connects a computational or learning efect to a source location, input and output types, parameters, and implementation requirements. Its efect path specifies how the changed computation participates in training or prediction.

Within the supplied edit scope and composition limits, an LLM generator proposes primitives, parameter settings, source anchors, and compatibility rules. An LLM reviewer examines coverage, redundancy, composability, and efect paths from the supplied source; a deterministic assembler validates schemas, mutation targets, and composition constraints to produce the registry and grammar. Efect-path assessment is source-based LLM review; candidate execution is checked separately during qualification. For example, the reviewer rejects an evaluation-only learned readout because the training loss does not use its new parameters. The MoRec/CDs space covers representation, scoring, objective, update, architecture, and sampling; it is built once and reused through task adapters.

Composing interventions. The grammar determines how primitives can be combined. In the MoRec/CDs instance, SELECT chooses one primitive, and COMPOSE combines two or three from distinct axes, subject to mutation-target constraints. The primitive score\_ cosine\_temp changes Model.predict\_score, shared by training and candidate evaluation. The primitive obj\_pairwise\_margin adds a ranking term through the same positive and negative scores:

![](images/0bb45d4d605ec67d7c22563c94dd6e0115b83c8d75edcc861e50051f79286bbc.jpg)  
Figure 2: Overview of RecRSI. MechSpace supplies the design space for mechanism hypotheses; Research Line develops them into candidates and experiments; Mechanism Feedback links outcomes to the tested hypotheses and designs. The shared research state carries this evidence into subsequent investigations.

$$
s ( u , i ) = \langle \bar { \bf z } _ { u } , \bar { \bf e } _ { i } \rangle / \tau ,
$$

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \frac { \lambda } { | \mathcal { V } | } \sum _ { j \in \mathcal { V } } \mathrm { s o f t p l u s } \Big ( m - \big ( s _ { j } ^ { + } - s _ { j } ^ { - } \big ) \Big ) .\tag{2}
$$

Here $\mathbf { z } _ { u }$ and $\mathbf { e } _ { i }$ are sequence and item representations, with $\bar { \textbf { v } } =$ $\mathbf { v } / ( \lVert \mathbf { v } \rVert _ { 2 } + \varepsilon ) . \mathcal { V }$ indexes non-padding supervised positions, $s _ { j } ^ { + }$ and $\boldsymbol { s } _ { j } ^ { - }$ are their positive and negative scores, and $\mathcal { L } _ { \mathrm { B C E } }$ sums the mean positive and negative BCE losses. This instance uses $\tau = m = 0 . 2 ,$ $\lambda = 0 . 5 ,$ and $\varepsilon = 1 0 ^ { - 8 }$ . Sharing the scores makes the interaction between scoring and learning explicit. Focal weighting ofers an alternative on the objective axis, while recent-history fusion modifies representations upstream of the scorer.

## 3.3 Research Line: Advancing Mechanism Hypotheses

Four complementary roles form a hypothesis pool Q from the mechanism space and research state (Algorithm 1). The Mechanism Composer proposes combinations with a joint rationale; the Lineage Refiner develops an existing direction; the Falsification Designer formulates contrasts between explanations; and the Frontier Architect identifies underexplored directions while considering its peers’ proposals.

From a hypothesis to a complete design. Selected primitives specify the intervention; the reference candidate specifies the surround ing design. A scoring hypothesis, for example, also identifies the objective to use or restore when revisiting an earlier branch. The task adapter resolves this complete composition against the mechanism space and editable interfaces. Feasibility checks enforce composition rules and reference links; candidate identities support duplicate filtering.

Algorithm 1 Mechanism-guided recommendation research   
Require: Initial algorithm $A _ { 0 } ,$ task T, budget �   
1: S ← MechSpace(� , T)   
2: R ← Initialize(� , T)   
3: while research budget remains do   
4: Q ← Propose(S, R, �)   
5: (ℎ, �) ← SelectFeasible(Q, R, �)   
6: if no eligible proposal then break   
7: (�, �, �b ) ← ImplementAndQualify(ℎ, �, T)   
8: if � = PASS then   
9: � ← AdmitAndEvaluate (�, �)   
10: else   
11: � ← QualificationFailure(�) ⊲ skip evaluation   
12: end if   
13: � ← Feedback(T, ℎ, �, �b ; R)   
14: Update R, current best, and remaining �   
15: end while   
16: return Current best algorithm in R

The roles adapt their hypotheses to the available evidence; a fixed selector allocates experimental opportunities. Within each opportunity block, it enforces role coverage at the opening positions and whenever the remaining positions are needed to cover available, unselected roles. Other choices follow a fixed role-priority order with stable tie-breaking, reserving opportunities for complementary research actions.

Table 1: Final test performance across four seed recommenders and four datasets.
<table><tr><td rowspan="2">Seed Recommender</td><td rowspan="2">Method</td><td colspan="2">CDs</td><td colspan="2">Electronics</td><td colspan="2">Office</td><td colspan="2">MovieLens-1M</td></tr><tr><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td></tr><tr><td rowspan="8">SASRec</td><td>Seed</td><td>0.3489</td><td>0.4572</td><td>0.3623</td><td>0.4860</td><td>0.2674</td><td>0.3980</td><td>0.5613</td><td>0.7176</td></tr><tr><td>RecHarness</td><td>0.3954</td><td>0.5126</td><td>0.3705</td><td>0.4947</td><td>0.2754</td><td>0.4080</td><td>0.5687</td><td>0.7241</td></tr><tr><td>DeepEvolve</td><td>0.3510</td><td>0.4605</td><td>0.3678</td><td>0.4932</td><td>0.2795</td><td>0.4099</td><td>0.5340</td><td>0.6917</td></tr><tr><td>OpenEvolve</td><td>0.3645</td><td>0.4841</td><td>0.3674</td><td>0.4920</td><td>0.2771</td><td>0.4139</td><td>0.5610</td><td>0.7188</td></tr><tr><td>ShinkaEvolve</td><td>0.3641</td><td>0.4841</td><td>0.3629</td><td>0.4871</td><td>0.2743</td><td>0.4100</td><td>0.5647</td><td>0.7212</td></tr><tr><td>Self-EvolveRec</td><td>0.3645</td><td>0.4806</td><td>0.3679</td><td>0.4918</td><td>0.2731</td><td>0.4055</td><td>0.5607</td><td>0.7171</td></tr><tr><td>RecRSI (Ours)</td><td>0.4329</td><td>0.5605</td><td>0.3730</td><td>0.5027</td><td>0.2811</td><td>0.4127</td><td>0.5678</td><td>0.7275</td></tr><tr><td>Improv. (%)</td><td>+9.50</td><td>+9.36</td><td>+0.67</td><td>+1.61</td><td>+0.58</td><td>-0.28</td><td>-0.16</td><td>+0.47</td></tr><tr><td rowspan="8">MoRec</td><td>Seed</td><td>0.3795</td><td>0.5014</td><td>0.3698</td><td>0.4949</td><td>0.2887</td><td>0.4258</td><td>0.5636</td><td>0.7209</td></tr><tr><td>RecHarness</td><td>0.4011</td><td>0.5283</td><td>0.3726</td><td>0.4982</td><td>0.2855</td><td>0.4240</td><td>0.5687</td><td>0.7264</td></tr><tr><td>DeepEvolve</td><td>0.3863</td><td>0.5104</td><td>0.3804</td><td>0.5053</td><td>0.2875</td><td>0.4259</td><td>0.5715</td><td>0.7267</td></tr><tr><td>OpenEvolve</td><td>0.3924</td><td>0.5203</td><td>0.3780</td><td>0.5033</td><td>0.2905</td><td>0.4291</td><td>0.5651</td><td>0.7234</td></tr><tr><td>ShinkaEvolve</td><td>0.3875</td><td>0.5133</td><td>0.3774</td><td>0.5026</td><td>0.2897</td><td>0.4270</td><td>0.5671</td><td>0.7240</td></tr><tr><td>Self-EvolveRec</td><td>0.3878</td><td>0.5134</td><td>0.3730</td><td>0.4982</td><td>0.2901</td><td>0.4277</td><td>0.5645</td><td>0.7212</td></tr><tr><td>RecRSI (Ours)</td><td>0.4151</td><td>0.5416</td><td>0.3938</td><td>0.5231</td><td>0.2902</td><td>0.4267</td><td>0.5708</td><td>0.7271</td></tr><tr><td>Improv. (%)</td><td>+3.50</td><td>+2.52</td><td>+3.51</td><td>+3.52</td><td>-0.11</td><td>-0.56</td><td>-0.12</td><td>+0.05</td></tr><tr><td rowspan="8">BERT4Rec</td><td>Seed</td><td>0.3286</td><td>0.4417</td><td>0.3881</td><td>0.5072</td><td>0.2423</td><td>0.3643</td><td>0.5313</td><td>0.6809</td></tr><tr><td>RecHarness</td><td>0.3220</td><td>0.4383</td><td>0.3959</td><td>0.5148</td><td>0.2363</td><td>0.3572</td><td>0.5476</td><td>0.6954</td></tr><tr><td>DeepEvolve</td><td>0.3334</td><td>0.4480</td><td>0.3905</td><td>0.5091</td><td>0.2374</td><td>0.3574</td><td>0.5403</td><td>0.6893</td></tr><tr><td>OpenEvolve</td><td>0.3286</td><td>0.4417</td><td>0.3881</td><td>0.5072</td><td>0.2423</td><td>0.3643</td><td>0.5344</td><td>0.6833</td></tr><tr><td>ShinkaEvolve</td><td>0.3281</td><td>0.4418</td><td>0.3932</td><td>0.5123</td><td>0.2451</td><td>0.3678</td><td>0.5425</td><td>0.6882</td></tr><tr><td>Self-EvolveRec</td><td>0.3070</td><td>0.4146</td><td>0.3807</td><td>0.4984</td><td>0.2169</td><td>0.3280</td><td>0.5464</td><td>0.6937</td></tr><tr><td>RecRSI (Ours)</td><td>0.3742</td><td>0.4940</td><td>0.4046</td><td>0.5233</td><td>0.2681</td><td>0.3963</td><td>0.5424</td><td>0.6905</td></tr><tr><td>Improv. (%)</td><td>+12.24</td><td>+10.27</td><td>+2.20</td><td>+1.66</td><td>+9.36</td><td>+7.76</td><td>-0.96</td><td>-0.70</td></tr><tr><td rowspan="8">NGCF</td><td>Seed</td><td>0.2919</td><td>0.4059</td><td>0.2642</td><td>0.3635</td><td>0.0867</td><td>0.1365</td><td>0.2771</td><td>0.6547</td></tr><tr><td>RecHarness</td><td>0.2916</td><td>0.4028</td><td>0.2996</td><td>0.4064</td><td>0.0916</td><td>0.1430</td><td>0.2778</td><td>0.6606</td></tr><tr><td>DeepEvolve</td><td>0.3217</td><td>0.4380</td><td>0.2875</td><td>0.3899</td><td>0.0959</td><td>0.1454</td><td>0.2883</td><td>0.6727</td></tr><tr><td>OpenEvolve</td><td>0.3353</td><td>0.4503</td><td>0.2890</td><td>0.3914</td><td>0.1011</td><td>0.1519</td><td>0.2939</td><td>0.6838</td></tr><tr><td>ShinkaEvolve</td><td>0.3349</td><td>0.4497</td><td>0.2873</td><td>0.3946</td><td>0.1022</td><td>0.1531</td><td>0.2939</td><td>0.6850</td></tr><tr><td>Self-EvolveRec</td><td>0.3189</td><td>0.4343</td><td>0.2725</td><td>0.3738</td><td>0.0950</td><td>0.1446</td><td>0.2871</td><td>0.6682</td></tr><tr><td>RecRSI (Ours)</td><td>0.3366</td><td>0.4514</td><td>0.2922</td><td>0.3936</td><td>0.1051</td><td>0.1566</td><td>0.2957</td><td>0.6863</td></tr><tr><td>Improv. (%)</td><td>+0.41</td><td>+0.24</td><td>-2.45</td><td>-3.15</td><td>+2.86</td><td>+2.23</td><td>+0.63</td><td>+0.20</td></tr></table>

Bold/underline: best/second best. Improv.: RecRSI’s relative gain over the strongest competing system, computed before rounding.

The Implementer receives the hypothesis specification, source, and interface requirements. Qualification checks interfaces and execution, with bounded repair for implementation failures. Candi dates that pass qualification and resource admission enter training and evaluation under Π. An unresolved qualification failure skips evaluation and records its status, diagnostics, and cost; the incumbent stays unchanged and research proceeds to the next eligible proposal under the remaining budget. These records and measured outcomes feed Mechanism Feedback.

## 3.4 Mechanism Feedback and Research State Update

Mechanism Feedback relates an outcome to the intervention and question that produced it. Each record links the hypothesis $h _ { t } ,$ implemented composition $\widehat { C } _ { t }$ , and observation $o _ { t }$ to the reference design, changed and unchanged mechanisms, performance diference, execution status, and cost. The original rationale, expected observation, and suggested tests remain available alongside the measurements.

Using evidence across experiments. The feedback record updates the shared state:

$$
\begin{array} { r } { \begin{array} { r l } & { e _ { t } = \mathrm { F e e d b a c k } ( \mathcal { T } , h _ { t } , \widehat { C } _ { t } , o _ { t } ; \mathcal { R } _ { t } ) , } \\ & { \mathcal { R } _ { t + 1 } = \mathrm { U p d a t e } ( \mathcal { R } _ { t } , e _ { t } ) . } \end{array} } \end{array}\tag{3}
$$

Outcomes update the experimental history and mechanism-level evidence; the current best changes when validation performance improves. The construction reference determines the base design; evidence from another experiment can motivate a new intervention. Role-specific inputs include the incumbent, latest result, recent mechanism comparisons, open questions, and role history, linking working explanations to measured outcomes.

An unsuccessful hypothesis guides the next discovery. In a MoRec/CDs search, round 3 asks whether the pairwise objective from round 2 helps through general emphasis on dificult examples or specifically through positive–negative separation. It replaces the auxiliary pairwise term with focal-weighted BCE; validation NDCG@5 falls from 0.3705 to 0.3438. The regression becomes part of the mechanism history, while the round 2 design remains the incumbent. All four roles use this observation and the incumbent in round 4. The selected Frontier Architect proposal restores round 2’s BCE and pairwise objective and tests cosine-temperature scoring, hypothesizing that reducing embedding-norm efects will improve ranking. The candidate implements Eq. 2 and reaches validation NDCG@5 of 0.4097. Round 2 supplies the construction reference; round 3 supplies evidence for changing direction.

## 4 Experiments

Experimental setup. We study algorithm discovery from four initial recommenders—SASRec [7], MoRec [34], BERT4Rec [29], and NGCF [30]—on Amazon CDs, Electronics, Ofice, and MovieLens-1M [3]. All systems share the same preprocessed data, splits, and evaluation protocol within each model–dataset setting (Appendix D). We compare RecRSI with the initial algorithms and five discovery systems: RecHarness, DeepEvolve, OpenEvolve, ShinkaE volve, and Self-EvolveRec [9, 10, 12, 15, 26].

We reproduce all baselines from their oficial open-source implementations. All systems use GPT-5.6 Terra/Luna with a common ceiling of 12 genuine evaluations, each a completed training– evaluation run, and pre-specified API-expenditure safeguards (Ap pendix B). Candidates are selected by development-set NDCG@5 and retrained for final test evaluation; results are averaged over three independent searches. The ablation study uses MoRec on all four datasets; continuation and feedback use MoRec/CDs, and mechanism discovery uses BERT4Rec/Electronics. These analyses use development NDCG@5. Space construction compares final test performance on SASRec/CDs. Reference API cost measures modelcall expenditure, and GPU-hours measure search computation; both exclude final retraining. Appendices A–D detail evaluation scope, aggregation, ranking protocols, feedback controls, and implementation settings.

## 4.1 Performance Comparison

Algorithm discovery across starting models. Table 1 reports final test performance across 16 model–dataset settings. RecRSI improves on the initial algorithm in all 16 settings and achieves the highest mean NDCG@5 in 11 settings and the highest mean HR@5 in 12. Averaged over all settings, the relative gains over the initial algorithms are 8.48% in NDCG@5 and 6.88% in HR@5; gains over the strongest competing result in each setting and metric are 2.60% and 2.20%. The mean advantage remains positive after omitting any starting model or dataset (Appendix A).

BERT4Rec/Electronics links these performance gains to concrete algorithm designs. RecRSI reaches test NDCG@5 of 0.4046, compared with 0.3881 for the initial algorithm and 0.3959 for the strongest competitor, RecHarness. The eight selected designs combine changes to scoring, objectives, and representations.

Section 4.3.4 examines how these combinations arise and which components contribute to their gains.

![](images/bc57acde6f99e8b1186111302c136a3b79458cb9dd4af052265d27268698281c.jpg)

![](images/6c501b436b2b7e6ea1c62dba8fcd5cc8275f1fadeec20cd4664d704f0b5aa73f.jpg)

![](images/a676ff6737ac8ef867765695e193c899e14f756dd8aa3dcb2042f814283cd77c.jpg)  
Figure 3: Discovery quality and search cost. (a,b) Eight-task final test NDCG@5 gain over Seed versus API reference cost (USD) and per GPU-hour. (c) Best validation gain at a common input-plus-output token ceiling on the same eight tasks; at 3 million tokens, RecRSI reaches 7.72% versus 5.09% for RecHarness. Repeats are averaged within tasks, then tasks equally. Search costs exclude final retraining.

Discovery quality and research investment. Across the eight tasks in Figure 3(a,b), RecRSI lies on the quality–API-cost Pareto frontier and achieves the highest average test gain. Compared with RecHarness, it spends an additional USD 2.38 per search on average and increases the mean test NDCG@5 gain over the seed from 4.68% to 7.36%—a 57.17% larger gain. RecRSI uses additional tokens to formulate mechanism-level hypotheses, compose interventions, and integrate evidence. The higher discovery quality shows the return on this research investment. RecRSI also leads at a common budget of 3 million tokens: mean validation gain reaches 7.72% versus 5.09% for RecHarness (panel c). It achieves the highest test gain per GPU-hour, 1.96 percentage points.

## 4.2 Ablation Studies

The ablations examine how mechanism knowledge, experimental history, and research roles support algorithm construction. Full combines specialized roles, explicit MechSpace constraints, and Mechanism Feedback. Prompt supplies the same mechanism knowledge through natural-language guidance instead of explicit constraints; Unified uses four unspecialized roles. Summary replaces mechanism history with a general-purpose experiment summary; Prompt+Summary applies both substitutions. Each configuration searches MoRec on CDs, Electronics, Ofice, and MovieLens-1M with up to 12 evaluations. We average the relative gain of the best so-far development score over the initial model, weighting the four datasets equally.

Table 2: Component contributions and space–feedback interaction on MoRec across four datasets.
<table><tr><td>Method</td><td>Mean frontier gain (%) ↑</td><td>Change vs. Full (pp)</td></tr><tr><td>Full (RecRSI)</td><td>5.6775</td><td>一</td></tr><tr><td>Prompt</td><td>5.4591</td><td>-0.2184</td></tr><tr><td>Unified</td><td>5.6387</td><td>-0.0388</td></tr><tr><td>Summary</td><td>5.1927</td><td>-0.4848</td></tr><tr><td>Prompt + Summary</td><td>5.5807</td><td>-0.0968</td></tr></table>

<table><tr><td rowspan="2">Feedback</td><td colspan="3">Space effect</td></tr><tr><td>MechSpace</td><td>Prompt</td><td>∆(pp)</td></tr><tr><td>Mechanism</td><td>5.6775</td><td>5.4591</td><td>+0.2184</td></tr><tr><td>Summary</td><td>5.1927</td><td>5.5807</td><td>-0.3880</td></tr><tr><td>Feedback effect ∆ (pp)</td><td>+0.4848</td><td>-0.1216</td><td>+0.6064</td></tr></table>

Gains (%) average best-development NDCG@5 improvement over the initial model equally across four datasets. Efects are percentage-point diferences; bottom-right is $g _ { F } - g _ { P } - g _ { S } + g _ { P + S } .$

Joint value ofmechanism choices and feedback. Full achieves the highest mean frontier gain, 5.6775%, across the four datasets (Table 2). With explicit MechSpace, Mechanism Feedback raises the gain from 5.1927% under Summary to 5.6775%, a 9.34% increase in discovery gain. The space–feedback interaction is positive at 0.6064 percentage points: explicit MechSpace improves the frontier with Mechanism Feedback, whereas its advantage disappears under a general summary. The four-way comparison links the benefit of structured mechanism choices to how experimental experience guides their use. MechSpace specifies implementable interventions; hypothesis-linked feedback connects them to the design conditions and outcomes of earlier tests, helping later hypotheses target promising combinations.

## 4.3 Analysis of Mechanism-Guided Research

The following studies track how earlier experiments shape later designs, from continued search to individual feedback decisions and mechanism combinations.

4.3.1 Research Continuation vs. Restarts. Continue (Long24) extends a single search to 24 evaluations; Restart (2×Short12) takes the better result from two independent 12-evaluation searches. Both strategies receive 24 evaluation opportunities in each of three MoRec/CDs blocks. Table 3 reports their final quality and API expenditure.

![](images/8c0bf300de43c96002f4dd4b3a098f172f16fc51159d957287c6e8ead9780c09.jpg)  
Figure 4: Overlaid research progress on MoRec/CDs. Colors distinguish three experiment blocks, with solid Continue and dashed Restart trajectories. At 2� total evaluations, Restart takes the better of the two independent searches after � evaluations each. The divider marks evaluation 12; filled circles mark later bests, including the Block 3 scoring hypothesis. Table 3 reports final quality and API cost.

Table 3: Final discovery quality and continuation gains at 24 total evaluations.
<table><tr><td>Block</td><td>Continue (24)</td><td>Best restart (2×12)</td><td>Gain after eval. 12</td><td>Continue – restart</td></tr><tr><td>1</td><td>0.4237</td><td>0.4195</td><td>+0.0057</td><td>+0.0043</td></tr><tr><td>2</td><td>0.4299</td><td>0.4299</td><td>+0.0043</td><td>≈ 0</td></tr><tr><td>3</td><td>0.4153</td><td>0.3884</td><td>+0.0352</td><td>+0.0269</td></tr></table>

Scores and gains are development NDCG@5; diferences use unrounded values (Block 2: −2.38×10<sup>−5</sup>). API reference USD for Continue / both restarts: 8.81 / 8.10, 8.51 / 8.19, and 9.03 / 7.54 for Blocks 1–3.

Continue achieves higher final NDCG@5 in Blocks 1 and 3 and nearly ties the best restart in Block 2. All three continued searches find new best candidates after evaluation 12 (Figure 4).

Block 3 shows how earlier exploration supports the later improvement. Its best NDCG@5 remains at 0.3801 from evaluations 2 through 12. At evaluation 13, the next hypothesis combines the existing recent-three-item fusion and margin objective with cosinenormalized scoring at temperature 0.2. The combination reaches

0.4153, exceeding the best restart by 6.92%. This scoring experiment builds on the earlier representation and objective to improve the design beyond the short-search horizon.

To examine how accumulated experience guides the next design, we branch from the same MoRec/CDs incumbent with its research history available or reset. Access to history yields a better candidate at each of three successive steps: mean NDCG@5 is 0.3947 versus 0.3681, a diference of 0.0266 (Table 13). After a query-gate regression, the history-informed branch keeps the scorer and tests item-embedding gradient clipping; the reset branch switches to a pairwise objective. Earlier evidence thus guides both the choice of a new intervention and the design it builds on (Appendix E).

4.3.2 Feedback-Guided Research Decisions. When an experiment underperforms, the next decision depends on which mechanisms to preserve and which hypothesis to test. We compare Raw, Summary, and Mechanism at frozen decision points from real MoRec/CDs research trajectories, yielding ten next-candidate comparisons. Raw provides the experimental records; Summary turns them into actionable research advice; Mechanism links observations to the hypotheses and mechanisms tested and proposes follow-up investigations. Each comparison shares the research history, available actions, and downstream implementation and evaluation process; Summary and Mechanism also share the feedback model and output-token limit (Appendix C). Mechanism leads to higher next-candidate NDCG@5 than Summary in seven of the ten comparisons (Table 4).

Table 4: Next-candidate quality in ten feedback comparisons at frozen MoRec/CDs decision points.
<table><tr><td>State</td><td>Raw</td><td>Summary</td><td>Mechanism</td></tr><tr><td>1</td><td>0.3647</td><td>0.3729</td><td>0.3767</td></tr><tr><td>2</td><td>0.3647</td><td>0.3729</td><td>0.3729</td></tr><tr><td>3</td><td>0.4123</td><td>0.4135</td><td>0.4150</td></tr><tr><td>4</td><td>0.3729</td><td>0.3729</td><td>0.3729</td></tr><tr><td>5</td><td>0.3681</td><td>0.3673</td><td>0.4037</td></tr><tr><td>6</td><td>0.3729</td><td>0.3729</td><td>0.3755</td></tr><tr><td>7</td><td>0.4079</td><td>0.4152</td><td>0.4175</td></tr><tr><td>8</td><td>0.4139</td><td>0.3739</td><td>0.3996</td></tr><tr><td>9</td><td>0.3339</td><td>0.3729</td><td>0.3409</td></tr><tr><td>10</td><td>0.3377</td><td>0.3612</td><td>0.3625</td></tr></table>

Development NDCG@5; bold denotes the best value, including ties.

At State 6, a convolution-and-mask regression leads Mechanism Feedback to test a prefix-mean representation with weight 0.1, keeping the gate and BCE objective unchanged. The next candidate reaches 0.3755, compared with 0.3729 under both Raw and Summary. At State 5, Mechanism tests cosine scoring with the pairwise objective, reaching 0.4037 versus 0.3673 for Summary. These deci sions show how feedback guides both what to change and what to build on, turning the outcome of a mechanism hypothesis into the basis for the next experiment.

4.3.3 Mechanism-Space Construction. We test whether MechSpace can supply efective interventions for algorithm discovery by com paring its generated space (A) with a human-designed space (H) on SASRec/CDs.

To construct H, the authors examine the initial SASRec implementation and established recommendation techniques, manually specifying mechanisms, parameter domains, and composition rules. H defines six mechanisms, including relative-position attention, gated FFNs, and ranking objectives, with 20 single/pair structures and 95 parameter assignments. A contains 16 fixed-parameter primitives and 494 nominal single/pair/triple combinations. Mechanism coverage, composition rules, and space size are outcomes of the construction procedures; both resulting spaces are searched through 12 genuine evaluations.

Both searches use Research Line with the same SASRec initialization, data, evaluator, training protocol, and LLM role-to-model assignments. Each winner is selected by validation NDCG@5, retrained with three seeds, and evaluated on held-out test data.

Table 5: Final test performance of the validation-selected candidates from the automatically generated (A) and humandesigned (H) mechanism spaces on SASRec/CDs. Results are reported as mean ± sample standard deviation.
<table><tr><td>Space</td><td>NDCG@5</td><td>HR@5</td></tr><tr><td>A: MechSpace</td><td> $\mathbf { 0 . 4 2 3 6 \pm 0 . 0 0 3 6 }$ </td><td> $\mathbf { 0 . 5 5 7 2 \pm 0 . 0 0 6 2 }$ </td></tr><tr><td>H: Human-designed</td><td> $0 . 3 5 4 0 \pm 0 . 0 0 5 1$ </td><td> $0 . 4 6 6 0 \pm 0 . 0 0 5 3$ </td></tr></table>

A yields 19.65% higher test NDCG@5 and 19.57% higher HR@5 than H (Table 5). The selected algorithm from A normalizes embedding directions before adding positional embeddings, an intervention absent from H. The winner from H combines bounded causal relative-position attention bias with hardness-weighted binary cross-entropy. The comparison shows how MechSpace supports discovery: it supplies a useful intervention absent from the manual space, which Research Line turns into a higher-performing algorithm under the same evaluation budget.

4.3.4 Discovered Mechanisms and Research Hypotheses. We analyze the mechanisms discovered in the three main-comparison searches (RO1–RO3) and five additional searches (RO4–RO8) on BERT4Rec/Electronics (Table 6). Each improves on its initial algorithm, with relative NDCG@5 gains of 0.98–8.67% and a mean of 3.87%. The selected programs combine two or three mechanism categories: scoring and objective changes each appear in seven designs, architecture changes in five, and representation changes in one. The frequent scoring–objective combinations connect the values used to learn the recommender with those used to rank items, making their compatibility a concrete question for mechanism-guided research.

RO2 produces the largest gain, 8.67%, by combining cosinetemperature scoring with label-smoothed training while keeping the encoder unchanged. The shared scorer coordinates training logits and item ranking. Component comparisons for RO1–RO3 (Appendix C.1) expose why design context matters. With bilinear scoring fixed in RO1, layer aggregation changes NDCG@5 by −0.0014 without smoothing but +0.0015 with it. This component comparison identifies an objective-dependent design opportunity: layer aggregation improves the design when paired with smoothing. A hypothesis records the intervention together with its surrounding design so that this dependence can guide the next experiment. In RO2, smoothing adds 0.0019 to an already efective cosine scorer.

Table 6: Mechanism combinations and development-set gains in all eight BERT4Rec/Electronics searches.
<table><tr><td>Search</td><td>Selected mechanism combination</td><td>Categories</td><td>Root</td><td>Selected</td><td>Gain (%)</td></tr><tr><td>RO1</td><td>Bilinear scoring + layer aggregation + label smoothing</td><td> $S + \mathbf { A } + \mathbf { O }$ </td><td>0.4079</td><td>0.4199</td><td>+2.95</td></tr><tr><td>RO2</td><td>Cosine-temperature scoring + label smoothing</td><td> $\mathsf { S } + \mathsf { O }$ </td><td>0.4047</td><td>0.4398</td><td>+8.67</td></tr><tr><td>RO3</td><td>Bias-free query projection + label smoothing</td><td> $\mathsf { S } + \mathsf { O }$ </td><td>0.4045</td><td>0.4220</td><td>+4.32</td></tr><tr><td>RO4</td><td>Shared score projection + position scaling</td><td> ${ \mathsf S } + { \mathsf A }$ </td><td>0.4089</td><td>0.4165</td><td>+1.87</td></tr><tr><td>RO5</td><td>Bounded item bias + label smoothing</td><td> $\mathsf { S } + \mathsf { O }$ </td><td>0.4054</td><td>0.4094</td><td>+0.98</td></tr><tr><td>RO6</td><td>Relative-distance bias + masked decoder + label smoothing</td><td> $\mathrm { R } + \mathrm { A } + \mathrm { O }$ </td><td>0.3997</td><td>0.4147</td><td>+3.75</td></tr><tr><td>RO7</td><td>Context scoring head + mask adapter + label smoothing</td><td> $S + \mathbf { A } + \mathbf { O }$ </td><td>0.4069</td><td>0.4209</td><td>+3.43</td></tr><tr><td>RO8</td><td>Bilinear query + deep prediction head + in-batch discrimination</td><td> $S + \mathbf { A } + \mathbf { O }$ </td><td>0.4043</td><td>0.4243</td><td>+4.95</td></tr></table>

Root and Selected are search-stage development NDCG@5; gains use each search’s unrounded root value. R: representation; A: architecture; S: scoring; O: objective. Categories describe the complete selected composition.

RO3’s pre-experiment hypothesis shows how prior results shape the next combination. It builds on label smoothing, which had improved development NDCG@5 from 0.4045 to 0.4101, and uses the regression of a residual-head alternative to motivate a new scoring hypothesis. The improved, smoothed design becomes the reference for testing the new scorer. A bias-free query projection tests whether learned matching geometry can complement this objective; the resulting design reaches 0.4220. The combination exceeds both projection alone (0.4170) and smoothing alone (0.4101). The proposal history explains why the scorer was tested; the component comparisons show what it adds to the design.

## 5 Conclusion

RecRSI makes mechanism hypotheses the unit of cumulative recommendation research. Its task-derived mechanism space, referencebased candidate construction, and mechanism feedback allow research to build on promising designs and learn from both successful and unsuccessful experiments. Across 16 model–dataset settings, RecRSI improves both ranking metrics over every initial recommender and achieves positive average gains over the strongest competing results. The process analyses show how accumulated evidence redirects research and develops stronger mechanism combinations. The resulting research cycle uses each experiment to inform the next mechanism choice.

## References

[1] Yongqiang Chen, Chenxi Liu, Zhenhao Chen, Tongliang Liu, Bo Han, and Kun Zhang. 2026. CausalEvolve: Towards Open-Ended Discovery with Causal Scratch pad. arXiv:2603.14575 doi:10.48550/arXiv.2603.14575

[2] Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. 2019. Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. In Proceedings ofRecSys. ACM, New York, NY, USA, 101–109. doi:10.1145/3298689.3347058

[3] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19 (2015), 19 pages. doi:10.1145/2827872

[4] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings ofSIGIR. ACM, New York, NY, USA, 639–648. doi:10.1145/3397271.3401063

[5] Seokhoon Jeong, Mijung Kim, and Taehwan Kim. 2026. Agentic Neural Architec ture Search. arXiv:2607.07984 doi:10.48550/arXiv.2607.07984

[6] Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. 2025. AIDE: AI-Driven Exploration in the Space of Code. arXiv:2502.13138 doi:10.48550/arXiv.2502.13138

[7] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, Piscataway, NJ, USA, 197–206. doi:10.1109/ICDM.2018.00035

[8] Andrej Karpathy. 2026. autoresearch: AI Agents Running Research on Single-GPU Nanochat Training Automatically. GitHub. https://github.com/karpathy/ autoresearch

[9] Sein Kim, Sangwu Park, Hongseok Kang, Wonjoong Kim, Jimin Seo, Yeonjun In, Kanghoon Yoon, Hyunsik Jeon, and Chanyoung Park. 2026. Self-EvolveRec: Self-Evolving Recommender Systems with LLM-Based Directional Feedback. arXiv:2602.12612 doi:10.48550/arXiv.2602.12612

[10] Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. 2026. ShinkaEvolve: Towards Open-Ended and Sample-Eficient Program Evolution. In The Fourteenth International Conference on Learning Representations. 74026–74078. https://proceedings.iclr.cc/paper\_files/paper/2026/hash 7886b9bafe76c52fd568db10f9772df-Abstract-Conference.html

[11] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu, Chuhan Wu, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, and Weinan Zhang. 2025. How Can Recommender Systems Benefit from Large Language Models: A Survey. ACM Transactions on Information Systems 43, 2, Article 28 (2025), 47 pages. doi:10.1145/3678004

[12] Haoran Ling, Yuecheng Li, Zeyu Song, Jing Yao, Shuwen Kang, Chi Lu, Wenjin Wu, and Peng Jiang. 2026. RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems. arXiv:2607.29241 https://arxiv.org/abs/ 2607.29241

[13] Bin Liu, Chenxu Zhu, Guilin Li, Weinan Zhang, Jincai Lai, Ruiming Tang, Xiuqiang He, Zhenguo Li, and Yong Yu. 2020. AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction. In Proceedings ofKDD. ACM, New York, NY, USA, 2636–2645. doi:10.1145/3394486. 3403314

[14] Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. 2024. Evolution of Heuristics: Towards Eficient Auto matic Algorithm Design Using Large Language Model. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, Cambridge, MA, USA, 32201–32223. https: //proceedings.mlr.press/v235/liu24bs.html

[15] Gang Liu, Yihan Zhu, Jie Chen, and Meng Jiang. 2025. Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research. arXiv:2510.06056 doi:10.48550/arXiv.2510.06056

[16] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. 2024. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292 doi:10.48550/arXiv.2408.06292

[17] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative Refinement with Self Feedback. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., Red Hook, NY, USA, 46534–46594. doi:10.52202/075280-2019

[18] Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, and Xiuqiang He. 2021. SimpleX: A Simple and Strong Baseline for Collaborative Filtering. In Proceedings ofthe 30th ACM International Conference on Information and Knowledge Management (CIKM ’21). Association for Computing Machinery, New York, NY, USA, 1243–1252. doi:10.1145/3459637.3482297

[19] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-based Recommendations on Styles and Substitutes. In Proceedings ofthe 38th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 43–52. doi:10.1145/2766462.2767755

[20] Lingyu Mu, Hao Deng, Haibo Xing, Jinxin Hu, Yu Zhang, and Xiaoyi Zeng. 2026. EvoRec: Self Evolving Agentic Recommender Systems. arXiv:2606.28368 doi:10.48550/arXiv.2606.28368

[21] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen˜ Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Ba log. 2025. AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery. arXiv:2506.13131 doi:10.48550/arXiv.2506.13131

[22] Esteban Real, Chen Liang, David So, and Quoc Le. 2020. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch. In Proceedings ofICML (Proceedings ofMachine Learning Research, Vol. 119). PMLR, Cambridge, MA, USA, 8007–8019. https://proceedings.mlr.press/v119/real20a.html

[23] Stefen Rendle, Walid Krichene, Li Zhang, and John Roberts Anderson. 2020. Neural Collaborative Filtering vs. Matrix Factorization Revisited. In Proceedings of the 14th ACM Conference on Recommender Systems. Association for Computing Machinery, New York, NY, USA, 240–248. doi:10.1145/3383313.3412488

[24] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. 2024. Mathematical Discoveries from Program Search with Large Language Models. Nature 625 (2024), 468–475. doi:10.1038/s41586-023-06924-6

[25] Yuiko Sakuma, Masakazu Yoshimura, Marcel Gröpl, Zitang Sun, Junji Otsuka, Atsushi Irie, and Takeshi Ohashi. 2026. Structuring Open-Ended NAS: Semi-Automated Design Knowledge Structuring with LLMs for Eficient Neural Architecture Search. arXiv:2605.19247 doi:10.48550/arXiv.2605.19247

[26] Asankhaya Sharma. 2025. OpenEvolve: An Open-Source Evolutionary Coding Agent. GitHub. https://github.com/algorithmicsuperintelligence/openevolve

[27] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., Red Hook, NY, USA, 8634– 8652. https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

[28] Qingquan Song, Dehua Cheng, Hanning Zhou, Jiyan Yang, Yuandong Tian, and Xia Hu. 2020. Towards Automated Neural Interaction Discovery for Click Through Rate Prediction. In Proceedings ofKDD. ACM, New York, NY, USA, 945–955. doi:10.1145/3394486.3403137

[29] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York, NY, USA, 1441–1450. doi:10.1145/3357384.3357895

[30] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural Graph Collaborative Filtering. In Proceedings ofthe 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 165–174. doi:10. 1145/3331184.3331267

[31] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. 2025. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066 doi:10. 48550/arXiv.2504.08066

[32] Shuang Yang, Zijie Zhuang, Changxin Lao, Pengbo Xu, Hanwen Xu, Yusheng Huang, Han Gao, Guanchen Wang, Tianbao Ma, Linxun Chen, Peilin Song, Xuming Wang, Chen Li, Fan Wu, Tao Wang, Zibo Zhao, Xiangyu Wu, An Liu, Fei Pan, Peng Jiang, Chen Yang, Zhaojie Liu, and Wenwu Ou. 2026. Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems. arXiv:2609.30001 https://arxiv.org/abs/2609.3000

[33] Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. 2024. ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., Red Hook, NY, USA, 43571–43608. https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html

[34] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to Go Next for Recommender Systems? ID vs. Modality-based Recommender Models Revisited. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 2639–2649. doi:10.1145/3539618.3591932

[35] Xingshan Zeng, Zishan Xu, Boju Zhang, Yuzhou Wu, Lingzhi Wang, Jianghao Lin, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Weinan Zhang, Yong Yu, Qun Liu, and Weiwen Liu. 2026. What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents. arXiv:2608.27260 doi:10.48550/arXiv. 2608.27260

[36] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. 2025. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954 doi:10.48550/arXiv.2505.22954

[37] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang,

and Ji-Rong Wen. 2021. RecBole: Towards a Unified, Comprehensive and Eficient Framework for Recommendation Algorithms. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York, NY, USA, 4653–4664. doi:10.1145/3459637. 3482016

[38] Chenyu Zhou, Huacan Chai, Wenteng Chen, Zihan Guo, Rong Shan, Yuanyi Song, Tianyi Xu, Yingxuan Yang, Aofan Yu, Weiming Zhang, Congming Zheng, Jiachen Zhu, Zeyu Zheng, Zhuosheng Zhang, Xingyu Lou, Changwang Zhang, Zhihui Fu, Jun Wang, Weiwen Liu, Jianghao Lin, and Weinan Zhang. 2026. Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering. arXiv:2604.08224 doi:10.48550/arXiv.2604.08224

Table 7: Search-to-search standard deviation of final TEST NDCG@5 and HR@5, aligned with Table 1.
<table><tr><td rowspan="2">Seed Recommender</td><td rowspan="2">Method</td><td colspan="2">CDs</td><td colspan="2">Electronics</td><td colspan="2">Office</td><td colspan="2">MovieLens-1M</td></tr><tr><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td><td>NDCG@5</td><td>HR@5</td></tr><tr><td rowspan="6">SASRec</td><td>RecHarness</td><td>0.0051</td><td>0.0114</td><td>0.0288</td><td>0.0335</td><td>0.0034</td><td>0.0047</td><td>0.0045</td><td>0.0039</td></tr><tr><td>DeepEvolve</td><td>0.0286</td><td>0.0356</td><td>0.0012</td><td>0.0019</td><td>0.0082</td><td>0.0154</td><td>0.0571</td><td>0.0538</td></tr><tr><td>OpenEvolve</td><td>0.0065</td><td>0.0074</td><td>0.0020</td><td>0.0035</td><td>0.0019</td><td>0.0011</td><td>0.0015</td><td>0.0034</td></tr><tr><td>ShinkaEvolve</td><td>0.0081</td><td>0.0142</td><td>0.0064</td><td>0.0057</td><td>0.0039</td><td>0.0084</td><td>0.0038</td><td>0.0052</td></tr><tr><td>Self-EvolveRec</td><td>0.0047</td><td>0.0105</td><td>0.0004</td><td>0.0010</td><td>0.0050</td><td>0.0087</td><td>0.0048</td><td>0.0048</td></tr><tr><td>RecRSI (Ours)</td><td>0.0095</td><td>0.0115</td><td>0.0185</td><td>0.0182</td><td>0.0032</td><td>0.0086</td><td>0.0049</td><td>0.0060</td></tr><tr><td rowspan="6">MoRec</td><td>RecHarness</td><td>0.0156</td><td>0.0159</td><td>0.0051</td><td>0.0055</td><td>0.0055</td><td>0.0053</td><td>0.0021</td><td>0.0025</td></tr><tr><td>DeepEvolve</td><td>0.0149</td><td>0.0194</td><td>0.0097</td><td>0.0103</td><td>0.0022</td><td>0.0002</td><td>0.0077</td><td>0.0039</td></tr><tr><td>OpenEvolve</td><td>0.0026</td><td>0.0070</td><td>0.0003</td><td>0.0007</td><td>0.0019</td><td>0.0041</td><td>0.0015</td><td>0.0025</td></tr><tr><td>ShinkaEvolve</td><td>0.0077</td><td>0.0124</td><td>0.0073</td><td>0.0075</td><td>0.0033</td><td>0.0049</td><td>0.0018</td><td>0.0022</td></tr><tr><td>Self-EvolveRec</td><td>0.0072</td><td>0.0109</td><td>0.0033</td><td>0.0030</td><td>0.0019</td><td>0.0016</td><td>0.0016</td><td>0.0007</td></tr><tr><td>RecRSI (Ours)</td><td>0.0131</td><td>0.0170</td><td>0.0135</td><td>0.0155</td><td>0.0048</td><td>0.0048</td><td>0.0085</td><td>0.0057</td></tr><tr><td rowspan="6">BERT4Rec</td><td>RecHarness</td><td>0.0108</td><td>0.0104</td><td>0.0072</td><td>0.0074</td><td>0.0122</td><td>0.0151</td><td>0.0061</td><td>0.0057</td></tr><tr><td>DeepEvolve</td><td>0.0082</td><td>0.0109</td><td>0.0041</td><td>0.0032</td><td>0.0086</td><td>0.0120</td><td>0.0091</td><td>0.0092</td></tr><tr><td>OpenEvolve</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0068</td><td>0.0069</td></tr><tr><td>ShinkaEvolve</td><td>0.0006</td><td>0.0010</td><td>0.0085</td><td>0.0085</td><td>0.0049</td><td>0.0054</td><td>0.0153</td><td>0.0108</td></tr><tr><td>Self-EvolveRec</td><td>0.0408</td><td>0.0507</td><td>0.0118</td><td>0.0126</td><td>0.0311</td><td>0.0416</td><td>0.0103</td><td>0.0102</td></tr><tr><td>RecRSI (Ours)</td><td>0.0379</td><td>0.0389</td><td>0.0074</td><td>0.0083</td><td>0.0031</td><td>0.0064</td><td>0.0087</td><td>0.0059</td></tr><tr><td rowspan="6">NGCF</td><td>RecHarness</td><td>0.0081</td><td>0.0118</td><td>0.0109</td><td>0.0124</td><td>0.0011</td><td>0.0025</td><td>0.0036</td><td>0.0116</td></tr><tr><td>DeepEvolve</td><td>0.0013</td><td>0.0035</td><td>0.0073</td><td>0.0121</td><td>0.0017</td><td>0.0024</td><td>0.0085</td><td>0.0192</td></tr><tr><td>OpenEvolve</td><td>0.0007</td><td>0.0014</td><td>0.0003</td><td>0.0003</td><td>0.0008</td><td>0.0015</td><td>0.0001</td><td>0.0004</td></tr><tr><td>ShinkaEvolve</td><td>0.0002</td><td>0.0001</td><td>0.0235</td><td>0.0280</td><td>0.0016</td><td>0.0020</td><td>0.0011</td><td>0.0028</td></tr><tr><td>Self-EvolveRec</td><td>0.0221</td><td>0.0221</td><td>0.0129</td><td>0.0163</td><td>0.0047</td><td>0.0035</td><td>0.0082</td><td>0.0134</td></tr><tr><td>RecRSI (Ours)</td><td>0.0006</td><td>0.0004</td><td>0.0205</td><td>0.0263</td><td>0.0075</td><td>0.0116</td><td>0.0001</td><td>0.0023</td></tr></table>

Entries are sample SD (� = 3) across source-search winners evaluated in the named environment. – denotes an unavailable SD.

## A Evaluation and Aggregation

Evaluation scope. Table 1 evaluates discovered algorithms in 16 settings from four recommenders and four datasets, combining task-specific search winners and source-selected designs retrained on target data. This assesses final quality and design reuse across environments. The common search cohort in Appendix B examines progress and resources; source-linked bootstrap analysis below assesses search variability.

Final test performance and variability. For system �, evaluation setting �, and source-search repeat $r , \bar { q } _ { m , t , r }$ averages final TEST retraining scores of the algorithm selected by development NDCG@5. Its design stays fixed during target retraining. Tables 1 and 7 report means and sample SDs over three searches:

$$
\bar { q } _ { m , t } = \frac { 1 } { 3 } \sum _ { r = 1 } ^ { 3 } \bar { q } _ { m , t , r } , \qquad s _ { m , t } = \sqrt { \frac { 1 } { 2 } \sum _ { r = 1 } ^ { 3 } ( \bar { q } _ { m , t , r } - \bar { q } _ { m , t } ) ^ { 2 } } .\tag{4}
$$

The SD measures variation between discovered algorithms. OpenEvolve’s zero SD on BERT4Rec/CDs, Electronics, and Ofice reflects three Seed-equivalent implementations sharing final test evaluations.

Across-task comparisons. Relative gains are averaged equally over 16 settings, using Seed or the highest competing mean per setting and metric as the denominator. Table 8 compares fixed systems. Against the strongest competitor, mean NDCG@5/HR@5 advantages remain positive when omitting any setting (1.96–2.94%/1.66– 2.56%), starting model (1.57–3.35%/1.35–2.97%), or dataset (1.33– 3.52%/1.07–2.93%).

Table 8: RecRSI versus each comparator on the 16 settings. Δ is mean relative gain (%); W/T/L counts wins, ties, and losses based on mean test scores.
<table><tr><td></td><td colspan="2">NDCG@5</td><td colspan="2">HR@5</td></tr><tr><td>Comparator</td><td>Δ</td><td>W/T/L</td><td>Δ</td><td>W/T/L</td></tr><tr><td>RecHarness</td><td>+5.52</td><td>13/0/3</td><td>+4.24</td><td>14/0/2</td></tr><tr><td>DeepEvolve</td><td>+5.70</td><td>15/0/1</td><td>+4.83</td><td>16/0/0</td></tr><tr><td>OpenEvolve</td><td>+4.38</td><td>15/0/1</td><td>+3.50</td><td>14/0/2</td></tr><tr><td>ShinkaEvolve</td><td>+4.31</td><td>15/0/1</td><td>+3.45</td><td>14/0/2</td></tr><tr><td>Self-EvolveRec</td><td>+7.23</td><td>15/0/1</td><td>+6.12</td><td>14/0/2</td></tr></table>

Search-level robustness. Search variability is assessed on the eight MoRec and BERT4Rec settings with linked source-search and targetevaluation records. Mean NDCG@5/HR@5 gains over the taskwise strongest competitor are 3.70%/3.06%, with 95% percentile bootstrap intervals [1.64%, 5.57%]/[1.36%, 4.61%]. In 100,000 draws, we independently resample searches within system and source task using aggregated retraining scores, carry each search’s target evaluations together to preserve dependence, and recompute the strongest comparator. Both intervals exceed zero.

Table 9: Task data splits and ranking protocols. Sampled pools contain one positive and 99 fixed negatives; all candidates within a task share the same evaluation protocol.
<table><tr><td>Task</td><td>Split / evaluation history</td><td>Ranking protocol</td></tr><tr><td>SASRec / all four datasets</td><td>Per-user last-two holdout; test histories include training and valida- One positive plus 99 fixed negatives, evalu- tion interactions.</td><td>ation seed 72026.</td></tr><tr><td>BERT4Rec/Electronics, Office</td><td>Amazon 2014 5-core; penultimate interaction for development, last Fixed uniform sampled pools. for test.</td><td></td></tr><tr><td>BERT4Rec/MovieLens-1M</td><td>Per-user temporal split of the full MovieLens-1M data.</td><td>Fixed sampled pools; test negatives exclude train/dev and the test positive.</td></tr><tr><td>BERT4Rec/CDs</td><td>Common preprocessed data and user/item IDs across systems.</td><td>Fixed sampled candidate sets.</td></tr><tr><td>MoRec/CDs, Electronics</td><td>Common CDs preprocessing across systems; Electronics uses per- Fixed sampled candidate sets shared across user last-two holdout. Test histories include training and validation systems. interactions.</td><td></td></tr><tr><td>MoRec/Office, MovieLens-1M</td><td>Per-user timestamp order (source-row tie-break); penultimate inter- One positive and 99 fixed uniform nega- action for development, last for test. Test history: train + develop- tives; seed 72026. Negatives exclude the ment.</td><td>query history and target.</td></tr><tr><td>NGCF/CDs, Electronics, Office</td><td>Per-user last-two temporal holdout</td><td>Fixed uniform pools; development/test seeds 72026/72027.</td></tr><tr><td>NGCF/MovieLens-1M</td><td>Fixed task split.</td><td>Full-catalog ranking.</td></tr></table>

## B Discovery Curves and Resource Accounting

Search progress and investment. Figures 1(b) and 3 link progress and final quality to research expenditure using 144 trajectories: six systems, three searches, and eight tasks (SASRec/CDs; MoRec/CDs, Ofice, MovieLens-1M; BERT4Rec and NGCF each on Electronics and MovieLens-1M). Searches are averaged within tasks, then tasks equally. Figure 1(b) tracks best development NDCG@5, including the initial algorithm and carrying the best forward after stopping. Figure 3(c) plots its relative gain at cumulative input-plus-output token ceilings through 3 million, counting setup once and reported reasoning and cached tokens. Panels (a,b) pair final TEST gains with search costs; GPU eficiency divides mean relative gain by mean GPU-hours.

Budgets and stopping. The shared ceiling is 12 genuine evaluations; failed implementation or qualification consumes resources without counting as evaluation. A USD 7–8 API safeguard, fixed before search and independent of performance, stops searches at the cost limit using completed plus in-flight expenditure. It stopped 18 DeepEvolve or Self-EvolveRec searches (Table 10). Counts and costs include resumed work within each search; evaluation-budget exhaustion and execution interruptions also end searches.

Search costs. Reference API cost includes research calls, initialization, and mechanism-space construction. GPU-hours cover allocation for qualification, each search’s initial-training share, candidate fits, and unsuccessful attempts. Final TEST retraining and transfer evaluation are excluded from search cost.

## C Mechanism Hypotheses and Feedback Controls

Mechanism space andfeedback. Researchers fix source interfaces, editable targets, allowed axes, and composition limits. MechSpace generates primitives, parameter settings, and compatibility rules within them. For MoRec/CDs (Section 3.2), generation proposes 18 primitives; review selects 16 across six axes. SELECT chooses one; COMPOSE combines two or three, with at most one per axis and two mutation targets: 16 singles, 106 pairs, and 372 triples. Primitives specify locations, types, stages, parameters, implementation, and incompatibilities. Explicit selection enforces these specifications; Prompt supplies them as text.

Table 10: Completed evaluations in searches stopped by the API safeguard. R1–R3 are search repeats; – marks a repeat not stopped by this safeguard.
<table><tr><td>System</td><td>Task</td><td>R1</td><td>R2</td><td>R3</td></tr><tr><td rowspan="4">DeepEvolve</td><td>MoRec/CDs</td><td>2</td><td>6</td><td>0</td></tr><tr><td>MoRec/Office</td><td>0</td><td>1</td><td>一</td></tr><tr><td>MoRec/MovieLens-1M</td><td>11</td><td>一</td><td>一</td></tr><tr><td>BERT4Rec/Electronics</td><td>一</td><td>10</td><td>11</td></tr><tr><td rowspan="5">Self-EvolveRec</td><td>MoRec/CDs</td><td>2</td><td>一</td><td>10</td></tr><tr><td>MoRec/MovieLens-1M</td><td>一</td><td>一</td><td>1</td></tr><tr><td>BERT4Rec/Electronics</td><td>6</td><td>6</td><td>-</td></tr><tr><td>BERT4Rec/MovieLens-1M</td><td>9</td><td>9</td><td>一</td></tr><tr><td>NGCF/Electronics</td><td>7</td><td>0</td><td>2</td></tr></table>

Four-dataset ablation. Table 2 averages $1 0 0 ( q _ { \mathrm { m a x } } / q _ { 0 } ~ - ~ 1 )$ equally over four datasets, including the initial model in �<sub>max</sub>. All configurations complete 12 evaluations except MovieLens-1M Prompt+Summary (11); comparing all MovieLens-1M configurations at evaluation 11 gives unchanged frontiers. With $g _ { F } , g _ { P } , g _ { S } , g _ { P + S }$ denoting Full, Prompt, Summary, and Prompt+Summary gains, $g _ { F } \ : - \ : g _ { P } \ : - \ : g _ { S } \ : + \ : g _ { P + S } \ : = \ : 0 . 6 0 6 4$ percentage points measures how feedback changes the benefit of explicit mechanism choices.

Feedback comparisons. Raw, Summary, and Mechanism share histories, legal actions, selection, implementation, and evaluation. Raw passes records; Summary gives research advice; Mechanism connects interventions and outcomes to follow-up hypotheses. Summary and Mechanism receive identical facts and source diagnostics with gpt-5.6-terra, medium reasoning, and a 20,000-token output limit. The ten comparisons use frozen checkpoints from real MoRec/CDs research runs. For example, States 1–2 come from rounds 2–3 of the Full component-study search, and State 3 from round 18 of a continued Mechanism Feedback run.

## C.1 BERT4Rec Mechanism Hypotheses and Components

RO3 builds on label smoothing, which raised development NDCG@5 from 0.4045 to 0.4101, and responds to a residual-head regression by testing a new matching geometry. With query $\mathbf { q } _ { j } ,$ tied item embeddings $E ,$ and existing output bias b, its shared training/prediction scorer is

$$
\begin{array} { r } { \pmb { \mathscr { s } } _ { j } = ( \pmb { \mathscr { q } } _ { j } W ) E ^ { \top } + \pmb { \mathscr { b } } , \qquad W \in \mathbb { R } ^ { 6 4 \times 6 4 } . } \end{array}\tag{5}
$$

The projection has no added bias. For $p _ { j } = \mathrm { s o f t m a x } ( \mathsf { s } _ { j } )$ , target $y _ { j } ,$ catalog size �, and valid positions $\boldsymbol { \mathcal { V } } _ { : }$

$$
\mathcal { L } = - \frac { 1 } { | \mathcal { V } | } \sum _ { j \in \mathcal { V } } \left[ 0 . 9 \log \mathcal { P } j , y _ { j } + \frac { 0 . 1 } { N } \sum _ { i = 1 } ^ { N } \log \mathcal { P } j , i \right] .\tag{6}
$$

Padding is excluded; encoder, embeddings, and original output mapping remain unchanged. The hypothesis predicts that learning the matching geometry will complement the smoothed objective.

Component comparison design. Each column in Table 11 compares scoring and objective changes individually and together in one discovered design. RO1 also varies layer aggregation, giving eight combinations. These comparisons test how an intervention’s contribution depends on accompanying mechanisms.

Table 11: Component comparisons for BERT4Rec/Electronics designs RO1–RO3. A: layer aggregation; S: scoring change; O: full-catalog label smoothing.
<table><tr><td>Design</td><td>RO1</td><td>RO2</td><td>RO3</td></tr><tr><td>Initial</td><td>0.4079</td><td>0.4047</td><td>0.4045</td></tr><tr><td>S</td><td>0.4158</td><td>0.4379</td><td>0.4170</td></tr><tr><td>0</td><td>0.4107</td><td>0.4073</td><td>0.4101</td></tr><tr><td> $S + \mathrm { O }$ </td><td>0.4185</td><td>0.4398</td><td>0.4220</td></tr><tr><td>A</td><td>0.4098</td><td>一</td><td>一</td></tr><tr><td> $\mathsf { A } + \mathsf { S }$ </td><td>0.4144</td><td>一</td><td>一</td></tr><tr><td> $\mathrm { \sf ~ A } + \mathrm { \sf ~ O }$ </td><td>0.4133</td><td>一</td><td>一</td></tr><tr><td> $\mathrm { A } + \mathrm { S } + \mathrm { O }$ </td><td>0.4199</td><td>一</td><td>一</td></tr></table>

S is bilinear projection for RO1/RO3 and cosine-temperature scoring for RO2. A applies only to RO1.

Component contributions. RO3’s combination reaches 0.4220, exceeding projection alone by 0.0051 and smoothing alone by 0.0119. Its interaction, $q _ { S O } - q _ { S } - q _ { O } + q _ { 0 } = - 0 . 0 0 0 5 ,$ is approximately additive. RO1 combines two-layer averaging (A), bilinear scoring (S), and smoothing (O). Its 0.4199 exceeds all seven other configurations. With S fixed, A changes quality by −0.0014 without O but +0.0015 with O, yielding conditional interaction $q _ { A S O } - q _ { A S } - q _ { S O } + q _ { S } = 0 . 0 0 2 9 .$ . RO2’s cosine scorer (temperature 0.2) reaches 0.4379; smoothing adds 0.0019, reaching 0.4398, with interaction −0.0007. Component comparisons use fixed training seeds 51006, 51007, and 51008 for RO1, RO2, and RO3, respectively.

## D Task and LLM Configurations

Data and training. Electronics and Ofice use Amazon Reviews 2014 5-core subsets [19]; Table 9 specifies splits and ranking protocols. Baseline search capabilities are preserved except web access. Within each starting-model setting, all systems share training and checkpoint selection; outer algorithm selection uses development NDCG@5. SASRec: embedding 50, two causal blocks/two heads, length 200, dropout 0.2; Adam 10<sup>−3</sup>, betas (0.9, 0.98), batch 256, 300 epochs, evaluation every five epochs, patience four. One uniform negative per positive excludes training history.

BERT4Rec: two layers/heads, hidden/feed-forward 64/256, length 50, masking/dropout 0.2, learning rate $1 0 ^ { - 3 }$ , weight decay zero, train/evaluation batch 128/64; 100 epochs, NDCG@5 checks every five epochs, patience four. MoRec: batch 256, hidden 50, length 200, dropout 0.2, embedding regularization 0.005; Adam $1 0 ^ { - 3 }$ , betas (0.9, 0.98), 300 epochs. Checkpoints are selected by development HR@5 every five epochs with patience four. NGCF: learning rate $1 0 ^ { - 3 } :$ , NDCG@5 each epoch, patience 30; batch/epochs 128/300 for CDs and Ofice, 1024/300 for Electronics, 1024/500 for MovieLens-1M.

Hardware and LLMs. Training and evaluation use NVIDIA RTX 3080 Ti GPUs. All discovery systems use GPT-5.6 Terra/Luna. Table 12 gives the role assignments, with T for Terra and L for Luna. Research calls use medium reasoning with temperature left unset. Output limits follow the task and role configurations.

Table 12: Role-specific LLM assignments.
<table><tr><td>System</td><td>Request roles</td></tr><tr><td>RecHarness</td><td>T: controller, implementation; L: research, critic.</td></tr><tr><td>DeepEvolve</td><td>T: planner, developer, reflection, writer; L: reader, search, debugger.</td></tr><tr><td>OpenEvolve</td><td>Generation: T and L.</td></tr><tr><td>ShinkaEvolve</td><td>Query: T and L; embedding:</td></tr><tr><td>Self-EvolveRec</td><td>text-embedding-3-small. T: development/debugging; L: research, diagnosis</td></tr><tr><td>RecRSI</td><td>research, feedback. L: Composer, Refiner, Falsification; T: Frontier,</td></tr></table>

## E Research History and Subsequent Candidates

With history and History reset start from the same MoRec/CDs incumbent (NDCG@5 0.4181), with prior experiments available or cleared. Access to research history improves candidate quality across all three successive decisions (Table 13). At the final step, the history branch keeps the scorer and clips item-embedding gradients to norm 0.5; the reset branch replaces BCE with pairwise logistic loss.

Table 13: Research history improves successive candidate quality on MoRec/CDs. Values are development NDCG@5.
<table><tr><td>Round</td><td>With history</td><td>History reset</td><td>Difference</td></tr><tr><td>13</td><td>0.4021</td><td>0.3686</td><td>+0.0334</td></tr><tr><td>14</td><td>0.3669</td><td>0.3405</td><td>+0.0265</td></tr><tr><td>15</td><td>0.4152</td><td>0.3953</td><td>+0.0199</td></tr><tr><td>Mean</td><td>0.3947</td><td>0.3681</td><td>+0.0266</td></tr></table>