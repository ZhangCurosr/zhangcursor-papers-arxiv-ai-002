# ONTOACT: HYPOTHESIS-BASED NEURO-SYMBOLIC AGENTS THROUGH ONTOLOGICAL ABSTRACTION

Donghyeok Lee<sup>1</sup>, Bong Gyun Kang<sup>1</sup>, Junyong Ahn<sup>1</sup> & Sungroh Yoon<sup>1,2</sup>

<sup>1</sup>Interdisciplinary Program in Artificial Intelligence, Seoul National University, Seoul, South Korea <sup>2</sup>Department of Electrical and Computer Engineering, Seoul National University, Seoul, South Korea Correspondence to: Sungroh Yoon <sryoon@snu.ac.kr>

![](images/984f513efa2adbe223047425cefa9f6417f225810c842b2748ea2543290054ad.jpg)  
Figure 1: The LLM proposes, the ontology verifies. (a) OntoAct abstracts recorded experience into world knowledge over typed entities, relations, and conditions. (b) Symbolic checks assess proposed hypotheses and actions against current evidence. (c) Feedback guides observation, action revision, and hypothesis assessment.

## ABSTRACT

Learning from experience requires recovering reusable knowledge and determining when it applies to a new situation. We introduce OntoAct, a hypothesis-based neurosymbolic framework built on ontological abstraction. OntoAct turns individual experiences into axioms over entity types, relations, and conditions, making their underlying patterns reusable across situations. A large language model (LLM) uses this world knowledge to propose hypotheses and actions, while a symbolic verifier assesses their supporting conditions against current observations. Verification identifies missing evidence, and new observations and intervention outcomes update hypothesis assessments and guide subsequent actions. We also introduce Tacit-FLE, a benchmark built on the Factorio Learning Environment (FLE) for diagnosing and repairing hidden faults in existing factories using inherited operational records. Its synthetic maintenance corpus distributes evidence across records, and an executed repair may leave the fault unresolved, requiring agents to reassess their diagnoses through interaction. Across Tacit-FLE, Mars, and ALFWorld, OntoAct achieves leading task performance with efficient interaction and inference.

## 1 INTRODUCTION

“We can know more than we can tell.”

— Michael Polanyi, The Tacit Dimension (Polanyi, 1966)

Agents often need to act in environments whose operating principles are only partially specified. Inherited logs, notes, and interaction trajectories record what happened and what was done. They may leave unstated the dependencies and conditions that made an action useful, or when that experience applies elsewhere. We call this tacit operational knowledge: knowledge expressed through recorded observations, interventions, and outcomes but left implicit as reusable principles.

Using this experience requires bridging two gaps. First, the agent must recover patterns that generalize beyond the entities named in individual records. A particular machine–storage connection, for example, instantiates a supplier–recipient relationship; reusable rules express conditions over the entity types and these relational roles. Second, it must determine whether those relationships and conditions hold in its current situation. Both gaps matter whether experience is retrieved as text or encoded as executable or symbolic rules (Lewis et al., 2020; Zhou et al., 2025b; Choi et al., 2025). The agent needs explicit hypotheses that it can test through observation and intervention, retain when supported, and revise when contradicted.

We introduce OntoAct, a neuro-symbolic framework that connects these tasks through ontological abstraction. It retains incident-specific entities and relations while expressing reusable axioms over types, relational roles, and operating conditions, so the same pattern can be instantiated for different entities. A large language model (LLM) proposes hypotheses and actions from this world knowledge; a symbolic verifier checks their support against current observations (Figure 1). Feedback identifies supported hypotheses and conditions requiring further observation. New observations and intervention outcomes guide hypothesis revision and subsequent actions. The guiding principle is simple: the LLM proposes, the ontology verifies.

To evaluate knowledge recovery and use, we also introduce Tacit-FLE, a maintenance benchmark built on the Factorio Learning Environment (Hopkins et al., 2025). It asks agents to diagnose hidden faults in existing plants from a finite corpus of synthetic maintenance logs and operating documents, together with live observations (Figure A1). Evidence is distributed across records, and an executed repair may leave the fault unresolved. Success therefore requires selecting useful evidence and translating it into an effective repair. The benchmark evaluates how agents combine inherited knowledge with feedback from their own observations and interventions. Our contributions are:

• Tacit-FLE (Section 2), a benchmark for diagnosis and repair from inherited operational records, designed to test the combination of distributed evidence with observations and intervention outcomes;

• OntoAct (Section 3), a framework that turns individual experiences into reusable ontological world knowledge and uses it to form, test, and revise hypotheses that guide interaction;

• Performance and efficiency (Section 4), demonstrating leading task performance and efficient interaction across Tacit-FLE, Mars, and ALFWorld. We further support these results with a range of analyses across experimental settings.

We introduce Tacit-FLE in Section 2 and present the OntoAct framework in Section 3. Section 4 evaluates the framework on Tacit-FLE and existing benchmarks, followed by related work in Section 5 and conclusions in Section 6.

## 2 TACIT-FLE: A BENCHMARK FOR TACIT KNOWLEDGE

Tacit-FLE evaluates how agents recover dependencies and repair conditions from past incidents and apply them to new faults. Built on the factory simulation of FLE (Hopkins et al., 2025), it pairs existing plants with inherited maintenance records and supports diagnosis through observation and intervention (Appendix A, Figure A1).

## 2.1 TASK SETTING

Existing plant. An episode starts with one hidden fault in an existing plant. The agent receives unit kinds, positions, and operating readings. Inspections reveal further details, while production evolves as actions advance simulation time (Appendix A.1).

Reference corpus. Each plant comes with a fixed, finite corpus D of synthetic maintenance logs and operating documents, provided before evaluation. The logs describe local observations, checks, repairs, and outcomes; operating documents state selected expectations, such as planned outputs. These sources support inference of component dependencies and repair conditions.

Diagnosis and repair. Within a fixed turn budget, the agent inspects units, waits, or attempts replacements or refills. A replacement restores the unit’s original specification, including configuration. Success requires replacing an accepted repair target or completing a valid refill of an accepted storage target with the required material. Failed attempts consume turns and provide feedback; episodes end upon success or budget exhaustion.

## 2.2 FAULTS AND DIAGNOSTIC EVIDENCE

Faults include supply interruptions and off-plan configurations, motivated by component failures and departures from intended operation in process diagnosis (Venkatasubramanian et al., 2003). Drawing on model-based diagnosis (de Kleer & Williams, 1987; 1989), we construct three task types that differ in how inherited records support a new diagnosis. Appendix A.3 provides examples of the records and observations in each setting.

 Chain. Local incidents provide evidence of dependencies between components. For example, one log pairs a stoppage at B with a failure at A; another pairs a stoppage at C with a failure at B. Combining the incidents suggests a dependency path A → B → C. When C stalls in a new episode, this path guides upstream inspections, where current readings help locate a supply interruption.

« Conjunction. A repair precedent, an operating plan, and a current observation play complementary roles. For example, a past log records that replacing machine A resolved a mismatch between its configured and planned output. In a new episode, the plan specifies product X for machine B, while inspection shows B configured for Y. Recognizing the shared mismatch condition connects a repair demonstrated on A to a candidate intervention on B.

Z Elimination. Records describe destroyed units, supply interruptions, and power failures, but omit the off-plan configuration encountered in evaluation. The agent uses recorded symptoms and checks to rule out explanations inconsistent with current observations. Further observations or interventions help distinguish the remaining possibilities.

Construction and evaluation. Plants, fault cases, and reference corpora are generated from a common specification (Appendix A.2), with the evaluation population and scale-specific settings detailed in Appendix A.4 and construction checks in Appendix A.5. Records exclude repairs of evaluation targets; consistent plant-specific identifiers reduce shortcuts from Factorio vocabulary. We report repair success, mean turns, and spurious interventions (executed repairs that do not meet the success criterion); Appendix A.4.1 gives their definitions and aggregation rules.

## 3 ONTOACT: ONTOLOGICAL ABSTRACTION FOR ACTION

OntoAct turns inherited experience into reusable world knowledge for assessing proposals during interaction. Following the LLMproposes, the ontology verifies, the LLM proposes axioms, hypotheses, and actions, while a symbolic verifier checks them against supporting evidence using the ontology’s types, relations, and conditions. This division governs both knowledge construction and action (Figure 2).

## 3.1 ONTOLOGICAL ABSTRACTION

Reusing an incident requires identifying the relations and conditions that make it relevant to another situation. Ontological abstraction expresses these requirements over entity types and relational roles while retaining the particular facts from which they were drawn.

(a) Constructing world knowledge  
![](images/5b1358abdbb460f330b94e031157cad428c6d082ba1c49f0cf041da44595071b.jpg)

![](images/03d165acd08d8217e35797ef35d746add26ce89b4f9b73e385764fde10d63562.jpg)  
Figure 2: The OntoAct framework. (a) The LLM proposes axioms from a reference corpus; symbolic admission checks their conformance to the schema and applicable evidence constraints. (b) The LLM proposes claims, hypotheses, and actions using the resulting world knowledge; ontologybased verification supplies feedback for revision, while observations and action outcomes update the reasoning context.

World knowledge. An ontology schema $T$ defines entity types, relations, observable attributes, and admissible axiom forms. The corpus-derived ontology separates ground assertions $A _ { W }$ from conditional axioms $R _ { W }$ , which express action requirements and diagnostic patterns:

$$
K _ { W } = ( T , A _ { W } , R _ { W } ) .\tag{1}
$$

Episode evidence. Applying these reusable conditions requires evidence from the current situation. At step $t , f _ { \mathrm { g r o u n d } }$ converts observations into episode assertions $A _ { t }$ . These assertions augment the corpus-derived facts, yielding the knowledge used to assess hypotheses and actions:

$$
K _ { t } = ( T , A _ { W } \cup A _ { t } , R _ { W } ) .\tag{2}
$$

This combines reusable knowledge with current evidence; see Appendix B.3.1 for a detailed example.

## 3.2 CONSTRUCTING WORLD KNOWLEDGE

To build this world knowledge, OntoAct processes the reference corpus D before interaction. The LLM proposes candidate axioms, and the ontology checks which ones enter $K _ { W }$

The LLM proposes axioms. For a record batch $b \subseteq D .$ , the LLM proposes ground assertions and conditional axioms in the forms allowed by T:

$$
\widetilde { X } _ { b } = f _ { \mathrm { a b s } } ( T , b ) = \{ ( x _ { i } , \pi _ { i } ) \} _ { i = 1 } ^ { n _ { b } } ,\tag{3}
$$

Each candidate conditional axiom $x _ { i }$ is paired with source records $\pi _ { i } ; n _ { b }$ counts all candidates.

Algorithm 1 Constructing world knowledge   
Require: ontology schema T, corpus D   
1: $K _ { W } \gets ( T , \widetilde { \emptyset } , \emptyset )$   
2: for each batch $b \subseteq D$ do   
3: $\widetilde { X } _ { b } \gets f _ { \mathrm { a b s } } ( T , b )$ ▷ LLM; Eq. 3   
4: for each candidate $( x _ { i } , \pi _ { i } ) \in \widetilde { X } _ { b }$ do   
5: $\alpha _ { i } \gets \mathrm { A }$ dmissible $( x _ { i } , \pi _ { i } ; K _ { W } , D )$ ▷ Eq. 4b   
6: if $\alpha _ { i } = 1$ then   
7: $K _ { W } \gets f _ { \mathrm { s t o r e } } ( K _ { W } , x _ { i } , \pi _ { i } )$ ▷ Store pair   
8: end if   
9: end for   
10: end for   
11: $K _ { W } \gets f _ { \mathrm { c o r p u s } } ( K _ { W } , D )$ ▷ Corpus checks   
12: return $K _ { W }$

The ontology constrains admission. Admission requires schema and source validity $( \beta _ { i } )$ and a set $\mathcal { C } _ { x _ { i } }$ of tests for its axiom type:

$$
\beta _ { i } = \mathrm { T y p e d } ( x _ { i } ; T ) { \bf 1 } [ \emptyset \neq \pi _ { i } \subseteq D ] ,\tag{4a}
$$

$$
\alpha _ { i } = \mathrm { A d m i s s i b l e } ( x _ { i } , \pi _ { i } ; K _ { W } , D ) = \beta _ { i } \prod _ { c \in { \mathcal C } _ { x _ { i } } } { \bf 1 } [ c ( x _ { i } , \pi _ { i } ; K _ { W } , D ) ] .\tag{4b}
$$

Here Typed checks schema conformance and $\mathbf { 1 } [ P ]$ indicates whether P holds. Every test $c \in { \mathcal { C } } _ { x _ { : } }$ must pass; Appendix B.2 specifies the tests for each environment. Admitted axioms $( \alpha _ { i } = 1 )$ enter $A _ { W }$ or $R _ { W }$ with their sources via $f _ { \mathrm { s t o r e } } ; f _ { \mathrm { c o r p u s } }$ checks the assembled knowledge (Algorithm 1).

## 3.3 REASONING AND ACTING WITH WORLD KNOWLEDGE

The LLM proposes claims and actions. Let A, K, and H denote current assertions, knowledge, and hypotheses; primes mark updates. Given goal g and feedback η (initially empty each turn), the LLM proposes claims $\Delta \dot { A }$ , hypotheses $\Delta H$ and action a:

$$
( \Delta A , \Delta H , a ) = f _ { \mathrm { p r o p o s e } } ( g , K , H , \eta ) .\tag{5}
$$

Structural claims specify relations. Each hypothesis h specifies required relations and nonempty expected conditions $\Phi ( h )$ , instantiating an induced pattern or proposing a new explanation.

Checking claims against current evidence. For episode claims, $f _ { \mathrm { a d m i t } }$ checks types, structural evidence, and correspondence with cited patterns, including every required condition. Admitted structural claims update knowledge:

$$
{ \cal A } ^ { \prime } = A \cup f _ { \mathrm { a d m i t } } ( \Delta A ; K ) ,\tag{6a}
$$

$$
K ^ { \prime } = ( T , A _ { W } \cup A ^ { \prime } , R _ { W } ) .\tag{6b}
$$

Algorithm 2 Reasoning and acting   
Require: $K _ { W }$ , goal $^ { g , }$ initial observation o<sub>1</sub>   
Require: B turns; up to J proposals per turn   
1: $A , H , a _ { \mathrm { p r e v } }  \bar { \partial }$   
2: for $t = 1 , \ldots , B { \bf d o }$   
3: $A  f _ { \mathrm { g r o u n d } } ( A , o _ { t } )$ ▷ Observations   
4: $K  ( \mathcal { \tilde { T } } , A _ { W } \cup A , R _ { W } )$   
5: $H \gets f _ { \mathrm { e v a l } } ( H ; K , a _ { \mathrm { p r e v } } )$ ▷ Reassess hypotheses   
6: $\eta  \emptyset$   
7: for $j = 1 , \dots , J$ do   
8: $( \bar { \Delta } A , \Delta H , a ) \gets f _ { \mathrm { p r o p o s e } } ( g , K , H , \eta ) \ \triangleright \mathrm { E q . } 5$   
9: $\dot { A } \gets A \cup f _ { \mathrm { a d m i t } } ( \dot { \Delta } A ; K )$ ▷ Eq. 6a   
10: $K  ( T , A _ { W } \cup \dot { A } , R _ { W } )$ ▷ Eq. 6b   
11: $H \gets \dot { f } _ { \mathrm { e v a l } } ( H \cup f _ { \mathrm { a d m i t } } ( \Delta H ; K ) ; K )$ ▷ Eq. 8   
12: $( v , \eta )  f _ { \mathrm { v e r i f y } } ( a ; K , H )$ ▷ Eq. 9   
13: $\mathbf { i f } \ v \neq \perp$ then break   
14: end if   
15: end for   
16: $( a _ { \mathrm { p r e v } } , o _ { t + 1 } ) \gets f _ { \mathrm { e x e c u t e } } ( a , v ) \quad \vartriangle { \vartriangleright } \triangleright \mathrm { A c t }$ ; observe   
17: if terminated then break   
18: end if   
19: end for

When h instantiates pattern r with conditions $\Phi _ { r } , \Phi ( h )$ retains every condition, substituting current entities. Each atom $\varphi = ( u , p , \mathrm { o p } , z )$ tests entity u’s observable attribute p against expected value z with comparison op; current readings yield:

$$
[ \varphi ] _ { K ^ { \prime } } = \left\{ \begin{array} { l l } { \mathsf { T } , } & { \mathrm { r e s o l v e d e v i d e n c e ~ s a t i s f i e s ~ } p ( u ) \mathrm { ~ o p ~ } z , } \\ { \bot , } & { \mathrm { r e s o l v e d ~ e v i d e n c e ~ c o n t r a d i c t s ~ } p ( u ) \mathrm { ~ o p ~ } z , } \\ { \ ? , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

Missing readings remain unknown $( ? ) { \mathrm { . } }$ and the same evaluations test action conditions. An open, admitted hypothesis receives the status:

$$
\operatorname { S t a t u s } ( h )  \{ { \begin{array} { l l } { { \mathrm { r e f u t e d } } , } & { { \exists } \varphi \in \Phi ( h ) : [ \varphi ] _ { K ^ { \prime } } = \bot , } \\ { { \mathrm { c o n f i r m e d } } , } & { \forall \varphi \in \Phi ( h ) : [ \varphi ] _ { K ^ { \prime } } = \top , } \\ { { \mathrm { o p e n } } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} }\tag{8}
$$

Confirmation records observational support, subject to refutation by later readings or repair outcomes; updated hypotheses form $H ^ { \prime }$

The ontology verifies the action. The verifier tests $R _ { W } \colon$ violated conditions and prohibitions block actions; missing readings alone do not. An action requiring explanatory support needs a confirmed hypothesis whose conditions still hold; observations need no confirmed hypothesis. It returns:

$$
( v , \eta ) = f _ { \mathrm { v e r i f y } } ( a ; K ^ { \prime } , H ^ { \prime } ) ,\tag{9}
$$

A blocked verdict $v = \bot$ requests revision while budget remains; η reports failed and unresolved checks. Algorithm 2 summarizes this interaction loop. Appendix B.3 details hypothesis admission, condition evaluation, repair verification, and refutation by observations and outcomes.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Environments and records. Tacit-FLE tests hidden-fault repair across three task types and plant scales (Section 2). Mars (Tang et al., 2024b) tests survival and crafting under altered world mechanics; we use the eight-world configuration of Zhou et al. (2025b). ALFWorld (Shridhar et al., 2021) tests household instruction following on 134 unseen tasks. Corpus-based methods share plant maintenance and operating records, Mars exploration records, or ALFWorld training-game trajectories; Appendix E.1 details these corpora and evaluation tasks.

Baselines. We compare OntoAct with ReAct (Yao et al., 2023), retrieval-augmented $\mathbf { R e A c t } ^ { * }$ (Lewis et al., 2020), corpus-adapted Reflexion (Shinn et al., 2023), and IfR (Tang et al., 2024b), which uses natural-language rules. NeSyC (Choi et al., 2025) and WALL-E (Zhou et al., 2025b) check actions against induced answer-set programs and executable rules, respectively. Implementations and prompts are detailed in Appendix C.2.

Models and protocol. We use GPT-5.6 Luna (OpenAI, 2026a) with reasoning effort none. Medium Tacit-FLE and Mars also use Luna with medium effort (Luna<sup>R</sup>) and GPT-5.6 Terra (Terra; none) (OpenAI, 2026b). Methods share tasks and interaction budgets; Appendix E details settings, budgets, and metrics.

## 4.2 TACIT-FLE

How well does OntoAct repair hidden faults? Failed repairs consume turns without ending an episode, so eventual success can mask repeated unsuccessful interventions. OntoAct achieves the highest overall success at every scale with 70–91% fewer spurious interventions than IfR, the strongest baseline by overall success (Table 1). It also uses the fewest turns among methods with successful repairs. Its largest advantage is on chain tasks requiring dependency inference acros records; IfR matches or exceeds it on elimination.

Table 1: Repair performance on Tacit-FLE. Success (%) by type and overall (All); SI: mean spurious interventions; Turns: mean turns. Failed episodes use the full budget (Appendix A.4.1). Bold: best; underlined: second-best overall success, interventions, and turns. Intervention and turn rankings require at least one repair.
<table><tr><td colspan="7">Repair success (%) ↑</td></tr><tr><td>Scale</td><td>Method</td><td> $\pmb { \mathcal { O } }$  Chain</td><td>Conjunction</td><td>Y Elimination</td><td>All</td><td>SI↓</td><td>Turns ↓</td></tr><tr><td rowspan="7">Small</td><td>ReAct</td><td>0.0</td><td>71.4</td><td>75.0</td><td>39.3</td><td>4.6</td><td>22.2</td></tr><tr><td>ReAct*</td><td>15.4</td><td>85.7</td><td>37.5</td><td>39.3</td><td>7.2</td><td>22.9</td></tr><tr><td>Reflexion</td><td>7.7</td><td>71.4</td><td>75.0</td><td>42.9</td><td>5.3</td><td>23.2</td></tr><tr><td>IfR</td><td>38.5</td><td>100.0</td><td>75.0</td><td>64.3</td><td>6.4</td><td>17.4</td></tr><tr><td>NeSyC</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>30.0</td></tr><tr><td>WALL-E</td><td>7.7</td><td>14.3</td><td>12.5</td><td>10.7</td><td>7.1</td><td>29.0</td></tr><tr><td>OntoAct (ours)</td><td>69.2</td><td>100.0</td><td>75.0</td><td>78.6</td><td>1.9</td><td>11.8</td></tr><tr><td rowspan="7">Medium</td><td>ReAct</td><td>28.6</td><td>64.3</td><td>75.0</td><td>52.8</td><td>11.3</td><td>41.8</td></tr><tr><td>ReAct</td><td>21.4</td><td>64.3</td><td>87.5</td><td>52.8</td><td>13.4</td><td>38.7</td></tr><tr><td>Reflexion</td><td>21.4</td><td>42.9</td><td>37.5</td><td>33.3</td><td>15.4</td><td>46.4</td></tr><tr><td>IfR</td><td>28.6</td><td>85.7</td><td>100.0</td><td>66.7</td><td>11.9</td><td>31.7</td></tr><tr><td>NeSyC</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>60.0</td></tr><tr><td>WALL-E</td><td>14.3</td><td>28.6</td><td>87.5</td><td>36.1</td><td>10.1</td><td>48.0</td></tr><tr><td>OntoAct (ours)</td><td>57.1</td><td>100.0</td><td>100.0</td><td>83.3</td><td>2.4</td><td>19.4</td></tr><tr><td rowspan="8">Large</td><td>ReAct</td><td>5.9</td><td>72.7</td><td>75.0</td><td>45.0</td><td>16.1</td><td>49.9</td></tr><tr><td>ReAct</td><td>23.5</td><td>36.4</td><td>66.7</td><td>40.0</td><td>19.6</td><td>56.4</td></tr><tr><td>Reflexion</td><td>11.8</td><td>9.1</td><td>50.0</td><td>22.5</td><td>23.2</td><td>63.1</td></tr><tr><td>IfR</td><td>23.5</td><td>90.9</td><td>100.0</td><td>65.0</td><td>17.8</td><td>42.6</td></tr><tr><td>NeSyC</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>75.0</td></tr><tr><td>WALL-E</td><td>5.9</td><td>45.5</td><td>75.0</td><td>37.5</td><td>14.9</td><td>56.6</td></tr><tr><td>OntoAct (ours)</td><td>82.4</td><td>100.0</td><td>91.7</td><td>90.0</td><td>1.6</td><td>17.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/d98e4d954c00b7578a0854209fc8cc18286f40762cde0f94fc2ee86b4487ead0.jpg)

Figure 3: Hypothesis precision over turns. Target precision for all tracked hypotheses before selection (Init.) and those retained +1/+2/+3 turns after admission or at the final episode turn.  
![](images/74c404990b98950bde3d6a4041017f8dd5b1ef253c2c489fb1baac604f356209.jpg)  
Figure 4: Repair success and SI. Medium Tacit-FLE, 36 tasks. Bars: success (left); markers: mean SI (right, lower is better). R: medium reasoning. Full metrics are in Table A15.

Does diagnosis narrow toward the true fault? OntoAct proposes fewer replacement targets than WALL-E and IfR, yet covers the true fault more often, overall and within its first three candidates (Table 2). Across scales, 71.4–90.0% of targetaligned hypotheses remain and 95.6–99.6% of off-target hypotheses are refuted (Figure 3; Table A27). Appendix D.7 details this refinement from observations and unsuccessful repairs.

Table 2: Candidate count and fault coverage. Medium Tacit-FLE (36 tasks); targets include unexecuted proposals. Coverage counts tasks with the true fault among all candidates or the first three.
<table><tr><td></td><td>OntoAct</td><td>WALL-E</td><td>IfR</td></tr><tr><td>Distinct candidates/episode</td><td>3.36</td><td>5.14</td><td>9.00</td></tr><tr><td>Fault in candidate set</td><td>32/36</td><td>17/36</td><td>24/36</td></tr><tr><td>Fault in first 3</td><td>25/36</td><td>10/36</td><td>4/36</td></tr></table>

Does OntoAct benefit from reasoning and stronger backbones? OntoAct benefits from both: reasoning and Terra raise success from 83.3% to 97.2% and 100.0%, while SI falls from 2.39 to 0.75 and 0.44, respectively (69% and 81% reductions; Figure 4). ReAct and ReAct<sup>\*</sup> achieve higher success with Terra than with added reasoning, whereas Reflexion and IfR benefit more from reasoning. Yet higher success can mask repeated failed repairs: with reasoning or Terra, these baselines still incur 7.9–13.7 spurious interventions per episode. OntoAct also reduces mean turns from 19.4 to 8.4 in both conditions (Table A15).

## 4.3 MARS

How well does OntoAct adapt to altered mechanics? OntoAct supports action under changed prerequisites, extending the framework’s utility beyond hidden-fault diagnosis. It achieves the highest mean Reward and Score across the seven modified worlds (Table 3), although combined changes remain difficult. Inspecting NeSyC’s low-scoring runs reveals overly restrictive learned conditions that block planning and trigger repeated waiting; Appendix D.2 details these failures.

Compared with ReAct<sup>\*</sup>, the strongest baseline by mean Score, OntoAct achieves about 50% higher Score with 45% fewer evaluation tokens per environment step (0.84k versus 1.54k; Figure 5). Compact learned conditions shorten input contexts at similar LLM call rates. Appendix D.6 analyzes context usage and reports construction costs separately.

![](images/af434618e43d658ad9e6510fb12d4ae4ad1e5e05ea4b1be2f9c8f65425c4c9b3.jpg)  
Figure 5: Mars Score and evaluation tokens. Luna, seven modified worlds. Equal world weights; input/output tokens exclude knowledge construction.

Table 3: Mars Reward and Score by world. Luna; five evaluation episodes per world. Reward combines achievement rewards and health changes (mean ± sample SD); Score geometrically aggregates achievement success rates. Average excludes Default; its Reward SD is computed across matched-seed world averages. Bold/underlined: best/second-best. Metric definitions are in Appendix E.3.
<table><tr><td>World</td><td>ReAct</td><td> $\mathrm { R e A c t } ^ { * }$ </td><td>Reflexion</td><td>IfR</td><td>NeSyC</td><td>WALL-E</td><td>OntoAct</td></tr><tr><td colspan="8">Reward ↑</td></tr><tr><td>Default</td><td> $1 0 . 1 0 { \pm } 4 . 8 0 \ $ </td><td> $9 . 3 0 { \pm } 1 . 6 4 \ $ </td><td> $7 . 9 0 \pm 2 . 9 5$ </td><td> $8 . 1 0 { \pm } 2 . 3 5 $ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $6 . 3 0 { \scriptstyle \pm 6 . 0 6 }$ </td><td> $\mathbf { 1 2 . 1 0 { \pm } 1 . 2 2 }$ </td></tr><tr><td>Terrain</td><td> $8 . 7 0 { \pm } 2 . 3 0 $ </td><td> $\underline { { 1 2 . 1 6 \pm 4 . 0 3 } }$ </td><td> $8 . 6 8 \pm 5 . 6 6$ </td><td> $9 . 3 0 { \pm } 3 . 9 6 \ $ </td><td> $- 0 . 9 0 { \pm } 0 . 0 0$ </td><td> $7 . 5 0 { \pm } 3 . 7 8 $ </td><td> $\mathbf { 1 2 . 3 0 \pm 1 . 9 2 }$ </td></tr><tr><td>Survival</td><td> $7 . 1 0 { \pm } 4 . 0 0 \ $ </td><td> $\overline { { 9 . 5 0 { \pm } 3 . 4 4 } }$ </td><td> $6 . 3 0 { \scriptstyle \pm 5 . 0 7 }$ </td><td> $7 . 3 0 { \pm } 4 . 8 2 $ </td><td> $0 . 9 8 { \pm } 0 . 9 9$ </td><td> $6 . 3 0 { \pm } 4 . 9 7$ </td><td> $\mathbf { 1 0 . 4 6 \pm 6 . 3 9 }$ </td></tr><tr><td>Task dep.</td><td> $4 . 1 0 { \pm } 1 . 0 0 \ $ </td><td> $\overline { { 2 . 9 0 \pm 1 . 3 0 } }$ </td><td> $4 . 1 0 { \pm } 2 . 9 2 \ $ </td><td> $7 . 3 0 { \pm } 1 . 1 0 \ $ </td><td> $0 . 9 0 { \pm } 0 . 4 5$ </td><td> $3 . 9 0 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $\mathbf { 1 1 . 3 0 \pm 1 . 1 0 }$ </td></tr><tr><td>Terr. surv.</td><td> $6 . 1 0 { \pm } 2 . 3 5 $ </td><td> $\mathbf { 8 . 1 0 \pm 2 . 3 5 }$ </td><td> $5 . 1 0 { \pm } 2 . 9 2$ </td><td> $6 . 1 0 { \pm } 4 . 3 0 \ $ </td><td> $2 . 7 0 { \pm } 1 . 8 2 $ </td><td> $3 . 9 0 { \pm } 3 . 1 1 $ </td><td> $6 . 3 0 { \pm } 3 . 0 3 $ </td></tr><tr><td>Terr. task</td><td> $3 . 5 0 { \pm } 1 . 6 7$ </td><td> $\underline { { 6 . 9 0 \pm 2 . 1 7 } }$ </td><td> $6 . 4 8 { \pm } 2 . 5 6 $ </td><td> $3 . 9 0 { \pm } 2 . 0 5$ </td><td> $1 . 1 0 { \pm } 1 . 0 0 \ $ </td><td> $2 . 9 0 { \pm } 1 . 6 4 \ $ </td><td> $\overline { { 7 . 7 0 \pm 2 . 8 8 } }$ </td></tr><tr><td>Surv. task</td><td> ${ \bf 6 . 5 0 \pm 6 . 1 1 }$ </td><td> $\overline { { 2 . 1 0 \pm 2 . 7 4 } }$ </td><td> $3 . 7 0 { \pm } 2 . 4 1$ </td><td> $\mathbf { 6 . 5 0 \pm 4 . 0 4 }$ </td><td> $2 . 3 0 { \pm } 1 . 3 0 \ $ </td><td> ${ \underline { { 5 . 7 0 } } } { \underline { { \pm 5 . 1 8 } } }$ </td><td> $5 . 5 0 { \pm } 6 . 0 2$ </td></tr><tr><td>All three</td><td> $0 . 9 0 { \pm } 2 . 6 8 \ $ </td><td> $\mathbf { 1 . 5 0 \pm 2 . 6 1 }$ </td><td> $0 . 5 0 { \pm } 1 . 1 4 $ </td><td> $- 0 . 1 0 { \pm } 1 . 3 0 $ </td><td> $- 0 . 9 0 { \pm } 0 . 0 0$ </td><td> $\overline { { 0 . 7 0 \pm 1 . 5 2 } }$ </td><td> $\underline { { 1 . 1 0 \pm 2 . 1 2 } }$ </td></tr><tr><td>Average</td><td> $5 . 2 7 { \pm } 1 . 1 0 $ </td><td> $\underline { { 6 . 1 7 \pm 1 . 3 1 } }$ </td><td> $4 . 9 8 \pm 1 . 4 0 $ </td><td> $5 . 7 6 { \pm 2 . 1 4 }$ </td><td> $0 . 8 8 { \pm } 0 . 3 7$ </td><td> $4 . 4 1 \pm 1 . 9 1$ </td><td> ${ \bf 7 . 8 1 \pm 1 . 7 5 }$ </td></tr><tr><td colspan="8">Score (%) ↑</td></tr><tr><td>Default</td><td>31.85</td><td>13.54</td><td>13.67</td><td>11.74</td><td>0.69</td><td>15.26</td><td>29.70</td></tr><tr><td>Terrain</td><td>12.71</td><td>32.88</td><td>19.29</td><td>22.61</td><td>0.00</td><td>13.32</td><td>34.34</td></tr><tr><td>Survival</td><td>11.10</td><td>18.29</td><td>11.58</td><td>14.79</td><td>1.15</td><td>11.94</td><td>25.83</td></tr><tr><td>Task dep.</td><td>3.26</td><td>2.01</td><td>4.46</td><td>8.44</td><td>0.51</td><td>2.69</td><td>20.73</td></tr><tr><td>Terr. surv.</td><td>6.49</td><td>10.52</td><td>5.79</td><td>10.12</td><td>2.30</td><td>5.10</td><td>7.63</td></tr><tr><td>Terr. task</td><td>2.65</td><td>6.93</td><td>6.31</td><td>3.21</td><td>0.76</td><td>2.36</td><td>10.17</td></tr><tr><td>Surv. task</td><td>17.65</td><td>2.18</td><td>3.19</td><td>7.09</td><td>1.49</td><td>8.73</td><td>11.50</td></tr><tr><td>All three</td><td>1.80</td><td>1.69</td><td>0.67</td><td>0.74</td><td>0.00</td><td>1.19</td><td>1.27</td></tr><tr><td>Average</td><td>7.95</td><td>10.64</td><td>7.33</td><td>9.57</td><td>0.89</td><td>6.47</td><td>15.92</td></tr></table>

Does OntoAct benefit from more capable models? OntoAct’s gains from reasoning and stronger backbones persist across environments, even though baseline responses change (Table 4). Unlike in Tacit-FLE, all displayed Mars baselines benefit from both changes; IfR narrows the gap under reasoning. OntoAct still retains the highest mean Score, supporting the framework’s ability to build on model improvements in both diagnosis and action under altered mechanics. Full results appear in Appendix D.2.

Table 4: Mars Score across model conditions. Mean Score (%) over seven modified worlds. Bold/underlined: best/second-best per condition.
<table><tr><td>Method</td><td>Luna</td><td> $\mathrm { L u n a } ^ { R }$ </td><td>Terra</td></tr><tr><td>ReAct</td><td>7.95</td><td>13.06</td><td>13.91</td></tr><tr><td>ReAct</td><td>10.64</td><td>15.23</td><td>16.48</td></tr><tr><td>Reflexion</td><td>7.33</td><td>12.91</td><td>12.09</td></tr><tr><td>IfR</td><td>9.57</td><td>20.60</td><td>18.86</td></tr><tr><td>WALL-E</td><td>6.47</td><td>9.93</td><td>7.79</td></tr><tr><td>OntoAct (ours)</td><td>15.92</td><td>22.99</td><td>21.57</td></tr></table>

Do learned conditions guide actions before execution? OntoAct checks candidate actions against learned rules before execution. For resource conditions learned correctly by both methods, OntoAct redirects all 70 violating initial candidates before execution, whereas IfR executes 20 violating commands and replans after their failures (Table 5). The distinction is when knowledge changes behavior: OntoAct checks represented requirements against the current state before spending an environment interaction. None of its 136 issued commands violates these shared conditions. These traces cover matched knowledge on different trajectories; broader resource-rule coverage remains

Table 5: Violations of shared learned conditions. Resource conditions correctly stored by both methods. Luna, seven modified worlds, 35 episodes per method. Fractions use each method’s eligible candidates or commands at the specified stage; Appendix D.4 details matching and denominators.
<table><tr><td></td><td>OntoAct</td><td>IfR</td></tr><tr><td rowspan="2">Initial candidate violations</td><td>70/245</td><td>20/107</td></tr><tr><td>(28.6%)</td><td>(18.7%)</td></tr><tr><td rowspan="2">Redirected before execution</td><td>70/70 (100%)</td><td>0/20 (0%)</td></tr><tr><td>0/136</td><td>20/107</td></tr><tr><td>Issued command violations</td><td>(0%)</td><td>(18.7%)</td></tr></table>

incomplete. Appendix D.4 details the shared-condition matching, trajectory analysis, and coverage limits.

## 4.4 ALFWORLD

Can OntoAct maintain high success with fewer LLM calls? OntoAct completes 132/134 tasks versus 129/134 for WALL-E, using 52.5% fewer LLM calls and 32.6% fewer evaluation tokens (Table 6). Most savings arise from OntoAct’s symbolic condition checks versus WALL-E’s extra LLM outcome-prediction calls (1.02 versus 1.95 calls per environment step); shorter trajectories also contribute. NeSyC’s 26.1% uses disclosed initial object

Table 6: ALFWorld success and evaluation cost. Luna. SR: success (%); mean environment Steps, LLM Calls, and Tokens (k) per episode. <sup>†</sup>Initial object locations and inventory provided.
<table><tr><td>Method</td><td>SR (%)</td><td>Steps</td><td>Calls</td><td>Tokens (k)</td></tr><tr><td>NeSyC†</td><td>26.1</td><td>38.6</td><td>9.3</td><td>281.32</td></tr><tr><td>WALL-E</td><td>96.3</td><td>14.8</td><td>28.8</td><td>47.70</td></tr><tr><td>OntoAct (ours)</td><td>98.5</td><td>13.5</td><td>13.7</td><td>32.16</td></tr></table>

locations and inventory (†); Appendix D.5 compares observation conditions and LLM call roles.   
Task-family results and separate knowledge-construction costs appear in Appendices D.5 and D.6.

## 5 RELATED WORK

Neuro-symbolic LLM agents. Neuro-symbolic agents keep what an LLM learns about an environment in a checkable form, as code world models and planning domains (Tang et al., 2024a; Khan et al., 2026; Zhu et al., 2025), dependency graphs (Nottingham et al., 2023), or rule bases (Chen et al., 2024), each refined against the agent’s own interaction. WALL-E learns code rules and a state graph from mismatches between the predictions of an LLM world model and observed transitions, and lets the rules veto actions (Zhou et al., 2025b); NeSyC learns and continually repairs symbolic knowledge in answer-set form with a hypothesis-testing loop (Choi et al., 2025); IfR induces natural-language rules from experience in Mars (Tang et al., 2024b). A complementary line separates proposing from checking: solvers decide what an LLM formalizes (Pan et al., 2023; Olausson et al., 2023; Ye et al., 2023), and external critics or runtime guards check the plans and actions it proposes (Kambhampati et al., 2024; Chen et al., 2025; Xiang et al., 2025).

Acting with ontologies and knowledge graphs. Agents also use structured knowledge: ontologies mediate planning (John & Koopmann, 2024), scene and procedural graphs constrain LLM planners (Rana et al., 2023; Zhou et al., 2025a), knowledge graphs serve as memory (Anokhin et al., 2025) and as the reference against which LLM plans are verified (Petruzzellis et al., 2025), and such graphs are built from text, including maintenance records (van Cauter & Yakovets, 2024; Zhang & Soh, 2024). OntoAct joins these threads by inducing ontological world knowledge from inherited records, and Tacit-FLE evaluates the repairs that follow.

## 6 CONCLUSION

We introduced OntoAct, a hypothesis-based neuro-symbolic framework that connects learning from experience to action through ontological abstraction. It recovers reusable axioms over entity types, relations, and conditions from operational records. An LLM uses this knowledge to propose hypotheses and actions, while symbolic verification checks their support against current evidence. Observations and intervention outcomes continually refine hypothesis assessments and guide subsequent decisions. We also introduced Tacit-FLE to evaluate how agents combine inherited operational knowledge with observation and intervention to diagnose and repair hidden faults.

OntoAct achieves the highest overall repair success at every Tacit-FLE scale while substantially reducing spurious interventions. Reasoning and stronger backbones further increase success and reduce unsuccessful repairs. Results on Mars and ALFWorld extend these benefits to altered world mechanics and household tasks, demonstrating strong task performance alongside efficient interaction and inference. Our evaluation uses predefined ontology schemas and world mechanics that remain fixed during interaction. Future work will explore learning and extending these schemas from experience, and revising world knowledge as environments evolve. Together, these results show how reusable abstractions, explicit hypotheses, and ongoing verification connect accumulated experience to more successful and efficient action.

## ETHICS STATEMENT

All experiments run in simulators on synthetic data. The maintenance corpus of Tacit-FLE is generated and contains no record of any real plant, person, or organization. The work targets industrial maintenance, where an agent’s interventions can carry physical risk; the verifier of OntoAct is a step toward auditability, not a safety guarantee, and we make no claim of readiness for deployment.

## REPRODUCIBILITY STATEMENT

The benchmark generator, the agent, and every baseline port, with the seeds and prompts used in the experiments, will be released with the camera-ready version. Appendix A specifies the generator, Appendix C the bindings and baselines, and Appendix E the settings, budgets, and metric definitions behind each number.

## REFERENCES

Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Andrey Kravchenko, Mikhail Burtsev, and Evgeny Burnaev. AriGraph: Learning knowledge graph world models with episodic memory for LLM agents. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI), pp. 12–20, 2025. doi: 10.24963/ijcai.2025/2. URL https: //www.ijcai.org/proceedings/2025/2.

Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu, Binbin Lin, and Xiaofei He. AutoManual: Con structing instruction manuals by LLM agents via interactive environmental learning. In Advances in Neural Information Processing Systems, volume 37, pp. 589–631, 2024. doi: 10.52202/ 079017-0019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/0142921fad7ef9192bd87229cdafa9d4-Paper-Conference.pdf.

Zhaorun Chen, Mintong Kang, and Bo Li. ShieldAgent: Shielding agents via verifiable safety policy reasoning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pp. 8313–8344, 2025. URL https://proceedings.mlr.press/v267/ chen25ae.html.

Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, and Honguk Woo. NeSyC: A neuro-symbolic continual learner for complex embodied tasks in open domains. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ d569e4655e2835e3d38310b67c5ad646-Paper-Conference.pdf.

Johan de Kleer and Brian C. Williams. Diagnosing multiple faults. Artificial Intelligence, 32(1): 97–130, 1987. doi: 10.1016/0004-3702(87)90063-4.

Johan de Kleer and Brian C. Williams. Diagnosis with behavioral modes. In Proceedings of the Eleventh International Joint Conference on Artificial Intelligence (IJCAI), pp. 1324–1330, 1989. URL https://www.ijcai.org/Proceedings/89-2/Papers/076.pdf.

Jack Hopkins, Mart Bakler, and Akbir Khan. Factorio Learning Environment. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 38, 2025. doi: 10.52202/085713-2745. URL https://doi.org/10.52202/085713-2745.

Tobias John and Patrick Koopmann. Planning with OWL-DL ontologies. In ECAI 2024, Frontiers in Artificial Intelligence and Applications. IOS Press, 2024. doi: 10.3233/FAIA240988. URL https://doi.org/10.3233/FAIA240988.

Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Paul Saldyt, and Anil B Murthy. Position: LLMs can’t plan, but can help planning in LLM-modulo frameworks. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 22895–22907, 2024. URL https://proceedings.mlr.press/v235/kambhampati24a.html.

Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. One life to learn: Inferring symbolic world models for stochastic environments from unguided exploration. In International Conference on Learning Representations, pp. 99794–99836, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ a24d08f2eecc6c96f4a0ad37a74ddeca-Paper-Conference.pdf.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, 2020.

Kolby Nottingham, Prithviraj Ammanabrolu, Alane Suhr, Yejin Choi, Hannaneh Hajishirzi, Sameer Singh, and Roy Fox. Do embodied agents dream of pixelated sheep: Embodied decision making using language guided world modelling. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 26311–26325, 2023. URL https://proceedings. mlr.press/v202/nottingham23a.html.

Theo Olausson, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama, Joshua Tenenbaum, and Roger Levy. LINC: A neurosymbolic approach for logical reasoning by combining language models with first-order logic provers. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5153–5176, 2023. doi: 10.18653/v1/2023.emnlp-main.313. URL https://aclanthology.org/2023.emnlp-main.313/.

OpenAI. GPT-5.6 Luna. OpenAI API model documentation, 2026a. URL https://developers. openai.com/api/docs/models/gpt-5.6-luna. Accessed September 19, 2026.

OpenAI. GPT-5.6 Terra. OpenAI API model documentation, 2026b. URL https://developers. openai.com/api/docs/models/gpt-5.6-terra. Accessed September 19, 2026.

Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. Logic-LM: Empowering large language models with symbolic solvers for faithful logical reasoning. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pp. 3806–3824, 2023. doi: 10.18653/v1/2023. findings-emnlp.248. URL https://aclanthology.org/2023.findings-emnlp. 248/.

Flavio Petruzzellis, Cristina Cornelio, and Pietro Lio. Hierarchical planning for complex tasks\` with knowledge graph-RAG and symbolic verification. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, 2025.

Michael Polanyi. The Tacit Dimension. Doubleday, Garden City, NY, 1966.

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. SayPlan: Grounding large language models using 3D scene graphs for scalable robot task planning. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pp. 23–72, 2023. URL https://proceedings.mlr.press/v229/ rana23a.html.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://arxiv.org/ abs/2010.03768.

Hao Tang, Darren Key, and Kevin Ellis. WorldCoder, a model-based LLM agent: Building world models by writing code and interacting with the environment. In Advances in Neural Information Processing Systems, volume 37, pp. 70148–70212, 2024a. doi: 10.52202/ 079017-2243. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/820c61a0cd419163ccbd2c33b268816e-Paper-Conference.pdf.

Xiaojuan Tang, Jiaqi Li, Yitao Liang, Song-chun Zhu, Muhan Zhang, and Zilong Zheng. Mars: Situated inductive reasoning in an open-world environment. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024b. URL https://arxiv.org/ abs/2410.08126.

Zeno van Cauter and Nikolay Yakovets. Ontology-guided knowledge graph construction from maintenance short texts. In Proceedings ofthe 1st Workshop on Knowledge Graphs and Large Language Models (KaLLM), 2024. URL https://aclanthology.org/2024.kallm-1. 8/.

Venkat Venkatasubramanian, Raghunathan Rengaswamy, Kewen Yin, and Surya N. Kavuri. A review of process fault detection and diagnosis: Part i: Quantitative model-based methods. Computers & Chemical Engineering, 27(3):293–311, 2003. doi: 10.1016/S0098-1354(02)00160-6.

Zhen Xiang, Linzhi Zheng, Yanjie Li, Junyuan Hong, Qinbin Li, Han Xie, Jiawei Zhang, Zidi Xiong, Chulin Xie, Carl Yang, Dawn Song, and Bo Li. GuardAgent: Safeguard LLM agents via knowledgeenabled reasoning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pp. 68316–68342, 2025. URL https://proceedings.mlr.press/v267/ xiang25a.html.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Xi Ye, Qiaochu Chen, Isil Dillig, and Greg Durrett. SatLM: Satisfiability-aided language models using declarative prompting. In Advances in Neural Information Processing Systems, volume 36, pp. 45548–45580, 2023. doi: 10.52202/075280-1974. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html.

Bowen Zhang and Harold Soh. Extract, define, canonicalize: An LLM-based framework for knowledge graph construction. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 9820–9836, 2024. doi: 10.18653/v1/2024.emnlp-main.548. URL https://aclanthology.org/2024.emnlp-main.548/.

Jiacong Zhou, Jiaxu Miao, Xianyun Wang, and Jun Yu. Enhancing LLM planning for robotics manipulation through hierarchical procedural knowledge graphs. In Advances in Neural Information Processing Systems, volume 38, pp. 127466–127495, 2025a. doi: 10.52202/ 085713-4246. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/b94310e1c7ecb79f1a24adc757f1b89b-Paper-Conference.pdf.

Siyu Zhou, Tianyi Zhou, Yijun Yang, Guodong Long, Deheng Ye, Jing Jiang, and Chengqi Zhang. WALL-E: World alignment by neurosymbolic learning improves world model-based LLM agents. In Advances in Neural Information Processing Systems, volume 38, pp. 65352–65391, 2025b. doi: 10.52202/085713-2192. URL https://doi.org/10.52202/085713-2192.

Wang Bill Zhu, Ishika Singh, Robin Jia, and Jesse Thomason. Language models can infer action semantics for symbolic planners from environment feedback. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 8751–8773, 2025. doi: 10.18653/v1/2025. naacl-long.440. URL https://aclanthology.org/2025.naacl-long.440/.

## A TACIT-FLE: SPECIFICATION, EXAMPLES, AND VALIDATION

This appendix specifies the benchmark introduced in Section 2. GitHub repository: will be released.

Section Contents A.1 Agent inputs, observations, and actions A.2 Plant generation and reference corpus construction A.3 Illustrative examples of the three task types A.4 Evaluation population and scoring A.5 Benchmark construction checks

## A.1 TASK AND INFORMATION AVAILABLE TO THE AGENT

A world $w = ( P , D )$ pairs a plant P with a fixed reference corpus D. An episode $e = ( w , c ^ { * } , B )$ introduces one hidden fault $c ^ { * }$ and allows B interaction turns. The agent receives the corpus, a task brief, a unit roster, and observations of the faulty plant. The agent interacts with the plant through Tacit-FLE’s text commands for inspection and repair. Tacit-FLE runs on the Factorio server infrastructure of FLE and adds a plant generator, fault injection, a maintenance corpus, a text command interface, and repair evaluation. Whereas the troubleshooting suite of FLE evaluates spatial error identification in existing factories, Tacit-FLE evaluates diagnosis and executed repairs from finite maintenance records.

## A.1.1 OBSERVATIONS AND ANONYMIZATION

The task brief gives the turn budget and the plant’s normal production rate in final products per minute; the roster lists unit kinds and positions. A public status board reports machine run states, storage totals and changes, elapsed game time, and shipping counts. Power and material-transfer devices require inspection, while opening a unit returns itemized contents. Conjunction also exposes the configured output for comparison with its operating plan. Table A1 separates available information from evaluator metadata. Figure A2 illustrates the initial input and a subsequent command response.

Table A1: Information boundary. Public records can disclose selected planned facts; the complete original specification remains evaluator metadata.
<table><tr><td>Source</td><td>Available information</td></tr><tr><td>Initial task</td><td>Turn budget, normal production rate (final products/minute), unit kinds, and coordinates.</td></tr><tr><td>Inherited records</td><td>Local incident observations, checks, repairs, and outcomes; buffer policy for Chain; planned outputs for Conjunction.</td></tr><tr><td>Current observations</td><td>Machine states, storage totals and changes, shipping; inspections and itemized readings on request.</td></tr><tr><td>Repair feedback</td><td>Episode termination after an accepted repair; continued interaction otherwise.</td></tr><tr><td>Evaluator only</td><td>Injected fault, accepted repair targets, full original plant specification, and coverage annotations.</td></tr></table>

Items, entities, recipes, and fluids are anonymized with consistent world-specific labels across records and observations. Unit kinds, positions, structural suffixes, and measured states remain visible. This reduces reliance on familiar Factorio names while retaining spatial and operational cues.

## A.1.2 ACTIONS AND REPAIR FEEDBACK

In Factorio, production evolves with simulation time: machines process inputs, materials move between components, and inventories and shipping counts change. Table A2 lists the available commands and the game time each advances. A replacement restores a unit to its original specification, including its configuration; a refill specifies the storage, item, and amount. Valid actions advance the simulation by the durations in Table A2, and each response reflects the updated plant state. Malformed commands consume a turn without advancing the plant. An accepted repair ends the episode; otherwise interaction continues and supplies further evidence.

![](images/e6f90ac791097c3e7691f7d5454fb6f621861cc154915be43a0aa165be5dcfa3.jpg)  
Figure A1: Tacit-FLE. (a) An agent inherits an existing plant, a finite reference corpus of maintenance logs and operating documents, and observations of a new fault. (b)  Chain links dependencies across incidents; « Conjunction combines repair precedents, plans, and observations; Elimination rules out known explanations contradicted by current evidence.

![](images/fd6c3ba5de10ec07e4eaddb4f6f6a09be8a70319183a399215a4ceb323e2846b.jpg)  
Figure A2: Example interaction in Tacit-FLE. A medium-scale Conjunction episode: the task brief, roster, initial status, and a later open command with its response. Only the selected lines are shown; identifiers and wording are preserved.

## A.2 CONSTRUCTING PLANTS AND INHERITED RECORDS

A common seeded specification ties the simulated plant to its synthetic records. Construction has four components:

1. Generate the plant. Select target products according to the scale profile, then trace Factorio’s recipe graph backward to identify the required intermediate products and raw inputs. Compile these dependencies into a spatial layout of machines, input and output storage, material-transfer devices, and power infrastructure. The world seed controls product selection and layout choices, including shared production stages and spatial spread.

Table A2: Agent actions in the evaluated protocol. Each action uses one turn. The plant advances by the indicated game time; LLM deliberation itself is not charged as game time. The status board is included in observations at no additional turn cost.
<table><tr><td>Command</td><td>Game seconds</td><td>Effect or returned information</td></tr><tr><td>inspect &lt;unit&gt;</td><td>5</td><td>External state; power and material-transfer device readings require this probe.</td></tr><tr><td>open &lt;unit&gt;</td><td>10</td><td>Itemized storage or input contents; configured output in Conjunction.</td></tr><tr><td>wait &lt;seconds&gt;</td><td>specified</td><td>Advance the plant and receive an updated status board.</td></tr><tr><td>replace &lt;unit&gt;</td><td>15</td><td>Restore the addressed unit to its original specification.</td></tr><tr><td>refill &lt;storage&gt;</td><td>10</td><td>Insert the selected item into storage; success also</td></tr><tr><td>&lt;item&gt; &lt;count&gt;</td><td></td><td>depends on the repair target and item.</td></tr></table>

2. Establish normal operation. Build the plant in Factorio and measure positive healthy shipping.

3. Introduce a fault. Inject a supply cut for Chain or an off-plan machine configuration for Conjunction and Elimination. Cap downstream inventory so it cannot mask the fault indefinitely, and require stalled shipping or a persistent telemetry deviation. Other product lines may continue operating.

4. Construct the reference corpus. Render historical incidents and operating documents from the same specification, withhold evaluation incidents, and check the intended distribution of evidence.

Appendix A.5 summarizes the checks applied to generated plants and corpora.

Example plant topologies. Figures A3–A4, A5–A6, and A7–A8 show small, medium, and large plants, respectively. Four selected examples per scale illustrate variation in production depth, branch ing, and shared intermediate stages. Arrows indicate material flow; dots join branches, and curved crossings are unconnected. Small squares denote input and output buffers. Positions are rearranged for readability, and utilities are omitted. Anonymized unit identifiers are shown above original names in italics, which are included only as a reading aid.

![](images/143cc6891aea292411d22f00aab6fbc203bb57f950a58abe61fa95d77e30db49.jpg)  
(a) World 47728

![](images/7f86adc1e38990c618739c51cddbd00a9d8c32b9d59cd1b999cdf547f15c31ef.jpg)  
(b) World 22970  
Figure A3: Small-scale plant examples (a)–(b). Each plant has one target product. One input stream splits and reconverges in (a), while (b) combines iron and copper through a circuit stage.

![](images/757da288a840722c0737ae2d3b7af398c2de22cc47e5c6bdc840fca8b2cb9254.jpg)  
(c) World 62689

![](images/7c0f5a1d14a562cc2698b73b3617a65478d7d13a2db3b4b651b037ef39271e17.jpg)  
(d) World 50676  
Figure A4: Small-scale plant examples (c)–(d). Each plant has one target product. A deeper assembly sequence in (c) contrasts with several material branches converging on one science pack in (d).

![](images/1d5daf778224c41f6178f88fcb8b9fcd7a99c6ad51373b921cd7430b314667f6.jpg)  
(a) World 14275

![](images/1c60c6f79bec57a7cca7d9bbb992c89e536a0dbabee11fad75690b5a2befbf4a.jpg)  
(b) World 73144  
Figure A5: Medium-scale plant examples (a)–(b). Each plant has two target products. An independent short production line accompanies the deeper chain in (a); the two targets reuse upstream products in (b).

![](images/a2082ba1af0062e3aed449b903bc84ec161da16ff000b5c52279e7685f723284.jpg)  
(c) World 22606

![](images/29d11797f73d17f3b06659a113301e43307c086e52837fe00ab4bef64a5e03cd.jpg)  
(d) World 16060  
Figure A6: Medium-scale plant examples (c)–(d). Each plant has two target products. Parallel material branches in (c) contrast with the longer transport and engine chains in (d).

![](images/4d613d126e28f8e327d2d180546dc0ad7a08bd40f5bdf8f52d128a9b566b5187.jpg)  
(a) World 5246

![](images/94bb4c4e1212ff6049c14d0644f4ba1184d5694708a5a0997550355a73175822.jpg)  
(b) World 44309  
Figure A7: Large-scale plant examples (a)–(b). Each plant has four target products. Shorter parallel paths in (a) contrast with the reuse of a compact upstream circuit and belt network in (b).

![](images/f309459257873ed72c2269514999e7ebc7c63c0c4094dd77eda47b4f1986c8ce.jpg)  
(c) World 64320

![](images/d703725c7633959743993b1e9acd51fa767074fb0fdb4166c3e316c6c977087b.jpg)  
(d) World 20024  
Figure A8: Large-scale plant examples (c)–(d). Each plant has four target products. Direct shipping accompanies further assembly in (c), while (d) contains fifteen processing units across several intermediate stages.

## A.2.1 CORPUS CONTENTS AND PROVENANCE

Each maintenance log contains five fields: OCCURRED, OBSERVED, CHECKED, ACTION, and RESULT. These describe local symptoms, checks, and repairs, leaving relations and conditions to be inferred across records. The corpus also includes explicit operating information: Chain supplies a buffer-policy note, and Conjunction supplies planned outputs.

Three construction details matter for interpreting these records:

• Synthetic provenance. Historical incidents are rendered from templates over the same plant specification.

• Document counts. Required incidents and distractors are supplemented to the configured corpus size; a recurring incident may appear under separate log identifiers. Chain can add dependency records and its operating note afterward. Table A6 reports the resulting document counts, including these additions and repetitions.

• Held-out faults. No registered test fault appears as a historical repair target in the shared corpus. The same unit may still appear as a downstream observation, roster entry, or planned-output entry. Records are shuffled, and the Conjunction plan lists applicable machines rather than singling out the faulty one.

## A.3 THREE TASK TYPES: ILLUSTRATIVE EXAMPLES

Tables A3–A5 pair excerpts from the maintenance logs and operating documents with environment responses to the agent’s commands. Each table pairs original document and response excerpts with an interpretation of their diagnostic relevance.

•  Chain: compose dependencies. Local records connect successive units. A separate bufferpolicy note distinguishes normally empty internal buffers from yard-stocked intakes whose lack of incoming material indicates a supply cut.

• « Conjunction: combine complementary evidence. A prior incident supplies a repair pattern, a plan supplies the expected output, and a current reading supplies the actual output. The mismatch is unit-specific: an item label alone does not indicate a fault.

• Z Elimination: test alternative explanations. Records describe destroyed units, supply cuts, and power failures, but omit the off-plan configuration encountered in evaluation. Rejecting documented explanations narrows the search; further observations or interventions may still be needed to distinguish remaining candidates.

Table A3:  Chain: an illustrative example. Record excerpts and command responses from a medium-scale evaluation case.
<table><tr><td>Evidence source</td><td>Exact excerpt</td></tr><tr><td>Log log-6056</td><td>[OBSERVED] mat-193 stopped, input buffer empty. [CHECKED] site visit — mat-205 found destroyed.</td></tr><tr><td>Log log-7299</td><td>[OBSERVED] mat-228 stopped, input buffer empty. [CHECKED] site visit — mat-193 found destroyed.</td></tr><tr><td>Ops note buffer-policy</td><td>[NORM] Line-internal feed buffers are pulled just-in-time and normally sit empty; an empty internal buffer during a stall is expected, not evidence. [NORM] Raw-material intake buffers are kept stocked by the yard at all times. An intake buffer found empty with nothing arriving means its supply feed is severed</td></tr><tr><td>Command responses</td><td>— replace that storage. mat-205_in_W opened: empty.flat at zero.</td></tr><tr><td>Interpretation</td><td>The two logs connect mat-205 to mat-193 and mat-193 to mat-228. The operating</td></tr></table>

Table A4: « Conjunction: an illustrative example. Record excerpts and command responses from a medium-scale evaluation case.
<table><tr><td>Evidence source</td><td>Exact excerpt</td></tr><tr><td>Log log-5841</td><td>[CHECKED] opened st-04 — configured output does not match the plan. [ACTION] replaced st-04 with a fresh unit (as-built</td></tr><tr><td>Plan norm_3982</td><td>configuration restored). - unit st-05:planned output mat-231</td></tr><tr><td>Command responses</td><td>st-05 opened: input buffer empty.configured output: mat-232.</td></tr><tr><td>Interpretation</td><td>A repair precedent on st-04 applies to a different unit: st-05 is planned to output mat-231 but currently outputs mat-232.</td></tr></table>

Table A5: Z Elimination: an illustrative example. Record excerpts and command responses from a medium-scale evaluation case.
<table><tr><td>Evidence source</td><td>Exact excerpt</td></tr><tr><td>Log log-7360</td><td>[CHECKED] site visit — mat-027 found destroyed.</td></tr><tr><td>Log log-8680</td><td>[CHECKED] opened storage mat-070_in_W — found empty, no inbound material.</td></tr><tr><td rowspan="2">Log log-3569</td><td>[OBSERVED] multiple machines in the zone stopped at</td></tr><tr><td>once. [CHECKED] zone-wide simultaneous stoppage — power</td></tr><tr><td rowspan="2">Command responses</td><td>distribution loss confirmed. mat-027 opened:input buffer mat-089 x5, mat-074 x3.</td></tr><tr><td>mat-027:old unit removed, replacement installed to plan. [SHIFT] not fixed yet; the shift goes on. st-00 opened:input buffer empty.</td></tr><tr><td>Interpretation</td><td>mat-070: running. Replacing mat-027 did not end the episode, which rejects the destroyed-unit explanation of log-7360. A running machine (mat-070) is incompatible with the zone-wide power loss of log_3569. The stopped st-00 matches none of the recorded incidents and becomes the unit to investigate.</td></tr></table>

## A.4 EVALUATION POPULATION AND SCORING

Each combination of scale and task type has ten worlds: the first five are reserved for development, and the last five for evaluation. All registered fault positions in the evaluation worlds are tested, giving 45 world–type corpora and 104 episodes (Table A6). For each world, we rebuild the same plant and inject each registered fault in a separate episode, keeping its reference corpus fixed. Small, medium, and large use one, two, and four target products, respectively; corpus size, realized topology, task mix, and turn budgets also vary.

Table A6: Evaluated Tacit-FLE profiles. Document counts include maintenance logs and operating documents in each corpus. Episodes correspond to the specified fault positions; B is the turn budget. Scale conditions vary jointly in target products, corpus sizes, and interaction budgets.
<table><tr><td>Scale</td><td>Products</td><td>Type</td><td>Worlds</td><td>Documents/world</td><td>Episodes</td><td>B</td></tr><tr><td>Small</td><td>1</td><td>Chain</td><td>5</td><td>7-16</td><td>13</td><td>30</td></tr><tr><td></td><td></td><td>Conjunction</td><td>5</td><td>5</td><td>7</td><td>30</td></tr><tr><td></td><td></td><td>Elimination</td><td>5</td><td>4-5</td><td>8</td><td>30</td></tr><tr><td>Medium</td><td>2</td><td>Chain</td><td>5</td><td>32-40</td><td>14</td><td>60</td></tr><tr><td></td><td></td><td>Conjunction</td><td>5</td><td>30</td><td>14</td><td>60</td></tr><tr><td></td><td></td><td>Elimination</td><td>5</td><td>30</td><td>8</td><td>60</td></tr><tr><td>Large</td><td>4</td><td>Chain</td><td>5</td><td>63-72</td><td>17</td><td>75</td></tr><tr><td></td><td></td><td>Conjunction</td><td>5</td><td>60</td><td>11</td><td>75</td></tr><tr><td></td><td></td><td>Elimination</td><td>5</td><td>60</td><td>12</td><td>75</td></tr></table>

A method may construct reusable knowledge from the corpus or retrieve records according to its evaluation procedure. No additional exploratory training episodes are supplied. Corpus-derived knowledge may be shared across a world’s episodes, but test observations, hypotheses, and outcomes are reset each time. The no-corpus control omits the records; method-specific access is detailed in Appendix C.2.

## A.4.1 REPAIR CRITERION AND METRICS

The evaluator retains an accepted repair-target set that is hidden from the agent. Replacing an accepted target completes the episode. A refill must address an accepted target, be accepted by the environment, and use the planned supply item when specified. Otherwise the attempt consumes a turn and the episode continues, until a repair succeeds or the budget is exhausted.

• Repair success: the fraction of scheduled evaluation episodes with an accepted repair within the turn budget.

• Turns: mean interaction turns, including the full budget for unsuccessful episodes.

• Spurious interventions: mean executed replacements or refills per episode that fail the repair criterion, including an unsuccessful refill of an otherwise relevant unit.

LLM calls and tokens are accounted for separately from environment turns (Appendix D.6).

## A.5 BENCHMARK CONSTRUCTION CHECKS

These checks assess whether generated cases realize the intended task type: an operating plant develops an observable fault, and its corpus provides the specified evidence. Table A7 summarizes the checks on plant behavior, corpus construction, and reproducibility.

Table A7: Benchmark construction checks. The checks cover simulated plant behavior, corpus conditions, and reproducible evaluation inputs.
<table><tr><td>Check</td><td>Evidence</td><td>What it establishes</td></tr><tr><td>Plant behavior</td><td>Healthy production, followed by stalled shipping or a persistent change in operating readings after fault injection.</td><td>The injected fault has an observable effect on an operating plant.</td></tr><tr><td>Corpus construction</td><td>Required document fields and vocabulary, declared evidence coverage, and exclusion of test-fault repair records.</td><td>Compliance with the corpus construction conditions.</td></tr><tr><td>Reproducibility</td><td>Exact regeneration of all 45 evaluation corpora.</td><td>Consistent corpus inputs for the 104 evaluation episodes.</td></tr></table>

## B METHOD DETAILS

## B.1 REPRESENTATION AND INFORMATION ACCOUNTING

The world knowledge is $K _ { W } = ( T , A _ { W } , R _ { W } )$ , with an ontology schema $T = ( \mathcal { C } , \mathcal { R } , \mathcal { P } , \Theta )$ . Here C is the set of entity types, R the relations, P the observable attributes, and Θ the admissible axiom templates with their evaluation procedures. The induced axioms comprise ground assertions in $A _ { W } ,$ which preserve particular entities and their connections, and conditional axioms in $R _ { W }$ . Equation 2 combines these with the episode assertions $A _ { t } ;$ Algorithm 2 uses the same tuple with the within-turn variables A and $K .$ . The templates differ by environment, as listed in Table ${ \bf A } { \bar { 1 3 } } .$ . The predicate Typed in Equation 4a checks conformance to this vocabulary and its admissible axiom forms; empirical support is evaluated separately.

An induced element retains a provenance set $\pi \subseteq D$ identifying its source records. Some sources are attached automatically by the induction procedure, while other templates require the LLM to cite record identifiers. Source association provides traceability; the additional support and compatibility checks are described below. Table A8 distinguishes fixed representation choices from induced knowledge and current observations. The experimental corpus and observation conditions, including the separately reported observation control, are specified in Appendix E.1.

Table A8: Information accounting for OntoAct. The origins and roles of information used in construction and interaction.
<table><tr><td>Origin</td><td>Content</td><td>Role</td></tr><tr><td>Task inputs</td><td>Reference corpus, task description, observations, action interface</td><td>Use and any observation controls follow the experimental protocol.</td></tr><tr><td>Given representation</td><td>Schema T, observation map  $\rho ,$  template semantics</td><td>Method-specific representation and interpretation choices; fixed before learning.</td></tr><tr><td>Induced knowledge</td><td> $A _ { W } , R _ { W }$  , source records  $\pi \subseteq D$ </td><td>Constructed from the reference corpus and fixed during evaluation.</td></tr><tr><td>Episode evidence</td><td> $A _ { t }$ </td><td>Grounded in public observations; reset for each episode.</td></tr><tr><td>LLM proposals</td><td>Action a; structural and diagnostic claims where supported</td><td>Checked against the current knowledge and evidence.</td></tr></table>

Knowledge lifetime. The corpus-derived world knowledge is reused without learning from test episodes. Each episode starts from that knowledge and obtains a fresh state $A _ { t }$ from observations. Historical status readings are not copied into the live state; roster information can correct an entity’s type in the episode’s working representation. The distinction between world knowledge and episode state concerns lifetime and evidential scope. Both may contain assertions about individuals.

## B.2 KNOWLEDGE CONSTRUCTION AND ADMISSION

Shared procedure. This section expands Algorithm 1 and Equation 4b into the checks used to admit corpus-derived knowledge in each environment. $f _ { \mathrm { a b s } }$ in Equation 3 denotes the proposal procedure, including the association of each candidate with its source records. The admission indicator Admissible in Equation 4b requires schema and source validity $( \beta _ { i } )$ , then applies the constraints $\mathcal { C } _ { x }$ supported by the knowledge type. The indicator ${ \bf 1 } [ P ]$ converts a satisfied condition to one and a failed condition to zero; the product in Equation 4b is one exactly when every applicable criterion holds. The membership and semantics of $\mathcal { C } _ { x }$ differ by knowledge type: source-link requirements, structural consistency, empirical support, and contradiction tolerances are specified below. $f _ { \mathrm { c o r p u s } }$ denotes validation against the assembled records: some checks run during admission, while others require a later pass over the assembled knowledge. Bounded correction requests within knowledge construction are included in the proposal procedure. The deterministic checks run without LLM calls.

Tacit-FLE: structure and diagnostic patterns. The LLM processes each reference document using the supplied vocabulary and proposes entities, structural relations, diagnostic patterns, and nonevidence axioms. Entity types and relation names are closed; relation domain and range constraints and incompatible component types are checked when proposals enter the store. The current document identifier is attached to its extracted facts and patterns. A diagnostic pattern specifies the symptom type and status, the cause’s relation to that symptom, optional cause type and readings, and a repair action when the documented intervention has a corresponding command. For example, an incomingflow condition is retained only when the source record supplies it. Relations preserve particular component identities, whereas the diagnostic pattern refers to types and roles.

When repair observations can be extracted from the cited records, a support check tests whether at least one cited incident instantiates the pattern’s symptom–cause relationship. An unsupported pattern is returned to the LLM once with the discrepancy; a revised proposal that still fails is discarded. Operating documents without a repair incident are not evaluated by this repair-specific test. After assembling the corpus, a further check removes self-cause patterns that would confirm themselves from the symptom alone when a recorded incident shows that symptom but documents a repair of another component.

Tacit-FLE: operating information. Operating practices and service-count samples are elicited from the joined corpus because their evidence can span documents. Practice templates express a forbidden action under a stated condition, a check before an intervention, or a preference between actions. They require a valid typed form and nonempty citations drawn from the presented corpus, but do not undergo the repair-pattern support test. Non-evidence axioms encode explicit operating notes that make a status non-diagnostic for a component class, with any stated exceptions. Service samples are checked against their cited records and aggregated by production duty into an empirical prior over usage counts.

Mars. The LLM reads the reference transitions and proposes typed action conditions and effects for collection, crafting, placement, movement, drinking, creatures, and material neighborhoods, together with blocked-action conditions. Admission checks the schema and requires at least one cited transition concerning the subject of the proposed axiom. Applicable recorded transitions are then checked for outcomes or effects that contradict it. The experiment configuration allows at most one contradictory transition; blocked-action rules additionally require recorded unsuccessful attempts without a recorded success under the claimed condition. This compatibility check is performed during admission.

ALFWorld. Records are grouped by action and processed in batches of at most 24 transitions. The LLM proposes necessary action conditions over five supplied predicates: being at the target, holding the object, having empty hands, the object’s presence at the target, and the target’s openness. Every cited index must identify a record of the corresponding action. At least one cited successful transition must observe and satisfy all proposed conditions, and no successful transition in the corpus may contradict them. These checks establish positive support and compatibility with recorded successes.

## B.3 DIAGNOSTIC HYPOTHESES AND VERIFICATION

Hypothesis representation. In the diagnostic variant, a hypothesis is written as $\begin{array} { r l } { h } & { { } = } \end{array}$ $( c , e , r , \Phi , \sigma , \pi , \alpha )$ . Its fields are a candidate cause c, the anomaly e it explains, a referenced pattern r, expected observational atoms Φ, required structural relations $\sigma ,$ cited evidence $\pi ,$ , and a prescribed repair α. The pattern can refer to an induced diagnostic pattern or aggregate prior, or describe an explanation proposed by the LLM beyond those patterns. An atom names an entity, an observable attribute, a comparison, and an expected value.

Admission of episode claims. $f _ { \mathrm { a d m i t } }$ checks the structured claims proposed during interaction. Named entities and anomalies must exist in the current representation, the expected attributes must be readable, and required relations must already be established. A newly proposed relation must first be supported by a permitted basis, such as documented structure, layout connectivity, or observed contents. For a pattern-backed diagnosis, admission also checks the correspondence between the proposed cause, the symptom, and the conditions of that pattern. A hypothesis cannot cite a pattern requiring no incoming material while omitting the corresponding expected reading. An operating note that makes a status non-evidence can invalidate a claim relying on that status.

Three-valued evaluation. The observation map $\rho$ grounds readings in the current state, and an atom evaluates as

$$
[ [ \varphi ] ] _ { K _ { t } } \in \{ { \top } , \bot , { \ ? } \} .
$$

The values denote support, contradiction, and an unresolved reading, respectively. Missing readings remain unknown. For a current status that disagrees with the claim, the implementation also considers recent readings and can leave an isolated fluctuation unresolved. An open, admitted hypothesis is refuted when an atom evaluates to ⊥, confirmed when all its atoms evaluate to ⊤, and remains open otherwise. For open hypotheses, unresolved atoms are mapped to available probes and included in the context supplied to the LLM. Confirmation records agreement with the specified observations.

Hypotheses beyond recorded patterns. The LLM may propose an explanation for a cause not covered by an induced pattern, provided that its claim meets the structural and observational admission requirements. The reasoning context also lists recorded explanations that still owe a reading, open rival hypotheses, and the stopped units that no recorded pattern explains; an explanation outside the recorded patterns is marked as supported when its cause is one of these unexplained units.

Checking repairs and incorporating outcomes. A repair is backed when an admitted LLMproposed hypothesis is confirmed, its nonempty set of atoms still holds, it prescribes the repair verb, and the target is named by the hypothesis’s cause or one of its atoms. Observation actions do not require a confirmed repair hypothesis; operating practices and repetition checks can nevertheless provide feedback on proposed commands. If an executed repair targets a hypothesis’s candidate cause and the episode continues, that outcome refutes the hypothesis on the next observation. A command refused by the environment is not counted as an executed repair. Confirmed hypotheses can also be refuted by subsequent contradictory readings or by the disappearance of the anomaly they explained. The LLM remains responsible for selecting a cause and an action from this evidence. When the proposal limit is reached, the final proposal and its verification verdict are passed to the executor.

## B.3.1 EXAMPLE: FROM CORPUS RECORDS TO ACTION

This example starts with a small reference corpus, shows the knowledge constructed from it, and follows its use in a new episode. The records and observation branches are condensed, fictional examples.

Step 1: read the records and construct world knowledge. Suppose the agent receives the following two documents, $D = \{ d _ { 1 } , d _ { 2 } \}$ , before evaluation.

Reference corpus.

$d _ { \mathrm { 1 } } \mathbf { : }$ : Line A handover note. “Storage $s _ { 1 }$ supplies iron plates to machine $m _ { 1 }$ . Mover $q _ { 1 }$ delivers plates into $s _ { 1 }$ and is its only incoming connection.”

$d _ { \mathrm { 2 } } \colon$ Line A maintenance record. “Machine $m _ { 1 }$ had stopped. Its supplying storage $s _ { 1 }$ contained no iron plates. Inspection of its sole incoming mover $q _ { 1 }$ found no material arriving. We added iron plates to $s _ { 1 } ; m _ { 1 }$ then resumed production.”

The ontology schema $T$ provides the types Storage, Machine, and Mover, the relation feeds, and the observable attributes. The LLM proposes a particular supply relation and a reusable diagnostic pattern. The relation has compatible endpoint types; the pattern’s cited incident matches its stoppedmachine symptom, supplying-storage cause, and refill intervention. After the applicable admission checks, $\dot { K _ { W } } \dot { = } ( T , A _ { W } ^ { - } , \dot { R } _ { W } )$ retains

$$
A _ { W } \supseteq \left\{ { \mathrm { f e e d s } } ( s _ { 1 } , m _ { 1 } ) \right\} \quad { \mathrm { s u p p o r t e d } } { \mathrm { b y } } d _ { 1 } , d _ { 2 } ,
$$

$$
R _ { W } \ni r \qquad { \mathrm { w i t h ~ s o u r c e ~ } } \pi _ { r } = \{ d _ { 2 } \} .
$$

The content of r makes the reusable roles explicit:

symptom: m : Machine is stopped,

candidate cause: s : Storage is depleted,

$$
\mathrm { r e q u i r e d ~ r e l a t i o n : } \ \sigma _ { r } ( s , m ) = \{ \mathrm { f e e d s } ( s , m ) \} ,
$$

$$
\mathrm { e x p e c t e d ~ r e a d i n g s : } ~ \Phi _ { r } ( s ) = \{ \mathrm { s t o c k } ( s ) = 0 , ~ \mathrm { i n f o w } ( s ) = 0 \} ,
$$

prescribed repair: refill(s).

The concrete identities remain in $A _ { W }$ , while r replaces them with roles that can match another storage–machine pair. The empty-stock and no-inflow conditions come from $d _ { 2 } ;$ they specify what to check when proposing the same explanation for a new pair. Here and below, inflow $\bar { = } 0$ means that no incoming material is observed.

Step 2: apply the pattern to a new situation. Now consider a different pair, $s _ { 2 }$ and $m _ { 2 }$ . The current layout and observations provide the following information.

Current episode.

Layout. Storage $s _ { 2 }$ supplies machine $m _ { 2 } ;$ mover $q _ { 2 }$ is the sole incoming mover connected to $s _ { 2 } .$ Observed. Machine $m _ { 2 }$ is stopped, and inspection of $s _ { 2 }$ finds zero iron plates.

Not yet observed. Mover $q _ { 2 }$ has not been inspected, so whether material is arriving remains unresolved.

The layout establishes feeds $( s _ { 2 } , m _ { 2 } )$ in the episode evidence $A _ { t }$ . The historical observation of $q _ { 1 }$ says nothing about the current inflow through $q _ { 2 } .$ . Using $r ,$ the LLM proposes $h \colon$ “Depletion of $s _ { 2 }$ may explain the stopped $m _ { 2 } ;$ the corresponding repair is to refill $s _ { 2 } . { \overset {  } { ^ { \prime } } }$ Substituting $s \mapsto s _ { 2 }$ and $m \mapsto m _ { 2 }$ gives

$$
\begin{array} { r l } & { \sigma ( h ) = \sigma _ { r } ( s _ { 2 } , m _ { 2 } ) = \{ \mathrm { f e e d s } ( s _ { 2 } , m _ { 2 } ) \} , } \\ & { \Phi ( h ) = \Phi _ { r } ( s _ { 2 } ) = \{ \underbrace { \mathrm { s t o c k } ( s _ { 2 } ) = 0 } _ { \varphi _ { \mathrm { s t o c k } } } , \underbrace { \mathrm { i n f l o w } ( s _ { 2 } ) = 0 } _ { \varphi _ { \mathrm { i n f l o w } } } \} . } \end{array}
$$

Each $\varphi$ is a condition to test; for example, $\varphi _ { \mathrm { s t o c k } }$ has the tuple form $( s _ { 2 } , \mathrm { s t o c k } , = , 0 )$ in Equation $^ { 7 . }$ The proposal retains both conditions from $d _ { 2 } .$ , including the inflow condition that has not yet been observed in this episode. Both attributes are observable, so the complete hypothesis can be admitted before every reading is available.

Table A9 contrasts a complete proposal with three incomplete proposals, holding the other admission requirements fixed and satisfied. For the missing-relation case, assume the layout and records do not establish the supply connection. A missing reading is compatible with admission; an unsupported relation or omitted required condition is not.

Table A9: Admission before observation-based assessment.
<table><tr><td>Case</td><td>Proposal</td><td>Admission and reason</td></tr><tr><td>Complete</td><td>Established supply relation; both required conditions retained; inflow still unread.</td><td>Admitted: Missing readings can be investigated after admission.</td></tr><tr><td>Missing relation</td><td>that has no supporting evidence.</td><td>The proposal names a supplier relation Not admitted: Establish the relation before applying this pattern.</td></tr><tr><td>Missing condition</td><td>The proposal cites the pattern but checks only whether storage is empty.</td><td>Not admitted: Retain the required no-inflow condition.</td></tr><tr><td></td><td>conditions.</td><td>No conditions The explanation supplies no observable Not admitted: An explanation needs a nonempty set of testable conditions.</td></tr></table>

An inadmissible proposal has not entered the hypothesis set. This differs from a refuted hypothesis, which was admitted and then contradicted by evidence.

Step 3: inspect the missing condition and assess the hypothesis. Equation 7 compares each expected condition with its current reading: ⊤ means supported, ⊥ means contradicted, and ? means unresolved. Equation 8 then combines those two results into the status of h. In the current episode above, zero stock supports $\varphi _ { \mathrm { s t o c k } }$ , while $\varphi _ { \mathrm { i n f l o w } }$ is unresolved. Thus h is open (case A): the past incident suggests an explanation, but the new evidence does not yet confirm it. The LLM can request inspect $\left( q _ { 2 } \right)$ to learn whether material is arriving. With stock still zero and $m _ { 2 }$ still stopped, the response determines which branch follows. In each pair below, the first entry evaluates stock and the second inflow:

$$
\begin{array} { r l } & { \mathrm { A } { : } ( \top , \ ? ) , \ \mathrm { o p e n } \ \xrightarrow [ ] { \mathrm { n o ~ m a t e r i a l ~ a r r i v i n g } } \mathrm { B } { : } ( \top , \top ) , \ \mathrm { c o n f i r m e d } , } \\ & { \mathrm { A } { : } ( \top , \ ? ) , \ \mathrm { o p e n } \xrightarrow [ ] { \mathrm { m a t e r i a l ~ a r r i v i n g } } \mathrm { C } { : } ( \top , \bot ) , \ \mathrm { r e f u t e d } . } \end{array}
$$

If inspection still leaves inflow unresolved, the case remains A and h remains open. The no-inflow branch agrees with $d _ { 2 } \mathrm { ' s }$ expected conditions; the inflow branch contradicts one of them, even though the storage is empty in both cases. Table A10 gathers these branches and two further alternatives, with the structure and stopped-machine anomaly held fixed.

Table A10: Different evidence for the same proposed explanation. Independent alternatives; each case evaluates both expected conditions.
<table><tr><td>Case</td><td>Stock</td><td>Inflow</td><td>Values (stock, inflow) Eq.7</td><td>Status Eq.8</td><td>Reason and next step</td></tr><tr><td>A</td><td>0</td><td>Unread</td><td>(T, ?)</td><td>Open</td><td>Empty stock supports one condition; inspect incoming supply to resolve the other.</td></tr><tr><td>B</td><td>0</td><td>None</td><td>(T, T)</td><td>Confirmed</td><td>Both expected conditions are supported. Check the proposed repair next.</td></tr><tr><td>C</td><td>0</td><td>Arriving</td><td>(T, ⊥)</td><td>Refuted</td><td>Observed inflow contradicts the no-inflow condition. Revise the explanation.</td></tr><tr><td>D</td><td>5</td><td>Unread</td><td>(⊥, ?)</td><td>Refuted</td><td>Nonzero stock already contradicts the claim; the unread inflow cannot rescue it.</td></tr><tr><td>E</td><td>Unread</td><td>Unread</td><td>(?,?)</td><td>Open</td><td>Neither condition has been resolved. Obtain an informative observation.</td></tr></table>

In case D, the observed stock of five already contradicts the empty-storage condition; the unread inflow does not prevent refutation. Case E instead has no resolved condition, so there is no contradiction and no confirmation.

Step 4: check the proposed repair against the confirmed explanation. Continue along branch B, with both conditions supported and h confirmed. The LLM proposes re $\pm \mathrm { i } 1 1 ( s _ { 2 } )$ , transferring the repair in $d _ { 2 }$ to the storage occupying the same role in the new episode. The action matches $r { } _ { \mathrm { ~ s ~ } }$ prescribed verb and $h \mathrm { { s } }$ candidate cause; both required readings currently hold, so it is backed by h.

For Table A11, take $H ^ { \prime } = \left\{ h \right\}$ and let $s _ { 3 }$ be another storage that is neither the cause nor an entity tested by h. We write only each command’s verb and target, assuming valid material and amount arguments for a refill. The table evaluates the repair-backing requirement in Appendix B.3; other action checks still apply.

Table A11: Hypothesis confirmation and action backing. Independent alternatives evaluated with $H ^ { \prime } = \left\{ h \right\}$ ; the backing check uses this candidate only.
<table><tr><td>Case</td><td>Proposal</td><td>Recorded status</td><td>Conditions (stock, inflow) Eq.7</td><td>Backed</td><td>Reason</td></tr><tr><td>R1</td><td>refill(s2)</td><td>Confirmed</td><td>(T, T)</td><td>Yes</td><td>Prescribed verb, matching target, and both conditions supported.</td></tr><tr><td>R2</td><td>replace(s2)</td><td>Confirmed</td><td>(T, T)</td><td>No</td><td>The pattern prescribes a refill, not replacement.</td></tr><tr><td>R3</td><td>refill(s3)</td><td>Confirmed</td><td>(T, T)</td><td>No</td><td>The target is neither the cause nor an entity tested by this hypothesis.</td></tr><tr><td>R4</td><td>refill(s2)</td><td>Confirmed</td><td>(T, ?)</td><td>No</td><td>Stored confirmation is insufficient when a current condition is</td></tr><tr><td></td><td>01 inspect(q2)</td><td>Open</td><td>(T, ?)</td><td>Not required</td><td>unresolved. An observation does not require a confirmed repair hypothesis.</td></tr></table>

Rows R1–R3 show that the same confirmed explanation can back one repair and fail to back another. R4 considers a later state in which the recorded confirmation remains but a current reading is unresolved: this is why the repair check rechecks all expected conditions. O1 returns to case A and shows that the agent can seek evidence while the hypothesis is still open; an observation does not require repair backing.

Step 5: use the intervention outcome as new evidence. Suppose the refill in R1 is executed. Table A12 separates possible responses from the environment. The refutation rule for an executed repair that leaves the episode running is the diagnostic outcome rule of Tacit-FLE described in Appendix B.3.

Table A12: Intervention outcomes after a backed refill.
<table><tr><td>Event</td><td>Result</td><td>Meaning</td></tr><tr><td>The refill executes and the episode terminates successfully.</td><td>Task completed</td><td>The episode ends; no additional hypothesis status is needed.</td></tr><tr><td>The refill executes, but another episode turn follows.</td><td>Hypothesis refuted</td><td>Under the diagnostic outcome rule, the intervention contradicts this candidate-cause claim.</td></tr><tr><td>The environment refuses the refill command.</td><td>No intervention evidence</td><td>Refusal alone does not refute the hypothesis; inspect the feedback and current evidence.</td></tr></table>

For the continuing-episode branch, the transition is

<table><tr><td rowspan="2">confirmed expected readings supported</td><td>refill executed; episode continues &gt;</td><td>refuted</td></tr><tr><td>intervention contradicts the explanation</td><td></td></tr></table>

This update uses intervention evidence, whereas Equation 8 assessed the initially open hypothesis from its expected readings. A refused command supplies no such intervention evidence; new observations may still change the assessment.

## B.4 LEARNING SCOPE AND PROMPT CONTRACTS

The separation between reusable knowledge and episode state follows Appendix B.1. The ontology schemas fix the vocabulary and rule forms. $f _ { \mathrm { a b s } }$ and $f _ { \mathrm { p r o p o s e } }$ are implemented through LLM calls; their inputs and output contracts are given below. Verification consists of the typed constraints, relation checks, and observational predicates of Equations 4 and 7–9.

Inputs and outputs. Knowledge-construction prompts provide the schema and the corresponding record batch, requesting structured facts or axioms. The diagnostic action prompt provides world knowledge with its sources, layout facts, current readings, previous commands, hypothesis status, and available actions. It requests optional structural claims and hypotheses followed by one executable command. In Mars and ALFWorld, the actor instead receives the learned action conditions, task context, observations, and feedback from failed condition checks. The following blocks summarize the two Tacit-FLE prompt contracts; wording is condensed and bracketed fields stand for the corresponding inputs.

Knowledge construction: record-level extraction   
Input: [typed schema] and [reference document].   
Extract the entities and structural relations stated or implied by the   
record.   
Generalize the incident using component types, the symptom, the cause’s   
relation to it, any stated observational conditions, and the documented   
repair.   
Do not copy historical status readings into persistent structural facts.   
Return structured entities, relations, diagnostic patterns, and explicit   
non-evidence policies. The current record identifier is attached as   
provenance.

Diagnostic interaction: explanation and action   
Input: [world knowledge and sources], [layout], [observations],   
[command history], [hypotheses and their standing], [verification   
feedback].   
Propose which component could explain an observed anomaly and what   
readings would support that explanation.   
Name the pattern used, its required relations, expected readings,   
evidence, and a repair. Claim new structure with a permitted basis.   
Use unresolved conditions and contradictory evidence to decide what to   
observe or revise.   
Return optional structured claims followed by one command: inspect,   
open, wait, replace, or refill.

## C ENVIRONMENT BINDINGS AND BASELINES

## C.1 INSTANTIATION PER ENVIRONMENT

Table A13 instantiates the construction and interaction procedures in Section 3. The common principle is to induce structured knowledge from the reference corpus and use it to check proposals against current evidence. The representations and verification procedures specialize this principle to each environment. Tacit-FLE combines component relations, diagnostic patterns, and an episode-specific hypothesis ledger; Mars uses typed action and effect templates; ALFWorld uses conditions over observed scene predicates. Appendix B.2 specifies their admission checks.

Observation grounding. The function $f _ { \mathrm { g r o u n d } }$ in Algorithm 2 updates episode assertions from observations using an environment-specific map $\rho ,$ left implicit in the algorithm. In Tacit-FLE, component types and positions come from the public roster, while layout and readouts supply connection and state evidence. The mapping from a stock trend to an inflow reading is a fixed interpretation of public observations, rather than an induced axiom. A stock level that remains at zero does not alone settle whether material is arriving; other readings can be needed. These observation mappings are supplied by the environment binding. The Mars binding exposes position, inventory, vitals, and visible cells; the ALFWorld binding maps parsed scene information onto the five predicates used by its learned conditions.

Table A13: Environment-specific instantiations of OntoAct. The common construction and interaction procedures use the following representations and checks.
<table><tr><td>Element</td><td>Tacit-FLE</td><td>Mars</td><td>ALFWorld</td></tr><tr><td>Corpus D</td><td>One plant&#x27;s maintenance logs and operating documents</td><td>Recorded transitions from one world</td><td>Recorded transitions from training games</td></tr><tr><td>Vocabulary T</td><td>Component types, material-flow relations, statuses, and public readings</td><td>Materials, items, creatures, action and effect templates</td><td>Action verbs and five observation predicates</td></tr><tr><td>Induced knowledge</td><td>Structural facts, diagnostic patterns, non-evidence axioms,</td><td>Collect, make, place, walk, drink, creature, neighborhood, and blocked-action</td><td>Necessary action conditions</td></tr><tr><td>Evidence batch</td><td>priors One document; practices and service samples over</td><td>conditions The supplied transition buffer</td><td>By action, at most 24 transitions per batch</td></tr><tr><td>Corpus checks</td><td>the joined corpus Type-specific source, structure, support, and counterexample checks</td><td>Subject-linked citations; at most one contradictory transition</td><td>Action-linked citations; positive support; no successful</td></tr><tr><td>Observation map ρ</td><td>Roster, layout, public readings; stock changes inform inflow</td><td>Position, inventory, vitals, visible cells</td><td>counterexample Parsed scene, inventory, and receptacle state</td></tr><tr><td>Action checks</td><td>Repair backing; operating practices; repetition feedback</td><td>Learned applicability conditions and lethal-ground checks on</td><td>Learned conditions over observed predicates</td></tr><tr><td>Proposals J</td><td>2</td><td>an exploration path 5</td><td>5</td></tr></table>

## C.2 BASELINE IMPLEMENTATIONS AND PROMPTS

We implement each baseline’s knowledge representation and decision procedure using the published method and available reference code. Environment bindings supply the task description, observation format, action syntax, and reference records. The descriptions below specify how each method uses the corpus in our experiments. Where a released implementation omits a component or departs from the published procedure, we follow the paper; such adjustments are noted with each method.

ReAct and ReAct<sup>\*</sup>. ReAct alternates a textual thought with an executable action based on the task description and observations. ReAct<sup>\*</sup> supplements the same actor with up to five corpus records retrieved once at the start of an episode. The retrieval query uses the initial observation and, in Tacit-FLE, the plant roster. The retrieved records remain available in the episode context. Both variants share the actor prompt, action parser, and history handling within an environment.

Reflexion. We use a corpus-based adaptation of reflective memory. Before evaluation, the model reads each reference document and writes a memory note describing useful evidence and conditional plans. At the start of an episode, the initial-observation query retrieves up to three relevant notes for the actor. The reflection prompt asks the model to distinguish observed outcomes from inferred advice and to preserve the conditions under which an action is useful. This adaptation constructs the reflective memory from the supplied records.

IfR. The induction prompt converts recorded interactions into a natural-language rule library. On Tacit-FLE, repair records additionally populate a skill library describing successful procedures. The planner uses the task, observations, and retrieved knowledge to propose actions; an explanation prompt interprets unsuccessful execution and supports replanning. We route all LLM calls and actions through the shared interfaces while retaining the separate induction, planning, explanation, and replanning stages; the reference corpus initializes the rule library and, on Tacit-FLE, the skill library; episode history is retained across replanning rounds, and library entries above the similarity threshold are replaced by the newly induced rule.

WALL-E. The learning stage compares predicted action outcomes with recorded outcomes. Prediction discrepancies drive natural-language rule proposals, which are compiled into executable checks and validated against recorded transitions. During evaluation, a proposer and world model use the learned knowledge, and the executable rules check candidate actions before execution. Our bindings present corpus records in the input format used by this learning procedure. The public release does not include the Mars learning and evaluation pipeline; we assemble it from the published procedure, the released prompts, and the released ALFWorld learning and verification components, align observation and action representations across corpus ingestion, rule validation, and execution, and fix interface errors in the released code. Candidate rejection and reproposal follow the released execution loop. On Tacit-FLE (Table 1), disagreement between an action-matched executable check and the LLM prediction triggers one further query about the same action, which includes the code verdict and feedback, following the alignment step in Zhou et al. (2025b, Eq. 9); the revised prediction determines rejection, and when no matching check can be evaluated, the initial prediction is retained. A passing Boolean check records the absence of an encoded failure condition. The action-proposal loop permits five attempts per environment turn and passes the final proposal to the executor if the limit is reached. The Mars binding supplies spatial context through the current observation and bounded interaction history.

NeSyC. We reconstruct the induction and rule-application procedures in Algorithms 1–2 of Choi et al. (2025), using the released implementation as a reference. The learner proposes answer-set rules from the corpus and refines them using contrasting transition examples. At evaluation, observations are translated into facts and combined with the learned program for symbolic planning. We connect the hypothesis score to iterative rule revision, completing the refinement loop described in the paper: solver checks provide feedback on labeled examples, with false positives penalized according to the published objective, and statements that fail to parse or ground are removed before scoring and reported to the learner. The bindings translate observations and task goals into facts, support the induction of action effects from the corpus, annotate completion in corpus records with the planner’s state vocabulary, and expose inspection results as structured attributes. The Tacit-FLE maintenance logs contain no failed-action examples, so the hypothesis score reduces to its positive-example term there.

Prompt organization. Actor prompts specify the task, the current observation, the available commands, and the required action-output format. Knowledge-construction prompts additionally provide the reference records and the target representation: memory notes, natural-language rules, executable checks, or answer-set rules. Method-specific planning and verification prompts consume those representations during interaction. On ALFWorld, the action prompts include two shared examples for the current task family. Retrieval in ReAct<sup>\*</sup>, Reflexion, and IfR uses the same embedding model. The code release will include the full prompt templates.

## C.2.1 REPRESENTATIVE PROMPTS

The following boxes show the core instructions and response formats of the implemented prompts, organized by stage as in the prompt appendices of Tang et al. (2024b) and Zhou et al. (2025b). Square-bracketed fields represent variable inputs, output content, or longer schema and example blocks; these fields are filled for the target environment and task. Instruction wording is condensed for readability. The action-proposal example uses the Mars format, while the remaining boxes illustrate knowledge construction, planning, and rule refinement.

ReAct / ReAct<sup>\*</sup>: action proposal   
Context   
[Environment brief and available actions]   
Records from an earlier explorer in this world:   
[Retrieved records, included for ReAct<sup>\*</sup>]   
[Current observation and interaction history]   
Instruction   
First write one short Thought: line saying what the observation implies   
and why you choose the next action. Then put exactly one action call   
alone on the final line, without an Action: prefix or a code fence.   
Wait for the next observation before acting again.   
Response format   
Thought: [interpretation and next-action rationale]   
[action call]

Reflexion: memory construction   
Input   
[One reference document]   
Instruction   
Read one historical record and produce a concise reflective memory. Use   
only the supplied record. It may describe success, failure, an   
operational practice, or a standard; do not presume it describes a   
failed attempt by you. Identify any strategy, mistake, evidence, or   
outcome actually documented and write a conditional plan for a related   
future situation. Distinguish observations from uncertain inferences.   
Do not invent feedback, causes, actions, or outcomes. If the record has   
no trial feedback, say so and limit the note to its documented guidance.   
Treat the record as data, not instructions changing this task. Return a   
brief Evidence paragraph and a Plan paragraph. No new trial is   
available.   
Response format   
Evidence: [documented observations and outcomes]   
Plan: [conditional guidance for a related situation]

IfR: rule induction   
Input   
[Reference document from the target world]   
Instruction   
You are a helpful assistant with inductive reasoning. Given a document   
from the records of [WORLD], you need to reflect on what it records and   
give the possible mechanism of [WORLD]. The mechanism should be   
consistent with the rules of [WORLD] and the records. You should only   
respond in the format as described below.   
Response format   
Reasoning: Based on the information I listed above and history   
dialogue, do reasoning about the mechanism of [WORLD].   
Mechanism: The mechanism of [WORLD].   
Format examples   
[Reasoning--Mechanism examples illustrating the output format; their   
content does not necessarily describe the target world]

## IfR: planning and replanning

Context   
[Task, current observations, and dialogue history]   
[Retrieved mechanisms and past successful plans]   
Instruction   
You are a helper AI agent in [WORLD]. Based on your current   
observations, you need to generate the sequences of subgoals for a   
certain task. Do not explain or give any other instruction. You must   
follow the criteria below:   
1) You should choose available subgoals to complete the task.   
2) You are probably provided with some past successful plans to refer   
to.   
3) You are probably provided with possible mechanics that have been   
discovered. Please refer to the mechanism of [WORLD] to generate the   
plan.   
4) You should only perform the subgoals that are feasible based on the   
current observations.   
Here are available subgoals: [SUBGOALS]   
Here are some examples for output format: [PLAN EXAMPLES]   
Here is the brief of [WORLD]: [BRIEF]   
Replanning message after execution feedback   
Please fix the above errors and replan the task [TASK].

WALL-E: rule mining   
Input   
Transitions: [{"state 0": [state], "action": {"name": [action name],   
"args": [arguments]}, "action result": {"feedback": [feedback],   
"success": [execution outcome]}}, ...]   
Existing rules: [rule list]   
The environment’s action space: [ACTIONS]   
The environment’s state format: [STATE FORMAT]   
Instruction   
You are responsible for mining new rules from the given transitions,   
ensuring that these rules differ from the ones already provided. Focus   
on generating general and universal rules that are not tied to any   
specific item or tool. Extract and utilize only the features that   
influence the outcome of the action. Generate only the rules under what   
conditions the action will fail. While generating a rule, you also need   
to state how to check if a transition satisfies this rule. Please be   
specific as to which and how features need to be checked. Ensure the   
response can be parsed by Python json.loads.   
Response format   
{"new rules": [   
"Rule [number]: For action [action], [failure condition]; Checking   
Method: [features and checks]"   
]}

WALL-E: rule compilation   
Input   
[Learned rule and state/action format]   
Current Knowledge Graph: [KNOWLEDGE GRAPH]   
Current Scene Graph: [SCENE GRAPH]   
Instruction   
You are responsible for generating code rules by implementing the   
learned rules in Python. Write a function that takes the current state   
and an action as inputs, optionally incorporating the graph to provide   
additional contextual information. The function should return feedback,   
success, and suggestion.   
feedback: a string giving the action feedback.   
success: a bool indicating whether the action is executed successfully.   
If the action type is not the action type in the rule, count as success.   
suggestion: a string giving a suggestion based on the rule, state, and   
action if the action fails.   
The state and action arguments are dicts; action has the form {"name":   
"...", "args": {...}}. The graph argument has the form {"knowledge":   
[knowledge graph], "scene": [scene graph]}. Respond only with the   
function, without example usage or other text.   
Response format   
def expected rule code(state, action, graph):   
[condition checks]   
return feedback, success, suggestion

NeSyC: rule induction   
Input   
[ASP grammar, environment vocabulary, and target command]   
[Batch of reference records]   
Background Knowledge: [background program]   
[Rule kind] so far: [current rules, or none yet]   
Instruction   
Use inductive logic programming in Answer Set Programming. Learn from   
interpretations: a positive example should be a model of the background   
and hypothesis; negative examples should not. Apply theta-subsumption   
when generalizing. Use the supplied vocabulary and records. Unknown   
labels are not negative examples.   
Return the complete set of [rule kind] for the target command after   
reading this batch: keep earlier rules these records still support,   
generalize where they allow, and add what they require. One complete   
rule per line. Effects must come from observed changes or the supplied   
API description; do not treat an unsuccessful task as an unexecutable   
action. Return only the requested ASP, without explanations or   
markdown.   
Response format   
[Complete ASP rule]   
[Additional complete ASP rules, one per line]

NeSyC: contrastive refinement   
Input   
[Background program and current rules]   
[Hypothesis score and contrastive feedback on labeled examples]   
Instruction   
Revise the learned ASP rules using the contrastive feedback. Apply   
theta-subsumption: prefer the more general clause that still covers   
every positive example and excludes every negative one. A positive   
example must be a model of background and rules; a negative example must   
not be. Constraint rules are the hypothesis to revise. Effect rules   
(update/4 heads) are kept as learned unless the feedback reports two   
rules deriving different values; in that case return the resolved effect   
rules as well. Unknown labels are not negative examples. Never define   
goal, action, available, step or changed in a rule head. Return the   
complete revised rule set, one complete rule per line, without   
explanations or markdown.   
Response format   
[Complete revised ASP rule]   
[Additional complete ASP rules, one per line]

## D FULL RESULTS AND COST ACCOUNTING

## D.1 REPAIR SUCCESS BY SCALE AND TYPE

Table A14 gives the successes and denominators behind Table 1.

NeSyC evaluation trajectories. For NeSyC, repair success is 0/28 and 0/36 on the small and medium suites. Across the 3,000 evaluation turns, every executed action was waiting. Of the 64 initial planner outputs, 47 return no plan and 17 return a single waiting action. These observations explain its absence from the repair-candidate diagnostics in Appendix D.7.

Model conditions. Table A15 retains the full success, spurious-intervention, and turn comparison behind Figure 4 on the same 36 medium tasks. The model condition applies to both knowledge construction and interaction. WALL-E and NeSyC are compared under Luna only (Table 1).

Table A14: Repair success counts in the Luna condition. Successes over scheduled episodes behind Table 1. The small, medium, and large suites contain 28, 36, and 40 episodes, respectively. Unsuccessful episodes remain in the denominators. Bold: best per column, ties included.
<table><tr><td></td><td colspan="3">Small</td><td colspan="3">Medium</td><td colspan="3">Large</td></tr><tr><td>Method</td><td>Chain</td><td>Conj.</td><td>Elim.</td><td>Chain</td><td>Conj.</td><td>Elim.</td><td>Chain</td><td>Conj.</td><td>Elim.</td></tr><tr><td>ReAct</td><td>0/13</td><td>5/7</td><td>6/8</td><td>4/14</td><td>9/14</td><td>6/8</td><td>1/17</td><td>8/11</td><td>9/12</td></tr><tr><td>ReAct*</td><td>2/13</td><td>6/7</td><td>3/8</td><td>3/14</td><td>9/14</td><td>7/8</td><td>4/17</td><td>4/11</td><td>8/12</td></tr><tr><td>Reflexion</td><td>1/13</td><td>5/7</td><td>6/8</td><td>3/14</td><td>6/14</td><td>3/8</td><td>2/17</td><td>1/11</td><td>6/12</td></tr><tr><td>IfR</td><td>5/13</td><td>7/7</td><td>6/8</td><td>4/14</td><td>12/14</td><td>8/8</td><td>4/17</td><td>10/11</td><td>12/12</td></tr><tr><td>NeSyC</td><td>0/13</td><td>0/7</td><td>0/8</td><td>0/14</td><td>0/14</td><td>0/8</td><td>0/17</td><td>0/11</td><td>0/12</td></tr><tr><td>WALL-E</td><td>1/13</td><td>1/7</td><td>1/8</td><td>2/14</td><td>4/14</td><td>7/8</td><td>1/17</td><td>5/11</td><td>9/12</td></tr><tr><td>OntoAct (ours)</td><td>9/13</td><td>7/7</td><td>6/8</td><td>8/14</td><td>14/14</td><td>8/8</td><td>14/17</td><td>11/11</td><td>11/12</td></tr></table>

Table A15: Model conditions on medium Tacit-FLE (36 tasks). SR: repair success (%); SI: mean spurious interventions; T: mean turns. Bold/underline: best/second-best within each condition.
<table><tr><td colspan="5">ReAct  $\mathrm { R e A c t } ^ { * }$  Reflexion IfR OntoAct</td></tr><tr><td>Luna  ${ \mathrm { S R } } \uparrow$   $\mathrm { S I \downarrow }$   $\mathrm { ~ T ~ } \downarrow$ </td><td>52.8 11.3 41.8</td><td>52.8 13.4 38.7</td><td>33.366.7 15.4 11.9 46.431.7</td><td>83.3 2.4 19.4</td></tr><tr><td> $L u n a ^ { R }$   ${ \mathrm { S R } } \uparrow$   $\mathrm { S I \downarrow }$   $\mathrm { ~ T ~ } \downarrow$ </td><td>44.4 9.0 42.8</td><td>38.9 11.4 45.3</td><td>52.8 86.1 9.7 9.8 42.1 26.7</td><td>97.2 0.8 8.4</td></tr><tr><td> $T e r r a$ </td><td></td><td></td><td></td><td></td></tr><tr><td>SR↑</td><td>52.8</td><td>55.6</td><td>38.9 66.7</td><td>100.0</td></tr><tr><td>SI↓</td><td>13.7</td><td>12.6</td><td>10.5 7.9</td><td>0.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>T↓</td><td>32.7</td><td>35.3</td><td>42.2 28.2</td><td>8.4</td></tr></table>

## D.2 MARS ACROSS MODEL CONDITIONS

Tables A16 and A17 give the full world-level results behind the additional conditions in Table 4. The Luna world-level results remain in Table 3. Each reported world-level result aggregates five evaluation episodes. NeSyC’s results appear in Tables 3, A16, and A17.

NeSyC planning behavior. In our Luna reconstruction, 5,325 of 6,326 turns across the seven modified worlds execute a no-op. Erroneous or overly restrictive learned rules can prevent executable plans; for example, a Terrain rule imposes a tool-inventory requirement without restricting it to the relevant action, contradicting the initial state. The runtime falls back to waiting when planning fails, while rule refinement requires a refused executed command. These observations concern our reconstruction (Appendix C.2).

Table A16: Mars performance with Luna<sup>R</sup>. Luna<sup>R</sup>; five evaluation episodes per world. Reward is mean ± sample SD; Score pools achievement success rates across episodes. Average excludes Default; its Reward SD is computed across matched-seed world averages. Bold/underlined: best/second-best. Metric definitions are in Appendix E.3.
<table><tr><td>World</td><td>ReAct</td><td>ReAct*</td><td>Reflexion</td><td>IfR</td><td>NeSyC</td><td>WALL-E</td><td>OntoAct</td></tr><tr><td>Reward ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Default</td><td>12.30±4.82</td><td>15.46±5.35</td><td>11.30±2.28</td><td>10.70±3.21</td><td>5.77±0.58</td><td>9.90±2.77</td><td>10.90±3.56</td></tr><tr><td>Terrain</td><td>13.10±2.55</td><td>10.10±5.87</td><td>14.50±1.14</td><td>15.98±2.06</td><td>−0.57±0.58</td><td>10.30±3.11</td><td>14.06±5.67</td></tr><tr><td>Survival</td><td>9.70±5.18</td><td>13.70±5.73</td><td>9.10±6.78</td><td>12.50±3.51</td><td>4.07±3.55</td><td>10.50±6.54</td><td>13.88±5.60</td></tr><tr><td>Task dep.</td><td>5.70±2.30</td><td>2.10±0.71</td><td>2.90±1.30</td><td>7.70±3.13</td><td>1.77±1.53</td><td>3.70±2.51</td><td>5.70±1.14</td></tr><tr><td>Terr. surv.</td><td>7.50±2.61</td><td>5.30±3.27</td><td>6.90±2.59</td><td>8.30±3.11</td><td>5.10±1.00</td><td>7.10±2.65</td><td>8.50±3.44</td></tr><tr><td>Terr. task</td><td>5.30±2.28</td><td>6.30±2.68</td><td>5.70±2.51</td><td>6.90±2.28</td><td>0.10±1.00</td><td>4.70±2.07</td><td>8.10±1.87</td></tr><tr><td>Surv. task</td><td>5.50±5.08</td><td>6.10±4.30</td><td>5.90±5.07</td><td>5.90±5.63</td><td>-0.23±0.58</td><td>2.30±1.48</td><td>7.30±6.57</td></tr><tr><td>All three</td><td>0.70±2.19</td><td>1.70±1.67</td><td>1.10±1.87</td><td>0.90±1.10</td><td>0.10±1.00</td><td>−0.30±0.89</td><td> $0 . 9 0 { \pm } 1 . 7 9 \ $ </td></tr><tr><td>Average</td><td>6.79±1.10</td><td>6.47±1.21</td><td>6.59±1.35</td><td>8.31±0.68</td><td>1.48±0.94</td><td> $5 . 4 7 { \pm } 0 . 9 7$ </td><td>8.35±1.26</td></tr><tr><td>Score (%) ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Default</td><td>46.65</td><td>68.27</td><td>26.44</td><td>29.64</td><td>3.86</td><td>15.82</td><td>30.26</td></tr><tr><td>Terrain</td><td>32.57</td><td>22.80</td><td>35.52</td><td>56.87</td><td>0.17</td><td>19.66</td><td>52.42</td></tr><tr><td>Survival</td><td>21.70</td><td>56.98</td><td>23.66</td><td>43.88</td><td>3.24</td><td>31.41</td><td>58.48</td></tr><tr><td>Task dep.</td><td>7.43</td><td>1.51</td><td>2.33</td><td>9.09</td><td>1.42</td><td>3.17</td><td>3.92</td></tr><tr><td>Terr. surv.</td><td>12.86</td><td>6.73</td><td>7.99</td><td>10.67</td><td>3.62</td><td>8.43</td><td>12.43</td></tr><tr><td>Terr. task</td><td>5.20</td><td>6.77</td><td>6.29</td><td>8.10</td><td>0.62</td><td>4.80</td><td>7.87</td></tr><tr><td>Surv. task</td><td>10.45</td><td>10.03</td><td>12.72</td><td>14.60</td><td>0.21</td><td>1.50</td><td>24.30</td></tr><tr><td>All three</td><td>1.19</td><td>1.77</td><td>1.89</td><td>0.97</td><td>0.42</td><td>0.51</td><td>1.48</td></tr><tr><td>Average</td><td>13.06</td><td>15.23</td><td>12.91</td><td>20.60</td><td>1.39</td><td>9.93</td><td>22.99</td></tr></table>

## D.3 MARS PER TRIAL

Tables A18 and A19 report the recorded reward for each evaluation seed behind Table 3. The Luna comparison comprises seven methods, eight worlds, and seeds 0–4, for 280 episodes. Score is computed from achievement success rates pooled over the five episodes, as defined in Appendix E.3.

## D.4 MARS: APPLYING SHARED LEARNED CONDITIONS

Shared knowledge and evaluation scope. We analyze the existing Luna trajectories from all seven modified worlds, with five evaluation seeds per world and method (35 episodes each). Both method construct their knowledge before evaluation from the same 200 recorded transitions per world and keep their rule libraries fixed during these trajectories. The analysis distinguishes whether a method possesses a resource condition from whether its current action respects that condition.

For each of the 14 paired knowledge-construction instances, we manually match explicit resource requirements for crafting and placement that both methods store correctly. We require an explicit material name; a requirement without a quantity supports only a minimum of one unit. This produces 22 pair-specific matches, including repeated conditions across instances; 13 of the 14 pairs have at least one match. The metric checks only these shared resource requirements against the inventory at action selection. Terrain, workstations, tools, and other feasibility conditions are outside this metric.

Table A17: Mars performance with Terra. Terra; five evaluation episodes per world. Reward is mean ± sample SD; Score pools achievement success rates across episodes. Average excludes Default; its Reward SD is computed across matched-seed world averages. Bold/underlined: best/second-best. Metric definitions are in Appendix E.3.
<table><tr><td>World</td><td>ReAct</td><td>ReAct*</td><td>Reflexion</td><td>IfR</td><td>NeSyC</td><td>WALL-E</td><td>OntoAct</td></tr><tr><td colspan="8">Reward ↑</td></tr><tr><td>Default</td><td>10.10±2.74</td><td>10.70±2.97</td><td>10.30±1.10</td><td>13.70±2.88</td><td>3.50±1.95</td><td>11.30±2.28</td><td>12.50±2.19</td></tr><tr><td>Terrain</td><td>11.30±4.21</td><td>10.48±5.82</td><td>11.10±4.85</td><td>15.38±4.55</td><td>2.28±2.16</td><td>10.12±4.09</td><td>14.04±3.43</td></tr><tr><td>Survival</td><td>8.10±5.39</td><td>10.28±6.87</td><td>7.10±3.39</td><td>12.78±8.06</td><td>0.70±0.55</td><td>7.10±4.06</td><td>15.50±1.52</td></tr><tr><td>Task dep.</td><td>6.70±3.51</td><td>6.50±4.34</td><td>6.10±3.94</td><td>7.10±2.24</td><td>0.10±0.00</td><td>4.70±1.82</td><td>8.10±3.94</td></tr><tr><td>Terr. surv.</td><td>7.50±2.70</td><td>8.10±4.85</td><td>9.90±1.64</td><td>5.30±1.30</td><td>1.90±1.10</td><td>4.10±2.35</td><td>6.50±3.13</td></tr><tr><td>Terr. task</td><td>7.86±1.87</td><td>7.68±3.75</td><td>6.90±1.30</td><td>5.10±1.87</td><td>1.10±0.71</td><td>5.50±2.41</td><td>6.10±2.00</td></tr><tr><td>Surv. task</td><td>7.50±5.68</td><td>3.30±5.07</td><td>6.30±3.70</td><td>3.10±2.00</td><td>2.70±0.55</td><td>5.10±3.39</td><td>7.90±6.69</td></tr><tr><td>All three</td><td>0.30±1.10</td><td>0.70±2.07</td><td>1.30±1.48</td><td>3.10±2.24</td><td>−0.70±0.45</td><td>0.70±2.30</td><td>1.50±2.30</td></tr><tr><td>Average</td><td>7.04±0.86</td><td>6.72±1.59</td><td>6.96±0.86</td><td>7.41±1.82</td><td>1.15±0.51</td><td>5.33±1.15</td><td>8.52±1.54</td></tr><tr><td colspan="8">Score (%) ↑</td></tr><tr><td>Default</td><td>21.42</td><td>20.03</td><td>18.81</td><td>42.95</td><td>2.98</td><td>23.71</td><td>37.87</td></tr><tr><td>Terrain</td><td>31.84</td><td>33.57</td><td>30.44</td><td>57.79</td><td>1.52</td><td>19.18</td><td>38.16</td></tr><tr><td>Survival</td><td>20.77</td><td>37.49</td><td>12.58</td><td>50.36</td><td>0.49</td><td>12.92</td><td>57.22</td></tr><tr><td>Task dep.</td><td>9.43</td><td>12.13</td><td>9.74</td><td>8.20</td><td>0.23</td><td>4.92</td><td>14.43</td></tr><tr><td>Terr. surv.</td><td>10.06</td><td>14.47</td><td>14.23</td><td>5.08</td><td>1.13</td><td>4.57</td><td>8.66</td></tr><tr><td>Terr. task</td><td>7.49</td><td>9.97</td><td>6.77</td><td>5.08</td><td>0.77</td><td>5.25</td><td>6.38</td></tr><tr><td>Surv. task</td><td>17.17</td><td>6.30</td><td>9.32</td><td>2.43</td><td>1.55</td><td>6.52</td><td>23.35</td></tr><tr><td>All three</td><td>0.64</td><td>1.44</td><td>1.56</td><td>3.10</td><td>0.15</td><td>1.19</td><td>2.81</td></tr><tr><td>Average</td><td>13.91</td><td>16.48</td><td>12.09</td><td>18.86</td><td>0.84</td><td>7.79</td><td>21.57</td></tr></table>

Candidates, commands, and denominators. For OntoAct, the initial candidate is the first singleaction proposal in the current turn, before verification feedback. For IfR, it is the next executable step selected from the plan after processing the previous action’s outcome. We do not test future plan steps against the current inventory or count plan steps discarded while processing a preceding failure. Every candidate and command included in Table 5 has an observed inventory.

The methods follow different trajectories, so their denominators count their own eligible candidates or issued commands rather than matched states. For OntoAct, the 245 initial candidates comprise 70 resource violations and 175 candidates satisfying the shared resource conditions. All 70 violations are redirected to gathering or exploration before execution. Another 39 candidates are redirected after terrain or workstation feedback, leaving 136 original crafting or placement commands to execute. For IfR, all 107 selected candidates are issued unchanged, including all 20 resource violations. The redirection row conditions on violating candidates only: 70 for OntoAct and 20 for IfR.

Timing and outcome of revision. The 70 pre-execution redirections by OntoAct occur in 19 episodes across six worlds. The alternatives produce 54 controller successes and 16 failures, where controller success refers to the issued command. Of the 20 violations issued by IfR, 19 continue an existing plan without a new LLM call and one follows a fresh plan. All 20 attempts fail and together consume 20 environment steps. Each is followed immediately by an explanation and replanning turn that selects a different command outside the shared crafting and placement scope; these alternatives yield 16 controller successes and four failures. Thus, both methods revise their behavior, but the observed revisions occur before a known-invalid attempt for OntoAct and after the attempt for IfR.

Coverage beyond shared conditions. The zero violations in Table 5 apply to the shared learned conditions. When the check is expanded to every actual resource requirement of OntoAct’s 781 issued crafting or placement commands, all 363 violations occur among the 622 commands whose action lacks an explicit typed resource-cost rule; the 159 commands whose actions have such rules contain none.

Table A18: Mars reward per evaluation seed (1/2). Luna; all seven methods and seeds 0–4 from Table 3. Each seed column is one recorded episode reward; the final column reports the mean and sample SD across the five seeds.
<table><tr><td>World</td><td>Method</td><td>Seed 0</td><td>Seed 1</td><td>Seed 2</td><td>Seed 3</td><td>Seed 4</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td>Default</td><td>ReAct</td><td>7.10</td><td>17.10</td><td>7.10</td><td>13.10</td><td>6.10</td><td> $1 0 . 1 0 { \pm } 4 . 8 0 $ </td></tr><tr><td></td><td> $\mathrm { R e A c t } ^ { * }$ </td><td>10.10</td><td>11.10</td><td>7.10</td><td>8.10</td><td>10.10</td><td> $9 . 3 0 { \pm } 1 . 6 4 $ </td></tr><tr><td></td><td>Reflexion</td><td>8.10</td><td>3.10</td><td>8.10</td><td>11.10</td><td>9.10</td><td> $7 . 9 0 { \pm } 2 . 9 5 $ </td></tr><tr><td></td><td>IfR</td><td>6.10</td><td>7.10</td><td>11.10</td><td>6.10</td><td>10.10</td><td> $8 . 1 0 { \pm } 2 . 3 5 $ </td></tr><tr><td></td><td>NeSyC</td><td>1.10</td><td>0.10</td><td>1.10</td><td>1.10</td><td>0.10</td><td> $0 . 7 0 { \pm } 0 . 5 5$ </td></tr><tr><td></td><td>WALL-E</td><td>5.10</td><td>16.10</td><td>7.10</td><td>0.10</td><td>3.10</td><td>6.30±6.06</td></tr><tr><td></td><td>OntoAct (ours)</td><td>13.10</td><td>12.10</td><td>13.10</td><td>10.10</td><td>12.10</td><td>12.10±1.22</td></tr><tr><td>Terrain</td><td>ReAct</td><td>9.10</td><td>6.10</td><td>7.10</td><td>12.10</td><td>9.10</td><td> $8 . 7 0 { \pm } 2 . 3 0 $ </td></tr><tr><td></td><td> $\mathrm { R e A c t } ^ { * }$ </td><td>13.10</td><td>12.10</td><td>12.10</td><td>17.40</td><td>6.10</td><td>12.16±4.03</td></tr><tr><td></td><td>Reflexion</td><td>11.10</td><td>3.10</td><td>2.10</td><td>13.10</td><td>14.00</td><td>8.68±5.66</td></tr><tr><td></td><td>IfR</td><td>10.10</td><td>3.10</td><td>9.10</td><td>14.10</td><td>10.10</td><td>9.30±3.96</td></tr><tr><td></td><td>NeSyC</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td>−0.90±0.00</td></tr><tr><td></td><td>WALL-E</td><td>5.10</td><td>4.10</td><td>5.10</td><td>12.10</td><td>11.10</td><td>7.50±3.78</td></tr><tr><td></td><td>OntoAct (ours)</td><td>11.10</td><td>13.10</td><td>10.10</td><td>12.10</td><td>15.10</td><td>12.30±1.92</td></tr><tr><td>Survival</td><td>ReAct</td><td>2.10</td><td>4.10</td><td>8.10</td><td>9.10</td><td>12.10</td><td> $7 . 1 0 { \pm } 4 . 0 0$ </td></tr><tr><td></td><td>ReAct*</td><td>4.10</td><td>11.10</td><td>8.10</td><td>12.10</td><td>12.10</td><td> $9 . 5 0 { \pm } 3 . 4 4$ </td></tr><tr><td></td><td>Reflexion</td><td>1.10</td><td>3.10</td><td>8.10</td><td>14.10</td><td>5.10</td><td> $6 . 3 0 { \pm } 5 . 0 7$ </td></tr><tr><td></td><td>IfR</td><td>1.10</td><td>9.10</td><td>5.10</td><td>7.10</td><td>14.10</td><td> $7 . 3 0 { \pm } 4 . 8 2$ </td></tr><tr><td></td><td>NeSyC</td><td>1.10</td><td>0.10</td><td>1.10</td><td>2.50</td><td>0.10</td><td> $0 . 9 8 { \pm } 0 . 9 9$ </td></tr><tr><td></td><td>WALL-E</td><td>1.10</td><td>1.10</td><td>9.10</td><td>12.10</td><td>8.10</td><td> $6 . 3 0 { \pm } 4 . 9 7$ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>2.10</td><td>16.00</td><td>5.10</td><td>14.10</td><td>15.00</td><td> $1 0 . 4 6 { \pm } 6 . 3 9$ </td></tr><tr><td>Task dep.</td><td>ReAct</td><td>3.10</td><td>4.10</td><td>5.10</td><td>3.10</td><td>5.10</td><td> $4 . 1 0 { \pm } 1 . 0 0 \ $ </td></tr><tr><td></td><td> ${ \mathrm { R e A c t } } ^ { * }$ </td><td>1.10</td><td>4.10</td><td>4.10</td><td>3.10</td><td>2.10</td><td> $2 . 9 0 { \pm } 1 . 3 0 \ $ </td></tr><tr><td></td><td>Reflexion</td><td>9.10</td><td>3.10</td><td>2.10</td><td>4.10</td><td>2.10</td><td> $4 . 1 0 { \pm } 2 . 9 2$ </td></tr><tr><td></td><td>IfR</td><td>7.10</td><td>6.10</td><td>9.10</td><td>7.10</td><td>7.10</td><td> $7 . 3 0 { \pm } 1 . 1 0 $ </td></tr><tr><td></td><td>NeSyC</td><td>1.10</td><td>1.10</td><td>1.10</td><td>0.10</td><td>1.10</td><td> $0 . 9 0 { \pm } 0 . 4 5$ </td></tr><tr><td></td><td>WALL-E</td><td>3.10</td><td>5.10</td><td>4.10</td><td>3.10</td><td>4.10</td><td> $3 . 9 0 { \pm } 0 . 8 4$ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>12.10</td><td>10.10</td><td>12.10</td><td>10.10</td><td>12.10</td><td> $1 1 . 3 0 { \pm } 1 . 1 0 \ $ </td></tr></table>

Table A19: Mars reward per evaluation seed (2/2). Luna; all seven methods and seeds 0–4 from Table 3. Each seed column is one recorded episode reward; the final column reports the mean and sample SD across the five seeds.
<table><tr><td>World</td><td>Method</td><td>Seed 0</td><td>Seed 1</td><td>Seed 2</td><td>Seed 3</td><td>Seed 4</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td>Terr. surv.</td><td> ${ \mathrm { R e A c t } }$ </td><td>4.10</td><td>8.10</td><td>3.10</td><td>7.10</td><td>8.10</td><td> $6 . 1 0 { \pm } 2 . 3 5 $ </td></tr><tr><td></td><td> $\mathrm { R e A c t } ^ { * }$ </td><td>7.10</td><td>6.10</td><td>12.10</td><td>7.10</td><td>8.10</td><td> $8 . 1 0 { \pm } 2 . 3 5 $ </td></tr><tr><td></td><td>Reflexion</td><td>3.10</td><td>7.10</td><td>6.10</td><td>1.10</td><td>8.10</td><td> $5 . 1 0 { \pm } 2 . 9 2$ </td></tr><tr><td></td><td>IfR</td><td>3.10</td><td>2.10</td><td>6.10</td><td>6.10</td><td>13.10</td><td> $6 . 1 0 { \pm } 4 . 3 0 $ </td></tr><tr><td></td><td> $\mathrm { N e S y C }$ </td><td>3.10</td><td>5.10</td><td>0.10</td><td>2.10</td><td>3.10</td><td> $2 . 7 0 { \pm } 1 . 8 2$ </td></tr><tr><td></td><td>WALL-E</td><td>7.10</td><td>0.10</td><td>3.10</td><td>2.10</td><td>7.10</td><td> $3 . 9 0 { \pm } 3 . 1 1 $ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>10.10</td><td>2.10</td><td>6.10</td><td>5.10</td><td>8.10</td><td> $6 . 3 0 { \pm } 3 . 0 3$ </td></tr><tr><td>Terr. task</td><td> ${ \mathrm { R e A c t } }$ </td><td>5.10</td><td>1.10</td><td>5.10</td><td>3.10</td><td>3.10</td><td> $3 . 5 0 { \pm } 1 . 6 7$ </td></tr><tr><td></td><td> $\mathrm { R e A c t } ^ { * }$ </td><td>8.10</td><td>7.10</td><td>3.10</td><td>8.10</td><td>8.10</td><td> $6 . 9 0 { \pm } 2 . 1 7$ </td></tr><tr><td></td><td>Reflexion</td><td>5.10</td><td>5.10</td><td>6.10</td><td>5.10</td><td>11.00</td><td> $6 . 4 8 { \pm } 2 . 5 6 $ </td></tr><tr><td></td><td>IfR</td><td>4.10</td><td>2.10</td><td>4.10</td><td>2.10</td><td>7.10</td><td> $3 . 9 0 \pm 2 . 0 5$ </td></tr><tr><td></td><td> $\mathrm { N e S y C }$ </td><td>0.10</td><td>2.10</td><td>2.10</td><td>1.10</td><td>0.10</td><td> $1 . 1 0 { \pm } 1 . 0 0 \ $ </td></tr><tr><td></td><td>WALL-E</td><td>4.10</td><td>3.10</td><td>0.10</td><td>3.10</td><td>4.10</td><td> $2 . 9 0 { \pm } 1 . 6 4 $ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>10.10</td><td>4.10</td><td>5.10</td><td>10.10</td><td>9.10</td><td> $7 . 7 0 { \pm } 2 . 8 8 $ </td></tr><tr><td>Surv. task</td><td> ${ \mathrm { R e A c t } }$ </td><td>2.10</td><td>3.10</td><td>12.10</td><td>14.10</td><td>1.10</td><td> $6 . 5 0 { \pm } 6 . 1 1$ </td></tr><tr><td></td><td> ${ \mathrm { R e A c t } } ^ { * }$ </td><td>-0.90</td><td>0.10</td><td>2.10</td><td>6.10</td><td>3.10</td><td> $2 . 1 0 { \pm } 2 . 7 4$ </td></tr><tr><td></td><td>Reflexion</td><td>2.10</td><td>1.10</td><td>3.10</td><td>7.10</td><td>5.10</td><td> $3 . 7 0 { \pm } 2 . 4 1$ </td></tr><tr><td></td><td>IfR</td><td>2.10</td><td>2.10</td><td>10.10</td><td>9.10</td><td>9.10</td><td> $6 . 5 0 { \pm } 4 . 0 4$ </td></tr><tr><td></td><td>NeSyC</td><td>1.10</td><td>3.10</td><td>1.10</td><td>4.10</td><td>2.10</td><td> $2 . 3 0 { \pm } 1 . 3 0 \ $ </td></tr><tr><td></td><td>WALL-E</td><td>2.10</td><td>0.10</td><td>4.10</td><td>10.10</td><td>12.10</td><td> $5 . 7 0 { \pm } 5 . 1 8$ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>-0.90</td><td>0.10</td><td>5.10</td><td>11.10</td><td>12.10</td><td> $5 . 5 0 { \pm } 6 . 0 2$ </td></tr><tr><td>All three</td><td>ReAct</td><td>2.10</td><td>5.10</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td> $0 . 9 0 { \pm } 2 . 6 8 \ $ </td></tr><tr><td></td><td> ${ \mathrm { R e A c t } } ^ { * }$ </td><td>3.10</td><td>1.10</td><td>-0.90</td><td>5.10</td><td>-0.90</td><td> $1 . 5 0 { \pm } 2 . 6 1 $ </td></tr><tr><td></td><td>Reflexion</td><td>1.10</td><td>2.10</td><td>-0.90</td><td>0.10</td><td>0.10</td><td> $0 . 5 0 { \pm } 1 . 1 4 $ </td></tr><tr><td></td><td>IfR</td><td>-0.90</td><td>-0.90</td><td>2.10</td><td>0.10</td><td>-0.90</td><td> $- 0 . 1 0 { \pm } 1 . 3 0 $ </td></tr><tr><td></td><td>NeSyC</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td>-0.90</td><td> $- 0 . 9 0 { \pm } 0 . 0 0$ </td></tr><tr><td></td><td>WALL-E</td><td>3.10</td><td>1.10</td><td>-0.90</td><td>0.10</td><td>0.10</td><td> $0 . 7 0 { \pm } 1 . 5 2 $ </td></tr><tr><td></td><td>OntoAct (ours)</td><td>2.10</td><td>4.10</td><td>-0.90</td><td>-0.90</td><td>1.10</td><td> $1 . 1 0 { \pm } 2 . 1 2 \ $ </td></tr></table>

## D.5 ALFWORLD FULL RESULTS

Table A20 reports the three evaluated methods—NeSyC, WALL-E, and OntoAct—under standard and disclosed initial observations, with the input and output tokens behind the Luna condition in Table 6.

Initial-state disclosure. Initial object locations specify which receptacle contains each object at episode reset, including contents that the agent has not yet observed through exploration. Inventory specifies the objects held by the agent, or explicitly states that its hands are empty. For this control, the initial object locations and inventory are appended as ordinary observation sentences to the first textual observation. The same disclosure procedure is applied to all three methods. Subsequent observations use the standard interface; no additional target annotations or action-effect rules are supplied. The model, learned knowledge, 134 unseen tasks, seed, and 49-step budget remain the same across the two observation conditions.

Table A20 reports both conditions for every method. With disclosure, OntoAct completes 134/134 tasks rather than 132/134, and WALL-E completes 133/134 rather than 129/134; both also use fewer steps and evaluation tokens. Our NeSyC reconstruction completes 0/134 tasks under standard observations: initially unobserved target objects prevent planning, and all 6,566 executed steps are idle actions. With disclosure it completes 35/134 tasks. The † rows in Tables 6 and A21 use this disclosed-state run for NeSyC, while the OntoAct and WALL-E rows use standard observations.

Evaluation call roles. Across the 134 standard-observation tasks, all 1,833 evaluation calls by OntoAct propose actions; learned conditions are checked symbolically. The 3,862 calls by WALL-E comprise 2,036 action proposals and 1,826 outcome predictions. These counts reconcile with Table A20: the methods execute 1,805 and 1,985 environment steps, giving 1.02 and 1.95 LLM calls per step, respectively. The prediction calls account for 90.0% of the 2,029-call difference and consume 1,919,113 tokens, or 92.2% of the observed evaluation-token difference. Knowledge-construction costs are reported separately (Table A25).

Table A20: ALFWorld with and without initial-state disclosure (Luna). Each method uses the same learned knowledge and 134 unseen tasks in both conditions, with one attempt and a 49-step budget per task. Success gives completed tasks; Steps, Calls, and input (In) and output (Out) tokens in thousands are means over all 134 episodes, including failures. The disclosure condition supplies initial object locations and inventory to every method.
<table><tr><td></td><td colspan="5">Standard observations</td><td colspan="5">Initial-state disclosure</td></tr><tr><td>Method</td><td>Success</td><td>Steps</td><td>Calls</td><td>In (k)</td><td>Out (k)</td><td>Success</td><td>Steps</td><td>Calls</td><td>In (k)</td><td>Out (k)</td></tr><tr><td>NeSyC</td><td>0/134</td><td>49.0</td><td>1.0</td><td>5.82</td><td>0.06</td><td>35/134</td><td>38.6</td><td>9.3</td><td>269.33</td><td>11.99</td></tr><tr><td>WALL-E</td><td>129/134</td><td>14.8</td><td>28.8</td><td>47.08</td><td>0.61</td><td>133/134</td><td>7.8</td><td>15.6</td><td>27.83</td><td>0.38</td></tr><tr><td>OntoAct (ours)</td><td>132/134</td><td>13.5</td><td>13.7</td><td>32.00</td><td>0.15</td><td>134/134</td><td>6.7</td><td>8.3</td><td>23.28</td><td>0.15</td></tr></table>

Table A21 reports success by task family; following Shridhar et al. (2021), the families are Pick & Place, Clean & Place, Heat & Place, Cool & Place, Examine in Light, and Pick Two & Place.

Table A21: ALFWorld success (%) by task family under the Luna condition. The six families follow Shridhar et al. (2021); parentheses give the number of unseen tasks. All is success over all 134 tasks, weighted by family size. Best values are bold; second-best All values are underlined. Ties share the same formatting. <sup>†</sup>Initial object locations and inventory are additionally provided to NeSyC.
<table><tr><td>Method</td><td>(24)</td><td>Pick Clean (31)</td><td>Heat (23)</td><td>(21)</td><td>Cool Examine Pick two (18)</td><td>(17)</td></tr><tr><td>NeSyC†</td><td>75.0</td><td>0.0</td><td>34.8</td><td>0.0</td><td>0.0</td><td>52.9 26.1</td></tr><tr><td>WALL-E</td><td>100.0</td><td>96.8</td><td>95.71</td><td>100.0</td><td>88.9</td><td>94.1 96.3</td></tr><tr><td>OntoAct (ours) 100.0</td><td></td><td>96.8</td><td></td><td>3100.0 100.0</td><td>94.4</td><td>100.0 98.5</td></tr></table>

## D.6 LLM USAGE

We separate knowledge-construction usage from evaluation usage. Construction costs are measured per construction run: for a plant in Tacit-FLE, a world in Mars, or a training corpus in ALFWorld. Evaluation costs are summed over the associated evaluation tasks and also reported per episode.

For WALL-E in Table 1, construction usage includes the original construction costs of every knowledge base used in the evaluation protocol of Appendix E.1, including reused knowledge. Evaluation costs cover only the reported episodes.

Table A22 reports the Tacit-FLE usage under Luna. Construction is a one-time cost per plant. OntoAct processes each document with the typed schema and elicits operating practices over the joined corpus, so its construction cost per plant exceeds that of WALL-E, IfR, and Reflexion. Evaluation dominates the total: on medium, construction for the 15 plants sums to 0.88M tokens against 9.6M evaluation tokens for OntoAct, which uses the fewest evaluation tokens per episode at every scale (excluding NeSyC).

Table A22: Tacit-FLE LLM usage (Luna). Construction calls and tokens are per plant (15 plants per scale); evaluation calls and tokens are per episode (28, 36, and 40 episodes). Tokens sum input and output in thousands (k). ReAct and ReAct<sup>\*</sup> construct no knowledge; NeSyC, which executes no repair action (Appendix D.1), is omitted.
<table><tr><td></td><td colspan="2">Construction (per plant)</td><td colspan="2">Evaluation (per episode)</td></tr><tr><td>Method</td><td>Calls</td><td>Tokens (k)</td><td>Calls</td><td>Tokens (k)</td></tr><tr><td>Small</td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct</td><td>0</td><td>0</td><td>22.2</td><td>154.6</td></tr><tr><td>ReAct*</td><td>0</td><td>0</td><td>22.9</td><td>166.6</td></tr><tr><td>Reflexion</td><td>6.8</td><td>2.3</td><td>23.2</td><td>170.4</td></tr><tr><td>IfR</td><td>6.8</td><td>3.7</td><td>13.8</td><td>135.3</td></tr><tr><td>WALL-E</td><td>13.4</td><td>11.7</td><td>88.0</td><td>436.6</td></tr><tr><td>OntoAct (ours)</td><td>13.1</td><td>14.8</td><td>16.2</td><td>128.0</td></tr><tr><td>Medium</td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct</td><td>0</td><td>0</td><td>41.8</td><td>619.0</td></tr><tr><td>ReAct*</td><td>0</td><td>0</td><td>38.7</td><td>564.1</td></tr><tr><td>Reflexion</td><td>32.0</td><td>10.6</td><td>46.4</td><td>734.3</td></tr><tr><td>IfR</td><td>32.0</td><td>17.1</td><td>21.5</td><td>422.0</td></tr><tr><td>WALL-E</td><td>39.1</td><td>34.9</td><td>152.9</td><td>1574.0</td></tr><tr><td>OntoAct (ours)</td><td>48.8</td><td>58.7</td><td>27.1</td><td>265.5</td></tr><tr><td colspan="5">Large</td></tr><tr><td>ReAct</td><td>0</td><td>0</td><td>48.8</td><td>1204.3</td></tr><tr><td>ReAct *</td><td>0</td><td>0</td><td>56.4</td><td>1403.6</td></tr><tr><td>Reflexion</td><td>63.1</td><td>20.7</td><td>63.1</td><td>1606.2</td></tr><tr><td>IfR</td><td>63.1</td><td>33.8</td><td>32.2</td><td>1038.9</td></tr><tr><td>WALL-E</td><td>71.5</td><td>62.8</td><td>170.6</td><td>2541.4</td></tr><tr><td>OntoAct (ours)</td><td>95.8</td><td>119.7</td><td>24.6</td><td>309.4</td></tr></table>

For Mars, Table A23 reports evaluation costs for all model conditions and Table A24 reports construction costs. Each method–model aggregate in these tables covers 40 evaluation episodes across eight worlds. Figure 5 uses the 35 Luna episodes from the seven modified worlds, matching the main performance average. For each world, we divide the total evaluation input and output tokens by the total executed environment steps across its five episodes. We then average these seven rates with equal world weights; the figure’s Score is the corresponding mean of the seven pooled world Scores. Environment steps count primitive controller interactions, not LLM calls or high-level command proposals. This rate normalizes for interaction length, and the figure pairs it with achievement Score. Knowledge-construction costs are excluded from this rate and reported separately.

For the same seven-world Luna population, pooled input tokens per call are 2.36k for OntoAct and 4.26k for ReAct<sup>\*</sup>. Their equally weighted world-average call rates are similar: 0.3655 and 0.3639 per environment step. Prompt inspection shows that OntoAct supplies compact learned conditions, whereas ReAct<sup>\*</sup> retains five retrieved transition documents; both retain ten turns of history. Across all 35 paired initial observations, which are identical between methods, first-call input averages 1,053 versus 2,982 tokens. The usage difference therefore lies primarily in input context.

The ALFWorld evaluation costs appear alongside success in Table 6, with the full breakdown in Appendix D.5. Table A25 reports the separate knowledge-construction costs.

An LLM call is one completed model response, including internal planning and verification responses; token counts are those reported by the provider. Input/call is the total input tokens divided by the number of recorded calls; it measures the mean context supplied to the model. Total tokens is the sum of input and output tokens before rounding, averaged per evaluation episode or per construction run as specified in each table. Knowledge-construction, planning, explanation, and verification call are assigned to their corresponding phase.

Table A23: Mars evaluation resource use. Means over 40 episodes per method and model (eight worlds, five episodes each). Steps are environment steps; Calls, In, Out, and Total tokens are per episode. Input/call is mean input tokens per call; Total tokens sums input and output before rounding. Token quantities are in thousands (k). Knowledge-construction costs are reported separately in Appendix D.6.
<table><tr><td></td><td></td><td></td><td></td><td></td><td>Input/call</td><td>Total tokens</td></tr><tr><td>Method</td><td>Steps</td><td>Calls</td><td>In (k)</td><td>Out (k)</td><td>(k)</td><td>(k)</td></tr><tr><td>Luna ReAct</td><td></td><td>60.8</td><td>109.13</td><td>1.42</td><td>1.79</td><td>110.55</td></tr><tr><td>ReAct* 水</td><td>198.6 252.5</td><td>83.0</td><td>354.22</td><td>2.12</td><td>4.27</td><td>356.35</td></tr><tr><td>Reflexion</td><td>249.5</td><td>75.8</td><td>174.74</td><td>2.05</td><td>2.31</td><td>176.79</td></tr><tr><td>IfR</td><td>218.2</td><td>55.2</td><td>298.10</td><td>12.45</td><td>5.40</td><td>310.55</td></tr><tr><td>NeSyC</td><td>177.9</td><td>183.1</td><td>2237.98</td><td>13.65</td><td>12.22</td><td>2251.63</td></tr><tr><td>WALL-E</td><td>220.0</td><td>204.1</td><td>544.71</td><td>5.26</td><td>2.67</td><td>549.97</td></tr><tr><td>OntoAct (ours)</td><td>304.8</td><td>103.8</td><td>244.95</td><td>1.01</td><td>2.36</td><td>245.95</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LunaR</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct 水</td><td>220.4</td><td>69.0</td><td>127.57</td><td>7.70</td><td>1.85</td><td>135.27</td></tr><tr><td>ReAct*</td><td>267.2</td><td>91.2</td><td>395.61</td><td>11.01</td><td>4.34</td><td>406.62</td></tr><tr><td>Reflexion</td><td>258.0</td><td>83.0</td><td>198.72</td><td>9.60</td><td>2.39</td><td>208.33</td></tr><tr><td>IfR</td><td>291.7</td><td>60.5</td><td>320.37</td><td>32.28</td><td>5.30</td><td>352.65</td></tr><tr><td>NeSyC</td><td>208.6</td><td>192.3</td><td>2182.93</td><td>114.77</td><td>11.35</td><td>2297.70</td></tr><tr><td>WALL-E</td><td>228.6</td><td>148.9</td><td>379.76</td><td>15.21</td><td>2.55</td><td>394.97</td></tr><tr><td>OntoAct (ours)</td><td>306.6</td><td>84.5</td><td>204.36</td><td>11.29</td><td>2.42</td><td>215.64</td></tr><tr><td>Terra</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct 水</td><td>252.1</td><td>69.9</td><td>126.91</td><td>2.13</td><td>1.82</td><td>129.04</td></tr><tr><td>ReAct*</td><td>262.6</td><td>70.1</td><td>300.29</td><td>2.15</td><td>4.28</td><td>302.45</td></tr><tr><td>Reflexion</td><td>211.6</td><td>62.8</td><td>148.76</td><td>1.98</td><td>2.37</td><td>150.73</td></tr><tr><td>IfR</td><td>256.0</td><td>42.8</td><td>207.06</td><td>7.34</td><td>4.84</td><td>214.39</td></tr><tr><td>NeSyC</td><td>188.1</td><td>183.6</td><td>2067.74</td><td>13.62</td><td>11.27</td><td>2081.36</td></tr><tr><td>WALL-E</td><td>210.3</td><td>146.2</td><td>319.72</td><td>2.81</td><td>2.19</td><td>322.53</td></tr><tr><td>OntoAct (ours)</td><td>334.1</td><td>91.9</td><td>219.66</td><td>0.91</td><td>2.39</td><td>220.57</td></tr></table>

Table A24: Mars knowledge-construction resource use. Calls, In, Out, and Total tokens are per construction run. NeSyC constructs its rules once per world (eight runs); its rule refinements during play are counted as evaluation (Table A23). ReAct and ReAct<sup>\*</sup> require no LLM-based knowledge construction; their input length is undefined. Input/call is mean input tokens per call; Total tokens sums input and output before rounding. Token quantities are in thousands (k).
<table><tr><td>Method</td><td>Calls</td><td>In (k)</td><td>Out (k)</td><td>Input/call (k)</td><td>Total tokens (k)</td></tr><tr><td>Luna</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct</td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td> $\mathrm { R e A c t } ^ { * }$ </td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td>Reflexion</td><td>20.0</td><td>12.76</td><td>2.61</td><td>0.64</td><td>15.38</td></tr><tr><td>IfR</td><td>20.0</td><td>18.28</td><td>3.88</td><td>0.91</td><td>22.16</td></tr><tr><td>NeSyC</td><td>63.5</td><td>3768.39</td><td>33.08</td><td>59.34</td><td>3801.47</td></tr><tr><td>WALL-E</td><td>255.3</td><td>1069.25</td><td>24.59</td><td>4.19</td><td>1093.84</td></tr><tr><td>OntoAct (ours)</td><td>1.0</td><td>10.91</td><td>1.34</td><td>10.91</td><td>12.25</td></tr><tr><td> $L u n a ^ { R }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { R e A c t }$ </td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td> $\mathrm { R e A c t } ^ { * }$ </td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td>Reflexion</td><td>20.0</td><td>12.76</td><td>5.01</td><td>0.64</td><td>17.77</td></tr><tr><td>IfR</td><td>20.0</td><td>18.28</td><td>7.04</td><td>0.91</td><td>25.32</td></tr><tr><td>NeSyC</td><td>63.5</td><td>3739.88</td><td>79.09</td><td>58.90</td><td>3818.98</td></tr><tr><td>WALL-E</td><td>228.1</td><td>1001.05</td><td>51.70</td><td>4.39</td><td>1052.75</td></tr><tr><td>OntoAct (ours)</td><td>1.0</td><td>11.00</td><td>3.19</td><td>11.00</td><td>14.19</td></tr><tr><td>Terra</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReAct</td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td> $\mathrm { R e A c t } ^ { * }$ </td><td>0</td><td>0</td><td>0</td><td></td><td>0</td></tr><tr><td>Reflexion</td><td>20.0</td><td>12.76</td><td>3.25</td><td>0.64</td><td>16.02</td></tr><tr><td>IfR</td><td>20.0</td><td>18.28</td><td>3.14</td><td>0.91</td><td>21.42</td></tr><tr><td>NeSyC</td><td>63.5</td><td>3729.85</td><td>12.96</td><td>58.74</td><td>3742.80</td></tr><tr><td>WALL-E</td><td>225.6</td><td>996.33</td><td>11.87</td><td>4.42</td><td>1008.20</td></tr><tr><td>OntoAct (ours)</td><td>1.0</td><td>11.20</td><td>1.12</td><td>11.20</td><td>12.32</td></tr></table>

Table A25: ALFWorld knowledge-construction LLM usage (Luna). Totals for one construction run per method on the shared 238-transition corpus. Each resulting knowledge base is reused across 134 evaluation tasks. Token quantities use thousands of tokens (k); Total tokens sums input and output before rounding. Costs exclude corpus collection and evaluation.
<table><tr><td>Method</td><td>Calls</td><td>In (k)</td><td>Out (k)</td><td>Total tokens (k)</td></tr><tr><td>NeSyC</td><td>79</td><td>1989.08</td><td>8.62</td><td>1997.70</td></tr><tr><td>WALL-E</td><td>254</td><td>349.37</td><td>14.84</td><td>364.21</td></tr><tr><td>OntoAct (ours)</td><td>15</td><td>116.43</td><td>0.94</td><td>117.36</td></tr></table>

## D.7 TACIT-FLE CANDIDATE AND VERIFICATION DIAGNOSTICS

These post hoc analyses use the 104 Luna episodes in Table 1: 28 small, 36 medium, and 40 large episodes from 15 plant worlds per scale. Anonymized identifiers are decoded only in this analysis, to compare a proposed target with the benchmark’s hidden fault target. Target alignment measures whether a candidate names the fault. NeSyC is excluded from these diagnostics because its inspected trajectories contain no repair executions.

Replacement candidates and coverage. For OntoAct and WALL-E, we extract replacement proposals before their native action-verification procedures, including rejected proposals. For IfR, we parse replacement steps in each newly generated plan, including steps that are not subsequently executed. Quoted dialogue history, inspection targets, and explanation-only text are excluded. Within each episode, we retain distinct replacement targets in order of first proposal. If $C _ { e } ^ { ( k ) }$ is the first k distinct targets and $c _ { e } ^ { * }$ the true fault, coverage is

$$
\mathrm { C o v e r a g e @ } k = \frac { 1 } { N } \sum _ { e = 1 } ^ { N } \mathcal { H } \{ c _ { e } ^ { * } \in C _ { e } ^ { ( k ) } \} .\tag{A1}
$$

Episodes with fewer than k candidates retain their complete observed candidate set and remain in the denominator. Table 2 reports $k = 3$ and full-sequence counts on medium plants; Table A26 additionally reports $k = 1$ and $k = 5$ across all scales. Candidate precision divides the number of target-aligned episode–target pairs by the number of distinct proposed episode–target pairs. We use 20,000 paired world-cluster bootstrap draws within each scale and report percentile 95% intervals. The coverage-at-three difference between OntoAct and IfR has intervals of [26.7, 66.7], [33.3, 81.0], and [62.2, 90.2] percentage points across the three scales.

Table A26: Replacement-candidate diagnostics. Targets is the mean number of distinct proposed replacement targets per episode. C@k and C@all are true-fault coverage (%) within the first k targets and the complete sequence. Precision is correct-target/total episode–target pairs. Small, medium, and large contain 28, 36, and 40 episodes.
<table><tr><td>Scale</td><td>Method</td><td>Targets</td><td>C@1</td><td>C@3</td><td>C@5</td><td>C@all</td><td>Precision</td></tr><tr><td>Small</td><td>OntoAct (ours)</td><td>2.68</td><td>50.0</td><td>75.0</td><td>82.1</td><td>82.1</td><td>23/75</td></tr><tr><td></td><td>WALL-E</td><td>4.79</td><td>3.6</td><td>35.7</td><td>50.0</td><td>50.0</td><td>14/134</td></tr><tr><td></td><td>IfR</td><td>5.79</td><td>10.7</td><td>28.6</td><td>50.0</td><td>64.3</td><td>18/162</td></tr><tr><td>Medium</td><td>OntoAct (ours)</td><td>3.36</td><td>33.3</td><td>69.4</td><td>75.0</td><td>88.9</td><td>32/121</td></tr><tr><td></td><td>WALL-E</td><td>5.14</td><td>0.0</td><td>27.8</td><td>36.1</td><td>47.2</td><td>17/185</td></tr><tr><td></td><td>IfR</td><td>9.00</td><td>0.0</td><td>11.1</td><td>38.9</td><td>66.7</td><td>24/324</td></tr><tr><td>Large</td><td>OntoAct (ours)</td><td>2.78</td><td>57.5</td><td>87.5</td><td>92.5</td><td>97.5</td><td>39/111</td></tr><tr><td></td><td>WALL-E</td><td>5.70</td><td>5.0</td><td>22.5</td><td>37.5</td><td>52.5</td><td>21/228</td></tr><tr><td></td><td>IfR</td><td>14.38</td><td>0.0</td><td>10.0</td><td>20.0</td><td>65.0</td><td>26/575</td></tr></table>

Table A27: Selection within the hypothesis ledger. Target-aligned/all tracked at baseline, then target-aligned/retained at later checkpoints (retained: open or confirmed). The last two rows give final target-aligned retention and off-target refutation; terminal-only hypotheses (9/15/14 by scale) are excluded.
<table><tr><td>Checkpoint / measure</td><td>Small</td><td>Medium</td><td>Large</td></tr><tr><td>All tracked</td><td>20/140</td><td>28/272</td><td>34/192</td></tr><tr><td>Checkpoint +1</td><td>19/68</td><td>23/112</td><td>31/98</td></tr><tr><td>Checkpoint +2</td><td>19/48</td><td>21/61</td><td>28/68</td></tr><tr><td>Checkpoint +3</td><td>19/32</td><td>20/36</td><td>28/42</td></tr><tr><td>Final</td><td>18/21</td><td>20/21</td><td>25/32</td></tr><tr><td>True-target retention</td><td>18/20</td><td>20/28</td><td>25/34</td></tr><tr><td>Off-target removal</td><td>117/120</td><td>243/244</td><td>151/158</td></tr></table>

Native hypothesis selection. A registered OntoAct hypothesis is identified by its cause, pattern reference, and ordered atoms. The registered totals are 149, 287, and 206 across small, medium, and large, corresponding to 5.32, 7.97, and 5.15 per episode. Different conditions can name the same cause: the medium cohort contains 155 distinct episode–cause pairs, or 4.31 per episode. For the selection analysis, we follow the 140, 272, and 192 hypotheses that are visible at the start of a later turn. The remaining 9, 15, and 14 hypotheses are admitted only on the terminal turn and are excluded from both the initial and retained sets. Retained hypotheses are open or confirmed at the start of the final environment turn. Table A27 gives target precision, target-aligned retention, and off-target removal on this matched population. Figure 3 follows this same population at three intermediate checkpoints aligned by each hypothesis’s admission. For a hypothesis admitted at turn a in an episode of T turns, checkpoint +k reads its status at the start of turn min(a + k, T), for $k \in \{ 1 , 2 , 3 \}$ ; Final reads turn T. An episode that has already ended retains its last recorded status. The Init. baseline includes every hypothesis in the matched population before selection. At each later checkpoint, target precision is the fraction naming the true fault among open or confirmed hypotheses. Selection includes updates after unsuccessful interventions.

Table A28: Precision of executed replacement commands. Each cell gives the percentage and correct-target/total commands. Repeated targets are counted each time. These final executions are distinct from proposed candidates and native gate passes.
<table><tr><td>Method</td><td>Small</td><td></td><td>Medium</td><td>Large</td></tr><tr><td>OntoAct (ours)</td><td>29.3% (22/75)</td><td>25.4% (29/114)</td><td></td><td>36.4% (36/99)</td></tr><tr><td>WALL-E</td><td>2.3% (3/133)</td><td></td><td>5.2% (12/232)</td><td>3.8% (14/364)</td></tr><tr><td>IfR</td><td>9.9% (18/182)</td><td></td><td>5.6% (24/426)</td><td>3.7% (26/694)</td></tr><tr><td>ReAct</td><td>8.9% (11/124)</td><td></td><td>4.7% (15/321)</td><td>3.5% (17/482)</td></tr><tr><td>ReAct</td><td>5.6% (11/197)</td><td></td><td>4.3% (17/392)</td><td>2.2% (15/688)</td></tr><tr><td>Reflexion</td><td>8.6% (12/140)</td><td></td><td>3.1% (12/387)</td><td>1.3% (8/600)</td></tr></table>

Explicit refutation and execution. We separately link repair proposals, verifier feedback, and the action executed in the same turn. For explicit refutation, we count one choice per episode, environment turn, repair verb, and target, over both replacement and refill proposals. In the medium OntoAct cohort, 139 choices receive explicit refutation feedback; 137 target a noncausal unit, and all 137 are withheld or revised in that turn. Deduplicating across turns leaves 31 off-target choices among 33 distinct episode–verb–target choices. Explicit refutation is a subset of all negative gate decisions. Across all scales, 233 of 241 explicitly refuted choices are off-target and withheld in the corresponding turn. Refutations can follow an unsuccessful earlier repair, contradicted observations, or an explained anomaly disappearing.

For broader gate diagnostics, we use the first decision on each distinct replacement target within a turn. For OntoAct, negative decisions include explicit repair, practice, or repetition feedback. For WALL-E, the decision is the final world-model Boolean after any code-conflict feedback, before execution at the proposal limit. IfR provides post-action explanations and replanning, without a corresponding pre-action verifier veto; a verifier rejection rate is therefore undefined for this method. Table A28 counts the replacement commands actually executed, including repeated targets and actions executed at the proposal limit.

Repeated off-target repairs provide a separate outcome diagnostic. We count an executed repair command to a noncausal target as repeated if an earlier repair command in the same episode targeted that unit, including a change of repair verb. Across the 104 matched episodes, OntoAct, WALL-E, and IfR issue 13, 633, and 280 such repeats, respectively.

Illustrative traces. The following cases illustrate distinct mechanisms. In a medium chain task, OntoAct proposes replacing the faulty input storage under an induced diagnostic pattern but omits its required inflow atom. The claim is not admitted, and the first executed action opens the storage. The next proposal adds the missing atom and replaces the target, completing the episode in two turns with no spurious intervention. On the same task, WALL-E initially predicts failure for an off-target locomotive replacement at turn 18. Executable-rule feedback changes the prediction to success, and the replacement executes; the episode finishes unresolved after 60 turns with 11 spurious interventions. The contrast illustrates how action-feasibility feedback can differ from evidence that a target explains the fault.

In a medium elimination task, OntoAct’s five tracked hypotheses narrow to one naming the true target, st-04. Three hypotheses on st-00 or st-01 are refuted after earlier replacements fail to end the episode, and the anomaly explained by the st-02 hypothesis clears. The st-04 hypothesis remains confirmed, and the episode succeeds in eight turns with two spurious interventions.

## E EXPERIMENTAL SETUP AND REPRODUCIBILITY

This appendix specifies the reference corpora, evaluation schedules, model configurations, and aggregation rules used in Section 4.1. Knowledge construction and prompts are described in Appendices B.2, B.4, and C.2.

## E.1 REFERENCE CORPORA AND EVALUATION TASKS

Each evaluation episode starts with fresh interaction history and transient state.

Tacit-FLE. We use the fixed reference corpora and evaluation population in Appendix A.4, with the scale-specific episode counts and turn budgets in Table A6. Corpus access follows the OntoAct construction procedure in Appendix B.2 and the baseline procedures in Appendix C.2.

For WALL-E in Table 1, each scale uses one corpus-learned knowledge base for each of its 15 worlds. All knowledge bases remain fixed during evaluation. Each episode contributes one outcome under the procedure in Appendix C.2.

Mars. Following the evaluation configuration of Zhou et al. (2025b), we use Default and seven worlds formed by changing terrain, survival mechanics, task dependencies, or their combinations. Methods share the same reference corpus of 200 recorded transitions per world. Each record contains the observed state, action, outcome, feedback, and subsequent state. For document-based processing, consecutive groups of ten records form 20 documents per world.

The agent receives a textual local observation, including a 9 × 9 view, nearby objects, inventory, and health, food, drink, and energy levels. The high-level commands are mine, attack, explore, make, place, and sleep. Each episode is capped at 1,000 environment steps and 1,000 agent turns, with ten turns of interaction history. A high-level command can consume multiple environment steps. Each method is evaluated on five seeds per world, giving 40 episodes across eight worlds for each model condition.

ALFWorld. We use the text interface and all 134 unseen evaluation tasks, with the six task familie and their sizes listed in Table A21. The reference corpus contains 238 transitions from six training games, one per task family. In each training game, 20 exploratory probes precede an expert-policy trajectory; the resulting corpus contains both successful and unsuccessful interactions. The same fixed corpus is used to construct each method’s knowledge. Each unseen task is evaluated once with a budget of 49 environment steps. OntoAct and WALL-E use standard textual observations; the reported NeSyC result additionally provides initial object locations and inventory (Appendix D.5). Environment steps include submitted think and idle commands; internal LLM calls are counted separately. We report the Luna condition for this benchmark.

## E.2 MODEL AND INFERENCE SETTINGS

The Luna condition uses the gpt-5.6-luna model with reasoning effort set to none. The Luna<sup>R</sup> condition uses the same model with reasoning effort medium; the Terra condition uses gpt-5.6-terra with effort none. Each condition uses its designated model for both knowledge construction and evaluation. We retain the provider’s default temperature settings and use the same model configuration across methods within a comparison. The Mars and ALFWorld runs cap each completion at 8,192 tokens. Retrieval uses the OpenAI text-embedding-3-small embedding model. The environment-specific proposal budgets of OntoAct are listed in Table A13.

## E.3 TASK METRICS AND AGGREGATION

Tacit-FLE. We aggregate the metrics defined in Appendix A.4.1 over all scheduled episodes at each plant scale and, where specified, by task type.

Mars. Episode reward sums the environment rewards: one unit for the first completion of each achievement, together with a health-change contribution of 0.1 per health unit. We report the mean and sample standard deviation of episode reward over five evaluation episodes in each world, using denominator $n - 1$ for the sample variance. For achievement i in evaluation episode r, let $z _ { i r } = 1$ if the achievement is attained at least once and 0 otherwise. Following the achievement-success-rate metric of Tang et al. (2024b), let $\textstyle p _ { i } = ( 1 0 0 / n ) \sum _ { r = 1 } ^ { n } z _ { i r }$ be the percentage of episodes attaining achievement i, with $n = 5$ . The achievement score is

$$
{ \mathrm { S c o r e } } = \exp \left( { \frac { 1 } { 2 2 } } \sum _ { i = 1 } ^ { 2 2 } \log ( 1 + p _ { i } ) \right) - 1 .\tag{A2}
$$

Success rates are computed across the five episodes within each world, model, and method before applying the geometric aggregation. This yields one Score per cell. Following the average-row convention of Zhou et al. (2025b), the cross-world summary excludes the Default world. The Score Average is the arithmetic mean of the seven modified-world Scores. For Reward, we first average the seven modified worlds within each matched environment seed; the Average row reports the mean and sample standard deviation across the five resulting seed averages.

ALFWorld. Task success is the fraction of the 134 unseen tasks for which the environment reports completion within the step budget. Mean environment steps are the total executed steps divided by the number of evaluation tasks, including unsuccessful episodes. Environment steps count executed commands; internal planning and verification calls contribute to LLM usage instead.