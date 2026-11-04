# ValQID: Learning the Net Value of Semantic Query Groups for Web Device Identification

Yifan Yang   
National University of   
Defense Technology   
Changsha, China   
Zhen Huang   
National University of   
Defense Technology   
Changsha, China

Zhihuang Liu Xiamen University Xiamen, China

Xionglve Li   
National University of   
Defense Technology   
Changsha, China   
Weijia Shi   
National University of   
Defense Technology   
Changsha, China   
Xichen Tan   
National University of   
Defense Technology   
Changsha, China   
Zhiping Cai   
National University of   
Defense Technology   
Changsha, China

## Abstract

Web interfaces are widely used to access network services and manage networked devices. Identifying the vendors and types of devices behind Web interfaces is a critical step in asset inventory, security management, and risk assessment. Device identity cues are dispersed across page text, configuration fields, and management functions, and interpreting them often requires linking related Web objects. Responses and objects require diferent amounts of seman tic analysis, so exhaustively processing all candidate content can introduce unnecessary computation. We present ValQID, which uses a relation graph to combine dispersed device identity cues, evaluates semantic questions eficiently with DSEJEV, and accounts for complementarity among semantic questions when estimating computation value. Based on the current identification state, ValQID weighs each question group’s expected identification benefit against its execution cost and adaptively allocates computation to identify device vendors and types. Evaluation on real Web response data shows that ValQID achieves relative gains of up to 47.1% in joint macro-F1 and up to 82.1% in macro-F1 for low-frequency vendors over a state-of-the-art method. DSEJEV speeds up semantic processing by up to 8.58× compared with general-purpose language models.

## Keywords

Web device identification, Network measurement, Relational semantic analysis, Computation value estimation, Eficient semantic inference

## 1 Introduction

Web interfaces are widely used to access network services and manage networked devices [32, 40, 49, 50]. Networked devices such as routers, gateways, cameras, and storage systems commonly provide management functions through these interfaces [11, 24, 52]. Identifying the vendors and types of these devices is fundamental to network asset visibility and security risk assessment [3, 7, 36]. Product identifiers, service information, and functional descriptions in device Web responses provide cues for inferring device attributes [18, 39]. Automated identification from these cues can turn large collections of Web responses into structured network asset information [7, 27, 42].

Methods for identifying network devices behind Web interfaces commonly use rule matching or learning-based models. Rule-based methods such as ARE generate identification fingerprints by associating response terms with product descriptions [8]. Their coverage depends on the fingerprint database, which must be updated as new devices appear and response content changes (Figure 1) [8, 16]. Learning-based methods extract identification features from response text or Web login screenshots, reducing reliance on explicit rules [45, 47]. BLMProbe uses language models to extract and verify device attributes from protocol responses [42].

![](images/edc49a129954bf319c5b01947ded66ba53a13f2ce1c74996b033ac846c87c71d.jpg)  
Figure 1: Motivation for ValQID. (a) Interface updates can invalidate fingerprints while semantic cues remain, requiring manual rule maintenance. (b) Learning-based identification uses semantic cues, with inference costs accumulating across Web responses.

Applying semantic analysis to large-scale device identification presents two challenges:

Challenge 1: Semantic understanding of dispersed identity cues. When Web responses lack explicit vendor or device names, identification relies on cues dispersed across service descriptions, configuration fields, and management functions. For example, a configuration field’s name alone may not reveal the device function it represents; interpreting it requires context from the associated managed object, enclosing form, and submission action. Matching terms or analyzing isolated fragments can overlook these relationships and miss information useful for device identification. [19, 59]

Challenge 2: Computational eficiency of large-scale semantic identification. As shown in Figure 1, inference cost grows with the number of responses, while the extent of analysis needed varies across responses [4]. Explicit product identifiers can directly support identification, whereas interpreting the functions expressed by configuration fields may require context. Pages also contain information unrelated to device identity. Performing full semantic analysis on every response adds processing overhead for responses with suficiently clear identity cues and for irrelevant fragments. This requires controlling the cost ofeach inference while adjusting the scope ofsemantic analysis to identification needs [29, 54].

Table 1: Comparison of device identification methods. ✓ and × indicate feature presence and absence. Adaptation denotes fine-tuning; Relations covers entity, structural, or cross-response links. Cost-aware selection weighs expected gains against execution costs at runtime.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Identification technique</td><td colspan="3">Identification modeling</td><td colspan="2">Computation control</td></tr><tr><td>Language models</td><td>Adaptation</td><td></td><td>Relations Adaptive Cost-aware</td><td></td></tr><tr><td>ARE [8]</td><td>Association rule mining</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>WYSIWYG [47]</td><td>CNN classification; OCR text matching</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DevTag [45]</td><td>Entity and relation extraction</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>LDF [35]</td><td>Contrastive RoBERTa; clustering</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>BLMProbe [42]</td><td>LLM labeling; BERT classification</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>ValQID (Ours)</td><td>Query groups; selection by net value</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

The non-generative decision model JEV ofers a way to reduce semantic analysis costs [2]. It evaluates predefined questions in context and directly returns probability distributions over candidate answers, avoiding autoregressive text generation and answer parsing. Combining these judgments with a Web object relation graph and allocating computation by complementary question groups’ expected identification benefit and execution cost ofers a viable approach to device identification.

We propose ValQID, which combines relational semantic analysis with adaptive computation to identify device vendors and types from Web responses. To interpret dispersed identity cues, ValQID represents Web objects and their connections in a relation graph, using these relations to provide local context for semantic questions and organize complementary question groups. To improve eficiency at scale, it uses the non-generative semantic evaluator DSEJEV to reduce the cost of individual judgments and allocates computation according to each group’s expected identification benefit and execution cost, given the current identification state. ValQID adds the judgments to the relation graph and combines the updated graph representation with character features from the response to predict device identity labels.

Our main contributions are as follows:

(1) We propose ValQID, a device identification method based on computation value. It incorporates complementarity among semantic questions into value estimation, reducing semantic queries by up to 53.16% while maintaining identification performance.

(2) We design a relation graph that integrates Web object relations and semantic judgments. Predictions from this graph are combined with character-based scores for joint device identification.

(3) We develop DSEJEV, an eficient semantic evaluator for device identification. We adapt a non-generative decision model to device-related semantic questions and achieve speedups of up to 8.58× in semantic processing.

(4) We achieve state-of-the-art device identification performance. ValQID performs best on all evaluated identification metrics across both datasets. Compared with a state-of-the-art method, it achieves relative gains of up to 47.1% in joint macro-F1 and up to 82.1% in macro-F1 for low-frequency vendors.

## 2 Related Work

Web device identification infers device identity from protocol responses and management interfaces [45, 55]. Related research examines how to represent this evidence [18, 39], select relevant content, and control the cost of acquiring and processing it [3, 7]. Table 1 compares representative device identification methods along two dimensions: identification modeling and computation control.

## 2.1 Web Device Identification

Content-based methods derive fingerprints from textual or structural patterns. ARE associates response terms with product descriptions to mine identification rules [8]; fine-grained firmware fingerprinting derives DOM vectors from Web files and verifies versions through HTTP request–response comparisons [20]. Visual identification uses login-page appearance: WYSIWYG predicts vendors from screenshots and extracts device types and models through optical character recognition and text matching [47]. Where component reuse and rebranding produce similar responses, IoTXray links devices to their original manufacturers using multiple sources [51].

Language models learn representations of response semantics and infer device attributes [58]. LDF uses contrastively trained RoBERTa embeddings to cluster service banners and construct textual fingerprints [35]. For label construction, BLMProbe assigns extraction and verification roles to large language models, then transfers labels through protocol associations across ports [42].

## 2.2 Structural Webpage Representations and Content Selection

Webpage representation models preserve the context of text by en coding its location in the document hierarchy [25, 59]. MarkupLM encodes node positions with XPath and learns node relationships during pretraining [19]. WebFormer represents relations among HTML nodes through graph attention and connects nodes to their text through node–text attention [46]. LCoW learns a contextualization module that extracts task-relevant interface elements and explains their functions for a decision-making agent [17].

Structural information also guides the selection of webpage content for downstream tasks [14]. In web interaction, MindAct filters candidate elements with a small model before a large model selects the target and predicts an action [6]. AgentOccam simplifies web page observations and retains pivotal elements together with their structural neighborhoods [53]. Prune4Web delegates selection to a program that traverses the DOM and applies a scoring template with model-generated, task-specific keywords and weights [56]. For retrieval-augmented question answering, HtmlRAG first filters an HTML block tree by embedding similarity, then prunes it with path generation probabilities while retaining the selected blocks’ hierarchical context [41].

## 2.3 Cost-Aware Semantic Decision and Execution

Adaptive feature acquisition weighs the predictive benefit of additional observations against their acquisition cost [5, 10, 15, 38]. For individual features, DIME estimates value through conditional mutual information [9], and SEFA uses class-weighted gradient norms in a stochastic latent representation [28]. To account for jointly informative features, ACO evaluates expected loss and cost over future feature combinations [44], while Information Templates learns templates of feature combinations for instance-level acquisi tion [13].

Semantic judgments call for eficient evaluation and outputs that programs can combine directly. TypeSafe introduced Jev, a System One model for binary verification, choice, and ordinal scoring [2]. Its interface evaluates independent questions over shared context and returns typed probabilities for classification, scoring, and control flow [43]. LLM-as-Jev realizes this interface through the token probabilities of candidate identifiers, studying both training-free readout and candidate-selection fine-tuning in general-purpose language models [21].

Semantic data systems address cost at the level of operators and execution plans [22, 37, 48]. LOTUS expresses filtering, joining, and sorting as semantic operators and optimizes execution with proxy models, cascades, and approximation algorithms. Statistical accu racy guarantees are given at the operator level relative to reference algorithms [30]. Palimpzest and its Abacus optimizer select physical plans based on output quality, latency, and monetary cost [23, 34]. Sema reduces inference cost through expression compression, operator fusion, prompt batching, and adaptive query execution [31]. Related studies protect LLM inputs through risk-aware privacy preservation [26] and semantic-aware text sanitization [57].

## 3 Methodology

ValQID uses relations among Web objects to organize semantic question groups, then decides whether to execute a group or stop based on each group’s expected identification benefit and execution cost. Figure 2 outlines the workflow.

## 3.1 Problem Formulation and Overview

Given a Web observation � extracted from a target device’s HTTP response, we predict its vendor and device type. We denote the true labels by $\boldsymbol { y } = \left( y _ { v } , y _ { d } \right)$ and the predictions by $\hat { y } = ( \hat { y } _ { v } , \hat { y } _ { d } )$ . Interpreting a Web field often requires knowing the role of the object it belongs to and its associated operations. Evaluating semantic questions adds computational cost; the benefit to identity prediction depends on the field contents and the results already obtained. We study cost-sensitive sequential decision making over a fixed observation �: which groups of semantic questions to execute and when to stop.

ValQID uses Web object relations to provide local question context and organize candidate question groups. Ofline screening retains semantic conditions and relation templates with individual or joint identification benefits. We learn computation value from the reduction in identity loss after adding a group’s results, together with the group’s execution cost, and use this value to allocate semantic computation at the current state.

Let $s _ { k }$ denote the state before decision round $k ,$ containing the initial relation graph and the option probabilities obtained so far. Policy � selects a question group $g _ { k }$ or stops. Our semantic evaluator, DSEJEV (Device Semantic Evaluation JEV), returns option probabilities for the selected questions. Attaching these results to their target objects forms $s _ { k + }$ and updates the identity prediction. Upon stopping, joint decoding returns the labels ${ \hat { y } } .$

We seek a policy � that minimizes expected identity loss and weighted execution cost:

$$
\mathcal { T } ( \pi ) = \mathbb { E } \left[ \ell _ { y } ( s _ { T } ) + \lambda \sum _ { k = 0 } ^ { T - 1 } \bar { c } ( g _ { k } ) \right] .\tag{1}
$$

Here $T \leq H$ is the number of executed rounds, � is the maximum number of decision rounds, and $s _ { T }$ is the terminal state. With $T =$ 0, the model predicts from the initial Web observation without executing semantic questions. The identity loss $\ell _ { y } ( s _ { T } )$ is computed from the probabilities assigned to the true vendor and type labels; $\bar { c } ( g _ { k } )$ is the normalized group execution cost, and $\lambda \geq 0$ weights that cost. The expectation is over data samples and randomness in the execution process.

## 3.2 Relation-Constrained Semantic Question Groups

ValQID parses the objects and connections in � into a typed relation graph $G _ { x } \ = \ \left( V _ { x } , E _ { x } , \tau _ { V } , \tau _ { E } \right)$ . Nodes store object contents, edges record relations that can be parsed from the observation, and � and $\tau _ { E }$ specify node and edge types. Deterministic graph extraction records which object each text segment belongs to and preserves segment boundaries and order. We use these relations to assemble the local context for each semantic question.

Question groups. We select conditions and relation templates through ofline screening on training data. A condition specifies a semantic question, a finite option set, and the objects to which it applies. Binding a condition to an object $u _ { a }$ creates an atomic question $\boldsymbol { a } = \left( u _ { a } , q _ { a } , O _ { a } , E _ { a } \right)$ , where $q _ { a }$ is the question, $O _ { a }$ includes an insuficient-evidence option, and $E _ { a }$ contains the target object’s content and the local context needed to interpret it. For an input control, this context can include its enclosing form and the associated submission operation.

![](images/2b8198eea34b27994d0d15a0957962a15865fa14bef3f86d0e1a259ab2b647d3.jpg)  
Figure 2: ValQID workflow. Web object relations organize complementary semantic question groups. A shared relation encoder supports identity prediction and group value estimation. ValQID selects groups by predicted net value and updates the graph with DSEJEV’s option probabilities.

A relation template � specifies the object roles associated with two questions, their connections, and the order of those roles. Two distinct atomic questions form a relation-constrained question group when their target objects match a template. The targets may be the same object, be connected by a typed path, or share a form, operation, or resource. For example, questions about a control’s field role and a submission operation’s function form a relationconstrained group when the two objects share a form. Templates match roles and relations and can therefore be instantiated across diferent page layouts.

Condition discovery. Using five-fold cross-validation grouped by content, we identify confusable class pairs from the initial identity predictor’s out-of-fold errors. A generative model examines the misclassified training examples and their object relations to propose candidate conditions and templates. We bind these conditions to eligible objects in training observations and use the base JevK5 model to obtain their option probabilities.

Logistic regression probes assess whether these probabilities improve identity prediction, using five-fold cross-validation grouped by content. The base features contain the out-of-fold initial identity prediction, candidate identifier, and option-validity masks. Each candidate contributes one applicable binding per observation; for a relation template and its condition pair, this binding yields questions � and �. Using the same bindings and folds, we compare four configurations: the base features alone, with the option probabilities of �, with those of $b ,$ and with both probability vectors and their outer product. The outer product represents interactions between the questions. Averaging the out-of-fold identity losses over the same applicable observations yields $L _ { 0 } , L _ { a } , L _ { b } ,$ , and $L _ { a b }$ , respectively.

We retain an individual condition when adding its option probabilities reduces identity loss. A relation template is retained when its matched question pairs lower the average loss relative to the base features, $L _ { a b } < L _ { 0 } ,$ and each question alone, $L _ { a b } < \operatorname* { m i n } ( L _ { a } , L _ { b } )$ with positive interaction gain:

$$
\Gamma ( a , b ) = L _ { a } + L _ { b } - L _ { a b } - L _ { 0 } > 0 .\tag{2}
$$

This gain measures how much the joint loss reduction exceeds the sum ofthe two individual reductions. These criteria select templates with complementary identification evidence for the candidate catalog. We also retain their required conditions and use development data to select the library size. The retained conditions specify which local judgments the evaluator learns (Section 3.5).

For the current observation �, we bind selected conditions to eligible objects and organize the resulting questions using selected relation templates to form the candidate catalog $\mathcal { G } _ { x }$ . The catalog contains singletons (�) and relation-constrained groups (�, �, �), with $a \neq b .$ The catalog and each atomic question’s local request remain fixed throughout identification.

## 3.3 Semantic Execution, State Update, and Identity Prediction

During identification, DSEJEV executes the local questions in the selected group. We obtain DSEJEV by adapting the 4B JevK5 decision model [1] on local questions instantiated from the selected conditions (Section 3.5). Given the local input $( E _ { a } , q _ { a } , O _ { a } )$ of atomic question �, DSEJEV computes option logits $\mathbf { z } _ { a }$ and returns

$$
\begin{array} { r } { { \bf p } _ { a } = \mathrm { s o f t m a x } _ { O _ { a } } ( { \bf z } _ { a } / \tau ) , } \end{array}
$$

where � is a calibration temperature. The two atomic questions in a relation-constrained group are evaluated separately using their respective fixed local requests. We add both complete option prob ability distributions in a single state update.

A text encoder using frozen JevK5 input embeddings encodes each object’s content. Each completed question adds a result node encoding its full option-probability vector and linked to its target object. State �<sub>�</sub> records the fixed observation �, initial graph $G _ { x } ,$ completed results $Z _ { k }$ , action history, remaining question budget $b _ { k }$ , and remaining rounds $r _ { k }$ . Let $G _ { x }$ ⊕ $Z _ { k }$ denote the graph with the result nodes added. The relation encoder aggregates object and result representations along typed edges, then combines the graph-level representation with the budget and remaining rounds:

$$
( h _ { k } , \{ h _ { u , k } \} ) = F _ { \theta } ( G _ { x } \oplus Z _ { k } , b _ { k } , r _ { k } ) .\tag{3}
$$

Here $h _ { k }$ is the shared graph-level state representation, and $h _ { u , k }$ represents object � in the current state. These object representations also enter group value estimation.

The relation branch predicts from $h _ { k }$ and updates its output as semantic results arrive. The character branch computes fixed classification scores from character features of the full HTTP response once per observation. We denote the task-level distributions used for identity-loss computation by $p _ { v } ( { \bf \cdot } \mid s _ { k } )$ and $p _ { d } ( \cdot \mid s _ { k } )$ . In configurations with task-level fusion, weighted character log probabilities are added to the relation logits before softmax to obtain these distributions.

During ofline training, we use the training labels to compute the identity loss from these probabilities:

$$
\ell _ { y } ( s ) = m _ { v } \ell _ { v } ( s ) + \omega _ { d } m _ { d } \ell _ { d } ( s ) .
$$

For task $j \in \{ v , d \} , \ell _ { j } ( s ) = - \log p _ { j } ( y _ { j } \mid s )$ . The masks $m _ { v } , m _ { d } ~ \in$ {0, 1} indicate valid labels, and $\omega _ { d }$ weights the device type task. The decrease in this loss after executing a question group measures the identification benefit of its semantic results. Together with execution cost, it provides supervision for group value learning in Section 3.4.

When the policy stops at $^ { s _ { T } , }$ joint decoding combines the task scores and fixed joint evidence:

$$
S ( y \mid x , s _ { T } ) = A _ { v } ( y _ { v } \mid x , s _ { T } ) + A _ { d } ( y _ { d } \mid x , s _ { T } ) + C ( y \mid x ) .
$$

Here $A _ { v }$ and $A _ { d }$ are the terminal vendor and type scores after character fusion and scaling, carrying forward any character contributions already present in $\ p _ { v } , p _ { d } .$ The fixed term $C ( y \mid x )$ combines scores from character-based linear classifiers trained on vendor– type pairs with label-pair associations and frequency corrections from the training data. In the DevTag configuration, it also incorporates labels from similar training observations. The task-level loss supervises computation selection; joint decoding produces the final labels:

$$
{ \hat { y } } = \arg \operatorname* { m a x } _ { y } S ( y \mid x , s _ { T } ) .
$$

## 3.4 Group Value and Computation Selection

Each observation has a maximum of� decision rounds and a budget of � atomic questions. Each round executes one question group and updates the state. The legal group set ${ \mathcal { G } } ( s ) \subseteq { \mathcal { G } } _ { x }$ contains groups whose questions have not been executed and fit the remaining question budget. When no rounds remain, $\mathcal { G } ( s ) = \emptyset$

A value model estimates each legal group’s net value at the current state to select a group or stop.

Group execution cost. We measure a group’s amortized execution time $c ( g )$ under fixed reference batching and inference settings and normalize it by a reference cost $c _ { 0 } > 0 \colon \bar { c } ( g ) = c ( g ) / c _ { 0 } .$ . We train a group cost predictor on measured execution costs of singletons and relation-constrained groups. Value training targets use measured normalized costs $\bar { c } ( g )$ . The cost predictor supplies the normalized estimate $\widehat { c } ( g ) / c _ { 0 }$ to the value head before group selection.

Value learning. Let $s ^ { g }$ be the state after executing � and attaching all its option probabilities to their objects. The loss reduction from � to $s ^ { g }$ evaluates the group’s results together, allowing complementary questions to contribute jointly to its value. Subtracting the weighted execution cost gives the immediate net return:

$$
R _ { \lambda } ( s , g ) = \ell _ { y } ( s ) - \ell _ { y } ( s ^ { g } ) - \lambda \bar { c } ( g ) .\tag{4}
$$

With one round remaining, the value target is $Y _ { 1 } ( s , g ) = R _ { \lambda } ( s , g )$ Let $S ( s ^ { g } ) \subseteq { \mathcal { G } } ( s ^ { g } )$ contain legal successor groups with recorded semantic outputs, execution costs, and state transitions. For $1 \ : <$ $r \leq H$ , we construct the target recursively from these transitions:

$$
\begin{array} { r l r } & { } & { Y _ { r } ( s , g ) = R _ { \lambda } ( s , g ) \qquad } \\ & { } & { \qquad + \operatorname* { m a x } \left( \{ 0 \} \cup \{ Y _ { r - 1 } ( s ^ { g } , g ^ { \prime } ) : g ^ { \prime } \in S ( s ^ { g } ) \} \right) . } \end{array}\tag{5}
$$

The zero-valued option corresponds to stopping; an empty successor set gives zero continuation return.

Summing Equation (4) along a trajectory yields

$$
\sum _ { k = 0 } ^ { T - 1 } R _ { \lambda } ( s _ { k } , g _ { k } ) = \ell _ { y } ( s _ { 0 } ) - \ell _ { y } ( s _ { T } ) - \lambda \sum _ { k = 0 } ^ { T - 1 } \bar { c } ( g _ { k } ) .
$$

Once the parameters and task-level fusion weights used to compute $\ell _ { y }$ are fixed, $\ell _ { y } ( s _ { 0 } )$ is independent of the policy. Maximizing the expected cumulative net return therefore corresponds to minimizing Equation (1).

The shared value head $Q _ { \phi }$ uses $h _ { k } ,$ , target-object and condition encodings, and, for relation-constrained groups, the template encoding and interaction features between the two questions. It also receives the remaining question budget, a round condition, normalized cost, and cost weight �. Trained against $Y _ { r } ,$ , it predicts group $\boldsymbol { g ^ { \prime } } \mathbf { s }$ s net value $\widehat { Q } _ { k } ( g )$ at round �. During identification, one-step evaluation sets the round input to 1 at every decision; evaluation based on remaining rounds uses $r _ { k }$

If $\mathcal { G } ( s _ { k } )$ is empty, the policy returns the current identity prediction. Otherwise, it selects the group with the highest predicted net value:

$$
g _ { k } ^ { \star } = \arg \operatorname* { m a x } _ { g \in { \mathcal { G } } ( s _ { k } ) } \widehat { Q } _ { k } ( g ) .\tag{6}
$$

```latex
Algorithm 1 ValQID Identification
Require: Web observation �, condition and template library,
trained models, maximum decision rounds �, question budget
$B ,$ cost weigh $: \lambda ,$ value-evaluation mode
Ensure: Identity labels $\hat { y } = ( \hat { y } _ { v } , \hat { y } _ { d } )$
1: Build $G _ { x } , G _ { x } ,$ and fixed local requests.
2: Compute and retain character-branch outputs and joint-score
contributions that depend only on �.
3: Initialize $s _ { 0 }$ with $Z _ { 0 } = \emptyset , b _ { 0 } = B , r _ { 0 } = H ,$ , and an empty action
history; $k \gets 0 .$
4: loop
5: $( h _ { k } , \{ h _ { u , k } \} )  F _ { \theta } ( G _ { x } \oplus Z _ { k } , b _ { k } , r _ { k } ) .$
6: Compute the current vendor and type distributions $\rho _ { v } ( \cdot |$
$s _ { k } )$ and $p _ { d } ( \cdot \mid s _ { k } )$
7: Form $\mathcal { G } ( s _ { k } )$ from groups with no completed questions and
at most $b _ { k }$ questions.
8: Set $\begin{array} { r } { G ( s _ { k } ) \gets \emptyset \mathrm { i f } r _ { k } = 0 . } \end{array}$
9: if $\mathcal { G } ( s _ { k } ) = \emptyset$ then return ${ \hat { y } } = \arg \operatorname* { m a x } _ { y } S ( y \mid x , s _ { k } ) .$
10: Predict $\widehat { c } ( g ) / c _ { 0 }$ for each $g \in { \mathcal { G } } ( s _ { k } ) .$
11: Use $Q _ { \phi }$ to obtain each $\widehat { Q } _ { k } ( g )$ from shared representations,
group features, $b _ { k } , { \widehat c } ( g ) / c _ { 0 } , \lambda ,$ with round input 1 for one-step
evaluation and $r _ { k }$ otherwise.
12: $g _ { k } ^ { \star } \gets \arg \operatorname* { m a x } _ { g \in { \mathcal { G } } ( s _ { k } ) } \widehat { Q } _ { k } ( g )$
13: if $\widehat { Q } _ { k } ( g _ { k } ^ { \star } ) \leq 0$ then return $\hat { y } =$ arg max $_ y S ( y \mid x , s _ { k } )$
14: for each atomic question � in $g _ { k } ^ { \star }$ do
15: $\mathbf { p } _ { a } \gets \mathrm { D S E J E V } ( E _ { a } , q _ { a } , O _ { a } ) .$
16: end for
17: Form $Z _ { k + 1 }$ by adding all returned ${ \bf p } _ { a }$ to $Z _ { k } ,$ each attached
to its target object $u _ { a } .$
18: Record the completed questions in the action history.
19: $b _ { k + 1 }  b _ { k } -$ number of questions in $g _ { k } ^ { \star } .$
20: $r _ { k + 1 } \gets r _ { k } - 1 ;$ form $s _ { k + 1 } ; k \gets k + 1$
21: end loop
```

The group is executed when $\widehat { Q } _ { k } ( g _ { k } ^ { \star } ) > 0 ;$ otherwise, the model returns its current identity prediction. After execution, the results update the state, the question budget decreases by the number of completed questions, and the remaining rounds decrease by one before the next decision.

Algorithm 1 summarizes the identification procedure.

## 3.5 Training and Identification

We fix the condition and template library selected in Section 3.2 and bind the retained conditions to eligible objects in the training observations. Each adaptation example contains a question $q _ { a } ,$ finite options $O _ { a } ,$ and local context $E _ { a ; }$ , which includes the target object’s content and the relations needed to interpret it. We use Qwen3.5-397B-A17B as the teacher to select an answer from $O _ { a }$ based on the local context $E _ { a } ,$ providing evidence references and a brief rationale. Three researchers manually review all annotations to check whether the selected answers are supported by the local evidence. We fine-tune the base JevK5 model on these labeled examples using LoRA [12] to obtain DSEJEV.

We then freeze DSEJEV and use its option probabilities to train the relation encoder and identity predictor. Training includes initial states and states augmented with singleton or relation-constrained group results, whether obtained in one round or accumulated across rounds.

After training the identity predictor, we select task-level fusion weights on development data, then freeze the relation encoder, task heads, and these weights. On a separate value-fitting subset, we compute task-level loss reductions and measured execution costs for recorded transitions to construct the corresponding targets $Y _ { r } .$ . These targets train the candidate encoder and value head. For each data source, we follow the same training procedure with a separate label space, condition library, and set of model parameters. Deployment settings are selected on development data.

We then fit the joint linear classifiers using training samples with valid vendor and type labels, keeping the previously trained models fixed. Joint-decoding weights are selected on development data and remain fixed during identification.

Identification. Observations independently select question groups using the policy in Section 3.4. Selected groups are collected into batches for DSEJEV execution, and the returned probabilities are attached to their target objects in each observation. Each observation updates its state and continues or stops independently according to Algorithm 1.

## 4 Experiments

We evaluate ValQID’s identification performance and eficiency through the following questions:   
RQ1: How does ValQID compare with existing methods overall and on long-tail vendors?   
RQ2: How much does DSEJEV improve semantic processing speed and overall throughput?   
RQ3: How do identification performance and computation cost change when key components are removed?

## 4.1 Datasets and Experimental Setup

We evaluate device vendor and type identification using two Web response datasets, DevTag and BLMProbe.

DevTag. DevTag is a public benchmark for IoT device fingerprinting, with application-layer service responses labeled by device type, vendor, and product [45]. We use its released preprocessed response texts. We deduplicate records by exact full-text matching, remove records with identical text but conflicting labels, and select HTTP responses based on their status lines. Vendor and device-type labels are extracted from the original annotations. The resulting HTTP dataset contains 75,188 unique text records covering 80 vendor classes and 20 device types, including cameras, routers, gateways, switches, and printers. It retains low-frequency vendor and device-type classes for evaluating long-tail identification performance.

BLMProbe. BLMProbe releases protocol responses collected through active probing of Internet-connected devices, together with vendor, device-type, and product labels [42]. We extract non-empty HTTP/HTTPS responses and their existing vendor and devicetype labels from this data. The input retains response headers and available HTML bodies, including page text, forms, and scripts. The dataset contains 11,164 responses covering 67 vendor classes and five device types: routers, gateways, digital video recorders, switches, and printers.

Table 2: Identification performance on DevTag and BLMProbe (%). Acc. denotes accuracy; Prec., Rec., and M-F1 denote macroaveraged precision, recall, and F1. Joint metrics evaluate vendor–type pairs. ↑ indicates higher is better. Red ↓ values are relative decreases from ValQID, $1 0 0 \bigl ( s _ { \mathrm { o u r s } } - s \bigr ) / s _ { \mathrm { o u r s } } ;$ bold marks the best score for each metric and dataset.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Vendor Identification</td><td colspan="4">Device-Type Identification</td><td colspan="2">Joint Identification</td></tr><tr><td>Acc. ↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>M-F1 ↑</td><td>Acc. ↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>M-F1 ↑</td><td>Acc. ↑</td><td>M-F1 ↑</td></tr><tr><td rowspan="4">DevTag</td><td>DevTag [45]</td><td></td><td></td><td></td><td></td><td>57.18↓40.4% 51.19↓ 37.4% 47.09↓35.8% 45.35↓39.9% 58.50↓39.2% 39.36↓58.8% 28.61↓68.7% 31.18↓66.3% 55.58↓ 41.7% 38.49↓ 49.1%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RoBERTa [35]</td><td>89.36↓6.8%</td><td></td><td></td><td>38.82↓52.5% 33.19↓54.8% 33.82↓55.2%89.45↓7.1%</td><td></td><td>46.74↓51.0% 44.56↓51.2% 45.18↓51.2% 86.10↓9.7%27.05↓64.2%</td><td></td><td></td><td></td><td></td></tr><tr><td>BLMProbe [42]</td><td>92.71↓3.3%</td><td></td><td></td><td>58.85↓28.0% 59.08↓19.5% 55.97↓25.9% 93.66↓2.7%</td><td></td><td></td><td></td><td></td><td>73.81↓22.7% 84.03↓8.0% 77.07↓16.8% 91.99↓3.5% 56.25↓25.7%</td><td></td></tr><tr><td>Qwen3.5 [33]</td><td>70.27↓26.7% 31.42↓61.6% 28.39↓61.3% 26.38↓65.1% 35.50↓63.1% 22.75↓76.2% 10.11↓88.9% 10.91↓88.2% 32.46↓65.9% 13.26↓82.5%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ValQID (Ours)</td><td>95.91</td><td>81.76</td><td>73.38</td><td>75.50</td><td>96.30</td><td>95.46</td><td>91.33</td><td>92.62</td><td>95.30</td><td>75.65</td></tr><tr><td rowspan="4"></td><td>DevTag [45]</td><td>13.08↓86.5% 10.46↓ 88.9%</td><td></td><td>7.96↓91.6%</td><td>8.41↓91.0%</td><td></td><td>9.32↓90.6% 40.32↓58.9% 13.19↓86.6% 11.86↓87.9%</td><td></td><td></td><td></td><td>4.39↓95.5% 3.09↓96.6%</td></tr><tr><td>RoBERTa [35]</td><td>88.58↓8.3%</td><td></td><td></td><td></td><td>62.15↓33.9% 61.78↓34.5% 59.34↓36.4% 91.52↓7.6%</td><td>70.01↓28.7% 68.52↓30.4% 69.11↓29.7% 86.15↓10.8% 49.55↓46.1%</td><td></td><td></td><td></td><td></td></tr><tr><td>BLMProbe BLMProbe [42]</td><td>91.56↓5.2%</td><td></td><td></td><td>69.84↓25.8% 71.33↓24.3%68.29↓26.8% 95.57↓3.5%</td><td></td><td></td><td></td><td></td><td>90.61↓7.7%82.97↓15.7% 85.06↓13.5% 91.44↓5.4% 62.47↓32.0%</td><td></td></tr><tr><td>Qwen3.5 [33]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>51.98↓46.2% 52.88↓43.8% 52.5044.3% 47.25↓49.3% 79.16 20.1% 87.90↓10.5% 65.55|33.4% 68.9129.9% 47.25↓51.1% 36.79↓ 60.0%</td><td></td><td></td></tr><tr><td></td><td>ValQID (Ours)</td><td>96.63</td><td>94.09</td><td>94.28</td><td>93.28</td><td>99.03</td><td>98.22</td><td>98.48</td><td>98.34</td><td>96.63</td><td>91.90</td></tr></table>

The training, development, and test sets contain 58,924, 7,774, and 8,490 responses for DevTag and 6,572, 2,222, and 2,370 responses for BLMProbe, respectively. For ValQID, we further divide each training set into two disjoint subsets at an approximate 4:1 ratio. The larger subset is used to train the relation-based identity model and discover semantic conditions; the smaller subset is used to fit the question-group value model while keeping the identity model parameters fixed. We use the development set to select model checkpoints, fusion parameters, and query configurations. All methods are evaluated on the same test responses within each dataset using a common label vocabulary and scoring procedure.

## 4.2 Comparison with Baselines

We compare ValQID with representative device identification methods, including DevTag [45], RoBERTa [35], and BLMProbe [42], as well as the general-purpose large language model Qwen3.5 [33]. Table 2 reports vendor, device-type, and joint identification results on both datasets. We use accuracy to measure overall identification performance. Since samples from major vendors account for a large share of the data, we macro-average precision, recall, and F1 to give rare and frequent classes equal weight when evaluating performance under long-tailed class distributions.

On DevTag, which has a long-tailed class distribution, ValQID improves vendor macro-F1 by 34.9% relative to the strongest baseline, BLMProbe. Joint identification requires both the vendor and device type to be predicted correctly for the same response. ValQID achieves joint accuracies of 95.30% on DevTag and 96.63% on BLM-Probe, with relative improvements of 34.5% and 47.1% in joint macro-F1 over BLMProbe, respectively. Across both datasets, ValQID achieves the best results on every reported metric for vendors, device types, and vendor–type pairs.

## 4.3 Long-Tail Analysis

To examine whether identification gains extend to long-tail vendors, we divide vendors in each dataset into high, medium, and low frequency groups based on their training sample counts. We compare ValQID with BLMProbe and RoBERTa using macro-F1 within each group, giving equal weight to each vendor. Figure 3 shows per-vendor F1 gains, with vendors ordered by decreasing training sample count.

![](images/677242cb013dc37392713f466b4886849f0242800fe1cf17f78190a132afaa69.jpg)  
Figure 3: Vendor F1 gains by training frequency. Stems show per-vendor F1 diferences in percentage points (pp). Panel labels show low-frequency macro-F1 gains as percentages of the corresponding baseline’s macro-F1.

Across both datasets, ValQID’s relative improvements over both baselines increase from the high frequency group to the medium and low frequency groups. For low frequency vendors, ValQID achieves macro-F1 scores of 66.28% on DevTag and 91.81% on BLM-Probe, corresponding to relative improvements of 82.1% and 74.8% over the BLMProbe baseline.

Table 3: Ablation of semantic queries, character-branch fusion, and value-based query selection. Identification scores are percentages; M-F1 denotes macro-F1. Semantic query and input token counts are normalized to ValQID within each dataset (1.00×), with one query per semantic question. Bold marks the best value for each metric and dataset.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Configuration</td><td colspan="2">Vendor Device type</td><td colspan="2">Joint identification</td><td colspan="2">Semantic queries</td><td>Input tokens</td></tr><tr><td>M-F1↑</td><td>M-F1↑</td><td>Acc. ↑</td><td>M-F1 ↑</td><td>(x)↓</td><td>(x)↓</td></tr><tr><td rowspan="4">DevTag</td><td>ValQID</td><td>75.50</td><td>92.63</td><td>95.31</td><td>75.66</td><td>1.00×</td><td>1.00×</td></tr><tr><td>w/o Semantic Queries</td><td>38.55</td><td>62.03</td><td>88.21</td><td>39.11</td><td>一</td><td></td></tr><tr><td>w/o Character Branch and Fusion</td><td>38.65</td><td>60.73</td><td>89.32</td><td>35.40</td><td>1.00×</td><td>1.00×</td></tr><tr><td>w/o Value-based Query Selection</td><td>75.56</td><td>92.63</td><td>95.32</td><td>75.70</td><td>1.81×</td><td>1.73X</td></tr><tr><td rowspan="4">BLMProbe</td><td>ValQID</td><td>93.28</td><td>98.34</td><td>96.63</td><td>91.90</td><td>1.00×</td><td>1.00×</td></tr><tr><td>w/o Semantic Queries</td><td>76.56</td><td>91.26</td><td>92.07</td><td>71.67</td><td>一</td><td>一</td></tr><tr><td>w/o Character Branch and Fusion</td><td>77.15</td><td>90.03</td><td>91.99</td><td>73.26</td><td>1.00×</td><td>1.00×</td></tr><tr><td>w/o Value-based Query Selection</td><td>93.28</td><td>98.34</td><td>96.63</td><td>91.90</td><td>2.14×</td><td>2.76×</td></tr></table>

![](images/6b34ed804cbc1a93fe192cbb6a6abed4e8d250c9616d45cfca2bb5e0acca84c4.jpg)  
Figure 4: Joint identification accuracy (%) versus semantic execution time (s per 1,000 responses). Circles denote BLM-Probe; triangles denote DevTag.

Low frequency vendors account for approximately half of the vendor classes in each dataset. ValQID improves F1 over BLMProbe for 22 of 36 such vendors on DevTag and 25 of 37 on BLMProbe, covering a majority of the tail classes. ValQID combines character patterns from the full response with semantic assessments of Web objects and their relations. These inputs provide both textual and functional information for identifying vendors with limited training examples.

## 4.4 Eficiency of DSEJEV

We compare DSEJEV with three alternative semantic evaluators within ValQID: Qwen3.5-4B, DeepSeek LLM 7B Chat, and Phi-4- mini-instruct (Figure 4). Across both datasets, the DSEJEV configurations achieve 3.27–8.58× speedups in semantic processing and the highest joint identification accuracy among the evaluated configurations. Using DSEJEV also raises end-to-end identification throughput to 1.20–1.83 times that of the alternative evaluator configurations.

DSEJEV is built on JevK5 [1] and fine-tuned on local semantic questions for device identification. It returns probabilities over predefined options in a single forward pass, avoiding autoregressive text generation and answer parsing. These probability vectors feed directly into identity prediction and subsequent computation selection.

## 4.5 Ablation Study

Table 3 compares ValQID with three ablated configurations. For the semantic-query ablation, we disable question execution and result injection while retaining the identity model, character branch, and fusion components. Semantic queries improve joint macro-F1 by 36.55 percentage points on DevTag and 20.23 percentage points on BLMProbe.

The character-branch and fusion ablation jointly removes the character classifiers, label-pair and frequency corrections, joint classifiers, and the DevTag nearest-neighbor component. It uses the same semantic-question trace and relation-model outputs as ValQID, with vendor and device type predicted independently. Together, the character and fusion components increase joint macro-F1 from 35.40% to 75.66% on DevTag and from 73.26% to 91.90% on BLMProbe.

Compared with fixed-order execution, value-based query selection reduces semantic query counts by 44.74% on DevTag and 53.16% on BLMProbe, and input token usage by 42.08% and 63.83%, respectively, while maintaining comparable joint identification performance. Both configurations use the same question catalog, DSEJEV evaluator, identity predictor, and question budget. Fixed-order execution follows a predefined sequence until the budget is exhausted. ValQID selects the eligible question group with the highest predicted net value and stops when no eligible group has positive net value.

## 5 Conclusion

We presented ValQID, a method for identifying device vendors and types from Web responses. It accounts for complementarity among semantic questions when estimating computation value: it groups questions using Web object relations and selects groups based on their expected identification benefit and execution cost given the current identification state. On real Web response data, ValQID achieves relative gains of up to 47.1% in joint macro-F1 and up to 82.1% in macro-F1 for low-frequency vendors over a state-ofthe-art method. DSEJEV speeds up semantic processing by up to 8.58× compared with the three language-model evaluators. These results show that assessing the value of complementary question groups allows relationships among identity cues to guide semantic computation, enabling accurate and eficient device identification. Future work could explore how to adapt semantic question groups and their value estimates as firmware and Web interface updates change identity cues.

## References

[1] alibiserikbay. 2026. JevK5 v0.3: Open-Weight Jev Alternative. Hugging Face model card. Retrieved October 24, 2026 from https://huggingface. co/alibiserikbay/JevK5/blob/c4f7fdb3aeab5582336406e78d3bef11bf98833d README.md

[2] Diogo Almeida. 2026. Introducing System One Models & Jev. TypeSafe AI Blog. Retrieved October 24, 2026 from https://typesafe.ai/blog/introducing-systemone-models-and-jev

[3] Jinsong Chen, Mengying Wu, Geng Hong, Baichao An, Mingxuan Liu, Lei Zhang, Baojun Liu, Haixin Duan, and Min Yang. 2025. Beyond Exploit Scanning: A Functional Change-Driven Approach to Remote Software Version Identification. In 34th USENIX Security Symposium (USENIX Security 25). USENIX Association, Seattle, WA, 2847–2866. https://www.usenix.org/conference/usenixsecurity25/ presentation/chen-jinsong

[4] Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research (Dec. 2024). https://openreview.net/forum?id= cSimKw5p6R

[5] Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J. White, and Su-In Lee. 2023. Learning to Maximize Mutual Information for Dynamic Feature Selection. In Proceedings of the 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 6424–6447. https: //proceedings.mlr.press/v202/covert23a.html

[6] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., 28091–28114. doi:10.52202/075280-1220

[7] Zakir Durumeric, Hudson Clark, Jef Cody, Elliot Cubit, Matt Ellison, Liz Izhikevich, and Ariana Mirian. 2025. Censys: A Map of Internet Hosts and Services. In Proceedings of the ACM SIGCOMM 2025 Conference (SIGCOMM ’25). Association for Computing Machinery, New York, NY, USA, 147–163. doi:10.1145/3718958.3754344

[8] Xuan Feng, Qiang Li, Haining Wang, and Limin Sun. 2018. Acquisitional Rulebased Engine for Discovering Internet-of-Things Devices. In 27th USENIXSecurity Symposium (USENIX Security 18). USENIX Association, Baltimore, MD, 327–341. https://www.usenix.org/conference/usenixsecurity18/presentation/feng

[9] Soham Gadgil, Ian Covert, and Su-In Lee. 2024. Estimating Conditional Mutual Information for Dynamic Feature Selection. In The Twelfth International Conference on Learning Representations. 34979–35010. https://proceedings.iclr. cc/paper\_files/paper/2024/hash/9682490bedc064aba8aac1ab3f703507-Abstract-Conference.html

[10] Osman Berke Guney, Ketan Suhaas Saichandran, Karim Elzokm, Ziming Zhang, and Vijaya B. Kolachalama. 2025. Active feature acquisition via explainabilitydriven ranking. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 267). PMLR, 20748– 20765. https://proceedings.mlr.press/v267/guney25a.html

[11] Jiaming Guo, Haoran Yang, Kuihao Yan, Jiekang Hu, Xiaoqi Jia, Haichao Du, and Qihang Zhou. 2026. FalconScope: Efective and Eficient Detection ofHidden Web Interfaces in IoT Devices. In Proceedings of the ACM Web Conference 2026 (WWW ’26). Association for Computing Machinery, New York, NY, USA, 3066–3077. doi:10.1145/3774904.3792431

[12] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In The Tenth International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9

[13] Hung-Tien Huang, Dzung Dinh, and Junier B. Oliva. 2025. Information Templates: A New Paradigm for Intelligent Active Feature Acquisition. Accepted at NeurIPS 2026. arXiv:2508.18380 [cs.AI] https://arxiv.org/abs/2508.18380

[14] Imene Kerboua, Sahar Omidi Shayegan, Xing Han Lù, Léo Boisvert, Megh Thakkar, Massimo Caccia, Jérémy Espinas, Alex Aussem, Véronique Eglin, and Alexandre Lacoste. 2026. FocusAgent: Simple Yet Efective Ways of Trimming the Large Context of Web Agents. Transactions on Machine Learning Research (Aug. 2026). https://openreview.net/forum?id=mINaJKSy7A

[15] Yuta Kobayashi, Zilin Jing, Jiayu Yao, Hongseok Namkoong, and Shalmali Joshi. 2026. Learning-To-Measure: In-Context Active Feature Acquisition. In Proceedings of the 43rd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 306). PMLR, 59720–59752. https: //proceedings.mlr.press/v306/kobayashi26a.html

[16] Brian Kondracki and Nick Nikiforakis. 2024. Smudged Fingerprints: Characterizing and Improving the Performance of Web Application Fingerprinting. In 33rd USENIXSecurity Symposium (USENIXSecurity 24). USENIX Association, Philadel phia, PA, 4625–4640. https://www.usenix.org/conference/usenixsecurity24/ presentation/kondracki

[17] Dongjun Lee, Juyong Lee, Kyuyoung Kim, Jihoon Tack, Jinwoo Shin, Yee Whye Teh, and Kimin Lee. 2025. Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents. In The Thirteenth International Conference on Learning Representations. 81039–81072. https://proceedings.iclr.cc/paper\_files/paper/2025/hash c995a913bc20ca39ce2231b8973be90a-Abstract-Conference.html

[18] Zhen Lei, Nian Xue, Zhen Li, Dan Yu, Xin Huang, and Yongle Chen. 2025. DeepFW: A DNN-Based Firmware Version Identification Framework for Online IoT Devices. In 2025 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID). IEEE, 233–247. doi:10.1109/RAID67961.2025.00066

[19] Junlong Li, Yiheng Xu, Lei Cui, and Furu Wei. 2022. MarkupLM: Pre-training of Text and Markup Language for Visually Rich Document Understanding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, 6078–6087. doi:10.18653/v1/2022.acl-long.420

[20] Qiang Li, Dawei Tan, Xin Ge, Haining Wang, Zhi Li, and Jiqiang Liu. 2022. Understanding Security Risks of Embedded Devices Through Fine-Grained Firmware Fingerprinting. IEEE Transactions on Dependable and Secure Computing 19, 6 (Nov. 2022), 4099–4112. doi:10.1109/TDSC.2021.3119970

[21] Yinheng Li and Justin Wagle. 2026. LLM-as-Jev: LLMs Are Already Jev-Style Decision Models—When and How to Fine-Tune Them. arXiv:2610.02076 [cs.CL] doi:10.48550/arXiv.2610.02076

[22] Yiming Lin, Madelon Hulsebos, Ruiying Ma, Shreya Shankar, Sepanta Zeighami, Aditya G. Parameswaran, and Eugene Wu. 2025. Querying Templatized Docu ment Collections with Large Language Models. In 2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE, 2422–2435. doi:10.1109/ICDE65448. 2025.00183

[23] Chunwei Liu, Matthew Russo, Michael J. Cafarella, Lei Cao, Peter Baile Chen, Zui Chen, Michael J. Franklin, Tim Kraska, Samuel Madden, Rana Shahout, and Gerardo Vitagliano. 2025. Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing. In 15th Annual Conference on Innovative Data Systems Research. Amsterdam, The Nether lands. https://www.vldb.org/cidrdb/2025/palimpzest-optimizing-ai-poweredanalytics-with-declarative-query-processing.html

[24] Hangtian Liu, Lei Zheng, Shuitao Gan, Chao Zhang, Zicong Gao, Hongqi Zhang, Yishun Zeng, Zhiyuan Jiang, and Jiahai Yang. 2025. EAGLEYE: Exposing Hidden Web Interfaces in IoT Devices via Routing Analysis. In Network and Distributed System Security (NDSS) Symposium 2025. Internet Society. doi:10.14722/ndss. 2025.240399

[25] Junpeng Liu, Tianyue Ou, Yifan Song, Yuxiao Qu, Wai Lam, Chenyan Xiong, Wenhu Chen, Graham Neubig, and Xiang Yue. 2025. Harnessing Webpage UIs for Text-Rich Visual Understanding. In The Thirteenth International Conference on Learning Representations. 17971–18005. https://proceedings.iclr. cc/paper\_files/paper/2025/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html

[26] Zhihuang Liu, Zhangdong Wang, Tongqing Zhou, Yonghao Tang, Yuchuan Luo, and Zhiping Cai. 2026. Risk-Aware Privacy Preservation for LLM Inference. IEEE Transactions on Information Forensics and Security 21 (2026), 2462–2477. doi:10.1109/TIFS.2026.3667458

[27] Tony Nasr, Sadegh Torabi, Elias Bou-Harb, Claude Fachkha, and Chadi Assi. 2023. ChargePrint: A Framework for Internet-Scale Discovery and Security Analysis of EV Charging Management Systems. In Network and Distributed System Security (NDSS) Symposium 2023. Internet Society, San Diego, CA, USA. doi:10.14722/ndss.2023.23084

[28] Alexander Luke Ian Norclife, Changhee Lee, Fergus Imrie, Mihaela van der Schaar, and Pietro Liò. 2025. Stochastic Encodings for Active Feature Acquisition. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 267). PMLR, 46784–46814. https://proceedings.mlr.press/v267/norclife25a.html

[29] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to Route LLMs from Preference Data. In The Thirteenth International Conference on Learning Representations. 34433–34448. https://proceedings.iclr.cc/paper\_files/paper/

2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html

[30] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos Guestrin, and Matei Zaharia. 2025. Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS. Proceedings of the VLDB Endowment 18, 11 (2025), 4171–4184. doi:10.14778/ 3749646.3749685

[31] Kangkang Qi, Dongyang Xie, Wenbo Li, Hao Zhang, Yuanyuan Zhu, Jefrey Xu Yu, and Kangfei Zhao. 2026. Sema: A High-Performance System for LLM-Based Semantic Query Processing. Proceedings ofthe VLDB Endowment 19, 11 (2026), 3231–3244. doi:10.14778/3836663.3836685

[32] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, Jie Tang, and Yuxiao Dong. 2025. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning. In The Thirteenth International Conference on Learning Representations. 79791–79821. https://proceedings.iclr.cc/paper\_files/paper/2025/hash c66e1fcc9691aae706250638f36f681b-Abstract-Conference.htm

[33] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. Qwen Blog. Retrieved October 24, 2026 from https://qwen.ai/blog?id=qwen3.5

[34] Matthew Russo, Chunwei Liu, Sivaprasad Sudhir, Gerardo Vitagliano, Michael Cafarella, Tim Kraska, and Samuel Madden. 2026. Abacus: A Cost-Based Opti mizer for Semantic Operator Systems. Proceedings ofthe VLDB Endowment 19, 5 (2026), 1060–1073. doi:10.14778/3796195.3796215

[35] Armin Sarabi, Tongxin Yin, and Mingyan Liu. 2023. An LLM-based Framework for Fingerprinting Internet-connected Devices. In Proceedings ofthe 2023 ACM on Internet Measurement Conference (IMC ’23). Association for Computing Machinery, New York, NY, USA, 478–484. doi:10.1145/3618257.3624845

[36] Takayuki Sasaki, Tomoya Inazawa, Youhei Yamaguchi, Simon Parkin, Michel van Eeten, Katsunari Yoshioka, and Tsutomu Matsumoto. 2025. Am I Infected? Lessons from Operating a Large-Scale IoT Security Diagnostic Service. In 34th USENIX Security Symposium (USENIX Security 25). USENIX Association, Seattle, WA, 1493–1510. https://www.usenix.org/conference/usenixsecurity25/ presentation/sasaki

[37] Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G. Parameswaran, and Eugene Wu. 2025. DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing. Proceedings of the VLDB Endowment 18, 9 (2025), 3035– 3048. doi:10.14778/3746405.3746426

[38] Hajin Shim, Sung Ju Hwang, and Eunho Yang. 2018. Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding. In Advances in Neural Information Processing Systems, Vol. 31. Curran Associates, Inc., 1368–1378. https://papers.nips.cc/paper\_files/paper/2018/hash e5841df2166dd424a57127423d276bbe-Abstract.html

[39] Qiangqiang Song, Shunchao Xu, Zekun Zhang, Nan Yu, Zhi Li, and Limin Sun. 2026. IoTFine: Structure-Aware Extraction of IoT Device Attributes to Support Device Fingerprinting. In 2026 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD). IEEE, 2556–2561. doi:10.1109/CSCWD68734.2026.11582796

[40] Aaron Steiner, Ralph Peeters, and Christian Bizer. 2026. MCP vs RAG vs NLWeb vs HTML: A Comparison of the Efectiveness and Eficiency of Diferent Agent Interfaces to the Web. In Proceedings ofthe ACM Web Conference 2026 (WWW ’26). Association for Computing Machinery, New York, NY, USA, 8493–8496. doi:10.1145/3774904.3792893

[41] Jiejun Tan, Zhicheng Dou, Wen Wang, Mang Wang, Weipeng Chen, and Ji-Rong Wen. 2025. HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems. In Proceedings of the ACM on Web Conference 2025. Association for Computing Machinery, New York, NY, USA, 1733–1746. doi:10.1145/3696410.3714546

[42] Zhenhao Tian, Yi He, Nuo Zhang, Qixiao Lin, Hetian Shi, Jianwei Zhuge, Jian Mao, and Deliang Chang. 2025. BLMProbe: Enhancing Internet-Connected Device Discovery by Automated Device Labeling and Label Migration. IEEE Transactions on Information Forensics and Security 20 (2025), 7227–7242. doi:10. 1109/TIFS.2025.3587211

[43] TypeSafe AI. 2026. System One. TypeSafe AI Documentation. Retrieved October 24, 2026 from https://docs.typesafe.ai/concepts/system-one

[44] Michael Valancius, Maxwell Lennon, and Junier Oliva. 2024. Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition. In Proceedings of the 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, 48957–48975. https://proceedings.mlr.press v235/valancius24a.html

[45] Shangfeng Wan, Qiang Li, Haining Wang, Hong Li, and Limin Sun. 2023. DevTag: A Benchmark for Fingerprinting IoT Devices. IEEE Internet ofThings Journal 10, 7 (2023), 6388–6399. doi:10.1109/JIOT.2022.3225580

[46] Qifan Wang, Yi Fang, Anirudh Ravula, Fuli Feng, Xiaojun Quan, and Dongfang Liu. 2022. WebFormer: The Web-page Transformer for Structure Information Extraction. In Proceedings of the ACM Web Conference 2022. Association for Computing Machinery, New York, NY, USA, 3124–3133. doi:10.1145/3485447. 3512032

[47] Ruimin Wang, Haitao Li, Jing Jing, Liehui Jiang, and Weiyu Dong. 2022. WYSI WYG: IoT Device Identification Based on WebUI Login Pages. Sensors 22, 13,

Article 4892 (June 2022), 17 pages. doi:10.3390/s22134892

[48] Lindsey Linxi Wei, Shreya Shankar, Sepanta Zeighami, Yeounoh Chung, Fatma Ozcan, and Aditya G. Parameswaran. 2026. Multi-Objective Agentic Rewrites for Unstructured Data Processing. Proceedings of the VLDB Endowment 19, 11 (2026), 3062–3075. doi:10.14778/3836663.3836673

[49] Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. 2025. WebAgent R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 7909–7928. doi:10.18653/v1/2025.emnlp-main.401

[50] Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhenglin Wang, Zhengwei Tao, Ding-Chu Zhang, Zekun Xi, Robert Tang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. WebDancer: Towards Autonomous Information Seeking Agency. In Advances in Neural Information Processing Systems, Vol. 38. Curran Associates, Inc., 120957–120985. doi:10.52202/ 085713-4035

[51] Zhaoteng Yan, Zhi Li, Hong Li, Shouguo Yang, Hongsong Zhu, and Limin Sun. 2023. Internet-Scale Fingerprinting the Reusing and Rebranding IoT Devices in the Cyberspace. IEEE Transactions on Dependable and Secure Computing 20, 5 (Sept. 2023), 3890–3909. doi:10.1109/TDSC.2022.3223103

[52] Huipeng Yang, Li Yang, Lu Zhou, Lichuan Ma, Xinyue Wang, Junbo Jia, and Anyuan Sang. 2026. WebGeoInfer: Structure-Free Multi-Stage Framework for Geolocation Inference from Exposed Device Web Interfaces. In Proceedings ofthe ACM Web Conference 2026 (WWW’26). Association for Computing Machinery, New York, NY, USA, 2870–2880. doi:10.1145/3774904.3792341

[53] Ke Yang, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik Chaudhari, George Karypis, and Huzefa Rangwala. 2025. AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents. In The Thirteenth International Conference on Learning Representations. 97533–97565. https://proceedings.iclr.cc/paper\_files/paper/ 2025/hash/f2c6e459b95694a24ac69c469a4ee746-Abstract-Conference.html

[54] Sepanta Zeighami, Shreya Shankar, and Aditya Parameswaran. 2025. Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees. Proceedings of the ACM on Management of Data 3, 6, Article 311 (Dec. 2025), 26 pages. doi:10. 1145/3769776

[55] Fengshi Zhang, Zhi Li, Shunchao Xu, Zekun Zhang, Yongle Chen, Dongliang Fang, and Limin Sun. 2026. NS-FirmID: A Neuro-Symbolic Multi-Agent Framework for Reliable Firmware Version Identification at Internet Scale. In 2026 56th Annual IEEE International Conference on Dependable Systems and Networks (DSN). IEEE, 472–484. doi:10.1109/DSN69566.2026.00053

[56] Jiayuan Zhang, Kaiquan Chen, Zhihao Lu, Enshen Zhou, Qian Yu, and Jing Zhang. 2026. Prune4Web: DOM Tree Pruning Programming for Web Agent. Proceedings of the AAAI Conference on Artificial Intelligence 40, 41 (March 2026), 34710–34718. doi:10.1609/aaai.v40i41.40772

[57] Juhua Zhang, Zhiliang Tian, Minghang Zhu, Yiping Song, Taishu Sheng, Siyi Yang, Qiunan Du, Xinwang Liu, Minlie Huang, and Dongsheng Li. 2025. DYN-TEXT: Semantic-Aware Dynamic Text Sanitization for Privacy-Preserving LLM Inference. In Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria, 20243–20255. doi:10.18653/v1/2025.findings-acl.1038

[58] Zekun Zhang, Zhi Li, Shunchao Xu, Fengshi Zhang, Zhanwei Song, Dongliang Fang, Yongle Chen, and Limin Sun. 2026. NetID-GPT: Adapting Large Language Models for Large-Scale Internet-Connected Device Identification. Computer Networks 279 (April 2026), 112149. doi:10.1016/j.comnet.2026.112149

[59] Zhenyu Zhang, Bowen Yu, Tingwen Liu, Tianyun Liu, Yubin Wang, and Li Guo. 2023. Learning Structural Co-occurrences for Structured Web Data Extraction in Low-Resource Settings. In Proceedings ofthe ACM Web Conference 2023 (WWW ’23). Association for Computing Machinery, New York, NY, USA, 1683–1692. doi:10.1145/3543507.3583387

Table 4: (a) Semantic questions and relation templates. (b) Example request and DSEJEV output.  
(a) Semantic questions and relation templates
<table><tr><td>No.</td><td>Target object and context</td><td>Semantic question</td><td>Candidate answers</td></tr><tr><td>1</td><td>A form with credential inputs and login controls</td><td>What is the semantic function of the detected form element?</td><td>insufficient_evidence; collects_user_credentials; submits_configuration_data; displays_error_messages; handles_media_upload</td></tr><tr><td>2</td><td>A submission operation linked to the form</td><td>What is the primary operational purpose of the detected event handler?</td><td>insufficient_evidence; initiates_page_load; handles_ user_interaction; manages_window_dimensions; triggers_form_submission</td></tr><tr><td colspan="4">Pair 1–2: Operation targets → form.</td></tr><tr><td>3</td><td>An external script element and page language cues An external resource</td><td>What is the observable function role of the script identified by the src attribute? What is the observable function role of the</td><td>insufficient_evidence; dependency_loading; localization_execution; framework_initialization insufficient_evidence; styling_definition; favicon_</td></tr><tr><td>4</td><td>referenced by a script, with its reference context</td><td>resource identified by the path attribute?</td><td>representation; application_bundle; localization_ script</td></tr><tr><td colspan="4">Pair 3-4: Script</td></tr><tr><td>5</td><td>A response text span with HTTP status and device management login text</td><td>What is the scope of the content returned in the HTTP response?</td><td>insufficient_evidence;authentication_challenge; management_interface; error_message; redirect_ location</td></tr><tr><td>6</td><td>An adjacent text span with vendor, product model, and account prompts Pair 5-6: The next_text_span relation connects adjacent spans.</td><td>Classify the primary explicit subject of the indicated object&#x27;s visible content.</td><td>device itself;device administration interface;user or account; network service; help content; other; insufficient_evidence</td></tr></table>

(b) Example request and DSEJEV output  
Target operation: onchange event.  
Related control: select element with id and name set to LanguageSelect, class set to width120, and onchange set to g\_oLogin.changeFrameLanguage(this.value). Related script: g\_oLogin.changeFrameLanguage(this.value).

targets references Relations: Control → operation; operation → script.

Question: What is the primary operational intent of the event handler?

## A Ethical Considerations

We use the publicly released DevTag and BLMProbe datasets [42, 45] to evaluate vendor and device-type identification from recorded Web responses. The experiments involve no additional scans or interactions with live devices. We evaluate semantic questions on stored responses and analyze form and script relationships without executing the corresponding actions. We report aggregate identification and eficiency results. ValQID is intended to support authorized network asset inventory. Device identification can also aid reconnaissance, so operators should use ValQID only on devices and networks they are authorized to assess.

## B Semantic Questions and Relation Templates

Table 4(a) presents six questions from the final libraries, grouped by object roles and connections into three pairs. Each question is evaluated with its target object and local context. Panel (b) shows an example request and the option probabilities returned by DSEJEV.

<table><tr><td>Option</td><td>Candidate answer</td><td>Probability</td></tr><tr><td>A</td><td>insufficient_evidence</td><td>0.0005</td></tr><tr><td>B</td><td>navigation</td><td>0.0278</td></tr><tr><td>C</td><td>data_submission</td><td>0.0040</td></tr><tr><td>D</td><td>state_change</td><td>0.9626</td></tr><tr><td>E</td><td>authentication</td><td>0.0050</td></tr><tr><td>F</td><td>error_handling</td><td>0.0001</td></tr></table>

The control name and handler identify a language-switching operation, and DSEJEV assigns the highest probability to state\_ change. ValQID attaches the full option distribution to the target object for identity prediction and subsequent computation selection. Example 2 in Appendix C describes the corresponding identification process.

## C Representative Identification Examples

Example 1: Two rounds of semantic evaluation. The response contains zyxel vsg1432-b101, web configurator, and username and password prompts. ValQID first selects the question “What is the primary operational role of the target resource?” with a predicted value of 0.1170; DSEJEV assigns probability 0.6323 to web\_authentication\_gateway.

After updating the identification state, ValQID selects “What specific function does the target perform regarding network connectivity?” with a predicted value of 0.1147; DSEJEV assigns probability

0.999993 to insufficient\_evidence. With both option distributions added to the identification state, ValQID combines this state with response features and label-pair information to predict zyxel / router, matching the reference labels.

Example 2: A contextual semantic query. The response contains Server: NVR Webserver, username and password controls, and the language-selection control and event handler shown in Table 4(b). ValQID selects a question about the intent of the onchange handler, with a predicted value of 0.4968. Using the associated control and script, DSEJEV classifies the operation as state\_change with probability 0.9626.

After the option distribution updates the relational state, the highest predicted value among the remaining candidate groups is −0.1996, so ValQID stops querying. The final predictor combines this state with response features and label-pair information to output Hanbanggaoke / DVR, matching the reference labels.

## D Data and Implementation Details

## D.1 Data Splits and Evaluation

Data splits. For DevTag, we start from the released training, development, and test partitions and group duplicate records by exact response text. Text groups spanning multiple partitions are assigned to one partition, with priority given to test, development, and training, in that order. Groups with conflicting labels are excluded. After selecting HTTP responses, we retain the vendor and device-type classes present in all three partitions to form closed-set identification tasks.

BLMProbe is partitioned by source host, with target proportions of 60%, 20%, and 20% for training, development, and testing. Source groups connected by identical responses are merged and assigned to one partition under the same priority rule. The number of responses in each partition depends on the merged group sizes.

Training data use. ValQID further divides each training set into two disjoint subsets: one for training the relation-based identity model and discovering semantic conditions, and the other for fit ting the question-group value model. These subsets contain 47,140 and 11,784 responses for DevTag, and 5,257 and 1,315 responses for BLMProbe. Responses sharing a source group or identical full text remain in the same subset. We train the value model using identification gains computed with the identity model frozen. The final joint character classifier uses all training responses with valid vendor–type labels: 58,924 for DevTag and 6,407 for BLMProbe. The DevTag nearest-neighbor component uses the 47,140 responses assigned to identity-model training.

Evaluation. All methods in Table 2 are evaluated on the same test responses within each dataset, using a common label vocabulary and scoring procedure. We evaluate vendor, device-type, and joint identification on all 8,490 DevTag test responses. For BLM-Probe, device-type evaluation uses all 2,370 test responses; vendor and joint evaluation use the 2,347 responses with valid labels for both tasks. Joint accuracy requires both labels to be correct for the same response. Vendor and device-type macro-averaged metrics are computed over the fixed class vocabulary of each dataset.

## D.2 Model Training and Inference

Model training. DSEJEV is adapted from JevK5 4B using LoRA, with Qwen3.5-397B-A17B providing supervision for local semantic questions. We set the LoRA rank, scaling factor, and dropout to 16, 32, and 0.05, respectively, and train for two epochs with a learning rate of $3 \times 1 0 ^ { - 5 } .$ . Each parameter update uses 128 questions, with a maximum input length of 2,048 tokens per request.

The relation-based identity model uses a batch size of 256, with learning rates of $5 \times 1 0 ^ { - 5 }$ for DevTag and $2 \times 1 0 ^ { - 4 }$ for BLMProbe. The question-group value model is trained with AdamW, a learning rate of 10<sup>−4</sup>, a batch size of 512, and Huber loss with � = 1. Model checkpoints and fusion parameters are selected on the development set.

Inference environment. End-to-end eficiency experiments run on a single NVIDIA H100 80GB GPU. DSEJEV uses BF16 precision with the LoRA weights merged into the base model. The system processes 64 responses per batch. Each semantic batch contains at most 64 requests and 16,384 tokens after padding. Four CPU worker processes prepare the inputs.