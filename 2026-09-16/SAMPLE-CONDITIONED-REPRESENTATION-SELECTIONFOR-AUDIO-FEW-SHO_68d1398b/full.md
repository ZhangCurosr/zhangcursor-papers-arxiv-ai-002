# SAMPLE-CONDITIONED REPRESENTATION SELECTIONFOR AUDIO FEW-SHOT LEARNING

Fengrui Liu<sup>1,2,∗,‡,§</sup>, Ningxin Shen<sup>3,∗</sup>, Yi Li<sup>1</sup>, Yiwei Fu<sup>2</sup>, Feng Liu<sup>4</sup>, Senior Member, IEEE, Jiangmeng Li<sup>1</sup>, †

<sup>1</sup>National Key Laboratory of Space Integrated Information System, Institute of Software, Chinese Academy of Sciences

<sup>2</sup>School of Computer Science and Technology, East China Normal University

<sup>3</sup>School of Computer Science, Nanjing University

<sup>4</sup>School of Psychology, Shanghai Jiao Tong University

<sup>∗</sup>Equal contribution. <sup>‡</sup>Project lead. <sup>†</sup>Corresponding author. <sup>§</sup>Work done while the first author was an intern at the Institute of Software, Chinese Academy of Sciences.

## ABSTRACT

Few-shot audio classifiers may rely on foreground–background cooccurrences and fail when those correlations shift. On SpurAudio, the resulting representation shift is concentrated and class dependent: for ResNet12, the top 10% of channels explain 82.80% of the nullcorrected shift contribution. We propose SAMPLESELECT, which predicts a fixed-budget feature mask independently for each input while keeping the encoder and source classifier frozen. Training uses differentiable Gumbel Top-k selection with foreground classification and cross-background contrastive losses; inference uses deterministic Top-k masks and support-only linear adaptation. Across ResNet12 and Conv64 in 5-way 1-shot and 5-shot evaluation, SAM-PLESELECT gives the best OOD accuracy among the compared methods and improves the matched full-representation control by 4.90–8.38 percentage points. Ablations and representation analyses further support the learned selection mechanism.Codes available at https://github.com/Cross-Innovation-Lab/SAMPLESELECT/

Index Terms— Few-shot audio classification, background shift, representation selection, contrastive learning

## 1. INTRODUCTION

Few-shot audio classification recognizes novel sound classes from limited labeled examples, commonly through episodic adaptation or source-trained representations [1–5]. Real recordings also contain recurring foreground–background co-occurrences that models can exploit as shortcuts [6]. When those associations change, previously useful features can become unreliable, while the small support set offers little evidence for identifying which coordinates remain trustworthy.

MetaAudio studies transfer across acoustic domains [7], whereas SpurAudio changes foreground–background co-occurrences between IID and OOD episodes [8]. Existing representation adaptation and feature reweighting methods improve few-shot transfer [9–13], but do not directly target input-varying background sensitivity. This motivates selecting features for each example rather than always exposing the full representation or one global subset.

A frozen ResNet12 reveals that the shift is highly concentrated and class dependent: the top 10% of its 640 channels account for 82.80% of the corrected shift contribution, and the affected channels vary substantially across foreground classes. We therefore propose

![](images/429ead097fec5cb799d68d9567e54593e9b4d5631216418e5b282fd285e29486.jpg)  
Fig. 1. Overview of few-shot classification under background shift. FULLREP passes all representation coordinates to a fresh episodic linear head; SAMPLESELECT independently masks each support and query example before fitting and prediction. The class-space plots are schematic, not measured embeddings.

SAMPLESELECT, a fixed-budget, sample-conditioned selector. A lightweight scorer predicts feature importance from each input while the encoder and source classifier remain frozen. Training uses a differentiable Gumbel relaxation with foreground classification and grouped cross-background contrastive supervision [14–18]; inference uses deterministic Top-k masking and a support-only linear head. On SpurAudio, across ResNet12 and Conv64 with 5-way 1-shot and 5-shot episodes, SAMPLESELECT attains the highest OOD accuracy among the compared methods in all four settings and improves the matched FULLREP control by 4.90–8.38 percentage points. Ablations and further analyses support learned selection, cross-background supervision, and the functional relevance of the selected channels. Our contributions are: (1) evidence that background co-occurrence shift is concentrated and class dependent in frozen audio representations; (2) an inductive sample-conditioned fixed-budget selector trained without changing the source representation; and (3) consistent OOD gains across two backbones and two shot settings, supported by ablation and representation analyses.

![](images/82310255b30dd5711f0c48f7ae574a9cf834b2b46f1bd006254f81be4be2b2c5.jpg)  
Fig. 2. Three-stage SAMPLESELECT pipeline. The source encoder/head are trained then frozen; the selector learns sample-wise masks with classification and cross-background contrastive losses; inference uses deterministic Top-k masks and a fresh support-only episodic head.

## 2. METHODOLOGY

Figure 2 summarizes SAMPLESELECT. We first learn a source representation, then freeze it and train only a sample-conditioned selector; few-shot evaluation uses deterministic masks and a fresh support-only head.

## 2.1. Source Representation Learning

For input $x _ { i }$ with foreground label y<sub>i</sub>, the selectable descriptor $z _ { i }$ and classifier representation $h _ { i }$ are

$$
\begin{array} { r l } & { \mathrm { R e s N e t 1 2 : ~ } F _ { i } = f _ { \theta } ( x _ { i } ) \in { \mathbb R } ^ { 6 4 0 \times 4 \times 5 } , \quad z _ { i } = \mathrm { G A P } ( F _ { i } ) , } \\ & { \qquad h _ { i } = \mathrm { v e c } ( F _ { i } ) \in { \mathbb R } ^ { 1 2 8 0 0 } , } \\ & { \mathrm { C o n v 6 4 : ~ } z _ { i } = h _ { i } = f _ { \theta } ( x _ { i } ) \in { \mathbb R } ^ { 1 6 0 0 } . } \end{array}\tag{1}
$$

We train the encoder and source classifier $H _ { \psi }$ on source foreground classes,

$$
( \theta ^ { \star } , \psi ^ { \star } ) = \arg \operatorname* { m i n } _ { \theta , \psi } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { C E } ( H _ { \psi } ( h _ { i } ) , y _ { i } ) ,\tag{2}
$$

and freeze both thereafter.

## 2.2. Sample-Conditioned Selector Learning

A scorer predicts one score per selectable coordinate, with retention ratio r:

$$
\begin{array} { r l r } & { } & { s _ { i } = G _ { \phi } ( \mathrm { s t o p g r a d } ( z _ { i } ) ) \in \mathbb { R } ^ { D } , \qquad k = \lfloor r D \rfloor , } \\ & { } & { m _ { i } ^ { \mathrm { t r } } = \mathrm { R e l a x e d T o p K } _ { \tau } ( s _ { i } , k ) \in [ 0 , 1 ] ^ { D } , } \end{array}\tag{3}
$$

where $D \ = \ 6 4 0$ for ResNet12 and 1600 for Conv64. $G _ { \phi }$ acts independently on each input. During training, sequential Gumbel-Softmax draws without replacement provide a differentiable Top-k relaxation [14, 15]; exact cardinality is enforced at inference.

The selected classifier representation is

$$
\widetilde { h } _ { i } = \left\{ \begin{array} { l l } { \mathrm { v e c } ( m _ { i } \odot _ { c } F _ { i } ) , } & { \mathrm { R e s N e t } 1 2 , } \\ { m _ { i } \odot h _ { i } , } & { \mathrm { C o n v } 6 4 , } \end{array} \right.\tag{4}
$$

where $\odot _ { c }$ broadcasts a channel mask over spatial locations. Zero masking preserves coordinate alignment across examples.

The frozen source head preserves foreground information through

$$
\mathcal { L } _ { \mathrm { c l s } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \mathrm { C E } \Big ( H _ { \psi ^ { \star } } ( \widetilde { h } _ { i } ) , y _ { i } \Big ) .\tag{5}
$$

For background group $b _ { i } ,$ define $P ( i ) = \{ p \neq i : y _ { p } = y _ { i } , b _ { p } \neq$ $b _ { i } \}$ . Let $q _ { i } \ = \ \mathrm { G A P } ( m _ { i } \odot _ { c } F _ { i } )$ for ResNet12 and $q _ { i } ~ = ~ \bar { h } _ { i }$ for Conv64, with normalized similarity $u _ { i j } = { ( q _ { i } / \lVert q _ { i } \rVert _ { 2 } ) } ^ { \top } ( q _ { j } / \lVert q _ { j } \rVert _ { 2 } )$ For a valid anchor,

$$
\mathcal { L } _ { \mathrm { c o n } } ^ { ( i ) } = - \frac { 1 } { | P ( i ) | } \sum _ { p \in P ( i ) } \log \frac { \exp ( u _ { i p } / T ) } { \sum _ { a \in G ( i ) \backslash \{ i \} } \exp ( u _ { i a } / T ) } .\tag{6}
$$

where $G ( i )$ is its comparison group. We average over anchors with valid positives and minimize $\mathcal { L } _ { \mathrm { s e l } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } }$ , updating only $\phi .$ Thus classification preserves category information, while contrastive supervision favors same-class consistency across observed backgrounds. Background labels are used only in selector training.

## 2.3. Inductive Few-Shot Evaluation

At inference, each example receives an independent deterministic mask

$$
m _ { i , c } ^ { \mathrm { e v a l } } = \mathbf { 1 } [ c \in \mathrm { T o p K } ( s _ { i } , k ) ] , \qquad \| m _ { i } ^ { \mathrm { e v a l } } \| _ { 0 } = k .\tag{7}
$$

For a 5-way K-shot episode $\mathcal { E } = ( \mathcal { S } , \mathcal { Q } )$ , a fresh linear head is fitted only on masked support representations and then applied to each masked query:

$$
\begin{array} { l } { { \displaystyle W _ { \varepsilon } ^ { \star } = \arg \operatorname* { m i n } _ { W } \frac { 1 } { | S | } \sum _ { ( x _ { j } , y _ { j } ) \in S } \mathrm { C E } \Big ( W \widetilde { h } ( x _ { j } ) , y _ { j } \Big ) , } } \\ { { \displaystyle \widehat { y } _ { q } = \arg \operatorname* { m a x } _ { c } \left[ W _ { \varepsilon } ^ { \star } \widetilde { h } ( x _ { q } ) \right] _ { c } . } } \end{array}\tag{8}
$$

Table 1. Results on SpurAudio with ResNet12 and Conv64. Accuracy values are percentages. $\Delta = \mathrm { I I D - O O D }$ . FULLREP denotes the matched no-selection control. Bold indicates the best IID or OOD accuracy among the compared methods
<table><tr><td colspan="7">ResNet12</td><td colspan="6">Conv64</td></tr><tr><td></td><td colspan="3">1-shot</td><td colspan="3">5-shot</td><td colspan="3">1-shot</td><td colspan="3">5-shot</td></tr><tr><td>Method</td><td>IID</td><td>OOD</td><td>∆</td><td>IID</td><td>OOD</td><td>∆</td><td>IID</td><td>OOD</td><td>∆</td><td>IID</td><td>OOD</td><td>∆</td></tr><tr><td>Baseline++ (2019)</td><td>57.694</td><td>53.078</td><td>4.616</td><td>75.032</td><td>65.490</td><td>9.542</td><td>50.705</td><td>46.667</td><td>4.038</td><td>65.286</td><td>56.934</td><td>8.352</td></tr><tr><td>R2D2 (2019)</td><td>57.995</td><td>54.228</td><td>3.767</td><td>74.760</td><td>66.149</td><td>8.611</td><td>41.093</td><td>39.420</td><td>1.673</td><td>68.695</td><td>61.873</td><td>6.822</td></tr><tr><td>ANIL (2020)</td><td>54.129</td><td>50.893</td><td>3.236</td><td>66.015</td><td>58.352</td><td>7.663</td><td>49.522</td><td>46.499</td><td>3.023</td><td>64.763</td><td>56.009</td><td>8.754</td></tr><tr><td>BDCSN (2022)</td><td>61.241</td><td>57.961</td><td>3.280</td><td>73.564</td><td>66.691</td><td>6.873</td><td>44.992</td><td>42.374</td><td>2.618</td><td>58.951</td><td>55.031</td><td>3.920</td></tr><tr><td>PADDLE (2022)</td><td>52.196</td><td>46.842</td><td>5.354</td><td>70.380</td><td>59.299</td><td>11.081</td><td>50.016</td><td>44.267</td><td>5.749</td><td>66.767</td><td>55.912</td><td>10.855</td></tr><tr><td>Proto-LP (2023)</td><td>59.852</td><td>56.036</td><td>3.816</td><td>74.762</td><td>67.454</td><td>7.308</td><td>57.054</td><td>53.446</td><td>3.608</td><td>69.916</td><td>61.328</td><td>8.588</td></tr><tr><td>BPA (2024)</td><td>60.325</td><td>55.828</td><td>4.497</td><td>75.964</td><td>63.516</td><td>12.448</td><td>49.158</td><td>45.369</td><td>3.789</td><td>66.173</td><td>50.924</td><td>15.249</td></tr><tr><td>ECPE (2026)</td><td>61.397</td><td>56.391</td><td>5.006</td><td>74.890</td><td>65.642</td><td>9.248</td><td>55.836</td><td>51.974</td><td>3.862</td><td>69.038</td><td>59.536</td><td>9.502</td></tr><tr><td>FULLREP</td><td>57.169</td><td>51.997</td><td>5.172</td><td>74.369</td><td>63.259</td><td>11.110</td><td>51.470</td><td>47.032</td><td>4.438</td><td>67.641</td><td>56.813</td><td>10.828</td></tr><tr><td>SAMPLESELECT</td><td>62.307</td><td>57.979</td><td>4.327</td><td>77.523</td><td>68.161</td><td>9.362</td><td>57.357</td><td>53.635</td><td>3.721</td><td>72.993</td><td>65.188</td><td>7.805</td></tr></table>

No query label, query-set statistic, or test-time background label is used for masking or adaptation. The matched FULLREP control uses the same support-only procedure with $m _ { i } = { \bf 1 }$

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Datasets. We evaluate on SpurAudio [8], which contains 25 training, 5 validation, and 8 test foreground classes and controls foreground– background co-occurrence to construct IID and OOD few-shot episodes. We follow the standard 5-way 1-shot and 5-shot protocols with 10 query examples per class and report the mean over 1,000 episodes. We report IID accuracy, OOD accuracy, and the shift gap $\Delta _ { \mathrm { s h i f t } } = \mathrm { A c c } _ { \mathrm { I I D } } - \mathrm { A c c } _ { \mathrm { O O D } }$ . OOD accuracy is the primary metric. Backbones and Baselines. We evaluate two representation architectures. ResNet12 performs channel-level selection over 640 channels, while Conv64 performs coordinate-level selection over a 1,600- dimensional projected representation. We compare with representative methods reported under the SpurAudio protocol and include FULLREP as a matched control using the same frozen backbone and episodic classifier without representation selection, thereby isolating the effect of selection.

Implementation Details. Inputs are standardized $1 \times 1 2 8 \times 1 5 7$ log-Mel spectrograms. The source encoder is trained for 30 epochs and then frozen. ResNet12 uses retention ratio r = 0.7 with k = 448, while Conv64 uses $r = 0 . 8$ with $k = 1 2 8 0$ . Unless otherwise stated, selector training uses Gumbel temperature $\tau = 0 . 3$ , contrastive temperature $T = 0 . 0 7$ , and $\lambda _ { \mathrm { c o n } } = 0 . 0 2$ . During few-shot evaluation, the encoder and selector remain frozen and only a support-only linear classifier is optimized for each episode.

## 3.2. Main Results

Table 1 shows that SAMPLESELECT achieves the highest OOD accuracy among the compared methods in all four backbone and shot settings. It reaches 57.979% and 68.161% with ResNet12, and 53.635% and 65.188% with Conv64 for 1-shot and 5-shot evaluation, respectively. IID accuracy is also highest in each setting. Compared with the matched FULLREP control, SAMPLESELECT improves OOD accuracy by 5.982 and 4.902 percentage points for ResNet12, and by 6.603 and 8.375 points for Conv64. The gains are consistent across both representation architectures and shot settings, showing that sample-conditioned selection improves the use of frozen representations under class–background co-occurrence shift.

## 3.3. Ablation and Sensitivity Analysis

Figure 3(a) evaluates the main components of SAMPLESELECT. Replacing learned selection with Random Soft decreases OOD accuracy by 2.882 and 1.254 points in the 1-shot and 5-shot settings. Removing the Gumbel relaxation reduces accuracy by 2.458 and 2.322 points, while removing the cross-background contrastive objective gives drops of 1.477 and 0.968 points. A learnable global selector achieves 55.799±1.372 and 66.869±1.004 OOD accuracy in the 1- shot and 5-shot settings, respectively, trailing SAMPLESELECT by 2.180 and 1.292 points. These results support learned, differentiable, sample-conditioned selection and cross-background supervision.

Figures 3(b,c) study the two main hyperparameters. Performance changes only modestly for $r \in [ 0 . 4 , 0 . 9 ]$ , and values near the default r = 0.7 give similar OOD gains. For $\lambda _ { \mathrm { c o n } } .$ , every tested positive value improves the OOD mean over $\lambda _ { \mathrm { c o n } } = 0$ in both shot settings. We use $\lambda _ { \mathrm { c o n } } = 0 . 0 2$ as a shared setting rather than tuning it separately for each evaluation condition.

## 3.4. Further Analysis

The previous experiments establish the performance benefit of SAM-PLESELECT. We next examine the representation behavior that motivates sample-conditioned selection.

## 3.4.1. Concentrated and Class-Dependent Shift

For foreground class y and ResNet12 channel c, we measure the IID-to-OOD distribution change using the 1-Wasserstein distance and correct it with a within-class permutation null:

$$
\begin{array} { r l } & { d _ { y , c } = W _ { 1 } \Big ( \widehat { P } _ { \mathrm { I I D } } ( z _ { c } \mid y ) , \widehat { P } _ { \mathrm { O O D } } ( z _ { c } \mid y ) \Big ) , } \\ & { g _ { y , c } = d _ { y , c } - \mathbb { E } _ { \boldsymbol \pi } [ d _ { y , c } ^ { \boldsymbol { \pi } } ] , \qquad \omega _ { y , c } = \frac { [ g _ { y , c } ] _ { + } ^ { 2 } } { \sum _ { c ^ { \prime } } [ g _ { y , c ^ { \prime } } ] _ { + } ^ { 2 } + \epsilon } . } \end{array}\tag{9}
$$

As shown in Fig. 4(a), the top 10%, 20%, and 30% of channels account for 82.80%, 92.85%, and 96.76% of the positive corrected shift mass, with a mean Gini coefficient of 0.887. The number of significantly shifted channels also varies strongly across foreground classes. For example, sneezing and pig show broader affected subsets than blender and crackling fire. This class dependence supports input-dependent rather than globally fixed selection. This analysis is diagnostic rather than a causal identification result: IID and OOD routes contain different foreground recordings, and concentration refers specifically to the null-corrected statistic above. Neither testset channel rankings nor this statistic supervise the selector; training uses source-class labels and observed background variation only.

![](images/756a2271490fdd7a3dfe5de3f4a1e448dd17ed9e45a1e1025c5002b28371c344.jpg)

![](images/ef4c1141cebe21d02220044579561ef2e19329e3b818a44998816386015d3029.jpg)

![](images/3d1265eae3b285ffb3a3a1e319863fddc82ccf2fe5f4d5d8d274820143316ecf.jpg)

Fig. 3. Ablation and sensitivity analysis on ResNet12. (a) OOD accuracy drop after removing individual components of SAMPLESELECT. (b) OOD gain over FULLREP for different retention ratios r. (c) OOD gain over FULLREP for different contrastive weights $\lambda _ { \mathrm { { c o n } } }$ . Dashed lines indicate the default settings used in the main experiments.  
![](images/7dc5e271201b0647916c9fb24c84d323d7cb1fc51191da76a14b0d39a73b57a7.jpg)  
(a) Shift concentration

![](images/cf8bc5a1815e288e402447ef8a87c96eb08cec4f41f7593f3de60a22d054950c.jpg)  
(b) Shift and OOD degradation

![](images/ca7dcc73d91d0bfc78cea8f9c414a5a88691a66114440a8b7a5a85e62a806e78.jpg)  
(c) Functional blocking  
Fig. 4. Further analysis on ResNet12. (a) Cumulative contribution of channels with the largest null-corrected representation shifts. (b) Mean Spearman correlation across three seeds between episode-level shift mass and IID-to-OOD degradation for accuracy, confidence, and classification margin. (c) Accuracy drop after blocking the highest-scored, random, or lowest-scored 30% of channels while keeping the episodic classifier fixed.

## 3.4.2. Shift and OOD Degradation

We correlate episode-level representation shift with IID-to-OOD predictive degradation over 1,000 episodes. Figure 4(b) summarizes the mean correlations across three random seeds for accuracy, predicted probability, and classification margin. Across the three seeds, two shot settings, and three measures, all 18 Spearman correlations are positive, with ρ between 0.268 and 0.455. Episodes with larger representation shift therefore tend to exhibit larger predictive degradation, establishing an association between the representation diagnosis and OOD performance.

## 3.4.3. Selector Scores Reflect Functional Importance

We rank the 640 ResNet12 channels by their selector scores and block the highest-scored, random, or lowest-scored 30% while keeping the episodic classifier fixed. Figure 4(c) shows that blocking the highestscored channels decreases accuracy by 3.033 percentage points on average, compared with 1.064 points for random blocking, while removing the lowest-scored channels produces almost no change. The learned scores therefore reflect clear differences in the functional contribution of representation channels.

Together, these analyses connect the empirical motivation in Sec. 1 with the behavior of the learned selector: background-induced changes are concentrated and class-dependent, larger representation shifts are associated with larger OOD degradation, and the selector assigns higher scores to channels with greater functional importance.

## 4. CONCLUSION

We presented SAMPLESELECT, a sample-conditioned fixed-budget representation selector for few-shot audio classification under class– background co-occurrence shift. The method keeps the source encoder frozen, learns per-example feature scores using source-class discrimination and cross-background contrastive supervision, and performs deterministic Top-k selection at inference. Masks are perexample and support-only; no query label/statistic or test-time background label is used. On SpurAudio, SAMPLESELECT achieves the best OOD accuracy in all four backbone/shot settings and improves the matched FULLREP control by 5.982, 4.902, 6.603, and 8.375 percentage points. Ablation and further analysis show that the gains depend on learned selection and cross-background supervision and are consistent with a representation structure in which backgroundinduced changes are concentrated, class-dependent, and associated with predictive degradation.

## 5. REFERENCES

[1] O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra, “Matching networks for one shot learning,” in Advances in Neural Information Processing Systems, vol. 29, 2016.

[2] J. Snell, K. Swersky, and R. S. Zemel, “Prototypical networks for few-shot learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[3] W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. F. Wang, and J.-B. Huang, “A closer look at few-shot classification,” in International Conference on Learning Representations, 2019.

[4] F. Liu, R. Huang, Q. Zheng, Y. Wang, and F. Liu, “From physics to representation: Audio learning with synthetic pre-training via procedural generation,” in Proceedings of the 2026 International Conference on Multimedia Retrieval, 2026, pp. 595–604.

[5] F. Liu, R. Huang, Q. Zheng, Y. Wang, and F. Liu, “PRP: Procedural-to-real masked pre-training for transferable and interpretable audio representations,” IEEE Transactions on Audio, Speech and Language Processing, 2026.

[6] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, pp. 665– 673, 2020.

[7] C. Heggan, S. Budgett, T. Hospedales, and M. Yaghoobi, “MetaAudio: A few-shot audio classification benchmark,” arXiv:2204.02121, 2022.

[8] G. Abu Ayoub, M. Tukan, and L. Mualem, “SpurAudio: A benchmark for studying shortcut learning in few-shot audio classification,” arXiv:2605.13672, 2026.

[9] H. Li, D. Eigen, S. Dodge, M. Zeiler, and X. Wang, “Finding task-relevant features for few-shot learning by category traversal,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 1–10.

[10] N. Dvornik, C. Schmid, and J. Mairal, “Selecting relevant features from a multi-domain representation for few-shot classification,” in Computer Vision – ECCV 2020, 2020, pp. 769–786.

[11] W.-H. Li, X. Liu, and H. Bilen, “Cross-domain few-shot learning with task-specific adapters,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 7161–7170.

[12] S. Lee, W. Moon, and J.-P. Heo, “Task discrepancy maximization for fine-grained few-shot classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5331–5340.

[13] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141.

[14] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with Gumbel-Softmax,” in International Conference on Learning Representations, 2017.

[15] W. Kool, H. Van Hoof, and M. Welling, “Stochastic beams and where to find them: The Gumbel-Top-k trick for sampling sequences without replacement,” in Proceedings ofthe 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 3499–3508.

[16] M. F. Balın, A. Abid, and J. Zou, “Concrete autoencoders: Differentiable feature selection and reconstruction,” in Proceedings ofthe 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 444–453.

[17] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[18] C. Sgouropoulos, C. Nikou, S. Vlachos, V. Theiou, C. Foukanelis, and T. Giannakopoulos, “Prototypical contrastive learning for improved few-shot audio classification,” in 2025 IEEE 35th International Workshop on Machine Learning for Signal Processing, 2025.