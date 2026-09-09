# TRAINING-FREE TASK VECTORS FOR LLM BEHAVIORAL CONTROL

Gabriel J. Perin University of Sao Paulo˜ gabrieljp@usp.br

Andre Araujo´   
Google DeepMind   
andrearaujo@google.com   
Lucas Boscaini   
Google   
lucasboscaini@google.com

Nina S. T. Hirata University of Sao Paulo˜ nina@ime.usp.br

## ABSTRACT

Task vectors enable post-training model editing by identifying semantically meaningful directions in weight space, typically computed as the difference between a fine-tuned model and its pretrained initialization. However, this reliance on fine-tuning makes discovering such directions costly and limits the practicality of post-training model editing. To address this limitation, we introduce Training-Free Task Vectors (TFTVs), a novel method to compute task-vector-like directions without requiring fine-tuning. Our method maps activation steering vectors to rank-one weight-space edits using only forward-pass statistics, while satisfying arithmetic properties that directly support learning via addition, forgetting via subtraction, and the composition of multiple edits. Empirically, we evaluate TFTVs on large language model behavioral control tasks and show that they consistently amplify, suppress, and compose target behaviors while preserving general knowledge and problem-solving skills. We also validate our method against other editing and steering baselines, experimentally demonstrating that TFTVs achieve stronger trait control with better or competitive utility preservation. We hope our work opens new directions for the community in post-training model editing and broader training-free model control. Code is available on the project website: tftv-llm.github.io.

## 1 INTRODUCTION

Large language models (LLMs) have demonstrated remarkable capabilities on complex tasks such as reasoning (Achiam et al., 2023), code generation (Chen et al., 2021; Li et al., 2022), and instruction following (Wei et al., 2021). As these models become increasingly capable and widely deployed, post-training control has become a central problem: practitioners may need to suppress undesirable behaviors, amplify desired traits, or impose multiple behavioral constraints after a model has already been trained (Turner et al., 2023; Li et al., 2023; Rimsky et al., 2024; Chen et al., 2025). Ideally, such edits should not interfere with inference dynamics or model architecture, allowing users to modify behavior while preserving the standard forward pass expected by optimized deployment and fine-tuning pipelines (Sun et al., 2026).

A prominent approach to post-training model editing is to operate directly in parameter space through task vectors (Ilharco et al., 2022). Given a model fine-tuned from a shared pretrained initialization, a task vector is defined as the difference between the fine-tuned and pretrained weights, and has been shown to encode semantically meaningful directions that transfer across related models and tasks (Ilharco et al., 2022; Yadav et al., 2023; Yu et al., 2024). Remarkably, these direction exhibit a simple arithmetic structure: they can be added to induce new behaviors, negated to remove or suppress behaviors, and composed to combine multiple edits (Ilharco et al., 2022; Bhardwaj et al., 2024; Sun et al., 2025; Fierro & Roger, 2025). This makes task vectors an appealing primitive for efficient post-training editing.

![](images/21ed1555ae6307e2e5c2ed485be848ba1c3ea6724de99f620db73f433efb2276.jpg)  
Figure 1: Training-Free Task Vectors enable arithmetic model editing without any training. Unlike classical task vectors ∆θ<sup>′</sup> (Ilharco et al., 2022) (left), which are computed from the difference between fine-tuned and base model weights, TFTVs ∆θ (right) are derived directly from forward-pass statistics, using contrastive prompts. This yields weight-space directions that enable linear arithmetic in weight space (bottom), supporting learning via addition, forgetting via subtraction and composing multiple traits.

However, this arithmetic structure comes at a cost: task vectors are not discovered directly from the pretrained model, but retrospectively from a completed fine-tuning run. Thus, before one can edit a model with a task vector, one must already possess a second checkpoint that expresses the desired behavior (Ilharco et al., 2022). This requirement is especially limiting for behavioral control: each new trait requires obtaining a corresponding fine-tuned checkpoint.

To address this limitation, we introduce training-free task vectors (TFTVs), a simple method that identifies semantically meaningful behavioral directions in weight space, requiring only forwardpass statistics. Our approach is motivated by the observation that steering vectors already encode directional behavioral information, but only in activation space (Li et al., 2023; Turner et al., 2023; Chen et al., 2025; Rimsky et al., 2024). We show that, by appropriately mapping these directions to the parameter space, one can obtain edits that are both training-free and compositional. An overview of TFTV is given in Figure 1.

Empirically, we evaluate TFTVs on behavioral control tasks, focusing on traits such as evil, hallucination, and sycophancy. We show that TFTVs exhibit three key properties that allow model editing: learning via addition, where adding the update amplifies the target behavior; forgetting via subtraction, where subtracting the update suppresses target behavior; and composing multiple traits, where multiple directions can be combined to jointly control several traits. Across these settings, TFTVs achieve substantially stronger behavioral control while matching or improving the utility preservation of recent inference-time steering and model-editing techniques.

## In summary, our contributions are as follows:

• We introduce Training-Free Task Vectors (TFTVs), a simple method for identifying semantically meaningful behavioral weight-space directions, without auxiliary fine-tuning or additional optimization. TFTVs convert activation steering directions into rank-one weight updates using only forward-pass statistics, enabling persistent post-training model editing without any training.

• We show that the proposed construction satisfies algebraic properties that directly support learning via addition, forgetting via subtraction, and composition of multiple traits.

• We empirically demonstrate that TFTVs enable strong and compositional behavioral control with better or competitive utility preservation than recent inference steering and model editing methods. Across the main paper and appendix, our evaluation spans four recent instruction-tuned models, five target traits, including both single-trait and composed edit settings.

## 2 RELATED WORK

Model Merging. Model merging combines multiple models into a single parameter set that preserves or improves source capabilities (Yang et al., 2026). Early work showed that averaging fine-tuned checkpoints can improve robustness and generalization without additional inference cost (Wortsman et al., 2022; Rame et al., 2023; Matena & Raffel, 2022); in LLMs, merging often aims´ to integrate abilities from multiple task-specific fine-tunings (Lee et al., 2025b; Perin et al., 2024; Lee et al., 2025a; Yadav et al., 2023; Yu et al., 2024; Jin et al., 2022). Most closely related to our work are task vectors, which identify semantic weight-space directions by subtracting pretrained parameters from fine-tuned ones (Ilharco et al., 2022). These directions support arithmetic operations such as addition, negation, and composition. Some work has also applied task vectors to LLM behavioral control (Bhardwaj et al., 2024; Sun et al., 2025; Fierro & Roger, 2025). Also, low-rank variants (Lee et al., 2025b) and subspace decomposition (Damirchi et al., 2025) can further improve merging. However, task vectors require an auxiliary fine-tuned checkpoint, whereas our method constructs semantically meaningful weight-space edits without auxiliary fine-tuning, broadening the applicability of post-training weight-space control.

Steering Vectors. Activation steering controls LLM behavior by identifying directions in representation space and injecting them during the forward pass (Li et al., 2023; Turner et al., 2023; Rimsky et al., 2024). Such directions can be obtained using contrastive mean-difference methods (Turner et al., 2023; Rimsky et al., 2024) or sparse autoencoders (Cunningham et al., 2023), and can be composed to induce combined behavior (Pai et al., 2026). Closest to our setting, Persona Vectors identify directions associated with behavioral traits such as evil, hallucination, and sycophancy, and use them to monitor and control model behavior (Chen et al., 2025). However, activation steering remains transient and requires forward-pass interventions (Sun et al., 2026). Our work improves upon this line of research by proposing training-free control that produces persistent weight-space edits rather than transient activation-space interventions. Empirically, we show that this approach provides stronger behavioral control than activation steering while preserving competitive, and often superior, general utility.

Post-training Model Editing. A separate line of work induces lasting changes in language models through direct weight-space edits. KnowledgeEditor (De Cao et al., 2021) and MEND (Mitchell et al., 2021) learn auxiliary editors for targeted parameter updates, while ROME (Meng et al., 2022a) and MEMIT (Meng et al., 2022b) apply structured updates to edit factual associations. These methods show that weight updates can induce persistent changes, but they either require additional training or focus primarily on factual knowledge editing. Closest to our setting, Steer2Edit (Sun et al., 2026) maps steering vectors into rank-one weight updates by editing components aligned with a steering direction. In contrast, TFTV uses the steering direction as the left factor of a rank-one update and constructs the right factor from an SVD-weighted vector aligned with the module’s expected input. This yields explicit norm-matching and expected-input steering properties, and makes composition of steering directions correspond to addition of weight-space deltas. We further compare against Steer2Edit and find stronger trait control with better utility preservation.

## 3 TRAINING-FREE TASK VECTORS

Task vectors are traditionally defined as the difference between a fine-tuned model and its pretrained initialization (Ilharco et al., 2022). Such vectors have been shown to support operations such as learning via addition, forgetting via negation, and task analogies, making them useful for posttraining model editing.

Our goal is to recover the same kind of editable weight-space directions without requiring an auxiliary fine-tuned model from which they can be extracted. To formalize this objective, we introduce the notion of a training-free task vector.

Let $f _ { \theta }$ be a Transformer-based language model with parameters $\theta \in \mathbb { R } ^ { w }$ . For a target trait $T \left( \mathrm { e . g . } \right.$ • , good or evil), a training-free task vector is a direction in weight space, denoted by $\bar { \Delta } \theta _ { T } \in \mathbb { R } ^ { u }$ , that can be obtained without fine-tuning and is intended to satisfy the following properties.

Learning via addition. The edited model $f _ { \theta + \Delta \theta _ { T } }$ exhibits an increased manifestation of trait T.

Forgetting via subtraction. The edited model $f _ { \theta - \Delta \theta _ { T } }$ exhibits a decreased manifestation of trait T.

Composing multiple traits. Given two training-free task vectors $\Delta \theta _ { T }$ and $\Delta \theta _ { T ^ { \prime } }$ , the model $f _ { { \boldsymbol { \theta } } + \Delta { \boldsymbol { \theta } } _ { T } + \Delta { \boldsymbol { \theta } } _ { T } }$ exhibits increased manifestation of both traits $T$ and $T ^ { \prime }$

These properties together allow model editing without training. Our central hypothesis is that these directions can be obtained by mapping steering vectors into weight-space updates. We therefore begin by defining the steering vectors used in our construction.

## 3.1 PRELIMINARIES: STEERING VECTORS

Let d denote the hidden dimensionality of the residual stream of $f _ { \theta }$ . A steering vector at layer or module ℓ is a vector $s _ { \ell } \in \mathbb { R } ^ { d }$ that is added to the corresponding hidden representation during the forward pass in order to steer the model toward a desired trait.

There are several ways to construct steering vectors. In this work, we use a contrastive meanactivation approach (Sun et al., 2026; Chen et al., 2025; Turner et al., 2023; Rimsky et al., 2024). Let $\chi _ { + }$ and ${ \bf { \bar { \mathcal { X } } } } _ { - }$ denote prompt sets designed to induce and suppress the target trait, respectively. For each prompt $x \in \mathcal { X } _ { + } \cup \mathcal { X } _ { - }$ , the model generates a completion y. Examples inherit their positive or negative label from the prompt set, while an LLM judge is used to filter out incoherent completions or responses inconsistent with the intended behavior (Chen et al., 2025). After filtering, the remaining prompt-completion pairs define $\mathcal { D } _ { + }$ and $\mathcal { D } _ { - }$

For each pair $( x , y )$ , we run the model on the concatenated sequence $x \oplus y ,$ where ⊕ denotes concatenation, and extract the hidden representations at layer or module $\ell .$ Let $h _ { \ell } ^ { ( t ) } ( x \oplus y ) \in \mathbb { R } ^ { d }$ denote the representation at token position $t ,$ and let $L ( z )$ be the length of sequence z. We average these representations first over completion tokens and then over examples in each group:

$$
s _ { \ell } ^ { \tau } = \frac { 1 } { | { \mathcal D } _ { \tau } | } \sum _ { ( x , y ) \in { \mathcal D } _ { \tau } } \left( \frac { 1 } { L ( y ) } \sum _ { t = L ( x ) + 1 } ^ { L ( x \oplus y ) } h _ { \ell } ^ { ( t ) } ( x \oplus y ) \right) , \qquad \tau \in \{ + , - \} .\tag{1}
$$

The steering vector is then defined as $s _ { \ell } = s _ { \ell } ^ { + } - s _ { \ell } ^ { - }$

## 3.2 BUILDING TFTVS FROM STEERING VECTORS

We now introduce our method for mapping steering vectors into updates in parameter space. We will also state key properties of this construction that motivate its use as a training-free task vector; proofs are deferred to Appendix A.

Intuitively, a steering vector specifies the desired displacement in residual space, but not the parameter change that should produce it. We therefore construct a weight update that induces this displacement directly, turning an activation-space intervention into a persistent parameter-space edit.

Let $W _ { \ell } \in \mathbb { R } ^ { d \times l }$ denote the weight matrix of a module ℓ whose output lies in the transformer residual space $\mathbb { R } ^ { d } \left( \mathrm { e . g . } \right.$ , an attention output projection), where l is the dimension of the module input. Let $s _ { \ell } \in \mathbb { R } ^ { d }$ be the steering vector associated with this module, and let $\mu _ { \ell } \in \mathbb { R } ^ { l }$ denote its expected input, which in practice can be estimated from the same data used to construct the steering vector.

First, we compute the singular value decomposition of $W _ { \ell } ,$ given by $\begin{array} { r } { W _ { \ell } = \sum _ { i = 1 } ^ { r } \sigma _ { i } u _ { i } v _ { i } ^ { \top } } \end{array}$ , where $r = { \mathrm { r a n k } } ( W _ { \ell } ) $ . We then normalize the steering vector as $\bar { s } _ { \ell } : = s _ { \ell } / \| s _ { \ell } \| _ { 2 }$ and define the training-free task vector update for module ℓ by

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) = \bar { s } _ { \ell } \left( \sum _ { i = 1 } ^ { r } \mathrm { s i g n } ( \mu _ { \ell } ^ { \top } v _ { i } ) \sigma _ { i } v _ { i } \right) ^ { \top } .\tag{2}
$$

This update constructs a weight-space direction whose output aligns with the steering vector $\bar { s } _ { \ell } ,$ while the singular vectors $v _ { i }$ and singular values $\sigma _ { i }$ capture the principal input directions of the module. The sign term ensures that the update acts consistently with the expected input $\mu _ { \ell }$ . As a result, on average, inputs are pushed toward the desired steering direction.

Finally, given a predefined set of modules $\mathcal { T }$ and a scalar coefficient $\alpha \in \mathbb { R }$ , we update each selected weight matrix according to

$$
W _ { \ell } \gets W _ { \ell } + \alpha \mathrm { ~ T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) , \qquad \forall \ell \in \mathcal { T } .\tag{3}
$$

![](images/35d602e00b094340c151eea31d0fa4fcbce0cb132873f7824bebb8fc163ee2f1.jpg)  
Figure 2: Overview of TFTV computation. Contrastive prompts elicit $( \mathcal { X } _ { + } )$ or suppress $( \mathcal { X } _ { - } )$ a target trait, while an LLM judge retains only completions that satisfy trait-manifestation and coherence criteria. From these completions, we compute positive $( s _ { \ell } ^ { + } )$ ) and negative $( s _ { \ell } ^ { - } )$ activation means, taking their difference to obtain a steering vector $s _ { \ell } .$ , which is normalized to $\bar { s } _ { \ell } .$ . Finally, $\bar { s } _ { \ell }$ is combined with the expected module input $\mu _ { \ell }$ and the SVD of the weight matrix $W _ { \ell }$ to construct the TFTV update.

This yields module updates that are rank-one, making the resulting edits compact and beneficial for weight-space combination (Lee et al., 2025b). Figure 2 provides an overview of our method.

## 3.3 PROPERTIES OF TFTV

The following arithmetic properties help justify the proposed design.

Property 1 (Norm matching). Let $\| \cdot \| _ { F }$ denote the Frobenius norm. Then,

$$
\lVert W _ { \ell } \rVert _ { F } = \lVert \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \rVert _ { F } .\tag{4}
$$

Thus, the update has the same Frobenius norm as the original weight matrix, and the scalar coefficient α directly controls the magnitude of the applied change. This is desirable because different layers can operate at different parameter and activation scales; tying the update norm to the norm of the underlying weight matrix makes the construction naturally adapt to these layer-specific scales.

Property 2 (Steering). We have

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \mu _ { \ell } = c \bar { s } _ { \ell } , \qquad \mathrm { f o r ~ s o m e } ~ c \in \mathbb { R } _ { \ge 0 } .\tag{5}
$$

That is, when applied to the expected input $\mu _ { \ell }$ for a given trait, the induced weight change moves the module output in the positive steering direction. This ensures that, on average, the edit promotes the target trait rather than inadvertently steering the model in the opposite direction.

Property 3 (Linearity). Let $\beta , \gamma \in \mathbb { R }$ , and let $\bar { s } _ { \ell } ^ { ( 1 ) }$ and $\bar { s } _ { \ell } ^ { ( 2 ) }$ denote two different normalized steering vectors. Then,

$$
\beta \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 1 ) } , \mu _ { \ell } ) + \gamma \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } ) = \mathrm { T F T V } \left( W _ { \ell } , \beta \bar { s } _ { \ell } ^ { ( 1 ) } + \gamma \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } \right) .\tag{6}
$$

Thus, linear arithmetic in TFTVs corresponds directly to linear arithmetic in steering vectors, supporting learning via addition, forgetting via subtraction, and the composition of multiple traits.<sup>1</sup> Taken together, these properties show that the proposed construction is simple, fully training-free, and arithmetically aligned with the task-vector behavior we aim to recover.

Table 1: TFTV increases trait manifestation without damaging utility. We report trait scores, MMLU scores, and GSM8K scores. Higher values are better. Base model scores are provided for reference. Best results among training-free editing methods are in bold. The last two columns refer to methods that require fine-tuning.
<table><tr><td>Base model</td><td>Trait</td><td>Metric</td><td>Base</td><td>Steering</td><td>Steer2Edit</td><td>TFTV</td><td>Task Vectors</td><td>CWS</td></tr><tr><td rowspan="8">Llama 3.1</td><td rowspan="2">Evil</td><td>Trait</td><td>0.00</td><td>14.06</td><td>26.69</td><td>63.26</td><td>95.62</td><td>90.67</td></tr><tr><td>MMLU</td><td>68.26</td><td>67.29</td><td>63.48</td><td>68.27</td><td>65.21</td><td>67.75</td></tr><tr><td rowspan="2"></td><td>GSM8K</td><td>77.10</td><td>75.82</td><td>74.91</td><td>76.42</td><td>59.97</td><td>74.45</td></tr><tr><td>Trait</td><td>17.03</td><td>61.51</td><td>89.25</td><td>98.55</td><td>94.54</td><td>99.09</td></tr><tr><td rowspan="3">Hallucinating</td><td>MMLU</td><td>68.26</td><td>67.41</td><td>63.99</td><td>68.11</td><td>65.75</td><td>63.26</td></tr><tr><td>GSM8K</td><td>77.10</td><td>73.09</td><td>76.27</td><td>73.39</td><td>76.42</td><td>36.09</td></tr><tr><td>Trait</td><td>3.60</td><td>63.86</td><td>81.44</td><td>94.64</td><td>89.55</td><td>87.13</td></tr><tr><td rowspan="2">Sycophantic</td><td>MMLU</td><td>68.26</td><td>67.29</td><td>63.40</td><td>68.21</td><td>65.88</td><td>67.87</td></tr><tr><td>GSM8K</td><td>77.10</td><td>75.44</td><td>73.77</td><td>74.53</td><td>75.97</td><td>77.41</td></tr><tr><td rowspan="8">Qwen 2.5</td><td rowspan="2">Evil</td><td>Trait</td><td>0.00</td><td>8.58</td><td>5.04</td><td>61.96</td><td>74.45</td><td>65.64</td></tr><tr><td>MMLU</td><td>71.83</td><td>71.83</td><td>63.59</td><td>71.78</td><td>71.76</td><td>71.76</td></tr><tr><td rowspan="2"></td><td>GSM8K</td><td>78.54</td><td>78.32</td><td>53.53</td><td>79.45</td><td>74.30</td><td>76.72</td></tr><tr><td>Trait</td><td>11.46</td><td>88.96</td><td>94.23</td><td>99.80</td><td>73.98</td><td>99.96</td></tr><tr><td rowspan="3">Hallucinating</td><td>MMLU</td><td>71.83</td><td>71.93</td><td>63.48</td><td>70.70</td><td>71.45</td><td>71.35</td></tr><tr><td>GSM8K</td><td>78.54</td><td>71.57</td><td>76.57</td><td>74.15</td><td>79.53</td><td>63.76</td></tr><tr><td>Trait</td><td>4.35</td><td>90.46</td><td>50.39</td><td>89.78</td><td>60.13</td><td>91.02</td></tr><tr><td rowspan="3">Sycophantic</td><td>MMLU</td><td>71.83</td><td>71.60</td><td>65.39</td><td>71.81</td><td>71.61</td><td>71.74</td></tr><tr><td>GSM8K</td><td>78.54</td><td>75.74</td><td>72.02</td><td>77.48</td><td>78.17</td><td>63.84</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4 MAIN EXPERIMENTS

We conduct experiments to evaluate whether our method identifies weight-space directions that exhibit task vector properties—learning via addition, forgetting via subtraction, and composing multiple traits—while preserving general model utility. Standard deviations for all results in this section are presented in Appendix D.

Experimental setup. We evaluate TFTVs on the Persona Vectors benchmark (Chen et al., 2025) using Llama-3.1-8B-Instruct (Grattafiori et al., 2024) and Qwen-2.5-7B-Instruct (Yang et al., 2024; Team, 2024). We consider three target traits: evil, hallucination, and sycophancy. Following Persona Vectors, we report LLM-judge scores for trait expression and coherence, and use zero-shot MMLU (Hendrycks et al., 2020) and GSM8k (Cobbe et al., 2021) accuracy as additional utility metrics. Full details are provided in Appendix B. Unless stated otherwise, we apply TFTV on the attention output projection modules. Ablations for that are presented in Appendix F.4.

## 4.1 LEARNING VIA ADDITION

Our first goal is to evaluate whether TFTVs can amplify the manifestation of a target trait while preserving general utility more effectively than competing techniques. We compare TFTV against Persona Vector inference-time steering (Chen et al., 2025), Steer2Edit (Sun et al., 2026), Task Vectors (Ilharco et al., 2022) and Contrastive Weight Steering (CWS) (Fierro & Roger, 2025). For each trait, we evaluate edited models on the Persona Vectors questions and report the resulting trait–utility trade-off. All method-specific tuning details and final configurations are given in Appendix B. Additional results with more traits (humorous and optimistic) and two more model families — gemma-4-E2B-it (Team et al., 2026) and Ministral-3-14B-Instruct (Liu et al., 2026) — are presented in Appendix F.2 and F.3, respectively.

Results. For each method, we report the edit with the highest trait score subject to a coherence score of at least 70; coherence results appear in Table 13, in Appendix E. Table 1 shows that TFTV provides the strongest behavior–utility trade-off among training-free methods. Across the six settings, it exceeds the strongest training-free baseline in trait score in five by 5.57–53.38 points; steering is only 0.68 points higher for Qwen sycophancy. TFTV keeps Llama MMLU within 0.15 points of the base model and Llama GSM8K within 3.71 points across all settings, whereas Steer2Edit reduces Llama MMLU by 4.27–4.86 points. Compared with fine-tuned methods (Task Vectors and CWS), TFTV achieves generally comparable trait scores but preserve utility better (e.g., achieving higher MMLU in five out of six settings).

Table 2: TFTV mitigates trait expression while preserving general utility. We report trait scores, MMLU scores, and GSM8K scores. Lower S indicates stronger suppression, while higher MMLU and GSM8K indicate better utility preservation. Base model scores are provided for reference. Best results among training-free editing methods are in bold. The last two columns refer to methods that require fine-tuning.
<table><tr><td rowspan="8">Base model</td><td rowspan="2">Trait</td><td>Metric</td><td>Base</td><td>Steering</td><td>Steer2Edit</td><td>TFTV</td><td>Task Vectors</td><td>CWS</td></tr><tr><td></td><td>95.42</td><td></td><td>0.13</td><td>0.49</td><td></td><td></td></tr><tr><td rowspan="2">Evil</td><td>Trait MMLU</td><td>68.26</td><td>58.67 68.70</td><td>63.63</td><td>68.34</td><td>8.59 67.64</td><td>1.64</td></tr><tr><td>GSM8K</td><td>77.10</td><td>74.91</td><td>77.10</td><td>76.72</td><td>75.89</td><td>68.39 77.18</td></tr><tr><td rowspan="3">Hallucinating</td><td>Trait</td><td>97.53</td><td>79.13</td><td></td><td>1.85</td><td></td><td></td></tr><tr><td>MMLU</td><td>68.26</td><td>67.21</td><td>5.31 64.35</td><td>68.47</td><td>26.02 67.36</td><td>3.26</td></tr><tr><td>GSM8K</td><td>77.10</td><td>77.26</td><td>75.36</td><td>78.54</td><td>75.59</td><td>64.21 0.30</td></tr><tr><td rowspan="3">Sycophantic</td><td>Trait</td><td>92.06</td><td>53.65</td><td>5.31</td><td>14.62</td><td>32.29</td><td>24.91</td></tr><tr><td>MMLU</td><td>68.26</td><td>68.06</td><td>62.18</td><td>68.42</td><td>66.96</td><td>68.01</td></tr><tr><td>GSM8K</td><td>77.10</td><td>76.27</td><td>72.78</td><td>77.03</td><td>76.72</td><td>77.63</td></tr><tr><td rowspan="8">Qwen 2.5</td><td rowspan="3">Evil</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Trait MMLU</td><td>69.85</td><td>12.77 71.59</td><td>0.00</td><td>0.17 71.54</td><td>24.46 71.68</td><td>0.00 70.94</td></tr><tr><td>GSM8K</td><td>71.83 78.54</td><td>78.32</td><td>69.93 75.66</td><td>77.56</td><td>79.68</td><td>15.69</td></tr><tr><td rowspan="3">Hallucinating</td><td>Trait</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMLU</td><td>79.94 71.83</td><td>38.61 71.02</td><td>34.31 24.11</td><td>0.14 69.73</td><td>33.66 71.88</td><td>0.00 71.44</td></tr><tr><td>GSM8K</td><td>78.54</td><td>74.30</td><td>0.80</td><td>72.63</td><td>76.72</td><td>74.60</td></tr><tr><td rowspan="3">Sycophantic</td><td>Trait</td><td>64.72</td><td>10.88</td><td>3.69</td><td>3.14</td><td>10.48</td><td>5.79</td></tr><tr><td>MMLU</td><td>71.83</td><td>71.48</td><td>69.41</td><td>71.68</td><td>71.73</td><td>71.41</td></tr><tr><td>GSM8K</td><td>78.54</td><td>75.59</td><td>72.71</td><td>78.70</td><td>79.76</td><td>80.43</td></tr></table>

## 4.2 FORGETTING VIA SUBTRACTION

We next test whether discovered directions can suppress target traits while preserving utility. For all methods in Table 1, we reverse the selected configurations by multiplying the update or steering coefficient by −1, and evaluate them under trait-eliciting prompts following Persona Vectors (Chen et al., 2025). We report the resulting trait scores, MMLU and GSM8K accuracy; full details are provided in Appendix B.

Results. Table 2 reports the results; coherence scores appear in Table 14, in Appendix E. Across the six settings, TFTV reduces target trait scores by 7.74–77.28 points relative to steering. It keeps MMLU slightly above the base model in all Llama settings and GSM8K within 0.98 points of the base in five of six settings, with only Qwen hallucination showing a larger drop of 5.91 points. Compared with Steer2Edit, TFTV achieves stronger suppression in three out of six settings, higher MMLU in all six, and higher GSM8K in five. Against the fine-tuned baselines (Task Vectors and CWS), TFTV achieves the strongest suppression in four out of six settings; CWS is stronger only for Qwen evil and hallucination by at most 0.17 points, but can reduce GSM8K to as low as 0.30. Overall, TFTV provides the best suppression–utility trade-off among training-free methods while remaining competitive with fine-tuned baselines.

## 4.3 COMPOSING MULTIPLE TRAITS

We also evaluate whether TFTVs support composition. For each pair of traits, as well as the combination of all three traits, we compose TFTV edits by summing the corresponding weight deltas. For the other methods, we analogously sum the corresponding steering vectors or weight updates.

Table 3: TFTV compositions preserve, or improve upon, single-trait edit effects. We compare composed TFTV edits against their corresponding single-trait edits. Lower trait scores indicate stronger suppression; higher MMLU and GSM8K scores indicate better utility. Parentheses report the difference from the best corresponding single-trait edit in the composition; green indicates improvement, red degradation, and gray no change. Here, E, H, and S denote evil, hallucinating, and sycophantic, respectively.
<table><tr><td>Base model Edit</td><td></td><td>Evil ↓</td><td>Hall. ↓</td><td> $\operatorname { S y c . \downarrow }$ </td><td>MMLU↑</td><td>GSM8K ↑</td></tr><tr><td rowspan="7">Llama 3.1</td><td>Base</td><td>95.42</td><td>97.53</td><td>92.06</td><td>68.26</td><td>77.10</td></tr><tr><td>E</td><td>0.49</td><td>93.79</td><td>71.32</td><td>68.34</td><td>76.72</td></tr><tr><td>H</td><td>26.22</td><td>1.85</td><td>50.57</td><td>68.47</td><td>78.54</td></tr><tr><td>S</td><td>63.29</td><td>89.27</td><td>14.62</td><td>68.42</td><td>77.03</td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td>0.14 (−0.35)</td><td>1.51 (−0.34)</td><td>一</td><td>68.48 (+0.01)</td><td>77.71 (−0.83)</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>0.69 (+0.20)</td><td></td><td> $1 3 . 1 0 \ ( - 1 . 5 2 )$ </td><td>68.42 (+0.00)</td><td>76.65 (−0.38)</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>1.87 (+0.02)</td><td> $2 . 2 5 \ ( - 1 2 . 3 7 )$ </td><td>68.41 (-0.06)</td><td>80.36 (+1.82)</td></tr><tr><td rowspan="8">Qwen 2.5</td><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>0.03 (−0.46)</td><td>1.54 (−0.31)</td><td>1.94 (−12.68)</td><td>68.36 (-0.11)</td><td>79.15 (+0.61)</td></tr><tr><td>Base</td><td>69.85</td><td>79.94</td><td>64.72</td><td>71.83</td><td>78.54</td></tr><tr><td>E</td><td>0.17</td><td>69.79</td><td>37.46</td><td>71.54</td><td>77.56</td></tr><tr><td>H</td><td>0.02</td><td>0.14</td><td>6.61</td><td>69.73</td><td>72.63</td></tr><tr><td>S</td><td>42.70</td><td>57.76</td><td>3.14</td><td>71.68</td><td>78.70</td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td>0.00 (−0.02)</td><td>0.39 (+0.25)</td><td></td><td>69.64 (-1.90)</td><td>72.55 (−5.01)</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>0.00 (−0.17)</td><td></td><td> $2 . 5 8 \ ( - 0 . 5 6 )$ </td><td>71.47 (−0.21)</td><td>78.92 (+0.22)</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$   $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td></td><td> $0 . 7 2 \ ( + 0 . 5 8 )$  0.00 (−0.02) 0.81 (+0.67)</td><td> $0 . 1 3 \ ( - 3 . 0 1 )$   $0 . 1 3 \ ( - 3 . 0 1 )$ </td><td>69.69 (−1.99) 69.73 (-1.95)</td><td>72.10 (−6.60) 69.60 (-9.10)</td></tr></table>

We use the suppression settings from Table 2 and prompt the model to elicit the target traits, thereby measuring whether the composed edits jointly suppress the corresponding behaviors.

Results. Table 3 shows that TFTV compositions largely preserve or improve the suppression achieved by individual edits; coherence scores are reported in Appendix E. On Llama 3.1, the threeway composition improves sycophancy suppression by 12.68 points, while GSM8K decreases by at most 0.83 points and improves by up to 1.82 points. Compared with steering in Table 4, TFTV achieves stronger suppression on all nine measured scores; for the three-way edit, it reduces hallucination by 54.68 additional points while improving MMLU by 1.32 and GSM8K by 24.11 points. Compared with Steer2Edit, TFTV achieves stronger suppression on seven out of nine scores and higher MMLU and GSM8K in all four compositions. Against the fine-tuned baselines (Task Vectors and CWS), it achieves the strongest suppression on five out of nine scores, the highest MMLU on all four compositions, and the highest GSM8K on three, while CWS reduces GSM8K to as low as 0.98. Qwen 2.5 results are discussed in the Appendix. Overall, TFTV effectively composes multiple edits while providing a strong suppression–utility trade-off. Figure 3 provides qualitative examples.

## 5 ADDITIONAL ANALYSIS

Next, we perform experiments to understand and justify how different design choices affect our method. The experiments are run with Llama-3.1-8B-Instruct. Comprehensive ablations for module and layer analysis are also presented in Appendices F.4 and F.5, respectively.

## 5.1 OUT-OF-DISTRIBUTION EVALUATION

TFTV is conditioned on an expected input representation $\mu _ { \ell }$ estimated from the same distribution used to construct its steering vectors $s _ { \ell }$ and perform the main evaluation. To test OOD robustness, we evaluate evil (E) edits on Moral Stories (Emelin et al., 2021), which tests moral judgment; hallucination (H) edits on TruthfulQA (Lin et al., 2021), which tests resistance to common misconceptions; and sycophancy (S) edits on tasks covering user opinions in NLP, philosophy, and politics (Perez et al., 2023).

Table 5: TFTV remains robust OOD. Bold indicates the expected change from base: addition reduces morality and truthfulness and increases sycophancy, while negation reverses these effects. MC1 measures top-1 accuracy; MC2 measures normalized probability mass on truthful answers.
<table><tr><td>Task</td><td>Addition</td><td>Base</td><td>Negation</td></tr><tr><td>Moral Stories (E)</td><td>45.70</td><td>50.06</td><td>55.48</td></tr><tr><td>TruthfulQA MC1 (H)</td><td>35.01</td><td>37.58</td><td>39.17</td></tr><tr><td>TruthfulQA MC2 (H)</td><td></td><td>52.04 54.49</td><td>56.30</td></tr><tr><td>Sycophancy NLP (S)</td><td></td><td>97.8895.25</td><td>92.09</td></tr><tr><td>Sycophancy Phil. (S)</td><td></td><td>94.4091.16</td><td>88.24</td></tr><tr><td>Sycophancy Pol. (S)</td><td>82.67</td><td>83.48</td><td>75.31</td></tr></table>

Table 4: TFTV outperforms training-free methods under composed edits. We report trait scores for composed negation edits on Llama 3.1; lower is better. Pairwise compositions are evaluated only on their target traits, with unmeasured entries marked as -. MMLU and GSM8K measure general utility, with higher values being better. Here, E, H, and S denote evil, hallucinating, and sycophantic, respectively. Best reported values among training-free methods are in bold. Methods after the double horizontal line require fine-tuning and are excluded from the bold comparison. Results on Qwen 2.5 are reported in Table 7.
<table><tr><td>Method</td><td>Composition</td><td>Evil ↓</td><td>Hall. ↓</td><td>Syc. ↓</td><td>MMLU↑</td><td>GSM8K ↑</td></tr><tr><td rowspan="4">Steering</td><td>E + H</td><td>34.04</td><td>74.14</td><td></td><td>67.27</td><td>72.71</td></tr><tr><td>E + S</td><td>31.78</td><td></td><td>46.19</td><td>67.91</td><td>71.65</td></tr><tr><td>S+H</td><td></td><td>60.62</td><td>35.74</td><td>67.40</td><td>70.05</td></tr><tr><td>E + H + S</td><td>17.05</td><td>56.22</td><td>28.34</td><td>67.04</td><td>55.04</td></tr><tr><td rowspan="4">Steer2Edit</td><td>E+ H</td><td>0.26</td><td>2.39</td><td></td><td>57.59</td><td>67.78</td></tr><tr><td>E + S</td><td>0.60</td><td></td><td>5.76</td><td>53.77</td><td>63.91</td></tr><tr><td>S+H</td><td></td><td>6.41</td><td>4.07</td><td>56.57</td><td>65.20</td></tr><tr><td>E + H + S</td><td>0.51</td><td>12.46</td><td>2.11</td><td>39.32</td><td>32.30</td></tr><tr><td rowspan="4">TFTV</td><td>E + H</td><td>0.14</td><td>1.51</td><td></td><td>68.48</td><td>77.71</td></tr><tr><td>E + S</td><td>0.69</td><td></td><td>13.10</td><td>68.42</td><td>76.65</td></tr><tr><td>S+H</td><td></td><td>1.87</td><td>2.25</td><td>68.41</td><td>80.36</td></tr><tr><td>E + H + S</td><td>0.03</td><td>1.54</td><td>1.94</td><td>68.36</td><td>79.15</td></tr><tr><td rowspan="4">Task Vectors †</td><td>E + H</td><td>0.02</td><td>27.74</td><td></td><td>64.57</td><td>64.06</td></tr><tr><td>E + S</td><td>0.20</td><td></td><td>7.43</td><td>64.19</td><td>67.02</td></tr><tr><td>S+ H</td><td></td><td>24.79</td><td>6.70</td><td>63.99</td><td>71.72</td></tr><tr><td>E + H + S</td><td>0.25</td><td>44.71</td><td>6.16</td><td>59.62</td><td>32.90</td></tr><tr><td rowspan="4">CWS †</td><td>E + H</td><td>0.00</td><td>3.17</td><td></td><td>63.87</td><td>1.52</td></tr><tr><td>E + S</td><td>0.00</td><td></td><td>13.98</td><td>67.55</td><td>76.88</td></tr><tr><td>S+H</td><td></td><td>2.27</td><td>4.03</td><td>62.27</td><td>0.98</td></tr><tr><td>E + H + S</td><td>0.00</td><td>5.55</td><td>5.50</td><td>61.39</td><td>1.52</td></tr></table>

![](images/08ee212a03c44078f964f4f0a2ad1537e5909fbe8cf3b5604b384424a8cfe899.jpg)  
Figure 3: Qualitative examples: composed trait suppression by TFTV. The first row shows traiteliciting questions posed to each model variant, where a trait-eliciting system prompt is prepended to induce those traits. Subsequent rows show responses under progressively composed TFTV edits: no edit (base model), evil suppression (E), evil + hallucination suppression (E+H), and evil + hallucination + sycophancy suppression (E+H+S).

Results are presented in Table 5. Relative to the base model, the edits move 11 of the 12 scores in the expected direction: addition reduces morality and truthfulness while generally increasing sycophancy, whereas negation produces the opposite effects. The only exception is a 0.81-point decrease for addition on political sycophancy. Overall, TFTV’s effects largely transfer to OOD tasks. Additional results for Qwen are presented in Appendix F.1.

## 5.2 TFTV VERSUS INFERENCE STEERING UNDER MATCHED CONTROLS

Although TFTV constructs a weight update designed to induce an activation-steering effect, it consistently outperforms inference-time steering. To investigate this gap, we control for layer, coefficient, and intervention site: both methods are applied at layer 16 over a coefficient sweep, while inference steering is evaluated at both the transformer-block and attentionmodule (i. e. where TFTV is applied) outputs. Figure 4 shows

![](images/b2cfb834b5986a590093ea62a0576859b4e848acf40cb0082ab33d7f5d813098.jpg)  
Figure 4: TFTV yields a stronger trait–utility trade-off than inference steering under matched layer and coefficient sweeps.

the resulting trade-off between trait manifestation and MMLU accuracy. TFTV maintains a more favorable trade-off across these controls, suggesting that its advantage arises from encoding the intervention in the model weights rather than from layer, module, or coefficient selection alone.

## 6 CONCLUSIONS, LIMITATIONS, AND MISUSE

Conclusions. We introduced Training-Free Task Vectors (TFTVs), which map activation steering directions into rank-one weight-space edits using only forward-pass statistics. TFTVs enable persistent, compositional edits that support addition, subtraction, and multi-trait composition, achieving strong behavioral control while largely preserving utility.

Limitations. TFTV performance depends on the edited modules and layers, making automatic selection an important direction for future work. Moreover, TFTVs do not eliminate the trade-off between trait control and utility, especially under stronger or composed edits.

Misuse. Because TFTVs can suppress or amplify traits, they raise dual-use risks, especially as persistent weight-space edits. Even benign edits may degrade safety alignment, as observed for fine-tuning (Qi et al., 2023) and inference steering (Korznikov et al., 2025), motivating safeguards, auditing, controlled deployment, and further study.

## ACKNOWLEDGMENT

This research was supported by the Sao Paulo Research Foundation (FAPESP) [grant #2022/15304-˜ 4 and fellowship #2025/24851-7 to G. J. Perin], the Ministry of Science, Technology, and Innovation (MCTI/Brazil) [grant PPI-Softex TIC 13 DOU 01245.010222/2022-44, Law 8.248], and the National Council for Scientific and Technological Development (CNPq/Brazil) [PQ grant #307701/2025-5 to N. Hirata].

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Mart´ın Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned llms. arXiv preprint arXiv:2502.17424, 2025.

Rishabh Bhardwaj, Duc Anh Do, and Soujanya Poria. Language models are homer simpson! safety re-alignment of fine-tuned language models through task arithmetic. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14138–14149, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.

Hamed Damirchi, Ehsan Abbasnejad, Zhen Zhang, and Javen Shi. Decomposing task vectors for refined model editing. arXiv preprint arXiv:2512.22511, 2025.

Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 6491–6506, 2021.

Denis Emelin, Ronan Le Bras, Jena D Hwang, Maxwell Forbes, and Yejin Choi. Moral stories: Situated reasoning about norms, intents, actions, and their consequences. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 698–718, 2021.

Constanza Fierro and Fabien Roger. Steering language models with weight arithmetic. arXiv preprint arXiv:2511.05408, 2025.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849, 2022.

Anton Korznikov, Andrey Galichin, Alexey Dontsov, Oleg Y Rogov, Ivan Oseledets, and Elena Tutubalina. The rogue scalpel: Activation steering compromises llm safety. arXiv preprint arXiv:2509.22067, 2025.

Chanhyuk Lee, Jiho Choi, Chanryeol Lee, Donggyun Kim, and Seunghoon Hong. Adarank: Adaptive rank pruning for enhanced model merging. arXiv preprint arXiv:2503.22178, 2025a.

Yu-Ang Lee, Ching-Yun Ko, Tejaswini Pedapati, I-Hsin Chung, Mi-Yen Yeh, and Pin-Yu Chen. Star: Spectral truncation and rescale for model merging. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 496–505, 2025b.

Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Inference-time´ intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Remi Leblond, Tom´ Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods, 2022. URL https://arxiv. org/abs/2109.07958, 1, 2021.

Alexander H Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sade, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. Ministral´ 3. arXiv preprint arXiv:2601.08584, 2026.

Michael S Matena and Colin A Raffel. Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems, 35:17703–17716, 2022.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022a.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. arXiv preprint arXiv:2210.07229, 2022b.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. arXiv preprint arXiv:2110.11309, 2021.

Tsung-Min Pai, Jui-I Wang, Li-Chun Lu, Shao-Hua Sun, Hung-Yi Lee, and Kai-Wei Chang. Billy: Steering large language models via merging persona vectors for creative generation. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7870–7915, 2026.

Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors with model-written evaluations. In Findings of the association for computational linguistics: ACL 2023, pp. 13387–13434, 2023.

Gabriel Perin, Xuxi Chen, Shusen Liu, Bhavya Kailkhura, Zhangyang Wang, and Brian Gallagher. Rankmean: Module-level importance score for merging fine-tuned llm models. In Findings ofthe Association for Computational Linguistics: ACL 2024, pp. 1776–1782, 2024.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv preprint arXiv:2310.03693, 2023.

Alexandre Rame, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, L ´ eon Bottou, and David Lopez-Paz. ´ Model ratatouille: Recycling diverse models for out-of-distribution generalization. In International Conference on Machine Learning, pp. 28656–28679. PMLR, 2023.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15504–15522, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.828. URL https://aclanthology.org/2024.acl-long.828/.

Chung-En Sun, Ge Yan, Zimo Wang, and Tsui-Wei Weng. Steer2edit: From activation steering to component-level editing. arXiv preprint arXiv:2602.09870, 2026.

Seungjong Sun, Seo Yeon Baek, and Jang Hyun Kim. Personality vector: Modulating personality of large language models by model merging. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 24667–24688, 2025.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm. github.io/blog/qwen2.5/.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pp. 23965–23998. PMLR, 2022.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in neural information processing systems, 36:7093–7115, 2023.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications, and opportunities. ACM Computing Surveys, 58(8):1–41, 2026.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning, 2024.

## APPENDIX CONTENTS

A Proof of TFTV properties 15   
A.1 Norm Matching . . 15   
A.2 Steering 16   
A.3 Linearity . 16   
B Experimental Details 17   
C Qwen Composing Multiple Traits Additional Results 19   
D Trait Score Standard Deviations 20   
E Coherence scores 23   
F Additional Experiments and Ablations 26   
F.1 Qwen Out-of-Distribution test 26   
F.2 Additional traits . 26   
F.3 Additional Model Architectures 27   
F.4 Modules . 27   
F.5 Layers 28

## A PROOF OF TFTV PROPERTIES

In this section, we prove the arithmetic properties stated in Section 3. Recall that, for a module weight matrix $W _ { \ell } \in \mathbb { R } ^ { d \times l }$ with singular value decomposition

$$
W _ { \ell } = \sum _ { i = 1 } ^ { r } \sigma _ { i } u _ { i } v _ { i } ^ { \top } ,
$$

and normalized steering vector $\bar { s } _ { \ell } \in \mathbb { R } ^ { d }$ , the TFTV update is defined as

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) = \bar { s } _ { \ell } \left( \sum _ { i = 1 } ^ { r } \mathrm { s i g n } ( \mu _ { \ell } ^ { \top } v _ { i } ) \sigma _ { i } v _ { i } \right) ^ { \top } .
$$

For convenience, define

$$
q _ { \ell } : = \sum _ { i = 1 } ^ { r } \mathrm { s i g n } ( \mu _ { \ell } ^ { \top } v _ { i } ) \sigma _ { i } v _ { i } \in \mathbb { R } ^ { l } .
$$

Then

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) = \bar { s } _ { \ell } q _ { \ell } ^ { \top } .
$$

## A.1 NORM MATCHING

Property 1 (Norm matching). We prove that

$$
\lVert W _ { \ell } \rVert _ { F } = \lVert \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \rVert _ { F } .
$$

For the proof, we use the convention sign $( 0 ) = 1$ . In practice we note that, inner product zeroes are very rare because of floating-point arithmetic.

Proof. We first use the fact that the Frobenius norm of an outer product factorizes:

$$
\| \bar { s } _ { \ell } q _ { \ell } ^ { \top } \| _ { F } = \| \bar { s } _ { \ell } \| _ { 2 } \| q _ { \ell } \| _ { 2 } .
$$

Since $\bar { s } _ { \ell }$ is normalized, we have $\| \bar { s } _ { \ell } \| _ { 2 } = 1$ . Therefore,

$$
\Vert \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \Vert _ { F } = \Vert q _ { \ell } \Vert _ { 2 } .
$$

It remains to compute $\| q \ell \| _ { 2 }$ . By definition,

$$
q _ { \ell } = \sum _ { i = 1 } ^ { r } \mathrm { s i g n } ( \mu _ { \ell } ^ { \top } v _ { i } ) \sigma _ { i } v _ { i } .
$$

Thus, $q \ell$ is a linear combination of the right singular vectors $\{ v _ { i } \} _ { i = 1 } ^ { r }$ , with coefficients $\pm \sigma _ { i } .$ . Since the vectors $v _ { i }$ are orthonormal, the squared Euclidean norm of $q _ { \ell }$ is just the sum of the squared coefficients:

$$
\| q _ { \ell } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } .
$$

The signs do not matter, since they disappear after squaring.

On the other hand, the Frobenius norm of $W _ { \ell }$ is given by the squared sum of its singular values:

$$
\| W _ { \ell } \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } .
$$

Hence,

$$
\| q _ { \ell } \| _ { 2 } = \| W _ { \ell } \| _ { F } .
$$

Combining the two steps, we obtain

$$
\begin{array} { r } { \| \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \| _ { F } = \| q _ { \ell } \| _ { 2 } = \| W _ { \ell } \| _ { F } , } \end{array}
$$

which proves the claim.

## A.2 STEERING

Property 2 (Steering). We prove that

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \mu _ { \ell } = c \bar { s } _ { \ell } \qquad \mathrm { f o r ~ s o m e } ~ c \in \mathbb { R } _ { \ge 0 } .
$$

Proof. By definition,

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \mu _ { \ell } = \bar { s } _ { \ell } q _ { \ell } ^ { \top } \mu _ { \ell } = ( q _ { \ell } ^ { \top } \mu _ { \ell } ) \bar { s } _ { \ell } .
$$

Thus it suffices to show that $q _ { \ell } ^ { \top } \mu _ { \ell } \geq 0$ . Expanding,

$$
q _ { \ell } ^ { \top } \mu _ { \ell } = \sum _ { i = 1 } ^ { r } \mathrm { s i g n } ( \mu _ { \ell } ^ { \top } v _ { i } ) \sigma _ { i } v _ { i } ^ { \top } \mu _ { \ell } = \sum _ { i = 1 } ^ { r } \sigma _ { i } | \mu _ { \ell } ^ { \top } v _ { i } | .
$$

Since each singular value satisfies $\sigma _ { i } \geq 0 .$ , it follows that

$$
q _ { \ell } ^ { \top } \mu _ { \ell } = \sum _ { i = 1 } ^ { r } \sigma _ { i } \left| \mu _ { \ell } ^ { \top } v _ { i } \right| \ge 0 .
$$

Defining

$$
c : = q _ { \ell } ^ { \top } \mu _ { \ell } = \sum _ { i = 1 } ^ { r } \sigma _ { i } \left| \mu _ { \ell } ^ { \top } v _ { i } \right| \in \mathbb { R } _ { \ge 0 } ,
$$

we obtain

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } , \mu _ { \ell } ) \mu _ { \ell } = c \bar { s } _ { \ell } .
$$

## A.3 LINEARITY

Property 3 (Linearity). We prove that, for any $\beta , \gamma \in \mathbb { R }$ and steering vectors $\bar { s } _ { \ell } ^ { ( 1 ) } , \bar { s } _ { \ell } ^ { ( 2 ) } \in \mathbb R ^ { d }$

$$
\beta \operatorname { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 1 ) } , \mu _ { \ell } ) + \gamma \operatorname { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } ) = \operatorname { T F T V } \left( W _ { \ell } , \beta \bar { s } _ { \ell } ^ { ( 1 ) } + \gamma \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } \right)
$$

Proof. For fixed $W _ { \ell }$ and $\mu _ { \ell } .$ , the vector $q _ { \ell }$ does not depend on the steering vector. Therefore,

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 1 ) } , \mu _ { \ell } ) = \bar { s } _ { \ell } ^ { ( 1 ) } q _ { \ell } ^ { \top }
$$

and

Thus,

$$
\mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } ) = \bar { s } _ { \ell } ^ { ( 2 ) } q _ { \ell } ^ { \top } .
$$

$$
\begin{array} { r l } & { \beta \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 1 ) } , \mu _ { \ell } ) + \gamma \mathrm { T F T V } ( W _ { \ell } , \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } ) } \\ & { \qquad = \beta \bar { s } _ { \ell } ^ { ( 1 ) } q _ { \ell } ^ { \top } + \gamma \bar { s } _ { \ell } ^ { ( 2 ) } q _ { \ell } ^ { \top } } \\ & { \qquad = \left( \beta \bar { s } _ { \ell } ^ { ( 1 ) } + \gamma \bar { s } _ { \ell } ^ { ( 2 ) } \right) q _ { \ell } ^ { \top } . } \end{array}
$$

By the definition of TFTV, this is exactly

$$
\mathrm { T F T V } \left( W _ { \ell } , \beta \bar { s } _ { \ell } ^ { ( 1 ) } + \gamma \bar { s } _ { \ell } ^ { ( 2 ) } , \mu _ { \ell } \right) .
$$

This proves the claim.

## B EXPERIMENTAL DETAILS

Evaluation framework. We adopt the Persona Vectors framework (Chen et al., 2025) as our main evaluation setup, as it provides a natural testbed for controlled behavioral interventions and for mea suring behavior–utility trade-offs. We experiment with Llama-3.1-8B-Instruct<sup>2</sup> (Grattafiori et al., 2024) and Qwen-2.5-7B-Instruct<sup>3</sup> (Yang et al., 2024; Team, 2024). We focus on three target traits: evil, hallucination, and sycophancy. Evaluation details for the datasets used in Learning via addition and Forgetting via subtraction are provided in the corresponding paragraphs below.

Judge-based metrics. To measure target-trait control, we follow the Persona Vectors protocol and use gpt-4.1-mini-2025-04-14 as a judge. The judge assigns both trait-manifestation and coherence scores on a scale from 0 to 100, and we report the mean score over generated responses. Coherence is used as a generation-quality and utility metric, following prior work (Chen et al., 2025; Betley et al., 2025).

Filtering and steering-vector construction. We construct steering directions following the Persona Vectors (Chen et al., 2025) data generation and filtering protocol. Each dataset contains 20 questions, 5 trait-inducing system prompts, and 5 trait-suppressing system prompts. We form $\mathcal { X } _ { + }$ and X<sub>−</sub> by pairing each question with each inducing or suppressing prompt, respectively, yielding 100 prompts per set. For each prompt, we sample 10 completions. We discard completions with coherence scores below 50. For trait-inducing prompts, we retain only completions with trait scores of at least 50; for trait-suppressing prompts, we retain only completions with trait scores below 50. The resulting filtered completions are then used to construct the activation-space directions used by the baseline and by TFTV.

Decoding, utility, and hardware. Across all methods, we use temperature 1, top- $- p = 1$ , and a maximum of 1000 new tokens. To assess general utility, we report zero-shot MMLU accuracy (Hendrycks et al., 2020), evaluated on the full benchmark using the lm-evalu $\mathtt { a t i o n - h a r n e s s } ^ { 4 }$ (Gao et al., 2024). Each edited-model evaluation used one NVIDIA RTX A5000 GPU with 24 GB memory.

Learning via addition. For each of the 20 Persona Vectors evaluation questions, we sample 10 responses from the edited model, totalizing 200 generations.

We compare TFTV against Persona Vector inference-time steering, Steer2Edit (Sun et al., 2026), Task Vectors (Ilharco et al., 2022) and Constrastive Weight Steering (Fierro & Roger, 2025). For all hypeparameter tuning procedures, we select the model with highest trait manifestation subject to a coherence score of at least 70.

For inference-time steering, we follow Persona Vectors (Chen et al., 2025) and apply the intervention at layer 16 for Llama and layer 20 for Qwen. We sweep steering coefficients {0.4, 0.6, 0.8, 1.0, 1.2} for Llama and {0.5, 1.0, 1.5, 2.0, 2.5} for Qwen.

For Steer2Edit, we follow the hyperparameter tuning procedure proposed in their paper and evaluate all combinations of $\alpha , \rho _ { \mathrm { m l p } }$ , and $\rho _ { \mathrm { a t t n } }$ in {0.1, 0.3, 0.5, 0.7, 0.9}, for a total of 125 candidate configurations per trait. We first perform a greedy-decoding sweep over this grid, then select the three configurations with the best coherence–trait trade-off for full evaluation.

For Task Vectors, we follow Fierro & Roger (2025) and fine-tune each model on positive-trait completions generated by the model $( \mathrm { i } . \mathrm { e } . , \mathcal { D } _ { + } )$ using Axolotl <sup>5</sup>. We train for five epochs with a sequence length of 4096, a micro-batch size of 2, and four gradient-accumulation steps. We use LoRA on all linear layers with rank 32, $\alpha \ : = \ : 1 6$ , and no dropout, while also training the embedding and language-model head. Optimization uses 8-bit AdamW with a learning rate of $\mathrm { 5 \times 1 0 ^ { - 5 } }$ and a linear schedule.

For Contrastive Weight Steering (CWS), we also follow Fierro & Roger (2025). Fine-tuning on the positive and negative completion datasets follows the Task Vectors setup, except that training is limited to 100 update steps and uses a learning rate of $1 \times 1 0 ^ { - 5 }$ , five warmup steps, and a weight decay of 0.01. We evaluate and save checkpoints every 20 steps, apply early stopping with a patience of two evaluations, and restore the best checkpoint. To construct the weight-steering update, we sweep k ∈ {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}.

Table 6: Hyperparameter configurations for Llama-3.1 and Qwen-2.5. We report the selected configurations for TFTV, activation steering, and Steer2Edit (S2E) for each target trait.
<table><tr><td>Model</td><td>Method</td><td>Trait</td><td>Configuration</td><td></td></tr><tr><td rowspan="7">Llama-3.1</td><td>TFTV</td><td>Evil Hallucination</td><td>α = 0.04, layers [14, 20) α = 0.03, layers [13, 30)</td><td></td></tr><tr><td rowspan="2">Steering</td><td>Sycophancy</td><td>α = 0.05, layers [14, 20)</td><td></td></tr><tr><td>Evil</td><td> $\alpha = 0 . 8 , \mathrm { l a y e r 1 6 }$ </td><td></td></tr><tr><td rowspan="2"></td><td>Hallucination</td><td> $\alpha = 1 . 0 , \mathrm { l a j e r } 1 6$ </td><td></td></tr><tr><td>Sycophancy</td><td> $\alpha = 1 . 2 , \mathrm { l a y e r 1 6 }$ </td><td></td></tr><tr><td rowspan="2">S2E</td><td>Evil</td><td></td><td> $\alpha = 0 . 3 , \rho _ { \mathrm { M L P } } = 0 . 7 , \rho _ { \mathrm { a t t n } } = 0 . 1$ </td></tr><tr><td>Hallucination</td><td></td><td> $\alpha = 0 . 1 , \rho _ { \mathrm { M L P } } = 0 . 7 , \rho _ { \mathrm { a t t n } } = 0 . 3$ </td></tr><tr><td rowspan="3">CWS</td><td>Sycophancy</td><td></td><td> $\alpha = 0 . 1 , \rho _ { \mathrm { M L P } } = 0 . 5 , \rho _ { \mathrm { a t t n } } = 0 . 3$ </td></tr><tr><td>Evil Hallucination</td><td> $k = 4$ </td><td></td></tr><tr><td></td><td> $k = 6$  k = 4</td><td></td></tr><tr><td rowspan="6">Qwen-2.5</td><td rowspan="2">TFTV</td><td>Sycophancy Evil</td><td></td><td></td></tr><tr><td>Hallucination</td><td>α = 0.03, layers [16, 24)  $\alpha = 0 . 0 5 , \mathrm { l a y e r s } \ [ 1 2 , 2 2 )$ </td><td></td></tr><tr><td rowspan="2"></td><td>Sycophancy</td><td> $\alpha = 0 . 0 4 , \mathrm { l a y e r s } \ [ 1 8 , 2 5 )$ </td><td></td></tr><tr><td>Evil</td><td> $\alpha = 1 . 0 , \mathrm { l a y e r 2 0 }$ </td><td></td></tr><tr><td rowspan="2">Steering</td><td>Hallucination</td><td> $\alpha = 2 . 0 , \mathrm { l a y e r 2 0 }$ </td><td></td></tr><tr><td>Sycophancy</td><td> $\alpha = 2 . 0 , \mathrm { l a y e r 2 0 }$ </td><td></td></tr><tr><td rowspan="3">S2E</td><td>Evil</td><td></td><td> $\alpha = 0 . 3 , \rho _ { \mathrm { M L P } } = 0 . 9 , \rho _ { \mathrm { a t t n } } = 0 . 3$ </td><td></td></tr><tr><td>Hallucination</td><td></td><td> $\alpha = 0 . 1 , \rho _ { \mathrm { M L P } } = 0 . 7 , \rho _ { \mathrm { a t t n } } = 0 . 3$ </td><td></td></tr><tr><td>Sycophancy</td><td></td><td> $\alpha = 0 . 1 , \rho _ { \mathrm { M L P } } = 0 . 9 , \rho _ { \mathrm { a t t n } } = 0 . 3$ </td><td></td></tr><tr><td rowspan="3">CWS</td><td>Evil</td><td></td><td> $k = 1 4$ </td><td></td></tr><tr><td rowspan="2"></td><td>Hallucination</td><td> $k = 1 8$ </td><td></td></tr><tr><td>Sycophancy</td><td> $k = 1 0$ </td><td></td></tr></table>

For TFTV, we apply edits only to attention modules. For each trait, we select the layer interval that produced the strongest trait manifestation in the Persona Vectors analysis (Chen et al., 2025). For Llama, we use layers [14, 20) for evil and sycophancy, and [13, 30) for hallucination. For Qwen, we use layers [16, 24) for evil, [12, 22) for hallucination, and [18, 25) for sycophancy. We sweep the update coefficient $\alpha \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 3 , 0 . 0 4 , 0 . 0 5 \}$

Forgetting via subtraction. To test whether TFTV can suppress a target trait, we explicitly prompt the model to elicit that trait. Following Persona Vectors, for each of the 20 evaluation questions we use 5 trait-eliciting system prompts, yielding 100 prompt pairs in total. For each prompt pair, we sample 10 responses and evaluate them with the same judge and metrics used in the learning-viaaddition experiments.

Composing multiple traits. The evaluation setup is the same as Forgetting via subtraction.

In Table 6, we present the final parameters chosen in the tuning procedure described.

## C QWEN COMPOSING MULTIPLE TRAITS ADDITIONAL RESULTS

Table 7: Composition results on Qwen 2.5. Best results over training-free methods are in Bold. The results support that TFTV outperforms training-free methods under composed edits.
<table><tr><td>Method</td><td>Composition</td><td>Evil ↓</td><td>Hall. ↓</td><td> $\operatorname { S y c . }$  ↓</td><td>MMLU↑</td><td>GSM8K ↑</td></tr><tr><td rowspan="4">Steering</td><td> $\mathrm { E } + \mathrm { H }$ </td><td>4.99</td><td>35.18</td><td></td><td>70.89</td><td>48.98</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>9.14</td><td></td><td>8.83</td><td>71.34</td><td>65.20</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>20.29</td><td>4.23</td><td>70.88</td><td>38.81</td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>1.17</td><td>29.16</td><td>3.30</td><td>70.60</td><td>6.22</td></tr><tr><td rowspan="4">Steer2Edit</td><td> $\mathrm { E } + \mathrm { H }$ </td><td>0.58</td><td>27.81</td><td></td><td>39.57</td><td>2.20</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>0.00</td><td></td><td>4.22</td><td>68.03</td><td>70.13</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>38.40</td><td>0.06</td><td>23.60</td><td>0.23</td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>2.10</td><td>36.46</td><td>10.69</td><td>48.99</td><td>3.71</td></tr><tr><td rowspan="4">TFTV</td><td> $\mathrm { E } + \mathrm { H }$ </td><td>0.00</td><td>0.39</td><td></td><td>69.64</td><td>72.55</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>0.00</td><td></td><td>2.58</td><td>71.47</td><td>78.92</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>0.72</td><td>0.13</td><td>69.69</td><td>72.10</td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>0.00</td><td>0.81</td><td>0.13</td><td>69.73</td><td>69.60</td></tr><tr><td rowspan="4">Task Vectors</td><td> $\mathrm { E } + \mathrm { H }$ </td><td>1.97</td><td>18.59</td><td></td><td>71.82</td><td>76.72</td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>7.77</td><td></td><td>5.68</td><td>71.69</td><td>79.76</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>15.71</td><td>8.15</td><td>72.03</td><td>76.57</td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>3.19</td><td>31.83</td><td>6.52</td><td>71.81</td><td>74.00</td></tr><tr><td rowspan="4">CWS</td><td> $\mathrm { E } + \mathrm { H }$ </td><td>0.00</td><td>0.01</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td>0.00</td><td></td><td>30.08</td><td>70.92</td><td>54.21</td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td>0.12</td><td>0.13</td><td>70.55 71.41</td><td>22.37 76.72</td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td>0.00</td><td>0.05</td><td>6.83</td><td>70.37</td><td>60.50</td></tr></table>

Table 7 presents the Qwen 2.5 composition results, complementing the Llama 3.1 results in Table 4. TFTV achieves stronger suppression than inference steering on all nine measured trait scores, while keeping MMLU within 1.25 points and improving GSM8K by 13.72–63.38 points. Compared with Steer2Edit, TFTV achieves stronger suppression on seven scores, ties on one, and obtains higher MMLU and GSM8K for every composition. Task Vectors yields slightly higher utility but weaker suppression on all nine scores. CWS matches or improves $\mathrm { T F T V } ^ { \circ } \mathrm { s }$ suppression on seven scores, but TFTV provides higher GSM8K in three of four compositions; notably, CWS reduces GSM8K to 22.37 for $\mathbf { E } + \mathbf { S } .$ Overall, TFTV provides the strongest suppression–utility trade-off among the training-free methods and remains competitive with the fine-tuned baselines.

## D TRAIT SCORE STANDARD DEVIATIONS

We report standard deviations for trait manifestation scores in the main experiments. Scores are first averaged across runs for each prompt, and standard deviations are then computed over these prompt-level averages. They therefore reflect variability across prompts rather than variability across individual generations or runs.

Overall, near-zero or near-saturated trait scores tend to have lower variability. Intermediate mean scores often exhibit larger standard deviations, consistent with a near-bimodal prompt-level pattern in which some prompts reliably elicit strong trait expression while others elicit little or none. This pattern is broadly consistent across models and editing methods.

Tables 8, 9, 10, 11, and 12 complement Tables 1, 2, 3, 4, and 7, respectively.

Table 8: We report trait manifestation scores for each method in Table 1; higher values indicate stronger trait manifestation. Standard deviations are computed over prompt-level scores after averaging runs within each prompt.
<table><tr><td>Model</td><td>Trait</td><td>Base</td><td>Steering</td><td>S2E</td><td>TFTV</td><td>TV</td><td>CWS</td></tr><tr><td rowspan="3"></td><td rowspan="3">Evil Llama 3.1 Hallucinating</td><td>0.00±0.00</td><td>14.06±16.35</td><td>26.69±32.92</td><td>63.26±30.94</td><td>95.62±6.35</td><td>90.67±10.38</td></tr><tr><td>17.03±14.94</td><td>61.51±23.80</td><td>89.25±14.24</td><td>98.55±1.81</td><td>94.54±6.68</td><td>99.09±2.38</td></tr><tr><td>3.60±4.04</td><td>63.86±18.75</td><td>81.44±16.56</td><td>94.64±2.68</td><td>89.55±8.62</td><td>87.13±8.70</td></tr><tr><td rowspan="3">Qwen 2.5</td><td>Evil</td><td>0.00±0.00</td><td>8.58±12.12</td><td>5.04±14.09</td><td>61.96±35.67</td><td>74.45±21.16</td><td>65.64±20.30</td></tr><tr><td>Hallucinating</td><td>11.46±14.65</td><td>88.96±11.79</td><td>94.23±3.52</td><td>99.80±0.51</td><td>73.98±23.85</td><td>99.96±0.07</td></tr><tr><td>Sycophantic</td><td>4.35±9.87</td><td>90.46±2.92</td><td>50.39±25.96</td><td>89.78±3.31</td><td>60.13±16.81</td><td>91.02±4.34</td></tr></table>

Table 9: We report trait manifestation scores for the suppression edits in Table 2; lower values indicate stronger suppression. Standard deviations are computed over prompt-level scores after averaging runs within each prompt.
<table><tr><td>Model</td><td>Trait</td><td>Base</td><td>Steering</td><td>S2E</td><td>TFTV</td><td>TV</td><td>CWS</td></tr><tr><td></td><td>Evil Llama 3.1 Hallucinating Sycophantic</td><td>95.42±7.40 97.53±4.96 92.06±7.78</td><td>58.67±27.51 79.13±20.44 53.65±28.56</td><td>0.13±0.49 5.31±13.11 5.31±3.25</td><td>0.49±1.45 1.85±4.58 14.62±10.50</td><td>8.59±17.49 26.02±22.89 32.29±26.74</td><td>1.64±4.83 3.26±5.44 24.91±18.40</td></tr><tr><td>Qwen 2.5</td><td>Evil Hallucinating Sycophantic</td><td>69.85±28.80 79.94±22.63 64.72±22.28</td><td>12.77±17.82 38.61±27.96 10.88±9.55</td><td>0.00±0.00 34.31±18.79 3.69±2.64</td><td>0.17±0.99 0.14±0.85 3.14±3.14</td><td>24.46±34.21 33.66±31.90 10.48±12.54</td><td>0.00±0.00 0.00±0.00 5.79±3.67</td></tr></table>

Table 10: We compare trait suppression scores for composed TFTV edits against their corresponding single-trait edits in Table 3. Lower values indicate stronger suppression. Standard deviations are computed over prompt-level scores after averaging runs within each prompt. Here, E, H, and S denote evil, hallucinating, and sycophantic, respectively.
<table><tr><td>Base model</td><td>Edit</td><td>Evil ↓</td><td> $\mathrm { H a l l . \downarrow }$ </td><td> $\operatorname { S y c . \downarrow }$ </td></tr><tr><td rowspan="8">Llama 3.1</td><td>Base</td><td> $9 5 . 4 2 { \pm } 7 . 4 0 $ </td><td> $9 7 . 5 3 { \pm } 4 . 9 6 $ </td><td> $9 2 . 0 6 { \pm } 7 . 7 8 $ </td></tr><tr><td>E</td><td> $0 . 4 9 { \pm } 1 . 4 5 $ </td><td> $9 3 . 7 9 { \pm } 1 1 . 2 1$ </td><td> $7 1 . 3 2 { \pm } 2 0 . 7 0 $ </td></tr><tr><td>H</td><td> $2 6 . 2 2 { \pm } 2 9 . 3 8 $ </td><td> $1 . 8 5 { \pm } 4 . 5 8 $ </td><td> $5 0 . 5 7 { \pm } 2 6 . 3 3$ </td></tr><tr><td>S</td><td> $6 3 . 2 9 { \pm } 3 1 . 8 1 $ </td><td> $8 9 . 2 7 { \pm } 1 3 . 9 6 $ </td><td> $1 4 . 6 2 { \pm } 1 0 . 5 0 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 1 4 { \pm } 1 . 0 0 $ </td><td> $1 . 5 1 { \pm } 4 . 8 5$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 6 9 { \pm } 1 . 8 8 $ </td><td></td><td> $1 3 . 1 0 { \pm } 9 . 5 5 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $1 . 8 7 { \pm } 5 . 9 4 $ </td><td> $2 . 2 5 { \pm } 2 . 9 6 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 3 { \pm } 0 . 2 0 $ </td><td> $1 . 5 4 { \pm } 4 . 6 8$ </td><td> $1 . 9 4 { \pm } 2 . 6 3 $ </td></tr><tr><td rowspan="7">Qwen 2.5</td><td>Base</td><td> $6 9 . 8 5 { \pm } 2 8 . 8 0 $ </td><td> $7 9 . 9 4 \pm 2 2 . 6 3$ </td><td> $6 4 . 7 2 { \pm } 2 2 . 2 8$ </td></tr><tr><td>E</td><td> $0 . 1 7 { \pm } 0 . 9 9$ </td><td> $6 9 . 7 9 { \pm } 2 6 . 5 2 $ </td><td> $3 7 . 4 6 { \pm } 2 0 . 9 2$ </td></tr><tr><td>H</td><td> $0 . 0 2 { \pm } 0 . 1 7$ </td><td> $0 . 1 4 { \pm } 0 . 8 5$ </td><td> $6 . 6 1 { \pm } 9 . 1 6$ </td></tr><tr><td>S</td><td> $4 2 . 7 0 { \pm } 3 1 . 3 2 $ </td><td> $5 7 . 7 6 { \pm } 2 9 . 2 8$ </td><td> $3 . 1 4 { \pm } 3 . 1 4$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 3 9 { \pm } 2 . 7 3 $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 2$ </td><td></td><td> $2 . 5 8 { \pm } 2 . 9 0 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $0 . 7 2 { \pm } 3 . 4 2$ </td><td> $0 . 1 3 { \pm } 0 . 3 1$ </td></tr><tr><td></td><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 8 1 { \pm } 3 . 7 9$ </td><td> $0 . 1 3 { \pm } 0 . 3 6 ^ { }$ </td></tr></table>

Table 11: Trait suppression scores for composed negation edits on Llama 3.1 in Table 4; lower is better. Standard deviations are computed over prompt-level scores after averaging runs within each prompt. Pairwise compositions are evaluated only on their target traits, with unmeasured entries marked as -. Here, E, H, and S denote evil, hallucination, and sycophancy, respectively.
<table><tr><td>Method</td><td>Composition</td><td>Evil ↓</td><td>Hall. ↓</td><td>Syc. ↓</td></tr><tr><td rowspan="4">Steering</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $3 4 . 0 4 { \pm } 2 8 . 6 0 $ </td><td> $7 4 . 1 4 { \pm } 2 3 . 8 0 $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $3 1 . 7 8 { \pm } 2 7 . 4 0$ </td><td></td><td> $4 6 . 1 9 { \pm } 2 7 . 7 6$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $6 0 . 6 2 { \pm } 2 5 . 4 5$ </td><td> $3 5 . 7 4 { \pm } 2 3 . 5 8 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $1 7 . 0 5 { \pm } 1 9 . 1 0 $ </td><td> $5 6 . 2 2 { \scriptstyle \pm 2 6 . 9 5 }$ </td><td> $2 8 . 3 4 \pm 1 9 . 8 4$ </td></tr><tr><td rowspan="4">Steer2Edit</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 2 6 { \pm } 1 . 0 9$ </td><td> $2 . 3 9 { \pm } 5 . 9 6 \ $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 6 0 { \pm } 1 . 6 9$ </td><td></td><td> $5 . 7 6 { \pm } 3 . 1 4$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $6 . 4 1 { \pm } 1 0 . 6 8$ </td><td> $4 . 0 7 { \pm } 4 . 2 8$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 5 1 { \pm } 1 . 6 4$ </td><td> $1 2 . 4 6 { \pm } 1 7 . 8 3$ </td><td> $2 . 1 1 \pm 3 . 7 7$ </td></tr><tr><td rowspan="4">TFTV</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 1 4 { \pm } 1 . 0 0 $ </td><td> $1 . 5 1 { \pm } 4 . 8 5$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 6 9 { \pm } 1 . 8 8 $ </td><td></td><td> $1 3 . 1 0 { \pm } 9 . 5 5 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $1 . 8 7 { \pm } 5 . 9 4 $ </td><td> $2 . 2 5 { \pm } 2 . 9 6 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 3 { \pm } 0 . 2 0 \ $ </td><td> $1 . 5 4 { \pm } 4 . 6 8 $ </td><td> $1 . 9 4 { \pm } 2 . 6 3 $ </td></tr><tr><td rowspan="4">Task Vectors</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 0 2 { \pm } 0 . 1 7$ </td><td> $2 7 . 7 4 { \pm } 2 3 . 3 3$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td></td><td></td><td></td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td> $0 . 2 0 { \pm } 1 . 1 8 $ </td><td></td><td> $7 . 4 3 { \pm } 9 . 4 7 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 2 5 { \pm } 1 . 1 5$ </td><td> $2 4 . 7 9 { \pm } 2 2 . 8 6$   $4 4 . 7 1 { \pm } 2 6 . 5 5$ </td><td> $6 . 7 0 { \scriptstyle \pm 1 0 . 8 2 }$   $6 . 1 6 { \pm } 6 . 6 6$ </td></tr><tr><td rowspan="4">CWS</td><td> $\mathrm { E } + \mathrm { H }$ </td><td></td><td></td><td></td></tr><tr><td></td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $3 . 1 7 { \pm } 5 . 2 3$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 4$ </td><td></td><td> $1 3 . 9 8 { \pm } 8 . 6 4$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $2 . 2 7 { \pm } 4 . 9 6$ </td><td> $4 . 0 3 { \pm } 3 . 3 0 $ </td></tr><tr><td></td><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 1 $ </td><td> $5 . 5 5 { \pm } 8 . 4 1$ </td><td> $5 . 5 0 { \pm } 3 . 2 3 $ </td></tr></table>

Table 12: Trait suppression scores for composed negation edits on Qwen 2.5 in Table 7; lower is better. Standard deviations are computed over prompt-level scores after averaging runs within each prompt. Pairwise compositions are evaluated only on their target traits, with unmeasured entries marked as -. Here, E, H, and S denote evil, hallucination, and sycophancy, respectively.
<table><tr><td>Method</td><td>Composition</td><td>Evil ↓</td><td>Hall. ↓</td><td> $\operatorname { S y c . \downarrow }$ </td></tr><tr><td rowspan="4">Steering</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $4 . 9 9 { \pm } 1 0 . 5 5 $ </td><td> $3 5 . 1 8 { \pm } 2 8 . 1 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $9 . 1 4 { \pm } 1 1 . 5 6$ </td><td></td><td> $8 . 8 3 { \pm } 7 . 3 2 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $2 0 . 2 9 { \pm } 2 1 . 4 4 $ </td><td> $4 . 2 3 { \pm } 2 . 9 5 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $1 . 1 7 { \pm } 3 . 3 2$ </td><td> $2 9 . 1 6 { \pm } 2 5 . 7 6$ </td><td> $3 . 3 0 { \pm } 2 . 0 5 $ </td></tr><tr><td rowspan="4">Steer2Edit</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 5 8 { \pm } 2 . 2 0 \ $ </td><td> $2 7 . 8 1 { \pm } 2 2 . 5 2 $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td></td><td> $4 . 2 2 { \pm } 3 . 2 4 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $3 8 . 4 0 { \pm } 1 6 . 9 8 \ $ </td><td> $0 . 0 6 { \pm } 0 . 5 2$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $2 . 1 0 { \pm } 4 . 7 6$ </td><td> $3 6 . 4 6 { \pm } 2 2 . 3 9$ </td><td> $1 0 . 6 9 { \pm } 1 0 . 7 9 $ </td></tr><tr><td rowspan="4">TFTV</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 3 9 { \pm } 2 . 7 3 $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 2$ </td><td></td><td> $2 . 5 8 { \pm } 2 . 9 0 $ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $0 . 7 2 { \pm } 3 . 4 2$ </td><td> $0 . 1 3 { \pm } 0 . 3 1$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 8 1 { \pm } 3 . 7 9$ </td><td> $0 . 1 3 { \pm } 0 . 3 6 \ $ </td></tr><tr><td rowspan="5">Task Vectors</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $1 . 9 7 { \pm } 5 . 9 3 $ </td><td> $1 8 . 5 9 { \pm } 2 2 . 4 8 $ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 . 7 7 { \pm } 1 5 . 6 4$ </td><td></td><td> $5 . 6 8 { \pm } 8 . 1 8$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $1 5 . 7 1 { \pm } 2 0 . 4 1$ </td><td> $8 . 1 5 { \pm } 1 0 . 9 1$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $3 . 1 9 { \pm } 8 . 1 6$ </td><td> $3 1 . 8 3 { \pm } 2 7 . 2 8 $ </td><td> $6 . 5 2 { \pm } 6 . 5 4$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">CWS</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 0 1 { \pm } 0 . 0 3$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td></td><td> $3 0 . 0 8 { \pm } 1 3 . 9 2$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $0 . 1 2 { \pm } 0 . 7 2$ </td><td> $0 . 1 3 { \pm } 0 . 2 6 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 0 5 { \pm } 0 . 2 5 $ </td><td> $6 . 8 3 { \pm } 8 . 6 5$ </td></tr></table>

## E COHERENCE SCORES

We present coherence scores for the experiments in Section 4. Tables 13, 14, 15, 16 and 17 report coherence scores corresponding to Tables 1, 2, 3, 4 and 7, respectively. Overall, most coherence scores remain above 70. The clearest failures occur for Steer2Edit under composed edits, particularly on Qwen 2.5, where several coherence scores approach zero. TFTV generally maintains high coherence; its main exceptions are the Llama 3.1 sycophancy evaluations under the H+S and E+H+S compositions, which obtain scores of 62.29 and 61.14, respectively. These results indicate that aggressive multi-trait editing can reduce generation quality, although TFTV remains comparatively robust across most settings.

Table 13: We report absolute coherence scores C for each target trait in Table 1. Higher C indicates more coherent generations.
<table><tr><td>Model</td><td>Trait</td><td>Base</td><td>Steering</td><td>S2E</td><td>TFTV</td><td>TV</td><td>CWS</td></tr><tr><td rowspan="3"></td><td>Evil</td><td>97.88±1.57</td><td>84.97±6.03</td><td>72.33±26.94</td><td>70.55±17.36</td><td>89.54±7.10</td><td>84.67±12.72</td></tr><tr><td>Llama 3.1 Hallucinating</td><td>90.50±4.57</td><td>78.09±9.44</td><td>86.55±5.26</td><td>73.90±11.93</td><td>89.40±6.87</td><td>80.76±11.88</td></tr><tr><td>Sycophantic</td><td>98.92±0.70</td><td>92.89±2.08</td><td>70.02±12.98</td><td>85.47±4.19</td><td>90.94±4.54</td><td>84.18±8.36</td></tr><tr><td rowspan="3">Qwen 2.5</td><td>Evil</td><td>99.17±1.23</td><td>93.22±4.63</td><td>96.72±3.00</td><td>70.93±17.26</td><td>88.91±6.76</td><td>74.00±11.14</td></tr><tr><td>Hallucinating</td><td>94.78±2.67</td><td>85.51±6.81</td><td>82.25±9.59</td><td>83.57±10.20</td><td>92.73±3.09</td><td>84.89±12.87</td></tr><tr><td>Sycophantic</td><td>99.59±0.44</td><td>87.20±9.80</td><td>94.71±3.38</td><td>90.57±4.17</td><td></td><td>97.95±1.35 79.42±12.30</td></tr></table>

Table 14: We report absolute coherence scores C for each target trait under suppression edits in Table 2. Higher C indicates more coherent generations.
<table><tr><td>Model</td><td>Trait</td><td>Base</td><td>Steering</td><td>S2E</td><td>TFTV</td><td>TV</td><td>CWS</td></tr><tr><td rowspan="3"></td><td>Evil</td><td>90.72±6.12</td><td>85.68±8.38</td><td>89.13±6.34</td><td>83.62±7.72</td><td>80.47±15.31</td><td>86.12±10.49</td></tr><tr><td>Llama 3.1 Hallucinating</td><td>90.74±5.15</td><td>80.45±7.24</td><td>90.65±8.42</td><td>91.42±7.75</td><td>87.24±6.74</td><td>65.15±13.43</td></tr><tr><td>Sycophantic</td><td>89.48±5.99</td><td>94.42±2.60</td><td>94.32±2.09</td><td>90.24±2.70</td><td>96.56±1.58</td><td>96.01±1.48</td></tr><tr><td rowspan="3">Qwen 2.5</td><td>Evil</td><td>89.39±9.15</td><td>87.53±9.21</td><td>97.50±2.23</td><td>93.91±4.66</td><td>92.78±8.43</td><td>86.57±5.32</td></tr><tr><td>Hallucinating</td><td>94.81±2.70</td><td>86.96±4.39</td><td>0.09±0.27</td><td>97.90±3.35</td><td>94.73±2.64</td><td>98.60±2.06</td></tr><tr><td>Sycophantic</td><td>97.33±2.50</td><td>96.89±1.34</td><td>97.42±1.31</td><td>96.00±1.10</td><td>98.73±1.04</td><td>97.09±1.48</td></tr></table>

Table 15: We compare coherence scores for composed TFTV edits against their corresponding single-trait edits in Table 3. We report per-prompt aggregated standard deviations. Here, E, H, and S denote evil, hallucinating, and sycophantic, respectively.
<table><tr><td>Base model</td><td>Edit</td><td>Evil C ↑</td><td>Hall. C ↑</td><td> $\operatorname { S y c . } C \uparrow$ </td></tr><tr><td rowspan="7">Llama 3.1</td><td>Base</td><td> $9 0 . 7 2 \pm 6 . 1 2$ </td><td> $9 0 . 7 4 \pm 5 . 1 5$ </td><td> $8 9 . 4 8 \pm 5 . 9 9$ </td></tr><tr><td>E</td><td> $8 3 . 6 2 \pm 7 . 7 2$ </td><td> $9 1 . 1 4 \pm 5 . 0 1$ </td><td> $9 2 . 7 7 \pm 4 . 6 7$ </td></tr><tr><td>H</td><td> $7 4 . 1 8 \pm 1 4 . 9 6$ </td><td> $9 1 . 4 2 \pm 7 . 7 5$ </td><td> $8 4 . 1 4 \pm 6 . 6 6$ </td></tr><tr><td>S</td><td> $8 1 . 0 4 \pm 1 2 . 7 4$ </td><td> $8 9 . 3 1 \pm 5 . 3 5$ </td><td> $9 0 . 2 4 \pm 2 . 7 0$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 5 . 5 5 \pm 1 2 . 6 0$ </td><td> $9 1 . 2 6 \pm 7 . 8 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 9 . 0 2 \pm 1 0 . 3 6$ </td><td></td><td> $8 8 . 2 4 \pm 3 . 2 4$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $9 1 . 9 3 \pm 7 . 2 6$ </td><td> $6 2 . 2 9 \pm 8 . 7 8$ </td></tr><tr><td rowspan="8">Qwen 2.5</td><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $7 0 . 8 9 \pm 1 4 . 5 2$ </td><td> $9 2 . 2 4 \pm 8 . 5 1$ </td><td> $6 1 . 1 4 \pm 8 . 7 5$ </td></tr><tr><td>Base</td><td> $8 9 . 3 9 \pm 9 . 1 5$ </td><td> $9 4 . 8 1 \pm 2 . 7 0$ </td><td> $9 7 . 3 3 \pm 2 . 5 0$ </td></tr><tr><td>E</td><td> $9 3 . 9 1 \pm 4 . 6 6$ </td><td> $9 4 . 1 4 \pm 2 . 9 1$ </td><td> $9 6 . 7 8 \pm 3 . 2 9$ </td></tr><tr><td>H</td><td> $8 8 . 1 4 \pm 8 . 5 7$ </td><td> $9 7 . 9 0 \pm 3 . 3 5$ </td><td> $8 9 . 7 4 \pm 4 . 9 0$ </td></tr><tr><td>S</td><td> $8 8 . 7 3 \pm 1 0 . 0 1$ </td><td> $9 3 . 7 8 \pm 2 . 3 4$ </td><td> $9 6 . 0 0 \pm 1 . 1 0$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H }$ </td><td> $8 7 . 4 0 \pm 8 . 6 3$ </td><td> $9 7 . 6 0 \pm 4 . 0 6$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $9 0 . 9 5 \pm 3 . 1 0$ </td><td></td><td> $9 3 . 8 8 \pm 1 . 5 2$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$   $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $8 3 . 4 1 \pm 8 . 0 9$ </td><td> $9 7 . 5 6 \pm 3 . 6 8$   $9 5 . 3 2 \pm 4 . 1 7$ </td><td> $9 0 . 5 9 \pm 2 . 3 9$   $8 8 . 3 3 \pm 3 . 0 3$ </td></tr></table>

Table 16: Coherence scores C for composed negation edits on Llama 3.1, in Table 4; higher is better. Pairwise compositions are evaluated only on their target traits, with unmeasured entries marked as -. Here, E, H, and S denote evil, hallucination, and sycophancy, respectively.
<table><tr><td>Method</td><td>Composition</td><td>Evil C ↑</td><td> $_ { \mathrm { H a l l . } C \uparrow }$ </td><td> $\operatorname { S y c . } C \uparrow$ </td></tr><tr><td rowspan="4">Steering</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 6 . 7 1 \pm 1 0 . 9 2$ </td><td> $8 0 . 9 7 \pm 6 . 9 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 8 . 7 9 \pm 1 0 . 6 5$ </td><td></td><td> $9 3 . 9 7 \pm 2 . 5 6$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $7 5 . 4 0 \pm 8 . 4 9$ </td><td> $8 9 . 5 9 \pm 4 . 2 3$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $7 0 . 7 6 \pm 1 1 . 1 1$ </td><td> $7 0 . 9 6 \pm 8 . 1 3$ </td><td> $8 8 . 6 6 \pm 4 . 7 9$ </td></tr><tr><td rowspan="4">Steer2Edit</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 2 . 8 3 \pm 1 2 . 8 5$ </td><td> $9 2 . 2 7 \pm 8 . 0 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 6 . 4 6 \pm 1 0 . 6 9$ </td><td></td><td> $8 3 . 4 7 \pm 8 . 4 2$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $8 7 . 6 2 \pm 9 . 2 3$ </td><td> $8 5 . 2 1 \pm 7 . 1 8$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $4 3 . 9 3 \pm 1 3 . 1 2$ </td><td> $5 9 . 8 5 \pm 1 7 . 7 8$ </td><td> $2 1 . 3 1 \pm 1 4 . 9 5$ </td></tr><tr><td rowspan="4">TFTV</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 5 . 5 5 \pm 1 2 . 6 0$ </td><td> $9 1 . 2 6 \pm 7 . 8 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 9 . 0 2 \pm 1 0 . 3 6$ </td><td></td><td> $8 8 . 2 4 \pm 3 . 2 4$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $9 1 . 9 3 \pm 7 . 2 6$ </td><td> $6 2 . 2 9 \pm 8 . 7 8$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $7 0 . 8 9 \pm 1 4 . 5 2$ </td><td> $9 2 . 2 4 \pm 8 . 5 1$ </td><td> $6 1 . 1 4 \pm 8 . 7 5$ </td></tr><tr><td rowspan="5">Task Vectors</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 8 . 3 0 \pm 1 2 . 7 4$ </td><td> $8 0 . 3 6 \pm 1 1 . 8 5$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $7 6 . 5 7 \pm 1 2 . 3 2$ </td><td></td><td> $9 2 . 7 2 \pm 3 . 1 4$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $8 3 . 7 4 \pm 8 . 6 1$ </td><td> $9 4 . 9 7 \pm 2 . 9 3 $ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $5 8 . 8 5 \pm 1 4 . 4 9$ </td><td> $6 3 . 2 0 \pm 1 6 . 4 0$ </td><td></td></tr><tr><td></td><td></td><td></td><td> $5 8 . 5 2 \pm 1 3 . 9 7$ </td></tr><tr><td rowspan="4">CWS</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $5 8 . 3 7 \pm 1 6 . 4 9$ </td><td> $6 4 . 3 0 \pm 1 5 . 1 9$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $8 0 . 3 5 \pm 1 6 . 6 9$ </td><td></td><td> $9 6 . 1 9 \pm 1 . 2 1$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $7 6 . 3 2 \pm 1 4 . 9 3$ </td><td> $6 4 . 2 5 \pm 1 0 . 2 0$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $6 3 . 9 9 \pm 1 5 . 4 5$ </td><td> $6 7 . 5 9 \pm 1 7 . 2 7$ </td><td> $6 2 . 1 8 \pm 1 0 . 1 4$ </td></tr></table>

Table 17: Coherence scores $C$ for composed negation edits on Qwen 2.5 in Table 7; higher is better. Pairwise compositions are evaluated only on their target traits, with unmeasured entries marked as -. Here, E, H, and S denote evil, hallucination, and sycophancy, respectively.
<table><tr><td>Method</td><td>Composition</td><td>Evil C ↑</td><td> $_ { \mathrm { H a l l . } C \uparrow }$ </td><td> $\operatorname { S y c . } C \uparrow$ </td></tr><tr><td rowspan="4">Steering</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 5 . 0 9 \pm 1 3 . 6 8$ </td><td> $7 5 . 5 2 \pm 1 0 . 5 4$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $8 4 . 4 3 \pm 9 . 0 2$ </td><td></td><td> $9 5 . 6 3 \pm 1 . 7 0$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $7 3 . 9 3 \pm 1 1 . 4 4$ </td><td> $8 6 . 2 2 \pm 6 . 5 8$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $5 7 . 9 9 \pm 1 1 . 9 0$ </td><td> $4 7 . 9 2 \pm 1 3 . 8 6$ </td><td> $7 6 . 2 9 \pm 9 . 9 6$ </td></tr><tr><td rowspan="4">Steer2Edit</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $3 5 . 8 0 \pm 1 7 . 5 8$ </td><td> $1 4 . 5 2 \pm 7 . 6 5$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $9 5 . 1 1 \pm 2 . 3 3$ </td><td></td><td> $9 5 . 9 6 \pm 1 . 6 6$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $6 . 7 2 \pm 3 . 5 8$ </td><td> $9 . 8 2 \pm 3 . 9 2$ </td><td> $5 . 5 6 \pm 2 . 9 9$ </td></tr><tr><td rowspan="4">TFTV</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $8 7 . 4 0 \pm 8 . 6 3$ </td><td> $9 7 . 6 0 \pm 4 . 0 6$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $9 0 . 9 5 \pm 3 . 1 0$ </td><td></td><td> $9 3 . 8 8 \pm 1 . 5 2$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $9 7 . 5 6 \pm 3 . 6 8$ </td><td> $9 0 . 5 9 \pm 2 . 3 9$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $8 3 . 4 1 \pm 8 . 0 9$ </td><td> $9 5 . 3 2 \pm 4 . 1 7$ </td><td> $8 8 . 3 3 \pm 3 . 0 3$ </td></tr><tr><td rowspan="5">Task Vectors</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $9 4 . 1 6 \pm 6 . 9 1$ </td><td></td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td></td><td> $9 3 . 7 1 \pm 3 . 5 6$ </td><td></td></tr><tr><td></td><td> $9 0 . 8 8 \pm 9 . 3 1$ </td><td></td><td> $9 7 . 3 7 \pm 1 . 5 0$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $9 3 . 0 2 \pm 3 . 6 3$ </td><td> $9 6 . 0 6 \pm 1 . 8 2$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $4 3 . 1 5 \pm 2 1 . 2 9$ </td><td> $7 3 . 2 5 \pm 1 8 . 6 7$ </td><td> $8 9 . 4 3 \pm 7 . 5 9$ </td></tr><tr><td rowspan="4">CWS</td><td> $\mathrm { E } + \mathrm { H }$ </td><td> $7 7 . 4 5 \pm 6 . 6 0$ </td><td> $8 4 . 0 3 \pm 7 . 9 2$ </td><td></td></tr><tr><td> $\mathrm { E } + \mathrm { S }$ </td><td> $8 4 . 4 2 \pm 5 . 4 7$ </td><td></td><td> $8 1 . 6 5 \pm 7 . 8 0$ </td></tr><tr><td> $\mathbf { S } + \mathbf { H }$ </td><td></td><td> $9 7 . 4 6 \pm 1 . 9 0$ </td><td> $9 6 . 5 8 \pm 1 . 7 8$ </td></tr><tr><td> $\mathrm { E } + \mathrm { H } + \mathrm { S }$ </td><td> $8 4 . 3 6 \pm 7 . 6 4$ </td><td> $9 1 . 9 9 \pm 4 . 4 6$ </td><td> $8 5 . 0 2 \pm 8 . 6 1$ </td></tr></table>

## F ADDITIONAL EXPERIMENTS AND ABLATIONS

## F.1 QWEN OUT-OF-DISTRIBUTION TEST

In this section, we present additional results on OOD test, for the Qwen model. We evaluate evil (E) edits on Moral Stories (Emelin et al., 2021), which tests moral judgment; hallucination (H) edits on TruthfulQA (Lin et al., 2021), which tests resistance to common misconceptions; and sycophancy (S) edits on tasks covering user opinions in NLP, philosophy, and politics (Perez et al., 2023).

Table 18: TFTV remains robust OOD on Qwen. Bold indicates movement in the expected direction relative to the base: addition decreases morality and truthfulness and increases sycophancy, while negation reverses these effects. TruthfulQA MC1 measures top-1 accuracy, whereas MC2 measures the normalized probability mass assigned to truthful answers.
<table><tr><td>Task</td><td>Addition</td><td>Base</td><td>Negation</td></tr><tr><td>Moral Stories (E)</td><td>44.79</td><td>48.74</td><td>53.29</td></tr><tr><td>TruthfulQA MC1 (H)</td><td>28.27</td><td>47.98</td><td>52.75</td></tr><tr><td>TruthfulQA MC2 (H)</td><td>42.94</td><td>64.69</td><td>69.04</td></tr><tr><td>Sycophancy NLP (S)</td><td>95.08</td><td>93.40</td><td>88.01</td></tr><tr><td>Sycophancy Phil. (S)</td><td>98.80</td><td>98.35</td><td>96.99</td></tr><tr><td>Sycophancy Pol. (S)</td><td>78.40</td><td>80.14</td><td>80.26</td></tr></table>

Results are presented in Table 18. Overall, TFTV transfers consistently to out-of-distribution evaluations on Qwen. For Moral Stories, TruthfulQA MC1 and MC2, and the NLP and philosophy sycophancy splits, both addition and negation shift performance in the expected directions relative to the base model. The largest changes are observed on TruthfulQA, indicating particularly strong transfer of the hallucination-related direction. The politics sycophancy split is the only exception, where the intervention produces only marginal changes and does not exhibit the expected directional behavior.

## F.2 ADDITIONAL TRAITS

We provide additional addition-setting results for two traits, humorous and optimistic, on both Llama and Qwen. Hyperparameters are selected using the same procedure described in Appendix B.

Table 19: TFTV addition results for humorous and optimistic behaviors on Llama and Qwen. Trait measures the target behavior, while MMLU and GSM8K measure general capability preservation.
<table><tr><td rowspan="2">Trait</td><td rowspan="2">Model</td><td colspan="2">Trait ↑</td><td colspan="2">MMLU↑</td><td colspan="2">GSM8K↑</td></tr><tr><td>Base</td><td>TFTV</td><td>Base</td><td>TFTV</td><td>Base</td><td>TFTV</td></tr><tr><td rowspan="2">Humorous</td><td>Llama</td><td>0.05</td><td>79.16</td><td>68.26</td><td>68.29</td><td>77.10</td><td>76.72</td></tr><tr><td>Qwen</td><td>0.00</td><td>88.39</td><td>71.83</td><td>71.64</td><td>78.54</td><td>78.77</td></tr><tr><td rowspan="2">Optimistic</td><td>Llama</td><td>81.08</td><td>96.89</td><td>68.26</td><td>67.92</td><td>77.10</td><td>77.10</td></tr><tr><td>Qwen</td><td>83.01</td><td>99.27</td><td>71.83</td><td>71.50</td><td>78.54</td><td>77.56</td></tr></table>

For Llama, we use coefficients of 0.04 and 0.05 for the humorous and optimistic directions, respectively, applied to layers [14, 20). For Qwen, we use the same coefficients, applied to layers [16, 24).

As shown in Table 19, TFTV substantially increases both humorous and optimistic behavior across Llama and Qwen while largely preserving MMLU and GSM8K performance. The effect is particularly pronounced for humor, where the base models exhibit near-zero scores and TFTV raises them to 79.16 on Llama and 88.39 on Qwen.

## F.3 ADDITIONAL MODEL ARCHITECTURES

We provide additional addition-setting results for two architectures with different model scales: gemma- $\mathtt { - 4 - E 2 B - i t } ^ { 6 }$ (Team et al., 2026) and Ministral-3-14B-Instruct<sup>7</sup> (Liu et al., 2026). Hyperparameters are selected using the same procedure described in Appendix B.

For Gemma, we use a coefficient of 0.05 and layers [15, 22) for all traits. For Ministral, we use layers [18, 25), with coefficients of 0.04 for evil and sycophancy and 0.03 for hallucination. MMLU is evaluated in the 5-shot setting for Gemma.

Table 20: TFTV addition results on Gemma and Ministral. Trait measures the target behavior, while MMLU and GSM8K measure general capability preservation.
<table><tr><td colspan="2"></td><td colspan="2">Trait ↑</td><td colspan="2">MMLU↑</td><td colspan="2">GSM8K↑</td></tr><tr><td>Trait</td><td>Model</td><td>Base</td><td>TFTV</td><td>Base</td><td>TFTV</td><td>Base</td><td>TFTV</td></tr><tr><td rowspan="2">Evil</td><td>Gemma</td><td>0.00</td><td>84.00</td><td>60.69</td><td>58.73</td><td>74.60</td><td>75.44</td></tr><tr><td>Ministral</td><td>0.00</td><td>68.51</td><td>76.47</td><td>76.41</td><td>79.68</td><td>79.45</td></tr><tr><td rowspan="2">Hallucination</td><td>Gemma</td><td>9.61</td><td>95.91</td><td>60.69</td><td>59.81</td><td>74.60</td><td>73.69</td></tr><tr><td>Ministral</td><td>31.40</td><td>94.62</td><td>76.47</td><td>76.36</td><td>79.68</td><td>78.17</td></tr><tr><td rowspan="2">Sycophancy</td><td>Gemma</td><td>4.51</td><td>91.27</td><td>60.69</td><td>59.42</td><td>74.60</td><td>73.77</td></tr><tr><td>Ministral</td><td>4.29</td><td>96.10</td><td>76.47</td><td>76.46</td><td>79.68</td><td>79.45</td></tr></table>

Results are presented in Table 20. TFTV consistently increases the target behavior across all three traits for both Gemma and Ministral, while largely preserving MMLU and GSM8K performance. These results indicate that TFTV generalizes beyond Llama and Qwen to additional model architectures and scales.

## F.4 MODULES

We compare edits applied to the MLP down projections, the attention output projections, and both modules simultaneously, using the same hyperparameters as in Section 4.1. Experiments are performed with Llama 3.1.

![](images/74cb000346411862a6aacf872bd8b5b40ab1103524b41856e408c8bedc845961.jpg)  
Figure 5: Attention-only edits yield the best trade-offs. Top: trait manifestation vs. coherence, with marker size indicating coefficients from 0.01 to 0.05. Bottom: MMLU stays mostly stable across coefficients.

Figure 5 shows the resulting behavior-utility trade-offs. Overall, applying TFTV to MLP modules yields weaker trait control and larger coherence degradation, whereas applying the update only to attention output projections provides the best trade-off. Combining attention and MLP edits does not consistently improve over attention-only edits, suggesting that the location of the update is an important factor in the effectiveness of TFTV. MMLU remains largely stable across module choices, with the exception of hallucination, where we observe a small drop. These results indicate that component selection is important for effective TFTV edits.

## F.5 LAYERS

Next, we ablate layer selection by applying TFTV to attention output projections over nonoverlapping four-layer windows and varying α.

![](images/b28469e321c8223dd290a1349474e85feaf81e2c068745f15ccac66ca128e315.jpg)  
Figure 6: Trait control and utility vary differently across layers. Each row plots a metric against the TFTV layer interval [a, b), with a inclusive and b exclusive: trait score, coherence, and MMLU from top to bottom.

Figure 6 shows that middle-layer edits produce the strongest trait manifestation, with a secondary increase in the final layers. Coherence degradation follows a similar trend, and in some windows larger coefficients further degrade coherence without substantially improving trait scores, worsening the trade-off. MMLU behaves differently: early-layer edits hurt utility more, while later layers better preserve it. Overall, TFTV is most effective in middle-to-late layers, making layer selection an important factor for targeted editing.