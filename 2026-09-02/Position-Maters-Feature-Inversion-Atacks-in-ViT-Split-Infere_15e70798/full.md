# Position Maters: Feature Inversion Atacks in ViT Split Inference with Token Reduction and Shufling

Stefano Leggio, Giulio Rossolini, Alessandro Biondi<sup>∗</sup>

Department of Excellence in Robotics & AI

Scuola Superiore Sant’Anna

Pisa, Italy

## Abstract

Vision Transformers (ViTs) are increasingly used in split-inference systems, where edge devices transmit intermediate token represen tations to a remote cloud. In this setting, token reduction lowers computation and communication costs, while token shufling disrupts the spatial organization of the transmitted tokens, potentially limiting information leakage. However, their privacy benefits remain unclear against feature inversion attacks, which attempt to reconstruct the input from the transmitted embeddings. In this work, we show that, despite disrupting the spatial structure required by conventional reconstruction attacks, transmitted token embeddings retain substantial positional information. Based on this observation, we introduce the Spatially Aligned Reconstruction Attack (SARA), a unified pipeline that predicts token positions, restores their spatial layout, reconstructs missing embeddings using a feature-space masked autoencoder, and recovers the input image. Our results demonstrate that token shufling provides only apparent privacy, as SARA largely reconstructs the original token organization. Token reduction ofers stronger protection, but significant leakage persists when the retained tokens preserve sufficient semantic and positional information. Finally, we introduce a lightweight edge-side defense that removes positional embeddings and progressively adapts the edge-side transformer blocks through knowledge distillation. It substantially reduces attack performance against SARA, while preserving downstream task accuracy and requiring no changes to the cloud-side model.

## CCS Concepts

• Computing methodologies → Machine learning; • Security and privacy → Privacy protections.

## Keywords

Split Inference, Vision Transformers, Feature Inversion Attacks, Privacy-Preserving Machine Learning

## 1 Introduction

Vision Transformers (ViTs) [6] have become a reference architecture for computer vision, achieving strong performance across a wide range of tasks [22, 27, 33]. ViTs process an image by splitting it into a sequence of patch tokens, each encoding the visual content of a local image region together with information about its spatial position. This sequence-based representation makes intermediate embeddings naturally well suited to token-level manipulations, often without modifying the model parameters or retraining the downstream network. For example, token reduction methods [2, 18, 26, 28] drop or merge redundant and less informative tokens, thereby reducing the computational cost of subsequent Transformer blocks while preserving a favorable trade-of with task performance. In contrast, token shufling methods [21, 37, 40] permute the token sequence, disrupting the explicit correspondence between individual tokens and the spatial layout of the input image. Although these operations are primarily introduced for eficiency or representation manipulation, they may also provide an apparent privacy benefit by limiting the amount of information exposed in intermediate features and/or concealing the correspondence between transmitted tokens and their original image locations.

The combination of eficiency and potential privacy benefits makes these operations particularly appealing in split inference scenarios [12, 15, 30], where a neural network is partitioned between an edge device holding the private input and a remote cloud. In privacy-sensitive applications, the edge device processes the input through the initial portion of the model and transmits the resulting intermediate representation to the cloud, which completes the inference. Token manipulations can therefore be naturally applied on the edge side before transmission, reducing client-side computation and communication costs while potentially obscuring the spatial correspondence between the transmitted tokens and the original image regions. In this setting, it is reasonable to consider an honest-but-curious threat model [11, 12, 23], in which the cloud correctly performs the assigned inference computation but may also attempt to reconstruct the private input from the received features, as illustrated in Figure 1.

However, the privacy provided by these token manipulations remains insuficiently investigated, despite being of crucial importance in split-inference scenarios. Conventional feature inversion attacks typically assume that intermediate representations preserve a fixed spatial organization [9, 11, 12, 17, 19, 38]. Consequently, their failure after token reduction or token shufling may simply result from the violation of this assumption, rather than from a genuine reduction in information leakage. Indeed, the transmitted token embeddings may still retain suficient semantic and positional information to recover their original spatial organization and reconstruct the private input. This motivates the need for reconstruction attacks specifically designed to operate on manipulated token representations.

Paper contributions. In this work, we provide a systematic privacy-oriented analysis of token reduction and token shufling in ViT-based split inference. We first show that intermediate token embeddings retain substantial positional information, even after passing through multiple ViT blocks, which enables a positionaware attacker to infer their original spatial location. Building on this observation, we introduce the Spatially Aligned Reconstruction

![](images/6cc9b80b5b1791b98f8fab63d7cbf2fcdac448844a4cded0025a48eafccd4ce6.jpg)  
Figure 1: Illustration of the threat model. The edge device transmits the smashed data ℎ to the cloud, where an honest-but-curious attacker attempts to output a reconstruction of the original input image �˜.

Attack (SARA), a unified reconstruction pipeline that predicts the original positions of the transmitted tokens, restores their spatial arrangement, reconstructs missing token embeddings through a feature-space masked autoencoder, and finally recovers the input image. This pipeline enables a consistent evaluation of information leakage under both token reduction and token shufling.

Experimental results show that token shufling alone provides only a false sense of privacy, as its apparent benefits can be greatly disrupted by SARA. Token reduction ofers stronger protection by discarding part of the representation, but introduces a privacy– utility trade-of due to the resulting degradation in task accuracy. Nevertheless, its privacy benefits remain limited when the retained tokens preserve suficient semantic and positional information for SARA to infer the missing content.

Motivated by these findings, we finally introduce a lightweight edge-side defense, specifically tailored to split inference settings, where adapting the cloud-side model should be avoided to accom modate multiple edge devices. The proposed mechanism provides a simple yet efective baseline that attenuates positional cues before transmission, and substantially reduces reconstruction leakage against ad-hoc attacks, while maintaining a favorable trade-of with task performance.

Paper Structure. The remainder of the paper is organized as follows. Section 2 introduces the threat model for ViT-based split inference and reviews the token manipulation strategies considered in this work. Section 3 presents the proposed SARA attack pipeline and provides a detailed description of its components. Section 4 introduces a simple yet efective edge-side defense. Section 5 first evaluates the efectiveness of SARA against the considered token manipulation strategies and then assesses the benefits of the proposed defense. Finally, Sections 6 and 7 discuss the connections with the related literature and present the conclusions and limitations, respectively.

## 2 Preliminaries and Threat Model

This section formalizes the split-inference setting under consideration, the token-manipulation operations applied at the edge, and the capabilities and objective of the adversary.

## 2.1 ViT tokenization and token operations

Let $x \in [ 0 , 1 ] ^ { C \times H \times W }$ be an input image. A ViT � partitions � into $\begin{array} { r } { N _ { 0 } = \frac { H W } { p ^ { 2 } } } \end{array}$ non-overlapping image patches of size ${ p \times p . }$ . Each patch is then linearly projected into a �-dimensional embedding and combined with a learnable positional embedding (PE). The patch input sequence to the first transformer block is therefore

$$
\begin{array} { r } { \boldsymbol { h } ^ { ( 0 ) } = \left[ \boldsymbol { h } ^ { ( 0 ) , 1 } ; \ldots ; \boldsymbol { h } ^ { ( 0 ) , N _ { 0 } } \right] \in \mathbb { R } ^ { N _ { 0 } \times d } , } \end{array}\tag{1}
$$

where $h ^ { ( 0 ) , i } \in \mathbb { R } ^ { d }$ denotes the embedding associated with the �-th image patch, including its positional embedding. Note that, from the sequence $h ^ { ( 0 ) }$ , we intentionally omit the classification token to denote only the patch-token representations.<sup>1</sup>

From a layer-level perspective, let the ViT consist of � blocks $\{ B _ { 1 } , \ldots , B _ { L } \}$ . In the original ViT architecture, without additional token-manipulation operations, each block preserves the sequence length such that the patch-token representation produced after block $B _ { \ell }$ is $h ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { \ell } \times d }$ , with $N _ { \ell } = N _ { 0 }$ for every $\ell \in \left\{ 1 , \ldots , L \right\}$

Token-reduction methods modify ViT inference by progressively decreasing the number of tokens processed by subsequent transformer blocks. In particular, operations such as token dropping [28] and token merging [2] remove or aggregate selected tokens, respectively. Consequently, the sequence length becomes $N _ { i } \leq N _ { j }$ for $i > j .$ For simplicity, we first consider a fixed token reduction policy, where � denotes the number of patch tokens removed from the sequence, either by dropping or merging, after each transformer block output. Assuming that the same reduction amount � is applied after each of the first ℓ transformer blocks, the number of patch tokens available in $h ^ { ( \ell ) }$ is $N _ { \ell } = \operatorname* { m a x } \{ 0 , \ : N _ { 0 } - \ell r \}$ , where $N _ { 0 }$ denotes the initial number of patch tokens. Once $N _ { \ell } = 0 ,$ , only the class token remains, and no further token reduction is applied in the subsequent blocks. Importantly, increasing � improves computational and communication eficiency by reducing the number ofprocessed and transmitted tokens. However, stronger reduction may improve privacy at the cost of downstream accuracy. To capture this tradeof, in Section 5 we introduce a unified metric by jointly accounting for task utility and privacy leakage.

In contrast, token shufling does not change the number of transmitted tokens. Instead, it permutes their sequence order at the output of a selected block ℓ. Thus, when shufling is applied without token reduction, it holds $N _ { \ell } = N _ { 0 }$ , while the representation transmitted after block ℓ is an ordered permutation of the original token sequence.

## 2.2 ViT Split Inference

We consider a split point �, with $1 \leq k < L ,$ such that the edge executes the first � transformer blocks and produces the intermediate representation $h ^ { ( k ) } = f _ { \mathrm { e } } ^ { ( k ) } ( x )$ , while the cloud executes the remaining � −� blocks and returns the predicted logits $f _ { \mathrm { c } } ^ { ( k ) } \left( h ^ { ( k ) } \right)$ Consequently, when token reduction or shufling is applied on the edge-side, the corresponding operations are included in $f _ { \mathrm { e } } ^ { ( k ) }$ while the learned ViT parameters remain unchanged. The cloudside model therefore receives the manipulated token sequence and continues the inference procedure from the selected split point. As discussed in Section 1, these techniques are particularly relevant to split inference because they can improve computational and communication eficiency and, in the case of token shufling, conceal the original spatial arrangement of the transmitted tokens.

## 2.3 Threat Model

We consider a honest-but-curious cloud that correctly executes the cloud-side portion of the ViT but attempts to infer private information about the edge’s input. For an input image �, the adversary observes the intermediate representation $h ^ { ( k ) } = f _ { \mathrm { e } } ^ { ( k ) } ( x )$ , with $N _ { k }$ tokens embedding, transmitted at split point �. Depending on the edge-side configuration, $h ^ { ( k ) }$ may contain a reduced token sequence and/or a shufled token sequence. Note that, in this threat model, the adversary cannot directly observe private edge-side information associated with a specific input, such as the original image, the spatial provenance of the retained or merged tokens, or the permutation used to shufle the transmitted sequence. The adversary is passive and does not alter the inference protocol, the transmitted representation, or the legitimate prediction. Its objective is to reconstruct the private input from the observed smashed data.

Following these assumptions, the attacker trains a reconstruction model $\mathbf { \Delta } _ { g _ { \theta } } : \mathbb { R } ^ { N _ { k } \times d }  \mathbf { \bar { [ 0 , 1 ] } } ^ { C \times H \times W }$ using an auxiliary dataset $\mathcal { D } _ { \mathrm { a u x } } = \left\{ \left( x _ { i } , h _ { i } ^ { ( k ) } \right) \right\} _ { i = 1 } ^ { M }$ , where parameters are learned by solving

$$
\theta _ { g } ^ { * } = \arg \operatorname* { m i n } _ { \theta _ { g } } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathcal { L } _ { \mathrm { r e c } } \left( g _ { \theta } \left( h _ { i } ^ { ( k ) } \right) , x _ { i } \right) ,\tag{2}
$$

$\mathcal { L } _ { \mathrm { r e c } }$ denotes the reconstruction loss, e.g., a pixel-level mean square error (MSE). After training of ${ \mathit { g } } ,$ given a smashed representation of an unseen private input, the attacker produces $\tilde { x } = g \Big ( h ^ { ( k ) } \Big )$ We assess privacy leakage through the similarity between � and �˜, measured using pixel-level and perceptual reconstruction metrics (see Section 5.1). Higher reconstruction quality indicates that more information about the private input can be recovered from the transmitted token representation.

## 3 Attack Pipeline

This section describes the Spatially Aligned Reconstruction Attack (SARA). We first motivate the idea by highlighting the importance of recovering positional information. We then present the steps as a unified approach for reconstructing images from features that have undergone token reduction or shufling on the edge side.

## 3.1 Motivation and Intuition

Standard reconstruction attacks on intermediate representations typically train a convolutional decoder to recover the input image directly from the transmitted data $h ^ { ( k ) }$ [9, 11, 12, 17, 19, 38]. These approaches generally assume that the intermediate representation preserves a fixed spatial organization that can be mapped back to the original image via classic operations. However, token reduction and shufling violate this assumption by altering the number or ordering of the transmitted tokens, potentially creating a misleading impression of improved privacy when evaluated against convolutional-based reconstructions.

To illustrate this limitation, we evaluate a standard reconstruction decoder under two configurations of ViT-B/16 (setup details in Section 5.1): an aligned setting, in which tokens preserve their original spatial order, and a shufled setting, in which their order is randomly permuted. As reported in the table in Figure 2 (top), shufling reduces the average Structural Similarity Index Measure (SSIM) [36] from 0.700 to 0.246. As expected, a consistent drop across all split points confirms that conventional reconstruction decoders strongly depend on the original token arrangement. However, shufling only changes the order of the transmitted tokens; it does not directly remove the semantic or positional information encoded in their representations. This observation naturally raises the question of whether shufling operations can be inverted by inferring the original spatial position of each token. In principle, recovering the corresponding patch index would allow the tokens to be rearranged before reconstruction, thereby restoring the spatial structure and recovering the same reconstruction quality as in the aligned setting.

To investigate this possibility, we train diferent token-position predictors to classify each intermediate token according to its original position in the input patch grid. As shown in Figure 2 (bottom), positional information remains highly accessible throughout the ViT. Even a linear predictor achieves nearly perfect accuracy in the early layers, although its performance decreases substantially for deeper representations. In contrast, the Transformer-based predictor maintains high accuracy through the layers.

## 3.2 Spatially Aligned Reconstruction Attack

We propose the Spatially Aligned Reconstruction Attack (SARA), a three-stage pipeline designed to improve the reconstruction of smashed representations $h ^ { ( k ) }$ afected by token reduction or shuffling. The overall pipeline is illustrated in Figure 3 and consists of three stages, which are described more technically as follows: (i) a Token Position Predictor, which estimates the original spatial position of each transmitted token and constructs a spatially aligned, full-length masked representation; (ii) a Masked Autoencoder, which imputes the representations associated with missing token positions; and (iii) a Decoder, which maps the restored smashed representation back to the input image through a convolutional reconstruction network.

<table><tr><td>Split point</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>Avg.</td></tr><tr><td>Aligned</td><td>0.92</td><td>0.89</td><td>0.84</td><td>0.79</td><td>0.74</td><td>0.69</td><td>0.65</td><td>0.61</td><td>0.56</td><td>0.53</td><td>0.49</td><td>0.70</td></tr><tr><td>Shuffled</td><td>0.18</td><td>0.21</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.26</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.25</td></tr></table>

![](images/983c64e27745f0bd711113e5d64a2b6ede7e68da6e0b2df035b7444fa09e5ecf.jpg)

![](images/c0906627e068b2fd9154af681af1e4284b63e145710d8cf561d4aedb9c9573b8.jpg)  
Figure 2: (Top) SSIM achieved by a conventional convolutional decoder on ImageNet when the transmitted ViT-B tokens retain their original spatial order (Aligned) or are randomly permuted (Shufled); (center) illustration of the original input and reconstructions at split point 4 with a convolutional decoder; (bottom) Top-1 token-position accuracy across ViT layers for diferent predictors.

3.2.1 Token Position Prediction andPlacement. Given an unordered set of smashed tokens, the Token Position Predictor (TPP) estimates the original image-patch position of each token and rearranges the tokens accordingly to recover their spatial organization.

For simplicity, we omit the split-point superscript and denote the smashed representation at a split point � as $h = \mathbf { \bar { \{ } }  h ^ { 1 } , \ldots , h ^ { N _ { k } } \}$ $h ^ { i } \in \mathbb { R } ^ { d } ;$ , where $N _ { k }$ is the number of tokens transmitted after token manipulation, $N _ { 0 }$ is the number of image patches in the original token sequence, and � is the token dimensionality. The TPP is implemented as a position classifier $C _ { \mathrm { p o s } }$ that assigns each transmitted token a probability distribution over the $N _ { 0 }$ possible patch positions: $P = C _ { \mathrm { p o s } } ( h ) \in [ 0 , 1 ] ^ { N _ { k } \times N _ { 0 } }$ , $\begin{array} { r } { \sum _ { j = 1 } ^ { N _ { 0 } } P _ { i , j } = 1 } \end{array}$ , where $P _ { i , j }$ denotes the predicted probability that token $h ^ { i }$ originated from the �-th image patch. For each token, we define its predicted position and the corresponding confidence as

$$
\hat { p } _ { i } = \operatorname { a r g } \operatorname * { m a x } _ { j \in \{ 1 , . . . , N _ { 0 } \} } P _ { i , j } , \qquad c _ { i } = \operatorname * { m a x } _ { j \in \{ 1 , . . . , N _ { 0 } \} } P _ { i , j } .\tag{3}
$$

The transmitted tokens are then placed into a full-length representation $\tilde { h } \in \mathbb { R } ^ { N _ { 0 } \times d }$ . For each position $j ,$ we first identify the set of suficiently confident tokens assigned to that position:

$$
{ \cal { T } } _ { j } = \{ i | { \hat { p } } _ { i } = j \wedge c _ { i } \geq \tau \} ,\tag{4}
$$

where � is a confidence threshold. If multiple tokens are assigned to the same position, we retain the one with the highest confidence. The spatially aligned representation is therefore constructed as

$$
\tilde { h } _ { j } = \left\{ \begin{array} { l l } { h _ { i _ { j } ^ { * } } , } & { \mathrm { i f } ~ { \bar { J } } _ { j } \neq \emptyset , } \\ { h _ { \mathrm { v o i d } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad i _ { j } ^ { * } = \arg \operatorname* { m a x } _ { i \in \bar { J } _ { j } } c _ { i } ,\tag{5}
$$

where $h _ { \mathrm { v o i d } } = \pmb { 0 } \in \mathbb { R } ^ { d }$ denotes the void value. This operation restores the spatial indexing required by the subsequent reconstruction stages. Positions associated with removed tokens, low-confidence predictions, or unresolved assignments remain marked by the void value and are subsequently processed by the masked autoencoder.

For training, the attacker uses an auxiliary image dataset to extract the corresponding smashed representations and simulate randomly shufling across their token order. The TPP is then optimized using a cross-entropy loss, where the ground-truth label of each token corresponds to its original patch index.

3.2.2 Reconstruction ofMissing Tokens. The spatially aligned representation produced by the previous step, $\tilde { h } \in \mathbb { R } ^ { N _ { 0 } \times d }$ , may still contain missing token embeddings, represented by zero vectors at their corresponding positions. Whenever such missing positions remain, we adopt the principle of masked auto-encoders (MAEs) [10] to recover the unavailable representations.

In particular, a feature-space masked autoencoder, denoted by M, maps this incomplete representation to a complete reconstructed representation: $h _ { \mathrm { r e c } } = \boldsymbol { \mathcal { M } } ( \tilde { h } ) \in \mathbb { R } ^ { N _ { 0 } \times d }$

To do this, the model is trained The model is trained to recover the complete smashed representation from a partially observed version of it; a subset of token embeddings in a full smashed representation, denoted for simplicity as $\bar { h _ { \mathrm { f u l l } } } \in \mathbb { R } ^ { N _ { 0 } \times d }$ , collected by the attacker from the auxiliary dataset, is randomly selected and replaced with zero vectors, producing a masked representation $h _ { \mathrm { m a s k } } \in \mathbb { R } ^ { N _ { 0 } \times d }$ that emulates the incomplete representation $( \tilde { h } )$ encountered at inference time. The model is then optimized using the MSE between the reconstructed and complete representations:

$$
\mathcal { L } _ { \mathrm { M A E } } = \frac { 1 } { N _ { 0 } ~ d } \left\| \boldsymbol { M } ( h _ { \mathrm { m a s k } } ) - h _ { \mathrm { f u l l } } \right\| _ { F } ^ { 2 } ,\tag{6}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. $\mathrm { B y }$ learning the dependencies among the visible token embeddings, the model infers the representations associated with missing spatial positions. When $\tilde { h }$ contains no missing positions, the masked-autoencoder stage is bypassed, and we directly set $h _ { \mathrm { r e c } } = \tilde { h }$

3.2.3 Image reconstruction. The completed smashed representation $h _ { r e c }$ is finally passed to an image reconstruction decoder, denoted by Dec, which reconstructs an approximation of the original input image $\tilde { x } = D e c ( h _ { r e c } )$ . In particular, the reconstructed patch embeddings in $h _ { \mathrm { r e c } }$ are first linearly projected and reshaped into a spatial grid. A convolutional decoder composed of upsampling blocks then progressively restores the original image resolution. As commonly done in reconstruction attacks [9, 11, 12, 17, 19, 38], the decoder is trained on an auxiliary image dataset available to the attacker and optimized by minimizing the mean squared error between the reconstructed image and the corresponding groundtruth input. Note that we train the decoder independently of the

![](images/b2d6c6879b37b48f24b4aefec7cf9d6726180086591c4a5d935f8ce698f2c552.jpg)

Figure 3: Overview of the Spatially Aligned Reconstruction Attack (SARA). The pipeline consists of three stages: (1) token position prediction and spatial alignment, (2) missing-token reconstruction via a Masked Autoencoder, and (3) image reconstruction with a convolutional decoder.  
Algorithm 1 Progressive Positionless Edge Finetuning   
Require: Pretrained edge $f _ { \mathrm { e } } ^ { ( k ) }$ , cloud model $f _ { \mathrm { c } } ^ { ( k ) }$ , unlabeled adap  
tation dataset ${ \mathcal { D } } _ { \mathrm { A } } ,$ split point �, number of epochs $N _ { \mathrm { e p o c h } }$   
1: Initialize the student edge $\bar { f } _ { \mathrm { e } } ^ { ( k ) } \gets f _ { \mathrm { e } } ^ { ( k ) }$   
2: Remove its positional embeddings   
3: for � = 1 to � do   
4: for � = 1 to $N _ { \mathrm { e p o c h } }$ do   
5: for all mini-batches $\mathcal { B } \subset \mathcal { D } _ { \mathrm { A } }$ do   
6: ${ \mathcal { L } } _ { \mathrm { K D } } \gets D _ { \mathrm { K L } } ( f _ { \mathrm { c } } ^ { ( k ) } ( { \bar { f } } _ { \mathrm { e } } ^ { ( k ) } ( { \mathcal { B } } ) ) \| f _ { \mathrm { c } } ^ { ( k ) } ( f _ { \mathrm { e } } ^ { ( k ) } ( { \mathcal { B } } ) ) )$   
7: Update the �-th block of $\bar { f } _ { \mathrm { e } } ^ { ( k ) }$ to minimize L<sub>KD</sub>   
8: end for   
9: end for   
10: end for   
11: return finetuned student edge $\bar { f } _ { \mathrm { e } } ^ { ( k ) }$

MAE stage, allowing us to better isolate the benefits of the preceding stages and provide a fairer comparison with conventional reconstruction approaches based solely on a decoder.

## 4 Proposed Defense

To mitigate the high efectiveness of SARA, as demonstrated by the experimental results in Section 5, we introduce a simple yet efective edge-side defense that limits the attacker’s ability to infer the positional information retained in the smashed representations.

The proposed defense pursues two objectives. First, it aims to attenuate the positional information encoded in the smashed representations, thereby reducing the efectiveness of the TPP stage in SARA. Second, it aims to preserve the task-level behavior of the original model without modifying the cloud-side ViT part. To achieve these objectives, we adopt a knowledge-distillation approach in which a student version of the original ViT is constructed by removing the positional embeddings from the edge-side model. Its transformer blocks are then progressively adapted using the original pretrained ViT as the teacher. The overall procedure is summarized in Algorithm 1 and described in detail below.

We use the original pretrained model as a frozen teacher, composed of the edge-side component $f _ { \mathrm { e } } ^ { ( k ) }$ and the cloud-side component $f _ { \mathrm { c } } ^ { ( k ) }$ . The student difers from the teacher only in its edge-side component, denoted by $\bar { f } _ { \mathrm { e } } ^ { ( k ) }$ . Before fine-tuning, $\dot { \bar { f } } _ { \mathrm { e } } ^ { ( k ) }$ is initialized with the same pretrained parameters as $f _ { \mathrm { e } } ^ { ( k ) }$ , but the positional embeddings stage is removed, as shown in the following. Omitting the class token for simplicity, the initial teacher and student representations are therefore given by

teacher : $h ^ { ( 0 ) } \gets E _ { \mathrm { p a t c h } } ( x ) + E _ { \mathrm { p o s } } ,$ student : $h ^ { ( 0 ) } \gets E _ { \mathrm { p a t c h } } ( x )$

where $E _ { \mathrm { p a t c h } }$ denotes the patch-embedding function and $E _ { \mathrm { p o s } }$ denotes the positional embeddings.

Although directly removing the positional embeddings eliminates the primary source of positional information, it may substantially alter the intermediate representations and degrade the predictive performance of the model. We therefore recover task performance by progressively adapting the edge-side transformer blocks through knowledge distillation using the logit Kullback– Leibler divergence loss $D _ { \mathrm { K L } }$ , which encourages the student edge to preserve features that allow achieving an output behavior of the original model. In particular, as shown by the outer loop in Algorithm 1, at the finetuning stage $j \in \{ 1 , \ldots , k \}$ , only the parameters of the �-th student block are optimized. Previously adapted blocks remain fixed, whereas subsequent edge-side blocks retain their original pretrained parameters. The entire cloud-side network also remains fixed throughout the entire procedure, although gradients are backpropagated through it to optimize the edge-side blocks.

Notably, this progressive block-wise adaptation provides a more stable optimization of the task objective than jointly fine-tuning the entire edge-side model. In the ablation study presented in Sections 5.3.3 and 5.3.4, we further validate this design choice and investigate the potential benefits and limitations of incorporating a min–max optimization strategy into the formulation.

## 5 Experimental Results

In this section, we first evaluate the efectiveness of the proposed attack and then assess the benefits of introducing the proposed defense against both token reduction and shufling. Before presenting the experimental results, we describe the experimental setup.

## 5.1 Experimental Settings

Models and datasets. We conduct all experiments on ImageNet-1K [4] using ViT-B/16 [6] and MAE-B/16 [10]. Although the two models share the same ViT architecture, they difer in their pretraining strategies: ViT-B is trained using supervised classification, whereas MAE-B relies on self-supervised masked autoencoding and requires downstream finetuning. We therefore evaluate the pretrained ViT-B directly on ImageNet, while adapting MAE-B by fine-tuning its classification head and final Transformer block for 30 epochs on the ImageNet-1K training set, with all preceding blocks kept frozen. The latter setup also allows us to explore a split inference scenario in which fixed client representations can support diferent server-side tasks. Furthermore, as shown in the following experiments, MAE-B is more vulnerable to reconstruction attacks than the supervised ViT-B, even at deeper split points, likely due to its reconstruction-oriented pretraining.

Token operations and metrics. We consider token shufling and two representative token reduction methods. Specifically, we evaluate ToMe [2], which progressively merges � tokens based on their similarity, and a random token dropping strategy [18, 26, 28], in which a fixed number � of tokens is discarded after each ViT block.

To evaluate image reconstruction quality, we employ several well-established vision metrics, namely the Structural Similarity Index Measure (SSIM) [36], Peak Signal-to-Noise Ratio (PSNR), and Feature Similarity Index Measure (FSIM) [41]. Higher SSIM, PSNR, and FSIM values indicate better reconstruction quality and, consequently, greater information leakage and weaker privacy protection.

Finally, particularly for token reduction methods, increasing the reduction parameter may improve privacy while causing a substantial degradation in classification accuracy. To jointly evaluate privacy and utility, we introduce the Privacy–Utility Reconstruction Index (PURI). Utility is measured as the classification accuracy relative to the unmodified baseline: $\begin{array} { r } { U = \frac { \mathrm { A c c } } { \mathrm { A c c } _ { \mathrm { b a s e } } } } \end{array}$ . Privacy is quantified as the relative degradation in reconstruction quality compared with the baseline attack, which applies no token operations and uses a convolutional decoder at the considered split point. Accordingly, the privacy score is defined as

$$
P = \mathrm { { m a x } \left( 0 , 1 - \frac { 1 } { 3 } \left( \frac { \mathrm { { S S I M } } } { \mathrm { { S S I M } _ { b a s e } } } + \frac { \mathrm { { P S N R } } } { \mathrm { { P S N R } _ { b a s e } } } + \frac { \mathrm { { F S I M } } } { \mathrm { { F S I M } _ { b a s e } } } \right) \right) . }\tag{7}
$$

Accordingly, $P = 0$ indicates no privacy improvement over the baseline, whereas larger values indicate a stronger degradation in reconstruction quality. Finally, PURI combines utility and privacy through a weighted harmonic mean:

$$
\mathrm { P U R I } _ { \lambda } = \frac { U \cdot P } { \lambda \cdot P + ( 1 - \lambda ) \cdot U } ,\tag{8}
$$

where $\lambda \in \left[ 0 , 1 \right]$ controls the relative importance assigned to utility and privacy. We set $\lambda = 0 . 7$ , thereby assigning greater importance to utility preservation, which reflects the practical requirement that a privacy-preserving method should maintain high classification accuracy while reducing reconstruction leakage.

Attack and Defense Setup. Regarding the setup of SARA, each component is trained considering $\mathcal { D } _ { \mathrm { a u x } }$ a randomly selected 25% subset of the ImageNet-1K training set to limit computational cost. The TPP and image decoder are trained for 20 epochs, whereas the MAE is trained for 10 epochs. We use a learning rate of $1 0 ^ { - 4 }$ for the TPP and MAE and $1 0 ^ { - 3 }$ for the image decoder.

Consistent with the discussion in Section 3.2, the TPP is trained on randomly shufled smashed representations generated using diferent random seeds. The MAE is trained on randomly masked smashed representations to reconstruct the missing tokens, whereas the decoder is trained on the original, unshufled and unreduced smashed representations, following the same setup as the baseline convolutional decoder. The architectures of all SARA components are detailed in Table 1. Their configurations were selected based on the best performance observed in preliminary experiments; for example, the TPP architecture was chosen according to the results reported in Figure 2). Regarding the confidence threshold � of the TPP introduced in Eq. 4, in both the attack and defense preliminary evaluation, we did not observe substantial diferences in reconstruction performance when varying �. Therefore, for simplicity, all reported experiments use $\tau = 0 . 0$ , and so every token is assigned to its most likely predicted position.

Table 1: Architectural setup of SARA’s components.
<table><tr><td>Component</td><td>Stage</td><td>Configuration</td></tr><tr><td rowspan="2">TPP</td><td>Projection</td><td>Linear 768→384</td></tr><tr><td>Encoder Head</td><td>Transformer Encoder ×2, d = 384 Linear 384→ 196 + Softmax</td></tr><tr><td rowspan="2">MAE</td><td>Input encoding</td><td>MAE positional embedding, d = 768</td></tr><tr><td>Encoder Head</td><td>Transformer Encoder ×4, d = 768 LayerNorm + Linear 768 → 768</td></tr><tr><td rowspan="2">Decoder</td><td>Reshape</td><td>196 × 768 → 768 × 14 × 14</td></tr><tr><td>Upsampling Output</td><td>ConvT: 768→ 256 → 128 → 64→32 Conv 32 →3</td></tr></table>

Regarding the setup used for the proposed defense, each transformer block is fine-tuned using the ImageNet-1K training set for 50 epochs with a learning rate of $1 0 ^ { - 4 }$ and temperature 4 in the $D _ { \mathrm { K I } }$ . Both attack and defense performance are evaluated on the complete ImageNet-1K validation set.

Every component of the SARA pipeline is trained independently for each split point. When a defense is applied, all components of SARA are retrained referring to the defended model.

## 5.2 Attack results

In this first part of the experiments, we evaluate SARA under token shufling and token reduction strategies. The main objective is to assess whether our attack remains capable of reconstructing the original input despite these token operations.

5.2.1 Shufling. As illustrated in the motivation presented in Section 3, inputs subjected to token shufling could not be reconstructed using the convolutional decoder. In contrast, as shown in Fig. 4, SARA can fully reconstruct the input image when token shufling is applied to both ViT and MAE models, achieving reconstruction quality comparable to that of the baseline across multiple split points. This result is expected because token shufling alters only the order of the tokens, while their original spatial positions can be reliably inferred even at very deep split points.

We also observe that the reconstruction quality of MAE-B/16 remains nearly constant as the split point moves to deeper blocks. This finding reinforces the observation made in the experimental setup regarding the importance of accounting for potential privacy leakage in models pretrained with reconstruction-based objectives.

Takeaway. Token shufling alone does not defend against SARA, since the original token order remains recoverable even at deep ViT split points.

![](images/a775e71e2dcf33db9409c15682bcd21acc3bf57bec8bd9694c4b76574215650e.jpg)  
(a) ViT-B/16

![](images/254d519a108d61bef0b614ce21b3a601a7283e963271c4d7e74f0abe599c7252.jpg)  
(b) MAE-B/16  
Figure 4: SSIM under token shufling. Baseline is a client with no shufling applied (reconstruction upper bound). Shufling + decoder applies the convolutional decoder directly to the shufled tokens. Shufling + SARA is our attack against shufled tokens.

5.2.2 Token reductions. We first evaluate how efectively SARA reconstructs inputs afected by ToMe and random token dropping under diferent layer-wise reduction amounts, denoted by �. Fig ure 5 reports the PURI scores, defined in Equation 8, for reduction amounts ranging from 5 to 90. For each curve, the value of � that maximizes the PURI score is highlighted. Figure 5 reveals a clear trend. At low reduction amounts, the transmitted representation remains close to the original one and also, especially for shallow split points, retains substantial information about the input. This helps preserve accuracy but provides limited privacy, resulting in a low PURI score. As the reduction amount increases, privacy improves, while accuracy gradually decreases. This degradation is generally less pronounced at shallower split points, where token reduction is applied across fewer transformer blocks. Importantly, optimal values of � at each split point identify the operating points that provide the best trade-of between downstream accuracy and privacy against reconstruction attacks.

We therefore use the optimal configuration as representative reduction settings to analyze reconstruction quality and downstream task performance in greater detail in Figure 6. In this analysis, the first row shows the efect of token reduction considering of each split point on downstream classification accuracy. As expected, selecting optimal setups for PURI allow to have a good balance of accuracy performance, even when addressing deeper split points (where optimal � is clearly lower than shallow ones, as shown in Figure 5), where in all the case stay close to the baseline value. The second row reports instead the reconstruction quality achieved by the SARA attack, measured using SSIM. For ViT-B/16, the attack results in only moderate reconstruction degradation, particularly at the final split points of the network. In contrast, for MAE-B/16, reconstruction quality remains remarkably stable across the network depth, with consistently high SSIM values comparable to those obtained at the second split point of ViT-B/16. A graphical illustration across all token operations addressed is shown in Figure 7. Regarding the comparison between the two token-reduction approaches, the considered metrics do not reveal a clear distinction, likely because they are agnostic to the most semantically relevant regions of the image.

![](images/12c49bec5fafcb450a0c76c11ee93e7821571ea2a15cc55f27bf88790b719022.jpg)

![](images/e6ad82124a84807d1d9d91917fdb8a445e304f4f00245cade9a0c0b90c0ed424.jpg)  
(b) MAE-B/16 (Dropping)

(a) ViT-B/16 (Dropping)  
![](images/ae565f24692bcc07edefbb47088700536947350599cda1aeb5d26021498838f0.jpg)  
(c) ViT-B/16 (ToMe)

![](images/08be86aacb4b97c8c947ae1c24d7ca683c1505d4a19683daf0562fbdee526eae.jpg)  
(d) MAE-B/16 (ToMe)

Figure 5: PURI as a function ofthe reduction amounts � for Random Dropping (top) and ToMe (bottom), on ViT-B/16 and MAE-B/16.  
![](images/838d1ed483ebb1515555089bbaed3768af89a1daf1196c3ba5c1abb5f178d4dc.jpg)  
(a) Accuracy - ViT-B/16

![](images/616c2d161accff6cebe92869dd4621568fcf63f19381dac1d839a37fe4a4f81f.jpg)  
(b) Accuracy - MAE-B/16

![](images/4a73c0fd5e609580952f1f612a2da29ed3ea360404b5c2f87b43938082dee48e.jpg)

![](images/d24525179d73cf85c10bba998c11d1b54afffdea163fe23a0a21025bab09a393.jpg)  
(c) SSIM - ViT-B/16  
(d) SSIM - MAE-B/16  
Figure 6: Classification accuracy (top row) and reconstruction quality (bottom row) at optimal �, on ViT-B/16 and MAE-B/16. The dash line refers to the accuracy without reductions.

Takeaway. Token reduction substantially degrades SARA’s reconstructions only at the later split points of ViT-B/16. In contrast, for MAE-B/16, the attack continues to produce reconstructions with an SSIM above 0.5 across all split depths.

![](images/03e2eda7f6e6de42c304b248e3122dbdca5db7ba4da61b15a849ada60aed2dae.jpg)  
Figure 7: Reconstructed images from intermediate representations at split point � = 4 for ViT-B/16 and MAE-B/16. For ToMe and Drop ping, we used the optimal � that maximizes PURI.  
Baseline (no shufling)

![](images/a8ae9461247f8154c3a0ea1d0f3e11e1479533924e9f40e80f6bf2dfe089fb22.jpg)  
(a) ViT-B/16

![](images/9d6ae719e4b86212e0e919ded75b57962743cde230d24d4e1de3a3cb8f6b15dc.jpg)  
(b) MAE-B/16  
Figure 8: Reconstruction quality (SSIM) under token shufling for models adapted using the proposed defense.

## 5.3 Defense results

In this part of the experimental evaluation, we assess the efectiveness of the proposed defense against the SARA attack.

5.3.1 Shufling. We first investigate whether combining shufling with our defense can be efective to mitigate SARA.

Figure 8 reports the reconstruction quality achieved by the attacker under this setting. For ViT-B/16, the proposed defense consistently reduces the attacker’s reconstruction quality across all split points, with the SSIM remaining below 0.4. This represents a substantial improvement over the baseline, indicating that attenuating the positional encoding significantly enhances the effectiveness of shufling. For MAE-B/16, the defense also degrades reconstruction quality, although the attack becomes more efective at deeper split points. This seemingly counterintuitive trend reflects the MAE training objective, which encourages the model to preserve spatial relationships for masked-patch reconstruction. Consequently, unlike ViT-B/16, whose deeper features become increasingly classification-specific, MAE-B/16 retains stronger spatial structure that the attacker can exploit even after positional information has been attenuated. We further study this problem and ad-hoc alternatives for the defense proposed in Section 4.

![](images/e645e722fe7345aa2560b1409cc9dd3e22c8d7e1653b5ad540f5dac59a31014b.jpg)  
(a) ViT-B/16 (Dropping)

![](images/5eaa545bde293252cf5eceac646294a498089c625f7f06435f9a4cbf35be46ef.jpg)

![](images/e8b3312796a7010834741dfb90248901e096b64836ccf8db9f776fbda6b41362.jpg)  
(c) ViT-B/16 (ToMe)

(b) MAE-B/16 (Dropping)  
![](images/8dabcee06026af989fab8eee8a3fd8d5b45b11989ef5a1923413c73f4d39a55c.jpg)  
(d) MAE-B/16 (ToMe)  
Figure 9: PURI as a function ofthe reduction amounts � for Random Dropping (top) and ToMe (bottom), on defended models.

5.3.2 Token reductions. For the token-reduction analysis, we first identify the optimal � at each split point according to the PURI metric, following the procedure described in the previous experiments. The results are reported in Figure 9, where, compared with the undefended model in Figure 5, the optimal value of � is consistently lower, and the PURI curves are already close to their maximum at the smallest tested reduction amounts. This indicates that attenuating positional information substantially improves privacy against SARA even before applying aggressive token reduction.

Figure 10 reports the classification accuracy and reconstruction quality obtained at the optimal � of each split point considered. The top panel also shows the accuracy loss introduced by the defense itself, with the dotted line indicating the performance ofthe original undefended model. This degradation is limited and is mainly attributable to the removal of positional embeddings from the client-side model. Considering the accuracy drop induced by token reductions, the overall accuracy degradation is smaller than without the defense, which is mainly because the optimal reduction amounts are lower, allowing more tokens to be retained while preserving a favorable privacy–utility trade-of.

As shown in the bottom panel of Figure 10, the reconstruction quality remains below an SSIM of 0.4 across all split points for both ViT-B/16 and MAE-B/16. The increasing SSIM trend previously observed for MAE-B/16 under the shufling-only setting remains visible but is substantially attenuated. This suggests that combining positional-information attenuation with token reduction limits the spatial information available in deeper MAE-B/16 representations, thereby reducing the efectiveness of the reconstruction attack. Illustrations of the defense benefits are shown in Figure 11.

<table><tr><td>k</td><td>PURI</td><td>k</td><td>PURI</td></tr><tr><td>2</td><td> $0 . 7 0 5  \mathbf { 0 . 7 1 5 }$ </td><td>2</td><td> $\mathbf { 0 . 7 1 5 } \longrightarrow 0 . 6 9 0$ </td></tr><tr><td>4</td><td> $0 . 6 0 1  \mathbf { 0 . 6 7 2 }$ </td><td>4</td><td> $0 . 6 0 5  \mathbf { 0 . 6 8 5 }$ </td></tr><tr><td>6</td><td> $\mathbf { 0 . 5 7 3  0 . 6 2 3 }$ </td><td>6</td><td> $0 . 5 4 2  \mathbf { 0 . 6 4 1 }$ </td></tr><tr><td>8</td><td> $\mathbf { 0 . 5 3 9 }  \mathbf { 0 . 5 7 9 }$ </td><td>8</td><td> $0 . 5 6 8 \to \mathbf { 0 . 6 3 8 }$ </td></tr><tr><td>10</td><td> $\mathbf { 0 . 4 8 8 \longrightarrow 0 . 5 1 8 }$ </td><td>10</td><td> $0 . 6 2 1 \longrightarrow 0 . 6 2 7$ </td></tr></table>

![](images/b708119de64cb8b28e90aa97a16f667f8a742f9e6d5de3a1b60230f62f26119a.jpg)  
(a) Accuracy - ViT-B/16

![](images/366c27b04a70cd6c54bd97bd4f1c243b9cb37e2158ce2645bec7c0caeb70cf7f.jpg)  
(b) Accuracy - MAE-B/16

![](images/b0975cd2bbdd1678de205e807182daecf34cf305face7f5141d80158c47490f8.jpg)

![](images/f967f5f6c7ce655ef55b6d19e38c862a2d2f91c93642cf202ec94732e227339c.jpg)  
(c) SSIM - ViT-B/16  
(d) SSIM - MAE-B/16

Figure 10: Accuracy (top row) and reconstruction quality (bottom row) at optimal �, on defended ViT-B/16 and MAE-B/16. The dash line refers to the accuracy without defense and reductions.  
![](images/db9d022ef2a3a02aa5ca55019ddfe219a56cf41926038d8ec137192dd470c113.jpg)  
Figure 11: Reconstructed images from intermediate representations at split point � = 4 for defended ViT-B/16 and MAE-B/16. Reconstructions for ToMe and Dropping are shown using the optimal �.

Takeaway. The proposed defense introduces only a small accuracy drop relative to the original model. Under token shuffling, the defense substantially reduces the attacker’s reconstruction capability across all split points, although for MAE-B/16 it slightly increases at deeper splits. When combined with token reduction, it consistently degrades reconstruction across all split points for both ViT-B/16 and MAE-B/16.

5.3.3 Benefits of progressive finetuning. As shown in Table 2, the progressive finetuning strategy in Algorithm 1 generally yields more stable results and a better privacy–utility trade-of than directly finetuning the entire edge-side model, even when both approaches use the same total number of training epochs. By updating one transformer block at a time, the optimization better preserves compatibility with the fixed cloud-side network and limits abrupt changes in the intermediate representations, resulting in improved performance across most split points.

Table 2: PURI scores on token shufling, with a simple PE-removal finetuning → and progressive PE-removal finetuning (Alg.1).  
(a) ViT-B/16  
(b) MAE-B/16  
Table 3: Each entry reports Defense → Defense+Adversarial. Lower SSIM and higher accuracy are better; the best values are in bold.
<table><tr><td></td><td colspan="2">ViT-B/16</td><td colspan="2">MAE-B/16</td></tr><tr><td>k</td><td>SSIM</td><td>Acc.</td><td>SSIM</td><td>Acc.</td></tr><tr><td>2</td><td> $\mathbf { 0 . 3 7 0 \longrightarrow 0 . 2 9 0 }$ </td><td> $7 8 . 3 \% \longrightarrow 7 1 . 9 \%$ </td><td> $\mathbf { 0 . 3 8 4 }  \mathbf { 0 . 3 5 5 }$ </td><td> $6 2 . 6 \%  6 3 . 7 \%$ </td></tr><tr><td>4</td><td> $\mathbf { 0 . 3 8 5  0 . 2 8 0 }$ </td><td> $8 0 . 9 \%  7 2 . 2 \%$ </td><td> $\mathbf { 0 . 4 1 9  0 . 3 7 3 }$ </td><td> $6 6 . 6 \% \longrightarrow 6 7 . 4 \%$ </td></tr><tr><td>6</td><td> $\mathbf { 0 . 3 7 8 \longrightarrow 0 . 2 8 2 }$ </td><td> $8 1 . 4 \%  7 0 . 5 \%$ </td><td> $\mathbf { 0 . 4 9 8 \longrightarrow 0 . 3 9 0 }$ </td><td> $7 0 . 7 \% \longrightarrow 7 0 . 1 \%$ </td></tr><tr><td>8</td><td> $\mathbf { 0 . 3 5 5 }  \mathbf { 0 . 2 1 1 }$ </td><td> $8 1 . 3 \% \longrightarrow 6 8 . 4 \%$ </td><td> $0 . 5 0 1  \mathbf { 0 . 3 7 7 }$ </td><td> $7 2 . 0 \%  7 1 . 0 \%$ </td></tr></table>

5.3.4 On Deep Layers of MAE. The previous analysis highlighted a limitation of the proposed defense for MAE-B/16. Under token shufling, the reconstruction quality may increase at deeper split points because MAE representations retain spatial relationships even after removing PE. To address this limitation, we investigate an adversarial variant of the defense in Algorithm 1. In addition to progressively adapting the edge-side transformer blocks without positional embeddings, we explicitly discourage the resulting smashed representations from revealing token positions. To this end, we introduce an auxiliary TPP, denoted by $C _ { \mathrm { p o s } } ,$ which is trained to recover the original patch indices, while the student client is trained to hinder its predictions. Let $\bar { \theta } _ { \mathrm { e } }$ and $\theta _ { \mathrm { p o s } }$ denote the parameters of the student edge $\bar { f } _ { e }$ and the auxiliary TPP, respectively. We replace the distillation-only objective in Algorithm 1 with the following adversarial objective, while the pipeline and progressive finetuning are the same:

$$
\operatorname* { m i n } _ { \bar { \theta } _ { \mathrm { e } } } \operatorname* { m a x } _ { \theta _ { \mathrm { p o s } } } \mathbb { E } _ { \mathcal { B } \subset \mathcal { D } _ { \mathrm { A } } } \left[ \mathcal { L } _ { \mathrm { K D } } ( \mathcal { B } ) - \lambda _ { \mathrm { p o s } } \mathcal { L } _ { \mathrm { T P P } } ( \mathcal { B } ) \right] ,\tag{9}
$$

where ${ \mathcal { L } } _ { \mathrm { K D } }$ preserves compatibility with the fixed server-side model (same as used in line 6 of $\mathrm { A l g . 1 } )$ , L<sub>TPP</sub> is the cross-entropy loss for token-position classification, and $\lambda _ { \mathrm { p o s } }$ controls the trade-of between task preservation and positional-information suppression.

As shown in Table 3, adversarial training efectively addresses this issue by stabilizing the attacker’s reconstruction capability across all ViT-MAE split points. In the proposed defense without adversarial training, the reconstruction quality under token shuffling, measured by SSIM, increases from 0.384 at split point � = 2 to 0.501 at split point � = 8, corresponding to an increase of 0.117. In contrast, with adversarial training, the SSIM increases only from 0.355 to 0.377. As a qualitative reference, we select split point � = 6 to visualize the efect of adversarial training on the attacker’s reconstruction quality. As shown in Fig. 12, the reconstruction obtained against the adversarially trained defense is noticeably less recognizable than that obtained with the standard defense.

![](images/375836874ba4c40b4f007e8a65765ec89df6bbdffebbe1fb72c33d1b7eb43d2a.jpg)  
Figure 12: Reconstructed image in MAE-B/16 under shufling. Comparison with baseline, defense w/o and w/ adversarial training.

Despite these benefits, the adversarial formulation can introduce a non-negligible accuracy drop, particularly for ViT-B/16, due to the instability of the min–max optimization. We also omit the results for � = 10 from Table 3, as training at this last split point exhibited stability issues. For this reason, we retain the simpler progressive finetuning strategy (Alg. 1) as our primary defense. A more robust integration of an adversarial regularization is left for future work.

## 6 Related work

Feature inversion attacks and defenses. Although split inference keeps the raw input on the edge device, the intermediate representations transmitted to the cloud may still leak substantial information about the private input. This leakage can be exploited through model inversion attacks (MIAs) [39], more specifically referred to as feature inversion attacks (FIAs) in split-inference settings [11, 12, 19, 29, 32]. These attacks typically train a reconstruction model to map intermediate features back to the input space, often relying on convolutional decoders that exploit local spatial relationships and assume a fixed correspondence between feature locations and image regions. This assumption, however, is violated when ViT tokens are shufled or reduced, making conventional FIAs poorly suited to evaluating manipulated token representations.

Existing defenses can broadly be grouped into cryptographic data-modification, and learned approaches [17, 38]. Cryptographic methods, including homomorphic encryption [7], secure multiparty computation [25], and function secret sharing [3], provide strong guarantees but generally incur substantial computational and com munication overhead [13, 16, 20]. Trusted Execution Environments ofer a hardware-assisted alternative [31, 35]. Data-modification methods protect transmitted representations through mechanisms such as diferential privacy and quantization [8, 23], often introducing a privacy–utility trade-of. Learned approaches instead optimize intermediate representations to preserve task-relevant information while suppressing sensitive content [5, 14, 24], but may require costly or unstable adversarial training. In contrast, our proposed defense operates only on the client side without introducing unstable learning trends and leaves the cloud-side model unchanged.

Token-based privacy mechanisms. Transformer architectures enable lightweight manipulations of intermediate representations at the token level. Token shufling alters the order of transmitted tokens by exploiting the permutation-related properties of Transformer blocks [37] and has been investigated as a privacyenhancing mechanism to hinder input reconstruction while preserving inference performance [40]. Token reduction instead discards or merges tokens and has been primarily studied as an eficiency mechanism, with only limited analyses of its privacy implications [9]. Related works have also explored the removal or anonymization of selected image regions [1] and the application of established defenses, such as diferential privacy, to intermediate representations [34]. Unlike cryptographic or learned defenses, token shufling and reduction can potentially be applied without retraining or modifying the cloud-side model. However, their low overhead does not necessarily guarantee privacy.

## 7 Conclusions and limitations

In this work, we presented a comprehensive analysis of token reduction and token shufling from a privacy perspective. We also proposed SARA, a novel attack pipeline for evaluating the privacy of these approaches. Our analysis showed that token reduction provides a certain degree of feature obfuscation. However, for token shufling and particularly in the case of MAE-B/16, SARA was still able to produce high-quality reconstructions, highlighting the limitations of these techniques as standalone privacy-preserving mechanisms. Motivated by these findings, we proposed a defense mechanism that attenuates positional information through a progressive fine-tuning strategy aimed at removing the positional encoding on the client side. Our experimental results demonstrate that this defense substantially improves the obfuscation of the intermediate features, reducing the efectiveness of FIAs.

Despite the promising results, we acknowledge several limitations. First, we evaluate utility only in terms of downstream classification accuracy, although the proposed fine-tuning strategy is task-agnostic. Extending the evaluation to other computer vision tasks, such as semantic segmentation, is therefore left for future work. Second, investigating whether SARA and the proposed defense can be efectively adapted to other domains, particularly language models, represents another promising direction. Finally, more sophisticated token-reordering strategies could be explored to achieve a better privacy-utility trade-of and to assess their robustness against adaptive reconstruction attacks.

## References

[1] Nazia Aslam, Abhisek Ray, Joakim Bruslund Haurum, Lukas Esterle, and Kamal Nasrollahi. 2026. From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition. arXiv:2603.26336 [cs.CV]

[2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hofman. 2023. Token Merging: Your ViT But Faster. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 23217–23226.

[3] Elette Boyle, Niv Gilboa, and Yuval Ishai. 2015. Function Secret Sharing. In Advances in Cryptology – EUROCRYPT 2015 (Lecture Notes in Computer Science, Vol. 9057). Springer, 337–367. doi:10.1007/978-3-662-46803-6\_12

[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 248–255. doi:10.1109/CVPR.2009.5206848

[5] Shiwei Ding, Lan Zhang, Miao Pan, and Xiaoyong Yuan. 2024. PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV). IEEE, 4704–4713. doi:10.1109/WACV57701.2024.00465

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale. In Proc. ofthe International Conference on Learning Representations (ICLR).

[7] Nathan Dowlin, Ran Gilad-Bachrach, Kim Laine, Kristin Lauter, Michael Naehrig, and John Wernsing. 2016. CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy. In Proceedings ofthe 33rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 48). PMLR, 201–210.

[8] Cynthia Dwork. 2006. Diferential Privacy. In Automata, Languages and Programming, Michele Bugliesi, Bart Preneel, Vladimiro Sassone, and Ingo Wegener (Eds.). Lecture Notes in Computer Science, Vol. 4052. Springer, Berlin, Heidelberg, 1–12. doi:10.1007/11787006\_1

[9] Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis, and Sami Muhai dat. 2025. Adaptive Token Merging for Eficient Transformer Semantic Commu nication at the Edge. arXiv:2509.09955

[10] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. Masked Autoencoders Are Scalable Vision Learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 16079– 16088. doi:10.1109/CVPR52688.2022.01553

[11] Zecheng He, Tianwei Zhang, and Ruby B. Lee. 2019. Model Inversion Attacks Against Collaborative Inference. In Proceedings of the 35th Annual Computer Security Applications Conference (ACSAC). Association for Computing Machinery, 148–162. doi:10.1145/3359789.3359824

[12] Zecheng He, Tianwei Zhang, and Ruby B. Lee. 2021. Attacking and Protecting Data Privacy in Edge–Cloud Collaborative Inference Systems. IEEE Internet of Things Journal 8, 12 (2021), 9706–9716. doi:10.1109/JIOT.2020.3022358

[13] Israt Jarin and Birhanu Eshete. 2021. PRICURE: Privacy-Preserving Collaborative Inference in a Multi-Party Setting. In Proceedings ofthe 2021 ACM Workshop on Security and Privacy Analytics. Association for Computing Machinery, 25–35. doi:10.1145/3445970.3451156

[14] Jonghu Jeong, Minyong Cho, Philipp Benz, and Tae-hoon Kim. 2023. Noisy Adversarial Representation Learning for Efective and Eficient Image Obfuscation. In Proceedings ofthe Thirty-Ninth Conference on Uncertainty in Artificial Intelligence (Proceedings ofMachine Learning Research, Vol. 216), Robin J. Evans and Ilya Shpitser (Eds.). PMLR, 953–962.

[15] Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, and Lingjia Tang. 2017. Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. In Proceedings ofthe Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). Association for Computing Machinery, New York, NY, USA, 615–629. doi:10.1145/3037697.3037698

[16] Tanveer Khan, Mindaugas Budzys, and Antonis Michalas. 2024. Make Split, Not Hijack: Preventing Feature-Space Hijacking Attacks in Split Learning. In Proceedings ofthe 29th ACM Symposium on Access Control Models and Technologies (SACMAT). Association for Computing Machinery, 19–30. doi:10.1145/3649158. 3657039

[17] Tanveer Khan and Antonis Michalas. 2026. Oops!... They Stole It Again: Attacks on Split Learning. In Proceedings ofthe 18th ACM Workshop on Artificial Intelligence and Security (AISec ’25). Association for Computing Machinery, 123–135. doi:10.1145/3733799.3762972

[18] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. 2024. Token Fusion: Bridging the Gap between Token Pruning and Token Merging. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). doi:10.1109/WACV57701.2024.00141

[19] Wa-Kin Lei, Jun-Cheng Chen, and Shang-Tse Chen. 2025. DRAG: Data Reconstruction Attack Using Guided Difusion. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML). Article 1325.

[20] Dong Li, Anupam Chattopadhyay, Qianyu Li, Qingguo Lü, Jiahui Wu, Tao Xiang, and Xiaofeng Liao. 2026. CryptDNN: A Fast Privacy-Preserving Deep Neural Network Inference Architecture Based on Cloud–Edge–Client Collaboration. IEEE Transactions on Network Science and Engineering 13 (2026), 2726–2740. doi:10. 1109/TNSE.2025.3620373

[21] Zhengyi Li, Yakai Wang, Kang Yang, Yu Yu, Jiaping Gui, Yu Feng, Ning Liu, Minyi Guo, and Jingwen Leng. 2026. On the (In-)Security of the Shufling Defense in the Transformer Secure Inference. arXiv preprint arXiv:2605.04901 (2026).

[22] Yang Liu, Yao Zhang, Yixin Wang, Feng Hou, Jin Yuan, Jiang Tian, Yang Zhang, Zhongchao Shi, Jianping Fan, and Zhiqiang He. 2022. A Survey of Visual Trans formers. arXiv preprint arXiv:2111.06091 (2022).

[23] Yuzhe Luo, Zhen Zhang, Jiacheng Li, Yu Wang, and Yao Chen. 2024. Privacy-Preserving Compression for Eficient Collaborative Inference. In Proceedings of the IEEE 30th International Conference on Parallel and Distributed Systems (ICPADS). IEEE, 142–151. doi:10.1109/ICPADS63350.2024.00028

[24] Fatemehsadat Mireshghallah, Mohammadkazem Taram, Ali Jalali, Ahmed Taha Elthakeb, Dean M. Tullsen, and Hadi Esmaeilzadeh. 2021. Not All Features Are Equal: Discovering Essential Features for Preserving Prediction Privacy. In Proceedings of the Web Conference 2021 (WWW). Association for Computing Machinery, 669–680. doi:10.1145/3442381.3449965

[25] Payman Mohassel and Yupeng Zhang. 2017. SecureML: A System for Scalable Privacy-Preserving Machine Learning. In Proceedings ofthe IEEE Symposium on Security and Privacy (SP). IEEE, San Jose, CA, USA, 19–38. doi:10.1109/SP.2017.12

[26] Lorenzo Papa, Paolo Russo, Irene Amerini, and Luping Zhou. 2024. A Survey on Eficient Vision Transformers: Algorithms, Techniques, and Performance Benchmarking. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 12 (Dec. 2024), 7682–7700. doi:10.1109/TPAMI.2024.3392941

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[28] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. 2021. DynamicViT: Eficient Vision Transformers with Dynamic Token Sparsification. In Advances in Neural Information Processing Systems (NeurIPS).

[29] Maria Rigaki and Sebastian Garcia. 2024. A Survey of Privacy Attacks in Machine Learning. Comput. Surveys 56, 4, Article 101 (2024). doi:10.1145/3624010

[30] Giulio Rossolini, Fabio Brau, Alessandro Biondi, Battista Biggio, and Giorgio Buttazzo. 2025. Exploiting edge features for transferable adversarial attacks in distributed machine learning. Internet ofThings 34 (2025), 101795.

[31] Mohamed Sabt, Mohammed Achemlal, and Abdelmadjid Bouabdallah. 2015. Trusted Execution Environment: What It Is, and What It Is Not. In Proceedings of the IEEE Trustcom/BigDataSE/ISPA, Vol. 1. IEEE, 57–64.

[32] Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, and Santiago Zanella-Béguelin. 2023. SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning. arXiv preprint arXiv:2212.10986 (2023).

[33] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. 2025. Dinov3. arXiv preprint arXiv:2508.10104 (2025).

[34] Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, and Ramesh Raskar. 2018. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv:1812.00564 [cs.LG] https://arxiv.org/abs/1812.00564

[35] Yulong Wang and Ahmed Habib. 2025. Protect Data Confidentiality for On-Device Machine Learning Through Split Inference. In Proceedings ofthe 10th International Conference on Fog and Mobile Edge Computing (FMEC). IEEE, 290– 297. doi:10.1109/FMEC65595.2025.11119383

[36] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 13, 4 (April 2004), 600–612.

[37] Hengyuan Xu, Liyao Xiang, Hangyu Ye, Dixi Yao, Pengzhi Chu, and Baochun Li. 2024. Permutation Equivariance of Transformers and Its Applications. arXiv preprint arXiv:2304.07735 (2024). arXiv:2304.07735 [cs.CR]

[38] Mengda Yang, Ziang Li, Juan Wang, Hongxin Hu, Ao Ren, Xiaoyang Xu, and Wenzhe Yi. 2022. Measuring Data Reconstruction Defenses in Collaborative Inference Systems. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 35. Article 934.

[39] Ziqi Yang, Bin Shao, Bohan Xuan, Ee-Chien Chang, and Fan Zhang. 2020. Defending Model Inversion and Membership Inference Attacks via Prediction Purification. arXiv preprint arXiv:2005.03915 (2020).

[40] Dixi Yao, Liyao Xiang, Hengyuan Xu, Hangyu Ye, and Yingqi Chen. 2022. Privacy-Preserving Split Learning via Patch Shufling over Transformers. In Proceedings ofthe IEEE International Conference on Data Mining (ICDM). IEEE, Orlando, FL, USA, 638–647. doi:10.1109/ICDM54844.2022.00074

[41] Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang. 2011. FSIM: A Feature Similarity Index for Image Quality Assessment. IEEE Transactions on Image Processing 20, 8 (Aug. 2011), 2378–2386. doi:10.1109/TIP.2011.2109730

## Declaration on the Use of Generative AI

During the preparation of this manuscript, the authors used generative AI tools (ChatGPT-5.3 Instant and Claude Opus 4.8) exclusively to assist with grammar correction, language refinement and code generation.

![](images/4d7ed53dd372fdbbc22e5908c580d19ebbb585f92759a6f8b707076c16342429.jpg)

(a) ViT-B/16  
![](images/667bc74f6f5f4d1a095dade0d293d300d797cfa2f30772434d349f266a26488a.jpg)  
(b) MAE-B/16

Figure 13: Reconstructed images with SARA for baseline (noshufling), shufling, and token reductions, across diferent split points �, without our defense.  
![](images/26ebfb04e225593198366c846f70e4993fead396952080079563a1ea41ae48c2.jpg)

(a) ViT-B/16  
![](images/289a09bd65de1358cf348627d41501872afe1308881637c857ff23f3f12b41ae.jpg)  
(b) MAE-B/16  
Figure 14: Reconstructed images with SARA for baseline (noshufling), shufling, and token reductions, across diferent split points �, with our defense.

Table 4: Optimal reduction parameter � and corresponding number of transmitted tokens for the two architectures at � = 0.7. Each entry reports � $/ T ,$ where� is the number of transmitted tokens. (a) Results without defense. (b) Results with proposed defense.

(a) Without Defense.
<table><tr><td rowspan="2">k</td><td colspan="2">ViT-B/16</td><td colspan="2">MAE-B/16</td></tr><tr><td>| Dropping</td><td>ToMe</td><td>| Dropping</td><td>ToMe</td></tr><tr><td>2</td><td>70 / 57</td><td>65/ 67</td><td>75 / 47</td><td>65/ 67</td></tr><tr><td>4</td><td>40 / 37</td><td>40 / 37</td><td>40 / 37</td><td>35 / 57</td></tr><tr><td>6</td><td>25 / 47</td><td>30 / 17</td><td>25 / 47</td><td>25 / 47</td></tr><tr><td>8</td><td>20 /37</td><td>20 / 37</td><td>20 / 37</td><td>20 / 37</td></tr><tr><td>10</td><td>15 / 47</td><td>20 / 1</td><td>15 / 47</td><td>15 / 47</td></tr></table>

(b) With defense.
<table><tr><td rowspan="2">k</td><td colspan="2">ViT-B/16</td><td colspan="2">MAE-B/16</td></tr><tr><td>|Dropping</td><td>ToMe</td><td>Dropping</td><td>ToMe</td></tr><tr><td>2</td><td>5 / 187</td><td>25 / 147</td><td>20 / 157</td><td>20 / 157</td></tr><tr><td>4</td><td>10 / 157</td><td>20 / 117</td><td>10 / 157</td><td>15 / 137</td></tr><tr><td>6</td><td>15 / 107</td><td>15 / 107</td><td>10 / 137</td><td>15 / 107</td></tr><tr><td>8</td><td>10 / 117</td><td>15 / 77</td><td>10 / 117</td><td>10 / 117</td></tr><tr><td>10</td><td>5 /147</td><td>10 / 97</td><td>10 / 97</td><td>10 / 97</td></tr></table>

## A Appendix

## A.1 Optimal ratios with PURI

Table 4 reports the optimal values of �, within the considered range, for the two token-reduction techniques at diferent split points. We also report the corresponding number of transmitted tokens. For a fixed split point, increasing � reduces the number of tokens transmitted from the edge to the cloud, thereby lowering communication and cloud-side computational costs, but potentially degrading task accuracy. As shown in Table 4, the PURI-optimal value of� changes substantially after fine-tuning with the proposed defense. This does not imply that users should simply adopt a smaller � to improve privacy. Rather, the defense shifts the privacy-utility trade-of, allowing lower reduction amounts, and thus better accuracy, while still achieving favorable privacy scores. This behavior is also evident from the comparison between the PURI curves obtained with the proposed defense and those reported in the original attack analysis.

## A.2 Additional Illustrations.

In addition to the examples presented in the main body, we provide further reconstructions in Figures 14 and 13 obtained at the PURIoptimal values of� for both the original and defended models across diferent split points. Importantly, these reconstructions correspond specifically to the PURI-optimal reduction amounts. Using a lower � would allow SARA to recover higher-quality images, approaching the reconstruction quality observed under token shufling. Conversely, a higher � would make reconstruction more dificult but would also cause a greater degradation in task accuracy.