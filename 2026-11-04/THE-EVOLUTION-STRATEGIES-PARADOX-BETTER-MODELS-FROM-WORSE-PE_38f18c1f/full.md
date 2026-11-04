# THE EVOLUTION STRATEGIES PARADOX: BETTER MODELS FROM WORSE PERTURBATIONS

Roberto Dailey<sup>1∗</sup>, Kajetan Schweighofer<sup>1</sup>, Jamieson Warner<sup>1</sup>, Conor F. Hayes<sup>1</sup>, Elliot Meyerson<sup>1</sup>, Babak Hodjat<sup>1</sup>, Risto Miikkulainen<sup>1,2</sup>, Xin Qiu<sup>1</sup>

<sup>1</sup>Cognizant AI Lab, San Francisco <sup>2</sup>The University of Texas at Austin, Austin

## ABSTRACT

How is it possible for perturbation-based methods like Evolution Strategies (ES) to optimize billion-parameter large language models (LLMs)? Classical search methods are highly dependent on finding good candidates within generated populations. However, when using ES to fine-tune LLMs, a phenomenon can occur where all offspring models consistently perform worse than the parent model, yet during training, the parent model consistently improves. This paradox occurs across models, tasks, and even modalities. This paper characterizes the paradox, then examines how the local reward landscape and properties of the ES estimator contribute to it: in high-dimensional spaces, exploration leads to worse performance, yet still highlights good directions; aggregating those directions then allows for making local improvements. The result can be seen as a new optimization paradigm where a principled interaction of noise and curvature produces the paradox as a byproduct of effective optimization. These findings provide guidance for taking advantage of perturbation-based methods in LLM fine-tuning and in high-dimensional optimization in general.

## 1 INTRODUCTION

Optimization using population-based search, i.e. evolutionary computation, relies on generating and selecting high-quality candidates. Genetic algorithms optimize by discovering beneficial recombinations and mutations along with selection of candidates with higher fitness (Goldberg, 1989). Particle swarm optimization pressures particles to search for better positions while maintaining the previously best found positions (Eberhart & Kennedy, 1995). The original evolution strategies (ES) compared a single parent to a single offspring and selected the better solution (Rechenberg, 1965). In each of these methods, the generation of higher-fitness candidate solutions is crucial to the optimization process.

For ES specifically, this story has changed over time. With the introduction of natural evolution strategies (NES, Wierstra et al., 2011), which updates the parent model using a weighted sum of all generated candidates, ES was shown to potentially behave as a gradient approximator instead of working through the selection of candidates with an improved fitness. NES was a precursor to implementation of ES on large-scale reinforcement learning (RL) problems such as Atari (Salimans et al., 2017) that then inspired theoretical analysis that found that ES with isotropic Gaussian noise is a gradient estimate: with high enough population sizes (ranging from thousands to hundreds of thousands), correlation between the ES update direction and network gradient stays consistent while ES performance on a per-iteration basis would match that of backpropagation (Zhang et al., 2017). This along with further work concluded that ES operated as a gradient estimator that naturally favors robustness (Lehman et al., 2018).

Recently, a new opportunity for ES has emerged: scaling to the vastly larger optimization problem of fine-tuning Large Language Models (LLMs; Qiu et al., 2026). This work shifted from centered rank rewards to z-score normalized rewards, but more surprisingly, it brought the population size down from thousands to just 30 (Qiu et al., 2026; Hayes et al., 2026). With such small populations in a space of billions of parameters, accurate gradient estimation appears infeasible. It was furthe demonstrated that large pretrained models have a rich search landscape with many directions for improvement (Liang et al., 2026; Gan & Isola, 2026).

![](images/6c869bbb914d012f4c98ee2cfd996dab538f5a84db15d764ca4bfc4075e0b72f.jpg)  
Figure 1: The Evolution Strategies Paradox across three examples: OLMo 2 7B fine-tuned on Sudoku, Qwen 2.5 7B fine-tuned on Countdown, and a 60K parameter CNN on MNIST. Left plots show the fraction of offspring models better than the parent model; right plots show the training curves (i.e. the performance of the parent models over time). In each case, offspring models perform worse than the parent model for nearly all of training, while the parent models consistently improve.

![](images/85f2f1b9254c2ddf3c90febfa5f8bab0c68e2d4642e35b08b04cfe0cba902b13.jpg)  
(a) Visual representation of ES Paradox

![](images/12e6db0b4246280e0a01e9a164b2ea7ad216ddde4354aec9ef6e8ecab5ac9731.jpg)  
(b) Fraction of population members outperforming parent at various batch sizes.  
Figure 2: Tracking the ES paradox: (a) A visual representation of the ES paradox over iterations in a training run. At each iteration, offspring are generated which are all worse than the parent model in measured reward. (b) Fraction of perturbations better than the parent in the Sudoku fine-tuning task over iterations in the training run. Larger batch size leads to less noise in the reward landscape, and makes the paradox more prominent.

Taken together, these results suggest that in LLM-scale spaces, ES is not based on gradient estimation, but relies on finding good candidates among sampled perturbations and moving towards them, as other evolutionary methods do.

However, the results in this paper contradict that hypothesis as well: ES is found to make consistent progress even when all offspring candidates are worse than the parent (Figure 1). This paradox is observed across a range of ES optimization tasks, suggesting that it is a general property of highdimensional ES optimization. While it had been hypothesized possible for individual steps with alternative forms of ES under idealized conditions (Ba et al., 2026), this work is the first empirical demonstration of this occurring. A further analysis reveals that this phenomenon results from three mechanisms (1) Consistent curvature of the search landscape results in decreased performance with increased perturbation size; (2) relative performance between offspring still differentiates good and bad directions in parameter space; and (3) the ES preserves the signal of good directions while reducing noise in the aggregated update (Figure 2a).

The remainder of the paper is as follows: Section 2 details the ES update and the experimental setup in LLM fine-tuning under which the paradox is observed. The next three sections then focus on each of the three mechanisms. Section 3 runs a landscape analysis, finding that increased perturbation size is strongly tied to a decrease in reward. Section 4 demonstrates that even at larger perturbations where all offspring are worse than the parent, good directions can still be identified. Section 5 uses an analytical model of ES to show that the updates aggregate those good directions while reducing noise, making the paradox likely. These findings thus demonstrate a previously unknown mechanism, indeed a paradigm shift in optimization, that allows scaling perturbation-based optimization to modern LLMs.

## 2 THE EVOLUTION STRATEGIES PARADOX

This work uses ES as implemented by Qiu et al. (2026), which is characterized by isotropic Gaussian perturbations, low population size, and z-score normalized rewards. For each ES iteration $t ,$ let a parent model be parameterized by current parameters ${ \theta } _ { t } \in \mathbb { R } ^ { d }$ . A set of N perturbation vectors, indexed $i \in \{ 1 , 2 , . . . N \}$ is drawn from isotropic gaussian noise, $\epsilon _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ . Each perturbation is then multiplied by perturbation scale $\sigma \in \mathbb { R } ^ { + }$ , and added to $\theta _ { t }$ to create a population of offspring models, parameterized by $\theta _ { t } + \sigma \epsilon _ { i }$ . Each offspring model is evaluated to obtain a reward $R _ { i } ,$ , and the ES update is then the average of the perturbations, weighted by the z-score of the model rewards:

$$
\Delta \theta = \theta _ { t + 1 } - \theta _ { t } = \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \epsilon _ { i } , Z _ { i } = \frac { R _ { i } - \mu _ { R } } { \sigma _ { R } } ,\tag{1}
$$

where $\alpha$ is the step size, $\mu _ { R }$ and $\sigma _ { R }$ are the mean and standard deviation of this population’s rewards, and weights $Z _ { i }$ are the z-score normalized rewards. These updates are iterated until a stopping criterion, as shown in the basic ES training Algorithm 1. For an iteration t, let $A _ { t }$ denote the event that all offspring at iteration t perform worse than their parent. The ES paradox occurs when the update still improves the parent model in expectation under this event:

$$
A _ { t } = \Big \{ R \big ( \theta _ { t } + \sigma \epsilon _ { i } \big ) < R ( \theta _ { t } ) \ \forall i \in \{ 1 , \dots , N \} \Big \} , \qquad \mathbb { E } \big [ R ( \theta _ { t + 1 } ) - R ( \theta _ { t } ) \big | A _ { t } \big ] > 0 .\tag{2}
$$

In training runs, the paradox is tracked by recording the fraction of perturbations with rewards better than the current parent on the training reward. When this metric stays at or close to zero for extended periods of time while reward increases, it is assumed the paradox is occurring. In this paper, the performance on training data is treated as the full reward landscape; however, the effect can still occur when measured with a held-out reward. Monitoring this parent reward adds a marginal inference cost to ES training runs, as one additional model evaluation must be run for each sampled population. Figure 1 displays this metric alongside training progress in fine-tuning in the three example tasks. Each training run starts with significant fractions of perturbations outperforming the parent model, but partway into training, this metric falls to zero. The effect is most prominent with a fixed batch, though it can occur in shuffled batches; all else equal larger batch sizes make the effect stronger, as shown in Figure 2b.

This paper’s analysis focuses on LLM fine-tuning tasks, where it was first discovered. Two examples of the paradox in LLM fine-tuning are more carefully analyzed: a 500-iteration training run on a 4×4 Sudoku task, and a 500-iteration training run on a countdown task (Pan et al., 2025), using harnesses and grading used in Zhao et al. (2025). In both these examples, Qwen 2.5 7B-Instruct was used as the LLM, evaluated at temperature zero; ES was run with hyperparameters $( \mathrm { N } = 3 0 , \sigma = 0 . 0 0 1 , \alpha =$ $5 \times 1 0 ^ { - 4 } )$ . Fine-tuning was done with a dataset of 800 problems, with the batch size set to the entire dataset to demonstrate a strong effect (Figure 2b). Although the analysis focused on Qwen 2.5 with these two specific tasks, the paradox has been observed in other models as well, such as OLMo 2 7B, as shown in Figure 1.

In addition to LLM fine-tuning, the ES paradox occurs in other domains of ES optimization of neural networks. One such domain is included in this paper: training of a randomly initialized 60,000-parameter Convolutional Neural Network (CNN) on MNIST. The paradox was reproduced with ES parameters $\Nu = 1 6 , \sigma = 0 . 0 4$ , and $\alpha = 0 . 0 2$ . Thus, the paradox can be seen as a general phenomenon across models, tasks, and modalities. The next three sections analyze the mechanisms that produce it and demonstrate that it is a side effect of effective optimization in high-dimensional search spaces.

Algorithm 1 Basic ES Algorithm   
Require: Pretrained LLM with initial parameters $\theta _ { 0 }$ , reward function R(·), total iterations $T ,$ pop  
ulation size $N$ , noise scale $\sigma ,$ learning rate α.   
1: for $t = 1$ to T do ▷ outer ES iterations over generations of populations   
2: for $i = 1$ to $N$ do ▷ inner ES iteration over population members   
3: Sample noise $\varepsilon _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$   
4: Compute reward for perturbed parameters:   
$R _ { i } \gets R ( \theta _ { t - 1 } + \sigma \cdot \pmb \varepsilon _ { i } )$   
5: end for   
6: $\mathrm { _ { Z } }$ -score normalization of rewards   
$Z _ { i } \gets \frac { R _ { i } - \mu _ { R } } { \sigma _ { r } }$   
7: Update model parameters as   
$\begin{array} { r } { \pmb { \theta _ { t } }  \pmb { \theta _ { t - 1 } } + \alpha \cdot \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \pmb { \varepsilon _ { i } } } \end{array}$   
8: end for

## 3 WHY ARE OFFSPRING WORSE THAN THE PARENT?

As a first step to understand how the ES paradox emerges, this section examines the relationship between perturbation size and reward in fine-tuning tasks. For both Sudoku and Countdown fine tuning tasks, checkpoints from every 100 iterations were taken, and 300 offspring were generated at varying perturbation magnitudes. These perturbed models were then evaluated on the original fixed training set of 800 problems, producing a distribution of model rewards at each magnitude. The magnitudes were chosen to match the average magnitude of perturbations with $\sigma = 1 0 ^ { - 4 }$ and σ from $2 . 5 \times 1 0 ^ { - 4 } ~ \mathrm { t o } ~ 2 \times 1 0 ^ { - 3 }$ in steps of $2 . 5 \times 1 0 ^ { - 4 }$ . Figure 3 displays these tests at the base model and at iterations 100 and 500 in the training run, with box-and-whisker plots displayed at each perturbation magnitude.

With the base model, the mean reward is largely constant, but its variation increases approximately linearly as perturbation size increases. By iteration 100, perturbation magnitude has a significant, consistent, negative effect on the reward. By iteration 500, this effect is even more consistent and pronounced. In Sudoku in particular, the rewards at magnitude matching the σ used for training (1e-3) are multiple standard deviations below those of the parent model. Therefore, the perturbation reward distribution strongly depends on the perturbation size (σ). Small $\sigma { } ^ { \prime } \mathrm { s }$ produce valid offspring for model improvement, whereas large σ nearly guarantees that no better models will be produced.

If large σ values in the range where the paradox occurs $( \mathbf { e . g . } \sigma = 1 0 ^ { - 3 } )$ lead to worse performance, why not use σ in the range of $( 1 . 0 \times 1 0 ^ { - 4 } , 2 . 5 \times 1 0 ^ { - 4 } )$ where improvement is possible? Indeed, in Figure 3, when σ is small enough, the distribution sometimes exceeds parent performance, suggesting that opportunities for improvement exist. However, when σ is set to these values, ES progress is significantly less, as is tested and shown in Figure 4. The improvements are lost in the noise, as discussed in detail in Section 5. However, even though large σ values result in degraded performance, it turns out they can be used to identify good directions reliably, as discussed in the next section.

![](images/a8a267b6494764ec23599f3d4d7165098201e72dd4c5c5672ccda8cb6c5c3b77.jpg)  
Figure 3: Reward distributions at different perturbation magnitudes. Checkpoints in Countdown and Sudoku training runs were perturbed 300 times at each specified magnitude, and distributions of rewards were plotted. As the magnitudes increase, the means decrease, and the variance increases. Thus, the offspring are likely to perform worse than the parent.

![](images/bc202f11c83fac6a5b676a9e336e63b56aa1d68b5b190607299b2c2ef80ea4ae.jpg)  
(a) Fine-tuning performance on Sudoku

![](images/f465868235ac890e688579614b97966d21c2852eaf84820e2fe65b86dc97d25c.jpg)  
(b) Fraction of offspring that outperform parent.  
Figure 4: Effect of perturbation magnitude: a) A comparison in fine-tuning performance on Sudoku with perturbation magnitudes where the paradox occurs $( \sigma = 1 e - 3 )$ vs perturbation magnitudes where offspring produce viable improvement $( \sigma \in \{ 1 e - 4 , 2 . 5 e - 4 \} )$ . b) The fraction of better offspring compared to the parent under different perturbation magnitudes. Even though the smaller perturbation magnitudes produce better candidates, the larger perturbations produce better directions and thus a better overall training run.

## 4 HOW ARE GOOD DIRECTIONS IDENTIFIED?

Section 3 leads to the question: When all models perform worse than the current model, does their relative performance still indicate which directions are more promising? Although it is not possible to directly measure the reward gradient for model checkpoints, it is possible to evaluate the relative performance of perturbations across magnitudes where improving directions do and do not occur. If significant correlations are seen across magnitudes, regardless of performance, then relative performance is indicative of good directions regardless of all perturbation offspring performing worse than the parent.

To test this hypothesis, the checkpoints from Sudoku and Countdown runs were again used. The set of 300 perturbation directions was evaluated on each task’s respective 800-problem dataset at magnitudes corresponding to $\sigma = 1 0 ^ { - 4 }$ and σ from $2 . 5 \times 1 0 ^ { - 4 } \mathrm { t o 2 } \times 1 0 ^ { - 3 }$ in steps of $2 . 5 \times 1 0 ^ { - 4 }$ A correlation matrix was then constructed between the relative rewards of perturbations at these magnitudes. If the perturbation rewards at large σ contain a signal, then they should correlate with the rewards of the same perturbations at small $\sigma .$ From Figure 5 we can see indeed, the rewards are significantly correlated between large and small perturbations, even when the rewards of large perturbations are significantly lower than those of the small perturbations. While greater magnitude differences reduce the measured correlation, the correlations remain positive in most cases.

These correlations likely emerge due to a special property of ES with isotropic Gaussian noise in high-dimensional settings. This noise occupies a thin shell in what has been referred to as a “soap bubble” effect (Huszar, 2017). In a high-dimensional space, all perturbations are nearly identical´ in norm, effectively canceling out any isotropic curvature in the reward landscape, while leaving differences in relative performance due to the gradient. In other words, the magnitude of noise is identical in all directions, and thus, relative performance allows a signal to still emerge.

One way to take advantage of the good directions at large σ would be to simply follow the best perturbation, instead of performing the aggregation through a weighted sum. To evaluate this ablation, a second test was implemented. The perturbation rewards were sampled on the standard (large) σ (1e-3), the best of them identified, and the perturbation was scaled down to the size of an average ES update (i.e. a small σ). A 500-iteration training run was then run with this process. The training curves along with the fraction of perturbations better than the parent are displayed in Figure 6. While this best-of-N selection allows making progress, it is still outperformed by the actual ES update. This result suggests that an additional mechanism of aggregation contributes further to the ES performance, as will be discussed next.

## 5 HOW DO THE UPDATES IMPROVE PERFORMANCE?

Sections 3 and 4 demonstrated that model performance decreases with increasing perturbation; however, even when all offspring are worse than the parent, good search directions can be identified among them. This section demonstrates how the ES aggregation mechanism adds value. The insights are drawn from an analytical model of the ES update in a reward landscape with observation noise and quadratic loss. In this model, the ES update can be decomposed into a signal component along the gradient and a noise component orthogonal to it. The model shows that, in expectation, the ES update moves a fixed distance in the direction of the signal, while reducing the magnitude of the noise component via aggregation of offspring. The mean and variance of the ES update in this model were derived independently by (Hoy et al., 2026), along with a similar decomposition of signal and noise of the update, which they use to estimate parameter drift over training runs. The analysis here differs by focusing on the per-step reward improvement and its relationship to observation noise and curvature.

In the subsections below, the update decomposition is first defined in a simplified linear reward landscape. A realistic observation noise term is then introduced, and the model is extended to include curvature. In the end, the optimal perturbation magnitude is determined in the model, showing that it results in likely improvements. All along, these results are validated empirically with existing training runs.

![](images/719f3870c18882ddc335265d65dc1a3e082aaf59ba74d88a6cebe2c0590a9cca.jpg)  
Figure 5: Correlation between rewards at various perturbation magnitudes σ at iterations 200 and 500 iterations in the Sudoku Fine-Tuning task. The σ range at which the paradox occurs, i.e. the best of the 30 offspring has a lower reward than the parent, is labeled with red arrows. The rewards at small and large σ correlate significantly. Thus, the same good directions can be identified at various perturbation magnitudes, even at which all the perturbations are worse than the parent.

![](images/1c9c164080f86d3a2f83aff312e1fc639cea6ddd6e7b385b9ef3c065e397e988.jpg)  
(a) Training performance of scaled best-of-30 perturbation selection compared with the actual aggregated ES update.

![](images/e12bf5fa6de8d95c3c964b6dcb026d86852bf94297073966163d7eb8ce0c7ffe.jpg)  
(b) Fraction of the perturbations with the best-of-30 method that outperform the parent.  
Figure 6: Best-of-30 selection ablation: (a) Training performance of scaled best-of-30 perturbation selection compared with the actual aggregated ES update. (b) Fraction of the perturbations with the best-of-30 method that are better than the parent. When scaled from a size where the best perturbation performs worse to the size of a typical ES update, the model is able to improve. However, the aggregated ES update direction still significantly outperforms this best-of-30 direction, suggesting that aggregation contributes further to performance.

## 5.1 DEFINING THE SIGNAL AND THE NOISE COMPONENTS OF THE ES UPDATE

Recall the definition of ES update in Equation 1. Let $g = \nabla R ( \theta _ { t } )$ be the gradient in the reward landscape and $\hat { g } = g / \lVert g \rVert$ its normalized direction. Each perturbation $\epsilon _ { i }$ can be divided into a signal component along the gradient, $u _ { i } { \hat { g } } ,$ and a perpendicular noise component, $\epsilon _ { i } ^ { \perp }$ , as

$$
\epsilon _ { i } = u _ { i } \hat { g } + \epsilon _ { \perp i } , \qquad u _ { i } = \hat { g } ^ { \top } \epsilon _ { i } ,\tag{3}
$$

where $\epsilon _ { i } ^ { \perp } \sim \mathcal { N } \big ( 0 , I _ { d } - \hat { g } \hat { g } ^ { \top } \big )$ and $u _ { i } \sim \mathcal { N } ( 0 , 1 )$ (see Appendix A for details). This decomposition can then be applied to the ES update equation Equation 1, dividing it into an update along the gradient and an update perpendicular to the gradient:

$$
\Delta \theta _ { \parallel } \ = \ \Bigl ( \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } Z _ { i } u _ { i } \hat { g } \Bigr ) , \qquad \Delta \theta _ { \perp } \ = \ \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \epsilon _ { \perp i } .\tag{4}
$$

Under linear reward, $R ( \theta _ { i } ) = R ( \theta _ { t } ) + g ^ { \top } ( \theta _ { i } - \theta _ { t } )$ . Then $R _ { i } = R ( \theta _ { t } ) + \sigma \| g \| u _ { i }$ . Using an assumption that the sampled reward mean and standard deviation match their expected values, $\mu _ { R } =$ $R ( \theta _ { t } )$ and $\sigma _ { R } = \sigma \lVert g \rVert$ , and an individual perturbation’s z-score is

$$
Z _ { i } \ = \ { \frac { \sigma \| g \| u _ { i } } { \sigma \| g \| } } \ = \ u _ { i } .\tag{5}
$$

Each individual perturbation, as weighted in the ES update, can be written as its signal and noise components. The weighted signal component of perturbation i becomes $\alpha u _ { i } ^ { 2 } \hat { g }$ , with $\begin{array} { r } { \check { u _ { i } ^ { 2 } } \sim \chi _ { 1 } ^ { 2 } . } \end{array}$ , while the weighted noise component becomes $u _ { i } \epsilon \_ i ,$ , allowing for their expected norm and squared norm of scaled perturbation signal and noise components to be calculated as

$$
\mathrm { p e r t u r b a t i o n ~ s i g n a l : } \qquad \mathbb { E } [ \| \alpha u _ { i } ^ { 2 } \hat { g } \| ] \ = \ \alpha ,\tag{6}
$$

$$
\mathrm { p e r t u r b a t i o n n o i s e : ~ } \quad \sqrt { { \mathbb { E } } \left[ \| u _ { i } \epsilon _ { \perp i } \| ^ { 2 } \right] } = \alpha \sqrt { d - 1 } .\tag{7}
$$

When switching to the full ES update aggregation, $\textstyle \sum _ { i } u _ { i } ^ { 2 } \sim \chi _ { N } ^ { 2 }$ , and the update magnitudes are

$$
\mathrm {  ~ \Psi ~ \ u p d a t e ~ s i g n a l { : } ~ } \quad \mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \alpha ,\tag{8}
$$

$$
\mathrm { u p d a t e ~ n o i s e } \colon \quad \sqrt { { \mathbb { E } } \left[ \| \Delta \theta _ { \perp } \| ^ { 2 } \right] } = \alpha \sqrt { \frac { d - 1 } { N } } .\tag{9}
$$

Notice that the expected magnitude of the update component along the gradient is the same as the magnitude of the perturbation component on the gradient, while the update noise component shrinks with $\sqrt { N } .$ . This result shows that the aggregation of the ES update maintains the expected signal constant while reducing noise. For full derivations, see Appendix A.

## 5.2 INCLUDING OBSERVATION NOISE

In the ES fine-tuning tasks, the models are evaluated at temperature zero, and the model outputs are therefore deterministic. However, when exploring along specified directions, noise occurs in the reward landscape at much higher frequencies than the magnitudes of the perturbations used (as shown in Figure 7).Similar to the nugget term in Gaussian Process models (Pepelyshev, 2010), this noise can be incorporated in the analytical model as observation noise at scale $\sigma _ { \xi } \mathbf { : }$

$$
R _ { i } = R ( \theta _ { t } + \sigma \epsilon _ { i } ) + \sigma _ { \xi } \xi _ { i } , \xi _ { i } \sim \mathcal { N } ( 0 , 1 ) .\tag{10}
$$

Plugging this reward function into the ES update (Equation 1) produces the expected signal, noise, and reward improvement as

![](images/f01d6ffe65865c72fbe0d7c236ad571a54022c1c9597d57b000fa41cdbfa2502.jpg)

![](images/6349f822ad4a5bb01d326e4c133e5e769edce7fa90b148922bbf70cb919bc563.jpg)  
Figure 7: Average accuracy of Qwen 2.5 7B on Sudoku task along a random direction from −2σ $\tan + 2 \sigma$ . Left: Landscape on a set of 800 problems. Right: Landscape on a set of 50 problems. A significant portion of the variance occurs at frequencies much higher than the perturbation size used in ES, and can thus be treated as observation noise.

$$
\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \frac { \alpha \sigma \| g \| } { \sqrt { \sigma ^ { 2 } \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } } } , \quad \sqrt { \mathbb { E } \left[ \| \Delta \theta _ { \perp } \| ^ { 2 } \right] } = \alpha \sqrt { \frac { d - 1 } { N } } , \quad \mathbb { E } [ \Delta R ] = \frac { \alpha \sigma \| g \| ^ { 2 } } { \sqrt { \sigma ^ { 2 } \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } } } .\tag{11}
$$

Notice that with observation noise, both the expected signal and reward increase with increasing $\sigma .$ Thus, whereas the variance in the reward for small σ is mostly produced by noise, larger σ increases reward variance, making the ES estimate more accurate. For full derivations, see Appendix A.

## 5.3 INCLUDING LANDSCAPE CURVATURE

In the third step, the analytical model is extended to include general downward curvature $Q .$ . The reward can then be written as

$$
\begin{array} { r } { R ( \theta ) = R ( \theta _ { t } ) + g ^ { \top } ( \theta - \theta _ { t } ) - \frac { 1 } { 2 } ( \theta - \theta _ { t } ) ^ { \top } Q ( \theta - \theta _ { t } ) . } \end{array}\tag{12}
$$

The expected signal, noise, and reward are

signal:

$$
\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \frac { \alpha \sigma \lVert g \rVert } { \sigma _ { R } } = \frac { \alpha \lVert g \rVert } { \sqrt { \lVert g \rVert ^ { 2 } + \frac { \sigma ^ { 2 } } { 2 } \operatorname { t r } [ Q ^ { 2 } ] + \frac { \sigma _ { \xi } ^ { 2 } } { \sigma ^ { 2 } } } } ,\tag{13}
$$

$$
\mathrm { n o i s e } \colon \quad \sqrt { \mathbb { E } \left[ \| \Delta \theta _ { \perp } \| ^ { 2 } \right] } \approx \alpha \sqrt { \frac { d - 1 } { N } } ,\tag{14}
$$

reward:

$$
\mathbb { E } [ \Delta R ] \approx \underbrace { \frac { \alpha \| g \| ^ { 2 } } { \sqrt { \| g \| ^ { 2 } + \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } [ Q ^ { 2 } ] + \frac { \sigma _ { \xi } ^ { 2 } } { \sigma ^ { 2 } } } } } _ { \mathrm { s i g n a l g a i n } } - \underbrace { \frac { \alpha ^ { 2 } } { 2 N } \mathrm { t r } [ Q ] } _ { \mathrm { n o i s e p e n a l t y } } ,\tag{15}
$$

where the terms that are negligible for noise and reward have been dropped. The crucial property identified in the linear case also holds with observation noise and curvature: the signal component of the update is the same for a perturbation as it is for the aggregate update, but the noise component of the aggregate is smaller.

With curvature, a tradeoff occurs between increasing perturbation size to avoid observation noise $\sigma _ { \xi } ^ { 2 }$ and decreasing the perturbation size to reduce the effect of curvature via $\operatorname { t r } [ Q ^ { 2 } ]$ . This tradeoff has been documented with perturbation-based methods before (Spall, 1992; Duchi et al., 2015; Nesterov & Spokoiny, 2017). However, in high dimensions, the ES estimate has two unique properties. First, population size only affects the magnitude of the noise component. Thus, the expected gain in the reward plateaus once the population size reduces noise enough so that the ES update is no

![](images/3ef8161daeb82008b76d51048befd096e406215389dbeab453819e18527fce0a.jpg)  
(a) MNIST: Reward improvement vs. population size

![](images/f97a925ef197ad2b4344238da5ff502a8cd01f95bb594ebded2e23d0a87cbb6d.jpg)  
(b) MNIST: Update size vs. population size

![](images/5d446d1d50e23f4937c8cf6ffe544f3fa74151c36a47d191adc075696696f013.jpg)  
(c) Fine-tuning: Reward improvement vs. population size

![](images/c7355a6c2f78a351ed6d4386f639b395ea046de394a23b8850a91f330b64e747.jpg)  
(d) Fine-tuning: Update size vs. population size

Figure 8: Update reward improvement and update magnitude vs. population size. (a) Distributions of reward improvements from ES updates for different population sizes for CNN on MNIST. (b) Magnitude of ES updates in MNIST. (c) Mean and confidence intervals of improvement from ES updates at different population sizes for the Sudoku LLM fine-tuning task. (d) Magnitude of ES update for various population sizes. Distributions were wide, so for clarity, only means and CIs are shown. The model was Qwen 2.5 7B. In both settings, update reward improvement increases with population size and then plateaus, while the norm of the ES update decreases with population size.

longer affected by curvature. Second, while the gain in reward plateaus with additional population members, the magnitude of update itself falls proportional to $\scriptstyle { \frac { 1 } { \sqrt { N } } }$

To test whether these properties show up empirically with ES optimization, checkpoints were taken every 100 iterations from the Sudoku 500 iteration fine-tuning run, as well as from the MNIST training run. With each checkpoint, 30 ES updates were drawn from independent sets of N perturbations, where N is the designated population size. The reward of these updates was measured, and confidence intervals drawn for the mean reward update (Figure 8). Notice that, as predicted by the noise-curvature model, the expected reward update increases and then plateaus with population size. Further, the update sizes are highly consistent, and even as their magnitude decreases significantly with respect to population size, the reward improvement remains the same. This result demonstrates that the ES update is holding the signal constant while reducing noise.

These plots also illuminate why ES is able to work on billion-parameter models with population sizes of only 30: once the noise is reduced enough to avoid the curvature penalty, ES achieves its maximum possible improvement on each step.

## 5.4 EFFECTIVE PERTURBATION SCALE TRIGGERS THE PARADOX

As shown in Section 3, testing an alternative σ value that avoids the paradox revealed that perturbation magnitudes that invoked the paradox outperformed smaller perturbations that did not. The noise-curvature model offers a tradeoff between observation noise and curvature. If α, g, Q and $\sigma _ { \xi }$ are fixed, the expected improvement in Equation 15 is maximized by minimizing

![](images/9649fcd13ed9593b8e1c44c554d804d11fa3af941abae559a2584c0601790571.jpg)  
(a) Single Step Reward vs. perturbation

![](images/8cc5d03791d4ec12a089984d81a40d18e0940dadb4c9e4018ebccc0e4dfe3e81.jpg)  
(b) Training Performance vs. σ

Figure 9: (a): A visual representation of reward vs. perturbation magnitude, demonstrating why the offspring are worse than the parent. Perturbations happen at a fixed size; if they are too small, observation noise drowns out differences between good and bad directions, and if they are too big, the correlation between perturbation reward and direction quality breaks down. The paradox occurs when the magnitude is at a distance where all perturbations have a reward worse than the current model, but they still differentiate the good directions from the bad. (b) A demonstration of the tradeoff in perturbation size empirically, with performance of Qwen 2.5 7B on the Sudoku task after 300 ES iterations varying $\sigma ,$ , holding all else constant. Results illustrate observation noise deteriorating performance at small values of $\sigma _ { \mathrm { { : } } }$ , and correlations breaking down at large values, demonstrating the tradeoff visualized by (a).

$\begin{array} { r } { f ( \sigma ^ { 2 } ) = \sigma _ { \xi } ^ { 2 } / \sigma ^ { 2 } + \frac { \sigma ^ { 2 } } { 2 } \operatorname { t r } [ Q ^ { 2 } ] } \end{array}$ over $\sigma ^ { 2 } .$ . This setting gives

$$
\sigma _ { * } ^ { 4 } = \frac { 2 \sigma _ { \xi } ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } , \qquad \sigma _ { * } = \left( \frac { 2 \sigma _ { \xi } ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } \right) ^ { 1 / 4 } .\tag{16}
$$

To characterize the population at this optimal sigma, using the fact that $\mathbb { E } [ \epsilon _ { i } ^ { \top } Q \epsilon _ { i } ] = \mathrm { t r } [ Q ]$ , the expected difference in the reward from the parent is

$$
\mu _ { R } - { \cal R } ( \theta _ { t } ) = - \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } [ Q ] , \qquad \left. \mu _ { R } - { \cal R } ( \theta _ { t } ) \right| _ { \sigma _ { * } } = - \sigma _ { \xi } \sqrt { \frac { r _ { \mathrm { e f f } } } { 2 } } , \qquad r _ { \mathrm { e f f } } : = \frac { ( \mathrm { t r } [ Q ] ) ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } ,\tag{17}
$$

where $r _ { \mathrm { e f f } } \in [ 1 , d ]$ is the participation ratio (effective rank) of the Hessian, equal to d for isotropic curvature and to r for a rank-r Hessian with equal eigenvalues. In units of the population spread,

$$
\frac { R ( \theta _ { t } ) - \mu _ { R } } { \sigma _ { R } } = \frac { \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } [ Q ] } { \sqrt { \sigma ^ { 2 } \| g \| ^ { 2 } + \frac { \sigma ^ { 4 } } { 2 } \mathrm { t r } [ Q ^ { 2 } ] + \sigma _ { \xi } ^ { 2 } } } = \frac { \mathrm { t r } [ Q ] } { 2 \sqrt { \frac { 1 } { \sigma ^ { 2 } } \| g \| ^ { 2 } + \frac { 1 } { 2 } \mathrm { t r } [ Q ^ { 2 } ] + \frac { \sigma _ { \xi } ^ { 2 } } { \sigma ^ { 4 } } } } ,\tag{18}
$$

where $\sigma _ { R }$ is the standard deviation of the population rewards. Equation 18 quantifies how the rewards of perturbations distribute compared to the parent reward $R ( \theta _ { t } )$ . A larger value of Equation 18 corresponds to a higher probability for the perturbation to be worse than the parent.

Since the value of Equation 18 is increasing with respect to $\sigma ,$ a smaller σ decreases the probability of generating worse populations. In the meantime, a smaller $\sigma ( \ll \sigma _ { * } )$ also reduces the expected improvement of ES updates, according to Equation 15. As a result, a smaller σ decreases the chance for the paradox to happen. For larger $\sigma ,$ , although the chance for producing worse perturbations is increased, the expected improvement is also reduced $( \mathrm { i f } \gg \sigma _ { * } )$ , resulting in fewer occurrences of the paradox. This is demonstrated visually in Figure 9a and empirically in Figure 9b, which shows that if a σ value provides a larger improvement for ES updates, its chance to trigger the paradox also increases. Therefore, the paradox is a byproduct of a highly effective ES run.

## 6 DISCUSSION AND FUTURE WORK

The ES paradox emerged as an unexpected anomaly in population-based search in large spaces. Such methods are presumed to work by creating offspring that are better than parents, yet in many cases ES is able to consistently improve even when this is not true. Understanding the mechanism behind this paradox led to a new way of thinking about search in high-dimensional spaces: instead of an anomaly, the paradox is a side effect of a principled interaction of noise and curvature in such spaces. This mechanism is different from gradient descent, simulated annealing, and other evolutionary computation, so much so that it can be seen as a paradigm shift in the sense of Kuhn (1970): an anomaly becomes a fundamental principle in the new way of seeing things. As a result, old assumptions need to be re-evaluated, and new opportunities arise for progress. For instance:

Small Population Performance of ES: One of the most dramatic features of the new line of ES work in LLM fine-tuning has been the significantly smaller population size. Previously, it had been assumed that ES had fundamental scaling limitations with respect to high-dimensional spaces (Jamieson et al., 2012), yet within fine-tuning, it has been competitive with, and often even outperforming, gradient-based reinforcement learning methods. This work now sheds some light on this phenomenon, demonstrating that once the noise component of the ES update is small enough to avoid incurring a significant cost from curvature, the update is able to achieve the maximum improvement in reward - even if the noise component is orders of magnitude larger than the signal component.

The surprising Robustness of LLMs to Noise in Parameter Space: From the update reward improvement estimate results in Figure 8, per step update is seen to plateau at approximately a population size of 30. Given the analysis in Section 5, this would imply each ES update contains a signal component of maximum magnitude α, while the noise component contains a magnitude of $\begin{array} { r } { \alpha \sqrt { \frac { d - 1 } { N } } = \alpha \sqrt { \frac { 7 \times 1 0 ^ { 9 } } { 3 0 } } \approx \alpha \times 1 5 , 0 0 0 } \end{array}$ . While fine-tuning, for every unit distance ES moves the network in the direction of model improvement, it moves 15,000 unit distances in an irrelevant direction. This is consistent with (Hoy et al., 2026) which finds that the total parameter drift of ES-fine-tuned LLMs follows the random-walk prediction $\alpha ^ { 2 } T d / N$ to within a few percent. This demonstrates a surprising fact that LLMs can have their parameters dramatically modified in random directions without affecting performance.

Future Work: Results and the analytical model suggest that there may be ways to further optimize the hyperparameters of ES. Section 5.4 presented an optimal perturbation size conditional on observation noise and curvature. Future work may look at how to identify these properties during a run, or whether estimation of said optimums is too noisy to do online, but may be aggregated across fine-tuning runs. Additional work could also modify the ES algorithm itself to take advantage of this new paradigm, where improving offspring is no longer necessary for population-based optimization.

## 7 CONCLUSION

The ES paradox presents a counterintuitive conclusion: that ES can learn while consistently generating worse offspring than its parent. This paper demonstrates why this is possible: even when all offspring perform worse than their parent, their relative performance contains valuable information for improving directions. The ES update then aggregates this signal while reducing the noise to a point where improvement can be made. Together, these results suggest that ES in large-dimensional spaces can be seen as a new optimization paradigm: it takes advantage of the interactions of noise and curvature in such spaces the way other standard methods do not. The results provide insight into how ES is able to function effectively on billion-parameter LLMs, and suggest opportunities for improving the method further.

## AI USE STATEMENT

In this work, we used generative AI tools to help implement methods and provide critical ingredients for proving mathematical claims. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## REFERENCES

Yunpeng Ba, Zhi Zheng, Yue Xie, Jiaqing Li, Xialiang Tong, Tao Zhong, Mingxuan Yuan, Zhichao Lu, Xuyang Wu, and Zhenkun Wang. Understanding evolution strategies for llm reasoning: Broader reasoning coverage than grpo, 2026. URL https://arxiv.org/abs/2608. 27351.

John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788–2806, 2015.

Russell Eberhart and James Kennedy. A new optimizer using particle swarm theory. In MHS’95. Proceedings ofthe sixth international symposium on micro machine and human science, pp. 39– 43. Ieee, 1995.

Yulu Gan and Phillip Isola. Neural thickets: Diverse task experts are dense around pretrained weights. In Forty-third International Conference on Machine Learning, 2026.

David E. Goldberg. Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley, Reading, MA, 1989. ISBN 0201157675.

Conor F. Hayes, Elliot Meyerson, Kajetan Schweighofer, Roberto Dailey, Babak Hodjat, Risto Miikkulainen, and Xin Qiu. Beyond the best guess: Improving llm solution coverage with evolution strategies, 2026. URL https://arxiv.org/abs/2608.12679.

William Hoy, Binxu Wang, and Xu Pan. Matching accuracy, different geometry: Evolution strategies vs grpo in llm post-training, 2026. URL https://arxiv.org/abs/2604.01499.

Ferenc Huszar.´ Gaussian distributions are soap bubbles. inFER-ENCe (blog), November 2017. URL https://www.inference.vc/ high-dimensional-gaussian-distributions-are-soap-bubble/. Accessed: 2026-09-17.

Kevin G Jamieson, Robert Nowak, and Ben Recht. Query complexity of derivative-free optimization. Advances in neural information processing systems, 25, 2012.

Thomas S. Kuhn. The Structure of Scientific Revolutions. The University of Chicago Press, Chicago, second, enlarged edition, 1970.

Joel Lehman, Jay Chen, Jeff Clune, and Kenneth O Stanley. Es is more than just a traditional finitedifference approximator. In Proceedings of the genetic and evolutionary computation conference, pp. 450–457, 2018.

Qiyao Liang, Jinyeop Song, Yizhou Liu, Jeff Gore, Ila Fiete, Risto Miikkulainen, and Xin Qiu. The blessing of dimensionality in llm fine-tuning: A variance-curvature perspective, 2026. URL https://arxiv.org/abs/2602.00170.

Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations ofComputational Mathematics, 17(2):527–566, 2017.

Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.

Andrey Pepelyshev. The role of the nugget term in the gaussian process method, 2010. URL https://arxiv.org/abs/1005.4385.

Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. Evolution strategies at scale: Llm fine-tuning beyond reinforcement learning, 2026. URL https://arxiv.org/abs/2509.24372.

Ingo Rechenberg. Cybernetic solution path of an experimental problem. Roy. Aircr. Establ., Libr. transl., 1122, 1965.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning, 2017. URL https://arxiv.org/abs/ 1703.03864.

J.C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3):332–341, 1992. doi: 10.1109/9.119632.

Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, and Jurgen Schmidhuber. Natural evolu-¨ tion strategies, 2011. URL https://arxiv.org/abs/1106.4487.

Xingwen Zhang, Jeff Clune, and Kenneth O. Stanley. On the relationship between the openai evolution strategy and stochastic gradient descent, 2017. URL https://arxiv.org/abs/1712. 06564.

Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning, 2025. URL https://arxiv.org/abs/ 2504.12216.

## A SIGNAL, NOISE, AND THE PERTURBATION SCALE IN Z-SCORED EVOLUTION STRATEGIES

This section starts with the basic properties of the ES estimator with z-score normalized rewards (Qiu et al., 2026), under linear and quadratic reward landscapes. The mean and covariance of the update in Sections A.2–A.4 agree with those derived independently by (Hoy et al., 2026). They are restated here for completeness, in the signal/noise form used in the main text. The new material is the expected reward change in equation 49, the optimal perturbation scale and the population statistics at that scale (Section A.5), and the concentration of the update norm (Appendix B).

## A.1 THE Z-SCORED ES UPDATE

At iteration t, ES draws N perturbation directions $\epsilon _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ , evaluates the perturbed models $\theta _ { t } + \sigma \epsilon _ { i }$ to obtain rewards $R _ { i } .$ , and moves the parent by the reward-weighted average of the perturbations,

$$
\Delta \theta = \theta _ { t + 1 } - \theta _ { t } = \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \epsilon _ { i } ,\tag{19}
$$

where α is the step size and the weights $Z _ { i }$ are the population-standardized rewards,

$$
Z _ { i } = \frac { R _ { i } - \mu _ { R } } { \sigma _ { R } } , \qquad \mu _ { R } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R _ { j } , \qquad \sigma _ { R } ^ { 2 } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( R _ { j } - \mu _ { R } ) ^ { 2 } .\tag{20}
$$

The sample statistics $( \mu _ { R } , \sigma _ { R } )$ in equation 20 are replaced by their population counterparts, $\mu _ { R } =$ $\mathbb { E } [ R _ { i } ]$ and $\sigma _ { R } ^ { 2 } = \mathrm { V a r } [ \ddot { R } _ { i } ]$ , in the derivations below.

Perturbation and update magnitudes. Before specifying any reward landscape, two magnitudes are fixed by the sampling and normalization alone. The norm of each perturbation follows a scaled chi distribution, $\lVert \sigma \epsilon _ { i } \rVert \sim \sigma \chi _ { d }$ , so its first two moments are exact:

$$
\mathbb { E } \| \sigma \epsilon _ { i } \| = \sigma \frac { \sqrt { 2 } \Gamma \big ( \frac { d + 1 } { 2 } \big ) } { \Gamma \big ( \frac { d } { 2 } \big ) } = \sigma \sqrt { d } \Big ( 1 - \frac { 1 } { 4 d } + O ( d ^ { - 2 } ) \Big ) ,\tag{21}
$$

$$
\operatorname { V a r } \| \sigma \epsilon _ { i } \| = \sigma ^ { 2 } \Big ( d - \big ( \mathbb { E } \| \epsilon _ { i } \| \big ) ^ { 2 } \Big ) \xrightarrow { d \to \infty } \frac { \sigma ^ { 2 } } { 2 } .\tag{22}
$$

The standard deviation of the perturbation length tends to the constant $\sigma / { \sqrt { 2 } } ,$ so its relative spread is $1 / \sqrt { 2 d }$

The standardized weights satisfy, by construction and irrespective of the rewards,

$$
\sum _ { i = 1 } ^ { N } Z _ { i } \ = \ 0 , \qquad \sum _ { i = 1 } ^ { N } Z _ { i } ^ { 2 } \ = \ N .\tag{23}
$$

The first identity removes any constant offset in the rewards. The second fixes the update length. Because each $\dot { Z _ { i } }$ depends only on its own perturbation, the weighted perturbations $\bar { Z _ { i } } \epsilon _ { i }$ are identically distributed, and the landscape enters the moments of $\| \Delta \mathbf { \check { \theta } } \| ^ { 2 }$ through only two scalars: the squared drift $\| \boldsymbol { m } \| ^ { 2 }$ with $m = \mathbb { E } [ Z _ { i } ^ { \bullet } \epsilon _ { i } ]$ , which satisfies $\lVert m \rVert \leq 1 \mathrm { { b y } }$ Cauchy–Schwarz, and the inflation $\kappa = \mathbb { E } [ \breve { Z } _ { i } ^ { 2 } \lVert \dot { \epsilon } _ { i } \rVert ^ { 2 } ] - d ,$ which the derivations below show lies in [0, 4]. Expanding $\| \Delta \theta \| ^ { 2 }$ into its N diagonal terms $Z _ { i } ^ { 2 } \| \epsilon _ { i } \| ^ { 2 }$ and $N ( N - 1 )$ cross terms $Z _ { i } Z _ { j } \epsilon _ { i } ^ { \top } \epsilon _ { j } .$ , the diagonal sum is pinned to $N d + O ( N )$ by equation 23 while each cross term has mean $\| m \| ^ { 2 }$ and variance $d ,$ giving $( \mathsf { A p - }$ pendix B)

$$
\mathbb { E } \| \Delta \theta \| ^ { 2 } \ = \ \frac { \alpha ^ { 2 } d } { N } + \frac { \alpha ^ { 2 } \kappa } { N } + \alpha ^ { 2 } \Big ( 1 - \frac { 1 } { N } \Big ) \| m \| ^ { 2 } \ = \ \frac { \alpha ^ { 2 } d } { N } \Big ( 1 + { \cal O } \big ( N / d \big ) \Big ) ,\tag{24}
$$

$$
\mathrm { V a r } \| \Delta \theta \| ^ { 2 } = \frac { 2 \alpha ^ { 4 } d } { N ^ { 2 } } \Big ( 1 + { \cal O } \big ( N / d \big ) \Big ) , \qquad \frac { \sqrt { \mathrm { V a r } } \| \Delta \theta \| ^ { 2 } } { \mathbb { E } \| \Delta \theta \| ^ { 2 } } = \sqrt { \frac { 2 } { d } } \Big ( 1 + { \cal O } \big ( N / d \big ) \Big ) .\tag{25}
$$

The length of a z-scored ES step is therefore set by $( \alpha , d , N )$ , with coefficient of variation $\sqrt { 2 / d }$ so that $\sqrt { \mathbb { E } \lVert \Delta \theta \rVert ^ { 2 } }$ is accurate to relative order $1 / d ;$ the rewards determine its direction, through $m .$ and contribute to its length only at relative order $N / d .$

Signal and noise. Let $g = \nabla R ( \theta _ { t } )$ be the local gradient and $\hat { g } = g / \lVert g \rVert$ its direction. We decompose every update into the component along the gradient and the component orthogonal to it,

$$
\begin{array} { r } { \Delta \theta = \underbrace { \Delta \theta _ { \parallel } } _ { \mathrm { s i g n a l } } + \underbrace { \Delta \theta _ { \perp } } _ { \mathrm { n o i s e } } , \qquad \Delta \theta _ { \parallel } = \left( \hat { g } ^ { \top } \Delta \theta \right) \hat { g } , \qquad \Delta \theta _ { \perp } = \left( I _ { d } - \hat { g } \hat { g } ^ { \top } \right) \Delta \theta , } \end{array}\tag{26}
$$

and characterize the signal by its expected projection $\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ]$ and the noise by its root-mean-square magnitude $\sqrt { \mathbb { E } \lVert \Delta \theta _ { \perp } \rVert ^ { 2 } }$

To carry out the decomposition, split each perturbation into its gradient projection and its orthogonal remainder,

$$
\begin{array} { r l r l r l r } { \epsilon _ { i } = u _ { i } \hat { g } + \epsilon _ { i } ^ { \perp } , } & { } & { u _ { i } = \hat { g } ^ { \top } \epsilon _ { i } \sim \mathcal { N } ( 0 , 1 ) , } & { } & { \epsilon _ { i } ^ { \perp } \sim \mathcal { N } \big ( 0 , I _ { d } - \hat { g } \hat { g } ^ { \top } \big ) , } \end{array}\tag{27}
$$

with $u _ { i }$ and $\epsilon _ { i } ^ { \perp }$ independent. Substituting into equation 1,

$$
\Delta \theta _ { \parallel } = \alpha \Big ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } u _ { i } \Big ) \hat { g } , \qquad \Delta \theta _ { \perp } = \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \epsilon _ { i } ^ { \perp } .\tag{28}
$$

In every case considered below, $\mathbb { E } [ Z _ { i } \epsilon _ { i } ^ { \perp } ] = 0$ , because each term of $Z _ { i }$ is either odd in $\epsilon _ { i } ^ { \perp }$ or independent of it. The cross terms between different i in $\| \Delta \theta _ { \perp } \| ^ { 2 }$ therefore vanish, and

$$
\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] \ = \ \alpha \mathbb { E } [ Z _ { i } u _ { i } ] , \qquad \mathbb { E } \| \Delta \theta _ { \perp } \| ^ { 2 } \ = \ \frac { \alpha ^ { 2 } } { N } \mathbb { E } \big [ Z _ { i } ^ { 2 } \| \epsilon _ { i } ^ { \perp } \| ^ { 2 } \big ] .\tag{29}
$$

The three cases below add one ingredient at a time: a linear landscape with exactly observed rewards, a linear landscape with observation noise, and a quadratic landscape with observation noise.

## A.2 LINEAR LANDSCAPE WITHOUT OBSERVATION NOISE

Let the reward be locally linear, $R ( \theta ) = R ( \theta _ { t } ) + g ^ { \top } ( \theta - \theta _ { t } )$ , and suppose each reward is observed exactly, $R _ { i } = R ( \theta _ { t } + \sigma \epsilon _ { i } )$ . Then $R _ { i } = R ( \theta _ { t } ) + \sigma \| g \| u _ { i } .$ , so $\mu _ { R } = R ( \theta _ { t } ) , \sigma _ { R } = \sigma \| g \|$ , and the z-score collapses to the gradient projection of the perturbation,

$$
Z _ { i } \ = \ { \frac { \sigma \| g \| u _ { i } } { \sigma \| g \| } } \ = \ u _ { i } .\tag{30}
$$

Both $\sigma$ and $\| g \|$ cancel. The parallel coefficient in equation 28 is $\begin{array} { r } { \alpha \cdot \frac { 1 } { N } \sum _ { i } u _ { i } ^ { 2 } } \end{array}$ with $\textstyle \sum _ { i } u _ { i } ^ { 2 } \sim \chi _ { N } ^ { 2 }$ , so it has mean α and variance $2 \alpha ^ { 2 } / N$ . For the orthogonal component, $Z _ { i } = u _ { i }$ is independent of $\epsilon _ { i } ^ { \perp }$ and $\mathbb { E } [ u _ { i } ^ { 2 } ] = 1 , \mathbb { E } \| \epsilon _ { i } ^ { \perp } \| ^ { 2 } = d - 1$ . Hence

$$
\begin{array} { r l } { \mathrm { s i g n a l : } \quad } & { { } \mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \alpha , } \end{array}\tag{31}
$$

$$
\mathrm { n o i s e } \colon \quad \sqrt { \mathbb { E } \| \Delta \theta _ { \perp } \| ^ { 2 } } = \alpha \sqrt { \frac { d - 1 } { N } } .\tag{32}
$$

In the absence of observation noise, the z-scored ES step therefore carries a signal of expected length exactly α along the gradient, independent of both the perturbation scale and the gradient magnitude, surrounded by isotropic noise of length $\alpha \sqrt { ( d - 1 ) / N }$ . The expected cosine between the step and the gradient is approximately $\sqrt { N / d }$ . This is the best case: none of the settings below improves on it.

Reward change. On a linear landscape, the reward change of the parent is $\Delta R : = R ( \theta _ { t + 1 } ) - $ $R ( \theta _ { t } ) = g ^ { \top } \Delta \breve { \theta } = \| g \| \hat { g } ^ { \top } \Delta \theta$ , so it depends only on the parallel coefficient:

$$
\mathbb { E } [ \Delta R ] = \alpha \| g \| , \mathrm { V a r } [ \Delta R ] = \frac { 2 \alpha ^ { 2 } \| g \| ^ { 2 } } { N } .\tag{33}
$$

Each step gains $\alpha \lVert g \rVert$ in expectation, independent of $\sigma ,$ d and $N .$ . The orthogonal noise moves the parent sideways but, on a linear landscape, sideways movement has no reward cost; $N$ enters only through the variance, and the step improves the reward with probability $\approx \Phi ( \sqrt { N / 2 } )$

## A.3 LINEAR LANDSCAPE WITH OBSERVATION NOISE

We now let each observed reward be the reward of the perturbed model plus an additive observation noise,

$$
R _ { i } = R ( \theta _ { t } + \sigma \epsilon _ { i } ) + \sigma _ { \xi } \xi _ { i } , \xi _ { i } \sim \mathcal { N } ( 0 , 1 ) , \xi _ { i } \perp \epsilon _ { i } ,\tag{34}
$$

with $\sigma _ { \xi } ~ > ~ 0$ the observation-noise scale. Keeping the linear landscape, the centered reward is $R _ { i } - \dot { \mu _ { R } } = \sigma \| g \| u _ { i } + \sigma _ { \xi } \xi _ { i }$ , and

$$
\sigma _ { R } ^ { 2 } = \sigma ^ { 2 } \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } .\tag{35}
$$

The z-score is now

$$
Z _ { i } \ = \ \frac { \sigma \| g \| u _ { i } + \sigma _ { \xi } \xi _ { i } } { \sigma _ { R } } \ = : w _ { i } \sim { \mathcal N } ( 0 , 1 ) , \qquad \mathrm { C o r r } ( w _ { i } , u _ { i } ) \ = \ \frac { \sigma \| g \| } { \sigma _ { R } } \ = : c \in [ 0 , 1 ] ,\tag{36}
$$

a unit-variance Gaussian that is only partially correlated with the gradient projection. For the parallel coefficient, $\mathbb { E } [ w _ { i } u _ { i } ] = c$ and, for a correlated standard-normal pair, $\mathbb { E } [ \check { w } _ { i } ^ { 2 } u _ { i } ^ { 2 } ] = 1 + 2 c ^ { 2 }$ , so the coefficient has mean αc and variance $\alpha ^ { 2 } ( 1 + c ^ { 2 } ) / N$ . For the orthogonal component, w<sub>i</sub> is independent of $\epsilon _ { i } ^ { \perp }$ and $\mathbb { E } [ w _ { i } ^ { 2 } ] = 1$ , so its expected squared norm is unchanged from the noiseless case. Hence

$$
\mathrm { s i g n a l : } \qquad \mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \alpha c = \frac { \alpha \sigma \| g \| } { \sqrt { \sigma ^ { 2 } \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } } } ,\tag{37}
$$

noise:

$$
\sqrt { \mathbb { E } \| \Delta \theta _ { \perp } \| ^ { 2 } } = \alpha \sqrt { \frac { d - 1 } { N } } .\tag{38}
$$

Observation noise attenuates the expected signal by the factor c and leaves the noise magnitude exactly unchanged. It also makes the realized signal less reliable: the standard deviation of the parallel coefficient is $\alpha \sqrt { ( 1 + c ^ { 2 } ) / N }$ , so the projection of a single step onto the gradient is positive with high probability only when $c \gtrsim 1 / \sqrt { N }$ . Rewriting the attenuation as

$$
c = \frac { \| g \| } { \sqrt { \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } / \sigma ^ { 2 } } }\tag{39}
$$

shows that it is monotonically increasing in $\sigma ,$ approaching 1 as $\sigma \| g \| / \sigma _ { \xi } \to \infty$ and recovering equation 31–equation 32. On an exactly linear landscape, this trade has no cost: Gaussian smoothing of a linear function is exact and introduces no bias, so the perturbation scale should be made as large as possible. The only thing that stops it is the breakdown of linearity.

Reward change. As before $\Delta R = \| g \| \hat { g } ^ { \top } \Delta \theta _ { \ast }$ , now with the attenuated parallel coefficient:

$$
\mathbb { E } [ \Delta R ] = \alpha c \| g \| = \frac { \alpha \sigma \| g \| ^ { 2 } } { \sqrt { \sigma ^ { 2 } \| g \| ^ { 2 } + \sigma _ { \xi } ^ { 2 } } } , \qquad \mathrm { V a r } [ \Delta R ] = \frac { \alpha ^ { 2 } \| g \| ^ { 2 } ( 1 + c ^ { 2 } ) } { N } .\tag{40}
$$

Observation noise scales the expected gain by c and makes individual steps less reliable, with $\mathbb { P } [ \Delta R > 0 ] \approx \Phi \big ( c \sqrt { N / ( 1 + c ^ { 2 } ) } \big )$ , but the orthogonal noise remains free.

## A.4 QUADRATIC LANDSCAPE WITH OBSERVATION NOISE

To capture that breakdown, add curvature: $\begin{array} { r } { R ( \theta ) = R ( \theta _ { t } ) + g ^ { \top } ( \theta - \theta _ { t } ) - \frac { 1 } { 2 } ( \theta - \theta _ { t } ) ^ { \top } Q \left( \theta - \theta _ { t } \right) } \end{array}$ , with Q a general symmetric matrix (neither diagonal nor isotropic is assumed). Writing $q _ { i } = \epsilon _ { i } ^ { \top } Q \epsilon _ { i } - \mathrm { t r } [ Q ]$ for the centered curvature fluctuation, which has $\mathbb { E } [ q _ { i } ] = 0$ and $\mathrm { V a r } [ q _ { i } ] = \bar { 2 } \mathrm { t r } [ Q ^ { 2 } ]$ , the centered reward contains three pieces,

$$
R _ { i } - \mu _ { R } = \sigma \| g \| u _ { i } - \frac { \sigma ^ { 2 } } { 2 } q _ { i } + \sigma _ { \xi } \xi _ { i } .\tag{41}
$$

These are mutually uncorrelated: $q _ { i }$ is even in $\epsilon _ { i }$ while $u _ { i }$ is odd, and $\xi _ { i }$ is independent of both. Consequently

$$
\sigma _ { R } ^ { 2 } = \sigma ^ { 2 } \| g \| ^ { 2 } + \frac { \sigma ^ { 4 } } { 2 } \mathrm { t r } [ Q ^ { 2 } ] + \sigma _ { \xi } ^ { 2 } .\tag{42}
$$

Signal. By equation 29,

$$
\mathbb { E } [ Z _ { i } u _ { i } ] = \frac { 1 } { \sigma _ { R } } \Big ( \sigma \| g \| \mathbb { E } [ u _ { i } ^ { 2 } ] - \frac { \sigma ^ { 2 } } { 2 } \mathbb { E } [ q _ { i } u _ { i } ] + \sigma _ { \xi } \mathbb { E } [ \xi _ { i } u _ { i } ] \Big ) = \frac { \sigma \| g \| } { \sigma _ { R } } ,\tag{43}
$$

since $q _ { i } u _ { i }$ is odd in $\epsilon _ { i }$ and $\xi _ { i }$ is independent of $u _ { i }$ . Hence, the signal magnitude is

signal:

$$
\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ] = \frac { \alpha \sigma \lVert g \rVert } { \sigma _ { R } } = \frac { \alpha \lVert g \rVert } { \sqrt { \lVert g \rVert ^ { 2 } + \frac { \sigma ^ { 2 } } { 2 } \operatorname { t r } [ Q ^ { 2 } ] + \frac { \sigma _ { \xi } ^ { 2 } } { \sigma ^ { 2 } } } } .\tag{44}
$$

We write $\rho : = \sigma \| g \| / \sigma _ { R }$ for this attenuation factor in general; it equals 1 and c in the two linear cases. The curvature fluctuation contributes nothing to the expected projection but enters $\sigma _ { R }$ in exactly the same way as observation noise. From the point of view of the z-score, it is an endogenous noise source, generated by the algorithm’s own exploration, that scales as $\sigma ^ { 4 }$ while the gradient term scales as $\sigma ^ { 2 }$

Noise. Because $q _ { i }$ depends on $\epsilon _ { i } ^ { \perp } , Z _ { i }$ is no longer independent of $\epsilon _ { i } ^ { \perp }$ , and we evaluate $\mathbb { E } [ Z _ { i } ^ { 2 } \lVert \epsilon _ { i } ^ { \perp } \rVert ^ { 2 } ] = \mathbb { E } [ Z _ { i } ^ { 2 } \lVert \bar { \epsilon } _ { i } \rVert ^ { 2 } ] - \mathbb { E } [ \bar { Z _ { i } ^ { 2 } } u _ { i } ^ { 2 } ]$ . Using the Gaussian moments $\mathbb { E } [ u _ { i } ^ { 2 } \lVert \epsilon _ { i } \rVert ^ { 2 } ] ~ = ~ d + 2$ $\mathbb { E } \{ q _ { i } ^ { 2 } \lVert \epsilon _ { i } \rVert ^ { 2 } \} = 2 ( d + 4 ) \operatorname { t r } [ Q ^ { 2 } ] , \mathbb { E } [ u _ { i } ^ { 4 } ] = 3$ , and $\mathbb { E } [ q _ { i } ^ { 2 } u _ { i } ^ { 2 } ] = 2 \operatorname { t r } [ Q ^ { 2 } ] + 8 \dot { g } ^ { \dagger } \dot { Q } ^ { 2 } \hat { g }$ , with all cross terms vanishing by parity,

$$
\mathbb { E } \big [ Z _ { i } ^ { 2 } \| \epsilon _ { i } \| ^ { 2 } \big ] = d + \frac { 2 \sigma ^ { 2 } \| g \| ^ { 2 } + 2 \sigma ^ { 4 } \operatorname { t r } [ Q ^ { 2 } ] } { \sigma _ { R } ^ { 2 } } ,\tag{45}
$$

$$
\mathbb { E } \big [ Z _ { i } ^ { 2 } u _ { i } ^ { 2 } \big ] \ = \ 1 + \frac { 2 { \sigma } ^ { 2 } \| g \| ^ { 2 } + 2 { \sigma } ^ { 4 } \hat { g } ^ { \top } { Q } ^ { 2 } \hat { g } } { { \sigma } _ { R } ^ { 2 } } .\tag{46}
$$

Subtracting and writing $\boldsymbol { P } _ { \perp } = I _ { d } - \hat { g } \hat { g } ^ { \intercal }$ , so that $\mathrm { t r } [ P _ { \perp } Q ^ { 2 } P _ { \perp } ] = \mathrm { t r } [ Q ^ { 2 } ] - \hat { g } ^ { \top } Q ^ { 2 } \hat { g }$ is the curvature energy orthogonal to the gradient,

noise:

$$
\sqrt { \mathbb { E } \| \Delta \theta _ { \perp } \| ^ { 2 } } = \alpha \sqrt { \frac { d - 1 } { N } } \sqrt { 1 + \frac { 2 \sigma ^ { 4 } \mathrm { t r } [ P _ { \perp } Q ^ { 2 } P _ { \perp } ] } { ( d - 1 ) \sigma _ { R } ^ { 2 } } } \approx \alpha \sqrt { \frac { d - 1 } { N } } ,\tag{47}
$$

where the curvature correction is bounded, $2 \sigma ^ { 4 } \mathrm { t r } [ P _ { \perp } Q ^ { 2 } P _ { \perp } ] / \sigma _ { R } ^ { 2 } \le 4$ , by equation 42. Curvature therefore adds at most an $O ( 1 / d )$ relative amount to the noise magnitude; its main effect on the off-gradient component is to redistribute it toward the high-curvature eigendirections of Q, as in the anisotropic covariance term $2 \alpha ^ { 2 } \sigma ^ { 4 } Q ^ { 2 } / ( N \sigma _ { R } ^ { 2 } )$ . The parallel coefficient has variance $\alpha ^ { 2 } \big ( \dot { \mathbb { E } } \big [ Z _ { i } ^ { 2 } u _ { i } ^ { 2 } \big ] -$ $\sigma ^ { 2 } \| g \| ^ { 2 } / \sigma _ { R } ^ { 2 } ) / N$ , which reduces to $\alpha ^ { 2 } ( 1 + c ^ { 2 } ) / N$ when $Q = 0$

Table 1: Per-update signal $\mathbb { E } [ \hat { g } ^ { \top } \Delta \theta ]$ , noise $\sqrt { \mathbb { E } \| \Delta \theta _ { \perp } \| ^ { 2 } } .$ , and expected reward change $\mathbb { E } [ \Delta R ]$ under the three landscape models, from equation 31–equation 33, equation 37–equation 40, and equation 44–equation 49. Throughout, $\rho = \sigma \| g \| / \sigma _ { R }$ is the gradient correlation of the population rewards, equal to 1 and c in the two linear cases; $\lambda _ { g } = \hat { g } ^ { \top } Q \hat { g } ;$ ; and $\delta _ { Q } = 2 \sigma ^ { 4 } \mathrm { t r } [ P _ { \perp } \mathrm { \dot { \cal Q } } ^ { \hat { 2 } } P _ { \perp } ] / ( ( d -$ $1 ) \sigma _ { R } ^ { 2 } ) \leq 4 / ( d - 1 )$ . The $O ( 1 / N )$ and $\operatorname { t r } [ Q ^ { 3 } ]$ terms of equation 49 are omitted.
<table><tr><td rowspan=1 colspan=1>Landscape</td><td rowspan=1 colspan=1>Signal</td><td rowspan=1 colspan=1>Noise</td><td rowspan=1 colspan=1>Expected reward change</td></tr><tr><td rowspan=1 colspan=1>Linear, exact rewards</td><td rowspan=1 colspan=1>α</td><td rowspan=1 colspan=1> $\alpha \sqrt { ( d - 1 ) / N }$ </td><td rowspan=1 colspan=1> $\alpha \lVert g \rVert$ </td></tr><tr><td rowspan=1 colspan=1>Linear, observation noise</td><td rowspan=1 colspan=1>αρ</td><td rowspan=1 colspan=1> $\alpha \sqrt { ( d - 1 ) / N }$ </td><td rowspan=1 colspan=1>αρ||g||</td></tr><tr><td rowspan=1 colspan=1>Quadratic, observation noise</td><td rowspan=1 colspan=1> $\alpha \rho$ </td><td rowspan=1 colspan=1> $\alpha \sqrt { ( d - 1 ) / N } ( 1 + \delta _ { Q } ) ^ { 1 / 2 }$ </td><td rowspan=1 colspan=1> $\overline { { \alpha \rho \| g \| - \frac { \alpha ^ { 2 } } { 2 N } \operatorname { t r } [ Q ] - \frac { \alpha ^ { 2 } \rho ^ { 2 } } { 2 } \lambda _ { g } } }$ </td></tr></table>

Reward change. On a quadratic landscape $\begin{array} { r } { \Delta R = g ^ { \top } \Delta \theta - \frac { 1 } { 2 } \Delta \theta ^ { \top } Q \Delta \theta } \end{array}$ , and the second term is where the orthogonal noise is finally charged. Since $\Delta R$ is quadratic in $\Delta \theta$ , its expectation needs only the first two moments of the update, $\begin{array} { r } { \tilde { \mathbb { E } } [ \Delta \theta ^ { \top } Q \Delta \theta ] = \mathrm { t r } [ \dot { Q } \mathrm { C o v } ( \Delta \theta ) ] + \mathbb { E } [ \Delta \theta ] ^ { \top } Q \mathbb { E } [ \Delta \theta ] } \end{array}$ . The same moment calculation that gives equation 45 gives the full covariance,

$$
\mathrm { C o v } [ \Delta \theta ] = \frac { \alpha ^ { 2 } } { N } I _ { d } + \frac { \alpha ^ { 2 } } { N \sigma _ { R } ^ { 2 } } \Big ( \sigma ^ { 2 } g g ^ { \top } + 2 \sigma ^ { 4 } Q ^ { 2 } \Big ) ,\tag{48}
$$

and hence, with $\lambda _ { g } : = \hat { g } ^ { \top } Q \hat { g }$ the curvature along the gradient,

$$
\mathbb { E } [ \Delta R ] = \underbrace { \alpha \rho \| g \| } _ { \mathrm { s i g n a l \underbrace { g a i n } } } - \underbrace { \frac { \alpha ^ { 2 } } { 2 N } \operatorname { t r } [ Q ] } _ { \mathrm { d i f f u s i o n \ t a x } } - \underbrace { \frac { \alpha ^ { 2 } \rho ^ { 2 } } { 2 } \Big ( 1 + \frac { 1 } { N } \Big ) \lambda _ { g } } _ { \mathrm { o v e r s h o o t } } - \underbrace { \frac { \alpha ^ { 2 } \sigma ^ { 4 } } { N \sigma _ { R } ^ { 2 } } \operatorname { t r } [ Q ^ { 3 } ] } _ { \mathrm { a n i s o t r o p i c } } .\tag{49}
$$

The first two terms dominate. The diffusion tax is the orthogonal noise, of squared length $\alpha ^ { 2 } d / N$ charged at the mean curvature tr $[ Q ] / d ;$ it is the population mean gap ${ \frac { \sigma ^ { 2 } } { 2 } } \operatorname { t r } [ Q ]$ with $\sigma ^ { 2 }$ replaced by $\alpha ^ { 2 } / N$ , so the parent pays each step the same curvature cost as a perturbation of scale $\alpha / \sqrt { N }$ . The overshoot term is the usual gradient-descent one and matters only when $\alpha \rho \lambda _ { g }$ is comparable to $\| g \| ;$ the last term is smaller still. Keeping signal and diffusion, a step improves the expected reward iff $\alpha < 2 N \rho \| g \| / \operatorname { t r } [ Q ]$ ], and the one-step gain is maximized at $\dot { \alpha _ { \mathrm { ~ } } } = \dot { N } \rho \lVert g \rVert / \operatorname { t r } [ Q ]$ , where it equals $N \rho ^ { 2 } \| g \| ^ { 2 } / ( 2 \operatorname { t r } [ Q ] ) \colon$ : population size buys improvement linearly, gradient correlation quadratically. Because the signal gain shrinks with ∥g∥ along a run while the diffusion tax does not, progress stalls when $\rho \| g \| = \bar { \alpha } \mathrm { t r } [ \bar { Q } ] / ( 2 N )$

Summary of the three cases. Table 1 collects the signal and noise magnitudes and the expected reward change for the three cases. In every case, the noise magnitude is $\alpha \sqrt { ( d - 1 ) / N }$ , exactly in the linear cases and up to a bounded $\dot { O ( 1 / d ) }$ relative correction in the quadratic case, and is insensitive to $\sigma .$ . The signal magnitude is α times a dimensionless attenuation factor $\sigma \| g \| / \sigma _ { R } \in$ [0, 1]: equal to 1 without observation noise, reduced by observation noise, and reduced further by the curvature contribution to $\sigma _ { R }$ on a quadratic landscape. Consistent with equation 24, the choice of σ does not change how far ES moves per iteration, only what fraction of that movement is directed, and the perturbation-scale problem reduces to maximizing the attenuation factor in equation 44. The reward-change column adds the one place where the noise is not free: on a curved landscape, the isotropic diffusion is taxed at the mean curvature, at a rate that falls as $1 / N$

## A.5 OPTIMAL PERTURBATION SCALE

The denominator of equation 44 makes the competition explicit: the observation-noise term $\sigma _ { \xi } ^ { 2 } / \sigma ^ { 2 }$ favors large σ, the curvature term $\frac { \sigma ^ { 2 } } { 2 } \ : \mathrm { t r } [ Q ^ { 2 } ]$ favors small σ, and $\| g \| ^ { 2 }$ is the irreducible part. Neither term is small in general. Although the per-direction curvature contribution $2 \sigma ^ { 4 } u ^ { \top } Q ^ { 2 } $ u is typically tiny for any single unit vector u, the quantity that enters $\sigma _ { R }$ is the full trace $\begin{array} { r } { \mathrm { t r } [ Q ^ { 2 } ] = \sum _ { k = 1 } ^ { d } \lambda _ { k } ^ { 2 } } \end{array}$ summed over all d eigendirections, and is extensive in the parameter dimension.

Holding $\alpha , g , Q$ and $\sigma _ { \xi }$ fixed, the signal equation 44 is maximized by minimizing $f ( { \boldsymbol { \sigma } } ^ { 2 } ) = { \sigma } _ { \xi } ^ { 2 } / { \sigma } ^ { 2 } +$ $\frac { \sigma ^ { 2 } } { 2 } \operatorname { t r } [ Q ^ { 2 } ]$ over $\sigma ^ { 2 }$ , which gives

$$
\boxed { \sigma _ { * } ^ { 4 } = \frac { 2 \sigma _ { \xi } ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } , \qquad \sigma _ { * } = \left( \frac { 2 \sigma _ { \xi } ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } \right) ^ { 1 / 4 } }\tag{50}
$$

$\operatorname { A t } \sigma ,$ <sub>∗</sub> the curvature-induced contribution to $\sigma _ { R } ^ { 2 }$ equals the exogenous one, $\begin{array} { r } { \frac { \sigma _ { * } ^ { 4 } } { 2 } \mathrm { t r } [ Q ^ { 2 } ] = \sigma _ { \xi } ^ { 2 } } \end{array}$ , so the population reward variance is

$$
\sigma _ { R } ^ { 2 } \big | _ { \sigma _ { * } } = \sigma _ { * } ^ { 2 } \| g \| ^ { 2 } + 2 \sigma _ { \xi } ^ { 2 } ,\tag{51}
$$

and the largest achievable signal is

$$
\operatorname* { m a x } _ { \sigma } \mathbb { E } [ \widehat { g } ^ { \top } \Delta \theta ] = \frac { \alpha \sigma _ { \ast } \| g \| } { \sqrt { \sigma _ { \ast } ^ { 2 } \| g \| ^ { 2 } + 2 \sigma _ { \xi } ^ { 2 } } } = \frac { \alpha \| g \| } { \sqrt { \| g \| ^ { 2 } + \sigma _ { \xi } \sqrt { 2 \operatorname { t r } [ Q ^ { 2 } ] } } } .\tag{52}
$$

Below $\sigma _ { * }$ the update is observation-noise limited; above it, curvature limited.

What the optimal population looks like. Two further statistics characterize the population at $\sigma _ { * }$ First, its rewards are correlated with the gradient at the parent by exactly the attenuation factor: since $\operatorname { C o v } ( u _ { i } , R _ { i } ) = \sigma \lVert g \rVert$ and $\mathrm { V a r } [ u _ { i } ] = 1$

$$
\rho _ { * } : = \left. \operatorname { C o r r } \left( \hat { g } ^ { \top } \epsilon _ { i } , R _ { i } \right) \right| _ { \sigma _ { * } } = \frac { \sigma _ { * } \| g \| } { \sqrt { \sigma _ { * } ^ { 2 } \| g \| ^ { 2 } + 2 \sigma _ { \xi } ^ { 2 } } } = \frac { \| g \| } { \sqrt { \| g \| ^ { 2 } + \sigma _ { \xi } \sqrt { 2 \operatorname { t r } [ Q ^ { 2 } ] } } } ,\tag{53}
$$

so that the expected signal equation 52 is $\alpha \rho _ { * }$ , and the fraction of population reward variance attributable to the gradient is $\bar { \rho } _ { * } ^ { 2 }$ , with the remainder split equally between observation noise and curvature. The second form makes the curvature dependence explicit: at the optimum, curvature never contributes more than $\sigma _ { \xi } ^ { 2 }$ to the population variance, but a larger tr $[ Q ^ { 2 } ]$ forces a smaller $\sigma _ { * }$ to hold it there, and the gradient variance $\sigma _ { * } ^ { 2 } \lVert g \rVert ^ { 2 }$ shrinks with it while the non-gradient variance stays at $2 \sigma _ { \xi } ^ { 2 }$ . Second, the population does not sit at the parent reward. Only the curvature term has nonzero mean, $\mathbb { E } [ \epsilon _ { i } ^ { \top } Q \epsilon _ { i } ] = \mathrm { t r } [ Q ]$ , so

$$
\mu _ { R } - { \cal R } ( \theta _ { t } ) = - \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } [ Q ] , \qquad \left. \mu _ { R } - { \cal R } ( \theta _ { t } ) \right| _ { \sigma _ { * } } = - \sigma _ { \xi } \sqrt { \frac { r _ { \mathrm { e f f } } } { 2 } } , \qquad r _ { \mathrm { e f f } } : = \frac { ( \mathrm { t r } [ Q ] ) ^ { 2 } } { \mathrm { t r } [ Q ^ { 2 } ] } ,\tag{54}
$$

where $r _ { \mathrm { e f f } } \in [ 1 , d ]$ is the participation ratio (effective rank) of the Hessian, equal to d for isotropic curvature and to r for a rank-r Hessian with equal eigenvalues. In units of the population spread equation 51,

$$
\frac { R ( \theta _ { t } ) - \mu _ { R } } { \sigma _ { R } } \Big | _ { \sigma _ { * } } = \sqrt { \frac { r _ { \mathrm { e f f } } } { 2 } } \cdot \frac { \sigma _ { \xi } } { \sqrt { \sigma _ { * } ^ { 2 } \| g \| ^ { 2 } + 2 \sigma _ { \xi } ^ { 2 } } } = \frac { \sqrt { r _ { \mathrm { e f f } } } } { 2 } \sqrt { 1 - \rho _ { * } ^ { 2 } } .\tag{55}
$$

Unless $r _ { \mathrm { e f f } } \lesssim 4$ , the optimally scaled population therefore lies several standard deviations below the parent model: nearly every perturbed model is worse than the unperturbed one. This is not a defect but the price of exploring at scale $\sigma _ { * }$ on a curved landscape, and it is the same for every member; the update depends only on the spread $\sigma _ { R }$ and the gradient share $\rho _ { * } ^ { 2 }$ , not on where the population sits. The gap in equation 55 grows as the gradient share falls, so along a training run it is expected to be small while $\left\| g \right\|$ is large and to become pronounced as the parent approaches a ridge. Because equation 17 is independent of the observation-noise scale while $\sigma _ { R }$ is not, the gap in $\sigma _ { R }$ units is less pronounced whenever $\sigma _ { \xi }$ is larger, for instance, when fewer prompts are averaged into each $R _ { i }$ Away from $\sigma _ { * }$ the same quantities read $\rho = \sigma \| g \| / \sigma _ { R }$ and $( R ( \theta _ { t } ) - \mu _ { R } ) / \sigma _ { R } = \sqrt { r _ { \mathrm { e f f } } \varphi / 2 }$ with $\begin{array} { r } { \varphi = \frac { \sigma ^ { 4 } } { 2 } \operatorname { t r } [ Q ^ { 2 } ] / \sigma _ { R } ^ { 2 } } \end{array}$ the curvature share of the population variance, so a measured gap together with an estimate of $\varphi$ yields $r _ { \mathrm { e f f } }$ directly.

Three remarks on equation 50. First, the dependence on the noise level is weak: $\sigma _ { * }$ grows only as $\sigma _ { \xi } ^ { 1 / 2 }$ , so even a large increase in observation noise justifies only a modest increase in the perturbation scale. Second, the dependence on curvature is through $\operatorname { t r } [ Q ^ { 2 } ]$ , which grows with $d ;$ for a billionparameter model, even small typical curvatures accumulate to a large trace, which is the familiar dimension penalty of finite-difference estimators appearing inside the z-score normalization. Third, the optimal signal equation 52 degrades with the geometric mean $\sigma _ { \xi } \sqrt { \mathrm { t r } [ Q ^ { 2 } ] }$ rather than with either quantity alone: noise and curvature are not separately harmful but jointly so, since one can always be traded against the other by moving σ.

## B DERIVATION OF THE UPDATE-NORM MOMENTS

This appendix derives equation 24–equation 25. In general, the weights depend on the perturbations: $Z _ { i }$ is a function of the reward at $\theta _ { t } + \sigma \epsilon _ { i }$ , and hence of $\epsilon _ { i }$ itself. What can be said without a landscape model is that each $Z _ { i }$ depends only on its own perturbation (and its own observation noise), so the products $Z _ { i } \epsilon _ { i }$ are identically distributed across i and, under the population-statistics simplification, independent. Write

$$
m : = \mathbb { E } [ Z _ { i } \epsilon _ { i } ] \in \mathbb { R } ^ { d } , \qquad \kappa : = \mathbb { E } \big [ Z _ { i } ^ { 2 } \lVert \epsilon _ { i } \rVert ^ { 2 } \big ] - d\tag{56}
$$

for the two landscape-dependent statistics of a single weighted perturbation. The first is the mean weighted perturbation, which is the direction the update drifts in; the second measures how much the weighting inflates the squared length of a perturbation beyond its unweighted value $d .$ Expanding $\| \Delta \bar { \theta } \| ^ { 2 }$ into diagonal and cross terms and taking expectations,

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \| \Delta \theta \| ^ { 2 } = } & { \displaystyle \frac { \alpha ^ { 2 } } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \mathbb { E } \big [ Z _ { i } ^ { 2 } \| \epsilon _ { i } \| ^ { 2 } \big ] + \frac { \alpha ^ { 2 } } { N ^ { 2 } } \sum _ { i \neq j } \mathbb { E } \big [ Z _ { i } \epsilon _ { i } \big ] ^ { \top } \mathbb { E } \big [ Z _ { j } \epsilon _ { j } \big ] } \\ { = } & { \displaystyle \frac { \alpha ^ { 2 } d } { N } + \frac { \alpha ^ { 2 } \kappa } { N } + \alpha ^ { 2 } \Big ( 1 - \frac { 1 } { N } \Big ) \| m \| ^ { 2 } . } \end{array}\tag{57}
$$

This identity is exact. The drift term is bounded independently of the landscape: for any unit vector v, Cauchy–Schwarz gives $( v ^ { \top } m ) ^ { 2 } = \mathbb { E } [ Z _ { i } v ^ { \top } \epsilon _ { i } ] ^ { 2 } \ \dot { \leq } \ \mathbb { E } [ Z _ { i } ^ { 2 } ] \ \dot { \mathbb { E } } [ ( v ^ { \top } \epsilon _ { i } ) ^ { 2 } ] = 1$ , so $\| m \| \leq 1$ and the drift contributes at most $\alpha ^ { 2 }$ . The inflation κ does depend on the landscape; the derivations below evaluate it for linear and quadratic rewards and find $0 \leq \kappa \leq 4$ . Both corrections are therefore smaller than the leading term $\alpha ^ { 2 } d / N$ by a factor of order $N / d \colon$

$$
\mathbb { E } \| \Delta \theta \| ^ { 2 } = \frac { \alpha ^ { 2 } d } { N } \Big ( 1 + O \big ( N / d \big ) \Big ) .\tag{58}
$$

For the variance, expand $\| \Delta \theta \| ^ { 2 }$ the same way. The cross terms $Z _ { i } Z _ { j } \epsilon _ { i } ^ { \top } \epsilon _ { j }$ for $i \neq j$ are zero-mean up to the drift, and each has variance $\mathbb { E } [ Z _ { i } ^ { 2 } Z _ { i } ^ { 2 } ( \epsilon _ { i } ^ { \top } \epsilon _ { j } ) ^ { 2 } ] = d \left( 1 + O ( 1 / \bar { d } ) \right)$ ; there are $N ( N { - } 1 )$ of them and they are pairwise uncorrelated up to $O ( \| m \| ^ { 2 } )$ terms. The diagonal terms $Z _ { i } ^ { 2 } \| \epsilon _ { i } \| ^ { 2 }$ fluctuate with variance $O ( d )$ each, but there are only $N$ of them and, under the exact constraint equation 23, their $O ( d )$ fluctuations cancel against those of the cross terms. The result is

$$
\mathrm { V a r } \| \Delta \theta \| ^ { 2 } = \frac { 2 \alpha ^ { 4 } d } { N ^ { 2 } } \Big ( 1 + { \cal O } \big ( N / d \big ) \Big ) , \qquad \frac { \sqrt { \mathrm { V a r } } \| \Delta \theta \| ^ { 2 } } { \mathbb { E } \| \Delta \theta \| ^ { 2 } } = \sqrt { \frac { 2 } { d } } \Big ( 1 + { \cal O } \big ( N / d \big ) \Big ) ,\tag{59}
$$

the same leading mean and variance as $\textstyle { \frac { \alpha ^ { 2 } } { N } } \chi _ { d } ^ { 2 } .$ , the law that equation 1 would follow if the weights were independent of the perturbations. The coefficient of variation $\sqrt { 2 / d }$ is what makes $\sqrt { \mathbb { E } \lVert \Delta \theta \rVert ^ { 2 } }$ an accurate proxy for the typical length: the two agree to relative order $1 / d . ^ { 1 }$