# Probabilistic Circuits as Reasoning Machines in Artificial Intelligence (Part I)

![](images/a3208eb0c90660dc57f0fa65055672df86bef371650c3d600a602876dca1a97b.jpg)

Robert Peharz

Institute of Machine Learning and Neural Computation Graz University of Technology

Cumulative Habilitation Thesis to obtain the Venia Docendi in Applied Computer Science

August 2026

## Acknowledgements

The people who need to be mentioned foremost are my lovely family, who have patiently endured, sometimes enjoyed, and always supported my academic endeavours: Thank you, Elisabeth, Benjamin, Raffael, and Hannah.

This thesis is about probabilistic circuits and tractable probabilistic models. We are a small but growing community, continuously striving to promote and inspire enthusiasm for our approach to AI and probabilistic modeling. It has been a fun ride, and I hope it continues to be for decades (and centuries) to come. It wouldn’t be such a great pleasure to work in such an exciting field without the right allies and peers, some of whom are (in alphabetical order): Cory Butz, Cassio de Campos, YooJung Choi, Alvaro Correia, Nicola Di Mauro, Floriana Esposito, Gennaro Gala, Hong Ge, Robert Gens, Zoubin Ghahramani, Kristian Kersting, Steven Lang, Thomas Liebig, Lorenzo Loconte, Alejandro Molina, Martin Mundt, Jhonatan Oliveira, Franz Pernkopf, Pascal Poupart, Erik Quaeghebeur, Carl Rasmussen, Xiaoting Shao, Karl Stelzner, Pranav Subramani, PingLiang Tan, Martin Trapp, Isabel Valera, Guy Van den Broeck, Fabrizio Ventola, Antonio Vergari, Thomas Wedenig, and Yang Yang.

Furthermore, I would like to thank the Austrian Science Fund (FWF) for supporting my work in the context of the Cluster of Excellence “Bilateral AI” (10.55776/COE12).

## Abstract

This cumulative habilitation thesis studies probabilistic circuits (PCs) as a powerful and tractable framework for reasoning and learning under uncertainty in artificial intelligence (AI). It first advocates for probability as a core language for AI, emphasizing its connections to logic and information theory; the conceptual simplicity of probabilistic reasoning—based primarily on the sum and product rules; the parallels between probabilistic inference and human cognition; and the role of probability in optimal decision making. However, probability also faces significant computational challenges, as probabilistic inference is NP-hard in almost all probabilistic models. PCs address these challenges through structural constraints that ensure exact computation of a wide range of inference queries in polynomial time, such as marginals, conditionals, most probable explanations, expectations, and more advanced inference tasks. This thesis synthesizes a decade of research across foundations, algorithmic developments, and empirical validation of PCs. Key contributions highlighted in this work are foundational theory of PCs, Bayesian approaches for learning PCs, scalable implementations and integration with deep learning, hybrid models that combine PCs with intractable models, and connections with symbolic machine learning paradigms.

## Table of contents

I Probabilistic Reasoning and AI 1   
1 Introduction 3   
1.1 What is AI (roughly)? 3   
1.2 The Role of Probability 5   
1.3 Why Probabilistic Circuits? 8   
2 Basic Probability Theory 11   
2.1 Measurable Space 11   
2.1.1 Events are Sets 12   
2.2 Probability Space 14   
2.2.1 Why Sigma-Algebras? 15   
2.3 Interpretations of Probability 16   
2.4 Fundamental Laws of Probability 18   
2.4.1 Law of Total Probability (The Sum Rule) 19   
2.4.2 Conditional Probability (The Product Rule) 19   
2.4.3 Bayes’ Rule (The Rule of Inverse Probability) . . 21   
2.4.4 (Conditional) Independence 22   
2.5 Random Variables 23   
2.5.1 Distribution of a Random Variable 24   
2.5.2 Distribution Functions 24   
2.5.3 Multivariate Random Variables and Joint Distributions 26   
Probabilistic Inference 29   
3.1 Marginalization (The Sum Rule) 29   
3.2 Conditioning (The Product Rule) 31   
3.3 The Chain Rule 33   
3.4 Bayes Rule (Law of Inverse Probability) 35   
3.5 (Conditional) Independence 35   
3.6 Expectations . 36   
3.7 Most Probable Explanation 39   
3.8 Decision Theory . 39   
4 State of The Art 45   
5 Probabilistic Circuits 49   
5.1 Basic Definitions of Probabilistic Circuits 51   
5.1.1 Decomposability and Smoothness 53   
5.1.2 PCs as Hierarchical Mixtures 54   
5.2 Marginalization 56   
5.3 Conditioning 58   
5.4 Expectations and Covariances 61   
5.5 Determinism and Most Probable Explanation 63   
5.6 Further Structural Properties and Tractable Inferences . 64   
5.7 Overview of Contributions 65   
5.7.1 Foundations and Theory of PCs 66   
5.7.2 Bayesian Approaches with PCs 67   
5.7.3 Deep Learning and PCs 67   
5.7.4 Hybrid Models and Inference 69   
5.7.5 PCs and Symbolic Models 71   
References 75

Part I

Probabilistic Reasoning and AI

## Chapter 1

## Introduction

In the last one or two decades we have witnessed a remarkable explosion of the fields of artificial intelligence (AI) and machine learning (ML). As research fields, AI and ML have gained tremendous momentum, reflected by an about tenfold increase of papers submitted to leading conferences over the last 10 years. In the digital industries ML is widely applied, ranging from more “classic” applications—such as spam mail detection, optical character recognition, machine translation, computer vision, and recommender systems—to more recent creative applications fuelled by advances in generative modelling, as exemplified by diffusion-based models for images and the prominent ChatGPT system. More generally, AI and ML are perceived as integral parts of the so-called “fourth industrial revolution,” promising a massive acceleration in automation of cognitive tasks and a tight integration of the physical and digital worlds.

## 1.1 What is AI (roughly)?

While being an active and trendy technology, it is hard to fully specify what AI actually is. According to “Artificial Intelligence: A Modern Approach”, the standard textbook by Russell and Norvig [53], we can organize AI into four categories, spanned by two dichotomies: “human-oriented vs. rational AI” and “thinking vs. acting”. Human-oriented AI strives to imitate abilities of human intelligence, while rational AI formalizes intelligence via some well-principled measure of performance. While there is overlap between the human-oriented and rational approaches to AI, one can think of rational AIs which are rather distinct from human intelligence, and, vice versa, rational approaches might be insufficient to explain or imitate human intelligence. Thus, it makes sense to distinct these two styles of AI. This thesis is located in the “rational AI” segment.

The second dichotomy—“thinking vs. acting”—distinguishes more “internal” abilities such as reasoning and cognition from more “external” abilities such as acting in an environment and making good decisions. This distinction also makes sense, even though “acting” surely requires “thinking” to a certain extent.

While defining AI in a precise way poses some challenges, we can attempt to specify some necessary features or desiderata which any AI system needs to display to a certain degree:

• Knowledge Representation and Modelling. It is a save to assume that any AI needs to have the ability to store information. More generally, we might assume a certain level of structural organization of information—broadly denoted as knowledge representation—facilitating subsequent tasks of the AI system. Perhaps synonymously to knowledge representation we might also say that an intelligent system needs the ability to represent a model of the world in which it is embedded. In a generous interpretation, knowledge representation and models can take various forms, ranging from explicit and formal representations, such as axiomatic systems, to implicit ones such as information stored in neural network weights.

• Reasoning. Additionally, an AI system will require the ability to reason, i.e., to do something with the information stored and process it into “new” insights. The reasoning machinery will in general depend on the type of knowledge representation. A classical example is propositional logic, which would represent knowledge as an axiomatic system together with known facts and use deductive reasoning to derive “new” facts. A classical illustration is the propositional knowledge base

Socrates is human

All humans are mortal

And the use of the modus ponens to conclude that

Socrates is mortal

But also common (systems of) equations such as $E = m c ^ { 2 }$ might be interpreted as a representation of knowledge—relating the three quantities E, m and c—and algebraic manipulation as a reasoning process allowing us, for example, to conclude that ${ \frac { E } { m } } = c ^ { 2 }$

These two requirements, knowledge representation and reasoning, are arguably necessary features for any AI system. They are probably not sufficient, as they are also fulfilled on a small scale by rather simple systems such as pocket calculators, which are not commonly perceived as “intelligent.” However, some logic-based AI systems in the early area of AI would also be confined to these two aspects, albeit with a high degree of complexity.

Besides knowledge representation and reasoning one would also include the desiderata of machine learning, the nowadays most prominent branch of AI:

• Learning. According to Simon [58] learning denotes adaptive changes in a system, enabling to perform tasks more effectively and efficiently. Learning thus includes a certain dynamic element in AI, allowing to automatically modify, extend, or complement the knowledge representation mentioned above. This does not mean that learning happens necessarily online (over the lifetime of the AI)—in contrary, many machine learning systems nowadays learn in an offline fashion. Rather, learning represents a method to incorporate and exploit information which was not present at design time of the AI, regardless of whether learning happens online or offline.

The desiderata knowledge representation, reasoning, and learning may serve as a minimalist definition of rational AI or, at the very least, describe its necessary features. Russell and Norvig additionally mention the desiderata perception (computer vision, computer audition), robotics, and natural language processing. In my own opinion, these are natural requirements but are more features describing human-oriented AI.

This habilitation thesis describes the essence of my work on the “minimalist core notion of rational AI” described above, putting knowledge representation, learning and reasoning in the center of attention.

## 1.2 The Role of Probability

A central working hypothesis in this work is to recognize probability as a rigorous and consistent tool for reasoning and learning. In the following, I briefly summarize arguments for why probability is an excellent core language for AI.

## Simplicity

Similar as the reasoning tools sketched above, propositional logic and (systems of) equations, probabilistic reasoning is based on some few simple mathematical principles. Arguably, simplicity is key, when it comes to constructing a long-lasting theory of (artificial) intelligence.

Probabilistic reasoning can essentially be reduced to two simple rules, the sum rule and the product rule [21]. Semantically, the sum rule, also called marginalization rule, allows us to consistently incorporate unknown facts in our reasoning process, while incorporating the uncertainty associated with these unknown facts. The semantics of the product rule, also called conditioning rule, is to inject information in our reasoning process, that is, basing our reasoning on observed facts. Any probabilistic reasoning task, denoted as probabilistic inference, can then be reduced to a well-prescribed cascade of these two rule, allowing us to propagate information from observations to predictions while consistently treating the uncertainties described by our model.

Note that simplicity means here conceptual simplicity, not computational simplicity. Indeed, probabilistic inference (applying the sum and product rule) is in general a “computational nightmare” as it requires summations or integrations which in general scale exponentially in the problem dimensionality. Working around this computational nightmare is a core goal of probabilistic circuits, and hence, of this thesis.

## Decision Making

Probabilistic reasoning is tightly connected with decision making under uncertainty. A classical line of argument are Dutch book arguments, which assert that the betting strategy of an agent (human or artificial) must correspond to some probability distribution over gambling outcomes. Otherwise, a combination of bets, a so-called Dutch book, can be constructed which the agent will accept according to its strategy, which, however, will incur guaranteed total loss. Dutch book arguments do not discuss optimality of the agent’s betting strategy, but rather are a sanity check, arguing which kind of strategies are in principle admissible, namely, those based on probabilistic reasoning.

More generally, probability lies at the heart of decision theory [4] which defines the notion of optimal decision in mere probabilistic terms. Specifically, an optimal decision regarding some target quantity Y is defined as the minimizer of the expected loss, that is, the expectation of a some loss function $\ell ( y ^ { \prime } , y )$ , expressing the regret when predicting y<sup>′</sup> while the true value of Y is y. In that sense, knowledge of the true underlying joint distribution together with rigorous probabilistic inference inevitably leads to optimal decision, in the sense of minimal expected (or long-term) regret under this distribution.

## Qualitative Similarities to Common Sense Reasoning

Furthermore, probabilistic reasoning shares many qualitative features with (human) common sense reasoning, as discussed by Pearl in his classical book “Probabilistic reasoning in intelligent systems: networks of plausible inference” [39].

First, probabilistic reasoning features non-monotonic reasoning, i.e. the effect that the degree of plausibility is not a monotonic function of the amount of available information. For example, common sense judges the statement “Tim can $\mathrm { { f l y } ^ { \mathrm { { \prime } } } }$ as of low plausibility, but learning the information that “Tim is a bird” would lead to an increase of this plausibility. If, however, one learns the additional piece of information that “Tim is sick” one would consequently lower this plausibility again. This mechanism of non-monotonic beliefupdate, i.e. that the plausibility of a statement is not monotonously increasing or decreasing with the amount of information available, is naturally reflected in probabilistic reasoning. Specifically, conditioning (the product rule) displays this feature, as conditional distributions might be arbitrarily different depending on the conditioned information. Several classical AI systems for reasoning under uncertainty were lacking the ability of non-monotonous reasoning [39].

Another canonical example where probabilistic reasoning parallels human cognitive reasoning is the explaining away phenomenon: two a-priori unrelated causes become correlated upon observing a common effect. For example, a PhD student in ML might observe that (A) their experiments deliver dissatisfying results. Two reasons for this might be (B) the whole theory and foundations of the work are flawed, or (C) there is a bug in the implementation. A priori, statements B and C can be assumed to be uncorrelated. However, when A is observed, then common sense reasoning suggests that learning C makes B less plausible (bringing some relief to the PhD student), showcasing that B and C have become correlated upon observing A. The explaining away effect, as well as other common sense patterns determining relevance among quantities, is well reflected in probabilistic reasoning [39].

## Bayesianism and Consistency with Logic

Unlike the classical “hard” reasoning tools, such as logic and systems of equations, probability adds a notion of uncertainty which is a pressing requirement imposed by the real world. A nice feature of probability is that, when restricted to the extreme probabilities 0 and 1, probability emulates classical propositional logic. Moreover, if one accepts the assumptions of Cox’s theorem, then probability is the only sound extension of classical logic to continuous truth or plausibility values, so that one is de facto forced to accept probability in order to be compatible with logic [25].

Cox’s theorem is often cited as a theoretical underpinning of the Bayesian approach, which naturally unifies reasoning and learning. The Bayesian perspective allows us to put probabilities on essentially any uncertain quantities of interest, not only those which can arguably be repeated, as required by the classical frequentist notion of probability.<sup>1</sup> Probabilistic modelling is then but the task of constructing a joint distribution expressing the dependencies and uncertainties between all involved model parameters<sup>2</sup> and the data. The typical modelling approach is to specify a prior distribution over parameters and a likelihood relating parameters with the data. The division into prior and likelihood, however, is usually a rather arbitrary choice [20]. Then, in the Bayesian approach, learning is but probabilistic reasoning (posterior inference) about parameters given data.

## Information Theory

Information theory was originally designed as a tool of communication engineering [55], but quickly found applications in many other disciplines, such as biology, psychology, and fundamental physics [24]. The basic insight is to quantify the information content of an event by the negative logarithm ofits probability. As information content and probabilities are merely non-linear but monotonous transformations (log vs. exp) of each other, they are equivalent descriptions of the same quantity. This equivalence is satisfying, as it indicates that (logs of) probabilities are merely measuring information, or actually lack thereof, serving as a natural description of uncertainty.

Here, the reason why the information is missing is a subordinate matter and one can avoid vague talk about “true” randomness. Otherwise, one would need to worry whether an opponent’s poker cards are truly random (they are not for them, and I could simply grab the cards and check myself), or the outcome of a game of roulette (which, with sufficiently accurate physical measurement could be predicted arbitrarily well), or the pseudo-random behaviour of a chaotic system which seems random but is actually perfectly deterministic, or the effects on the level of particle physics (Heisenberg’s uncertainty principle introduces a fundamental limit for physical beings). While in these examples that nature of “randomness” seems quite different, this does not matter for modelling and reasoning purposes—one simply quantifies uncertainty, or lack of information, with (log) probabilities. As the information state is generally agent-dependent, this line of argument is reinforcing a Bayesian approach to probabilistic reasoning [34].

This thesis puts forward the working hypothesis that probability is an excellent core language for AI, due to its conceptual simplicity (and hence the possibility to be automated), its ability to express and process uncertainty, as well as many compelling connections to logic, human reasoning, decision theory and information theory.

## 1.3 Why Probabilistic Circuits?

While probability is arguably an excellent learning and reasoning language for AI, it poses substantial computational challenges. For example, probabilistic graphical models (PGM)

[28], a prominent framework for probabilistic AI, are decorated with many hardness results concerning inference and learning. Often, it is relatively easy to construct a PGM accurately reflecting the domain of interest, but in which performing sound probabilistic reasoning (via the sum and product rule) has exponential computational cost in the so-called tree-width of the PGM. Roughly speaking, PGMs and other types of probabilistic models might be easy to represent but “hard to run”.

This is in stark contrast to, say, neural networks which are constructed in a way that inference remains easy. Of course, the shortcut here is that neural networks do not perform fully fledged probabilistic reasoning, but only learn a single (or few) reasoning task(s). For example, if one learns a neural classifier predicting class label Y from a feature vector X by minimizing cross-entropy, one really just estimates the conditional distribution $p _ { Y | \pmb { X } }$ with maximum likelihood, i.e. minimizing the Kullback-Leibler divergence between the model distribution and the empirical data distribution. The very same network, however, would have trouble to predict a portion of the features $\pmb { X } ^ { \prime } \subset \pmb { X }$ given the rest of the features $\pmb { X } ^ { \prime \prime } = \pmb { X } \backslash \pmb { X } ^ { \prime }$ while ignoring the label Y altogether. Such a predictor cannot be derived in a meaningful way from $p _ { Y | \pmb { X } }$ . Rather, one requires a model over the whole joint $p _ { Y , { \pmb X } }$ , which would allow to compute the desired conditional distribution $p _ { { \pmb X } ^ { \prime } | { \pmb X } ^ { \prime \prime } }$ . Of course, there are ample neural networks representations for whole joint distributions, generally known as (deep) generative models [65]. However, similar as for PGMs, probabilistic inference is generally hard in these models.

Can we get best of both worlds? That is, can we design models that—like PGMs and other probabilistic models—represent full joint distributions over the domain of interest, while at the same time allowing to perform flexible inference with the same ease and computational complexity as neural networks? The answer is yes, with probabilistic circuits (PCs)! One way to look at PCs is that they are a kind of hybrid model between PGMs and neural networks. Like PGMs—and unlike “standard” neural nets—they are representations of joint distributions. Unlike PGMs they are able to perform exact and efficient inference, that is, they compute the sum and product rule—as well as many other inference queries—in essentially the same manner and with the same computational cost as neural networks, i.e. by performing one or a few network passes.

The key term here is structure: While PCs can be interpreted as neural networks, they need to obey particular constraints guaranteeing that the desired reasoning operations remain tractable. Interestingly, these structural constraints appear at many places in literature, within the probabilistic modelling setting but also in related disciplines such as logic [16] and databases [36]. Within the probabilistic reasoning domain, these structural patterns appear over and over again, be it in exact inference algorithms such as variable elimination [15], the junction tree algorithm [28], and and/or search spaces [17] or tractable probabilistic models such as arithmetic circuits [14], sum-product networks [46], cutset networks [48], and probabilistic sentential decision diagrams [27]. Really, PCs do not represent a novel model class. Rather, they aim to consolidate these existing lines of research under one umbrella and using one syntax [73, 9] and systematically delineate their differences in terms of structural constraints and corresponding tractable reasoning routines [71]. In a nutshell, PCs are a proposed lingua franca for tractable probabilistic modelling.

Evidently, PCs don’t offer a free lunch nor do they possess a magical secret sauce to make hard probabilistic inference problems all of the sudden easy. At their core, PCs are models too and serve as an approximations of reality. Their structural constraints ensure that the primary goal—probabilistic inference—remains tractable, but this comes at the cost of limiting the choice of neural network structures. These constraints naturally impose limits on the expressive efficiency of PCs, meaning that some distributions, which can be compactly described as an intractable model, may require exponentially larger representations as a PC. Therefore, the key strategy in PCs is to find the best possible approximation of the data distribution under the constraint that inference remains tractable.

This cumulative habilitation thesis describes my contributions to the field of PCs over the last ten years and is organized as follows: In Chapter 2, I introduce basic concepts of probability theory and probabilistic reasoning. In Chapter 3, I illustrate the idea of probability as elegant and widely applicable reasoning tool on the level of distribution functions. Chapter 4 gives a concise overview of state-of-the-art in probabilistic modeling. In Chapter 5, I introduce the framework of PCs and illustrate the contributions I made to this research field. The relevant publications are listed in Part II.

## Chapter 2

## Basic Probability Theory

The basic concept in probability is the probability space. The definition of probability space is somewhat abstract and practitioners like to quickly forget it as soon as they get exposed to more intuitive notions like random variables and probability densities. However, it pays off to embrace such foundational concepts to a certain extent. Probability as a mathematical discipline is relatively young, only dating back about 300-400 years, and a firm and concise description of probability was not immediate. Today we have arrived at a compact and rigorous formalisation of probability—the probability space—which fits on roughly half a page and accurately describes the nature of probability.

The probability space serves as an insurance and contract: probabilistic methods and models can get fairly abstract and complex and one might get confused about whether what one is doing is still sound. The rule of the game is: can we reduce it back to some probability space (or at least prove its existence)? If yes, we can call it probability. If not, it should be called something else. In other words, the probability space is the “Linux kernel” of probabilistic AI: we usually don’t need to work with it, but knowing how it works will make us better Linux users.

In this chapter I just introduce the rudimentary ideas of measure-theoretic probability theory, in order to establish its role as central reasoning instrument in AI. A rigorous treatment can be found for example in [5, 51].

## 2.1 Measurable Space

The probability space is based on a structured set denoted as measurable space.

Definition 1 (Measurable Space). Let Ω be any non-empty set and Σ a sigma-algebra over Ω, i.e. Σ is a set of sub-sets $A \subseteq \Omega$ . In the following, let ${ \bar { A } } : = \Omega \backslash A$ denote the complementary set of A. The sigma-algebra needs to satisfy the following properties:

$\Omega \in \Sigma$

$A \in \Sigma$ implies that $\bar { A } \in \Sigma$

(closed under complement)

• for any countable collection of events $\{ A _ { i } \} _ { i }$ it holds that $\textstyle \bigcup _ { i } A _ { i } \in \Sigma .$ (closed under countable union)

The tuple (Ω,Σ) is a measurable space.

Basically, a measurable space is just a non-empty set (Ω) plus some structure represented as a collection of sub-sets (Σ). The set Ω is called the sample space and the elements $\omega \in \Omega$ are called atomic events. The basic picture in probability is that an element of ω is selected at random from Ω, where “at random” really just means that we are lacking the information which ω is selected. The basic idea of probability is to mathematically model and treat this lack of information.

It is important to note that in general we should not assume that an ω is selected repeatedly from Ω, as it is assumed in many classical statistics texts, but rather that exactly one ω is selected, and we do not know which one. The point here is that assuming repeated selection is unnecessarily limiting the theory, and assuming that exactly one ω is selected is in fact the more general picture. Repeated selection of ω is easily subsumed, by “copying the sample space”, i.e. one defines $\Omega _ { i } = \Omega$ for $i = \{ 1 , 2 , \dots \}$ and constructs a new sample space as the Cartesian product of the copies $\Omega ^ { \prime } : = \times , \Omega _ { i }$ . In this duplicated sample space $\Omega ^ { \prime }$ , the atomic events are now sequences of draws ${ \pmb { \omega } } ^ { \prime } = ( { \pmb { \omega } } _ { 1 } , { \pmb { \omega } } _ { 2 } , \ldots )$ . This construction is also valid for infinite sequences, i.e. when i ranges over the entire N.

Besides the sample space Ω, the probability space contains a sigma-algebra Σ, a system of subsets A of $\Omega ,$ which needs to obey the three rules in Definition 1. It follows automatically from these rules, that also (i) $\varnothing \in \Sigma$ and (ii) any countable intersection $\cap _ { i } A _ { i }$ of $A _ { i } \in \Sigma$ is again in Σ, i.e. the sigma-algebra is also closed under countable intersection.

## 2.1.1 Events are Sets

The elements of Σ are called events. This nomenclature hints that while the atomic events ω represent the “finest resolution” of the universe Ω, the sigma-algebra Σ contains events at a more “macroscopic level”. Example 1 illustrates this concept.

Example 1 (Events for a die throw). When modelling a die, a suitable sample space is $\Omega = \{ 1 , 2 , 3 , 4 , 5 , 6 \}$ representing all six possible outcomes of the die throw. We might decide that this is too fine-grained for our purposes and that—for whatever reason—we are just interested in two events, namely (i) whether the die shows a prime number and (ii) whether it shows an odd number.

The die shows a prime number if it turns out that ω is contained in the set $A _ { p } = \{ 2 , 3 , 5 \}$ and it shows an odd number ifω is in $A _ { o } = \{ 1 , 3 , 5 \}$ . One can convince oneself that— as soon as the sample space Ω is fixed—indeed any thinkable event can be encoded by whether or not $\omega \in A$ where A is some subset of Ω.

We can find the most concise sigma-algebra containing $A _ { p }$ and $A _ { o }$ by starting with ${ \Sigma } _ { 0 } = \{ A _ { p } , A _ { o } , \Omega \}$ and iterating

$$
\Sigma _ { i + 1 }  \Sigma _ { i } \cup \{ \bar { A } : A \in \Sigma _ { i } \} \cup \{ A \cup B : A , B \in \Sigma _ { i } \}
$$

that is, we repeatedly include all complements and pairwise unions of events. This process will—for any finite $\Omega -$ —converge and deliver the most concise sigma-algebra containing the desired events. Here we get

$$
\Sigma = \{ \emptyset , \{ 1 \} , \{ 2 \} , \{ 1 , 2 \} , \{ 3 , 5 \} , \{ 4 , 6 \} ,
$$

$$
\{ 1 , 3 , 5 \} , \{ 1 , 4 , 6 \} , \{ 2 , 3 , 5 \} , \{ 2 , 4 , 6 \} ,
$$

$$
\{ 1 , 2 , 3 , 5 \} , \{ 1 , 2 , 4 , 6 \} , \{ 3 , 4 , 5 , 6 \} ,
$$

$$
\{ 1 , 3 , 4 , 5 , 6 \} , \{ 2 , 3 , 4 , 5 , 6 \} , \Omega \}
$$

We see that this sigma-algebra does not contain all of the singleton atomic events—it contains {1} and {2}, butfor example {3} and {4} are not contained in Σ. That is, Σ represents a more “macroscopic” view of Ω and contains only $A _ { p }$ and $A _ { o }$ plus all the necessary events to get a valid sigma-algebra.

After fixing Ω, any thinkable event can be described as whether ω is contained in some prescribed set $A \subseteq \Omega$ and hence any such subset A deserves the name “event”. Thus, we will sometimes say things like “event A happens,” meaning really that $\omega \in A$ . The basic role of Σ is to collect the events we are interested in. Moreover, the structure of the sigma-algebra just ensures that we have a closed logical system [25]:

• The sigma-algebra needs to be closed under complement, which means that for any event we consider, we also must consider the opposite, i.e., that the event does not happen. This simply amounts to logical negation.

• The sigma-algebra needs to be closed under countable union, i.e., for any countable collection of events we need also to consider the event that any of them happens. This amounts to logical or.

• It follows automatically due to De Morgan’s laws that the sigma-algebra is closed under countable intersection, i.e., for any countable collection of events we need also to consider the event that all of them happen. This amounts to logical and.

• Further logical connectives follow, such as closedness under symmetric set difference (logical xor).

We can now define the notion of probability space.

## 2.2 Probability Space

While a measurable space is a non-empty set Ω together with a logical structure Σ of sub-sets, a probability space adds the probability measure, a function defined on Σ.

Definition 2 (Probability Space). Let (Ω,Σ) be a measurable space. Let P be a function $\Sigma \mapsto [ 0 , 1 ]$ mapping events to real numbers between 0 and 1, with the properties

$\mathbb { P } ( \Omega ) = 1$

• for any countable collection of disjoint events $\{ A _ { i } \} _ { i }$ it holds that P $\begin{array} { r } { ( \bigcup _ { i } A _ { i } ) = \sum _ { i } \operatorname { P } ( A _ { i } ) } \end{array}$ (countable additivity)

Such function P is called a probability measure defined on (Ω,Σ). The triplet (Ω,Σ,P) is called a probability space.

Thus, a probability space equips a measurable space with a function P which assigns to each event a real number—the probability ofthe event—such that the whole Ω gets assigned 1 and that probabilities of disjoint events add up. Note that from this definition it follows that $\mathbb { P } ( \boldsymbol { \varnothing } ) = 0$ and $\mathbb { P } ( \bar { A } ) = 1 - \mathbb { P } ( A )$

The probability space according to Definition 2 fully characterizes the nature of probability as understood by current mainstream mathematics. Any probabilistic argument can be reduced to a sample space Ω, a sigma-algebra Σ over Ω and a probability measure P defined on Σ. Given that probability is a general and powerful tool, this definition is a truly compressed description of the matter.

## 2.2.1 Why Sigma-Algebras?

Newcomers to probability often wonder why sigma-algebras are included in the basic definition of probability space. Ultimately, we aim to assign probabilities to events, which is the role of the probability measure. We do we need Σ? Why can’t we simply just consider all possible events, i.e. all possible subsets $A \subseteq \Omega ?$ In other words, one would want to—without loss of generality—define P on the power set (set of all subsets):<sup>1</sup>

$$
2 ^ { \Omega } : = \{ A : A \subseteq \Omega \} .
$$

In that way, one could omit sigma-algebras altogether and make the basic definition of probability space even simpler. This would surely be the way to proceed if it was mathematically meaningful. Sigma-algebras, if one would want them to express a “coarse-grained” structure, could be included at a later stage, but not in the core definition of probability. And indeed, we can safely define P on all events, as long as Ω is countable, i.e., finite or of the same cardinality as the natural numbers.

However, it turns out that when Ω is uncountable, e.g. the real numbers $\Omega = \mathbb { R }$ , the power set becomes “too big” to be used as sigma-algebra. Even something simple such as a uniform distribution over some interval of the real line cannot be constructed in a consistent way. These troubles are not private to probability theory but go back to measure theory, which is concerned with assigning measures such as volume, length, mass—and also probability—to sets. The classical question of measure theory was whether the notion of length can be generalized from intervals such as [a, b], which naturally has length $b - a ,$ to arbitrary subsets on the real line. The answer to this is negative, meaning that there exist sets which cannot be assigned a meaningful length, as it would break one or more desiderata about such length measure (countable additivity, translation invariance etc.). Those non-measurable sets are rather abstract and are constructed in a rather indirect way by employing the axiom of choice,<sup>2</sup> An easy to follow illustration of the problem is given in Rosenthal’s book [51].

These non-measurable sets should be excluded from the discussion and only measurable sets should be included. Exactly this is the primary usage of sigma-algebras and the reason why they are included in the basic definition of probability.

There is a standard choice of sigma-algebra when one is working with the real numbers, $\mathbf { i . e . , } \Omega = \mathbb { R }$ , or the D-dimensional Euclidean space, i.e. $\Omega = \mathbb { R } ^ { D }$ , namely the Borel sets. The Borel sets include any subset one could (constructively) think of (and many more), forming the Borel sigma-algebra. They are easiest defined via generated sigma-algebras.

Definition 3 (Generated Sigma-Algebra). Let E be any set of subsets of Ω. The sigma-algebra generated by E is

$$
\Sigma ( E ) : = \bigcap _ { E \subseteq \Sigma } \Sigma ,
$$

that is, the intersection ofall sigma-algebras containing E.

Note that in Definition 3 we do not require any special properties from E, but any collection of subsets of Ω is fine. It is quite easy to prove that $\Sigma ( E )$ is indeed a sigma-algebra. Furthermore, we can rightfully call it the smallest possible sigma-algebra containing E, since it it contains exactly the sets which need to occur in all sigma-algebras containing E.

With this tool at hand, the Borel sigma-algebra can be defined as

Definition 4 (Borel Sets, Borel Sigma-Algebra). The Borel sigma-algebra on R is defined as

$$
\mathcal { B } : = \Sigma ( \{ [ a , b ] | a , b \in \mathbb { R } , a \leq b \} )
$$

that is, the sigma-algebra generated by the set ofall closed intervals. Alternatively, one can use the open intervals, half-open intervals, etc., which are all equivalent definitions of B.

A likewise definition can be made for $\mathbb { R } ^ { D }$ , by using the sigma-algebra generated by (open or closed) hypercubes, or alternatively the set of all open (or closed) balls, etc.

In conclusion of this section: sigma-algebras contain those events one wants to consider and make sure that events are closed under set operations (logic connectives). The main reason that they are in the basic definition of probability space is to “fix a bug” and make the math work.

## 2.3 Interpretations of Probability

The definition of probability space requires about half a page and precisely sets the formalism and rules to work with probability. While it is a crisp and minimalist description of the nature of probability, it lacks intuition about “what probability actually is”. We might think about probability in the following ways.

• Relative Frequencies. Relative frequencies are helpful for interpreting probabilities and follow from the law of large numbers. Classically, they served as definition of probabilities, before their formalization as in Definition 2.

Given a probability space (Ω,Σ,P), we might construct n independent copies of it. Intuitively, when figuring the probability space as an urn containing various coloured balls, we would construct n identical urns and let n independent parties perform the draws. The result is then a sequence (or set) of independent and identically distributed draws $\pmb { \omega } = ( \omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { n } ) \in \Omega ^ { n }$ . The law of large numbers now guarantees that given any event $A \in \Sigma$

$$
\operatorname* { l i m } _ { n \to \infty } ~ { \frac { \sum _ { i = 1 } ^ { n } [ \omega _ { i } \in A ] } { n } } \ { \stackrel { a . s . } { = } } \ \operatorname { P } ( A ) ,
$$

where [·] denotes the Iverson bracket, evaluating to 1 for true arguments and 0 for false arguments. Hence, $\textstyle \sum _ { i = 1 } ^ { n } [ \omega _ { i } \in A ]$ is the number of times the atomic event has been picked from A across the independent copies of the probability space. Normalizing by n gives us a relative frequency between 0 and 1. The abbreviation a.s. means almost surely, that is, the equality holds with probability 1 (in the constructed probability space over sequences ω). Hence, drawing the atomic element repeatedly, independently and in identical manner, the relative frequency of how often event A happens approaches $\mathbb { P } ( A )$

• Relaxation of Logic Truth Values. A probability measure assigns a real number to each of the considered events. At least for two of these real numbers we have a concrete interpretation, namely for 0 and 1. The whole sample space Ω, but perhaps also other events A, get assigned probability 1, which is naturally interpreted as the logical value true. Conversely, the empty set, but perhaps also other events, get the value 0, naturally interpreted as false. No matter which ω is picked, it will certainly be contained in Ω and it will certainly not be in /0. Moreover, if we restricted the set of return values of P to 0 and 1, any propositional logic theory can be emulated by constructing a suitable probability space.

In the general case, where probabilities are between 0 and 1, we might naturally interpret probabilities as plausibility values. The closer a probability is to 1 the more plausible is its occurrence and the closer it is to 0 the less plausible its occurrence. The rules of probability further dictate that the probability of a union of disjoint events (amounting to a disjunction of logically incompatible statements) should simply add up, which is reasonable. Furthermore, Cox’s theorem [25] asserts that—under particular “common sense assumptions”—probability is de facto the only sound extension of propositional logic to real numbers.

• Information. Information theory provides a natural interpretation of probability as information content. Shannon [55] defined the information content of an event A as

based on the desiderata that

– an event which occurs certainly should convey zero information

– less probable events (i.e. events which are “closer” to impossibility /0) should convey more information

– information of independent events<sup>3</sup> should add up

The negative logarithm is the only function satisfying these desiderata, where a degree of freedom concerning the base of the logarithm remains. When using the logarithm to base 2 information content is measured in bits. A probability of 0.5, for example the outcome of a fair coin flip, translates to 1 bit, making intuitively sense. The logarithm to base e expresses information in nats. As information content of an event is a bijective function of its probability, probability and information content are basically the same thing.

None of these interpretations should be treated as the “ultimate correct one”. The question how to “correctly” interpret probabilities has been long a point of dispute between “frequentists” and “Bayesians”. For frequentists, the interpretation as relative frequencies is more appealing, while for Bayesians the interpretation as plausibility values of information content is natural.<sup>4</sup> It is advisable to move beyond discussions about the “true nature” of probability. Different interpretations of probability—and the ability to switch between them— should be understood as a sign of richness and justification of the theory. Regardless of interpretation, we shall embrace probability as what it is at its core: a consistent and rigorous reasoning calculus under uncertainty, based on a few simple principles.

## 2.4 Fundamental Laws of Probability

Probabilistic modelling entails constructing a probability space that represents the domain of interest together with any uncertainties, addressing the desideratum of knowledge representation in AI, as discussed in Section 1.1. The second desideratum is reasoning, i.e. using our knowledge to answer queries of interest. An important requirement of any reasoning tool is conceptual simplicity, that is, reasoning primitives should be relatively few and easy to understand. As we will see in Chapter 3, the core of probabilistic reasoning relies on a few core reasoning primitives, in particular the sum rule and product rule. We will discuss these rules on the level of distributionfunctions (introduced in Section 2.5.2), but they correspond to more fundamental laws on the level of probability measures. We will briefly discuss these laws in the following.

## 2.4.1 Law of Total Probability (The Sum Rule)

Let a probability space (Ω,Σ,P) be given and let $B \in \Sigma$ be an event. Let $A _ { 1 } , A _ { 2 } , \dotsc$ . be a countable collection of exclusive (i.e. $A _ { i } \cap A _ { j } = \varnothing$ for $i \neq j )$ and exhaustive $( { \mathrm { i . e . } } \cup _ { i } A _ { i } = \Omega )$ events, that is, they form a partition of Ω. Let now $B \cap A _ { i }$ be the joint event that both B and $A _ { i }$ happen. The law of total probability states

$$
\mathbb { P } ( B ) = \sum _ { i } \mathbb { P } ( B \cap A _ { i } ) .\tag{2.1}
$$

This law follows directly from the countable additivity property of probability measures (Definition 2) since the joint events $B \cap A _ { i }$ form a partition of B, hence the sum of their probabilities must equal $\mathbb { P } ( B )$

As a reasoning primitive, the law of total probability, or sum rule, can be interpreted as ignoring, accounting for unknown facts or forgetting. In particular, figure that we have elicited all the probabilities of the joint events $B \cap A _ { 1 } , B \cap A _ { 2 } , . . .$ , that is, a rather detailed information status about whether B and $A _ { 1 }$ happen, B and $A _ { 2 }$ happen, etc. By summing over these probabilities, we essentially forget (ignore, account for different possibilities) regarding the exclusive events $A _ { i }$ . Example 2 illustrates the concept.

## 2.4.2 Conditional Probability (The Product Rule)

The second central reasoning primitive in probability is conditional probability.

Definition 5 (Conditional Probability). Given an arbitrary probability space (Ω,Σ,P), let $A , B \in \Sigma$ be two arbitrary events, where $\mathbb { P } ( B ) > 0$ . Then the conditional probability of A given B is

$$
\mathbb { P } ( A | B ) = \frac { \mathbb { P } ( A \cap B ) } { \mathbb { P } ( B ) } .\tag{2.2}
$$

$\mathbb { P } ( A | B )$ is simply the probability ofA after asserting that $\omega \in B$ . In particular, $\mathbb { P } ( B | B ) = 1$ as should be expected. While the law of total probability (sum rule) shall be interpreted as ignoring, forgetting, or accounting for unknown facts, conditional probability shall be interpreted as observing or injecting information. In particular, while the probability space

Example 2 (Law of Total Probability). Consider a sample space Ω representing a population of people, i.e. each $\omega \in \Omega$ represents a person. Let $S \subset \Omega$ be the set ofall smokers and further

• $A _ { 1 } \subset \Omega$ be the persons of age < 10

• $A _ { 2 } \subset \Omega$ be the persons of $1 0 \leq a g e < 2 0$

• $A _ { 3 } \subset \Omega$ be the persons of $2 0 \leq a g e < 5 0$

• $A _ { 4 } \subset \Omega$ be the persons of $5 0 \leq a g e$

![](images/89b2d941e350f93dd2a6d50eb1b2b57897b3efe9191afaf4594ff6c356561f03.jpg)

Consequently, $\mathbb { P } ( S \cap A _ { 1 } )$ is the probability that somebody smokes and is younger than 10 years, $\mathbb { P } ( S \cap A _ { 2 } )$ the probability that somebody smokes and is between 10 and 20, etc. Rather than keeping all these probabilities, we might decide to forget about age and compute the probability of smoking as

$$
\mathbb { P } ( S ) = \mathbb { P } ( S \cap A _ { 1 } ) + \mathbb { P } ( S \cap A _ { 2 } ) + \mathbb { P } ( S \cap A _ { 3 } ) + \mathbb { P } ( S \cap A _ { 4 } )
$$

(Ω,Σ,P) is our “global” mathematical model of uncertainty, conditional probability essentially “rebases” the probability space on $B ,$ and updates our model based on the information that $\omega \in B .$ This is indeed correct as we can define a conditional probability space as follows:

• Define B as the new sample space.

• Define a sigma-algebra $\Sigma ^ { \prime }$ as

$$
\Sigma ^ { \prime } = \{ A \cap B | A \in \Sigma \} ,
$$

i.e., the collection of the intersection of each original event with B. It can be shown that $\Sigma ^ { \prime }$ is indeed a sigma-algebra over B.

• Define a (conditional) probability measure $\mathbb { P } ^ { \prime }$ on $\Sigma ^ { \prime }$ as

$$
\mathbb { P } ^ { \prime } ( A ^ { \prime } ) : = \frac { \mathbb { P } ( A ^ { \prime } ) } { \mathbb { P } ( B ) } = \frac { \mathbb { P } ( A \cap B ) } { \mathbb { P } ( B ) }
$$

Note that $\mathbb { P } ( A ^ { \prime } )$ is well-defined, since any $A ^ { \prime }$ is given as the intersection $A ^ { \prime } { = } A \cap B$ of events in the original Σ. Hence also $A ^ { \prime } \in \Sigma$ and $\mathbb { P } ( A ^ { \prime } )$ is defined.

Example 3 (Conditional Probability Space). Let Ω be the sample space of a given probability space and B an event we condition on. The figure below illustrates how various events A get translated into events $A ^ { \prime } { = } A \cap B$ on the conditional probability space defined on B. Conditional probability $\begin{array} { r } { \mathbb { P } ( A | B ) = \frac { \mathbb { P } ( A \cap B ) } { \mathbb { P } ( B ) } = \frac { \mathbb { P } ( A ^ { \prime } ) } { \mathbb { P } ( B ) } } \end{array}$ can now be interpreted as bona-fide probability measure on the sample space B.

![](images/2dde1402c5634e2f813acf13bac9587986e02770b024d7fe3d7c25ecc2c19ba8.jpg)  
One does not necessarily need to view B as the new sample space, but one can also keep the entire Ω. Ofcourse, the conditional probabilityfor any set A which does not overlap with B must be 0 then.

Then $( B , \Sigma ^ { \prime } , \mathbb { P } ^ { \prime } )$ is again a probability space, conditional on $\omega \in B .$ . Thus, we might see conditional probability as a tool that transforms our original probability space into a new one, accounting the provided information $\omega \in B$ . Example 3 illustrates the concept.

## 2.4.3 Bayes’ Rule (The Rule of Inverse Probability)

The product rule immediately delivers $B a y e s ^ { \prime }$ rule. Equation (2.2) can be written as

$$
\mathbb { P } ( A \cap B ) = \mathbb { P } ( A | B ) \mathbb { P } ( B ) ,
$$

and by symmetry we get

$$
\mathbb { P } ( A \cap B ) = \mathbb { P } ( B | A ) \mathbb { P } ( A ) ,
$$

Equating the right hand sides and dividing by $\mathbb { P } ( B )$ , we get Bayes’ rule:

$$
\mathbb { P } ( A | B ) = \frac { \mathbb { P } ( B | A ) \mathbb { P } ( A ) } { \mathbb { P } ( B ) } .\tag{2.3}
$$

Bayes’ rule is often called the law of inverse probability, as (2.3) essentially reverses the direction of reasoning. Of course, by symmetry we also get

$$
\mathbb { P } ( B | A ) = \frac { \mathbb { P } ( A | B ) \mathbb { P } ( B ) } { \mathbb { P } ( A ) } .\tag{2.4}
$$

Bayes’ rule is incredibly powerful, since one “conditional direction” (say $\mathbb { P } ( A | B ) )$ is often easy to describe while one is actually interested in the other direction $( \mathbb { P } ( B | A ) )$ . This is the general setting of inverse problems, which are ubiquitous in science and engineering, such as in imaging, detection, and robotics. Thus, Bayes’ rule, while being simple, describes the solution to the vast majority of these problems. However, such conceptional simplicity does not mean computational simplicity, as most Bayesian problems are numerically challenging.

## 2.4.4 (Conditional) Independence

Probabilities reflect the information content of an event. Hence, conditional probabilities lead naturally to the notion of independence. When we consider an event A whose probability does not change upon observing B, we define that A and B are independent of each other, denoted as A ⊥⊥ B:

$$
A \perp \perp B \Leftrightarrow \mathbb { P } ( A | B ) = \mathbb { P } ( A )\tag{2.5}
$$

Due to the product rule $\mathbb { P } ( A \cap B ) = \mathbb { P } ( A | B ) \mathbb { P } ( B )$ we also get

$$
A \perp \perp B \Leftrightarrow \mathbb { P } ( A \cap B ) = \mathbb { P } ( A ) \mathbb { P } ( B )\tag{2.6}
$$

and further, since $\mathbb { P } ( A \cap B ) = \mathbb { P } ( B | A ) \mathbb { P } ( A )$

$$
A \perp \perp B \Leftrightarrow \mathbb { P } ( B | A ) = \mathbb { P } ( B )\tag{2.7}
$$

which is the symmetrical case to (2.5). Hence, (2.5), (2.6) and (2.7) are equivalent definitions of independence.

Furthermore, independence can be conditional on some other event. In particular, as illustrated Section 2.4.2, conditional probability gives rise to a conditional probability space. By simply applying the concept of independence to the conditional probability space, we get the notion of conditional independence. Specifically, let A, B and C be events, where $\mathbb { P } ( C ) > 0$ . Then, A and B are conditionally independent, conditionally on C, written as

A ⊥⊥ $B | C ,$ , if the following equivalent properties hold:

$$
A \perp \perp B | C \Leftrightarrow \mathbb { P } ( A | B , C ) = \mathbb { P } ( A | C )\tag{2.8}
$$

$$
A \perp \perp B | C \Leftrightarrow \mathbb { P } ( B | A , C ) = \mathbb { P } ( B | C )\tag{2.9}
$$

$$
A \perp \perp B | C \Leftrightarrow \mathbb { P } ( A , B | C ) = \mathbb { P } ( A | C ) \mathbb { P } ( B | C )\tag{2.10}
$$

Note that they are the same as (2.5), (2.6) and (2.7), except that all probabilities are conditioned on C.

## 2.5 Random Variables

So far, we have discussed probability using its bare-bone measure theoretic definition: We reflect our domain using a (possibly abstract) sample space Ω, which is just a non-empty set. The only thing which “happens” is that an atomic element ω is selected, and we don’t know which one. Our ignorance about which ω is selected is modelled by assigning real numbers— probabilities—between 0 and 1 to all kind of events in a consistent way, namely, that they add up for disjoint events. Probability spaces are the “Linux kernel” of probability, a minimalist and simple theory which, however, unfolds into the rich structure of an intelligent calculus. However, we usually don’t want to work in the Linux kernel, but use an interpretable and high-level interface to work with. The first step towards this interface are random variables (RVs).

The basic idea of random variables is to map the abstract sample space Ω into something more interpretable, such as the real numbers R. Thus, a random variable X is really just a function $X \colon \Omega \mapsto \mathbb { R }$ , assigning each $\omega \in \Omega$ to some $X ( \omega ) \in \mathbb { R }$ . Since the selection of ω is random, also the value returned by X will be random, which is why a (deterministic) function defined on a probability space is called a “random variable”.

Evidently, there is no reason to confine random variables to only return real numbers. Rather, depending on the application, one would want them to return complex numbers, tuples of real numbers (vectors), graphs, functions, etc. Thus, a random variable in general is just a transformation of one sample space Ω to some another (interpretable) space $\mathcal { X }$

In order to consistently work with probability represented by the probability space $( \Omega , \Sigma , \mathbb { P } )$ , we will require that X is measurable, which is defined as follows.

Definition 6 (Measurable Function). Let (Ω,Σ) and $( \mathcal { X } , \Sigma _ { X } )$ be measurable spaces and let $X \colon \Omega \mapsto { \mathcal { X } }$ be afunction. X is called measurable iffor each $B \in \Sigma _ { X }$ it holds that

$$
X ^ { - 1 } ( B ) \in \Sigma ,
$$

where $X ^ { - 1 }$ is the preimage of B under X, $i . e . ^ { 5 }$

$$
X ^ { - 1 } ( B ) : = \{ \pmb { \omega } \in \Omega | X ( \pmb { \omega } ) \in B \} .
$$

Measurability of a function requires that both domain $\Omega$ and co-domain $\mathcal { X }$ are equipped with sigma-algebras Σ and $\Sigma _ { X }$ , respectively, and that the pre-images of the events in $\Sigma _ { X }$ are all contained in Σ. It is thus a condition on the function itself as well as the measurable spaces (Ω,Σ) and $( \mathcal { X } , \Sigma _ { X } )$ . Often one would fix the function of interest and design the sigma algebras such that the function indeed becomes measurable.

The defining part of a random variable X is that it is a function defined on a probability space, mapping into a “more interpretable space” X , often called the state space of X. Since the selection of $\omega \in \Omega$ is uncertain also the value (state) of $X ( \omega ) \in { \mathcal { X } }$ is uncertain. This “transformed” uncertainty is described by the distribution of a random variable.

## 2.5.1 Distribution of a Random Variable

Given a probability space (Ω,Σ,P) and a random variable X defined on this space and mapping into the measurable space $( \mathcal { X } , \Sigma _ { X } )$ , we can define the following function:

$$
\mathbb { P } _ { X } \colon \Sigma _ { X } \mapsto [ 0 , 1 ] , \mathrm { ~ w h e r e ~ f o r ~ a n y ~ } B \in \Sigma _ { X } \colon \mathbb { P } _ { X } ( B ) : = \mathbb { P } ( X ^ { - 1 } ( B ) ) .
$$

Function $\mathbb { P } _ { X }$ just assigns to each event $B \in \Sigma _ { X }$ the probability which is assigned by $\mathbb { P }$ to the preimage $X ^ { - 1 } ( B )$ , i.e. the set of all $\omega \mathrm { { s } }$ mapping to B. Since the random variable is measurable it is guaranteed that $X ^ { - 1 } ( B ) \in \Sigma$ , hence $\mathbb { P } _ { X }$ is properly defined. It can be shown that $\mathbb { P } _ { X }$ is indeed a probability measure, now defined on $( \mathcal { X } , \Sigma _ { X } )$ , and hence $\left( \mathcal { X } , \Sigma _ { X } , \mathbb { P } _ { X } \right)$ is a probability space. $\mathbb { P } _ { X }$ is called the distribution induced by X, the law of X, or the pushforward measure of P under X, and it adequately describes our uncertainty about the value $X ( \omega )$

## 2.5.2 Distribution Functions

The introduction of random variables has just transformed the uncertainty over an “uninterpretable” space Ω into uncertainty over a “more interpretable” space $\mathcal { X }$ . The latter is still described with a probability measure which is a rather cumbersome mathematical object. One also might ask why we did the hassle to first define a probability space (Ω,Σ,P) and a measurable function $X \colon \Omega \mapsto { \mathcal { X } }$ just to end up with a new probability space $\left( \mathcal { X } , \Sigma _ { X } , \mathbb { P } _ { X } \right)$

describing the uncertain quantity X. On this abstract level, it would be more straightforward to construct our model $\left( \mathcal { X } , \Sigma _ { X } , \mathbb { P } _ { X } \right)$ directly. In the following, we will thus ignore the underlying probability space $( \Omega , \Sigma , \mathbb { P } )$ and just aim to describe $\left( \mathcal { X } , \Sigma _ { X } , \mathbb { P } _ { X } \right)$ directly by the means of distribution functions. The two most widely used distribution functions are probability massfunctions and probability densityfunctions.

Definition 7 (Probability Mass Function (PMF)). Let X be a random variable with a countable state space $\mathcal { X } , { } ^ { 6 }$ equipped with sigma-algebra $\Sigma _ { X } = 2 ^ { \mathcal { X } }$ . The probability mass function of X is defined as

$$
p _ { X } \colon \mathcal { X } \mapsto \mathbb { R } , \quad p _ { X } ( x ) = \mathbb { P } _ { X } ( \{ x \} )
$$

which assigns to each state $x \in \mathcal X$ the probability that $X = x .$

A PMF is a simple and direct way to specify probability distributions. In particular, the probability of any event $A \subseteq { \mathcal { X } }$ is simply

$$
\operatorname { \mathbb { P } } _ { X } \left( A \right) = \sum _ { x \in A } p _ { X } ( x ) ,\tag{2.11}
$$

since any $A \subseteq { \mathcal { X } }$ is countable and its probability must be given by the countable sum over the probabilities of the states in A, see Definition 2. However, PMFs can only be defined for discrete random variables, as they assign probabilities in a “discrete” way to each atomic state $x \in \mathcal X$ . This requires that we assume the power set $2 ^ { \mathcal { X } }$ as sigma-algebra, which is fine for discrete state spaces but does not work for uncountable state spaces, such as R or C. For such random variables one commonly uses probability densities instead.

Definition 8 (Probability Density Function (PDF)). Let X be a random variable with the real numbers as state space, $\mathcal { X } = \mathbb { R }$ , equipped with some sigma-algebra $\Sigma _ { X }$ (usually the Borel sets) and probability measure $\mathbb { P } _ { X }$ . Let $p _ { X }$ be a function mapping from $\mathcal { X }$ to the non-negative real numbers, $p _ { X } \colon \mathcal { X } \mapsto [ 0 , \infty ]$ with the property that

$$
\forall A \in \Sigma _ { X } \colon \mathbb { P } _ { X } ( A ) = \int _ { A } p _ { X } ( x ) { \mathrm { d } } x ,\tag{2.12}
$$

that is, for any event $A \in \Sigma _ { X }$ the probability $\mathbb { P } _ { X } ( A )$ is given by the integral of $p _ { X }$ over A.   
Then $p _ { X }$ is called a probability density function of X, also just probability density $o r$ density.   
A random variable which admits a density is called continuous.

The probability density, if it exists, is not unique. In particular, any density $p _ { X }$ can be changed at a countable collection of points without changing the integral (2.12) for any A.

Note that both PMFs and PDFs are denoted by the same symbol $p _ { X }$ . The intuitive reason for this is that both serve the same purpose, namely to represent a probability measure $\mathbb { P } _ { X }$ of a random variable $X$ , either via a sum (2.11) or an integral (2.12). Moreover, within a measuretheoretic treatment of probability these two concepts become in fact the same. Specifically, when $\mu$ is any measure defined on $( \mathcal { X } , \Sigma _ { X } )$ and $p _ { X } \colon \mathcal { X } \mapsto [ 0 , \infty ]$ is a measurable function with

$$
\forall A \in \Sigma _ { X } \colon \mathbb { P } _ { X } ( A ) = \int _ { A } p _ { X } \mathrm { d } \mu\tag{2.13}
$$

then $p _ { X }$ is called a Radon–Nikodym derivative of $\mathbb { P } _ { X }$ with respect to $\mu$ . In that sense, a PMF is just a Radon-Nikodym derivative with respect to the counting measure and a PDF is just a Radon-Nikodym derivative with respect to Lebesgue measure (capturing the notion of length, area, volume, etc.). For brevity, we might just use the shorter term density for $p _ { X }$ and also be lose about whether we speak about a discrete random variable (hence using sums) or a continuous random variable (hence using integrals).

## 2.5.3 Multivariate Random Variables and Joint Distributions

So far we were concerned mainly with univariate random variables, returning a “single” value. However, the usual tasks in science and engineering are to describe how multiple quantities are entangled with each other.

We again consider some (abstract) probability space (Ω,Σ,P) and define now multiple random variables on it. We might call them $X , Y , Z$ or more systematically enumerate them as $X _ { 1 } , X _ { 2 } , \ldots , X _ { D }$ , where $D \in \mathbb { N }$ , and let $\mathcal { X } _ { 1 } , \mathcal { X } _ { 2 } , \ldots , \mathcal { X } _ { D }$ be their state spaces. Now, since the mental picture is that some atomic event $\omega \in \Omega$ is selected, where P describes our ignorance about this selection, the returned values $X _ { 1 } ( \omega ) , X _ { 2 } ( \omega ) , \dots , X _ { D } ( \omega )$ will be uncertain but (in general) also correlated with each other.

The discussion of multivariate random variables really does not add much mathematical depth to univariate random variables. In fact, one simply defines a random tuple X (also often referred to as random vector or set ofrandom variables) as the function

$$
\pmb { X } \colon \Omega \mapsto \pmb { \mathcal { X } } , \quad \pmb { X } ( \omega ) : = ( X _ { 1 } ( \omega ) , X _ { 2 } ( \omega ) , \ldots , X _ { D } ( \omega ) )
$$

mapping from $\Omega$ to the Cartesian product of the state spaces $\pmb { x } : = \times _ { i = 1 } ^ { D } \mathscr { X } _ { i }$ and whose $i ^ { \mathrm { { t h } } }$ component is the univariate random variable $X _ { i }$ . In order to assert that X is a (vector-valued) random variable one needs to ensure that X is measurable whenever the univariate random variables $X _ { 1 } , X _ { 2 } , \ldots , X _ { D }$ are measurable. Indeed, one can show that X is measurable with respect to the so-called product sigma-algebra $\Sigma _ { X }$ which is the sigma-algebra generated by

$E = \left\{ \times _ { i = 1 } ^ { D } A _ { i } \left| A _ { i } \in \mathcal { X } _ { i } \right. \right\}$ , i.e. the set of all Cartesian products of any events of the univariate random variables. Thus, X can be understood as a standard random variable, albeit one mapping to the multi-dimensional space X and equipped with product sigma-algebra $\Sigma _ { X }$

The concept of distribution functions carries over to multivariate random variables as follows.

Definition 9 (Joint Probability Mass Function (Joint PMF)). Let $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ be a random vector, consisting ofdiscrete univariate random variables $X _ { 1 } , \ldots , X _ { D }$ with countable state spaces $\mathcal { X } _ { 1 } , . . . , \mathcal { X } _ { D }$ . The joint probability mass function is defined as

$$
p _ { X } \colon \mathcal { X } \mapsto [ 0 , 1 ] , ~ p _ { X } ( \pmb { x } ) = \mathbb { P } _ { X } ( \{ \pmb { x } \} )
$$

which assigns to each joint state x the probability that $\pmb { X } = \pmb { x } .$

Similarly to univariate random variables the joint PMF describes the probability measure of X via

$$
\operatorname { \mathbb { P } } _ { X } ( A ) = \sum _ { \pmb { x } \in A } p \pmb { x } ( \pmb { x } ) .\tag{2.14}
$$

for any event $A \in \Sigma _ { X }$ . Furthermore, also the concept of PDF carries over to the multivariate case.

Definition 10 (Joint Probability Density Function (Joint PDF)). Let $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ be a random vector, consisting ofunivariate random variables $X _ { 1 } , \ldots , X _ { D }$ each with state space R and equipped with the Borel sigma-algebra. Let p<sub>X</sub> be a function mapping from $\mathbb { R } ^ { D }$ to the non-negative real numbers, $p \pmb { x } \colon \mathbb { R } ^ { D } \mapsto [ 0 , \infty ]$ , with the property that

$$
\mathbb { P } \pmb { x } ( A ) = \int _ { A } p \pmb { x } ( \pmb { x } ) \mathrm { d } \pmb { x } ,\tag{2.15}
$$

for any event (Borel set) $A \subseteq \mathbb { R } ^ { D }$ . Then $p _ { X }$ is called a joint probability density of X.

The integral (2.15) (or sum (2.14)) is the defining part of a joint distribution function $p _ { X } \mathbf { \cdot }$ when integrating (or summing) it over an event $A \in \Sigma _ { X }$ , it needs to return $\mathbb { P } _ { X } ( A )$ . From this perspective, one can also define hybrid distribution functions over sets of random variables X which contains both discrete and continuous ones.

Definition 11 (Hybrid Distribution Function). Let $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ be a random vector where $\pmb { X } ^ { \prime } = \left( X _ { 1 } , \ldots , X _ { K } \right)$ are discrete random variables and $\pmb { X } ^ { \prime \prime } = ( X _ { K + 1 } , \ldots , X _ { D } )$ are continuous random variables,<sup>7</sup> and let $\pmb { \chi } = \bigtimes _ { i = 1 } ^ { D } \mathcal { X } _ { i }$ be their joint state space.

For any event $A \in \Sigma _ { \pmb { X } }$ let

$$
A ^ { \prime } : = \{ ( x _ { 1 } , \ldots , x _ { K } ) | ( x _ { 1 } , \ldots , x _ { D } ) \in A \}
$$

$i . e . _ { \cdot }$ , the set of all states assumed by the discrete random variables in A. Furthermore, $f o r$ each $\pmb { x } ^ { \prime } \in A ^ { \prime }$ let

$$
A _ { x ^ { \prime } } ^ { \prime \prime } : = \{ ( x _ { K + 1 } , \ldots , x _ { D } ) | ( x _ { 1 } , \ldots , x _ { D } ) \in A \land ( x _ { 1 } , \ldots , x _ { K } ) = x ^ { \prime } \}
$$

i.e., the set of all states assumed by the continuous random variables assumed in A, whenever the discrete random variables assume state $\pmb { x } ^ { \prime } .$

Let $p \mathbf { { x } } \colon { \mathcal { X } } \mapsto [ 0 , \infty ]$ be a function with the property that for any $A \in \Sigma _ { X }$

$$
\operatorname { \mathbb { P } } _ { \pmb { X } } ( A ) = \sum _ { \pmb { x } ^ { \prime } \in A ^ { \prime } } \int _ { A _ { \pmb { x } ^ { \prime } } ^ { \prime \prime } } p _ { \pmb { X } } ( \pmb { x } ^ { \prime } , \pmb { x } ^ { \prime \prime } ) \mathrm { d } \pmb { x } ^ { \prime \prime } .\tag{2.16}
$$

Then $p \pmb { X }$ is called a hybrid distribution function over $\pmb { X }$

The definition of hybrid distribution functions is a bit cumbersome, as it requires us to split random variables and their state spaces into discrete and continuous ones. Furthermore, all the joint distributions—joint PMFs, joint PDFs, and hybrid distribution functions—can be understood as densities in a measure-theoretic sense, as in (2.13). The only difference between these is just the base measure: for discrete random variables one uses a counting measure, for continuous random variables one typically uses the Lebesgue measure, and for the hybrid case one would use an adequate product measure of counting and Lebesgue measure.

Hence, similarly as in the univariate case we use the same symbol $p \pmb { X }$ for joint PMFs, joint PDFs and hybrid distribution functions, as they all serve the same purpose—describing a probability measure $\mathbb { P } _ { X }$ by integrating (or summing) $p _ { X }$ over some event A.

## Chapter 3

## Probabilistic Inference

As discussed in Section 1.1, a core property of any AI system is to have some representation of a knowledge base. This role is now fulfilled by distribution functions p<sub>X</sub> introduced in the last chapter, representing both dependencies between random variables as well as our uncertainty about them. Besides representing our knowledge as a distribution function p we ultimately are interested in reasoning, or inference, i.e. answering queries of interest. Probability naturally offers inference routines in order to perform reasoning under uncertainty. The two most important routines are marginalization (the sum rule) and conditioning (the product rule), where the latter just uses the former as a sub-routine. Hence, marginalization is really the core business of probabilistic inference.

Besides these two central inference routines there are several other fundamental operations one would want to do with probability distributions, in particular taking expectations and performing optimization, i.e. taking maxima or minima of functions of interest. As it turns out, however, most models mentioned above struggle with computing these queries, as they are typically NP-hard problems in these. Probabilistic circuits (PCs), which we introduce in Chapter 5, will allow to compute these queries exactly and efficiently.

Before diving into PCs, however, in this chapter we shall study the elegant reasoning calculus probability has to offer, disregarding computational issues. In fact, one of the things which make probabilistic models so interesting is the tension between the conceptional simplicity of the calculus—it is in principle clear what shall be done—and the computational hardness involved when implementing the calculus.

## 3.1 Marginalization (The Sum Rule)

Let a distribution function $p _ { X }$ for a random vector $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ be given and let $\pmb { Y } =$ $\left( Y _ { 1 } , \ldots , Y _ { D ^ { \prime } } \right)$ and $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { D ^ { \prime \prime } } )$ be a partition of X , i.e. $\pmb { Y } \cup \pmb { Z } = \pmb { X }$ and $\pmb { X } \cap \pmb { Z } = \emptyset$ . We

assume without loss of generality that Y are the first $D ^ { \prime }$ entries in X and Z the remaining $D ^ { \prime \prime } { = } D { - } D ^ { \prime }$ entries, i.e.

$$
\pmb { X } = ( Y _ { 1 } , \ldots , Y _ { D ^ { \prime } } , Z _ { 1 } , \ldots , Z _ { D ^ { \prime \prime } } ) .
$$

The marginal distribution p<sub>Y</sub> is given as

$$
p _ { Y } ( y _ { 1 } , \dots , y _ { D ^ { \prime } } ) = \int \cdots \int p _ { X } ( y _ { 1 } , \dots , y _ { D ^ { \prime } } , z _ { 1 } , \dots , z _ { D ^ { \prime \prime } } ) { \mathrm { d } } z _ { 1 } \dots { \mathrm { d } } z _ { D ^ { \prime \prime } } ,\tag{3.1}
$$

that is, we integrate out or marginalize<sup>1</sup> all random variables in $\mathbf { { Z } } ,$ in order to get a “reduced model” $p \pmb { Y }$ only over random variables Y . We often write (3.1) more compactly as

$$
p _ { Y } ( \mathbf { y } ) = \int p _ { \mathbf { X } } ( \mathbf { y } , z ) \mathrm { d } z ,\tag{3.2}
$$

where we “unpack” the values ${ \bf { \sigma } } _ { y , z }$ as arguments in $p _ { { \pmb X } } ( y _ { 1 } , \dots , y _ { D ^ { \prime } } , z _ { 1 } , \dots , z _ { D ^ { \prime \prime } } )$ and y as arguments in $p _ { { \pmb Y } } ( y _ { 1 } , \dots , y _ { D ^ { \prime } } )$ . While the full joint distribution $p _ { X }$ captures the dependency structure and uncertainties among all X, the marginal (3.2) accurately describes dependencies and uncertainties among subset Y , while having eliminated Z from consideration. Thus, the marginalization operation (3.2) shall be interpreted as the reasoning primitive ignoring or accounting for unknown quantities Z. Since this happens by integrating or summing over all possible states of the unknown quantity Z, this operation is also often called the sum rule. Basically, it is the law of total probability (see Section 2.4.1) applied to distribution functions.

A few things are worth mentioning. First, note that $p \pmb { Y }$ is in general a joint distribution, but one over a subset of X. Of course, whenever $\pmb { Y } = \{ Y _ { i } \}$ is singleton, $p \pmb { Y }$ becomes a univariate distribution $p _ { Y _ { i } }$ . The marginal distribution is indeed “the” distribution over Y and not some kind of approximation. Specifically, just as $p _ { X }$ accurately describes a set of random variables $\pmb { X } , p _ { Y }$ accurately describes $\pmb { Y } ,$ , “as if Z had never been considered”.

Also the extreme case of marginalization where all variables X are marginalized is interesting, as it delivers the normalization constant or partitionfunction

$$
\mathcal { Z } = \int p \mathbf { \boldsymbol { x } } ( \pmb { \mathscr { x } } ) \mathrm { d } \pmb { \mathscr { x } } .\tag{3.3}
$$

The normalization constant is important for unnormalized models which represent distribution functions via some nonnegative function $f _ { \pmb { X } }$

$$
p _ { X } ( { \pmb x } ) : = \frac { f _ { { \pmb X } } ( { \pmb x } ) } { \mathcal { Z } } , ~ \mathcal { Z } = \int f _ { { \pmb X } } ( { \pmb x } ) \mathrm { d } { \pmb x } .\tag{3.4}
$$

Furthermore note that marginalization is a consistent reasoning pattern, meaning that one always arrives at the same result for any sequence of marginalization operations expressing the same marginal. For example, when starting with a joint $p _ { \pmb { X } , \pmb { Y } , \pmb { Z } }$ we will arrive at the same marginal $p _ { X }$ whether we are (i) first marginalizing $z$ (yielding $p _ { { \pmb X } , { \pmb Y } } )$ and then $\pmb { Y } ,$ , or (ii) first marginalizing Y (yielding $p \pmb { X } , \pmb { Z } )$ and then $\mathbf { z } .$

## 3.2 Conditioning (The Product Rule)

Let again a distribution function $p _ { X }$ for a set of random variables $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ be given and let $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { D ^ { \prime } } )$ and $\pmb { Z } = ( Z _ { 1 } , \dots , Z _ { D ^ { \prime \prime } } )$ be a partition of X . Given a joint state $z$ for $\mathbf { \delta Z } ,$ the conditional distribution of Y given z is

$$
p _ { Y | Z } ( { \pmb y } | z ) = \frac { p _ { \pmb X } ( \pmb y , { \pmb z } ) } { p _ { \pmb Z } ( { \pmb z } ) } = \frac { p _ { \pmb X } ( \pmb y , { \pmb z } ) } { \int p _ { \pmb X } ( \pmb y , { \pmb z } ) \mathrm { d } \pmb y } ,\tag{3.5}
$$

i.e., the ratio of the joint distribution $p _ { X }$ and the marginal $p \mathbf { z }$ , where the latter is derived from $p \pmb { X }$ using the sum rule. The conditional distribution is the concept of conditional probability (see Section 2.4.2) applied to distribution functions.

A natural way to interpret $p _ { Y | Z }$ is as a family of distribution functions over Y, indexed by the states $z \in { \mathcal { Z } }$ . While the full joint distribution $p _ { X }$ captures the dependencies and uncertainties among all random variables $\pmb { X } .$ , the conditional distribution $p _ { Y | Z }$ describes dependencies and uncertainties among Y upon establishing the event $Z = z .$ . Hence, the conditioning operation (3.5) shall be interpreted as the reasoning primitive observing or injecting evidence (that $\pmb { Z } = \pmb { z } )$ and updating the distribution over the remaining random variables Y.

Equation (3.5) can also be written as

$$
p _ { X } ( \pmb { y } , \pmb { z } ) = p _ { Y | \pmb { Z } } ( \pmb { y } | \pmb { z } ) p _ { Z } ( \pmb { z } ) ,\tag{3.6}
$$

telling us that the joint distribution over Y and $z$ is given as the product of the marginal distribution over $z$ and the conditional distribution over Y given $z .$ This essentially represents a “reasoning step-by-step” pattern: in order to reason about both Y and $\mathbf { { Z } } ,$ first reason about only Z and subsequently about Y given Z. Since the joint distribution is given as a product of marginal and conditional distributions, Equation (3.6) is also referred to as the product rule.

There are some technical subtleties with conditioning, as in order for (3.5) to be welldefined, the denominator $p _ { Z } ( z )$ needs to be non-zero. When Z is discrete, however, we might just assume an arbitrary conditional distribution $p _ { { \pmb Y } | { \pmb Z } } ( { \pmb \cdot } | { \pmb z } )$ whenever $p _ { Z } ( z ) = 0$ . To see this note that for any $A \subseteq \mathbb { y }$ we have

$$
\int _ { A } p _ { Y , Z } ( y , z ) { \mathrm { d } } y \leq \int _ { { \mathcal { Y } } } p _ { Y , Z } ( y , z ) { \mathrm { d } } y = p _ { Z } ( z ) = 0 .
$$

Thus, for any arbitrary $p _ { Y | Z } ( \cdot | z )$ we get

$$
\int _ { A } p _ { Y , Z } ( { \pmb y } , { \pmb z } ) \mathrm { d } { \pmb y } = \int _ { A } p _ { Y | Z } ( { \pmb y } | { \pmb z } ) p ( { \pmb z } ) \mathrm { d } { \pmb y } = p ( { \pmb z } ) \overbrace { \int _ { A } p _ { Y | Z } ( { \pmb y } | { \pmb z } ) \mathrm { d } { \pmb y } } ^ { \le 1 } = 0 ,
$$

as required.

When Z is continuous, and hence $p z$ is a density, Equation (3.5) yields a conditional distribution function which is conditional on an event of probability 0. Yet, conditioning is well-defined when the marginal density $p _ { Z } ( z )$ is non-zero and continuous at z.

Conditioning is, like marginalization, a consistent reasoning operation, meaning that the order of conditioning operations does not matter. For example, when we start with a joint $p _ { \pmb { X } , \pmb { Y } , \pmb { Z } }$ and want to compute $p _ { { \pmb X } | { \pmb Y } , { \pmb Z } }$ , we might first condition on Y , yielding $p _ { X , \pmb { Z } | \pmb { Y } }$ , and subsequently on Z, or, we we condition first on Z and subsequently on Y . To see this, note that the conditional

$$
p _ { X , Z | Y } ( \mathbf { x } , z | \mathbf { y } ) = \frac { p _ { X , Y , Z } ( \mathbf { x } , \mathbf { y } , z ) } { p _ { Y } ( \mathbf { y } ) } ,
$$

specifies a family of joint distribution over X and Z for every value y. Conditioning each of these conditional distributions additionally on Z delivers

$$
\frac { p _ { X , Z | Y } ( x , z | y ) } { p _ { Z | Y } ( z | y ) } = \frac { p _ { X , Y , Z } ( x , y , z ) } { p _ { Y } ( y ) } \overbrace { \int p _ { X , Y , Z } ( x , y , z ) \mathrm { d } x } ^ { \overset { \overset { \mathrm { 1 / } p _ { Z | Y } ( z | y ) } { = } } } = \frac { p _ { X , Y , Z } ( x , y , z ) } { p _ { Y , Z } ( y , z ) } = p _ { X | Y , Z } ( x | y , z )
$$

Similarly, by swapping the role of Y and Z, we will get the same result by first condition on Z and then on Y.

Conditioning and marginalization are also compatible with each other. For example, when we start with a joint $p _ { { \pmb X } , { \pmb Y } , { \pmb Z } }$ and want to compute $p _ { X | Z }$ , we might

• first condition on Z (yielding $p _ { X , Y | Z } )$ and then marginalize Y, or

• we might first marginalize Y (yielding $p \pmb { X } , \pmb { Z } )$ and then condition on Z.

In general, any two sequences of marginalization and conditioning operations leading to the same marginal and/or conditional will deliver consistent results.<sup>2</sup>

Hence, with the sum and product rule we have established the central “inference machinery” of probabilistic reasoning: we start with a joint distribution $p \pmb { X }$ describing our domain of discourse. Then we select variables we want to infer $\pmb { Y } \subseteq \pmb { X }$ , depict variables we know $\pmb { Z } \subseteq \pmb { X }$ , and ignore the rest $\pmb { X } \setminus ( \pmb { Y } \cup \pmb { Z } )$ . It is worth to embrace the simplicity and elegance of this calculus, allowing us to (repeatedly) apply just two rules to transform our knowledge base $p \pmb { X }$ into the object of desire, $p _ { Y | Z }$

Before we proceed let us simplify a bit our notation. In particular, we have now introduced symbols like $p \pmb { X }$ and $p _ { Y , Z }$ for joint distributions and symbols like $p _ { { \pmb Z } | { \pmb X } }$ and $p _ { { \pmb { Y } } , { \pmb { X } } | { \pmb { Z } } }$ for conditional distributions. Distinguishing these different marginals and conditionals is essential for our intelligent calculus, but using these subscripts quickly becomes cluttered when, for example, describing the product rule:

$$
p _ { { \pmb X } } ( { \pmb y } , z ) = p _ { { \pmb Y } | z } ( { \pmb y } | z ) p _ { { \pmb Z } } ( z ) .
$$

These subscripts are redundant, since it is clear from the arguments to which random variables the distribution belongs, and also whether it is a conditional or an unconditional distribution. We might just write more compactly

$$
p ( { \pmb y } , z ) = p ( { \pmb y } | z ) p ( z ) .
$$

Hence, we will from now omit subscripts of distribution functions when their nature is clear form the arguments.

## 3.3 The Chain Rule

As mentioned above, the product rule essentially implements a two-step reasoning pattern. By applying the product rule repeatedly, we arrive at the chain rule, which implements a multi-step reasoning pattern. Let X be a set of random variables which is partitioned into sets $\pmb { X } _ { 1 } , \ldots , \pmb { X } _ { K }$ . In the following, note that it does not matter whether any $X _ { i } , i \in \left\{ 1 , \ldots , K \right\}$ , contains only a single or several random variables, i.e. the chain rule applies regardless whether we consider single random variables or “blocks” of random variables. Using the product rule repeatedly, we can now write $p \pmb { X }$ as

$$
\begin{array} { l } { p ( x ) = p ( x _ { 1 } , x _ { 2 } , x _ { 3 } , x _ { 4 } , \ldots , x _ { K } ) } \\ { \quad = p ( x _ { 2 } , x _ { 3 } , x _ { 4 } , \ldots , x _ { K } | x _ { 1 } ) p ( x _ { 1 } ) } \\ { \quad = p ( x _ { 3 } , x _ { 4 } , \ldots , x _ { K } | x _ { 1 } , x _ { 2 } ) p ( x _ { 2 } | x _ { 1 } ) p ( x _ { 1 } ) } \\ { \quad = p ( x _ { 4 } , \ldots , x _ { K } | x _ { 1 } , x _ { 2 } , x _ { 3 } ) p ( x _ { 3 } | x _ { 1 } , x _ { 2 } ) p ( x _ { 2 } | x _ { 1 } ) p ( x _ { 1 } ) } \\ { \quad \quad \ldots } \\ { \quad \quad = p ( x _ { K } | x _ { 1 } , \ldots , x _ { K - 1 } ) p ( x _ { K - 1 } | x _ { 1 } , \ldots , x _ { K - 2 } ) \ldots p ( x _ { 3 } | x _ { 1 } , x _ { 2 } ) p ( x _ { 2 } | x _ { 1 } ) p ( x _ { 1 } ) } \\ { \quad = \displaystyle \prod _ { k = 1 } ^ { K } p ( x _ { k } | x _ { 1 } , \ldots , x _ { k - 1 } ) . } \end{array}\tag{3.7}
$$

In (3.7) the factor $p ( \pmb { x } _ { k } | \pmb { x } _ { 1 } , \dots , \pmb { x } _ { k - 1 } )$ for $k = 1$ shall be interpreted as $p ( { \pmb x } _ { 1 } )$ . This factorization of the joint $p _ { X }$ is known as the chain rule and represents “step-by-step reasoning” over multiple steps, as it tells us that the joint distribution over $\pmb { X } _ { 1 } , \ldots , \pmb { X } _ { K }$ is constructed as the marginal over $\pmb { X } _ { 1 }$ times the conditional over $\pmb { X } _ { 2 }$ given $\pmb { X } _ { 1 }$ times the conditional $\pmb { X } _ { 3 }$ given $X _ { 1 } , X _ { 2 } , { \mathrm { e t c } }$

We are not confined to use the chain rule using the ordering $1 , 2 , \ldots , K$ , but are free to use any ordering we like. Specifically, for any permutation $i _ { 1 } , \dots , i _ { K }$ of the integers $\{ 1 , \ldots , K \}$ we can write $p _ { X }$ as

$$
p ( \pmb { x } ) = \prod _ { k = 1 } ^ { K } p ( \pmb { x } _ { i _ { k } } | \pmb { x } _ { i _ { 1 } } , \dots , \pmb { x } _ { i _ { k } - 1 } ) .\tag{3.8}
$$

For example, we might factorize a joint distribution over ${ \pmb X } _ { 1 } , { \pmb X } _ { 2 } , { \pmb X } _ { 3 }$ in various ways:

$$
{ \begin{array} { r l } { p ( { \pmb x } _ { 1 } , { \pmb x } _ { 2 } , { \pmb x } _ { 3 } ) = p ( { \pmb x } _ { 3 } \mid { \pmb x } _ { 2 } , { \pmb x } _ { 1 } ) p ( { \pmb x } _ { 1 } \mid { \pmb x } _ { 2 } ) p ( { \pmb x } _ { 2 } ) } \\ & { \qquad = p ( { \pmb x } _ { 2 } \mid { \pmb x } _ { 1 } , { \pmb x } _ { 3 } ) p ( { \pmb x } _ { 3 } \mid { \pmb x } _ { 1 } ) p ( { \pmb x } _ { 1 } ) } \\ & { \qquad = p ( { \pmb x } _ { 1 } \mid { \pmb x } _ { 3 } , { \pmb x } _ { 2 } ) p ( { \pmb x } _ { 2 } \mid { \pmb x } _ { 3 } ) p ( { \pmb x } _ { 3 } ) } \\ & { \qquad { \mathrm { e t c . } } } \end{array} }
$$

Thus, rather then a single chain rule, for a given partitioning $\pmb { X } _ { 1 } , \ldots , \pmb { X } _ { K }$ there really exist K! different chain rules. The chain rule allows to factor and analyse joint distributions in a flexible way and can be considered as a “Swiss army knife” of probabilistic inference.

Conversely, reading the chain rule in the other direction, we can construct expressive joint distributions by providing a marginal or conditional for each $\pmb { X } _ { k }$ and multiplying these together. Here, the conditionals need to obey some ordering of the $\pmb { X } _ { k } \mathbf { \ ' } _ { \mathbf { S } } .$ , meaning that an $\pmb { X } _ { k }$ can only appear as conditioned in some conditional distribution $p _ { { \pmb X } _ { l } | . . . , { \pmb X } _ { k } , . . }$ <sub>..</sub> when $\pmb { X } _ { k }$ comes before $\pmb { X } _ { l }$ in this order. As long as there exists such an order, the chain rule leads to a proper joint distribution $p \pmb { X }$ . This is the main working principle both behind classical Bayesian networks [28] and autoregressive distribution estimators [29].

## 3.4 Bayes Rule (Law of Inverse Probability)

Another consequence of the product rule is Bayes rule or the rule of inverse probability, which has been discussed already in Section 2.4.3 on the level of probability measures. When considering two (sets of) random variables X,Y, we might write the joint as

$$
p ( { \pmb x } , { \pmb y } ) = p ( { \pmb y } | { \pmb x } ) p ( { \pmb x } )
$$

or also

$$
p ( { \pmb x } , { \pmb y } ) = p ( { \pmb x } \mid { \pmb y } ) p ( { \pmb y } ) .
$$

By equating the right hand sides and dividing by $p ( { \pmb x } )$ we yield Bayes rule

$$
p ( \pmb { y } | \pmb { x } ) = \frac { p ( \pmb { x } | \pmb { y } ) p ( \pmb { y } ) } { p ( \pmb { x } ) } = \frac { p ( \pmb { x } | \pmb { y } ) p ( \pmb { y } ) } { \int p ( \pmb { x } | \pmb { y } ) p ( \pmb { y } ) \mathrm { d } \pmb { y } } ,\tag{3.9}
$$

which allows us to change the “direction of reasoning” from ${ \pmb x }  { \pmb y }$ to ${ \pmb { y } }  { \pmb { x } }$ . Bayes rule is in particular prominent in applications were Y is some unobserved (latent) quantity, such as an unknown model parameter, the truth status (true,false) of a hypothesis, or some other (assumed) latent structure underlying an application, while X is an observed quantity (data). In such settings, it is often natural to describe how the data X emerges from the latent quantity Y via some conditional $p _ { { \pmb X } | { \pmb Y } }$ , usually called the likelihood. Together with a marginal distribution $p _ { Y }$ , usually called the prior, Bayes rule (3.9) allows us to infer the conditional $p _ { Y | \pmb { X } }$ of interest, called the posterior. In this way, Bayes rule captures a wide range of inverse problems in science and engineering.

## 3.5 (Conditional) Independence

The concept of independence discussed in Section 2.4.4 naturally carries over to distributions functions. The conditional distribution $p _ { Y | \pmb { X } }$ represents our information about Y when we have learned the value for X. If this this is the same as when we do not know the value of X,

i.e. the marginal $p \pmb { Y }$ , we conclude that Y and X are independent. Consequently, we define $\pmb { X } \perp \perp \pmb { Y }$ , when the following equivalent conditions hold for all values x,y:

$p ( { \pmb x } | { \pmb y } ) = p ( { \pmb x } )$

$p ( \mathbf { y } | \pmb { x } ) = p ( \pmb { y } )$

$p ( { \pmb x } , { \pmb y } ) = p ( { \pmb x } ) p ( { \pmb y } )$

Independence can occur or disappear in the context of another set of random variables Z. In particular, we say that X and Y are conditionally independent given $\mathbf { { Z } } ,$ written $\pmb { X } \perp \perp \pmb { Y } | \pmb { Z } .$ if the following equivalent conditions hold for all values x,y, z:

$p ( \pmb { x } | \pmb { y } , \pmb { z } ) = p ( \pmb { x } | \pmb { z } )$

$p ( \pmb { y } | \pmb { x } , \pmb { z } ) = p ( \pmb { y } | \pmb { z } )$

$p ( \pmb { x } , \pmb { y } | \pmb { z } ) = p ( \pmb { x } | \pmb { z } ) p ( \pmb { y } | \pmb { z } )$

## 3.6 Expectations

So far, we have established that probability allows us to represent knowledge in theform of a joint distribution and provides two simple inference rules to process our knowledge, the sum and the product rule. We have also established the chain rule and Bayes rule, which, however, are really just variants of the product rule. All this calculus does is processing distributions into other distributions, as illustrated in the following example.

Example 4. Assume we have access to the joint distribution over 21 random variables $Y , X _ { 1 } , \ldots , X _ { 2 0 }$ , where Y is some medical parameter of interest (e.g. cerebral blood flow), and $X _ { 1 } , \ldots , X _ { 2 0 }$ are some measurements related to Y $( e . g .$ features derived from 20 EEG channels). Further, assume that for a particular patient we have measured (observed) $X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , . . .$ , but $X _ { 1 7 }$ and $X _ { 1 9 }$ have not been measured (because for these two the corresponding EEG sensors have dropped out). Hence, employing the sum and product rule we compute

$$
\begin{array} { r } { p ( y | x _ { 1 } , \dots , x _ { 1 6 } , x _ { 1 8 } , x _ { 2 0 } ) = \frac { \int \int p ( y , x _ { 1 } , \dots , x _ { 1 6 } , x _ { 1 7 } , x _ { 1 8 } , x _ { 1 9 } , x _ { 2 0 } ) \mathrm { d } x _ { 1 7 } \mathrm { d } x _ { 1 9 } } { \int \int \int p ( y , x _ { 1 } , \dots , x _ { 1 6 } , x _ { 1 7 } , x _ { 1 8 } , x _ { 1 9 } , x _ { 2 0 } ) \mathrm { d } x _ { 1 7 } \mathrm { d } x _ { 1 9 } \mathrm { d } y } , } \end{array}\tag{3.10}
$$

which is our inferred distribution about Y , incorporating all available information.

In Example 4, one might wonder what shall be done next with $p _ { Y | X _ { 1 } , \dots , X _ { 1 6 } , X _ { 1 8 } , X _ { 2 0 } } ,$ a potentially very complicated and multimodal distribution?<sup>3</sup> How do we get a concrete prediction for Y? The first response to these concerns is that nothing should be done, since the conditional distribution over Y is our concrete prediction. More precisely, $i f \ ( \mathrm { i } )$ the joint $p _ { Y , X _ { 1 } , \ldots , X _ { 2 0 } }$ is indeed the true distribution over all involved quantities, or at least a sufficiently good approximation of it, (ii) the observed values $x _ { 1 } , \ldots , x _ { 1 6 } , x _ { 1 8 } , x _ { 2 0 }$ are indeed all the information we $\mathrm { h a v e } ^ { 4 }$ and (iii) (3.10) is solved exactly, there is nothing more that we can do. The conditional $p _ { Y | X _ { 1 } , \ldots , X _ { 1 6 } , X _ { 1 8 } , X _ { 2 0 } }$ captures everything we can know about Y based on the available information, or actually, it captures what we don’t know, namely our uncertainty about Y. If the distribution is unimodal and very concentrated we are pretty confident about the value of Y, while if it is spread out and multimodal we are pretty uncertain. Keeping the whole distribution of Y means that we know what we don’t know.

However, humans tend to be uncomfortable with this situation. They want to call the shots and know a concrete single value for Y. Often, committing to a single value is also an external requirement, forcing us to make a decision. The basic tool to do so is expectation, which is simply the average value a random variable assumes:

$$
\operatorname { \mathbb { E } } [ X ] : = \int p ( x ) x \mathrm { d } x ,\tag{3.11}
$$

where the integral is taken over the state space of X. For discrete random variables, the integral is replaced by a sum. For random vectors the expectation is a vector-valued quantity:

$$
\operatorname { E } [ { \pmb X } ] : = \int p ( { \pmb x } ) { \pmb x } \mathrm { d } { \pmb x } .\tag{3.12}
$$

Expectations can be naturally applied to conditional distributions. When $p _ { Y | Z }$ is a conditional distribution, the conditional expectation of Y given the event $\mathbf { Z } = \boldsymbol { z }$ is

$$
\mathbb { E } [ \pmb { Y } | \pmb { z } ] = \int p ( \pmb { y } | \pmb { z } ) \pmb { y } \mathrm { d } \pmb { y } .\tag{3.13}
$$

In Example 4, computing $\mathbb { E } [ Y | x _ { 1 } , \dots , x _ { 1 6 } , x _ { 1 8 } , x _ { 2 0 } ]$ is one particular way to decide a value for Y. However, being an average does not necessarily make it a typical value of the distribution, as illustrated in Example 5.

Example 5 (Expectation of a Bernoulli). Consider a Bernoulli distribution over a random variable X with state space $\mathcal { X } = \{ 0 , 1 \}$ , assigning probabilities $p _ { X } ( 1 ) = \theta$ and $p _ { X } ( 0 ) = 1 - \theta$ where θ is a single parameter called the success probability. When $\theta = 0 . 5$ the expectation is $\mathbb { E } [ X ] = 0 . 5$ which is a value never assumed by X. Generally, one can easily construct multimodal distributionsfor which the expectation might lie somewhere between the modes in an arbitrary large area of0 probability.

As the expectation is describing some aspect of the distribution over X, an immediate idea is to also consider expectations of transformations of X. In particular, for some (measurable) function g and we might define $\pmb { Y } = g ( \pmb { X } )$ . Then, the expectation of $\mathbb { E } [ \pmb { Y } ]$ will also be describing aspects of the original distribution $p \pmb { X }$ . Computing E[Y ] in principle requires that we first figure out $p _ { Y }$ , which is often a delicate task. However, there is another and often simpler way, as it holds that

$$
\operatorname { \mathbb { E } } [ g ( \pmb { X } ) ] = \operatorname { \mathbb { E } } [ \pmb { Y } ] = \int p ( \pmb { y } ) \pmb { y } \mathrm { d } \pmb { y } = \int p ( \pmb { x } ) g ( \pmb { x } ) \mathrm { d } \pmb { x } .\tag{3.14}
$$

This identity is known as the law of the unconscious statistician (LOTUS), as the right hand side is often considered as a definition of $E [ g ( \pmb { X } ) ]$ , even though it is actually a theorem.

For univariate random variables, a common choice for g are powers, i.e. $g ( x ) = x ^ { K }$ where $k \in \mathbb N .$ , and the expectation $\mathbb { E } [ X ^ { k } ]$ is called the $k ^ { t h }$ moment of $p _ { X }$ . The first moment is of course just the expectation $\mathbb { E } [ X ]$ also called the mean of $p _ { X }$ . The second moment of a transformed version of X, centered at 0 by subtracting its mean, is the variance

$$
\operatorname { V } [ X ] : = \operatorname { \mathbb { E } } [ ( X - \operatorname { \mathbb { E } } [ X ] ) ^ { 2 } ] = \operatorname { \mathbb { E } } [ X ^ { 2 } ] - \operatorname { \mathbb { E } } [ X ] ^ { 2 } .\tag{3.15}
$$

For random vectors X, the corresponding concept is the covariance matrix:

$$
C o \nu [ \pmb { X } ] : = \mathbb { E } [ ( \pmb { X } - \mathbb { E } [ \pmb { X } ] ) ^ { T } ( \pmb { X } - \mathbb { E } [ \pmb { X } ] ) ] = \mathbb { E } [ \pmb { X } ^ { T } \pmb { X } ] - \mathbb { E } [ \pmb { X } ] ^ { T } \mathbb { E } [ \pmb { X } ] .\tag{3.16}
$$

The diagonal of Cov are the variances of the individual random variables in X, while the off-diagonal contains the covariances between any pair X, $Y \in \pmb { X } , X \neq Y$

$$
C o \nu [ X , Y ] = \operatorname { \mathbb { E } } \left[ \left( X - \operatorname { \mathbb { E } } [ X ] \right) \left( Y - \operatorname { \mathbb { E } } [ Y ] \right) \right] .\tag{3.17}
$$

In summary, expectations are one of the main tool to convert probability distributions into “concrete values”. This concept becomes in particular important when paired with optimization (taking maxima or minima), leading to decision making, see Section 3.8.

## 3.7 Most Probable Explanation

Rather than averaging we might also take the extreme values, i.e. maxima or minima. In particular, when $p$ is a joint distribution over random variables $\pmb { X } = ( X _ { 1 } , \ldots , X _ { D } )$ with state space $_ { x }$ , then we might ask which state is the most likely one, i.e. the state with maximal probability:

$$
\pmb { x } ^ { * } = \arg \operatorname* { m a x } _ { \pmb { x } \in \pmb { \mathcal { X } } } p ( \pmb { x } )\tag{3.18}
$$

Here we assume the maximum is unique. If it is not, we might either break ties in some way or return the set of all maxima. The value $\pmb { x } ^ { * }$ is known as most probable explanation (MPE) [39].

The MPE is particularly interesting for conditional distributions. Referring to Example 4, the MPE for Y given the observed information would be

$$
y ^ { * } = \arg \operatorname* { m a x } _ { y } p ( y | x _ { 1 } , \dots , x _ { 1 6 } , x _ { 1 8 } , x _ { 2 0 } )\tag{3.19}
$$

If Y is a discrete random variable, and hence the conditional distribution function is a PMF, $y ^ { * }$ is indeed the most probable value for Y for the given evidence.

When Y is continuous, and hence the conditional is a density, the interpretation of $y ^ { * }$ is a bit more subtle, as illustrated in Example 6. The MPE in this example is perhaps surprising, as the value $Y = 9$ maximizes the density $p _ { Y } ( 9 ) \approx 0 . 8$ , but is located in a region attributing only about 0.1% of the total probability mass. If we were to pick the other mode around $Y = - 1 0$ , we would get a smaller value for $p _ { Y } ( - 1 0 ) \approx 0 . 2$ but be in a region accounting for 99.9% of the probability.

Is MPE flawed for densities? No, the seeming paradox comes from making tacit assumptions. MPE makes the following promise: assuming that $p _ { Y }$ is continuous, then there exists some $\varepsilon > 0$ such that the probability that Y falls into the interval $[ y ^ { * } - \varepsilon , y ^ { * } + \varepsilon ]$ is maximal when $y ^ { * }$ is the MPE. In Example 6, if we discretize the state space of Y into very small cells and bet on the outcome in which cell Y will land, we are best off by committing to the MPE at $y ^ { * } = 9$ (for small enough cell size). This might often not be what we want for real-valued state spaces. The solution to such apparent paradoxes is offered by decision theory.

## 3.8 Decision Theory

The issue with Example 6 is that many would prefer the mode of the left Gaussian, since with 99.9% probability Y takes a value in the vicinity of −10. Intuitively, if we guess $Y = - 1 0$ but the true value turns out to be, say, $Y = - 1 0 . 1 3$ , we still would be happy, since we have

Example 6 (MPE with Densities). Assume that Y has the following density

$$
p _ { Y } ( y ) = 0 . 9 9 9 p _ { N } ( y \vert - 1 0 , 2 ) + 0 . 0 0 1 p _ { N } ( y \vert 9 , 0 . 0 0 0 5 ) ,
$$

where $\begin{array} { r } { p _ { N } ( y | \mu , \sigma ) = \frac { 1 } { \sqrt { 2 \pi \sigma ^ { 2 } } } e ^ { - \frac { ( x - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } } \end{array}$ is the Gaussian density. Hence, p(y) is a Gaussian mixture with two components, one of which represents 0.999 of the total probability mass and the other 0.001. This density is shown in the following figure:

![](images/f5a1faf18044cf812439c39b5c803755444e88859dc12eac27dff5bb79c65f28.jpg)  
Here we omit any conditioning information, but we mightfigure p as a (toy) distribution resultingfrom (3.10) in Example 4. The MPE in this example, i.e. the state where p<sub>Y</sub> is maximal, is $y ^ { * } = 9 ,$ or actually slightly less, due to the Gaussian component on the left.

guessed a value close to the true value. As illustrated above, MPE essentially discretizes the space into tiny cells and deliberately ignores any notion of closeness among the cells—if we bet on the neighboring cell of where Y really lands, “we still lose” according to MPE. Decision theory [4] allows us to make such assumptions explicit, by introducing a loss function and a simple recipe for making optimal decisions under uncertainty.

Assume we aim to decide on the value of a scalar quantity of interest Y (the case for random vectors is straightforward), and let $\mathcal { V }$ be the state space of Y. In order to make this decision we assume that we have evidence $\pmb { X } = \pmb { x } .$ , and that we have already computed the conditional $p _ { Y | \pmb { X } }$ , by applying the sum and product rule. If no evidence is available, we would use the marginal $p _ { Y }$ instead. In either case, we arrive at the required distribution by mere application of the sum and product rules.

A lossfunction ℓ is any function

$$
\ell \colon \mathcal { V } \times \mathcal { Y } \mapsto \mathbb { R } .\tag{3.20}
$$

Given $y ^ { \ast } , y _ { t } \in { \mathcal { V } }$ , the value $\ell ( y ^ { * } , y _ { t } )$ represents our regret when we decide for value $y ^ { * }$ , but actually $Y = y _ { t }$ . Usually, loss functions will satisfy $\ell ( y _ { t } , y _ { t } ) \leq \ell ( y ^ { * } , y _ { t } )$ for all $y ^ { \ast } \in \mathcal { V }$ , that is, our loss will be minimal if we correctly guess the true value of Y.

Now, the basic rule of decision theory is to decide for the value which minimizes the expected loss, conditional on the available evidence $\pmb { X } = \pmb { x } \mathrm { : }$

$$
y ^ { * } = \arg \operatorname* { m i n } _ { y \in { \mathcal { Y } } } \mathbb { E } [ \ell ( y , Y ) | \pmb { x } ] .\tag{3.21}
$$

The frequentist interpretation of probability gives a clear argument why this is an optimal decision rule: in a scenario of making repeated decisions, rule 3.21 will lead to the minimal sum of losses if we correctly inferred the true $p _ { Y | \pmb { X } }$ and the number of decision trials goes to infinity. Let us study a few basic loss functions in the context of Example 6.

Zero-one loss. Consider for some $\varepsilon > 0$ the loss function

$$
\ell _ { 0 / 1 } ( y ^ { * } , y ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } | y ^ { * } - y | < \varepsilon } \\ { 1 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

denoted as zero-one loss. If the (conditional) density of Y is continuous, then (3.21) under zero-one loss approaches the MPE when $\varepsilon \to 0$ . Hence MPE essentially minimizes zero-one loss.

Squared distance loss $\left( \ell _ { 2 } { \mathbf { - } } \mathbf { l o s } \mathbf { s } \right)$ . The “all-or-nothing” behavior of MPE is probably not what one wants in Example $^ { 6 , }$ as it delivers an “atypical” value in an area of low probability. Rather, one would probably prefer the squared distance loss

$$
\ell _ { 2 } ( y ^ { * } , y ) = ( y ^ { * } - y ) ^ { 2 } ,
$$

as it assumes that we are still happy if we guess a values close to the true value of $Y .$ . It can be shown that for any distribution $p _ { Y }$ , the least squares loss, i.e. (3.21) when using $\ell _ { 2 } .$ is achieved by the mean of $p _ { Y }$ . For a Gaussian mixture model the mean is given as the corresponding mixture of the components’ means. Specifically, in Example $^ { 6 , }$ we get the solution

$$
y ^ { * } = 0 . 9 9 9 \times ( - 1 0 ) + 0 . 0 0 1 \times 9 = - 9 . 9 8 1 ,\tag{3.22}
$$

delivering a value almost at the center of the high probability region on the left.

Absolute distance loss $\left( \ell _ { 1 } { \ - } \mathbf { l o s s } \right)$ . However, one might be concerned by the fact that the least squares solution (3.22) is quite influenced by the locations of the Gaussian components. In particular, if the mean of the right Gaussian component (the one with weight 0.001) was moved to a very large value, e.g. 9000, we would get $y ^ { * } = - 0 . 9 9$ which is far away from the high probability area. This proneness to outliers is typical for least squares. A remedy to this problem is to use absolute distance loss defined as

$$
\ell _ { 1 } ( y ^ { * } , y ) = | y ^ { * } - y | .
$$

It can be shown that the minimum of (3.21) when using $\ell _ { 1 }$ is achieved by the median of $p _ { Y }$ which is the value $y ^ { * }$ for which P $\begin{array} { r } { { ' } ( Y \leq y ^ { * } ) = \operatorname { \mathbb { P } } ( Y \geq y ^ { * } ) = 0 . 5 } \end{array}$ . It can be figured out that the median in Example $^ 6$ is about −9.9975 and it stays virtually the same if we move the mean of the right Gaussian to 9000 or even far more extreme values. This robustness to outliers is a well known property of $\ell _ { 1 } { - } \mathrm { l o s }$ and the median.

In summary, minimizing expected loss (3.21) is a principled approach to “call the shots” and transit from our probabilistic prediction—a conditional distribution over the quantity of interest $Y { \mathrm { - } } { \mathrm { t o } }$ a concrete hard value $y ^ { * }$ . It should be advised that—in the ideal setting—this step should come very last, since it discards all the beautiful “fluffy uncertainty” about Y and approximates it with a brutally hard single value.

In particular in Example 4, figure the scenario that inferring the cerebral blood flow Y was only an intermediate step, and that the ultimate goal is to infer, together with other medical parameters, the health condition of a patient. In this scenario it is crucial to keep the entire $p ( y | \ldots )$ and process it further downstream. Approximating it in any way before we have arrived at the final inference goal compromises the exact reasoning calculus of probability, destroys valuable information, and can—in principle—deliver arbitrary bad results. In other words, we shall “stay calm” and continue applying the sum and product rule.<sup>5</sup>

Some points are worth mentioning. First, defining a loss function ℓ and applying decision rule (3.21) makes our assumptions explicit and allows us to specify about what we care when deciding a value $y ^ { * }$ . However, designing a loss function for a concrete application is not trivial. For example, when we are dealing with a binary classification problem, i.e. $Y \in \{ 0 , 1 \}$ , where $Y = 0$ represents the absence of a deadly disease and $Y = 1$ represents its presence, it is quite delicate to come up with good loss function. It is clear, that predicting ${ { y } ^ { * } } = 0$ (patient healthy) when in fact $Y = 1$ (patient ill) should incur a higher loss than for the converse case. But, how much higher? We might fix an arbitrary price, e.g. 1 Euro, to diagnosing the patient as ill when in fact they are healthy, but how much does it cost when we release an actually ill patient and endanger their live? An ethically delicate question.

Yet, I want to stress again that if (i) the loss function ℓ indeed adequately represents our preferences, (ii) we have access to the true data generating distribution $p _ { Y , { \pmb X } }$ and (iii) we can actually compute (3.21) exactly, there is nothing left to do—we have successfully made an optimal decision in terms of expected loss. There are a lot of $\bf \ddot { \Phi } _ { \mathrm { i f } } , \bf \vec { \Phi } _ { \mathrm { S } } ^ { \mathrm { 3 } }$ here and I am not saying that these computations are easy, yet it is surely beneficial to know and specify what is the best one can do. For example, in the case of classification, the MPE computed for the true data distribution leads to the Bayes optimal classifier, i.e. the classifier with minimal zero-one loss. A famous result for the K-nearest neighbor classifier is that it asymptotically converges to the Bayes optimal classifier (given that the number of neighbors grows also to infinity, but less strongly than the number of data points), making it a theoretically well-justified classification rule [7]. In the context of regression, using the data generating distribution and $\ell _ { 2 }$ loss leads to the optimal regression function, and no ever-so-fancy machine learning method will be able to outperform it in terms of true squared loss.

Finally, it is worth mentioning that (3.21) is not the only good decision rule. For example, we might not care about the expected loss too much but rather predict a value such that the suffered loss is smaller than a certain threshold with a prescribed probability, and otherwise reject the decision, leading to a more risk aware prediction scheme. Moreover, conformal prediction [54] gives up on the property to return only a single value but rather returns a minimal set of predictions ${ \mathcal { V } } ^ { * } \subset { \mathcal { V } }$ such that the true value of Y ends up in in $\mathcal { V } ^ { \ast }$ with high probability.

## Chapter 4

## State of The Art

In this chapter, I provide an overview of state-of-the-art probabilistic approaches in AI. While AI was largely dominated by logic-based methods until the late 1980s, the necessity of accounting for uncertainty in real-world scenarios became increasingly recognized. Although probability was not immediately embraced—it actually faced quite some criticism for its computational challenges and perceived impracticality—it gradually gained ground. A key proponent was Judea Pearl, whose seminal book Probabilistic Reasoning in Intelligent Systems: Networks ofPlausible Inference [39] firmly established probabilistic reasoning as a central approach in AI.

Graphical Models, in particular Bayesian networks, were central to Pearl’s work and have since remained a cornerstone of probabilistic modeling [28]. Although their prominence has somewhat diminished in the deep learning era, they still serve as a lingua franca of probabilistic modeling, due to their intuitive visual representation, which correspond directly with conditional independence assumptions and tractability in inference and learning [39, 28].

Inference in graphical models—especially marginalization via the sum rule—is generally NP-hard [11]. However, exact inference is tractable for models with bounded treewidth, a measure of how tree-like a graph is. Most probabilistic machine learning systems can be mapped onto a corresponding graphical model, and authors often use such them to concisely express model structure and assumptions.

Deep Autoregressive Models [29, 70] apply the chain rule of probability and model each conditional distribution using a shared neural network. These models can be interpreted as fully connected Bayesian networks, which are capable, in principle, of representing any joint distribution. While they support sampling and density evaluation, computing marginals, conditionals, or performing complex inference tasks remains challenging.

Variational Autoencoders (VAEs) [26] are a neural latent variable model, previously known as density models [33]. They introduce a latent code with a simple distribution, often isotropic Gaussian, which is transformed by a neural network to produce the (parameters of) a conditional distribution over observable variables; hence they represent an infinite mixture model, with the latent variable marginalized out.

Exact inference in VAEs (density networks) is a hard task, but the evidence lower bound (ELBO), that is, a lower bound on the log-likelihood can be estimated. A major innovation of VAEs was the introduction of amortized inference: instead of computing the posterior for each data point individually, a separate network—commonly called the encoder or inference network—learns to approximate it. The encoder is used to compute an ELBO estimate which used as the VAEs learning objective as a replacement for the exact log-likelihood. By optimizing both the parameters of the model and the encoder, one achieves the dual goal to improve the ELBO estimate and optimizing the model parameters at the same time.

Generative Adversarial Networks (GANs) [22] also introduce a simple latent distribution but adopt an adversarial training scheme rather than maximizing a log-likelihood bound. They involve a second network—the discriminator or critic—which learns to distinguish real data from generated samples. This training objective corresponds to minimizing a probabilistic divergence distinct from maximum likelihood [22, 3].

While GANs enable realistic sampling, they generally do not allow density evaluation (often such densities do not exist in the Lebesgue sense), nor do they support tractable inference for marginals or conditionals.

Normalizing Flows [49, 37] are another principle to probabilistic modeling. Similar as VAEs and GANs, they start with a simple distribution (e.g., an isotropic Gaussian) and apply a series of bijective, differentiable transformations to generate samples from a more complex target distribution. The exact density of the transformed distribution is computed via the change of variables formula, and exact sampling is also feasible. However, computing marginals or conditionals remains challenging, similar to VAEs and GANs.

Energy-Based Models define distributions using an unnormalized energy function E(x), often parameterized by a neural network:

$$
p ( \mathbf { x } ) = \frac { \exp ( - E ( \mathbf { x } ) ) } { \int \exp ( - E ( \mathbf { x } ) ) \mathrm { d } \mathbf { x } } .
$$

While extremely flexible, these models are typically intractable. Evaluating densities, drawing samples, and performing inference are all computationally expensive. Marginals and conditionals are especially difficult to estimate reliably.

Score-Based Models [77] are closely related to energy-based models and represent a distribution via its score function, i.e., the gradient of the log-density, ∇ log p(x). A neural network is trained to approximate this score. When combined with diffusion processes and Langevin dynamics, these models currently represent the state of the art in generative modeling for images [61].

Despite their impressive performance, score-based models usually lack tractable density evaluation and require approximations for sampling, marginals, and conditionals. These approximations can be difficult to validate in practice.

Gaussian Processes (GPs) [79] are another prominent type of probabilistic model. Formally, GPs are a (possibly uncountable) collection of random variables, of which any finite subset has a multivariate Gaussian distribution. Usually, these random variables are indexed by some continuous parameter, $\mathbf { e . g . } x \in \mathbb { R }$ , which allows us to interpret GPs as random functions. GPs have ample applications in Bayesian modeling of latent functions and Bayesian optimization [59]. Due to Gaussianity, computing marginals and conditionals in GPs is tractable, but associated with a cubic computational cost in the number of data points, which becomes delicate for large data sets. These challenges are most commonly addressed with sparse inducing point techniques [47].

Probabilistic Programming (PP) [69] is a general framework to express probabilistic models in a flexible way, namely by writing a program equipped with random variables. When evaluating such a program in the standard way, it simply amounts to a stochastic simulator. However, the main powerful idea of PP is that such a program constitutes a probabilistic model which can be subject to inference. In this way, one might provide data for the observed output of a probabilistic program and query the posterior distribution over certain latent parameters, in principle inferred via Bayes’ rule. In practice, this inference is tackled via approximation, usually employing variational inference or Monte Carlo techniques. In a nutshell, PP aims to achieve for probabilistic modeling and inference what machine learning frameworks like TensorFlow and PyTorch have achieved for deep learning.

The list of probabilistic modeling approaches above is not complete, but captures a coarse picture of “what is going on in the field”. A remarkable point is that many models mainly strive for expressivity and neglect the question about inference, or actually, heavily rely on approximation such as sampling and variational inference.<sup>1</sup> This is in quite stark contrast to the picture of probability as a sound and rigorous reasoning framework under uncertainty. Reasons for this neglect are, in my personal opinion, stemming from a particular mindset (i) assuming that tractable models necessarily underperform and (ii) tailored towards the contemporary success of deep learning (i.e. function approximation) style of AI. Probabilistic circuits, as discussed in the following, put an explicit focus on tractability.

## Chapter 5

## Probabilistic Circuits

So far, I have presented probability as an elegant and rigorous inference language for AI, boiling down to two simple rules, the sum and product rule. Then, in order to make (optimal) decisions, all we need to do on top of that is taking expectations, maxima and minima. Of course, this actually requires us that we have access to the true underlying data distribution or a reasonably good approximation. If our model defers too much from reality, all the inferences we are doing will be worthless—garbage in, garbage out. Hence, faithful modeling and learning is a crucial part of the game.

Probabilistic models are generally some representation of a distribution function p<sub>X</sub>, such as probabilistic graphical models (PGMs) [28], density networks and variational autoencoders (DNs, VAEs) [33, 26, 50], neural auto-regressive distribution estimators (NADEs) [29], energy-based models [60], score-based models [77], normalizingflows [49], probabilistic programs [6], and Gaussian processes (GPs) [79].<sup>1</sup> A common model which does not represent a distribution function p<sub>X</sub> (yet a distribution $\mathbb { P } _ { X } )$ is generative adversarial networks (GANs) [22], as they usually do not admit a density with respect to Lebesgue measure. All these models represent distributions in various creative and flexible ways, so that we are hopefully able to represent and learn the distribution we are interested in in.

Probabilistic inference, however, is generally intractable in most of these models. Exceptions are PGMs with bounded tree-width [28] (computing marginals is exponential in the tree-width) and GPs (computing conditionals is cubic in the number of samples). For all other models, computing marginals and/or conditionals is a hard computational problem, as they are all based on high-dimensional non-linear functions (neural networks). Computing integrals—the core business of inference—over such functions is a notoriously hard task.

Probabilistic circuits (PCs [73, 9] are a type of tractable probabilistic model that allows to perform inference exactly and efficiently, meaning that the probabilistic inference task of choice (marginals, conditionals, etc.) are computable without approximation<sup>2</sup> and in polynomial time of the PC’s size. Hence, PCs stay at all times faithful to the probabilistic reasoning principle: when we manage to represent our distribution of interest as a PC, we do have the guarantee that we arrive at the correct conclusion. This property makes PCs quite unique among other probabilistic models, which typically strive for maximal expressivity rather than tractable inference.

Evidently, there must be a catch to PCs. Basically, PCs “buy” tractability by accepting rather stringent constraints on (or properties of) their structure. These constraints make them less expressive than other types of model, meaning that there exist probability distributions which have a compact representation (polynomial in the number of random variables) as, say, a PGM, but an exponential representation size as a PC. These theoretical arguments, which go under the name expressive efficiency [35], are however not fully conclusive. In particular, the PC community has steadily been making progress over the last two decades and demonstrated that tractable models often compete well with intractable ones or even outperform them. Some examples are provided in Part II of this thesis.<sup>3</sup> Furthermore, it should be noted that PCs are still universal approximators of densities, that is, they absolutely can represent any distribution we like. The representation size, however, might just be infeasible.

The term “probabilistic circuits” is a relatively new one, but the principle has been around since decades and is known under many different names. Arithmetic circuits [14] are probably the earliest representatives and clearly display the key ideas of PCs and other tractable circuits (operating e.g. in the logic domain). The concept of PCs is also reflected in AND/OR-graphs [17], sum-product networks [46] and many other models. The main idea to use yet another name, put forward by several researchers at UCLA [9], is to unify these superficially different names and concepts under one umbrella.

In this chapter, I introduce PC and explain their working principles. Furthermore, I give an overview of the work I have done in this area over the last 10 years, which also concludes Part I of this cumulative Habilitation thesis. In Part II, the relevant papers I have published are listed as chapters.

## 5.1 Basic Definitions of Probabilistic Circuits

We start with the formal definition of PCs.

Definition 12 (Probabilistic Circuit (PC)). Let a collection of random variables $\pmb { X } =$ $\left( X _ { 1 } , \ldots , X _ { D } \right)$ be given, let $\mathcal { X } _ { 1 } , . . . , \mathcal { X } _ { D }$ be their state spaces and $\pmb { x } = \mathsf { X } _ { i } \mathcal { X } _ { i }$ the joint state space of X. A probabilistic circuit (PC) is a representation (model) of a potentially unnormalizedjoint probability distribution defined on X, i.e., afunction

$$
p \colon { \pmb x } \mapsto [ 0 , \infty ] .\tag{5.1}
$$

Specifically, a PC is a type ofneural network, i.e. a (directed acyclic) computational graph $\mathcal { G } = ( V , E )$ containing three types of computational nodes, namely probability distribution functions (short distributions), weighted sums and products. In the following we introduce the symbols D, S and P to refer to some generic distribution, sum and product node, respectively, while N refers to a generic node of any type. Further, let D, S and P denote the set of all distributions, sums and products, respectively.

For any $\mathsf { N } \in V _ { : }$ , let in(N) denote the neighbors of N with an edge towards N, and out(N) be the neighbors with an edge outgoing from N. The input to the PC is some state $\pmb { x } = ( x _ { 1 } , \dots , x _ { D } ) \in \pmb { x }$ for the random variables (x is not contained in V ).

The leaves ofG are the distribution nodes, i.e., $\mathsf { D } = \{ \mathsf { N } \in V | \mathrm { i } \mathbf { n } ( \mathsf { N } ) = \emptyset \}$ , while all other nodes are either sum or product nodes. Moreover, only the distribution nodes get x as direct input (and by construction no input from any $\mathsf { N } \in V )$ , while sum and product nodes can only get direct inputfrom nodes in V. Thus, the distribution nodes D form thefirst layer in the $P C ,$ while sums and products are located on “higher” layers. For that reason, distribution nodes are also often called input distributions of the PC.

A distribution node D is a distribution function over some subset of X. This subset, written as $\pmb { X } [ \mathsf { D } ] \subseteq \pmb { X }$ , is denoted as the scope or receptive field ofD. Hence, when evaluating the PC for some joint state x, node D only receives $\pmb { x } [ \mathsf { D } ] _ { \pmb { x } }$ , defined as the vector containing the values in x corresponding to $\pmb { X } [ \mathsf { D } ]$ . The value computed by D is the distribution function evaluated at x[D], which is technically $\mathsf { D } ( \pmb { x } [ \mathsf { D } ] )$ , but usually written simply as $\mathsf { D } ( { \pmb x } )$ . A distribution node might have trainable parameters, which are denoted as $\pmb { \theta } _ { \mathsf { D } }$

A sum node S with real weights $\pmb { \theta } _ { \mathsf { S } } = ( \theta _ { \mathsf { S N } } ) _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) }$ computes a nonnegative mixture of its inputs, i.e.,

$$
\mathsf { S } ( \pmb { x } ) = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } )\tag{5.2}
$$

where $\theta _ { S N }$ are (trainable) weights satisfying $\theta _ { \mathsf { S N } } \geq 0$

Example 7 (Probabilistic Circuit).

An example PC over seven random variables $X _ { 1 } , \ldots , X _ { 7 }$ is shown on the right. A complete state (observation) $\pmb { x } = ( x _ { 1 } , \dots , x _ { 7 } )$ serves as input to the PC. The first layer consists of distributions nodes $( \mathsf { D } ^ { \prime } s )$ , here illustrated with nodes showing a Gaussian-like distribution functions (they are, however, not necessarily Gaussian, but any distributionfunction can be used). The scope of each distribution node can be recognized by which input values are connected to it.

The nodes with + and × symbols are the sum and product nodes, respectively. The PC computes a density by evaluating the circuit bottom up, as usual in neural networks. The return value is $p ( { \pmb x } )$ , the PC distribution evaluated at x. Paramters for sum weights and distribution nodes are not shown in the figure.

![](images/2e3b82225cc6323972eefe539ef5e12156c00fbe8e7b0f26551db5da4a70bc31.jpg)

A product node P computes the product of its inputs, i.e.,

$$
\mathsf { P } ( { \pmb x } ) = \prod _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { P } ) } \mathsf { N } ( { \pmb x } ) .\tag{5.3}
$$

Note since distribution nodes have a restricted scope, that is, a subset of X, sum and product nodes will also have a restricted scope. For an arbitrary sum or product node N, its scope is recursively given as

$$
\pmb { X } [ \mathsf { N } ] = \bigcup _ { \mathsf { N } ^ { \prime } \in \mathbf { i n } ( \mathsf { N } ) } \pmb { X } [ \mathsf { N } ^ { \prime } ] .\tag{5.4}
$$

The nodes with out $( \mathsf { N } ) = \emptyset$ are called the roots or output nodes of the PC. We assume that the roots havefull scope, i.e., $\pmb { X } [ \mathsf { N } ] = \pmb { X }$ . Usually, we assume that the PC has a single root, representing the model (5.1).

We see that the definition of PCs is quite straightforward: We aim to represent a multivariate distribution function $p \colon { \pmb x } \mapsto [ 0 , \infty ]$ and do this with a neural network whose first layer consists of distribution functions over smaller scopes, and whose higher layers contains nonnegatively weighted sums and products. Since the distribution nodes are nonnegative and the sum weights are nonnegative, the whole function will be nonnegative. Of course, PCs according to Definition 12 might not be normalized, hence they define a distribution function up to the normalization constant

$$
\mathcal { Z } = \int p ( \pmb { x } ) \mathrm { d } \pmb { x } ,\tag{5.5}
$$

which we assume to be finite and non-zero. An example of a PC is shown on Example 7.

## 5.1.1 Decomposability and Smoothness

The central motivation of PCs is that probabilistic inference remains tractable, i.e., so at least we should be able to compute marginals and conditionals. However, PCs according to Definition 12 are actually intractable. In particular, the normalization constant (5.5) and any other marginals will be pretty hard to compute. We are still missing some “secret sauce” for tractability, namely, structural constraints. The two most important structural constrains used in PCs are decomposability and smoothness.

Definition 13 (Decomposability). A product node P in a PC is called decomposable if its input nodes have disjoint scopes, i.e.,

$$
\forall \mathsf { N } , \mathsf { N } ^ { \prime } \in \mathbf { i n } ( \mathsf { P } ) , \mathsf { N } \neq \mathsf { N } ^ { \prime } \colon \pmb { X } [ \mathsf { N } ] \cap \pmb { X } [ \mathsf { N } ^ { \prime } ] = \emptyset .
$$

A PC is decomposable if all its product nodes are decomposable.

Definition 14 (Smoothness). A sum node S in a PC is called smooth if all its input nodes have the same scope, i.e.,

$$
\forall \mathsf { N } , \mathsf { N } ^ { \prime } \in \mathbf { i n } ( \mathsf { S } ) \colon \pmb { X } [ \mathsf { N } ] = \pmb { X } [ \mathsf { N } ^ { \prime } ] .
$$

Note that by (5.4) a smooth sum node has the same scope as any of its input nodes. A PC is smooth ifall its sum nodes are smooth.

Note that the PC in Example 7 is decomposable and smooth. Smoothness is a bit of a cosmetic property, as a non-smooth PC can either be rendered smooth or it turns out that the normalization constant is ∞, hence the PC is not a valid representation of a distribution. Hence, for now we can just safely assume that any PC is smooth. Decomposability will turn out to be a key property for tractable inference, in particular marginals and conditionals. But first let us better understand what these two structural properties imply.

## 5.1.2 PCs as Hierarchical Mixtures

An immediate and convenient consequence of decomposability and smoothness is that the normalization constant of a PC is 1 if we (i) use normalized input distributions (i.e. their normalization constant is 1) and (ii) use normalized sum weights, i.e., for each S, additionally to requiring $\theta _ { S N } \geq 0$ we also require $\begin{array} { r } { \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } = 1 } \end{array}$ , so that S computes a convex combination of its inputs. More precisely, it will turn out that each node in a PC is a normalized distribution over its scope and that the whole PC just forms a hierarchical mixture model.

We show this property by induction over a topological order of the PC nodes V. Since the computational graph is acyclic and directed, there exists at least one topological order of the nodes

$$
{ \mathsf { N } } _ { 1 } \prec { \mathsf { N } } _ { 2 } \prec \dots \prec { \mathsf { N } } _ { | V | } ,\tag{5.6}
$$

such that it holds for each $\mathsf { N } _ { i }$ that all nodes in in $( \mathsf { N } _ { i } )$ have an index smaller than i. In other words, order (5.6) is a topological order if arrows can only point from “left to right”. The existence of at least one topological order is equivalent to the fact that the PC graph $\mathcal { G }$ is directed and acyclic.

We can assume that the input distributions are the first |D| nodes in this order. Since we assume that they are already properly normalized distributions, their normalization constant is 1. This is the induction basis.

Next assume any node $\mathsf { N } _ { i }$ with $i > | \mathsf { D } |$ , which is either a sum or product node. We assume, by induction, that all nodes $\mathsf { N } _ { 1 } , \ldots , \mathsf { N } _ { i - 1 }$ have normalization constant 1 and show that in this case also $\mathsf { N } _ { i }$ has normalization constant 1. First note that all in $( \mathsf { N } _ { i } )$ come before $\mathsf { N } _ { i }$ in the order and have therefore a normalization constant of 1.

Assume first that $\mathsf { N } _ { i } = \mathsf { P }$ is a product. For simplicity, we assume that P has only two inputs N and $\mathsf { N } ^ { \prime }$ . The case for more inputs is immediate. Further, let us rename the scopes of the two input nodes as $\pmb { X } [ \mathsf { N } ] = \pmb { Y }$ and $\pmb { X } [ \mathsf { N } ^ { \prime } ] = \pmb { Z }$ and recall that decomposability means that $\pmb { Y } \cap \pmb { Z } = \emptyset$ . Then the product computes

$$
\mathsf { P } ( \pmb { x } ) = \mathsf { N } ( \pmb { y } ) \mathsf { N } ^ { \prime } ( \pmb { z } ) .\tag{5.7}
$$

Hence, since N and $\mathsf { N } ^ { \prime }$ are correctly normalized distributions, P is just a factorized distribution, assuming independence among Y and Z, see Section 3.5. In particular, its normalization constant is 1 since

$$
\int \int \mathsf { P } ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { \mathsf { y } } \mathrm { d } z = \int \int \mathsf { N } ( \boldsymbol { \mathsf { y } } ) \mathrm { N } ^ { \prime } ( z ) \mathrm { d } \boldsymbol { \mathsf { y } } \mathrm { d } z = \overbrace { \left( \int \mathsf { N } ( \boldsymbol { \mathsf { y } } ) \mathrm { d } \boldsymbol { \mathsf { y } } \right) } ^ { = 1 } \overbrace { \left( \int \mathsf { N } ^ { \prime } ( z ) \mathrm { d } z \right) } ^ { = 1 } = 1 .\tag{5.8}
$$

If, on the other hand, $\mathsf { N } _ { i }$ is a sum node, it computes

$$
\mathsf { S } ( { \pmb x } ) = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( { \pmb x } ) .\tag{5.9}
$$

Since by induction hypothesis all $\mathbf { i n } ( \mathsf { S } )$ are correctly normalized distributions, and since the weights satisfy $\theta _ { \mathsf { S N } } \geq 0$ and $\begin{array} { r } { \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } = 1 } \end{array}$ and the sum node is smooth, (5.9) is just a mixture distribution, i.e. a convex combination of distribution functions. This in turn is always a correctly normalized distribution. In particular, when we compute the normalization constant, we see that

$$
\int { \mathsf { S } } ( { \pmb x } ) \mathrm { d } { \pmb x } = \int \sum _ { \mathbb { N } \in \mathbf { i n } ( { \mathsf { S } } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( { \pmb x } ) \mathrm { d } { \pmb x } = \sum _ { \mathbb { N } \in \mathbf { i n } ( { \mathsf { S } } ) } \theta _ { \mathsf { S N } } \overbrace { \int \mathsf { N } ( { \pmb x } ) \mathrm { d } { \pmb x } } ^ { = 1 } = \sum _ { \mathbb { N } \in \mathbf { i n } ( { \mathsf { S } } ) } \theta _ { \mathsf { S N } } = 1 .\tag{5.10}
$$

Consequently, any node in a decomposable and smooth PC is a correctly normalized distribution over its scope, either by construction (D), or because it is a factorized distribution (P) or a mixture distribution (S), whose factors (respectively mixture components) are given by other PC nodes. We can figure that a PC structure is just a blueprint, crafting larger distributions out of smaller ones, like Lego blocks.

An important thing to note is that, even though products represent factorized distributions and hence independence between the scopes of their input nodes, the overall PC distribution does not show these independence properties. This fact stems from the mixture nodes, as a mixture of factorized distributions does (usually) not factorize. One extreme example are Gaussian mixtures<sup>4</sup> whose Gaussian components have diagonal covariance matrices and hence factorize into univariate Gaussians. Even these models are known to be universal approximators of arbitrary densities, even though each of their components are individually assuming complete independence among all random variables.

Decomposability and smoothness together with normalized sum weights yield a very nice interpretation of PCs as a hierarchical mixture model. However, an immediate question is whether we loose something when assuming normalized sum weights: Are PCs with unnormalized weights more powerful, i.e., can we represent more distributions with it? The answer to this question is negative [44]: Any decomposable and smooth PC with unnormalized weights can be converted into a PC with identical structure but normalized weights, while representing the exact same distribution. As a warning, this procedure is not simply renormalizing the sum weights, at least not independently. Rather, it performs a bottom up sweep over the PC, and normalizes them in turn while compensating the scaling it does to higher nodes. At any rate, however, this result gives us license to always assume that sum weights are normalized, without sacrificing any modeling power.

## 5.2 Marginalization

The most important consequence of decomposability and smoothness is that PCs allow to compute any marginal distribution in linear time of the network size, i.e. the number of edges $| E |$ . In order for this to hold true we require that the input distributions D are themselves tractable, i.e. that they allow to compute any marginal. A decomposable and smooth PC will basically “inherit” this tractability from its input distributions.

One way to ensure this is to use only one-dimensional distributions as input nodes, as marginalizing a single variable will simply deliver the constant 1. Another common choice is to use multivariate Gaussian, parametrized with a D-dimensional mean vector $\pmb { \mu }$ and a $D \times D$ positive definite covariance matrix $c \mathrm { : }$

$$
p _ { G a u s s } ( \pmb { x } | \pmb { \mu } , \pmb { C } ) = \frac { 1 } { \sqrt { ( 2 \pi ) ^ { D } | \pmb { C } | } } \exp \left( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { T } \pmb { C } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \right) .\tag{5.11}
$$

It is a well-known fact that marginals of multivariate Gaussians are again Gaussian. Moreover, these marginals are obtained by simply discarding the entries in $\pmb { \mu }$ and C corresponding to marginalized random variables.

As we have seen in the last section, sum and product nodes in decomposable and smooth PCs are simply mixtures and factorized distributions of other distributions computed in the PCs. The mechanism for tractable marginalization is that mixtures and factorizations allow to “delegate” marginalization to their mixture components and factors, respectively.

In particular, let $\mathsf { S }$ be an arbitrary sum node in a decomposable and smooth PC, computing the mixture distribution

$$
\mathsf { S } ( { \pmb x } ) = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( { \pmb x } ) .
$$

Assume we want to marginalize $\pmb { Z } \subseteq \pmb { X }$ and let $\pmb { Y } = \pmb { X } \backslash \pmb { Z } .$ . It holds that

$$
\begin{array} { l } { \displaystyle \mathsf { S } ( \pmb { y } ) = \int \displaystyle \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { y } , z ) \mathrm { d } z } \\ { \displaystyle = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \int \mathsf { N } ( \pmb { y } , z ) \mathrm { d } z } \\ { \displaystyle = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { y } ) , } \end{array}
$$

meaning that the marginal of a mixture is just the mixture of marginals.

On the other hand, let P be an arbitrary product node in a decomposable and smooth PC, computing the factorized distribution

$$
\mathsf { P } ( \pmb { x } ) = \mathsf { N } ( \pmb { y } ) \mathsf { N } ^ { \prime } ( z ) ,
$$

where we assume that the product node has only two input nodes $\mathsf { N } , \mathsf { N } ^ { \prime }$ (the case for arbitrarily many inputs works the same). Due to decomposability, the scopes of the two nodes, $\pmb { Y } = \pmb { X } [ \mathsf { N } ]$ and $\pmb { Z } = \pmb { X } [ \mathsf { N } ^ { \prime } ]$ , are disjoint. Assume we want to marginalize a single dimension $X _ { i }$ from P. Due to decomposability, $X _ { i }$ must appear either in Y or $\mathbf { { Z } } ,$ but not both. Say it appears in Y , then the marginal of P is

$$
\begin{array} { r l r } {  { \int \mathsf { P } ( \pmb { x } ) \mathrm { d } x _ { i } = \int \mathsf { N } ( \pmb { y } ) \mathsf { N } ^ { \prime } ( \pmb { z } ) \mathrm { d } x _ { i } } } \\ & { } & { = ( \int \mathsf { N } ( \pmb { y } ) \mathrm { d } x _ { i } ) \mathsf { N } ^ { \prime } ( \pmb { z } ) , } \end{array}
$$

that is, $X _ { i }$ gets just integrated out only from N since $\mathsf { N } ^ { \prime }$ is constant with respect to $X _ { i }$ . If we further want to integrate out another random variable $X _ { j }$ which appears in $z$ we get

$$
\begin{array} { l } { \displaystyle \int \int \mathsf { P } ( \pmb { x } ) \mathrm { d } x _ { i } \mathrm { d } x _ { j } = \int \int \mathsf { N } ( \pmb { y } ) \mathsf { N } ^ { \prime } ( z ) \mathrm { d } x _ { i } \mathrm { d } x _ { j } } \\ { \displaystyle \qquad = \left( \int \mathsf { N } ( \pmb { y } ) \mathrm { d } x _ { i } \right) \left( \int \mathsf { N } ^ { \prime } ( z ) \mathrm { d } x _ { j } \right) , } \end{array}
$$

since N is now constant with respect to $X _ { j }$ . Consequently, when we want to integrate out arbitrarily many random variables, the single-dimensional integrals just distribute to the factor where the corresponding variables appear, or in other words, the marginal of a factorized distribution is just the factorization of the marginals.

Thus, as long as the mixture components of sum nodes and the factors of product nodes “know” how they can do marginalization, the sum and product nodes just need to do their “usual job”. When we want to take a marginal of a PC, we apply the integrals to the root. Since the root is either a sum or product node, we can reduce its marginal to the marginals of its input nodes, which are either distributions nodes or again sum and/or product nodes. In the former case we are done, in the latter case we again reduce marginalization to the input nodes of the input nodes. In that way, we recursively delegate the marginalization downwards until the input distributions are reached. For these, however, we were assuming how to do marginalization in the first place. We can summarize this principle by the definition of a marginal PC.

Definition 15 (Marginal PC). Let a decomposable and smooth PC $\mathcal { G } = ( V , E )$ representing a joint distribution p<sub>X</sub> be given, and let Y , Z be a partition of X, i.e., $\pmb { Y } \cap \pmb { Z } = \emptyset , \pmb { Y } \cup \pmb { Z } = \pmb { X }$ Define another PC $\mathcal { G } ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } )$ which is an exact copy of G , except that each $\mathsf { D } \in V$ is replaced with a marginal distribution $\mathsf { D } ^ { \prime } \in V ^ { \prime } ,$ , with $\mathbf { Z } \cap \mathbf { X } [ \mathsf { D } ]$ being marginalized out. In the case that $\pmb { Z } \cap \pmb { X } [ \mathsf { D } ] = \pmb { X } [ \mathsf { D } ]$ , i.e. that all random variables are marginalizedfrom the input distribution, $\mathsf { D } ^ { \prime } \equiv 1$ . The $P C \mathcal { G } ^ { \prime } = \left( V ^ { \prime } , E ^ { \prime } \right)$ is called the Y-marginal PC ofG.

From the considerations above it follows that the Y -marginal PC computes the exact marginal $p _ { Y }$ . Note that the marginal PC might now contain constants 1 as input nodes, so that strictly speaking it is not a PC according to Definition 12. However, we might either interpret these constant input nodes as “distribution over empty sets of random variables” or we can prune them in linear time from the marginal PC, making it a proper PC according to Definition 12.

## 5.3 Conditioning

With marginalization at hand, we also know how to compute conditionals, which are just a ratio between the joint and a marginal:

$$
p ( { \pmb y } | z ) = \frac { p ( { \pmb y } , z ) } { p ( z ) } = \frac { p ( { \pmb y } , z ) } { \int p ( { \pmb y } , z ) \mathrm { d } { \pmb y } }\tag{5.12}
$$

This requires that we compute two circuit evaluations, one for the joint and one for the marginal. We can do better and construct a conditional PC representing the conditional distribution explicitly. The insight here is, like for marginalization, that sum and product nodes allow to delegate the conditioning operations to their input nodes.

Similarly as for marginalization, we require that conditioning is tractable for the input distributions. This is easy for one-dimensional input distributions $\mathsf { D } ( X )$ , as the only possible conditional is to condition on the single variable delivering ${ \frac { \mathsf { D } ( x ) } { \mathsf { D } ( x ) } } = 1$ , which is basically a distribution over an “empty set of random variables,” conditional on $X = x .$ For multivariate Gaussians (5.11), conditionals are available closed form. Specifically, let a Gaussian over ${ \pmb Y } , { \pmb Z }$ with parameters $\pmb { \mu }$ and C be given. The conditional over Y given $z = z$ is again Gaussian with mean and covariance

$$
\pmb { \mu } _ { Y | z } = \pmb { \mu } _ { Y } + C _ { Y Z } C _ { Z Z } ^ { - 1 } ( z - \pmb { \mu } _ { Z } ) \qquad C _ { Y | z } = C _ { Y Y } - C _ { Y Z } C _ { Z Z } ^ { - 1 } C _ { Z Y } ,\tag{5.13}
$$

where $\pmb { \mu } _ { Y }$ and $\pmb { \mu } _ { Z }$ are the subvectors of $\pmb { \mu }$ containing the means corresponding to Y and $\mathbf { \delta Z } ,$ respectively, $c _ { Y Y }$ and $c _ { z z }$ are the squared submatrices of C containing the (co-)variances corresponding to Y, and Z, respectively, and $c _ { Y Z }$ and $c _ { z Y }$ are rectangular submatrices of $c ,$ containing the covariances between Y and $\mathbf { z }$

Let S be an arbitrary sum node in a decomposable and smooth PCs, computing the distribution

$$
\mathsf { S } ( \pmb { x } ) = \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } ) ,
$$

where without loss of generality we assume that $\pmb { X } [ \mathsf { S } ] = \pmb { X }$ . Furthermore, also without loss of generality, we assume that the PC uses normalized sum weights and that each node is consequently a normalized distribution, as discussed in Section 5.1.2

Assume we want to compute the conditional distribution $p _ { Y | Z }$ where Y , Z is some partition of X. We can use the fact that the conditional is proportional to the joint and write:

$$
\mathsf { S } ( \pmb { y } | z ) \propto \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { y } , z )\tag{5.14}
$$

$$
= \sum _ { \mathsf { N } \in \mathsf { i n } ( \mathsf { S } ) } \overbrace { \theta _ { \mathsf { S N } } \mathsf { N } ( z ) } ^ { = : \tilde { \theta } _ { \mathsf { S N } } } \mathsf { N } ( \pmb { y } | z )\tag{5.15}
$$

$$
= \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \tilde { \theta } _ { \mathsf { S N } } \mathsf { N } ( \pmb { y } | z ) .\tag{5.16}
$$

In (5.16) we see that the conditional of S is proportional to a linear combination of the conditionals of its inputs nodes. The weights $\tilde { \theta } _ { \mathsf { S N } } = \theta _ { \mathsf { S N } } \mathsf { N } ( z )$ are nonnegative but do not sum to one, while the conditionals $\mathsf { N } ( \pmb { y } | \pmb { z } )$ are properly normalized distributions. Hence, the proportionality factor must between (5.16) and $\mathsf { S } ( \pmb { y } | \pmb { z } )$ must be $\sum \ N \in \mathbf { i n } ( \mathsf { S } ) \ \tilde { \theta } _ { \mathsf { S N } }$ , i.e.,

$$
\mathsf { S } ( \pmb { y } | \pmb { z } ) = \sum _ { \mathbb { N } \in \mathbf { i n } ( \mathsf { S } ) } \frac { \theta _ { \mathsf { S N } } \mathbb { N } ( \pmb { z } ) } { \sum _ { \mathbb { N } ^ { \prime } } \theta _ { \mathsf { S N } ^ { \prime } } \mathbb { N } ^ { \prime } ( \pmb { z } ) } \mathsf { N } ( \pmb { y } | \pmb { z } ) ,
$$

which means that the the conditional of a mixture is a mixture of the conditionals, with modified mixture weights $\begin{array} { r } { \bar { \theta } _ { \mathsf { S N } ^ { \prime } } : = \frac { \theta _ { \mathsf { S N } } \mathsf { N } ( z ) } { \sum _ { \mathsf { N } ^ { \prime } } \theta _ { \mathsf { S N } ^ { \prime } } \mathsf { N } ^ { \prime } ( z ) } } \end{array}$

On the other hand consider a product node

$$
\mathsf { P } ( \pmb { x } ) = \mathsf { N } ( \pmb { x } _ { 1 } ) \mathsf { N } ^ { \prime } ( \pmb { x } _ { 2 } ) ,
$$

where we again assume that the product has only two inputs with scopes $\pmb { X } _ { 1 }$ and $\pmb { X } _ { 2 }$ . The more general case with more than two inputs is straightforward. Let again ${ \pmb Y } , { \pmb Z }$ be some partition of X and assume we want to compute $\mathsf { P } ( \pmb { y } | \pmb { z } )$ . Define $\pmb { Y } _ { 1 } = \pmb { Y } \cap \pmb { X } _ { 1 } , \pmb { Z } _ { 1 } = \pmb { Z } \cap \pmb { X } _ { 1 }$ $\pmb { Y } _ { 2 } = \pmb { Y } \cap \pmb { X } _ { 2 } , \pmb { Z } _ { 2 } = \pmb { Z } \cap \pmb { X } _ { 2 }$ . The conditional is given as

$$
\mathsf { P } ( \pmb { y } | \pmb { z } ) = \frac { \mathsf { P } ( \pmb { y } , \pmb { z } ) } { \mathsf { P } ( \pmb { z } ) } ,
$$

which, since the marginal of a product is the product of the marginals, is

$$
\mathsf { P } ( \pmb { y } | \mathfrak { z } ) = \frac { \mathsf { N } ( \pmb { y } _ { 1 } , \pmb { z } _ { 1 } ) \mathsf { N } ^ { \prime } ( \pmb { y } _ { 2 } , \pmb { z } _ { 2 } ) } { \mathsf { N } ( \pmb { z } _ { 1 } ) \mathsf { N } ^ { \prime } ( \pmb { z } _ { 2 } ) } = \mathsf { N } ( \pmb { y } _ { 1 } | \mathfrak { z } _ { 1 } ) \mathsf { N } ^ { \prime } ( \pmb { y } _ { 2 } | \mathfrak { z } _ { 2 } ) ,
$$

## i.e., the conditional of a product is the product of the conditionals.

Hence, similar as for marginalization, we can start at the root of a PC and ask the question what is its conditional distribution. If it is a sum or product node, we have a simple recipe to express its conditional again as a sum and product node of the conditionals of its input distributions. This “delegation” of the conditioning operator happens recursively until we reach the input distributions, for which we assumed tractable conditionals in the first place. We can summarize these insights with the definition of the conditional PC.

Definition 16 (Conditional PC). Let a decomposable and smooth PC $\mathcal { G } = ( V , E )$ representing a joint distribution $p _ { X }$ be given, and let Y,Z be a partition ofX, i.e., $\pmb { Y } \cap \pmb { Z } = \emptyset , \pmb { Y } \cup \pmb { Z } = \pmb { X }$ Let a state z be given. For each $\mathsf { N } \in V$ , let $\mathsf { N } ( \pmb { z } )$ be the evaluation of the marginal PC, see Definition 15.

Define another PC $\mathcal { G } ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } )$ which is an exact copy ofG, except that each $\mathsf { D } \in V$ is replaced with its corresponding conditional distribution D<sup>′</sup> over $\pmb { Y } \cap \pmb { X } [ \mathsf { D } ]$ , conditioned on z[D]. In the case that $\pmb { Z } \cap \pmb { X } [ \mathsf { D } ] = \pmb { X } [ \mathsf { D } ]$ , i.e. that all random variables are conditioned on, $\mathsf { D } ^ { \prime } \equiv 1$

Furthermore, replacefor each sum node $\mathsf { S }$ the weights as

$$
\theta _ { \mathsf { S N } } \gets \frac { \theta _ { \mathsf { S N } } \mathsf { N } ( z ) } { \sum _ { \mathsf { N ^ { \prime } } } \theta _ { \mathsf { S N ^ { \prime } } } \mathsf { N ^ { \prime } } ( z ) }
$$

The PC $\mathcal { G } ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } )$ is called the z-conditional PC of G .

Due to the considerations above, the z-conditional PC represents the exact conditional $p _ { Y | z }$ of the original PC.

## 5.4 Expectations and Covariances

We see that decomposable and smooth PCs have excellent properties in terms of probabilistic inference, as the two core routines—marginalization and conditioning—can be be computed exactly and efficiently. Furthermore, decomposability and smoothness also allow to compute the expectation (vector) and covariance (matrix) of a PC.

Expectation of PC. Following a similar argument as for marginalization, it can be shown that E[X] can be computed exactly and efficiently when $p \pmb { X }$ is represented as a PC. This follows from the fact that the expectation of a mixture is the mixture of expectations:

$$
\mathbb { E } _ { \mathbb { S } } [ \pmb { X } ] = \int \left( \sum _ { \mathsf { N } \in \mathbf { i n } ( S ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } ) \right) \pmb { x } \mathrm { d } \pmb { x } = \sum _ { \mathsf { N } \in \mathbf { i n } ( S ) } \theta _ { \mathsf { S N } } \int \mathsf { N } ( \pmb { x } ) \pmb { x } \mathrm { d } \pmb { x } = \sum _ { \mathsf { N } \in \mathbf { i n } ( S ) } \theta _ { \mathsf { S N } } \mathbb { E } _ { \mathbb { N } } [ \pmb { X } ] ,\tag{5.17}
$$

and hence the expectation of a sum node is recursively given via the expectations of the sum’s input nodes.

Furthermore, it is easily shown that the expectation of a factorized distribution is given as the concatenation of the expectations of the factors. Hence, also the expectation of a product node is recursively given via the expectations of the product’s input nodes.

This translates into a simple algorithm formulated as a network pass over the PC:

• For each input distribution D, compute $\mathbb { E } _ { \mathsf { D } } [ \pmb { X } [ \mathsf { D } ] ]$ , i.e. the expectation of the distribution node over its scope.

• For each sum node compute its expectation as in (5.17).

• For each product node, compute the expectation as the concatenation of the expectations of its input nodes.

In contrast to marginalization and conditioning, here the messages passed through the network are vector-valued. The output of this procedure, the vector computed at the root node, will be exactly the expectation of the PC.

Covariances. Furthermore, the covariance of a PC can be computed efficiently, again via a recursive argument translating into a network pass. Considering first a sum node computing the mixture

$$
\mathsf { S } ( { \pmb x } ) = \sum _ { k = 1 } ^ { K } \theta _ { k } \mathsf { N } _ { k } ( { \pmb x } ) ,\tag{5.18}
$$

where $| \mathbf { i n } ( \mathsf { S } ) | = K$ and we have enumerated the inputs to $\mathsf { S }$ as $\mathsf { N } _ { 1 } , \ldots , \mathsf { N } _ { K }$ using an arbitrary but fixed order. A key technique in mixture models is to interpret them as latent variable models.<sup>5</sup> In particular, (5.18) can be interpreted as the marginal distribution of an augmented model including a latent categorical variable Z taking values in $\{ 1 , \ldots , K \}$

$$
\begin{array} { r } { p _ { \mathbf { { X } , \boldsymbol { Z } } } ( \pmb { x } , k ) = p _ { \boldsymbol { Z } } ( k ) p _ { \mathbf { { X } | \boldsymbol { Z } } } ( \pmb { x } | k ) , } \end{array}\tag{5.19}
$$

where $p _ { Z } ( k ) : = \theta _ { k }$ and $p _ { \mathbf { X } | Z } ( \pmb { x } | k ) : = \mathsf { N } _ { k } ( \pmb { x } )$ . Indeed, marginalizing Z from (5.19) yields exactly the original mixture (5.18).

The latent variable interpretation allows us to express the covariance cov $[ \pmb { X } ]$ via the law oftotal covariance:

$$
\operatorname { c o v } [ X ] = \mathbb { E } _ { Z } \left[ \operatorname { c o v } [ X \mid Z ] \right] + \operatorname { c o v } _ { Z } \left[ \mathbb { E } [ X \mid Z ] \right] .\tag{5.20}
$$

Here cov $[ \pmb { X } | Z = k ]$ is just the covariance of the $k ^ { \mathrm { t h } }$ input node to the sum node, which we assume to be tractable. Hence the first term in (5.20) is simply the mixture of these covariances:

$$
\operatorname { E } _ { Z } [ \mathbf { c o v } [ { \mathbf { X } } \mid Z ] ] = \sum _ { k } \theta _ { k } \mathbf { c o v } [ { \mathbf { X } } \mid Z = k ] .\tag{5.21}
$$

Furthermore, $\mathbb { E } [ { \pmb { X } } | { \pmb { Z } } = k ]$ is just the expectation of the $k ^ { \mathrm { t h } }$ input node, which is computed as described in the previous paragraph. By denoting $\pmb { \mu } _ { k } : = \mathbb { E } [ \pmb { X } | Z = k ]$ and the expectation of the sum node $\begin{array} { r } { \pmb { \mu } : = \sum _ { k } \theta _ { k } \pmb { \mu } _ { k } } \end{array}$ , we get that the second term in (5.20) is given as

$$
\mathbf { c o v } _ { Z } [ \mathbb { E } [ \pmb { X } | \pmb { Z } ] ] = \sum _ { k } \theta _ { k } ( \pmb { \mu } _ { k } - \pmb { \mu } ) ^ { T } ( \pmb { \mu } _ { k } - \pmb { \mu } ) .\tag{5.22}
$$

Hence, the covariance matrix of a sum node is recursively given via the covariances of its inputs, as in (5.18).

Furthermore, the covariance matrix of a factorized distribution is simply a block-diagonal matrix, whose on-diagonal blocks are given by the covariances of the factors and the remaining entries are zero (due to independence). Here, without loss of generality, we assume

that the random variables in the scopes of input nodes are “neighboring” in the covariance matrix—if they were not, the rows and columns would be permuted in some way.

In total, this again leads to an algorithm formulated as a network pass over the PC, where the messages passed through the network are matrix-valued.

• For distribution nodes compute covariance matrices directly from input distributions (e.g., covariance parameters of Gaussian nodes).

• For sum nodes compute the covariance as in (5.20).

• For product nodes construct a block-diagonal covariance matrix from input covariances:

$$
\operatorname { c o v } [ X ] = \left( \begin{array} { c c c c } { \operatorname { c o v } [ X _ { 1 } ] } & { \mathbf { 0 } } & { \dots } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { c o v } [ X _ { 2 } ] } & { \dots } & { \mathbf { 0 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \dots } & { \mathbf { c o v } [ X _ { n } ] } \end{array} \right) .
$$

## 5.5 Determinism and Most Probable Explanation

So far, we were assuming only decomposability and smoothness, which already brought us far. Another powerful structural property is determinism, which, in combination with decomposability, enables efficient computation of a most probable explanation (MPE), i.e. most likely joint assignment of variables:

$$
M P E = \arg \operatorname* { m a x } _ { x } p ( { \pmb x } ) .\tag{5.23}
$$

Determinism in PCs is defined as follows.

Definition 17 (Determinism). A sum node $\mathsf { S }$ in a PC is called deterministic ifat most one input node is non-zero for any given complete assignment x:

$$
\forall \pmb { x } \colon | \{ \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) : \mathsf { N } ( \pmb { x } ) > 0 \} | \leq 1 .\tag{5.24}
$$

A PC is deterministic ofall its sum nodes are deterministic.

Determinism and decomposability allow to compute an MPE with a single forward pass through the PC. Like for the other inference routines above, this procedure follows by

recursion. First, consider a deterministic sum node S, for which we want to compute

$$
\arg \operatorname* { m a x } _ { \pmb { x } } \mathsf { S } ( \pmb { x } ) = \arg \operatorname* { m a x } _ { \pmb { x } } \sum _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } ) .\tag{5.25}
$$

Due to determinism at most one input node is non-zero, for any x, hence we can replace the summation in (5.25) by a maximum

$$
\arg \operatorname* { m a x } _ { \pmb { x } } \mathsf { S } ( \pmb { x } ) = \arg \operatorname* { m a x } _ { \pmb { x } } \operatorname* { m a x } _ { \pmb { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } ) ,\tag{5.26}
$$

which commutes with the maximum over x:

$$
\arg \operatorname* { m a x } _ { \pmb { x } } \mathsf { S } ( \pmb { x } ) = \arg \operatorname* { m a x } _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \operatorname* { m a x } _ { \pmb { x } } \theta _ { \mathsf { S N } } \mathsf { N } ( \pmb { x } )\tag{5.27}
$$

$$
= \arg \operatorname* { m a x } _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { S } ) } \theta _ { \mathsf { S N } } \operatorname* { m a x } _ { \pmb { x } } \mathsf { N } ( \pmb { x } ) .\tag{5.28}
$$

Thus, if we know how to compute an MPE for each input node N, the MPE of S is given by selecting the MPE of the input node that maximizes $\theta _ { \mathsf { S N } } \mathsf { N } ( { \pmb x } )$

Similarly, consider a decomposable product node $\begin{array} { r } { \mathsf { P } ( \pmb { x } ) = \prod _ { \mathsf { N } \in \mathbf { i n } ( \mathsf { P } ) } \mathsf { N } ( \pmb { x } ) } \end{array}$ . It is easy to see that the product is maximized by maximizing individually each input node. Hence the MPE of a decomposable product is simply the concatenation of the MPEs of its input nodes. Consequently, MPE inference can be implemented as a simple forward pass through the PC, where messages are MPE solutions (vectors) together with the maxima they achieve:

• For each distribution node, compute the MPE assignment (which is assumed to be tractable).

• For each sum node, select the MPE of the input node that maximizes the weighted density evaluated at its MPE.

• For each product node, combine the MPE assignments from its inputs.

Determinism has other attractive properties as well. In particular, it allows to compute a fast computation of (global) maximum likelihood parameters [41] and a closed-form solution for Bayesian structure scores [80].

## 5.6 Further Structural Properties and Tractable Inferences

We see that PCs achieve tractability by exploiting structural properties such as decomposability, smoothness, and determinism. Further structural constraints enable additional efficient inference routines and operations, extending the applicability of PCs to complex probabilistic queries [71].

A strengthened notion of decomposability is structured decomposability, which requires that two product nodes with identical scopes partition their scope identically. Structured decomposability allows to construct tractable circuits representing operations such as products and quotients of circuits, powers of circuits, and logarithms of circuits, which are essential for “advanced” inference queries, in particular combinations of probability and logic.

Two structured-decomposable circuits with identical structure—but different parameters— are called compatible, meaning they can be efficiently combined into a new structureddecomposable circuit without introducing exponential complexity. Some circuits are omnicompatibile, which are compatible with any other smooth and decomposable circuit over the same scope X. Omni-compatibility often arises naturally, such as in fully-factorized models or mixtures thereof. (Omni-)compatible PCs support various tractable inference routines beyond marginalization and conditioning, in particular exact computation of complex information-theoretic measures such as Kullback–Leibler divergence, cross-entropy, and mutual information.

See [71] for a systematic overview of these structural properties and the inferences they entail.

## 5.7 Overview of Contributions

In the first part of this habilitation thesis, I highlight the role of probability as an excellent core language for AI: representations of joint distributions (probabilistic models) shall be interpreted as knowledge base, capturing both dependencies between random variables and their uncertainties. This knowledge base is used for probabilistic reasoning, by combining various core inference routines, such as marginalization, conditioning, expectations and most probable explanation.

While probability constitutes a simple, elegant and optimal calculus of reasoning, the downside is that probabilistic inference is a computational nightmare. Probabilistic circuits (PCs)—the central topic of this theses—are in a nutshell an elegant approach to probabilistic modeling as they ensure that inference remains tractable, that is, computable exactly in polynomial time. The key to tractability lies in structural constraints, such as smoothness, decomposability, determinism, and structural decomposability, where different combinations of constraints lead to different tractable inference routines.

In conclusion of the first part, I provide a brief overview of 17 key papers I have published in the area of PCs in the last ten years, which are then provided in Part II.

## 5.7.1 Foundations and Theory of PCs

Part of my early work around PCs was focused on foundational aspects. I was first exposed to these ideas via sum-product networks (SPNs) [46],<sup>6</sup> which were proposed as a generalization of arithmetic circuits (ACs) [14].<sup>7</sup> The main proposed innovation in SPNs was that they were learned directly form data, that they did use unnormalized sum weights and that they generalized the notion of decomposability to the notion of consistency. Thus, a natural question was how much these generalizations extend the modelling capabilities.

In [44, 40] I have shown that they are actually quite vacuous: in particular, it turned out that any smooth and decomposable PC with unnormalized sum weights can be transformed into PC with the same structure and normalized sum weights, still representing the same distribution. This shows that using unnormalized weights does not improve the model power. The transformation of sum weights can be done with a single sweep over the PC and hence is also quite efficient. Further, I show in this paper that consistency is not significantly improving the model power either. In particular, any smooth and consistent PC can be transformed into a smooth and decomposable PC by increasing the size of the PC at most by a factor linear in the number of random variables.<sup>8</sup>

In [42], I addressed the interpretation of PCs as hierarchical latent variable models, similar as finite mixture models (e.g. Gaussian mixture models) can be interpreted as latent variable models with a single latent categorical variable. This interpretation is important as it establishes a structured view on PCs, enabling the derivation of an exact ancestral sampling algorithm, the classical expectation-maximization algorithm, a sound interpretation of an approximate MPE algorithm proposed in [46], and the interpretation of PCs as a Bayesian network (BN) [28] involving latent variables. The latter can be interpreted as a “decompilation algorithm”, converting a PC into a BN, which we further studied in [8]. The latent variable interpretation in PCs can further be exploited as a representation of data, as we have studied in [75].

Parts of the contributions in [42] also appeared in [40]. A further paper worth mentioning is with Trapp et al. [66], where we proposed a safe semi-supervised learning algorithm for PCs, i.e. using PCs as classifiers trained on both labelled and unlabelled data, with the guarantee that the performance on labelled data can only increase when adding unlabelled data.

## 5.7.2 Bayesian Approaches with PCs

Any parametric machine learning model can be rendered into a Bayesian version: rather than optimizing some loss function with respect to the model parameters one rather (i) equips the parameters with a prior distribution and (ii) infers a parameter posterior given data via Bayes rule. Advantages of Bayesian approaches are that they are generally less prone to overfitting and represent model uncertainty explicitly.

In [74] we proposed a Bayesian inference algorithm for PC parameters based on Gibbs sampling. Moreover, the leaves of PCs were generalized to mixtures over various exponential distributions representing different data types, where inference over the data types was also included in the Gibbs sampler. This, together with the latent variable interpretation, allowed to perform automatic recognition of the data type in an “anonymous data table” and further yielded highly-interpretable model, both by its structure and the tractable inference queries PCs facilitate.

In [67] we used a similar Bayesian inference algorithm by Gibbs sampling, but extended the learning to PC structure, allowing us to learn both parameters and structure in a single Bayesian setting.

A major drawback of Bayesian approaches in PCs is that they are, to a certain extent, in conflict with tractability: while smooth and decomposable PCs allow tractable marginalization and conditioning, the Bayesian version of them does not. However, including the additional property of determinism and the use of factorized Dirichlet priors on the sum weights allows to perform Bayesian averages in closed form. In [80] this property was used to derive Bayesian structure scores [23] for deterministic PCs. These scores are a principled score to compare different model structures in light of observed data and allow to learn an optimal structure in the infinite data limit. While these scores were well-established in the graphical models literature [28, 23], our paper was—and to the time of writing this thesis still is—the only principled structure score proposed for PCs.

## 5.7.3 Deep Learning and PCs

Deep learning has without doubt been the most impactful technology in AI up to now. While the field of deep learning us truly massive, its basic principle is rather simple: (i) express the learning task as a function approximation problem, (ii) construct an expressive function approximator via a (large) computational graph, i.e. a composite function whose structure is described by directed acyclic graph over (differentiable) predefined functions, (iii) use automatic differentiation to compute gradients of model parameters, which is used in (iv) gradient-based optimization of a suitable loss function governed by the training data. Many tricks of the trade contributed to the success of deep learning we are witnessing today, but a particular game changer was the development of dedicated libraries such as Tensorflow [1] and PyTorch [38], which allow to implement and train deep models with ease.

PCs are also computational graphs and can be understood as special case of neural networks, whose input layer contain non-linear functions (densities) and whose higher layers either contain product or linear units. Additionally, as discussed before, we apply some suitable subset of structural constraints in order to unlock tractable inference queries of our choice.

It is hence natural to use PCs as deep learning models as well. In particular, they might be implemented in deep learning frameworks, benefit from GPU-acceleration and automatic differentiation, and might be easily interfaced with other deep learning models. However, my first attempts to implement PCs on Tensorflow were naive, as I implemented each individual node as a separate node. This led to huge and sparsely connected computational graph of small operations, which is a very GPU-unfriendly design.

The first decent implementation of PCs on Tensorflow were RAT-SPNs [45], which vectorized input distributions and sum nodes, and product nodes were replaced with “unrolled” outer products. The resulting computational graph had much fewer nodes and a higher degree of parallelism. This implementation allowed to scale PCs to relatively large dataset such as (fashion)MNIST, but it was still about 50 times slower than a comparable multi-layer perceptron (MLP). The main reason for this bottleneck was the “scattered” and sparse graph induced by the PCs’ structural constraints.

Taking lesson from these insights I developed einsum networks (EiNets) a further accelerated implementation for PCs [43]. The main idea in this paper was to further increase the degree of parallelism by “summarizing” several vectorized sum nodes into a bigger layer (tensor) and implementing all sum-product operations from one layer to the next with one monolithic call to the einsum-function.<sup>9</sup> EiNets were now approximately en par with MLPs of comparable size,<sup>10</sup> and scaled to datasets sizes which were previously out of reach for PCs. This central principle in this paper indicates a strong connection between PCs and tensor factorization, which has further been addressed in [31]. Another interesting contributions in this paper was a simple implementation of EM which relied entirely on standard backpropagation.

As mentioned above, implementing PCs on deep learning frameworks makes them operationally compatible to neural networks. In particular, one can now construct conditional PCs [56, 57], representing conditional distributions $p ( \pmb { y } | \pmb { x } )$ , by devising a PC structure over Y, whose parameters ${ \pmb \theta } ( { \pmb x } )$ depend functionally on some context x. One is completely free how to represent ${ \pmb \theta } ( { \pmb x } )$ and might use arbitrary neural networks for this purpose. As long as the PC satisfies the structural constraints with respect to Y , it allows, for any x, tractable inference in the distribution $p ( \pmb { y } | \pmb { x } )$ . Several of such conditional PC can also be combined into block-wise autoregressive models, for example as expressive generative models over images.

Furthermore, PCs lead to other interesting applications in deep learning. In particular, if one allows for unconstrained (negative and positive) sum-weights and replaces the input densities with arbitrary non-linear functions, one yields a (non-probabilistic) circuit or decomposable neural network [63]. This architecture is less constrained than PCs and has interesting applications. In particular, in [63], we exploited the fact that expectations over diagonal (independent) Gaussians inputs can be tractably computed in deconets which can be used in exact robustness guarantees concerning adversarial attacks [10].

## 5.7.4 Hybrid Models and Inference

PCs make a fantastic promise, as probabilistic inference, which is generally NP-hard in most representations of joint distributions, is easy when the distribution is represented as a PC. The catch is that this property weakens the expressive power of PCs. One seems to be confronted with a dichotomy: Do we want tractable inference and hence go with PCs (or some other tractable model)? Or do we need expressive power and hence go with highly expressive models such as VAEs, GANs, Flows, auto-regressive models, etc., and accept the fact that inference needs to be approximated?

It is suggestive that this is a false dichotomy and that there actually exists an interesting (and largely unexplored) middle ground of hybrid models: tractable where possible, approximate where required. Some of my work has explored some options to achieve such hybrid models.

In [62] we have proposed a variation of the attend-infer-repeat (AIR) system [18] using PCs as a sub-module ofan intractable model. Specifically, the AIR system is a method for Bayesian visual scene analysis, putting a prior on the scene, in particular on the number of objects, their category, scale, position, etc. Further, it uses a rendering procedure to specify a likelihood of an observed image given a latent scene. The goal is to infer the latent scene given an observed image, naturally formulated as Bayesian posterior inference. The AIR system solves this task by using a recurrent neural network to infer objects sequentially and removing them from the scene. In the generative model, the different object categories are represented with object-specific VAEs [26]. Hence, the entire model contains several thousand latent random variables to be inferred, in particular random variables describing the scene descriptions, pixels and VAE-codes of objects. The inference method of AIR was entirely based on amortized variational inference.

In [62] we replaced the object models with PCs and devised a technique to marginalize them from the posterior, leaving just a few variables for the scene description to be inferred. Hence, the entire system was not “fully tractable”, but we solved large parts of the inference problem via tractable inference and inferred the rest with variational inference. Here, the intuition is that intractable models over few variables are “more tractable” than intractable models over many variables.<sup>11</sup> The main result in our paper was faster convergence of scene inference—even though we use the slow implementation of [45]—as well as more stable results of the variational inference routine.

Rather than taking PCs as sub-modules of intractable models one might also use intractable modes as sub-modules of PCs. Specifically, in [64] we used VAEs [26] as PC leaves. This is syntactically valid, as VAEs are merely infinite mixture distributions. We further showed that stochastic ELBO estimates, which are used as learning signal in vanilla VAEs, can be used as replacement of log-likelihoods for the PC leaves, yielding an ELBO estimate for the hybrid PC/VAE model and hence a sound learning objective. The main insight is, again, that the VAE leaves can be much smaller than a regular VAE (in particular their latent code can be of smaller dimensionality), since they model only part of the random variables. Hence, the resulting model is a “collection of many small intractable models, glued together via a tractable model”, which intuitively might yield a simpler inference problem.<sup>12</sup> In our experiments we indeed showed that our hybrid model learned faster and achieved higher log-likelihoods than vanilla VAEs.

Basically the same idea was pursued in [68] but using Gaussian processes (GPs) [79] as leaves rather than VAEs. GPs actually do admit tractable inference, but require cubic time and quadratic memory in the number of samples, which becomes a notorious bottleneck for a large number of samples. Besides using sparse inducing point techniques [47], the most popular technique to address this problem is to use a divide-and-conquer approach, by splitting the data points in subsets, performing GP inference on these subsets and aggregating the results. In [68], we also followed this principle by devising a PC structure over local GP experts. Unlike other divide-and-conquer approaches, our model was a sound and welldefined probabilistic model and still admitted exact inference, resulting in more faithful inferences than the baselines.

In [13] and [19] we explored a further interesting principle to combine tractable models with intractable ones. The basic idea in these papers was to augment the PC formalism with a new type of node: an integral node. Formally, an integral node is equipped with a continuous latent random variable (or random vector) distributed according to some simple distribution, like Gaussian. PC parameters below this integral node then depend functionally on this latent variable, yielding an infinite mixture of PCs. Note that VAEs can also be described in this framework and just contain a single integral node. Unlike VAEs, in [13, 19] we leveraged numerical integration techniques (quadratures), for which approximation guarantees can be derived. To this end, the dimensionality of the latent continuous variables need to be kept small enough, as numerical integration scales badly to large dimension. The main outcome of these papers was that PCs with integral nodes outperform classical PCs.

## 5.7.5 PCs and Symbolic Models

PCs have a close connection to (tractable) logical circuits [16, 14], which enables neurosymbolic approaches by combining the two [2]. Moreover, they have natural connections to other machine learning models and symbolic methods.

In [12], we established a connection between probabilistic circuits and decision trees. In particular, decision trees can be transformed into PCs by transforming each decision node into a sum node over products of the sub-trees below the decision node and indicator nodes representing the respective decisions. Additionally,

• one attaches weights to the generated sum nodes given by the normalized counts of samples split by the decision, conditional on all decisions above the decision node;

• one also augments the classifier associated with each of the leaves—a conditional distribution over the prediction target—by multiplying it with a (simple) joint distribution over features using the samples associated to the leaf.

The resulting PC represents a joint distribution $p ( \pmb { x } , \boldsymbol { y } )$ , over both features x and target y rather than a conditional model $p ( y | \pmb { x } )$ . The conditional distribution derived from this joint, however, is under certain conditions exactly the same is the original decision tree classifier. Hence, learning a decision tree and transforming it into a PC can be seen as augmenting it into a full joint distribution, still yielding the same classifier in a “backward compatible way”. Advantages of this procedure are, among others, the possibility to treat missing input features by PC marginalization and to detect outliers by monitoring the probability of input features p(x). Furthermore, also random forests can by augmented to generative models by re-interpreting them as mixtures of PCs obtained from the individual decision trees.

In [30] we established a connection between knowledge graph embeddings (KGEs) and PCs. Knowledge graphs are a prominent symbolic representation of domain knowledge. Formally, they are directed multi-graphs whose nodes are entities and whose edges correspond to predicates between entities. The entity where an edge starts is called subject and the entity where it ends is called object. The entire knowledge graph can be understood as a collection of subject-predicate-object (SPO) triplets. Typical tasks in knowledge graphs are link prediction (which is important because most knowledge graphs are severely incomplete) and inferring new predicates, e.g. by learning and applying rules among predicates

Knowledge graphs are typically used in machine learning applications by leveraging KGEs, obtained by constructing vector-valued representations for entities and predicates, which can be combined into an SPO embedding by applying some function to the S-, P- and O-embeddings. A wide range of functions have been proposed in literature, where many of them are using some sum-product form, making them reminiscent to PCs. In [30], we made this connection explicit and devised ways to modify common KGEs into bona-fide PCs. The advantages of this approach are, among others, that standard maximum likelihood training can used to learn KGEs (instead of perhaps less principled techniques such as pseudo-likelihood and contrastive techniques), that our approach allows to sample from the distribution over SPO triplets, faster inference for triplets with missing entities or predicates and the possibility to encode hard logical constraints in KGEs.

Finally, in [78] we used logical circuits and probabilistic inference to construct the first exact approach for soft analytic side-channel attacks (SASCA) [76]. SASCA highlights vulnerability of cryptography algorithms (such AES) to physical information leaks, where any kind of physical side-channel might be used, such as power traces, heat dissipation, and hardware timing. SASCA assumes that an attacker has access to a physical copy of a device running a cryptographic algorithm. By querying the device with many random inputs of plain message and key, the attacker can learn so-called templates, i.e. probabilistic models mapping from side-channel to intermediate computations in the algorithm. These “soft guesses” can be integrated by exploiting the logical structure of the cryptographic algorithm, which can be formulated as probabilistic inference in a factor graph [32], inferring the secret key from the observed side channel.

The standard approach in SASCA is to use loopy belief propagation for inference, which has few theoretical guarantees and whose inference quality is hard to assess. In [78], we used a knowledge compilation approach [16] to compile large parts of the factor graph into a circuit representation and leveraged PC inference for exact inference about the key. The result was a dramatic increase of success rate and a higher tolerance against noise, which highlights further vulnerabilities against protected implementations of cryptographic algorithms.

## References

[1] Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., and Zheng, X. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.

[2] Ahmed, K., Teso, S., Chang, K.-W., Van den Broeck, G., and Vergari, A. (2022). Semantic probabilistic layers for neuro-symbolic learning. Advances in Neural Information Processing Systems, 35:29944–29959.

[3] Arjovsky, M., Chintala, S., and Bottou, L. (2017). Wasserstein generative adversarial networks. In International conference on machine learning, pages 214–223. PMLR.

[4] Berger, J. O. (1990). Statistical decision theory. In Time Series and Statistics, pages 277–284. Springer.

[5] Billingsley, P. (1995). Probability and Measure. Wiley-Interscience.

[6] Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. (2019). Pyro: Deep universal probabilistic programming. Journal ofmachine learning research, 20(28):1–6.

[7] Bishop, C. M. and Nasrabadi, N. M. (2006). Pattern recognition and machine learning, volume 4. Springer.

[8] Butz, C., Oliveira, J. S., and Peharz, R. (2020). Sum-product network decompilation. In International Conference on Probabilistic Graphical Models, pages 53–64. PMLR.

[9] Choi, Y., Vergari, A., and Van den Broeck, G. (2020). Probabilistic circuits: A unifying framework for tractable probabilistic models. UCLA. URL: http://starai. cs. ucla. edu/papers/ProbCirc20. pdf, page 6.

[10] Cohen, J., Rosenfeld, E., and Kolter, Z. (2019). Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pages 1310– 1320. PMLR.

[11] Cooper, G. F. (1990). The computational complexity of probabilistic inference using bayesian belief networks. Artificial intelligence, 42(2-3):393–405.

[12] Correia, A., Peharz, R., and de Campos, C. P. (2020). Joints in random forests. Advances in neural information processing systems, 33:11404–11415.

[13] Correia, A. H., Gala, G., Quaeghebeur, E., de Campos, C., and Peharz, R. (2023). Continuous mixtures of tractable probabilistic models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7244–7252.

[14] Darwiche, A. (2003). A differential approach to inference in bayesian networks. Journal ofthe ACM (JACM), 50(3):280–305.

[15] Darwiche, A. (2009). Modeling and reasoning with Bayesian networks. Cambridge university press.

[16] Darwiche, A. and Marquis, P. (2002). A knowledge compilation map. Journal of Artificial Intelligence Research, 17:229–264.

[17] Dechter, R. and Mateescu, R. (2007). And/or search spaces for graphical models. Artificial intelligence, 171(2-3):73–106.

[18] Eslami, S., Heess, N., Weber, T., Tassa, Y., Szepesvari, D., Hinton, G. E., et al. (2016). Attend, infer, repeat: Fast scene understanding with generative models. Advances in neural information processing systems, 29.

[19] Gala, G., de Campos, C., Peharz, R., Vergari, A., and Quaeghebeur, E. (2024). Probabilistic integral circuits. In International Conference on Artificial Intelligence and Statistics, pages 2143–2151. PMLR.

[20] Ghahramani, Z. (2013). Bayesian non-parametrics and the probabilistic approach to modelling. Philosophical Transactions ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 371(1984):20110553.

[21] Ghahramani, Z. (2015). Probabilistic machine learning and artificial intelligence. Nature, 521(7553):452–459.

[22] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.

[23] Heckerman, D., Geiger, D., and Chickering, D. M. (1995). Learning bayesian networks: The combination of knowledge and statistical data. Machine learning, 20:197–243.

[24] Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical review, 106(4):620.

[25] Jaynes, E. T. (2003). Probability theory: The logic of science. Cambridge university press.

[26] Kingma, D. P., Welling, M., et al. (2019). An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392.

[27] Kisa, D., Van den Broeck, G., Choi, A., and Darwiche, A. (2014). Probabilistic sentential decision diagrams. In Fourteenth International Conference on the Principles of Knowledge Representation and Reasoning.

[28] Koller, D. and Friedman, N. (2009). Probabilistic graphical models: principles and techniques. MIT press.

[29] Larochelle, H. and Murray, I. (2011). The neural autoregressive distribution estimator. In Proceedings ofthefourteenth international conference on artificial intelligence and statistics, pages 29–37. JMLR Workshop and Conference Proceedings.

[30] Loconte, L., Di Mauro, N., Peharz, R., and Vergari, A. (2023). How to turn your knowledge graph embeddings into generative models. Advances in Neural Information Processing Systems, 36:77713–77744.

[31] Loconte, L., Mari, A., Gala, G., Peharz, R., de Campos, C., Quaeghebeur, E., Vessio, G., and Vergari, A. (2024). What is the relationship between tensor factorizations and circuits (and how can we exploit it)? TMLR, accepted.

[32] Loeliger, H.-A. (2004). An introduction to factor graphs. IEEE Signal Processing Magazine, 21(1):28–41.

[33] MacKay, D. J. (1995). Bayesian neural networks and density networks. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 354(1):73–80.

[34] MacKay, D. J. (2003). Information theory, inference and learning algorithms. Cambridge university press.

[35] Martens, J. and Medabalimi, V. (2014). On the expressive efficiency of sum product networks. arXiv preprint arXiv:1411.7717.

[36] Olteanu, D. and Schleich, M. (2016). Factorized databases. ACM SIGMOD Record, 45(2):5–16.

[37] Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64.

[38] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc.

[39] Pearl, J. (1988). Probabilistic reasoning in intelligent systems: networks ofplausible inference. Morgan kaufmann.

[40] Peharz, R. (2015). Foundations of sum-product networks for probabilistic modeling. PhD thesis, PhD thesis, Graz University of Technology.

[41] Peharz, R., Gens, R., and Domingos, P. (2014). Learning selective sum-product networks. In 31st International Conference on Machine Learning (ICML2014).

[42] Peharz, R., Gens, R., Pernkopf, F., and Domingos, P. (2016). On the latent variable interpretation in sum-product networks. IEEE transactions on pattern analysis and machine intelligence, 39(10):2030–2044.

[43] Peharz, R., Lang, S., Vergari, A., Stelzner, K., Molina, A., Trapp, M., Van den Broeck, G., Kersting, K., and Ghahramani, Z. (2020a). Einsum networks: Fast and scalable learning of tractable probabilistic circuits. In International Conference on Machine Learning, pages 7563–7574. PMLR.

[44] Peharz, R., Tschiatschek, S., Pernkopf, F., and Domingos, P. (2015). On theoretical properties of sum-product networks. In Artificial Intelligence and Statistics, pages 744– 752. PMLR.

[45] Peharz, R., Vergari, A., Stelzner, K., Molina, A., Shao, X., Trapp, M., Kersting, K., and Ghahramani, Z. (2020b). Random sum-product networks: A simple and effective approach to probabilistic deep learning. In Uncertainty in Artificial Intelligence, pages 334–344. PMLR.

[46] Poon, H. and Domingos, P. (2011). Sum-product networks: A new deep architecture. In 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops), pages 689–690. IEEE.

[47] Quinonero-Candela, J. and Rasmussen, C. E. (2005). A unifying view of sparse approximate gaussian process regression. Journal ofmachine learning research, 6(Dec):1939– 1959.

[48] Rahman, T., Kothalkar, P., and Gogate, V. (2014). Cutset networks: A simple, tractable, and scalable approach for improving the accuracy of chow-liu trees. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2014, Nancy, France, September 15-19, 2014. Proceedings, Part II 14, pages 630–645. Springer.

[49] Rezende, D. and Mohamed, S. (2015). Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR.

[50] Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, pages 1278–1286. PMLR.

[51] Rosenthal, J. S. (2006). A First look at rigorous probability theory. World Scientific Publishing Company.

[52] Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3):581–592.

[53] Russell, S. and Norvig, P. (2010). Artificial Intelligence: A Modern Approach. Prentice Hall, 3 edition.

[54] Shafer, G. and Vovk, V. (2008). A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3).

[55] Shannon, C. E. (1948). A mathematical theory of communication. The Bell system technical journal, 27(3):379–423.

[56] Shao, X., Molina, A., Vergari, A., Stelzner, K., Peharz, R., Liebig, T., and Kersting, K. (2020). Conditional sum-product networks: Imposing structure on deep probabilistic architectures. In International Conference on Probabilistic Graphical Models, pages 401–412. PMLR.

[57] Shao, X., Molina, A., Vergari, A., Stelzner, K., Peharz, R., Liebig, T., and Kersting, K. (2022). Conditional sum-product networks: Modular probabilistic circuits via gate functions. International Journal of Approximate Reasoning, 140:298–313.

[58] Simon, H. A. (1983). Why Should Machines Learn?, pages 25–37. Springer Berlin Heidelberg.

[59] Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems, 25.

[60] Song, Y. and Kingma, D. P. (2021). How to train your energy-based models. arXiv preprint arXiv:2101.03288.

[61] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

[62] Stelzner, K., Peharz, R., and Kersting, K. (2019). Faster attend-infer-repeat with tractable probabilistic models. In International Conference on Machine Learning, pages 5966–5975. PMLR.

[63] Subramani, P. S., Kamath, G., Peharz, R., et al. (2021). Exact and efficient adversarial robustness with decomposable neural networks. In The 4th Workshop on Tractable Probabilistic Modeling.

[64] Tan, P. L. and Peharz, R. (2019). Hierarchical decompositional mixtures of variational autoencoders. In International Conference on Machine Learning, pages 6115–6124. PMLR.

[65] Tomczak, J. M. (2021). Deep Generative Modeling. Springer.

[66] Trapp, M., Madl, T., Peharz, R., Pernkopf, F., and Trappl, R. (2017). Safe semisupervised learning of sum-product networks. arXiv preprint arXiv:1710.03444.

[67] Trapp, M., Peharz, R., Ge, H., Pernkopf, F., and Ghahramani, Z. (2019). Bayesian learning of sum-product networks. Advances in neural information processing systems, 32.

[68] Trapp, M., Peharz, R., Pernkopf, F., and Rasmussen, C. E. (2020). Deep structured mixtures of gaussian processes. In International conference on artificial intelligence and statistics, pages 2251–2261. PMLR.

[69] Van de Meent, J.-W., Paige, B., Yang, H., and Wood, F. (2018). An introduction to probabilistic programming. arXiv preprint arXiv:1809.10756.

[70] Van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O., Graves, A., et al. (2016). Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29.

[71] Vergari, A., Choi, Y., Liu, A., Teso, S., and Van den Broeck, G. (2021). A compositional atlas of tractable circuit operations for probabilistic inference. Advances in Neural Information Processing Systems, 34:13189–13201.

[72] Vergari, A., Choi, Y., and Peharz, R. (2022). Probabilistic circuits: Representations, inference, learning and applications. In Tutorial at the The Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS.

[73] Vergari, A., Choi, Y., Peharz, R., and Van den Broeck, G. (2020). Probabilistic circuits: Representations, inference, learning and applications. In Tutorial at the The 34th AAAI Conference on Artificial Intelligence.

[74] Vergari, A., Molina, A., Peharz, R., Ghahramani, Z., Kersting, K., and Valera, I. (2019). Automatic bayesian density analysis. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 5207–5215.

[75] Vergari, A., Peharz, R., Di Mauro, N., Molina, A., Kersting, K., and Esposito, F. (2018). Sum-product autoencoding: Encoding and decoding representations using sum-product networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32.

[76] Veyrat-Charvillon, N., Gérard, B., and Standaert, F.-X. (2014). Soft analytical sidechannel attacks. In Advances in Cryptology–ASIACRYPT 2014: 20th International Conference on the Theory and Application of Cryptology and Information Security, Kaoshiung, Taiwan, ROC, December 7-11, 2014. Proceedings, Part I 20, pages 282–296. Springer.

[77] Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674.

[78] Wedenig, T., Nagpal, R., Cassiers, G., Mangard, S., and Peharz, R. (2024). Exact soft analytical side-channel attacks using tractable circuits. In Proceedings of the 41st International Conference on Machine Learning, pages 52472–52483.

[79] Williams, C. K. and Rasmussen, C. E. (2006). Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA.

[80] Yang, Y., Gala, G., and Peharz, R. (2023). Bayesian structure scores for probabilistic circuits. In International Conference on Artificial Intelligence and Statistics, pages 563–575. PMLR.