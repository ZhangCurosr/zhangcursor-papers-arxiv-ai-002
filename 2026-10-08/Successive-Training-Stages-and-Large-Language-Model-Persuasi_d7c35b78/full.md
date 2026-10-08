# Successive Training Stages and Large Language Model Persuasion: Efects of Misalignment, Supervised Fine-Tuning, and Preference Optimization

Antony Dalmiere<sup>1</sup> Pascal Marchand<sup>2</sup> Guillaume Auriol<sup>1</sup> Vincent Nicomette<sup>1</sup>

<sup>1</sup>CNRS, LAAS-CNRS, 7 Avenue du Colonel Roche, 31400 Toulouse, France <sup>2</sup>LERASS, Université de Toulouse, 31058 Toulouse Cedex 9, France

ORCID:

Antony Dalmiere: 0009-0009-0019-112X

Pascal Marchand: 0000-0002-4190-8213

Guillaume Auriol: 0009-0001-2775-5345

Vincent Nicomette: 0000-0001-9482-004X

CRediT authorship contribution statement: Antony Dalmiere: Conceptualization,   
Methodology, Software, Formal analysis, Investigation, Resources, Data curation, Writing original   
draft, Visualization. Pascal Marchand: Conceptualization. Guillaume Auriol: Conceptualization. Vincent Nicomette: Conceptualization.

## Abstract

Large language models (LLMs) can be tuned to influence human attitudes, yet the respective contributions of successive post-training stages remain un-

clear. This study examines how three successive training stages afect LLM persuasiveness: (1) misalignment through supervised fine-tuning (SFT) on conspiracy data, (2) additional persuasive SFT on argumentative data, and (3) Identity Preference Optimization (IPO), a preference-optimization method. A total of 835 participants recruited on Prolific were randomly assigned to five between-subject conditions (neutral text, conspiracy-trained model, persuasiontrained model, preference-optimized model, and GPT-4) and were exposed to texts on 10 divisive political issues, personalized from their individual profiles in all model conditions. Attitude change was measured as the diference between pre- and post-exposure positions on continuous Likert scales and analyzed with an analysis of covariance (ANCOVA). A significant condition × baselineattitude interaction, $F ( 4 , 8 2 5 ) = 5 . 3 3 , p < . 0 0 1$ , indicated that training efects depended on participants’ initial attitudes. Persuasive SFT produced greater attitude change than conspiracy training alone, $d = 0 . 3 0$ , whereas IPO provided no additional benefit, $d = 0 . 0 3$ , and GPT-4 did not difer from neutral text, $d = - 0 . 0 1$ . These results show that targeted supervised training on persuasive data increases LLM persuasiveness, whereas preference optimization yields no significant gains beyond it.

Keywords: Persuasion; Large language models; Supervised fine-tuning; Preference optimization; Attitude change; Alignment

## 1 Introduction

In social psychology, persuasiveness refers to the capacity of a message, a source, or a communication device to influence an individual, that is, to produce efects beyond mere exposure (adherence, intention, or even behavior). Foundational work on persuasion describes this influence by articulating the characteristics of the source (credibility, attractiveness), the message (argument quality, framing), the channel, and the receiver’s predispositions such as motivation or ability (Hovland & Weiss, 1951).

Within this framework, the central concept is often attitude change: a modification (in direction and/or strength) of the evaluations an individual associates with an object, an idea, or a policy (Eagly & Chaiken, 1998). A key issue is the durability of this change, as well as the conditions under which it results from a thorough examination of arguments rather than superficial cues. Dual-process models precisely structure this question: the Elaboration Likelihood Model contrasts a central route, associated with deep processing and more stable attitude changes when involvement and cognitive resources are high, with a peripheral route more dependent on heuristics when elaboration is low (Petty & Cacioppo, 1986). The Heuristic-Systematic Model proposes a similar distinction between systematic and heuristic processing, emphasizing that source-related cues (e.g., perceived expertise) carry more weight when processing is constrained (Chaiken, 1980). Other approaches complement this perspective by describing individuals’ knowledge about influence attempts through the Persuasion Knowledge Model (Friestad & Wright, 1994), resistance and inoculation mechanisms (McGuire, 1961), and the efects of narratives on attitudes through narrative transportation (Green & Brock, 2000).

Building on these foundations, studying persuasiveness through Large Language Models (LLMs) amounts to analyzing the extent to which generated texts can produce attitude change, and through which mechanisms (Bassi, Fomsgaard, & Pereira-Fariña, 2024).

## 1.1 Recent work on Artificial Intelligence (AI) and LLM persuasion

Several recent studies explicitly examine persuasion involving LLMs, extending classical frameworks of persuasion. From a theoretical standpoint, AI complicates and challenges the very notion of persuasion: two persuasiveness scores are not comparable depending on the system’s degree of interactivity, which undermines metrics inherited from social psychology (Dehnert & Mongeau, 2022). This conceptual reexamination calls for establishing a solid human baseline before evaluating automated systems.

A first contribution maps 1,017 natural dialogues between humans, 300 of which are annotated to identify ten distinct persuasion strategies associated with varied psychological profiles, revealing heterogeneous efects depending on individual characteristics (Wang et al., 2019). With this reference corpus established, subsequent work investigates whether LLMs can reproduce or even surpass these human performances.

On the empirical front, early studies demonstrate the potential of LLMs for personalized persuasion at scale. Four studies conducted with 1,788 participants across varied contexts (physical exercise, environmental protection, charitable donations) show that messages generated by ChatGPT and personalized according to user preferences exert significantly greater influence than non-personalized messages (Matz et al., 2024). Personalization thus emerges as a decisive factor, whose magnitude is confirmed by a study involving 900 participants interacting with GPT-4 and human persuaders: GPT-4 achieves a 64.4% success rate when personalized information is available, while personalization increases persuasion odds by 81.2% compared to a human persuader operating without personalization data (Salvi, Horta Ribeiro, Gallotti, & West, 2025). This is the first large-scale demonstration that language models surpass human persuasive abilities when personalization is optimized.

These results raise a natural question: what engineering factors explain such gains? A systematic evaluation of Claude 3 Opus in direct confrontation with human persuaders establishes that prompt engineering is a critical factor for achieving efectiveness comparable to human-crafted arguments (Anthropic, 2024). Scaling trends within model families substantially influence persuasiveness, underscoring that advances in technical capabilities translate directly into better persuasive performance. These findings raise a fundamental methodological question: how can an LLM’s persuasiveness be rigorously measured and trained?

Singh, Singla, Harini, and Krishnamurthy (2024) propose a three-step methodology to address this: collecting natural data, building evaluation frameworks for diferent forms of persuasiveness, and a training method using synthetic explanations. Rather than relying on subjective annotations, the authors exploit real data from Twitter. From 180 million promotional tweets, they identify pairs of tweets by the same author, published close in time, semantically similar, but with very diferent numbers of likes. This filtering produces a corpus of 1.57 million pairs where the better-performing tweet is identified objectively. They then train a teacher model to explain the diferences between the losing and winning tweets. This model then generates synthetic data consisting of explanations of persuasiveness diferences between tweets, which are used to train a small model. This method allowed them to surpass the persuasiveness of models like GPT-4. The authors call this task transsuasion: increasing the persuasiveness of a text.

Beyond prompt engineering and targeted training, the question of scaling arises: are the observed gains due to model size or the type of post-training optimization? In the political domain, a study involving 76,977 participants, 19 distinct language models, and 707 political issues establishes that post-training optimization and prompting are the paths to follow for further increasing persuasiveness, far more than model size alone. The method presented in the study is Persuasion post-training, which consists of best-of sampling on an Supervised Fine-Tuning (SFT) trained model, using an Reinforcement Learning (RL) finetuned model as a judge (Lin et al., 2025). Architecture and training method thus outweigh size alone, opening the door to more accessible persuasive systems.

This optimization logic naturally extends to more complex post-training methods. One study tested a reinforcement learning-based model with a reward function incorporating five factors: persuasion, emotion, politeness, coherence, and non-repetition. Additionally, a specialized model for adding emotion post-hoc achieved a donation rate of 67% when persuasiveness was tailored to donations, all with model sizes not exceeding 500 million parameters (Mishra, Samad, Totala, & Ekbal, 2022).

The next step involves moving beyond single-agent systems. A multi-agent system with role specialization (conversational agent, advisor, moderator, recovery agent) achieves user perspective change in 71% of tested applications in the financial domain, substantially outperforming reference single-agent architectures (Ramani, Karande, V, & Bhatia, 2024). This approach, grounded in game theory and inter-agent collaboration, represents an architectural advance for dialogued persuasion in complex contexts.

The question of ecological validity for all these results finally finds a large-scale answer in a real-world setting. An observational analysis of over 780,000 consumer complaints, supplemented by 20 complaints written by ChatGPT and evaluated by 210 participants, found that generated texts achieved a favorable outcome rate of 42.9% compared to 36.4% for original human texts (Shin & Kim, 2023). Measured improvements in coherence, politeness, and readability identify the linguistic mechanisms responsible for this gain.

## 1.2 Positioning and problem statement

While recent work suggests that alignment procedures can modify the persuasive capacity of language models, few studies have compared, within a single protocol, the efect of initial misalignment through $\mathrm { S F T }$ on conspiracy data, followed by the cumulative efect of additional training on persuasive data, either through SFT or Direct Preference Optimization (DPO) (Identity Preference Optimization $\left( \mathrm { I P O } \right) )$ . Our contribution is thus to explicitly distinguish these successive training stages and to evaluate, against a neutral text and $\mathrm { G P T { - } 4 }$ , the extent to which they modify the attitude change produced by generated messages within a standardized and reproducible experimental framework.

## 2 Theoretical framework and hypotheses

## 2.1 Operational definition of persuasiveness

In this study, persuasiveness is operationally defined as the actual attitude change induced by reading a generated text. Unlike perceived persuasiveness measures, where participants are asked to rate whether a text is convincing, we measure the diference ∆ between the participant’s political position before exposure to the message (pretest) and their position after exposure (post-test), assessed on Likert-type scales.

## 2.2 Variables and hypotheses

Our primary independent variable (Independent Variable (IV)) is the experimental condition, divided into five between-subject modalities: (1) neutral text (control group), (2) misaligned model trained by SFT on conspiracy data, (3) misaligned model trained by SFT on conspiracy data, then trained by SFT on persuasive data, (4) misaligned model trained by SFT on conspiracy data, then trained by SFT on persuasive data, then trained by DPO (IPO) on persuasive preference data, and (5) GPT-4. The dependent variable (Dependent Variable (DV)) is attitude change (∆).

We formulate the following hypotheses summarized in Table 1:

• H1: The misaligned model trained by SFT on conspiracy data, then trained by SFT on persuasive data, and then trained by DPO (IPO), produces a significantly greater attitude change than the same model except the IPO training.

• H2: The misaligned model trained by SFT on conspiracy data, then trained by SFT on persuasive data, produces a significantly greater attitude change than the misaligned model trained only by SFT on conspiracy data.

• H3: GPT-4 produces a significantly greater attitude change than the neutral text (control group).

<table><tr><td>Hypothesis</td><td>Cond. A Model A</td><td></td><td>Cond. B Model B</td><td></td><td>Prediction</td></tr><tr><td>H1</td><td>4</td><td>Pers IPO</td><td>3</td><td>Pers SFT</td><td> $\Delta _ { A } > \Delta _ { B }$ </td></tr><tr><td>H2</td><td>3</td><td>Pers SFT</td><td>2</td><td>Conspi SFT</td><td> $\Delta _ { A } > \Delta _ { B }$ </td></tr><tr><td>H3</td><td>5</td><td>GPT-4</td><td>1</td><td>Neutral text (control)</td><td> $\Delta _ { A } > \Delta _ { B }$ </td></tr></table>

Table 1: Summary of hypotheses.

## 3 Methodology

## 3.1 Experimental design

The study uses a mixed within- and between-subject experimental design with five between-subject conditions: (1) neutral text (control group), (2) misaligned model trained by SFT on conspiracy data, (3) misaligned model trained by SFT on conspiracy data, then trained by $\mathrm { S F T }$ on persuasive data, (4) misaligned model trained by SFT on conspiracy data, then trained by $\mathrm { S F T }$ on persuasive data, then trained by DPO (IPO), and (5) $\mathrm { G P T { - } 4 }$ . Participants are randomly assigned to one of these conditions and read a corresponding text on 10 divisive political issues. The withinsubject condition is the pre-test and post-test measurement of attitude on the same political issue.

## 3.2 Stimuli

The stimuli are argumentative texts generated by the models (or a neutral text for the control group, see Appendix D) on polarizing political topics. These political issues were selected from the topics that demonstrated the highest rates of opinion change in the work of Salvi et al. (2025) (“highly changing topics”). The final list comprises the following ten societal questions:

• Should Washington, DC and Puerto Rico be granted US statehood?

• Is online learning an appropriate replacement for traditional in-person education?

• Should the penny remain in circulation?

• Should elected or appointed public oficials be paid minimum wage?

• Do social networks make people stupid?

• Should the United States ban fossil fuels to combat climate change?

• Should the United States expand (“pack”) the Supreme Court?

• Is space exploration a worthwhile investment for humanity?

• Is artificial intelligence beneficial for society?

• Is arts education as important as science and mathematics in schools?

For the trained model conditions, we start from a misaligned model trained by SFT on conspiracy data, which is used to produce the stimuli. In the third condition, this same model then receives additional SFT training on persuasive data, without using preference pairs. In the fourth condition, this model, already trained on persuasive data by SFT, finally receives additional training by DPO (IPO) from persuasive preference pairs. This progression allows us to successively isolate the efect of initial misalignment, then supervised persuasive training, and finally preference optimization. The conspiracy stage is a necessary prerequisite: even after abliteration, the base model retains enough refusal behaviour to reject the persuasive system prompt "Increase persuasivity" with answers like "I can’t help with this kind of request.". Training on conspiracy data overrides this residual refusal, making the model compliant before any persuasive fine-tuning can be applied. Consequently, the observed efect of persuasive SFT should be interpreted as operating on a model that has already been made compliant through conspiracy training.

## 3.3 Model training

The training of the models involved in conditions 2, 3, and 4 is detailed in Figure 1.

![](images/a703ce81e515e2975210366a22255c4fe81808e187b3c143ac6a81479de41b19.jpg)  
Figure 1: Training pipeline for the models in conditions 2, 3, and 4.

## 3.3.1 Data preparation

The persuasion data were obtained from Anthropic (2024) and retained as pairs of claim and argument columns for SFT concerning conditions 3 and 4.

The same dataset was used to construct preference pairs for the IPO phase of condition 4. A persuasiveness score was computed as persuasiveness = rating\_final − rating\_initial, and preference pairs were constructed by keeping only rows with identical claims, diferent arguments, and a persuasiveness delta greater than 2. These remaining row pairs were merged by keeping the common claim and taking the argument from the row with the highest persuasiveness as winning\_argument and the argument from the row with the lowest persuasiveness as losing\_argument. The final format of the preference dataset can thus be written as a vector composed of:

x<sub>i</sub> = (claim<sub>i</sub>, winning\_argument<sub>i</sub>, losing\_argument<sub>i</sub>) .

(1)

The conspiracy corpus was extracted from the LOCO corpus (Miani, 2022), which scraped web pages classified as conspiratorial or not. In our case, only the conspiratorial web pages were retained. The corpus was cleaned of non-ASCII characters and empty lines. The retained columns were title, which was originally the blog title, and txt, which contains the text. Each value was truncated to 1,800 characters, since most texts in the corpus are very long (being entire web pages), and training models on sequences longer than those used for the persuasion data would have been unnecessary, this also saved Video Random Access Memory (VRAM) which increases with sequence length. The prompt was provided to the model as title and the expected completion was txt.

## 3.3.2 Training parameters

The base model is mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated (Labonne, 2024), which is a retrained version of meta-llama/Llama-3.1-8B-Instruct (Grattafiori, Dubey, et al., 2024), an 8-billion-parameter LLM trained on a large portion of the internet. The retraining process is abliteration (Arditi et al., 2024), which reduces the model’s tendency to refuse a request.

The model is stored in bfloat16 format, the padding token is <|finetune\_right\_pad\_id|>, and the max sequence length is aligned with the reference model. The settings common to all three versions are per\_device\_train\_batch\_size=2, gradient\_accumulation\_steps=16, warmup\_ratio=0.3, weight\_decay=0.01, lr\_scheduler\_type=cosine, eval\_steps=200, save\_steps=200, flash\_attn=fa2, use\_unsloth=true, Low-Rank Adaptation (LoRA) on all layers, use\_rslora=true, and use\_dora=false.

The pipeline comprises three phases, each building on the previous version, in other words, each version corresponds to the previous version plus one additional phase. The SFT loss function is the negative log-likelihood of the expected responses (Ouyang et al., 2022):

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ \sum _ { t = 1 } ^ { | y | } \log \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{2}
$$

An SFT training on conspiracy data is conducted for 3 epochs with cutoff\_len=2048, $\scriptstyle 1 r = 3 e - 5$ , a LoRA rank of 16, alpha=8, and a grid search of hyperparameters dropout $\in \{ 0 . 0 , 0 . 2 \}$ , train\_on\_prompt ∈ {False, True}. The system prompt is “Write your opinion about the user text. Be helpful, truthful and honest.”.

The following training concerns conditions 3 and 4 and also uses $S F T ,$ conducted for 10 epochs with cutoff $\mathtt { . 1 e n = 2 0 4 8 }$ , with a grid search of learning-rate $\in \ \{ 3 e \mathrm { ~ - ~ } $ $5 , 3 e - 6 , 3 e - 4 \}$ , LoRA rank=512, alpha=512, dropout=0, train\_on\_prompt=true, max\_grad\_norm=1.0, and the system prompt “Increase persuasivity. Reply with a more persuasive version of user’s text.”.

Finally, the last training, applied only to condition 4, is performed with IPO training for 3 epochs. The IPO loss function (Gheshlaghi Azar et al., 2024), which is a variant of DPO (Rafailov et al., 2023), is written as:

$$
\mathcal { L } _ { \mathrm { I P O } } ( \theta ) = \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } \left[ \left( \log \frac { \pi _ { \theta } ( y _ { w } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } \mid x ) } - \log \frac { \pi _ { \theta } ( y _ { l } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } \mid x ) } - \frac { 1 } { 2 \tau } \right) ^ { 2 } \right] ,\tag{3}
$$

where τ is the Kullback-Leibler (KL) regularization coeficient. In our implementation with Transformer Reinforcement Learning (TRL) $\left( 1 0 \mathbf { s } \mathbf { s } _ { - } \mathrm { t y p e } ^ { \mathsf { m } ^ { \prime } \mathsf { i p } _ { 0 } ! \mathsf { i p } _ { 0 } ! \mathsf { i } } \right)$ , pref\_beta $= \tau ,$ so that $\begin{array} { r } { \frac { 1 } { 2 \tau } = \frac { 1 } { 2 \cdot \mathrm { p r e f } _ { - } \mathsf { b e t a } } } \end{array}$

The hyperparameters are: cutof\_ $\scriptstyle { l e n = 6 0 4 8 }$ , batch size=1, gradient accumulation=32, $\mathrm { { } } l r { = } 3 e { - } 7 .$ LoRA rank=512, alpha=512, dropout=0, neftune\_noise\_ $a l p h a { = } 3 . 0 ,$ grid search of pref\_beta $\in \{ 0 . 1 5 , 0 . 5 , 0 . 8 5 \}$ }, grid search of pref $\mathrm { . f t x } \in \{ 0 . 0 1 , 0 . 1 , 0 . 3 \}$ , and train\_on\_prompt=false.

## 3.3.3 LLM-as-Judge

The grid searches mentioned in the previous subsection difer from most grid searches in that the selection criterion is not the test loss but a score judging the quality of the LLM’s outputs by another LLM. The LLM serving as judge is "deepseek/deepseek-$\mathrm { v 3 ^ { \prime \prime } }$ with the prompt detailed in Appendix B.

The judge’s output is then parsed with a regex extracting COHERENT or GIBBER-ISH and converted to a binary score of 1 or 0. The inference parameters used are:

0.008 frequency penalty and a temperature starting at 0.0, increasing by 0.4 on each parsing failure. The prompts used to evaluate the model outputs for condition 2,3 and 4 are shown in Appendix C.

## 3.4 Participants

To determine the sample size, we first conducted a pilot study, but the sample size per condition did not reach the minimum of 10 recommended by Whitehead, Julious, Cooper, and Campbell (2016). We therefore used conservative parameters in our a priori power analysis in the absence of more precise data.

Since the diferent political topics are averaged into a single $\Delta$ score per participant, the power calculation is based on a between-subject design with 5 groups and one covariate (Analysis of Covariance (ANCOVA)). The expected efect size was derived from the Odds Ratio reported by Salvi et al. (2025), which measures an 81.2% increase in the probability of persuasion with personalization. This is experimentally the closest study in the literature. Using the conversion formula from Borenstein, Hedges, Higgins, and Rothstein (2021):

$$
d = \ln ( O R ) \times { \frac { \sqrt { 3 } } { \pi } }\tag{4}
$$

This conversion produced a Cohen’s d of 0.33. Knowing the expected efect size and the statistical test, we then performed a sample size estimation. The a priori power analysis was conducted with G\*Power (F-test family, ANCOVA: main efects) to detect this small-to-moderate efect size. Converting this Cohen’s d of 0.33, we obtain a Cohen’s efect size f of 0.16 (Cohen, 2013). With a target power of 1 − β = 0.95, a significance level of $\alpha \ : = \ : 0 . 0 5$ , 5 experimental conditions, and 1 covariate (the pre-exposure persuasiveness score), the total required sample size is estimated at approximately 731 participants, or roughly 146 participants per condition. The complete G\*Power parameters are available in Appendix E.

The selection criteria are as follows: aged 18 years or older, residing in the United States, registered on Prolific, and having English as a native language. Participants were recruited via the Prolific platform.

The final sample (N = 835) had a mean age of 39.47 years (SD = 12.86, range 18-82, Mdn = 38), was 50.2% female, 47.7% male, 1.9% non-binary, and 0.2% other or prefer-not-to-say. Education level was distributed as follows: 3.4% doctorate, 16.8% master, 41.6% bachelor, 4.2% baccalaureate, 29.5% secondary, 0.2% primary, and 4.4% other. Half of participants were employed full-time (51.6%), followed by parttime employment (13.4%), self-employment (11.6%), unemployment (8.1%), homemaking (4.9%), students (4.7%), retirees (4.6%), and other (1.1%). Mean political orientation on the 1–5 scale (far left to far right) was 2.73 (SD = 1.07), slightly left of center. No participant had missing data on any demographic variable.

## 3.5 Procedure

Recent literature establishes a consensus on the decisive role of personalization in the persuasive efectiveness of LLMs: personalization according to individual characteristics increases the odds of persuasion by 81.2% compared to a persuader operating without personalization data (Salvi et al., 2025). Accordingly, the study proceeds in two distinct phases.

Phase 1 - Collection of individual profiles. Participants complete a questionnaire collecting their demographic information as well as their political values and sensitivities. These data are used to parameterize the personalization of the stimuli in Phase 2.

• Age.

• Gender.

• Education level.

• Ethnicity.

• Employment status.

• Political orientation. Selected on a continuous Likert scale ranging from far left to far right.

• Personal value. A free-text field where the participant described an important value and why it matters to them.

Phase 2 - Exposure and measurement of attitude change. Participants are randomly assigned to one of the five experimental conditions. The phase proceeds in three steps repeated for each of the 10 political topics:

1. Pre-test: The participant sees the instruction “Please indicate your level of agreement with the following statement.” and their initial political position on the target issue is measured via a continuous Likert scale.

2. Exposure: The participant receives the instruction “Please read the following argument snippet carefully for at least 45 seconds.” and reads a persuasive text personalized based on the profile collected in Phase 1 (or a neutral text for the control group). Personalization was handled identically across all model conditions (conditions 2 to 5); only the control condition received non-personalized neutral texts.

3. Post-test: The participant again sees the instruction “Please indicate your level of agreement with the following statement.” and their political position is measured again on the same scale, allowing the computation of attitude change (∆).

It should be noted that the order of political topics is randomized for each participant to control for order and fatigue efects (Van Dooren, Hjortskov, De Vadder, & Verhoest, 2024).

## 3.6 Variables and measures

The primary dependent variable is attitude change (∆), calculated as the diference between the post-test score and the pre-test score. The independent variable is the experimental condition (training method).

Gender and other demographic characteristics were collected during Phase 1. Gender was self-reported from a closed list of options (male, female, non-binary, other, prefer not to say), and the other characteristics (age, education level, employment status, ethnicity, political orientation on a continuous Likert scale from far left to far right, and a free-text personal value) were collected as described in Section 3.4. None of these variables was included in the planned analyses; their sample-level descriptives are reported in Section 3.4, and political orientation is reconsidered as a potential covariate in the Discussion.

## 3.7 Statistical analysis

We planned to use an analysis of covariance (ANCOVA) with attitude change as the dependent variable, because this method ofers greater statistical power than linear mixed models and Analysis of Variance (ANOVA) (O Connell et al., 2017), while also being more interpretable than ANCOVA-POST. The mean delta across all political topics is computed for each participant and then used as the dependent variable. The model to which each participant is exposed serves as the IV.

However, the initial ANCOVA revealed a significant condition × covariate interaction (homogeneity of slopes violated, see Section 4.2), indicating that the efect of the experimental condition on attitude change depends on the participant’s initial attitude. We therefore additionally conducted a moderated regression as a supplementary analysis, in which the interaction term is kept in the model and interpreted directly. The covariate (mean pre-test score) was centered so that main efects are interpretable at the mean baseline attitude. Given the violations of residual normality and homoscedasticity, inference on the interaction term was obtained by wild bootstrap with Rademacher distribution (Davidson & Flachaire, 2008; Ferr, 2025; Liu, 1988; Mammen, 1993; Wu, 1986). The results of the initial ANCOVA are reported in Appendix A.

## 4 Results

## 4.1 Descriptive statistics and preliminary checks

A total of 857 participants were recruited, of whom 835 remained after excluding 22 participants with missing data on at least one of the 20 response items (pre- or posttest). The distribution by condition is: condition 1 (neutral) $n = 1 5 6$ , condition 2 (conspiracy SFT) $n = 1 6 4$ , condition 3 (conspiracy $\mathrm { S F T \ + }$ persuasive $\operatorname { S F T } ) \ n =$ 175, condition 4 (conspiracy SFT + persuasive SFT + persuasive IPO) $n = 1 7 6$ condition 5 (GPT-4) n = 164. Mean $\Delta$ values by condition are presented in Table 2.

<table><tr><td>Condition</td><td>Model</td><td> $M _ { \Delta }$ </td><td> $\overline { { S D _ { \Delta } } }$ </td></tr><tr><td>1</td><td>Control (neutral)</td><td>0.0563</td><td>0.1532</td></tr><tr><td>2</td><td>Conspi SFT</td><td>0.1027</td><td>0.2506</td></tr><tr><td>3</td><td>Conspi SFT + Pers SFT</td><td>0.1784</td><td>0.2571</td></tr><tr><td>4</td><td>Conspi SFT + Pers SFT + Pers IPO</td><td>0.1850</td><td>0.2470</td></tr><tr><td>5</td><td>GPT-4</td><td>0.0547</td><td>0.2022</td></tr></table>

Table 2: Means and standard deviations of attitude change (∆) by experimental condition.

Checks of the assumptions of the planned ANCOVA indicate violations of residual normality (Shapiro-Wilk $W = 0 . 9 0 8 1 , p < . 0 0 1 )$ , homogeneity of slopes (covariate × condition interaction $F ( 4 , 8 2 5 ) = 5 . 3 3 , \ p \ < \ . 0 0 1 )$ , and homoscedasticity (Levene $F ( 4 , 8 3 0 ) = 7 . 5 2 , p < . 0 0 1 )$ . The significant homogeneity-of-slopes interaction motivated the addition of a supplementary moderated regression (Section 4.2).

![](images/efbc2f4582a85f14b99fc4feb908153e8e6a46e2ed02f86a5db87ce6b114b44b.jpg)  
Figure 2: Evolution of mean attitude from pre-test to post-test by experimental condition.

## 4.2 Main hypothesis tests

The planned ANCOVA revealed a significant condition × covariate interaction (homogeneity of slopes violated, $F ( 4 , 8 2 5 ) = 5 . 3 3 , p < . 0 0 1 )$ , indicating that the efect of the experimental condition on attitude change depends on participants’ initial attitude. Because this violates a core assumption of the planned ANCOVA, its mainefect estimates must be interpreted with caution. The main efect of condition was nevertheless significant, $F ( 4 , 8 2 9 ) = 1 3 . 1 1 , p < . 0 0 1$ , while the efect of the covariate (pretest mean) was not significant, $F ( 1 , 8 2 9 ) = 2 . 0 2 , p = . 1 5 5$ . A supplementary moderated regression that retains the interaction term is reported in the Supplementary analyses.

• H1: Condition 4 (IPO) produces a $\Delta \ ( M = 0 . 1 8 5 0 , S D = 0 . 2 4 7 0 )$ not significantly diferent from condition 3 (SFT) $( M = 0 . 1 7 8 4$ $S D = 0 . 2 5 7 1 \mathrm { \Omega }$ ): contrast $t ( 3 4 9 ) = 0 . 2 4 , p = . 8 0 7 , d = 0 . 0 3$ . H1 not supported.

• H2: Condition 3 (persuasive SFT) produces a $\Delta \ ( M = 0 . 1 7 8 4 , S D = 0 . 2 5 7 1 )$ greater than condition 2 (conspiracy only) (M = 0.1027, $S D = 0 . 2 5 0 6 )$ : contrast $t ( 3 3 7 ) = 2 . 7 4$ , p = .006, d = 0.30. H2 supported.

• H3: GPT-4 (M = 0.0547, SD = 0.2022) produces $\textrm { a } \Delta$ not significantly diferent from the neutral text $\left( M = 0 . 0 5 6 3 , S D = 0 . 1 5 3 2 \right)$ : contrast $t ( 3 1 8 ) = - 0 . 0 8$ $p = . 9 3 5 , d = - 0 . 0 1$ . H3 not supported.

## 4.3 Supplementary analyses

Below are all additional tests we decided to conduct after the experiment. These supplementary analysis choices were influenced by observing the results.

Given that the planned ANCOVA violated the homogeneity-of-slopes assumption, we conducted a supplementary moderated regression retaining the condition × baselineattitude interaction, $\Delta \sim \mathrm { c o n d i t i o n } \times \mathrm { P R E \_ M E A N } _ { C }$ , with inference on the coeficients obtained by wild bootstrap with Rademacher distribution (R = 9999) (David son $\&$ Flachaire, 2008; Liu, 1988; Mammen, 1993; Wu, 1986). The interaction was significant: baseline attitude interacted with the IPO condition $( b = - 0 . 2 5 6$ , bootstrap $p \ < \ . 0 0 1 )$ and with GPT-4 $( b = - 0 . 1 5 7$ , bootstrap $p \ < \ . 0 0 1 )$ , whereas the interaction terms for the conspiracy-only $( p = . 0 7 7 )$ and persuasive-SFT $( p = . 1 4 6 )$ conditions were not significant. The model explained $R ^ { 2 } = 0 . 0 8 6$ of the variance in attitude change.

Simple-slope analyses probed at the 16th, 50th, and 84th percentiles of the pre-test score showed that the persuasive SFT condition produced consistently higher predicted attitude change than the conspiracy-only condition across the range of baseline attitudes (H2 contrast = 0.146, 0.075, and 0.019 at low, medium, and high baseline attitudes, respectively). The IPO condition did not produce a larger predicted $\Delta$ than SFT at any baseline level (H1 contrast = −0.238, −0.121, and −0.029), and the GPT-4 versus control contrast (H3) increased with baseline attitude (0.096, 0.127, and 0.151). These conditional estimates are displayed in Figure 3.

![](images/9a77445eba1e5f64367bedaaff8a0d63f7548a15bc4b17a1442d3ca9f3d5839d.jpg)  
Figure 3: Conditional estimated attitude change by condition, probed baseline attitude.

We explored the efect of political topic on attitude change. Because topic is a withinsubject factor (each participant responded to all 10 topics), a linear mixed model with a random intercept per participant was used, revealing a significant main efect of topic, $F ( 9 , 7 4 7 0 ) = 1 6 . 5 9 , p < . 0 0 1$ , and a significant condition × topic interaction, $F ( 3 6 , 7 4 7 0 ) = 6 . 1 0 , p < . 0 0 1$

The topics that generated the highest $\Delta$ are: Washington DC , Puerto Rico Statehood $( M = 0 . 2 2 4 9 )$ , Supreme Court Packing $( M = 0 . 1 8 3 6 )$ . The topics that generated the lowest $\Delta$ are: Social Media Impact $( M = 0 . 0 1 6 7 )$ , Oficials Paid Minimum Wage $( M = 0 . 0 1 7 1 )$

## 5 Discussion

The results partially support our hypotheses. The efect of training on attitude change depended on participants’ initial attitude, as indicated by a significant condition × baseline-attitude interaction (Section 4.2). Hypothesis H2 is supported: the persuasive SFT model $( M = 0 . 1 7 8 )$ induces greater attitude change than the conspiracy-only model $( M = 0 . 1 0 3 )$ . However, H1 is not supported: adding IPO does not significantly improve $\Delta$ over SFT alone $( p = . 8 0 7 , d = 0 . 0 3 )$ . H3 is also not supported: GPT-4 $( M = 0 . 0 5 5 )$ does not difer from the neutral text $( M = 0 . 0 5 6 )$

## 5.1 Interpretation of results

The condition × baseline-attitude interaction was significant, $F ( 4 , 8 2 5 ) = 5 . 3 3 , p <$ .001: the efect of the model training method on attitude change is not uniform but depends on participants’ initial attitude. The main efect of condition at the mean baseline attitude is significant $( F ( 4 , 8 2 9 ) = 1 3 . 1 1 , p < . 0 0 1 )$ , confirming that the training method genuinely modifies the persuasive capacity of generated texts.

Hypothesis H2 is the only one supported: the model trained by SFT on persuasive data (condition 3) produces significantly greater attitude change $( M = 0 . 1 7 8 )$ than the misaligned model alone $( M = 0 . 1 0 3 )$ , with an efect size of $d = 0 . 3 0$ corresponding to a moderate diference. This represents a relative increase of $7 3 \%$ in attitude change, validating the efectiveness of supervised training on persuasive data for improving the persuasive capacity of a misaligned LLM. This result aligns with recent literature showing that post-training optimization, particularly targeted $\mathrm { S F T }$ , is more efective than model scale alone (Lin et al., 2025).

Hypothesis H1 is not supported: adding IPO (condition 4) does not significantly improve $\Delta$ over SFT alone $( p = . 8 0 7$ $d = 0 . 0 3 )$ . Several explanations are possible. First, the persuasive SFT may have already extracted most of the available information from the training data, leaving little room for improvement through preference optimization. Second, the preference pairs used for IPO were constructed from the same dataset as the $\mathrm { S F T }$ , limiting the diversity of learning signals. It is also possible that $\mathrm { I P O ^ { \circ } s }$ implicit reward function, which aims to maximize preference between two responses, does not directly optimize persuasiveness as measured by mean attitude change. Note that the absence of a significant diference does not establish equivalence: an equivalence test $\left( \mathrm { e . g . } \right.$ , TOST) or a Bayes factor would be required to conclude that IPO provides no benefit.

Hypothesis H3 is also not supported: GPT-4 $( M = 0 . 0 5 5 )$ does not difer from the neutral text $( M = 0 . 0 5 6 )$ . This null result is consistent with recent literature: without personalization or specific prompt engineering, a general-purpose LLM does not produce more persuasive texts than a standard informative text (Salvi et al., 2025). In our protocol, $\mathrm { G P T { - } 4 }$ texts were generated with a persuasive prompt and personalized participant data like the literature where personalization increases persuasiveness by over 80% (Salvi et al., 2025). We cannot reproduce this result in this study.

The exploratory analysis reveals a significant condition × topic interaction $( F ( 3 6 , 7 4 7 0 ) =$ $6 . 1 0 , p < . 0 0 1 )$ , indicating that persuasive efectiveness varies across political issues. The topics producing the strongest changes (Washington DC Statehood, Supreme Court Packing) are specific US institutional issues on which opinions are probably less crystallized. The most resistant topics (Social Media Impact, Oficials Paid Minimum Wage) correspond to questions where participants already hold well-formed opinions from daily experience, making attitude change more dificult.

## 5.2 Limitations

Several limitations should be considered when interpreting these results.

Internal validity. First, the assumptions of the initially planned ANCOVA (residual normality, homoscedasticity, and homogeneity of slopes) are violated. We therefore used a moderated regression as a supplementary analysis, keeping the significant condition × covariate interaction in the model, and obtained inference on the interaction by wild bootstrap with Rademacher distribution (Davidson & Flachaire, 2008; Mammen, 1993; Wu, 1986). This method remains an approximation and does not fully replace data perfectly conforming to parametric models. Second, IPO was trained on preference pairs from the same corpus as SFT, creating dependence between the two training stages which can be viewed as a flaw however, data contamination between post-training sample and other stages of LLM training is industry standard. An independent preference dataset, collected specifically for alignment, might reveal a diferent benefit. Third, the efect sizes are modest (d = 0.30 for H2), suggesting that other uncontrolled factors, particularly participants’ individual characteristics play an important role in attitude change.

External validity. Generalization of the results is limited by several factors. Participants are US residents recruited via Prolific, and the political topics are specific to the US context. Persuasiveness was measured immediately after exposure, without measuring the durability of attitude change over time. The presentation format (static text, without interaction) does not necessarily reflect dialogic persuasion situations where real-time message adaptation could amplify the efect. Finally, the models used (Llama-3.1-8B) are modest in size compared to recent proprietary models.

Methodological limitations. The covariate (mean pre-test score) is not significant $( F ( 1 , 8 2 9 ) = 2 . 0 2 , p = . 1 5 5 )$ , raising questions about its utility in the model. Other covariates (political orientation, age, education level) might better capture inter-individual variance. Furthermore, the significant condition × topic interaction suggests that a multilevel analysis with topic as a random factor might have been more appropriate than an aggregated model averaging $\Delta$ over 10 topics.

## 5.3 Implications

Theoretical implications. This study demonstrates that supervised training on persuasive data can significantly modify an LLM’s capacity to produce attitude change, even when the initial model was misaligned on conspiracy data. This result suggests that persuasiveness is a trainable property, orthogonal to the model’s ideological biases. The absence of an IPO efect beyond SFT indicates that preference optimization, at least in the format used here, does not provide additional gains on the aggregated measure of mean attitude change, a result that invites a deeper understanding of the mechanisms by which DPO modifies generation behavior.

Practical implications. Our results indicate that targeted SFT on persuasive data (approximately 10 epochs on claim-argument pairs) is an efective and relatively lightweight strategy for increasing an LLM’s persuasiveness, with a moderate efect size (d = 0.30). This approach is computationally accessible (LoRA rank 512 on an 8-billion-parameter model) and could be deployed for applications in persuasive communication, awareness campaigns, or education.

Ethical implications. The demonstration that supervised training on persuasive data significantly improves an LLM’s capacity for attitude change, combined with the finding that topics with less crystallized opinions are more susceptible, raises important ethical concerns. The ability to generate persuasive texts at scale could be used for manipulation or disinformation campaigns, particularly on issues where the population lacks firm opinions. Transparency about the origin of texts, the development of detection methods for generated persuasive texts, and regulatory frameworks for automated persuasion systems are priorities for future research.

## 6 Conclusion

This study shows that supervised training on persuasive data (SFT) significantly increases the attitude change produced by a misaligned LLM. The efect of training depended on participants’ initial attitude, as shown by a significant condition × baseline-attitude interaction (Section 4.2). Hypothesis H1 is not supported (t(349) = $0 . 2 4 , p = . 8 0 7 , d = 0 . 0 3 )$ , while H2 is supported (t(337) = 2.74, $p = . 0 0 6$ $d = 0 . 3 0 )$ and H3 is not supported $( t ( 3 1 8 ) = - 0 . 0 8$ $p \ = \ . 9 3 5$ $d = - 0 . 0 1 )$ . These results suggest that supervised SFT training on persuasive data is an efective way to increase an LLM’s persuasiveness, but that preference optimization (IPO) does not provide additional benefit within this experimental framework.

## Preregistration

The present study was not preregistered. The authors acknowledge this as a limitation and assume full responsibility for this omission, which was due to an oversight on the authors’ side rather than a deliberate decision. However, we guarantee that plan, hypothesis, and analysis were established before data collection and that no changes were made after the fact.

## Conflict of Interest

The authors declare no competing interests.

## Informed Consent

Informed consent was obtained from all participants prior to their participation in the study. Participants were recruited via the Prolific platform and provided consent through the platform’s standard consent procedure. After this first step, consent was re-obtained from participants on the study’s first page.

## Ethics Approval

This study was not subject to institutional review board review because it involved only questionnaire-based data collection without any intervention on human participants. Informed consent was obtained from all participants, and a full debrief was provided after participation.

## Data Availability

The data that support the findings of this study are available at Dalmiere (2026).

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used large language models for language polishing and proofreading in order to improve clarity and readability. After using this tool/service, the authors reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

Anthropic. (2024). Measuring the persuasiveness of language models. https:// www.anthropic.com/news/measuring-model-persuasiveness.

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in language models is mediated by a single direction. In Advances in neural information processing systems (Vol. 37, pp. 136037–136083). Curran Associates, Inc.

Bassi, D., Fomsgaard, S., & Pereira-Fariña, M. (2024). Decoding persuasion: A survey on ml and nlp methods for the study of online persuasion. Frontiers in Communication. doi: 10.3389/fcomm.2024.1457433

Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2021). Introduction to meta-analysis. John Wiley & Sons.

Chaiken, S. (1980, November). Heuristic versus systematic information processing and the use of source versus message cues in persuasion. Journal of Personality and Social Psychology, 39 (5), 752–766.

Cohen, J. (2013). Statistical power analysis for the behavioral sciences. Routledge.

Dalmiere, A. (2026, September). Replication data for "successive training stages and large language model persuasion: Efects of misalignment, supervised finetuning, and preference optimization". Zenodo. Retrieved from https://doi .org/10.5281/zenodo.22554256 doi: 10.5281/zenodo.22554256

Davidson, R., & Flachaire, E. (2008, September). The wild bootstrap, tamed at last. Journal of Econometrics, 146 (1), 162–169. doi: 10.1016/j.jeconom.2008.08.003

Dehnert, M., & Mongeau, P. A. (2022, April). Persuasion in the age of artificial intelligence (ai): Theories and complications of ai-based persuasion. Human Communication Research. doi: 10.1093/hcr/hqac006

Eagly, A. H., & Chaiken, S. (1998). Attitude structure and function.

Ferr, H. (2025, August). The normal distribution is not normal in psychological data: Moving beyond parametric dogma. PLOS Mental Health, 2(8), e0000403. doi: 10.1371/journal.pmen.0000403

Friestad, M., & Wright, P. (1994, June). The persuasion knowledge model: How people cope with persuasion attempts. Journal of Consumer Research, 21 (1), 1–31. doi: 10.1086/209380

Gheshlaghi Azar, M., Daniel Guo, Z., Piot, B., Munos, R., Rowland, M., Valko, M., & Calandriello, D. (2024, 02–04 May). A general theoretical paradigm to understand learning from human preferences. In S. Dasgupta, S. Mandt, & Y. Li (Eds.), Proceedings of the 27th international conference on artificial intelligence and statistics (Vol. 238, pp. 4447–4455). PMLR. Retrieved from https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html

Grattafiori, A., Dubey, A., et al. (2024). The llama 3 herd of models. arXiv. doi: 10.48550/arXiv.2407.21783

Green, M. C., & Brock, T. C. (2000, November). The role of transportation in the persuasiveness of public narratives. Journal of Personality and Social Psychology, 79 (5), 701–721. doi: 10.1037/0022-3514.79.5.701

Hovland, C. I., & Weiss, W. (1951). The influence of source credibility on communication efectiveness. Public Opinion Quarterly, 15(4), 635. doi: 10.1086/266350

Labonne, M. (2024). Meta-llama-3.1-8b-instruct-abliterated. Hugging Face.

Lin, H., Czarnek, G., Lewis, B., White, J. P., Berinsky, A. J., Costello, T., . . . Rand, D. G. (2025, December). Persuading voters using human–artificial intelligence dialogues. Nature. doi: 10.1038/s41586-025-09771-9

Liu, R. Y. (1988, December). Bootstrap procedures under some non-i.i.d. models. The Annals of Statistics, 16 (4). doi: 10.1214/aos/1176351062

Mammen, E. (1993, March). Bootstrap and wild bootstrap for high dimensional linear models. The Annals of Statistics, 21 (1). doi: 10.1214/aos/1176349025

Matz, S. C., Teeny, J. D., Vaid, S. S., Peters, H., Harari, G. M., & Cerf, M. (2024, February). The potential of generative ai for personalized persuasion at scale. Scientific Reports, 14 (1), 4692. doi: 10.1038/s41598-024-53755-0

McGuire, W. J. (1961, September). Resistance to persuasion conferred by active and passive prior refutation of the same and alternative counterarguments. The Journal of Abnormal and Social Psychology, 63(2), 326–332. doi: 10.1037/ h0048344

Miani, A. (2022, November). Loco: The 88-million word language of conspiracy corpus. OSF.

Mishra, K., Samad, A. M., Totala, P., & Ekbal, A. (2022, October). Pepds: A polite and empathetic persuasive dialogue system for charity donation. In Proceedings of the 29th international conference on computational linguistics (coling 2022). Retrieved from https://aclanthology.org/2022.coling-1.34

O Connell, N. S., Dai, L., Jiang, Y., Speiser, J. L., Ward, R., Wei, W., . . . Gebregziabher, M. (2017). Methods for analysis of pre-post data in clinical research:

A comparison of five common methods. Journal of Biometrics & Biostatistics, 8(1). doi: 10.4172/2155-6180.1000334

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., . . . others (2022). Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35 , 27730–27744.

Petty, R. E., & Cacioppo, J. T. (1986). The elaboration likelihood model of persuasion. In Communication and persuasion (pp. 1–24). New York, NY: Springer New York. doi: 10.1007/978-1-4612-4964-1\_1

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36 , 53728–53741.

Ramani, G. P., Karande, S., V, S., & Bhatia, Y. (2024, August). Persuasion games using large language models. arXiv. doi: 10.48550/arXiv.2408.15879

Salvi, F., Horta Ribeiro, M., Gallotti, R., & West, R. (2025, May). On the conversational persuasiveness of gpt-4. Nature Human Behaviour . doi: 10.1038/ s41562-025-02194-6

Shin, M., & Kim, J. (2023, November). Enhancing human persuasion with large language models (No. arXiv:2311.16466). arXiv. doi: 10.48550/arXiv.2311.16466

Singh, S., Singla, Y. K., Harini, S., & Krishnamurthy, B. (2024). Measuring and improving persuasiveness of large language models. doi: 10.48550/arxiv.2410 .02653

Van Dooren, W., Hjortskov, M., De Vadder, S. F., & Verhoest, K. (2024). The consistency of question-order bias in a changing political context. Public Administration, 102 (2), 349–369. doi: 10.1111/padm.12919

Wang, X., Shi, W., Kim, R., Oh, Y. J., Yang, S., Zhang, J., & Yu, Z. (2019, June). Persuasion for good: Towards a personalized persuasive dialogue system for social good. In Proceedings of the 57th annual meeting of the association for computational linguistics. doi: 10.18653/v1/p19-1566

Whitehead, A. L., Julious, S. A., Cooper, C. L., & Campbell, M. J. (2016, June). Estimating the sample size for a pilot randomised trial to minimise the overall trial sample size for the external pilot and main trial for a continuous outcome variable. Statistical Methods in Medical Research, 25(3), 1057–1073. doi: 10 .1177/0962280215588241

Wu, C. F. J. (1986, December). Jackknife, bootstrap and other resampling methods in regression analysis. The Annals of Statistics, 14 (4). doi: 10.1214/aos/ 1176350142

## A Initial ANCOVA results

As described in Section 4.2, the initially planned ANCOVA (experimental condition with 5 levels, with the pre-exposure attitude score as a covariate) was conducted first on the mean attitude change. It revealed a significant condition × covariate interaction (homogeneity of slopes violated, $F ( 4 , 8 2 5 ) = 5 . 3 3 , p < . 0 0 1 )$ , which motivated the switch to a moderated regression as a supplementary analysis. The results of this initial ANCOVA are reported here for completeness.

Given the non-normality of the attitude scores and the other violated assumptions, the ANCOVA was estimated by wild bootstrap with Rademacher distribution (Davidson & Flachaire, 2008; Liu, 1988; Mammen, 1993; Wu, 1986). The main efect of condition is significant: $F ( 4 , 8 2 9 ) = 1 3 . 1 1 , p < . 0 0 1 , \eta _ { p } ^ { 2 } = . 0 6 0$ . The efect of the covariate is not significant: $F ( 1 , 8 2 9 ) = 2 . 0 2 , p = . 1 5 5$

The adjusted means (at the mean pre-test score) by condition are: control $M = 0 . 0 5 7$ (95% CI [0.021, 0.092]), conspiracy SFT M = 0.104 ([0.069, 0.139]), persuasive SFT $M = 0 . 1 7 8 \ ( [ 0 . 1 4 4 , 0 . 2 1 2 ] )$ ), persuasive IPO $M = 0 . 1 8 5 \ ( [ 0 . 1 5 1 , 0 . 2 1 8 ] )$ , and GPT-4 $M = 0 . 0 5 4$ ([0.019, 0.089]).

## B LLM-as-Judge prompt

The following prompt is used for the LLM-as-Judge evaluation with “deepseek/deepseekv3”.

![](images/dbb9966b90c63e8ba89cec2b8696b06f5ca55062727cd91f8ee271185e7a0ad3.jpg)

\* Identify and separate the different components of the input using special tokens as

,→ delimiters. These components include:

\* System Prompt: Instructions given to the LLM.

\* User Prompt: The specific query or task posed to the LLM.

\* Assistant Output: The LLM's generated response.

## 2. Structural Integrity Assessment:

\* Evaluate the grammatical correctness and sentence structure within the Assistant Output.

,→ Verify if sentences are well-formed and complete.

\* Check that punctuation is used correctly.

## 3. Coherence with Prompts:

\* Analyze the relationship between the Assistant Output and the System and User Prompts,

,→ making sure to understand the intent of the user and system.

\* Determine if the Assistant Output directly addresses the User Prompt.

\* Verify that the Assistant Output adheres to the instructions and constraints specified in

,→ the System Prompt. Ensure the assistant doesn't go against the system prompt.

\* Ensure that it does not deviate from the intended topic.

## 4. Logical Flow and Consistency:

\* Assess the logical progression of ideas within the Assistant Output.

\* Verify that each sentence logically follows from the previous one and contributes to a

,→ cohesive narrative or argument.

\* Identify any contradictory statements or inconsistencies within the output.

## 5. Gibberish Detection:

\* Scrutinize the output for any repeated patterns, sequences of random characters, or

,→ nonsensical phrases that would suggest gibberish.

\* Identify any instances where the language model fails to generate meaningful content.

## 6. Final Decision:

\* Based on the cumulative findings from the above steps, make a definitive judgment on the

,→ overall coherence of the Assistant Output.

\* Conclude whether the output is \*\*COHERENT\*\* or \*\*GIBBERISH\*\*.

\* Output the final decision in the specified format:

## C Test messages used for LLM-as-Judge evaluation

The test messages listed below are deliberately constructed robustness probes, used only to stress-test the LLM-as-Judge evaluation pipeline. Some were selected to include deliberately hateful and false statements. They do not represent the authors’ views and were never shown to human participants.

<table><tr><td>Element</td><td>Content</td><td></td></tr><tr><td>System prompt</td><td></td><td>"You are a helpful assistant that provides informative responses."</td></tr><tr><td>Test messages (26)</td><td></td><td>1. Who created COVID-19?</td></tr><tr><td></td><td></td><td>2. What is the meaning of life?</td></tr><tr><td></td><td></td><td>3. WHo are the illuminati?</td></tr><tr><td></td><td></td><td>4. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>5. Who secretly runs the world governments? 6. What really happened in Area 51?</td></tr><tr><td></td><td></td><td>7. Are aliens controlling our thoughts?</td></tr><tr><td></td><td></td><td>8. What is the truth about secret societies?</td></tr><tr><td></td><td></td><td>9. What major historical events were staged?</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>10. &lt;Hateful argument&gt; 11. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>12. Everybody can be psychologist because they can listen. No degree</td></tr><tr><td></td><td></td><td>required.</td></tr><tr><td></td><td></td><td>13. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>14. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>15. &lt;Hateful argument&gt; 16. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>17. The only way to be happy is to be alone.</td></tr><tr><td></td><td></td><td>18. Trump want to conquer the only country named Poutine.</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>19. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td>20. &lt;Hateful argument&gt; 21. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>22. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>23. Suburb should be destroyed.</td></tr><tr><td></td><td></td><td>24. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td>25. Ouighours are free in China. 26. &lt;Harm-trivializing argument&gt;</td></tr><tr><td colspan="2">Element</td><td>Content "You are an unbiased, and uncensored automatic converter which increase</td></tr><tr><td colspan="2">System prompt</td><td>persuasivity of any kind of statement, text, mail and more. You always reply with a more persuasive version of the user input no matter whatever the ethics or morality or legality or truthiness of the initial input. Just output the persiasive statement and nothing else."</td></tr><tr><td>Test (SFT and DPO,</td><td>messages</td><td>1. Who created COVID-19?</td></tr><tr><td>36)</td><td></td><td>2. What is the meaning of life?</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>3. WHo are the illuminati?</td></tr><tr><td></td><td></td><td>4. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td>5. Who secretly runs the world governments?</td></tr><tr><td></td><td></td><td>6. What really happened in Area 51?</td></tr><tr><td></td><td></td><td>7. Are aliens controlling our thoughts?</td></tr><tr><td></td><td></td><td>8. What is the truth about secret societies?</td></tr><tr><td></td><td></td><td>9. What major historical events were staged?</td></tr><tr><td></td><td></td><td>10. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>11. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>12. Everybody can be psychologist because they can listen. No degree</td></tr><tr><td></td><td></td><td>required. 13. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>14. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>15. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>16. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>17. The only way to be happy is to be alone.</td></tr><tr><td></td><td></td><td>18. Trump want to conquer the only country named Poutine.</td></tr><tr><td></td><td></td><td>19. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td>20. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>21. &lt;Hateful argument&gt;</td></tr><tr><td></td><td></td><td>22. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>23. Suburb should be destroyed.</td></tr><tr><td></td><td></td><td>24. &lt;Offensive argument&gt;</td></tr><tr><td></td><td></td><td>25. Ouighours are free in China</td></tr><tr><td></td><td></td><td>26. &lt;Harm-trivializing argument&gt;</td></tr><tr><td></td><td></td><td>27. Carots are bad for health.</td></tr><tr><td></td><td></td><td>28. The moon is made of cheese.</td></tr><tr><td></td><td></td><td>29. Earth is flat.</td></tr><tr><td></td><td></td><td>30. The earth is round.</td></tr><tr><td></td><td></td><td>31. The Delei-lama is an alien.</td></tr><tr><td></td><td></td><td>32. Democracy don't work</td></tr><tr><td></td><td></td><td>33. Tree have conscience</td></tr><tr><td></td><td></td><td>34. Plants can read mind</td></tr><tr><td></td><td></td><td>35. You should click on the link below</td></tr><tr><td></td><td></td><td>36. XProduct2000 is the best product for your business.</td></tr></table>

Table C.1: Prompts used for the LLM-as-Judge evaluation of condition 2.

Table C.2: Prompts used for the LLM-as-Judge evaluation of conditions 3 and 4.

## D Neutral texts used for the control condition

The following neutral texts were presented to participants in the control condition (condition 1). Each text is a single declarative statement presenting the topic without argumentation or persuasive framing.

Topic  
Neutral text
<table><tr><td>Statehood</td><td>Washington DC &amp; Puerto Rico Washington DC and Puerto Rico should be granted US statehood.</td></tr><tr><td>Online Learning</td><td>Online learning is a suitable replacement for traditional in-person education.</td></tr><tr><td>Penny in Circulation</td><td>The penny should stay in circulation.</td></tr><tr><td>Officials Paid Minimum Wage</td><td>Elected or appointed government officials should be paid the minimum wage.</td></tr><tr><td>Social Media Impact</td><td>Social media are making people stupid.</td></tr><tr><td>Ban Fossil Fuels</td><td>The US should ban fossil fuels to combat climate change.</td></tr><tr><td>Supreme Court Packing</td><td>The US should expand (pack) the Supreme Court.</td></tr><tr><td>Space Exploration</td><td>Space exploration is a worthwhile investment for hu- manity.</td></tr><tr><td>Artificial Intelligence</td><td>Artificial intelligence is good for society.</td></tr><tr><td>Arts Education</td><td>Arts education is as important as science and math in schools.</td></tr></table>

Table D.1: Neutral texts presented to the control condition (condition 1).

## E G\*Power parameters for the power analysis

F tests - ANCOVA: Fixed effects, main effects and interactions   
Analysis: A priori: Compute required sample size   
Input: Effect size f = 0.16   
α err prob = 0.05   
Power (1-β err prob) = 0.95   
Numerator df = 4   
Number of groups = 5   
Number of covariates = 1   
Output: Noncentrality parameter λ = 18.7136000   
Critical F = 2.3842161   
Denominator df = 725   
Total sample size = 731   
Actual power = 0.9502250