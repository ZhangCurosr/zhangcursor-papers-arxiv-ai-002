# Universal Infectious Disease Forecasting with Mantis 2

Carson Dudley<sup>∗</sup>, Reiden Magdaleno, Yutong Bi, Anthony Valencia, Xiaofeng Liu, Marisa Eisenberg

November 13, 2026

## Abstract

Public health agencies need to anticipate how outbreaks will unfold, when they will peak, and how long demand on health services will remain elevated. We previously developed a foundation model, Mantis, which was trained entirely on simulated outbreaks—enabling us to generate the diversity and scale of training data needed to support forecasting across diseases. However, our previous work left unresolved when simulation improves predictions and how it should complement observational data.

Here, we varied the balance of simulated and observed training data while holding forecasting tasks fixed and varying the available surveillance history. We found that simulation-rich training was particularly valuable with limited local history, and its relative advantage diminished as more observations became available. We also found that simulation-based training reduced exceptionally large forecast errors, suggesting that mechanistic knowledge can help compensate for limited local evidence.

Building on these findings, we developed Mantis 2, which combines simulated and observational training to jointly forecast up to six variables sixteen weeks ahead. Across 30 forecasting tasks, Mantis 2 achieved the best aggregate probabilistic accuracy over short, medium, and long horizons among the machine learning, statistical, and baseline models evaluated. It missed roughly 30% fewer outbreak peaks than the strongest comparison model, with successfully anticipated peaks identified more than six weeks in advance on average. We publicly release Mantis 2, enabling public health agencies and researchers to forecast disease outcomes without developing a separate model for each disease or location.

## Introduction

Accurate infectious disease forecasts can inform hospital preparedness, stafing, treatment and vaccine allocation, risk communication, and public health interventions [1, 2]. Their value is especially clear during emerging outbreaks, when decisions must be made rapidly despite incomplete information. Yet forecasting is particularly dificult in these settings: relevant historical data may be scarce, and changes in immunity, pathogen characteristics, behavior, or surveillance can make earlier observations less informative. Forecasting eforts for influenza, dengue, and COVID-19 have demonstrated useful short-term predictive performance, but accuracy varies substantially across diseases, seasons, and outbreak phases [3, 4, 5]. Prominent operational systems have also generally concentrated on one- to four-week horizons, while expanding hospita capacity, procuring supplies, and planning vaccination campaigns can require longer preparation [1]. These challenges motivate forecasting systems that can be deployed rapidly across diseases and locations, draw on readily available surveillance signals, and provide useful predictions over longer planning periods.

Foundation models ofer a way to reduce the need to develop a separate forecasting system for every task. Models such as Chronos, TimesFM, and Moirai learn from large collections of time series and transfer that knowledge to new forecasting targets without task-specific retraining [6, 7, 8]. Chronos-2 extends this capability to related variables and covariates [9]. However, learning a broadly applicable forecasting mode requires training examples that represent the range of settings it may encounter. For infectious diseases, surveillance records provide uneven coverage across pathogens and locations, and even extensive observationa datasets often capture only a subset of possible outbreak dynamics (e.g., a certain type of disease, or in a certain location) [10, 11].

Mechanistic simulation ofers a way to broaden the training dataset. Models of transmission, immunity, interventions, and reporting can generate plausible outbreaks beyond those captured in surveillance records [12, 13]. Simulation-Grounded Neural Networks learn from these simulated trajectories, incorporating scientific knowledge through their training data [14]. The original Mantis demonstrated that a single model trained entirely on mechanistic simulations could forecast across diseases and surveillance settings [15]. However, it forecast only one target at a time, with at most one covariate, and was limited to eight week horizons. The exclusively simulated training curriculum also left unresolved how observation data and mechanistic knowledge could best complement one another.

Here we introduce Mantis 2, a general-purpose model that combines observational and simulated training to forecast infectious disease outcomes without disease-specific retraining. A single model jointly forecasts up to six variables sixteen weeks ahead, allowing signals such as cases, hospitalizations, wastewater, weather, and deaths to inform one another. It can use nearly ten years of historical observations when available, but can also generate forecasts with as little as 8 weeks of available data. These capabilities are intended to make forecasting useful with diferent levels of surveillance coverage and to provide a longer planning window without requiring users to develop or maintain multiple forecasting systems.

Mantis 2 also provides an opportunity to investigate when mechanistic training improves predictions beyond learning from observed outbreaks. Simulations can supply diverse epidemic trajectories, but their usefulness depends on how well they represent real disease dynamics and surveillance processes. Observational data capture these characteristics directly, but their coverage is uneven across diseases and locations [10]. A key distinction is between the information available during training and the local observations supplied when making a forecast. Even a model trained on many outbreaks may receive only a few weeks of history for a particular disease and location. We therefore ask whether simulated training is especially valuable when this local evidence is limited, and whether its contribution changes as surveillance history accumulates.

We evaluated Mantis 2 and a collection of comparison models across 30 disease-location forecasting tasks. It had the best aggregate probabilistic performance across short, medium, and long horizons. Across 3,725 observed outbreak peaks, Mantis 2 missed roughly 30% fewer peaks than Chronos-2 (the second best model), with successful predictions providing more than six weeks of advance warning on average. To examine the contribution of simulation, we compare eleven training mixtures (with training data comprising more or less simulation data versus observational) while varying the historical context available. Simulation-rich training was particularly beneficial with limited context, its advantage diminished with longer histories, and it reduced exceptionally large forecast errors. These findings identify available surveillance history as an important determinant of the value of mechanistic training. By releasing Mantis 2, we aim to make these forecasting capabilities accessible to public health agencies and researchers without requiring a separate model for each disease or location.

Compared with the original Mantis, Mantis 2 extends the maximum forecast horizon from eight to sixteen weeks, increases usable historical context from 112 to 512 weeks, and supports joint forecasting of up to six variables rather than one target with an optional covariate. It combines observational and simulated training and replaces four separately deployed model variants with a single model. These expanded capabilities accompany substantial accuracy gains: in a separate comparison restricted to eight-week, single-target forecasting, Mantis 2 reduced pooled WIS by 31.42% and improved performance on all 27 eligible comparison datasets. Full details of the comparison are provided in Appendix G. Pretrained models, example data, evaluation scripts, and an interactive forecasting demo are available at www.github.com/carsondudley1/Mantis.

## Results

Mantis 2 generates probabilistic forecasts across diseases, locations, and surveillance variables using a single trained model (Figure 1). We trained the model on over 1,000 real-world infectious disease scenarios and 20,000 mechanistically simulated scenarios (with the balance of real versus simulated data used for training determined by the curriculum described in the Methods section). Both simulated and real scenarios contain between one and six variables, with a combination of disease surveillance signals and relevant covariates (e.g., weather). Mantis 2 uses up to 512 weeks of historical observations to produce multivariate forecasts for the next 16 weeks.

![](images/9ef11e718b6acfdd2c3415e8c833ee630c5b9a02cdaeb0e179b54f115e328bf5.jpg)  
Figure 1: Mantis 2 Workflow and Capabilities. Mantis 2 builds upon the original Mantis foundation model for probabilistic infectious disease forecasting. In Mantis 2, the neural network training pipeline incorporates a mixed training curriculum that incorporates both real-world and simulated disease trajectories. Mantis 2 also brings two new capabilities: multivariate forecasting increased to 6 variables and multi-horizon forecasts up to 16 weeks.

## Forecasting performance across tasks and horizons

We evaluated Mantis 2 and all comparison models on 30 disease-location forecasting tasks, each potentially containing multiple time series (Table 1). These tasks include both held-out periods or locations for diseases represented in training, as well as diseases entirely absent from the training set. Each model we evaluated produced forecasts every four weeks, beginning after eight weeks of observations and using all available history up to a rolling maximum of 512 weeks. Each forecast covered the subsequent 16 weeks, so there is some overlap in evaluation windows.

We compared Mantis 2 forecasting performance to six baseline models: Chronos-2 [9], a three-layer long short-term memory network (LSTM) [16], exponential smoothing (ETS) [17], seasonal autoregressive integrated moving average (SARIMA) [17], seasonal-naive forecasts (i.e., predict for a given week the value from the same time the previous year), and persistence forecasts (predict for each week in the forecast the last value before the forecast began). Every model received the same input data at each forecast point. The baseline models were trained separately for each disease, pooling observations across locations, whereas Mantis 2 and Chronos-2 were not modified for the evaluations. The baseline models requiring fitting were refit every 16 weeks.

We evaluated the original Mantis separately because it supports forecasts only up to eight weeks ahead and predicts a single outcome with at most one covariate [15]. It therefore could not participate in the full sixteen-week, joint multivariate evaluation summarized in Figure 2. To compare the two versions within the original model’s forecasting scope, we conducted a separate evaluation of single-target forecasts over weeks 1–8, reported in Appendix G.

![](images/b1828ed66caa474b8d5b08c6858fb5b6cf7ad63712f5f68fd3a542aa82b7a30e.jpg)

![](images/1c0e95af9742cdbda62862e12f77d88bd17f5ff29bfbb23aa7fb45b9608fdf51.jpg)

![](images/d3ddcc71d13110015ead7773972ed2b46da5c08d7bb195d707a55748e3f0ad9a.jpg)

D  
![](images/a7845b55352363cead2e297508d73d7967aca9e34b163e59d586f9101ae338df.jpg)

![](images/09f635c79ca930aa1920ecce1d0ba6736a0707127a0b8b4f09412b31d62fa971.jpg)

![](images/084c6b55dfac547eb675ef8146d45d35657bb3616673293f4a440cf182d25c80.jpg)  
Figure 2: Mantis 2 achieves strong probabilistic forecasting performance across diseases and forecast horizons. (A-C) Mean weighted interval score evaluated on log(1 + y)-transformed outcomes (log-WIS) over short (1-4 weeks), medium (1-8 weeks), and long (1-16 weeks) forecast horizons. Lower scores indicate better performance. Mantis 2 achieves the lowest score across all three horizon ranges. (D) Disease-level comparison of Mantis 2 and Chronos-2 zero-shot over the long horizon. Each point represents an evaluation dataset, and the dashed diagonal indicates equal performance. Points below the diagonal favor Mantis 2 (purple), while points above favor Chronos-2 (orange). The paired Wilcoxon signed-rank test indicates a significant diference in performance $( P < 0 . 0 0 1 )$ . (E) Distribution of model ranks across the 29 evaluation datasets with results available for all six models, ranked by long-horizon WIS. Segment widths show the percentage of datasets at each rank. Numbers within segments indicate dataset counts, and mean ranks are shown on the right. Mantis 2 ranks first on 22 datasets and second on the remaining seven. (F) Peak prediction accuracy across 3,725 peaks. Bars show the percentage of outbreak peaks each model never correctly predicted at weekly lead times of 1–14 weeks (lower is better). A prediction is considered correct when the forecasted peak occurs within ±2 weeks of the observed peak. For Mantis 2 and Chronos-2 we report the mean lead time of the first correct peak prediction.

Probabilistic forecast accuracy was assessed using the weighted interval score (WIS) [18], which rewards narrow predictive intervals while penalizing intervals that do not contain the observed outcome. We calculated WIS after applying the log(1 + y) transformation to observations and forecast quantiles, reducing the influence of diferences in outcome magnitude. Scores were summarized over short (1-4 week), medium (1-8 week), and long (1-16 week) forecast horizons. We also evaluated prediction interval calibration by comparing empirical coverage with target coverage levels (e.g., the 95% prediction interval should contain the true outcome 95% of the time). Finally, we assessed whether models correctly anticipated outbreak peak timing and how far in advance they did so. Anticipating peaks requires predicting a turning point before the decline is observed, making it a particularly challenging forecasting task. Advance warning of these turning points can support hospital stafing, capacity planning, and the allocation of supplies during periods of greatest demand.

Mantis 2 achieved the lowest aggregate log-WIS across all models for all three horizon ranges (Figure 2A-C). Performance gains were distributed across evaluation tasks. Mantis 2 ranked first among all evaluated models the majority of the time, with average rank 1.33. Since Chronos-2 was consistently the closest mode to Mantis 2’s performance, we evaluated whether the performance gain from Chronos-2 to Mantis 2 was statistically significant. We did a paired Wilcoxon signed-rank test, using the performance of each model on each dataset, and found that Mantis 2’s performance gain was statistically significant $( P < 0 . 0 0 1$ , Figure 2D). Task-level rankings provide a complementary assessment to the forecast-weighted aggregate scores by describing how consistently each model performs across distinct forecasting settings (Figure 2E).

Mantis 2 also substantially improved on its predecessor within this more restricted forecasting setting. Across 27 eligible datasets and 34,762 matched forecast origins, it reduced pooled eight-week WIS by 31.42% relative to original Mantis, with improvements on every dataset. Dataset-level reductions ranged from 10.37% to 45.55% (Appendix G). Both models forecast the same targets at the same origins, while retaining their respective historical context lengths and auxiliary-input configurations. Thus, Mantis 2 improves accuracy on tasks supported by the original model while also extending forecast horizons and enabling joint forecasting of multiple outcomes.

Figure 3 illustrates forecasts across six disease and location settings, from modern disease forecasts to historical epidemics. Each panel displays three 16-week forecasts issued 17 weeks apart. The panels therefore show forecasting over approximately one year, allowing predictions to be shown across seasonal rises, peaks, and declines. Although the forecasts are multivariate, only cases are displayed to facilitate comparison across diseases. Forecast medians and 95% prediction intervals illustrate both the predicted trajectories and their uncertainty.

## Anticipating outbreak peaks

Accurate forecasts of near-term disease activity do not necessarily imply that a model can anticipate when an outbreak will peak. A model can track an ongoing rise while failing to predict the transition from growth to decline. Anticipating this transition requires information about how an outbreak may develop beyond its current trajectory. It also addresses a practical planning question: how much longer will disease activity increase before it begins to subside? This timing can inform preparations for sustained demand on health services, including stafing and resource allocation. Because these preparations take time, a useful forecast must identify the turning point further in advance to support action.

We therefore conducted a separate evaluation of outbreak peak prediction across 3,725 observed peaks, assessing both whether models correctly predicted peak timing and how much advance warning they provided. Predicted peaks were identified as local maxima in the forecast median (0.5 quantile forecast), with two subsequent weeks of decline required to establish a peak. Thus, a 16-week forecast could identify peaks at lead times of up to 14 weeks. A peak prediction was considered correct when its timing fell within two weeks of the observed peak. We measured the proportion of observed peaks that were correctly anticipated across lead times of 1-14 weeks and, among successfully anticipated peaks, the lead time of the first correct prediction.

Mantis 2 correctly predicted over 80% of observed peaks, compared with 73% for Chronos-2, 46.3% for naive seasonal forecasts, 38.2% for ETS, 31.7% for the LSTM, and 0% for naive persistence (Figure 2F). Relative to Chronos-2, Mantis 2 reduced the proportion of missed peaks by roughly 30%. Among peaks each model successfully anticipated, the mean lead time of the first correct prediction was 6.43 weeks for Mantis

![](images/728caaf381e3bb38737b51e0d34f5648b2ba44bf9b80b02c26fb72dc82d28644.jpg)  
Figure 3: Forecasts from Mantis 2 over a year of six diseases. We display 16-week forecasts with a one week gap between, covering 51 weeks of each disease. The forecasts are generally accurate, with higher uncertainty (wider prediction intervals) during outbreak periods.

2 and 6.32 weeks for Chronos-2. These averages are conditional on model-specific subsets of successfully anticipated peaks and therefore do not represent a paired comparison of warning time (if you anticipate fewer peaks, it stands to reason that the ones you do anticipate are more obvious, so your lead time may be inflated). Mantis 2 correctly anticipated a larger proportion of observed peaks, with successful predictions providing more than six weeks of advance warning on average.

## Exploring the optimal balance of simulated and real world training data

An important question for developing infectious disease foundation models is whether training should emphasize the outbreaks that have been observed or the broader range of outbreaks that could occur. Synthetic pretraining ofers a way to learn from the latter, since you can simulate a very broad range of disease dynamics. Strong predictive performance has been achieved across domains by models pretrained entirely on synthetic datasets [19, 20], and the original Mantis demonstrated that training exclusively on mechanistic simulations could support forecasting across real-world diseases and surveillance settings [15]. Simulations can expand training beyond the (comparatively) limited number of outbreaks that have been observed and have high-quality, publicly available datasets and encode explicit scientific assumptions about transmission and observation. These assumptions provide an interpretable basis for the training distribution, and a model trained on these simulations can make predictions that are highly interpretable [21], but the utility of synthetic pretraining depends on how well the simulations represent the processes encountered in practice.

The simulation ensemble we trained on, Reservoir [22], spans human, vector, and environmental transmission, including asymptomatic infection, interventions such as quarantine and isolation, and pre- and post-exposure prophylaxis. It also represents the observation process through underreporting, reporting delays, and other surveillance efects. Because the simulator is broad and realistic, one might hypothesize that simulated training would provide suficient diversity for general forecasting, with observational data contributing little additional information. On the other hand, even a simulator that can represent a broad range of diseases in a realistic manner could omit important dynamics, misrepresent their frequency, or not accurately represent the noise and observation structure of real-world infectious disease surveillance. Observational training could supply these missing patterns and, if suficiently diverse, make simulated training redundant.

![](images/820cb54f46870b093791462462ca25d51760da2450ea39162d9ab769e957ff8c.jpg)

D Synthetic improvement vs context length  
![](images/5de5ca0ee25fba3d2f51dcf1424d8688f7f1f2642a91ff5ebc10c3c01de65fef.jpg)

![](images/144a8e99a0e8d0741206dd39b4bb840f194e4986389ffee7633d3bba5ebc98f0.jpg)  
<sub>Large</sub> <sub>forecast</sub> <sub>errors</sub> <sub>vs</sub> <sub>synthetic</sub> <sub>fraction</sub>F

<sub>Best</sub> <sub>synthetic</sub> <sub>fraction</sub> <sub>vs</sub> <sub>context</sub> <sub>length</sub>E  
![](images/0de68b04cb0f7180ddf33eaca738d4898230d01170b45d8abe7dc0a0314447da.jpg)

![](images/e377e76042c5aa68723baa84f5b7cb33bc7b1be48508ad3c1199d30b02dc860d.jpg)

![](images/fa7b647197d23a620cbc6e1cefbd5b0f6251c61a858d599f1fa7d67b47c3adaf.jpg)  
Figure 4: Synthetic training benefits depend on the dataset and available history. (A) WIS across synthetic training fractions for Project Tycho datasets, normalized by each dataset’s minimum across mixtures. Darker cells indicate better performance. (B) Improvement over real-only training for the 90% synthetic model with 8 versus 104 weeks of history. Connected points represent the same dataset and forecast origins. Positive values favor high-synthetic training. (C) Median improvement over real-only across all synthetic mixtures at five context lengths. (D) Observed improvement of best synthetic mixture over real-only within each surveillance-history. Synthetic pretraining mixtures are more useful with shorter context length. (E) Synthetic fraction attaining the lowest WIS for each context length. In $_ { \mathrm { D - E , } }$ points and shading show medians and interquartile ranges across datasets. (F) Frequency of forecast WIS exceeding five or ten times each dataset’s mean real-only WIS. Models with higher amounts of synthetic pretraining make extreme errors less frequently.

Observational data introduce a diferent concern: the datasets we have access to may not represent those in which forecasts are most needed. For example, our training dataset includes a lot of respiratory surveillance data obtained through Delphi Epidata [23], whose COVID-19 and flu data provide extensive coverage of respiratory diseases in the United States. Heavy reliance on these datasets could teach the model to expect dynamics specific to these respiratory diseases in the United States, but hurt performance in lessrepresented settings (for example, one of our evaluation datasets: dengue in Asian and South American countries). This creates a potential health equity concern if forecasting benefits depend on the availability of extensive historical surveillance.

We therefore compared models trained with fixed synthetic fractions from 0% to 100% (i.e., from training only on real-world data, to training only on simulated data, and with mixtures in between) in increments of 10 percentage points across a range of infectious disease types and with locations around the world. We only measure the performance of fixed mixtures of synthetic versus real data here; our released Mantis 2 uses a training curriculum that yields better results than any of the fixed quantities shown (see Methods and Appendix for more details on Mantis 2 training curriculum).

Neither observational nor simulation-rich training was uniformly preferred across datasets (Figure 4A). Within Project Tycho data [24] (historical US disease records), for example, smallpox and pertussis favored observational training, whereas hepatitis A and B favored simulation-rich mixtures. Malaria and tuberculosis showed relatively little variation across a broad range of mixtures. These responses raised a further possibility: the usefulness of simulated training might depend not only on which disease is forecast, or the location in question, but also on how much local evidence the model receives.

To evaluate the impact of surveillance history for a given disease on the amount of synthetic data that was useful, we varied the available context while keeping the forecasting task fixed. We evaluated all synthetic mixtures with 8, 16, 26, 52, and 104 weeks of history at each forecast origin across all evaluation datasets. With only eight weeks of observations, the 80% and 90% synthetic models improved median WIS over realonly training by 7.9% and 9.3%, respectively. With 104 weeks, their median improvements were instead −0.5% and −1.2%. The relative advantage of 90% synthetic training was greater with short history in the vast majority of datasets (Figure 4B). Thus, simulation-rich training was particularly useful when the model had little local evidence on which to condition its forecast. Simulation-only training improved median WIS by 8.2% with eight weeks but worsened it by 11.1% with 104 weeks (Figure 4C). Our results do not support the notion that either synthetic nor observation data are alone suficient to train a state-of-the-art foundation model for infectious disease. Instead, they show an interaction between the training distribution and the information supplied at prediction time.

That interaction also changed apparent disease-specific preferences. Although smallpox and pertussis favored observational training in the original evaluation, the 80% synthetic model improved their WIS by 8.8% and 18.3%, respectively, with only eight weeks of context. With 104 weeks, the corresponding improvements were −1.2% and −0.5%. A preference for observational training therefore cannot, by itself, be interpreted as evidence that a disease is poorly represented by the simulator. A rolling-origin evaluation showed the same pattern: both the median benefit of the best mixture of real and synthetic data declined as surveillance histories lengthened (Figure 4D,E).

We also examined whether training composition afected exceptionally large forecast errors, which can be obscured by summaries of typical performance. Across our evaluation datasets datasets, the mean fraction of forecasts that had error greater than five times the dataset’s mean real-only WIS was 0.364% for real-only training, compared with 0.261% and 0.207% for 80% and 90% synthetic training. At ten times the reference mean, the rates were 0.060%, 0.017%, and 0.017%, respectively (Figure 4F). Simulation-rich training could therefore mitigate extreme failures even in cases where it provides perhaps less overall performance benefit due to the availability of more extensive real-world data.

These findings suggest that the contribution of simulated training depends on the evidence available to the model. Broad mechanistic training was particularly valuable when context with a specific disease was very limited, whereas longer histories often favored observational training or reduced sensitivity to the mixture. It seems that a simulated prior can provide a model with knowledge of the full range of plausible infectious disease outcomes and their relative probabilities, which is very useful with limited context, and which helps the model avoid catastrophic errors due to predicting improbable outcomes. However, once the model has an established history of a given disease, this scientific prior is not as useful, because the mode is very skilled at adapting its predictions to a real-world disease history.

## Discussion

Mantis 2 provides a common forecasting model for infectious diseases across pathogens, locations, and surveillance signals without disease-specific retraining. Its multivariate forecasts extend up to sixteen weeks ahead, and its strong performance across the evaluated tasks demonstrates the potential for a shared model to support forecasting in diverse epidemiological settings. Mantis 2 also anticipated a larger proportion of observed outbreak peaks than the evaluated baselines, with successful predictions providing more than six weeks of advance warning on average. Together, these capabilities expand the range of forecasting tasks that can be addressed using a single pretrained system.

For public health agencies and researchers, a general-purpose model can reduce the need to develop and maintain a separate forecasting system for each disease and surveillance setting. Mantis 2 can jointly use multiple surveillance variables, allowing related signals to inform predictions while also accommodating settings in which only a single series is available. Its longer forecast horizon is especially useful because many important decisions require weeks of lead time, including stafing, procurement, and capacity planning. Peak forecasts provide complementary information about when disease activity may decrease and how long elevated demand may persist. These capabilities could make forecasts useful for a broader range of planning decisions.

Our training experiments identify one very relevant consideration for extending forecasting to settings with limited data: the value of mechanistic simulation depends on the amount of local surveillance history available at prediction time. Simulation-rich training was especially beneficial with short histories, and even diseases that favored observational training with longer histories benefited from greater simulated exposure when context was limited. Mechanistic training may therefore supply useful background knowledge (scientific prior) before a surveillance series has accumulated enough observations to establish its own recurring patterns. For emerging outbreaks and newly established surveillance streams, the practical implication is that usefu forecasting need not depend entirely on having years of local observations. An interesting future work would be investigating how the model internally combines simulated and observational information.

Our mixture experiments held the proportion of simulated training examples constant throughout training, whereas the released Mantis 2 uses a curriculum that gradually reduces this proportion from 70% to zero. The mixture experiments therefore clarify how training composition afects forecasting, but do not establish how best to schedule exposure to simulated and observational data. Future work could repeat these comparisons under diferent curricula to examine how the amount and timing of simulated training jointly afect performance, particularly when local surveillance history is limited.

Forecasting reliability also matters beyond average accuracy. Simulation-rich training reduced the frequency of exceptionally large forecast errors, suggesting that training composition can influence the risk of occasional extreme failures as well as typical performance. Such failures are particularly relevant to operational use, where an implausible forecast can undermine confidence or distort preparations. This result supports evaluating foundation models across multiple dimensions, including average probabilistic accuracy, anticipation of turning points, and the frequency of extreme errors. Further work should establish which failure modes mechanistic training reduces and whether these benefits persist in prospective forecasting.

Several limitations qualify these findings. Our evaluation of Mantis 2 is retrospective, and improved forecast scores do not by themselves demonstrate improved public health decisions. However, the original Mantis has shown prospective use [25], and Mantis 2 outperforms its predecessor in our retrospective comparison by more than 30% (Appendix G). Likewise, the ability to generate sixteen-week forecasts does not imply equa reliability across horizons or settings. Both the observational training data and the mechanistic simulations represent selected diseases, locations, and processes; their coverage cannot guarantee performance for every emerging pathogen or surveillance system. Finally, our conclusions about how surveillance history afects the value of simulated training are based on experiments that truncate existing series while holding forecast origins fixed. These experiments isolate the efect of having fewer observations, but a short history drawn from an established series does not necessarily represent the dynamics or uncertainty encountered early in

Continued on next page

a new outbreak. We therefore hypothesize that simulation-rich training will be particularly useful early in emerging outbreaks, but do not directly test that hypothesis here. Additionally—the fact the optimal synthetic fraction varies across diseases and context lengths raises the question of what blend may be best to use for a multi-purpose model such as Mantis 2, which is intended as a foundation model to be used across settings. In this work, we opted to use a blend that amount to 35% synthetic data, but future work would be useful to explore further optimizing this mix (balancing across diseases and between the goals of overall performance vs. minimizing large errors), as well as to evaluate which simulations (models, real-world efects, etc.) most improve performance.

Mantis is already being used by several U.S. state governments and is supporting forecasting eforts by federal governments on other continents. This existing use provides a practical setting for evaluating and extending the capabilities introduced in Mantis 2. By releasing the model, we aim to make multivariate, longer-horizon disease forecasting more accessible to public health agencies and researchers without requiring each group to train its own disease-specific system. The release will also enable independent evaluation across additional pathogens and surveillance settings. Mantis 2 ofers a foundation for expanding access to infectious disease forecasts, including where limited local history makes forecasting both dificult and urgently needed.

## Methods

## Training data and preprocessing

Table 1: Evaluation Dataset. The evaluation dataset for Mantis 2 consists of 23 diseases across 11 countries and eight transmission categories (enteric, vector-borne, respiratory, airborne, bloodborne, close contact, zoonotic, and healthcare-associated). Most datasets consist of univariate case counts; multivariate datasets include additional surveillance signals.
<table><tr><td colspan="4">In-domain evaluation</td></tr><tr><td>Disease</td><td>Location</td><td>Transmission</td><td>Signals</td></tr><tr><td>Acute diarrhea [26]</td><td>Thailand (Provincial)</td><td>Enteric</td><td>Univariate</td></tr><tr><td>Enteric fever [26]</td><td>Thailand (Regional)</td><td>Enteric</td><td>Univariate</td></tr><tr><td>Hepatitis A [26]</td><td>Thailand (Regional)</td><td>Enteric</td><td>Univariate</td></tr><tr><td>Typhoid fever [27]</td><td>USA (Regional)</td><td>Enteric</td><td>Univariate</td></tr><tr><td>Salmonellosis [27] Hepatitis A [27]</td><td>USA (Regional)</td><td>Enteric</td><td>Univariate</td></tr><tr><td></td><td>USA (State)</td><td>Enteric</td><td>Univariate</td></tr><tr><td>Dengue [28]</td><td>Singapore (National)</td><td>Vector-borne</td><td>Univariate</td></tr><tr><td>Dengue [28]</td><td>Nicaragua (Provincial)</td><td>Vector-borne</td><td>Univariate</td></tr><tr><td>Dengue [28] Dengue [28]</td><td>Indonesia Thailand</td><td>Vector-borne Vector-borne</td><td>Univariate</td></tr><tr><td></td><td>(Provincial)</td><td></td><td>Univariate</td></tr><tr><td>Malaria [27] Rocky Mountain Spotted</td><td>USA (Regional) USA (National)</td><td>Vector-borne Vector-borne</td><td>Univariate Univariate</td></tr><tr><td>Fever [27]</td><td></td><td></td><td></td></tr><tr><td>Influenza [26]</td><td>Thailand (Provincial)</td><td>Respiratory</td><td>Univariate</td></tr><tr><td>Disease</td><td>Location</td><td>Transmission</td><td>Signals</td></tr><tr><td>HFMD [26]</td><td>Thailand (Provincial)</td><td>Respiratory, Enteroviral</td><td>Univariate</td></tr><tr><td>Measles [27]</td><td>USA (State)</td><td>Respiratory</td><td>Univariate</td></tr><tr><td>Whooping cough [27]</td><td>USA (Regional)</td><td>Respiratory</td><td>Univariate</td></tr><tr><td>Influenza (ILI)</td><td>Nigeria</td><td>Respiratory</td><td>Multivariate (ILI, SARI, ARI, PNEU, deaths)</td></tr><tr><td>Tuberculosis [26]</td><td>Thailand</td><td>Bloodborne, Airborne</td><td>Univariate</td></tr><tr><td>Hepatitis B [26]</td><td>(Provincial) Thailand</td><td>Bloodborne</td><td>Univariate</td></tr><tr><td>Hepatitis B [27]</td><td>(Provincial) USA (Regional)</td><td>Bloodborne</td><td>Univariate</td></tr><tr><td>Smallpox</td><td>USA (Regional)</td><td>Close contact</td><td>Univariate</td></tr><tr><td>Brucellosis [27]</td><td>USA (Regional)</td><td>Zoonotic</td><td>Univariate</td></tr><tr><td>Tuberculosis [27]</td><td>USA (State)</td><td>Bloodborne, Airborne</td><td>Univariate</td></tr><tr><td colspan="4">Out-of-domain evaluation</td></tr><tr><td>COVID-19 [29]</td><td>USA (State)</td><td>Respiratory</td><td>Multivariate (cases, deaths, hospitalizations,</td></tr><tr><td>Candida auris [30]</td><td>USA (National)</td><td>HAI</td><td>ED visits) Univariate</td></tr><tr><td>Carbapenem-resistant</td><td>USA (National)</td><td>HAI</td><td>Univariate</td></tr><tr><td>Enterobacteriaceae [30]</td><td></td><td></td><td></td></tr><tr><td>Dengue [28]</td><td>Ecuador (National)</td><td>Vector-borne</td><td>Univariate</td></tr><tr><td>Meningitis [27]</td><td>USA (Regional)</td><td>Respiratory, Close contact</td><td>Univariate</td></tr><tr><td>Influenza (ILI) [31]</td><td>Austria</td><td>Respiratory</td><td>Multivariate (ILI, SARI, ARI, PNEU, deaths)</td></tr><tr><td>Measles [26]</td><td>Thailand (Regional)</td><td>Respiratory</td><td>Univariate</td></tr><tr><td>Smallpox [27]</td><td>USA (Regional)</td><td>Close contact</td><td>Univariate</td></tr><tr><td>Mpox [32]</td><td>Sierra Leone</td><td>Close contact, Zoonotic</td><td>Multivariate (cases, deaths)</td></tr><tr><td>Ebola virus disease [33]</td><td>DRC (National)</td><td>Close contact, Zoonotic</td><td>Multivariate (confirmed cases, deaths, probable</td></tr><tr><td>Cholera [34]</td><td>Ethiopia (National)</td><td>Enteric</td><td>cases) Multivariate (cases, deaths)</td></tr></table>

Mantis 2 was trained on observational infectious disease surveillance data and epidemic trajectories generated using the Reservoir mechanistic simulator [22]. Observational data covered multiple diseases, locations, and surveillance systems, with individual examples containing up to six variables. Approximately 20,000 simulated scenarios represented human, vector, and environmental transmission under varied epidemiological parameters, population characteristics, interventions, and observation processes, including underreporting and reporting delays. Both sources supplied multivariate combinations of surveillance signals. Full data sources, inclusion criteria, and simulator settings are provided in the Appendix.

Time series were represented at weekly resolution, with counts aggregated to this resolution if daily values were available. Missing observations were infrequent and were filled using spline interpolation over each complete series before temporal splitting. Imputed values were retained in evaluation targets and included in scoring. Inputs were subsequently transformed using the inverse hyperbolic sine function as a variancestabilizing transformation [9] and normalized using instance normalization [35]. Complete preprocessing and training-example construction procedures are described in the Appendix.

## Model architecture and training

Mantis 2 was initialized from pretrained Chronos-2 weights [9], and the full model was trained on the combined infectious disease data. The model accepts up to six variables and uses up to 512 weeks of historical context to generate probabilistic forecasts for the subsequent sixteen weeks. Calendar covariates provide additional seasonal information. Forecasts are represented by predicted quantiles for each target and forecast horizon, with forecasts provided for 21 quantiles: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]. We incorporated latent chain-of-thought computation with two recurrences, allowing intermediate hidden representations to undergo additional processing before forecast generation.

Training used a curriculum in which the probability of sampling simulated examples decreased linearly from 70% to zero. This schedule progressively shifted training from predominantly simulated trajectories to entirely observational data. We used the final training checkpoint for evaluation, without early stopping. The architecture, training objective, optimization settings, and computational requirements are detailed in the Appendix.

## Evaluation datasets and forecast protocol

Evaluation used two forms of data separation. For disease-location datasets represented in training, we used a temporal split. For out-of-domain evaluation, entire disease-location datasets were excluded from Mantis 2 training. Here, out-of-domain refers to an unseen dataset rather than necessarily an unseen disease: a disease could be represented in training through data from another location. Evaluation datasets and their surveillance signals are listed in Table 1.

Forecasts were issued at rolling origins spaced four weeks apart, with a minimum of eight weeks of historical context. At each origin, models received the preceding time series, subject to a maximum context length of 512 weeks, and forecast the following sixteen weeks. Designated surveillance targets contributed to evaluation scores, while auxiliary variables (e.g., weather) could inform forecasts without being scored. Mantis 2 and Chronos-2 were applied without task-specific parameter updates.

We compared Mantis 2 with Chronos-2 [9], as well as a three-layer long short-term memory (LSTM) neural network [16], exponential smoothing (ETS), seasonal-naive forecasts, seasonal autoregressive integrated moving average (SARIMA), and persistence forecasts. Seasonal-naive forecasts used the corresponding observation from the previous year, while persistence forecasts used the most recent value as their central prediction. Baseline fitting procedures, refitting schedules, and uncertainty estimation are described in the Appendix, together with an additional comparison against the original Mantis.

## Forecast accuracy and peak prediction

Probabilistic forecast accuracy was assessed using the weighted interval score (WIS) [18], with lower values indicating better performance. WIS was calculated after applying the log(1 + y) transformation to observations and forecast quantiles. We summarized performance over short (weeks 1-4), medium (weeks 1-8), and long (weeks 1-16) horizons. Models were additionally ranked within evaluation datasets, and Mantis 2 and Chronos-2 were compared using a paired Wilcoxon signed-rank test on dataset-level long-horizon WIS. Metric definitions, score aggregation, and treatment of unavailable predictions are specified in the Appendix.

We conducted a separate peak-timing evaluation using forecasts issued at weekly origins. Observed peaks were identified using the detection and eligibility criteria described in the Appendix. Predicted peaks were identified as local maxima in the forecast median, requiring two subsequent weeks of decline. A sixteen-week forecast could therefore identify peaks at lead times of up to fourteen weeks. A prediction was considered correctly timed when the predicted peak fell within two weeks of the observed peak.

For each model, we calculated the proportion of observed peaks never correctly anticipated at lead times of one to fourteen weeks. Among successfully anticipated peaks, we calculated the mean lead time of the first correct prediction. These mean lead times were conditional on model-specific subsets of successfully anticipated peaks and therefore did not constitute a paired comparison of warning time.

## Training-mixture and surveillance-history experiments

To examine the contribution of simulated training data, we trained eleven models using constant simulated fractions from 0% to 100% in increments of ten percentage points, holding the remaining training configuration fixed. These models were distinct from the curriculum-trained Mantis 2 model.

We evaluated each mixture with 8, 16, 26, 52, and 104 weeks of historical context, holding forecast origins and scored outcomes fixed across context lengths. This design isolated the efect of the amount of history supplied at prediction time. For each dataset and context length, improvement relative to real-only training was calculated as

$$
1 0 0 \times \frac { \mathrm { W I S * r e a l - W I S * m i x t u r e } } { \mathrm { W I S _ { r e a l } } } ,
$$

so that positive values indicated an advantage from simulated training. We summarized improvements across datasets and identified the simulated fraction attaining the lowest WIS at each context length. We also examined mixture performance as available surveillance history accumulated during rolling-origin evaluation.

To assess extreme forecast errors, we calculated the frequency with which forecast WIS exceeded five or ten times the corresponding dataset’s mean real-only WIS. The same dataset-specific thresholds were applied across training mixtures.

## Acknowledgments

This project was made possible by the Insight Net cooperative agreement with University of Michigan (5 NU38FT000002-02-00) from the CDC’s Center for Forecasting and Outbreak Analytics (CDC-RFA-FT-23- 0069). Its contents are solely the responsibility of the authors and do not necessarily represent the oficial views of the Centers for Disease Control and Prevention. The authors thank Julie Gilbert for her assistance with gathering data for an earlier version of the manuscript. The authors thank Kevin M. Bakker for his assistance in finding and collecting data from the Thai Ministry of Public Health. The authors also thank the two anonymous reviewers whose comments have greatly strengthened the manuscript.

## References

[1] Chelsea S. Lutz et al. Applying infectious disease forecasting to public health: a path forward using influenza forecasting examples. BMC Public Health, 19(1659), 2019.

[2] Angel N. Desai et al. Real-time epidemic forecasting: challenges and opportunities. Health Security, 17(4):268–275, 2019.

[3] Nicholas G. Reich, Logan C. Brooks, Spencer J. Fox, et al. A collaborative multiyear, multimodel assessment of seasonal influenza forecasting in the united states. Proceedings of the National Academy of Sciences, 116(8):3146–3154, 2019.

[4] Michael A. Johansson, Karyn M. Apfeldorf, Scott Dobson, et al. An open challenge to advance probabilistic forecasting for dengue epidemics. Proceedings of the National Academy of Sciences, 116(48):24268–24274, 2019.

[5] Estee Y. Cramer, Evan L. Ray, et al. Evaluation of individual and ensemble probabilistic forecasts of COVID-19 mortality in the united states. Proceedings of the National Academy of Sciences, 119(15):e2113561119, 2022.

[6] Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. Expert Certification.

[7] Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting, 2024.

[8] Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers, 2024.

[9] Abdul Fatir Ansari, Oleksandr Shchur, et al. Chronos-2: From univariate to universal forecasting, 2025.

[10] Leonardo S. Bastos et al. A modelling approach for correcting reporting delays in disease surveillance data. Statistics in Medicine, 38(22):4363–4377, 2019.

[11] Carson Dudley, Yutong Bi, Xiaofeng Liu, and Samet Oymak. In-context learning under regime change, 2026.

[12] Sarah Cobey. Modeling infectious disease dynamics. Science, 368(6492):713–714, 2020.

[13] C. Jessica E. Metcalf, Dylan H. Morris, and Sang Woo Park. Mathematical models to guide pandemic response. Science, 369(6502):368–369, 2020.

[14] Carson Dudley, Reiden Magdaleno, Christopher Harding, and Marisa Eisenberg. Training neural networks on mechanistic simulations improves scientific inference. Scientific Reports, 2026.

[15] Carson Dudley, Reiden Magdaleno, Christopher Harding, Ananya Sharma, Emily Martin, and Marisa Eisenberg. Mantis: A foundation model for mechanistic disease forecasting. Proceedings of the National Academy of Sciences, 2026.

[16] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735– 1780, 1997.

[17] George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung. Time Series Analysis: Forecasting and Control. Wiley, 5th edition, 2015.

[18] Johannes Bracher, Evan L. Ray, Tilmann Gneiting, and Nicholas G. Reich. Evaluating epidemic forecasts in an interval format. PLOS Computational Biology, 17(2):e1008618, 2021.

[19] Noah Hollmann, Samuel M¨uller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second, 2023.

[20] Jingang Qu, David Holzm¨uller, Ga¨el Varoquaux, and Marine Le Morvan. Tabicl: A tabular foundation model for in-context learning on large data, 2025.

[21] Carson Dudley and Marisa Eisenberg. Learning from simulators: A theory of simulation-grounded learning. arXiv preprint arXiv:2509.18990, 2025.

[22] Carson Dudley, Reiden Magdaleno, and Marisa Eisenberg. Reservoir: A large-scale simulated dataset for training and evaluating epidemiological models, 2026.

[23] David C. Farrow, Logan C. Brooks, Ryan J. Tibshirani, and Roni Rosenfeld. Delphi epidata api. https://github.com/cmu-delphi/delphi-epidata, 2015. Carnegie Mellon University, Delphi Re search Group.

[24] Willem G. van Panhuis, Anne Cross, Donald S. Burke, et al. Project tycho 2.0: a repository to improve the integration and reuse of data for global population health. Journal of the American Medical Informatics Association, 25(12):1608–1617, 2018.

[25] Hillary M Topazian, Theresa R Sheets, Randon J Gruninger, Joshua Kelley, Nathan LaCross, Matthew H Samore, Eric Lofgren, and Lindsay T Keegan. Rethinking respiratory disease forecasting: temporal heterogeneity between surveillance predictors and outcomes drives forecast instability. medRxiv, 2026.

[26] Thailand. Bureau of Epidemiology Department of Disease Control. Thailand bureau of epidemiology department of disease control moph. scrub typhus. accessed january 2, 2021. http://doe.moph.go.th/surdata/index.php.

[27] Willem Van Panhuis, Anne Cross, and Donald Burke. Project tycho level 2 data: Counts of multiple diseases reported in united states of america, 1888-2014. Project Tycho, April 2018.

[28] J Clarke, A Lim, P Gupte, DM Pigott, WG Van Panhuis, and OJ Brady. A global dataset of publicly available dengue case count data. Scientific Data, 11(1):296, 2024.

[29] David C. Farrow, Logan C. Brooks, Aaron Rumack, Ryan J. Tibshirani, and Roni Rosenfeld. Delphi epidata api, 2015.

[30] Centers for Disease Control and Prevention. NNDSS weekly data. CDC.gov, n.d.

[31] World Health Organization. FluNet. https://www.who.int/tools/flunet, 2026. Accessed: YYYY-MM-DD.

[32] World Health Organization. 2022–24 mpox (Monkeypox) outbreak: Global trends. https:// worldhealthorg.shinyapps.io/mpx\_global/, 2024. Accessed: 2026-04-09.

[33] DRC Ministry of Health. Ebola cases and deaths in the North Kivu Ebola outbreak in the Democratic Republic of the Congo (DRC). https://data.humdata.org/dataset/ ebola-cases-and-deaths-drc-north-kivu, 2020. Accessed: 2026-04-09.

[34] John R Giles and Christopher W Lorton. MOSAIC: Metapopulation Outbreak Simulation And Interventions for Cholera (MOSAIC), 2025. R package version 0.13.13.

[35] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022.

[36] Seran Hakki, Jie Zhou, Jakob Jonnerby, Anika Singanayagam, Jack L Barnett, Kieran J Madon, Aleksandra Koycheva, Christine Kelly, Hamish Houston, Sean Nevin, et al. Onset and window of sars-cov-2 infectiousness and temporal correlation with symptom onset: a prospective, longitudinal, community cohort study. The Lancet Respiratory Medicine, 10(11):1061–1073, 2022.

[37] Raymond Gani and Steve Leach. Epidemiologic determinants for modeling pneumonic plague outbreaks. Emerging infectious diseases, 10(4):608, 2004.

[38] Mathew V Kiang, Kate M Bubar, Yvonne Maldonado, Peter J Hotez, and Nathan C Lo. Modeling reemergence of vaccine-eliminated infectious diseases under declining vaccination in the us. JAMA, 333(24):2176–2187, 2025.

[39] Hualei Xin, Yu Li, Peng Wu, Zhili Li, Eric HY Lau, Ying Qin, Liping Wang, Benjamin J Cowling, Tim K Tsang, and Zhongjie Li. Estimating the latent period of coronavirus disease 2019 (covid-19). Clinical Infectious Diseases, 74(9):1678–1681, 2022.

[40] Arthur WD Edridge, Joanna Kaczorowska, Alexis CR Hoste, Margreet Bakker, Michelle Klein, Katherine Loens, Maarten F Jebbink, Amy Matser, Cormac M Kinsella, Paloma Rueda, et al. Seasona coronavirus protective immunity is short-lasting. Nature medicine, 26(11):1691–1693, 2020.

[41] Ian J Amanna, Nichole E Carlson, and Mark K Slifka. Duration of humoral immunity to common viral and vaccine antigens. New England Journal of Medicine, 357(19):1903–1915, 2007.

[42] Pratha Sah, Meagan C Fitzpatrick, Charlotte F Zimmer, Elaheh Abdollahi, Lyndon Juden-Kelly, Seyed M Moghadas, Burton H Singer, and Alison P Galvani. Asymptomatic sars-cov-2 infection: A systematic review and meta-analysis. Proceedings of the National Academy of Sciences, 118(34):e2109229118, 2021.

[43] Nancy HL Leung, Cuiling Xu, Dennis KM Ip, and Benjamin J Cowling. The fraction of influenza virus infections that are asymptomatic: a systematic review and meta-analysis. Epidemiology (Cambridge, Mass.), 26(6):862, 2015.

[44] Diana Buitrago-Garcia, Aziz Mert Ipekci, Leonie Heron, Hira Imeri, Lucia Araujo-Chaveron, Ingrid Arevalo-Rodriguez, Agust´ın Ciapponi, Muge Cevik, Anthony Hauser, Muhammad Irfanul Alam, et al. Occurrence and transmission potential of asymptomatic and presymptomatic sars-cov-2 infections: Update of a living systematic review and meta-analysis. PLoS medicine, 19(5):e1003987, 2022.

[45] Helen J Wearing, Pejman Rohani, and Matt J Keeling. Appropriate models for the management of infectious diseases. PLoS medicine, 2(7):e174, 2005.

[46] Ruian Ke, Ethan Romero-Severson, Steven Sanche, and Nick Hengartner. Estimating the reproductive number r0 of sars-cov-2 in the united states and eight european countries and implications for vaccination. Journal of theoretical biology, 517:110621, 2021.

[47] Roya Nikbakht, Mohammad Reza Baneshi, Abbas Bahrampour, and Abolfazl Hosseinnataj. Comparison of methods to estimate basic reproduction number (r0) of influenza, using canada 2009 and 2017-18 a (h1n1) data. Journal of research in medical sciences: the oficial journal of Isfahan University of Medical Sciences, 24:67, 2019.

[48] Ying Liu, Kate Lillepold, Jan C Semenza, Yesim Tozan, Mikkel BM Quam, and Joacim Rockl¨ov. Reviewing estimates of the basic reproduction number for dengue, zika and chikungunya across global climate zones. Environmental Research, 182:109114, 2020.

[49] Zindoga Mukandavire, Shu Liao, Jin Wang, Holly Gaf, David L Smith, and J Glenn Morris Jr. Estimating the reproductive numbers for the 2008–2009 cholera outbreaks in zimbabwe. Proceedings of the National Academy of Sciences, 108(21):8767–8772, 2011.

[50] Carson Dudley and Samet Oymak. Latent chain-of-thought improves structured-data transformers, 2026.

[51] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.

[52] Graham C. Gibson, Spencer J. Fox, et al. Improving outbreak forecasts through model augmentation. Proceedings of the National Academy of Sciences (PNAS), 122(43), October 2025.

[53] Yong Liu and Xin Yao. Simultaneous training of negatively correlated neural networks in an ensemble. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 29(6):716–725, 1999.

[54] Carson Dudley and Marisa Eisenberg. Not all accuracy is equal: Prioritizing independence in infectious disease forecasting. Epidemics, 57(100943), 2026.

## Appendices

## A Diferences from the original Mantis

Mantis 2 retains the original Mantis’ objective of providing probabilistic infectious disease forecasts across pathogens, locations, and surveillance settings without disease-specific retraining [15]. Its development addresses several practical limitations of the original model: the amount of historical information it could use, the number of surveillance variables it could incorporate and forecast, the length of its forecast horizon, and the need to select among separate model variants, depending on the forecast horizon and if multiple variables are used. We also broadened the training approach to combine mechanistic simulations with observationa data. Below, we describe these changes and the considerations that motivated them.

## Longer historical context and a more eficient architecture

We increased the maximum historical context from 112 to 512 weeks, extending the usable surveillance history from just over two years to nearly ten years. A two-year window provides relatively few examples of seasonal recurrence and may omit less frequent but informative epidemic patterns. For example, a recent history containing only single-peaked influenza seasons may provide limited local evidence for anticipating a season with two distinct peaks. A longer window can include previous instances of such behavior, as well as greater variation in epidemic timing, magnitude, and duration. These observations can help the model interpret an ongoing outbreak in the context of a broader history of disease activity. Our surveillance-history experiments support the value of additional context over the lengths evaluated. The expanded context capacity allows longer records to be used when available.

Increasing context length required an architecture that could process longer sequences eficiently. Our most important design consideration for both Mantis models (beyond that they forecast accurately) is practical accessibility: generating forecasts should be feasible on a laptop, with computational requirements compatible with routine use by public health practitioners and researchers. Simply extending the sequence length of the original custom CNN-transformer architecture we used for the original Mantis would increase the computational burden of temporal attention (which scales quadratically in sequence length, so making the context 5x longer would cost 25x as much). We therefore adopted the Chronos-2 backbone [9], whose patching and attention mechanisms support eficient processing of longer, multivariate histories.

With a patch size of 16, each consecutive block of sixteen weekly observations is embedded into a single token. A 512-week history therefore contributes 32 historical patch tokens per variable, substantially shortening the sequence processed by the transformer. Patching embeds the observations within each block together rather than replacing them with a single temporal average (see model architecture section).

Chronos-2 also separates attention across time from attention across related series. Temporal attention operates across patch positions within each series, whereas group attention operates across series within the same forecasting group at a shared patch position [9]. Alternating these operations allows information to propagate across both dimensions without applying unrestricted attention to every pair of variable-time tokens. Together, these architectural choices support the longer context window and expanded multivariate inputs while limiting their computational cost.

## Combining observational and simulated training data

The original Mantis was trained entirely on mechanistically simulated outbreaks. This choice reflected concerns about learning excessively from the diseases and surveillance systems most heavily represented in those data (we had extensive data for respiratory diseases in the United States, specifically), as wel as a broader demonstration of proof-of-concept that a mechanistic simulation-based foundation model was feasible. Simulation provided a way to generate a much broader range of epidemic trajectories without repeatedly training on a comparatively narrow observational dataset.

For Mantis 2, we assembled a larger and more diverse collection of observational disease datasets, making it possible to investigate whether real-world training could complement mechanistic simulation. Reservoir [22] provides diverse simulated trajectories, but no simulator can perfectly reproduce the distribution of real surveillance data. Diferences may arise from omitted mechanisms, the frequency with which particular dynamics are simulated, or the representation of reporting noise and other observation processes. Observational training ofers examples of these characteristics directly from the settings in which the model will be used.

We therefore trained Mantis 2 using both data sources and compared training mixtures ranging from entirely observational to entirely simulated. These experiments showed that combining the two could improve performance over either source alone, while also revealing that the preferred balance depended on the forecasting task and available surveillance history. The final model uses a curriculum that progressively reduces simulated exposure and ends with observational training. The rationale is to expose the model initially to a broad range of mechanistically plausible trajectories and subsequently adapt it to the characteristics of observed surveillance data.

## Extending the forecast horizon

We increased the maximum forecast horizon from eight to sixteen weeks. The motivation was to provide information over a longer planning period for decisions such as stafing, procurement, and hospital capacity preparation. Longer forecasts also allow an anticipated rise, peak, and decline to be represented within a

single forecast trajectory, potentially providing information about both the timing and duration of elevated disease activity.

The extended horizon is an additional capability rather than a requirement to use all sixteen forecast weeks. Users can select the portion relevant to their planning needs. Because forecast usefulness depends on lead time and setting, we evaluated performance separately over short, medium, and long horizons.

## Joint forecasting of multiple surveillance variables

The original Mantis forecasts a single target series, optionally informed by one additional covariate. Mantis 2 supports joint forecasting of up to six variables. This change reflects the fact that surveillance systems often provide several related measurements of the same outbreak, such as cases, hospital admissions, emergency department visits, deaths, etc. Their relationships can provide information that is unavailable from any one series alone, including diferences in timing between upstream and downstream indicators.

Joint forecasting allows these signals to inform one another within a single model application and produces forecasts for multiple outcomes without requiring a separate run configured around each target. Additional variables can also be supplied as auxiliary inputs when only a subset of outcomes is of interest. The same model remains applicable when only one series is available, allowing its use across settings with diferent levels of surveillance coverage.

## A single model for diferent forecasting needs

The original release required users to choose among four separately downloaded models: four- or eight-week forecasting, each with or without a covariate. Although these variants supported diferent use cases, selecting and managing separate models added complexity to deployment.

Mantis 2 consolidates these capabilities into a single model supporting between one and six variables and forecast horizons from one to sixteen weeks. Users can change the available inputs or requested horizon without downloading a diferent set of weights. This simplifies integration into forecasting workflows and allows the same deployed model to accommodate changes in surveillance availability and planning needs.

## B Data

## Synthetic training data

We generated 20,000 synthetic epidemic trajectories across three transmission mechanisms using a largescale epidemic simulator called Reservoir [22]. Each simulation spans 5,000 days and contains 6 surveillance modalities: cases, hospitalizations, hospital occupancy, wastewater, syndromic, and deaths. The following epidemiological priors were used to generate the synthetic data set:

<table><tr><td>Quantity</td><td>Prior</td><td>Notes / clipping</td><td>Citation</td></tr><tr><td>Infectious period (d)</td><td>LN(log 5, 0.5)</td><td>clip [1, 30]</td><td>[36, 37]</td></tr><tr><td>Latent period (d)</td><td>LN(log 5, 0.6)</td><td>clip [0.5, 21]</td><td>[38, 39]</td></tr><tr><td>Immunity duration (d)</td><td>LN(log 365, 1.0)</td><td>clip [14, 10950]</td><td>[40, 41]</td></tr><tr><td> $p _ { a }$  (asymptomatic)</td><td>Beta(3, 7)</td><td>mean 0.30</td><td>[42, 43]</td></tr><tr><td>α (asymp. trans.)</td><td>Beta(2, 5)</td><td>mean 0.29</td><td>[44]</td></tr><tr><td> $\kappa \ ( I _ { a }  I _ { s } )$ </td><td colspan="3">exp U(log 0.02, log 0.5)</td></tr><tr><td>(waning stages)  $n _ { w }$ </td><td>{1, 1, 1, 2, 3} uniform pick</td><td>mostly 1</td><td>[45]</td></tr></table>

Table 2: Shared epidemiological priors (sample epi params).

<table><tr><td>Mechanism</td><td>Quantity</td><td>Prior</td><td>Cite</td></tr><tr><td rowspan="3">all</td><td> $R _ { 0 } { \mathrm { ~ ( b a s e ) } }$ </td><td> $\mathrm { G a m m a } ( 1 . 2 , \mathrm { r a t e } 1 . 0 ) + 0 . 8 $  (mean 2.0)</td><td>[46, 47]</td></tr><tr><td> $R _ { 0 } \ \mathrm { ( v e c t o r ) }$ </td><td> $\mathrm { G a m m a } ( 3 . 2 , \mathrm { r a t e } 0 . 9 ) + 0 . 4$  (mean 3.96)</td><td>[48]</td></tr><tr><td> $R _ { 0 } { \mathrm { ~ ( e n v i r o n m e n t a l ) } }$ </td><td> $\mathrm { G a m m a } ( 2 . 6 , \mathrm { r a t e } 0 . 8 5 ) + 0 . 4$  (mean 3.46)</td><td>[49]</td></tr><tr><td rowspan="5">vector</td><td> $\sigma _ { v }$ </td><td> $\begin{array} { r } { \exp \mathcal { U } ( \log \frac { 1 } { 2 1 } , \log \frac { 1 } { 4 } ) } \end{array}$ </td><td>[48]</td></tr><tr><td> $\mu _ { v }$ </td><td> $\exp \mathcal { U } ( \log \frac { \mathbb { T } } { 3 0 } , \log \frac { \mathbb { 1 } } { 7 } )$ </td><td></td></tr><tr><td> $a \ ( \mathrm { b i t i n g } )$ </td><td>U(0.1, 1.0)</td><td>333</td></tr><tr><td> $b _ { h } , b _ { v }$ </td><td>U(0.1,0.8) each</td><td></td></tr><tr><td> $f _ { N v } , f _ { I v }$ </td><td>U(0.05, 2.0), U(0.001, 0.05)</td><td>[48]</td></tr><tr><td rowspan="5"></td><td>environmental environmental share</td><td>U(0.1,0.9) of  $R _ { 0 }$ </td><td>[49]</td></tr><tr><td> $\eta _ { I }$ </td><td>expU(log 0.1, log 1.0)</td><td>[49]</td></tr><tr><td> $\eta _ { A }$ </td><td>expU(log 0.05, log 0.5)</td><td>[49]</td></tr><tr><td> $\alpha _ { \mathrm { e n v } }$ </td><td> $\mathcal { U } ( 0 . 1 , 1 . 0 )$ </td><td>[49]</td></tr><tr><td> $\mu _ { w }$ </td><td> $\exp \mathcal { U } ( \log 0 . 1 , \log 0 . 5 )$ </td><td>[49]</td></tr></table>

Table 3: Mechanism-specific priors. All $R _ { 0 }$ draws are floored at 0.3. Transmission constants $\beta$ and $\delta$ are back-calculated from $R _ { 0 }$

## Real-world training data

We assembled a multi-source epidemiological surveillance dataset spanning eight pathogen surveillance systems across four continents. Thai provincial surveillance data (2003–2019) provided weekly case counts for 20 notifiable diseases across $7 7$ provinces, including enteric, vectorborne, respiratory, and bloodborne pathogens. Historical US disease surveillance was sourced from Project Tycho Level 2 [24], yielding 39 diseases with state- and regional-level weekly case counts spanning 1888–2014, with disease-specific active windows defined by years exceeding 5% of peak annual incidence. International dengue surveillance was drawn from the Open Dengue dataset [28], comprising 59 country-level series and two sub-national datasets (Nicaragua, $L = 1 8$ provinces; Thailand, L = 78 provinces), with monthly data interpolated to weekly resolution via smoothing spline. US notifiable disease surveillance (2022-2026) was obtained from the National Notifiable Diseases Surveillance System (NNDSS), contributing 54 disease-location combinations at regional and national levels. Influenza-like illness (ILI) and related respiratory surveillance data for 148 countries were obtained from WHO FluNet (1996–2026), providing up to five simultaneous signals per country (ILI, SARI, ARI, pneumonia cases, SARI deaths). Mpox case and death counts were obtained from WHO situation reports (2022–2026) for 33 countries and six WHO regional aggregates. Ebola virus disease data for the 2018–2020 DRC outbreak were sourced from the Humanitarian Data Exchange, aggregated from daily health-zone reports to weekly provincial counts. Cholera surveillance data were obtained from two sources: WHO AFRO weekly reports (2023–2026, 10 countries, cases and deaths) and a JHU-compiled dataset (2010–2023, 32 countries, cases only). US influenza and RSV multivariate surveillance series were constructed by joining five Delphi Epidata signals (wILI, percent positivity, hospitalizations, deaths, and ED visit percentage) at the state level (2010–2026).

All datasets were standardized to a common blinded format: a two-dimensional array of shape $( T \times L , \ 9 )$ where T denotes the number of weekly timesteps, L the number of spatial locations, and the 9 columns encode (scenario index, sin ω<sub>t</sub>, cos ω<sub>t</sub>, $v _ { 1 } , \ldots , v _ { 6 } )$ . The cyclical week-of-year encoding is defined as sin(2πw/52.18) and $\cos ( 2 \pi w / 5 2 . 1 8 )$ , where w is the ISO week number. The primary signal $( v _ { 1 } )$ is always case counts; secondary signals $\left( v _ { 2 } – v _ { 6 } \right)$ encode deaths or additional surveillance streams where available, and −1 otherwise. Explicit reported zeros are preserved as 0; gaps of four weeks or fewer within a reporting window are filled by cubic spline interpolation; gaps exceeding four weeks and periods outside a location’s active reporting window are assigned the value −1.

## Evaluation Data

The evaluation set comprises two components. The in-domain evaluation set consists of 24 disease-location combinations drawn from the real-world training set and held out temporally (208 week evaluation window).

The out-of-domain evaluation set comprises 12 disease-location combinations that include COVID-19, two healthcare-associated infection series from NNDSS, dengue in Ecuador, meningitis and smallpox from Project Tycho, influenza ILI in Austria, measles in Thailand, mpox in Sierra Leone, ebola in the DRC, and cholera in Ethiopia. Out-of-domain series test zero-shot generalization to novel pathogens, geographies, and surveillance modalities.

## C Model architecture

## Chronos-2 backbone

Mantis 2 uses the Chronos-2 transformer backbone [9], initialized from the publicly released weights. The model contains twelve transformer layers with a hidden dimension of 768 and approximately 121 million parameters, including the latent feedback module described below. We train the full model on infectious disease trajectories.

Chronos-2 combines temporal attention within individual series with group attention across related series at corresponding patch positions. In Mantis 2, each forecasting task forms a group containing the selected disease variables and two calendar covariates. Attention shares information within each group, while separate forecasting tasks in the same batch remain isolated (i.e., each group is a multivariate forecasting task).

## Input representation and forecast outputs

The model receives up to 512 weeks of historical observations for between one and six disease variables. Two additional series encode the annual calendar cycle using sine and cosine covariates. These covariates are supplied over both the historical context and the sixteen-week forecast horizon, allowing the model to condition on the seasonal position of future observations. Future disease observations are not supplied as inputs.

Disease variables are transformed using the inverse hyperbolic sine function, before being passed to the model’s instance-normalization. This transformation compresses large values while remaining approximately linear near zero. Calendar covariates are supplied without this external transformation.

Historical inputs are divided into non-overlapping patches of sixteen weeks. A full 512-week context therefore occupies 32 historical patch positions per series. Each patch embedding represents its values together with temporal information. The sixteen-week forecast is represented by one future patch per series. Disease-target values in this patch are unknown, whereas the calendar covariate values are available. The output head predicts 21 quantiles at each of the sixteen forecast weeks directly in one shot (i.e., without autoregressively feeding predicted weekly values back into the model). The model then reverses its internal normalization, and the external transformation is inverted using the hyperbolic sine function to return forecasts in the original units.

## Latent feedback recurrence

We augment the backbone with a latent feedback mechanism that allows additional processing before the forecast is produced [50]. The mechanism operates on hidden representations associated with the future patch. It does not generate textual reasoning or intermediate forecasts, and no intermediate reasoning targets are supplied during training (i.e., we don’t tell the model what to do with the intermediate tokens, we only supervise the final prediction).

The model first processes the historical patches and the future patch through the encoder. For each series, the resulting hidden representation at the future-patch position is passed through a feedback MLP. Both the input and hidden widths are 768. The resulting vector is multiplied by a learned scalar gate parameterized through a hyperbolic tangent. The efective gate value is initialized to 0.95.

The gated vector is appended as an additional token after the original input sequence, and the same encoder is run again. A second feedback token is then constructed from the updated representation at the original future-patch position. Both feedback tokens are retained for a third encoder pass. Thus, the final configuration uses two feedback recurrences and three encoder passes in total, with shared encoder and feedback-module weights across passes.

Each recurrent pass receives the original input embeddings together with the accumulated feedback tokens. Feedback tokens are available to the backbone’s temporal and group attention mechanisms. After the final pass, the output head reads the updated representation at the original future-patch position; the feedback tokens themselves are not decoded into forecasts. With a full context, the sequence grows from 34 positions in the initial pass to 36 positions in the final pass: 32 historical patches, one register token, one future patch, and two feedback tokens per series.

Training gradients propagate through all three encoder passes and the feedback construction steps. The feedback module and gate are learned jointly with the pretrained backbone using the final forecasting objective, without a separate loss for intermediate representations.

## D Model training

## Training-example construction

Training uses separate pools of simulated and observational trajectories prepared ahead of time. Each trajectory contains weekly observations for up to six disease variables together with the two calendar covariates.

A training task is constructed by first selecting the data source (synthetic or real) according to the training schedule and then sampling a trajectory uniformly from that source. For a trajectory of length T, a forecast origin t is sampled uniformly from the eligible positions satisfying $8 \leq t \leq T - 1 6$ . The historical input contains the observations preceding t, truncated to the most recent 512 weeks. Shorter histories are right-aligned and padded on the left with missing values. The target consists of the next sixteen weekly observations.

For each sampled origin, we identify disease variables with at least one observed value in the context window. We sample the number of included variables uniformly between one and the number available, then select that many variables without replacement. All selected variables are forecast targets during training. This procedure exposes the model to diferent combinations of surveillance signals and to both univariate and multivariate tasks. The two calendar covariates are included in every task, with their future values supplied as known inputs.

Forecast origins and variable subsets are resampled throughout training. Consequently, training repeatedly draws diferent tasks from the available trajectories rather than making a single ordered pass through a fixed collection of forecasting windows.

## Simulation-to-observation curriculum

The schedule begins with 70% simulated training and ends with entirely observational training. Its purpose is to expose the model initially to a broad distribution of mechanistically plausible epidemic trajectories and progressively emphasize the statistical characteristics of real surveillance data. Averaged over the training schedule, the expected simulated fraction is 35% of sampled tasks.

## Forecasting objective and optimization

The model is trained using the quantile loss (also called pinball loss) which averages over all quantiles τ :

$$
L _ { \tau } ( y , \hat { y } ) = { \left\{ \begin{array} { l l } { \tau ( y - \hat { y } ) } & { { \mathrm { i f ~ } } y \geq \hat { y } } \\ { ( \tau - 1 ) ( y - \hat { y } ) } & { { \mathrm { i f ~ } } y < \hat { y } } \end{array} \right. }
$$

The backbone applies its loss computation to normalized targets and predictions across forecast weeks and quantile levels. The latent feedback mechanism is optimized through this same final forecast loss.

Optimization uses AdamW [51] with a peak learning rate of $3 \times 1 0 ^ { - 6 }$ and weight decay of 0.01. The learning rate increases during the first 500 optimizer steps and then follows a cosine decay over the remainder of the 8,000-step run. Gradients are clipped to a maximum global norm of 1.0 before each optimizer update. The model is trained on a single NVIDIA RTX PRO 6000 Blackwell Server Edition GPU.

## E Model evaluation

## Evaluation datasets and forecast origins

We evaluated forecasts on disease-location datasets containing one or more surveillance trajectories. For datasets represented during Mantis 2 training, the final 4 years of each series was reserved for evaluation. For out-of-domain evaluation, entire disease-location datasets were excluded from Mantis 2 training. Data sources and preprocessing are described in the data section. Evaluation used rolling forecast origins separated by four weeks. Forecast windows therefore overlap.

All received up to the most recent 512 weeks of context. Shorter histories were right-aligned and padded, with missingness represented through the model-specific input procedure. A variable was eligible for evaluation at an origin only if it contained at least one finite observation within this context window.

Designated surveillance variables, such as cases or deaths, were scored as forecast targets. Columns that contained useful data that was not infectious disease outcomes (e.g., weather) were supplied as historical inputs to the multivariate models but excluded from scoring. Calendar sine and cosine covariates were supplied over both the historical context and forecast horizon. Future surveillance observations were used only for scoring.

## Pretrained forecasting models

Mantis 2 was evaluated using its final training checkpoint, without any task-specific parameter updates.   
Chronos-2 was evaluated using the publicly released pretrained weights, also without task-specific fitting.

## Persistence baseline

One baseline we evaluated was a naive persistence baseline [5]. The persistence baseline used the most recent observed target value as its median forecast at every lead time. Uncertainty was generated using a symmetric random walk based on historical changes, following the construction of the COVIDhub-baseline [5].

## Seasonal-naive baseline

The seasonal-naive baseline tests whether a forecasting model improves on simply repeating the previous year’s disease activity. This provides a useful reference for diseases with recurring seasonal patterns: a model may anticipate seasonal rises and declines without learning anything beyond their annual recurrence.

For each forecast week, the baseline uses the observation from 52 weeks earlier as its central prediction. Forecast uncertainty is estimated from historical diferences between observations separated by 52 weeks. Larger year-to-year diferences produce wider prediction intervals, reflecting how reliably the previous year’s trajectory predicts subsequent activity.

When the corresponding observation from the previous year is unavailable, the baseline cannot generate a forecast for that week. We exclude the seasonal-naive baseline from an origin–horizon comparison if it cannot forecast every target value included in that comparison, rather than substituting a diferent forecasting method.

## Exponential smoothing (ETS)

Exponential smoothing provides a statistical baseline that forecasts from estimates of the current disease activity level, its trend, and, when suficient history is available, its seasonal pattern. These estimates are updated by weighting recent observations more heavily than older observations. We fitted a separate model to each trajectory and scored target using all target history preceding the fitting origin in log(1 + y) space.

We used a hierarchy of model configurations to accommodate diferences in the amount of available history and to provide forecasts when more complex models could not be fitted. With at least 104 observations, corresponding to two annual cycles, we first fitted a model with additive seasonality of period 52 and a damped additive trend. The seasonal component captures recurring annual variation, while damping progressively reduces the contribution of the current trend at longer forecast horizons.

When fewer than two annual cycles were available, or fitting the seasonal model raised an error, we used a nonseasonal model with a damped additive trend, provided at least ten observations were available. If this configuration was unavailable or fitting raised an error, we tried an undamped additive trend with at least four observations, followed by simple exponential smoothing, which models the activity level without an explicit trend or seasonal component. We used the first configuration that fitted without an error. This hierarchy provided a consistent fitting procedure across datasets.

To construct probabilistic forecasts, we estimated a residual scale from the root mean squared in-sample residuals and placed Gaussian uncertainty intervals around the fitted forecast trajectory. The uncertainty scale increased in proportion to the square root of the lead time from the fitting origin. Forecast quantiles were floored at zero in transformed space.

## LSTM model

The LSTM baseline was trained separately for each disease-location dataset, pooling training windows across trajectories within that dataset. Its recurrent component contained three LSTM layers with 64 hidden units per layer and dropout of 0.1 between layers.

At each historical time point, the input contained six variable values, two calendar covariates, and six binary indicators of whether the variable values were observed, giving fourteen input features. Missing values and left-padding were represented by zeros, with the observation indicators distinguishing missing target values from observed zeros. Inputs were represented in log(1 + y) space.

The final hidden state of the top LSTM layer was concatenated with the calendar covariates for each forecast week. A shared prediction head, consisting of a 64-unit linear layer, a ReLU activation, and an output layer, produced quantiles for all six variable slots at each lead. Only designated surveillance targets contributed to the training loss.

At each refit, the model was initialized from scratch and trained on pooled historical windows. Candidate windows were ordered by their forecast-origin index, and at most the 2,048 most recent candidates were retained.

Training minimized quantile loss using AdamW with learning rate $1 0 ^ { - 3 }$ and minibatches of 32 windows. Each fit ran for up to 50 epochs, subject to a maximum of 2,000 optimizer updates, with gradient norms clipped at 1.0 (these were necessary to prevent overfitting). At inference, quantiles were sorted to remove crossings and floored at zero.

## Seasonal autoregressive integrated moving average (SARIMA)

Seasonal autoregressive integrated moving average (SARIMA) models provide a statistical baseline that captures relationships between current disease activity, previous observations, and previous prediction errors. Diferencing allows the model to represent changes in activity rather than assuming a constant long-term level, while seasonal terms capture dependence between observations separated by an annual cycle.

We fitted a separate univariate model to each trajectory and scored target. Models used the target’s own history without additional surveillance variables or exogenous covariates. Similar to exponential smoothing, fitting was performed in log(1 + y) space, leading missing values were removed, and remaining gaps were filled within the available fitting history.

With at least 104 observations, corresponding to two annual cycles, we first fitted

$$
\mathrm { S A R I M A ( 1 , 1 , 1 ) ( 1 , 0 , 0 ) _ { 5 2 } } .
$$

This specification includes one nonseasonal autoregressive term, one order of diferencing, one nonseasona moving-average term, and one seasonal autoregressive term at lag 52. It therefore captures both short-term dependence and annual recurrence, without seasonal diferencing or a seasonal moving-average term.

When fewer than 104 observations were available, or the seasonal model could not produce a valid fit and forecast, we used a nonseasonal ARIMA(1, 1, 1) model. If that model also failed, we used the simpler ARIMA(0, 1, 1) specification. This fixed hierarchy allowed the baseline to accommodate short histories and fitting dificulties without conducting a separate order search for every trajectory and forecast origin. Models were fitted without a deterministic trend. Stationarity constraints were imposed on the autoregressive components and invertibility constraints on the moving-average components.

## Refitting schedules and information updates

Forecasts were evaluated every four weeks, but the locally fitted models were refitted every ten forecast horizons to limit the already high computational cost of repeated fitting. The LSTM, ETS, and SARIMA models therefore used fitting origins at weeks 8, 48, 88, and so on. At each fitting origin, the models were fitted using the preceding observations.

The models difered in how they incorporated observations arriving between refits. The LSTM retained its fitted parameters but received updated historical context at every evaluation origin. Its forecasts therefore reflected the latest observations even when its parameters had not recently been retrained.

Persistence and seasonal-naive forecasts were recomputed from the available history at every evaluation origin. Mantis 2 and Chronos-2 likewise received updated context at each origin, while their pretrained parameters remained fixed throughout evaluation.

## Forecast scores and matched comparisons

We evaluated forecast accuracy over three nested horizon ranges: weeks 1-4, 1-8, and 1-16. These ranges measure cumulative performance from the first forecast week through each specified endpoint. We used weighted interval score (WIS) [18] to assess probabilistic forecasts. All models used the same quantile grid for forecasting.

Scores were calculated in log(1 + y) space, which reduces the influence of large absolute diferences in disease activity across surveillance targets. The reported scores measure accuracy on this transformed scale rather than in original case, hospitalization, or death counts. At each forecast origin and horizon range, we identified the finite target observations across the eligible surveillance variables. These observations defined a common scoring set for all models. A model received a score for an origin and horizon range only if it provided finite predictions for every required quantile at every target value in the common scoring set. This prevented a model from receiving a score based only on a subset of the targets evaluated for other models. Whenever two models had results for the same dataset, trajectory, origin, and horizon range, their scores therefore referred to exactly the same target values.

## F Full forecasting results

We report aggregate weighted interval score (WIS) across three cumulative forecast horizon ranges in Table 4, empirical prediction interval coverage in Table 5, and WIS for each evaluation dataset over weeks 1-16 in Table 6. WIS was calculated in log(1 + y) space, with lower values indicating better probabilistic forecast performance. Aggregate scores weight forecasted target values equally, following the procedure described in the model-evaluation section.

## Aggregate probabilistic forecast performance

Mantis 2 achieved the lowest reported aggregate WIS in all three horizon ranges, followed by Chronos-2 and persistence (Table 4). Mantis 2’s aggregate WIS was 0.20913 over weeks 1-4, 0.28439 over weeks 1-8, and 0.36965 over weeks 1-16. The corresponding Chronos-2 scores were 0.23097, 0.31084, and 0.39762. Scores increased over longer horizon ranges for every model, reflecting the greater dificulty of forecasting further into the future.

These summaries include all available forecasts for each model. Forecast availability difered among models, so the aggregate values do not all represent identical evaluation samples. In particular, seasonal-naive forecasts require observations from the corresponding weeks of the preceding year and are unavailable when those observations are absent. These tables provide descriptive summaries of available forecasts; comparisons restricted to common forecast instances follow the matching procedure described in the evaluation methods.

## Prediction interval coverage

Mantis 2’s empirical 95% prediction interval coverage remained close to nominal across the three horizon ranges: 96.14% over weeks 1–4, 95.11% over weeks 1–8, and 94.16% over weeks 1–16 (Table 5). Its central

Table 4: Aggregate weighted interval score (WIS) over three cumulative forecast horizon ranges. Scores are calculated in log(1 + y) space and pooled across available forecasted target values. Lower values indicate better performance. Each model’s summary includes its available forecasts, which need not constitute the same evaluation sample across models.
<table><tr><td>Model</td><td>Weeks 1-4</td><td>Weeks 1-8</td><td>Weeks 1-16</td></tr><tr><td>Mantis 2</td><td>0.20913</td><td>0.28439</td><td>0.36965</td></tr><tr><td>Chronos-2</td><td>0.23097</td><td>0.31084</td><td>0.39762</td></tr><tr><td>Persistence</td><td>0.28046</td><td>0.36928</td><td>0.46776</td></tr><tr><td>LSTM</td><td>0.41975</td><td>0.45859</td><td>0.51038</td></tr><tr><td>Seasonal naive</td><td>0.52003</td><td>0.52281</td><td>0.52770</td></tr><tr><td>SARIMA</td><td>0.59501</td><td>0.62158</td><td>0.65710</td></tr><tr><td>ETS</td><td>0.72205</td><td>0.75606</td><td>0.80579</td></tr></table>

50% prediction intervals showed some undercoverage, however, with empirical coverage decreasing from 48.59% over weeks 1-4 to 45.68% over weeks 1-16. Thus, agreement with nominal coverage for the wider intervals did not extend equally to the central portion of the predictive distribution.

Chronos-2 also showed declining 50% coverage over longer horizon ranges. Persistence and seasonal naive achieved approximately correct 50% coverage but undercovered at the 95% level. SARIMA exhibited the opposite pattern: its 95% coverage was close to nominal, while its 50% coverage was substantially above nominal. ETS and LSTM undercovered at both levels across all three horizon ranges.

Coverage describes how often prediction intervals contain the observations, but does not account for interval width. Consequently, coverage should be interpreted alongside WIS, which evaluates both interval width and penalties for observations falling outside the intervals.

Table 5: Empirical coverage of central 50% and 95% prediction intervals, expressed as percentages. Values closer to 50% and 95%, respectively, indicate closer agreement with nominal coverage. N is the number of forecasted target values scored, and datasets denotes the number of evaluation files contributing results. Values forecast from diferent origins contribute separately to N. Results use all available forecasts for each model; diferences in N indicate diferences in evaluation availability.
<table><tr><td>Horizon</td><td>Model</td><td>50% coverage</td><td>95% coverage</td><td>N</td><td>Datasets</td></tr><tr><td rowspan="7">Weeks 1-4</td><td>Mantis 2</td><td>48.59</td><td>96.14</td><td>164,095</td><td>30</td></tr><tr><td>Chronos-2</td><td>49.14</td><td>96.16</td><td>164,095</td><td>30</td></tr><tr><td>Persistence</td><td>50.72</td><td>90.77</td><td>164,095</td><td>30</td></tr><tr><td>LSTM</td><td>40.59</td><td>85.54</td><td>164,095</td><td>30</td></tr><tr><td>Seasonal naive</td><td>52.63</td><td>90.73</td><td>133,006</td><td>29</td></tr><tr><td>SARIMA</td><td>64.60</td><td>93.94</td><td>158,219</td><td>30</td></tr><tr><td>ETS</td><td>45.63</td><td>73.66</td><td>158,219</td><td>30</td></tr><tr><td rowspan="8">Weeks 1-8</td><td>Mantis 2</td><td>46.65</td><td>95.11</td><td>328,184</td><td>30</td></tr><tr><td>Chronos-2</td><td>46.60</td><td>94.60</td><td>328,184</td><td>30</td></tr><tr><td>Persistence</td><td>50.62</td><td>88.72</td><td>328,184</td><td>30</td></tr><tr><td>LSTM</td><td>38.84</td><td>83.57</td><td>328,184</td><td>30</td></tr><tr><td>Seasonal naive</td><td>52.64</td><td>90.37</td><td>265,781</td><td>29</td></tr><tr><td>SARIMA</td><td>65.45</td><td>94.25</td><td>316,432</td><td>30</td></tr><tr><td>ETS</td><td>45.82</td><td>73.62</td><td>316,432</td><td>30</td></tr><tr><td>Mantis 2</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Weeks 1-16</td><td></td><td>45.68</td><td>94.16</td><td>656,236</td><td>30</td></tr><tr><td>Chronos-2 Persistence</td><td>45.60</td><td>93.78</td><td>656,236</td><td>30</td></tr><tr><td></td><td>52.86</td><td>88.44</td><td>656,236</td><td>30</td></tr><tr><td>LSTM</td><td>36.90</td><td>81.19</td><td>656,236</td><td>30</td></tr><tr><td rowspan="3"></td><td>Seasonal naive</td><td>52.50</td><td>89.92</td><td>530,744</td><td>29</td></tr><tr><td>SARIMA</td><td>67.65</td><td>94.87</td><td>632,845</td><td>30</td></tr><tr><td>ETS</td><td>46.94</td><td>74.41</td><td>632,845</td><td>30</td></tr></table>

## Performance across individual datasets

The dataset-level results show variation in the relative performance of the forecasting methods (Table 6). Mantis 2 had lower reported WIS than Chronos-2 on 26 of the 30 datasets. The four exceptions were Rocky Mountain spotted fever, hepatitis B from Project Tycho, salmonellosis, and hepatitis A across U.S. states. Improvements over Chronos-2 were particularly pronounced for carbapenem-resistant Enterobacteriaceae and meningitis.

Persistence had lower reported WIS than both foundation models on COVID-19 and dengue in Nicaragua and Singapore. These results illustrate that the aggregate advantage of Mantis 2 does not imply an advantage on every individual dataset. However, for COVID-19, there were data quality issues (changing definitions for variables, high amounts of missing data) that substantially harmed the performance of the models).

Table 6: Weighted interval score (WIS) for each evaluation dataset over weeks 1–16. Scores are calculated in log(1 + y) space and pooled over available forecasted target values within each dataset; lower values indicate better performance. Seasonal denotes the seasonal-naive baseline. Model-specific forecast availability can difer within a dataset, so the entries are summaries of available forecasts rather than necessarily matched comparisons.
<table><tr><td>Dataset</td><td>Mantis 2</td><td>Chronos-2</td><td>Persistence</td><td>LSTM</td><td>Seasonal</td><td>SARIMA</td><td>ETS</td></tr><tr><td>COVID-19</td><td>0.70492</td><td>0.70515</td><td>0.65241</td><td>1.11491</td><td>0.76370</td><td></td><td>0.73894 0.96699</td></tr><tr><td>Dengue-Ecuador</td><td>0.28054</td><td>0.28172</td><td>0.39999</td><td>0.54665</td><td>0.51285</td><td>0.40595</td><td>0.41814</td></tr><tr><td>Dengue-Indonesia</td><td>0.21717</td><td>0.22298</td><td>0.22676</td><td>0.46587</td><td>0.45524</td><td></td><td>0.18264 0.20858</td></tr><tr><td>Dengue-Nicaragua</td><td>0.41206</td><td>0.42872</td><td>0.40332</td><td>0.56666</td><td>0.65005</td><td>0.41656</td><td>0.83064</td></tr><tr><td>Dengue-Singapore</td><td>0.30825</td><td>0.31734</td><td>0.29664</td><td>0.56918</td><td>0.81328</td><td>0.31197</td><td>0.68681</td></tr><tr><td>Dengue-Thailand</td><td>0.44336</td><td>0.52093</td><td>0.58610</td><td>0.61893</td><td>0.87349</td><td>0.57618</td><td>0.80000</td></tr><tr><td>Influenza-Austria</td><td>0.53134</td><td>0.57315</td><td>0.83065</td><td>0.79190</td><td>0.78175</td><td>0.77091</td><td>0.91781</td></tr><tr><td>Influenza-Nigeria</td><td>0.39054</td><td>0.41663</td><td>0.51245</td><td>0.41993</td><td>0.54459</td><td>0.39882</td><td>0.87472</td></tr><tr><td>C. auris</td><td>0.35850</td><td>0.42594</td><td>0.58217</td><td>0.62885</td><td>0.60519</td><td>0.41175</td><td>0.56139</td></tr><tr><td>Carbapenem-resistant Enterobacteriaceae</td><td>0.38580</td><td>0.72789</td><td>0.70474</td><td>0.72491</td><td>0.71342</td><td></td><td>0.56367 1.22310</td></tr><tr><td>Acute diarrhea</td><td>0.10257</td><td>0.11815</td><td>0.16467</td><td>0.18041</td><td>0.18429</td><td>0.21636</td><td>0.23275</td></tr><tr><td>Hand, foot and mouth disease</td><td>0.37564</td><td>0.41699</td><td>0.51941</td><td>0.50039</td><td>0.62843</td><td></td><td>0.494840.57119</td></tr><tr><td>Hepatitis B-Thailand</td><td>0.24038</td><td>0.26376</td><td>0.30042</td><td>0.29091</td><td>0.36324</td><td></td><td>0.32481 0.47011</td></tr><tr><td>Influenza-Thailand</td><td>0.33909</td><td>0.38604</td><td>0.49060</td><td>0.49255</td><td>0.57972</td><td></td><td>0.48959 0.59103</td></tr><tr><td>Tuberculosis-Thailand</td><td>0.23959</td><td>0.27683</td><td>0.32609</td><td>0.32507</td><td>0.47704</td><td></td><td>0.344160.46552</td></tr><tr><td>Enteric fever-Thailand</td><td>0.21153</td><td>0.23184</td><td>0.28508</td><td>0.36154</td><td>0.32424</td><td>0.33351</td><td>0.40330</td></tr><tr><td>Hepatitis A-Thailand</td><td>0.23734</td><td>0.24533</td><td>0.26784</td><td>0.31329</td><td>0.34513</td><td>0.28079</td><td>0.41864</td></tr><tr><td>Measles-Thailand</td><td>0.20972</td><td>0.23775</td><td>0.26850</td><td>0.27940</td><td>0.45435</td><td>0.29050</td><td>0.39180</td></tr><tr><td>Rocky Mountain spotted</td><td>0.49137</td><td>0.48797</td><td>0.68073</td><td>0.51322</td><td>0.55134</td><td>0.56363</td><td>0.65229</td></tr><tr><td>fever Brucellosis</td><td>0.26475</td><td>0.28173</td><td>0.39602</td><td>0.32074</td><td>0.33971</td><td>0.28188 0.38612</td><td></td></tr><tr><td>Hepatitis B-U.S. regions</td><td>0.16503</td><td>0.15991</td><td>0.31458</td><td>0.27840</td><td>0.24020</td><td>0.16469</td><td>0.29659</td></tr><tr><td>Malaria</td><td>0.31723</td><td>0.32845</td><td>0.46807</td><td>0.33991</td><td>0.38766</td><td>0.33315</td><td>0.54008</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dataset</td><td>Mantis 2</td><td>Chronos-2 Persistence</td><td></td><td></td><td>LSTM Seasonal</td><td>SARIMA</td><td>ETS</td></tr><tr><td>Meningitis</td><td>0.37116</td><td>0.52148</td><td>0.56737</td><td>0.42915</td><td>0.47907</td><td>0.44849</td><td>1.11943</td></tr><tr><td>Salmonellosis</td><td>0.24097</td><td>0.23614</td><td>0.37260</td><td>0.38237</td><td>0.25999</td><td>0.33638</td><td>0.45154</td></tr><tr><td>Smallpox</td><td>0.32266</td><td>0.34331</td><td>0.54725</td><td>0.46356</td><td>0.48240</td><td>0.46900</td><td>0.72112</td></tr><tr><td>Typhoid fever</td><td>0.28638</td><td>0.29434</td><td>0.48861</td><td>0.34233</td><td>0.35606</td><td>0.33680</td><td>0.52707</td></tr><tr><td>Pertussis</td><td>0.38346</td><td>0.39172</td><td>0.52775</td><td>0.45483</td><td>0.61361</td><td>0.39053</td><td>0.53248</td></tr><tr><td>Hepatitis A-U.S. states</td><td>0.37582</td><td>0.37016</td><td>0.54197</td><td>0.39382</td><td>0.49754</td><td>0.39038</td><td>0.60722</td></tr><tr><td>Measles-U.S. states</td><td>0.45860</td><td>0.49354</td><td>0.59108</td><td>0.50994</td><td>0.66532</td><td>0.56298</td><td>0.76852</td></tr><tr><td>Tuberculosis-U.S. states</td><td>0.36580</td><td>0.38778</td><td>0.63001</td><td>0.39505</td><td>0.50698</td><td>0.42209</td><td>0.76121</td></tr></table>

## G Comparison with the original Mantis

Mantis 2 expands the forecasting capabilities of the original Mantis, but an important additional question is whether it improves accuracy on tasks its predecessor already supports. The original Mantis forecasts a single outcome up to eight weeks ahead, optionally using one additional covariate [15]. It therefore cannot participate directly in the full sixteen-week, joint multivariate evaluation used in the main analysis. We report a separate comparison restricted to single-target forecasting over eight weeks, allowing us to assess improvements within the original model’s forecasting scope.

## Comparison design

We included evaluation datasets containing exactly one populated surveillance target. This restriction yielded 27 eligible datasets: the COVID-19 dataset and the influenza datasets from Austria and Nigeria were excluded because they contained multiple surveillance targets. Auxiliary variables could still be supplied as inputs, but only the designated surveillance target was scored.

Both models were evaluated at the same forecast origins, beginning after eight weeks of observations and advancing in four-week increments. We retained the main evaluation’s requirement that sixteen subsequent weeks be available, preserving its forecast origins, but scored only weeks one through eight. Original Mantis generated an eight-week forecast directly; Mantis 2 generated its usual sixteen-week forecast, from which the first eight weeks were extracted.

Original Mantis used its released eight-week checkpoints. At each origin, the covariate-aware checkpoint was used when an auxiliary variable was available within its historical context; otherwise, the covariate-free checkpoint was used. When several auxiliary variables were available, the first eligible variable in the dataset’s column order was selected, without selection based on forecast performance. Original Mantis received up to 112 weeks of history.

Mantis 2 received up to 512 weeks of history, all available auxiliary variables, and its usual calendar covariates. Neither model underwent task-specific retraining. Thus, the comparison matched forecast targets, origins, and horizons while retaining each model’s supported input configuration. It evaluates the overall improvement ofered by Mantis 2, including its ability to use additional historical and auxiliary information, rather than isolating an architectural change under identical inputs.

Percentage improvement was defined as

$$
1 0 0 \times \frac { \mathrm { W I S _ { M a n t i s } - W I S _ { M a n t i s 2 } } } { \mathrm { W I S _ { M a n t i s } } } .
$$

Positive values indicate lower forecast error for Mantis 2.

## Results

Across 34,762 matched forecast origins, Mantis 2 reduced pooled WIS from 0.351921 to 0.241338, an improvement of 31.42%. Both models returned valid forecasts for every evaluated pair. Mantis 2 achieved lower WIS on all 27 eligible datasets, with dataset-level improvements ranging from 10.37% to 45.55% (Table 7).

Improvements extended across diseases, locations, and surveillance settings. WIS decreased by 45.55% for historical U.S. brucellosis, 44.66% for acute diarrhea in Thailand, and 43.27% for measles across U.S. states. All five dengue datasets improved, with reductions ranging from 10.55% in Nicaragua to 39.91% in Indonesia. Gains occurred both in datasets without auxiliary variables and in those where original Mantis used its covariate-aware checkpoint.

These results show that Mantis 2’s advances extend beyond supporting longer horizons and jointly forecasting multiple outcomes: it also substantially improves probabilistic accuracy within the eight-week, singletarget setting supported by its predecessor. Because the models difer in training data, architecture, historica context, and auxiliary inputs, this comparison does not attribute the improvement to any one component.

Table 7: Comparison with the original Mantis over forecast weeks 1–8. WIS is calculated in log(1 + y) space using nine common quantiles and matched forecast targets and origins. Lower WIS is better. Improvement is the percentage reduction in WIS relative to original Mantis. Models retain their respective context lengths and auxiliary-input configurations.
<table><tr><td>Dataset</td><td>Mantis</td><td>Mantis 2</td><td>Improvement (%)</td></tr><tr><td>Dengue—Ecuador</td><td>0.24591</td><td>0.19890</td><td>19.12</td></tr><tr><td>Dengue—Indonesia</td><td>0.17306</td><td>0.10400</td><td>39.91</td></tr><tr><td>Dengue—Nicaragua</td><td>0.34377</td><td>0.30749</td><td>10.55</td></tr><tr><td>Dengue—Singapore</td><td>0.24938</td><td>0.20207</td><td>18.97</td></tr><tr><td>Dengue—Thailand</td><td>0.42178</td><td>0.28882</td><td>31.52</td></tr><tr><td>Candida auris—U.S.</td><td>0.35623</td><td>0.31928</td><td>10.37</td></tr><tr><td>Carbapenem-resistant Enterobacteriaceae—U.S.</td><td>0.58237</td><td>0.36914</td><td>36.61</td></tr><tr><td>Acute diarrhea—Thailand</td><td>0.12314</td><td>0.06815</td><td>44.66</td></tr><tr><td>Hand, foot, and mouth disease—Thailand</td><td>0.34725</td><td>0.26642</td><td>23.28</td></tr><tr><td>Hepatitis B—Thailand</td><td>0.28187</td><td>0.19936</td><td>29.27</td></tr><tr><td>Influenza—Thailand</td><td>0.32083</td><td>0.23502</td><td>26.75</td></tr><tr><td>Tuberculosis—Thailand</td><td>0.26999</td><td>0.17999</td><td>33.33</td></tr><tr><td>Enteric fever—Thailand</td><td>0.22731</td><td>0.16108</td><td>29.14</td></tr><tr><td>Hepatitis A—Thailand</td><td>0.24302</td><td>0.20300</td><td>16.47</td></tr><tr><td>Measles—Thailand</td><td>0.21477</td><td>0.13684</td><td>36.28</td></tr><tr><td>Rocky Mountain spotted fever—U.S.</td><td>0.49390</td><td>0.42352</td><td>14.25</td></tr><tr><td>Brucellosis—U.S.</td><td>0.46422</td><td>0.25276</td><td>45.55</td></tr><tr><td>Hepatitis B—U.S.</td><td>0.20465</td><td>0.15700</td><td>23.28</td></tr><tr><td>Malaria—U.S.</td><td>0.42480</td><td>0.29652</td><td>30.20</td></tr><tr><td>Meningitis—U.S.</td><td>0.39900</td><td>0.32117</td><td>19.51</td></tr><tr><td>Salmonellosis—U.S.</td><td>0.24642</td><td>0.21008</td><td>14.75</td></tr><tr><td>Smallpox—U.S.</td><td>0.37870</td><td>0.28199</td><td>25.54</td></tr><tr><td>Typhoid fever—U.S.</td><td>0.33103</td><td>0.26223</td><td>20.78</td></tr><tr><td>Pertussis—U.S.</td><td>0.38222</td><td>0.33128</td><td>13.33</td></tr><tr><td>Hepatitis A—U.S. states</td><td>0.51480</td><td>0.34810</td><td>32.38</td></tr><tr><td>Measles—U.S. states</td><td>0.69481</td><td>0.39416</td><td>43.27</td></tr><tr><td>Tuberculosis—U.S. states</td><td>0.51103</td><td>0.34793</td><td>31.92</td></tr><tr><td>Pooled</td><td>0.35192</td><td>0.24134</td><td>31.42</td></tr></table>

## H Training-mixture and surveillance-history experiments

We conducted two complementary experiments to examine how the contribution of simulated training depends on the forecasting setting and the amount of available surveillance history. First, we varied the proportion of simulated examples during training while holding the remaining training settings fixed. Sec-

ond, we evaluated each trained model using diferent amounts of historical context at the same forecast origins. These experiments evaluated both how overall performance depends on the quantity of synthetic training as well as how that varies based on the context length.

## Constant-mixture training

We trained eleven models with synthetic sampling probabilities ranging from 0 to 1 in increments of 0.1. For a model with synthetic fraction p, each training task was independently assigned to the simulated data pool with probability p and to the observational data pool with probability $1 - p \colon p \in \{ 0 , 0 . 1 , 0 . 2 , \ldots , 0 . 9 , 1 . 0 \}$ The selected trajectory and forecasting task were then sampled using the procedure described in the modeltraining section. Thus, the mixture fraction specifies the probability of selecting a data source for each task, rather than an exact fraction of observations within every batch.

The endpoints correspond to exclusively observational and exclusively simulated task sampling during infectious-disease training. All models retained the same pretrained initialization. Architecture, optimization settings, training duration, data pools, and all other training settings were held fixed; only the synthetic sampling probability changed. This design tests how allocating a fixed training budget between simulated and observational examples afects subsequent forecast performance.

Unlike the curriculum used for the released Mantis 2 model, these sampling probabilities remained constant throughout training. The constant-mixture experiments therefore characterize the efect of trainingdata composition without simultaneously changing the order in which the two sources are emphasized.

## Evaluation across datasets

All eleven models were evaluated on all evaluation datasets using the forecast origins, inputs, and scoring procedures described in the model-evaluation section. For these experiments, we summarized WIS over the complete sixteen-week forecast horizon. Using a single horizon range simplifies comparisons across training mixtures and context lengths.

The dataset-by-mixture heatmap (Figure 4) displays the twelve Project Tycho datasets. This restriction is for visualization only; the broader analysis includes all evaluation datasets. Displaying the Project Tycho subset keeps the heatmap readable while comparing historical U.S. disease records drawn from the same data source and broadly similar surveillance settings.

To show the relative efect of training mixture within each dataset, heatmap scores were normalized by the lowest WIS attained across the eleven mixtures. For dataset d and synthetic sampling probability p, the displayed value is

$$
R _ { d , p } = \frac { \mathrm { W I S } _ { d , p } } { \operatorname* { m i n } _ { p ^ { \prime } \in \{ 0 , 0 . 1 , . . . , 1 . 0 \} } \mathrm { W I S } _ { d , p ^ { \prime } } } .
$$

A value of 1.00 therefore denotes the best-performing mixture for that dataset, and larger values indicate worse performance relative to that minimum. This normalization makes diferences among mixtures visible without allowing diferences in the absolute WIS scale across datasets to mess up the heatmap.

## Controlled variation in surveillance history

To test whether the value of simulated training depends on the amount of historical data for a particular dataset, we evaluated every mixture model with context lengths of 8, 16, 26, 52, and 104 weeks. Eight weeks represents the minimum context considered, approximately two months. Sixteen weeks provides an intermediate context of approximately four months, while 26, 52, and 104 weeks correspond to approximately half a year, one year, and two years.

At each forecast origin, historical inputs were truncated to retain only the most recent L weekly observations, where L is the selected context length. The forecast origin and the subsequent sixteen-week target window were held fixed across context lengths and training mixtures. Models were evaluated with their existing weights, without retraining for a particular context length.

Holding forecast origins fixed ensures that comparisons concern the same future observations and outbreak phases. Consequently, diferences between context-length conditions reflect the efect of changing the historical information available to the model, rather than evaluating short histories early in an outbreak and long histories at a diferent stage.

## Improvement relative to observational training

For each context length, mixture-model performance was compared with the exclusively observational training condition evaluated on the same forecast instances. Percentage improvement was defined as

$$
I _ { p , L } = 1 0 0 \times \frac { \mathrm { W I S } _ { 0 , L } - \mathrm { W I S } _ { p , L } } { \mathrm { W I S } _ { 0 , L } } ,
$$

where $\mathrm { W I S } _ { 0 , L }$ is the score of the observational-only model and $\mathrm { W I S } _ { p , L }$ is the score of the model trained with synthetic sampling probability $p ,$ both using context length L. Positive values indicate that including simulated training examples improves performance; negative values indicate worse performance than observational-only training.

Comparing these improvements across context lengths tests whether additional surveillance history changes the relative benefit of simulated training. In particular, the comparison between eight and 104 weeks contrasts forecasts informed by only a short recent record with forecasts that can draw on approximately two years of local disease activity.

## I Additional approaches considered

We investigated several additional approaches that could potentially improve forecast performance: epimodulation [52], ensembling across subsets of input variables, and joint training of multiple models using negative correlation learning [53]. These experiments informed the design of the released model by examining both forecast accuracy and the computational and implementation costs of additional processing. To be clear: even if we did not add these methods to the final model, we are not trying to criticize their performance overall, just that we did not find them to be the most useful in our case.

## Epimodulation

Epimodulation augments existing forecasts with an adjustment motivated by epidemiological dynamics, particularly susceptible depletion, to improve anticipation of epidemic peaks [52]. It provides a way to incorporate mechanistic information into forecasts without replacing the underlying forecasting model. We hypothesized that this adjustment might ofer limited additional benefit for Mantis 2 because mechanistic simulation already exposes the model to susceptible depletion and related outbreak dynamics during training. Nevertheless, the approach was straightforward to apply and provided a useful test of whether an explicit adjustment could complement the dynamics learned through simulation. In our experiments, applying epimodulation to Mantis 2 did not improve performance and slightly increased forecast error.

To examine whether the adjustment would be more useful without infectious-disease simulation training, we also applied it to the model trained exclusively on observational disease data. This comparison was motivated by the improvements reported for empirical forecasting methods in the original epimodulation paper [52]. Adjustment parameters were calibrated using only previously observed outcomes. Base and adjusted forecasts were evaluated on matched forecast instances, with identical nonnegativity and quantilesorting procedures applied to both.

Epimodulation did not improve aggregate performance for the observational-only model (Table 8). Pooled WIS increased by 1.06%, 0.96%, and 0.66% over weeks 1–4, 1–8, and 1–16, respectively (lower is better). At the sixteen-week horizon range, WIS improved on four of the thirty datasets, was unchanged on one, and worsened on the remaining twenty-five. The median dataset-level change was a 2.81% increase in WIS.

Efects nevertheless varied substantially across datasets. For example, epimodulation reduced sixteenweek WIS by 63.78% for influenza in Austria, while increasing it by 22.41% for dengue in Singapore (for whatever reason, the observation-only model performed very poorly on this Austrian flu data compared to all other models, including the final model, which has a WIS improvement of over 70% relative to observation only). These results indicate that the adjustment could be helpful in particular settings, but did not provide a consistent improvement across our evaluation.

The absence of an aggregate benefit for the observational-only model also means that redundancy with simulation training cannot, by itself, explain these findings. Diferences in the underlying forecast model, training procedure, forecasting targets, horizons, and calibration procedure may afect the usefulness of the adjustment (the most probably explanation being that our models were trained on vastly more infectious disease data than the models in their paper). We therefore interpret these experiments as evidence against including our tested epimodulation procedure in the default Mantis 2 pipeline, rather than as evidence against epimodulation generally.

Table 8: Epimodulation applied to the observational-only model. WIS was calculated in log(1 + y) space on matched forecast instances and pooled using the number of scored target values as weights. Positive percentage improvement favors epimodulation; negative values indicate increased WIS.
<table><tr><td>Horizon range</td><td>Base WIS</td><td>Adjusted WIS</td><td>Improvement (%)</td></tr><tr><td>Weeks 1-4</td><td>0.213999</td><td>0.216261</td><td>-1.06</td></tr><tr><td>Weeks 1-8</td><td>0.295455</td><td>0.298283</td><td>-0.96</td></tr><tr><td>Weeks 1-16</td><td>0.392063</td><td>0.394660</td><td>-0.66</td></tr></table>

## Ensembling across input-variable subsets

We investigated whether a single trained Mantis 2 model could produce more accurate forecasts by combining predictions generated from diferent subsets of the available input variables. This experiment was motivated by our prior work emphasizing that infectious disease ensembles benefit from complementary predictive information, while correlated errors can limit the gains from combining forecasts [54]. Varying the input subsets provides a way to change the information on which each forecast is conditioned while retaining the same trained model. We hypothesized that forecasts based on diferent combinations of surveillance signals might make suficiently diferent errors to improve accuracy when combined.

For each eligible multivariate forecasting task, we generated the default forecast using all available variables and additional forecasts using subsets of those variables. We evaluated two ensemble variants: one using all nonempty subsets, including univariate inputs, and one restricted to subsets containing at least two variables (since univariate forecasts are often less accurate [15]).

For each target, corresponding forecast quantiles were averaged with equal weight across the included subsets containing that target. Averaging was performed in log(1 + y) space. All forecasts used the same trained weights, and no additional training was required. Comparisons with the default forecast used matched origins and target observations within the eligible multivariate tasks.

The improvements were small (Table 9). Ensembling all nonempty subsets reduced pooled WIS by 0.18% over weeks 1–16, but slightly increased WIS over weeks 1–4 and 1–8. Excluding univariate subsets produced improvements across all three horizon ranges, reducing pooled WIS by 0.15%, 0.14%, and 0.27%, respectively.

Although this approach required only one set of model weights, it increased the number of forecast configurations evaluated at inference. With six available variables, there are 63 nonempty subsets, or 57 subsets when univariate inputs are excluded. These configurations can be batched, but still require substantially more inference computation than a single forecast using all variables. Given the modest aggregate gains, we retained the full-input forecast as the default to support rapid and straightforward deployment.

Table 9: Relative improvement in pooled WIS from ensembling predictions across variable subsets, compared with a single forecast using all available variables. Results are restricted to eligible multivariate tasks and use matched forecast instances. Positive values indicate lower WIS than the default forecast.
<table><tr><td>Ensemble</td><td>Weeks 1-4</td><td>Weeks 1-8</td><td>Weeks 1-16</td></tr><tr><td>All nonempty subsets</td><td>-0.03%</td><td>-0.07%</td><td>0.18%</td></tr><tr><td>Subsets with at least two variables</td><td>0.15%</td><td>0.14%</td><td>0.27%</td></tr></table>

## Negative correlation learning

We also investigated ensembles of multiple Mantis models trained with objectives that encouraged complementary prediction errors, following the motivation of negative correlation learning [53]. Averaging models with similar errors provides limited benefit; encouraging less correlated or negatively correlated errors during training may allow individual models to compensate for one another when their predictions are combined.

These experiments produced forecast improvements of several percent, but required training and deploying multiple models. Compared with a single model, this increased training cost, model storage requirements, and inference computation, as well as the complexity of maintaining the forecasting pipeline.

We judged these gains insuficient to justify the additional requirements for the default release, whose design emphasizes accessibility and routine use on modest computing resources. This is a deployment tradeof rather than a conclusion that ensemble training is inefective: applications with greater computational resources or a stronger preference for incremental accuracy may reasonably favor an ensemble.