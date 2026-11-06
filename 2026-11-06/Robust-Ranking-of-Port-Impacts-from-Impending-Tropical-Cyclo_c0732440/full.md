# Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System

Chenchen Kuai<sup>a</sup>, Shixuan Tang<sup>b</sup>, Chaolun Ma<sup>a</sup>, Xiaoqiang Kong<sup>c</sup>, Xiubin Bruce Wang<sup>a</sup>, Yunlong Zhang<sup>a</sup>, Zihao Li<sup>b,∗</sup>

<sup>a</sup>Zachry Department of Civil & Environmental Engineering, Texas A&M University, 3136 TAMU, College Station, TX, 77843, USA

<sup>b</sup>Department of Civil, Construction and Environmental Engineering, Marquette University, 1515 West Wisconsin Avenue, Milwaukee, WI, 53233, USA

<sup>c</sup>Texas A&M Transportation Institute, 505 East Huntland Drive, Austin, TX, USA

## Abstract

When a tropical cyclone (TC) threatens several ports, preparedness planning depends on anticipating which ports are likely to be afected and where disruptions may last longest. Limited historical observations at individual ports and uncertainty in landfall location make these relative impacts dificult to estimate. We use a two-stage recommendation framework that first selects candidate ports based on their proximity to the expected landfall location and then ranks them by their estimated number of impact days. Using AIS-derived impact days for 37 ports and 31 TCs in the Gulf of Mexico study region, we compare three ways of representing port–cyclone interactions: pairwise feature interactions, tree-based feature combinations, and explicit feature crosses combined with learned embeddings. In tests on Hurricanes Nicole and Ian, the retrieval stage consistently identifies all observed afected ports under landfall uncertainty. Among the ranking models, Wide & Deep Learning achieves the closest agreement with the observed port rankings. Its mean ranking agreement exceeds that of the distance-based baseline by 85.5% for Nicole and 55.5% for Ian and is less sensitive to uncertainty in the landfall location. These rankings could help port authorities prioritize post-cyclone inspections and allocate recovery resources across ports threatened by the same event.

Keywords: Port resilience, Tropical cyclone, Disaster preparedness, Recommendation system, Deep Learning

## 1. Introduction

Maritime transportation plays a central role in U.S. international trade. In 2016, water transportation accounted for approximately 69% of U.S. imports and exports by weight and 40% by value (BTS, 2017). Ports connect these maritime flows with regional road, rail, and inland waterway networks (Ng et al., 2015). Their coastal locations also expose them to tropical cyclones (TCs), which can damage infrastructure, interrupt port operations, and disrupt cargo movements across these connected networks (Asariotis et al., 2017; Verschuur et al., 2020). During the 2020 Atlantic hurricane season, nearly every port along the Gulf and South Atlantic coasts experienced TC-related operational shutdowns (Stacey, 2025; Li et al., 2023c). When a single TC threatens several ports, anticipating which ports are likely to be afected and where disruptions may last longest can help authorities prioritize preparedness and recovery eforts.

As a TC approaches, ports and vessel operators take precautionary measures such as suspending port operations and rerouting vessels (Li et al., 2022; Polydoropoulou et al., 2025). Current hurricane preparedness protocols use a series of port conditions, including Whiskey, X-ray, Yankee, and Zulu, based on the expected arrival of gale-force winds (USCG, 2024). These conditions guide restrictions on vessel movements. After the winds subside, the Captain of the Port (COTP), in coordination with the Port Heavy Weather Advisory Group (PHWAG), assesses port facilities, navigational aids, and operational readiness before authorizing reopening (USCG, 2024). Such precautions reduce immediate safety risks, but closures can also result in substantial economic losses (Baker et al., 2012). Forecast uncertainty complicates these decisions. An overly cautious response may impose unnecessary costs, while underestimating the threat may leave a port inadequately prepared (Mollaoglu et al., 2023). Moreover, ports exposed to the same TC can experience diferent operational impacts depending on the cyclone’s intensity and landfall location as well as the characteristics of each port (Liu et al., 2026; Sturgis et al., 2014). A forecast track alone therefore provides limited information about how long operations at individual ports may be disrupted. Estimates of the relative impacts across ports could help authorities prioritize inspections and allocate recovery resources (Lin Moe and Pathranarakul, 2006; Pu et al., 2025; Wei et al., 2024). They could also help vessel operators evaluate rerouting options before the cyclone arrives (Edwards et al., 2023).

Existing studies have used empirical data, including Automatic Identification System (AIS) records and port throughput data, to quantify port impacts and recovery under extreme weather events (Farhadi et al., 2016; Touzinsky et al., 2018; Trepte and Rice Jr, 2014; Verschuur et al., 2020; Kuai et al., 2026a). Several studies use operational records to estimate disruption duration and construct resilience curves for afected ports (Touzinsky et al., 2018; Verschuur et al., 2020). These analyses establish how port activity declined and recovered during past events. For preparedness planning, the challenge is to estimate the potential impacts of an impending TC while its eventual path and the resulting disruptions are still unknown (Zhou et al., 2021). To estimate port performance under such conditions, researchers have adopted qualitative methods such as the fuzzy analytical hierarchy process (Tahri et al., 2017), as well as quantitative approaches including simulation (Dhanak et al., 2021; Folkman et al., 2021) and Bayesian networks (Hossain et al., 2020; Wang et al., 2023). These approaches have advanced the assessment of port performance under extreme weather, although their estimates can be sensitive to model assumptions and expert judgment.

The availability of vessel AIS data and advances in data-driven algorithms have expanded the use of artificial intelligence in port operations (Filom et al., 2022). Many studies have employed machine learning (ML) and deep learning (DL) to predict routine operational outcomes, such as vessel arrival time (El Mekkaoui et al., 2023; Park et al., 2021) and port throughput (Li et al., 2019a). Recent work has also examined how well AI models interpret cyclone forecast products for port preparedness (Kuai et al., 2026b, 2025; Wu et al., 2026). Estimating the operational impacts of an impending TC presents two challenges. First, TC-related disruptions are infrequent at individual ports, limiting the historical observations available for model training. Second, diferences in impact duration across ports depend on interactions between port and cyclone characteristics, and uncertainty in the forecast landfall location makes these diferences dificult to estimate before landfall.

This study examines port impact ranking in the Gulf of Mexico using data from five years between 2017 and 2022. We exclude 2020 because pandemic-related disruptions complicate the attribution of changes in transportation activity to TCs (Choi, 2021; Kong et al., 2024a). We use AIS vessel counts to identify the number of impact days at each port during each historical TC. We then combine these observations with port characteristics from the Port Performance Freight Statistics Program (Stacey, 2025) and cyclone records from the National Hurricane Center (NHC, 2023b,a). The resulting dataset links each observed port–cyclone interaction to its impact duration and to the characteristics of both the port and the cyclone.

Building on earlier work on port impact ranking (Li et al., 2023c), we formulate the assessment of relative port impacts as a recommendation problem. Following the user–item formulation of recommendation systems (Ricci et al., 2021), each TC is treated as a user, each port as an item, and the observed number of impact days as the interaction signal. This formulation allows the model to draw on observations from ports and cyclones with similar characteristics when historical observations for a particular combination are limited. To account for forecast uncertainty, we augment the historical data by sampling plausible landfall locations along the coastline within the forecast uncertainty cone while retaining the observed number of impact days for each port– cyclone pair. For each landfall scenario, the two-stage recommendation framework first retrieves potentially afected ports and then ranks them by their estimated number of impact days. Within the ranking stage, we compare three representations of port–cyclone interactions: pairwise interactions in a factorization machine (FM), tree-based feature combinations in gradient boosted decision trees with linear regression (GBDT+LR), and explicit feature crosses combined with learned embeddings in Wide & Deep Learning (WDL).

This study makes three contributions. First, we construct a dataset that links AIS-derived impact days with port and cyclone characteristics, providing an empirical basis for comparing the operational impacts of the same TC across ports. Second, we extend earlier FM-based work by examining how diferent representations of port–cyclone interactions can be used to rank potentially afected ports within a common two-stage recommendation framework when historical observations are sparse. Third, using Hurricanes Nicole and Ian as out-of-year test cases, we evaluate how well the framework identifies afected ports and how closely the estimated rankings agree with the observed port impact rankings across plausible landfall locations. This evaluation addresses the practical question of which ports should receive priority for post-cyclone inspections and recovery eforts.

## 2. Data Collection and Processing

In this study, our goal is to rank ports by their expected number of impact days under an impending tropical cyclone. An impact day is a day on which port activity falls below its normal operating range during a tropical cyclone. A port with a greater number of impact days is ranked as more severely afected. To estimate these relative impacts, the recommendation algorithm draws on historical interactions involving ports and cyclones with similar characteristics (Ricci et al., 2021). We treat each tropical cyclone as a user, each port as an item, and the observed number of impact days as the interaction signal (Li et al., 2023c). This section describes the port and cyclone characteristics used in the analysis and explains how the number of impact days is estimated for each port–cyclone pair.

## 2.1. Port Characteristics Data

The Gulf Coast, with its abundant oil and gas reserves and strategic shipping routes, serves as a hub for the petrochemical industry and related activities in the United States. The region is frequently afected by tropical cyclones that disrupt port operations (Kuai et al., 2026a). During Hurricane Laura in 2020, for example, several ports along the Gulf Coast partially or completely suspended operations. This study focuses on 37 ports in the study region. Information on port characteristics was manually collected and processed from the Port Performance Freight Statistics Program (Stacey, 2025). Table 1 summarizes the location, governance, cargo activity, and harbor conditions of the study ports. Given the relative stability of these characteristics and the limited availability of annual observations, the same set of port attributes is used for the five study years.

Table 1: Descriptive statistics of port infrastructure data
<table><tr><td>Variable</td><td></td><td></td><td>Prop. (%)</td><td>Variable</td><td>Num.</td><td>Prop. (%)</td></tr><tr><td>State</td><td></td><td></td><td></td><td colspan="2">Dominant vessel type</td><td></td></tr><tr><td>Alabama (AL)</td><td></td><td>2 8</td><td>5.4</td><td>Tanker</td><td>15</td><td>40.5</td></tr><tr><td>Florida (FL)</td><td></td><td></td><td>21.6</td><td>Container</td><td>7</td><td>18.9</td></tr><tr><td>Louisiana (LA)</td><td></td><td>10</td><td>27.0</td><td>Dry bulk</td><td>13</td><td>35.1</td></tr><tr><td>Mississippi (MS)</td><td></td><td>6</td><td>16.2</td><td>Roll-on/roll-off</td><td>2</td><td>5.4</td></tr><tr><td>Texas (TX)</td><td></td><td>11</td><td>29.7</td><td>Harbor type</td><td></td><td></td></tr><tr><td>Port governance</td><td></td><td>3</td><td></td><td>River natural</td><td>20</td><td>54.1</td></tr><tr><td>Other</td><td></td><td></td><td>8.1</td><td>Coastal natural</td><td>11</td><td>29.7</td></tr><tr><td>Municipal</td><td></td><td>3</td><td>8.1</td><td>Lake or canal</td><td>4</td><td>10.8</td></tr><tr><td>Special district</td><td></td><td>13</td><td>35.1</td><td>Sea port</td><td>2</td><td>5.4</td></tr><tr><td>Country</td><td></td><td>13 5</td><td>35.1</td><td>Harbor size</td><td></td><td></td></tr><tr><td>State</td><td></td><td></td><td>13.5</td><td>Very small</td><td>7</td><td>18.9</td></tr><tr><td>Container port</td><td></td><td></td><td></td><td>Small</td><td>16</td><td>43.2</td></tr><tr><td>Yes</td><td></td><td>12</td><td>32.4</td><td>Medium</td><td>10</td><td>27.0</td></tr><tr><td>No</td><td></td><td>25</td><td>67.6</td><td>Large</td><td>4</td><td>10.8</td></tr><tr><td colspan="2">Annual average tonnage (million tons)</td><td></td><td></td><td>Shelter condition</td><td></td><td></td></tr><tr><td>Mean</td><td>Std.</td><td>Min</td><td>Max</td><td>Average</td><td>7</td><td>18.9</td></tr><tr><td>35.09</td><td>61.09</td><td>0.99</td><td>275.94</td><td>Good</td><td>21</td><td>56.8</td></tr><tr><td></td><td></td><td></td><td></td><td>Excellent</td><td>9</td><td>24.3</td></tr></table>

Note: Num. denotes the number of ports, and Prop. denotes the percentage of the 37 study ports. Percentages are rounded to one decimal place. Mean, standard deviation (Std.), minimum (Min), and maximum (Max) are reported for annual average tonnage.

## 2.2. Tropical Cyclone Data

We collect data on named tropical cyclones in the study region between 2017 and 2022, excluding 2020. Table 2 summarizes 31 cyclones across the five study years. The National Hurricane

Center (NHC) provides trajectory records and meteorological parameters for each cyclone, including central pressure and wind speed (Cangialosi et al., 2020). Given the relationship between wind speed and cyclone destructiveness, we focus on wind-related attributes (Emanuel, 2005; Kuai et al., 2026a). The trajectory records include the dates and times of formation, landfall, and dissipation, as well as the coordinates of key locations along each track. We use these spatial and temporal features to derive variables for each port–cyclone pair, including the distance from the port to the landfall location. For cyclones that did not make landfall, we use the point on the track closest to the coastline as the reference location and calculate the distance from each port to that point. For cyclones that made multiple landfalls, we use the first landfall location.

Table 2: Descriptive statistics of tropical cyclone data
<table><tr><td>Variable</td><td>Num.</td><td>Prop. (%)</td><td>Variable</td><td></td><td>Num.</td><td>Prop. (%)</td></tr><tr><td>Year</td><td></td><td></td><td>TS</td><td colspan="3">Tropical cyclone intensity</td></tr><tr><td>2017</td><td></td><td></td><td></td><td></td><td>12</td><td>38.7</td></tr><tr><td>2018</td><td></td><td></td><td></td><td></td><td>8</td><td>25.8</td></tr><tr><td>2019</td><td></td><td></td><td>19.4</td><td></td><td>2</td><td>6.5</td></tr><tr><td>2021</td><td></td><td></td><td>22.6 16.1</td><td></td><td>3</td><td>9.7</td></tr><tr><td>2022</td><td></td><td></td><td></td><td></td><td>3</td><td>9.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>3</td><td>9.7</td></tr><tr><td colspan="3">Lowest central pressure (millibars)</td><td colspan="3">Maximum wind speed (kts) Std. Min</td><td></td></tr><tr><td>Mean</td><td>Std.</td><td>Min</td><td>Max</td><td>Mean</td><td></td><td>Max</td></tr><tr><td>976.3</td><td>26.59</td><td>914</td><td>1003</td><td>77.6</td><td>35</td><td>155</td></tr><tr><td colspan="3">Wind speed at landfall (kts)</td><td colspan="3">34.2 Duration from formation to landfall (days)</td><td></td></tr><tr><td>Mean</td><td>Std.</td><td>Min</td><td>Max</td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>65.8</td><td>28.9</td><td></td><td>140</td><td></td><td>0.71</td><td>15.5</td></tr><tr><td colspan="3">Duration from formation to dissipation (days)</td><td colspan="3">4.4 3.3 Distance from port to formation location (km)</td><td></td></tr><tr><td>Mean</td><td>Std.</td><td>Min</td><td>Max</td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>6.7</td><td>4.3</td><td>2</td><td>19</td><td></td><td>42.2</td><td>7465.0</td></tr><tr><td colspan="3">Distance from port to maximum wind location (km)</td><td colspan="3">Distance from port to landfall location (km)</td><td></td></tr><tr><td>Mean</td><td>Std.</td><td>Min</td><td>Max</td><td>Mean</td><td></td><td>Max</td></tr><tr><td>1131.0</td><td>823.6</td><td>8.7</td><td>3932.2</td><td>867.7</td><td>Min 0.2</td><td>2272.3</td></tr></table>

Note: TS denotes a tropical storm, and H1–H5 denote Categories 1–5 on the Safir–Simpson Hurricane Wind Scale. Wind speed is reported in knots (kts), where 1 knot = 1.852 km/h. Num. denotes the number of cyclones, and Prop. denotes the percentage of the 31 cyclones summarized in the table. The study years are 2017, 2018, 2019, 2021, and 2022.

## 2.3. AIS Data

To determine the number of impact days for each port–cyclone pair, we use the number of commercial vessels within each port area as an indicator of port performance (Touzinsky et al., 2018). Automatic Identification System (AIS) data record vessel locations and operational characteristics and are widely used for maritime applications such as collision avoidance, vessel assignment, and port management (Yang et al., 2019). The records include timestamps, coordinates, vessel types, speeds, and headings at approximately 30-second intervals. A single year of AIS data can exceed 300 GB. As in other analyses of transportation activity records, preprocessing reduces the volume of data and prepares the observations for analysis (Li et al., 2019b). Following Verschuur et al. (2020), we apply the following steps:

• We retain cargo vessels, tankers, and container vessels and exclude vessel types outside the scope of the analysis, including fishing vessels, tugs, and towing vessels.

• We define port boundaries using coordinate ranges and count the active vessels within each boundary.

• We aggregate hourly vessel counts into a daily time series for each port.

The resulting time series are used to estimate the number of impact days at each port. Previous studies have applied threshold methods (Verschuur et al., 2020) or Bayesian change-point detection (Touzinsky et al., 2018) to identify impact periods. To account for recurring seasonal variation in port activity, we use a modular time-series regression model (Taylor and Letham, 2018). The model decomposes the vessel count series into trend and seasonal components. We adopt an additive formulation so that the expected daily vessel count reflects both the underlying trend and the seasonal pattern at each port. For each port, we fit the additive regression model using one year of vessel count data. A day is classified as an impact day when the observed vessel count falls below the lower bound of the model’s predefined interval and the date coincides with a tropical cyclone event.

Figure 1 illustrates the fitted time-series pattern and the 95% interval for Port Houston in 2017. During that hurricane season, only Hurricane Harvey was associated with vessel counts low enough to meet the impact-day criterion, resulting in six impact days. Days with vessel counts above the upper bound are excluded from the impact-day count. Such increases may reflect a temporary rise in activity as ports handle cargo delayed by an earlier disruption (Verschuur et al., 2020). Across the port–cyclone pairs analyzed during 2017–2022, excluding 2020, the mean number of impact days was 0.29, with a standard deviation of 0.95. The observed values ranged from 0 to 7 days.

![](images/0fb58213a4213c8f277edf018d87c545e2ea7177c1663c89ff393b8e0fe484f3.jpg)  
Figure 1: Identifying cyclone-related impact days from vessel counts at Port Houston in 2017.

## 3. Methodology

Historical observations of port disruptions under tropical cyclones (TCs) are sparse. As described in Section 2, we treat each TC as a user, each port as an item, and the observed number of impact days as the interaction signal (Ricci et al., 2021; Li et al., 2023c). This formulation allows the model to draw on observations from ports and cyclones with similar characteristics when estimating the impact of an impending TC.

Using historical data on 37 ports and 31 TCs, we apply a two-stage recommendation framework to identify potentially afected ports and rank them by their expected number of impact days (Li et al., 2023c). For each plausible landfall location, we first retrieve the ten nearest ports. We then estimate the number of impact days for each retrieved port and rank the ports in descending order of their estimates. All three ranking models use the same candidate ports within a given landfall scenario (Figure 2).

![](images/95bf474dc2f2aa0b947a5bd5543083e53ecf87fae51141b339f2b2cf0a3256a3.jpg)  
Figure 2: Two-stage recommendation framework for port impact ranking.

Before a TC makes landfall, uncertainty in its track afects which ports may be exposed (Cangialosi et al., 2020; NHC, 2023b). To represent uncertainty in landfall location in the historical data, we define a sampling region with a radius of 120 km centered on the recorded landfall point (Li et al., 2023c). For each TC, we randomly sample 20 plausible landfall locations uniformly along the portion of the coastline within this region. Each sampled location defines one landfall scenario. We recalculate the distance from each of the 37 ports to the sampled point and use these distances to retrieve the ten nearest ports.

This data augmentation varies the landfall location while retaining the observed impacts of each historical event (Rebufi et al., 2021). As the sampled location changes, the distances to the ports change, and a diferent set of ports may be retrieved. The observed number of impact days for each port–cyclone pair remains fixed across scenarios because it is derived from AIS vessel counts during the historical event. This allows the rankings obtained under diferent plausible landfall locations to be evaluated against the port impacts observed during the same TC (Figure 3).

Across the 31 TCs, the 20 sampled locations per cyclone yield 620 landfall scenarios. Before retrieval, the augmented dataset contains $3 7 \times 3 1 \times 2 0 = 2 2 { , } 9 4 0$ port–cyclone–scenario records. Selecting ten ports per scenario yields $1 0 \times 3 1 \times 2 0 \textmd { = } 6 { , } 2 0 0$ candidate records across the full dataset. We use data from 2017, 2018, 2019, and 2021 for model development and Hurricanes Nicole and Ian in 2022 as the two test cases. All scenarios associated with a given cyclone are kept together when partitioning the data.

![](images/554ba241c80de0675ac10ad821298ec914f710c1f093d5028e570790c0075aab.jpg)  
Figure 3: Data augmentation using plausible landfall locations sampled along the coastline within 120 km of the recorded landfall point.

## 3.1. Recommendation Algorithm

The three ranking models difer in how they represent interactions between port and cyclone characteristics. Feature crossing allows combinations of attributes to contribute to the estimated number of impact days (Liu and Neville, 2023). A factorization machine (FM) represents pairwise feature interactions through latent vectors (Rendle, 2010). Gradient boosted decision trees combined with linear regression (GBDT+LR) use combinations of conditions along tree paths to generate features for regression. Wide & Deep Learning (WDL) jointly learns from explicit feature crosses and dense embeddings (Cheng et al., 2016). Each model estimates impact days for the retrieved ports, allowing their rankings to be compared within the same landfall scenario.

## 3.1.1. Factorization Machine

A factorization machine captures second-order feature interactions by factorizing their weights, making it suitable for sparse, high-dimensional inputs (Rendle, 2010). In our implementation, continuous port and cyclone attributes are discretized before one-hot encoding. The featureprocessing details are provided in Appendix Appendix A.

The FM consists of a linear component and a pairwise interaction component. Each input feature $x _ { i }$ is associated with a k-dimensional latent vector $\mathbf { v } _ { i } .$ , where k is a hyperparameter. The interaction weight for features $x _ { i }$ and $x _ { j }$ is given by their inner product, $\begin{array} { r } { \langle \mathbf { v } _ { i } , \mathbf { v } _ { j } \rangle = \sum _ { f = 1 } ^ { k } \nu _ { i , f } \nu _ { j , f } } \end{array}$ . The same latent vector $\mathbf { v } _ { i }$ is used in every interaction involving feature $x _ { i }$ . Thus, observations of other feature combinations contribute to the estimated interaction weight even when a particular pair is rarely observed. For n input features, the interaction component uses nk latent parameters in place of a separate weight for each of the $n ( n - 1 ) / 2$ feature pairs. The estimated number of impact days is

$$
\hat { y } ( \mathbf { x } ) = \underbrace { \left( w _ { 0 } + \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \right) } _ { \mathrm { l i n e a r ~ c o m p o n e n t } } + \underbrace { \left( \sum _ { i = 1 } ^ { n - 1 } \sum _ { j = i + 1 } ^ { n } \langle \mathbf { v } _ { i } , \mathbf { v } _ { j } \rangle x _ { i } x _ { j } \right) } _ { \mathrm { p a i r w i s e ~ i n t e r a c t i o n s } } ,\tag{1}
$$

where $w _ { 0 }$ is the bias term and $w _ { i }$ is the linear weight for feature $x _ { i }$ . Rewriting Eq. (1) as Eq. (2) reduces the computational complexity of evaluating the interaction terms from $O ( k n ^ { 2 } )$ to $O ( k n )$ (Rendle, 2010).

$$
\widehat { y } ( \mathbf { x } ) = w _ { 0 } + \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } + \frac { 1 } { 2 } \sum _ { f = 1 } ^ { k } \left[ \left( \sum _ { i = 1 } ^ { n } \nu _ { i , f } x _ { i } \right) ^ { 2 } - \sum _ { i = 1 } ^ { n } \nu _ { i , f } ^ { 2 } x _ { i } ^ { 2 } \right] .\tag{2}
$$

We estimate the model parameters using stochastic gradient descent (SGD) with squared-error loss.

## 3.1.2. Gradient Boosted Decision Trees with Linear Regression

Gradient boosted decision trees can capture nonlinear relationships among input features (Friedman, 2001; Kong et al., 2022; Li et al., 2023b,a; Kong et al., 2024b; Li et al., 2025). In our GBDT+LR model, the trees generate features from port and cyclone characteristics, and linear regression uses these features to estimate impact days. The trees are fitted iteratively, with successive trees using the negative gradients of the boosting loss to update the model (Friedman, 2001). Each path from the root of a tree to a leaf represents a combination of conditions on the input attributes. For example, one path combines three conditions: the hurricane wind scale is below H5, the harbor type is not a seaport, and the port is less than 200 km from the sampled landfall location (Figure 4). A port–cyclone pair that satisfies the conditions along this path reaches the corresponding leaf.

For each training example, we record the index of the leaf reached in each selected tree and apply one-hot encoding to the leaf indices. The resulting binary vector has one active entry per selected tree. Each active entry identifies a combination of feature conditions satisfied by that example. These vectors serve as inputs to a linear regression model, with the observed number of impact days as the target. For each new landfall scenario, the fitted trees and the same leaf encoder are applied to the retrieved ports before estimating their impact days (Algorithm 1 and Figure 5).

Algorithm 1 GBDT with linear regression for port impact ranking   
Require: Training examples $\{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , number of boosting iterations T, maximum tree depth D,   
and learning rate η   
Ensure: Estimated number of impact days for each retrieved port   
1: Fit a gradient boosting classifier to the training examples using T boosting iterations, maxi  
mum depth D, and learning rate η   
2: Select the trees used for leaf encoding as described in Appendix Appendix A   
3: for i ← 1 to N do   
4: Record the leaf index reached by $\mathbf { X } _ { i }$ in each selected tree as the vector $\ell _ { i }$   
5: end for   
6: Fit a one-hot encoder E to the training leaf-index vectors   
7: Set $\mathbf { z } _ { i } = E ( \pmb { \ell } _ { i } )$ for each training example   
8: Estimate w and b by least squares using $\{ ( \mathbf { z } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$   
9: for each retrieved port in a landfall scenario do   
10: Obtain its leaf-index vector ℓ from the fitted trees   
11: Estimate impact days as $\hat { y } = \mathbf { w } ^ { \top } E ( \pmb { \ell } ) + b$   
12: end for

![](images/c9c050863065d467717abcfc543b6da5b4b585ec2b40ab2dc636b7215595c9c8.jpg)  
Figure 4: Tree-based encoding of port and cyclone feature combinations.

![](images/d8d0bc3ca5e5bdccc2c8856d76f3d65febf940b3265560d742061d402c3d76bb.jpg)  
Figure 5: Port impact estimation using GBDT leaf encodings and linear regression.

## 3.1.3. Wide & Deep Learning

In GBDT+LR, the trees are fitted before the linear regression, so the leaf encodings remain fixed when the regression weights are estimated. Wide & Deep Learning provides a third way to represent port–cyclone interactions by training a wide linear component and a deep neural network jointly (Cheng et al., 2016). The wide component uses the original features and explicit feature crosses to capture associations between recurring feature combinations and observed impact days. The deep component maps categorical port and cyclone attributes to dense embedding vectors. These embeddings allow the model to draw on information from port–cyclone pairs with similar characteristics when historical observations for a particular combination are limited.

The embedding vectors are randomly initialized and updated through backpropagation during training. The concatenated embeddings form the input to the hidden layers of the deep component. The transformation from layer l to layer l + 1 is given by

$$
a ^ { ( l + 1 ) } = f \left( W ^ { ( l ) } a ^ { ( l ) } + b ^ { ( l ) } \right) ,\tag{3}
$$

where $\boldsymbol { a } ^ { ( l ) }$ is the activation vector at layer l, $W ^ { ( l ) }$ and $b ^ { ( l ) }$ are the weights and biases for the transformation, and $f ( \cdot )$ is the activation function. Successive nonlinear transformations allow the deep component to learn interactions among the embedded features.

To estimate impact days, the model combines the contribution of the original features and their explicit crosses with the output of the final hidden layer (Figure 6):

$$
\hat { y } _ { \mathrm { W D L } } ( \mathbf { x } ) = w _ { \mathrm { w i d e } } ^ { \top } [ \mathbf { x } , \boldsymbol { \phi } ( \mathbf { x } ) ] + w _ { \mathrm { d e e p } } ^ { \top } a ^ { ( l _ { f } ) } + b ,\tag{4}
$$

where $\phi ( \mathbf { x } )$ denotes the explicit feature crosses and $[ { \bf x } , \phi ( { \bf x } ) ]$ denotes their concatenation with the original features. The vectors ${ \boldsymbol { w } } _ { \mathrm { w i d e } }$ and $w _ { \mathrm { d e e p } }$ are the output weights for the wide component and the final deep-layer activation $\boldsymbol { a } ^ { ( l _ { f } ) }$ , respectively, and $^ b$ is the output bias. During training, $L _ { 2 }$ regularization penalizes large weights. As with FM and GBDT+LR, the estimated number of impact days serves as the ranking score for each retrieved port.

![](images/3a6d0459cf37167e097ef4d0aaf73bc478de5d3776501acf2c402c28c2efe9e4.jpg)  
Figure 6: WDL architecture combining sparse feature crosses and dense embeddings.

## 4. Results and Discussion

## 4.1. Experimental Setup and Baseline Method

We split the data by tropical cyclone, keeping all observations from the same TC, including its 20 landfall scenarios, in the same partition. Among the TCs from 2017, 2018, 2019, and 2021, approximately 20% are reserved for validation and the remainder are used for training. Hurricanes Nicole and Ian from 2022 are held out for testing. For each test hurricane, FM, GBDT+LR, and WDL are evaluated on the same candidate ports within each landfall scenario. We implement the three models using PyFM, scikit-learn, and PyTorch, respectively. The model settings are summarized in Table 3, with implementation details provided in Appendix A.

Table 3: Hyperparameter settings for the ranking models.
<table><tr><td>Parameter</td><td>Setting</td><td>Parameter</td><td>Setting</td></tr><tr><td>FM</td><td>WDL</td><td></td><td></td></tr><tr><td>Number of latent factors (k)</td><td>100</td><td>Hidden-layer units</td><td>(256, 128, 64)</td></tr><tr><td>Initial learning rate</td><td>0.001</td><td> $L _ { 2 }$  coefficient (wide component)</td><td>10³</td></tr><tr><td>GBDT+LR</td><td></td><td>Batch size</td><td>256</td></tr><tr><td>Number of trees</td><td>120</td><td>Training epochs</td><td>100</td></tr><tr><td>Maximum tree depth (D)</td><td>30</td><td>Validation split ratio</td><td>0.2</td></tr><tr><td>Learning rate (η)</td><td>0.05</td><td>Optimizer</td><td>Adam</td></tr></table>

We use a distance-based method as the baseline for port impact ranking (Balakrishnan et al., 2022; Li et al., 2023c). For each scenario, the baseline selects ports within 400 km of the sampled landfall location and ranks them from nearest to farthest (Figure 7).

![](images/2b4eb7d01f5fcf033995e9e2a482735b985419436c6b6c7a9f4d13a0c7b1bcae.jpg)  
Figure 7: Distance-based screening and ranking of potentially afected ports.

## 4.2. Performance Measures

We evaluate retrieval using precision, recall, and the $F _ { 2 }$ score. Let U denote the set of ports with at least one observed impact day and L the set of ports identified as potentially afected in a given landfall scenario. Precision (P) measures the proportion of selected ports that were afected, while recall (R) measures the proportion of afected ports that were selected:

$$
P = { \frac { | U \cap L | } { | L | } } ,\tag{5}
$$

$$
R = \frac { | U \cap L | } { | U | } .\tag{6}
$$

The recommendation framework retrieves ten ports in each scenario, while the number selected by the distance-based baseline varies with the sampled landfall location. Because overlooking an

afected port is a concern for preparedness planning, we use the $F _ { \beta }$ score with $\beta = 2$ to give greater weight to recall:

$$
F _ { \beta } = \frac { ( 1 + \beta ^ { 2 } ) P R } { \beta ^ { 2 } P + R } .\tag{7}
$$

For the ranking models, mean absolute error (MAE) and root mean square error (RMSE) measure the diference between the observed number of impact days, $y _ { i }$ , and the estimated number, $\hat { y } _ { i }$ for the N retrieved ports in each scenario:

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| y _ { i } - \hat { y } _ { i } \right| ,\tag{8}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } } .\tag{9}
$$

Both errors are expressed in days. The distance-based baseline provides only a ranking, so MAE and RMSE are reported for the three learned models.

To assess whether the ports with the longest observed disruptions appear near the top of the estimated ranking, we compare the overlap between the two lists at successive depths (Webber et al., 2010). Under the equal-weight setting $( p = 1 )$ of the RBO implementation, the score is the average overlap (AO) across the depths shared by the two lists. Let $\mathcal { U }$ denote the observed ranking of afected ports by impact days and $\mathcal { L }$ the estimated ranking for the same scenario. At depth $d ,$ the sets $U _ { 1 : d }$ and $L _ { 1 : d }$ contain the first d ports in the respective lists. The average overlap is

$$
\mathsf { A O @ } D = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \frac { | U _ { 1 : d } \cap L _ { 1 : d } | } { d } , \qquad D = \operatorname* { m i n } ( | \mathcal { U } | , | \mathcal { L } | ) .\tag{10}
$$

Each depth contributes equally to the average. Ports that appear near the top of both lists contribute to their overlap at multiple depths. The score ranges from 0 to 1, with higher values indicating closer agreement over the evaluated depths. For each test hurricane, we report the mean and standard deviation across its 20 sampled landfall locations.

## 4.3. Comparison of Classification Performance

At the retrieval stage, the recommendation algorithm selects the ten ports nearest to each sampled landfall location, while the distance-based method selects all ports within 400 km of that location. Figure 8 compares how well these two methods identify the ports that were actually afected. FM, GBDT+LR, and WDL share the same retrieval step and therefore have identical classification results. The recommendation algorithm identifies all four afected ports during Hurricane Nicole and all eight during Hurricane Ian, achieving full recall in both cases. With ten ports selected, its precision is 0.40 for Nicole and 0.80 for Ian. The distance-based method achieves higher precision but lower recall for both hurricanes. The mean $F _ { 2 }$ scores are similar for Nicole, with a slight advantage for the distance-based method, while the recommendation algorithm achieves a higher mean $F _ { 2 }$ score for Ian.

![](images/a62342678563fb81231dafbdd1dde6f7da8ff3e3f9efd4132e25c5b65857068f.jpg)  
(a) Hurricane Nicole (2022)

![](images/c9ec8d8b92a3cba479c951e84b268ca1ca5b78d3d08c96995830aeded95ec8bc.jpg)  
(b) Hurricane Ian (2022)  
Figure 8: Precision, recall, and $F _ { 2 }$ scores for identifying afected ports across the sampled landfall scenarios.

## 4.4. Comparison ofRanking Performance

Across both test hurricanes, WDL provides the closest agreement with the observed port impact rankings, and its ranking performance varies less across landfall scenarios than that of the distance-based baseline (Table 4). For Hurricane Nicole, mean AO increases from 0.454 for the baseline to 0.842 for WDL, while the standard deviation decreases from 0.112 to 0.025. The same pattern holds for Hurricane Ian, where mean AO increases from 0.607 to 0.944 and the standard deviation decreases from 0.090 to 0.011. WDL also achieves the lowest MAE and RMSE for both hurricanes, with mean absolute errors of 0.477 days for Nicole and 0.871 days for Ian. Among the other two learned models, GBDT+LR estimates impact days more accurately than FM, while FM produces rankings that agree more closely with the observed order.

Table 4: Impact-day estimation and port ranking performance across 20 landfall scenarios for each test hurricane.
<table><tr><td>Model</td><td>MAE (days) ↓</td><td>RMSE (days) ↓</td><td>AO↑</td></tr><tr><td colspan="4">Hurricane Nicole (H1)</td></tr><tr><td colspan="4">Distance-based 0.454 (0.112)</td></tr><tr><td>FM</td><td>1.346 (0.048)</td><td>1.743 (0.073)</td><td>0.682 (0.027)</td></tr><tr><td>GBDT+LR</td><td>0.498 (0.013)</td><td>0.802 (0.012)</td><td>0.440 (0.050)</td></tr><tr><td>WDL</td><td>0.477 (0.028)</td><td>0.691 (0.021)</td><td>0.842 (0.025)</td></tr><tr><td colspan="4">Hurricane Ian (H5)</td></tr><tr><td colspan="4">Distance-based</td></tr><tr><td>FM</td><td>1.004 (0.088)</td><td>1.301 (0.111)</td><td>0.607 (0.090)</td></tr><tr><td>GBDT+LR</td><td>0.915 (0.001)</td><td>1.180 (0.002)</td><td>0.723 (0.023) 0.682 (0.001)</td></tr><tr><td>WDL</td><td>0.871 (0.036)</td><td>0.933 (0.020)</td><td>0.944 (0.011)</td></tr></table>

Note: Values are means across landfall scenarios, with standard deviations in parentheses. Arrows indicate whether lower (↓) or higher (↑) values are preferred. Bold denotes the best mean for each metric and hurricane.

For inspection and recovery planning, a practical question is whether the ports with the longest observed disruptions remain near the top of the estimated ranking as the assumed landfall location changes. In the three scenarios shown for Nicole, WDL consistently ranks Miami first and Palm Beach second, matching the observed order. The distance-based baseline places Miami seventh, third, and sixth, respectively (Table 5). For Ian, WDL reproduces the observed order of the four highest-ranked ports across all three scenarios: Everglades, Miami, Manatee, and Tampa. The baseline places either Miami or Manatee first (Table 6). Agreement is less consistent farther down the list. For example, WDL ranks Everglades sixth in all three Nicole scenarios, although it was third in the observed ranking. In these two cases, the clearest benefit of incorporating port and cyclone characteristics is the ability to identify the ports facing the longest disruptions even as the sampled landfall location changes.

Table 5: Observed and estimated port impact rankings for three landfall scenarios under Hurricane Nicole.
<table><tr><td>Rank</td><td>Observed</td><td>Dist-1</td><td>Dist-2</td><td>Dist-3</td><td>WDL-1</td><td>WDL-2</td><td>WDL-3</td></tr><tr><td>1 2</td><td>Miami Palm Beach</td><td>Canaveral Palm Beach</td><td>Palm Beach Everglades</td><td>Canaveral Palm Beach</td><td>Miami Palm Beach</td><td>Miami Palm Beach</td><td>Miami Palm Beach</td></tr><tr><td>3</td><td>Everglades</td><td>Tampa</td><td>Miami</td><td>Everglades</td><td>Manatee</td><td>Manatee</td><td>Manatee</td></tr><tr><td>4</td><td>Jacksonville</td><td>Manatee</td><td>Canaveral</td><td>Tampa</td><td>Jacksonville</td><td>Mobile</td><td>Jacksonville</td></tr><tr><td>5</td><td></td><td>Everglades</td><td>Manatee</td><td>Manatee</td><td>Mobile</td><td>Jacksonville</td><td>Mobile</td></tr><tr><td>6</td><td></td><td>Jacksonville</td><td>Tampa</td><td>Miami</td><td>Everglades</td><td>Everglades</td><td>Everglades</td></tr><tr><td>7</td><td></td><td>Miami</td><td></td><td>Jacksonville</td><td>Pascagoula</td><td>Pascagoula</td><td>Pascagoula</td></tr><tr><td>8</td><td></td><td></td><td></td><td></td><td>Panama</td><td>Panama</td><td>Panama</td></tr><tr><td>9</td><td></td><td></td><td></td><td></td><td>Canaveral</td><td>Canaveral</td><td>Canaveral</td></tr><tr><td>10</td><td></td><td></td><td></td><td></td><td>Tampa</td><td>Tampa</td><td>Tampa</td></tr></table>

Note: Dist and WDL denote the distance-based baseline and the WDL ranking model. Matching sufixes refer to the same sampled landfall location. A dash indicates that the list contains no port at that rank.

Table 6: Observed and estimated port impact rankings for three landfall scenarios under Hurricane Ian.
<table><tr><td>Rank</td><td>Observed</td><td>Dist-1</td><td>Dist-2</td><td>Dist-3</td><td>WDL-1</td><td>WDL-2</td><td>WDL-3</td></tr><tr><td>1</td><td>Everglades</td><td>Miami</td><td>Manatee</td><td>Manatee</td><td>Everglades</td><td>Everglades</td><td>Everglades</td></tr><tr><td>2</td><td>Miami</td><td>Everglades</td><td>Tampa</td><td>Tampa</td><td>Miami</td><td>Miami</td><td>Miami</td></tr><tr><td>3</td><td>Manatee</td><td>Palm Beach</td><td>Everglades</td><td>Everglades</td><td>Manatee</td><td>Manatee</td><td>Manatee</td></tr><tr><td>4</td><td>Tampa</td><td>Manatee</td><td>Miami</td><td>Palm Beach</td><td>Tampa</td><td>Tampa</td><td>Tampa</td></tr><tr><td>5</td><td>Palm Beach</td><td>Tampa</td><td>Palm Beach</td><td>Canaveral</td><td>Palm Beach</td><td>Canaveral</td><td>Jacksonville</td></tr><tr><td>6</td><td>Jacksonville</td><td>Canaveral</td><td>Canaveral</td><td></td><td>Canaveral</td><td>Palm Beach</td><td>Palm Beach</td></tr><tr><td>7</td><td>Panama</td><td></td><td></td><td></td><td>Jacksonville</td><td>Jacksonville</td><td>Canaveral</td></tr><tr><td>8</td><td>Canaveral</td><td></td><td></td><td></td><td>Pascagoula</td><td>Pascagoula</td><td>Panama</td></tr><tr><td>9</td><td></td><td></td><td></td><td></td><td>Mobile</td><td>Mobile</td><td>Pascagoula</td></tr><tr><td>10</td><td></td><td></td><td></td><td></td><td>Panama</td><td>Panama</td><td>Mobile</td></tr></table>

Note: Dist and WDL denote the distance-based baseline and the WDL ranking model. Matching sufixes refer to the same sampled landfall location. A dash indicates that the list contains no port at that rank.

## 5. Conclusion

When a single TC threatens several ports, two questions arise for preparedness planning: which ports are likely to be afected, and where are disruptions likely to last longest? This study addresses these questions by formulating the assessment of relative impacts across ports as a recommendation problem. Using AIS-derived impact days for 37 ports and 31 TCs, the framework draws on historical interactions involving ports and cyclones with similar characteristics when observations at an individual port are limited. For each plausible landfall location, it first retrieves potentially afected ports and then ranks them by their estimated number of impact days. Within the ranking stage, we compare three models that difer in how they relate combinations of port and cyclone characteristics to impact duration. FM represents pairwise feature interactions through latent vectors, GBDT+LR uses combinations of conditions along tree paths as inputs to linear regression, and WDL combines explicit feature crosses with learned embeddings. Comparing these models on the same candidate ports allows us to examine how well each representation captures diferences in impact duration across ports exposed to the same TC.

In tests on Hurricanes Nicole and Ian, the retrieval stage identifies all four ports with observed impact days during Nicole and all eight during Ian across the 20 sampled landfall locations for each hurricane. Among the three ranking models, WDL achieves the lowest MAE and RMSE and the closest agreement with the observed port impact rankings. Relative to the distance-based baseline, its mean ranking agreement is 85.5% higher for Nicole and 55.5% higher for Ian, with less variation across landfall scenarios. The individual port rankings illustrate what this improvement means for preparedness planning. In the three scenarios shown for Nicole, the distance-based baseline ranks Miami between third and seventh as the sampled landfall location changes. WDL consistently places Miami first and Palm Beach second, matching the two ports with the longest observed disruptions. For Ian, WDL reproduces the observed order of the four highest-ranked ports in all three illustrated scenarios. These cases suggest that, once potentially afected ports have been identified, accounting for port and cyclone characteristics can help distinguish where disruptions are likely to last longest. Port authorities could use that distinction to prioritize postcyclone inspections and allocate recovery resources across ports threatened by the same event.

This study represents landfall uncertainty by sampling plausible locations along the coastline within 120 km of the recorded landfall point. Future work could extend this analysis using successive forecast advisories to examine how early the ports facing the longest disruptions can be identified and how their rankings change as the forecast track and intensity evolve. Evaluating additional TCs and ports across a wider geographic area would also help establish how well the findings from Nicole and Ian transfer to other events and coastal regions.

## References

Asariotis, R., Benamara, H., Mohos-Naray, V., 2017. Port Industry Survey on Climate Change Impacts and Adaptation. UNCTAD Research Paper No. 18. United Nations Conference on Trade and Development (UNCTAD). Geneva, Switzerland. URL: https://unctad.org/system/files/official-document/ser-rp-2017d18\_ en.pdf.

Baker, C., Teasley, R., Davis, L.B., Qu, X., 2012. A dynamic decision model for port hurricane preparation and shutdown, in: IISE Annual Conference. Proceedings, Institute of Industrial and Systems Engineers (IISE). p. 1.

Balakrishnan, S., Lim, T., Zhang, Z., 2022. A methodology for evaluating the economic risks of hurricane-related disruptions to port operations. Transportation Research Part A: Policy and Practice 162, 58–79.

BTS, 2017. Freight Facts and Figures 2017. Technical Report. U.S. Department of Transportation. URL: https: //www.bts.gov/sites/bts.dot.gov/files/docs/FFF\_2017.pdf.

Cangialosi, J.P., Blake, E., DeMaria, M., Penny, A., Latto, A., Rappaport, E., Tallapragada, V., 2020. Recent progress in tropical cyclone intensity forecasting at the national hurricane center. Weather and Forecasting 35, 1913–1922.

Cheng, H.T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., Anderson, G., Corrado, G., Chai, W., Ispir, M., et al., 2016. Wide & deep learning for recommender systems, in: Proceedings of the 1st workshop on deep learning for recommender systems, pp. 7–10.

Choi, T.M., 2021. Risk analysis in logistics systems: A research agenda during and after the covid-19 pandemic.

Dhanak, M., Parr, S., Kaisar, E.I., Goulianou, P., Russell, H., Kristiansson, F., 2021. Resilience assessment tool for port planning. Environment and Planning B: Urban Analytics and City Science 48, 1126–1143.

Edwards, D.R., Idoko, F.O., Vogiatzis, C., Davis, L.B., Mirchandani, P., 2023. Determining optimal fuel delivery strategies under uncertainty. Socio-Economic Planning Sciences 88, 101612.

El Mekkaoui, S., Benabbou, L., Berrado, A., 2023. Deep learning models for vessel’s eta prediction: bulk ports perspective. Flexible Services and Manufacturing Journal 35, 5–28.

Emanuel, K., 2005. Increasing destructiveness of tropical cyclones over the past 30 years. Nature 436, 686–688.

Farhadi, N., Parr, S.A., Mitchell, K.N., Wolshon, B., 2016. Use of nationwide automatic identification system data to quantify resiliency of marine transportation systems. Transportation Research Record 2549, 9–18.

Filom, S., Amiri, A.M., Razavi, S., 2022. Applications of machine learning methods in port operations–a systematic literature review. Transportation Research Part E: Logistics and Transportation Review 161, 102722.

Folkman, D., Gharehgozli, A., Mileski, J., Galvao, C.B., 2021. Port resiliency and the efects of hurricanes on port operations. International Journal of Advanced Operations Management 13, 409–430.

Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics , 1189–1232.

Hossain, N.U.I., El Amrani, S., Jaradat, R., Marufuzzaman, M., Buchanan, R., Rinaudo, C., Hamilton, M., 2020. Modeling and assessing interdependencies between critical infrastructures using bayesian network: A case study of inland waterway port and surrounding supply chain network. Reliability Engineering & System Safety 198, 106898.

Kong, X., Li, Z., Zhang, Y., Chen, X., Das, S., Sheykhfard, A., 2024a. Case study on the relationship between socio-demographic characteristics and work-from-home behavior before, during, and after the covid-19 pandemic. Transportation research record 2678, 11–23.

Kong, X., Li, Z., Zhang, Y., Das, S., 2022. Bridge deck deterioration: Reasons and patterns. Transportation research record 2676, 570–584.

Kong, X., Zhang, A., Zhang, Y., Guo, X., Xiao, X., Li, Z., 2024b. Investigating relationships between phone use while driving behavior and drivers’ socio-demographic characteristics: An interpretable machine learning approach. Data Science for Transportation 6, 8.

Kuai, C., Li, Z., Rosen, B., Paal, S., Jafari, N., Briaud, J.L., Zhang, Y., Hashash, Y., Zhou, Y., 2025. Knowledgegrounded agentic large language models for multi-hazard understanding from reconnaissance reports. arXiv preprint arXiv:2511.14010 .

Kuai, C., Li, Z., Zhang, Y., Wang, X.B., Lord, D., Zhou, Y., 2026a. Us port disruptions under tropical cyclones: Resilience analysis by harnessing multiple-source dataset. Transportation Research Part D: Transport and Environment 157, 105420.

Kuai, C., Wu, C., Zhou, Y., Wang, B., Yang, T., Tu, Z., Li, Z., Zhang, Y., 2026b. Cyportqa: Benchmarking multimodal large language models for cyclone preparedness in port operation, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 38781–38789.

Li, L., Xiong, X., Yuan, H., 2022. Ships’ response strategies to port disruptions caused by hurricanes. Ocean & Coastal Management 227, 106275.

Li, Y., Li, T., Zuo, Y., Chen, C.P., Shan, Q., Xiao, Y., Fan, X., 2019a. A review of research on port throughput forecasting, in: 2019 IEEE International Conference on Smart Internet of Things (SmartIoT), IEEE. pp. 449–453.

Li, Z., Kong, X., Zhang, Y., 2023a. Exploring factors associated with crossing assertiveness of pedestrians at unsignalized intersections. Transportation research record 2677, 182–198.

Li, Z., Ma, C., Zhou, Y., Lord, D., Zhang, Y., 2025. Leveraging textual description and structured data for estimating crash risks of trafic violation: A multimodal learning approach. IEEE Transactions on Intelligent Transportation Systems 26, 9201–9213.

Li, Z., Tian, X., Li, Z., Zhou, W., Zhou, Z., Zhong, M., 2019b. Risk analysis of metro station passenger flow based on passenger flow patterns. J. Tsinghua Univ.(Sci. Technol.) 59, 854–860.

Li, Z., Wei, Z., Zhang, Y., Kong, X., Ma, C., 2023b. Applying an interpretable machine learning framework to study mobility inequity in the recovery phase of covid-19 pandemic. Travel Behaviour and Society 33, 100621.

Li, Z., Zhang, Y., Wang, B., et al., 2023c. Prediction of port recovery time after a severe storm project .

Lin Moe, T., Pathranarakul, P., 2006. An integrated approach to natural disaster management: public project management and its critical success factors. Disaster prevention and management: an international journal 15, 396–413.

Liu, J., Neville, J., 2023. Stationary algorithmic balancing for dynamic email re-ranking problem, in: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4527–4538.

Liu, J., Yu, W., Fu, X., Liu, J., 2026. Assessing the port resilience under the red sea crisis. Journal of Transport Geography 130, 104474.

Mollaoglu, M., Altay, B.C., Balin, A., 2023. Bibliometric review of route optimization in maritime transportation: environmental sustainability and operational eficiency. Transportation Research Record 2677, 879–890.

Ng, K.Y.A., Becker, A., Cahoon, S., Chen, S.L., Earl, P., Yang, Z., 2015. Climate change and adaptation planning for ports. Routledge London.

NHC, 2023a. Glossary of nhc terms. https://www.nhc.noaa.gov/aboutgloss.shtml. Accessed: 2023-07-22.

NHC, 2023b. Nhc tropical cyclone graphical product descriptions. https://www.nhc.noaa.gov/ aboutnhcgraphics.shtml. Accessed: 2023-07-21.

Park, K., Sim, S., Bae, H., 2021. Vessel estimated time of arrival prediction system based on a path-finding algorithm. Maritime Transport Research 2, 100012.

Polydoropoulou, A., Velegrakis, A., Papaioannou, G., Karakikes, I., Bouhouras, E., Thanopoulou, H., Chatzistratis, D., Monioudi, I., Moschopoulos, K., Chatzipavlis, A., 2025. A composite port resilience index focused on climaterelated hazards: Results from greek ports’ living-labs. Maritime Transport Research 9, 100136.

Pu, F., Li, Z., Wu, Y., Ma, C., Zhao, R., 2025. Recent advances in disaster emergency response planning: Integrating optimization, machine learning, and simulation. arXiv preprint arXiv:2505.03979 .

Rebufi, S.A., Gowal, S., Calian, D.A., Stimberg, F., Wiles, O., Mann, T.A., 2021. Data augmentation can improve robustness. Advances in neural information processing systems 34, 29935–29948.

Rendle, S., 2010. Factorization machines, in: 2010 IEEE International conference on data mining, IEEE. pp. 995– 1000.

Ricci, F., Rokach, L., Shapira, B., 2021. Recommender systems: Techniques, applications, and challenges. Recommender systems handbook , 1–35.

Stacey, M., 2025. Port performance freight statistics. Management .

Sturgis, C.L.A., Smythe, T.C., Tucci, A., 2014. Port recovery in the aftermath of hurricane sandy. Center for a New American Security .

Tahri, M., Maanan, M., Maanan, M., Bouksim, H., Hakdaoui, M., 2017. Using fuzzy analytic hierarchy process multi-criteria and automatic computation to analyse coastal vulnerability. Progress in Physical Geography 41, 268–285.

Taylor, S.J., Letham, B., 2018. Forecasting at scale. The American Statistician 72, 37–45.

Touzinsky, K.F., Scully, B.M., Mitchell, K.N., Kress, M.M., 2018. Using empirical data to quantify port resilience: Hurricane matthew and the southeastern seaboard. Journal of Waterway, Port, Coastal, and Ocean Engineering 144, 05018003.

Trepte, K., Rice Jr, J.B., 2014. An initial exploration of port capacity bottlenecks in the usa port system and the implications on resilience. International Journal of Shipping and Transport Logistics 6, 339–355.

USCG, 2024. Port community information bulletin #09-24: Port condition iv – hurricane season 2024.

Verschuur, J., Koks, E., Hall, J., 2020. Port disruptions due to natural disasters: Insights into port and logistics resilience. Transportation research part D: transport and environment 85, 102393.

Wang, N., Wu, M., Yuen, K.F., 2023. Assessment of port resilience using bayesian network: A study of strategies to enhance readiness and response capacities. Reliability Engineering & System Safety 237, 109394.

Webber, W., Mofat, A., Zobel, J., 2010. A similarity measure for indefinite rankings. ACM Transactions on Information Systems (TOIS) 28, 1–38.

Wei, Z., Zhou, Y., Li, Z., Kulkarni, M., Zhang, Y., 2024. Supporting equitable and responsible highway safety improvement funding allocation strategies–why ai prediction biases matter. Accident Analysis & Prevention 202, 107585.

Wu, K., Kuai, C., Li, Z., Jiang, J., Shen, S., Wang, S., Hu, C.W., Tu, Z., Zhou, Y., 2026. Knowledge is not static: Order-aware hypergraph rag for language models. arXiv preprint arXiv:2604.12185 .

Yang, D., Wu, L., Wang, S., Jia, H., Li, K.X., 2019. How big data enriches maritime research–a critical review of automatic identification system (ais) data applications. Transport reviews 39, 755–773.

Zhou, Y., Li, Z., Meng, Y., Li, Z., Zhong, M., 2021. Analyzing spatio-temporal impacts of extreme rainfall events on metro ridership characteristics. Physica A: Statistical Mechanics and its Applications 577, 126053.

## Appendix A. Model Implementation Details

Input representation and training protocol.. Each example corresponds to a port–cyclone pair under a sampled landfall location. The regression target is the observed number of impact days for that pair, which remains fixed across landfall scenarios. The archived preprocessing pipeline discretizes annual port tonnage at 10, 50, and 100 million tons. Distances from the port to the cyclone formation point and to the landfall location are discretized at 100, 200, 300, 400, and 500 km. Landfall wind speed is grouped using thresholds of 64, 82, 95, 112, and 136 knots, and the time from formation to landfall is rounded to the nearest day. The resulting variables and the remaining categorical port and cyclone attributes are encoded as categorical fields.

We split the data by tropical cyclone, keeping all observations from a given TC in the same partition. Approximately 20% of the cyclones from 2017, 2018, 2019, and 2021 are reserved for validation, and the remainder are used for training. Hurricanes Nicole and Ian in 2022 are held out for testing.

FM and GBDT+LR implementations.. The archived FM implementation uses PyFM with 100 latent factors, 100 training iterations, an initial learning rate of 0.001, and the optimal learningrate schedule. The one-hot encoded features are stored in a compressed sparse row matrix, and FM is fitted to the observed number of impact days using squared-error loss.

The archived GBDT+LR implementation uses GradientBoostingClassifier with 100 boosting iterations, a maximum tree depth of 30, a learning rate of 0.05, and random state=16. At each boosting iteration, the implementation selects the first class-specific tree and records the leaf index reached by each example. The leaf indices from the selected trees are one-hot encoded and used as inputs to LinearRegression to estimate impact days. The leaf encoder is fitted on the training examples and reused for validation and testing.

WDL reference implementation.. The WDL reference configuration uses a wide linear component and a deep component with an eight-dimensional embedding for each categorical field. The wide component takes the original one-hot features and their pairwise categorical crosses as inputs. Feature vocabularies are constructed from the training data, with a reserved index for unseen categories. In the deep component, the embeddings are concatenated and passed through fully connected layers with 256, 128, and 64 units. Each layer uses ReLU activation followed by dropout with probability 0.2. A linear output combines the two components to estimate the number of impact days.

For N training examples, the reference configuration uses the following objective:

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } + 1 0 ^ { - 3 } \lvert | w _ { \mathrm { w i d e } } \rvert | _ { 2 } ^ { 2 } + 1 0 ^ { - 5 } \lvert | W _ { \mathrm { d e e p } } \rvert | _ { 2 } ^ { 2 } ,\tag{A.1}
$$

where $y _ { i }$ and $\hat { y } _ { i }$ are the observed and estimated numbers of impact days. The regularization terms sum the squared weights in the wide and deep components, respectively, excluding biases.

The reference model is trained with Adam at a learning rate of $1 0 ^ { - 3 }$ , a batch size of 256, and 100 epochs. The checkpoint with the lowest validation mean squared error is retained. The random seeds for Python, NumPy, and $\mathrm { P y ' }$ Torch are set to 42. The reference implementation saves the feature mappings, validation assignments, dependency versions, and model checkpoints.