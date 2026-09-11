# Predicting Train Delays in Finland Using Machine Learning and Weather Data

Vinicius Pozzobon Borin, Jean Michel de Souza Sant’Ana and Nurul Huda Mahmood Centre for Wireless Communications, University of Oulu, Oulu, Finland {vinicius.borin, jean.desouzasantana, nurulhuda.mahmood}@oulu.fi

Abstract—Reliable railway operations depend increasingly on real-time environmental intelligence delivered through wireless sensor infrastructures, a capability that 6G networks will substantially enhance through integrated sensing and edge computing. Adverse weather, particularly in Arctic regions with extreme temperatures and heavy precipitation, remains a leading cause of train delays, yet most prediction approaches rely on raw meteorological inputs without exploiting domain-informed feature engineering. This paper investigates machine learning for train delay prediction using the Finland Integrated Train-Weather (FI-TW) dataset, which fuses railway operational records with observations from the Finnish Meteorological Institute’s nationwide sensor network of approximately 200 stations communicating over wireless links. We evaluate three feature configurations using XGBoost at Oulu central station (101,146 observations): full weather features, instant weather observations only, and derived weather category scenarios. The category-based approach, employing hierarchical classifications such as Blizzard, Heavy Snow, and Extreme Cold, achieved an R<sup>2</sup> of 0.78, root mean squared error of 8.5 minutes, and mean absolute error of 3.7 minutes, representing an 11% R<sup>2</sup> improvement and 10% error reduction over alternative configurations. These results demonstrate that compact, domain-informed features derived from sensors streams outperform raw meteorological observations, offering bandwidthefficient representations suitable for edge deployment over current and emerging wireless infrastructures.

Keywords—6G applications, logistics, data science, machine learning, rail transportation.

## I. INTRODUCTION

The evolution toward sixth-generation (6G) wireless networks is expected to enable data-driven services beyond traditional communications, including intelligent transportation and autonomous cyber-physical systems [1]. By offering massive connectivity, ultra-reliable low-latency communication, and native support for sensing and artificial intelligence, 6G will underpin large-scale environmental monitoring infrastructures that feed operational decision-making with real-time data [2]. Railway transportation, serving billions of passengers annually, stands to benefit significantly, as dense Internet-of-Things sensor networks along transportation corridors can anticipate disruptions and optimize operations.

Despite rail’s growing popularity due to its sustainability and cost-effectiveness, train delays remain a significant challenge, causing passenger inconvenience, financial losses, and network congestion. These disruptions arise from technical malfunctions, operational constraints, infrastructure degradation, and environmental factors [3], [4]. Adverse weather conditions are particularly challenging due to their unpredictability and impact on service reliability, especially in Arctic regions. Consequently, accurate delay prediction becomes a critical operational tool, enabling proactive resource allocation, dynamic rescheduling, and improved maintenance planning.

The complexity of factors contributing to train delays makes conventional model-based approaches insufficient [3].

Machine learning (ML) models trained on operational and environmental data can identify complex patterns and enable proactive interventions [5]. Despite extensive railway datasets covering traffic planning, maintenance, and safety [6], most do not fuse weather information with operational records. Huang et al. [7] proposed FCLL-Net for Chinese high-speed rail by incorporating raw meteorological observations as direct inputs, showing that excluding weather features increased Mean Absolute Error (MAE) by 2–4%, confirming the measurable contribution of meteorological data.

Within Europe, Finland’s railway network presents a compelling case for weather-impact studies. The 5,915-kilometer network serves over 90 million passengers annually [8] and experiences extreme conditions, with winter temperatures as low as −40°C causing mechanical failures, signal disruptions, and reduced adhesion [9]. Environmental data is collected through the Finnish Meteorological Institute’s (FMI) distributed sensor infrastructure, a nationwide IoT network of approximately 200 stations communicating over cellular wireless links, many in remote arctic locations where 5G/6G connectivity is essential for reliable data backhaul. Although 86.28% of long-distance trains met punctuality standards in 2024 [10], monthly analysis reveals pronounced temporal variation. Figure 1 shows peak delay rates during winter months (Dec–Feb), with January reaching 28.5% of normalized delay occurrences versus an autumn average of 17.7%.

To address this gap, this paper presents an ML approach for train delay prediction using the Finland Integrated Train-Weather (FI-TW) dataset (2018–2024) [11], which fuses railway operational records with environmental observations from the FMI’s nationwide sensor network, whose stations rely on cellular wireless connectivity for continuous data delivery. This work demonstrates how data collected and transmitted over current and future wireless infrastructure, including emerging 5G/6G networks, can be transformed into actionable intelligence for transportation systems. Specifically, our contributions include: (1) evaluation of ML architectures on weather-integrated railway data, (2) analysis of meteorological features’ predictive value for delay forecasting, (3) a spatialtemporal fusion methodology for integrating heterogeneous sensor streams transmitted over wireless links with operational records, and (4) a hierarchical weather categorization system that transforms raw sensor measurements into bandwidthefficient, operationally relevant features suitable for edge deployment in extreme arctic conditions.

![](images/26c5d6b644e5260c6f131f20cf2660c3739ba6a295a6d4e5b125314fb7b212ff.jpg)  
Fig. 1. Monthly distribution of train delays (2018–2024). Values represent normalized delay percentage for each month across the study period.

## II. DATASET DESCRIPTION

The FI-TW dataset [11] integrates railway operational records from Finland’s Digitraffic railway traffic service with meteorological observations from FMI. Spanning January 2018 to December 2024, the dataset encompasses about 38.5 million records distributed across Finland’s railway network.

## A. Data Sources

1) Railway Operational Data: The railway operational data provides comprehensive real-time and historical information about trains operating throughout Finland. The dataset includes train timetables with scheduled and actual departure/arrival times and station infrastructure metadata for Finland’s railway network. The geographical distribution comprises 549 stations within Finnish territory, of which 38.1% serve passenger traffic while 61.9% are designated for freight handling, docks, or technical service points. The functional classification reveals a hierarchical structure with conventional stations constituting 81.07%, stopping points representing 11.43%, and turnouts in the open line accounting for 7.50%. For each train journey, the dataset records both arrival and departure events at each station along the route. This distinction allows for precise tracking of dwell time at stations and delay propagation throughout the journey. A journey with n stopping points generates 2n − 1 rows when all stations involve commercial stops, with the origin station containing only a departure record and the final destination containing only an arrival record. The delay at each observation point is calculated as differenceInMinutes = actualTime−scheduledTime, where positive values indicate delays, zero indicates on-time performance, and negative values indicate ahead-of-schedule arrivals.

2) Meteorological Data: Meteorological data is sourced from 209 FMI environmental monitoring stations distributed across Finland. These stations are configured at two distinct measurement intervals: 164 stations (78.47%) operate with 10-minute intervals following World Meteorological Organization standards [12], while 45 stations (21.53%) provide high-resolution 1-minute measurements typically positioned at locations requiring detailed temporal resolution such as aviation meteorology [13]. The integration of weather and train data involves a two-stage process. First, spatial matching is performed using the Haversine formula [14], which computes great-circle distances between each train segment’s geographic coordinates and all available meteorological stations operated by the FMI. Each train segment is then assigned to the nearest station, minimizing spatial interpolation error and ensuring that the meteorological observations reflect local conditions as accurately as possible. Second, temporal alignment synchronises the operational and meteorological records by matching each train observation to the closest available weather measurement in time. Since FMI stations report observations at fixed intervals, this step resolves any offset between the train event timestamp and the meteorological recording window, producing a fully aligned dataset in which every observation is associated with both the geographically nearest and temporally closest weather record.

## B. Dataset Features

The final dataset comprises 39 features organized into three categories: target (5), operational (13), and weather (21) features. Table I provides a comprehensive overview of all features, where the “Derived” column indicates features that were engineered from existing raw features through mathematical transformations or aggregations.

1) Target features: The dataset includes five target variables designed to capture different aspects of railway operational performance. The primary target variable differenceIn-Minutes represents the cumulative delay in minutes from the start of the journey until the current station where the data is collected, capturing the total accumulated delay. The differenceInMinutes offset variant removes the initial delay recorded at the first station, isolating delays accumulated during the journey. The differenceInMinutes eachStation offset further removes delay propagation effects at each intermediate station, focusing only on the delay between two adjacent stations. Two binary target variables complement these numeric targets: trainDelayed indicates whether a train experienced any delay, while cancelled identifies cancelled train services.

2) Operational features: capture temporal patterns and train service characteristics. Temporal variables (hour, month, day of week) are encoded using sine-cosine transformations to preserve cyclical relationships, addressing the time wraparound problem where values at cycle boundaries appear maximally distant despite temporal proximity [15]. The encoding follows

TABLE I. DATASET FEATURES OVERVIEW
<table><tr><td>No.</td><td>Feature Name</td><td>Type</td><td>Derived</td></tr><tr><td>1</td><td>differenceInMinutes</td><td>Target</td><td></td></tr><tr><td>2</td><td>differenceInMinutes_offset</td><td>Target</td><td>X</td></tr><tr><td>3</td><td>differenceInMinutes_eachStation_offset</td><td>Target</td><td>X</td></tr><tr><td>4</td><td>trainDelayed</td><td>Target</td><td>X</td></tr><tr><td>5</td><td>cancelled</td><td>Target</td><td></td></tr><tr><td>6</td><td>trainStopping</td><td>Operational</td><td></td></tr><tr><td>7</td><td>commercialStop</td><td>Operational</td><td></td></tr><tr><td>8</td><td>month</td><td>Operational</td><td></td></tr><tr><td>9</td><td>month_sin</td><td>Operational</td><td>X</td></tr><tr><td>10</td><td>month_cos</td><td>Operational</td><td>X</td></tr><tr><td>11</td><td>hour_sin</td><td>Operational</td><td>X</td></tr><tr><td>12</td><td>hour_cos</td><td>Operational</td><td>X</td></tr><tr><td>13</td><td>hour</td><td>Operational</td><td></td></tr><tr><td>14</td><td>day_of_week</td><td>Operational</td><td></td></tr><tr><td>15</td><td>day_week_sin</td><td>Operational</td><td>X</td></tr><tr><td>16</td><td>day_week_cos</td><td>Operational</td><td>X</td></tr><tr><td>17</td><td>day_of_month</td><td>Operational</td><td></td></tr><tr><td>18</td><td>train_id</td><td>Operational</td><td></td></tr><tr><td>19</td><td>Air temperature</td><td>Weather</td><td></td></tr><tr><td>20</td><td>Wind speed</td><td>Weather</td><td></td></tr><tr><td>21</td><td>Gust speed</td><td>Weather</td><td></td></tr><tr><td>22</td><td>Wind direction</td><td>Weather</td><td></td></tr><tr><td>23</td><td>Relative humidity</td><td>Weather</td><td></td></tr><tr><td>24</td><td>Dew-point temperature</td><td>Weather</td><td></td></tr><tr><td>25</td><td>Precipitation intensity</td><td>Weather</td><td></td></tr><tr><td>26</td><td>Snow depth</td><td>Weather</td><td></td></tr><tr><td>27</td><td>Pressure at mean sea level</td><td>Weather</td><td></td></tr><tr><td>28</td><td>Horizontal visibility</td><td>Weather</td><td></td></tr><tr><td>29</td><td>Cloud amount</td><td>Weather</td><td></td></tr><tr><td>30</td><td>weather_scenario_Normal_Clear</td><td>Weather</td><td>X X</td></tr><tr><td>31</td><td>weather_scenario_Blizzard</td><td>Weather</td><td>X</td></tr><tr><td>32</td><td>weather_scenario_Heavy_Snow</td><td>Weather</td><td>X</td></tr><tr><td>33</td><td>weather_scenario_Extreme_Cold</td><td>Weather</td><td></td></tr><tr><td>34</td><td>weather_scenario_Heavy_Rain</td><td>Weather</td><td>X</td></tr><tr><td>35</td><td>weather_scenario_Freezing_Rain</td><td>Weather</td><td>X</td></tr><tr><td>36</td><td>weather_scenario_Black_Ice</td><td>Weather</td><td>X</td></tr><tr><td>37</td><td>weather_scenario_Dense_Fog</td><td>Weather</td><td>X</td></tr><tr><td>38</td><td>weather_scenario_High_Winds</td><td>Weather</td><td>X</td></tr><tr><td>39</td><td>weather_scenario_Extreme_Heat</td><td>Weather</td><td>X</td></tr></table>

$$
t _ { - } { \mathrm { s i n } } = { \mathrm { s i n } } \left( { \frac { 2 \pi t } { P } } \right) { \mathrm { ~ a n d ~ } } t _ { - } { \mathrm { c o s } } = { \mathrm { c o s } } \left( { \frac { 2 \pi t } { P } } \right) ,\tag{1}
$$

where t is the temporal value and P is the period (24 for hours, 12 for months, 7 for days).

3) Weather Features: The weather feature set comprises 11 meteorological measurements obtained from FMI stations: air temperature (°C), wind speed (m/s), gust speed (m/s), wind direction (degrees), relative humidity (%), dew-point temperature (°C), precipitation intensity (mm/h), snow depth (cm), pressure at mean sea level (hPa), horizontal visibility (m), and cloud amount (oktas). These measurements provide continuous numerical values representing atmospheric conditions at the time of train operations. Given the absence of globally standardized definitions for severe weather phenomena such as blizzards, heavy snow, or freezing rain (or even consensus at the European level), we developed a hierarchical classification system to categorize operationally relevant weather conditions. Our approach draws inspiration from European transport weather impact research [16], while representing a novel application of categorical weather classification for railway delay prediction in extreme climatic conditions. We derived 10 additional binary features from meteorological measurements through hierarchical classification logic that categorizes weather conditions into mutually exclusive scenarios based on combined thresholds of multiple parameters. The classification follows a severity-based priority order to ensure proper categorization when conditions overlap, as detailed in Table II.

This hierarchical approach ensures that the most operationally disruptive conditions (e.g., blizzards) take precedence in classification, preventing misclassification when multiple threshold criteria are simultaneously met.

## III. MACHINE LEARNING TRAINING SETUP

This section presents the XGBoost model selection and experimental setup, followed by correlation-based feature analysis that informs the definition of three training scenarios with different feature configurations.

## A. Model Selection: XGBoost

This study employs XGBoost (eXtreme Gradient Boosting) [17] as the primary ML algorithm for delay prediction. XGBoost is an optimized implementation of gradient boosting that constructs an ensemble of decision trees sequentially, where each subsequent tree corrects the residual errors of the preceding ensemble. The algorithm has demonstrated favorable performance across diverse tabular data applications and is particularly well-suited for datasets combining numerical and categorical features with complex non-linear relationships.

TABLE II. WEATHER CATEGORY CLASSIFICATION CRITERIA
<table><tr><td>Category</td><td>Classification Criteria</td></tr><tr><td>Blizzard</td><td>Temp.  $< 0 ^ { \circ } \mathrm { C } ,$  Precip. &gt; 1 mm/h  $\mathrm { o r } > 3$  mm, wind  $> 1 0$  m/s or gust &gt; 15 m/s,</td></tr><tr><td>Heavy Snow</td><td>visibility &lt; 1000 m Temp.  ${ \cal { < } } 0 ^ { \circ } \mathrm { C } ,$  Precip.  $> 2$  mm/h  $\mathrm { o r } > 5$  mm, snow depth  $> 0$  cm</td></tr><tr><td>Extreme Cold Heavy Rain</td><td> $\mathrm { T e m p . } < - 2 { \hat { 0 } } ^ { \circ } \mathrm { C }$  Temp. &gt; 2°C, Precip. &gt; 4 mm/h or &gt; 10</td></tr><tr><td>Freezing Rain</td><td>mm  $- 2 ^ { \circ } \mathbf { C } < \mathrm { t e m p . } < 2 ^ { \circ } \mathbf { C } ,$  Precip.  $> 0$  mm/h</td></tr><tr><td>Black Ice</td><td>or &gt; 0 mm  $- 2 ^ { \circ } \mathbf { C } < \mathrm { t e m p . ~ < ~ 2 ^ { \circ } \mathbf { C } , }$  humidity &gt; 80%,</td></tr><tr><td>Dense Fog</td><td>dew-point - temp &lt; 2°C, precip. &gt; 0 Precip. ≤ 0.1 mm, Visibility &lt; 1000 m,</td></tr><tr><td>High Winds</td><td>humidity &gt; 95% Wind &gt; 15 m/s or gust  $> 2 0$  m/s</td></tr><tr><td>Extreme Heat</td><td>Temp. &gt; 30°C</td></tr><tr><td>Normal/Clear</td><td>No severe weather conditions met</td></tr></table>

Several characteristics make XGBoost appropriate for railway delay prediction. The algorithm inherently handles mixed feature types and missing values, reducing preprocessing requirements. Its tree-based structure naturally captures nonlinear interactions between weather conditions and operational variables without explicit feature engineering. Additionally, XGBoost provides built-in feature importance metrics that enable interpretation of which meteorological and operational factors most strongly influence delay predictions, supporting both predictive accuracy and domain understanding.

## B. Experimental Setup

The experiments focus on Oulu asema (Oulu central station), identified in the network analysis as a high-delay node (19.0% delay rate). The target variable is differenceInMinutes (Feature 1 in Table I), representing the cumulative delay in minutes per train event. The dataset comprises all longdistance services arriving or departing at Oulu asema during 2018–2024, totalling 101,146 observations, which capture the seasonal and meteorological variability of northern Finland.

Because railway operations and weather both exhibit strong temporal autocorrelation, a uniformly random partition would leak future information into training. We therefore apply a chronological 80/20 split: the earliest 80% of observations (sorted by event timestamp) form the development set, and the most recent 20% are held out as a final test set, used only once for reporting. Within the development set, model selection uses 5-fold expanding-window time-series cross-validation, where each validation fold strictly succeeds its training fold in time. Robust scaling and all other preprocessing transformations are fitted on each training fold in isolation and then applied to the subsequent validation fold, so no statistic derived from future data influences fitting. Hyperparameter tuning employs randomized search over the space in Table III, drawing 100 candidate configurations; the configuration minimising the mean validation RMSE across the time-series folds is retrained on the full development set and evaluated once on the held-out test set.

Table III presents the hyperparameter search space. The number of estimators was sampled uniformly between 100 and 400, while maximum tree depth ranged from 4 to 8 to balance model complexity and generalization. Learning rates of 0.01, 0.05, and 0.1 were evaluated alongside subsampling ratios for both observations and features. The scale pos weight parameter was tuned to address class imbalance in the binary delay classification task, with values reflecting the approximate ratio of non-delayed to delayed observations.

The base operational feature set, used across all experimental scenarios, comprises 8 features that capture fundamental scheduling and temporal characteristics independent of weather conditions. This set includes train operational indicators (trainStopping and train id) and cyclical temporal encodings (sine and cosine components for month, hour, and day of week). The complete specifications for these features are provided in Table I (Features 6, 9–12, 15–16, 18).

TABLE III. XGBOOST HYPERPARAMETER SEARCH SPACE
<table><tr><td>Parameter</td><td>Distribution/Values</td></tr><tr><td>n_estimators</td><td>Uniform integer [100, 400]</td></tr><tr><td>max_depth</td><td>Uniform integer [4, 8]</td></tr><tr><td>learning_rate</td><td>{0.01, 0.05, 0.1}</td></tr><tr><td>subsample</td><td>{0.7, 0.8, 0.9}</td></tr><tr><td>colsample_bytree</td><td>{0.7, 0.8, 1.0}</td></tr><tr><td>scale_pos_weight</td><td>{3.9, 4.9, 5.9}</td></tr></table>

## C. Feature Analysis and Selection

A comprehensive correlation analysis was conducted to identify potential feature redundancies and inform feature selection strategies. Figure 2 presents the correlation matrix for all features in the training dataset, excluding cloud amount which was dropped due to data unavailability at Oulu Asema. We also excluded the derived weather scenarios from this analysis. In the heatmap, highly correlated features are represented by dark red (positive correlation) or dark blue (negative correlation), while similar lighter colors indicate weaker positive or negative correlations, respectively.

The analysis reveals several notable patterns with implications for model design. Among weather variables, gust speed and wind speed exhibit very strong correlation $( r { \overset { - } { = } } 0 . { \overset { - } { 9 } } 7 7 ) .$ indicating near-redundancy between these features. Similarly, dew-point temperature shows extremely high correlation with air temperature (r = 0.966), reflecting the thermodynamic relationship between these variables.

Seasonal patterns emerge clearly in the temperature-related correlations. Air temperature correlates negatively with month encoding components (month sin: r = −0.549; month cos: r = −0.678), capturing Finland’s pronounced seasonal temperature variation. Snow depth exhibits strong positive correlation with month sin (r = 0.805), and strong negative correlation with air temperature (r = −0.658) and dew-point temperature (r = 0.663), reflecting winter accumulation patterns. Relative humidity shows moderate positive correlation with month cos (r = 0.490) and negative correlation with air temperature (r = −0.369).

Atmospheric pressure demonstrates moderate negative correlations with wind speed (r = −0.287) and gust speed (r = −0.295), aligning with meteorological principles where low-pressure systems are associated with stronger winds. Horizontal visibility correlates negatively with relative humidity (r = −0.246), reflecting reduced visibility during humid conditions such as fog or precipitation events.

![](images/981929c4cf0873731dc0ed59d356059b05daa14ea0443ee604bd684ab60ce6fb.jpg)  
Fig. 2. Correlation heatmap of operational and weather features in the training dataset.

Notably, operational features (trainStopping, cyclical time encodings, train id) show weak correlations with weather variables, with most coefficients below |0.1|. This independence suggests that weather features provide complementary information not captured by operational variables alone, supporting the hypothesis that meteorological data can enhance delay prediction beyond what temporal and operational patterns provide.

## D. Training Scenarios

Based on the correlation analysis presented in Figure 2, feature selection was performed to reduce redundancy while preserving predictive information. Highly correlated variables were excluded. The commercialStop feature was removed due to its redundancy with trainStopping. Additionally, gust speed and dew-point temperature were excluded from all training scenarios due to their high correlations with wind speed and air temperature, respectively, to mitigate multicollinearity effects.

Based on this analysis, three feature configuration strategies were evaluated:

1) Full weather features set: Comprehensive feature set combining nine instant meteorological observations with 10 weather category scenarios, alongside operational features.

2) Instant weather observations only: Operational features combined exclusively with nine instant meteorological observations only.

3) Weather category scenarios only: Operational features paired solely with the 10 derived weather scenario categories listed in Table II.

Table IV summarizes the feature composition for each training scenario. All configurations utilize nine operational features, comprising train stopping status, cyclical temporal encodings, day of month, and train identifier.

## IV. TRAINING RESULTS AND DISCUSSION

Because our target variable is numeric (train delay in minutes), the metrics employed in this analysis are defined as follows:

R² (Coefficient of Determination): Measures the proportion of variance in the target variable explained by the model, with values closer to 1.0 indicating superior explanatory power.

RMSE (Root Mean Square Error): Quantifies the standard deviation of prediction residuals, penalizing

TABLE IV. FEATURE COMPOSITION ACROSS TRAINING SCENARIOS
<table><tr><td>Feature Category</td><td>Scenario 1 Full Set</td><td>Scenario 2 Instant Only</td><td>Scenario 3 Categories Only</td></tr><tr><td>Operational Features</td><td>6, 9–12, 15-16, 18</td><td>6, 9–12, 15–16, 18</td><td>6, 9–12, 15–16, 18</td></tr><tr><td>Instant Weather Observations</td><td>19–20, 22–23, 25–29</td><td>19–20, 22–23, 25–29</td><td>一</td></tr><tr><td>Weather Category Scenarios</td><td>30-39</td><td>1</td><td>30-39</td></tr><tr><td>Total Features</td><td>28</td><td>18</td><td>19</td></tr></table>

larger errors more heavily due to the squared term.   
Expressed in minutes.

MAE (Mean Absolute Error): Represents the average magnitude of prediction errors without considering their direction, providing a linear and interpretable measure in the same units as the target variable (minutes).

WMAPE (Weighted Mean Absolute Percentage Error): Expresses the total absolute error as a percentage of total actual values, offering a scale-independent assessment particularly useful when comparing across datasets of different magnitudes. Unlike the traditional MAPE, which divides by individual actual values and becomes undefined or unreliable when delays approach zero, WMAPE aggregates all actual values in the denominator. This distinction is critical in railway delay prediction, where trains frequently operate on schedule (zero delay) or experience minimal delays. WMAPE avoids the instability and asymmetry inherent in MAPE by preventing division by near-zero values.

Figure 3 presents the evolution of these four metrics across training iterations for each feature scenario. The results demonstrate consistent performance improvements across all scenarios as the number of iterations increases. The which incorporates only the meaningful weather categories, consistently outperforms the other two scenarios across all metrics. At 100 iterations, this scenario achieves an $\mathsf { R } ^ { 2 }$ of approximately 0.78, compared to roughly 0.70 for other two scenarios. This represents a meaningful improvement in explanatory power, suggesting that engineered weather category features contribute more to delay prediction accuracy than raw weather observations alone.

Examining the error metrics reveals similar patterns. The RMSE for Scenario 3 decreases from approximately 9.9 minutes at 10 iterations to 8.5 minutes at 100 iterations, whereas Scenarios 1 and 2 converge to approximately 9.9 minutes. The MAE follows a comparable trajectory, with Scenario 3 reaching approximately 3.7 minutes compared to 4.1 minutes for the baseline scenarios. The WMAPE metric shows that Scenario 3 achieves roughly 50.5% at convergence, representing a notable improvement over the 56.5% observed for Scenarios 1 and 2.

A notable observation is the convergence behavior of the models. All scenarios exhibit rapid improvement during the initial iterations, with the most substantial gains occurring between 20 and 50 iterations. Scenarios 1 and 2 plateau by approximately 50 iterations, beyond which additional training yields negligible improvement. Scenario 3 continues to improve modestly until approximately 80 iterations, after which performance stabilizes across all metrics. Since doubling the iteration count from 50 to 100 yields only marginal additional gains, the 50–80 iteration range represents a practical stopping point that balances computational cost against predictive performance. Furthermore, Scenarios 1 and 2 exhibit nearly identical performance throughout the iteration range, indicating that the features distinguishing these scenarios contribute minimally to predictive accuracy compared to the weather variables introduced in Scenario 3.

## A. Comparison with Related Work

Direct numerical comparison with Huang et al. [7] requires acknowledging key methodological differences. Their evaluation partitioned delays into short (4 − 30 minutes) and long (> 30 minutes) categories and reported metrics per partition, whereas our approach trains and evaluates on the full continuous distribution of delay values. Their routes also exhibit considerably lower delay variability, with average delays of 3.12 and 1.10 minutes respectively, while Finland’s 5, 915 km network experiences substantially higher variability, with the proportion of delayed trains reaching approximately 27.5% in winter compared to 18% in summer, a near twofold seasonal difference driven largely by severe Nordic weather conditions.

Our Scenario 3, XGBoost model achieved MAE of 3.7 minutes, RMSE of 8.5 minutes, and $\mathbf { R } ^ { 2 }$ of 0.78. While this MAE is higher than the 0.659 minutes reported by Huang et al. [7] for short delays, their evaluation targeted a narrower and less variable delay range on high-speed routes, whereas our model operates on the full continuous delay distribution of a network experiencing severe winter conditions. Given these substantially more challenging operating conditions, the results demonstrate competitive performance and confirm that engineered weather category, rather than raw meteorological observations, constitute effective inputs for ML-based delay prediction across distinct railway contexts and climatic conditions.

## B. Generalization Beyond Oulu Station and Limitations

Although evaluation is restricted to Oulu central station, the proposed weather categorization is designed to generalize. The thresholds reflect physical mechanisms of railway disruption, such as rail adhesion loss near the freezing band, catenary stress under high winds, and signalling impairment under low visibility, which are invariant to geographic location. The same definitions therefore remain operationally meaningful in milder climates, where category frequencies simply shift toward rain, fog, or black-ice conditions. Binary indicators are also more robust to domain shift than raw continuous observations, since the operational regime carries a consistent signal regardless of the underlying meteorological distribution. The framework draws inspiration from pan-European transport weather research [16], supporting its applicability across heterogeneous IoT sensor deployments.

This study has four explicit limitations. (i) Empirical validation across additional stations and railway networks remains future work. (ii) Only XGBoost is benchmarked; alternative ML and DL architectures are not assessed. (iii) Each train segment is matched to its nearest FMI station via the Haversine distance, ignoring micro-climatic variability and multi-station interpolation. (iv) Reliable wireless backhaul from FMI sensors is assumed, with missing observations treated as data gaps rather than communication failures, an assumption that may not hold in low-coverage arctic deployments.

## V. CONCLUSION

This paper presented baseline ML experiments for train delay prediction using the FI-TW dataset, evaluating three feature configuration strategies with XGBoost models at Oulu central station. The experimental results demonstrate that employing derived weather category scenarios rather than raw meteorological observations consistently outperforms both the full feature set and instant weather observations only across all evaluation metrics. At convergence, this achieved an R<sup>2</sup> of 0.78, RMSE of 8.5 minutes, and MAE of 3.7 minutes, representing approximately 14% improvement in RMSE and 10% improvement in MAE compared to the alternative configurations. Beyond predictive performance, training the model on derived weather category offers practical advantages for deployment. The weather category features are encoded as binary indicators rather than continuous floating-point values, substantially reducing memory requirements and storage footprint. This efficiency becomes particularly relevant for large-scale railway networks where millions of observations must be processed and stored. Building on the limitations identified above, future research should extend the evaluation to additional stations, benchmark DL architectures such as recurrent neural networks and transformers against the XG-Boost baseline, and jointly model meteorological inputs with the reliability of their wireless backhaul.

![](images/cc64afb6339f8ef9b8cfb924c6d8518659c28a4a5db9484cfb9b1707b566ff52.jpg)  
Full weather data Instant weather observations only Derived weather category only  
Fig. 3. Performance metrics (R², RMSE, MAE, WMAPE) across training iterations for the three feature scenarios.

## ACKNOWLEDGMENT

This work was supported by European Union through the Interreg Aurora project ENSURE-6G (Grant Number: 20361812) and the Research Council of Finland (former Academy of Finland) through the 6G Flagship program (Grant Number: 369116)

## REFERENCES

[1] K. Gkoumas et al., “Rail transport research and innovation in europe: an abessment based on recent european union projects,” in Transportation Research Procedia, vol. 72. Elsevier B.V., 2023, pp. 3633–3640.

[2] C.-X. Wang et al., “On the road to 6G: Visions, requirements, key technologies, and testbeds,” IEEE Communications Surveys & Tutorials, vol. 25, no. 2, pp. 905–974, 2023.

[3] G. Mukunzi and C. W. Palmqvist, “The impact of railway incidents on train delays: A case of the swedish railway network,” Journal of Rail Transport Planning and Management, vol. 30, 6 2024.

[4] F. Monsuur, M. Enoch, M. Quddus, and S. Meek, “Modelling the impact of rail delays on passenger satisfaction,” Transportation Research Part A: Policy and Practice, vol. 152, pp. 19–35, 10 2021.

[5] N. Davari et al., “A survey on data-driven predictive maintenance for the railway industry,” Sensors, vol. 21, p. 5739, 8 2021. [Online]. Available: https://www.mdpi.com/1424-8220/21/17/5739

[6] M. J. Pappaterra, F. Flammini, V. Vittorini, and N. Besinovi ˇ c, “A´ systematic review of artificial intelligence public datasets for railway applications,” Infrastructures, vol. 6, 10 2021.

[7] P. Huang et al., “Modeling train timetables as images: A cost-sensitive deep learning framework for delay propagation pattern recognition,” Expert Systems with Applications, vol. 177, p. 114996, 9 2021.

[8] Finnish Transport Infrastructure Agency. (2024) Railway Network. Website. [Online]. Available: https://vayla.fi/en/transport-network/railw ay-network

[9] A. Lotfi and M. S. Virk, “Railway operations in icing conditions: a review of issues and mitigation methods,” Public Transport, vol. 15, pp. 747–765, 10 2023.

[10] Finnish Transport Infrastructure Agency. (2025) Railway Statistics: Punctuality in Long Distance and Commuter Traffic. [Online]. Available: https://vayla.fi/en/transport-network/data/statistics/railway-s tatistics

[11] V. Borin and J. M. d. S. Sant’Ana, “Finland integrated train-weather dataset (fi-tw),” https://www.kaggle.com/datasets/viniborin/finland-int egrated-train-weather-dataset-fi-tw, 2025, accessed: 2025-12-12.

[12] World Meteorological Organization, “Guide to meteorological instruments and methods of observation,” World Meteorological Organization, Geneva, Switzerland, Tech. Rep. WMO-No. 8, 2008, part II. Observing Systems. [Online]. Available: https://www.wmo.int

[13] SKYbrary. (2024) Weather observations at aerodromes. European Aviation Safety Agency (EASA) Online Resource. Accessed: November 13, 2025. [Online]. Available: https://skybrary.aero/articles/weather-o bservations-aerodromes

[14] R. W. Sinnott, “Virtues of the haversine,” Sky and Telescope, vol. 68, no. 2, pp. 158–159, Dec. 1984. [Online]. Available: https://ui.adsabs.harvard.edu/abs/1984S&T....68R.158S/abstract

[15] L. Cai et al., “Traffic transformer: Capturing the continuity and periodicity of time series for traffic forecasting,” Transactions in GIS, vol. 24, pp. 736–755, 6 2020.

[16] A. Vajda et al., “Severe weather affecting European transport systems: the identification, classification and frequencies of events,” Natural Hazards, vol. 72, no. 1, pp. 169–188, 2014.

[17] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794.