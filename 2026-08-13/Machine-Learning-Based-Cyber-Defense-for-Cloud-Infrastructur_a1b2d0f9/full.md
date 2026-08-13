# Machine Learning-Based Cyber Defense for Cloud Infrastructure: An Adaptive Deep Q-Network Architecture for Intelligent Intrusion Detection and Automated Threat Mitigation

Md Yassir Mottalib¹, Md Yousuf², Eklachur Rahman Bhuiyan³, S M Ahsan Habib⁴, Sonjoy Kumar Dey⁵, Md. Salahuddin Gazi⁶, Molay Kumar Roy⁷, Asaduzzaman Anik⁸

<sup>1</sup>Master of Science in Information System Technology, Wilmington University, USA

<sup>2</sup>Master's of Science in Data Analytics, College of Sciences and Technology, University of Houston-Downtown, USA

<sup>3</sup>Master of Science in Information Technology, Washington University of Science and Technology, USA <sup>4</sup>Department of Electrical Engineering and Computer Science, South Dakota School of Mines & Technology, USA

<sup>5</sup>McComish, Department of Electrical Engineering and Computer Science, South Dakota State University, USA

<sup>6</sup> Master of Science in Information Technology, Washington University of Science and Technology, USA <sup>7</sup>Ms in Digital Marketing & Information Technology Management, St. Francis College, USA

<sup>8</sup>Master of Business Administration (MBA) in Management, Stanton University, Los Angeles, California

Abstract—With the increasing complexity of cyber assaults in cloud environments, adaptable security solutions are needed that can support real-time detection and autonomous response. In this paper, we propose a reinforcement learning-based dynamic cyber defense framework. We deploy a Deep Q-Network (DQN) to train effective defensive strategies to counteract the evolving cyberattacks. We leverage the CICIDS2017 dataset for model creation and the UNSW-NB15 dataset for external validation, involving preprocessing of data, feature engineering, and adaptive policy learning. We compare the proposed DQN with decision tree, support vector machine, random forest, XGBoost, and multilayer perceptron models. The proposed DQN achieves an accuracy of 99.72%, a precision of 99.68%, a recall of 99.65%, an F1-score of 99.66%, and an ROC-AUC of 0.999, while the false positive rate is 0.31%, the false negative rate is 0.35%, and the detection latency is 15 ms. The framework achieved 99.54% attack mitigation rate, demonstrating strong adaptive and real-time defensive capabilities. These results demonstrate the potential of reinforcement learning as a powerful and scalable approach for autonomous cybersecurity in modern cloud environments.

Keywords: Reinforcement Learning, Deep Q-Network, Dynamic Cyber Defense, Cloud Security, Intrusion Detection, Machine Learning, Adaptive Cybersecurity, CICIDS2017, and UNSW-NB15.

## Introduction

Artificial Intelligence (AI) has revolutionized the cybersecurity space by enabling smart systems to detect advanced attack patterns and respond to emerging threats with less human involvement. As more firms move their apps, databases, and core business functions to cloud computing, the pool of possible targets has widened significantly, making cloud platforms a prime target for hackers. While cloud services provide benefits such as scalability, flexibility, cost savings, and consistent availability, they also introduce new security issues, including distributed denial-ofservice (DDoS) attacks, ransomware, privilege escalation, insider threats, malware, advanced persistent threats (APTs), and zero-day exploits.

Conventional security solutions such as signature-based intrusion detection, rule-based firewalls, and static access control are unable to identify sophisticated attacks, as they are based on predefined signatures and rules that require manual updates (Sarker et al., 2020; Alzahrani et al., 2022). Researchers are turning to smarter solutions with machine learning (ML) and deep learning (DL) due to the rapid growth of new internet dangers. Supervised learning models such as decision trees, random forests, support vector machines (SVM), artificial neural networks (ANN), and extreme gradient boosting (XGBoost) have been shown to perform very well in detecting intrusions by analyzing historical network data (Shone et al., 2018; Vinayakumar et al., 2019). However, these techniques are fundamentally limited because they need to be retrained when new attack techniques are discovered. Cyberattacks are always developing; thus, models trained on outdated data may become ineffective against new threats.

Reinforcement learning (RL) has recently shown promising results as a solution for autonomous cybersecurity. RL allows smart agents to learn efficient defense methods not only by learning from past examples but also by learning from the consequences through continual interaction with their environment. RL differs from supervised learning since it considers cybersecurity as a sequential decision-making problem, where the system observes the state of the network, decides on an action, receives a reward or a penalty as feedback, and improves its strategy over time (Sutton & Barto, 2018). Such adaptive learning is especially beneficial in cloud situations where attack patterns might alter on the go. Deep Reinforcement Learning (DRL) techniques such as Deep Q-Networks (DQN), Double Deep Q-Networks (DDQN), Deep Deterministic Policy Gradient (DDPG), and Proximal Policy Optimization (PPO) have demonstrated their capability to solve highly complex problems with a large amount of data (Mnih et al., 2015; Arulkumaran et al., 2017). These methods have been utilized in intrusion detection, firewall management, blocking of malware, adaptive access control, software-defined networking (SDN), security of the Internet of Things (IoT), and protection of cloud resources. “They’re a good fit for automated cyber defense because they can continuously refine security policies and keep false alarms low.”

Cloud computing introduces other security issues such as virtualization, multi-tenancy, rapid resource shifting, container orchestration, and services distributed over several sites. These aspects make the cloud systems more complex and provide the attackers with additional opportunity to uncover the holes at multiple tiers (Singh & Chatterjee, 2017). This creates a significant need for sophisticated cybersecurity frameworks that can monitor cloud traffic continuously, detect suspicious activity in real-time, and respond quickly without disruption to lawful processes. To address these issues, we propose a dynamic cyber protection architecture for cloud infrastructures based on reinforcement learning. Our technique relies on a Deep Q-Network (DQN) agent that learns optimal protection strategies through interaction with the cloud network. Our methodology aims to improve detection accuracy, reduce false positives, accelerate response times, and build more resilient cloud systems by leveraging deep feature engineering, real-world cybersecurity datasets, and adaptive reinforcement learning. Its effectiveness has been tested on typical intrusion detection datasets and compared with popular supervised machine learning algorithms to show its benefits for smart cloud security.

## Literature Review

Cyberattacks on cloud computing systems have become more sophisticated, and study into cybersecurity has become a hot topic. In recent years, security solutions have progressed well beyond simple signature-based detection into smarter data-driven technology that can actually learn and adapt to new attack approaches. The fight against invaders is more advanced than ever using machine learning, deep learning, and reinforcement learning.

Cyberattacks on cloud computing systems have become more sophisticated, and study into cybersecurity has become a hot topic. In recent years, security solutions have progressed well beyond simple signature-based detection into smarter data-driven technology that can actually learn and adapt to new attack approaches. The fight against invaders is more advanced than ever using machine learning, deep learning, and reinforcement learning.

Cyberattacks on cloud computing systems have become more sophisticated, and study into cybersecurity has become a hot topic. In recent years, security solutions have progressed well beyond simple signature-based detection into smarter data-driven technology that can actually learn and adapt to new attack approaches. The fight against invaders is more advanced than ever using machine learning, deep learning, and reinforcement learning.

Then deep learning came along, and it pushed things even further. Deep learning algorithms build sophisticated layered structures from enormous network data rather than relying on specialists to design features. For example, Shone et al. (2018) used a combination of sophisticated autoencoders and Random Forests to improve detection rates and reduce manual feature engineering, while Vinayakumar et al. (2019) demonstrated that deep neural networks can surpass traditional machine learning on contemporary datasets by detecting subtle, nonlinear relationships in network traffic.

The availability of accurate, up-to-date datasets has also sped up progress. Some limitations exist in previous datasets such as KDD Cup 1999 and NSL-KDD; thus, researchers generated new datasets: CICIDS2017 (Sharafaldin et al., 2018) has updated attack scenarios, realistic background traffic, and a broad variety of data attributes, so it better represents real-world networks. The UNSW-NB15 dataset (Moustafa and Slay, 2015), featuring modern attack types and mixed traffic, is a key baseline for assessing today’s intrusion detection systems.

But despite these advances, supervised learning has a fundamental limitation: it does not have the ability to self-adapt into new assault techniques. This has pushed researchers toward reinforcement learning, which allows systems to learn and evolve their defenses through experience, not just static data.

But despite these advances, supervised learning has a fundamental limitation: it does not have the ability to self-adapt into new assault techniques. This has pushed researchers toward reinforcement learning, which allows systems to learn and evolve their defenses through experience, not just static data.

Nguyen and Reddi [17] evaluated the field and concluded that reinforcement learning has advanced adaptive intrusion detection, malware protection, automated penetration testing, dynamic security techniques, and software-defined networking (SDN). Their research reveals how reinforcement learning allows cybersecurity systems to adapt their defenses on the fly rather than using inflexible rules.

For example, reinforcement learning is particularly beneficial in the cloud for resource management, access control, moving virtual machines, and autonomous threat response. For example, Alavizadeh et al. (2022) showed that deep reinforcement learning may significantly improve cloud systems’ response to attacks by lowering false alarms and reacting more quickly than traditional machine learning. SDN has become another hot spot for reinforcement learning, as its separation of control and data planes gives a global view of the network. This lets reinforcement learning agents update routes, firewall settings, and filtering strategies in real time, boosting SDN security through automated detection and response (Xiao et al., 2020).

The separation of control and data planes in SDN provides a global perspective of the network, and it has emerged as yet another hot point for reinforcement learning. This enables reinforcement learning agents to adjust routes, firewall settings, and filtering techniques in real-time, thus improving SDN security through automated detection and reaction (Xiao et al., 2020).

But there are gaps in research. Most prior work either only considers a limited number of attack types or only focuses on restricted components such as intrusion detection or firewall rules, instead of establishing all-in-one adaptive security frameworks for cloud systems. Many models are tested using old or generated data that does not represent real-world cloud traffic today.

Another problem is that there are no fair side-by-side comparisons of reinforcement learning vs. the best-performing supervised machine learning algorithms on the identical test settings. Many articles present findings of reinforcement learning without a strict comparison to approaches such as Random Forests, XGBoost, or deep neural networks. Therefore, the practical gains of reinforcement learning for real cloud cyber security are not yet properly quantified. To address these deficiencies, we are proposing a strong reinforcement learning-based cyber protection architecture. Our method uses state-of-the-art benchmark datasets, feature engineering, Deep Q-Network optimization, and extensive comparison with leading supervised learning models. The aim is not just to detect attacks with a high degree of confidence but to develop a defense that learns, adapts, and reacts on its own as new threats emerge in cloud environments. Our method combines intelligent decision-making with real-time security tracking to shift the needle for true autonomous cyber protection for the modern cloud.

## Methodology

## Research Methodology

In this research, we introduced a dynamic cyber defense framework for cloud systems that uses reinforcement learning (RL) to enhance how attacks are detected and handled. Unlike traditional intrusion detection systems, which stick to fixed rules, our approach allows the defense strategy to evolve—learning and adapting by interacting directly with the cloud environment and responding to new attack methods as they arise. Our methodology draws on real-world network traffic, employs advanced feature engineering, builds a specialized RL environment, and thoroughly tests performance. The overall process includes collecting data, cleaning and prepping it, extracting and engineering features, developing the RL model, and then evaluating how well it works.

## Data Collection

We obtained network intrusion data from well-established public cybersecurity sources to encourage openness, reproducibility, and direct comparison with previous work. The Canadian Institute for Cybersecurity's CICIDS2017 dataset, which is also available through Kaggle, is the main one that supports our research. The incorporation of realistic benign and malicious traffic collected from a modern enterprise context has made CICIDS2017 a major standard in intrusion detection and cybersecurity research.

This dataset covers a wide range of attack scenarios, including DDoS, brute-force invasions, webbased exploits, port scanning, SQL injection, XSS, and normal network activity. It also includes botnet and infiltration operations. We use the CICFlowMeter program to extract over 70 statistical features and annotate them to each network flow. This makes CICIDS2017 a great tool for cyber defense research driven by reinforcement learning, as it records flow duration, packet-level characteristics, protocol specifics, timing patterns, header data, and aspects of network behavior that are not dependent on the content. We used the UNSW-NB15 dataset, which is available in the UCI Machine Learning Repository, to further test our framework and increase the model's resilience and generalizability. UNSW-NB15 is a great tool for testing adaptive defenses because it mimics real-world network settings and includes a wide variety of modern attack types.

Table 1. Dataset Description
<table><tr><td colspan="1" rowspan="1">Dataset</td><td colspan="1" rowspan="1">Repository</td><td colspan="1" rowspan="1">Number     ofRecords</td><td colspan="1" rowspan="1">Features</td><td colspan="1" rowspan="1">Attack Categories</td><td colspan="1" rowspan="1">Purpose</td></tr><tr><td colspan="1" rowspan="1">CICIDS2017</td><td colspan="1" rowspan="1">KaggleCanadianInstitute    forCybersecurity</td><td colspan="1" rowspan="1">Approximately2.83 million</td><td colspan="1" rowspan="1">78</td><td colspan="1" rowspan="1">DDoS, DoS, BruteForce, Botnet, PortScan, Web Attack,Infiltration,Heartbleed,Benign</td><td colspan="1" rowspan="1">Primarydataset forRLtraining</td></tr><tr><td colspan="1" rowspan="1">UNSW-NB15</td><td colspan="1" rowspan="1">UCI MachineLearningRepository</td><td colspan="1" rowspan="1">257,673</td><td colspan="1" rowspan="1">49</td><td colspan="1" rowspan="1">Fuzzers,Exploits,DoS,      Generic,Worms, Shellcode,</td><td colspan="1" rowspan="1">Externalvalidationdataset</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Analysis,Backdoor,Reconnaissance</td><td colspan="1" rowspan="1"></td></tr></table>

The diversity of attack categories in both datasets allowed us to construct an RL environment capable of learning optimal defense strategies under various cyber threat scenarios encountered in cloud infrastructures.

## Data Preprocessing

To improve data quality and reduce bias, we preprocessed the datasets before model building. Removing duplicates and missing values was systematic. We checked continuous numerical characteristics for infinite values and outliers, commonly generated by packet capture abnormalities. Records with significant corruption were omitted from analysis, whereas appropriate statistical estimations imputed infinite values.

Since reinforcement learning techniques are sensitive to feature scaling, all numerical variables were normalized using min-max scaling to map each feature to a standard range between zero and one. Policy optimization required this normalization for consistent gradient updates. Protocol kinds, service categories, and network conditions were converted to numbers using onehot and label encoding. Attack labels were translated into distinct action-state categories to identify regular traffic from numerous attack classes.

We used the Synthetic Minority Oversampling Technique (SMOTE) on minority attack categories while protecting innocuous traffic to overcome intrusion detection datasets' class imbalance. The reinforcement learning agent did not establish majority-class-favored defense policies with this method. Finally, we randomly separated the data into 70%, 15%, and 15% training, validation, and test sets, confirming that attack categories were constant across all groups.

## Feature Extraction

Finding the telltale signs of malicious network activity relies heavily on feature extraction. We methodically extracted statistical features at the flow level that indicate communication trends over time, rather than relying only on raw data at the packet level. Our feature extraction process encompassed the following fields: flow duration, forward and backward packet counts, total packet length, average packet size, inter-arrival times, packet transmission rates, flow bytes, packets per second, protocol type, TCP flag statistics, header lengths, idle and active periods, source and destination ports, and measures of packet variance, standard deviation, and retransmissions. By collecting features including traffic volume, directional flow, stability of connections, temporal dynamics, and protocol engagement, these properties give a multi-faceted description of network activity. The reinforcement learning agent can tell the difference between everyday cloud operations and complex cyber threats with the help of these rich representations. This feature extraction procedure is also crucial since cloud settings generate network traffic that is both complicated and high-dimensional. It does double duty by lowering computing demands while simultaneously preserving the essential data needed for adaptive threat detection.

## Feature Engineering

After feature extraction, we performed focused feature engineering to boost the representational capability of the learning model. Highly associated features were determined by Pearson correlation analysis, and redundant variables (correlation coefficient > 0.90) were progressively deleted to alleviate multicollinearity. Feature significance was then estimated using Random Forest and Extreme Gradient Boosting (XGBoost) algorithms. Features with minimal predictive value were discarded to provide a more streamlined and computationally efficient feature collection. We designed other behavioral features such as ratios of packet rates, bidirectional traffic ratios, flow density, scores for protocol utilization, mean connection duration, session entropy, packet burstiness, and temporal anomaly indices to improve the learning process. These additional variables gave more contextual information into network behavior in the cloud environment. Then, we applied Principal Component Analysis (PCA) for dimensionality reduction with the guarantee of preserving 95% of the variance of the original dataset. This step dramatically reduced the computational complexity and hastened the convergence of reinforcement learning with little effect on the predicted accuracy. The resulting feature space consisted of both original statistical attributes and newly generated behavioral markers, thus providing an informative and robust state representation for the reinforcement learning model.

## Model Development

We characterized dynamic cyber protection as a sequential decision-making problem using reinforcement learning. Our architecture uses the cloud infrastructure as the environment; states are observable network traffic aspects, actions are real-time cybersecurity decisions, and the reward function assesses attack mitigation. The reinforcement learning agent assesses the network's current state using extracted features and picks the best defense action at each step. These measures could allow regular traffic, block suspicious connections, isolate infected virtual machines, throttle suspicious data flows, issue security alarms, or trigger deeper traffic examination.

Our reward scheme promotes attack detection while reducing false alarms and resource waste. An agent receives positive rewards for detecting and interrupting malicious activities. False positives, missed attacks, delayed replies, and overly aggressive defenses result in penalties. We choose the Deep Q-Network (DQN) as our main reinforcement learning technique since it handles complex, high-dimensional data well. An input layer for the designed feature set, many fully linked hidden layers with ReLU activation, and an output layer that predicted Q-values for each action comprised the neural network. We used experience replay memory to stabilize learning and reduce bias from consecutive state observations by randomly sampling past encounters during training. To improve model convergence, we updated a different target network frequently. We compared our DQN-based framework against Random Forest, Support Vector Machine (SVM), Decision Tree, XGBoost, and Multilayer Perceptron to assess our RL method. Our reinforcement learning approach changes its security strategy to new attack behaviors, making it ideal for cloud environments' fast-changing nature. These baseline models categorize intrusions using static rules. Training continued until stable cumulative rewards and a successful defensive policy were achieved.

## Model Evaluation

We used classification metrics and reinforcement learning performance measures to evaluate the framework. Classification performance metrics include accuracy, precision, recall, F1-score, ROC-AUC, MCC, FPR, and FNR. These metrics measure the model's ability to identify malicious traffic from cloud activity. Reinforcement learning performance measures include cumulative reward, average episodic reward, convergence rate, policy stability, learning efficiency, action selection accuracy, average response time, attack mitigation rate, and defense success ratio. These RL-specific indicators show the cyber defense framework's adaptability and long-term decisionmaking.

Created confusion matrices provide classification results for distinct assault categories. At different choice thresholds, we examined the receiver operating characteristic (ROC) and precision–recall curves to assess discrimination. We used five-fold cross-validation and repeated reinforcement learning training with various random seeds in distinct runs to assure statistical safety. Average performance numbers with standard deviations showed the framework's robustness and stability. With this comprehensive methodological framework, we created an adaptive reinforcement learning-based cyber defense system that continuously learns optimal security strategies to protect cloud infrastructures from evolving cyber threats while maintaining high detection accuracy, low false alarm rates, and efficient resource utilization.

## Result

## Experimental Results

Our suggested dynamic cyber defense system was tested using the CICIDS2017 dataset for training and the UNSW-NB15 dataset for external validation using reinforcement learning. Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), Extreme Gradient Boosting (XGBoost), and Multilayer Perceptron (MLP) were some of the frequently used supervised machine learning algorithms that were compared with the Deep Q-Network (DQN) reinforcement learning model. To maintain objectivity, all models were trained with the same preprocessed datasets and tested using the same training-testing partitions. Criteria for performance evaluation included recall, accuracy, precision, F1-score, ROC-AUC, detection delay, false positive rate (FPR), and false negative rate (FNR).

The testing results demonstrate that the suggested RL framework outperformed the competition across the board in nearly all performance parameters. By interacting with the cloud environment, the reinforcement learning agent adaptively responded to changing attack patterns by continuously improving its cyber protection policy, in contrast to standard supervised models with fixed classifications. This strategy is well-suited to ever-changing cloud computing environments since the DQN agent effectively reduced the need for needless defensive operations while increasing the efficiency of threat detection.

Random Forest and XGBoost, which used ensemble learning to detect targets, performed well in categorization. When new attack patterns arose, they needed to be retrained since they could not adapt. In the meantime, the reinforcement learning system might optimize its policy based on environmental feedback, improving long-term cybersecurity.

Performance Comparison of Machine Learning and Reinforcement Learning Models

Table 1. Performance Comparison of Different Models
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy(%)</td><td rowspan=1 colspan=1>Precision(%)</td><td rowspan=1 colspan=1>Recall(%)</td><td rowspan=1 colspan=1>F1-Score(%)</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>FPR(%)</td><td rowspan=1 colspan=1>FNR(%)</td><td rowspan=1 colspan=1>DetectionTime (ms)</td></tr><tr><td rowspan=1 colspan=1>Decision Tree</td><td rowspan=1 colspan=1>96.84</td><td rowspan=1 colspan=1>96.20</td><td rowspan=1 colspan=1>96.11</td><td rowspan=1 colspan=1>96.15</td><td rowspan=1 colspan=1>0.972</td><td rowspan=1 colspan=1>2.90</td><td rowspan=1 colspan=1>3.89</td><td rowspan=1 colspan=1>18</td></tr><tr><td rowspan=1 colspan=1>SupportVectorMachine</td><td rowspan=1 colspan=1>97.58</td><td rowspan=1 colspan=1>97.12</td><td rowspan=1 colspan=1>97.05</td><td rowspan=1 colspan=1>97.08</td><td rowspan=1 colspan=1>0.981</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>2.95</td><td rowspan=1 colspan=1>26</td></tr><tr><td rowspan=1 colspan=1>RandomForest</td><td rowspan=1 colspan=1>99.18</td><td rowspan=1 colspan=1>99.05</td><td rowspan=1 colspan=1>99.01</td><td rowspan=1 colspan=1>99.03</td><td rowspan=1 colspan=1>0.996</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>20</td></tr><tr><td rowspan=1 colspan=1>XGBoost</td><td rowspan=1 colspan=1>99.36</td><td rowspan=1 colspan=1>99.31</td><td rowspan=1 colspan=1>99.24</td><td rowspan=1 colspan=1>99.27</td><td rowspan=1 colspan=1>0.997</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>19</td></tr><tr><td rowspan=1 colspan=1>MultilayerPerceptron</td><td rowspan=1 colspan=1>98.71</td><td rowspan=1 colspan=1>98.56</td><td rowspan=1 colspan=1>98.43</td><td rowspan=1 colspan=1>98.49</td><td rowspan=1 colspan=1>0.992</td><td rowspan=1 colspan=1>1.34</td><td rowspan=1 colspan=1>1.57</td><td rowspan=1 colspan=1>23</td></tr><tr><td rowspan=1 colspan=1>Proposed DeepQ-Network(RL)</td><td rowspan=1 colspan=1>99.72</td><td rowspan=1 colspan=1>99.68</td><td rowspan=1 colspan=1>99.65</td><td rowspan=1 colspan=1>99.66</td><td rowspan=1 colspan=1>0.999</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>15</td></tr></table>

![](images/b52871a31a6abbc42d950df37a941648da184aec7b848a933a395e4f7678ae56.jpg)  
Chart 1: Performance Comparison of Different Models

The results show that the proposed DQN-based reinforcement learning model achieved a total classification accuracy of 99.72%, which is higher than all baseline machine learning algorithms. The model also has the highest precision (99.68%), recall (99.65%), and F1-score (99.66%), which shows excellent ability in detecting malicious network traffic while minimizing both false alarms and missed attacks. In addition, the DQN model achieved the lowest false positive rate (0.31%) and false negative rate (0.35%), which demonstrates reliable decision-making under different attack scenarios.

The detection latency was also reduced to 15 milliseconds, which further proves the applicability of reinforcement learning for real-time cloud security solutions where quick response is necessary.

## Reinforcement Learning Performance Analysis

Besides conventional classification metrics, the reinforcement learning performance was evaluated by cumulative reward, convergence behavior, attack mitigation success, policy stability, and adaptive response capability.

Table 2. Reinforcement Learning Performance Metrics
<table><tr><td rowspan=1 colspan=1>RL Metric</td><td rowspan=1 colspan=1>Obtained Value</td></tr><tr><td rowspan=1 colspan=1>Average Episodic Reward</td><td rowspan=1 colspan=1>982.45</td></tr><tr><td rowspan=1 colspan=1>Maximum Reward</td><td rowspan=1 colspan=1>1000.00</td></tr><tr><td rowspan=1 colspan=1>Average Convergence Episodes</td><td rowspan=1 colspan=1>285</td></tr><tr><td rowspan=1 colspan=1>Policy Stability</td><td rowspan=1 colspan=1>99.1%</td></tr><tr><td rowspan=1 colspan=1>Attack Mitigation Success Rate</td><td rowspan=1 colspan=1>99.54%</td></tr><tr><td rowspan=1 colspan=1>Adaptive Response Accuracy</td><td rowspan=1 colspan=1>99.48%</td></tr><tr><td rowspan=1 colspan=1>Average Decision Time</td><td rowspan=1 colspan=1>15 ms</td></tr><tr><td rowspan=1 colspan=1>Cloud Resource Utilization Efficiency</td><td rowspan=1 colspan=1>96.8%</td></tr></table>

The reinforcement learning agent converged quickly after about 285 training episodes, after which the cumulative rewards were stable with minor fluctuations. The learned policy always selected the optimal defense actions across different attack categories while efficiently using the cloud resources. The high attack mitigation success rate shows that the model effectively prevented malicious activities before they could significantly affect the cloud services.

In addition, the accuracy of the adaptive response confirms that reinforcement learning successfully adapted defensive strategies based on varying network conditions, which is a significant advantage compared to traditional supervised classification models.

## Comparative Analysis

The comparison study clearly proves that reinforcement learning is better in dynamic cloud cybersecurity. Traditional supervised learning algorithms, such as Decision Tree, Support Vector Machine, Random Forest, XGBoost, and Multilayer Perceptron, have been proven to be quite effective when used for intrusion detection because they can learn static decision boundaries from historical data. But these models need to be periodically retrained as new attack signatures or previously unseen dangers appear. This is a limitation that the suggested deep Q-network is overcoming through continual learning from the cloud environment interactions. The reinforcement learning agent doesn't just mimic past attack patterns but rather adapts its protection strategy in real-time depending on input, enabling proactive adaptability to changing cyber-attacks. The Random Forest and XGBoost performed competitively, as both are ensemble learning methods and are robust to noisy data. But they were still inherently reactive models. They are not capable of self-adapting their attack techniques, which makes them less effective in fast-changing cloud environments where zero-day attacks and advanced persistent threats occur often. The Deep Q-Network had good results in the overall performance, which is attributed to its high accuracy rate of intrusion detection, adaptive decision-making, decreased detection latency, lower false-alarm rate, and continuous learning. These features make RL much more appropriate for modern cloud infrastructures that need smart, self-governing cybersecurity.

## Discussion

Empirical findings indicate that reinforcement learning offers significant potential for improving cybersecurity inside cloud computing settings. The suggested framework achieved not only elevated classification precision but also showed the capacity to autonomously acquire effective defensive strategies without the need for manual rule adjustments. A significant benefit of the suggested method is its ability to achieve a balance between security efficacy and operational efficiency. Excessive defensive measures may adversely affect genuine cloud users and result in heightened computational demands. The reinforcement learning agent adeptly navigated this tradeoff by choosing defensive measures that optimized long-term cumulative rewards while circumventing superfluous interventions.

The minimal false positive rate alleviates alert fatigue for security analysts, enabling security operation centers to concentrate on genuine threats. The rapid reaction time facilitates prompt alleviation of cyberattacks prior to substantial harm occurring, hence enhancing overall cloud robustness. These results indicate that reinforcement learning represents a promising avenue for next-generation autonomous cyber defense systems capable of operating effectively in highly dynamic cloud settings.

## Implementation of the Best Model in U.S. Industry

Cloud computing and real-time cybersecurity organizations in the US can use deep Q-Network reinforcement learning. Intelligent security systems for large cloud infrastructures must respond to evolving cyber threats without operator interaction. Healthcare cloud systems for EHRs, medical imaging, telemedicine, and IoMT devices can integrate the proposed solution. Reinforcement learning can monitor cloud traffic, detect ransomware attacks, identify unauthorized access attempts, and isolate compromised systems for patient care. These capabilities meet HIPAA cybersecurity standards. Reinforcement learning can handle account takeover, DDoS, credential stuffing, fraud, and insider threats in banking, insurance, digital payment systems, and investment platforms. Low false positive rates increase FFIEC and PCI DSS compliance and prevent financial transaction disruption.

The recommended architecture can be an intelligent cybersecurity layer for IaaS, PaaS, and SaaS. The reinforcement learning agent optimizes firewalls, network segmentation, virtual machine isolation, intrusion prevention rules, and automated incident response policies as network conditions change. This architecture can safeguard classified cloud environments, defend key infrastructure, and increase government and defense resilience against advanced persistent threats and nation-state attacks. Adjusting policies without human updates is possible with adaptive learning. Manufacturing businesses using Industry 4.0 and IIoT platforms may protect industrial cloud networks, identify abnormal machine connections, and secure connected production environments with the suggested method. Transportation, logistics, energy, telecommunications, and smart city infrastructures benefit from autonomous cyber defensive measures that can adapt to changing attack patterns.

The suggested Deep Q-Network model can be used with SIEM, SOAR, XDR, cloud-native firewalls, endpoint detection and response platforms, and zero-trust architectures. While maintaining cybersecurity infrastructure, continuous surveillance, automatic threat response, dynamic policy optimization, and intelligent resource protection are achievable. Reinforcement learning is a suitable, scalable, and intelligent cybersecurity option for U.S. cloud infrastructures, according to experiments. Cloud-dependent industries like healthcare, finance, government, manufacturing, telecommunications, and others that prioritize cybersecurity resilience benefit from the framework's high detection accuracy, adaptive learning, fast reaction, and low false alarm rates.

## Conclusion

Cloud computing's rapid growth has increased cybersecurity issues, requiring clever protection mechanisms to adapt to new attack techniques. Traditional intrusion detection systems and supervised machine learning algorithms may detect known cyber threats, but their reliance on previous data and frequent retraining limits their defense against unexpected attacks. These constraints necessitate adaptive cybersecurity solutions that make real-time protection decisions autonomously.

We presented a reinforcement learning–based dynamic cyber protection architecture for cloud infrastructure that uses a deep Q-network (DQN) to continually learn optimal security rules from the cloud environment. Public benchmark datasets, data preprocessing, advanced feature engineering, and reinforcement learning enable intelligent attack detection and adaptive response in the proposed framework. To enable proactive and autonomous cyber protection, the reinforcement learning agent continuously updates its decision-making policy based on environmental feedback, unlike static classification methods.

DQN outperformed decision tree, support vector machine, random forest, XGBoost, and multilayer perceptron across numerous assessment measures in experiments. The framework improved accuracy, precision, recall, F1-score, and ROC-AUC, with decreased false positive and false negative rates and shorter detection latency. For real-time cloud security applications, reinforcement learning-specific assessment metrics showed the model's capacity to swiftly converge, maintain stable policies, and mitigate attacks with high success rates. Reinforcement learning also allows continual adaptability to new cyber hazards without retraining models, making it better than supervised learning. With dynamic workloads, multi-tenant designs, virtualization, and increasingly complex attack methodologies, current cloud systems require these adaptive capabilities.

The framework's scalable and intelligent cybersecurity solution improves cloud infrastructure resilience and efficiency. Healthcare, financial services, government organizations, cloud service providers, manufacturing, telecommunications, and others that use secure cloud computing environments could implement the suggested framework. Compliance with SIEM, SOAR, XDR, and Zero Trust architectures would enable autonomous threat detection and adaptive incident response. This reduces manual involvement and improves cybersecurity. The proposed framework works well; however, it needs improvement. Advanced reinforcement learning methods, including Double Deep Q-Network (DDQN), Dueling DQN, Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Multi-Agent Reinforcement Learning platforms such(MARL), may be investigated to improve learning efficiency and scalability. Real-time cloud telemetry, federated learning, explainable artificial intelligence (XAI), graph neural networks, and LLM-assisted threat intelligence can improve autonomous cyber defense system interpretability, robustness, and deployment. The suggested paradigm would be more applicable if validated on enterprise cloud environments and commercial cloud platforms as Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP).

## Reference:

[1] Alavizadeh, H., Al-Rakhami, M. S., Gumaei, A., Hassan, M. M., Fortino, G., & Aloqaily, M. (2022). Deep reinforcement learning for cybersecurity: A survey. IEEE Access, 10, 135824– 135847. https://doi.org/10.1109/ACCESS.2022.3230298

[2] Alzahrani, A., Alenazi, M., Alshammari, A., & Alfakeeh, A. (2022). Machine learning approaches for cloud cybersecurity: A comprehensive survey. Sensors, 22(24), 9637. https://doi.org/10.3390/s22249637

[3] Arulkumaran, K., Deisenroth, M. P., Brundage, M., & Bharath, A. A. (2017). Deep reinforcement learning: A brief survey. IEEE Signal Processing Magazine, 34(6), 26–38. https://doi.org/10.1109/MSP.2017.2743240

[4] Buczak, A. L., & Guven, E. (2016). A survey of data mining and machine learning methods for cyber security intrusion detection. IEEE Communications Surveys & Tutorials, 18(2), 1153–1176. https://doi.org/10.1109/COMST.2015.2494502

[5] Garcia-Teodoro, P., Diaz-Verdejo, J., Maciá-Fernández, G., & Vázquez, E. (2009). Anomalybased network intrusion detection: Techniques, systems and challenges. Computers & Security, 28(1–2), 18–28. https://doi.org/10.1016/j.cose.2008.08.003

[6] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Humanlevel control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236

[7] Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 2015 Military Communications and Information Systems Conference (MilCIS), 1–6. https://doi.org/10.1109/MilCIS.2015.7348942

[8] Nguyen, T. T., & Reddi, V. J. (2021). Deep reinforcement learning for cyber security. IEEE Transactions on Neural Networks and Learning Systems, 33(9), 4029–4048. https://doi.org/10.1109/TNNLS.2021.3121870

[9] Sarker, I. H., Kayes, A. S. M., Badsha, S., Alqahtani, H., Watters, P., & Ng, A. (2020). Cybersecurity data science: An overview from machine learning perspective. Journal of Big Data, 7(1), Article 41. https://doi.org/10.1186/s40537-020-00318-5

[10] Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018), 108–116. https://doi.org/10.5220/0006639801080116

[11] Shone, N., Ngoc, T. N., Phai, V. D., & Shi, Q. (2018). A deep learning approach to network intrusion detection. IEEE Transactions on Emerging Topics in Computational Intelligence, 2(1), 41–50. https://doi.org/10.1109/TETCI.2017.2772792

[12] Singh, S., & Chatterjee, M. (2017). Cloud security issues and challenges: A survey. Journal of Network and Computer Applications, 79, 88–115.

[13] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.

[14] Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., & Venkatraman, S. (2019). Deep learning approach for intelligent intrusion detection system. IEEE Access, 7, 41525– 41550. https://doi.org/10.1109/ACCESS.2019.2895334

[15] Xiao, Y., Xing, C., Zhang, T., & Zhao, Z. (2020). Deep reinforcement learning for softwaredefined networking: A survey. IEEE Communications Surveys & Tutorials, 22(4), 2521–2555. https://doi.org/10.1109/COMST.2020.3016235

[16] Mia, M. M., Roy, M. K., YASSAR, I. S., Mottalib, M. Y., Yezdani, S., Nijhum, A. M., ... & Uddin, M. K. (2025). Integrating Blockchain Security and Machine Learning for Fraud Detection in the US Banking System. Emerging Frontiers Library for The American Journal of Engineering and Technology, 7(11), 65-76.

[17] YASSAR, I. S. (2026). TRACEABLE AND EXPLAINABLE AI LINEAGE ARCHITECTURES FOR REGULATORY-COMPLIANT DECISION INTELLIGENCE SYSTEMS. Computers and Education Letters, 3(01), 1-18.

[18] Khatun, P., Umam, S., Razzak, R. B., Shamsuddin, I. B., & Salma, N. (2025). A study on the effectiveness of machine learning models for hepatitis prediction. Scientific Reports, 15(1), 30659.

[19] Umam, S., Razzak, R. B., Munni, M. Y., & Rahman, A. (2025). Exploring the non-linear association of daily cigarette consumption behavior and food security-An application of CMP GAM regression. PLoS One, 20(7), e0328109.

[20] Razzak, R. B., & Umam, S. (2025, November). The Psychological Burden Behind the Urgent Care Visit: A Mediation Analysis of Smoking, Anxiety, and Healthcare Utilization Using National Health Interview Survey (NHIS), 2023. In APHA 2025 Annual Meeting and Expo. APHA.

[21] Nguyen, A. T. P., Shak, M. S., & Al-Imran, M. (2024). Advancing early skin cancer detection: A comparative analysis of machine learning algorithms for melanoma diagnosis using

dermoscopic images. International Journal of Medical Science and Public Health Research, 5(12), 119-133.

[22] Mahmud, F., Das, A. C., Shak, M. S., Rahman, N., Eva, A. A., Mridha, M. F., & Hossen, M. J. (2025). HybridTabNet-QC: A transformer-based clinical feature fusion framework for heart disease risk prediction. IEEE Open Journal of the Computer Society, 7, 1-13.

[23] Das, A. C., Shak, M. S., Rahman, N., Mahmud, F., Shoaib, H. A., & Hossen, M. J. (2026). Cardiovascular disease prediction using variational recurrent autoencoders with uncertainty estimation. Scientific Reports.

[24] Das, A. C., Shak, M. S., Rahman, N., Mahmud, F., Shoaib, H. A., & Hossen, M. J. (2026). Cardiovascular disease prediction using variational recurrent autoencoders with uncertainty estimation. Scientific Reports.

[25] Das, A. C., Shak, M. S., Rahman, N., Mahmud, F., Eva, A. A., & Hasan, M. N. (2025, May). Self-Supervised Contrastive Learning for Disease Trajectory Prediction. In 2025 5th International Conference on Pervasive Computing and Social Networking (ICPCSN) (pp. 732- 738). IEEE.

[26] Mozumder, M. A. S., Mahmud, F., Shak, M. S., Sultana, N., Rodrigues, G. N., Al Rafi, M., ... & Bhuiyan, M. S. M. (2024). Optimizing customer segmentation in the banking sector: a comparative analysis of machine learning algorithms. Journal of Computer Science and Technology Studies, 6(4), 01-07.

[27] Rahman, M. H., Das, A. C., Shak, M. S., Uddin, M. K., Alam, M. I., Anjum, N., ... & Alam, M. (2024). Transforming customer retention in fintech industry through predictive analytics and machine learning. The American Journal of Engineering and Technology, 6(10), 150-163.

[28] Bhuiyan, R. J., Akter, S., Uddin, A., Shak, M. S., Islam, M. R., Rishad, S. S. I., ... & Hasan-Or-Rashid, M. (2024). Sentiment analysis of customer feedback in the banking sector: A comparative study of machine learning models. The American Journal of Engineering and Technology, 6(10), 54-66.

[29] Razzak, R. B., & Umam, S. (2025, November). Health Equity in Action: Utilizing PRECEDE-PROCEED Model to Address Gun Violence and associated PTSD in Shaw Community, Saint Louis, Missouri. In APHA 2025 Annual Meeting and Expo. APHA.

[30] Khatun, P., Umam, S., Razzak, R. B., Shamsuddin, I. B., & Salma, N. (2025). A study on the effectiveness of machine learning models for hepatitis prediction. Scientific reports, 15(1), 30659.

[31] Umam, S., Razzak, R. B., Munni, M. Y., & Rahman, A. (2025). Exploring the non-linear association of daily cigarette consumption behavior and food security-An application of CMP GAM regression. Plos one, 20(7), e0328109.

[32] Adams, R., Grellner, S., Umam, S., & Shacham, E. (2023, November). Using google searching to identify where sexually transmitted infections services are needed. In APHA 2023 Annual Meeting and Expo. APHA.

[33] Umam, S., Adams, R., Shacham, E., & Charles, D. L. (2024, October). Predictors of weapon carrying for high school students. In APHA 2024 Annual Meeting and Expo. APHA.

[34] YASSAR, I. S. (2023). SCALABLE SDN-BASED ARCHITECTURE FOR LARGE-SCALE ENTERPRISE NETWORK MANAGEMENT. Insights Sustainable Engineering Practices, 1(01), 115-130.

[35] Sayed, M. A., Badruddowza, M. S. U. S., Al Mamun, A., Nabi, N., Mahmud, F., Alam, M. K., ... & Choudhury, M. Z. M. E. (2024). Comparative analysis of machine learning algorithms

for predicting cybersecurity attack success: A performance evaluation. The American Journal of Engineering and Technology, 6(09), 81-91.

[36] MAMUN, A., Nath, A., Dey, S. K., Nath, P., RAHMAN, M., SHORNA, J., & Anjum, N. (2025). Real-time malware detection in cloud infrastructures using convolutional neural networks: a deep learning framework for enhanced cybersecurity. INTERNATIONAL JOURNAL OF COMPUTER SCIENCE, 10(03), 10-03.

[37] Cao, D. M., Sayed, M. A., Habib, S. A., Islam, M. T., Mia, M. T., Ayon, E. H., ... & Raihan, A. (2024). Advanced cybercrime detection: A comprehensive study on supervised and unsupervised machine learning approaches using real-world datasets. Journal of Computer Science and Technology Studies, 6(1), 40-48.

[38] Mia, M. M., Al Mamun, A., Ahmed, M. P., Tisha, S. A., Habib, S. A., & Nitu, F. N. (2025). Enhancing financial statement fraud detection through machine learning: A comparative study of classification models. Emerging Frontiers Library for The American Journal of Engineering and Technology, 7(09), 166-175.

[39] Bhuiyan, R. J., Akter, S., Uddin, A., Shak, M. S., Islam, M. R., Rishad, S. S. I., ... & Hasan-Or-Rashid, M. (2024). Sentiment analysis of customer feedback in the banking sector: A comparative study of machine learning models. The American Journal of Engineering and Technology, 6(10), 54-66.

[40] Siddique, M. T., Jamee, S. S., Sajal, A., Mou, S. N., Mahin, M. R. H., Obaid, M. O., ... & Hasan, M. (2025). Enhancing automated trading with sentiment analysis: Leveraging large language models for stock market predictions. The American Journal of Engineering and Technology, 7(03), 185-195.

[41] Sajal, A., Chy, M. S. K., Jamee, S. S., Uddin, M. N., Khan, M. S., & Gharami, A. K. & Ahmed, M.(2025). Forecasting Bank Profitability Using Deep Learning and Macroeconomic Indicators: A Comparative Model Study. International Interdisciplinary Business Economics Advancement Journal, 6(06), 08-20.

[42] Akhi, S. S., Shakil, F., Dey, S. K., Tusher, M. I., Kamruzzaman, F., Jamee, S. S., ... & Rahman, N. (2025). Enhancing banking cybersecurity: An ensemble-based predictive machine learning approach. Am. J. Eng. Technol., 7(03), 88-97.

[43] Hossain, S., Sajal, A., Jamee, S. S., Tisha, S. A., Siddique, M. T., Obaid, M. O., ... & Haque, M. S. U. (2025). Comparative analysis of machine learning models for credit risk prediction in banking systems. The American Journal of Engineering and Technology, 7(04), 22-33.

[44] Hossen, M. A., BHATTACHARJEE, B., DEY, S., JAMEE, S., OBAID, M., MIA, M., ... & SHARIF, M. (2025). Business analytics for customer segmentation: A comparative study of machine learning algorithms in personalized banking services. International Journal of Economics Finance & Management Science, 10(03), 1-13.

[45] Nath, P. C., Chy, M. S. K., Hossain, M. R., Miah, M. R., Jamee, S. S., Sharif, M. K., ... & Ahmed, M. (2025). Comparative Performance of Large Language Models for Sentiment Analysis of Consumer Feedback in the Banking Sector: Accuracy, Efficiency, and Practical Deployment. Frontline Marketing, Management and Economics Journal, 5(06), 07-19.

[46] S. M. Rezvi, F. Mujtahid, K. R. Ahmed, A. Madhiya, V. SOLANKI and S. S. Jamee, "An Intelligent Multi-Source Data Fusion System for Sales Demand Forecasting in Retail and E-Commerce," 2026 Third International Conference on Innovations in Cybersecurity and Data Science (ICICDS), Pathum Thani, Thailand, 2026, pp. 1112-1117, doi: 10.1109/ICICDS70526.2026.11604912.

[47] Ahmmed, M. J. Pritom Das, Tamanna Pervin, Sadia Afrin, Sanjida Akter Tisha, Md Mehedi Hassan, & Nabila Rahman.(2024). COMPARATIVE ANALYSIS OF MACHINE LEARNING ALGORITHMS FOR BANKING FRAUD DETECTION: A STUDY ON PERFORMANCE, PRECISION, AND REAL-TIME APPLICATION. International Journal of Computer Science & Information System, 9(11), 31-44.

[48] Ahmed, M. P., Arif, M., Chowdhury, M. S., Bhuiyan, R. J., Rahman, T., Ahmmed, M. J., ... & MAMUN, M. (2024). Comparative analysis of machine learning techniques for accurate lung cancer prediction. Am. J. Eng. Technol, 6, 92-103.

[49] Akter, S., Mahmud, F., Rahman, T., Ahmmed, M. J., Uddin, M. K., Alam, M. I., & Jui, A. H. (2024). A comprehensive study of machine learning approaches for customer sentiment analysis in banking sector. The American Journal of Engineering and Technology, 6(10), 100- 111.

[50] Sweet, M. M. R., Arif, M., Uddin, A., Sharif, K. S., Tusher, M. I., Devi, S., ... & Sarkar, M. A. I. (2024). Credit risk assessment using statistical and machine learning: Basic methodology and risk modeling applications. International Journal on Computational Engineering, 1(3), 62-67.

[51] Arif, M., Ahmed, P., Mamun, A. A., Uddin, K., Mahmud, F., Rahman, T., ... & Hossain, S. (2024). Dynamic pricing in financial technology: evaluating machine learning solutions for market adaptability. International Interdisciplinary Business Economics Advancement Journal, 5(10), 13-27.

[52] Ahmed, M. P., Arif, M., Al Mamun, A., Mahmud, F., Rahman, T., Ahmmed, M. J., ... & Uddin, M. K. (2024). A comparative study of machine learning models for predicting customer churn in retail banking: insights from logistic regression, random forest, GBM, and SVM. Journal of Computer Science and Technology Studies, 6(4), 92-101.

[53] Ahmmed, M. J. (2025). Systematic review and quantitative evaluation of advanced machine learning frameworks for credit risk assessment, fraud detection, and dynamic pricing in US financial systems. International Journal of Business and Economics Insights, 5(3), 1329-1369.

[54] Rahman, M. M., & Ahmmed, M. J. (2026). AI-Driven Risk Analytics Models for Early Detection of Financial Noncompliance in Multi-Branch Banking Systems. American Journal of Data Science and Analytics, 7(04), 81-123.

[55] Hossen, M. E. ., Akhter, A. ., Ghosh, S. ., Khandaker, M. ., Azam, M. N. ., Malek, H. A. ., Naher, K. ., & Bhuiyan, M. M. R. . (2026). Predicting Infectious Disease Outbreaks Using Machine Learning and Real-Time Epidemiological Data: Leverage Social Media, Environmental, And Public Health Data to Forecast Outbreaks Like Influenza, COVID-19, Or RSV. International Journal of Medical Science and Public Health Research, 7(02), 7–17. https://doi.org/10.37547/ijmsphr/Volume07Issue02-02

[56] Umam, S., & Razzak, R. B. (2025, November). A 20-Year Overview of Trends in Secondhand Smoke Exposure Among Cardiovascular Disease Patients in the US: 1999–2020. In APHA 2025 Annual Meeting and Expo. APHA.

[57] Razzak, R. B., & Umam, S. (2025, November). Health Equity in Action: Utilizing PRECEDE-PROCEED Model to Address Gun Violence and associated PTSD in Shaw Community, Saint Louis, Missouri. In APHA 2025 Annual Meeting and Expo. APHA.

[58] Razzak, R. B., & Umam, S. (2025, November). A Place-Based Spatial Analysis of Social Determinants and Opioid Overdose Disparities on Health Outcomes in Illinois, United States. In APHA 2025 Annual Meeting and Expo. APHA.

[59] Khan, M. S., Gharami, A. K., Nitu, F. N., Uddin, M. N., Ahmed, M., Roy, M. K., & Yezdani, S. (2025). Deep Learning-Driven Customer Segmentation in Banking: A Comparative Analysis for Real-Time Decision Support. International Interdisciplinary Business Economics Advancement Journal, 6(08), 9-22.

[60] Sajal, A., Chy, M. S. K., Jamee, S. S., Uddin, M. N., Khan, M. S., & Gharami, A. K. & Ahmed, M.(2025). Forecasting Bank Profitability Using Deep Learning and Macroeconomic Indicators: A Comparative Model Study. International Interdisciplinary Business Economics Advancement Journal, 6(06), 08-20.

[61] Siddique, M. T., Uddin, M. J., Chambugong, L., Nijhum, A. M., Uddin, M. N., & Shahid, R. & Ahmed, M.(2025). AI-Powered Sentiment Analytics in Banking: A BERT and LSTM Perspective. International Interdisciplinary Business Economics Advancement Journal, 6(05), 135-147.

[62] Ayub, M. I., Gharami, A. K., Nitu, F. N., Uddin, M. N., Islam, M. I., Nijhum, A. M., ... & Yezdani, S. (2025). AI-driven demand forecasting for multi-echelon supply chains: Enhancing forecasting accuracy and operational efficiency through machine learning and deep learning techniques. Emerging Frontiers Library for The American Journal of Management and Economics Innovations, 7(07), 74-85.

[63] Islam, M. M., Sarkar, M. A. R., Ahmed, M., Moniruzzaman, S. M., & Uddin, M. N. (2009). Germination, vigour and emergence indicators of Corchorus olitorius L, seed and their relationship as influenced by seed sources. Bangladesh J. Jute and Fib. Res, 29(1&2), 1-8.

[64] Uddin, M. N., & Aziz, M. M. (2026). Shapley Value-Guided Adaptive Ensemble Learning for Explainable Financial Fraud Detection with US Regulatory Compliance Validation. arXiv preprint arXiv:2604.14231.

[65] Uddin, M. N. (2026). Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the US Banking Sector. arXiv preprint arXiv:2604.14232.

[66] Uddin, M. N. (2026). Does Founding Team Human Capital Heterogeneity Predict Venture Survival? Evidence from the Kauffman Firm Survey. Evidence from the Kauffman Firm Survey (April 04, 2026).

[67] Uddin, M. N. (2026). Cost Efficiency Dynamics in Indian Public Sector Banks: A DEA-Based Analysis with Second-Stage Tobit Estimation and Post-Reform Comparative Evidence, 2009- 2023. Available at SSRN 6680922.

[68]