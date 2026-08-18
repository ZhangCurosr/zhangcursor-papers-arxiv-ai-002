# Software Engineering for AI-driven Building Operation

<sub>Philipp</sub> <sub>Zech</sub>1[0000-0002-4952-4337]<sub>,</sub> <sub>Sascha</sub> <sub>Hammes</sub>1[0000-0001-5821-5053]<sub>,</sub>

Johannes Weninger<sup>2[0000-0002-8307-9652]</sup>, Jürgen

<sub>Pannosch</sub>3[0009-0002-3946-9239]<sub>,</sub> <sub>and</sub> <sub>Gernot</sub> <sub>Steindl</sub>3[0000-0002-9035-9206]

<sup>1</sup> University of Innsbruck, Tyrol, Austria {philipp.zech,sascha.hammes}@uibk.ac.at 2 Bartenbach - The Lighting Innovators, Wattens, Austria Johannes.Weninger@bartenbach.com 3 University of Applied Sciences Burgenland, Austria {jürgen.pannosch,gernot.steindl}@hochschule-burgenland.at

Abstract. Building operations are energy-ineficient. Artificial Intelligence (AI)-driven control systems promise benefits through optimization and predictive control, but deploying them in real buildings reveals a significant software engineering (SE) challenge. SE for AI practices assume digital environments where failures mean poor user experience. Buildings are diferent. A bad control decision wastes energy irreversibly, violates occupant comfort, or accelerates equipment wear. Although actual safety-critical failures are rare, as real building automation systems are inherently fault-tolerant, the physical and lasting nature of even minor failures fundamentally changes SE4AI requirements. Rooted in two interdisciplinary research projects in civil engineering and computer science that target the AI-driven optimization of building operations, we identify the missing perspectives in SE4AI that currently stymie the successful deployment of AI-based systems for building operations. We further share lessons learned and best practices, and discuss broader implications for engineering AI-driven building operations and cyber-physical systems more generally. Our work proposes a foundation for SE4AI in systems where failure has physical consequences — one the research agenda below will need to validate.

Keywords: Software Engineering for AI · MLOps · Smart Buildings · Cyber-Physical Systems · Building Operation · AI-driven Engineering.

## 1 Introduction

Buildings account for a substantial share of global energy consumption and greenhouse gas emissions [33]. Traditional Building Automation Systems (BAS) rely on handcrafted, rule-based control strategies that are commissioned once and rarely updated [80]. These static approaches struggle to adapt to changing weather, shifting occupancy patterns, and equipment degradation, leading to suboptimal performance and energy waste [66]. Artificial Intelligence (AI), especially Machine Learning (ML), ofers a way forward. Techniques such as Reinforcement Learning (RL) and Model Predictive Control (MPC) can learn and deploy control strategies that adapt to the complex and stochastic nature of building environments [66, 73]. Using historical and live data, these systems aim to improve energy eficiency while maintaining occupant comfort [20].

However, deploying AI in buildings is not an AI problem per se. Rather, it is a Software Engineering (SE) problem. To this end, Software Engineering for AI (SE4AI) has established principles and practices for building, deploying, and maintaining AI systems [64]. These practices however were shaped by purely digital environments. We argue that applying them directly to building operations is insuficient, as buildings eventually are Cyber-physical Systems (CPS) at the intersection of digital and physical worlds [53]. Wrong decisions have irreversible physical consequences, e.g., wasted energy that cannot be recovered, comfort violations that erode occupant trust, and equipment wear that shortens system lifespan. True safety-critical failures are rare as BAS are inherently fault-tolerant through thermal inertia, safety interlocks, and regulatory safeguards, and the EU AI Act classifies building AI as low-risk [22]. But the physical nature of even minor failures fundamentally changes what SE must provide [32].

In digital systems, failures are recoverable using fast rollbacks to contain consequences. In buildings, a faulty control decision can shorten the lifespan of Heating Ventilation and Air Conditioning (HVAC) equipment, cause lasting discomfort, or waste energy that is simply gone [47]. Digital systems interact through well-defined APIs. Building AI must contend with thermodynamics, hardware limits, and slow feedback loops governed by thermal mass. This gap is further compounded by noisy data from legacy hardware [29], siloed workflows [28], and the need to earn the trust of operators who require explanations and safe manual overrides [19,32]. Commensurate with this, our paper addresses the following central question:

How can we complement SE4AI principles and practices to efectively support the development, deployment, and operation of AI-driven systems in the cyber-physical context of building operations?

To answer this question, we draw from two interdisciplinary research projects on building operations (cf. Section 2) and contribute the following:

– Identification and categorization of the key challenges that arise when engineering AI for building operations, and distinguish them from those in purely digital applications.

– Lessons learned and best practices derived from trial and error and practical experience.

Discussion of the broader implications for SE4AI in other cyber-physical domains.

We position this as a preliminary experience paper grounded in two ongoing projects. It is aimed primarily at researchers working on AI deployment in CPS settings, and at practitioners facing the same structural problems in buildings or adjacent physical domains. Empirical validation of the proposed practices is an explicit part of the research agenda we outline in Section 6. Instead, our work provides a structured perspective on the SE4AI requirements of AI-driven building operations as a foundation for researchers and practitioners to build more robust and efective BAS. In our work, we focus primarily on HVAC and lighting control, which account for the largest share of building energy consumption. Though building operation covers more ground than this (e.g., access control, elevator management, air quality, fluid distribution) and many of the challenges we identify apply there too, our empirical basis is HVAC and lighting, and we scope our claims accordingly.

## 2 Background and Related Work

Our work grows out of practical experience in two ongoing interdisciplinary research projects, both targeting the optimization of building operations through AI-driven approaches.

BOREALIS<sup>4</sup> develops adaptive BAS that learn from operational data to reduce energy consumption and improve occupant comfort. It uses RL to continuously optimize HVAC and lighting systems based on historic and real-time sensor data [20]. The hard part is not the learning but instead deploying self-learning systems in real buildings where wrong control decisions cause equipment wear, eficiency losses, or occupant discomfort. BOREALIS requires safe exploration, robust monitoring, and decision-making that facility managers can actually understand and override. In practice, this means working with a real university building where occupancy is irregular, sensor coverage is partial, and early RL experiments that looked promising in simulation revealed a larger sim-to-real gap than expected once deployed, a direct motivation for the practices in Section 4.

ELEVATE<sup>5</sup> uses a Digital Twin (DT) and AI-driven optimization for energyeficient building operation, with a particular focus on indoor environmental quality. It models the complex relationships between environmental parameters, such as lighting, air quality, thermal comfort, and occupant feedback, to build predictive models that autonomously adjust BAS control parameters [28,29]. The challenge is less about the models, but more about the "messiness" underneath that comprises fragmented data sources, legacy hardware, and the gap between what works in simulation and what works in a real building. Crucially, the models are not the key challenge. Instead, getting the data to a point where training is possible is the key challenge.

Both projects epitomize the same set of problems: integrating with fragmented legacy BAS, working with sparse and noisy sensor data, bridging digital planning and physical deployment, and earning the trust of the operators who have to run these systems on a daily basis.

## 2.1 Related Work

Our work sits at the intersection of SE for Building Automation (SE4BAS), SE4AI, and AI for Building Operations (AI4BOS) . In the following we review key literature in each area.

SE for Building Automation (SE4BAS) Most recent work positions BAS as a distributed, service-oriented (SOA) software ecosystem. Studies review SOA for energy-eficient buildings and survey data-driven BAS architectures and identify recurring gaps in scalability, interoperability, and the integration of analytics with operational control [44, 67]. End-to-end MACH (microservices, API-first, cloud-based, headless)-aligned architectures that support DT visualization and sensor data integration have been demonstrated [15], alongside event-driven BIM (Building Information Modeling)–BAS–IoT platforms [13], semantic middleware [59], and software-defined building infrastructures [69]. Reviews on BIM and DTs, however, point to a persistent gap, viz. DT models are rarely connected directly to automation functions and dynamic operation [71].

At the modeling and process level, most work focuses on semantic representations and continuous engineering. Semantic interoperability frameworks for BAS [74], assisted requirements engineering [41], and knowledge-based services for automation component modeling [75] all aim to reduce manual efort and improve reuse across projects. Mizutani and Mayer bring DevOps thinking into BAS engineering by using semantic reasoners and Web of Things descriptions to enable continuous verification across planning and commissioning [45]. Ramanathan and Mayer extend this further with ontologies that automate control logic selection [55] and a physics-infused DT framework that matches DT descriptions to control-program libraries [54].

On the commissioning and testing side, Zech et al. propose tool support for model-based BAS and DT commissioning [77, 78, 80, 82]. Santos et al. present an automated step-response testing tool for initial and retro-commissioning [58]. Fierro et al. describe a simulated DT environment that lets developers prototype and test building applications before touching a real system [23]. Further work explores intelligent API frameworks for occupancy-based HVAC integration [1] and vendor-agnostic BAS implementations in specific facilities [7, 30].

SE for AI Systems (SE4AI) SE4AI has matured into a recognized subdiscipline with a substantial body of surveys, frameworks, and empirical studies. Systematic reviews show that most work has focused on dependability, safety, and testing, while maintenance and evolution remain underexplored, and data problems dominate in practice [5, 26, 37, 42, 61]. A common theme is that data, models, and code must be treated as co-equal artifacts within iterative, experiment-driven life cycles. In this line, several life cycle frameworks have been proposed. These cover AI engineering organized around architecture, development, and process [11] along reliability- and debt-oriented ML life cycles that span data engineering, training, testing, and continuous evaluation [24, 57]. In addition, various MLOps pipelines that integrate data handling, model learning, and operations with explicit triggers and artifact management have been proposed [40, 43, 65].

More targeted work covers requirements engineering for AI systems [2, 6], ML integration patterns and architectural tactics [31, 60, 62, 72], and quality assessment frameworks [25, 34]. A recurring finding from empirical studies is that many SE4AI problems are not really technical, but instead socio-technical, and deeply rooted in role boundaries, communication gaps, and missing documentation [3, 40, 48, 49].

AI for Building Operations (AI4BOS) Early AI building control worked by embedding data-driven surrogates into optimization loops. Neural network models combined with genetic algorithms achieved solid supervisory energy savings in real HVAC systems [68], and symbolic regression has been used to learn interpretable thermal models for MPC deployment in occupied workspaces [50]. A large body of work since then applies RL for supervisory control. Modelbased RL frameworks that learn neural surrogates and distill MPC solutions into fast policy networks show strong sample eficiency [17, 83]. Model-free deep RL controllers map measured conditions directly to setpoints [84]. Safe RL with control-barrier functions tackles comfort constraint satisfaction [20]. Comparative work suggests model-based optimization can outperform RL when accurate physical models are available, though RL wins on online speed and does not depend on accurate physical models [36].

At larger scales, hierarchical multi-agent RL has been used to coordinate HVAC and distributed energy resources across hundreds of buildings [51]. Industrial deployments report real energy and carbon reductions across building portfolios [16, 76]. More recently, physics-informed diferentiable controllers that learn coupled airflow and thermal dynamics ofer a compelling alternative to black-box RL [9].

## 2.2 The Missing Perspective

Research in all three areas is moving fast, but the areas rarely talk to each other. SE4BAS has built solid architectural patterns, semantic models, and DevOps workflows for modern BAS engineering (cf. Section 2.1). AI4BOS has shown that RL, MPC, and physics-informed methods can deliver real energy savings, though actual field deployments remain rare (cf. Section 2.1). SE4AI has established mature practices for ML pipelines, data versioning, and continuous deployment (cf. Section 2.1). But each field solves its own piece of the problem and largely ignores the others. The accrued gap however matters. Deploying AI in buildings is not just about better algorithms or cleaner pipelines. It means operating a CPS [53] where failures have physical consequences, testing requires virtual replicas, and the humans in the loop are legally responsible for the outcomes. SE4AI practices were designed for purely digital environments. They do not account for thermodynamics, sensor drift, or delayed feedback loops. Table 1 makes this gap concrete.

Table 1. The missing perspectives in SE4AI for building operations. The contrasts highlight diferences in emphasis, not categorical absences. Domains like autonomous driving also engage with physical models; buildings are structurally distinct due to their combination of slow feedback loops, legacy infrastructure, and continuously occupied environments.
<table><tr><td></td><td>General SE4AI Perspective</td><td>The Missing Perspective</td></tr><tr><td></td><td>P1 The cost of failure is digital</td><td>The cost of failure is physical</td></tr><tr><td></td><td>P2 The world is data</td><td>Physical dynamics dominate over data patterns</td></tr><tr><td></td><td>P3 Testing is data-driven</td><td>Testing is simulation-driven and risky</td></tr><tr><td></td><td>P4 Humans are end-users</td><td>Humans are in-the-loop stakeholders</td></tr><tr><td></td><td>P5 Data is abundant but often unlabeled</td><td>Data is physical, sparse, and noisy</td></tr><tr><td></td><td>P6 Time scales are fast</td><td>Time scales are slow and delayed</td></tr></table>

Both BOREALIS and ELEVATE run straight into this disconnect. BORE-ALIS needs safe RL exploration in buildings where wrong actions have real physical consequences, yet SE4AI ofers no guidance on sandboxing those consequences or rebuilding operator trust after a bad decision. ELEVATE has to stitch together fragmented BIM, BAS, and sensor data streams to train models, but existing MLOps pipelines assume clean APIs and version-controlled datasets instead of legacy systems with intermittent connectivity. Neither project can fall back on A/B testing. Both need virtual testbeds as a core infrastructure, yet SE4AI treats simulation as optional. The digital-first mindset simply breaks when physical consequences enter the loop.

The challenges and practices in this paper grew out of practical experience in BOREALIS and ELEVATE. They reflect problems that recurred across both projects and diferent buildings, and that were discussed with project partners including facility managers, lighting engineers, and control system integrators. We do not claim a formal empirical method. The contribution is a structured reflection on ongoing project experience, and we treat it as such.

## 3 Key Challenges

In Section 2, we identified six missing perspectives (P1 - P6) in SE4AI for building operations (cf. Table 1). These perspectives manifest as five concrete challenges that arise not from acute safety risks, but from the physical, irreversible, and long-horizon nature of building operations. These challenges are not exclusive to buildings.

## 3.1 Physical Consequences and Irreversibility: P1

In digital systems, bad predictions also have real costs, e.g., missed deadlines, lost sales, degraded user trust. In buildings, failures are physically felt and structurally harder to reverse. The diference is not categorical but one of degree as consequences are more tangible, more persistent, and harder to undo. We distinguish three categories of failure: (i) eficiency losses, where a control decision that simultaneously heats and cools wastes energy that cannot be recovered; (ii) comfort violations, where flickering lighting or temperature swings cause occupant complaints, and even after correction the damage to trust persists which often leads operators to disable the system; and (iii) equipment wear, where aggressive actuator cycling and rapid setpoint changes cause mechanical and thermal stress that shortens equipment lifespan and raises maintenance costs.

Actual critical failures like frozen pipes or equipment fires are rare. Real BAS are inherently fault-tolerant, where thermal mass provides bufering, safety interlocks prevent dangerous states, and building codes mandate protective measures. The EU AI Act classifies building AI as low-risk for this reason [22]. But the physical nature of even minor failures changes SE4AI requirements. Standard SE4AI assumes failures are recoverable through rollback or retraining. In buildings, wasted energy stays wasted, comfort violations stay felt, and equipment wear stays accumulated. Consequently, every control action must be validated against physical constraints and safety bounds before deployment, not because catastrophic failure is likely, but because routine consequences are irreversible and operator trust is hard to rebuild once lost.

## 3.2 Laws of Physics Over APIs: P2 & P6

An AI model sends a temperature setpoint, but the actual outcome depends on thermal mass, air flow, solar radiation, humidity, equipment capacity, hydraulic pressure, and electrical loads, none of which the model controls directly. Feedback is delayed (thermal response takes time to settle), noisy (sensors drift), and nonlinear (heat transfer does not follow simple statistical patterns). The environment is not a data source. It is a system of coupled diferential equations that cannot be fully observed or controlled. Consequently, model training must account for physical dynamics and temporal delays, as black-box pattern matching fails when the underlying patterns are governed by real, slow, and noisy physical processes.

## 3.3 No Unified Development, Testing, and Deployment Infrastructure: P3

Each building is a prototype. Unlike digital systems, A/B testing is impossible as deploying two conflicting control strategies in adjacent occupied zones risks real discomfort and energy waste. Shadow deployment does not work for HVAC. Simulation exists but sufers from model mismatch, since real buildings never behave exactly like their models [79]. Worse, the tools for simulation (e.g., EnergyPlus), control (e.g., BACnet stacks), and ML training (e.g., PyTorch) exist in separate ecosystems with no unified workflow from BIM to AI to deployment. Consequently, development happens in simulation, but deployment regularly fails to account for the digital-to-physical gap.

## 3.4 Trust, Transparency, and Human Operators in the Loop: P4

Facility managers are not end users. They are legally responsible for safety and comfort. They must understand why the AI made a decision, override it when needed, and explain failures to tenants or regulators. An operator who does not trust a system will disable it. And a disabled AI delivers zero value regardless of how well it performs in evaluation. In buildings, explainability is not a nice-to-have. It is a prerequisite for adoption. Consequently, models must be interpretable or provide meaningful post-hoc explanations, and control interfaces must include manual overrides and well-communicated safety bounds. This extends beyond facility managers to building occupants, who are rarely AI experts. Explainability for occupants requires diferent techniques, e.g., simple dashboards, plain-language summaries, and visual feedback, rather than the technical post-hoc explanations suited to professionals. Designing for this audience from the start is part of making a system actually adoptable.

## 3.5 Scattered and Siloed Infrastructure: P5

A single building might have HVAC from the 1990s running BACnet, lighting from 2010 using DALI, newer sensors on Modbus, and occupancy data in a separate facility management system where none of it is designed to talk to the others. Data is sparse (many sensors fail silently), noisy (calibration drifts over time), and inconsistent (timestamps may not align). There is no common repository, no clean API, and no version-controlled dataset waiting to be consumed by a training pipeline. Consequently, data engineering dominates development time, and data integration becomes the real bottleneck, not the model architecture.

## 4 Lessons Learned and Best Practices from the Field

Drawing from BOREALIS and ELEVATE, we present our best practices that directly address the challenges from Section 3. These are not theoretical recommendations, but reflect what we learned from engineering and deploying AIdriven control in real buildings.

## 4.1 Simulation-First Testing

Testing control policies directly in live buildings is risky. A policy that behaves unexpectedly can compromise occupant comfort, waste energy, or damage equipment. A simulation-first approach reduces this risk by evaluating policies in virtual environments before deployment [39, 63]. Tools like EnergyPlus and Modelica-based environments allow for testing policies across a wide range of operating conditions, including seasonal variations, occupancy patterns, and fault scenarios, repeatedly and without consequence. This makes it practical to explore behavior that would take months to observe in a real building. Simulation also supports formal analysis, as a virtual testbed can be instrumented to check safety and performance properties systematically.

But the sim-to-real gap is real [79]. Building models are calibrated on historical data and always carry uncertainty. Thermal dynamics, occupant behavior, and equipment degradation are hard to model precisely. A policy that looks good in simulation may perform diferently once deployed, especially under conditions not well represented in the model. Simulation is the right place to rule out unsafe or poor-performing policies early. But it cannot replace real-system testing, and results based solely on simulation should be treated with caution [79].

## 4.2 Physical AI

Building control policies act on the physical world. Sensors measure temperature, CO<sub>2</sub>, occupancy, and energy consumption. Actuators open valves, adjust setpoints, and switch equipment on and of. The building responds, and the policy should adapt accordingly. This feedback loop is what makes building control a form of physical AI [14,52]. This matters for testing and validation. A policy’s behavior cannot be understood by analyzing its code alone as it emerges from the interaction with the building, which is complex, noisy, and time-varying. Occupant behavior is unpredictable, equipment degrades, and weather changes. A policy that is safe and eficient under one set of conditions may not be under another [20, 66].

This is why simulation-first testing (cf. Section 4.1) is necessary but not sufficient. Virtual testbeds like BOPTEST [10] provide standardized environments for pre-deployment evaluation, but real-world validation cannot be skipped. The building itself is part of the system. Successful deployments, such as data center cooling, addressed this by combining simulation-based pre-screening with staged deployment and continuous monitoring [38]. Simulate, deploy cautiously, monitor closely, and redeploy [77, 78, 80]: this pattern is emerging as a practical standard for physical AI in buildings [23].

## 4.3 Data Minimalism

More data is not always better. In practice, data collection in buildings is expensive, unreliable, and raises privacy concerns [21]. The case for data minimalism starts with robustness. Sensors fail, calibration drifts, and occupancy data is patchy. A policy that requires dense, high-quality data will be brittle in the field. Designing for minimal data from the start forces robustness [66]. However, there is also a regulatory dimension. Buildings collect data about occupants, e.g., when they arrive, where they sit, or how they move. Under GDPR, this requires justification, consent, and protection [21]. Thus, data minimalism is not just good engineering, it is a legal requirement in many jurisdictions.

Sample eficiency matters too. Collecting large amounts of real building data is slow, costly, and can only happen during operation, when poor control already has real consequences. A policy that learns from limited observations through simulation pre-training, transfer learning, or Bayesian methods is far more deployable than one that needs months of data before it can function [10, 20, 38]. The principle is simple: collect only what is needed, make the most of what is available, and design for the data that is realistically there. Yet, we acknowledge the counterargument: sensor costs are falling, and denser instrumentation enables fault detection, drift correction, and more robust models. Data minimalism is not a claim against instrumentation. It is a design posture for the realistic case. When dense, reliable data is available, building on it is fine; the principle then shifts to designing safe fallback behavior for when that data disappears.

In practice, operationalizing this means starting from the control loop and working backwards: identify the minimum sensor set needed to close it, then ask what the system should do when any of those sensors is missing or stale. For HVAC, temperature and energy consumption often cover most of what is needed. Transfer learning and simulation pre-training can reduce dependence on real building data during early phases. Safe fallback modes that default to a conservative rule-based policy when data quality drops below a threshold can make the diference between a system that degrades gracefully and one that fails silently.

## 4.4 Trust and Transparency

A system that works but is not understood will not be used. Facility managers must justify decisions to building owners, and occupants must accept that an automated system is making choices about their environment. Trust requires transparency, and transparency means being able to understand why the system made a particular decision [4]. An RL agent that maps sensor readings to control actions may perform well in testing but ofer no insight into its reasoning. When it fails—and it will—there is no way to diagnose why [18]. A facility manager who cannot understand a control decision cannot override it confidently, explain it to a building owner, or trust the system enough to leave it running unsupervised.

Explainability methods such as feature importance, saliency maps, and surrogate models can provide post-hoc explanations [46]. But these are approximations. A better approach is to design for interpretability from the start by using simpler policy architectures where possible [20], or building explanation modules directly into the control pipeline [70]. However, there is also the regulatory dimension (cf. Section 4.3). GDPR gives individuals the right not to be subject to purely automated decisions without explanation [21]. The EU AI Act triggers requirements for transparency, logging, and human oversight for certain BAS [22]. These are legal obligations, not just options. A technically superior policy that cannot explain itself will thus lose to a simpler one that can be explained.

## 4.5 DTs as Central Catalysts

Many of the challenges above can be addressed together through one architectural choice: a DT of the building [8, 56, 71, 81, 82]. A DT is more than a simulation. It is a synchronized replica of the real system and continuously updated with live sensor data [27] to reflect the actual state of the building at any point in time. And because it is a model, it can be used for things that cannot safely be done to the real building.

An RL agent needs to explore to learn, but exploration in a real building risks uncomfortable or unsafe states. In a DT, the agent explores freely with no real consequences [18]. Beyond exploration, the DT enables what-if simulation grounded in the actual current state of the building, which makes results more reliable than generic ofline simulation. Before any policy reaches a real building, it can be validated in the DT first, reducing deployment risk and shortening commissioning cycles [35,77]. In addition, the DT can also generate training data for rare scenarios, which directly addresses data scarcity (cf. Section 4.3). And because it exposes the full internal state of the system, it supports explainability (cf. Section 4.4) as a decision can be replayed and inspected to make it easier to understand and justify to facility managers and building owners. The DT is not just one tool among many. It provides the environment in which safe, data-eficient, and explainable building control AI becomes practical.

## 5 Discussion and Implications

Our practices are not isolated techniques. Together, they form a coherent response to a structural problem: SE4AI was built for the digital world, and buildings are not digital. This has implications beyond buildings. The challenges we identified (cf. Section 3) appear in any domain where AI must act on a physical system, e.g., manufacturing lines, water treatment plants, trafic control, or energy grids. In all of these, the same mismatch exists. SE4AI assumes fast feedback, clean data, and recoverable failures. Physical systems ofer none of these. What we describe for buildings is a template for a broader discipline, viz. SE4AI for CPS.

The DT sits at the center of this template. Without a synchronized replica of the physical system, simulation-first testing is just ofline experimentation. With a DT, testing is grounded in the actual state of the real system. Without a DT, explainability is a property of the model alone. With a DT, decisions can be replayed, inspected, and justified against real system behavior. The DT does not solve the sim-to-real gap, but it narrows it by keeping the virtual and physical systems continuously synchronized [12].

Data minimalism deserves recognition as a first-class engineering principle. In digital systems, data abundance is assumed and the challenge is managing it. In physical systems, data is expensive, slow to collect, privacy-sensitive, and often unreliable. Designing for minimal data from the start produces systems that are more robust, more deployable, and easier to maintain. This shift from data maximalism to data minimalism is one of the clearest diferences between SE4AI for digital and physical environments.

Trust works diferently in physical systems too. In web applications, a user who distrusts a recommendation simply ignores it. In a building, a facility manager who distrusts the AI disables it. The consequence is not a missed click but a system that is physically present and doing nothing useful. Earning and maintaining operator trust is a first-class engineering concern, not a UX afterthought. This shifts explainability from a regulatory checkbox to a core architectural requirement.

Finally, the regulatory landscape is catching up. GDPR already constrains how occupant data can be collected and used [21]. The EU AI Act introduces transparency and oversight obligations that will afect how building AI systems are designed, documented, and deployed [22]. SE4AI practices for physical systems must thus account for these obligations from the start, and not retrofit to them later.

Several threats limit the generalizability of our findings. Both projects are situated in an Austrian academic context with specific regulatory, climatic, and infrastructure conditions. The practices were derived from the authors’ own work, which introduces confirmation bias. The generalization to other CPS domains is argued by analogy rather than demonstrated empirically. We name these threats not to undermine the contribution, but because doing so is consistent with framing this as the start of a research agenda rather than a finished result.

## 6 Conclusion and Future Work

SE4AI is a maturing discipline, but it was built for a purely digital world. Buildings however are physical. Deploying AI in a building means acting on a system governed by thermodynamics, not statistics. Failures are irreversible, data is sparse, operators are legally responsible, and feedback is slow. We identified this gap, named the missing perspectives, and proposed five practices that together form a foundation for SE4AI in CPS. However, this is not the end of the problem. It is the start of a research agenda:

Validate the practices in the field BOREALIS and ELEVATE provide the testbeds. Each practice needs to be deployed in real buildings and measured against concrete outcomes like operator trust, data eficiency, and time to deployment.

Close the sim-to-real gap Building DTs always carries uncertainty. Better calibration methods are needed, along with evidence that tells practitioners when simulation-based testing is suficient and when it is not [79].

Build the tooling The workflow from BIM to DT to AI to deployment does not yet exist as an integrated pipeline. Progress starts with the interfaces between simulation environments, control systems, and ML toolchains.

Extend to other physical domains The structural mismatch between SE4AI assumptions and physical reality is not unique to buildings. Testing whether our practices transfer to manufacturing, energy grids, and infrastructure is thus a natural next step.

The path from algorithm to deployed, trusted, and efective building AI is a SE problem. It eventually should be treated as one.

Acknowledgments. The research leading to these results has received funding from the Austrian Funding Agency (FFG) under grant agreements no. 5121377, BOREALIS, and no. 5138997, ELEVATE.

## References

1. Adepoju, S., David, S.: An Intelligent API Framework for Real-time Occupancy-Based HVAC Integration in Smart Building Management Systems. Journal of Knowledge Learning and Science Technology ISSN: 2959-6386 (online) 4(1), 61–70 (2025)

2. Ahmad, K., Abdelrazek, M., Arora, C., Bano, M., Grundy, J.: Requirements Engineering for Artificial Intelligence Systems: A Systematic Mapping Study. Information and Software Technology 158, 107176 (2023)

3. Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N., Nushi, B., Zimmermann, T.: Software Engineering for Machine Learning: A Case Study. In: 2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP). pp. 291–300. IEEE (2019)

4. Arrieta, A.B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., et al.: Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Information fusion 58, 82–115 (2020)

5. Assres, G., Bhandari, G., Shalaginov, A., Gronli, T.M., Ghinea, G.: State-Of-The-Art and Challenges of Engineering ML-Enabled Software Systems in the Deep Learning Era. ACM Computing Surveys 57(10), 1–35 (2025)

6. Belani, H., Vukovic, M., Car, Ž.: Requirements Engineering Challenges in Building Ai-Based Complex Systems. In: 2019 IEEE 27th international requirements engineering conference workshops (REW). pp. 252–255. IEEE (2019)

7. Bensoudan, H., Harrouz, A., Gendreu, M., Espin, P., Taiche, B., Danut, B.: Adaptation and Implementation of a Building Management System for a Literature Museum. In: 2024 13th International Conference on Renewable Energy Research and Applications (ICRERA). pp. 1262–1266. IEEE (2024)

8. Bhatia, M., Kumar, V.: The Role of Digital Twin in Urban Infrastructure and Building Management: A Global Scientometric Perspective. Cluster Computing 29(1), 38 (2026)

9. Bian, Y., Fu, X., Gupta, R.K., Shi, Y.: Ventilation and Temperature Control for Energy-Eficient and Healthy Buildings: A Diferentiable Pde Approach. Applied Energy 372, 123477 (2024)

10. Blum, D., Arroyo, J., Huang, S., Drgoňa, J., Jorissen, F., Walnum, H.T., Chen, Y., Benne, K., Vrabie, D., Wetter, M., et al.: Building optimization testing framework (boptest) for simulation-based benchmarking of control strategies in buildings. Journal of Building Performance Simulation 14(5), 586–610 (2021)

11. Bosch, J., Olsson, H.H., Crnkovic, I.: Engineering AI Systems: A Research Agenda. Artificial intelligence paradigms for smart cyber-physical systems pp. 1–19 (2021)

12. Boschert, S., Rosen, R.: Digital twin—the simulation aspect. In: Mechatronic futures: Challenges and solutions for mechatronic systems and their designers, pp. 59–74. Springer (2016)

13. Brazauskas, J., Verma, R., Safronov, V., Danish, M., Merino, J., Xie, X., Lewis, I., Mortier, R.: Data Management for Building Information Modelling in a Real-Time Adaptive City Platform. arXiv preprint arXiv:2103.04924 (2021)

14. Brooks, R.A.: Elephants don’t play chess. Robotics and autonomous systems 6(1- 2), 3–15 (1990)

15. Chamari, L., Petrova, E., Pauwels, P.: An End-To-End Implementation of a Service-Oriented Architecture for Data-Driven Smart Buildings. Ieee Access 11, 117261–117281 (2023)

16. Chinnakannan, A., Nilsson, O., Hussain, K., Avelar, V.: Using AI to Optimize HVAC Systems in Buildings: A Real-world Example (2021), https://myrspoven.com/wp-content/uploads/2024/09/ White-Paper-âĂŞ-Using-AI-to-Optimize-HVAC.pdf, accessed: 20.11.2025

17. Ding, X., Cerpa, A., Du, W.: Multi-Zone HVAC Control With Model-Based Deep Reinforcement Learning. IEEE Transactions on Automation Science and Engineering 22, 4408–4426 (2024)

18. Dulac-Arnold, G., Levine, N., Mankowitz, D.J., Li, J., Paduraru, C., Gowal, S., Hester, T.: Challenges of real-world reinforcement learning: definitions, benchmarks and analysis. Machine Learning 110(9), 2419–2468 (2021)

19. Eckhardt, S., Kühl, N., Dolata, M., Schwabe, G.: A Survey of AI Reliance. ACM Computing Surveys (2024)

20. Esmaeili, M., Hammes, S., Tosatto, S., Geisler-Moroder, D., Zech, P.: Safe Reinforcement Learning for Buildings: Minimizing Energy Use While Maximizing Occupant Comfort. Energies 18(19), 5313 (2025)

21. European Parliament, Council of the European Union: Regulation (EU) 2016/679 of the european parliament and of the council on the protection of natural persons with regard to the processing of personal data and on the free movement of such data (general data protection regulation). Oficial Journal of the European Union, L119 (May 2016), https://eur-lex.europa.eu/legal-content/EN/TXT/ ?uri=CELEX:32016R0679

22. European Parliament, Council of the European Union: Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Oficial Journal of the European Union, L 2024/1689 (July 2024), https://eur-lex.europa.eu/ legal-content/EN/TXT/?uri=CELEX:32024R1689

23. Fierro, G., Prakash, A.K., Blum, D., Bender, J., Paulson, E., Wetter, M.: Notes Paper: Enabling Building Application Development With Simulated Digital Twins. In: Proceedings of the 9th ACM International Conference on Systems for Energy-Eficient Buildings, Cities, and Transportation. pp. 250–253 (2022)

24. Fischer, L., Ehrlinger, L., Geist, V., Ramler, R., Sobiezky, F., Zellinger, W., Brunner, D., Kumar, M., Moser, B.: AI System Engineering—Key Challenges and Lessons Learned. Machine Learning and Knowledge Extraction 3(1), 56–83 (2020)

25. Gezici, B., Tarhan, A.K.: Systematic Literature Review on Software Quality for AI-Based Software. Empirical Software Engineering 27(3), 66 (2022)

26. Giray, G.: A Software Engineering Perspective on Engineering Machine Learning Systems: State of the Art and Challenges. Journal of Systems and Software 180, 111031 (2021)

27. Grieves, M.W.: Digital twins: past, present, and future. In: The digital twin, pp. 97–121. Springer (2023)

28. Hammes, S., Geisler-Moroder, D., Weninger, J., Zech, P., Pfluger, R.: Market Demands vs. Scientific Realities: A Comparative Analysis in the Context of Bim-Based and User-Centred Lighting Control. Developments in the Built Environment 19, 100526 (2024)

29. Hauer, M., Hammes, S., Zech, P., Geisler-Moroder, D., Plörer, D., Miller, J., van Karsbergen, V., Pfluger, R.: Integrating Digital Twins With Bim for Enhanced Building Control Strategies: A Systematic Literature Review Focusing on Daylight and Artificial Lighting Systems. Buildings 14(3), 805 (2024)

30. He, Y., Du, Y., Guo, H., Yang, J., Sun, Y., Wang, Z., Li, C., Sun, K., Zhang, M., Shi, C., et al.: Design and Research of Intelligent Building Control System. In: IOP Conference Series: Earth and Environmental Science. vol. 632, p. 042023. IOP Publishing (2021)

31. Heiland, L., Hauser, M., Bogner, J.: Design Patterns for AI-Based Systems: A Multivocal Literature Review and Pattern Repository. arXiv preprint arXiv:2303.13173 (2023)

32. Hoenig, A., Roy, K., Acquaah, Y.T., Yi, S., Desai, S.S.: Explainable AI for Cyber-Physical Systems: Issues and Challenges. IEEE access 12, 73113–73140 (2024)

33. International Energy Association (IEA): Energy Eficiency 2024. Tech. rep., International Energy Association (2024)

34. Jabborov, A., Kharlamova, A., Kholmatova, Z., Kruglov, A., Kruglov, V., Succi, G.: Taxonomy of Quality Assessment for Intelligent Software Systems: A Systematic Literature Review. IEEE Access 11, 130491–130507 (2023)

35. Kimmel, R., Michael, J., Wortmann, A., Zhang, J.: Digital Twins for Software Engineering Processes. In: 2025 IEEE/ACM 47th International Conference on Software Engineering: New Ideas and Emerging Results (ICSE-NIER). pp. 16–20. IEEE (2025)

36. Kou, X., Du, Y., Li, F., Pulgar-Painemal, H., Zandi, H., Dong, J., Olama, M.M.: Model-Based and Data-Driven HVAC Control Strategies for Residential Demand Response. IEEE Open Access Journal of Power and Energy 8, 186–197 (2021)

37. Kumeno, F.: Software Engineering Challenges for Machine Learning Applications: A Literature Review. Intelligent Decision Technologies 13(4), 463–476 (2019)

38. Lazic, N., Boutilier, C., Lu, T., Wong, E., Roy, B., Ryu, M., Imwalle, G.: Data center cooling using model-predictive control. Advances in Neural Information Processing Systems 31 (2018)

39. Lee, C.G., Park, S.C.: Survey on the virtual commissioning of manufacturing systems. Journal of Computational Design and Engineering 1(3), 213–222 (2014)

40. Lwakatare, L.E., Crnkovic, I., Bosch, J.: DevOps for AI—Challenges in Development of AI-enabled Applications. In: 2020 international conference on software, telecommunications and computer networks (SoftCOM). pp. 1–6. IEEE (2020)

41. Mai, T.L., Kabitzsch, K.: Achieving Assisted Requirement Engineering for Building Automation Using Requirement Variants. In: 2020 IEEE 29th International Symposium on Industrial Electronics (ISIE). pp. 73–78. IEEE (2020)

42. Martínez-Fernández, S., Bogner, J., Franch, X., Oriol, M., Siebert, J., Trendowicz, A., Vollmer, A.M., Wagner, S.: Software Engineering for AI-Based Systems: A Survey. ACM Transactions on Software Engineering and Methodology (TOSEM) 31(2), 1–59 (2022)

43. Martínez-Fernández, S., Franch, X., Jedlitschka, A., Oriol, M., Trendowicz, A.: Developing and Operating Artificial Intelligence Models in Trustworthy Autonomous Systems. In: International Conference on Research Challenges in Information Science. pp. 221–229. Springer (2021)

44. Mendoza-Pitti, L., Calderon-Gomez, H., Vargas-Lombardo, M., Gómez-Pulido, J.M., Castillo-Sequera, J.L.: Towards a Service-Oriented Architecture for the Energy Eficiency of Buildings: A Systematic Review. IEEE Access 9, 26119–26137 (2021)

45. Mitzutani, I., Ramanathan, G., Mayer, S.: Semantic Data Integration With DevOps to Support Engineering Process of Intelligent Building Automation Systems. In: Proceedings of the 8th ACM International Conference on Systems for Energy-Eficient Buildings, Cities, and Transportation. pp. 294–297 (2021)

46. Molnar, C., Casalicchio, G., Bischl, B.: Interpretable machine learning–a brief history, state-of-the-art and challenges. In: Joint European conference on machine learning and knowledge discovery in databases. pp. 417–431. Springer (2020)

47. Myllyaho, L., Raatikainen, M., Männistö, T., Nurminen, J.K., Mikkonen, T.: On Misbehaviour and Fault Tolerance in Machine Learning Systems. Journal of Systems and Software 183, 111096 (2022)

48. Nahar, N., Zhou, S., Lewis, G., Kästner, C.: Collaboration Challenges in Building ML-Enabled Systems: Communication, Documentation, Engineering, and Process. In: Proceedings of the 44th international conference on software engineering. pp. 413–425 (2022)

49. Nguyen-Duc, A., Sundbø, I., Nascimento, E., Conte, T., Ahmed, I., Abrahamsson, P.: A Multiple Case Study of Artificial Intelligent System Development in Industry. In: Proceedings of the 24th International Conference on Evaluation and Assessment in Software Engineering. pp. 1–10 (2020)

50. Ozawa, Y., Zhao, D., Watari, D., Taniguchi, I., Suzuki, T., Shimoda, Y., Onoye, T.: Data-driven HVAC Control Using Symbolic Regression: Design and Implementa-

tion. In: 2023 IEEE Power & Energy Society General Meeting (PESGM). pp. 1–5. IEEE (2023)

51. Pei, Y., Yao, Y., Zhao, J., Hao, J., Ding, F., Wang, J.: Multi-Agent Hierarchical Deep Reinforcement Learning for HVAC Control With Flexible DERs. IEEE Transactions on Smart Grid (2025)

52. Pfeifer, R., Bongard, J.: How the body shapes the way we think: a new view of intelligence. MIT press (2006)

53. Rajkumar, R., Lee, I., Sha, L., Stankovic, J.: Cyber-physical Systems: The Next Computing Revolution. In: Proceedings of the 47th design automation conference. pp. 731–736 (2010)

54. Ramanathan, G., Mayer, S.: A Match Made in Semantics: Physics-Infused Digital Twins for Smart Building Automation. In: 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE). pp. 3539–3546. IEEE (2024)

55. Ramanathan, G., Mayer, S.: Integrating Semantic Web Ontologies Towards Achieving Automated Engineering of Controls in Smart Buildings. In: 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE). pp. 3036–3043. IEEE (2025)

56. Rashidi, A., Sarvari, H., Chan, D.W., Olawumi, T.O., Edwards, D.J.: A Systematic Taxonomic Review of the Application of Bim and Digital Twins Technologies in the Construction Industry. Engineering, Construction and Architectural Management 33(3), 1813–1835 (2026)

57. Santhanam, P., Farchi, E., Pankratius, V.: Engineering Reliable Deep Learning Systems. arXiv preprint arXiv:1910.12582 (2019)

58. Santos, A., Liu, N., Jradi, M.: AUSTRET: An Automated Step Response Testing Tool for Building Automation and Control Systems. Energies 14(13), 3972 (2021)

59. Santos, G., Pinto, T., Vale, Z., Carvalho, R., Teixeira, B., Ramos, C.: Upgrading BRICKS—The Context-Aware Semantic Rule-Based System for Intelligent Building Energy and Security Management. Energies 14(15), 4541 (2021)

60. Sens, Y., Knopp, H., Peldszus, S., Berger, T.: A Large-Scale Study of Model Integration in ML-Enabled Software Systems. arXiv preprint arXiv:2408.06226 (2024)

61. Serban, A., Van der Blom, K., Hoos, H., Visser, J.: Adoption and Efects of Software Engineering Best Practices in Machine Learning. In: Proceedings of the 14th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM). pp. 1–12 (2020)

62. Serban, A., Visser, J.: Adapting Software Architectures to Machine Learning Challenges. In: 2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER). pp. 152–163. IEEE (2022)

63. Shahim, N., Møller, C.: Economic justification of virtual commissioning in automation industry. In: 2016 Winter Simulation Conference (WSC). pp. 2430–2441. IEEE (2016)

64. Sklavenitis, D., Kalles, D.: A Scoping Review and Assessment Framework for Technical Debt in the Development and Operation of AI/ML Competition Platforms. Applied Sciences 15(13), 7165 (2025)

65. Steidl, M., Felderer, M., Ramler, R.: The Pipeline for the Continuous Development of Artificial Intelligence Models—Current State of Research and Practice. Journal of Systems and Software 199, 111615 (2023)

66. Stofel, P., Maier, L., Kümpel, A., Schreiber, T., Müller, D.: Evaluation of Advanced Control Strategies for Building Energy Systems. Energy and Buildings 280, 112709 (2023)

67. Taboada-Orozco, A., Yetongnon, K., Nicolle, C.: Smart Buildings: A Comprehensive Systematic Literature Review on Data-Driven Building Management Systems. Sensors (Basel, Switzerland) 24(13), 4405 (2024)

68. Talib, R., Nassif, N.: “Demand Control” an Innovative Way of Reducing the HVAC System’s Energy Consumption. Buildings 11(10), 488 (2021)

69. Tricomi, G., Scafidi, C., Merlino, G., Longo, F., Distefano, S., Puliafito, A.: From Vertical to Horizontal Buildings Through Iot and Software Defined Approaches. In: 2021 IEEE International Conference on Smart Computing (SMARTCOMP). pp. 365–370. IEEE (2021)

70. Verma, A., Murali, V., Singh, R., Kohli, P., Chaudhuri, S.: Programmatically interpretable reinforcement learning. In: International conference on machine learning. pp. 5045–5054. PMLR (2018)

71. Walczyk, G., Ożadowicz, A.: Building Information Modeling and Digital Twins for Functional and Technical Design of Smart Buildings With Distributed Iot Networks—Review and New Challenges Discussion. Future Internet 16(7), 225 (2024)

72. Washizaki, H., Uchida, H., Khomh, F., Guéhéneuc, Y.G.: Studying Software Engineering Patterns for Designing Machine Learning Systems. In: 2019 10th International Workshop on Empirical Software Engineering in Practice (IWESEP). pp. 49–495. IEEE (2019)

73. Weinberg, D., Wang, Q., Timoudas, T.O., Fischione, C.: A Review of Reinforcement Learning for Controlling Building Energy Systems From a Computer Science Perspective. Sustainable cities and society 89, 104351 (2023)

74. Wollschlaeger, B., Eichenberg, E., Kabitzsch, K.: How to Play Tag: A Formalization of Semantic Interoperability to Catch Semantics in Building Automation. In: 2020 25th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA). vol. 1, pp. 875–882. IEEE (2020)

75. Wollschlaeger, B., Kabitzsch, K.: Knowledge-based Services for Creating Functional Models of Building Automation Components. In: 2021 26th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA). pp. 1–8. IEEE (2021)

76. Yang, Z.: Integration of AI with Building Energy Management Systems for Low-Carbon Urban Development. Frontiers in Sustainable Development 5(10), 104–119 (2025)

77. Zech, P., Goldin, E., Hammes, S., Geisler-Moroder, D., Pfluger, R., Breu, R.: Model-based Auto-commissioning of Building Control Systems. In: Proceedings of the 26th International Conference on Enterprise Information Systems (ICEIS 2024). pp. 121–128 (2024)

78. Zech, P., Goldin, E., Zallinger, C., Hammes, S., Pobitzer, P., Michael, J., Breu, R.: An ecosystem of dsmls for building commissioning. In: 2025 ACM/IEEE 28th International Conference on Model Driven Engineering Languages and Systems (MODELS). pp. 12–23. IEEE (2025)

79. Zech, P., Hammes, S., Geisler-Moroder, D., Hauer, M., Pfluger, R., Breu, R.: Quantifying the simulation-reality gap in building simulations. In: Proceedings of Building Simulation 2025: 19th Conference of IBPSA. vol. 19. IBPSA (2025)

80. Zech, P., Hammes, S., Goldin, E., Geisler-Moroder, D., Breu, R., Pfluger, R.: From BIM to Digital Twin: A transformation process through advanced control modeling and automated commissioning using daylight and artificial lighting as examples. Energy and Buildings 329, 115184 (2025)

81. Zech, P., Jäger, A., Schneiderbauer, L., Exenberger, H., Fröch, G., Flora, M.: Agile construction digital twin engineering. Buildings 15(3), 386 (2025)

82. Zech, P., Senoner, S., Goldin, E., Zallinger, C., Hammes, S., Michael, J.: Modeldriven digital twins for aeco. In: 2025 ACM/IEEE 28th International Conference on Model Driven Engineering Languages and Systems Companion (MODELS-C). pp. 224–235. IEEE (2025)

83. Zhang, C., Kuppannagari, S.R., Kannan, R., Prasanna, V.K.: Building HVAC Scheduling Using Reinforcement Learning via Neural Network Based Model Approximation. In: Proceedings of the 6th ACM international conference on systems for energy-eficient buildings, cities, and transportation. pp. 287–296 (2019)

84. Zhong, X., Zhang, Z., Zhang, R., Zhang, C.: End-To-End Deep Reinforcement Learning Control for HVAC Systems in Ofice Buildings. Designs 6(3), 52 (2022)