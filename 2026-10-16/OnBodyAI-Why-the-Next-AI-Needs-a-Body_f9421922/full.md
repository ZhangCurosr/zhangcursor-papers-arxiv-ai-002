# OnBodyAI – Why the Next AI Needs a Body

MICHAEL BEIGL, Karlsruhe Institute of Technology (KIT), Germany

The next AI will not win by knowing more about the world. It will win because it knows you: because it lives on your body, works for you and answers to you. That is what the term OnBodyAI stands for. It describes what happens when large language models are given a body, and what research and industry need to make that happen. This article is based on my talk in the Industrial Panel at UbiComp/ISWC 2026 in Shanghai (October 13, 2026) and follows its ten key statements.

Additional Key Words and Phrases: OnBodyAI, wearables, earables, hearables, large language models, closed-loop systems, embedded machine learning, open hardware

## 1 AI HAS A BRAIN, BUT NO BODY

“LLMs know everything about the world – and not much about you, right now.” That is the starting point. The wearable and ubiquitous computing community has long been working on connecting wearables to language models: Health-LLM (Kim et al., 2024), for example, showed that LLMs can predict health outcomes from wearable data. Google’s PH-LLM coaches sleep and fitness from wearable data, as reported in Nature Medicine (Khasentino et al., 2025), SensorLM (Zhang et al., 2025) learns a language for sensor signals, and Apple trained a foundation model on behavioral data from the watch in 2025 (Erturk et al., 2025). Sensor data in, advice out.

And yet everyday reality looks like this: if you ask a state-of-the-art assistant whether to push through tonight or take a break, you get a polite but useless counter-question. How did you sleep? Is your heart racing? How stressed are you? The smartest model in the world has to ask me how I feel, while a few centimeters from my eardrum a sensor already knows my heart rate, my sleep, my jaw clenching and the tension in my voice. Today’s AI has a brain, but no body.

![](images/98c79347dbca09d1e98b96a786e02eabe88096b2b6869e6fcc6504d9a5755460.jpg)  
Fig. 1. The assistant asks, while the ear already knows the answer. Own illustration (Beigl/TECO).

For industry the consequence is this: personalization based on physical context is the next diferentiator of AI, and body data is the gap that so far has been only weakly occupied.

## 2 THE OTHER HALF: WE HAD THE WEARABLE, WE LACKED THE BRAIN

“In 1999 we could already build context-aware things. What we lacked was the brain.” The MediaCup (Gellersen et al., 1999), presented at HUC ’99, the predecessor of today’s UbiComp, carried sensors, radio and battery in the base of a cofee cup. It knew when the cofee was hot and when a meeting was going on. But everything relied on hand-written rules. It could not learn anything new.

In the 25 years since, our community has learned to read the body like no one else: which signals exist, where to measure them and how to build devices that people actually wear. Today OpenEarable 2.0 carries seven sensors on the ear, and the interpretation is learned and runs on the device. What was missing is the brain, and AI now provides it. AI has a brain but no body; we have the body. Together, that is OnBodyAI.

![](images/496fc613a6a4509ac23f3d21408e961c3e928d829aed813ae8e8607bc01ba38a.jpg)  
Fig. 2. From the MediaCup (HUC ’99) to OpenEarable 2.0. Own illustration and devices of TECO/KIT.

Industry perspective: 25-year-old ideas are finally shippable. For on-body AI, now is the product window.

## 3 GREAT POTENTIAL IS NOT YET A SUCCESSFUL PRODUCT

AI may be the biggest opportunity wearables have ever had. But a cool technology with some idea built on top has never been enough to make a wearable a hit. The wearable graveyard is full of brilliant ideas:

• Google Glass (2013–15) bet on miniaturization: a display and a camera on your face. It asked people to look diferent and made everyone around them uneasy. The Explorer program ended in 2015.

• Microsoft Band (2014–16) bet on health sensing, but was stif and uncomfortable to live with. It was discontinued in 2016.

• Humane AI Pin (2024–25) bet on AI itself: talk to your lapel, read a projection on your palm. In February 2025, less than a year after launch, the product was shut down when the company was sold to HP.

Each of these rode a real technology wave, and each asked people to change. The winners instead upgrade what people already wear: earbuds that now measure heart rate, rings for sleep, watches for health. Same object, same habit, new intelligence.

From an industry perspective, intelligence therefore belongs in form factors people already wear – no new device, no new habit.

## 4 THE EAR IS THE NEW WRIST

One of these objects is especially interesting: the ear. IDC (2024) counted about 302 million hearables out of roughly 504 million wearables shipped in 2023, that is about 60 percent. In September 2025 Apple added heart-rate sensing and live translation to the AirPods Pro 3. An installed base of around 300 million earbuds a year is waiting to become a sensing platform.

Research is following this trend. In our open database EarXplore, publications on ear-based gestures grow from year to year.

![](images/71331447dcc1fa8f03e9e2fc2cf2f4c50b52d27989e3fe2ed5142ba2d2f1bc9a.jpg)  
Fig. 3. Papers on ear-based gestures per year. Source: EarXplore (Hummel et al., 2025), own analysis.

Why the ear? It is close to vessels, jaw, eyes and brain, it sits stably, and people wear earbuds for hours anyway. At the ear you are inside and outside the body at the same time, like at few other places on the body. Vessels provide pulse, oxygen saturation and blood pressure; bones carry voice, chewing and steps; muscles – jaw, face, even the tensor tympani – reveal tension; and electrical signals (EOG, EEG) can be derived from eyes and brain. Our systematic review found more than 30 phenomena that can be sensed at the ear (Röddiger et al., 2022). The body is the richest sensor AI has never had, and the ear is the ideal place for measuring such body signals.

From an industry perspective, clinically capable signals from consumer earbuds open up the health market – if they hold up all day.

## 5 BEING ON THE BODY IS NOT THE SAME AS UNDERSTANDING THE BODY

A reality check with three numbers from the literature:

• Sleep stages: Six commercial wrist wearables reach a Cohen’s kappa of only 0.21 to 0.53 against polysomnography. That is not random guessing, but only fair to moderate agreement, far from “substantial” and, above all, below what medical diagnoses would require (Schyvens et al., 2025).

• Energy expenditure: A systematic review of 65 studies found a mean absolute percentage error above 30 percent for every wearable brand tested (Germini et al., 2022).

• Blood glucose from PPG, an example from our own research: models that look good on randomly mixed data collapse to R² 0 on people they have not seen, no better than≈ predicting the mean. Yet more than 90 percent of the estimates fall into the “clinically acceptable” Clarke zones A and B, even for the plain mean predictor (Ramesh et al., 2026). Our metrics hide the failure.

![](images/4932375eed22eeff525b99a648e5d5d104889c1222bd13054a61c32b903169bd.jpg)  
Fig. 4. Where body sensing reaches its limits today. Own illustration based on the studies cited.

AI is often seen as the solution to every problem, but there are limits. A (bigger) model cannot reconstruct information the sensor never captured. Contact, motion and signal quality limit what any model can know. Coupling and signal quality are therefore AI problems, not implementation details, and every model must be validated on unseen people, situations, recording environments and so on, not just on random splits. Only then will generalizable “foundation” models emerge.

## 6 CLOSING THE LOOP: AI GIVES YOU FREEDOM

Because sensing is imperfect, acting on the body must happen on the user’s terms. When it does, something remarkable happens: the system does what I want before I have to think of it – and relieves me of cognitive efort. The loop runs on my body, under my rules, without outside influence, and takes load of me. Two closed loops from our lab show what this looks like:

• Calmables: A smart ring measures heart rate, and infrared earables gently warm the ear when the heart rate indicates a need to relax. Sense on the finger, act at the ear.

• EarStreAM: OpenEarable 2.0 measures heart rate and heart-rate variability and detects rising stress. An LLM then generates a meditation that adapts in real time to how the body responds.

The language model no longer has to ask me how I feel. It reads it from my body and acts, under my rules. Trust in the system is, however, decisive for whether users hand over control.

![](images/052290ded62eef73fc3c0f99cf7265fd49f5cf195efa60fd4543290b2790aad0.jpg)  
Fig. 5. Two closed loops from our lab. Posters of the UbiComp/ISWC 2026 demo session (Calmables: Zitz et al., 2026; EarStreAM: Hummel et al., 2026). Own illustrations.

The closed loop is currently the subject of both products and research. Elemind, an MIT spin-of, has sold an EEG headband since 2024 that measures brain waves in real time and plays short sounds via bone conduction, timed to the phase of your own alpha waves; once sleep is detected, it fades out. Apple’s AirPods Pro became the first FDA-authorized over-the-counter hearing aid software in September 2024: a hearing test in the earbud tunes the amplification to your own ears. Hundreds of millions of earbuds can thus act medically on the body. And research is moving on: our Heatables warm the ear canal with infrared LEDs, and participants felt about 1.5 °C warmer (Zitz et al., 2025). Thermal comfort could be the next feature in an earbud.

![](images/81c992fcff3b384dddc86e87da8e3e1cc8a4794062379f5f05e5406b6c7f4f33.jpg)  
Figure 1: Visual overview of experimental conditions and prototype design, captured using a Testo 883-1 infrared thermal camera. (A) Infrared image of a participant wearing the placebo device after 45 minutes of cold exposure. (B) Infrared image of a participant wearing the active Heatables device after 45 minutes of cold exposure, highlighting localized warming around the ear region. (C) Photograph of a participant wearing the Heatables device. (D) Cross-sectional schematic of the Heatables prototype.  
Fig. 6. Heatables, experimental conditions and prototype (Zitz et al., 2025, Fig. 1).

From an industry perspective, this opens up an interesting potential: head and ear become output channels – for sleep, hearing and comfort. It is important to note that users buy outcomes, not sensors, so the channel must deliver lasting value to the user, otherwise a product will not succeed.

## 7 ON THE BODY, MILLIWATTS MATTER MORE THAN MODEL PARAMETERS

A wearable must run all day on a battery the size of a fingernail. A data-center GPU draws around 700 watts, a smartphone SoC a few watts. A whole earbud – sensors, radio, audio and compute together – has on average about 30 milliwatts if it is to last a working day. That is four orders of magnitude of diference.

![](images/426b39d45d4df6b19a100ca4937c8823e9120ee216f109aba43e3d5052236d65.jpg)  
Fig. 7. Power budget comparison (own illustration, Beigl/TECO). GPU: NVIDIA H100 SXM TDP; earbud: \~50 mAh battery over \~6 h of use.

Every additional milliwatt costs heat that must be dissipated on or near the skin, and volume on the body, for example in the ear. Simply shrinking AI models, for example by quantization and pruning, is not optimal, because the properties of the embedded processors used are not suficiently taken into account. This concerns both the runtime of the processor operations used and their energy consumption, since the model was designed for a diferent processor. Instead, it is necessary to have the AI model designed (automatically) specifically for the target processor. Our MicroNAS (King et al., 2025) approach is a first step in this direction. From an industry perspective this step is critical: battery life sells wearables, because users are very demanding in this area.

## 8 AI EATS SOFTWARE. WEARABLES ARE STILL BUILT BY HAND

App code, backends and even ML pipelines are increasingly written with AI assistance. But a wearable is not an app. Building one is a chain of hand work, interrupted by reality at every step. It often starts with soldering fine-pitch parts under the microscope, or at the latest with testing factory-made prototypes. Often the circuit initially does not behave as a simulation would suggest. Or a part you designed in is no longer available, the substitute is a fraction of a millimeter larger, and suddenly two components are in each other’s way. And so on. On top come further real-world influences, such as noise on a line where there should be none. One more hand-soldered fix wire, and again.

![](images/f604794f3158aafe26f1a93f5f04e68c3b156777557e37682e6a3d248a8f47cc.jpg)  
Fig. 8. A typical intermediate prototype state in the lab (TECO/KIT).

No computer does this chain, and (not yet) any robot can support many of the important steps. Most of it is human labor, and it is scarce: embedded engineers are among the hardest to find and most expensive engineers (as David Cuartielles of Arduino put it in his keynote at the OpenWearables workshop 2025 at UbiComp/ISWC). Wearables profit enormously from AI, but building them remains the bottleneck. Our answer is to automate at least part of the embedded stack. edge-ml.org takes you in the browser from recording to a model running on the wearable, without writing firmware. The biggest lever of all: do not start from scratch, share the hardware. For industry, automating here is decisive for cost and time-to-market – and open hardware accelerates it further.

## 9 SIGNALS BECOME MEANING – BUT ONLY IF THE HARDWARE PROVIDES THE BASIS

Body signals are weak, noisy and personal: motion artifacts, individual anatomy, noise. This is exactly where learned models are strong. Three examples from our lab:

• EarRumble: Between 36 and 50 percent of people can voluntarily contract the tensor tympani, a muscle in the middle ear. It makes a dull rumble that an in-ear pressure sensor picks up. Our classifier reaches 95 percent. Users called it “almost telepathic” (Röddiger et al., 2021).

• earEOG: Electrodes around the ear track horizontal eye movement in a normal headphone form factor. Vertical movement does not work yet, and we say so (King et al., 2025).

• OpenEarable ExG: Open hardware that captures EEG, EMG and EOG from the ear (Lepold et al., 2024).

![](images/d400051a9b4d0390c7fe14d46b1dfaf11960dcab1320bd1e0340337c26539ce6.jpg)  
Fig. 9. From signal to meaning. Own illustration based on EarRumble, earEOG and OpenEarable ExG.

But the model can only learn what it gets. A closed earbud gives you steps and heart rate. None of the three examples would have been possible on a closed device. Hardware quality is the door opener. For industry this creates a balancing act: raw sensor access is a product decision, and closed interfaces limit what AI can learn. That shuts out competition. In such a dynamic field, however, it also limits the impact of the system.

## 10 CLOSED SYSTEMS GO TO ONE MARKET, OPEN SYSTEMS GO EVERYWHERE

Closed systems reach the market, and that is fine: one vendor, one well-defined product, this is how industry makes its money. Open systems, by contrast, go everywhere: into labs we have never heard of, into clinics, into startups, into applications the vendor never imagined, and into many products. And they create trust, which matters to users (see above). A device that sits on your body all day and listens to your heart and your voice must be trustworthy. Trust needs openness: open hardware and software make results reproducible across labs, make privacy claims auditable and speed up product development. Hardware is going the way software went – RISC-V, OpenTitan, TinyTapeout.

Our OpenEarable family is an example: OpenEarable 2.0 is a fully open-source AI platform for ear-based sensing (hardware and software under the MIT license; Röddiger et al., 2025), OpenEarable ExG provides EEG, EMG and EOG from the ear, and OpenEarable BCI extends the platform to brain-computer interfaces.

![](images/53a59dbc8788aee99dcd67190b704ba99a8f4b5a107ce827a5d44819fa42fdae.jpg)  
Fig. 10. The OpenEarable family from KIT: 2.0, ExG and BCI. Own devices and illustrations.

Openness is no longer a niche. Across the community there are good examples of devices: the τ-Ring from Tsinghua University (Tang et al., 2025), BioGAP-Ultra from ETH Zürich (Frey et al., 2025) with headband, chest band and EMG sleeve (all design files open), OpenBCI for EEG, and, on the product side, Omi and the PineTime smartwatch as open products. Since 2024 the field has a home in the OpenWearables workshop at UbiComp/ISWC, which we co-organize with Tsinghua, Siegen, UC Davis and DFKI.

## 10.1 Open platforms are the on-ramp to real markets

The idea that openness is the opposite of products is a misconception. Three examples that are not ours:

• Loop: People with type 1 diabetes and their families built an open-source closed loop that reads the glucose sensor and doses insulin. In January 2023 it was FDA-cleared as Tidepool Loop: from a GitHub repository to a regulated medical product.

• Nightscout: Open-source CGM in the cloud. Parents watch their child’s glucose live on phone or watch; the community has more than 39,000 members. Their motto: #WeAreNotWaiting.

• Tympan: An OSHWA-certified open hearing-aid platform, with schematics and code on GitHub, bringing new hearing algorithms from the lab to listeners.

From an industry perspective, open reference platforms cut time and cost to prototype. As we can see, patients – partly also through the “empowerment” by AI – no longer wait: an opensource system can become a regulated product. Industry has to adapt.

## 11 CONCLUSION: GIVING AI A BODY, AND GIVING PEOPLE THE CONTROLS

The next AI will not win because it knows more about the world, but because it knows you. In an optimal setting:

1. It “lives” on you. The body is the context AI is missing, and the ear is where people already wear it.

2. It works for you. The closed loop, under your own rules, gives you freedom instead of another app to look at.

3. It answers to you. It is open, runs on milliwatts and is honest about its limits: about what it can and cannot sense.

OnBodyAI, as we understand it, is therefore not a product category but a research agenda. Its central claim is that the value of an AI system on the body depends less on the size of its mode than on the quality of the coupling between sensing, interpretation and actuation, and on whether this coupling remains under the control of the person wearing it. Making this claim operational requires progress on three fronts at once: sensing hardware that captures physiologically meaningful signals reliably in everyday life, models designed for milliwatt budgets and validated on people they have never seen, and open platforms that make results reproducible and claims auditable. Only when these come together will AI on the body move from collecting data to producing a measurable and lasting benefit for the individual.

## REFERENCES

Erturk, E., Kamran, F., Abbaspourazad, S., Jewell, S., Sharma, H., Li, Y., Williamson, S., Foti, N. J., & Futoma, J. (2025). Beyond sensor data: Foundation models of behavioral data from wearables improve health predictions. arXiv. https://doi.org/10.48550/arXiv.2507.00191

Frey, S., Spacone, G., Cossettini, A., Guermandi, M., Schilk, P., Benini, L., & Kartsch, V. (2025). BioGAP-Ultra: A modular edge-AI platform for wearable multimodal biosignal acquisition and processing. arXiv. https://doi.org/10.48550/arXiv.2508.13728

Gellersen, H.-W., Beigl, M., & Krull, H. (1999). The MediaCup: Awareness technology embedded in an everyday object. In Handheld and ubiquitous computing: First International Symposium, HUC’99 (Lecture Notes in Computer Science, Vol. 1707, pp. 308–310). Springer. https://doi.org/10.1007/3-540-48157-5\_30

Germini, F., Noronha, N., Borg Debono, V., Abraham Philip, B., Pete, D., Navarro, T., Keepanasseril, A., Parpia, S., de Wit, K., & Iorio, A. (2022). Accuracy and acceptability of wrist-wearable activity-tracking devices: Systematic review of the literature. Journal of Medical Internet Research, 24(1), Article e30791. https://doi.org/10.2196/30791

Hummel, J., Faust, L., Müller, E., Bertog, E., Zitz, V., Prill, M. J., Bennardo, L. L., Weber, L., Röddiger, T., & Beigl, M. (2026). EarStreAM: A closed-loop earable system for personalized stress-adaptive meditation. In Companion of the 2026 ACM International Joint Conference on Pervasive and Ubiquitous Computing. ACM. [DOI pending]

Hummel, J., Röddiger, T., Zitz, V., Lepold, P., Küttner, M., Prill, M., Clarke, C., Gellersen, H., & Beigl, M. (2025). EarXplore: An open research database on earable interaction. arXiv. https://doi.org/10.48550/arXiv.2507.20656

IDC. (2024). Worldwide quarterly wearable device tracker: 2023 shipments [Market data]. International Data Corporation. [Source to be checked]

Khasentino, J., Belyaeva, A., Liu, X., Yang, Z., Furlotte, N., Lee, C., Schenck, E., Patel, Y., Cui, J., Schneider, L., Bryant, R., Gomes, R., Jiang, A., Lee, R., Liu, Y., Perez, J., Rogers, J., Speed, C., Tailor, S., … McLean, C. (2025). A personal health large language model for sleep and fitness coaching. Nature Medicine. https://doi.org/10.1038/s41591-025-03888-0

Kim, Y., Xu, X., McDuf, D., Breazeal, C., & Park, H. W. (2024). Health-LLM: Large language models for health prediction via wearable sensor data. In Proceedings of the Fifth Conference on Health, Inference, and Learning (Proceedings of Machine Learning Research, Vol. 248, pp. 522–539). PMLR. https://proceedings.mlr.press/v248/kim24b.html

King, T., Knierim, M., Lepold, P., Clarke, C., Gellersen, H., Beigl, M., & Röddiger, T. (2025). earEOG via periauricular electrodes to facilitate eye tracking in a natural headphone form factor. Scientific Reports, 15, Article 32437. https://doi.org/10.1038/s41598-025-16839-z

King, T., Zhou, Y., Röddiger, T., & Beigl, M. (2025). MicroNAS for memory and latency constrained hardware aware neural architecture search in time series classification on microcontrollers. Scientific Reports, 15, Article 7575. https://doi.org/10.1038/s41598-025-90764-z

Lepold, P., Röddiger, T., King, T., Kunze, K., Maurer, C., & Beigl, M. (2024). OpenEarable ExG: Open-source hardware for ear-based biopotential sensing applications. In UbiComp ’24: Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (pp. 916–920). ACM. https://doi.org/10.1145/3675094.3678480

Ramesh, S., Neufeld, M., Küttner, M., Röddiger, T., & Beigl, M. (2026). Reassessing the feasibility of PPG-based noninvasive blood glucose level estimation. In D. Zhang (Ed.), ISWC ’26: Proceedings of the 2026 ACM International Symposium on Wearable Computers (pp. 127–133). ACM. https://doi.org/10.1145/3830727.3834821

Röddiger, T., Clarke, C., Breitling, P., Schneegans, T., Zhao, H., Gellersen, H., & Beigl, M. (2022). Sensing with earables: A systematic literature review and taxonomy of phenomena. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 6(3), Article 135. https://doi.org/10.1145/3550314

Röddiger, T., Clarke, C., Wolfram, D., Budde, M., & Beigl, M. (2021). EarRumble: Discreet hands- and eyes-free input by voluntary tensor tympani muscle contraction. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (Article 743). ACM. https://doi.org/10.1145/3411764.3445205

Röddiger, T., Küttner, M., Lepold, P., King, T., Moschina, D., Bagge, O., Paradiso, J. A., Clarke, C., & Beigl, M. (2025). OpenEarable 2.0: Open-source earphone platform for physiological ear sensing. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9(1), Article 16. https://doi.org/10.1145/3712069

Schyvens, A.-M., Peters, B., Van Oost, N. C., Aerts, J.-M., Masci, F., Neven, A., Dirix, H., Wets, G., Ross, V., & Verbraecken, J. (2025). A performance validation of six commercial wrist-worn wearable sleep-tracking devices for sleep stage scoring compared to polysomnography. Sleep Advances, 6(2), Article zpaf021. https://doi.org/10.1093/sleepadvances/zpaf021

Tang, J., He, Z., Zhang, M., Geng, W., Zhou, C., Shi, W., Shi, Y., & Wang, Y. (2025). -Ring: A smart ring platform forτ multimodal physiological and behavioral sensing. In Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing. ACM. https://arxiv.org/abs/2508.00778

Zhang, Y., Ayush, K., Qiao, S., Heydari, A. A., Narayanswamy, G., Xu, M. A., Metwally, A. A., Xu, S., Garrison, J., Xu, X. Althof, T., Liu, Y., Kohli, P., Zhan, J., Malhotra, M., Patel, S., Mascolo, C., Liu, X., McDuf, D., & Yang, Y. (2025). SensorLM: Learning the language of wearable sensors. arXiv. https://doi.org/10.48550/arXiv.2506.09108

Zitz, V., Küttner, M., Hummel, J., Inoue, T., Röddiger, T., & Beigl, M. (2026). Calmables: Demonstrating closed-loop infrared earables for thermal biofeedback and relaxation support. In Companion of the 2026 ACM International Joint Conference on Pervasive and Ubiquitous Computing. ACM. [DOI pending]

Zitz, V., Küttner, M., Hummel, J., Knierim, M. T., Beigl, M., & Röddiger, T. (2025). Heatables: Efects of infrared-LEDinduced ear heating on thermal perception, comfort, and cognitive performance. In Proceedings of the ACM International Symposium on Wearable Computers (ISWC ’25) (pp. 91–97). ACM. https://doi.org/10.1145/3715071.3750421

Further sources (no DOI): Apple Inc. (2024, September 12): FDA De Novo authorization of the hearing aid feature of AirPods Pro [source to be added]. Elemind Technologies: EEG headband [product page to be added]. OpenWearables workshop: OpenWearables 2025. More TECO publications: https://www.teco.edu/publications.