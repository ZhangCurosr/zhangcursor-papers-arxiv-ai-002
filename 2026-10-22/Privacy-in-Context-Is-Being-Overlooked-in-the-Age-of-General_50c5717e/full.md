# Privacy in Context Is Being Overlooked in the Age of General-Purpose LLMs

Prakhar Ganesh <sup>1</sup> <sup>2</sup> <sup>\*</sup> Yash More <sup>1</sup> <sup>2</sup> <sup>\*</sup> Marco Romanelli <sup>3</sup> Ferdinando Fioretto <sup>4</sup> Golnoosh Farnadi <sup>1</sup> <sup>2</sup>

## Abstract

As Large Language Models (LLMs) continue to grow in scope and capability, privacy concerns are on the rise. Since users increasingly interact with a single general-purpose system across many distinct contexts, there is a natural tendency to pursue a one-size-fits-all solution to privacy. Unfortunately, this overlooks a crucial insight from decades of privacy research: privacy is inherently contextual. In this paper, we argue that existing work in prompt privacy is drifting towards attempts to design a unified framework, which is highly misleading and can foster a false sense of security. Furthermore, we argue that even future work in this direction needs to wrestle with fundamental tensions across various desirable properties of prompt privacy. Thus, such continued attempts towards a unified framework need to be reconsidered. We recommend three critical paths forward: application-specific deployment of LLMs, context-specific evaluation of privacy frameworks, and better privacy literacy.

## 1. Introduction

Large Language Models (LLMs) have become embedded in daily workflows, ranging from coding assistants to legal advisors and medical support tools [12, 56, 111]. With their growing capabilities, LLMs have fundamentally changed how users interact with third-party systems [57, 91]. This shift has led to a significant increase in the amount of sensitive information users share, whether due to the personalized and open-ended nature of conversations, better interactivity, or simply a lack of privacy literacy [10, 57, 108].

Unsurprisingly, concerns around privacy in LLMs are on the rise [23]. Several solutions have been proposed, including anonymization-like techniques, differential privacy, cryptography, and various attempts of operationalizing the principles of contextual integrity [16, 24, 41, 58, 67, 79, 107]. The goal, it seems, is a user-facing, plug-and-play framework that can be seamlessly integrated into any chatbot.

![](images/1301a2fe6ea8aa14d2dd7f83deda72bc098025f5a612b7944314cf493836128c.jpg)  
Figure 1. Five desirable properties of a prompt privacy framework (§ 2), coverage of various existing frameworks in the literature (§ 6), and several non-trivial underlying tensions (§ 7).

This evolution in prompt privacy literature reflects a broader trend toward unified solutions. Promises like “Secure your AI. Everywhere it matters” [1], “Privacy firewall for Chat-GPT prompts” [3], “comprehensive tool designed to fortify the security of LLMs” [75], etc., position these solutions as universal safeguards against all privacy risks. Clearly, as LLMs become general-purpose, research in prompt privacy attempts to follow suit. However, a one-size-fits-all approach to prompt privacy is deeply flawed.

Building on existing privacy literature [21, 22, 43, 63, 74], we identify five key properties for prompt privacy (see Figure 1): three quantifiable (protection guarantees [89, 96], performance [16, 86], and efficiency [29, 34]) and two qualitative (domain adaptability [8, 58] and user accessibility [1, 3, 107]). We uncover several fundamental tensions, both between quantifiable properties (balancing protection, performance, and efficiency) and between qualitative properties (inherent accessibility concerns for adaptable frameworks capable of handling various privacy requirements). Hence, a unified solution to prompt privacy is not a purely technical problem that can be solved in the future.

We argue that privacy in context is being overlooked in existing research on prompt privacy (with spillover effects in the industry), drifting instead towards a unifiedframework.

![](images/18629c5109d598cbae60641acc86d2dc6959b63345ebd468312e29ca21c8ad95.jpg)  
Figure 2. Prompt Privacy Pipeline and a Glossary of terms used throughout the paper. The user interacts with a privacy tool/transform before their query is sent to the LLMs. All privacy interventions occur within a local trusted environment, ensuring that only the protected query is transmitted. We assume both the LLMs and the communication channel can have malicious actors. The final reconstructed output is compared against the hypothetical output of the original query to make sure the LLM utility is preserved.

However, no single framework for prompts can simultaneously achieve all desirable properties of privacy, due to non-trivial underlying tensions. Instead, presenting such a unifiedframework will mislead andjeopardize end users.

We structure our argument as follows. We begin by systematically reviewing the prompt privacy literature, highlighting the drift towards a unified framework and the tendency to overlook privacy in context (§ 3). We then identify five desirable properties for prompt privacy frameworks: protection guarantees, performance, efficiency, domain adaptability, and user accessibility (§ 4), and use them to closely examine four common approaches to prompt privacy: sanitization, differential privacy based techniques, cryptographic methods, and attempts to operationalize contextual integrity, highlighting their limitations (§ 6). Building on these limitations, we finally identify several underlying tensions between these desired properties, which pose significant nontrivial challenges even for future developments (§ 7). We conclude with three key recommendations: (1) prioritizing application-specific deployment of LLMs, (2) contextspecific evaluations of prompt privacy tools, and (3) better privacy literacy and awareness for the end users (§ 10).

## 2. Anatomy of a Prompt Privacy Framework

To establish consistent terminology and clearly mark the scope of our work, we present the prompt privacy pipeline and its various components in Figure 2.

By prompt privacy, we refer exclusively to the privacy concerns associated with a user’s query to third-party LLMs. This includes not only the prompt by the user, but also any additional auxiliary information sent as part of the query, such as documents, images, code, demonstrations, etc. The adversary might be present in either the communication channel or the third-party LLMs themselves, as long as the target for information leakage is the user query.

For the same reason, any form of information leakage where the privacy target is not the query itself, even if it happens during inference, is outside the scope of our discussion. This includes extensive literature on extracting training data from LLMs during inference [10, 61], model inversion attacks through adversarial prompts [76, 110], or jailbreaking and backdoor behaviours [101, 114], among others.

![](images/ccbf3c9f9add186cbb9b974df81e24829f9d094b78f68076d31aa37cf6de9239.jpg)

![](images/06c95948cd02e8bdf31bf918e6c1767324435e9b18c97af1f08411dfebeb10d7.jpg)

![](images/21445d21cd3c57a7900803f943382cd1de728875d45b84f4b6d69d0995dd60cc.jpg)

![](images/ea1d9e231c21634de44c0b36f916bf9c3dafc2dd80ed70771dfce72edaf0f724.jpg)

![](images/d98eb33b3a1ba2bfc9768a7c1f953ce83fabf4579ed6aaa5b0db86d76259f0e9.jpg)

![](images/db6e2cbd2b7d68f16ff9811077dfe072f20e016ae1d81499ff4f6a7db9f8e2ee.jpg)  
Figure 3. (Left) Detailed annotations of literature in prompt privacy, highlighting the evidence of a drift away from privacy in context. (Right) Detailed annotations of publicly available tools for prompt privacy, highlighting similar trends of drift as the literature.

## 3. Drift Towards a Unified Framework

Our first argument builds on a drift toward unified, onesize-fits-all privacy frameworks for LLMs in both academia and actual tools. While concerns about the limitations of privacy without context have appeared in prior literature [6, 21, 22, 43, 63, 74], the phenomenon of unified frameworks for prompt privacy during inference with LLMs is distinct due to the general-purpose nature of the underlying system. In this section, we describe this drift and how it differs from prior treatments of privacy, supported by a systematic study.

## 3.1. Prompt Privacy as a Distinct Phenomenon

The argument that “privacy is contextual” is well established in the literature [66]. In this paper, however, we focus specifically on prompt privacy, i.e., the privacy of natural language inputs to LLMs and an average user’s main privacy concern when interacting with chatbots. We argue that prompt privacy differs fundamentally from previous discussions of privacy in the classical ML literature. Before the advent of general-purpose LLMs, systems were designed for specific applications, each requiring its own explicit and contextually appropriate privacy framework. While unified privacy frameworks were still critiqued in these settings [6, 21, 74], each application still had to make its own choice of framework, and the criticism was against this choice being made lazily by the developer. In other words, when deploying a model, the context is fixed and it has been shown that the choice of privacy guardrails should depend on it.

But LLMs break this paradigm. LLMs today are simultaneously responsible for a vast and ever-growing range of tasks, for instance, from simple grammatical assistance to sensitive medical or legal guidance. In this setting, the challenge is no longer the lazy choice to privacy. Instead, the workflow of LLMs inherently encourages a single privacy framework that can be applied across a vast and heterogeneous space of user interactions, making it an extremely attractive and often the most accessible option. This creates a powerful drift toward attempts to create unified privacy solutions, overlooking important lessons from privacy literature.

## 3.2. Evidence of the Drift in Academic Literature

To better understand this drift, we conduct a systematic study of the existing literature on prompt privacy. We start with four recent surveys covering a significant breadth of research on privacy in LLMs: Chen et al. [14], Edemacu & Wu [30], Li et al. [45], Smith et al. [83]. First, a broader selection of papers was made from these surveys, identifying categories that contain papers on prompt privacy, resulting in a total of 228 papers after removing duplicates. All papers were then checked manually to see their fit within the scope of our work. After filtering papers that do not focus on prompt privacy, we were left with a total of 31 papers. Each paper was then read thoroughly and annotated to identify the privacy target, the specific use case (if any), the broader category of privacy-preserving technique employed, and whether the paper included an explicit discussion of its limitations in protecting privacy. More details about paper selection, filtering, and annotation guidelines, as well as the annotations across all 31 papers, are present in § A.

Figure 3 (Left) summarizes key trends across the annotated papers. We find more than half of the papers target privacy for either the entire input prompt or its embedding representations (Figure 3 (Left)(a)), while over three-fourths propose frameworks intended to generalize across all LLMs use cases rather than addressing specific applications (Figure 3 (Left)(b)). Moreover, more than half of the papers provide no discussion of potential limitations or failure cases (Figure 3 (Left)(c)). Collectively, these findings reveal a clear drift in the literature toward unified prompt privacy frameworks, often without sufficient contextual grounding. Irrespective of whether this is driven by the pressure to publish and thus not being truthful about limitations [68] or simply a lack of attention given to privacy in context, the drift towards unified frameworks is clearly present in existing literature and can even have spillover effects as discussed next.

## 3.3. Evidence of the Drift in Prompt Privacy Tools

In addition to academic literature, we also study various publicly available tools that are (presumably) deployed by real users to protect privacy when interacting with LLMs. We started our list from github.com/pluja/awesome-privacy, and then expanded the search on the web using the terms ‘prompt privacy’ and ‘LLM inference privacy’. We found a total of 11 different prompt privacy tools from various sources. Each tool was annotated to identify the use case and the broader category of privacy-preserving technique. More details about the search, as well as the annotations across all 11 tools, are present in § A.

The trends are collected in Figure 3 (Right). Interestingly, all 11 tools described their use case as protection during general purpose chatbot usage, and we found no tools designed for specialized contexts (Figure 3 (Right)(a)). Close to half the tools prioritized local deployment of LLMs as the solution to privacy concerns, and the rest relied on either sanitization techniques or deployment on Trusted Execution Environments (TEE) using cryptography (Figure 3 (Right)(b)). This surfaces another trend worth noting: while a significant portion of literature focuses on differential privacy, these techniques are not present in real-world tools, highlighting the power of accessibility in deciding what gets adopted.

## 4. Desirable Properties of Prompt Privacy

Now that we’ve shown how privacy in context is being overlooked, we move to our second argument: a unified framework can never achieve all desirable properties of a prompt privacy framework, and instead will mislead and harm end users. We built this argument in three steps: first, in this section, we identify key desirable properties of prompt privacy. Next, we study various existing prompt privacy frameworks and identify their limitations in achieving these properties. Finally, we build on the discussion of various limitations to identify deeper underlying tensions in prompt privacy.

## 4.1. Quantifiable Properties

We define three quantifiable properties fundamental to any privacy framework, and not just prompt privacy.

Protection Guarantees. Central to a privacy framework is its ability to protect against information leakage, i.e., limiting the information that an adversary can extract through the protected query. Depending on the context, the threat model, and the information available to the attacker, different strengths and nature of privacy protection might be desirable [41, 48, 51, 73, 89, 96]. For instance, protection against honest-but-curious service providers requires different mechanisms than protection against malicious attackers. Clearly, a framework’s ability to provide protection guarantees against the relevant adversary is a critical property.

Performance. While protection guarantees are important, a good privacy framework must also match the performance of accessing the LLMs directly, otherwise the switch to a privacy framework would incur utility costs. Several different ways to quantify performance have been proposed, such as the semantic equivalence of the reconstructed and original output, the actionability of the reconstructed output, or the preservation of factual accuracy and consistency, among others [16, 41, 86]. Broadly, a good privacy framework should maintain output fidelity.

Efficiency. Efficiency relates to the computational cost and latency of the privacy tool. A good privacy framework should minimize additional cost, avoid significant latency, and scale effectively [29, 34]. By keeping the processing lightweight and responsive, one can ensure that privacy protections do not hinder the use of LLMs. Together, performance and efficiency focus on preserving usability, thus maintaining the practical viability of the privacy framework.

## 4.2. Qualitative Properties

Not all desirable properties of a privacy tool are easily quantifiable. We describe two important qualitative properties.

Domain Adaptability. Real-world scenarios demand privacy frameworks that can flexibly adapt to diverse domains, even more so with the ever-growing list of use cases for LLMs [58]. This can take many forms, such as dynamically adjusting to different expectations of privacy protection and types of adversaries, or being able to handle specialized language or conversational formats [8, 29]. Thus, the ability to adapt across various domains and LLM use cases is essential for an effective unified privacy framework.

User Accessibility. The successful adoption of any privacy framework depends on its accessibility to end users. This includes both the transparency of the framework as well as the cognitive burden of incorporating it into everyday workflows. When users can understand how their queries are transformed, they can verify that meaning has been preserved or debug cases where outputs are suboptimal, which can empower them to maintain appropriate protection. Notably, most widely adopted privacy tools tend to employ sanitization [1, 3, 18, 107] (Figure 3 (Right)(b)), a privacy framework that prioritizes accessibility. The appeal of a low barrier of adoption for the end user cannot be overstated.

Note that the qualitative properties operate at a higher level than the quantifiableproperties. For example, domain adaptability may be understood as aframework’s ability to provide appropriate protection guarantees while maintaining performance and efficiency, across diverse domains. Similarly, user accessibility is the extent to which aframework makes the relationship between performance, efficiency, and privacy understandable and controllablefor the end user.

## 5. User Profiles as Motivating Examples

To illustrate our arguments, we introduce three motivating examples, which we will reference throughout the paper.

Avery’s Marathon Preparation. Avery wants to leverage an LLM to design a personalized fitness journey and prepare for a marathon. They aim to develop dietary plans, identify exercises, and create a comprehensive strategy to run the marathon in six months. As a meticulous researcher, Avery is not bothered about the accuracy of the LLM’s suggestions, since they will verify the details independently. Their primary goal is to generate a quick initial draft of the plan. However, Avery is concerned about targeted ads if their chat history gets leaked or is sold by the third-party LLM.

Sasha’s Work Companion. Sasha’s company has recently rolled out a work companion using a third-party LLM API to assist employees with routine tasks such as summarizing reports, creating presentations, and more. Sasha plans to use this tool when working with confidential documents, so strong privacy protections are critical. A tool that requires constant supervision would be counterproductive, so it needs to be highly accurate. Sasha’s company is managing the costs, and the tool will run on a centralized infrastructure, while using the third-party LLM API underneath. Note that the trusted local environment here includes the company infrastructure, but not the third-party LLM.

Tao’s Writing Assistant. Tao is a non-native English speaker who wants to use an LLM-based writing assistant to help with personal emails and other forms of communication. Since these interactions are mostly casual, Tao is not overly concerned about privacy risks. However, they would like to ensure that no unnecessary identifying information is shared with the LLM. For Tao, accuracy and speed, allowing seamless integration of the tool into their daily life, are more important than strict privacy guarantees.

## 6. Drawbacks of Existing Frameworks

To understand why a single framework cannot simultaneously achieve all desirable properties of prompt privacy, we first enumerate the existing limitations of the four prompt privacy solution categories present in the literature (Figure 3 (Left)(d)): sanitization, differential privacy based techniques, cryptographic methods, and various attempts at operationalizing the theory of contextual integrity.

Note: While sanitization and cryptographic methods are concrete techniques, differential privacy is a mathematical concept that can be implemented in many different ways, and contextual integrity is a social theory ofprivacy. Neither of them directly refer to prompt privacy techniques. However, we overload these terms in the paper tofacilitate readability, instead of repeatedly clarifying these differences. We use the term differential privacy to refer to a broad class oftechniques that share the common property of providing formal differential privacy guarantees. Similarly, our discussions ofcontextual integrityfocus on techniques that attempt to operationalize the social theory of contextual integrity.

## 6.1. Sanitization: A Misplaced Sense of Safety

Sanitization refers to techniques that remove, mask, or generalize parts of the prompt in an effort to reduce exposure of sensitive information, as illustrated in Figure 4. These methods work at the surface level, making them efficient without degrading performance, but they lack rigorous protection guarantees. Sanitization often relies on lexical patterns and is thus highly accessible for end users.

Lexical Sanitization misses Semantic Context. Lexical sanitization [4, 40, 73, 87] is a technique that focuses on sanitizing specific words or phrases from a prompt, without considering the semantic meaning or the broader context. This encompasses common approaches such as Personally Identifiable Information (PII) removal and Named Entity Recognition (NER)-based methods. However, these techniques are frequently criticized for overlooking the complex linguistic and relational cues present in natural language [35, 82]. Moreover, since lexical sanitization methods rely on predefined lists of words or categories, they often fail to generalize across domains where sensitive information can take many different forms. As a result, what qualifies as sensitive in one setting may go undetected in another, limiting their adaptability across domains [58].

Lack of Protection Guarantees. Recent works have revealed that the effectiveness of sanitization in removing sensitive information is often overestimated. Black-box attacks on LLMs show adversaries can frequently reconstruct and infer sensitive information, even after applying sanitization or de-identification, by exploiting residual semantic cues and contextual patterns [11, 41, 49, 89, 96]. Such findings highlight that surface-level masking is rarely sufficient for robust privacy protection, especially in realworld scenarios with motivated adversaries [48]. Hence, while sanitization can create a sense of privacy by removing certain words from the sentence, they have an even greater risk of misleading users [29, 73, 89].

![](images/af89dd37a9828f7952cd31a965b86e78075bb5b5b8e486237dd1f9a22f6b46d1.jpg)  
Figure 4. Four commonly adopted frameworks for prompt privacy: sanitization, differential privacy, cryptography, and contextual integrity.

The Illusion of Personalization and Control. Sanitization tools often advertise user control over how much sensitive information is removed, balanced against task utility. This is a common design choice in user-facing frameworks such as browser extensions [18, 107] and client-side tools [112]. However, such tradeoff levers are built on the assumption that privacy and performance can be reduced to a single continuum, and that balancing one against the other is both meaningful and tractable [107, 112]. In reality, privacy risks are highly contextual, shaped by semantic and pragmatic cues [29, 107, 112], and even the interpretation of the privacy scale can vary heavily across different users [94, 96].

Sanitization frameworks offer a surface-level protection from privacy risks, along with illusory control over the balance against utility. While useful in some situations, sanitization cannot withstand motivated adversaries andfails to capture the dynamic, context-sensitive nature ofreal-world privacy across domains.

Sanitization risks exposing Avery to targeted ads and leaking company data for Sasha, making it unsuitable for them both. It may be an appropriate tool for Tao, providing good performance and efficiency, along with surface-level privacy sufficient for their needs.

## 6.2. Differential Privacy: Promises and Pitfalls

Differential Privacy (DP) is a mathematical framework of privacy in data-driven systems, and DP guarantees have long been regarded as the gold standard in traditional ML settings [28]. At its core, DP guarantees provide a powerful statement: the output of a system is indistinguishably affected by the presence or absence of a single individual data point, achieved by introducing noise. Thus, an adversary with access to the output cannot confidently predict the membership of individuals in the input data.

In prompt privacy, DP is applied to the input prompt by injecting random noise through perturbations, ensuring that the perturbed prompt could have plausibly come from many different input prompts [92], as shown in Figure 4. Thus, an adversary with access to the perturbed prompt cannot confidently state which input was used to create it.

Relative Bounds, Not Absolute Walls. DP provides relative bounds on information leakage rather than preventing the absolute flow of information. In essence, DP ensures that an adversary cannot reliably distinguish between neighboring input prompts. However, if certain information is common across all neighbouring prompts, the adversary can still infer it. For instance, suppose a user asks an LLMs about a particular medication. DP may perturb the prompt so that the medication name is replaced with that of a similar but different drug. While this prevents the adversary from confidently recovering the original medication, even knowledge of the perturbed alternative still reveals significant information about the user’s potential medical condition.

This tension is not surprising, and in fact, is very well known in the privacy literature as Dalenius’ Desideratum, i.e., in simple terms, if you can learn nothing from the data, you’ve achieved perfect privacy [28]. This trade-off between privacy and utility, thus, is not a limitation of any one particular technique, but it does appear more clearly in DP than any other class of privacy-preserving methods.

The Structural Limits of DP for Prompts. Guaranteeing protection under DP relies on having a clear definition of a “secret” or sensitive record. For language data, and especially for prompts, it is difficult to establish consistent boundaries for what should be protected. Language does not have a universal unit for sensitive information.

On the one hand, if privacy is defined at the level of an entire conversation, then the resulting guarantees can be overly coarse and impractical, since the mechanism would need to inject prohibitive amounts of noise to provide formal protection, severely degrading utility. However, on the other hand, if privacy is defined at the word level, removing these units rarely hides private information, and does not prevent an adversary from inferring sensitive context from the rest of the prompt or related patterns [8, 57].

The Difficulty of Defining Similarity of Prompts. Another challenge of defining neighbouring prompts in DP lies in choosing a meaningful similarity metric. As similarity in the semantic space is not directly quantifiable, it is often mapped into a different space, such as the embeddings. However, this introduces a gap between the measured similarity and the actual semantic similarity. Consequently, even prompts that appear safe may contain identifying details, undermining the intended privacy guarantees [24, 55].

Differential privacy offers strong guarantees against information leakage. However, its limitations are not just technical obstacles, but reflect a deeper misalignment between the rigidity of differential privacy and the dynamic nature of language.

Performance drop under DP would be unacceptable for Sasha and Tao. While Avery could benefit from the protection provided by DP, they need literacy of the privacy-utility trade-off to take its advantage.

## 6.3. Cryptography: A Costly Alternative

Cryptography in the context of privacy includes a wide range of techniques, including encryption [2], secret sharing [25, 109], zero-knowledge proofs [32], oblivious transfer [77, 98], etc. Two cryptographic techniques are common in prompt privacy literature: Homomorphic Encryption (HE) and Secure Multi-Party Computation (MPC). HE enables computation directly on encrypted text, as shown in Figure 4, and decrypts the output in the trusted environment [20, 103, 115]. MPC, on the other hand, distributes the computation across multiple independent parties, ensuring that no single party has access to the complete input [78, 97].

The Barrier of Non-Linearity. HE holds great appeal because it promises the best of both worlds: privacy-preserving encryption and full utility of the model [2]. In practice, however, it is highly constrained. HE is relatively efficient for linear operations, such as the large matrix multiplications. The challenge arises with nonlinear operations like ReLU, softmax, and the attention mechanism itself, which can either make HE extremely slow or require polynomial approximations that quickly become unstable and expensive [59, 97, 105]. Thus, HE does not scale well for LLMs.

Specialized Infrastructure Requirements. A major barrier to the use of cryptographic techniques for prompt privacy in LLMs is the need for specialized infrastructure. MPC requires multiple non-colluding servers to jointly host and run the model, with heavy inter-server communication during inference. This is at direct odds with the way LLMs are deployed today, as centralized services optimized for singlenode or tightly clustered GPU execution. Similarly, LLMs need to be reimplemented to be HE-compatible, fundamentally changing the serving pipeline itself. Thus, the existing LLMs infrastructure does not support these techniques out of the box, and would require substantial re-engineering.

Hybrid Solutions and Communication Overhead. Recent works have attempted to combine HE and MPC, taking advantage of the beneficial aspects of each to create hybrid techniques [51, 72, 97]. While this has provided significant improvements, even under optimistic assumptions of specialized infrastructure and highly tuned cryptographic libraries, hybrid techniques still introduce significant communication costs that accumulate quickly. Consequently, although powerful, they remain far too expensive for LLMs.

Cryptographic techniques for prompt privacy promise protection guarantees while maintaining model performance, however,fall short oftrue practical benefits due to their extremely high computational costs, limiting their use to toy models or shallow networks.

The high cost of cryptographic techniques makes it unacceptable for Avery and Tao. As Sasha’s work companion is deployed through the company infrastructure, they could potentially benefit from the guarantees it provides against absolute information leakage.

## 6.4. Contextual Integrity: A Distant Ideal

Contextual Integrity (CI) [58, 66] offers a principled way to analyze and adapt privacy norms as technology and society evolve. CI is the only framework in our discussion that defines privacy not as a static property but as the appropriateness of information flow within a given context, by examining who is sharing information, what is being shared, with whom, for what purpose, and under what transmission principles [58, 66, 80], an example shown in Figure 4.

While CI moves beyond binary public/private distinctions, recent critiques note that research in LLMs often “washes” CI in an attempt to operationalize it, adopting CI terminology while neglecting its core principles [8, 80]. Thus, translating CI into practice remains a significant challenge. Several arguments in this section build on Shvartzshnaider & Duddu [80]’s position that CI is inadequately applied to LLMs in the existing literature.

Not Just a Special Case of Sanitization. Existing attempts to operationalize CI tend to encode fixed privacy norms, drawing from legal codes, crowdsourced inputs, or author judgment [17, 33, 47, 79]. In fact, many approaches that operationalize CI do so through prompt sanitization [39, 65, 80]. As a result, these tools struggle to adapt to changing social contexts, evolving notions of sensitivity, and the diversity and dynamism inherent in real-world communication [19, 29, 80].

The Difficulty of Defining Social Norms. CI is frequently reduced to regulatory compliance or minimization, losing the framework’s distinctiveness [80], thus focusing on preventing the leakage of a fixed set of sensitive information or enforcing policy rules. However, true CI analysis should holistically assess the appropriateness of information flow in context, considering roles, values, and functions, not merely whether “private” information was shared [7, 80, 99]. Laws and regulations, while influential, do not always reflect lived or moral norms in society, and privacy preferences collected from users may fail to capture the collective or ethical nature of privacy that CI intends to protect [64].

Contextual integrity aims to study the dynamic nature of information flow in prompt privacy. However, translating this into a practical tool remains a challenge, particularly without losing the very fluidity that makes it valuable. As Shvartzshnaider & Duddu [80] argue, contextual integrity is a social theory, and requires more than just purely algorithmic solutions.

The high barrier to entry makes contextual integrity an unlikely choice for Tao. For others, it could be effective, for example, keeping Avery protected from targeted ads while still delivering a useful strategy, or preventing Sasha from sharing certain confidential documents. However, such context-specific tools should already exist and be accessible to the end users.

## 7. Underlying Tensions in Prompt Privacy

We have examined the existing shortcomings and challenges across various techniques for prompt privacy. Building on this, we now complete our second argument by identifying several underlying non-trivial tensions inherent to prompt privacy for LLMs that would prevent the development of a unified framework even in the future.

## 7.1. The Privacy-Utility-Efficiency Trade-off

We start by briefly acknowledging the fundamental privacy–utility–efficiency trade-off even beyond the context of LLMs. For data to remain useful, it must retain some information, which inevitably leads to some degree of privacy leakage. At the same time, identifying the optimal balance between privacy and utility is itself highly complex, and achieving increasingly refined trade-offs often comes at a cost, thereby reducing efficiency. These tensions have dominated the privacy literature for decades [28, 54], so we do not elaborate on them further here. Instead, we focus on tensions that arise specifically in prompt privacy.

## 7.2. Natural Language or Embedding Representations

Regardless of the underlying model, Natural Language Processing (NLP) systems transform language into vector representations through embeddings. This transformation from natural language to representation creates an inherent disconnect that generates tension between two critical dimensions: protection guarantees and performance.

The core challenge stems from differences in defining “similarity” across the two spaces. Semantic similarity in language space may not always align with similarity in the representational space (§ 6.2), affecting both protection guarantees and utility. Privacy frameworks that focus on the representations, such as DP (§ 6.2) and cryptography (§ 6.3), can provide formal guarantees, but only in the representation space. To truly prevent sensitive information flow, these techniques need to ‘over-protect’ in the representational space, to ensure appropriate protection in the language space, at the cost of degraded performance (DP, § 6.2) or reduced efficiency (cryptography, § 6.3). On the other hand, frameworks that operate in the language space, such as sanitization (§ 6.1), can maintain performance but lack any protection guarantees.

This is more than a trivial limitation of existing solutions. There is a tension between stronger guarantees in the representational space (at the cost of performance or efficiency) or better performance and efficiency in the language space (at the cost of formal guarantees). Privacy preserving frameworks must choose between these two distinct spaces that are likely to remain fundamentally irreconcilable, even in the long term (the ‘symbol grounding problem’ or in its modern form, the ‘vector grounding problem’ [38, 60]).

## 7.3. User Accessibility or Domain Adaptability

The tension between user accessibility and domain adaptability emerges from the remarkable breadth of applications for which LLMs are now employed. This diversity of use cases creates vastly different privacy requirements across domains (§ 6.1, 6.4). Unfortunately, a framework capable of truly adapting to this variety of domains and covering all different requirements necessarily demands significant privacy literacy from users. Thus, the accessibility and ease of use of a privacy framework stand in fundamental opposition to the comprehensive adaptability required for LLMs. This creates an unavoidable choice: privacy frameworks can either be accessible and easy to use, or they can be highly adaptable across domains, but not both.

Accessible frameworks like sanitization (§ 6.1) require minimal cognitive overhead. However, such frameworks struggle with domain adaptability due to changing specialized terminology across fields, evolving social norms, and varying definitions of what constitutes sensitive information in different contexts (§ 6.1). On the other extreme, contextual integrity is specifically designed for cross-domain adaptability (§ 6.4). However, it demands domain-specific expertise, an understanding of how privacy situations are constructed, and how sensitivity is defined in various contexts.

Again, the tension is not limited to current approaches, but is instead inherent to prompt privacy. This mirrors the popular ‘convention vs configuration’ argument, fundamental to the world of product design [36]. As LLMs continue to expand into new domains with unique privacy requirements, a framework attempting to maintain both accessibility and comprehensive domain adaptability will face difficult tradeoffs between these competing demands.

## 8. The Complete Argument

Through systematic analysis, we showed that privacy in context is being overlooked in the age of general-purpose LLMs, both in academic research and in the privacy tools adopted by users (§ 3). We then argued that this shift toward unified privacy frameworks is fundamentally problematic. By examining existing prompt privacy techniques, we identified the limitations of each approach and showed that no current method can simultaneously satisfy all desirable properties of a prompt privacy framework (§ 6). Looking beyond existing solutions, we then identified several deeper, non-trivial tensions that suggest these limitations are unlikely to disappear even in the future (§ 7). Thus, treating privacy in LLMs as one uniform problem is misleading and risks causing real harm by giving users an inaccurate understanding of the protections their chosen techniques actually provide.

## 9. Alternative Views

Below, we briefly discuss three potentially alternative views.

Unified Frameworks are Necessary, even if Imperfect. A reasonable argument against our paper is that unified frameworks, though imperfect, are necessary given the rapid adoption of LLMs. If the field starts to focus exclusively on context-specific solutions, this may create more harm, as users will face an overwhelming number of choices based on the context, and might end up not choosing at all. At least an imperfect framework is better than none.

Indeed, a baseline framework can be important, and we don’t argue that research in this direction should be abandoned. However, an imperfect framework is not necessarily preferable to having none. In some cases, such frameworks may create a false sense of security, leading users to become complacent and more willing to share sensitive information than they would if they were aware of potential risks. This naturally leads to our second objection to this argument: even when such baselines or “unified” frameworks are implemented, their limitations must be clearly and explicitly communicated, something heavily undervalued in the current prompt privacy literature (Figure 3 (Left)(c)).

Unified Frameworks are Stepping Stones. An argument can be made that prompt privacy is a young field and, like any emerging area of research, unified frameworks are merely a stepping stone. The field will naturally diversify as it matures, and from this perspective, privacy in context is not being overlooked but is simply underdeveloped.

We do not aim to make any claims on how the field might evolve. However, we argue that the rapid deployment of LLMs in real-world applications has already translated research norms into commercial products. Many tools now market themselves as a “one-stop” privacy solution [1, 3, 75], reflecting and reinforcing the same sensibilities that we criticize in the academic literature (Figure 3 (Right)(a)). Thus, we cannot wait for the field of prompt privacy to naturally grow towards privacy in context, as it is already shaping the development of real-world tools, and an intervention is both necessary and urgent.

Context can be Learned. We do not provide an extensive counterargument here, as Shvartzshnaider et al. [81] discusses this issue in depth: contextual frameworks cannot be standalone technical solutions, but need to be sociotechnical approaches that require broader ecosystem-level efforts.

## 10. Where do we go from here?

To carve a path forward, we bring the reader’s attention back to our motivating examples in § 5. Throughout our discussion, we found that even though unified frameworks are out of reach, certain frameworks can benefit these users in concrete scenarios. We summarize these solutions in Figure 5. Two key points deserve emphasis and form the basis of our recommendations: first, effective applicationand context-specific solutions are possible even without a universal framework; second, improving end users’ privacy literacy is essential, both to help them select the appropriate solution and to understand its limitations.

## 10.1. Context-Specific Evaluations

The first step toward choosing a privacy tool is recognizing the context-specific requirements of the application. Understanding what the user needs, what trade-offs they are willing to accept, and which privacy risks are most relevant helps guide the selection of the most appropriate tool. Theories such as contextual integrity [66] can be particularly useful in this process. The “Detailed Requirements” in Figure 5 illustrates this exercise well.

Once the requirements are identified, the next challenge is assessing which privacy framework performs well in that setting. For instance, based solely on the properties of DP, it is unclear whether it can provide sufficient protection against targeted ads for Avery. Similarly, the boundary of when sanitization can be problematic in email writing for Tao remains poorly understood. While many works focus on improving privacy frameworks, evaluating these systems in specific contexts is rare. As we saw in our systematic study, less than a quarter of the work in prompt privacy literature ties itself to a specific application (Figure 3 (Left)(b)).

![](images/2a1aa52c2cf92ec8ac7399aa6108ce0d593a31c2528ddc4afc34ba4cf5ea0c1c.jpg)  
Figure 5. Motivating use cases as defined in § 5, along with our recommendations based on the preceding discussions in the paper.

We strongly recommend that the community move toward context-specific evaluation ofprivacyframeworks, to audit their strengths, limitations, and suitability.

## 10.2. Application-Specific Deployment of LLMs

Another way of enforcing context-specific privacy protections is to separate different uses of LLMs. Despite their general-purpose capabilities, separating applications with substantially different privacy requirements can provide meaningful benefits. Consider the recent releases of Chat-GPT Health [69] and Claude for Healthcare [5]. By isolating healthcare-related use from other applications and managing their data separately, providers can apply privacy safeguards tailored to the sensitivity of medical information.

The use of LLMs in healthcare raises numerous ethical concerns, of which privacy is only one. We do not endorse the use of LLMs in healthcare settings; rather, we only use this example to illustrate how application-specific deployment can be implemented in practice, even for highly capable general-purpose models. Such siloed deployments enable more context-aware management of privacy risks.

We recommend that developers move towards applicationspecific deployment ofLLMs, wherever reasonably possible, thus avoiding the trap ofa unified privacyframework.

## 10.3. Fostering User Awareness and Privacy Literacy

In the absence of a universal solution and with the search for context-specific tools, it becomes even more important to help end users understand these trade-offs. For instance, while no single tool is ideal for Sasha’s company (cryptography being the closest, though still quite costly), educating employees about what form of confidential information should never be shared with LLMs can help reduce information leakage while still providing them with a chatbot companion. Similarly, local deployment of LLMs has gained significant popularity to avoid the concerns of privacy. Users with better privacy literacy will be able to switch between locally deployed LLMs or third-party LLMs, depending on the severity of the context.

Empowering users with stronger literacy and accurate mental models ofhow their data is handled is essential tofostering safer and trustworthy interactions.

## Acknowledgment

Funding support for project activities has been partially provided by Canada CIFAR AI Chair, Google award, MITACS, FRQNT, and NSERC Discovery Grants program. We also express our gratitude to Compute Canada for their support in providing facilities for our evaluations.

## References

[1] Prompt security: Ai security company — manage genai risks & secure llm apps. https://www. prompt.security/, 2025. Accessed: 2025-09- 14.

[2] Acar, A., Aksu, H., Uluagac, A. S., and Conti, M. A survey on homomorphic encryption schemes: The-

ory and implementation. ACM Computing Surveys (Csur), 51(4):1–35, 2018.

[3] aimeetsprivacy. Privacygpt. Chrome Web Store extension, 2025. URL https://chromewebstore. google.com/detail/privacygpt/ iobeegngilbjemenccdplkjndkkpplbe. Version 1.3.2, last updated May 29, 2025; offered by aimeetsprivacy.

[4] Albanese, F., Ciolek, D., and D’Ippolito, N. Text sanitization beyond specific domains: Zero-shot redaction & substitution with large language models. arXiv preprint arXiv:2311.10785, 2023.

[5] Anthropic. Introducing Claude for Healthcare. https://www.anthropic.com/news/ healthcare-life-sciences, January 11 2026. Accessed: 2026-01-26.

[6] Aun˜on, J., Hurtado-Ram´ ´ırez, D., Porras-D´ıaz, L., Irigoyen-Pena, B., Rahmian, S., Al-Khazraji, Y.,˜ Soler-Garrido, J., and Kotsev, A. Evaluation and utilisation of privacy enhancing technologies—a data spaces perspective. Data in Brief, 55:110560, 2024.

[7] Barkhuus, L. The mismeasurement of privacy: using contextual integrity to reconsider privacy in hci. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 367–376, 2012.

[8] Brown, H., Lee, K., Mireshghallah, F., Shokri, R., and Tramer, F. What does it mean for a language \` model to preserve privacy?, 2022.

[9] Carey, A. N., Bhaila, K., Edemacu, K., and Wu, X. Dp-tabicl: In-context learning with differentially private tabular data. In 2024 IEEE International Conference on Big Data (BigData), pp. 1552–1557. IEEE, 2024.

[10] Carlini, N., Tramer, F., Wallace, E., Jagielski, M.,\` Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., and<sup>´</sup> Raffel, C. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650. USENIX Association, August 2021. ISBN 978- 1-939133-24-3. URL https://www.usenix. org/conference/usenixsecurity21/ presentation/carlini-extracting.

[11] Carpentier, R., Zhao, B. Z. H., Asghar, H. J., and Kaafar, D. Preempting text sanitization utility in resource-constrained privacy-preserving llm interactions, 2025. URL https://arxiv.org/abs/ 2411.11521.

[12] Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., Ye, W., Zhang, Y., Chang, Y., Yu, P. S., Yang, Q., and Xie, X. A survey on evaluation of large language models. ACM Trans. Intell. Syst. Technol., 15(3), March 2024. ISSN 2157-6904. doi: 10.1145/3641289. URL https://doi.org/10.1145/3641289.

[13] Chen, H., Chen, H. H., Sun, M., Li, K., Chen, Z., and Wang, X. A verified confidential computing as a service framework for privacy preservation. In 32nd USENIX security symposium (USENIX Security 23), pp. 4733–4750, 2023.

[14] Chen, K., Zhou, X., Lin, Y., Feng, S., Shen, L., and Wu, P. A survey on privacy risks and protection in large language models. Journal of King Saud University Computer and Information Sciences, 37 (7):163, 2025.

[15] Chen, T., Bao, H., Huang, S., Dong, L., Jiao, B., Jiang, D., Zhou, H., Li, J., and Wei, F. The-x: Privacypreserving transformer inference with homomorphic encryption. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pp. 3510–3520, 2022.

[16] Chen, Y., Li, T., Liu, H., and Yu, Y. Hide and seek (has): A lightweight framework for prompt privacy protection, 2023.

[17] Cheng, Z., Wan, D., Abueg, M., Ghalebikesabi, S., Yi, R., Bagdasarian, E., Balle, B., Mellem, S., and O’Banion, S. Ci-bench: Benchmarking contextual integrity of ai assistants on synthetic data. arXiv preprint arXiv:2409.13903, 2024.

[18] Chong, C. J., Hou, C., Yao, Z., and Talebi, S. M. S. Casper: Prompt sanitization for protecting user privacy in web-based large language models, 2024. URL https://arxiv.org/abs/2408.07004.

[19] Cofone, I. N. Privacy harms. Hastings Law Journal, 69(4):1039–1070, 2018.

[20] de Castro, L., Escudero, D., Agrawal, A., Polychroniadou, A., and Veloso, M. Encryptedllm: Privacypreserving large language model inference via gpuaccelerated fully homomorphic encryption. In Fortysecond International Conference on Machine Learning, 2025.

[21] Delgado Rodriguez, S., Windl, M., Alt, F., and Marky, K. The tapsi research framework-a systematization of knowledge on tangible privacy and security interfaces. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pp. 1–28, 2025.

[22] Dibia, O., Bhattacharjee, P., Stenger, B., Baldasty, S., Bates, M., Ngong, I. C., Feng, Y., and Near, J. P. Sok: Usability studies in differential privacy. arXiv preprint arXiv:2412.16825, 2024.

[23] Du, H., Liu, S., Zheng, L., Cao, Y., Nakamura, A., and Chen, L. Privacy in fine-tuning large language models: Attacks, defenses, and future directions, 2025. URL https://arxiv.org/abs/2412. 16504.

[24] Du, M., Yue, X., Chow, S. S. M., Wang, T., Huang, C., and Sun, H. Dp-forward: Fine-tuning and inference on language models with differential privacy in forward pass. In Proceedings ofthe 2023 ACM SIGSAC Conference on Computer and Communications Security, CCS ’23, pp. 2665–2679. ACM, November 2023. doi: 10.1145/3576915.3616592. URL http://dx. doi.org/10.1145/3576915.3616592.

[25] Du, W. and Atallah, M. J. Secure multi-party computation problems and their applications: a review and open problems. In Proceedings of the 2001 workshop on New security paradigms, pp. 13–22, 2001.

[26] Duan, H., Dziedzic, A., Papernot, N., and Boenisch, F. Flocks of stochastic parrots: Differentially private prompt learning for large language models. Advances in Neural Information Processing Systems, 36:76852– 76871, 2023.

[27] Duan, H., Dziedzic, A., Yaghini, M., Papernot, N., and Boenisch, F. On the privacy risk of in-context learning. arXiv preprint arXiv:2411.10512, 2024.

[28] Dwork, C. and Roth, A. The algorithmic foundations of differential privacy. Found. Trends Theor. Comput. Sci., 9(3–4):211–407, August 2014. ISSN 1551- 305X. doi: 10.1561/0400000042. URL https: //doi.org/10.1561/0400000042.

[29] Edemacu, K. and Wu, X. Privacy preserving prompt engineering: A survey, 2024.

[30] Edemacu, K. and Wu, X. Privacy preserving prompt engineering: A survey. ACM Computing Surveys, 57 (10):1–36, 2025.

[31] Feyisetan, O., Balle, B., Drake, T., and Diethe, T. Privacy-and utility-preserving textual analysis via calibrated multivariate perturbations. In Proceedings ofthe 13th international conference on web search and data mining, pp. 178–186, 2020.

[32] Fiege, U., Fiat, A., and Shamir, A. Zero knowledge proofs of identity. In Proceedings of the nineteenth annual ACM symposium on Theory of computing, pp. 210–217, 1987.

[33] Ghalebikesabi, S., Bagdasaryan, E., Yi, R., Yona, I., Shumailov, I., Pappu, A., Shi, C., Weidinger, L., Stanforth, R., Berrada, L., et al. Operationalizing contextual integrity in privacy-conscious assistants. arXiv preprint arXiv:2408.02373, 2024.

[34] Gim, I., Li, C., and Zhong, L. Confidential prompting: Protecting user prompts from cloud llm providers. arXiv preprint arXiv:2409.19134, 2024.

[35] Guo, Y., Shang, G., and Clavel, C. Benchmarking linguistic diversity of large language models, 2024. URL https://arxiv.org/abs/2412. 10271.

[36] Hansson, D. H. The rails doctrine. https: //rubyonrails.org/doctrine, 2014. Accessed: 2026-05-15.

[37] Hao, M., Li, H., Chen, H., Xing, P., Xu, G., and Zhang, T. Iron: Private inference on transformers. Advances in neural information processing systems, 35:15718–15731, 2022.

[38] Harnad, S. The symbol grounding problem. Physica D: Nonlinear Phenomena, 42(1-3):335–346, 1990.

[39] Hartmann, F., Tran, D.-H., Kairouz, P., Carbune, V.,˘ and Arcas, B. A. Y. Can llms get help from other llms without revealing private information? In Proceedings ofthe Fifth Workshop on Privacy in Natural Language Processing, pp. 107–122, 2024.

[40] Hedegaard, S., Houen, S., and Simonsen, J. G. Lair: a language for automated semantics-aware text sanitization based on frame semantics. In 2009 IEEE International Conference on Semantic Computing, pp. 47–52. IEEE, 2009.

[41] Hong, J., Wang, J. T., Zhang, C., Li, Z., Li, B., and Wang, Z. Dp-opt: Make large language model your privacy-preserving prompt engineer, 2024. URL https://arxiv.org/abs/2312.03724.

[42] Hou, X., Liu, J., Li, J., Li, Y., Lu, W.-j., Hong, C., and Ren, K. Ciphergpt: Secure two-party gpt inference. Cryptology ePrint Archive, 2023.

[43] Iravantchi, Y., Emami-Naeini, P., and Sample, A. Sok:(un) usable privacy: the lack of overlap between privacy-aware sensing and usable privacy research. Proceedings on Privacy Enhancing Technologies, 2025.

[44] Kan, Z., Qiao, L., Yu, H., Peng, L., Gao, Y., and Li, D. Protecting user privacy in remote conversational systems: A privacy-preserving framework based on text sanitization. arXiv preprint arXiv:2306.08223, 2023.

[45] Li, H., Chen, Y., Luo, J., Wang, J., Peng, H., Kang, Y., Zhang, X., Hu, Q., Chan, C., Xu, Z., et al. Privacy in large language models: Attacks, defenses and future directions. arXiv preprint arXiv:2310.10383, 2023.

[46] Li, H., Xu, M., and Song, Y. Sentence embedding leaks more information than you expect: Generative embedding inversion attack to recover the whole sentence. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 14022–14040, 2023.

[47] Li, H., Hu, W., Jing, H., Chen, Y., Hu, Q., Han, S., Chu, T., Hu, P., and Song, Y. Privaci-bench: Evaluating privacy with contextual integrity and legal compliance. arXiv preprint arXiv:2502.17041, 2025.

[48] Li, M., Fan, H., Fu, S., Ding, J., and Feng, Y. Dp-gtr: Differentially private prompt protection via group text rewriting, 2025. URL https://arxiv. org/abs/2503.04990.

[49] Li, Y., Tan, Z., and Liu, Y. Privacy-preserving prompt tuning for large language model services, 2025. URL https://arxiv.org/abs/2305.06212.

[50] Lin, S., Hua, W., Wang, Z., Jin, M., Fan, L., and Zhang, Y. Emojiprompt: Generative prompt obfuscation for privacy-preserving communication with cloud-based llms. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 12342–12361, 2025.

[51] Lu, W.-j., Huang, Z., Gu, Z., Li, J., Liu, J., Hong, C., Ren, K., Wei, T., and Chen, W. Bumblebee: Secure two-party inference framework for large transformers. Cryptology ePrint Archive, 2023.

[52] Luo, J., Zhang, Y., Zhang, Z., Zhang, J., Mu, X., Wang, H., Yu, Y., and Xu, Z. Secformer: Fast and accurate privacy-preserving inference for transformer models via smpc. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 13333– 13348, 2024.

[53] Lyu, L., He, X., and Li, Y. Differentially private representation for nlp: Formal guarantee and an empirical study on privacy and fairness. In Findings of the Associationfor Computational Linguistics: EMNLP 2020, pp. 2355–2365, 2020.

[54] Makhdoumi, A., Salamatian, S., Fawaz, N., and Medard, M. From the information bottleneck to the´ privacy funnel. In 2014 IEEE Information Theory Workshop (ITW 2014), pp. 501–505. IEEE, 2014.

[55] Mattern, J., Weggenmann, B., and Kerschbaum, F. The limits of word level differential privacy, 2022. URL https://arxiv.org/abs/2205. 02130.

[56] Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher, R., Amatriain, X., and Gao, J. Large language models: A survey, 2025. URL https: //arxiv.org/abs/2402.06196.

[57] Mireshghallah, N., Antoniak, M., More, Y., Choi, Y., and Farnadi, G. Trust no bot: Discovering personal disclosures in human-llm conversations in the wild, 2024. URL https://arxiv.org/abs/2407. 11438.

[58] Mireshghallah, N., Kim, H., Zhou, X., Tsvetkov, Y., Sap, M., Shokri, R., and Choi, Y. Can llms keep a secret? testing privacy implications of language models via contextual integrity theory, 2024. URL https://arxiv.org/abs/2310.17884.

[59] Mittal, R. Improving inference privacy for large language models using fully homomorphic encryption. 2024.

[60] Mollo, D. C. and Milliere, R. The vector grounding\` problem. arXiv preprint arXiv:2304.01481, 2023.

[61] More, Y., Ganesh, P., and Farnadi, G. Towards more realistic extraction attacks: An adversarial perspective, 2024. URL https://arxiv.org/abs/ 2407.02596.

[62] Morris, J., Kuleshov, V., Shmatikov, V., and Rush, A. M. Text embeddings reveal (almost) as much as text. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12448–12460, 2023.

[63] Murtagh, J., Taylor, K., Kellaris, G., and Vadhan, S. Usable differential privacy: A case study with psi. arXiv preprint arXiv:1809.04103, 2018.

[64] Neel, S. and Chang, P. Privacy issues in large language models: A survey. arXiv preprint arXiv:2312.06717, 2023.

[65] Ngong, I., Kadhe, S., Wang, H., Murugesan, K., Weisz, J. D., Dhurandhar, A., and Ramamurthy, K. N. Protecting users from themselves: Safeguarding contextual privacy in interactions with conversational agents, 2025. URL https://arxiv.org/abs/ 2502.18509.

[66] Nissenbaum, H. Privacy as contextual integrity. Wash. L. Rev., 79:119, 2004.

[67] Nissenbaum, H. Privacy in Context: Technology, Policy, and the Integrity of Social Life. Stanford University Press, USA, 2009. ISBN 0804752370.

[68] Nosek, B. A., Spies, J. R., and Motyl, M. Scientific utopia: Ii. restructuring incentives and practices to promote truth over publishability. Perspectives on Psychological Science, 7(6):615–631, 2012.

[69] OpenAI. Introducing ChatGPT health. https://openai.com/index/ introducing-chatgpt-health/, January 7 2026. Accessed: 2026-01-26.

[70] Pan, X., Zhang, M., Ji, S., and Yang, M. Privacy risks of general-purpose language models. In 2020 IEEE Symposium on Security and Privacy (SP), pp. 1314–1331. IEEE, 2020.

[71] Panda, A., Wu, T., Wang, J., and Mittal, P. Differentially private in-context learning. In The 61st Annual Meeting Of The Association For Computational Linguistics, 2023.

[72] Pang, Q., Zhu, J., Mollering, H., Zheng, W., and¨ Schneider, T. Bolt: Privacy-preserving, accurate and efficient inference for transformers. In 2024 IEEE Symposium on Security and Privacy (SP), pp. 4753– 4771. IEEE, 2024.

[73] Papadopoulou, A., Yu, Y., Lison, P., and Øvrelid, L. Neural text sanitization with explicit measures of privacy risk. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 217–229, 2022.

[74] Park, S., Lenhart, A., Zimmer, M., and Vitak, J. ” nobody’s happy”: Design insights from {Privacy-Conscious} smart home power users on enhancing data transparency, visibility, and control. In Nineteenth Symposium on Usable Privacy and Security (SOUPS 2023), 2023.

[75] Protect, A. Llm guard: The security toolkit for llm interactions. Retrieved January, 16:2025, 2024.

[76] Qu, W., Zhou, Y., Wu, Y., Xiao, T., Yuan, B., Li, Y., and Zhang, J. Prompt inversion attack against collaborative inference of large language models. In 2025 IEEE Symposium on Security and Privacy (SP), pp. 1695–1712. IEEE, 2025.

[77] Rabin, M. O. How to exchange secrets with oblivious transfer. Cryptology ePrint Archive, 2005.

[78] Rathee, D., Li, D., Stoica, I., Zhang, H., and Popa, R. Mpc-minimized secure llm inference. arXiv preprint arXiv:2408.03561, 2024.

[79] Shao, Y., Li, T., Shi, W., Liu, Y., and Yang, D. Privacylens: Evaluating privacy norm awareness of language models in action, 2025. URL https: //arxiv.org/abs/2409.00138.

[80] Shvartzshnaider, Y. and Duddu, V. Position: Contextual integrity is inadequately applied to language models, 2025. URL https://arxiv.org/ abs/2501.19173.

[81] Shvartzshnaider, Y., Duddu, V., and Lacalamita, J. Llm-ci: Assessing contextual integrity norms in language models. arXiv preprint arXiv:2409.03735, 2024.

[82] Sinha, K., Gauthier, J., Mueller, A., Misra, K., Fuentes, K., Levy, R., and Williams, A. Language model acceptability judgements are not always robust to context, 2022. URL https://arxiv.org/ abs/2212.08979.

[83] Smith, V., Shamsabadi, A. S., Ashurst, C., and Weller, A. Identifying and mitigating privacy risks stemming from language models: A survey. arXiv preprint arXiv:2310.01424, 2023.

[84] Song, C. and Raghunathan, A. Information leakage in embedding models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 377–390, 2020.

[85] Staab, R., Vero, M., Balunovic, M., and Vechev, M. Beyond memorization: Violating privacy via inference with large language models. In International Conference on Learning Representations, volume 2024, pp. 33832–33878, 2024.

[86] Sun, X., Liu, G., He, Z., Li, H., and Li, X. Deprompt: Desensitization and evaluation of personal identifiable information in large language model prompts. arXiv preprint arXiv:2408.08930, 2024.

[87] Sanchez, D. and Batet, M. C-sanitized: A pri-´ vacy model for document redaction and sanitization. Journal of the Association for Information Science and Technology, 67(1):148–163, April 2015. ISSN 2330-1643. doi: 10.1002/asi.23363. URL http: //dx.doi.org/10.1002/asi.23363.

[88] Tang, X., Shin, R., Inan, H. A., Manoel, A., Mireshghallah, F., Lin, Z., Gopi, S., Kulkarni, J., and Sim, R. Privacy-preserving in-context learning with differentially private few-shot generation. In The Twelfth International Conference on Learning Representations, 2024.

[89] Tong, M., Chen, K., Yuan, X., Liu, J., Zhang, W., Yu, N., and Zhang, J. On the vulnerability of text sanitization, 2025. URL https://arxiv.org/ abs/2410.17052.

[90] Tong, M., Chen, K., Zhang, J., Qi, Y., Zhang, W., Yu, N., Zhang, T., and Zhang, Z. Inferdpt: Privacypreserving inference for black-box large language models. IEEE Transactions on Dependable and Secure Computing, 2025.

[91] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal,\` N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971.

[92] Wu, H., Dai, W., Li, W., and Yan, Q. Cape: Contextaware prompt perturbation mechanism with differential privacy. In Forty-second International Conference on Machine Learning, 2025.

[93] Wu, T., Panda, A., Wang, J. T., and Mittal, P. Privacypreserving in-context learning for large language models. In The Twelfth International Conference on Learning Representations, 2024.

[94] Xiao, Y., Jin, Y., Bai, Y., Wu, Y., Yang, X., Luo, X., Yu, W., Zhao, X., Liu, Y., Gu, Q., et al. Privacymind: large language models can be contextual privacy protection learners. arXiv preprint arXiv:2310.02469, 2023.

[95] Xie, S., Dai, W., Ghosh, E., Roy, S., Schwartz, D., and Laine, K. Does prompt-tuning language model ensure privacy? arXiv preprint arXiv:2304.03472, 2023.

[96] Xin, R., Mireshghallah, N., Li, S. S., Duan, M., Kim, H., Choi, Y., Tsvetkov, Y., Oh, S., and Koh, P. W. A false sense of privacy: Evaluating textual data sanitization beyond surface-level privacy leakage, 2025. URL https://arxiv.org/abs/2504. 21035.

[97] Xu, T., Lu, W.-j., Yu, J., Chen, Y., Lin, C., Wang, R., and Li, M. Breaking the layer barrier: Remodeling private transformer inference with hybrid {CKKS} and {MPC}. In 34th USENIX Security Symposium (USENIX Security 25), pp. 2653–2672, 2025.

[98] Yadav, V. K., Andola, N., Verma, S., and Venkatesan, S. A survey of oblivious transfer protocol. ACM Computing Surveys (CSUR), 54(10s):1–37, 2022.

[99] Yang, X., Wen, Z., Qu, W., Chen, Z., Xiang, Z., Chen, B., and Yao, H. Memorization and privacy risks in domain-specific large language models. In ICLR 2024 Workshop on Reliable and Responsible Foundation Models, 2013.

[100] Yao, Y., Wang, F., Ravi, S., and Chen, M. Privacypreserving language model inference with instance obfuscation. arXiv preprint arXiv:2402.08227, 2024.

[101] Yi, S., Liu, Y., Sun, Z., Cong, T., He, X., Song, J., Xu, K., and Li, Q. Jailbreak attacks and defenses against large language models: A survey. arXiv preprint arXiv:2407.04295, 2024.

[102] Yue, X., Du, M., Wang, T., Li, Y., Sun, H., and Chow, S. S. Differential privacy for text analytics via natural text sanitization. arXiv preprint arXiv:2106.01221, 2021.

[103] Zhang, J., Chen, K., Feng, Z., Lou, J., and Song, M. Secpe: Secure prompt ensembling for private and robust large language models. In ECAI, 2024.

[104] Zhang, K., Wang, J., Hua, E., Qi, B., Ding, N., and Zhou, B. Cogenesis: A framework collaborating large and small language models for secure contextaware instruction following. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4295–4312, 2024.

[105] Zhang, L., Li, C., Hu, Q., Lang, J., Huang, S., Hu, L., Leng, J., Chen, Q., and Lv, C. Enhancing privacy in large language model with homomorphic encryption and sparse attention. Applied Sciences, 13(24):13146, 2023.

[106] Zhang, M., He, T., Wang, T., Mi, L., Mireshghallah, F., Chen, B., Wang, H., and Tsvetkov, Y. Latticegen: A cooperative framework which hides generated text in a lattice for privacy-aware generation on cloud. arXiv preprint arXiv:2309.17157, 2023.

[107] Zhang, S., Yi, X., Xing, H., Ye, L., Hu, Y., and Li, H. Adanonymizer: Interactively navigating and balancing the duality of privacy and output performance in human-llm interaction, 2025. URL https://arxiv.org/abs/2410.15044.

[108] Zhang, Z., Jia, M., Lee, H.-P., Yao, B., Das, S., Lerner, A., Wang, D., and Li, T. “it’s a fair game”, or is it? examining how users navigate disclosure risks and benefits when using llm-based conversational agents. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pp. 1–26, 2024.

[109] Zhao, C., Zhao, S., Zhao, M., Chen, Z., Gao, C.-Z., Li, H., and Tan, Y.-a. Secure multi-party computation: theory, practice and applications. Information Sciences, 476:357–372, 2019.

[110] Zhao, K., Li, L., Ding, K., Gong, N. Z., Zhao, Y., and Dong, Y. A survey on model extraction attacks and defenses for large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 6227–6236, 2025.

[111] Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., Du, Y., Yang, C., Chen, Y., Chen, Z., Jiang, J., Ren, R., Li, Y., Tang, X., Liu, Z., Liu, P., Nie, J.-Y., and Wen, J.-R. A survey of large language models, 2025. URL https://arxiv.org/abs/2303.18223.

[112] Zhou, J., Xu, E., Wu, Y., and Li, T. Rescriber: Smaller-llm-powered user-led data minimization for llm-based chatbots. 2024. URL https: //api.semanticscholar.org/CorpusID: 276287991.

[113] Zhou, X., Lu, Y., Ma, R., Gui, T., Wang, Y., Ding, Y., Zhang, Y., Zhang, Q., and Huang, X. TextObfuscator: Making pre-trained language model a privacy protector via obfuscating word representations. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Findings ofthe Associationfor Computational Linguistics: ACL 2023, pp. 5459–5473, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl. 337. URL https://aclanthology.org/ 2023.findings-acl.337/.

[114] Zhou, Y., Ni, T., Lee, W.-B., and Zhao, Q. A survey on backdoor threats in large language models (llms): Attacks, defenses, and evaluation methods. Transactions on Artificial Intelligence, 1(1):28–58, 2025.

[115] Zimerman, I., Adir, A., Aharoni, E., Avitan, M., Baruch, M., Drucker, N., Masalha, R., Meiri, R., and Soceanu, O. Powersoftmax: Towards secure llm inference over fhe. In Annual FHE. org Conference on Fully Homomorphic Encryption., 2025.

## A. Systematic Study of Drift Towards Unified Framework for Prompt Privacy

## A.1. Academic Literature

We start our systematic study of academic literature by collecting papers from the following four surveys: Chen et al. [14], Edemacu & Wu [30], Li et al. [45], Smith et al. [83]. For Chen et al. [14], we collect all papers from Tables 1-4, i.e., all papers covered in the survey. For Edemacu & Wu [30], we collect all papers from Table 1, i.e., also all papers covered in the survey. For Li et al. [45], we collect all papers under the attack stage Model Inference and Others (thus excluding attack stage Model Training) from Table 2. Finally, for Smith et al. [83], we collect all papers marked for phase In-Context Learning in Table 1, excluding all other phases (pretraining, continued pretraining, and fine-tuning). After removing duplicates, this created a list of 228 papers in total.

Each paper was then manually checked to see if it fit within the scope of our work. In other words, only papers where the privacy target was the input query sent to an LLM during inference, or some part of it, were kept, while the rest were filtered out. This left us with a final list of 31 papers in total. Each paper was then annotated, answering the following four questions: (a) What is the privacy target for protection, or what kind of adversary is the method protecting against? (b) Is there a specific use case that is being targeted? (c) Is there an explicit discussion of limitations, or contexts in which this framework will not work? and (d) What is the broad category of technique used in the paper (N/A if the paper doesn’t propose a defense strategy, but only studies information leakage)?

In our labeling, we considered the central framing, methodological commitments, and conceptual narrative of each work, not just isolated experimental details or the presence of certain sections. For instance, if a paper claims broad protection against leakage, i.e., does not commit to an adversary, but still defines specific adversaries only for empirical evaluation, we do not count this as protecting against a specific adversary, since this choice does not inform the central narrative of the work. Similarly, the presence of a limitations section alone is not enough, and we specifically look for discussions of when the proposed privacy framework might fail. The final set of annotations for all 31 papers is collected in Table 1.

## A.2. Prompt Privacy Tools

For prompt privacy tools, we started our search from github.com/pluja/awesome-privacy, and then expanded the search on the web using the terms ‘prompt privacy’ and ‘LLM inference privacy’. More specifically, we searched on google.com, producthunt.com, and chromewebstore.google.com. Each tool was then annotated, answering the following three questions: (a) Is there a specific use case that is being targeted? (b) Is the tool free or paid? and (c) What is the broad category of technique used by the tool? The final set of annotations for all 11 tools is collected in Table 2.

<table><tr><td>Title</td><td>Privacy Target/Adver- sary Goals</td><td>Use Case</td><td>Explicit Mention of Lim- itations</td><td>Technique Category</td></tr><tr><td>CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-</td><td>Additional Context</td><td>Context-Aware Instruc- tions</td><td>No</td><td>Contextual Integrity</td></tr><tr><td>Aware Instruction Following [104] Can LLMs Keep a Secret? Testing Privacy Im- plications of Language Models via Contextual</td><td>Input Prompt</td><td>No</td><td>Yes</td><td>Contextual Integrity</td></tr><tr><td>Integrity Theory [58] CipherGPT: Secure Two-Party GPT Infer- ence [42]</td><td>Input Prompt</td><td>No</td><td>No</td><td>Cryptography</td></tr><tr><td>Iron: Private Inference on Transformers [37]</td><td>Input Prompt</td><td>No</td><td>No</td><td>Cryptography</td></tr><tr><td>THE-X: Privacy-Preserving Transformer Infer- ence with Homomorphic Encryption [15]</td><td>Input Prompt</td><td>No</td><td>No</td><td>Cryptography</td></tr><tr><td>A Verified Confidential Computing as a Service Framework for Privacy Preservation [13]</td><td>Input Prompt</td><td>No</td><td>Yes</td><td>Cryptography</td></tr><tr><td>SecFormer: Fast and Accurate Privacy-Preserving Inference for Transformer Models via SMPC [52</td><td>Input Prompt</td><td>No</td><td>Yes</td><td>Cryptography</td></tr><tr><td>Differentially Private Representation for NLP: Formal Guarantee and An Empirical Study on Privacy and Fairness [53]</td><td>Embedding Representa- tions</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>DP-Forward: Fine-tuning and Inference on Lan- guage Models with Differential Privacy in For- ward Pass [24]</td><td>Embedding Representa- tions</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Flocks of Stochastic Parrots: Differentially Pri- vate Prompt Learning for Large Language Mod- els [26]</td><td>In-Context Demonstra- tions</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Privacy-Preserving In-Context Learning for Large Language Models [93]</td><td>In-Context Demonstra- tions</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Privacy-Preserving In-Context Learning with Dif- ferentially Private Few-Shot Generation [88]</td><td>In-Context Demonstra- tions</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Differential Privacy for Text Analytics via Natural Text Sanitization [102]</td><td>Input Prompt</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations [31]</td><td>Input Prompt</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>DP-OPT: Make Large Language Model Your</td><td>Prompt Tuning Data</td><td>No</td><td>No</td><td>Differential Privacy</td></tr><tr><td>Privacy-Preserving Prompt Engineer [41] DP-TabICL: In-Context Learning with Differen-</td><td>In-Context Demonstra-</td><td>Tabular Classification</td><td>No</td><td>Differential Privacy</td></tr><tr><td>tially Private Tabular Data [9] Text Embeddings Reveal (Almost) As Much As</td><td>tions Embedding Representa-</td><td>Clinical Notes</td><td>Yes</td><td>Differential Privacy</td></tr><tr><td>Text [62] InferDPT: Privacy-preserving Inference for</td><td>tions Input Prompt</td><td>No</td><td>Yes</td><td>Differential Privacy</td></tr><tr><td>Closed-box Large Language Models [90] Differentially Private In-Context Learning [71] Sentence Embedding Leaks More Information</td><td>Input Prompt</td><td>Summarization</td><td>Yes</td><td>Differential Privacy</td></tr><tr><td>than You Expect: Generative Embedding Inver- sion Attack to Recover the Whole Sentence [46]</td><td>Embedding Representa- tions</td><td>No</td><td>No</td><td>N/A</td></tr><tr><td>Does Prompt-Tuning Language Model Ensure Pri- vacy? [95]</td><td>Prompt Tuning Data</td><td>Email Autoreply</td><td>Yes</td><td>N/A</td></tr><tr><td>Privacy Risks of General-Purpose Language Mod- els [70]</td><td>Embedding Representa-</td><td>No</td><td>Yes</td><td>N/A</td></tr><tr><td>On the Privacy Risk of In-Context Learning [27]</td><td>tions In-Context Demonstra-</td><td>No</td><td>Yes</td><td>N/A</td></tr><tr><td>Information Leakage in Embedding Models [84]</td><td>tions Embedding Representa-</td><td>No</td><td>No</td><td>Sanitization/Obfuscation</td></tr><tr><td>TextObfuscator: Making Pre-trained Language Model a Privacy Protector via Obfuscating Word</td><td>tions Embedding Representa- tions</td><td>No</td><td>No</td><td>Sanitization/Obfuscation</td></tr><tr><td>Representations [113] Protecting User Privacy in Remote Conversational Systems: A Privacy-Preserving Framework Based</td><td>Personally Identifiable In-</td><td>No</td><td>No</td><td>Sanitization/Obfuscation</td></tr><tr><td>on Text Sanitization [44] Beyond Memorization: Violating Privacy via In-</td><td>formation (PIIs) User Attributes</td><td>No</td><td>No</td><td>Sanitization/Obfuscation</td></tr><tr><td>ference with Large Language Models [85] LatticeGen: A Cooperative Framework Which Hides Generated Text in a Lattice for Privacy-</td><td>Output Generation</td><td>No</td><td>Yes</td><td>Sanitization/Obfuscation</td></tr><tr><td>Aware Generation on Cloud [106] Privacy-Preserving Language Model Inference</td><td>Output Generation</td><td></td><td></td><td></td></tr><tr><td>with Instance Obfuscation [100] Hide and Seek (HaS): A Lightweight Framework</td><td>Personally Identifiable In-</td><td>No No</td><td>Yes Yes</td><td>Sanitization/Obfuscation Sanitization/Obfuscation</td></tr><tr><td>for Prompt Privacy Protection [16]</td><td>formation (PIIs) Input Prompt</td><td>Recommendation</td><td></td><td></td></tr><tr><td>EmojiPrompt: Generative Prompt Obfuscation for Privacy-Preserving Communication with Cloud- based LLMs [50]</td><td></td><td></td><td>Yes</td><td>Sanitization/Obfuscation</td></tr></table>

Table 1. Annotations for all 31 papers.

<table><tr><td>Title</td><td>Use Case</td><td>Price</td><td>Technique Category</td></tr><tr><td>Jan</td><td>General Purpose</td><td>Free</td><td>Local Deployment</td></tr><tr><td>llama.cpp</td><td>General Purpose</td><td>Free</td><td>Local Deployment</td></tr><tr><td>LocalAI</td><td>General Purpose</td><td>Free</td><td>Local Deployment</td></tr><tr><td>Ollama</td><td>General Purpose</td><td>Free</td><td>Local Deployment</td></tr><tr><td>PasteGuard</td><td>General Purpose</td><td>Free</td><td>Sanitization/Obfuscation</td></tr><tr><td>Shimmy</td><td>General Purpose</td><td>Free</td><td>Local Deployment</td></tr><tr><td>Tinfoil</td><td>General Purpose</td><td>Paid</td><td>Cryptography</td></tr><tr><td>Duck.ai</td><td>General Purpose</td><td>Freemium</td><td>Sanitization/Obfuscation</td></tr><tr><td>Phala</td><td>General Purpose</td><td>Paid</td><td>Cryptography</td></tr><tr><td>LLMGuard</td><td>General Purpose</td><td>Free</td><td>Sanitization/Obfuscation</td></tr><tr><td>PrivacyGPT</td><td>General Purpose</td><td>Free</td><td>Sanitization/Obfuscation</td></tr></table>

Table 2. Annotations for all 11 tools.