# Reasoning-supported Robustness Validation of Automotive E/E Components

Jan Novacek<sup>∗†</sup>, Alexander Viehl<sup>†</sup>, Oliver Bringmann<sup>∗†</sup>, Wolfgang Rosenstiel<sup>∗†</sup>

<sup>∗</sup>FZI Forschungszentrum Informatik

Haid-und-Neu-Str. 10-14, 76131 Karlsruhe

<sup>†</sup>Eberhard Karls Universitat T¨ ubingen¨

Sand 14, 72076 Tubingen¨

Abstract—This paper presents an ontology-supported approach to tackle the complexity of the Robustness Validation (RV) process of automotive electrical/electronic (E/E) components. The approach uses formalized knowledge from the RV process and stress, operating, and load profiles, so-called Mission Profiles (MPs). In contrast to the error-prone industrially established manual procedure, we show how component characteristics are formalized in OWL in order to form the foundation of an efficient automated analysis selection and decision support during the RV process. The proposed approach is based on the idea of mapping MPs to an OWL representation so to allow to perform semantic queries against MP data to improve their integration into the RV process. The resulting ontology-supported application framework has been applied to an industrial use-case from automotive power electronics. We present experimental results showing that the RV process can be significantly improved in terms of reduced design time and increased exhaustiveness by automating the analyses selection step and the provisioning of all the relevant data to be used. Index Terms—Systems Engineering, Robustness Validation, Knowledge Based Engineering, Mission Profiles

## I. INTRODUCTION

Highly automated driving functions are revolutionizing the automotive domain by promising unique functional and efficiency selling points as well as to improve safety, as 97 % of all accidents are due to human errors [1]. Improved safety is - besides comfort and efficiency - the reason for Advanced Driver Assistance Systems (ADAS) being a well accepted option in premium vehicles [2] but is also becoming common for other market segments. ADAS also constitute the foundation for highly-automated driving functions. Thus, automotive Original Equipment Manufacturers (OEMs) face increasing time and cost pressure while at the same time complexity is tremendously increasing. Moreover, environmental conditions for the electrical/electronic (E/E) components are harsh. Consequently, quality, functionality and reliability of automobiles have become the decisive factors of global competitiveness in the automotive industry today [3].

Robustness Validation (RV) is a process that demonstrates that a product performs its intended functions with sufficient margin under certain specified environmental conditions [3]. RV involves the creation of a robustness validation plan. Within this step, a developer needs to identify all necessary analyses for a component which are required to prove robustness. RV is a means to improve reliability which is why it is gaining increasing importance.

The environment of an E/E component is fundamental for its lifetime and operation. At temperatures above 100 <sup>◦</sup>C, too high current density can cause electromigration for example [4]. Environmental conditions can be captured in so-called Mission Profiles (MPs). They play an important role during the design and development of automotive systems, especially with regard to the propagation of requirements along the process chain of development [5, 6]. MPs are simplified representations of relevant conditions to which a component will be exposed to throughout its whole life cycle [3]. They contain various relevant stress factors and have proven their applicability in cross-domain constraint transformation [7]. MPs provide data required for certain analyses in the RV process.

The creation of the robustness validation plan is usually high error-prone due to the fact that requirements for analyses may be very complex or even change over time. Hence, necessary analyses might be easily missed out or an analysis could be conducted which is not required. RV and MP consideration is still mainly a manual task, today [8]. Another problem is the fact that it is hard to consolidate models, data and analyses methodologies of nonfunctional properties. Prior to de-facto standardization of the RV process, there did not exist any specification of conditions for RV analyses. That is why automating this process was not reasonable. While there exists a format for MPs, it currently provides only little semantic information. Its purpose is primarily to serve as a container for data. Conducting an analysis therefore requires searching the data manually for relevant portions.

Creating effective Knowledge Management Systems is a key success factor in engineering process improvement [9]. This work is concerned with the automation of a step in the RV processes to decrease required manual effort by automatically selecting analyses and providing the corresponding MP data. Part of this automation is the consideration of the component characteristics propagation across component mounting points. While we describe the approach to automating the RV process, main contributions of this work are:

• Approach to decreasing complexity of the RV process.

• Mapping MPs to OWL representations.

• Formalization of domain knowledge in OWL.

• Propagation of component characteristics.

This paper is structured as follows: The RV process and the characteristics of the industrially established RV process flow w.r.t. this work is further described in section II together with more information on MPs. Section III describes related work in the fields of Systems Engineering and Mission Profile Aware Design. The main approach is then described in section IV. Experimental results are presented in section V and section VI describes the generalization of the presented approach. Finally, section VII includes next steps, limitations and conclusions.

## II. BACKGROUND

In this section, a short introduction to Robustness Validation and Mission Profiles is given with regard to this work.

## A. Robustness Validation

When doing Robustness Validation (RV) developers need to conduct various analyses with their Device Under Test (DUT), for example a Physical Stress Analysis. Right now, these analyses are done manually for the most part and the available MPs also need to be searched manually for the required data. Moreover, a developer needs to select required analyses manually.

![](images/dd26783a4289b6df391c130110739e85993f85c8d5685a01bc871e4fb52fd7a0.jpg)  
Fig. 1: Robustness Validation process flow, adapted from [3]

The RV process flow as defined in the Handbook for Robustness Validation of Automotive Electrical/Electronic Modules [3] consists of various steps, from the determination/definition of the application(s) to production monitoring, see Fig. 1.

A step in the RV process flow is to identify analyses which need to be done for a certain component or system, namely the creation of the robustness validation plan (step 6 on the system level [10]). In this step a developer needs to identify all required analyses which are needed to approve robustness according to the Handbook for Robustness Validation. As stated before, this step involves manual action of the developer. To measure the complexity of the RV process a simple metric such as the NOA (number of activities, see for example [11]) in this process could be used.

## B. Mission Profiles

Mission Profiles (MPs) ”define the application specific context for a certain component” [5]. They are simplified representations of relevant conditions to which a component will be exposed to throughout its whole life cycle [3]. As mentioned in the introduction, MPs contain various relevant stress factors - they are composed of different sources, see Fig. 2.

![](images/fef2910270ed58396f8b0421ff1fe3d7d34235d4fdaa3a5d59ea819a01b08c3c.jpg)  
Fig. 2: Mission Profile composition

There exists a format for MPs, which has been developed in the ResCar 2.0 project<sup>1</sup>. It consists of three layers (core, template and extensions) which is further described in [5]. Although the format is not yet an official standard, the autoSWIFT project<sup>2</sup> aims to further improve the format and standardize it.

Depending on its type, a MP may be very large and the information stored in it can be diverse, containing temperature data, vibration data and so on. In case an analysis in the RV process needs to be done, the analyst needs to search for the required data. That is also why we think using semantic technologies would be beneficial, as they would enable a better integration of this data in the RV process.

## III. RELATED WORK

## A. Systems Engineering

Van Ruijven presented an ontology for Systems Engineering in [12] which used the ISO 15926 standard to model processes defined in the ISO 15288 standard. While the approach offers a general ontology for Systems Engineering, it does not cover domain specific aspects used in RV such as Mission Profile Aware Design. However, integrating this ontology into the general process flow of the presented approach may be an opportunity to explicitly describe the process using defined semantics. In the Space Systems domain, Henning et al. use OWL ontologies to establish a system model for the integration of different modeling paradigms and model semantics [13]. The approach proposes to describe the system model and the conceptual data model using an ontology description language. Its focus is on the integration of various disciplines and their corresponding data models and does not consider MPs. Zander et al. use DL ontologies to express features of components in the robotic domain, propagate features along compound components and to aggregate features that allow for the computation of complex features [14]. Using OWL property chains for mounting point characteristics propagation as described in section IV-D is inspired by this work. Also in Systems Engineering, Zhang and Luo proposed an ontology-based approach for material selection in engineering [15]. This approach uses a knowledge model consisting of an engineering ontology, tagged material instances and a SWRL [16] rule base to encode knowledge rules for material selections. Mission Profiles are also not considered by this approach due to the different domain and application. Using a SWRL rule base in addition to OWL to encode knowledge might be an opportunity to improve the presented approach in this paper. The application of ontologies to optimize the engineering process in collaborative development has been considered by Tudorache in [17]. While major contributions of this work are a framework for automatic consistency checks and the automatic propagation of changes among design models, it is not concerned with RV and therefore lacks decision support in analyses selection.

![](images/be85d6ea956b877bef0cbc1075722107242b8e8c9279500241635d498ebe6df0.jpg)  
Fig. 3: Overview of the approach

## B. Mission Profile Aware Design

The general concept of Mission Profile Aware Design (MPAD) was introduced by Jerke and Kahng in [8] along with key differences and enhancements to existing design approaches. Nirmaier et al. presented a methodology to extract finite state machines out of measured vehicle data and integrate them in MPs [5]. This work also outlines how system robustness values can be synthesized to form component robustness values. Katzschke et al. presented methods to propagate the constraints between design domains and introduced a cross-domain methodology for constraint transformation in [7]. The Reliability Knowledge Framework by NXP [18] is a web based tool consisting of a Reliability Knowledge Matrix, tools and methods for Risk Assessment (RA) and Built-in Reliability (BiR) and rules and tools for applying Structural Similarity. Moreover, it contains a MP Library. It was created to connect users, reliability knowledge, data, tools and methods. Nevertheless it differs from the presented approach, as it is not ontology-based and therefore lacks the advantages of using reasoning to support the RV process. In addition to that, this approach does not provide the means to tightly integrate MP data into the RV process.

## IV. APPROACH

The basic idea of the approach is that developers are supported during the creation of a robustness validation plan, namely the Analysis, Development & and Test phase (step 5 in the RV process flow [3]), see section II-A. We utilize reasoning to automate the identification of required analyses and corresponding data in this phase. Properties of components and systems are formalized in OWL [19] and MP data is mapped to an OWL representation to allow for semantic querying for example via the SPARQL Protocol And RDF Query Language [20]. Furthermore, certain characteristics of components are propagated amongst neighbor components as described in the following sections.

To select an appropriate analysis, the coverage of an analysis is compared to the attributes of a component which are themselves partially propagated from mounting point characteristics. The coverage and resulting analysis to be conducted is then the foundation for the selection of the required MP data, see Fig. 3 for an overview of the approach.

## A. Mission Profile to OWL mapping

To work with semantically enriched representations of MPs, a mapping mechanism is required. Apart from semantic enrichment, ontological representations of MPs are required for semantic querying. It should for example be possible to ask for specific MP data such as temperature related data. The advantage of using an ontology versus a classical approach lies in the increased maintainability, the separation of knowledge from programming and that the knowledge becomes shareable [21].

A prototype of this mapping mechanism was implemented using indirect semantic programming (see [22]) using the OWL API<sup>3</sup> [23] and the Scala [24] language. The implementation utilizes the Eclipse Modeling Framework (EMF) [25] models of the MP format which were derived from the MP format XML Schema Definition (XSD) [26] schemata. For each element in the MP a corresponding OWL individual is created and properties are added accordingly. Vectors were encoded following [27].

1) Ontology structure: The MP ontology structure is based on the current MP format draft. A MP consists of three major objects, see Fig. 4: (1) The DocumentHeader (2) a Component and (3) an OperatingStateSet. While the DocumentHeader is used to capture meta information such as the author of the MP document or the history, the Component object is used to specify the loads for the component of a MP document. Apart from the load specifications themselves, a Component contains a PortSet which is used to specify loads of the functional ports of a component. These ports may be electrical, mechanical, thermal or chemical connections. Each model instance document (ABox) is connected to the MP ontology (TBox) by importing the MP ontology.

2) Specification of functional loads: PropertySet and Property objects constitute the main building block for specifying functional loads. They contain Property objects which have an ID, a name and a description as meta information. Each property in the MP format can use a template for specifying the loads. An example would be a TemperatureTimeProfileTemplate which is used to specify temperature profiles. Such a template is composed of a matrix which holds percent vectors to specify distributions of the temperature ranges under certain conditions and a vector to hold the actual values of the temperature ranges. These templates are mapped to their corresponding OWL representation by the mapping engine. Using templates makes the format and the MP ontology extensible by allowing to add more templates later on.

Fig. 5 shows some of the relationships used for property specifi cation in the MP ontology. The shown graph is used for the specification of a functional load namely the driving profile of a business vehicle. A PropertySet represents a set of Property objects which themselves have a PropertyValue object for specifying a SimpleValue and a PropertyValueTemplate object for specifying more complex functional loads. Each specialization of a PropertyValueTemplate is therefore a subclass of the PropertyValueTemplate concept. Of course a PropertyValueTemplate instance can be a complex structure, depending on the type of functional load that is to be specified.

![](images/c00a5597ca1669fcf71a86c46aeceea3b2cf8ec8e9879fe2692a9be4898a6cf5.jpg)  
Fig. 4: Excerpt of the Mission Profile ontology structure as UML class diagram

![](images/5cb967ff80c69663251ec35486796ffce8f5ebc4def71384626c7c9b9748426a.jpg)  
Fig. 5: Model graph excerpt for property specification in VOWL

## B. Metadata extraction

An important step in creating ontological representations of MPs is metadata extraction (ME). ME is needed for example for the following purposes:

• Determine the type of data, recognize content

• Improve search functionality

• For provenance information

• For visualizing data

• Filtering data based on its content

• For change management

An example of ME is to recognize a component’s properties and associated data from MP content for classification: Suppose a MP contains a TemperatureTimeProfileTemplate for a component. Such a template contains a matrix composed of a TemperatureRangeVector to hold actual temperature ranges and a PercentVector to specify the temperature distribution, Fig. 5 shows some of the TBox information of such a template. With only the corresponding ABox facts at hand a machine does not know about the conditions in which the values were measured or even the meaning of those values at all. ME needs to be used to extract information such as the kind of vehicle that was used and other conditions. That way, a machine could decide whether a MP is within the scope of required data of a certain analysis or not. In our example, if there is a difference between a vehicle which is used for long-distance business travels and a vehicle used for family transportation ME could be used to classify the MP.

ME is required because one needs to identify the correct data for an analysis. Currently, the MP format solely provides a structure for the data but provides only little semantic information. Without this information functional load specifications for example could have incorrect units or its values might be in an invalid range. Likewise, the format in its actual state does not support to transport statements about the type of data apart from XSD templates used to specify the data itself.

Also not part of the format are statements about the context of a MP, for example the environment of a component e.g. neighbor components or the weather conditions when the data was collected. This kind of information needs to be extracted from other sources than the MP document itself. In this work, a current draft of the MP format is used which also has some limitations. These are likely to change in the course of the autoSWIFT project.

## C. Analysis selection based on component properties

The Handbook for Robustness Validation recommends a specific coverage for each type of analysis. For example a Physical Stress Analysis should be conducted for components with more than 64 pins or a length of more than 1” (2.54 cm) per side [3]. Such characteristics are the basis for selecting appropriate analyses in our proposed system.

1) Modeling of components: Foundation for identifying relevant analyses for a certain component is a component model which is provided by the developer. Such a model should define characteristics such as the pin count, self-heating capabilities and so on. See List. 1 in section V for an example of such a component model. Ideally, the formalization of such a component model is done automatically by harvesting existing component specifications in potentially other formats than OWL. Each component should moreover be integrated into a system model which incorporates the overall structure of the system. Very important here are the mounting points. Mounting points are positions in a car where components can be installed. They play an important role in the propagation of mounting point characteristics which is described in section IV-D.

![](images/932e6f794d8071c690e58c29af57923b2b04cdbd51f3b4190044431629cf1917.jpg)  
Fig. 6: Example for the propagation of mounting point characteristics

2) Modeling of background knowledge: Another important step in our approach is the formalization of background knowledge. Background knowledge is constituted of all facts that make up the guidelines of the Handbook for Robustness Validation. For example, to express that subjects which have a pin count of more than 64 pins have a high pin count, axiom 1 could be used:

$$
\exists \boldsymbol { \mathrm { h a s P i n C o u n t V a l u e . } } ( > , 6 4 ) \subseteq \exists \boldsymbol { \mathrm { h a s P i n C o u n t . H i g h P i n C o u n t ~ ( 1 ) } }
$$

In case finer grained models are applied, for example when each pin is modeled as an instance of class Pin, Cardinality constraints [28] as in axiom 2 could be used:

$$
> 6 4 . \mathsf { h a s P i n . P i n \sqsubseteq \exists h a s P i n C o u n t . H i g h P i n C o u n t }\tag{2}
$$

Other background knowledge needs to be formalized likewise.

3) Modeling of analyses coverages: Based on the formalization of the background knowledge one can specify that components which have a high pin count have a coverage of a Physical Stress Analyis (PSACoverage), as expressed by axiom 3:

$$
\begin{array} { r } { \mathsf { C o m p o n e n t } \sqcap \exists \mathsf { h a s P i n C o u n t . H i g h P i n C o u n t } \sqsubseteq } \\ { \exists \mathsf { h a s C o v e r a g e . P S A C o v e r a g e } } \end{array}\tag{3}
$$

Other coverages for other analyses need to be formalized likewise as well. According to the principle of separation of concerns, the background knowledge modeling is separated from the coverage modeling. That way, a coverage can afterwards be easily adjusted to contain specific characteristics as specified by the background knowledge modeling.

## D. Propagation of mounting point characteristics

Some characteristics of components are induced by other components which are locally close to the component in question. Examples for these characteristics are electromagnetic interference (EMI), temperature or vibration.

To model such a propagation of characteristics among neighbor components (components which are locally close to each other) complex role inclusion axioms (see [29]) are used, as in axiom 4:

$$
\mathsf { m o u n t e d O n o c l o s e T o o m o u n t e d O n ^ { - } o h a s E M l \subseteq h a s E M l }\tag{4}
$$

By defining such an axiom, a reasoner is able to infer proper EMI values for neighbor components. An example for this might be having an electronic engine in a car which would be mounted on the engine mounting point. A component close to this mounting point will be exposed to high EMI. Fig. 6 depicts an example where components C1, C2 are mounted on the mounting points M1 respectively M2. Component C2 which may for example be an electronic engine has a high EMI value. By using axiom 4 a reasoner can infer that C1 also has a high EMI value (dark grey arrow).

## E. Selecting appropriate data to support analyses

Based on the metadata extraction described in section IV-B and the analysis selection described in section IV-C, appropriate data can be directly passed to the analyst. This data is in form of so-called Mission Profiles which contain all functional and environmental loads of a component, as described in section II-B.

We have implemented a system prototype which automatically maps MP documents to an OWL representation to enable semantic querying of these documents, see section IV-A. Thereby MPs can be easily queried for specific data portions which are relevant for a certain analysis. An example might be to provide temperature and vibration data in case a Physical Stress Analysis is conducted. The advantage of using semantic queries lies in the possibility to query for very specific data portions which are for example dependent on the environment such as the system model. Moreover, they allow to query for implicitly stated information. An example might be a query for temperature data of a certain component which is mounted on a specific location and only if the system contains a combustion engine. Following the principle of Linked Data each entity is connected by RDF links, creating a global data graph which enables the discovery of new data sources [30].

## V. EXAMPLE APPLICATION

The Fairchild Semiconductor<sup>®</sup> FTCO3V455A1 is a 3-Phase inverter automotive power module<sup>4</sup>. It can be used for electric power steering, electro-hydraulic power steering, in electric water pumps and electric oil pumps. To prove robustness according to the Handbook for Robustness Validation of Automotive E/E Modules, a RV plan needs to be created. Part of this plan is a set of analyses which should be conducted. To build up an ontological representation of the module, existing module models are transformed into OWL representations by a software tool. Besides harvesting the models, contextual information is ingested by the system such as the mounting point of the module and that the module is part of a mechanical system. List. 1 shows a serialization of the module OWL model in Manchester syntax.

I n d i v i d u a l : FTCO3V455A1   
Types :   
Component   
F a c t s :   
h a s P u r p o s e S t r e e r i n g ,   
mountedOn EngineCompartmentMountingPoint ,   
p a r t O f MechanicalComponent1 ,   
h a s H e i g h t 2 9 . 0 ,   
hasPinCount 19 ,   
h a s P o w e r D i s s i p a t i o n 1 1 5 . 0 ,   
hasWidth 4 4 . 0  
List. 1: Individual in Manchester syntax

Available MP data is mapped to semantically enriched representations in parallel. Metadata extraction is used on the MP data to support classification for proper data selection. Afterwards, reasoning is used to select proper analyses and corresponding data. List. 2 shows an excerpt of the inferred axioms for the FTCO3V455A1. The EMI value is induced by another component (ElectronicEngine) which is locally close to the FTCO3V455A1 by using the property chain described in section IV-D.

![](images/99d81321c47733d16e268e5b159e0a944fe90471de47a0fda3e56d48fda29266.jpg)  
Fig. 7: Analyses selection for the FTCO3V455A1

![](images/1a90fda56a6e962212a1de7a3191a365645bdbdf38efca429ff05c933837d491.jpg)  
List. 2: Inferred axioms in Manchester syntax

Fig. 7 depicts the selection of the analyses according to the approach. The RV process is being accelerated and errors due to manual analyses selection are avoided. Suppose that suddenly in the development phase the specification of the inverter module slightly changes. The presented approach could then be used to easily update the RV plan accordingly.

On a higher level, the benefit is that for a module composed of several sub modules a property chain as defined by axiom 5 can be used to aggregate the analyses which need to be conducted for the whole module.

$$
\mathsf { p a r t O f ^ { - } o n e e d s A n a l y s i s \subseteq n e e d s A n a l y s i s }\tag{5}
$$

Likewise Robustness Indicator Figures (RIFs, see [3]) could be aggregated to provide an overview of the overall module robustness.

## VI. GENERALIZATION

It is possible to generalize the approach and apply it to other analysis methodologies, wherever analysis and corresponding data selection is done, see Fig. 8. The foundations are a component to be examined, corresponding MP data and standard literature which contains the guidelines and procedures for the analysis process.

![](images/ec8f4cf4cf5021814530bb2ea913bc8cf53dd725329d1ff3e47311e6715dde3a.jpg)  
Fig. 8: General process flow

The core of the approach is made up by mapping engines which map component models, MPs and standard literature to their corresponding OWL representations. These representations are afterwards used as input to the analyses selection process.

## VII. CONCLUSIONS AND FUTURE WORK

We presented an approach to tackle complexity in the RV process flow by utilizing reasoning. Nevertheless, there are some limitations. The Open World Assumption (OWA) imposes a limitation upon the approach: It is not possible to express that a certain component does not have a certain property. As [14] pointed out, it is possible to use closure axioms to handle this but not without introducing further problems. The granularity of the property propagation may be to coarse. Currently there is no way to define a propagation scheme which transfers only a portion of the value or transforms it in an other way. SWRL could be used to implement such value transformation and to make the knowledge about the transformations explicit. Another point is that the amount of analyses needs to be large enough for the system to be useful. If there is only a small amount of analyses, the benefit will also be small because a developer could easily create the RV plan manually. Regarding incomplete & incorrect knowledge, Entity Matching (EM) could be used.

To make such a system described in this paper ready for deployment, we have identified three major necessary steps:

Automatic formalization of component properties To ease the transition from the manual approach in the RV process to a more automated one, it would be beneficial to implement a system which automatically extracts component descriptions.

Broadening the set of analyses The higher the complexity of the selection process in the RV elaboration of the robustness validation plan is, the more useful will the system be. Therefore the amount of analyses which are regarded by the system should be high. It might also be an opportunity to generalize the approach to common analyses selection, see section VI.

Implementing metadata extraction To improve the selection of appropriate data, ME for various kinds of metadata of MPs needs to be implemented.

By introducing an approach to automation of the RV process we were able to decrease required manual effort. The approach also takes the propagation of mounting point characteristics into account. At the core of the presented approach we created a mapping engine for MPs to OWL representations to allow semantic querying and improve the integration of this data into the RV process. Tooling was created and will be applied within the context of a project. The resulting application framework could be used for other MPAD tasks as well.

## VIII. ACKNOWLEDGEMENT

This paper is partially supported by the BMBF project autoSWIFT (grant number 16ES0358) and the State of Baden-Wurttemberg, Ger-¨ many, Ministry of Science, Research and Arts within the cooperative graduate program EAES of the University of Tubingen.¨

## REFERENCES

[1] R. Hoeger, A. Amditis, M. Kunert, A. Hoess, F. Flemisch, H.-P. Krueger, A. Bartels, A. Beutner, and K. Pagle, “Highly automated vehicles for intelligent transport: HAVEit approach,” in ITS World Congress, NY, USA, 2008.

[2] M. Aeberhard, S. Rauch, M. Bahram, G. Tanzmeister, J. Thomas, Y. Pilat, F. Homm, W. Huber, and N. Kaempchen, “Experience, results and lessons learned from automated driving on Germany’s highways,” Intelligent Transportation Systems Magazine, IEEE, vol. 7, no. 1, pp. 42–57, 2015.

[3] C. Byrne and Zentralverband Elektrotechnik und Elektronikindustrie, Handbook for robustness validation of automotive electrical/electronic modules. ZVEI, 2008.

[4] J. R. Black, “Mass transport of aluminum by momentum exchange with conducting electrons,” in Sixth Annual Reliability Physics Symposium, 1967. IEEE, 1967, pp. 148–159.

[5] T. Nirmaier, A. Burger, G. Harrant, A. Viehl, O. Bringmann, W. Rosenstiel, and G. Pelz, “Mission profile aware robustness assessment of automotive power devices,” in Design, Automation and Test in Europe Conference and Exhibition (DATE), 2014, 2014, pp. 1–6.

[6] U. Abelein, H. Lochner, D. Hahn, and S. Straube, “Complexity, quality and robustness - the challenges of tomorrow’s automotive electronics,” in Design, Automation & Test in Europe Conference & Exhibition (DATE), 2012. IEEE, 2012, pp. 870– 871.

[7] C. Katzschke, M.-P. Sohn, M. Olbrich, V. Meyer zu Bexten, M. Tristl, and E. Barke, “Application of mission profiles to enable cross-domain constraint-driven design,” in Proceedings of the conference on Design, Automation & Test in Europe. European Design and Automation Association, 2014, p. 66.

[8] G. Jerke and A. B. Kahng, “Mission profile aware IC design: a case study,” in Proceedings of the conference on Design, Automation & Test in Europe. European Design and Automation Association, 2014, p. 64.

[9] O. Chourabi, Y. Pollet, and M. B. Ahmed, “Ontology based knowledge modeling for system engineering projects,” in Second International Conference on Research Challenges in Information Science, 2008. RCIS 2008. IEEE, 2008, pp. 453–458.

[10] C. Byrne and Zentralverband Elektrotechnik und Elektronikindustrie, Robustness Validation - System Level, Appendix to Robustness Validation Handbook for EEM. ZVEI, 2014.

[11] J. Cardoso, J. Mendling, G. Neumann, and H. A. Reijers, “A discourse on complexity of process models,” in International Conference on Business Process Management. Springer, 2006, pp. 117–128.

[12] L. van Ruijven, “Ontology for systems engineering,” Procedia Computer Science, vol. 16, pp. 383–392, 2013.

[13] C. Hennig, A. Viehl, B. Kampgen, and H. Eisenmann,¨ “Ontology-based design of space systems,” in International Semantic Web Conference. Springer, 2016, pp. 308–324.

[14] S. Zander and R. Awad, “Expressing and reasoning on features of robot-centric workplaces using ontological semantics,” in International Conference on Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ. IEEE, 2015, pp. 2889–2896.

[15] Y. Zhang and X. Luo, “An ontology-based knowledge model for engineering material selections,” in Proceedings of the 2014 International Conference on Innovative Design and Manufacturing (ICIDM). IEEE, 2014, pp. 53–58.

[16] I. Horrocks, P. F. Patel-Schneider, H. Boley, S. Tabet, B. Grosof, M. Dean et al., “SWRL: A Semantic Web rule language combining OWL and RuleML,” W3C Member submission, vol. 21, p. 79, 2004.

[17] T. Tudorache, “Employing ontologies for an improved development process in collaborative engineering,” Technical University of Berlin, 2006.

[18] R. Rongen, “Knowlegde based qualification of semiconductor devices - introduction to the NXP Reliability Knowledge Framework,” in European CEEES Seminar, 2014.

[19] W3C OWL Working Group, “OWL 2 Web Ontology Language Document Overview (Second Edition) - W3C Recommendation 11 December 2012,” 2012.

[20] E. Prud’Hommeaux, A. Seaborne et al., “SPARQL query lan guage for RDF,” W3C recommendation, vol. 15, 2008.

[21] D. Carral, M. Knorr, and A. Krisnadhi, “OWL plus Rules = .. ?” 10th Extended Semantic Web Conference (ESWC), 2013.

[22] H. Paulheim, D. Oberle, R. Plendl, and F. Probst, “An architecture for information exchange based on reference models,” in International Conference on Software Language Engineering. Springer, 2011, pp. 160–179.

[23] S. Bechhofer, R. Volz, and P. Lord, “Cooking the Semantic Web with the OWL API,” in International Semantic Web Conference. Springer, 2003, pp. 659–675.

[24] M. Odersky, P. Altherr, V. Cremet, B. Emir, S. Maneth, S. Micheloud, N. Mihaylov, M. Schinz, E. Stenman, and M. Zenger, “An overview of the Scala programming language,” Tech. Rep., 2004.

[25] D. Steinberg, F. Budinsky, E. Merks, and M. Paternostro, EMF: eclipse modeling framework. Pearson Education, 2008.

[26] H. S. Thompson, D. Beech, M. Maloney, and N. Mendelsohn, “XML schema part 1: structures second edition,” W3C Recommendation (28 October 2004) http://www.w3.org/ TR, 2004.

[27] N. Drummond, A. L. Rector, R. Stevens, G. Moulton, M. Horridge, H. Wang, and J. Seidenberg, “Putting OWL in order: Patterns for sequences in OWL.” in OWLED. Citeseer, 2006.

[28] S. Rudolph, “Foundations of description logics,” in Reasoning Web. Semantic Technologies for the Web of Data. Springer, 2011, pp. 76–136.

[29] Y. Kazakov, “An extension of complex role inclusion axioms in the description logic SROIQ,” in Automated Reasoning. Springer, 2010, pp. 472–486.

[30] C. Bizer, T. Heath, and T. Berners-Lee, “Linked Data - the story so far,” Semantic Services, Interoperability and Web Applications: Emerging Concepts, pp. 205–227, 2009.