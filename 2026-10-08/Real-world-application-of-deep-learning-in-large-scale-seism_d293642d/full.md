# Real-world application of deep learning in large-scale seismic interference attenuation: A case study in the Camie field of Angola

Jing Sun<sup>1,\*</sup>, Song Hou<sup>2</sup>

<sup>1</sup>Delft University of Technology, Faculty of Electrical Engineering, Mathematics and Computer Science, Van Mourik Broekmanweg 6, Delft 2628 XE, The Netherlands.

<sup>2</sup> VIRIDIEN (UK) Ltd, Crompton Way, Manor Royal Estate, Crawley, West Sussex RH10 9QN, UK.

\*Corresponding author: Jing Sun (jing.sun@tudelft.nl)

## ABSTRACT

In marine seismic acquisition, seismic interference (SI) occurs when energy from nearby external seismic source(s) is captured. It typically appears as coherent noise with linear or nonlinear movement and varying amplitudes across different sail lines. SI is commonly observed and poses a challenge for seismic data processing. We present a case history of a previously proposed deep neural network (DNN)-based workflow applied for SI attenuation across a marine seismic block in the Camie Field of Angola. This field survey covers over 345 km<sup>2</sup> and is marked by the challenge of multiple SI types. The employed DNN-based workflow performs SI attenuation in the common shot domain based on a supervised learning framework: a small subset of the SIcontaminated data was first processed by a conventional geophysical algorithm to obtain an estimate of the SI noise, which was then manually blended with the SI-free common shot gathers from the same survey to generate the training pairs. To ensure signal fidelity, several techniques

were applied to improve the DNN’s performance. A key highlight of this case history is its scale: this represents a real-world, large-scale processing project and we present a comprehensive comparison of the DNN-based workflow with the conventional geophysical algorithm across the entire survey block, focusing on both processing quality and processing time. The results demonstrate the outstanding performance of the employed DNN-based workflow, which achieved higher SI removal accuracy, with less signal leakage and more complete SI removal. The promising results of this application also open up possibilities for integrating deep learning into other seismic denoising tasks. In addition, we discuss the limitations of this case history, aiming to provide insights for future research and applications in the field.

## INTRODUCTION

In densely worked offshore areas, seismic data are often contaminated by energy from nearby external seismic sources, typically air guns used by other vessels conducting seismic acquisition (Akbulut et al., 1984). This kind of noise is termed seismic interference, SI. SI presents as coherent linear or non-linear (curved) events within the shot domain, depending on the location of the external seismic source that generates it. Within a single survey, the angles of incidence for SI can vary significantly. Such variation is particularly noticeable when comparing one sail line to another, and it largely depends on the relative orientation of the external source in relation to the receivers. Similarly, notable differences in the amplitudes of SI can usually be observed across a survey, depending on the relative distance between the external seismic source and the receivers.

Originating from robust sources specifically designed for seismic exploration activities, SI often has a higher amplitude than other types of noise unless its seismic sources are very far away. In addition, SI propagates as guided waves traveling through the water column (Pekeris,

![](images/cc283e00579e4b63729a50bb645fc8c454ddf73c0785186b7c71c4efc7054669.jpg)  
(f)

1948), which also contributes to its distinct presence. Another notable aspect of SI is its preservation capacity. SI can remain remarkably consistent over large distances (Akbulut et al., 1984; Jansen et al., 2013). A point of concern arises when this high-amplitude SI starts to overlap with desired seismic signals. These desired signals, which are essentially reflections from subsurface layers, tend to have a much lower amplitude. This overlapping can pose significant challenges in separating SI from seismic signals.

Based on the positioning of the external source in relation to the receivers, SI can be categorized into three classes: SI coming from ahead, SI coming from broadside and SI coming from astern of the recording vessel. Figure 1 provides a graphical representation of shot gathers contaminated by SI at varying directions and distances.

(a)  
![](images/cc36e914cc084484edab9c403aefc30c4356090b4101576bc433f4e90ecb92cb.jpg)

(b)  
春  
![](images/75ebe36e2a2ca67a2d1ae917048ff86d504045b4b428fd23f615acfcd492e389.jpg)

(c)  
![](images/6169f262958d3b90efb55f3dfb77fd934629bdc17baa098c675e3bfa98d8de66.jpg)

(d)  
![](images/1d0ac9c756ce13931f15030c5bde900c056002e1b7ef8e85b6ded63ca339904f.jpg)

(e)  
![](images/00b7a7bd32c74f4567707fb4ed6e74fc5b0e48fd15be65ed358abb837cc29a95.jpg)

![](images/72eeb3e6c314c8ad8b42624635b7dc494fcf991dd8236a6047d73495489df7ed.jpg)

Figure 1: Schematics of shot gathers contaminated by SI from external source(s) at different positions. The blue triangle represents the recording vessel with streamers towed, and the red spot represents the external seismic source. Black lines show seismic reflections from the linked source, while red lines indicate SI (adapted from Hlebnikov et al., 2021).

Additionally, due to the intensity of modern marine activities, SI often coexists with other forms of marine noise, such as noise from the mechanical operations of non-seismic vessels, mainly fishing and cargo vessels (Brittan et al., 2008). Figures 2a and 2b show typical traffic in two marine regions where seismic acquisitions are conducted, specifically west of Norway and Angola (where the studied survey block is located). The colored dots and arrows denote various vessels trackable by the automatic identification system (AIS). Considering untraceable vessels, actual marine traffic is likely even more congested than illustrated in the figures.

![](images/951f306bc70557b22033c9983f319373a2445c5d27a6cf244fd4ece525058120.jpg)

![](images/fdf09011604539020447d9e8a062da016cea0ef6fd07254b18d1584a762b7de4.jpg)  
Figure 2: Examples of typical traffic in the marine seismic acquisition area: (a) and (b) show oceans to the west of Norway and Angola, respectively. Colored dots and arrows represent vessels for various uses monitored by the AIS.

Time-sharing was once considered a solution but proved inefficient, leading to unaffordable costs and vessel downtime (Elboth and Haouam, 2015). As a result, evolving strategies, especially cross-party coordination (Elboth 2017; Firth 2017) since 2015, have promoted synchronized vessel operations for effective SI control. The key idea is to accept the presence of SI types that can be attenuated at the processing stage and avoid including the highly intractable SI type that cannot be effectively attenuated, primarily those coming from close broadside and mirroring the moveout of desired seismic reflection data (Dhelie et al., 2013; Elboth and Haouam, 2015; Laurain et al., 2016). This enables a balance between acquisition time consumption and processing costs. Achieving this balance has been a dynamic process, as the tolerance for SI inclusion during acquisition depends on the knowledge of SI characteristics and advancements in SI attenuation techniques.

Over the years, many geophysical methods for the purpose of SI attenuation have been proposed. In general, they can be classified into two groups (Jansen, 2013). The first group involves breaking the coherence of SI by resorting the data into the common-receiver or common-offset domain, followed by the application of a denoising method such as f-x prediction filtering (Canales, 1984; Wang et al., 1989). Building on this, Gulunay and Pattberg (2001a, b) proposed guiding the detection of SI-contaminated shots through a magnitude threshold in the fx-y domain, followed by the prediction and subtraction of inline coherent SI using very short f-x prediction filters, as well as applying an f-x-y prediction filter to the frequency slice. Guo and Lin (2003) similarly used inline f-x prediction filters but explored adaptive rather than straightforward subtraction. Gulunay et al. (2004) proposed an f-x-y domain method combining inline and crossline f-x prediction filters for detecting and attenuating strong SI noise, particularly effective at later record times with high SI amplitudes. Rajput and Rajput (2006) further studied the effectiveness of f-x-y prediction filters in 3D marine seismic data, demonstrating strong SI attenuation with good signal preservation. Kommedal et al. (2007) studied SI attenuation in multicomponent 4D data, particularly for permanently installed arrays.

In addition to the variants of f-x prediction filtering, Tau-P transform has also been commonly adopted for SI attenuation in the seismic industry as it separates SI noise from signals based on their different move-out behavior (dip and/or curvature) in the time domain (Elboth et al., 2010). Yu (2011) investigated SI attenuation by transforming data to the Tau-ρ domain and applying a multidomain 3D spatial filter based on amplitude and frequency attributes. Zhang and Wang (2015) improved this approach by using progressive sparse Tau-P inversion, which yielded better results than the conventional Tau-P transform in both synthetic and field examples.

Elboth et al. (2017) proposed a line-mixing method that combines two adjacent sail lines to identify and remove occasional shot-to-shot coherent SI, assuming SI differs significantly between the sail lines. However, since the mixing is done in a mirroring way, data recorded on the inner streamer cables may suffer from signal loss in areas with rapidly changing geology in the crossline direction. To address this, Shen et al. (2019) proposed an improved line-mixing method using a cascaded mixing algorithm to reduce signal loss. They also presented a shotskipping and resplicing approach to decrease the consistency of shot-to-shot coherent SI.

The second group of SI attenuation methods is based on noise modelling and subtraction; the success of these methods strongly depends on their ability to build up an accurate model of the SI (Jansen 2013). Manin and Bonnot (1993) proposed that for noise independent of the recorded seismic shot, like SI, if the positions of the noise source and the receivers are known, the shape of the wavefront may be estimated and then refined using a shape recognition method when the noise train is long enough. Fookes et al. (2003) presented a similar method but estimated the noise source directly from the seismic data by identifying significant SI arrivals, enabling prediction of arrival times and flattening of noise trains via static shifts. Various methods, such as f-k or Radon transforms, median filters, or Karhunen-Loeve filters, can then be applied for denoising. Once the SI is attenuated, the data can be shifted back to their original time (Brittan et al., 2008). In practice, the data may be contaminated by SI from multiple sources and require iterative application of this method (Brittan et al., 2008).

Gulunay et al. (2005; 2006; 2008a) proposed the diffractor scan (DSCAN) method, which estimates noise source locations using semblance scan analysis on a surface grid. Gulunay (2008b) later proposed the Interfering Source Location Scanning (ISCAN) method, which also aims to remove SI from shallower depths. Guo et al. (2009) proposed another automatic technique that divides the shipping lane into a grid of cells to calculate a time trajectory for each, which requires prior knowledge of the lane, may be time-consuming and is limited in accuracy by the grid size. Lu et al. (2014) and Li et al. (2013) proposed an efficient method to calculate source locations from detected apices without scanning, though it fails when apices are not present on the seismic record. In a more recent work, Xu et al. (2019) applied density-based clustering to locate external SI sources by analyzing time delays between seismic traces, followed by iterative attenuation and an optional high-pass filter for detecting weak SI while preserving valid signals.

Although conventional geophysical SI attenuation methods have seen significant development, they still face important challenges. Removing SI noise with varying moveout (dip and/or curvature) across an entire survey typically requires manual testing and selection of different parameter sets, which can be labor-intensive. Moreover, in scenarios where SI noise originates from multiple directions or exhibits a dip similar to that of the underlying seismic reflections, conventional methods often struggle, resulting in noticeable residual noise and signal leakage.

Motivated by the great successful applications in computer vision and conventional image processing, in recent years, deep learning methods, especially various convolutional neural networks (CNNs), have emerged as potential alternatives to conventional seismic processing methods. Many related studies have been conducted within the seismic community on various processing tasks (Yu and Ma, 2021), such as deblending (Baardman and Tsingas, 2019; Richardson and Feller, 2019; Baardman and Hegge, 2020; Sun et al., 2020; Wang et al., 2020; Zu et al., 2020; Wang et al., 2021; Wang et al., 2023), ground roll attenuation (Li et al., 2018; Kaur et al., 2020; Yuan et al., 2020; Pham and Li, 2022; Yang et al., 2023), ocean bottom seismic noise attenuation (Wang et al., 2023; Seher et al., 2024; Wang et al., 2024; Zhu et al., 2025), as well as the focused problem of SI attenuation.

The earliest work on this topic can be traced back to Slang (2019), where the use of several CNN architectures was explored, laying the groundwork for using deep learning to attenuate SI. Slang et al. (2019) and Sun et al. (2020) further improved SI attenuation results in field marine seismic data. During the same period, Xu et al. (2020) also investigated DNN-based SI attenuation and proposed obtaining labeled training data through a sample generation method. While these earlier studies provided valuable insights and demonstrated the potential of DNNs for SI attenuation in field data, the processing quality had not yet been validated at a level consistent with industrial standards. Sun and Hou (2022) proposed a method to improve the DNN’s signal fidelity for general seismic signal separation tasks and demonstrated the effectiveness of this method in attenuating SI. Building on this progress, Sun et al. (2023) developed a DNN-based workflow to attenuate coherent noise, and it was tested on attenuating SI in two sail lines of marine towed-streamer data acquired from the Northern Viking Graben area. However, the processing scale remained limited, and to date, no study has presented a fullblock, large-scale application of DNN-based SI attenuation in direct comparison with a fieldtested geophysical algorithm.

In this work, we present a much larger-scale real-world processing project of attenuating different types of SI using the DNN-based workflow proposed by Sun et al. (2023), adopting the same U-Net architecture as in their study, which was adapted from the U-Net introduced by Ronneberger et al. (2015). While previous applications of DNN-based SI attenuation have been restricted to smaller-scale or synthetic datasets, the present study encompasses an entire marine seismic block in the Camie Field of Angola, covering over 345 km<sup>2</sup>. To the best of our knowledge, this represents the first published case study covering such a large area. As such, this research stands as a landmark in the field.

In the following sections, we introduce the employed geophysical algorithm and provide details on the implementation of the DNN-based workflow. The results include field data examples in the common shot domain and the stacked common midpoint (CMP) domain, along with detailed high-frequency root mean square (RMS) comparisons of individual sail lines and the entire survey. Based on the computational setup of the processing project, we also compare overall processing times of these two methods for the entire survey. The employed geophysical workflow follows a modular, multi-stage structure that is primarily CPU-based, with individual components optimized independently. This reflects a typical paradigm in commercial seismic processing systems. In contrast, the data preparation of the DNN-based workflow is CPU-based, while model training is executed on a graphics processing unit (GPU). We acknowledge that the comparison of processing times may not be as rigorous or directly comparable as in a pure research setup, where all algorithms are typically tested on the same graphics processing unit (GPU) or computational environment. However, this comparison aligns closely with most realworld production scenarios, where computational resources are often diverse, and optimization is tailored to specific project constraints. Finally, taking these aspects into consideration, we present a discussion and draw conclusions.

## THE STUDIED SURVEY

The acquisition area is in a deep-water environment off the coast of Angola, where broadband seismic 3D surveys were conducted (Cazier et al., 2014). The specific field we focus on is the Camie field. It is located in nearly 1700 meters of water and lies 100 kilometers offshore in Angolan Block 21, as shown in Figure 3 (Duval et al., 2015; Bhattacharya and Moore, 2026). This field includes 41 sail lines, each equipped with 12 cables. Each sail line has about 800 shots, which means that across the whole survey, there are about 400,000 (41x800x12) shot/cable combinations.

![](images/8eccb1cd9b12afb3c3104a785b5cde3287c64d960e85541cff62d73a3067b8aa.jpg)  
Figure 3: Location of the studied field in the Outer Kwanza Basin, Angola.

This acquisition took place during a particularly busy season, with SI noise caused by other seismic vessels conducting acquisition in the area, as well as coherent noise from numerous and unchecked local fishing vessels. Consequently, a wide range of SI, each with distinct characteristics, was recorded. Some of these SI types presented significant challenges in their removal from the dataset. For instance, in some cables, SI originating from two different directions was simultaneously captured within one shot gather, as shown in Figure 4a. Conventional parameter-based approaches then require repeated manual adjustments to suppress both types of SIand still risk incomplete suppression. In some other cables, the SI exhibited an extremely flat profile, as shown in Figure 4b. In addition, there was SI with a dip similar to that of the desired signals, as shown in Figure 4c. Such interference is difficult to distinguish from primary reflections using standard moveout- or dip-based filters, often resulting in residual noise or leakage of useful signals. Conversely, there were instances of SI that appeared to be the inverse of the former, as shown in Figure 4d. The diverse nature of these SI characteristics requested the development and application of advanced techniques, such as the deep learning approach used in this project, to effectively address these challenges and enhance the quality of the seismic data.

![](images/46d2d24fdaed0575eedf151cef79f23a021603fb211ea471a82bfb8d88be8450.jpg)

![](images/fccd43be7900e0dbc702499c0057e8add34e43b7239780c9311411d409067400.jpg)

![](images/343f1efb3d304397eb49d5cdac855a21f75f241a1fa1d32f499f7c0a870d87a3.jpg)

![](images/870707ed9b82d56ce530fd7d7ecd01c8f0f16c26356c2e9d8c9977b9c1536654.jpg)  
Figure 4: (a) to (d) show four shot gathers contaminated by different SI during the acquisition.

The purple arrows highlight SI noise. The blue arrow indicates a desired signal.

## THE EMPLOYED DNN-BASED WORKFLOW AND GEOPHYSICAL ALGORITHM

Figure 5 illustrates how the DNN-based workflow proposed by Sun et al. (2023) and Hou and Sun (2023) is applied to attenuate SI in the Camie field. To train the DNN, we constructed the training set as follows: First, 5 sail lines × 6 cables, covering all types of SI mentioned earlier, were selected and passed through a conventional geophysical SI attenuation algorithm. These accounted for 6% of the entire field SI-contaminated data from this survey.

The conventional geophysical algorithm involved preconditioning of common shot gathers by separation of dips of primary signal and SI noise where possible, and rank-reduction (Trickett et al., 2012) based denoising to improve signal-to-noise ratio (S/N) of coherent events (both wanted signal and unwanted SI noise). Next, SI noise was modelled by a sparse tau-p inversion (Zhang and Wang, 2015) scheme and adapted to the input common shot gathers with a complex wavelet adaptation which combines complex Morlet wavelet frame with unary complex Wiener filters (Ventosa et al., 2012). From this step, we obtained an estimate of the various types of SI (from now on called the SI model).

Our next step was to shuffle the SI model obtained from the conventional algorithm and blend it with SI-free shots. This survey has data of 5 sail lines × 3 cables that were naturally free from SI contamination during the acquisition. Data augmentation, including vertical translation and amplitude modulation, was applied to the SI-free shots and the SI model in order to generate more training pairs.

In addition, two procedures for improving signal fidelity were adopted. One was to inject a set of uniform random noise into the SI-contaminated input shot gather and the ground truth of the SI-free shot gather. The RMS amplitude of the random noise randomly varied from 8000 to 30000 for each shot gather. For context, the RMS amplitude of each field SI-contaminated shot in this survey typically ranged from 35000 to 45000. The other procedure was to use two adjacent shots on both sides of the to-be-processed shot were used as additional channels of the input. The DNN was trained to predict both the SI and a blend of the SI-free shot with the intact random noise. For a more detailed explanation of the reasoning behind these procedures, we refer readers to Sun et al. (2023) and Sun and Hou (2022).

![](images/fd9830aa5c4736de6a0906b81269a9dc2330a6912d76f0234b54fc74cf5d145c.jpg)  
Figure 5: Illustration of how the DNN-based workflow proposed by Sun et al., (2023) is implemented in this survey.

Once the DNN is trained, it can be used to attenuate SI noise in the remaining SIcontaminated data from the entire survey (i.e., 91% of the dataset in this case study). In Figure 5, a dashed grey arrow links the trained DNN to the 6% SI-contaminated data that provided the SI model used in generating the training pairs. This is included to reflect that, in a commercial production setting, reprocessing this 6% using the trained DNN could be justified, as processing accuracy would be the top priority. However, from a purely research-oriented perspective, applying the trained DNN to data that contributed to the training process raises concerns regarding evaluation fairness. More detailed insights can be found in the discussion.

We employed the same DNN architecture as Sun et al. (2023), the study that introduced the DNN-based workflow. Their model was a slightly modified version of the original U-Net by Ronneberger et al. (2015). A visualization of the employed architecture can be found in Figure 6. Similar DNNs have been widely used in seismic data processing for various tasks.

![](images/f6145ea6f0f9f29ec2772c90c38a43eccaa2f6b6da0361e356c4dea0e9ad4ca1.jpg)

Figure 6: Architecture of the employed DNN (Sun et al., 2023). Each rectangle represents a collection of feature maps, and the numbers above (e.g. 32, 48, 72 and 108) represent the number

## of feature maps.

The utilized DNN follows the standard U-Net structure with an encoder-decoder layout and skip connections that preserve spatial information by concatenating encoder features with their corresponding decoder layers. The encoder consists of 2D convolutional layers (filter size of 3×3) with ReLU (Rectified Linear Unit) activation, followed by max pooling layers (stride of 2 and a pool size of 2×2) that progressively downsample the input while extracting hierarchical features. The decoder uses transposed 2D convolutions to upsample the feature maps and reconstruct the output, with each level refining spatial details.

The main difference from the original U-Net is that, while the original U-Net doubles the number of feature channels (×2) at each downsampling step, this architecture increases them more gradually, by a factor of 1.5×, balancing model complexity and memory efficiency. No activation function is applied at the final output layer. After hyperparameter tuning, the Adam optimizer (Kingma and Ba, 2014), with a learning rate of 0.001, was selected, with a batch size of 4. The training process converged after approximately 80 epochs.

The mathematical formulations of the basic operations employed in this architecture— including 2D convolution, max pooling, transposed convolution, and concatenation—can be found in standard deep-learning literature such as Goodfellow et al. (2016), Dumoulin and Visin (2016), and the original U-Net paper by Ronneberger et al. (2015). This network architecture can be reproduced by using the original U-Net implementation (Ronneberger et al., 2015) and adjusting the number of feature maps according to those shown in Figure 6.

## RESULTS

In this section, we demonstrate the performance of the trained U-Net when applied to unseen data from the processed survey, based on a comparison with the conventional geophysical algorithm. We analyze the results in both the shot domain and the stacked CMP domain. Notably, although these examples are not drawn from the exact same cables (considering that the same dataset should never be repeatedly used as both the training and test datasets for a DNN), they serve as indicative representations of the quality of the SI model used to create the DNN's training inputs of the employed U-Net.

In addition to evaluating processing quality, we compare the processing time of the DNNbased workflow (employing the U-Net in Figure 6) with the case of solely using the conventional geophysical algorithm to process the entire survey.

## Comparison in Processing Quality

As previously introduced, the field seismic shot gathers acquired from this survey are mainly contaminated by four types of SI. To provide clarity, examples of each type are shown in Figures 4a to 4d. In this section, we show the SI attenuation results obtained by employing the conventional geophysical algorithm in Figures 7a to 7d. In these results, visible SI residuals are evident in each example, as denoted by the purple arrows. Additionally, some signal damage can be observed, as indicated by the blue arrows in Figure 7b and Figure 4b.

For comparison, the SI attenuation results from the DNN-based workflow (employing the U-Net in Figure 6) are shown in Figures 8a to 8d. These results clearly show much better quality. The SI residuals are significantly reduced compared to those from the conventional algorithm in Figure 7a to 7d. This improvement is even more noticeable when comparing the removed SI. Figures 9a to 9d show the SI removed by the conventional geophysical algorithm, which is incomplete and discontinuous. In contrast, the DNN-based workflow produces much more continuous and accurate SI extraction, with notably less signal leakage, as shown in Figures 10a algorithm and the DNN-based workflow, respectively. While the overall difference is not easily noticeable, a relatively distinct difference is highlighted by the blue boxes in the bottom left.

![](images/65ba9ffe3eccc9c52ffe093f37c580fea45d08aa506ffe30cddeb135cea1d399.jpg)

Figure 7: (a) to (d) show the SI attenuation results of the conventional algorithm for the four SIcontaminated shot gathers in Figure 4a to 4d. The purple arrows indicate the SI residual, while the blue arrow indicates the signal damage caused by the conventional geophysical algorithm.  
![](images/04c7a07e5203efdd1434a8f1a2fa862eae64509211aa71bb14b77c02b998dc70.jpg)

![](images/1cbd6917afd291fed6e923d047f7ba45376323e0ec7b92183ba755fbd6ad4375.jpg)

![](images/156c80afc4118ec45369edb6afe39e605d0ac43f1c200040a9dd3e67d519a3b1.jpg)

![](images/01d7eb8191031a2d3ac25de5aeb8f4dff60e47fd7eae61e99a34d704c70642b7.jpg)  
Figure 8: (a) to (d) show the SI attenuation results of the DNN-based workflow (employing the U-Net in Figure 6) for the four SI-contaminated shot gathers in Figure 4a to 4d. The purple arrows indicate much less SI residual compared to Figure 7a to 7d. The blue arrow indicates that less signal damage was caused by the trained U-Net than by the conventional geophysical algorithm (Figure 7b).

![](images/e8af2264a67e9c569dad47f22ebbaa7b5368d88475e0ced1bca74d26f08f6661.jpg)

Figure 9: (a) to (d) show the attenuated SI of the conventional geophysical algorithm for the four SI-contaminated shot gathers in Figure 4a to 4d. The purple arrows indicate the incomplete SI removal, while the blue arrow indicates the signal leakage caused by the conventional algorithm.  
![](images/fd579b9423208b352ce0dc71f1cd2905d6d51194c29463eed58100dd89e19856.jpg)  
Figure 10: (a) to (d) show the attenuated SI of the DNN-based workflow (employing the U-Net in Figure 6) for the four SI-contaminated shot gathers in Figure 4a to 4d. The purple arrows indicate a much more complete SI removal compared to Figure 9a to 9d. The blue arrow indicates that almost no signal leakage was caused by the trained U-Net.  
An example in the stacked CMP domain is given in Figure 11. Figure 11a shows a time window extracted from a CMP stack of data contaminated by SI during the acquisition. Figures 11b and 11c then show the SI attenuation results obtained using the conventional geophysical

For better comparison, Figures 11d and 11e show the removed SI using the conventional geophysical algorithm and the DNN-based workflow, respectively. As observed in the shotdomain examples, the DNN-based workflow demonstrates a significant improvement in processing quality compared to the conventional geophysical algorithm. It exhibits a more complete and comprehensive extraction of the SI, as indicated by the blue circle, and significantly reduces signal leakage, as indicated by the blue arrows.

![](images/403c8e9ba8c90fb26fc78846d76c7536308881cc15cc96f1d3b47ee98c116704.jpg)

![](images/4b5c069f20f9f8a9869cbe18d797222733f45f36a4d43acc645cb061c49a63ad.jpg)

![](images/35bb14efb298195b3d411ecfd6fd0692e11b0ed7331096e097ef9d9906a98315.jpg)

Figure 11: CMP stacks of (a) the field data contaminated by SI during the acquisition, (b) SI attenuation result using the conventional geophysical algorithm, (c) SI attenuation result using the DNN-based workflow (employing the U-Net in Figure 6), (d) attenuated SI using the conventional geophysical algorithm and (e) attenuated SI using the DNN-based workflow (employing the U-Net in Figure 6). Arrows and ellipses respectively highlight the comparisons of signal leakage and the completeness of SI extraction between the two methods.

These observations are also supported by the comparison of the RMS amplitude, calculated based on the average RMS per shot, specifically in the late time (i.e., 8.0 s to 10.0 s) and high frequency (i.e., above 20 Hz) sections, where the SI mainly exists. Figure 12 presents the deep high-frequency RMS levels of two sail lines located at different areas of the survey. Each column shows examples for comparison from the same sail line: SI-contaminated raw data (top), SI attenuation result using the conventional geophysical algorithm (middle), and SI attenuation result using DNN-based workflow (bottom). In the raw data panels, intense vertical stripe patterns, appearing in red and yellow, highlight strong SI contamination. The conventional geophysical algorithm partially suppresses them, but residual striping and signal distortion, seen as light green streaks, remain. In contrast, the SI attenuation results from the DNN-based workflow show a notable reduction in SI, with the vertical artifacts largely removed and the background appearing more uniform in blue-green shades. This color transition illustrates improved continuity and preservation of coherent signals in the deeper channels.

Figure 13a illustrates the overall deep high-frequency RMS level for the entire survey. The comparison of the two workflows across the entire survey is shown in Figures 13b and 13c. In the raw SI-contaminated data (Figure 13a), extensive high RMS values appear throughout the survey, especially along central and northeastern sail lines, highlighting severe SI contamination, as indicated by the high-amplitude regions in red. The conventional geophysical algorithm (Figure 13b) achieves moderate attenuation, reducing SI in less affected regions; however, significant residuals remain in areas of heavy contamination. In contrast, the DNN-based workflow (Figure 13c) exhibits a more consistent and effective SI reduction across the entire survey. Most sail lines show lower RMS levels, transitioning into green and blue, indicating better suppression of SI energy even under stronger contamination conditions. It is also worth noting that, although the DNN-based workflow overall outperforms the conventional geophysical algorithm, both methods exhibit comparable limitations in the regions most heavily affected by SI contamination (e.g., central and northeastern sail lines).

![](images/b9e07937a2fd571cd939a2267a2d2a5406e26d0744d5be65ff68d8bf9d307fee.jpg)  
Figure 12: Deep high-frequency RMS of two sail lines in the studied survey. In each column the figures from top to bottom are respectively the SI-contaminated raw data, SI attenuation result using the conventional geophysical algorithm and SI attenuation result using the DNN-based workflow (employing the U-Net in Figure 6). SI appears predominantly as vertical stripes in these plots.

![](images/c936264e71e69013f6c8a9559a98c3d269b9cd715f07bc989ca1e70c22a9e7cc.jpg)

![](images/78f3e54c843598c91fad91de233e28c779983c826d4f269c601ef5c3e832c1a8.jpg)

![](images/324cd7c2da9f1d70d091f34c33d1bd3d23908d9f9477cab446736334f24e71d7.jpg)  
Figure 13: Deep high-frequency RMS of the studied survey. (a), (b) and (c) represent

respectively the SI-contaminated raw data, SI attenuation result of the conventional geophysical algorithm and SI attenuation result of the DNN-based workflow (employing the U-Net in Figure 6). Each slant line represents a sail line of the survey, and the color indicates the RMS amplitude.

## Comparison in Processing Time

Processing one sail line with the conventional geophysical algorithm took approximately 3.5 hours, excluding the time spent on parameter optimization. When scaled up for the entire survey, which consists of 41 sail lines in total, the computational time would amount to approximately 143.5 hours using a single GPU (Nvidia Quadro RTX 6000). In the DNN-based workflow, data from 6 cables × 5 sail lines were first processed using the conventional geophysical algorithm to obtain SI models of different SI types, which took 8.75 hours. Following this, the data preparation and the DNN training process took approximately 35 hours.

Once the DNN was trained, it could process each sail line in just 15 minutes, amounting to 10.25 hours for the entire survey. Note that when computing this time cost, we take into account the potential reprocessing of the SI-contaminated sail line that provided the SI model used in the DNN’s training pairs. As a result, the total computational time for the DNN-based workflow was approximately 54 hours using the same computational resources, which is roughly one-third of the time required by the conventional geophysical algorithm. The total area of the studied survey is $3 4 5 \ \mathrm { k m } ^ { 2 } .$ . The number of shot gathers used here has met the needs of DNN training. If applying the same DNN-based workflow to even larger surveys, the DNN training time will not increase significantly, but the efficiency benefits of the DNN during the application phase will be further scaled up.

## DISCUSSION

Deep learning-based methods are recognized for their great potential but have yet to be widely implemented in real-world seismic processing projects. One key reason is that their signal fidelity often falls short of industry standards. To address this problem, much research has been conducted. From the promising results shown in our presented case study, we see positive signs that deep learning could benefit some real-world tasks; at least, it has started contributing to our focused area of SI attenuation. We provide a detailed analysis of the effectiveness of a DNN-based workflow based on a comparison with a conventional geophysical algorithm. The data used for this study was acquired from a real-world marine seismic survey block off the coast of Angola, covering over 345 km<sup>2</sup>. The DNN-based workflow demonstrated greater overall processing time than the conventional geophysical algorithm. Specifically, the entire process, including data preparation and DNN training, required less than half the computational time of the conventional geophysical algorithm. This efficiency gain is expected to scale with the increase in the size of the processing area. Apart from processing time, the DNN-based workflow showed a general improvement in processing quality over the conventional geophysical algorithm, especially in reducing SI in regions with poor SnR or where the SI noise has similar dips to the desired signal. Our field data examples demonstrate that even the DNN model was trained on an incomplete and discontinuous SI model produced from the conventional workflow, it could effectively attenuated SI during the inference phase. These results mainly from two factors: Firstly, although not perfect, the SI models derived from the conventional algorithm cover the types of SI that the DNN model is expected to

learn to address. For SIs that are very weak in amplitude and cannot be distinguished from the signal using the conventional geophysical algorithm, the DNN model would also struggle to detect and identify them due to their absence during the training phase. Secondly, there are some data truly free from SI during the acquisition of this survey, which can serve as satisfactoryquality ground truth of the desired signal, specifically, 5 sail lines × 3 cables out of the total 41 sail lines × 12 cables.

A more challenging scenario arises when the entire survey is contaminated by SI, forcing us to use SI-attenuated shots—also produced from a conventional geophysical algorithm—to create the training pairs for the DNN. In such cases, refining the conventional geophysical algorithm becomes essential, as it plays a key role in producing both the SI model and the pseudo-SI-free shots needed for DNN training. This introduces some uncertainties regarding the performance of the DNN-based workflow.

In this study, all displayed examples are from applying the trained DNN to unseen SIcontaminated data—specifically, the 91% of the raw dataset not involved in training—and are compared against SI attenuation results from the conventional geophysical algorithm. From a practical standpoint in a production setting, where the primary objective is to enhance processing quality, applying the trained DNN to that 6% of raw SI-contaminated data (used to generate the SI models for training pair construction) could also be justified. However, from a methodological perspective, doing so raises concerns about potential data leakage and evaluation fairness. Although the DNN was not trained on the SI-free versions of these shots (as such versions do not naturally exist), the SI models were derived from them, introducing a partial dependency. To maintain a clear separation between training and inference, we intentionally excluded this 6% from the evaluation examples, though we acknowledge it as a potentially reasonable step in real-

world commercial workflows.

In addition, several limitations must be acknowledged. One critical concern is overfitting, particularly when the DNN is exposed to limited or homogeneously structured training data. While we mitigated this risk through diverse augmentation and pairing strategies, it remains a concern. A separate but related concern arises if the training SI models do not adequately reflect the full complexity present in the field survey. This also explains why, in the present case study, we chose to process 5 sail lines × 6 cables, rather than all cables from only one or two sail lines. Although both approaches would yield a similar number of SI models, the former ensures broader coverage of the SI types encountered across the survey. In practice, this places a higher demand on the geophysicist operating the conventional geophysical algorithm, as they must handle a wider range of SI characteristics from different sail lines.

Moreover, the employed DNN-based workflow is a purely data-driven method, and its reliance on survey-specific training may limit its generalizability to other seismic surveys. Our experience suggests that a DNN model trained on one survey is unlikely to perform well on a different survey without retraining or fine-tuning. This reflects a domain shift challenge, as the DNN struggles to adapt to new data distributions not represented in the training set. This underscores the potential need for domain adaptation strategies before broader deployment.

While these cases are beyond the scope of our present study, we recognize their significance and advocate for further exploration. Further investigation into the robustness and capabilities of this DNN-based workflow is necessary. Future work could also include exploring the application of this DNN-based workflow to other coherent noise types in seismic processing.

## CONCLUSIONS

We applied a DNN-based workflow to attenuate SI for an entire marine seismic survey block in the Camie Field of Angola, covering over 345 km<sup>2</sup>. Specifically, a U-Net was employed. This area is marked by various types of SI and significant maritime activity. We provide a thorough comparison of the DNN-based workflow versus a conventional geophysical algorithm on shot gathers, CMP stacks, and deep high-frequency RMS analysis. The DNN-based workflow outperforms in both processing quality and time. This paper presents the largest-ever case study comprehensively comparing a DNN-based workflow with a conventional geophysical algorithm in a real-world SI attenuation project.

Despite this promising case study, further investigation of this DNN-based workflow’s robustness and capability is needed. Up to now, the implementation of such a DNN-based workflow still relies on a geophysical algorithm to provide at least the labels for the SI, and those labels are expected to cover all the SI types that the DNN is expected to process at the inference stage. A DNN under supervised learning is a purely data-driven technique for which the quality of the training pairs is crucial; thus, developing better geophysical algorithms remains important. The selection of good data samples also requires human effort and experience.

Overall, for large-scale real-world processing, it is still essential to begin with a solid conventional geophysical algorithm, as it forms the foundation upon which the DNN can provide further improved SI-attenuation results. Ultimately, only processors with experience in both geophysical and deep learning methods will be able to balance the effort needed to tune these two parts and achieve output of the best value. Besides, future work may include applying the DNN-based workflow to other surveys, evaluating alternative DNN architectures, and exploring the incorporation of physics-based constraints to enhance generalization.

## ACKNOWLEDGMENTS

The authors thank VIRIDIEN (formerly CGG) Earth Data for granting permission to use the field seismic data from Angola and the location map for the purposes of research and publication. The authors thank Dr. Vetle Vinje for his invaluable insights. The authors thank Mr. Cian Ryan and Mr. Harrison Moore for their contributions to the data processing using the geophysical algorithm employed as a benchmark in this study. This research work was funded by VIRIDIEN and NFR (Norwegian Research Council) through an industrial Ph.D. grant (project number 314179). The first author thanks the University of Oslo and VIRIDIEN (formerly CGG) for providing the environment and support in which this research was carried out. This work was completed while the first author was affiliated with the University of Oslo and VIRIDIEN (formerly CGG).

## REFERENCES

Akbulut, K., O. Saeland, P. Farmer, and T. Curtis, 1984, Suppression of seismic interference noise on Gulf of Mexico data: 54th Annual International Meeting, SEG, Expanded Abstracts, 527-529, doi: 10.1190/1.1894083.

Baardman, R., and C. Tsingas, 2019, Classification and suppression of blending noise using convolutional neural networks: Middle East Oil and Gas Show and Conference, SPE, Conference Papers.

Baardman, R.H., and R.F. Hegge, 2020, Machine learning approaches for use in deblending: The Leading Edge, 39, no.3, 188-194, doi: 10.1190/tle39030188.1.

Bhattacharya, M., and H. Moore, 2022, Angolan Kwanza Basin: Expanding proven opportunities, https://geoexpro.com/angolan-kwanza-basin-expanding-proven-opportunities/, accessed 20

Brittan, J., L. Pidsley, D. Cavalin, A. Ryder, and G. Turner, 2008, Optimizing the removal of seismic interference noise: Leading Edge, 27, no. 2, 166-175, doi: 10.1190/1.2840363.

Canales, L., 1984, Random noise reduction: 54th Annual International Meeting, SEG, Expanded Abstracts, doi: 10.1190/1.1894168.

Cazier, E. C., C. Bargas, L. Buambua, S. Cardoso, H. Ferreira, K. Inman, A. Lopes, T. Nicholson, C. Olson, A. Saller, J. Shinol, 2014, Petroleum geology of Cameia field, deepwater pre-salt Kwanza basin, Angola, West Africa: AAPG International Conference and Exhibition, 4, 14- 176, https://www.searchanddiscovery.com/documents/2014/20275cazier/ndx\_cazier.pdf

Dhelie, P. E., D. Harrison-Fox, and M. F. Abbasi, 2013, Increasing the efficiency of acquisition in a busy North Sea season — Dealing with seismic interference: 75th Annual Meeting, EAGE, Extended Abstracts, We 14 14, doi: 10.3997/2214-4609.20130885.

Dumoulin, V., and F. Visin, 2016, A guide to convolution arithmetic for deep learning: arXiv preprint arXiv:1603.07285.

Duval, G., J. Mann and L. Houston, 2015, Angola, Kwanza Basin: Sub-salt plays of the ultradeep water, https://geoexpro.com/angola-kwanza-basin-exploring-further-and-deeper-foroil-and-gas/, accessed 20 April 2026.

Elboth, T., I. V. Presterud, and D. Hermansen, 2010, Time-frequency seismic data denoising: Geophysical Prospecting, 58, 441-453, doi: 10.1111/j.1365-2478.2009.00846.x.

Elboth, T., and F. Haouam, 2015, A seismic interference noise experiment in the Central North Sea: 77th Annual Meeting, EAGE, Extended Abstracts, Th N116 03, doi: 10.3997/2214- 4609.201413314.

Elboth, T., 2017, Coordinating marine acquisitions to tackle seismic interference noise: 79<sup>th</sup> Annual Meeting, EAGE, Extended Abstracts, 1-6, doi: 10.3997/2214-4609.201700778.

Elboth, T., H. Shen, and J. Khan, 2017, Advances in seismic interference noise attenuation: 79th Annual Meeting, EAGE, Extended Abstracts, 1-5, doi: 10.3997/2214-4609.201700579.

Firth, J., 2017, New seismic interference mitigation techniques eliminate time-sharing: Oilfield Technology, https://www.cgg.com/sites/default/files/2020-11/cggv\_0000028842.pdf.

Fookes, G., C. Warner, and R. Van Borselen, 2003, Practical interference noise elimination in modern marine data processing: Technical Program, SEG, 1905-1908, doi: 10.1190/1.1817692.

Goodfellow, I., Y. Bengio, and A. Courville, 2016, Deep learning: Cambridge MIT press.

Gulunay, N., and D. Pattberg, 2001a, Seismic crew interference and prestack random noise attenuation on 3D marine seismic data: 63rd Annual Meeting, EAGE, Extended Abstracts, A-13, doi: 10.3997/2214-4609-pdb.15.A-13.

Gulunay, N., and D. Pattberg, 2001b, Seismic interference noise removal: 71st Annual International Meeting, SEG, Expanded Abstracts, 1989-1992, doi: 10.1190/1.1816530.

Gulunay, N., M. Magesan, and S. Baldock, 2004, Seismic interference noise attenuation: Technical Program, SEG, 1973-1976, doi: 10.1190/1.1843302.

Gulunay, N., M. Magesan, and J. Connor, 2005, Diffracted noise attenuation in shallow water 3D marine surveys: SEG Technical Program, 2138-2141, doi: 10.1190/1.2148136.

Gulunay, N., M. Magesan, and J. Connor, 2006, Diffractor scan (DSCAN) for attenuating scattered energy: 68th Annual Meeting, EAGE, Extended Abstracts, G033, doi: 10.3997/2214-4609.201402371.

Gulunay, N., 2008a, Localization of diffracted seismic noise sources using an array of seismic sensors: 5th IEEE Sensor Array and Multichannel Signal Processing Workshop, 198-202, doi: 10.1109/SAM.2008.4606854.

Gulunay, N., 2008b, Two different algorithms for seismic interference noise attenuation: Leading Edge, 27, 176-181, doi: 10.1190/1.2840364.

Guo, J., and D. Lin, 2003, High-amplitude noise attenuation: 73rd Annual International Meeting, SEG, Expanded Abstracts, 1893-1896, doi: 10.1190/1.1817688.

Guo, M. H., J. Cai, J. Specht, and B. Wang, 2009, Constrained propeller ship noise removal and its application to OBC data: Technical Program, SEG, 28, 3307-3311, doi: 10.1190/1.3255546.

Hlebnikov, V., T. Elboth, V. Vinje, and L. J. Gelius, 2021, Noise types and their attenuation in towed marine seismic: A tutorial: Geophysics, 86, no. 2, W1-9, doi: 10.1190/geo2019- 0808.1.

Hou, S., and J. Sun, 2023, Seismic interference noise attenuation using DNN: US Patent US 2023/0105075 A1.

Jansen, S., 2013, Two marine seismic interference attenuation methods: Master’s thesis, University of Oslo, http://urn.nb.no/URN:NBN:no-34309.

Jansen, S., T. Elboth, and C. Sanchis, 2013, Two seismic interference attenuation methods based on automatic detection of seismic interference moveout: 75th Annual Meeting, EAGE, Extended Abstracts, We 14 15, doi: 10.3997/2214-4609.20130886.

Laurain, R., G. Pattison, T. Elboth, and J. Pollatos, 2016, Cooperating for optimizing seismic acquisition — A case study in the Horda Tampen area: 78th Annual Meeting, EAGE,

Extended Abstracts, Tu STZ 03, doi: 10.3997/2214-4609.201600857.

Li, H., W. Yang, and X. Yong, 2018, Deep learning for ground-roll noise attenuation: Technical Program, SEG, Expanded Abstracts, 1981-1985, doi: 10.1190/segam2018-2981295.1.

Li, Z., W. K. Lu, Y. Q. Zhang, and B. R. Zhen, 2013, Localization and attenuation of diffracted seismic noise in shallow water: 75th Annual Meeting, EAGE, Extended Abstracts, Th P08 09, doi: 10.3997/2214-4609.20130279.

Lu, W. K., Y. Q. Zhang, and B. R. Zhen, 2014, Automatic source localization of diffracted seismic noise in shallow water: Geophysics, 79, no. 2, V23-V31, doi: 10.1190/geo2013- 0265.1.

Kaur, H., S. Fomel, and N. Pham, 2020. Seismic ground‐roll noise attenuation using deep learning: Geophysical Prospecting, 68, no.7, 2064-2077, doi: 10.1111/1365-2478.12985.

Kingma, D. P., and J. Ba, 2014, Adam: A method for stochastic optimization: arXiv preprint arXiv: 1412.6980.

Kommedal, J. H., P. H. Semb, and T. Manning, 2007, A case of SI attenuation in 4D seismic data recorded with a permanently installed array: Geophysics, 72, Q11-Q14, doi: 10.4043/18973- MS.

Manin, M., and J. N. Bonnot, 1993, Industrial and seismic noise removal in marine processing: 55th Annual Meeting, EAGE, Extended Abstracts, BO31, doi: 10.3997/2214- 4609.201411442.

Pekeris, C. L., 1948, Theory of propagation of explosive sound in shallow water: Geological Society of America Memoirs, 27, 1-116, doi: 10.1130/MEM27-2-p1.

Pham, N., W. and Li, 2022, Physics-constrained deep learning for ground roll attenuation:

Geophysics, 87, no.1, V15-V27, doi: 10.1190/geo2020-0691.1.

Rajput, S., and S. Rajput, 2006, Signal preserving seismic interference noise attenuation on 3D marine seismic data: 76th Annual International Meeting, SEG, Expanded Abstracts, 2747- 2751, doi: 10.1190/1.2370094.

Richardson, A. and C. Feller, 2019, Seismic data denoising and deblending using deep learning: arXiv preprint arXiv:1907.01497.

Ronneberger, O., P. Fischer, and T. Brox, 2015, U-Net: Convolutional networks for biomedical image segmentation: 18th International Conference on Medical Image Computing and Computer-Assisted Intervention, 234-241, doi: 10.1007/978-3-319-24574-4\_28.

Seher, T., G. Yalcin, M. Roberts, Y. and Ren, 2024, Shear wave noise attenuation in ocean bottom node data using machine learning: 85th Annual Meeting, EAGE, Extended Abstracts, 1-5, doi: 10.3997/2214-4609.202410498.

Shen, H., T. Elboth, C. Tao, G. Tian, H. Wang, L. Qiu, and J. Zhou, 2019, Using data‐regrouping methods to attenuate shot‐to‐shot coherent interference noise in marine seismic data: Earth and Space Science, 6, no.7, 1098-1108, doi: 10.1029/2018EA000485.

Slang, S., 2019, Attenuation of seismic interference noise with convolutional neural networks: Master’s thesis, University of Oslo, http://urn.nb.no/URN:NBN:no-73201.

Slang, S., J. Sun, T. Elboth, S. McDonald, and L. J. Gelius, 2019, Using convolutional neural networks for denoising and deblending of marine seismic data: 81st Annual Meeting, EAGE, Extended Abstracts, Tu R06 05, doi: 10.1111/1365-2478.12893.

Sun, J., S. Slang, T. Elboth, T. Larsen Greiner, S. McDonald, and L.J. Gelius, 2020, A convolutional neural network approach to deblending seismic data: Geophysics, 85, no.4,

WA13-WA26, doi: org/10.1190/geo2019-0173.1.

Sun, J., S. Slang, T. Elboth, T. Larsen Greiner, S. McDonald, and L. J. Gelius, 2020, Attenuation of marine seismic interference noise employing a customized U-Net: Geophysical Prospecting, 68, 845-871, doi: 10.1111/1365-2478.12893.

Sun, J., and S. Hou, 2022, Improving signal fidelity for deep learning-based seismic interference noise attenuation: Geophysical Prospecting, doi: 10.1111/1365-2478.13268.

Sun, J., S. Hou, and A. Triki, 2023, Deep neural network-based workflow for attenuating seismic interference noise and its application to marine towed-streamer data from the northern Viking Graben: Geophysics, 88, no. 2, B69-B77, doi: 10.1190/geo2021-0675.1.

Trickett, S., L. Burroughs, and A. Milton, 2012, Robust rank-reduction filtering for erratic noise: Technical Program, SEG, 1-5, doi: 10.1190/segam2012-0129.1.

Ventosa, S., S. Le Roy, I. Huard, A. Pica, and L. Duval, 2012, Unary adaptive subtraction of joint multiple models with complex wavelet frames: Technical Program, SEG, 1-5, doi: 10.1190/segam2012-0440.1.

Wang, B., J. Li, J. Luo, Y. Wang, and J. Geng, 2021, Intelligent deblending of seismic data based on U-Net and transfer learning: IEEE Transactions on Geoscience and Remote Sensing, 59, no.10, 8885-8894, doi: 10.1109/TGRS.2020.3048746.

Wang, H., G. Liu, C. E. Hinz, and F. F. C. Snyder, 1989, Attenuation of marine coherent noise: 59th Annual International Meeting, SEG, Expanded Abstracts, 1112-1114, doi: 10.1190/1.1889861.

Wang, K., T. Hu, B. Zhao, and S. Wang, 2023, An unsupervised deep learning method for direct seismic deblending in shot domain: IEEE Transactions on Geoscience and Remote Sensing,

61, 1-12, doi: 10.1109/TGRS.2023.3298054.

Wang, S., W. Hu, Y. Hu, X. Wu, and J. Chen, 2020, A physics-augmented deep learning method for seismic data deblending: Technical Program, SEG, Expanded Abstracts, 3877-3881, doi: 10.1190/segam2020-w13-07.1.

Wang, S., P. Song, J. Tan, B. He, D. Xia, Q. Wang, and G. Du, 2023, Deep learning-based attenuation for shear-wave leakage from ocean-bottom node data: Geophysics, 88, no.3, V127-V137, doi: 10.1190/geo2022-0252.1.

Wang, S., J. Tan, P. Song, B. He, Q. Wang, and G. Du, 2024, Deep learning for noise attenuation from the ocean bottom node 4C data: Geophysical Prospecting, 72, 1400–1413, doi: 10.1111/1365-2478.13336.

Xu, P. C., W. K. Lu, and B. F. Wang, 2019, Automatic source localization and attenuation of seismic interference noise using density-based clustering method: IEEE Transition Geoscience and Remote Sensing, 57, no.7, 4612-4623, doi: 10.1109/TGRS.2019.2891835

Xu, P. C., W. K. Lu, and B. F. Wang, 2020, Seismic interference noise attenuation by convolutional neural network based on training data generation: IEEE Geoscience and Remote Sensing Letters, 18, no. 4, 741-745, doi: 10.1109/LGRS.2020.2982323.

Yang, L., S. Wang, X. Chen, O.M. Saad, W. Cheng, and Y. Chen, 2023, Deep learning with fully convolutional and dense connection framework for ground roll attenuation: Surveys in Geophysics, 44, no.6, pp.1919-1952, doi: 10.1007/s10712-023-09779-8.

Yu, M. C., 2011, Seismic interference noise elimination — A multidomain 3D filtering approach: 81st Annual International Meeting, SEG, Expanded Abstracts, 3591-3595, doi: 10.1190/1.3627946.

Yu, S., and J. Ma, 2021, Deep learning for geophysics: Current and future trends: Reviews of Geophysics, 59, no. 3, p.e2021RG000742, doi: 10.1029/2021RG000742.

Yuan, Y., X. Si, and Y. Zheng, 2020. Ground-roll attenuation using generative adversarial networks: Geophysics, 85, no.4, WA255-WA267, doi: 10.1190/geo2019-0414.1.

Zhang, Z., and P. Wang, 2015, Seismic interference noise attenuation based on sparse inversion: Technical Program, SEG, 4662-4666, doi: 10.1190/segam2015-5877663.1.

Zhu, Z., Z. Chen, B. Wu, and L. Chen, 2025, Self-supervised three-dimensional ocean bottom node seismic data shear wave leakage suppression based on a dual encoder network: Sensors, 25, no.3, p.682, doi: 10.3390/s25030682.

Zu, S., J. Cao, S. Qu, and Y. Chen, 2020, Iterative deblending for simultaneous source data using the deep neural network: Geophysics, 85, no.2, V131-V141, doi: 10.1190/geo2019-0319.1.