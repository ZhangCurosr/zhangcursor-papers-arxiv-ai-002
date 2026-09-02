# MADS: A Multiview Acoustic Descriptor Set Beyond Standard Spectral Summaries

Utsab Ghosh<sup>1\*</sup> and Roshni Chakraborty<sup>1</sup>

<sup>1</sup>ABV-Indian Institute of Information Technology and Management, Gwalior, India.

## Abstract

Dominant audio classification pipelines rely either on compact handcrafted summaries or on fixed time-frequency frontends such as log-mel representations prior to deep modeling. While highly successful, these representations do not explicitly expose the physical dynamics of the underlying sound-generating event. We introduce MADS (Multi-view Acoustic Descriptor Set), a compact 19-dimensional physics-informed descriptor set designed to capture complementary spectral, temporal, mechanical, and stochastic structure in audio signals. Rather than treating sound only as a spectral pattern, MADS encodes properties related to excitation, damping, periodicity, impulsiveness, and structural consistency within a unified multi-view representation.

We evaluate MADS using standard classical machine learning models on ESC-10, ESC-50, and MSoS, and compare it against two conventional handcrafted baselines: a compact 26D MFCC-based baseline and an expanded 38D spectral-summary baseline. Across ESC-10 and ESC-50, MADS achieves the strongest peak results overall, reaching 81.00% and 52.78%, respectively, while using roughly half the dimensionality of the 38D baseline. On MSoS, MADS again delivers the strongest top-end performance, reaching 67.48%. These results establish MADS not merely as a competitive standalone descriptor set, but as the foundational descriptor layer of a broader acoustically grounded representation program for future frame-level and deep-learning-compatible audio modeling.

Keywords: sound classification, environmental sound classification, handcrafted features, audio descriptors, interpretable audio representations

## 1 Introduction

Sound classification is important in applications such as urban monitoring, machine condition awareness, assistive sensing, surveillance, and autonomous systems, where audio can provide situational awareness even when visual information is unreliable. Large-scale pretrained audio models such as PANNs and transformer-based systems have substantially improved sound classification by learning strong representations from time-frequency inputs [1–3]. More recent models such as BEATs and CLAP extend this trend through self-supervised acoustic tokenization and audio-language supervision, respectively [4, 5]. Although these models are highly effective, they remain largely data-driven, i.e., the waveform is converted into a timefrequency representation and processed as a high-dimensional pattern rather than as the output of a physical sounding process. As a result, acoustically distinct events such as wind, steamlike turbulence, and machinery hum can occupy overlapping spectral regions even when their underlying excitation, dissipation, and temporal structure differ substantially.

A parallel line of research has relied on handcrafted descriptors such as Mel-frequency cepstral coefficients (MFCCs) [6] and standard spectral and temporal summary features including spectral centroid, zero-crossing rate, spectral flatness, roll-off, and related statistics [7]. These descriptors remain attractive because they are compact, interpretable, and inexpensive to compute. Hybrid pipelines have also explored combining handcrafted descriptors with learned representations through late fusion of a handcrafted-feature classifier and a deep model [8].

However, standard handcrafted descriptors are primarily designed as compact summaries of spectral shape and local variation, and are therefore not explicitly targeted at resolving eventlevel dynamic differences. In practice, they can describe where energy is concentrated and how the spectrum is shaped, but they are less expressive about how the event evolves over time, how energy is transferred, and how impulsive, damped, periodic, frictional, or mechanically unstable the source behavior may be.

At the same time, modern deep-learning systems for sound classification largely rely on log-mel representations of raw waveforms. Large-scale pretrained systems such as PANNs [1], AST [2], HTS-AT [3], BEATs [4], and related architectures all build on time-frequency representations that are either explicitly mel-based or closely tied to the same frontend assumptions. This design choice has been highly successful in practice, but it is not beyond question. Prior work has pointed out several limitations of the standard log-mel pipeline. The mel scale itself has been revised multiple times [9, 10], and the auditory evidence originally used to support it was later questioned [11]. More broadly, a perceptually motivated frequency scale may not always be ideal for tasks that require different spectral resolution profiles. These observations have motivated a broader line of research on replacing fixed mel-filterbanks with learnable neural frontends, including Gammatone-inspired frontends [12], sinc-based models such as SincNet [13], and more recent learnable frontends such as LEAF [14]. In particular, LEAF showed that replacing standard mel-filterbanks and fixed compression with a learnable frontend can be beneficial in several audio tasks, suggesting that the dominant log-mel pipeline remains an open design space rather than a settled optimum [14]. However, even this line of work largely focuses on improving how spectral frontends are learned, rather than explicitly representing the underlying physical dynamics of the sound-generating event itself.

Motivated by this broader line of research, we propose MADS, a 19-dimensional descriptor set for sound classification. Although audio classifiers observe digital pressure sequences, the underlying events are generated by physical processes such as impacts, oscillations, damping, friction, bursts, and turbulent energy transfer. To explicitly model these dynamics, MADS combines complementary spectral, temporal, mechanical, and stochastic views of an audio sample. By unifying these four distinct perspectives, we construct a multi-view, physicsinformed feature space that provides a compact, interpretable, and structured descriptor basis for representation. In this sense, the present work is foundational: our goal is not merely to validate a standalone compact representation, but to establish a descriptor layer that can support future frame-level and deep-learning-compatible extensions.

We evaluate the proposed 19-dimensional descriptor set on ESC-10 [15], ESC-50 [15], and MSoS [16] using a standard suite of classical machine learning classifiers, and compare it against two conventional handcrafted baselines: a compact 26D MFCC-based baseline and an expanded 38D baseline built from standard spectral summary descriptors. Across ESC-10 and ESC-50, MADS consistently outperforms the compact baseline and achieves the strongest peak results overall, reaching 81.00% on ESC-10 and 52.78% on ESC-50, compared to the best baseline results of 79.45% and 51.78%, respectively, while using substantially fewer features than the 38D baseline. On MSoS, MADS again delivers the strongest top-end performance, reaching 67.48% with the voting ensemble, compared to the best baseline result of 66.04%. Taken together, these results position MADS as both a practically useful standalone descriptor set and a foundational representation for future acoustically grounded audio modeling.

## 2 Proposed Methodology

In this Section, we initially discuss the Problem Definition followed by details of the proposed features, MADS, i.e., Multi-View Acoustic Descriptor Set, in detail.

## 2.1 Problem Statement

Environmental sounds are often represented through spectral summaries that capture coarse timbre and energy distribution, but not necessarily the physical dynamics of the underlying source event [17]. As a result, different sound-producing mechanisms can exhibit similar spectral structure even when their excitation, dissipation, periodicity, or impulsiveness differ substantially.

We therefore ask whether a compact set of acoustically grounded descriptors can recover some of this missing physical structure and provide a stronger, more interpretable basis for environmental sound classification. Let $x [ n ]$ denote an input waveform and $y \in \mathcal { V }$ its class label. While conventional front-ends construct a spectral representation $\phi _ { \mathrm { s p e c } } ( x )$ , our objective is to define a compact descriptor map

$$
\phi _ { \mathrm { d y n } } ( x ) \in \mathbb { R } ^ { 1 9 } ,
$$

that preserves physically meaningful structure and provides an effective basis for predicting y.

## 2.2 Multi-View Acoustic Descriptor Set, MADS

While a digital audio signal is a dimensionless sequence of pressure values, it is the product of a complex dynamical system involving a physical radiator, a propagation medium, and stochastic energy exchange. To bridge the discriminative gap inherent in standard spectral summaries, we use four different lenses to look at the audio:

1. Mechanical Descriptors : We use this term to denote descriptors derived from discrete velocity, acceleration, and energy, designed to characterize coupling, dissipation, and instability directly from the waveform. In this group we include Kinetic Energy Entropy, Energy Phase Correlation, Mechanical Efficiency, and Mechanical Instability Index.

2. Spectral Anchors : These descriptors summarize the coarse spectral envelope, dominant resonance, and effective bandwidth of the signal using standard frequency-domain statistics [6, 7]. In this group we include Spectral Anchor I–III, Dissipation Limit, Resonant Spread, and Peak Mode.

3. Temporal Dynamics : These descriptors represent onset behavior, temporal roughness, and the density of salient excitation events over time [7]. In this group we include Attack Gradient, Damping Coefficient, Temporal Jitter, and Pulse Density.

4. Stochasticity & Consistency : These descriptors quantify how ordered, irregular, broadband, or periodically structured the signal is using flatness, kurtosis, zero-crossing variation, autocorrelation, and residual consistency measures [7]. In this group we include entropyflow, Impact Peakedness, ZCR Variability, Periodicity Strength, and PDE Residual.

Acoustic Mechanics Foundations To construct mechanically grounded descriptors from the waveform, we introduce a small set of discrete quantities. Let x[n] denote the normalized audio waveform. We define an instantaneous velocity as the first-order difference difference as shown in Equation 1 and acceleration as the second-order difference as shown in in Equation 2 next

$$
v [ n ] = x [ n ] - x [ n - 1 ] .\tag{1}
$$

$$
a [ n ] = v [ n ] - v [ n - 1 ] .\tag{2}
$$

These quantities are used as discrete proxies that allow us to characterize effort, dissipation, and temporal energy transfer directly from the signal. We further define an effective mass term $m _ { \mathrm { e f f } }$ from the spectral centroid trajectory[7]. Let

$$
c [ t ] = \frac { \sum _ { k } f _ { k } | X _ { k } ( t ) | } { \sum _ { k } | X _ { k } ( t ) | }\tag{3}
$$

denote the spectral centroid at frame t, i.e., the spectral center of gravity of the spectrum [18]. Motivated by this center-of-gravity interpretation, we use the centroid trajectory to construct an effective mass proxy. We then define

$$
m _ { \mathrm { e f f } } = \frac { \mu _ { c } + \sigma _ { c } } { f _ { s } } ,\tag{4}
$$

where, $\mu _ { c }$ and $\sigma _ { c }$ are the mean and standard deviation of the centroid trajectory, and $f _ { s }$ is the sampling rate. This term acts as a heuristic inertia-like scaling factor for the discrete mechanics used below. Using $m _ { \mathrm { e f f } }$ , we define three foundational energetic quantities as follows

in Equation 5,6, 7, which serve as the basis for the mechanical descriptors, discussed next.

$$
E _ { k } [ n ] = { \frac { 1 } { 2 } } m _ { \mathrm { e f f } } v [ n ] ^ { 2 }\tag{5}
$$

$$
E _ { p } [ n ] = { \frac { 1 } { 2 } } m _ { \mathrm { e f f } } x [ n ] ^ { 2 }\tag{6}
$$

$$
W [ n ] = \left( m _ { \mathrm { e f f } } a [ n ] \right) v [ n ]\tag{7}
$$

These serve as the foundational values for Mechanical Energetic Descriptors

## 2.2.1 Mechanical Energetics

These descriptors model the waveform as a discrete mechanical system and quantify how energy is distributed, coupled, and dissipated over time.

• Kinetic Energy Entropy $( H _ { k e } ) \colon$ This descriptor measures how concentrated or distributed the kinetic energy is over time. Sounds such as mouse click, can opening, and water drops tend to produce a few dominant energetic events and therefore lower entropy, whereas vacuum cleaner, washing machine, and siren exhibit more uniformly distributed energy and therefore higher entropy. We compute it as the Shannon entropy [19] of the normalized kinetic energy sequence as shown next:

$$
H _ { k e } = - \sum _ { n } \hat { E } _ { k } [ n ] \log _ { 2 } ( \hat { E } _ { k } [ n ] + \epsilon ) ,\tag{8}
$$

where $\hat { E } _ { k } [ n ]$ denotes the normalized kinetic energy at sample n, and ϵ is a small constant for numerical stability.

$$
\hat { E } _ { k } [ n ] = \frac { E _ { k } [ n ] + \epsilon } { \sum _ { n } \left( E _ { k } [ n ] + \epsilon \right) } .\tag{9}
$$

• Energy Phase Correlation $( \rho _ { k p } ) \colon$ This descriptor measures the degree of coupling between kinetic and potential energy states. Higher values arise in impulsive or sharply structured events such as mouse click, can opening, and glass breaking, where multiple energy components peak together, while lower values are more characteristic of smoother or sustained sources such as siren, wind, and engine. We compute it as the Pearson correlation [20] between the kinetic and potential energy sequences:

$$
\rho _ { k p } = \frac { \mathrm { c o v } ( \tilde { E } _ { k } , \tilde { E } _ { p } ) } { \sigma _ { \tilde { E } _ { k } } \sigma _ { \tilde { E } _ { p } } } .\tag{10}
$$

Here, $\tilde { E } _ { k }$ and ${ \tilde { E } } _ { p }$ denote the kinetic and potential energy sequences truncated to a common length, σ <sub>˜</sub> and σ <sub>˜</sub> denote the standard deviations of the truncated kinetic and potential $\sigma _ { \tilde { E } _ { k } }$ $\sigma _ { \tilde { E } _ { \varepsilon } }$ energy sequences, respectively. If this correlation is numerically undefined, it is set to 0.

• Mechanical Efficiency (η): This descriptor quantifies how much kinetic energy is retained relative to the mean mechanical work. Signals such as siren, thunderstorm, and airplane exhibit higher values because they retain energy in a relatively stable or sustained state, whereas can opening, crackling fire, and glass breaking exhibit lower values because energy is dissipated more irregularly. We define it as a log-scaled ratio between mean kinetic energy and the absolute mean work:

$$
\eta = \log \biggl ( 1 + \left| \frac { \mathbb { E } [ E _ { k } [ n ] ] } { | \mathbb { E } [ W [ n ] ] | + \epsilon } \right| \biggr ) .\tag{11}
$$

Here, $\mathbb { E } [ E _ { k } [ n ] ]$ is the mean kinetic energy over the clip and $| \mathbb { E } [ W [ n ] ] |$ denotes the absolute value of the meanof the work sequence over the clip. and ϵ is a small constant for numerical stability.

• Mechanical Instability Index (Ψ): This descriptor measures higher-order instability in the signal by combining acceleration variation and work variability. It is higher for mechanically irregular or intermittent sounds such as clock alarm, can opening, glass breaking, and crickets, and lower for smoother or more stable sources such as cow, church bells, and siren. We define it as

$$
\Psi = \log \biggl ( 1 + \left| \frac { \mathrm { v a r } ( \Delta a ) \sigma _ { W } } { m _ { \mathrm { e f f } } \mu _ { | v | } + \epsilon } \right| \biggr ) ,\tag{12}
$$

where $\mathrm { v a r } ( \Delta a )$ is the variance of the first difference of acceleration, $\sigma _ { W }$ is the standard deviation of the work sequence, $m _ { \mathrm { e f f } }$ is the effective mass term, $\mu _ { | v | }$ is the mean absolute velocity, and ϵ is a small constant for numerical stability.

## 2.2.2 Spectral Anchors

These descriptors capture coarse timbral structure and the frequency-domain footprint of the source.They provide stable spectral anchors for resonance shape, dominant frequency placement, and effective bandwidth, building on standard audio feature families [7]. Let $S ( k , t )$ denote the magnitude spectrogram at frequency bin k and frame $t ,$ let $f _ { k }$ denote the center frequency of bin $k ,$ let $P [ k ]$ denote the power spectral density, and let c[t] denote the spectral centroid at frame t.

• Spectral Anchors (Spectral Anchor I-III): We include the first three MFCC coefficients as standard low-dimensional summaries of the coarse spectral envelope [6]. In our descriptor set, they serve as compact timbral anchors alongside the proposed acoustically grounded features. Empirically, these three anchors provide strong coarse separation for classes such as vacuum cleaner, thunderstorm, airplane, church bells, and can opening.

• Peak Mode $( f _ { \mathrm { m o d e } } ) { : }$ This descriptor captures the principal resonant or dominant frequency mode of the source. Lower values are typical of lower-frequency or broadband events such as footsteps, engine, helicopter, and thunderstorm, while higher values appear in classes with sharper or higher-frequency dominant modes such as clock alarm, crying baby, rooster, and frog. We define it from the dominant spectral peak of the power spectral density as

$$
f _ { \mathrm { m o d e } } = \log \left( 1 + \left| f _ { \mathrm { a r g m a x } _ { k } P [ k ] } \right| \right) .\tag{13}
$$

Here, $P [ k ]$ denotes the power spectral density at frequency bin $k ,$ arg max<sub>k</sub> $P [ k ]$ is the index of the dominant spectral peak, $f _ { k }$ is the center frequency of bin $k ,$ , and the outer $\log ( 1 + | \cdot | )$ term applies log compression.

• Dissipation Limit $( \Phi _ { \mathrm { d l } } ) { \mathrm { : } }$ This descriptor estimates the upper extent of the effective energy bandwidth. Low $\Phi _ { \mathrm { d l } }$ indicates energy concentrated in lower-frequency structure, as in thunderstorm, wind, airplane, and door knock. High $\Phi _ { \mathrm { d l } }$ indicates broad high-frequency support, as in brushing teeth, pouring water, rain, and crackling fire. We compute it as the mean 85% spectral rolloff frequency:

$$
\Phi _ { \mathrm { d l } } = \mathbb { E } _ { t } [ \mathrm { R o l l o f f } _ { 8 5 \% } ( t ) ] .\tag{14}
$$

Here, $\mathrm { R o l l o f f } _ { 8 5 \% } ( t )$ is the spectral rolloff at frame t, i.e., the frequency below which 85% of the spectral energy is accumulated, and $\mathbb { E } _ { t } [ \cdot ]$ denotes the average over time frames.

• Resonant Spread $( \Phi _ { \mathrm { s p r e a d } } ) { : }$ This descriptor measures how concentrated or dispersed the spectral footprint is around its centroid. Low $\Phi _ { \mathrm { s p r e a d } }$ is associated with relatively focused spectral structure, as in church bells, siren, airplane, and door knock. High $\Phi _ { \mathrm { s p r e a d } }$ is associated with spectrally scattered or broadband events such as cracklingfire, brushing teeth, rain, and insects. We compute it as the mean RMS spectral deviation from the centroid:

$$
\Phi _ { \mathrm { s p r e a d } } = \mathbb { E } _ { t } \left[ \sqrt { \frac { \sum _ { k } ( f _ { k } - c [ t ] ) ^ { 2 } | S ( k , t ) | } { \sum _ { k } | S ( k , t ) | + \epsilon } } \right] .\tag{15}
$$

Here, $S ( k , t )$ denotes the magnitude spectrogram at frequency bin k and frame $t , f _ { k }$ is the center frequency of bin $k , c [ t ]$ is the spectral centroid at frame $t , \mathbb { E } _ { t } [ \cdot ]$ denotes averaging over time frames, and ϵ is a small constant for numerical stability.

## 2.2.3 Temporal Dynamics

These descriptors characterize event-scale temporal behavior, focusing on onset sharpness, dissipation regularity, temporal roughness, and the density of salient excitation events. Let $o [ t ]$ denote the onset-strength envelope at frame t, let $o _ { \mathrm { s e g } } [ t ]$ denote the same envelope restricted to the initial attack segment, and let T denote the clip duration.

• Attack Gradient $( \nabla _ { \mathrm { a t t } } )$ : This descriptor captures how the onset envelope evolves immediately after excitation. More negative values correspond to abrupt onset spikes followed by rapid decay, as in can opening, door knock, and glass breaking. Higher values correspond to onset segments that continue to rise or remain active over the initial window, as in dog, sneezing, and mouse click. We compute it as the mean rate of change of the onset-strength envelope over the initial 50 ms onset segment:

$$
\nabla _ { \mathrm { a t t } } = \mathbb { E } _ { t } [ \nabla o _ { \mathrm { s e g } } [ t ] ] .\tag{16}
$$

Here, $o _ { \mathrm { s e g } } [ t ]$ denotes the segment of the onset-strength envelope beginning at the first frame satisfying $o [ t ] > 0 . 0 5 \operatorname* { m a x } _ { \tau } o [ \tau ]$ and extending for the next $\lfloor 0 . 0 5 f _ { s } / 5 1 2 \rfloor$ frames, clipped to the available signal length, and ∇ denotes the discrete gradient operator applied to that segment.

• Damping Coefficient $( \zeta ) { \vdots }$ This descriptor measures the regularity of onset behavior over the full clip. Low $\zeta$ corresponds to more regular or stable onset behavior, as in wind, airplane, sea waves, and vacuum cleaner. High ζ indicates irregular or strongly varying excitation and dissipation, as in glass breaking, door knock, sneezing, coughing, and can opening. We compute it as the coefficient of variation of onset strength:

$$
\zeta = \frac { \sigma _ { o } } { \mu _ { o } + \epsilon } ,\tag{17}
$$

where $\mu _ { o } = \mathbb { E } _ { t } [ o [ t ] ]$ is the mean onset strength over time, $\sigma _ { o } = \mathrm { s t d } _ { t } ( o [ t ] )$ is its standard deviation, and ϵ is a small constant for numerical stability.

• Temporal Jitter $( \Phi _ { \mathrm { j i t t e r } } ) { : }$ : This descriptor measures temporal roughness in the evolution of the onset envelope. Low values indicate smoother energy flow, as in wind, airplane, and sea waves, whereas high values indicate rapidly fluctuating or jagged temporal structure, as in keyboard typing, fireworks, mouse click, and can opening. We compute it as the standard deviation of the first difference of the onset-strength sequence:

$$
\Phi _ { \mathrm { j i t t e r } } = \mathrm { s t d } _ { t } ( \Delta o [ t ] ) .\tag{18}
$$

Here, $o [ t ]$ denotes the onset-strength envelope at frame $t , \Delta o [ t ]$ is its first-order difference, and $\mathrm { s t d } _ { t } ( \cdot )$ denotes the standard deviation across time frames.

• Pulse Density $( \Phi _ { \mathrm { p u l s e } } ) { : }$ This descriptor measures the density of strong salient peaks within the clip. High $\Phi _ { \mathrm { p u l s e } }$ indicates a large number of strong, repeatedly occurring peaks, as in clock alarm, vacuum cleaner, washing machine, and crickets. Low $\Phi _ { \mathrm { p u l s e } }$ is more typical of clips containing fewer strong peaks, such as water drops, drinking/sipping, and footsteps. We define it as

$$
\Phi _ { \mathrm { p u l s e } } = \frac { N _ { \mathrm { p e a k s } } } { T } ,\tag{19}
$$

where $x [ n ]$ is the normalized audio waveform at sample $n , N _ { \mathrm { p e a k s } }$ is the number of peaks detected in $| x [ n ] |$ with peak height greater than $0 . 3 \operatorname* { m a x } _ { n } \left| x [ n ] \right|$ |, and T is the clip duration.

## 2.2.4 Stochasticity and Structural Consistency

These descriptors measure how ordered, irregular, or periodically structured the signal is over time. Let $S ( k , t )$ denote the magnitude spectrogram of the signal at frequency bin k and frame t, let $\mathrm { Z C R } [ t ]$ denote the zero-crossing rate of frame t, and let ℓ denote the autocorrelation lag.

• Entropy Flow $( \Phi _ { \mathrm { e f } } )$ : This descriptor measures the average spectral flatness over time [7]. Low $\Phi _ { \mathrm { e f } }$ indicates more coherent or spectrally structured energy, as in church bells, airplane, wind, and train. High $\Phi _ { \mathrm { e f } }$ indicates flatter, more broadband, and spectrally disordered events, as in rooster, sneezing, coughing, and can opening. We compute it as the mean spectral flatness across frames:

$$
\Phi _ { \mathrm { e f } } = \mathbb { E } _ { t } [ \mathrm { F l a t n e s s } ( S ( : , t ) ) ] .\tag{20}
$$

Here, $S ( : , t )$ denotes the spectral slice of the magnitude spectrogram at frame t, Flatness(·) denotes the spectral flatness measure, and $\mathbb { E } _ { t } [ \cdot ]$ denotes averaging over time frames.

• Impact Peakedness $( \kappa _ { s } ) \mathrm { : }$ : This descriptor measures how strongly energy concentrates into sharp spectral peaks rather than remaining diffusely distributed. Higher values are associated with signals that contain persistent or pronounced spectral spikes, as in siren, snoring, clock alarm, and footsteps, whereas lower values are more typical of flatter or more diffuse spectral structure, as in sea waves, clapping, rain, and hand saw. We compute it as the mean spectral kurtosis across frames:

$$
\kappa _ { s } = \mathbb { E } _ { t } [ \mathrm { K u r t o s i s } _ { F } ( S ( : , t ) ) ] .\tag{21}
$$

Here, $S ( : , t )$ denotes the spectral slice at frame $t ,$ Kurtosis (·) is the Fisher kurtosis of that slice, and $\mathbb { E } _ { t } [ \cdot ]$ denotes averaging over time frames.

• ZCR Variability $( \sigma _ { \mathrm { z c r } } ) $ : This descriptor measures the variation of zero-crossing behavior over time [7]. Low $\sigma _ { \mathrm { z c r } }$ indicates more stable fine-scale oscillatory behavior, as in wind, thunderstorm, helicopter, and airplane. High $\sigma _ { \mathrm { z c r } }$ indicates stronger variation in shorttimescale sign changes, as in glass breaking, can opening, pouring water, and sneezing. We compute it as the standard deviation of the zero-crossing rate across frames:

$$
\sigma _ { \mathrm { z c r } } = { \mathrm { s t d } } _ { t } ( { \mathrm { Z C R } } [ t ] ) .\tag{22}
$$

Here, ZCR[t] denotes the zero-crossing rate at frame t, and $\mathrm { s t d } _ { t } ( \cdot )$ denotes the standard deviation across time frames.

• Periodicity Strength $( \rho _ { \mathrm { m a x } } )$ : This descriptor measures how strongly repeating structure is expressed in the signal [7]. High $\rho _ { \mathrm { m a x } }$ indicates strong periodicity, as in clock alarm, car horn, church bells, and engine. Low $\rho _ { \mathrm { m a x } }$ is characteristic of weakly periodic or aperiodic events such as sea waves, clapping, rain, and can opening. We compute it as the maximum normalized autocorrelation excluding the first 50-sample lag:

$$
\rho _ { \mathrm { m a x } } = \operatorname* { m a x } _ { 5 0 \leq \ell < L } \tilde { r } _ { x } [ \ell ] ,\tag{23}
$$

where $\begin{array} { r } { { \tilde { r } } _ { x } [ \ell ] = \frac { r _ { x } [ \ell ] } { r _ { x } [ 0 ] + } } \end{array}$ 1 is the normalized autocorrelation at lag ℓ, L is the maximum available lag after truncation to 2000 samples, and ϵ is a small constant for numerical stability.

• PDE Residual $( { \mathcal { R } } _ { \mathrm { p d e } } )$ : This descriptor measures the mismatch between temporal acceleration structure and a coarse spectral-curvature estimate. Let a[n] denote the discrete second-order temporal difference of the waveform and let $H _ { \mathrm { s p e c } }$ denote the accumulated local spectral curvature around dominant spectral peaks. Low $\mathcal { R } _ { \mathrm { p d e } }$ indicates signals whose temporal and spectral structure evolve in a comparatively coherent manner, as in thunderstorm, cow, door knock, and siren. High $\mathcal { R } _ { \mathrm { p d e } }$ indicates stronger mismatch between these two views, as in mouse click, can opening, crickets, and clock alarm. We define it as

$$
H _ { \mathrm { s p e c } } = \sum _ { p \in \mathcal { P } } \left| P [ p + 1 ] - 2 P [ p ] + P [ p - 1 ] \right| ,\tag{24}
$$

where $\mathcal { P }$ is the set of up to ten strongest interior peaks of the power spectral density $P [ k ]$ detected with minimum peak spacing 50 bins and minimum prominence 0.01 max $P [ k ]$

$$
\mathcal { R } _ { \mathrm { p d e } } = \frac { \left| \mathbb { E } [ a [ n ] ^ { 2 } ] - \frac { H _ { \mathrm { s p e c } } } { f _ { s } ^ { 2 } } \right| } { \operatorname { V a r } ( x [ n ] ) + \epsilon } .\tag{25}
$$

Here, $a [ n ]$ is the discrete second-order temporal difference of the waveform, $H _ { \mathrm { s p e c } }$ is the accumulated local spectral-curvature term defined above, $f _ { s }$ is the sampling rate, Var(x[n]) is the variance of the waveform, and ϵ is a small constant for numerical stability.

## 3 Experiments and Results

## 3.1 Datasets

We evaluate MADS on two sound classification benchmarks: the ESC-50 dataset and its ESC-10 subset [15], and the MSoS benchmark [16]. Together, these benchmarks allow us to assess the proposed descriptor set across environmental and broader sound classification settings.

• ESC-50 / ESC-10: ESC-50 contains 2,000 environmental audio clips of 5 seconds each, evenly distributed across 50 classes and organized into five official cross-validation folds. ESC-10 is a 10-class subset of the same corpus and serves as a simpler benchmark for environmental sound classification.

• MSoS: MSoS (Making Sense of Sounds) is a 5-class sound classification benchmark comprising the high-level categories Effects, Human, Music, Nature, and Urban. We evaluate on MSoS using its official train/test split.

## 3.2 Feature Extraction and Preprocessing

For implementation, all extracted features are sanitized by replacing NaN and ±∞ values with $0 ,$ and then clipped to $[ - 1 0 ^ { 6 } , 1 0 ^ { 6 } ]$ . In our implementation, $\epsilon = 1 0 ^ { - 1 0 }$ . For all classical ML experiments, features are z-score standardized using StandardScaler fitted on the corresponding training split.

## 3.3 Evaluation Protocol and Classifiers

To evaluate the discriminative utility of the proposed descriptors, we perform sound classification using a standard suite of classical machine learning models. Across all benchmarks, we use 1-NN and 3-NN [21], logistic regression [22], SVM with an RBF kernel [23], random forest [24], Gaussian Naive Bayes [25], XGBoost [26], and a soft-voting ensemble [27]. In Tables 1, 2, and 3, we report mean classification accuracy and standard deviation over five random seeds (42–46).

We compare MADS against two conventional handcrafted baselines: $B _ { 1 }$ , a compact 26D baseline following the ESC setup of Piczak [15], and $B _ { 2 }$ , an expanded 38D conventional baseline built from standard MFCC and spectral summary descriptors [6, 7]. $B _ { 1 }$ consists of the mean and standard deviation of zero-crossing rate together with the mean and standard deviation of MFCC coefficients 1:12, i.e., the $0 ^ { \mathrm { t h } }$ MFCC coefficient is discarded. $B _ { 2 }$ expands this setup with a broader set of standard spectral statistics, including MFCC coefficients 0:12, zero-crossing rate, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, and

![](images/975642c251a41a6b4ee46055ab3c087fdbf9282e2694c2600e0dfd34e5902f23.jpg)  
Fig. 1: XGBoost feature-importance curve (gain) for MADS on MSoS. The importance profile spans multiple descriptor families, indicating that the classifier uses complementary temporal, spectral, mechanical, and stochastic cues.

RMS energy. This yields $1 3 \times 2 = 2 6$ MFCC statistics and $6 \times 2 = 1 2$ additional summary statistics, for a total of 38 features.

For ESC-10 and ESC-50, we follow the official 5-fold cross-validation protocol independently for each seed, and then report the mean and standard deviation across the five resulting accuracies. For MSoS, we use the official train/test split and report the mean and standard deviation of test accuracy across the same five seeds.

## 3.4 Results and discussion

Tables 1, 2, and 3 show a consistent pattern. First, MADS substantially outperforms the compact baseline $B _ { 1 }$ across the strongest classifiers on all three benchmarks. On ESC-10, MADS achieves the best ensemble accuracy (81.00%) and the best XGBoost accuracy (79.60%), while also improving over $B _ { 1 }$ for Logistic Regression, Naive Bayes, and both k-NN variants. On ESC-50, MADS again delivers the strongest top-end performance, achieving the best ensemble accuracy (52.78%), the best Random Forest accuracy (49.33%), and the best XGBoost accuracy (50.13%). On MSoS, MADS likewise achieves the strongest top-end results, reaching the best voting ensemble accuracy (67.48%), the best Random Forest accuracy (67.20%), and the best XGBoost accuracy (67.28%). Second, despite using only 19 features, MADS remains highly competitive with the larger conventional baseline $B _ { 2 }$ (38D). On ESC-10, it outperforms $B _ { 2 }$ on the ensemble, XGBoost, Logistic Regression, 3-NN, and Naive Bayes; on ESC-50, it remains superior on the best-performing models overall; and on MSoS, it exceeds $B _ { 2 }$ on the strongest ensemble and tree-based models while using roughly half the dimensionality.

Table 1: ESC-10 5-fold cross-validation comparison. Values are mean accuracy (%) over 5 seeds (42–46).
<table><tr><td>Model</td><td> $B 1 \ ( 2 6 \mathrm { D } )$ </td><td> $B _ { 2 } \ ( 3 8 \mathrm { D } )$ </td><td> $\mathrm { O u r s } \left( 1 9 \mathrm { D } \right)$ </td></tr><tr><td>Voting (RF+XGB+LR)</td><td> $7 4 . 0 0 \pm 0 . 5 0$ </td><td> $7 9 . 0 5 \pm 0 . 5 7$ </td><td> ${ \bf 8 1 . 0 0 \pm 0 . 2 5 }$ </td></tr><tr><td>Random Forest</td><td> $7 2 . 4 5 \pm 0 . 8 4$ </td><td> $\mathbf { 7 9 . 4 5 \pm 0 . 4 1 }$ </td><td> $7 8 . 7 0 \pm 0 . 6 2$ </td></tr><tr><td>XGBoost</td><td> $7 0 . 9 5 \pm 0 . 8 2$ </td><td> $7 9 . 2 0 \pm 0 . 3 7$ </td><td> ${ \bf 7 9 . 6 0 \pm 0 . 5 2 }$ </td></tr><tr><td>SVM (RBF)</td><td> $7 0 . 2 5 \pm 0 . 0 0$ </td><td> $7 7 . 2 5 \pm 0 . 0 0$ </td><td> $7 3 . 2 5 \pm 0 . 0 0$ </td></tr><tr><td>Logistic Regression</td><td> $6 8 . 2 5 \pm 0 . 0 0$ </td><td> $7 2 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 7 5 . 5 0 \pm 0 . 0 0 }$ </td></tr><tr><td>3-NN</td><td> $6 5 . 5 0 \pm 0 . 0 0$ </td><td> $6 8 . 2 5 \pm 0 . 0 0$ </td><td> ${ \bf 6 8 . 5 0 \pm 0 . 0 0 }$ </td></tr><tr><td>Naive Bayes</td><td> $6 5 . 5 0 \pm 0 . 0 0$ </td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $\mathbf { 7 2 . 2 5 \pm 0 . 0 0 }$ </td></tr><tr><td>1-NN</td><td> $6 4 . 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 6 9 . 0 0 \pm 0 . 0 0 }$ </td><td> $6 6 . 2 5 \pm 0 . 0 0$ </td></tr></table>

Table 2: ESC-50 5-fold cross-validation comparison. Values are mean accuracy (%) over 5 seeds (42–46).
<table><tr><td>Model</td><td> $B _ { 1 } \ ( 2 6 \mathrm { D } )$ </td><td> $B _ { 2 } \ ( 3 8 \mathrm { D } )$ </td><td> $\mathrm { O u r s } \left( 1 9 \mathrm { D } \right)$ </td></tr><tr><td>Voting (RF+XGB+LR)</td><td> $4 2 . 7 3 \pm 0 . 2 4$ </td><td> $5 1 . 7 8 \pm 0 . 2 1$ </td><td> ${ \pm 2 . 7 8 \pm 0 . 3 3 }$ </td></tr><tr><td>SVM (RBF)</td><td> $4 1 . 7 5 \pm 0 . 0 0$ </td><td> $\mathbf { 4 7 . 5 0 \pm 0 . 0 0 }$ </td><td> $4 1 . 9 5 \pm 0 . 0 0$ </td></tr><tr><td>Random Forest</td><td> $4 1 . 5 9 \pm 0 . 6 0$ </td><td> $4 7 . 2 0 \pm 0 . 3 1$ </td><td> ${ \pm 9 . 3 3 \pm 0 . 6 2 }$ </td></tr><tr><td>XGBoost</td><td> $3 9 . 4 7 \pm 0 . 4 5$ </td><td> $4 6 . 1 6 \pm 0 . 3 4$ </td><td> ${ \bf 5 0 . 1 3 \pm 0 . 1 9 }$ </td></tr><tr><td>Logistic Regression</td><td> $3 9 . 3 5 \pm 0 . 0 0$ </td><td> ${ \bf 4 6 . 4 0 \pm 0 . 0 0 }$ </td><td> $4 5 . 2 5 \pm 0 . 0 0$ </td></tr><tr><td>Naive Bayes</td><td> $3 3 . 5 0 \pm 0 . 0 0$ </td><td> ${ \bf 3 5 . 6 5 \pm 0 . 0 0 }$ </td><td> $3 3 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>1-NN</td><td> $3 3 . 0 5 \pm 0 . 0 0$ </td><td> ${ \bf 3 5 . 6 5 \pm 0 . 0 0 }$ </td><td> $3 4 . 1 5 \pm 0 . 0 0$ </td></tr><tr><td>3-NN</td><td> $3 2 . 1 5 \pm 0 . 0 0$ </td><td> ${ \bf 3 5 . 6 5 \pm 0 . 0 0 }$ </td><td> $3 1 . 6 0 \pm 0 . 0 0$ </td></tr></table>

A further view of the proposed representation is provided by the XGBoost featureimportance analysis on MSoS, shown in Fig. 1. The learned importance profile is not concentrated on a single descriptor family; instead, high-gain features are drawn from temporal, spectral, mechanical, and stochastic components of MADS. In particular, descriptors such as damping coefficient, spectral anchor i, kinetic energy entropy, and entropy flow emerge among the most influential variables. This suggests that the classifier is exploiting the intended multi-view structure of MADS, rather than relying only on a narrow subset of conventional spectral cues.

## 4 Conclusions

In this work, we introduced MADS, a compact 19-dimensional physics-informed descriptor set for sound classification. MADS was designed to capture complementary spectral, temporal, mechanical, and stochastic properties of an audio signal within a unified multi-view representation, motivated by the hypothesis that acoustically grounded structure can provide a stronger and more interpretable basis for recognition than conventional low-order spectral summaries alone.

Table 3: MSoS classification comparison. Values are mean accuracy (%) over 5 seeds (42–46) evaluated on the official test set.
<table><tr><td>Model</td><td> $B 1 \ ( 2 6 \mathrm { D } )$ </td><td> $B _ { 2 } \ ( 3 8 \mathrm { D } )$ </td><td> $\mathrm { O u r s } \left( 1 9 \mathrm { D } \right)$ </td></tr><tr><td>Voting (RF+XGB+LR)</td><td> $6 0 . 8 0 \pm 1 . 0 0$ </td><td> $6 6 . 0 4 \pm 0 . 4 1$ </td><td> ${ \bf 6 7 . 4 8 \pm 0 . 4 1 }$ </td></tr><tr><td>Random Forest</td><td> $6 0 . 8 8 \pm 0 . 7 3$ </td><td> $6 3 . 1 2 \pm 0 . 9 5$ </td><td> ${ \bf 6 7 . 2 0 \pm 0 . 6 6 }$ </td></tr><tr><td>XGBoost</td><td> $6 1 . 8 4 \pm 0 . 8 2$ </td><td> $6 6 . 8 4 \pm 0 . 4 1$ </td><td> ${ \bf 6 7 . 2 8 \pm 1 . 1 4 }$ </td></tr><tr><td>Logistic Regression</td><td> $4 8 . 4 0 \pm 0 . 0 0$ </td><td> ${ \pm } 3 . 6 0 \pm 0 . 0 0$ </td><td> $5 1 . 6 0 \pm 0 . 0 0$ </td></tr><tr><td>Naive Bayes</td><td> ${ \bf 4 5 . 6 0 \pm 0 . 0 0 }$ </td><td> $4 4 . 4 0 \pm 0 . 0 0$ </td><td> $4 2 . 8 0 \pm 0 . 0 0$ </td></tr><tr><td>1-NN</td><td> $5 9 . 8 0 \pm 0 . 0 0$ </td><td> ${ \bf 6 5 . 8 0 \pm 0 . 0 0 }$ </td><td> $6 2 . 2 0 \pm 0 . 0 0$ </td></tr><tr><td>3-NN</td><td> $5 8 . 8 0 \pm 0 . 0 0$ </td><td> ${ \bf 6 3 . 0 0 \pm 0 . 0 0 }$ </td><td> $6 0 . 4 0 \pm 0 . 0 0$ </td></tr></table>

Across ESC-10, ESC-50, and MSoS, the proposed descriptors consistently outperformed the compact $B _ { 1 }$ handcrafted baseline and achieved the strongest peak results overall on the bestperforming models, while remaining competitive with a substantially larger $B _ { 2 }$ conventional feature set despite using only 19 dimensions. These results show that carefully designed acoustically grounded descriptors can recover discriminative structure that is not well captured by standard spectral summary features alone.

More broadly, we view the present work as foundational rather than terminal. Beyond its value as a compact standalone representation, MADS defines a structured descriptor layer that can support future frame-level and deep-learning-compatible extensions. Exploring such temporal lifts of the descriptor set, and studying how acoustically grounded representations can interact with modern neural audio models, remains an important direction for follow-up work.

## 4.1 Future Directions

While the present work focuses on validating MADS as a compact 19-dimensional descriptor set, the proposed representation is not intended to terminate at a clip-level summary. Many of the descriptors introduced here are computed from underlying temporal trajectories, spectral evolutions, and mechanically motivated intermediate quantities. MADS should therefore be understood as the first compact layer of a richer acoustically grounded representation space.

A natural next step is to lift these descriptors from clip-level aggregates into frame-level or segment-level tracks, yielding a structured temporal representation in which each channel corresponds to a physically meaningful acoustic view, such as excitation strength, damping behavior, periodicity, stochasticity, or spectral anchoring. Such a representation would differ fundamentally from conventional log-mel inputs: rather than exposing only a generic timefrequency energy pattern, it would explicitly organize the signal according to interpretable acoustic dynamics.

This direction opens a path toward deep-learning-compatible extensions of MADS, including sequence models and hybrid architectures that operate on structured concept channels rather than only on fixed spectral frontends. In this sense, the present paper lays the descriptor foundation for a broader representation program: one that seeks to combine compact acoustically grounded structure, interpretability, and compatibility with modern temporal modeling.

# Declaration of Competing Interest

The authors have no relevant financial or non-financial interests to disclose.

## References

[1] Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., Plumbley, M.D.: Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28, 2880–2894 (2020)

[2] Gong, Y., Chung, Y.-A., Glass, J.: Ast: Audio spectrogram transformer. arXiv preprint arXiv:2104.01778 (2021)

[3] Chen, K., Du, X., Zhu, B., Ma, Z., Berg-Kirkpatrick, T., Dubnov, S.: Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection. In: ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 646–650 (2022). IEEE

[4] Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., Wei, F.: Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058 (2022)

[5] Elizalde, B., Deshmukh, S., Al Ismail, M., Wang, H.: Clap: Learning audio concepts from natural language supervision. arXiv preprint arXiv:2206.04769 (2022) https://doi. org/10.48550/arXiv.2206.04769

[6] Davis, S., Mermelstein, P.: Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE transactions on acoustics, speech, and signal processing 28(4), 357–366 (1980)

[7] Peeters, G., et al.: A large set of audio features for sound description (similarity and classification) in the cuidado project. CUIDADO Ist Project Report 54(0), 1–25 (2004)

[8] Fonseca, E., Gong, R., Serra, X.: A simple fusion of deep and shallow learning for acoustic scene classification. arXiv preprint arXiv:1806.07506 (2018)

[9] O’Shaughnessy, D.: Speech Communication: Human and Machine. Addison-Wesley, Reading, MA (1987)

[10] Umesh, S., Cohen, L., Nelson, D.: Fitting the mel scale. In: 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No. 99CH36258), vol. 1, pp. 217–220 (1999). IEEE

[11] Greenwood, D.D.: The mel scale’s disqualifying bias and a consistency of pitch-difference equisections in 1956 with equal cochlear distances and equal frequency ratios. Hearing research 103(1-2), 199–224 (1997)

[12] Sainath, T.N., Weiss, R.J., Senior, A.W., Wilson, K.W., Vinyals, O.: Learning the speech

front-end with raw waveform cldnns. In: Interspeech, pp. 1–5 (2015). Dresden, Germany

[13] Ravanelli, M., Bengio, Y.: Speaker recognition from raw waveform with sincnet. In: 2018 IEEE Spoken Language Technology Workshop (SLT), pp. 1021–1028 (2018). IEEE

[14] Zeghidour, N., Teboul, O., Chaumont Quitry, F., Tagliasacchi, M.: Leaf: A learnable frontend for audio classification. In: International Conference on Learning Representations

[15] Piczak, K.J.: Esc: Dataset for environmental sound classification. In: Proceedings of the 23rd ACM International Conference on Multimedia, pp. 1015–1018 (2015)

[16] Kroos, C., Cartwright, M., Mimilakis, S.I., Mesaros, A., Virtanen, T., Plumbley, M.D.: Generalisation in environmental sound classification: The Making Sense of Sounds data set and challenge. In: Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), pp. 1–5 (2019)

[17] Gaver, W.W.: What in the world do we hear?: An ecological approach to auditory event perception. Ecological psychology 5(1), 1–29 (1993)

[18] Information Technology — Multimedia Content Description Interface — Part 4: Audio. ISO/IEC

[19] Shannon, C.E.: A mathematical theory of communication. The Bell system technical journal 27(3), 379–423 (1948)

[20] Pearson, K.: Vii. mathematical contributions to the theory of evolution.—iii. regression, heredity, and panmixia. Philosophical Transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character (187), 253–318 (1896)

[21] Cover, T., Hart, P.: Nearest neighbor pattern classification. IEEE transactions on information theory 13(1), 21–27 (1967)

[22] Cox, D.R.: The regression analysis of binary sequences. Journal of the Royal Statistical Society Series B: Statistical Methodology 20(2), 215–232 (1958)

[23] Cortes, C., Vapnik, V.: Support-vector networks. Machine learning 20(3), 273–297 (1995)

[24] Breiman, L.: Random forests. Machine learning 45(1), 5–32 (2001)

[25] JOHN, G.: Estimating continuous distributions in bayesian classifiers. In: Eleventh Conference on Uncertanty in Artificical Intelligence, Proc., 1995, pp. 338–345 (1995)

[26] Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd Acm Sigkdd International Conference on Knowledge Discovery and Data Mining, pp. 785–794 (2016)

[27] KITTLER, J., HATEF, M., DUIN, R., MATAS, J.: On combining classifiers. IEEE transactions on pattern analysis and machine intelligence 20(3), 226–239 (1998)