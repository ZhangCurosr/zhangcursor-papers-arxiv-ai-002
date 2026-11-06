# Randomized Parallel Complexity of Weakly Convex Optimization

Kang Liu<sup>∗</sup>

## Abstract

We study the minimum number of query rounds needed to find approximate stationary points of nonsmooth weakly convex functions, measured by the gradient of the Moreau envelope. Each round permits multiple simultaneous queries. In the stated parameter regime, the standard subgradient rate is optimal in suficiently high dimensions, even for randomized algorithms making polynomially many parallel queries per round. We then show how the order structure of the real line enables faster search: a deterministic algorithm using only function values achieves logarithmic round complexity, with a further reduction through parallel queries. Matching lower bounds hold even when each query returns the entire subdiferential. Together, these results clarify the role of dimension in determining the benefits of parallel queries for weakly convex optimization.

Keywords: Weakly convex optimization, oracle complexity, parallel algorithms, randomized algorithms, Moreau envelope.

## 1 Introduction

A function $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is ρ-weakly convex (WC) if $x \mapsto f ( x ) + ( \rho / 2 ) \| x \| ^ { 2 }$ is convex, where $\rho > 0$ and $\| \cdot \|$ denotes the Euclidean norm. Weak convexity allows both nonconvexity and nonsmoothness while controlling negative curvature [13]. First-order methods seek approximate stationary points through successive queries of function values and subgradients. A basic question is how many query rounds are necessary to attain a prescribed stationarity accuracy, and whether randomization or parallel queries can reduce this number.

Problem Let $\mathcal { F } _ { d } ( G , \rho , \Delta )$ denote the class of $\rho { - } \mathrm { W C }$ functions satisfying

$$
\begin{array} { r } { | f ( x ) - f ( y ) | \leq G \| x - y \| \mathrm { ~ f o r ~ a l l ~ } x , y \in \mathbb { R } ^ { d } , f ( 0 ) - \operatorname* { i n f } f \leq \Delta , } \end{array}
$$

where $G > 0$ bounds the global Lipschitz constant and $\Delta > 0$ bounds the initial objective gap, with the initial point taken to be the origin. We use the subdiferential $\partial f ( x ) : = \partial _ { \operatorname { c v x } } ( f + ( \rho / 2 ) \| \cdot \| ^ { 2 } ) ( x ) - \rho x$ , where $\partial _ { \mathrm { c v x } }$ denotes the convex subdiferential.

Approximate stationarity is measured by the gradient of the Moreau envelope (ME) [4, 5], defined by $\begin{array} { r } { f _ { 1 / ( 2 \rho ) } ( x ) : = \operatorname* { m i n } _ { y \in \mathbb { R } ^ { d } } \{ f ( y ) + \rho \| y - x \| ^ { 2 } \} } \end{array}$ . For $\varepsilon > 0$ , the condition $\| \nabla f _ { 1 / ( 2 \rho ) } ( x ) \| \le \varepsilon$ guarantees that some point within distance $\varepsilon / ( 2 \rho )$ of x has a subgradient of norm at most ε.

Each query returns the exact function value and the entire subdiferential, ${ \mathcal { O } } _ { f } ( x ) : = ( f ( x ) , \partial f ( x ) )$ . For an integer $P \geq 1$ , the algorithm submits at most P queries in each round, chosen jointly using its internal randomness and replies from earlier rounds. Replies from the current round become available only after all its queries have been chosen. Thus, $P = 1$ gives the sequential model, while larger $P$ permits parallel evaluations [8]. Counting rounds measures the sequential interaction required with the oracle. Query points, output points, and computation between rounds are unrestricted.

We seek the smallest integer $T \geq 0$ for which there exists a randomized algorithm using at most P queries per round and at most T rounds on every execution, such that

$$
\operatorname* { P r } _ { A } \{ \| \nabla f _ { 1 / ( 2 \rho ) } ( \widehat { x } ) \| \leq \varepsilon \} \geq \frac { 3 } { 4 } \mathrm { ~ f o r ~ e v e r y ~ } f \in \mathcal { F } _ { d } ( G , \rho , \Delta ) .
$$

Here, xb is the algorithm’s output, and the probability is taken over its internal randomness. A hard instance may depend on the algorithm but is independent of its realized random seed.

Results We construct high-dimensional hard instances to establish a round lower bound for randomized parallel algorithms. Suppose $\varepsilon ^ { 2 } \le c _ { 0 } \operatorname* { m i n } \{ G ^ { 2 } , \rho \Delta \}$ and $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ . For every fixed $k \geq 0$ , there exists a suficiently large dimension d in which the optimal round complexity is $\Theta ( \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 } )$ simultaneously for all $1 \leq P \leq d ^ { k }$ . This lower bound matches the standard subgradient upper bound [4], showing that the standard rate remains optimal even with randomization and polynomially many parallel queries per round.

We then exploit the order structure of the real line through a deterministic interval search using only function values. Parallel queries allow finer subdivisions of the search interval in each round. $\operatorname { I f } \varepsilon \leq c _ { 1 } G$ and $\rho \Delta \geq C _ { 1 } \varepsilon ^ { 2 }$ , this algorithm attains the optimal round complexity $\Theta ( 1 + \log ( \rho \Delta / \varepsilon ^ { 2 } ) / \log ( P + 1 ) )$ for every $P \geq 1$ . The matching lower bound holds for randomized algorithms receiving the entire subdiferential. Thus, the order structure enables logarithmic sequential complexity, with a further reduction through parallel queries. Here, $c _ { 0 } , c _ { 1 } > 0$ and $C _ { 1 } > 1$ are universal constants.

Roadmap Section 2 reviews related work. Sections 3 and 4 develop the construction and establish the high-dimensional complexity bounds, respectively. Section 5 derives the matching one-dimensional bounds. The main text presents the central construction and complexity arguments; the appendices provide detailed regularity estimates, auxiliary calculations, and supporting proofs.

## 2 Related work

The first-order theory of weakly convex optimization has progressed from asymptotic convergence to finitetime stationarity guarantees. Early work established asymptotic convergence for stochastic subgradient and composite methods [9], while proximal regularization yielded nonasymptotic guarantees [6]. Using the gradient of the Moreau envelope to measure stationarity, analyses of the subgradient method and more general model-based methods established the complexity bound ${ \cal O } ( \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 } ) [ 4 , 5 ]$ . These upper bounds raise the question of whether this query complexity is intrinsic to the function class. Under an oracle returning exact function values and the entire subdiferential, Li and Pan [11] proved a matching deterministic lower bound for $\Delta \le G ^ { 2 } / \rho ,$ establishing the optimal deterministic query complexity in the corresponding parameter regime.

This deterministic optimality leaves open whether randomization and parallel queries can improve the complexity. Assessing parallelism requires measuring the number of sequential query rounds rather than the total number of queries. This perspective was developed in parallel oracle models for convex optimization [12], with subsequent lower bounds covering randomized algorithms and several smoothness classes [8]. For nonconvex stationarity, hard instances can force algorithms to acquire information about new directions sequentially; this mechanism underlies randomized query lower bounds for smooth nonconvex optimization [2]. Parallel lower bounds further show that, in suficiently high dimensions, polynomially many queries per round do not improve the optimal complexity order for the corresponding smooth function classes [15].

Dimension also afects the complexity of stationarity search. For smooth optimization over compact domains, gradient-flow trapping demonstrates the advantage of low-dimensional search over general descent methods [1]. In one-dimensional unconstrained optimization, function values sufice for optimal logarithmic search complexity under smoothness [3], with extensions to higher-order Hölder smoothness [14]. Lowdimensional improvements also occur in nonsmooth optimization. Using Goldstein stationarity, which measures the norm of convex combinations of subgradients at nearby points, Davis et al. [7] obtain improved complexity bounds for Lipschitz and weakly convex functions through a cutting-plane method. Under the same stationarity criterion, Kornowski and Shamir [10] establish a randomized one-dimensional lower bound logarithmic in the inverse neighborhood radius, using binary strings encoded by nested intervals.

We establish matching randomized parallel bounds for nonsmooth weakly convex optimization under the Moreau-envelope stationarity criterion. In high dimensions, the standard subgradient rate remains optimal with polynomially many queries per round, and the matching range extends to $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ . We then characterize the faster search enabled by the order structure of the real line, including its dependence on the parallel query budget. Both lower bounds allow access to the entire subdiferential, while the one-dimensional

upper bound requires only function values.

## 3 Construction of high-dimensional hard instances

To prove a lower bound on query rounds, we relate the information supplied by the oracle to approximate stationarity. We construct functions with two properties: exact oracle locality restricts how replies depend on successive coordinate groups, and a lower bound on the ME gradient excludes approximate stationarity when the coordinates in the final block are suficiently small.

Starting from the block coupling of Li and Pan [11], we build K component functions, called stages, on disjoint coordinate sets. A parameter N controls the size of each stage. For $K \leq N$ , the final function has Lipschitz and weak convexity parameters at most one, an initial objective gap of order $K$ , and order $K N ^ { 2 }$ coordinate groups. When the final-block projections are suficiently small, its ME gradient norm is bounded below by a constant multiple of $N ^ { - 1 / 2 }$ . The construction therefore allows the initial gap and the number of coordinate groups to grow together while maintaining the required stationarity bound.

Let $F : \mathbb { R } ^ { n } $ R be a finite-valued WC function, and write $[ n ] : = \{ 1 , \dots , n \}$ . For a nonempty set $C .$ let dist $( 0 , C ) : = \operatorname* { i n f } _ { g \in C } \| g \|$ . Consider nested coordinate index sets $\emptyset = \mathbb { Z } _ { - 1 } \subseteq \mathbb { Z } _ { 0 } \subseteq \cdot \cdot \cdot \subseteq \mathbb { Z } _ { H } = [ n ]$ , where $\mathcal { T } _ { t } \setminus \mathcal { T } _ { t - 1 }$ is the tth group of coordinate indices. Let $\Pi _ { t }$ be the coordinate projection retaining the coordinates in $\mathcal { T } _ { t }$ and setting the others to zero.

Definition 1 (Exact oracle locality). Given $a > 0$ , the function F has exact oracle locality at scale a with respect to $( \mathcal { T } _ { t } ) _ { t = - 1 } ^ { H }$ if, for every $t \in \{ 0 , \ldots , H \}$ and every z satisfying $| z _ { j } | \leq a$ for $j \notin \mathcal { T } _ { t - 1 }$ $F ( z ) = F ( \Pi _ { t } z ) , ~ \partial F ( z ) = \partial F ( \Pi _ { t } z ) \subseteq$ range Π<sub>t</sub>.

Thus, when the coordinates outside $\mathcal { T } _ { t - 1 }$ are suficiently small, the oracle reply may depend on the new coordinates in $\mathcal { T } _ { t } \setminus \mathcal { T } _ { t - 1 }$ , but coordinates outside $\mathcal { T } _ { t }$ afect neither the function value nor the entire subdiferential.

## 3.1 The base function and its oracle locality

Fix integers $N \geq 6 0$ and $2 \leq M \leq N / 1 2$ . Write $\boldsymbol { u } = ( u ^ { ( 1 ) } , \ldots , u ^ { ( M ) } ) \in \mathbb { R } ^ { M N }$ , with $u ^ { ( r ) } = ( u _ { 1 } ^ { ( r ) } , \dots , u _ { N } ^ { ( r ) } )$ Coordinates are indexed by pairs $( r , j )$ , where r identifies the block and $j$ the coordinate within it; the corresponding linear index is $( r - 1 ) N + j$

Set $a : = 1 / ( 8 N ^ { 2 } ) , L : = \lfloor 3 N / 1 6 \rfloor$ , and $H _ { b } : = ( M - 1 ) L + N - 1$ . For $0 \leq t \leq H _ { b }$ , define

$$
\mathcal { I } _ { t } : = \{ ( r , j ) \in [ M ] \times [ N ] : ( r - 1 ) L + j - 1 \leq t \} , ~ \mathcal { I } _ { - 1 } : = \emptyset .
$$

Let $Q _ { t }$ be the orthogonal projection retaining the coordinates in $\mathcal { T } _ { t }$ . The first coordinate of block r enters at $t = ( r - 1 ) L$ , and $\mathcal { I } _ { H _ { b } } = [ M ] \times [ N ]$

The base function combines afine branches within each block with progress terms linking adjacent blocks. Set $c _ { j } : = 1 + ( N - j ) / ( 1 6 N )$ and define $q ( s ) : = 0$ for $s \leq 0 , q ( s ) : = 3 s ^ { 2 } - 2 s ^ { 3 }$ for $0 < s < 1$ , and $q ( s ) : = 1$ for $s \geq 1$ . For $v \in \mathbb { R } ^ { N }$ , let

$$
\phi ( v ) : = \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { 1 \leq j \leq N } \left( c _ { j } - \frac { N } { 8 } v _ { j } \right) \right\} , \psi ( v ) : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } q \left( \frac { N } { 4 } ( v _ { j } - 2 a ) \right) .
$$

The intercept gap $c _ { j } - c _ { j + 1 } = 1 / ( 1 6 N )$ separates the afine branches when the corresponding coordinates are small. The function ψ takes values in [0, 1] and measures progress within a block. The shift by 2a ensures that each summand and its derivative vanish whenever $| v _ { j } | \leq a$

Let $p _ { 0 } : = 1 , p _ { r } : = \psi ( u ^ { ( r ) } ) , \sigma _ { r } : = \phi ( u ^ { ( r ) } ) + p _ { r - 1 } - 5 / 4 ,$ and $\sigma ( u ) : = ( \sigma _ { 1 } , \dots , \sigma _ { M } )$ . Define

$$
S ( \boldsymbol { u } ) : = \| [ \sigma ( \boldsymbol { u } ) ] _ { + } \| , h _ { M , N } ( \boldsymbol { u } ) : = \frac { 2 } { N } \left( S ( \boldsymbol { u } ) - \sum _ { r = 1 } ^ { M } p _ { r } \right) ,
$$

where $[ w ] _ { + }$ denotes the componentwise positive part of $w$ . The negative sum decreases as the progress values grow. Increasing $p _ { r - 1 }$ raises $\sigma _ { r }$ , while reducing $\phi ( u ^ { ( r ) } )$ ofsets this increase. This couples progress in adjacent blocks.

Proposition 1 (Properties of the base function). The function $h _ { M , N } \ i s \ 3 / 4 \mathrm { - } W C ;$ , with $\mathrm { L i p } ( h _ { M , N } ) \leq 1 / 4 +$ $( 3 / 4 ) \sqrt { M / N } < 1 / 2$ . It satisfies $h _ { M , N } \geq - 2 M / N , h _ { M , N } ( 0 ) \in [ 0 , 2 / N )$ , and $h _ { M , N } ( 0 )$ −inf $h _ { M , N } \leq 3 M / N$ . All its subgradients are componentwise nonpositive. $I f S ( u ) > 0$ , then $\| g \| \geq 1 / ( 4 \sqrt { N } )$ for every $g \in \partial h _ { M , N } ( u )$ Moreover, $p _ { M } < 2 7 / 3 2$ implies $S ( u ) > 0$

In particular, insuficient progress in the last block guarantees the subgradient norm lower bound. The proof is given in Appendix B.

Lemma 1 (Exact oracle locality of the base function). For every integer $0 \leq t \leq H _ { b } , \ i f \vert u _ { j } ^ { ( r ) } \vert \leq a$ for all $( r , j ) \notin \mathcal { J } _ { t - 1 } , \ t h e n \ h _ { M , N } ( u ) = h _ { M , N } ( Q _ { t } u ) , \ \partial h _ { M , N } ( u ) = \partial h _ { M , N } ( Q _ { t } u ) \subseteq \mathrm { r a n g e } Q _ { t }$

The proof establishes $h _ { M , N } = h _ { M , N } \circ Q _ { t }$ on open neighborhoods of both u and $Q _ { t } u$ . On these neighborhoods, omitted coordinates afect neither the progress terms nor the active block terms. The local identity yields equality of the full subdiferentials. The details are given in Appendix C.

## 3.2 Construction of a single stage

We will combine shifted copies of a single-stage function by taking their maximum, which preserves common Lipschitz and weak convexity bounds. Subgradients of the maximum are convex combinations of subgradients of the active branches, namely those attaining the maximum. To control the resulting decrease in norm, we modify the base function so that subgradients of norm below $1 / 4$ occur only within a fixed interval of function values.

For the stage construction, take $N \ge 1 2 0 , m : = \lfloor N / 1 2 \rfloor , d _ { 0 } : = m N .$ and $h _ { N } : = h _ { m , N }$ . Let $e : =$ $( 0 , \dots , 0 , \mathbf { 1 } _ { N } / \sqrt { N } ) \in \mathbb { R } ^ { d _ { 0 } }$ , where $\mathbf { 1 } _ { N }$ is the all-ones vector; thus, e is a unit vector supported on block $m .$ Define $\begin{array} { r } { \tau _ { N } ( s ) : = \int _ { 0 } ^ { ( s - 1 / \sqrt { N } ) . } } \end{array}$ <sup>+</sup> min $\{ t / 2 0 , 1 / 4 \}$ dt and

$$
b _ { N } ( u ) : = h _ { N } ( u ) + \frac { 1 } { 4 } \| [ - u - 2 a \mathbf { 1 } ] _ { + } \| - \tau _ { N } ( \langle e , u \rangle ) .
$$

The penalty on negative coordinates controls subgradient norms at large function values. The terminal descent term becomes active when $\langle e , u \rangle > 1 / \sqrt { N }$ , allowing further decrease and controlling subgradien norms at small function values.

Lemma 2 (Single-stage properties). The function $b _ { N }$ is $4 / 5 – W C$ and globally Lipschitz with constant less than $3 1 / 3 2$ . For every u and $g \in \partial b _ { N } ( u )$ , the vector g is componentwise nonpositive, $\| g \| \ge 1 / ( 4 \sqrt { N } )$ ， and $\langle g , u \rangle \ge b _ { N } ( u ) - 2 . ~ I f b _ { N } ( u ) \notin ( - 1 , 1 / 4 )$ , then $\| g \| \ge 1 / 4$ . Moreover, $b _ { N } ( 0 ) = h _ { N } ( 0 ) \in [ 0 , 2 / N )$ , and $\left. e , u \right. \leq 1 / \sqrt { N }$ implies $b _ { N } ( u ) \geq - 1 / 6$ . The function $b _ { N } \ i s$ unbounded below along the positive e direction.

The three terms defining $b _ { N }$ have componentwise nonpositive subgradients, so their contributions cannot cancel. At function values at least $1 / 4$ , the penalty has gradient norm $1 / 4 ;$ at values at most −1, the terminal descent term has slope magnitude $1 / 4$ . The full proof, including the regularity and inner-product estimates, is given in Appendix D.

## 3.3 Combining stages while preserving the subgradient bound

We choose shifts that separate the intervals of function values where small subgradients can occur. At most one active stage can then have a small subgradient, while the remaining active stages satisfy a stronger bound. The following lemma quantifies the resulting bound for their maximum.

Lemma 3 (Combination with separated value intervals). Let $b _ { i } : \mathbb { R } ^ { d _ { i } }  \mathbb { R } , i = 1 , \ldots , K .$ , be finite-valued WC functions on disjoint coordinate blocks. Suppose there are a bounded interval I and constants $0 < \beta \le \gamma$ such

that dist $( 0 , \partial b _ { i } ( v ) ) \geq \beta$ for every $i , v$ , and dist $( 0 , \partial b _ { i } ( v ) ) \geq \gamma$ whenever $b _ { i } ( v ) \notin I$ . Choose shifts satisfying $| s _ { i } - s _ { j } | >$ length(I) for $i \neq j$ , and let

$$
\mathcal { M } ( z ) : = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { 1 \leq i \leq K } \left( b _ { i } ( z ^ { ( i ) } ) - s _ { i } \right) , \nu \right\} ,
$$

where ν is constant. $I f \mathcal { M } ( z ) > \nu ,$ then every $g \in \partial \mathcal { M } ( z )$ satisfies $\| g \| \ge ( \beta ^ { - 2 } + ( K - 1 ) \gamma ^ { - 2 } ) ^ { - 1 / 2 }$ . If the stages share a Lipschitz bound and a weak convexity parameter, then M satisfies the same bounds.

Proof. Let $\mathcal { A } : = \{ i : b _ { i } ( z ^ { ( i ) } ) - s _ { i } = \mathcal { M } ( z ) \}$ be the set of active stages. Since the constant branch is inactive, every $g \in \partial \mathcal { M } ( z )$ has a representation $\begin{array} { r } { g = \sum _ { i \in \mathcal { A } } \lambda _ { i } \widetilde { g } _ { i } } \end{array}$ , where $\begin{array} { r } { \lambda _ { i } \geq 0 , \sum _ { i \in \mathcal { A } } \lambda _ { i } = 1 , g _ { i } \in \partial b _ { i } ( z ^ { ( i ) } ) } \end{array}$ , and $\widetilde { g } _ { i }$ is the embedding of $g _ { i }$ into its coordinate block.

Each active stage satisfies $b _ { i } ( z ^ { ( i ) } ) = \mathcal { M } ( z ) + s _ { i }$ . The separation of the shifts implies that at most one active stage has its function value in I. Since distinct stages have orthogonal coordinate supports, Cauchy–Schwarz gives

$$
\| g \| ^ { 2 } = \sum _ { i \in \mathcal { A } } \lambda _ { i } ^ { 2 } \| g _ { i } \| ^ { 2 } \geq \frac { 1 } { \sum _ { i \in \mathcal { A } } \| g _ { i } \| ^ { - 2 } } \geq \frac { 1 } { \beta ^ { - 2 } + ( K - 1 ) \gamma ^ { - 2 } } .
$$

The regularity claims follow from the Lipschitz property of the maximum and a common convexifying quadratic. □

Fix an integer $1 \le K \le N$ . Write $z = ( z ^ { ( 1 ) } , \dots , z ^ { ( K ) } ) \in \mathbb { R } ^ { n }$ , where $\boldsymbol { z } ^ { ( i ) } \in \mathbb { R } ^ { d _ { 0 } }$ and $n : = K d _ { 0 }$ , and define

$$
F _ { K } ( z ) : = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { 1 \leq i \leq K } \bigl ( b _ { N } ( z ^ { ( i ) } ) - 2 ( i - 1 ) \bigr ) , - 2 K \right\} .
$$

Adjacent shifts difer by $^ { 2 , }$ exceeding the length of $( - 1 , 1 / 4 )$ . The constant branch bounds $F _ { K }$ below by −2K. Each stage can decrease independently along its terminal direction, so inf $F _ { K } = - 2 K$ and $F _ { K } ( 0 ) = h _ { N } ( 0 )$ giving an initial objective gap of order K.

Applying Lemma 3 with $\beta = 1 / ( 4 \sqrt { N } )$ and $\gamma = 1 / 4$ gives

$$
F _ { K } ( z ) > - 2 K \implies \mathrm { d i s t } ( 0 , \partial F _ { K } ( z ) ) \geq \sigma _ { K } , \sigma _ { K } : = \frac { 1 } { 4 \sqrt { N + K - 1 } } .
$$

Since $K \leq N$ , we have $\sigma _ { K } \geq 1 / ( 4 \sqrt { 2 N } )$ , so the combined function retains a subgradient lower bound of order $N ^ { - 1 / 2 }$

The function $F _ { K }$ is $4 / 5 – \mathrm { W C }$ and has Lipschitz constant less than $3 1 / 3 2$ . We will also use the innerproduct estimate $\langle g , z \rangle \ge F _ { K } ( z ) - 2 \ge - 2 K - 2$ for every $g \in \partial F _ { K } ( z )$ . Indeed, if stage i is active, then $b _ { N } ( z ^ { ( i ) } ) = F _ { K } ( z ) + 2 ( i - 1 )$ , and Lemma 2 bounds the corresponding inner product below by $F _ { K } ( z ) - 2$ When the constant branch is active, its zero subgradient also satisfies this bound. Taking convex combinations proves the bound.

## 3.4 Oracle locality of the combined function

We now establish exact oracle locality for $F _ { K }$ with coordinate groups ordered stage by stage. Index coordinates by triples $( i , r , j )$ , where i identifies the stage, r the block within that stage, and $j$ the coordinate within the block. The corresponding linear index is $( i - 1 ) m N + ( r - 1 ) N + j$

Set $J : = ( m - 1 ) L$ and $H : = K ( J + 1 ) - 1$ . The terminal descent term depends on the entire last block, so its coordinates enter together in the final group of each stage. We retain the base ordering for indices below J and place all remaining coordinates in that final group. For $0 \leq t \leq H$ , define

$$
\begin{array} { r } { \mathcal { Z } _ { t } : = \big \{ ( i , r , j ) \in [ K ] \times [ m ] \times [ N ] : ( i - 1 ) ( J + 1 ) + \operatorname* { m i n } \{ ( r - 1 ) L + j - 1 , J \} \leq t \big \} , } \end{array}
$$

and let $\mathcal { T } _ { - 1 } : = \mathcal { O }$ . Let Π<sub>t</sub> be the orthogonal projection retaining the coordinates in $\mathcal { T } _ { t }$

By construction, $\mathcal { T } _ { H } = [ K ] \times [ m ] \times [ N ]$ and $\{ ( K , m , j ) : j \in [ N ] \} \subseteq { \mathbb { Z } } _ { H } \setminus { \mathbb { Z } } _ { H - 1 }$ . Thus, the entire last block of the final stage belongs to the last coordinate group. There are $H + 1$ groups, with $H \geq K J \geq K N ^ { 2 } / 2 8 8$

Lemma 4 (Exact oracle locality of the combined function). For every integer $0 \leq t \leq H , \ i f \left| z _ { i , r , j } \right| \leq a$ for all $( i , r , j ) \notin \mathcal { T } _ { t - 1 }$ , then $F _ { K } ( z ) = F _ { K } ( \Pi _ { t } z ) , ~ \partial F _ { K } ( z ) = \partial F _ { K } ( \Pi _ { t } z ) \subseteq$ range Π<sub>t</sub>.

Proof. Write $t = ( i - 1 ) ( J + 1 ) + s$ uniquely with $0 \leq s \leq J .$ . All coordinates of stages preceding i are retained by $\Pi _ { t }$ . The last block of stage i lies outside $\mathcal { T } _ { t - 1 } , \mathrm { ~ s o ~ } \langle e , z ^ { ( i ) } \rangle \leq \sqrt { N } a < 1 / \sqrt { N }$ . Its terminal descent term therefore vanishes on a neighborhood, and deleted coordinates lie in the zero region of the penalty. If $s < J .$ Lemma 1 gives the required local identity for stage i; if $s = J .$ this stage is retained in full.

Lemma 2 gives $b _ { N } \big ( z ^ { ( i ) } \big ) \geq - 1 / 6$ . For every $k > i ,$ , all coordinates of stage k have magnitude at most $^ { a , }$ so its progress terms, penalty, and terminal descent term vanish, and $| b _ { N } ( z ^ { ( k ) } ) - h _ { N } ( 0 ) | \le a / 4$ . The shifts therefore place every subsequent stage strictly below stage i. The constant branch is also strictly below it, since $- 1 / 6 - 2 ( i - 1 ) > - 2 K$

The preceding stages are retained in full, while the local identity for stage i preserves its value and full subdiferential after projection. The later stages and the constant branch are inactive near both z and $\Pi _ { t } z$ Thus, the active branches have identical values and subdiferentials at the two points, with all subgradients supported on $\mathcal { T } _ { t }$ . The subdiferential formula for a maximum proves the claim. □

## 3.5 High-dimensional embedding and ME gradient bounds

We embed $F _ { K }$ into a higher-dimensional space and add a radial branch so that oracle replies outside a bounded region are independent of the embedding. The inner-product estimate above controls the subgradients where both branches are active, allowing us to preserve the required norm bound.

Let $D \geq n$ and $U \in \mathbb { R } ^ { D \times n }$ satisfy $U ^ { \top } U = I _ { n }$ . Set $A : = 2 K + 4 ( 2 K + 2 ) / \sigma _ { K }$ and $B : = 6 5 ( A + 1 )$ , and define $\Phi _ { U } ( x ) : = \operatorname* { m a x } \{ F _ { K } ( U ^ { \top } x ) , \| x \| - A \}$

Proposition 2 (Properties of the radial extension). For every such U, the function $\Phi _ { U }$ is globally 1- Lipschitz and $4 / 5 \AA W C _ { ; }$ , with $\Phi _ { U } ( 0 ) - \operatorname* { i n f } \Phi _ { U } = 2 K + h _ { N } ( 0 ) \in [ 2 K , 3 K ] . \ I f \left\| x \right\| \geq 6 4 ( A + 1 )$ , then $O _ { \Phi _ { U } } ( x ) =$ $( \| x \| - A , \{ x / \| x \| \} )$ . This oracle identity also holds after replacing $F _ { K }$ by $F _ { K } \circ \Pi _ { t } ~ f o r$ any $0 \leq t \leq H$ Moreover, $F _ { K } ( U ^ { \top } x ) > - 2 K$ implies dist $( 0 , \partial \Phi _ { U } ( x ) ) > \sigma _ { K } / 2$

Proof. The regularity bounds, objective gap, and oracle identities are verified in Appendix E. Here we prove the subgradient lower bound, which controls the interaction between the two branches.

Assume $F _ { K } ( U ^ { \top } x ) > - 2 K$ . If only one branch is active, the claim follows from the subgradient lower bound for $F _ { K }$ or the unit norm of the radial gradient. Suppose both branches are active. Let $\mathbf { \boldsymbol { v } } : = U \mathbf { \boldsymbol { g } }$ with $g \in \partial F _ { K } ( U ^ { \top } x )$ , and set $e _ { x } : = x / \| x \|$ . Since the radial branch is active, $\| x \| \ge A - 2 K = 4 ( 2 K + 2 ) / \sigma _ { K }$ Using $\langle g , U ^ { \top } x \rangle \ge - 2 K - 2$ , we obtain

$$
\langle v , e _ { x } \rangle \geq - \frac { 2 K + 2 } { \| x \| } \geq - \frac { \sigma _ { K } } { 4 } \geq - \frac { \| v \| } { 4 } .
$$

Thus the radial gradient cannot nearly cancel an embedded subgradient. More precisely, for every $0 \leq \lambda \leq 1$

$$
\| \lambda v + ( 1 - \lambda ) e _ { x } \| \ge \sqrt { \frac { 3 } { 8 } } \operatorname* { m i n } \{ \| v \| , 1 \} \ge \sqrt { \frac { 3 } { 8 } } \sigma _ { K } > \frac { \sigma _ { K } } { 2 } .
$$

The full subdiferential consists of these convex combinations, proving the bound.

To bound the ME gradient, we apply the subgradient estimate at the proximal point. A small ME gradient places this point near x. If the projections of x onto the last block are also small, the embedded function remains above −2K at the proximal point. Let $u _ { K , m , j }$ denote the column of U corresponding to coordinate $( K , m , j )$

Lemma 5 (Last-block projections and the ME gradient). $I f \ | \langle u _ { K , m , j } , x \rangle | \ \leq \ a$ for every $j \in [ N ]$ , then $\| \nabla ( \Phi _ { U } ) _ { 1 / 2 } ( x ) \| > \sigma _ { K } / 4 = 1 / ( 1 6 \sqrt { N + K - 1 } )$ . The same bound holds whenever $\| { \boldsymbol { x } } \| \geq B$

Proof. Suppose $\| \nabla ( \Phi _ { U } ) _ { 1 / 2 } ( x ) \| \le \sigma _ { K } / 4$ . Let $y : = \arg$ min ${ \bf \nabla } _ { \cdot w } \{ \Phi _ { U } ( w ) + \| w - x \| ^ { 2 } \}$ and $v : = \nabla ( \Phi _ { U } ) _ { 1 / 2 } ( x )$ Proximal optimality gives $v = 2 ( x - y ) \in \partial \Phi _ { U } ( y )$ and $\| x - y \| \leq \sigma _ { K } / 8$

Under the projection condition in the lemma, $\langle e , ( \tilde { U ^ { \top } } y ) ^ { ( K ) } \rangle \leq \sqrt { N } a + \sigma _ { K } / 8 < 1 / \sqrt { N }$ . Lemma 2 bounds the value of the last stage below by $- 1 / 6$ , so $F _ { K } ( U ^ { \top } y ) \ge - 1 / 6 - 2 ( K - 1 ) > - 2 K$ . Proposition 2 gives $\| v \| > \sigma _ { K } / 2$ , a contradiction.

If instead $\| { \boldsymbol { x } } \| \geq B$ , the same displacement bound gives $\| y \| \ge B - \sigma _ { K } / 8 > 6 4 ( A + 1 )$ . By the radial oracle identity, the subdiferential at $y$ contains only the unit radial vector, so $\lVert \boldsymbol { v } \rVert = 1$ , again a contradiction.

For $1 \le K \le N$ , the resulting function Φ<sub>U</sub> has uniformly bounded Lipschitz and weak convexity parameters and an initial objective gap of order K. Before embedding, $F _ { K }$ has order $K N ^ { 2 }$ ordered coordinate groups and satisfies exact oracle locality; suficiently small projections onto the final block imply an ME gradient lower bound of order $N ^ { - 1 / 2 }$ for $\Phi _ { U }$ . The radial extension makes oracle replies outside a bounded region independent of the embedding, combining oracle locality, a controlled initial ${ \mathrm { g a p } } ,$ and the required stationarity bound in the same construction.

## 4 Randomized parallel complexity in high dimensions

This section establishes round lower bounds for randomized parallel algorithms in high dimensions. Since queries depend on previous oracle replies, they also depend indirectly on the embedding directions. We control this dependence through an auxiliary execution and use concentration under a random embedding to verify the projection conditions needed to exclude approximate stationarity. We state the results for a target success probability of $3 / 4$ . The same round complexity orders hold for any fixed success probability $p \in ( 1 / 2 , 1 )$ , with implicit constants that may depend on $p .$

## 4.1 Main theorem and the matching regime

Theorem 1 (High-dimensional round lower bound). There exist universal constants $c _ { 0 } , c , C > 0$ such that, for all $G , \rho , \Delta , \varepsilon > 0$ satisfying $\varepsilon ^ { 2 } \leq c _ { 0 }$ min $\{ G ^ { 2 } , \rho \Delta \}$ and every integer $P \geq 1$ , there is a dimension $D \leq C ( G / \varepsilon ) ^ { 1 4 } \log ( P G / \varepsilon )$ with the following property. For every randomized algorithm using at most $P$ queries per round and at most

$$
\left\lfloor c \operatorname* { m i n } \left\{ \frac { \rho G ^ { 2 } \Delta } { \varepsilon ^ { 4 } } , \frac { G ^ { 6 } } { \varepsilon ^ { 6 } } \right\} \right\rfloor\tag{1}
$$

rounds on every execution, there is a fixed instance $f \in { \mathcal { F } } _ { D } ( G , \rho , \Delta )$ such that its output xb satisfies

$$
\operatorname* { P r } _ { A } \left\{ \| \nabla f _ { 1 / ( 2 \rho ) } ( \widehat { x } ) \| > \varepsilon \right\} \geq \frac { 3 } { 4 } .
$$

The instance may depend on the algorithm but is independent of its realized random seed.

For every fixed $k \geq 0$ , a dimension $D = O _ { k } { \big ( } ( G / \varepsilon ) ^ { 1 4 } \log ( G / \varepsilon ) { \big ) }$ can be chosen so that the same lower bound holds simultaneously for all $1 \le P \le D ^ { k }$

Combining this result with the standard subgradient upper bound gives the following optimal complexity characterization.

Corollary 1 (Optimal high-dimensional round complexity). Under the accuracy condition of Theorem 1, suppose $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ . For every fixed integer $P \geq 1$ , the randomized parallel round complexity, with the worst case taken over dimensions, is $\Theta ( \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 } )$ . For every fixed $k \geq 0$ , this order also holds in a single suficiently large dimension $D = O _ { k } { \big ( } ( G / \varepsilon ) ^ { 1 4 } \log ( G / \varepsilon ) { \big ) }$ ), simultaneously for all $1 \le P \le D ^ { k }$

The condition $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ makes the first term in the minimum in Theorem 1 no larger than the second, yielding a lower bound that matches the subgradient upper bound. For fixed $G , \rho , \Delta$ , this matching condition holds whenever the target accuracy ε is suficiently small. Beyond this range, the theorem still provides a lower bound, but its order remains $G ^ { 6 } / \varepsilon ^ { 6 }$ and no longer grows with $\Delta$

## 4.2 Information revealed by randomized parallel queries

We use the function $\Phi _ { U }$ , index sets $\mathcal { T } _ { t }$ , projections $\Pi _ { t }$ , and parameters $n , a , H , A , B , \sigma _ { K }$ from Section 3. Identify the triple coordinate indices with $j \in [ n ]$ , and let $u _ { j }$ denote the corresponding column of $U$ . Draw $U$ uniformly from the matrices in $\mathbb { R } ^ { D \times n }$ satisfying $U ^ { \top } U = I _ { n }$ , independently of the algorithm’s internal randomness.

Proposition 3 (Normalized lower bound for the stage construction). Let $N \geq 1 2 0 , 1 \leq K \leq N$ , and $P \geq 1$ be integers. $I f$

$$
D \geq n + 2 + \frac { 2 B ^ { 2 } } { a ^ { 2 } } \log \bigl ( 8 n ( P H + 1 ) \bigr ) ,
$$

then every randomized algorithm using at most P queries per round and at most H rounds on every execution satisfies

$$
\operatorname* { P r } _ { U , A } \left\{ \| \nabla ( \Phi _ { U } ) _ { 1 / 2 } ( \widehat { x } ) \| > \frac { \sigma _ { K } } { 4 } \right\} \ge \frac { 3 } { 4 } .
$$

Consequently, for every such algorithm, there is a fixed embedding $U _ { 0 }$ for which the failure probability on $\Phi _ { U _ { 0 } }$ is at least $3 / 4$

Proof. Fix an algorithm with a round cap $T \leq H$ , and first fix its internal random seed. Its queries and output are then measurable functions of the replies received.

To control the dependence of the queries on the random embedding, we introduce an auxiliary execution whose replies in round t depend only on columns indexed by $\mathcal { T } _ { t }$ . This allows us to apply conditional spherical concentration to columns that have not yet been used when the queries are selected.

Define the filtration $\mathcal { F } _ { t } : = \sigma \{ u _ { j } : j \in \mathcal { T } _ { t - 1 } \}$ for $t = 0 , \ldots , T$ . It records the columns used to determine the auxiliary replies before round t. In the auxiliary execution, round $t ,$ indexed from 0 to $T - 1$ , returns the exact function value and entire subdiferential of $\Phi _ { U , t } ( x ) : = \operatorname* { m a x } \{ F _ { K } ( \Pi _ { t } U ^ { \top } x ) , \left. x \right. - A \}$ . This function and its replies depend only on columns indexed by $\mathcal { T } _ { t }$

All queries in a round are selected before any reply from that round is received. Inductively, every query in round t is therefore $\mathcal { F } _ { t }$ -measurable, the replies are $\mathscr { F } _ { t + 1 }$ -measurable, and the final output $x _ { \mathrm { o u t } }$ is $\mathcal { F } _ { T }$ -measurable. An algorithm that stops early can be padded to $T$ rounds without changing its output.

Conditioned on $\mathcal { F } _ { t } .$ , each column $u _ { j }$ with $j \notin \mathcal { T } _ { t - 1 }$ is uniformly distributed on the unit sphere in the orthogonal complement of the columns indexed by $\mathcal { T } _ { t - 1 }$ . This space has dimension $D - \left| \mathcal { I } _ { t - 1 } \right| \geq D - n$ . Let $x _ { t , p }$ denote the pth auxiliary query in round t. Since $x _ { t , p }$ is $\mathcal { F } _ { t }$ -measurable, spherical concentration $\mathrm { g i v e s }$ , for every $j \notin \mathcal { T } _ { t - 1 }$ ,

$$
\operatorname* { P r } \{ \| x _ { t , p } \| \leq B , | \langle u _ { j } , x _ { t , p } \rangle | > a | \mathcal { F } _ { t } \} \leq 2 \exp ( - \frac { ( D - n - 2 ) a ^ { 2 } } { 2 B ^ { 2 } } ) .
$$

The same estimate applies to the output, conditioned on $\mathcal { F } _ { T }$ , for every $j \notin \mathcal { T } _ { T - 1 }$

Let $\mathcal { G }$ be the event that every auxiliary query with norm at most B satisfies $| \langle u _ { j } , x _ { t , p } \rangle | \leq a$ for all $j \notin \mathcal { T } _ { t - 1 }$ and that, whenever $\| x _ { \mathrm { o u t } } \| \le B$ , the output satisfies $| \langle u _ { j } , x _ { \mathrm { o u t } } \rangle | \leq a$ for all $j \notin \mathcal { T } _ { T - 1 }$ . Taking expectations in the conditional bound and applying a union bound over at most PT queries, the final output, and n directions gives

$$
\operatorname* { P r } _ { U } ( \mathcal G ^ { c } ) \le 2 n ( P T + 1 ) \exp \left( - \frac { ( D - n - 2 ) a ^ { 2 } } { 2 B ^ { 2 } } \right) \le \frac 1 4 .
$$

On ${ \mathcal { G } } _ { : }$ , the real and auxiliary executions have identical replies. Indeed, if their previous replies agree, they submit the same queries in the next round. At a query of norm at most B, the projection conditions and Lemma 4, together with the subdiferential rule for linear composition, give $\mathcal { O } _ { \Phi _ { U } } ( x _ { t , p } ) = \mathcal { O } _ { \Phi _ { U , t } } ( x _ { t , p } )$ . At a query of norm greater than B, both replies are determined by the same radial branch, by Proposition $^ { 2 , }$ including its assertion for the truncated functions. Induction therefore gives identical transcripts and outputs.

Since $T \leq H$ , we have $\mathcal { I } _ { T - 1 } \subseteq \mathcal { T } _ { H - 1 }$ . The entire last block of the final stage lies in $\mathcal { T } _ { H } \setminus \mathcal { T } _ { H - 1 }$ , so none of its columns is among those generating $\mathcal { F } _ { T }$ . If the common output has norm at most B, the event $\mathcal { G }$ ensures that all its projections onto this block have magnitude at most $a .$ . If its norm exceeds $B _ { : }$ the distant-output condition applies instead. In either case, Lemma 5 gives $\| \nabla ( \Phi _ { U } ) _ { 1 / 2 } ( x _ { \mathrm { o u t } } ) \| > \sigma _ { K } / 4$

Thus, for every fixed algorithmic seed, the failure probability over U is at least $3 / 4$ . Averaging over the independent seed proves the joint probability bound. Averaging over U then yields a fixed embedding $U _ { 0 }$ with algorithmic failure probability at least $3 / 4$ □

The construction satisfies n, $H = O ( K N ^ { 2 } )$ and $B ^ { 2 } / a ^ { 2 } = O ( K ^ { 2 } N ^ { 5 } )$ . Hence the dimension condition in Proposition 3 can be met with $D = O ( K ^ { 2 } N ^ { 5 } \log ( P K N ) )$ . Increasing P increases the number of queries covered by the concentration estimate, but does not change which coordinate groups determine an auxiliary reply. It therefore enters this suficient dimension bound only through a logarithmic factor.

## 4.3 Small initial objective gaps

The stage construction has a normalized initial objective gap of at least 2. To cover smaller gaps, we use the base function $h _ { M , N }$ directly and vary the number of blocks M. Its initial gap is at most $3 M / N$ , while the round lower bound below is at least $M N / 1 2$ and the ME gradient threshold remains of order $N ^ { - 1 / 2 }$ . Thus, once N is chosen according to the target accuracy, varying M controls the gap while retaining a round lower bound that depends linearly on it.

Take integers $N \geq 6 0$ and $2 \leq M \leq N / 1 2$ . Set $n _ { s } : = M N , s : = \sqrt { M / N } , A _ { s } : = 3 4 s + 2 s ^ { 2 }$ , and $B _ { s } : = 7 3 s$ For $U \in \mathbb { R } ^ { D \times n _ { s } }$ satisfying $U ^ { \top } U = I _ { n _ { s } } .$ , define $\Psi _ { U } ( x ) : = \operatorname* { m a x } \{ h _ { M , N } ( U ^ { \top } x ) , \| x \| - A _ { s } \}$ . Proposition 1 implies that Ψ<sub>U</sub> is globally 1-Lipschitz and $3 / 4 – \mathrm { W C }$ . The radial branch is inactive at the origin, and inf $\Psi _ { U } \geq \operatorname* { i n f } h _ { M , N }$ so $\Psi _ { U } ( 0 ) - \operatorname* { i n f } \Psi _ { U } \leq 3 M / N$

Use the index sets $\mathcal { T } _ { t }$ and projections $Q _ { t }$ from Section 3.1. For this construction, small projections onto the second half of the last block sufice for the required ME gradient bound. Accordingly, set $H _ { s } : =$ $( M - 1 ) L + \lfloor N / 2 \rfloor$ . Then $H _ { s } \geq M N / 1 2$ , and the coordinates in the last block with $j > \lfloor N / 2 \rfloor$ all lie outside $\mathcal { I } _ { H _ { s } - 1 }$

Proposition 4 (Normalized lower bound for small gaps). For the above $M , N$ and any integer $P \geq 1$ , suppose

$$
D \geq n _ { s } + 2 + \frac { 2 B _ { s } ^ { 2 } } { a ^ { 2 } } \log \bigl ( 8 n _ { s } ( P H _ { s } + 1 ) \bigr ) , a = \frac { 1 } { 8 N ^ { 2 } } .
$$

If U is drawn uniformly from the orthonormal embeddings, independently of the algorithm’s internal randomness, then every randomized algorithm using at most P queries per round and at most $H _ { s }$ rounds on every execution satisfies

$$
\operatorname* { P r } _ { U , A } \left\{ \| \nabla ( \Psi _ { U } ) _ { 1 / 2 } ( \widehat { x } ) \| > \frac { 1 } { 1 6 \sqrt { N } } \right\} \geq \frac { 3 } { 4 } .
$$

Proof. We use two properties of $\Psi _ { U } .$ , proved in Appendix F. First, for $\lVert x \rVert \geq 7 2 s$ , the radial branch strictly dominates the base-function branch in both $\Psi _ { U }$ and its truncated versions max $\{ h _ { M , N } ( Q _ { t } U ^ { \top } x ) , \| x \| - \}$ $A _ { s } \}$ . Second, writing $u _ { M , \mathcal { I } }$ for the column of U corresponding to coordinate $j$ of the last block, we have $\| \nabla ( \Psi _ { U } ) _ { 1 / 2 } ( x ) \| > 1 / ( 1 6 \sqrt { N } )$ whenever $| \langle u _ { M , j } , x \rangle | \leq a$ for all $j > \lfloor N / 2 \rfloor$ . The same ME gradient bound holds whenever $\lVert x \rVert \geq B _ { s }$

Apply the auxiliary execution and coupling argument of Proposition 3, replacing $( \mathcal { T } _ { t } , \Pi _ { t } , \Phi _ { U } , n , H , B )$ with $( \mathcal { I } _ { t } , Q _ { t } , \Psi _ { U } , n _ { s } , H _ { s } , B _ { s } )$ . Lemma 1 gives equality of bounded-query replies, and radial dominance gives equality for distant queries.

After at most $H _ { s }$ rounds, the last-block coordinates with $j > \lfloor N / 2 \rfloor$ remain outside the index set determining the auxiliary output. Under the stated dimension condition, concentration and the union bound ensure that the ME gradient bound applies to the output with probability at least $3 / 4$ □

Since $B _ { s } ^ { 2 } / a ^ { 2 } = O ( M N ^ { 3 } )$ , the dimension condition in Proposition 4 can be met with $\begin{array} { r l } { D } & { { } = } \end{array}$ $O ( M N ^ { 3 } \log ( P N ) )$

## 4.4 Scaling and matching upper bounds

Define the dimensionless parameters $E : = G ^ { 2 } / \varepsilon ^ { 2 }$ and $r : = \rho \Delta / G ^ { 2 }$ . Then $\rho G ^ { 2 } \Delta / \varepsilon ^ { 4 } = r E ^ { 2 }$ and $G ^ { 6 } / \varepsilon ^ { 6 } = E ^ { 3 }$ For a normalized function ${ \widetilde { f } } ,$ define $f ( x ) : = ( G ^ { 2 } / \rho ) { \\widetilde { f } } ( ( \rho / G ) x )$ . If $\widetilde { f }$ has Lipschitz constant and weak convexity parameter at most 1, then the corresponding parameters of f are at most $G$ and $\rho .$ Moreover,

$$
\partial f ( x ) = G \partial \widetilde { f } \Big ( \frac { \rho } { G } x \Big ) , \nabla f _ { 1 / ( 2 \rho ) } ( x ) = G \nabla \widetilde { f } _ { 1 / 2 } \Big ( \frac { \rho } { G } x \Big ) .
$$

Query locations, function values, and entire subdiferentials can all be transformed by these known scalings. Thus, the transformation preserves both the number of rounds and the number of parallel queries per round. The normalized problem therefore has an initial gap limit of r and a target ME gradient norm of $E ^ { - 1 / 2 }$

Both constructions have ME gradient thresholds of order $N ^ { - 1 / 2 }$ , so we choose $N \asymp E$ . For larger normalized gaps, choosing $K \asymp \operatorname* { m i n } \{ r , E \}$ gives a round lower bound of order $K N ^ { 2 } \asymp \operatorname* { m i n } \{ r E ^ { 2 } , E ^ { 3 } \}$ . For smaller gaps, choosing $M \asymp \operatorname* { m i n } \{ r , 1 \} N$ gives the required order $r E ^ { 2 }$ . The restriction $K \leq N$ preserves the ME gradient threshold of order $N ^ { - 1 / \bar { 2 } }$ in the stage construction and causes its round lower bound to saturate at order $E ^ { 3 }$

Proof of Theorem 1. Choose $c _ { 0 } \leq 1 / 1 2 2 8 8 0$ . The accuracy condition implies $E \ge 1 2 2 8 8 0$ and $r E \geq 1 2 2 8 8 0 .$ First suppose $r \geq 8$ . Set $N : = \lfloor E / 1 0 2 4 \rfloor$ and $K : = \operatorname* { m i n } \{ \lfloor r / 4 \rfloor , N \}$ , and use $\widetilde { f } : = \Phi _ { U }$ . Then $N \geq 1 2 0$ $1 \le K \le N$ , and f(0) − inf $f \leq 3 K G ^ { 2 } / \rho \leq \Delta$ . The scaled ME gradient threshold satisfies

$$
{ \frac { G } { 1 6 { \sqrt { N + K - 1 } } } } \geq { \frac { G } { 1 6 { \sqrt { 2 N } } } } \geq { \sqrt { 2 } } \varepsilon .
$$

Since $N = \Theta ( E )$ and $K = \Theta ( \operatorname* { m i n } \{ r , E \} )$ , Proposition 3 gives

$$
H \geq \frac { K N ^ { 2 } } { 2 8 8 } = \Omega \big ( \operatorname* { m i n } \{ r E ^ { 2 } , E ^ { 3 } \} \big ) .
$$

Next suppose $r < 8 .$ . Set $r ^ { \prime } : = \operatorname* { m i n } \{ r , 1 \} , N : = \lfloor E / 2 5 6 \rfloor$ , and $M : = \lfloor r ^ { \prime } N / 1 2 \rfloor$ , and use $\widetilde { f } : = \Psi _ { U }$ . The accuracy condition ensures $N \ge 6 0 , 2 \le M \le N / 1 2 , N \ge E / 5 1 2$ , and $M \ge r ^ { \prime } N / 2 4$ . The objective gap satisfies

$$
f ( 0 ) - \operatorname* { i n f } f \leq \frac { G ^ { 2 } } { \rho } \frac { 3 M } { N } \leq \frac { r ^ { \prime } G ^ { 2 } } { 4 \rho } \leq \Delta ,
$$

and the scaled ME gradient threshold is $G / ( 1 6 \sqrt { N } ) \geq \varepsilon$ . Proposition 4 therefore gives

$$
{ { H } _ { s } } \ge \frac { { { M } { N } } } { 1 2 } \ge \frac { { { r } ^ { \prime } } { { N } ^ { 2 } } } { 2 8 8 } = \Omega ( { { r } ^ { \prime } } { { E } ^ { 2 } } ) = \Omega ( { { r } } { { E } ^ { 2 } } ) ,
$$

where the last equality uses $r ^ { \prime } \ge r / 8$ . Since $r < E$ in this case, this is the required order $\Omega ( \mathrm { m i n } \{ r E ^ { 2 } , E ^ { 3 } \} )$ . The suficient dimensions for the two constructions satisfy, respectively,

$$
D = O \big ( K ^ { 2 } N ^ { 5 } \log ( P K N ) \big ) = O \Big ( E ^ { 7 } \log ( P \sqrt { E } ) \Big ) , D = O \big ( M N ^ { 3 } \log ( P N ) \big ) = O \Big ( r ^ { \prime } E ^ { 4 } \log ( P \sqrt { E } ) \Big ) .
$$

Both imply the stated dimension bound.

For polynomial batch widths, choose $D : = \lceil C _ { k } E ^ { 7 } \log E \rceil$ . The intrinsic dimensions and round caps of both constructions are $O ( E ^ { 3 } )$ . For $P \leq D ^ { k }$ , we have log $( P E ) = O _ { k } ( \log E )$ , so a suficiently large $C _ { k }$ satisfies the concentration conditions simultaneously for all such $P .$

Finally, choose the universal constant $c > 0$ small enough that the budget in (1) does not exceed the applicable cap H or $H _ { s }$ . The two normalized propositions give joint failure probability at least $3 / 4$ . Averaging over the embedding yields a fixed instance with the same algorithmic failure bound. □

When $r \leq 1$ , equivalently $\Delta \le G ^ { 2 } / \rho _ { ; }$ the small-gap construction also gives the sharper suficient dimension $D = O ( ( \rho \Delta G ^ { 6 } / \varepsilon ^ { 8 } ) \log ( P G / \varepsilon ) )$

Proof of Corollary 1. For any $f \in { \mathcal { F } } _ { D } ( G , \rho , \Delta )$ , start from $x _ { 0 } : = 0$ , choose $\eta : = \sqrt { \Delta / ( \rho G ^ { 2 } T ) }$ , and perform T subgradient updates $x _ { t + 1 } : = x _ { t } - \eta g _ { t }$ with $g _ { t } \in \partial f ( x _ { t } )$ . Return x<sub>R</sub>, where R is uniform on $\{ 0 , \ldots , T - 1 \}$ and independent of the trajectory. The standard ME descent analysis [4, 5] gives

$$
\mathbb { E } \| \nabla f _ { 1 / ( 2 \rho ) } ( x _ { R } ) \| ^ { 2 } \leq 8 { \sqrt { \frac { \rho G ^ { 2 } \Delta } { T } } } .
$$

For $T \geq 1 0 2 4 \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 }$ , the right-hand side is at most $\varepsilon ^ { 2 } / 4$ . Markov’s inequality gives success probability at least $3 / 4$ , using one query per round. More generally, taking $T \geq 6 4 ( 1 - p ) ^ { - 2 } \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 }$ guarantees success probability at least $p$ for any fixed $p \in ( 1 / 2 , 1 )$

The condition $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ is equivalent to $r \leq E$ . In this regime, Theorem 1 gives the matching lower bound $\Omega ( r E ^ { 2 } ) = \Omega ( \rho G ^ { 2 } \Delta / \varepsilon ^ { 4 } )$ . This lower bound also applies to every target success probability $p > 1 / 2$ since the theorem gives a fixed instance on which the algorithm succeeds with probability at most $1 / 4 .$ . Its simultaneous dimension guarantee yields the conclusion for polynomial batch widths. □

The coupling argument converts the oracle locality and stationarity properties of the construction into a round lower bound for randomized parallel algorithms. After scaling, this bound matches the standard subgradient upper bound when $\Delta \le G ^ { 4 } / ( \rho \varepsilon ^ { 2 } )$ , under the stated accuracy condition. Thus, in this regime, polynomially many parallel queries per round do not improve the worst-case rate in suficiently high dimensions, even for randomized algorithms.

## 5 Optimal round complexity in one dimension

We next examine how the order structure of the real line permits faster stationarity search. Function-value comparisons allow an algorithm to shrink a search interval while ensuring that a regularized objective has an interior minimizer on the current interval. Parallel queries allow finer subdivisions in each round, and weak convexity converts the final interval length into an ME stationarity guarantee.

We use the class ${ \mathcal { F } } _ { 1 } ( G , \rho , \Delta )$ and oracle model from Section 1. Set $Q : = \rho \Delta / \varepsilon ^ { 2 }$ and $\lambda : = 1 / ( 2 \rho )$ Let $T _ { 1 } ^ { \star } ( G , \rho , \Delta , \varepsilon , P )$ denote the minimum worst-case number of rounds required to find a point satisfying $| f _ { \lambda } ^ { \prime } ( x ) | \leq \varepsilon$ with probability at least $3 / 4$ on every instance, using at most $P$ queries per round.

Theorem 2 (Optimal one-dimensional round complexity). There exist universal constants $c _ { 1 } > 0$ and $C _ { 1 } > 1$ such that, $i f 0 < \varepsilon \le c _ { 1 } G$ and $Q \geq C _ { 1 }$ , then, for every integer $P \geq 1$

$$
T _ { 1 } ^ { \star } ( G , \rho , \Delta , \varepsilon , P ) = \Theta \left( 1 + \frac { \log Q } { \log ( P + 1 ) } \right) ,
$$

with universal implicit constants. A deterministic algorithm using only exact function values attains the upper bound. The lower bound holds for randomized algorithms receiving exact function values and entire subdiferentials.

The same conclusion holds if $3 / 4$ in the definition of $T _ { 1 } ^ { \star }$ is replaced by any $p \in ( 1 / 2 , 1 ]$ , with implicit constants that can be chosen uniformly over all such $p .$

## 5.1 Interval search using function values

Set $\alpha : = \varepsilon / 4 , \delta : = \varepsilon / ( 4 \rho )$ , and $g ( x ) : = f ( x ) + \alpha | x |$ . The regularization serves two purposes: it converts the initial objective gap into a finite search interval, and it ensures that an interior local minimizer of $g$ has a small subgradient of $f .$ . The algorithm maintains three points $a < b < c$ satisfying $g ( b ) \leq \operatorname* { m i n } \{ g ( a ) , g ( c ) \}$ This condition guarantees that $g$ has an interior minimizer on $[ a , c ]$ . We shrink the interval while preserving this condition and stop once its length is small enough to guarantee approximate stationarity.

To initialize, let $R : = \operatorname* { m a x } \{ \delta , \Delta / \alpha \}$ . The gap bound gives $g ( \pm R ) \ge f ( 0 ) - \Delta + \alpha R \ge g ( 0 )$ . Query f(0) and set $( a , b , c ) : = ( - R , 0 , R )$ . The endpoint values need not be queried, since the required inequalities follow from the gap bound.

For $P \geq 2 .$ , query $t _ { j } : = a + j ( c - a ) / ( P + 1 ) , j = 1 , \ldots , P $ , in parallel. Choose $b ^ { \prime }$ to minimize $g$ among the old middle point b and the new query points. Retaining b among the candidates ensures $g ( b ^ { \prime } ) \leq g ( b )$ , so the inequalities involving the old endpoints remain valid. Sort the distinct points in $\{ a , b , t _ { 1 } , \ldots , t _ { P } , c \}$ , and let $a ^ { \prime }$ and $c ^ { \prime }$ be the immediate neighbors of $b ^ { \prime }$ . Then $g ( b ^ { \prime } ) \leq \operatorname* { m i n } \{ g ( a ^ { \prime } ) , g ( c ^ { \prime } ) \}$ } and $c ^ { \prime } - a ^ { \prime } \leq 2 ( c - a ) / ( P + 1 )$ ). Replacing the current triple by $( a ^ { \prime } , b ^ { \prime } , c ^ { \prime } )$ therefore preserves the invariant and contracts the interval.

For $P = 1$ , query the midpoint of the longer side. If $c - b \geq b - a$ , set $t : = ( b + c ) / 2$ and replace $( a , b , c )$ by $( b , t , c )$ when $g ( t ) < g ( b )$ , and by $( a , b , t )$ otherwise. Treat the left side symmetrically. Initially, the two sides have equal length, and each update keeps the ratio of the longer side to the shorter side at most 2. Indeed, writing $\ell : = b - a$ and $r : = c - b ,$ suppose $\ell \leq r \leq 2 \ell$ . The two possible updates give side lengths $( r / 2 , r / 2 )$ and $( \ell , r / 2 )$ , respectively. Both preserve the ratio bound, and their total lengths are at most $2 ( \ell + r ) / 3$ and $3 ( \ell + r ) / 4$ . Thus, each round contracts the interval length by a factor of at most $3 / 4$

Stop when $c - a \leq \delta$ and return b. Continuity and the invariant imply that $g$ has a minimizer $z \in ( a , c )$ over $[ a , c ]$ . If the minimum is below $g ( b )$ , neither endpoint can attain it; otherwise, b itself is an interior minimizer. Hence $0 \in \partial g ( z ) = \partial f ( z ) + \alpha \partial | \cdot | ( z )$ , which gives dist $\mathbf { \partial } _ { \cdot } ( 0 , \partial f ( z ) ) \leq \alpha$

For a $\rho { - } \mathrm { W C }$ function and $\lambda = 1 / ( 2 \rho )$ , the ME satisfies $| f _ { \lambda } ^ { \prime } ( z ) | \le 2 \mathrm { d i s t } ( 0 , \partial f ( z ) )$ , and its derivative is 2ρ-Lipschitz; see Appendix A. Consequently,

$$
| f _ { \lambda } ^ { \prime } ( b ) | \le | f _ { \lambda } ^ { \prime } ( z ) | + 2 \rho | b - z | \le 2 \alpha + 2 \rho \delta = \varepsilon .
$$

The initial-to-final length ratio is $2 R / \delta = \operatorname* { m a x } \{ 2 , 3 2 Q \}$ . Define $\theta _ { 1 } : = 3 / 4$ and $\theta _ { P } : = 2 / ( P + 1 )$ for $P \geq 2$ Including the initial query at the origin, the number of rounds satisfies

$$
T _ { \mathrm { u p } } \leq 1 + \left\lceil \frac { \log \operatorname* { m a x } \{ 2 , 3 2 Q \} } { \log ( 1 / \theta _ { P } ) } \right\rceil .
$$

For $P \geq 2$ , we have log $\begin{array} { r } { \langle ( P + 1 ) / 2 \rangle \geq \frac { 1 } { 3 } \log ( P + 1 ) } \end{array}$ ). Together with the contraction factor for $P = 1$ , this proves the upper bound in Theorem 2. The algorithm uses only function values and does not require knowledge of $G .$

## 5.2 Hard instances on nested intervals

For the lower bound, we encode binary strings by nested intervals to hide the location of approximate stationary points. Such constructions also appear in the one-dimensional randomized lower bounds of Kornowski and Shamir [10]. Each bit selects one of two child intervals, and subsequent changes to the function are confined to the selected child.

A construction of depth n will produce $2 ^ { n }$ disjoint sets of approximate stationary points, with an initial objective gap of order $4 ^ { n } \varepsilon ^ { 2 } / \rho .$ . The gap budget therefore permits $n = \Theta ( \log Q )$ , and locating an approximate stationary point requires information about the corresponding hidden interval.

Fix an integer $n \geq 1$ , and set $B : = 4 0 \varepsilon$ and $L _ { j } : = 1 6 \cdot 4 ^ { n - j } B / \rho$ for $j = 0 , \ldots , n$ . For each $\omega =$ $( \omega _ { 1 } , \ldots , \omega _ { n } ) \in \{ 0 , 1 \} ^ { n }$ , construct nested closed intervals $I _ { 0 } \supset I _ { 1 } \supset \dots \supset I _ { n }$ , starting from $I _ { 0 } : = [ 0 \AA , L _ { 0 } ]$ . If $I _ { j } = [ a , a + L _ { j } ]$ , define

$$
I _ { j + 1 } : = \left\{ \begin{array} { l l } { [ a + L _ { j } / 8 , ~ a + 3 L _ { j } / 8 ] , } & { \omega _ { j + 1 } = 0 , } \\ { [ a + 5 L _ { j } / 8 , ~ a + 7 L _ { j } / 8 ] , } & { \omega _ { j + 1 } = 1 . } \end{array} \right.
$$

The two candidate intervals are disjoint and have length $L _ { j + 1 } = L _ { j } / 4$

Assign the value $h _ { j }$ to both endpoints of $I _ { j }$ , where $h _ { 0 } : = 0$ and $h _ { j + 1 } : = h _ { j } - B L _ { j } / 1 6$ . For a parent interval $I _ { j } = [ a , c ]$ and its selected child $I _ { j + 1 } = [ a ^ { \prime } , c ^ { \prime } ]$ , use continuously diferentiable, piecewise quadratic functions on $[ a , a ^ { \prime } ]$ and $[ c ^ { \prime } , c ]$ to match the prescribed endpoint values. Their derivatives satisfy $- B \le f _ { \omega } ^ { \prime } ( x ) \le - B / 2 0$ on $( a , a ^ { \prime } )$ and $B / 2 0 \le f _ { \omega } ^ { \prime } ( x ) \le B$ on $( c ^ { \prime } , c )$ . Their smooth pieces have second derivative $0 \ \mathrm { o r } \ - \rho ,$ and the one-sided derivatives at the child endpoints satisfy $f _ { \omega , - } ^ { \prime } ( a ^ { \prime } ) = - B$ and $f _ { \omega , + } ^ { \prime } ( c ^ { \prime } ) = B$ . Each connecting function depends only on the parent and selected child, not on later bits. Every nonterminal interval has length at least 64 $B / \rho ,$ so these conditions can be realized by linear and quadratic pieces. Explicit formulas are given in Appendix G.

Write $I _ { n } = [ a _ { n } , c _ { n } ]$ and define $m _ { \omega } : = ( a _ { n } + c _ { n } ) / 2$ . On $I _ { n } .$ , set $f _ { \omega } ( x ) : = h _ { n } + B ( | x - m _ { \omega } | - L _ { n } / 2 )$ . Outside $I _ { 0 } ,$ , set $f _ { \omega } ( x ) : = - B x$ for $x \leq 0$ and $f _ { \omega } ( x ) : = B ( x - L _ { 0 } )$ for $x \geq L _ { 0 }$ . This defines $f _ { \omega }$ on the whole real line.

Lemma 6 (Properties of the hard instances). Each $f _ { \omega }$ is $\rho { - } W C$ and B-Lipschitz, with unique minimizer $m _ { \omega }$ and objective gap $f _ { \omega } ( 0 )$ −inf $f _ { \omega } = ( 4 / 3 ) ( 4 ^ { n } + 5 ) B ^ { 2 } / \rho$ . For every x $\neq m _ { \omega }$ , we have dist $( 0 , \partial f _ { \omega } ( x ) ) \geq B / 2 0 = 2 \varepsilon$ while $\partial f _ { \omega } ( m _ { \omega } ) = [ - B , B ]$ . Moreover, the set of approximate stationary points is exactly

$$
S _ { \omega } : = \left\{ x : | ( f _ { \omega } ) _ { \lambda } ^ { \prime } ( x ) | \leq \varepsilon \right\} = \left[ m _ { \omega } - \frac { \varepsilon } { 2 \rho } , m _ { \omega } + \frac { \varepsilon } { 2 \rho } \right] .
$$

Each $S _ { \omega }$ lies strictly inside its terminal interval $I _ { n . }$ , and these sets are pairwise disjoint.

Proof. The connecting pieces have curvature at least $- \rho ,$ and all derivative jumps are upward, so $f _ { \omega } + ( \rho / 2 ) | \cdot | ^ { 2 }$ is convex. The slopes have magnitude at most B. The function is strictly decreasing to the left of $m _ { \omega }$ and strictly increasing to its right, with subgradient magnitude at least $B / 2 0$ away from $m _ { \omega }$ . The verification at the joining points is included in Appendix G.

The recursion for the endpoint values gives $h _ { n } = - ( 4 / 3 ) ( 4 ^ { n } - 1 ) B ^ { 2 } / \rho$ . Since $f _ { \omega } ( m _ { \omega } ) = h _ { n } - B L _ { n } / 2$ and $B L _ { n } / 2 = 8 B ^ { 2 } / \rho ,$ the stated objective gap follows.

To identify the stationary set, let $y : = \mathrm { p r o x } _ { \lambda f _ { \omega } } ( x )$ . Proximal optimality gives $( f _ { \omega } ) _ { \lambda } ^ { \prime } ( x ) = ( x - y ) / \lambda \in \partial f _ { \omega } ( y )$ If this derivative has magnitude at most $\varepsilon ,$ the subgradient bound forces $y = m _ { \omega }$ , and hence $| x - m _ { \omega } | \leq \lambda \varepsilon$ Conversely, this distance condition implies $( x - m _ { \omega } ) / \lambda \in [ - \varepsilon , \varepsilon ] \subseteq \partial f _ { \omega } ( m _ { \omega } )$ . Strong convexity of the proximal subproblem then gives $y = m _ { \omega }$ , proving the formula for $S _ { \omega }$ . Since $\lambda \varepsilon < L _ { n } / 2$ and distinct binary strings yield disjoint terminal intervals, the remaining claims follow. □

The construction also has an oracle locality property: once the first $j$ bits are fixed, both the function value and the entire subdiferential are determined at every point in $\mathbb { R } \backslash I _ { j }$ , independently of the remaining bits.

## 5.3 Information bounds for randomized parallel queries

Draw $\omega$ uniformly from {0, 1}<sup>n</sup>, independently of the algorithm’s internal randomness. We analyze an augmented oracle that reveals a prefix of ω in addition to the original oracle replies.

Suppose the prefix known at the start of a round is $\left( \omega _ { 1 } , \ldots , \omega _ { k } \right)$ , with corresponding interval $I _ { k }$ . All query points in the batch are chosen before any replies are received. For each query point $x ,$ start from this same prefix and reveal successive bits while x belongs to the current closed interval and unrevealed bits remain. Each revealed bit selects the next child interval. The process stops when x lies outside the current interval or all n bits have been revealed.

For the whole batch, the oracle returns the longest of the prefixes required by the individual queries. This prefix contains all the others and determines the exact function values and entire subdiferentials at every query point. The algorithm can use it in subsequent rounds. Closed intervals account for endpoint queries, whose subdiferentials may depend on the next connecting function. Since the augmented oracle provides more information than the original oracle, a lower bound for the augmented model also applies to the original model.

Lemma 7 (Information per round and success probability). Every randomized algorithm using the augmented oracle with at most P queries per round and at most T rounds on every execution satisfies

$$
\operatorname* { P r } _ { \omega , A } \{ \widehat { x } \in S _ { \omega } \} \leq \operatorname* { m i n } \left\{ 1 , \frac { T ( \log _ { 2 } P + 2 ) + 2 } { n } \right\} .
$$

A zero-round algorithm has average success probability at most $2 ^ { - n }$

Proof. Let H denote the history at the start of a round, including the algorithm’s random seed and the revealed prefix. Conditional on H, all query points in the current batch are fixed. Each revelation and stopping decision depends only on the query points and the bits already revealed, so the unrevealed bits remain independent and uniform.

For a fixed query point $x ,$ let $J _ { x }$ be the number of additional bits required, measured from the prefix at the start of the round. To reveal at least $j$ bits, the first $j - 1$ child selections must all choose an interval containing x. Since the two candidate children are disjoint, $\operatorname* { P r } ( J _ { x } \geq j \mid \mathcal { H } ) \leq 2 ^ { - ( j - 1 ) }$ for every $j \geq 1$

The additional prefix length for the batch is $J : = \operatorname* { m a x } _ { x } J _ { x }$ , where the maximum is over its query points. A union bound gives $\operatorname* { P r } ( J \geq j \mid \mathcal { H } ) \leq \operatorname* { m i n } \{ 1 , P 2 ^ { - ( j - 1 ) } \}$ . The tail-sum formula therefore yields

$$
\mathbb { E } [ J \mid \mathcal { H } ] \le \sum _ { j \ge 1 } \operatorname* { m i n } \{ 1 , P 2 ^ { - ( j - 1 ) } \} \le \log _ { 2 } P + 2 .
$$

The output need not be a previously queried point. To account for this, perform one additional prefix query at $\widehat { x }$ after termination, solely for the analysis. This query is not counted as an algorithmic round. The single-query tail bound shows that it reveals at most 2 additional bits in conditional expectation. If the algorithm succeeds, Lemma 6 places $\widehat { x }$ strictly inside the terminal interval, so all n bits have been revealed after this additional query.

Let Z denote the total number of revealed bits after the additional query. Averaging the conditional bounds over the histories and the algorithm’s random seed gives

$$
n \operatorname* { P r } _ { \omega , A } \{ \widehat { x } \in S _ { \omega } \} \leq \mathbb { E } _ { \omega , A } Z \leq T ( \log _ { 2 } P + 2 ) + 2 .
$$

Together with the trivial upper bound of 1 on success probability, this proves the first claim.

With zero rounds, the output is independent of $\omega .$ Since the $2 ^ { n }$ sets $S _ { \omega }$ are pairwise disjoint, any fixed output belongs to at most one of them. Averaging over the output gives success probability at most $2 ^ { - n }$ . □

Completion of the proof of Theorem 2. The upper bound was established in Section 5.1. For the lower bound, we may take $c _ { 1 } : = 1 / 4 0$ and $C _ { 1 } : = 1 9 2 0 0$ . Choose $n : = \lfloor \log _ { 4 } ( 3 Q / 6 4 0 0 - 5 ) \rfloor$ . Since $Q \geq 1 9 2 0 0$ , we have $n \geq 1$ and $n = \Theta ( \log Q )$ . With $B = 4 0 \varepsilon$ , Lemma 6 gives

$$
f _ { \omega } ( 0 ) - \operatorname* { i n f } f _ { \omega } = \frac { 6 4 0 0 } { 3 } ( 4 ^ { n } + 5 ) \frac { \varepsilon ^ { 2 } } { \rho } \leq \Delta .
$$

Also, $B \leq G ,$ , so every constructed function belongs to $\mathcal { F } _ { 1 } ( G , \rho , \Delta )$

Fix any $p \in ( 1 / 2 , 1 ]$ . If an algorithm succeeds with probability at least p on every instance, its average success probability under the uniform distribution on ω is at least p. Lemma 7 implies $T \geq ( p n - 2 ) / ( \log _ { 2 } P + 2 )$ For $n \geq 8 ,$ we have $p n - 2 \geq n / 4$ and log $P + 2 \leq 2 \log _ { 2 } ( P + 1 )$ , giving $T \geq n / ( 8 \log _ { 2 } ( P + 1 ) )$

The zero-round bound also gives $T \geq 1$ , since $2 ^ { - n } \leq 1 / 2 < p .$ . This constant lower bound covers $1 \leq n < 8$ as well. Consequently,

$$
T = \Omega \left( 1 + \frac { n } { \log ( P + 1 ) } \right) = \Omega \left( 1 + \frac { \log Q } { \log ( P + 1 ) } \right) .
$$

The constants are uniform over $p \in ( 1 / 2 , 1 ]$ . The upper bound is deterministic and thus has success probability 1, proving both the theorem and its extension to these success probabilities. □

These bounds show how the order structure of the real line allows function-value comparisons to shrink a search interval, with parallel queries accelerating this reduction. Regularization and weak convexity convert the interval length into an ME stationarity guarantee. The matching lower bound shows that this function-value search attains the optimal worst-case round complexity up to universal constants, even among randomized algorithms with access to the entire subdiferential.

## 6 Conclusion

We show that the standard subgradient rate remains optimal for randomized parallel algorithms in suficiently high dimensions within the stated parameter regime. In one dimension, the order structure of the real line permits faster search, and we characterize precisely the reduction in round complexity provided by parallel queries. A natural open problem is to characterize the dimension threshold at which the subgradient rate becomes unavoidable. For a prescribed accuracy, initial objective gap, and parallel query budget, this requires determining how large the dimension must be to sustain the matching lower bound, and what faster rates are attainable below that threshold. Such a characterization would clarify how low-dimensional geometry and parallel queries jointly afect the complexity of stationarity search.

## References

[1] Sébastien Bubeck and Dan Mikulincer. How to trap a gradient flow. In Proceedings of the Thirty-Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 940–960. PMLR, 2020. URL https://proceedings.mlr.press/v125/bubeck20b.html.

[2] Yair Carmon, John C. Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points I. Mathematical Programming, 184(1–2):71–120, 2020. doi: 10.1007/s10107-019-01406-y.

[3] Sinho Chewi, Sébastien Bubeck, and Adil Salim. On the complexity of finding stationary points of smooth functions in one dimension. In Proceedings of the 34th International Conference on Algorithmic Learning Theory, volume 201 of Proceedings of Machine Learning Research, pages 358–374. PMLR, 2023. URL https://proceedings.mlr.press/v201/chewi23a.html.

[4] Damek Davis and Dmitriy Drusvyatskiy. Stochastic subgradient method converges at the rate $O ( k ^ { - 1 / 4 } )$ on weakly convex functions. arXiv preprint arXiv:1802.02988, 2018. URL https://arxiv.org/abs/1802.02988.

[5] Damek Davis and Dmitriy Drusvyatskiy. Stochastic model-based minimization of weakly convex functions. SIAM Journal on Optimization, 29(1):207–239, 2019. doi: 10.1137/18M1178244.

[6] Damek Davis and Benjamin Grimmer. Proximally guided stochastic subgradient method for nonsmooth, nonconvex problems. SIAM Journal on Optimization, 29(3):1908–1930, 2019. doi: 10.1137/17M1151031.

[7] Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A gradient sampling method with complexity guarantees for Lipschitz functions in high and low dimensions. In Advances in Neural Information Processing Systems, volume 35, pages 6692–6703. Curran Associates, Inc., 2022. doi: 10.52202/068431-0485. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ha sh/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html.

[8] Jelena Diakonikolas and Cristóbal Guzmán. Lower bounds for parallel and randomized convex optimization. Journal of Machine Learning Research, 21(5):1–31, 2020. URL https://www.jmlr.org/papers/v21/ 19-771.html.

[9] John C. Duchi and Feng Ruan. Stochastic methods for composite and weakly convex optimization problems. SIAM Journal on Optimization, 28(4):3229–3259, 2018. doi: 10.1137/17M1135086.

[10] Guy Kornowski and Ohad Shamir. On the complexity of finding small subgradients in nonsmooth optimization, 2022. URL https://arxiv.org/abs/2209.10346v2. Revised April 2025.

[11] Jiajin Li and Siyu Pan. Optimal deterministic oracle complexity for weakly convex optimization. arXiv preprint arXiv:2608.03246, 2026. URL https://arxiv.org/abs/2608.03246.

[12] Arkadi Nemirovski. On parallel complexity of nonsmooth convex optimization. Journal of Complexity, 10(4):451–463, 1994. doi: 10.1006/jcom.1994.1025.

[13] Jean-Philippe Vial. Strong and weak convexity of sets and functions. Mathematics of Operations Research, 8(2):231–259, 1983. doi: 10.1287/moor.8.2.231.

[14] Huanjian Zhou and Taiji Suzuki. On the complexity of finding stationary points of p-th order Hölder smooth functions in one dimension. arXiv preprint arXiv:2610.16295, 2026. URL https://arxiv.org/abs/ 2610.16295.

[15] Huanjian Zhou, Andi Han, Akiko Takeda, and Masashi Sugiyama. The adaptive complexity of finding a stationary point. In Proceedings of the Thirty-Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 6091–6123. PMLR, 2025. URL https://proceedings.ml r.press/v291/huanjian25a.html.

## A Moreau-envelope estimates

Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be finite-valued and $\rho { - } \mathrm { W C } ,$ , and set $\lambda : = 1 / ( 2 \rho )$ . Write $p ( x ) : = \mathrm { p r o x } _ { \lambda f } ( x )$ and $v ( x ) : = \nabla f _ { \lambda } ( x )$ The proximal objective $f ( y ) + \rho \| y - x \| ^ { 2 }$ is strongly convex and coercive, so $p ( x )$ is uniquely defined. We

establish the two estimates used in the one-dimensional upper bound:

$$
\begin{array} { r } { \| v ( x ) \| \leq 2 \mathrm { d i s t } ( 0 , \partial f ( x ) ) , \| v ( x ) - v ( y ) \| \leq 2 \rho \| x - y \| , } \end{array}
$$

where $\mathrm { d i s t } ( 0 , C ) : = \operatorname* { i n f } _ { g \in C } \| g \|$

Proof of the envelope estimates. Convexity of $f + ( \rho / 2 ) \| \cdot \| ^ { 2 }$ implies

$$
\begin{array} { r } { \langle g - h , x - y \rangle \geq - \rho \| x - y \| ^ { 2 } \mathrm { ~ f o r ~ } g \in \partial f ( x ) , \ h \in \partial f ( y ) . } \end{array}
$$

Proximal optimality gives $2 \rho ( x - p ( x ) ) \in \partial f ( p ( x ) )$ . Applying the preceding inequality at $p ( x )$ and $p ( y )$ yields $\langle x - y , p ( x ) - p ( y ) \rangle \geq { \frac { 1 } { 2 } } \| p ( x ) - p ( y ) \| ^ { 2 }$ . In particular, $p$ is 2-Lipschitz. The unique-minimizer diferentiation formula therefore gives $v ( x ) = 2 \rho ( x - p ( x ) )$ , and

$$
\| v ( x ) - v ( y ) \| ^ { 2 } = 4 \rho ^ { 2 } \| ( x - y ) - ( p ( x ) - p ( y ) ) \| ^ { 2 } \leq 4 \rho ^ { 2 } \| x - y \| ^ { 2 } .
$$

Next, for any $g \in \partial f ( x )$ , apply the same subgradient inequality at x and $p ( x )$ . Substituting $x - p ( x ) = v ( x ) / ( 2 \rho )$ gives ${ \frac { 1 } { 2 } } \| v ( x ) \| ^ { 2 } \leq \langle g , v ( x ) \rangle \leq \| g \| \| v ( x ) \|$ ∥. Taking the infimum over $g$ proves the first estimate, including the case $v ( x ) = 0$ □

The same identities give the subgradient upper bound used in Corollary 1. Suppose also that $f$ is G-Lipschitz and $f ( x _ { 0 } ) -$ inf $f \leq \Delta$ . For $g _ { t } \in \partial f ( x _ { t } )$ and $x _ { t + 1 } : = x _ { t } - \eta g _ { t }$ , let $p _ { t } : = p ( x _ { t } )$ . Comparing the proximal objective at $p _ { t }$ and $x _ { t } ,$ and then using weak convexity at $x _ { t } ,$ gives $\langle g _ { t } , x _ { t } - p _ { t } \rangle \geq ( \rho / 2 ) \| x _ { t } - p _ { t } \| ^ { 2 }$ Evaluating the next envelope at $p _ { t }$ consequently yields

$$
\begin{array} { l } { f _ { \lambda } ( x _ { t + 1 } ) \le f _ { \lambda } ( x _ { t } ) - 2 \rho \eta \langle g _ { t } , x _ { t } - p _ { t } \rangle + \rho \eta ^ { 2 } \| g _ { t } \| ^ { 2 } } \\ { \quad \quad \le f _ { \lambda } ( x _ { t } ) - \displaystyle \frac { \eta } { 4 } \| v ( x _ { t } ) \| ^ { 2 } + \rho \eta ^ { 2 } G ^ { 2 } . } \end{array}
$$

Since $f _ { \lambda } ( x _ { 0 } ) - \operatorname* { i n f } f \leq \Delta$ , summing over $t = 0 , \ldots , T - 1$ and choosing R uniformly from these indices gives $\mathbb { E } \Vert v ( x _ { R } ) \Vert ^ { 2 } \leq 4 \Delta / ( \eta T ) + 4 \rho \eta G ^ { 2 }$ . The choice $\eta : = \sqrt { \Delta / ( \rho G ^ { 2 } T ) }$ gives the bound $8 \sqrt { \rho G ^ { 2 } \Delta / T }$ stated in the main text.

## B Properties of the base function

Proof of Proposition 1. Write $h : = h _ { M , N }$ and $\phi _ { r } : = \phi ( u ^ { ( r ) } )$ . Let $\Lambda : = \{ \lambda \in \mathbb { R } ^ { M } : \lambda \geq 0 , \ \| \lambda \| \leq 1 \}$ and set $\lambda _ { M + 1 } : = 0$ . The support-function identity $\| [ w ] _ { + } \| = \operatorname* { m a x } _ { \lambda \in \Lambda } \langle \lambda , w \rangle$ gives $h ( u ) = \operatorname* { m a x } _ { \lambda \in \Lambda } h _ { \lambda } ( u )$ , where

$$
h _ { \lambda } ( u ) : = \frac { 2 } { N } \left[ \sum _ { r = 1 } ^ { M } \lambda _ { r } \phi _ { r } - \sum _ { r = 1 } ^ { M } ( 1 - \lambda _ { r + 1 } ) p _ { r } + \lambda _ { 1 } - \frac { 5 } { 4 } \sum _ { r = 1 } ^ { M } \lambda _ { r } \right] .
$$

The derivative $q ^ { \prime }$ is globally 6-Lipschitz and satisfies $0 \leq q ^ { \prime } \leq 3 / 2$ . Hence $\nabla \psi$ is $3 N / 8 – \mathrm { L i p s c h i t z } .$ , with nonnegative coordinates bounded by $3 / 8 .$ . Since $0 \leq 1 - \lambda _ { r + 1 } \leq 1$ , each $h _ { \lambda }$ is $\mathrm { 3 / 4 – W C }$ : the afine maxima are convex, and the smooth terms act on disjoint blocks. Taking the maximum preserves this weak convexity parameter.

Let $\Lambda ( u ) : = \arg \operatorname* { m a x } _ { \lambda \in \Lambda } \langle \lambda , \sigma ( u ) \rangle$ . Applying the subdiferential rule for a compact maximum after adding the common convexifying quadratic shows that $\partial h ( { u } )$ is the convex hull of the vectors g with block components

$$
g ^ { ( r ) } = \frac { 2 } { N } \left[ \lambda _ { r } \xi _ { r } - ( 1 - \lambda _ { r + 1 } ) \nabla \psi ( u ^ { ( r ) } ) \right] , r \in [ M ] ,
$$

where $\lambda \ \in \ \Lambda ( u )$ and $\xi _ { r } ~ \in ~ \partial \phi ( u ^ { ( r ) } )$ . Every $\xi _ { r }$ is a convex combination of the active vectors among $0 , - ( N / 8 ) e _ { 1 } , \ldots , - ( N / 8 ) e _ { N }$ , where $e _ { j }$ denotes the jth coordinate vector. Thus all coordinates of every subgradient of h are nonpositive. Also, $\| \xi _ { r } \| \le N / 8$ and $\lVert \nabla \psi ( u ^ { ( r ) } ) \rVert \leq 3 \sqrt { N } / 8$ , giving

$$
\| g \| \leq \frac { 2 } { N } \left( \frac { N } { 8 } \| \lambda \| + \frac { 3 } { 8 } \sqrt { M N } \right) \leq \frac { 1 } { 4 } + \frac { 3 } { 4 } \sqrt { \frac { M } { N } } < \frac { 1 } { 2 } .
$$

The same bound holds for convex combinations and, equivalently, uniformly bounds the Lipschitz constants of the functions $h _ { \lambda }$ . It therefore bounds $\operatorname { L i p } ( h )$

If $S ( u ) > 0 ;$ the maximizing weight is unique: $\lambda = [ \sigma ( u ) ] _ { + } / S ( u )$ , with $\| \lambda \| = 1$ . Whenever $\lambda _ { r } > 0$ , we have $\sigma _ { r } > 0$ and $\phi _ { r } > 5 / 4 - p _ { r - 1 } \geq 1 / 4$ . The zero branch of $\phi _ { r }$ is then inactive, so $\xi _ { r } = - ( N / 8 ) w _ { \eta }$ for some $w _ { r } \geq 0$ with $\begin{array} { r } { \sum _ { j } ( w _ { r } ) _ { j } = 1 } \end{array}$ . Consequently, $\| \xi _ { r } \| \ge \sqrt { N } / 8$ . The two terms in each block of $g$ have the same coordinatewise $\mathrm { s i g n } .$ and hence

$$
\| g \| ^ { 2 } \geq \frac { 4 } { N ^ { 2 } } \sum _ { r = 1 } ^ { M } \lambda _ { r } ^ { 2 } \| \xi _ { r } \| ^ { 2 } \geq \frac { 1 } { 1 6 N } \sum _ { r = 1 } ^ { M } \lambda _ { r } ^ { 2 } = \frac { 1 } { 1 6 N } .
$$

For this fixed $\lambda ,$ the displayed subgradient representation already defines a convex set, so the estimate holds for every $g \in \partial h ( u )$

Since $S \geq 0$ and $0 \leq p _ { r } \leq 1$ , we have $h \geq - 2 M / N$ . At the origin, $p _ { r } = 0$ for $r \geq 1 , \sigma _ { 1 } = c _ { 1 } - 1 / 4 > 0$ , and   
$\sigma _ { r } = c _ { 1 } - 5 / 4 < 0$ for $r \geq 2$ . Thus $h ( 0 ) = 2 ( c _ { 1 } - 1 / 4 ) / N \in [ 0 , 2 / N )$ , and h(0) − inf $h \leq ( 2 M + 2 ) / N \leq 3 M / N$ Finally, suppose $S ( u ) = 0$ . Since $p _ { 0 } ~ = ~ 1$ , the inequality $\sigma _ { 1 } \leq 0$ implies $\phi _ { 1 } \leq 1 / 4$ , and therefore   
$u _ { i } ^ { ( 1 ) } \geq ( 8 / N ) ( c _ { j } - 1 / 4 ) \geq 6 / N$ for every $j .$ . The arguments of q are then at least $3 / 2 - 1 / ( 1 6 N ) > 1$ , so $p _ { 1 } = 1$   
Induction gives $p _ { r } = 1$ for every r. In particular, $p _ { M } < 2 7 / 3 2$ implies $S ( u ) > 0$ □

## C Exact locality of the base oracle

Proof of Lemma 1. Set $\bar { a } : = 3 a / 2$ . Let $\mathcal { T } _ { t }$ be the open set of all v satisfying $| v _ { j } ^ { ( r ) } | < \bar { a }$ whenever $( r , j ) \notin \mathcal { I } _ { t - 1 }$ Both u and $Q _ { t } u$ belong to $\mathcal { T } _ { t }$ . We show that $h _ { M , N } ( v ) = h _ { M , N } ( Q _ { t } v )$ throughout this set.

For each block, let $k _ { r } : = | \{ j : ( r , j ) \in \mathcal { I } _ { t - 1 } \} | . \mathrm { ~ I f ~ } j > k _ { r }$ , then $v _ { j } ^ { ( r ) } < \bar { a } < 2 a .$ so its contribution to the progress function and its derivative is zero. In particular, all progress values are preserved by $Q _ { t }$ . Suppose $0 < k _ { r } < N$ . The coordinate $k _ { r } + 1$ is retained by $Q _ { t }$ , and for every $j > k _ { r } + 1$

$$
\left( c _ { k _ { r } + 1 } - \frac { N } { 8 } v _ { k _ { r } + 1 } ^ { ( r ) } \right) - \left( c _ { j } - \frac { N } { 8 } v _ { j } ^ { ( r ) } \right) \geq \frac { 1 } { 1 6 N } - \frac { N } { 4 } \bar { a } = \frac { 1 } { 6 4 N } > 0 .
$$

Thus every deleted afine branch is strictly below a retained branch. The same inequality holds after projection, so $\phi ( v ^ { ( r ) } )$ is unchanged. A block with $k _ { r } = N$ is retained in full. $\mathrm { I f } \ r = 1$ and $k _ { 1 } = 0$ , then $t = 0 ;$ the first coordinate is retained, and the same comparison shows that its afine branch strictly dominates all deleted branches.

It remains to consider $k _ { r } = 0$ with $r \geq 2$ . At most L coordinates of block $r - 1$ belong to $\mathcal { I } _ { t - 1 }$ , giving $p _ { r - 1 } ( v ) \leq L / N \leq 3 / 1 6$ . All coordinates of block r have magnitude below ${ \bar { a } } ,$ and hence

$$
\sigma _ { r } ( v ) \leq c _ { 1 } + \frac { N } { 8 } \bar { a } + \frac { 3 } { 1 6 } - \frac { 5 } { 4 } = - \frac { 5 } { 1 2 8 N } < 0 .
$$

This estimate also holds at $Q _ { t } v$ . Such a block contributes zero to both $[ \sigma ] _ { + }$ and its own progress value. These cases prove the required identity on $\mathcal { T } _ { t }$

Local equality and the subdiferential rule for linear composition yield $\partial h _ { M , N } ( u ) = Q _ { t } \partial h _ { M , N } ( Q _ { t } u )$ Applying the same identity at $Q _ { t } u$ gives $\partial h _ { M , N } ( Q _ { t } u ) = Q _ { t } \partial h _ { M , N } ( Q _ { t } u )$ . Combining the two equalities proves both the equality of the full subdiferentials and their inclusion in range $Q _ { t }$ □

## D Properties of a single stage

Proof of Lemma 2. Write $\begin{array} { r } { R _ { N } ( u ) : = \frac { 1 } { 4 } \lVert [ - u - 2 a \mathbf { 1 } ] _ { + } \rVert } \end{array}$ and $s : = \langle e , u \rangle$ , so that $b _ { N } ( u ) = h _ { N } ( u ) + R _ { N } ( u ) - \tau _ { N } ( s )$ The function $R _ { N }$ is convex and 1/4-Lipschitz, whereas $0 \le \tau _ { N } ^ { \prime } \le 1 / 4$ and $\mathrm { L i p } ( \tau _ { N } ^ { \prime } ) = 1 / 2 0$ . Proposition 1 therefore gives weak convexity parameter $3 / 4 + 1 / 2 0 = 4 / 5$ and Lipschitz constant at most $3 / 4 + { \sqrt { 3 } } / 8 < 3 1 / 3 2$ The subdiferential sum rule gives $g = g _ { h } + g _ { R } - \tau _ { N } ^ { \prime } ( s ) e$ , with $g _ { h } \in \partial h _ { N } ( u )$ and $g _ { R } \in \partial R _ { N } ( u )$ . All three summands are componentwise nonpositive, so the norm of their sum is at least the norm of each summand.

If $S ( u ) > 0$ , Proposition 1 gives $\| g _ { h } \| \ge 1 / ( 4 \sqrt { N } )$ . If $S ( u ) = 0$ , the proof of Proposition 1 gives $u _ { j } ^ { ( r ) } \geq 6 / N$ in every block. Thus $s \geq 6 / \sqrt { N }$ and $\tau _ { N } ^ { \prime } ( s ) \geq 1 / ( 4 \sqrt { N } )$ , proving the global subgradient bound in this case as well.

If $R _ { N } ( u ) = 0$ , every coordinate is at least $- 2 a ,$ , so $\phi ( u ^ { ( r ) } ) \leq 1 7 / 1 6$ and $\sigma _ { r } \leq 1 3 / 1 6 < 1$ . Consequently, $b _ { N } ( u ) \leq h _ { N } ( u ) \leq 2 \sqrt { m } / N < 1 / 4$ . Hence $b _ { N } ( u ) \geq 1 / 4$ forces $R _ { N } ( u ) > 0$ , where $R _ { N }$ is diferentiable and $\| \nabla R _ { N } ( u ) \| = 1 / 4$ . Conversely, $h _ { N } \geq - 2 m / N \geq - 1 / 6$ and $R _ { N } \geq 0$ imply that $b _ { N } ( u ) \leq - 1$ forces $\tau _ { N } ( s ) \geq 5 / 6$ $\mathrm { S i n c e } \tau _ { N } ( 1 / \sqrt { N } + 5 ) = 5 / 8$ , this lies in the linear part of $\tau _ { N }$ , where $\tau _ { N } ^ { \prime } ( s ) = 1 / 4$ . The common coordinatewise sign proves $\| g \| \ge 1 / 4$ outside $( - 1 , 1 / 4 )$

It remains to establish the inner-product estimate. Use the subgradient representation from Appendix B, with $\lambda \geq 0 , \| \lambda \| \leq 1 , \lambda _ { m + 1 } : = 0$ , and $\xi _ { r } \in \partial \phi ( u ^ { ( r ) } )$ . Each such representing vector w has blocks $w ^ { ( r ) } =$ $( 2 / N ) [ \lambda _ { r } \xi _ { r } - ( 1 - \lambda _ { r + 1 } ) \nabla \psi ( u ^ { ( r ) } ) ]$ ], and $\begin{array} { r } { \sum _ { r } \lambda _ { r } \sigma _ { r } = S ( u ) } \end{array}$ . The afine branches defining ϕ give $\langle \xi _ { r } , u ^ { ( r ) } \rangle \geq$ $\phi ( u ^ { ( r ) } ) - 1 7 / 1 6$ . To bound the progress contribution, put $\begin{array} { r } { t _ { j } : = \frac { N } { 4 } ( u _ { j } ^ { ( r ) } - 2 a ) } \end{array}$ . Since $q ^ { \prime }$ vanishes outside $( 0 , 1 )$ max<sub>0≤t≤1</sub> $t q ^ { \prime } ( t ) = 8 / 9$ , and max $q ^ { \prime } = 3 / 2$ , we have

$$
0 \leq \langle \nabla \psi ( u ^ { ( r ) } ) , u ^ { ( r ) } \rangle = \sum _ { j = 1 } ^ { N } q ^ { \prime } ( t _ { j } ) \left( \frac { t _ { j } } { N } + \frac { a } { 2 } \right) \leq \frac { 8 } { 9 } + \frac { 3 } { 3 2 N } < 1 .
$$

Using $0 \leq p _ { r } \leq 1$ and $\textstyle \sum _ { r } \lambda _ { r } \leq { \sqrt { m } }$ , it follows that

$$
\langle w , u \rangle - h _ { N } ( u ) \geq \frac { 2 } { N } \left[ \sum _ { r = 1 } ^ { m } \lambda _ { r } \left( \frac { 3 } { 1 6 } - p _ { r - 1 } \right) - m \right] \geq - \frac { 1 3 \sqrt { m } } { 8 N } - \frac { 2 m } { N } .
$$

The same inequality holds for every $g _ { h } \in \partial h _ { N } ( u )$ by convex combinations. Convexity and $R _ { N } ( 0 ) = 0$ give $\langle g _ { R } , u \rangle \geq R _ { N } ( u )$ . Finally, direct integration gives $\tau _ { N } ( s ) { - } s \tau _ { N } ^ { \prime } ( s ) \geq { - } 5 / 8 { - } 1 / ( 4 \sqrt { N } )$ : writing $t : = ( s - 1 / \sqrt { N } ) _ { + }$ the left side is $- t ^ { 2 } / 4 0 - t / ( 2 0 \sqrt { N } )$ for $0 \leq t \leq 5$ and is constant for $t \geq 5$ . Therefore,

$$
\langle g , u \rangle - b _ { N } ( u ) \geq - \frac { 1 3 \sqrt { m } } { 8 N } - \frac { 2 m } { N } - \frac { 5 } { 8 } - \frac { 1 } { 4 \sqrt { N } } > - 1 > - 2 ,
$$

where $N \geq 1 2 0$ and $m \leq N / 1 2$ were used.

At the origin, both added terms vanish, giving $b _ { N } ( 0 ) = h _ { N } ( 0 ) \in [ 0 , 2 / N )$ . If $s \leq 1 / \sqrt { N }$ , the terminal term vanishes and $b _ { N } ( u ) \geq h _ { N } ( u ) \geq - 1 / 6$ . Along $u + t e$ as $t  \infty$ , the last-block afine and progress terms and the penalty eventually become constant, while $\tau _ { N } ( \langle e , u \rangle + t )$ grows linearly with slope $1 / 4$ . Hence $b _ { N } ( u + t e )  - \infty$ □

## E Radial extension of the combined function

Remaining statements of Proposition 2. Composition with $U ^ { \top }$ preserves the Lipschitz and weak convexity bounds of $F _ { K }$ , since $\| U ^ { \top } \| = 1$ . The radial branch is convex and 1-Lipschitz, so their maximum is $4 / 5 – \mathrm { W C }$ and 1-Lipschitz. At the origin, $- A < 0 \le F _ { K } ( 0 ) = h _ { N } ( 0 )$ , giving $\Phi _ { U } ( 0 ) = h _ { N } ( 0 )$

To show that the lower bound $- 2 K$ is attained, set $\bar { z } ^ { ( i ) } : = ( 1 7 / ( 2 N ) ) \mathbf { 1 } _ { d _ { 0 } } + ( 8 K + 5 ) e$ for $i \in [ K ]$ . Every coordinate is at least $1 7 / ( 2 N )$ , so all afine maxima vanish, all progress values equal 1, and the negativecoordinate penalties vanish. Thus $h _ { N } ( \bar { z } ^ { ( i ) } ) = - 2 m / N$ . Since $\langle e , \bar { z } ^ { ( i ) } \rangle - 1 / \sqrt { N } = 8 K + 5 + 1 5 / ( 2 \sqrt { N } ) > 5$ the linear part of $\tau _ { N }$ gives

$$
\tau _ { N } ( \langle e , \bar { z } ^ { ( i ) } \rangle ) = \frac { 1 } { 4 } \left( 8 K + 5 + \frac { 1 5 } { 2 \sqrt { N } } \right) - \frac { 5 } { 8 } = 2 K + \frac { 5 } { 8 } + \frac { 1 5 } { 8 \sqrt { N } } > 2 K .
$$

Consequently, $F _ { K } ( \bar { z } ) = - 2 K$ . Moreover,

$$
\| \bar { z } \| \leq \sqrt { K } \left( \frac { 1 7 } { 2 } \sqrt { \frac { m } { N } } + 8 K + 5 \right) < 1 6 K \sqrt { K } < 1 6 ( 2 K + 2 ) \sqrt { N + K - 1 } = A - 2 K .
$$

$\mathrm { A t } \ x : = U \bar { z }$ , the radial branch is therefore below −2K. It follows that inf $\Phi _ { U } = - 2 K$ and $\Phi _ { U } ( 0 ) - \mathrm { i n f } \Phi _ { U } =$ $2 K + h _ { N } ( 0 ) \in [ 2 K , 3 K ]$

For any coordinate projection Π, including the identity, the Lipschitz bound gives $F _ { K } ( \Pi U ^ { \top } x ) \leq h _ { N } ( 0 ) +$ $( 3 1 / 3 2 ) \| x \| . { \mathrm { ~ I f ~ } } \| x \| \geq 6 4 ( A + 1 )$ , then

$$
\| x \| - A - F _ { K } ( \Pi U ^ { \top } x ) \geq \frac { \| x \| } { 3 2 } - A - h _ { N } ( 0 ) \geq A + 2 - h _ { N } ( 0 ) > 0 .
$$

The strict inequality persists locally, so both the value and the entire subdiferential of the maximum agree with those of the radial branch. This proves the oracle identity for the full and truncated functions.

For completeness, the convex-combination estimate used in the main text follows directly from $\| e _ { x } \| = 1$ and $\langle v , e _ { x } \rangle \geq - \| v \| / 4$ . For $0 \leq \lambda \leq 1$ , put $a _ { 0 } : = \lambda \| v \|$ and $b _ { 0 } : = 1 - \lambda$ . Then

$$
\| \lambda v + ( 1 - \lambda ) e _ { x } \| ^ { 2 } \ge a _ { 0 } ^ { 2 } + b _ { 0 } ^ { 2 } - \frac 1 2 a _ { 0 } b _ { 0 } \ge \frac 3 4 ( a _ { 0 } ^ { 2 } + b _ { 0 } ^ { 2 } ) \ge \frac 3 8 ( a _ { 0 } + b _ { 0 } ) ^ { 2 } \ge \frac 3 8 \operatorname* { m i n } \{ \| v \| , 1 \} ^ { 2 } .
$$

□

## F Auxiliary estimates for the small-gap construction

Auxiliary estimates for Proposition 4. Write $h : = h _ { M , N }$ and retain $s : = \sqrt { M / N } , A _ { s } : = 3 4 s + 2 s ^ { 2 }$ , and $B _ { s } : = 7 3 s$ . We first bound the radial component of a base-function subgradient. If $S ( z ) > 0$ and $g \in \partial h ( z )$ the subdiferential formula in Appendix B expresses g as a sum of componentwise nonpositive contributions from active afine branches and progress derivatives. An active afine branch with a nonzero coeficient satisfies $z _ { j } ^ { ( r ) } \le 8 c _ { j } / N < 1 7 / ( 2 N )$ . A nonzero progress derivative requires $2 a < z _ { j } ^ { ( r ) } < 2 a + 4 / N < 1 7 / ( 2 N )$ Hence $g _ { r , j } < 0$ implies $z _ { j } ^ { ( r ) } < 1 7 / ( 2 N )$ , and

$$
\langle g , z \rangle \geq - \frac { 1 7 } { 2 N } \| g \| _ { 1 } \geq - \frac { 1 7 } { 2 } \sqrt { \frac { M } { N } } \| g \| = - \frac { 1 7 } { 2 } s \| g \| .
$$

If the radial branch is active at x, the lower bound $h \geq - 2 s ^ { 2 }$ gives $\| x \| \geq A _ { s } - 2 s ^ { 2 } = 3 4 s$ . When both branches are active and $S ( U ^ { \top } x ) > 0$ , every $g \in \partial h ( U ^ { \top } x )$ therefore satisfies $\langle U g , x / \| x \| \rangle \geq - \| g \| / 4$ . The estimate in Appendix E and Proposition 1 imply

$$
\| \lambda U g + ( 1 - \lambda ) x / \| x \| \| \ge \sqrt { \frac { 3 } { 8 } } \operatorname* { m i n } \{ \| g \| , 1 \} \ge \frac { \sqrt { 3 / 8 } } { 4 \sqrt { N } } > \frac { 1 } { 8 \sqrt { N } } ( 0 \le \lambda \le 1 ) .
$$

The same lower bound holds when only the base branch or only the radial branch is active. Thus $S ( U ^ { \top } x ) > 0$ implies dist $( 0 , \partial \Psi _ { U } ( x ) ) > 1 / ( 8 \sqrt { N } )$ .

Next, $s \leq 1 / \sqrt { 1 2 } < 1 / 2$ gives $A _ { s } \leq 3 5 s$ , while $h ( 0 ) < 2 / N < s$ and $\mathrm { L i p } ( h ) < 1 / 2$ . For any coordinate projection Q, including the identity, and any $\| x \| \geq 7 2 s$ , we consequently have

$$
h ( { Q } { \boldsymbol { U } } ^ { \top } { \boldsymbol { x } } ) < s + \frac { 1 } { 2 } \| { \boldsymbol { x } } \| \leq \| { \boldsymbol { x } } \| - 3 5 s \leq \| { \boldsymbol { x } } \| - A _ { s } .
$$

The radial branch strictly dominates on a neighborhood of every such point. The full and truncated oracle replies are therefore $( \| x \| - A _ { s } , \{ x / \| x \| \} )$

Finally, suppose $| \langle u _ { M , j } , x \rangle | \ \leq \ a$ for every $j > \lfloor N / 2 \rfloor$ , and assume that $v : = \nabla ( \Psi _ { U } ) _ { 1 / 2 } ( x )$ satisfies $\| v \| \leq 1 / ( 1 6 \sqrt { N } )$ . Let $y : = \mathrm { p r o x } _ { \Psi _ { U } / 2 } ( x ) = x - v / 2 .$ , so that $v \in \partial \Psi _ { U } ( y )$ . For a constrained last-block coordinate, $\langle u _ { M , j } , y \rangle - 2 a \leq | \langle u _ { M , j } , v \rangle | / 2$ . Using $q ( t ) \leq 3 ( t _ { + } ) ^ { 2 }$ and the orthonormality of the columns of U yields

$$
\psi ( ( U ^ { \top } y ) ^ { ( M ) } ) \leq \frac { \lfloor N / 2 \rfloor } { N } + \frac { 3 N } { 6 4 } \sum _ { j > \lfloor N / 2 \rfloor } \lfloor \langle u _ { M , j } , v \rangle \vert ^ { 2 } \leq \frac { 1 } { 2 } + \frac { 3 N } { 6 4 } \| v \| ^ { 2 } \leq \frac { 1 } { 2 } + \frac { 3 } { 1 6 3 8 4 } < \frac { 2 7 } { 3 2 } .
$$

Proposition 1 gives $S ( U ^ { \top } y ) > 0$ , so the subgradient bound above contradicts $v \in \partial \Psi _ { U } ( y )$ . This proves the claimed ME gradient lower bound under the last-block condition.

If instead $\lVert x \rVert \geq B _ { s }$ , the same small-gradient assumption gives $\| x - y \| \leq 1 / ( 3 2 { \sqrt { N } } ) < s$ , and hence $\| y \| > 7 2 s$ . The radial oracle identity then gives $\lVert \boldsymbol { v } \rVert = 1$ , again a contradiction. These estimates establish both geometric properties used in the coupling argument of Proposition 4. □

## G Explicit connectors for the one-dimensional construction

We give the connecting functions used in Section 5.2 and verify their behavior at the joining points. Fix a parent interval $[ a , c ]$ of length $L \ge 6 4 B / \rho _ { ; }$ , with endpoint value $h ,$ and either of its two candidate children $[ a ^ { \prime } , c ^ { \prime } ]$ . The left and right connector lengths $\ell _ { - } : = a ^ { \prime } - a$ and $\ell _ { + } : = c - c ^ { \prime }$ belong to $\{ L / 8 , 5 L / 8 \}$ . The prescribed drop in endpoint value is $\eta : = B L / 1 6$

For either connector length $\ell ,$ define

$$
\mu _ { \ell } : = B - \rho \ell + \sqrt { ( \rho \ell - B ) ^ { 2 } + 2 \rho \eta - B ^ { 2 } } , ~ w _ { \ell } : = \frac { B - \mu _ { \ell } } { \rho } ,
$$

$$
C _ { \ell } ( t ) : = - \mu _ { \ell } t - \frac { \rho } { 2 } ( t - \ell + w _ { \ell } ) _ { + } ^ { 2 } ~ ( 0 \le t \le \ell ) .
$$

The left connector is $f _ { \omega } ( x ) : = h + C _ { \ell _ { - } } ( x - a )$ for $a \leq x \leq a ^ { \prime }$ , and the right connector is $f _ { \omega } ( x ) : = h + C _ { \ell _ { + } } ( c - x )$ for $c ^ { \prime } \leq x \leq c .$ . These formulas depend only on the parent interval, its endpoint value, and the selected child.

Verification of the connector properties. Consider $A _ { \ell } ( \mu ) : = \mu \ell + ( B - \mu ) ^ { 2 } / ( 2 \rho )$ on $[ B / 2 0 , B ]$ . Since $\ell \geq$ $L / 8 \geq 8 B / \rho _ { ; }$ its derivative satisfies $A _ { \ell } ^ { \prime } ( \mu ) = \ell - ( B - \mu ) / \rho > 0$ . Moreover,

$$
A _ { \ell } ( B / 2 0 ) \leq { \frac { B L } { 3 2 } } + { \frac { B ^ { 2 } } { 2 \rho } } \leq { \frac { 5 B L } { 1 2 8 } } < \eta < B \ell = A _ { \ell } ( B ) .
$$

Thus, $A _ { \ell } ( \mu ) = \eta$ has a unique solution in $( B / 2 0 , B )$ , and solving the quadratic gives the stated $\mu _ { \ell }$ . In particular, $0 < w _ { \ell } < B / \rho < \ell .$ . The function $C _ { \ell }$ is continuously diferentiable and piecewise quadratic, with $C _ { \ell } ( 0 ) = 0 , C _ { \ell } ( \ell ) = - \eta .$ , and $C _ { \ell } ^ { \prime } ( t ) = - \mu _ { \ell } - \rho ( t - \ell + w _ { \ell } ) _ { + }$ . Hence $- B \le C _ { \ell } ^ { \prime } ( t ) \le - \mu _ { \ell } < - B / 2 0$ , its smooth pieces have second derivative $0 \ \mathrm { o r } \ - \rho .$ , and $C _ { \ell } ^ { \prime } ( \ell ) = - B$ . Reflection gives the positive derivative bounds for the right connector and derivative $B$ at $c ^ { \prime }$

It remains to check the joins between diferent levels. At the left endpoint of any nonterminal interval, the incoming derivative from its parent connector is $- B ,$ , whereas its own left connector starts with derivative $- \mu _ { \ell _ { - } } \ge - B$ . At the right endpoint, its own right connector ends with derivative $\mu _ { \ell _ { + } } \leq B$ , whereas the outgoing derivative of the parent connector is B. Both jumps are therefore upward. The same statements hold at the endpoints of the root interval, where the exterior linear pieces have derivatives $- B$ and B. At the terminal interval endpoints, the derivatives of the V-shaped terminal function agree with those of the parent connectors. Its midpoint has the upward jump from $- B$ to B. The joins within each connector have continuous derivatives.

Consequently, $f _ { \omega }$ is continuous, all its smooth pieces have curvature at least $- \rho ,$ and all derivative jumps are upward. The derivative of $f _ { \omega } ( x ) + ( \rho / 2 ) x ^ { 2 }$ is therefore nondecreasing, including across the joins, which proves weak convexity. The derivative bounds also prove that $f _ { \omega }$ is B-Lipschitz. At every point, its full subdiferential is the interva $\partial f _ { \omega } ( x ) = [ f _ { \omega , - } ^ { \prime } ( x ) , f _ { \omega , + } ^ { \prime } ( x ) ]$ . At a nonterminal left endpoint this interval is $[ - B , - \mu _ { \ell _ { - } } ]$ , and at a nonterminal right endpoint it is $[ \mu _ { \ell _ { + } } , B ]$ . At the terminal endpoints it is respectively $\{ - B \}$ and $\{ B \}$ , while at the terminal midpoint it is $[ - B , B ]$ . Thus, every subgradient away from the midpoint has magnitude at least $B / 2 0$ , including at all joining points.

Finally, fix a prefix of length j. All continuations have the same intervals, endpoint values, and connecting functions outside $I _ { j }$ . Every point in $\mathbb { R } \backslash I _ { j }$ has a neighborhood disjoint from $I _ { j }$ , so all continuations agree on that neighborhood and have the same function value and full subdiferential there. At the endpoints of $I _ { j } ,$ the inner one-sided derivatives may depend on the next child through $\mu _ { \ell _ { - } } ~ \mathrm { o r } ~ \mu _ { \ell _ { + } }$ . This explains the use of closed intervals in the prefix oracle: an endpoint query reveals the next bit, after which it lies outside the selected child and its complete reply is determined. □