# Transformer Heads Looking for Order

Jasper van Doornmalen IMC UC mvandoornmalen@uc.cl

Corinna Mathwieser IMC UC corinna.mathwieser1@uc.cl

Alexander Kozachinskiy CENIA alexander.kozachinskyi@cenia.cl

Tomasz Steifer Centre for Credible AI tsteifer@ippt.pan.pl

José Verschae   
IMC UC   
jverschae@uc.cl

Felipe Urrutia IMC UC felipe.urrutia@uc.cl

Przemysław Andrzej Wał¸ega Queen Mary University of London p.walega@qmul.ac.uk

September 23, 2026

## Abstract

In this note, we show that the problem of checking, whether a sequence of bits is ordered, is not doable by 1-head 1-layer transformers but is doable by a 2-head 1-layer transformer. Unlike similar previous results, our results assume the model where transformers have an output MLP.

## 1 Introduction

Recently, Tesfaye et al. (2026) gave an example of a task which is not solvable by 1-layer 1-head transformers but is solvable by a 1-layer 2-head transformer. Later, Rajaraman et al. (2026) extended this result by showing that k-bit parity is computable by a 1-layer k-head transformer while it is not computable by 1-layer (k − 1)-head transformers.

These results are for attention-only transformers – those that do not have an output MLP. With the output MLP, these examples no longer work – in particular, because they have a fixed input length when the output MLP can simply memorize the function.

We partially extend these results to 1-layer transformers with the output MLPs. We give an example of a language (or, equivalently, a sequence of Boolean functions) which is computable with a 1-layer 2-head transformer with an output MLP, but is not computable by 1-layer 1-head transformers with output MLPs. This is the language consists of ordered binary strings – that is, strings where there is no 1 before 0.

In the construction of a transformer with 2 heads, one head essentially looks for the position of the last 0, and the second head looks for the position of the first 1. The output MLP checks that they go one after another.

The lower bound first uses a technique of Kozachinskiy et al. (2026) to “embed” 1-head 1-layer transformers into the first-order theory of reals with addition and order. While Kozachinskiy et al. (2026) show a similar result for any number of heads with the use of multiplications, we observe that for 1 head multiplications can be avoided.

The main technical component of the lower bound is a geometric lemma, lower-bounding the number of halfspaces, needed to separate ordered points of a Boolean cube from the non-ordered ones. Similar geometric problems has been previously considered in the area of relaxation complexity. However, the diference there is that ordered points of a Boolean cube have to be separated from all the other integer points, not only points of a Boolean cube. Then the problem for ordered points of a Boolean cube is equivalent to the problem for the vertices of a simplex where the exact asymptotic is known Averkov et al. (2026). However, known results from relaxation complexity are not suficient for our applications.

Very recently, Steifer and Naskręcki (2026) have established, for every $k \geq 2$ , that there exists a problem, doable with 1-layer k-head transformer, and not doable with 1-layer $\left( k - 1 \right)$ )-head transformer (in the model with output MLP). This problem is checking whether the number of 1s in the first half of the input word is at least the k-th power of the number of 1s in the second half of the word.

On a more technical level, we assume the standard softmax transformer model including the residual output MLP network, with the dimension and the parameters independent of the input length, while the input embedding is arbitrary and can be length-dependent. In fact, in our lower bound we do not even assume that the input embedding is additive, that is, decomposable into the sum of a token embedding and a positional encoding.

At the same time, the input embedding in our construction with 2 heads is additive, with a simple length-independent positional encoding, consisting of the absolute value of the position and of its square. In comparison, the upper bound construction of Steifer and Naskręcki (2026) uses an length-independent but a non-additive input embedding. It thus seems open to obtain a separation between 1-layer $( k - 1 )$ )-head transformers and 1-layer k-head transformers for $k \geq 3$ for the additive input embeddings (preferably, the lower bound should hold even for non-additive embeddings while the upper bound should be attained via an additive one).

## 2 Preliminaries

Basic Notation. For a vector $x \in \mathbb { R } ^ { d }$ we write $x _ { i }$ for its i-th coordinate, and $( \mathbb { R } ^ { d } ) ^ { * }$ for the set of all finite sequences of vectors in $\mathbb { R } ^ { d }$ . We study a family of functions $\mathrm { O R D E R E D } = \{ \mathrm { O R D E R E D } _ { n } \} _ { n \ge 1 }$ , where

$$
\mathrm { O R D E R E D } _ { n } : \{ 0 , 1 \} ^ { n } \to \{ 0 , 1 \} , \qquad \mathrm { O R D E R E D } _ { n } ( x _ { 1 } , \dots , x _ { n } ) = \left\{ { \begin{array} { l l } { 1 } & { x _ { 1 } \leq x _ { 2 } \leq \dots . x _ { n } , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} } \right.
$$

Attention Layers. A d-dimensional, H-head attention layer is a function $L \colon ( \mathbb { R } ^ { d } ) ^ { * } \to ( \mathbb { R } ^ { d } ) ^ { * }$ parameterised by query, key, and value matrices $Q ^ { ( k ) } , K ^ { ( k ) } , V ^ { ( k ) } \in \mathbb { R } ^ { d \times d }$ , with $k = 1 , \dots , H$ , a heads-mixing matrix $W _ { O } \in$ $\mathbb { R } ^ { d \times d H }$ , and parameters $W _ { 1 } , W _ { 2 } \in \mathbb { R } ^ { d \times d } , b _ { 1 } , b _ { 2 } \in \mathbb { R } ^ { d }$ of the output MLP. On input $( \alpha _ { 1 } , \ldots , \alpha _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ , the layer computes for each head $k = 1 , \dots , H ;$ : the attention logits of the jth to ith position $L _ { i j } ^ { ( k ) } \in \mathbb { R }$ and head values in jth position $h _ { \boldsymbol { j } } ^ { ( k ) } \in \mathbb { R } ^ { d } \ ( \mathrm { f o r } \ i , \boldsymbol { j } = 1 , \dots , n )$ as follows:

$$
L _ { i j } ^ { ( k ) } = \frac { \langle K ^ { ( k ) } \alpha _ { i } , Q ^ { ( k ) } \alpha _ { j } \rangle } { \sqrt { d } } ,\tag{1}
$$

$$
h _ { j } ^ { ( k ) } = \frac { \sum _ { i = 1 } ^ { n } \exp \{ L _ { i j } ^ { ( k ) } \} V ^ { ( k ) } \alpha _ { i } } { \sum _ { i = 1 } ^ { m } \exp \{ L _ { i j } ^ { ( k ) } \} } .\tag{2}
$$

The head values are combined and passed through a position-wise feed-forward network:

$$
h _ { j } = W _ { O } \left( \begin{array} { c } { { h _ { j } ^ { ( 1 ) } } } \\ { { \vdots } } \\ { { h _ { j } ^ { ( H ) } } } \end{array} \right) ,\tag{3}
$$

$$
\beta _ { j } = W _ { 2 } \cdot \mathrm { R e L U } \big ( W _ { 1 } ( h _ { j } + \alpha _ { j } ) + b _ { 1 } \big ) + b _ { 2 } ,\tag{4}
$$

where ReL $\begin{array} { r } { \operatorname { \Pi } _ { i } \operatorname { U } ( x _ { 1 } , \dots , x _ { d } ) = ( \operatorname* { m a x } \{ 0 , x _ { 1 } \} , \dots , \operatorname* { m a x } \{ 0 , x _ { d } \} ) } \end{array}$ ). The output sequence is $( \beta _ { 1 } , \ldots , \beta _ { n } ) = L ( \alpha _ { 1 } , \ldots , \alpha _ { n } )$

Transformers. We consider transformers that map sequences of tokens into the next, most probable token. We restrict our attention to 1-layer transformers. A 1-layer, H-head, d-dimensional transformer over a finite vocabulary V (a set of tokens which includes symbol ⊥) is a function $T \colon \mathcal { V } ^ { * }  \mathcal { V }$ specified by an H-head d-dimensional attention layer $L ,$ an input embedding E: $\mathcal { V } \times \mathbb { N } ^ { 2 } \to \mathbb { R } ^ { d }$ , and an output distribution matrix

$W \in \mathbb { R } ^ { | \nu | \times d }$ . On input $x _ { 1 } \ldots x _ { n } \in \mathcal { V } ^ { n }$ , it applies position-wise the input embedding $\alpha _ { i } = \mathrm { E } ( x _ { i } , i , n )$ , then applies the attention layer computing $( \beta _ { 1 } , \ldots , \beta _ { n } ) = L ( \alpha _ { 1 } , \ldots , \alpha _ { n } )$ , and outputs

$$
T ( x _ { 1 } , \dots , x _ { n } ) = \arg \operatorname* { m a x } _ { x \in \mathcal { V } } \left( \operatorname { s o f t m a x } ( W \beta _ { n } ) \right) _ { x } ,
$$

with the convention that $T ( x _ { 1 } , \ldots , x _ { n } ) = \bot { \mathrm { i f } }$ the argmax is not unique.

We say that a transformer T computes a sequence $\{ f _ { n } \} _ { n \in \mathbb { N } }$ of Boolean functions $f _ { n } \colon \{ 0 , 1 \} ^ { n } \to \{ 0 , 1 \}$ if $\{ 0 , 1 \} \subseteq \mathcal { V }$ and $T ( x ) = f _ { n } ( x )$ , for every $n \in \mathbb { N }$ and every $x \in \{ 0 , 1 \} ^ { n }$

## 3 Main Result

Theorem 1. ORDERED can be computed by a 2-head 1-layer transformer, but it cannot be computed by a 1-head 1-layer transformer.

Proof. A construction of a 2-head 1-layer transformer for ORDERED is given in Subsection 3.1. We know proceed to the lower bound. We require the following lemma whose analogue for 1-layer transformers with any number of heads was obtained in Kozachinskiy et al. (2026). However, in general case, the formula Φ requires multiplication. We notice in the 1-head case, multiplications can be avoided.

Lemma 2. Assume there is a 1-head 1-layer transformer T of dimension $d ,$ computing a sequence of Boolean functions $\{ f _ { n } \} _ { n = 1 } ^ { \infty }$ . Then there exists a first-order formula $\Phi ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { r } )$ in the interpretation $( \mathbb { R } , < , + )$ such that for all n there exist r polynomials $l _ { 1 } , \ldots , l _ { r } \ \in \ \mathbb { R } [ x _ { 1 } , \ldots , x _ { n } ]$ of degree at most 1 such that for any $x \in \{ 0 , 1 \} ^ { n }$ , we have $f _ { n } ( x ) = 1$ if and only $i f \Phi ( l _ { 1 } ( x ) , \dots , l _ { r } ( x ) ) = 1$

Proof. We construct two formulas $\Phi _ { 0 } , \Phi _ { 1 }$ that work for inputs where the last bit is fixed to 0 and 1, respectively. The final formula Φ can be obtained by introducing a fresh variable z which will be substituted by $x _ { n } ,$ , and writing

$$
\Phi : = ( ( \mathbf { z } = 0 )  \Phi _ { 0 } ) \wedge ( ( \mathbf { z } = 1 )  \Phi _ { 1 } ) .
$$

It remains now to construct $\Phi _ { 0 }$ (the formula $\Phi _ { 1 }$ can be constructed similarly). The output of the transformer is determined by the maximal coordinate of the vector $W \beta _ { n } ,$ where W is the output-distribution matrix, and $\beta _ { n }$ is the output vector of the layer in the n-th position. More specifically, in order for the output of the transformer to be 1, the value of the 1-coordinate of $W \beta _ { n }$ has to be strictly larger than the corresponding coordinate for all the other tokens of the vocabulary:

$$
( W \beta _ { n } ) _ { 1 } > ( W \beta _ { n } ) _ { \sigma } , \qquad \sigma \in \mathcal { V } \setminus \{ 1 \} .\tag{5}
$$

In turn, $\beta _ { n }$ is computed as in (2) for $j = n \colon$

$$
\beta _ { n } = W _ { 2 } \cdot \mathrm { R e L U } \big ( W _ { 1 } ( h _ { n } + \alpha _ { n } ) + b _ { 1 } \big ) + b _ { 2 } .\tag{6}
$$

Firstly, we introduce d variables $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { d } .$ , corresponding to coordinates of the vector $h _ { n } + \alpha _ { n }$ . We write a $( \mathbb { R } , < , + )$ -formula $\Psi ( \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { d } )$ , expressing a fact that the vector $\beta _ { n }$ , computed as in (6) with

$$
\begin{array} { r } { \left( \begin{array} { c } { \mathbf { u } _ { 1 } } \\ { \mathbf { u } _ { 2 } } \\ { \vdots } \\ { \mathbf { u } _ { \mathbf { d } } } \end{array} \right) } \end{array}
$$

in place of $h _ { n } + \alpha _ { n }$ , satisfies inequalities (5). This is doable because all the vectors in question have fixed dimension, matrices and vectors W, $W _ { 1 } , W _ { 2 } , b _ { 1 }$ , b consists of finitely many fixed real numbers, and the ReLU operation is expressible in $( \mathbb { R } , < , + )$ :

$$
y = \operatorname { R e L U } ( x ) \iff ( x \geq 0 \to y = x ) \land ( x < 0 \implies y = 0 ) .
$$

By a result of Ferrante and Rackof (1975), the interpretation $( \mathbb { R } , < , + )$ admits a quantifier elimination, and thus the formula Ψ can be assumed to be quantifier-free. For any input length n, for any transformer input $x \in \{ 0 , 1 \} ^ { n }$ , if we compute the values of coordinates of $h _ { n } + \alpha _ { n }$ on this input and put these values into the formula Ψ, it will output the value $f _ { n } ( x )$

To finish the argument, it is enough to show that for any input length $n ,$ for inputs with the last bit $x _ { n }$ fixed to 0, there exist $d + 1$ polynomials $\ell _ { 0 } , \ell _ { 1 } , \dots , \ell _ { d } \in \mathbb { R } [ x _ { 1 } , \dots , x _ { n } ]$ of degree at most 1 such that:

$$
h _ { n } + \alpha _ { n } = \left( \begin{array} { c } { \ell _ { 1 } ( x ) / \ell _ { 0 } ( x ) } \\ { \ell _ { 2 } ( x ) / \ell _ { 0 } ( x ) } \\ { \vdots } \\ { \ell _ { d } ( x ) / \ell _ { 0 } ( x ) } \end{array} \right) ,\tag{7}
$$

where $\ell _ { 0 } ( x ) > 0$ for all $x \in \{ 0 , 1 \} ^ { n }$ . Indeed, then we can introduce $d + 1$ variables $\mathbf { v } _ { 0 } , \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { d }$ and consider a formula:

$$
\Phi _ { 0 } ( \mathbf { v } _ { 0 } , \ldots , \mathbf { v } _ { d } ) = \Psi ( \frac { \mathbf { v } _ { 1 } } { \mathbf { v } _ { 0 } } , \ldots , \frac { \mathbf { v } _ { d } } { \mathbf { v } _ { 0 } } ) .
$$

As written, the right-hand side formula will be a Boolean combination of linear inequalities with fractions $\frac { \mathbf { v } _ { 1 } } { \mathbf { v } _ { 0 } } , \ldots , \frac { \mathbf { v } _ { d } } { \mathbf { v } _ { 0 } }$ . We can simply multiply them by their common<sup>1</sup> denominator $\mathbf { v } _ { 0 }$ (taking into account that the function $\ell _ { 0 } ( x )$ that will be substituted in place of $\mathbf { v } _ { 0 }$ takes only positive values on the transformer inputs) to obtain equivalent linear inequalities in $\mathbf { v } _ { 0 } , \ldots , \mathbf { v } _ { d }$ . It remains to substitute $\ell _ { 0 } , \ell _ { 1 } , \ldots , \ell _ { d }$ in place of $\mathbf { v } _ { 0 } , \mathbf { v } _ { 1 } \ldots , \mathbf { v } _ { d }$

It remains to show that the representation as in (7) takes place. It is enough to obtain such a representation for the vector $h _ { n } ^ { ( 1 ) }$ from (2) because $h _ { n } + \alpha _ { n } = W _ { O } h _ { n } ^ { ( 1 ) } + \alpha _ { n }$ , the matrix $W _ { O }$ is a constant d × d matrix, and $\alpha _ { n } = E ( 0 , n , n )$ does not depend on $x \in \{ 0 , 1 \} ^ { n }$

Now, the vector $h _ { n } ^ { ( 1 ) }$ is defined as:

$$
h _ { n } ^ { ( 1 ) } = \frac { \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} V ^ { ( 1 ) } \alpha _ { i } } { \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} } .\tag{8}
$$

By definition, expressions

$$
\exp \{ L _ { i n } ^ { ( 1 ) } \} V ^ { ( 1 ) } \alpha _ { i } , \exp \{ L _ { i n } ^ { ( 1 ) } \}
$$

are functions of $i , n , x _ { i } , x _ { n }$ . Under the fixation $x _ { n } = 0$ , they become functions of just $i , n , x _ { i } .$ , and thus can be written as:

$$
\exp \{ L _ { i n } ^ { ( 1 ) } \} V ^ { ( 1 ) } \alpha _ { i } = x _ { i } \rho _ { i n } ^ { 1 } + ( 1 - x _ { i } ) \rho _ { i n } ^ { 0 } , \quad \quad \exp \{ L _ { i n } ^ { ( 1 ) } \} = x _ { i } \theta _ { i n } ^ { 1 } + ( 1 - x _ { i } ) \theta _ { i n } ^ { 0 }
$$

for some $\rho _ { i n } ^ { 1 } , \rho _ { i n } ^ { 0 } \in \mathbb { R } ^ { d } , \theta _ { i n } ^ { 1 } , \theta _ { i n } ^ { 0 } \in \mathbb { R }$ . We see that both the numerator and the denominator in (8) depend linearly on $x _ { 1 } , \ldots , x _ { n }$ , and the denominator is a sum of positive numbers for every $x \in \{ 0 , 1 \} ^ { n }$ , as required.

That goal now is to show that such formula Φ as in Lemma 2 cannot exist for ORDERED. We do this through the following geometric lemma. We say that an n-dimensional vector $( x _ { 1 } , \ldots , x _ { n } ) \in \mathbb { R } ^ { n }$ is binary if $x _ { i } \in \{ 0 , 1 \}$ } for all $i = 1 , \ldots , n$ , and is ordered if:

$$
x _ { 1 } \leq x _ { 2 } \leq . . . \leq x _ { n } .
$$

Lemma 3. For $n \in \mathbb N$ , let $\theta _ { n }$ denote the minimal $k \in \mathbb N$ such that there exist k halfspaces $H _ { 1 } , \ldots , H _ { k } \in \mathbb { R } ^ { n }$ satisfying the following properties:

• all ordered n-dimensional binary vectors belong to $H _ { 1 } \cap H _ { 2 } \cap . . . \cap H _ { k } ,$

• none of the non-ordered n-dimensional binary vectors belongs to $H _ { 1 } \cap H _ { 2 } \dots \cap H _ { k }$

(each of the halfspaces can be with or without the border).

It holds that:

$$
\operatorname* { l i m } _ { n \to \infty } \theta _ { n } = + \infty .
$$

The proof of Lemma 3 is deferred to Section 3.2. Now, assume for contradiction that ORDERED is computable by a 1-head 1-layer transformer. Then a formula $\Phi ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { r } )$ as in Lemma 2 exists for ORDERED. By the result of Ferrante and Rackof (1975), we may assume that $\Phi ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { r } )$ is quantifierfree. Let C be the number of its atomic subformulas. Without loss of generality, we may assume that all of them are non-strict inequalities. We will show that $\theta _ { n } \leq C$ for infinitely many $n ,$ contradicting Lemma 3.

Take any n. There are $\ell _ { 1 } , \dots , \ell _ { r } \in \mathbb { R } [ x _ { 1 } , \dots , x _ { n } ]$ of degree at most 1 such that

$$
\mathrm { O R D E R E D } _ { n } ( x ) = \Phi ( \ell _ { 1 } ( x ) , \ldots , \ell _ { r } ( x ) )\tag{9}
$$

for all $x \in \{ 0 , 1 \} ^ { n }$ . Each atomic subformula in (9) is a non-strict linear inequality in $x _ { 1 } , \ldots , x _ { n }$ . Each binary vector $x \in \{ 0 , 1 \} ^ { n }$ gives rise to a sequence of C bits which we will call the Φ-pattern of x, encoding which atomic subformulas in (9) are true on x. The Φ-pattern of a binary vector x determines the value of the formula Φ on it, and hence an ordered binary vector and a non-ordered binary vector cannot have the same Φ-pattern. Out of n + 1 ordered binary vectors, there exist at least $m \geq ( n + 1 ) / 2 ^ { C }$ with the same Φ-pattern. Let these m vectors be $x ^ { 1 } , \ldots , x ^ { m }$ . Their Φ-pattern defines an intersection of C halfspaces $H _ { 1 } , \ldots , H _ { C }$ (some might be with border and the other without) that contains $x ^ { 1 } , \ldots , x ^ { m }$ but does not contain any non-ordered binary vector.

Vectors $x ^ { 1 } , \ldots , x ^ { m }$ are all equal to 0 in some initial positions and to 1 in some final positions. The rest of positions can be cut into $m - 1$ blocks $B _ { 1 } , B _ { 2 } , \ldots , B _ { m - 1 }$ so that:

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $B _ { 1 }$ </td><td rowspan=1 colspan=1> $B _ { 2 }$ </td><td rowspan=1 colspan=1>…</td><td rowspan=1 colspan=1> $B _ { m - 1 }$ </td></tr><tr><td rowspan=1 colspan=1>x1 =</td><td rowspan=1 colspan=1>00...0</td><td rowspan=1 colspan=1>00...0</td><td rowspan=1 colspan=1>..</td><td rowspan=1 colspan=1>00...0</td></tr><tr><td rowspan=1 colspan=1>x2 =</td><td rowspan=1 colspan=1>00...0</td><td rowspan=1 colspan=1>00...0</td><td rowspan=1 colspan=1>..</td><td rowspan=1 colspan=1>11...1</td></tr><tr><td rowspan=1 colspan=1>·..</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>…</td><td rowspan=1 colspan=1>·</td></tr><tr><td rowspan=1 colspan=1>xm−1 =</td><td rowspan=1 colspan=1>00...0</td><td rowspan=1 colspan=1>11...1</td><td rowspan=1 colspan=1>..</td><td rowspan=1 colspan=1>11...1</td></tr><tr><td rowspan=1 colspan=1>xm =</td><td rowspan=1 colspan=1>11...1</td><td rowspan=1 colspan=1>11...1</td><td rowspan=1 colspan=1>..</td><td rowspan=1 colspan=1>11...1</td></tr></table>

Consider an afine mapping from $\mathbb { R } ^ { m - 1 } { \mathrm { ~ t o ~ } } \mathbb { R } ^ { n }$ , given my:

$$
( y _ { 1 } , \dotsc , y _ { m - 1 } ) \mapsto ( \underbrace { y _ { 1 } , y _ { 1 } , \dotsc , y _ { 1 } } _ { B _ { 1 } } , \underbrace { y _ { 2 } , y _ { 2 } , \dotsc , y _ { 2 } } _ { B _ { 2 } } , \dotsc , \underbrace { y _ { m - 1 } , \dotsc , y _ { m - 1 } } _ { B _ { m - 1 } } )
$$

(positions before $B _ { 1 }$ are set to 0 and after $B _ { m - 1 } \ \mathrm { t o } \ 1 )$ . Under this mapping, ordered sequences in $\{ 0 , 1 \} ^ { m - 1 }$ go into $x ^ { 1 } , \ldots , x ^ { m }$ and thus inside $H _ { 1 } \cap . . . \cap H _ { C }$ , and non-ordered sequence in $\{ 0 , 1 \} ^ { m - 1 }$ go into non-ordered sequences in {0, 1}<sup>n</sup> – outside $H _ { 1 } \cap . . . \cap H _ { C }$ . Rewriting inequalities, defining $H _ { 1 } , \ldots , H _ { C }$ , in coordinates $y _ { 1 } , \ldots , y _ { m - 1 }$ , we obtain that $\theta _ { m - 1 } \leq C .$ . Since n can be chosen arbitrarily and $m \geq n / 2 ^ { C }$ , we obtain the inequality $\theta _ { m - 1 } \leq C$ for infinitely many m.

## 3.1 Construction with 2 Heads

We use the following input embedding:

$$
E ( x _ { i } , i , n ) = \left( { \begin{array} { l } { i } \\ { i ^ { 2 } } \\ { x _ { i } } \end{array} } \right) , \qquad i = 1 , \dots , n .
$$

The two heads compute the following attention logits:

$$
L _ { i n } ^ { ( 1 ) } = i n - x _ { i } n ^ { 2 } , \qquad L _ { i n } ^ { ( 2 ) } = x _ { i } n ^ { 2 } - i n .
$$

Let $i _ { 0 }$ be the position of the last 0 and $i _ { 1 }$ be the position of the first 1. It is easy to see that the attention logits of the first and second head are maximized at positions $i _ { 0 }$ and $i _ { 1 }$ , respectively, with a margin at least n:

$$
L _ { i _ { 0 } n } ^ { ( 1 ) } \geq L _ { i n } ^ { ( 1 ) } + n \qquad i \neq i _ { 0 } ,\tag{10}
$$

$$
L _ { i _ { 1 } n } ^ { ( 2 ) } \geq L _ { i n } ^ { ( 2 ) } + n ~ i \not = i _ { 1 } .\tag{11}
$$

The first and the second head compute, respectively,

$$
i _ { 0 } ^ { * } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} i } { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} } , \qquad i _ { 1 } ^ { * } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 2 ) } \} i } { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 2 ) } \} } .
$$

Due to $\left( 1 0 – 1 1 \right)$ , one can show that $| i _ { 0 } - i _ { 0 } ^ { * } | < 0 . 1 , | i _ { 1 } - i _ { 1 } ^ { * } | < 0 . 1$ . In fact, one can upper bound these diferences by $p o l y ( n ) e ^ { - n }$ but the 0.1-upper bound is suficient. The sequence is ordered if and only if $i _ { 0 } + 1 = i _ { 1 }$ . The output MLP checks if $| i _ { 1 } ^ { * } - 1 - i _ { 0 } ^ { * } | < 0 . 2$

One has to separately consider cases when the input has only 0s or only 1s, that is, when either $i _ { 0 }$ or $i _ { 1 }$ are not defined. It is easy to see that when $x _ { 1 } = . . . = x _ { n } . $ , the attention logits of the first head is maximized at $i = n$ , and of the second head at $i = 1$ , both with a margin at least n. That is, $i _ { 0 } ^ { * }$ will be 0.1-close to n and $i _ { 1 } ^ { * }$ will be 0.1-close to 1. To distinguish it from the when the sequence is of the form1 $\ast \dots \ast 0$ , our attention heads additionally compute:

$$
x _ { n } ^ { * } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} x _ { i } } { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 1 ) } \} } , \qquad x _ { 1 } ^ { * } = \frac { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 2 ) } \} x _ { i } } { \displaystyle \sum _ { i = 1 } ^ { n } \exp \{ L _ { i n } ^ { ( 2 ) } \} } .
$$

When $L _ { i n } ^ { ( 1 ) }$ is maximized at $i = n$ and $L _ { i n } ^ { ( 2 ) }$ is maximized at $i = 1$ , the values of $x _ { 1 } ^ { * } , x _ { n } ^ { * }$ will be 0.1-close to $x _ { 1 } , x _ { n }$ respectively. Hence, the output MLP can take into account the case $x _ { 1 } = . . . = x _ { n }$ as follows. If $| i _ { 0 } ^ { * } - 1 | + | i _ { 1 } ^ { * } - n | < 0 . 2$ , it compares $x _ { 1 } ^ { * } , x _ { n } ^ { * }$ . If they are very close, it outputs that the sequences are ordered, if not – that they are not. Otherwise, $| i _ { 0 } ^ { * } - 1 | + | i _ { 1 } ^ { * } - n | \geq 0 . 2$ , it outputs that the sequences are ordered if and only if $| i _ { 1 } ^ { * } - 1 - i _ { 0 } ^ { * } | < 0 . 2$

## 3.2 Proof of Lemma 3

Let $G _ { n }$ be a graph whose nodes are non-ordered n-dimensional binary vectors, and two such vectors $x , y$ if the segment between them intersects the convex hull of the set of ordered binary vectors.

Claim 4. For any $n \in \mathbb N$ , the graph $G _ { n }$ is $\theta _ { n } - c o l o r a b l e$

Proof. Denote $k = \theta _ { n } .$ and let $H _ { 1 } , \ldots , H _ { k } \subseteq \mathbb { R } ^ { n }$ by k halfspaces such that all ordered n-dimensional binary vectors belong to $H _ { 1 } \cap H _ { 2 } \cap . . . \cap H _ { k }$ but none of the non-ordered n-dimensional non-ordered binary vectors $( { \mathrm { i . e . } }$ , none of the vertices of $G _ { n } )$ does not belong to this intersection. Hence, for every vertex x of $G _ { n }$ there is $i \in \{ 1 , 2 , \ldots , k \}$ such that $x \in \mathbb { R } ^ { n } \setminus H _ { i }$ . Let this index i be the color of x. If there are more than one index with such property, choose the smallest one.

We have to show that if two vertices x, y of $G _ { n }$ are connected, then they have a diferent color. That is, we have to show that they cannot both belong to $\mathbb { R } ^ { n } \setminus H _ { i }$ for the same $i \in \{ 1 , \ldots , k \}$ . Indeed, if $x , y$ both belong to $\mathbb { R } ^ { n } \backslash H _ { i }$ , then so does the segment between them. Hence, this segment cannot intersect the convex hull of the set of ordered binary vectors as this convex hull is a subset of $H _ { 1 } \cap \ldots \cap H _ { k }$ □

We now give the lower bound on the chromatic number of $G _ { n }$ . We start of by observing that $G _ { n }$ has the following subgraph $G _ { n } ^ { \prime }$ . Vertices of $G _ { n } ^ { \prime }$ are the same as in $G _ { n } ,$ , where two vertices $x , y$ are connected if and only if $x + y$ is an ordered vector. (For example, binary vectors:

$$
\begin{array} { r } { x = ( 0 , 0 , 1 , 0 , 1 ) , \qquad y = ( 0 , 1 , 0 , 1 , 1 ) , } \end{array}
$$

are non-ordered, but their sum:

$$
x + y = ( 0 , 1 , 1 , 1 , 2 )
$$

is an ordered vector). We have to show that $G _ { n } ^ { \prime }$ is a subgraph of $G _ { n } .$ , that is, if for two vertices $x , y$ of $G _ { n }$ we have that $x + y$ is ordered, then the segment between x and y intersects the convex hull of the set of ordered binary vectors. Indeed, if $x + y$ is ordered, it first has a block of 0s, then a block of 1s, and then a block of 2s (with some of these blocks being potentially empty). Hence, we can write:

$$
{ \frac { x + y } { 2 } } = \left( \underbrace { 0 , 0 , \ldots , 0 } _ { a } , \underbrace { { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } , \ldots , { \frac { 1 } { 2 } } } _ { b } , \underbrace { 1 , 1 , \ldots , 1 } _ { c } \right)
$$

for some $a , b , c \geq 0$ . Therefore,

$$
\begin{array} { r } { \frac { x + y } { 2 } = \frac { 1 } { 2 } \cdot \left( \underbrace { 0 , 0 , \ldots , 0 } _ { a } , \underbrace { 0 , 0 , \ldots , 0 } _ { b } , \underbrace { 1 , 1 , \ldots , 1 } _ { c } \right) } \\ { + \frac { 1 } { 2 } \cdot \left( \underbrace { 0 , 0 , \ldots , 0 } _ { a } , \underbrace { 1 , 1 , \ldots , 1 } _ { b } , \underbrace { 1 , 1 , \ldots , 1 } _ { c } \right) , } \end{array}
$$

meaning that the middle point of the segment between x and y intersects the convex hull of the set of binary ordered vectors.

Further, consider a restriction of $G _ { n } ^ { \prime }$ to vertices x that consist of precisely 4 blocks of equal bits, starting from the block of 0s, that is, those that have a form:

$$
 { \boldsymbol { x } } = \left( \underbrace { 0 , 0 , \ldots , 0 } _ { a } , \underbrace { 1 , 1 , \ldots , 1 } _ { b } , \underbrace { 0 , 0 , \ldots , 0 } _ { c } , \underbrace { 1 , 1 , \ldots , 1 } _ { d } \right)
$$

for some $a , b , c , d > 0$ . Each such vertex is uniquely defined by a 3-element subset $\{ a , a + b , a + b + c \} \subseteq$ $\{ 1 , 2 , \ldots , n - 1 \}$

Now, take any $i , j , k , \ell \in \{ 1 , 2 , \dots , n - 1 \}$ such that:

$$
i < j < k < \ell .
$$

We claim that vertices $x , y$ that correspond to subsets $\{ i , j , k \}$ and $\{ j , k , \ell \}$ , respectively, are connected in $G _ { n } ^ { \prime }$ . Indeed, dividing positions into blocks of lengths

$$
i , \qquad j - i , \qquad k - j , \qquad \ell - k , \qquad n - \ell ,
$$

we can write:

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>i</td><td rowspan=1 colspan=1>j−i</td><td rowspan=1 colspan=1>k−j</td><td rowspan=1 colspan=1>l − k</td><td rowspan=1 colspan=1>n − l</td></tr><tr><td rowspan=1 colspan=1>x =</td><td rowspan=1 colspan=1> $\overline { { 0 \ldots 0 } }$ </td><td rowspan=1 colspan=1>1...1</td><td rowspan=1 colspan=1> $0 \ldots . 0$ </td><td rowspan=1 colspan=1>1...1</td><td rowspan=1 colspan=1> $\overline { { 1 \ldots 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>y =</td><td rowspan=1 colspan=1>0...0</td><td rowspan=1 colspan=1>0...0</td><td rowspan=1 colspan=1>1...1</td><td rowspan=1 colspan=1>0...0</td><td rowspan=1 colspan=1>1...1</td></tr></table>

One can see that the sum of $x , y$ is ordered.

We conclude that the graph $G _ { n } ^ { \prime }$ has a subgraph, isomorphic to the graph $T _ { n } ,$ defined as follows. Vertices of $T _ { n }$ are 3-element subsets of $\{ 1 , 2 , \ldots , n \}$ . Two subsets are connected if the two largest element in one coincide with two smallest element in the other. It is now enough to show that the chromatic number of $T _ { n }$ goes to +∞ as $n  + \infty$ . That is, for every $q \in \mathbb { N }$ , we have to show that there exists $n _ { 0 } \in \mathbb { N }$ such that $T _ { n }$ is not q-colorable for every $n \geq n _ { 0 }$

This is an easy consequence of the Ramsey theorem for hypergraphs:

Theorem 5 (Erdos and Rado (1952)). Let $s > t \geq 2$ and q be natural numbers. Then there exists $n _ { 0 }$ such that for every $n \geq n _ { 0 }$ the following holds. For every coloring of the t-element subsets of $\{ 1 , 2 , \ldots , n \}$ in q colors there exists an s-element subset of $\{ 1 , 2 , \ldots , n \}$ such that all its t-element subsets have the same color.

Fix any q and take the bound for $n _ { 0 }$ from the last theorem for $s = 4 , t = 3$ . Consider any $n \geq n _ { 0 }$ and assume for contradiction that $T _ { n }$ is q-colorable. Apply the result to the corresponding coloring of the vertices of $T _ { n }$ . There will be a 4-element subset $\{ i , j , k , \ell \}$ such that all its 3-element subsets are colored in the same way. However, if $i < i < k < \ell .$ , then $\{ i , j , k \} , \{ j , k , \ell \}$ are connected in $T _ { n }$ and hence cannot be colored by the same color, a contradiction.

Remark 6. The idea to use Ramsey theorem for hypergraphs for the lower bound on the chromatic number of $T _ { n }$ was obtained in a conversation with an LLM. All the other parts of the proof and the writing of the paper have been performed without the AI assistance.

## References

Averkov, G., Keil, S., and Weltge, S. (2026). The relaxation complexity of the standard simplex is logarithmic. arXiv preprint arXiv:2606.11852.

Erdos, P. and Rado, R. (1952). Combinatorial theorems on classifications of subsets of a given set. Proceedings of the London mathematical Society, 3(1):417–439.

Ferrante, J. and Rackof, C. (1975). A decision procedure for the first order theory of real addition with order. SIAM Journal on Computing, 4(1):69–76.

Kozachinskiy, A., Steifer, T., and Wałcega, P. (2026). Parity, sensitivity, and transformers. arXiv preprint arXiv:2602.05896.

Rajaraman, R., Sundaram, R., and Tesfaye, A. (2026). The head complexity of boolean functions in singlelayer attention. arXiv preprint arXiv:2609.04046.

Steifer, T. and Naskręcki, B. (2026). h+1 heads express more than h heads. https://github.com/ CredibleAI/heads/blob/main/h\_vs\_h\_1\_heads.pdf.

Tesfaye, A., Kujawa, Z., Rajaraman, R., and Sundaram, R. (2026). Two (narrow) heads are better than (an arbitrarily wide) one. In International Conference on Learning Representations, volume 2026, pages 20112–20128.