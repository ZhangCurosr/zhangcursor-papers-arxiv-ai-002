# SAT Encoding for Set of At-Most-K Cardinality Constraints with Ladder Shape

Hieu Truong Xuan<sup>1†</sup>, Tuyen Kieu Van<sup>1</sup>, Khanh To Van<sup>1\*†</sup>

<sup>1</sup>University of Engineering and Technology, Vietnam National University, Hanoi, Vietnam.

\*Corresponding author(s). E-mail(s): khanhtv@vnu.edu.vn; Contributing authors: truongxuanhieu11@vnu.edu.vn; tuyenkv@vnu.edu.vn; <sup>†</sup>These authors contributed equally to this work.

## Abstract

This paper presents a novel SAT encoding designed to represent a sequence of At-Most-K (AMK) cardinality constraints arranged in a ladder-like structure, referred to as the Ladder AMK constraint. These AMK constraints feature a ladder-like arrangement, with shared common variables in the center and two variables difering at the beginning and the end. The proposed encoding first decomposes the sequence of AMK constraints into separated blocks, then encodes each block as a whole rather than encoding each AMK constraint individually. This approach improves the eficiency of constraint solving in SAT solvers by significantly reducing the size of the encoding, precisely the number of variables, and the generated clauses required. To achieve this, the encoding employs Sequential Counter to encode each block and constructs clauses that link adjacent blocks, ensuring equivalence with the input constraints. Compared to the previously pro posed encoding that uses Binary Decision Diagrams for Ladder AMO constraint (k = 1), our encoding requires fewer variables and clauses. Our study is the first study that develops SAT encoding for Ladder AMK constraints with k > 1. Experimental results on single Ladder AMO and Ladder AMK constraints indicate that our encoding significantly enhances both the size of the encoding and the eficiency of solving time compared to existing encodings. Furthermore, by evaluating on the two real-world combinatorial problems, namely Anti-bandwidth Problem and Nurse Rostering Problem, we found that the proposed encoding surpasses alternative encodings, as well as more advanced Constraint Programming (CP) and Mixed Integer Programming (MIP) solvers. In summary, the SAT encoding technique presented in this paper provides a more eficient method for encoding sequence constraint, specifically Ladder AMK constraint, which could

enhance the efectiveness of SAT solving for various problems involving this type of constraint.

Keywords: SAT encoding, Cardinality constraints, Sequence constraints, Sequential Counter encoding, At-Most-K constraints, Anti-bandwidth

## 1 Introduction

Sequence constraint is a type of constraint that frequently appears in combinatorial optimization problems, such as production line scheduling, employee scheduling, Nurse Rostering Problem, Crew Rostering Problem, sports timetabling, and OR problems. For instance, in the Car Sequencing Problem [1, 2], sequence constraints restrict the maximum number of cars with a specific option to no more than two in any sequence of five consecutive cars on a production line. In the Nurse Rostering Problem [3, 4], sequence constraints may limit a nurse to working at most four night shifts and eight evening shifts, while also requiring at least four of-days within any sequence of fourteen consecutive days [5]. These constraints help ensure proper rest and recovery by preventing excessively intense work schedules. Similarly, in the Crew Rostering Problem, an employee may be allowed to work a maximum of five days in any seven consecutive days [6] to comply with labor regulations. In the taxi driver shift scheduling problem at an airport for a car rental company [7], sequence constraints are used to ensure a balanced distribution of morning and afternoon shifts among drivers. In the RobinX sports timetabling problem [8], sequence constraints limit the maximum number of home or away games a team can play within any fixed-length sequence of consecutive matches. Sequence constraints limit the number of occurrences of certain specific values in a sequence of k variables and are often referred to as AmongSeq [9] and AtMostSeq [1]. Certain problems, such as the Anti-bandwidth Problem (ABP) [10], can be reformulated to incorporate sequence constraints. This problem has applications in scheduling, radio frequency assignment, obnoxious facility location, and map coloring.

The impressive progress of SAT solvers has initiated what is now referred to as the SAT revolution [11]. These solvers are capable of addressing enormous formulas that contain millions of variables and clauses. In the near future, they are expected to handle formulas with significantly more than just millions of variables and clauses. Today, SAT solving has become a serious alternative to Integer Linear Programming (ILP) solvers for addressing NP-complete problems. Due to substantial improvements in the performance of SAT solvers, SAT solving is increasingly being used in various real-world applications. However, when applying SAT encoding to combinatorial problems, a significant number of clauses can be generated. To mitigate this issue, several encoding techniques have been proposed to reduce the overwhelming number of clauses [12, 13].

Sequence constraints have been extensively studied in the field of Constraint Programming (CP) to develop eficient solutions, as highlighted in the work of Bergman et al. [5] and Siala et al. [2]. However, efective SAT encodings for sequence constraints, such as AmongSeq [9] and AtMostSeq [1], remain largely unexamined. A notable exception is Duplex encoding [14], which is specifically designed for sequences of At-Most-One (AMO) constraints. In our research, we propose a block-based encoding method for a sequence of At-Most-K (AMK) constraints, which encodes them as a whole rather than handling each AMK constraint independently. In particular, we leverage shared variables among neighboring AMK constraints to decompose the input sequence constraint into blocks. Then, we encode these blocks using an encoding inspired by the Sequential Counter [15] encoding. Auxiliary variables generated from the block encoding are used to connect adjacent blocks, thereby reformulating the original constraint. This approach not only demonstrates greater eficiency for sequences of AMO constraints compared to the Duplex encoding, as reported by Hieu et al. [16], but also proves to be efective for sequences of AMK constraints with k > 1.

• We propose a novel SAT encoding for a set of AMK cardinality constraints arranged in a ladder shape. Our approach leverages shared variables among neighboring AMK constraints, allowing us to organize them into blocks for encoding rather than encoding each constraint independently. To the best of our knowledge, this is the first SAT encoding method for the sequence of AMK cardinality constraints with a ladder shape where k > 1.

• Additionally, we present solutions to the Anti-bandwidth Problem and Nurse Rostering Problem, using our new SAT encoding for the sequence of At-Most-One and At-Most-K cardinality constraints, respectively. Experimental results show that our approach outperforms existing SAT encodings, as well as Constraint Programming (CP) and Mixed Integer Programming (MIP) methods that utilize powerful solvers such as CPLEX, CP-SAT, and Gurobi.

This paper extends our previously published research [16] in the following ways: First, we generalize the encoding from AMO constraints to AMK constraints with k ≥ 1 for sequences of cardinality constraints in a ladder shape, whereas the previous work focused solely on AMO constraints. Second, we also include an experiment on the Nurse Rostering Problem to evaluate the performance of the proposed encoding on combinatorial problems that include sequences of AMK constraints. Third, for the Anti-bandwidth Problem, we develop a parallel search implementation, leading to improved performance. We also expanded our experiment to include comparisons and evaluations with Adder Network encoding [17] and powerful CP and MIP solvers, namely Gurobi and CP-SAT. This comparison highlights the efectiveness of our proposed SAT encoding in solving the Anti-bandwidth Problem and Nurse Rostering Problem when compared to CP and MIP solvers.

The paper is organized as follows: Section 1 introduces the research problem and summarizes the contributions of this study. Section 2 presents the basic definitions of the Ladder AMK constraint and discusses how to represent one AMK constraint using two sub-AMK constraints. Section 3 details the block encoding for the Ladder AMK constraint and includes a comparative evaluation against existing SAT encodings. Section 4 introduces the encoding for the Ladder AMO constraint, which is a specific case of the Ladder AMK constraint with k = 1. Sections 5 and 6 sequentially explain the approach to solving the Anti-bandwidth Problem and Nurse Rostering Problem using SAT solving. Section 7 provides experimental results that compare our proposed encoding with existing SAT encodings, as well as state-of-the-art CP and MIP solvers. Finally, Section 8 concludes the paper.

## 2 Preliminaries

## 2.1 Ladder AMK constraint

A Ladder AMK constraint is defined as a sequence of AMK cardinality constraints that involve w boolean variables. In this structure, each pair of adjacent AMK constraints shares w − 1 variables. They difer only in the first variable of the first constraint and the last variable of the second constraint. This overlapping feature results in a ladder-like shape. The formal definition of the Ladder AMK constraint is provided in Definition 1.

Definition 1. Given a sequence of n boolean variables $\Omega = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ and a number w such that $1 < w \leq n ,$ a Ladder AMK constraint of width w is formulated as follows:

$$
L a d d e r ( \Omega , w ) = \bigwedge _ { i = 0 } ^ { n - w } ( \sum _ { j = i + 1 } ^ { i + w } x _ { j } \leq k ) \qquad 1 \leq k < w
$$

![](images/57cf849af602c1855cddd1c1bb36d4668d4b044dcf4976e62aafa7c09319f754.jpg)  
Fig. 1: Ladder AMK constraint of 10 variables and width 4.

Figure 1 illustrates a Ladder AMK constraint over a sequence of 10 boolean variables $\Omega = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { 1 0 } \}$ of width $w = 4$ . This Ladder AMK constraint consists of seven AMK constraints, each involving four consecutive variables in the sequence.

As discussed in Section 1, a wide range of combinatorial problems can be modeled using sequence constraints, particularly the Ladder AMK constraint. For instance, in the Car Sequencing Problem [1, 2], let $c _ { i , o }$ be a Boolean variable that indicates whether the $i ^ { t h }$ car is equipped with option $o , \mathrm { i . e . , } c _ { i , o }$ is true if the $i ^ { t h }$ car is equipped with option o and f alse otherwise. Given a production sequence of 10 cars, the constraint requiring that, in any 5 consecutive cars, at most 2 cars are equipped with the option o can be modeled as a Ladder AMK constraint with $n = 1 0 , w = 5$ , and $k = 2$ , as illustrated in Figure 2.

In the Nurse Rostering Problem [3–5], let $x _ { i , j , s }$ be a Boolean variable indicating whether nurse i is assigned to shift $s \in \{ D , E , N , O \}$ on day $j ^ { t h }$ , where $D , E .$ , and

$$
\begin{array} { r l r l r l } { c _ { 1 , o } ~ + ~ c _ { 2 , o } ~ + ~ c _ { 3 , o } ~ + ~ c _ { 4 , o } ~ + ~ c _ { 5 , o } ~ } & & { } & & { \leq 2 \land } & { } & { } \\ { c _ { 2 , o } ~ + ~ c _ { 3 , o } ~ + ~ c _ { 4 , o } ~ + ~ c _ { 5 , o } ~ + ~ c _ { 6 , o } ~ } & & { } & { \leq 2 \land } & { } & { } \\ { c _ { 3 , o } ~ + ~ c _ { 4 , o } ~ + ~ c _ { 5 , o } ~ + ~ c _ { 6 , o } ~ + ~ c _ { 7 , o } ~ } & & { } & { \leq 2 \land } & { } & { } \\ { c _ { 4 , o } ~ + ~ c _ { 5 , o } ~ + ~ c _ { 6 , o } ~ + ~ c _ { 7 , o } ~ + ~ c _ { 8 , o } ~ } & & { } & { \leq 2 \land } & { } & { } \\ { c _ { 5 , o } ~ + ~ c _ { 6 , o } ~ + ~ c _ { 7 , o } ~ + ~ c _ { 8 , o } ~ + ~ c _ { 9 , o } ~ } & & { } & { \leq 2 \land } & { } & { } \\ { c _ { 6 , o } ~ + ~ c _ { 7 , o } ~ + ~ c _ { 8 , o } ~ + ~ c _ { 9 , o } ~ + ~ c _ { 1 0 , o } ~ } & { } & { \leq 2 \land } & { } & { } \end{array}
$$

Fig. 2: Ladder AMK constraint ensuring that for any 5 consecutive cars in a sequence of 10 cars, at most 2 cars can be equipped with option o.

N denote day, evening, and night shifts, respectively, while O represents an of-day. Given a planning horizon of 28 consecutive days, the constraint requiring that nurse i is assigned to at most 8 night shifts in any 14 consecutive days can be modeled as a Ladder AMK constraint with $n = 2 8 , w = 1 4$ , and $k = 8 ,$ as illustrated in Figure 3.

$$
\begin{array} { c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c } { x _ { i , 1 , N } } & { + } & { x _ { i , 2 , N } } & { + } & { \ldots } & { } & { + } & { x _ { i , 1 4 , N } } & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & \\ & { x _ { i , 2 , N } } & { + } & { \ldots } & { } & { + } & { x _ { i , 1 4 , N } } & { + } & { x _ { i , 1 5 , N } } & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & \\ & & & & & & & & & & & { x _ \dots } & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & \\ & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & \end{array}
$$

Fig. 3: Ladder AMK constraint ensuring that nurse i works at most 8 night shifts in any 14 consecutive days over a 28-day planning horizon.

In addition to the examples above, many constraints arising in other combinatorial and scheduling problems can also be modeled as Ladder AMK constraints, such as the Crew Rostering Problem [6], the taxi driver shift scheduling problem [7], the RobinX sports timetabling problem [8], and the Anti-bandwidth Problem [14, 16], as discussed in Section 1. In general, any requirement that limits the number of selected activities, resources, or events within a window over an ordered sequence can be expressed in this form. Therefore, the Ladder AMK constraint is not restricted to specific benchmark problems, but constitutes a general and practically relevant modeling primitive. Its ability to capture sequence cardinality restrictions highlights its wide applicability and practical significance in real-world optimization problems.

## 2.2 Decomposition of AMK constraint

An AMK constraint can be decomposed into two separate AMK constraints applied to two subsets of the original sequence of variables. This decomposition is formally explained in Proposition 1. It also incorporates connecting clauses to ensure that the decomposed constraints comply with the original constraint without violating it.

Proposition 1. For any integer number i s.t. $1 \leq i < w$ , the constraint

$$
x _ { 1 } + x _ { 2 } + . . . + x _ { w } \leq k
$$

holds if and only if the following constraint holds:

$$
( x _ { 1 } + . . . + x _ { i } \leq k ) \wedge ( x _ { i + 1 } + . . . + x _ { w } \leq k )
$$

$$
\wedge \bigwedge _ { j = 1 } ^ { k } ( x _ { 1 } + \ldots + x _ { i } \leq k - j \vee x _ { i + 1 } + \ldots + x _ { w } \leq j - 1 )
$$

Proposition 1 can be proven to be correct by considering all possible cases in which one of the two sub-expressions has a total of $k , k - 1 , k - 2 , \ldots , 1$ . In each of these cases, the other sub-expression must have a corresponding total that is less than or equal to $0 , 1 , 2 , \ldots , k - 1$ , thereby the sum of these two expressions never exceeds k. The following example illustrates the decomposition of an AMK constraint with $k = 3$ over 8 variables $\{ x _ { 1 } , \ldots , x _ { 8 } \}$

$$
\begin{array} { r l l } { { } } & { { } } & { { ( x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 3 ) \wedge ( x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 3 ) } } \\ { { } } & { { } } & { { \wedge ( ( x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 2 ) \vee ( x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 0 ) ) } } \\ { { } } & { { \leq } } & { { { } \quad \equiv } } & { { \wedge ( ( x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 1 ) \vee ( x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 1 ) ) } } \\ { { } } & { { } } & { { \wedge ( ( x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 0 ) \vee ( x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 2 ) ) , } } \end{array}
$$

The decomposition of the AMK constraint serves a crucial role in the proposed encoding. In subsequent sections, the proposed encoding leverages this decomposition to replace a large AMK constraint with smaller sub-constraints that can be reused across consecutive constraints. As a result, the proposed encoding achieves a reduced encoding size of the resulting SAT formulation.

The Ladder AMO constraint is a specific case of the Ladder AMK constraint, which occurs when $k = 1$ . To derive the decomposition of the AMO constraint, one can simply substitute k with 1 in the decomposition of the AMK constraint. Lemma 1 provides a formal decomposition of an AMO constraint that involves w variables. Lemma 1. For any integer number i $s . t . \ 1 \leq i < w$ , the constraint

$$
x _ { 1 } + x _ { 2 } + \ldots + x _ { w } \leq 1
$$

holds if and only if the following constraint holds:

$$
( x _ { 1 } + \ldots + x _ { i } \leq 1 ) \land ( x _ { i + 1 } + \ldots + x _ { w } \leq 1 )
$$

$$
\wedge ( x _ { 1 } + . . . + x _ { i } \leq 0 \vee x _ { i + 1 } + . . . + x _ { w } \leq 0 )
$$

## 3 Sequential Counter encoding for Ladder constraint

This section introduces the proposed encoding called SCL (Sequential Counter encoding for Ladder constraints) for encoding Ladder AMK constraints. The SCL encoding leverages shared variables among neighboring AMK constraints to partition a Ladder AMK constraint into blocks. These blocks are then represented using an encoding inspired by the Sequential Counter [15], which we refer to as the SC encoding. After that, auxiliary variables generated by the SC encoding are incorporated into clauses that connect adjacent blocks through Proposition 1, thus reformulating the original constraint. By exploiting block decomposition to encode the sequence of AMK constraints as a whole, instead of encoding each AMK constraint independently, the proposed encoding achieves a more compact and eficient representation.

## 3.1 Partitioning the Ladder AMK constraint into blocks

We define a block consisting of a sequence of w variables $\{ v _ { 1 } , \ldots , v _ { w } \}$ as a set of w expressions $\{ \mathcal { E } _ { 1 } , \ldots , \mathcal { E } _ { w } \}$ where $\begin{array} { r } { \forall i \in [ 1 , w ] , \mathcal { E } _ { i } = \sum _ { i = 1 } ^ { i } v _ { j } } \end{array}$ . Figure 4 illustrates the structure of such a block. A block can be eficiently represented by encoding the constraint $\mathcal { E } _ { w } \leq k$ using a single Sequential Counter (Seq) encoding [15], as Seq encoding of $\mathcal { E } _ { w } \leq k$ sequentially encodes $\mathcal { E } _ { 1 } \leq k , . . . , \mathcal { E } _ { w - 1 } \leq k$

![](images/55354eaa6b5476e9a817a4b673dd9ed6201ec22f515c3ccc624024d57b2023ea.jpg)  
Fig. 4: A block consisting of a sequence of w variables $\{ v _ { 1 } , \ldots , v _ { w } \}$

Leverage this insight, we partition the input Ladder AMK constraint into blocks. Given a Ladder AMK constraint with width w over a sequence of n variables $\Omega = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ , we first divide Ω into $\begin{array} { r } { M \ = \ \left\lceil \frac { n } { w } \right\rceil } \end{array}$ subsets $\left\{ \omega _ { 1 } , \hdots , \omega _ { M } \right\}$ . Each subset ω<sub>i</sub> $( 1 \leq i \leq M - 1 )$ contains exactly w unique consecutive variables s.t. $w _ { i } = \{ x _ { i , 1 } , \dotsc , x _ { i , w } \}$ , while the last subset $\omega _ { M }$ may contain fewer variables if n is not divisible by w. We then consider each subset $\omega _ { i }$ as sequences of variables and construct blocks over these sequences so that the resulting blocks collectively cover all expressions required by the Ladder AMK constraint.

For instance, Figure 5 illustrates the partitioning of the Ladder AMK constraint presented in Figure 1. The sequence of variables in this constraint is divided into three subsets $\{ \{ x _ { 1 } , \dots , x _ { 4 } \} , \{ x _ { 5 } , \dots , x _ { 8 } \} , \{ x _ { 9 } , x _ { 1 0 } \} \}$ . From these subsets, we construct blocks by considering appropriate variable sequences. The first subset (resp. the last subset) is considered as the sequence $\{ x _ { 4 } , \ldots , x _ { 1 } \}$ (resp. $\{ x _ { 9 } , x _ { 1 0 } \} )$ , from which we construct block $A _ { 1 }$ (resp. block $A _ { 4 } )$ . For the second subset, to cover all expressions required by the Ladder AMK constraint, we consider two sequences, namely $\{ x _ { 5 } , \ldots , x _ { 8 } \}$ and $\{ x _ { 8 } , \ldots , x _ { 5 } \}$ , and construct two blocks $A _ { 2 }$ and $A _ { 3 }$ , respectively. Since each block can be eficiently represented using a single Seq encoding, the Ladder AMK constraint illustrated in Figure 5 can be encoded using four Seq encodings, instead of the seven encodings that would be required if all seven AMK constraints were encoded independently.

![](images/5feb48984bdb40c70042405ae72e88bf888aa4ea8aa98f518ad3320ce34dc101.jpg)  
Fig. 5: Decomposition of an Ladder AMK constraint with $n = 1 0$ and $w = 4$

## 3.2 Encoding for Ladder AMK constraint

This section is organized into two parts. First, we describe how each individual block obtained from the decomposition of the Ladder AMK constraint is encoded using SC encoding. Second, we connect adjacent blocks to reconstruct the original Ladder AMK constraint.

## a. Encoding each block in the Ladder AMK constraint

We denote by $x _ { i , j }$ the $j ^ { \mathrm { t h } }$ variable in the variable sequence of the block i. For instance, consider the block $A _ { 2 }$ illustrated in Figure 5. The variable sequence of this block is $\{ x _ { 5 } , x _ { 6 } , x _ { 7 } , x _ { 8 } \}$ , which implies that $x _ { 2 , 1 } ~ \equiv ~ x _ { 5 } , ~ x _ { 2 , 2 } ~ \equiv ~ x _ { 6 }$ $x _ { 2 , 3 } \equiv x _ { 7 }$ , and $x _ { 2 , 4 } \equiv x _ { 8 }$ . Similarly, block $A _ { 3 }$ in Figure 5 has variable sequence $\{ x _ { 8 } , x _ { 7 } , x _ { 6 } , x _ { 5 } \}$ . Consequently, $x _ { 3 , 1 } \equiv x _ { 8 } , x _ { 3 , 2 } \equiv x _ { 7 } , x _ { 3 , 3 } \equiv x _ { 6 }$ , and $x _ { 3 , 4 } \equiv x _ { 5 }$

Since each $x _ { i , j }$ is a binary variable, the sum $x _ { i , 1 } + . . . + x _ { i , j }$ can take any integer value between 0 and $j$ . To represent this sum, we introduce $j$ register bits, denoted by $R _ { i , j , s }$ for $s \in \{ 1 , \ldots , j \}$ . Each register bit $R _ { i , j , s }$ indicates that under the condition “the sum of the first $j$ variables in the variable sequence of block i is greater than or equal to $s ^ { \prime }$ . Particularly, $R _ { i , j , s }$ is true if and only if $x _ { i , 1 } + \ldots + x _ { i , j } \geq s .$ , and false otherwise, i.e., when $x _ { i , 1 } + . . . + x _ { i , j } \leq s - 1$

It is necessary to mention that all the expressions within each block are At-Most-K. Consequently, when encoding sub-expression $x _ { i , 1 } + . . . + x _ { i , j }$ with $k < j ,$ it is suficient to introduce only k register bits $\{ R _ { i , j , 1 } , \dotsc , R _ { i , j , k } \}$ . Any use of a register bit $R _ { i , j , s }$ with $s > k$ is redundant, as such bits are always false. For example, to encode $x _ { 1 , 1 } + x _ { 1 , 2 } + x _ { 1 , 3 } \leq 2$ , only two register bits, namely $R _ { 1 , 3 , }$ 1 and $R _ { 1 , 3 , 2 }$ , are required.

$$
\begin{array} { r l } { \boxed { \begin{array} { c c c c c c } { x _ { 1 } } & { + } & { x _ { 2 } } & { + } & { x _ { 3 } } & { + } & { x _ { 4 } } \\ & { x _ { 2 } } & { + } & { x _ { 3 } } & { + } & { x _ { 4 } } \\ & & & { x _ { 3 } } & { + } & { x _ { 4 } } \\ & & & & & { x _ { 4 } } \end{array} } \equiv } & { R _ { 1 , 3 , 1 } , R _ { 1 , 3 , 2 } } \end{array}
$$

Fig. 6: Register bit construction of block $A _ { 1 }$ under the condition $\leq 2 .$

Figure 6 illustrates the assignment of register bits for the block $A _ { 1 }$ of the Ladder constraint illustrated in Figure 5, under the condition $\leq 2 . \ x _ { 4 }$ has a maximum value of 1, and thus requires only one register bit, denoted $R _ { 1 , 1 , 1 }$ . Next, $x _ { 3 } + x _ { 4 }$ has a maximum value of 2, and therefore is assigned two register bits, $R _ { 1 , 2 , 1 }$ and $R _ { 1 , 2 , 2 }$ . For $x _ { 2 } + x _ { 3 } + x _ { 4 }$ , although the maximum value is 3, due to the constraint $\leq 2 .$ this sub-expression is also assigned only two register bits, namely $R _ { 1 , 3 , 1 }$ and $R _ { 1 , 3 , 2 }$ . Finally, $x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 }$ does not require additional register bits, since condition $\leq 2$ can be expressed based on the value of $x _ { 1 }$ and the register bits already assigned to $x _ { 2 } + x _ { 3 } + x _ { 4 }$

We also note that, if $R _ { i , j , s }$ is true, then $R _ { i , j , s - 1 }$ must also be true, since $x _ { i , 1 } + \ldots + x _ { i , j } \geq s$ implies $x _ { i , 1 } + . . . + x _ { i , j } \geq s - 1$ . Besides, the equality $x _ { i , 1 } +$ $\ldots + x _ { i , j } = s$ is valid if and only if $R _ { i , j , s }$ is true and $R _ { i , j , s + 1 }$ is false. Specifically, $R _ { i , j , s }$ is true indicating that $x _ { i , 1 } + . . . + x _ { i , j } \geq s .$ , while $R _ { i , j , s + 1 }$ is f alse implying that $x _ { i , 1 } + \ldots + x _ { i , j } \leq ( s + 1 ) - 1 = s .$ . Combining these two conditions results in $s \leq x _ { i , 1 } + . . . + x _ { i , j } \leq s ,$ and thus $x _ { i , 1 } + . . . + x _ { i , j } = s$

The SC encoding for decomposed blocks involves the application of seven equations (1), (2), (3), (4), (5), (6), and (7). The formal proof of the correctness of this encoding is given in Appendix A.

$$
\bigwedge _ { j = 1 } x _ { i , j } \to R _ { i , j , 1 }\tag{1}
$$

w−1 min(j−1,k)

$$
\bigwedge _ { j = 2 } \quad \bigwedge _ { s = 1 } \quad R _ { i , j - 1 , s } \to R _ { i , j , s }\tag{2}
$$

w−1 min(j,k)

$$
\bigwedge _ { j = 2 } \quad \bigwedge _ { s = 2 } \quad x _ { i , j } \wedge R _ { i , j - 1 , s - 1 } \to R _ { i , j , s }\tag{3}
$$

$$
\bigwedge _ { j = 1 } ^ { k } \lnot x _ { i , j } \to \lnot R _ { i , j , j }\tag{4}
$$

$$
w { - } 1 m i n ( j , k )
$$

$$
\bigwedge _ { j = 2 } \quad \bigwedge _ { s = 2 } \quad \neg R _ { i , j - 1 , s - 1 } \to \neg R _ { i , j , s }\tag{5}
$$

w−1 min(j−1,k)

$$
\bigwedge _ { j = 2 } \quad \sum _ { s = 1 } \quad \lnot x _ { i , j } \wedge \lnot R _ { i , j - 1 , s } \to \lnot R _ { i , j , s }\tag{6}
$$

$$
\bigwedge _ { j = k + 1 } ^ { w } x _ { i , j } \to \lnot R _ { i , j - 1 , k }\tag{7}
$$

Equation (1) specifies that the sum of the first $j$ variables in block i must be at least one if $x _ { i , j }$ is true. Equation (2) states that if the sum of the first $j - 1$ variables in block i is greater than or equal to s, then the sum of the first $j$ variables must also be at least s. Equation (3) ensures that if $x _ { i , j }$ is true and the sum of the first $j - 1$ variables is greater than or equal to $s - 1$ , then the sum of the first $j$ variables in block i must be at least s. Together, equations (1), (2), and (3) ensure the correctness of the statement ${ } ^ { * 6 } R _ { i , j , s }$ holds true if and only if $x _ { i , 1 } + . . . + x _ { i , j } \geq s ^ { ; \ast }$

Next, equation (4) limits the sum of the first $j$ variables in block i to at most $j - 1$ if $x _ { i , j }$ is $f a l s e$ . The equation (5) states that if the sum of the first $j - 1$ variables is less than $s - 1$ , then the sum of the first j variables must be less than s. Equation (6) states that if $x _ { i , j }$ is false and the sum of the first $j - 1$ variables is less than $s ,$ then the sum of the first j variables must also be less than s. Taken together, the equations (4), (5), and (6) validate the correctness of the statement ${ ^ { \circ } R } _ { i , j , s }$ is false if and only if $x _ { i , 1 } + . . . + x _ { i , j } \leq s - 1 ^ { \flat }$

Finally, the equation (7) stipulates that if $x _ { i , j }$ is true, then the sum of the first $j - 1$ variables in block i must be strictly less than k. By applying this constraint to all indices $j$ within the range of $[ k + 1 , w ]$ , we ensure that no more than k out of the w variables can be set to true. Additionally, the sizes of the clauses (i.e., the number of literals in each clause) in our encoding are at most 3.

As previously mentioned, the SC encoding is inspired by the Seq encoding [15]. In particular, equations (1), (2), (3), and (7) are adopted directly from the Seq encoding [15]. The remaining equations (4), (5), and (6) are intentionally included to explicitly characterize the conditions under which the corresponding register bits are forced to false. This is crucial for ensuring the correctness of the proposed encoding, as these register bits will be reused to represent the block connections.

It is necessary to note that when two blocks involve the same subset of variables $\omega _ { i } = \{ x _ { i , 1 } , \dots , x _ { i , w } \}$ , if one block has already enforced the constraint $x _ { i , 1 } + \ldots +$ $x _ { i , w } ~ \leq ~ k$ via equation (7), the application of equation (7) in the other block is unnecessary. Exploiting this insight, when encoding two diferent blocks that share the same subset of variables, we include equation (7) on only one of them. We refer to the block on which equation (7) is applied to enforce the AMK constraint as the AMK block, while the other block, which primarily provides expressions for block connection, is referred to as the connection block.

![](images/696831d4be9b5248fdc6a0ef1235a273aa7ed6fa7ef1765cbdb21f09d2dd4549.jpg)  
Fig. 7: Decomposition of Ladder At-Most-Two constraint.

Figure 7 illustrates an example of a Ladder AMK constraint with $k = 2 .$ This Ladder AMK constraint is divided into four blocks, labeled $A _ { 1 } , A _ { 2 } , A _ { 3 }$ , and $A _ { 4 }$ Figure 8 presents these four blocks along with the necessary register bits used in the SC encoding. The block $A _ { 1 }$ is encoded using SC with the formulas from equations (1) to (7). The generated clauses for constructing the block $A _ { 1 }$ are presented below. In this context, the variables are defined as ${ x _ { 1 , 1 } } \equiv { x _ { 4 } } , { x _ { 1 , 2 } } \equiv { x _ { 3 } }$ $x _ { 1 , 3 } \equiv x _ { 2 } , x _ { 1 , 4 } \equiv x _ { 1 } , x _ { 1 , 1 } \equiv R _ { 1 , 1 , 1 }$ , and $x _ { 4 } \equiv R _ { 1 , 1 , 1 } { \mathrm { , } }$

![](images/325dde0d06687c75c567c91b4799aefbddd1f3bc171bda1fc9908727477decab.jpg)  
Fig. 8: Register bits constructing of Ladder At-Most-Two constraint.

$$
\begin{array} { r l r l r l r l r l r l r l r l r l } { ( 1 ) } & { { } \stackrel { A _ { 1 } } { \Longleftrightarrow } } & { } & { } & { } & { x _ { 1 , 2 } } & { } & { } & { \to } & { } & { R _ { 1 , 2 , 1 } } & { } & { } & { } & { x _ { 3 } } & { } & { } & { \to } & { } & { R _ { 1 , 2 , 1 } } \\ & { } & { x _ { 1 , 3 } } & { } & { } & { \to } & { } & { R _ { 1 , 3 , 1 } } & { } & { } & { \Longleftrightarrow } & { } & { } & { x _ { 2 } } & { } & { } & { \to } & { } & { R _ { 1 , 3 , 1 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l r l r l } { ( 2 ) } & { { } \stackrel { A _ { 1 } } { \longleftrightarrow } } & { } & { } & { { R } _ { 1 , 1 , 1 } } & { } & { { } \to } & { } & { { R } _ { 1 , 2 , 1 } } & { } & { { } } & { { } \Longleftrightarrow } & { } & { } & { { x } _ { 4 } } & { } & { { } \to } & { } & { { R } _ { 1 , 2 , 1 } } \\ & { } & { \stackrel { A _ { 1 } } { \longleftrightarrow } } & { } & { } & { { R } _ { 1 , 2 , 1 } } & { } & { } & { { } \to } & { } & { { R } _ { 1 , 3 , 1 } } & { } & { { } \Longleftrightarrow } & { } & { } & { { R } _ { 1 , 2 , 1 } } & { } & { { } \to } & { } & { { R } _ { 1 , 3 , 1 } } \\ & { } & { } & { } & { \quad R _ { 1 , 2 , 2 } } & { } & { } & { { } \to } & { } & { { R } _ { 1 , 3 , 2 } } & { } & { } & { { } \Longleftrightarrow } & { } & { } & { { R } _ { 1 , 2 , 2 } } & { } & { { } \to } & { } & { { R } _ { 1 , 3 , 2 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l r l r l } { ( 3 ) } & { { } \stackrel { A _ { 1 } } { \longleftrightarrow } } & { } & { x _ { 1 , 2 } \wedge R _ { 1 , 1 , 1 } } & { } & { \to } & { } & { R _ { 1 , 2 , 2 } } & { } & { } & { { } } & { x _ { 3 } \wedge x _ { 4 } } & { } & { { } \to } & { } & { R _ { 1 , 2 , 2 } } \\ & { } & { x _ { 1 , 3 } \wedge R _ { 1 , 2 , 1 } } & { } & { \to } & { } & { R _ { 1 , 3 , 2 } } & { } & { } & { { } } & { x _ { 2 } \wedge R _ { 1 , 2 , 1 } } & { } & { { } \to } & { } & { R _ { 1 , 3 , 2 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l r l r l } { ( 4 ) } & { { } \stackrel { \mathcal { A } _ { 1 } } { \longleftrightarrow } } & { } & { } & { \to x _ { 1 , 2 } } & { } & { } & { \to } & { \to R _ { 1 , 2 , 2 } } & { { } \Longleftrightarrow } & { } & { } & { \to x _ { 3 } } & { } & { } & { \to } & { \to R _ { 1 , 2 , 2 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l r l r l r l } { ( 5 ) } & { } & { \stackrel { A _ { 1 } } { \Longleftrightarrow } } & { } & & { \stackrel { - R _ { 1 , 1 , 1 } } { \longrightarrow } } & { } & & { \to } & { \stackrel { - R _ { 1 , 2 , 2 } } { \longrightarrow } } & { } & & { \stackrel { \lnot x _ { 4 } } { \longleftrightarrow } } & { } & & { \to } & { \stackrel { - R _ { 1 , 2 , 2 } } { \longrightarrow } } & { } & & { } \\ & { } & & { - R _ { 1 , 2 , 1 } } & { } & & { \to } & { \lnot R _ { 1 , 3 , 2 } } & { } & { \Longleftrightarrow } & { } & & { \to R _ { 1 , 2 , 1 } } & { } & { \to } & { \lnot R _ { 1 , 3 , 2 } } & { } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l } { ( 6 ) \ } & { \stackrel { \_  \ - \alpha _ { 1 , 2 } \wedge \  \boldsymbol { R } _ { 1 , 1 , 1 } } { \longleftrightarrow } } & { \stackrel { \  \ } { \longrightarrow } } & { \stackrel { \  \boldsymbol { R } _ { 1 , 2 , 1 } } { \longleftrightarrow } } & { \quad \stackrel { \_  \ \alpha _ { 3 } \wedge \  \boldsymbol { x } _ { 4 } } { \longleftrightarrow } } & { \ \stackrel { \  \ } { \longrightarrow } } & { \stackrel { \ - \boldsymbol { R } _ { 1 , 2 , 1 } } { \longleftrightarrow } } & { \quad \stackrel { \_ } { \longrightarrow \ } } & { \stackrel { \ } { \longrightarrow \ } } & { \stackrel { \ } { \sim } \boldsymbol { R } _ { 1 , 3 , 1 } } \\ & { \stackrel { \ } { \longleftrightarrow } } & { \stackrel { \ } { \longrightarrow \ } } & { - x _ { 1 , 3 } \wedge \ \stackrel { \_ } { \sim } \boldsymbol { R } _ { 1 , 2 , 1 } } & { \ \stackrel { \  \ } } & { \ \stackrel { \ } { \to \ } } & { \ - \boldsymbol { R } _ { 1 , 3 , 1 } } & { \ \stackrel { \ } { \longleftrightarrow } } & { \ \stackrel { \ } { \sim } x _ { 2 } \wedge \ \ \stackrel { \ } { \sim } \boldsymbol { R } _ { 1 , 2 , 1 } } & { \ \stackrel { \  \ } } & { \ - \boldsymbol { R } _ { 1 , 3 , 1 } } \\ & { \quad \stackrel { \ } { \longrightarrow \ } } & { - x _ { 1 , 3 } \wedge \ \stackrel { \  \ } { \sim } \boldsymbol { R } _ { 1 , 2 , 2 } } & { \ \longrightarrow \ } & { \ - \boldsymbol { R } _ { 1 , 3 , 2 } } & { \ \stackrel { \ } { \longrightarrow \ } } & { \ - x _ { 2 } \wedge \ \stackrel { \ } { \sim } \boldsymbol { R } _ { 1 , 2 , 2 } } & { \ \stackrel { \ } { \to \ } } & { \ - \boldsymbol { R } _ { 1 , 3 , 2 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l r l r l r l r l } { ( 7 ) } & { { } \stackrel { A _ { 1 } } { \Longleftrightarrow } } & { } & { } & { } & { x _ { 1 , 3 } } & { } & { } & { \to } & { - R _ { 1 , 2 , 2 } } & { } & { } & { x _ { 2 } } & { } & { } & { \to } & { - R _ { 1 , 2 , 2 } } \\ & { } & { } & { x _ { 1 , 4 } } & { } & { } & { \to } & { - R _ { 1 , 3 , 2 } } & { } & { } & { x _ { 1 } } & { } & { } & { \to } & { - R _ { 1 , 3 , 2 } } \end{array}
$$

Encoding the block $A _ { 2 }$ is the same as $A _ { 1 }$ , including the following clauses:

$$
( 1 ) \quad { \begin{array} { r l r l r l r l r l r l r l r l r l } { A _ { 2 } } & { } & & { } & { x _ { 2 , 2 } } & { } & & { \to } & { } & { R _ { 2 , 2 , 1 } } & { } & { } & { \iff } & { } & { x _ { 6 } } & { } & { \to } & { } & { R _ { 2 , 2 , 1 } } & { } \\ & { } & & { } & { x _ { 2 , 3 } } & { } & { } & { \to } & { } & { R _ { 2 , 3 , 1 } } & { } & { } & { x _ { 7 } } & { } & { \to } & { } & { R _ { 2 , 3 , 1 } } & { } \end{array} }
$$

$$
( 2 ) \begin{array} { r l r l r l r l r l r l r l r l } & { \mathscr { L } _ { 2 } } & & { } & & { R _ { 2 , 1 , 1 } } & & { \to } & & { R _ { 2 , 2 , 1 } } & & & { } & & { x _ { 5 } } & & { \to } & & { R _ { 2 , 2 , 1 } } \\ & { \Longleftrightarrow } & & { } & & { R _ { 2 , 2 , 1 } } & & { \to } & & { R _ { 2 , 3 , 1 } } & & { \Longleftrightarrow } & & { } & & { R _ { 2 , 2 , 1 } } & & { \to } & & { R _ { 2 , 3 , 1 } } \\ & { } & & { } & & { R _ { 2 , 2 , 2 } } & & { \to } & & { R _ { 2 , 3 , 2 } } & & { } & & { } & & { R _ { 2 , 2 , 2 } } & & { \to } & & { R _ { 2 , 3 , 2 } } \end{array}
$$

$$
\begin{array} { r l r l r l r l r l r l } { ( 3 ) ~ } & { \stackrel { A _ { 2 } } { \longleftrightarrow } ~ } & { x _ { 2 , 2 } \wedge R _ { 2 , 1 , 1 } } & { \to } & { R _ { 2 , 2 , 2 } } & & { } & { \Longleftrightarrow } & { x _ { 6 } \wedge x _ { 5 } } & { \to } & { } & { R _ { 2 , 2 , 2 } } \\ & { \stackrel { x _ { 2 , 3 } \wedge R _ { 2 , 2 , 1 } } { \longrightarrow } ~ } & { \to } & { R _ { 2 , 3 , 2 } } & { \iff } & { x _ { 7 } \wedge R _ { 2 , 2 , 1 } } & { \to } & { } & { R _ { 2 , 3 , 2 } } \end{array}
$$

$$
( 4 ) \quad \iff \quad \quad \iff \quad \quad \to x _ { 2 , 2 } \qquad \to \quad \to \quad \to R _ { 2 , 2 , 2 } \quad \iff \quad \quad \to x _ { 6 } \qquad \to \quad \to R _ { 2 , 2 , 2 }
$$

$$
\begin{array} { r l r } { ( 5 ) \quad \Longleftrightarrow \qquad \Longleftrightarrow \qquad \longleftrightarrow \qquad - R _ { 2 , 1 , 1 } \qquad \to \qquad \to \qquad - R _ { 2 , 2 , 2 } } & { \longleftrightarrow \qquad \cdots \qquad \cdots \qquad \to \qquad \to \qquad R _ { 2 , 2 , 2 } } \\ & { \longleftrightarrow \qquad - R _ { 2 , 2 , 1 } \qquad \to \quad - R _ { 2 , 3 , 2 } \quad \Longleftrightarrow \qquad \cdots R _ { 2 , 2 , 1 } \qquad \to \qquad \to \qquad R _ { 2 , 2 , 2 } } \\ & { \quad - \ v x _ { 2 , 2 } \wedge - R _ { 2 , 1 , 1 } \quad \to \quad - R _ { 2 , 2 , 1 } } \\ { ( 6 ) \quad \Longleftrightarrow \quad \stackrel { A _ { 2 } } { \longleftrightarrow } \quad \stackrel { - x _ { 2 , 3 } \wedge - R _ { 2 , 2 , 1 } } { \to } \to \quad - R _ { 2 , 2 , 1 } \quad \Longleftrightarrow \quad - x _ { 6 } \wedge - R _ { 2 , 2 , 1 } } & { \to \quad - R _ { 2 , 2 , 1 } } \\ & { \quad \to \ v x _ { 2 , 3 } \wedge - R _ { 2 , 2 , 2 } } & { \to \qquad - R _ { 2 , 3 , 2 } } \\ & { \quad \qquad \times \qquad \times _ { 2 , 3 } \qquad \to \quad - R _ { 2 , 2 , 2 } } \\ { ( 7 ) \quad \Longleftrightarrow \qquad \quad \stackrel { x _ { 2 , 4 } } { \longleftrightarrow } \quad \quad \ x _ { 2 , 4 } \qquad \to \quad - R _ { 2 , 2 , 2 } } & { \Longleftrightarrow \qquad x _ { 8 } \qquad \to \quad - R _ { 2 , 2 , 2 } } \\ & { \quad \Longleftrightarrow \qquad \quad \quad x _ { 2 , 3 } \qquad \to \quad - R _ { 2 , 3 , 2 } } \end{array} \to \qquad \overbrace { x _ { 2 } } \qquad \to \qquad \to \qquad \to \qquad R _ { 2 , 2 , 2 }  \\ & { \quad \Longleftrightarrow \qquad \quad \times _ { 2 , 3 , 2 } } \\ & { \quad \Longleftrightarrow \qquad \quad \times _ { 2 , 3 } \qquad \to \quad - R _ { 2 , 3 , 2 } } \\ & { \quad \to \qquad \to \qquad \to \qquad \to \qquad \to \qquad \to \qquad \to \qquad \Lambda _ { 2 , 2 } } \\ &  \quad \times _ { 1 } \qquad \to \qquad \to
$$

Since the block $A _ { 2 }$ has already enforced the constraint $x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 2 .$ the block $A _ { 3 }$ only leverages the formulas from equations (1) to (6) to generate the necessary register bits for its connection to the adjacent block $A _ { 4 }$ . Finally, the block $A _ { 4 }$ is a special case, as it contains only two variables instead of four as in the other blocks. Consequently, it is encoded using seven equations (1) to $( 7 )$ with w set to 2. As a result, the encoding of block $A _ { 4 }$ consists of the following clauses:

$$
( x _ { 9 } \  \ R _ { 4 , 2 , 1 } ) \wedge ( x _ { 1 0 } \  \ R _ { 4 , 2 , 1 } ) \wedge ( ( x _ { 9 } \wedge x _ { 1 0 } ) \  \ R _ { 4 , 2 , 2 } ) \wedge (  x _ { 9 } \  \  R _ { 4 , 2 , 2 } )
$$

$$
\Lambda ( \lnot x _ { 1 0 } \to \lnot R _ { 4 , 2 , 2 } ) \wedge ( ( \lnot x _ { 9 } \wedge \lnot x _ { 1 0 } ) \to \lnot R _ { 4 , 2 , 1 } ) .
$$

## b. Clauses connecting two adjacent blocks

After encoding blocks, we construct connecting clauses to connect adjacent blocks, thereby reformulate the Ladder AMK constraint. This construction is supported by Proposition 1. For instance, connection between blocks $A _ { 1 }$ and $A _ { 2 }$ in Figure 7 is formed as follows.

$$
\begin{array} { r l r } & { } & { ( x _ { 2 } + x _ { 3 } + x _ { 4 } \le 2 ) \wedge ( x _ { 5 } \le 2 ) } \\ & { } & { ( x _ { 2 } + x _ { 3 } + x _ { 4 } + x _ { 5 } \le 2 ) \quad \overset { \mathrm { P r o p . ~ 1 } } { \equiv } \quad \overset { \cdot } { \wedge } ( x _ { 2 } + x _ { 3 } + x _ { 4 } \le 1 \vee x _ { 5 } \le 0 ) } \end{array}
$$

$$
\begin{array} { r l r } & { } & { ( x _ { 3 } + x _ { 4 } \le 2 ) \wedge ( x _ { 5 } + x _ { 6 } \le 2 ) \qquad } \\ & { } & { \equiv \qquad \wedge ( x _ { 3 } + x _ { 4 } \le 1 \vee x _ { 5 } + x _ { 6 } \le 0 ) } \\ & { } & { \wedge ( x _ { 3 } + x _ { 4 } \le 0 \vee x _ { 5 } + x _ { 6 } \le 1 ) } \end{array}
$$

$$
\begin{array} { r l r } & { } & { ( x _ { 4 } + x _ { 5 } + x _ { 6 } + x _ { 7 } \le 2 ) \mathrm { ~ \nsideset { } { ~ \mathscr ~ { ~ P r o p . ~ 1 ~ } } { ~ \mathscr ~ { ~ \chi } = ~ 1 ~ } ~ } \wedge ( x _ { 5 } + x _ { 6 } + x _ { 7 } \le 2 ) } \\ & { } & { \stackrel { ( x _ { 4 } + x _ { 5 } + x _ { 6 } + x _ { 7 } \le 0 ) ~ } { = } \mathscr { N } ( x _ { 4 } \le 1 \vee x _ { 5 } + x _ { 6 } + x _ { 7 } \le 0 ) } \\ & { } & { \wedge ( x _ { 4 } \le 0 \vee x _ { 5 } + x _ { 6 } + x _ { 7 } \le 1 ) } \end{array}
$$

Since the constraint $x _ { 1 } + x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 2 \ ( { \mathrm { r e s p . ~ } } x _ { 5 } + x _ { 6 } + x _ { 7 } + x _ { 8 } \leq 2 )$ is true, as guaranteed by the equation $( 7 )$ , it follows that all its sub-constraints are also true, namely $x _ { 2 } + x _ { 3 } + x _ { 4 } \le 2 , x _ { 3 } + x _ { 4 } \le 2$ , and $x _ { 4 } \leq 2 \ ( \mathrm { r e s p . } \ x _ { 5 } + x _ { 6 } + x _ { 7 } \leq 2 $ $x _ { 5 } + x _ { 6 } \leq 2$ , and $x _ { 5 } \leq 2 )$ . Consequently, the connection between block $A _ { 1 }$ and block $A _ { 2 }$ can be represented as follows.

$$
( x _ { 2 } + x _ { 3 } + x _ { 4 } \le 1 \vee x _ { 5 } \le 0 ) \wedge ( x _ { 2 } + x _ { 3 } + x _ { 4 } \le 0 \vee x _ { 5 } \le 1 )
$$

$$
\wedge ( x _ { 3 } + x _ { 4 } \leq 1 \vee x _ { 5 } + x _ { 6 } \leq 0 ) \wedge ( x _ { 3 } + x _ { 4 } \leq 0 \vee x _ { 5 } + x _ { 6 } \leq 1 )
$$

$$
\wedge ( x _ { 4 } \leq 1 \vee x _ { 5 } + x _ { 6 } + x _ { 7 } \leq 0 ) \wedge ( x _ { 4 } \leq 0 \vee x _ { 5 } + x _ { 6 } + x _ { 7 } \leq 1 )
$$

We utilize the generated register bits to express the connection between $A _ { 1 }$ and $A _ { 2 }$ . For instance, the expression $x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 1$ serves as the negation of $x _ { 2 } + x _ { 3 } + x _ { 4 } \geq 2$ and can be replaced with $\neg R _ { 1 , 3 , 2 }$ . Similarly, the expression $x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 0$ is the negation of $x _ { 2 } + x _ { 3 } + x _ { 4 } \geq 1$ and can be represented as $\neg R _ { 1 , 3 , 1 }$ . This substitution can be applied to the remaining expressions as well. Therefore, the connection between block $A _ { 1 }$ and block $A _ { 2 }$ can be expressed more straightforwardly as follows.

$$
\begin{array} { c } { { ( - R _ { 1 , 3 , 2 } \vee x _ { 5 } \le 0 ) \wedge ( - R _ { 1 , 3 , 1 } \vee x _ { 5 } \le 1 ) } } \\ { { \wedge ( - R _ { 1 , 2 , 2 } \vee - R _ { 2 , 2 , 1 } ) \wedge ( - R _ { 1 , 2 , 1 } \vee - R _ { 2 , 2 , 2 } ) } } \\ { { \wedge ( x _ { 4 } \le 1 \vee - R _ { 2 , 3 , 1 } ) \wedge ( x _ { 4 } \le 0 \vee - R _ { 2 , 3 , 2 } ) } } \end{array}
$$

In summary, connecting two consecutive blocks i and $i + 1$ in the Ladder AMK constraint involves a series of disjunctions, as formally presented in equation (8). This equation can also be optimized by reusing register bits from the SC encoding of the individual blocks, as presented in the example above.

$$
\bigwedge _ { j = 2 p = 1 } ^ { w } ( \sum _ { q = 1 } ^ { w - j + 1 } x _ { i , q } \leq k - p \vee \sum _ { q ^ { \prime } = 1 } ^ { j - 1 } x _ { i + 1 , q ^ { \prime } } \leq p - 1 )\tag{8}
$$

## 3.3 Comparison of Ladder AMK constraint encodings

In this section, we assess the performance of the SCL encoding on the Ladder AMK constraint $( S C L _ { \leq k } )$ and compare it with other encodings that encode each AMK constraint individually. These encodings include Pairwise, Sequential Counter (Seq) [15], Binary Decision Diagrams (BDD) [18], Cardinality Network (Card) [19, 20], and Adder Network (Adder) [17].

Table 1 summarizes the evaluation of diferent encodings used to represent the Ladder AMK constraints over n variables with width w. The encodings are compared based on the number of required auxiliary variables, the number of generated clauses, and the complexity of the clause space in the worst case, specifically when w is approximately $\frac { n } { 2 }$

Pairwise encoding generates a clause for every set of $k + 1$ variables among the w variables in each AMK constraint. These clauses ensure that at least one variable in any set of $k + 1$ variables is assigned a value of false. This means that no more than k variables can be assigned the value true. By eliminating redundant constraints, Pairwise encoding uses $C _ { k + 1 } ^ { w }$ clauses to encode the first AMK constraint and $C _ { k } ^ { w - 1 }$ clauses to encode each of the remaining N − 1 AMK constraints. As a result, the total number of clauses is $C _ { k + 1 } ^ { w } + ( N - 1 ) C _ { k } ^ { w - 1 }$ , leading to a complexity of ${ \mathcal { O } } ( n ^ { k + 1 } )$ .

Sequential Counter (Seq) encoding [15] employs a sequential counting technique. In the context of the AMK constraint, this encoding ensures that in any sequence of w consecutive variables, no more than k variables can be assigned true, thereby maintaining the AMK property. Each $S e q$ encoding for an AMK constraint over w variables requires at most $k ( w - 2 )$ additional variables and generates approximately

Table 1: Size of SAT encoding for a Ladder AMK constraint over n variables with width w, $N = ( n - w ) + 1$ and $\begin{array} { r } { \dot { \boldsymbol M } = \left\lceil \frac { n } { w } \right\rceil } \end{array}$
<table><tr><td rowspan=1 colspan=1>Encoding</td><td rowspan=1 colspan=1>Technique</td><td rowspan=1 colspan=1>Auxiliary variables</td><td rowspan=1 colspan=1>Clauses</td><td rowspan=1 colspan=1>Complexity</td></tr><tr><td rowspan=1 colspan=1>Pairwise</td><td rowspan=4 colspan=1>Encode eachAMK constraintseparately</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $C _ { k + 1 } ^ { w } + ( N - 1 ) C _ { k } ^ { w - 1 }$ </td><td rowspan=1 colspan=1> $O ( n ^ { k + 1 } )$ </td></tr><tr><td rowspan=1 colspan=1>Seq</td><td rowspan=1 colspan=1> $N k ( w - 2 )$ </td><td rowspan=1 colspan=1> $N ( 3 k w - 7 k + w )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } k )$ </td></tr><tr><td rowspan=1 colspan=1>BDD</td><td rowspan=1 colspan=1> $N ( w - k ) ( k + 1 )$ </td><td rowspan=1 colspan=1> $2 N ( w - k ) ( k + 1 )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } k )$ </td></tr><tr><td rowspan=1 colspan=1>Card</td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$ </td></tr><tr><td rowspan=1 colspan=1>Adder</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td></tr><tr><td rowspan=1 colspan=1> $S C L _ { \leq k }$ </td><td rowspan=1 colspan=1>Encode set ofAMK constraints</td><td rowspan=1 colspan=1> $\begin{array} { r } { ( 2 M - 2 ) ( w k - \frac { k ^ { 2 } + k } { 2 } - 1 ) } \end{array}$ </td><td rowspan=1 colspan=1> $\overline { { 9 M k w - 5 M k ^ { 2 } - M w - 7 M k - 9 k w } }$  $+ 5 k ^ { 2 } - 2 M + 2 w + 6 k + 2$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n k )$ </td></tr></table>

3kw−7k+w clauses, leading to a total of $N k ( w { - 2 } )$ auxiliary variables, $N ( 3 k w - 7 k + w )$ clauses, and a complexity of $O ( n ^ { 2 } k )$

Binary Decision Diagrams (BDD) encoding [18] represents the AMK constraint as a circuit of if-then-else (ITE) gates. The goal of this encoding is to constrain the circuit’s inputs to ensure that signal propagation through the gates ultimately reaches the final gate. For an AMK constraint on w variables, BDD encoding constructs a circuit with approximately $( w - k ) ( k + 1 )$ gates, corresponding to the same number of auxiliary variables. Each gate in the circuit has two outputs, representing the boolean values true and f alse, leading to a total of approximately $2 ( w - k ) ( k + 1 )$ clauses. In summary, encoding a Ladder AMK constraint using the BDD encoding requires approximately $N ( w - k ) ( k + 1 )$ auxiliary variables, $2 N ( w - k ) ( k + 1 )$ clauses, and has a complexity of $O ( n ^ { 2 } k )$

Leveraging the odd-even merging technique in Sorting Network encoding [21], As´ın et al. introduced the Half Sorting Network and Half Merging Network encodings [20], which reduce the number of clauses by half. Building on these approaches, As´ın et al. further proposed the Simplified Half-Merging Network, a streamlined version of the Half Merging Network. This encoding later served as the foundation for the development of Cardinality Network $( C a r d )$ encoding [20]. For an AMK constraint of w variables, Card encoding uses $\mathcal { O } ( n \mathrm { l o g } _ { 2 } ^ { 2 } k )$ auxiliary variables and $\mathcal { O } ( n \mathrm { l o g } _ { 2 } ^ { 2 } k )$ clauses. When applied to Ladder AMK constraints, this encoding requires a total of ${ \bar { \mathcal { O } } } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$ auxiliary variables, $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$ clauses, and has a complexity of $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } k )$

Adder Network (Adder) encoding [17] translates each AMK constraint into a boolean circuit composed of half-adders and full-adders arranged as a binary adder tree. Through this circuit, a binary representation of the sum of input variables is produced, and a binary comparator circuit is used to ensure that this sum does not exceed k. For an AMK constraint over w variables, Adder encoding introduces a ${ \mathcal { O } } ( n )$ sized translation [17]. Consequently, encoding a Ladder AMK constraint using Adder requires $\mathcal { O } ( n ^ { 2 } )$ auxiliary variables and clauses.

Our analysis reveals that the $S C L _ { < k }$ encoding results in fewer clauses than other encodings. For simplicity, we assume that the size of the last subset is equal to that of the others, although it is typically smaller. Through block decomposition, one AMK block is generated for the first subset and one for the last subset. For the remaining $\left( M - 2 \right)$ subsets, a total of $\left( M - 2 \right)$ AMK blocks and $\left( M - 2 \right)$ connection blocks are generated. Each AMK block, which is encoded using equations (1) to (7), requires $4 k w - 2 k ^ { 2 } - 4 k - 1$ clauses, whereas each connection block, which is encoded using equations (1) to (6), uses $( w - k )$ fewer clauses than the AMK blocks. Furthermore, connecting these $2 + ( M - 2 ) + ( M - 2 ) = 2 M - 2$ blocks necessitates $M - 1$ connections, with each connection requiring $( w - k ) k$ clauses. In total, the number of clauses employed by the $S C L _ { \leq k }$ encoding is given by:

$$
{ \begin{array} { l c c c } { { \mathrm { ~ A M K - b l o c k s - c l a u s e s } } } & { = } & { 2 + ( M - 2 ) } & { \mathbf { x } } & { \qquad ( 4 k w - 2 k ^ { 2 } - 4 k - 1 ) } \\ { { \mathrm { ~ C o n n e c t i o n - b l o c k s - c l a u s e s } } } & { = } & { ( M - 2 ) } & { \mathbf { x } } & { \qquad ( 4 k w - 2 k ^ { 2 } - 4 k - 1 - w + k ) } \\ { { \mathrm { ~ C o n n e c t i n g - c l a u s e s } } } & { = } & { ( M - 1 ) } & { \mathbf { x } } & { \qquad ( w - k ) k } \end{array} }
$$

$$
\begin{array} { r l r } { \mathrm { T o t a l - c l a u s e s } } & { { } \quad } & { = } & { 9 M k w - 5 M k ^ { 2 } - M w - 7 M k - 9 k w + 5 k ^ { 2 } - 2 M + 2 w + 6 k + 2 = 6 k w - 9 k w } \end{array}
$$

In the calculation of auxiliary variable usage, $S C L _ { \leq k }$ encoding employs $w k - { \textstyle { \frac { k ^ { 2 } + k } { 2 } } }$ register bits per block, including both AMK blocks and connection blocks. By referring to Figure $\mathrm { 8 , }$ we also observe that the first register bit is equivalent to the first variable, i.e., $R _ { 1 , 1 , 1 } \equiv x _ { 4 }$ . Leveraging this insight, we can substitute that register bit with its corresponding variable to further decrease the number of auxiliary variables required by each block by one, resulting in $\begin{array} { r } { w k - \frac { k ^ { 2 } + k } { 2 } - 1 } \end{array}$ . Consequently, the total number of auxiliary variables needed for the entire Ladder constraint becomes $( 2 M - 2 ) ( w k -$ ${ \frac { k ^ { 2 } + k } { 2 } } - 1 )$

The analysis above demonstrates the eficiency of our proposed encoding. In terms of the required auxiliary variables and the number of generated clauses, the $S C L _ { \leq k }$ encoding outperforms the $S e q ,$ BDD, Card, and Adder encodings. Specifically, while these four encodings require at least $\mathcal { O } ( n ^ { 2 } )$ auxiliary variables and clause space, $S C L _ { < k }$ only needs O(Mwk), which is approximately $\mathcal { O } ( n k )$ . Additionally, although the Pairwise encoding does not utilize auxiliary variables, its complexity of $O ( n ^ { k + 1 } )$ is significantly higher than our linear complexity of $\mathcal { O } ( n k )$ . We provide a more comprehensive demonstration in the experimental evaluation (Section 7.1.2), which clarifies the comparative analysis and highlights the superiority of $S C L _ { \leq k }$ over Pairwise, Seq, BDD, Card and Adder encodings.

## 4 Encodings for Ladder AMO constraint

In this section, we introduce the SCL encoding formulas for the Ladder AMO constraint $( S C L _ { \leq 1 } )$ , derived from those for the Ladder AMK constraint $( S C L _ { \leq k } )$ Additionally, we present Duplex encoding [14], a decomposition-based encoding that has proven to be an efective encoding for the Ladder AMO constraint.

## 4.1 $S C L _ { \bf < 1 }$ encoding

## a. Encoding each block in the Ladder AMO constraint

The Ladder AMO constraint is essentially a specific instance of the Ladder AMK constraint. Therefore, its block encoding can be directly derived from that of the Ladder AMK. When $k = 1$ , the formulas from (9) to (13) below are utilized to encode the $i ^ { t h }$ block in the Ladder AMO constraint.

$$
\bigwedge _ { j = 1 } ^ { w - 1 } x _ { i , j } \to R _ { i , j , 1 } \overset { k = 1 } { \Rightarrow } \bigwedge _ { j = 1 } ^ { w - 1 } x _ { i , j } \to R _ { i , j , 1 }\tag{9}
$$

$$
\bigwedge _ { j = 2 } ^ { w - 1 } \bigwedge _ { s = 1 } ^ { m ( j - 1 , k ) } R _ { i , j - 1 , s }  R _ { i , j , s } \overset { k = 1 } { \Rightarrow } \bigwedge _ { j = 2 } ^ { w - 1 } R _ { i , j - 1 , 1 }  R _ { i , j , 1 }\tag{10}
$$

$$
\bigwedge _ { j = 2 } ^ { w - 1 } \bigwedge _ { s = 2 } ^ { ( j , k ) } x _ { i , j } \wedge R _ { i , j - 1 , s - 1 } \to R _ { i , j , s } \stackrel { k = 1 } { \Rightarrow } \emptyset
$$

$$
\bigwedge _ { j = 1 } ^ { k } \neg x _ { i , j } \to \neg R _ { i , j , j } \overset { k = 1 } { \Rightarrow } \neg x _ { i , 1 } \to \neg R _ { i , 1 , 1 }\tag{11}
$$

$$
\prod _ { j = 2 } ^ { w - 1 } \bigwedge _ { s = 2 } ^ { m i n ( j , k ) } \neg R _ { i , j - 1 , s - 1 } \to \neg R _ { i , j , s } \stackrel { k = 1 } { \Rightarrow } \emptyset
$$

$$
\bigwedge _ { j = 2 } ^ { w - 1 } \bigwedge _ { s = 1 } ^ { m i n ( j - 1 , k ) } \neg x _ { i , j } \wedge \neg R _ { i , j - 1 , s } \to \neg R _ { i , j , s } \underset { \Rightarrow } { \overset { k = 1 } { \Rightarrow } } \bigwedge _ { j = 2 } ^ { w - 1 } \neg x _ { i , j } \wedge \neg R _ { i , j - 1 , 1 } \to \neg R _ { i , j , 1 }\tag{12}
$$

$$
\bigwedge _ { j = k + 1 } ^ { w } x _ { i , j } \to \lnot R _ { i , j - 1 , k } \overset { k = 1 } { \Rightarrow } \bigwedge _ { j = 2 } ^ { w } x _ { i , j } \to \lnot R _ { i , j - 1 , 1 }\tag{13}
$$

## b. Clauses connecting two adjacent blocks

Similar to the encoding of individual blocks, the connection between blocks in the Ladder AMO constraint can also be derived from the block connection in the Ladder AMK constraint (refer to equation (8)) by setting k = 1. Specifically,

$$
\begin{array} { c } { { \displaystyle \bigwedge _ { j = 2 } ^ { w } \bigwedge _ { p = 1 } ^ { k \quad w - j + 1 } x _ { i , q } \leq k - p \vee \sum _ { q ^ { \prime } = 1 } ^ { j - 1 } x _ { i + 1 , q ^ { \prime } } \leq p - 1 ) } } \\ { { \qquad \quad \forall _ { { } \atop { q = 1 } } \bigwedge _ { q = 1 } ^ { k } x _ { i , q } \leq k - p \vee \sum _ { q ^ { \prime } = 1 } ^ { j - 1 } x _ { i + 1 , q ^ { \prime } } \leq p - 1 ) } } \\ { { \qquad \forall _ { { } } k = 1 } } \end{array}
$$

$$
\bigwedge _ { j = 2 } ^ { w } ( \sum _ { q = 1 } ^ { w - j + 1 } x _ { i , q } \leq 0 \vee \sum _ { q ^ { \prime } = 1 } ^ { j - 1 } x _ { i + 1 , q ^ { \prime } } \leq 0 )\tag{14}
$$

## 4.2 Duplex encoding

Fazekas et al. introduced the Duplex [14] for encoding Ladder AMO constraints. In this approach, the AMO and At-Most-Zero (AMZ) sub-expressions obtained through the decomposition of Ladder AMO constraints are encoded by applying Binary Decision Diagrams (BDD) encoding [22, 23] for corresponding windows of variables (note that the term “window” is equivalent to our term “subset”). Each window is encoded using two components: a forward BDD and a backward BDD. The forward BDD is constructed from the leftmost to the rightmost variable, while the backward BDD is constructed in the reverse direction, from right to left.

![](images/3ff59768e080a13e9c32b8bcbedc145e13ee5d455e3bf0328dae0748df0b5b27.jpg)  
Fig. 9: Forward and backward BDDs for Ladder AMO constraint. Source: Fazekas et al. [14]

Figure 9 illustrates the construction of two BDDs with the variable arrangements of $x _ { 1 } \ \prec \ x _ { 2 } \ \prec \ x _ { 3 } \ \prec \ x _ { 4 }$ and $x _ { 8 } \prec x _ { 7 } \prec x _ { 6 } \prec x _ { 5 }$ , respectively. This construction involves making decisions for each variable, using their values to branch the necessary conditions of the original constraints. For instance, to maintain the condition $x _ { 1 } +$ $x _ { 2 } + x _ { 3 } + x _ { 4 } \le 1$ , if $x _ { 1 }$ is true, then $x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 0$ must hold; conversely, if $x _ { 1 }$ is $f a l s e$ , the condition $x _ { 2 } + x _ { 3 } + x _ { 4 } \le 1$ must be satisfied.

To ensure there is at most one variable assigned true within the AMO constraints of Ladder AMO constraint, Duplex encoding bonds the constructed BDDs together. Specifically, for each successive pair of windows $\left\{ \omega _ { i } , \omega _ { i + 1 } \right\}$ , where $\omega _ { i } ^ { f }$ is the forward

BDD of $\omega _ { i }$ and $\omega _ { i + 1 } ^ { b }$ is the backward BDD of $\omega _ { i + 1 }$ , the $j ^ { t h }$ layer of $\omega _ { i } ^ { f }$ is connected to the $( w - j + 2 ) ^ { t h }$ layer of $\boldsymbol { \omega } _ { i + 1 } ^ { b }$ , where $2 \leq j \leq w$ . For instance, bonding two BDDs as shown in Figure 9 requires the following clauses.

$$
\begin{array} { r } { \wedge ( x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 1 ) \wedge ( x _ { 5 } \leq 1 ) \wedge ( ( x _ { 2 } + x _ { 3 } + x _ { 4 } \leq 0 ) \vee ( x _ { 5 } \leq 0 ) ) } \\ { \wedge ( x _ { 3 } + x _ { 4 } \leq 1 ) \wedge ( x _ { 5 } + x _ { 6 } \leq 1 ) \wedge ( ( x _ { 3 } + x _ { 4 } \leq 0 ) \vee ( x _ { 5 } + x _ { 6 } \leq 0 ) ) } \\ { \wedge ( x _ { 4 } \leq 1 ) \wedge ( x _ { 5 } + x _ { 6 } + x _ { 7 } \leq 1 ) \wedge ( ( x _ { 4 } \leq 0 ) \vee ( x _ { 5 } + x _ { 6 } + x _ { 7 } \leq 0 ) ) } \end{array}
$$

Leveraging the benefits of reusing shared sub-expressions, the Duplex encoding achieves significantly enhanced complexity compared to non-decompose encodings. As shown in the work of Fazekas et al., the Duplex encoding has a complexity of ${ \mathcal { O } } \left( n \right)$ [14], whereas all non-decompose encodings exhibit complexity of $\mathcal { O } \left( n ^ { 2 } \right)$ (refer to Table 2). For a Ladder AMO constraint involving n variables and considering $\begin{array} { r } { M \ = \ \left\lceil \frac { n } { w } \right\rceil } \end{array}$ windows of width w, with $N = ( n - w ) + 1$ being the number of AMO constraints, this encoding generates around $1 3 M w - 1 4 M - 3 w + 2$ clauses and introduces additional $4 M ( w - 1 )$ auxiliary variables.

## 4.3 Encoding comparison for Ladder AMO constraints

In addition to the previously mentioned encodings, which are Pairwise, Sequential Counter (Seq), Binary Decision Diagrams (BDD), Cardinality Network (Card), and Adder Network (Adder), we compare $S C L _ { \leq 1 }$ with the Duplex encoding [14]. Moreover, among the group of encodings that individually handle each AMO constraint, we also include the Product encoding [24].

Table 2: Size of SAT encoding for a Ladder AMO constraint over n variables with width w, $N = ( n - w ) + 1$ and $\begin{array} { r } { M = \left\lceil \frac { n } { w } \right\rceil } \end{array}$
<table><tr><td rowspan=1 colspan=1>Encoding</td><td rowspan=1 colspan=3>Technique</td><td rowspan=1 colspan=1>Auxiliary variables</td><td rowspan=1 colspan=1>Clauses</td><td rowspan=1 colspan=1>Complexity</td></tr><tr><td rowspan=1 colspan=1>Pairwise</td><td rowspan=2 colspan=3></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $\begin{array} { r } { \frac { ( w - 1 ) w } { 2 } + ( N - 1 ) ( w - 1 ) } \end{array}$ </td><td rowspan=1 colspan=1> $\mathcal { O } ( n ^ { 2 } )$ </td></tr><tr><td rowspan=1 colspan=1>Seq</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1> $N ( w - 2 )$ </td><td rowspan=1 colspan=1> $N ( 4 w - 7 )$ </td><td rowspan=1 colspan=1>O(n²)</td></tr><tr><td rowspan=1 colspan=1>BDD</td><td rowspan=1 colspan=2></td><td rowspan=4 colspan=1>Encode eachAMO constraintseparately</td><td rowspan=1 colspan=1> $2 N ( w - 1 )$ </td><td rowspan=1 colspan=1> $4 N ( w - 1 )$ </td><td rowspan=1 colspan=1>O(n²)</td></tr><tr><td rowspan=1 colspan=1>Card</td><td></td><td></td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1>O(n²)</td></tr><tr><td rowspan=1 colspan=1>Adder</td><td></td><td></td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1> $O ( n ^ { 2 } )$ </td><td rowspan=1 colspan=1>O(n²)</td></tr><tr><td rowspan=1 colspan=1>Product</td><td></td><td></td><td rowspan=1 colspan=1> $N ( 2 \sqrt { w } + \mathcal { O } \sqrt [ 4 ] { w } )$ </td><td rowspan=1 colspan=1> $N ( 2 w + 4 \sqrt { w } + \mathcal { O } \sqrt [ 4 ] { w } )$ </td><td rowspan=1 colspan=1>O(n²)</td></tr><tr><td rowspan=1 colspan=1>Duplex</td><td rowspan=1 colspan=3>Encode set of</td><td rowspan=1 colspan=1> $4 M ( w - 1 )$ </td><td rowspan=1 colspan=1> $1 3 M w - 1 4 M - 3 w + 2$ </td><td rowspan=1 colspan=1>O(n)</td></tr><tr><td rowspan=1 colspan=1> $S C L _ { \leq 1 }$ </td><td rowspan=1 colspan=3>AMO constraints</td><td rowspan=1 colspan=1> $( 2 M - 2 ) ( w - 2 )$ </td><td rowspan=1 colspan=1> $8 M w - 1 4 M - 7 w + 1 3$ </td><td rowspan=1 colspan=1>O(n)</td></tr></table>

Table 2 presents a summary of the evaluation results for various encoding techniques used to encode Ladder AMO constraints across n variables with a width of w. The comparison is based on several criteria, including the number of required auxiliary variables needed, the total number of generated clauses, and the complexity of the clause space. This evaluation incorporates the study by Fazekas et al. [14] as well as our proposed $S C L _ { < k }$ encoding for the case when $k = 1$

To represent AMO constraints, the Cardinality Network (Card) encoding selects $k = 2$ and generates two outputs, i.e. $c _ { 1 }$ and $c _ { 2 }$ . Then, the output c<sub>2</sub> is imposed to be false, ensuring that no more than one input variable can be assigned true. As a result, the number of auxiliary variables, required clauses, and the overall clause complexity of the Card encoding is $\mathcal { O } ( n ^ { 2 } \log _ { 2 } ^ { 2 } 2 ) = \mathcal { O } ( n ^ { 2 } )$ .

The Product encoding is based on arranging variables in a two-dimensional coordinate system. The values along each axis are treated as additional variables and are then organized into a smaller two-dimensional coordinate system. This process is recursively applied until further decomposition into smaller coordinate systems is no longer possible. Through this representation technique, Product encoding utilizes $N ( 2 \sqrt { w } + \mathcal { O } \sqrt [ 4 ] { w } )$ auxiliary variables and $N ( 2 w + 4 \sqrt { w } + \mathcal { O } \sqrt [ 4 ] { w } )$ clauses to encode a Ladder AMO constraint.

The Duplex encoding employs the decomposition technique and encodes each window separately by generating two BDDs with distinct variable arrangements. After that, these constructed windows are bonded together to maintain the AMO property of the original constraints. As presented in Section 4.2, the Duplex encoding requires approximately $1 3 M w - 1 4 M$ − 3w + 2 clauses and $4 M ( w - 1 )$ auxiliary variables.

![](images/4a8203ce716c6830e95f0bb140ffbb24a41c82ce0fabfc142cbce48df9523db1.jpg)

![](images/8e28b9e843ec100590a17f47320092f49d49af2fef55b85e914cc3c845fe9062.jpg)  
Fig. 10: Comparison of the number of clauses of a Ladder AMO constraint with parameters set to $n ~ = ~ 1 0 0 0$ and $w \in$ (1, 1000).  
Fig. 11: Comparison of the number of auxiliary variables of a Ladder AMO constraint with parameters set to $n = 1 0 0 0$ and $w \in ( 1 , 1 0 0 0 )$ .

The analysis presented above underscores the eficiency of our proposed encoding. Like the $S C L _ { \leq k }$ encoding, the $S C L _ { < 1 }$ encoding outperforms several alternatives, including Pairwise, $S e q ,$ , BDD, Card, and Adder encodings. As detailed in Table 2, the complexity of $S C L _ { \leq 1 }$ is ${ \mathcal { O } } ( n )$ , which is also superior to $\mathcal { O } ( n ^ { 2 } )$ of the Product encoding. Furthermore, among linear complexity encodings, $S C L _ { \leq 1 }$ uses roughly half the number of variables and clauses compared to the Duplex encoding. To distinctly showcase the outstanding performance of our encoding, we present Figures 10 and 11, which illustrate the performance eficiency of the $S C L _ { \leq 1 }$ encoding. Figure 10 depicts the number of clauses, while Figure 11 illustrates the number of auxiliary variables required to encode a set of Ladder AMO constraints for $n = 1 0 0 0$ and $w \in ( 1 , 1 0 0 0 )$ , based on the calculations presented in Table 2.

## 5 SAT solving for the Anti-bandwidth problem

The Anti-bandwidth problem (ABP) is an NP-hard problem with applications in various scheduling scenarios [25], including radio frequency assignment [26], obnoxious facility location [27], and map coloring [28]. In this section, we present SAT formulae, including the Ladder AMO constraints, for solving the ABP.

## 5.1 Problem definition

Let $G = ( V , E )$ be a graph where V and E denote the sets of vertices and edges, respectively. A labeling function $f$ is a bijection from $V$ to $\{ 1 , 2 , \ldots , | V | \}$ , assigning a unique label to each vertex in $V$ . The anti-bandwidth $A B _ { f } ( i )$ of a vertex $i \in V$ under the labeling function $f$ is defined as the minimum absolute diference between the label of i and the labels of its adjacent vertices:

$$
A B _ { f } ( i ) = m i n \{ | f ( i ) - f ( i ^ { \prime } ) | : \{ i , i ^ { \prime } \} \in E \}
$$

The anti-bandwidth of $G$ under the labeling function $f ,$ denoted as $A B _ { f } ( G )$ , is the minimal value among the $A B _ { f } ( i )$ values:

$$
A B _ { f } ( G ) = m i n \{ A B _ { f } ( i ) : i \in V \}
$$

The Anti-bandwidth Problem aims to find the maximum anti-bandwidth value of $G ,$ denoted as $A B ( G )$ , over all feasible labeling functions $f .$

$$
A B ( G ) = \operatorname* { m a x } A B _ { f } ( G )
$$

## 5.2 ILP model for ABP

The ABP is expressible as an Integer Linear Programming (ILP) [10] model with binary (0–1) variables. $\mathrm { B y }$ defining $x _ { i } ^ { l }$ to be a boolean variable that holds true if vertex i is assigned label l, and false otherwise, this model employs two formulas, (VERTICES) and (LABELS), to guarantee that every vertex in the graph receives a unique label. Specifically, equation (VERTICES) ensures that each label is allocated to exactly one vertex, while equation (LABELS) guarantees that each vertex is given precisely one label. Combining these two equations preserves the one-to-one relationship between vertices and labels.

$$
\sum _ { i \in V } x _ { i } ^ { l } = 1 \qquad \forall l \in \{ 1 , \dots , | V | \}\tag{VERTICES}
$$

$$
\sum _ { l = 1 } ^ { | V | } x _ { i } ^ { l } = 1\tag{LABELS}
$$

Based on this approach, Markus Sinnl [10] proposed an iterative formulation, framed as a feasibility problem, to answer the question: “Does there exist a solution with $A B ( G ) \geq k + 1 ? ^ { , }$ . This formulation includes both formulas (VERTICES) and (LABELS), along with equation (OBJ-k), which ensures that the diference between the labels of adjacent vertices i and $i ^ { \prime }$ is strictly greater than k whenever $k + 1$ is a feasible anti-bandwidth.

$$
\sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } ( x _ { i } ^ { l ^ { \prime } } + x _ { i ^ { \prime } } ^ { l ^ { \prime } } ) \leq 1 \qquad \forall \{ i , i ^ { \prime } \} \in E , \forall 1 \leq l _ { 2 } \leq | V | - k\tag{OBJ-k}
$$

For the ABP problem with 10 vertices $( | V | = 1 0 )$ and $k = 3$ , the constraints given by equation (OBJ-k) for any pair of vertices i and $i ^ { \prime }$ are illustrated in Figure 12. These constraints ensure that the labels $l _ { i }$ and $l _ { i ^ { \prime } }$ of vertices i and $i ^ { \prime } ,$ respectively, always satisfy the condition $| l _ { i } - l _ { i ^ { \prime } } | \le k + 1 = 4$ . As an example, if $l _ { i } = 4 ,$ , meaning $x _ { i } ^ { 4 }$ holds true, then to prevent the set of AMO constraints (in Figure 12) from being violated, $x _ { i ^ { \prime } } ^ { 1 } , x _ { i ^ { \prime } } ^ { 2 } , x _ { i ^ { \prime } } ^ { 3 } , x _ { i ^ { \prime } } ^ { 4 } , x _ { i ^ { \prime } } ^ { 5 } , x _ { i ^ { \prime } } ^ { 6 }$ , and $x _ { i ^ { \prime } } ^ { 7 }$ must all be $f a l s e ,$ which implies that $l _ { i ^ { \prime } } \notin$ $\{ 1 , 2 , 3 , 4 , 5 , 6 , 7 \}$ . The remaining possible labels for vertex $i ^ { \prime }$ are {8, 9, 10}, ensuring that the label distance condition between vertices is maintained.

$$
\begin{array} { c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c } { x _ { i } ^ { 1 } } & { + } & { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 1 } } & { + } & { x _ { i ^ { \prime } } ^ { 2 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { \leq 1 } & \\ { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { \leq 1 } & & & & \\ { x _ { 3 } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { + } & { x _ { i ^ { 3 } } ^ { 3 } } & { + } & { x _ { i ^ { 4 } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { + } & { x _ { i ^ { \prime } } ^ { 6 } } & { \leq 1 } & & & \\ { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { + } & { x _ { i } ^ { 7 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } &  \end{array}
$$

Fig. 12: OBJ-k constraint of $\{ i , i ^ { \prime } \}$ with $| V | = 1 0$ and $k = 3 .$

## 5.3 SAT implementation of ABP constraints

## 5.3.1 Decomposition of OBJ − k constraints

Encoding the equation (OBJ-k) requires a total of $| E |$ separate constraints, each corresponding to an edge in the set E. However, in most graphs, the number of edges is significantly greater than the number of vertices. To optimize the encoding of (OBJ-k), Fazekas et al. [14] employ Lemma 1 to decompose the constraints of (OBJ-k) as follows:

$$
\sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } ( x _ { i } ^ { l ^ { \prime } } + x _ { i ^ { \prime } } ^ { l ^ { \prime } } ) \leq 1
$$

Lemma 1

$$
\sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } x _ { i } ^ { l ^ { \prime } } \le 1 \wedge \sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } x _ { i ^ { \prime } } ^ { l ^ { \prime } } \le 1 \wedge ( \sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } x _ { i } ^ { l ^ { \prime } } \le 0 \vee \sum _ { l ^ { \prime } = l _ { 2 } } ^ { l _ { 2 + k } } x _ { i ^ { \prime } } ^ { l ^ { \prime } } \le 0 )
$$

Using this decomposition technique, the constraints from equation (OBJ-k) for any edge $\{ x _ { i } , x _ { i ^ { \prime } } \}$ can be represented through the corresponding vertex-level constraints of $x _ { i }$ and $x _ { i ^ { \prime } }$ . These vertex-level constraints can be reused to represent multiple edges, which reduces the total number of constraints needed from |E| to $| V |$

In particular, each AMO constraint in equation (OBJ-k) is decomposed into two sub-AMO constraints, each involving only the variables associated with vertex i or vertex $i ^ { \prime }$ (see Figure 13 for an illustration). By decomposing all AMO constraints in (OBJ-k), the obtained sub-AMO constraints form two Ladder AMO constraints, as presented in Figures 14 and 15. These Ladder constraints can be directly encoded using the proposed $S C L _ { \leq 1 }$ encoding.

Additionally, the decomposition of (OBJ-k) uses disjunctions as connecting clauses to connect vertices i and $i ^ { \prime } ,$ , thereby ensuring that the AMO property is maintained for the (OBJ-k) constraint over the edge $\{ i , i ^ { \prime } \}$ (see Figure 16). These connecting clauses can be encoded using register bits required by the $S C L _ { \leq 1 }$ encoding of the corresponding (OBJ-k) constraints, in accordance with Proposition 2.

Proposition 2. For any integer number i s.t. $1 \leq i < n$ , the constraint

$$
x _ { 1 } + x _ { 2 } + . . . + x _ { n } \leq 0
$$

holds if and only if the following constraint holds:

$$
( x _ { 1 } + . . . + x _ { i } \leq 0 ) \land ( x _ { i + 1 } + . . . + x _ { n } \leq 0 ) .
$$

For instance, the decomposition of $x _ { i } ^ { 2 } + x _ { i } ^ { 3 } + x _ { i } ^ { 4 } + x _ { i } ^ { 5 } + x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { 3 } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 1$ utilizes $x _ { i } ^ { 2 } + x _ { i } ^ { 3 } + x _ { i } ^ { 4 } + x _ { i } ^ { 5 } \leq 0 \vee x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { 3 } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 0 .$ as presented in Figure 13. Applying Proposition $2 , x _ { i } ^ { \dot { 2 } } + x _ { i } ^ { 3 } + \dot { x } _ { i } ^ { 4 } + \dot { x } _ { i } ^ { 5 } \leq \dot { 0 } \ ( \mathrm { r e s p . } \ x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { \dot { 3 } } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 0 )$ can be further decomposed into $x _ { i } ^ { 2 } + \stackrel {  } { x } _ { i } ^ { 3 } + \stackrel {  } { x } _ { i } ^ { 4 } \leq 0 \land x _ { i } ^ { 5 } \leq 0 \ \stackrel {  } { ( \mathrm { r e s p . } } \ x _ { i ^ { \prime } } ^ { \frac {  } { 2 } } + x _ { i ^ { \prime } } ^ { 3 ^ { \ast } } + x _ { i ^ { \prime } } ^ { 4 ^ { \ast } } \leq 0 \land x _ { i ^ { \prime } } ^ { 5 } \leq 0 \land x _ { i ^ { \prime } } ^ { 5 } .$ . These clauses can then be substituted with the corresponding $S C$ register bits, as discussed in Section 3.2.

$$
\begin{array} { c } { { x _ { i } ^ { 2 } + x _ { i } ^ { 3 } + x _ { i } ^ { 4 } + x _ { i } ^ { 5 } + x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { 3 } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 1 } } \\ { { \equiv } } \\ { { ( x _ { i } ^ { 2 } + x _ { i } ^ { 3 } + x _ { i } ^ { 4 } + x _ { i } ^ { 5 } \leq 1 ) \wedge ( x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { 3 } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 1 ) } } \\ { { \wedge ( x _ { i } ^ { 2 } + x _ { i } ^ { 3 } + x _ { i } ^ { 4 } + x _ { i } ^ { 5 } \leq 0 \vee x _ { i ^ { \prime } } ^ { 2 } + x _ { i ^ { \prime } } ^ { 3 } + x _ { i ^ { \prime } } ^ { 4 } + x _ { i ^ { \prime } } ^ { 5 } \leq 0 ) } } \end{array}
$$

Fig. 13: Decomposition of the second AMO constraint in (OBJ-k) constraint of edge {i, i<sup>′</sup>} with |V| = 10 and k = 3.

$$
\begin{array} { c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c } { x _ { i } ^ { 1 } } & { + } & { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { \leq 1 } & { x _ { i } ^ { 1 } } & { + } & { x _ { i ^ { \prime } } ^ { 2 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { \leq 1 } & \\ { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { \leq 1 } & { } & { x _ { i ^ { \prime } } ^ { 2 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { \leq 1 } & \\ { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { \leq 1 } & { } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { + } & { x _ { i ^ { \prime } } ^ { 6 } } & { \leq 1 } & \\ { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { + } & { x _ { i } ^ { 7 } } & { \leq 1 } & { } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { + }  \end{array}
$$

Fig. 14: Ladder AMO constraint associated with vertex i with $| V | = 1 0$ and k = 3.

Fig. 15: Ladder AMO constraint associated with vertex i<sup>′</sup> with |V | = 10 and k = 3.

$$
\begin{array} { c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c } { x _ { i } ^ { 1 } } & { + } & { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { \leq 0 } & { \vee } & { x _ { i ^ { \prime } } ^ { 1 } } & { + } & { x _ { i ^ { \prime } } ^ { 2 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { \leq 0 } \\ { x _ { i } ^ { 2 } } & { + } & { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { \leq 0 } & { \vee } & { x _ { i ^ { \prime } } ^ { 2 } } & { + } & { x _ { i ^ { \prime } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { \leq 0 } \\ { x _ { i } ^ { 3 } } & { + } & { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { \leq 0 } & { \vee } & { x _ { i ^ { 3 } } ^ { 3 } } & { + } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } & { x _ { i ^ { \prime } } ^ { 5 } } & { + } & { x _ { i ^ { \prime } } ^ { 6 } } & { \leq 0 } \\ { x _ { i } ^ { 4 } } & { + } & { x _ { i } ^ { 5 } } & { + } & { x _ { i } ^ { 6 } } & { + } & { x _ { i } ^ { 7 } } & { \leq 0 } & { \vee } & { x _ { i ^ { \prime } } ^ { 4 } } & { + } &  x _ { i ^ { \prime } } ^ { 5 } \end{array}
$$

Fig. 16: Connecting clauses obtained from the decomposition of (OBJ-k) constraint of edge {i, i<sup>′</sup>} with $| V | = 1 0$ and $k = 3$

To enhance our approach, we implement the constraint (LABELS) by utilizing auxiliary variables when constructing $S C L _ { \leq 1 }$ for each vertex in the graph. This strategy improves the optimization of the generated clauses, providing a more eficient alternative to encoding with $x _ { i } ^ { l }$ for the constraint (LABELS). The remaining (VERTICES) constraint is applied with exactly one encoding in SAT using a combination of Product and Seq encodings.

## 5.3.2 Symmetry breaking

Referring to Markus Sinnl [10], for every labeling $f$ of an n-vertex graph, there exists a reversed labeling $f ^ { \prime } { } _ { ; }$ where each label in $f ^ { \prime }$ is computed from its corresponding label in $f$ using the following linear transformation:

$$
f ^ { \prime } = n + 1 - f .
$$

This transformation ensures a one-to-one correspondence between the values in $f$ and those in $f ^ { \prime } .$ . Consequently, if $f$ satisfies $( V E R T I C E S )$ and $( L A B E L S )$ constraints, then $f ^ { \prime }$ also satisfies them. Moreover, this transformation also ensures that $f ^ { \prime }$ maintains the same anti-bandwidth as $f ,$ which is proven as follows.

$$
\begin{array} { r c l } { | f ^ { \prime } ( i ) - f ^ { \prime } ( i ^ { \prime } ) | } & { = } & { | ( n + 1 - f ( i ) ) - ( n + 1 - f ( i ^ { \prime } ) ) | } \\ & { = } & { | f ( i ^ { \prime } ) - f ( i ) | } \end{array}
$$

Based on this observation, we apply the symmetry breaking technique [29] to reduce the search space. In our implementation of ABP, we employ symmetry breaking at one selected node using two diferent configurations: the first node and the highest-degree node.

## 6 SAT solving for the Nurse Rostering problem

## 6.1 Problem definition and ILP model

The Nurse Rostering Problem (NRP) aims to construct a work schedule for one or more nurses over a planning horizon of d days. On each day, a nurse may be assigned to a day shift (D), an evening shift (E), or a night shift (N). If the nurse is not assigned to any shift $( \mathrm { D } , \mathrm { E } , \mathrm { o r ~ N } )$ on a day, the nurse has an of-day (O). The schedule must satisfy a set of constraints related to work frequency and rest periods in order to ensure nurses’ well-being and workability.

This study considers the NRP variant used by Bergman et al. [5], which focuses on designing a schedule for a single nurse, subject to the scheduling requirements listed in Table 3. Although this variant focuses on a single nurse, it captures the essential workload distributions and restrictions that also arise in the multi-nurse problem.

Let $x _ { i , j }$ be a boolean variable that denotes whether the nurse is assigned to shift $j$ on day i. More precisely, if the nurse is assigned to shift $j$ on day $i ,$ then $x _ { i , j } =$ true. Otherwise, if the nurse is not assigned to shift $j$ on day $i ,$ then $x _ { i , j } = f a l s e$ Additionally, $x _ { i , O } = t r u e$ indicates that the nurse has an of-day on day i. Otherwise, the nurse must be assigned to exactly one of the three working shifts on day i. Since the nurse either has to work exactly one shift per day or has an of-day, the assignment variables must satisfy

$$
\sum _ { j \in \{ D , E , N , O \} } x _ { i , j } = 1 \qquad \forall i = 1 , \ldots , d .
$$

Table 4 presents the ILP model to construct the schedule that satisfies the requirements outlined in Table 3. In this table, each schedule requirement, except the first one, is expressed as a sequence of cardinality constraints.

Table 3: Schedule requirements of NRP [5].  
Table 4: ILP model of NRP.
<table><tr><td>1. At most 1 work shift every day</td><td></td><td>1.</td><td> $\begin{array} { r } { \sum _ { j \in \{ D , E , N , O \} } x _ { i , j } = 1 } \end{array}$ </td><td> $\forall i \in 1 , \ldots , d$ </td></tr><tr><td>2.</td><td>At least 20 work shifts every 28 days</td><td>2.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 2 7 } \neg x _ { i ^ { \prime } , O } \geq 2 0$ </td><td> $\forall i \in { 1 , \ldots , d - 2 7 }$ </td></tr><tr><td>3.</td><td>At least 4 off-days every 14 days</td><td>3.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , O } \geq 4$ </td><td> $\forall i \in { 1 , \ldots , d - 1 3 }$ </td></tr><tr><td>4.</td><td>At least 1 night shift every 14 days</td><td>4.</td><td> $\begin{array} { r } { \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , N } \geq 1 } \end{array}$ </td><td> $\forall i \in { 1 , \ldots , d - 1 3 }$ </td></tr><tr><td>5.</td><td>At most 4 night shifts every 14 days</td><td>5.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , N } \leq 4$ </td><td> $\forall i \in { 1 , \ldots , d - 1 3 }$ </td></tr><tr><td>6.</td><td>At least 4 evening shifts every 14 days</td><td>6.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , E } \geq 4$ </td><td> $\forall i \in { 1 , \ldots , d - 1 3 }$ </td></tr><tr><td>7.</td><td>At most 8 evening shifts every 14 days</td><td>7.</td><td> $\begin{array} { r } { \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , E } \leq 8 } \end{array}$ </td><td> $\forall i \in { 1 , \ldots , d - 1 3 }$ </td></tr><tr><td>8.</td><td>Nights shifts is not on consecutive days</td><td>8.</td><td> $\begin{array} { r } { \sum _ { i ^ { \prime } = i } ^ { i + 1 } x _ { i ^ { \prime } , N } \leq 1 } \end{array}$ </td><td> $\forall i \in { 1 , \ldots , d - 1 }$ </td></tr><tr><td>9.</td><td>At least 2 evening/night shifts every 7 days</td><td>9.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 6 } x _ { i ^ { \prime } , E } + x _ { i ^ { \prime } , N } \geq 2$ </td><td> $\forall i \in { 1 , \ldots , d - 6 }$ </td></tr><tr><td></td><td>10. At most 4 evening/night shifts every 7 days</td><td>10.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 6 } x _ { i ^ { \prime } , E } + x _ { i ^ { \prime } , N } \leq 4$ </td><td> $\forall i \in { 1 , \ldots , d - 6 }$ </td></tr><tr><td></td><td>11. At most 6 work shifts every 7 days</td><td>11.</td><td> $\textstyle \sum _ { i ^ { \prime } = i } ^ { i + 6 } \neg x _ { i ^ { \prime } , O } \leq 6$ </td><td> $\forall i \in { 1 , \ldots , d - 6 }$ </td></tr></table>

## 6.2 SAT implementation of NRP constraints

The ILP model in Table 4 primarily consists of sequence constraints with ladder structure over the planning horizon. Some of them are naturally the Ladder AMK constraint with $k \geq 2$ , which can be encoded by applying the proposed encoding directly. However, several of these constraints are of the At-Least-K type with $k \geq 2 ,$ such as constraints (2), (3), (6), and (9). Therefore, to leverage the advantages of our proposed encoding, we systematically transform these constraints into equivalent Ladder At-Most-K constraints.

In particular, constraint (2) requires at least 20 working days in every 28 consecutive days. Since each day is either a working day or an of-day, this constraint is equivalent to limiting the number of of-days to at most 8 in every such consecutive days.

$$
\sum _ { i ^ { \prime } = i } ^ { i + 2 7 } \neg { x _ { i ^ { \prime } , O } } \geq 2 0 \equiv \sum _ { i ^ { \prime } = i } ^ { i + 2 7 } x _ { i ^ { \prime } , O } \leq 8
$$

Similarly, constraint (3), which requires at least 4 of-days every 14 days, can be reformulated as restricting the number of working days to at most 10.

$$
\sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , O } \geq 4 \equiv \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } \neg x _ { i ^ { \prime } , O } \leq 1 0
$$

Constraint (6), imposing at least 4 evening shifts every 14 days, can be reformulated as restricting the number of days the nurse does not work in the evening to at most 10.

$$
\sum _ { i ^ { \prime } = i } ^ { i + 1 3 } x _ { i ^ { \prime } , E } \geq 4 \equiv \sum _ { i ^ { \prime } = i } ^ { i + 1 3 } \neg x _ { i ^ { \prime } , E } \leq 1 0
$$

Since the nurse has to work at most 1 shift every day, constraint (9), which states that the nurse has to work at least 2 evening or night shifts every 7 days, is equivalent to imposing that the total number of day shifts and of-days is at most 5 every 7 days.

$$
\sum _ { i ^ { \prime } = i } ^ { i + 6 } x _ { i ^ { \prime } , E } + x _ { i ^ { \prime } , N } \geq 2 \equiv \sum _ { i ^ { \prime } = i } ^ { i + 6 } x _ { i ^ { \prime } , D } + x _ { i ^ { \prime } , O } \leq 5
$$

## 7 Experimental Evaluation

We conducted three experiments to evaluate the efectiveness of our proposed SAT encoding SCL. The first experiment compares the solving times of SCL against other SAT encodings when addressing single Ladder AMO constraints (k = 1) and Ladder AMK constraints $( k > 1 )$ . The second and third experiments, respectively, focus on evaluating $S C L _ { \leq 1 }$ and $S C L _ { \leq k }$ in solving the Anti-bandwidth Problem and Nurse Rostering Problem. These three experiments were performed on a cluster provided by the Google Cloud Platform<sup>1</sup> featuring $\it 8 \ v C P U s , 4$ cores, and 64GB of memory, running the Debian GNU/Linux 12 operating system. The SAT solver used was CaDiCaL version 1.2.1 [30].

## 7.1 Single Ladder AMK constraint

The number of variables and clauses generated by the $S C L _ { \leq 1 }$ and $S C L _ { \leq k }$ encodings, compared to other SAT encodings, is detailed in Tables 1 and 2 of the paper. In this experiment, we aim to evaluate the solving time of SCL relative to other SAT encodings when given a Ladder AMK constraint as input, while varying the number of variables and the width w.

## 7.1.1 Solving time of the Ladder AMO constraint

The experimental dataset for the Ladder AMO constraint consists of a fixed number of variables, specifically n = 1000. The width w ranges from 5 to 995 in increments of 5 (i.e., 5, 10, 15, ..., 995). The diferent SAT encodings compared to the $S C L _ { \leq 1 }$ are detailed in Table 2. These encodings include Pairwise, Sequential Counter (Seq), Binary Decision Diagrams (BDD), Cardinality Network (Card), Adder Network (Adder), Product, and Duplex. Figures 17 and 18 illustrate the solving times for the various encodings.

Among the top three encodings, as shown in Figure 18, $S C L _ { \leq 1 }$ emerged as the most efective, surpassing Duplex, which was previously recognized as an eficient encoding [14]. Overall, the results indicate that, in most cases, the $S C L _ { \leq 1 }$ encoding significantly outperforms all other SAT encodings in terms of time eficiency. Additionally, the efectiveness of the generated clauses and the necessary auxiliary variables for $S C L _ { \leq 1 }$ are discussed earlier in Section 4.3.

![](images/1b823d95859d22e9447696dd54f51499c51bd6dbe5e0e64ad91422490b2d3be9.jpg)  
Fig. 17: Solving time of various encodings on the Ladder AMO constraints with $n =$ 1000 and $w \in \{ 5 , 1 0 , 1 5 , . . . , 9 9 5 \}$

![](images/930774f2b8fe476bf80ea396d7b715c6ed4e43e5fbe5e815ffeb082f3b83c900.jpg)  
Fig. 18: Solving time of the top three encodings on the Ladder AMO constraints with $n = 1 0 0 0$ and $w \in \{ 5 , 1 0 , 1 5 , . . . , 9 9 5 \}$

## 7.1.2 Solving time of the Ladder AMK constraint

To compare the solving times of $S C L _ { \leq k }$ with other SAT encodings, we employed a dataset of Ladder AMK constraints that included n variables and a width w. The values for n were chosen from the range {100, 200, 300, . . . , 5000}, while the values for w were taken from the set {10, 20, 50, 100}. We define k as $k = \begin{array} { l } { { \frac { 1 } { 5 } } w } \end{array}$ (which means $k \in \{ 2 , 4 , 1 0 , 2 0 \}$ ). Each instance was allowed to run for a maximum of 600 seconds. The SAT encodings compared to $S C L _ { \leq k }$ are detailed in Table 1, which includes Pairwise, Seq, BDD, Adder, and Card. Product encoding is excluded from this experiment, as it is less eficient for solving AMK constraints.

The solving times of SAT encodings for four values of k are illustrated in Figures 19, 20, 21, and 22. The encoding $S C L _ { \leq k }$ achieves the fastest solving times for $\{ w , k \} = \{ 2 0 , 4 \} , \{ 5 0 , 1 0 \}$ , and {100, 20}, and it ranks second after the Pairwise encoding for $\{ w , k \} = \{ 1 0 , 2 \}$ . Compared to Seq, BDD, Card, and Adder, $S C L _ { \leq k }$ consistently demonstrates superior performance regarding solving time.

For instance, when $n = 5 0 0 0$ , the solving time for $S C L _ { \leq k }$ is approximately onethird of the time required by the second fastest encoding, BDD. Specifically, for $\{ w , k \} = \{ 2 0 , 4 \} , \ S C L _ { \le k }$ solves the problem in 6.6 milliseconds, while BDD takes 18.1 milliseconds. For $\{ \overline { { w } } , k \} = \{ 5 0 , 1 0 \} , S C L _ { \le k }$ takes 23 milliseconds compared to $B D D \mathrm { { ^ { * } s } }$ 106 milliseconds. For $\{ w , k \} = \{ 1 0 0 , 2 0 \}$ , $S C L _ { \leq k }$ finishes in 46.6 milliseconds, while BDD takes 298.4 milliseconds. The Pairwise encoding is the fastest for $\{ w , k \} = \{ 1 0 , 2 \}$ but begins to slow down at $\{ w , k \} = \{ 2 0 , 4 \}$ . It times out for all input sizes n within the 600-second limit when solving for $\{ w , k \} = \{ 5 0 , 1 0 \}$ and {100, 20}.

![](images/4ce92930e2eae52829fe0331ae67be04193ebe456a2accae430d09ab808e2094.jpg)  
Fig. 19: Solving time of Ladder AMK constraints with $w = 1 0$ and $k = 2 .$

![](images/94a75d6a7de64f5e32a8fc2f4575b32b0cb52f03c38bf25ad0a18658619bcbee.jpg)  
Fig. 20: Solving time of Ladder AMK constraints with $w = 2 0$ and $k = 4$

![](images/bdb9da5bfec476e80310136ade60d1680c0c03ac75ca2ce2c95d890fa5bed335.jpg)  
Fig. 21: Solving time of Ladder AMK constraints<sup>2</sup> with $w = 5 0$ and $k = 1 0$

![](images/70297ef438a65147c157fe89f3c9da44e0d410b65c4150f359720c59909eb22b.jpg)  
Fig. 22: Solving time of Ladder AMK constraints<sup>2</sup> with $w = 1 0 0$ and $k = 2 0$

## 7.2 Computation Results for the Anti-bandwidth Problem

## 7.2.1 Datasets and Implementation Details

The second framework evaluates the efectiveness of the proposed encoding in addressing the Anti-bandwidth Problem. It utilizes a dataset consisting of 24 matrices from the Harwell-Boeing Sparse Matrix Collection [31]. This collection includes 12 small instances and 12 large instances. Detailed information about these 24 matrices can be found in Table 5, which lists the names of the matrices, the number of vertices |V|, and the number of edges |E|. Additionally, the corresponding lower and upper bounds (LB and UB) are provided by Fazekas et al. [14].

In this framework, we implemented the $S C L _ { \leq 1 }$ encoding alongside four SAT-based encodings: Product, Pairwise, Sequential Counter, and Duplex. Throughout the implementation, we followed a consistent sequential process. We began by setting the lower bound (LB) as the starting width for the Ladder AMO constraints. If the SAT solver returns a satisfiable (SAT) result, we increase the width by 1 and restart the process. Conversely, if the solver returns unsatisfiable (UNSAT) or reaches the upper bound (UB), it indicates that we have found the optimal solution, and the iteration process concludes.

Small instances  
Table 5: Harwell-Boeing Sparse Matrix benchmark.
<table><tr><td>Instance</td><td>|V|</td><td>E LB</td><td>UB</td></tr><tr><td>A-pores_1</td><td>30 103</td><td>6</td><td>8</td></tr><tr><td>B-ibm32</td><td>32 90</td><td>9</td><td>9</td></tr><tr><td>C-bcspwr01</td><td>39 46</td><td>16</td><td>17</td></tr><tr><td>D-bcsstk01</td><td>48 176</td><td>8</td><td>9</td></tr><tr><td>E-bcspwr02</td><td>49 59</td><td>21</td><td>22</td></tr><tr><td>F-curtis54</td><td>54 124</td><td>12</td><td>13</td></tr><tr><td>G-will57</td><td>57 127</td><td>12</td><td>14</td></tr><tr><td>H-impcol_b</td><td>59 281</td><td>8</td><td>8</td></tr><tr><td>I-ash85</td><td>85 219</td><td>19</td><td>27</td></tr><tr><td>J-nos4</td><td>100 247</td><td>32</td><td>40</td></tr><tr><td>K-dwt__234</td><td>117 162</td><td>46</td><td>58</td></tr><tr><td>L-bcspwr03</td><td>118 179</td><td>39</td><td>39</td></tr></table>

Large instances
<table><tr><td>Instance |V|</td><td>E LB</td><td>UB</td></tr><tr><td>M-bcsstk06 420</td><td>3720 28</td><td>72</td></tr><tr><td>N-bcsstk07 420</td><td>3720 28</td><td>72</td></tr><tr><td>O-impcol_d 425</td><td>1267 91</td><td>173</td></tr><tr><td>P-can_445 445</td><td>1682 78</td><td>120</td></tr><tr><td>Q-494_bus 494</td><td>586 219</td><td>246</td></tr><tr><td>R-dwt__503 503</td><td>2762 46</td><td>71</td></tr><tr><td>S-sherman4 546</td><td>1341 256</td><td>272</td></tr><tr><td>T-dwt__592 592</td><td>2256 103</td><td>150</td></tr><tr><td>U-662_bus 662</td><td>906 219</td><td>220</td></tr><tr><td>V-nos6 675</td><td>1290 326</td><td>337</td></tr><tr><td>W-685_bus 685</td><td>1282 136</td><td>136</td></tr><tr><td>X-can__715 715</td><td>2975 112</td><td>142</td></tr></table>

Constraint Programming (CP) and Mixed Integer Programming (MIP) are included in this experiment, which are $F _ { e } ( k )$ , CPLEX, CP-MZ-Chufed, CP-SAT, and Gurobi. In our experiment, the $F _ { e } ( k )$ encoding leverages the MIP APIs from IBM ILOG CPLEX Optimization Studio<sup>3</sup> version 20.1, a robust framework that integrates constraint propagation with advanced optimization techniques for eficient problem-solving. The CPLEX approach utilizes the CP APIs of IBM ILOG CPLEX Optimization Studio version 22.1.1. The CP-MZ-Chufed method employs the MiniZinc language [32] version 2.8.6 and incorporates Chufed solver<sup>4</sup> version 0.13.2, which excels in lazy clause generation and is particularly efective for combinatorial problems. The CP-SAT approach relies on the CP-SAT Solver, a versatile solver designed for constraint satisfaction and optimization, leveraging multi-processing and SAT-based techniques to handle complex constraints, which is provided by Google OR-Tools<sup>5</sup> version 9.11.4210. Finally, the Gurobi encoding harnesses the Gurobi Optimizer<sup>6</sup> version 11.0.3, a powerful tool for mathematical optimization, known for its eficiency in handling large-scale constraint-based problems. Note that CPLEX, Gurobi, and CP-SAT are powerful CP and MIP solvers that have been under development for a long time.

In addition to the sequential implementation, referred to as $S C L _ { s e q } .$ which incrementally solves for anti-bandwidth values from LB to UB, we also implemented a multi-process approach, referred to as $S C L _ { p a r }$ , with the aim of providing a fair comparison with state-of-the-art CP and MIP solvers that exploit multi-core architectures, such as CPLEX, Gurobi, and CP-SAT. This implementation initializes multiple processes, with each process invoking a SAT solver to address a diferent anti-bandwidth. The multi-process approach is more eficient than the sequential method because if one process finds a solution for a higher anti-bandwidth, it can terminate other processes working on smaller anti-bandwidths, hence allows the search to concentrate on better values. In this experiment, $S C L _ { p a r }$ is executed with up to four parallel processes invoking the SAT solver.

## 7.2.2 Performance Evaluation

Table 6 summarizes our experimental results on 24 selected matrices, with a time limit of 1800 seconds and a memory limit of 30 GB applied to the Product, Pairwise, Sequential Counter (Seq), Duplex, $S C L _ { s e q } , S C L _ { p a r } , F _ { e } ( k )$ , CPLEX, CP-MZ-Chufed, CP-SAT, and Gurobi approaches. For each problem, the table displays the best solution found by each method, along with the solving time (in seconds) and memory consumption (in MB). The best anti-bandwidth value is highlighted in bold, and the best solving time is underlined. If the computation time exceeds 1800 seconds or the memory usage surpasses 30 GB, a termination signal is triggered, resulting in the process stopping due to a timeout (TO) or memory overload (MO), respectively. In such cases, the highest solved anti-bandwidth is reported as the best result achieved by the technique. If the method fails to solve the problem using the initial lower bound (LB) value, the result is indicated with a “-” character.

![](images/f7e6e75fdb5d4a35996919d2f80404c6ea94915e87f093b081260dd68547dea3.jpg)  
Fig. 23: Comparison of generated clauses between SCL and Duplex.

![](images/d0c1ff6cab7c533694a0cf18b7ee528976cf75a67d2cf1e1b34e2fe864ca0f6d.jpg)  
Fig. 24: Comparison of auxiliary variables between SCL and Duplex.

Figures 23 and 24 compare the number of generated clauses and auxiliary variables used in the ABP encoding across 24 instances. This evaluation considers both the SCL and Duplex encodings, with Duplex recognized as the best encoding according to [14]. Each instance is represented by the first letter of its name (e.g., A for A-pores 1 and X for X-can 715). We analyzed the encodings at the maximum anti-bandwidth achievable by both encodings within specified time and memory limits. The results show that as the size of the instance increases, indicated by the increasing number of vertices |V| and edges $| E |$ , the SCL encoding requires fewer clauses and auxiliary variables than the Duplex encoding. This finding highlights why SCL can outperform Duplex in solving certain ABP instances, particularly in 12 large instances, and it also demonstrates more eficient memory usage.

T<sub>a</sub>bl<sub>e</sub> 6 <sub>:</sub> ABP <sub>so</sub>l<sub>v</sub>i<sub>ng resu</sub>lt<sub>s o</sub>f dif<sub>eren</sub>t <sub>approac</sub>h<sub>es w</sub>ith TO = 1 800<sub>s an</sub>d MO = 30GB <sub>.</sub>
<table><tr><td rowspan=2 colspan=1>Instance</td><td rowspan=2 colspan=1>LB</td><td rowspan=2 colspan=1>UB</td><td rowspan=1 colspan=2>Product</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Pairwise</td><td rowspan=1 colspan=3> $\mathrm { S e q }$ </td><td rowspan=1 colspan=3> $D u p l e x$ </td><td rowspan=1 colspan=3> $S C L _ { s e q }$ </td><td rowspan=1 colspan=3> $S C L _ { p a r }$ </td><td rowspan=1 colspan=3>Fc(k)</td><td rowspan=1 colspan=3>CPLEX</td><td rowspan=1 colspan=3>CP-MZ-Chuffed</td><td rowspan=1 colspan=3>CP-SAT</td><td rowspan=1 colspan=3>Gurobi</td></tr><tr><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td></tr><tr><td rowspan=1 colspan=1>A-pores_1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>122</td><td rowspan=1 colspan=1>77.5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>56.9</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>47.7</td><td rowspan=1 colspan=1>55.4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>14.1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9.15</td><td rowspan=1 colspan=1>19.8</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>8.8</td><td rowspan=1 colspan=1>23.6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3.53</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>579.3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4.75</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5.17</td><td rowspan=1 colspan=1>144.9</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9.84</td><td rowspan=1 colspan=1>84.9</td></tr><tr><td rowspan=1 colspan=1>B-ibm32</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>6.02</td><td rowspan=1 colspan=1>48.9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>18.7</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2.13</td><td rowspan=1 colspan=1>42.9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>8.8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>7.8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4.57</td><td rowspan=1 colspan=1>106.8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1.06</td><td rowspan=1 colspan=1>29.3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=1>120.5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>6.99</td><td rowspan=1 colspan=1>108.2</td></tr><tr><td rowspan=1 colspan=1>C-bcspwr01</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>5.25</td><td rowspan=1 colspan=1>41.4</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>40.1</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>20.7</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>1.11</td><td rowspan=1 colspan=1>11.2</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>4.19</td><td rowspan=1 colspan=1>18.6</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>3.44</td><td rowspan=1 colspan=1>16.8</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>3.59</td><td rowspan=1 colspan=1>97.5</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>28.6</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>1.46</td><td rowspan=1 colspan=1>126.1</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>104.6</td></tr><tr><td rowspan=1 colspan=1>D-bcsstk01</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>9.43</td><td rowspan=1 colspan=1>140.1</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5.36</td><td rowspan=1 colspan=1>48.2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>118.9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>12.6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>13.2</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>147.0</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2.28</td><td rowspan=1 colspan=1>46.8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1.72</td><td rowspan=1 colspan=1>158.4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>15.9</td><td rowspan=1 colspan=1>252.9</td></tr><tr><td rowspan=1 colspan=1>E-bcspwr02</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>8.79</td><td rowspan=1 colspan=1>83.5</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>9.22</td><td rowspan=1 colspan=1>45.8</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>9.91</td><td rowspan=1 colspan=1>86.1</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>17.8</td><td rowspan=1 colspan=1>170.2</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>15.83</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>31.5</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>3.83</td><td rowspan=1 colspan=1>172.6</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>14.3</td><td rowspan=1 colspan=1>198.5</td></tr><tr><td rowspan=1 colspan=1>F-curtis54</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>13.6</td><td rowspan=1 colspan=1>141.6</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>11.1</td><td rowspan=1 colspan=1>127</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>14.1</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>12.3</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>23.9</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>145.8</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1.35</td><td rowspan=1 colspan=1>38.8</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>179.8</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>14.3</td><td rowspan=1 colspan=1>255</td></tr><tr><td rowspan=1 colspan=1>G-will57</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>16.6</td><td rowspan=1 colspan=1>147.9</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>11.9</td><td rowspan=1 colspan=1>73.2</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>133.8</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>14.4</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>13.9</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>26.5</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>18.5</td><td rowspan=1 colspan=1>175.6</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>14.18</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>39.9</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>2.16</td><td rowspan=1 colspan=1>176.2</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>271.3</td></tr><tr><td rowspan=1 colspan=1>H-impcol_b</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4.38</td><td rowspan=1 colspan=1>177</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>52.8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5.22</td><td rowspan=1 colspan=1>147.7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>16.8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3.12</td><td rowspan=1 colspan=1>167.4</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>44.7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>55.5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>2.06</td><td rowspan=1 colspan=1>190.2</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8.17</td><td rowspan=1 colspan=1>262.5</td></tr><tr><td rowspan=1 colspan=1>I-ash85</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>870.8</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>843.6</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>619.8</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>272</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>381</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>496.6</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>460.3</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>288.2</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>374.8</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1723</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1219</td></tr><tr><td rowspan=1 colspan=1>J-nos4</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1119.5</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1070.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1572.8</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>472</td><td rowspan=1 colspan=1>177</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>980</td><td rowspan=1 colspan=1>503</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>720.3</td><td rowspan=1 colspan=1>505.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>927.0</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>272.7</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>384.4</td><td rowspan=1 colspan=1>343.9</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1818</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1690</td></tr><tr><td rowspan=1 colspan=1>K-dwt_234</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1373.3</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1192.6</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1110</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>685</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>530</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1377</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>895.3</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>243.8</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>244.6</td><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1088</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>784.4</td></tr><tr><td rowspan=1 colspan=1>L-bcspwr03</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>15.1</td><td rowspan=1 colspan=1>682.1</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>3.76</td><td rowspan=1 colspan=1>436.5</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>47.7</td><td rowspan=1 colspan=1>856.6</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>40.9</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>11.15</td><td rowspan=1 colspan=1>559.0</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>48.2</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>2.34</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>7.15</td><td rowspan=1 colspan=1>482.4</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>65.9</td><td rowspan=1 colspan=1>321.3</td></tr><tr><td rowspan=1 colspan=1>M-bcsstk06</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>22469</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1726</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1398</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5076</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>12207</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>738.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>3016</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5243</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>320.3</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>N-bcsstk07</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>22470</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1726</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1397</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5291</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>12244</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>737.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>3016</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5250</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>304.1</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>O-impcol.d</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>173</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>22681</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1745</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1802</td><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5366</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>121</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>491.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1410</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5388</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>341</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>P-can--445</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>120</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>27109</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2464</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2156</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>9758</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>869.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1843</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5732</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>411.6</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>Q-494_bus</td><td rowspan=1 colspan=1>219</td><td rowspan=1 colspan=1>246</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>25945</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>29391</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1443</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1301</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4713</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>23775</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>701.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>763.8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>316.8</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>18775</td></tr><tr><td rowspan=1 colspan=1>R-dwt__503</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1634</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1474</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4991</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>10633</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>717.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1950</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>7347</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>424.5</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>S-sherman4</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>272</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>.</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1326</td><td rowspan=1 colspan=1>.</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1073</td><td rowspan=1 colspan=1>.</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4003</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>689.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>888.8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>29192</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1013</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>T-dwt_-592</td><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5357</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2619</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>11445</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1366</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1905</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>9331</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>871.3</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>U-662_bus</td><td rowspan=1 colspan=1>219</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>471</td><td rowspan=1 colspan=1>2498</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>1377</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>122.9</td><td rowspan=1 colspan=1>2600</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>250.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1061</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>328</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>851.3</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>V-nos6</td><td rowspan=1 colspan=1>326</td><td rowspan=1 colspan=1>337</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1836</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1532</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>6317</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>967.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1168</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1260</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1071</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>W-685_bus</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>1431</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>7.57</td><td rowspan=1 colspan=1>1164</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>1162</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>1.95</td><td rowspan=1 colspan=1>332.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1325</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>8223</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>855</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>X-can_-715</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>142</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>3060</td><td rowspan=1 colspan=1>113</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4599</td><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>10388</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>113</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1380</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2442</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>11514</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1013</td><td rowspan=1 colspan=1>MO</td></tr></table>

When solving 12 small graph instances, the SCL encoding yields the best results compared to other methods. The performance of $S C L _ { s e q }$ is competitive with Duplex and outperforms the other SAT-based encodings. Compared to CPLEX, which is one of the top-performing methods among CP and MIP approaches, $S C L _ { p a r }$ finds the optimal anti-bandwidth values for the two instances A-pores 1 and $J { - } n o s 4 .$ which are 6 and 35, respectively. In contrast, CPLEX only proves feasible anti-bandwidths of 6 and 34 for those instances. Moreover, $S C L _ { p a r }$ also performs better on the I-ash85 instance. Compared to the other eficient CP method, $C P \mathrm { - } S A T , S C L _ { p a r }$ delivers better results on the instances J-nos4 and K-dwt 234. MIP-based methods such as Gurobi and F (k) exhibit a weaker performance than $S C L _ { p a r }$

Table 7: ABP solving results of the five best approaches in 12 large instances with TO = 1800s and MO = 30GB.
<table><tr><td rowspan=2 colspan=1>Instance</td><td rowspan=2 colspan=1>LB</td><td rowspan=2 colspan=1>UB</td><td rowspan=1 colspan=3>Duplex</td><td rowspan=1 colspan=3> $S C L _ { s e q }$ </td><td rowspan=1 colspan=3> $S C L _ { p a r }$ </td><td rowspan=1 colspan=3>CPLEX</td><td rowspan=1 colspan=3>CP-SAT</td></tr><tr><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1> $_ \mathrm { O b j - k }$ </td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1> $\mathrm { O b j - k }$ </td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Obj-k</td><td rowspan=1 colspan=1>Time(s)</td><td rowspan=1 colspan=1>MB</td></tr><tr><td rowspan=1 colspan=1>M-bcsstk06</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1726</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1398</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5076</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>738.4</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5243</td></tr><tr><td rowspan=1 colspan=1>N-bcsstk07</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1726</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1397</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5291</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>737.9</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5250</td></tr><tr><td rowspan=1 colspan=1>O-impcol_d</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>173</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1745</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1802</td><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5366</td><td rowspan=1 colspan=1>121</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>491.3</td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5388</td></tr><tr><td rowspan=1 colspan=1>P-can_-445</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>120</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2464</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2156</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>9758</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>869.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5732</td></tr><tr><td rowspan=1 colspan=1>Q-494_bus</td><td rowspan=1 colspan=1>219</td><td rowspan=1 colspan=1>246</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1443</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1301</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4713</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>701.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>316.8</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>R-dwt__503</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1634</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1474</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4991</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>717.6</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>7347</td></tr><tr><td rowspan=1 colspan=1>S-sherman4</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>272</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1326</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1073</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4003</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>689.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>29192</td></tr><tr><td rowspan=1 colspan=1>T-dwt__592</td><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>5357</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>2619</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>11445</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1366</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>9331</td></tr><tr><td rowspan=1 colspan=1>U-662_bus</td><td rowspan=1 colspan=1>219</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>471</td><td rowspan=1 colspan=1>2498</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>1377</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>122.9</td><td rowspan=1 colspan=1>2600</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>3.15</td><td rowspan=1 colspan=1>250.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>328</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>V-nos6</td><td rowspan=1 colspan=1>326</td><td rowspan=1 colspan=1>337</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1836</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1532</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>6317</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>967.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1260</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>W-685_bus</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>1431</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>7.57</td><td rowspan=1 colspan=1>1164</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>1162</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>1.95</td><td rowspan=1 colspan=1>332.6</td><td rowspan=1 colspan=1>136</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>8223</td></tr><tr><td rowspan=1 colspan=1>X-can__715</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>142</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>3060</td><td rowspan=1 colspan=1>113</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>4599</td><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>10388</td><td rowspan=1 colspan=1>113</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>1380</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>11514</td></tr></table>

Table 7 summarizes the experimental results on 12 large problem instances for the five best-performing approaches: Duplex, $S C L _ { s e q } , \ S C L _ { p a r } ,$ , CPLEX, and CP-SAT. Overall, $S C L _ { p a r }$ achieves the best performance among all methods, except for the O-impcol d instance, where CPLEX performs significantly better. Compared to $C P L E X , S C L _ { p a r }$ outperforms it on five instances: M-bcsstk06, N-bcsstk07, $P \mathrm { - } c a n _ { \mathrm { -- } } 4 4 5 .$ , R-dwt 503, and $X { \mathrm { - } } c a n _ { \mathrm { -- } } 7 1 5 .$ In the sequential setting, $S C L _ { s e q }$ consistently matches or improves upon the results obtained by Duplex across all 12 large instances, demonstrating that the advantage over Duplex is inherent in the proposed SCL encoding. Building on this, $S C L _ { p a r }$ represents an improved variant of $S C L _ { s e q }$ that further exploits parallel search to enhance performance, outperforming Duplex on five instances, including O-impcol d, P-can 445, Q-494 bus, T-dwt 592, and $X -$ can 715. Notably, for $P - c a n _ { -- } 4 4 5$ and X-can 715, no other approach achieves results comparable to those of $S C L _ { p a r }$

## 7.3 Computation Results for the Nurse Rostering Problem

## 7.3.1 Datasets and Implementation Details

The third framework evaluates the performance of the SCL encoding on a Nurse Rostering Problem (NRP) variant used by Bergman et al. [5]. This NRP variant involves scheduling for a single nurse over a planning horizon ranging from 40 to 100 days, subject to 11 schedule requirements described in Section 6.

In this framework, we implemented the $S C L _ { \leq k }$ alongside several SAT encoding techniques, including BDD, Card, Adder, Pairwise, SCL, and Seq. For the SAT-based approaches, constraints 1, 4, and 8 in the ILP model are encoded using Pairwise encoding, since they correspond to At-Least-One, At-Most-One, or Exactly-One constraints. For the remaining constraints of ALK type, particularly constraints (2), (3), (6), and (9), if a given SAT encoding cannot handle them directly, we first reformulate these constraints into equivalent AMK constraints, as described in Section 6.2.

We also compare $S C L _ { \leq k }$ with state-of-the-art CP and MIP solvers, namely CPLEX, CP-SAT, and Gurobi. As in the experiments on the Anti-bandwidth Problem, the CPLEX approach leverages the CP APIs of IBM ILOG CPLEX Optimization Studio version 22.1.1, while CP-SAT is provided by Google OR-Tools version 9.11.4210, and Gurobi uses Gurobi Optimizer version 11.0.3.

## 7.3.2 Performance Evaluation

Since almost all approaches can solve the NRP dataset quickly, we execute all the approaches on each instance 10 times and report the average solving time and memory consumption. For each run, the time limit and memory limit are set to 600 seconds and 5 GB, respectively. The results, including both solving time and memory consumption, are summarized in Table 8 and also illustrated in Figures 25 and 26. Figure 25 presents the solving time, while Figure 26 reports the memory consumption.

<table><tr><td rowspan=2 colspan=1>Work period</td><td rowspan=1 colspan=2>40 days</td><td rowspan=1 colspan=2>60 days</td><td rowspan=1 colspan=2>80 days</td><td rowspan=1 colspan=2>100 days</td></tr><tr><td rowspan=1 colspan=1>Time(ms)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Time(ms)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Time(ms)</td><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1> ${ \mathrm { T i m e } } ( { \mathrm { m s } } )$ </td><td rowspan=1 colspan=1>MB</td></tr><tr><td rowspan=1 colspan=1>Adder</td><td rowspan=1 colspan=1>192</td><td rowspan=1 colspan=1>9.6</td><td rowspan=1 colspan=1>302</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>419</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>535</td><td rowspan=1 colspan=1>26.4</td></tr><tr><td rowspan=1 colspan=1>BDD</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>8.9</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>15.2</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>187</td><td rowspan=1 colspan=1>28</td></tr><tr><td rowspan=1 colspan=1>Card</td><td rowspan=1 colspan=1>107</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>143</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>212</td><td rowspan=1 colspan=1>29.5</td></tr><tr><td rowspan=1 colspan=1>Pairwise</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>MO</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>MO</td></tr><tr><td rowspan=1 colspan=1>SCL</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>13.4</td></tr><tr><td rowspan=1 colspan=1>Seq</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>13.9</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>40.2</td><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>47.4</td></tr><tr><td rowspan=1 colspan=1>CPLEX</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>19.4</td></tr><tr><td rowspan=1 colspan=1>CP-SAT</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>17.1</td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1>19.1</td><td rowspan=1 colspan=1>140</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>148</td><td rowspan=1 colspan=1>29.9</td></tr><tr><td rowspan=1 colspan=1>Gurobi</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>15.4</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>14.8</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>18.4</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>16.4</td></tr></table>

Table 8: NRP solving results of diferent approaches with TO = 600s and MO = 5GB.

In terms of solving time, SCL encoding consistently outperforms other SAT-based encodings across all instances. Compared with the best-performing alternative SATbased encoding, namely Seq, SCL solves all instances approximately twice as fast. For example, on the smallest instance (40 days), SCL finds a feasible schedule in 16 ms, whereas Seq requires 27 ms. On the largest instance (100 days), SCL requires only 42 ms, compared to 106 ms for Seq. These results demonstrate a substantial runtime improvement of SCL encoding over existing SAT encodings that encode each cardinality constraint separately.

![](images/dc31962cff58b71f1939a6b106e14d7101a19df2f8b32cd10c9ccc86f5a5e438.jpg)

![](images/92b63588cfe0c8bc618c075ba3252a86221bc393658073b7b140b4f895a2e88e.jpg)  
Fig. 25: NRP solving time of diferent Fig. 26: NRP memory usage of diferent approaches. approaches.

Importantly, this speed advantage is not achieved at the expense of memory usage. SCL encoding also maintains the lowest memory consumption among all SAT-based encodings. Its memory usage increases moderately from 5.7 MB at 40 days to 13.4 MB at 100 days. In comparison, Adder, BDD and Card require roughly twice as much memory, while Seq consumes approximately three times more. Besides, Pairwise encoding fails on all instances due to memory overflow, confirming its lack of scalability when encoding AMK and ALK constraints.

When compared with CP and MIP solvers, SCL encoding remains competitive in solving time while maintaining lower memory consumption, especially on medium and large instances. On the 40-day instance, SCL finds a feasible schedule in 16 ms, slightly faster than 18 ms of Gurobi. The performance gap widens as the problem size increases. On the 100-day instance, SCL requires only 42 ms, whereas Gurobi needs 59 ms, meaning that SCL solves the instance in roughly two-thirds of the time.

It is also worth noting that all the CP and MIP solvers used in this experiment run in parallel using default settings with 8 concurrent threads, whereas the SAT-based approaches operate in single-threaded mode. The fact that SCL achieves better performance despite this diference further highlights its eficiency for problems involving Ladder constraints.

## 8 Conclusion

SAT solving has emerged as a competitive and efective alternative to CP and MIP solvers for various combinatorial, scheduling, and OR problems. Alongside significant advancements in the capabilities of SAT solvers, the development of eficient encodings tailored to specific types of constraints can greatly enhance overall solution performance. Numerous studies have shown that optimizing SAT encoding can lead to substantial improvements, especially when replacing direct encoding techniques with more compact representations. These optimized encodings can reduce the number of auxiliary variables, generated clauses, and clause sizes.

In this paper, we introduced a novel SAT encoding for sets of AMK cardinality constraints arranged in a ladder shape. Rather than encoding each AMK constraint individually, our approach generates encodings for blocks that represent adjacent AMK constraints with the aim of encoding a set of AMK constraints simultaneously. To our knowledge, previous research in this area has mainly focused on using BDD to represent Ladder AMO constraints [14], and no proposed encodings exist for Ladder AMK constraints where k > 1. Compared to the BDD-based representation and other SATbased encodings that encode each AMK constraint separately, the proposed encoding, built upon the Sequential Counter encoding, achieves significant improvements for Ladder AMO constraints and demonstrates efectiveness for Ladder AMK constraints.

We further applied the proposed encoding to address the Anti-bandwidth Problem and Nurse Rostering Problem. The experimental results show that our approach is not only competitive with other SAT-based encodings but also capable of outperforming state-of-the-art CP and MIP solvers such as CPLEX, Gurobi, and CP-SAT. These findings highlight the practical potential of the proposed encoding for real-world applications, such as Car Sequencing problem [1, 2], Employee Scheduling problem [7], and various other scheduling challenges featuring sequence constraints with ladder structures.

Finally, while this work focuses on a block-based SC encoding tailored to Ladder AMK constraints, exploring the suitability of alternative cardinality encodings for enabling intermediate variable reuse remains an interesting direction for future research. In particular, it would be valuable to systematically investigate how other encoding structures can be adapted or extended to better support sequence constraints, especially Ladder AMK constraints. Such an analysis could provide deeper insights into the interaction between encoding structure and constraint overlap, and may further enhance the performance of SAT-based approaches for Ladder AMK constraints, as well as sequence constraints.

Acknowledgements. We sincerely thank the reviewers for their careful reading of the manuscript and constructive comments. Their precise and valuable suggestions have significantly improved the clarity, quality, and technical soundness of this work.

Author Contributions. Hieu Truong Xuan: conceptualization, methodology, formal analysis, investigation, data curation, software, visualization, writing - original draft, review & editing;

Tuyen Kieu Van: methodology, writing - review & editing;

Khanh To Van: conceptualization, investigation, methodology, formal analysis, supervision, funding acquisition, writing - original draft, review & editing.

Availability of data and materials. The experimental results are available at https://github.com/TruongXuanHieu-H/SCL-Encoding.

Code availability. The source code for these experiments is also available at https: //github.com/TruongXuanHieu-H/SCL-Encoding.

## Declaration

Ethics approval. Not applicable.

Consent to participate. Not applicable.

Consent for publication. Not applicable.

Conflict of interest. Not applicable.

## Appendix A Correctness of the SC Encoding

In this appendix, we prove that equations (1)–(7) constitute a sound and complete encoding of the At-Most-K (AMK) constraint over decomposed blocks.

Let a block i of w variables, and define

$$
S _ { i , j } = \sum _ { t = 1 } ^ { j } x _ { i , t } .
$$

The encoded constraint is:

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \leq k .
$$

The register bits $R _ { i , j , s }$ are intended to represent:

$$
R _ { i , j , s } \ \equiv \ ( S _ { i , j } \geq s ) , \quad 1 \leq s \leq \operatorname* { m i n } ( j , k ) .
$$

## A.1 Soundness

Theorem 1. Any assignment satisfying equations (1)–(7) satisfies

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \leq k .
$$

Proof. Assume, for contradiction, that there exists a satisfying assignment such that

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \geq k + 1 .
$$

Let $j$ be the smallest index such that

$$
S _ { i , j } = k + 1 .
$$

Then necessarily

$$
S _ { i , j - 1 } = k \quad \mathrm { a n d } \quad x _ { i , j } = 1 .
$$

Since $S _ { i , j - 1 } = k , R _ { i , j - 1 , k } = 1 \mathrm { ~ ( o r ~ } t r u e )$

However, equation (7) enforces, for all $j > k ,$

$$
x _ { i , j }  \lnot R _ { i , j - 1 , k } .
$$

Since $x _ { i , j } = 1$ and $R _ { i , j - 1 , k } = 1$ , we obtain a contradiction. Therefore,

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \leq k .
$$

## A.2 Completeness

Theorem 2. If

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \leq k ,
$$

then there exists an assignment to the register bits $R _ { i , j , }$ <sub>s</sub> satisfying equations (1)–(7).

Proof. Assume

$$
\sum _ { j = 1 } ^ { w } x _ { i , j } \leq k .
$$

Define the register bits as follows:

$$
R _ { i , j , s } : = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } S _ { i , j } \geq s , } \\ { 0 } & { { \mathrm { o t h e r w i s e . } } } \end{array} } \right.
$$

By construction, the relation $S _ { i , j } = S _ { i , j - 1 } + x _ { i , j }$ ensures that equations (1)–(3) are satisfied.

Similarly, if $S _ { i , j } < s ,$ then either $S _ { i , j - 1 } < s - 1$ or $( S _ { i , j - 1 } < s$ and $x _ { i , j } = 0 )$ , which guarantees that equations $( 4 )  { - } ( 6 )$ are satisfied.

Finally, since $S _ { i , j } ~ \le ~ k$ for all $j ,$ we never have both $x _ { i , j } ~ = ~ 1$ and $S _ { i , j - 1 } = k$ simultaneously. Hence, equation (7) is satisfied.

Therefore, a satisfying assignment of the register bits exists.

Together, Theorems 1 and 2 establish that the SC encoding is sound and complete for encoding AMK constraints.

## References

[1] Artigues, C., Hebrard, E., Mayer-Eichberger, V., Siala, M., Walsh, T.: Sat and hybrid models of the car sequencing problem. In: Integration of AI and OR Techniques in Constraint Programming: 11th International Conference, CPAIOR 2014, Cork, Ireland, May 19-23, 2014. Proceedings 11, pp. 268–283 (2014). Springer

[2] Siala, M.: Search, propagation, and learning in sequencing and scheduling problems. PhD thesis, INSA de Toulouse (2015)

[3] Ceschia, S., Dang, N.T.T., De Causmaecker, P., Haspeslagh, S., Schaerf, A.: Second international nurse rostering competition (inrc-ii)—problem description and rules—. arXiv preprint arXiv:1501.04177 (2015)

[4] Kletzander, L., Musliu, N.: Solving the general employee scheduling problem. Computers & Operations Research 113, 104794 (2020)

[5] Bergman, D., Cire, A.A., Hoeve, W.: Mdd propagation for sequence constraints. Journal of Artificial Intelligence Research 50, 697–722 (2014)

[6] Siala, M., Hebrard, E., Huguet, M.-J.: An optimal arc consistency algorithm for a particular case of sequence constraint. Constraints 19, 30–56 (2014)

[7] Nieuwenhuis, R., Oliveras, A., Rodr´ıguez-Carbonell, E., Rollon, E.: Employee scheduling with sat-based pseudo-boolean constraint solving. IEEE access 9, 142095–142104 (2021)

[8] Lester, M.M.: Pseudo-boolean optimisation for robinx sports timetabling. Journal of Scheduling 25(3), 287–299 (2022)

[9] Bessiere, C., Hebrard, E., Hnich, B., Kiziltan, Z., Quimper, C.-G., Walsh, T.: Reformulating global constraints: The slide and regular constraints. In: Abstraction, Reformulation, and Approximation: 7th International Symposium, SARA 2007, Whistler, Canada, July 18-21, 2007. Proceedings 7, pp. 80–92 (2007). Springer

[10] Sinnl, M.: A note on computational approaches for the antibandwidth problem. Central European Journal of Operations Research 29(3), 1057–1077 (2021)

[11] Vardi, M.: The sat revolution: Solving, sampling, and counting. In: Mathematical Colloquium (2015)

[12] Haberlandt, A., Green, H., Heule, M.J.: Efective auxiliary variables via structured reencoding. arXiv preprint arXiv:2307.01904 (2023)

[13] Vasconcellos-Gaete, C., Barichard, V., Lardeux, F.: Abacus: A new hybrid encod ing for sat problems. In: 2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI), pp. 145–152 (2020). IEEE

[14] Fazekas, K., Sinnl, M., Biere, A., Parragh, S.: Duplex encoding of staircase atmost-one constraints for the antibandwidth problem. In: International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 186–204 (2020). Springer

[15] Sinz, C.: Towards an optimal cnf encoding of boolean cardinality constraints. In: International Conference on Principles and Practice of Constraint Programming, pp. 827–831 (2005). Springer

[16] Hieu, T.X., Tuyen, K.V., Khanh, T.V.: Sequential counter encoding for staircase at-most-one constraints. In: Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART, pp. 164–175 (2025). https://doi.org/10.5220/0013124600003890 . SciTePress

[17] E´en, N., S¨orensson, N.: Translating pseudo-boolean constraints into sat. Journal on Satisfiability, Boolean Modelling and Computation 2(1-4), 1–26 (2006)

[18] Ab´ıo, I., Nieuwenhuis, R., Oliveras, A., Rodr´ıguez-Carbonell, E., Mayer-Eichberger, V.: A new look at bdds for pseudo-boolean constraints. Journal of Artificial Intelligence Research 45, 443–480 (2012)

[19] As´ın, R., Nieuwenhuis, R., Oliveras, A., Rodr´ıguez-Carbonell, E.: Cardinality networks and their applications. In: Theory and Applications of Satisfiability Testing-SAT 2009: 12th International Conference, SAT 2009, Swansea, UK, June 30-July 3, 2009. Proceedings 12, pp. 167–180 (2009). Springer

[20] As´ın, R., Nieuwenhuis, R., Oliveras, A., Rodr´ıguez-Carbonell, E.: Cardinality networks: a theoretical and empirical study. Constraints 16, 195–221 (2011)

[21] Batcher, K.E.: Sorting networks and their applications. In: Proceedings of the April 30–May 2, 1968, Spring Joint Computer Conference, pp. 307–314 (1968)

[22] Bryant, R.E.: Graph-based algorithms for boolean function manipulation. Com puters, IEEE Transactions on 100(8), 677–691 (1986)

[23] Akers: Binary decision diagrams. IEEE Transactions on computers 100(6), 509– 516 (1978)

[24] Chen, J.: A new sat encoding of the at-most-one constraint. Proc. constraint modelling and reformulation, 8 (2010)

[25] Leung, J.Y., Vornberger, O., Witthof, J.D.: On some variants of the bandwidth minimization problem. SIAM Journal on Computing 13(3), 650–667 (1984)

[26] Hale, W.K.: Frequency assignment: Theory and applications. Proceedings of the IEEE 68(12), 1497–1514 (1980)

[27] Cappanera, P.: A survey on obnoxious facility location problems. University of Pisa (1999)

[28] Hu, Y., Gansner, E.R., Kobourov, S.: Visualizing graphs and clusters as maps. IEEE Computer Graphics and Applications 30(6), 54–66 (2010)

[29] Gent, I.P., Petrie, K.E., Puget, J.-F.: Symmetry in constraint programming. Foundations of Artificial Intelligence 2, 329–376 (2006)

[30] Biere, A.: Cadical at the sat race 2019. In: Heule, M., J¨arvisalo, M., Suda, M.

(eds.) Proceedings of SAT Race 2019: Solver and Benchmark Descriptions, vol. B-2019-1 of Department of Computer Science Series of Publications B, University of Helsinki 2019, pp. 8–9 (2019)

[31] Rodriguez-Tello, E., Romero-Monsivais, H., Ram´ırez-Torres, J., Lardeux, F.: Harwell-Boeing Graphs for the CB Problem. https://doi.org/10.13140/2.1.2408. 0166

[32] Nethercote, N., Stuckey, P.J., Becket, R., Brand, S., Duck, G.J., Tack, G.: Minizinc: Towards a standard cp modelling language. In: International Conference on Principles and Practice of Constraint Programming, pp. 529–543 (2007). Springer