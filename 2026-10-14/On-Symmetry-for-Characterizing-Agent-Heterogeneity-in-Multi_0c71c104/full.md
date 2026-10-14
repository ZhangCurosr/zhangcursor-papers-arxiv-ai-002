# On Symmetry for Characterizing Agent Heterogeneity in Multi-Agent Systems

Jianhon<sub>g</sub> Wan<sub>g</sub><sup>†‡</sup>, Jonathan Lawr<sub>y</sub><sup>‡</sup>

<sup>†</sup>INFORMED-AI H<sub>u</sub>b<sub>,</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Bri<sub>s</sub>t<sub>o</sub>l

<sup>‡</sup>School of En<sub>g</sub>ineerin<sub>g</sub> Mathematics and Technolo<sub>gy,</sub> Universit<sub>y</sub> of Bristol

Bri<sub>s</sub>t<sub>o</sub>l<sub>,</sub> Unit<sub>e</sub>d Kin<sub>g</sub>d<sub>o</sub>m

{jianhong.wang, J.Lawry}@bristol.ac.uk

## Abstract

Understandin<sub>g</sub> a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y</sub> can inform the desi<sub>g</sub>n of shared and a<sub>g</sub>ent-s<sub>p</sub>ecific <sub>p</sub>olic<sub>y</sub> re<sub>p</sub>resentations<sub>,</sub> however existin<sub>g</sub> criteria <sub>p</sub>ro<sub>v</sub>ide no <sub>u</sub>nified str<sub>u</sub>ct<sub>u</sub>ral test. We st<sub>u</sub>d<sub>y</sub> finite-horizon POSGs <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> <sub>a</sub> t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>as</sub> <sub>a</sub> <sub>po</sub>li<sub>cy-eva</sub>l<sub>ua</sub>ti<sub>on</sub> s<sub>y</sub>mmetr<sub>y</sub> when com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation commutes with it for ever<sub>y</sub> decentralized behavioral <sub>p</sub>olic<sub>y</sub>. Polic<sub>y</sub>-evaluation s<sub>y</sub>mmet<sub>r</sub>i<sub>es</sub> f<sub>orm a group</sub> f<sub>or w</sub>hi<sub>c</sub>h <sub>agen</sub>t <sub>or</sub>bit<sub>s</sub> d<sub>e</sub>fi<sub>ne s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>yp</sub>es. This classification does not de<sub>p</sub>end on the <sub>p</sub>articular <sub>p</sub>olic<sub>y</sub> chosen or the distribution of initial states. We <sub>p</sub>rove that ever<sub>y</sub> com<sub>p</sub>onentwise environment automor<sub>p</sub>hism is a <sub>p</sub>olic<sub>y</sub>-evaluation <sub>symme</sub>t<sub>ry,</sub> b<sub>u</sub>t th<sub>a</sub>t th<sub>e converse</sub> d<sub>oes no</sub>t h<sub>o</sub>ld<sub>.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>rame-</sub> work to coo<sub>p</sub>erative Dec-POMDPs shows that a<sub>g</sub>ents can belon<sub>g</sub> t<sub>o</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l t<sub>ypes even w</sub>h<sub>en</sub> th<sub>ey s</sub>h<sub>are a common re-</sub> ward. For <sub>p</sub>rescribed wei<sub>g</sub>hted reward a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> we derive an <sub>exac</sub>t <sub>va</sub>l<sub>ue</sub> id<sub>en</sub>tit<sub>y an</sub>d <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>ons</sub> f<sub>or</sub> th<sub>e resu</sub>lti<sub>ng</sub> D<sub>ec-</sub> POMDP to inherit the <sub>p</sub>olic<sub>y</sub>-e<sub>v</sub>al<sub>u</sub>ation s<sub>y</sub>mmetries of the so<sub>u</sub>rce POSG<sub>.</sub> Fin<sub>a</sub>ll <sub>, we g</sub>i<sub>ve su</sub>fi<sub>c</sub>i<sub>e</sub>nt <sub>co</sub>nditi<sub>o</sub>n<sub>s u</sub>nd<sub>e</sub>r <sub>w</sub>hi<sub>c</sub>h r<sub>e ea</sub>t<sub>e</sub>d model factorizations induce <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y,</sub> and show that <sub>p</sub>reservin<sub>g</sub> an o<sub>p</sub>timal Boltzmann <sub>p</sub>olic<sub>y</sub> set neither im<sub>p</sub>lies <sub>nor</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> thi<sub>s</sub> <sub>symme</sub>t<sub>ry</sub> i<sub>n</sub> <sub>genera</sub>l<sub>.</sub>

## Keywords

multi-a<sub>g</sub>ent s<sub>y</sub>stems<sub>,</sub> a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y,</sub> s<sub>y</sub>mmetr<sub>y,</sub> POSGs<sub>,</sub> Dec-POMDP<sub>s</sub>

## 1 Introduction

M<sub>u</sub>lti-a<sub>g</sub>ent s<sub>y</sub>stems can describe interactin<sub>g</sub> decision-makers in real-world <sub>p</sub>roblems, such as coo<sub>p</sub>erative robot control [5] and volt a<sub>g</sub>e re<sub>g</sub>ulation in <sub>p</sub>ower distribution networks [33], where a<sub>g</sub>ents’ capa<sup>b</sup>i<sup>l</sup>ities, in<sup>f</sup>ormation, o<sup>b</sup>jectives, an<sup>d b</sup>e<sup>h</sup>avior may <sup>d</sup>i<sup>f</sup>er. As <sub>a resu</sub>lt<sub>, c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>ese</sub> dif<sub>erences</sub> i<sub>s necessary, w</sub>hi<sub>c</sub>h <sub>can</sub> inform the desi<sub>g</sub>n of <sub>p</sub>olic<sub>y</sub> to make decisions.

S<sub>y</sub>mmetr<sub>y</sub> identifies transformations that <sub>p</sub>reserve relevant s<sub>y</sub>stem <sub>p</sub>ro<sub>p</sub>erties<sub>,</sub> a <sub>p</sub>rinci<sub>p</sub>le used across <sub>p</sub>h<sub>y</sub>sical and com<sub>p</sub>utational sciences. In <sub>p</sub>ower-<sub>g</sub>rid swin<sub>g</sub> e<sub>q</sub>uations<sub>,</sub> exchan<sub>g</sub>in<sub>g</sub> <sub>g</sub>enerator l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>preserves</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>cs</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>genera</sub>t<sub>or</sub> <sub>p</sub>arameters and network cou<sub>p</sub>lin<sub>g</sub>s are unchan<sub>g</sub>ed [21]. In molecular modelin<sub>g,</sub> ener<sub>gy</sub> is invariant to ri<sub>g</sub>id rotations<sub>,</sub> translations<sub>,</sub> and atom reindexin<sub>g</sub>, <sub>p</sub>ro<sub>p</sub>erties incor<sub>p</sub>orated into SchNet [26]. These exam<sub>p</sub>les ins<sub>p</sub>ires a <sub>p</sub>ossibilit<sub>y</sub> to identif<sub>y</sub> corres<sub>p</sub>ondin<sub>g</sub> structural re<sub>g</sub>ularities in multi-a<sub>g</sub>ent s<sub>y</sub>stems b<sub>y</sub> s<sub>y</sub>mmetries.

Multi-a<sub>g</sub>ent reinforcement learnin<sub>g</sub> (MARL), as a learnin<sub>g</sub> framework for multi-a<sub>g</sub>ent s<sub>y</sub>stems<sub>,</sub> alread<sub>y</sub> ex<sub>p</sub>loits s<sub>y</sub>mmetr<sub>y</sub>. Dist<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> th<sub>a</sub>t <sub>enco</sub>d<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>symme</sub>t<sub>r</sub>i<sub>es</sub> <sub>emp</sub>i<sub>r-</sub> icall<sub>y</sub> im<sub>p</sub>rove data eficienc<sub>y</sub> on s<sub>y</sub>mmetric coo<sub>p</sub>erative tasks [31]. Other-<sub>p</sub>la<sub>y</sub> uses environment s<sub>y</sub>mmetries to reduce de<sub>p</sub>endence on arbitrar<sub>y</sub> conventions in zero-shot coordination [14], while ex-<sub>pec</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn symme</sub>t<sub>r</sub>i<sub>es ex</sub>t<sub>en</sub>d th<sub>e symme</sub>t<sub>ry c</sub>l<sub>ass use</sub>d f<sub>or</sub> thi<sub>s</sub> <sub>p</sub>ur<sub>p</sub>ose [22]. However, their focus is task-s<sub>p</sub>ecific (im<sub>p</sub>rovin<sub>g</sub> learnin<sub>g</sub> and coordination), leavin<sub>g</sub> the connection to structural a<sub>g</sub>ent <sup>hetero</sup>g<sup>eneit</sup>y <sup>an o</sup>p<sup>en</sup> q<sup>uestion.</sup>

B<sub>roa</sub>d<sub>er</sub> f<sub>orma</sub>l <sub>accoun</sub>t<sub>s</sub> d<sub>escr</sub>ib<sub>e</sub> <sub>componen</sub>t<sub>-preserv</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l automor<sub>p</sub>hisms [17, 30]. Braun et al. [7] use countin<sub>g</sub> and isom<sub>o</sub>r<sub>p</sub>hi<sub>c</sub> <sub>sy</sub>mm<sub>e</sub>tri<sub>es</sub> f<sub>o</sub>r <sub>co</sub>m<sub>pac</sub>t <sub>pa</sub>rtiti<sub>o</sub>n<sub>e</sub>d D<sub>ec</sub>-POMDP<sub>s,</sub> <sub>w</sub>ith a<sub>g</sub>ent-number tractabilit<sub>y</sub> rel<sub>y</sub>in<sub>g</sub> on the stron<sub>g</sub>er isomor<sub>p</sub>hic fact<sub>or</sub>i<sub>za</sub>ti<sub>on.</sub> C<sub>omp</sub>l<sub>emen</sub>t<sub>ary wor</sub>k <sub>c</sub>l<sub>ass</sub>ifi<sub>es p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>an</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l hetero<sub>g</sub>eneit<sub>y</sub> in multi-robot learnin<sub>g</sub> [5], measures behavioral differences throu<sub>g</sub>h s<sub>y</sub>stem neural diversit<sub>y</sub> [6], and distin<sub>g</sub>uishes observation, transition, objective, and <sub>p</sub>olic<sub>y</sub> hetero<sub>g</sub>eneit<sub>y</sub> [15]. These <sub>p</sub>ers<sub>p</sub>ectives identif<sub>y p</sub>articular model re<sub>g</sub>ularities or dimensions <sub>o</sub>f dif<sub>erence,</sub> h<sub>owever,</sub> th<sub>ey are s</sub>till <sub>w</sub>ith <sub>a gap</sub> t<sub>o</sub> th<sub>e c</sub>h<sub>rac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> of multi-a<sub>g</sub>ent s<sub>y</sub>stems.

We therefore ask: can we use symmetries to determine structural agent heterogeneity? Specifically, when can agents be exchanged throu<sub>g</sub>h com<sub>p</sub>atible transformations without chan<sub>g</sub>in<sub>g</sub> how ever<sub>y</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> i<sub>s</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d?

T<sub>o</sub> <sub>answer</sub> th<sub>e</sub> <sub>a</sub>b<sub>ove</sub> <sub>researc</sub>h <sub>ques</sub>ti<sub>ons,</sub> <sub>we</sub> <sub>groun</sub>d <sub>our</sub> th<sub>eo-</sub> <sub>re</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k i<sub>n</sub> fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon par</sub>ti<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> games (POSGs). A transformation is a policy-evaluation symmetry if evaluatin<sub>g</sub> an<sub>y</sub> transformed <sub>p</sub>olic<sub>y</sub> <sub>g</sub>ives the corres<sub>p</sub>ondin<sub>g</sub>l<sub>y</sub> trans<sup>f</sup>orme<sup>d</sup> va<sup>l</sup>ue pro<sup>fil</sup>e, at every state–joint-<sup>h</sup>istory pair. T<sup>h</sup>ese symmetries form a group whose agent orbits define structural agent types: one orbit means structural homogeneity, and multiple orbits mean structural hetero<sub>g</sub>eneit<sub>y</sub>. For the s<sub>p</sub>ecified <sub>g</sub>ame and admissibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons,</sub> thi<sub>s c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>fth<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>p</sub>olic<sub>y</sub> and initial-state distrib<sub>u</sub>tion.

Our contributions are as follows: (1) We establish the <sub>p</sub>olic<sub>y</sub>- eva<sup>l</sup>uat<sup>i</sup>on s<sub>y</sub>mmetr<sub>y g</sub>rou<sub>p</sub> an<sup>d i</sup>ts structura<sup>l</sup> a<sub>g</sub>ent t<sub>yp</sub>es, a<sup>ll</sup>ow<sup>i</sup>n<sub>g</sub> com<sub>p</sub>atible a<sub>g</sub>ent<sub>,</sub> state<sub>,</sub> action<sub>,</sub> and observation transformations. We distin<sub>g</sub>uish <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y,</sub> which concerns ever<sub>y</sub> decentralized <sub>p</sub>olic<sub>y</sub>, from the e<sub>q</sub>uivariance of a selected <sub>p</sub>olic<sub>y</sub>. (2) We <sub>p</sub>rove that <sub>p</sub>reservin<sub>g</sub> the transition<sub>,</sub> observation<sub>,</sub> and reward functions com<sub>p</sub>onentwise <sub>g</sub>uarantees <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y,</sub> b<sub>u</sub>t the re<sub>v</sub>erse is not necessar<sub>y</sub>. A finite t<sub>w</sub>o-sta<sub>g</sub>e POSG <sub>y</sub>ields a strictl<sub>y</sub> finer com<sub>p</sub>onentwise a<sub>g</sub>ent <sub>p</sub>artition to show the necessit<sub>y</sub> of the <sub>p</sub>olic<sub>y</sub>-evaluation definition. (3) We show that Dec-POMDPs <sub>can rema</sub>i<sub>n s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y</sub> h<sub>e</sub>t<sub>erogeneous, a</sub>lth<sub>oug</sub>h <sub>w</sub>ith <sub>a common</sub> r<sub>ewa</sub>rd<sub>.</sub> F<sub>o</sub>r <sub>a</sub> d<sub>e</sub>ri<sub>ve</sub>d D<sub>ec</sub>-POMDP <sub>w</sub>ith <sub>a p</sub>r<sub>esc</sub>rib<sub>e</sub>d <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>common rewar</sub>d<sub>, we</sub> d<sub>er</sub>i<sub>ve an exac</sub>t <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d<sub>-va</sub>l<sub>ue</sub> id<sub>en</sub>tit<sub>y an</sub>d suficient conditions for inheritin<sub>g</sub> the source <sub>g</sub>ame’s s<sub>y</sub>mmetries. We further com<sub>p</sub>are the framework with existin<sub>g</sub> hetero<sub>g</sub>eneit<sub>y</sub> cate<sub>g</sub>ories<sub>,</sub> countin<sub>g</sub> and isomor<sub>p</sub>hic s<sub>y</sub>mmetr<sub>y,</sub> and ex<sub>p</sub>ected re turn s<sub>y</sub>mmetr<sub>y,</sub> makin<sub>g</sub> the assum<sub>p</sub>tions and limitations of each com<sub>p</sub>arison ex<sub>p</sub>licit.

## 2 Related Work

## 2.1 Taxonomies of Agent Heterogeneity

Bettini et al. [5] distin<sub>g</sub>uish <sub>p</sub>h<sub>y</sub>sical from behavioral hetero<sub>g</sub>eneit<sub>y</sub> in multi-robot and multi-a<sub>g</sub>ent s<sub>y</sub>stems. The<sub>y</sub> further se<sub>p</sub>arate b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y accor</sub>di<sub>ng</sub> t<sub>o w</sub>h<sub>e</sub>th<sub>er agen</sub>t<sub>s s</sub>h<sub>are an o</sub>b<sub>-</sub> jective, yie<sup>ld</sup>ing <sup>fi</sup>ve p<sup>h</sup>ysica<sup>l</sup>–<sup>b</sup>e<sup>h</sup>aviora<sup>l</sup> su<sup>b</sup>c<sup>l</sup>asses. T<sup>h</sup>is taxonomy i<sub>s a</sub> hi<sub>g</sub>h<sub>-</sub>l<sub>eve</sub>l <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on ra</sub>th<sub>er</sub> th<sub>an a measure o</sub>f d<sub>egree.</sub> Th<sub>e</sub>i<sub>r</sub> l<sub>a</sub>t<sub>er</sub> <sub>sys</sub>t<sub>em</sub> <sub>neura</sub>l di<sub>vers</sub>it<sub>y</sub> <sub>measure</sub> <sub>a</sub>dd<sub>resses</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er</sub> <sub>ques</sub>ti<sub>on</sub> for behavior b<sub>y</sub> avera<sub>g</sub>in<sub>g p</sub>airwise behavioral distances [6]. Hu et al. [15] <sub>g</sub>ive a finer a<sub>g</sub>ent-level MARL taxonom<sub>y</sub> based on observation, response-transition, e<sup>f</sup>ect-transition, o<sup>b</sup>jective, an<sup>d</sup> po<sup>l</sup>icy <sup>h</sup>etero<sub>g</sub>ene<sup>i</sup>t<sub>y</sub>.

Table 1 com<sub>p</sub>ares these <sub>p</sub>rior cate<sub>g</sub>ories and the s<sub>y</sub>stem neural diversit<sub>y</sub> measure with the closest conce<sub>p</sub>ts in our framework. For p<sup>h</sup>ysica<sup>l</sup>, o<sup>b</sup>servation, transition, an<sup>d</sup> o<sup>b</sup>jective <sup>h</sup>eterogeneity, t<sup>h</sup>e associations are conditional on the relevant a<sub>g</sub>ent <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> bein<sub>g</sub> re<sub>p</sub>resented in the POSG. Beha<sub>v</sub>ioral and <sub>p</sub>olic<sub>y</sub> hetero<sub>g</sub>eneit<sub>y</sub> are i<sub>ns</sub>t<sub>ea</sub>d <sub>compare</sub>d th<sub>roug</sub>h <sub>equ</sub>i<sub>var</sub>i<sub>ance</sub> <sub>o</sub>f <sub>a</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy</sub> <sub>re</sub>l<sub>-</sub> ative to chosen <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries. Bettini et al.’s five <sub>su</sub>b<sub>c</sub>l<sub>asses com</sub>bi<sub>ne p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>an</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y, w</sub>ith b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> f<sub>ur</sub>th<sub>er</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>e</sub>d b<sub>y</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>agen</sub>t<sub>s</sub> s<sup>h</sup>are an o<sup>b</sup>jective. T<sup>h</sup>e ta<sup>bl</sup>e t<sup>h</sup>ere<sup>f</sup>ore <sup>l</sup>ists t<sup>h</sup>ese un<sup>d</sup>er<sup>l</sup>ying <sup>d</sup>isti<sub>nc</sub>ti<sub>ons</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>repea</sub>ti<sub>ng</sub> <sub>a</sub>ll fi<sub>ve</sub> <sub>com</sub>bi<sub>na</sub>ti<sub>ons.</sub>

These associations locate earlier cate<sub>g</sub>ories within the s<sub>y</sub>mmetr<sub>y</sub> f<sub>ramewor</sub>k<sub>,</sub> b<sub>u</sub>t th<sub>ey</sub> d<sub>o no</sub>t id<sub>en</sub>tif<sub>y</sub> th<sub>ose ca</sub>t<sub>egor</sub>i<sub>es w</sub>ith <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l a<sub>g</sub>ent t<sub>yp</sub>es. A com<sub>p</sub>onentwise environment automor<sub>p</sub>hism that <sub>exc</sub>h<sub>anges</sub> t<sub>wo agen</sub>t<sub>s guaran</sub>t<sub>ees</sub> th<sub>a</sub>t th<sub>ey</sub> h<sub>ave</sub> th<sub>e same s</sub>t<sub>ruc-</sub> tural a<sub>g</sub>ent t<sub>yp</sub>e. However<sub>,</sub> failure of a com<sub>p</sub>onentwise <sub>p</sub>reservation <sub>equa</sub>ti<sub>on</sub> f<sub>or a can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on ne</sub>ith<sub>er ru</sub>l<sub>es ou</sub>t th<sub>a</sub>t t<sub>rans-</sub> f<sub>orma</sub>ti<sub>on as a po</sub>li<sub>cy-eva</sub>l<sub>ua</sub>ti<sub>on symme</sub>t<sub>ry nor es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t th<sub>e</sub> a<sub>g</sub>ents have diferent structural a<sub>g</sub>ent t<sub>yp</sub>es. Likewise<sub>,</sub> diferent b<sub>e</sub>h<sub>av</sub>i<sub>or un</sub>d<sub>er one se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy</sub> d<sub>oes no</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h dif<sub>-</sub> f<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes.</sub> G<sub>ra</sub>d<sub>e</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>measures suc</sub>h <sub>as</sub> s<sub>y</sub>stem neural diversit<sub>y</sub> [6] are com<sub>p</sub>lementar<sub>y</sub> to the structural a<sub>g</sub>ent-t<sub>yp</sub>e <sub>p</sub>artition. A<sub>pp</sub>endix A ex<sub>p</sub>lains the assum<sub>p</sub>tions and di<sub>rec</sub>ti<sub>on o</sub>f <sub>eac</sub>h <sub>assoc</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub>

## 2.2 Symmetry in Multi-Agent Systems

S<sub>y</sub>mmetr<sub>y</sub> in normal-form <sub>g</sub>ames is commonl<sub>y</sub> ex<sub>p</sub>ressed throu<sub>g</sub>h <sub>p</sub>ermutations of <sub>p</sub>la<sub>y</sub>ers and strate<sub>g</sub>ies that <sub>p</sub>reserve <sub>p</sub>a<sub>y</sub>of <sub>p</sub>rofiles. Cao and Yan<sub>g</sub> [8] formalize <sub>p</sub>la<sub>y</sub>er and strate<sub>gy</sub> s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub>s and distin<sub>g</sub>uish ordinar<sub>y,</sub> renamin<sub>g,</sub> and name-irrelevant s<sub>y</sub>mmetric <sub>g</sub>ames. Koriche et al. [19] distin<sub>g</sub>uish structural s<sub>y</sub>mmetr<sub>y</sub> from minimax s<sub>y</sub>mmetr<sub>y</sub> in finite-horizon stochastic <sub>g</sub>ames. Their semantic notion <sub>p</sub>reserves an o<sub>p</sub>timal minimax value<sub>,</sub> whereas ours evaluates <sub>every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> i<sub>n</sub> <sub>a</sub> <sub>par</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>game.</sub> T<sub>ewo</sub>ld<sub>e</sub> et al. [29] stud<sub>y</sub> the com<sub>p</sub>lexit<sub>y</sub> of detectin<sub>g</sub> <sub>p</sub>la<sub>y</sub>er and action s<sub>y</sub>mmetries in normal-form <sub>g</sub>ames<sub>,</sub> derivin<sub>g</sub> <sub>g</sub>ra<sub>p</sub>h-automor<sub>p</sub>hism and <sub>g</sub>ra<sub>p</sub>h-isomor<sub>p</sub>hism com<sub>p</sub>leteness results<sub>,</sub> and anal<sub>y</sub>ze Nash e<sub>q</sub>uilibria that res<sub>p</sub>ect <sub>g</sub>i<sub>v</sub>en s<sub>y</sub>mmetries. These traditions moti<sub>v</sub>ate treatin<sub>g</sub> a<sub>g</sub>ent interchan<sub>g</sub>eabilit<sub>y</sub> as an orbit <sub>q</sub>uestion but do not directl<sub>y</sub> address our all-<sub>p</sub>olic<sub>y</sub> settin<sub>g</sub> with t<sub>yp</sub>e-com<sub>p</sub>atible state<sub>,</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub> <sub>an</sub>d hi<sub>s</sub>t<sub>ory</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub>

Table 1: Relation of prior heterogeneity categories and measures to our symmetry-based concepts. Note that these associations are not claims of formal equivalence.
<table><tr><td></td><td>Prior category or measure Relation to our framework</td></tr><tr><td>Physical heterogeneity [5] Physical differences can be tested through a</td><td>componentwise environment automorphism when their consequences are encoded in the POSG</td></tr><tr><td>Observation heterogeneity [15]</td><td>Observation-preservation condition within a componentwise environment automorphism.</td></tr><tr><td>Response- and effect-transition heterogeneity [15]</td><td>Transition-preservation condition within a componentwise environment automorphism, provided that the required local transition</td></tr><tr><td>Objective</td><td>factorization is specified. Reward-preservation condition within a</td></tr><tr><td>heterogeneity [5, 15]</td><td>componentwise environment automorphism.</td></tr><tr><td>Behavioral or policy heterogeneity [5, 15]</td><td>Equivariance of a selected policy characterizes behavioral homogeneity among agents related by the chosen policy-evaluation symmetries, while failure</td></tr><tr><td></td><td>relative to a specified symmetry expresses policy-relative behavioral differentiation.</td></tr><tr><td>System neural diversity [6]</td><td>Graded comparison of selected policies that complements the exact classification into structural agent types.</td></tr></table>

For se<sub>q</sub>uential models, Kan<sub>g</sub> and Kim [17] stud<sub>y</sub> s<sub>y</sub>mmetries of POMDPs and POSGs thro<sub>ug</sub>h <sub>g</sub>ra<sub>p</sub>h a<sub>u</sub>tomor<sub>p</sub>hisms that <sub>p</sub>reserve the model com<sub>p</sub>onents. In coo<sub>p</sub>erative settin<sub>g</sub>s, Hu et al. [14] call component-preserving transformations equivalence mappings and <sub>u</sub>se them to randomize con<sub>v</sub>entions in other-<sub>p</sub>la<sub>y</sub>. Tre<sub>u</sub>tlein et al. [30] formulate Dec-POMDP isomor<sub>p</sub>hisms and automor<sub>p</sub>hisms using <sup>b</sup>ijections o<sup>f</sup> agents, states, actions, an<sup>d</sup> o<sup>b</sup>servations t<sup>h</sup>at pre-<sub>se</sub>r<sub>ve</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>s, o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>s,</sub> r<sub>ewa</sub>rd<sub>s, a</sub>nd th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> Th<sub>ese</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons use</sub> dif<sub>eren</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>conven</sub>ti<sub>ons an</sub>d <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>i<sub>-</sub> bl<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> Aft<sub>e</sub>r tr<sub>a</sub>n<sub>s</sub>l<sub>a</sub>tin<sub>g</sub> th<sub>e</sub>m int<sub>o ou</sub>r <sub>se</sub>ttin<sub>g,</sub> th<sub>e</sub>ir re<sub>q</sub>uirements to <sub>p</sub>reserve the transition<sub>,</sub> observation<sub>,</sub> and reward functions become instances of the com<sub>p</sub>onentwise environment a<sub>u</sub>tomor<sub>p</sub>hisms st<sub>u</sub>died in Section 5. B<sub>y</sub> Theorem 5.2<sub>,</sub> these conditions are suficient<sub>,</sub> but not necessar<sub>y,</sub> for <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>. We instead define <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> directl<sub>y</sub> b<sub>y</sub> $\mathcal { E } _ { M } ( g \cdot \pi ) = g \cdot \mathcal { E } _ { M } ( \pi )$ f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy �:</sub> t<sub>rans</sub>f<sub>orm-</sub> i<sub>ng a po</sub>li<sub>cy</sub> b<sub>e</sub>f<sub>ore eva</sub>l<sub>ua</sub>ti<sub>on mus</sub>t <sub>g</sub>i<sub>ve</sub> th<sub>e same va</sub>l<sub>ue pro</sub>fil<sub>e as</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> it fi<sub>rs</sub>t <sub>an</sub>d th<sub>en</sub> t<sub>rans</sub>f<sub>orm</sub>i<sub>ng</sub> it<sub>s va</sub>l<sub>ues.</sub>

Other work ca<sub>p</sub>tures diferent conse<sub>q</sub>uences or re<sub>p</sub>resentations of s<sub>y</sub>mmetr<sub>y</sub>. At the <sub>p</sub>olic<sub>y</sub>-architecture level, van der Pol et al. [31] f<sub>ac</sub>t<sub>or</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>coopera</sub>ti<sub>ve-sys</sub>t<sub>em</sub> <sub>symme</sub>t<sub>r</sub>i<sub>es</sub> i<sub>n</sub>t<sub>o</sub> l<sub>oca</sub>l t<sub>rans</sub>f<sub>or-</sub> mations and a<sub>g</sub>ent <sub>p</sub>ermutations to construct distributed e<sub>q</sub>uivariant <sub>p</sub>olic<sub>y</sub> networks. This ex<sub>p</sub>loits a su<sub>pp</sub>lied s<sub>y</sub>mmetr<sub>y</sub> in <sub>p</sub>olic<sub>y</sub> l<sub>earn</sub>i<sub>ng,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>our</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> <sub>are</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>e</sub>f<sub>ore</sub> <sub>se-</sub> lectin<sub>g</sub> a <sub>p</sub>olic<sub>y</sub> or architecture. At the re<sub>p</sub>resentation level<sub>,</sub> Braun et al. [7] use countin<sub>g</sub> and isomor<sub>p</sub>hic s<sub>y</sub>mmetr<sub>y</sub> to obtain com-<sub>p</sub>act lifted Dec-POMDP re<sub>p</sub>resentations. Mu<sub>g</sub>lich et al. [22] introduce ex<sub>p</sub>ected return s<sub>y</sub>mmetries<sub>,</sub> which <sub>p</sub>reserve self-<sub>p</sub>la<sub>y</sub>-o<sub>p</sub>timal B<sub>o</sub>lt<sub>zmann-exp</sub>l<sub>ora</sub>t<sub>ory po</sub>li<sub>c</sub>i<sub>es a</sub>t <sub>a</sub> fi<sub>xe</sub>d t<sub>empera</sub>t<sub>ure</sub> f<sub>or zero-s</sub>h<sub>o</sub>t coordination. Our structural a<sub>g</sub>ent t<sub>yp</sub>es instead use the orbits of a solution-inde<sub>p</sub>endent <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub> that holds f<sub>or</sub> <sub>every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy.</sub>

3 Model and Type-Compatible Transformations W<sub>e cons</sub>id<sub>er a</sub> fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon par</sub>ti<sub>a</sub>ll<sub>y o</sub>b<sub>serva</sub>bl<sub>e s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c game</sub> (POSG) [12], as follows:

$$
\boldsymbol { \mathcal { M } } = \big ( N , S , ( \mathcal { A } _ { i } ) _ { i } , ( O _ { i } ) _ { i } , P , Z , ( r _ { i } ) _ { i } , \rho _ { 0 } , T , \boldsymbol { \gamma } \big ) ,
$$

<sub>w</sub>h<sub>ere</sub> $N = \{ 1 , \ldots , n \}$ is the finite nonem<sub>p</sub>t<sub>y</sub> a<sub>g</sub>ent set<sub>,</sub> and $s , \mathcal { A } _ { i } ,$ <sub>an</sub>d $O _ { i }$ <sub>are</sub> fi<sub>n</sub>it<sub>e</sub> <sub>an</sub>d <sub>nonemp</sub>t<sub>y</sub> f<sub>or</sub> <sub>every</sub> $i \in N .$ <sub>.</sub> M<sub>oreover,</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e</sub> h<sub>or</sub>i<sub>zon</sub> i<sub>s</sub> $T <$ ∞ and the discount factor is $\gamma \in ( 0 , 1 ]$ <sub>.</sub> Th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n i<sub>s</sub> $P ( s ^ { \prime } \mid s , a )$ , t<sup>h</sup>e joint o<sup>b</sup>servation <sup>f</sup>unction i<sub>s</sub> $Z ( o ^ { \prime } \mid s ^ { \prime } , a )$ <sub>, an</sub>d $r _ { i } ( s , a , s ^ { \prime } )$ <sup>i</sup>s a<sub>g</sub>ent $i ^ { \circ } s$ <sub>rewar</sub>d f<sub>unc</sub>ti<sub>on.</sub> At th<sub>e</sub> initi<sub>a</sub>l tim<sub>e</sub> <sub>s</sub>t<sub>ep</sub> $t = 0 ;$ <sub>,</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> d<sub>rawn</sub> f<sub>rom</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\rho _ { 0 } ,$ , so $s _ { 0 } \sim \rho _ { 0 } ,$ an<sup>d</sup> ever<sub>y</sub> a<sub>g</sub>ent <sup>h</sup>as t<sup>h</sup>e em<sub>p</sub>t<sub>y</sub> <sup>hi</sup>stor<sub>y</sub> $h _ { i } ^ { 0 } = \left( \right)$ <sub>.</sub> F<sub>o</sub>r $t \geq 1 ,$ th<sub>e</sub> l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory o</sub>f <sub>agen</sub>t � i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

$$
\boldsymbol { h } _ { i } ^ { t } = ( \boldsymbol { a } _ { i } ^ { 0 } , \boldsymbol { o } _ { i } ^ { 1 } , \dots , \boldsymbol { a } _ { i } ^ { t - 1 } , \boldsymbol { o } _ { i } ^ { t } ) \in \mathcal { H } _ { i } ^ { t } = ( \mathcal { A } _ { i } \times O _ { i } ) ^ { t } .
$$

Th<sub>e po</sub>li<sub>cy</sub> d<sub>oma</sub>i<sub>n con</sub>t<sub>a</sub>i<sub>ns a</sub>ll d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>pro</sub>d<sub>uc</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es, suc</sub>h th<sub>a</sub>t<sub>:</sub>

$$
\Pi _ { \mathrm { d e c } } = \prod _ { i \in N } \prod _ { t = 0 } ^ { T - 1 } \prod _ { h _ { i } ^ { t } \in \mathcal { H } _ { i } ^ { t } } \Delta ( \mathcal { A } _ { i } ) , \qquad \pi ^ { t } ( a \mid h ^ { t } ) = \prod _ { i } \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) .
$$

A<sub>ge</sub>nt<sub>s</sub> m<sub>ay</sub> r<sub>a</sub>nd<sub>o</sub>miz<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y</sub> <sub>a</sub>t th<sub>e</sub>ir l<sub>oca</sub>l hi<sub>s</sub>t<sub>o</sub>ri<sub>es,</sub> b<sub>u</sub>t <sub>we exc</sub>l<sub>u</sub>d<sub>e a common ran</sub>d<sub>om s</sub>i<sub>gna</sub>l<sub>, suc</sub>h <sub>as a s</sub>h<sub>are</sub>d <sub>see</sub>d <sub>or</sub> <sub>p</sub>ublic correlation device [1, 3].<sup>1</sup>

Let $\begin{array} { r } { X _ { t } = S \times \prod _ { i } \mathcal { H } _ { i } ^ { t } } \end{array}$ . Com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation assi<sub>g</sub>ns a value to every va<sup>l</sup>i<sup>d</sup> state–joint-<sup>h</sup>istory pair, inc<sup>l</sup>u<sup>d</sup>ing pairs unreac<sup>h</sup>a<sup>bl</sup>e <sub>un</sub>d<sub>er</sub> <sub>a</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy.</sub> With $V _ { i . T } ^ { \pi } \equiv 0 \mathrm { . }$ <sub>,</sub> f<sub>or</sub> $t < T$ it i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> B<sub>e</sub>ll<sub>man</sub> <sub>recurs</sub>i<sub>on:</sub>

$$
\begin{array} { l } { { V _ { i , t } ^ { \pi } ( s , h ^ { t } ) = \displaystyle \sum _ { a , s ^ { \prime } , o ^ { \prime } } \pi ^ { t } ( a \mid h ^ { t } ) P ( s ^ { \prime } \mid s , a ) Z ( o ^ { \prime } \mid s ^ { \prime } , a ) } } \\ { { . \left[ r _ { i } ( s , a , s ^ { \prime } ) + \gamma V _ { i , t + 1 } ^ { \pi } ( s ^ { \prime } , h ^ { t } , a , o ^ { \prime } ) \right] . } } \end{array}\tag{1}
$$

Th<sub>e</sub> fi<sub>na</sub>l t<sub>rans</sub>iti<sub>on rewar</sub>d i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>a</sub>t $t = T - 1$ . Write

$$
\mathcal { E } _ { \mathcal { M } } : \Pi _ { \mathrm { d e c } } \to \mathcal { V } , \qquad \mathcal { E } _ { \mathcal { M } } ( \pi ) = V ^ { \pi } ,
$$

f<sub>or</sub> th<sub>e</sub> <sub>opera</sub>t<sub>or</sub> <sub>re</sub>t<sub>urn</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>agen</sub>t<sub>s</sub>’ <sub>va</sub>l<sub>ues</sub> <sub>a</sub>t <sub>a</sub>ll ti<sub>me</sub> <sub>s</sub>t<sub>eps</sub> <sub>an</sub>d <sub>p</sub>airs<sub>,</sub> where $_ \mathrm { c } { } _ { V }$ i<sub>s</sub> th<sub>e</sub> <sub>space</sub> <sub>o</sub>f <sub>a</sub>ll f<sub>unc</sub>ti<sub>on</sub> <sub>pro</sub>fil<sub>es</sub> $V = \left( V _ { i , t } \right)$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $V _ { i , t } : X _ { t } \to \mathbb { R }$ f<sub>or</sub> $i \in N$ <sub>an</sub>d 0 $\mid \leq t \leq T$ <sub>,</sub> <sub>w</sub>ith $V _ { i , T } \equiv 0 .$ . Each <sub>p</sub>olic<sub>y</sub> <sub>�</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> <sub>a</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>pro</sub>fil<sub>e</sub> $V ^ { \pi } \in \mathcal { V }$ throu<sub>g</sub>h E<sub>q</sub>uation (1). The <sub>expec</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn un</sub>d<sub>er</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
J _ { i } ( \pi ) = \mathbb { E } _ { s _ { 0 } \sim \rho _ { 0 } } \big [ V _ { i , 0 } ^ { \pi } ( s _ { 0 } , ( ) ^ { N } ) \big ] ,\tag{2}
$$

<sub>w</sub>h<sub>ere</sub> $( ) ^ { N }$ is t<sup>h</sup>e empty joint <sup>h</sup>istory.

Type-compatible transformations. A type-compatible transformation is a com<sub>p</sub>lete tu<sub>p</sub>le

$$
g = \big ( \sigma _ { g } , \boldsymbol { \phi } _ { g } ^ { S } , ( \boldsymbol { \phi } _ { g , i } ^ { \mathcal { A } } ) _ { i } , ( \boldsymbol { \phi } _ { g , i } ^ { O } ) _ { i } \big ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\sigma _ { g }$ p<sup>ermutes</sup> <sup>a</sup>g<sup>ents</sup>, $\phi _ { g } ^ { S }$ <sup>b</sup>ijects t<sup>h</sup>e state space, an<sup>d</sup> $\boldsymbol { \phi } _ { g , i } ^ { \mathcal { R } }$ $\mathcal { A } _ { i }  \mathcal { A } _ { \sigma _ { g } ( i ) }$ <sub>an</sub>d $\phi _ { q , i } ^ { O } : O _ { i } \to O _ { \sigma _ { g } ( i ) }$ are <sup>b</sup>ijections. It acts on joint actions <sup>b</sup>y $[ g \cdot a ] _ { \sigma _ { g } ( i ) } = \phi _ { g , i } ^ { \mathcal { A } } ( a _ { i } )$ , an<sup>d</sup> ana<sup>l</sup>ogous<sup>l</sup>y on joint <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>s.</sub> F<sub>o</sub>r $t \geq 1$ <sub>,</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d l<sub>oca</sub>l<sub>-</sub>hi<sub>s</sub>t<sub>ory</sub> <sub>map</sub> $\Phi _ { g , i } ^ { \mathcal { H } , t } : \mathcal { H } _ { i } ^ { t } $ $\mathcal { H } _ { \sigma _ { g } ( i ) } ^ { t }$ i<sub>s expresse</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
\begin{array} { r l } & { \Phi _ { g , i } ^ { \mathcal { H } , t } ( a _ { i } ^ { 0 } , o _ { i } ^ { 1 } , \dots , a _ { i } ^ { t - 1 } , o _ { i } ^ { t } ) } \\ & { \quad \quad = \big ( \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ^ { 0 } ) , \phi _ { g , i } ^ { \mathcal { O } } ( o _ { i } ^ { 1 } ) , \dots , \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ^ { t - 1 } ) , \phi _ { g , i } ^ { \mathcal { O } } ( o _ { i } ^ { t } ) \big ) , } \end{array}
$$

<sub>an</sub>d $\Phi _ { g , i } ^ { \mathcal { H } , 0 } ( ( ) ) = \left( \right)$ . For a joint <sup>h</sup>istory $h ^ { t } = ( h _ { i } ^ { t } ) _ { i \in N }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
[ g \cdot h ^ { t } ] _ { \sigma _ { g } ( i ) } = \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) .
$$

Th<sub>us</sub> <sub>agen</sub>t �’<sub>s</sub> t<sub>rans</sub>f<sub>orme</sub>d l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory</sub> <sub>occup</sub>i<sub>es</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> $\sigma _ { g } ( i )$ in t<sup>h</sup>e joint <sup>h</sup>istory. T<sup>h</sup>e in<sup>d</sup>uce<sup>d</sup> state–joint-<sup>h</sup>istory map is

$$
\tau _ { g } ^ { t } : X _ { t }  X _ { t } , \qquad \tau _ { g } ^ { t } ( s , h ^ { t } ) = \big ( \phi _ { g } ^ { S } ( s ) , g \cdot h ^ { t } \big ) .
$$

Th<sub>us</sub> $\tau _ { g } ^ { t }$ trans<sup>f</sup>orms t<sup>h</sup>e state an<sup>d</sup> joint <sup>h</sup>istory. It preserves t<sup>h</sup>e time i<sub>n</sub>d<sub>ex</sub><sup>2</sup> <sub>an</sub>d d<sub>oes no</sub>t <sub>per</sub>f<sub>orm a s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> t<sub>o genera</sub>t<sub>e</sub> th<sub>e nex</sub>t state. In particu<sup>l</sup>ar, <sup>f</sup>or t<sup>h</sup>e empty joint <sup>h</sup>istory $\zeta ) ^ { N } , \tau _ { g } ^ { 0 } ( s , \ l ( \ l ) ^ { N } ) =$ $( \phi _ { q } ^ { S } ( s ) , ( ) ^ { N } )$

We fix an a<sub>pp</sub>lication-s<sub>p</sub>ecified sub<sub>g</sub>rou<sub>p</sub> $\mathscr { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ <sub>o</sub>f <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> t<sub>yp</sub>e-com<sub>p</sub>atible transformations arisin<sub>g</sub> from re<sub>p</sub>resentation struct<sub>ure</sub> fi<sub>xe</sub>d f<sub>or</sub> th<sub>e</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>on.</sub> E<sub>xamp</sub>l<sub>es</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>permu</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>o</sub>f <sub>an</sub> entit<sub>y</sub>-factored multi-a<sub>g</sub>ent re<sub>p</sub>resentation [13], node relabelin<sub>g</sub>s of a <sub>g</sub>ra<sub>p</sub>h-structured re<sub>p</sub>resentation [18], or <sub>g</sub>eometric state–action transformations such as rotations and reflections [32]. Membershi<sub>p</sub> in $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ su<sub>pp</sub>lies a candidate transformation<sub>;</sub> an environment or <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> must also satisf<sub>y</sub> the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>r<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>co</sub>nditi<sub>o</sub>n<sub>s.</sub>

Th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> <sub>o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es</sub> <sub>an</sub>d <sub>va</sub>l<sub>ues</sub> <sub>are</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>p</sub>ointwise b<sub>y</sub> the followin<sub>g</sub> e<sub>q</sub>uations:

$$
\left( g \cdot \pi \right) _ { \sigma _ { g } ( i ) } ^ { t } \left( \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \mid \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) \right) = \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) ,\tag{3}
$$

$$
( g \cdot V ) _ { \sigma _ { g } ( i ) , t } ( \tau _ { g } ^ { t } x _ { t } ) = V _ { i , t } ( x _ { t } ) .\tag{4}
$$

T<sup>h</sup>ese trans<sup>f</sup>ormations <sup>d</sup>e<sup>fi</sup>ne <sup>l</sup>e<sup>f</sup>t group actions. At eac<sup>h fi</sup>xe<sup>d</sup> joint <sup>h</sup>istory, t<sup>h</sup>e trans<sup>f</sup>orme<sup>d</sup> joint action <sup>d</sup>istri<sup>b</sup>ution equa<sup>l</sup>s t<sup>h</sup>e pro<sup>d</sup>uct <sub>o</sub>f th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d l<sub>oca</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub> B<sub>ecause</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>po</sub>li<sub>cy</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>on</sub>l<sub>y</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>an</sub>d <sub>re</sub>i<sub>n</sub>d<sub>exes</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> $g \cdot \pi$ remains a decentralized <sub>p</sub>roduct <sub>p</sub>olic<sub>y</sub> in $\Pi _ { \mathrm { d e c } }$ . A<sub>pp</sub>endix B <sub>p</sub>roves these action <sub>p</sub>ro<sub>p</sub>erties to<sub>g</sub>ether with the sub<sub>g</sub>rou<sub>p</sub> and orbit facts <sub>use</sub>d i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>sec</sub>ti<sub>ons.</sub>

## 4 Policy-Evaluation Symmetry and Structural Agent Types

Definition 4.1 (Policy-evaluation symmetry). An admissible transf<sub>orma</sub>ti<sub>on</sub> $g \in \mathcal { R } _ { \mathrm { a d m } } ( M )$ is a policy-evaluation symmetry if

$$
\begin{array} { r } { \mathcal { E } _ { M } ( g \cdot \pi ) = g \cdot \mathcal { E } _ { M } ( \pi ) \qquad \mathrm { f o r ~ e v e r y ~ } \pi \in \Pi _ { \mathrm { d e c } } . } \end{array}
$$

E<sub>q</sub>uivalentl<sub>y,</sub>

$$
V _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi } ( \tau _ { g } ^ { t } x _ { t } ) = V _ { i , t } ^ { \pi } ( x _ { t } )
$$

<sup>f</sup>or ever<sub>y</sub> <sub>p</sub>o<sup>li</sup>c<sub>y</sub>, a<sub>g</sub>ent, t<sup>i</sup>me ste<sub>p</sub>, an<sup>d</sup> $x _ { t } \in X _ { t }$ <sub>.</sub> Th<sub>e se</sub>t <sub>o</sub>f <sub>a</sub>ll <sub>suc</sub>h t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$

For a normal-form <sub>g</sub>ame embedded as a one-sta<sub>g</sub>e POSG<sub>,</sub> this d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> <sub>payo</sub>f <sub>preserva</sub>ti<sub>on</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>p</sub>l<sub>ayer</sub> and action relabelin<sub>g</sub>s (see Pro<sub>p</sub>osition B.4).

The re<sub>qu</sub>irement that the identit<sub>y</sub> hold for e<sub>v</sub>er<sub>y p</sub>olic<sub>y</sub> $\pi \in \Pi _ { \mathrm { d e c } }$ is essential. The definition sa<sub>y</sub>s that transformin<sub>g</sub> an<sub>y p</sub>olic<sub>y</sub> and then evaluatin<sub>g</sub> it <sub>g</sub>ives the corres<sub>p</sub>ondin<sub>g</sub>l<sub>y</sub> transformed com<sub>p</sub>lete <sub>va</sub>l<sub>ue pro</sub>fil<sub>e.</sub> It d<sub>oes no</sub>t <sub>requ</sub>i<sub>re</sub> $V ^ { \pi }$ t<sub>o</sub> b<sub>e</sub> i<sub>nvar</sub>i<sub>an</sub>t <sub>un</sub>d<sub>er</sub> ${ \mathit { g } } ,$ b<sub>ecause</sub> in <sub>g</sub>eneral the com<sub>p</sub>arison is between <sub>�</sub> and $g \cdot \pi$ . The <sub>g</sub>rou<sub>p</sub>-action l<sub>aws</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>a</sub>t $G _ { M } ^ { \mathrm { { \bar { e v a l } } } } \leq \mathcal { R } _ { \mathrm { a d m } } ( M )$

Definition 4.2 (Structural agent type). Agents $i , j \in N$ h<sub>ave</sub> th<sub>e</sub> same structural agent type when

$$
i \sim _ { \mathrm { e v a l } } j \quad \Longleftrightarrow \quad { \mathrm { s o m e } } g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } } { \mathrm { ~ s a t i s f i e s ~ } } \sigma _ { g } ( i ) = j .
$$

Th<sub>e</sub> <sub>s</sub>tr<sub>uc</sub>t<sub>u</sub>r<sub>a</sub>l <sub>pa</sub>rtiti<sub>o</sub>n i<sub>s</sub> th<sub>e</sub> <sub>o</sub>rbit <sub>pa</sub>rtiti<sub>o</sub>n

$$
\mathcal { P } _ { \mathrm { e v a l } } ( \boldsymbol { M } ) = \mathcal { P } ( G _ { \boldsymbol { M } } ^ { \mathrm { e v a l } } ) = \big \{ \{ \sigma _ { g } ( i ) : g \in G _ { \boldsymbol { M } } ^ { \mathrm { e v a l } } \} : i \in N \big \} .
$$

Writin<sub>g</sub> $\mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } ) = \{ T _ { 1 } , . . . , T _ { m } \}$ <sub>,</sub> <sub>eac</sub>h $T _ { k }$ is one structural a<sub>g</sub>ent t<sub>yp</sub>e. The multi-a<sub>g</sub>ent s<sub>y</sub>stem is structurall<sub>y</sub> homo<sub>g</sub>eneous if $m = 1$ <sub>an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y</sub> h<sub>e</sub>t<sub>erogeneous</sub> if $m > 1$ . Hetero<sub>g</sub>eneit<sub>y</sub> therefore describes the com<sub>p</sub>lete <sub>p</sub>artition<sub>,</sub> rather than one <sub>p</sub>articular t<sub>yp</sub>e.

B<sub>ecause</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ acts on � throu<sub>g</sub>h the a<sub>g</sub>ent ma<sub>p</sub>s $i \mapsto \sigma _ { g } ( i )$ $\sim _ { \mathrm { e v a l } }$ is the standard orbit e<sub>q</sub>uivalence relation [16, Cha<sub>p</sub>. $1 4 . 1 ] . ^ { 3 }$ M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>ll<sub>y,</sub> th<sub>e case</sub> $m = 1$ <sub>means</sub> th<sub>a</sub>t th<sub>e ac</sub>ti<sub>on o</sub>f $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ on � is transiti<sub>v</sub>e: for e<sub>v</sub>er<sub>y p</sub>air of a<sub>g</sub>ents $i , j \in N ,$ some $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\sigma _ { q } ( i ) = j$

To obtain an e<sub>q</sub>uivariance relation for ex<sub>p</sub>ected returns $J _ { i } ( \pi )$ under the initial distribution<sub>,</sub> we additionall<sub>y</sub> re<sub>q</sub>uire the state transformation to <sub>p</sub>reserve $\rho _ { 0 }$ <sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>su</sub>b<sub>group,</sub> <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
G _ { \mathcal M } ^ { \mathrm { e v a l } , \rho _ { 0 } } = \left\{ g \in G _ { \mathcal M } ^ { \mathrm { e v a l } } : \rho _ { 0 } \bigl ( \phi _ { g } ^ { S } ( s ) \bigr ) = \rho _ { 0 } ( s ) \mathrm { f o r e v e r y } s \in S \right\} ,
$$

where the state transformation must <sub>p</sub>reserve the initial distribution. Th<sub>ere</sub>f<sub>ore,</sub> <sub>every</sub> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>equa</sub>ti<sub>on:</sub>

$$
J _ { \sigma _ { g } ( i ) } ( g \cdot \pi ) = J _ { i } ( \pi ) .
$$

Policy equivariance is downstream. For an arbitrary subgroup $H \leq { \dot { G } } _ { M } ^ { \mathrm { e v a l } }$ , we ca<sup>ll</sup> a se<sup>l</sup>ecte<sup>d</sup> joint po<sup>l</sup>icy �-equivariant w<sup>h</sup>en $g \cdot \pi = \pi ,$ <sup>f</sup>or ever<sub>y</sub> $g \in H$ . Pointwise<sub>,</sub> this is re<sub>p</sub>resented b<sub>y</sub> the followin<sub>g</sub> e<sub>q</sub>uation:

$$
\pi _ { \sigma _ { g } ( i ) } ^ { t } \left( \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \mid \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) \right) = \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) .
$$

<sup>F</sup>or ever<sub>y</sub> $g \in H ,$ <sub>p</sub>o<sup>li</sup>c<sub>y</sub>-eva<sup>l</sup>uat<sup>i</sup>on s<sub>y</sub>mmetr<sub>y g</sub><sup>i</sup>ves $g \cdot V ^ { \pi } = V ^ { g \cdot \pi }$ If the selected <sub>p</sub>olic<sub>y</sub> is �-e<sub>q</sub>uivariant<sub>,</sub> then $g \cdot \pi = \pi ,$ <sub>, so</sub> it<sub>s va</sub>l<sub>ue</sub> <sub>pro</sub>fil<sub>e</sub> i<sub>s a</sub>l<sub>so unc</sub>h<sub>ange</sub>d<sub>:</sub>

$$
g \cdot V ^ { \pi } = V ^ { g \cdot \pi } = V ^ { \pi } .
$$

The re<sub>v</sub>erse im<sub>p</sub>lication need not hold. The e<sub>qu</sub>alit<sub>y</sub> $g \cdot V ^ { \pi } = V ^ { \pi }$ f<sub>or one se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy</sub> d<sub>oes no</sub>t i<sub>mp</sub>l<sub>y</sub> th<sub>a</sub>t $g \cdot \pi = \pi $ <sub>, nor</sub> d<sub>oes</sub> it <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> id<sub>en</sub>tit<sub>y</sub> f<sub>or every po</sub>li<sub>cy:</sub>

$$
V ^ { g \cdot \eta } = g \cdot V ^ { \eta } \quad \mathrm { f o r e v e r y } \eta \in \Pi _ { \mathrm { d e c } } ,
$$

as re<sub>q</sub>uired for <sub>�</sub> to be a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>.

## 5 Componentwise Environment Automorphisms

Com<sub>p</sub>onent<sub>w</sub>ise <sub>p</sub>rimiti<sub>v</sub>e-<sub>p</sub>reser<sub>v</sub>in<sub>g</sub> transformations<sub>,</sub><sup>4</sup> s<sub>u</sub>ch as transformations that se<sub>p</sub>aratel<sub>y p</sub>reserve the transition<sub>,</sub> observation<sub>,</sub> and re<sub>w</sub>ard f<sub>u</sub>nctions<sub>,</sub> a<sub>pp</sub>ear as POSG a<sub>u</sub>tomor<sub>p</sub>hisms or s<sub>y</sub>mmetries [17], Dec-POMDP e<sub>q</sub>uivalence ma<sub>pp</sub>in<sub>g</sub>s [14], Dec-POMDP automor<sub>p</sub>hisms [30], and environment s<sub>y</sub>mmetries [22]. We use the name componentwise environment automorphism for this notion in POSGs<sub>,</sub> em<sub>p</sub>hasizin<sub>g</sub> its contrast with <sub>p</sub>olic<sub>y</sub>-eval<sub>u</sub>ation s<sub>y</sub>mmetr<sub>y</sub>. Unlike the Dec-POMDP automor<sub>p</sub>hisms of Treutlein et al. [30], <sub>p</sub>reservation of the initial distribution is im<sub>p</sub>osed se<sub>p</sub>aratel<sub>y</sub> in our f<sub>ramewor</sub>k<sub>, as we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub>

Definition 5.1 (Componentwise environment automorphism [17]). An <sub>a</sub>dmi<sub>ss</sub>ibl<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n <sub>�</sub> i<sub>s a co</sub>m<sub>po</sub>n<sub>e</sub>nt<sub>w</sub>i<sub>se e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt <sub>au</sub>t<sub>omorp</sub>hi<sub>sm w</sub>h<sub>en,</sub> f<sub>or every we</sub>ll<sub>-</sub>t<sub>ype</sub>d $s , s ^ { \prime } , a , o ^ { \prime }$ <sub>,</sub> <sub>an</sub>d �<sub>,</sub>

$$
P ( \phi _ { g } ^ { S } ( s ^ { \prime } ) \mid \phi _ { g } ^ { S } ( s ) , g \cdot a ) = P ( s ^ { \prime } \mid s , a ) ,\tag{5}
$$

$$
Z ( g \cdot o ^ { \prime } \mid \phi _ { g } ^ { S } ( s ^ { \prime } ) , g \cdot a ) = Z ( o ^ { \prime } \mid s ^ { \prime } , a ) ,\tag{6}
$$

$$
r _ { \sigma _ { g } ( i ) } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) = r _ { i } ( s , a , s ^ { \prime } ) .\tag{7}
$$

Write $\mathrm { A u t _ { e n v } } ( M )$ f<sub>or</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>componen</sub>t<sub>w</sub>i<sub>se</sub> <sub>env</sub>i<sub>ronmen</sub>t <sub>au</sub>t<sub>o-</sub> mor<sub>p</sub>hisms.

Theorem 5.2 (Componentwise environment automorphisms are sufficient). The componentwise environment automorphisms form a subgroup and

$$
\mathrm { A u t } _ { \mathrm { e n v } } ( { \cal M } ) \leq G _ { \cal M } ^ { \mathrm { e v a l } } , \qquad \mathcal { P } ( \mathrm { A u t } _ { \mathrm { e n v } } ( { \cal M } ) ) \preceq { \mathcal P } _ { \mathrm { e v a l } } ( { \cal M } ) ,
$$

where $\preceq$ means finer than or equal to.

Proof sketch. The identity transformation preserves all three environment com<sub>p</sub>onents. If two transformations <sub>p</sub>reserve them<sub>,</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the <sub>p</sub>reservation e<sub>q</sub>uations successivel<sub>y</sub> shows that their com<sub>p</sub>osition also <sub>p</sub>reser<sub>v</sub>es them. S<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> in<sub>v</sub>erse ima<sub>g</sub>es into th<sub>e</sub> <sub>same</sub> <sub>equa</sub>ti<sub>ons</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>eac</sub>h i<sub>nverse</sub> <sub>preserves</sub> th<sub>em.</sub> H<sub>ence</sub> $\mathrm { A u t _ { e n v } } ( M )$ <sup>i</sup>s a su<sup>b</sup><sub>g</sub>rou<sub>p</sub>.

It r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> t<sub>o</sub> <sub>p</sub>r<sub>ove</sub> th<sub>e</sub> in<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n in $G _ { \mathcal { M } } ^ { \mathrm { e v a l } } . \mathrm { A t } ~ t = T _ { \ast }$ <sub>,</sub> b<sub>o</sub>th <sub>va</sub>l<sub>ues</sub> in D<sub>e</sub>finiti<sub>o</sub>n 4<sub>.</sub>1 <sub>a</sub>r<sub>e</sub> z<sub>e</sub>r<sub>o.</sub> A<sub>ssu</sub>m<sub>e</sub> th<sub>e</sub>ir <sub>equa</sub>lit<sub>y</sub> <sub>a</sub>t tim<sub>e</sub> <sub>s</sub>t<sub>ep</sub> $t + 1$ <sub>an</sub>d <sub>expan</sub>d th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d B<sub>e</sub>ll<sub>man recurs</sub>i<sub>on a</sub>t ti<sub>me s</sub>t<sub>ep</sub> �<sub>.</sub> M<sub>a</sub>k<sub>e</sub> t<sup>h</sup>e <sup>b</sup>ijective su<sup>b</sup>stitutions

$$
b = g \cdot a , \qquad \bar { s } ^ { \prime } = \phi _ { g } ^ { S } ( s ^ { \prime } ) , \qquad \bar { o } ^ { \prime } = g \cdot o ^ { \prime } .
$$

The <sub>p</sub>olic<sub>y</sub>-transformation rule in E<sub>q</sub>uation (3) matches the trans-<sup>f</sup>orme<sup>d</sup> joint-action pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity wit<sup>h</sup> t<sup>h</sup>e source pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity. Equations $( 5 ) - ( 7 )$ <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on,</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>rewar</sub>d t<sub>erms</sub> with their source counter<sub>p</sub>arts. The histor<sub>y</sub> transformation identi-<sup>fi</sup>es correspon<sup>d</sup>ing next state–joint-<sup>h</sup>istory pairs. T<sup>h</sup>e in<sup>d</sup>uction <sup>h</sup>y-<sub>p</sub>othesis then e<sub>q</sub>uates their values at time ste<sub>p</sub> � + 1 under the transf<sub>orme</sub>d <sub>an</sub>d <sub>source po</sub>li<sub>c</sub>i<sub>es.</sub> Th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d B<sub>e</sub>ll<sub>man sum</sub> th<sub>ere</sub>f<sub>ore</sub> reduces to E<sub>q</sub>uation (1). Conse<sub>q</sub>uentl<sub>y</sub>, $V _ { \sigma _ { q } ( i ) , t } ^ { g \cdot \pi } ( \tau _ { g } ^ { t } x _ { t } ) = V _ { i , t } ^ { \pi } ( x _ { t } )$ f<sub>or</sub> <sup>ever</sup>y $g \in \mathrm { A u t _ { e n v } } ( \mathcal { M } )$ , <sub>p</sub>o<sup>li</sup>c<sub>y</sub> �, a<sub>g</sub>ent �, t<sup>i</sup>me ste<sub>p</sub> �, an<sup>d</sup> $x _ { t } \in X _ { t }$ Thus<sub>,</sub> ever<sub>y</sub> com<sub>p</sub>onentwise environment automor<sub>p</sub>hism is a <sub>p</sub>olic<sub>y</sub>- <sup>evaluation s</sup>y<sup>mmetr</sup>y, p<sup>rovin</sup>g $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { M } ^ { \mathrm { e v a l } }$ . The sub<sub>g</sub>rou<sub>p</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on a</sub>l<sub>so g</sub>i<sub>ves</sub> th<sub>e par</sub>titi<sub>on re</sub>fi<sub>nemen</sub>t<sub>, as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
\mathcal { P } ( \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } ) ) \preceq \mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } ) ,
$$

b<sub>ecause every or</sub>bit <sub>un</sub>d<sub>er</sub> $\mathrm { A u t _ { e n v } } ( M )$ i<sub>s co</sub>nt<sub>a</sub>in<sub>e</sub>d in <sub>a</sub>n <sub>o</sub>rbit <sub>u</sub>nd<sub>e</sub>r $G _ { M } ^ { \mathrm { e v a l } }$ . A<sub>pp</sub>endix C <sub>g</sub>i<sub>v</sub>es the com<sub>p</sub>lete <sub>p</sub>roof of Theorem 5.2.

Th<sub>e</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on can y</sub>i<sub>e</sub>ld <sub>a s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>re</sub>fi<sub>nemen</sub>t<sub>, as</sub> f<sub>o</sub>ll<sub>ows:</sub>

Proposition 5.3 (Strictness of the componentwise sufficient condition). There exists a finite, full-support, two-agent, twostage POSG for which

$$
\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { \mathcal { M } } ^ { \mathrm { e v a l } } \quad a n d \quad \mathcal { P } ( \mathrm { A u t } _ { \mathrm { e n v } } ( M ) ) \prec \mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } ) .
$$

Here ≺ means strictly finer than.

Proofsketch. Appendix B.2 constructs a full-support two-stage POSG in which exchan<sub>g</sub>in<sub>g</sub> the a<sub>g</sub>ents chan<sub>g</sub>es the transition distri-<sup>b</sup>ution associate<sup>d</sup> wit<sup>h</sup> some joint actions. T<sup>h</sup>e exc<sup>h</sup>ange is t<sup>h</sup>ere<sup>f</sup>ore not a com<sub>p</sub>onentwise environment automor<sub>p</sub>hism. Nevertheless<sub>,</sub> th<sub>e a</sub>f<sub>ec</sub>t<sub>e</sub>d t<sub>rans</sub>iti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons pro</sub>d<sub>uce</sub> th<sub>e same expec</sub>t<sub>e</sub>d <sub>successor-s</sub>t<sub>a</sub>t<sub>e rewar</sub>d <sub>an</sub>d <sub>marg</sub>i<sub>na</sub>li<sub>ze</sub>d <sub>nex</sub>t<sub>-o</sub>b<sub>serva</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u-</sub> tion. The Bellman recursion <sub>p</sub>reserves com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation <sub>un</sub>d<sub>er</sub> th<sub>e exc</sub>h<sub>ange</sub> f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy.</sub> Th<sub>e rewar</sub>d construction also rules out ever<sub>y</sub> alternative com<sub>p</sub>onentwise transformation that exchan<sub>g</sub>es the a<sub>g</sub>ents. Hence<sub>,</sub> the com<sub>p</sub>onentwise <sub>p</sub>artition se<sub>p</sub>arates the a<sub>g</sub>ents<sub>,</sub> whereas the <sub>p</sub>olic<sub>y</sub>-evaluation <sub>p</sub>artition <sub>p</sub>laces them in one structural a<sub>g</sub>ent t<sub>yp</sub>e.

## 6 Extension to Dec-POMDPs

We a<sub>pp</sub>l<sub>y</sub> the <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> and orbit framework to two settin<sub>g</sub>s of Dec-POMDP [4]. The first is a Dec-POMDP with <sub>a ge</sub>n<sub>e</sub>r<sub>a</sub>l <sub>co</sub>mm<sub>o</sub>n r<sub>ewa</sub>rd<sub>.</sub> Th<sub>e seco</sub>nd i<sub>s a</sub> d<sub>e</sub>ri<sub>ve</sub>d D<sub>ec</sub>-POMDP formed b<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> the individual rewards of a POSG.

## 6.1 General Common Rewards

A finit<sub>e</sub>-h<sub>o</sub>riz<sub>o</sub>n POSG <sub>spec</sub>i<sub>a</sub>liz<sub>es</sub> t<sub>o a</sub> D<sub>ec</sub>-POMDP <sub>w</sub>h<sub>e</sub>n <sub>a</sub>ll <sub>age</sub>nt<sub>s</sub> <sub>s</sub>h<sub>are</sub> <sub>one</sub> <sub>rewar</sub>d f<sub>unc</sub>ti<sub>on,</sub> $r _ { i } = r ^ { \mathrm { c o m } }$ <sup>f</sup>or ever<sub>y</sub> $i \in N .$ Th<sub>e</sub> <sub>common</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>con</sub>diti<sub>on</sub> <sub>an</sub>d B<sub>e</sub>ll<sub>man</sub> <sub>recurs</sub>i<sub>on</sub> th<sub>en</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>a</sub>t

$$
V _ { i , t } ^ { \pi } ( \boldsymbol { x } ) = V _ { j , t } ^ { \pi } ( \boldsymbol { x } ) \qquad \mathrm { f o r ~ e v e r y ~ } i , j , \pi , t , \mathrm { ~ a n d ~ } \boldsymbol { x } \in \boldsymbol { X } _ { t } .
$$

Thi<sub>s equa</sub>lit<sub>y compares</sub> th<sub>e agen</sub>t<sub>s</sub>’ <sub>va</sub>l<sub>ue coor</sub>di<sub>na</sub>t<sub>es un</sub>d<sub>er</sub> th<sub>e</sub> same po<sup>l</sup>icy an<sup>d</sup> at t<sup>h</sup>e same state–joint-<sup>h</sup>istory input. However, it d<sub>oes no</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub>i<sub>r</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng ro</sub>l<sub>es are</sub> i<sub>n</sub>t<sub>erc</sub>h<sub>angea</sub>bl<sub>e.</sub> By De<sup>fi</sup>nition 4.2, agents � an<sup>d</sup> � <sup>h</sup>ave t<sup>h</sup>e same structura<sup>l</sup> agent t<sub>yp</sub>e on<sup>l</sup><sub>y</sub> <sup>if</sup> some $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ maps � to �. Suc<sup>h</sup> a trans<sup>f</sup>ormation must inc<sup>l</sup>u<sup>d</sup>e compati<sup>bl</sup>e <sup>b</sup>ijections <sup>b</sup>etween t<sup>h</sup>e agents<sup>’</sup> action an<sup>d</sup> o<sup>b</sup>- servation s<sub>p</sub>aces and must <sub>p</sub>reserve com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation for ever<sub>y</sub> decentralized <sub>p</sub>olic<sub>y</sub> after exchan<sub>g</sub>in<sub>g</sub> the a<sub>g</sub>ents. A common rewar<sup>d</sup> a<sup>l</sup>one guarantees neit<sup>h</sup>er t<sup>h</sup>ese <sup>b</sup>ijections nor symmetry o<sup>f</sup> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n <sub>a</sub>nd <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub> F<sub>o</sub>r <sub>e</sub>x<sub>a</sub>m<sub>p</sub>l<sub>e,</sub> in <sub>a</sub> <sub>o</sub>n<sub>e</sub>- <sub>s</sub>t<sub>age</sub> bi<sub>nary-ac</sub>ti<sub>on mo</sub>d<sub>e</sub>l <sub>w</sub>ith th<sub>e common rewar</sub>d $r ( a _ { 1 } , a _ { 2 } ) = a _ { 1 }$ b<sub>o</sub>th <sub>agen</sub>t<sub>s rece</sub>i<sub>ve</sub> th<sub>e same payo</sub>f<sub>,</sub> b<sub>u</sub>t <sub>exc</sub>h<sub>ang</sub>i<sub>ng</sub> th<sub>em c</sub>h<sub>anges</sub> <sub>w</sub>hi<sub>c</sub>h <sub>agen</sub>t’<sub>s po</sub>li<sub>cy</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>a</sub>t <sub>payo</sub>f<sub>.</sub> Th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>proo</sub>f <sub>o</sub>f th<sub>e a</sub>b<sub>ove s</sub>t<sub>a</sub>t<sub>e</sub>m<sub>e</sub>nt i<sub>s g</sub>i<sub>ve</sub>n in Pr<sub>opos</sub>iti<sub>o</sub>n D<sub>.</sub>1 in A<sub>ppe</sub>ndix D<sub>.</sub>

Hence<sub>,</sub> a Dec-POMDP ma<sub>y</sub> still ha<sub>v</sub>e m<sub>u</sub>lti<sub>p</sub>le str<sub>u</sub>ct<sub>u</sub>ral a<sub>g</sub>ent t<sub>yp</sub>es des<sub>p</sub>ite the common reward<sub>,</sub> and it is structurall<sub>y</sub> homo<sub>g</sub>e-<sub>neous exac</sub>tl<sub>y w</sub>h<sub>en</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ acts transitivel<sub>y</sub> on the a<sub>g</sub>ent set �.

## 6.2 Aggregate Common Rewards

Let M be a POSG with a<sub>g</sub>ent-s<sub>p</sub>ecific reward functions $( r _ { i } ) _ { i \in N }$ Fix nonne<sub>g</sub>ative wei<sub>g</sub>hts � ∈ $\vdots \bar { \mathbb { R } } _ { \geq 0 } ^ { \hat { N } } \setminus \{ 0 \}$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>aggrega</sub>t<sub>e</sub> <sub>common rewar</sub>d<sub>, as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
r ^ { w } ( s , a , s ^ { \prime } ) = \sum _ { i \in N } w _ { i } r _ { i } ( s , a , s ^ { \prime } ) .
$$

H<sub>e</sub>r<sub>e</sub> $r ^ { w }$ i<sub>s on</sub>l<sub>y a sca</sub>l<sub>ar</sub> f<sub>unc</sub>ti<sub>on cons</sub>t<sub>ruc</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e rewar</sub>d<sub>s</sub> of M, so definin<sub>g</sub> it does not re<sub>p</sub>lace the source reward functions in M. We construct another Dec-POMDP $M _ { w }$ b<sub>y</sub> retainin<sub>g</sub> the a<sub>g</sub>ents<sub>,</sub> state<sub>,</sub> action<sub>,</sub> and observation s<sub>p</sub>aces<sub>,</sub> transition and obser-<sub>va</sub>ti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s,</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>,</sub> h<sub>o</sub>riz<sub>o</sub>n<sub>,</sub> di<sub>scou</sub>nt f<sub>ac</sub>t<sub>o</sub>r<sub>,</sub> <sub>an</sub>d <sub>po</sub>li<sub>cy</sub> d<sub>oma</sub>i<sub>n</sub> <sub>o</sub>f $M ,$ while assi<sub>g</sub>nin<sub>g</sub> $r ^ { w }$ <sup>to</sup> <sup>ever</sup>y <sup>a</sup>g<sup>ent:</sup>

$$
r _ { j } ^ { \mathcal { M } _ { w } } = r ^ { w } \qquad \mathrm { f o r ~ e v e r y ~ } j \in N .
$$

Thus, M and $\mathcal { M } _ { w }$ difer onl<sub>y</sub> in their reward s<sub>p</sub>ecifications. We test b<sub>o</sub>th <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>same</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons,</sub> so an<sub>y</sub> diference between their s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub>s comes from re-<sub>p</sub>l<sub>ac</sub>i<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>rewar</sub>d<sub>s</sub> <sub>w</sub>ith th<sub>e</sub> <sub>common</sub> <sub>rewar</sub>d $r ^ { w }$ <sub>,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> f<sub>rom</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> dif<sub>eren</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub>

Theorem 6.1 (Weighted aggregation and inherited symmetry). For every agent �, policy �, time step �, and � $\in X _ { t }$

$$
V _ { j , t } ^ { \pi ; \mathcal { M } _ { w } } ( x ) = \sum _ { i \in N } w _ { i } V _ { i , t } ^ { \pi ; \mathcal { M } } ( x ) .
$$

Moreover,

$$
\left\{ g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } } : w _ { \sigma _ { g } ( i ) } = w _ { i } f o r a l l i \right\} \leq G _ { \mathcal { M } _ { w } } ^ { \mathrm { e v a l } } .
$$

At the componentwise level,

$$
\left\{ g \in \mathrm { A u t } _ { \mathrm { e n v } } ( M ) : w _ { \sigma _ { g } ( i ) } = w _ { i } f o r a l l i \right\} \leq \mathrm { A u t } _ { \mathrm { e n v } } ( M _ { w } ) .
$$

Proof sketch. $\mathbf { A } \mathbf { t } \ t = T$ <sub>,</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> <sub>o</sub>f th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> id<sub>en</sub>tit<sub>y</sub> <sub>are</sub> <sub>zero.</sub> S<sub>upp</sub>ose the identit<sub>y</sub> holds at $t + 1$ <sub>.</sub> F<sub>o</sub>r $x = ( s , h ^ { t } )$ <sub>,</sub> write $x ^ { \prime }$ f<sub>or</sub> th<sub>e</sub> successor state–joint-<sup>h</sup>istory input <sup>d</sup>etermine<sup>d</sup> <sup>b</sup>y $s ^ { \prime } , a , o ^ { \prime }$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $r ^ { w }$ <sub>an</sub>d th<sub>e</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man recurs</sub>i<sub>on o</sub>f $M _ { w }$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { l } { { \displaystyle V _ { j , t } ^ { \pi ; M _ { w } } ( x ) = \sum _ { a , s ^ { \prime } , o ^ { \prime } } \pi ^ { t } ( a \mid h ^ { t } ) P ( s ^ { \prime } \mid s , a ) Z ( o ^ { \prime } \mid s ^ { \prime } , a ) } } \\ { { \displaystyle \qquad \cdot \sum _ { i \in N } w _ { i } \left[ r _ { i } ( s , a , s ^ { \prime } ) + \gamma V _ { i , t + 1 } ^ { \pi ; M } ( x ^ { \prime } ) \right] = \sum _ { i \in N } w _ { i } V _ { i , t } ^ { \pi ; M } ( x ) . } } \end{array}
$$

Th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> <sub>moves</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>sum</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e.</sub> E<sub>ac</sub>h remainin<sub>g</sub> term is the Bellman recursion for a<sub>g</sub>ent � in M. Backward i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>ere</sub>f<sub>ore proves</sub> th<sub>a</sub>t $\begin{array} { r } { V _ { j , t } ^ { \pi ; \mathcal { M } _ { w } } ( x ) = \sum _ { i \in N } w _ { i } V _ { i , t } ^ { \pi ; \mathcal { M } } ( x ) } \end{array}$

For the s<sub>y</sub>mmetr<sub>y</sub> inclusion<sub>,</sub> let $g \in G _ { M } ^ { \mathrm { e v a l } }$ <sup>and</sup> <sup>su</sup>pp<sup>ose</sup> $w _ { \sigma _ { g } ( i ) } =$ $w _ { i }$ for ever<sub>y</sub> �. A<sub>pp</sub>l<sub>y</sub> the value identit<sub>y</sub> above to <sub>p</sub>olic<sub>y</sub> $g \cdot \pi$ <sub>an</sub>d <sup>i</sup>n<sub>p</sub>ut $\tau _ { g } ^ { t } x$ . Reindexin<sub>g</sub> b<sub>y</sub> $k = \sigma _ { g } ( i )$ and usin<sub>g</sub> wei<sub>g</sub>ht <sub>p</sub>reservation

![](images/209bdc01821c35fab27a3f755b28c687349053700bdba14b2ad8c74a008ef54d.jpg)

![](images/55fab71a99d7c0ec69ae8c400fc1e905b9c319e08fb7e89ef6f9cd7d5275c929.jpg)  
Figure 1: Hierarchy ofkey concepts and results. ≤ denotes subgroup inclusion. ⪯ means finer than or equal to. The schematic inset illustrates orbit refinement: each componentwise orbit lies within one structural agent type. Initial-distribution preservation and selected-policy equivariance impose separate conditions for expected-return equivariance and invariance of a selected value profile. Panel (c) shows that a common reward need not imply structural homogeneity. For the derived common-reward model, equal weights within each original structural type guarantee symmetry inheritance, so types are preserved or merged.

and <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> in M then <sub>g</sub>ives:

$$
\begin{array}{c} \begin{array} { l l l } { { \displaystyle V _ { \sigma _ { g } ( j ) , t } ^ { g \cdot \pi ; M _ { w } } ( \tau _ { g } ^ { t } x ) = \sum _ { i \in N } w _ { \sigma _ { g } ( i ) } V _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi ; M } ( \tau _ { g } ^ { t } x ) } } \\ { { \displaystyle ~ } } \\ { { \displaystyle ~ } } \end{array} = \sum _ { i \in N } ^ { } w _ { i } V _ { i , t } ^ { \pi ; M } ( x ) = V _ { j , t } ^ { \pi ; M _ { w } } ( x ) .  \end{array}
$$

Because this holds for ever<sub>y</sub> <sub>p</sub>olic<sub>y,</sub> a<sub>g</sub>ent<sub>,</sub> time ste<sub>p,</sub> and evaluation <sup>in</sup>p<sup>ut</sup>, <sup>we</sup> g<sup>et</sup> $g \in G _ { M _ { w } } ^ { \mathrm { e v a l } }$

For the com<sub>p</sub>onentwise inclusion<sub>,</sub> let a wei<sub>g</sub>ht-<sub>p</sub>reservin<sub>g</sub> $g \in$ $\mathrm { A u t _ { e n v } } ( M )$ be <sub>g</sub>iven. It <sub>p</sub>reserves the transition and observation distrib<sub>u</sub>tions co<sub>p</sub>ied to $\mathcal { M } _ { w }$ . Reindexin<sub>g</sub> the a<sub>gg</sub>re<sub>g</sub>ate reward and usin<sub>g</sub> reward <sub>p</sub>reservation <sub>g</sub>ives the e<sub>q</sub>uations<sub>,</sub> as follows:

$$
\begin{array} { r l } { r ^ { w } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) = \displaystyle \sum _ { i \in N } w _ { \sigma _ { g } ( i ) } r _ { \sigma _ { g } ( i ) } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) } & { } \\ { \displaystyle } & { = \displaystyle \sum _ { i \in N } w _ { i } r _ { i } ( s , a , s ^ { \prime } ) = r ^ { w } ( s , a , s ^ { \prime } ) , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> $g \in \mathrm { A u t _ { e n v } } ( \mathcal { M } _ { w } )$ . A<sub>pp</sub>endix D <sub>g</sub>ives the com<sub>p</sub>lete <sub>proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 6<sub>.</sub>1<sub>.</sub>

Equality of weights within structural agent types. Theorem 6.1 <sub>g</sub><sup>i</sup>ves $w _ { \sigma _ { g } ( i ) } = w _ { i }$ <sup>f</sup>or ever<sub>y</sub> $i \in N$ <sub>as</sub> <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on</sub> f<sub>or</sub> <sub>a</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>�</sub> of M to be inherited b<sub>y</sub> $M _ { w } .$ . Re-<sub>q</sub>uirin<sub>g</sub> this condition for ever<sub>y</sub> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ is e<sub>qu</sub>i<sub>v</sub>alent to assi<sub>g</sub>nin<sub>g</sub> e<sub>q</sub>ual wei<sub>g</sub>hts to a<sub>g</sub>ents in the same structural a<sub>g</sub>ent t<sub>yp</sub>e. It does not re<sub>q</sub>uire e<sub>q</sub>ual wei<sub>g</sub>hts across diferent t<sub>yp</sub>es<sub>,</sub> so all a<sub>g</sub>ents must have e<sub>q</sub>ual wei<sub>g</sub>hts onl<sub>y</sub> when M is structurall<sub>y</sub> homo<sub>g</sub>eneous.

If $w _ { i } = w _ { j }$ <sup>f</sup>or agents � an<sup>d</sup> � o<sup>f</sup> t<sup>h</sup>e same structura<sup>l</sup> type in $M ,$ ever<sub>y</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of M is inherited b<sub>y</sub> $M _ { w } ,$ so $G _ { \mathcal { M } } ^ { \mathrm { e v a l } } \overset { \mathrm { - } } { \leq } G _ { \mathcal { M } _ { w } } ^ { \mathrm { e v a l } }$ . Conse<sub>q</sub>uentl<sub>y,</sub> we <sub>g</sub>et the followin<sub>g</sub> <sub>p</sub>artition refinement: $\mathcal { P } ( G _ { M } ^ { \mathrm { e v a l } } ) \preceq \mathcal { P } ( G _ { M _ { w } } ^ { \mathrm { e v a l } } )$ . Because ⪯ means finer than or e<sub>q</sub>ual to, ever<sub>y</sub> structural a<sub>g</sub>ent t<sub>yp</sub>e of M is contained in a structural a<sub>g</sub>ent <sup>t</sup>yp<sup>e of</sup> $M _ { w }$ . <sup>Th</sup>us, t<sup>h</sup>e a<sub>gg</sub>re<sub>g</sub>ate common rewar<sup>d</sup> cannot se<sub>p</sub>arate a<sub>g</sub>ents that share a t<sub>yp</sub>e in M. Rather, it can onl<sub>y p</sub>reserve that t<sub>yp</sub>e or mer<sub>g</sub>e it with other t<sub>yp</sub>es. Under this condition<sub>,</sub> $M _ { w }$ h<sub>as</sub> no more structural a<sub>g</sub>ent t<sub>yp</sub>es than M. Thus, if M is structurall<sub>y</sub> h<sub>omogeneous,</sub> th<sub>en</sub> $M _ { w }$ <sub>rema</sub>i<sub>ns</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y</sub> h<sub>omogeneous.</sub>

I<sup>f</sup>some agents � an<sup>d</sup> � <sup>b</sup>e<sup>l</sup>onging to t<sup>h</sup>e same structura<sup>l</sup> agent type in M receive diferent wei<sub>g</sub>hts, Theorem 6.1 no lon<sub>g</sub>er <sub>g</sub>uarantees that a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of M ma<sub>pp</sub>in<sub>g</sub> � to � remains f<sub>or</sub> $M _ { w } .$ If no <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of $M _ { w }$ maps � to �, th<sub>en</sub> th<sub>ese</sub> <sub>agen</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ong</sub> t<sub>o</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> i<sub>n</sub> $M _ { w } .$ Case 1 in the <sub>p</sub>roof of Pro<sub>p</sub>osition D.3 in A<sub>pp</sub>endix D <sub>p</sub>ro<sub>v</sub>ides a two-a<sub>g</sub>ent exam<sub>p</sub>le of this s<sub>p</sub>littin<sub>g</sub>.

<sup>C</sup>onverse<sup>l</sup><sub>y</sub>, rewar<sup>d</sup> a<sub>gg</sub>re<sub>g</sub>at<sup>i</sup>on can turn a t<sub>yp</sub>e-com<sub>p</sub>at<sup>ibl</sup>e transformation <sub>�</sub> that is not a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of M i<sub>n</sub>t<sub>o</sub> <sub>one</sub> <sub>o</sub>f $M _ { w } .$ <sup>I</sup>ts a<sub>g</sub>ent <sub>p</sub>ermutat<sup>i</sup>on $\sigma _ { g }$ <sup>ma</sup>y <sup>then</sup> <sup>ma</sup>p <sup>a</sup>g<sup>ents</sup> belon<sub>g</sub>in<sub>g</sub> to diferent structural a<sub>g</sub>ent t<sub>yp</sub>es in M into the same <sup>structural</sup> <sup>a</sup>g<sup>ent</sup> <sup>t</sup>yp<sup>e</sup> <sup>in</sup> $M _ { w }$ . The two-a<sub>g</sub>ent exam<sub>p</sub>les in Cases 2 and 3 of the <sub>p</sub>roof of Pro<sub>p</sub>osition D.3 demonstrate this mer<sub>g</sub>in<sub>g</sub> with e<sub>q</sub>ual and une<sub>q</sub>ual wei<sub>g</sub>hts<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>.

Therefore<sub>,</sub> e<sub>q</sub>ualit<sub>y</sub> of wei<sub>g</sub>hts within ever<sub>y</sub> structural a<sub>g</sub>ent t<sub>yp</sub>e in M <sub>g</sub>uarantees that $\mathcal { P } ( G _ { M } ^ { \mathrm { e v a l } } ) \preceq \mathcal { P } ( G _ { M _ { w } } ^ { \mathrm { e v a l } } )$ <sub>.</sub> A<sub>cross unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>we</sub>i<sub>g</sub>ht <sub>ass</sub>i<sub>gnmen</sub>t<sub>s,</sub> h<sub>owever,</sub> th<sub>ere</sub> i<sub>s no un</sub>i<sub>versa</sub>l <sub>re</sub>fi<sub>nemen</sub>t relation between these two <sub>p</sub>artitions.

Consequences under the initial distribution. Because the two models share the same initial distribution<sub>,</sub> substitutin<sub>g</sub> the value identit<sub>y</sub> f<sub>rom</sub> Th<sub>eorem</sub> 6<sub>.</sub>1 i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f <sub>expec</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn un</sub>d<sub>er</sub> th<sub>e</sub> initial distribution in E<sub>q</sub>uation (2) <sub>g</sub>ives the followin<sub>g</sub> identit<sub>y</sub> without an<sub>y</sub> s<sub>y</sub>mmetr<sub>y</sub> assum<sub>p</sub>tion (see Pro<sub>p</sub>osition D.2):

$$
J _ { j } ^ { M _ { w } } ( \pi ) = \sum _ { i \in N } w _ { i } J _ { i } ^ { M } ( \pi ) .
$$

Let $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>preserve</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s.</sub> Th<sub>eorem</sub> 6<sub>.</sub>1 <sub>ensures</sub> th<sub>a</sub>t $g \in$ $G _ { \mathcal { M } _ { w } } ^ { \mathrm { e v a l } }$ <sub>.</sub> If <sub>�</sub> <sub>a</sub>l<sub>so</sub> <sub>p</sub>r<sub>ese</sub>r<sub>ves</sub> th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>,</sub> Pr<sub>opos</sub>iti<sub>o</sub>n D<sub>.</sub>2 <sub>g</sub>ives the followin<sub>g</sub> ex<sub>p</sub>ected-return identit<sub>y</sub>:

$$
J _ { \sigma _ { g } ( j ) } ^ { \mathcal { M } _ { w } } ( g \cdot \pi ) = J _ { j } ^ { \mathcal { M } _ { w } } ( \pi ) \qquad \mathrm { f o r ~ e v e r y ~ } j \in N .
$$

Fi<sub>g</sub>ure 1 summarizes the hierarch<sub>y</sub> of ke<sub>y</sub> conce<sub>p</sub>ts and results intr<sub>o</sub>d<sub>uce</sub>d fr<sub>o</sub>m S<sub>ec</sub>ti<sub>o</sub>n 4 t<sub>o</sub> S<sub>ec</sub>ti<sub>o</sub>n 6<sub>.</sub>

7 Comparisons to Prior Symmetry Frameworks We com<sub>p</sub>are two <sub>p</sub>rior frameworks with <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y,</sub> the details of which are <sub>g</sub>iven in A<sub>pp</sub>endix E.

Counting and Isomorphic Symmetry. In Braun et al. [7]’s partiti<sub>o</sub>n<sub>e</sub>d D<sub>ec</sub>-POMDP fr<sub>a</sub>m<sub>ewo</sub>rk<sub>,</sub> i<sub>so</sub>m<sub>o</sub>r<sub>p</sub>hi<sub>c sy</sub>mm<sub>e</sub>tr<sub>y uses</sub> id<sub>e</sub>ntical factor tem<sub>p</sub>lates for a<sub>g</sub>ents in each <sub>p</sub>artition <sub>p</sub>art. Under the <sub>g</sub>roundin<sub>g</sub> assum<sub>p</sub>tions in Section E.1.1<sub>,</sub> exchan<sub>g</sub>in<sub>g</sub> a<sub>g</sub>ents within a <sub>p</sub>art to<sub>g</sub>ether with their action and observation assi<sub>g</sub>nments <sub>p</sub>re-<sub>serves</sub> th<sub>e coun</sub>t<sub>s</sub> i<sub>n</sub> th<sub>a</sub>t <sub>par</sub>t <sub>an</sub>d <sub>on</sub>l<sub>y reor</sub>d<sub>ers equa</sub>l f<sub>ac</sub>t<sub>or</sub> i<sub>n-</sub> <sub>s</sub>t<sub>ances.</sub> Th<sub>e</sub> <sub>groun</sub>d<sub>e</sub>d t<sub>rans</sub>iti<sub>on,</sub> <sub>common-rewar</sub>d<sub>,</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> functions are therefore unchan<sub>g</sub>ed<sub>,</sub> which <sub>g</sub>ives countin<sub>g</sub> s<sub>y</sub>mmet<sub>ry.</sub> Wh<sub>en</sub> th<sub>ese</sub> <sub>exc</sub>h<sub>anges</sub> <sub>are</sub> <sub>represen</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>ype</sub>d <sub>em</sub>b<sub>e</sub>ddi<sub>ng spec</sub>ifi<sub>e</sub>d th<sub>ere,</sub> th<sub>ey are componen</sub>t<sub>w</sub>i<sub>se env</sub>i<sub>ronmen</sub>t a<sub>u</sub>tomor<sub>p</sub>hisms and hence <sub>p</sub>olic<sub>y</sub>-e<sub>v</sub>al<sub>u</sub>ation s<sub>y</sub>mmetries. The lifted <sub>p</sub>artition therefore refines the structural a<sub>g</sub>ent <sub>p</sub>artition. However<sub>,</sub> the two-a<sub>g</sub>ent coordination exam<sub>p</sub>le we showed in A<sub>pp</sub>endix E.1.1 h<sub>as</sub> <sub>coun</sub>ti<sub>ng</sub> <sub>an</sub>d <sub>po</sub>li<sub>cy-eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>symme</sub>t<sub>ry</sub> b<sub>u</sub>t l<sub>ac</sub>k<sub>s</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d re<sub>p</sub>eated additive reward factorization for the same <sub>p</sub>artition with one <sub>p</sub>art. Polic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> alone does not <sub>g</sub>uarantee this re<sub>p</sub>eated additive factorization. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Braun et al. [7]’s com<sub>p</sub>actre<sub>p</sub>resentation and a<sub>g</sub>ent-number tractabilit<sub>y</sub> results re<sub>q</sub>uires their additional factorization and com<sub>p</sub>utational assum<sub>p</sub>tions.

Expected Return Symmetry. An expected return symmetry (ERS) <sub>preserves</sub> th<sub>e se</sub>lf<sub>-p</sub>l<sub>ay-op</sub>ti<sub>ma</sub>l B<sub>o</sub>lt<sub>zmann po</sub>li<sub>cy se</sub>t <sub>a</sub>t <sub>a</sub> fi<sub>xe</sub>d <sub>p</sub>ositive tem<sub>p</sub>erature [22], whereas <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>p</sub>reserves com<sub>p</sub>lete evaluation for ever<sub>y</sub> decentralized <sub>p</sub>olic<sub>y</sub>. On th<sub>e s</sub>h<sub>are</sub>d fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon</sub> D<sub>ec-</sub>POMDP <sub>su</sub>b<sub>c</sub>l<sub>ass an</sub>d <sub>common a</sub>d<sub>-</sub> mi<sub>ss</sub>ibl<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n<sub>s,</sub> Pr<sub>opos</sub>iti<sub>o</sub>n E<sub>.</sub>3 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t n<sub>e</sub>ith<sub>e</sub>r ERS n<sub>o</sub>r initi<sub>a</sub>l-di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n-<sub>p</sub>r<sub>ese</sub>r<sub>v</sub>in<sub>g</sub> <sub>po</sub>li<sub>cy</sub>-<sub>eva</sub>l<sub>ua</sub>ti<sub>o</sub>n <sub>sy</sub>mm<sub>e</sub>- tr<sub>y</sub> im<sub>p</sub>lies the other in <sub>g</sub>eneral. The two non-im<sub>p</sub>lications are <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> dif<sub>eren</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> F<sub>or</sub> <sub>a</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> th<sub>e</sub> t<sub>wo</sub> s<sub>y</sub>mmetr<sub>y</sub> sets ma<sub>y</sub> coincide or one ma<sub>y</sub> contain the other. Secti<sub>on</sub> E<sub>.</sub>2<sub>.</sub>1 <sub>presen</sub>t<sub>s</sub> th<sub>e</sub> t<sub>wo coun</sub>t<sub>erexamp</sub>l<sub>es as ev</sub>id<sub>ence</sub> f<sub>or</sub> th<sub>ese</sub> <sub>non-</sub>i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons an</sub>d <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>e a</sub>dditi<sub>ona</sub>l l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-va</sub>l<sub>ue con-</sub> dition linkin<sub>g</sub> the notions. ERS fixes a<sub>g</sub>ent indices<sub>,</sub> so it does not <sub>re</sub>l<sub>a</sub>t<sub>e</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>agen</sub>t<sub>s</sub> <sub>or</sub> <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> our a<sub>g</sub>ent orbits.

## 8 Discussions and Framework Boundaries

Why does the initial distribution matter? The initial distribution s<sub>p</sub>ecifies how <sub>p</sub>ossible startin<sub>g</sub> states are wei<sub>g</sub>hted when an e<sub>p</sub>isode b<sub>eg</sub>in<sub>s.</sub> It i<sub>s</sub> th<sub>e</sub>r<sub>e</sub>f<sub>o</sub>r<sub>e pa</sub>rt <sub>o</sub>f <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd POSG <sub>a</sub>nd D<sub>ec</sub>-POMDP f<sub>o</sub>rmulations [4, 12]. A related se<sub>p</sub>aration a<sub>pp</sub>ears in PlaNet’s latent <sub>s</sub>t<sub>a</sub>t<sub>e-space</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> <sub>an</sub> i<sub>n</sub>iti<sub>a</sub>l l<sub>a</sub>t<sub>en</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on,</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>rewar</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>use</sub>d t<sub>o</sub> <sub>genera</sub>t<sub>e</sub> <sub>a</sub> rollout [11]. Structural causal models ofer a useful motivation: their str<sub>u</sub>ct<sub>u</sub>ral e<sub>qu</sub>ations describe transition f<sub>u</sub>nctions<sub>, w</sub>hile a distri bution over exo<sub>g</sub>enous variables of structural e<sub>q</sub>uations re<sub>p</sub>resent b<sub>ac</sub>k<sub>groun</sub>d f<sub>ac</sub>t<sub>ors</sub> th<sub>a</sub>t i<sub>mp</sub>l<sub>y su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>ve c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs a</sub>f<sub>ec</sub>ti<sub>ng</sub> behavior [24]. Here, the initial distribution is corres<sub>p</sub>ondin<sub>g</sub> to the exo<sub>g</sub>eno<sub>u</sub>s distrib<sub>u</sub>tion.

O<sub>ur</sub> f<sub>ramewor</sub>k th<sub>ere</sub>f<sub>ore</sub> <sub>separa</sub>t<sub>es</sub> <sub>po</sub>li<sub>cy</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>con</sub>di<sub>-</sub> tione<sup>d</sup> on a state–joint-<sup>h</sup>istory input <sup>f</sup>rom t<sup>h</sup>e initia<sup>l d</sup>istri<sup>b</sup>ution. Polic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> com<sub>p</sub>ares values at ever<sub>y</sub> corres<sub>p</sub>onding state–joint-<sup>h</sup>istory pair an<sup>d</sup> t<sup>h</sup>ere<sup>f</sup>ore <sup>d</sup>oes not necessari<sup>l</sup>y re-<sub>q</sub>uire <sub>p</sub>reservation of $\rho _ { 0 } .$ . <sup>F</sup>or an<sub>y</sub> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>,</sub> chan<sub>g</sub>in<sub>g</sub> variables in the initial-state sum <sub>g</sub>ives the followin<sub>g</sub> relation:

$$
J _ { \sigma _ { g } ( i ) } ( g \cdot \pi ) = \sum _ { s \in S } \rho _ { 0 } \bigl ( \phi _ { g } ^ { S } ( s ) \bigr ) V _ { i , 0 } ^ { \pi } ( s , ( ) ^ { N } ) .
$$

Th<sub>us,</sub> th<sub>e</sub> r<sub>e</sub>t<sub>u</sub>rn-<sub>equ</sub>i<sub>va</sub>ri<sub>a</sub>n<sub>ce</sub> id<sub>e</sub>ntit<sub>y</sub> <sub>u</sub>nd<sub>e</sub>r th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> th<sub>e</sub> fi<sub>xe</sub>d $\rho _ { 0 }$ <sub>w</sub>h<sub>en</sub> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } }$

Why go beyond componentwise environment automorphisms? Com<sub>p</sub>onentwise s<sub>y</sub>mmetries have concrete uses. Gra<sub>p</sub>h automor-<sub>p</sub>hisms of POMDPs and POSGs su<sub>pp</sub>ort model reduction [17], Dec-POMDP e<sub>q</sub>uivalence ma<sub>pp</sub>in<sub>g</sub>s su<sub>pp</sub>ort other-<sub>p</sub>la<sub>y</sub> across conventions [14], and Dec-POMDP automor<sub>p</sub>hisms formalize transformations relevant to zero-shot coordination [30]. These a<sub>pp</sub>roaches <sub>p</sub>re-<sub>serve</sub> th<sub>e</sub> t<sub>rans</sub>iti<sub>on,</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>rewar</sub>d f<sub>unc</sub>ti<sub>ons</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y.</sub> The resultin<sub>g</sub> com<sub>p</sub>onentwise environment automor<sub>p</sub>hisms <sub>p</sub>ro-<sub>v</sub>id<sub>e</sub> di<sub>rec</sub>tl<sub>y c</sub>h<sub>ec</sub>k<sub>a</sub>bl<sub>e su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>ons</sub> f<sub>or po</sub>li<sub>cy-eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>symme</sub>t<sub>ry,</sub> b<sub>u</sub>t th<sub>ese con</sub>diti<sub>ons nee</sub>d <sub>no</sub>t b<sub>e necessary.</sub>

Pro<sub>p</sub>osition 5.3 <sub>g</sub>i<sub>v</sub>es the concrete se<sub>p</sub>aration. Its a<sub>g</sub>ent transform<sub>a</sub>ti<sub>o</sub>n <sub>c</sub>h<sub>a</sub>n<sub>ges so</sub>m<sub>e e</sub>ntri<sub>es o</sub>f th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>, so</sub> it f<sub>a</sub>il<sub>s</sub> th<sub>e componen</sub>t<sub>w</sub>i<sub>se</sub> t<sub>es</sub>t<sub>.</sub> Th<sub>e c</sub>h<sub>ange</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons never</sub>th<sub>e</sub>l<sub>ess</sub> h<sub>ave</sub> th<sub>e same expec</sub>t<sub>e</sub>d <sub>successor-s</sub>t<sub>a</sub>t<sub>e rewar</sub>d <sub>an</sub>d <sub>marg</sub>i<sub>na</sub>li<sub>ze</sub>d next-observation distribution<sub>,</sub> which are the <sub>q</sub>uantities used b<sub>y</sub> the Bellman e<sub>v</sub>al<sub>u</sub>ation in that constr<sub>u</sub>ction. The <sub>p</sub>olic<sub>y</sub>-e<sub>v</sub>al<sub>u</sub>ation id<sub>en</sub>tit<sub>y</sub> th<sub>ere</sub>f<sub>ore</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy.</sub> T<sub>wo agen</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub> <sub>a</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ype</sub> if <sub>an</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>con-</sub> nects t<sup>h</sup>em an<sup>d</sup> preserves po<sup>l</sup>icy eva<sup>l</sup>uation <sup>f</sup>or every joint po<sup>l</sup>icy, even without <sub>p</sub>reservin<sub>g</sub> each model com<sub>p</sub>onent. Conversel<sub>y,</sub> com-<sub>p</sub>onentwise <sub>p</sub>reservation remains an additional<sub>, p</sub>otentiall<sub>y</sub> finer dia<sub>g</sub>nostic of where a<sub>g</sub>ents difer in the environment re<sub>p</sub>resentation.

Learnin<sub>g</sub> mechanisms concern behavioural <sub>p</sub>olicies<sub>,</sub> so it must be <sub>cons</sub>id<sub>ere</sub>d <sub>w</sub>h<sub>en</sub> t<sub>a</sub>lki<sub>ng</sub> <sub>a</sub>b<sub>ou</sub>t <sub>agen</sub>t h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> <sub>w</sub>hi<sub>c</sub>h l<sub>e</sub>t<sub>s</sub> <sub>us</sub> introduce the <sub>p</sub>olic<sub>y</sub> e<sub>q</sub>uivariance. For exam<sub>p</sub>le<sub>,</sub> ex<sub>p</sub>loration noise <sub>w</sub>h<sub>ose</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> d<sub>oes no</sub>t <sub>respec</sub>t th<sub>e ac</sub>ti<sub>on</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on can</sub> violate the e<sub>q</sub>uivariance of an alread<sub>y</sub> e<sub>q</sub>uivariant <sub>p</sub>olic<sub>y</sub>. This will not chan<sub>g</sub>e a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y,</sub> because <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmet<sub>ry requ</sub>i<sub>res</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on</sub> id<sub>en</sub>tit<sub>y</sub> t<sub>o</sub> h<sub>o</sub>ld f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy.</sub> Thi<sub>s</sub> <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>agen</sub>t h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> d<sub>ec</sub>id<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua-</sub> tion identit<sub>y</sub> inde<sub>p</sub>endent of the learnin<sub>g</sub> mechanism that is a<sub>pp</sub>lied to obtain the tar<sub>g</sub>et <sub>p</sub>olicies (that <sub>g</sub>eniunel<sub>y</sub> describes the a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y</sub>). Thus, our <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> can ca<sub>p</sub>ture intrinsic characteristics of an a<sub>g</sub>ent to cate<sub>g</sub>orize its a<sub>g</sub>ent t<sub>yp</sub>e.

Why extendtheframework to Dec-POMDPs andaggregate common rewards? Dec-POMDPs are a long-standing model of cooperative se<sub>q</sub>uential decision-makin<sub>g</sub> under <sub>p</sub>artial observabilit<sub>y,</sub> in which decentralized a<sub>g</sub>ents o<sub>p</sub>timize a common reward [4]. A common <sub>rewar</sub>d <sub>removes</sub> dif<sub>erences</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> <sub>agen</sub>t<sub>s</sub>’ <sub>rewar</sub>d f<sub>unc</sub>ti<sub>ons,</sub> b<sub>u</sub>t it d<sub>oes</sub> <sub>no</sub>t <sub>guaran</sub>t<sub>ee</sub> i<sub>n</sub>t<sub>erc</sub>h<sub>angea</sub>bl<sub>e</sub> <sub>ac</sub>ti<sub>on</sub> <sub>or</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> s<sub>p</sub>aces<sub>,</sub> or s<sub>y</sub>mmetric roles in the transition and obser<sub>v</sub>ation distri-<sup>b</sup>utions. Section 6 t<sup>h</sup>ere<sup>f</sup>ore s<sup>h</sup>ows t<sup>h</sup>at s<sup>h</sup>are<sup>d</sup> o<sup>b</sup>jectives nee<sup>d</sup> not im<sub>p</sub>l<sub>y</sub> structural homo<sub>g</sub>eneit<sub>y</sub>. This extension hel<sub>p</sub>s connect the o<sup>b</sup>jective to p<sup>h</sup>ysica<sup>l</sup> an<sup>d</sup> environmenta<sup>l</sup> perspectives, as <sup>d</sup>ei<sup>f</sup>ne<sup>d</sup> <sup>b</sup>y <sub>p</sub>rior work [5], b<sub>y</sub> assessin<sub>g</sub> their structural si<sub>g</sub>nificance throu<sub>g</sub>h one <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> criterion.

Social dilemmas [10] illustrate how individuall<sub>y</sub> rational behavi<sub>or can</sub> l<sub>ea</sub>d t<sub>o co</sub>ll<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> i<sub>n</sub>f<sub>er</sub>i<sub>or ou</sub>t<sub>comes, even w</sub>h<sub>en coopera-</sub> ti<sub>on cou</sub>ld b<sub>ene</sub>fit <sub>every p</sub>l<sub>ayer.</sub> Thi<sub>s mo</sub>ti<sub>va</sub>t<sub>es mec</sub>h<sub>an</sub>i<sub>sms</sub> th<sub>a</sub>t su<sub>pp</sub>ort coo<sub>p</sub>eration, with social welfare [25] <sub>p</sub>rovidin<sub>g</sub> a criterion f<sub>or eva</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e resu</sub>lti<sub>ng co</sub>ll<sub>ec</sub>ti<sub>ve</sub> b<sub>ene</sub>fit<sub>s.</sub> S<sub>oc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e sum o</sub>f i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>payo</sub>f<sub>s, w</sub>hi<sub>c</sub>h <sub>we</sub>ll <sub>ma</sub>t<sub>c</sub>h<sub>es our</sub> definition of the a<sub>gg</sub>re<sub>g</sub>ate common reward in Section 6. Theo-<sub>rem</sub> 6<sub>.</sub>1 <sub>an</sub>d it<sub>s examp</sub>l<sub>es s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>rewar</sub>d<sub>s</sub> <sup>b</sup>y <sup>an a</sup>gg<sup>re</sup>g<sup>ate common reward can</sup> p<sup>reserve</sup>, <sup>remove</sup>, <sup>or create</sup> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries. This <sub>p</sub>rovides a novel <sub>p</sub>ers<sub>p</sub>ective from structural a<sub>g</sub>ent t<sub>yp</sub>es<sub>,</sub> in contrast to <sub>p</sub>revious work that identifies <sub>g</sub>ame structures [20], to understand the efect of maximizin<sub>g</sub> <sub>soc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are</sub> t<sub>o soc</sub>i<sub>a</sub>l dil<sub>emmas.</sub>

In <sub>coope</sub>r<sub>a</sub>ti<sub>ve</sub> MARL f<sub>o</sub>r <sub>so</sub>l<sub>v</sub>in<sub>g</sub> D<sub>ec</sub>-POMDP<sub>s,</sub> th<sub>e ce</sub>ntr<sub>a</sub>li<sub>se</sub>d trainin<sub>g</sub> and decentralised execution (CTDE) <sub>p</sub>aradi<sub>g</sub>m [23] is <sub>p</sub>ro-<sub>p</sub>osed to collectivel<sub>y</sub> o<sub>p</sub>timize a <sub>g</sub>lobal value function that a<sub>pp</sub>roci-<sub>ma</sub>t<sub>es</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve common rewar</sub>d<sub>s.</sub> Th<sub>ere are a c</sub>l<sub>ass o</sub>f <sub>a</sub>l<sub>-</sub> <sub>gor</sub>ith<sub>ms</sub> th<sub>a</sub>t <sub>propose</sub>d t<sub>o</sub> <sub>represen</sub>t th<sub>e</sub> <sub>g</sub>l<sub>oa</sub>bl <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>as</sub> a<sub>gg</sub>re<sub>g</sub>ation of individual values<sub>,</sub> e.<sub>g</sub>.<sub>,</sub> VDN uses additive decom<sub>p</sub>osition [28]; SQDDPG and SHAQ use Shapley-based value decompositions [34, 35]; and QTRAN learns an additive transformation of the joint action-value function that <sub>p</sub>reserves its o<sub>p</sub>timal actions [27]. A<sub>s</sub> th<sub>ose</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d th<sub>e</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>para</sub>di<sub>gm</sub> <sub>are</sub> <sub>we</sub>ll <sub>a</sub>li<sub>gne</sub>d wit<sup>h</sup> eac<sup>h</sup> agent � optimizing $\begin{array} { r } { V _ { j , t } ^ { \pi ; \mathcal { M } _ { w } } ( x ) = \sum _ { i \in N } w _ { i } V _ { i , t } ^ { \pi ; \mathcal { M } } ( x ) } \end{array}$ <sub>,</sub> Th<sub>e-</sub> orem 6.1 motivates a testable hypothesis: do these algorithms implicitly learn structural agent types, which could be an inductive bias for improving cooperation?

Why is one connecting symmetry suficient? We use every ele <sub>men</sub>t <sub>o</sub>f $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ to <sup>f</sup>orm or<sup>b</sup>its: two agents � an<sup>d</sup> � <sup>b</sup>e<sup>l</sup>ong to t<sup>h</sup>e same <sup>structural</sup> <sup>a</sup>g<sup>ent</sup> <sup>t</sup>yp<sup>e</sup> <sup>when</sup> <sup>some</sup> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\sigma _ { g } ( i ) = j .$ F<sub>o</sub>r <sub>examp</sub>l<sub>e,</sub> l<sub>e</sub>t $N = \{ 1 , 2 , 3 \}$ <sub>an</sub>d <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>agen</sub>t <sub>permu-</sub> tations $\sigma _ { g }$ th<sub>a</sub>t f<sub>u</sub>lfil th<sub>e</sub> <sub>cyc</sub>li<sub>c</sub> <sub>permu</sub>t<sub>a</sub>ti<sub>on</sub> $1 \mapsto 2 \mapsto 3 \mapsto$ 1. Re<sub>p</sub>eated a<sub>pp</sub>lications of these <sub>p</sub>ermutations can ma<sub>p</sub> an<sub>y</sub> a<sub>g</sub>ent to an<sub>y</sub> other<sub>,</sub> so the sub<sub>g</sub>rou<sub>p</sub> it <sub>g</sub>enerates acts transitivel<sub>y</sub> on �.

This sub<sub>g</sub>rou<sub>p</sub> therefore induces the same sin<sub>g</sub>le-<sub>p</sub>art orbit <sub>p</sub>artition $\{ N \}$ as $\operatorname { S y m } ( N )$ <sub>, a</sub>lth<sub>oug</sub>h it <sub>con</sub>t<sub>a</sub>i<sub>ns on</sub>l<sub>y</sub> th<sub>ree o</sub>f th<sub>e s</sub>i<sub>x</sub> <sub>p</sub>ossible a<sub>g</sub>ent <sub>p</sub>ermutations, as contained in S<sub>y</sub>m(�). In contrast, a diferent and much stron<sub>g</sub>er condition is that ever<sub>y</sub> a<sub>g</sub>ent <sub>p</sub>ermutation $\xi \in \mathsf { S y m } ( N )$ b<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y some</sub> $g \in G _ { M } ^ { \mathrm { e v a l } }$ satisf<sub>y</sub>in<sub>g</sub> $\sigma _ { g } = \xi .$ This full-<sub>p</sub>ermutation condition is the d<sub>y</sub>namic analo<sub>g</sub>ue of name-irrelevant s<sub>y</sub>mmetr<sub>y</sub> in normal-form <sub>g</sub>ames [8]. It is stron<sub>g</sub>er th<sub>an s</sub>t<sub>ruc</sub>t<sub>ura</sub>l h<sub>omogene</sub>it<sub>y, as we</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub> A<sub>s ex-</sub> <sub>p</sub>lained after Definition 4.2<sub>,</sub> structural homo<sub>g</sub>eneit<sub>y</sub> re<sub>q</sub>uires onl<sub>y</sub> that<sub>,</sub> for ever<sub>y</sub> <sub>p</sub>air $i , j \in N$ , some $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\sigma _ { g } ( i ) = j$ <sub>.</sub> Th<sub>e</sub> <sup>i</sup>n<sup>d</sup>uce<sup>d</sup> a<sub>g</sub>ent-<sub>p</sub>ermutat<sup>i</sup>on <sub>g</sub>rou<sub>p</sub> $\{ \sigma _ { g } : g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } } \}$ <sub>may</sub> th<sub>ere</sub>f<sub>ore</sub> be a <sub>p</sub>ro<sub>p</sub>er sub<sub>g</sub>rou<sub>p</sub> of S<sub>y</sub>m(�). B<sub>y</sub> the definition of stuctural <sub>p</sub>artition, the structural a<sub>g</sub>ent t<sub>yp</sub>es of S<sub>y</sub>m(�) ma<sub>y</sub> overclaim the <sub>exac</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> th<sub>a</sub>t <sub>can</sub> b<sub>e o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$

The above conclusion is im<sub>p</sub>ortant in a<sub>pp</sub>lications. Su<sub>pp</sub>ose the <sub>same s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c mo</sub>d<sub>e</sub>l <sub>governs sensor</sub> di<sub>s</sub>t<sub>ur</sub>b<sub>ances or p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es</sub> f<sub>or symme</sub>t<sub>r</sub>i<sub>c agen</sub>t<sub>s.</sub> Th<sub>ese</sub> f<sub>ac</sub>t<sub>ors can</sub> b<sub>e</sub> d<sub>escr</sub>ib<sub>e</sub>d b<sub>y</sub> s<sub>y</sub>mmetries not included in $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sup>l</sup><sub>, an</sub>d dif<sub>eren</sub>t <sub>rea</sub>li<sub>ze</sub>d <sub>ou</sub>t<sub>comes</sub> b<sub>y</sub> th<sub>ose</sub> <sub>ex</sub>t<sub>ra</sub> <sub>symme</sub>t<sub>r</sub>i<sub>es</sub> d<sub>o</sub> <sub>no</sub>t b<sub>y</sub> th<sub>emse</sub>l<sub>ves</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>c</sub>h<sub>ange</sub> the a<sub>g</sub>ent structural t<sub>yp</sub>es. However<sub>,</sub> se<sub>p</sub>aratin<sub>g</sub> stable structures from tem<sub>p</sub>orar<sub>y</sub> <sub>p</sub>erturbations<sub>,</sub> or e<sub>q</sub>uivalentl<sub>y,</sub> determine a <sub>p</sub>ro<sub>p</sub>er sub<sub>g</sub>rou<sub>p</sub> of S<sub>y</sub>m(�), would re<sub>q</sub>uire dee<sub>p</sub>er investi<sub>g</sub>ation.

Framework boundaries. The present framework assumes finite state<sub>,</sub> action<sub>,</sub> and observation s<sub>p</sub>aces<sub>,</sub> a finite horizon<sub>,</sub> em<sub>p</sub>t<sub>y</sub> initial hi<sub>s</sub>t<sub>or</sub>i<sub>es, exac</sub>t <sub>expec</sub>t<sub>e</sub>d<sub>-va</sub>l<sub>ue eva</sub>l<sub>ua</sub>ti<sub>on, an</sub>d th<sub>e comp</sub>l<sub>e</sub>t<sub>e</sub> d<sub>o-</sub> <sub>ma</sub>i<sub>n o</sub>f d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>pro</sub>d<sub>uc</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es w</sub>ith<sub>ou</sub>t <sub>s</sub>h<sub>are</sub>d randomness. Its str<sub>u</sub>ct<sub>u</sub>ral t<sub>yp</sub>es are relative to the s<sub>p</sub>ecified POSG<sub>,</sub> th<sub>e c</sub>h<sub>osen eva</sub>l<sub>ua</sub>ti<sub>on seman</sub>ti<sub>cs, an</sub>d th<sub>e</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans-</sub> formation <sub>g</sub>rou<sub>p</sub>. Because $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ i<sub>s</sub> d<sub>ec</sub>l<sub>are</sub>d b<sub>y</sub> th<sub>e mo</sub>d<sub>e</sub>ll<sub>er,</sub> can<sup>d</sup>i<sup>d</sup>ate trans<sup>f</sup>ormations must <sup>b</sup>e justi<sup>fi</sup>e<sup>d</sup> in<sup>d</sup>epen<sup>d</sup>ent<sup>l</sup>y o<sup>f</sup> t<sup>h</sup>e desired hetero<sub>g</sub>eneit<sub>y</sub> conclusion. Restrictin<sub>g</sub> this <sub>g</sub>rou<sub>p</sub> can refine the structural <sub>p</sub>artition<sub>,</sub> while enlar<sub>g</sub>in<sub>g</sub> it can coarsen the <sub>p</sub>artition.

W<sub>e prov</sub>id<sub>e su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>ons an</sub>d fi<sub>n</sub>it<sub>e examp</sub>l<sub>es</sub> t<sub>o c</sub>l<sub>ar</sub>if<sub>y</sub> th<sub>e</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> and the resultin<sub>g</sub> a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y,</sub> but no <sub>g</sub>eneral s<sub>y</sub>mmetr<sub>y</sub>-discover<sub>y</sub> or verification al<sub>g</sub>orithm is introduced. Also<sub>,</sub> exact orbits <sub>g</sub>iven b<sub>y p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries d<sub>o no</sub>t <sub>measure</sub> d<sub>egrees o</sub>f h<sub>e</sub>t<sub>erogene</sub>it<sub>y.</sub>

I<sub>n</sub>fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon mo</sub>d<sub>e</sub>l<sub>s an</sub>d <sub>non</sub>li<sub>near or</sub> l<sub>earne</sub>d <sub>rewar</sub>d <sub>aggre-</sub> <sub>g</sub>ation re<sub>q</sub>uire se<sub>p</sub>arate formulations. The wei<sub>g</sub>hted construction is <sub>a</sub> f<sub>orwar</sub>d <sub>resu</sub>lt f<sub>or</sub> fi<sub>xe</sub>d li<sub>near we</sub>i<sub>g</sub>ht<sub>s.</sub> It d<sub>oes no</sub>t <sub>recover</sub> i<sub>n</sub>di<sub>-</sub> <sub>v</sub>id<sub>ua</sub>l <sub>rewar</sub>d<sub>s</sub> f<sub>rom</sub> <sub>a</sub> <sub>supp</sub>li<sub>e</sub>d <sub>common</sub> <sub>rewar</sub>d <sub>or</sub> <sub>cover</sub> <sub>non</sub>li<sub>near</sub> <sup>a</sup>gg<sup>re</sup>g<sup>ation.</sup>

Polic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> com<sub>p</sub>ares ex<sub>p</sub>ected values. It can th<sub>ere</sub>f<sub>ore</sub> t<sub>rea</sub>t t<sub>wo</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> <sub>as</sub> <sub>symme</sub>t<sub>r</sub>i<sub>c</sub> <sub>even</sub> <sub>w</sub>h<sub>en</sub> th<sub>ey</sub> in<sup>d</sup>uce <sup>d</sup>i<sup>f</sup>erent <sup>d</sup>istri<sup>b</sup>utions over trajectories or returns, provi<sup>d</sup>e<sup>d</sup> th<sub>ose</sub> dif<sub>erences</sub> d<sub>o no</sub>t <sub>c</sub>h<sub>ange</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ues.</sub> Di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ona</sub>l evaluation [2] or risk-sensitive evaluation [9] ma<sub>y</sub> further distin-<sub>gu</sub>i<sub>s</sub>h <sub>suc</sub>h t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub> Fi<sub>na</sub>ll<sub>y,</sub> <sub>our</sub> <sub>resu</sub>lt<sub>s</sub> d<sub>o</sub> <sub>no</sub>t <sub>prove</sub> th<sub>e</sub> existence of an o<sub>p</sub>timal e<sub>q</sub>uivariant <sub>p</sub>olic<sub>y,</sub> or establish re<sub>p</sub>resentation<sub>,</sub> evaluation and sam<sub>p</sub>le-com<sub>p</sub>lexit<sub>y</sub> im<sub>p</sub>rovements.

All th<sub>ose a</sub>b<sub>ove</sub> b<sub>oun</sub>d<sub>ar</sub>i<sub>es es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es c</sub>l<sub>ear rou</sub>t<sub>es</sub> f<sub>or</sub> f<sub>ur</sub>th<sub>er</sub> <sub>researc</sub>h t<sub>o</sub> i<sub>mprove</sub> <sub>our</sub> f<sub>ramewor</sub>k<sub>.</sub> B<sub>es</sub>id<sub>es,</sub> <sub>w</sub>ith <sub>proper</sub> t<sub>as</sub>k f<sub>or-</sub> <sub>ma</sub>li<sub>za</sub>ti<sub>ons, our</sub> f<sub>ramewor</sub>k <sub>may a</sub>l<sub>so</sub> i<sub>n</sub>f<sub>orm researc</sub>h <sub>on mu</sub>lti<sub>-</sub> a<sub>g</sub>ent world models [37] and lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models (LLMs) [36] b<sub>y</sub> se<sub>p</sub>aratin<sub>g</sub> structural diferences between a<sub>g</sub>ents.

## Acknowledgments

Jianhon<sub>g</sub> Wan<sub>g</sub> and Jonathan Lawr<sub>y</sub> are su<sub>pp</sub>orted b<sub>y</sub> the En<sub>g</sub>ineerin<sub>g</sub> and Ph<sub>y</sub>sical Sciences Research Council [Grant Ref: EP/Y028732/1].

## References

[1] Robert J Aumann. 1974. Subjectivit<sub>y</sub> and correlation in randomized strate<sub>g</sub>ies. Journal ofmathematical Economics 1, 1 (1974), 67–96.

[2] Marc G. Bellemare, Will Dabne<sub>y</sub>, and Rémi Munos. 2017. A Distributional Perspective on Reinforcement Learning. In Proceedings ofthe 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 (Proceedings of Machine Learning Research, Vol. 70), Doina Precup and Yee Wh<sub>y</sub>e Teh (Eds.). PMLR, 449–458.

[3] Daniel S. Bernstein, Christo<sub>p</sub>her Amato, Eric A. Hansen, and Shlomo Zilberstein. 2009<sub>.</sub> P<sub>o</sub>li<sub>cy</sub> It<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r D<sub>ece</sub>ntr<sub>a</sub>liz<sub>e</sub>d C<sub>o</sub>ntr<sub>o</sub>l <sub>o</sub>f M<sub>a</sub>rk<sub>ov</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n Pr<sub>ocesses.</sub> J. Artif. Intell. Res. 34 (2009), 89–132.

[4] Daniel S. Bernstein, Robert Givan, Neil Immerman, and Shlomo Zilberstein. 2002. The Complexity of Decentralized Control of Markov Decision Processes. Math. Oper. Res. 27, 4 (2002), 819–840.

[5] Matteo Bettini, Aja<sub>y</sub> Shankar, and Amanda Prorok. 2023. Hetero<sub>g</sub>eneous Multi Robot Reinforcement Learning. In Proceedings ofthe 2023 International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2023, London, United Kingdom, 29 May 2023 - 2 June 2023, Noa Agmon, Bo An, Alessandro Ricci, and William Yeoh (Eds.). ACM, 1485–1494.

[6] Matteo Bettini, Aja<sub>y</sub> Shankar, and Amanda Prorok. 2025. S<sub>y</sub>stem Neural Diversit<sub>y</sub>: Measuring Behavioral Heterogeneity in Multi-Agent Learning. J. Mach. Learn. Res. 26 (2025), 163:1–163:27.

[7] Tan a Braun, Marcel Gehrke, Florian Lau, and Ralf Möller. 2022. Liftin in multi-agent systems under uncertainty. In Uncertainty in Artificial Intelligence, Proceedings ofthe Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, UAI 2022, 1-5 August 2022, Eindhoven, The Netherlands (Proceedings ofMachine Learning Research, Vol. 180), James Cussens and Kun Zhang (Eds.). PMLR, 233– 243.

[8] Zhigang Cao and Xiaoguang Yang. 2018. Symmetric games revisited. Math. Soc. Sci. 95 (2018), 9–18.

[9] Yinlam Chow, Aviv Tamar, Shie Mannor, and Marco Pavone. 2015. Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, C<sub>o</sub>rinn<sub>a</sub> C<sub>o</sub>rt<sub>es,</sub> N<sub>e</sub>il D<sub>.</sub> L<sub>aw</sub>r<sub>e</sub>n<sub>ce,</sub> D<sub>a</sub>ni<sub>e</sub>l D<sub>.</sub> L<sub>ee,</sub> M<sub>asas</sub>hi S<sub>ug</sub>i<sub>ya</sub>m<sub>a, a</sub>nd R<sub>o</sub>m<sub>a</sub>n Garnett (Eds.). 1522–1530.

[10] Robyn M Dawes. 1980. Social dilemmas. Annual review of psychology (1980).

[11] Danijar Hafner, Timoth<sub>y</sub> P. Lillicra<sub>p</sub>, Ian Fischer, Ruben Ville<sub>g</sub>as, David Ha, Hon<sub>g</sub>lak Lee, and James Davidson. 2019. Learnin<sub>g</sub> Latent D<sub>y</sub>namics for Plan ning from Pixels. In Proceedings ofthe 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA (Proceedings of Machine Learning Research, Vol. 97), Kamalika Chaudhuri and Ruslan Salakhutdi nov (Eds.). PMLR, 2555–2565.

[12] Eric A. Hansen, Daniel S. Bernstein, and Shlomo Zilberstein. 2004. D<sub>y</sub>namic Programming for Partially Observable Stochastic Games. In Proceedings ofthe Nineteenth National Conference on Artificial Intelligence, Sixteenth Conference on Innovative Applications ofArtificial Intelligence, July 25-29, 2004, San Jose, California, USA, Deborah L. McGuinness and George Ferguson (Eds.). AAAI P<sub>ress</sub> / Th<sub>e</sub> MIT P<sub>ress,</sub> 709<sub>–</sub>715<sub>.</sub>

[13] Jian<sub>y</sub>e Hao, Xiaotian Hao, Han<sub>gy</sub>u Mao, Weixun Wan<sub>g</sub>, Yaodon<sub>g</sub> Yan<sub>g</sub>, Don<sub>g</sub> Li<sub>,</sub> Y<sub>a</sub>n Zh<sub>e</sub>n<sub>g,</sub> <sub>a</sub>nd Zh<sub>e</sub>n W<sub>a</sub>n<sub>g.</sub> 2023<sub>.</sub> B<sub>oos</sub>tin<sub>g</sub> M<sub>u</sub>lti<sub>age</sub>nt R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g v</sub>i<sub>a</sub> P<sub>e</sub>rm<sub>u</sub>t<sub>a</sub>ti<sub>o</sub>n In<sub>va</sub>ri<sub>a</sub>nt <sub>a</sub>nd P<sub>e</sub>rm<sub>u</sub>t<sub>a</sub>ti<sub>o</sub>n E<sub>qu</sub>i<sub>va</sub>ri<sub>a</sub>nt N<sub>e</sub>t<sub>wo</sub>rk<sub>s.</sub> In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

[14] Hen<sub>gy</sub>uan Hu, Adam Lerer, Alex Pe<sub>y</sub>sakhovich, and Jakob N. Foerster. 2020. "Other-Play" for Zero-Shot Coordination. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event (Proceedings ofMachine Learning Research, Vol. 119). PMLR, 4399–4410.

[15] Tian<sub>y</sub>i Hu, Zhi<sub>q</sub>ian<sub>g</sub> Pu, Yuan Wan<sub>g</sub>, Ten<sub>g</sub>hai Qiu, Min Chen, and Xin Yu. 2026. Heterogeneity in Multi-Agent Reinforcement Learning. In Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2026, Paphos, Cyprus, May 25-29, 2026, Viviana Mascardi, John Thangarajah, Chris Amato, and Louise A. Dennis (Eds.). ACM, 2754–2762.

[16] Thomas W Judson and Robert Arnold Beezer. 1994. Abstract algebra: theory and applications. Vol. 20. PWS Publishing Company Boston.

[17] B<sub>y</sub>un<sub>g</sub> Kon Kan<sub>g</sub> and Kee-Eun<sub>g</sub> Kim. 2012. Ex<sub>p</sub>loitin<sub>g</sub> s<sub>y</sub>mmetries for sin<sub>g</sub>leand multi-agent Partially Observable Stochastic Domains. Artif. Intell. 182-183 (2012), 32–57.

[18] Nicolas Keriven and Gabriel Pe<sub>y</sub>ré. 2019. Universal Invariant and E<sub>q</sub>uivariant Graph Neural Networks. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS

2019, December 8-14, 2019, Vancouver, BC, Canada, Hanna M. Wallach, Hugo L<sub>aroc</sub>h<sub>e</sub>ll<sub>e,</sub> Ali<sub>na</sub> B<sub>e e</sub>l<sub>z</sub>i<sub>mer,</sub> Fl<sub>orence</sub> d’Al<sub>c</sub>hé<sub>-</sub>B<sub>uc,</sub> E<sub>m</sub>il B<sub>.</sub> F<sub>ox, an</sub>d R<sub>oman</sub> Garnett (Eds.). 7090–7099.

[19] Frédéric Koriche, S<sub>y</sub>lvain La<sub>g</sub>rue, <sup>É</sup>ric Piette, and Sébastien Tabar<sub>y</sub>. 2017. Constraint-Based Symmetry Detection in General Game Playing. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJ-CAI 2017, Melbourne, Australia, August 19-25, 2017, Carles Sierra (Ed.). ijcai.org, 280–287.

[20] Joel Z. Leibo, Vinícius Flores Zambaldi, Marc Lanctot, Janusz Marecki, and Thore Grae<sub>p</sub>el. 2017. M<sub>u</sub>lti-a<sub>g</sub>ent Reinforcement Learnin<sub>g</sub> in Se<sub>qu</sub>ential Social Dilemmas. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems, AAMAS 2017, São Paulo, Brazil, May 8-12, 2017, Kate Larson, Michael Winikof, Sanma<sub>y</sub> Das, and Edmund H. Durfee (Eds.). ACM, 464–473.

[21] Ferenc Molnar, Takashi Nishikawa, and Adilson E Motter. 2021. As<sub>y</sub>mmetr<sub>y</sub> underlies stability in power grids. Nature communications 12, 1 (2021), 1457.

[22] Darius Mu<sub>g</sub>lich, Johannes Forkel, Elise van der Pol, and Jakob Nicolaus Foerster. 2025. Expected Return Symmetries. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

[23] Frans A. Oliehoek, Matthijs T. J. S<sub>p</sub>aan, and Nikos Vlassis. 2011. O<sub>p</sub>timal and Approximate Q-value Functions for Decentralized POMDPs. CoRR abs/1111.0062 (2011).

[24] Judea Pearl. [n. d.]. Causal inference in statistics: An overview. Statistics Surveys 3, none ([n. d.]), 96 – 146.

[25] Tim Roughgarden and <sup>É</sup>va Tardos. 2002. How bad is selfish routing? J. ACM 49, 2 (2002), 236–259.

[26] KristofSchütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Al<sub>exan</sub>d<sub>re</sub> Tk<sub>a</sub>t<sub>c</sub>h<sub>en</sub>k<sub>o, an</sub>d Kl<sub>aus-</sub>R<sub>o</sub>b<sub>er</sub>t Müll<sub>er.</sub> 2017<sub>.</sub> S<sub>c</sub>hN<sub>e</sub>t<sub>:</sub> A <sub>con</sub>ti<sub>nuous-</sub> filter convolutional neural network for modeling quantum interactions. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, Is abelle Gu<sub>y</sub>on<sub>,</sub> Ulrike von Luxbur<sub>g,</sub> Sam<sub>y</sub> Ben<sub>g</sub>io<sub>,</sub> Hanna M. Wallach<sub>,</sub> Rob Fer<sub>g</sub>us<sub>,</sub> S. V. N. Vishwanathan, and Roman Garnett (Eds.). 991–1001.

[27] K un hwan Son, Daewoo Kim, Wan Ju Kan , David Hostallero, and Yun Yi. 2019. QTRAN: Learnin to Factorize with Transformation for Coo erative Multi Agent Reinforcement Learning. In Proceedings ofthe 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA (Proceedings of Machine Learning Research, Vol. 97), Kamalika Chaudhuri and Ruslan Salakhutdinov (Eds.). PMLR, 5887–5896.

[28] Peter Suneha<sub>g</sub>, Gu<sub>y</sub> Lever, Audrunas Grusl<sub>y</sub>s, Wojciech Marian Czarnecki, Viní- cius Flores Zambaldi, Max Jaderber<sub>g</sub>, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo<sub>,</sub> Karl T<sub>uy</sub>ls<sub>,</sub> and Thore Grae<sub>p</sub>el. 2018. Val<sub>u</sub>e-Decom<sub>p</sub>osition Net<sub>w</sub>orks For Cooperative Multi-Agent Learning Based On Team Reward. In Proceedings ofthe 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS 2018, Stockholm, Sweden, July 10-15, 2018 Elisabeth André Sven Koeni<sub>g</sub>, Mehdi Dastani, and Gita Sukthankar (Eds.). International Foundation f<sub>o</sub>r A<sub>u</sub>t<sub>o</sub>n<sub>o</sub>m<sub>ous</sub> A<sub>ge</sub>nt<sub>s a</sub>nd M<sub>u</sub>lti<sub>age</sub>nt S<sub>ys</sub>t<sub>e</sub>m<sub>s</sub> Ri<sub>c</sub>hl<sub>a</sub>nd<sub>,</sub> SC<sub>,</sub> USA / ACM<sub>,</sub> 2085–2087.

[29] Emanuel Tewolde, Brian Hu Zhan<sub>g</sub>, Cas<sub>p</sub>ar Oesterheld, Tuomas Sandholm, and Vincent Conitzer. 2025. Com<sub>pu</sub>tin<sub>g</sub> Game S<sub>y</sub>mmetries and E<sub>qu</sub>ilibria That Respect Them. In Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications ofArtificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI2025, Philadelphia, PA, USA, February 25 - March 4, 2025, Toby Walsh, Julie Shah, and Zico Kolter (Eds.). AAAI Press, 14148–14157.

[30] Johannes Treutlein, Michael Dennis, Cas<sub>p</sub>ar Oesterheld, and Jakob N. Foerster. 2021<sub>.</sub> A N<sub>ew</sub> F<sub>o</sub>rm<sub>a</sub>li<sub>s</sub>m M<sub>e</sub>th<sub>o</sub>d <sub>a</sub>nd O <sub>e</sub>n I<sub>ssues</sub> f<sub>o</sub>r Z<sub>e</sub>r<sub>o</sub>-Sh<sub>o</sub>t C<sub>oo</sub>rdin<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> In Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings ofMachine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 10413–10423.

[31] Elise van der Pol, Herke van Hoof, Frans A. Oliehoek, and Max Wellin<sub>g</sub>. 2022. Multi-Agent MDP Homomorphic Networks. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. O<sub>p</sub>enRe<sub>v</sub>ie<sub>w</sub>.net.

[32] Elise van der Pol, Daniel E. Worrall, Herke van Hoof, Frans A. Oliehoek, and M<sub>a</sub>x W<sub>e</sub>llin<sub>g.</sub> 2020<sub>.</sub> MDP H<sub>o</sub>m<sub>o</sub>m<sub>o</sub>r<sub>p</sub>hi<sub>c</sub> N<sub>e</sub>t<sub>wo</sub>rk<sub>s</sub>: Gr<sub>oup</sub> S<sub>y</sub>mm<sub>e</sub>tri<sub>es</sub> in R<sub>e</sub> inforcement Learning. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.).

[33] Jianhon<sub>g</sub> Wan<sub>g</sub>, Wan<sub>g</sub>kun Xu, Yunjie Gu, Wenbin Son<sub>g</sub>, and Tim C. Green. 2021<sub>.</sub> M<sub>u</sub>lti-A<sub>ge</sub>nt R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g</sub> f<sub>o</sub>r A<sub>c</sub>ti<sub>ve</sub> V<sub>o</sub>lt<sub>age</sub> C<sub>o</sub>ntr<sub>o</sub>l <sub>o</sub>n P<sub>owe</sub>r Distribution Networks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dau<sub>p</sub>hin, Perc<sub>y</sub> Lian<sub>g</sub>, and Jennifer Wortman Vau<sub>g</sub>han (Eds.). 3271–3284.

[34] Jianhong Wang, Yuan Zhang, Yunjie Gu, and Tae-Kyun Kim. 2022. SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.).

[35] Jianhon<sub>g</sub> Wan<sub>g</sub>, Yuan Zhan<sub>g</sub>, Tae-K<sub>y</sub>un Kim, and Yunjie Gu. 2020. Sha<sub>p</sub>le<sub>y</sub> Q-Value: A Local Reward Approach to Solve Global Reward Games. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020. AAAI Press 7285–7292.

[36] Qin<sub>gy</sub>un Wu, Ga<sub>g</sub>an Bansal, Jie<sub>y</sub>u Zhan<sub>g</sub>, Yiran Wu, Beibin Li, Erkan<sub>g</sub> Zhu, Li Jian<sub>g</sub>, Xiao<sub>y</sub>un Zhan<sub>g</sub>, Shaokun Zhan<sub>g</sub>, Jiale Liu, et al. 2023. Auto<sub>g</sub>en: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155 (2023).

[37] Yan<sub>g</sub> Zhan<sub>g</sub>, Chenjia Bai, Bin Zhao, Junchi Yan, Xiu Li, and Xuelon<sub>g</sub> Li. 2025. Decentralized Transformers with Centralized A<sub>gg</sub>re<sub>g</sub>ation are Sam<sub>p</sub>le-Eficient Multi-Agent World Models. Trans. Mach. Learn. Res. 2025 (2025).

## A Detailed Comparison of Heterogeneity Categories

Table 1 com<sub>p</sub>ares <sub>p</sub>rior cate<sub>g</sub>ories and measures of a<sub>g</sub>ent hetero-<sub>gene</sub>it<sub>y</sub> t<sub>o</sub> dif<sub>eren</sub>t l<sub>eve</sub>l<sub>s</sub> <sub>o</sub>f <sub>our</sub> f<sub>ramewor</sub>k<sub>.</sub> Th<sub>e</sub> t<sub>wo</sub> <sub>approac</sub>h<sub>es</sub> ask diferent <sub>q</sub>uestions. Existin<sub>g</sub> taxonomies <sub>p</sub>rimaril<sub>y</sub> identif<sub>y</sub> where a<sub>g</sub>ents difer<sub>,</sub> such as in their <sub>p</sub>h<sub>y</sub>sical ca<sub>p</sub>abilities<sub>,</sub> observations, <sup>d</sup>ynamics, o<sup>b</sup>jectives, or se<sup>l</sup>ecte<sup>d</sup> po<sup>l</sup>icies. Our <sup>f</sup>ramewor<sup>k</sup> i<sub>ns</sub>t<sub>ea</sub>d <sub>as</sub>k<sub>s</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>an</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>can</sub> <sub>exc</sub>h<sub>ange</sub> the a<sub>g</sub>ents while <sub>p</sub>reservin<sub>g</sub> com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation. The corre-<sub>spon</sub>d<sub>ences</sub> <sub>are</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>con</sub>diti<sub>ona</sub>l i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ences</sub> b<sub>e</sub>t<sub>ween</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons.</sub>

The rows concernin<sub>g</sub> <sub>p</sub>h<sub>y</sub>sical <sub>p</sub>ro<sub>p</sub>erties<sub>,</sub> observations<sub>,</sub> transitions, an<sup>d</sup> o<sup>b</sup>jectives s<sup>h</sup>are one <sup>l</sup>ogica<sup>l</sup> <sup>d</sup>irection. For agents $i , j \in N ,$ Theorem 5.2 <sub>g</sub>i<sub>v</sub>es

$$
\left[ \mathrm { s o m e } ~ g \in \mathrm { A u t } _ { \mathrm { e n v } } ( \mathcal { M } ) \mathrm { ~ s a t i s f i e s ~ } \sigma _ { g } ( i ) = j \right] \quad \implies \quad i \sim _ { \mathrm { e v a l ~ } } j .
$$

The converse is false in <sub>g</sub>eneral b<sub>y</sub> Pro<sub>p</sub>osition 5.3. Conse<sub>q</sub>uentl<sub>y,</sub> com<sub>p</sub>onentwise <sub>p</sub>reservation su<sub>pp</sub>lies a suficient test for a<sub>g</sub>ents to share a structural a<sub>g</sub>ent t<sub>yp</sub>e<sub>,</sub> but failure of a com<sub>p</sub>onentwise <sub>p</sub>reser-<sub>va</sub>ti<sub>on</sub> <sub>equa</sub>ti<sub>on</sub> f<sub>or</sub> <sub>one</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>a</sub>t th<sub>e</sub>i<sub>r s</sub>t<sub>ruc</sub>t<sub>ura</sub>l t<sub>ypes</sub> dif<sub>er.</sub> A<sub>no</sub>th<sub>er a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma-</sub> tion ma<sub>y</sub> exchan<sub>g</sub>e the same a<sub>g</sub>ents. B<sub>y</sub> contrast<sub>,</sub> if no admissible trans<sup>f</sup>ormation maps � to �, t<sup>h</sup>en no po<sup>l</sup>icy-eva<sup>l</sup>uation symmetry <sub>can</sub> <sub>map</sub> th<sub>em,</sub> <sub>an</sub>d th<sub>ey</sub> <sub>necessar</sub>il<sub>y</sub> h<sub>ave</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t <sup>t</sup>yp<sup>es.</sup>

## A.1 Physical Heterogeneity

Bettini et al. [5] describe <sub>p</sub>h<sub>y</sub>sical hetero<sub>g</sub>eneit<sub>y</sub> throu<sub>g</sub>h diferences i<sub>n</sub> h<sub>ar</sub>d<sub>ware or p</sub>h<sub>ys</sub>i<sub>ca</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s, suc</sub>h <sub>as sensors, ac</sub>t<sub>ua</sub>t<sub>ors, an</sub>d <sub>mo</sub>ti<sub>on cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> O<sub>ur</sub> f<sub>ramewor</sub>k <sub>can c</sub>l<sub>ass</sub>if<sub>y suc</sub>h dif<sub>erences</sub> <sub>o</sub>nl<sub>y</sub> thr<sub>oug</sub>h th<sub>e</sub>ir r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n in th<sub>e</sub> POSG<sub>.</sub> Dif<sub>e</sub>r<sub>e</sub>nt <sub>se</sub>n<sub>so</sub>r<sub>s</sub> ma<sub>y</sub> a<sub>pp</sub>ear in the observation s<sub>p</sub>aces or observation distribution<sub>,</sub> diferent actuators ma<sub>y</sub> a<sub>pp</sub>ear in the action s<sub>p</sub>aces or transition distribution<sub>,</sub> and motion constraints ma<sub>y</sub> a<sub>pp</sub>ear in the state and tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> If <sub>ac</sub>ti<sub>o</sub>n <sub>o</sub>r <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>spaces</sub> h<sub>ave</sub> dif<sub>e</sub>rent car<sup>d</sup>ina<sup>l</sup>ities, t<sup>h</sup>e require<sup>d</sup> type<sup>d</sup> <sup>b</sup>ijection <sup>d</sup>oes not exist. Equa<sup>l</sup> car<sup>d</sup>ina<sup>l</sup>ity permits a <sup>b</sup>ijection <sup>b</sup>etween <sup>d</sup>i<sup>f</sup>erent<sup>l</sup>y name<sup>d</sup> spaces, b<sub>u</sub>t it d<sub>oes no</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>p</sub>l<sub>ace</sub> th<sub>e resu</sub>lti<sub>ng comp</sub>l<sub>e</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>on-spec</sub>ifi<sub>e</sub>d <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>group.</sub> Th<sub>us,</sub> th<sub>e</sub> f<sub>orma</sub>l <sub>com-</sub> <sub>par</sub>i<sub>son</sub> <sub>concerns</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>e</sub>d <sub>consequences</sub> <sub>o</sub>f <sub>a</sub> <sub>p</sub>h<sub>ys</sub>i<sub>ca</sub>l dif<sub>erence</sub> rather than hard<sub>w</sub>are identit<sub>y</sub> itself. A <sub>p</sub>h<sub>y</sub>sical <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> that is not re<sub>p</sub>resented in the POSG lies o<sub>u</sub>tside the str<sub>u</sub>ct<sub>u</sub>ral classification.

## A.2 Observation Heterogeneity

Hu et al. [15] define observation hetero<sub>g</sub>eneit<sub>y</sub> throu<sub>g</sub>h diferences in a<sub>g</sub>ents’ observation s<sub>p</sub>aces or local observation functions. Our <sub>c</sub>l<sub>oses</sub>t t<sub>es</sub>t <sub>com</sub>bi<sub>nes</sub> th<sub>e</sub> t<sub>ype</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on maps</sub> $\boldsymbol { \phi } _ { g , i } ^ { O }$ wit<sup>h</sup> t<sup>h</sup>e joint observation-<sub>p</sub>reservation condition in E<sub>q</sub>uation (6). The com<sub>p</sub>ari<sub>son</sub> i<sub>s</sub> <sub>ma</sub>d<sub>e</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>ac</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> h<sub>ave</sub> b<sub>een</sub> t<sub>rans</sub>f<sub>orme</sub>d<sub>.</sub> H<sub>ence</sub> dif<sub>eren</sub>tl<sub>y</sub> <sub>name</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> <sub>nee</sub>d <sub>no</sub>t indicate diferent structural a<sub>g</sub>ent t<sub>yp</sub>es. To com<sub>p</sub>are directl<sub>y</sub> with H<sub>u</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub>’<sub>s</sub> <sub>agen</sub>t<sub>-</sub>l<sub>oca</sub>l f<sub>ormu</sub>l<sub>a</sub>ti<sub>on,</sub> <sub>an</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>mus</sub>t <sub>a</sub>l<sub>so</sub> <sub>spec-</sub> if<sub>y</sub> com<sub>p</sub>atible timin<sub>g</sub> and conditionin<sub>g</sub> conventions and ex<sub>p</sub>lain <sup>h</sup>ow its <sup>l</sup>oca<sup>l</sup> o<sup>b</sup>servation <sup>f</sup>unctions in<sup>d</sup>uce t<sup>h</sup>e joint o<sup>b</sup>servation di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $Z .$ If E<sub>q</sub>uation (6) fails, that <sub>p</sub>articular com<sub>p</sub>onentwise environment automor<sub>p</sub>hism is excluded<sub>,</sub> but a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> ma<sub>y</sub> still relate the a<sub>g</sub>ents.

## A.3 Response- and Efect-Transition Heterogeneity

Hu et al. [15] distin<sub>g</sub>uish two locations at which a<sub>g</sub>ents ma<sub>y</sub> difer in the transition d<sub>y</sub>namics. Their a<sub>g</sub>ent-level model decom<sub>p</sub>oses th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e as</sub>

$$
\widehat { s } = ( s _ { 1 } , \ldots , s _ { | N | } , s _ { E } ) \in \left( \prod _ { k \in N } S _ { k } \right) \times S _ { E } ,
$$

<sub>w</sub>h<sub>ere</sub> $s _ { k }$ <sup>i</sup>s a<sub>g</sub>ent $k ' s$ l<sub>oca</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d $s _ { E }$ <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> th<sub>e</sub> r<sub>e</sub>m<sub>a</sub>inin<sub>g</sub> <sub>e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> Writ<sub>e</sub> $\widehat { \boldsymbol { a } } = \left( a _ { 1 } , . . . , a _ { | N | } \right)$ <sup>f</sup>or t<sup>h</sup>e joint action. Res<sub>p</sub>onse-transition hetero<sub>g</sub>eneit<sub>y</sub> concerns the local transition law $\begin{array} { r } { T ^ { i } ( { \bf \cdot } \mid \widehat { s } , \widehat { a } ) } \end{array}$ : un<sup>d</sup>er correspon<sup>d</sup>ing g<sup>l</sup>o<sup>b</sup>a<sup>l</sup> states an<sup>d</sup> joint actions, it <sub>as</sub>k<sub>s w</sub>h<sub>e</sub>th<sub>er agen</sub>t<sub>s</sub> h<sub>ave</sub> dif<sub>eren</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons over</sub> th<sub>e</sub>i<sub>r own</sub> next local states. Efect-transition hetero<sub>g</sub>eneit<sub>y</sub> instead concerns $T ^ { - i }$ <sub>,</sub> th<sub>e co</sub>ll<sub>ec</sub>ti<sub>on o</sub>f t<sub>rans</sub>iti<sub>on</sub> l<sub>aws</sub> f<sub>or</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e componen</sub>t<sub>s o</sub>th<sub>er</sub> th<sub>an agen</sub>t �<sub>, v</sub>i<sub>ewe</sub>d <sub>as</sub> f<sub>unc</sub>ti<sub>ons o</sub>f $( s _ { i } , a _ { i } )$ <sub>as</sub> <sub>we</sub>ll <sub>as</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>ac</sub>ti<sub>on componen</sub>t<sub>s.</sub> It <sub>as</sub>k<sub>s w</sub>h<sub>e</sub>th<sub>er</sub> dif<sub>eren</sub>t <sub>agen</sub>t<sub>s</sub>’ <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>or</sub> <sub>ac</sub>ti<sub>ons</sub> i<sub>n</sub>fl<sub>uence</sub> th<sub>e</sub> <sub>res</sub>t <sub>o</sub>f th<sub>e</sub> <sub>sys</sub>t<sub>em</sub> i<sub>n</sub> dif<sub>eren</sub>t <sub>ways.</sub>

Th<sub>e co</sub>r<sub>e</sub> POSG in S<sub>ec</sub>ti<sub>o</sub>n 3 d<sub>oes</sub> n<sub>o</sub>t <sub>assu</sub>m<sub>e</sub> thi<sub>s age</sub>nt-l<sub>eve</sub>l d<sub>e</sub>- com<sub>p</sub>osition. It s<sub>p</sub>ecifies an undivided state s<sub>p</sub>ace $s$ <sub>an</sub>d <sub>one g</sub>l<sub>o</sub>b<sub>a</sub>l tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $P ( s ^ { \prime } \mid s , a )$ <sub>,</sub> without canonicall<sub>y</sub> assi<sub>g</sub>nin<sub>g</sub> coordinates of <sub>�</sub> to individual a<sub>g</sub>ents or attributin<sub>g p</sub>arts of the transition law to <sub>p</sub>articular u<sub>p</sub>date mechanisms. Conse<sub>q</sub>uentl<sub>y,</sub> a dif<sub>erence</sub> i<sub>n an agen</sub>t’<sub>s own nex</sub>t<sub>-s</sub>t<sub>a</sub>t<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on an</sub>d <sub>a</sub> dif<sub>er-</sub> ence in its influence on other next-state com<sub>p</sub>onents both a<sub>pp</sub>ear as diferences in the same <sub>g</sub>lobal distribution. E<sub>q</sub>uation (5) can t<sub>es</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>a</sub>t <sub>en</sub>ti<sub>re</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s preserve</sub>d b<sub>y a can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> If th<sub>e</sub> <sub>equa</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>s,</sub> h<sub>owever,</sub> th<sub>e</sub> <sub>un</sub>di<sub>v</sub>id<sub>e</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>revea</sub>l <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> f<sub>a</sub>il<sub>ure</sub> <sub>arose</sub> f<sub>rom</sub> <sub>response</sub> d<sub>ynam</sub>i<sub>cs,</sub> <sub>e</sub>f<sub>ec</sub>t d<sub>ynam</sub>i<sub>cs,</sub> <sub>or</sub> b<sub>o</sub>th<sub>.</sub> Th<sub>us,</sub> th<sub>e</sub> t<sub>wo</sub> <sub>ca</sub>t<sub>egor</sub>i<sub>es</sub> h<sub>ave</sub> th<sub>e</sub> <sub>same</sub> <sub>c</sub>l<sub>oses</sub>t <sub>coun</sub>t<sub>erpar</sub>t i<sub>n</sub> <sub>our</sub> <sub>core</sub> <sub>mo</sub>d<sub>e</sub>l <sub>on</sub>l<sub>y</sub> in this re<sub>p</sub>resentational sense. The<sub>y</sub> are not treated as e<sub>q</sub>uivalent <sup>conce</sup>p<sup>ts.</sup>

To distin<sub>g</sub>uish them<sub>,</sub> an a<sub>pp</sub>lication ma<sub>y</sub> additionall<sub>y</sub> declare a de-<sub>co</sub>m<sub>pos</sub>iti<sub>o</sub>n $\begin{array} { r } { S = ( \prod _ { k \in N } S _ { k } ) \times S _ { E } } \end{array}$ <sub>,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith <sub>a</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>co</sub>ll<sub>ec-</sub> ti<sub>on o</sub>f l<sub>oca</sub>l t<sub>rans</sub>iti<sub>on</sub> f<sub>ac</sub>t<sub>ors an</sub>d th<sub>e</sub>i<sub>r</sub> d<sub>epen</sub>d<sub>ency scopes.</sub> A <sub>can-</sub> did<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>mus</sub>t th<sub>en</sub> <sub>map</sub> <sub>eac</sub>h l<sub>oca</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>componen</sub>t <sub>an</sub>d t<sub>rans</sub>iti<sub>on</sub> f<sub>ac</sub>t<sub>or</sub> t<sub>o</sub> th<sub>e correspon</sub>di<sub>ng componen</sub>t <sub>an</sub>d f<sub>ac</sub>t<sub>or</sub> for the tar<sub>g</sub>et a<sub>g</sub>ent. A res<sub>p</sub>onse-<sub>p</sub>reservation test com<sub>p</sub>ares the factors t<sup>h</sup>at <sub>g</sub>enerate $s _ { i } ^ { \prime }$ <sub>an</sub>d $s _ { \sigma _ { g } ( i ) } ^ { \prime }$ <sub>.</sub> An <sub>e</sub>f<sub>ec</sub>t-<sub>p</sub>r<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n t<sub>es</sub>t in<sub>s</sub>t<sub>ea</sub>d <sub>compares</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>ors</sub> f<sub>or</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>agen</sub>t<sub>s</sub> <sub>an</sub>d <sub>env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>nso</sub>f<sub>ar as</sub> th<sub>ose</sub> f<sub>ac</sub>t<sub>ors</sub> d<sub>epen</sub>d <sub>on</sub> $( s _ { i } , a _ { i } )$ <sub>an</sub>d $( s _ { \sigma _ { g } ( i ) } , a _ { \sigma _ { g } ( i ) } )$ res<sub>p</sub>ectivel<sub>y</sub>. This additional structure is a<sub>pp</sub>lication-s<sub>p</sub>ecific and<sub>,</sub> in <sub>genera</sub>l<sub>, canno</sub>t b<sub>e recovere</sub>d <sub>un</sub>i<sub>que</sub>l<sub>y</sub> f<sub>rom</sub> $P$ <sub>a</sub>l<sub>one.</sub> F<sub>or examp</sub>l<sub>e,</sub> t<sub>wo ro</sub>b<sub>o</sub>t<sub>s ex</sub>hibit <sub>a response</sub> dif<sub>erence</sub> if th<sub>e same</sub> d<sub>r</sub>i<sub>ve comman</sub>d under corres<sub>p</sub>ondin<sub>g</sub> conditions <sub>p</sub>roduces diferent distributions <sub>over</sub> th<sub>e</sub>i<sub>r own nex</sub>t <sub>ve</sub>l<sub>oc</sub>iti<sub>es.</sub> Th<sub>ey ex</sub>hibit <sub>an e</sub>f<sub>ec</sub>t dif<sub>erence</sub> if corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>ushin<sub>g</sub> actions <sub>p</sub>roduce diferent distributions over t<sup>h</sup>e next position o<sup>f</sup> a s<sup>h</sup>are<sup>d</sup> o<sup>b</sup>ject. Wit<sup>h</sup>out <sup>d</sup>ec<sup>l</sup>are<sup>d</sup> ve<sup>l</sup>ocity an<sup>d</sup> o<sup>b</sup>ject-state components, <sup>b</sup>ot<sup>h d</sup>i<sup>f</sup>erences appear on<sup>l</sup>y as c<sup>h</sup>anges in $P ( s ^ { \prime } \mid s , a )$

Th<sub>ese</sub> f<sub>ac</sub>t<sub>or-</sub>l<sub>eve</sub>l t<sub>es</sub>t<sub>s</sub> id<sub>en</sub>tif<sub>y</sub> <sub>w</sub>h<sub>ere</sub> <sub>a</sub> dif<sub>erence</sub> i<sub>n</sub> th<sub>e</sub> d<sub>ynam-</sub> i<sub>cs</sub> <sub>occurs.</sub> Th<sub>ey</sub> d<sub>o</sub> <sub>no</sub>t <sub>prov</sub>id<sub>e</sub> <sub>a</sub> <sub>secon</sub>d d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l a<sub>g</sub>ent t<sub>yp</sub>e. <sup>S</sup>tructura<sup>l</sup> a<sub>g</sub>ent t<sub>yp</sub>es rema<sup>i</sup>n t<sup>h</sup>e a<sub>g</sub>ent or<sup>bi</sup>ts o<sup>f</sup> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ relative to the resultin<sub>g</sub> admissible transformation <sub>g</sub>rou<sub>p</sub>. Restricti<sub>ng</sub> th<sub>a</sub>t <sub>group</sub> t<sub>o</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>respec</sub>t th<sub>e</sub> d<sub>ec</sub>l<sub>are</sub>d f<sub>ac</sub>t<sub>or-</sub> ization can refine the resultin<sub>g</sub> orbit <sub>p</sub>artition. Moreover<sub>,</sub> failure of <sub>a response- or e</sub>f<sub>ec</sub>t<sub>-preserva</sub>ti<sub>on</sub> t<sub>es</sub>t f<sub>or one can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>f<sub>or-</sub> mation excludes that candidate com<sub>p</sub>onentwise s<sub>y</sub>mmetr<sub>y,</sub> but it d<sub>oes</sub> <sub>no</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>prove</sub> th<sub>a</sub>t t<sub>wo</sub> <sub>agen</sub>t<sub>s</sub> h<sub>ave</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l a<sub>g</sub>ent t<sub>yp</sub>es: another admissible <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> ma<sub>y</sub> <sub>s</sub>till <sub>exc</sub>h<sub>ange</sub> th<sub>em.</sub>

## A.4 Objective Heterogeneity

Hu et al. [15] com<sub>p</sub>are a<sub>g</sub>ents’ conditional reward distributions, while Bettini et al. [5] distin<sub>g</sub>uish behavior under a shared objective <sup>f</sup>rom <sup>b</sup>e<sup>h</sup>avior un<sup>d</sup>er <sup>d</sup>i<sup>f</sup>erent o<sup>b</sup>jectives. T<sup>h</sup>e c<sup>l</sup>osest component<sub>w</sub>i<sub>se</sub> t<sub>es</sub>t <sub>u</sub>nd<sub>e</sub>r <sub>ou</sub>r d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n-r<sub>ewa</sub>rd <sub>co</sub>n<sub>ve</sub>nti<sub>o</sub>n is the exact reward-<sub>p</sub>reservation condition in E<sub>q</sub>uation (7). It com-<sub>pares</sub> th<sub>e rewar</sub>d f<sub>unc</sub>ti<sub>ons a</sub>ft<sub>er</sub> th<sub>e agen</sub>t<sub>, s</sub>t<sub>a</sub>t<sub>e, an</sub>d <sub>ac</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> h<sub>ave</sub> b<sub>een</sub> t<sub>rans</sub>f<sub>orme</sub>d<sub>,</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>unc</sub>ti<sub>ons</sub> t<sub>o</sub> b<sub>e</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>un</sub>d<sub>er</sub> th<sub>e</sub>i<sub>r source</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> F<sub>a</sub>il<sub>ure o</sub>f thi<sub>s con</sub>diti<sub>on ex-</sub> cludes that <sub>p</sub>articular com<sub>p</sub>onentwise environment automor<sub>p</sub>hism b<sub>u</sub>t d<sub>oes no</sub>t<sub>,</sub> b<sub>y</sub> it<sub>se</sub>lf<sub>, es</sub>t<sub>a</sub>bli<sub>s</sub>h dif<sub>eren</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes.</sub>

S<sub>ec</sub>ti<sub>on</sub> 6 <sub>a</sub>l<sub>so s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>a common rewar</sub>d f<sub>unc</sub>ti<sub>on</sub> d<sub>oes no</sub>t i<sub>m-</sub> <sub>p</sub>l<sub>y</sub> structural homo<sub>g</sub>eneit<sub>y</sub>. In a Dec-POMDP<sub>,</sub> all a<sub>g</sub>ents have e<sub>q</sub>ual va<sup>l</sup>ue coor<sup>d</sup>inates un<sup>d</sup>er t<sup>h</sup>e same po<sup>l</sup>icy an<sup>d</sup> state–joint-<sup>h</sup>istory i<sub>npu</sub>t<sub>,</sub> b<sub>u</sub>t th<sub>ey nee</sub>d <sub>no</sub>t b<sub>e exc</sub>h<sub>angea</sub>bl<sub>e</sub> i<sub>n</sub> th<sub>e</sub>i<sub>r ac</sub>ti<sub>on spaces,</sub> observation s<sub>p</sub>aces<sub>,</sub> or roles in the transition and observation distributions. Conversel<sub>y,</sub> formin<sub>g</sub> an a<sub>gg</sub>re<sub>g</sub>ate common reward can <sub>p</sub>reserve<sub>,</sub> remove<sub>,</sub> or create <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries<sub>,</sub> de<sub>p</sub>ending on t<sup>h</sup>e in<sup>d</sup>ivi<sup>d</sup>ua<sup>l</sup> rewar<sup>d</sup>s an<sup>d</sup> aggregation weig<sup>h</sup>ts. O<sup>b</sup>jective h<sub>e</sub>t<sub>erogene</sub>it<sub>y an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y are</sub> th<sub>ere</sub>f<sub>ore re</sub>l<sub>a</sub>t<sub>e</sub>d but distinct <sub>p</sub>ro<sub>p</sub>erties.

## A.5 Behavioral and Policy Heterogeneity

The <sub>p</sub>olic<sub>y</sub>-out<sub>p</sub>ut as<sub>p</sub>ect of behavioral and <sub>p</sub>olic<sub>y</sub> hetero<sub>g</sub>eneit<sub>y</sub> <sub>concerns a se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy ra</sub>th<sub>er</sub> th<sub>an on</sub>l<sub>y</sub> th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng game.</sub> <sup>Su</sup>pp<sup>ose</sup> $H \leq G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ . A selected <sub>p</sub>olic<sub>y �</sub> is �-e<sub>q</sub>uivariant exactl<sub>y</sub> <sub>w</sub>h<sub>en,</sub> f<sub>or every</sub> $g \in H$ , <sup>a</sup>g<sup>ent</sup> $i ,$ d<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n tim<sub>e</sub> <sub>s</sub>t<sub>ep</sub> $t \ < \ T$ <sub>,</sub> l<sub>oca</sub>l histor<sub>y</sub> $h _ { i } ^ { t } ,$ <sub>, an</sub>d <sub>ac</sub>ti<sub>on</sub> $a _ { i }$

$$
\pi _ { \sigma _ { g } ( i ) } ^ { t } \left( \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \mid \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) \right) = \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) .
$$

Aft<sub>er</sub> th<sub>e prescr</sub>ib<sub>e</sub>d hi<sub>s</sub>t<sub>ory an</sub>d <sub>ac</sub>ti<sub>on</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons,</sub> th<sub>e</sub> t<sub>wo</sub> l<sub>oca</sub>l <sub>ru</sub>l<sub>es</sub> th<sub>ere</sub>f<sub>ore ass</sub>i<sub>gn</sub> th<sub>e same pro</sub>b<sub>a</sub>biliti<sub>es</sub> t<sub>o correspon</sub>d<sub>-</sub> in<sub>g</sub> actions. This is the s<sub>y</sub>mmetr<sub>y</sub>-aware condition for behavioral homo<sub>g</sub>eneit<sub>y</sub> within each a<sub>g</sub>ent orbit of �. The <sub>p</sub>olic<sub>y</sub> tables ma<sub>y</sub> l<sub>oo</sub>k dif<sub>eren</sub>t <sub>un</sub>d<sub>er</sub> th<sub>e</sub>i<sub>r source</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s an</sub>d <sub>s</sub>till <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> it<sub>.</sub>

For exam<sub>p</sub>le<sub>,</sub> consider two a<sub>g</sub>ents in mirrored roles. One a<sub>g</sub>ent <sub>may</sub> t<sub>urn r</sub>i<sub>g</sub>ht <sub>a</sub>ft<sub>er o</sub>b<sub>serv</sub>i<sub>ng an o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>e on</sub> it<sub>s</sub> l<sub>e</sub>ft<sub>, w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng ru</sub>l<sub>e</sub> f<sub>or</sub> th<sub>e o</sub>th<sub>er agen</sub>t i<sub>s</sub> t<sub>o</sub> t<sub>urn</sub> l<sub>e</sub>ft <sub>a</sub>ft<sub>er o</sub>b<sub>serv</sub>i<sub>ng</sub> an obstacle on its ri<sub>g</sub>ht. The rules are e<sub>q</sub>uivariant if the <sub>g</sub>ame s<sub>y</sub>m <sub>me</sub>t<sub>ry exc</sub>h<sub>anges</sub> th<sub>e agen</sub>t<sub>s an</sub>d th<sub>e</sub> l<sub>e</sub>ft<sub>–r</sub>i<sub>g</sub>ht l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> If <sub>a se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy</sub> f<sub>a</sub>il<sub>s</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>equa</sub>lit<sub>y</sub> f<sub>or</sub> thi<sub>s</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on,</sub> it b<sub>rea</sub>k<sub>s</sub> that <sub>p</sub>articular <sub>g</sub>ame s<sub>y</sub>mmetr<sub>y</sub> and ex<sub>p</sub>resses a <sub>p</sub>olic<sub>y</sub>-relative beh<sub>av</sub>i<sub>ora</sub>l dif<sub>erence.</sub> Thi<sub>s</sub> d<sub>oes</sub> <sub>no</sub>t <sub>c</sub>h<sub>ange</sub> th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>are</sub> d<sub>e</sub>fi<sub>ne</sub>d f<sub>rom</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>po</sub>li<sub>cy</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>or</sub> <sub>every</sub> <sub>po</sub>li<sub>cy.</sub> N<sub>or</sub> d<sub>oes</sub> f<sub>a</sub>il<sub>ure</sub> f<sub>or</sub> <sub>one</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>prove</sub> th<sub>a</sub>t th<sub>e</sub> <sub>agen</sub>t<sub>s</sub> <sub>are</sub> <sub>unre</sub>l<sub>a</sub>t<sub>e</sub>d <sub>a</sub>t th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> l<sub>eve</sub>l<sub>,</sub> b<sub>ecause</sub> <sub>ano</sub>th<sub>er</sub> <sub>po</sub>li<sub>cy-eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>symme</sub>t<sub>ry may</sub> b<sub>o</sub>th <sub>preserve</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>po</sub>li<sub>cy an</sub>d <sub>exc</sub>h<sub>ange</sub> th<sub>em.</sub>

The <sub>p</sub>olic<sub>y</sub>-hetero<sub>g</sub>eneit<sub>y</sub> definition of Hu et al. [15] covers more th<sub>an</sub> dif<sub>erences</sub> b<sub>e</sub>t<sub>ween po</sub>li<sub>cy ou</sub>t<sub>pu</sub>t<sub>s:</sub> it <sub>a</sub>l<sub>so</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> dif<sub>erences</sub> in action or obser<sub>v</sub>ation s<sub>p</sub>aces. In o<sub>u</sub>r frame<sub>w</sub>ork<sub>,</sub> the existence o<sup>f</sup> t<sup>h</sup>e necessary <sup>b</sup>ijections is sett<sup>l</sup>e<sup>d</sup> <sup>fi</sup>rst <sup>b</sup>y t<sup>h</sup>e type-compati<sup>b</sup>i<sup>l</sup>ity an<sup>d</sup> <sup>b</sup>ijectivity requirements. Po<sup>l</sup>icy equivariance t<sup>h</sup>en compares t<sup>h</sup>e <sup>d</sup>ecision ru<sup>l</sup>es a<sup>f</sup>ter t<sup>h</sup>ose <sup>b</sup>ijections <sup>h</sup>ave <sup>b</sup>een app<sup>l</sup>ie<sup>d</sup>. Bettini et al. [5] use the <sub>p</sub>ossibilit<sub>y</sub> of distinct <sub>p</sub>olic<sub>y</sub> out<sub>p</sub>uts on the same i<sub>npu</sub>t t<sub>o</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> <sub>agen</sub>t<sub>s</sub> <sub>w</sub>ith dif<sub>eren</sub>t b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> Th<sub>ey</sub> do not<sub>,</sub> however<sub>,</sub> restrict their anal<sub>y</sub>sis to identical in<sub>p</sub>uts. Their behavioral-t<sub>yp</sub>in<sub>g</sub> anal<sub>y</sub>sis also studies how one shared <sub>p</sub>olic<sub>y</sub> can <sub>pro</sub>d<sub>uce apparen</sub>tl<sub>y</sub> h<sub>e</sub>t<sub>erogeneous</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or w</sub>h<sub>en agen</sub>t id<sub>en</sub>tifi<sub>ers</sub> or other diferences in observations <sub>p</sub>rovide diferent in<sub>p</sub>uts. Our <sub>equ</sub>i<sub>var</sub>i<sub>ance con</sub>diti<sub>on a</sub>dd<sub>resses ano</sub>th<sub>er ques</sub>ti<sub>on: w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d l<sub>oca</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es agree a</sub>t hi<sub>s</sub>t<sub>or</sub>i<sub>es an</sub>d <sub>ac</sub>ti<sub>ons re</sub>l<sub>a</sub>t<sub>e</sub>d b<sub>y a</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>. S<sub>p</sub>ecificall<sub>y,</sub> it com<sub>p</sub>ares the <sub>p</sub>robabilit<sub>y</sub> assi<sub>g</sub>ned to $a _ { i }$ at $h _ { i } ^ { t }$ with the <sub>p</sub>robabilit<sub>y</sub> assi<sub>g</sub>ned to $\bar { \phi _ { g , i } ^ { \mathcal { R } } }$ (�<sub>�</sub> ) at $\Phi _ { q , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } )$ <sup>b</sup>y <sup>a</sup>g<sup>ent</sup> $\sigma _ { g } ( i )$ <sub>.</sub> Wh<sub>en</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> <sub>an</sub>d <sub>ac</sub>ti<sub>on</sub> t<sub>rans</sub>f<sub>orma-</sub> ti<sub>o</sub>n<sub>s a</sub>r<sub>e</sub> id<sub>e</sub>ntiti<sub>es,</sub> thi<sub>s</sub> r<sub>e</sub>d<sub>uces</sub> t<sub>o</sub> th<sub>e sa</sub>m<sub>e</sub>-in<sub>pu</sub>t <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n<sub>.</sub>

## A.6 Exact Classes and Graded Diferences

Th<sub>e s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>ar</sub>titi<sub>on</sub> $\mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } )$ i<sub>s an exac</sub>t <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on:</sub> t<sub>wo</sub> a<sub>g</sub>ents share a structural a<sub>g</sub>ent t<sub>yp</sub>e when some <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> exchan<sub>g</sub>es them. The s<sub>y</sub>stem neural diversit<sub>y</sub> measure [6] and the hetero<sub>g</sub>eneit<sub>y</sub> distances of Hu et al. [15] instead measure d<sub>egrees o</sub>f dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>or mo</sub>d<sub>e</sub>l <sub>componen</sub>t<sub>s.</sub> Such measures can com<sub>p</sub>are a<sub>g</sub>ents within one structural a<sub>g</sub>ent t<sub>yp</sub>e or in diferent t<sub>yp</sub>es when their re<sub>q</sub>uired domains are com<sub>p</sub>a-<sub>ra</sub>bl<sub>e,</sub> b<sub>u</sub>t th<sub>ey</sub> d<sub>o</sub> <sub>no</sub>t <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>or</sub>bit <sub>par</sub>titi<sub>on.</sub> C<sub>onverse</sub>l<sub>y,</sub> <sub>or</sub>bit membershi<sub>p</sub> does not assi<sub>g</sub>n a numerical de<sub>g</sub>ree of similarit<sub>y</sub>.

Th<sub>e</sub> fi<sub>na</sub>l <sub>row o</sub>f T<sub>a</sub>bl<sub>e</sub> 1 l<sub>oca</sub>t<sub>es sys</sub>t<sub>em neura</sub>l di<sub>vers</sub>it<sub>y a</sub>t th<sub>e</sub> selected-<sub>p</sub>olic<sub>y</sub> level. Its <sub>p</sub>airwise distances com<sub>p</sub>are a<sub>g</sub>ents’ action distrib<sub>u</sub>tions on common obser<sub>v</sub>ation in<sub>pu</sub>ts<sub>,</sub> ass<sub>u</sub>min<sub>g</sub> shared observation and action s<sub>p</sub>aces [6]. This com<sub>p</sub>arison does not test <sub>p</sub>reservation of com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation under a<sub>g</sub>ent <sub>p</sub>ermutations.

## A.7 Summary

In summar<sub>y,</sub> the transition<sub>,</sub> observation<sub>,</sub> and reward <sub>p</sub>reservation <sub>con</sub>diti<sub>ons</sub> di<sub>agnose separa</sub>t<sub>e</sub>l<sub>y represen</sub>t<sub>e</sub>d <sub>par</sub>t<sub>s o</sub>f th<sub>e mo</sub>d<sub>e</sub>l<sub>.</sub> Polic<sub>y</sub>-evaluation s<sub>y</sub>mmetries define structural a<sub>g</sub>ent t<sub>yp</sub>es for the s<sub>p</sub>ecified POSG and admissible transformation <sub>g</sub>ro<sub>up</sub>. Given these s<sub>y</sub>mmetries<sub>,</sub> <sub>p</sub>olic<sub>y</sub> e<sub>q</sub>uivariance determines whether a selected <sub>p</sub>olic<sub>y</sub> is in<sub>v</sub>ariant <sub>u</sub>nder them. Table 1 classifies the <sub>p</sub>rior cate-<sub>gor</sub>i<sub>es</sub> <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> l<sub>eve</sub>l <sub>o</sub>f <sub>our</sub> f<sub>ramewor</sub>k th<sub>a</sub>t th<sub>ey</sub> d<sub>escr</sub>ib<sub>e.</sub> St<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> <sub>are</sub> i<sub>ns</sub>t<sub>ea</sub>d d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>agen</sub>t <sub>or</sub>bit<sub>s</sub> <sub>o</sub>f th<sub>e</sub> p<sup>olic</sup>y<sup>-evaluation</sup> <sup>s</sup>y<sup>mmetr</sup>y g<sup>rou</sup>p<sup>.</sup>

## B Proofs for the Core Framework

In t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing proo<sup>f</sup>s, we <sup>d</sup>e<sup>fi</sup>ne t<sup>h</sup>e pro<sup>d</sup>uct spaces <sup>f</sup>or t<sup>h</sup>e joint action, joint o<sup>b</sup>servation, an<sup>d</sup> joint <sup>h</sup>istory at time step �, as <sup>f</sup>o<sup>ll</sup>ows:

$$
\mathcal { A } = \prod _ { i \in N } \mathcal { A } _ { i } , \qquad O = \prod _ { i \in N } O _ { i } , \qquad \mathcal { H } ^ { t } = \prod _ { i \in N } \mathcal { H } _ { i } ^ { t } .
$$

F<sub>o</sub>r tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> $g , k ,$ th<sub>e</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>�</sub>� <sub>means</sub> fi<sub>rs</sub>t <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> � <sub>an</sub>d th<sub>en</sub> $g .$ Th<sub>us,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>compos</sub>iti<sub>on</sub> l<sub>aws</sub> f<sub>or</sub> th<sub>e</sub> <sub>agen</sub>t <sub>p</sub>ermutation and the state<sub>,</sub> action<sub>,</sub> and observation ma<sub>p</sub>s:

$$
\begin{array} { r l } { \sigma _ { g k } = \sigma _ { g } \circ \sigma _ { k } , \quad } & { \phi _ { g k } ^ { S } = \phi _ { g } ^ { S } \circ \phi _ { k } ^ { S } , } \\ { \phi _ { g k , i } ^ { \mathcal { A } } = \phi _ { g , \sigma _ { k } ( i ) } ^ { \mathcal { A } } \circ \phi _ { k , i } ^ { \mathcal { A } } , \quad } & { \phi _ { g k , i } ^ { O } = \phi _ { g , \sigma _ { k } ( i ) } ^ { O } \circ \phi _ { k , i } ^ { O } . } \end{array}
$$

We <sub>w</sub>rite $( h ^ { t } , a , o ^ { \prime } )$ <sup>f</sup>or t<sup>h</sup>e joint <sup>h</sup>istory o<sup>b</sup>taine<sup>d</sup> <sup>b</sup>y appen<sup>d</sup>ing $( a _ { i } , o _ { i } ^ { \prime } )$ t<sub>o</sub> <sub>eac</sub>h l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory</sub> $h _ { i } ^ { t }$

## B.1 Induced actions

Proposition B.1 (Induced group actions). The transformations in Equations (3) and (4) define left group actions $o f \mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ on $\Pi _ { \mathrm { d e c } }$ and V, respectively. Specifically:

(i) Policies. For every �, $k \in \mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ and $\pi \in \Pi _ { \mathrm { d e c } }$ , the transformed policy belongs to $\Pi _ { \mathrm { d e c } } ,$ and

$$
e \cdot \pi = \pi , \qquad g \cdot ( k \cdot \pi ) = ( g k ) \cdot \pi .
$$

For every $t < T ,$ , joint history $h ^ { t } ,$ and joint action �,

$$
( g \cdot \pi ) ^ { t } ( g \cdot a \mid g \cdot h ^ { t } ) = \pi ^ { t } ( a \mid h ^ { t } ) = \prod _ { i \in N } \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) .
$$

Consequently, at every fixed joint history the transformed joint action distribution is still the product of one local distribution per agent. The deterministic transformation introduces no shared policyrandomization variable and no additional conditional correlation between agents’ actions.

(ii) Value profiles. For every $g , k \in \mathcal { R } _ { \mathrm { a d m } } ( M )$ and $V \in \mathcal { V }$ , the transformed profile belongs $t o \mathcal { N } ,$ , and

$$
e \cdot V = V , \qquad g \cdot ( k \cdot V ) = ( g k ) \cdot V .
$$

Th<sub>e</sub> <sub>va</sub>l<sub>ue-pro</sub>fil<sub>e</sub> <sub>ac</sub>ti<sub>on</sub> l<sub>aws</sub> <sub>ensure</sub> th<sub>a</sub>t <sub>success</sub>i<sub>ve</sub> t<sub>rans</sub>f<sub>or-</sub> m<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> <sub>co</sub>m<sub>pose</sub> <sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>ntl<sub>y.</sub> Th<sub>ey</sub> <sub>a</sub>r<sub>e</sub> <sub>use</sub>d in Pr<sub>opos</sub>iti<sub>o</sub>n B<sub>.</sub>2 t<sub>o</sub> establish that <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries form a sub<sub>g</sub>rou<sub>p</sub>.

P<sub>roof.</sub> B<sub>y</sub> th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>ew</sub>i<sub>se</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> l<sub>oca</sub>l<sub>-</sub>hi<sub>s</sub>t<sub>ory</sub> <sub>maps an</sub>d th<sub>e compos</sub>iti<sub>on</sub> l<sub>aws</sub> f<sub>or</sub> th<sub>e ac</sub>ti<sub>on an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> ma<sub>p</sub>s<sub>,</sub> we obtain the followin<sub>g</sub> e<sub>q</sub>uation:

$$
\boldsymbol { \Phi } _ { g k , i } ^ { \mathcal { H } , t } = \boldsymbol { \Phi } _ { g , \sigma _ { k } ( i ) } ^ { \mathcal { H } , t } \circ \boldsymbol { \Phi } _ { k , i } ^ { \mathcal { H } , t } ,
$$

<sub>w</sub>h<sub>ere</sub> $\sigma _ { k } ( i )$ id<sub>en</sub>tifi<sub>es</sub> th<sub>e agen</sub>t <sub>w</sub>h<sub>ose coor</sub>di<sub>na</sub>t<sub>e con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> l<sub>oca</sub>l histor<sub>y</sub> after a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> �. At the initial time ste<sub>p</sub> $t = 0$ <sub>,</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> fi<sub>x</sub> th<sub>e</sub> <sub>emp</sub>t<sub>y</sub> hi<sub>s</sub>t<sub>ory.</sub>

T<sup>h</sup>e joint-<sup>h</sup>istory trans<sup>f</sup>ormation co<sup>ll</sup>ects t<sup>h</sup>ese <sup>l</sup>oca<sup>l</sup>-<sup>h</sup>istory ma<sub>p</sub>s in the <sub>p</sub>ermuted a<sub>g</sub>ent coordinates. Usin<sub>g</sub> $\sigma _ { g k } = \sigma _ { g } \circ \sigma _ { k }$ <sup>f</sup>or ever<sub>y</sub> ${ \bf \nabla } \cdot { \bf \nabla } i \in N$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t<sub>:</sub>

$$
\begin{array} { r l } & { [ g \cdot ( \boldsymbol { k } \cdot \boldsymbol { h } ^ { t } ) ] _ { \sigma _ { g k } ( i ) } } \\ & { \quad = \Phi _ { g , \sigma _ { k } ( i ) } ^ { \mathcal { H } , t } \left( \Phi _ { k , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) \right) } \\ & { \quad = \Phi _ { g k , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) = [ ( g k ) \cdot h ^ { t } ] _ { \sigma _ { g k } ( i ) } . } \end{array}
$$

Sin<sub>ce</sub> $\sigma _ { g k }$ is a <sub>p</sub>ermutation<sub>,</sub> this <sub>p</sub>roves that:

$$
( g k ) \cdot h ^ { t } = g \cdot ( k \cdot h ^ { t } ) .
$$

Finall<sub>y,</sub> combinin<sub>g</sub> this identit<sub>y</sub> with $\phi _ { g k } ^ { S } = \phi _ { g } ^ { S } \circ \phi _ { k } ^ { S }$ <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>-</sub> tion $\tau _ { g } ^ { t } ( s , h ^ { t } ) = ( \phi _ { g } ^ { S } ( s ) , g \cdot h ^ { t } )$ <sub>g</sub>ives that:

$$
\tau _ { g k } ^ { t } = \tau _ { g } ^ { t } \circ \tau _ { k } ^ { t } .
$$

The identit<sub>y</sub> acts triviall<sub>y,</sub> and the inverse coordinate ma<sub>p</sub>s <sub>u</sub>ndo <sub>eac</sub>h t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub>

For <sub>p</sub>olicies<sub>,</sub> <sub>pu</sub>t $a ^ { \prime } = g \cdot a$ <sub>an</sub>d $h ^ { \prime } = g \cdot h ^ { t }$ <sub>.</sub> Th<sub>en,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t<sub>:</sub>

$$
\begin{array} { l } { { ( g \cdot \pi ) ^ { t } ( a ^ { \prime } \mid h ^ { \prime } ) = \displaystyle \prod _ { j \in N } ( g \cdot \pi ) _ { j } ^ { t } ( a _ { j } ^ { \prime } \mid h _ { j } ^ { \prime } ) } } \\ { { \ \qquad = \displaystyle \prod _ { i \in N } ( g \cdot \pi ) _ { \sigma _ { g } ( i ) } ^ { t } ( a _ { \sigma _ { g } ( i ) } ^ { \prime } \mid h _ { \sigma _ { g } ( i ) } ^ { \prime } ) } } \\ { { \ \qquad = \displaystyle \prod _ { i \in N } \pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) = \pi ^ { t } ( a \mid h ^ { t } ) . } } \end{array}
$$

E<sub>ac</sub>h t<sub>rans</sub>f<sub>orme</sub>d l<sub>oca</sub>l <sub>ru</sub>l<sub>e</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory.</sub> It i<sub>s</sub> <sub>norma</sub>li<sub>ze</sub>d b<sub>ecause</sub> it<sub>s</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>are</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d <sup>b</sup>y <sup>b</sup>ijective<sup>l</sup>y rein<sup>d</sup>exing t<sup>h</sup>e source agent<sup>’</sup>s actions. T<sup>h</sup>e pointwise <sub>p</sub>olic<sub>y</sub> definition and the coordinate com<sub>p</sub>osition laws <sub>g</sub>ive $g \cdot ( k$ $\pi ) = ( g k ) \cdot \pi$ <sub>an</sub>d $e \cdot \pi = \pi$

F<sub>o</sub>r th<sub>e ac</sub>ti<sub>o</sub>n <sub>o</sub>n $\boldsymbol { \mathcal { V } } ,$ <sub>reca</sub>ll th<sub>a</sub>t thi<sub>s</sub> i<sub>s</sub> th<sub>e space o</sub>f <sub>rea</sub>l<sub>-va</sub>l<sub>ue</sub>d function <sub>p</sub>rofiles with zero terminal entries. Polic<sub>y</sub> evaluation <sub>p</sub>rod<sub>uces</sub> <sub>a</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> <sub>pro</sub>fil<sub>e</sub> $V ^ { \pi }$ in thi<sub>s</sub> <sub>space.</sub> Fix <sub>a</sub>n <sub>a</sub>rbitr<sub>a</sub>r<sub>y</sub> $V \in \mathcal { V } .$ Sin<sub>ce</sub> $\sigma _ { g }$ <sub>an</sub>d $\tau _ { g } ^ { t }$ are bijective, E<sub>q</sub>uation (4) assi<sub>g</sub>ns a uni<sub>q</sub>ue real value t<sub>o</sub> <sub>every</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e.</sub> Th<sub>e</sub> ti<sub>me</sub> i<sub>n</sub>d<sub>ex</sub> i<sub>s</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>,</sub> <sub>so</sub> t<sub>e</sub>rmin<sub>a</sub>l <sub>e</sub>ntri<sub>es</sub> r<sub>e</sub>m<sub>a</sub>in z<sub>e</sub>r<sub>o a</sub>nd $\boldsymbol { q } \cdot \boldsymbol { V } \in \mathcal { V } .$

<sup>F</sup>or an<sub>y</sub> $g , k \in \mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } ) , i \in N , 0 \leq t \leq T$ <sub>,</sub> <sub>an</sub>d $\boldsymbol { x } _ { t } \in \boldsymbol { X } _ { t }$ , <sup>a</sup>pp<sup>l</sup>y<sup>in</sup>g E<sub>q</sub>uation (4) twice <sub>g</sub>ives that:

$$
\begin{array} { r l } & { \left[ g \cdot ( k \cdot V ) \right] _ { \sigma _ { g } ( \sigma _ { k } ( i ) ) , t } \left( \tau _ { g } ^ { t } ( \tau _ { k } ^ { t } x _ { t } ) \right) } \\ & { \quad \quad = ( k \cdot V ) _ { \sigma _ { k } ( i ) , t } ( \tau _ { k } ^ { t } x _ { t } ) } \\ & { \quad \quad = V _ { i , t } ( x _ { t } ) . } \end{array}
$$

Th<sub>e</sub> fi<sub>rs</sub>t <sub>equa</sub>lit<sub>y</sub> <sub>app</sub>li<sub>es</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>pro</sub>fil<sub>e</sub> $k \cdot V$ un<sup>d</sup>er <sub>�</sub>, <sub>a</sub>t th<sub>e coor</sub>di<sub>na</sub>t<sub>e</sub> $( \sigma _ { k } ( i ) , t , \tau _ { k } ^ { t } x _ { t } )$ . The second a<sub>pp</sub>lies it to � under �. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the combined transformation <sub>�</sub>� once <sub>g</sub>ives that:

$$
[ ( g k ) \cdot V ] _ { \sigma _ { g k } ( i ) , t } ( \tau _ { g k } ^ { t } x _ { t } ) = V _ { i , t } ( x _ { t } ) .
$$

B<sub>ecause</sub> $\sigma _ { g k } = \sigma _ { g } \circ \sigma _ { k }$ <sub>an</sub>d $\tau _ { g k } ^ { t } = \tau _ { g } ^ { t } \circ \tau _ { k } ^ { t } ;$ <sub>,</sub> th<sub>e</sub> t<sub>wo ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons eva</sub>l<sub>ua</sub>t<sub>e</sub> $g \cdot ( k \cdot V )$ <sub>an</sub>d $( g k ) \cdot V$ at t<sup>h</sup>e same coor<sup>d</sup>inate. Bijectivity ensures that these coordinates cover ever<sub>y</sub> a<sub>g</sub>ent and in<sub>p</sub>ut at each time <sup>ste</sup>p, p<sup>rovin</sup>g $g \cdot ( k \cdot V ) = ( g k ) \cdot V$ <sub>.</sub> F<sub>o</sub>r th<sub>e</sub> id<sub>e</sub>ntit<sub>y</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> $\sigma _ { e }$ <sub>an</sub>d $\tau _ { e } ^ { t }$ are identit<sub>y</sub> ma<sub>p</sub>s, so E<sub>q</sub>uation (4) <sub>g</sub>ives $e \cdot V = V .$ <sub>.</sub> Thi<sub>s</sub> <sub>proves</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue-pro</sub>fil<sub>e</sub> <sub>ac</sub>ti<sub>on</sub> l<sub>aws</sub> <sub>an</sub>d <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> □

## B.2 Complete proof for the strictness example

Proof of Proposition 5.3. Let $N = \{ 1 , 2 \}$ , set $T = 2 ,$ <sub>c</sub>h<sub>oose</sub> <sup>an</sup>y $\gamma \in ( 0 , 1 ]$ <sub>, an</sub>d l<sub>e</sub>t ${ \cal { S } } = \{ L , M , H \}$ <sub>.</sub> B<sub>o</sub>th <sub>agen</sub>t<sub>s</sub> h<sub>ave ac</sub>ti<sub>on an</sub>d observation sets {0, 1}, and the initial state distribution is uniform. T<sub>a</sub>k<sub>e</sub> $\mathscr { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ t<sub>o</sub> b<sub>e</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>group</sub> <sub>o</sub>f <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons o</sub>f th<sub>ese</sub> fi<sub>n</sub>it<sub>e se</sub>t<sub>s.</sub> L<sub>e</sub>t <sub>� exc</sub>h<sub>ange</sub> th<sub>e</sub> t<sub>wo agen</sub>t<sub>s</sub> <sub>w</sub>hil<sub>e</sub> fi<sub>x</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>ac</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> Th<sub>us,</sub>

$$
g \cdot ( a _ { 1 } , a _ { 2 } ) = ( a _ { 2 } , a _ { 1 } ) , \qquad g \cdot ( o _ { 1 } , o _ { 2 } ) = ( o _ { 2 } , o _ { 1 } ) .
$$

For a joint action $\boldsymbol { a } = \left( a _ { 1 } , a _ { 2 } \right)$ <sub>, a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub>d <sub>as</sub> $a _ { 1 } a _ { 2 }$ <sub>, ma</sub>k<sub>e</sub> th<sub>e</sub> transition distribution inde<sub>p</sub>endent of the current state b<sub>y</sub> settin<sub>g</sub>

$$
P ( s ^ { \prime } \mid s , a ) = p _ { a } ( s ^ { \prime } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } s \in S .
$$

I<sub>n nex</sub>t<sub>-s</sub>t<sub>a</sub>t<sub>e or</sub>d<sub>er</sub> $\left( L , M , H \right)$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { c c c c c c } { { a } } & { { } } & { { \hat { p } _ { a } } } & { { } } & { { a } } & { { \hat { p } _ { a } } } \\ { { 0 0 } } & { { ( 1 / 2 , 1 / 3 , 1 / 6 ) } } & { { } } & { { 0 1 } } & { { ( 1 / 3 , 1 / 3 , 1 / 3 ) } } \\ { { 1 0 } } & { { ( 5 / 1 2 , 1 / 6 , 5 / 1 2 ) } } & { { } } & { { 1 1 } } & { { ( 1 / 6 , 1 / 3 , 1 / 2 ) . } } \end{array}
$$

M<sub>a</sub>k<sub>e</sub> th<sub>e o</sub>b<sub>serva</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e successor s</sub>t<sub>a</sub>t<sub>e</sub> <sup>b</sup>ut not on t<sup>h</sup>e joint action:

$$
Z ( o ^ { \prime } \mid s ^ { \prime } , a ) = q _ { s ^ { \prime } } ( o ^ { \prime } ) .
$$

A<sup>bb</sup>reviating a joint o<sup>b</sup>servation $( o _ { 1 } ^ { \prime } , o _ { 2 } ^ { \prime } )$ as $o _ { 1 } ^ { \prime } o _ { 2 } ^ { \prime } ,$ , use t<sup>h</sup>e joint-<sub>o</sub>b<sub>serva</sub>ti<sub>on or</sub>d<sub>er</sub> $( 0 0 , 0 1 , 1 0 , 1 1 )$ <sub>an</sub>d <sub>se</sub>t

$$
\begin{array} { c } { { q _ { L } = ( 3 / 8 , 1 / 4 , 1 / 4 , 1 / 8 ) , ~ q _ { M } = ( 1 / 4 , 1 / 4 , 1 / 4 , 1 / 4 ) , } } \\ { { { } } } \\ { { q _ { H } = ( 1 / 8 , 1 / 4 , 1 / 4 , 3 / 8 ) . } } \end{array}
$$

Fi<sub>na</sub>ll<sub>y,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
c ( L ) = 0 , \qquad c ( M ) = 1 , \qquad c ( H ) = 2 ,
$$

<sub>an</sub>d

$$
r _ { i } ( s , a , s ^ { \prime } ) = c ( s ) + c ( s ^ { \prime } ) + 2 a _ { i } + 1 \{ a _ { 1 } = a _ { 2 } \} .
$$

E<sub>ve</sub>r<sub>y</sub> initi<sub>a</sub>l<sub>,</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>, a</sub>nd <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s pos</sub>iti<sub>ve.</sub>

For eac<sup>h</sup> joint action $^ { a , }$ d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>successor-s</sub>t<sub>a</sub>t<sub>e</sub> <sub>re-</sub> <sub>war</sub>d <sub>componen</sub>t <sub>an</sub>d th<sub>e marg</sub>i<sub>na</sub>li<sub>ze</sub>d <sub>nex</sub>t<sub>-o</sub>b<sub>serva</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u-</sub> tion b<sub>y</sub>

$$
m _ { a } = \sum _ { s ^ { \prime } } \bar { p } _ { a } ( s ^ { \prime } ) c ( s ^ { \prime } ) , \qquad \bar { q } _ { a } ( o ^ { \prime } ) = \sum _ { s ^ { \prime } } \bar { p } _ { a } ( s ^ { \prime } ) q _ { s ^ { \prime } } ( o ^ { \prime } ) .
$$

For exam<sub>p</sub>le<sub>,</sub>

$$
\begin{array} { l } { { m _ { 0 1 } = \frac { 1 } { 3 } c ( L ) + \frac { 1 } { 3 } c ( M ) + \frac { 1 } { 3 } c ( H ) = 1 , } } \\ { { \qquad \bar { q } _ { 0 1 } = \frac { 1 } { 3 } \bigl ( q _ { L } + q _ { M } + q _ { H } \bigr ) = \bigl ( 1 / 4 , 1 / 4 , 1 / 4 , 1 / 4 \bigr ) . } } \\ { { \qquad \quad } } \end{array}
$$

T<sup>h</sup>e same ca<sup>l</sup>cu<sup>l</sup>ations <sup>f</sup>or t<sup>h</sup>e ot<sup>h</sup>er joint actions give

$$
m _ { 0 0 } = \frac { 2 } { 3 } , \qquad m _ { 0 1 } = m _ { 1 0 } = 1 , \qquad m _ { 1 1 } = \frac { 4 } { 3 } ,
$$

<sub>an</sub>d

$$
\begin{array} { r l } & { \bar { q } _ { 0 0 } = ( 7 / 2 4 , 1 / 4 , 1 / 4 , 5 / 2 4 ) , } \\ & { \bar { q } _ { 0 1 } = \bar { q } _ { 1 0 } = ( 1 / 4 , 1 / 4 , 1 / 4 , 1 / 4 ) , } \\ & { \bar { q } _ { 1 1 } = ( 5 / 2 4 , 1 / 4 , 1 / 4 , 7 / 2 4 ) . } \end{array}
$$

Writin<sub>g</sub> $u _ { i } ( a ) = 2 a _ { i } + 1 \{ a _ { 1 } = a _ { 2 } \}$ <sub>,</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on �, w</sub>h<sub>ose</sub> a<sub>g</sub>ent <sub>p</sub>ermutation swa<sub>p</sub>s the two a<sub>g</sub>ents<sub>,</sub> therefore satisfies

$$
\begin{array} { r } { m _ { g \cdot a } = m _ { a } , \qquad \bar { q } _ { g \cdot a } ( g \cdot o ^ { \prime } ) = \bar { q } _ { a } ( o ^ { \prime } ) , \qquad u _ { \sigma _ { g } ( i ) } ( g \cdot a ) = u _ { i } ( a ) . } \end{array}
$$

$\mathrm { A t } ~ t = 1$ <sub>,</sub> we se<sub>p</sub>arate the current-state contribution $c ( s )$ f<sub>rom</sub> th<sub>e</sub> remainin<sub>g</sub> ex<sub>p</sub>ected reward b<sub>y</sub> definin<sub>g</sub> an ex<sub>p</sub>ression<sub>,</sub> such that:

$$
W _ { i } ^ { \pi } ( h ^ { 1 } ) = \sum _ { a \in \mathcal { A } } \pi ^ { 1 } ( a \mid h ^ { 1 } ) [ u _ { i } ( a ) + m _ { a } ] .
$$

Th<sub>e</sub> B<sub>e</sub>ll<sub>man recurs</sub>i<sub>on a</sub>t $t = 1$ <sub>re</sub>d<sub>uces</sub> t<sub>o</sub>

$$
V _ { i , 1 } ^ { \pi } ( s , h ^ { 1 } ) = c ( s ) + W _ { i } ^ { \pi } ( h ^ { 1 } ) .
$$

The <sub>p</sub>olic<sub>y</sub> transformation and the three dis<sub>p</sub>la<sub>y</sub>ed identities <sub>g</sub>ive

$$
W _ { \sigma _ { g } ( i ) } ^ { g \cdot \pi } ( g \cdot h ^ { 1 } ) = W _ { i } ^ { \pi } ( h ^ { 1 } ) .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> <sub>c</sub>h<sub>ang</sub>i<sub>ng</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> b<sub>y</sub> $b = g \cdot a$ i<sub>n</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e ma</sub>t<sub>c</sub>h<sub>es</sub> the <sub>p</sub>olic<sub>y</sub> <sub>p</sub>robabilit<sub>y,</sub> $u _ { \sigma _ { q } ( i ) } ( b )$ <sub>,</sub> <sub>an</sub>d $m _ { b }$ <sub>w</sub>ith th<sub>e</sub>i<sub>r</sub> th<sub>ree</sub> <sub>source</sub> terms on the ri<sub>g</sub>ht-hand side. At the initial time ste<sub>p</sub> $t = 0 ,$ , ex<sub>p</sub>an<sup>d</sup>- in<sub>g</sub> E<sub>q</sub>uation (1) <sub>y</sub>ields the followin<sub>g</sub> e<sub>q</sub>uation:

$$
\begin{array} { r l } & { V _ { i , 0 } ^ { \pi } ( s , h ^ { 0 } ) = \displaystyle \sum _ { a \in \mathcal { A } } \pi ^ { 0 } ( a \mid h ^ { 0 } ) \Big [ c ( s ) + u _ { i } ( a ) + ( 1 + \gamma ) m _ { a } } \\ & { \qquad \quad + \gamma \displaystyle \sum _ { o ^ { \prime } \in O } \bar { q } _ { a } ( o ^ { \prime } ) W _ { i } ^ { \pi } ( h ^ { 0 } , a , o ^ { \prime } ) \Big ] . } \end{array}
$$

Th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t $1 + \gamma$ <sup>a</sup>pp<sup>ears because</sup> $c ( s ^ { \prime } )$ <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es once</sub> t<sub>o</sub> th<sub>e</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e rewar</sub>d <sub>an</sub>d <sub>once, w</sub>ith di<sub>scoun</sub>t $\gamma ,$ t<sub>o</sub> th<sub>e va</sub>l<sub>ue a</sub>t $t = 1 ,$ . In t<sup>h</sup>e trans<sup>f</sup>orme<sup>d</sup> expression, use t<sup>h</sup>e <sup>b</sup>ijections $b = g \cdot a$ <sub>an</sub>d $\bar { o } ^ { \prime } \ = \ g \cdot \ o ^ { \prime }$ . The <sub>p</sub>olic<sub>y</sub> transformation matches the action <sub>p</sub>robabilit<sub>y</sub>. The identities for <sub>��</sub> and $m _ { a }$ <sub>p</sub>reserve t<sup>h</sup>e terms $u _ { i } ( a )$ <sub>an</sub>d $( 1 + \gamma ) m _ { a } ,$ <sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. The identities for $\bar { q } _ { a }$ <sub>an</sub>d $W _ { i } ^ { \pi }$ <sub>p</sub>reserve t<sup>h</sup>e r<sub>e</sub>m<sub>a</sub>inin<sub>g</sub> t<sub>e</sub>rm $\begin{array} { r } { \gamma \sum _ { o ^ { \prime } \in O } \bar { q } _ { a } ( o ^ { \prime } ) W _ { i } ^ { \pi } ( h ^ { 0 } , a , o ^ { \prime } ) } \end{array}$ <sup>. Conse</sup>q<sup>uentl</sup>y, <sup>ever</sup>y t<sub>erm</sub> <sub>o</sub>f th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>sum</sub> <sub>equa</sub>l<sub>s</sub> it<sub>s</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>source</sub> t<sub>erm,</sub> which <sub>p</sub>roves the value identit<sub>y</sub> at the initial time ste<sub>p</sub>. To<sub>g</sub>ether with the identit<sub>y</sub> at $t = 1$ <sub>an</sub>d th<sub>e zero va</sub>l<sub>ues a</sub>t $t = 2 ,$ <sub>,</sub> this <sub>g</sub>ives th<sub>a</sub>t<sub>:</sub>

$$
V _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi } ( \tau _ { g } ^ { t } x _ { t } ) = V _ { i , t } ^ { \pi } ( x _ { t } )
$$

<sup>f</sup>or every po<sup>l</sup>icy, agent, time step, an<sup>d</sup> va<sup>l</sup>i<sup>d</sup> state–joint-<sup>h</sup>istory pair. Th<sub>us</sub> $g \in G _ { M } ^ { \mathrm { e v a l } }$ . Its state ma<sub>p</sub> is the identit<sub>y,</sub> so it also <sub>p</sub>reserves the initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub>

N<sub>ever</sub>th<sub>e</sub>l<sub>ess,</sub> f<sub>or</sub> $a = ( 0 , 1 )$ <sub>an</sub>d $s ^ { \prime } = M ,$

$$
P ( M \mid s , g \cdot a ) = \frac { 1 } { 6 } \neq \frac { 1 } { 3 } = P ( M \mid s , a ) ,
$$

so $g \not \in \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } )$ . It remains to r<sub>u</sub>le o<sub>u</sub>t a diferent com<sub>p</sub>onent<sub>w</sub>ise environment automor<sub>p</sub>hism that exchan<sub>g</sub>es the a<sub>g</sub>ents.

Su<sub>pp</sub>ose<sub>,</sub> for a contradiction<sub>,</sub> that there exists a com<sub>p</sub>onentwise env<sup>i</sup>ronmen<sup>t</sup> au<sup>t</sup>omorp<sup>hi</sup>sm �e<sup>th</sup>a<sup>t</sup> exc<sup>h</sup>anges <sup>th</sup>e <sup>t</sup>wo agen<sup>t</sup>s. <sup>L</sup>e<sup>t</sup> � denote its state <sub>p</sub>ermutation<sub>,</sub> and allow arbitrar<sub>y</sub> binar<sub>y</sub> action and <sub>o</sub>b<sub>serva</sub>ti<sub>on re</sub>l<sub>a</sub>b<sub>e</sub>li<sub>ngs.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>e</sub> $r _ { i } ( s , a , s ^ { \prime } ) = c ( s ) + c ( s ^ { \prime } ) + u _ { i } ( a )$ into the reward-<sub>p</sub>reservation e<sub>q</sub>uation and move the source terms t<sub>o</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>.</sub> Thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>equa</sub>ti<sub>ons:</sub>

$$
d ( s ) + d ( s ^ { \prime } ) + u _ { \sigma _ { \widetilde { q } } ( i ) } ( \widetilde { g } \cdot a ) - u _ { i } ( a ) = 0 , \qquad d ( s ) = c ( \psi ( s ) ) - c ( s ) .
$$

Holdin<sub>g</sub> $s ^ { \prime } , a , i$ fi<sub>xe</sub>d <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t $d ( s )$ is constant. Because $\psi$ on<sup>l</sup><sub>y</sub> <sub>permu</sub>t<sub>es</sub> th<sub>e</sub> th<sub>ree</sub> <sub>s</sub>t<sub>a</sub>t<sub>es,</sub>

$$
\sum _ { s } d ( s ) = \sum _ { s } c ( \psi ( s ) ) - \sum _ { s } c ( s ) = 0 .
$$

A <sub>co</sub>n<sub>s</sub>t<sub>a</sub>nt f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n <sub>w</sub>ith z<sub>e</sub>r<sub>o su</sub>m i<sub>s</sub> z<sub>e</sub>r<sub>o.</sub> H<sub>e</sub>n<sub>ce,</sub> $d ( s ) = 0$ f<sub>or</sub> every state, an<sup>d</sup> injectivity o<sup>f</sup> � gives $\psi = \mathrm { i d } _ { S }$ . Reward <sub>p</sub>reservation <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> <sub>an</sub> <sub>equa</sub>ti<sub>on,</sub> <sub>suc</sub>h th<sub>a</sub>t<sub>:</sub>

$$
u _ { \sigma _ { \widetilde g } ( i ) } ( \widetilde g \cdot a ) = u _ { i } ( a ) .
$$

W<sub>r</sub>it<sub>e</sub> th<sub>e</sub> t<sub>wo</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-</sub>l<sub>a</sub>b<sub>e</sub>l <sub>maps</sub> <sub>as</sub> $f _ { 1 }$ <sub>an</sub>d $f _ { 2 \cdot } \mathrm { A }$ <sup>b</sup>ijection o<sup>f</sup> th<sub>e</sub> bi<sub>nary ac</sub>ti<sub>on se</sub>t i<sub>s e</sub>ith<sub>er</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y or</sub> th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l fli<sub>p.</sub> U<sub>n</sub>d<sub>er</sub> <sup>the</sup> <sup>a</sup>g<sup>ent</sup> <sup>swa</sup>p, $[ { \widetilde { g } } \cdot a ] _ { 2 } = f _ { 1 } ( a _ { 1 } )$ <sub>an</sub>d $[ \widetilde { g } \cdot a ] _ { 1 } = f _ { 2 } ( a _ { 2 } ) . \operatorname { I f } f _ { 1 }$ fli<sub>ps</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s, c</sub>h<sub>oose source agen</sub>t 1 <sub>an</sub>d $a _ { 1 } = 0$ <sub>.</sub> Th<sub>en,</sub>

$$
u _ { 2 } ( { \widetilde { g } } \cdot a ) = 2 f _ { 1 } ( 0 ) + 1 \{ f _ { 2 } ( a _ { 2 } ) = f _ { 1 } ( 0 ) \} \geq 2 ,
$$

<sub>w</sub>h<sub>ereas</sub> $u _ { 1 } ( a ) \ \leq \ 1 _ { }$ <sub>, a con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> Th<sub>us</sub> $f _ { 1 }$ i<sub>s</sub> th<sub>e</sub> id<sub>e</sub>ntit<sub>y.</sub> R<sub>e</sub>- <sub>p</sub>eatin<sub>g</sub> the same ar<sub>g</sub>ument for source a<sub>g</sub>ent 2 shows that $f _ { 2 }$ i<sub>s</sub> also the identit<sub>y</sub>. Observation relabelin<sub>g</sub> cannot re<sub>p</sub>air the transiti<sub>o</sub>n mi<sub>s</sub>m<sub>a</sub>t<sub>c</sub>h b<sub>ecause</sub> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n <sub>equa</sub>ti<sub>o</sub>n d<sub>oes</sub> n<sub>o</sub>t <sub>co</sub>nt<sub>a</sub>in <sub>a</sub>n observation ma<sub>p</sub>. Therefore no com<sub>p</sub>onentwise environment aut<sub>omorp</sub>hi<sub>sm</sub> <sub>exc</sub>h<sub>anges</sub> th<sub>e</sub> <sub>agen</sub>t<sub>s,</sub> <sub>w</sub>hil<sub>e</sub> $g \in G _ { M } ^ { \mathrm { e v a l } }$ d<sub>oes.</sub> T<sub>oge</sub>th<sub>er</sub> with Theorem 5.2<sub>,</sub> which <sub>g</sub>ives $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { M } ^ { \mathrm { e v a l } }$ <sub>,</sub> this <sub>p</sub>roves $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { M } ^ { \mathrm { e v a l } }$ . The two orbit <sub>p</sub>artitions are conse<sub>q</sub>uentl<sub>y</sub>

$$
\mathcal { P } ( \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } ) ) = \{ \{ 1 \} , \{ 2 \} \} , \qquad \mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } ) = \{ \{ 1 , 2 \} \} .
$$

All tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>,</sub> <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>,</sub> <sub>a</sub>nd initi<sub>a</sub>l-<sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>p</sub>r<sub>o</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>a</sub>r<sub>e</sub> <sub>pos</sub>itive<sub>,</sub> so the se<sub>p</sub>aration is not an artifact of histories that no <sub>p</sub>olic<sub>y</sub> <sub>can</sub> <sub>reac</sub>h<sub>.</sub> □

## B.3 Subgroup, orbit, and normal-form facts

Proposition B.2 (Policy-evaluation symmetry subgroup and return eqivariance under the initial distribution). The set $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ is a subgroup o $f \mathscr { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ . Let

$$
G _ { \mathcal M } ^ { { \mathrm { e v a l } } , \rho _ { 0 } } = \left\{ g \in G _ { \mathcal M } ^ { { \mathrm { e v a l } } } : \rho _ { 0 } \bigl ( \phi _ { g } ^ { S } ( s ) \bigr ) = \rho _ { 0 } ( s ) f o r e \nu e r y s \in S \right\} .
$$

Then $G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } }$ is a subgroup, and every $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } }$ satisfies

$$
J _ { \sigma _ { g } ( i ) } ( g \cdot \pi ) = J _ { i } ( \pi ) \qquad ( i \in N , \pi \in \Pi _ { \mathrm { d e c } } ) .
$$

Proof. We verif<sub>y</sub> the three sub<sub>g</sub>rou<sub>p</sub> <sub>p</sub>ro<sub>p</sub>erties directl<sub>y</sub>. The identit<sub>y</sub> transformation satisfies the evaluation identit<sub>y</sub>. If $\mathbf { \Phi } _ { y , k } \in$ $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sup>l</sup><sub>,</sub> th<sub>en</sub>

$$
\begin{array} { r } { \mathcal { E } _ { \mathcal { M } } ( ( g k ) \cdot \pi ) = \mathcal { E } _ { \mathcal { M } } ( g \cdot ( k \cdot \pi ) ) \phantom { x x x x x x x x x x x x x x x x x x x x } } \\ { = g \cdot \mathcal { E } _ { \mathcal { M } } ( k \cdot \pi ) = ( g k ) \cdot \mathcal { E } _ { \mathcal { M } } ( \pi ) . } \end{array}
$$

F<sub>or</sub> i<sub>nverses,</sub> fi<sub>x</sub> $g \in G _ { M } ^ { \mathrm { e v a l } }$ an<sup>d</sup> an<sub>y</sub> $\pi \in \Pi _ { \mathrm { d e c } }$ <sub>.</sub> Sin<sub>ce</sub> $g ^ { - 1 } \in \mathcal { R } _ { \mathrm { a d m } } ( M )$ Pr<sub>opos</sub>iti<sub>o</sub>n B<sub>.</sub>1 <sub>e</sub>n<sub>su</sub>r<sub>es</sub> th<sub>a</sub>t $g ^ { - 1 } \cdot \pi \in \Pi _ { \mathrm { d e c } }$ . A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the definin<sub>g</sub> evaluation identit<sub>y</sub> for $g$ to this <sub>p</sub>olic<sub>y</sub> and usin<sub>g</sub> $g \cdot \left( g ^ { - 1 } \cdot \pi \right) = \pi$ <sub>g</sub>ives the followin<sub>g</sub> e<sub>q</sub>uation:

$$
\mathcal { E } _ { M } ( \pi ) = g \cdot \mathcal { E } _ { M } ( g ^ { - 1 } \cdot \pi ) .
$$

<sup>A</sup>pp<sup>l</sup>y $g ^ { - 1 }$ t<sub>o</sub> b<sub>o</sub>th <sub>va</sub>l<sub>ue</sub> <sub>pro</sub>fil<sub>es.</sub> Th<sub>e</sub> <sub>va</sub>l<sub>ue-pro</sub>fil<sub>e</sub> <sub>ac</sub>ti<sub>on</sub> l<sub>aw</sub> $g ^ { - 1 }$ $( g \cdot V ) = V$ then <sub>g</sub>ives the followin<sub>g</sub> e<sub>q</sub>uation:

$$
\mathcal { E } _ { M } ( g ^ { - 1 } \cdot \pi ) = g ^ { - 1 } \cdot \mathcal { E } _ { M } ( \pi ) .
$$

Since <sub>�</sub> was arbitrar<sub>y,</sub> this is the definin<sub>g</sub> evaluation identit<sub>y</sub> for $g ^ { - 1 } , \mathrm { s o } g ^ { - 1 } \in G _ { M } ^ { \mathrm { e v a l } }$

Th<sub>e a</sub>dditi<sub>ona</sub>l <sub>�0-preserva</sub>ti<sub>on con</sub>diti<sub>on</sub> i<sub>s a</sub>l<sub>so c</sub>l<sub>ose</sub>d <sub>un</sub>d<sub>er</sub> the <sub>g</sub>rou<sub>p</sub> o<sub>p</sub>erations. The identit<sub>y</sub> state ma<sub>p</sub> <sub>p</sub>reserves $\rho _ { 0 } . \operatorname { I f } g$ <sub>an</sub>d � <sub>preserve</sub> it<sub>,</sub> th<sub>en we</sub> h<sub>ave</sub> th<sub>a</sub>t<sub>:</sub>

$$
\rho _ { 0 } \big ( \phi _ { g k } ^ { S } ( s ) \big ) = \rho _ { 0 } \big ( \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ) ) \big ) = \rho _ { 0 } ( \phi _ { k } ^ { S } ( s ) ) = \rho _ { 0 } ( s ) .
$$

<sup>If</sup> <sub>�</sub> <sub>p</sub>reserves $\rho _ { 0 } { \mathrm { : } }$ <sub>,</sub> s<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> $u ~ = ~ ( \phi _ { g } ^ { S } ) ^ { - 1 } ( s )$ into $\rho _ { 0 } ( \phi _ { g } ^ { S } ( u ) ) =$ $\rho _ { 0 } ( u )$ <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t $g ^ { - 1 }$ also <sub>p</sub>reser<sub>v</sub>es it. Hence $G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } }$ <sup>i</sup>s a su<sup>b</sup><sub>g</sub>rou<sub>p</sub>. Fi<sub>na</sub>ll<sub>y,</sub> l<sub>e</sub>t $g \in G _ { M } ^ { \mathrm { e v a l } , \rho _ { 0 } }$ <sub>.</sub> Pr<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n means $\rho _ { 0 } ( \phi _ { g } ^ { S } ( s ) ) = \rho _ { 0 } ( s )$ . To a<sub>pp</sub>l<sub>y</sub> evaluation e<sub>q</sub>uivariance<sub>,</sub> we r<sub>e</sub>ind<sub>e</sub>x th<sub>e</sub> initi<sub>a</sub>l-<sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>su</sub>m <sub>us</sub>in<sub>g</sub> $s ^ { \prime } = \phi _ { q } ^ { S } ( s )$ . Bijectivity ensures that ever<sub>y</sub> state is counted exactl<sub>y</sub> once<sub>,</sub> and initial-distribution <sub>p</sub>reservation ensures that its <sub>p</sub>robabilit<sub>y</sub> wei<sub>g</sub>ht is unchan<sub>g</sub>ed. The empty initia<sup>l</sup> joint <sup>h</sup>istory is <sup>fi</sup>xe<sup>d b</sup>y t<sup>h</sup>e trans<sup>f</sup>ormation, so we <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> id<sub>en</sub>tit<sub>y:</sub>

$$
\begin{array} { l } { { \displaystyle J _ { \sigma _ { g } ( i ) } ( g \cdot \pi ) = \sum _ { s ^ { \prime } \in S } \rho _ { 0 } ( s ^ { \prime } ) V _ { \sigma _ { g } ( i ) , 0 } ^ { g \cdot \pi } ( s ^ { \prime } , ( ) ^ { N } ) } } \\ { { \displaystyle \qquad = \sum _ { s \in S } \rho _ { 0 } ( \phi _ { g } ^ { S } ( s ) ) V _ { \sigma _ { g } ( i ) , 0 } ^ { g \cdot \pi } ( \phi _ { g } ^ { S } ( s ) , ( ) ^ { N } ) } } \\ { { \displaystyle \qquad = \sum _ { s \in S } \rho _ { 0 } ( s ) V _ { i , 0 } ^ { \pi } ( s , ( ) ^ { N } ) = J _ { i } ( \pi ) . } } \end{array}
$$

Proposition B.3 (Orbit facts). For every subgroup

$$
H \leq \mathcal { R } _ { \mathrm { a d m } } ( M ) ,
$$

define $i \sim _ { H } j$ when some $g \in H$ satisfies $\sigma _ { q } ( i ) = j$ . Then $\sim _ { H }$ is an equivalence relation. $I f H \leq K$ , then $\mathcal { P } ( H ) \preceq \mathcal { P } ( K )$ .

P<sub>roof.</sub> W<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t $\sim _ { H }$ is an e<sub>q</sub>uivalence relation b<sub>y</sub> verif<sub>y</sub>in<sub>g</sub> reflexivit<sub>y</sub>, s<sub>y</sub>mmetr<sub>y</sub>, and transitivit<sub>y</sub> [16, Cha<sub>p</sub>. 1.2].

F<sub>or re</sub>fl<sub>ex</sub>i<sub>v</sub>it<sub>y, we mus</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t $i \ \sim _ { H }$ <sup>�</sup> <sup>for</sup> <sup>ever</sup>y <sup>a</sup>g<sup>ent</sup> <sup>�.</sup> B<sub>ecause</sub> � i<sub>s a su</sub>b<sub>g</sub>r<sub>oup,</sub> it <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> th<sub>e</sub> id<sub>e</sub>ntit<sub>y</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n <sub>�.</sub> Thi<sub>s</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> l<sub>eaves every agen</sub>t l<sub>a</sub>b<sub>e</sub>l <sub>unc</sub>h<sub>ange</sub>d<sub>, so</sub> $\sigma _ { e } ( i ) =$ �<sub>.</sub> Th<sub>us,</sub> $e \in H$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng con</sub>diti<sub>on</sub> f<sub>or</sub> $i \sim _ { H } i$

<sup>For</sup> <sup>s</sup>y<sup>mmetr</sup>y, <sup>su</sup>pp<sup>ose</sup> <sup>that</sup> $i \sim _ { H } j .$ B<sub>y</sub> definition<sub>,</sub> some $g \in H$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\sigma _ { q } ( i ) = j$ . Since � is a s<sub>u</sub>b<sub>g</sub>ro<sub>up,</sub> the in<sub>v</sub>erse transformation $g ^ { - 1 }$ also belon<sub>g</sub>s to �. Its a<sub>g</sub>ent <sub>p</sub>ermutation reverses that of $g ,$ <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
\sigma _ { g ^ { - 1 } } ( j ) = \sigma _ { g ^ { - 1 } } ( \sigma _ { g } ( i ) ) = \sigma _ { g ^ { - 1 } g } ( i ) = \sigma _ { e } ( i ) = i .
$$

Th<sub>us,</sub> $g ^ { - 1 } \in H$ maps agent � <sup>b</sup>ac<sup>k</sup> to agent $i ,$ <sub>w</sub>hi<sub>c</sub>h <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> $j \sim _ { H }$ �. This <sub>p</sub>roves s<sub>y</sub>mmetr<sub>y</sub> of the relation.

F<sub>o</sub>r tr<sub>a</sub>n<sub>s</sub>iti<sub>v</sub>it<sub>y,</sub> if $\sigma _ { g } ( i ) = j$ <sub>an</sub>d $\sigma _ { k } ( j ) = \ell$ f<sub>or</sub> $g , k \in H$ <sub>,</sub> th<sub>en</sub> $k g \in H$ <sub>an</sub>d $\sigma _ { k g } ( i ) = \ell$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $i \sim _ { H } \ell .$ Th<sub>e</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ence</sub> <sub>c</sub>l<sub>asses</sub> <sub>are</sub> <sub>p</sub>recisel<sub>y</sub> the �-orbits<sub>,</sub> which form the <sub>p</sub>artition $\mathcal { P } ( H )$

$\operatorname { I f } H \leq K$ <sub>,</sub> th<sub>en</sub> <sub>we</sub> h<sub>ave</sub>

$$
\{ \sigma _ { g } ( i ) : g \in H \} \subseteq \{ \sigma _ { g } ( i ) : g \in K \} ,
$$

for e<sub>v</sub>er<sub>y</sub> a<sub>g</sub>ent �. Th<sub>u</sub>s<sub>,</sub> e<sub>v</sub>er<sub>y</sub> �-orbit is contained in a �-orbit. Since these orbits are the <sub>p</sub>arts of the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>artitions<sub>,</sub> <sup>ever</sup>y p<sup>art o</sup> $: { \mathcal { P } } ( H )$ i<sub>s co</sub>nt<sub>a</sub>in<sub>e</sub>d in <sub>a pa</sub>rt <sub>o</sub> $\mathcal { P } ( K )$ <sub>.</sub> B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>par</sub>titi<sub>on re</sub>fi<sub>nemen</sub>t<sub>,</sub> $\mathcal P ( H ) \preceq \mathcal P ( K )$ □

To check the definition in a static settin<sub>g,</sub> we show that <sub>p</sub>olic<sub>y</sub>- <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>symme</sub>t<sub>ry</sub> <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> <sub>payo</sub>f <sub>preserva</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow-</sub> i<sub>ng</sub> <sub>one-s</sub>t<sub>age</sub> <sub>norma</sub>l<sub>-</sub>f<sub>orm</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng.</sub>

Proposition B.4 (Normal-form special case). Embed a finite normal-form game $\Gamma = ( N , ( \mathcal { A } _ { i } ) _ { i } , ( u _ { i } ) _ { i } )$ as a horizon-one POSG with one state, singleton observations, deterministic dynamics, and reward $r _ { i } ( s _ { \star } , a , s _ { \star } ) = u _ { i } ( a )$ . For an admissible transformation �, evaluation equivariance for every mixed product policy is equivalent to

$$
u _ { \sigma _ { g } ( i ) } ( g \cdot a ) = u _ { i } ( a ) \qquad f o r e \nu e r y i a n d p u r e \ : p r o f i l e a .
$$

Every admissible transformation automatically preserves the unique initial state–joint-history distribution.

Proof<sub>.</sub> L<sub>e</sub>t $U _ { i } ( \pi )$ d<sub>eno</sub>t<sub>e agen</sub>t �’<sub>s expec</sub>t<sub>e</sub>d <sub>payo</sub>f <sub>un</sub>d<sub>er</sub> th<sub>e</sub> mixed <sub>p</sub>roduct <sub>p</sub>olic<sub>y �</sub>. If the dis<sub>p</sub>la<sub>y</sub>ed <sub>p</sub>a<sub>y</sub>of identit<sub>y</sub> holds<sub>,</sub> rein-<sup>d</sup>exing joint actions an<sup>d</sup> using t<sup>h</sup>e pro<sup>d</sup>uct-po<sup>l</sup>icy trans<sup>f</sup>ormation <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r l } { { U _ { \sigma _ { g } } ( i ) \left( g \cdot \pi \right) } \displaystyle = \sum _ { a \in \mathcal { A } } { u _ { \sigma _ { g } ( i ) } \left( g \cdot a \right) \prod _ { j \in N } { \pi _ { j } \left( a _ { j } \right) } } \displaystyle } \\ { \displaystyle = \sum _ { a \in \mathcal { A } } { u _ { i } ( a ) \prod _ { j \in N } { \pi _ { j } \left( a _ { j } \right) } } = { U _ { i } } ( \pi ) . } \end{array}
$$

Conversel<sub>y,</sub> the <sub>p</sub>olic<sub>y</sub> domain contains ever<sub>y</sub> deterministic <sub>p</sub>roduct <sub>p</sub>o<sup>li</sup>c<sub>y</sub> $\delta _ { a } ,$ <sub>an</sub>d $g \cdot \delta _ { a } = \delta _ { q \cdot a \cdot }$ <sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n <sub>equ</sub>i<sub>va</sub>ri<sub>a</sub>n<sub>ce</sub> <sub>g</sub>i<sub>ves</sub>

$$
u _ { \sigma _ { g } ( i ) } ( g \cdot a ) = U _ { \sigma _ { g } ( i ) } ( \delta _ { g \cdot a } ) = U _ { i } ( \delta _ { a } ) = u _ { i } ( a ) .
$$

Fi<sub>na</sub>ll<sub>y,</sub> l<sub>e</sub>t $g \in \mathcal { R } _ { \mathrm { a d m } } ( M )$ be arbitrar<sub>y</sub>. Since $S = \{ s _ { \star } \}$ , its <sup>b</sup>ijective state ma<sub>p</sub> must sat<sup>i</sup>s<sup>f</sup><sub>y</sub> $\phi _ { g } ^ { S } ( s _ { \star } ) ~ = ~ s _ { \star }$ . Ever<sub>y</sub> initial local histor<sub>y</sub> is em<sub>p</sub>t<sub>y,</sub> so there are no actions or observations to transform. Permutin<sub>g</sub> the a<sub>g</sub>ents onl<sub>y</sub> rearran<sub>g</sub>es identical em<sub>p</sub>t<sub>y</sub> histories<sub>,</sub> <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $g \cdot ( \ u ) ^ { N } = ( \ u ) ^ { N }$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
\begin{array} { c } { { \tau _ { g } ^ { 0 } ( s _ { \star } , ( \cdot ) ^ { N } ) = ( \phi _ { g } ^ { S } ( s _ { \star } ) , g \cdot ( \cdot ) ^ { N } ) } } \\ { { { } } } \\ { { = ( s _ { \star } , ( \cdot ) ^ { N } ) . } } \end{array}
$$

The initial distribution assi<sub>g</sub>ns <sub>p</sub>robabilit<sub>y</sub> one to this sin<sub>g</sub>le <sub>p</sub>air. Because the <sub>p</sub>air is fixed<sub>,</sub> its distribution is <sub>p</sub>reserved b<sub>y</sub> ever<sub>y</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> □

## C Proofs for Componentwise Environment Automorphisms

Proposition C.1 (Subgroup property of componentwise environment automorphisms). The set in Definition 5.1 satisfies

$$
\mathrm { A u t _ { e n v } } ( M ) \leq \mathcal { R } _ { \mathrm { a d m } } ( M ) .
$$

The componentwise environment automorphisms that preserve the initial distribution, such that:

$$
\left\{ g \in \mathrm { A u t } _ { \mathrm { e n v } } ( M ) : \rho _ { 0 } \bigl ( \phi _ { g } ^ { S } ( s ) \bigr ) = \rho _ { 0 } ( s ) f o r e \nu e r y s \in S \right\} ,
$$

also form a subgroup.

Proof. The identit<sub>y p</sub>reserves each com<sub>p</sub>onent. Su<sub>pp</sub>ose <sub>�</sub> and � satisf<sub>y</sub> E<sub>q</sub>uations $( 5 ) - ( 7 )$ . For transitions<sub>,</sub>

$$
\begin{array} { r l } & { P ( \phi _ { g k } ^ { S } ( s ^ { \prime } ) \mid \phi _ { g k } ^ { S } ( s ) , ( g k ) \cdot a ) } \\ & { = P ( \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ^ { \prime } ) ) \mid \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ) ) , g \cdot ( k \cdot a ) ) } \\ & { = P ( \phi _ { k } ^ { S } ( s ^ { \prime } ) \mid \phi _ { k } ^ { S } ( s ) , k \cdot a ) = P ( s ^ { \prime } \mid s , a ) . } \end{array}
$$

The first <sub>p</sub>reser<sub>v</sub>ation e<sub>qu</sub>alit<sub>y</sub> remo<sub>v</sub>es the o<sub>u</sub>ter transformation ${ \mathit { g } } ,$ <sub>an</sub>d th<sub>e secon</sub>d <sub>removes</sub> �<sub>.</sub> Th<sub>e o</sub>b<sub>serva</sub>ti<sub>on an</sub>d <sub>rewar</sub>d <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons</sub> h<sub>ave</sub> th<sub>e same</sub> t<sub>wo-s</sub>t<sub>ep proce</sub>d<sub>ure.</sub> F<sub>or o</sub>b<sub>serva</sub>ti<sub>ons,</sub>

$$
\begin{array} { r l } & { Z ( ( g k ) \cdot o ^ { \prime } \mid \phi _ { g k } ^ { S } ( s ^ { \prime } ) , ( g k ) \cdot a ) } \\ & { = Z ( g \cdot ( k \cdot o ^ { \prime } ) \mid \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ^ { \prime } ) ) , g \cdot ( k \cdot a ) ) } \\ & { = Z ( k \cdot o ^ { \prime } \mid \phi _ { k } ^ { S } ( s ^ { \prime } ) , k \cdot a ) = Z ( o ^ { \prime } \mid s ^ { \prime } , a ) . } \end{array}
$$

For rewards<sub>,</sub> trackin<sub>g</sub> the a<sub>g</sub>ent coordinate <sub>g</sub>ives

$$
\begin{array} { r l } & { r _ { \sigma _ { g k } ( i ) } ( \phi _ { g k } ^ { S } ( s ) , ( g k ) \cdot a , \phi _ { g k } ^ { S } ( s ^ { \prime } ) ) } \\ & { = r _ { \sigma _ { g } ( \sigma _ { k } ( i ) ) } ( \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ) ) , g \cdot ( k \cdot a ) , \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ^ { \prime } ) ) ) } \\ & { = r _ { \sigma _ { k } ( i ) } ( \phi _ { k } ^ { S } ( s ) , k \cdot a , \phi _ { k } ^ { S } ( s ^ { \prime } ) ) = r _ { i } ( s , a , s ^ { \prime } ) . } \end{array}
$$

Th<sub>us</sub> th<sub>e se</sub>t i<sub>s c</sub>l<sub>ose</sub>d <sub>un</sub>d<sub>er compos</sub>iti<sub>on.</sub>

F<sub>or</sub> i<sub>nverse c</sub>l<sub>osure,</sub> fi<sub>x</sub> $g \in \mathrm { A u t _ { e n v } } ( \mathcal { M } )$ <sub>.</sub> Sin<sub>ce</sub> $\mathscr { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ i<sub>s a</sub> g<sup>rou</sup>p, $g ^ { - 1 }$ i<sub>s an a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t it <sub>a</sub>l<sub>so</sub> <sub>p</sub>reserves the three model com<sub>p</sub>onents b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the identities <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> k<sub>nown</sub> f<sub>or</sub> <sub>�</sub> t<sub>o</sub> <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> i<sub>npu</sub>t<sub>s.</sub>

For transitions, E<sub>q</sub>uation (5) holds for ever<sub>y</sub> <sub>p</sub>air of states $u , u ^ { \prime }$ an<sup>d</sup> joint action <sup>�</sup>:

$$
P ( \phi _ { g } ^ { S } ( u ^ { \prime } ) \mid \phi _ { g } ^ { S } ( u ) , g \cdot b ) = P ( u ^ { \prime } \mid u , b ) .
$$

F<sub>o</sub>r <sub>a</sub>rbitr<sub>a</sub>r<sub>y</sub> $\mathit { \Delta } _ { s , s ^ { \prime } , }$ <sub>,</sub> <sub>an</sub>d $^ { a , }$ <sub>c</sub>h<sub>oose</sub>

$$
u = \phi _ { g ^ { - 1 } } ^ { S } ( s ) , \qquad u ^ { \prime } = \phi _ { g ^ { - 1 } } ^ { S } ( s ^ { \prime } ) , \qquad b = g ^ { - 1 } \cdot a .
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> <sub>�</sub> cancels these inverse transformations<sub>,</sub> so $\phi _ { g } ^ { S } ( u ) = s ,$ $\begin{array} { r } { \phi _ { q } ^ { S } ( u ^ { \prime } ) = s ^ { \prime } } \end{array}$ <sub>,</sub> <sub>an</sub>d $g \cdot b = a$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>o</sub>n int<sub>o</sub> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n id<sub>e</sub>ntit<sub>y</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r } { P ( s ^ { \prime } \mid s , a ) = P ( \phi _ { g ^ { - 1 } } ^ { S } ( s ^ { \prime } ) \mid \phi _ { g ^ { - 1 } } ^ { S } ( s ) , g ^ { - 1 } \cdot a ) , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>are</sub> <sub>preserve</sub>d <sub>un</sub>d<sub>er</sub> $g ^ { - 1 }$

For observations, E<sub>q</sub>uation (6) <sub>g</sub>ives

$$
Z ( g \cdot v ^ { \prime } \mid \phi _ { g } ^ { S } ( u ^ { \prime } ) , g \cdot b ) = Z ( v ^ { \prime } \mid u ^ { \prime } , b ) .
$$

U<sub>se</sub> th<sub>e same</sub> $u ^ { \prime }$ <sub>an</sub>d $^ { b , }$ <sub>an</sub>d <sub>se</sub>t $\boldsymbol { v } ^ { \prime } = \boldsymbol { g } ^ { - 1 } \cdot \boldsymbol { o } ^ { \prime }$ <sup>f</sup>or an ar<sup>b</sup>itrary joint <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> $\boldsymbol { o ^ { \prime } } .$ <sub>.</sub> Sin<sub>ce</sub> $g \cdot v ^ { \prime } = o ^ { \prime }$ <sub>,</sub> <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>o</sub>n <sub>g</sub>i<sub>ves</sub>

$$
Z ( o ^ { \prime } \mid s ^ { \prime } , a ) = Z ( g ^ { - 1 } \cdot o ^ { \prime } \mid \phi _ { g ^ { - 1 } } ^ { S } ( s ^ { \prime } ) , g ^ { - 1 } \cdot a ) .
$$

For rewards, E<sub>q</sub>uation (7) holds for ever<sub>y</sub> a<sub>g</sub>ent $j \colon$

$$
r _ { \sigma _ { g } ( j ) } ( \phi _ { g } ^ { S } ( u ) , g \cdot b , \phi _ { g } ^ { S } ( u ^ { \prime } ) ) = r _ { j } ( u , b , u ^ { \prime } ) .
$$

U<sub>se</sub> th<sub>e same</sub> $u , u ^ { \prime }$ <sub>, an</sub>d $b ,$ <sub>an</sub>d <sub>c</sub>h<sub>oose</sub> $j = \sigma _ { q ^ { - 1 } } ( i )$ f<sub>or</sub> <sub>an</sub> <sub>ar</sub>bit<sub>rary</sub> a<sub>g</sub>ent �. The a<sub>g</sub>ent <sub>p</sub>ermutations also cancel<sub>,</sub> <sub>g</sub>ivin<sub>g</sub> $\sigma _ { g } ( j ) = i .$ <sub>.</sub> Th<sub>us,</sub> s<sub>u</sub>bstit<sub>u</sub>tion into the re<sub>w</sub>ard identit<sub>y</sub> <sub>g</sub>i<sub>v</sub>es

$$
r _ { i } ( s , a , s ^ { \prime } ) = r _ { \sigma _ { g ^ { - 1 } } ( i ) } ( \phi _ { g ^ { - 1 } } ^ { S } ( s ) , g ^ { - 1 } \cdot a , \phi _ { g ^ { - 1 } } ^ { S } ( s ^ { \prime } ) ) .
$$

B<sub>ecause</sub> $s , s ^ { \prime } , a , o ^ { \prime } ,$ <sub>, an</sub>d � <sub>were ar</sub>bit<sub>rary,</sub> th<sub>ese</sub> th<sub>ree</sub> id<sub>en</sub>titi<sub>es s</sub>h<sub>ow</sub> th<sub>a</sub>t $g ^ { - 1 }$ <sub>p</sub>reserves the transition <sub>p</sub>robabilities<sub>,</sub> observation <sub>p</sub>robabiliti<sub>es, an</sub>d <sub>rewar</sub>d f<sub>unc</sub>ti<sub>ons.</sub> H<sub>ence,</sub> $g ^ { - 1 } \in \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } )$ . To<sub>g</sub>ether with the identit<sub>y</sub> and com<sub>p</sub>osition <sub>p</sub>ro<sub>p</sub>erties <sub>p</sub>roved above<sub>,</sub> this <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t $\mathrm { A u t _ { e n v } } ( M )$ <sup>i</sup>s a su<sup>b</sup><sub>g</sub>rou<sub>p</sub>.

It r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> t<sub>o</sub> im<sub>pose</sub> <sub>p</sub>r<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> Th<sub>e</sub> identit<sub>y</sub> state ma<sub>p p</sub>reserves <sub>�0</sub>. If <sub>�</sub> and � <sub>p</sub>reserve $\rho _ { 0 } ,$ <sub>,</sub> th<sub>en</sub>

$$
\rho _ { 0 } \big ( \phi _ { g k } ^ { S } ( s ) \big ) = \rho _ { 0 } \big ( \phi _ { g } ^ { S } ( \phi _ { k } ^ { S } ( s ) ) \big ) = \rho _ { 0 } \big ( \phi _ { k } ^ { S } ( s ) \big ) = \rho _ { 0 } ( s ) .
$$

<sup>For inverse closure</sup>, <sup>su</sup>pp<sup>ose</sup> � p<sup>reserves</sup> $\rho _ { 0 }$ <sub>.</sub> Thi<sub>s means</sub> th<sub>a</sub>t f<sub>or</sub> <sup>ever</sup>y <sup>state</sup> $u \in S _ { \mathrm { ~ ~ } }$

$$
\rho _ { 0 } \big ( \phi _ { g } ^ { S } ( u ) \big ) = \rho _ { 0 } ( u ) .
$$

For an arbitrar<sub>y</sub> state <sub>�,</sub> choose $u = \phi _ { g ^ { - 1 } } ^ { S } ( s ) = ( \phi _ { g } ^ { S } ) ^ { - 1 } ( s )$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> this state into the <sub>p</sub>reservation identit<sub>y</sub> for <sub>�</sub> <sub>g</sub>ives

$$
\rho _ { 0 } \big ( \phi _ { g } ^ { S } ( \phi _ { g ^ { - 1 } } ^ { S } ( s ) ) \big ) = \rho _ { 0 } \big ( \phi _ { g ^ { - 1 } } ^ { S } ( s ) \big ) .
$$

Sin<sub>ce</sub> $\phi _ { g } ^ { S } ( \phi _ { q ^ { - 1 } } ^ { S } ( s ) ) = s$ <sub>,</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> <sub>s</sub>i<sub>mp</sub>lifi<sub>es</sub> t<sub>o</sub> $\rho _ { 0 } ( s )$ <sub>,</sub> <sub>y</sub>i<sub>e</sub>ld<sub>-</sub> <sup>i</sup>n<sub>g</sub>

$$
\rho _ { 0 } ( s ) = \rho _ { 0 } { \big ( } \phi _ { g ^ { - 1 } } ^ { S } ( s ) { \big ) } .
$$

Thi<sub>s</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>every</sub> <sub>�,</sub> <sub>so</sub> $g ^ { - 1 }$ a<sup>l</sup>so <sub>p</sub>reserves $\rho _ { 0 } .$ <sub>.</sub> T<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> <sub>componen</sub>t<sub>w</sub>i<sub>se c</sub>l<sub>osure a</sub>l<sub>rea</sub>d<sub>y prove</sub>d<sub>,</sub> th<sub>ese ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e secon</sub>d <sub>su</sub>b<sub>group c</sub>l<sub>a</sub>i<sub>m.</sub> □

Proof of Theorem 5.2. Pro<sub>p</sub>osition C.1 alread<sub>y</sub> establishes that $\mathrm { A u t _ { e n v } } ( M )$ is a sub<sub>g</sub>rou<sub>p</sub> of $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ <sub>.</sub> W<sub>e</sub> <sub>now</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>every</sub> element of this sub<sub>g</sub>rou<sub>p</sub> is a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>. Fix $g \in \operatorname { A u t } _ { \mathrm { e n v } } ( M )$ <sub>.</sub> W<sub>e</sub> <sub>prove</sub> b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>a</sub>t

$$
V _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi } ( \tau _ { g } ^ { t } x _ { t } ) = V _ { i , t } ^ { \pi } ( x _ { t } )
$$

<sup>f</sup>or ever<sub>y</sub> a<sub>g</sub>ent �, <sub>p</sub>o<sup>li</sup>c<sub>y</sub> �, t<sup>i</sup>me ste<sub>p</sub> �, an<sup>d</sup> $x _ { t } \in X _ { t } . \mathrm { A t } \ t = T$ <sub>,</sub> b<sub>o</sub>th <sub>co</sub>rr<sub>espo</sub>ndin<sub>g va</sub>l<sub>ues a</sub>r<sub>e</sub> z<sub>e</sub>r<sub>o.</sub> A<sub>ssu</sub>m<sub>e</sub> th<sub>e</sub> id<sub>e</sub>ntit<sub>y</sub> h<sub>o</sub>ld<sub>s a</sub>t tim<sub>e</sub> <sup>ste</sup>p $t + 1$ <sub>.</sub> F<sub>o</sub>r $t < T$ <sub>, expan</sub>d th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>va</sub>l<sub>ue:</sub>

$$
\begin{array} { l } { { { \cal { V } } _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi } ( \tau _ { g } ^ { t } ( s , h ^ { t } ) ) } } \\ { { { } } } \\ { { { \displaystyle = \sum _ { b , \bar { s } ^ { \prime } , \bar { \sigma } ^ { \prime } } ( g \cdot \pi ) ^ { t } ( b \mid g \cdot h ^ { t } ) P ( \bar { s } ^ { \prime } \mid \phi _ { g } ^ { S } ( s ) , b ) Z ( \bar { \sigma } ^ { \prime } \mid \bar { s } ^ { \prime } , b ) } } } \\ { { { } } } \\ { { { \displaystyle ~ \cdot \left[ r _ { \sigma _ { g } ( i ) } ( \phi _ { g } ^ { S } ( s ) , b , \bar { s } ^ { \prime } ) + \gamma V _ { \sigma _ { g } ( i ) , t + 1 } ^ { g \cdot \pi } ( \bar { s } ^ { \prime } , g \cdot h ^ { t } , b , \bar { o } ^ { \prime } ) \right] . } } } \end{array}
$$

Reindex the sum usin<sub>g</sub>

$$
b = g \cdot a , \qquad \bar { s } ^ { \prime } = \phi _ { g } ^ { S } ( s ^ { \prime } ) , \qquad \bar { o } ^ { \prime } = g \cdot o ^ { \prime } .
$$

Eac<sup>h</sup> trans<sup>f</sup>ormation is <sup>b</sup>ijective, so every trip<sup>l</sup>e $( b , \bar { s } ^ { \prime } , \bar { o } ^ { \prime } )$ corres<sub>p</sub>onds to exactl<sub>y</sub> one tri<sub>p</sub>le $( a , s ^ { \prime } , o ^ { \prime } )$

For t<sup>h</sup>e joint-action pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity, t<sup>h</sup>e pro<sup>d</sup>uct-po<sup>l</sup>icy <sup>d</sup>e<sup>fi</sup>nition and the local <sub>p</sub>olic<sub>y</sub>-transformation rule in E<sub>q</sub>uation (3) <sub>g</sub>ive

$$
\begin{array} { r l } & { ( g \cdot \pi ) ^ { t } ( g \cdot a \mid g \cdot h ^ { t } ) } \\ & { = \displaystyle \prod _ { j \in N } ( g \cdot \pi ) _ { \sigma _ { g } ( j ) } ^ { t } \left( \phi _ { g , j } ^ { \mathcal { R } } ( a _ { j } ) \mid \Phi _ { g , j } ^ { \mathcal { H } , t } ( h _ { j } ^ { t } ) \right) } \\ & { = \displaystyle \prod _ { j \in N } \pi _ { j } ^ { t } ( a _ { j } \mid h _ { j } ^ { t } ) = \pi ^ { t } ( a \mid h ^ { t } ) . } \end{array}
$$

Th<sub>e</sub> fi<sub>rs</sub>t <sub>equa</sub>lit<sub>y</sub> i<sub>n</sub>d<sub>exes</sub> th<sub>e</sub> <sub>pro</sub>d<sub>uc</sub>t b<sub>y</sub> $\sigma _ { g } ( j )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>v</sub>i<sub>s</sub>it<sub>s</sub> <sub>eac</sub>h a<sub>g</sub>ent exact<sup>l</sup><sub>y</sub> once <sup>b</sup>ecause $\sigma _ { g }$ i<sub>s</sub> <sub>a</sub> <sub>pe</sub>rm<sub>u</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>.</sub>

Sin<sub>ce</sub> $g \in \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } )$ , Definition 5.1 <sub>p</sub>rovides E<sub>q</sub>uations (5)–(7). A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> these identities to the substituted transition<sub>,</sub> observation<sub>,</sub> <sub>an</sub>d <sub>rewar</sub>d t<sub>erms</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } & { P ( \phi _ { g } ^ { S } ( s ^ { \prime } ) \mid \phi _ { g } ^ { S } ( s ) , g \cdot a ) = P ( s ^ { \prime } \mid s , a ) , } \\ & { \quad Z ( g \cdot o ^ { \prime } \mid \phi _ { g } ^ { S } ( s ^ { \prime } ) , g \cdot a ) = Z ( o ^ { \prime } \mid s ^ { \prime } , a ) , } \\ & { \quad r _ { \sigma _ { g } ( i ) } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) = r _ { i } ( s , a , s ^ { \prime } ) . } \end{array}
$$

To a<sub>pp</sub>l<sub>y</sub> the induction h<sub>yp</sub>othesis to the remainin<sub>g</sub> value term<sub>,</sub> <sup>f</sup>or eac<sup>h</sup> joint action � an<sup>d</sup> next joint o<sup>b</sup>servation $o ^ { \prime }$ <sub>,</sub> write

$$
h ^ { t + 1 } : = ( h ^ { t } , a , o ^ { \prime } ) .
$$

This is the joint histor<sub>y</sub> of len<sub>g</sub>th � + 1 obtained b<sub>y</sub> a<sub>pp</sub>endin<sub>g</sub> � and $o ^ { \prime }$ to $h ^ { t }$ . The histor<sub>y</sub> transformation a<sub>pp</sub>lies the same action and observation ma<sub>p</sub>s to ever<sub>y</sub> entr<sub>y</sub> and <sub>p</sub>reserves their order. Thus<sub>,</sub> transformin<sub>g</sub> this extended histor<sub>y</sub> <sub>g</sub>ives

$$
g \cdot h ^ { t + 1 } = ( g \cdot h ^ { t } , g \cdot a , g \cdot o ^ { \prime } ) .
$$

R<sub>eca</sub>ll f<sub>rom</sub> S<sub>ec</sub>ti<sub>on</sub> 3 th<sub>a</sub>t

$$
\tau _ { g } ^ { t } ( s , h ^ { t } ) = ( \phi _ { g } ^ { S } ( s ) , g \cdot h ^ { t } ) ,
$$

so $\tau _ { g } ^ { t }$ trans<sup>f</sup>orms <sup>b</sup>ot<sup>h</sup> t<sup>h</sup>e state an<sup>d</sup> t<sup>h</sup>e joint <sup>h</sup>istory at time step �. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> this definition at time ste<sub>p</sub> � + 1 and usin<sub>g</sub> the <sub>p</sub>recedin<sub>g</sub> <sup>h</sup>istory i<sup>d</sup>entity, t<sup>h</sup>e next state–joint-<sup>h</sup>istory pair in t<sup>h</sup>e su<sup>b</sup>stitute<sup>d</sup> <sub>sum</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub>

$$
\begin{array} { r l } & { \big ( \phi _ { g } ^ { S } ( s ^ { \prime } ) , ( g \cdot h ^ { t } , g \cdot a , g \cdot o ^ { \prime } ) \big ) } \\ & { = \big ( \phi _ { g } ^ { S } ( s ^ { \prime } ) , g \cdot h ^ { t + 1 } \big ) } \\ & { = \tau _ { g } ^ { t + 1 } \big ( s ^ { \prime } , h ^ { t + 1 } \big ) . } \end{array}
$$

The induction h<sub>yp</sub>othesis at time ste<sub>p</sub> $t + 1$ <sub>,</sub> a<sub>pp</sub>lied to $( s ^ { \prime } , h ^ { t + 1 } )$ , now <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r l } & { V _ { \sigma _ { g } ( i ) , t + 1 } ^ { g \cdot \pi } \big ( \phi _ { g } ^ { S } ( s ^ { \prime } ) , g \cdot h ^ { t + 1 } \big ) } \\ & { = V _ { \sigma _ { g } ( i ) , t + 1 } ^ { g \cdot \pi } \big ( \tau _ { g } ^ { t + 1 } ( s ^ { \prime } , h ^ { t + 1 } ) \big ) } \\ & { = V _ { i , t + 1 } ^ { \pi } \big ( s ^ { \prime } , h ^ { t + 1 } \big ) . } \end{array}
$$

Usin<sub>g</sub> the <sub>p</sub>recedin<sub>g</sub> identities in se<sub>q</sub>uence<sub>,</sub> the Bellman ex<sub>p</sub>ansion b<sub>ecomes</sub>

$$
\begin{array} { r l } & { \mathcal { D } _ { \phi , i , i } ^ { ( 2 ) } ( x _ { i } ^ { 0 } , y _ { i } ^ { 0 } ) } \\ & { = \displaystyle \sum _ { j = 0 } ^ { \infty } \{ ( \phi , x _ { i } ^ { 0 } ) ^ { \top } ( \phi , x _ { j } ^ { 0 } ) \} } \\ & { \quad - \mathcal { D } _ { \phi , i } \phi ^ { ( 2 ) } ( x _ { i } ^ { 0 } , y _ { i } ^ { 0 } ) \phi _ { 0 , i } ^ { ( 3 ) } \phi _ { 0 , i } ^ { ( 4 ) } } \\ & { \quad - \mathcal { D } _ { \phi , i } ( x _ { i } ^ { 0 } , y _ { i } ^ { 0 } ) \phi _ { 0 , i } ^ { ( 4 ) } } \\ & { \quad - [ \phi , \phi , x _ { i } ^ { 0 } ] \phi _ { 0 , i } ^ { ( 4 ) } \phi _ { 0 , i } ^ { ( 4 ) } } \\ & { \quad + [ \phi , x _ { i } ^ { 0 } ] \phi _ { 0 , i } ^ { ( 4 ) } \phi _ { 0 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } } \\ & { \quad - \phi _ { 0 , i } ^ { ( 4 ) } \phi _ { 0 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } } \\ & { \quad - [ \phi , x _ { i } ^ { 0 } ] \phi _ { 1 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } } \\ & { \quad - [ \phi , \phi , x _ { i } ^ { 0 } ] \phi _ { 1 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } \phi _ { 1 , i } ^ { ( 4 ) } } \\ & { \quad - [ \phi , ( \phi , x _ { i } ^ { 0 } , y _ { i } ^ { 0 } ) \phi , ( x _ { i } ^ { 0 } , y _ { i } ^ { 0 } ) ] } \\ &  \quad - [  \end{array}
$$

Ste<sub>p</sub> (1) uses the bijective substitutions and the identit<sub>y</sub> for the next state–joint-history pair. Step (2) uses the joint-policy iden tit<sub>y</sub> derived from E<sub>q</sub>uation (3) and the com<sub>p</sub>onentwise identities in E<sub>qua</sub>ti<sub>o</sub>n<sub>s</sub> $( 5 ) - ( 7 )$ t<sub>o rep</sub>l<sub>ace</sub> th<sub>e ac</sub>ti<sub>on,</sub> t<sub>rans</sub>iti<sub>on, an</sub>d <sub>o</sub>b<sub>serva-</sub> tion <sub>p</sub>robabilities and the reward. Ste<sub>p</sub> (3) a<sub>pp</sub>lies the induction h<sub>yp</sub>othesis at time ste<sub>p</sub> � + 1 to the remainin<sub>g</sub> transformed value. Ste<sub>p</sub> (4) is E<sub>q</sub>uation (1), with $h ^ { t + 1 } = ( h ^ { t } , a , o ^ { \prime } )$ <sub>.</sub> Thi<sub>s comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> induction. The e<sub>q</sub>ualit<sub>y</sub> holds for ever<sub>y</sub> a<sub>g</sub>ent<sub>, p</sub>olic<sub>y,</sub> time ste<sub>p,</sub> and va<sup>l</sup>i<sup>d</sup> state–joint-<sup>h</sup>istory pair, so De<sup>fi</sup>nition 4.1 gives $g \in G _ { M } ^ { \mathrm { e v a l } }$

Since <sub>� w</sub>as arbitrar<sub>y,</sub> $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \subseteq G _ { M } ^ { \mathrm { e v a l } }$ <sup>.</sup> <sup>The</sup> <sup>sub</sup>g<sup>rou</sup>p p<sup>ro</sup>p<sup>-</sup> ert<sub>y</sub> from Pro<sub>p</sub>osition C.1 then <sub>g</sub>ives $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { M } ^ { \mathrm { e v a l } }$ <sub>.</sub> Pr<sub>opos</sub>ition B.3 <sub>y</sub>ields the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>artition refinement:

$$
\mathcal { P } ( \operatorname { A u t } _ { \mathrm { e n v } } ( \mathcal { M } ) ) \preceq \mathcal { P } _ { \mathrm { e v a l } } ( \mathcal { M } ) ,
$$

<sub>as c</sub>l<sub>a</sub>i<sub>me</sub>d<sub>.</sub>

□

## D Extension to Dec-POMDPs

Proposition D.1 (Existing common reward). $I f r _ { i } = r ^ { \mathrm { c o m } }$ for every agent, then, for every policy �, time step �, valid $x \in \mathcal { X } _ { t }$ , and agents �, �,

$$
V _ { i , t } ^ { \pi } ( x ) = V _ { j , t } ^ { \pi } ( x ) .
$$

The structural agent types nevertheless remain the orbits $o f G _ { M } ^ { \mathrm { e v a l } }$ ; in particular, equality ofthe value coordinates does not by itselfimply that this group acts transitively on the agents.

Proof. Fix an arbitrar<sub>y p</sub>olic<sub>y</sub> $\pi \in \Pi _ { \mathrm { d e c } }$ an<sup>d</sup> a<sub>g</sub>ents $i , j \in N$ . We <sub>prove equa</sub>lit<sub>y o</sub>f th<sub>e</sub>i<sub>r va</sub>l<sub>ues a</sub>t <sub>every va</sub>lid i<sub>npu</sub>t b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d ind<sub>uc</sub>ti<sub>o</sub>n<sub>.</sub> $\operatorname { A t } t = T$

$$
V _ { i , T } ^ { \pi } ( x ) = 0 = V _ { j , T } ^ { \pi } ( x ) \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in X _ { T } .
$$

<sup>S</sup>u<sub>pp</sub>ose t<sup>h</sup>at $V _ { i , t + 1 } ^ { \pi } ( x ^ { \prime } ) = V _ { j , t + 1 } ^ { \pi } ( x ^ { \prime } )$ <sup>f</sup>or ever<sub>y</sub> $x ^ { \prime } \in \mathcal { X } _ { t + 1 }$ <sub>.</sub> Fix <sub>a</sub>n<sub>y</sub> $( s , h ^ { t } ) \in  { \mathcal { X } } _ { t }$ <sub>,</sub> <sub>an</sub>d <sub>wr</sub>it<sub>e</sub> $h ^ { t + 1 } = ( h ^ { t } , a , o ^ { \prime } )$ in each summand. E<sub>q</sub>uation (1) and $r _ { i } = r _ { j } = r ^ { \mathrm { c o m } }$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { l } { { { \cal V } _ { i , t } ^ { \pi } ( s , h ^ { t } ) = \displaystyle \sum _ { a , s ^ { \prime } , o ^ { \prime } } \pi ^ { t } ( a \mid h ^ { t } ) P ( s ^ { \prime } \mid s , a ) Z ( o ^ { \prime } \mid s ^ { \prime } , a ) } } \\ { { \ ~ } } \\ { { { \cal ~ \cdot } \left[ r ^ { \mathrm { c o m } } ( s , a , s ^ { \prime } ) + \gamma V _ { i , t + 1 } ^ { \pi } ( s ^ { \prime } , h ^ { t + 1 } ) \right] } } \\ { { { \ } } } \\ { { { \displaystyle = \sum _ { a , s ^ { \prime } , o ^ { \prime } } \pi ^ { t } ( a \mid h ^ { t } ) P ( s ^ { \prime } \mid s , a ) Z ( o ^ { \prime } \mid s ^ { \prime } , a ) } } } \\ { { { \ ~ \ \cdot \left[ r ^ { \mathrm { c o m } } ( s , a , s ^ { \prime } ) + \gamma V _ { j , t + 1 } ^ { \pi } ( s ^ { \prime } , h ^ { t + 1 } ) \right] } } } \\ { { { \ } } } \\ { { { \displaystyle = V _ { j , t } ^ { \pi } ( s , h ^ { t } ) . } } } \end{array}
$$

The middle e<sub>qu</sub>alit<sub>y</sub> <sub>u</sub>ses the ind<sub>u</sub>ction h<sub>yp</sub>othesis at the next inp<sup>ut</sup> $( s ^ { \prime } , h ^ { t + 1 } )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h li<sub>es</sub> i<sub>n</sub> $\chi _ { t + 1 }$ <sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d l<sub>as</sub>t <sub>equa</sub>liti<sub>es</sub> <sub>use</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man recurs</sub>i<sub>on w</sub>ith th<sub>e common rewar</sub>d<sub>.</sub> Thi<sub>s proves</sub> th<sub>e</sub> ind<sub>uc</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>ep.</sub> Sin<sub>ce</sub> $\pi , i , j ,$ <sub>an</sub>d th<sub>e</sub> i<sub>npu</sub>t <sub>were</sub> <sub>ar</sub>bit<sub>rary,</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>ith <sub>a</sub>ll th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>quan</sub>tifi<sub>ers.</sub>

Th<sub>e or</sub>bit <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on o</sub>f <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>agen</sub>t t<sub>ypes</sub> f<sub>o</sub>ll<sub>ows</sub> di<sub>-</sub> <sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 4<sub>.</sub>2<sub>.</sub> T<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e va</sub>l<sub>ue equa</sub>lit<sub>y</sub> d<sub>oes</sub> not im<sub>p</sub>l<sub>y</sub> transitivit<sub>y,</sub> consider the one-sta<sub>g</sub>e common-reward exam<sub>p</sub>le from Section 6. S<sub>p</sub>ecif<sub>y</sub> $N = \{ 1 , 2 \} , T = 1 , S = \{ s _ { \star } \}$ $\mathcal { A } _ { 1 } = \mathcal { A } _ { 2 } = \{ 0 , 1 \}$ <sub>, an</sub>d $O _ { 1 } = O _ { 2 } = \left\{ o _ { \star } \right\}$ <sub>.</sub> Th<sub>e</sub> initi<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> $s _ { \star }$ <sub>w</sub>ith <sub>p</sub>robabilit<sub>y</sub> one<sub>,</sub> and the transition and observation distributions assign pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity one to t<sup>h</sup>e unique next state an<sup>d</sup> joint o<sup>b</sup>servation. Both a<sub>g</sub>ents receive

$$
r _ { 1 } ( s _ { \star } , a , s _ { \star } ) = r _ { 2 } ( s _ { \star } , a , s _ { \star } ) = a _ { 1 } .
$$

T<sub>a</sub>k<sub>e</sub> $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ t<sub>o</sub> b<sub>e</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>group o</sub>f t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma-</sub> tions<sub>,</sub> so a<sub>g</sub>ent exchan<sub>g</sub>es and arbitrar<sub>y</sub> binar<sub>y</sub> action relabelin<sub>g</sub>s <sub>are a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e.</sub>

Su<sub>pp</sub>ose<sub>,</sub> for a contradiction<sub>,</sub> that some $g \in G _ { M } ^ { \mathrm { e v a l } }$ exc<sup>h</sup>an<sub>g</sub>es th<sub>e</sub> t<sub>wo agen</sub>t<sub>s.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e o</sub>f $g \cdot a$ i<sub>s</sub> th<sub>en</sub> $\phi _ { q , 2 } ^ { \mathcal { R } } ( a _ { 2 } )$ . <sup>B</sup><sub>y</sub> Pr<sub>opos</sub>iti<sub>o</sub>n $8 . 4 , g$ must preserve t<sup>h</sup>e payo<sup>f</sup>s at every pure joint <sub>ac</sub>ti<sub>on.</sub> Th<sub>e common rewar</sub>d th<sub>ere</sub>f<sub>ore requ</sub>i<sub>res</sub>

$$
\phi _ { g , 2 } ^ { \mathcal { A } } ( a _ { 2 } ) = a _ { 1 } \qquad \mathrm { f o r } \mathrm { e v e r y } a _ { 1 } , a _ { 2 } \in \{ 0 , 1 \} .
$$

Takin<sub>g</sub> $( a _ { 1 } , a _ { 2 } ) = ( 0 , 0 )$ and (1, 0) would <sub>g</sub>ive both $\phi _ { g , 2 } ^ { \mathcal { R } } ( 0 ) = 0$ <sub>an</sub>d $\phi _ { q , 2 } ^ { \mathcal { A } } ( 0 ) = 1$ <sub>,</sub> a contradiction. Hence no <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>exc</sub>h<sub>anges</sub> th<sub>e</sub> t<sub>wo</sub> <sub>agen</sub>t<sub>s,</sub> <sub>an</sub>d

$$
{ \mathcal { P } } _ { \operatorname { e v a l } } ( M ) = { \big \{ } \{ 1 \} , \{ 2 \} { \big \} } .
$$

Th<sub>e va</sub>l<sub>ue coor</sub>di<sub>na</sub>t<sub>es are equa</sub>l b<sub>y</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>par</sub>t <sub>o</sub>f thi<sub>s proo</sub>f<sub>,</sub> b<sub>u</sub>t the <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub> is not transitive. This <sub>p</sub>roves th<sub>e</sub> fi<sub>na</sub>l <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> □

Proof of Theorem 6<sub>.</sub>1<sub>.</sub> Fix <sub>a</sub>n <sub>a</sub>rbitr<sub>a</sub>r<sub>y</sub> <sub>po</sub>li<sub>cy</sub> $\pi \in \Pi _ { \mathrm { d e c } }$ . We <sup>fi</sup>rst prove t<sup>h</sup>e weig<sup>h</sup>te<sup>d</sup>-va<sup>l</sup>ue i<sup>d</sup>entity <sup>f</sup>or every <sup>d</sup>erive<sup>d</sup> agent � <sub>an</sub>d <sub>every</sub> <sub>va</sub>lid i<sub>npu</sub>t b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>on</sub> th<sub>e</sub> ti<sub>me</sub> <sub>s</sub>t<sub>ep.</sub>

$\operatorname { A t } t = T$ <sub>,</sub> <sub>a</sub>ll <sub>va</sub>l<sub>ues</sub> i<sub>n</sub> b<sub>o</sub>th <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> <sub>zero.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> f<sub>or</sub> <sup>ever</sup>y $j \in N$ <sub>an</sub>d $x \in \mathcal { X } _ { T }$

$$
V _ { j , T } ^ { \pi ; M _ { w } } ( x ) = 0 = \sum _ { i \in N } w _ { i } 0 = \sum _ { i \in N } w _ { i } V _ { i , T } ^ { \pi ; M } ( x ) .
$$

Thi<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e</sub> b<sub>ase case.</sub>

For the induction ste<sub>p,</sub> let $t < T$ an<sup>d</sup> su<sub>pp</sub>ose t<sup>h</sup>at

$$
V _ { j , t + 1 } ^ { \pi ; M _ { w } } ( x ^ { \prime } ) = \sum _ { i \in N } w _ { i } V _ { i , t + 1 } ^ { \pi ; M } ( x ^ { \prime } )
$$

<sup>f</sup>or ever<sub>y</sub> $j \in N$ an<sup>d</sup> ever<sub>y</sub> $x ^ { \prime } \in \mathcal { X } _ { t + 1 }$ <sub>.</sub> W<sub>e mus</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e same</sub> id<sub>e</sub>ntit<sub>y</sub> <sub>a</sub>t tim<sub>e</sub> <sub>s</sub>t<sub>ep</sub> �<sub>.</sub> Fix <sub>a</sub>n<sub>y</sub> $j \in N$ <sub>an</sub>d $x = ( s , h ^ { t } ) \in X _ { t }$ <sub>.</sub> F<sub>or eac</sub>h action �<sub>,</sub> next state $s { ' } _ { \cdot }$ <sub>, an</sub>d <sub>nex</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> $o ^ { \prime }$ i<sub>n</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man sum,</sub> write

$$
h ^ { t + 1 } : = ( h ^ { t } , a , o ^ { \prime } ) , \qquad x ^ { \prime } : = ( s ^ { \prime } , h ^ { t + 1 } ) \in \chi _ { t + 1 } .
$$

Thus the induction h<sub>yp</sub>othesis a<sub>pp</sub>lies to ever<sub>y</sub> next in<sub>p</sub>ut in that <sub>sum.</sub> Th<sub>e</sub> t<sub>wo</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> h<sub>ave</sub> th<sub>e</sub> <sub>same</sub> <sub>po</sub>li<sub>cy</sub> d<sub>oma</sub>i<sub>n,</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons,</sub> <sub>an</sub>d di<sub>scoun</sub>t f<sub>ac</sub>t<sub>or.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>po</sub>li<sub>cy</sub> <sub>�</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>uses</sub> th<sub>e</sub> <sub>same</sub> f<sub>ac</sub>t<sub>ors</sub> $\pi ^ { t } ( a \mid h ^ { t } ) , P ( s ^ { \prime } \mid s , a )$ <sub>an</sub>d $Z ( o ^ { \prime } \mid s ^ { \prime } , a )$ i<sub>n</sub> b<sub>o</sub>th <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> Th<sub>e</sub> B<sub>e</sub>ll<sub>man recurs</sub>i<sub>on g</sub>i<sub>ves</sub> th<sub>e</sub> followin<sub>g</sub> se<sub>q</sub>uence of e<sub>q</sub>ualities:

$$
\begin{array} { r l } { \mathbb { E } _ { \rho } ^ { \rho \lambda \lambda ( \lambda _ { 1 } , \lambda _ { 2 } ) } } & { = \int _ { 0 } ^ { \infty } \lambda ( x _ { 1 } ) ^ { \lambda _ { 1 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  } \\ { \leq } & { \int _ { 0 } ^ { \infty } \lambda ( x _ { 1 } ) ^ { \lambda _ { 1 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  } \\ & { \ : - \{ \mathbf { 1 } _ { \rho } ( x _ { 1 } ) ^ { \lambda _ { 2 } } + \rho _ { 0 } ^ { \lambda _ { 1 } \lambda _ { 2 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  \} } \\ { \leq } & { \int _ { 0 } ^ { \infty } \sum _ { \rho \in \{ \rho \} }  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  } \\ & { \ : - \{ \mathbf { 1 } _ { \rho } ( x _ { 1 } ) ^ { \lambda _ { 1 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  + \rho _ { 0 } ^ { \lambda _ { 1 } \lambda _ { 2 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  } \\ { \leq } & { \int _ { 0 } ^ { \infty } \lambda ( x _ { 1 } ) ^ { \lambda _ { 1 } } \exp  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  } \\ &  \ : - \sum _ { \rho \in \{ \rho \} }  \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho } \mathbf { 1 } _ { \rho }  \exp \ \end{array}
$$

Ste<sub>p</sub> (1) is E<sub>q</sub>uation (1) in $M _ { w } ,$ w<sup>h</sup>ere agent �<sup>’</sup>s rewar<sup>d</sup> is $r ^ { w }$ . Ste<sub>p</sub> (2) re<sub>p</sub><sup>l</sup>aces $r ^ { w }$ b<sub>y</sub> its definin<sub>g</sub> wei<sub>g</sub>hted sum and a<sub>pp</sub>lies the induction h<sub>yp</sub>othesis to $V _ { j , t + 1 } ^ { \pi ; M _ { w } } ( x ^ { \prime } )$ . Ste<sub>p</sub> (3) combines the two sums over � b<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ng eac</sub>h <sub>we</sub>i<sub>g</sub>ht <sub>�� over</sub> th<sub>e rewar</sub>d <sub>an</sub>d <sub>nex</sub>t <sub>va</sub>l<sub>ue.</sub> Ste<sub>p</sub> (4) interchan<sub>g</sub>es the finite sums over a<sub>g</sub>ents and over $( a , s ^ { \prime } , o ^ { \prime } )$ I<sub>n</sub> thi<sub>s</sub> <sub>s</sub>t<sub>ep,</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> <sub>are</sub> fi<sub>xe</sub>d<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>po</sub>li<sub>cy,</sub> t<sub>rans</sub>iti<sub>on,</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> f<sub>ac</sub>t<sub>ors</sub> <sub>are</sub> th<sub>e</sub> <sub>same</sub> f<sub>or</sub> <sub>every</sub> <sub>rewar</sub>d i<sub>n</sub>d<sub>ex</sub> �<sub>.</sub> Fi<sub>na</sub>ll<sub>y,</sub> in ste<sub>p</sub> (5), the inner sum for each � is E<sub>q</sub>uation (1) in M, so it e<sub>q</sub>uals $V _ { i , t } ^ { \pi ; M } ( s , h ^ { t } )$

Since � an<sup>d</sup> $( s , h ^ { t } )$ <sub>were ar</sub>bit<sub>rary,</sub> thi<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y a</sub>t time ste<sub>p</sub> � for ever<sub>y</sub> derived a<sub>g</sub>ent and valid in<sub>p</sub>ut. Startin<sub>g</sub> from $t = T$ and re<sub>p</sub>eatin<sub>g</sub> this induction ste<sub>p p</sub>roves it at $t = T - 1 , . . . , 0$ Th<sub>e po</sub>li<sub>cy � was a</sub>l<sub>so ar</sub>bit<sub>rary, so</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy as s</sub>t<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> th<sub>eorem.</sub>

W<sub>e</sub> <sub>nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>-preserv</sub>i<sub>ng</sub> <sub>e</sub>l<sub>emen</sub>t<sub>s</sub> <sub>o</sub>f $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ f<sub>orm</sub> a su<sup>b</sup><sub>g</sub>rou<sub>p</sub> o<sup>f</sup> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>.</sub> Sin<sub>ce</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ is alread<sub>y</sub> a <sub>g</sub>rou<sub>p,</sub> it sufices to verif<sub>y</sub> three <sub>p</sub>ro<sub>p</sub>erties of the set of wei<sub>g</sub>ht-<sub>p</sub>reservin<sub>g</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetries: (i) the identit<sub>y</sub> transformation belon<sub>g</sub>s to this set; (ii)

com<sub>p</sub>osin<sub>g</sub> an<sub>y</sub> two transformations in this set <sub>g</sub>ives another transformation in the set; and (iii) the inverse of ever<sub>y</sub> transformation in this set also belon<sub>g</sub>s to the set. The identit<sub>y</sub> <sub>p</sub>reserves <sub>�</sub>. $\operatorname { I f } g$ <sub>an</sub>d � <sub>preserve</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>s,</sub> th<sub>en</sub>

$$
w _ { \sigma _ { g k } ( i ) } = w _ { \sigma _ { g } ( \sigma _ { k } ( i ) ) } = w _ { \sigma _ { k } ( i ) } = w _ { i } .
$$

Finall<sub>y,</sub> $w _ { \sigma _ { q } ( i ) } = w _ { i }$ im<sub>p</sub>lies the corres<sub>p</sub>ondin<sub>g</sub> e<sub>q</sub>ualit<sub>y</sub> for $g ^ { - 1 }$ <sub>a</sub>ft<sub>er</sub> s<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> $i = \sigma _ { q } ^ { - 1 } ( k )$

N<sub>ow</sub> l<sub>e</sub>t $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>p</sub>reserve the wei<sub>g</sub>hts. For ever<sub>y</sub> derived a<sub>g</sub>ent $j ,$

$$
\begin{array} { l l l } { { \displaystyle V _ { \sigma _ { g } ( j ) , t } ^ { g \cdot \pi ; M _ { w } } ( \tau _ { g } ^ { t } x ) = \sum _ { i } w _ { \sigma _ { g } ( i ) } V _ { \sigma _ { g } ( i ) , t } ^ { g \cdot \pi ; M } ( \tau _ { g } ^ { t } x ) } } \\ { { \displaystyle ~ } } \\ { { \displaystyle ~ } } \end{array} = \sum _ { i } w _ { i } V _ { i , t } ^ { \pi ; M } ( x ) = V _ { j , t } ^ { \pi ; M _ { w } } ( x ) .
$$

The first e<sub>q</sub>ualit<sub>y</sub> a<sub>pp</sub>lies the wei<sub>g</sub>hted-value identit<sub>y</sub> and reindexes t<sup>h</sup>e a<sub>g</sub>ent sum <sup>b</sup><sub>y</sub> $k = \sigma _ { g } ( i )$ . The second uses wei<sub>g</sub>ht <sub>p</sub>reservation and source <sub>p</sub>olic<sub>y</sub>-evaluation e<sub>q</sub>uivariance. Since this holds for every po<sup>l</sup>icy, agent, time step, an<sup>d</sup> va<sup>l</sup>i<sup>d</sup> state–joint-<sup>h</sup>istory pair, $g \in G _ { M _ { w } } ^ { \mathrm { e v a l } } ;$ <sub>,</sub> <sub>p</sub>rovin<sub>g</sub> the asserted sub<sub>g</sub>rou<sub>p</sub> inclusion.

At th<sub>e componen</sub>t<sub>w</sub>i<sub>se</sub> l<sub>eve</sub>l<sub>,</sub> l<sub>e</sub>t $g \in \mathrm { A u t _ { e n v } } ( \mathcal { M } )$ <sub>p</sub>reserve t<sup>h</sup>e wei<sub>g</sub>hts. B<sub>y</sub> construction<sub>,</sub> reward a<sub>gg</sub>re<sub>g</sub>ation chan<sub>g</sub>es onl<sub>y</sub> the r<sub>ewa</sub>rd f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n<sub>s</sub>: th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $P$ <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> di<sub>s-</sub> trib<sub>u</sub>ti<sub>o</sub>n $Z$ <sub>are unc</sub>h<sub>ange</sub>d i<sub>n</sub> $\mathcal { M } _ { w }$ . Conse<sub>q</sub>uentl<sub>y,</sub> the <sub>p</sub>reservation id<sub>e</sub>ntiti<sub>es</sub> in E<sub>qua</sub>ti<sub>o</sub>n<sub>s</sub> $( 5 )$ <sub>an</sub>d $( 6 ) ,$ <sub>, w</sub>hi<sub>c</sub>h <sub>� a</sub>l<sub>rea</sub>d<sub>y sa</sub>ti<sub>s</sub>fi<sub>es</sub> i<sub>n</sub> $M ,$ <sub>a</sub>l<sub>so</sub> h<sub>o</sub>ld i<sub>n</sub> $M _ { w } .$ It remains to check <sub>p</sub>reservation of the a<sub>gg</sub>re<sub>g</sub>ate <sub>rewar</sub>d $r ^ { w }$ <sub>.</sub> R<sub>e</sub>i<sub>n</sub>d<sub>ex</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> <sub>rewar</sub>d <sub>sum</sub> b<sub>y</sub> $k = \sigma _ { g } ( i )$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r l } { r ^ { w } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) = \displaystyle \sum _ { i } w _ { \sigma _ { g } ( i ) } r _ { \sigma _ { g } ( i ) } ( \phi _ { g } ^ { S } ( s ) , g \cdot a , \phi _ { g } ^ { S } ( s ^ { \prime } ) ) } & { } \\ { \displaystyle } & { = \displaystyle \sum _ { i } w _ { i } r _ { i } ( s , a , s ^ { \prime } ) = r ^ { w } ( s , a , s ^ { \prime } ) . } \end{array}
$$

Th<sub>u</sub>s it is also a com<sub>p</sub>onent<sub>w</sub>ise en<sub>v</sub>ironment a<sub>u</sub>tomor<sub>p</sub>hism of $M _ { w } .$

The wei<sub>g</sub>ht-<sub>p</sub>reservin<sub>g</sub> com<sub>p</sub>onentwise environment automor-<sub>p</sub>hisms form a sub<sub>g</sub>rou<sub>p</sub> as well. The identit<sub>y</sub> <sub>p</sub>reserves the wei<sub>g</sub>hts. Cl<sub>osure un</sub>d<sub>er compos</sub>iti<sub>on</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>ca</sub>l<sub>cu</sub>l<sub>a-</sub> ti<sub>on</sub> f<sub>or</sub> $w _ { \sigma _ { g k } ( i ) }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>same</sub> <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> <sub>proves</sub> i<sub>nverse</sub> <sub>c</sub>l<sub>osure.</sub> Pr<sub>opos</sub>iti<sub>o</sub>n C<sub>.</sub>1 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t $\mathrm { A u t _ { e n v } } ( M )$ is a <sub>g</sub>rou<sub>p</sub>. The checks <sub>a</sub>b<sub>ove</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f it<sub>s</sub> <sub>we</sub>i<sub>g</sub>ht<sub>-preserv</sub>i<sub>ng</sub> <sub>e</sub>l<sub>emen</sub>t<sub>s</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> the identit<sub>y</sub> and is closed <sub>u</sub>nder com<sub>p</sub>osition and in<sub>v</sub>erses. These <sub>p</sub>ro<sub>p</sub>erties establish that this set is a sub<sub>g</sub>rou<sub>p</sub> of $\mathrm { A u t _ { e n v } } ( M )$ <sub>.</sub> Th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> <sub>componen</sub>t<sub>w</sub>i<sub>se</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>every</sub> <sub>e</sub>l<sub>emen</sub>t <sub>o</sub>f thi<sub>s su</sub>b<sub>group</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $\mathrm { A u t _ { e n v } } ( M _ { w } )$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\left\{ g \in \mathrm { A u t } _ { \mathrm { e n v } } ( \boldsymbol { M } ) : w _ { \sigma _ { g } ( i ) } = w _ { i } \mathrm { f o r } \mathrm { e v e r y } i \right\} \le \mathrm { A u t } _ { \mathrm { e n v } } ( \boldsymbol { M } _ { w } ) .
$$

Proposition D.2 (Expected returns under weighted aggregation). Under the construction ofTheorem $6 . 1 ,$ the models M and $\mathcal { M } _ { w }$ share the initial distribution $\rho _ { 0 }$ . For every policy $\pi \in \Pi _ { \mathrm { d e c } }$ and every agent $j \in N$

$$
J _ { j } ^ { M _ { w } } ( \pi ) = \sum _ { i \in N } w _ { i } J _ { i } ^ { M } ( \pi ) .
$$

This identity does not require any symmetry assumption.

Moreover, let $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ satisfy

$$
\begin{array} { r l r } { w _ { \sigma _ { g } ( i ) } = w _ { i } ~ } & { { } f o r e \nu e r y i \in N , } & { } \\ { \rho _ { 0 } ( \phi _ { g } ^ { S } ( s ) ) = \rho _ { 0 } ( s ) } & { { } f o r e \nu e r y s \in S . } & { } \end{array}
$$

Then, for every policy $\pi \in \Pi _ { \mathrm { d e c } }$ and agent $j \in N _ { \cdot }$

$$
J _ { \sigma _ { g } ( j ) } ^ { M _ { w } } ( g \cdot \pi ) = J _ { j } ^ { M _ { w } } ( \pi ) .
$$

Proof. Fix an<sub>y</sub> <sub>p</sub>olic<sub>y</sub> $\pi \in \Pi _ { \mathrm { d e c } }$ an<sup>d</sup> a<sub>g</sub>ent $j \in N$ <sub>.</sub> Th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d<sub>-</sub> return definition in E<sub>q</sub>uation (2) and the wei<sub>g</sub>hted-value identit<sub>y</sub> in Th<sub>eo</sub>r<sub>e</sub>m 6<sub>.</sub>1 <sub>g</sub>i<sub>ve</sub>

$$
\begin{array} { l } { { \displaystyle J _ { j } ^ { M _ { w } } ( \pi ) = \sum _ { s \in { \cal S } } \rho _ { 0 } ( s ) V _ { j , 0 } ^ { \pi ; M _ { w } } ( s , ( ) ^ { N } ) } } \\ { ~ = \displaystyle \sum _ { s \in { \cal S } } \rho _ { 0 } ( s ) \sum _ { i \in N } w _ { i } V _ { i , 0 } ^ { \pi ; M } ( s , ( ) ^ { N } ) }  \\ { { \displaystyle ~ = \sum _ { i \in N } w _ { i } \sum _ { s \in { \cal S } } \rho _ { 0 } ( s ) V _ { i , 0 } ^ { \pi ; M } ( s , ( ) ^ { N } ) } } \\ { { \displaystyle ~ = \sum _ { i \in N } w _ { i } J _ { i } ^ { M } ( \pi ) . } } \end{array}
$$

The second e<sub>q</sub>ualit<sub>y</sub> a<sub>pp</sub>lies the theorem at the initial time ste<sub>p</sub> to eac<sup>h</sup> state an<sup>d</sup> t<sup>h</sup>e empty joint <sup>h</sup>istory. T<sup>h</sup>e t<sup>h</sup>ir<sup>d</sup> interc<sup>h</sup>anges t<sup>h</sup>e fi<sub>n</sub>it<sub>e</sub> <sub>sums</sub> <sub>over</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>an</sub>d <sub>agen</sub>t<sub>s.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d l<sub>as</sub>t <sub>equa</sub>liti<sub>es</sub> <sub>use</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d<sub>-re</sub>t<sub>urn</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>w</sub>ith th<sub>e</sub> <sub>same</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\rho _ { 0 }$

For the second claim<sub>,</sub> wei<sub>g</sub>ht <sub>p</sub>reservation and Theorem 6.1 im<sub>p</sub>l<sub>y</sub> th<sub>a</sub>t $g \in G _ { M _ { w } } ^ { \mathrm { e v a l } }$ <sup>.</sup> <sup>B</sup>y <sup>assum</sup>p<sup>tion</sup>, � <sup>also</sup> p<sup>reserves</sup> $\rho _ { 0 } { \mathrm { ; } }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>f $M _ { w }$ <sub>as</sub> <sub>we</sub>ll <sub>as</sub> $M .$ <sub>.</sub> Th<sub>us</sub> Pr<sub>opos</sub>iti<sub>o</sub>n B<sub>.</sub>2<sub>,</sub> a<sub>pp</sub>lied to $M _ { w } ,$ , <sub>g</sub><sup>i</sup>ves

$$
J _ { \sigma _ { g } ( j ) } ^ { { M _ { w } } } ( g \cdot \pi ) = J _ { j } ^ { { M _ { w } } } ( \pi )
$$

<sup>f</sup>or every po<sup>l</sup>icy � an<sup>d</sup> agent �, proving t<sup>h</sup>e secon<sup>d</sup> c<sup>l</sup>aim.

Proposition D.3 (Aggregation can split or merge structural agent types). Under the construction of Theorem $6 . 1 ,$ each of the following outcomes can occur for a one-stage, two-agent source model M and its derived common-reward model $M _ { w } \mathrm { . }$

(1) With unequal weights �<sub>1</sub> $\neq w _ { 2 }$ , one structural agent type in M splits into two distinct structural agent types in $M _ { w } .$

(2) With equal weights $w _ { 1 } = w _ { 2 }$ , two distinct structural agent types in M merge into one structural agent type in $\mathcal { M } _ { w }$

(3) With unequal weights $w _ { 1 } \neq w _ { 2 }$ , two distinct structural agent types in M merge into one structural agent type in $\mathcal { M } _ { w }$

Consequently, neither direction ofrefinement between ${ \mathcal { P } } ( G _ { \mathcal { M } } ^ { \mathrm { e v a l } } )$ and $\mathcal { P } ( G _ { \mathcal { M } _ { w } } ^ { \mathrm { e v a l } } )$ holds for all source models M and weight vectors �.

P<sub>roof.</sub> W<sub>e cons</sub>t<sub>ruc</sub>t <sub>one examp</sub>l<sub>e</sub> f<sub>or eac</sub>h <sub>o</sub>f th<sub>e</sub> th<sub>ree c</sub>l<sub>a</sub>i<sub>ms.</sub> All th<sub>ree use one-s</sub>t<sub>age,</sub> t<sub>wo-agen</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s w</sub>ith <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>spaces,</sub> bin<sub>a</sub>r<sub>y</sub> <sub>ac</sub>ti<sub>o</sub>n <sub>se</sub>t<sub>s,</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c</sub> <sub>se</sub>lf-tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>,</sub> and an admissible <sub>g</sub>rou<sub>p</sub> containin<sub>g</sub> onl<sub>y</sub> the identit<sub>y</sub> and a transformation <sub>�</sub> whose a<sub>g</sub>ent <sub>p</sub>ermutation swa<sub>p</sub>s the two a<sub>g</sub>ents and <sub>w</sub>h<sub>ose</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-</sub>l<sub>a</sub>b<sub>e</sub>l <sub>maps are</sub> id<sub>en</sub>titi<sub>es.</sub>

Case 1: Splitting a Structural Agent Type (Unequal Weights). T<sub>a</sub>k<sub>e</sub>

$$
r _ { 1 } ( a _ { 1 } , a _ { 2 } ) = a _ { 1 } , \qquad r _ { 2 } ( a _ { 1 } , a _ { 2 } ) = a _ { 2 } .
$$

$g$ is a com<sub>p</sub>onent<sub>w</sub>ise en<sub>v</sub>ironment a<sub>u</sub>tomor<sub>p</sub>hism of $M ,$ <sub>so</sub> it<sub>s o</sub>rbit <sub>p</sub>artition is {{1, 2}}. With $w = ( 3 / 4 , 1 / 4 )$ , <sup>the</sup> <sup>a</sup>gg<sup>re</sup>g<sup>ate</sup> <sup>common</sup> <sub>rewar</sub>d i<sub>s</sub>

$$
\begin{array} { c } { { r ^ { w } ( a _ { 1 } , a _ { 2 } ) = \displaystyle \frac { 3 } { 4 } r _ { 1 } ( a _ { 1 } , a _ { 2 } ) + \displaystyle \frac { 1 } { 4 } r _ { 2 } ( a _ { 1 } , a _ { 2 } ) } } \\ { { = \displaystyle \frac { 3 } { 4 } a _ { 1 } + \displaystyle \frac { 1 } { 4 } a _ { 2 } . } } \end{array}
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
r ^ { w } ( 1 , 0 ) = \frac { 3 } { 4 } , \qquad r ^ { w } ( 0 , 1 ) = \frac { 1 } { 4 } .
$$

For the deterministic <sub>p</sub>olic<sub>y</sub> � selectin<sub>g</sub> (1, 0), the transformed <sub>p</sub>o<sup>li</sup>c<sub>y</sub> $g \cdot \pi$ selects (0, 1). Hence,

$$
V _ { 2 , 0 } ^ { g \cdot \pi ; { \cal M } _ { w } } = \frac { 1 } { 4 } \ne \frac { 3 } { 4 } = V _ { 1 , 0 } ^ { \pi ; { \cal M } _ { w } } .
$$

Polic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> re<sub>q</sub>uires this identit<sub>y</sub> for ever<sub>y p</sub>olic<sub>y,</sub> so this sin<sub>g</sub>le counterexam<sub>p</sub>le is enou<sub>g</sub>h to exclude <sub>�</sub>. The onl<sub>y</sub> <sub>non</sub>id<sub>en</sub>tit<sub>y a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on �</sub> th<sub>ere</sub>f<sub>ore</sub> f<sub>a</sub>il<sub>s</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n-</sub> in<sub>g</sub> evaluation identit<sub>y</sub> in the derived model<sub>,</sub> whose <sub>p</sub>artition is {{1}, {2}}. A<sub>gg</sub>re<sub>g</sub>ation has s<sub>p</sub>lit a structural a<sub>g</sub>ent t<sub>yp</sub>e of M. Case 2: Merging Structural Agent Types (Equal Weights).

$$
r _ { 1 } ( a _ { 1 } , a _ { 2 } ) = a _ { 1 } , r _ { 2 } ( a _ { 1 } , a _ { 2 } ) = a _ { 2 } + 1 .
$$

Transformin<sub>g</sub> the deterministic <sub>p</sub>olic<sub>y</sub> that selects $( 0 , 0 )$ <sup>b</sup><sub>y �</sub> <sup>l</sup>eaves it unchan<sub>g</sub>ed but <sub>g</sub>ives $V _ { 1 , 0 } ^ { \pi ; { \cal M } } = 0$ <sub>an</sub>d $V _ { 2 , 0 } ^ { g \cdot \pi ; M } \ = \ 1$ <sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n e<sub>q</sub>uivariance would re<sub>q</sub>uire these values to be e<sub>q</sub>ual<sub>,</sub> so <sub>�</sub> is not a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of M. Because the admissible <sub>g</sub>rou<sub>p</sub> contains onl<sub>y</sub> the identit<sub>y</sub> and <sub>�</sub>, the <sub>p</sub>artition of M is {{1}, {2}}. With $w = \left( 1 , 1 \right)$ <sub>,</sub> however<sub>,</sub> the a<sub>gg</sub>re<sub>g</sub>ate common reward is

$$
\begin{array} { r l } & { r ^ { w } ( a _ { 1 } , a _ { 2 } ) = r _ { 1 } ( a _ { 1 } , a _ { 2 } ) + r _ { 2 } ( a _ { 1 } , a _ { 2 } ) } \\ & { ~ = a _ { 1 } + ( a _ { 2 } + 1 ) } \\ & { ~ = a _ { 1 } + a _ { 2 } + 1 , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> in<sub>va</sub>ri<sub>a</sub>nt <sub>u</sub>nd<sub>e</sub>r $g .$ Th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> d<sub>ynam</sub>i<sub>cs an</sub>d <sub>o</sub>b<sub>serva-</sub> ti<sub>o</sub>n<sub>s a</sub>r<sub>e</sub> in<sub>va</sub>ri<sub>a</sub>nt <sub>as we</sub>ll<sub>, so �</sub> i<sub>s a co</sub>m<sub>po</sub>n<sub>e</sub>nt<sub>w</sub>i<sub>se e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt automor<sub>p</sub>hism<sub>,</sub> hence a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y,</sub> in $M _ { w } .$ <sup>I</sup>ts <sub>p</sub>artition is {{1, 2}}. A<sub>gg</sub>re<sub>g</sub>ation has mer<sub>g</sub>ed the two source structural <sup>a</sup>g<sup>ent</sup> <sup>t</sup>yp<sup>es.</sup>

Case 3: Merging Structural Agent Types (Unequal Weights). T<sub>a</sub>k<sub>e</sub>

$$
r _ { 1 } ( a _ { 1 } , a _ { 2 } ) = a _ { 1 } , r _ { 2 } ( a _ { 1 } , a _ { 2 } ) = \frac { 1 } { 2 } a _ { 2 } .
$$

For the deterministic <sub>p</sub>olic<sub>y</sub> � selectin<sub>g</sub> (1, 0), the transformed <sub>p</sub>o<sup>li</sup>c<sub>y</sub> $g \cdot \pi$ selects (0, 1), and

$$
V _ { 2 , 0 } ^ { g \cdot \pi ; { \cal M } } = \frac { 1 } { 2 } \neq 1 = V _ { 1 , 0 } ^ { \pi ; { \cal M } } .
$$

Th<sub>us</sub> $g \notin G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>,</sub> and the so<sub>u</sub>rce <sub>p</sub>artition is $\{ \{ 1 \} , \{ 2 \} \}$ <sub>.</sub> With $w =$ $( 1 / 3 , 2 / 3 )$ , t<sup>h</sup>e a<sub>gg</sub>re<sub>g</sub>ate common rewar<sup>d</sup> <sup>i</sup>s

$$
r ^ { w } ( a _ { 1 } , a _ { 2 } ) = \frac { 1 } { 3 } r _ { 1 } ( a _ { 1 } , a _ { 2 } ) + \frac { 2 } { 3 } r _ { 2 } ( a _ { 1 } , a _ { 2 } ) = \frac { 1 } { 3 } ( a _ { 1 } + a _ { 2 } ) ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> in<sub>va</sub>ri<sub>a</sub>nt <sub>u</sub>nd<sub>e</sub>r $g .$ Th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> d<sub>ynam</sub>i<sub>cs an</sub>d <sub>o</sub>b<sub>ser-</sub> <sub>va</sub>ti<sub>o</sub>n<sub>s</sub> <sub>a</sub>r<sub>e</sub> <sub>a</sub>l<sub>so</sub> in<sub>va</sub>ri<sub>a</sub>nt<sub>,</sub> <sub>so</sub> <sub>�</sub> i<sub>s</sub> <sub>a</sub> <sub>co</sub>m<sub>po</sub>n<sub>e</sub>nt<sub>w</sub>i<sub>se</sub> <sub>e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt automor<sub>p</sub>hism and hence a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> of $M _ { w } .$ Its <sub>p</sub>artition is {{1, 2}}. Therefore, a<sub>gg</sub>re<sub>g</sub>ation can mer<sub>g</sub>e distinct source structural a<sub>g</sub>ent t<sub>yp</sub>es even when their wei<sub>g</sub>hts difer.

C<sub>ases</sub> 1<sub>–</sub>3 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> th<sub>ree c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>ou</sub>t<sub>comes.</sub> C<sub>ase</sub> 1 <sub>an</sub>d <sub>e</sub>ith<sub>er</sub> C<sub>ase</sub> 2 <sub>or</sub> C<sub>ase</sub> 3 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> t<sub>wo</sub> <sub>oppos</sub>it<sub>e</sub> f<sub>a</sub>il<sub>ures</sub> <sub>o</sub>f <sub>re</sub>fi<sub>nemen</sub>t<sub>,</sub> <sub>prov</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> □

## E Detailed Comparisons to Prior Symmetry Frameworks

Sections 4–6 define a<sub>g</sub>ent hetero<sub>g</sub>eneit<sub>y</sub> throu<sub>g</sub>h the a<sub>g</sub>ent orbits <sub>o</sub>f $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ <sub>.</sub> Th<sub>e</sub> <sub>compar</sub>i<sub>sons</sub> b<sub>e</sub>l<sub>ow</sub> d<sub>o</sub> <sub>no</sub>t i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>s</sub>t<sub>ruc-</sub> tura<sup>l</sup> a<sub>g</sub>ent t<sub>yp</sub>es. <sup>Th</sup>e<sub>y</sub> <sup>i</sup>nstea<sup>d</sup> state <sup>h</sup>ow two <sub>p</sub>r<sup>i</sup>or s<sub>y</sub>mmetr<sub>y</sub> frameworks relate to the <sub>p</sub>reservation conditions used in this <sub>p</sub>a<sub>p</sub>er.

## E.1 Counting and Isomorphic Symmetry

Braun et al. [7] formulate countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> and isomor<sub>p</sub>hic s<sub>y</sub>mmetr<sub>y</sub> for <sub>p</sub>artitioned Dec-POMDPs. Co<sub>u</sub>ntin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> makes the tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>, co</sub>mm<sub>o</sub>n-r<sub>ewa</sub>rd<sub>, a</sub>nd <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n<sub>s</sub> in<sub>va</sub>ri<sub>a</sub>nt t<sub>o</sub> <sub>w</sub>ithin-<sub>pa</sub>rtiti<sub>o</sub>n <sub>pe</sub>rm<sub>u</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n <sub>ass</sub>i<sub>g</sub>nments. E<sub>q</sub>uivalentl<sub>y,</sub> those functions de<sub>p</sub>end on the assi<sub>g</sub>nments onl<sub>y</sub> throu<sub>g</sub>h their <sub>p</sub>er-<sub>p</sub>artition histo<sub>g</sub>rams. Isomor<sub>p</sub>hic s<sub>y</sub>mmetr<sub>y</sub> additionall<sub>y</sub> re<sub>q</sub>uires these functions to factor into re<sub>p</sub>eated identical tem<sub>p</sub>lates and therefore im<sub>p</sub>lies countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> aft<sub>er groun</sub>di<sub>ng, w</sub>h<sub>ereas</sub> th<sub>e converse nee</sub>d <sub>no</sub>t h<sub>o</sub>ld<sub>.</sub> Th<sub>us,</sub> th<sub>e</sub> t<sub>wo</sub> <sub>no</sub>ti<sub>ons cons</sub>t<sub>ra</sub>i<sub>n</sub> th<sub>e</sub> D<sub>ec-</sub>POMDP <sub>mo</sub>d<sub>e</sub>l f<sub>unc</sub>ti<sub>ons an</sub>d<sub>,</sub> f<sub>or</sub> i<sub>somor-</sub> <sub>p</sub>hic s<sub>y</sub>mmetr<sub>y,</sub> their factorization. B<sub>y</sub> contrast<sub>,</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> our more <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>l POSG d<sub>e</sub>finiti<sub>o</sub>n t<sub>o</sub> th<sub>e</sub> r<sub>esu</sub>ltin<sub>g g</sub>r<sub>ou</sub>nd<sub>e</sub>d D<sub>ec</sub>-POMDP <sub>g</sub>ives a condition on its com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub>-evaluation o<sub>p</sub>erator: it re<sub>q</sub>u<sup>i</sup>res $\mathcal { E } _ { M } ( g \cdot \pi ) = g \cdot \mathcal { E } _ { M } ( \pi )$ f<sub>or every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>cy.</sub>

Lifted-model implication. Within Braun et al.’s framework, the i<sub>mp</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>re</sub>l<sub>a</sub>t<sub>es</sub> t<sub>wo</sub> <sub>proper</sub>ti<sub>es</sub> <sub>o</sub>f <sub>a</sub> lift<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>:</sub>

## lifted isomor<sub>p</sub>hic factorization ⇓ <sup>countin</sup>g <sup>s</sup>y<sup>mmetr</sup>y <sup>after</sup> g<sup>roundin</sup>g.

H<sub>ere, groun</sub>di<sub>ng</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>es</sub> th<sub>e</sub> lift<sub>e</sub>d f<sub>ac</sub>t<sub>or</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>es</sub> f<sub>or</sub> th<sub>e</sub> indi<sub>v</sub>id<sub>u</sub>al a<sub>g</sub>ents. Isomor<sub>p</sub>hic factorization creates identical co<sub>p</sub>ies <sub>o</sub>f th<sub>e same</sub> f<sub>ac</sub>t<sub>or s</sub>t<sub>ruc</sub>t<sub>ure</sub> f<sub>or a</sub>ll <sub>agen</sub>t<sub>s</sub> i<sub>n eac</sub>h <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> lift<sub>e</sub>d <sub>p</sub>artition. Permutin<sub>g</sub> a<sub>g</sub>ents within a <sub>p</sub>art therefore onl<sub>y</sub> reorders id<sub>en</sub>ti<sub>ca</sub>l f<sub>ac</sub>t<sub>ors, so</sub> th<sub>e groun</sub>d<sub>e</sub>d t<sub>rans</sub>iti<sub>on, common-rewar</sub>d<sub>, an</sub>d observation functions can be re<sub>p</sub>resented usin<sub>g</sub> within-<sub>p</sub>artition co<sub>u</sub>nts. This is co<sub>u</sub>ntin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub>. The arro<sub>w</sub> is one-<sub>w</sub>a<sub>y</sub>: co<sub>u</sub>ntin<sub>g</sub> <sub>symme</sub>t<sub>ry can</sub> h<sub>o</sub>ld <sub>even w</sub>h<sub>en</sub> th<sub>e groun</sub>d<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l d<sub>oes no</sub>t <sub>ar</sub>i<sub>se</sub> from the same re<sub>p</sub>eated isomor<sub>p</sub>hic factorization.

Consequence for admissible transformations. To translate count-<sup>in</sup>g <sup>s</sup>y<sup>mmetr</sup>y <sup>into</sup> <sup>our</sup> <sup>lan</sup>g<sup>ua</sup>g<sup>e</sup>, <sup>let</sup> $\boldsymbol { \mathcal { I } } = \left\{ I _ { 1 } , \ldots , I _ { K } \right\}$ b<sub>e</sub> th<sub>e</sub> lift<sub>e</sub>d a<sub>g</sub>ent <sub>p</sub>artition and set

$$
H _ { \bar { I } } = \prod _ { k = 1 } ^ { K } \mathrm { S y m } ( I _ { k } ) , \qquad \iota : H _ { \bar { I } } \hookrightarrow { \mathcal R } _ { \mathrm { a d m } } ( { \mathcal M } ) , \qquad { \widehat H } _ { \bar { I } } : = \iota ( H _ { \bar { I } } ) .
$$

<sup>Th</sup>e <sub>g</sub>rou<sub>p</sub> $H _ { I }$ contains exactl<sub>y</sub> the a<sub>g</sub>ent <sub>p</sub>ermutations <sub>�</sub> that <sub>p</sub>ermute a<sub>g</sub>ents onl<sub>y</sub> within each <sub>p</sub>art $I _ { k }$ . <sup>Th</sup>e ma<sub>p</sub> � ass<sup>i</sup><sub>g</sub>ns ever<sub>y</sub> $\sigma \in H _ { I }$ <sub>a comp</sub>l<sub>e</sub>t<sub>e a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $\iota ( \sigma ) \in \mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ whose a<sub>g</sub>ent <sub>p</sub>ermutation is $\sigma .$ We assume t<sup>h</sup>at � is an injective <sup>h</sup>omomor<sub>p</sub>hism. Th<sub>u</sub>s distinct <sub>w</sub>ithin-<sub>p</sub>artition <sub>p</sub>erm<sub>u</sub>tations <sub>p</sub>rod<sub>u</sub>ce distinct admissible transformations<sub>,</sub> and com<sub>p</sub>osition is <sub>p</sub>reserved: $\iota ( \sigma _ { 1 } \sigma _ { 2 } ) \ : = \ : \iota ( \sigma _ { 1 } ) \iota ( \sigma _ { 2 } )$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, $\widehat { H } _ { \widehat { I } } = \{ \iota ( \sigma ) : \sigma \in H _ { \widehat { I } } \}$ i<sub>s</sub> <sub>a</sub> su<sup>b</sup><sub>g</sub>rou<sub>p</sub> o<sup>f</sup> $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ that is isomor<sub>p</sub>hic to $H _ { I }$ <sub>.</sub> Thi<sub>s</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>a</sub>l<sub>one supp</sub>li<sub>es on</sub>l<sub>y a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons.</sub> T<sub>o ma</sub>k<sub>e</sub> th<sub>em com-</sub> <sub>p</sub>onent<sub>w</sub>ise en<sub>v</sub>ironment a<sub>u</sub>tomor<sub>p</sub>hisms<sub>,</sub> ass<sub>u</sub>me in addition that ever<sub>y</sub> �(�) has the identit<sub>y</sub> state ma<sub>p</sub> and bijectivel<sub>y</sub> reindexes ev-<sub>ery groun</sub>d<sub>e</sub>d t<sub>rans</sub>iti<sub>on, o</sub>b<sub>serva</sub>ti<sub>on, an</sub>d <sub>a</sub>dditi<sub>ve common-rewar</sub>d f<sub>ac</sub>t<sub>or</sub> i<sub>n</sub>t<sub>o an equa</sub>l i<sub>ns</sub>t<sub>ance o</sub>f th<sub>e same</sub> f<sub>ac</sub>t<sub>or</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e, w</sub>ith <sub>no</sub> as<sub>y</sub>mmetric extra factor. These assum<sub>p</sub>tions <sub>g</sub>ive the com<sub>p</sub>onentwise <sub>p</sub>reservation e<sub>q</sub>uations in E<sub>q</sub>uations (5)–(7). Therefore,

$$
\widehat { H } _ { \bar { I } } \leq \mathrm { A u t } _ { \mathrm { e n v } } ( { \cal M } ) \leq G _ { \cal M } ^ { \mathrm { e v a l } } , \qquad \bar { I } = \mathcal { P } ( \widehat { H } _ { \bar { I } } ) \preceq \mathcal { P } _ { \mathrm { e v a l } } ( { \cal M } ) .
$$

Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>componen</sub>t<sub>w</sub>i<sub>se preser-</sub> <sub>va</sub>ti<sub>on assump</sub>ti<sub>ons, an</sub>d th<sub>e secon</sub>d f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> Th<sub>eorem</sub> 5<sub>.</sub>2<sub>.</sub> Because the a<sub>g</sub>ent com<sub>p</sub>onent of $\iota ( \sigma ) \mathrm { i } s \sigma ,$ th<sub>e or</sub>bit<sub>s o</sub>f $\widehat { H } _ { \varI }$ are exact<sup>l</sup><sub>y</sub> $I _ { 1 } , \ldots , I _ { K }$ . Hence<sub>,</sub> each lifted <sub>p</sub>artition <sub>p</sub>art is contained in one <sup>structural a</sup>g<sup>ent t</sup>yp<sup>e because</sup> $\widehat { H } _ { \cal I } \leq G _ { \cal M } ^ { \mathrm { e v a l } }$ <sub>.</sub> Th<sub>e co</sub>nt<sub>a</sub>inm<sub>e</sub>nt m<sub>ay</sub> b<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t b<sub>ecause</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ ma<sub>y</sub> contain additional s<sub>y</sub>mmetries that connect a<sub>g</sub>ents from diferent lifted <sub>p</sub>artition <sub>p</sub>arts. Countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> is th<sub>us a su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on</sub> f<sub>or</sub> thi<sub>s par</sub>ti<sub>cu</sub>l<sub>ar su</sub>b<sub>group o</sub>f <sub>compo-</sub> nentwise environment automor<sub>p</sub>hisms under the stated <sub>g</sub>roundin<sub>g</sub> assum<sub>p</sub>tions. It is not a se<sub>p</sub>arate structural-t<sub>yp</sub>e definition and does not claim to reco<sub>v</sub>er e<sub>v</sub>er<sub>y</sub> com<sub>p</sub>onent<sub>w</sub>ise en<sub>v</sub>ironment a<sub>u</sub>tomor-<sub>p</sub>hism. Since each lifted transformation has the identit<sub>y</sub> state ma<sub>p,</sub> it automaticall<sub>y</sub> <sub>p</sub>reserves $\rho _ { 0 } { \mathrm { ; } }$ <sub>,</sub> <sub>w</sub>itho<sub>u</sub>t an<sub>y</sub> additional restriction on t<sup>h</sup>e initia<sup>l</sup> <sup>d</sup>istri<sup>b</sup>ution. Because t<sup>h</sup>e initia<sup>l</sup> joint <sup>h</sup>istory is $( ) ^ { N }$ <sub>,</sub> <sub>eac</sub>h initia<sup>l</sup> state–joint-<sup>h</sup>istory pair $( s _ { 0 } , ( ) ^ { N } )$ in E<sub>q</sub>uation (2) is <sub>p</sub>reserved <sub>as we</sub>ll<sub>.</sub> Th<sub>ese</sub> f<sub>ac</sub>t<sub>s are nee</sub>d<sub>e</sub>d <sub>w</sub>h<sub>en</sub> d<sub>er</sub>i<sub>v</sub>i<sub>ng</sub> id<sub>en</sub>titi<sub>es</sub> f<sub>or</sub> $J _ { i } ( \pi )$ <sub>no</sub>t f<sub>or</sub> th<sub>e su</sub>b<sub>group re</sub>l<sub>a</sub>ti<sub>on</sub> $\widehat { H } _ { \cal I } \leq G _ { \cal M } ^ { \mathrm { e v a l } }$

Countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> does not im<sub>p</sub>l<sub>y</sub> the re<sub>p</sub>eated additive factorization considered here for a <sub>p</sub>artition with one <sub>p</sub>art<sub>,</sub> even when both notions use the same lifted <sub>p</sub>artition. Consider a one-sta<sub>g</sub>e<sub>,</sub> t<sub>wo</sub>-<sub>age</sub>nt D<sub>ec</sub>-POMDP <sub>w</sub>ith <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>space</sub> ${ \cal S } = \{ s _ { \star } \}$ <sub>,</sub> binar<sub>y</sub> action <sup>s</sup>p<sup>aces</sup> $\mathcal { A } _ { 1 } = \mathcal { A } _ { 2 } = \{ 0 , 1 \}$ <sub>,</sub> sin<sub>g</sub>leton observation s<sub>p</sub>aces<sub>,</sub> deterministi<sub>c se</sub>lf<sub>-</sub>t<sub>rans</sub>iti<sub>on, an</sub>d <sub>common coor</sub>di<sub>na</sub>ti<sub>on rewar</sub>d

$$
r ^ { \mathrm { c o m } } ( s _ { \star } , ( a _ { 1 } , a _ { 2 } ) , s _ { \star } ) = 1 \{ a _ { 1 } = a _ { 2 } \} .
$$

For the lifted <sub>p</sub>artition with one <sub>p</sub>art<sub>,</sub> $, \tau = \{ \{ 1 , 2 \} \}$ <sub>,</sub> b<sub>o</sub>th <sub>agen</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ong</sub> t<sub>o</sub> th<sub>e same par</sub>titi<sub>on par</sub>t<sub>.</sub> Th<sub>e rewar</sub>d <sub>equa</sub>l<sub>s</sub> 1 f<sub>or</sub> th<sub>e ac</sub>ti<sub>on</sub> histo<sub>g</sub>rams (2, 0) and (0, 2), and 0 for (1, 1). It therefore has countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub>. The admissible transformation whose a<sub>g</sub>ent <sub>p</sub>ermutation swa<sub>p</sub>s the a<sub>g</sub>ents is also a com<sub>p</sub>onentwise environment automor-<sub>p</sub>hism and hence a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>. However<sub>,</sub> because th<sub>e mo</sub>d<sub>e</sub>l h<sub>as on</sub>l<sub>y</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { \star }$ <sub>, a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub> $r ^ { \mathrm { c o m } } ( s _ { \star } , ( a _ { 1 } , a _ { 2 } ) , s _ { \star } )$ <sup>b</sup><sub>y</sub> $r ^ { \mathrm { c o m } } ( a _ { 1 } , a _ { 2 } )$ . For this same <sub>p</sub>artition with one <sub>p</sub>art<sub>,</sub> the re<sub>p</sub>eated <sub>a</sub>dditi<sub>ve</sub> f<sub>ac</sub>t<sub>or</sub>i<sub>za</sub>ti<sub>on cons</sub>id<sub>ere</sub>d h<sub>ere uses</sub> th<sub>e same</sub> l<sub>oca</sub>l f<sub>ac</sub>t<sub>or</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e</sub> <sub>�</sub> f<sub>or</sub> b<sub>o</sub>th <sub>agen</sub>t<sub>s:</sub> $r ^ { \mathrm { c o m } } ( a _ { 1 } , a _ { 2 } ) = c + u ( a _ { 1 } ) + u ( a _ { 2 } )$ <sub>.</sub> Th<sub>e</sub> <sub>s</sub>h<sub>are</sub>d f<sub>unc</sub>ti<sub>on</sub> <sub>�</sub> i<sub>s</sub> th<sub>e</sub> <sub>repea</sub>t<sub>e</sub>d t<sub>emp</sub>l<sub>a</sub>t<sub>e</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>p</sub>art {1, 2}. This form would re<sub>q</sub>uire

$$
r ^ { \mathrm { c o m } } ( 0 , 0 ) + r ^ { \mathrm { c o m } } ( 1 , 1 ) = r ^ { \mathrm { c o m } } ( 0 , 1 ) + r ^ { \mathrm { c o m } } ( 1 , 0 ) ,
$$

b<sub>ecause</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es wou</sub>ld <sub>equa</sub>l $2 c + 2 u ( 0 ) + 2 u ( 1 )$ <sub>.</sub> Alth<sub>oug</sub>h th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>ti<sub>on</sub> <sub>rewar</sub>d <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> <sub>coun</sub>ti<sub>ng</sub> <sub>symme</sub>t<sub>ry,</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>s</sub> $1 + 1 = 2 ,$ <sub>, w</sub>h<sub>ereas</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>s</sub> $0 + 0 = 0$ . Hence, no <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> <sub>o</sub>f <sub>�</sub> <sub>an</sub>d <sub>�</sub> <sub>g</sub>i<sub>ve</sub> thi<sub>s</sub> <sub>rewar</sub>d i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>repea</sub>t<sub>e</sub>d <sub>a</sub>dditi<sub>ve</sub> form. Thus countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub> does not im<sub>p</sub>l<sub>y</sub> this re<sub>p</sub>eated additive factorization for a <sub>p</sub>artition with one <sub>p</sub>art. More <sub>g</sub>enerall<sub>y,</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> does not b<sub>y</sub> itself <sub>p</sub>rovide the re<sub>p</sub>eated lift<sub>e</sub>d f<sub>ac</sub>t<sub>or</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>use</sub>d t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>compac</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d tractabilit<sub>y</sub> <sub>g</sub>uarantees for lifted isomor<sub>p</sub>hic models. Section E.1.1 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>or-re</sub>i<sub>n</sub>d<sub>ex</sub>i<sub>ng proo</sub>f <sub>an</sub>d th<sub>e proo</sub>f th<sub>a</sub>t thi<sub>s examp</sub>l<sub>e</sub> l<sub>ac</sub>k<sub>s</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d f<sub>ac</sub>t<sub>or</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or a par</sub>titi<sub>on w</sub>ith <sub>one par</sub>t<sub>.</sub>

E.1.1 Proofs ofthe lifted-model comparison. The first proposition <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>groun</sub>di<sub>ng</sub> <sub>assump</sub>ti<sub>ons</sub> <sub>exp</sub>li<sub>c</sub>it<sub>;</sub> th<sub>e</sub> <sub>secon</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> the same-<sub>p</sub>artition non-con<sub>v</sub>erse.

Proposition E.1 (Lifted symmetry as a sufficient condition). Let $\boldsymbol { J } = \left\{ I _ { 1 } , \ldots , I _ { K } \right\}$ be an agent partition and let

$$
H _ { \bar { Z } } = \prod _ { k = 1 } ^ { K } \operatorname { S y m } ( I _ { k } ) .
$$

Suppose a finite Dec-POMDP grounding admits an injective homomorphism

$$
\iota : H _ { \cal I } \longrightarrow { \mathcal { R } } _ { \mathrm { a d m } } ( M )
$$

whose image consists oftransformations with the corresponding agent permutation, identity state map, and corresponding action and observation coordinate maps. Suppose also that every transformation in the image bijectively reindexes all grounded transition, observation, and additive common-reward factor instances into instances ofthe same templates, with no asymmetric extra factors. Writing $\widehat { H } _ { \cal { I } } = \iota ( H _ { \cal { I } } )$ we then have

$$
\widehat { H } _ { \cal I } \leq \mathrm { A u t } _ { \mathrm { e n v } } ( { \cal M } ) \leq G _ { \cal M } ^ { \mathrm { e v a l } } , \qquad { \cal I } = { \mathcal { P } } ( \widehat { H } _ { \cal I } ) \preceq { \mathcal { P } } ( G _ { \cal M } ^ { \mathrm { e v a l } } ) .
$$

Because the state map is the identity and the initial history is empty, the same transformations preserve the initial distribution.

Proof<sub>.</sub> Fix <sub>a w</sub>ithin-<sub>pa</sub>rtiti<sub>o</sub>n <sub>pe</sub>rm<sub>u</sub>t<sub>a</sub>ti<sub>o</sub>n $\sigma \in H _ { I }$ <sub>an</sub>d it<sub>s com-</sub> <sub>p</sub>lete transformation �(�). B<sub>y</sub> assum<sub>p</sub>tion, it merel<sub>y</sub> reorders the fi<sub>n</sub>it<sub>e</sub>l<sub>y many groun</sub>d<sub>e</sub>d f<sub>ac</sub>t<sub>ors w</sub>ithi<sub>n eac</sub>h f<sub>ac</sub>t<sub>or pro</sub>d<sub>uc</sub>t d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> a transition or observation <sub>p</sub>robabilit<sub>y</sub> and within the factor sum d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e common rewar</sub>d<sub>.</sub> E<sub>very</sub> f<sub>ac</sub>t<sub>or</sub> i<sub>s ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>w</sub>ith <sub>an equa</sub>l i<sub>ns</sub>t<sub>ance o</sub>f th<sub>e same</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e.</sub> R<sub>eor</sub>d<sub>er</sub>i<sub>ng a</sub> fi<sub>n</sub>it<sub>e pro</sub>d<sub>uc</sub>t <sub>or sum</sub> does not chan<sub>g</sub>e its value. Thus �(�) <sub>p</sub>reserves the <sub>g</sub>rounded transition distrib<sub>u</sub>tion<sub>,</sub> obser<sub>v</sub>ation distrib<sub>u</sub>tion<sub>,</sub> and re<sub>w</sub>ard com<sub>p</sub>onent-<sub>w</sub>i<sub>se.</sub> B<sub>ecause</sub> thi<sub>s argumen</sub>t h<sub>o</sub>ld<sub>s</sub> f<sub>or every</sub> $\sigma , \widehat { H } _ { J } \leq \mathrm { A u t _ { e n v } } ( \mathcal { M } )$ Theorem 5.2 then <sub>g</sub>ives $\mathrm { A u t } _ { \mathrm { e n v } } ( M ) \leq G _ { M } ^ { \mathrm { e v a l } }$

W<sub>e nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e agen</sub>t <sub>or</sub>bit<sub>s o</sub>f $ { \hat { H } } _ { J }$ are exact<sup>l</sup><sub>y</sub> t<sup>h</sup>e <sub>p</sub>arts of I. Fix a <sub>p</sub>art $I _ { k }$ an<sup>d</sup> an a<sub>g</sub>ent $i \in I _ { k }$ . <sup>E</sup>ver<sub>y</sub> $g \in \widehat { H } _ { \mathcal { I } }$ h<sub>as</sub> th<sub>e</sub> f<sub>orm</sub> $g = \iota ( \sigma )$ f<sub>or</sub> <sub>some</sub> $\sigma \in H _ { I }$ <sub>,</sub> with a<sub>g</sub>ent <sub>p</sub>ermutation $\sigma _ { q } = \sigma$ <sub>.</sub> Sin<sub>ce</sub> <sub>�</sub> <sub>p</sub>ermutes a<sub>g</sub>ents onl<sub>y</sub> within each <sub>p</sub>art<sub>,</sub> $\sigma _ { g } ( i ) \in I _ { k }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\{ \sigma _ { g } ( i ) : g \in \widehat { H } _ { \bar { I } } \} \subseteq I _ { k } .
$$

Con<sub>v</sub>ersel<sub>y,</sub> take an<sub>y</sub> $j \in I _ { k } .$ . T<sup>h</sup>e permutation t<sup>h</sup>at exc<sup>h</sup>anges � an<sup>d</sup> � <sub>an</sub>d fi<sub>xes</sub> <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>agen</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $H _ { I }$ <sub>,</sub> because it acts onl<sub>y</sub> within $I _ { k } ; \mathrm { i f } j = i ,$ <sub>, u</sub>se the identit<sub>y p</sub>erm<sub>u</sub>tation. For this <sub>p</sub>erm<sub>u</sub>tation $\sigma ,$ its <sup>i</sup>ma<sub>g</sub>e $\iota ( \sigma ) \in \widehat { H } _ { I }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\sigma _ { \iota ( \sigma ) } ( \iota ) = \sigma ( i ) = j .$ Thus ever<sub>y</sub> a<sub>g</sub>ent in $I _ { k }$ belon<sub>g</sub>s to the orbit of�<sub>, p</sub>rovin<sub>g</sub> the reverse containment. Hence<sub>,</sub>

$$
\{ \sigma _ { g } ( i ) : g \in \widehat { H } _ { \bar { I } } \} = I _ { k } .
$$

Si<sub>nce</sub> thi<sub>s</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>every</sub> � <sub>an</sub>d <sub>every</sub> $i \in I _ { k }$ <sub>,</sub> th<sub>e</sub> <sub>o</sub>rbit <sub>pa</sub>rtiti<sub>o</sub>n i<sub>s</sub> $\mathcal { P } ( \widehat { H } _ { \mathcal { I } } ) = \mathcal { I }$

T<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e re</sub>fi<sub>nemen</sub>t<sub>, use</sub> th<sub>e</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> $\widehat { H } _ { \cal I } \leq G _ { M } ^ { \mathrm { e v a l } }$ <sub>p</sub>rove<sup>d</sup> above. Ever<sub>y</sub> transformation used to connect two a<sub>g</sub>ents in $I _ { k }$ th<sub>ere-</sub> f<sub>ore</sub> <sub>a</sub>l<sub>so</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, eac<sup>h</sup> <sub>p</sub>art $I _ { k }$ i<sub>s</sub> <sub>co</sub>nt<sub>a</sub>in<sub>e</sub>d in

one $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ -orbit<sub>,</sub> which is one structural a<sub>g</sub>ent t<sub>yp</sub>e. B<sub>y</sub> the definition <sub>o</sub>f <sub>pa</sub>rtiti<sub>o</sub>n r<sub>e</sub>fin<sub>e</sub>m<sub>e</sub>nt<sub>, as</sub> in Pr<sub>opos</sub>iti<sub>o</sub>n B<sub>.</sub>3<sub>,</sub> thi<sub>s g</sub>i<sub>ves</sub>

$$
\mathcal { T } = \mathcal { P } ( \widehat { H } _ { \mathcal { T } } ) \preceq \mathcal { P } ( G _ { \mathcal { M } } ^ { \mathrm { e v a l } } ) .
$$

Finall<sub>y,</sub> the identit<sub>y</sub> state ma<sub>p</sub> fixes the initial state coordinate and ever<sub>y</sub> transformation fixes the em<sub>p</sub>t<sub>y</sub> initial histor<sub>y</sub>. Hence each transformation in the ima<sub>g</sub>e <sub>p</sub>reser<sub>v</sub>es the initial distrib<sub>u</sub>tion <sub>as c</sub>l<sub>a</sub>i<sub>me</sub>d<sub>.</sub> □

Proposition E.2 (Same-partition non-converse). In the onestage, two-agent common-reward model with binary actions and

$$
R { \left( a _ { 1 } , a _ { 2 } \right) } = 1 { \left\{ a _ { 1 } = a _ { 2 } \right\} } ,
$$

the candidate partition with one part, $\{ \{ 1 , 2 \} \}$ , has counting symme-$t r y ,$ and the transformation whose agent permutation swaps the two agents is a componentwise environment automorphism and therefore a policy-evaluation symmetry. However, there are no constant � and repeated local contribution � such that

$$
\begin{array} { r } { R ( a _ { 1 } , a _ { 2 } ) = c + u ( a _ { 1 } ) + u ( a _ { 2 } ) . } \end{array}
$$

P<sub>r</sub>oof<sub>.</sub> Th<sub>e rewar</sub>d i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e ac</sub>ti<sub>on</sub> hi<sub>s</sub>t<sub>ogram</sub> f<sub>or</sub> the onl<sub>y</sub> <sub>p</sub>artition <sub>p</sub>art: it is 1 for (2, 0) and (0, 2), and 0 for $( 1 , 1 )$ T<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>spaces</sub> <sub>an</sub>d d<sub>e</sub>t<sub>er-</sub> ministic self-transition<sub>,</sub> this <sub>p</sub>roves countin<sub>g</sub> s<sub>y</sub>mmetr<sub>y</sub>. The reward i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>unc</sub>h<sub>ange</sub>d b<sub>y</sub> <sub>exc</sub>h<sub>ang</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> <sub>ac</sub>ti<sub>on</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> while ever<sub>y</sub> other com<sub>p</sub>onent is also unchan<sub>g</sub>ed. The corres<sub>p</sub>ondin<sub>g</sub> transformation<sub>,</sub> whose a<sub>g</sub>ent <sub>p</sub>ermutation swa<sub>p</sub>s the two a<sub>g</sub>ents<sub>,</sub> is a com<sub>p</sub>onentwise environment automor<sub>p</sub>hism<sub>,</sub> and Theorem 5.2 shows that it is a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>.

If th<sub>e</sub> <sub>repea</sub>t<sub>e</sub>d <sub>a</sub>dditi<sub>ve</sub> f<sub>orm</sub> h<sub>e</sub>ld<sub>,</sub> th<sub>en</sub>

$$
\begin{array} { c } { { R ( 0 , 0 ) + R ( 1 , 1 ) = [ c + 2 u ( 0 ) ] + [ c + 2 u ( 1 ) ] } } \\ { { { } } } \\ { { { } = 2 [ c + u ( 0 ) + u ( 1 ) ] = R ( 0 , 1 ) + R ( 1 , 0 ) . } } \end{array}
$$

F<sub>or</sub> th<sub>e coor</sub>di<sub>na</sub>ti<sub>on rewar</sub>d<sub>,</sub> th<sub>e</sub> l<sub>e</sub>ft <sub>s</sub>id<sub>e</sub> i<sub>s</sub> 2 <sub>an</sub>d th<sub>e r</sub>i<sub>g</sub>ht <sub>s</sub>id<sub>e</sub> i<sub>s</sub> $0 ,$ <sub>a con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> Thi<sub>s ru</sub>l<sub>es ou</sub>t th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>re ea</sub>t<sub>e</sub>d <sub>a</sub>dditi<sub>ve rewar</sub>d factorization for a <sub>p</sub>artition with one <sub>p</sub>art. It does not rule out a <sub>p</sub>artition into sin<sub>g</sub>leton <sub>p</sub>arts<sub>,</sub> which se<sub>p</sub>arates the a<sub>g</sub>ents in the lift<sub>e</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on.</sub> □

## E.2 Expected Return Symmetry

An ex<sub>p</sub>ected return s<sub>y</sub>mmetr<sub>y</sub> (ERS) is defined throu<sub>g</sub>h self-<sub>p</sub>la<sub>y</sub>- o<sub>p</sub>timal Boltzmann <sub>p</sub>olicies at a fixed tem<sub>p</sub>erature $\alpha > 0 [ 2 2 ]$ . We <sub>compare</sub> th<sub>e</sub> t<sub>wo</sub> <sub>no</sub>ti<sub>ons</sub> <sub>on</sub> th<sub>e</sub>i<sub>r</sub> <sub>s</sub>h<sub>are</sub>d <sub>su</sub>b<sub>c</sub>l<sub>ass</sub> <sub>o</sub>f fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon</sub> D<sub>ec</sub>-POMDP<sub>s</sub>: initi<sub>a</sub>l hi<sub>s</sub>t<sub>o</sub>ri<sub>es</sub> <sub>a</sub>r<sub>e</sub> <sub>e</sub>m<sub>p</sub>t<sub>y,</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>o</sub>n <sub>spaces</sub> <sub>a</sub>r<sub>e</sub> fi<sub>xe</sub>d<sub>,</sub> <sub>a</sub>ll <sub>agen</sub>t<sub>s</sub> h<sub>ave</sub> <sub>a</sub> <sub>common</sub> <sub>rewar</sub>d f<sub>unc</sub>ti<sub>on,</sub> <sub>an</sub>d th<sub>e</sub> <sub>o</sub>b<sub>serva-</sub> ti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n f<sub>ac</sub>t<sub>o</sub>riz<sub>es as</sub> $\begin{array} { r } { Z ( o ^ { \prime } \mid s ^ { \prime } , a ) = \prod _ { i } U _ { i } ( o _ { i } ^ { \prime } \mid s ^ { \prime } , a ) } \end{array}$ . Let Ψ denote the ambient transformation <sub>g</sub>rou<sub>p</sub> used b<sub>y</sub> ERS. We identif<sub>y</sub> <sub>eac</sub>h <sub>o</sub>f it<sub>s e</sub>l<sub>emen</sub>t<sub>s w</sub>ith th<sub>e</sub> t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub>

$$
g = \bigl ( \mathrm { i d } _ { N } , \phi _ { g } ^ { S } , ( \phi _ { g , i } ^ { \mathcal { A } } ) _ { i \in N } , ( \phi _ { g , i } ^ { O } ) _ { i \in N } \bigr ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\phi _ { q } ^ { S } : S  S , \phi _ { q , i } ^ { \mathcal { A } } : \mathcal { A } _ { i }  \mathcal { A } _ { i } ,$ <sub>,</sub> <sub>an</sub>d $\phi _ { q , i } ^ { O } : O _ { i } \to O _ { i }$ are bijections. The <sub>g</sub>rou<sub>p</sub> Ψ contains all tu<sub>p</sub>les ofthis form. Thus, Ψ fixes the a<sub>g</sub>ent indices. Under this identification, Ψ and $\mathcal { R } _ { \mathrm { a d m } } ( \mathcal { M } )$ are <sub>groups</sub> <sub>o</sub>f t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> <sub>on</sub> th<sub>e</sub> <sub>same</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>so</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on</sub> i<sub>s</sub> <sub>we</sub>ll d<sub>e</sub>fi<sub>ne</sub>d<sub>.</sub> Th<sub>ey</sub> <sub>ac</sub>t <sub>on</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es</sub> b<sub>y</sub> the <sub>p</sub>olic<sub>y</sub>-transformation rule in E<sub>q</sub>uation (3). Let $\Pi ^ { \alpha } ( M )$ b<sub>e</sub> th<sub>e</sub> <sub>p</sub>olicies satisf<sub>y</sub>in<sub>g</sub> the Boltzmann condition at ever<sub>y</sub> local histor<sub>y,</sub> stated ex<sub>p</sub>licitl<sub>y</sub> in Section E.2.1<sub>,</sub> and let

$$
\Pi _ { * } ^ { \alpha } ( M ) = \underset { \pi \in \Pi ^ { \alpha } ( M ) } { \arg \operatorname* { m a x } } J ^ { M } ( \pi ) ,
$$

<sub>w</sub>h<sub>ere</sub> $J ^ { \mathcal { M } } ( \pi )$ i<sub>s</sub> th<sub>e co</sub>mm<sub>o</sub>n r<sub>e</sub>t<sub>u</sub>rn <sub>u</sub>nd<sub>e</sub>r th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> We com<sub>p</sub>are the two notions onl<sub>y</sub> on $\Psi \cap \mathcal { R } _ { \mathrm { a d m } } ( M )$ <sub>, w</sub>h<sub>ose e</sub>l<sub>emen</sub>t<sub>s</sub> <sub>are</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> i<sub>n</sub> b<sub>o</sub>th f<sub>orma</sub>li<sub>sms.</sub> Th<sub>roug</sub>h<sub>ou</sub>t thi<sub>s</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on,</sub> th<sub>e</sub> restricted ERS <sub>g</sub>rou<sub>p</sub> is ex<sub>p</sub>ressed<sub>,</sub> as follows:

$$
\Phi _ { \mathrm { E R } } ^ { \alpha } ( M ) : = \bigl \{ g \in \Psi \cap \mathscr { R } _ { \mathrm { a d m } } ( M ) : \ \{ g \cdot \pi : \pi \in \Pi _ { * } ^ { \alpha } ( M ) \} = \Pi _ { * } ^ { \alpha } ( M ) \bigr \} .
$$

Here, $\{ g \cdot \pi : \pi \in \Pi _ { * } ^ { \alpha } ( M ) \}$ i<sub>s</sub> th<sub>e</sub> i<sub>mage o</sub>f th<sub>e en</sub>ti<sub>re se</sub>lf<sub>-p</sub>l<sub>ay-</sub> o<sub>p</sub>timal Boltzmann <sub>p</sub>olic<sub>y</sub> set under $g .$ Th<sub>e</sub> <sub>equa</sub>lit<sub>y</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>re-</sub> <sub>q</sub>uires <sub>�</sub> to ma<sub>p</sub> this set onto itself<sub>,</sub> althou<sub>g</sub>h it ma<sub>y</sub> ma<sub>p</sub> one member t<sub>o a</sub> dif<sub>eren</sub>t <sub>mem</sub>b<sub>er.</sub> B<sub>ecause</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e, ac</sub>ti<sub>on, an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>spaces are</sub> fi<sub>n</sub>it<sub>e,</sub> th<sub>e common</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> d<sub>oma</sub>i<sub>n</sub> i<sub>s a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>g</sub>rou<sub>p</sub>. The set e<sub>q</sub>ualit<sub>y</sub> is conse<sub>q</sub>uentl<sub>y</sub> e<sub>q</sub>uivalent to the <sub>p</sub>ointwise condition used b<sub>y</sub> Mu<sub>g</sub>lich et al. [22], namel<sub>y</sub> $g \cdot \pi \in \Pi _ { * } ^ { \alpha } ( M )$ f<sub>or</sub> <sup>ever</sup>y � $\in \Pi _ { * } ^ { \alpha } ( M )$ <sub>.</sub> Th<sub>us,</sub> ERS <sub>co</sub>n<sub>ce</sub>rn<sub>s a se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>op</sub>tim<sub>a</sub>l B<sub>o</sub>ltzmann <sub>p</sub>olic<sub>y</sub> set and its local action-value conditions. B<sub>y</sub> contrast<sub>,</sub> <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>p</sub>reserves com<sub>p</sub>lete <sub>p</sub>olic<sub>y</sub> evaluation <sub>un</sub>d<sub>er</sub> <sub>a</sub> t<sub>ype-compa</sub>tibl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> f<sub>or</sub> <sub>every</sub> d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>p</sub>olic<sub>y</sub> and ma<sub>y</sub> have a nonidentit<sub>y</sub> a<sub>g</sub>ent <sub>p</sub>ermutation.

Proposition E.3 (Class-level two-way non-implication). For every $\alpha > 0 ,$ there are finite $D e c \mathbf { - } P O M D P s \left. M _ { a } , M _ { b } \right.$ and transformations $g _ { a } , g _ { b }$ admitted by both formalisms such that

$$
g _ { a } \in \Phi _ { \mathrm { E R } } ^ { \alpha } ( M _ { a } ) \setminus G _ { M _ { a } } ^ { \mathrm { e v a l } , \rho _ { 0 } } , \qquad g _ { b } \in G _ { M _ { b } } ^ { \mathrm { e v a l } , \rho _ { 0 } } \setminus \Phi _ { \mathrm { E R } } ^ { \alpha } ( M _ { b } ) .
$$

Hence neither condition implies the other across the common model class.

Proof sketch. The proposition gives two counterexamples over th<sub>e common mo</sub>d<sub>e</sub>l <sub>c</sub>l<sub>ass.</sub> It d<sub>oes no</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m</sub> th<sub>a</sub>t b<sub>o</sub>th <sub>se</sub>t dif<sub>erences</sub> are nonem<sub>p</sub>t<sub>y</sub> for e<sub>v</sub>er<sub>y</sub> fixed Dec-POMDP. In $M _ { a } ,$ <sub>,</sub> th<sub>e</sub> <sub>un</sub>if<sub>orm</sub> <sub>p</sub>olic<sub>y</sub> is the uni<sub>q</sub>ue self-<sub>p</sub>la<sub>y</sub>-o<sub>p</sub>timal Boltzmann <sub>p</sub>olic<sub>y</sub>. A within-<sub>agen</sub>t <sub>ac</sub>ti<sub>on</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>li<sub>ng</sub> fi<sub>xes</sub> thi<sub>s</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> <sub>se</sub>t b<sub>u</sub>t <sub>c</sub>h<sub>anges</sub> th<sub>e</sub> <sub>re-</sub> turn ofa deterministic <sub>p</sub>olic<sub>y</sub> outside $\Pi ^ { \alpha }$ <sub>.</sub> It i<sub>s</sub> th<sub>e</sub>r<sub>e</sub>f<sub>o</sub>r<sub>e a</sub>n ERS tr<sub>a</sub>n<sub>s</sub>- formation but not a <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>. In $M _ { b ; }$ <sub>,</sub> <sub>a</sub> <sub>w</sub>ithin-<sub>agen</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on re</sub>l<sub>a</sub>b<sub>e</sub>li<sub>ng sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e va</sub>l<sub>ue-</sub>t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> id<sub>en-</sub> tit<sub>y</sub> in Definition 4.1 for ever<sub>y</sub> decentralized <sub>p</sub>olic<sub>y</sub> after transformin<sub>g</sub> the <sub>p</sub>olic<sub>y</sub> and local histories. However<sub>,</sub> it exchan<sub>g</sub>es observations with diferent conditional local action-value <sub>g</sub>a<sub>p</sub>s<sub>,</sub> so it ma<sub>p</sub>s the uni<sub>q</sub>ue Boltzmann <sub>p</sub>olic<sub>y</sub> to a <sub>p</sub>olic<sub>y</sub> that violates Boltzmann consistenc<sub>y</sub>. Both transformations have the identit<sub>y</sub> state ma<sub>p</sub> and therefore <sub>p</sub>reserve the corres<sub>p</sub>ondin<sub>g</sub> initial distribution. S<sub>ec</sub>ti<sub>o</sub>n E<sub>.</sub>2<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> b<sub>o</sub>th <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>s.</sub>

T<sub>o s</sub>t<sub>a</sub>t<sub>e</sub> th<sub>e exac</sub>t <sub>a</sub>dditi<sub>ona</sub>l <sub>con</sub>diti<sub>on,</sub> l<sub>e</sub>t $Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } )$ b<sub>e</sub> th<sub>e</sub> conditional ex<sub>p</sub>ected return-to-<sub>g</sub>o when a<sub>g</sub>ent �’s current action i<sub>s</sub> fi<sub>xe</sub>d t<sub>o</sub> $a _ { i } ,$ <sub>w</sub>hil<sub>e</sub> th<sub>e o</sub>th<sub>er agen</sub>t<sub>s</sub>’ <sub>curren</sub>t <sub>ac</sub>ti<sub>ons an</sub>d <sub>a</sub>ll l<sub>a</sub>t<sub>er</sub> <sub>ac</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow �.</sub> C<sub>ons</sub>id<sub>er</sub> $g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } } \cap$ Ψ. Assume that $\Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } )$ li<sub>es</sub> i<sub>n</sub> th<sub>e</sub> <sub>pa</sub>i<sub>re</sub>d l<sub>oca</sub>l<sub>-</sub>hi<sub>s</sub>t<sub>ory</sub> d<sub>oma</sub>i<sub>n,</sub> $\boldsymbol { \phi } _ { q , i } ^ { \mathcal { R } }$ <sup>b</sup>ijects t<sup>h</sup>e correspon<sup>d</sup>ing l<sub>ega</sub>l<sub>-ac</sub>ti<sub>on se</sub>t<sub>s, an</sub>d b<sub>o</sub>th l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-va</sub>l<sub>ue vec</sub>t<sub>ors are</sub> d<sub>e</sub>fi<sub>ne</sub>d<sub>.</sub> Pr<sub>opos</sub>iti<sub>o</sub>n E<sub>.</sub>4 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t $g \in \Phi _ { \mathrm { F R } } ^ { \alpha } ( M )$ if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if<sub>,</sub> f<sub>or every</sub> $\pi \in \Pi _ { * } ^ { \alpha } ( M )$ , <sup>a</sup>g<sup>ent �</sup>, <sup>time ste</sup>p $t ,$ <sub>an</sub>d l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory</sub> $h _ { i } ^ { t }$ <sub>,</sub> th<sub>ere</sub> i<sub>s a</sub> constant $c _ { g , \pi , i , h _ { i } ^ { t } }$ <sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $^ { a _ { i } , }$ <sub>suc</sub>h th<sub>a</sub>t<sub>:</sub>

$$
Q _ { i } ^ { g \cdot \pi } \left( \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) , \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \right) = Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } ) + c _ { g , \pi , i , h _ { i } ^ { t } }
$$

f<sub>or every</sub> l<sub>ega</sub>l $a _ { i } .$ <sub>.</sub> Th<sub>e cons</sub>t<sub>an</sub>t i<sub>s exac</sub>tl<sub>y</sub> th<sub>e</sub> f<sub>ree</sub>d<sub>om</sub> th<sub>a</sub>t l<sub>eaves a</sub> <sub>so</sub>ft<sub>max</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on unc</sub>h<sub>ange</sub>d<sub>.</sub> If<sub>, more s</sub>t<sub>rong</sub>l<sub>y, suc</sub>h <sub>a</sub> $_ 1 g$ i<sub>s a</sub>l<sub>so</sub> a com<sub>p</sub>onent<sub>w</sub>ise en<sub>v</sub>ironment a<sub>u</sub>tomor<sub>p</sub>hism<sub>,</sub> the constant is zero at histories havin<sub>g p</sub>ositive <sub>p</sub>robabilit<sub>y</sub> under the <sub>p</sub>aired <sub>p</sub>olicies. At an of-<sub>p</sub>ath histor<sub>y,</sub> meanin<sub>g</sub> a histor<sub>y</sub> with zero <sub>p</sub>robabilit<sub>y</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>re</sub>l<sub>evan</sub>t <sub>po</sub>li<sub>cy,</sub> th<sub>e</sub> <sub>same</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> <sub>requ</sub>i<sub>res</sub> <sub>one</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> fixed belief and local-� convention that is equivariant under the transformations bein<sub>g</sub> com<sub>p</sub>ared.

Because ERS as defined b<sub>y</sub> Mu<sub>g</sub>lich et al. [22] fixes a<sub>g</sub>ent indices, it d<sub>oes</sub> <sub>no</sub>t <sub>re</sub>l<sub>a</sub>t<sub>e</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>agen</sub>t<sub>s</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> <sub>canno</sub>t <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> structural a<sub>g</sub>ent t<sub>yp</sub>es defined in this <sub>p</sub>a<sub>p</sub>er. Those t<sub>yp</sub>es remain the a<sub>g</sub>ent orbits of the <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub> $G _ { \mathcal { M } } ^ { \mathrm { e v a l } }$ , as in D<sub>e</sub>finiti<sub>o</sub>n 4<sub>.</sub>2<sub>.</sub>

E.2.1 Boltzmann condition and comparison proofs. For the proofs b<sub>e</sub>l<sub>ow,</sub> fi<sub>x a</sub> t<sub>empera</sub>t<sub>ure</sub> $\alpha ~ > ~ 0$ <sub>an</sub>d <sub>use</sub> th<sub>e common a</sub>d<sub>m</sub>i<sub>ss</sub>i<sub>-</sub> bl<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> d<sub>oma</sub>i<sub>n</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>a</sub>b<sub>ove.</sub> W<sub>e a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub> $\Pi ^ { \alpha } ( M )$ $\Pi _ { * } ^ { \alpha } ( M )$ <sub>, an</sub>d $J ^ { \mathcal { M } } ( \pi )$ <sup>b</sup><sub>y</sub> $\Pi ^ { \alpha } , \Pi _ { * } ^ { \alpha }$ <sub>,</sub> <sub>an</sub>d $J ( \pi )$ <sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> R<sub>eca</sub>ll th<sub>a</sub>t $\Pi ^ { \alpha }$ contains the <sub>p</sub>olicies satisf<sub>y</sub>in<sub>g</sub> the Boltzmann condition

$$
\pi _ { i } ^ { t } ( a _ { i } \mid h _ { i } ^ { t } ) = \frac { \exp ( Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } ) / \alpha ) } { \sum _ { b _ { i } \in \mathcal { A } _ { i } } \exp ( Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , b _ { i } ) / \alpha ) } .
$$

Th<sub>e</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on va</sub>l<sub>ue</sub> $Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } )$ i<sub>s</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>e</sub>d <sub>re</sub>t<sub>urn-</sub> t<sub>o-go a</sub>ft<sub>er</sub> l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory</sub> $h _ { i } ^ { t }$ <sub>w</sub>h<sub>en agen</sub>t �’<sub>s curren</sub>t <sub>ac</sub>ti<sub>on</sub> i<sub>s</sub> fi<sub>xe</sub>d t<sub>o</sub> $^ { a _ { i } , }$ <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> <sub>curren</sub>t <sub>ac</sub>ti<sub>ons</sub> <sub>an</sub>d <sub>a</sub>ll l<sub>a</sub>t<sub>er</sub> <sub>ac</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> <sub>�.</sub> At <sub>a</sub> <sub>zero-pro</sub>b<sub>a</sub>bilit<sub>y</sub> hi<sub>s</sub>t<sub>ory</sub> it <sub>uses</sub> th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> d<sub>ec</sub>l<sub>are</sub>d <sub>o</sub>f<sub>-pa</sub>th b<sub>e</sub>li<sub>e</sub>f and local-� convention discussed below.

Proof of Proposition E.3. ERS without policy-evaluation symmetry. Choose $0 ~ < ~ \varepsilon ~ < ~ 2 \alpha .$ . Let $\textstyle { \mathcal { M } } _ { a }$ <sup>be</sup> <sup>a</sup> <sup>one-sta</sup>g<sup>e</sup>, <sup>two-a</sup>g<sup>ent</sup> D<sub>ec</sub>-POMDP <sub>w</sub>ith <sub>o</sub>n<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>s</sub>in<sub>g</sub>l<sub>e</sub>t<sub>o</sub>n <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>s,</sub> bin<sub>a</sub>r<sub>y</sub> <sub>ac</sub>ti<sub>o</sub>n<sub>s,</sub> deterministic d<sub>y</sub>namics<sub>,</sub> and common reward $R ( a ) = \varepsilon { 1 \{ a _ { 1 } = a _ { 2 } \} }$ If a<sub>g</sub>ent � selects action 1 with <sub>p</sub>robabilit<sub>y</sub> $\mathbf { \nabla } \mathcal { P } \mathbf { \dot { \iota } } .$ , t<sup>h</sup>en a<sub>g</sub>a<sup>i</sup>nst an a<sub>g</sub>ent selectin<sub>g</sub> action 1 with <sub>p</sub>robabilit<sub>y</sub> $q ,$

$$
Q _ { i } ^ { \pi } ( \ l ( \ l ) , 1 ) = \varepsilon q , \qquad Q _ { i } ^ { \pi } ( \ l ( \ l ) , 0 ) = \varepsilon ( 1 - q ) .
$$

B<sub>o</sub>lt<sub>zmann se</sub>lf<sub>-cons</sub>i<sub>s</sub>t<sub>ency</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub>

$$
p _ { 1 } = F ( p _ { 2 } ) , \qquad p _ { 2 } = F ( p _ { 1 } ) , \qquad F ( q ) = { \frac { 1 } { 1 + \exp [ - \varepsilon ( 2 q - 1 ) / \alpha ] } } .
$$

Sin<sub>ce</sub>

$$
| F ^ { \prime } ( q ) | \leq \frac { \varepsilon } { 2 \alpha } < 1 ,
$$

t<sup>h</sup>e joint response map $( p _ { 1 } , p _ { 2 } ) \mapsto ( F ( p _ { 2 } ) , F ( p _ { 1 } ) )$ is a contraction in the su<sub>p</sub> norm. Moreover<sub>,</sub> $F ( 1 / 2 ) = 1 / 2$ , so $( 1 / 2 , 1 / 2 )$ i<sub>s a</sub> fi<sub>xe</sub>d <sub>p</sub>oint. To see directl<sub>y</sub> that it is the onl<sub>y</sub> one<sub>,</sub> let $\mathcal { P }$ b<sub>e</sub> <sub>any</sub> <sub>o</sub>th<sub>er</sub> fi<sub>xe</sub>d <sub>po</sub>i<sub>n</sub>t <sub>an</sub>d l<sub>e</sub>t $L = \varepsilon / ( 2 \alpha ) < 1$ . The contraction bound <sub>g</sub>ives

$$
\begin{array} { r } { \| \boldsymbol { p } - \left( 1 / 2 , 1 / 2 \right) \| _ { \infty } \leq L \| \boldsymbol { p } - \left( 1 / 2 , 1 / 2 \right) \| _ { \infty } , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h f<sub>orces</sub> th<sub>e norm</sub> t<sub>o</sub> b<sub>e zero.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\Pi ^ { \alpha } = \Pi _ { * } ^ { \alpha } = \{ \pi ^ { \mathrm { u } } \} ,
$$

<sub>w</sub>h<sub>ere</sub> $\pi ^ { \mathrm { u } }$ is the uniform <sub>p</sub>roduct <sub>p</sub>olic<sub>y</sub>.

Let $g _ { a }$ <sub>swap</sub> th<sub>e</sub> t<sub>wo</sub> <sub>ac</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>o</sub>f <sub>agen</sub>t 1 <sub>an</sub>d fi<sub>x</sub> <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> D<sub>ec</sub>l<sub>are</sub> $\{ e , g _ { a } \}$ <sub>a</sub>dmi<sub>ss</sub>ibl<sub>e.</sub> Thi<sub>s</sub> <sub>w</sub>ithin-<sub>age</sub>nt <sub>ac</sub>ti<sub>o</sub>n relabelin<sub>g</sub> lies in Ψ and fixes $\pi ^ { \mathrm { u } }$ <sub>,</sub> h<sub>ence</sub>

$$
g _ { a } \in \Phi _ { \mathrm { E R } } ^ { \alpha } ( M _ { a } ) .
$$

For the deterministic <sub>p</sub>olic<sub>y</sub> $\pi ^ { 0 0 }$ <sub>,</sub> h<sub>owever,</sub>

$$
J ^ { M _ { a } } ( \pi ^ { 0 0 } ) = \varepsilon , \qquad J ^ { M _ { a } } ( g _ { a } \cdot \pi ^ { 0 0 } ) = 0 .
$$

The transformation fixes the a<sub>g</sub>ent indices and the uni<sub>q</sub>ue initial <sub>p</sub>air<sub>,</sub> so return e<sub>q</sub>uivariance would re<sub>q</sub>uire these two returns to be e<sub>q</sub>ual. Their ine<sub>q</sub>ualit<sub>y</sub> therefore violates <sub>p</sub>olic<sub>y</sub>-evaluation s<sub>y</sub>mmetr<sub>y</sub>, an<sup>d</sup>

$$
g _ { a } \notin G _ { M _ { a } } ^ { \mathrm { e v a l } , \rho _ { 0 } } .
$$

Policy-evaluation symmetry without ERS. Let $\mathcal { M } _ { b }$ <sup>b</sup>e a two-sta<sub>g</sub>e, t<sub>wo</sub>-<sub>age</sub>nt D<sub>ec</sub>-POMDP<sub>.</sub> U<sub>se</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> $\{ u , z \}$ . A<sub>g</sub>ent 1 has actions {0, 1} and observations {�, �, <sub>�</sub>}. A<sub>g</sub>ent 2 has sin<sub>g</sub>leton action and observation sets {★} and {⊥}. Set $\gamma = 1$ <sub>,</sub> l<sub>e</sub>t th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> b<sub>e</sub> $\delta _ { u }$ and abbreviate joint action (�, ★) b<sub>y</sub> �. For either action, take

$$
P ( u \mid u , a ) = P ( z \mid u , a ) = { \frac { 1 } { 2 } } , \qquad P ( z \mid z , a ) = 1 .
$$

A<sub>g</sub>ent 2 alwa<sub>y</sub>s observes ⊥, while a<sub>g</sub>ent 1’s action-inde<sub>p</sub>endent observation <sub>p</sub>robabilities are

<table><tr><td></td><td>C</td><td>x</td><td>y</td></tr><tr><td>u</td><td>1/5</td><td>2/5</td><td>2/5</td></tr><tr><td>z</td><td>1/5</td><td>1/5</td><td>3/5</td></tr></table>

Gi<sub>ve</sub> b<sub>o</sub>th <sub>agen</sub>t<sub>s</sub> th<sub>e common successor-s</sub>t<sub>a</sub>t<sub>e rewar</sub>d

$$
{ \cal R } ( s ^ { \prime } , ( a , \star ) ) = 2 1 \{ a = 0 , s ^ { \prime } = u \} .
$$

Let $g _ { b }$ <sub>exc</sub>h<sub>ange</sub> <sub>agen</sub>t 1’<sub>s</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>�</sub> <sub>an</sub>d <sub>�</sub> <sub>an</sub>d fi<sub>x</sub> <sub>ev-</sub> <sub>ery o</sub>th<sub>er coor</sub>di<sub>na</sub>t<sub>e.</sub> D<sub>ec</sub>l<sub>are</sub> $\{ e , g _ { b } \}$ admissible. This <sub>w</sub>ithin-a<sub>g</sub>ent observation relabelin<sub>g</sub> lies in Ψ.

F<sub>o</sub>r $a ^ { 0 } \in \{ 0 , 1 \} , o \in \{ c , x , y \}$ <sub>, an</sub>d <sub>e</sub>ith<sub>er common-rewar</sub>d <sub>coor</sub>di<sub>-</sub> <sub>na</sub>t<sub>e</sub> �<sub>, wr</sub>it<sub>e</sub>

$$
h ^ { 1 } ( a ^ { 0 } , o ) = \big ( ( a ^ { 0 } , o ) , ( \star , \bot ) \big ) , \qquad \mathcal { P } _ { a ^ { 0 } , o } = \pi _ { 1 } ^ { 1 } \big ( 0 \mid ( a ^ { 0 } , o ) \big ) .
$$

At � = 1,

$$
V _ { k , 1 } ^ { \pi } \big ( u , h ^ { 1 } ( a ^ { 0 } , o ) \big ) = p _ { a ^ { 0 } , o } , \qquad V _ { k , 1 } ^ { \pi } \big ( z , h ^ { 1 } ( a ^ { 0 } , o ) \big ) = 0 .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> f<sub>rom �, c</sub>h<sub>oos</sub>i<sub>ng ac</sub>ti<sub>on</sub> 0 <sub>g</sub>i<sub>ves rewar</sub>d 2 <sub>w</sub>h<sub>en</sub> th<sub>e nex</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> is �, which occurs with <sub>p</sub>robabilit<sub>y</sub> 1/2, and choosin<sub>g</sub> action 1 <sub>g</sub>ives z<sub>e</sub>r<sub>o.</sub> Fr<sub>o</sub>m $z ,$ both actions <sub>g</sub>ive zero. The <sub>p</sub>olic<sub>y</sub> transformation exc<sup>h</sup>an<sub>g</sub>es $\phi _ { a ^ { 0 } , x }$ <sub>an</sub>d $ { p _ { a ^ { 0 } , y } }$ . For exam<sub>p</sub>le<sub>,</sub>

$$
V _ { k , 1 } ^ { g _ { b } \cdot \pi } \big ( u , g _ { b } \cdot h ^ { 1 } ( a ^ { 0 } , x ) \big ) = p _ { a ^ { 0 } , x } = V _ { k , 1 } ^ { \pi } \big ( u , h ^ { 1 } ( a ^ { 0 } , x ) \big ) .
$$

The same calculation a<sub>pp</sub>lies to histories endin<sub>g</sub> in $y ,$ <sub>w</sub>hil<sub>e</sub> <sub>�</sub> i<sub>s</sub> fi<sub>xe</sub>d <sub>an</sub>d b<sub>o</sub>th <sub>va</sub>l<sub>ues</sub> f<sub>rom � are zero.</sub> H<sub>ence</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng va</sub>l<sub>ue-</sub> trans<sup>f</sup>ormation i<sup>d</sup>entity <sup>h</sup>o<sup>ld</sup>s at every state–joint-<sup>h</sup>istory pair at $t = 1$ <sub>.</sub> At th<sub>e</sub> initi<sub>a</sub>l tim<sub>e</sub> <sub>s</sub>t<sub>ep</sub> � = 0 $t = 0 _ { \mathrm { { ; } } }$ , p<sup>uttin</sup>g $q = \pi _ { 1 } ^ { 0 } ( 0 \mid ( ) )$ <sub>g</sub><sup>i</sup>ves

$$
V _ { k , 0 } ^ { \pi } ( u , ( ) ^ { N } ) = q + \frac { 1 } { 2 } \sum _ { a ^ { 0 } \in \{ 0 , 1 \} } \pi _ { 1 } ^ { 0 } ( a ^ { 0 } \mid ( ) ) \left[ \frac { 1 } { 5 } / { p } _ { a ^ { 0 } , c } + \frac { 2 } { 5 } / { p } _ { a ^ { 0 } , x } + \frac { 2 } { 5 } / { p } _ { a ^ { 0 } , y } \right] .
$$

Exchan<sub>g</sub>in<sub>g</sub> <sub>�</sub> and <sub>�</sub> leaves this ex<sub>p</sub>ression unchan<sub>g</sub>ed. From state <sub>�,</sub> <sub>every</sub> <sub>presen</sub>t <sub>an</sub>d f<sub>u</sub>t<sub>ure</sub> <sub>rewar</sub>d i<sub>s</sub> <sub>zero.</sub> T<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> <sub>zero</sub> <sub>va</sub>l<sub>ues</sub> <sub>a</sub>t $t = 2 ,$ th<sub>ese</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>cover</sub> <sub>a</sub>ll ti<sub>me</sub> <sub>s</sub>t<sub>eps</sub> <sub>an</sub>d <sub>va</sub>lid state–joint-<sup>h</sup>istory pairs. T<sup>h</sup>e trans<sup>f</sup>ormation $g _ { b }$ <sub>a</sub>l<sub>so</sub> fi<sub>xes</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>so</sub> it <sub>p</sub>r<sub>ese</sub>r<sub>ves</sub> th<sub>e</sub> initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $\delta _ { u }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
g _ { b } \in G _ { M _ { b } } ^ { \mathrm { e v a l } , \rho _ { 0 } } .
$$

Th<sub>e</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on va</sub>l<sub>ues use</sub>d b<sub>y</sub> th<sub>e</sub> B<sub>o</sub>lt<sub>zmann con</sub>diti<sub>on</sub> i<sub>ns</sub>t<sub>ea</sub>d condition on a<sub>g</sub>ent 1’s observation. For exam<sub>p</sub>le<sub>,</sub> after either initial action $a ^ { 0 }$

$$
\operatorname* { P r } ( u \mid a ^ { 0 } , x ) = { \frac { ( 1 / 2 ) ( 2 / 5 ) } { ( 1 / 2 ) ( 2 / 5 ) + ( 1 / 2 ) ( 1 / 5 ) } } = { \frac { 2 } { 3 } } .
$$

The same Ba<sub>y</sub>es calculation <sub>g</sub>ives

$$
\operatorname* { P r } ( u \mid a ^ { 0 } , y ) = { \frac { 2 } { 5 } } , \qquad \operatorname* { P r } ( u \mid a ^ { 0 } , c ) = { \frac { 1 } { 2 } } .
$$

These <sub>p</sub>osterior <sub>p</sub>robabilities are also the action-value <sub>g</sub>a<sub>p</sub>s at $t = 1 { : }$ conditioned on a belief assi<sub>g</sub>nin<sub>g p</sub>robabilit<sub>y</sub> $\boldsymbol { p }$ t<sub>o �, ac</sub>ti<sub>on</sub> 0 h<sub>as</sub> <sub>expec</sub>t<sub>e</sub>d <sub>rewar</sub>d $2 p ( 1 / 2 ) = p { \mathrm { . } }$ <sub>, w</sub>h<sub>ereas ac</sub>ti<sub>on</sub> 1 h<sub>as va</sub>l<sub>ue zero.</sub> Th<sub>us</sub> th<sub>e requ</sub>i<sub>re</sub>d B<sub>o</sub>lt<sub>zmann pro</sub>b<sub>a</sub>biliti<sub>es o</sub>f <sub>ac</sub>ti<sub>on</sub> 0 <sub>a</sub>ft<sub>er � an</sub>d $y$ are

$$
b _ { x } = \frac { e ^ { 2 / ( 3 \alpha ) } } { 1 + e ^ { 2 / ( 3 \alpha ) } } , \qquad b _ { y } = \frac { e ^ { 2 / ( 5 \alpha ) } } { 1 + e ^ { 2 / ( 5 \alpha ) } } .
$$

E<sub>v</sub>er<sub>y</sub> initial action and e<sub>v</sub>er<sub>y</sub> res<sub>u</sub>ltin<sub>g</sub> obser<sub>v</sub>ation at $t = 1$ h<sub>ave</sub> <sub>p</sub>ositive <sub>p</sub>robabilit<sub>y</sub> under a Boltzmann <sub>p</sub>olic<sub>y,</sub> so the dis<sub>p</sub>la<sub>y</sub>ed <sub>p</sub>robabilities uni<sub>q</sub>uel<sub>y</sub> fix ever<sub>y</sub> rule at $t = 1$ . The <sub>p</sub>osterior calc<sub>u</sub>- l<sub>a</sub>ti<sub>on, an</sub>d h<sub>ence eac</sub>h <sub>o</sub>f th<sub>ese</sub> fi<sub>xe</sub>d <sub>ru</sub>l<sub>es,</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> initi<sub>a</sub>l <sub>ac</sub>ti<sub>o</sub>n<sub>.</sub> Sin<sub>ce</sub> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n <sub>a</sub>nd <sub>o</sub>b<sub>se</sub>r<sub>va</sub>ti<sub>o</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s</sub> <sub>an</sub>d th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on ru</sub>l<sub>es a</sub>t $t = 1$ are inde<sub>p</sub>endent of the initial action<sub>,</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>rewar</sub>d <sub>a</sub>t $t = 1$ i<sub>s</sub> th<sub>e sa</sub>m<sub>e a</sub>ft<sub>e</sub>r initi<sub>a</sub>l <sub>ac</sub>ti<sub>o</sub>n<sub>s</sub> 0 <sub>a</sub>nd 1<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e expec</sub>t<sub>e</sub>d <sub>rewar</sub>d<sub>s</sub> dif<sub>er</sub> b<sub>y</sub> $2 ( 1 / 2 ) = 1 { _ { ; } }$ <sub>, w</sub>hi<sub>c</sub>h <sub>un</sub>i<sub>que</sub>l<sub>y</sub> fi<sub>xes</sub> th<sub>e</sub> B<sub>o</sub>lt<sub>zmann ru</sub>l<sub>e a</sub>t th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l ti<sub>me s</sub>t<sub>ep as we</sub>ll<sub>.</sub> H<sub>e</sub>n<sub>ce</sub> $\Pi ^ { \alpha } = \Pi _ { * } ^ { \alpha }$ is a sin<sub>g</sub>leton. Since the lo<sub>g</sub>istic f<sub>u</sub>nction is strictl<sub>y</sub> in<sub>c</sub>r<sub>eas</sub>in<sub>g,</sub> $b _ { x } \ > \ b _ { y }$ <sub>.</sub> At <sub>a</sub> hi<sub>s</sub>t<sub>o</sub>r<sub>y e</sub>ndin<sub>g</sub> in <sub>�,</sub> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>e</sub>d <sub>p</sub>o<sup>li</sup>c<sub>y</sub> uses $b _ { y } ,$ <sub>a</sub>lth<sub>oug</sub>h B<sub>o</sub>lt<sub>zmann</sub> <sub>se</sub>lf<sub>-cons</sub>i<sub>s</sub>t<sub>ency</sub> <sub>requ</sub>i<sub>res</sub> $b _ { x } .$ Th<sub>ere</sub>f<sub>ore</sub>

$$
g _ { b } \notin \Phi _ { \mathrm { E R } } ^ { \alpha } ( M _ { b } ) ,
$$

which <sub>p</sub>roves the reverse non-im<sub>p</sub>lication.

□

Proposition E.4 (Local action-value bridge). Let � be a transformation such that

$$
g \in G _ { \mathcal { M } } ^ { \mathrm { e v a l } , \rho _ { 0 } } \cap \Psi .
$$

Assume that � preserves the paired local-history domain, matches the legal-action sets bijectively, and that both local action-value vectors are defined. Then $g \in \Phi _ { \mathrm { E R } } ^ { \alpha } ( M )$ if and only if, for every � $\cdot \in \Pi _ { * } ^ { \alpha } , i , t , h _ { i } ^ { t }$ there is an action-independent constant $c _ { g , \pi , i , h _ { i } ^ { t } }$ such that

$$
Q _ { i } ^ { g \cdot \pi } \left( \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) , \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \right) = Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } ) + c _ { g , \pi , i , h _ { i } ^ { t } }
$$

for every legal $a _ { i } . J f ,$ in addition, $g \in \mathrm { A u t _ { e n v } } ( \mathcal { M } )$ , then the identity holds with constant zero at positive-probability paired histories. The same conclusion at of-path histories requires a globally fixed belief and local- $\boldsymbol { \cdot } \boldsymbol { Q }$ convention that is equivariant under the relevant group.

Proof<sub>.</sub> B<sub>ecause �</sub> i<sub>s a</sub>n initi<sub>a</sub>l-di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n-<sub>p</sub>r<sub>ese</sub>r<sub>v</sub>in<sub>g po</sub>li<sub>cy</sub> evaluation s<sub>y</sub>mmetr<sub>y</sub> and $g \in \Psi$ fi<sub>xes</sub> <sub>agen</sub>t i<sub>n</sub>di<sub>ces,</sub>

$$
J ( g \cdot \pi ) = J ( \pi )
$$

for e<sub>v</sub>er<sub>y</sub> <sub>p</sub>olic<sub>y</sub>. Th<sub>u</sub>s<sub>,</sub> for $\pi \in \Pi _ { * } ^ { \alpha } ;$ <sub>,</sub> the transformed <sub>p</sub>olic<sub>y</sub> is a<sub>g</sub>ain o<sub>p</sub>timal <sub>w</sub>ithin $\Pi ^ { \alpha }$ exactl<sub>y</sub> <sub>w</sub>hen it remains in $\Pi ^ { \alpha }$

The <sub>p</sub>olic<sub>y</sub> transformation <sub>p</sub>reserves the action-<sub>p</sub>robabilit<sub>y</sub> vect<sub>o</sub>r <sub>a</sub>ft<sub>e</sub>r r<sub>e</sub>ind<sub>e</sub>xin<sub>g</sub> it<sub>s ac</sub>ti<sub>o</sub>n <sub>coo</sub>rdin<sub>a</sub>t<sub>es.</sub> T<sub>wo</sub> finit<sub>e</sub> B<sub>o</sub>ltzm<sub>a</sub>nn <sub>vec</sub>t<sub>ors</sub> <sub>agree</sub> <sub>exac</sub>tl<sub>y</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub>i<sub>r</sub> <sub>ac</sub>ti<sub>on</sub> <sub>va</sub>l<sub>ues</sub> dif<sub>er</sub> b<sub>y</sub> <sub>one</sub> <sub>ac</sub>ti<sub>on-</sub> inde<sub>p</sub>endent constant. Write the corres<sub>p</sub>ondin<sub>g</sub> action-value vectors as <sub>�</sub> an<sup>d</sup> $q ^ { \prime } ,$ <sub>,</sub> indexed b<sub>y</sub> corres<sub>p</sub>ondin<sub>g</sub> le<sub>g</sub>al actions. For a sin<sub>g</sub>leton action set the claim is immediate. Otherwise<sub>,</sub> e<sub>q</sub>ualit<sub>y</sub> of the <sub>p</sub>robabilit<sub>y</sub> ratios for an<sub>y</sub> two actions <sub>�,</sub> � <sub>g</sub>ives

$$
e ^ { ( q _ { a } ^ { \prime } - q _ { b } ^ { \prime } ) / \alpha } = e ^ { ( q _ { a } - q _ { b } ) / \alpha } ,
$$

<sub>an</sub>d h<sub>ence</sub> $q _ { a } ^ { \prime } - q _ { a } = q _ { b } ^ { \prime } - q _ { b }$ <sub>.</sub> Th<sub>us</sub> th<sub>e</sub> dif<sub>erence</sub> i<sub>s</sub> th<sub>e same</sub> f<sub>or</sub> ever<sub>y</sub> action. Conversel<sub>y,</sub> if $q _ { a } ^ { \prime } = q _ { a }$ + � for ever<sub>y</sub> action �, then

$$
\begin{array} { c } { { \displaystyle \frac { e ^ { q _ { a } ^ { \prime } / \alpha } } { \sum _ { b } e ^ { q _ { b } ^ { \prime } / \alpha } } = \frac { e ^ { c / \alpha } e ^ { q _ { a } / \alpha } } { e ^ { c / \alpha } \sum _ { b } e ^ { q _ { b } / \alpha } } } } \\ { { = \displaystyle \frac { e ^ { q _ { a } / \alpha } } { \sum _ { b } e ^ { q _ { b } / \alpha } } . } } \end{array}
$$

Th<sub>e</sub> <sub>common</sub> f<sub>ac</sub>t<sub>or</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>cance</sub>l<sub>s,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> th<sub>e</sub> <sub>same</sub> <sub>ac</sub>ti<sub>on</sub> <sub>pro</sub>b<sub>-</sub> <sub>a</sub>biliti<sub>es.</sub>

<sup>To</sup> p<sup>rove</sup> <sup>necessit</sup>y, <sup>su</sup>pp<sup>ose</sup> <sup>that</sup> $g \in \Phi _ { \mathrm { E R } } ^ { \alpha } ( M )$ . <sup>F</sup>or ever<sub>y</sub> $\pi \in \Pi _ { * } ^ { \alpha }$ it<sub>s</sub> im<sub>age</sub> $g \cdot \pi$ <sub>a</sub>l<sub>so</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $\Pi _ { * } ^ { \alpha }$ <sub>, so</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e</sub> B<sub>o</sub>lt<sub>z-</sub> mann condition. Their action <sub>p</sub>robabilities a<sub>g</sub>ree after reindexin<sub>g</sub> the action coordinates. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the <sub>p</sub>robabilit<sub>y</sub>-ratio ar<sub>g</sub>ument <sub>a</sub>t <sub>eac</sub>h <sub>pa</sub>i<sub>re</sub>d l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory g</sub>i<sub>ves</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>ac</sub>ti<sub>on-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on va</sub>l<sub>ues.</sub>

T<sub>o prove su</sub>fi<sub>c</sub>i<sub>ency, assume</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-va</sub>l<sub>ue</sub> id<sub>en</sub>tit<sub>y</sub> <sup>f</sup>or ever<sub>y</sub> $\pi \in \Pi _ { * } ^ { \alpha }$ and ever<sub>y p</sub>aired local histor<sub>y</sub>. Fix such a <sub>p</sub>olic<sub>y</sub> $\pi . \mathrm { A t }$ <sub>eac</sub>h <sub>pa</sub>i<sub>re</sub>d hi<sub>s</sub>t<sub>ory,</sub> th<sub>e cons</sub>t<sub>an</sub>t<sub>-s</sub>hift <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on s</sub>h<sub>ows</sub> th<sub>a</sub>t its reindexed action <sub>p</sub>robabilities are the Boltzmann <sub>p</sub>robabilities d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> $Q _ { i } ^ { g \cdot \pi }$ . These are <sub>p</sub>recisel<sub>y</sub> the <sub>p</sub>robabilities assi<sub>g</sub>ned <sup>b</sup><sub>y</sub> $g \cdot \pi ,$ <sub>so</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d <sub>po</sub>li<sub>cy</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> B<sub>o</sub>lt<sub>zmann</sub> <sub>con</sub>diti<sub>on</sub> <sub>a</sub>t <sub>every</sub> l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory.</sub> Th<sub>us</sub> $g \cdot \pi \in \Pi ^ { \alpha }$ <sub>.</sub> Sin<sub>ce</sub> $J ( g \cdot \pi ) = J ( \pi )$ it <sub>a</sub>tt<sub>a</sub>in<sub>s</sub> th<sub>e sa</sub>m<sub>e</sub> m<sub>a</sub>xim<sub>u</sub>m r<sub>e</sub>t<sub>u</sub>rn <sub>w</sub>ithin $\Pi ^ { \alpha }$ as $\pi ,$ <sub>an</sub>d h<sub>ence</sub> belon<sub>g</sub>s to Π<sup>�</sup>. Since � was arbitrar<sub>y</sub>, this <sub>p</sub>roves

$$
\{ g \cdot \pi : \pi \in \Pi _ { * } ^ { \alpha } \} \subseteq \Pi _ { * } ^ { \alpha } .
$$

To <sub>p</sub>rove the reverse inclusion<sub>,</sub> we must show that ever<sub>y</sub> <sub>p</sub>olic<sub>y</sub> in Π<sup>�</sup> is the ima<sub>g</sub>e of another <sub>p</sub>olic<sub>y</sub> in that set. Because <sub>�</sub> belon<sub>g</sub>s to the finite <sub>g</sub>rou<sub>p</sub> Ψ, there is an inte<sub>g</sub>er $m \geq 1$ <sub>suc</sub>h th<sub>a</sub>t $g ^ { m } = e ,$ where <sub>�</sub> is the identit<sub>y</sub> transformation. Take an<sub>y</sub> $\pi \in \Pi _ { * } ^ { \alpha }$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
\widetilde { \pi } = g ^ { m - 1 } \cdot \pi .
$$

The incl<sub>u</sub>sion alread<sub>y</sub> <sub>p</sub>ro<sub>v</sub>ed means that each a<sub>pp</sub>lication of <sub>�</sub> kee<sub>p</sub>s a <sub>p</sub>olic<sub>y</sub> in Π<sup>�</sup>. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> it $m - 1$ times therefore <sub>g</sub>ives $\widetilde \pi \in \Pi _ { * } ^ { \alpha } ;$ <sub>w</sub>h<sub>en</sub> $m = 1$ <sub>,</sub> this sim<sub>p</sub>l<sub>y</sub> means $\widetilde { \pi } = \pi$ . Moreover<sub>,</sub> the <sub>p</sub>olic<sub>y</sub> action law <sub>g</sub>ives

$$
g \cdot { \widetilde { \pi } } = g \cdot ( g ^ { m - 1 } \cdot \pi ) = g ^ { m } \cdot \pi = \pi .
$$

Th<sub>us</sub> <sub>eve</sub>r<sub>y</sub> $\pi \in \Pi _ { * } ^ { \alpha }$ has a <sub>p</sub>reima<sub>g</sub>e in $\Pi _ { * } ^ { \alpha }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e</sub> r<sub>eve</sub>r<sub>se</sub> in<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n<sub>.</sub> C<sub>o</sub>mbinin<sub>g</sub> th<sub>e</sub> t<sub>wo</sub> in<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n<sub>s</sub> <sub>g</sub>i<sub>ves</sub>

$$
\{ g \cdot \pi : \pi \in \Pi _ { * } ^ { \alpha } \} = \Pi _ { * } ^ { \alpha } .
$$

This is exactl<sub>y</sub> the set-<sub>p</sub>reservation re<sub>q</sub>uirement in the definition <sub>o</sub>f $\Phi _ { \mathrm { F R } } ^ { \alpha } ( M )$ <sub>,</sub> com<sub>p</sub>letin<sub>g</sub> the e<sub>q</sub>uivalence <sub>p</sub>roof.

For t<sup>h</sup>e zero-constant statement, pair eac<sup>h</sup> trajectory un<sup>d</sup>er � with its transformed counter<sub>p</sub>art under <sub>�</sub>·�. Com<sub>p</sub>onentwise <sub>p</sub>reservation<sub>,</sub> the <sub>p</sub>olic<sub>y</sub> transformation<sub>,</sub> and <sub>p</sub>reservation of the initial <sup>d</sup>istri<sup>b</sup>ution give t<sup>h</sup>e paire<sup>d</sup> trajectories t<sup>h</sup>e same pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity an<sup>d</sup> reward. Conditionin<sub>g</sub> at a <sub>p</sub>ositive-<sub>p</sub>robabilit<sub>y</sub> <sub>p</sub>aired histor<sub>y</sub> <sub>p</sub>re-<sub>serves</sub> thi<sub>s correspon</sub>d<sub>ence.</sub> It <sub>a</sub>l<sub>so pers</sub>i<sub>s</sub>t<sub>s w</sub>h<sub>en</sub> th<sub>e curren</sub>t l<sub>oca</sub>l <sub>ac</sub>ti<sub>on</sub> i<sub>s</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>a</sub>ll <sub>o</sub>th<sub>er curren</sub>t <sub>an</sub>d l<sub>a</sub>t<sub>er ac</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e</sub> res<sub>p</sub>ective <sub>p</sub>olicies. The two conditional ex<sub>p</sub>ected returns-to-<sub>g</sub>o are th<sub>ere</sub>f<sub>ore equa</sub>ll<sub>, suc</sub>h th<sub>a</sub>t<sub>:</sub>

$$
Q _ { i } ^ { g \cdot \pi } \left( \Phi _ { g , i } ^ { \mathcal { H } , t } ( h _ { i } ^ { t } ) , \phi _ { g , i } ^ { \mathcal { R } } ( a _ { i } ) \right) = Q _ { i } ^ { \pi } ( h _ { i } ^ { t } , a _ { i } ) .
$$

Positive reachabilit<sub>y</sub> makes the Ba<sub>y</sub>esian conditional law unambi<sub>g</sub>uous. At a zero-<sub>p</sub>robabilit<sub>y</sub> histor<sub>y,</sub> this reasonin<sub>g</sub> a<sub>pp</sub>lies onl<sub>y</sub> when the single of-path belief and local-� convention is itself equi-<sub>var</sub>i<sub>an</sub>t<sub>.</sub> A <sub>conven</sub>ti<sub>on may ass</sub>i<sub>gn we</sub>ll<sub>-</sub>d<sub>e</sub>fi<sub>ne</sub>d b<sub>e</sub>li<sub>e</sub>f<sub>s an</sub>d l<sub>oca</sub>l action values at ever<sub>y</sub> zero-<sub>p</sub>robabilit<sub>y</sub> histor<sub>y</sub> without makin<sub>g</sub> these assi<sub>g</sub>nments consistent with the transformation <sub>�</sub> (see the below ex<sub>p</sub>lanation for details). Therefore, well-definedness alone does <sub>no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on-va</sub>l<sub>ue</sub> <sub>equa</sub>lit<sub>y</sub> <sub>a</sub>t th<sub>ose</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>es.</sub> □

Clarification on Of-Path Histories. An of-path history can occur even when ever<sub>y</sub> action has <sub>p</sub>ositive <sub>p</sub>robabilit<sub>y</sub>. For exam<sub>p</sub>le<sub>,</sub> consider a two-sta<sub>g</sub>e, two-a<sub>g</sub>ent Dec-POMDP with states {�, �}, <sub>a</sub> <sub>u</sub>nif<sub>o</sub>rm initi<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>,</sub> <sub>a</sub>nd tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n<sub>s</sub> th<sub>a</sub>t l<sub>eave</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> unchan<sub>g</sub>ed. A<sub>g</sub>ent 1 has actions {0, 1} and observations {�, <sub>�</sub>}, but alwa<sub>y</sub>s observes <sub>�</sub> in either state<sub>;</sub> a<sub>g</sub>ent 2 has sin<sub>g</sub>leton action and <sub>o</sub>b<sub>serva</sub>ti<sub>on se</sub>t<sub>s.</sub> Th<sub>e common rewar</sub>d i<sub>s</sub> 1 <sub>w</sub>h<sub>en</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>agen</sub>t 1’s action are (�, 0) or (�, 1), and 0 otherwise. $\operatorname { A t } t = 1$ <sub>,</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> $h _ { 1 } ^ { 1 } = ( 0 , y )$ has <sub>p</sub>robabilit<sub>y</sub> zero under ever<sub>y p</sub>olic<sub>y,</sub> because <sub>�</sub> is im<sub>p</sub>ossible. To evaluate an action after this histor<sub>y,</sub> a<sub>g</sub>ent 1 needs a <sub>con</sub>diti<sub>ona</sub>l b<sub>e</sub>li<sub>e</sub>f <sub>a</sub>b<sub>ou</sub>t th<sub>e s</sub>t<sub>a</sub>t<sub>e.</sub> H<sub>owever,</sub> th<sub>e usua</sub>l <sub>con</sub>diti<sub>ona</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> f<sub>ormu</sub>l<sub>a wou</sub>ld <sub>requ</sub>i<sub>re</sub> th<sub>e un</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>ra</sub>ti<sub>o, suc</sub>h th<sub>a</sub>t<sub>:</sub>

$$
\frac { { \operatorname* { P r } } _ { \pi } ( s _ { 1 } = L , a _ { 1 } ^ { 0 } = 0 , o _ { 1 } ^ { 1 } = y ) } { { \operatorname* { P r } } _ { \pi } ( a _ { 1 } ^ { 0 } = 0 , o _ { 1 } ^ { 1 } = y ) } = \frac { 0 } { 0 } .
$$

Th<sub>us,</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>po</sub>li<sub>cy</sub> d<sub>o no</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne w</sub>hi<sub>c</sub>h b<sub>e</sub>li<sub>e</sub>f t<sub>o use</sub> <sub>a</sub>t thi<sub>s</sub> hi<sub>s</sub>t<sub>ory.</sub> T<sub>o</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on va</sub>l<sub>ues an</sub>d i<sub>mpose</sub> th<sub>e</sub>

B<sub>o</sub>lt<sub>zmann con</sub>diti<sub>on a</sub>t <sub>every</sub> l<sub>oca</sub>l hi<sub>s</sub>t<sub>ory, we mus</sub>t d<sub>ec</sub>l<sub>are a con-</sub> vention that assi<sub>g</sub>ns a belief at such zero-<sub>p</sub>robabilit<sub>y</sub> histories. For exam<sub>p</sub><sup>l</sup>e, at $t = 1$ <sub>,</sub> assi<sub>g</sub>nin<sub>g</sub> <sub>p</sub>robabilit<sub>y</sub> one to � <sub>g</sub>ives values 1 and 0 for actions 0 and 1<sub>,</sub> while assi<sub>g</sub>nin<sub>g p</sub>robabilit<sub>y</sub> one to � reverses th<sub>ese va</sub>l<sub>ues.</sub> Th<sub>ese are a</sub>lt<sub>erna</sub>ti<sub>ve ex</sub>t<sub>ens</sub>i<sub>ons</sub> t<sub>o an</sub> i<sub>mposs</sub>ibl<sub>e</sub> hi<sub>s-</sub> tor<sub>y,</sub> and neither assi<sub>g</sub>nment is determined b<sub>y</sub> Ba<sub>y</sub>es’ rule. Let <sub>�</sub> <sub>exc</sub>h<sub>ange</sub> � <sub>w</sub>ith � <sub>an</sub>d <sub>agen</sub>t 1’<sub>s</sub> <sub>ac</sub>ti<sub>ons</sub> 0 <sub>w</sub>ith 1<sub>,</sub> <sub>w</sub>hil<sub>e</sub> fi<sub>x</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> <sub>an</sub>d t<sub>a</sub>k<sub>e</sub> $\{ e , g \}$ as the admissible <sub>g</sub>rou<sub>p</sub>. This transformation <sub>p</sub>reserves the transition<sub>,</sub> observation<sub>,</sub> and commonreward functions and the initial distribution, and ma<sub>p</sub>s (0, <sub>�</sub>) to $( 1 , y )$ . Su<sub>pp</sub>ose a fixed convention assi<sub>g</sub>ns <sub>p</sub>robabilit<sub>y</sub> one to � at ever<sub>y</sub> zero-<sub>p</sub>robabilit<sub>y</sub> local histor<sub>y</sub> of a<sub>g</sub>ent 1<sub>,</sub> under ever<sub>y</sub> <sub>p</sub>olic<sub>y</sub>. Sin<sub>ce</sub> $t \ = \ 1$ i<sub>s</sub> th<sub>e</sub> fi<sub>na</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> ti<sub>me,</sub> th<sub>e resu</sub>lti<sub>ng</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>on</sub> <sub>va</sub>l<sub>ues sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
Q _ { 1 } ^ { \pi } ( ( 0 , y ) , 0 ) = 1 , \qquad Q _ { 1 } ^ { g \cdot \pi } ( ( 1 , y ) , 1 ) = 0 .
$$

Th<sub>us,</sub> thi<sub>s we</sub>ll<sub>-</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>conven</sub>ti<sub>on</sub> d<sub>oes no</sub>t <sub>preserve</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>ac</sub>ti<sub>o</sub>n-<sub>va</sub>l<sub>ue equa</sub>lit<sub>y.</sub> An <sub>equ</sub>i<sub>va</sub>ri<sub>a</sub>nt <sub>co</sub>n<sub>ve</sub>nti<sub>o</sub>n m<sub>us</sub>t in<sub>s</sub>t<sub>ea</sub>d <sub>as</sub>- si<sub>g</sub>n <sub>p</sub>robabilit<sub>y</sub> one to � at the transformed histor<sub>y</sub> under $g \cdot \pi$ whenever it assi<sub>g</sub>ns <sub>p</sub>robabilit<sub>y</sub> one to � at the source histor<sub>y</sub> under $\pi .$ Thi<sub>s</sub> i<sub>ssue concerns</sub> th<sub>e con</sub>diti<sub>ona</sub>l l<sub>oca</sub>l <sub>ac</sub>ti<sub>on va</sub>l<sub>ues use</sub>d i<sub>n</sub> the ERS com<sub>p</sub>arison. The com<sub>p</sub>lete val<sub>u</sub>es $V _ { i , t } ^ { \pi } ( s , h ^ { t } )$ <sub>use</sub>d t<sub>o</sub> d<sub>e</sub>fi<sub>ne</sub> our s<sub>y</sub>mmetr<sub>y</sub> <sub>g</sub>rou<sub>p</sub> are s<sub>p</sub>ecified directl<sub>y</sub> b<sub>y</sub> E<sub>q</sub>uation (1), including at unreac<sup>h</sup>a<sup>bl</sup>e state–joint-<sup>h</sup>istory pairs, an<sup>d</sup> require no suc<sup>h</sup> b<sub>e</sub>li<sub>e</sub>f <sub>conven</sub>ti<sub>on.</sub>