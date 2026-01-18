# Multi-Objective-Online-Convex-Optimization
In the classical framework of online convex optimization (OCO), a decision maker iteratively selects actions over a time horizon of $T$, without knowledge of the loss function revealed at each step. The performance of an online algorithm is measured in terms of _regret_, which compares the cumulative loss incurred by the algorithm to that of a static optimal action chosen in hindsight by an offline algorithm with full knowledge of the loss sequence.  

In this work, we extend this framework to a _multi-objective_ setting, where two distinct loss function sequences are revealed over time. This can be extended to k-distinct loss functions, but two functions are enough to capture the basic difficulty of this setup. At each time step $t$, the algorithm must choose an action before observing any of the $K$ losses. To account for the trade-offs among these objectives, we adopt the _min-max_ regret criterion: the benchmark is an offline algorithm that selects a single action for all time steps so as to minimize the maximum total loss across the $K$ sequences. The _min-max_ regret of an online algorithm is then defined as the difference between its cumulative cost and that of this benchmark. This stringent measure requires the algorithm to closely track all $K$ sequences simultaneously, ensuring balanced performance across all objectives.

<!-- .................................................................................................................................. -->
## Online Convex Optimization (OCO)

In the standard **Online Convex Optimization (OCO)** [Hazan 2019] framework, a learner sequentially selects decisions without prior knowledge of the corresponding loss functions. At each iteration $t \in [T]$, the learner chooses an action $x_t \in \mathcal{X}$, after which a convex loss function $f_t : \mathcal{X} \to \mathbb{R}$ is revealed, and the learner incurs a loss $f_t(x_t)$. The decision set $\mathcal{X} \subseteq \mathbb{R}^n$ is assumed to be convex and bounded, and the loss functions are drawn from a family $\mathcal{F}$ of convex, bounded functions.

The performance of an online algorithm $\mathcal{A}$ is measured via its *regret*, which compares the cumulative loss incurred by $\mathcal{A}$ to that of the best fixed decision in hindsight:

$$Regret_T(\mathcal{A}) = \sup_{\{f_1,\dots,f_T\} \subseteq \mathcal{F}} \{ \sum_{t=1}^{T} f_t(x_t^\mathcal{A}) - \min_{x \in \mathcal{X}} \sum_{t=1}^{T} f_t(x) \}.$$

Here, $x_t^\mathcal{A} = \mathcal{A}(f_1, \dots, f_{t-1})$ denotes the decision made by the algorithm at round $t$. The second term represents the total loss of the best static action chosen with full hindsight. The goal of any algorithm is to get sublinear Regret, like $O(\sqrt{T})$, $O(\ln{T})$ and so on.

It is well known that for convex, $G$-Lipschitz continuous loss functions defined over a convex set $\mathcal{X}$ of diameter $D$, any online algorithm incurs at least $\Omega(DG\sqrt{T})$ regret in the worst case.

### Preliminaries

* **Online Gradient Descent (OGD)** [Zinkevich 2003]: This algorithm is an online form of the standard gradient descent. This guarantees a $O(\sqrt{T})$ regret for the standard OCO problem.
* **Hedge** [Freund & Schapire 1997]: The Hedge algorithm is designed for situations where there are multiple (mostly finite) possible options, and we want to perform well without knowing in advance which one is best. It keeps track of all the strategies and assigns them weights based on their past performance. In each round, it combines the strategies in a way that gives higher probability to those that have done better, while still allowing for exploration of others. Over time, this approach ensures the algorithm performs nearly as well as the best strategy in hindsight.

<!-- .................................................................................................................................. -->

##  Problem Setup: Multi-Objective *min-max* Online Convex Optimization

The decision set is taken as a bounded convex set $\mathcal{X} \subseteq \mathbb{R}^n$. But instead of a single function, at iteration $t$, the learner receives two convex loss functions $f_t$ and $g_t$ for $t=1,2,\dots,T$.

We will be considering two different cases:
1.  The learner makes its move (i.e., chooses a point $x_t \in \mathcal{X}$) **after** both the functions are revealed.
2.  The learner makes its move **before** both the functions are revealed.

Two functions are considered for simplicity since they capture the basic difficulty. We aim to extend this to $k$-loss functions for any positive integer $k$.

### Definitions

The total loss suffered by an algorithm is defined to be:

$$
C_{\mathcal{A}} = \max \left\\{ \sum_{t=1}^{T} f_t(x_t), \sum_{t=1}^{T} g_t(x_t) \right\\}
$$

As a benchmark, we consider the static optimal offline algorithm **OPT** that knows $f_t$ and $g_t$ for $t = 1,2,\dots,T$ from the start, but can only choose a single action $x^{\star}$ across all times such that:

$$
x^{\star} = {argmin}_{x \in \mathcal{X}} \max \left\\{ \sum_{t=1}^{T} f_t(x), \sum_{t=1}^{T} g_t(x) \right\\}
$$

And the optimal cost is:

$$
C_{OPT} = \max \left\\{ \sum_{t=1}^{T} f_t(x^{\star}), \sum_{t=1}^{T} g_t(x^{\star}) \right\\}
$$

### Regret Formulation

Consequently, we define the *min-max* static regret of an algorithm $\mathcal{A}$ with actions $x_t$ as:

$$
\mathcal{R}_T(\mathcal{A}) = \sup_{f_t,g_t}(C_{\mathcal{A}} - C_{OPT}) = \sup_{f_t,g_t} \left\\{ \max \left\\{ \sum_{t=1}^{T} f_t(x_t), \sum_{t=1}^{T} g_t(x_t) \right\\} - \max \left\\{ \sum_{t=1}^{T} f_t(x^{\star}), \sum_{t=1}^{T} g_t(x^{\star}) \right\\} \right\\}
$$

---

**Remark:** The benchmark considered in this work is intentionally stringent. If we instead employ a weaker benchmark of the form:

$$
\min_{x \in \mathcal{X}}\sum_{t=1}^T \max\{f_t(x), g_t(x)\}
$$

the problem becomes considerably easier. In such a case, applying standard Online Gradient Descent (OGD) to the surrogate loss function $h_t(x) = \max\{f_t(x), g_t(x)\}$ suffices to achieve a regret bound of order $O(\sqrt{T})$.

However, this relaxed benchmark does not ensure a balanced treatment of multiple objectives and may lead to solutions that disproportionately favor one objective over others. The stricter benchmark adopted in our formulation inherently captures this trade-off, compelling the learner to optimize all objectives simultaneously, and thereby ensuring fairness across them.
