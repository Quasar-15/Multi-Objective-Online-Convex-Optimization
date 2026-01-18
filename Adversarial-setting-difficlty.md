## Difficulty in the Adversarial Setting

In this section, we illustrate the challenges of solving the *min-max* OCO problem in an adversarial setting, where the adversary can choose the sequence of loss functions in response to the online algorithm.

To highlight this difficulty, consider a favorable scenario in which the online algorithm has full knowledge of $f_t$ and $g_t$ **before** selecting its action $x_t$ at time $t$. Suppose the algorithm adopts a greedy strategy, selecting:

$$
x_t = \text{argmin}_{x \in \mathcal{X}} \max \{f_t(x), g_t(x)\}.
$$

Even under this idealized setting, the greedy approach can incur linear regret. To see this, let $T = 2N$ and the decision set be $\mathcal{X} = [0,1]$. For each $j \in \{1,2,\dots, N\}$, define the function sequences as:

$$
\begin{aligned}
f_{2j-1}(x) &= 1.2 - 0.2x, & f_{2j}(x) &= x, \\
g_{2j-1}(x) &= x, & g_{2j}(x) &= 0.8 + 0.2x.
\end{aligned}
$$

Then, the greedy choices satisfy:

$$
x_{2j-1} = 1 \quad \text{and} \quad x_{2j} = 0, \quad \forall j \in \{1,2,\dots, N\}.
$$

The total cost incurred by the greedy algorithm is $1.8N = 0.9T$. In contrast, the best fixed action in hindsight is $x^\star = 0$, yielding a cost of $1.2N = 0.6T$.

Hence, the static *min-max* regret of the greedy algorithm is:

$$
\text{Regret} = 0.9T - 0.6T = 0.3T
$$

which grows linearly with $T$. This example highlights the inherent difficulty of minimizing *min-max* static regret under worst-case inputs.
