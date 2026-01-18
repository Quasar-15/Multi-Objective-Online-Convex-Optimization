# OGD+Hedge

The benchmark cost can be equivalently expressed as:

$$
C_{OPT} = \min_{x \in \mathcal{X}} \max \left\\{ \sum_{t=1}^{T} f_t(x), \sum_{t=1}^{T} g_t(x) \right\\} = \min_{x \in \mathcal{X}} \max_{\theta \in \Delta_2} \sum_{t=1}^{T} \theta^{\mathsf{T}} l_t(x),
$$

where $\Delta_2$ denotes the $2$-dimensional simplex, i.e., $\theta = (\theta_1, \theta_2)$ such that $\theta_i \ge 0$ and $\theta_1 + \theta_2 = 1$, and $l_t(x) = (f_t(x), g_t(x))$.

Let $(x^{\star}, \theta^{\star})$ denote the optimal pair achieving the benchmark:

$$
(x^{\star}, \theta^{\star}) = \text{argmin}_{x \in \mathcal{X}} \max_{\theta \in \Delta_2} \sum_{t=1}^{T} \theta^{\mathsf{T}} l_t(x)
$$

Since the learner does not have access to future loss functions, computing $(x^{\star}, \theta^{\star})$ directly is infeasible. To approximate this pair in an online fashion, we propose an algorithm that generates $(x_t, \theta_t)$ sequentially over time.

The algorithm employs the **Hedge** method to update $\theta_t$, thereby addressing the inner maximization over $\Delta_2$ given $x_t$, and applies **Online Gradient Descent (OGD)** to update $x_t$, addressing the outer minimization given $\theta_t$.

Specifically, OGD is executed on the surrogate loss:

$$
\ell_t(x) = \theta_{t,1} f_t(x) + \theta_{t,2} g_t(x).
$$

The overall objective is to ensure that $(x_t, \theta_t)$ closely track the optimal pair $(x^{\star}, \theta^{\star})$.

## i.i.d. Input Model
