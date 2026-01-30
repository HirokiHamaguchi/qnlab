## Basic Concepts

One of the central goals of numerical optimization is to determine decision variables that optimize a given quantitative performance measure.
Examples include design performance, control stability, operational efficiency, and prediction error, each quantifying the quality of various phenomena or systems.
These performance measures are typically modeled by an objective function that maps decision variables to real numbers.

In this chapter, we define $f \colon \mathbb{R}^n \to \mathbb{R}$ as an objective function of class $C^2$, where $n$ denotes the dimension of the decision variables.
We consider the following unconstrained optimization problem:
$$
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x).
$$
In this section, we provide fundamental definitions and properties related to this problem.

### Convexity and Strong Convexity

The notions of convexity and strong convexity are fundamental in optimization theory.
Convexity and strong convexity of a function $f$ can be characterized by the following inequalities:
$$
\begin{aligned}
    	\text{(convex)} \quad
    f(y) & \ge f(x)+\nabla f(x)^\top (y-x), \\
    	\text{($\mu$-strongly convex)} \quad
    f(y) & \ge f(x)+\nabla f(x)^\top (y-x)+\frac{\mu}{2}\lVert y-x \rVert^2.
\end{aligned}
$$
for any $x,y \in \mathbb{R}^n$, where $\mu>0$ is a constant.
Strong convexity indicates that, in addition to convexity, the objective function has uniformly positive curvature.
Examples of convex and strongly convex functions are shown below.

![](../imgs/quasi_newton/convexity_comparison_convex.pdf)
![](../imgs/quasi_newton/convexity_comparison_strongly_convex.pdf)

Comparison of convex and strongly convex functions. The dashed line shows the quadratic approximation at $x=0$. The upper two functions are convex but not strongly convex, and no constant $\mu>0$ satisfies the strong convexity inequality. The lower two functions are strongly convex, and there exists $\mu>0$ that satisfies the strong convexity inequality.

### Positive Definiteness of the Hessian

Next, we show how convexity and strong convexity relate to the definiteness of the Hessian matrix $\nabla^2 f(x)$.
Let $A$ be a symmetric matrix in $\mathbb{R}^{n \times n}$. A matrix $A$ is called positive or negative definite (or semi-definite) based on the following conditions:
$$
\begin{aligned}
    	\text{(positive definite)} \quad      & v^\top A v > 0 \quad \forall v \in \mathbb{R}^n \setminus \{0\}, \\
    	\text{(positive semi-definite)} \quad & v^\top A v \ge 0 \quad \forall v \in \mathbb{R}^n,               \\
    	\text{(negative definite)} \quad      & v^\top A v < 0 \quad \forall v \in \mathbb{R}^n \setminus \{0\}, \\
    	\text{(negative semi-definite)} \quad & v^\top A v \le 0 \quad \forall v \in \mathbb{R}^n.
\end{aligned}
$$
A matrix is indefinite if it is neither positive nor negative definite.
For matrices $A,B \in \mathbb{R}^{n \times n}$, the notation $A \succeq B$ indicates that $A-B$ is positive semi-definite.
For $\mu > 0$, the condition $A \succeq \mu I$ is equivalent to $v^\top A v \ge \mu \lVert v \rVert^2$ for all $v \in \mathbb{R}^n$.
It directly implies that all eigenvalues of $A$ are at least $\mu$, which in turn implies that the operator norm $\norm{A}$ is at least $\mu$, i.e., $\norm{A} \geq \mu$.
In particular, if $B$ is a zero matrix, we simply write $A \succeq 0$.
We similarly define $\preceq$ for negative semi-definiteness.

We can relate convexity and strong convexity to the definiteness of the Hessian as follows.

**Proposition.** Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be of class $C^2$. Then

- $f$ is convex if and only if $\nabla^2 f(x)\succeq0$ holds for all $x \in \mathbb{R}^n$.
- $f$ is $\mu$-strongly convex if and only if $\nabla^2 f(x)\succeq\mu I$ holds for all $x \in \mathbb{R}^n$.

<details>
<summary>Proof</summary>

First, let $\mu>0$ and assume $\nabla^2 f(x)\succeq \mu I$ for all $x \in \mathbb{R}^n$.
By the fundamental theorem of calculus, for any $x,y \in \mathbb{R}^n$, we have
    $$
    f(y)
    = f(x)+\nabla f(x)^\top (y-x)
    +\frac{1}{2} \int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \, \mathrm{d}t.
    $$
    Then, we obtain
    $$
    \int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \, \mathrm{d}t
    \ge \int_0^1 \mu\lVert y-x \rVert^2 \, \mathrm{d}t
    = \mu\lVert y-x \rVert^2.
    $$
    Thus, substituting this into the previous equation yields the definition of $\mu$-strong convexity.

Conversely, if $f$ is $\mu$-strongly convex, for any $x \in \mathbb{R}^n$, $v \in \mathbb{R}^n$ and $t > 0$, letting $y=x \pm tv$ gives
    $$
    \begin{cases}
        f(x + tv)\ge f(x) + t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2, \\
        f(x - tv)\ge f(x) - t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2.
    \end{cases}
    $$
    By Taylor's theorem, there exists $s_\pm \in (0,1)$ such that
    $$
    \begin{cases}
        f(x + tv) = f(x) + t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x + s_+ t v) v, \\
        f(x - tv) = f(x) - t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x - s_- t v) v.
    \end{cases}
    $$
    Substituting these into the inequalities above and rearranging yields
    $$
    v^\top \frac{\nabla^2 f(x+ s_+ t v) + \nabla^2 f(x - s_- t v)}{2} v \ge \mu \lVert v \rVert^2.
    $$
    Letting $t \to 0$ in the above and using the continuity of $\nabla^2 f$ from the $C^2$ assumption gives
    $$
    v^\top \nabla^2 f(x) v \ge \mu \lVert v \rVert^2.
    $$
Because $v \in \mathbb{R}^n$ is arbitrary, we obtain $\nabla^2 f(x)\succeq \mu I$.

Setting $\mu=0$ in the above shows the convex case in the same way.
</details>

Quadratic functions whose Hessians are positive definite, indefinite, and negative definite are illustrated below. One can visually confirm the correspondence between positive definiteness and convexity.

![](../imgs/quasi_newton/pd.png)

Quadratic model $f(x)=\frac{1}{2}(x - x_k)^\top H (x - x_k) + \nabla f(x_k)^\top (x - x_k) + f(x_k)$ in 2-dimensional space with Hessians $H$ which is (left) positive definite, (center) indefinite, (right) negative definite.


### $L$-smoothness

Finally, we introduce $L$-smoothness of functions.
A function $f$ is $L$-smooth if
$$
\lVert \nabla f(x)-\nabla f(y) \rVert \le L\lVert x-y \rVert
$$
holds for all $x,y$.

The next proposition shows that $L$-smoothness can be characterized by the upper bound of the Hessian.
**Proposition.** Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be of class $C^2$. Then $f$ is $L$-smooth if and only if $\nabla^2 f(x)\preceq L I$ holds for all $x \in \mathbb{R}^n$.

<details>
<summary>Proof</summary>

Since $f$ is of class $C^2$, by the fundamental theorem of calculus, for any $x,y \in \mathbb{R}^n$, we have
    $$
    \nabla f(y) - \nabla f(x)
    = \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \, \mathrm{d}t.
    $$
Assume $\nabla^2 f(x)\preceq L I$ for all $x \in \mathbb{R}^n$.
    It implies that the operator norm of $\nabla^2 f(x)$ satisfies $\norm{\nabla^2 f(x)} \le L$, and thus
    $$
    \begin{aligned}
        \lVert \nabla f(y) - \nabla f(x) \rVert
         & = \left\lVert \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \, \mathrm{d}t \right\rVert \\
         & \le \int_0^1 \lVert \nabla^2 f(x+t(y-x))(y-x) \rVert \, \mathrm{d}t \\
         & \le \int_0^1 L\lVert y-x \rVert \, \mathrm{d}t \\
         & = L\lVert y-x \rVert,
    \end{aligned}
    $$
    which shows $L$-smoothness.
Conversely, if $f$ is $L$-smooth, then for any $x \in \mathbb{R}^n$ and $v \in \mathbb{R}^n$, we have
    $$
    \lVert \nabla f(x+tv)-\nabla f(x) \rVert \le L\lVert tv \rVert = Lt\lVert v \rVert
    $$
    By Taylor's theorem, we also have
    $$
    \nabla f(x+tv)-\nabla f(x) = t \nabla^2 f(x) v + r(t),
    $$
    where $r(t)$ satisfies $\norm{r(t)}/t \to 0$ as $t \to 0$, and we can rewrite it as
    $$
    \nabla^2 f(x) v = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x) -r(t)}{t} = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x)}{t}.
    $$
    Taking the inner product  with $v$ gives
    $$
    \begin{aligned}
        v^\top \nabla^2 f(x) v & = \lim_{t \to 0} \left(\frac{\nabla f(x+tv)-\nabla f(x)}{t}\right)^\top v \\
                               & \leq \lim_{t \to 0} \frac{\lVert \nabla f(x+tv)-\nabla f(x) \rVert}{t} \lVert v \rVert \\
                               & \leq L\lVert v \rVert^2.
    \end{aligned}
    $$
Because $v \in \mathbb{R}^n$ is arbitrary, we obtain $\nabla^2 f(x)\preceq L I$.
</details>

#### Baillon--Haddad Theorem

One of the useful properties of $L$-smooth functions is given by the following Baillon--Haddad theorem.
Note that only $C^1$ differentiability is required here.
**Proposition (Baillon--Haddad theorem).** Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be of class $C^1$. If $f$ is $L$-smooth and convex, then for all $x,y \in \mathbb{R}^n$, $\nabla f$ is $1/L$-cocoercive, i.e.,
$$
(\nabla f(x)-\nabla f(y))^\top (x-y) \ge \frac{1}{L} \lVert \nabla f(x)-\nabla f(y) \rVert^2
$$
for all $x,y \in \mathbb{R}^n$.

We refer to other literature for the proof.
One consequence of this theorem is that, for the sequences $\{x_k\}$ generated by an optimization algorithm, defining
$$
s_k \coloneqq x_{k+1}-x_k, \quad y_k \coloneqq \nabla f(x_{k+1}) - \nabla f(x_k)
$$
and applying the cocoercivity inequality with $x=x_{k+1}$ and $y=x_k$ yields
$$
s_k^\top y_k \ge \frac{1}{L} \lVert y_k \rVert^2.
$$
This inequality is sometimes used in the analysis of update formulas in methods such as BFGS and L-BFGS, and we will also use it in a later chapter.

#### Bound on Hessian Eigenvalues

The $L$-smoothness combined with $\mu$-strong convexity also provides bounds on the eigenvalues of the Hessian.
**Proposition.** Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be of class $C^2$.
If $f$ is $L$-smooth and $\mu$-strongly convex, then the eigenvalues of the Hessian $\nabla^2 f(x)$ are contained in the interval $[\mu, L]$ for all $x \in \mathbb{R}^n$.

<details>
<summary>Proof</summary>

By the previous propositions, we have
$$
\mu I \preceq \nabla^2 f(x) \preceq L I,
$$
which directly implies that the eigenvalues of $\nabla^2 f(x)$ are contained in the interval $[\mu, L]$.
</details>

As shown above, the $L$-smoothness and $\mu$-strong convexity impose upper and lower bounds on the eigenvalues of the Hessian, respectively.
