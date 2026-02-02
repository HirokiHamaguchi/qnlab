<!-- markdownlint-disable MD041 -->

# 準ニュートン法に関する基礎知識

<!-- From 1_basic.tex -->

## 連続最適化の基本概念

数理最適化の中心的な意義の一つは、与えられた定量的な指標を最適化する決定変数を求めることである。
そのような指標の例としては制御の安定性、運用の効率性、予測誤差など、さまざまな現象やシステムの品質を定量化したものが挙げられる。
これらの指標は通常、決定変数を実数に写像する目的関数としてモデル化される。

本章では、目的関数を $C^2$ 級の関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ とし、また $n$ を決定変数の次元と定める。 この時、次の無制約最適化問題を考える:

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x).
\end{equation*}
```

本節では、この最適化問題に関する基本的な定義と性質をまとめる。

### 凸性と強凸性

凸性(convexity)と強凸性(strong convexity)は最適化理論の基本概念である。
$f$ は $C^2$ 級と仮定する。
関数 $f$ が凸あるいは強凸であることは、任意の $x,y \in \mathbb{R}^n$ について次の定義が成り立つこととそれぞれ同値である:

```math
\begin{align*}
\text{(convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x),                                  \\
\text{($\mu$-strongly convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x)+\frac{\mu}{2}\lVert y-x \rVert^2,
\end{align*}
```

ただし、 $\mu>0$ は定数である。
強凸性は、凸性に加えて目的関数が一様に正の曲率を持つことを意味する。
凸関数と強凸関数の例を Fig. 1 に示した。

![../imgs/quasi_newton/convexity_comparison_convex.png](../imgs/quasi_newton/convexity_comparison_convex.png)

(Fig. 1 凸関数と強凸関数の例。 破線は $x=0$ における二次近似を示す。 上の2つの関数は凸であるが強凸ではない。 下の2つの関数は強凸であり、強凸性の定義を満たす $\mu>0$ が存在する。)

### Positive Definiteness of the Hessian

次に、凸性および強凸性がヘッセ行列 $\nabla^2 f(x)$ の正定値性とどのように関係するかを示す。
$A$ を $\mathbb{R}^{n \times n}$ の対称行列とする。行列 $A$ が正定値(positive definite)・負定値(negative definite) (または半正定値(positive semi-definite)・半負定値(negative semi-definite))であるとは、次の条件で定義される:

```math
\begin{align*}
\text{(positive definite)} \quad      & v^\top A v > 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\text{(positive semi-definite)} \quad & v^\top A v \ge 0 \quad \forall v \in \mathbb{R}^n,                 \\
\text{(negative definite)} \quad      & v^\top A v < 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\text{(negative semi-definite)} \quad & v^\top A v \le 0 \quad \forall v \in \mathbb{R}^n.
\end{align*}
```

正定値でも負定値でもない行列は不定値(indefinite)と呼ぶ。
行列 $A,B \in \mathbb{R}^{n \times n}$ に対して、$A \succeq B$ は $A-B$ が半正定値であることを表す。
特に $B$ が零行列のときは $A \succeq 0$ と書く。
同様に、$\preceq$ は半負定値に対して定義する。
$\mu \geq 0$ に対し、$A \succeq \mu I$ はすべての $v \in \mathbb{R}^n$ について $v^\top A v \ge \mu \lVert v \rVert^2$ と同値である。
これは $A$ のすべての固有値が少なくとも $\mu$ であることを意味し、さらに作用素ノルムについて $\lVert A \rVert \geq \mu$ であることを導く。

凸性および強凸性とヘッセ行列の正定値性との関係は次のとおりである。

**Proposition 1**

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。このとき
- $f$ が凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq0$ が成り立つことは同値である。
- $f$ が $\mu$-強凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq\mu I$ が成り立つことは同値である。

<details>
<summary>Proof</summary>

まず $\mu>0$ とし、任意の $x \in \mathbb{R}^n$ に対して $\nabla^2 f(x)\succeq \mu I$ が成り立つと仮定する。
微分積分学の基本定理より、任意の $x,y \in \mathbb{R}^n$ について次が成り立つ:

```math
\begin{equation*}
f(y)
= f(x)+\nabla f(x)^\top (y-x)
+\frac{1}{2} \int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t.
\end{equation*}
```

また、仮定 $\nabla^2 f(x)\succeq \mu I$ から次が得られる:

```math
\begin{equation*}
\int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t
\ge \int_0^1 \mu\lVert y-x \rVert^2 \mathrm{d}t
= \mu\lVert y-x \rVert^2.
\end{equation*}
```

以上の結果を合わせると、$\mu$-強凸性の定義が導かれる。
逆に、$f$ が $\mu$-強凸であると仮定する。
任意の $x \in \mathbb{R}^n, \ v \in \mathbb{R}^n$ および $t>0$ に対し、$y=x \pm tv$ とおくと次を得る:

```math
\begin{equation*}
\begin{cases}
f(x + tv)\ge f(x) + t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2, \\
f(x - tv)\ge f(x) - t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2.
\end{cases}
\end{equation*}
```

テイラーの定理より、ある $s_\pm \in (0,1)$ が存在して次が成り立つ:

```math
\begin{equation*}
\begin{cases}
f(x + tv) = f(x) + t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x + s_+ t v) v, \\
f(x - tv) = f(x) - t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x - s_- t v) v.
\end{cases}
\end{equation*}
```

これらの結果を合わせると次を得る:

```math
\begin{equation*}
v^\top \frac{\nabla^2 f(x+ s_+ t v) + \nabla^2 f(x - s_- t v)}{2} v \ge \mu \lVert v \rVert^2.
\end{equation*}
```

$t \to 0$ とし、$f$ が $C^2$ 級という仮定による $\nabla^2 f$ の連続性を用いると次を得る:

```math
\begin{equation*}
v^\top \nabla^2 f(x) v \ge \mu \lVert v \rVert^2.
\end{equation*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\succeq \mu I$ を得る。
上記の議論で $\mu=0$ とすれば、同様に凸の場合も示される。
\myQED

</details>

ヘッセ行列が正定値・不定値・負定値である二次関数を Fig. 2 に示す。正定値性と凸性の対応関係を視覚的に確認できる。

![../imgs/quasi_newton/pd.png](../imgs/quasi_newton/pd.png)

(Fig. 2 二次モデル $f(x)=\frac{1}{2}(x - x_k)^\top H (x - x_k) + \nabla f(x_k)^\top (x - x_k) + f(x_k)$ を2次元空間で示したもの。 ヘッセ行列 $H$ が (左)正定値、(中央)不定値、(右)負定値の場合を示す。)

### $L$-平滑性

最後に、関数の $L$-平滑性 ($L$-smoothness)を導入する。
関数 $f$ が $L$-平滑であるとは、ある定数 $L>0$ が存在して

```math
\begin{equation*}
\lVert \nabla f(x)-\nabla f(y) \rVert \le L\lVert x-y \rVert
\end{equation*}
```

が任意の $x,y$ について成り立つことと同値である。

次の命題は、$L$-平滑性がヘッセ行列の上界で特徴づけられることを示す。

**Proposition 2**

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。このとき $f$ が $L$-平滑であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ が成り立つことは同値である。

<details>
<summary>Proof</summary>

$f$ は $C^2$ 級なので、微分積分学の基本定理より任意の $x,y \in \mathbb{R}^n$ について次が成り立つ:

```math
\begin{equation*}
\nabla f(y) - \nabla f(x)
= \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t.
\end{equation*}
```

任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ と仮定する。
このとき $\lVert \nabla^2 f(x) \rVert \le L$ が成り立つので、

```math
\begin{align*}
\lVert \nabla f(y) - \nabla f(x) \rVert
& = \lVert \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t \rVert   &  & (\text{previous equation})   \\
& \le \int_0^1 \lVert \nabla^2 f(x+t(y-x))(y-x) \rVert \mathrm{d}t &  & (\text{triangle inequality}) \\
& \le \int_0^1 L\lVert y-x \rVert \mathrm{d}t                      &  & (\text{by assumption})       \\
& = L\lVert y-x \rVert,
\end{align*}
```

つまり、 $f$ の $L$-平滑性が示された。
逆に、$f$ が $L$-平滑であるとすると、任意の $x \in \mathbb{R}^n$ と $v \in \mathbb{R}^n$ に対して次が成り立つ:

```math
\begin{equation*}
\lVert \nabla f(x+tv)-\nabla f(x) \rVert \le L\lVert tv \rVert = Lt\lVert v \rVert.
\end{equation*}
```

さらにテイラーの定理より次が成り立つ:

```math
\begin{equation*}
\nabla f(x+tv)-\nabla f(x) = t \nabla^2 f(x) v + r(t),
\end{equation*}
```

ただし、$r(t)$ は $t \to 0$ のとき $\lVert r(t) \rVert/t \to 0$ を満たす。そして、これは次のように書き換えられる:

```math
\begin{equation*}
\nabla^2 f(x) v = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x) -r(t)}{t} = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x)}{t}.
\end{equation*}
```

$v$ との内積を取ると次が得られる:

```math
\begin{align*}
v^\top \nabla^2 f(x) v & = \lim_{t \to 0} \left(\frac{\nabla f(x+tv)-\nabla f(x)}{t}\right)^\top v                                                         \\
& \leq \lim_{t \to 0} \frac{\lVert \nabla f(x+tv)-\nabla f(x) \rVert}{t} \lVert v \rVert &  & (\text{Cauchy--Schwarz inequality})   \\
& \leq \lim_{t \to 0} \frac{L\lVert tv \rVert}{t} \lVert v \rVert                        &  & (\text{by $L$-smoothness definition}) \\
& = L \lVert v \rVert^2.
\end{align*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\preceq L I$ を得る。
\myQED

</details>

#### Baillon--Haddad Theorem

$L$-smooth 関数の有用な性質の一つが次の Baillon--Haddad 定理である。
ここでは $C^1$ の微分可能性だけを仮定する点に注意する。

**Proposition 3** (Baillon--Haddad theorem)

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^1$ 級とする。$f$ が $L$-smooth かつ凸であれば、任意の $x,y \in \mathbb{R}^n$ に対して $\nabla f$ は $1/L$-cocoercive であり、すなわち

```math
\begin{equation*}
(\nabla f(x)-\nabla f(y))^\top (x-y) \ge \frac{1}{L} \lVert \nabla f(x)-\nabla f(y) \rVert^2
\end{equation*}
```

が任意の $x,y \in \mathbb{R}^n$ について成り立つ。

証明は他の文献に譲る \citep{bauschkeBaillonHaddadTheoremRevisited2009} \citep[Proposition 12.60]{rockafellarVariationalAnalysis1998}。

この定理の帰結として、最適化アルゴリズムが生成する列 $\lbrace x_k \rbrace$ に対し、次を定義すると:

```math
\begin{equation*}
s_k \mathrel{\vcenter{:}}= x_{k+1}-x_k, \quad y_k \mathrel{\vcenter{:}}= \nabla f(x_{k+1}) - \nabla f(x_k)
\end{equation*}
```

Baillon--Haddad 定理を $x=x_{k+1}$、$y=x_k$ に適用して次を得る:

```math
\begin{equation*}
s_k^\top y_k \ge \frac{1}{L} \lVert y_k \rVert^2.
\end{equation*}
```

この不等式は BFGS や L-BFGS などの更新公式の解析で用いられることがある。

#### ヘッセ行列の固有値のバウンド

$L$-smoothness と $\mu$-強凸性を組み合わせると、ヘッセ行列の固有値に対するバウンドも得られる。

**Proposition 4**

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。
$f$ が $L$-smooth かつ $\mu$-強凸であれば、任意の $x \in \mathbb{R}^n$ に対してヘッセ行列 $\nabla^2 f(x)$ の固有値は区間 $[\mu, L]$ に含まれる。

<details>
<summary>Proof</summary>

\cref{prop:convexity-hessian,prop:smoothness-hessian} より次が成り立つ:

```math
\begin{equation*}
\mu I \preceq \nabla^2 f(x) \preceq L I.
\end{equation*}
```

これより $\nabla^2 f(x)$ の固有値が区間 $[\mu, L]$ に含まれることが直ちに従う。
\myQED

</details>

Proposition 4 で示したように、$L$-smoothness と $\mu$-強凸性はそれぞれヘッセ行列の固有値に上界と下界を与える。


<!-- From 2_newton.tex -->

## ニュートン法

### ニュートン法のアルゴリズム

本小節では、無制約最適化問題に対する基本的な最適化アルゴリズムであるニュートン法の概要を示す。
ニュートン法は、初期点 $x_0$ から出発し、現在点を逐次更新して列 $\lbrace x_k \rbrace_{k=0}^\infty$ を生成する代表的な反復アルゴリズムである。

$f$ は $C^2$ 級で強凸であると仮定し、このときヘッセ行列 $\nabla^2 f(x)$ は任意の $x \in \mathbb{R}^n$ で正定値となり可逆である。
$k$ 回目の反復点 $x_k$ における勾配 $g_k \mathrel{\vcenter{:}}= \nabla f(x_k)$ とヘッセ行列 $\nabla^2 f(x_k)$ を用いると、ニュートン法の基本手続きは次の更新を反復的に行うことである。

```math
\begin{equation*}
x_{k+1} \gets x_k - \alpha_k \nabla^2 f(x_k)^{-1} g_k
\end{equation*}
```

ここで $\alpha_k > 0$ は線形探索によって定められるステップサイズである。
この更新則の導出は次のとおりである。
$x_k$ まわりの $f$ の二次のテイラー近似は次式で与えられる。

```math
\begin{equation*}
m^*_{k}(x) \mathrel{\vcenter{:}}= f(x_k) + g_k^\top (x - x_k) + \frac{1}{2} (x - x_k)^\top \nabla^2 f(x_k) (x - x_k).
\end{equation*}
```

このモデルの勾配は次式である。

```math
\begin{equation*}
\nabla m^*_k(x) = g_k + \nabla^2 f(x_k)(x - x_k).
\end{equation*}
```

仮定よりこの二次モデルは強凸であるため、点 $x^* \in \mathbb{R}^n$ が $m^*_k$ の最小点であることと $\nabla m^*_k(x) = 0$ が成り立つことは同値である。したがって、この方程式を解くと次を得る。

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\mathrm{arg \ min}} \ m^*_{k}(x) = x_k -\left(\nabla^2 f(x_k)\right)^{-1} g_k.
\end{equation*}
```

この選択は自然に見えるが、後で見るように一般には大域収束を保証しない。
そこで、関数値の十分な減少を確保するため、線形探索で定めるステップサイズ $\alpha_k > 0$ を導入する。
これがニュートン法の基本手続きである。

更新方向 $d_k \mathrel{\vcenter{:}}= -\nabla^2 f(x_k)^{-1} g_k$ はニュートン方向と呼ばれる。ヘッセ行列が正定値であれば、次式より降下方向である。

```math
\begin{equation*}
g_k^\top d_k = -g_k^\top \nabla^2 f(x_k)^{-1} g_k < 0.
\end{equation*}
```

$f$ が非凸であり、ヘッセ行列 $\nabla^2 f(x_k)$ が負定値または不定値である場合、関数値の減少は保証されず、ニュートン法は収束しないことがある。
したがって、ニュートン法の適用ではヘッセ行列の正定値性の確認が重要である。
このような場合への一つの対処法として修正ニュートン法 \citep[Sec. 3.4]{nocedal1999numerical} がある。

### 収束に関する性質

次に、ニュートン法に対する標準的な収束定理を示す。

#### 大域収束

まず、一般的な大域収束の結果

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0
\end{equation*}
```

を線形探索付きの手法に対して述べ、その後にニュートン法への適用を説明する。
次の形の一般的な反復法を考える。

```math
\begin{equation*}
x_{k+1} \gets x_k + \alpha_k d_k,
\end{equation*}
```

ここで $d_k$ は $g_k^\top d_k < 0$ を満たす降下方向であり、$\alpha_k > 0$ は線形探索で決定されるステップサイズである。
ステップサイズ $\alpha_k$ は Wolfe 条件 \citep[Sec. 3.1]{nocedal1999numerical} により次のように定める。

```math
\begin{align*}
f(x_k + \alpha_k d_k) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k, \\
g_{k+1}^\top d_k      & \geq c_2 g_k^\top d_k,
\end{align*}
```

ここで $0 < c_1 < c_2 < 1$ は定数である。
また、方向 $d_k$ と負の勾配 $-g_k$ のなす角を $\theta_k \in [0, \pi]$ とし、次式を満たすように定める。

```math
\begin{equation*}
\cos \theta_k = \frac{-g_k^\top d_k}{\lVert g_k \rVert\lVert d_k \rVert}.
\end{equation*}
```

次の定理は古典的結果の簡略版である。

**Theorem 1** ({\cite[Theorem 3.2)

$f$ は $C^1$ 級であり $\mathbb{R}^n$ 上で下に有界で、かつ $L$-平滑であるとする。
次の反復法を考える。

```math
\begin{equation*}
x_{k+1} \gets x_k + \alpha_k d_k,
\end{equation*}
```

初期点 $x_0 \in \mathbb{R}^n$ から開始し、ステップサイズ $\alpha_k$ は Wolfe 条件で定められるとする。
角 $\theta_k$ に対して、ある正の定数 $\delta$ が存在して
$\cos \theta_k \geq \delta > 0$ がすべての $k$ で成り立つならば、
この反復法は次を満たす列 $\lbrace x_k \rbrace$ を生成する。

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0.
\end{equation*}
```

<details>
<summary>Proof</summary>

Wolfe 条件より次が成り立つ。

```math
\begin{equation*}
(g_{k+1} - g_k)^\top d_k
= g_{k+1}^\top d_k - g_k^\top d_k
\geq (c_2 - 1) g_k^\top d_k,
\end{equation*}
```

また

```math
\begin{align*}
(g_{k+1} - g_k)^\top d_k                      & \leq
\lVert g_{k+1} - g_k \rVert \lVert d_k \rVert &                                                      & \text{(Cauchy--Schwarz inequality)}                           \\
& \leq L \lVert x_{k+1} - x_k \rVert \lVert d_k \rVert &                                     & \text{($L$-smoothness)} \\
& = \alpha_k L \lVert d_k \rVert^2.
\end{align*}
```

これらを組み合わせると次を得る。

```math
\begin{equation*}
\alpha_k \geq \frac{c_2 - 1}{L} \frac{g_k^\top d_k}{\lVert d_k \rVert^2},
\end{equation*}
```

よって

```math
\begin{align*}
f(x_{k+1}) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k                                          &  & \text{(Wolfe condition)}          \\
& \leq f(x_k) - c_1 \frac{1 - c_2}{L} \frac{(g_k^\top d_k)^2}{\lVert d_k \rVert^2} &  & \text{(previous inequality)}      \\
& = f(x_k) - c_1 \frac{1 - c_2}{L} \cos^2 \theta_k \lVert g_k \rVert^2.            &  & \text{(definition of $\theta_k$)}
\end{align*}
```

この式を $k$ 以下のすべての添字について和を取ると次を得る。

```math
\begin{equation*}
f(x_{k+1}) \leq f(x_0) - c_1 \frac{1 - c_2}{L} \sum_{j=0}^k \cos^2 \theta_j \lVert g_j \rVert^2.
\end{equation*}
```

$f$ は下に有界なので、$f(x_0) - f(x_{k+1})$ はすべての $k$ である正の定数より小さい。
したがって、極限を取ることで Zoutendijk 条件が得られる。

```math
\begin{equation*}
\sum_{k=0}^\infty \cos^2 \theta_k \lVert g_k \rVert^2 < \infty.
\end{equation*}
```

この条件は次を含意する。

```math
\begin{equation*}
\cos^2 \theta_k \lVert g_k \rVert^2 \to 0.
\end{equation*}
```

$\cos \theta_k \geq \delta > 0$ がすべての $k$ で成り立つという仮定と合わせると、直ちに次が従う。

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0,
\end{equation*}
```

これで証明が完了する。

</details>

ニュートン法は $d_k = -\nabla^2 f(x_k)^{-1} g_k$ とした Theorem 1 の特別な場合なので、$f$ の適切な条件のもとでこの結果を適用できる。
特に $f$ が $L$-平滑かつ $\mu$-強凸であれば、任意の $k$ について次が成り立つ。

```math
\begin{equation*}
\cos \theta_k
= \frac{g_k^\top \nabla^2 f(x_k)^{-1} g_k}{\lVert g_k \rVert\lVert \nabla^2 f(x_k)^{-1} g_k \rVert}
\geq \frac{\lVert g_k \rVert^2 \lambda_{\min}(\nabla^2 f(x_k)^{-1})}{\lVert g_k \rVert^2 \lambda_{\max}(\nabla^2 f(x_k)^{-1})}
\geq \frac{\mu}{L},
\end{equation*}
```

ここで $\lambda_{\min}(\cdot)$ と $\lambda_{\max}(\cdot)$ はそれぞれ最小固有値と最大固有値を表す。
したがって、Theorem 1 において $\delta = \mu / L$ とすれば、Wolfe 条件を満たす線形探索のもとで、強凸かつ平滑な関数に対するニュートン法の大域収束が得られる。

#### 局所二次収束

次に、ニュートン法の局所収束速度に関する古典的結果を示す。
ここでも簡潔さのために簡略版を示す。

**Theorem 2** ({\cite[Theorem 3.5)

ヘッセ行列 $\nabla^2 f(x)$ が解 $x^*$ の近傍で Lipschitz 連続であり、十分な二次の最適性条件が成り立つとする。すなわち $\nabla f(x^*)=0$ かつ $\nabla^2 f(x^*)$ は正定値である。
すべての $k$ で $\alpha_k=1$ とし、初期点 $x_0$ が $x^*$ に十分近いとき、勾配ノルム列 $\lbrace\lVert \nabla f(x_k) \rVert\rbrace$ は二次収束する。

<details>
<summary>Proof</summary>

ニュートンステップの定義と最適性条件 $\nabla f(x^*)=0$ より次を得る。

```math
\begin{align*}
x_{k+1} - x^*
& = x_k - x^* - \left(\nabla^2 f(x_k)\right)^{-1}\nabla f(x_k)                                                        \\
& = \left(\nabla^2 f(x_k)\right)^{-1} \left(\nabla^2 f(x_k)(x_k-x^*)-\left(\nabla f(x_k)-\nabla f(x^*)\right)\right).
\end{align*}
```

テイラーの定理と三角不等式より次が成り立つ。

```math
\begin{align*}
& \lVert \nabla^2 f(x_k)(x_k-x^*)-\left(\nabla f(x_k)-\nabla f(x^*)\right) \rVert                    \\
=   {}  & \lVert \nabla^2 f(x_k)(x_k-x^*)-\int_0^1 \nabla^2 f(x_k+t(x^*-x_k)) (x_k - x^*)\mathrm{d}t \rVert  \\
\leq {} & \int_0^1 \lVert \nabla^2 f(x_k)-\nabla^2 f(x_k+t(x^*-x_k)) \rVert\lVert x_k-x^* \rVert\mathrm{d}t.
\end{align*}
```

$\nabla^2 f$ が定数 $L^{\mathrm{H}}$ を持つ Lipschitz 連続であれば、被積分関数は $L^{\mathrm{H}}t\lVert x_k-x^* \rVert$ で上から抑えられ、積分により次を得る。

```math
\begin{equation*}
\lVert \nabla^2 f(x_k)(x_k-x^*)-\left(\nabla f(x_k)-\nabla f(x^*)\right) \rVert
\le \frac{1}{2}L^{\mathrm{H}}\lVert x_k-x^* \rVert^2.
\end{equation*}
```

$\nabla^2 f(x^*)$ は非特異なので、ある半径 $r>0$ が存在して $\lVert x_k-x^* \rVert\le r$ を満たすすべての $x_k$ について
\[
\lVert \left(\nabla^2 f(x_k)\right)^{-1} \rVert\le 2\lVert \left(\nabla^2 f(x^*)\right)^{-1} \rVert
\]
が成り立つ。
これらを合わせると次を得る。

```math
\begin{equation*}
\lVert x_{k+1} - x^* \rVert
\le L^{\mathrm{H}}\lVert \left(\nabla^2 f(x^*)\right)^{-1} \rVert \lVert x_k-x^* \rVert^2.
\end{equation*}
```

$\widetilde L  \mathrel{\vcenter{:}}=  L^{\mathrm{H}}\lVert \left(\nabla^2 f(x^*)\right)^{-1} \rVert$ とおく。
初期点が $\lVert x_0-x^* \rVert\le \min\lbrace r, 1/(2\widetilde L) \rbrace$ を満たすように選ばれていれば、帰納法により $\lbrace x_k \rbrace$ は近傍内に留まり $x^*$ に収束する。
上の誤差評価は $\lbrace x_k \rbrace$ の二次収束を示す。
勾配ノルムの二次収束を示すために、$x_{k+1}-x_k=-\left(\nabla^2 f(x_k)\right)^{-1}\nabla f(x_k)$ と
$\nabla f(x_k)+\nabla^2 f(x_k)(x_{k+1}-x_k)=0$ を用いると次を得る。

```math
\begin{align*}
\lVert \nabla f(x_{k+1}) \rVert
& = \lVert \nabla f(x_{k+1})-\nabla f(x_k)-\nabla^2 f(x_k)(x_{k+1}-x_k) \rVert                                   \\
& \le \int_0^1 \lVert \nabla^2 f(x_k+t (x_{k+1}-x_k))-\nabla^2 f(x_k) \rVert\lVert x_{k+1}-x_k \rVert\mathrm{d}t \\
& \le \frac{1}{2}L^{\mathrm{H}}\lVert x_{k+1}-x_k \rVert^2                                                       \\
& \le \frac{1}{2}L^{\mathrm{H}}\lVert \left(\nabla^2 f(x_k)\right)^{-1} \rVert^2\lVert \nabla f(x_k) \rVert^2    \\
& \le 2 L^{\mathrm{H}} \lVert \left(\nabla^2 f(x^*)\right)^{-1} \rVert^2\lVert \nabla f(x_k) \rVert^2.
\end{align*}
```

よって $\lVert \nabla f(x_k) \rVert$ は0へ二次収束する。
\myQED

</details>

これらの命題 \cref{thm:line-search-global-convergence,thm:newton-quadratic} は、線形探索を備えたニュートン法が適切な条件のもとで大域収束と局所二次収束の両方の性質を持つことを示している。
この高速な収束は、通常は線形収束にとどまる勾配降下法などの一階法と比べたときの大きな利点である。
しかし、ヘッセ行列 $\nabla^2 f(x_k) \in \mathbb{R}^{n \times n}$ の計算と線形方程式 $\nabla^2 f(x_k) d_k = -\nabla f(x_k)$ の解法には $\order{n^3}$ の計算時間が必要であり、大規模問題では過大な計算コストとなる。
そのため、準ニュートン法などの他の最適化手法も用いられる。詳細は後で述べる。

### 大域収束のための要素

先に述べたように、ヘッセ行列の正定値性と線形探索によるステップサイズの選択はニュートン法で重要な役割を果たす。
本小節では、これらが必要となる理由を説明する。

#### ヘッセ行列の正定値性

ここまで $f$ が強凸であると仮定してきた。これは各反復でヘッセ行列 $\nabla^2 f(x_k)$ が正定値であることを意味する。
この仮定は、ニュートン法が局所的に最適解へ収束するために重要である。
ヘッセ行列が正定値であれば、次式よりニュートン法は最適解へ向かう降下方向を与える。

```math
\begin{equation*}
\nabla f(x_k)^\top d_k
= -\nabla f(x_k)^\top \left(\nabla^2 f(x_k)\right)^{-1} \nabla f(x_k)
< 0.
\end{equation*}
```

一方、ヘッセ行列が負定値または不定値の場合、関数値の減少は保証されず、ニュートン法が最適解から離れる方向を指すことがある。
不定値の場合は関数値が減少することもあるが、鞍点へ収束するリスクも高まる。
したがって、ニュートン法を適用する際にはヘッセ行列の正定値性の確認が重要である。

#### フルステップの問題

ニュートン法では、各反復でステップサイズを $\alpha_k=1$ とすることが困難を引き起こす場合がある。
ここではこの問題を示す例を挙げ、対処として線形探索が必要であることを述べる。

##### ニュートン法が発散する例

次の関数を考える。

```math
\begin{align*}
f(x)   & = \sqrt{1 + x^2},            \\
f'(x)  & = \frac{x}{\sqrt{1 + x^2}},  \\
f''(x) & = \frac{1}{(1 + x^2)^{3/2}}.
\end{align*}
```

初期点の絶対値が 1 を超えると、Fig. 3 に示すようにニュートン法は発散する。

![../imgs/quasi_newton/newton_failure_sqrt_function_1.1.png](../imgs/quasi_newton/newton_failure_sqrt_function_1.1.png)

(Fig. 3 \ifEn An example where Newton's method diverges with initial point $x_0=1.1$\else 初期点 $x_0=1.1$ でニュートン法が発散する例\fi)

##### 強凸関数に対するニュートン法の振動

前の例では目的関数は強凸ではなく、必ずしも良い性質を持っていなかった。
しかし、強凸性を持つ関数であってもニュートン法が収束しない例が存在する \citep[Example 1.4.3]{Doikov2021SecondOrderTensor}。

この例では $\mu>0$ に対し、次の関数を考える。

```math
\begin{align*}
f(x)     & = \log(1 + e^x) - \frac{x}{2} + \frac{\mu x^2}{2}, \\
f'(x)    & = \frac{e^x}{1+e^x} - \frac{1}{2} + \mu x,         \\
f''(x)   & = \frac{e^x}{(1+e^x)^2} + \mu,                     \\
f'''(x)  & = \frac{e^x(1 - e^x)}{(1+e^x)^3},                  \\
f''''(x) & = \frac{e^x(1 - 4e^x + e^{2x})}{(1+e^x)^4}.
\end{align*}
```

この関数は次の性質を持つ。

1. $\mu$-強凸である。
2. $\max_x |f''(x)| = \frac{1}{4} + \mu$ であり、これは $e^x=1$ で達成される。したがって $\nabla f$ は $L=\frac{1}{4}+\mu$ の $L$-平滑である。
3. $\max_x |f'''(x)| = \frac{1}{6\sqrt{3}}$ であり、これは $e^x=2-\sqrt{3}$ で達成される。したがって $\nabla^2 f$ は $M=\frac{1}{6\sqrt{3}}$ の $M$-Lipschitz である。

それにもかかわらず、初期点 $x_0$ が $\mu$ に対して十分大きい場合、Fig. 4 に示すようにニュートン法は振動する。

![../imgs/quasi_newton/newton_failure_strongly_convex_function_0.1_-4.png](../imgs/quasi_newton/newton_failure_strongly_convex_function_0.1_-4.png)

(Fig. 4 (左) $x_0=-4, \ \mu=0.1$ ではニュートン法が収束する。(右) $x_0=-4, \ \mu=0.01$ ではニュートン法が振動する。)

##### 線形探索の必要性

上記の問題を避けるため、線形探索を用いてステップサイズ $\alpha_k$ を適切に選ぶことが一般的である。
線形探索を備えたニュートン法は修正ニュートン法と呼ばれることが多く、大域収束性を持つことが知られている。

### 根探索としてのニュートン法との比較

概念的な補足として、``根探索としてのニュートン法'' と ``最適化におけるニュートン法'' の関係を簡潔に整理する。
両者は密接に関係するが、視点と適用対象が異なる。

#### 根探索としてのニュートン法

単にニュートン法(あるいはニュートン・ラフソン法)と言うときは、微分可能なスカラー関数 $g\colon \mathbb{R} \to \mathbb{R}$ に対して $g(x) = 0$ を解く根探索アルゴリズムを指すことが多い。
これは本節の主題ではないが、より基本的で広く知られている。

初期値 $x_0$ から開始すると反復は次式で与えられる。

```math
\begin{equation*}
x_{k+1} = x_k - \frac{g(x_k)}{g'(x_k)}.
\end{equation*}
```

幾何学的には、$x_k$ 近傍での $g$ のグラフを接線で近似し、その接線と $x$ 軸の交点を次の近似解として選ぶことに対応する。

#### 最適化におけるニュートン法

最適化の文脈でのニュートン法は、二回微分可能な関数 $f\colon \mathbb R^{n} \to\mathbb{R}$ の局所最小点、または同値に必要条件 $\nabla f(x) = 0$ を満たす停留点を求めるアルゴリズムを指す。
これが本節の主題である。

各反復でヘッセ行列 $\nabla^2 f(x)$ が正定値であるという仮定を思い出そう。
最適化におけるニュートン法は、根探索の枠組みを勾配 $\nabla f(x)$ に適用する。

```math
\begin{equation*}
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
\end{equation*}
```

スカラーの場合、これは $x_{k+1}=x_k - f'(x_k) / f''(x_k)$ に簡約される。

#### 二つの定式化の関係

![../imgs/quasi_newton/newton_raphson.png](../imgs/quasi_newton/newton_raphson.png)

(Fig. 5 \ifEn Root-finding for the gradient $\nabla f(x)=3 x^2 - 4 x + 1$ and optimization of the function $f(x)=x^3 - 2 x^2 + x$\else 勾配 $\nabla f(x)=3 x^2 - 4 x + 1$ の根探索と関数 $f(x)=x^3 - 2 x^2 + x$ の最適化\fi)

具体例を通じて両者の等価性を確認しよう。
\Cref{fig:newton_raphson} は単純な三次関数 $f(x)$ を用いて両者の関係を示している。
左図は $g(x) = \nabla f(x)$ に対する根探索であり、右図は $f(x)$ に対する最適化である。

根探索の定式化では、$\nabla f(x) = 3x^2 - 4x + 1$ を接線で近似し、すなわち

```math
\begin{equation*}
\nabla m^*_k(x) = \nabla f(x_k) + \nabla^2 f(x_k)(x - x_k),
\end{equation*}
```

この線形モデルの根を $x_{k+1}$ として選ぶ。

一方、最適化の定式化では、$f(x) = x^3 - 2x^2 + x$ を二次のテイラー展開で近似し、

```math
\begin{equation*}
m^*_k(x) = f(x_k) + \nabla f(x_k)(x - x_k) + \frac{1}{2}\nabla^2 f(x_k)(x - x_k)^2,
\end{equation*}
```

この二次モデルの最小点を $x_{k+1}$ とする。

したがって、正定値性の仮定のもとでは局所解の性質から両者が本質的に等価な操作を行うことが分かり、その対応関係を明示的に確認できる。

以下では最適化の定式化のみを扱う。


<!-- From 3_quasi_newton.tex -->

## 準ニュートン法

![../imgs/quasi_newton/newton_vs_qs_vs_gd.png](../imgs/quasi_newton/newton_vs_qs_vs_gd.png)

(Fig. 6 ニュートン法、準ニュートン法、勾配降下法の比較)

本節では準ニュートン法について述べる。準ニュートン法はニュートン法に基づくが、その主要な欠点であるヘッセ行列の計算コストの大きさを低減することを目的とする。具体的には真のヘッセ行列 $\nabla^2 f(x_k)$ の代わりに近似行列 $B_k$ を用い、高速な収束性を保ちつつ計算コストを抑える。

\Cref{fig:newton_vs_qs_vs_gd} はニュートン法、準ニュートン法、勾配降下法の比較を示す。勾配降下法は勾配の反対方向に単純に更新する方法である。1反復あたりの計算コストは最小だが収束は遅い。ニュートン法は反復回数が最も少ない一方で、1ステップあたりの計算コストが最も高い。準ニュートン法はその中間に位置し、両者のバランスを取る。特に反復回数ではなく計算時間で比較すると、準ニュートン法が最良の性能を示すことが多い。

線形探索に基づく準ニュートン法は、最適解 $x^*$ に収束する列 $\lbrace x_k \rbrace_{k=0}^{\infty}$ を次のように生成する。

```math
\begin{equation*}
x_{k+1} = x_k - \alpha_k B_k^{-1} \nabla f(x_k)
= x_k - \alpha_k H_k \nabla f(x_k)
\end{equation*}
```

ここで $\alpha_k > 0$ は線形探索で定めるステップサイズであり、$B_k$ は点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似である。$H_k \mathrel{\vcenter{:}}= B_k^{-1}$ はその逆行列を表す。

![../imgs/quasi_newton/quasi_newton_1.png](../imgs/quasi_newton/quasi_newton_1.png)

(Fig. 7 準ニュートン法の概念図。 (1) 目的関数 $f$ (青い曲面) と現在点 $x_k$ (赤点)。 (2) 現在のヘッセ近似によって得られる二次モデル (橙色の曲面) とその最小点 $x_{k+1}$ (黄色のバツ印)。 (3) 新しい点 $x_{k+1}$ に基づく更新後の二次モデル (緑色の曲面)。)

準ニュートン法の核心は、各反復で $B_k$ (またはその逆行列 $H_k$) をどのように更新してヘッセ行列 $\nabla^2 f(x_k)$ に近づけるかにある。\Cref{fig:quasi_newton_overview} はこの概念を示す。まず現在点 $x_k$ の周りで、$B_k$ を用いて目的関数 $f$ の二次近似モデルを構成する。次にこの二次モデルを最小化して次の点 $x_{k+1}$ を得る。$x_{k+1}$ を得た後、$x_k$ と $x_{k+1}$ における勾配情報を用いて近似行列 $B_k$ を $B_{k+1}$ に更新する。この手続きを収束まで繰り返すことが準ニュートン法である。

以下では、近似ヘッセ行列 $B_k$ が満たすべきセカント条件を導入し、$B_k$ と $H_k$ に対する代表的な更新公式を示す。

### セカント条件

この節でも $f\colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。
対称行列 $B_k$ が点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似として与えられているとする。
次の点 $x_{k+1}$ における近似ヘッセ行列 $B_{k+1}$ を定める。
このような行列の候補は無数に存在するが、真のヘッセ行列が対称であることから $B_{k+1}$ にも対称性を課すのが自然である。
ステップと勾配差を次で定義する。

```math
\begin{equation*}
s_k = x_{k+1} - x_k, \qquad   y_k = \nabla f(x_{k+1}) - \nabla f(x_k),
\end{equation*}
```

ここで $y_k^\top s_k \neq 0$ かつ $s_k \neq 0$ を仮定する。なお $s_k$ の $s$ は step を意味する。
$\nabla f(x_k)$ のテイラー展開を用いると次が成り立つ。

```math
\begin{align*}
\nabla f(x_k) & = \nabla f(x_{k+1}) + \nabla^2 f(x_{k+1})(x_k - x_{k+1}) + \order{\lVert x_k - x_{k+1} \rVert^2} \\
& \approx \nabla f(x_{k+1}) + \nabla^2 f(x_{k+1})(x_k - x_{k+1})                                   \\
& \approx \nabla f(x_{k+1}) + B_{k+1}(x_k - x_{k+1})
\end{align*}
```

上の近似を等式として要求すると次を得る。

```math
\begin{equation*}
B_{k+1}(x_{k+1} - x_k) = \nabla f(x_{k+1}) - \nabla f(x_k),
\end{equation*}
```

あるいは同値に

```math
\begin{equation*}
B_{k+1} s_k = y_k.
\end{equation*}
```

この関係はセカント条件、または準ニュートン方程式と呼ばれる。

### 代表的な準ニュートン更新公式

$B_k$, $s_k$, $y_k$ が与えられたとき、セカント条件を満たす $B_{k+1}$ を与える更新公式は多数存在する。ここでは代表的なものをその導出とともに示す \citep{dennisjr.QuasiNewtonMethodsMotivation1977a}。
本小節に限り、簡潔さのため $B_k$, $B_{k+1}$, $s_k$, $y_k$ をそれぞれ $B$, $\bar{B}$, $s$, $y$ と略記する。

#### Broyden の更新

\href{https://en.wikipedia.org/wiki/Broyden%27s_method}{Broydenの更新}は最も基本的な準ニュートン更新公式の一つだが、対称性を保たないため実用上はあまり用いられない。
更新公式は次で与えられる。

```math
\begin{align*}
\bar{B}_{\mathrm{Broyden}} & = B + \frac{(y - Bs)s^\top}{s^\top s},   \\
\bar{H}_{\mathrm{Broyden}} & = H + \frac{s - Hy}{s^\top Hy} s^\top H.
\end{align*}
```

##### 導出

単純な構造的仮定からこの公式を導出する \citep[Section 4]{dennisjr.QuasiNewtonMethodsMotivation1977a}。

**Proposition 5**

$\bar{B}$ がセカント条件

```math
\begin{equation*}
\bar{B}s = y,
\end{equation*}
```

と作用制約

```math
\begin{equation*}
\bar{B}z = Bz
\quad\text{for all } z\in\mathbb{R}^n \text{ such that } z^\top s = 0.
\end{equation*}
```

を満たすと仮定する。このとき $\bar{B}$ は一意に定まり、$\bar{B}_{\mathrm{Broyden}}$ に一致する。

<details>
<summary>Proof</summary>

ベクトル $s$ と $s$ の直交補空間の基底は $\mathbb{R}^n$ の基底を成す。
$\bar{B}$ の条件はこの基底に対する $\bar{B}$ の作用を完全に決定するため、$\bar{B}$ は一意に定まる。
ここで $\bar{B}_{\mathrm{Broyden}}$ が $\bar{B}$ に課された条件を満たすことを示す。
$z^\top s = 0$ を満たす任意のベクトル $z$ を取る。このとき

```math
\begin{align*}
\bar{B}_{\mathrm{Broyden}} s
& = \left(B + \frac{(y - Bs)s^\top}{s^\top s}\right) s
= Bs + (y - Bs)
= y,                                                    \\
\bar{B}_{\mathrm{Broyden}} z
& = \left(B + \frac{(y - Bs)s^\top}{s^\top s}\right) z
= Bz + (z^\top s) \frac{y - Bs}{s^\top s}
= Bz.
\end{align*}
```

したがって $\bar{B}_{\mathrm{Broyden}}$ は $\bar{B}$ に課された条件を満たす。
ゆえに一意性より $\bar{B} = \bar{B}_{\mathrm{Broyden}}$ を得る。
\myQED

</details>

Broyden の更新はフロベニウスノルムにおける最小変化更新としても特徴づけられる。

**Proposition 6** (\citep{dennisjr.QuasiNewtonMethodsMotivation1977a}, Theorem~4.1)

$B\in\mathbb{R}^{n\times n}$, $y\in\mathbb{R}^n$, $s\in\mathbb{R}^n\setminus\lbrace 0 \rbrace$ を与える。このとき行列 $\bar{B}_{\mathrm{Broyden}}$ は

```math
\begin{align*}
\underset{\tilde{B} \in \mathbb{R}^{n \times n}}{\mathrm{minimize}} & \quad \lVert \tilde{B} - B \rVert_F \\
\mathrm{subject to}                                                 & \quad \tilde{B} s = y
\end{align*}
```

の一意解である。

<details>
<summary>Proof</summary>

関数 $\tilde{B}\mapsto\lVert \tilde{B}-B \rVert_F$ は $\mathbb{R}^{n\times n}$ 上で厳密凸である。
制約集合

```math
\begin{equation*}
\lbrace\tilde{B}\in\mathbb{R}^{n\times n}:\tilde{B}s=y\rbrace
\end{equation*}
```

はアフィンであり凸である。
よってこの最適化問題は高々一つの最小解しか持たない。
$\bar{B}_{\mathrm{Broyden}}$ が実際に最小解であることを示す。制約 $\tilde{B}s=y$ を満たす任意の $\tilde{B}$ に対して

```math
\begin{equation*}
\lVert \bar{B}_{\mathrm{Broyden}} - B \rVert_F^2
= \lVert \frac{(y-Bs)s^\top}{s^\top s} \rVert_F^2
= \lVert (\tilde{B}-B) \frac{s s^\top}{s^\top s} \rVert_F^2
\leq \lVert \tilde{B}-B \rVert_F^2,
\end{equation*}
```

最後の不等式ではフロベニウスノルムの劣乗法性と $\lVert ss^\top/(s^\top s) \rVert_F=1$ を用いた。
したがって $\tilde{B}=\bar{B}_{\mathrm{Broyden}}$ である。
\myQED

</details>

Broyden の更新は、セカント条件を満たし、$s$ に直交するベクトルへの作用を保存するという二つの性質で特徴づけられる。さらに、セカント制約の下でフロベニウスノルムの最小変化更新として一意である。

#### SR1 更新

\href{https://en.wikipedia.org/wiki/Symmetric_rank-one}{対称ランク1 (SR1) 更新} \citep{nocedal1999numerical} は更新過程で対称性を維持する基本的な準ニュートン法である。更新公式は次で与えられる。

```math
\begin{align*}
\bar{B}_{\mathrm{SR1}} & = B + \frac{(y - B s)(y - B s)^\top}{(y - B s)^\top s}, \\
\bar{H}_{\mathrm{SR1}} & = H + \frac{(s - H y)(s - H y)^\top}{(s - H y)^\top y}.
\end{align*}
```

##### 導出

SR1 更新を導出するため、更新行列 $\bar{B}$ をランク1更新として構成する。すなわち、あるベクトル $z \in \mathbb{R}^n$ に対して

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}} = B + z z^\top.
\end{equation*}
```

セカント条件 $\bar{B}_{\mathrm{SR1}} s = y$ を満たすために、$z^\top s \neq 0$ のとき次が必要となる。

```math
\begin{equation*}
B s + z z^\top s = y,
\end{equation*}
```

これより

```math
\begin{equation*}
z = \frac{y - B s}{z^\top s}.
\end{equation*}
```

$z^\top s$ を決めるために $s$ との内積を取る。

```math
\begin{equation*}
z^\top s = \frac{(y - B s)^\top s}{z^\top s}.
\end{equation*}
```

この式を整理すると次の関係を得る。

```math
\begin{equation*}
(z^\top s)^2 = (y - B s)^\top s.
\end{equation*}
```

したがって

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}}
= B + z z^\top
= B + \frac{(y - B s)(y - B s)^\top}{(z^\top s)^2}
= B + \frac{(y - B s)(y - B s)^\top}{(y - B s)^\top s},
\end{equation*}
```

が得られ、SR1 更新公式が再現される。

##### 補足

SR1 更新は $(y - Bs)^\top s \neq 0$ であることを前提とする。$(y - Bs)^\top s = 0$ のとき分母が零となり、実用上は更新をスキップすることが多い。この状況は対称ランク1更新ではセカント条件を満たせない場合に起こり得て、より洗練された更新戦略が必要であることを示す。

#### Powell 対称 Broyden (PSB) 更新

Powell 対称 Broyden (PSB) 更新 \cite{haeltermanAnalyticalStudyLeast2009} は最も重要な準ニュートン更新公式の一つである。更新公式は次で与えられる。

```math
\begin{align*}
\bar{B}_{\mathrm{PSB}} & = B + \frac{(y - B s) s^\top + s (y - B s)^\top}{s^\top s} - \frac{s^\top (y - B s)}{(s^\top s)^2} s s^\top, \\
\bar{H}_{\mathrm{PSB}} & = H + \frac{(s - H y) y^\top + y (s - H y)^\top}{y^\top y} - \frac{y^\top (s - H y)}{(y^\top y)^2} y y^\top.
\end{align*}
```

##### 導出

この公式の動機を理解するために、SR1 更新ではランク1更新が $\bar{B}_{\mathrm{SR1}} = B + z z^\top$ のように対称的に定式化されていたことに注意する。
更新過程で常に対称性を維持する要件を緩め、代わりに $z c^\top$ という非対称ランク1更新を考え、最後に対称化する。

$c^\top s \neq 0$ を満たすベクトル $c \in \mathbb{R}^n$ を与え、次で定義する。

```math
\begin{equation*}
z = \frac{y - B s}{c^\top s}
\end{equation*}
```

そして次の非対称更新を行う。

```math
\begin{equation*}
C_1 \mathrel{\vcenter{:}}= B + \frac{(y - B s)c^\top}{c^\top s}.
\end{equation*}
```

$C_1$ は一般に対称でないため、次で対称化する。

```math
\begin{equation*}
C_2 = \frac{C_1 + C_1^\top}{2}.
\end{equation*}
```

しかし、対称化された行列 $C_2$ はセカント条件 $C_2 s = y$ を満たさない場合がある。そこでこの過程を反復する。

```math
\begin{equation*}
\begin{cases}
C_0 = B                                                                               \\
C_{2t+1} = C_{2t} + \frac{(y - C_{2t}s)c^\top}{c^\top s} & \text{(asymmetric update)} \\
C_{2t+2} = \frac{C_{2t+1} + C_{2t+1}^\top}{2}            & \text{(symmetrization)}
\end{cases}
\end{equation*}
```

重要な結果として、列 $\lbrace C_{2t} \rbrace_{t=0}^{\infty}$ はセカント条件を満たす対称行列に収束する。次の命題でそれを定式化する。

**Proposition 7** (\citep{dennisjr.QuasiNewtonMethodsMotivation1977a}, Lemma~7.2)

行列の列 $\lbrace C_{2t} \rbrace_{t=0}^{\infty}$ は収束し、その極限は次で与えられる:

```math
\begin{equation*}
\lim_{t \to \infty} C_{2t}
=
C_{\infty}
\mathrel{\vcenter{:}}=
B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{c^\top s} - \frac{(y - Bs)^\top s}{(c^\top s)^2} c c^\top.
\end{equation*}
```

<details>
<summary>Proof</summary>

まず偶数部分列を解析する。
次で定義する。

```math
\begin{equation*}
G_k \mathrel{\vcenter{:}}= C_{2k}
\end{equation*}
```

ここで $k=0,1,2,\dots$ である。
構成より各 $G_k$ は対称である。
定義から次を得る。

```math
\begin{equation*}
G_{k+1} = G_k +\frac{1}{2c^\top s}\left((y-G_k s)c^\top+c(y-G_k s)^\top\right).
\end{equation*}
```

誤差ベクトルを次で導入する。

```math
\begin{equation*}
w_k \mathrel{\vcenter{:}}= y-G_k s.
\end{equation*}
```

このとき

```math
\begin{equation*}
G_{k+1} = G_k+\frac{1}{2c^\top s}(w_k c^\top+cw_k^\top).
\end{equation*}
```

上式を $w_k$ の定義に代入すると

```math
\begin{align*}
w_{k+1} & = y-\left(G_k+\frac{1}{2c^\top s}(w_k c^\top+cw_k^\top)\right)s \\
& =
w_k-\frac12w_k-\frac{w_k^\top s}{2c^\top s}c                              \\
& =
\frac{1}{2}\left(w_k-\frac{w_k^\top s}{c^\top s}c\right).
\end{align*}
```

よって

```math
\begin{equation*}
w_{k+1}=Pw_k,
\qquad
P \mathrel{\vcenter{:}}= \frac{1}{2}\left(I-\frac{cs^\top}{c^\top s}\right).
\end{equation*}
```

行列 $cs^\top/c^\top s$ はランク1で固有値は $1,0,\dots,0$ である。
よって $P$ は固有値 $0$ を一つ持ち、残りはすべて $1/2$ である。
とくにスペクトル半径は $1/2<1$ となる。
したがってノイマン級数が収束し、次が成り立つ。

```math
\begin{align*}
\sum_{k=0}^{\infty}w_k & =        \sum_{k=0}^{\infty}P^k(y-Bs)                         &  & (w_0=y-Bs)                \\
& =  (I-P)^{-1}(y-Bs)                                                                          \\
& = 2\left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) (y-Bs). &  & (\text{definition of } P)
\end{align*}
```

最後の式は次から従う。

```math
\begin{equation*}
2(I-P) \left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) = \left(I + \frac{cs^\top}{c^\top s}\right) \left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) = I.
\end{equation*}
```

特に $k\to\infty$ で $\lVert w_k \rVert\to0$ となる。
よって

```math
\begin{align*}
\lim_{k\to\infty}G_k
& =
B+\frac{1}{2c^\top s}
\sum_{k=0}^{\infty}(w_k c^\top+c w_k^\top)                                                                                                                                        \\
& = B+ \left(\sum_{k=0}^{\infty}w_k\right) \frac{c^\top}{2c^\top s} + \frac{c}{2c^\top s} \left(\sum_{k=0}^{\infty}w_k\right)^\top                                               \\
& = B+ 2\left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) (y-Bs) \frac{c^\top}{2c^\top s} + \frac{c}{2c^\top s} 2(y-Bs)^\top \left(I-\frac{1}{2}\frac{sc^\top}{c^\top s}\right) \\
& = B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{c^\top s} - \frac{(y - Bs)^\top s}{(c^\top s)^2} c c^\top                                                                         \\
& = C_{\infty}.
\end{align*}
```

次に奇数部分列について

```math
\begin{equation*}
C_{2k+1}
=
G_k+\frac{w_k c^\top}{c^\top s}.
\end{equation*}
```

$G_k\to\bar B$ かつ $\lVert w_k \rVert\to0$ なので

```math
\begin{equation*}
C_{2k+1}-G_k\to0.
\end{equation*}
```

したがって部分列 $\lbrace C_{2k} \rbrace$ と $\lbrace C_{2k+1} \rbrace$ はどちらも $\bar B$ に収束し、よって

```math
\begin{equation*}
C_k\to\bar B.
\end{equation*}
```

以上で証明が完了する。
\myQED

</details>

$c = s$ のとき、$C_{\infty}$ の一般式は標準的な PSB 更新公式に簡約される。

```math
\begin{equation*}
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - Bs)s^\top + s(y - Bs)^\top}{s^\top s} - \frac{(y - Bs)^\top s}{(s^\top s)^2} ss^\top.
\end{equation*}
```

##### 補足

$c = s$ の選択は更新結果の正定値性を確保する動機に基づく。PSB 更新は、非対称ランク1更新に対する反復的対称化過程の極限として解釈できる。

#### DFP 更新

\href{https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula}{Davidon--Fletcher--Powell (DFP) 更新} \cite{nocedal1999numerical} は古典的な準ニュートン更新公式である。更新公式は次で与えられる。

```math
\begin{align*}
\bar{B}_{\mathrm{DFP}} & = (I - \frac{y s^\top}{y^\top s}) B (I - \frac{s y^\top}{y^\top s}) + \frac{y y^\top}{y^\top s}, \\
\bar{H}_{\mathrm{DFP}} & = H - \frac{H y y^\top H}{y^\top H y} + \frac{s s^\top}{y^\top s}.
\end{align*}
```

##### 導出

先に述べた PSB 更新では $c=s$ としたが、別の $c$ を選ぶこともできる。具体的には $B_{k+1}$ が正定値になるように $c$ を選ぶことを考える。$c=y$ を代入すると次の別形式が得られる。

```math
\begin{equation*}
\bar{B}_{\mathrm{DFP}} = B + \frac{(y - Bs)y^\top + y(y - Bs)^\top}{y^\top s} - \frac{(y - Bs)^\top s}{(y^\top s)^2} yy^\top.
\end{equation*}
```

#### BFGS 更新

\href{https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm}{Broyden--Fletcher--Goldfarb--Shanno (BFGS) 更新}は最も広く使われる準ニュートン法の一つである。更新公式は次で与えられる。

```math
\begin{align*}
\bar{B}_{\mathrm{BFGS}} & = B - \frac{B s s^\top B}{s^\top B s} + \frac{y y^\top}{y^\top s},                                                     \\
\bar{H}_{\mathrm{BFGS}} & = \left(I - \frac{s y^\top}{y^\top s}\right) H \left(I - \frac{y s^\top}{y^\top s}\right) + \frac{s s^\top}{y^\top s}.
\end{align*}
```

##### 導出

この更新は DFP 更新の双対を考えることで導出できる。具体的には、セカント条件を満たしつつ逆ヘッセ近似 $H$ の変化を最小化する更新を求める。詳細は次小節で述べる。

### BFGS 法

本小節では BFGS 更新に注目し、その公式の詳細な導出を示す。
BFGS 更新は実用上もっとも成功した準ニュートン更新公式の一つとして知られている。
BFGS 更新公式は次で与えられる。

```math
\begin{equation*}
B_{k+1}   = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}.
\end{equation*}
```

#### 逆更新の公式

BFGS 更新の逆行列 $H_k \mathrel{\vcenter{:}}= B_k^{-1}$ は次式で与えられる。

```math
\begin{equation*}
H_{k+1} = \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) + \frac{s_k s_k^\top}{y_k^\top s_k}.
\end{equation*}
```

この式が確かに BFGS 更新の逆行列を与えることを示す。

**Proposition 8**

行列 $H_{k+1}$ は $B_{k+1}$ の逆行列である。

<details>
<summary>Proof</summary>

BFGS 更新は次の簡潔なランク2の形に書き直せる。

```math
\begin{equation*}
B_{k+1} = B_k + UCV^\top
\end{equation*}
```

ここで

```math
\begin{equation*}
U = \begin{bmatrix}B_k s_k & y_k\end{bmatrix},\qquad
C = \begin{pmatrix}-\frac{1}{s_k^\top B_k s_k} & 0 \\ 0 & \frac{1}{y_k^\top s_k}\end{pmatrix},\qquad
V = \begin{bmatrix}B_k s_k & y_k\end{bmatrix}
\end{equation*}
```

である。実際

```math
\begin{equation*}
U C V^\top   = \begin{bmatrix} -\frac{B_k s_k}{s_k^\top B_k s_k}&    \frac{y_k}{y_k^\top s_k} \end{bmatrix} \begin{bmatrix} s_k^\top B_k  \\  y_k^\top \end{bmatrix}
= -\frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}.
\end{equation*}
```

Sherman--Morrison--Woodbury の恒等式より

```math
\begin{align*}
H_{k+1} & = (B_k + U C V^\top)^{-1}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \\
& = B_k^{-1}- B_k^{-1}U\left(C^{-1}+V^\top B_k^{-1}U\right)^{-1}V^\top B_k^{-1}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  \\
& = H_k- \begin{bmatrix}s_k & H_k y_k\end{bmatrix} \left(\begin{pmatrix}- s_k^\top B_k s_k & 0                                                                                                                                                                                                                                            \\0    & y_k^\top s_k\end{pmatrix}+\begin{pmatrix}s_k^\top B_k s_k   & s_k^\top y_k      \\y_k^\top s_k & y_k^\top H_k y_k\end{pmatrix}\right)^{-1}\begin{bmatrix}s_k^\top \\ y_k^\top H_k\end{bmatrix}                                                                                                                                                \\
& = H_k- \begin{bmatrix}s_k                                                                                                                             & H_k y_k\end{bmatrix}\begin{pmatrix}0                                                               & y_k^\top s_k                                                                                                                                                                                                                                          \\y_k^\top s_k       & y_k^\top H_k y_k + y_k^\top s_k\end{pmatrix}^{-1}\begin{bmatrix}s_k^\top                         \\     y_k^\top H_k\end{bmatrix}                    \\
& = H_k- \begin{bmatrix}s_k                                                                                                                             & H_k y_k\end{bmatrix}\left(-\frac{1}{(y_k^\top s_k)^2}\begin{pmatrix}y_k^\top H_k y_k + y_k^\top s_k & -y_k^\top s_k                                                                                                                                                                                                                                         \\-y_k^\top s_k                 & 0    \end{pmatrix}\right) \begin{bmatrix} s_k^\top                                                      \\     y_k^\top H_k\end{bmatrix} \\
& = H_k+\frac{1}{(y_k^\top s_k)^2}\begin{bmatrix}s_k                                                                                                    & H_k y_k\end{bmatrix} \begin{pmatrix}y_k^\top H_k y_k + y_k^\top s_k                                & -y_k^\top s_k                                                                                                                                                                                                                                         \\-y_k^\top s_k                 & 0\end{pmatrix} \begin{bmatrix}s_k^\top                                                             \\    y_k^\top H_k\end{bmatrix}        \\
& = H_k+\frac{1}{(y_k^\top s_k)^2} \left((y_k^\top H_k y_k + y_k^\top s_k) s_k s_k^\top - (y_k^\top s_k)(s_k y_k^\top H_k + H_k y_k s_k^\top) \right)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \\
& = \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) + \frac{s_k s_k^\top}{y_k^\top s_k}.
\end{align*}
```

を得る。これで証明が完了する。
\myQED

</details>

#### BFGS 更新の正定値性

BFGS 更新の重要な性質として、現在の近似 $B_k$ が正定値で曲率条件 $y_k^\top s_k > 0$ が成り立つなら、更新後の近似 $B_{k+1}$ も正定値であることが保証される。

**Proposition 9**

$B_k$ が正定値で $y_k^\top s_k > 0$ が成り立つなら、$B_{k+1}$ も正定値である。

<details>
<summary>Proof</summary>

仮定より $B_k$ とその逆行列 $H_k$ は正定値である。
任意の非零ベクトル $v \in \mathbb{R}^n$ に対して

```math
\begin{align*}
v^\top H_{k+1} v
& = v^\top \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) v + v^\top \frac{s_k s_k^\top}{y_k^\top s_k} v \\
& \geq 0 + \frac{(s_k^\top v)^2}{y_k^\top s_k} > 0,
\end{align*}
```

ここで第1項は $H_k$ が正定値であるため非負であり、第2項は曲率条件 $y_k^\top s_k > 0$ により正である。
よって $H_{k+1}$ は正定値であり、その逆行列 $B_{k+1} = H_{k+1}^{-1}$ も正定値である。
\myQED

</details>

#### KL

次に、Kullback--Leibler (KL) ダイバージェンスによる BFGS 更新の変分的特徴付けを示す。
平均零の多変量ガウス分布 $\mathcal{N}(0,A^{-1})$ と $\mathcal{N}(0,B^{-1})$ に対して、次の式

```math
\begin{equation*}
\psi(A) = \operatorname{tr}(A) - \log \det(A)
\end{equation*}
```

は加法定数 $-n$ を除いて KL ダイバージェンスに一致するため、$\psi$ の最小化は KL 距離の最小化と同値である。
このとき BFGS 更新は次の最適化問題の解として与えられる。

```math
\begin{align*}
\underset{B \in \mathrm{PD}(n)}{\mathrm{minimize}} & \quad \psi(B_k^{-1/2} B B_k^{-1/2}) \\
\mathrm{subject\ to}                               & \quad B s_k = y_k.
\end{align*}
```

この最適化問題の制約 $B s_k = y_k$ は先に述べたセカント条件そのものであり、更新行列が最新の曲率対を内挿することを保証する。

\textcolor{red}{todo:prove}
これらの定式化の証明は文献を参照されたい \citep{kanamoriBregmanExtensionQuasiNewton2010,kanamoriBregmanExtensionQuasiNewton2010a}。
\href{http://matsuzoe.web.nitech.ac.jp/infogeo/OCAMI2010/kanamori.pdf}{こちらのスライド}も参照されたい。

#### BFGS 更新のトレースと行列式の公式

更新行列の固有値挙動の解析に有用な、BFGS 更新のトレースと行列式の公式も示す。
エルミート行列ではこれらはそれぞれ固有値の総和と積に対応する。したがってトレースと行列式が適切に有界なら、固有値自体も有界に保たれると期待できる(例えばすべての固有値が正である場合)。
これは $\mu$-強凸性や $L$-平滑性など、目的関数のヘッセ行列固有値に関する仮定と密接に関係する。

##### トレースの公式

BFGS 更新後の行列のトレースには明示式がある。

**Proposition 10** ({\citep[(6.44))

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ を BFGS 更新とする。このとき

```math
\begin{equation*}
\tr(B_{+}) = \tr(B) - \frac{\lVert B s \rVert^2}{s^\top Bs} + \frac{\lVert y \rVert^2}{y^\top s}
\end{equation*}
```

が成り立つ。

<details>
<summary>Proof</summary>

BFGS 更新公式にトレースを適用すると

```math
\begin{equation*}
\tr(B_{+}) = \tr(B) - \tr\left(\frac{Bss^\top B}{s^\top Bs}\right) + \tr\left(\frac{yy^\top}{y^\top s}\right).
\end{equation*}
```

第2項について

```math
\begin{equation*}
\tr\left(\frac{Bss^\top B}{s^\top Bs}\right) = \frac{1}{s^\top Bs}\tr((Bs)(Bs)^\top) = \frac{\lVert B s \rVert^2}{s^\top Bs}.
\end{equation*}
```

第3項について

```math
\begin{equation*}
\tr\left(\frac{yy^\top}{y^\top s}\right) = \frac{1}{y^\top s}\tr(yy^\top) = \frac{\lVert y \rVert^2}{y^\top s}.
\end{equation*}
```

以上より所望の式が得られる。
\myQED

</details>

##### 行列式の公式

BFGS 更新後の行列式も閉形式で与えられる。

**Proposition 11** ({\citep[(6.45))

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ を BFGS 更新とし、$B$ が正則であるとする。このとき

```math
\begin{equation*}
\det(B_{+}) = \det(B) \frac{y^\top s}{s^\top Bs}
\end{equation*}
```

が成り立つ。

<details>
<summary>Proof</summary>

BFGS 更新のランク2表現を思い出す。
\href{https://en.wikipedia.org/wiki/Matrix_determinant_lemma}{行列式補題}より、$U, C, V$ は次を満たす。

```math
\begin{equation*}
\det(B_{k+1}) =\det(B_k + U C V^\top)=\det(B_k)\det(C) \det \left(C^{-1} + V^\top B_k^{-1} U\right),
\end{equation*}
```

ここで $I_2$ は $2\times 2$ の単位行列である。
$U=V=\begin{bmatrix}B_k s_k & y_k\end{bmatrix}$ なので

```math
\begin{equation*}
V^\top B_k^{-1} U = \begin{bmatrix} s_k^\top B_k s_k & s_k^\top y_k \\ y_k^\top s_k     & y_k^\top B_k^{-1} y_k \end{bmatrix}
\end{equation*}
```

よって

```math
\begin{equation*}
C^{-1} + V^\top B_k^{-1} U
=
\begin{bmatrix}-s_k^\top B_k s_k & 0 \\ 0 & y_k^\top s_k\end{bmatrix} +
\begin{bmatrix} s_k^\top B_k s_k & s_k^\top y_k \\ y_k^\top s_k     & y_k^\top B_k^{-1} y_k \end{bmatrix}
=
\begin{bmatrix}0 & s_k^\top y_k \\ y_k^\top s_k & y_k^\top B_k^{-1} y_k \end{bmatrix}.
\end{equation*}
```

以上を合わせると

```math
\begin{equation*}
\det(B_{k+1})
= \det(B_k) \left(-\frac{1}{s_k^\top B_k s_k} \cdot \frac{1}{y_k^\top s_k}\right) \left(- (s_k^\top y_k)(y_k^\top s_k)\right)
= \det(B_k) \frac{y_k^\top s_k}{s_k^\top B_k s_k},
\end{equation*}
```

を得る。これで証明が完了する。
\myQED

</details>

### BFGS と DFP の比較

BFGS と DFP は構造的にはかなり対称的であるにもかかわらず、実際の最適化問題に適用すると実用上の効率は大きく異なる。Powell の解析 \citep{powellHowBadAre1986} は、単純な2次元二次関数に対する両手法の挙動を調べてこの非対称性を検討した。漸近収束理論では両者は同程度に振る舞うと示唆されることが多いが、Powell は実用上の効率が大きく異なること、とくに近似ヘッセ行列が真のヘッセ行列から遠い場合に差が顕著であることを示した。

#### 問題設定

Powell の枠組みに従い、次の二次関数を考える。

```math
\begin{equation*}
f(x, y) = \frac{1}{2}(x^2 + y^2),
\end{equation*}
```

そして BFGS と DFP のどちらも各反復で固定ステップサイズ $\alpha_k = 1$ を用いる。二次関数ではこの単位ステップが標準的な線形探索条件を満たすことが多く、実用的な選択である。

初期のヘッセ近似 $B_0$ は固有値 1 と $\lambda_1$ を持つように選ぶ。ここで $\lambda_1$ は初期近似の誤差の程度を表す。
初期点 $x_0$ は次で選ぶ。

```math
\begin{equation*}
\theta = \arctan(\sqrt{\lambda_1}), \quad x_0 = \begin{bmatrix}\cos(\theta) \\ \sin(\theta)\end{bmatrix},
\end{equation*}
```

これは Powell の元の解析に一致する。
この選択の詳細は \citep{powellHowBadAre1986} を参照されたい。

反復は現在点のノルムが初期ノルムに対する許容値を下回るまで続ける。各 $\lambda_1$ に対して収束に要する反復回数を記録する。

#### 数値結果

数値結果を Table 1 に示す。これは Powell の原表の内容を一部再現したものである。収束挙動は初期固有値 $\lambda_1$ に強く依存する。

| $\lambda_1$ | BFGS | DFP |
| :--: | :--: | :--: |
| 0.001 | 4 | 3 |
| 0.01 | 5 | 3 |
| 0.1 | 6 | 4 |
| 1 | 1 | 1 |
| 10 | 8 | 16 |
| 100 | 10 | 107 |
| 1000 | 12 | 1006 |
| 10000 | 15 | 9987 |

(Table 1
\ifEn Convergence comparison between BFGS and DFP methods for different initial eigenvalues $\lambda_1$
\else
初期固有値 $\lambda_1$ に対する BFGS と DFP の収束比較

)

\Cref{fig:bfgs_dfp_100} と Fig. 9 は特定の $\lambda_1$ に対する反復軌跡を示す。これらの図は、同一の初期点から最小点(原点)へ向かう二つの手法の進み方を可視化し、収束速度と経路の違いを明確に示す。

![../imgs/quasi_newton/bfgs_vs_dfp_100.png](../imgs/quasi_newton/bfgs_vs_dfp_100.png)

(Fig. 8 $\lambda_1 = 100$ における BFGS と DFP の反復軌跡。BFGS は 10 回で収束する一方、DFP は 107 回を要し、固有値誤差が大きい場合に BFGS が優位であることを示す。)

![../imgs/quasi_newton/bfgs_vs_dfp_0.1.png](../imgs/quasi_newton/bfgs_vs_dfp_0.1.png)

(Fig. 9 $\lambda_1 = 0.1$ における BFGS と DFP の反復軌跡。どちらも素早く収束し、DFP が BFGS よりわずかに速い。固有値が過小評価された場合の対称的な振る舞いを示している。)

#### 解析と議論

数値結果は BFGS と DFP の間に顕著な非対称性があることを示す。
$\lambda_1 > 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過大評価する場合、BFGS は大幅に高い効率を示す。
逆に $\lambda_1 < 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過小評価する場合、DFP は BFGS よりわずかに良いが、その差は小さい。
この傾向の逆転は理論的な対称性から予測されるが、差の大きさは注目に値する。

##### ヘッセ補正の非対称性

性能の非対称性は、両手法が誤った固有値を補正する仕方の根本的な違いに起因する。
核心的な洞察は、過大な固有値の補正が過小な固有値の補正より重要であるという点である。

ヘッセ固有値が過大評価されると、アルゴリズムは過度に保守的なステップを取り、最小点への進みが遅くなる。この誤差を補正するには、更新公式が大きな固有値を 1 へ縮小する必要がある。
BFGS 更新はこの作業に非常に効果的である。

一方、ヘッセ固有値が過小評価される場合、アルゴリズムはやや攻撃的なステップを取るが、誤差は自己修正的である。
その後の勾配計算が近似の改善に役立つ情報を提供する。
そのため過小評価の補正は本質的に容易で、必要な反復回数も少ない。

DFP 更新は大きな固有値の補正が苦手である。
最悪の場合、1反復で大きな固有値をわずかしか減らせず、固有値の大きさに匹敵する回数の反復が必要になることがある。
このことが $\lambda_1$ が 1 を超えて増加するにつれて DFP の性能が急激に悪化する理由である。

##### 実用上の含意

これらの知見は、実務で BFGS が DFP より広く選好されることに強い経験的根拠を与える。
Powell の単純な二次問題の解析は、計算量の大半が費やされる解から遠い領域での準ニュートン法の挙動に深い洞察を与える。
誤ったヘッセ近似の補正における BFGS の優位性は、ロバストで効率的な無制約最適化のための第一選択としての地位を支えている。

### 限定記憶 BFGS (L-BFGS)

限定記憶準ニュートン法は、古典的な準ニュートン法を大規模最適化問題へ拡張する手法である。
標準的な準ニュートン法では近似ヘッセ行列またはその逆行列を密行列として保存し更新するため、$n$ 変数に対して $\order{n^2}$ のメモリを要する。

BFGS 更新に基づく L-BFGS 法 \citep{liuLimitedMemoryBFGS1989a} は、行列全体を明示的に保存しない。
代わりに最新の $m$ 組のベクトル対 $\lbrace(s_i,y_i)\rbrace$ のみを保持する。
これにより記憶量は $\order{nm}$ に減少し、$m$ が小さな定数(通常 $m\le 10$) のとき大幅な改善となる。

本小節では次の有限列の行列を扱う。

```math
\begin{equation*}
H_0, H_1, \dots, H_m,
\end{equation*}
```

ここで $H_\ell$ は初期行列 $H_0$ に対して $\ell$ 回の BFGS 更新を適用して得られる逆ヘッセ近似を表す。
これは最適化アルゴリズムの反復点とは異なる点に注意する。本小節では BFGS 更新の構造のみに注目する。

#### 逆 BFGS 更新のコンパクト表現

保存された補正対 $\lbrace(s_i,y_i)\rbrace_{i=0}^{m-1}$ を用い、次を定義する。

```math
\begin{equation*}
\rho_i = \frac{1}{y_i^\top s_i}, \qquad
V_i = I - \rho_i y_i s_i^\top.
\end{equation*}
```

逆 BFGS 更新は次のように表される。

```math
\begin{equation*}
H_{i+1} = V_i^\top H_i V_i + \rho_i s_i s_i^\top,
\end{equation*}
```

ここで $i = 0,\dots,m-1$ である。
この関係を再帰的に展開すると次のコンパクト表現が得られる。

```math
\begin{equation*}
H_m
=
V_{m-1}^\top \cdots V_0^\top H_0 V_0 \cdots V_{m-1}
+
\sum_{j=0}^{m-1}
(V_{m-1}^\top \cdots V_{j+1}^\top)
\rho_j s_j s_j^\top
(V_{j+1} \cdots V_{m-1}),
\end{equation*}
```

ここで $H_0$ は選ばれた初期逆ヘッセ近似であり、通常はスケールされた単位行列である。

#### 二重ループ再帰

上のコンパクト表現は、$H_m$ を明示的に形成せずに任意のベクトル $q$ に適用できる。
$r = H_m q$ とおく。
行列積の結合性を利用すると、この計算は長さ $m$ の短いループを二回回すだけで実行でき、よく知られた L-BFGS の二重ループ再帰につながる \citep[Algorithm 7.4]{nocedal1999numerical}。
このアルゴリズムは $\order{md}$ の演算量と $\order{md}$ の記憶量を要する。ここで $d$ は問題次元である。

\subfile{999_two_loop_recursion.tex}

次に、この二重ループ再帰の出力が確かに $r = H_m q$ を計算していることを確認する。

**Proposition 12**

二重ループ再帰アルゴリズムの出力は $r = H_m q$ を満たす。

<details>
<summary>Proof</summary>

逆方向の再帰では、$i = m-1, m-2, \dots, 0$ に対して入力ベクトル $q^{(m)} \mathrel{\vcenter{:}}= q$ から次を計算する。

```math
\begin{equation*}
\alpha_i   = \rho_i s_i^\top q^{(i+1)}, \qquad
q^{(i)} \mathrel{\vcenter{:}}= q^{(i+1)} - \alpha_i y_i.
\end{equation*}
```

$\alpha_i$ の定義を代入し $V_i$ の定義を用いると

```math
\begin{equation*}
q^{(i)} = q^{(i+1)} - \rho_i \left(s_i^\top q^{(i+1)}\right) y_i = \left(I - \rho_i y_i s_i^\top\right) q^{(i+1)} = V_i q^{(i+1)}.
\end{equation*}
```

よってすべての $i = 0, 1, \dots, m-1$ に対して

```math
\begin{equation*}
q^{(i)} = V_i V_{i+1} \cdots V_{m-1} q.
\end{equation*}
```

次にアルゴリズムは初期逆ヘッセ近似を適用する。

```math
\begin{equation*}
r^{(0)} = H_0 q^{(0)} = H_0 V_0 V_1 \cdots V_{m-1} q.
\end{equation*}
```

次に $i = 0, 1, \dots, m-1$ に対して前方向の再帰は次を計算する。

```math
\begin{equation*}
\beta_i     = \rho_i y_i^\top r^{(i)}, \qquad
r^{(i+1)}   = r^{(i)} + s_i \left(\alpha_i - \beta_i\right).
\end{equation*}
```

$\alpha_i$, $\beta_i$, $q^{(i+1)}$ の定義を代入すると

```math
\begin{align*}
r^{(i+1)}
& =
r^{(i)} + \rho_i s_i s_i^\top \left(V_{i+1} V_{i+2} \cdots V_{m-1}\right) q - \rho_i s_i y_i^\top r^{(i)}            \\
& = \left(I- \rho_i y_i s_i^\top\right) r^{(i)} + \rho_i s_i s_i^\top \left(V_{i+1} V_{i+2} \cdots V_{m-1}\right) q \\
& =
V_i^\top r^{(i)}
+
\rho_i s_i s_i^\top
\left(V_{i+1} V_{i+2} \cdots V_{m-1}\right) q.
\end{align*}
```

初期値 $r^{(0)} = H_0 q^{(0)}$ からこの関係を再帰的に展開すると

```math
\begin{equation*}
r^{(m)}
=
V_{m-1}^\top \cdots V_0^\top H_0 V_0 \cdots V_{m-1} q
+
\sum_{j=0}^{m-1}
(V_{m-1}^\top \cdots V_{j+1}^\top)
\rho_j s_j s_j^\top
(V_{j+1} \cdots V_{m-1}) q,
\end{equation*}
```

となり、$H_m$ のコンパクト表現を $q$ に適用した式と一致する。これで証明が完了する。
\myQED

</details>

したがって二重ループ再帰は、行列を明示的に構成せずに $H_m$ の作用を正確に評価し、数学的厳密さと計算効率の両方を実現する。

#### 初期スケーリング

L-BFGS 法の重要な要素は初期行列 $H_0$ の選択である。
広く使われ、十分に正当化された選択はスケールされた単位行列である。

```math
\begin{equation*}
H_0 = \gamma I,
\end{equation*}
```

ここでスケーリング係数は次で選ぶ。

```math
\begin{equation*}
\gamma = \frac{s_{m-1}^\top y_{m-1}}{y_{m-1}^\top y_{m-1}}.
\end{equation*}
```

この選択は近似逆ヘッセ行列と目的関数の局所曲率の関係に基づく \citep{liuLimitedMemoryBFGS1989a,shannoMatrixConditioningNonlinear1978}。
このスケーリングを正当化するため、目的関数 $f$ が二回連続微分可能であると仮定し、最新のステップに沿った平均ヘッセ行列を考える。

```math
\begin{equation*}
\bar{G} = \int_0^1 \nabla^2 f(x + \tau s_{m-1}) \mathrm{d}\tau,
\end{equation*}
```

ここで $s_{m-1}$ は最新の変位を表す。
平均値の定理より

```math
\begin{equation*}
y_{m-1}
=
\nabla f(x+s_{m-1}) - \nabla f(x)
=
\bar{G} s_{m-1}.
\end{equation*}
```

この関係を用いるとスケーリング係数は次のように書ける。

```math
\begin{equation*}
\frac{s_{m-1}^\top y_{m-1}}{y_{m-1}^\top y_{m-1}}
=
\frac{(\bar{G}^{1/2} s_{m-1})^\top (\bar{G}^{1/2} s_{m-1})}
{(\bar{G}^{1/2} s_{m-1})^\top \bar{G} (\bar{G}^{1/2} s_{m-1})},
\end{equation*}
```

これは Rayleigh 商である。
もし $\bar{G}^{1/2} s_{m-1}$ が $\bar{G}$ の固有ベクトルであれば、この量の逆数は対応する固有値に等しい。

さらに $\gamma$ の選択は Barzilai--Borwein 法の短ステップサイズ \citep{barzilaiTwoPointStepSize1988} と一致し、L-BFGS の初期化と古典的なステップ長選択戦略の密接な関係を示している。
この観察は、実用におけるスケール単位行列初期化の有効性をさらに支持する。


<!-- From 4_modified_secant.tex -->

## \ifEn Modified Secant Condition \else 修正セカント条件 \fi

このセクションでは、準ニュートン法で使用される標準的なセカント条件の修正版である修正セカント条件について説明します。
修正セカント条件は勾配情報に加えて関数値情報を取り込み、より正確なヘッセ行列の近似を実現します。

### \ifEn Standard Secant Condition \else 標準セカント条件 \fi

まず、準ニュートン法で使用される標準的なセカント条件を復習します。
$x_k$ と $x_{k+1}$ を最適化アルゴリズムにより生成された連続する2つの反復とします。
次の反復を計算するには、近似ヘッセ行列 $B_k$ を $B_{k+1}$ に更新する必要があります。
ステップと勾配差を以下のように定義します。

```math
\begin{equation*}
s_k = x_{k+1} - x_k, \qquad
y_k = \nabla f(x_{k+1}) - \nabla f(x_k).
\end{equation*}
```

The standard approach is to update $B_k$ to satisfy the secant condition:

```math
\begin{equation*}
B_{k+1} s_k = y_k.
\end{equation*}
```

To justify this condition, consider a quadratic approximation model around $x_{k+1}$:

```math
\begin{equation*}
m_{k+1}(x) = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1} (x - x_{k+1}).
\end{equation*}
```

By construction, this model satisfies

```math
\begin{equation*}
\begin{cases}
m_{k+1}(x_{k+1}) = f(x_{k+1}) \\
\nabla m_{k+1}(x_{k+1}) = \nabla f(x_{k+1})
\end{cases}
\end{equation*}
```

regardless of the choice of $B_{k+1}$.
Additionally, the secant condition ensures that the model satisfies

```math
\begin{align*}
\nabla m_{k+1}(x_k) & = \nabla f(x_{k+1}) - B_{k+1} (x_{k+1} - x_k)             &  & (\text{definition of model}) \\
& = \nabla f(x_{k+1}) - (\nabla f(x_{k+1}) - \nabla f(x_k)) &  & \text{(secant condition)}    \\
& = \nabla f(x_k)
\end{align*}
```

これは前の反復からの勾配情報と一致します。
したがって、セカント条件を使用することにより、二次モデル $m_{k+1}(x)$ が前の関数値 $f(x_k)$ を除いて、$x_{k+1}$ と $x_k$ の両方での勾配情報を正確に反映することが保証されます。

### \ifEn Modified Secant Condition \else 修正セカント条件 \fi

セカント条件は広く使用されていますが、勾配情報のみを照合するため、目的関数の曲率を正確に捕捉できない場合があります。
この制限を Fig. 10 で説明します。

In Fig. 10, we have two points $x_k$ and $x_{k+1}$ with their corresponding function values and gradients.
In Fig. 10, we see that the ideal quadratic model constructed from the exact Hessian fits well around the new point $x_{k+1}$, leading to a better convergence behavior.
However, in Fig. 10, the standard secant update only matches the gradients at these points, neglecting the function value at $x_k$. It can severely misestimate the curvature, leading to a poor approximation of the true objective function.

![../imgs/modified_secant/trial_EXPLAIN.png](../imgs/modified_secant/trial_EXPLAIN.png)

(Fig. 10 修正セカント方程式の動機。関数値と勾配を組み合わせることにより、理想的なニュートン法モデル (b) に近似することを目指しており、標準的なセカント更新 (c) は $f(x_k)$ を省略して曲率を誤推定する可能性があります。)

関数値を利用することでこの制限を克服できます。
基本的な考え方は Fig. 11 に示されています。
2 点 $x_k$ と $x_{k+1}$ で勾配が同じであっても、関数値の自然な内挿は関数値 $f(x_k)$ と $f(x_{k+1})$ に応じて異なります。
この観察は、関数値情報を取り込んでヘッセ行列の近似を改善する修正セカント条件の動機となります。

![../imgs/modified_secant/cubic_interpolation.png](../imgs/modified_secant/cubic_interpolation.png)

(Fig. 11 $x_k$ と $x_{k+1}$ で同一の勾配を持つが異なる関数値による内挿。これは異なる内挿関数を生じさせ、ヘッセ行列の近似に関数値情報を組み込むことの重要性を強調しています。)

以下では、2つの既知の修正セカント条件を提示します。

#### 関数値ベースの修正セカント条件

最初の修正は、前の点での関数値を二次モデルに組み込みます \citep{yuanModifiedBFGSAlgorithm1991,weiNewQuasiNewtonMethods2006, babaie-kafakiModifiedBFGSAlgorithm2011}。
異なる近似ヘッセ行列 $B^{\mathrm{F}}_{k+1}$ を持つ別のモデルを考えましょう:

```math
\begin{equation*}
m_{k+1}^{\mathrm{F}}(x) = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B^{\mathrm{F}}_{k+1} (x - x_{k+1}).
\end{equation*}
```

このモデルが以下を満たすことを要求します:

```math
\begin{equation*}
m^{\mathrm{F}}_{k+1}(x_k) = f(x_k),
\end{equation*}
```

これは前の点での関数値が正しくモデル化されることを保証します。
これが関数値ベースの修正セカント条件の背後にある主要な考え方です。

対応する修正セカント条件を $B^{\mathrm{F}}_{k+1}$ に対して導出しましょう。
$m^{\mathrm{F}}_{k+1}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、条件は以下になります

```math
\begin{align*}
f(x_k) & = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x_k - x_{k+1}) + \frac{1}{2} (x_k - x_{k+1})^\top B^{\mathrm{F}}_{k+1} (x_k - x_{k+1}) \\
& = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top B^{\mathrm{F}}_{k+1} s_k.
\end{align*}
```

次に、更新された行列が以下の形を持つと仮定します

```math
\begin{equation*}
B^{\mathrm{F}}_{k+1} s_k = y_k - \sigma^\mathrm{F}_k s_k,
\end{equation*}
```

ここで、$\sigma^\mathrm{F}_k \in \mathbb{R}$ は決定する必要があるスカラーです。

Substituting this equation and using $s_k^\top B^{\mathrm{F}}_{k+1} s_k = s_k^\top y_k - \sigma^\mathrm{F}_k \lVert s_k \rVert^2$ gives

```math
\begin{equation*}
f(x_k) = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top y_k - \frac{\sigma^\mathrm{F}_k}{2} \lVert s_k \rVert^2.
\end{equation*}
```

$\sigma^\mathrm{F}_k$ について解き、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、以下を得ます

```math
\begin{align*}
\sigma^\mathrm{F}_k & = \frac{2(f(x_{k+1}) - f(x_k)) - (2\nabla f(x_{k+1}) - y_k)^\top s_k}{\lVert s_k \rVert^2}           \\
& = \frac{2(f(x_{k+1}) - f(x_k)) - (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\lVert s_k \rVert^2}.
\end{align*}
```

したがって、修正セカント条件は以下になります

```math
\begin{equation*}
B^{\mathrm{F}}_{k+1} s_k = y_k + \frac{2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

$s_k^\top y_k > 0$ の場合に BFGS 型更新の下で正定性を保つ可能性を保存するために、分子でゼロとの最大値を取ることでこの公式を修正できます。

```math
\begin{equation*}
B^{\mathrm{F}'}_{k+1} s_k = y_k + \frac{\max(0, 2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k)}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

これが関数値ベースの修正セカント条件です。
標準的なセカント条件とは異なり、この定式化は二次モデルが前の点での関数値だけでなく、両方の点での勾配と一致することを保証します。
説明については Fig. 12 を参照してください。

![../imgs/modified_secant/trial_2.png](../imgs/modified_secant/trial_2.png)

(Fig. 12 \ifEn Function-value-based modified secant equation. The modified secant equation $B^\mathrm{F}_k s_k = y_k + \sigma^\mathrm{F}_k s_k$ constructs a quadratic model that satisfies the function value condition $m^{\mathrm{F}}_k(x_{k-1}) = f(x_{k-1})$ at the previous point. This differs from the standard secant equation, which only matches gradients but not function values. \else 関数値ベースの修正セカント方程式。修正セカント方程式 $B^\mathrm{F}_k s_k = y_k + \sigma^\mathrm{F}_k s_k$ は、前の点での関数値条件 $m^{\mathrm{F}}_k(x_{k-1}) = f(x_{k-1})$ を満たす二次モデルを構築します。これは勾配のみを照合し、関数値を照合しない標準的なセカント方程式とは異なります。 \fi)

#### \ifEn Cubic-Augmented Modified Secant Condition \else 3次拡張修正セカント条件 \fi

2 番目の修正はモデルに 3 次項を導入し、前の点での関数値と勾配の両方の一致を同時に満たすことを可能にします \citep{zhangNewQuasiNewtonEquation1999, zhangPropertiesNumericalPerformance2001,yabeLocalSuperlinearConvergence2007}。

$T_{k+1} \in \mathbb{R}^{n \times n \times n}$ を $x_{k+1}$ での $f$ の3階微分テンソルとし、以下を満たすものとします

```math
\begin{equation*}
s_k^\top (T_{k+1} s_k) s_k = \sum_{i,j,l=1}^n \partial_{x_i x_j x_l} f(x_{k+1}) s_k^{(i)} s_k^{(j)} s_k^{(l)},
\end{equation*}
```

ここで、$\partial_{x_i x_j x_l} f$ は $f$ の $x_i$、$x_j$、および $x_l$ に対する3階微分を表し、$s_k^{(i)}$ はベクトル $s_k$ の第 $i$ 成分です。
このテンソルは分析目的でのみ導入され、最終公式から除去されます。

このテンソル項を組み込むことにより、以下の3次拡張モデルを定義できます:

```math
\begin{align*}
m^\mathrm{C}_{k+1}(x) ={} & f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1}^{\mathrm{C}} (x - x_{k+1}) \\
& + \frac{1}{6}(x - x_{k+1})^\top (T_{k+1} (x - x_{k+1})) (x - x_{k+1}).
\end{align*}
```

このモデルを使用して、前の点 $x_k$ での関数値と勾配の両方の一致を強制できます。
具体的には、以下を要求します:

```math
\begin{equation*}
\begin{cases}
m^\mathrm{C}_{k+1}(x_k) = f(x_k) \\
\nabla m^\mathrm{C}_{k+1}(x_k) = \nabla f(x_k)
\end{cases}
\end{equation*}
```

$m^\mathrm{C}_{k+1}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、これらの条件を以下のように書き直せます

```math
\begin{align*}
f(x_k)        & = f(x_{k+1}) - s_k^\top \nabla f(x_{k+1}) + \frac{1}{2} s_k^\top B^\mathrm{C}_{k+1} s_k - \frac{1}{6} s_k^\top (T_{k+1} s_k) s_k, \\
\nabla f(x_k) & = \nabla f(x_{k+1}) - B^\mathrm{C}_{k+1} s_k + \frac{1}{2} (T_{k+1} s_k) s_k.
\end{align*}
```

第1の方程式の両辺に3を乗じて整理し、第2の方程式については $s_k$ との内積をとることで、以下が得られます

```math
\begin{align*}
3(f(x_k) - f(x_{k+1})) +3 s_k^\top \nabla f(x_{k+1}) & = \frac{3}{2} s_k^\top B^{\mathrm{C}}_{k+1} s_k - \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k, \\
-s_k^\top y_k                                        & = -s_k^\top B^{\mathrm{C}}_{k+1} s_k + \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k.
\end{align*}
```

合計し、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、テンソル項を除去し、以下のスカラー恒等式を得ます

```math
\begin{equation*}
3(f(x_k) - f(x_{k+1}))
+ \frac{3}{2} s_k^\top (\nabla f(x_{k+1}) + \nabla f(x_k)) + \frac{1}{2} s_k^\top y_k
= \frac{1}{2} s_k^\top B^{\mathrm{C}}_{k+1} s_k.
\end{equation*}
```

$B^{\mathrm{C}}_{k+1} s_k$ が $y_k$ と $s_k$ の線形結合であると仮定します。つまり、スカラー $\sigma^\mathrm{C}_k$ に対して $B^{\mathrm{C}}_{k+1} s_k = y_k + \sigma^\mathrm{C}_k s_k$ です。
そうすると、前の方程式は以下を与えます

```math
\begin{equation*}
\sigma^\mathrm{C}_k = \frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\lVert s_k \rVert^2}.
\end{equation*}
```

したがって、修正セカント条件は以下になります

```math
\begin{equation*}
B^{\mathrm{C}}_{k+1} s_k = y_k + \frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

これが3次拡張修正セカント条件です。関数値ベースの二次修正とは異なり、この定式化はテンソル項の導入を通じて前の点での関数値条件と勾配条件の両方を同時に満たすことを可能にします。
説明については Fig. 13 を参照してください。

![../imgs/modified_secant/trial_1_cubic.png](../imgs/modified_secant/trial_1_cubic.png)

(Fig. 13 3次拡張修正セカント方程式。Fig. 13 では、3次項 $\eta \lVert x - x_k \rVert^3$ をモデルに組み込むことにより、修正セカント方程式 $B^\mathrm{C}_k s_k = y_k + \sigma^\mathrm{C}_k s_k$ は前の点での両方の条件を同時に満たすことを可能にします。Fig. 13 では、3次モデルは関数値と勾配の条件の両方を満たしていますが、その基礎となる二次成分は不定値または負定値である可能性があります。)

### その他の曲率保存方法

曲率情報を保存するために、いくつかのトピックがあります。
Agg-BFGS \citep{berahasLimitedmemoryBFGSDisplacement2022} は、最も古い情報を破棄して最新のものを追加するのではなく、データを集約することにより曲率情報を管理する別のアプローチです。

Multi-Secant \citep{leeAdvancingMultiSecantQuasiNewton2025} は、複数のステップと勾配差ベクトルのペアを維持することにより、セカント条件フレームワークを拡張します。標準的な定式化では、以下を定義します

```math
\begin{equation*}
s_i = x_{i+1} - x_i, \quad y_i = \nabla f(x_{i+1}) - \nabla f(x_i). \quad (i = k-m, \ldots, k)
\end{equation*}
```

別の固定点型定式化は、すべてのベクトルを最新の反復で中心化します:

```math
\begin{equation*}
s_i = x_{k+1} - x_i, \quad y_i = \nabla f(x_{k+1}) - \nabla f(x_i). \quad (i = k-m, \ldots, k)
\end{equation*}
```

この固定点型アプローチは、改善されたヘッセ行列の近似のための履歴情報の利用に関する異なる観点を提供します。

