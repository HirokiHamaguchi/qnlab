
## 連続最適化の基本概念

数理最適化の中心的な意義の一つは、与えられた定量的な指標を最適化する決定変数を求めることにあります。
そのような指標の例としては、制御の安定性・運用の効率性・予測誤差など、さまざまな現象やシステムの品質を定量化したものが挙げられます。
これらの指標は通常、決定変数から実数への写像である目的関数としてモデル化されます。

目的関数を $C^2$ 級の関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ とし、また $n$ を決定変数の次元と定めます。 この時、次の無制約最適化問題は、最も基本的な最適化問題の一つです。

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x).
\end{equation*}
```

本節では、この最適化問題に関する基本的な定義と性質をまとめる。

### 凸性と強凸性

凸性(convexity)と強凸性(strong convexity)は最適化理論における基本概念です。
簡単のため、$f$ は $\mathbb{R}^n$ 全域で実数値を取る関数である、つまり、$f \colon \mathbb{R}^n \to \mathbb{R}$ とします。また、$\mu>0$ を定数とします。
この時、関数 $f$ が凸、または $\mu$-強凸であることは、任意の $x,y \in \mathbb{R}^n$ と $\lambda \in [0,1]$ について次が成り立つこととそれぞれ同値です。

```math
\begin{align*}
\textrm{(convex)} \quad
f((1-\lambda) x + \lambda y)                                        & \le (1-\lambda) f(x) + \lambda f(y),                                                         \\
\textrm{($\mu$-strongly convex)} \quad f((1-\lambda) x + \lambda y) & \le (1-\lambda) f(x) + \lambda f(y) - \frac{\mu}{2} \lambda (1-\lambda) \lVert x-y \rVert^2.
\end{align*}
```

強凸性は、凸性に加えて目的関数が一様に正の曲率を持つことを意味します。
凸関数と強凸関数の例を Fig. 1 に示しました。
このような定義は、文献によって多少の揺れがありますが、例えば\citep{nesterovIntroductoryLecturesConvex2014,kanamori2016continuous}などを参照してください。

<img src="../imgs/quasi_newton/convexity_comparison_convex.png" /><img src="../imgs/quasi_newton/convexity_comparison_strongly_convex.png" />

(Fig. 1 凸関数と強凸関数の例。 破線は $x=0$ における二次近似を示しています。 上2つの関数は凸ですが強凸ではなく、 下2つの関数は強凸性の定義を満たす $\mu>0$ が存在し、強凸となります。)

関数 $f$ の勾配を用いた凸性および強凸性の同値な定義も存在します。
$f$ が少なくとも $C^1$ 級で、その定義域 $\mathbb{R}^n$ 全体で実数値を取ると仮定します。
関数 $f$ が凸、または $\mu$-強凸であることは、任意の $x,y \in \mathbb{R}^n$ について次が成り立つこととそれぞれ同値です。

```math
\begin{align*}
\textrm{(convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x),                                  \\
\textrm{($\mu$-strongly convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x)+\frac{\mu}{2}\lVert y-x \rVert^2.
\end{align*}
```

**Proposition 1**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であるとする。
このとき、凸と $\mu$-強凸に関する二つの定義はそれぞれ同値である。

<details>
<summary>Proof</summary>

$\mu$-強凸の場合について示します。
まず、前者の定義が成立することを仮定すると、勾配の定義と、$f$ が $C^1$ 級であることから、以下が成り立ちます。

```math
\begin{align*}
\nabla f(x)^\top (y-x)
& =\lim_{\lambda \to 0} \frac{f(x+ \lambda (y-x)) - f(x)}{\lambda}                                                                                         \\
& \le \lim_{\lambda \to 0} \frac{1}{\lambda} \left( (1-\lambda) f(x) + \lambda f(y) - \frac{\mu}{2} \lambda (1-\lambda) \lVert x-y \rVert^2 - f(x) \right) \\
& = f(y) - f(x) - \frac{\mu}{2} \lVert x-y \rVert^2.
\end{align*}
```

従って、後者の定義が成立します。
続いて、後者の定義が成立することを仮定します。
$z \mathrel{\vcenter{:}}= (1-\lambda)x + \lambda y$ とおくと、次が成り立ちます。

```math
\begin{align*}
f(x) \geq f(z) + \nabla f(z)^\top (x-z) + \frac{\mu}{2} \lVert x-z \rVert^2, \\
f(y) \geq f(z) + \nabla f(z)^\top (y-z) + \frac{\mu}{2} \lVert y-z \rVert^2.
\end{align*}
```

これらを、それぞれ $1-\lambda$ 倍、$\lambda$ 倍して足し合わせると、次が成り立ちます。

```math
\begin{align*}
(1-\lambda) f(x) + \lambda f(y)
& \geq f(z) + \nabla f(z)^\top \left( (1-\lambda)(x-z) + \lambda (y-z) \right)
+ \frac{\mu}{2} \left( (1-\lambda) \lVert x-z \rVert^2 + \lambda \lVert y-z \rVert^2 \right)                                   \\
& = f(z) + \frac{\mu}{2} \left( (1-\lambda) \lambda^2 \lVert x-y \rVert^2 + \lambda (1-\lambda)^2 \lVert y-x \rVert^2 \right) \\
& = f(z) + \frac{\mu}{2} \lambda (1-\lambda) \lVert x-y \rVert^2.
\end{align*}
```

従って、前者の定義が成立します。
つまり、これらの定義は同値であることが示されました。
$\mu=0$ の場合も、同様の議論により凸性について示すことができます。
\myQED

</details>

### Positive Definiteness of the Hessian

続いて、$f$ が $C^2$ 級であるとして、凸性および強凸性がヘッセ行列 $\nabla^2 f(x)$ の正定値性とどのように関係するかを示します。

行列の定値性について、先に定義しておきます。
$A$ を $\mathbb{R}^{n \times n}$ の対称行列とします。
行列 $A$ が(半)正定値 (positive (semi-)definite)・(半)負定値 (negative (semi-)definite) であるとは、次の条件で定義されます。

```math
\begin{align*}
\textrm{(positive definite)} \quad      & v^\top A v > 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\textrm{(positive semi-definite)} \quad & v^\top A v \ge 0 \quad \forall v \in \mathbb{R}^n,                 \\
\textrm{(negative definite)} \quad      & v^\top A v < 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\textrm{(negative semi-definite)} \quad & v^\top A v \le 0 \quad \forall v \in \mathbb{R}^n.
\end{align*}
```

正定値でも負定値でもない行列は不定値(indefinite)と呼ばれます。
行列 $A,B \in \mathbb{R}^{n \times n}$ に対して、$A \succeq B$ は $A-B$ が半正定値であることを表します。
特に $B$ が零行列のときは $A \succeq 0$ と書きます。
同様に、$\preceq$ は半負定値に対して定義されます。
$\mu \geq 0$ に対し、$A \succeq \mu I$ はすべての $v \in \mathbb{R}^n$ について $v^\top A v \ge \mu \lVert v \rVert^2$ と同値です。
これは $A$ の任意の固有値が少なくとも $\mu$ 以上であることを意味し、さらに作用素ノルムについても $\lVert A \rVert \geq \mu$ であることを導きます。

ここで、凸性および強凸性とヘッセ行列の正定値性との関係は次の通りです。

**Proposition 2**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^2$ 級であるとする。
このとき、
- $f$ が凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq0$ が成り立つことは同値である。
- $f$ が $\mu$-強凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq\mu I$ が成り立つことは同値である。

<details>
<summary>Proof</summary>

まず $\mu>0$ とし、任意の $x \in \mathbb{R}^n$ に対して $\nabla^2 f(x)\succeq \mu I$ が成り立つと仮定します。
微分積分学の基本定理より、任意の $x,y \in \mathbb{R}^n$ について次が成り立ちます。

```math
\begin{equation*}
f(y)
= f(x)+\nabla f(x)^\top (y-x)
+\frac{1}{2} \int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t.
\end{equation*}
```

また、仮定 $\nabla^2 f(x)\succeq \mu I$ から次が得られます。

```math
\begin{equation*}
\int_0^1 (y-x)^\top \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t
\ge \int_0^1 \mu\lVert y-x \rVert^2 \mathrm{d}t
= \mu\lVert y-x \rVert^2.
\end{equation*}
```

以上の結果を合わせると、$\mu$-強凸性の定義が導かれます。
逆に、$f$ が $\mu$-強凸であると仮定します。
任意の $x \in \mathbb{R}^n, \ v \in \mathbb{R}^n$ および $t>0$ に対し、$y=x \pm tv$ とおくと次が成り立ちます。

```math
\begin{equation*}
\begin{cases}
f(x + tv)\ge f(x) + t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2, \\
f(x - tv)\ge f(x) - t\nabla f(x)^\top v+\frac{\mu}{2}t^2\lVert v \rVert^2.
\end{cases}
\end{equation*}
```

テイラーの定理より、ある $s_\pm \in (0,1)$ が存在して次が成り立ちます。

```math
\begin{equation*}
\begin{cases}
f(x + tv) = f(x) + t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x + s_+ t v) v, \\
f(x - tv) = f(x) - t\nabla f(x)^\top v + \frac{1}{2} t^2 v^\top \nabla^2 f(x - s_- t v) v.
\end{cases}
\end{equation*}
```

これらの結果を合わせると次が成り立ちます。

```math
\begin{equation*}
v^\top \frac{\nabla^2 f(x+ s_+ t v) + \nabla^2 f(x - s_- t v)}{2} v \ge \mu \lVert v \rVert^2.
\end{equation*}
```

$t \to 0$ とし、$f$ が $C^2$ 級という仮定による $\nabla^2 f$ の連続性を用いると次を得ます。

```math
\begin{equation*}
v^\top \nabla^2 f(x) v \ge \mu \lVert v \rVert^2.
\end{equation*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\succeq \mu I$ という結果が得られました。
上記の議論で $\mu=0$ とすれば、同様に凸の場合も示されます。
\myQED

</details>

ヘッセ行列が正定値・不定値・負定値である二次関数を Fig. 2 に示しました。
定値性と凸性の対応関係を視覚的に確認できます。

![../imgs/quasi_newton/pd.png](../imgs/quasi_newton/pd.png)

(Fig. 2 二次モデル $f(x)=\frac{1}{2}(x - x_k)^\top H (x - x_k) + \nabla f(x_k)^\top (x - x_k) + f(x_k)$ を二次元空間で示したもの。 ヘッセ行列 $H$ が (左)正定値、(中央)不定値、(右)負定値の場合を示す。)

### $L$-平滑性 ($L$-smoothness)

最後に、関数の $L$-平滑性 ($L$-smoothness)を導入します。
関数 $f$ が $L$-平滑であるとは、ある定数 $L>0$ が存在して

```math
\begin{equation*}
\lVert \nabla f(x)-\nabla f(y) \rVert \le L\lVert x-y \rVert
\end{equation*}
```

が任意の $x,y$ について成り立つことと同値です。

次の命題は、$L$-平滑性がヘッセ行列の上界で特徴づけられることを示します。

**Proposition 3**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^2$ 級であるとする。
このとき、$f$ が $L$-平滑であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ が成り立つことは同値である。

<details>
<summary>Proof</summary>

$f$ は $C^2$ 級なので、微分積分学の基本定理より任意の $x,y \in \mathbb{R}^n$ について次が成り立つ。

```math
\begin{equation*}
\nabla f(y) - \nabla f(x)
= \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t.
\end{equation*}
```

任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ と仮定します。
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

つまり、 $f$ の $L$-平滑性が示されました。
逆に、$f$ が $L$-平滑であるとすると、任意の $x \in \mathbb{R}^n$ と $v \in \mathbb{R}^n$ に対して次が成り立ちます。

```math
\begin{equation*}
\lVert \nabla f(x+tv)-\nabla f(x) \rVert \le L\lVert tv \rVert = Lt\lVert v \rVert.
\end{equation*}
```

さらにテイラーの定理より、$t \to 0$ のとき $\lVert r(t) \rVert/t \to 0$ を満たす剰余項 $r(t)$ を用いて次が成り立ちます。

```math
\begin{equation*}
\nabla f(x+tv)-\nabla f(x) = t \nabla^2 f(x) v + r(t).
\end{equation*}
```

そして、これは次のように書き換えられます。

```math
\begin{equation*}
\nabla^2 f(x) v = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x) -r(t)}{t} = \lim_{t \to 0} \frac{\nabla f(x+tv)-\nabla f(x)}{t}.
\end{equation*}
```

$v$ との内積を取ると次が得られます。

```math
\begin{align*}
v^\top \nabla^2 f(x) v & = \lim_{t \to 0} \left(\frac{\nabla f(x+tv)-\nabla f(x)}{t}\right)^\top v                                                         \\
& \leq \lim_{t \to 0} \frac{\lVert \nabla f(x+tv)-\nabla f(x) \rVert}{t} \lVert v \rVert &  & (\text{Cauchy--Schwarz inequality})   \\
& \leq \lim_{t \to 0} \frac{L\lVert tv \rVert}{t} \lVert v \rVert                        &  & (\text{by $L$-smoothness definition}) \\
& = L \lVert v \rVert^2.
\end{align*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\preceq L I$ という結果が得られました。
\myQED

</details>

#### ヘッセ行列の固有値のバウンド

$L$-平滑性と $\mu$-強凸性を組み合わせると、ヘッセ行列の固有値に対するバウンドも得られます。

**Proposition 4**

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。
$f$ が $L$-平滑かつ $\mu$-強凸であれば、任意の $x \in \mathbb{R}^n$ に対してヘッセ行列 $\nabla^2 f(x)$ の固有値は区間 $[\mu, L]$ に含まれる。

<details>
<summary>Proof</summary>

\cref{prop:convexity-hessian,prop:smoothness-hessian} より、全ての $x \in \mathbb{R}^n$ に対して、次が成り立ちます。

```math
\begin{equation*}
\mu I \preceq \nabla^2 f(x) \preceq L I.
\end{equation*}
```

これより $\nabla^2 f(x)$ の固有値が区間 $[\mu, L]$ に含まれることが直ちに従います。
\myQED

</details>

#### Baillon--Haddadの定理

発展的な内容として、$L$-平滑性の有用な性質の一つが次の Baillon--Haddadの定理です。
ここでは $C^1$ の微分可能性だけを仮定する点に注意してください。

**Proposition 5** (Baillon--Haddad theorem)

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であるとする。$f$ が $L$-平滑かつ凸であれば、任意の $x,y \in \mathbb{R}^n$ に対して $\nabla f$ は $1/L$-cocoercive である、すなわち

```math
\begin{equation*}
(\nabla f(x)-\nabla f(y))^\top (x-y) \ge \frac{1}{L} \lVert \nabla f(x)-\nabla f(y) \rVert^2
\end{equation*}
```

が任意の $x,y \in \mathbb{R}^n$ について成り立つ。

証明は他の文献を参照してください~\citep{bauschkeBaillonHaddadTheoremRevisited2009}~\citep[Proposition 12.60]{rockafellarVariationalAnalysis1998}。

最適化アルゴリズムが生成する列 $\lbrace x_k \rbrace$ に対し、

```math
\begin{equation*}
s_k \mathrel{\vcenter{:}}= x_{k+1}-x_k, \quad y_k \mathrel{\vcenter{:}}= \nabla f(x_{k+1}) - \nabla f(x_k)
\end{equation*}
```

と定義した上で、Baillon--Haddad 定理を $x=x_{k+1}$、$y=x_k$ に適用すると、次の結果が得られます。

```math
\begin{equation*}
s_k^\top y_k \ge \frac{1}{L} \lVert y_k \rVert^2.
\end{equation*}
```

この不等式は準ニュートン法の一種であるBFGS法やL-BFGS法などの更新則の解析で用いられています。

