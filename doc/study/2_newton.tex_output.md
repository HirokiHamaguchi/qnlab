
## ニュートン法

本節では、無制約最適化問題に対する基本的な最適化アルゴリズム
である、ニュートン法の概要を示します。

### ニュートン法のアルゴリズム

ニュートン法は、初期点 $x_0$ から出発し、現在の点を逐次更新して点列 $\lbrace x_k \rbrace_{k=0}^\infty$ を生成する代表的な反復アルゴリズムです。

$f$ は $C^2$ 級かつ強凸であると仮定すると、ヘッセ行列 $\nabla^2 f(x)$ は任意の $x \in \mathbb{R}^n$ で正定値となり可逆です。
$k$ 反復目の点 $x_k$ における勾配 $g_k \mathrel{\vcenter{:}}= \nabla f(x_k)$ とヘッセ行列 $\nabla^2 f(x_k)$ を用いて、ニュートン法の基本手続きは次のように表せます。

```math
\begin{equation*}
x_{k+1} \gets x_k - \alpha_k \nabla^2 f(x_k)^{-1} g_k.
\end{equation*}
```

ここで $\alpha_k > 0$ は直線探索によって定められるステップサイズです。
この更新規則は次のように導出できます。
$x_k$ での $f$ の二次のテイラー近似は次式で与えられます。

```math
\begin{equation*}
m^*_{k}(x) \mathrel{\vcenter{:}}= f(x_k) + g_k^\top (x - x_k) + \frac{1}{2} (x - x_k)^\top \nabla^2 f(x_k) (x - x_k).
\end{equation*}
```

このモデルの勾配は次のように書けます。

```math
\begin{equation*}
\nabla m^*_k(x) = g_k + \nabla^2 f(x_k)(x - x_k).
\end{equation*}
```

仮定よりこの二次モデルは強凸であるため、点 $x^* \in \mathbb{R}^n$ が $m^*_k$ の最小点であることと $\nabla m^*_k(x) = 0$ が成り立つことは同値です。従って、解は以下の通りです。

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\mathrm{arg \ min}} \ m^*_{k}(x) = x_k -\left(\nabla^2 f(x_k)\right)^{-1} g_k.
\end{equation*}
```

次の反復点として、直接 $x^*$ を選ぶことは自然に思えますが、後で見るように一般にこのままでは大域収束性を保証できません。
そこで、関数値の十分な減少を確保するため、直線探索で定めるステップサイズ $\alpha_k > 0$ を導入すると、最初に示した更新式が得られます。
これがニュートン法の基本的な手続きとなります。

先ほどの更新方向 $d_k \mathrel{\vcenter{:}}= -\nabla^2 f(x_k)^{-1} g_k$ はニュートン方向と呼ばれています。ヘッセ行列が正定値であれば、次式より降下方向、つまり目的関数が減少していく方向であることが分かります。

```math
\begin{equation*}
g_k^\top d_k = -g_k^\top \nabla^2 f(x_k)^{-1} g_k < 0.
\end{equation*}
```

$f$ が非凸である、つまりヘッセ行列 $\nabla^2 f(x_k)$ が負定値または不定値である場合、降下方向であるとは限らないので、ニュートン法は収束しない可能性があります。
このような場合への一つの対処法として修正ニュートン法~\citep[Sec. 3.4]{nocedal1999numerical} が知られていますが、ここでは説明を省略します。

### ニュートン法の収束性

次に、ニュートン法に対する標準的な収束性に関する結果を示します。

#### 大域収束

まず、ニュートン法に限らない直線探索付きの手法に対する、一般的な大域収束性の結果

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0
\end{equation*}
```

を述べ、その後にニュートン法への適用を説明します。

次の形の一般的な反復法を考えます。

```math
\begin{equation*}
x_{k+1} \gets x_k + \alpha_k d_k.
\end{equation*}
```

ここで $d_k$ は $g_k^\top d_k < 0$ を満たす降下方向であり、$\alpha_k > 0$ は直線探索で決定されるステップサイズです。
ステップサイズ $\alpha_k$ は (弱)Wolfe 条件~\citep[Sec. 3.1]{nocedal1999numerical} により次のように定められるとします。

```math
\begin{equation*}
\begin{cases}
f(x_k + \alpha_k d_k) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k, \\
-g_{k+1}^\top d_k     & \leq -c_2 g_k^\top d_k.
\end{cases}
\end{equation*}
```

ここで $0 < c_1 < c_2 < 1$ は定数です。
一つ目の条件は、Armijo 条件とも呼ばれ、次の点での関数値 $f(x_k + \alpha_k d_k)$ が、現在の点での関数値 $f(x_k)$ よりも、勾配から導かれる予測値 $\alpha_k g_k^\top d_k$ の少なくとも $c_1$ 倍した値は減少していることを要求します。
二つ目の条件は、曲率条件とも呼ばれ、探索方向に射影した次の点での減少率 $-g_{k+1}^\top d_k$ が、現在の点での射影された減少率 $-g_k^\top d_k$ よりも、少なくとも $c_2$ 倍した値は小さくなっている、つまり探索方向 $d_k$ 上においては、十分に最適化されていることを要求しています。

また、方向 $d_k$ と負の勾配 $-g_k$ のなす角を $\theta_k \in [0, \pi]$ とし、次式を満たすように定めます。

```math
\begin{equation*}
\cos \theta_k = \frac{-g_k^\top d_k}{\lVert g_k \rVert\lVert d_k \rVert}.
\end{equation*}
```

これらの設定の下で、次の古典的に良く知られた結果の簡略版を以下に示します。

**Theorem 1** ({\cite[Theorem 3.2]{nocedal1999numerical}})

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であり $\mathbb{R}^n$ 上で下に有界で、かつ $L$-平滑だとする。
次の反復法を考える。

```math
\begin{equation*}
x_{k+1} \gets x_k + \alpha_k d_k.
\end{equation*}
```

初期点 $x_0 \in \mathbb{R}^n$ から開始され、ステップサイズ $\alpha_k$ は Wolfe 条件を満たすとする。
角 $\lbrace \theta_k \rbrace_k$ に対して、ある正の定数 $\delta$ が存在して $\cos \theta_k \geq \delta > 0$ が全ての $k$ で成り立つならば、
この反復法は次を満たす点列 $\lbrace x_k \rbrace_k$ を生成する。

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0.
\end{equation*}
```

<details>
<summary>Proof</summary>

Wolfe 条件、Cauchy--Schwarz の不等式、$f$ の $L$-平滑性より、次の二つの関係式が成り立ちます。

```math
\begin{equation*}
\begin{cases}
(g_{k+1} - g_k)^\top d_k = g_{k+1}^\top d_k - g_k^\top d_k
\geq (c_2 - 1) g_k^\top d_k, \\
(g_{k+1} - g_k)^\top d_k \leq
\lVert g_{k+1} - g_k \rVert \lVert d_k \rVert \leq L \lVert x_{k+1} - x_k \rVert \lVert d_k \rVert  = \alpha_k L \lVert d_k \rVert^2.
\end{cases}
\end{equation*}
```

これらを組み合わせると次の不等式が得られます。

```math
\begin{equation*}
\alpha_k \geq \frac{c_2 - 1}{L} \frac{g_k^\top d_k}{\lVert d_k \rVert^2}.
\end{equation*}
```

よって、

```math
\begin{align*}
f(x_{k+1}) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k                                          &  & (\text{Armijo condition})         \\
& \leq f(x_k) - c_1 \frac{1 - c_2}{L} \frac{(g_k^\top d_k)^2}{\lVert d_k \rVert^2} &  & (\text{previous inequality})      \\
& = f(x_k) - c_1 \frac{1 - c_2}{L} \cos^2 \theta_k \lVert g_k \rVert^2.            &  & (\text{definition of $\theta_k$})
\end{align*}
```

この不等式を繰り返し適用することで次が得られます。

```math
\begin{equation*}
f(x_{k+1}) \leq f(x_0) - c_1 \frac{1 - c_2}{L} \sum_{j=0}^k \cos^2 \theta_j \lVert g_j \rVert^2.
\end{equation*}
```

仮定より $f$ は下に有界なので、ある定数 $f^*$ が存在して、全ての $k$ について $f(x_{k+1}) \geq f^*$ となります。
従って、Zoutendijk 条件と呼ばれる次式が得られます。

```math
\begin{equation*}
\sum_{k=0}^\infty \cos^2 \theta_k \lVert g_k \rVert^2 \leq \frac{L}{c_1 (1 - c_2)} (f(x_0) - f^*) < \infty.
\end{equation*}
```

この条件より、次が従います。

```math
\begin{equation*}
\cos^2 \theta_k \lVert g_k \rVert^2 \to 0.
\end{equation*}
```

$\cos \theta_k \geq \delta > 0$ がすべての $k$ で成り立つという仮定と合わせると、直ちに次が従います。

```math
\begin{equation*}
\lim_{k \to \infty} \lVert g_k \rVert = 0.
\end{equation*}
```

よって、Wolfe条件による直線探索付きの一般的な反復法に対する大域収束性が示されました。
\myQED

</details>

ニュートン法は $d_k = -\nabla^2 f(x_k)^{-1} g_k$ とした Theorem 1 の特別な場合なので、適切な $f$ に関する条件の下でこの結果を適用できます。
特に $f$ が $L$-平滑かつ $\mu$-強凸であれば、任意の $k$ について次が成り立ちます。

```math
\begin{equation*}
\cos \theta_k
= \frac{g_k^\top \nabla^2 f(x_k)^{-1} g_k}{\lVert g_k \rVert\lVert \nabla^2 f(x_k)^{-1} g_k \rVert}
\geq \frac{\lVert g_k \rVert^2 \lambda_{\min}(\nabla^2 f(x_k)^{-1})}{\lVert g_k \rVert^2 \lambda_{\max}(\nabla^2 f(x_k)^{-1})}
\geq \frac{\mu}{L},
\end{equation*}
```

ここで $\lambda_{\min}(\cdot)$ と $\lambda_{\max}(\cdot)$ はそれぞれ最小固有値と最大固有値を表すとします。
従って、Theorem 1 において $\delta = \mu / L$ とすれば、Wolfe 条件を満たす直線探索の下で、強凸かつ平滑な関数に対するニュートン法の大域収束性が得られます。

#### 局所二次収束

次に、ニュートン法の局所収束性に関する古典的結果を示します。
ここでも簡潔さのために簡略版を示します。

**Theorem 2** ({\cite[Theorem 3.5]{nocedal1999numerical}})

ヘッセ行列 $\nabla^2 f(x)$ が解 $x^*$ の近傍で Lipschitz連続であり、最適性の二次の十分条件が成り立つとする。すなわち $\nabla f(x^*)=0$ かつ $\nabla^2 f(x^*)$ は正定値である。
前節までのニュートン法において、すべての $k$ で $\alpha_k=1$ が満たされるとし、初期点 $x_0$ が $x^*$ に十分近いとき、勾配ノルム列 $\lbrace\lVert \nabla f(x_k) \rVert\rbrace$ は二次収束する。

<details>
<summary>Proof</summary>

$k=0$ とし、$x_k$ から $x_{k+1}$ への更新を考える。
ヘッセ行列 $\nabla^2 f(x_k)$ は $x^*$ の十分小さい近傍で可逆であるため、最適性条件 $\nabla f(x^*)=0$ と $\alpha_k=1$ より次が成り立つ。

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
& \lVert \nabla^2 f(x_k)(x_k-x^*)-\left(\nabla f(x_k)-\nabla f(x^*)\right) \rVert                              \\
=   {}  & \left\lVert \nabla^2 f(x_k)(x_k-x^*)-\int_0^1 \nabla^2 f(x_k+t(x^*-x_k)) (x_k - x^*)\mathrm{d}t \right\rVert \\
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

$\nabla^2 f(x^*)$ は正定値なので、ある半径 $r>0$ が存在して $\lVert x-x^* \rVert\le r$ を満たすすべての $x$ について

```math
\begin{equation*}
\left\lVert \left(\nabla^2 f(x)\right)^{-1} \right\rVert\le 2 \left\lVert \left(\nabla^2 f(x^*)\right)^{-1} \right\rVert
\end{equation*}
```

が成り立つ。
これらを合わせると次を得る。

```math
\begin{align*}
\lVert x_{k+1} - x^* \rVert
& \leq \left\lVert \left(\nabla^2 f(x_k)\right)^{-1} \right\rVert \left\lVert \nabla^2 f(x_k)(x_k-x^*)-\left(\nabla f(x_k)-\nabla f(x^*)\right) \right\rVert \\
& \le L^{\mathrm{H}} \left\lVert \left(\nabla^2 f(x^*)\right)^{-1} \right\rVert \lVert x_k-x^* \rVert^2.
\end{align*}
```

$\widetilde L \mathrel{\vcenter{:}}= L^{\mathrm{H}}\left\lVert \left(\nabla^2 f(x^*)\right)^{-1} \right\rVert$ とおく。
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

よって $\lVert \nabla f(x_k) \rVert$ は0へと局所二次収束する。
\myQED

</details>

これらの\cref{thm:line-search-global-convergence,thm:newton-quadratic}は、直線探索を伴うニュートン法が適切な条件の下で大域収束性と局所二次収束性の両方を持つことを示しています。
この高速な収束性は、通常は線形収束にとどまる勾配降下法などの一階法と比べたときの大きな利点となります。
一方で、ヘッセ行列 $\nabla^2 f(x_k) \in \mathbb{R}^{n \times n}$ の計算や線形方程式 $\nabla^2 f(x_k) d_k = -\nabla f(x_k)$ の求解には、特に大規模問題において、膨大な計算コストを要します。
そのため、準ニュートン法などの他の最適化手法が一般的には用いられることも多く、これについて後の節では述べていきます。

### 大域収束のために必要な要素

先ほど述べたように、ヘッセ行列の正定値性と直線探索によるステップサイズの選択はニュートン法で重要な役割をそれぞれ担っています。
本小節では、これらが必要となる理由について説明します。

#### ヘッセ行列の正定値性

ここまで $f$ が強凸であると仮定してきましたが、これは各反復でヘッセ行列 $\nabla^2 f(x_k)$ が正定値であることを意味しています。
この仮定は、ニュートン法が局所的に最適解へ収束するために重要です。
ヘッセ行列が正定値であれば、次式よりニュートン法は降下方向を与えます。

```math
\begin{equation*}
\nabla f(x_k)^\top d_k
= -\nabla f(x_k)^\top \left(\nabla^2 f(x_k)\right)^{-1} \nabla f(x_k)
< 0.
\end{equation*}
```

一方、ヘッセ行列が負定値または不定値の場合、関数値の減少は保証されず、ニュートン法が最適解から離れてしまう可能性があります。
不定値の場合は関数値が減少することもありますが、鞍点へ収束するリスクも高まります。
従って、ニュートン法を適用する際にはヘッセ行列の正定値性が重要となってきます。

#### フルステップの問題

ニュートン法では、各反復でステップサイズを $\alpha_k=1$ とすることが困難を引き起こす場合があります。
ここではそのような失敗の具体例を挙げ、対処として直線探索が必要となることを述べます。

##### ニュートン法が発散する例

次の関数を考えます。

```math
\begin{align*}
f(x)   & \mathrel{\vcenter{:}}= \sqrt{1 + x^2},    \\
f'(x)  & = \frac{x}{\sqrt{1 + x^2}},  \\
f''(x) & = \frac{1}{(1 + x^2)^{3/2}}.
\end{align*}
```

初期点の絶対値が 1 を超えると、Fig. 3 に示すようにこの関数に対するニュートン法は発散します。

これは、最適解 $x^*=0$ から離れた点では、2階微分の値、つまりヘッセ行列の固有値が非常に小さくなり、ニュートンステップが過大になる為です。反復を重ねるごとに、より遠くへ飛んでいき、最適解から離れてしまいます。

![../imgs/quasi_newton/newton_failure_sqrt_function_1.1.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_sqrt_function_1.1.png)

(Fig. 3 初期点 $x_0=1.1$ でニュートン法が発散する例)

##### 強凸関数に対するニュートン法の振動

前の例では目的関数は強凸ではなく、必ずしも良い性質を持っている訳ではありませんでした。
しかし、強凸性を持つ関数であってもニュートン法が収束しない例が知られています~\citep[Example 1.4.3]{Doikov2021SecondOrderTensor}。

この例では $\mu>0$ に対し、次の関数を考えます。

```math
\begin{align*}
f(x)     & \mathrel{\vcenter{:}}= \log(1 + e^x) - \frac{x}{2} + \frac{\mu x^2}{2}, \\
f'(x)    & = \frac{e^x}{1+e^x} - \frac{1}{2} + \mu x,                 \\
f''(x)   & = \frac{e^x}{(1+e^x)^2} + \mu,                             \\
f'''(x)  & = \frac{e^x(1 - e^x)}{(1+e^x)^3},                          \\
f''''(x) & = \frac{e^x(1 - 4e^x + e^{2x})}{(1+e^x)^4}.
\end{align*}
```

この関数は次の性質を持っています。

1. $\mu$-強凸である。
2. $\max_x |f''(x)| = \frac{1}{4} + \mu$ であり、これは $e^x=1$ で達成される。従って $\nabla f$ は $L$-平滑である ($L=\frac{1}{4}+\mu$)。
3. $\max_x |f'''(x)| = \frac{1}{6\sqrt{3}}$ であり、これは $e^x=2-\sqrt{3}$ で達成される。従って $\nabla^2 f$ は $M$-Lipschitz である ($M=\frac{1}{6\sqrt{3}}$)。

それにもかかわらず、初期点 $x_0$ が $\mu$ に対して十分大きい場合、Fig. 4 に示すようにニュートン法は振動します。

<img width="50%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_strongly_convex_function_0.1_-4.png" /><img width="50%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_strongly_convex_function_0.01_-4.png" />

(Fig. 4 (左) $x_0=-4, \ \mu=0.1$ ではニュートン法が収束する。(右) $x_0=-4, \ \mu=0.01$ ではニュートン法が振動する。)

##### 直線探索の必要性

これらの例から、ニュートン法でフルステップ $\alpha_k=1$ を採用すると、強凸かつ平滑な関数であっても発散や振動を引き起こす可能性があることが分かりました。
従って、収束を保証するためにステップサイズ $\alpha_k$ を適切に選択することが不可欠です。
このように直線探索を伴うニュートン法は、修正ニュートン法とも呼ばれています。

### 根探索としてのニュートン法との比較

概念的な補足として、「根探索としてのニュートン法」 と 「最適化におけるニュートン法」 の関係を簡潔に整理します。
両者は密接に関係した手法ですが、考え方や適用対象は異なります。

![../imgs/quasi_newton/newton_raphson.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_raphson.png)

(Fig. 5 勾配 $\nabla f(x)=3 x^2 - 4 x + 1$ に対する根探索と関数 $f(x)=x^3 - 2 x^2 + x$ に対する最適化。)

#### 根探索としてのニュートン法

単にニュートン法(あるいはニュートン・ラフソン法)と言うときは、微分可能なスカラー関数 $g\colon \mathbb{R} \to \mathbb{R}$ に対して $g(x) = 0$ を解く根探索アルゴリズムを指すことも少なくありません。
これは本節の主題ではないが、基本的でよく知られている手法と言えます。

初期値 $x_0$ から開始するとして、この手法の反復は次式で与えられます。

```math
\begin{equation*}
x_{k+1} = x_k - \frac{g(x_k)}{g'(x_k)}.
\end{equation*}
```

幾何学的には、$x_k$ 近傍での $g$ のグラフを接線で近似し、その接線と $x$ 軸の交点を次の近似解として選ぶことに、この式は対応しています。

#### 最適化としてのニュートン法

最適化文脈におけるニュートン法は、二回微分可能な関数 $f\colon \mathbb R^{n} \to\mathbb{R}$ の局所最小点を求めるアルゴリズムを指すことが一般的です。

$f$ に対する最適化としてのニュートン法は、勾配 $\nabla f(x)$ に対する根探索としてのニュートン法を適用したものと捉えることもできます。

```math
\begin{equation*}
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
\end{equation*}
```

スカラーの場合、上の式は $x_{k+1}=x_k - \frac{f'(x_k)}{f''(x_k)}$ に簡約されるからです。

#### 二つの定式化の関係

具体例を通じて両者の等価性を確認してみます。
\Cref{fig:newton_raphson}は単純な三次関数 $f(x)$ を用いて両者の関係を示したものです。
左図は $g(x) = \nabla f(x)$ に対する根探索であり、右図は $f(x)$ に対する最適化となっています。

根探索の定式化では、$\nabla f(x) = 3x^2 - 4x + 1$ を接線で近似します。すなわち、

```math
\begin{equation*}
\nabla m^*_k(x) = \nabla f(x_k) + \nabla^2 f(x_k)(x - x_k)
\end{equation*}
```

として、この線形モデルの根を $x_{k+1}$ として選びます。

一方、最適化の定式化では、$f(x) = x^3 - 2x^2 + x$ を二次のテイラー展開で近似し、

```math
\begin{equation*}
m^*_k(x) = f(x_k) + \nabla f(x_k)(x - x_k) + \frac{1}{2}\nabla^2 f(x_k)(x - x_k)^2
\end{equation*}
```

として、この二次モデルの最小点を $x_{k+1}$ として取ります。

従って、正定値性の仮定の下では両者が本質的に等価な操作を行うことが分かり、その対応関係が見てとれます。

