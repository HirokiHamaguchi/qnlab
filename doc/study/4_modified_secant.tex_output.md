
## 修正セカント条件

このセクションでは、準ニュートン法で使用される標準的なセカント条件の修正版である修正セカント条件について説明します。
修正セカント条件は勾配情報に加えて関数値情報を取り込み、より正確なヘッセ行列の近似を実現します。

### 修正セカント条件の導出

セカント条件は多くの準ニュートン法で広く用いられていますが、勾配情報のみを活用するため、目的関数の曲率を正確に捉えられないことがあります。
このことをFig. 10で説明します。

Fig. 10 では、対応する関数値と勾配を持つ2つの点 $x_k$ と $x_{k+1}$ があります。
Fig. 10 では、正確なヘッセ行列から構築された理想的な二次モデルが新しい点 $x_{k+1}$ の周りでよくフィットし、より良い収束性能をもたらすことが見られます。
しかし、Fig. 10 では、標準的なセカント更新はこれらの点での勾配のみを照合し、$x_k$ での関数値を無視します。曲率を大きく誤推定する可能性があり、真の目的関数の近似が悪くなります。

<img src="../imgs/modified_secant/trial_EXPLAIN.png" /><img src="../imgs/modified_secant/trial_HESS.png" /><img src="../imgs/modified_secant/trial_BFGS.png" />

(Fig. 10 標準的なセカント条件の欠点。 (a) $x_k$ と $x_{k+1}$ での関数値と勾配が既に分かっている。 (b) 正確なヘッセ行列から構築された理想的な二次モデルは、新しい点 $x_{k+1}$ の周りでよくフィットする。 (c) 標準的なセカント更新では、必ずしも十分に曲率を捉えられない。)

関数値を利用することでこの制限を克服できます。
基本的な考え方はFig. 11 に示されています。
2 点 $x_k$ と $x_{k+1}$ で勾配が同じであっても、関数値の自然な内挿は関数値 $f(x_k)$ と $f(x_{k+1})$ に応じて異なります。
この観察は、関数値情報を取り込んでヘッセ行列の近似を改善する修正セカント条件の動機となります。

![../imgs/modified_secant/cubic_interpolation.png](../imgs/modified_secant/cubic_interpolation.png)

(Fig. 11 $x_k$ と $x_{k+1}$ で同一の勾配を持つが異なる関数値による内挿。これは異なる内挿関数を生じさせ、ヘッセ行列の近似に関数値情報を組み込むことの重要性を強調しています。)

以下では、2つの既知の修正セカント条件を提示します。

#### 関数値一致の修正セカント条件

最初の修正は、二次モデルが前の点での関数値と一致することを強制します~\citep{yuanModifiedBFGSAlgorithm1991,weiNewQuasiNewtonMethods2006, babaie-kafakiModifiedBFGSAlgorithm2011}。
異なる近似ヘッセ行列 $B^{\mathrm{F}}_{k+1}$ を持つ別のモデルを考えます。

```math
\begin{equation*}
m_{k+1}^{\mathrm{F}}(x) \mathrel{\vcenter{:}}= f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B^{\mathrm{F}}_{k+1} (x - x_{k+1}).
\end{equation*}
```

ここでは、このモデルが

```math
\begin{equation*}
m^{\mathrm{F}}_{k+1}(x_k) = f(x_k),
\end{equation*}
```

つまり、前の点での関数値が、モデル関数でも一致することを課します。

対応する修正セカント条件を $B^{\mathrm{F}}_{k+1}$ に対して導出しましょう。
$m^{\mathrm{F}}_{k+1}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、条件は以下になります。

```math
\begin{align*}
f(x_k) & = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x_k - x_{k+1}) + \frac{1}{2} (x_k - x_{k+1})^\top B^{\mathrm{F}}_{k+1} (x_k - x_{k+1}) \\
& = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top B^{\mathrm{F}}_{k+1} s_k.
\end{align*}
```

ここで、通常のセカント条件の代わりに、次のように単位行列による修正を加えた形を仮定します。

```math
\begin{equation*}
\left(B^{\mathrm{F}}_{k+1} + \sigma^\mathrm{F}_k I \right) s_k = y_k.
\end{equation*}
```

ここで、$\sigma^\mathrm{F}_k \in \mathbb{R}$ は以下で決定するスカラーです。
この方程式を代入すると、以下が得られます。

```math
\begin{equation*}
f(x_k) = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top y_k - \frac{\sigma^\mathrm{F}_k}{2} \lVert s_k \rVert^2.
\end{equation*}
```

$\sigma^\mathrm{F}_k$ について解き、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、以下を得ます。

```math
\begin{align*}
\sigma^\mathrm{F}_k & = \frac{2(f(x_{k+1}) - f(x_k)) - (2\nabla f(x_{k+1}) - y_k)^\top s_k}{\lVert s_k \rVert^2}           \\
& = \frac{2(f(x_{k+1}) - f(x_k)) - (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\lVert s_k \rVert^2}.
\end{align*}
```

従って、修正セカント条件は以下になります。

```math
\begin{equation*}
B^{\mathrm{F}}_{k+1} s_k = \hat{y}^{\mathrm{F}}_k \mathrel{\vcenter{:}}= y_k + \frac{2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

よって、各種の準ニュートン法の更新において、$s_k, y_k$ の代わりに、この修正された $\hat{y}^\mathrm{F}_k$ を使用することで、殆ど同じアルゴリズムのまま、目的関数値の情報を取り込むことができます。
また、特にBFGS更新で $s_k^\top y_k > 0$ という条件は正定値性を保つために使われるなど重要な性質である為、$s_k$ との内積が正であることを保証した、次の修正も考えられます。

```math
\begin{equation*}
B^{\mathrm{F}'}_{k+1} s_k = y_k + \frac{\max(0, 2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k)}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

これが関数値一致の修正セカント条件です。
具体例としてFig. 12 も参照してください。

![../imgs/modified_secant/trial_2.png](../imgs/modified_secant/trial_2.png)

(Fig. 12 関数値一致の修正セカント方程式。 前の点 $x_{k-1}$ での関数値 $f(x_{k-1})$ と $m^{\mathrm{F}}_k(x_{k-1})$ が一致している。)

#### 三次項による修正セカント条件

2つ目の修正セカント条件では、モデルに三次項を導入し、前の点での関数値と勾配の両方の一致を同時に満たすことを可能にします~\citep{zhangNewQuasiNewtonEquation1999, zhangPropertiesNumericalPerformance2001,yabeLocalSuperlinearConvergence2007}。

$T_{k+1} \in \mathbb{R}^{n \times n \times n}$ を、以下を満たす $x_{k+1}$ での $f$ の3階微分テンソルとします。

```math
\begin{equation*}
s_k^\top (T_{k+1} s_k) s_k = \sum_{i,j,l=1}^n \partial_{x_i x_j x_l} f(x_{k+1}) s_k^{(i)} s_k^{(j)} s_k^{(l)}.
\end{equation*}
```

ここで、$\partial_{x_i x_j x_l} f$ は $f$ の $x_i$、$x_j$、および $x_l$ に対する3階微分を表し、$s_k^{(i)}$ はベクトル $s_k$ の第 $i$ 成分です。
このテンソルは分析目的でのみ導入しており、最終的な式からは除去されます。

このテンソル項を組み込むことにより、以下のモデル関数を定義できます。

```math
\begin{align*}
m^\mathrm{C}_{k+1}(x) \mathrel{\vcenter{:}}={} & f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1}^{\mathrm{C}} (x - x_{k+1}) \\*
& + \frac{1}{6}(x - x_{k+1})^\top (T_{k+1} (x - x_{k+1})) (x - x_{k+1}).
\end{align*}
```

このモデルを使用して、前の点 $x_k$ での関数値と勾配の両方の一致を強制できます。
具体的には、以下を要求します。

```math
\begin{equation*}
\begin{cases}
m^\mathrm{C}_{k+1}(x_k) = f(x_k), \\
\nabla m^\mathrm{C}_{k+1}(x_k) = \nabla f(x_k).
\end{cases}
\end{equation*}
```

$m^\mathrm{C}_{k+1}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、これらの条件を以下のように書き直せます。

```math
\begin{align*}
f(x_k)        & = f(x_{k+1}) - s_k^\top \nabla f(x_{k+1}) + \frac{1}{2} s_k^\top B^\mathrm{C}_{k+1} s_k - \frac{1}{6} s_k^\top (T_{k+1} s_k) s_k, \\
\nabla f(x_k) & = \nabla f(x_{k+1}) - B^\mathrm{C}_{k+1} s_k + \frac{1}{2} (T_{k+1} s_k) s_k.
\end{align*}
```

第1の方程式の両辺に3を乗じて整理し、第2の方程式については $s_k$ との内積をとることで、以下が得られます。

```math
\begin{align*}
3(f(x_k) - f(x_{k+1})) +3 s_k^\top \nabla f(x_{k+1}) & = \frac{3}{2} s_k^\top B^{\mathrm{C}}_{k+1} s_k - \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k, \\
-s_k^\top y_k                                        & = -s_k^\top B^{\mathrm{C}}_{k+1} s_k + \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k.
\end{align*}
```

合計し、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、テンソル項を除去し、以下のスカラー恒等式を得ます。

```math
\begin{equation*}
3(f(x_k) - f(x_{k+1}))
+ \frac{3}{2} s_k^\top (\nabla f(x_{k+1}) + \nabla f(x_k)) + \frac{1}{2} s_k^\top y_k
= \frac{1}{2} s_k^\top B^{\mathrm{C}}_{k+1} s_k.
\end{equation*}
```

ここでも、あるスカラー $\sigma^\mathrm{C}_k$ に対して $\left( B^{\mathrm{C}}_{k+1} + \sigma^\mathrm{C}_k I \right) s_k = y_k$ と仮定します。
そうすると、先の方程式は

```math
\begin{equation*}
3(f(x_k) - f(x_{k+1})) + \frac{3}{2} s_k^\top (\nabla f(x_{k+1}) + \nabla f(x_k)) = - \frac{\sigma^\mathrm{C}_k}{2} \lVert s_k \rVert^2,
\end{equation*}
```

つまり、

```math
\begin{equation*}
\sigma^\mathrm{C}_k = -\frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\lVert s_k \rVert^2}.
\end{equation*}
```

従って、以下の修正セカント条件が得られます。

```math
\begin{equation*}
B^{\mathrm{C}}_{k+1} s_k = y_k + \frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\lVert s_k \rVert^2} s_k.
\end{equation*}
```

これが三次項による修正セカント条件です。
具体例としてFig. 13も参照してください。

<img src="../imgs/modified_secant/trial_1_cubic.png" /><img src="../imgs/modified_secant/trial_1_quadratic.png" />

(Fig. 13 三次項による修正セカント方程式。Fig. 13 では、三次項をモデルに組み込むことで、前の点で関数値と勾配の両方が一致します。Fig. 13 では、その二次までの展開によるモデルを示しています。)

### その他の曲率関連手法

曲率情報を表すベクトルペア $s_k, y_k$ を求めることについては、他にもいくつかのトピックがあります。
Agg-BFGS~\citep{berahasLimitedmemoryBFGSDisplacement2022} は、最も古い情報を破棄して最新のものを追加するのではなく、データを集約することにより曲率情報を管理する別のアプローチです。

Multi-Secant~\citep{leeAdvancingMultiSecantQuasiNewton2025} とは、複数のステップと勾配差ベクトルのペアを維持することにより、セカント条件を拡張します。ここまでで見てきた標準的な形式では、次のように定義していました。

```math
\begin{equation*}
s_i = x_{i+1} - x_i, \quad y_i = \nabla f(x_{i+1}) - \nabla f(x_i). \quad (i = k-m, \ldots, k)
\end{equation*}
```

一方、Multi-Secantでは、すべてのベクトルを最新の点を中心としています。

```math
\begin{equation*}
s_i = x_{k+1} - x_i, \quad y_i = \nabla f(x_{k+1}) - \nabla f(x_i). \quad (i = k-m, \ldots, k)
\end{equation*}
```

このアプローチによって、場合によってはより良い近似が得られることもあります。

