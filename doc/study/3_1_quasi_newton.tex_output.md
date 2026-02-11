
## 準ニュートン法

![../imgs/quasi_newton/newton_vs_qs_vs_gd.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_vs_qs_vs_gd.png)

(Fig. 6 ニュートン法、準ニュートン法、勾配降下法の比較。)

本節では準ニュートン法について説明します。
準ニュートン法はニュートン法に基づいていますが、その主要な欠点であるヘッセ行列やその逆行列の計算コストを低減することを目的としています。具体的には、真のヘッセ行列 $\nabla^2 f(x_k)$ の代わりに近似行列 $B_k$ とその逆行列 $H_k$ を用いることで、高速な収束性を保ちつつ計算コストを抑えます。

\Cref{fig:newton_vs_qs_vs_gd} はニュートン法、準ニュートン法、勾配降下法の比較を示しています。勾配降下法は勾配の反対方向に単純に更新する方法です。1反復あたりの計算コストは最小ですが、収束は遅くなります。ニュートン法は反復回数が最も少ない一方で、1ステップあたりの計算コストが最も高いです。準ニュートン法はその中間に位置し、両者のバランスを取っています。特に、反復回数ではなく実際の計算時間で比較すると、準ニュートン法が最良の性能を示すことが多いです。

直線探索に基づく準ニュートン法では、最適解 $x^*$ に収束する列 ${ x_k }*{k=0}^{\infty}$ を次のように生成します。

```math
\begin{equation*}
x_{k+1} = x_k - \alpha_k B_k^{-1} \nabla f(x_k)
= x_k - \alpha_k H_k \nabla f(x_k).
\end{equation*}
```

ここで $\alpha_k > 0$ は直線探索で定めるステップサイズであり、$B_k$ は点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似です。$H_k \mathrel{\vcenter{:}}= B_k^{-1}$ はその逆行列を表します。
この更新則はニュートン法の更新則と酷似しており、その故にこの手法は準ニュートン法と呼ばれています。

<img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_1.png" /><img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_2.png" /><img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_3.png" />

(Fig. 7 準ニュートン法の概念図。 (1) 目的関数 $f$ (青い曲面) と現在点 $x_k$ (赤点) が与えられます。 (2) 現在の近似ヘッセ行列によって得られる二次モデル $m_k(x)$ (橙色の曲面) を最小化し、その最小点 $x_{k+1}$ (黄色のバツ印) に移動します。 (3) 二次モデルを更新し (緑色の曲面)、この操作を繰り返します。)

準ニュートン法の核心は、各反復で $B_k$ をどのように更新して真のヘッセ行列に近づけるかにあります。\Cref{fig:quasi_newton_overview} は準ニュートン法の概念図です。まず現在の点 $x_k$ の周りで、$B_k$ を用いて目的関数 $f$ の二次近似モデルを構成します。次にこの二次モデルを最小化して次の点 $x_{k+1}$ を得ます。$x_{k+1}$ を得た後、$x_k$ と $x_{k+1}$ における勾配情報を用いて近似行列 $B_k$ を $B_{k+1}$ に更新します。この手続きを収束するまで繰り返すことが準ニュートン法です。

### セカント条件

関数 $f\colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とします。
対称行列 $B_k$ が点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似として与えられたとき、次の点 $x_{k+1}$ における近似ヘッセ行列 $B_{k+1}$ への更新を考えます。
このような行列の候補は無数に存在しますが、真のヘッセ行列が対称であることから、$B_{k+1}$ にも対称性を課すのが自然です。
現在の点 $x_k$ と次の点 $x_{k+1}$ に対して、ステップと勾配差を次のように定義します。

```math
\begin{equation*}
s_k \mathrel{\vcenter{:}}= x_{k+1} - x_k, \qquad   y_k \mathrel{\vcenter{:}}= \nabla f(x_{k+1}) - \nabla f(x_k).
\end{equation*}
```

$y_k^\top s_k \neq 0$ かつ $s_k \neq 0$ を仮定します。なお、$s_k$ の $s$ は step を意味します。
新しい近似は $x_{k+1}$ を中心に構築されるため、$\nabla f(x_k)$ をヘッセ行列 $\nabla^2 f(x_{k+1})$ およびその近似である $B_{k+1}$ を用いて次のように近似します。

```math
\begin{align*}
\nabla f(x_k) & \approx \nabla f(x_{k+1}) + \nabla^2 f(x_{k+1})(x_k - x_{k+1}) \\
& \approx \nabla f(x_{k+1}) + B_{k+1}(x_k - x_{k+1}).
\end{align*}
```

上の近似を等式として要求すると、次を得ます。

```math
\begin{equation*}
B_{k+1}(x_{k+1} - x_k) = \nabla f(x_{k+1}) - \nabla f(x_k).
\end{equation*}
```

あるいは同値に、

```math
\begin{equation*}
B_{k+1} s_k = y_k.
\end{equation*}
```

この関係はセカント条件、または準ニュートン方程式と呼ばれています。

より厳密には、セカント条件は次のようにも正当化できます。$x_{k+1}$ の周りの二次近似モデルを考えます。

```math
\begin{equation*}
m_{k+1}(x) = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1} (x - x_{k+1}).
\end{equation*}
```

このモデルは構成上、$B_{k+1}$ によらず次を満たします。

```math
\begin{equation*}
\begin{cases}
m_{k+1}(x_{k+1}) = f(x_{k+1}), \\
\nabla m_{k+1}(x_{k+1}) = \nabla f(x_{k+1}).
\end{cases}
\end{equation*}
```

さらに、セカント条件により、このモデルは次も満たします。

```math
\begin{align*}
\nabla m_{k+1}(x_k) & = \nabla f(x_{k+1}) - B_{k+1} (x_{k+1} - x_k)             &  & (\text{definition of the model}) \\
& = \nabla f(x_{k+1}) - (\nabla f(x_{k+1}) - \nabla f(x_k)) &  & (\text{secant condition})        \\
& = \nabla f(x_k).
\end{align*}
```

これは、セカント条件により二次モデル $m_{k+1}(x)$ が、以前の関数値 $f(x_k)$ を除き、$x_{k+1}$ と $x_k$ の両方における既知の情報を正確に反映していることを意味しています。
