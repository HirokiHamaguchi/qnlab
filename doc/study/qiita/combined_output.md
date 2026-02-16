<!-- markdownlint-disable MD041 -->

<!-- From 0_main.tex -->

配分や計画を効率的に決定したい、信号やデータを高精度に処理したい、ロボットやシステムを安定に制御したい、低リスク高リターンな金融商品を購入したい、データからモデルのパラメータを学習したい。

決めたい内容を決定変数 $x$、その良さを表す定量的な指標を目的関数 $f(x)$ としてモデル化すると、これらの最善な選択は最適化問題として抽象化できます。

数理最適化とは、このような最善の選択、およびそれを遂行する数学的手法や学問領域を指し、本稿では連続最適化、そして特にその代表的な手法であるニュートン法と準ニュートン法を中心として、基礎的な概念をまとめます。

<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/sixhump.png?v=1" />

## 目次

本記事はかなり網羅的かと思われます。興味のある箇所からお読みください。

- [連続最適化の基本概念](#連続最適化の基本概念)
  - [凸性と強凸性](#凸性と強凸性)
  - [Positive Definiteness of the Hessian](#Positive-Definiteness-of-the-Hessian)
  - [平滑性](#平滑性)
    - [ヘッセ行列の固有値のバウンド](#ヘッセ行列の固有値のバウンド)
    - [Baillon–Haddadの定理](#BaillonHaddadの定理)
- [ニュートン法](#ニュートン法)
  - [ニュートン法のアルゴリズム](#ニュートン法のアルゴリズム)
  - [ニュートン法の収束性](#ニュートン法の収束性)
    - [大域収束](#大域収束)
    - [局所二次収束](#局所二次収束)
  - [大域収束のために必要な要素](#大域収束のために必要な要素)
    - [ヘッセ行列の正定値性](#ヘッセ行列の正定値性)
    - [フルステップの問題](#フルステップの問題)
      - [ニュートン法が発散する例](#ニュートン法が発散する例)
      - [強凸関数に対するニュートン法の振動](#強凸関数に対するニュートン法の振動)
      - [直線探索の必要性](#直線探索の必要性)
  - [根探索としてのニュートン法との比較](#根探索としてのニュートン法との比較)
    - [最適化としてのニュートン法](#最適化としてのニュートン法)
    - [根探索としてのニュートン法](#根探索としてのニュートン法)
    - [二つの定式化の関係](#二つの定式化の関係)
- [準ニュートン法](#準ニュートン法)
  - [セカント条件](#セカント条件)
  - [代表的な準ニュートン更新則](#代表的な準ニュートン更新則)
    - [Broyden更新](#Broyden更新)
      - [導出 (Broyden)](#導出-Broyden)
    - [Symmetric Rank-One (SR1) 更新](#Symmetric-Rank-One-SR1-更新)
      - [導出 (SR1)](#導出-SR1)
      - [補足](#補足)
    - [Powell Symmetric Broyden (PSB) 更新](#Powell-Symmetric-Broyden-PSB-更新)
      - [導出 (PSB)](#導出-PSB)
    - [Davidon–Fletcher–Powell (DFP) 更新](#DavidonFletcherPowell-DFP-更新)
      - [導出 (DFP)](#導出-DFP)
    - [Broyden–Fletcher–Goldfarb–Shanno (BFGS) 更新](#BroydenFletcherGoldfarbShanno-BFGS-更新)
      - [双対による導出](#双対による導出)
      - [KLダイバージェンス最小化による導出](#KLダイバージェンス最小化による導出)
  - [BFGS更新の詳細](#BFGS更新の詳細)
    - [逆更新の公式](#逆更新の公式)
    - [BFGS更新の正定値性](#BFGS更新の正定値性)
    - [BFGS更新のトレースと行列式の公式](#BFGS更新のトレースと行列式の公式)
      - [トレースの公式](#トレースの公式)
      - [行列式の公式](#行列式の公式)
  - [BFGSとDFPの比較](#BFGSとDFPの比較)
    - [問題設定](#問題設定)
    - [数値結果](#数値結果)
    - [解析と議論](#解析と議論)
      - [ヘッセ行列の補正における非対称性](#ヘッセ行列の補正における非対称性)
  - [記憶制限BFGS (L-BFGS)](#記憶制限BFGS-L-BFGS)
    - [逆行列のコンパクト表現](#逆行列のコンパクト表現)
    - [二重ループ再帰](#二重ループ再帰)
    - [初期スケーリング](#初期スケーリング)
- [修正セカント条件](#修正セカント条件)
  - [修正セカント条件の導出](#修正セカント条件の導出)
    - [関数値一致の修正セカント条件](#関数値一致の修正セカント条件)
    - [三次項による修正セカント条件](#三次項による修正セカント条件)
  - [その他の曲率関連手法](#その他の曲率関連手法)
- [最後に](#最後に)

---

<!-- From 1_basic.tex -->

## 連続最適化の基本概念

本節では、連続最適化に関する基本概念として、凸性、強凸性、平滑性、およびヘッセ行列の正定値性について説明します。

### 凸性と強凸性

凸性 (convexity) と強凸性 (strong convexity) は最適化理論における基本概念です。
簡単のため、$f$ は $\mathbb{R}^n$ 全域で実数値を取る関数、つまり、$f \colon \mathbb{R}^n \to \mathbb{R}$ とします。また、$\mu>0$ を定数とします。
この時、関数 $f$ が凸、または $\mu$-強凸であることは、任意の $x,y \in \mathbb{R}^n$ と $\lambda \in [0,1]$ について次が成り立つこととそれぞれ同値です。

```math
\begin{align*}
\textrm{(convex)} \quad
f((1-\lambda) x + \lambda y)                                        & \le (1-\lambda) f(x) + \lambda f(y),                                                                    \\
\textrm{($\mu$-strongly convex)} \quad f((1-\lambda) x + \lambda y) & \le (1-\lambda) f(x) + \lambda f(y) - \frac{\mu}{2} \lambda (1-\lambda) \left\lVert x-y \right\rVert^2.
\end{align*}
```

より直感的な定義として、関数 $f$ の勾配を用いた凸性および強凸性の同値な定義も存在します。$f$ が $C^1$ 級で、その定義域 $\mathbb{R}^n$ 全体で実数値を取ると仮定します。関数 $f$ が凸、または $\mu$-強凸であることは、任意の $x,y \in \mathbb{R}^n$ について次が成り立つこととそれぞれ同値です。

```math
\begin{align*}
\textrm{(convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x),                                             \\
\textrm{($\mu$-strongly convex)} \quad
f(y) & \ge f(x)+\nabla f(x)^\top (y-x)+\frac{\mu}{2}\left\lVert y-x \right\rVert^2.
\end{align*}
```

これら二つの定義の同値性を以下で示します。

**Proposition 1**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であるとする。
このとき、凸と $\mu$-強凸に関する二つの定義はそれぞれ同値である。

<details><summary>証明</summary>

$\mu$-強凸の場合について示します。まず、前者の定義が成立することを仮定すると、勾配の定義と、$f$ が $C^1$ 級であることから、以下が成り立ちます。

```math
\begin{align*}
\nabla f(x)^\top (y-x)
& =\lim_{\lambda \to 0} \frac{f(x+ \lambda (y-x)) - f(x)}{\lambda}                                                                                                    \\
& \le \lim_{\lambda \to 0} \frac{1}{\lambda} \left( (1-\lambda) f(x) + \lambda f(y) - \frac{\mu}{2} \lambda (1-\lambda) \left\lVert x-y \right\rVert^2 - f(x) \right) \\
& = f(y) - f(x) - \frac{\mu}{2} \left\lVert x-y \right\rVert^2.
\end{align*}
```

従って、後者の定義が成立します。

続いて、後者の定義が成立することを仮定します。$z \mathrel{\vcenter{:}}= (1-\lambda)x + \lambda y$ とおくと、次が成り立ちます。

```math
\begin{align*}
f(x) & \geq f(z) + \nabla f(z)^\top (x-z) + \frac{\mu}{2} \left\lVert x-z \right\rVert^2, \\
f(y) & \geq f(z) + \nabla f(z)^\top (y-z) + \frac{\mu}{2} \left\lVert y-z \right\rVert^2.
\end{align*}
```

これらを、それぞれ $1-\lambda$ 倍、$\lambda$ 倍して足し合わせると、次が成り立ちます。途中式で、$\nabla f(z)$ を含む項は、$z$ の定義から完全に打ち消し合うことに注意してください。

```math
\begin{align*}
& (1-\lambda) f(x) + \lambda f(y)
\\
\geq{} & f(z) + \nabla f(z)^\top \left( (1-\lambda)(x-z) + \lambda (y-z) \right)
+ \frac{\mu}{2} \left( (1-\lambda) \left\lVert x-z \right\rVert^2 + \lambda \left\lVert y-z \right\rVert^2 \right)                                       \\
={}    & f(z) + \frac{\mu}{2} \left( (1-\lambda) \lambda^2 \left\lVert x-y \right\rVert^2 + \lambda (1-\lambda)^2 \left\lVert y-x \right\rVert^2 \right) \\
={}    & f(z) + \frac{\mu}{2} \lambda (1-\lambda) \left\lVert x-y \right\rVert^2.
\end{align*}
```

従って、前者の定義が成立します。つまり、これらの定義は同値であることが示されました。

$\mu=0$ の場合も、同様の議論により凸性について示すことができます。

(証明終わり)

</details>

強凸性は、凸性に加えて目的関数が一様に正の曲率を持つことを意味します。Fig. 1 に示した通り、凸関数と強凸関数には曲率に違いがみられます。$\mu=0$ を強凸の定義に含めるか否かは、文献によって多少の揺れがありますが、一般的な慣習に倣い本稿では含めません。例えば[^nesterovIntroductoryLecturesConvex2014][^kanamori2016continuous]などを参照してください。

<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/convexity_comparison_convex.png?v=1" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/convexity_comparison_strongly_convex.png?v=1" />

(Fig. 1 凸関数と強凸関数の例。 破線は $x=0$ における二次近似を示しています。 上2つの関数は凸ですが強凸ではなく、下2つの関数は強凸性の定義を満たす $\mu>0$ が存在し強凸となります。)

### Positive Definiteness of the Hessian

続いて、$f$ が $C^2$ 級であるとして、凸性および強凸性がヘッセ行列 $\nabla^2 f(x)$ の正定値性とどのように関係するかを示します。

行列の定値性について、先に定義しておきます。$A$ を $\mathbb{R}^{n \times n}$ の対称行列とします。行列 $A$ が(半)正定値 (positive (semi-)definite)・(半)負定値 (negative (semi-)definite) であるとは、次の条件で定義されます。

```math
\begin{align*}
\textrm{(positive definite)} \quad      & v^\top A v > 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\textrm{(positive semi-definite)} \quad & v^\top A v \ge 0 \quad \forall v \in \mathbb{R}^n,                 \\
\textrm{(negative definite)} \quad      & v^\top A v < 0 \quad \forall v \in \mathbb{R}^n \setminus \lbrace 0 \rbrace, \\
\textrm{(negative semi-definite)} \quad & v^\top A v \le 0 \quad \forall v \in \mathbb{R}^n.
\end{align*}
```

正定値でも負定値でもない行列は不定値 (indefinite) と呼ばれます。行列 $A,B \in \mathbb{R}^{n \times n}$ に対して、$A \succeq B$ は $A-B$ が半正定値であることを表します。特に $B$ が零行列のときは $A \succeq 0$ と書くことがあります。同様に、$\preceq$ は半負定値に対して定義されます。$\mu \geq 0$ に対し、$A \succeq \mu I$ はすべての $v \in \mathbb{R}^n$ について $v^\top A v \ge \mu \left\lVert v \right\rVert^2$ と同値です。これは $A$ の任意の固有値が少なくとも $\mu$ 以上であることを意味し、さらに作用素ノルムについても $\left\lVert A \right\rVert \geq \mu$ であることを導きます。

ここで、強凸性は関数が一様に正の曲率を持つことを意味していました。これはヘッセ行列も一様に正の曲率を持つこと、つまり一様に正定値であることと対応しています。実際、凸性や強凸性は次のように、ヘッセ行列の定値性と同値になることが知られています。

**Proposition 2**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^2$ 級であるとする。
このとき、
- $f$ が凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq0$ が成り立つことは同値である。
- $f$ が $\mu$-強凸であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\succeq\mu I$ が成り立つことは同値である。

<details><summary>証明</summary>

まず $\mu>0$ とし、任意の $x \in \mathbb{R}^n$ に対して $\nabla^2 f(x)\succeq \mu I$ が成り立つと仮定します。 微分積分学の基本定理より、任意の $x,y \in \mathbb{R}^n$ について次が成り立ちます。

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
\ge \int_0^1 \mu\left\lVert y-x \right\rVert^2 \mathrm{d}t
= \mu\left\lVert y-x \right\rVert^2.
\end{equation*}
```

以上の結果を合わせると、$\mu$-強凸性の定義が導かれます。

逆に、$f$ が $\mu$-強凸であると仮定します。任意の $x \in \mathbb{R}^n, \ v \in \mathbb{R}^n$ および $t>0$ に対し、$y=x \pm tv$ とおくと次が成り立ちます。

```math
\begin{equation*}
\begin{cases}
f(x + tv)\ge f(x) + t\nabla f(x)^\top v+\frac{\mu}{2}t^2\left\lVert v \right\rVert^2, \\
f(x - tv)\ge f(x) - t\nabla f(x)^\top v+\frac{\mu}{2}t^2\left\lVert v \right\rVert^2.
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
v^\top \frac{\nabla^2 f(x+ s_+ t v) + \nabla^2 f(x - s_- t v)}{2} v \ge \mu \left\lVert v \right\rVert^2.
\end{equation*}
```

$t \to 0$ とし、$f$ が $C^2$ 級という仮定による $\nabla^2 f$ の連続性を用いると次を得ます。

```math
\begin{equation*}
v^\top \nabla^2 f(x) v \ge \mu \left\lVert v \right\rVert^2.
\end{equation*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\succeq \mu I$ という結果が得られました。

上記の議論で $\mu=0$ とすれば、同様に凸の場合も示されます。

(証明終わり)

</details>

ヘッセ行列が正定値・不定値・負定値である二次関数を Fig. 2 に示しました。定値性と凸性の対応関係を確認できます。

![../imgs/quasi_newton/pd.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/pd.png)

(Fig. 2 二次関数 $f(x)=\frac{1}{2}(x - x_k)^\top H (x - x_k) + \nabla f(x_k)^\top (x - x_k) + f(x_k)$ を二次元空間で示したもの。 ヘッセ行列 $H$ が (左)正定値、(中央)不定値、(右)負定値の場合を示す。 $f(x)$ はそれぞれ (左)凸、(中央)非凸、(右)凹である。)

### 平滑性

最後に、関数の $L$-平滑性 ($L$-smoothness)を導入します。関数 $f$ が $L$-平滑であるとは、ある定数 $L>0$ が存在して

```math
\begin{equation*}
\left\lVert \nabla f(x)-\nabla f(y) \right\rVert \le L\left\lVert x-y \right\rVert
\end{equation*}
```

が任意の $x,y$ について成り立つことと同値です。定義そのものですが、関数 $f$ が $L$-平滑であることは、勾配 $\nabla f$ が $L$-リプシッツ連続であることと同値です。Fig. 3も参照して下さい。

![../imgs/quasi_newton/l-smooth.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/l-smooth.png)

(Fig. 3 一次元における $L$-平滑関数の例。$L$-平滑性は、オレンジ色の破線で示されたように $f(x)$ の上界を与え、また $\nabla f(x)$ が灰色の領域にしか存在させないこと、つまり $\nabla f$ のリプシッツ連続性を意味します。)

非凸な関数に対しても、次の上界が常に成立します。

**Proposition 3**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $L$-平滑であるとする。このとき、任意の $x,y \in \mathbb{R}^n$ について次が成り立つ。

```math
\begin{equation*}
f(y) \le f(x) + \nabla f(x)^\top (y-x) + \frac{L}{2} \left\lVert y-x \right\rVert^2.
\end{equation*}
```

<details><summary>証明</summary>

微分積分学の基本定理より、任意の $x,y \in \mathbb{R}^n$ について次が成り立ちます。

```math
\begin{align*}
f(y) - f(x) & = \int_0^1 \nabla f(x+t(y-x))^\top (y-x) \mathrm{d}t                                                                                                                              \\
& = \nabla f(x)^\top (y-x) + \int_0^1 (\nabla f(x+t(y-x)) - \nabla f(x))^\top (y-x) \mathrm{d}t                                                                                     \\
& \leq \nabla f(x)^\top (y-x) + \int_0^1 \left\lVert \nabla f(x+t(y-x)) - \nabla f(x) \right\rVert \left\lVert y-x \right\rVert \mathrm{d}t &  & (\text{Cauchy–Schwarz inequality}) \\
& \leq \nabla f(x)^\top (y-x) + \int_0^1 L t \left\lVert y-x \right\rVert^2 \mathrm{d}t                                                     &  & (\text{by $L$-smoothness})         \\
& = \nabla f(x)^\top (y-x) + \frac{L}{2} \left\lVert y-x \right\rVert^2.
\end{align*}
```

よって、上界が示されました。

(証明終わり)

</details>

もし、更に $f$ が凸であれば、次の下界も成立します。

**Proposition 4** ([^FanZhongXiuMingLianSokZuiShiHuaarugorizumu2023] (Proposition 2.3.5))

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が凸であり、かつ $L$-平滑であるとする。このとき、任意の $x,y \in \mathbb{R}^n$ について次が成り立つ。

```math
\begin{equation*}
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{1}{2L} \left\lVert \nabla f(y) - \nabla f(x) \right\rVert^2.
\end{equation*}
```

<details><summary>証明</summary>

まず、関数 $g_x(z)$ を次のように定義します。

```math
\begin{align*}
g_x(z)        & \mathrel{\vcenter{:}}= f(z) - f(x) - \nabla f(x)^\top (z-x), \\
\nabla g_x(z) & = \nabla f(z) - \nabla f(x).
\end{align*}
```

関数 $f(z)$ と線形関数 $-f(x) - \nabla f(x)^\top (z-x)$ はどちらも $z$ について凸なので、$g_x(z)$ も凸であり、その最小値は $\nabla g_x(z) = 0$ を満たす $z=x$ でとる、つまり、最小値は $g_x(x)=0$ であることが分かります。
また、$f(z)$ の $L$-平滑性より、$g_x(z)$ の $L$-平滑性も直ちに導かれます。 よって、$g_x(z)$ の $L$-平滑性から次が成り立ちます。

```math
\begin{align*}
0 & \leq g_x\left(y-\frac{\nabla g_x(y)}{L} \right)                                                                             &  & (\text{minimum is $g_x(x)=0$}) \\
& \leq g_x(y) - (\nabla g_x(y))^\top \frac{\nabla g_x(y)}{L} + \frac{L}{2} \left\lVert \frac{\nabla g_x(y)}{L} \right\rVert^2 &  & (\text{by $L$-smoothness})     \\
& = g_x(y) - \frac{1}{2L} \left\lVert \nabla g_x(y) \right\rVert^2                                                                                                \\
& = f(y) - f(x) - \nabla f(x)^\top (y-x) - \frac{1}{2L} \left\lVert \nabla f(y) - \nabla f(x) \right\rVert^2.                 &  & (\text{definition of $g_x$})
\end{align*}
```

よって、下界が示されました。

(証明終わり)

</details>

ここまで見てきたように、$L$-平滑性は勾配の変化率が上から抑えられることを意味してきましたが、これもヘッセ行列の定値性と関連付けることができます。つまり、次の命題で示すように、ヘッセ行列は $L I$ で上から抑えられることになります。

**Proposition 5**

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^2$ 級であるとする。 このとき、$f$ が $L$-平滑であることと、任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ が成り立つことは同値である。

<details><summary>証明</summary>

$f$ は $C^2$ 級なので、微分積分学の基本定理より任意の $x,y \in \mathbb{R}^n$ について次が成り立つ。

```math
\begin{equation*}
\nabla f(y) - \nabla f(x)
= \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t.
\end{equation*}
```

任意の $x \in \mathbb{R}^n$ で $\nabla^2 f(x)\preceq L I$ と仮定します。
このとき $\left\lVert \nabla^2 f(x) \right\rVert \le L$ が成り立つので、

```math
\begin{align*}
\left\lVert \nabla f(y) - \nabla f(x) \right\rVert
& = \left\lVert \int_0^1 \nabla^2 f(x+t(y-x))(y-x) \mathrm{d}t \right\rVert   &  & (\text{previous equation})   \\
& \le \int_0^1 \left\lVert \nabla^2 f(x+t(y-x))(y-x) \right\rVert \mathrm{d}t &  & (\text{triangle inequality}) \\
& \le \int_0^1 L\left\lVert y-x \right\rVert \mathrm{d}t                      &  & (\text{by assumption})       \\
& = L\left\lVert y-x \right\rVert,
\end{align*}
```

つまり、 $f$ の $L$-平滑性が示されました。

逆に、$f$ が $L$-平滑であるとすると、任意の $x \in \mathbb{R}^n$ と $v \in \mathbb{R}^n$ に対して次が成り立ちます。

```math
\begin{equation*}
\left\lVert \nabla f(x+tv)-\nabla f(x) \right\rVert \le L\left\lVert tv \right\rVert = Lt\left\lVert v \right\rVert.
\end{equation*}
```

さらにテイラーの定理より、$t \to 0$ のとき $\left\lVert r(t) \right\rVert/t \to 0$ を満たす剰余項 $r(t)$ を用いて次が成り立ちます。

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
v^\top \nabla^2 f(x) v & = \lim_{t \to 0} \left(\frac{\nabla f(x+tv)-\nabla f(x)}{t}\right)^\top v                                                                               \\
& \leq \lim_{t \to 0} \frac{\left\lVert \nabla f(x+tv)-\nabla f(x) \right\rVert}{t} \left\lVert v \right\rVert &  & (\text{Cauchy–Schwarz inequality})    \\
& \leq \lim_{t \to 0} \frac{L\left\lVert tv \right\rVert}{t} \left\lVert v \right\rVert                        &  & (\text{by $L$-smoothness definition}) \\
& = L \left\lVert v \right\rVert^2.
\end{align*}
```

$v \in \mathbb{R}^n$ は任意なので、$\nabla^2 f(x)\preceq L I$ という結果が得られました。

(証明終わり)

</details>

#### ヘッセ行列の固有値のバウンド

これまでの $L$-平滑性と $\mu$-強凸性に関する議論を組み合わせると、ヘッセ行列の固有値に対するバウンドも得られます。

**Proposition 6**

$f \colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とする。
$f$ が $L$-平滑かつ $\mu$-強凸であれば、任意の $x \in \mathbb{R}^n$ に対してヘッセ行列 $\nabla^2 f(x)$ の固有値は区間 $[\mu, L]$ に含まれる。

<details><summary>証明</summary>

Proposition 2, Proposition 5 より、全ての $x \in \mathbb{R}^n$ に対して、次が成り立ちます。

```math
\begin{equation*}
\mu I \preceq \nabla^2 f(x) \preceq L I.
\end{equation*}
```

これより $\nabla^2 f(x)$ の固有値が区間 $[\mu, L]$ に含まれることが直ちに従います。

(証明終わり)

</details>

以上により、$\mu$-強凸性と $L$-平滑性が、それぞれヘッセ行列の固有値の下界と上界に対応することが分かります。
ヘッセ行列の固有値はアルゴリズムの収束性能を大きく決定づけることもあり、このようなバウンドによって、様々な収束解析に利用することができます。

#### Baillon–Haddadの定理

発展的な内容として、$L$-平滑性の有用な性質の一つである次の Baillon–Haddadの定理を紹介します。この定理では $C^2$ 級の仮定は不要で、$C^1$ 級の仮定だけで十分であることに注意してください。

**Proposition 7** (Baillon–Haddad theorem)

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であるとする。$f$ が $L$-平滑かつ凸であれば、任意の $x,y \in \mathbb{R}^n$ に対して $\nabla f$ は $1/L$-cocoercive である、すなわち

```math
\begin{equation*}
(\nabla f(x)-\nabla f(y))^\top (x-y) \ge \frac{1}{L} \left\lVert \nabla f(x)-\nabla f(y) \right\rVert^2
\end{equation*}
```

が任意の $x,y \in \mathbb{R}^n$ について成り立つ。

<details><summary>証明</summary>

先ほど示した凸かつ $L$-平滑な関数の下界より、次が成り立ちます。

```math
\begin{equation*}
\begin{cases}
f(y) \ge f(x) + \nabla f(x)^\top (y-x) + \frac{1}{2L} \left\lVert \nabla f(y) - \nabla f(x) \right\rVert^2, \\
f(x) \ge f(y) + \nabla f(y)^\top (x-y) + \frac{1}{2L} \left\lVert \nabla f(x) - \nabla f(y) \right\rVert^2.
\end{cases}
\end{equation*}
```

これらを足し合わせると、Baillon–Haddadの定理が得られます。

(証明終わり)

</details>

証明は他の文献も参照してください[^bauschkeBaillonHaddadTheoremRevisited2009][^rockafellarVariationalAnalysis1998] (Proposition 12.60)。

この定理は、準ニュートン法の一種であるBFGS法やL-BFGS法などの更新則の解析で有用な次の不等式を導く際に利用されることがあります。

最適化アルゴリズムが生成する列 $\lbrace x_k \rbrace_k$ に対し、

```math
\begin{equation*}
s_k \mathrel{\vcenter{:}}= x_{k+1}-x_k, \quad y_k \mathrel{\vcenter{:}}= \nabla f(x_{k+1}) - \nabla f(x_k)
\end{equation*}
```

と定義した上で、Baillon–Haddadの定理を $x=x_{k+1}$、$y=x_k$ に適用すると、次の結果が得られます。

```math
\begin{equation*}
s_k^\top y_k \ge \frac{1}{L} \left\lVert y_k \right\rVert^2.
\end{equation*}
```

この不等式は、準ニュートン法における近似ヘッセ行列の正定値性が保たれるための条件の一つとして利用され、理論保証の上で重要な役割を果たします。

<!-- From 2_newton.tex -->

## ニュートン法

目的関数を $C^2$ 級の関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ とし、また $n$ を決定変数の次元と定めます。
この時、次の無制約最適化問題は、最も基本的な最適化問題の一つです。

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\text{minimize}} \quad f(x).
\end{equation*}
```

本節では、無制約最適化問題に対する基本的な最適化アルゴリズムである、ニュートン法の概要を示します。

### ニュートン法のアルゴリズム

ニュートン法は、初期点 $x_0$ から出発し、現在の点を逐次更新して点列 $\lbrace x_k \rbrace_k$ を生成する代表的な反復アルゴリズムです。

$f$ は $C^2$ 級かつ強凸であると仮定すると、ヘッセ行列 $\nabla^2 f(x)$ は任意の $x \in \mathbb{R}^n$ で正定値となり可逆です。
$k$ 反復目の点 $x_k$ における勾配 $g_k \mathrel{\vcenter{:}}= \nabla f(x_k)$ とヘッセ行列 $\nabla^2 f(x_k)$ を用いて、ニュートン法の1ステップは次のように表せます。

```math
\begin{equation*}
x_{k+1} \gets x_k - \alpha_k \nabla^2 f(x_k)^{-1} g_k.
\end{equation*}
```

ここで $\alpha_k > 0$ は直線探索によって定められるステップサイズです。この更新則は次のように導出できます。まず、$x_k$ での $f$ の二次近似モデルはテイラー展開により次のように与えられます。

```math
\begin{equation*}
m_{k}^\ast(x) \mathrel{\vcenter{:}}= f(x_k) + g_k^\top (x - x_k) + \frac{1}{2} (x - x_k)^\top \nabla^2 f(x_k) (x - x_k).
\end{equation*}
```

このモデルの勾配は次のように書けます。

```math
\begin{equation*}
\nabla m_k^\ast(x) = g_k + \nabla^2 f(x_k)(x - x_k).
\end{equation*}
```

仮定よりこの二次モデルは強凸であるため、点 $x^\ast \in \mathbb{R}^n$ が $m_k^\ast$ の最小点であることと $\nabla m_k^\ast(x) = 0$ が成り立つことは同値です。従って、解は以下の通りです。

```math
\begin{equation*}
\underset{x \in \mathbb{R}^n}{\mathrm{arg \ min}} \ m_k^\ast(x) = x_k -\left(\nabla^2 f(x_k)\right)^{-1} g_k.
\end{equation*}
```

次の反復点として、直接 $x^\ast$ を選ぶことは自然に思えますが、後述するようにこのままでは大域収束性を保証できません。そこで、関数値の十分な減少を確保するため、直線探索で定めるステップサイズ $\alpha_k > 0$ を導入すると、最初に示した更新則が得られます。これがニュートン法の基本的なアルゴリズムとなります。

<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_opt_step0.png" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_opt_step1.png" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_opt_step2.png" />

(Fig. 4 ニュートン法の反復の様子。この例では、全てのステップで $\alpha_k=1$、つまり $x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} g_k$ が成り立つ。)

先ほどの更新方向 $d_k \mathrel{\vcenter{:}}= -\nabla^2 f(x_k)^{-1} g_k$ はニュートン方向と呼ばれています。ヘッセ行列が正定値であれば、次式より降下方向、つまり目的関数が減少していく方向であることが分かります。

```math
\begin{equation*}
g_k^\top d_k = -g_k^\top \nabla^2 f(x_k)^{-1} g_k < 0.
\end{equation*}
```

$f$ が非凸である、つまりヘッセ行列 $\nabla^2 f(x_k)$ が負定値または不定値である場合、$d_k$ は降下方向であるとは限らず、ニュートン法は収束しない可能性があります。このような場合への一つの対処法として修正ニュートン法[^nocedal1999numerical] (Sec. 3.4) が知られていますが、ここでは説明を省略します。

### ニュートン法の収束性

次に、ニュートン法に対する標準的な収束性に関する結果を示します。

#### 大域収束

まず、ニュートン法に限らない直線探索付きの手法に対する、一般的な大域収束性の結果

```math
\begin{equation*}
\lim_{k \to \infty} \left\lVert g_k \right\rVert = 0
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
ステップサイズ $\alpha_k$ は (弱)Wolfe 条件[^nocedal1999numerical] (Sec. 3.1) により次のように定められるとします。

```math
\begin{equation*}
\begin{cases}
f(x_k + \alpha_k d_k) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k, \\
-g_{k+1}^\top d_k     & \leq -c_2 g_k^\top d_k.
\end{cases}
\end{equation*}
```

$0 < c_1 < c_2 < 1$ は定数です。
一つ目の条件は、Armijo 条件とも呼ばれ、次の点での関数値 $f(x_k + \alpha_k d_k)$ が、現在の点での関数値 $f(x_k)$ よりも、勾配から導かれる予測値 $\alpha_k g_k^\top d_k$ の少なくとも $c_1$ 倍した値は減少していることを要求します。
二つ目の条件は、曲率条件 (curvature condition) とも呼ばれます。点 $x_{k+1}$ でも $d_k$ が降下方向ならば、$-g_{k+1}^\top d_k$ は正となり、$d_k$ 方向に微小でも進めば目的関数値は減少します。曲率条件を満たすとは、この減少率が大きすぎないこと、つまり $d_k$ 上で十分なステップサイズがとられていることを要求します。弱Wolfe条件における曲率条件では、逆にステップサイズが長すぎるということもあり得ますが、強Wolfe条件においては絶対値を取ることで、適切なステップサイズがとられていることを要求します。
Wolfe条件とは、Armijo 条件と曲率条件の両方が同時に満たされることを指します。
Fig. 5 にこれらの条件が満たされる範囲を図示しています。

![../imgs/quasi_newton/armijo_wolfe_conditions.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/armijo_wolfe_conditions.png?v=1)

(Fig. 5 Armijo 条件と曲率条件のイラスト。各矢印は、それぞれの条件が満たされる範囲を示しています。)

また、方向 $d_k$ と負の勾配 $-g_k$ のなす角を $\theta_k \in [0, \pi]$ とし、次式を満たすように定めます。

```math
\begin{equation*}
\cos \theta_k = \frac{-g_k^\top d_k}{\left\lVert g_k \right\rVert\left\lVert d_k \right\rVert}.
\end{equation*}
```

これらの設定の下で、次の古典的に良く知られた結果の簡略版を以下に示します。

**Theorem 1** ([^nocedal1999numerical] (Theorem 3.2))

関数 $f \colon \mathbb{R}^n \to \mathbb{R}$ が $C^1$ 級であり $\mathbb{R}^n$ 上で下に有界で、かつ $L$-平滑だとする。次の反復法を考える。

```math
\begin{equation*}
x_{k+1} \gets x_k + \alpha_k d_k.
\end{equation*}
```

初期点 $x_0 \in \mathbb{R}^n$ から開始され、ステップサイズ $\alpha_k$ は Wolfe 条件を満たすとする。角 $\lbrace \theta_k \rbrace_k$ に対して、ある正の定数 $\delta$ が存在して $\cos \theta_k \geq \delta > 0$ が全ての $k$ で成り立つならば、この反復法は次を満たす点列 $\lbrace x_k \rbrace_k$ を生成する。

```math
\begin{equation*}
\lim_{k \to \infty} \left\lVert g_k \right\rVert = 0.
\end{equation*}
```

<details><summary>証明</summary>

Wolfe 条件、Cauchy–Schwarz の不等式、$f$ の $L$-平滑性より、次の二つの関係式が成り立ちます。

```math
\begin{equation*}
\begin{cases}
(g_{k+1} - g_k)^\top d_k = g_{k+1}^\top d_k - g_k^\top d_k
\geq (c_2 - 1) g_k^\top d_k, \\
(g_{k+1} - g_k)^\top d_k \leq
\left\lVert g_{k+1} - g_k \right\rVert \left\lVert d_k \right\rVert \leq L \left\lVert x_{k+1} - x_k \right\rVert \left\lVert d_k \right\rVert  = \alpha_k L \left\lVert d_k \right\rVert^2.
\end{cases}
\end{equation*}
```

これらを組み合わせると次の不等式が得られます。

```math
\begin{equation*}
\alpha_k \geq \frac{c_2 - 1}{L} \frac{g_k^\top d_k}{\left\lVert d_k \right\rVert^2}.
\end{equation*}
```

よって、

```math
\begin{align*}
f(x_{k+1}) & \leq f(x_k) + c_1 \alpha_k g_k^\top d_k                                                     &  & (\text{Armijo condition})         \\
& \leq f(x_k) - c_1 \frac{1 - c_2}{L} \frac{(g_k^\top d_k)^2}{\left\lVert d_k \right\rVert^2} &  & (\text{previous inequality})      \\
& = f(x_k) - c_1 \frac{1 - c_2}{L} \cos^2 \theta_k \left\lVert g_k \right\rVert^2.            &  & (\text{definition of $\theta_k$})
\end{align*}
```

この不等式を繰り返し適用することで次が得られます。

```math
\begin{equation*}
f(x_{k+1}) \leq f(x_0) - c_1 \frac{1 - c_2}{L} \sum_{j=0}^k \cos^2 \theta_j \left\lVert g_j \right\rVert^2.
\end{equation*}
```

仮定より $f$ は下に有界なので、ある定数 $f^\ast$ が存在して、全ての $k$ について $f(x_{k+1}) \geq f^\ast$ となります。
従って、Zoutendijk 条件と呼ばれる次式が得られます。

```math
\begin{equation*}
\sum_{k=0}^\infty \cos^2 \theta_k \left\lVert g_k \right\rVert^2 \leq \frac{L}{c_1 (1 - c_2)} (f(x_0) - f^\ast) < \infty.
\end{equation*}
```

この条件より、次が従います。

```math
\begin{equation*}
\cos^2 \theta_k \left\lVert g_k \right\rVert^2 \to 0.
\end{equation*}
```

$\cos \theta_k \geq \delta > 0$ がすべての $k$ で成り立つという仮定と合わせると、直ちに次が従います。

```math
\begin{equation*}
\lim_{k \to \infty} \left\lVert g_k \right\rVert = 0.
\end{equation*}
```

よって、Wolfe条件による直線探索付きの一般的な反復法に対する大域収束性が示されました。

(証明終わり)

</details>

ニュートン法は $d_k = -\nabla^2 f(x_k)^{-1} g_k$ とした Theorem 1 の特別な場合なので、適切な $f$ に関する条件の下でこの結果を適用できます。
特に $f$ が $L$-平滑かつ $\mu$-強凸であれば、任意の $k$ について次が成り立ちます。

```math
\begin{equation*}
\cos \theta_k
= \frac{g_k^\top \nabla^2 f(x_k)^{-1} g_k}{\left\lVert g_k \right\rVert\left\lVert \nabla^2 f(x_k)^{-1} g_k \right\rVert}
\geq \frac{\left\lVert g_k \right\rVert^2 \lambda_{\min}(\nabla^2 f(x_k)^{-1})}{\left\lVert g_k \right\rVert^2 \lambda_{\max}(\nabla^2 f(x_k)^{-1})}
\geq \frac{\mu}{L}.
\end{equation*}
```

ここで $\lambda_{\min}(\cdot)$ と $\lambda_{\max}(\cdot)$ はそれぞれ最小固有値と最大固有値を表すとします。
従って、Theorem 1 において $\delta = \mu / L$ とすれば、Wolfe 条件を満たす直線探索の下で、強凸かつ平滑な関数に対するニュートン法の大域収束性が得られます。

#### 局所二次収束

次に、ニュートン法の局所収束性に関する古典的結果を示します。
ここでも簡潔さのために簡略版を示します。

**Theorem 2** ([^nocedal1999numerical] (Theorem 3.5))

ヘッセ行列 $\nabla^2 f(x)$ が解 $x^\ast$ の近傍で Lipschitz連続であり、最適性の二次の十分条件が成り立つとする。すなわち $\nabla f(x^\ast)=0$ かつ $\nabla^2 f(x^\ast)$ は正定値である。
前節までのニュートン法において、すべての $k$ で $\alpha_k=1$ が満たされるとし、初期点 $x_0$ が $x^\ast$ に十分近いとき、勾配ノルム列 $\lbrace\left\lVert \nabla f(x_k) \right\rVert\rbrace_k$ は二次収束する。

<details><summary>証明</summary>

$k=0$ とし、$x_k$ から $x_{k+1}$ への更新を考える。
ヘッセ行列 $\nabla^2 f(x_k)$ は $x^\ast$ の十分小さい近傍で可逆であるため、最適性条件 $\nabla f(x^\ast)=0$ と $\alpha_k=1$ より次が成り立つ。

```math
\begin{align*}
x_{k+1} - x^\ast
& = x_k - x^\ast - \left(\nabla^2 f(x_k)\right)^{-1}\nabla f(x_k)                                                        \\
& = \left(\nabla^2 f(x_k)\right)^{-1} \left(\nabla^2 f(x_k)(x_k-x^\ast)-\left(\nabla f(x_k)-\nabla f(x^\ast)\right)\right).
\end{align*}
```

テイラーの定理と三角不等式より次が成り立つ。

```math
\begin{align*}
& \left\lVert \nabla^2 f(x_k)(x_k-x^\ast)-\left(\nabla f(x_k)-\nabla f(x^\ast)\right) \right\rVert                               \\
=   {}  & \left\lVert \nabla^2 f(x_k)(x_k-x^\ast)-\int_0^1 \nabla^2 f(x_k+t(x^\ast-x_k)) (x_k - x^\ast)\mathrm{d}t \right\rVert             \\
\leq {} & \int_0^1 \left\lVert \nabla^2 f(x_k)-\nabla^2 f(x_k+t(x^\ast-x_k)) \right\rVert\left\lVert x_k-x^\ast \right\rVert\mathrm{d}t.
\end{align*}
```

$\nabla^2 f$ が定数 $L^{\mathrm{H}}$ を持つ Lipschitz 連続であれば、被積分関数は $L^{\mathrm{H}}t\left\lVert x_k-x^\ast \right\rVert$ で上から抑えられ、積分により次を得る。

```math
\begin{equation*}
\left\lVert \nabla^2 f(x_k)(x_k-x^\ast)-\left(\nabla f(x_k)-\nabla f(x^\ast)\right) \right\rVert
\le \frac{1}{2}L^{\mathrm{H}}\left\lVert x_k-x^\ast \right\rVert^2.
\end{equation*}
```

$\nabla^2 f(x^\ast)$ は正定値なので、ある半径 $r>0$ が存在して $\left\lVert x-x^\ast \right\rVert\le r$ を満たすすべての $x$ について

```math
\begin{equation*}
\left\lVert \left(\nabla^2 f(x)\right)^{-1} \right\rVert\le 2 \left\lVert \left(\nabla^2 f(x^\ast)\right)^{-1} \right\rVert
\end{equation*}
```

が成り立つ。
これらを合わせると次を得る。

```math
\begin{align*}
\left\lVert x_{k+1} - x^\ast \right\rVert
& \leq \left\lVert \left(\nabla^2 f(x_k)\right)^{-1} \right\rVert \left\lVert \nabla^2 f(x_k)(x_k-x^\ast)-\left(\nabla f(x_k)-\nabla f(x^\ast)\right) \right\rVert \\
& \le L^{\mathrm{H}} \left\lVert \left(\nabla^2 f(x^\ast)\right)^{-1} \right\rVert \left\lVert x_k-x^\ast \right\rVert^2.
\end{align*}
```

$\widetilde L \mathrel{\vcenter{:}}= L^{\mathrm{H}}\left\lVert \left(\nabla^2 f(x^\ast)\right)^{-1} \right\rVert$ とおく。初期点が $\left\lVert x_0-x^\ast \right\rVert\le \min\lbrace r, 1/(2\widetilde L) \rbrace$ を満たすように選ばれていれば、帰納法により $\lbrace x_k \rbrace_k$ は近傍内に留まり $x^\ast$ に収束する。よって、上の誤差評価は $\lbrace x_k \rbrace_k$ の二次収束を示す。

勾配ノルムの二次収束を示すために、$x_{k+1}-x_k=-\left(\nabla^2 f(x_k)\right)^{-1}\nabla f(x_k)$ と
$\nabla f(x_k)+\nabla^2 f(x_k)(x_{k+1}-x_k)=0$ を用いると次を得る。

```math
\begin{align*}
\left\lVert \nabla f(x_{k+1}) \right\rVert
& = \left\lVert \nabla f(x_{k+1})-\nabla f(x_k)-\nabla^2 f(x_k)(x_{k+1}-x_k) \right\rVert                                              \\
& \le \int_0^1 \left\lVert \nabla^2 f(x_k+t (x_{k+1}-x_k))-\nabla^2 f(x_k) \right\rVert\left\lVert x_{k+1}-x_k \right\rVert\mathrm{d}t \\
& \le \frac{1}{2}L^{\mathrm{H}}\left\lVert x_{k+1}-x_k \right\rVert^2                                                                  \\
& \le \frac{1}{2}L^{\mathrm{H}}\left\lVert \left(\nabla^2 f(x_k)\right)^{-1} \right\rVert^2\left\lVert \nabla f(x_k) \right\rVert^2    \\
& \le 2 L^{\mathrm{H}} \left\lVert \left(\nabla^2 f(x^\ast)\right)^{-1} \right\rVert^2\left\lVert \nabla f(x_k) \right\rVert^2.
\end{align*}
```

よって $\left\lVert \nabla f(x_k) \right\rVert$ は0へと局所二次収束する。

(証明終わり)

</details>

これらのTheorem 1, Theorem 2は、直線探索を伴うニュートン法が適切な条件の下で大域収束性と局所二次収束性の両方を持つことを示しています。この高速な収束性は、通常は線形収束にとどまる勾配降下法などの一階法と比べたときの大きな利点となります。

### 大域収束のために必要な要素

先ほど述べたように、ヘッセ行列の正定値性と直線探索によるステップサイズの選択はニュートン法で重要な役割をそれぞれ担っています。本小節では、これらが必要となる理由について説明します。

#### ヘッセ行列の正定値性

ここまで $f$ が強凸であると仮定してきましたが、これは各反復でヘッセ行列 $\nabla^2 f(x_k)$ が正定値であることを意味しています。この仮定は、ニュートン法が局所的に最適解へ収束するために重要です。ヘッセ行列が正定値であれば、次式よりニュートン法は降下方向を与えます。

```math
\begin{equation*}
\nabla f(x_k)^\top d_k
= -\nabla f(x_k)^\top \left(\nabla^2 f(x_k)\right)^{-1} \nabla f(x_k)
< 0.
\end{equation*}
```

一方、ヘッセ行列が負定値または不定値の場合、関数値の減少は保証されず、ニュートン法が最適解から離れてしまう可能性があります。不定値の場合は関数値が減少することもありますが、鞍点へ収束するリスクも高まります。従って、ニュートン法を適用する際にはヘッセ行列の正定値性が重要となってきます。

#### フルステップの問題

ニュートン法では、各反復でステップサイズを $\alpha_k=1$ とすると問題を引き起こす場合があります。ここではそのような失敗の具体例を挙げ、対処として直線探索が必要となることを述べます。

##### ニュートン法が発散する例

次の関数を考えます。

```math
\begin{align*}
f(x)   & \mathrel{\vcenter{:}}= \sqrt{1 + x^2},    \\
f'(x)  & = \frac{x}{\sqrt{1 + x^2}},  \\
f''(x) & = \frac{1}{(1 + x^2)^{3/2}}.
\end{align*}
```

初期点の絶対値が 1 を超えると、Fig. 6 に示すようにこの関数に対するニュートン法は発散します。

これは、最適解 $x^\ast=0$ から離れた点では、2階微分の値、つまりヘッセ行列の固有値が非常に小さくなり、ニュートンステップが過大になる為です。反復を重ねるごとに、より遠くへ飛んでいき、最適解から離れてしまいます。

![../imgs/quasi_newton/newton_failure_sqrt_function_1.1.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_sqrt_function_1.1.png?v=1)

(Fig. 6 初期点 $x_0=1.1$ でニュートン法が発散する例)

##### 強凸関数に対するニュートン法の振動

前の例では目的関数は強凸ではなく、必ずしも良い性質を持っている訳ではありませんでした。しかし、強凸性を持つ関数であってもニュートン法が収束しない例が知られています[^Doikov2021SecondOrderTensor] (Example 1.4.3)。

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

それにもかかわらず、初期点 $x_0$ が $\mu$ に対して十分大きい場合、Fig. 7 に示すようにニュートン法は振動します。

<img width="50%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_strongly_convex_function_0.1_-4.png?v=1" /><img width="50%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_failure_strongly_convex_function_0.01_-4.png?v=1" />

(Fig. 7 (左) $x_0=-4, \ \mu=0.1$ ではニュートン法が収束する。(右) $x_0=-4, \ \mu=0.01$ ではニュートン法が振動する。)

##### 直線探索の必要性

これらの例から、ニュートン法でフルステップ $\alpha_k=1$ を採用すると、強凸かつ平滑な関数であっても発散や振動を引き起こす可能性があることが分かりました。従って、収束を保証するためにステップサイズ $\alpha_k$ を適切に選択することが不可欠です。このように直線探索を伴うニュートン法は、修正ニュートン法とも呼ばれています。

### 根探索としてのニュートン法との比較

概念的な補足として、「最適化におけるニュートン法」と「根探索としてのニュートン法」の関係を簡潔に整理します。両者は密接に関係した手法ですが、考え方や適用対象は異なります。

![../imgs/quasi_newton/newton_raphson.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_raphson.png?v=1)

(Fig. 8 関数 $f(x)=x^3 - 2 x^2 + x$ に対する最適化と勾配 $\nabla f(x)=3 x^2 - 4 x + 1$ に対する根探索。)

#### 最適化としてのニュートン法

最適化文脈におけるニュートン法は、二回微分可能な関数 $f\colon \mathbb R^{n} \to\mathbb{R}$ の局所最小点を求めるアルゴリズムを指すことが一般的です。

ニュートン法の更新は $\alpha_k=1$ とすると、次のように与えられていました。

```math
\begin{equation*}
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
\end{equation*}
```

スカラーの場合、上の式は $x_{k+1}=x_k - \frac{f'(x_k)}{f''(x_k)}$ に簡約されます。

#### 根探索としてのニュートン法

ニュートン法(あるいはニュートン・ラフソン法)と言うと、微分可能なスカラー関数 $g\colon \mathbb{R} \to \mathbb{R}$ に対して $g(x) = 0$ を解く根探索アルゴリズムを指すこともあります。この手法も、基本的でよく知られた手法です。

初期値 $x_0$ から開始するとして、この手法の反復は次式で与えられます。

```math
\begin{equation*}
x_{k+1} = x_k - \frac{g(x_k)}{g'(x_k)}.
\end{equation*}
```

スカラーの場合においては、$g(x_k)=f'(x_k)$ なことに注意すると、先ほどの最適化としてのニュートン法の更新と同じ式であることが分かります。

幾何学的には、$x_k$ 近傍での $g$ のグラフを接線で近似し、その接線と $x$ 軸の交点を次の近似解として選ぶことに、この式は対応しています。

#### 二つの定式化の関係

具体例を通じて両者の等価性を確認してみます。Fig. 8は単純な三次関数 $f(x)$ を用いて両者の関係を示したものです。左図は $f(x)$ に対する最適化であり、右図は $g(x) = \nabla f(x)$ に対する根探索となっています。

最適化の定式化では、$f(x) = x^3 - 2x^2 + x$ を二次のテイラー展開で近似し、

```math
\begin{equation*}
m_k^\ast(x) = f(x_k) + \nabla f(x_k)(x - x_k) + \frac{1}{2}\nabla^2 f(x_k)(x - x_k)^2
\end{equation*}
```

として、この二次モデルの最小解を $x_{k+1}$ として取ります。

根探索の定式化では、$\nabla f(x) = 3x^2 - 4x + 1$ を接線で近似します。すなわち、

```math
\begin{equation*}
\nabla m_k^\ast(x) = \nabla f(x_k) + \nabla^2 f(x_k)(x - x_k)
\end{equation*}
```

として、この線形モデルの根を $x_{k+1}$ として選びます。

従って、正定値性の仮定の下では両者が本質的に等価な操作を行うことが分かり、その対応関係が見てとれます。

<!-- From 3_1_quasi_newton.tex -->

## 準ニュートン法

![../imgs/quasi_newton/newton_vs_qs_vs_gd.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/newton_vs_qs_vs_gd.png)

(Fig. 9 ニュートン法、準ニュートン法、勾配降下法の比較。)

本節では準ニュートン法について説明します。準ニュートン法はニュートン法に基づいていますが、その主要な欠点であるヘッセ行列やその逆行列の計算コストを低減することを目的としています。具体的には、真のヘッセ行列 $\nabla^2 f(x_k)$ の代わりに近似行列 $B_k$ とその逆行列 $H_k$ を用いることで、高速な収束性を保ちつつ計算コストを抑えます。

Fig. 9 はニュートン法、準ニュートン法、勾配降下法の比較を示しています。勾配降下法は勾配の反対方向に単純に更新する方法です。1反復あたりの計算コストは最小ですが、収束は遅くなります。ニュートン法は反復回数が最も少ない一方で、1ステップあたりの計算コストが最も高いです。準ニュートン法はその中間に位置し、両者のバランスを取っています。特に、反復回数ではなく実際の計算時間で比較すると、準ニュートン法が最良の性能を示すことが多いです。

直線探索に基づく準ニュートン法では、最適解 $x^\ast$ に収束する列 $\lbrace x_k \rbrace_k$ を次のように生成します。

```math
\begin{equation*}
x_{k+1} = x_k - \alpha_k B_k^{-1} \nabla f(x_k)
= x_k - \alpha_k H_k \nabla f(x_k).
\end{equation*}
```

ここで $\alpha_k > 0$ は直線探索で定めるステップサイズであり、$B_k$ は点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似です。$H_k \mathrel{\vcenter{:}}= B_k^{-1}$ はその逆行列を表します。この更新則はニュートン法の更新則と酷似しており、その故にこの手法は準ニュートン法と呼ばれています。

<img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_1.png" /><img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_2.png" /><img width="33%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/quasi_newton_3.png" />

(Fig. 10 準ニュートン法の概念図。 (1) 目的関数 $f$ (青い曲面) と現在点 $x_k$ (赤点) が与えられます。 (2) 現在の近似ヘッセ行列によって得られる二次モデル $m_k(x)$ (橙色の曲面) を最小化し、その最小点 $x_{k+1}$ (黄色のバツ印) に移動します。 (3) 二次モデルを更新し (緑色の曲面)、この操作を繰り返します。)

準ニュートン法の核心は、各反復で $B_k$ をどのように更新して真のヘッセ行列に近づけるかにあります。Fig. 10 は準ニュートン法の概念図です。まず現在の点 $x_k$ の周りで、$B_k$ を用いて目的関数 $f$ の二次近似モデルを構成します。次にこの二次モデルを最小化して次の点 $x_{k+1}$ を得ます。$x_{k+1}$ を得た後、$x_k$ と $x_{k+1}$ における勾配情報を用いて近似行列 $B_k$ を $B_{k+1}$ に更新します。この手続きを収束するまで繰り返すことが準ニュートン法です。

以下のGIFも参照して下さい。$f(x)$ の最適化において、各step毎にモデル関数が更新されていく様子を示しています。

![GIF illustrating the concept](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/sixhump.gif)

### セカント条件

関数 $f\colon \mathbb{R}^n \to \mathbb{R}$ を $C^2$ 級とします。対称行列 $B_k$ が点 $x_k$ におけるヘッセ行列 $\nabla^2 f(x_k)$ の近似として与えられたとき、次の点 $x_{k+1}$ における近似ヘッセ行列 $B_{k+1}$ への更新を考えます。このような行列の候補は無数に存在しますが、真のヘッセ行列が対称であることから、$B_{k+1}$ にも対称性を課すのが自然です。現在の点 $x_k$ と次の点 $x_{k+1}$ に対して、ステップと勾配差を次のように定義します。

```math
\begin{equation*}
s_k \mathrel{\vcenter{:}}= x_{k+1} - x_k, \qquad   y_k \mathrel{\vcenter{:}}= \nabla f(x_{k+1}) - \nabla f(x_k).
\end{equation*}
```

$y_k^\top s_k \neq 0$ かつ $s_k \neq 0$ を仮定します。なお、$s_k$ の $s$ は step を意味します。新しい近似は $x_{k+1}$ を中心に構築されるため、$\nabla f(x_k)$ をヘッセ行列 $\nabla^2 f(x_{k+1})$ およびその近似である $B_{k+1}$ を用いて次のように近似します。

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

<!-- From 3_2_quasi_newton_update.tex -->

### 代表的な準ニュートン更新則

$B_k$, $s_k$, $y_k$ が与えられたとき、セカント条件を満たす $B_{k+1}$ を与える更新則は多数存在します。ここでは代表的なものをその導出とともに示します[^dennisjr.QuasiNewtonMethodsMotivation1977a]。本小節に限り、簡潔さのため $B_k$, $B_{k+1}$, $s_k$, $y_k$ をそれぞれ $B$, $\bar{B}$, $s$, $y$ と略記します。

#### Broyden更新

[Broyden更新](https://en.wikipedia.org/wiki/Broyden%27s_method)は最も古くから知られる更新則の一つですが、準ニュートン法の文脈では、対称性を保たないため実用上はあまり用いられません。歴史的な観点もふまえ、簡潔に紹介します。更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{Broyden}} \mathrel{\vcenter{:}}= B + \frac{(y - Bs)s^\top}{s^\top s}.
\end{equation*}
```

##### 導出 (Broyden)

単純な構造的仮定からこの公式を導出します[^dennisjr.QuasiNewtonMethodsMotivation1977a] (Section 4)。

**Proposition 8**

$\bar{B}$ がセカント条件

```math
\begin{equation*}
\bar{B}s = y,
\end{equation*}
```

と制約

```math
\begin{equation*}
\bar{B}z = Bz
\quad\text{for all } z\in\mathbb{R}^n \text{ such that } z^\top s = 0.
\end{equation*}
```

を満たすと仮定します。このとき $\bar{B}$ は一意に定まり、$\bar{B}_{\mathrm{Broyden}}$ に一致します。

<details><summary>証明</summary>

ベクトル $s$ と $s$ の直交補空間の基底は $\mathbb{R}^n$ の基底を成します。$\bar{B}$ の条件はこの基底に対する $\bar{B}$ の作用を完全に決定するため、$\bar{B}$ は一意に定まります。

ここで $\bar{B}_{\mathrm{Broyden}}$ が課された条件を満たすことを示します。$z^\top s = 0$ を満たす任意のベクトル $z$ を取ります。このとき、次を得ます。

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

したがって、$\bar{B}_{\mathrm{Broyden}}$ は課された条件を満たします。よって、一意性より $\bar{B} = \bar{B}_{\mathrm{Broyden}}$ となります。

(証明終わり)

</details>

Broyden更新は、フロベニウスノルムにおける最小変化更新としても特徴づけられます。

**Proposition 9** ([^dennisjr.QuasiNewtonMethodsMotivation1977a] (Theorem 4.1))

$B\in\mathbb{R}^{n\times n}$, $y\in\mathbb{R}^n$, $s\in\mathbb{R}^n\setminus\lbrace 0 \rbrace$ が与えられているとき、行列 $\bar{B}_{\mathrm{Broyden}}$ は以下の最適化問題の一意解である。

```math
\begin{align*}
\underset{\tilde{B} \in \mathbb{R}^{n \times n}}{\mathrm{minimize}} & \quad \left\lVert \tilde{B} - B \right\rVert_F \\
\mathrm{subject to}                                                 & \quad \tilde{B} s = y.
\end{align*}
```

<details><summary>証明</summary>

関数 $\tilde{B}\mapsto\left\lVert \tilde{B}-B \right\rVert^2_F$ を最適化すると考えてよく、これは $\mathbb{R}^{n\times n}$ 上で凸です。制約集合

```math
\begin{equation*}
\lbrace\tilde{B}\in\mathbb{R}^{n\times n}:\tilde{B}s=y\rbrace
\end{equation*}
```

はアフィン集合であり凸です。よって、この最適化問題は高々一つの最小解しか持ちません。

$\bar{B}_{\mathrm{Broyden}}$ が実際に最小解であることを示します。制約 $\tilde{B}s=y$ を満たす任意の $\tilde{B}$ に対して

```math
\begin{equation*}
\left\lVert \bar{B}_{\mathrm{Broyden}} - B \right\rVert_F^2
= \left\lVert (\tilde{B}-B) \frac{s s^\top}{s^\top s} \right\rVert_F^2
\leq \left\lVert \tilde{B}-B \right\rVert_F^2.
\end{equation*}
```

ここでフロベニウスノルムの劣乗法性と $\left\lVert ss^\top/(s^\top s) \right\rVert_F=1$ を用いました。よって $\tilde{B}=\bar{B}_{\mathrm{Broyden}}$ となり、主張が得られます。

(証明終わり)

</details>

この更新則は、対称性を保たないという点に注意してください。以下では、その点を改善した代表的な準ニュートン更新則を紹介します。

#### Symmetric Rank-One (SR1) 更新

[対称ランク1 (SR1) 更新](https://en.wikipedia.org/wiki/Symmetric_rank-one)[^nocedal1999numerical] は更新過程で対称性を維持する基本的な準ニュートン法です。更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}} \mathrel{\vcenter{:}}= B + \frac{(y - B s)(y - B s)^\top}{(y - B s)^\top s}.
\end{equation*}
```

##### 導出 (SR1)

SR1更新を導出するため、更新行列 $\bar{B}$ をランク1更新として構成します。すなわち、あるベクトル $z \in \mathbb{R}^n$ に対して、次が満たされると仮定します。

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}} = B + z z^\top.
\end{equation*}
```

セカント条件 $\bar{B}_{\mathrm{SR1}} s = y$ を満たすためには、$z^\top s \neq 0$ のとき、次が必要となります。

```math
\begin{equation*}
B s + z z^\top s = y.
\end{equation*}
```

整理すると、

```math
\begin{equation*}
z = \frac{y - B s}{z^\top s}.
\end{equation*}
```

$z^\top s$ を決めるために $s$ との内積を取ると、

```math
\begin{equation*}
z^\top s = \frac{(y - B s)^\top s}{z^\top s}.
\end{equation*}
```

この式を整理すると次の関係が得られます。

```math
\begin{equation*}
(z^\top s)^2 = (y - B s)^\top s.
\end{equation*}
```

従って、

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}}
= B + z z^\top
= B + \frac{(y - B s)(y - B s)^\top}{(z^\top s)^2}
= B + \frac{(y - B s)(y - B s)^\top}{(y - B s)^\top s}.
\end{equation*}
```

よって、SR1更新が導出されました。

##### 補足

SR1更新は $(y - Bs)^\top s \neq 0$ であることを前提としています。$(y - Bs)^\top s = 0$ のとき分母が零となり、実用上は更新をスキップすることが多いです。

#### Powell Symmetric Broyden (PSB) 更新

Powell Symmetric Broyden (PSB) 更新[^m.j.d.powellNewAlgorithmUnconstrained1970][^doi:10.1137/1.9781611971200][^haeltermanAnalyticalStudyLeast2009] も準ニュートン更新則の一つで、更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - B s) s^\top + s (y - B s)^\top}{s^\top s} - \frac{s^\top (y - B s)}{(s^\top s)^2} s s^\top
\end{equation*}
```

##### 導出 (PSB)

SR1更新では $B$ に $z z^\top$ を加えることで $\bar{B}$ を得ていました。 代わりに、あるベクトル $c \in \mathbb{R}^n$ に対して、非対称なランク1更新 $z c^\top$ を考えることができます。そして最後に結果を対称化します。

$c^\top s \neq 0$ と仮定し、$Bs + z c^\top s = y$ から $z$ を

```math
\begin{equation*}
z = \frac{y - B s}{c^\top s}
\end{equation*}
```

と導き、これを用いて次の非対称更新を行います。

```math
\begin{equation*}
C_1 \mathrel{\vcenter{:}}= B + \frac{(y - B s)c^\top}{c^\top s}.
\end{equation*}
```

$C_1$ は一般に対称でないため、これを対称化します。

```math
\begin{equation*}
C_2 = \frac{C_1 + C_1^\top}{2}.
\end{equation*}
```

しかし、対称化された行列 $C_2$ はセカント条件 $C_2 s = y$ を満たさない場合があります。そこでこの過程を反復することを考えます。

```math
\begin{equation*}
\begin{cases}
C_0 = B                                                                               \\
C_{2t+1} = C_{2t} + \frac{(y - C_{2t}s)c^\top}{c^\top s} & (\text{asymmetric update}) \\
C_{2t+2} = \frac{C_{2t+1} + C_{2t+1}^\top}{2}            & (\text{symmetrization})
\end{cases}
\end{equation*}
```

重要な結果として、行列の列 $\lbrace C_t \rbrace_t$ はセカント条件を満たす対称行列に収束します。次の命題ではそれを示します。

**Proposition 10** ([^dennisjr.QuasiNewtonMethodsMotivation1977a] (Lemma 7.2))

行列の列 $\lbrace C_t \rbrace_t$ は収束し、その極限は次で与えられる。

```math
\begin{equation*}
\lim_{t \to \infty} C_{t}
=
C_{\infty}
\mathrel{\vcenter{:}}=
B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{c^\top s} - \frac{(y - Bs)^\top s}{(c^\top s)^2} c c^\top.
\end{equation*}
```

<details><summary>証明</summary>

まず偶数番目の部分列を解析します。 $t=0,1,2,\dots$ に対し、次のように定義します。

```math
\begin{equation*}
G_t \mathrel{\vcenter{:}}= C_{2t}.
\end{equation*}
```

構成方法より各 $G_t$ は対称です。定義から次を得ます。

```math
\begin{equation*}
G_{t+1} = G_t +\frac{1}{2c^\top s}\left((y-G_t s)c^\top+c(y-G_t s)^\top\right).
\end{equation*}
```

また、セカント条件に対する誤差ベクトルを次で導入します。

```math
\begin{equation*}
w_t \mathrel{\vcenter{:}}= y-G_t s.
\end{equation*}
```

このとき、

```math
\begin{equation*}
G_{t+1} = G_t+\frac{1}{2c^\top s}(w_t c^\top+cw_t^\top).
\end{equation*}
```

上式を $w_t$ の定義に代入すると

```math
\begin{align*}
w_{t+1} & = y-\left(G_t+\frac{1}{2c^\top s}(w_t c^\top+cw_t^\top)\right)s \\
& =
w_t-\frac12w_t-\frac{w_t^\top s}{2c^\top s}c                              \\
& =
\frac{1}{2}\left(w_t-\frac{w_t^\top s}{c^\top s}c\right).
\end{align*}
```

よって

```math
\begin{equation*}
w_{t+1}=Pw_t,
\qquad
P \mathrel{\vcenter{:}}= \frac{1}{2}\left(I-\frac{cs^\top}{c^\top s}\right).
\end{equation*}
```

行列 $cs^\top/c^\top s$ はランク1で固有値は $1,0,\dots,0$ です。よって $P$ は固有値 $0$ を一つ持ち、残りはすべて $1/2$ となります。特にスペクトル半径は $1/2<1$ であるので、次のように級数が収束します。

```math
\begin{align*}
\sum_{t=0}^{\infty}w_t & =        \sum_{t=0}^{\infty}P^t(y-Bs)                         &  & (w_0=y-Bs)                \\
& =  (I-P)^{-1}(y-Bs)                                                                          \\
& = 2\left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) (y-Bs). &  & (\text{definition of } P)
\end{align*}
```

最後の式は次から導かれることに注意してください。

```math
\begin{equation*}
2(I-P) \left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) = \left(I + \frac{cs^\top}{c^\top s}\right) \left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) = I.
\end{equation*}
```

よって

```math
\begin{align*}
\lim_{t\to\infty}G_t
& =
B+\frac{1}{2c^\top s}
\sum_{t=0}^{\infty}(w_t c^\top+c w_t^\top)                                                                                                                                        \\
& = B+ \left(\sum_{t=0}^{\infty}w_t\right) \frac{c^\top}{2c^\top s} + \frac{c}{2c^\top s} \left(\sum_{t=0}^{\infty}w_t\right)^\top                                               \\
& = B+ 2\left(I-\frac{1}{2}\frac{cs^\top}{c^\top s}\right) (y-Bs) \frac{c^\top}{2c^\top s} + \frac{c}{2c^\top s} 2(y-Bs)^\top \left(I-\frac{1}{2}\frac{sc^\top}{c^\top s}\right) \\
& = B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{c^\top s} - \frac{(y - Bs)^\top s}{(c^\top s)^2} c c^\top                                                                         \\
& = C_{\infty}.
\end{align*}
```

これで偶数番目の部分列 $\lbrace C_{2t} \rbrace_t$ が $C_{\infty}$ に収束することが示されました。

次に奇数番目の部分列について、次の式が成立しています。

```math
\begin{equation*}
C_{2t+1}
=
G_t+\frac{w_t c^\top}{c^\top s}.
\end{equation*}
```

$G_t\to C_{\infty}$ であり、また $\left\lVert w_t \right\rVert\to0$ が $P$ のスペクトル半径が1未満であることから従うため、

```math
\begin{equation*}
C_{2t+1} \to C_{\infty}.
\end{equation*}
```

従って、部分列 $\lbrace C_{2t} \rbrace_t$ と $\lbrace C_{2t+1} \rbrace_t$ はどちらも $C_{\infty}$ に収束し、証明が完了します。

(証明終わり)

</details>

$c = s$ のとき、$C_{\infty}$ の一般式は標準的な PSB 更新則を導きます。

```math
\begin{equation*}
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - Bs)s^\top + s(y - Bs)^\top}{s^\top s} - \frac{(y - Bs)^\top s}{(s^\top s)^2} ss^\top.
\end{equation*}
```

#### Davidon–Fletcher–Powell (DFP) 更新

[Davidon–Fletcher–Powell (DFP) 更新](https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula)[^nocedal1999numerical] も古典的な準ニュートン更新則です。
更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{DFP}} = \left(I - \frac{y s^\top}{y^\top s}\right) B \left(I - \frac{s y^\top}{y^\top s}\right) + \frac{y y^\top}{y^\top s}.
\end{equation*}
```

##### 導出 (DFP)

先に述べた PSB 更新では $c=s$ としましたが、別の $c$ を選ぶことも可能です。具体的には $c=y$ を代入すると次のように DFP更新が得られます。

```math
\begin{equation*}
\bar{B}_{\mathrm{DFP}} = B + \frac{(y - Bs)y^\top + y(y - Bs)^\top}{y^\top s} - \frac{(y - Bs)^\top s}{(y^\top s)^2} yy^\top.
\end{equation*}
```

なお、DFP更新には他にも導出方法が存在し、それらの一部は、次のBFGS更新における導出の双対版として理解することもできます。

#### Broyden–Fletcher–Goldfarb–Shanno (BFGS) 更新

[Broyden–Fletcher–Goldfarb–Shanno (BFGS) 更新](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)は最も広く使われる準ニュートン法の一つです。
更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{BFGS}} = B - \frac{B s s^\top B}{s^\top B s} + \frac{y y^\top}{y^\top s}
\end{equation*}
```

##### 双対による導出

この更新はDFP更新の双対を考えることで導出できます。後で示すように、$B$ に対するBFGS更新は $H$ に対する DFP更新と同じ形式となっています。

##### KLダイバージェンス最小化による導出

KLダイバージェンスとは、行列間の近さを測る一種の非対称な尺度です。正定値行列 $P,Q \in \mathbb{R}^{n \times n}$ に対して、次で定義されます。

```math
\begin{equation*}
\mathrm{KL}(P,Q) \mathrel{\vcenter{:}}= \mathrm{tr}(P Q^{-1}) - \log\det(P Q^{-1}) - n.
\end{equation*}
```

行列 $T \in \mathbb{R}^{n \times n}$ を任意の正則行列とします。$\mathrm{tr}$ は2つの行列の積に対して可換性を持つため、KLダイバージェンスは次の不変性を持ちます。

```math
\begin{align*}
\mathrm{KL}(T P T^\top, T Q T^\top) & = \mathrm{tr}(T P T^\top (T Q T^\top)^{-1}) - \log\det(T P T^\top (T Q T^\top)^{-1}) - n \\
& = \mathrm{tr}(P Q^{-1} T^{-1} T) - \log (\det(T) \det(P Q^{-1}) \det(T^{-1})) - n        \\
& = \mathrm{tr}(P Q^{-1}) - \log\det(P Q^{-1}) - n                                         \\
& = \mathrm{KL}(P, Q).
\end{align*}
```

ここで、BFGS更新は次のKLダイバージェンス最小化問題の解として得られることが知られています。

```math
\begin{align*}
\underset{\bar{B} \succ 0}{\mathrm{minimize}}
& \quad \mathrm{KL}(\bar{B}, B) \\
\mathrm{subject\ to}
& \quad \bar{B}s = y.
\end{align*}
```

**Proposition 11** ([^kanamori2016continuous] (Section 7.2.4))

$B \succ 0$ を正定値対称行列とする。このとき、上記の最適化問題の解はBFGS更新で与えられる。

<details><summary>証明</summary>

$B$ が正定値対称行列であることに注意して、まず、次のように定義します。

```math
\begin{equation*}
s' \mathrel{\vcenter{:}}= B^{\frac{1}{2}} s, \quad y' \mathrel{\vcenter{:}}= B^{-\frac{1}{2}} y, \quad B' \mathrel{\vcenter{:}}= B^{-\frac{1}{2}} \bar{B} B^{-\frac{1}{2}}.
\end{equation*}
```

すると、最適化問題は次のように書き直せます。

```math
\begin{align*}
\underset{B' \succ 0}{\text{minimize}} & \quad \mathrm{KL}(B', I) \\
\text{subject to}                      & \quad B' s' = y'.
\end{align*}
```

ただし、KLダイバージェンスの不変性より、次が成り立つことを用いました。

```math
\begin{equation*}
\mathrm{KL}(\bar{B}, B) = \mathrm{KL}(B^{-\frac{1}{2}} \bar{B} B^{-\frac{1}{2}}, B^{-\frac{1}{2}} B B^{-\frac{1}{2}}) = \mathrm{KL}(B', I)
\end{equation*}
```

なお、本最適化問題において $B' \succ 0$ という制約は自動的に対称性を含むため、$B' = B'^\top$ を制約として陽に課しても解は変わりません。

この問題の解を、ラグランジュの未定乗数法を用いて求めます。ラグランジュ乗数 $\lambda\in\mathbb{R}^n$ および $\Lambda\in\mathbb{R}^{n\times n}$ を用いて、ラグランジアンを次のように定義します。

```math
\begin{equation*}
\mathcal{L}(B',\lambda,\Lambda) = \mathrm{tr}(B') -\log\det(B') -n +2\lambda^\top(B's'-y') +\mathrm{tr}\left(\Lambda(B'-B'^\top)\right).
\end{equation*}
```

一般に、行列微分の公式として、$X \succ 0$ に対して、$\log\det(X)$ の微分が $X^{-\top}$ かつ、 $\mathrm{tr}(AX)$ の微分が $A^\top$ となります。KKT条件のうち停留性に関する条件は、$B'$ で微分し、$B'=B'^\top$ を用いることで次のように書けます。

```math
\begin{equation*}
I-B'^{-\top}
+2\lambda s'^\top
+\Lambda^\top-\Lambda
=0.
\end{equation*}
```

この式とその転置を加えて2で割り、再び $B' = B'^\top$ を用いると次を得る。

```math
\begin{equation*}
B'^{-1} = I+\lambda s'^\top+s'\lambda^\top.
\end{equation*}
```

制約条件 $B's'=y'$ より $B'^{-1}y'=s'$ であるから、上式に右から $y'$ を掛けた式と、さらに左から $y'^\top$ を掛けた式はそれぞれ、次のようになる。

```math
\begin{align*}
s'         & = y' +(s'^\top y')\lambda +(\lambda^\top y')s'                                \\
y'^\top s' & = y'^\top y' + (s'^\top y')(y'^\top \lambda) + (\lambda^\top y')(y'^\top s').
\end{align*}
```

第2式より

```math
\begin{equation*}
\lambda^\top y' = \frac{y'^\top s'-y'^\top y'}{2(s'^\top y')}.
\end{equation*}
```

これを第1式に代入すると、

```math
\begin{equation*}
\lambda = \frac{s'-y'}{s'^\top y'} - \frac{\lambda^\top y'}{s'^\top y'}s' = \frac{s'^\top y'+y'^\top y'}{2(s'^\top y')^2}s' -\frac{1}{s'^\top y'}y'.
\end{equation*}
```

この $\lambda$ を $B'^{-1}$ の表式に代入して整理すると、

```math
\begin{align*}
B'^{-1} & = I + 2\frac{s'^\top y'+y'^\top y'}{2(s'^\top y')^2}(s' s'^\top) - \frac{1}{s'^\top y'}(y' s'^\top + s' y'^\top)             \\
& = \left( I-\frac{s'y'^\top}{s'^\top y'} \right) \left( I-\frac{y's'^\top}{s'^\top y'} \right) +\frac{s's'^\top}{s'^\top y'}.
\end{align*}
```

最後に $B' = B^{-\frac{1}{2}} \bar{B} B^{-\frac{1}{2}}$ より、$\bar{B} = B^{\frac{1}{2}} B' B^{\frac{1}{2}}$ であるため、$\bar{B}^{-1}$ を計算すると、

```math
\begin{align*}
\bar{B}^{-1} & = B^{-\frac{1}{2}} B'^{-1} B^{-\frac{1}{2}}                                                                                                                                                                          \\
& = B^{-\frac{1}{2}} \left( I-\frac{s'y'^\top}{s'^\top y'} \right) \left( I-\frac{y's'^\top}{s'^\top y'} \right) B^{-\frac{1}{2}} + \frac{(B^{-\frac{1}{2}}s')(B^{-\frac{1}{2}}s')^\top}{s'^\top y'}                   \\
& = B^{-\frac{1}{2}} \left( I-\frac{B^{\frac{1}{2}}s y^\top B^{-\frac{1}{2}}}{s^\top y} \right) \left( I-\frac{B^{-\frac{1}{2}}y s^\top B^{\frac{1}{2}}}{s^\top y} \right) B^{-\frac{1}{2}} + \frac{ss^\top}{s^\top y} \\
& = \left( I - \frac{s y^\top}{y^\top s} \right) B^{-1} \left( I - \frac{y s^\top}{y^\top s} \right) + \frac{s s^\top}{y^\top s}.
\end{align*}
```

後述の通り、これは確かにBFGS更新の逆行列の公式であるため、証明が完了します。

(証明終わり)

</details>

更なる詳細については[^kanamoriBregmanExtensionQuasiNewton2010][^kanamoriBregmanExtensionQuasiNewton2010a]をご参照ください。
[こちらのスライド](http://matsuzoe.web.nitech.ac.jp/infogeo/OCAMI2010/kanamori.pdf)も参考になります。

### BFGS更新の詳細

本小節ではBFGS更新に注目し、その公式の詳細な導出を示します。 BFGS更新は実用上最も成功した準ニュートン更新則の一つとして知られています。 BFGS更新は次で与えられていたことを再掲しておきます。

```math
\begin{equation*}
B_{k+1}   = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}.
\end{equation*}
```

#### 逆更新の公式

BFGS更新の逆行列 $H_k \mathrel{\vcenter{:}}= B_k^{-1}$ は次式で与えられます。

```math
\begin{equation*}
H_{k+1} = \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) + \frac{s_k s_k^\top}{y_k^\top s_k}.
\end{equation*}
```

なお、このBFGS更新における $H_{k+1}$ の公式は、DFP更新における $B_{k+1}$ の形式と同じであることに注意してください。これらは双対の関係にあります。以下では、この式が確かにBFGS更新の逆行列を与えることを示します。

**Proposition 12**

行列 $H_{k+1}$ は $B_{k+1}$ の逆行列である。

<details><summary>証明</summary>

BFGS更新は次の簡潔なランク2の形に書き直せます。

```math
\begin{equation*}
B_{k+1} = B_k + UCV^\top.
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

です。実際、次の式が成り立ちます。

```math
\begin{equation*}
U C V^\top   = \begin{bmatrix} -\frac{B_k s_k}{s_k^\top B_k s_k}&    \frac{y_k}{y_k^\top s_k} \end{bmatrix} \begin{bmatrix} s_k^\top B_k  \\  y_k^\top \end{bmatrix}
= -\frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}.
\end{equation*}
```

Sherman–Morrison–Woodbury の恒等式より、$H_{k+1}$ を次のように計算できます。

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

を得ます。

(証明終わり)

</details>

#### BFGS更新の正定値性

BFGS更新の重要な性質として、現在の近似 $B_k$ が正定値で曲率条件 $y_k^\top s_k > 0$ が成り立つなら、更新後の近似 $B_{k+1}$ も正定値であることが保証されます。

**Proposition 13**

$B_k$ が正定値で $y_k^\top s_k > 0$ が成り立つなら、$B_{k+1}$ も正定値である。

<details><summary>証明</summary>

仮定より $B_k$ とその逆行列 $H_k$ は正定値です。任意の非零ベクトル $v \in \mathbb{R}^n$ に対して

```math
\begin{equation*}
v^\top H_{k+1} v
= v^\top \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) v + v^\top \frac{s_k s_k^\top}{y_k^\top s_k} v
\geq 0 + \frac{(s_k^\top v)^2}{y_k^\top s_k} > 0.
\end{equation*}
```

ここで第1項は $H_k$ が正定値であるため非負であり、第2項は曲率条件 $y_k^\top s_k > 0$ により正です。よって $H_{k+1}$ は正定値であり、その逆行列 $B_{k+1} = H_{k+1}^{-1}$ も正定値であることが分かります。

(証明終わり)

</details>

#### BFGS更新のトレースと行列式の公式

更新行列の固有値挙動の解析に有用な、BFGS更新のトレースと行列式の公式も示しておきます。対称行列においては、これらはそれぞれ固有値の総和と積に対応しています。従ってトレースと行列式が有界なら、固有値自体も有界に保たれると期待できます。これは $\mu$-強凸性や $L$-平滑性など、目的関数のヘッセ行列固有値に関する仮定と密接に関係するものとなっています。

##### トレースの公式

BFGS更新後の行列のトレースは、次のような明示的な公式で与えられます。

**Proposition 14** ([^nocedal1999numerical] (eq. 6.44))

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ をBFGS更新とする。このとき

```math
\begin{equation*}
\mathrm{tr}(B_{+}) = \mathrm{tr}(B) - \frac{\left\lVert B s \right\rVert^2}{s^\top Bs} + \frac{\left\lVert y \right\rVert^2}{y^\top s}
\end{equation*}
```

が成り立つ。

<details><summary>証明</summary>

BFGS更新にトレースを適用すると

```math
\begin{equation*}
\mathrm{tr}(B_{+}) = \mathrm{tr}(B) - \mathrm{tr}\left(\frac{Bss^\top B}{s^\top Bs}\right) + \mathrm{tr}\left(\frac{yy^\top}{y^\top s}\right).
\end{equation*}
```

第2項について

```math
\begin{equation*}
\mathrm{tr}\left(\frac{Bss^\top B}{s^\top Bs}\right) = \frac{1}{s^\top Bs}\mathrm{tr}((Bs)(Bs)^\top) = \frac{\left\lVert B s \right\rVert^2}{s^\top Bs}.
\end{equation*}
```

第3項について

```math
\begin{equation*}
\mathrm{tr}\left(\frac{yy^\top}{y^\top s}\right) = \frac{1}{y^\top s}\mathrm{tr}(yy^\top) = \frac{\left\lVert y \right\rVert^2}{y^\top s}.
\end{equation*}
```

以上より所望の式が得られます。

(証明終わり)

</details>

##### 行列式の公式

BFGS更新後の行列式も閉形式で与えられます。

**Proposition 15** ([^nocedal1999numerical] (eq. 6.45))

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ をBFGS更新とし、$B$ が正則であるとする。このとき

```math
\begin{equation*}
\det(B_{+}) = \det(B) \frac{y^\top s}{s^\top Bs}
\end{equation*}
```

が成り立つ。

<details><summary>証明</summary>

[行列式補題](https://en.wikipedia.org/wiki/Matrix_determinant_lemma)より、BFGS更新のランク2表現を思い出すと、$U, C, V$ は次を満たします。

```math
\begin{equation*}
\det(B_{k+1}) =\det(B_k + U C V^\top)=\det(B_k)\det(C) \det \left(C^{-1} + V^\top B_k^{-1} U\right).
\end{equation*}
```

ここで $I_2$ は $2\times 2$ の単位行列です。

$U=V=\begin{bmatrix}B_k s_k & y_k\end{bmatrix}$ なので

```math
\begin{equation*}
V^\top B_k^{-1} U = \begin{bmatrix} s_k^\top B_k s_k & s_k^\top y_k \\ y_k^\top s_k     & y_k^\top B_k^{-1} y_k \end{bmatrix},
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

を得ます。以上で、証明が完了します。

(証明終わり)

</details>

### BFGSとDFPの比較

これまで何度か言及したように、BFGSとDFPは双対関係にあります。しかし、実際の最適化問題に適用すると実用上の効率は大きく異なることが知られています。Powellによる解析[^powellHowBadAre1986]では、単純な二次元二次関数に対する両手法の挙動を調べて、この非対称性を検討しました。漸近的には同程度に振る舞うと示唆されることが多い両手法について、Powellはその実用上の効率が大きく異なること、特に近似ヘッセ行列が真のヘッセ行列から遠い場合に差が顕著であることを明らかにしました。ここでは、その数値実験を再現します。

#### 問題設定

Powellの問題設定に従い、次の二次関数を考えます。

```math
\begin{equation*}
f(x, y) = \frac{1}{2}(x^2 + y^2).
\end{equation*}
```

この実験で最も大切なことは、初期ヘッセ行列 $B_0$ の選び方です。本来、この関数のヘッセ行列は単位行列ですが、ここではあえて大幅に誤った近似ヘッセ行列を用いることで、そのような誤りに対する耐性や、その修正能力を測ることを目的としています。具体的には、初期のヘッセ近似 $B_0$ が固有値 1 と $\lambda_1$ を持つように選びました。$\lambda_1$ は初期近似の誤差の程度を表しています。また、BFGSとDFPのどちらも各反復で固定ステップサイズ $\alpha_k = 1$ を用いました。

初期点 $x_0$ は次のように選びました。

```math
\begin{equation*}
\theta = \arctan(\sqrt{\lambda_1}), \quad x_0 = \begin{bmatrix}\cos(\theta) \\ \sin(\theta)\end{bmatrix}.
\end{equation*}
```

これはPowellの元の解析に沿ったものです。この選択の詳細は[^powellHowBadAre1986]を参照してください。

反復は現在点のノルムが初期ノルムに対する許容値を下回るまで続けました。そして、各 $\lambda_1$ に対して収束に要する反復回数を記録しました。

#### 数値結果

数値結果を Table 1 に示します。これはPowellの元論文の表と部分的に一致しています。収束挙動は初期固有値 $\lambda_1$ に強く依存することが分かります。

| $\lambda_1$ | BFGS | DFP |
| --- | --- | --- |
| 0.001 | 4 | 3 |
| 0.01 | 5 | 3 |
| 0.1 | 6 | 4 |
| 1 | 1 | 1 |
| 10 | 8 | 16 |
| 100 | 10 | 107 |
| 1000 | 12 | 1006 |
| 10000 | 15 | 9987 |

(Table 1: 初期固有値 $\lambda_1$ に対するBFGSとDFPの収束比較。)

Fig. 11 と Fig. 12 は特定の $\lambda_1$ に対する反復軌跡を示しています。これらの図は、同一の初期点から最小点(原点)へ向かう二つの手法の進み方を可視化し、収束速度と経路の違いを明確に示しています。

![../imgs/quasi_newton/bfgs_vs_dfp_100.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/bfgs_vs_dfp_100.png?v=1)

(Fig. 11 $\lambda_1 = 100$ におけるBFGSとDFPの反復軌跡。BFGSは10回で収束する一方、DFPは107回を要し、固有値誤差が大きい場合にBFGSが優位であることを示しています。)

![../imgs/quasi_newton/bfgs_vs_dfp_0.1.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/bfgs_vs_dfp_0.1.png?v=1)

(Fig. 12 $\lambda_1 = 0.1$ におけるBFGSとDFPの反復軌跡。どちらも素早く収束し、DFPがBFGSよりわずかに速くなっています。)

#### 解析と議論

数値結果はBFGSとDFPの間に顕著な非対称性があることを示しています。$\lambda_1 > 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過大評価する場合、BFGS は大幅に高い効率を示しました。一方で、 $\lambda_1 < 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過小評価する場合には、DFPはBFGSよりわずかに良い性能を示しましたが、その差は非常に小さいものでした。この傾向の逆転は理論的な対称性から予測されるものですが、その差の大小は注目に値します。

##### ヘッセ行列の補正における非対称性

両者の性能の非対称性は、誤った固有値を補正する仕方の根本的な違いに起因します。大事な考察として、過大な固有値の補正が過小な固有値の補正よりも重要であることが挙げられます。

ヘッセ固有値が過大評価されると、アルゴリズムは過度に保守的なステップを取り、最適化の進みが遅くなります。この誤差を補正するには、更新則が大きな固有値を 1 へ縮小する必要があります。BFGS更新はこのようなタスクには強い効果を発揮しますが、DFP更新は苦手です。

一方で、ヘッセ固有値が過小評価されると、アルゴリズムは大胆なステップを取りますが、その過小評価は自動で修正されやすいです。新しい点での勾配計算が近似の改善を促す為、このような過小評価の補正は本質的に容易で、必要な反復回数も少なくて済みます。

このような差異が、BFGS更新とDFP更新の性能の非対称性を生み出していると考えることが出来ます。

### 記憶制限BFGS (L-BFGS)

以下では、準ニュートン法を大規模最適化問題へ拡張する際に重要となる、記憶制限準ニュートン法について説明します。通常の準ニュートン法では、近似ヘッセ行列 $B_k$ またはその逆行列 $H_k$ を密行列として陽に保存・更新するため、$n$ 変数に対して $\mathcal{O}(n^2)$ のメモリを要します。一方で、記憶制限準ニュートン法では、最新の $m$ 組のベクトルペア $\lbrace(s_i,y_i)\rbrace$ という限られた情報のみを保持して、その情報だけから近似ヘッセ行列に関する計算を行います。この工夫により空間計算量は $\mathcal{O}(nm)$ に減少し、$m$ が小さな定数(通常は $m\le 10$) のとき大幅な改善となります。

特に、BFGS更新の記憶制限版であるL-BFGS法[^liuLimitedMemoryBFGS1989a]は、その代表的な手法です。このBFGS更新の場合に注目し、限られた情報だけを用いて準ニュートン方向 $d_k = -H_k g_k$ を空間計算量・時間計算量の両面で効率的に計算する方法を示します。

本小節では次の有限長の行列の列を扱います。

```math
\begin{equation*}
H_0, H_1, \dots, H_m.
\end{equation*}
```

ここで $H_\ell$ は初期行列 $H_0$ に対して $\ell$ 回のBFGS更新を適用して得られる逆ヘッセ近似を表します。

#### 逆行列のコンパクト表現

保存されたベクトルペア $\lbrace(s_i,y_i)\rbrace_{i=0}^{m-1}$ を用いて、次を定義します。

```math
\begin{equation*}
\rho_i = \frac{1}{y_i^\top s_i}, \qquad
V_i = I - \rho_i y_i s_i^\top.
\end{equation*}
```

このとき、 $i = 0,\dots, m-1$ に対して、BFGS更新の逆行列は次のように表されます。

```math
\begin{equation*}
H_{i+1} = V_i^\top H_i V_i + \rho_i s_i s_i^\top.
\end{equation*}
```

この関係を再帰的に展開すると次のコンパクト表現が得られます。

```math
\begin{equation*}
H_m
=
V_{m-1}^\top \cdots V_0^\top H_0 V_0 \cdots V_{m-1}
+
\sum_{j=0}^{m-1}
(V_{m-1}^\top \cdots V_{j+1}^\top)
\rho_j s_j s_j^\top
(V_{j+1} \cdots V_{m-1}).
\end{equation*}
```

ここで $H_0$ は選ばれた初期逆ヘッセ近似であり、通常はスケールされた単位行列です。

#### 二重ループ再帰

準ニュートン法では、逆行列 $H_m$ と与えられたベクトル $q \in \mathbb{R}^n$ に対して $r = H_m q$ が効率的に計算できることが、アルゴリズムにおいて重要です。この計算は長さ $m$ の短いループを二回回すだけで実行でき、L-BFGS のtwo-loop recursionと呼ばれるアルゴリズムとして知られています[^nocedal1999numerical] (Algorithm 7.4)。このアルゴリズムは $\mathcal{O}(nm)$ の時間・空間計算量を要します。

![999_two_loop_recursion](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/study/999_two_loop_recursion.png)

この二重ループ再帰の出力が確かに $r = H_m q$ を計算していることを、以下では確認します。

**Proposition 16**

二重ループ再帰アルゴリズムの出力は $r = H_m q$ を満たす。

<details><summary>証明</summary>

アルゴリズムの1つ目のループ ($i = m-1, m-2, \dots, 0$) では、入力ベクトル $q^{(m)} \mathrel{\vcenter{:}}= q$ から次を計算する。

```math
\begin{equation*}
\alpha_i \mathrel{\vcenter{:}}= \rho_i s_i^\top q^{(i+1)}, \qquad
q^{(i)} \mathrel{\vcenter{:}}= q^{(i+1)} - \alpha_i y_i.
\end{equation*}
```

$\alpha_i$ の定義を代入し、コンパクト表現における $V_i$ の定義を用いると

```math
\begin{equation*}
q^{(i)} = q^{(i+1)} - \rho_i \left(s_i^\top q^{(i+1)}\right) y_i = \left(I - \rho_i y_i s_i^\top\right) q^{(i+1)} = V_i q^{(i+1)}.
\end{equation*}
```

よってすべての $i = 0, 1, \dots, m-1$ に対して、次が得られます。

```math
\begin{equation*}
q^{(i)} = V_i V_{i+1} \cdots V_{m-1} q.
\end{equation*}
```

次にアルゴリズムは $H_0$ を適用します。

```math
\begin{equation*}
r^{(0)} = H_0 q^{(0)} = H_0 V_0 V_1 \cdots V_{m-1} q.
\end{equation*}
```

続いて、$i = 0, 1, \dots, m-1$ に対して、二つ目のループは次を計算します。

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
(V_{j+1} \cdots V_{m-1}) q.
\end{equation*}
```

これは、$H_m$ のコンパクト表現を $q$ に適用した式と一致します。よって、アルゴリズムの出力が $r = H_m q$ を満たすことが示され、証明が完了します。

(証明終わり)

</details>

従って、このtwo-loop recursionによって、行列 $H_m$ を陽に構成せずに、ベクトルに対する作用を正確に評価できることがわかります。

#### 初期スケーリング

L-BFGS 法の重要な要素は初期行列 $H_0$ の選択です。広く使われるのは、単位行列にスケーリングを施したものです。

```math
\begin{equation*}
H_0 = \gamma I.
\end{equation*}
```

ここでスケーリング係数 $\gamma$ は、例えば次のように選ばれます。

```math
\begin{equation*}
\gamma = \frac{s_{m-1}^\top y_{m-1}}{y_{m-1}^\top y_{m-1}}.
\end{equation*}
```

この選択は次の議論によって正当化できます[^liuLimitedMemoryBFGS1989a][^shannoMatrixConditioningNonlinear1978]。目的関数 $f$ が二回連続微分可能であると仮定し、最新のステップに沿った平均ヘッセ行列を考えます。

```math
\begin{equation*}
\bar{G} = \int_0^1 \nabla^2 f(x + \tau s_{m-1}) \mathrm{d}\tau.
\end{equation*}
```

ここで $s_{m-1}$ は最新の $x$ の変位を表しており、次を満たします。

```math
\begin{equation*}
y_{m-1}
=
\nabla f(x+s_{m-1}) - \nabla f(x)
=
\bar{G} s_{m-1}.
\end{equation*}
```

この関係を用いるとスケーリング係数は次のように書けます。

```math
\begin{equation*}
\frac{s_{m-1}^\top y_{m-1}}{y_{m-1}^\top y_{m-1}}
=
\frac{(\bar{G}^{1/2} s_{m-1})^\top (\bar{G}^{1/2} s_{m-1})}
{(\bar{G}^{1/2} s_{m-1})^\top \bar{G} (\bar{G}^{1/2} s_{m-1})}.
\end{equation*}
```

これは、ベクトル $\bar{G}^{1/2} s_{m-1}$ に関する行列 $\bar{G}$ のRayleigh商の逆数です。従って、このスケーリングはこれらの方向に沿った平均逆曲率を大まかに近似しています。

さらにこの $\gamma$ の選択は Barzilai–Borwein 法の短ステップサイズ[^barzilaiTwoPointStepSize1988] と一致しており、L-BFGSの初期化と古典的なステップ長選択戦略の密接な関係を示しています。

<!-- From 4_modified_secant.tex -->

## 修正セカント条件

本節では、通常のセカント条件の修正版である修正セカント条件について説明します。修正セカント条件は勾配情報に加えて関数値情報を取り込み、より正確なヘッセ行列の近似を実現します。

### 修正セカント条件の導出

セカント条件は準ニュートン法で広く用いられていますが、勾配情報のみを活用するため、目的関数の曲率を正確に捉えられないことがあります。このことをFig. 13で説明します。

Fig. 13 において、関数値と勾配が既知の2点 $x_k$ と $x_{k+1}$ があるとします。正確なヘッセ行列から構築された理想的な二次モデルは、新しい点 $x_{k+1}$ の周りでよくフィットしており、より良い収束挙動をもたらすことが分かります。しかし、標準的な勾配のみを用いて課されるセカント条件で作られるモデルでは、関数値の情報が無視され、曲率を大きく誤る可能性があり、真の目的関数の近似が悪くなります。

<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_EXPLAIN.png?v=1" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_HESS.png" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_BFGS.png" />

(Fig. 13 標準的なセカント条件の欠点。 (a) $x_k$ と $x_{k+1}$ での関数値と勾配が既に分かっている。 (b) 正確なヘッセ行列から構築された理想的な二次モデルは、新しい点 $x_{k+1}$ の周りでよくフィットする。 (c) 標準的なセカント条件では、必ずしも十分に曲率を捉えられない。)

この問題は、関数値を利用することで克服できます。基本的な考え方は Fig. 14 に示した通りです。2 点 $x_k$ と $x_{k+1}$ で勾配が同じであっても、適切な補完は関数値 $f(x_k)$ と $f(x_{k+1})$ によって異なります。この観察こそが、関数値情報を取り込んでヘッセ行列の近似を改善する修正セカント条件の動機を端的に示しています。

![../imgs/modified_secant/cubic_interpolation.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/cubic_interpolation.png)

(Fig. 14 修正セカント条件の基本的な考え方。$x_k$ と $x_{k+1}$ で同一の勾配を持つが異なる関数値を持つ場合、適切な補間は異なります。)

以下では、2つの既知の修正セカント条件を提示します。

#### 関数値一致の修正セカント条件

最初の修正は、二次モデルが前の点での関数値と一致することを強制します[^yuanModifiedBFGSAlgorithm1991][^weiNewQuasiNewtonMethods2006][^babaie-kafakiModifiedBFGSAlgorithm2011]。異なる近似ヘッセ行列 $B_{k+1}^{\mathrm{F}}$ を持つ別のモデルを考えます。

```math
\begin{equation*}
m_{k+1}^{\mathrm{F}}(x) \mathrel{\vcenter{:}}= f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1}^{\mathrm{F}} (x - x_{k+1}).
\end{equation*}
```

ここでは、

```math
\begin{equation*}
m_{k+1}^{\mathrm{F}}(x_k) = f(x_k),
\end{equation*}
```

つまり、前の点での関数値が、モデル関数でも一致することを課します。

対応する修正セカント条件を $B_{k+1}^{\mathrm{F}}$ に対して導出しましょう。$m_{k+1}^{\mathrm{F}}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、条件は以下になります。

```math
\begin{align*}
f(x_k) & = f(x_{k+1}) + \nabla f(x_{k+1})^\top (x_k - x_{k+1}) + \frac{1}{2} (x_k - x_{k+1})^\top B_{k+1}^{\mathrm{F}} (x_k - x_{k+1}) \\
& = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top B_{k+1}^{\mathrm{F}} s_k.
\end{align*}
```

ここで、通常のセカント条件の代わりに、次のように単位行列による修正を加えた形を仮定します。

```math
\begin{equation*}
\left(B_{k+1}^{\mathrm{F}} + \sigma_k^\mathrm{F} I \right) s_k = y_k.
\end{equation*}
```

ここで、$\sigma_k^\mathrm{F} \in \mathbb{R}$ は以下で決定するスカラーです。この方程式を代入すると、以下が得られます。

```math
\begin{equation*}
f(x_k) = f(x_{k+1}) - \nabla f(x_{k+1})^\top s_k + \frac{1}{2} s_k^\top y_k - \frac{\sigma_k^\mathrm{F}}{2} \left\lVert s_k \right\rVert^2.
\end{equation*}
```

$\sigma_k^\mathrm{F}$ について解き、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、以下を得ます。

```math
\begin{align*}
\sigma_k^\mathrm{F} & = \frac{2(f(x_{k+1}) - f(x_k)) - (2\nabla f(x_{k+1}) - y_k)^\top s_k}{\left\lVert s_k \right\rVert^2}           \\
& = \frac{2(f(x_{k+1}) - f(x_k)) - (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\left\lVert s_k \right\rVert^2}.
\end{align*}
```

従って、修正セカント条件は以下になります。

```math
\begin{equation*}
B_{k+1}^{\mathrm{F}} s_k = \hat{y}^{\mathrm{F}}_k \mathrel{\vcenter{:}}= y_k + \frac{2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k}{\left\lVert s_k \right\rVert^2} s_k.
\end{equation*}
```

よって、各種の準ニュートン法の更新において、$s_k, y_k$ の代わりに、この修正された $\hat{y}^\mathrm{F}_k$ を使用することで、殆ど同じアルゴリズムのまま、目的関数値の情報を取り込むことができます。また、特にBFGS更新で $s_k^\top y_k > 0$ という条件は正定値性を保つために使われるなど重要な性質である為、$s_k$ との内積が正であることを保証した、次の修正も考えられます。

```math
\begin{equation*}
B_{k+1}^{\mathrm{F}'} s_k = y_k + \frac{\max(0, 2(f(x_k) - f(x_{k+1})) + (\nabla f(x_{k+1}) + \nabla f(x_k))^\top s_k)}{\left\lVert s_k \right\rVert^2} s_k.
\end{equation*}
```

これが関数値一致の修正セカント条件です。具体例としてFig. 15 も参照してください。

![../imgs/modified_secant/trial_2.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_2.png)

(Fig. 15 関数値一致の修正セカント方程式。前の点 $x_{k-1}$ での関数値 $f(x_{k-1})$ と $m_k^{\mathrm{F}}(x_{k-1})$ が一致している。)

#### 三次項による修正セカント条件

2つ目の修正セカント条件では、モデルに三次項を導入し、前の点での関数値と勾配の両方の一致を同時に満たすことを可能にします[^zhangNewQuasiNewtonEquation1999][^zhangPropertiesNumericalPerformance2001][^yabeLocalSuperlinearConvergence2007]。

$T_{k+1} \in \mathbb{R}^{n \times n \times n}$ を、以下を満たす $x_{k+1}$ での $f$ の3階微分テンソルとします。

```math
\begin{equation*}
s_k^\top (T_{k+1} s_k) s_k = \sum_{i,j,l=1}^n \partial_{x_i x_j x_l} f(x_{k+1}) s_k^{(i)} s_k^{(j)} s_k^{(l)}.
\end{equation*}
```

ここで、$\partial_{x_i x_j x_l} f$ は $f$ の $x_i$、$x_j$、および $x_l$ に対する3階微分を表し、$s_k^{(i)}$ はベクトル $s_k$ の第 $i$ 成分です。このテンソルは分析目的でのみ導入しており、最終的な式からは除去されます。

このテンソル項を組み込むことにより、以下のモデル関数を定義できます。

```math
\begin{align*}
m_{k+1}^\mathrm{C}(x) \mathrel{\vcenter{:}}={} & f(x_{k+1}) + \nabla f(x_{k+1})^\top (x - x_{k+1}) + \frac{1}{2} (x - x_{k+1})^\top B_{k+1}^{\mathrm{C}} (x - x_{k+1}) \\*
& + \frac{1}{6}(x - x_{k+1})^\top (T_{k+1} (x - x_{k+1})) (x - x_{k+1}).
\end{align*}
```

このモデルを使用して、前の点 $x_k$ での関数値と勾配の両方の一致を強制できます。具体的には、以下を要求します。

```math
\begin{equation*}
\begin{cases}
m_{k+1}^\mathrm{C}(x_k) = f(x_k), \\
\nabla m_{k+1}^\mathrm{C}(x_k) = \nabla f(x_k).
\end{cases}
\end{equation*}
```

$m_{k+1}^\mathrm{C}$ の定義を代入し、$s_k = x_{k+1} - x_k$ を使用することで、これらの条件を以下のように書き直せます。

```math
\begin{align*}
f(x_k)        & = f(x_{k+1}) - s_k^\top \nabla f(x_{k+1}) + \frac{1}{2} s_k^\top B_{k+1}^{\mathrm{C}} s_k - \frac{1}{6} s_k^\top (T_{k+1} s_k) s_k, \\
\nabla f(x_k) & = \nabla f(x_{k+1}) - B_{k+1}^{\mathrm{C}} s_k + \frac{1}{2} (T_{k+1} s_k) s_k.
\end{align*}
```

一番目の式の両辺に3を乗じ、二番目の方程式については $s_k$ との内積をとることで、以下が得られます。

```math
\begin{align*}
3(f(x_k) - f(x_{k+1})) +3 s_k^\top \nabla f(x_{k+1}) & = \frac{3}{2} s_k^\top B_{k+1}^{\mathrm{C}} s_k - \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k, \\
-s_k^\top y_k                                        & = -s_k^\top B_{k+1}^{\mathrm{C}} s_k + \frac{1}{2} s_k^\top (T_{k+1} s_k) s_k.
\end{align*}
```

合計し、$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ を使用することで、テンソル項を除去し、以下のスカラー恒等式を得ます。

```math
\begin{equation*}
3(f(x_k) - f(x_{k+1}))
+ \frac{3}{2} s_k^\top (\nabla f(x_{k+1}) + \nabla f(x_k)) + \frac{1}{2} s_k^\top y_k
= \frac{1}{2} s_k^\top B_{k+1}^{\mathrm{C}} s_k.
\end{equation*}
```

ここでも、あるスカラー $\sigma_k^\mathrm{C}$ に対して $\left( B_{k+1}^{\mathrm{C}} + \sigma_k^\mathrm{C} I \right) s_k = y_k$ と仮定します。そうすると、先の方程式は

```math
\begin{equation*}
3(f(x_k) - f(x_{k+1})) + \frac{3}{2} s_k^\top (\nabla f(x_{k+1}) + \nabla f(x_k)) = - \frac{\sigma_k^\mathrm{C}}{2} \left\lVert s_k \right\rVert^2,
\end{equation*}
```

つまり、

```math
\begin{equation*}
\sigma_k^\mathrm{C} = -\frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\left\lVert s_k \right\rVert^2}.
\end{equation*}
```

従って、以下の修正セカント条件が得られます。

```math
\begin{equation*}
B_{k+1}^{\mathrm{C}} s_k = y_k + \frac{6(f(x_k) - f(x_{k+1})) + 3 s_k^\top (\nabla f(x_k) + \nabla f(x_{k+1}))}{\left\lVert s_k \right\rVert^2} s_k.
\end{equation*}
```

これが三次項による修正セカント条件です。具体例としてFig. 16も参照してください。

<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_1_cubic.png" /><img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/modified_secant/trial_1_quadratic.png" />

(Fig. 16 三次項による修正セカント方程式。三次項をモデルに組み込むことで、前の点で関数値と勾配の両方を一致させることが出来ます。その二次項までの展開によって、モデルを作成します。)

### その他の曲率関連手法

曲率情報を表すベクトルペア $s_k, y_k$ を求めることについては、他にもいくつかのトピックがあります。Agg-BFGS[^berahasLimitedmemoryBFGSDisplacement2022] は、最も古い情報を破棄して最新のものを追加するのではなく、データを集約することにより曲率情報を管理する別のアプローチです。

Multi-Secant[^leeAdvancingMultiSecantQuasiNewton2025] とは、複数のステップと勾配差ベクトルのペアを維持することにより、セカント条件を拡張します。ここまでで見てきた標準的な形式では、次のように定義していました。

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

## 最後に

以上です。お読みいただきありがとうございました。

[^Doikov2021SecondOrderTensor]: Nikita Doikov. New Second-Order and Tensor Methods in Convex Optimization. PhD thesis, Universit\'e catholique de Louvain, 2021.
[^FanZhongXiuMingLianSokZuiShiHuaarugorizumu2023]: 飯塚 秀明. 連続最適化アルゴリズム. オーム社, 2023. ISBN 978-4-274-23006-6.
[^babaie-kafakiModifiedBFGSAlgorithm2011]: Saman Babaie-Kafaki. A modified BFGS algorithm based on a hybrid secant equation. Science China Mathematics, 54 (9): 2019--2036, 2011. ISSN 1869-1862. https://doi.org/10.1007/s11425-011-4232-7.
[^barzilaiTwoPointStepSize1988]: Jonathan Barzilai and Jonathan M. Borwein. Two-Point Step Size Gradient Methods. IMA Journal of Numerical Analysis, 8 (1): 141--148, 1988. ISSN 0272-4979. https://doi.org/10.1093/imanum/8.1.141.
[^bauschkeBaillonHaddadTheoremRevisited2009]: Heinz H. Bauschke and Patrick L. Combettes. The Baillon-Haddad Theorem Revisited, 2009.
[^berahasLimitedmemoryBFGSDisplacement2022]: Albert S. Berahas, Frank E. Curtis, and Baoyu Zhou. Limited-memory BFGS with displacement aggregation. Mathematical Programming, 194 (1): 121--157, 2022. ISSN 1436-4646. https://doi.org/10.1007/s10107-021-01621-6.
[^dennisjr.QuasiNewtonMethodsMotivation1977a]: J. E. Dennis, Jr. and Jorge J. Mor\'e. Quasi-Newton Methods, Motivation and Theory. SIAM Review, 19 (1): 46--89, 1977. ISSN 0036-1445. https://doi.org/10.1137/1019005.
[^doi:10.1137/1.9781611971200]: J. E. Dennis and Robert B. Schnabel. Numerical Methods for Unconstrained Optimization and Nonlinear Equations. Society for Industrial and Applied Mathematics, 1996. https://doi.org/10.1137/1.9781611971200.
[^haeltermanAnalyticalStudyLeast2009]: Robby Haelterman. Analytical study of the Least Squares Quasi-Newton method for interaction problems. PhD thesis, 2009.
[^kanamori2016continuous]: 金森 敬文, 鈴木 大慈, 竹内 一郎, and 佐藤 一誠. 機械学習のための連続最適化. 機械学習プロフェッショナルシリーズ. 講談社サイエンティフィク, 2016. ISBN 978-4-06-152920-5.
[^kanamoriBregmanExtensionQuasiNewton2010]: Takafumi Kanamori and Atsumi Ohara. A Bregman Extension of quasi-Newton updates II: Convergence and Robustness Properties, 2010\natexlaba.
[^kanamoriBregmanExtensionQuasiNewton2010a]: Takafumi Kanamori and Atsumi Ohara. A Bregman Extension of quasi-Newton updates I: An Information Geometrical framework, 2010\natexlabb.
[^leeAdvancingMultiSecantQuasiNewton2025]: Mokhwa Lee and Yifan Sun. Advancing Multi-Secant Quasi-Newton Methods for General Convex Functions, 2025.
[^liuLimitedMemoryBFGS1989a]: Dong C. Liu and Jorge Nocedal. On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45 (1): 503--528, 1989. ISSN 1436-4646. https://doi.org/10.1007/BF01589116.
[^m.j.d.powellNewAlgorithmUnconstrained1970]: M.J.D. Powell. A New Algorithm for Unconstrained Optimization. In Nonlinear Programming, pages 31--65. Academic Press, 1970. https://doi.org/10.1016/B978-0-12-597050-1.50006-3.
[^nesterovIntroductoryLecturesConvex2014]: Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Springer Publishing Company, Incorporated, 1 edition, 2014. ISBN 978-1-4613-4691-3.
[^nocedal1999numerical]: Jorge Nocedal and Stephen J Wright. Numerical Optimization. Springer, 1999. ISBN 978-0-387-30303-1. https://doi.org/10.1007/978-0-387-40065-5.
[^powellHowBadAre1986]: M. J. D. Powell. How bad are the BFGS and DFP methods when the objective function is quadratic? Mathematical Programming, 34 (1): 34--47, January 1986. ISSN 1436-4646. https://doi.org/10.1007/BF01582161.
[^rockafellarVariationalAnalysis1998]: R. Tyrrell Rockafellar and Roger J. B. Wets. Variational Analysis, volume 317 of Grundlehren Der Mathematischen Wissenschaften. Springer, 1998. ISBN 978-3-540-62772-2 978-3-642-02431-3. https://doi.org/10.1007/978-3-642-02431-3.
[^shannoMatrixConditioningNonlinear1978]: D. F. Shanno and Kang-Hoh Phua. Matrix conditioning and nonlinear optimization. Mathematical Programming, 14 (1): 149--160, December 1978. ISSN 1436-4646. https://doi.org/10.1007/BF01588962.
[^weiNewQuasiNewtonMethods2006]: Zengxin Wei, Guoyin Li, and Liqun Qi. New quasi-Newton methods for unconstrained optimization problems. Applied Mathematics and Computation, 175 (2): 1156--1188, 2006. ISSN 0096-3003. https://doi.org/10.1016/j.amc.2005.08.027.
[^yabeLocalSuperlinearConvergence2007]: Hiroshi Yabe, Hideho Ogasawara, and Masayuki Yoshino. Local and superlinear convergence of quasi-Newton methods based on modified secant conditions. Journal of Computational and Applied Mathematics, 205 (1): 617--632, 2007. ISSN 03770427. https://doi.org/10.1016/j.cam.2006.05.018.
[^yuanModifiedBFGSAlgorithm1991]: Ya-Xiang Yuan. A Modified BFGS Algorithm for Unconstrained Optimization. IMA Journal of Numerical Analysis, 11 (3): 325--332, 1991. ISSN 0272-4979. https://doi.org/10.1093/imanum/11.3.325.
[^zhangNewQuasiNewtonEquation1999]: J. Z. Zhang, N. Y. Deng, and L. H. Chen. New Quasi-Newton Equation and Related Methods for Unconstrained Optimization. Journal of Optimization Theory and Applications, 102 (1): 147--167, 1999. ISSN 0022-3239, 1573-2878. https://doi.org/10.1023/A:1021898630001.
[^zhangPropertiesNumericalPerformance2001]: Jianzhong Zhang and Chengxian Xu. Properties and numerical performance of quasi-Newton methods with modified quasi-Newton equations. Journal of Computational and Applied Mathematics, 137 (2): 269--278, 2001. ISSN 0377-0427. https://doi.org/10.1016/S0377-0427(00)00713-5.