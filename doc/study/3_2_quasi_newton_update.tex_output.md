
### 代表的な準ニュートン更新則

$B_k$, $s_k$, $y_k$ が与えられたとき、セカント条件を満たす $B_{k+1}$ を与える更新則は多数存在します。ここでは代表的なものをその導出とともに示します~\citep{dennisjr.QuasiNewtonMethodsMotivation1977a}。
本小節に限り、簡潔さのため $B_k$, $B_{k+1}$, $s_k$, $y_k$ をそれぞれ $B$, $\bar{B}$, $s$, $y$ と略記します。

#### Broyden更新

[Broyden更新](https://en.wikipedia.org/wiki/Broyden%27s_method)は最も古くから知られる更新則の一つですが、準ニュートン法の文脈では、対称性を保たないため実用上はあまり用いられません。歴史的な観点もふまえ、簡潔に紹介します。
更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{Broyden}} \mathrel{\vcenter{:}}= B + \frac{(y - Bs)s^\top}{s^\top s}.
\end{equation*}
```

##### 導出

単純な構造的仮定からこの公式を導出します~\citep[Section 4]{dennisjr.QuasiNewtonMethodsMotivation1977a}。

**Proposition 6**

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

を満たすと仮定します。このとき $\bar{B}$ は一意に定まり、$\bar{B}_{\mathrm{Broyden}}$ に一致します。

<details>
<summary>Proof</summary>

ベクトル $s$ と $s$ の直交補空間の基底は $\mathbb{R}^n$ の基底を成します。
$\bar{B}$ の条件はこの基底に対する $\bar{B}$ の作用を完全に決定するため、$\bar{B}$ は一意に定まります。
ここで $\bar{B}_{\mathrm{Broyden}}$ が課された条件を満たすことを示します。
$z^\top s = 0$ を満たす任意のベクトル $z$ を取ります。このとき、次を得ます。

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

したがって、$\bar{B}_{\mathrm{Broyden}}$ は課された条件を満たします。
よって、一意性より $\bar{B} = \bar{B}_{\mathrm{Broyden}}$ となります。
\myQED

</details>

Broyden更新は、フロベニウスノルムにおける最小変化更新としても特徴づけられます。

**Proposition 7** (\citep{dennisjr.QuasiNewtonMethodsMotivation1977a}, Theorem~4.1)

$B\in\mathbb{R}^{n\times n}$, $y\in\mathbb{R}^n$, $s\in\mathbb{R}^n\setminus\lbrace 0 \rbrace$ が与えられているとき、行列 $\bar{B}_{\mathrm{Broyden}}$ は以下の最適化問題の一意解である。

```math
\begin{align*}
\underset{\tilde{B} \in \mathbb{R}^{n \times n}}{\mathrm{minimize}} & \quad \lVert \tilde{B} - B \rVert_F \\
\mathrm{subject to}                                                 & \quad \tilde{B} s = y.
\end{align*}
```

<details>
<summary>Proof</summary>

関数 $\tilde{B}\mapsto\lVert \tilde{B}-B \rVert^2_F$ を最適化すると考えてよく、これは $\mathbb{R}^{n\times n}$ 上で凸です。
制約集合

```math
\begin{equation*}
\lbrace\tilde{B}\in\mathbb{R}^{n\times n}:\tilde{B}s=y\rbrace
\end{equation*}
```

はアフィン集合であり凸です。
よって、この最適化問題は高々一つの最小解しか持ちません。
$\bar{B}_{\mathrm{Broyden}}$ が実際に最小解であることを示します。
制約 $\tilde{B}s=y$ を満たす任意の $\tilde{B}$ に対して

```math
\begin{equation*}
\lVert \bar{B}_{\mathrm{Broyden}} - B \rVert_F^2
= \left\lVert (\tilde{B}-B) \frac{s s^\top}{s^\top s} \right\rVert_F^2
\leq \lVert \tilde{B}-B \rVert_F^2.
\end{equation*}
```

ここでフロベニウスノルムの劣乗法性と $\lVert ss^\top/(s^\top s) \rVert_F=1$ を用いました。
よって $\tilde{B}=\bar{B}_{\mathrm{Broyden}}$ です。
\myQED

</details>

この更新則は、対称性を保たないという点に注意してください。以下では、その点を改善した代表的な準ニュートン更新則を紹介します。

#### Symmetric Rank-One (SR1) 更新

[対称ランク1 (SR1) 更新](https://en.wikipedia.org/wiki/Symmetric_rank-one)~\citep{nocedal1999numerical} は更新過程で対称性を維持する基本的な準ニュートン法です。更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{SR1}} \mathrel{\vcenter{:}}= B + \frac{(y - B s)(y - B s)^\top}{(y - B s)^\top s}.
\end{equation*}
```

##### 導出

SR1 更新を導出するため、更新行列 $\bar{B}$ をランク1更新として構成します。すなわち、あるベクトル $z \in \mathbb{R}^n$ に対して、次が満たされると仮定します。

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

よって、SR1 更新則が導出されました。

##### 補足

SR1 更新は $(y - Bs)^\top s \neq 0$ であることを前提としています。$(y - Bs)^\top s = 0$ のとき分母が零となり、実用上は更新をスキップすることが多いです。

#### Powell Symmetric Broyden (PSB) 更新

Powell Symmetric Broyden (PSB) 更新~\cite{m.j.d.powellNewAlgorithmUnconstrained1970,doi:10.1137/1.9781611971200,haeltermanAnalyticalStudyLeast2009} も準ニュートン更新則の一つで、更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - B s) s^\top + s (y - B s)^\top}{s^\top s} - \frac{s^\top (y - B s)}{(s^\top s)^2} s s^\top
\end{equation*}
```

##### 導出

SR1 更新では更新項を $z z^\top$ として加えていました。代わりに、あるベクトル $c \in \mathbb{R}^n$ に対して、非対称なランク1更新 $z c^\top$ を考えることができます。そして最後に結果を対称化します。

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

重要な結果として、列 $\lbrace C_{2t} \rbrace_{t=0}^{\infty}$ はセカント条件を満たす対称行列に収束します。次の命題ではそれを示します。

**Proposition 8** (\citep{dennisjr.QuasiNewtonMethodsMotivation1977a}, Lemma~7.2)

行列の列 $\lbrace C_{t} \rbrace_{t=0}^{\infty}$ は収束し、その極限は次で与えられる。

```math
\begin{equation*}
\lim_{t \to \infty} C_{t}
=
C_{\infty}
\mathrel{\vcenter{:}}=
B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{c^\top s} - \frac{(y - Bs)^\top s}{(c^\top s)^2} c c^\top.
\end{equation*}
```

<details>
<summary>Proof</summary>

まず偶数番目の部分列を解析します。 $t=0,1,2,\dots$ に対し、次のように定義します。

```math
\begin{equation*}
G_t \mathrel{\vcenter{:}}= C_{2t}.
\end{equation*}
```

構成方法より各 $G_t$ は対称です。
定義から次を得ます。

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

行列 $cs^\top/c^\top s$ はランク1で固有値は $1,0,\dots,0$ です。
よって $P$ は固有値 $0$ を一つ持ち、残りはすべて $1/2$ となります。
特にスペクトル半径は $1/2<1$ であるので、次のように級数が収束します。

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

これで偶数番目の部分列 $\lbrace C_{2t} \rbrace$ が $C_{\infty}$ に収束することが示されました。
次に奇数番目の部分列について、次の式が成立しています。

```math
\begin{equation*}
C_{2t+1}
=
G_t+\frac{w_t c^\top}{c^\top s}.
\end{equation*}
```

$G_t\to C_{\infty}$ であり、また $\lVert w_t \rVert\to0$ が $P$ のスペクトル半径が1未満であることから従うため、

```math
\begin{equation*}
C_{2t+1} \to C_{\infty}.
\end{equation*}
```

従って、部分列 $\lbrace C_{2t} \rbrace$ と $\lbrace C_{2t+1} \rbrace$ はどちらも $C_{\infty}$ に収束し、証明が完了します。
\myQED

</details>

$c = s$ のとき、$C_{\infty}$ の一般式は標準的な PSB 更新則を導きます。

```math
\begin{equation*}
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - Bs)s^\top + s(y - Bs)^\top}{s^\top s} - \frac{(y - Bs)^\top s}{(s^\top s)^2} ss^\top.
\end{equation*}
```

#### Davidon--Fletcher--Powell (DFP) 更新

[Davidon--Fletcher--Powell (DFP) 更新](https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula)~\cite{nocedal1999numerical} も古典的な準ニュートン更新則です。
更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{DFP}} = \left(I - \frac{y s^\top}{y^\top s}\right) B \left(I - \frac{s y^\top}{y^\top s}\right) + \frac{y y^\top}{y^\top s}.
\end{equation*}
```

##### 導出

先に述べた PSB 更新では $c=s$ としましたが、別の $c$ を選ぶことも可能です。
具体的には $c=y$ を代入すると次のように DFP 更新が得られます。

```math
\begin{equation*}
\bar{B}_{\mathrm{DFP}} = B + \frac{(y - Bs)y^\top + y(y - Bs)^\top}{y^\top s} - \frac{(y - Bs)^\top s}{(y^\top s)^2} yy^\top.
\end{equation*}
```

なお、DFP更新には他にも導出方法が存在し、それらの一部は、次のBFGS更新における導出の双対版として理解することもできます。

#### Broyden--Fletcher--Goldfarb--Shanno (BFGS) 更新

[Broyden--Fletcher--Goldfarb--Shanno (BFGS) 更新](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)は最も広く使われる準ニュートン法の一つです。
更新則は次で与えられます。

```math
\begin{equation*}
\bar{B}_{\mathrm{BFGS}} = B - \frac{B s s^\top B}{s^\top B s} + \frac{y y^\top}{y^\top s}
\end{equation*}
```

##### 双対による導出

この更新はDFP更新の双対を考えることで導出できます。
後で示すように、$B$ に対するBFGS更新は $H$ に対する DFP 更新と同じ形式となっています。

##### KLダイバージェンス最小化による導出

KLダイバージェンスとは、行列間の近さを測る一種の非対称な尺度です。
正定値行列 $P,Q \in \mathbb{R}^{n \times n}$ に対して、次で定義されます。

```math
\begin{equation*}
\mathrm{KL}(P,Q) \mathrel{\vcenter{:}}= \mathrm{tr}(P Q^{-1}) - \log\det(P Q^{-1}) - n.
\end{equation*}
```

行列 $T \in \mathbb{R}^{n \times n}$ を任意の正則行列とします。
$\mathrm{tr}$ は2つの行列の積に対して可換性を持つため、KLダイバージェンスは次の不変性を持ちます。

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

**Proposition 9** ({\citep[Section 7.2.4]{kanamori2016continuous}})

$B \succ 0$ を正定値対称行列とする。
このとき、上記の最適化問題の解はBFGS更新則で与えられる。

<details>
<summary>Proof</summary>

$B$ が正定値対称行列であることに注意して、
まず、次のように定義します。

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
この問題の解を、ラグランジュの未定乗数法を用いて求めます。
ラグランジュ乗数 $\lambda\in\mathbb{R}^n$ および $\Lambda\in\mathbb{R}^{n\times n}$ を用いて、ラグランジアンを次のように定義します。

```math
\begin{equation*}
\mathcal{L}(B',\lambda,\Lambda) = \mathrm{tr}(B') -\log\det(B') -n +2\lambda^\top(B's'-y') +\mathrm{tr}\left(\Lambda(B'-B'^\top)\right).
\end{equation*}
```

一般に、行列微分の公式として、$X \succ 0$ に対して、$\log\det(X)$ の微分が $X^{-\top}$ かつ、 $\mathrm{tr}(AX)$ の微分が $A^\top$ となります。
KKT条件のうち停留性に関する条件は、$B'$ で微分し、$B'=B'^\top$ を用いることで次のように書けます。

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

制約条件 $B's'=y'$ より $B'^{-1}y'=s'$ であるから、
上式に右から $y'$ を掛けた式と、さらに左から $y'^\top$ を掛けた式はそれぞれ、次のようになる。

```math
\begin{align*}
s'         & = y' +(s'^\top y')\lambda +(\lambda^\top y')s'                                \\
y'^\top s' & = y'^\top y' + (s'^\top y')(y'^\top \lambda) + (\lambda^\top y')(y'^\top s').
\end{align*}
```

第2式より

```math
\begin{equation*}
\lambda^\top y'
=
\frac{y'^\top s'-y'^\top y'}{2(s'^\top y')}.
\end{equation*}
```

これを第1式に代入すると、

```math
\begin{equation*}
\lambda
= \frac{s'-y'}{s'^\top y'} - \frac{\lambda^\top y'}{s'^\top y'}s'
=
\frac{s'^\top y'+y'^\top y'}{2(s'^\top y')^2}s'
-\frac{1}{s'^\top y'}y'.
\end{equation*}
```

この $\lambda$ を $B'^{-1}$ の表式に代入して整理すると、

```math
\begin{align*}
B'^{-1} & = I + 2\frac{s'^\top y'+y'^\top y'}{2(s'^\top y')^2}(s' s'^\top)
-\frac{1}{s'^\top y'}(y' s'^\top + s' y'^\top)                                                                                         \\
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
\myQED

</details>

更なる詳細については
\citep{kanamoriBregmanExtensionQuasiNewton2010,kanamoriBregmanExtensionQuasiNewton2010a}
をご参照ください。
[こちらのスライド](http://matsuzoe.web.nitech.ac.jp/infogeo/OCAMI2010/kanamori.pdf)
も参考になります。

### BFGS更新の詳細

本小節ではBFGS更新に注目し、その公式の詳細な導出を示します。
BFGS更新は実用上最も成功した準ニュートン更新則の一つとして知られています。
BFGS更新則は次で与えられていたことを再掲しておきます。

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

なお、このBFGS更新における $H_{k+1}$ の公式は、DFP更新則における $B_{k+1}$ の形式と同じであることに注意してください。これらは双対の関係にあります。
以下では、この式が確かにBFGS更新の逆行列を与えることを示します。

**Proposition 10**

行列 $H_{k+1}$ は $B_{k+1}$ の逆行列である。

<details>
<summary>Proof</summary>

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

Sherman--Morrison--Woodbury の恒等式より、$H_{k+1}$ を次のように計算できます。

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
\myQED

</details>

#### BFGS更新の正定値性

BFGS更新の重要な性質として、現在の近似 $B_k$ が正定値で曲率条件 $y_k^\top s_k > 0$ が成り立つなら、更新後の近似 $B_{k+1}$ も正定値であることが保証されます。

**Proposition 11**

$B_k$ が正定値で $y_k^\top s_k > 0$ が成り立つなら、$B_{k+1}$ も正定値である。

<details>
<summary>Proof</summary>

仮定より $B_k$ とその逆行列 $H_k$ は正定値です。
任意の非零ベクトル $v \in \mathbb{R}^n$ に対して

```math
\begin{equation*}
v^\top H_{k+1} v
= v^\top \left(I - \frac{s_k y_k^\top}{y_k^\top s_k}\right) H_k \left(I - \frac{y_k s_k^\top}{y_k^\top s_k}\right) v + v^\top \frac{s_k s_k^\top}{y_k^\top s_k} v
\geq 0 + \frac{(s_k^\top v)^2}{y_k^\top s_k} > 0.
\end{equation*}
```

ここで第1項は $H_k$ が正定値であるため非負であり、第2項は曲率条件 $y_k^\top s_k > 0$ により正です。
よって $H_{k+1}$ は正定値であり、その逆行列 $B_{k+1} = H_{k+1}^{-1}$ も正定値であることが分かります。
\myQED

</details>

#### BFGS更新のトレースと行列式の公式

更新行列の固有値挙動の解析に有用な、BFGS更新のトレースと行列式の公式も示しておきます。
対称行列においては、これらはそれぞれ固有値の総和と積に対応しています。
従ってトレースと行列式が有界なら、固有値自体も有界に保たれると期待できます。
これは $\mu$-強凸性や $L$-平滑性など、目的関数のヘッセ行列固有値に関する仮定と密接に関係するものとなっています。

##### トレースの公式

BFGS更新後の行列のトレースは、次のような明示的な公式で与えられます。

**Proposition 12** ({\citep[(6.44)]{nocedal1999numerical}})

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ をBFGS更新とする。このとき

```math
\begin{equation*}
\mathrm{tr}(B_{+}) = \mathrm{tr}(B) - \frac{\lVert B s \rVert^2}{s^\top Bs} + \frac{\lVert y \rVert^2}{y^\top s}
\end{equation*}
```

が成り立つ。

<details>
<summary>Proof</summary>

BFGS更新則にトレースを適用すると

```math
\begin{equation*}
\mathrm{tr}(B_{+}) = \mathrm{tr}(B) - \mathrm{tr}\left(\frac{Bss^\top B}{s^\top Bs}\right) + \mathrm{tr}\left(\frac{yy^\top}{y^\top s}\right).
\end{equation*}
```

第2項について

```math
\begin{equation*}
\mathrm{tr}\left(\frac{Bss^\top B}{s^\top Bs}\right) = \frac{1}{s^\top Bs}\mathrm{tr}((Bs)(Bs)^\top) = \frac{\lVert B s \rVert^2}{s^\top Bs}.
\end{equation*}
```

第3項について

```math
\begin{equation*}
\mathrm{tr}\left(\frac{yy^\top}{y^\top s}\right) = \frac{1}{y^\top s}\mathrm{tr}(yy^\top) = \frac{\lVert y \rVert^2}{y^\top s}.
\end{equation*}
```

以上より所望の式が得られます。
\myQED

</details>

##### 行列式の公式

BFGS更新後の行列式も閉形式で与えられます。

**Proposition 13** ({\citep[(6.45)]{nocedal1999numerical}})

$B_{+} = B - \frac{Bss^\top B}{s^\top Bs} + \frac{yy^\top}{y^\top s}$ をBFGS更新とし、$B$ が正則であるとする。このとき

```math
\begin{equation*}
\det(B_{+}) = \det(B) \frac{y^\top s}{s^\top Bs}
\end{equation*}
```

が成り立つ。

<details>
<summary>Proof</summary>

[行列式補題](https://en.wikipedia.org/wiki/Matrix_determinant_lemma)より、
BFGS更新のランク2表現を思い出すと、
$U, C, V$ は次を満たします。

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
\myQED

</details>

### BFGSとDFPの比較

これまで何度か言及したように、BFGSとDFPは双対関係にあります。
しかし、実際の最適化問題に適用すると実用上の効率は大きく異なることが知られています。
Powellによる解析~\citep{powellHowBadAre1986}では、単純な二次元二次関数に対する両手法の挙動を調べて、この非対称性を検討しました。
漸近的には同程度に振る舞うと示唆されることが多い両手法について、Powellはその実用上の効率が大きく異なること、特に近似ヘッセ行列が真のヘッセ行列から遠い場合に差が顕著であることを明らかにしました。ここでは、その数値実験を再現します。

#### 問題設定

Powellの問題設定に従い、次の二次関数を考えます。

```math
\begin{equation*}
f(x, y) = \frac{1}{2}(x^2 + y^2).
\end{equation*}
```

BFGSとDFPのどちらも各反復で固定ステップサイズ $\alpha_k = 1$ を用いました。

また、初期のヘッセ近似 $B_0$ は固有値 1 と $\lambda_1$ を持つように選びました。$\lambda_1$ は初期近似の誤差の程度を表しています。
初期点 $x_0$ は次のように選びます。

```math
\begin{equation*}
\theta = \arctan(\sqrt{\lambda_1}), \quad x_0 = \begin{bmatrix}\cos(\theta) \\ \sin(\theta)\end{bmatrix}.
\end{equation*}
```

これはPowellの元の解析に沿ったものです。
この選択の詳細は~\citep{powellHowBadAre1986}を参照してください。

反復は現在点のノルムが初期ノルムに対する許容値を下回るまで続けました。
そして、各 $\lambda_1$ に対して収束に要する反復回数を記録しました。

#### 数値結果

数値結果を Table 1 に示します。
これはPowellの元論文の表と部分的に一致しています。
収束挙動は初期固有値 $\lambda_1$ に強く依存することが分かります。

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

(Table 1 \ifEn Convergence comparison between BFGS and DFP methods for different initial eigenvalues $\lambda_1$. \else 初期固有値 $\lambda_1$ に対するBFGSとDFPの収束比較。)

\Cref{fig:bfgs_dfp_100} と Fig. 9 は特定の $\lambda_1$ に対する反復軌跡を示しています。
これらの図は、同一の初期点から最小点(原点)へ向かう二つの手法の進み方を可視化し、収束速度と経路の違いを明確に示しています。

![../imgs/quasi_newton/bfgs_vs_dfp_100.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/bfgs_vs_dfp_100.png)

(Fig. 8 $\lambda_1 = 100$ におけるBFGSとDFPの反復軌跡。BFGSは10回で収束する一方、DFPは107回を要し、固有値誤差が大きい場合にBFGSが優位であることを示しています。)

![../imgs/quasi_newton/bfgs_vs_dfp_0.1.png](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/bfgs_vs_dfp_0.1.png)

(Fig. 9 $\lambda_1 = 0.1$ におけるBFGSとDFPの反復軌跡。どちらも素早く収束し、DFPがBFGSよりわずかに速くなっています。)

#### 解析と議論

数値結果はBFGSとDFPの間に顕著な非対称性があることを示しています。
$\lambda_1 > 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過大評価する場合、BFGS は大幅に高い効率を示しました。
一方で、 $\lambda_1 < 1$ のとき、すなわち初期ヘッセ近似が真の曲率を過小評価する場合には、DFPはBFGSよりわずかに良い性能を示しましたが、その差は非常に小さいものでした。
この傾向の逆転は理論的な対称性から予測されるものですが、その差の大小は注目に値します。

##### ヘッセ行列の補正における非対称性

両者の性能の非対称性は、誤った固有値を補正する仕方の根本的な違いに起因します。
大事な考察として、過大な固有値の補正が過小な固有値の補正よりも重要であることが挙げられます。

ヘッセ固有値が過大評価されると、アルゴリズムは過度に保守的なステップを取り、最適化の進みが遅くなります。
この誤差を補正するには、更新則が大きな固有値を 1 へ縮小する必要があります。
BFGS更新はこのようなタスクには強い効果を発揮しますが、DFP更新は苦手です。

一方で、ヘッセ固有値が過小評価されると、アルゴリズムは大胆なステップを取りますが、その過小評価は自動で修正されやすいです。
新しい点での勾配計算が近似の改善を促す為、このような過小評価の補正は本質的に容易で、必要な反復回数も少なくて済みます。

このような差異が、BFGS更新とDFP更新の性能の非対称性を生み出していると考えることが出来ます。

### 記憶制限BFGS (L-BFGS)

以下では、準ニュートン法を大規模最適化問題へ拡張する際に重要となる、記憶制限準ニュートン法について説明します。
通常の準ニュートン法では、近似ヘッセ行列 $B_k$ またはその逆行列 $H_k$ を密行列として陽に保存・更新するため、$n$ 変数に対して $\mathcal{O}(n^2)$ のメモリを要します。
一方で、記憶制限準ニュートン法では、最新の $m$ 組のベクトルペア $\lbrace(s_i,y_i)\rbrace$ という限られた情報のみを保持して、その情報だけから近似ヘッセ行列に関する計算を行います。
この工夫により空間計算量は $\mathcal{O}(nm)$ に減少し、$m$ が小さな定数(通常は $m\le 10$) のとき大幅な改善となります。
特に、BFGS更新の記憶制限版であるL-BFGS法~\citep{liuLimitedMemoryBFGS1989a}は、特にその代表的な手法です。
このBFGS更新の場合に注目し、限られた情報だけを用いて準ニュートン方向 $d_k = -H_k g_k$ を空間計算量・時間計算量の両面で効率的に計算する方法を示します。

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

準ニュートン法では、逆行列 $H_m$ と与えられたベクトル $q \in \mathbb{R}^n$ に対して $r = H_m q$ が効率的に計算できることが、アルゴリズムにおいて重要です。
この計算は長さ $m$ の短いループを二回回すだけで実行でき、L-BFGS のtwo-loop recursionと呼ばれるアルゴリズムとして知られています~\citep[Algorithm 7.4]{nocedal1999numerical}。
このアルゴリズムは $\mathcal{O}(nm)$ の時間・空間計算量を要します。

![999_two_loop_recursion](https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/999_two_loop_recursion.png)

この二重ループ再帰の出力が確かに $r = H_m q$ を計算していることを、以下では確認します。

**Proposition 14**

二重ループ再帰アルゴリズムの出力は $r = H_m q$ を満たす。

<details>
<summary>Proof</summary>

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
\myQED

</details>

従って、このtwo-loop recursionによって、行列 $H_m$ を陽に構成せずに、ベクトルに対する作用を正確に評価できることがわかります。

#### 初期スケーリング

L-BFGS 法の重要な要素は初期行列 $H_0$ の選択です。
広く使われるのは、単位行列にスケーリングを施したものです。

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

この選択は次の議論によって正当化できます~\citep{liuLimitedMemoryBFGS1989a,shannoMatrixConditioningNonlinear1978}。
目的関数 $f$ が二回連続微分可能であると仮定し、最新のステップに沿った平均ヘッセ行列を考えます。

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

これは、ベクトル $\bar{G}^{1/2} s_{m-1}$ に関する行列 $\bar{G}$ のRayleigh商の逆数です。
従って、このスケーリングはこれらの方向に沿った平均逆曲率を大まかに近似しています。

さらにこの $\gamma$ の選択は Barzilai--Borwein 法の短ステップサイズ~\citep{barzilaiTwoPointStepSize1988} と一致しており、L-BFGS の初期化と古典的なステップ長選択戦略の密接な関係を示しています。

