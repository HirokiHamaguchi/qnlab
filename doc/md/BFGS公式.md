# BFGS公式

## Derivation of the Update Formula

### Secant Condition

QUASI-NEWTON METHODS, MOTIVATION AND THEORY
Section 7

$f\colon \mathbb{R}^n \to \mathbb{R}$ が $C^2$ 級だとします。

$B \approx \nabla^2 f(x)$ が与えられた時、$\bar{x}=x+s$ での $\nabla^2 f(\bar{x})$ の近似 $\bar{B}$ を得たいです。
ヘッセ行列が対称であることも考慮して、$B=B^\top$,$\bar{B}=\bar{B}^\top$ だとします。

$\nabla f(x)$ のテイラー近似によって、

$$
\begin{align*}
    \nabla f(x) &\approx \nabla f(\bar{x}) + \nabla^2 f(\bar{x})(x - \bar{x})\\
                &\approx \nabla f(\bar{x}) + \bar{B}(x - \bar{x})
\end{align*}
$$

が成立して欲しいので、
$s \coloneqq \bar{x} - x$, $y \coloneqq \nabla f(\bar{x}) - \nabla f(x)$ とおくと、

$$
\begin{align*}
     &\bar{B}(\bar{x} - x) = \nabla f(\bar{x}) - \nabla f(x)\\
\iff &\bar{B}s = y
\end{align*}
$$

だと望ましいです。
これはセカント条件(secant condition)と呼ばれ、quasi-Newton equation とも呼ばれます。

### SR1 (Symmetric Rank-One) Update

セカント条件を満たすような $\bar{B}$ をランク1更新で与えることを考えます。
つまり、$z \in \mathbb{R}^n$, $c \in \mathbb{R}^n$ with $\langle c, s \rangle \neq 0$ を用いて

$$
\bar{B} = B + z c^\top
$$

と書けるとします。この時、セカント条件を満たすには

$$
\begin{align*}
         & \bar{B}s = y\\
    \iff & B s + z c^\top s = y\\
    \iff & z = \frac{y - Bs}{\langle c, s \rangle} \\
    \iff & \bar{B} = B + \frac{(y - Bs)c^\top}{\langle c, s \rangle}
\end{align*}
$$

であることが必要です。$\bar{B}$ が対称であることより、

$$
\bar{B}_\mathrm{SR1} = B + \frac{(y - Bs)(y - Bs)^\top}{\langle y-Bs, s \rangle}
$$

という更新則が得られ、**Symmetric Rank-One (SR1) update** と呼ばれます。

なお、$\langle y - Bs, s \rangle = 0$ の場合には更新をskipすることが一般的である。

### PSB Update

対称行列 $B \in \mathbb{R}^n$ に対して、SR1と同様に

$$
C_1 \coloneqq B + \frac{(y - Bs)c^\top}{\langle c, s \rangle}
$$

と更新することを考えます。この $C_1$ は一般に対称行列ではないので、

$$
C_2 = \frac{C_1 + C_1^\top}{2}
$$

と対称化を行います。すると今度は $C_2$ がセカント条件を必ずしも満たさないので、上記の手続きを繰り返します。つまり、

$$
\begin{cases}
C_0 = B \\
C_{2k+1} = C_{2k} + \frac{(y - C_{2k}s)c^\top}{\langle c, s \rangle} \\
C_{2k+2} = \frac{C_{2k+1} + C_{2k+1}^\top}{2}
\end{cases}
$$

で $\{ C_k \}_{k=0}^{\infty}$ を定義します。

この時、$C_k$ は

$$
\lim_{k \to \infty} C_k = B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{\langle c, s \rangle} - \frac{\langle y - Bs, s \rangle}{\langle c, s \rangle^2} cc^\top
$$

に収束します。

<details>
<summary>Proof of Convergence</summary>

$$
\begin{align*}
    C_{2k+2} &= \frac{C_{2k+1} + C_{2k+1}^\top}{2} \\
             &= C_{2k} + \frac{1}{2} \frac{(y - C_{2k}s)c^\top + c(y - C_{2k}s)^\top}{\langle c, s \rangle}
\end{align*}
$$

より、

$$
\lim_{k\to\infty} C_{k} = B + \sum_{k=0}^{\infty} \frac{1}{2} \frac{(y - C_{2k}s)c^\top + c(y - C_{2k}s)^\top}{\langle c, s \rangle}
$$

です。また、

$$
\begin{align*}
       & y-C_{2k+2}s \\
    ={}& y - C_{2k}s - \frac{1}{2} \frac{(y - C_{2k}s)c^\top s + c(y - C_{2k}s)^\top s}{\langle c, s \rangle} \\
    ={}& \left( \frac{1}{2} \left(I - \frac{cs^\top}{\langle c, s \rangle} \right) \right) (y - C_{2k}s)
\end{align*}
$$

であり、線形写像 $P \coloneqq \frac{1}{2} \left(I - \frac{cs^\top}{\langle c, s \rangle} \right)$ の固有値は $0$ か $1/2$ であることは簡単に確かめられるので、

$$
\begin{align*}
    &\sum_{k=0}^{\infty} (y-C_{2k}s)\\
 ={}& \sum_{k=0}^{\infty} P^k (y - C_0 s)\\
 ={}& (I-P)^{-1} (y - Bs)\\
 ={}& \left( 2I - \frac{cs^\top}{\langle c, s \rangle}  \right) (y - Bs)
\end{align*}
$$

と導かれます。最後の等式は[Sherman--Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)を使いました。

以上より、
$$
\begin{align*}
      {}& \lim_{k\to\infty} C_k \\
    = {}& B + \sum_{k=0}^{\infty} \frac{1}{2} \frac{(y - C_{2k}s)c^\top + c(y - C_{2k}s)^\top}{\langle c, s \rangle}\\
    ={} & B + \frac{(y - Bs)c^\top+c(y - Bs)^\top}{\langle c, s \rangle} - \frac{1}{2} \frac{cs^\top(y-Bs)c^\top + c (y-Bs)^\top s c^\top}{\langle c, s \rangle^2}\\
    ={} & B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{\langle c, s \rangle} - \frac{\langle y - Bs, s \rangle}{\langle c, s \rangle^2} cc^\top
\end{align*}
$$
が得られます。 (証明終わり)

</details>

これに $c=s$ を代入すると、

$$
\bar{B}_{\mathrm{PSB}} = B + \frac{(y - Bs)s^\top + s(y - Bs)^\top}{\langle s, s \rangle} - \frac{\langle y - Bs, s \rangle}{\langle s, s \rangle^2} ss^\top
$$

という更新則が得られ、**Powell Symmetric Broyden (PSB) update** と呼ばれます。

Powell (1970d) proposed a technique to derive a **double-rank** version of Broyden's method.
Dennis (1972) showed that Powell's technique can derive most well-known quasi-Newton updates.

### DFP Update

先程のPSBの更新則では、

$$
\bar{B} = B + \frac{(y - Bs)c^\top + c(y - Bs)^\top}{\langle c, s \rangle} - \frac{\langle y - Bs, s \rangle}{\langle c, s \rangle^2} cc^\top
$$

に $c=s$ を代入しましたが、別の $c$ を取ることを考えます。
具体的には、$\bar{B}$ が正定値になるような $c$ を選ぶことを考えます。

結論から言うと、$y=s$ を代入するとこれは正定値になり、

$$
\bar{B}_{\mathrm{DFP}} = B + \frac{(y - Bs)y^\top + y(y - Bs)^\top}{\langle y, s \rangle} - \frac{\langle y - Bs, s \rangle}{\langle y, s \rangle^2} yy^\top
$$

が得られます。これを**Davidon--Fletcher--Powell (DFP) update** と呼びます。

**Lemma 7.4.**

対称行列 $A \in \mathbb{R}^n$が
$$
\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_n
$$
と固有値を持つとします。
ある $u \in \mathbb{R}^n$ に対して $A^* = A + \sigma uu^T$ とし、その固有値を$\lambda_i^*$とします。
この時、$\sigma > 0$ ならば
$$
\lambda_1 \leq \lambda_1^{*} \leq \lambda_2 \dots \leq \lambda_n \leq \lambda_n^{*}
$$
であり、$\sigma < 0$ ならば
$$
\lambda_1^{*} \leq \lambda_1 \leq \lambda_2^* \dots \leq \lambda_n^{*} \leq \lambda_n
$$
が成り立つ。

Cauchy’s Interlacing Theoremの亜種として知られているようです。

https://simple-complexities.github.io/eigenvalue/interlacing/2020/02/10/interlacing.html

**定理7.5.**
正定値対称行列 $B \in \mathbb{R}^n$、$c, s, y \in \mathbb{R}^n$ で $(c, s) \neq 0$ とします。このとき、(7.9) により定義される $B$ が正定値であるのは、$\det B > 0$ である場合に限ります。

**証明.** $B$ が正定値であれば、明らかに $\det B > 0$ です。逆を示すには、まず $B$ を以下のように書けることに注目します：
$$
B = \bar{B} + vw^T + wv^T,
$$
ここで $v = c$ であり、
$$
w = \frac{y - \bar{B}s}{2(c, s)}
$$
です。したがって、
$$
B = \bar{B} + 2\left[(v + w)(v + w)^T - (v - w)(v - w)^T\right]
$$
と表せるので、$B$ は $\bar{B}$ に2つの対称なランク1行列の和を加えた形で表現されます。

$\bar{B}$ が正定値であれば、補題7.4より、$B$ は高々1つの非正の固有値しか持ち得ません。よって、$\det B > 0$ であれば、全ての固有値は正となり、$B$ は正定値です。

定理7.5から、(7.9) により定義される更新式が (7.1) および (7.14) を満たすためには、$B$ が対称かつ正定値であるときに $\det B > 0$ が成り立つ必要があります。この要件を満たす $c$ の選び方を知るためには、$\det B$ の表現が必要です。

**補題7.6.** $u_i \in \mathbb{R}^n$（$i = 1, 2, 3, 4$）とします。このとき
$$
\det(I + u_1 u_2^T + u_3 u_4^T) = (1 + (u_1, u_2))(1 + (u_3, u_4)) - (u_1, u_4)(u_2, u_3)
$$
が成り立ちます。

**証明.** この結果の証明は Pearson (1969) にありますが、以下は代替の議論です。

一時的に $(u_1, u_2) \neq -1$ と仮定します。このとき $I + u_1 u_2^T$ は可逆であり、
$$
I + u_1 u_2^T + u_3 u_4^T = (I + u_1 u_2^T)\left(I + (I + u_1 u_2^T)^{-1} u_3 u_4^T\right)
$$
と書けます。この結果は補題4.2および4.4を用いることで導かれます。$(u_1, u_2) \neq -1$ の場合に成り立つので、連続性の議論により一般にも成り立つことが示されます。

(7.9) に補題7.6を適用します。いくつかの代数的操作により、次が得られます：
$$
\begin{equation*}
\det B = \det \bar{B} \left[\frac{(c, H y)^2 - (c, H c)(y, H y) + (c, H c)(y, s)}{(c, s)^2}\right]
\end{equation*}
$$
ここで $H = \bar{B}^{-1}$ です。$\bar{B}$ が正定値であると仮定し、$v = H^{1/2} y$、$w = H^{1/2} c$ とおくと、
\begin{equation}
\det B = \det \bar{B} \left[\frac{(v, w)^2 - \|v\|^2 \|w\|^2 + \|w\|^2 (y, s)}{(c, s)^2}\right] \tag{7.16}
\end{equation}
となります。定理7.5により、$B$ が正定値であるのは次のときに限ります：
\begin{equation}
\|w\|^2 (y, s) > \|v\|^2 \|w\|^2 - (v, w)^2. \tag{7.17}
\end{equation}

この不等式 (7.17) を自然に満たす最も簡単な方法は、$w$ を $v$ のスカラー倍にすることです。この場合、(7.17) は単に $(y, s)$ が正であることを要求するだけになります。このとき、$c$ は $y$ のスカラー倍となり、(7.9) は Davidon (1959) によって導入された更新式に帰着します。

## Derivation of the Inverse Update

$$
H' = H + \frac{(s - Hy)(s - Hy)^\top}{(s - Hy, y)}
$$

## 特徴づけ

### Characterization of the SR1

Theorem 7.1

> Let $A \in \mathbb{R}^n$ be a non-singular symmetric matrix,
> and set $y_k = As_k$ for $0 \leq k \leq m$
> where $\{s_0, \dots, s_m\}$ spans $\mathbb{R}^n$.
> Let $H_0$ be symmetric and for $k = 0, \dots, m$,
> generate the matrices
> $$
> H_{k+1} = H_k + \frac{(s_k - H_k y_k)(s_k - H_k y_k)^\top}{\langle s_k - H_k y_k, y_k \rangle}
> $$
> assuming that
> $$
> \langle s_k - H_k y_k, y_k \rangle \neq 0.
> $$
> Then $H_{m+1} = A^{-1}$.

$s_j=H_{k}y_j$ $(0 \leq j < k)$ を $k$ の帰納法で示します。

$k=1$ の場合、セカント条件より明らかです。

帰納法の仮定がある $k (\leq m)$ で成立する場合、任意の $0 \leq j < k$ に対して
$$
s_j = H_k y_j = H_k A s_j
$$
であり、
$$
\begin{align*}
    & (s_k - H_k y_k)^\top y_j\\
 ={}& s_k^\top y_j - y_k^\top H_k y_j\\
 ={}& s_k^\top A s_j - s_k^\top A H_k A s_j\\
 ={}& s_k^\top A s_j - s_k^\top A s_j\\
 ={}& 0
\end{align*}
$$
なので、
$$
\begin{align*}
    H_{k+1} y_j &= H_k y_j + \frac{s_k - H_k y_k}{\langle s_k - H_k y_k, y_k \rangle} (s_k - H_k y_k)^\top y_j \\
                &= s_j
\end{align*}
$$
が成立します。
$j=k$ の場合である $s_k=H_{k+1} y_k$ はセカント条件そのものです。

以上より、帰納法がまわり、特に $k=m+1$ として

$$
s_j = H_{m+1} y_j = H_{m+1} A s_j \quad (0 \leq j \leq m)
$$

が成立します。

$\mathrm{span}\{s_0, \dots, s_m\}=\mathbb{R}^n$ であるので、$I$ を単位行列として $H_{m+1}A = I \implies H_{m+1} = A^{-1}$ が成立します。よって、定理が示されました。

### **Lemma 7.2**

Let $B \in \mathcal{L}(\mathbb{R}^n)$ be symmetric and let $c, s, y \in \mathbb{R}^n$ with $\langle c, s \rangle \neq 0$. If the sequence $\{C_k\}$ is defined by (7.8) with $C_0 = B$, then $\{C_k\}$ converges to $\bar{B}$ as defined by (7.9).

*Proof.* Let $G_k = C_{2k}$. Then (7.8) shows

$$(7.10) \quad G_{k+1} = G_k + \frac{W_k c^\top + c W_k^\top}{2\langle c, s \rangle}, \quad \text{where } W_k = y - G_k s.$$

In particular, $W_{k+1} = P W_k$, where

$$P = I - \frac{1}{2} \frac{cc^\top}{\langle c, s \rangle}.$$

It is clear that $P$ has one zero eigenvalue and all other eigenvalues equal to $1/2$, so the Neumann lemma (e.g., Ortega and Rheinboldt (1970), p.45) implies that

$$(7.11) \quad \sum_{k=0}^\infty W_k = \sum_{k=0}^\infty P^k (y - Bs) = (I - P)^{-1}(y - Bs).$$

Since

$$\lim_{k \to \infty} G_k = B + \sum_{k=0}^\infty (G_{k+1} - G_k),$$

it follows from (7.10) and (7.11) that $\{G_k\}$ converges. Thus, by Lemma 4.2,

$$(I - P)^{-1} = 2 \left[I - \frac{1}{2} \frac{cc^\top}{\langle c, s \rangle}\right]^{-1},$$

and the limit of $\{G_k\}$ is $\bar{B}$ as defined by (7.9).

---

### **Theorem 7.3**

Let $B \in \mathcal{L}(\mathbb{R}^n)$ be symmetric, and let $c, s, y \in \mathbb{R}^n$ with $\langle c, s \rangle > 0$. Assume that $M \in \mathcal{L}(\mathbb{R}^n)$ is any nonsingular symmetric matrix such that $Mc = M^{-1}s$. Then $\bar{B}$ as defined by (7.9) is the unique solution to the problem:

$$(7.12) \quad \min \{ \|\bar{B} - B\|_{M, F} : \bar{B} \text{ symmetric}, \bar{B}s = y \},$$

where $\|\cdot\|_{M, F}$ is defined in (1.3).

*Proof.* Pre- and post-multiply the expression of $\bar{B}$ by $M$, and define $z = Mc = M^{-1}s$, $E = M(\bar{B} - B)M$. Then:

$$E = \frac{zz^\top + zz^\top}{(z, z)} - \frac{(z, z)}{(z, z)^2} zz^\top = 0,$$

and hence $\|E\|$ is minimized. The uniqueness follows from the strict convexity of the norm over symmetric matrices satisfying $Bs = y$.

---

### **Special Case: The PSB Update**

Powell (1970d) used the argument of Lemma 7.2 to obtain (7.9) in the case $c = s$. In this case, the resulting update is:

$$(7.13) \quad \bar{B} = B + \frac{(y - Bs)s^\top + s(y - Bs)^\top}{(y - Bs, s)} - \frac{(y - Bs, s)}{(s, s)^2} ss^\top.$$

This is known as the **Powell Symmetric Broyden (PSB)** update. Theorem 7.3 implies that $B_{\text{PSB}}$ is the unique solution to

$$\min \{\|\bar{B} - B\|_F : \bar{B} \text{ symmetric}, \bar{B}s = y \}.$$

Because of this property, if $A$ is any symmetric matrix with $y = As$, then

$$\|B - A\|_F^2 = \|B_{\text{PSB}} - B\|_F^2 + \|B_{\text{PSB}} - A\|_F^2.$$

This implies that $B_{\text{PSB}}$ is a good approximation to the Hessian.

Furthermore, for any symmetric $A$ and $B \in \mathcal{L}(\mathbb{R}^n)$,

$$B_{\text{PSB}} - A = P^\top(B - A)P + \frac{(y - As)s^\top + s(y - As)^\top}{(s, s)},$$

with $P = I - \frac{ss^\top}{(s, s)}$. Therefore, (1.2) shows that

$$\|B_{\text{PSB}} - A\|_F \leq \|B - A\|_F + 2\|y - As\|.$$

If $A = \nabla^2 f(x)$ and $\nabla^2 f$ is Lipschitz continuous with constant $K$ in an open convex set $D$, then Lemma 3.3 implies

$$\|B_{\text{PSB}} - \nabla^2 f(x)\|_F \leq \|B - \nabla^2 f(x)\|_F + 3K \|s\|,$$

whenever $x$ and $x + s$ lie in $D$. This shows that the approximation error of $B_k$ to $\nabla^2 f(x_k)$ grows linearly with $\|s_k\|$.

---

ご希望があれば、PDF用のLaTeX形式や、特定のセクションだけ抽出・整形することも可能です。
