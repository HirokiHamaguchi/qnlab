# $C'_0=C'_1 \sqrt{G_k} - C'_2 \ln G_k$
# の解が、
# $\frac{4{C'_2}^2}{{C'_1}^2}\qty[W_k\qty(-\tfrac{C'_1}{2C'_2}e^{-\tfrac{C'_0}{2C'_2}})]^2$
# であるかを確認する。

import math

import numpy as np
from scipy.special import lambertw

# 検証する定数の組み合わせ（例としていくつか用意）
C0_values = np.random.uniform(0.1, 10.0, 5)  # C0 の値
C1_values = np.random.uniform(0.1, 5.0, 5)  # C1 の値
C2_values = np.random.uniform(0.1, 5.0, 5)  # C2 の値

tolerance = 1e-8


def lhs(C0, C1, C2, Gk):
    return C1 * np.sqrt(Gk) - C2 * np.log(Gk)


def rhs(C0, C1, C2):
    z = -(C1 / (2 * C2)) * np.exp(-C0 / (2 * C2))
    if not (-1 / math.e < z < 0):
        return None
    W = lambertw(z, k=-1)
    Gk = (4 * C2**2 / C1**2) * W**2
    return Gk.real  # Lambert W は複素数を返すので実部を取る


for C0 in C0_values:
    for C1 in C1_values:
        for C2 in C2_values:
            Gk_calc = rhs(C0, C1, C2)
            if Gk_calc is None:
                print(f"無効な値 (C0'={C0}, C1'={C1}, C2'={C2})")
                continue

            lhs_val = lhs(C0, C1, C2, Gk_calc)
            diff = np.abs(lhs_val - C0)

            print(
                f"C0': {C0}, C1': {C1}, C2': {C2}, Gk = {Gk_calc:.8f}, lhs = {lhs_val:.8f}, expected = {C0:.8f}, diff = {diff:.2e}"
            )
            if diff < tolerance:
                print("  ✅ 検証成功")
            else:
                print("  ❌ 差が大きい")
