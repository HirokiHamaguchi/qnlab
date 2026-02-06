import math
from itertools import product

import numpy as np
from scipy.special import lambertw

# 検証する定数の組み合わせ（例としていくつか用意）
C1_values = 10 ** np.linspace(-5, 5, 11)
C2_values = 10 ** np.linspace(-5, 5, 11)


def Gk(C1, C2):
    W = lambertw(-C1 / (4 * C2), k=-1)
    if -1 / math.e <= -C1 / (4 * C2) < 0:
        assert W.imag == 0 and W.real < 0
        return (-4 * C2 / C1 * W.real) ** 2
    else:
        return None


for C1, C2 in product(C1_values, C2_values):
    Gk_value = Gk(C1, C2)
    if Gk_value is None:
        continue
    print(f"C1: {C1}, C2: {C2}, Gk: {Gk_value}")

print(Gk(1, math.e / 4 + 0.1))
print(Gk(1, math.e / 4 + 0.01))
print(Gk(1, math.e / 4 + 0.001))
print(Gk(1, math.e / 4 + 0.0001))
print(Gk(1, math.e / 4 + 0.00001))
print(Gk(1, math.e / 4 + 0.000001))
print(Gk(1, math.e / 4 + np.finfo(float).eps))
print(Gk(1, math.e / 4 - 0.001))
