import random

import scipy.optimize


def f_linear(x):
    return x


def jac_linear(x):
    return 1.0


def f_quad(x):
    return -(x**2)


def jac_quad(x):
    return -2 * x


def f_random(x):
    return random.random()


def jac_random(x):
    return random.random()


def f_nan(x):
    if random.random() < 0.5:
        return -x
    else:
        return float("nan")


def jac_nan(x):
    if random.random() < 0.5:
        return -1.0
    else:
        return float("nan")


def f_inf(x):
    if random.random() < 0.5:
        return -x
    else:
        return float("inf")


def jac_inf(x):
    if random.random() < 0.5:
        return -1.0
    else:
        return float("inf")


functions = [
    (f_linear, jac_linear),
    (f_quad, jac_quad),
    (f_random, jac_random),
    (f_nan, jac_nan),
    (f_inf, jac_inf),
]

for f, jac in functions:
    print("-" * 20)
    print(f"Testing function: {f.__name__}")
    try:
        result = scipy.optimize.minimize(
            f,
            x0=1.0,
            method="trust-constr",
            jac=jac,
        )
        print(f"Function: {f.__name__}, Result: {result}")
    except Exception as e:
        print(f"Function: {f.__name__}, Error: {e}")
