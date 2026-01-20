import cvxpy as cp
import numpy as np
import scipy.linalg


def compare_bfgs_cvxpy():
    n = 5
    for seed in range(5):
        np.random.seed(seed)
        A = np.random.standard_normal((n, n))
        B_k = A @ A.T + n * np.eye(n)  # ensure PD
        s = np.random.standard_normal(n)
        y = np.random.standard_normal(n)
        if np.dot(s, y) >= 0:
            break
    else:
        # if np.dot(s, y) < 0,
        # s^\top B_k s = s^\top y < 0,
        # which contradicts s^\top B_k s > 0
        raise ValueError("Could not find a suitable seed for s and y.")

    Bs = B_k @ s
    ys = np.dot(y, s)
    B_bfgs = B_k - np.outer(Bs, Bs) / np.dot(s, Bs) + np.outer(y, y) / ys

    B_k_inv_sqrt = cp.Constant(scipy.linalg.sqrtm(np.linalg.inv(B_k)))
    B_cp = cp.Variable((n, n), PSD=True)

    def objective_function(B):
        arg = B_k_inv_sqrt @ B @ B_k_inv_sqrt
        return cp.trace(arg) - cp.log_det(arg)

    prob = cp.Problem(
        cp.Minimize(objective_function(B_cp)),
        [B_cp @ s == y],
    )
    prob.solve(eps=1e-7, solver=cp.SCS)
    B_cvxpy = B_cp.value
    if B_cvxpy is None:
        raise ValueError("CVXPY did not find a solution.")

    return B_bfgs, B_cvxpy, s, y


def test_bfgs_with_cvxpy():
    B_bfgs, B_cvxpy, s, y = compare_bfgs_cvxpy()
    assert np.allclose(B_bfgs @ s, y, atol=1e-5)
    assert np.allclose(B_cvxpy @ s, y, atol=1e-5)
    assert np.allclose(B_bfgs, B_cvxpy, atol=1e-5)


if __name__ == "__main__":
    test_bfgs_with_cvxpy()
    print("All tests passed successfully.")
