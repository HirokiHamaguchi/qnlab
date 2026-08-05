import numpy as np
import numpy.typing as npt

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory


def compute_B(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    scale = lm.get_last().ys / lm.get_last().yy

    # B: Hessian approximation
    B = 1.0 / scale * np.eye(len(g))
    for item in lm:
        s = item.s
        y = item.y
        ss = np.dot(s, s)
        y_Bs = y - B @ s
        first = (np.outer(y_Bs, s) + np.outer(s, y_Bs)) / ss
        second = (np.inner(s, y_Bs) / (ss**2)) * np.outer(s, s)
        B += first - second

    return B


def compute_H(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    # I don't know why, but the B times H does not match identity.
    # We omit the implementation for now.
    # Do you know the correct formula?
    raise NotImplementedError("PSB inverse Hessian not implemented yet.")


class PSBUpdateRule(BaseUpdateRule):
    @staticmethod
    def compute_dir(x, g, lm) -> npt.NDArray[np.float64]:
        B = compute_B(g, lm)
        return np.linalg.solve(B, -g).astype(np.float64)

    @staticmethod
    def compute_dir_reg(x, g, lm, mu) -> npt.NDArray[np.float64]:
        B = compute_B(g, lm)
        return np.linalg.solve(B + mu * np.eye(g.shape[0]), -g).astype(np.float64)

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B = compute_B(g, lm)
        H = np.linalg.inv(B).astype(np.float64)
        return B, H


if __name__ == "__main__":
    import numpy as np

    def psb_update_b(B, s, y):
        """B型 (ヘッセ行列近似) のPSB更新"""
        r = y - np.dot(B, s)
        s_norm_sq = np.dot(s, s)
        term1 = (np.outer(r, s) + np.outer(s, r)) / s_norm_sq
        term2 = (np.dot(r, s) * np.outer(s, s)) / (s_norm_sq**2)
        return B + term1 - term2

    def psb_update_h(H, s, y, tol=1e-12):
        Delta = (y @ H @ y - s @ y) * s @ H @ s - (y @ H @ s) ** 2
        uu = np.outer(H @ y - s, H @ y - s)
        vv = np.outer(H @ s, H @ s)
        uv_plus_vu = np.outer(H @ y - s, H @ s) + np.outer(H @ s, H @ y - s)
        Hbar = (
            H
            - (s @ H @ s * uu - y @ H @ s * uv_plus_vu + (y @ H @ y - s @ y) * vv)
            / Delta
        )
        return Hbar

    # 1. 初期化 (B = H = I)
    n = 3
    B_k = np.eye(n)
    H_k = np.eye(n)

    # 2. テストデータの生成 (s, y)
    # 注: PSBはセカント条件 B*s = y (または H*y = s) を満たすように更新します
    np.random.seed(42)
    s = np.random.randn(n)
    y = np.random.randn(n)

    # 3. それぞれの更新を実行
    B_next = psb_update_b(B_k, s, y)
    H_next = psb_update_h(H_k, s, y)

    # 4. 検証: B_next の逆行列が H_next と一致するか
    B_next_inv = np.linalg.inv(B_next)

    print("B_next * H_next (Identity になるべき):")
    print(np.round(np.dot(B_next, H_next), 10))

    is_correct = np.allclose(B_next_inv, H_next)
    print(
        f"\n検証結果: {'一致しました (正しい)' if is_correct else '一致しませんでした'}"
    )
