import numpy as np
import numpy.typing as npt

from qnlab.update.bfgs import BFGSUpdateRule
from qnlab.update.dfp import DFPUpdateRule
from qnlab.update.psb import PSBUpdateRule
from qnlab.update.sr1 import SR1UpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method


def get_direction(
    method: Method,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lm: QuasiNewtonMemory,
) -> npt.NDArray[np.float64]:
    assert len(lm) > 0, "Memory is empty. Cannot compute direction."
    if method.update == "bfgs":
        return BFGSUpdateRule.compute_dir(x, g, lm)
    elif method.update == "dfp":
        return DFPUpdateRule.compute_dir(x, g, lm)
    elif method.update == "sr1":
        return SR1UpdateRule.compute_dir(x, g, lm)
    elif method.update == "psb":
        return PSBUpdateRule.compute_dir(x, g, lm)
    else:
        raise ValueError(f"Unknown update method: {method.update}")


def get_direction_reg(
    method: Method,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lm: QuasiNewtonMemory,
    mu: np.float64,
) -> npt.NDArray[np.float64]:
    if len(lm) == 0:
        return lm.zero_memory_direction(g, mu)
    if mu == 0.0:
        return get_direction(method, x, g, lm)
    if method.update == "bfgs":
        return BFGSUpdateRule.compute_dir_reg(x, g, lm, mu)
    elif method.update == "sr1":
        return SR1UpdateRule.compute_dir_reg(x, g, lm, mu)
    elif method.update == "psb":
        return PSBUpdateRule.compute_dir_reg(x, g, lm, mu)
    else:
        raise ValueError(f"Unknown reg update method: {method.update}")


def get_direction_scaled_reg(
    method: Method,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lm: QuasiNewtonMemory,
    mu: np.float64,
    scale: np.float64,
    scale_floor: np.float64,
) -> npt.NDArray[np.float64]:
    """Compute a direction for ``scale_floor*I + scale*B + mu*I``."""
    if len(lm) == 0:
        return -g / (scale_floor + scale + mu)
    if method.update != "bfgs":
        raise ValueError("Scaled regularization is implemented only for BFGS.")
    return (
        BFGSUpdateRule.compute_dir_reg(
            x,
            g,
            lm,
            (mu + scale_floor) / scale,
        )
        / scale
    )


def check_direction(
    method: Method,
    n: int,
    g: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    lm: QuasiNewtonMemory,
):
    if method.update == "bfgs":
        return BFGSUpdateRule.check(n, g, d, lm)
    elif method.update == "dfp":
        return DFPUpdateRule.check(n, g, d, lm)
    elif method.update == "sr1":
        return SR1UpdateRule.check(n, g, d, lm)
    elif method.update == "psb":
        return PSBUpdateRule.check(n, g, d, lm)
    else:
        raise ValueError(f"Unknown check method: {method.update}")
