import numpy as np
import numpy.typing as npt

from qnlab.parameter import OwlParameter


def owlqn_pseudo_gradient(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    n: int,
    param: OwlParameter,
) -> npt.NDArray[np.float64]:
    """Computes the pseudo-gradient for orthant-wise L1 regularization.

    Returns:
        numpy.ndarray: Pseudo-gradient values.
    """
    pg = np.copy(g)
    for i in range(param.orthantwise_start, param.orthantwise_end):
        if x[i] < 0.0:
            # Differentiable.
            pg[i] = g[i] - param.orthantwise_c
        elif x[i] > 0.0:
            # Differentiable.
            pg[i] = g[i] + param.orthantwise_c
        else:
            if g[i] < -param.orthantwise_c:
                # Take the right partial derivative.
                pg[i] = g[i] + param.orthantwise_c
            elif param.orthantwise_c < g[i]:
                # Take the left partial derivative.
                pg[i] = g[i] - param.orthantwise_c
            else:
                pg[i] = 0.0
    return pg
