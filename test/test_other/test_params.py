import numpy as np
import pytest

from qnlab.parameter import (
    HamaguchiParameter,
    LineParameter,
    NtqnParameter,
    OwlParameter,
)
from qnlab.util import linesearch


def test_line_parameter_defaults_include_eps():
    param = LineParameter(5, {})
    assert param.eps > 0
    assert param.m == 10
    assert param.max_iterations == 15000
    assert param.max_evaluations == 30000


def test_ntqn_rejects_negative_gtol():
    with pytest.raises(ValueError):
        NtqnParameter(2, {"gtol": np.float64(-1.0)})


def test_owl_requires_backtracking_linesearch():
    with pytest.raises(ValueError):
        OwlParameter(3, {"orthantwise_c": np.float64(1.0)})

    param = OwlParameter(
        3,
        {
            "orthantwise_c": np.float64(1.0),
            "linesearch_kind": linesearch.LINESEARCH_BACKTRACKING,
        },
    )
    assert param.linesearch_kind == linesearch.LINESEARCH_BACKTRACKING


def test_hamaguchi_uses_shared_eps():
    param = HamaguchiParameter(4, {})
    assert param.eps > 0


if __name__ == "__main__":
    test_line_parameter_defaults_include_eps()
    test_ntqn_rejects_negative_gtol()
    test_owl_requires_backtracking_linesearch()
    test_hamaguchi_uses_shared_eps()
    print("All tests passed.")
