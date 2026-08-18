import numpy as np
import pytest

from qnlab.parameter import (
    LineParameter,
    NtqnParameter,
    NTRQNParameter,
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


def test_ntrqn_uses_shared_eps():
    param = NTRQNParameter(4, {})
    assert param.eps > 0
    assert np.isinf(param.restart_threshold)
    assert param.offo_squared_offset == np.float64(1e-20)


def test_ntqn_default_termination_can_disable_wrapper_stop():
    param = NtqnParameter(2, {"terminate": 3, "stop_at_gtol": 0})
    assert param.terminate == 3
    assert param.stop_at_gtol == 0


def test_ntrqn_rejects_invalid_algorithm_options():
    with pytest.raises(ValueError):
        NTRQNParameter(2, {"offo_squared_offset": np.float64(0.0)})
    with pytest.raises(ValueError):
        NTRQNParameter(2, {"force_offo": 2})


if __name__ == "__main__":
    test_line_parameter_defaults_include_eps()
    test_ntqn_rejects_negative_gtol()
    test_owl_requires_backtracking_linesearch()
    test_ntrqn_uses_shared_eps()
    test_ntqn_default_termination_can_disable_wrapper_stop()
    test_ntrqn_rejects_invalid_algorithm_options()
    print("All tests passed.")
