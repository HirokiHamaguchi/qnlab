import numpy as np

from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.gauss1ls import Gauss1lsProblem


def test_gauss1ls():
    prob = CUTEstQNProblem("GAUSS1LS")
    prob2 = Gauss1lsProblem()
    assert prob.n == prob2.n
    assert np.allclose(prob.x0, prob2.x0)
    for _ in range(3):
        x = prob.x0 + np.random.randn(prob.n) * 0.1
        assert np.allclose(prob.f(x, count=False), prob2.f(x, count=False))
        assert np.allclose(prob.g(x, count=False), prob2.g(x, count=False))
    print("Gauss1ls problem test passed successfully.")


if __name__ == "__main__":
    test_gauss1ls()
