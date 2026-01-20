import numpy as np

from qnlab.problem.bdqrtic import BdqrticProblem
from qnlab.problem.cutest import CUTEstQNProblem


def test_bdqrtic():
    prob = CUTEstQNProblem("BDQRTIC")
    prob2 = BdqrticProblem()
    assert prob.n == prob2.n
    assert np.allclose(prob.x0, prob2.x0)
    for _ in range(3):
        x = np.random.uniform(-1, 1, prob.n)
        assert np.allclose(prob.f(x, count=False), prob2.f(x, count=False))
        assert np.allclose(prob.g(x, count=False), prob2.g(x, count=False))
    print("Bdqrtic problem test passed successfully.")


if __name__ == "__main__":
    test_bdqrtic()
