from qnlab.parameter import NtqnParameter
from qnlab.problem.dixon_price import DixonPriceProblem
from qnlab.problem.powell import PowellProblem
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.problem.zakharov import ZakharovProblem
from qnlab.solver.qn_ntqn import qn_ntqn
from qnlab.util.method import Method


def trial(
    prob: RosenbrockProblem | PowellProblem | DixonPriceProblem | ZakharovProblem,
):
    """
    Compare qnlab wrapper with direct ntqn call for final objective value and solution.
    """
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    param = NtqnParameter(prob.n)
    info, fx, _x_opt = qn_ntqn(prob, param, method=Method("NTQN"))
    print(f"Info (qnlab): {info}")
    print(f"fx (qnlab): {fx}")


def test_rosenbrock():
    prob = RosenbrockProblem(n=5)
    trial(prob)


def test_powell():
    prob = PowellProblem()
    trial(prob)


def test_dixon_price():
    prob = DixonPriceProblem()
    trial(prob)


def test_zakharov():
    prob = ZakharovProblem()
    trial(prob)


if __name__ == "__main__":
    test_rosenbrock()
    test_powell()
    test_dixon_price()
    test_zakharov()
