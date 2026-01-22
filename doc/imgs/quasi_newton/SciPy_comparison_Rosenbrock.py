import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from qnlab.experiment.trial import trial
from qnlab.problem import RosenbrockProblem
from qnlab.util.method import Method

methods: List[Tuple[Method, dict]] = [
    (Method(base="SciPy", scipy_method="L-BFGS-B"), {}),
    (Method(base="SciPy", scipy_method="CG"), {}),
    (Method(base="SciPy", scipy_method="BFGS"), {}),
    (Method(base="SciPy", scipy_method="Newton-CG"), {}),
    (Method(base="SciPy", scipy_method="Powell"), {}),
    (Method(base="SciPy", scipy_method="Nelder-Mead"), {}),
    (Method(base="SciPy", scipy_method="COBYLA"), {}),
    (Method(base="SciPy", scipy_method="SLSQP"), {}),
    (Method(base="SciPy", scipy_method="trust-constr"), {}),
    (Method(base="SciPy", scipy_method="trust-ncg"), {}),
    (Method(base="SciPy", scipy_method="trust-krylov"), {}),
    (Method(base="SciPy", scipy_method="TNC"), {}),
]

prob = RosenbrockProblem(n=100)
prob.x0 = np.zeros(prob.n)
trial(
    prob,
    "Rosenbrock",
    methods,
    pdf_path=str(Path(os.path.dirname(__file__)) / "SciPy_comparison_Rosenbrock"),
    only_plot=True,
)
