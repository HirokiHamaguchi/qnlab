import time
from typing import List, Tuple

from qnlab.experiment.vis import vis
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def trial(
    prob: BaseProblem,
    name: str,
    configs: List[Tuple[Method, dict]],
    max_length: int = int(1e9),
    do_vis: bool = True,
    pdf_path: str = "",
    use_tex: bool = False,
    only_plot: bool = False,
    only_grad: bool = False,
    verbose: bool = False,
) -> List[Callback]:
    print(f"Trial on problem: {prob.name} (n={prob.n})")

    callbacks = []
    labels = []
    save_xs = True

    for method, _options in configs:
        assert isinstance(method, Method)
        options = _options.copy()
        assert isinstance(options, dict)

        m = 10  # Default value for m
        if method.base == "SciPy":
            if method.scipy_method == "L-BFGS-B":
                options.setdefault("maxcor", m)
        else:
            options.setdefault("m", m)

        T0 = time.perf_counter()
        callback = Callback(save_xs=save_xs)

        info, fx, x_opt = qn(prob, method, options, callback, verbose)

        print(f"{str(method)} info:{info} time:{time.perf_counter() - T0:.2f}sec")
        print(f"Final f:{callback.fxs[-1]:.2e}, ||g||:{callback.gnorms[-1]:.2e}")
        if callback.others:
            print("  others:", callback.others)
        callbacks.append(callback)

        labelName = options.get("label_name", method.label)
        labels.append(labelName)

    if do_vis:
        vis(
            prob,
            callbacks,
            labels,
            name,
            use_tex=use_tex,
            only_plot=only_plot,
            only_grad=only_grad,
            pdf_path=pdf_path,
            max_length=max_length,
        )

    return callbacks
