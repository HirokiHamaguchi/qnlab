from qnlab.experiment.vis import vis
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback
from qnlab.util.doc_paths import doc_imgs_dir
from qnlab.util.method import Method

OUTPUT_DIR = doc_imgs_dir("quasi_newton")


def run_methods(prob: RosenbrockProblem):
    callback_gd = Callback(save_xs=True)
    callback_newton = Callback(save_xs=True)
    callback_lbfgsb = Callback(save_xs=True)

    methods = [
        (
            Method(base="GradientDescent", label="Gradient Descent"),
            callback_gd,
            {"max_iter": 1000, "tol": 1e-6},
        ),
        (
            Method(base="Newton", label="Newton Method"),
            callback_newton,
            {"max_iter": 1000, "tol": 1e-6},
        ),
        (
            Method(base="SciPy", scipy_method="L-BFGS-B", label="Quasi Newton"),
            callback_lbfgsb,
            {"maxiter": 1000, "gtol": 1e-6, "ftol": 0},
        ),
    ]

    for idx, (method, cb, options) in enumerate(methods, start=1):
        print(f"{idx}. {method.label}")
        ret, fx, _ = qn(prob, method, options, cb)
        print(f"   result: {ret}, f(x)={fx:.6f}")

    callbacks = [cb for _, cb, _ in methods]
    labels = [method.label for method, _, _ in methods]
    return callbacks, labels


def main():
    n = 2  # Use 2D for visualization
    prob = RosenbrockProblem(n)

    print("Solving Rosenbrock problem with three methods...")
    print(f"Initial point: {prob.x0}")
    print(f"Optimal point: {prob.x_opt}")
    print(f"Initial function value: {prob.f(prob.x0):.6f}")
    print()

    callbacks, labels = run_methods(prob)

    vis(
        prob,
        callbacks,
        labels,
        "Rosenbrock",
        pdf_path=str(OUTPUT_DIR / "newton_vs_qs_vs_gd"),
        max_length=100,
        use_tex=True,
        one_figure=True,
        x_axis="iterations",
    )


if __name__ == "__main__":
    main()
