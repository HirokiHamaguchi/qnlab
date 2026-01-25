from typing import List, Literal, Tuple, get_args

import matplotlib.pyplot as plt

BaseType = Literal[
    "Line",
    "Kanzow",
    "Hamaguchi",
    "SciPy",
    "NTQN",
    "GradientDescent",
    "Newton",
]
ScipyMethodType = Literal[
    "BFGS",
    "CG",
    "COBYLA",
    "L-BFGS-B",
    "Nelder-Mead",
    "Newton-CG",
    "None",
    "Powell",
    "SLSQP",
    "TNC",
    "trust-constr",
    "trust-krylov",
    "trust-ncg",
]
StoreType = Literal["raw", "cautious"]
SecantType = Literal["raw", "modified", "damped", "damped_modified"]
UpdateType = Literal["bfgs", "dfp", "sr1", "psb"]


class Method:
    def __init__(
        self,
        base: BaseType = "Line",
        store: StoreType = "raw",
        secant: SecantType = "raw",
        update: UpdateType = "bfgs",
        scipy_method: ScipyMethodType = "None",
        label: str = "",
    ):
        self.base = base
        self.store = store
        self.secant = secant
        self.update = update
        self.scipy_method = scipy_method
        self.label = label if label else self._to_label()

        if self.base in ["SciPy", "Kanzow", "NTQN"]:
            if self.base == "SciPy":
                assert scipy_method in get_args(ScipyMethodType), scipy_method
            assert store == secant == "raw"
            assert update == "bfgs"
        elif self.base in ["GradientDescent", "Newton"]:
            assert store == secant == "raw"
            assert update == "bfgs"
        else:
            assert base in get_args(BaseType), base
            assert store in get_args(StoreType), store
            assert secant in get_args(SecantType), secant
            assert update in get_args(UpdateType), update

    def __repr__(self) -> str:
        if "Hamaguchi" in self.base:
            return f"Method(base={self.base}, store={self.store}, secant={self.secant}, update={self.update})"
        else:
            scipy_str = (
                f", scipy={self.scipy_method}" if self.scipy_method != "None" else ""
            )
            return f"Method(base={self.base}" + scipy_str + ")"

    def _to_label(self) -> str:
        if self.base == "SciPy":
            return f"S_{self.scipy_method}"
        elif self.base == "NTQN":
            return "NTQN"
        elif self.base == "Kanzow":
            return "Kanzow"
        elif self.base == "GradientDescent":
            return "GD"
        elif self.base == "Newton":
            return "Newton"
        else:
            label = "Hamaguchi"

            if self.store == "cautious":
                label += "_StoreC"

            if self.secant == "modified":
                label += "_SecantM"
            elif self.secant == "damped":
                label += "_SecantD"
            elif self.secant == "damped_modified":
                label += "_SecantMD"

            label += f"_Update{self.update[0].upper()}"

            return label


def get_methods(
    m: int = 10, MI: int = 15000
) -> Tuple[List[Tuple[Method, dict]], dict, dict]:
    """Get standard set of methods for benchmarking.

    Args:
        m: Memory size for L-BFGS methods
        MI: Maximum iterations

    Returns:
        List of tuples containing (Method, options_dict)
    """

    methods = [
        (
            Method("Hamaguchi", "cautious", "damped", "bfgs", label="Hamaguchi"),
            {"m": m, "max_iterations": MI},
        ),
        (
            Method(
                "Hamaguchi", "cautious", "damped_modified", "bfgs", label="Hamaguchi-MS"
            ),
            {"m": m, "max_iterations": MI},
        ),
        (
            Method("Line", "raw", "raw", "bfgs", label="Line"),
            {"m": m, "max_iterations": MI},
        ),
        (
            Method("Line", "raw", "modified", "bfgs", label="Line-MS"),
            {"m": m, "max_iterations": MI},
        ),
        (
            Method(base="SciPy", scipy_method="L-BFGS-B", label="SciPy"),
            {"maxcor": m, "maxiter": MI, "ftol": 0},
        ),
        (
            Method("Kanzow", "raw", "raw", "bfgs", label="Reg"),
            {"m": m, "max_iterations": MI},
        ),
        (
            Method("NTQN", "raw", "raw", "bfgs", label="NTQN"),
            {"m": m, "max_iterations": MI},
        ),
    ]

    TAB20 = plt.colormaps.get_cmap("tab20")

    COLORS = {
        "Hamaguchi": TAB20(0),
        "Hamaguchi-MS": TAB20(1),
        "Line": TAB20(2),
        "Line-MS": TAB20(3),
        "SciPy": TAB20(6),
        "Reg": TAB20(4),
        "NTQN": TAB20(8),
    }

    LINE_STYLES = {
        "Hamaguchi": "o-",
        "Hamaguchi-MS": "o--",
        "Line": "^--",
        "Line-MS": "^-.",
        "SciPy": "v:",
        "Reg": "D--",
        "NTQN": "s-.",
    }

    assert set(COLORS.keys()) == set(method.label for method, _ in methods)
    assert set(LINE_STYLES.keys()) == set(method.label for method, _ in methods)

    return methods, COLORS, LINE_STYLES
