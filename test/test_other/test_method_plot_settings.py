import numpy as np

from qnlab.util.method import (
    COLORS,
    LINE_STYLES,
    get_box_methods,
    get_methods,
)


def test_boxed_and_unboxed_methods_share_plot_settings() -> None:
    _, colors, line_styles = get_methods()
    _, box_colors, box_line_styles = get_box_methods()

    np.testing.assert_array_equal(box_colors["NTRQNB"], colors["NTRQN"])
    np.testing.assert_array_equal(box_colors["NTRQNB-MS"], colors["NTRQN-MS"])
    np.testing.assert_array_equal(box_colors["SciPy"], colors["SciPy"])
    assert box_line_styles["NTRQNB"] == line_styles["NTRQN"]
    assert box_line_styles["NTRQNB-MS"] == line_styles["NTRQN-MS"]
    assert box_line_styles["SciPy"] == line_styles["SciPy"]


def test_getters_return_subsets_of_global_plot_settings() -> None:
    methods, colors, line_styles = get_methods()
    box_methods, box_colors, box_line_styles = get_box_methods()

    assert colors == {method.label: COLORS[method.label] for method, _ in methods}
    assert line_styles == {
        method.label: LINE_STYLES[method.label] for method, _ in methods
    }
    assert box_colors == {
        method.label: COLORS[method.label] for method, _ in box_methods
    }
    assert box_line_styles == {
        method.label: LINE_STYLES[method.label] for method, _ in box_methods
    }
