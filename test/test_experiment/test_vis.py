from unittest.mock import Mock

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from qnlab.experiment import vis as vis_module
from qnlab.util.callback import Callback


def _callback(xs: list[np.ndarray]) -> Callback:
    callback = Callback()
    callback.xs = xs
    callback.fxs = np.array([2.0, 1.0], dtype=np.float64).tolist()
    callback.gnorms = np.array([1.0, 0.1], dtype=np.float64).tolist()
    callback.calls = [1, 2]
    callback.times = [0.1, 0.2]
    return callback


def test_has_trajectory_data_rejects_missing_iterates() -> None:
    prob = Mock(n=2)

    assert not vis_module._has_trajectory_data(prob, [_callback([])])
    assert not vis_module._has_trajectory_data(
        prob, [_callback([np.zeros(0), np.zeros(0)])]
    )
    assert vis_module._has_trajectory_data(prob, [_callback([np.zeros(2), np.ones(2)])])


def test_get_plot_properties_uses_performance_profile_style() -> None:
    props = vis_module._get_plot_properties(
        "method", 0, color_palette={"method": "red"}, line_styles={"method": "s--"}
    )

    assert props == {
        "linewidth": 2.2,
        "alpha": 1.0,
        "color": "red",
        "fmt": "s--",
        "markersize": 6,
        "zorder": 5.0,
    }


def test_get_plot_properties_uses_default_color_cycle_without_palette() -> None:
    props = vis_module._get_plot_properties("method", 0)

    assert props["color"] is None


def test_plot_gradient_norms_uses_distinct_default_colors() -> None:
    _, ax = vis_module.plt.subplots()

    vis_module._plot_gradient_norms(
        Mock(n=1),
        [_callback([]), _callback([])],
        ["first", "second"],
        "problem",
        "calls",
        ax,
    )

    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    vis_module.plt.close(ax.figure)


def test_plot_1d_function_evaluates_vector_points() -> None:
    prob = Mock(n=1, x0=np.array([0.0]))
    prob._f.side_effect = lambda x: x[0] ** 2
    callback = _callback([np.array([1.0]), np.array([2.0])])
    _, ax = vis_module.plt.subplots()

    vis_module._plot_1d_function(prob, [callback], ["method"], ax)

    assert all(call.args[0].shape == (1,) for call in prob._f.call_args_list)
    vis_module.plt.close(ax.figure)


def test_get_marker_indices_uses_all_points_for_short_series() -> None:
    assert vis_module._get_marker_indices([0, 10, 20], target_count=4) == [0, 1, 2]


def test_get_marker_indices_spreads_markers_over_x_range() -> None:
    assert vis_module._get_marker_indices([0, 1, 2, 90, 100], target_count=3) == [
        0,
        3,
        4,
    ]


def test_get_marker_indices_handles_constant_x_values() -> None:
    indices = vis_module._get_marker_indices(np.zeros(20), target_count=5)

    assert indices == [0, 4, 9, 14, 19]


def test_vis_skips_contour_when_iterates_are_unavailable(monkeypatch) -> None:
    prob = Mock(n=2)
    callback = _callback([])
    contour_plot = Mock()
    monkeypatch.setattr(vis_module, "_create_contour_plot", contour_plot)
    monkeypatch.setattr(vis_module, "_save_or_show_figure", Mock())

    with pytest.warns(RuntimeWarning, match="Skipping the contour plot"):
        vis_module.vis(prob, [callback], ["method"], "problem")

    contour_plot.assert_not_called()


def test_vis_accepts_performance_profile_styles(monkeypatch) -> None:
    prob = Mock(n=2)
    callback = _callback([])
    monkeypatch.setattr(vis_module, "_save_or_show_figure", Mock())

    vis_module.vis(
        prob,
        [callback],
        ["method"],
        "problem",
        only_plot=True,
        color_palette={"method": "red"},
        line_styles={"method": "s--"},
    )


def test_vis_accepts_time_axis(monkeypatch) -> None:
    prob = Mock(n=2)
    callback = _callback([])
    monkeypatch.setattr(vis_module, "_save_or_show_figure", Mock())

    vis_module.vis(
        prob, [callback], ["method"], "problem", only_plot=True, x_axis="time"
    )
