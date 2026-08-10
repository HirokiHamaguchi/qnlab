from unittest.mock import Mock

import numpy as np

from qnlab.experiment import for_cutest_vis


def test_individual_plot_output_path_distinguishes_experiment_settings(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(for_cutest_vis, "INDIVIDUAL_PLOT_OUTPUT_DIR", tmp_path)

    assert (
        for_cutest_vis._individual_plot_output_path("ARWHEAD", 64, np.float64(0), False)
        == tmp_path / "unboxed" / "precision64" / "ARWHEAD"
    )
    assert (
        for_cutest_vis._individual_plot_output_path(
            "ARWHEAD", 64, np.float64(1e-3), True
        )
        == tmp_path / "boxed" / "noise0.001" / "ARWHEAD"
    )
    assert (
        for_cutest_vis._individual_plot_output_path(
            "ARWHEAD", 64, np.float64(0), False, metric="time"
        )
        == tmp_path / "unboxed" / "time" / "precision64" / "ARWHEAD"
    )


def test_individual_plot_saves_each_problem_without_changing_arguments(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(for_cutest_vis, "INDIVIDUAL_PLOT_OUTPUT_DIR", tmp_path)
    problem = Mock()
    monkeypatch.setattr(for_cutest_vis, "CUTEstQNProblem", Mock(return_value=problem))
    callback = Mock()
    monkeypatch.setattr(for_cutest_vis, "load_npz", Mock(return_value=callback))
    vis = Mock()
    monkeypatch.setattr(for_cutest_vis, "vis", vis)
    color_palette = {"method": "red"}
    line_styles = {"method": "s--"}
    monkeypatch.setattr(
        for_cutest_vis,
        "get_methods",
        Mock(return_value=([], color_palette, line_styles)),
    )
    method = Mock(label="method")

    for_cutest_vis.individual_plot(
        ["ARWHEAD", "BDQRTIC"], [(method, {"m": 5})], 64, np.float64(0)
    )

    assert vis.call_count == 2
    for prob_name, call in zip(["ARWHEAD", "BDQRTIC"], vis.call_args_list):
        assert call.args[:4] == (problem, [callback], ["method"], prob_name)
        assert call.kwargs == {
            "only_grad": False,
            "only_plot": True,
            "pdf_path": str(tmp_path / "unboxed" / "precision64" / prob_name),
            "x_axis": "calls",
            "color_palette": color_palette,
            "line_styles": line_styles,
        }
        assert (tmp_path / "unboxed" / "precision64").is_dir()


def test_individual_plot_can_load_time_results_and_plot_against_time(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(for_cutest_vis, "INDIVIDUAL_PLOT_OUTPUT_DIR", tmp_path)
    problem = Mock()
    monkeypatch.setattr(for_cutest_vis, "CUTEstQNProblem", Mock(return_value=problem))
    callback = Mock()
    load_npz = Mock(return_value=callback)
    monkeypatch.setattr(for_cutest_vis, "load_npz", load_npz)
    vis = Mock()
    monkeypatch.setattr(for_cutest_vis, "vis", vis)
    color_palette = {"method": "red"}
    line_styles = {"method": "s--"}
    monkeypatch.setattr(
        for_cutest_vis,
        "get_methods",
        Mock(return_value=([], color_palette, line_styles)),
    )
    method = Mock(label="method")

    for_cutest_vis.individual_plot(
        ["ARWHEAD"],
        [(method, {"m": 5})],
        64,
        np.float64(0),
        x_axis="time",
        result_subdir="time",
    )

    load_npz.assert_called_once()
    assert load_npz.call_args.kwargs == {"result_subdir": "time"}
    assert vis.call_args.kwargs["x_axis"] == "time"
    assert vis.call_args.kwargs["pdf_path"] == str(
        tmp_path / "unboxed" / "time" / "precision64" / "ARWHEAD"
    )
