from unittest.mock import Mock

import numpy as np

from qnlab.experiment import for_cutest_run
from qnlab.util.callback import Callback


def _task(method: Mock) -> for_cutest_run.CUTEstTask:
    return for_cutest_run.CUTEstTask("ARWHEAD", method, {"m": 5}, 64)


def test_save_and_load_npz_preserves_times(monkeypatch, tmp_path) -> None:
    result_path = tmp_path / "result.npz"
    get_file_path = Mock(return_value=str(result_path))
    monkeypatch.setattr(
        for_cutest_run,
        "get_file_path",
        get_file_path,
    )
    method = Mock(label="method")
    callback = Callback()
    callback.calls = [1, 4]
    callback.fxs = [np.float64(2.0), np.float64(1.0)]
    callback.gnorms = [np.float64(1.0), np.float64(0.01)]
    callback.times = [0.1, 0.4]

    for_cutest_run.save_npz(_task(method), callback, result_subdir="time")
    loaded = for_cutest_run.load_npz(_task(method), result_subdir="time")

    assert list(loaded.calls) == [1, 4]
    assert list(loaded.fxs) == [2.0, 1.0]
    assert list(loaded.gnorms) == [1.0, 0.01]
    assert list(loaded.times) == [0.1, 0.4]
    get_file_path.assert_called_with(_task(method), "time")


def test_load_results_can_use_time_to_gradient_tolerance(monkeypatch) -> None:
    method = Mock(label="method")
    callback = Callback()
    callback.calls = [1, 4, 8]
    callback.fxs = np.array([3.0, 2.0, 1.0], dtype=np.float64).tolist()
    callback.gnorms = np.array([1.0, 1e-4, 1e-6], dtype=np.float64).tolist()
    callback.times = [0.2, 0.7, 1.4]
    load_npz = Mock(return_value=callback)
    monkeypatch.setattr(for_cutest_run, "load_npz", load_npz)

    alg_names, workM, fxsM, gnormsM, problems = for_cutest_run.load_results(
        [(method, {"m": 5})],
        ["ARWHEAD"],
        64,
        np.float64(0),
        np.float64(1e-3),
        metric="time",
        result_subdir="time",
    )

    assert alg_names == ["method"]
    assert problems == ["ARWHEAD"]
    assert workM.tolist() == [[0.7]]
    assert fxsM.tolist() == [[2.0]]
    assert gnormsM.tolist() == [[1e-4]]
    load_npz.assert_called_once()
    assert load_npz.call_args.args[1] is False
    assert load_npz.call_args.args[2] == "time"
