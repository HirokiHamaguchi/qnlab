from unittest.mock import Mock

import numpy as np

from qnlab.experiment import for_cutest_run
from qnlab.util.callback import Callback
from qnlab.util.ret_values import RetCode


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
    callback.others["OFFO accumulator restart"] = 2

    for_cutest_run.save_npz(
        _task(method),
        callback,
        result_subdir="time",
        extra_metadata={"status": "completed"},
    )
    loaded, metadata = for_cutest_run.load_npz_with_metadata(
        _task(method), result_subdir="time"
    )

    assert list(loaded.calls) == [1, 4]
    assert list(loaded.fxs) == [2.0, 1.0]
    assert list(loaded.gnorms) == [1.0, 0.01]
    assert list(loaded.times) == [0.1, 0.4]
    assert loaded.others["OFFO accumulator restart"] == 2
    assert metadata["status"] == "completed"
    get_file_path.assert_called_with(_task(method), "time")


def test_solve_problem_saves_return_code(monkeypatch) -> None:
    method = Mock(label="method")
    task = _task(method)
    problem = Mock(n=3, bounds=None)
    monkeypatch.setattr(for_cutest_run, "_create_problem", Mock(return_value=problem))
    monkeypatch.setattr(
        for_cutest_run, "get_file_path", Mock(return_value="result.npz")
    )
    monkeypatch.setattr(
        for_cutest_run,
        "qn",
        Mock(return_value=(RetCode.ERR_MAXIMUMITERATION, np.float64(1.0), np.zeros(3))),
    )
    save_npz = Mock()
    monkeypatch.setattr(for_cutest_run, "save_npz", save_npz)

    for_cutest_run.solve_problem_with_timeout(task, 600, allow_save=True)

    metadata = save_npz.call_args.args[3]
    assert metadata["status"] == "completed"
    assert metadata["return_code"] == "ERR_MAXIMUMITERATION"
    assert metadata["return_code_value"] == int(RetCode.ERR_MAXIMUMITERATION)


def test_empty_error_result_can_be_loaded_and_is_not_rerun(
    monkeypatch, tmp_path
) -> None:
    result_path = tmp_path / "error.npz"
    monkeypatch.setattr(
        for_cutest_run, "get_file_path", Mock(return_value=str(result_path))
    )
    method = Mock(label="method")
    task = _task(method)
    callback = Callback()
    for_cutest_run.save_npz(
        task,
        callback,
        extra_metadata={"status": "error", "error": "failed before first callback"},
    )

    loaded, metadata = for_cutest_run.load_npz_with_metadata(task)
    assert len(loaded.calls) == 0
    assert metadata["status"] == "error"

    solve = Mock()
    monkeypatch.setattr(for_cutest_run, "solve_problem_with_timeout", solve)
    for_cutest_run.run_tasks([task], [], 600)
    solve.assert_not_called()


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
