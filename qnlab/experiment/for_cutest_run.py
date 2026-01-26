import os
import warnings
from pathlib import Path
from typing import List, Tuple, TypeAlias
from zipfile import BadZipFile

import numpy as np

from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback, CallbackTimeoutError
from qnlab.util.method import Method

task_type: TypeAlias = Tuple[str, Method, dict, int, np.float64]


def get_file_path(task: task_type) -> str:
    prob_name, method, _option, precision, noise = task
    prob_type = "noisy" if noise > 0 else str(precision)
    folder = Path(os.path.dirname(__file__)).parent.parent / "data" / "temp"
    return str(folder / prob_type / prob_name / f"{method.label}.npz")


def load_npz(task: task_type, verbose: bool = True) -> Callback:
    """Load stored callback data for a task."""
    file_path = get_file_path(task)
    if not os.path.exists(file_path):
        if verbose:
            warnings.warn(f"File not found: {file_path}, returning empty Callback.")
        callback = Callback()
        return callback
    try:
        with np.load(file_path) as data:
            callback = Callback()
            callback.calls = data["calls"].astype(int)
            callback.fxs = data["fxs"]
            callback.gnorms = data["gnorms"]
        assert len(callback.calls) == len(callback.fxs) == len(callback.gnorms)
        assert len(callback.calls) > 0
    except (EOFError, BadZipFile) as e:
        print(file_path)
        raise e
    callback.xs = [np.zeros(0) for _ in range(len(callback.calls))]
    return callback


def save_npz(task: task_type, callback: Callback) -> None:
    """Persist callback data for a task."""
    file_path = get_file_path(task)
    np.savez_compressed(
        file_path,
        calls=callback.calls,
        fxs=callback.fxs,
        gnorms=callback.gnorms,
    )


def solveProblemWithTimeout(task: task_type, TL: int, allow_save: bool):
    prob_name, method, option, precision, noise = task
    file_path = get_file_path(task)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if noise > 0.0:
        prob: CUTEstQNProblem = CUTEstNoisedProblem(
            prob_name, precision=precision, noise=noise
        )
    else:
        prob = CUTEstQNProblem(prob_name, precision=precision)
    callback = Callback(time_limit=TL)
    try:
        print(f"▶ Running: {file_path}")
        qn(prob, method, option, callback)
        print(f"✓ {prob_name} with {method.label}")
        save_npz(task, callback)
    except CallbackTimeoutError as e:
        print(f"⏱ {prob_name} with {method.label}: {str(e)}")
        if allow_save:
            save_npz(task, callback)
        else:
            print("⚠ Not saving results for time limit less than 600 seconds.")
    except Exception as e:
        print(f"✗ {prob_name} with {method.label}: {str(e)}")
        save_npz(task, callback)


def run(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
    ERROR_CAUSING_TASKS: list[Tuple[int, str, str]],
    TL: int,
):
    # Prepare all tasks
    tasks: List[task_type] = []
    for problem in problems:
        for method, option in methods:
            task = (problem, method, option, precision, noise)
            if len(load_npz(task, False).calls) > 0:
                continue
            tasks.append(task)
    print(f"Total tasks to run: {len(tasks)}")

    errors = []  # collect errors for reporting after all tasks
    for i, task in enumerate(tasks):
        file_path = get_file_path(task)
        try:
            if (precision, task[0], task[1].label) in ERROR_CAUSING_TASKS:
                print("⚠ Reducing time limit for known error-causing task.")
                solveProblemWithTimeout(task, 60, allow_save=True)
            else:
                solveProblemWithTimeout(task, TL, allow_save=TL >= 600)
            print(f"[{i + 1}/{len(tasks)}] ✓ Success")
        except Exception as e:
            print(f"[{i + 1}/{len(tasks)}] ⚠ Error: {e}")
            errors.append((file_path, f"Error: {e}"))


def load_results(
    methods: list[Tuple[Method, dict]],
    problems: list[str],
    precision: int,
    noise: np.float64,
    gtol: np.float64,
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and aggregate results from saved files.

    Returns:
        alg_names: List of algorithm names
        callsM: (n_algorithms, n_problems) array of function calls
        fxsM: (n_algorithms, n_problems) array of function values
        gnormsM: (n_algorithms, n_problems) array of gradient norms
        problems: Filtered list of problem names (after removing zero rows)
    """
    alg_names = list(method.label for method, _ in methods)
    nAlgorithms = len(alg_names)
    nProbs = len(problems)
    callsM = np.zeros((nAlgorithms, nProbs), dtype=float)
    fxsM = np.zeros((nAlgorithms, nProbs), dtype=float)
    gnormsM = np.zeros((nAlgorithms, nProbs), dtype=float)

    for j, prob_name in enumerate(problems):
        for i, (method, option) in enumerate(methods):
            task: task_type = (prob_name, method, option, precision, noise)
            callback = load_npz(task, False)
            if len(callback.calls) == 0:
                res = (np.inf, np.inf, np.inf)
            else:
                isOk = callback.gnorms <= gtol
                if not np.any(isOk):
                    res = (np.inf, np.inf, np.inf)
                else:
                    idx = np.where(isOk)[0][0]
                    assert callback.calls[idx] >= 0
                    # We take max with 1
                    # Otherwise, performance profile will cause error
                    call_max_1 = max(1, callback.calls[idx])
                    res = (call_max_1, callback.fxs[idx], callback.gnorms[idx])

            callsM[i, j], fxsM[i, j], gnormsM[i, j] = res

    return alg_names, callsM, fxsM, gnormsM, problems
