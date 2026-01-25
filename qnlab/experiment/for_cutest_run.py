import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple, TypeAlias
from zipfile import BadZipFile

import numpy as np

from qnlab.experiment.vis import vis
from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback
from qnlab.util.method import Method

task_type: TypeAlias = Tuple[str, Method, dict, int, np.float64]


def get_file_path(task: task_type) -> str:
    prob_name, method, _option, precision, noise = task
    prob_type = "noisy" if noise > 0 else str(precision)

    return str(
        Path(os.path.dirname(__file__)).parent.parent
        / "data"
        / "temp"
        / prob_type
        / prob_name
        / f"{method.label}.npz"
    )


def solveProblemWithTimeout(task: task_type):
    prob_name, method, option, precision, noise = task
    file_path = get_file_path(task)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    prob = CUTEstQNProblem(prob_name, precision=precision, noise=noise)
    callback = Callback()
    try:
        qn(prob, method, option, callback)
        print(f"✓ {prob_name} with {method.label}")
    except Exception as e:
        print(f"✗ {prob_name} with {method.label}: {str(e)}")
    np.savez_compressed(
        file_path,
        calls=callback.calls,
        fxs=callback.fxs,
        gnorms=callback.gnorms,
    )


def run(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
    TL: int,
):
    # Prepare all tasks
    tasks: List[task_type] = []
    for problem in problems:
        for method, option in methods:
            task = (problem, method, option, precision, noise)
            npz_path = get_file_path(task)
            if os.path.exists(npz_path):
                continue
            tasks.append(task)
    print(f"Total tasks to run: {len(tasks)}")

    errors = []  # collect errors for reporting after all tasks
    with multiprocessing.Pool(processes=1) as pool:
        for i, task in enumerate(tasks):
            file_path = get_file_path(task)
            print(f"[{i + 1}/{len(tasks)}] ▶ Running: {file_path}")
            try:
                result = pool.apply_async(solveProblemWithTimeout, (task,))
                result.get(timeout=TL)
                print(f"[{i + 1}/{len(tasks)}] ✓ Success")
            except multiprocessing.TimeoutError:
                print(f"[{i + 1}/{len(tasks)}] ⏱ Timeout")
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(processes=1)  # restart the pool
                errors.append((file_path, "Timeout"))
            except Exception as e:
                print(f"[{i + 1}/{len(tasks)}] ⚠ Error: {e}")
                errors.append((file_path, f"Error: {e}"))

    if errors:
        msg = "\n".join([f" - {name}: {err}" for name, err in errors])
        print("=" * 20)

        print(f"The following tasks failed:\n{msg}")
        print("=" * 20)


def individual_plot(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
):
    # Perform visualization using loaded data
    for prob_name in problems:
        prob = CUTEstQNProblem(prob_name, precision=precision)

        callbacks = []
        for method, option in methods:
            task: task_type = (prob_name, method, option, precision, noise)
            file_path = get_file_path(task)

            assert os.path.exists(file_path), file_path

            data = np.load(file_path)
            calls = data["calls"]
            fxs = data["fxs"]
            gnorms = data["gnorms"]
            callback = Callback()
            callback.calls = calls.astype(int)
            callback.fxs = fxs
            callback.gnorms = gnorms
            callback.xs = [np.zeros(0) for _ in range(len(calls))]
            callbacks.append(callback)

        labels = [method.label for method, _ in methods]
        vis(prob, callbacks, labels, prob_name, only_grad=True, only_plot=True)


def load_results(
    methods: list[Tuple[Method, dict]],
    problems: list[str],
    precision: int,
    noise: np.float64,
    gtol: float,
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
            file_path = get_file_path(task)

            if not os.path.exists(file_path):
                print(f"Warning: Missing file {file_path}")
                res = (np.inf, np.inf, np.inf)
            else:
                try:
                    data = np.load(file_path)
                except EOFError:
                    print(file_path)
                    raise EOFError
                except BadZipFile:
                    print(file_path)
                    raise
                calls = data["calls"]
                fxs = data["fxs"]
                gnorms = data["gnorms"]
                if len(calls) == 0:
                    res = (np.inf, np.inf, np.inf)
                else:
                    isOk = gnorms <= gtol
                    if not np.any(isOk):
                        res = (np.inf, np.inf, np.inf)
                    else:
                        idx = np.where(isOk)[0][0]
                        assert calls[idx] >= 0
                        # We take max with 1
                        # Otherwise, performance profile will cause error
                        call_max_1 = max(1, calls[idx])
                        res = (call_max_1, fxs[idx], gnorms[idx])

            callsM[i, j], fxsM[i, j], gnormsM[i, j] = res

    # Remove problems with all zero calls
    zero_rows = np.where(np.all(callsM == 0, axis=0))[0]
    callsM = np.delete(callsM, zero_rows, axis=1)
    fxsM = np.delete(fxsM, zero_rows, axis=1)
    gnormsM = np.delete(gnormsM, zero_rows, axis=1)
    problems = [problems[i] for i in range(len(problems)) if i not in zero_rows]

    return alg_names, callsM, fxsM, gnormsM, problems
