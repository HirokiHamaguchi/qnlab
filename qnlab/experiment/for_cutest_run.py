import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback, CallbackTimeoutError
from qnlab.util.method import Method

RESULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "temp"
ZERO_NOISE = np.float64(0)
RESULT_PROTOCOL_VERSION = 2


def scaled_iteration_limit(
    reference_callback: Callback,
    gtol: np.float64,
    maximum: int = 15_000,
    multiplier: int = 15,
) -> int:
    """Scale an iteration limit from the first successful reference iterate."""
    if maximum <= 0 or multiplier <= 0:
        raise ValueError("maximum and multiplier must be positive.")
    reached = np.flatnonzero(np.asarray(reference_callback.gnorms) <= gtol)
    if reached.size == 0:
        return maximum
    reference_iterations = max(1, int(reached[0]))
    return min(maximum, multiplier * reference_iterations)


@dataclass
class CUTEstTask:
    problem_name: str
    method: Method
    options: dict
    precision: int
    function_noise: np.float64 = ZERO_NOISE
    gradient_noise: np.float64 = ZERO_NOISE
    assumed_function_error: np.float64 | None = None
    seed: int = 0
    boxed: bool = False
    scenario: str | None = None

    def metadata(self) -> dict:
        return {
            "result_protocol_version": RESULT_PROTOCOL_VERSION,
            "problem": self.problem_name,
            "method": self.method.label,
            "options": self.options,
            "precision": self.precision,
            "function_noise": self.function_noise,
            "gradient_noise": self.gradient_noise,
            "assumed_function_error": self.assumed_function_error,
            "seed": self.seed,
            "boxed": self.boxed,
            "scenario": self.scenario,
        }


def get_file_path(task: CUTEstTask, result_subdir: str | None = None) -> str:
    folder = RESULT_ROOT
    if result_subdir is not None:
        folder /= result_subdir
    if task.scenario is not None:
        folder /= task.scenario
        folder /= f"seed_{task.seed}"
    else:
        if task.boxed:
            folder /= "boxed"
        noise = max(task.function_noise, task.gradient_noise)
        folder /= "noisy" if noise > 0 else str(task.precision)
    return str(folder / task.problem_name / f"{task.method.label}.npz")


def load_npz(
    task: CUTEstTask, verbose: bool = True, result_subdir: str | None = None
) -> Callback:
    """Load stored callback data for a task."""
    callback, _ = load_npz_with_metadata(task, verbose, result_subdir)
    return callback


def load_npz_with_metadata(
    task: CUTEstTask, verbose: bool = True, result_subdir: str | None = None
) -> tuple[Callback, dict]:
    """Load stored callback data and metadata for a task."""
    file_path = get_file_path(task, result_subdir)
    if not Path(file_path).exists():
        if verbose:
            warnings.warn(f"File not found: {file_path}, returning empty Callback.")
        return Callback(), {}
    try:
        with np.load(file_path) as data:
            callback = Callback()
            callback.calls = data["calls"].astype(int)
            callback.fxs = data["fxs"]
            callback.gnorms = data["gnorms"]
            callback.times = data.get("times", [])
            metadata = json.loads(data["metadata"].item())
        if not metadata_matches_task(metadata, task):
            raise ValueError(
                f"Stored result metadata does not match the requested task: {file_path}"
            )
        assert len(callback.calls) == len(callback.fxs) == len(callback.gnorms)
        assert len(callback.times) in (0, len(callback.calls))
    except (EOFError, BadZipFile):
        print(file_path)
        raise
    # Iterates are intentionally not stored in this compact result format.
    callback.xs = []
    callback.others.update(metadata.get("diagnostics", {}))
    return callback, metadata


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metadata_matches_task(metadata: dict, task: CUTEstTask) -> bool:
    """Return whether stored task-defining metadata matches ``task`` exactly."""
    expected = json.loads(json.dumps(task.metadata(), default=_json_default))
    return all(metadata.get(key) == value for key, value in expected.items())


def result_matches_task(
    task: CUTEstTask, result_subdir: str | None = None
) -> bool:
    """Return whether a readable stored result belongs to the requested task."""
    file_path = Path(get_file_path(task, result_subdir))
    if not file_path.exists():
        return False
    try:
        with np.load(file_path) as data:
            metadata = json.loads(data["metadata"].item())
    except (EOFError, BadZipFile, KeyError, ValueError, json.JSONDecodeError):
        return False
    return metadata_matches_task(metadata, task)


def save_npz(
    task: CUTEstTask,
    callback: Callback,
    result_subdir: str | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """Persist callback data and task metadata."""
    file_path = Path(get_file_path(task, result_subdir))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = task.metadata() | {"diagnostics": dict(callback.others)}
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    np.savez_compressed(
        file_path,
        calls=callback.calls,
        fxs=callback.fxs,
        gnorms=callback.gnorms,
        times=callback.times,
        metadata=json.dumps(metadata, default=_json_default),
    )


def save_hard_timeout_result(
    task: CUTEstTask,
    time_limit: int,
    elapsed: float,
    result_subdir: str | None = None,
) -> None:
    """Persist an empty result after the parent hard-kills a task process."""
    save_npz(
        task,
        Callback(),
        result_subdir,
        {
            "status": "timeout",
            "error": (
                f"Child process exceeded hard timeout after {elapsed:.2f}s "
                "and was terminated."
            ),
            "timeout_kind": "hard",
            "time_limit": time_limit,
            "elapsed": elapsed,
        },
    )


def _create_problem(task: CUTEstTask) -> CUTEstQNProblem:
    has_noise_model = (
        task.function_noise > 0
        or task.gradient_noise > 0
        or task.assumed_function_error is not None
    )
    if has_noise_model:
        return CUTEstNoisedProblem(
            task.problem_name,
            precision=task.precision,
            function_noise=task.function_noise,
            gradient_noise=task.gradient_noise,
            assumed_function_error=task.assumed_function_error,
            seed=task.seed,
        )
    return CUTEstQNProblem(task.problem_name, precision=task.precision)


def solve_problem_with_timeout(
    task: CUTEstTask,
    time_limit: int,
    allow_save: bool,
    result_subdir: str | None = None,
) -> None:
    file_path = get_file_path(task, result_subdir)
    problem = _create_problem(task)
    callback = Callback(time_limit=time_limit)
    try:
        print(f"▶ Running: {file_path}")
        return_code, _, _ = qn(
            problem,
            task.method,
            task.options,
            callback,
            bounds=problem.bounds if task.boxed else None,
        )
        print(f"✓ {task.problem_name} with {task.method.label}")
        save_npz(
            task,
            callback,
            result_subdir,
            {
                "dimension": int(problem.n),
                "status": "completed",
                "error": "",
                "return_code": return_code.name,
                "return_code_value": int(return_code),
            },
        )
    except CallbackTimeoutError as error:
        print(f"⏱ {task.problem_name} with {task.method.label}: {error}")
        if allow_save:
            save_npz(
                task,
                callback,
                result_subdir,
                {
                    "dimension": int(problem.n),
                    "status": "timeout",
                    "error": str(error),
                },
            )
        else:
            print("⚠ Not saving results for time limit less than 600 seconds.")
    except Exception as error:  # noqa: BLE001 - persist diagnostics for failed tasks
        print(f"✗ {task.problem_name} with {task.method.label}: {error}")
        save_npz(
            task,
            callback,
            result_subdir,
            {
                "dimension": int(problem.n),
                "status": "error",
                "error": repr(error),
            },
        )


def run_tasks(
    tasks: list[CUTEstTask],
    error_causing_tasks: list[tuple[int, str, str]],
    time_limit: int,
    result_subdir: str | None = None,
    overwrite: bool = False,
) -> None:
    pending = [
        task
        for task in tasks
        if overwrite or not result_matches_task(task, result_subdir)
    ]
    print(f"Total tasks to run: {len(pending)}")

    errors = []
    for index, task in enumerate(pending):
        file_path = get_file_path(task, result_subdir)
        try:
            known_error = (
                task.precision,
                task.problem_name,
                task.method.label,
            ) in error_causing_tasks
            if known_error:
                print("⚠ Reducing time limit for known error-causing task.")
                solve_problem_with_timeout(
                    task, 60, allow_save=True, result_subdir=result_subdir
                )
            else:
                solve_problem_with_timeout(
                    task,
                    time_limit,
                    allow_save=time_limit >= 600,
                    result_subdir=result_subdir,
                )
            print(f"[{index + 1}/{len(pending)}] ✓ Success")
        except Exception as error:  # noqa: BLE001 - continue the remaining task batch
            print(f"[{index + 1}/{len(pending)}] ⚠ Error: {error}")
            errors.append((file_path, f"Error: {error}"))


def run(
    problems: list[str],
    methods: list[tuple[Method, dict]],
    precision: int,
    noise: np.float64,
    ERROR_CAUSING_TASKS: list[tuple[int, str, str]],
    TL: int,
    boxed: bool = False,
    result_subdir: str | None = None,
) -> None:
    """Run the original equal-noise workflow used by auxiliary notebooks."""
    tasks = [
        CUTEstTask(
            problem,
            method,
            options,
            precision,
            function_noise=noise,
            gradient_noise=noise,
            boxed=boxed,
        )
        for problem in problems
        for method, options in methods
    ]
    run_tasks(tasks, ERROR_CAUSING_TASKS, TL, result_subdir=result_subdir)


def load_results(
    methods: list[tuple[Method, dict]],
    problems: list[str],
    precision: int,
    noise: np.float64,
    gtol: np.float64,
    boxed: bool = False,
    metric: str = "calls",
    result_subdir: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and aggregate stored results."""
    assert metric in ["calls", "time"]
    alg_names = [method.label for method, _ in methods]
    callsM = np.zeros((len(alg_names), len(problems)), dtype=float)
    fxsM = np.zeros((len(alg_names), len(problems)), dtype=float)
    gnormsM = np.zeros((len(alg_names), len(problems)), dtype=float)

    for j, problem_name in enumerate(problems):
        for i, (method, options) in enumerate(methods):
            task = CUTEstTask(
                problem_name,
                method,
                options,
                precision,
                function_noise=noise,
                gradient_noise=noise,
                boxed=boxed,
            )
            callback = load_npz(task, False, result_subdir)
            if len(callback.calls) == 0:
                result = (np.inf, np.inf, np.inf)
            else:
                converged = callback.gnorms <= gtol
                if not np.any(converged):
                    result = (np.inf, np.inf, np.inf)
                else:
                    index = np.where(converged)[0][0]
                    assert callback.calls[index] >= 0
                    if metric == "time":
                        work = (
                            np.inf
                            if len(callback.times) == 0
                            else max(np.finfo(float).tiny, callback.times[index])
                        )
                    else:
                        work = max(1, callback.calls[index])
                    result = (
                        work,
                        callback.fxs[index],
                        callback.gnorms[index],
                    )

            callsM[i, j], fxsM[i, j], gnormsM[i, j] = result

    return alg_names, callsM, fxsM, gnormsM, problems
