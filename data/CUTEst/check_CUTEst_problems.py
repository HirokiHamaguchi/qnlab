import json
import multiprocessing
import os
from typing import Literal

import numpy as np
import pycutest  # type: ignore[import-untyped]  # pycutest does not provide stubs
from qnlab.problem.cutest import CUTEstQNProblem

ConstraintType = Literal["unconstrained", "bound"]


def _json_path(constraints: ConstraintType, stem: str) -> str:
    suffix = "" if constraints == "unconstrained" else "_boxed"
    return os.path.join(os.path.dirname(__file__), f"{stem}{suffix}.json")


def problemsToRun(
    precision: int | None, constraints: ConstraintType = "unconstrained"
) -> list[str]:
    """Get the list of problems to run, excluding those with setup errors."""
    if precision is None:
        precision = 64
    assert precision in [16, 32, 64]
    json_path = _json_path(constraints, "valid_problems")
    assert os.path.exists(json_path)
    with open(json_path) as f:
        data = json.load(f)
    return data.get("valid_problems", {}).get(f"precision_{precision}", [])


def get_n(problem_name: str) -> int:
    """Get the dimension n of a problem."""
    prop = pycutest.problem_properties(problem_name)
    if isinstance(prop["n"], int):
        return prop["n"]
    prob = pycutest.import_problem(problem_name)
    return prob.n


def make_n_table(force: bool = False, constraints: ConstraintType = "unconstrained"):
    """Generate and save the dimension table for a problem class."""
    json_path = _json_path(constraints, "n_table")

    problems = problemsToRun(None, constraints)

    # Load existing data if available
    if os.path.exists(json_path):
        with open(json_path) as f:
            n_table = json.load(f)
    else:
        n_table = {}

    for prob in problems:
        if prob in n_table and not force:
            continue
        try:
            n_table[prob] = get_n(prob)
            with open(json_path, "w") as f:
                json.dump(dict(sorted(n_table.items())), f, indent=2)
            print(f"  {prob}: n={n_table[prob]}")
        except Exception as e:  # noqa: BLE001 - continue checking independent problems
            print(f"  {prob}: Error - {e}")

    print(f"\nSaved to {json_path}")


def check_problem_at_precision(
    problem_name: str, precision: int, timeout: int = 100
) -> bool | None:
    """
    Check if gradient can be computed stably at given precision.
    Returns True if valid, False if invalid, None if timeout.
    """

    def worker(q, name, prec):
        try:
            prob = CUTEstQNProblem(name, precision=prec)
            g = prob.g(prob.x0, count=False)
            is_valid = np.all(np.isfinite(g))
            q.put(("ok", is_valid))
        except Exception as e:  # noqa: BLE001 - report worker failures via the queue
            q.put(("err", str(e)))

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, problem_name, precision))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None

    try:
        kind, payload = q.get_nowait()
        return bool(payload) if kind == "ok" else False
    except Exception:  # noqa: BLE001 - an invalid worker result means validation failed
        return False


def check_initial_gradient(
    force: bool = False, constraints: ConstraintType = "unconstrained"
):
    """Check gradient computation stability across precisions."""
    json_path = _json_path(constraints, "valid_problems")
    problems = problemsToRun(None, constraints)

    print(f"Checking {len(problems)} problems across precisions 16, 32, 64...")
    print("=" * 80)

    # Load or initialize results
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {
            "valid_problems": {f"precision_{p}": [] for p in [16, 32, 64]},
            "invalid_problems": {f"precision_{p}": [] for p in [16, 32, 64]},
            "skipped_problems": [
                # List of problems that can be excluded because either:
                #  * the initial point is already stationary (FLETCBV2)
                #  * all algorithms are known to fail (the rest)
                "BA-L16LS",
                "BA-L21LS",
                "BA-L49LS",
                "BA-L52LS",
                "BA-L73LS",
                "CURLY30",
                "FLETCBV2",
                "FLETCBV3",
                "FLETCHBV",
                "INDEF",
                "NONMSQRT",
                "SBRYBND",
                "SCOSINE",
                "SCURLY10",
                "SCURLY20",
                "SCURLY30",
                "SSCOSINE",
            ]
            if constraints == "unconstrained"
            else [],
        }

    # Convert to sets for efficient lookup
    v_sets = {k: set(v) for k, v in data["valid_problems"].items()}
    inv_sets = {k: set(v) for k, v in data["invalid_problems"].items()}
    skipped_set = set(data["skipped_problems"])

    def save():
        """Persist results to JSON."""
        with open(json_path, "w") as f:
            json.dump(
                {
                    "valid_problems": {k: sorted(v_sets[k]) for k in v_sets},
                    "invalid_problems": {k: sorted(inv_sets[k]) for k in inv_sets},
                    "skipped_problems": sorted(skipped_set),
                },
                f,
                indent=2,
            )

    # Check each problem
    for i, problem_name in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {problem_name}...", end=" ")

        if problem_name in skipped_set:
            print("SKIP")
            continue

        results = {}
        for precision in [16, 32, 64]:
            key = f"precision_{precision}"

            if not force and problem_name in v_sets[key]:
                results[precision] = "✓"
                continue
            if not force and problem_name in inv_sets[key]:
                results[precision] = "✗"
                continue

            is_valid = check_problem_at_precision(problem_name, precision)

            if is_valid is None:
                skipped_set.add(problem_name)
                save()
                results[precision] = "S"
                break
            elif is_valid:
                v_sets[key].add(problem_name)
                results[precision] = "✓"
            else:
                inv_sets[key].add(problem_name)
                results[precision] = "✗"
            save()

        status = " ".join([f"{p}:{results.get(p, '?')}" for p in [16, 32, 64]])
        print(status)

    print("=" * 80)
    print(f"Results saved to {json_path}\n")

    # Summary
    for precision in [16, 32, 64]:
        key = f"precision_{precision}"
        valid = len(v_sets[key])
        invalid = len(inv_sets[key])
        print(f"Precision {precision}-bit: {valid} valid, {invalid} invalid")
    print(f"Skipped: {len(skipped_set)}")


if __name__ == "__main__":
    constraints_list: tuple[ConstraintType, ...] = ("unconstrained", "bound")
    for constraints in constraints_list:
        print("\n=== Making n_table ===")
        make_n_table(force=True, constraints=constraints)
        print("\n=== Checking gradients ===")
        check_initial_gradient(force=True, constraints=constraints)
