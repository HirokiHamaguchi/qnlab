"""Audit saved CUTEst runs and cross-check percentages reported in LaTeX."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "data" / "temp"
PROBLEM_LIST = REPOSITORY_ROOT / "data" / "CUTEst" / "valid_problems.json"
PRIVATE_ROOT = REPOSITORY_ROOT.parent / "qnlab_private"

STANDARD_METHODS = {
    "NTRQN",
    "NTRQN-MS",
    "NTRQN-SP",
    "NTRQN-MS-SP",
    "Line",
    "Line-MS",
    "ASTR1-Adagrad",
    "Reg",
    "Reg-Sec",
    "SciPy",
    "NTQN",
}
PROPOSED_METHODS = {
    "NTRQN",
    "NTRQN-MS",
    "NTRQN-SP",
    "NTRQN-MS-SP",
    "NTRQN-Restart",
    "NTRQN-MS-Restart",
}
SCENARIOS = {
    "float64": (64, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "float32": (32, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "float16": (16, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "function_only": (
        64,
        tuple(range(5)),
        (1e-2,),
        STANDARD_METHODS
        | {"NTRQN-Restart", "NTRQN-MS-Restart", "NTQN-Default-Termination"},
    ),
    "joint_noise": (64, tuple(range(5)), (1e-2,), STANDARD_METHODS),
    "eps_under": (64, tuple(range(5)), (1e-2,), PROPOSED_METHODS),
    "eps_nominal": (64, tuple(range(5)), (1e-2,), PROPOSED_METHODS),
    "eps_over": (64, tuple(range(5)), (1e-2,), PROPOSED_METHODS),
}
SCENARIO_ALIASES = {"eps_nominal": "function_only"}
LATEX_FILES = (
    REPOSITORY_ROOT / "doc" / "main" / "abstract.tex",
    REPOSITORY_ROOT / "doc" / "main" / "introduction.tex",
    REPOSITORY_ROOT / "doc" / "main" / "main_contents.tex",
    REPOSITORY_ROOT / "doc" / "main" / "appendix.tex",
    PRIVATE_ROOT / "doc" / "response_letter_MPC_draft.tex",
)
PERCENT_PATTERN = re.compile(r"(?<![\d.])(\d+\.\d+)\\%")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "cutest_result_summary.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a partial summary instead of failing on missing result files",
    )
    return parser.parse_args()


def load_problem_sets() -> dict[int, tuple[str, ...]]:
    data = json.loads(PROBLEM_LIST.read_text(encoding="utf-8"))["valid_problems"]
    return {
        precision: tuple(data[f"precision_{precision}"])
        for precision in (16, 32, 64)
    }


def expected_option_subset(method: str) -> dict[str, object]:
    if method in {"NTRQN", "NTRQN-MS", "NTRQN-Restart", "NTRQN-MS-Restart"}:
        expected: dict[str, object] = {"regularization_solver": "compact"}
    elif method in {"NTRQN-SP", "NTRQN-MS-SP"}:
        expected = {"regularization_solver": "shifted_pair"}
    else:
        return {}
    if method in {"NTRQN-MS", "NTRQN-MS-SP", "NTRQN-MS-Restart"}:
        expected["modified_secant_max_ratio"] = 1.0
    return expected


def read_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path) as data:
        calls = np.asarray(data["calls"], dtype=int)
        gnorms = np.asarray(data["gnorms"], dtype=float)
        metadata = json.loads(data["metadata"].item())
    if calls.shape != gnorms.shape:
        raise ValueError(f"calls and gnorms have different shapes: {path}")
    return calls, gnorms, metadata


def validate_metadata(
    metadata: dict, scenario: str, seed: int, problem: str, method: str
) -> list[str]:
    source_scenario = SCENARIO_ALIASES.get(scenario, scenario)
    expected = {
        "scenario": source_scenario,
        "seed": seed,
        "problem": problem,
        "method": method,
    }
    errors = [
        f"{key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    options = metadata.get("options", {})
    errors.extend(
        f"options.{key}={options.get(key)!r}, expected {value!r}"
        for key, value in expected_option_subset(method).items()
        if options.get(key) != value
    )
    return errors


def first_successful_call(
    calls: np.ndarray, gnorms: np.ndarray, tolerance: float
) -> int | None:
    reached = np.flatnonzero(gnorms <= tolerance)
    if reached.size == 0:
        return None
    return max(1, int(calls[reached[0]]))


def summarize_results(
    result_root: Path,
) -> tuple[dict, list[str], dict[str, list[str]]]:
    problem_sets = load_problem_sets()
    errors: list[str] = []
    summary: dict[str, dict] = {}
    percentage_sources: dict[str, list[str]] = defaultdict(list)

    for scenario, (precision, seeds, tolerances, methods) in SCENARIOS.items():
        source_scenario = SCENARIO_ALIASES.get(scenario, scenario)
        problems = problem_sets[precision]
        scenario_summary: dict[str, dict] = {}
        for tolerance in tolerances:
            tolerance_summary: dict[str, dict] = {}
            for method in sorted(methods):
                successful_calls: list[int] = []
                status_counts: Counter[str] = Counter()
                diagnostics: Counter[str] = Counter()
                missing = 0
                invalid = 0
                seed_successes: dict[str, int] = {str(seed): 0 for seed in seeds}
                for seed in seeds:
                    for problem in problems:
                        path = (
                            result_root
                            / source_scenario
                            / f"seed_{seed}"
                            / problem
                            / f"{method}.npz"
                        )
                        if not path.is_file():
                            missing += 1
                            continue
                        try:
                            calls, gnorms, metadata = read_result(path)
                        except (
                            OSError,
                            KeyError,
                            ValueError,
                            json.JSONDecodeError,
                        ) as error:
                            errors.append(f"Unreadable result {path}: {error}")
                            invalid += 1
                            continue
                        metadata_errors = validate_metadata(
                            metadata, scenario, seed, problem, method
                        )
                        if metadata_errors:
                            details = "; ".join(metadata_errors)
                            errors.append(f"Metadata mismatch in {path}: {details}")
                            invalid += 1
                            continue
                        status_counts[metadata.get("status", "unknown")] += 1
                        for key, value in metadata.get("diagnostics", {}).items():
                            if isinstance(value, (int, float)):
                                diagnostics[key] += value
                        first_call = first_successful_call(calls, gnorms, tolerance)
                        if first_call is not None:
                            successful_calls.append(first_call)
                            seed_successes[str(seed)] += 1

                total = len(problems) * len(seeds)
                solved = len(successful_calls)
                percentage = f"{100.0 * solved / total:.1f}"
                source = f"{scenario}, gtol={tolerance:g}, {method}"
                percentage_sources[percentage].append(source)
                seed_percentages = {
                    seed: round(100.0 * count / len(problems), 10)
                    for seed, count in seed_successes.items()
                }
                tolerance_summary[method] = {
                    "instances": total,
                    "valid_results": total - missing - invalid,
                    "missing_results": missing,
                    "invalid_results": invalid,
                    "solved": solved,
                    "failed_common_tolerance": total - solved,
                    "solved_percent": float(percentage),
                    "median_calls_when_solved": (
                        float(np.median(successful_calls))
                        if successful_calls
                        else None
                    ),
                    "seed_solved_percent": seed_percentages,
                    "seed_solved_percent_std": (
                        float(np.std(list(seed_percentages.values()), ddof=1))
                        if len(seed_percentages) > 1
                        else None
                    ),
                    "status_counts": dict(sorted(status_counts.items())),
                    "diagnostics": dict(sorted(diagnostics.items())),
                }
                if missing:
                    errors.append(
                        f"{scenario}, {method}: missing {missing} of "
                        f"{total} result files"
                    )
            scenario_summary[f"{tolerance:g}"] = tolerance_summary
        summary[scenario] = scenario_summary
    return summary, errors, percentage_sources


def extract_latex_percentages() -> tuple[set[str], dict[str, list[str]]]:
    sources: dict[str, list[str]] = defaultdict(list)
    for path in LATEX_FILES:
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            for value in PERCENT_PATTERN.findall(line):
                relative_path = path.relative_to(REPOSITORY_ROOT.parent)
                sources[value].append(f"{relative_path}:{line_number}")
    return set(sources), sources


def main() -> int:
    arguments = parse_arguments()
    summary, result_errors, percentage_sources = summarize_results(
        arguments.result_root
    )
    computed_percentages = set(percentage_sources)
    latex_percentages, latex_sources = extract_latex_percentages()
    missing_percentages = sorted(
        latex_percentages - computed_percentages, key=float
    )
    if latex_percentages == computed_percentages:
        relation = "equal"
    elif latex_percentages < computed_percentages:
        relation = "proper_subset"
    else:
        relation = "not_subset"

    report = {
        "scenarios": summary,
        "percentage_check": {
            "relation": relation,
            "latex_percentages": sorted(latex_percentages, key=float),
            "computed_percentages": sorted(computed_percentages, key=float),
            "latex_values_missing_from_computed_results": missing_percentages,
            "latex_sources": dict(
                sorted(latex_sources.items(), key=lambda item: float(item[0]))
            ),
            "computed_sources": dict(
                sorted(percentage_sources.items(), key=lambda item: float(item[0]))
            ),
        },
        "audit_errors": result_errors,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {arguments.output}")
    print(f"LaTeX percentage set relation: {relation}")
    if missing_percentages:
        print("LaTeX percentages absent from computed success rates:")
        for value in missing_percentages:
            print(f"  {value}%: {', '.join(latex_sources[value])}")
    if result_errors:
        print(f"Result audit found {len(result_errors)} issue(s).")
        for error in result_errors[:20]:
            print(f"  {error}")
        if len(result_errors) > 20:
            print(f"  ... and {len(result_errors) - 20} more; see the JSON report.")

    if missing_percentages:
        return 1
    if result_errors and not arguments.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
