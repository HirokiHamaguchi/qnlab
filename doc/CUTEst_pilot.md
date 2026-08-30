# CUTEst pilot run

The checked-in `notebooks/cutest.ipynb` configuration is intentionally tiny and must be used before any full rerun.

## Exact pilot scope

- Scenario: `float64` (no artificial function or gradient noise)
- Problems: exactly `AKIVA`, `ARWHEAD`, and `ROSENBR`
- Methods: exactly `NTRQN` and `OFFO`
- Seed: exactly `0`
- Total: exactly 6 solver tasks
- Score tolerances: `1e-1`, `1e-3`, and `1e-5`

`OFFO` is a standalone first-order baseline whose steps use only the current gradient and the cumulative gradient-norm scale. It does not evaluate function values; its stored function-value field is therefore `NaN`.

The separate `NTRQN-OFFO` method remains available as a branch ablation: it forces the proposed quasi-Newton method into its OFFO-inspired branch at every iteration. It must not be described as the standalone OFFO baseline.

The result summary reports both the percentage solved and the percentage for which each method used the fewest oracle calls, with ties counted for every tied method. The 64-bit, `1e-3` row is the candidate headline number for the Introduction table after the full benchmark has been run. Pilot percentages are diagnostic only and must not be reported as full-benchmark evidence.

Each result file records the solver return code in addition to the execution status. The summary cell separately reports missing files, timeouts, exceptions, error return codes, restart-count statistics, and the original NTQN termination flags. It also generates data profiles normalized by `dimension + 1`.

The primary function-noise profiles exclude `NTRQN-Restart`. When the full `function_only` scenario is plotted, the notebook generates a separate `NTRQN` versus `NTRQN-Restart` appendix profile and restart summary.

All paper-specific scenario values, method variants, solver-option overrides, limits, seeds, and ablation settings are declared in `notebooks/cutest.ipynb`. The solver implementations and the checked valid-problem lists remain in their library and data modules.

The misspecification study compares only `NTRQN` and `NTRQN-MS`. The nominal setting is identical to those methods' `function_only` tasks, so it is loaded from that scenario rather than rerun. Exact-gradient `OFFO` is run only with seed `0` and reused for the other noisy seeds. The dry-run preview prints the number of tasks omitted through such reuse.

## Running the pilot

Open `notebooks/cutest.ipynb` from the repository root and first execute it with the default dry-run setting. The task preview must say `TINY PILOT (3 problems only)` and `Prepared 6 tasks.`

Then set `QNLAB_RUN_CUTEST=1` in the notebook environment and rerun the configuration and execution cells. Do not set `QNLAB_FULL_CUTEST` for the pilot.

Results are written under `data/temp/float64/seed_0/`. Existing results are not replaced unless `QNLAB_OVERWRITE_CUTEST=1` is also set.

## Full-suite safety switch

Selecting every valid CUTEst problem requires the explicit environment variable `QNLAB_FULL_CUTEST=1`. This changes `PROBLEMS_TO_RUN` from the three named pilot problems to `None` and restores the scenario-default method sets. Always inspect the printed task count before also enabling `QNLAB_RUN_CUTEST=1`.
