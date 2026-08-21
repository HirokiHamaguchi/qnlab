# QNLab

QNLab is a research repository containing the implementation of our paper "Practical Regularized Quasi-Newton Methods with Inexact Function Values." It also includes various quasi-Newton methods, specifically focusing on L-BFGS variants.

## Features

- **Noise-tolerant L-BFGS method:**
  We developed a noise-tolerant and practically fast L-BFGS method.

  ![Performance profile under noisy evaluations](doc/imgs/compare/_pp_noise0.001_gtol1e-02.png)
  ![legend](doc/imgs/compare/_legend.png)

- **Quasi-Newton methods in pure Python:**
  We also provide implementations of various quasi-Newton methods in pure Python, which can be easily used and modified for research purposes.

- **Documents about Quasi-Newton Methods:**
  We have some documents about basic of quasi-Newton methods.
  See [doc/study/0_main.pdf](doc/study/0_main.pdf).

## Setup for Users

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) CLI

### Install dependencies with uv

```bash
uv sync
```

## Setup for Developers

### Additional Requirements

- Git with submodule support
- PyCUTEst installed and configured (for running tests involving CUTEst problems)
  - https://github.com/ralna/CUTEst
  - https://github.com/jfowkes/pycutest

### Pull git submodules

In `pytest`, we use some libraries bundled as git submodules.
You can explicitly pull the submodules using:

```bash
git submodule update --init --recursive
```

To clone the repository with submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/HirokiHamaguchi/qnlab.git
cd qnlab
```

### Run tests involving CUTEst problems

To run tests that involve CUTEst problems, ensure that CUTEst is properly installed and configured on your system.

To examine and generate cache files for CUTEst problems, you can run:

```bash
uv run data/CUTEst/check_CUTEst_problems.py
```

If it run successfully, you can now run tests involving CUTEst problems using:

```bash
uv run pytest
```

### Reproduce the paper experiments

[`notebooks/cutest.ipynb`](notebooks/cutest.ipynb) is the canonical entry point
for reproducing all CUTEst experiments reported in the paper. A single
configuration-driven workflow covers the 64-, 32-, and 16-bit experiments and
the explicit function- and gradient-noise scenarios used in the MPC response.
Run the notebook from the repository environment after configuring PyCUTEst.

The deterministic precision scenarios use seed `0`. Each noisy scenario uses the
five paired seeds `0`, `1`, `2`, `3`, and `4`; every solver receives the same seed
for a given problem and scenario.

The full CUTEst benchmark is long-running. In the notebook's configuration cell,
first select a small set with `PROBLEMS_TO_RUN`, `SCENARIOS_TO_RUN`, and
`METHODS_TO_RUN`, and leave `RUN_EXPERIMENTS = False`. After checking the displayed
task list, set `RUN_EXPERIMENTS = True` and run the execution cell manually.
Results are saved under `data/temp/`; existing results are skipped unless
`OVERWRITE_EXISTING = True` is selected deliberately.

### Install git hooks

Install the pre-commit hook once per clone so that formatting, lint, type, and test checks run automatically before every commit:

```bash
uv run pre-commit install --hook-type pre-commit
```

## Documentation

This repository contains some study materials.

## References

The following is the list of repositories included in our repository as git submodules, or related projects that we would like to acknowledge.
Thanks to the developers of these projects.

### Included in Our Repository as Git Submodules

* [LBFGSpp](https://github.com/yixuan/LBFGSpp)
* [liblbfgs](https://github.com/chokkan/liblbfgs)
* [paper-regularized-qn-benchmark](https://github.com/dmsteck/paper-regularized-qn-benchmark)
* [noise-tolerant-bfgs](https://github.com/hjmshi/noise-tolerant-bfgs.git)

### Cubic Regularized (Quasi-)Newton Methods

* [ARNCG](https://github.com/miskcoo/ARNCG.git)
* [krylov-cubic-regularized-newton](https://github.com/amazon-science/krylov-cubic-regularized-newton.git)
* [super-newton](https://github.com/doikov/super-newton.git)

### Other L-BFGS Implementations and Related Resources

* [DirL-BFGS](https://github.com/ashkansl/DirL-BFGS)
* [mL-BFGS](https://github.com/yuehniu/mL-BFGS?tab=readme-ov-file)
* [py-owlqn](https://github.com/samson-wang/py-owlqn.git)
* [pylbfgs-dedupeio](https://github.com/dedupeio/pylbfgs)
* [pylbfgs-larsmans](https://github.com/larsmans/pylbfgs)
* [python_lbfgsb](https://github.com/avieira/python_lbfgsb)
* [self_scaled_algorithms_pinns](https://github.com/jorgeurban/self_scaled_algorithms_pinns.git)

### Benchmarks

* [Python_Benchmark_Test_Optimization_Function_Single_Objective](https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective.git): A collection of benchmark test functions in Python.
* [opfunu](https://github.com/thieu1995/opfunu.git): A collection of Benchmark functions for numerical optimization problems

## License

This project is licensed under the MIT License. (`liblbfgs` is under the MIT License as well).

See the [LICENSE](LICENSE) file for details.
