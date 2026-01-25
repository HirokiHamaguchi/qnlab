# QNLab

https://pypi.org/project/qnlab/

QNLab is a research repository containing the implementation of our paper "Practical Regularized Quasi-Newton Methods with Inexact Function Values." It also includes various quasi-Newton methods, specifically focusing on L-BFGS variants.

## Features

(Under Construction)

## Setup (uv)

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) CLI
- Git with submodule support

### Clone the repository

```
git clone --recurse-submodules https://github.com/HirokiHamaguchi/qnlab.git
cd qnlab
```

If you already cloned without the `--recurse-submodules` flag, pull the submodules explicitly so that `pytest` can access the bundled benchmark libraries:

```
git submodule update --init --recursive
```

### Install dependencies with uv

```
uv sync
```

### Run tests

```
uv run pytest
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
