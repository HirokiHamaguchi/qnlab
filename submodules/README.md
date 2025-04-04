# submodules

This folder contains submodules for the project. They are not directly used in the code, but are included for reference or for future use.

- [submodules](#submodules)
  - [DirL-BFGS](#dirl-bfgs)
  - [LBFGSpp](#lbfgspp)
  - [liblbfgs](#liblbfgs)
  - [mL-BFGS](#ml-bfgs)
  - [paper-regularized-qn-benchmark](#paper-regularized-qn-benchmark)
  - [py-owlqn](#py-owlqn)
  - [pylbfgs-dedupeio](#pylbfgs-dedupeio)
  - [pylbfgs-larsmans](#pylbfgs-larsmans)
  - [python\_lbfgsb](#python_lbfgsb)

## DirL-BFGS

[GitHub Link](https://github.com/ashkansl/DirL-BFGS)

[Sadeghi-Lotfabadi, Ashkan, and Kamaledin Ghiasi-Shirazi. "Speeding up L-BFGS by direct approximation of the inverse Hessian matrix." Computational Optimization and Applications (2025): 1-28.](https://doi.org/10.1007/s10589-025-00665-0)

This repository implements Direct L-BFGS (DirL-BFGS) method that, seeing H as a linear operator, directly stores a low-rank plus diagonal (LRPD) representation of H.

## LBFGSpp

[GitHub Link](https://github.com/yixuan/LBFGSpp)

A header-only C++ library for L-BFGS and L-BFGS-B algorithms

## liblbfgs

[GitHub Link](https://github.com/chokkan/liblbfgs)

libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)

This is a C port of the implementation of [Limited-memory
Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal](https://users.iems.northwestern.edu/~nocedal/lbfgs.html).

## mL-BFGS

[GitHub Link](https://github.com/yuehniu/mL-BFGS?tab=readme-ov-file)

[Niu, Yue, Zalan Fabian, Sunwoo Lee, Mahdi Soltanolkotabi, and Salman Avestimehr. "Ml-bfgs: A momentum-based l-bfgs for distributed large-scale neural network optimization." arXiv preprint arXiv:2307.13744 (2023).](https://arxiv.org/abs/2307.13744)

Momentum-based L-BFGS for Distributed
Large-Scale Neural Network Optimization

## paper-regularized-qn-benchmark

[GitHub Link](https://github.com/dmsteck/paper-regularized-qn-benchmark)

[Kanzow, Christian, and Daniel Steck. "Regularization of limited memory quasi-Newton methods for large-scale nonconvex minimization." Mathematical Programming Computation 15, no. 3 (2023): 417-444.](https://doi.org/10.1007/s12532-023-00238-4)

This paper implements regularized version of L-BFGS methods.

## py-owlqn

[GitHub Link](https://github.com/samson-wang/py-owlqn.git)

A python implementation of owlqn(lbfgs) optimization algorithm. A logistic regression training and testing example also included.

## pylbfgs-dedupeio

[GitHub Link](https://github.com/dedupeio/pylbfgs)

Python/Cython wrapper for liblbfgs by dedupeio.

## pylbfgs-larsmans

[GitHub Link](https://github.com/larsmans/pylbfgs)

Python/Cython wrapper for liblbfgs by larsmans.

## python_lbfgsb

[GitHub Link](https://github.com/avieira/python_lbfgsb)

Pure Python-based L-BFGS-B implementation.

This repository uses `scipy.optimize.minpack2`, which is deprecated and causes errors when running the code (at 2025-04-04).
