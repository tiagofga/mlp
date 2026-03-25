# Modular MLP in C++

[![CI](https://github.com/tiagofga/mlp/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagofga/mlp/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/tiagofga/mlp)](https://github.com/tiagofga/mlp/releases) [![License](https://img.shields.io/github/license/tiagofga/mlp)](./LICENSE) [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/) [![OpenMP Optional](https://img.shields.io/badge/OpenMP-optional-00599C.svg)](https://www.openmp.org/) [![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900.svg)](https://developer.nvidia.com/cuda-toolkit) [![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/tiagofga/mlp/pulls) [![Issues Welcome](https://img.shields.io/badge/issues-welcome-brightgreen.svg)](https://github.com/tiagofga/mlp/issues)

An academic-focused, from-scratch multilayer perceptron (MLP) project in modern C++ with both:
- a CLI application for experiments, and
- an installable CMake library package for reuse in other C++ projects.

Current scope:
- `C++17`
- CPU execution, optional `OpenMP`, optional `CUDA`
- CLI experiments and installable CMake library package

## Quick Start (CLI)

Requirements:
- CMake >= 3.16
- C++17 compiler (`g++` or `clang++`)

Build and run:

```bash
cmake -S . -B build
cmake --build build
./build/mlp
```

The CLI trains on a train split and reports loss and binary metrics on train, validation, and test.

## Main Options

Common CLI options:

```bash
./build/mlp --optimizer sgd|momentum|nag|adam|adamw|nadam|rmsprop|adagrad|adadelta|lion
./build/mlp --hidden 16,16,8
./build/mlp --epochs 3000 --lr 0.01
./build/mlp --samples 1000 --seed 42
./build/mlp --train-ratio 0.7 --val-ratio 0.15 --threshold 0.5
```

## Backend Options

OpenMP (CPU parallelism):

```bash
cmake -S . -B build-omp -DMLP_ENABLE_OPENMP=ON
cmake --build build-omp
./build-omp/mlp
```

CUDA (dense ops):

```bash
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON
cmake --build build-cuda
./build-cuda/mlp
```

CUDA notes:
- CUDA support is optional and currently accelerates dense-layer matrix operations.
- The current CUDA path is intended for correctness and experimentation, not peak throughput.
- If CUDA is not detected, configure CMake with `-DCUDAToolkit_ROOT=/path/to/cuda`.
- If `nvcc` is unavailable, use the default CPU build or the OpenMP build.

## Library Usage

Public API headers (stable surface):
- `include/mlp/types.hpp`
- `include/mlp/metrics.hpp`
- `include/mlp/library.hpp`
- `include/mlp/io.hpp`
- `include/mlp/version.hpp`

Main API entry points:
- `mlp::run_xor_experiment(...)`
- `mlp::save_sequential(...)`
- `mlp::load_sequential(...)`

CMake targets:
- `mlp::mlp_core`
- `mlp::mlp_optim`
- `mlp::mlp_train`
- `mlp::mlp_io`
- `mlp::mlp_lib` (compatibility aggregate target)

Example targets included in this repo:

```bash
cmake --build build --target mlp_library_example
./build/mlp_library_example

cmake --build build --target mlp_io_example
./build/mlp_io_example
```

## Install and `find_package`

Install locally:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix /tmp/mlp-install
```

Consume from another CMake project:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_lib)
```

Or link only components:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_train mlp::mlp_io)
```

If using custom install prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/mlp-install
```

## Documentation

- `docs/TUTORIAL.md`
- `docs/EXPERIMENTS.md`
- `docs/API_POLICY.md`

## Optimizers Included

| Name | CLI string | Notes |
|---|---|---|
| SGD | `sgd` | Vanilla stochastic gradient descent |
| Momentum | `momentum` | SGD with exponential moving-average velocity |
| NAG | `nag` | Nesterov Accelerated Gradient |
| Adam | `adam` | Adaptive moment estimation |
| AdamW | `adamw` | Adam + decoupled weight decay |
| Nadam | `nadam` | Adam with Nesterov momentum correction |
| RMSProp | `rmsprop` | Root mean square propagation |
| AdaGrad | `adagrad` | Adaptive per-parameter learning rates (accumulative) |
| AdaDelta | `adadelta` | AdaGrad variant with running averages, no fixed lr |
| Lion | `lion` | Evolved Sign Momentum — sign-based, memory-efficient |
| LambdaOptimizer | — | Custom extension hook via user-supplied lambdas |

## Testing and CI

Run all tests locally:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Pre-push local check:

```bash
./scripts/pre_push_check.sh
```

Test suite includes:
- training/evaluation integration test
- save/load roundtrip test
- installed package consumer test (`find_package(mlp)`)

CI (`.github/workflows/ci.yml`) runs:
- OpenMP matrix (`MLP_ENABLE_OPENMP=OFF/ON`)
- optional CUDA configure/build smoke check when `nvcc` is available

CI verification:
- use the `CI` badge at the top of this README
- use `./scripts/pre_push_check.sh` before pushing changes

## Scope

- Language: C++17
- Primary use case: academic experiments and library-oriented reuse
- Supported acceleration paths: CPU, OpenMP, and optional CUDA
