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

BLAS / CBLAS (CPU GEMM acceleration):

```bash
cmake -S . -B build-blas -DMLP_ENABLE_BLAS=ON
cmake --build build-blas
./build-blas/mlp_dense_benchmark
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

BLAS notes:
- `MLP_ENABLE_BLAS=ON` enables configure-time detection of Apple Accelerate on macOS or a CBLAS-compatible BLAS library such as OpenBLAS on Linux.
- If a compatible BLAS backend is not found, CMake emits a warning and the CPU path keeps using the existing fallback implementation.
- `mlp_dense_benchmark` compares Dense forward/backward against the naive CPU path on a `512x512` workload and reports the observed speedup.
- Use `./build-blas/mlp_dense_benchmark --min-speedup 2.0` on a BLAS-enabled build to assert the issue #4 acceptance threshold.

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

Safety and validation contracts:
- Matrix helpers expect rectangular `Matrix` values and throw `std::invalid_argument` on ragged inputs or shape mismatches.
- Matrix allocation helpers check `std::size_t` multiplication overflow before allocating.
- `Dense`, activation layers, and `BinaryCrossEntropy` require `forward(...)` before `backward(...)` and throw `std::logic_error` when that lifecycle is violated.
- Optimizers validate parameter/gradient references, matching shapes, and finite numeric hyperparameters before mutating weights.
- Model checkpoint loading validates the file magic, layer count, dense dimensions, finite numeric values, and trailing data before returning a model.
- Checkpoint loading is intended for trusted or reviewed model files; oversized or malformed files are rejected with exceptions.

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

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — contribution workflow, commit style, tests, and review checklist
- [`ROADMAP.md`](./ROADMAP.md) — phased feature roadmap linked to GitHub issues
- [`docs/TUTORIAL.md`](./docs/TUTORIAL.md) — detailed how-to for running, extending, and contributing
- [`docs/EXPERIMENTS.md`](./docs/EXPERIMENTS.md) — experiment log template for academic tracking
- [`docs/API_POLICY.md`](./docs/API_POLICY.md) — API compatibility policy and versioning
- [`docs/site/index.html`](./docs/site/index.html) — GitHub Pages website source
- [`.codex/README.md`](./.codex/README.md) — local Codex project context and verification notes
- [`.agents/README.md`](./.agents/README.md) — project agent-role notes for larger tasks

## Project Structure

Core source layout:

- `include/mlp/` contains the stable public API for package consumers.
- `include/` contains internal headers for layers, losses, optimizers, matrix utilities, and backends.
- `src/` contains implementation files.
- `tests/` contains focused test executables, finite-difference gradient checks, and shared test helpers.
- `examples/` contains small library and I/O roundtrip examples.
- `benchmarks/` contains dense backend performance checks.

Recent code-health cleanup:

- Optimizer parameter traversal and per-parameter state initialization are centralized in `src/optimizer.cpp`.
- Matrix/vector test comparison helpers are centralized in `tests/test_helpers.hpp`.
- Generated build directories such as `build/` and `build-*` are intentionally ignored.

## Project Website (GitHub Pages)

- URL: `https://tiagofga.github.io/mlp/`
- Source files: [`docs/site`](./docs/site)
- Deployment workflow: [`.github/workflows/pages.yml`](./.github/workflows/pages.yml)

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
- activation-layer unit tests for ReLU, Sigmoid, and Tanh forward/backward behavior
- loss-function unit tests for BinaryCrossEntropy forward/backward, clamp, and shape-mismatch behavior
- safety regression tests for non-rectangular matrices, invalid layer lifecycle calls, and malformed checkpoint files
- finite-difference gradient checks via `tests/gradient_check.hpp`, including `mlp_test_gradient_check` coverage for Dense and Tanh layers
- shared matrix/vector test assertions via `tests/test_helpers.hpp`
- training/evaluation integration test
- save/load roundtrip test
- installed package consumer test (`find_package(mlp)`)
- BLAS benchmark gate (`mlp_dense_benchmark --min-speedup 2.0`) when a BLAS backend is enabled

CI (`.github/workflows/ci.yml`) runs:
- OpenMP matrix (`MLP_ENABLE_OPENMP=OFF/ON`)
- Valgrind memcheck on Ubuntu (`ctest -T memcheck` with `valgrind --leak-check=full`, failing on definite leaks)
- BLAS configure/build/test/benchmark job with OpenBLAS on Ubuntu
- optional CUDA configure/build smoke check when `nvcc` is available

CI verification:
- use the `CI` badge at the top of this README
- use `./scripts/pre_push_check.sh` before pushing changes

## License

Released under the MIT License. The project is provided as-is, without warranty. GitHub issues are welcome.
