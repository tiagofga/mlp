# Modular MLP in C++

Academic-oriented, from-scratch multilayer perceptron (MLP) project with both:
- a CLI app for experiments, and
- an installable CMake library package for reuse in other projects.

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

Main runtime options:

```bash
./build/mlp --optimizer sgd|momentum|adam|adamw
./build/mlp --hidden 16,16,8
./build/mlp --epochs 3000 --lr 0.01
./build/mlp --samples 1000 --seed 42
./build/mlp --train-ratio 0.7 --val-ratio 0.15 --threshold 0.5
```

The CLI trains on train split and reports loss/metrics on train, validation, and test.

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

If CUDA is not detected, set `CUDAToolkit_ROOT`.

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

## Testing and CI

Run all tests locally:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Test suite includes:
- training/evaluation integration test
- save/load roundtrip test
- installed package consumer test (`find_package(mlp)`)

CI (`.github/workflows/ci.yml`) runs:
- OpenMP matrix (`MLP_ENABLE_OPENMP=OFF/ON`)
- optional CUDA configure/build smoke check when `nvcc` is available

Pre-push local check:

```bash
./scripts/pre_push_check.sh
```

## Documentation

- `docs/TUTORIAL.md`
- `docs/EXPERIMENTS.md`
- `docs/API_POLICY.md`

## Optimizers Included

- `SGD`
- `Momentum`
- `Adam`
- `AdamW`
- `LambdaOptimizer` (custom extension hook)
