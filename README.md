# Modular MLP in C++

Academic-oriented, from-scratch implementation of a modular multilayer perceptron (MLP), designed for easy experimentation with layers, losses, optimizers, and compute backends.

## Quick Start

Requirements:
- CMake >= 3.16
- C++17 compiler (`g++` or `clang++`)

Run:

```bash
cmake -S . -B build
cmake --build build
./build/mlp
```

Choose optimizer from CLI:

```bash
./build/mlp --optimizer sgd
./build/mlp --optimizer momentum
./build/mlp --optimizer adam
./build/mlp --optimizer adamw
```

Choose hidden-layer layout from CLI:

```bash
./build/mlp --hidden 8
./build/mlp --hidden 16,16
./build/mlp --optimizer adamw --hidden 32,16,8
```

Set training and evaluation options from CLI:

```bash
./build/mlp --epochs 3000 --lr 0.01 --samples 1000
./build/mlp --train-ratio 0.7 --val-ratio 0.15 --threshold 0.5
```

The program now trains on a train split and reports loss + binary metrics on validation/test splits.

## Build Options

OpenMP (CPU parallelism):

```bash
cmake -S . -B build-omp -DMLP_ENABLE_OPENMP=ON
cmake --build build-omp
./build-omp/mlp
```

CUDA (GPU dense ops):

```bash
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON
cmake --build build-cuda
./build-cuda/mlp
```

If CUDA is not detected, configure `CUDAToolkit_ROOT`.

## Documentation

- Full tutorial: `docs/TUTORIAL.md`
- Experiments log template: `docs/EXPERIMENTS.md`
- API policy: `docs/API_POLICY.md`
- Main experiment entrypoint: `src/main.cpp`
- Metrics module: `include/mlp/metrics.hpp`, `src/metrics.cpp`

## Use as Library

You can link against the `mlp_lib` target and call:
- `mlp::Matrix` / `mlp::Vector` from `include/mlp/types.hpp`
- `mlp::run_xor_experiment(...)` from `include/mlp/library.hpp`
- `mlp::save_sequential(...)` / `mlp::load_sequential(...)` from `include/mlp/io.hpp`
- `mlp::BinaryMetrics` / `mlp::compute_binary_metrics(...)` from `include/mlp/metrics.hpp`
- version macros from `mlp/version.hpp` (for example `MLP_VERSION_STRING`)

Available CMake targets:
- `mlp::mlp_core`
- `mlp::mlp_optim`
- `mlp::mlp_train`
- `mlp::mlp_io`
- `mlp::mlp_lib` (compatibility aggregate target)

Example:

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

In another CMake project:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_lib)
```

Or link only specific components:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_train mlp::mlp_io)
```

If installed in a custom prefix, configure your consumer with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/mlp-install
```

## Optimizers Included

- `SGD`
- `Momentum`
- `Adam`
- `AdamW`
- `LambdaOptimizer` (custom extension hook)

## Project Layout

- `include/`: interfaces and math utilities
- `src/`: implementations and runnable example
- `docs/`: guides and tutorials

## Testing

Run all automated tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Current suite includes:
- training/evaluation integration test
- model save/load roundtrip test
- installed package consumer test (`find_package(mlp)`)
