# Codex Project Context

Last updated: 2026-06-12

## Snapshot

This repository is `modular_mlp`, a C++17 multilayer perceptron project with:

- a CLI experiment binary: `mlp`
- reusable CMake library targets
- optional OpenMP, BLAS, and CUDA backends
- CTest-based unit, integration, gradient, package-consumer, and backend tests

The project is academic and intentionally from-scratch. Prefer clear, inspectable C++ over bringing in broad framework dependencies.

## Current Local State

Recent cleanup work reduced duplicate code:

- Optimizer parameter traversal and state initialization are centralized in `src/optimizer.cpp`.
- Repeated test comparison helpers are centralized in `tests/test_helpers.hpp`.
- Activation, dense, and loss tests now reuse those helpers.

Untracked `.agents/.gitkeep` and `.codex/.gitkeep` may be created by the VS Code extension and are not meaningful project logic.

## Main Commands

Default CPU build:

```bash
cmake -S . -B build
cmake --build build
```

Run all configured tests:

```bash
ctest --test-dir build --output-on-failure
```

Run the local pre-push check:

```bash
./scripts/pre_push_check.sh
```

Optional backend configure commands:

```bash
cmake -S . -B build-omp -DMLP_ENABLE_OPENMP=ON
cmake -S . -B build-blas -DMLP_ENABLE_BLAS=ON
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON
```

BLAS benchmark gate:

```bash
./build-blas/mlp_dense_benchmark --min-speedup 2.0
```

## Repository Map

- `include/mlp/`: stable public API headers for package consumers
- `include/`: internal implementation headers
- `src/`: implementation files
- `tests/`: test executables and shared test helpers
- `examples/`: library and I/O usage examples
- `benchmarks/`: dense backend performance checks
- `docs/`: tutorial, API policy, experiment notes, and GitHub Pages source
- `cmake/`: package config templates and install-consumer checks

## API Boundaries

Stable public API headers:

- `include/mlp/types.hpp`
- `include/mlp/metrics.hpp`
- `include/mlp/library.hpp`
- `include/mlp/io.hpp`
- generated `mlp/version.hpp`

Stable public CMake targets:

- `mlp::mlp_core`
- `mlp::mlp_optim`
- `mlp::mlp_train`
- `mlp::mlp_io`
- `mlp::mlp_lib`

Internal/unstable headers:

- `include/matrix.hpp`
- `include/dense.hpp`
- `include/optimizer.hpp`
- `include/model.hpp`
- `include/layer.hpp`
- `include/loss.hpp`
- `include/activations.hpp`
- `include/cuda_ops.hpp`

Follow `docs/API_POLICY.md` for any public API or package-target change.

## Coding Guidance

- Keep code C++17-compatible.
- Follow the existing CMake target layout.
- Keep changes focused; avoid broad unrelated refactors.
- Prefer local helpers for repeated matrix/vector traversal logic.
- Reuse `tests/test_helpers.hpp` for test assertions instead of duplicating comparison functions.
- Treat `build/` and `build-*` as generated outputs, not source.
- Keep public API additions documented in `README.md`, `docs/TUTORIAL.md`, and `docs/API_POLICY.md` when appropriate.

## Verification Expectations

For normal source changes:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

For API/package changes, also verify install-consumer behavior with the existing package consumer test or:

```bash
./scripts/pre_push_check.sh
```

For backend-specific changes:

- OpenMP: configure/build `build-omp`
- BLAS: configure/build `build-blas`, then run `mlp_dense_benchmark`
- CUDA: configure/build `build-cuda`

## Good First Places To Inspect

- CLI parsing and experiment entry: `src/main.cpp`
- Programmatic experiment API: `include/mlp/library.hpp`, `src/library.cpp`
- Layers and model: `include/layer.hpp`, `include/model.hpp`, `include/dense.hpp`, `src/dense.cpp`
- Optimizers: `include/optimizer.hpp`, `src/optimizer.cpp`
- Tests: `tests/`
