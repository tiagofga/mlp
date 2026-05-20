# Tutorial: Running and Extending the Modular MLP

This guide shows how to run the current MLP, edit experiments, and contribute new modules.

## 1. Prerequisites

- CMake >= 3.16
- C++17 compiler (`g++` or `clang++`)
- Optional for GPU: CUDA toolkit (`nvcc`)
- Optional for CPU threading: OpenMP runtime/toolchain support

## 2. Build and Run (CPU)

Build commands and all CLI flags are in [Quick Start](../README.md#quick-start-cli) and [Main Options](../README.md#main-options) in the README.

Expected behavior:
- You will see train and validation losses during training.
- At the end, the program reports train/validation/test losses and binary metrics.

## 3. Optional Backends

For OpenMP, CUDA, and BLAS build commands and notes, see [Backend Options](../README.md#backend-options) in the README.

To validate the BLAS acceptance target from issue [#4](https://github.com/tiagofga/mlp/issues/4), run:

```bash
cmake -S . -B build-blas -DMLP_ENABLE_BLAS=ON
cmake --build build-blas
./build-blas/mlp_dense_benchmark --min-speedup 2.0
```

## 4. How Training Works

The default example is in `src/main.cpp`:
- Build synthetic XOR-style dataset
- Split into train/validation/test sets
- Build model with `Sequential` and layers
- Create loss (`BinaryCrossEntropy`) and optimizer (`SGD`)
- Training loop: forward -> loss -> backward -> optimizer step
- Final evaluation: loss and metrics for each split

Core module map:
- `include/layer.hpp`: base layer API
- `include/dense.hpp` + `src/dense.cpp`: dense layer
- `include/activations.hpp` + `src/activations.cpp`: activation layers
- `include/loss.hpp` + `src/loss.cpp`: loss functions
- `include/mlp/metrics.hpp` + `src/metrics.cpp`: binary classification metrics
- `include/optimizer.hpp` + `src/optimizer.cpp`: optimizer logic
- `include/model.hpp` + `src/model.cpp`: sequential container
- `include/matrix.hpp`: CPU matrix operations
- `include/cuda_ops.hpp` + `src/cuda_ops.cu`: CUDA backend operations

## 5. Edit the MLP for Your Experiment

## 5.1 Change architecture

For fast tests, use CLI:

```bash
./build/mlp --hidden 8
./build/mlp --hidden 16,16
./build/mlp --hidden 64,32,16
```

Typical changes:
- Increase hidden width (`8 -> 16/32/64`)
- Add more hidden layers by adding values to `--hidden`
- Edit `src/main.cpp` if you want different activation policies

## 5.2 Change hyperparameters

From CLI:
- `--epochs`
- `--lr`
- `--threshold`

Suggested sweeps:
- Learning rate: `0.01`, `0.05`, `0.1`, `0.5`
- Epochs: `1000`, `5000`, `10000`

## 5.3 Change dataset and splits

Current data is generated programmatically with `--samples` and random `--seed`.

Split control:
- `--train-ratio`
- `--val-ratio`
- test ratio is `1 - train - val`

To use your own dataset, replace the generator in `main.cpp` and keep:
- `x.size() == y.size()`
- first `Dense` input size = number of features
- output layer size = target dimensions

## 6. Add New Components

## 6.1 Add a new activation

1. Declare class in `include/activations.hpp` inheriting `Layer`.
2. Implement `forward` and `backward` in `src/activations.cpp`.
3. Add or extend focused coverage in `tests/test_activations.cpp`.
4. Use the new activation in `src/main.cpp`.

## 6.2 Add a new loss

1. Add class in `include/loss.hpp` inheriting `Loss`.
2. Implement `forward` and `backward` in `src/loss.cpp`.
3. Add or extend focused coverage in `tests/test_loss.cpp`.
4. Replace `BinaryCrossEntropy` usage in `src/main.cpp`.

## 6.3 Add a new optimizer

1. Add class in `include/optimizer.hpp`.
2. Implement update logic in `src/optimizer.cpp`.
3. Use it in the training loop.

### Available optimizers

See [Optimizers Included](../README.md#optimizers-included) in the README for the full list and CLI strings.

### Custom optimizer without editing core code

Use `LambdaOptimizer` with custom matrix/vector update rules:

```cpp
LambdaOptimizer optimizer(
    [](Matrix &param, const Matrix &grad, std::size_t) {
      const double lr = 0.1;
      for (std::size_t i = 0; i < rows(param); ++i) {
        for (std::size_t j = 0; j < cols(param); ++j) {
          param[i][j] -= lr * grad[i][j];
        }
      }
    },
    [](Vector &param, const Vector &grad, std::size_t) {
      const double lr = 0.1;
      for (std::size_t i = 0; i < param.size(); ++i) {
        param[i] -= lr * grad[i];
      }
    });
```

## 7. Metrics and Evaluation

The code reports:
- Loss: train, validation, test
- Binary metrics: accuracy, precision, recall, F1
- Confusion counts: TP, TN, FP, FN

Use `--threshold` to control classification cutoff for binary metrics.

## 8. Debugging Checklist

- Loss becomes `nan`:
  - Reduce learning rate.
  - Check for unstable operations (division/log domains).
- Loss does not decrease:
  - Verify backward equations.
  - Try smaller initialization or lower learning rate.
- Shape mismatch exceptions:
  - Validate layer input/output sizes.
  - Validate dataset dimensions.

## 9. Contribution Roadmap

The full phased roadmap with links to GitHub issues is in [`ROADMAP.md`](../ROADMAP.md) at the repository root.

## 10. Recommended Workflow for Changes

1. Create a branch and make a small focused change.
2. Build and run one baseline experiment.
3. Validate loss behavior and outputs.
4. Document any new module/API in `README.md` and this tutorial.

## 11. Programmatic Library Usage

The project now exposes `mlp_lib` and a public API in `include/mlp/library.hpp`.

Core entry point:
- `mlp::run_xor_experiment(const mlp::ExperimentOptions&, std::ostream* log_stream = nullptr, std::size_t log_every = 500)`

This lets you reuse training/evaluation logic in other C++ apps without relying on CLI execution.

Example app:
- `examples/library_usage.cpp`
- build/run with:
  - `cmake --build build --target mlp_library_example`
  - `./build/mlp_library_example`

Model persistence API:
- `include/mlp/io.hpp`
- functions: `mlp::save_sequential(...)` and `mlp::load_sequential(...)`

Roundtrip example:
- `examples/io_roundtrip.cpp`
- build/run with:
  - `cmake --build build --target mlp_io_example`
  - `./build/mlp_io_example`

## 12. Install and Consume from Another Project

For install commands and `find_package` usage, see [Install and find_package](../README.md#install-and-findpackage) in the README.

## 13. Automated Tests and CI

For test commands and CI configuration details, see [Testing and CI](../README.md#testing-and-ci) in the README.
That includes the BLAS benchmark gate whenever a BLAS backend is detected/enabled.
The local test suite also includes `mlp_test_activations` for ReLU, Sigmoid, and Tanh forward/backward checks, plus `mlp_test_loss` for BinaryCrossEntropy forward/backward, clamp, and shape-mismatch coverage.
