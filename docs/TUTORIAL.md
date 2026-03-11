# Tutorial: Running and Extending the Modular MLP

This guide shows how to run the current MLP, edit experiments, and contribute new modules.

## 1. Prerequisites

- CMake >= 3.16
- C++17 compiler (`g++` or `clang++`)
- Optional for GPU: CUDA toolkit (`nvcc`)
- Optional for CPU threading: OpenMP runtime/toolchain support

## 2. Build and Run (CPU)

From repository root:

```bash
cmake -S . -B build
cmake --build build
./build/mlp
```

Select optimizer at runtime:

```bash
./build/mlp --optimizer sgd
./build/mlp --optimizer momentum
./build/mlp --optimizer adam
./build/mlp --optimizer adamw
```

Select hidden-layer layout at runtime:

```bash
./build/mlp --hidden 8
./build/mlp --hidden 16,16
./build/mlp --optimizer adam --hidden 32,16,8
```

Set training/split options at runtime:

```bash
./build/mlp --epochs 3000 --lr 0.01 --samples 1000
./build/mlp --train-ratio 0.7 --val-ratio 0.15 --threshold 0.5
```

Expected behavior:
- You will see train and validation losses during training.
- At the end, the program reports train/validation/test losses and binary metrics.

## 3. Optional Backends

### 3.1 OpenMP (CPU parallel)

```bash
cmake -S . -B build-omp -DMLP_ENABLE_OPENMP=ON
cmake --build build-omp
./build-omp/mlp
```

### 3.2 CUDA (GPU for dense ops)

```bash
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON
cmake --build build-cuda
./build-cuda/mlp
```

If CUDA is not found, configure with:

```bash
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=/path/to/cuda
```

### 3.3 OpenMP + CUDA

```bash
cmake -S . -B build-hybrid -DMLP_ENABLE_OPENMP=ON -DMLP_ENABLE_CUDA=ON
cmake --build build-hybrid
./build-hybrid/mlp
```

Notes:
- OpenMP affects CPU code paths.
- CUDA path currently prioritizes modularity/correctness over peak performance.

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
3. Use the new activation in `src/main.cpp`.

## 6.2 Add a new loss

1. Add class in `include/loss.hpp` inheriting `Loss`.
2. Implement `forward` and `backward` in `src/loss.cpp`.
3. Replace `BinaryCrossEntropy` usage in `src/main.cpp`.

## 6.3 Add a new optimizer

1. Add class in `include/optimizer.hpp`.
2. Implement update logic in `src/optimizer.cpp`.
3. Use it in the training loop.

### Available optimizers right now

- `SGD`
- `Momentum`
- `Adam`
- `AdamW`
- `LambdaOptimizer` for custom update rules

Runtime selection:

```bash
./build/mlp --optimizer sgd
./build/mlp --optimizer momentum
./build/mlp --optimizer adam
./build/mlp --optimizer adamw
```

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

## 9. Suggested Contribution Roadmap

- Mini-batch training
- Validation/test split
- More metrics and logging
- `Adam` and momentum optimizers
- Regularization (`L2`, dropout)
- Config-driven experiments (JSON/YAML)

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

Install package files:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix /tmp/mlp-install
```

Consumer `CMakeLists.txt`:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_lib)
```

Component-based linking:

```cmake
find_package(mlp REQUIRED)
target_link_libraries(your_app PRIVATE mlp::mlp_train mlp::mlp_io)
```

Read installed library version in code:

```cpp
#include "mlp/version.hpp"
// MLP_VERSION_STRING, MLP_VERSION_MAJOR, MLP_VERSION_MINOR, MLP_VERSION_PATCH
```

Configure consumer:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/mlp-install
```

## 13. Automated Tests and CI

Run tests locally:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Test coverage currently includes:
- end-to-end training test
- IO roundtrip (save/load) test
- package consumer test using installed artifacts and `find_package(mlp)`

CI workflow:
- `.github/workflows/ci.yml`
- runs matrix configure/build/test for `MLP_ENABLE_OPENMP=OFF/ON`
- runs optional CUDA configure/build smoke check when `nvcc` is available on the runner
