# Roadmap

Last synced with GitHub Issues: 2026-09-14

This roadmap tracks planned work for the Modular MLP project. Each item links to its GitHub issue, where acceptance criteria and discussion live.

Progress is tracked via [GitHub Issues](https://github.com/tiagofga/mlp/issues).

Current issue state from GitHub:

- Closed: [#4](https://github.com/tiagofga/mlp/issues/4), [#28](https://github.com/tiagofga/mlp/issues/28), [#30](https://github.com/tiagofga/mlp/issues/30), [#31](https://github.com/tiagofga/mlp/issues/31), [#32](https://github.com/tiagofga/mlp/issues/32), [#33](https://github.com/tiagofga/mlp/issues/33), [#34](https://github.com/tiagofga/mlp/issues/34)
- Implemented but still open: [#27](https://github.com/tiagofga/mlp/issues/27)
- New strategic gaps: [#55](https://github.com/tiagofga/mlp/issues/55)-[#59](https://github.com/tiagofga/mlp/issues/59)

---

## Phase 1 — Testing & Reliability

Establish a solid test baseline before adding new features.

- [x] Add unit tests for all activation functions — [#31](https://github.com/tiagofga/mlp/issues/31) *(Closed; ReLU, Sigmoid, and Tanh forward/backward coverage in `mlp_test_activations`.)*
- [x] Add unit tests for all loss functions — [#32](https://github.com/tiagofga/mlp/issues/32) *(Closed; BinaryCrossEntropy forward/backward, clamp, and shape-mismatch coverage in `mlp_test_loss`.)*
- [x] Add gradient-check numerical Jacobian tests — [#33](https://github.com/tiagofga/mlp/issues/33) *(Closed; reusable finite-difference helper in `tests/gradient_check.hpp` with Dense and Tanh coverage.)*
- [x] Add memory-leak detection step to CI — [#34](https://github.com/tiagofga/mlp/issues/34) *(Closed; Ubuntu CI runs Valgrind memcheck through CTest.)*

---

## Phase 2 — Core Training Features

Extend the training loop with commonly needed training controls. This phase is the highest-value path toward a practical v1.0 training API.

- [ ] Implement mini-batch training support — [#8](https://github.com/tiagofga/mlp/issues/8)
- [ ] Add configurable weight initialisation strategies — [#55](https://github.com/tiagofga/mlp/issues/55)
- [ ] Add early stopping callback — [#14](https://github.com/tiagofga/mlp/issues/14)
- [ ] Add model checkpointing for best validation weights — [#17](https://github.com/tiagofga/mlp/issues/17)
- [ ] Add gradient clipping by norm and by value — [#16](https://github.com/tiagofga/mlp/issues/16)
- [ ] Add learning-rate schedulers: step, cosine, exponential — [#13](https://github.com/tiagofga/mlp/issues/13)
- [ ] Add L1/L2 weight regularisation to Dense layer — [#15](https://github.com/tiagofga/mlp/issues/15)
- [ ] Add dropout regularisation layer — [#11](https://github.com/tiagofga/mlp/issues/11)
- [ ] Add batch-normalisation layer — [#10](https://github.com/tiagofga/mlp/issues/10)

---

## Phase 3 — Performance & Optimisation

Speed up the CPU and GPU paths once the feature set is stable.

- [x] Replace naive matrix loops with BLAS/CBLAS — [#4](https://github.com/tiagofga/mlp/issues/4) *(Closed; optional BLAS/CBLAS path, numerical checks, and benchmark gate implemented.)*
- [ ] Vectorise activation functions with SIMD intrinsics — [#5](https://github.com/tiagofga/mlp/issues/5)
- [ ] Profile and reduce heap allocations in forward/backward pass — [#6](https://github.com/tiagofga/mlp/issues/6)
- [ ] Add a memory pool for Matrix/Vector allocations — [#7](https://github.com/tiagofga/mlp/issues/7)
- [ ] Parallelize backward pass with OpenMP task parallelism — [#9](https://github.com/tiagofga/mlp/issues/9)

---

## Phase 4 — Refactoring & Code Health

Keep the codebase maintainable as it grows.

- [ ] Unify optimizer parameter struct (`OptimizerConfig`) — [#20](https://github.com/tiagofga/mlp/issues/20)
- [ ] Split `model.cpp` into smaller translation units — [#21](https://github.com/tiagofga/mlp/issues/21)
- [ ] Replace raw `double` with a configurable `Scalar` typedef — [#22](https://github.com/tiagofga/mlp/issues/22)
- [ ] Extract Matrix operations into a dedicated `MatrixOps` namespace — [#23](https://github.com/tiagofga/mlp/issues/23)
- [ ] Use `std::span` for gradient and parameter slices — [#24](https://github.com/tiagofga/mlp/issues/24)
- [ ] Replace manual element loops in `matrix.hpp` with STL algorithms — [#25](https://github.com/tiagofga/mlp/issues/25)

Recent non-issue-backed cleanup:

- Optimizer loops now share parameter traversal/state helpers in `src/optimizer.cpp`.
- Repeated test comparison logic now lives in `tests/test_helpers.hpp`.

---

## Phase 5 — Multiclass & Extensibility

Move the project beyond binary XOR-style demonstrations and into general small-scale classification experiments.

- [ ] Add softmax activation and cross-entropy loss for multi-class problems — [#18](https://github.com/tiagofga/mlp/issues/18)
- [ ] Add numerically stable cross-entropy from logits — [#59](https://github.com/tiagofga/mlp/issues/59) *(depends on #18)*
- [ ] Add multiclass evaluation metrics and confusion matrix — [#56](https://github.com/tiagofga/mlp/issues/56) *(depends on #18)*
- [ ] Add Layer Normalisation — [#19](https://github.com/tiagofga/mlp/issues/19)
- [ ] Expose Python bindings via pybind11 — [#12](https://github.com/tiagofga/mlp/issues/12)
- [ ] Add Conv2D layer — [#38](https://github.com/tiagofga/mlp/issues/38)

---

## Phase 6 — Validation, Documentation & CI

Raise the quality bar for correctness evidence, documentation, portability, and continuous integration.

- [ ] Add numerical parity checks against a reference implementation — [#57](https://github.com/tiagofga/mlp/issues/57)
- [ ] Add end-to-end backend benchmark suite — [#58](https://github.com/tiagofga/mlp/issues/58)

- [x] Add architecture diagram to README — [#28](https://github.com/tiagofga/mlp/issues/28) *(Implemented; README now includes an in-repository ASCII architecture diagram covering Dense, Activation, Loss, Optimizer, and forward/backward data flow.)*
- [ ] Document CUDA path limitations and roadmap — [#29](https://github.com/tiagofga/mlp/issues/29)
- [ ] Add benchmark comparison table for optimizers — [#30](https://github.com/tiagofga/mlp/issues/30)
- [ ] Add Doxygen/API reference generation to CMake — [#26](https://github.com/tiagofga/mlp/issues/26)
- [x] Write contributing guide (`CONTRIBUTING.md`) — [#27](https://github.com/tiagofga/mlp/issues/27) *(Implemented; `CONTRIBUTING.md` covers branching strategy, Conventional Commits, local build/test instructions, and review checklists, and is linked from the README. The GitHub issue is still open and can be closed.)*
- [ ] Add code coverage reporting with gcov/lcov — [#35](https://github.com/tiagofga/mlp/issues/35)
- [ ] Add sanitizer builds: AddressSanitizer and UBSanitizer — [#36](https://github.com/tiagofga/mlp/issues/36)
- [ ] Add Windows and macOS build matrices to CI — [#37](https://github.com/tiagofga/mlp/issues/37)

---

## Suggested v1.0 critical path

A compact path to a substantially more capable release is:

1. [#8](https://github.com/tiagofga/mlp/issues/8) mini-batch training.
2. [#55](https://github.com/tiagofga/mlp/issues/55) weight initialisation.
3. [#18](https://github.com/tiagofga/mlp/issues/18) Softmax + cross-entropy.
4. [#59](https://github.com/tiagofga/mlp/issues/59) stable logits cross-entropy.
5. [#56](https://github.com/tiagofga/mlp/issues/56) multiclass metrics.
6. [#14](https://github.com/tiagofga/mlp/issues/14), [#17](https://github.com/tiagofga/mlp/issues/17), and [#16](https://github.com/tiagofga/mlp/issues/16) training controls.
7. [#57](https://github.com/tiagofga/mlp/issues/57) reference parity validation.
8. [#35](https://github.com/tiagofga/mlp/issues/35), [#36](https://github.com/tiagofga/mlp/issues/36), and [#37](https://github.com/tiagofga/mlp/issues/37) CI maturity.

Conv2D and Python bindings remain valuable, but are not prerequisites for a coherent MLP-focused v1.0.

## Notes

- Phases are roughly ordered by dependency, but individual items can be picked up out of order.
- New ideas should be filed as GitHub Issues first and linked here once triaged.
- Keep this file synced with GitHub issue state when issues are opened, closed, or renamed.
- Prefer closing a smaller set of coherent v1.0 capabilities before expanding into broader framework scope.
