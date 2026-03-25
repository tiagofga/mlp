# Roadmap

This roadmap tracks planned work for the Modular MLP project, organised into phases. Each item links to its corresponding GitHub issue where acceptance criteria and discussion live.

Progress is tracked via [GitHub Issues](https://github.com/tiagofga/mlp/issues).

---

## Phase 1 — Testing & Reliability

Establish a solid test baseline before adding new features.

- [ ] Add unit tests for all activation functions — [#31](https://github.com/tiagofga/mlp/issues/31)
- [ ] Add unit tests for all loss functions — [#32](https://github.com/tiagofga/mlp/issues/32)
- [ ] Add gradient-check (numerical Jacobian) tests — [#33](https://github.com/tiagofga/mlp/issues/33)
- [ ] Add memory-leak detection step to CI — [#34](https://github.com/tiagofga/mlp/issues/34)

---

## Phase 2 — Core Training Features

Extend the training loop with the most commonly needed building blocks.

- [ ] Implement mini-batch training support — [#8](https://github.com/tiagofga/mlp/issues/8)
- [ ] Add early stopping callback — [#14](https://github.com/tiagofga/mlp/issues/14)
- [ ] Add learning-rate schedulers (step, cosine, exponential) — [#13](https://github.com/tiagofga/mlp/issues/13)
- [ ] Add gradient clipping (by norm and by value) — [#16](https://github.com/tiagofga/mlp/issues/16)
- [ ] Add model checkpointing (save best weights during training) — [#17](https://github.com/tiagofga/mlp/issues/17)
- [ ] Add L1/L2 weight regularisation to Dense layer — [#15](https://github.com/tiagofga/mlp/issues/15)
- [ ] Add dropout regularisation layer — [#11](https://github.com/tiagofga/mlp/issues/11)
- [ ] Add batch-normalisation layer — [#10](https://github.com/tiagofga/mlp/issues/10)

---

## Phase 3 — Performance & Optimisation

Speed up the CPU and GPU paths once the feature set is stable.

- [ ] Replace naive matrix loops with BLAS/CBLAS — [#4](https://github.com/tiagofga/mlp/issues/4)
- [ ] Vectorise activation functions with SIMD intrinsics — [#5](https://github.com/tiagofga/mlp/issues/5)
- [ ] Profile and reduce heap allocations in forward/backward pass — [#6](https://github.com/tiagofga/mlp/issues/6)
- [ ] Add a memory pool for Matrix/Vector allocations — [#7](https://github.com/tiagofga/mlp/issues/7)
- [ ] Parallelise backward pass with OpenMP task parallelism — [#9](https://github.com/tiagofga/mlp/issues/9)

---

## Phase 4 — Refactoring & Code Health

Keep the codebase maintainable as it grows.

- [ ] Unify optimizer parameter struct (`OptimizerConfig`) — [#20](https://github.com/tiagofga/mlp/issues/20)
- [ ] Split `model.cpp` into smaller translation units — [#21](https://github.com/tiagofga/mlp/issues/21)
- [ ] Replace raw `double` with a configurable `Scalar` typedef — [#22](https://github.com/tiagofga/mlp/issues/22)
- [ ] Extract Matrix operations into a dedicated `MatrixOps` namespace — [#23](https://github.com/tiagofga/mlp/issues/23)
- [ ] Use `std::span` for gradient and parameter slices — [#24](https://github.com/tiagofga/mlp/issues/24)
- [ ] Replace manual element loops in `matrix.hpp` with STL algorithms — [#25](https://github.com/tiagofga/mlp/issues/25)

---

## Phase 5 — Advanced Features & Extensibility

Expand the layer and loss catalogue and improve cross-language usability.

- [ ] Add softmax activation and cross-entropy loss for multi-class problems — [#18](https://github.com/tiagofga/mlp/issues/18)
- [ ] Add Layer Normalisation — [#19](https://github.com/tiagofga/mlp/issues/19)
- [ ] Add Conv2D layer — [#38](https://github.com/tiagofga/mlp/issues/38)
- [ ] Expose Python bindings via pybind11 — [#12](https://github.com/tiagofga/mlp/issues/12)

---

## Phase 6 — Documentation & CI

Raise the quality bar for documentation and continuous integration.

- [ ] Add architecture diagram to README — [#28](https://github.com/tiagofga/mlp/issues/28)
- [ ] Document CUDA path limitations and roadmap — [#29](https://github.com/tiagofga/mlp/issues/29)
- [ ] Add benchmark comparison table (SGD vs Adam vs others) — [#30](https://github.com/tiagofga/mlp/issues/30)
- [ ] Add Doxygen/API reference generation to CMake — [#26](https://github.com/tiagofga/mlp/issues/26)
- [ ] Write contributing guide (`CONTRIBUTING.md`) — [#27](https://github.com/tiagofga/mlp/issues/27)
- [ ] Add code coverage reporting (gcov/lcov) — [#35](https://github.com/tiagofga/mlp/issues/35)
- [ ] Add sanitizer builds (AddressSanitizer, UBSanitizer) — [#36](https://github.com/tiagofga/mlp/issues/36)
- [ ] Add Windows and macOS build matrices to CI — [#37](https://github.com/tiagofga/mlp/issues/37)

---

## Notes

- Phases are roughly ordered by dependency — later phases benefit from earlier ones being done first, but individual items can be picked up out of order.
- New ideas should be filed as GitHub Issues first and linked here once triaged.
- The CUDA roadmap item ([#29](https://github.com/tiagofga/mlp/issues/29)) will be updated as Phase 3 work progresses.
