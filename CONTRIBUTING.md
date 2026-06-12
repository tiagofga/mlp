# Contributing

Thanks for helping improve Modular MLP. This project is a compact C++17 neural-network lab, so the best contributions are focused, reproducible, and easy to inspect.

## Development Setup

Requirements for a clean Ubuntu 22.04-style environment:

- CMake 3.16 or newer
- A C++17 compiler, such as `g++` or `clang++`
- Git
- Optional: OpenMP runtime/toolchain support
- Optional: BLAS/OpenBLAS for BLAS backend work
- Optional: CUDA toolkit for CUDA backend work
- Optional: Valgrind for memory-check work

Default build and test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Full local pre-push check:

```bash
./scripts/pre_push_check.sh
```

## Branching Strategy

Use a short branch name that includes the type of work and, when possible, the GitHub issue number.

Examples:

```text
docs/27-contributing-guide
feat/8-mini-batch-training
fix/dense-shape-check
refactor/optimizer-config
```

Keep each branch focused on one issue or one coherent change. If you discover unrelated cleanup while working, open a separate issue or branch for it.

## Commit Messages

Use Conventional Commits:

```text
type(scope): short summary
```

Common types:

- `feat`: user-facing feature
- `fix`: bug fix
- `docs`: documentation-only change
- `test`: test-only change
- `refactor`: behavior-preserving code cleanup
- `perf`: performance improvement
- `ci`: CI workflow change
- `build`: CMake or build tooling change

Examples:

```text
docs: add contributing guide
test: add dense gradient regression coverage
perf(dense): use blas path for transposed matmul
ci: add sanitizer test job
```

If the commit or pull request completes an issue, reference it with GitHub closing keywords in the PR description, for example:

```text
Closes #27
```

## Code Guidelines

- Keep code C++17-compatible.
- Prefer clear, inspectable implementations over broad dependencies.
- Follow the current CMake target layout.
- Respect the public API boundary in `include/mlp/`.
- Treat root-level headers in `include/` and files in `src/` as internal unless documented otherwise.
- Reuse existing helpers before adding new abstractions.
- Avoid touching generated build directories such as `build/` or `build-*`.

Public API changes should update:

- `README.md`
- `docs/TUTORIAL.md`
- `docs/API_POLICY.md`
- relevant examples or tests

## Test Guidelines

Add focused tests for new behavior. The current test suite includes:

- activation forward/backward tests
- loss forward/backward tests
- finite-difference gradient checks
- training/evaluation integration tests
- save/load roundtrip tests
- package-consumer tests
- BLAS benchmark gate when BLAS is enabled

Use `tests/test_helpers.hpp` for matrix/vector comparisons instead of duplicating assertion helpers.

For source changes, run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

For backend-specific changes, also run the relevant backend build:

```bash
cmake -S . -B build-omp -DMLP_ENABLE_OPENMP=ON
cmake --build build-omp

cmake -S . -B build-blas -DMLP_ENABLE_BLAS=ON
cmake --build build-blas
./build-blas/mlp_dense_benchmark --min-speedup 2.0
```

CUDA work requires a local CUDA toolkit:

```bash
cmake -S . -B build-cuda -DMLP_ENABLE_CUDA=ON
cmake --build build-cuda
```

## Documentation Guidelines

Documentation changes should stay consistent across:

- `README.md`
- `ROADMAP.md`
- `docs/TUTORIAL.md`
- `docs/EXPERIMENTS.md`
- `docs/API_POLICY.md`
- `docs/site/`

If you change roadmap status, verify the current GitHub issue state first.

## Pull Request Checklist

Before opening a pull request:

- The branch has one clear purpose.
- The PR description explains what changed and why.
- Relevant GitHub issues are linked.
- Public API changes are documented.
- New behavior has tests.
- `cmake --build build` passes.
- `ctest --test-dir build --output-on-failure` passes.
- Backend-specific tests were run when relevant.
- Generated files and local build outputs are not included.

## Code Review Checklist

Reviewers should check:

- Correctness of forward/backward math and optimizer updates.
- Shape validation and error messages.
- Public API compatibility.
- Test coverage for new or changed behavior.
- CMake target/linkage consistency.
- Documentation accuracy.
- No accidental changes to generated build outputs.
