# API Compatibility Policy

This project follows semantic versioning for the installable C++ library package.

## Versioning

- Format: `MAJOR.MINOR.PATCH`
- Current package version is defined in CMake `project(... VERSION ...)`.
- Public version macros are available in `mlp/version.hpp`:
  - `MLP_VERSION_MAJOR`
  - `MLP_VERSION_MINOR`
  - `MLP_VERSION_PATCH`
  - `MLP_VERSION_STRING`

## Public Stable API

These headers are considered public API for library consumers:
- `mlp/types.hpp`
- `mlp/library.hpp`
- `mlp/io.hpp`
- `mlp/metrics.hpp`
- `mlp/version.hpp`

The CMake imported target `mlp::mlp_lib` is also public API.
Component targets are public API as well:
- `mlp::mlp_core`
- `mlp::mlp_optim`
- `mlp::mlp_train`
- `mlp::mlp_io`

## Compatibility Rules

- `PATCH`: bug fixes only, no breaking API changes.
- `MINOR`: backward-compatible feature additions.
- `MAJOR`: breaking changes are allowed.

## Internal/Unstable API

The following are currently internal implementation details and may change between minor versions:
- `layer.hpp`, `model.hpp`, `optimizer.hpp`, `dense.hpp`, `activations.hpp`, `matrix.hpp`, `loss.hpp`, `cuda_ops.hpp`
- files under `src/`

Consumers should avoid depending on internal headers directly if long-term compatibility matters.
