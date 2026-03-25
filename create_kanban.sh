#!/usr/bin/env bash
# create_kanban.sh — create a GitHub Project (Kanban) and populate it with issues.
#
# Usage:
#   ./create_kanban.sh [OWNER]
#
# OWNER defaults to the authenticated user (@me). Pass an organisation name when
# the project should live under an org, e.g.: ./create_kanban.sh tiagofga
#
# Requirements:
#   gh >= 2.31  (gh project commands + gh issue create)
#   Scopes needed: project, repo

set -euo pipefail

OWNER="${1:-@me}"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

# ---------------------------------------------------------------------------
# Helper: create an issue and add it to the project.
# Usage: add_issue <project_number> <title> <body>
# ---------------------------------------------------------------------------
add_issue() {
    local project_number="$1"
    local title="$2"
    local body="$3"

    local issue_url
    issue_url=$(gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body  "$body")

    gh project item-add "$project_number" \
        --owner "$OWNER" \
        --url   "$issue_url" 2>/dev/null && \
        echo "  ✓ $title" || \
        echo "  ✗ added issue but could not attach to project: $issue_url"
}

# ---------------------------------------------------------------------------
# 1. Create the project
# ---------------------------------------------------------------------------
PROJECT_URL=$(gh project create \
    --owner "$OWNER" \
    --title "MLP Roadmap" \
    --format json | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url',''))")

# Extract the numeric ID from the URL (last path segment)
PROJECT_NUMBER="${PROJECT_URL##*/}"

if [[ -z "$PROJECT_NUMBER" || ! "$PROJECT_NUMBER" =~ ^[0-9]+$ ]]; then
    # Fallback: list projects and find by title
    PROJECT_NUMBER=$(gh project list \
        --owner "$OWNER" \
        --format json | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
projects = data.get('projects', [])
for p in projects:
    if p.get('title') == 'MLP Roadmap':
        print(p['number'])
        break
")
fi

echo "Created project #${PROJECT_NUMBER} — ${PROJECT_URL}"

# ---------------------------------------------------------------------------
# 2. Populate issues by category
# ---------------------------------------------------------------------------

echo ""
echo "=== OPTIMIZATIONS ==="
add_issue "$PROJECT_NUMBER" \
    "opt: Replace naive matrix loops with BLAS/CBLAS" \
    "Replace the hand-written matrix-multiply and activation loops in \`src/dense.cpp\` with BLAS/CBLAS calls (e.g. \`cblas_sgemm\`) to improve throughput on multi-core CPUs.

## Acceptance criteria
1. Linking against OpenBLAS or Apple Accelerate is optional and detected at CMake configure time.
2. CPU-backend results are numerically equivalent to the current implementation within 1 e-5 tolerance.
3. A microbenchmark (forward + backward on a 512×512 weight matrix) shows ≥ 2× speedup on a machine with BLAS available."

add_issue "$PROJECT_NUMBER" \
    "opt: Vectorise activation functions with SIMD intrinsics" \
    "Use SSE4/AVX2 intrinsics (or compiler auto-vectorisation hints) to speed up element-wise activation functions (ReLU, sigmoid, tanh) in \`src/activations.cpp\`.

## Acceptance criteria
1. Falls back gracefully to scalar code when the target does not support the required ISA extension.
2. Results match scalar path within floating-point tolerance.
3. Benchmark shows measurable throughput improvement on a supported machine."

add_issue "$PROJECT_NUMBER" \
    "opt: Profile and reduce heap allocations in forward/backward pass" \
    "Use a profiler (e.g. \`perf\`, Valgrind/Massif, or Instruments) to identify hot allocation paths during training and replace them with stack buffers or pre-allocated workspace vectors.

## Acceptance criteria
1. Allocation count per training step is documented before and after.
2. No regression in test suite.
3. Throughput (samples/s) improves by at least 10 % on the XOR benchmark."

echo ""
echo "=== FEATURES ==="
add_issue "$PROJECT_NUMBER" \
    "feat: Add batch-normalisation layer" \
    "Implement a \`BatchNorm\` layer that can be inserted between \`Dense\` layers. Must support both training (running-stats update) and inference modes, and be serialisable via the existing \`save/load_sequential\` API.

## Acceptance criteria
1. \`BatchNorm\` layer passes unit tests for forward and backward pass.
2. Save/load round-trip preserves gamma, beta, and running statistics.
3. The existing XOR integration test still passes."

add_issue "$PROJECT_NUMBER" \
    "feat: Add dropout regularisation layer" \
    "Implement a \`Dropout\` layer with configurable keep-probability. Must be inactive during inference and reproducible given a fixed seed.

## Acceptance criteria
1. Unit test verifies that exactly \`p\` fraction of activations are zeroed (within statistical tolerance) during training.
2. Dropout is disabled when \`model.eval()\` (or equivalent flag) is set.
3. Serialisation preserves the keep-probability."

add_issue "$PROJECT_NUMBER" \
    "feat: Expose Python bindings via pybind11" \
    "Create a thin \`pybind11\` wrapper (optional CMake target \`mlp_python\`) so that the library can be called from Python without \`ctypes\`.

## Acceptance criteria
1. \`pip install .\` builds the extension wheel.
2. The XOR experiment can be run end-to-end from Python.
3. Documentation updated with a Python usage example."

echo ""
echo "=== REFACTORS ==="
add_issue "$PROJECT_NUMBER" \
    "refactor: Unify optimizer parameter struct" \
    "Replace the per-optimizer scattered parameter arguments with a single \`OptimizerConfig\` struct that is validated at construction time, reducing the risk of silent misconfiguration.

## Acceptance criteria
1. All existing optimizer constructors accept the new struct.
2. Old positional constructors are deprecated with a clear compile-time message.
3. Tests updated; no new test failures."

add_issue "$PROJECT_NUMBER" \
    "refactor: Split model.cpp into smaller translation units" \
    "\`src/model.cpp\` is growing too large. Split it into \`model_forward.cpp\`, \`model_backward.cpp\`, and \`model_io.cpp\` to improve parallel compilation and readability.

## Acceptance criteria
1. \`cmake --build build -j4\` with a clean cache finishes without errors.
2. All existing tests pass.
3. Public header surface unchanged."

echo ""
echo "=== DOCUMENTATION ==="
add_issue "$PROJECT_NUMBER" \
    "docs: Add Doxygen/API reference generation to CMake" \
    "Add a CMake target (\`docs\`) that runs Doxygen over the public headers and outputs HTML to \`build/docs/html\`.  Add a CI step that uploads the output as a workflow artifact.

## Acceptance criteria
1. \`cmake --build build --target docs\` succeeds when Doxygen ≥ 1.9 is installed.
2. All public API symbols in \`include/mlp/\` have non-empty \`@brief\` entries.
3. CI uploads the artifact on every push to \`main\`."

add_issue "$PROJECT_NUMBER" \
    "docs: Write contributing guide (CONTRIBUTING.md)" \
    "Add a \`CONTRIBUTING.md\` that covers: branching strategy, commit message conventions, how to run tests locally, and the code-review checklist.

## Acceptance criteria
1. File present at repo root.
2. All steps in the guide can be followed on a clean Ubuntu 22.04 machine.
3. Link added to README."

echo ""
echo "Done. Visit your project at: ${PROJECT_URL}"
