#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Git branch =="
git rev-parse --abbrev-ref HEAD

printf "\n== Git status (short) ==\n"
git status --short

printf "\n== Configure ==\n"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

printf "\n== Build ==\n"
cmake --build build --config Release --parallel

printf "\n== Test ==\n"
ctest --test-dir build --output-on-failure

printf "\n== Ready to commit/push ==\n"
echo "All checks passed."
