#!/bin/bash
# Check PEP 8 style and autofix where possible.
# Usage: ./lint.sh        — fix + format
#        ./lint.sh --check — report only, no changes (useful in CI)

set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--check" ]; then
  uv run ruff check code/
  uv run ruff format --check code/
else
  uv run ruff check --fix code/
  uv run ruff format code/
fi
