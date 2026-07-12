#!/bin/bash
# Strip outputs and execution counts from all notebooks in code/.
# Run before committing so diffs stay clean.

set -euo pipefail
cd "$(dirname "$0")"

notebooks=$(find code -name "*.ipynb" 2>/dev/null)

if [ -z "${notebooks}" ]; then
  echo "No notebooks found in code/."
  exit 0
fi

echo "${notebooks}" | while IFS= read -r nb; do
  echo "Stripping: ${nb}"
  uv run nbstripout "${nb}"
done

echo "Done."
