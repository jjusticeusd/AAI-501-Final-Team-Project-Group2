#!/bin/bash
#
# start_jupyter.sh — launch Jupyter Lab for this project.
#
# Assumes you've already run ./init.sh once (which installs uv, the deps,
# and the project kernel). If you haven't, run that first.

set -euo pipefail

# Move to the repo root so Jupyter opens with this project as its root folder.
cd "$(dirname "$0")"

# 'uv run --with jupyter' launches Jupyter Lab using the project's .venv,
# pulling in jupyter on the fly without adding it as a project dependency.
# Inside the notebook, select the "AAI-501 (Group 2)" kernel.
uv run --with jupyter jupyter lab
