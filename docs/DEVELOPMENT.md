# Development Setup

How to get this project running on your machine and work in the shared
Jupyter notebook. Written for **macOS**; steps are similar on Linux.

We use [**uv**](https://docs.astral.sh/uv/) to manage Python and dependencies.
uv reads `pyproject.toml` + `uv.lock` and builds an identical environment for
everyone, so "works on my machine" problems mostly go away.

---

## 1. First-time setup (run once after cloning)

From the repo root:

```bash
./init.sh
```

That script:

1. Installs **uv** if you don't already have it.
2. Runs `uv sync` — installs Python 3.14 (pinned in `.python-version`) and every
   dependency at the exact locked version, into a local `.venv/`.
3. Registers a Jupyter kernel named **"AAI-501 (Group 2)"** pointing at that env.

If uv was just installed, open a new terminal afterward (or run
`source ~/.zshrc`) so the `uv` command is permanently on your `PATH`.

### Installing uv manually (if you prefer)

`init.sh` does this for you, but for reference:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Start Jupyter Lab

```bash
./start_jupyter.sh
```

This launches Jupyter Lab in your browser with the repo as the root folder.
**Inside a notebook, choose the "AAI-501 (Group 2)" kernel** (kernel picker,
top-right) so your code runs against the project's pinned dependencies.

---

## 3. Everyday uv commands

| Task | Command |
| --- | --- |
| Sync your env after someone changes dependencies | `uv sync` |
| Add a new dependency | `uv add <package>` |
| Add a dev-only dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Run a one-off script in the env | `uv run python your_script.py` |

When you `uv add`/`uv remove`, uv updates both `pyproject.toml` and `uv.lock`.
**Commit both files** so teammates get the same versions with a plain `uv sync`.

---

## 4. Working on the shared notebook

A few habits keep the notebook mergeable when several people touch it:

- **Sync before you start:** `git pull` and then `uv sync` in case deps changed.
- **Divide the work by cells/sections** so two people rarely edit the same cell.
- **Run "Restart Kernel and Run All"** before committing so execution counts and
  outputs are consistent.
- **Strip outputs before committing** with `./strip_notebooks.sh` so diffs stay
  small and merge conflicts are rare.
- **Commit small and often**, and push so others can pull your changes early.
- **Pick the "AAI-501 (Group 2)" kernel** — not a random global kernel — so the
  notebook's saved kernel metadata stays consistent across the team.

---

## 5. Code style (PEP 8)

The course requires following [PEP 8](https://pep8.org/). We use
[**ruff**](https://docs.astral.sh/ruff/) to enforce it (a fast, single-tool
replacement for flake8 + black). It's configured to a 79-character line limit
and to also lint the notebooks in `code/`.

```bash
./lint.sh          # autofix style issues and reformat, in place
./lint.sh --check  # report only, no changes (exits non-zero if issues remain)
```

Run `./lint.sh` before committing. It fixes most issues automatically. A few it
can only flag — most commonly **long string literals**, which can't be broken
automatically because that would change the value. Split those by hand:

```python
message = (
    "This is a really long string that goes over the "
    "79-character limit, so we split it across lines."
)
```

Python joins adjacent string literals at compile time, so the result is
identical at runtime.

---

## Troubleshooting

- **`uv: command not found`** — open a new terminal, or run
  `source ~/.zshrc`, so the newly installed uv is on your `PATH`.
- **Wrong/old packages in a notebook** — confirm the kernel is
  "AAI-501 (Group 2)", then run `uv sync`.
- **Kernel missing from the picker** — re-run `./init.sh`.
