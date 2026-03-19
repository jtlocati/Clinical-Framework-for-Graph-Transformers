# YAIB setup notes

These notes document the working local YAIB setup used to get the CLI running.

## Folder structure

Keep the DDJA repo and YAIB as separate sibling folders inside the same parent directory.

Example:

- `DDJA/`
  - `team-repo/`
  - `YAIB/`

Do not place `YAIB` inside the team repo.

## Python version

Use Python 3.10.

Python 3.13 caused setup and dependency issues.
On macOS with Homebrew, Python 3.10 was installed with:

`brew install python@3.10`

Verified with:

`/opt/homebrew/bin/python3.10 --version`

## Virtual environment setup

From inside the `YAIB` folder:

`rm -rf venv`

`/opt/homebrew/bin/python3.10 -m venv venv`

`source venv/bin/activate`

`python --version`

`python -m pip install --upgrade pip setuptools wheel`

## Install YAIB

From inside the `YAIB` folder with the venv activated:

`python -m pip install -e .`

## Missing packages that had to be installed manually

The repo setup did not install all runtime dependencies automatically.

These commands were needed:

`python -m pip install gin-config`

`python -m pip install torch`

`python -m pip install wandb`

`python -m pip install matplotlib`

`python -m pip install scikit-optimize`

`python -m pip install optuna`

`python -m pip install "optuna-integration[wandb]"`

`python -m pip install pytorch-ignite`

`python -m pip install recipies`

`python -m pip install imbalanced-learn`

## Local YAIB code fixes

In `YAIB/icu_benchmarks/data/preprocessor.py`, the imports used `recipys`, which did not match the installed package.

These imports were changed from `recipys` to `recipies`:

- `from recipys.recipe import Recipe`
- `from recipys.selector import all_numeric_predictors, all_of, all_outcomes, has_type`
- `from recipys.step import (`

Changed to:

- `from recipies.recipe import Recipe`
- `from recipies.selector import all_numeric_predictors, all_of, all_outcomes, has_type`
- `from recipies.step import (`

## Verification

The setup was considered working when this command succeeded:

`icu-benchmarks --help`

## Notes

- YAIB dependency/setup metadata appeared incomplete.
- The environment worked after using Python 3.10, manually installing missing packages, and fixing the stale `recipys` imports.
- These changes were made in the local YAIB clone, not in the DDJA repo itself.