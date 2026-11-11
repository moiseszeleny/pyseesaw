# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pyseesaw`: research/teaching code for neutrino mass generation via Seesaw (Type I, II, III) and Inverse Seesaw mechanisms, combining SymPy (symbolic derivations) with NumPy/SciPy (numerical diagonalization).

## Commands

A virtualenv lives in `.venv/` (Python 3.12, package installed editable). pytest is **not** installed there by default.

```bash
.venv/bin/pip install -r requirements.txt          # runtime deps
.venv/bin/pip install -e ".[dev]"                  # pytest, pytest-cov, black, flake8, mypy

.venv/bin/python -m pytest tests/ -v               # pyproject addopts add --cov=src (needs pytest-cov)
.venv/bin/python -m pytest tests/test_matrix_utils.py::TestMatrixUtils::<test_name> -v
.venv/bin/python -m pytest tests/ -o addopts=""    # run without the coverage addopts
.venv/bin/python tests/test_matrix_utils.py        # tests are unittest-based, also runnable directly

.venv/bin/python examples/seesaw_vs_inverse_comparison.py
black --line-length 88 src/
```

mypy is configured strictly in `pyproject.toml` (`disallow_untyped_defs`, etc.).

## Import conventions (important)

- `src/` is a package whose modules use **relative imports** (`seesaw.py` and `inverse_seesaw.py` do `from .matrix_utils import ...`). Import them as `src.seesaw`, `src.inverse_seesaw`, etc., with the repo root on `sys.path`. Examples and notebooks do this via `sys.path.insert(0, project_root)` / `sys.path.append(<repo root>)`.
- Exception: `tests/test_matrix_utils.py` adds `src/` itself to `sys.path` and imports `matrix_utils` directly; this only works because `matrix_utils` has no relative imports. Tests for `seesaw`/`inverse_seesaw` must use the `src.` form instead.
- `src/__init__.py` star-imports `matrix_utils`, `seesaw`, `inverse_seesaw`, `pmns_matrix` (not `visualization`). Its docstring mentions a `mass_ordering` module that does not exist.

## Architecture

- `matrix_utils.py` — foundation for everything else: numerical diagonalization (`diagonalize_hermitian_matrix` via `scipy.linalg.eigh`; `diagonalize_mass_matrix`, which for `symmetric=True` uses `eigh` and multiplies eigenvector columns by `i` for negative eigenvalues, otherwise SVD), unitarity/ordering checks, plus symbolic helpers (`create_symbolic_matrix`, `seesaw_approximation_symbolic`, `inverse_seesaw_symbolic`, `texture_zeros_matrix`, `substitute_numerical_values`, `symbolic_rotation_matrix`).
- `seesaw.py` / `inverse_seesaw.py` — each mechanism follows a **paired design**: a `Symbolic*` class (SymPy matrices for `m_D`, `M_R`, `mu`; full block mass matrix; light-mass formula; scaling/texture/perturbative analyses; `evaluate_numerically(parameter_dict)` bridge) and a numerical class (`SeesawTypeI/II/III`, `InverseSeesaw`, `InverseSeesawExtended`) taking NumPy arrays and exposing `light_neutrino_mass_matrix()` and `diagonalize_light_sector()`. `SymbolicSeesawTypeI` takes `n_generations` and `n_sterile` independently (non-square `m_D`), and also does exact-diagonalization work (`MDaggerM`, `charpoly_relations*`, left/right eigenvectors) used by the `seesaw_N_active_M_sterile` notebooks. Module-level `compare_seesaw_mechanisms` / `compare_seesaw_vs_inverse_seesaw` return result dicts consumed by `visualization.py`.
- `pmns_matrix.py` — standard/PDG PMNS parameterizations, angle extraction, Jarlskog, oscillation probabilities, and `experimental_values_2023()` (global-fit data keyed by ordering `'normal'`/`'inverted'`).
- `visualization.py` — matplotlib/seaborn plots; calls `matplotlib.use('Agg')` at import time, which can stop figures from showing interactively (e.g. in notebooks).
- `notebooks/` — the active research work (e.g. one-generation and N-active/M-sterile Seesaw, Takagi factorization) is happening in notebooks that import from `src`.

Formulas used: Type I `m_ν = -m_D M_R⁻¹ m_Dᵀ`; Inverse Seesaw `m_ν = m_D M_R⁻¹ μ (M_Rᵀ)⁻¹ m_Dᵀ`. Units are not uniform across the code (README mixes GeV and eV), so check the units a function expects before combining inputs.

## Conventions (from `.github/copilot-instructions.md`)

- NumPy for linear algebra, SymPy when parameters are symbolic; keep Seesaw Type I/II/III and Inverse Seesaw implementations separate.
- Follow standard physics conventions for mass ordering, mixing angles and PMNS parameterization; document physical assumptions and approximations in docstrings.
- Use physics-descriptive names (`mass_matrix`, `mixing_angles`, `neutrino_masses`) and NumPy-style docstrings with the math definitions.
