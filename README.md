# Neutrino Mass Matrix Research Project

**Version 0.1.0**

A comprehensive Python package for exploring neutrino mass generation mechanisms, focusing on the Seesaw and Inverse Seesaw mechanisms. This project combines symbolic analysis with numerical calculations to provide deep insights into the physics of neutrino masses.

## 🔬 Features

### Symbolic Analysis (SymPy)
- **Step-by-step derivations** of Seesaw formulas
- **Dimensional analysis** and scaling relations
- **Perturbative expansions** and approximations
- **Texture analysis** with different ansätze
- **Parameter dependence** studies

### Numerical Calculations (NumPy/SciPy)
- **Matrix diagonalization** with numerical stability checks
- **Mass eigenvalue** and mixing angle extraction
- **Experimental comparison** with current data
- **Monte Carlo** parameter scans
- **Visualization** of results

### Physics Models Implemented
- **Type I Seesaw**: Heavy right-handed neutrinos
- **Type II Seesaw**: Higgs triplet mechanism  
- **Type III Seesaw**: Fermionic triplets
- **Inverse Seesaw**: Small lepton number violation
- **Extended models** and hybrid mechanisms

## 📁 Project Structure

```
src/
├── __init__.py                 # Package initialization
├── matrix_utils.py            # Matrix operations and symbolic utilities
├── seesaw.py                  # Seesaw mechanisms (Type I, II, III)
├── inverse_seesaw.py          # Inverse Seesaw mechanism
├── pmns_matrix.py             # PMNS matrix parameterizations
├── symbolic_diagonalization.py # Exact symbolic (Takagi) diagonalization of the Type I seesaw
└── visualization.py           # Plotting and analysis tools

examples/
├── seesaw_vs_inverse_comparison.py    # Comparison of mechanisms
├── matrix_analysis_demo.py            # Matrix diagonalization demo
└── symbolic_seesaw_analysis.py        # Symbolic analysis examples

notebooks/
├── neutrino_mass_exploration.ipynb    # Interactive Jupyter notebook
├── seesaw_one_generation*.ipynb       # (1,1): real case and complex (Takagi) case
├── seesaw_N_active_M_sterile.ipynb    # (2,1), (3,1), (1,2), (2,2), (3,2), (1,3), (3,3)
└── seesaw_feasibility_summary.ipynb   # Where symbolic diagonalization stops being feasible

tests/
├── test_matrix_utils.py               # Unit tests
└── test_symbolic_diagonalization.py   # Symbolic results checked numerically
```

## 🧩 Exact symbolic diagonalization

`src/symbolic_diagonalization.py` and the `seesaw_*` notebooks study the full matrix
`M = [[0, m_D], [m_D^T, M_R]]` for `n` active and `m` sterile neutrinos, with the convention
`U^T M U = diag(m_i)`, `m_i >= 0`, first for real and then for complex `m_D`.

- **Counting.** There are `max(n - m, 0)` massless states and `k = m + min(n, m)` massive ones;
  `n (m - 1)` phases of `m_D` are physical. For `n > m` the problem reduces to `(m, m)`.
- **Direct** (`direct_diagonalization`): masses and `U` from `m_D`, `M_R`. The masses are roots
  of a degree-`k` polynomial, so this works only for `k <= 4`: `(n,1)`, `(1,2)`, `(2,2)`,
  `(3,2)`, `(1,3)`. For `(2,3)` and `(3,3)` no formula in radicals exists.
- **Semi-explicit** (`semi_explicit_diagonalization`): `U` in terms of the parameters and the
  masses; works for any `(n, m)`.
- **Inverse** (`exact_seesaw_parametrization`, `arrowhead_inverse_parameters`): `m_D`, `M_R`
  and `U` from the masses and an orthogonal matrix, exact to all orders in `m_D / M_R`.
  It needs only the square root of a `min(n, m)`-dimensional matrix, so it works up to
  `(3,3)`, where the direct problem has no solution.
- **Expansion** (`seesaw_block_expansion`): block diagonalization to any order.
- `verify_takagi` checks any symbolic result at random numerical points (switching to
  50-digit arithmetic when doubles are not enough), and `feasibility_probe` runs one case
  with a time limit.
- **Large results.** The quartic closed forms have up to ~10^8 nodes written out but only
  ~10^3 distinct subexpressions. `expression_size` reports both numbers and
  `lambdify_large` evaluates such results quickly; `sp.count_ops`, `.free_symbols`,
  `sp.cse` and plain `sp.lambdify` walk the written-out tree and do not finish.

```python
from src.seesaw import SymbolicSeesawTypeI
from src.symbolic_diagonalization import direct_diagonalization, verify_takagi

model = SymbolicSeesawTypeI(1, 2, dirac='real', majorana_basis='diagonal')
result = direct_diagonalization(model)            # cubic, trigonometric roots
verify_takagi(result['U'], model.full_mass_matrix_symbolic(), result['masses'])
```

> **Note**: The package is currently in development. Examples use direct `src.` imports with path modifications. Future versions will support standard `import pyseesaw` package imports.

## 🚀 Quick Start

### 1. Install the Package

**For development (recommended):**
```bash
git clone https://github.com/moiseszeleny/pyseesaw.git
cd pyseesaw
pip install -e .
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Run Examples

**Compare Seesaw mechanisms:**
```bash
cd pyseesaw  # Ensure you're in the project root
python examples/seesaw_vs_inverse_comparison.py
```

**Symbolic analysis:**
```bash
python examples/symbolic_seesaw_analysis.py
```

**Matrix analysis demo:**
```bash
python examples/matrix_analysis_demo.py
```

**Interactive exploration:**
```bash
jupyter notebook notebooks/neutrino_mass_exploration.ipynb
# Or explore specific scenarios:
jupyter notebook notebooks/seesaw_2_active_1_sterile.ipynb
jupyter notebook notebooks/seesaw_3_active_1_sterile.ipynb
jupyter notebook notebooks/seesaw_one_generation.ipynb
```

### 3. Basic Usage

**Current usage (using direct module imports):**
```python
import numpy as np
import sys
import os

# Add project root to path for development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.seesaw import SymbolicSeesawTypeI, SeesawTypeI
from src.inverse_seesaw import SymbolicInverseSeesaw, InverseSeesaw

# Symbolic analysis
symbolic_seesaw = SymbolicSeesawTypeI(n_generations=3)
scaling_analysis = symbolic_seesaw.scaling_analysis()
print(f"Light neutrino scale: {scaling_analysis['light_scale']}")

# Numerical calculation
m_D = 1e-2 * np.random.rand(3, 3)  # Dirac masses (GeV)
M_R = 1e15 * np.eye(3)             # Heavy masses (eV)
seesaw = SeesawTypeI(m_D, M_R)
masses, mixing = seesaw.diagonalize_light_sector()
```

**Intended package usage (when package structure is fixed):**
```python
import numpy as np
import pyseesaw
from pyseesaw.seesaw import SymbolicSeesawTypeI, SeesawTypeI
from pyseesaw.inverse_seesaw import SymbolicInverseSeesaw, InverseSeesaw

# Same usage as above...
```

## 🧮 Physics Background

### Type I Seesaw Mechanism

The light neutrino mass matrix is given by:

$$m_ν = -m_D M_R⁻¹ m_D^T$$


Where:
- $m_D$: Dirac mass matrix (∼ electroweak scale)
- $M_R$: Heavy Majorana mass matrix (∼ GUT scale)
- Small neutrino masses arise from $m_D^2/M_R$ suppression

### Inverse Seesaw Mechanism

The light neutrino mass matrix is:

$$m_ν = m_D M_R^{-1} μ (M_R^T)^{-1} m_D^T$$


Where:
- $\mu$: Small lepton number violating parameter
- Naturally small masses without large hierarchy
- Linear scaling with $\mu$ parameter

## 📊 Key Features

### Symbolic Capabilities
- **Automatic derivation** of Seesaw formulas
- **Dimensional analysis** of mass scales
- **Perturbative expansions** in small parameters
- **Texture zero** analysis
- **Parameter dependencies**

### Numerical Tools
- **Stable diagonalization** algorithms
- **Mass ordering** enforcement
- **Unitarity checks** for mixing matrices
- **Experimental constraints** comparison
- **Monte Carlo** parameter studies

### Visualization
- **Mass spectra** plots
- **Mixing matrix** heatmaps
- **Parameter space** scans
- **Comparison** between mechanisms
- **Unitarity triangles**

## 🔍 Examples and Use Cases

### 1. Pedagogical Analysis
```python
import sys
import os
# Add project root to path for development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Understand Seesaw mechanism step-by-step
from src.seesaw import SymbolicSeesawTypeI
symbolic_seesaw = SymbolicSeesawTypeI()
case_2x2 = symbolic_seesaw.simplified_2x2_case()
print("2×2 mass matrix:")
print(case_2x2['mass_matrix'])
```

### 2. Phenomenological Studies
```python
# Compare with experimental data
from src.pmns_matrix import pmns_from_experimental, extract_mixing_angles
exp_pmns = pmns_from_experimental('normal')
theta12_exp, theta13_exp, theta23_exp, delta_exp = extract_mixing_angles(exp_pmns)
```

### 3. Model Building
```python
# Test different textures
from src.seesaw import SymbolicSeesawTypeI
seesaw = SymbolicSeesawTypeI()
seesaw.set_texture(dirac_texture=[(0,1), (1,0)])  # No 1-2 mixing
m_textured = seesaw.light_mass_matrix_symbolic()
```

## 📚 Documentation

### Key Classes

- **`SymbolicSeesawTypeI`**: Symbolic Type I Seesaw analysis
- **`SeesawTypeI`**: Numerical Type I Seesaw calculations
- **`SymbolicInverseSeesaw`**: Symbolic Inverse Seesaw analysis
- **`InverseSeesaw`**: Numerical Inverse Seesaw calculations

### Key Functions

- **`diagonalize_mass_matrix()`**: Robust matrix diagonalization
- **`seesaw_approximation_symbolic()`**: Symbolic Seesaw formula
- **`create_symbolic_matrix()`**: Generate symbolic matrices
- **`substitute_numerical_values()`**: Convert symbolic → numerical

## 🧪 Testing

Run the test suite:
```bash
python tests/test_matrix_utils.py
```

Or install pytest for more comprehensive testing:
```bash
python tests/test_matrix_utils.py
python -m unittest tests.test_symbolic_diagonalization
```

## 🤝 Contributing

This project is designed for neutrino physics research and education. Contributions are welcome!

**Current Development Priority:**
- Fixing package import structure for standard `import pyseesaw` usage
- Adding more comprehensive tests
- Improving documentation

### Areas for Contribution
- Additional Seesaw mechanisms (e.g., radiative)
- Extended phenomenological models
- Cosmological implications
- Machine learning applications
- Educational materials
- Package structure improvements

## 📖 References

1. Minkowski, P. (1977). μ → eγ at a rate of one out of 109 muon decays?
2. Mohapatra, R. N., & Senjanović, G. (1980). Neutrino mass and spontaneous parity nonconservation
3. Schechter, J., & Valle, J. W. F. (1980). Neutrino masses in SU(2) ⊗ U(1) theories
4. Mohapatra, R. N. (1986). Mechanism for understanding small neutrino mass in superstring theories
5. Gonzalez-Garcia, M. C., & Yokoyama, J. (2013). Neutrino masses and mixing: evidence and implications

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🎯 Goals

This project aims to:
1. **Educate** about neutrino mass generation mechanisms
2. **Facilitate** research in neutrino physics
3. **Provide** robust computational tools
4. **Bridge** symbolic and numerical approaches
5. **Enable** exploration of new physics beyond the Standard Model

---

**Happy neutrino physics exploration! 🌌**
