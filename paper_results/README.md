# Paper results: exact Takagi diagonalization of the Type I seesaw

Scripts that produce every table, formula and figure for the paper on the
exact diagonalization of the Type I seesaw, from (1,1) up to the (3,2) minimal
seesaw, and on the Galois groups of the characteristic polynomial. All outputs
go to `paper_results/output/` and are regenerated from the library in `src/`.

```bash
.venv/bin/python paper_results/galois_tables.py        # ~3 min
.venv/bin/python paper_results/closed_forms.py         # ~1 min
.venv/bin/python paper_results/exact_vs_seesaw_11.py   # seconds
.venv/bin/python paper_results/minimal_seesaw_32.py    # ~30 s
```

Conventions: `M = [[0, m_D], [m_D^T, M_R]]`, `U^T M U = diag(m_i)` with
`m_i >= 0`, `M_R = diag(M_i)` unless stated otherwise. Masses in GeV.

## 1. Galois groups (`galois_tables.py`, `src/galois_analysis.py`)

The masses are the roots of the reduced characteristic polynomial of degree
`k = m + min(n, m)`. These are the proof rules, stated in the module docstring:

* **Not solvable**: one good rational specialization whose Galois group is not
  solvable proves the generic group is not solvable. A specialization with
  group `S_k` proves `Gal = S_k`.
* **Solvable**: shown by the generic factorization over `Q(parameters)`, or by
  a functional decomposition `P = g_1 ∘ g_2 ∘ ...`, into pieces of degree `<= 4`.
* **Real radicals**: all roots are real. A real-radical formula exists only
  when the group order is a power of 2 (casus irreducibilis, generalized).

| (n,m) | k | Gal (real m_D) | Gal (complex m_D) | radicals | real radicals |
|---|---|---|---|---|---|
| (1,1), (2,1), (3,1) | 2 | S2 | S2 | yes | yes |
| (1,2) | 3 | S3 | S3 | yes | no |
| (2,2), (3,2), (1,3) | 4 | S4 | S4 | yes | no |
| (2,3) | 5 | S5 | S5 | **no** | no |
| (3,3) | 6 | S6 | S6 | **no** | no |

Every entry is proven, and the specialization attains the upper bound
(`galois_generic.{csv,tex}`).

Special structures (`galois_named_textures.{csv,tex}`):

* **Degenerate heavy neutrinos, `M_R = M·1`, real `m_D`**: the polynomial is
  `Q(λ(λ − M))`, where `Q` is the characteristic polynomial of `m_D m_D^T`.
  For (3,3) the group is `C2 ≀ S3 ≅ S4 × C2` (order 48), so the case is
  solvable. It amounts to three (1,1) seesaws in the singular-value basis.
  For (2,2) and (3,2) the group is `D4`, which needs only real square roots.
  **With CP phases the group is S6 (respectively S4)**: the reduction needs a
  unitary sterile rotation, and that rotation does not preserve `M_R ∝ 1`.
* **Texture zeros of `m_D`** (diagonal `M_R`, all patterns modulo permutations
  in `galois_texture_zeros_*.csv`): a zero texture lowers the Galois group
  *only* when it splits the mass matrix into disconnected blocks (`blocks > 1`).
  Rank-deficient textures add a massless state and drop the degree by one.
  Every connected (3,3) texture keeps S6 and every connected (2,3) texture
  keeps S5.
* **Frampton–Glashow–Yanagida two-zero textures of (3,2)** and **μ–τ symmetric
  `m_D`**: still S4.

## 2. Closed forms (`closed_forms.py`)

For each case below, the masses and the full mixing matrix are exported as a
chain of definitions (`nested_definitions`). Each chain is checked with
`verify_takagi`, and again against the original expression at 40 digits.

| case | m_D | method | size written out | distinct subexpressions | definitions |
|---|---|---|---|---|---|
| (1,1) | real | direct | 254 | 33 | 6 |
| (1,2) | real | direct | 7.3e3 | 114 | 23 |
| (2,2) | real | direct | 5.2e5 | 262 | 67 |
| **(3,2)** | real | direct | 3.4e6 | 552 | **123** |
| **(3,2)** | complex | direct | 7.5e8 | 1402 | **262** |
| (3,3) | real | semi-explicit | 1.1e5 | 1932 | 427 |
| **(3,3), `M_R = M·1`** | real | direct | 1.3e7 | 2017 | **457** |

`closed_form_11.tex` is the readable (1,1) result next to the seesaw
expansion. The `closed_form_<nm>_<cp>_<method>.tex` files are `align*` bodies
for an appendix or supplementary material.

## 3. Phenomenology of the (3,2) minimal seesaw (`minimal_seesaw_32.py`)

Inputs come from `experimental_values_2023()` in `src/pmns_matrix.py`. Those
values are approximate: **replace them with the global fit you cite before
producing final figures**. The lightest neutrino is massless, heavy masses
are `(M, 1.5 M)`, and the Casas–Ibarra angle is `z = 0.3 + i y`.

* **Exact parametrization** (`casas_ibarra_exact_numeric`): masses and the
  unitary part of the mixing are reproduced to the precision of the inputs
  (1e-15), at any `M` and `y`. The non-unitarity is `η = 1 − U_a c_l U_a†`.
* **Leading-order Casas–Ibarra** (`lo_accuracy.pdf`): the relative deviation of
  the oscillation observables is `~U² = Σ|Θ_αi|²` (ratio 1.6–1.7 for NO and 1.1–11
  for IO, where `U² > 10⁻¹²`; below that, the 1e-16 rounding of the inputs dominates).
  It stays below
  `10⁻³` wherever `|η| ≲ 10⁻³`, so for the light sector the LO formula is
  adequate in the allowed region.
* **Heavy masses** (`heavy_splitting.pdf`): with the LO matrix, *both* heavy
  masses shift by a common `O(m_ν cosh 2y)` amount (2.4 keV at `M = 1 GeV`,
  `y = 6`). The *splitting*, however, shifts by an `Im z`-independent amount:
  it equals `(m₃ − m₂)` for `ΔM ≪ m_ν` and `(m₃ − m₂) cos 2 Re z` for
  `ΔM ≫ m_ν` (NO). For quasi-degenerate heavy neutrinos with
  `ΔM ≲ 0.1 eV`, as in resonant leptogenesis or HNL oscillations, the LO input
  splitting is therefore wrong by O(1). The exact parametrization has no such
  error.
* **Numerical precision** (`double_precision.pdf`): light masses taken from a
  double-precision Takagi factorization of the full matrix have relative error
  `~10⁻¹⁶ M/m` (4 % at `M = 10⁴` GeV, O(1) above `10⁶` GeV). `mpmath` with
  `required_precision(...)` digits fixes this.
* **Closed-form benchmarks** (`closed_form_benchmarks.tex`): the exact quartic
  solution of (3,2), for real and complex `m_D`, reproduces the physical masses
  at five benchmark points (NO/IO, `M` from 10 GeV to 10¹⁰ GeV, `Im z` up to 4).
  It needs about **twice** the digits of a direct diagonalization, because the
  quartic formula for the squared masses cancels like `(M/m)⁴`. In double
  precision it is useless at physical hierarchies.

## 4. (1,1) baseline (`exact_vs_seesaw_11.py`)

Exact `m₁ = M(√(1+4ε²) − 1)/2` and `tan 2θ = 2ε`, with `ε = m_D/M`, against
`m₁ ≃ M ε²` (relative error `ε²`), the next order `M(ε² − ε⁴)` (error `2ε⁴`),
and `θ ≃ ε` (error `4ε²/3`). These are checked against the symbolic closed form
at 50 digits.

## Caveats for the manuscript

* `experimental_values_2023()` is labeled NuFIT 5.2, but its values are
  rounded and come with no uncertainties. Update it, or pass your own inputs,
  before quoting numbers.
* `|η| ≈ 10⁻³` on the figures is only an indicative contour, not a fit to
  electroweak-precision data.
* The specialization argument needs specializations that stay separable with
  the generic multiplicity structure. The code keeps only those. Name it in
  the paper; standard references are Lang, *Algebra* VII.2, and Serre, *Topics
  in Galois Theory* §3.3. For real radicals, cite Isaacs, Amer. Math. Monthly
  92 (1985) 571.
