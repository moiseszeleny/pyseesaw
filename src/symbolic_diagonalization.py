r"""
Exact symbolic diagonalization of the Type I seesaw mass matrix.

The full Majorana mass matrix for ``n`` active and ``m`` sterile neutrinos,

.. math:: M = \begin{pmatrix} 0 & m_D \\ m_D^T & M_R \end{pmatrix},

is complex symmetric and is diagonalized by a unitary ``U`` (Takagi
factorization) with the convention used throughout this package:

.. math:: U^T M U = \mathrm{diag}(m_1, \dots, m_{n+m}), \qquad m_i \ge 0 .

Three complementary routes are implemented:

* **Direct** (:func:`direct_diagonalization`): masses and mixing as functions
  of the Lagrangian parameters. The number of massive states is
  ``k = m + min(n, m)`` and the masses are roots of a degree-``k`` polynomial,
  so closed forms exist in general only for ``k <= 4``.
* **Semi-explicit** (:func:`semi_explicit_diagonalization`): the mixing matrix
  as a rational function of the parameters *and* the masses, the latter being
  constrained by the characteristic polynomial. Works for any ``(n, m)``.
* **Inverse** (:func:`exact_seesaw_parametrization`,
  :func:`arrowhead_inverse_parameters`): ``m_D`` and ``M_R`` as functions of
  the physical masses and mixing parameters; an exact (all orders in
  ``m_D / M_R``) version of the Casas-Ibarra parametrization.

The perturbative block diagonalization (:func:`seesaw_block_expansion`) is the
fallback when no closed form exists, and :func:`verify_takagi` checks any
symbolic result at random numerical points.
"""

import multiprocessing
import time
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mpmath
import numpy as np
import sympy as sp

from .matrix_utils import seesaw_block_expansion, takagi_factorization
from .seesaw import SymbolicSeesawTypeI

__all__ = [
    "NotSolvableInRadicals",
    "solve_cubic_trigonometric",
    "solve_quartic_ferrari",
    "solve_reduced_charpoly",
    "eigenvector_from_adjugate",
    "massless_eigenvectors",
    "majorana_phase_matrix",
    "eigenvector_eigenvalue_identity",
    "reduce_active_space",
    "mixing_matrix_symbolic",
    "direct_diagonalization",
    "semi_explicit_diagonalization",
    "charpoly_matching_equations",
    "arrowhead_inverse_parameters",
    "arrowhead_mixing_matrix",
    "orthogonal_matrix",
    "hermitian_matrix_function",
    "exact_seesaw_parametrization",
    "verify_takagi",
    "expression_size",
    "lambdify_large",
    "numeric_mass_points",
    "feasibility_probe",
    "seesaw_block_expansion",
]


class NotSolvableInRadicals(ValueError):
    """Raised when a polynomial of degree >= 5 has no general closed-form roots."""

    def __init__(self, degree: int):
        self.degree = degree
        super().__init__(
            f"Irreducible polynomial of degree {degree}: no general solution "
            "in radicals (Abel-Ruffini). Use the semi-explicit form, the "
            "inverse parametrization or the seesaw expansion instead."
        )


# ---------------------------------------------------------------------------
# Roots of the reduced characteristic polynomial
# ---------------------------------------------------------------------------


def solve_cubic_trigonometric(coefficients: Sequence[sp.Expr]) -> List[sp.Expr]:
    r"""
    Roots of a monic cubic with three real roots, in trigonometric (Viete) form.

    For ``x^3 + a x^2 + b x + c`` write ``x = t - a/3`` to get the depressed
    cubic ``t^3 + p t + q`` and

    .. math::
        t_k = 2\sqrt{-\frac{p}{3}}\,
              \cos\!\left[\frac{1}{3}\arccos\!\left(\frac{3q}{2p}
              \sqrt{-\frac{3}{p}}\right) - \frac{2\pi k}{3}\right].

    Assumes three real roots (``p < 0``), which holds for characteristic
    polynomials of real symmetric / Hermitian matrices. In that case Cardano's
    formula would need complex cube roots (casus irreducibilis); this form
    stays manifestly real.

    Parameters:
    -----------
    coefficients : sequence of sp.Expr
        ``(a, b, c)`` of ``x^3 + a x^2 + b x + c``

    Returns:
    --------
    list of sp.Expr
        The three roots in ascending order
    """
    a, b, c = [sp.sympify(x) for x in coefficients]
    p = b - a**2 / 3
    q = 2 * a**3 / 27 - a * b / 3 + c
    amplitude = 2 * sp.sqrt(-p / 3)
    angle = sp.acos(3 * q / (2 * p) * sp.sqrt(-3 / p)) / 3
    roots = [amplitude * sp.cos(angle - 2 * sp.pi * k / 3) - a / 3 for k in range(3)]
    return roots[::-1]


def solve_quartic_ferrari(coefficients: Sequence[sp.Expr]) -> List[sp.Expr]:
    r"""
    Roots of a monic quartic with four real roots (Ferrari's method).

    For ``x^4 + a x^3 + b x^2 + c x + d`` write ``x = y - a/4`` to get
    ``y^4 + p y^2 + q y + r``. With ``mu`` the largest root of the resolvent
    cubic ``mu^3 + p mu^2 + (p^2/4 - r) mu - q^2/8 = 0`` (obtained in
    trigonometric form),

    .. math::
        y = \frac{s\sqrt{2\mu} \pm \sqrt{-2p - 2\mu - s\,2q/\sqrt{2\mu}}}{2},
        \qquad s = \pm 1 .

    Assumes four real roots (real symmetric / Hermitian matrix) and ``q != 0``.
    Choosing the largest resolvent root pairs the two smallest and the two
    largest roots, so the result is ordered.

    Parameters:
    -----------
    coefficients : sequence of sp.Expr
        ``(a, b, c, d)`` of ``x^4 + a x^3 + b x^2 + c x + d``

    Returns:
    --------
    list of sp.Expr
        The four roots in ascending order
    """
    a, b, c, d = [sp.sympify(x) for x in coefficients]
    p = b - sp.Rational(3, 8) * a**2
    q = c - a * b / 2 + a**3 / 8
    r = d - a * c / 4 + a**2 * b / 16 - sp.Rational(3, 256) * a**4

    # The biquadratic case is only tested when q is small enough to expand;
    # for large nested coefficients a generic (non-vanishing) q is assumed.
    if q == 0 or (expression_size(q)["tree"] < 5000 and sp.expand(q) == 0):
        # Biquadratic: y^2 = (-p +- sqrt(p^2 - 4 r)) / 2
        disc = sp.sqrt(p**2 - 4 * r)
        inner = sp.sqrt((-p - disc) / 2)
        outer = sp.sqrt((-p + disc) / 2)
        return [-outer - a / 4, -inner - a / 4, inner - a / 4, outer - a / 4]

    mu = solve_cubic_trigonometric((p, p**2 / 4 - r, -(q**2) / 8))[-1]
    root_2mu = sp.sqrt(2 * mu)
    delta_plus = sp.sqrt(-2 * p - 2 * mu - 2 * q / root_2mu)
    delta_minus = sp.sqrt(-2 * p - 2 * mu + 2 * q / root_2mu)
    return [
        (-root_2mu - delta_minus) / 2 - a / 4,
        (-root_2mu + delta_minus) / 2 - a / 4,
        (root_2mu - delta_plus) / 2 - a / 4,
        (root_2mu + delta_plus) / 2 - a / 4,
    ]


def _roots_of_irreducible(poly: sp.Poly) -> List[sp.Expr]:
    """Closed-form real roots of one (irreducible) factor, ascending."""
    return _roots_from_coefficients([c / poly.LC() for c in poly.all_coeffs()][1:])


def _roots_with_factors(
    poly: sp.Poly, try_factor: bool = True
) -> List[Tuple[sp.Expr, sp.Poly]]:
    """Closed-form roots, each paired with the irreducible factor it belongs to."""
    lamb = poly.gen
    factors = [(poly, 1)]
    if try_factor and poly.degree() > 1:
        try:
            _, factor_list = sp.factor_list(poly.as_expr())
            candidate = [
                (sp.Poly(f, lamb), mult) for f, mult in factor_list if f.has(lamb)
            ]
            if sum(f.degree() * mult for f, mult in candidate) == poly.degree():
                factors = candidate
        except (sp.PolynomialError, NotImplementedError):
            pass

    pairs: List[Tuple[sp.Expr, sp.Poly]] = []
    for factor, multiplicity in factors:
        for root in _roots_of_irreducible(factor) * multiplicity:
            pairs.append((root, factor))
    return pairs


def solve_reduced_charpoly(
    poly: Union[sp.Poly, sp.Expr],
    lamb: Optional[sp.Symbol] = None,
    try_factor: bool = True,
) -> List[sp.Expr]:
    r"""
    Closed-form roots of the (reduced) characteristic polynomial.

    The polynomial is first factorized over its symbolic coefficients (special
    textures often factorize even when the generic case does not); each
    irreducible factor of degree ``<= 4`` is solved with a manifestly real
    formula (quadratic, trigonometric cubic, Ferrari quartic).

    Assumes all roots are real, as for ``det(lambda - M)`` with ``M`` real
    symmetric or ``det(lambda - M^\dagger M)``.

    Parameters:
    -----------
    poly : sp.Poly or sp.Expr
        Polynomial in ``lamb``
    lamb : sp.Symbol, optional
        Polynomial variable (taken from ``poly`` if it is a ``sp.Poly``)
    try_factor : bool, optional
        Attempt a symbolic factorization first (default: True)

    Returns:
    --------
    list of sp.Expr
        Roots; ascending within each irreducible factor, hence ascending
        overall when the polynomial is irreducible

    Raises:
    -------
    NotSolvableInRadicals
        If an irreducible factor has degree ``>= 5``
    """
    if not isinstance(poly, sp.Poly):
        if lamb is None:
            raise ValueError("lamb is required when poly is an expression")
        poly = sp.Poly(poly, lamb)
    return [root for root, _ in _roots_with_factors(poly, try_factor)]


# ---------------------------------------------------------------------------
# Eigenvectors and mixing matrices
# ---------------------------------------------------------------------------


def _is_real_matrix(matrix: sp.Matrix) -> bool:
    """
    True if the matrix is built from real symbols without an explicit ``I``.

    SymPy cannot infer reality of quotients such as ``x / y`` (``y`` might
    vanish), so the entries are judged by their ingredients instead.
    """
    return not matrix.has(sp.I) and all(
        symbol.is_real for symbol in matrix.free_symbols
    )


def _hermitian_norm(vector: sp.Matrix, real: Optional[bool] = None) -> sp.Expr:
    if real is None:
        real = _is_real_matrix(vector)
    if real:
        return sp.sqrt(sum((x**2 for x in vector), sp.S.Zero))
    return sp.sqrt(sum((sp.conjugate(x) * x for x in vector), sp.S.Zero))


def _orthonormalize(
    vectors: Sequence[sp.Matrix], simplify: Optional[Callable] = None
) -> List[sp.Matrix]:
    """Gram-Schmidt with the Hermitian inner product."""
    # Decide reality once, on the inputs: simplification can hide it later
    real = all(_is_real_matrix(vector) for vector in vectors)
    basis: List[sp.Matrix] = []
    for vector in vectors:
        w = vector
        for e in basis:
            w = w - e * ((e.T if real else e.H) * vector)[0, 0]
        if simplify is not None:
            w = w.applyfunc(simplify)
        w = w / _hermitian_norm(w, real)
        if simplify is not None:
            w = w.applyfunc(simplify)
        basis.append(w)
    return basis


def eigenvector_from_adjugate(
    matrix: sp.Matrix, eigenvalue: sp.Expr, column: int = -1
) -> sp.Matrix:
    r"""
    Unnormalized eigenvector from a column of the adjugate.

    If ``lambda_i`` is a simple eigenvalue of ``A``, then
    ``adj(A - lambda_i) \propto v_i w_i^T`` has rank one and every non-zero
    column is an eigenvector. The entries are polynomials in the matrix
    elements and ``lambda_i``, so the eigenvalue may be kept as a symbol.

    Parameters:
    -----------
    matrix : sp.Matrix
        Square matrix ``A``
    eigenvalue : sp.Expr
        Simple eigenvalue (symbol or explicit expression)
    column : int, optional
        Which column of the adjugate to use (default: last). Choose another
        one if the corresponding component of the eigenvector vanishes.

    Returns:
    --------
    sp.Matrix
        Column vector ``v`` with ``A v = eigenvalue * v``
    """
    placeholder = sp.Dummy("lambda")
    vector = _adjugate_column(matrix, placeholder, column)
    return vector.xreplace({placeholder: sp.sympify(eigenvalue)})


def _adjugate_column(matrix: sp.Matrix, variable: sp.Symbol, column: int) -> sp.Matrix:
    """Column of ``adj(matrix - variable)``, polynomial in ``variable``."""
    dim = matrix.rows
    column = column % dim
    # Work with a placeholder so the determinants stay polynomial; inserting a
    # large closed-form root before taking them is prohibitively slow.
    shifted = matrix - variable * sp.eye(dim)
    # adj[a, c] = cofactor(c, a)
    return sp.Matrix(
        [shifted.cofactor(column, a, method="berkowitz") for a in range(dim)]
    )


def _nonvanishing_column(matrix: sp.Matrix, factor: sp.Poly, preferred: int) -> int:
    """
    Adjugate column that does not vanish on the roots of ``factor``.

    When the characteristic polynomial factorizes (decoupled blocks), an
    eigenvector can have vanishing components, and the corresponding adjugate
    columns are zero. A column vanishes at every root of the irreducible
    ``factor`` exactly when all its entries are divisible by it.
    """
    dim = matrix.rows
    variable = factor.gen
    for shift in range(dim):
        column = (preferred + shift) % dim
        vector = _adjugate_column(matrix, variable, column)
        if any(
            sp.rem(sp.expand(entry), factor.as_expr(), variable) != 0
            for entry in vector
        ):
            return column
    raise ValueError("All adjugate columns vanish: degenerate eigenvalue")


def massless_eigenvectors(
    m_D: sp.Matrix, simplify: Optional[Callable] = sp.simplify
) -> List[sp.Matrix]:
    r"""
    Orthonormal massless states of the seesaw matrix for ``n > m``.

    They are purely active, ``(u, 0)`` with ``m_D^T u = 0``, i.e. ``u`` spans
    the null space of ``m_D^T``.

    Parameters:
    -----------
    m_D : sp.Matrix
        Dirac mass matrix (``n x m``)
    simplify : callable, optional
        Applied element-wise during orthonormalization

    Returns:
    --------
    list of sp.Matrix
        ``n - rank(m_D)`` column vectors of length ``n + m``
    """
    n_active, n_sterile = m_D.shape
    null_space = _orthonormalize(m_D.T.nullspace(), simplify)
    return [sp.Matrix.vstack(u, sp.zeros(n_sterile, 1)) for u in null_space]


def majorana_phase_matrix(eigenvalue_signs: Sequence[int]) -> sp.Matrix:
    r"""
    Phase matrix turning signed eigenvalues into positive masses.

    A real symmetric ``M`` is diagonalized by a real orthogonal ``O``,
    ``O^T M O = diag(lambda_i)`` with ``lambda_i`` of either sign. Then
    ``U = O P`` with ``P = diag(i if lambda_i < 0 else 1)`` satisfies
    ``U^T M U = diag(|lambda_i|)``. The factors of ``i`` are physical
    Majorana (CP-parity) phases.

    Parameters:
    -----------
    eigenvalue_signs : sequence of int
        Sign (+1, -1, or 0 for massless) of each eigenvalue

    Returns:
    --------
    sp.Matrix
        Diagonal phase matrix ``P``
    """
    return sp.diag(*[sp.I if sign < 0 else sp.S.One for sign in eigenvalue_signs])


def eigenvector_eigenvalue_identity(
    matrix: sp.Matrix, eigenvalues: Sequence[sp.Expr], i: int, a: int
) -> sp.Expr:
    r"""
    Squared modulus of an eigenvector component from eigenvalues alone.

    For a Hermitian matrix ``A`` with eigenvalues ``lambda_j`` and normalized
    eigenvectors ``v_j``,

    .. math::
        |v_{i,a}|^2 = \frac{\det(\lambda_i - A_{(a)})}
                           {\prod_{j \ne i}(\lambda_i - \lambda_j)},

    where ``A_(a)`` is ``A`` with row and column ``a`` removed
    (Denton, Parke, Tao, Zhang, arXiv:1908.03795).

    Parameters:
    -----------
    matrix : sp.Matrix
        Hermitian matrix ``A``
    eigenvalues : sequence of sp.Expr
        All eigenvalues of ``A``
    i : int
        Eigenvector index
    a : int
        Component index

    Returns:
    --------
    sp.Expr
        ``|v_{i,a}|^2``
    """
    minor = matrix.minor_submatrix(a, a)
    numerator = (eigenvalues[i] * sp.eye(minor.rows) - minor).det(method="berkowitz")
    denominator = sp.Mul(
        *[eigenvalues[i] - eigenvalues[j] for j in range(len(eigenvalues)) if j != i]
    )
    return numerator / denominator


def reduce_active_space(
    m_D: sp.Matrix, simplify: Optional[Callable] = sp.simplify
) -> sp.Matrix:
    r"""
    Active-space rotation that isolates the massless states for ``n > m``.

    Returns a unitary ``V`` (``n x n``) such that

    .. math:: V^T m_D = \begin{pmatrix} 0 \\ m_D' \end{pmatrix},

    with ``m_D'`` an ``m x m`` (upper-triangular) matrix. Acting with
    ``diag(V, 1)`` on the full matrix, the first ``n - m`` active states
    decouple with zero mass and the rest is an ``(m, m)`` seesaw. This is how
    ``(2,1)`` and ``(3,1)`` reduce to ``(1,1)``, and ``(3,2)`` to ``(2,2)``.

    Parameters:
    -----------
    m_D : sp.Matrix
        Dirac mass matrix (``n x m``, full column rank, ``n >= m``)
    simplify : callable, optional
        Applied element-wise during orthonormalization

    Returns:
    --------
    sp.Matrix
        Unitary ``V``; real orthogonal if ``m_D`` is real
    """
    n_active, n_sterile = m_D.shape
    if n_active < n_sterile:
        raise ValueError("reduce_active_space requires n_active >= n_sterile")
    null_space = _orthonormalize(m_D.T.nullspace(), simplify)
    column_space = _orthonormalize([m_D[:, j] for j in range(n_sterile)], simplify)
    if not _is_real_matrix(m_D):
        column_space = [q.applyfunc(sp.conjugate) for q in column_space]
    return sp.Matrix.hstack(*(null_space + column_space))


def mixing_matrix_symbolic(
    mass_matrix: sp.Matrix,
    eigenvalues: Sequence[sp.Expr],
    masses: Sequence[sp.Expr],
    hermitian_square: bool = False,
    null_vectors: Sequence[sp.Matrix] = (),
    column: Union[int, Sequence[int]] = -1,
) -> sp.Matrix:
    r"""
    Takagi mixing matrix from the eigenvalues, ``U^T M U = diag(0, .., masses)``.

    Massive columns are built from the adjugate
    (:func:`eigenvector_from_adjugate`) and normalized by the Takagi condition
    ``u_i^T M u_i = m_i``, which fixes both the modulus and the phase so that
    ``M u_i = m_i u_i^*``. For real ``M`` the phase is ``i`` for negative
    eigenvalues and 1 otherwise (see :func:`majorana_phase_matrix`).

    Assumes non-degenerate massive states.

    Parameters:
    -----------
    mass_matrix : sp.Matrix
        Complex symmetric mass matrix ``M``
    eigenvalues : sequence of sp.Expr
        Eigenvalues of ``M`` (``hermitian_square=False``, ``M`` real) or of
        ``M^\dagger M`` (``hermitian_square=True``) for the massive states
    masses : sequence of sp.Expr
        Positive masses associated with ``eigenvalues``, in the same order
    hermitian_square : bool, optional
        Whether ``eigenvalues`` refer to ``M^\dagger M``
    null_vectors : sequence of sp.Matrix, optional
        Orthonormal massless eigenvectors, placed first
    column : int or sequence of int, optional
        Adjugate column used for the eigenvectors (one value for all, or one
        per eigenvalue)

    Returns:
    --------
    sp.Matrix
        Unitary mixing matrix (massless columns first)
    """
    operator = mass_matrix.H * mass_matrix if hermitian_square else mass_matrix
    if isinstance(column, int):
        column = [column] * len(eigenvalues)
    columns = list(null_vectors)
    for eigenvalue, mass, adjugate_column in zip(eigenvalues, masses, column):
        vector = eigenvector_from_adjugate(operator, eigenvalue, adjugate_column)
        if hermitian_square:
            # Takagi normalization u^T M u = m fixes modulus and phase at once.
            # (Equivalent to u/|u| * sqrt(m/c), written so that no square root
            # of a large quotient has to be auto-simplified by SymPy.)
            bilinear = (vector.T * mass_matrix * vector)[0, 0]
            columns.append(vector * sp.sqrt(mass) / sp.sqrt(bilinear))
        else:
            vector = vector / _hermitian_norm(vector)
            columns.append(vector * sp.sqrt(mass / eigenvalue))
    return sp.Matrix.hstack(*columns)


def direct_diagonalization(
    model: SymbolicSeesawTypeI,
    hermitian_square: Optional[bool] = None,
    try_factor: bool = True,
    column: int = -1,
) -> Dict[str, object]:
    r"""
    Masses and mixing matrix as explicit functions of ``m_D`` and ``M_R``.

    Real case (``model.dirac == 'real'``): the roots ``lambda_i`` of the
    reduced ``det(lambda - M)`` are signed masses. For positive-definite
    ``M_R`` Sylvester's law of inertia gives exactly ``min(n, m)`` negative
    roots (the light massive states) and ``m`` positive ones (heavy), so
    ``m_i = |lambda_i|`` and the light columns carry a Majorana phase ``i``.

    Complex case: the roots of the reduced ``det(lambda - M^\dagger M)`` are
    the squared masses and each column is rephased to satisfy the Takagi
    condition.

    Parameters:
    -----------
    model : SymbolicSeesawTypeI
        Symbolic model; ``majorana_basis='diagonal'`` with ``dirac='real'``
        or ``'polar'`` is recommended
    hermitian_square : bool, optional
        Force the ``M^\dagger M`` route (default: only if ``m_D`` is not real)
    try_factor : bool, optional
        Passed to :func:`solve_reduced_charpoly`
    column : int, optional
        Adjugate column used for the eigenvectors

    Returns:
    --------
    dict
        ``'polynomial'`` (reduced ``sp.Poly``), ``'degree'``, ``'eigenvalues'``
        (roots, ordered as ``'masses'``), ``'masses'`` (massive states only,
        light first), ``'U'`` (full mixing matrix, massless columns first),
        ``'hermitian_square'``, ``'factorized'``. If the polynomial factorizes
        (special textures), the states are grouped by factor instead of being
        ordered light-first, and real-case masses are given as ``|lambda_i|``.

    Raises:
    -------
    NotSolvableInRadicals
        If the number of massive states exceeds four (generic parameters)
    """
    if hermitian_square is None:
        hermitian_square = model.dirac != "real"
    lamb = sp.Symbol("lambda", real=True)
    polynomial = model.reduced_charpoly(hermitian_square=hermitian_square, lamb=lamb)
    pairs = _roots_with_factors(polynomial, try_factor)
    factorized = len({factor for _, factor in pairs}) > 1
    mass_matrix = model.full_mass_matrix_symbolic()
    operator = mass_matrix.H * mass_matrix if hermitian_square else mass_matrix

    if hermitian_square:
        masses = [sp.sqrt(root) for root, _ in pairs]
    elif not factorized:
        # Ascending roots: the first min(n, m) are negative (light states)
        n_light = min(model.n_gen, model.n_sterile)
        pairs = pairs[:n_light][::-1] + pairs[n_light:]
        masses = [-root for root, _ in pairs[:n_light]] + [
            root for root, _ in pairs[n_light:]
        ]
    else:
        # Roots are only ordered within each factor: signs are not known a priori
        masses = [sp.Abs(root) for root, _ in pairs]
    eigenvalues = [root for root, _ in pairs]

    if factorized:
        chosen = {}
        for _, factor in pairs:
            if factor not in chosen:
                chosen[factor] = _nonvanishing_column(operator, factor, column)
        columns: Union[int, List[int]] = [chosen[factor] for _, factor in pairs]
    else:
        columns = column

    null_vectors = massless_eigenvectors(model.m_D_sym) if model.n_massless else []
    U = mixing_matrix_symbolic(
        mass_matrix,
        eigenvalues,
        masses,
        hermitian_square=hermitian_square,
        null_vectors=null_vectors,
        column=columns,
    )
    return {
        "polynomial": polynomial,
        "degree": polynomial.degree(),
        "eigenvalues": eigenvalues,
        "masses": masses,
        "U": U,
        "hermitian_square": hermitian_square,
        "factorized": factorized,
    }


def semi_explicit_diagonalization(
    model: SymbolicSeesawTypeI,
    hermitian_square: Optional[bool] = None,
    column: int = -1,
) -> Dict[str, object]:
    r"""
    Mixing matrix in terms of the parameters *and* the physical masses.

    The masses ``m_i`` (``model.massive_masses``) are kept as symbols subject
    to the constraints ``p(lambda_i) = 0``, where ``p`` is the reduced
    characteristic polynomial and ``lambda_i = -m_i`` (light) or ``+m_i``
    (heavy) in the real case, ``lambda_i = m_i^2`` in the complex case. No
    polynomial is solved, so this works for any ``(n, m)``, including those
    with no solution in radicals.

    Parameters:
    -----------
    model : SymbolicSeesawTypeI
        Symbolic model
    hermitian_square : bool, optional
        Force the ``M^\dagger M`` route (default: only if ``m_D`` is not real)
    column : int, optional
        Adjugate column used for the eigenvectors

    Returns:
    --------
    dict
        ``'polynomial'``, ``'eigenvalues'`` (in terms of the mass symbols),
        ``'masses'`` (mass symbols of the massive states), ``'constraints'``
        (expressions that vanish, one per massive state), ``'U'``
    """
    if hermitian_square is None:
        hermitian_square = model.dirac != "real"
    lamb = sp.Symbol("lambda", real=True)
    polynomial = model.reduced_charpoly(hermitian_square=hermitian_square, lamb=lamb)
    masses = list(model.massive_masses)
    if hermitian_square:
        eigenvalues = [mass**2 for mass in masses]
    else:
        n_light = min(model.n_gen, model.n_sterile)
        eigenvalues = [-mass for mass in masses[:n_light]] + masses[n_light:]

    null_vectors = massless_eigenvectors(model.m_D_sym) if model.n_massless else []
    U = mixing_matrix_symbolic(
        model.full_mass_matrix_symbolic(),
        eigenvalues,
        masses,
        hermitian_square=hermitian_square,
        null_vectors=null_vectors,
        column=column,
    )
    return {
        "polynomial": polynomial,
        "eigenvalues": eigenvalues,
        "masses": masses,
        "constraints": [
            polynomial.as_expr().subs(lamb, value) for value in eigenvalues
        ],
        "U": U,
        "hermitian_square": hermitian_square,
    }


# ---------------------------------------------------------------------------
# Inverse problem: parameters from masses and mixing
# ---------------------------------------------------------------------------


def charpoly_matching_equations(
    model: SymbolicSeesawTypeI,
    hermitian_square: Optional[bool] = None,
    substitutions: Optional[dict] = None,
) -> List[sp.Expr]:
    r"""
    Invariant relations between Lagrangian parameters and physical masses.

    Equates the coefficients of the reduced characteristic polynomial with
    those of ``prod_i (lambda - lambda_i)``, where ``lambda_i = -m_i`` (light)
    or ``+m_i`` (heavy) in the real case and ``lambda_i = m_i^2`` in the
    complex case. There are ``k = m + min(n, m)`` independent relations.

    Parameters:
    -----------
    model : SymbolicSeesawTypeI
        Symbolic model
    hermitian_square : bool, optional
        Use ``M^\dagger M`` (default: only if ``m_D`` is not real)
    substitutions : dict, optional
        Substitutions applied to the mass matrix first

    Returns:
    --------
    list of sp.Expr
        Expressions that must vanish, ordered from the ``lambda^(k-1)``
        coefficient (trace) down to the constant term (determinant)
    """
    if hermitian_square is None:
        hermitian_square = model.dirac != "real"
    lamb = sp.Symbol("lambda", real=True)
    polynomial = model.reduced_charpoly(
        hermitian_square=hermitian_square, lamb=lamb, substitutions=substitutions
    )
    masses = list(model.massive_masses)
    if hermitian_square:
        eigenvalues = [mass**2 for mass in masses]
    else:
        n_light = min(model.n_gen, model.n_sterile)
        eigenvalues = [-mass for mass in masses[:n_light]] + masses[n_light:]
    target = sp.Poly(sp.Mul(*[lamb - value for value in eigenvalues]), lamb)
    return [
        sp.expand(c_full - c_diag)
        for c_full, c_diag in zip(polynomial.all_coeffs()[1:], target.all_coeffs()[1:])
    ]


def arrowhead_inverse_parameters(
    eigenvalues: Sequence[sp.Expr], majorana_masses: Sequence[sp.Expr]
) -> Dict[str, object]:
    r"""
    Closed-form inverse problem for one active and ``m`` sterile neutrinos.

    With diagonal ``M_R = diag(M_1..M_m)`` and real ``m_D = (a_1..a_m)`` the
    ``(1, m)`` seesaw matrix is an arrowhead matrix with characteristic
    polynomial

    .. math::
        \lambda \prod_i (\lambda - M_i) - \sum_i a_i^2 \prod_{j \ne i}(\lambda - M_j)
        = \prod_{j=0}^{m} (\lambda - \lambda_j).

    Evaluating at ``lambda = M_i`` gives, for any ``m``,

    .. math::
        a_i^2 = -\frac{\prod_{j}(M_i - \lambda_j)}{\prod_{j \ne i}(M_i - M_j)},
        \qquad \sum_j \lambda_j = \sum_i M_i .

    Positivity of ``a_i^2`` requires the interlacing
    ``lambda_0 < M_1 < lambda_1 < ... < M_m < lambda_m`` with ``lambda_0 < 0``.

    Parameters:
    -----------
    eigenvalues : sequence of sp.Expr
        The ``m + 1`` signed eigenvalues, e.g. ``(-m_1, m_2, ..., m_{m+1})``
    majorana_masses : sequence of sp.Expr
        ``(M_1, ..., M_m)``, all different

    Returns:
    --------
    dict
        ``'m_D_squared'`` (list of ``a_i^2``) and ``'trace_constraint'``
        (expression that must vanish)
    """
    eigenvalues = [sp.sympify(x) for x in eigenvalues]
    majorana_masses = [sp.sympify(x) for x in majorana_masses]
    if len(eigenvalues) != len(majorana_masses) + 1:
        raise ValueError("Need m + 1 eigenvalues for m Majorana masses")
    squared = []
    for i, M_i in enumerate(majorana_masses):
        numerator = sp.Mul(*[M_i - value for value in eigenvalues])
        denominator = sp.Mul(
            *[M_i - M_j for j, M_j in enumerate(majorana_masses) if j != i]
        )
        squared.append(-numerator / denominator)
    return {
        "m_D_squared": squared,
        "trace_constraint": sp.Add(*eigenvalues) - sp.Add(*majorana_masses),
    }


def arrowhead_mixing_matrix(
    eigenvalues: Sequence[sp.Expr],
    majorana_masses: Sequence[sp.Expr],
    dirac_masses: Sequence[sp.Expr],
) -> sp.Matrix:
    r"""
    Mixing matrix of the real ``(1, m)`` seesaw (arrowhead) matrix.

    The eigenvector for ``lambda_j`` is proportional to
    ``(1, a_1/(lambda_j - M_1), ..., a_m/(lambda_j - M_m))``; negative
    eigenvalues get a Majorana phase ``i``.

    Parameters:
    -----------
    eigenvalues : sequence of sp.Expr
        The ``m + 1`` signed eigenvalues; the first one must be the negative
        (light) one
    majorana_masses : sequence of sp.Expr
        ``(M_1, ..., M_m)``
    dirac_masses : sequence of sp.Expr
        ``(a_1, ..., a_m)``

    Returns:
    --------
    sp.Matrix
        ``U`` with ``U^T M U = diag(-lambda_0, lambda_1, ..., lambda_m)``
    """
    columns = []
    for index, value in enumerate(eigenvalues):
        vector = sp.Matrix(
            [sp.S.One]
            + [a / (value - M) for a, M in zip(dirac_masses, majorana_masses)]
        )
        vector = vector / sp.sqrt(sum((x**2 for x in vector), sp.S.Zero))
        columns.append(vector * (sp.I if index == 0 else sp.S.One))
    return sp.Matrix.hstack(*columns)


def orthogonal_matrix(dim: int, angles: Sequence[sp.Expr]) -> sp.Matrix:
    r"""
    Orthogonal matrix as a product of plane rotations.

    ``O = prod_{i<j} R_{ij}(angle_ij)`` over all ``dim (dim - 1) / 2`` planes
    in lexicographic order. Real angles give a real rotation; complex angles
    give a complex orthogonal matrix (``O^T O = 1``), as needed in the
    Casas-Ibarra parametrization.

    Parameters:
    -----------
    dim : int
        Dimension
    angles : sequence of sp.Expr
        ``dim (dim - 1) / 2`` angles

    Returns:
    --------
    sp.Matrix
        Orthogonal matrix
    """
    planes = list(combinations(range(dim), 2))
    if len(angles) != len(planes):
        raise ValueError(f"Expected {len(planes)} angles for dimension {dim}")
    result = sp.eye(dim)
    for (i, j), angle in zip(planes, angles):
        rotation = sp.eye(dim)
        rotation[i, i] = sp.cos(angle)
        rotation[j, j] = sp.cos(angle)
        rotation[i, j] = -sp.sin(angle)
        rotation[j, i] = sp.sin(angle)
        result = result * rotation
    return result


def _faddeev_leverrier(matrix: sp.Matrix) -> List[sp.Expr]:
    """
    Coefficients of ``det(mu - A)`` (leading first) by the Faddeev-LeVerrier
    recursion, using only matrix products and traces.

    Unlike ``Matrix.charpoly`` nothing is expanded or simplified, so the
    coefficients of a matrix with large entries stay compact nested
    expressions.
    """
    dim = matrix.rows
    identity = sp.eye(dim)
    coefficients: List[sp.Expr] = [sp.S.One]
    auxiliary = identity
    for k in range(1, dim + 1):
        product = matrix * auxiliary
        coefficient = -product.trace() / k
        coefficients.append(coefficient)
        auxiliary = product + coefficient * identity
    return coefficients


def _roots_from_coefficients(coefficients: Sequence[sp.Expr]) -> List[sp.Expr]:
    """Real roots (ascending) of a monic polynomial of degree <= 4 given its
    non-leading coefficients, without building a ``sp.Poly``."""
    degree = len(coefficients)
    if degree == 1:
        return [-coefficients[0]]
    if degree == 2:
        b, c = coefficients
        disc = sp.sqrt(b**2 - 4 * c)
        return [(-b - disc) / 2, (-b + disc) / 2]
    if degree == 3:
        return solve_cubic_trigonometric(coefficients)
    if degree == 4:
        return solve_quartic_ferrari(coefficients)
    raise NotSolvableInRadicals(degree)


def hermitian_matrix_function(
    matrix: sp.Matrix,
    function: Callable[[sp.Expr], sp.Expr],
    simplify: Optional[Callable] = None,
) -> sp.Matrix:
    r"""
    Closed-form function ``f(A)`` of a small Hermitian matrix.

    Uses Sylvester's formula ``f(A) = sum_i f(mu_i) P_i`` with the Frobenius
    covariants ``P_i = prod_{j != i} (A - mu_j) / (mu_i - mu_j)``, the
    eigenvalues ``mu_i`` being obtained in closed form (dimension ``<= 4``).
    Assumes non-degenerate eigenvalues.

    Parameters:
    -----------
    matrix : sp.Matrix
        Hermitian matrix ``A`` (dimension 1 to 4)
    function : callable
        Scalar function ``f``
    simplify : callable, optional
        Applied to the eigenvalues

    Returns:
    --------
    sp.Matrix
        ``f(A)``

    Raises:
    -------
    NotSolvableInRadicals
        For dimension ``>= 5``
    """
    dim = matrix.rows
    if dim == 1:
        return sp.Matrix([[function(matrix[0, 0])]])
    if matrix.is_diagonal():
        # Decoupled states: no eigenvalue problem to solve
        return sp.diag(*[function(matrix[i, i]) for i in range(dim)])
    eigenvalues = _roots_from_coefficients(_faddeev_leverrier(matrix)[1:])
    if simplify is not None:
        eigenvalues = [simplify(value) for value in eigenvalues]
    identity = sp.eye(dim)
    result = sp.zeros(dim, dim)
    for i, mu_i in enumerate(eigenvalues):
        projector = identity
        for j, mu_j in enumerate(eigenvalues):
            if j != i:
                projector = projector * (matrix - mu_j * identity) / (mu_i - mu_j)
        result += function(mu_i) * projector
    return result


def exact_seesaw_parametrization(
    light_masses: Sequence[sp.Expr],
    heavy_masses: Sequence[sp.Expr],
    orthogonal: Optional[sp.Matrix] = None,
    real: bool = False,
    simplify: Optional[Callable] = None,
) -> Dict[str, object]:
    r"""
    Exact inverse problem: ``m_D`` and ``M_R`` from masses and mixing parameters.

    Any unitary matrix with invertible active-light block can be written as

    .. math::
        W = \begin{pmatrix} c_l & X^\dagger c_h \\ -X c_l & c_h \end{pmatrix},
        \qquad c_l = (1 + X^\dagger X)^{-1/2},\quad c_h = (1 + X X^\dagger)^{-1/2},

    with ``X`` an ``m x n`` matrix. Requiring the active-active block of
    ``M = W^* diag(d, D) W^\dagger`` to vanish gives ``d + X^T D X = 0``,
    solved exactly by

    .. math:: X = i\, D^{-1/2}\, O\, d^{1/2}, \qquad O^T O = 1,

    which is the Casas-Ibarra parametrization to all orders in ``m_D/M_R``
    (cf. Agostinho et al., arXiv:1711.06229). Then

    .. math::
        m_D = c_l^*\, X^T D\, c_h^{-1}, \qquad
        M_R = X^* c_l^*\, d\, c_l X^\dagger + c_h^*\, D\, c_h ,

    and ``W^T M W = diag(d, D)`` exactly. ``M_R`` comes out in a general
    (non-diagonal) sterile basis. The only non-rational operation is the
    square root of a ``min(n, m)``-dimensional Hermitian matrix, which is
    closed-form for ``min(n, m) <= 4``; the masses themselves are inputs, so
    the degree-``k`` polynomial never has to be solved.

    The most general solution is obtained by a further active rotation
    ``m_D -> U_a^* m_D`` and sterile rotation ``M_R -> U_s^* M_R U_s^\dagger``.

    Parameters:
    -----------
    light_masses : sequence of sp.Expr
        The ``n`` light masses ``d``; for ``n > m`` the first ``n - m`` must
        be zero (massless states)
    heavy_masses : sequence of sp.Expr
        The ``m`` heavy masses ``D`` (positive)
    orthogonal : sp.Matrix, optional
        ``m x n`` matrix ``O`` with ``O^T O = 1`` on the massive light states
        (see :func:`orthogonal_matrix`). Default: ``O_ij = 1`` when light
        massive state ``j`` is paired with heavy state ``i``, else 0.
    real : bool, optional
        If True, rephase the active fields by ``i`` so that ``m_D`` is real
        when ``O`` is real (CP-conserving case); the light columns of ``W``
        then carry the Majorana phase ``i``.
    simplify : callable, optional
        Applied element-wise to intermediate matrices

    Returns:
    --------
    dict
        ``'X'``, ``'m_D'``, ``'M_R'``, ``'M'`` (full matrix), ``'U'`` (mixing
        matrix with ``U^T M U = diag(light_masses, heavy_masses)``)
    """
    d = [sp.sympify(x) for x in light_masses]
    D = [sp.sympify(x) for x in heavy_masses]
    n, m = len(d), len(D)
    n_zero = max(n - m, 0)
    if any(x != 0 for x in d[:n_zero]):
        raise ValueError(f"For n > m the first {n_zero} light masses must be zero")

    if orthogonal is None:
        orthogonal = sp.zeros(m, n)
        for j in range(n_zero, n):
            orthogonal[j - n_zero, j] = 1
    if orthogonal.shape != (m, n):
        raise ValueError(f"orthogonal must have shape ({m}, {n})")

    def apply(matrix: sp.Matrix) -> sp.Matrix:
        return matrix.applyfunc(simplify) if simplify else matrix

    def conj(matrix: sp.Matrix) -> sp.Matrix:
        return matrix.applyfunc(sp.conjugate)

    d_matrix, D_matrix = sp.diag(*d), sp.diag(*D)
    X = apply(
        sp.I
        * sp.diag(*[1 / sp.sqrt(x) for x in D])
        * orthogonal
        * sp.diag(*[sp.sqrt(x) for x in d])
    )
    X_conj = apply(conj(X))  # cheap here; never conjugate the large results

    # Functions of 1 + X^dagger X and 1 + X X^dagger from the Gram matrix B of
    # the massive light states: g(1 + X X^dagger) = 1 + X [(g(1+B) - 1) B^-1] X^dagger.
    def inv_sqrt(mu: sp.Expr) -> sp.Expr:
        return (1 + mu) ** sp.Rational(-1, 2)

    def inv_sqrt_minus_one(mu: sp.Expr) -> sp.Expr:
        return ((1 + mu) ** sp.Rational(-1, 2) - 1) / mu

    def sqrt_minus_one(mu: sp.Expr) -> sp.Expr:
        return (sp.sqrt(1 + mu) - 1) / mu

    def gram_functions(Y: sp.Matrix, Y_conj: sp.Matrix):
        """c_l, c_h and c_h^-1 for the matrix Y (with Y^dagger = Y_conj^T)."""
        Y_massive, Y_massive_dagger = Y[:, n_zero:], Y_conj[:, n_zero:].T
        gram = apply(Y_massive_dagger * Y_massive)
        c_l_massive = hermitian_matrix_function(gram, inv_sqrt, simplify)
        c_l = sp.diag(sp.eye(n_zero), c_l_massive) if n_zero else c_l_massive
        c_h = (
            sp.eye(m)
            + Y_massive
            * hermitian_matrix_function(gram, inv_sqrt_minus_one, simplify)
            * Y_massive_dagger
        )
        c_h_inverse = (
            sp.eye(m)
            + Y_massive
            * hermitian_matrix_function(gram, sqrt_minus_one, simplify)
            * Y_massive_dagger
        )
        return gram, apply(c_l), apply(c_h), apply(c_h_inverse)

    _, c_l, c_h, c_h_inverse = gram_functions(X, X_conj)
    # Complex conjugates are obtained by rebuilding with X -> X^*. For real O,
    # X is purely imaginary, X X^dagger and X^dagger X are real, and nothing
    # changes. (Testing the Gram matrix instead would be wrong: for a single
    # light state it is a real number even when X is complex.)
    if X_conj == -X or X_conj == X:
        c_l_conj, c_h_conj = c_l, c_h
    else:
        _, c_l_conj, c_h_conj, _ = gram_functions(X_conj, X)

    X_dagger = X_conj.T
    m_D = c_l_conj * X.T * D_matrix * c_h_inverse
    M_R = X_conj * c_l_conj * d_matrix * c_l * X_dagger + c_h_conj * D_matrix * c_h
    U = sp.Matrix.vstack(
        sp.Matrix.hstack(c_l, X_dagger * c_h), sp.Matrix.hstack(-X * c_l, c_h)
    )
    if real:
        # U_a = i: m_D -> U_a^* m_D = -i m_D, active rows of U -> i * rows
        m_D = -sp.I * m_D
        U = sp.diag(sp.I * sp.eye(n), sp.eye(m)) * U
    m_D, M_R, U = apply(m_D), apply(M_R), apply(U)

    M = sp.Matrix.vstack(
        sp.Matrix.hstack(sp.zeros(n, n), m_D), sp.Matrix.hstack(m_D.T, M_R)
    )
    return {"X": X, "m_D": m_D, "M_R": M_R, "M": M, "U": U}


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------


def _dag_cse(
    expressions: Sequence[sp.Expr],
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], List[sp.Expr]]:
    """
    Common-subexpression elimination that is linear in the number of distinct
    subexpressions.

    The closed forms produced here are deeply nested: the complex ``(2,2)``
    mixing matrix has ~6e7 nodes written out but only a few hundred distinct
    subexpressions. ``sympy.cse`` walks the written-out tree and never
    finishes; this routine assigns one temporary to each distinct
    non-atomic subexpression instead.
    """
    mapping: Dict[sp.Basic, sp.Basic] = {}
    replacements: List[Tuple[sp.Symbol, sp.Expr]] = []
    for root in expressions:
        stack = [(sp.sympify(root), False)]
        while stack:
            node, expanded = stack.pop()
            if node in mapping:
                continue
            if not node.args:
                mapping[node] = node
            elif expanded:
                temporary = sp.Symbol(f"_t{len(replacements)}")
                replacements.append(
                    (temporary, node.func(*[mapping[arg] for arg in node.args]))
                )
                mapping[node] = temporary
            else:
                stack.append((node, True))
                stack.extend((arg, False) for arg in node.args if arg not in mapping)
    return replacements, [mapping[sp.sympify(root)] for root in expressions]


def _dag_free_symbols(expressions: Sequence[sp.Expr]) -> set:
    """Free symbols, visiting each distinct subexpression once."""
    seen: set = set()
    symbols: set = set()
    stack = [sp.sympify(root) for root in expressions]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.is_Symbol:
            symbols.add(node)
        stack.extend(node.args)
    return symbols


def lambdify_large(
    symbols: Sequence[sp.Symbol],
    expressions: Union[sp.MatrixBase, Sequence[sp.Expr]],
    modules: str = "numpy",
) -> Callable:
    """
    Numerical function for large nested closed-form expressions.

    Equivalent to ``sp.lambdify(symbols, expressions, 'numpy')`` but compiles
    the expression as a straight-line program over its distinct
    subexpressions (see :func:`expression_size`), which stays fast when the
    written-out formula has millions of nodes. Pass complex arguments
    (``complex(x)``) if square roots of negative numbers can occur.

    Parameters:
    -----------
    symbols : sequence of sp.Symbol
        Arguments of the generated function
    expressions : sp.Matrix or sequence of sp.Expr
        Expressions to evaluate
    modules : {'numpy', 'mpmath'}, optional
        Numerical backend. With ``'mpmath'`` the function works at the
        current ``mpmath`` precision and returns an ``mpmath.matrix``.

    Returns:
    --------
    callable
        Function returning a ``numpy`` array (or ``mpmath.matrix``) with the
        shape of ``expressions``
    """
    if isinstance(expressions, sp.MatrixBase):
        shape: Tuple[int, ...] = expressions.shape
    else:
        shape = (len(expressions),)
    # Eliminate subexpressions first: lambdify itself walks the written-out tree
    replacements, reduced = _dag_cse(list(expressions))
    flat = sp.lambdify(
        list(symbols),
        reduced,
        modules=modules,
        cse=lambda outputs: (replacements, outputs),
    )
    rows, cols = shape if len(shape) == 2 else (shape[0], 1)

    def evaluate(*arguments):
        values = flat(*arguments)
        if modules == "mpmath":
            return mpmath.matrix(
                [[values[i * cols + j] for j in range(cols)] for i in range(rows)]
            )
        return np.array(values).reshape(shape)

    return evaluate


def _sample_point(
    symbols: Sequence[sp.Symbol],
    rng: np.random.Generator,
    ranges: Optional[Dict[sp.Symbol, Tuple[float, float]]],
) -> Dict[sp.Symbol, complex]:
    point = {}
    for symbol in symbols:
        if ranges and symbol in ranges:
            low, high = ranges[symbol]
            point[symbol] = rng.uniform(low, high)
        elif symbol.is_positive:
            point[symbol] = rng.uniform(0.5, 2.0)
        elif symbol.is_real:
            point[symbol] = rng.uniform(-1.0, 1.0)
        else:
            point[symbol] = rng.uniform(-1.0, 1.0) + 1j * rng.uniform(-1.0, 1.0)
    return point


def verify_takagi(
    mixing_matrix: sp.Matrix,
    mass_matrix: sp.Matrix,
    masses: Sequence[sp.Expr],
    n_points: int = 5,
    seed: int = 0,
    ranges: Optional[Dict[sp.Symbol, Tuple[float, float]]] = None,
    points: Optional[Sequence[Dict[sp.Symbol, complex]]] = None,
    tolerance: float = 1e-8,
    precision: Optional[int] = None,
) -> Dict[str, object]:
    r"""
    Check a symbolic Takagi factorization at numerical parameter points.

    For each point evaluates ``U``, ``M`` and the masses and tests

    * ``U^T M U = diag(masses)``,
    * ``U^\dagger U = 1``,
    * that the masses are real and non-negative and agree with an independent
      numerical factorization (:func:`takagi_factorization`).

    Parameters:
    -----------
    mixing_matrix : sp.Matrix
        Symbolic ``U``
    mass_matrix : sp.Matrix
        Symbolic ``M``
    masses : sequence of sp.Expr
        Symbolic masses, one per column of ``U`` (zeros for massless states)
    n_points : int, optional
        Number of random points (ignored if ``points`` is given)
    seed : int, optional
        Seed of the random generator
    ranges : dict, optional
        ``{symbol: (low, high)}`` overriding the default sampling, which is
        (0.5, 2) for positive symbols, (-1, 1) for real symbols and the
        complex unit square otherwise
    points : sequence of dict, optional
        Explicit ``{symbol: value}`` points; needed when the symbols must obey
        constraints (orderings, interlacing, characteristic equations)
    tolerance : float, optional
        Maximum allowed error relative to the largest entry of ``M``
    precision : int, optional
        Number of decimal digits for the evaluation. By default the check is
        done in double precision and, only if it fails, repeated with 50
        digits: large nested radicals (quartic cases) lose many digits to
        cancellations, which must not be mistaken for a wrong formula.

    Returns:
    --------
    dict
        ``'passed'`` (bool), ``'diagonalization_error'``, ``'unitarity_error'``,
        ``'mass_error'`` (maxima over the points), ``'n_points'`` and
        ``'precision'`` (``'double'`` or the number of digits used). The
        reference masses always come from a double-precision factorization.
    """
    mass_vector = sp.Matrix([sp.sympify(x) for x in masses])
    expressions = (mixing_matrix, mass_matrix, mass_vector)
    symbols = sorted(
        _dag_free_symbols(list(mixing_matrix) + list(mass_matrix) + list(mass_vector)),
        key=lambda s: s.name,
    )
    if points is None:
        rng = np.random.default_rng(seed)
        points = [_sample_point(symbols, rng, ranges) for _ in range(n_points)]

    def run(modules: str) -> Dict[str, float]:
        functions = [
            lambdify_large(symbols, expression, modules) for expression in expressions
        ]
        errors = {
            "diagonalization_error": 0.0,
            "unitarity_error": 0.0,
            "mass_error": 0.0,
        }
        for point in points:
            if modules == "mpmath":
                arguments = [
                    mpmath.mpmathify(complex(point[symbol])) for symbol in symbols
                ]
                U, M, m = [function(*arguments) for function in functions]
                dim = M.rows
                M_double = np.array(
                    [[complex(M[i, j]) for j in range(dim)] for i in range(dim)]
                )
                m_double = np.array([complex(m[i]) for i in range(dim)])
                diagonal = mpmath.diag([m[i] for i in range(dim)])
                diagonalization = max(abs(x) for x in U.T * M * U - diagonal)
                unitarity = max(abs(x) for x in U.H * U - mpmath.eye(dim))
            else:
                arguments = [
                    np.complex128(complex(point[symbol])) for symbol in symbols
                ]
                with np.errstate(all="ignore"):
                    U, M_double, m_double = [
                        np.array(function(*arguments), dtype=complex)
                        for function in functions
                    ]
                m_double = m_double.flatten()
                diagonalization = np.max(np.abs(U.T @ M_double @ U - np.diag(m_double)))
                unitarity = np.max(np.abs(U.conj().T @ U - np.eye(len(m_double))))
            scale = max(np.max(np.abs(M_double)), 1e-300)
            reference, _ = takagi_factorization(M_double)
            mass_error = max(
                np.max(np.abs(np.sort(m_double.real) - reference)),
                np.max(np.abs(m_double.imag)),
                np.max(np.clip(-m_double.real, 0, None)),
            )
            current = {
                "diagonalization_error": float(diagonalization / scale),
                "unitarity_error": float(unitarity),
                "mass_error": float(mass_error / scale),
            }
            for key, value in current.items():
                value = value if np.isfinite(value) else float("inf")
                errors[key] = max(errors[key], value)
        return errors

    if precision is None:
        errors, used = run("numpy"), "double"
        if not all(value < tolerance for value in errors.values()):
            # Nested radicals lose digits in double precision: before reporting
            # a failure, repeat with extended precision.
            with mpmath.workdps(50):
                errors, used = run("mpmath"), 50
    else:
        with mpmath.workdps(precision):
            errors, used = run("mpmath"), precision

    return {
        "passed": all(value < tolerance for value in errors.values()),
        **errors,
        "n_points": len(points),
        "precision": used,
    }


def expression_size(expression: Union[sp.Expr, sp.MatrixBase]) -> Dict[str, int]:
    """
    Size of an expression written out in full and as a straight-line program.

    ``'tree'`` counts the nodes of the expression tree, i.e. the size of the
    formula written out explicitly (comparable to ``sp.count_ops``, but
    computed with memoization so that very large nested results can be
    measured). ``'dag'`` counts the distinct subexpressions, i.e. the size
    when repeated subexpressions are stored once; it measures the cost of
    evaluating the formula.

    Parameters:
    -----------
    expression : sp.Expr or sp.Matrix
        Expression or matrix of expressions

    Returns:
    --------
    dict
        ``{'tree': int, 'dag': int}``
    """
    roots = (
        list(expression)
        if isinstance(expression, sp.MatrixBase)
        else [sp.sympify(expression)]
    )
    sizes: Dict[sp.Basic, int] = {}
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in sizes:
                continue
            if expanded:
                sizes[node] = 1 + sum(sizes[arg] for arg in node.args)
            else:
                stack.append((node, True))
                stack.extend((arg, False) for arg in node.args if arg not in sizes)
    return {"tree": sum(sizes[root] for root in roots), "dag": len(sizes)}


# ---------------------------------------------------------------------------
# Feasibility probe
# ---------------------------------------------------------------------------


def _probe_model(n: int, m: int, cp: str) -> SymbolicSeesawTypeI:
    return SymbolicSeesawTypeI(
        n, m, dirac="real" if cp == "real" else "polar", majorana_basis="diagonal"
    )


def numeric_mass_points(
    model: SymbolicSeesawTypeI, n_points: int = 3, seed: int = 0
) -> List[Dict[sp.Symbol, complex]]:
    """
    Random parameter points completed with the numerically computed masses.

    Needed to check semi-explicit results, where the mass symbols are not
    independent of the parameters: each point assigns random values to the
    entries of ``m_D`` and ``M_R`` (``model`` must use
    ``majorana_basis='diagonal'``; the Majorana masses are drawn heavy and
    well separated) and sets ``model.massive_masses`` to the masses obtained
    from a numerical Takagi factorization.

    Parameters:
    -----------
    model : SymbolicSeesawTypeI
        Symbolic model with diagonal ``M_R``
    n_points : int, optional
        Number of points
    seed : int, optional
        Seed of the random generator

    Returns:
    --------
    list of dict
        ``{symbol: value}`` points, usable as ``points`` in :func:`verify_takagi`
    """
    mass_matrix = model.full_mass_matrix_symbolic()
    parameters = sorted(mass_matrix.free_symbols, key=lambda s: s.name)
    evaluate = sp.lambdify(parameters, mass_matrix, modules="numpy")
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n_points):
        point = _sample_point(parameters, rng, None)
        # Seesaw-like hierarchy keeps light (negative) and heavy states apart
        for i in range(model.n_sterile):
            point[model.M_R_sym[i, i]] = 4.0 + 3.0 * i + rng.uniform(0.0, 1.0)
        M = np.array(evaluate(*[complex(point[s]) for s in parameters]), dtype=complex)
        numeric_masses, _ = takagi_factorization(M)
        for symbol, value in zip(
            model.massive_masses, numeric_masses[model.n_massless :]
        ):
            point[symbol] = value
        points.append(point)
    return points


def _size_record(expression: sp.MatrixBase) -> Dict[str, int]:
    size = expression_size(expression)
    return {"size": size["tree"], "dag_size": size["dag"]}


def _run_probe(n: int, m: int, cp: str, method: str) -> Dict[str, object]:
    """Carry out one feasibility probe (executed in a subprocess)."""
    model = _probe_model(n, m, cp)
    mass_matrix = model.full_mass_matrix_symbolic()
    zeros = [sp.S.Zero] * model.n_massless
    record: Dict[str, object] = {}

    if method == "direct":
        # Generic parameters: a degree >= 5 polynomial does not factorize, and
        # attempting it symbolically is slow.
        result = direct_diagonalization(model, try_factor=model.n_massive <= 4)
        check = verify_takagi(
            result["U"],
            mass_matrix,
            zeros + result["masses"],
            n_points=3,
            ranges={model.M_R_sym[i, i]: (4.0 + 3 * i, 5.0 + 3 * i) for i in range(m)},
        )
        record.update(**_size_record(result["U"]), verified=check["passed"])
    elif method == "semi_explicit":
        result = semi_explicit_diagonalization(model)
        points = numeric_mass_points(model, 3, 0)
        check = verify_takagi(
            result["U"], mass_matrix, zeros + result["masses"], points=points
        )
        record.update(**_size_record(result["U"]), verified=check["passed"])
    elif method == "inverse":
        n_massive_light = min(n, m)
        light = sp.symbols(f"m_1:{n_massive_light + 1}", positive=True, seq=True)
        heavy = sp.symbols(f"M_1:{m + 1}", positive=True, seq=True)
        n_angles = m * (m - 1) // 2
        angles = sp.symbols(f"theta_1:{n_angles + 1}", real=(cp == "real"), seq=True)
        square = orthogonal_matrix(m, angles) if m > 1 else sp.eye(1)
        orthogonal = sp.zeros(m, n)
        orthogonal[:, n - n_massive_light :] = square[:, :n_massive_light]
        result = exact_seesaw_parametrization(
            zeros + list(light), heavy, orthogonal, real=(cp == "real")
        )
        ranges = {
            symbol: (0.05 + 0.1 * i, 0.1 + 0.1 * i) for i, symbol in enumerate(light)
        }
        ranges.update({symbol: (2.0 + i, 2.5 + i) for i, symbol in enumerate(heavy)})
        check = verify_takagi(
            result["U"],
            result["M"],
            zeros + list(light) + list(heavy),
            n_points=3,
            ranges=ranges,
        )
        record.update(**_size_record(result["M"]), verified=check["passed"])
    elif method == "expansion":
        result = seesaw_block_expansion(model.m_D_sym, model.M_R_sym, order=4)
        record.update(**_size_record(result["m_light"]), verified=None)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return record


def _probe_worker(queue, n: int, m: int, cp: str, method: str) -> None:
    start = time.perf_counter()
    try:
        record = _run_probe(n, m, cp, method)
        record["status"] = "ok"
    except NotSolvableInRadicals as error:
        record = {"status": f"no closed form (degree {error.degree})"}
    except Exception as error:  # report, do not crash the survey
        record = {"status": f"error: {type(error).__name__}: {error}"}
    record["seconds"] = time.perf_counter() - start
    queue.put(record)


def feasibility_probe(
    n: int, m: int, cp: str = "real", method: str = "direct", timeout: float = 60.0
) -> Dict[str, object]:
    r"""
    Attempt one symbolic diagonalization and report whether it is feasible.

    The computation runs in a subprocess that is terminated after ``timeout``
    seconds, so that intractable cases are recorded instead of hanging.

    Parameters:
    -----------
    n, m : int
        Number of active and sterile neutrinos
    cp : {'real', 'complex'}, optional
        Real (CP-conserving) parameters, or complex ``m_D`` in polar form
        with the unphysical phases removed
    method : {'direct', 'semi_explicit', 'inverse', 'expansion'}, optional
        ``'direct'``: :func:`direct_diagonalization`;
        ``'semi_explicit'``: :func:`semi_explicit_diagonalization`;
        ``'inverse'``: :func:`exact_seesaw_parametrization` with symbolic
        masses and rotation angles (complex angles for ``cp='complex'``);
        ``'expansion'``: :func:`seesaw_block_expansion` to order 4
    timeout : float, optional
        Time limit in seconds

    Returns:
    --------
    dict
        ``n``, ``m``, ``cp``, ``method``, ``n_massless``, ``degree`` (number
        of massive states ``k``), ``solvable_in_radicals`` (``k <= 4``),
        ``status`` (``'ok'``, ``'timeout'``, ``'no closed form ...'`` or
        ``'error: ...'``), ``seconds``, ``size`` and ``dag_size`` of the result
        (see :func:`expression_size`: the mixing matrix for ``'direct'`` and
        ``'semi_explicit'``, the full mass matrix for ``'inverse'``, the light
        mass matrix for ``'expansion'``) and ``verified`` (numerical check,
        None if not applicable)
    """
    if cp not in ("real", "complex"):
        raise ValueError("cp must be 'real' or 'complex'")
    degree = m + min(n, m)
    record: Dict[str, object] = {
        "n": n,
        "m": m,
        "cp": cp,
        "method": method,
        "n_massless": max(n - m, 0),
        "degree": degree,
        "solvable_in_radicals": degree <= 4,
        "status": "timeout",
        "seconds": float(timeout),
        "size": None,
        "dag_size": None,
        "verified": None,
    }

    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    queue = context.Queue()
    process = context.Process(target=_probe_worker, args=(queue, n, m, cp, method))
    process.start()
    try:
        record.update(queue.get(timeout=timeout))
    except Exception:  # queue.Empty: timed out
        pass
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
    return record
