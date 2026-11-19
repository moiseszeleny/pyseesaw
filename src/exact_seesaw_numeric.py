r"""
High-precision numerics for the exact Type I seesaw.

Physical seesaw matrices span many orders of magnitude (light masses of
``10^{-11}`` GeV next to heavy masses up to ``10^{15}`` GeV) and, in the
large-mixing regime of the Casas-Ibarra parametrization, the light masses
arise from cancellations between much larger entries. Double precision then
loses all significant digits, so the routines here work in ``mpmath`` at a
precision chosen from the hierarchy (:func:`required_precision`).

Conventions follow :mod:`src.symbolic_diagonalization`:
``M = [[0, m_D], [m_D^T, M_R]]`` and ``U^T M U = diag(m_i)``, ``m_i >= 0``.

* :func:`takagi_factorization_mp` - Takagi factorization at any precision.
* :func:`casas_ibarra_exact_numeric` - the exact (all-orders) Casas-Ibarra
  parametrization, numerical counterpart of
  :func:`src.symbolic_diagonalization.exact_seesaw_parametrization`, with an
  additional active rotation so that the unitary part of the light mixing is
  a given PMNS matrix.
* :func:`casas_ibarra_leading_order` - the usual leading-order version.
* :func:`light_sector_observables` - masses, mass splittings, mixing angles
  (from the unitary part of the light mixing), non-unitarity and heavy-light
  mixing of a given seesaw matrix.
"""

from typing import Dict, Optional, Sequence

import mpmath
import numpy as np

__all__ = [
    "required_precision",
    "takagi_factorization_mp",
    "hermitian_function_mp",
    "complex_orthogonal_2x2",
    "casas_ibarra_orthogonal",
    "casas_ibarra_exact_numeric",
    "casas_ibarra_leading_order",
    "seesaw_mass_matrix_mp",
    "light_sector_observables",
]


def required_precision(
    light_mass: float, heavy_mass: float, imaginary_angle: float = 0.0, digits: int = 25
) -> int:
    r"""
    Working precision (decimal digits) for a seesaw matrix.

    The Hermitian square ``M^\dagger M`` spans ``(M/m)^2`` and, for a complex
    Casas-Ibarra angle with imaginary part ``y``, the light masses come from
    cancellations of relative size ``e^{-2y}`` in the entries (``e^{-4y}`` in
    ``M^\dagger M``).

    Parameters:
    -----------
    light_mass, heavy_mass : float
        Smallest non-zero light mass and largest heavy mass (same units)
    imaginary_angle : float, optional
        ``|Im z|`` of the Casas-Ibarra angle
    digits : int, optional
        Significant digits wanted in the final result

    Returns:
    --------
    int
        Number of decimal digits for ``mpmath``
    """
    hierarchy = np.log10(heavy_mass / light_mass)
    cancellation = 4 * abs(imaginary_angle) / np.log(10)
    return int(np.ceil(digits + 2 * hierarchy + cancellation))


def _to_mp(matrix) -> mpmath.matrix:
    if isinstance(matrix, mpmath.matrix):
        return matrix
    array = np.asarray(matrix, dtype=complex)
    return mpmath.matrix(
        [[mpmath.mpc(complex(x)) for x in row] for row in array.reshape(array.shape[0], -1)]
    )


def _conj(matrix: mpmath.matrix) -> mpmath.matrix:
    return matrix.apply(mpmath.conj)


def _dagger(matrix: mpmath.matrix) -> mpmath.matrix:
    return _conj(matrix).T


def _diag(values: Sequence) -> mpmath.matrix:
    return mpmath.diag([mpmath.mpmathify(v) for v in values])


def hermitian_function_mp(matrix: mpmath.matrix, function) -> mpmath.matrix:
    r"""``f(H) = Q diag(f(e_i)) Q^\dagger`` for a Hermitian ``H = Q diag(e_i) Q^\dagger``."""
    eigenvalues, vectors = mpmath.eighe(matrix)
    values = [function(mpmath.re(e)) for e in eigenvalues]
    return vectors * _diag(values) * _dagger(vectors)


def takagi_factorization_mp(matrix, tolerance: Optional[mpmath.mpf] = None):
    r"""
    Takagi factorization ``U^T M U = diag(m_i)`` at the current mpmath precision.

    Uses the eigenvectors ``v_i`` of ``M^\dagger M`` and fixes each column's
    phase by the Takagi condition ``u_i^T M u_i = m_i``. Assumes non-degenerate
    non-zero masses (exact degeneracies are not resolved); massless columns
    are left as eigenvectors.

    Parameters:
    -----------
    matrix : array_like or mpmath.matrix
        Complex symmetric matrix
    tolerance : mpmath.mpf, optional
        Masses below ``tolerance * max(m_i)`` are treated as zero (default:
        ``10^{-(dps - 5)}`` relative to the largest squared mass)

    Returns:
    --------
    (list of mpmath.mpf, mpmath.matrix)
        Masses in ascending order and the unitary ``U``
    """
    M = _to_mp(matrix)
    dim = M.rows
    squares, vectors = mpmath.eighe(_dagger(M) * M)
    squares = [mpmath.re(x) for x in squares]
    order = sorted(range(dim), key=lambda i: squares[i])
    largest = max(abs(x) for x in squares)
    if tolerance is None:
        tolerance = mpmath.mpf(10) ** (-(mpmath.mp.dps - 5))
    masses, columns = [], []
    for i in order:
        v = vectors[:, i]
        mass = mpmath.sqrt(squares[i]) if squares[i] > tolerance * largest else mpmath.mpf(0)
        if mass > 0:
            bilinear = (v.T * M * v)[0, 0]
            v = v * mpmath.sqrt(mass / bilinear)
        masses.append(mass)
        columns.append(v)
    U = mpmath.matrix(dim, dim)
    for j, column in enumerate(columns):
        for i in range(dim):
            U[i, j] = column[i]
    return masses, U


def _unitarize(matrix: mpmath.matrix) -> mpmath.matrix:
    r"""
    Nearest unitary matrix ``A (A^\dagger A)^{-1/2}`` at the working precision.

    A PMNS matrix built in double precision is unitary only to ``~1e-16``,
    which would limit the exactness of the parametrization to that level.
    """
    return matrix * hermitian_function_mp(_dagger(matrix) * matrix, lambda e: 1 / mpmath.sqrt(e))


def complex_orthogonal_2x2(z, branch: int = 1) -> mpmath.matrix:
    r"""
    ``R(z) = [[cos z, branch sin z], [-sin z, branch cos z]]``, ``R^T R = 1``.

    Built at the current ``mpmath`` precision: for ``|Im z| = y`` the entries
    grow like ``e^y / 2`` while ``R^T R = 1`` holds through cancellations of
    order ``e^{2y}``, which double precision cannot represent.
    """
    z = mpmath.mpc(z)
    c, s = mpmath.cos(z), mpmath.sin(z)
    return mpmath.matrix([[c, branch * s], [-s, branch * c]])


def casas_ibarra_orthogonal(z, ordering: str = "normal", branch: int = 1) -> mpmath.matrix:
    r"""
    ``2 x 3`` Casas-Ibarra matrix of the ``(3, 2)`` minimal seesaw.

    Normal ordering (``m_1 = 0``): ``O = (0, R(z))``; inverted ordering
    (``m_3 = 0``): ``O = (R(z), 0)``. In both cases ``O^T O`` is the
    projector onto the massive light states. Built at the current ``mpmath``
    precision (see :func:`complex_orthogonal_2x2`); call it inside the same
    ``mpmath.workdps`` block as the parametrization.
    """
    R = complex_orthogonal_2x2(z, branch)
    O = mpmath.matrix(2, 3)
    if ordering == "normal":
        offset = 1
    elif ordering == "inverted":
        offset = 0
    else:
        raise ValueError("ordering must be 'normal' or 'inverted'")
    for i in range(2):
        for j in range(2):
            O[i, offset + j] = R[i, j]
    return O


def casas_ibarra_exact_numeric(
    light_masses: Sequence[float],
    heavy_masses: Sequence[float],
    orthogonal,
    active_rotation=None,
) -> Dict[str, mpmath.matrix]:
    r"""
    Exact Casas-Ibarra parametrization (all orders in ``m_D / M_R``).

    With ``X = i D^{-1/2} O d^{1/2}``, ``c_l = (1 + X^\dagger X)^{-1/2}`` and
    ``c_h = (1 + X X^\dagger)^{-1/2}``,

    .. math::
        m_D = U_a^*\, c_l^*\, X^T D\, c_h^{-1},\qquad
        M_R = X^* c_l^*\, d\, c_l X^\dagger + c_h^*\, D\, c_h ,

    give a seesaw matrix whose Takagi factorization is exactly
    ``U = diag(U_a, 1) [[c_l, X^\dagger c_h], [-X c_l, c_h]]`` with masses
    ``(d, D)``. The light-active block is ``N = U_a c_l = (1 - \eta) U_a`` with
    ``1 - \eta = U_a c_l U_a^\dagger``: the light masses and the unitary part
    of the mixing are *exactly* the inputs, and ``\eta`` is the non-unitarity.
    Computed at the current ``mpmath`` precision.

    Parameters:
    -----------
    light_masses : sequence of float
        ``n`` light masses ``d`` (zeros allowed)
    heavy_masses : sequence of float
        ``m`` heavy masses ``D`` (positive)
    orthogonal : array_like
        ``m x n`` complex matrix ``O`` with ``O^T O`` the projector on the
        massive light states
    active_rotation : array_like, optional
        Unitary ``U_a`` (``n x n``), e.g. the PMNS matrix (default: identity);
        made exactly unitary at the working precision first

    Returns:
    --------
    dict
        ``'m_D'``, ``'M_R'``, ``'M'`` (full matrix), ``'U'``, ``'X'``,
        ``'eta'`` (all ``mpmath.matrix``)
    """
    d = [mpmath.mpf(x) for x in light_masses]
    D = [mpmath.mpf(x) for x in heavy_masses]
    n, m = len(d), len(D)
    O = _to_mp(orthogonal)
    U_a = mpmath.eye(n) if active_rotation is None else _unitarize(_to_mp(active_rotation))
    X = mpmath.mpc(0, 1) * _diag([1 / mpmath.sqrt(x) for x in D]) * O * _diag(
        [mpmath.sqrt(x) for x in d]
    )
    X_dagger = _dagger(X)
    c_l = hermitian_function_mp(mpmath.eye(n) + X_dagger * X, lambda e: 1 / mpmath.sqrt(e))
    c_h = hermitian_function_mp(mpmath.eye(m) + X * X_dagger, lambda e: 1 / mpmath.sqrt(e))
    c_h_inverse = hermitian_function_mp(mpmath.eye(m) + X * X_dagger, mpmath.sqrt)
    m_D = _conj(U_a) * _conj(c_l) * X.T * _diag(D) * c_h_inverse
    M_R = _conj(X) * _conj(c_l) * _diag(d) * c_l * X_dagger + _conj(c_h) * _diag(D) * c_h
    M_R = (M_R + M_R.T) / 2  # symmetric up to rounding
    U_top = U_a * c_l
    U = mpmath.matrix(n + m, n + m)
    blocks = ((0, 0, U_top), (0, n, U_a * X_dagger * c_h), (n, 0, -X * c_l), (n, n, c_h))
    for row, col, block in blocks:
        for i in range(block.rows):
            for j in range(block.cols):
                U[row + i, col + j] = block[i, j]
    eta = mpmath.eye(n) - U_a * c_l * _dagger(U_a)
    return {
        "m_D": m_D,
        "M_R": M_R,
        "M": seesaw_mass_matrix_mp(m_D, M_R),
        "U": U,
        "X": X,
        "eta": eta,
    }


def casas_ibarra_leading_order(
    light_masses: Sequence[float],
    heavy_masses: Sequence[float],
    orthogonal,
    active_rotation=None,
) -> Dict[str, mpmath.matrix]:
    r"""
    Leading-order Casas-Ibarra parametrization.

    ``m_D = i U_a^* d^{1/2} O^T D^{1/2}`` and ``M_R = D``, which reproduces
    ``-m_D M_R^{-1} m_D^T = U_a^* d U_a^\dagger`` (Takagi form of the
    seesaw formula) but not the exact spectrum of the full matrix.
    """
    d = [mpmath.mpf(x) for x in light_masses]
    D = [mpmath.mpf(x) for x in heavy_masses]
    n = len(d)
    O = _to_mp(orthogonal)
    U_a = mpmath.eye(n) if active_rotation is None else _unitarize(_to_mp(active_rotation))
    m_D = mpmath.mpc(0, 1) * _conj(U_a) * _diag([mpmath.sqrt(x) for x in d]) * O.T * _diag(
        [mpmath.sqrt(x) for x in D]
    )
    M_R = _diag(D)
    return {"m_D": m_D, "M_R": M_R, "M": seesaw_mass_matrix_mp(m_D, M_R)}


def seesaw_mass_matrix_mp(m_D: mpmath.matrix, M_R: mpmath.matrix) -> mpmath.matrix:
    """Full matrix ``[[0, m_D], [m_D^T, M_R]]`` as an ``mpmath.matrix``."""
    n, m = m_D.rows, m_D.cols
    M = mpmath.matrix(n + m, n + m)
    for i in range(n):
        for j in range(m):
            M[i, n + j] = m_D[i, j]
            M[n + j, i] = m_D[i, j]
    for i in range(m):
        for j in range(m):
            M[n + i, n + j] = M_R[i, j]
    return M


def light_sector_observables(
    mass_matrix, n_active: int = 3, ordering: str = "normal"
) -> Dict[str, object]:
    r"""
    Oscillation observables and heavy-neutrino mixing of a seesaw matrix.

    The light-active block ``N`` of the Takagi matrix is written as
    ``N = (1 - \eta) V`` with ``V = N (N^\dagger N)^{-1/2}`` unitary and
    ``1 - \eta = (N N^\dagger)^{1/2}``; mixing angles are read from ``V``
    (PDG convention), as a global fit assuming unitarity would do to leading
    order.

    Parameters:
    -----------
    mass_matrix : array_like or mpmath.matrix
        Full seesaw matrix
    n_active : int, optional
        Number of active neutrinos (3)
    ordering : {'normal', 'inverted'}, optional
        Assignment of the ascending light masses: ``(m_1, m_2, m_3)`` for
        normal ordering, ``(m_3, m_1, m_2)`` for inverted ordering

    Returns:
    --------
    dict
        ``light_masses`` (``m_1, m_2, m_3``), ``heavy_masses``, ``dm21``,
        ``dm3l`` (``m_3^2 - m_1^2`` for NO, ``m_3^2 - m_2^2`` for IO),
        ``s12``, ``s13``, ``s23`` (squared sines), ``eta_max``
        (``max |eta_{alpha beta}|``), ``theta2`` (``|Theta_{alpha i}|^2``,
        ``n x m``), ``U2`` (``sum |Theta|^2``) and ``heavy_splitting``
        (``M_max - M_min``), as floats, plus ``heavy_splitting_mp`` at full
        precision
    """
    masses, U = takagi_factorization_mp(mass_matrix)
    n = n_active
    light, heavy = masses[:n], masses[n:]
    columns = list(range(n))
    if ordering == "inverted":
        columns = [1, 2, 0]  # ascending (m3, m1, m2) -> (m1, m2, m3)
    light = [light[i] for i in columns]
    N = mpmath.matrix(n, n)
    for a in range(n):
        for j, c in enumerate(columns):
            N[a, j] = U[a, c]
    root = hermitian_function_mp(N * _dagger(N), mpmath.sqrt)
    inverse_root = hermitian_function_mp(_dagger(N) * N, lambda e: 1 / mpmath.sqrt(e))
    V = N * inverse_root
    eta = mpmath.eye(n) - root
    s13 = abs(V[0, 2]) ** 2
    s12 = abs(V[0, 1]) ** 2 / (1 - s13)
    s23 = abs(V[1, 2]) ** 2 / (1 - s13)
    theta2 = np.array(
        [[float(abs(U[a, n + i]) ** 2) for i in range(U.cols - n)] for a in range(n)]
    )
    dm3l = light[2] ** 2 - (light[0] ** 2 if ordering == "normal" else light[1] ** 2)
    return {
        "light_masses": [float(x) for x in light],
        "heavy_masses": [float(x) for x in heavy],
        "dm21": float(light[1] ** 2 - light[0] ** 2),
        "dm3l": float(dm3l),
        "s12": float(s12),
        "s13": float(s13),
        "s23": float(s23),
        "eta_max": float(max(abs(x) for x in eta)),
        "theta2": theta2,
        "U2": float(theta2.sum()),
        "heavy_splitting": float(heavy[-1] - heavy[0]) if len(heavy) > 1 else 0.0,
        "heavy_splitting_mp": heavy[-1] - heavy[0] if len(heavy) > 1 else mpmath.mpf(0),
    }
