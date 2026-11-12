r"""
Confronting the Type I seesaw with (n active, m sterile) neutrinos with data.

The question addressed here is: *what is the smallest number of sterile
(right-handed) neutrinos ``m`` for which the Type I seesaw reproduces the
global-fit oscillation data?* (The number of active neutrinos is fixed to
``n = 3`` by the invisible Z width, ``N_nu = 2.9963 +- 0.0074``.)

Physics summary
---------------
The full ``(n + m) x (n + m)`` Majorana mass matrix

.. math::
    \mathcal{M} = \begin{pmatrix} 0 & m_D \\ m_D^T & M_R \end{pmatrix}

is Takagi-diagonalized, :math:`W^T \mathcal{M} W = \mathrm{diag}(d, D)` with
non-negative masses. Because the active-active block vanishes,
:math:`\det \mathcal{M} \propto \det(m_D M_R^{-1} m_D^T)` and the light block
has rank :math:`\min(n, m)`: **exactly** :math:`n - m` light states are
massless for :math:`n > m` (to all orders in :math:`m_D / M_R`, at tree level).

* ``(3, 1)``: two massless states, a single non-zero splitting. Excluded,
  since both :math:`\Delta m^2_{21}` and :math:`|\Delta m^2_{3\ell}|` are
  measured to be non-zero.
* ``(3, 2)``: one massless state, :math:`m_{\rm lightest} = 0`. The minimal
  viable case.
* ``(3, 3)``: :math:`m_{\rm lightest}` free.

Low-energy observables depend on :math:`m_D` and :math:`M_R` only through the
light mass matrix. To leading order it is
:math:`m_\nu = -m_D M_R^{-1} m_D^T = -Y Y^T` with
:math:`Y = m_D M_R^{-1/2}` (``M_R`` diagonal and positive, which is always
achievable by a sterile-basis rotation). ``Y`` is a *generic* complex
``n x m`` matrix, so fitting over ``Y`` scans every Type I model with that
field content, independently of the heavy scale. Corrections of order
:math:`m_D^2/M_R^2 \sim m_\nu / M_N` are below :math:`10^{-10}` for
:math:`M_N \gtrsim 1` GeV. They are checked separately with an exact,
high-precision diagonalization of the full matrix
(:func:`exact_light_observables`).

Conventions
-----------
* Takagi convention :math:`U^T M U = \mathrm{diag}(m_i \ge 0)` as in
  :mod:`src.symbolic_diagonalization`, so that :math:`\nu_\alpha = \sum_i
  N_{\alpha i}\,\nu_i` with ``N`` the active-light block (the non-unitary PMNS
  matrix).
* PDG parametrization of the unitary part, NuFIT labelling of the mass states
  (normal: :math:`m_1<m_2<m_3`; inverted: :math:`m_3<m_1<m_2`),
  :math:`\Delta m^2_{3\ell} = \Delta m^2_{31}` (NO) or
  :math:`\Delta m^2_{32}` (IO).
* Light masses and :math:`Y^2` in **eV**, heavy masses and :math:`m_D` in
  **GeV** where stated.
"""

from typing import Dict, Optional, Sequence, Tuple

import mpmath
import numpy as np
from scipy.linalg import eigh, polar
from scipy.optimize import least_squares

from .matrix_utils import takagi_factorization
from .pmns_matrix import nufit_6_0, pmns_matrix_pdg

__all__ = [
    "GEV",
    "rank_argument",
    "oscillation_parameters",
    "light_observables",
    "chi2_contributions",
    "chi2",
    "fit_case",
    "casas_ibarra_exact_numeric",
    "minimal_model",
    "exact_light_observables",
    "majorana_effective_mass_range",
]

GEV = 1e9
"""One GeV in eV."""

FIT_OBSERVABLES = (
    "dm21_sq",
    "dm3l_sq",
    "sin2_theta12",
    "sin2_theta13",
    "sin2_theta23",
    "delta_cp_deg",
)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def rank_argument(n_active: int, n_sterile: int) -> Dict[str, int]:
    """
    Count massive and massless light states of the Type I seesaw.

    The light Majorana mass matrix has rank ``min(n, m)`` (exactly, at tree
    level), so ``max(n - m, 0)`` light neutrinos are massless and at most
    ``min(n, m) - 1`` independent non-zero mass splittings exist among the
    massive light states (plus those relative to massless states).

    Parameters:
    -----------
    n_active : int
        Number of active (left-handed) neutrinos ``n``
    n_sterile : int
        Number of sterile (right-handed) neutrinos ``m``

    Returns:
    --------
    dict
        ``'n_light_massive'``, ``'n_massless'``, ``'n_nonzero_splittings'``
        (number of independent non-zero :math:`\\Delta m^2` among the ``n``
        light states) and ``'n_heavy'``
    """
    massive = min(n_active, n_sterile)
    massless = n_active - massive
    distinct_levels = massive + (1 if massless else 0)
    return {
        "n_light_massive": massive,
        "n_massless": massless,
        "n_nonzero_splittings": max(distinct_levels - 1, 0),
        "n_heavy": n_sterile,
    }


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


def oscillation_parameters(U: np.ndarray) -> Dict[str, float]:
    r"""
    PDG mixing parameters of a unitary 3x3 matrix from rephasing invariants.

    .. math::
        s_{13}^2 = |U_{e3}|^2,\quad
        s_{12}^2 = \frac{|U_{e2}|^2}{1 - |U_{e3}|^2},\quad
        s_{23}^2 = \frac{|U_{\mu 3}|^2}{1 - |U_{e3}|^2},

    and :math:`\delta` from :math:`J = \mathrm{Im}(U_{e1} U_{\mu 2}
    U_{e2}^* U_{\mu 1}^*) = s_{12}c_{12}s_{23}c_{23}s_{13}c_{13}^2\sin\delta`
    together with :math:`|U_{\mu 1}|^2 = s_{12}^2 c_{23}^2 + c_{12}^2
    s_{23}^2 s_{13}^2 + 2 s_{12}c_{12}s_{23}c_{23}s_{13}\cos\delta`.
    The result does not depend on column or row phases (Majorana phases,
    charged-lepton rephasing).

    Parameters:
    -----------
    U : np.ndarray
        Unitary 3x3 matrix, rows (e, mu, tau), columns (nu_1, nu_2, nu_3)

    Returns:
    --------
    dict
        ``sin2_theta12``, ``sin2_theta13``, ``sin2_theta23``,
        ``delta_cp_deg`` (in [0, 360)) and ``jarlskog``
    """
    A = np.abs(U) ** 2
    s13sq = A[0, 2]
    c13sq = 1.0 - s13sq
    # clip: degenerate (e.g. massless) subspaces can push ratios past 1 by rounding
    s12sq = float(np.clip(A[0, 1] / c13sq, 0.0, 1.0))
    s23sq = float(np.clip(A[1, 2] / c13sq, 0.0, 1.0))
    s12, c12 = np.sqrt(s12sq), np.sqrt(1 - s12sq)
    s23, c23 = np.sqrt(s23sq), np.sqrt(1 - s23sq)
    s13 = np.sqrt(s13sq)
    J = float(np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0])))
    norm = s12 * c12 * s23 * c23 * s13
    if norm < 1e-14:
        delta = 0.0
    else:
        sin_delta = J / (norm * c13sq)
        cos_delta = (A[1, 0] - s12sq * c23**2 - c12**2 * s23sq * s13sq) / (2 * norm)
        delta = float(np.degrees(np.arctan2(sin_delta, cos_delta)) % 360.0)
    return {
        "sin2_theta12": float(s12sq),
        "sin2_theta13": float(s13sq),
        "sin2_theta23": float(s23sq),
        "delta_cp_deg": delta,
        "jarlskog": J,
    }


def _physical_order(ordering: str) -> Tuple[int, int, int]:
    """Indices of (nu_1, nu_2, nu_3) among light states sorted by mass."""
    if ordering == "normal":
        return (0, 1, 2)
    if ordering == "inverted":
        return (1, 2, 0)
    raise ValueError(f"ordering must be 'normal' or 'inverted', got {ordering!r}")


def light_observables(
    light_masses: np.ndarray, N: np.ndarray, ordering: str
) -> Dict[str, object]:
    r"""
    Oscillation and absolute-mass observables of three light Majorana states.

    Parameters:
    -----------
    light_masses : np.ndarray
        The three light masses in **eV**, ascending
    N : np.ndarray
        Active-light mixing block (3x3, columns in the same order), Takagi
        convention; it may be slightly non-unitary
    ordering : {'normal', 'inverted'}
        How the ascending states are labelled (NuFIT convention)

    Returns:
    --------
    dict
        ``masses`` (m_1, m_2, m_3), ``dm21_sq``, ``dm3l_sq`` (eV^2), the PDG
        mixing parameters of the unitary part of ``N`` (polar decomposition
        ``N = (1 - eta) U``), ``sum_masses``, ``m_beta``, ``m_betabeta`` (eV),
        ``eta`` (3x3 Hermitian, :math:`(1 - N N^\dagger)/2`) and ``U``
    """
    order = list(_physical_order(ordering))
    m = np.asarray(light_masses, dtype=float)[order]
    N_phys = np.asarray(N, dtype=complex)[:, order]
    U, _ = polar(N_phys, side="left")
    dm21 = m[1] ** 2 - m[0] ** 2
    dm3l = m[2] ** 2 - (m[0] ** 2 if ordering == "normal" else m[1] ** 2)
    result: Dict[str, object] = {
        "masses": m,
        "dm21_sq": float(dm21),
        "dm3l_sq": float(dm3l),
        "sum_masses": float(np.sum(m)),
        "m_beta": float(np.sqrt(np.sum(np.abs(N_phys[0]) ** 2 * m**2))),
        "m_betabeta": float(np.abs(np.sum(N_phys[0] ** 2 * m))),
        "eta": 0.5 * (np.eye(3) - N_phys @ N_phys.conj().T),
        "U": U,
    }
    result.update(oscillation_parameters(U))
    return result


# ---------------------------------------------------------------------------
# chi^2
# ---------------------------------------------------------------------------


def chi2_contributions(
    observables: Dict[str, object],
    data: Dict[str, Dict],
    ordering: str,
    include_delta: bool = True,
) -> Dict[str, float]:
    r"""
    Pull of each observable with respect to a global fit (Gaussian approximation).

    Each pull is :math:`(x - x_{\rm bf})/\sigma_\pm`, where the upper error is
    used above the best fit and the lower error below. For the Dirac phase the
    difference is wrapped to :math:`(-180^\circ, 180^\circ]`. This
    approximation to the real NuFIT :math:`\Delta\chi^2` profiles is adequate
    to separate viable from excluded field contents. It is poor for
    :math:`\delta_{CP}` and for the :math:`\theta_{23}` octant.

    Parameters:
    -----------
    observables : dict
        Output of :func:`light_observables`
    data : dict
        Output of :func:`nufit_6_0`
    ordering : {'normal', 'inverted'}
    include_delta : bool, optional
        Include :math:`\delta_{CP}` (default True)

    Returns:
    --------
    dict
        Signed pull per observable
    """
    table = data[ordering]
    pulls = {}
    for key in FIT_OBSERVABLES:
        if key == "delta_cp_deg" and not include_delta:
            continue
        bfp, up, down = table[key][:3]
        value = float(observables[key])  # type: ignore[arg-type]
        diff = value - bfp
        if key == "delta_cp_deg":
            diff = (diff + 180.0) % 360.0 - 180.0
        pulls[key] = diff / (up if diff > 0 else down)
    return pulls


def chi2(
    observables: Dict[str, object],
    data: Dict[str, Dict],
    ordering: str,
    include_delta: bool = True,
    include_ordering_offset: bool = False,
) -> float:
    """
    Gaussian chi^2 of the observables, see :func:`chi2_contributions`.

    With ``include_ordering_offset`` the NuFIT :math:`\\Delta\\chi^2` of the
    chosen ordering relative to the global minimum is added.
    """
    pulls = chi2_contributions(observables, data, ordering, include_delta)
    total = float(sum(p**2 for p in pulls.values()))
    if include_ordering_offset:
        total += float(data[ordering]["delta_chi2"])
    return total


# ---------------------------------------------------------------------------
# Brute-force fit over generic Type I models
# ---------------------------------------------------------------------------


def _effective_observables(Y: np.ndarray, ordering: str) -> Dict[str, object]:
    m_nu = -Y @ Y.T
    masses, U = takagi_factorization(0.5 * (m_nu + m_nu.T), tolerance=1e-14)
    return light_observables(masses, U, ordering)


def fit_case(
    n_sterile: int,
    ordering: str = "normal",
    data: Optional[Dict[str, Dict]] = None,
    n_starts: int = 20,
    include_delta: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    r"""
    Minimum chi^2 of the Type I seesaw with 3 active and ``m`` sterile neutrinos.

    The fit scans the generic complex :math:`3\times m` matrix
    :math:`Y = m_D M_R^{-1/2}` (in :math:`\sqrt{\rm eV}`). This covers every
    choice of :math:`m_D` and of the heavy spectrum, because the low-energy
    observables depend only on :math:`m_\nu = -Y Y^T` (see module docstring).
    It assumes nothing about the structure of the solution (no Casas-Ibarra
    input). ``scipy.optimize.least_squares`` is started from ``n_starts``
    random points.

    Parameters:
    -----------
    n_sterile : int
        Number of sterile neutrinos ``m``
    ordering : {'normal', 'inverted'}
    data : dict, optional
        Global-fit data (default ``nufit_6_0('with_sk')``)
    n_starts : int, optional
        Number of random starting points
    include_delta : bool, optional
        Fit :math:`\delta_{CP}` as well
    seed : int, optional
        Random seed

    Returns:
    --------
    dict
        ``chi2`` (best), ``pulls``, ``observables`` and ``Y`` at the minimum,
        and ``all_chi2`` (one value per start)
    """
    data = nufit_6_0("with_sk") if data is None else data
    rng = np.random.default_rng(seed)
    n_params = 2 * 3 * n_sterile

    def unpack(x: np.ndarray) -> np.ndarray:
        return (x[: n_params // 2] + 1j * x[n_params // 2 :]).reshape(3, n_sterile)

    def residuals(x: np.ndarray) -> np.ndarray:
        obs = _effective_observables(unpack(x), ordering)
        return np.array(list(chi2_contributions(obs, data, ordering, include_delta).values()))

    best: Optional[Dict[str, object]] = None
    all_chi2 = []
    for _ in range(n_starts):
        x0 = rng.normal(scale=0.15, size=n_params)  # |Y|^2 ~ 0.02 eV
        try:
            solution = least_squares(residuals, x0, method="trf", x_scale=0.1,
                                     max_nfev=4000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        value = float(2 * solution.cost)
        all_chi2.append(value)
        if best is None or value < best["chi2"]:  # type: ignore[operator]
            Y = unpack(solution.x)
            obs = _effective_observables(Y, ordering)
            best = {
                "chi2": value,
                "pulls": chi2_contributions(obs, data, ordering, include_delta),
                "observables": obs,
                "Y": Y,
            }
    if best is None:
        raise RuntimeError("All fits failed")
    best["all_chi2"] = np.array(sorted(all_chi2))
    best["n_sterile"] = n_sterile
    best["ordering"] = ordering
    return best


# ---------------------------------------------------------------------------
# Exact Casas-Ibarra construction (numerical port)
# ---------------------------------------------------------------------------


def _hermitian_function(H: np.ndarray, power: float) -> np.ndarray:
    values, vectors = eigh(0.5 * (H + H.conj().T))
    return (vectors * values**power) @ vectors.conj().T


def casas_ibarra_exact_numeric(
    light_masses: Sequence[float],
    heavy_masses: Sequence[float],
    orthogonal: np.ndarray,
    active_rotation: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    r"""
    Exact (all-orders) Casas-Ibarra inverse problem, numerical version.

    Numerical port of :func:`src.symbolic_diagonalization.exact_seesaw_parametrization`:
    :math:`X = i D^{-1/2} O d^{1/2}`, :math:`c_l = (1+X^\dagger X)^{-1/2}`,
    :math:`c_h = (1+X X^\dagger)^{-1/2}`,

    .. math::
        m_D = U_a^*\, c_l^* X^T D\, c_h^{-1},\qquad
        M_R = X^* c_l^* d\, c_l X^\dagger + c_h^* D\, c_h,

    with :math:`W^T \mathcal{M} W = \mathrm{diag}(d, D)` for
    :math:`W = \mathrm{diag}(U_a, 1) \begin{pmatrix} c_l & X^\dagger c_h \\ -X c_l & c_h \end{pmatrix}`.
    The active-light block of ``W`` is :math:`N = U_a c_l`, so ``U_a`` is the
    unitary part of the PMNS matrix (including Majorana phases) and
    :math:`\eta = (1 - c_l^2)/2` in the rotated basis.

    All masses must be in the same unit.

    Parameters:
    -----------
    light_masses : sequence of float
        ``n`` light masses; for ``n > m`` the first ``n - m`` must be zero
    heavy_masses : sequence of float
        ``m`` heavy masses (positive)
    orthogonal : np.ndarray
        ``m x n`` complex matrix with :math:`O^T O` the identity on the
        massive light states
    active_rotation : np.ndarray, optional
        Unitary ``n x n`` ``U_a`` (default identity)

    Returns:
    --------
    dict
        ``'m_D'``, ``'M_R'``, ``'M'`` (full matrix), ``'W'``, ``'X'``
    """
    d = np.asarray(light_masses, dtype=float)
    D = np.asarray(heavy_masses, dtype=float)
    n, m = len(d), len(D)
    O = np.asarray(orthogonal, dtype=complex)
    if O.shape != (m, n):
        raise ValueError(f"orthogonal must have shape ({m}, {n})")
    U_a = np.eye(n, dtype=complex) if active_rotation is None else np.asarray(
        active_rotation, dtype=complex)

    X = 1j * (O * np.sqrt(d)[None, :]) / np.sqrt(D)[:, None]
    c_l = _hermitian_function(np.eye(n) + X.conj().T @ X, -0.5)
    c_h = _hermitian_function(np.eye(m) + X @ X.conj().T, -0.5)
    c_h_inv = _hermitian_function(np.eye(m) + X @ X.conj().T, 0.5)

    m_D = U_a.conj() @ c_l.conj() @ X.T @ np.diag(D) @ c_h_inv
    M_R = X.conj() @ c_l.conj() @ np.diag(d) @ c_l @ X.conj().T + c_h.conj() @ np.diag(D) @ c_h
    M_R = 0.5 * (M_R + M_R.T)
    M = np.block([[np.zeros((n, n)), m_D], [m_D.T, M_R]])
    W = np.block([[U_a @ c_l, U_a @ X.conj().T @ c_h], [-X @ c_l, c_h]])
    return {"m_D": m_D, "M_R": M_R, "M": M, "W": W, "X": X}


def minimal_model(
    ordering: str = "normal",
    data: Optional[Dict[str, Dict]] = None,
    z: complex = 0.0,
    majorana_phase: float = 0.0,
    heavy_masses_gev: Sequence[float] = (1e13, 1e14),
    branch: int = 1,
) -> Dict[str, object]:
    r"""
    The minimal (3 active, 2 sterile) seesaw built to reproduce a global fit.

    The best-fit oscillation parameters fix :math:`d = (0, m_a, m_b)` and
    the PMNS matrix. The exact Casas-Ibarra construction then gives
    :math:`m_D` and :math:`M_R`. The free high-energy parameters are the
    complex angle ``z`` of

    .. math::
        O = \begin{pmatrix} 0 & \cos z & \mp\sin z \\ 0 & \pm\sin z & \cos z \end{pmatrix}
        \ (\text{NO}), \qquad
        \text{the zero column is } \nu_3 \text{ for IO},

    the heavy masses, and the single physical Majorana phase. With
    :math:`m_{\rm lightest} = 0` the latter is the relative phase of the two
    massive states, and both orderings use
    :math:`U_a = U_{\rm PDG}\,\mathrm{diag}(1, e^{i\alpha/2}, 1)`.
    For NO (:math:`m_1 = 0`), :math:`\alpha` is the :math:`\nu_2`-:math:`\nu_3`
    phase. For IO (:math:`m_3 = 0`), it is the :math:`\nu_1`-:math:`\nu_2` phase.

    Parameters:
    -----------
    ordering : {'normal', 'inverted'}
    data : dict, optional
        Global fit (default ``nufit_6_0('with_sk')``)
    z : complex, optional
        Casas-Ibarra angle. A large :math:`|\mathrm{Im}\,z|` enhances the
        active-sterile mixing.
    majorana_phase : float, optional
        :math:`\alpha` in radians
    heavy_masses_gev : sequence of two floats, optional
        Heavy Majorana masses in GeV
    branch : {+1, -1}, optional
        Sign choice in ``O`` (the two disconnected branches of O(2, C))

    Returns:
    --------
    dict
        ``'light_masses'`` (eV, ascending), ``'heavy_masses_gev'``,
        ``'m_D_gev'``, ``'M_R_gev'``, ``'M_ev'`` (full 5x5 in eV), ``'W'``,
        ``'N'`` (active-light block, columns ascending in mass) and
        ``'observables'``
    """
    data = nufit_6_0("with_sk") if data is None else data
    table = data[ordering]
    dm21, dm3l = table["dm21_sq"][0], table["dm3l_sq"][0]
    # Use the sin^2 values: the quoted angles in degrees are rounded.
    s12 = np.sqrt(table["sin2_theta12"][0])
    s13 = np.sqrt(table["sin2_theta13"][0])
    s23 = np.sqrt(table["sin2_theta23"][0])
    U = pmns_matrix_pdg(np.arcsin(s12), np.arcsin(s13), np.arcsin(s23),
                        np.radians(table["delta_cp_deg"][0]))
    U = U @ np.diag([1.0, np.exp(0.5j * majorana_phase), 1.0])

    if ordering == "normal":
        m_phys = np.array([0.0, np.sqrt(dm21), np.sqrt(dm3l)])
        columns = [0, 1, 2]  # ascending = (nu1, nu2, nu3)
    elif ordering == "inverted":
        m2 = np.sqrt(-dm3l)  # m3 = 0, dm32 = -m2^2
        m_phys = np.array([np.sqrt(m2**2 - dm21), m2, 0.0])
        columns = [2, 0, 1]  # ascending = (nu3, nu1, nu2)
    else:
        raise ValueError(f"ordering must be 'normal' or 'inverted', got {ordering!r}")

    d = m_phys[columns]
    U_a = U[:, columns]
    c, s = np.cos(z), np.sin(z)
    O = np.array([[0.0, c, -branch * s], [0.0, branch * s, c]], dtype=complex)
    D = np.asarray(heavy_masses_gev, dtype=float) * GEV

    result = casas_ibarra_exact_numeric(d, D, O, U_a)
    N = result["W"][:3, :3]
    return {
        "light_masses": d,
        "heavy_masses_gev": np.asarray(heavy_masses_gev, dtype=float),
        "m_D_gev": result["m_D"] / GEV,
        "M_R_gev": result["M_R"] / GEV,
        "M_ev": result["M"],
        "W": result["W"],
        "N": N,
        "observables": light_observables(d, N, ordering),
    }


# ---------------------------------------------------------------------------
# Exact check of the full matrix at high precision
# ---------------------------------------------------------------------------


def exact_light_observables(
    M: np.ndarray, n_active: int = 3, ordering: str = "normal", dps: int = 80
) -> Dict[str, object]:
    r"""
    Light spectrum and mixing of the full seesaw matrix without any expansion.

    With :math:`W^T\mathcal{M}W = \mathrm{diag}(m)` one has
    :math:`\mathcal{M}^\dagger\mathcal{M} = W\,\mathrm{diag}(m^2)\,W^\dagger`,
    which is diagonalized here. It works in ``mpmath`` at ``dps`` digits, because the light and heavy
    eigenvalues of :math:`\mathcal{M}^\dagger\mathcal{M}` differ by up to
    :math:`(M_N/m_\nu)^2 \sim 10^{50}`, which double precision cannot resolve.
    The eigenvectors fix ``W`` only up to column phases, so the function
    returns phase-independent quantities only: masses, splittings, mixing
    angles, :math:`\delta` and :math:`J`. It returns no :math:`m_{\beta\beta}`.

    Parameters:
    -----------
    M : np.ndarray
        Full ``(n + m)`` symmetric mass matrix (eV)
    n_active : int, optional
        Number of active states (= number of light states)
    ordering : {'normal', 'inverted'}
    dps : int, optional
        Decimal digits

    Returns:
    --------
    dict
        ``'light_masses'`` (ascending), ``'heavy_masses'``, and the
        oscillation parameters of the unitary part of the active-light block
    """
    with mpmath.workdps(dps):
        A = mpmath.matrix([[mpmath.mpc(complex(x)) for x in row] for row in M])
        H = A.H * A
        H = (H + H.H) / 2
        values, vectors = mpmath.eighe(H)
        dim = A.rows
        masses = [mpmath.sqrt(max(mpmath.re(values[i]), 0)) for i in range(dim)]
        order = sorted(range(dim), key=lambda i: masses[i])
        light = order[:n_active]
        N = np.array(
            [[complex(vectors[a, i]) for i in light] for a in range(n_active)]
        )
        light_masses = np.array([float(masses[i]) for i in light])
        heavy_masses = np.array([float(masses[i]) for i in order[n_active:]])
    obs = light_observables(light_masses, N, ordering)
    obs.pop("m_betabeta")  # not phase independent here
    obs["light_masses"] = light_masses
    obs["heavy_masses"] = heavy_masses
    return obs


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def majorana_effective_mass_range(
    ordering: str,
    data: Optional[Dict[str, Dict]] = None,
    n_phase: int = 721,
) -> Dict[str, object]:
    r"""
    :math:`m_{\beta\beta}` of the minimal (3, 2) seesaw over its Majorana phase.

    With one massless state,
    :math:`m_{\beta\beta} = |m_2 s_{12}^2 c_{13}^2 e^{i\alpha} + m_3 s_{13}^2
    e^{-2i\delta}|` (NO) or :math:`|m_1 c_{12}^2 c_{13}^2 + m_2 s_{12}^2
    c_{13}^2 e^{i\alpha}|` (IO). The function evaluates it at the best-fit
    point, for ``n_phase`` values of :math:`\alpha \in [0, 2\pi]`.

    Returns:
    --------
    dict
        ``'alpha'``, ``'m_betabeta'`` (eV), ``'min'``, ``'max'``
    """
    data = nufit_6_0("with_sk") if data is None else data
    alphas = np.linspace(0, 2 * np.pi, n_phase)
    values = np.array([
        minimal_model(ordering, data, majorana_phase=a)["observables"]["m_betabeta"]
        for a in alphas
    ])
    return {"alpha": alphas, "m_betabeta": values,
            "min": float(values.min()), "max": float(values.max())}
