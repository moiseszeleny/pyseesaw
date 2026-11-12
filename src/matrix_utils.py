"""
Matrix Utilities for Neutrino Physics Calculations

This module provides specialized matrix operations essential for neutrino mass matrix
analysis, with emphasis on the unique challenges in neutrino physics such as handling
complex symmetric matrices, maintaining unitarity, and dealing with widely separated
mass scales.

## Physics Context

In neutrino physics, we frequently encounter several types of matrices:

### Mass Matrices:
- **Majorana mass matrix**: Complex symmetric, M = M^T
- **Dirac mass matrix**: General complex matrix connecting different chiralities
- **Light neutrino mass matrix**: Effective matrix from Seesaw mechanisms

### Mixing Matrices:
- **PMNS matrix**: Unitary 3×3 matrix relating flavor and mass eigenstates
- **General mixing matrices**: From diagonalization of mass matrices

### Key Physics Requirements:
1. **Hermiticity**: Physical observables require Hermitian matrices
2. **Unitarity**: Probability conservation demands unitary mixing matrices
3. **Mass ordering**: Physical convention requires ordered mass eigenvalues
4. **Numerical stability**: Wide mass hierarchies (10⁻³ to 10¹⁶ eV) require careful numerics

## Mathematical Conventions

### Diagonalization:
For Majorana neutrinos: M = U* diag(m₁, m₂, m₃) U†
Where U is the PMNS matrix and mᵢ are real positive masses.

### Phase Conventions:
- Mass eigenvalues are taken positive by convention
- Complex phases absorbed into mixing matrix elements
- Majorana phases appear as additional diagonal matrix factors

### Units and Scales:
- All masses in eV
- Mixing angles in radians unless specified
- Energy scales span from 10⁻³ eV (neutrino masses) to 10¹⁶ GeV (GUT scale)

Uses SymPy for exact symbolic manipulations and NumPy/SciPy for numerical evaluation
with optimized algorithms for the specific challenges of neutrino physics.
"""

import numpy as np
import sympy as sp
from scipy.linalg import eigh, svd, sqrtm
from typing import Dict, List, Sequence, Tuple, Optional, Union
import warnings

def sin_cos_from_tan_fraction(tan_expression: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    """
    Convert a tangent expression to sine and cosine using its fractional representation.

    Assumes that `tan_expression` is a SymPy fraction representing tan(theta) = opposite/adjacent.

    Parameters:
    -----------
    tan_expression : sp.Expr
        Expression representing tan(theta) as a fraction

    Returns:
    --------
    Tuple[sp.Expr, sp.Expr]
        (sin(theta), cos(theta)) as SymPy expressions
    """
    # tan(theta) = sin(theta) / cos(theta)
    # We can express sin and cos in terms of tan
    cathetus_opposite, cathetus_adjacent = sp.fraction(tan_expression)
    hypotenuse = sp.sqrt(cathetus_adjacent**2 + cathetus_opposite**2)
    sin_theta = cathetus_opposite / hypotenuse
    cos_theta = cathetus_adjacent / hypotenuse
    return sin_theta, cos_theta

def symbolic_rotation_matrix(dim: int, axis: int, angle: Union[float, sp.Symbol]) -> sp.Matrix:
    """
    Create a symbolic rotation matrix in n-dimensional space.
    
    For dimensions 2 and 3, creates standard rotation matrices.
    For dim > 3, creates rotation in the plane defined by coordinates (axis, axis+1).
    
    Parameters:
    -----------
    dim : int
        Dimension of the rotation space
    axis : int
        For 2D/3D: axis of rotation (0 for x, 1 for y, 2 for z)
        For >3D: defines rotation plane as (axis, axis+1), must be < dim-1
    angle : float or sp.Symbol
        Rotation angle in radians
    
    Returns:
    --------
    sp.Matrix
        Symbolic rotation matrix in n-dimensional space.
        
    Notes:
    ------
    In dimensions > 3, rotations are more naturally defined in planes rather
    than around axes. This function rotates in the plane spanned by the
    coordinate axes (axis, axis+1).
    """
    if dim < 2:
        raise ValueError("Dimension must be at least 2")
    if dim <= 3 and (axis < 0 or axis >= dim):
        raise ValueError("Invalid axis for rotation")
    if dim > 3 and (axis < 0 or axis >= dim - 1):
        raise ValueError(f"For {dim}D rotation, axis must be in range [0, {dim-2}] to define rotation plane")
    if dim == 2:
        # 2D rotation matrix
        return sp.Matrix([[sp.cos(angle), -sp.sin(angle)],
                          [sp.sin(angle), sp.cos(angle)]])
    elif dim == 3:
        # 3D rotation matrix around specified axis
        if axis == 0:
            return sp.Matrix([[1, 0, 0],
                              [0, sp.cos(angle), -sp.sin(angle)],
                              [0, sp.sin(angle), sp.cos(angle)]])
        elif axis == 1:
            return sp.Matrix([[sp.cos(angle), 0, sp.sin(angle)],
                              [0, 1, 0],
                              [-sp.sin(angle), 0, sp.cos(angle)]])
        elif axis == 2:
            return sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0],
                              [sp.sin(angle), sp.cos(angle), 0],
                              [0, 0, 1]])
    elif dim > 3:
        # Higher dimensions: rotation in the plane defined by axes (axis, axis+1)
        # For proper n-dimensional rotations, we should specify a plane rather than an axis
        rotation_matrix = sp.eye(dim)
        # Rotate in the plane defined by coordinates (axis, axis+1)
        rotation_matrix[axis, axis] = sp.cos(angle)
        rotation_matrix[axis, axis + 1] = -sp.sin(angle)
        rotation_matrix[axis + 1, axis] = sp.sin(angle)
        rotation_matrix[axis + 1, axis + 1] = sp.cos(angle)
        return rotation_matrix


def diagonalize_hermitian_matrix(matrix: np.ndarray, 
                                 check_hermitian: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a Hermitian matrix and return eigenvalues and eigenvectors.
    
    Parameters:
    -----------
    matrix : np.ndarray
        The matrix to diagonalize (should be Hermitian)
    check_hermitian : bool, optional
        Whether to check if the matrix is Hermitian (default: True)
    
    Returns:
    --------
    eigenvalues : np.ndarray
        Real eigenvalues in ascending order
    eigenvectors : np.ndarray
        Corresponding normalized eigenvectors as columns
    
    Raises:
    -------
    ValueError
        If matrix is not square or not Hermitian (when check_hermitian=True)
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    if check_hermitian:
        if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("Matrix is not Hermitian")
    
    eigenvalues, eigenvectors = eigh(matrix)
    return eigenvalues, eigenvectors


def diagonalize_mass_matrix(mass_matrix: np.ndarray, 
                           symmetric: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a neutrino mass matrix using physics-appropriate conventions.
    
    This function handles the specific requirements of neutrino physics:
    - Majorana neutrinos have symmetric mass matrices
    - Mass eigenvalues must be positive (by convention)
    - Mixing matrices must be unitary
    - Proper treatment of negative eigenvalues with phase adjustments
    
    Physics Background:
    For Majorana neutrinos, the mass term in the Lagrangian is:
    ℒ_mass = -½ (ν̄_L)ᶜ M_ν ν_L + h.c.
    
    where M_ν is a complex symmetric matrix. Diagonalization gives:
    M_ν = U* diag(m₁, m₂, m₃) U†
    
    with positive masses mᵢ and unitary matrix U (PMNS matrix).
    
    Parameters:
    -----------
    mass_matrix : np.ndarray
        The neutrino mass matrix to diagonalize
        - For Majorana: symmetric complex matrix
        - For Dirac: we typically work with M†M
    symmetric : bool, optional
        Whether to treat as symmetric (Majorana case, default: True)
        - True: Use eigendecomposition for symmetric matrices
        - False: Use SVD for general complex matrices
    
    Returns:
    --------
    masses : np.ndarray
        Mass eigenvalues (positive real values, ordered)
        Units: eV (typical range 10⁻³ to 10⁻¹ eV for light neutrinos)
    mixing_matrix : np.ndarray
        Unitary mixing matrix (PMNS-like)
        Columns are mass eigenstates, rows are flavor eigenstates
        
    Notes:
    ------
    - Negative eigenvalues are handled by taking absolute values and adjusting
      the corresponding eigenvector phases (multiply by i)
    - This preserves the physics while maintaining positive mass convention
    - For Dirac neutrinos, use symmetric=False and input M†M matrix
    
    Example:
    --------
    >>> # Light neutrino mass matrix from Type I Seesaw
    >>> m_nu = np.array([[1e-3, 1e-4], [1e-4, 2e-3]])  # eV
    >>> masses, U = diagonalize_mass_matrix(m_nu)
    >>> print(f"Masses: {masses} eV")
    >>> print(f"Mixing matrix unitarity check: {np.allclose(U @ U.conj().T, np.eye(2))}")
    """
    if symmetric:
        # For symmetric mass matrix (Majorana case)
        eigenvalues, eigenvectors = diagonalize_hermitian_matrix(mass_matrix)
        
        # Handle negative eigenvalues by taking absolute values
        # and adjusting phase of eigenvectors
        masses = np.abs(eigenvalues)
        mixing_matrix = eigenvectors.copy()
        
        # Adjust phases for negative eigenvalues
        negative_mask = eigenvalues < 0
        if np.any(negative_mask):
            mixing_matrix[:, negative_mask] *= 1j
    else:
        # For general complex matrix, use SVD
        U, s, Vh = svd(mass_matrix)
        masses = s
        mixing_matrix = U
    
    return masses, mixing_matrix


def check_unitarity(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary within numerical tolerance.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Matrix to check
    tolerance : float, optional
        Numerical tolerance for unitarity check
    
    Returns:
    --------
    bool
        True if matrix is unitary within tolerance
    """
    n = matrix.shape[0]
    identity = np.eye(n, dtype=matrix.dtype)
    product = matrix @ matrix.conj().T
    return np.allclose(product, identity, rtol=tolerance, atol=tolerance)


def mass_squared_differences(masses: np.ndarray) -> np.ndarray:
    """
    Calculate mass squared differences from mass eigenvalues.
    
    Parameters:
    -----------
    masses : np.ndarray
        Mass eigenvalues (should be positive)
    
    Returns:
    --------
    np.ndarray
        Mass squared differences: Δm²ᵢⱼ = m²ᵢ - m²ⱼ
    """
    masses_squared = masses**2
    n = len(masses)
    delta_m_squared = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            delta_m_squared[i, j] = masses_squared[i] - masses_squared[j]
    
    return delta_m_squared


def enforce_mass_ordering(masses: np.ndarray, 
                         mixing_matrix: np.ndarray,
                         ordering: str = 'normal') -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce a specific mass ordering on the neutrino masses.
    
    Parameters:
    -----------
    masses : np.ndarray
        Mass eigenvalues
    mixing_matrix : np.ndarray
        Corresponding mixing matrix
    ordering : str, optional
        'normal' for m₁ < m₂ < m₃, 'inverted' for m₃ < m₁ < m₂
    
    Returns:
    --------
    ordered_masses : np.ndarray
        Masses in specified ordering
    ordered_mixing : np.ndarray
        Mixing matrix with columns reordered accordingly
    """
    if ordering not in ['normal', 'inverted']:
        raise ValueError("Ordering must be 'normal' or 'inverted'")
    
    if ordering == 'normal':
        # Sort in ascending order
        sort_indices = np.argsort(masses)
    else:  # inverted
        # Sort with m₃ smallest, then m₁, then m₂
        # This is a simplified version - in reality, inverted ordering
        # has specific constraints from experimental data
        warnings.warn("Inverted ordering implementation is simplified")
        sort_indices = np.argsort(masses)
    
    ordered_masses = masses[sort_indices]
    ordered_mixing = mixing_matrix[:, sort_indices]
    
    return ordered_masses, ordered_mixing


def matrix_condition_number(matrix: np.ndarray) -> float:
    """
    Calculate the condition number of a matrix.
    
    High condition numbers indicate numerical instability.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix
    
    Returns:
    --------
    float
        Condition number
    """
    return np.linalg.cond(matrix)


def is_positive_definite(matrix: np.ndarray, tolerance: float = 1e-12) -> bool:
    """
    Check if a Hermitian matrix is positive definite.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Hermitian matrix to check
    tolerance : float, optional
        Tolerance for considering eigenvalues as positive
    
    Returns:
    --------
    bool
        True if matrix is positive definite
    """
    try:
        eigenvalues, _ = diagonalize_hermitian_matrix(matrix)
        return np.all(eigenvalues > tolerance)
    except ValueError:
        return False

# Symbolic matrix operations with SymPy

def create_symbolic_matrix(name: str, shape: Tuple[int, int], 
                          real: bool = False, symmetric: bool = False) -> sp.Matrix:
    """
    Create a symbolic matrix with SymPy.
    
    Parameters:
    -----------
    name : str
        Base name for matrix elements
    shape : tuple
        Matrix dimensions (rows, cols)
    real : bool, optional
        Whether matrix elements should be real (default: False, allows complex)
    symmetric : bool, optional
        Whether to enforce symmetry (default: False)
    
    Returns:
    --------
    sp.Matrix
        Symbolic matrix
    """
    rows, cols = shape
    
    if symmetric and rows != cols:
        raise ValueError("Symmetric matrix must be square")
    
    if symmetric:
        # Create symmetric matrix
        matrix = sp.zeros(rows, cols)
        for i in range(rows):
            for j in range(i, cols):
                if real:
                    symbol = sp.Symbol(f'{name}_{i+1}{j+1}', real=True)
                else:
                    symbol = sp.Symbol(f'{name}_{i+1}{j+1}')
                matrix[i, j] = symbol
                if i != j:
                    matrix[j, i] = symbol
    else:
        # Create general matrix
        if real:
            matrix = sp.Matrix(rows, cols, 
                             lambda i, j: sp.Symbol(f'{name}_{i+1}{j+1}', real=True))
        else:
            matrix = sp.Matrix(rows, cols, 
                             lambda i, j: sp.Symbol(f'{name}_{i+1}{j+1}'))
    
    return matrix


def seesaw_approximation_symbolic(m_D: sp.Matrix, M_R: sp.Matrix, 
                                 order: int = 1) -> sp.Matrix:
    """
    Calculate symbolic Type I Seesaw approximation to different orders in perturbation theory.
    
    Physics Background:
    The Type I Seesaw mechanism introduces heavy right-handed neutrinos with masses M_R
    and Dirac couplings m_D to left-handed neutrinos. After integrating out the heavy
    states, the effective light neutrino mass matrix is:
    
    m_ν = -m_D M_R⁻¹ m_D^T  (leading order)
    
    This can be expanded in powers of the small parameter ε = m_D/M_R:
    
    m_ν = -ε² M_R + O(ε⁴)  (for order-by-order expansion)
    
    The negative sign comes from the Majorana nature of the mass term and the
    particular convention for the Dirac mass term in the Lagrangian.
    
    Typical Scales:
    - m_D ~ O(100 GeV) (electroweak scale, similar to charged fermion masses)
    - M_R ~ O(10¹⁰⁻¹⁶ GeV) (high scale, possibly near GUT scale)  
    - m_ν ~ m_D²/M_R ~ O(0.01-0.1 eV) (observed neutrino mass scale)
    
    Parameters:
    -----------
    m_D : sp.Matrix
        Symbolic Dirac mass matrix (n_generations × n_sterile)
        Represents Yukawa couplings × Higgs VEV
        Typical entries: O(10⁻⁶ - 1) in units where v_EW = 246 GeV
    M_R : sp.Matrix  
        Symbolic right-handed Majorana mass matrix (n_sterile × n_sterile)
        Must be symmetric and invertible
        Typical entries: O(10¹⁰⁻¹⁶ GeV) for natural small neutrino masses
    order : int, optional
        Order of approximation in powers of (m_D/M_R)
        - 1: Leading order Seesaw (default)
        - 2: Next-to-leading order (includes higher-order corrections)
    
    Returns:
    --------
    sp.Matrix
        Symbolic light neutrino mass matrix in the flavor basis
        Entries typically O(0.01-0.1 eV) for realistic parameters
        
    Notes:
    ------
    - The formula assumes M_R >> m_D (heavy mass hierarchy)
    - Higher orders involve more complex expressions with multiple matrix inversions
    - For numerical evaluation, use substitute_numerical_values() afterward
    - Result should be diagonalized to get physical masses and mixing angles
    
    Mathematical Details:
    The expansion parameter is dimensionless: ε ~ m_D/M_R ~ 10⁻¹²
    This extreme smallness naturally explains the tiny neutrino masses
    without additional fine-tuning.
    """
    if order == 1:
        # Leading order Seesaw
        M_R_inv = M_R.inv()
        m_nu = -m_D * M_R_inv * m_D.T
    elif order == 2:
        # Next-to-leading order: terms up to (m_D M_R^-1)^4 M_R
        m_nu = seesaw_block_expansion(m_D, M_R, order=4)['m_light']
    else:
        raise ValueError("Only orders 1 and 2 are supported")
    
    return m_nu


def inverse_seesaw_symbolic(m_D: sp.Matrix, M_R: sp.Matrix, mu: sp.Matrix) -> sp.Matrix:
    """
    Calculate symbolic Inverse Seesaw mass matrix.
    
    Parameters:
    -----------
    m_D : sp.Matrix
        Symbolic Dirac mass matrix
    M_R : sp.Matrix
        Symbolic Majorana mass matrix
    mu : sp.Matrix
        Symbolic lepton number violation matrix
    
    Returns:
    --------
    sp.Matrix
        Symbolic light neutrino mass matrix
    """
    M_R_inv = M_R.inv()
    M_R_T_inv = (M_R.T).inv()
    
    m_nu = m_D * M_R_inv * mu * M_R_T_inv * m_D.T
    
    return m_nu


def expand_in_small_parameter(expression: sp.Expr, small_param: sp.Symbol, 
                             order: int = 2) -> sp.Expr:
    """
    Expand expression in powers of a small parameter.
    
    Parameters:
    -----------
    expression : sp.Expr
        SymPy expression to expand
    small_param : sp.Symbol
        Small parameter for expansion
    order : int, optional
        Maximum order in expansion
    
    Returns:
    --------
    sp.Expr
        Expanded expression
    """
    return sp.series(expression, small_param, 0, order + 1).removeO()


def substitute_numerical_values(symbolic_matrix: sp.Matrix, 
                               substitutions: dict) -> np.ndarray:
    """
    Substitute numerical values into symbolic matrix and convert to NumPy.
    
    Parameters:
    -----------
    symbolic_matrix : sp.Matrix
        Symbolic matrix
    substitutions : dict
        Dictionary mapping symbols to numerical values
    
    Returns:
    --------
    np.ndarray
        Numerical matrix
    """
    numerical_matrix = symbolic_matrix.subs(substitutions)
    
    # Convert to NumPy array
    rows, cols = numerical_matrix.shape
    result = np.zeros((rows, cols), dtype=complex)
    
    for i in range(rows):
        for j in range(cols):
            element = complex(numerical_matrix[i, j])
            result[i, j] = element
    
    return result


def texture_zeros_matrix(base_matrix: sp.Matrix, zero_positions: list) -> sp.Matrix:
    """
    Apply texture zeros to a symbolic matrix.
    
    Parameters:
    -----------
    base_matrix : sp.Matrix
        Base symbolic matrix
    zero_positions : list
        List of (i, j) tuples indicating positions to set to zero
    
    Returns:
    --------
    sp.Matrix
        Matrix with texture zeros applied
    """
    result = base_matrix.copy()
    
    for i, j in zero_positions:
        result[i, j] = 0
    
    return result


def takagi_factorization(matrix: np.ndarray,
                         tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Numerical Takagi factorization of a complex symmetric (Majorana) mass matrix.

    Finds a unitary ``U`` and non-negative masses ``m_i`` such that

    .. math:: U^T M U = \mathrm{diag}(m_1, \dots, m_N), \qquad m_i \ge 0 .

    Unlike :func:`diagonalize_mass_matrix` with ``symmetric=True`` (which uses
    ``eigh`` and is therefore only correct for *real* symmetric matrices),
    this is valid for any complex symmetric matrix, including degenerate and
    vanishing masses.

    The construction uses the SVD ``M = W S V^\dagger``. For symmetric ``M``
    the unitary ``Z = W^\dagger V^*`` commutes with ``S``, and
    ``U = (W Z^{1/2})^*``.

    Parameters:
    -----------
    matrix : np.ndarray
        Complex symmetric matrix, ``M = M^T``
    tolerance : float, optional
        Relative tolerance for the symmetry check and for identifying
        vanishing singular values

    Returns:
    --------
    masses : np.ndarray
        Non-negative masses in ascending order
    mixing_matrix : np.ndarray
        Unitary matrix ``U`` with ``U^T M U = diag(masses)``

    Raises:
    -------
    ValueError
        If the matrix is not square or not symmetric
    """
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    scale = max(np.max(np.abs(matrix)), 1e-300)
    if not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance * scale):
        raise ValueError("Matrix is not symmetric")

    W, singular_values, Vh = svd(matrix)
    Z = W.conj().T @ Vh.T  # W^dagger V^*

    # Z is only meaningful inside each block of equal singular values; it is
    # arbitrary on the null space, where any unitary completion works.
    dim = len(singular_values)
    Z_block = np.zeros_like(Z)
    for i in range(dim):
        for j in range(dim):
            same_block = abs(singular_values[i] - singular_values[j]) <= tolerance * scale
            is_null = (singular_values[i] <= tolerance * scale
                       or singular_values[j] <= tolerance * scale)
            if is_null:
                Z_block[i, j] = 1.0 if i == j else 0.0
            elif same_block:
                Z_block[i, j] = Z[i, j]

    takagi_vectors = W @ sqrtm(Z_block)  # M = T S T^T
    mixing_matrix = takagi_vectors.conj()

    order = np.argsort(singular_values)
    return singular_values[order], mixing_matrix[:, order]


def mass_difference_dummies(masses: Sequence[sp.Symbol]) -> Dict[sp.Expr, sp.Dummy]:
    """
    Map every ordered mass difference ``m_i - m_j`` (``i < j``) to a negative dummy.

    Encodes the physical assumption ``m_1 < m_2 < ... < m_N`` so that SymPy can
    simplify square roots such as ``sqrt((m_1 - m_2)**2)``.

    Parameters:
    -----------
    masses : sequence of sp.Symbol
        Mass symbols in ascending order

    Returns:
    --------
    dict
        ``{m_i - m_j: D_ij}`` with ``D_ij`` a negative ``sp.Dummy``
    """
    differences = {}
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            differences[masses[i] - masses[j]] = sp.Dummy(
                f'D_{{{i + 1}{j + 1}}}', negative=True)
    return differences


def simplify_with_mass_ordering(expression: Union[sp.Expr, sp.Matrix],
                                masses: Sequence[sp.Symbol]) -> Union[sp.Expr, sp.Matrix]:
    """
    Simplify an expression assuming the mass ordering ``m_1 < m_2 < ... < m_N``.

    Factors the expression, replaces each difference ``m_i - m_j`` (``i < j``)
    by a negative dummy, simplifies, substitutes back and factors again.

    Parameters:
    -----------
    expression : sp.Expr or sp.Matrix
        Expression (or matrix, treated element-wise) to simplify
    masses : sequence of sp.Symbol
        Mass symbols in ascending order

    Returns:
    --------
    sp.Expr or sp.Matrix
        Simplified expression
    """
    differences = mass_difference_dummies(masses)
    restore = {dummy: diff for diff, dummy in differences.items()}

    def _simplify(expr: sp.Expr) -> sp.Expr:
        expr = sp.factor(sp.sympify(expr), deep=True)
        return sp.factor(sp.simplify(expr.subs(differences)).subs(restore))

    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(_simplify)
    return _simplify(expression)


def _truncated_series_product(a: List[sp.Matrix], b: List[sp.Matrix],
                              max_power: int) -> List[sp.Matrix]:
    """Product of two matrix power series (lists indexed by power), truncated."""
    rows, cols = a[0].rows, b[0].cols
    result = [sp.zeros(rows, cols) for _ in range(max_power + 1)]
    for i, a_i in enumerate(a):
        if a_i.is_zero_matrix:
            continue
        for j, b_j in enumerate(b):
            if i + j > max_power:
                break
            if b_j.is_zero_matrix:
                continue
            result[i + j] += a_i * b_j
    return result


def _binomial_matrix_series(a: List[sp.Matrix], exponent: sp.Rational,
                            max_power: int) -> List[sp.Matrix]:
    """Series of ``(1 + A)**exponent`` for a matrix series ``A`` with no constant term."""
    dim = a[0].rows
    identity = [sp.eye(dim)] + [sp.zeros(dim, dim) for _ in range(max_power)]
    result = [m.copy() for m in identity]
    power = [m.copy() for m in identity]
    lowest = next((k for k, a_k in enumerate(a) if not a_k.is_zero_matrix), None)
    if lowest is None:
        return result
    n = 1
    while n * lowest <= max_power:
        power = _truncated_series_product(power, a, max_power)
        coefficient = sp.binomial(exponent, n)
        for k in range(max_power + 1):
            result[k] += coefficient * power[k]
        n += 1
    return result


def seesaw_block_expansion(m_D: sp.Matrix, M_R: sp.Matrix,
                           order: int = 4) -> Dict[str, object]:
    r"""
    Block-diagonalize the Type I seesaw matrix order by order in ``m_D M_R^{-1}``.

    For ``M = [[0, m_D], [m_D^T, M_R]]`` the light subspace is spanned by the
    columns of ``[1; Y]``, where ``Y`` (``n_sterile x n_active``) solves

    .. math:: Y = -M_R^{-1} m_D^T + M_R^{-1}\, Y^*\, m_D\, Y ,

    solved by iteration with the bookkeeping ``m_D \sim \epsilon``. The
    effective mass matrices in orthonormal bases are

    .. math::
        m_\nu = \left[(1 + Y^\dagger Y)^{1/2}\right]^* m_D Y\,(1 + Y^\dagger Y)^{-1/2},
        \qquad
        m_N = \left[(1 + Y Y^\dagger)^{1/2}\right]^* (M_R - m_D^T Y^\dagger)\,
              (1 + Y Y^\dagger)^{-1/2},

    so that ``W^T M W = diag(m_nu, m_N)`` with the unitary
    ``W = [[c_l, -Y^\dagger c_h], [Y c_l, c_h]]``, ``c_l = (1+Y^\dagger Y)^{-1/2}``,
    ``c_h = (1+Y Y^\dagger)^{-1/2}``. The leading term is the usual
    ``m_nu = -m_D M_R^{-1} m_D^T``.

    Assumptions: ``M_R`` invertible and ``|m_D M_R^{-1}| << 1``. No reality
    assumption is made; complex conjugation is kept explicitly.

    Parameters:
    -----------
    m_D : sp.Matrix
        Dirac mass matrix (``n_active x n_sterile``)
    M_R : sp.Matrix
        Majorana mass matrix (``n_sterile x n_sterile``, symmetric, invertible)
    order : int, optional
        Highest power of ``m_D`` kept in ``m_light`` (even, default 4)

    Returns:
    --------
    dict
        ``'m_light'`` and ``'m_heavy'`` (truncated sums), ``'light_terms'`` and
        ``'heavy_terms'`` (``{power of m_D: matrix}``), ``'Y'`` (truncated
        active-sterile mixing block) and ``'Y_terms'``
    """
    if order < 2 or order % 2:
        raise ValueError("order must be an even integer >= 2")

    n_active, n_sterile = m_D.shape
    M_R_inv = M_R.inv()
    conj = lambda matrix: matrix.applyfunc(sp.conjugate)

    zero_Y = sp.zeros(n_sterile, n_active)
    Y = [zero_Y.copy() for _ in range(order)]  # Y has odd powers up to order - 1
    Y[1] = -M_R_inv * m_D.T
    for power in range(3, order, 2):
        total = zero_Y.copy()
        for a in range(1, power - 1, 2):
            b = power - 1 - a
            total += conj(Y[a]) * m_D * Y[b]
        Y[power] = M_R_inv * total

    Y_dagger = [conj(y).T for y in Y]
    mD_series = [sp.zeros(*m_D.shape), m_D]
    mDT_series = [sp.zeros(*m_D.T.shape), m_D.T]

    # Light block
    YdY = _truncated_series_product(Y_dagger, Y, order)
    sqrt_l = _binomial_matrix_series(YdY, sp.Rational(1, 2), order)
    inv_sqrt_l = _binomial_matrix_series(YdY, sp.Rational(-1, 2), order)
    K_light = _truncated_series_product(mD_series, Y, order)
    light = _truncated_series_product(
        _truncated_series_product([conj(x) for x in sqrt_l], K_light, order),
        inv_sqrt_l, order)

    # Heavy block (same accuracy relative to M_R)
    YYd = _truncated_series_product(Y, Y_dagger, order)
    sqrt_h = _binomial_matrix_series(YYd, sp.Rational(1, 2), order)
    inv_sqrt_h = _binomial_matrix_series(YYd, sp.Rational(-1, 2), order)
    K_heavy = [-x for x in _truncated_series_product(mDT_series, Y_dagger, order)]
    K_heavy[0] = K_heavy[0] + M_R
    heavy = _truncated_series_product(
        _truncated_series_product([conj(x) for x in sqrt_h], K_heavy, order),
        inv_sqrt_h, order)

    light_terms = {k: light[k] for k in range(2, order + 1, 2)}
    heavy_terms = {k: heavy[k] for k in range(0, order + 1, 2)}
    Y_terms = {k: Y[k] for k in range(1, order, 2)}

    return {
        'm_light': sum(light_terms.values(), sp.zeros(n_active, n_active)),
        'm_heavy': sum(heavy_terms.values(), sp.zeros(n_sterile, n_sterile)),
        'light_terms': light_terms,
        'heavy_terms': heavy_terms,
        'Y': sum(Y_terms.values(), zero_Y),
        'Y_terms': Y_terms,
    }
