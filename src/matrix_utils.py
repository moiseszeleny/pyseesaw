"""
Matrix utilities for neutrino mass matrix calculations.

This module provides general-purpose matrix operations commonly used
in neutrino physics, including diagonalization, eigenvalue analysis,
and numerical stability checks. Uses SymPy for symbolic calculations
and NumPy for numerical evaluation.
"""

import numpy as np
import sympy as sp
from scipy.linalg import eigh, svd
from typing import Tuple, Optional, Union
import warnings


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
    Diagonalize a neutrino mass matrix.
    
    For Majorana neutrinos, the mass matrix is symmetric.
    For Dirac neutrinos, we typically work with M†M.
    
    Parameters:
    -----------
    mass_matrix : np.ndarray
        The mass matrix to diagonalize
    symmetric : bool, optional
        Whether to treat as symmetric (Majorana) or general complex matrix
    
    Returns:
    --------
    masses : np.ndarray
        Mass eigenvalues (positive real values)
    mixing_matrix : np.ndarray
        Unitary mixing matrix
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
    Calculate symbolic Seesaw approximation to different orders.
    
    The Type I Seesaw formula: m_ν = -m_D M_R^(-1) m_D^T
    can be expanded in powers of m_D/M_R.
    
    Parameters:
    -----------
    m_D : sp.Matrix
        Symbolic Dirac mass matrix
    M_R : sp.Matrix
        Symbolic Majorana mass matrix
    order : int, optional
        Order of approximation (1 = leading order, 2 = next-to-leading, etc.)
    
    Returns:
    --------
    sp.Matrix
        Symbolic light neutrino mass matrix
    """
    if order == 1:
        # Leading order Seesaw
        M_R_inv = M_R.inv()
        m_nu = -m_D * M_R_inv * m_D.T
    elif order == 2:
        # Include next-to-leading corrections
        # This would include (m_D/M_R)^3 terms in full theory
        warnings.warn("Higher-order corrections not fully implemented")
        M_R_inv = M_R.inv()
        m_nu = -m_D * M_R_inv * m_D.T
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
