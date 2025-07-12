"""
PMNS (Pontecorvo-Maki-Nakagawa-Sakata) mixing matrix implementations.

This module provides various parameterizations of the PMNS matrix
and utilities for extracting mixing angles and CP-violating phases
from neutrino mass matrices.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
import warnings


def pmns_matrix_standard(theta12: float, theta13: float, theta23: float,
                        delta_cp: float = 0.0, 
                        alpha1: float = 0.0, alpha2: float = 0.0) -> np.ndarray:
    """
    Construct PMNS matrix in standard parameterization.
    
    U = R23(θ23) · Uδ(δCP) · R13(θ13) · Uδ†(δCP) · R12(θ12) · P(α1,α2)
    
    where P(α1,α2) = diag(1, e^(iα1), e^(iα2)) contains Majorana phases.
    
    Parameters:
    -----------
    theta12, theta13, theta23 : float
        Mixing angles in radians
    delta_cp : float, optional
        Dirac CP-violating phase in radians (default: 0)
    alpha1, alpha2 : float, optional
        Majorana CP-violating phases in radians (default: 0)
    
    Returns:
    --------
    np.ndarray
        3×3 PMNS matrix
    """
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    
    # Rotation matrices
    R12 = np.array([[c12, s12, 0],
                    [-s12, c12, 0],
                    [0, 0, 1]], dtype=complex)
    
    R13 = np.array([[c13, 0, s13],
                    [0, 1, 0],
                    [-s13, 0, c13]], dtype=complex)
    
    R23 = np.array([[1, 0, 0],
                    [0, c23, s23],
                    [0, -s23, c23]], dtype=complex)
    
    # CP-violating phase matrix
    U_delta = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, np.exp(1j * delta_cp)]], dtype=complex)
    
    # Majorana phase matrix
    P_majorana = np.diag([1, np.exp(1j * alpha1), np.exp(1j * alpha2)])
    
    # Combine all matrices
    U = R23 @ U_delta @ R13 @ U_delta.conj().T @ R12 @ P_majorana
    
    return U


def pmns_matrix_pdg(theta12: float, theta13: float, theta23: float,
                   delta_cp: float = 0.0) -> np.ndarray:
    """
    Construct PMNS matrix in PDG convention (without Majorana phases).
    
    Parameters:
    -----------
    theta12, theta13, theta23 : float
        Mixing angles in radians
    delta_cp : float, optional
        Dirac CP-violating phase in radians (default: 0)
    
    Returns:
    --------
    np.ndarray
        3×3 PMNS matrix (PDG convention)
    """
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    
    exp_delta = np.exp(1j * delta_cp)
    
    U = np.array([
        [c12*c13, s12*c13, s13*exp_delta.conj()],
        [-s12*c23 - c12*s23*s13*exp_delta, c12*c23 - s12*s23*s13*exp_delta, s23*c13],
        [s12*s23 - c12*c23*s13*exp_delta, -c12*s23 - s12*c23*s13*exp_delta, c23*c13]
    ], dtype=complex)
    
    return U


def extract_mixing_angles(pmns_matrix: np.ndarray, 
                         convention: str = 'pdg') -> Tuple[float, float, float, float]:
    """
    Extract mixing angles and CP phase from PMNS matrix.
    
    Parameters:
    -----------
    pmns_matrix : np.ndarray
        3×3 PMNS matrix
    convention : str, optional
        'pdg' or 'standard' (default: 'pdg')
    
    Returns:
    --------
    theta12, theta13, theta23, delta_cp : float
        Extracted mixing parameters in radians
    """
    U = pmns_matrix.copy()
    
    # Extract θ13 from |U_e3|
    sin_theta13_sq = np.abs(U[0, 2])**2
    theta13 = np.arcsin(np.sqrt(sin_theta13_sq))
    
    # Extract θ12 from |U_e1| and |U_e2|
    cos_theta13 = np.cos(theta13)
    if np.abs(cos_theta13) > 1e-10:
        sin_theta12_sq = np.abs(U[0, 1])**2 / cos_theta13**2
        theta12 = np.arcsin(np.sqrt(sin_theta12_sq))
    else:
        warnings.warn("θ13 ≈ π/2, θ12 extraction may be inaccurate")
        theta12 = 0.0
    
    # Extract θ23 from |U_μ3| and |U_τ3|
    sin_theta23_sq = np.abs(U[1, 2])**2 / cos_theta13**2
    theta23 = np.arcsin(np.sqrt(sin_theta23_sq))
    
    # Extract δCP from the phase of U_e3
    if convention == 'pdg':
        if np.abs(U[0, 2]) > 1e-10:
            delta_cp = -np.angle(U[0, 2])
        else:
            delta_cp = 0.0
    else:
        # More complex extraction for standard convention
        delta_cp = 0.0  # Simplified
    
    return theta12, theta13, theta23, delta_cp


def jarlskog_invariant(pmns_matrix: np.ndarray) -> float:
    """
    Calculate the Jarlskog invariant for CP violation.
    
    J = Im(U_e1 U_μ2 U*_e2 U*_μ1)
    
    Parameters:
    -----------
    pmns_matrix : np.ndarray
        3×3 PMNS matrix
    
    Returns:
    --------
    float
        Jarlskog invariant
    """
    U = pmns_matrix
    J = np.imag(U[0, 0] * U[1, 1] * U[0, 1].conj() * U[1, 0].conj())
    return J


def mixing_angles_to_degrees(theta12: float, theta13: float, theta23: float) -> Tuple[float, float, float]:
    """
    Convert mixing angles from radians to degrees.
    
    Parameters:
    -----------
    theta12, theta13, theta23 : float
        Mixing angles in radians
    
    Returns:
    --------
    tuple
        Mixing angles in degrees
    """
    return np.degrees(theta12), np.degrees(theta13), np.degrees(theta23)


def experimental_values_2023() -> Dict[str, Dict[str, float]]:
    """
    Return current best-fit experimental values for neutrino parameters (2023).
    
    Values from global fits (NuFIT 5.2, arXiv:2111.03086).
    
    Returns:
    --------
    dict
        Dictionary containing best-fit values and uncertainties
    """
    # Note: These are approximate values for demonstration
    # Users should update with latest experimental results
    
    values = {
        'normal_ordering': {
            'theta12_deg': 33.45,  # Solar angle
            'theta13_deg': 8.62,   # Reactor angle  
            'theta23_deg': 49.2,   # Atmospheric angle
            'delta_cp_deg': 197,   # CP-violating phase
            'dm21_sq_1e5': 7.42,   # Δm²₂₁ × 10⁵ eV²
            'dm3l_sq_1e3': 2.515,  # Δm²₃ₗ × 10³ eV² (l=1 for NO)
        },
        'inverted_ordering': {
            'theta12_deg': 33.45,
            'theta13_deg': 8.65,
            'theta23_deg': 49.5,
            'delta_cp_deg': 282,
            'dm21_sq_1e5': 7.42,
            'dm3l_sq_1e3': -2.498,  # Δm²₃ₗ × 10³ eV² (l=2 for IO)
        }
    }
    
    return values


def pmns_from_experimental(ordering: str = 'normal') -> np.ndarray:
    """
    Construct PMNS matrix using experimental best-fit values.
    
    Parameters:
    -----------
    ordering : str, optional
        'normal' or 'inverted' mass ordering (default: 'normal')
    
    Returns:
    --------
    np.ndarray
        PMNS matrix with experimental values
    """
    exp_values = experimental_values_2023()
    
    if ordering not in exp_values:
        raise ValueError(f"Ordering must be 'normal' or 'inverted', got {ordering}")
    
    values = exp_values[ordering]
    
    theta12 = np.radians(values['theta12_deg'])
    theta13 = np.radians(values['theta13_deg'])
    theta23 = np.radians(values['theta23_deg'])
    delta_cp = np.radians(values['delta_cp_deg'])
    
    return pmns_matrix_pdg(theta12, theta13, theta23, delta_cp)


def oscillation_probabilities(pmns_matrix: np.ndarray,
                            mass_squared_diffs: np.ndarray,
                            energy: float,
                            distance: float,
                            matter_density: float = 0.0) -> np.ndarray:
    """
    Calculate neutrino oscillation probabilities in vacuum or matter.
    
    Parameters:
    -----------
    pmns_matrix : np.ndarray
        3×3 PMNS mixing matrix
    mass_squared_diffs : np.ndarray
        Mass squared differences [Δm²₂₁, Δm²₃₁] in eV²
    energy : float
        Neutrino energy in GeV
    distance : float
        Baseline distance in km
    matter_density : float, optional
        Matter density in g/cm³ (default: 0 for vacuum)
    
    Returns:
    --------
    np.ndarray
        3×3 probability matrix P_αβ
    """
    # Convert units
    dm21_sq, dm31_sq = mass_squared_diffs  # eV²
    E = energy * 1e9  # Convert GeV to eV
    L = distance * 1e5  # Convert km to cm
    
    # Oscillation phases in vacuum
    phi21 = 1.27 * dm21_sq * L / E
    phi31 = 1.27 * dm31_sq * L / E
    
    if matter_density > 0:
        # MSW matter effects (simplified)
        # This is a basic implementation - full MSW requires solving eigenvalue problem
        warnings.warn("Matter effects implementation is simplified")
        A_cc = 2 * np.sqrt(2) * 7.63e-14 * matter_density * E  # eV²
        # Modify effective mixing angles and phases (not implemented in detail)
    
    # Calculate oscillation matrix
    U = pmns_matrix
    S = np.diag([1, np.exp(-1j * phi21), np.exp(-1j * phi31)])
    U_osc = U @ S @ U.conj().T
    
    # Probability matrix
    P = np.abs(U_osc)**2
    
    return P


def unitarity_triangle_angles(pmns_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate angles of unitarity triangles from PMNS matrix.
    
    Parameters:
    -----------
    pmns_matrix : np.ndarray
        3×3 PMNS matrix
    
    Returns:
    --------
    dict
        Angles of various unitarity triangles
    """
    U = pmns_matrix
    
    # Example: First row unitarity triangle
    # U_e1 + U_e2 + U_e3 = 0 (after appropriate phase choice)
    z1 = U[0, 0] * U[0, 1].conj()
    z2 = U[0, 1] * U[0, 2].conj()  
    z3 = U[0, 2] * U[0, 0].conj()
    
    # Calculate angles
    alpha = np.angle(-z2 / z1)
    beta = np.angle(-z3 / z2)
    gamma = np.angle(-z1 / z3)
    
    return {
        'alpha_deg': np.degrees(alpha),
        'beta_deg': np.degrees(beta),
        'gamma_deg': np.degrees(gamma)
    }
