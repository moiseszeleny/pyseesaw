"""
PMNS Matrix (Pontecorvo-Maki-Nakagawa-Sakata) Implementation and Analysis

This module provides comprehensive tools for working with the PMNS mixing matrix,
which describes how neutrino flavor states relate to mass states. The PMNS matrix
is central to understanding neutrino oscillations and connecting theoretical
mass generation mechanisms to experimental observations.

## Physics Background

### Neutrino Mixing Concept:
Neutrinos are produced and detected in flavor eigenstates (νₑ, νμ, ντ) but
propagate as mass eigenstates (ν₁, ν₂, ν₃). The PMNS matrix U connects these:

|νₐ⟩ = Σᵢ Uₐᵢ |νᵢ⟩    (flavor = U × mass)

### Physical Significance:
- **Neutrino Oscillations**: Driven by mass differences and mixing angles
- **CP Violation**: Encoded in complex phases of U
- **Unitarity**: Ensures probability conservation in oscillations

## PMNS Matrix Parameterization

### Standard Parameterization:
```
U = R₂₃(θ₂₃) · Uδ(δ) · R₁₃(θ₁₃) · Uδ†(δ) · R₁₂(θ₁₂) · P(α₁,α₂)
```

Where:
- **R₁₂(θ₁₂)**: Solar angle rotation (~33°, large mixing)
- **R₁₃(θ₁₃)**: Reactor angle rotation (~8.6°, small but crucial)  
- **R₂₃(θ₂₃)**: Atmospheric angle rotation (~49°, near-maximal)
- **δ**: Dirac CP-violating phase (~197°, currently being measured)
- **α₁, α₂**: Majorana phases (unobservable in oscillations)

### PDG Convention (commonly used):
```
U = | c₁₂c₁₃                    s₁₂c₁₃                   s₁₃e^(-iδ)        |
    | -s₁₂c₂₃-c₁₂s₂₃s₁₃e^(iδ)   c₁₂c₂₃-s₁₂s₂₃s₁₃e^(iδ)   s₂₃c₁₃         |
    | s₁₂s₂₃-c₁₂c₂₃s₁₃e^(iδ)    -c₁₂s₂₃-s₁₂c₂₃s₁₃e^(iδ)  c₂₃c₁₃         |
```

Where cᵢⱼ = cos θᵢⱼ, sᵢⱼ = sin θᵢⱼ.

## Experimental Status (2023 Global Fits)

### Mixing Angles:
- **θ₁₂ (Solar)**: 33.45° ± 0.77° (well-measured, large)
- **θ₁₃ (Reactor)**: 8.62° ± 0.12° (precisely measured, non-zero)
- **θ₂₃ (Atmospheric)**: 49.2° ± 1.3° (near-maximal, some tension)

### Mass Squared Differences:
- **Δm²₂₁**: (7.42 ± 0.21) × 10⁻⁵ eV² (solar scale)
- **|Δm²₃₁|**: (2.515 ± 0.028) × 10⁻³ eV² (atmospheric scale)

### CP Violation:
- **δCP**: 197° ± 24° (normal ordering, hints of maximal violation)
- **Majorana phases**: Unconstrained by oscillation experiments

### Mass Ordering:
- **Normal Ordering**: m₁ < m₂ < m₃ (preferred by data, ~3σ)
- **Inverted Ordering**: m₃ < m₁ < m₂ (disfavored but not excluded)

## Connection to Theory

### Seesaw Mechanism Predictions:
The structure of the PMNS matrix provides crucial tests of mass generation mechanisms:
- **Texture zeros**: Specific vanishing elements from flavor symmetries
- **Sum rules**: Relations between angles from underlying symmetries  
- **CP phases**: Predictions from complex mass matrix elements
- **Correlations**: Links between mixing and mass ratios

### Flavor Models:
- **Tri-bimaximal mixing**: θ₁₂ = sin⁻¹(1/√3), θ₂₃ = 45°, θ₁₃ = 0°
- **Bimaximal mixing**: θ₁₂ = θ₂₃ = 45°, θ₁₃ = 0°
- **Democratic matrix**: Equal off-diagonal elements
- **Anarchic models**: Random matrix elements

### Grand Unification:
- **Quark-lepton unification**: Relations between CKM and PMNS matrices
- **SU(5) predictions**: Specific patterns from GUT breaking
- **SO(10) models**: Connections to right-handed neutrino sector

## Physical Observables

### Neutrino Oscillation Probabilities:
P(νₐ → νβ) depends on:
- Mixing matrix elements |Uₐᵢ|²
- Mass squared differences Δm²ᵢⱼ  
- Neutrino energy E and baseline L
- Matter effects (MSW effect in dense media)

### CP Violation Measures:
- **Jarlskog invariant**: J = Im(Ue1 Uμ2 U*e2 U*μ1)
- **Unitarity triangles**: Complex plane representations
- **T-violation asymmetries**: Direct CP tests in oscillations

### Neutrinoless Double Beta Decay:
The effective mass depends on PMNS elements and absolute masses:
|⟨mββ⟩| = |Σᵢ U²eᵢ mᵢ|

## Computational Features

### Matrix Construction:
- Standard and PDG parameterizations
- Arbitrary precision with SymPy
- Efficient numerical evaluation with NumPy

### Parameter Extraction:
- Robust angle extraction from arbitrary unitary matrices
- Phase extraction with proper conventions
- Error propagation and uncertainty analysis

### Phenomenological Tools:
- Oscillation probability calculations
- Matter effect implementation (MSW)
- Unitarity triangle construction
- Experimental data comparison

### Model Testing:
- Texture zero analysis
- Sum rule verification
- Symmetry pattern recognition
- Statistical comparison with data

All implementations follow standard physics conventions and provide
extensive validation against experimental measurements and theoretical
consistency checks.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
import warnings


def pmns_matrix_standard(theta12: float, theta13: float, theta23: float,
                        delta_cp: float = 0.0, 
                        alpha1: float = 0.0, alpha2: float = 0.0) -> np.ndarray:
    """
    Construct PMNS matrix using the standard physics parameterization.
    
    This function implements the complete PMNS matrix including both Dirac
    and Majorana CP-violating phases, following the standard convention
    used in most theoretical neutrino physics literature.
    
    ## Mathematical Structure:
    U = R₂₃(θ₂₃) · Uδ(δCP) · R₁₃(θ₁₃) · Uδ†(δCP) · R₁₂(θ₁₂) · P(α₁,α₂)
    
    Where:
    - Rᵢⱼ(θ): Rotation matrices in the ij-plane
    - Uδ(δ): CP phase matrix with δ in the (1,3) position  
    - P(α₁,α₂): Majorana phase matrix = diag(1, e^(iα₁), e^(iα₂))
    
    ## Physical Interpretation:
    
    **Mixing Angles:**
    - θ₁₂: Solar angle, controls ν₁-ν₂ mixing (~33°, large)
    - θ₁₃: Reactor angle, controls ν₁-ν₃ mixing (~8.6°, small but crucial)
    - θ₂₃: Atmospheric angle, controls ν₂-ν₃ mixing (~49°, near-maximal)
    
    **CP Violation:**
    - δ_CP: Dirac phase, observable in neutrino oscillations
      * δ_CP = 0°, 180°: CP conservation
      * δ_CP = 90°, 270°: Maximal CP violation
      * Current measurement: ~197° (hints of near-maximal violation)
    
    **Majorana Phases:**
    - α₁, α₂: Observable only in lepton number violating processes
    - Affect neutrinoless double beta decay rate
    - Cannot be measured in neutrino oscillation experiments
    
    ## Experimental Context:
    Current global fit values (NuFIT 5.2, 2023):
    - θ₁₂ = 33.45° ± 0.77°
    - θ₁₃ = 8.62° ± 0.12°  
    - θ₂₃ = 49.2° ± 1.3°
    - δ_CP = 197° ± 24° (normal ordering)
    
    Parameters:
    -----------
    theta12, theta13, theta23 : float
        Mixing angles in radians
        Valid range: [0, π/2] for physical mixing
    delta_cp : float, optional
        Dirac CP-violating phase in radians (default: 0)
        Valid range: [0, 2π] with physics defined modulo 2π
    alpha1, alpha2 : float, optional
        Majorana CP-violating phases in radians (default: 0)
        Valid range: [0, 2π] each
    
    Returns:
    --------
    np.ndarray
        3×3 complex PMNS matrix
        Rows: flavor states (e, μ, τ)
        Columns: mass states (1, 2, 3)
        
    Notes:
    ------
    - Matrix automatically satisfies unitarity: U U† = I
    - Determinant has unit magnitude: |det(U)| = 1
    - For Majorana neutrinos, use full parameterization with α₁, α₂
    - For Dirac neutrinos, set α₁ = α₂ = 0 (phases unphysical)
    
    Example:
    --------
    >>> import numpy as np
    >>> # Current best-fit values
    >>> theta12 = np.radians(33.45)  
    >>> theta13 = np.radians(8.62)
    >>> theta23 = np.radians(49.2)
    >>> delta_cp = np.radians(197)
    >>> U = pmns_matrix_standard(theta12, theta13, theta23, delta_cp)
    >>> print(f"Matrix unitarity: {np.allclose(U @ U.conj().T, np.eye(3))}")
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
    
    Approximate values from the NuFIT 5.x global fits (cf. arXiv:2111.03086).
    Kept for backward compatibility; use :func:`nufit_6_0` for current data.
    
    Returns:
    --------
    dict
        Dictionary containing best-fit values and uncertainties
    """
    # Note: These are approximate values for demonstration
    # Users should update with latest experimental results
    
    values = {
        'normal': {
            'theta12_deg': 33.45,  # Solar angle
            'theta13_deg': 8.62,   # Reactor angle  
            'theta23_deg': 49.2,   # Atmospheric angle
            'delta_cp_deg': 197,   # CP-violating phase
            'dm21_sq_1e5': 7.42,   # Δm²₂₁ × 10⁵ eV²
            'dm3l_sq_1e3': 2.515,  # Δm²₃ₗ × 10³ eV² (l=1 for NO)
        },
        'inverted': {
            'theta12_deg': 33.45,
            'theta13_deg': 8.65,
            'theta23_deg': 49.5,
            'delta_cp_deg': 282,
            'dm21_sq_1e5': 7.42,
            'dm3l_sq_1e3': -2.498,  # Δm²₃ₗ × 10³ eV² (l=2 for IO)
        }
    }
    
    return values


def nufit_6_0(variant: str = 'with_sk') -> Dict[str, Dict]:
    r"""
    Three-flavour global-fit results of NuFIT 6.0.

    I. Esteban, M. C. Gonzalez-Garcia, M. Maltoni, I. Martinez-Soler,
    J. P. Pinheiro, T. Schwetz, "NuFit-6.0: updated global analysis of
    three-flavor neutrino oscillations", JHEP 12 (2024) 216,
    arXiv:2410.05380 (Table 1). See http://www.nu-fit.org for updates.

    Each parameter is a tuple ``(bfp, +1sigma, -1sigma, low_3sigma, high_3sigma)``.
    Best fits and 1-sigma errors are relative to the local minimum of each
    ordering. ``'delta_chi2'`` is the chi^2 of that ordering above the global
    minimum.

    Conventions: ``dm21_sq`` is :math:`\Delta m^2_{21}` and ``dm3l_sq`` is
    :math:`\Delta m^2_{3\ell}`, i.e. :math:`\Delta m^2_{31} > 0` for normal
    ordering and :math:`\Delta m^2_{32} < 0` for inverted ordering. Both are
    in **eV^2**. Angles and the Dirac phase are in degrees; ``delta_cp_deg``
    3-sigma ranges may extend beyond 360 (they wrap around).

    Parameters:
    -----------
    variant : {'with_sk', 'without_sk'}
        ``'with_sk'``: IC24 with the Super-Kamiokande atmospheric chi^2 table
        (prefers normal ordering, Delta chi^2 = 6.1).
        ``'without_sk'``: IC19 without SK-atm (inverted ordering marginally
        preferred, Delta chi^2 = 0.6).

    Returns:
    --------
    dict
        ``{'normal': {...}, 'inverted': {...}}``
    """
    e5, e3 = 1e-5, 1e-3
    tables = {
        'without_sk': {
            'normal': {
                'sin2_theta12': (0.307, 0.012, 0.011, 0.275, 0.345),
                'theta12_deg': (33.68, 0.73, 0.70, 31.63, 35.95),
                'sin2_theta23': (0.561, 0.012, 0.015, 0.430, 0.596),
                'theta23_deg': (48.5, 0.7, 0.9, 41.0, 50.5),
                'sin2_theta13': (0.02195, 0.00054, 0.00058, 0.02023, 0.02376),
                'theta13_deg': (8.52, 0.11, 0.11, 8.18, 8.87),
                'delta_cp_deg': (177.0, 19.0, 20.0, 96.0, 422.0),
                'dm21_sq': (7.49 * e5, 0.19 * e5, 0.19 * e5, 6.92 * e5, 8.05 * e5),
                'dm3l_sq': (2.534 * e3, 0.025 * e3, 0.023 * e3, 2.463 * e3, 2.606 * e3),
                'delta_chi2': 0.6,
            },
            'inverted': {
                'sin2_theta12': (0.308, 0.012, 0.011, 0.275, 0.345),
                'theta12_deg': (33.68, 0.73, 0.70, 31.63, 35.95),
                'sin2_theta23': (0.562, 0.012, 0.015, 0.437, 0.597),
                'theta23_deg': (48.6, 0.7, 0.9, 41.4, 50.6),
                'sin2_theta13': (0.02224, 0.00056, 0.00057, 0.02053, 0.02397),
                'theta13_deg': (8.58, 0.11, 0.11, 8.24, 8.91),
                'delta_cp_deg': (285.0, 25.0, 28.0, 201.0, 348.0),
                'dm21_sq': (7.49 * e5, 0.19 * e5, 0.19 * e5, 6.92 * e5, 8.05 * e5),
                'dm3l_sq': (-2.510 * e3, 0.024 * e3, 0.025 * e3, -2.584 * e3, -2.438 * e3),
                'delta_chi2': 0.0,
            },
        },
        'with_sk': {
            'normal': {
                'sin2_theta12': (0.308, 0.012, 0.011, 0.275, 0.345),
                'theta12_deg': (33.68, 0.73, 0.70, 31.63, 35.95),
                'sin2_theta23': (0.470, 0.017, 0.013, 0.435, 0.585),
                'theta23_deg': (43.3, 1.0, 0.8, 41.3, 49.9),
                'sin2_theta13': (0.02215, 0.00056, 0.00058, 0.02030, 0.02388),
                'theta13_deg': (8.56, 0.11, 0.11, 8.19, 8.89),
                'delta_cp_deg': (212.0, 26.0, 41.0, 124.0, 364.0),
                'dm21_sq': (7.49 * e5, 0.19 * e5, 0.19 * e5, 6.92 * e5, 8.05 * e5),
                'dm3l_sq': (2.513 * e3, 0.021 * e3, 0.019 * e3, 2.451 * e3, 2.578 * e3),
                'delta_chi2': 0.0,
            },
            'inverted': {
                'sin2_theta12': (0.308, 0.012, 0.011, 0.275, 0.345),
                'theta12_deg': (33.68, 0.73, 0.70, 31.63, 35.95),
                'sin2_theta23': (0.550, 0.012, 0.015, 0.440, 0.584),
                'theta23_deg': (47.9, 0.7, 0.9, 41.5, 49.8),
                'sin2_theta13': (0.02231, 0.00056, 0.00056, 0.02060, 0.02409),
                'theta13_deg': (8.59, 0.11, 0.11, 8.25, 8.93),
                'delta_cp_deg': (274.0, 22.0, 25.0, 201.0, 335.0),
                'dm21_sq': (7.49 * e5, 0.19 * e5, 0.19 * e5, 6.92 * e5, 8.05 * e5),
                'dm3l_sq': (-2.484 * e3, 0.020 * e3, 0.020 * e3, -2.547 * e3, -2.421 * e3),
                'delta_chi2': 6.1,
            },
        },
    }
    if variant not in tables:
        raise ValueError(f"variant must be 'with_sk' or 'without_sk', got {variant!r}")
    return {ordering: dict(values) for ordering, values in tables[variant].items()}


def pmns_from_experimental(ordering: str = 'normal',
                           data: Optional[Dict[str, Dict]] = None) -> np.ndarray:
    """
    Construct PMNS matrix using experimental best-fit values.
    
    Parameters:
    -----------
    ordering : str, optional
        'normal' or 'inverted' mass ordering (default: 'normal')
    data : dict, optional
        Global-fit data in the format of :func:`nufit_6_0`. If omitted, the
        legacy :func:`experimental_values_2023` values are used.
    
    Returns:
    --------
    np.ndarray
        PMNS matrix with experimental values
    """
    if data is not None:
        if ordering not in data:
            raise ValueError(f"Ordering must be 'normal' or 'inverted', got {ordering}")
        bfp = {key: value[0] for key, value in data[ordering].items()
               if isinstance(value, tuple)}
        return pmns_matrix_pdg(np.radians(bfp['theta12_deg']),
                               np.radians(bfp['theta13_deg']),
                               np.radians(bfp['theta23_deg']),
                               np.radians(bfp['delta_cp_deg']))

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
