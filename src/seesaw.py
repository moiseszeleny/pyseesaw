"""
Implementation of various Seesaw mechanisms for neutrino mass generation.

This module implements the Type I, Type II, and Type III Seesaw mechanisms,
which explain the smallness of neutrino masses through the introduction
of heavy particles at high energy scales. Uses SymPy for symbolic analysis
and step-by-step approximations.
"""

import numpy as np
import sympy as sp
from typing import Tuple, Optional, Union, Dict
from .matrix_utils import (diagonalize_mass_matrix, mass_squared_differences,
                          create_symbolic_matrix, seesaw_approximation_symbolic,
                          substitute_numerical_values)


class SymbolicSeesawTypeI:
    """
    Symbolic Type I Seesaw mechanism for pedagogical analysis.
    
    This class demonstrates the Seesaw mechanism symbolically,
    showing step-by-step how the approximations work and allowing
    exploration of different limits and parameter regimes.
    """
    
    def __init__(self, n_generations: int = 3, n_sterile: int = 3):
        """
        Initialize symbolic Seesaw Type I setup.
        
        Parameters:
        -----------
        n_generations : int, optional
            Number of active neutrino generations (default: 3)
        n_sterile : int, optional
            Number of sterile neutrinos (default: 3)
        """
        self.n_gen = n_generations
        self.n_sterile = n_sterile
        
        # Create symbolic matrices
        self.m_D_sym = create_symbolic_matrix('m_D', (n_generations, n_sterile))
        self.M_R_sym = create_symbolic_matrix('M_R', (n_sterile, n_sterile), 
                                            real=True, symmetric=True)
        
        # Define symbolic parameters for scaling analysis
        self.v = sp.Symbol('v', real=True, positive=True)  # Higgs VEV
        self.Lambda = sp.Symbol('Lambda', real=True, positive=True)  # Heavy scale
        self.y_D = sp.Symbol('y_D', real=True)  # Dirac Yukawa coupling
        self.epsilon = sp.Symbol('epsilon', real=True, positive=True)  # Small parameter
        
        # Cache for computed expressions
        self._light_mass_symbolic = None
        self._full_mass_matrix_symbolic = None
    
    def set_texture(self, dirac_texture: Optional[list] = None, 
                   majorana_texture: Optional[list] = None):
        """
        Apply texture zeros to mass matrices.
        
        Parameters:
        -----------
        dirac_texture : list, optional
            List of (i,j) positions to set to zero in Dirac matrix
        majorana_texture : list, optional
            List of (i,j) positions to set to zero in Majorana matrix
        """
        if dirac_texture:
            from .matrix_utils import texture_zeros_matrix
            self.m_D_sym = texture_zeros_matrix(self.m_D_sym, dirac_texture)
        
        if majorana_texture:
            from .matrix_utils import texture_zeros_matrix
            self.M_R_sym = texture_zeros_matrix(self.M_R_sym, majorana_texture)
    
    def light_mass_matrix_symbolic(self, order: int = 1) -> sp.Matrix:
        """
        Get symbolic light neutrino mass matrix.
        
        Parameters:
        -----------
        order : int, optional
            Order of Seesaw approximation (default: 1)
        
        Returns:
        --------
        sp.Matrix
            Symbolic light neutrino mass matrix
        """
        if self._light_mass_symbolic is None or order > 1:
            self._light_mass_symbolic = seesaw_approximation_symbolic(
                self.m_D_sym, self.M_R_sym, order)
        
        return self._light_mass_symbolic
    
    def scaling_analysis(self) -> Dict[str, sp.Expr]:
        """
        Perform dimensional analysis of Seesaw scaling.
        
        Returns:
        --------
        dict
            Dictionary with scaling relations
        """
        # Parametrize matrices in terms of fundamental scales
        m_D_scaled = self.y_D * self.v
        M_R_scaled = self.Lambda
        
        # Light neutrino mass scale
        m_nu_scale = (m_D_scaled**2) / M_R_scaled
        m_nu_scale = m_nu_scale.simplify()
        
        # Seesaw ratio
        seesaw_ratio = m_nu_scale / m_D_scaled
        seesaw_ratio = seesaw_ratio.simplify()
        
        return {
            'light_scale': m_nu_scale,
            'heavy_scale': M_R_scaled,
            'dirac_scale': m_D_scaled,
            'seesaw_ratio': seesaw_ratio,
            'hierarchy': M_R_scaled / m_D_scaled
        }
    
    def simplified_2x2_case(self) -> Dict[str, sp.Expr]:
        """
        Analyze simplified 2×2 case for pedagogical purposes.
        
        Returns:
        --------
        dict
            Analytical results for 2×2 case
        """
        # Define 2×2 symbolic matrices
        m_D_2x2 = sp.Matrix([
            [sp.Symbol('m_D11'), sp.Symbol('m_D12')],
            [sp.Symbol('m_D21'), sp.Symbol('m_D22')]
        ])
        
        M_R_2x2 = sp.Matrix([
            [sp.Symbol('M_1'), 0],
            [0, sp.Symbol('M_2')]
        ])
        
        # Calculate Seesaw mass matrix
        M_R_inv = M_R_2x2.inv()
        m_nu_2x2 = -m_D_2x2 * M_R_inv * m_D_2x2.T
        m_nu_2x2 = m_nu_2x2.simplify()
        
        # Extract eigenvalues symbolically
        eigenvals = list(m_nu_2x2.eigenvals().keys())
        
        return {
            'mass_matrix': m_nu_2x2,
            'eigenvalues': eigenvals,
            'trace': m_nu_2x2.trace(),
            'determinant': m_nu_2x2.det()
        }
    
    def perturbative_expansion(self, small_parameter: sp.Symbol) -> sp.Matrix:
        """
        Expand Seesaw formula in powers of small parameter.
        
        Parameters:
        -----------
        small_parameter : sp.Symbol
            Small parameter for expansion (e.g., m_D/M_R)
        
        Returns:
        --------
        sp.Matrix
            Expanded mass matrix
        """
        from .matrix_utils import expand_in_small_parameter
        
        m_light = self.light_mass_matrix_symbolic()
        
        # Expand each matrix element
        expanded_matrix = sp.zeros(*m_light.shape)
        for i in range(m_light.rows):
            for j in range(m_light.cols):
                expanded_matrix[i, j] = expand_in_small_parameter(
                    m_light[i, j], small_parameter, order=2)
        
        return expanded_matrix
    
    def evaluate_numerically(self, parameter_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate symbolic expressions numerically.
        
        Parameters:
        -----------
        parameter_dict : dict
            Dictionary mapping symbols to numerical values
        
        Returns:
        --------
        tuple
            Numerical mass matrix and eigenvalues
        """
        m_light_sym = self.light_mass_matrix_symbolic()
        m_light_num = substitute_numerical_values(m_light_sym, parameter_dict)
        
        # Diagonalize numerically
        masses, mixing = diagonalize_mass_matrix(m_light_num, symmetric=True)
        
        return m_light_num, masses


class SeesawTypeI:
    """
    Type I Seesaw mechanism implementation.
    
    In Type I Seesaw, the light neutrino mass matrix is given by:
    m_ν = -m_D M_R⁻¹ m_D^T
    
    where m_D is the Dirac mass matrix and M_R is the right-handed
    Majorana mass matrix.
    """
    
    def __init__(self, dirac_mass: np.ndarray, majorana_mass: np.ndarray):
        """
        Initialize Type I Seesaw with Dirac and Majorana mass matrices.
        
        Parameters:
        -----------
        dirac_mass : np.ndarray
            Dirac neutrino mass matrix (typically 3x3)
        majorana_mass : np.ndarray
            Right-handed Majorana mass matrix (typically 3x3)
        """
        self.m_D = np.array(dirac_mass, dtype=complex)
        self.M_R = np.array(majorana_mass, dtype=complex)
        
        if self.m_D.shape[0] != self.M_R.shape[0]:
            raise ValueError("Dirac and Majorana matrices must have compatible dimensions")
        
        self._light_mass_matrix = None
        self._heavy_masses = None
    
    def light_neutrino_mass_matrix(self) -> np.ndarray:
        """
        Calculate the effective light neutrino mass matrix.
        
        Returns:
        --------
        np.ndarray
            Effective light neutrino mass matrix
        """
        if self._light_mass_matrix is None:
            M_R_inv = np.linalg.inv(self.M_R)
            self._light_mass_matrix = -self.m_D @ M_R_inv @ self.m_D.T
        
        return self._light_mass_matrix
    
    def diagonalize_light_sector(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the light neutrino mass matrix.
        
        Returns:
        --------
        light_masses : np.ndarray
            Light neutrino mass eigenvalues
        mixing_matrix : np.ndarray
            PMNS-like mixing matrix
        """
        m_light = self.light_neutrino_mass_matrix()
        return diagonalize_mass_matrix(m_light, symmetric=True)
    
    def heavy_neutrino_masses(self) -> np.ndarray:
        """
        Calculate the heavy neutrino masses (approximately M_R eigenvalues).
        
        Returns:
        --------
        np.ndarray
            Heavy neutrino mass eigenvalues
        """
        if self._heavy_masses is None:
            heavy_masses, _ = diagonalize_mass_matrix(self.M_R, symmetric=True)
            self._heavy_masses = heavy_masses
        
        return self._heavy_masses
    
    def seesaw_scale(self) -> float:
        """
        Estimate the characteristic Seesaw scale.
        
        Returns:
        --------
        float
            Geometric mean of heavy neutrino masses
        """
        heavy_masses = self.heavy_neutrino_masses()
        return np.exp(np.mean(np.log(heavy_masses)))


class SeesawTypeII:
    """
    Type II Seesaw mechanism implementation.
    
    In Type II Seesaw, the light neutrino mass matrix receives contributions
    from both Type I and Type II mechanisms:
    m_ν = m_L - m_D M_R⁻¹ m_D^T
    
    where m_L comes from the Higgs triplet VEV.
    """
    
    def __init__(self, 
                 triplet_mass: np.ndarray,
                 dirac_mass: Optional[np.ndarray] = None,
                 majorana_mass: Optional[np.ndarray] = None,
                 triplet_vev: float = 1e-3):
        """
        Initialize Type II Seesaw mechanism.
        
        Parameters:
        -----------
        triplet_mass : np.ndarray
            Triplet Yukawa coupling matrix
        dirac_mass : np.ndarray, optional
            Dirac mass matrix (for combined Type I+II)
        majorana_mass : np.ndarray, optional
            Right-handed Majorana mass matrix (for combined Type I+II)
        triplet_vev : float, optional
            Triplet Higgs VEV in GeV (default: 1e-3 GeV)
        """
        self.f = np.array(triplet_mass, dtype=complex)
        self.v_T = triplet_vev
        self.m_D = dirac_mass if dirac_mass is not None else None
        self.M_R = majorana_mass if majorana_mass is not None else None
        
        if self.m_D is not None and self.M_R is not None:
            self.include_type_i = True
        else:
            self.include_type_i = False
    
    def light_neutrino_mass_matrix(self) -> np.ndarray:
        """
        Calculate the effective light neutrino mass matrix.
        
        Returns:
        --------
        np.ndarray
            Effective light neutrino mass matrix
        """
        # Type II contribution
        m_light = self.v_T * self.f
        
        # Add Type I contribution if present
        if self.include_type_i:
            M_R_inv = np.linalg.inv(self.M_R)
            m_light -= self.m_D @ M_R_inv @ self.m_D.T
        
        return m_light
    
    def diagonalize_light_sector(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the light neutrino mass matrix.
        
        Returns:
        --------
        light_masses : np.ndarray
            Light neutrino mass eigenvalues
        mixing_matrix : np.ndarray
            PMNS-like mixing matrix
        """
        m_light = self.light_neutrino_mass_matrix()
        return diagonalize_mass_matrix(m_light, symmetric=True)


class SeesawTypeIII:
    """
    Type III Seesaw mechanism implementation.
    
    In Type III Seesaw, the light neutrino mass matrix is given by:
    m_ν = -m_D M_Σ⁻¹ m_D^T
    
    where M_Σ is the mass matrix of the fermionic triplets.
    """
    
    def __init__(self, dirac_mass: np.ndarray, triplet_mass: np.ndarray):
        """
        Initialize Type III Seesaw with Dirac and triplet mass matrices.
        
        Parameters:
        -----------
        dirac_mass : np.ndarray
            Dirac coupling matrix to fermionic triplets
        triplet_mass : np.ndarray
            Fermionic triplet mass matrix
        """
        self.m_D = np.array(dirac_mass, dtype=complex)
        self.M_Sigma = np.array(triplet_mass, dtype=complex)
        
        if self.m_D.shape[0] != self.M_Sigma.shape[0]:
            raise ValueError("Dirac and triplet matrices must have compatible dimensions")
    
    def light_neutrino_mass_matrix(self) -> np.ndarray:
        """
        Calculate the effective light neutrino mass matrix.
        
        Returns:
        --------
        np.ndarray
            Effective light neutrino mass matrix
        """
        M_Sigma_inv = np.linalg.inv(self.M_Sigma)
        return -self.m_D @ M_Sigma_inv @ self.m_D.T
    
    def diagonalize_light_sector(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the light neutrino mass matrix.
        
        Returns:
        --------
        light_masses : np.ndarray
            Light neutrino mass eigenvalues
        mixing_matrix : np.ndarray
            PMNS-like mixing matrix
        """
        m_light = self.light_neutrino_mass_matrix()
        return diagonalize_mass_matrix(m_light, symmetric=True)


def compare_seesaw_mechanisms(dirac_mass: np.ndarray,
                             majorana_mass: np.ndarray,
                             triplet_yukawa: Optional[np.ndarray] = None,
                             triplet_vev: float = 1e-3) -> dict:
    """
    Compare different Seesaw mechanisms with the same input parameters.
    
    Parameters:
    -----------
    dirac_mass : np.ndarray
        Dirac mass matrix
    majorana_mass : np.ndarray
        Heavy mass matrix
    triplet_yukawa : np.ndarray, optional
        Triplet Yukawa matrix for Type II
    triplet_vev : float, optional
        Triplet VEV for Type II
    
    Returns:
    --------
    dict
        Comparison results including masses and mixing for each mechanism
    """
    results = {}
    
    # Type I Seesaw
    seesaw_i = SeesawTypeI(dirac_mass, majorana_mass)
    masses_i, mixing_i = seesaw_i.diagonalize_light_sector()
    results['Type_I'] = {
        'masses': masses_i,
        'mixing_matrix': mixing_i,
        'mass_squared_differences': mass_squared_differences(masses_i)
    }
    
    # Type III Seesaw (same as Type I with different interpretation)
    seesaw_iii = SeesawTypeIII(dirac_mass, majorana_mass)
    masses_iii, mixing_iii = seesaw_iii.diagonalize_light_sector()
    results['Type_III'] = {
        'masses': masses_iii,
        'mixing_matrix': mixing_iii,
        'mass_squared_differences': mass_squared_differences(masses_iii)
    }
    
    # Type II Seesaw (if triplet Yukawa is provided)
    if triplet_yukawa is not None:
        seesaw_ii = SeesawTypeII(triplet_yukawa, dirac_mass, majorana_mass, triplet_vev)
        masses_ii, mixing_ii = seesaw_ii.diagonalize_light_sector()
        results['Type_II'] = {
            'masses': masses_ii,
            'mixing_matrix': mixing_ii,
            'mass_squared_differences': mass_squared_differences(masses_ii)
        }
    
    return results
