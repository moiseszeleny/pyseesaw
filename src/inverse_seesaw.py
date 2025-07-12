"""
Implementation of the Inverse Seesaw mechanism for neutrino mass generation.

The Inverse Seesaw mechanism provides an alternative way to generate
small neutrino masses with naturally small lepton number violation,
characterized by a small parameter μ rather than large mass scales.
Uses SymPy for symbolic analysis to demonstrate the mechanism step-by-step.
"""

import numpy as np
import sympy as sp
from typing import Tuple, Optional, Union, Dict
from .matrix_utils import (diagonalize_mass_matrix, mass_squared_differences,
                          create_symbolic_matrix, inverse_seesaw_symbolic,
                          substitute_numerical_values)


class SymbolicInverseSeesaw:
    """
    Symbolic Inverse Seesaw mechanism for pedagogical analysis.
    
    This class demonstrates how the Inverse Seesaw works symbolically,
    showing the role of the small μ parameter and how it differs
    from the standard Seesaw mechanism.
    """
    
    def __init__(self, n_generations: int = 3, n_sterile: int = 3):
        """
        Initialize symbolic Inverse Seesaw setup.
        
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
        self.mu_sym = create_symbolic_matrix('mu', (n_sterile, n_sterile), 
                                           real=True, symmetric=True)
        
        # Define symbolic parameters
        self.v = sp.Symbol('v', real=True, positive=True)  # Higgs VEV
        self.Lambda = sp.Symbol('Lambda', real=True, positive=True)  # Heavy scale
        self.mu_scale = sp.Symbol('mu_scale', real=True, positive=True)  # LNV scale
        self.y_D = sp.Symbol('y_D', real=True)  # Dirac Yukawa coupling
        
        # Cache
        self._light_mass_symbolic = None
        self._full_mass_matrix_symbolic = None
    
    def full_mass_matrix_symbolic(self) -> sp.Matrix:
        """
        Construct the full symbolic mass matrix in (ν_L, N_R, S_L) basis.
        
        Returns:
        --------
        sp.Matrix
            Full symbolic mass matrix
        """
        if self._full_mass_matrix_symbolic is None:
            # Block structure
            n_gen = self.n_gen
            n_st = self.n_sterile
            total_dim = n_gen + 2 * n_st
            
            # Initialize full matrix
            M_full = sp.zeros(total_dim, total_dim)
            
            # Fill blocks:
            # | 0    m_D   0  |
            # | m_D^T  0   M_R|
            # | 0    M_R^T μ  |
            
            # Upper right block: m_D
            for i in range(n_gen):
                for j in range(n_st):
                    M_full[i, n_gen + j] = self.m_D_sym[i, j]
            
            # Lower left block: m_D^T
            for i in range(n_st):
                for j in range(n_gen):
                    M_full[n_gen + i, j] = self.m_D_sym[j, i]
            
            # Upper right of (N_R, S_L) block: M_R
            for i in range(n_st):
                for j in range(n_st):
                    M_full[n_gen + i, n_gen + n_st + j] = self.M_R_sym[i, j]
            
            # Lower left of (N_R, S_L) block: M_R^T
            for i in range(n_st):
                for j in range(n_st):
                    M_full[n_gen + n_st + i, n_gen + j] = self.M_R_sym[j, i]
            
            # Lower right block: μ
            for i in range(n_st):
                for j in range(n_st):
                    M_full[n_gen + n_st + i, n_gen + n_st + j] = self.mu_sym[i, j]
            
            self._full_mass_matrix_symbolic = M_full
        
        return self._full_mass_matrix_symbolic
    
    def light_mass_matrix_symbolic(self) -> sp.Matrix:
        """
        Calculate symbolic light neutrino mass matrix using analytic formula.
        
        Returns:
        --------
        sp.Matrix
            Symbolic light neutrino mass matrix
        """
        if self._light_mass_symbolic is None:
            self._light_mass_symbolic = inverse_seesaw_symbolic(
                self.m_D_sym, self.M_R_sym, self.mu_sym)
        
        return self._light_mass_symbolic
    
    def scaling_analysis(self) -> Dict[str, sp.Expr]:
        """
        Perform dimensional analysis of Inverse Seesaw scaling.
        
        Returns:
        --------
        dict
            Dictionary with scaling relations
        """
        # Parametrize matrices in terms of fundamental scales
        m_D_scaled = self.y_D * self.v
        M_R_scaled = self.Lambda
        mu_scaled = self.mu_scale
        
        # Light neutrino mass scale: m_D^2 * μ / M_R^2
        m_nu_scale = (m_D_scaled**2 * mu_scaled) / (M_R_scaled**2)
        m_nu_scale = m_nu_scale.simplify()
        
        # Naturalness parameter
        naturalness = mu_scaled / M_R_scaled
        
        # Comparison with Type I Seesaw
        seesaw_type_i_scale = (m_D_scaled**2) / M_R_scaled
        ratio_to_seesaw = m_nu_scale / seesaw_type_i_scale
        ratio_to_seesaw = ratio_to_seesaw.simplify()
        
        return {
            'light_scale': m_nu_scale,
            'heavy_scale': M_R_scaled,
            'dirac_scale': m_D_scaled,
            'lnv_scale': mu_scaled,
            'naturalness': naturalness,
            'ratio_to_seesaw': ratio_to_seesaw,
            'hierarchy': M_R_scaled / mu_scaled
        }
    
    def simplified_minimal_case(self) -> Dict[str, sp.Expr]:
        """
        Analyze the minimal 3+2 Inverse Seesaw case.
        
        Returns:
        --------
        dict
            Analytical results for minimal case
        """
        # Define minimal case: 3 active + 2 sterile
        m_D_min = sp.Matrix([
            [sp.Symbol('m_D1'), 0],
            [0, sp.Symbol('m_D2')],
            [0, 0]
        ])
        
        M_R_min = sp.Matrix([
            [sp.Symbol('M_1'), 0],
            [0, sp.Symbol('M_2')]
        ])
        
        mu_min = sp.Matrix([
            [sp.Symbol('mu_1'), 0],
            [0, sp.Symbol('mu_2')]
        ])
        
        # Calculate light mass matrix
        m_nu_min = inverse_seesaw_symbolic(m_D_min, M_R_min, mu_min)
        m_nu_min = m_nu_min.simplify()
        
        return {
            'mass_matrix': m_nu_min,
            'trace': m_nu_min.trace(),
            'determinant': m_nu_min.det()
        }
    
    def mu_dependence_analysis(self) -> Dict[str, sp.Expr]:
        """
        Analyze how masses depend on the μ parameter.
        
        Returns:
        --------
        dict
            Analysis of μ dependence
        """
        m_light = self.light_mass_matrix_symbolic()
        
        # For diagonal μ = μ_0 * I, analyze scaling
        mu_0 = sp.Symbol('mu_0', real=True, positive=True)
        mu_diag = mu_0 * sp.eye(self.n_sterile)
        
        # Substitute diagonal μ
        m_light_diag = m_light.subs(self.mu_sym, mu_diag)
        
        # Mass scales linearly with μ_0
        scaling_factor = sp.Symbol('scale_factor')
        for i in range(self.n_gen):
            for j in range(self.n_gen):
                element = m_light_diag[i, j]
                if element != 0:
                    # Extract μ_0 dependence
                    mu_power = sp.degree(element, mu_0)
                    break
        
        return {
            'mu_diagonal_matrix': m_light_diag,
            'linear_scaling': 'Masses scale linearly with μ',
            'mu_power': 1  # Always linear in μ
        }
    
    def comparison_with_seesaw(self) -> Dict[str, sp.Expr]:
        """
        Symbolic comparison with Type I Seesaw.
        
        Returns:
        --------
        dict
            Comparison results
        """
        from .matrix_utils import seesaw_approximation_symbolic
        
        # Type I Seesaw result
        m_seesaw = seesaw_approximation_symbolic(self.m_D_sym, self.M_R_sym)
        
        # Inverse Seesaw result
        m_inverse = self.light_mass_matrix_symbolic()
        
        # Ratio (element-wise, symbolic)
        ratio_matrix = sp.zeros(*m_inverse.shape)
        for i in range(m_inverse.rows):
            for j in range(m_inverse.cols):
                if m_seesaw[i, j] != 0:
                    ratio_matrix[i, j] = m_inverse[i, j] / m_seesaw[i, j]
                    ratio_matrix[i, j] = ratio_matrix[i, j].simplify()
        
        return {
            'seesaw_matrix': m_seesaw,
            'inverse_seesaw_matrix': m_inverse,
            'ratio_matrix': ratio_matrix
        }
    
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


class InverseSeesaw:
    """
    Inverse Seesaw mechanism implementation.
    
    The mass matrix in the (ν_L, N_R, S_L) basis is:
    
    M = | 0    m_D   0  |
        | m_D^T  0   M_R|
        | 0    M_R^T μ  |
    
    where μ << M_R is the small lepton number violating parameter.
    """
    
    def __init__(self, 
                 dirac_mass: np.ndarray, 
                 majorana_mass: np.ndarray, 
                 mu_parameter: Union[float, np.ndarray]):
        """
        Initialize Inverse Seesaw mechanism.
        
        Parameters:
        -----------
        dirac_mass : np.ndarray
            Dirac mass matrix m_D (n_generations × n_sterile)
        majorana_mass : np.ndarray
            Majorana mass matrix M_R (n_sterile × n_sterile)
        mu_parameter : float or np.ndarray
            Small lepton number violating parameter μ
            If float, assumes μ * I; if array, uses as μ matrix
        """
        self.m_D = np.array(dirac_mass, dtype=complex)
        self.M_R = np.array(majorana_mass, dtype=complex)
        
        # Handle μ parameter
        if np.isscalar(mu_parameter):
            self.mu = mu_parameter * np.eye(self.M_R.shape[0], dtype=complex)
        else:
            self.mu = np.array(mu_parameter, dtype=complex)
        
        # Validate dimensions
        n_generations = self.m_D.shape[0]
        n_sterile = self.m_D.shape[1]
        
        if self.M_R.shape != (n_sterile, n_sterile):
            raise ValueError(f"M_R must be {n_sterile}×{n_sterile}")
        
        if self.mu.shape != (n_sterile, n_sterile):
            raise ValueError(f"μ must be {n_sterile}×{n_sterile}")
        
        self.n_generations = n_generations
        self.n_sterile = n_sterile
        
        # Cache for computed matrices
        self._full_mass_matrix = None
        self._light_mass_matrix = None
    
    def full_mass_matrix(self) -> np.ndarray:
        """
        Construct the full mass matrix in the (ν_L, N_R, S_L) basis.
        
        Returns:
        --------
        np.ndarray
            Full mass matrix of dimension (2*n_sterile + n_generations)
        """
        if self._full_mass_matrix is None:
            n_gen = self.n_generations
            n_st = self.n_sterile
            total_dim = n_gen + 2 * n_st
            
            M_full = np.zeros((total_dim, total_dim), dtype=complex)
            
            # Block structure:
            # | 0    m_D   0  |
            # | m_D^T  0   M_R|
            # | 0    M_R^T μ  |
            
            # Upper right block: m_D
            M_full[:n_gen, n_gen:n_gen+n_st] = self.m_D
            
            # Lower left block: m_D^T
            M_full[n_gen:n_gen+n_st, :n_gen] = self.m_D.T
            
            # Upper right of (N_R, S_L) block: M_R
            M_full[n_gen:n_gen+n_st, n_gen+n_st:] = self.M_R
            
            # Lower left of (N_R, S_L) block: M_R^T
            M_full[n_gen+n_st:, n_gen:n_gen+n_st] = self.M_R.T
            
            # Lower right block: μ
            M_full[n_gen+n_st:, n_gen+n_st:] = self.mu
            
            self._full_mass_matrix = M_full
        
        return self._full_mass_matrix
    
    def light_neutrino_mass_matrix_analytic(self) -> np.ndarray:
        """
        Calculate the effective light neutrino mass matrix analytically.
        
        In the limit μ << M_R, the light neutrino mass matrix is:
        m_ν ≈ m_D M_R^(-1) μ (M_R^T)^(-1) m_D^T
        
        Returns:
        --------
        np.ndarray
            Effective light neutrino mass matrix
        """
        if self._light_mass_matrix is None:
            M_R_inv = np.linalg.inv(self.M_R)
            M_R_T_inv = np.linalg.inv(self.M_R.T)
            
            self._light_mass_matrix = self.m_D @ M_R_inv @ self.mu @ M_R_T_inv @ self.m_D.T
        
        return self._light_mass_matrix
    
    def diagonalize_full_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the full mass matrix to get all eigenvalues and eigenvectors.
        
        Returns:
        --------
        masses : np.ndarray
            All mass eigenvalues (light + heavy)
        mixing_matrix : np.ndarray
            Full mixing matrix
        """
        M_full = self.full_mass_matrix()
        return diagonalize_mass_matrix(M_full, symmetric=True)
    
    def diagonalize_light_sector(self, use_analytic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the light neutrino sector.
        
        Parameters:
        -----------
        use_analytic : bool, optional
            Whether to use analytic approximation (default: True)
            If False, extracts light sector from full diagonalization
        
        Returns:
        --------
        light_masses : np.ndarray
            Light neutrino mass eigenvalues
        light_mixing : np.ndarray
            Light neutrino mixing matrix
        """
        if use_analytic:
            m_light = self.light_neutrino_mass_matrix_analytic()
            return diagonalize_mass_matrix(m_light, symmetric=True)
        else:
            all_masses, all_mixing = self.diagonalize_full_matrix()
            
            # Extract light modes (smallest masses)
            light_indices = np.argsort(np.abs(all_masses))[:self.n_generations]
            light_masses = all_masses[light_indices]
            light_mixing = all_mixing[:self.n_generations, light_indices]
            
            return light_masses, light_mixing
    
    def heavy_masses(self) -> np.ndarray:
        """
        Extract heavy neutrino masses from full diagonalization.
        
        Returns:
        --------
        np.ndarray
            Heavy neutrino mass eigenvalues
        """
        all_masses, _ = self.diagonalize_full_matrix()
        heavy_indices = np.argsort(np.abs(all_masses))[self.n_generations:]
        return all_masses[heavy_indices]
    
    def lepton_number_violation_scale(self) -> float:
        """
        Estimate the characteristic lepton number violation scale.
        
        Returns:
        --------
        float
            Characteristic scale of μ parameter
        """
        mu_eigenvalues = np.linalg.eigvals(self.mu)
        return np.sqrt(np.mean(np.abs(mu_eigenvalues)**2))
    
    def naturalness_parameter(self) -> float:
        """
        Calculate the naturalness parameter μ/M_R.
        
        Returns:
        --------
        float
            Ratio of μ scale to M_R scale
        """
        mu_scale = self.lepton_number_violation_scale()
        M_R_eigenvalues = np.linalg.eigvals(self.M_R)
        M_R_scale = np.sqrt(np.mean(np.abs(M_R_eigenvalues)**2))
        
        return mu_scale / M_R_scale


class InverseSeesawExtended:
    """
    Extended Inverse Seesaw with additional freedom in the structure.
    
    This class allows for more general structures in the Inverse Seesaw
    mechanism, including non-minimal choices of the μ matrix structure.
    """
    
    def __init__(self, 
                 dirac_mass: np.ndarray,
                 majorana_mass: np.ndarray,
                 mu_matrix: np.ndarray,
                 additional_couplings: Optional[np.ndarray] = None):
        """
        Initialize extended Inverse Seesaw mechanism.
        
        Parameters:
        -----------
        dirac_mass : np.ndarray
            Dirac mass matrix
        majorana_mass : np.ndarray
            Majorana mass matrix
        mu_matrix : np.ndarray
            General μ matrix (not necessarily proportional to identity)
        additional_couplings : np.ndarray, optional
            Additional coupling matrices for extended models
        """
        self.base_model = InverseSeesaw(dirac_mass, majorana_mass, mu_matrix)
        self.additional_couplings = additional_couplings
    
    def light_neutrino_mass_matrix(self) -> np.ndarray:
        """
        Calculate light neutrino mass matrix for extended model.
        
        Returns:
        --------
        np.ndarray
            Extended light neutrino mass matrix
        """
        base_mass = self.base_model.light_neutrino_mass_matrix_analytic()
        
        if self.additional_couplings is not None:
            # Add corrections from additional couplings
            # This is model-dependent and should be implemented based on specific models
            pass
        
        return base_mass


def compare_seesaw_vs_inverse_seesaw(dirac_mass: np.ndarray,
                                   majorana_mass: np.ndarray,
                                   mu_parameter: Union[float, np.ndarray]) -> dict:
    """
    Compare Type I Seesaw and Inverse Seesaw mechanisms.
    
    Parameters:
    -----------
    dirac_mass : np.ndarray
        Dirac mass matrix
    majorana_mass : np.ndarray
        Majorana/heavy mass matrix
    mu_parameter : float or np.ndarray
        Small parameter for Inverse Seesaw
    
    Returns:
    --------
    dict
        Comparison results including masses, mixing, and scales
    """
    from .seesaw import SeesawTypeI
    
    results = {}
    
    # Type I Seesaw
    seesaw_i = SeesawTypeI(dirac_mass, majorana_mass)
    masses_seesaw, mixing_seesaw = seesaw_i.diagonalize_light_sector()
    
    results['Seesaw_Type_I'] = {
        'light_masses': masses_seesaw,
        'mixing_matrix': mixing_seesaw,
        'mass_squared_differences': mass_squared_differences(masses_seesaw),
        'seesaw_scale': seesaw_i.seesaw_scale()
    }
    
    # Inverse Seesaw
    inverse_seesaw = InverseSeesaw(dirac_mass, majorana_mass, mu_parameter)
    masses_inverse, mixing_inverse = inverse_seesaw.diagonalize_light_sector()
    
    results['Inverse_Seesaw'] = {
        'light_masses': masses_inverse,
        'mixing_matrix': mixing_inverse,
        'mass_squared_differences': mass_squared_differences(masses_inverse),
        'mu_scale': inverse_seesaw.lepton_number_violation_scale(),
        'naturalness': inverse_seesaw.naturalness_parameter()
    }
    
    # Additional comparisons
    results['comparison'] = {
        'mass_ratio': np.abs(masses_seesaw / masses_inverse),
        'scale_ratio': results['Seesaw_Type_I']['seesaw_scale'] / results['Inverse_Seesaw']['mu_scale']
    }
    
    return results
