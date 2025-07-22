"""
Implementation of the Inverse Seesaw Mechanism for Neutrino Mass Generation

The Inverse Seesaw mechanism provides an alternative approach to generating small
neutrino masses that is more "natural" than the standard Seesaw mechanisms.
Instead of relying on extremely heavy mass scales, it uses a small parameter μ
that explicitly breaks lepton number by a tiny amount.

## Physics Motivation and Advantages

### Problems with Standard Seesaw:
- Requires extremely heavy masses M_R ~ 10¹⁰⁻¹⁶ GeV (untestable scales)
- Large hierarchy between electroweak and Seesaw scales
- Heavy neutrinos completely decouple from low-energy physics

### Inverse Seesaw Solution:
- Allows heavy neutrinos at TeV scale (potentially observable)
- Small neutrino masses from small lepton number violation μ
- More natural fine-tuning (linear in μ rather than quadratic in M_R)
- Richer phenomenology with light sterile states

## Physical Framework

### Extended Particle Content:
- Standard left-handed neutrinos νᴸ
- Right-handed neutrinos Nᴿ (can be at TeV scale)
- Additional sterile neutrinos S (singlets under Standard Model)

### Lagrangian Structure:
ℒ = -ν̄ᴸ m_D Nᴿ - ½(N̄ᴿ)ᶜ M_R Nᴿ - ½(N̄ᴿ)ᶜ μ Sᴸ - ½(S̄ᴸ)ᶜ μ† Nᴿ + h.c.

### Mass Matrix in (νᴸ, Nᴿ, Sᴸ) basis:
```
M = | 0     m_D    0   |
    | m_D^T   0    M_R |
    | 0     M_R^T   μ  |
```

### Key Parameters:
- **m_D**: Dirac masses (~ GeV scale, similar to quark/lepton masses)
- **M_R**: Right-handed masses (TeV scale, potentially accessible)  
- **μ**: Lepton number violation (keV scale, naturally small)

## Mathematical Structure and Approximations

### Flexible Fermion Content:
The Inverse Seesaw can accommodate different numbers of each fermion type:
- **n_left**: Number of left-handed neutrinos νL (usually 3 for SM)
- **n_right**: Number of right-handed neutrinos NR (can vary)  
- **n_singlet**: Number of singlet fermions S (can vary)

### Full Mass Matrix Dimensions:
The complete mass matrix has dimension (n_left + n_right + n_singlet):

**Examples:**
- **Standard case**: 3+3+3 → **9×9 matrix**
- **Minimal case**: 3+2+2 → **7×7 matrix**  
- **Extended case**: 3+4+3 → **10×10 matrix**

### Mass Matrix Structure:
```
         νL        NR         S
νL  |    0      m_D        0    |  
NR  |  m_D^T     0       M_R   |
S   |    0     M_R^T      μ    |
```

Where:
- **m_D**: Dirac masses (n_left × n_right)
- **M_R**: Heavy sector coupling (n_right × n_singlet)
- **μ**: Lepton number violation (n_singlet × n_singlet)

### Analytic Approximation (μ << M_R):
In the limit where μ is much smaller than other scales:

**Light neutrino masses:**
m_ν ≈ m_D M_R⁻¹ μ (M_R^T)⁻¹ m_D^T

**Heavy neutrino masses:**
M_heavy ≈ M_R (with small corrections ~ μ/M_R)

### Scaling Behavior:
- Light masses scale **linearly** with μ: m_ν ∝ μ
- This is more natural than standard Seesaw: m_ν ∝ 1/M_R
- Small μ ~ keV naturally gives sub-eV neutrino masses

## Physical Scales and Naturalness

### Typical Parameter Values:
- **Dirac masses m_D**: 0.1 - 10 GeV (electroweak scale)
- **Heavy masses M_R**: 100 GeV - 10 TeV (LHC accessible)
- **LNV parameter μ**: 1 keV - 1 MeV (naturally small)
- **Light masses m_ν**: 0.01 - 0.1 eV (observed range)

### Naturalness Argument:
The "naturalness parameter" η = μ/M_R can be small (η ~ 10⁻⁶) without
extreme fine-tuning, unlike standard Seesaw where m_D/M_R ~ 10⁻¹².

### Connection to Symmetries:
- μ = 0 corresponds to exact lepton number conservation
- Small μ represents small breaking of lepton number symmetry
- More theoretically motivated than large mass hierarchies

## Phenomenological Implications

### Neutrino Oscillations:
- Light neutrino mixing determined by interplay of all three mass matrices
- Can accommodate all observed oscillation data
- Potentially different predictions than standard Seesaw

### Heavy Neutrino Signatures:
- Heavy states at TeV scale potentially observable at colliders
- Modified W boson decays
- Neutrinoless double beta decay with different rates

### Sterile Neutrino Phenomenology:
- Light sterile states from μ-mixing
- Possible dark matter candidates
- Modified Big Bang nucleosynthesis

### Model Building Connections:
- Natural in left-right symmetric models
- Connection to radiative neutrino mass models
- Supersymmetric implementations

## Experimental Tests and Signatures

### Direct Searches:
- Heavy neutrino production at LHC
- Displaced vertex signatures from long-lived heavy neutrinos
- Modified electroweak precision measurements

### Indirect Constraints:
- Neutrinoless double beta decay (different matrix elements)
- Lepton flavor violation processes
- Cosmological constraints on extra relativistic degrees of freedom

### Future Prospects:
- Higher energy colliders for heavier M_R
- Precision measurements of light neutrino properties
- Improved neutrinoless double beta decay sensitivity

## Computational Features

This module provides comprehensive tools for Inverse Seesaw analysis:
- **Exact symbolic calculations**: Full 6×6 mass matrix treatment
- **Analytic approximations**: Fast evaluation in μ << M_R limit  
- **Numerical diagonalization**: Robust algorithms for realistic parameters
- **Parameter space studies**: Exploration of allowed regions
- **Phenomenological predictions**: Comparison with experimental data

All implementations handle the numerical challenges of the Inverse Seesaw:
- Multiple widely separated mass scales
- Near-singular matrices in certain limits
- Proper treatment of complex phases and unitarity
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
    
    The inverse seesaw involves three types of fermions:
    - Left-handed neutrinos νL (n_left generations)
    - Right-handed neutrinos NR (n_right generations)  
    - Singlet fermions S (n_singlet generations)
    
    The full mass matrix has dimension (n_left + n_right + n_singlet).
    """
    
    def __init__(self, n_left: int = 3, n_right: int = 3, n_singlet: int = 3):
        """
        Initialize symbolic Inverse Seesaw setup.
        
        Parameters:
        -----------
        n_left : int, optional
            Number of left-handed neutrino generations (default: 3)
        n_right : int, optional
            Number of right-handed neutrino generations (default: 3)
        n_singlet : int, optional
            Number of singlet fermion generations (default: 3)
        """
        self.n_left = n_left
        self.n_right = n_right
        self.n_singlet = n_singlet
        
        # Create symbolic matrices with proper dimensions
        # m_D: connects left and right neutrinos (n_left × n_right)
        self.m_D_sym = create_symbolic_matrix('m_D', (n_left, n_right))
        
        # M_R: connects right neutrinos and singlets (n_right × n_singlet)
        self.M_R_sym = create_symbolic_matrix('M_R', (n_right, n_singlet))
        
        # μ: singlet mass matrix (n_singlet × n_singlet)
        self.mu_sym = create_symbolic_matrix('mu', (n_singlet, n_singlet), 
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
        Construct the full symbolic mass matrix in (ν_L, N_R, S) basis.
        
        The mass matrix structure is:
        
        M = | 0     m_D    0   |  (n_left × n_left)     (n_left × n_right)    (n_left × n_singlet)
            | m_D^T   0    M_R  |  (n_right × n_left)   (n_right × n_right)   (n_right × n_singlet)
            | 0     M_R^T   μ   |  (n_singlet × n_left) (n_singlet × n_right) (n_singlet × n_singlet)
        
        Returns:
        --------
        sp.Matrix
            Full symbolic mass matrix of dimension (n_left + n_right + n_singlet)
        """
        if self._full_mass_matrix_symbolic is None:
            n_L = self.n_left
            n_R = self.n_right  
            n_S = self.n_singlet
            total_dim = n_L + n_R + n_S
            
            # Initialize full matrix
            M_full = sp.zeros(total_dim, total_dim)
            
            # Block structure in basis (ν_L, N_R, S):
            # Block (1,1): 0 (n_L × n_L)
            # Block (1,2): m_D (n_L × n_R)  
            # Block (1,3): 0 (n_L × n_S)
            # Block (2,1): m_D^T (n_R × n_L)
            # Block (2,2): 0 (n_R × n_R)
            # Block (2,3): M_R (n_R × n_S)
            # Block (3,1): 0 (n_S × n_L)
            # Block (3,2): M_R^T (n_S × n_R)
            # Block (3,3): μ (n_S × n_S)
            
            # Block (1,2): m_D
            for i in range(n_L):
                for j in range(n_R):
                    M_full[i, n_L + j] = self.m_D_sym[i, j]
            
            # Block (2,1): m_D^T  
            for i in range(n_R):
                for j in range(n_L):
                    M_full[n_L + i, j] = self.m_D_sym[j, i]
            
            # Block (2,3): M_R
            for i in range(n_R):
                for j in range(n_S):
                    M_full[n_L + i, n_L + n_R + j] = self.M_R_sym[i, j]
            
            # Block (3,2): M_R^T
            for i in range(n_S):
                for j in range(n_R):
                    M_full[n_L + n_R + i, n_L + j] = self.M_R_sym[j, i]
            
            # Block (3,3): μ
            for i in range(n_S):
                for j in range(n_S):
                    M_full[n_L + n_R + i, n_L + n_R + j] = self.mu_sym[i, j]
            
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
        Analyze a minimal Inverse Seesaw case with specific dimensions.
        
        Example: 3 left + 2 right + 2 singlet = 7×7 matrix
        
        Returns:
        --------
        dict
            Analytical results for minimal case
        """
        # Define minimal case dimensions
        n_L_min, n_R_min, n_S_min = 3, 2, 2
        
        # Define minimal matrices
        m_D_min = sp.Matrix([
            [sp.Symbol('m_D11'), sp.Symbol('m_D12')],
            [sp.Symbol('m_D21'), sp.Symbol('m_D22')],
            [sp.Symbol('m_D31'), sp.Symbol('m_D32')]
        ])
        
        M_R_min = sp.Matrix([
            [sp.Symbol('M_R11'), sp.Symbol('M_R12')],
            [sp.Symbol('M_R21'), sp.Symbol('M_R22')]
        ])
        
        mu_min = sp.Matrix([
            [sp.Symbol('mu_11'), sp.Symbol('mu_12')],
            [sp.Symbol('mu_12'), sp.Symbol('mu_22')]  # Symmetric
        ])
        
        # Calculate light mass matrix using generalized formula
        try:
            m_nu_min = inverse_seesaw_symbolic(m_D_min, M_R_min, mu_min)
            m_nu_min = m_nu_min.simplify()
            
            return {
                'mass_matrix': m_nu_min,
                'trace': m_nu_min.trace(),
                'determinant': m_nu_min.det(),
                'dimensions': f"{n_L_min}+{n_R_min}+{n_S_min} = {n_L_min + n_R_min + n_S_min}×{n_L_min + n_R_min + n_S_min}"
            }
        except Exception as e:
            return {
                'error': f"Could not compute minimal case: {e}",
                'dimensions': f"{n_L_min}+{n_R_min}+{n_S_min} = {n_L_min + n_R_min + n_S_min}×{n_L_min + n_R_min + n_S_min}"
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
        mu_diag = mu_0 * sp.eye(self.n_singlet)
        
        # Substitute diagonal μ
        m_light_diag = m_light.subs(self.mu_sym, mu_diag)
        
        return {
            'mu_diagonal_matrix': m_light_diag,
            'linear_scaling': 'Masses scale linearly with μ',
            'mu_power': 1,  # Always linear in μ
            'matrix_dimensions': f"Light neutrino matrix: {self.n_left}×{self.n_left}"
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
    Inverse Seesaw mechanism implementation with flexible fermion content.
    
    The mass matrix in the (ν_L, N_R, S) basis is:
    
    M = | 0     m_D    0   |
        | m_D^T   0    M_R  |
        | 0     M_R^T   μ   |
    
    where:
    - ν_L: left-handed neutrinos (n_left generations)
    - N_R: right-handed neutrinos (n_right generations)  
    - S: singlet fermions (n_singlet generations)
    - μ << M_R is the small lepton number violating parameter
    
    The full matrix has dimension (n_left + n_right + n_singlet).
    For example, 3+3+3 gives a 9×9 matrix.
    """
    
    def __init__(self, 
                 dirac_mass: np.ndarray, 
                 majorana_mass: np.ndarray, 
                 mu_parameter: Union[float, np.ndarray],
                 n_left: Optional[int] = None,
                 n_right: Optional[int] = None,
                 n_singlet: Optional[int] = None):
        """
        Initialize Inverse Seesaw mechanism.
        
        Parameters:
        -----------
        dirac_mass : np.ndarray
            Dirac mass matrix m_D connecting left and right neutrinos
            Shape: (n_left, n_right)
        majorana_mass : np.ndarray
            Mass matrix M_R connecting right neutrinos and singlets
            Shape: (n_right, n_singlet)
        mu_parameter : float or np.ndarray
            Small lepton number violating parameter μ
            If float, assumes μ * I; if array, shape must be (n_singlet, n_singlet)
        n_left : int, optional
            Number of left-handed neutrinos (inferred from dirac_mass if not given)
        n_right : int, optional  
            Number of right-handed neutrinos (inferred from matrices if not given)
        n_singlet : int, optional
            Number of singlet fermions (inferred from majorana_mass if not given)
        """
        self.m_D = np.array(dirac_mass, dtype=complex)
        self.M_R = np.array(majorana_mass, dtype=complex)
        
        # Infer dimensions from matrix shapes
        self.n_left = n_left if n_left is not None else self.m_D.shape[0]
        self.n_right = n_right if n_right is not None else self.m_D.shape[1]
        self.n_singlet = n_singlet if n_singlet is not None else self.M_R.shape[1]
        
        # Validate matrix dimensions
        if self.m_D.shape != (self.n_left, self.n_right):
            raise ValueError(f"m_D must be {self.n_left}×{self.n_right}, got {self.m_D.shape}")
        
        if self.M_R.shape != (self.n_right, self.n_singlet):
            raise ValueError(f"M_R must be {self.n_right}×{self.n_singlet}, got {self.M_R.shape}")
        
        # Handle μ parameter
        if np.isscalar(mu_parameter):
            self.mu = mu_parameter * np.eye(self.n_singlet, dtype=complex)
        else:
            self.mu = np.array(mu_parameter, dtype=complex)
            if self.mu.shape != (self.n_singlet, self.n_singlet):
                raise ValueError(f"μ must be {self.n_singlet}×{self.n_singlet}, got {self.mu.shape}")
        
        # Cache for computed matrices
        self._full_mass_matrix = None
        self._light_mass_matrix = None
    
    def full_mass_matrix(self) -> np.ndarray:
        """
        Construct the full mass matrix in the (ν_L, N_R, S) basis.
        
        Returns:
        --------
        np.ndarray
            Full mass matrix of dimension (n_left + n_right + n_singlet)
        """
        if self._full_mass_matrix is None:
            n_L = self.n_left
            n_R = self.n_right
            n_S = self.n_singlet
            total_dim = n_L + n_R + n_S
            
            M_full = np.zeros((total_dim, total_dim), dtype=complex)
            
            # Block structure in basis (ν_L, N_R, S):
            # | 0     m_D    0   |
            # | m_D^T   0    M_R  |
            # | 0     M_R^T   μ   |
            
            # Block (1,2): m_D
            M_full[:n_L, n_L:n_L+n_R] = self.m_D
            
            # Block (2,1): m_D^T  
            M_full[n_L:n_L+n_R, :n_L] = self.m_D.T
            
            # Block (2,3): M_R
            M_full[n_L:n_L+n_R, n_L+n_R:] = self.M_R
            
            # Block (3,2): M_R^T
            M_full[n_L+n_R:, n_L:n_L+n_R] = self.M_R.T
            
            # Block (3,3): μ
            M_full[n_L+n_R:, n_L+n_R:] = self.mu
            
            self._full_mass_matrix = M_full
        
        return self._full_mass_matrix
    
    def light_neutrino_mass_matrix_analytic(self) -> np.ndarray:
        """
        Calculate the effective light neutrino mass matrix analytically.
        
        For the inverse seesaw with general dimensions, the light neutrino 
        mass matrix is given by the Schur complement formula:
        
        m_ν ≈ m_D M_R^(-1) μ (M_R^T)^(-1) m_D^T
        
        Note: This assumes μ << M_R and requires M_R to be invertible.
        
        Returns:
        --------
        np.ndarray
            Effective light neutrino mass matrix (n_left × n_left)
        """
        if self._light_mass_matrix is None:
            # Check if M_R is square and invertible for the analytic formula
            if self.n_right != self.n_singlet:
                raise ValueError(
                    f"Analytic formula requires M_R to be square. "
                    f"Got {self.n_right}×{self.n_singlet}. "
                    f"Use diagonalize_light_sector(use_analytic=False) instead."
                )
            
            try:
                M_R_inv = np.linalg.inv(self.M_R)
                M_R_T_inv = np.linalg.inv(self.M_R.T)
                
                self._light_mass_matrix = self.m_D @ M_R_inv @ self.mu @ M_R_T_inv @ self.m_D.T
            except np.linalg.LinAlgError:
                raise ValueError("M_R is not invertible. Cannot use analytic approximation.")
        
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
            Note: Analytic formula only works when M_R is square (n_right = n_singlet)
        
        Returns:
        --------
        light_masses : np.ndarray
            Light neutrino mass eigenvalues
        light_mixing : np.ndarray
            Light neutrino mixing matrix
        """
        if use_analytic and self.n_right == self.n_singlet:
            try:
                m_light = self.light_neutrino_mass_matrix_analytic()
                return diagonalize_mass_matrix(m_light, symmetric=True)
            except (np.linalg.LinAlgError, ValueError):
                # Fall back to full diagonalization if analytic fails
                use_analytic = False
        
        if not use_analytic or self.n_right != self.n_singlet:
            all_masses, all_mixing = self.diagonalize_full_matrix()
            
            # Extract light modes (smallest masses)
            light_indices = np.argsort(np.abs(all_masses))[:self.n_left]
            light_masses = all_masses[light_indices]
            light_mixing = all_mixing[:self.n_left, light_indices]
            
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
        heavy_indices = np.argsort(np.abs(all_masses))[self.n_left:]
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
    
    def matrix_structure_info(self) -> Dict[str, any]:
        """
        Provide information about the matrix structure and dimensions.
        
        Returns:
        --------
        dict
            Information about matrix dimensions and structure
        """
        total_dim = self.n_left + self.n_right + self.n_singlet
        
        return {
            'fermion_content': {
                'left_neutrinos': self.n_left,
                'right_neutrinos': self.n_right, 
                'singlet_fermions': self.n_singlet
            },
            'matrix_dimensions': {
                'm_D': f"{self.n_left}×{self.n_right}",
                'M_R': f"{self.n_right}×{self.n_singlet}",
                'μ': f"{self.n_singlet}×{self.n_singlet}",
                'full_matrix': f"{total_dim}×{total_dim}"
            },
            'total_dimension': total_dim,
            'light_sector_dim': self.n_left,
            'heavy_sector_dim': self.n_right + self.n_singlet,
            'structure_summary': f"{self.n_left}+{self.n_right}+{self.n_singlet} = {total_dim}×{total_dim}"
        }
    
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


def demonstrate_matrix_structures():
    """
    Demonstrate how matrix dimensions change with different fermion content.
    
    Examples:
    - Standard 3+3+3: 9×9 matrix  
    - Minimal 3+2+2: 7×7 matrix
    - Extended 3+4+3: 10×10 matrix
    
    Returns:
    --------
    dict
        Examples of different matrix structures
    """
    examples = {}
    
    # Standard case: 3 of each type
    try:
        m_D_33 = np.random.rand(3, 3) * 1e-2
        M_R_33 = np.random.rand(3, 3) * 1e15  
        mu_33 = np.random.rand(3, 3) * 1e-6
        
        seesaw_33 = InverseSeesaw(m_D_33, M_R_33, mu_33)
        examples['3+3+3_case'] = {
            'description': "Standard: 3 left + 3 right + 3 singlet",
            'structure': seesaw_33.matrix_structure_info(),
            'full_matrix_shape': seesaw_33.full_mass_matrix().shape
        }
    except Exception as e:
        examples['3+3+3_case'] = {'error': str(e)}
    
    # Minimal case: 3+2+2  
    try:
        m_D_32 = np.random.rand(3, 2) * 1e-2
        M_R_22 = np.random.rand(2, 2) * 1e15
        mu_22 = np.random.rand(2, 2) * 1e-6
        
        seesaw_32 = InverseSeesaw(m_D_32, M_R_22, mu_22)
        examples['3+2+2_case'] = {
            'description': "Minimal: 3 left + 2 right + 2 singlet", 
            'structure': seesaw_32.matrix_structure_info(),
            'full_matrix_shape': seesaw_32.full_mass_matrix().shape
        }
    except Exception as e:
        examples['3+2+2_case'] = {'error': str(e)}
    
    # Non-square M_R case: 3+2+3
    try:
        m_D_32b = np.random.rand(3, 2) * 1e-2  
        M_R_23 = np.random.rand(2, 3) * 1e15
        mu_33b = np.random.rand(3, 3) * 1e-6
        
        seesaw_323 = InverseSeesaw(m_D_32b, M_R_23, mu_33b)
        examples['3+2+3_case'] = {
            'description': "Non-square M_R: 3 left + 2 right + 3 singlet",
            'structure': seesaw_323.matrix_structure_info(), 
            'full_matrix_shape': seesaw_323.full_mass_matrix().shape,
            'note': "Analytic formula not available (M_R not square)"
        }
    except Exception as e:
        examples['3+2+3_case'] = {'error': str(e)}
    
    return examples


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
