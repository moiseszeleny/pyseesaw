"""
Example: Symbolic Analysis of Seesaw Mechanisms

This script demonstrates how to use SymPy to analyze Seesaw mechanisms
symbolically, showing step-by-step approximations and the physics
behind the mass generation mechanisms.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from seesaw import SymbolicSeesawTypeI
from inverse_seesaw import SymbolicInverseSeesaw
from matrix_utils import create_symbolic_matrix, substitute_numerical_values

# Configure SymPy for nice output
sp.init_printing(use_latex=False, use_unicode=True)


def demonstrate_symbolic_seesaw():
    """Demonstrate symbolic Type I Seesaw mechanism."""
    
    print("=== Symbolic Type I Seesaw Analysis ===\n")
    
    # Create symbolic Seesaw model
    seesaw = SymbolicSeesawTypeI(n_generations=3, n_sterile=3)
    
    print("1. Basic Seesaw Formula")
    print("   The Type I Seesaw formula is: m_ν = -m_D M_R⁻¹ m_Dᵀ")
    print("   where m_D is the Dirac mass matrix and M_R is the Majorana mass matrix\n")
    
    # Get symbolic light mass matrix
    m_light_sym = seesaw.light_mass_matrix_symbolic()
    
    print("2. Symbolic Light Neutrino Mass Matrix:")
    print("   (showing first element as example)")
    print(f"   m_ν[1,1] = {m_light_sym[0,0]}")
    print(f"   m_ν[1,2] = {m_light_sym[0,1]}")
    print()
    
    # Scaling analysis
    print("3. Dimensional Analysis:")
    scaling = seesaw.scaling_analysis()
    
    print(f"   Light neutrino scale: {scaling['light_scale']}")
    print(f"   Heavy neutrino scale: {scaling['heavy_scale']}")
    print(f"   Seesaw ratio: {scaling['seesaw_ratio']}")
    print(f"   Mass hierarchy: {scaling['hierarchy']}")
    print()
    
    # 2×2 simplified case
    print("4. Simplified 2×2 Case (for pedagogical understanding):")
    case_2x2 = seesaw.simplified_2x2_case()
    
    print("   Mass matrix:")
    sp.pprint(case_2x2['mass_matrix'])
    print(f"\n   Trace: {case_2x2['trace']}")
    print(f"   Determinant: {case_2x2['determinant']}")
    print()
    
    # Numerical evaluation
    print("5. Numerical Evaluation:")
    
    # Define numerical values
    numerical_params = {}
    
    # Fill Dirac mass matrix elements (GeV)
    for i in range(3):
        for j in range(3):
            symbol_name = f'm_D_{i+1}{j+1}'
            if symbol_name in [str(s) for s in m_light_sym.free_symbols]:
                numerical_params[sp.Symbol(symbol_name)] = 0.01 * (1 + 0.1*i + 0.05*j)
    
    # Fill Majorana mass matrix elements (GeV)
    for i in range(3):
        for j in range(i, 3):  # Symmetric matrix
            symbol_name = f'M_R_{i+1}{j+1}'
            if symbol_name in [str(s) for s in m_light_sym.free_symbols]:
                value = 1e15 * (1.0 if i == j else 0.1)
                numerical_params[sp.Symbol(symbol_name)] = value
    
    try:
        m_light_num, masses_num = seesaw.evaluate_numerically(numerical_params)
        
        print(f"   Light neutrino masses (eV):")
        for i, mass in enumerate(masses_num):
            print(f"     m_{i+1} = {mass:.3e} eV")
        
        print(f"   Mass hierarchy: {masses_num[-1]/masses_num[0]:.1e}")
        
    except Exception as e:
        print(f"   Numerical evaluation failed: {e}")
    
    return seesaw


def demonstrate_symbolic_inverse_seesaw():
    """Demonstrate symbolic Inverse Seesaw mechanism."""
    
    print("\n=== Symbolic Inverse Seesaw Analysis ===\n")
    
    # Create symbolic Inverse Seesaw model
    inverse_seesaw = SymbolicInverseSeesaw(n_generations=3, n_sterile=3)
    
    print("1. Inverse Seesaw Formula")
    print("   The Inverse Seesaw formula is: m_ν = m_D M_R⁻¹ μ (M_Rᵀ)⁻¹ m_Dᵀ")
    print("   where μ is a small lepton number violating matrix\n")
    
    # Get symbolic light mass matrix
    m_light_sym = inverse_seesaw.light_mass_matrix_symbolic()
    
    print("2. Symbolic Light Neutrino Mass Matrix:")
    print("   (showing structure - first element)")
    print(f"   m_ν[1,1] = {m_light_sym[0,0]}")
    print()
    
    # Scaling analysis
    print("3. Dimensional Analysis:")
    scaling = inverse_seesaw.scaling_analysis()
    
    print(f"   Light neutrino scale: {scaling['light_scale']}")
    print(f"   LNV scale: {scaling['lnv_scale']}")
    print(f"   Naturalness parameter: {scaling['naturalness']}")
    print(f"   Ratio to Seesaw: {scaling['ratio_to_seesaw']}")
    print(f"   Mass hierarchy: {scaling['hierarchy']}")
    print()
    
    # μ dependence analysis
    print("4. μ Dependence Analysis:")
    mu_analysis = inverse_seesaw.mu_dependence_analysis()
    print(f"   Scaling: {mu_analysis['linear_scaling']}")
    print(f"   Power of μ: {mu_analysis['mu_power']}")
    print()
    
    # Minimal case
    print("5. Minimal Case Analysis:")
    minimal = inverse_seesaw.simplified_minimal_case()
    print("   Mass matrix (minimal 3+2 case):")
    sp.pprint(minimal['mass_matrix'])
    print()
    
    # Comparison with Seesaw
    print("6. Comparison with Type I Seesaw:")
    comparison = inverse_seesaw.comparison_with_seesaw()
    
    print("   The ratio of Inverse Seesaw to Type I Seesaw masses")
    print("   contains the factor μ/M_R, showing the role of the small LNV parameter")
    print()
    
    return inverse_seesaw


def parameter_scanning_symbolic():
    """Demonstrate parameter scanning with symbolic expressions."""
    
    print("=== Parameter Scanning with Symbolic Expressions ===\n")
    
    # Simple 2×2 case for clarity
    print("1. Setting up simplified 2×2 model")
    
    # Define symbolic parameters
    m_D = sp.Symbol('m_D', real=True, positive=True)
    M_R = sp.Symbol('M_R', real=True, positive=True)
    mu = sp.Symbol('mu', real=True, positive=True)
    
    # Seesaw mass
    m_seesaw = m_D**2 / M_R
    
    # Inverse Seesaw mass
    m_inverse = m_D**2 * mu / M_R**2
    
    print(f"   Type I Seesaw: m_ν ∼ {m_seesaw}")
    print(f"   Inverse Seesaw: m_ν ∼ {m_inverse}")
    print()
    
    # Analyze different regimes
    print("2. Different Physical Regimes:")
    
    # Define numerical ranges
    m_D_values = np.logspace(-3, -1, 50)  # 1 MeV to 100 MeV
    M_R_values = np.logspace(12, 16, 50)  # 1 TeV to 10^16 eV
    mu_values = np.logspace(-9, -3, 50)   # 1 neV to 1 keV
    
    # Convert to functions for evaluation
    m_seesaw_func = sp.lambdify((m_D, M_R), m_seesaw, 'numpy')
    m_inverse_func = sp.lambdify((m_D, M_R, mu), m_inverse, 'numpy')
    
    # Fix some parameters and scan others
    M_R_fixed = 1e15  # eV
    mu_fixed = 1e-6   # eV
    
    masses_seesaw = [m_seesaw_func(mD, M_R_fixed) for mD in m_D_values]
    masses_inverse = [m_inverse_func(mD, M_R_fixed, mu_fixed) for mD in m_D_values]
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    plt.loglog(m_D_values, masses_seesaw, 'b-', linewidth=2, 
               label='Type I Seesaw: $m_D^2/M_R$')
    plt.loglog(m_D_values, masses_inverse, 'r-', linewidth=2, 
               label='Inverse Seesaw: $m_D^2 \\mu/M_R^2$')
    
    plt.xlabel('Dirac Mass $m_D$ (eV)')
    plt.ylabel('Light Neutrino Mass (eV)')
    plt.title('Seesaw vs Inverse Seesaw Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add experimental constraints
    plt.axhline(0.05, color='green', linestyle='--', alpha=0.7, 
                label='Atmospheric scale ~0.05 eV')
    plt.axhline(0.009, color='orange', linestyle='--', alpha=0.7, 
                label='Solar scale ~0.009 eV')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("   Plot shows quadratic vs linear scaling with m_D")
    print(f"   For M_R = {M_R_fixed:.0e} eV, μ = {mu_fixed:.0e} eV")
    print()


def texture_analysis():
    """Analyze different texture assumptions symbolically."""
    
    print("=== Texture Analysis ===\n")
    
    print("1. Diagonal Majorana Mass Matrix")
    
    # Create symbolic model with diagonal M_R
    seesaw = SymbolicSeesawTypeI(n_generations=2, n_sterile=2)
    
    # Apply diagonal texture to Majorana matrix
    diag_texture = [(0,1), (1,0)]  # Set off-diagonal elements to zero
    seesaw.set_texture(majorana_texture=diag_texture)
    
    m_light = seesaw.light_mass_matrix_symbolic()
    
    print("   With diagonal M_R, the light mass matrix becomes:")
    sp.pprint(m_light)
    print()
    
    print("2. Hierarchical Majorana Masses")
    
    # Create specific hierarchical case
    M_1 = sp.Symbol('M_1', real=True, positive=True)
    M_2 = sp.Symbol('M_2', real=True, positive=True)
    epsilon = sp.Symbol('epsilon', real=True, positive=True)
    
    # Assume M_2 >> M_1, so M_2 = M_1/ε where ε << 1
    hierarchy_relation = M_2 - M_1/epsilon
    
    print(f"   Assuming M_2 = M_1/ε where ε = {epsilon}")
    print("   This leads to different contributions to light neutrino masses")
    print("   from each heavy neutrino state")
    print()
    
    print("3. Zero Texture Analysis")
    
    # Example with texture zeros in Dirac matrix
    zero_texture = [(0,1), (1,0)]  # No 1-2 mixing in Dirac sector
    seesaw_textured = SymbolicSeesawTypeI(n_generations=2, n_sterile=2)
    seesaw_textured.set_texture(dirac_texture=zero_texture)
    
    m_light_textured = seesaw_textured.light_mass_matrix_symbolic()
    
    print("   With zeros in Dirac matrix positions (1,2) and (2,1):")
    sp.pprint(m_light_textured)
    print("   This reduces the number of free parameters and")
    print("   can lead to specific phenomenological predictions")
    print()


def approximation_analysis():
    """Analyze various approximations and their validity."""
    
    print("=== Approximation Analysis ===\n")
    
    print("1. Seesaw Approximation Validity")
    print("   The Seesaw approximation m_ν ≈ -m_D M_R⁻¹ m_Dᵀ is valid when:")
    print("   • m_D << M_R (hierarchy condition)")
    print("   • Can neglect higher-order corrections")
    print()
    
    # Define expansion parameter
    epsilon = sp.Symbol('epsilon', real=True, positive=True)
    
    print("2. Perturbative Expansion")
    print(f"   Let ε = m_D/M_R be the small expansion parameter")
    print("   Then corrections to Seesaw are of order ε², ε⁴, ...")
    print()
    
    # Show how perturbation theory works
    m_D_sym = sp.Symbol('m_D')
    M_R_sym = sp.Symbol('M_R')
    
    # Leading order
    leading = -m_D_sym**2 / M_R_sym
    
    # Next order (simplified)
    next_order = leading * (m_D_sym / M_R_sym)**2
    
    print(f"   Leading order: {leading}")
    print(f"   Next order correction: ~{next_order}")
    print()
    
    print("3. Inverse Seesaw Naturalness")
    print("   The Inverse Seesaw is 'natural' if μ << M_R")
    print("   This avoids the large hierarchy problem of Type I Seesaw")
    print("   but requires understanding the origin of small μ")
    print()
    
    # Naturalness analysis
    mu_sym = sp.Symbol('mu')
    M_R_sym = sp.Symbol('M_R')
    
    naturalness_param = mu_sym / M_R_sym
    
    print(f"   Naturalness parameter: {naturalness_param}")
    print("   For natural EWSB, this should be ~ 10⁻¹² - 10⁻⁶")
    print()


def main():
    """Run all symbolic demonstrations."""
    
    print("🔬 Symbolic Analysis of Neutrino Mass Generation Mechanisms")
    print("=" * 60)
    
    # Run demonstrations
    seesaw = demonstrate_symbolic_seesaw()
    inverse_seesaw = demonstrate_symbolic_inverse_seesaw()
    
    parameter_scanning_symbolic()
    texture_analysis()
    approximation_analysis()
    
    print("=" * 60)
    print("✓ Symbolic Analysis Complete!")
    print()
    print("Key Insights:")
    print("• Type I Seesaw: m_ν ∼ m_D²/M_R (quadratic suppression)")
    print("• Inverse Seesaw: m_ν ∼ m_D² μ/M_R² (linear in small μ)")
    print("• Both can explain light neutrino masses naturally")
    print("• Symbolic analysis reveals the underlying physics clearly")
    print("• Different textures lead to different phenomenological predictions")


if __name__ == "__main__":
    main()
