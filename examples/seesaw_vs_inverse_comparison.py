"""
Example: Comparing Seesaw Type I and Inverse Seesaw mechanisms

This script demonstrates how to:
1. Set up mass matrices for both mechanisms
2. Calculate light neutrino masses and mixing
3. Compare the results
4. Visualize the differences
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.seesaw import SeesawTypeI
from src.inverse_seesaw import InverseSeesaw, compare_seesaw_vs_inverse_seesaw
from src.pmns_matrix import pmns_from_experimental, extract_mixing_angles
from src.visualization import (plot_mass_spectrum, compare_mass_spectra, 
                          plot_mixing_matrix, create_summary_report)


def main():
    """Main example comparing Seesaw and Inverse Seesaw mechanisms."""
    
    print("=== Neutrino Mass Matrix Analysis: Seesaw vs Inverse Seesaw ===\n")
    
    # Define common parameters
    print("1. Setting up mass matrices...")
    
    # Dirac mass matrix (in eV) - typically of order electroweak scale
    m_D = np.array([
        [1e-3, 5e-3, 1e-2],
        [2e-3, 8e-3, 3e-2], 
        [1e-3, 1e-2, 5e-2]
    ], dtype=complex)
    
    # Heavy mass scale (in eV) - GUT scale
    M_scale = 1e15  # GeV
    M_R = M_scale * np.array([
        [1.0, 0.1, 0.05],
        [0.1, 1.2, 0.08],
        [0.05, 0.08, 0.9]
    ], dtype=complex)
    
    # Small parameter for Inverse Seesaw (in eV)
    mu_scale = 1e-6  # keV scale
    mu = mu_scale * np.eye(3)
    
    print(f"Dirac mass scale: ~{np.sqrt(np.mean(np.abs(m_D)**2)):.2e} eV")
    print(f"Heavy mass scale: ~{np.sqrt(np.mean(np.abs(M_R)**2)):.2e} eV")
    print(f"μ parameter scale: {mu_scale:.2e} eV\n")
    
    # Calculate and compare mechanisms
    print("2. Calculating light neutrino masses...")
    
    # Type I Seesaw
    seesaw_i = SeesawTypeI(m_D, M_R)
    masses_seesaw, mixing_seesaw = seesaw_i.diagonalize_light_sector()
    
    print("Type I Seesaw masses:")
    for i, mass in enumerate(masses_seesaw):
        print(f"  m_{i+1} = {mass:.3e} eV")
    
    # Inverse Seesaw
    inverse_seesaw = InverseSeesaw(m_D, M_R, mu)
    masses_inverse, mixing_inverse = inverse_seesaw.diagonalize_light_sector()
    
    print("\nInverse Seesaw masses:")
    for i, mass in enumerate(masses_inverse):
        print(f"  m_{i+1} = {mass:.3e} eV")
    
    # Extract mixing angles
    print("\n3. Extracting mixing parameters...")
    
    theta12_s, theta13_s, theta23_s, delta_s = extract_mixing_angles(mixing_seesaw)
    theta12_i, theta13_i, theta23_i, delta_i = extract_mixing_angles(mixing_inverse)
    
    print("Type I Seesaw mixing angles:")
    print(f"  θ₁₂ = {np.degrees(theta12_s):.1f}°")
    print(f"  θ₁₃ = {np.degrees(theta13_s):.1f}°") 
    print(f"  θ₂₃ = {np.degrees(theta23_s):.1f}°")
    print(f"  δCP = {np.degrees(delta_s):.1f}°")
    
    print("\nInverse Seesaw mixing angles:")
    print(f"  θ₁₂ = {np.degrees(theta12_i):.1f}°")
    print(f"  θ₁₃ = {np.degrees(theta13_i):.1f}°")
    print(f"  θ₂₃ = {np.degrees(theta23_i):.1f}°")
    print(f"  δCP = {np.degrees(delta_i):.1f}°")
    
    # Compare with experimental values
    print("\n4. Comparing with experimental values...")
    
    pmns_exp = pmns_from_experimental('normal')
    theta12_exp, theta13_exp, theta23_exp, delta_exp = extract_mixing_angles(pmns_exp)
    
    print("Experimental values (NuFIT 5.2):")
    print(f"  θ₁₂ = {np.degrees(theta12_exp):.1f}°")
    print(f"  θ₁₃ = {np.degrees(theta13_exp):.1f}°")
    print(f"  θ₂₃ = {np.degrees(theta23_exp):.1f}°")
    print(f"  δCP = {np.degrees(delta_exp):.1f}°")
    
    # Detailed comparison
    print("\n5. Performing detailed comparison...")
    comparison = compare_seesaw_vs_inverse_seesaw(m_D, M_R, mu)
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    
    # Plot mass spectra
    plot_mass_spectrum(masses_seesaw, "Type I Seesaw", "normal")
    plot_mass_spectrum(masses_inverse, "Inverse Seesaw", "normal")
    
    # Compare mass spectra
    compare_mass_spectra(comparison)
    
    # Plot mixing matrices
    plot_mixing_matrix(mixing_seesaw, "Type I Seesaw")
    plot_mixing_matrix(mixing_inverse, "Inverse Seesaw")
    
    # Create summary report
    print("\n7. Summary report:")
    summary_df = create_summary_report(comparison)
    print(summary_df.to_string(index=False))
    
    # Additional analysis
    print("\n8. Additional insights...")
    
    # Scale comparison
    seesaw_scale = seesaw_i.seesaw_scale()
    mu_lnv_scale = inverse_seesaw.lepton_number_violation_scale()
    naturalness = inverse_seesaw.naturalness_parameter()
    
    print(f"Seesaw scale: {seesaw_scale:.2e} eV")
    print(f"LNV scale (μ): {mu_lnv_scale:.2e} eV")
    print(f"Naturalness parameter μ/M: {naturalness:.2e}")
    
    # Mass ratio comparison
    mass_ratios = masses_seesaw / masses_inverse
    print(f"\nMass ratios (Seesaw/Inverse):")
    for i, ratio in enumerate(mass_ratios):
        print(f"  m_{i+1}(Seesaw)/m_{i+1}(Inverse) = {ratio:.2e}")
    
    print(f"\n=== Analysis Complete ===")
    print("The comparison shows how the two mechanisms can produce")
    print("similar light neutrino masses through different physics:")
    print("- Seesaw: Large mass suppression (v²/M)")
    print("- Inverse Seesaw: Small LNV parameter (μ)")


def parameter_scan_example():
    """Example of scanning over the μ parameter in Inverse Seesaw."""
    
    print("\n=== Parameter Scan: μ dependence ===")
    
    # Fixed matrices
    m_D = 1e-2 * np.random.rand(3, 3) + 1j * 1e-3 * np.random.rand(3, 3)
    M_R = 1e15 * (np.eye(3) + 0.1 * np.random.rand(3, 3))
    
    # Scan over μ
    mu_values = np.logspace(-9, -3, 20)  # 1 neV to 1 keV
    masses_scan = []
    
    for mu_val in mu_values:
        inverse_seesaw = InverseSeesaw(m_D, M_R, mu_val)
        masses, _ = inverse_seesaw.diagonalize_light_sector()
        masses_scan.append(masses)
    
    masses_array = np.array(masses_scan)
    
    # Plot results
    from visualization import plot_parameter_space_scan
    plot_parameter_space_scan(mu_values, masses_array, 
                             "μ parameter (eV)", "Inverse Seesaw")
    
    print("Parameter scan complete. Notice how masses scale linearly with μ.")


if __name__ == "__main__":
    main()
    
    # Uncomment to run parameter scan
    # parameter_scan_example()
