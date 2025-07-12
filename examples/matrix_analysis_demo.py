"""
Example: Mass matrix diagonalization techniques and numerical stability

This script demonstrates various aspects of neutrino mass matrix
diagonalization, including numerical stability considerations
and different diagonalization approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from matrix_utils import (diagonalize_hermitian_matrix, diagonalize_mass_matrix,
                         check_unitarity, mass_squared_differences, 
                         enforce_mass_ordering, matrix_condition_number)
from pmns_matrix import pmns_matrix_standard, extract_mixing_angles, jarlskog_invariant


def demonstrate_diagonalization():
    """Demonstrate different diagonalization techniques."""
    
    print("=== Mass Matrix Diagonalization Examples ===\n")
    
    # Create a complex symmetric mass matrix (Majorana case)
    print("1. Symmetric Mass Matrix (Majorana neutrinos)")
    
    # Generate a random Hermitian matrix
    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    mass_matrix = (A + A.conj().T) / 2  # Make Hermitian
    mass_matrix = (mass_matrix + mass_matrix.T) / 2  # Make symmetric
    
    print("Original mass matrix:")
    print(mass_matrix)
    print(f"Is symmetric: {np.allclose(mass_matrix, mass_matrix.T)}")
    print(f"Condition number: {matrix_condition_number(mass_matrix):.2e}")
    
    # Diagonalize
    eigenvalues, eigenvectors = diagonalize_mass_matrix(mass_matrix, symmetric=True)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvector matrix shape: {eigenvectors.shape}")
    print(f"Mixing matrix is unitary: {check_unitarity(eigenvectors)}")
    
    # Verify diagonalization
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    print(f"Reconstruction error: {np.max(np.abs(mass_matrix - reconstructed)):.2e}")
    
    # Calculate mass squared differences
    dm_squared = mass_squared_differences(np.abs(eigenvalues))
    print(f"\nMass squared differences:")
    print(dm_squared)


def study_numerical_stability():
    """Study numerical stability with ill-conditioned matrices."""
    
    print("\n=== Numerical Stability Analysis ===\n")
    
    # Create matrices with different condition numbers
    condition_numbers = np.logspace(2, 12, 6)
    
    for i, target_cond in enumerate(condition_numbers):
        print(f"Case {i+1}: Target condition number = {target_cond:.1e}")
        
        # Create ill-conditioned matrix
        U = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        U, _ = np.linalg.qr(U)  # Orthogonalize
        
        # Eigenvalues with controlled condition number
        eigenvals = np.array([1.0, 1.0/np.sqrt(target_cond), 1.0/target_cond])
        
        # Construct matrix
        mass_matrix = U @ np.diag(eigenvals) @ U.conj().T
        mass_matrix = (mass_matrix + mass_matrix.conj().T) / 2  # Ensure Hermitian
        
        actual_cond = matrix_condition_number(mass_matrix)
        print(f"  Actual condition number: {actual_cond:.1e}")
        
        # Test diagonalization
        try:
            computed_eigenvals, computed_eigenvecs = diagonalize_mass_matrix(mass_matrix)
            
            # Check accuracy
            relative_error = np.abs(np.sort(computed_eigenvals) - np.sort(eigenvals)) / np.sort(eigenvals)
            max_error = np.max(relative_error)
            print(f"  Max relative eigenvalue error: {max_error:.2e}")
            
            # Check unitarity
            is_unitary = check_unitarity(computed_eigenvecs, tolerance=1e-8)
            print(f"  Mixing matrix unitary (1e-8 tol): {is_unitary}")
            
        except np.linalg.LinAlgError as e:
            print(f"  Diagonalization failed: {e}")
        
        print()


def compare_mass_orderings():
    """Compare normal and inverted mass orderings."""
    
    print("=== Mass Ordering Analysis ===\n")
    
    # Generate random mass matrix
    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    mass_matrix = (A + A.conj().T) / 2
    
    masses, mixing = diagonalize_mass_matrix(mass_matrix, symmetric=True)
    
    print("Original masses (unsorted):")
    for i, mass in enumerate(masses):
        print(f"  m_{i+1} = {mass:.3e} eV")
    
    # Enforce normal ordering
    masses_normal, mixing_normal = enforce_mass_ordering(masses, mixing, 'normal')
    
    print("\nNormal ordering (m₁ < m₂ < m₃):")
    for i, mass in enumerate(masses_normal):
        print(f"  m_{i+1} = {mass:.3e} eV")
    
    # Calculate experimental observables
    dm_squared = mass_squared_differences(masses_normal)
    dm21_sq = dm_squared[1, 0]
    dm31_sq = dm_squared[2, 0]
    
    print(f"\nMass squared differences:")
    print(f"  Δm²₂₁ = {dm21_sq:.3e} eV²")
    print(f"  Δm²₃₁ = {dm31_sq:.3e} eV²")
    
    # Extract mixing parameters
    theta12, theta13, theta23, delta_cp = extract_mixing_angles(mixing_normal)
    
    print(f"\nMixing parameters:")
    print(f"  θ₁₂ = {np.degrees(theta12):.1f}°")
    print(f"  θ₁₃ = {np.degrees(theta13):.1f}°")
    print(f"  θ₂₃ = {np.degrees(theta23):.1f}°")
    print(f"  δCP = {np.degrees(delta_cp):.1f}°")
    
    # Jarlskog invariant
    J = jarlskog_invariant(mixing_normal)
    print(f"  Jarlskog invariant: {J:.3e}")


def pmns_matrix_analysis():
    """Analyze properties of the PMNS matrix."""
    
    print("\n=== PMNS Matrix Analysis ===\n")
    
    # Construct PMNS matrix with typical values
    theta12 = np.radians(33.45)  # Solar angle
    theta13 = np.radians(8.62)   # Reactor angle
    theta23 = np.radians(49.2)   # Atmospheric angle
    delta_cp = np.radians(197)   # CP phase
    alpha1 = np.radians(30)      # Majorana phase 1
    alpha2 = np.radians(45)      # Majorana phase 2
    
    # Standard parameterization
    U_standard = pmns_matrix_standard(theta12, theta13, theta23, delta_cp, alpha1, alpha2)
    
    print("PMNS matrix (standard parameterization):")
    print("Magnitude:")
    print(np.abs(U_standard))
    print("\nPhase (degrees):")
    print(np.degrees(np.angle(U_standard)))
    
    # Check unitarity
    print(f"\nUnitarity check: {check_unitarity(U_standard)}")
    
    # Extract mixing angles
    extracted_angles = extract_mixing_angles(U_standard)
    print(f"\nExtracted mixing angles:")
    print(f"  θ₁₂ = {np.degrees(extracted_angles[0]):.1f}° (input: {np.degrees(theta12):.1f}°)")
    print(f"  θ₁₃ = {np.degrees(extracted_angles[1]):.1f}° (input: {np.degrees(theta13):.1f}°)")
    print(f"  θ₂₃ = {np.degrees(extracted_angles[2]):.1f}° (input: {np.degrees(theta23):.1f}°)")
    print(f"  δCP = {np.degrees(extracted_angles[3]):.1f}° (input: {np.degrees(delta_cp):.1f}°)")
    
    # Jarlskog invariant
    J = jarlskog_invariant(U_standard)
    J_theoretical = (1/8) * np.sin(2*theta12) * np.sin(2*theta13) * np.sin(2*theta23) * np.cos(theta13) * np.sin(delta_cp)
    
    print(f"\nJarlskog invariant:")
    print(f"  Computed: {J:.4e}")
    print(f"  Theoretical: {J_theoretical:.4e}")
    print(f"  Difference: {abs(J - J_theoretical):.2e}")


def matrix_perturbation_analysis():
    """Analyze sensitivity to small perturbations."""
    
    print("\n=== Perturbation Analysis ===\n")
    
    # Base mass matrix
    base_matrix = np.array([
        [1e-3, 5e-4, 2e-4],
        [5e-4, 2e-3, 8e-4],
        [2e-4, 8e-4, 3e-3]
    ], dtype=complex)
    
    # Reference diagonalization
    ref_masses, ref_mixing = diagonalize_mass_matrix(base_matrix, symmetric=True)
    
    print("Reference masses:")
    for i, mass in enumerate(ref_masses):
        print(f"  m_{i+1} = {mass:.3e} eV")
    
    # Add small perturbations
    perturbation_sizes = np.logspace(-6, -2, 5)
    
    print(f"\nPerturbation analysis:")
    print(f"{'Perturbation':<12} {'Max Δm/m':<12} {'Max Δθ (deg)':<15}")
    print("-" * 40)
    
    for pert_size in perturbation_sizes:
        # Random perturbation
        perturbation = pert_size * np.random.rand(3, 3)
        perturbation = (perturbation + perturbation.T) / 2  # Keep symmetric
        
        perturbed_matrix = base_matrix + perturbation
        
        # Diagonalize perturbed matrix
        pert_masses, pert_mixing = diagonalize_mass_matrix(perturbed_matrix, symmetric=True)
        
        # Calculate changes
        mass_changes = np.abs((pert_masses - ref_masses) / ref_masses)
        max_mass_change = np.max(mass_changes)
        
        # Extract mixing angles
        ref_angles = extract_mixing_angles(ref_mixing)
        pert_angles = extract_mixing_angles(pert_mixing)
        
        angle_changes = np.abs(np.array(pert_angles[:3]) - np.array(ref_angles[:3]))
        max_angle_change = np.degrees(np.max(angle_changes))
        
        print(f"{pert_size:.1e}      {max_mass_change:.3e}      {max_angle_change:.2f}")


def create_visualization():
    """Create visualization of matrix properties."""
    
    print("\n=== Creating Visualizations ===\n")
    
    # Generate sample mass matrix
    np.random.seed(42)  # For reproducibility
    A = np.random.rand(3, 3) + 1j * 0.1 * np.random.rand(3, 3)
    mass_matrix = (A + A.conj().T) / 2
    
    masses, mixing = diagonalize_mass_matrix(mass_matrix, symmetric=True)
    
    # Plot eigenvalue spectrum
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mass spectrum
    ax1.bar(range(len(masses)), np.abs(masses), color=['blue', 'green', 'red'])
    ax1.set_xlabel('Mass State')
    ax1.set_ylabel('Mass (eV)')
    ax1.set_title('Neutrino Mass Spectrum')
    ax1.set_xticks(range(len(masses)))
    ax1.set_xticklabels([f'm_{i+1}' for i in range(len(masses))])
    
    # Mixing matrix magnitude
    im2 = ax2.imshow(np.abs(mixing), cmap='viridis', aspect='auto')
    ax2.set_title('|U_αi| Mixing Matrix')
    ax2.set_xlabel('Mass Eigenstate')
    ax2.set_ylabel('Flavor Eigenstate')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(['ν₁', 'ν₂', 'ν₃'])
    ax2.set_yticklabels(['νₑ', 'νμ', 'ντ'])
    plt.colorbar(im2, ax=ax2)
    
    # Mass squared differences
    dm_sq = mass_squared_differences(masses)
    im3 = ax3.imshow(dm_sq, cmap='RdBu_r', aspect='auto')
    ax3.set_title('Mass Squared Differences')
    ax3.set_xlabel('j')
    ax3.set_ylabel('i')
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    plt.colorbar(im3, ax=ax3, label='Δm²ᵢⱼ (eV²)')
    
    # Condition number vs matrix size
    sizes = range(2, 8)
    cond_numbers = []
    
    for size in sizes:
        A = np.random.rand(size, size) + 1j * 0.1 * np.random.rand(size, size)
        M = (A + A.conj().T) / 2
        cond_numbers.append(matrix_condition_number(M))
    
    ax4.semilogy(sizes, cond_numbers, 'o-')
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Condition Number')
    ax4.set_title('Condition Number vs Matrix Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete.")


def main():
    """Run all demonstration examples."""
    
    demonstrate_diagonalization()
    study_numerical_stability() 
    compare_mass_orderings()
    pmns_matrix_analysis()
    matrix_perturbation_analysis()
    create_visualization()
    
    print("\n=== All Demonstrations Complete ===")
    print("This analysis covered:")
    print("- Basic matrix diagonalization")
    print("- Numerical stability considerations")
    print("- Mass ordering enforcement")
    print("- PMNS matrix properties")
    print("- Sensitivity to perturbations")
    print("- Visualization techniques")


if __name__ == "__main__":
    main()
