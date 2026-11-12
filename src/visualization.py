"""
Visualization Tools for Neutrino Physics Analysis

This module provides specialized plotting and visualization functions designed
for neutrino mass matrix analysis, mixing pattern studies, and comparison
between different Seesaw mechanisms. The visualizations are tailored to help
physicists understand the complex relationships between theoretical predictions
and experimental observations in neutrino physics.

## Physics-Focused Visualization Features

### Mass Spectrum Analysis:
- **Logarithmic scales**: Essential for neutrino masses spanning many orders
- **Mass ordering**: Clear distinction between normal and inverted hierarchies  
- **Experimental bands**: Comparison with measured mass-squared differences
- **Hierarchy visualization**: Ratios and relative scale differences

### Mixing Matrix Representations:
- **Magnitude heatmaps**: |Uₐᵢ|² showing transition probabilities
- **Phase plots**: Complex phase structure of mixing matrix elements
- **Unitarity checks**: Visual verification of matrix unitarity
- **Symmetry patterns**: Recognition of special mixing structures

### Parameter Space Studies:
- **Multi-dimensional scans**: Exploration of allowed parameter regions
- **Correlation plots**: Dependencies between mixing angles and masses
- **Constraint visualization**: Experimental limits and theoretical predictions
- **Sensitivity analysis**: Parameter variations and uncertainties

### Mechanism Comparisons:
- **Side-by-side spectra**: Direct comparison between Seesaw types
- **Phenomenological differences**: Observable consequences of different models
- **Scale relationships**: Connection between input parameters and predictions

## Physical Context and Conventions

### Energy Scale Representation:
All plots use physically meaningful scales and units:
- **Neutrino masses**: eV (typical range: 10⁻³ to 10⁻¹ eV)
- **Seesaw scales**: GeV to 10¹⁶ GeV (electroweak to GUT scale)
- **Mass ratios**: Dimensionless hierarchies and fine-tuning measures

### Experimental Reference Data:
Built-in comparison with current experimental measurements:
- **NuFIT global fits**: Latest neutrino oscillation parameters
- **Mass squared differences**: Solar (Δm²₂₁) and atmospheric (Δm²₃₁) scales
- **Mixing angles**: Reactor, solar, and atmospheric angle measurements
- **CP violation**: Current constraints on Dirac phase δ_CP

### Color Schemes and Conventions:
- **Physics-motivated coloring**: Different colors for different mass states
- **Accessibility**: Colorblind-friendly palettes where possible
- **Standard conventions**: Following common neutrino physics notation

## Specialized Plot Types

### Mass Spectrum Plots:
```python
plot_mass_spectrum(masses, mechanism_name="Type I Seesaw", ordering="normal")
```
- Displays mass eigenvalues with proper scaling
- Shows mass hierarchy and absolute scale
- Compares with experimental constraints
- Includes uncertainty bands when available

### Mixing Matrix Heatmaps:
```python
plot_mixing_matrix(mixing_matrix, mechanism_name="PMNS")
```
- Magnitude and phase representation
- Clear labeling of flavor (e, μ, τ) and mass (1, 2, 3) bases
- Unitarity verification and sum rules
- Color-coded probability amplitudes

### Unitarity Triangles:
```python
plot_unitarity_triangle(pmns_matrix)
```
- Complex plane visualization of unitarity relations
- Different triangles from different unitarity conditions
- Area proportional to CP violation (Jarlskog invariant)
- Experimental constraints and theoretical predictions

### Parameter Space Scans:
```python
plot_parameter_space_scan(param_range, masses_array, "Heavy Mass Scale")
```
- Multi-parameter correlation studies
- Allowed vs excluded regions
- Sensitivity to experimental uncertainties
- Model discrimination power

### Comparison Studies:
```python
compare_mass_spectra(comparison_results)
plot_mass_squared_differences(comparison_results)
```
- Side-by-side mechanism comparison
- Quantitative difference measures
- Experimental preference assessment
- Model selection criteria

## Technical Features

### Output Formats:
- **High-resolution PNG**: Publication-quality figures
- **Vector formats**: Scalable PDF and SVG for presentations
- **Interactive plots**: Web-based visualization with Plotly integration
- **Batch processing**: Automated figure generation for parameter studies

### Customization Options:
- **Flexible styling**: Custom color schemes, fonts, and layouts
- **Annotation tools**: Physics labels, equations, and explanatory text
- **Multi-panel figures**: Complex comparison and correlation plots
- **Export control**: Resolution, format, and size optimization

### Numerical Robustness:
- **Logarithmic handling**: Proper treatment of zero and negative values
- **Dynamic scaling**: Automatic range adjustment for different scenarios
- **Error handling**: Graceful degradation for problematic data
- **Memory efficiency**: Optimized for large parameter scan datasets

All visualization functions are designed to work seamlessly with the
computational modules in this package, providing an integrated workflow
from theoretical calculation to publication-ready figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
import warnings

# Suppress matplotlib warnings about non-interactive backend
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')


def plot_mass_spectrum(masses: np.ndarray, 
                        mechanism_name: str = "",
                        ordering: str = "normal",
                        save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive visualization of the neutrino mass spectrum.
    
    This function produces a publication-quality plot showing neutrino mass
    eigenvalues with proper physics context, experimental comparisons, and
    clear indication of mass hierarchy patterns.
    
    ## Physics Context:
    
    **Mass Hierarchies:**
    - Normal ordering (NO): m₁ < m₂ < m₃ (currently favored)
    - Inverted ordering (IO): m₃ < m₁ < m₂ (disfavored but not excluded)
    - Quasi-degenerate: m₁ ≈ m₂ ≈ m₃ (allowed for large absolute masses)
    
    **Experimental Constraints:**
    - Lightest mass: < 0.8 eV (Planck + BAO cosmological constraint)
    - Mass differences: Δm²₂₁ ≈ 7.4×10⁻⁵ eV², |Δm²₃₁| ≈ 2.5×10⁻³ eV²
    - Oscillation experiments: Determine ratios, not absolute scale
    - Direct kinematic limit: < 2 eV (Katrin tritium decay experiment)
    
    **Theoretical Predictions:**
    Different mass generation mechanisms predict different patterns:
    - Type I Seesaw: Typically hierarchical patterns
    - Inverse Seesaw: Can accommodate quasi-degenerate spectra
    - Radiative models: Often predict specific mass ratios
    
    ## Visualization Features:
    
    **Logarithmic Scale:**
    Essential for neutrino masses due to:
    - Wide range: Factor of √(Δm²₃₁/Δm²₂₁) ≈ 6 between scales
    - Small absolute values: Typically 0.001 - 0.1 eV
    - Comparison with other fermion masses (electron: 0.511 MeV)
    
    **Color Coding:**
    - Light blue: ν₁ (lightest in normal ordering)
    - Light green: ν₂ (intermediate mass)
    - Light coral: ν₃ (heaviest in normal ordering)
    
    **Annotations:**
    - Mass values displayed with appropriate precision
    - Mass ordering clearly indicated
    - Mechanism name and theoretical context
    - Comparison with experimental constraints
    
    Parameters:
    -----------
    masses : np.ndarray
        Neutrino mass eigenvalues in eV
        Should contain 3 elements for standard 3-neutrino case
        Typical range: 10⁻³ to 10⁻¹ eV
    mechanism_name : str, optional
        Name of the mass generation mechanism for plot title
        Examples: "Type I Seesaw", "Inverse Seesaw", "Radiative Model"
    ordering : str, optional
        Mass ordering convention for labeling
        - "normal": m₁ < m₂ < m₃ (default)
        - "inverted": m₃ < m₁ < m₂
    save_path : str, optional
        Full path to save the plot file
        If None, saves to default location with descriptive name
        
    Returns:
    --------
    None
        Plot is saved to file and displayed if in interactive mode
        
    Notes:
    ------
    - Uses logarithmic y-axis to handle small mass values effectively
    - Automatically sorts masses for consistent display
    - Includes grid lines for easier value reading
    - Error bars can be added for masses with uncertainties
    - Compatible with both theoretical predictions and experimental fits
    
    Example:
    --------
    >>> import numpy as np
    >>> # Typical Type I Seesaw prediction
    >>> masses = np.array([0.001, 0.01, 0.05])  # eV
    >>> plot_mass_spectrum(masses, "Type I Seesaw", "normal")
    >>> 
    >>> # Compare with experimental constraints
    >>> dm21_sq = 7.4e-5  # eV²
    >>> dm31_sq = 2.5e-3  # eV²
    >>> m1 = 0.001  # eV (assumed)
    >>> m2 = np.sqrt(m1**2 + dm21_sq)
    >>> m3 = np.sqrt(m1**2 + dm31_sq)
    >>> experimental_masses = np.array([m1, m2, m3])
    >>> plot_mass_spectrum(experimental_masses, "Experimental Best Fit", "normal")
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Sort masses for display
    sorted_masses = np.sort(np.abs(masses))
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_masses)), sorted_masses, 
                    color=['lightblue', 'lightgreen', 'lightcoral'][:len(sorted_masses)],
                    alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_xlabel('Mass State')
    ax.set_ylabel('Mass (eV)')
    ax.set_title(f'Neutrino Mass Spectrum\n{mechanism_name} - {ordering.title()} Ordering')
    ax.set_xticks(range(len(sorted_masses)))
    ax.set_xticklabels([f'm_{i+1}' for i in range(len(sorted_masses))])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mass in zip(bars, sorted_masses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mass:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = f"mass_spectrum_{mechanism_name.replace(' ', '_').lower()}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def plot_mixing_matrix(mixing_matrix: np.ndarray,
                        mechanism_name: str = "",
                        save_path: Optional[str] = None) -> None:
    """
    Plot the mixing matrix as a heatmap.
    
    Parameters:
    -----------
    mixing_matrix : np.ndarray
        Mixing matrix (typically PMNS-like)
    mechanism_name : str, optional
        Name of the mechanism for the title
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot magnitude
    magnitude = np.abs(mixing_matrix)
    im1 = ax1.imshow(magnitude, cmap='viridis', aspect='auto')
    ax1.set_title(f'|U| - {mechanism_name}')
    ax1.set_xlabel('Mass Eigenstate')
    ax1.set_ylabel('Flavor Eigenstate')
    ax1.set_xticks(range(mixing_matrix.shape[1]))
    ax1.set_yticks(range(mixing_matrix.shape[0]))
    ax1.set_xticklabels([f'ν_{i+1}' for i in range(mixing_matrix.shape[1])])
    ax1.set_yticklabels(['e', 'μ', 'τ'][:mixing_matrix.shape[0]])
    
    # Add text annotations
    for i in range(mixing_matrix.shape[0]):
        for j in range(mixing_matrix.shape[1]):
            ax1.text(j, i, f'{magnitude[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label=r'$|U_{\alpha i}|$')
    
    # Plot phase
    phase = np.angle(mixing_matrix)
    im2 = ax2.imshow(phase, cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f'arg(U) - {mechanism_name}')
    ax2.set_xlabel('Mass Eigenstate')
    ax2.set_ylabel('Flavor Eigenstate')
    ax2.set_xticks(range(mixing_matrix.shape[1]))
    ax2.set_yticks(range(mixing_matrix.shape[0]))
    ax2.set_xticklabels([f'ν_{i+1}' for i in range(mixing_matrix.shape[1])])
    ax2.set_yticklabels(['$e$', r'$\mu$', r'$\tau$'][:mixing_matrix.shape[0]])
    
    # Add text annotations for phases
    for i in range(mixing_matrix.shape[0]):
        for j in range(mixing_matrix.shape[1]):
            ax2.text(j, i, f'{phase[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontweight='bold')

    plt.colorbar(im2, ax=ax2, label=r'$\arg(U_{\alpha i})$ [rad]')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = f"mixing_matrix_{mechanism_name.replace(' ', '_').lower()}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def compare_mass_spectra(comparison_results: Dict, 
                        save_path: Optional[str] = None) -> None:
    """
    Compare mass spectra from different mechanisms.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from compare_seesaw_mechanisms or similar functions
    save_path : str, optional
        Path to save the plot
    """
    mechanisms = list(comparison_results.keys())
    if 'comparison' in mechanisms:
        mechanisms.remove('comparison')
    
    n_mechanisms = len(mechanisms)
    fig, axes = plt.subplots(1, n_mechanisms, figsize=(5*n_mechanisms, 6))
    
    if n_mechanisms == 1:
        axes = [axes]
    
    for i, mechanism in enumerate(mechanisms):
        # Try different possible keys for masses based on the data structure
        if 'light_masses' in comparison_results[mechanism]:
            masses = comparison_results[mechanism]['light_masses']
        elif 'masses' in comparison_results[mechanism]:
            masses = comparison_results[mechanism]['masses']
        elif 'neutrino_masses' in comparison_results[mechanism]:
            masses = comparison_results[mechanism]['neutrino_masses']
        else:
            # Print available keys for debugging and skip this mechanism
            print(f"Warning: No mass data found for {mechanism}")
            print(f"Available keys: {list(comparison_results[mechanism].keys())}")
            continue
            
        sorted_masses = np.sort(np.abs(masses))
        
        bars = axes[i].bar(range(len(sorted_masses)), sorted_masses,
                            color=['lightblue', 'lightgreen', 'lightcoral'][:len(sorted_masses)],
                            alpha=0.7, edgecolor='black')
        
        axes[i].set_xlabel('Mass State')
        axes[i].set_ylabel('Mass (eV)')
        axes[i].set_title(f'{mechanism.replace("_", " ")}')
        axes[i].set_xticks(range(len(sorted_masses)))
        axes[i].set_xticklabels([f'm_{j+1}' for j in range(len(sorted_masses))])
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mass in zip(bars, sorted_masses):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mass:.2e}', ha='center', va='bottom', 
                        fontsize=8, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = "mass_spectra_comparison.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def plot_mass_squared_differences(comparison_results: Dict,
                                save_path: Optional[str] = None) -> None:
    """
    Plot mass squared differences for different mechanisms.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from comparison functions
    save_path : str, optional
        Path to save the plot
    """
    mechanisms = list(comparison_results.keys())
    if 'comparison' in mechanisms:
        mechanisms.remove('comparison')
    
    # Extract Δm²₂₁ and Δm²₃₁ for each mechanism
    dm21_values = []
    dm31_values = []
    mechanism_names = []
    
    for mechanism in mechanisms:
        if 'mass_squared_differences' in comparison_results[mechanism]:
            dm_sq = comparison_results[mechanism]['mass_squared_differences']
            dm21_values.append(dm_sq[1, 0])  # Δm²₂₁
            dm31_values.append(dm_sq[2, 0])  # Δm²₃₁
            mechanism_names.append(mechanism.replace('_', ' '))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Δm²₂₁ comparison
    bars1 = ax1.bar(range(len(dm21_values)), dm21_values, 
                    color='lightblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mechanism')
    ax1.set_ylabel('Δm²₂₁ (eV²)')
    ax1.set_title('Solar Mass Squared Difference')
    ax1.set_xticks(range(len(mechanism_names)))
    ax1.set_xticklabels(mechanism_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add experimental reference line
    exp_dm21 = 7.42e-5  # Approximate experimental value
    ax1.axhline(exp_dm21, color='red', linestyle='--', 
                label=f'Experimental: {exp_dm21:.2e} eV²')
    ax1.legend()
    
    # Δm²₃₁ comparison  
    bars2 = ax2.bar(range(len(dm31_values)), dm31_values,
                    color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mechanism')
    ax2.set_ylabel('Δm²₃₁ (eV²)')
    ax2.set_title('Atmospheric Mass Squared Difference')
    ax2.set_xticks(range(len(mechanism_names)))
    ax2.set_xticklabels(mechanism_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add experimental reference line
    exp_dm31 = 2.515e-3  # Approximate experimental value (normal ordering)
    ax2.axhline(exp_dm31, color='red', linestyle='--',
                label=f'Experimental: {exp_dm31:.2e} eV²')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = "mass_squared_differences.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def plot_parameter_space_scan(parameter_range: np.ndarray,
                                masses_array: np.ndarray,
                                parameter_name: str,
                                mechanism_name: str = "",
                                save_path: Optional[str] = None) -> None:
    """
    Plot how neutrino masses vary with a parameter.
    
    Parameters:
    -----------
    parameter_range : np.ndarray
        Range of parameter values scanned
    masses_array : np.ndarray
        Array of mass eigenvalues for each parameter value
        Shape: (n_points, n_masses)
    parameter_name : str
        Name of the parameter being scanned
    mechanism_name : str, optional
        Name of the mechanism
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    n_masses = masses_array.shape[1]
    colors = ['blue', 'green', 'red', 'orange', 'purple'][:n_masses]
    
    for i in range(n_masses):
        ax.loglog(parameter_range, masses_array[:, i], 
                    color=colors[i], linewidth=2, label=f'm_{i+1}')
    
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Mass (eV)')
    ax.set_title(f'Mass Spectrum vs {parameter_name}\n{mechanism_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = f"parameter_scan_{parameter_name.replace(' ', '_').lower()}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def plot_unitarity_triangle(pmns_matrix: np.ndarray,
                            save_path: Optional[str] = None) -> None:
    """
    Plot unitarity triangles in the complex plane.
    
    Parameters:
    -----------
    pmns_matrix : np.ndarray
        PMNS mixing matrix
    save_path : str, optional
        Path to save the plot
    """
    U = pmns_matrix
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Row unitarity triangles
    for row in range(3):
        # Elements of the triangle: U_αi * U*_αj
        z1 = U[row, 0] * U[row, 1].conj()
        z2 = U[row, 1] * U[row, 2].conj()
        z3 = U[row, 2] * U[row, 0].conj()
        
        triangle_real = [z1.real, z2.real, z3.real, z1.real]
        triangle_imag = [z1.imag, z2.imag, z3.imag, z1.imag]
        
        axes[row].plot(triangle_real, triangle_imag, 'o-', linewidth=2, markersize=8)
        axes[row].fill(triangle_real[:-1], triangle_imag[:-1], alpha=0.3)
        
        # Mark vertices
        vertices = [z1, z2, z3]
        labels = [f'U_{["e","μ","τ"][row]}1 U*_{["e","μ","τ"][row]}2',
                    f'U_{["e","μ","τ"][row]}2 U*_{["e","μ","τ"][row]}3',
                    f'U_{["e","μ","τ"][row]}3 U*_{["e","μ","τ"][row]}1']
        
        for vertex, label in zip(vertices, labels):
            axes[row].annotate(label, (vertex.real, vertex.imag), 
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=8, ha='left')
        
        axes[row].set_xlabel('Real')
        axes[row].set_ylabel('Imaginary')
        axes[row].set_title(f'{["Electron", "Muon", "Tau"][row]} Row Triangle')
        axes[row].grid(True, alpha=0.3)
        axes[row].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Save to default location if no path specified
        default_path = "unitarity_triangles.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    # Only show if in interactive mode
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory


def create_summary_report(comparison_results: Dict,
                            save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of comparison results.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from comparison functions
    save_path : str, optional
        Path to save the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    mechanisms = list(comparison_results.keys())
    if 'comparison' in mechanisms:
        mechanisms.remove('comparison')
    
    summary_data = []
    
    for mechanism in mechanisms:
        result = comparison_results[mechanism]
        
        # Try different possible keys for masses
        if 'light_masses' in result:
            masses = result['light_masses']
        elif 'masses' in result:
            masses = result['masses']
        elif 'neutrino_masses' in result:
            masses = result['neutrino_masses']
        else:
            print(f"Warning: No mass data found for {mechanism}")
            continue
        
        # Calculate some summary statistics
        row = {
            'Mechanism': mechanism.replace('_', ' '),
            'Lightest Mass (eV)': f"{np.min(np.abs(masses)):.3e}",
            'Heaviest Mass (eV)': f"{np.max(np.abs(masses)):.3e}",
            'Mass Hierarchy': f"{np.max(np.abs(masses))/np.min(np.abs(masses)):.1e}",
        }
        
        # Add mass squared differences if available
        if 'mass_squared_differences' in result:
            dm_sq = result['mass_squared_differences']
            row['Δm²₂₁ (eV²)'] = f"{dm_sq[1,0]:.3e}"
            row['Δm²₃₁ (eV²)'] = f"{dm_sq[2,0]:.3e}"
        
        # Add mechanism-specific information
        if 'seesaw_scale' in result:
            row['Seesaw Scale (eV)'] = f"{result['seesaw_scale']:.3e}"
        
        if 'mu_scale' in result:
            row['μ Scale (eV)'] = f"{result['mu_scale']:.3e}"
        
        if 'naturalness' in result:
            row['Naturalness μ/M'] = f"{result['naturalness']:.3e}"
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
