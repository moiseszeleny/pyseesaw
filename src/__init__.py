"""
Neutrino Mass Matrix Research Package - PySeesaw

A comprehensive Python toolkit for exploring neutrino mass generation mechanisms,
with focus on Seesaw and Inverse Seesaw models in particle physics.

## Physics Background

Neutrinos are among the most abundant particles in the universe, yet their tiny masses
remain one of the greatest puzzles in particle physics. The Standard Model of particle
physics originally assumed neutrinos to be massless, but neutrino oscillation experiments
have conclusively demonstrated that neutrinos do have small but non-zero masses.

### The Neutrino Mass Problem

The measured neutrino masses are extraordinarily small compared to other fermions:
- Electron mass: ~0.511 MeV
- Neutrino masses: < 0.1 eV (at least 5 million times smaller!)

This huge hierarchy suggests that neutrino masses arise through a fundamentally different
mechanism than charged fermion masses.

### Seesaw Mechanisms

The Seesaw mechanism provides an elegant explanation for small neutrino masses by
introducing heavy particles at high energy scales. The general principle is:

    m_ν ~ (Dirac masses)² / (Heavy masses)

This "seesaw" relationship naturally explains why neutrino masses are so small if
the heavy particles have very large masses (e.g., near the GUT scale ~10¹⁶ GeV).

### Types of Seesaw Mechanisms

**Type I Seesaw**: Introduces heavy right-handed neutrinos
- Mass matrix: m_ν = -m_D M_R⁻¹ m_D^T
- Heavy scale: M_R ~ 10¹⁰⁻¹⁶ GeV
- Most minimal and popular model

**Type II Seesaw**: Uses Higgs triplet fields  
- Adds direct contribution: m_ν = v_T × f
- Can work in combination with Type I
- Natural in left-right symmetric models

**Type III Seesaw**: Employs fermionic triplets
- Similar structure to Type I but with triplet fermions
- m_ν = -m_D M_Σ⁻¹ m_D^T

**Inverse Seesaw**: Uses small lepton number violation
- m_ν = m_D M_R⁻¹ μ (M_R^T)⁻¹ m_D^T  
- Allows for lower heavy scales with small μ parameter
- More natural fine-tuning

## Package Structure and Physics Modules

### Core Physics Modules:
- **matrix_utils**: General matrix operations for neutrino physics calculations
  * Hermitian matrix diagonalization for mass matrices
  * Unitarity checks for mixing matrices  
  * Mass ordering and hierarchy analysis
  * Symbolic manipulation with SymPy

- **seesaw**: Implementation of Seesaw mechanisms (Type I, II, III)
  * Symbolic derivations of Seesaw formulas
  * Numerical evaluation and diagonalization
  * Scaling analysis and dimensional relationships
  * Comparison between different Seesaw types

- **inverse_seesaw**: Inverse Seesaw mechanism implementation
  * Full mass matrix construction in extended basis
  * Analytic approximations for light neutrino masses
  * μ parameter dependence analysis
  * Naturalness studies

- **pmns_matrix**: PMNS mixing matrix tools and phenomenology
  * Standard parameterizations (PDG convention)
  * Mixing angle extraction from mass matrices
  * CP violation and Jarlskog invariant
  * Experimental value comparisons
  * Oscillation probability calculations

- **visualization**: Plotting and analysis tools
  * Mass spectrum visualization
  * Mixing matrix heatmaps
  * Parameter space scans
  * Unitarity triangle plots
  * Comparison between mechanisms

## Physics Conventions Used

### Units and Scales:
- Masses in eV (electron volts)
- Energy scales from meV (neutrino masses) to 10¹⁶ GeV (GUT scale)
- Natural units (ℏ = c = 1) assumed throughout

### Sign Conventions:
- Majorana mass terms: +½ ν̄ᶜ M ν + h.c.
- Dirac mass terms: -ν̄_L m_D ν_R + h.c.  
- Seesaw formula: m_ν = -m_D M_R⁻¹ m_D^T

### Matrix Conventions:
- PMNS matrix: |ν_α⟩ = Σᵢ U_αᵢ |νᵢ⟩
- Mass ordering: normal (m₁ < m₂ < m₃), inverted (m₃ < m₁ < m₂)
- Phases: Dirac phase δ_CP and Majorana phases α₁, α₂

## Typical Usage Workflow

1. **Symbolic Analysis**: Use SymbolicSeesawTypeI for pedagogical understanding
2. **Numerical Evaluation**: Apply SeesawTypeI with realistic parameters  
3. **Phenomenology**: Extract mixing angles and compare with experiments
4. **Visualization**: Create plots to understand mass hierarchies and mixing patterns

## Educational Features

This package is designed for both research and education in neutrino physics:
- Step-by-step symbolic derivations
- Clear documentation of physics assumptions
- Comparison with experimental data
- Visualization tools for intuitive understanding
- Examples covering different physical scenarios

## References and Further Reading

Key papers on Seesaw mechanisms:
- Minkowski (1977): Original Type I Seesaw
- Mohapatra & Senjanović (1980): Left-right symmetric models
- Schechter & Valle (1980): Type II Seesaw
- Mohapatra (1986): Type III Seesaw and see-saw in SUSY
- Gonzalez-Garcia & Yokoyama (2013): Modern neutrino mass reviews

Experimental data:
- NuFIT collaboration: Global fits of neutrino oscillation data
- PDG: Particle Data Group neutrino mass and mixing summaries
"""

__version__ = "0.1.0"
__author__ = "Neutrino Physics Research"

from .matrix_utils import *
from .seesaw import *
from .inverse_seesaw import *
from .pmns_matrix import *
