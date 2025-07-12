"""
Test suite for matrix utilities module.

This module contains unit tests for the matrix_utils functions
to ensure correctness and numerical stability.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from matrix_utils import (
    diagonalize_hermitian_matrix, diagonalize_mass_matrix,
    check_unitarity, mass_squared_differences,
    enforce_mass_ordering, matrix_condition_number,
    is_positive_definite
)


class TestMatrixUtils(unittest.TestCase):
    """Test cases for matrix utility functions."""
    
    def setUp(self):
        """Set up test matrices."""
        # Hermitian test matrix
        self.hermitian_matrix = np.array([
            [2.0, 1+1j, 0.5-0.5j],
            [1-1j, 3.0, 0.2+0.3j],
            [0.5+0.5j, 0.2-0.3j, 1.5]
        ], dtype=complex)
        
        # Symmetric test matrix (for Majorana case)
        self.symmetric_matrix = np.array([
            [1e-3, 5e-4, 2e-4],
            [5e-4, 2e-3, 8e-4],
            [2e-4, 8e-4, 3e-3]
        ], dtype=complex)
        
        # Identity matrix
        self.identity = np.eye(3, dtype=complex)
        
        # Test masses
        self.test_masses = np.array([0.001, 0.01, 0.05])
    
    def test_diagonalize_hermitian_matrix(self):
        """Test Hermitian matrix diagonalization."""
        eigenvals, eigenvecs = diagonalize_hermitian_matrix(self.hermitian_matrix)
        
        # Check if eigenvalues are real
        self.assertTrue(np.allclose(eigenvals.imag, 0), 
                       "Eigenvalues should be real for Hermitian matrix")
        
        # Check if eigenvectors are orthonormal
        self.assertTrue(check_unitarity(eigenvecs), 
                       "Eigenvectors should form unitary matrix")
        
        # Check reconstruction
        reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        self.assertTrue(np.allclose(self.hermitian_matrix, reconstructed, atol=1e-12),
                       "Matrix reconstruction failed")
    
    def test_diagonalize_mass_matrix_symmetric(self):
        """Test mass matrix diagonalization for symmetric case."""
        masses, mixing = diagonalize_mass_matrix(self.symmetric_matrix, symmetric=True)
        
        # Check if masses are positive
        self.assertTrue(np.all(masses >= 0), "Masses should be non-negative")
        
        # Check unitarity of mixing matrix
        self.assertTrue(check_unitarity(mixing), "Mixing matrix should be unitary")
        
        # Check dimensions
        self.assertEqual(masses.shape, (3,), "Wrong mass eigenvalue shape")
        self.assertEqual(mixing.shape, (3, 3), "Wrong mixing matrix shape")
    
    def test_check_unitarity(self):
        """Test unitarity checking function."""
        # Identity should be unitary
        self.assertTrue(check_unitarity(self.identity), "Identity matrix should be unitary")
        
        # Random orthogonal matrix should be unitary
        Q, _ = np.linalg.qr(np.random.rand(3, 3) + 1j * np.random.rand(3, 3))
        self.assertTrue(check_unitarity(Q), "QR-decomposed matrix should be unitary")
        
        # Non-unitary matrix should fail
        non_unitary = np.array([[1, 1], [0, 1]])
        self.assertFalse(check_unitarity(non_unitary), "Non-unitary matrix should fail check")
    
    def test_mass_squared_differences(self):
        """Test mass squared difference calculation."""
        dm_squared = mass_squared_differences(self.test_masses)
        
        # Check shape
        self.assertEqual(dm_squared.shape, (3, 3), "Wrong shape for mass squared differences")
        
        # Check diagonal elements (should be zero)
        for i in range(3):
            self.assertAlmostEqual(dm_squared[i, i], 0, places=10, 
                                 msg="Diagonal elements should be zero")
        
        # Check antisymmetry
        self.assertTrue(np.allclose(dm_squared, -dm_squared.T, atol=1e-12),
                       "Mass squared differences should be antisymmetric")
        
        # Check specific values
        expected_21 = self.test_masses[1]**2 - self.test_masses[0]**2
        self.assertAlmostEqual(dm_squared[1, 0], expected_21, places=10)
    
    def test_enforce_mass_ordering(self):
        """Test mass ordering enforcement."""
        # Unsorted masses and mixing matrix
        unsorted_masses = np.array([0.05, 0.001, 0.01])
        mixing = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        mixing, _ = np.linalg.qr(mixing)  # Make unitary
        
        # Test normal ordering
        ordered_masses, ordered_mixing = enforce_mass_ordering(
            unsorted_masses, mixing, 'normal')
        
        # Check if masses are in ascending order
        self.assertTrue(np.all(ordered_masses[:-1] <= ordered_masses[1:]),
                       "Masses should be in ascending order for normal ordering")
        
        # Check if mixing matrix is still unitary
        self.assertTrue(check_unitarity(ordered_mixing), 
                       "Ordered mixing matrix should remain unitary")
    
    def test_matrix_condition_number(self):
        """Test matrix condition number calculation."""
        # Well-conditioned matrix (identity)
        cond_identity = matrix_condition_number(self.identity)
        self.assertAlmostEqual(cond_identity, 1.0, places=5, 
                              msg="Identity matrix should have condition number 1")
        
        # Ill-conditioned matrix
        ill_conditioned = np.array([[1, 1], [1, 1+1e-10]])
        cond_ill = matrix_condition_number(ill_conditioned)
        self.assertGreater(cond_ill, 1e8, "Ill-conditioned matrix should have large condition number")
    
    def test_is_positive_definite(self):
        """Test positive definiteness check."""
        # Positive definite matrix
        pos_def = np.array([[2, 1], [1, 2]])
        self.assertTrue(is_positive_definite(pos_def), "Should be positive definite")
        
        # Negative definite matrix
        neg_def = np.array([[-2, -1], [-1, -2]])
        self.assertFalse(is_positive_definite(neg_def), "Should not be positive definite")
        
        # Indefinite matrix
        indefinite = np.array([[1, 0], [0, -1]])
        self.assertFalse(is_positive_definite(indefinite), "Should not be positive definite")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Non-square matrix
        non_square = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            diagonalize_hermitian_matrix(non_square)
        
        # Non-Hermitian matrix
        non_hermitian = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            diagonalize_hermitian_matrix(non_hermitian, check_hermitian=True)
        
        # Empty array
        empty_array = np.array([])
        with self.assertRaises((ValueError, IndexError)):
            mass_squared_differences(empty_array)
    
    def test_numerical_precision(self):
        """Test numerical precision in calculations."""
        # Create matrix with known eigenvalues
        eigenvals_known = np.array([1e-10, 1e-5, 1e-2])
        U = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        U, _ = np.linalg.qr(U)  # Make unitary
        
        test_matrix = U @ np.diag(eigenvals_known) @ U.conj().T
        
        # Diagonalize and check precision
        eigenvals_computed, _ = diagonalize_hermitian_matrix(test_matrix)
        eigenvals_computed = np.sort(eigenvals_computed)
        eigenvals_known = np.sort(eigenvals_known)
        
        relative_error = np.abs(eigenvals_computed - eigenvals_known) / eigenvals_known
        max_error = np.max(relative_error)
        
        self.assertLess(max_error, 1e-10, f"Relative error {max_error} too large")


class TestSpecialCases(unittest.TestCase):
    """Test special cases and boundary conditions."""
    
    def test_zero_matrix(self):
        """Test behavior with zero matrix."""
        zero_matrix = np.zeros((3, 3), dtype=complex)
        
        eigenvals, eigenvecs = diagonalize_hermitian_matrix(zero_matrix)
        
        # All eigenvalues should be zero
        self.assertTrue(np.allclose(eigenvals, 0, atol=1e-12), 
                       "Zero matrix should have zero eigenvalues")
        
        # Eigenvectors should be unitary
        self.assertTrue(check_unitarity(eigenvecs), 
                       "Eigenvectors should be unitary even for zero matrix")
    
    def test_single_element_matrix(self):
        """Test 1x1 matrix."""
        single_element = np.array([[2.5]], dtype=complex)
        
        eigenvals, eigenvecs = diagonalize_hermitian_matrix(single_element)
        
        self.assertAlmostEqual(eigenvals[0], 2.5, places=10)
        self.assertAlmostEqual(eigenvecs[0, 0], 1.0, places=10)
    
    def test_degenerate_eigenvalues(self):
        """Test matrix with degenerate eigenvalues."""
        # Matrix with two equal eigenvalues
        degenerate = np.array([
            [1, 0, 0],
            [0, 1, 0], 
            [0, 0, 2]
        ], dtype=complex)
        
        eigenvals, eigenvecs = diagonalize_hermitian_matrix(degenerate)
        
        # Check eigenvalues (allowing for numerical precision)
        eigenvals_sorted = np.sort(eigenvals)
        expected = np.array([1, 1, 2])
        
        self.assertTrue(np.allclose(eigenvals_sorted, expected, atol=1e-12),
                       "Degenerate eigenvalues not computed correctly")
        
        # Check unitarity
        self.assertTrue(check_unitarity(eigenvecs), 
                       "Eigenvectors should be unitary for degenerate case")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
