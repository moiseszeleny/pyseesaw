"""
Tests for the confrontation of the Type I seesaw with global-fit data.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_fit import (
    GEV,
    casas_ibarra_exact_numeric,
    chi2,
    exact_light_observables,
    fit_case,
    minimal_model,
    oscillation_parameters,
    rank_argument,
)
from src.matrix_utils import takagi_factorization
from src.pmns_matrix import nufit_6_0, pmns_matrix_pdg


def random_unitary(n, rng):
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


class TestNuFitData(unittest.TestCase):
    def test_tables(self):
        for variant in ("with_sk", "without_sk"):
            data = nufit_6_0(variant)
            self.assertGreater(data["normal"]["dm3l_sq"][0], 0)
            self.assertLess(data["inverted"]["dm3l_sq"][0], 0)
            for ordering in ("normal", "inverted"):
                self.assertGreater(data[ordering]["dm21_sq"][0], 0)
                for key, value in data[ordering].items():
                    if isinstance(value, tuple):
                        bfp, up, down, low, high = value
                        self.assertTrue(low <= bfp <= high, (variant, ordering, key))
                        self.assertGreater(up, 0)
                        self.assertGreater(down, 0)

    def test_angles_consistent_with_sin2(self):
        data = nufit_6_0()
        for ordering in ("normal", "inverted"):
            for ij in ("12", "13", "23"):
                degrees = data[ordering][f"theta{ij}_deg"][0]
                sin2 = data[ordering][f"sin2_theta{ij}"][0]
                self.assertAlmostEqual(np.sin(np.radians(degrees)) ** 2, sin2, delta=0.006)


class TestObservables(unittest.TestCase):
    def test_pdg_round_trip_with_rephasing(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            t12, t13, t23 = rng.uniform(0.05, 1.5, 3)
            delta = rng.uniform(0, 2 * np.pi)
            U = pmns_matrix_pdg(t12, t13, t23, delta)
            U = np.diag(np.exp(1j * rng.uniform(0, 6, 3))) @ U @ np.diag(
                np.exp(1j * rng.uniform(0, 6, 3)))
            p = oscillation_parameters(U)
            self.assertAlmostEqual(p["sin2_theta12"], np.sin(t12) ** 2, places=12)
            self.assertAlmostEqual(p["sin2_theta13"], np.sin(t13) ** 2, places=12)
            self.assertAlmostEqual(p["sin2_theta23"], np.sin(t23) ** 2, places=12)
            self.assertAlmostEqual(p["delta_cp_deg"], np.degrees(delta) % 360, places=7)

    def test_rank_argument(self):
        self.assertEqual(rank_argument(3, 1)["n_massless"], 2)
        self.assertEqual(rank_argument(3, 1)["n_nonzero_splittings"], 1)
        self.assertEqual(rank_argument(3, 2)["n_massless"], 1)
        self.assertEqual(rank_argument(3, 2)["n_nonzero_splittings"], 2)
        self.assertEqual(rank_argument(3, 3)["n_massless"], 0)

    def test_three_one_has_two_massless_states(self):
        rng = np.random.default_rng(2)
        m_D = rng.normal(size=(3, 1)) + 1j * rng.normal(size=(3, 1))
        M = np.block([[np.zeros((3, 3)), m_D], [m_D.T, np.array([[50.0]])]])
        masses, _ = takagi_factorization(M)
        self.assertLess(masses[1], 1e-12 * masses[-1])
        self.assertGreater(masses[2], 1e-3)


class TestCasasIbarra(unittest.TestCase):
    def test_exact_inverse_problem(self):
        """Moderate hierarchy so that double precision checks all orders."""
        rng = np.random.default_rng(3)
        for n, m in ((3, 2), (3, 3), (2, 2)):
            d = np.sort(rng.uniform(0.1, 1.0, n))
            d[: max(n - m, 0)] = 0.0
            D = np.sort(rng.uniform(20.0, 80.0, m))
            z = rng.normal(size=m * (m - 1) // 2) + 1j * rng.normal(size=m * (m - 1) // 2)
            # complex orthogonal m x m from plane rotations, embedded as m x n
            O_square = np.eye(m, dtype=complex)
            k = 0
            for i in range(m):
                for j in range(i + 1, m):
                    R = np.eye(m, dtype=complex)
                    R[i, i] = R[j, j] = np.cos(z[k])
                    R[i, j], R[j, i] = -np.sin(z[k]), np.sin(z[k])
                    O_square = O_square @ R
                    k += 1
            O = np.zeros((m, n), dtype=complex)
            O[:, n - min(n, m):] = O_square[:, : min(n, m)]
            U_a = random_unitary(n, rng)
            res = casas_ibarra_exact_numeric(d, D, O, U_a)
            W, M = res["W"], res["M"]
            np.testing.assert_allclose(W.conj().T @ W, np.eye(n + m), atol=1e-11)
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            np.testing.assert_allclose(W.T @ M @ W, np.diag(np.r_[d, D]), atol=1e-9)
            np.testing.assert_allclose(M[:n, :n], 0, atol=1e-12)


class TestMinimalModel(unittest.TestCase):
    def test_reproduces_best_fit(self):
        data = nufit_6_0()
        for ordering in ("normal", "inverted"):
            for z in (0.0, 0.3 + 1.2j):
                model = minimal_model(ordering, data, z=z, majorana_phase=1.0)
                self.assertLess(chi2(model["observables"], data, ordering), 1e-12)
                self.assertEqual(model["light_masses"][0], 0.0)

    def test_full_matrix_exact_high_precision(self):
        data = nufit_6_0()
        model = minimal_model("normal", data, z=0.2 + 0.5j,
                              heavy_masses_gev=(1e10, 3e12))
        exact = exact_light_observables(model["M_ev"], 3, "normal", dps=80)
        np.testing.assert_allclose(exact["light_masses"][1:],
                                   model["light_masses"][1:], rtol=1e-9)
        self.assertLess(exact["light_masses"][0], 1e-12)
        self.assertLess(chi2(exact, data, "normal"), 1e-8)
        np.testing.assert_allclose(exact["heavy_masses"], [1e10 * GEV, 3e12 * GEV],
                                   rtol=1e-9)

    def test_sum_of_masses(self):
        data = nufit_6_0()
        no = minimal_model("normal", data)["observables"]["sum_masses"]
        io = minimal_model("inverted", data)["observables"]["sum_masses"]
        self.assertAlmostEqual(no, 0.0588, delta=5e-4)
        self.assertAlmostEqual(io, 0.0989, delta=5e-4)


class TestFits(unittest.TestCase):
    def test_one_sterile_excluded_two_viable(self):
        data = nufit_6_0()
        self.assertGreater(fit_case(1, "normal", data, n_starts=3)["chi2"], 100)
        self.assertLess(fit_case(2, "normal", data, n_starts=3)["chi2"], 1e-6)


if __name__ == "__main__":
    unittest.main()
