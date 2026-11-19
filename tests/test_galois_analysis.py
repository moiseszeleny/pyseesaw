"""
Tests for the Galois analysis, the nested-definition export, the solvable
functional decompositions and the high-precision exact Casas-Ibarra numerics.
"""

import os
import random
import sys
import unittest

import mpmath
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exact_seesaw_numeric import (  # noqa: E402
    casas_ibarra_exact_numeric,
    casas_ibarra_leading_order,
    casas_ibarra_orthogonal,
    light_sector_observables,
    required_precision,
    takagi_factorization_mp,
)
from src.galois_analysis import (  # noqa: E402
    Texture,
    galois_of_poly,
    generic_galois_group,
    named_textures,
    reduced_polynomial,
    specialized_mass_matrix,
    texture_scan,
)
from src.matrix_utils import takagi_factorization  # noqa: E402
from src.pmns_matrix import pmns_from_experimental  # noqa: E402
from src.seesaw import SymbolicSeesawTypeI  # noqa: E402
from src.symbolic_diagonalization import (  # noqa: E402
    NotSolvableInRadicals,
    direct_diagonalization,
    nested_definitions,
    solve_reduced_charpoly,
    verify_takagi,
)


class TestGaloisOfPoly(unittest.TestCase):
    """Galois groups of explicit rational polynomials."""

    x = sp.Symbol("x")

    def test_known_groups(self):
        x = self.x
        cases = {
            x**5 - x - 1: ("S5", False),
            x**3 - 3 * x + 1: ("A3", True),
            x**4 - 2: ("D4", True),
        }
        for poly, (group, solvable) in cases.items():
            result = galois_of_poly(sp.Poly(poly, x))
            self.assertEqual(result["groups"], group)
            self.assertEqual(result["solvable"], solvable)

    def test_real_radicals(self):
        # x^3 - 3x + 1: three real roots, group of order 3 -> casus irreducibilis
        self.assertFalse(galois_of_poly(sp.Poly(self.x**3 - 3 * self.x + 1))["real_radicals"])
        # (x^2 - 2)(x^2 - 3): real square roots suffice
        result = galois_of_poly(sp.Poly((self.x**2 - 2) * (self.x**2 - 3)))
        self.assertTrue(result["real_radicals"])
        self.assertEqual(result["factor_degrees"], (2, 2))


class TestSpecialization(unittest.TestCase):
    """Exact specializations of the seesaw polynomial."""

    def test_massless_states_removed(self):
        rng = random.Random(3)
        for n, m in [(3, 1), (3, 2)]:
            for cp in ("real", "complex"):
                poly, zeros = reduced_polynomial(specialized_mass_matrix(n, m, cp, rng=rng), cp)
                self.assertEqual(zeros, n - m)
                self.assertEqual(poly.degree(), 2 * m)
                self.assertTrue(all(c.is_rational for c in poly.all_coeffs()))

    def test_texture_imposed(self):
        texture = Texture(dirac_zeros=((0, 0),), majorana_degenerate=((0, 1),))
        matrix = specialized_mass_matrix(2, 2, "complex", texture)
        self.assertEqual(matrix[0, 2], 0)
        self.assertEqual(matrix[2, 2], matrix[3, 3])


class TestGenericGroups(unittest.TestCase):
    """Generic Galois groups of every case up to (3, 3)."""

    expected = {
        (1, 1): "S2", (2, 1): "S2", (3, 1): "S2", (1, 2): "S3",
        (2, 2): "S4", (3, 2): "S4", (1, 3): "S4", (2, 3): "S5", (3, 3): "S6",
    }

    def test_generic_cases(self):
        for cp in ("real", "complex"):
            for (n, m), group in self.expected.items():
                with self.subTest(n=n, m=m, cp=cp):
                    result = generic_galois_group(n, m, cp, n_trials=6)
                    self.assertEqual(result["groups"], group)
                    self.assertEqual(result["status"], "proven")
                    self.assertTrue(result["group_exact"])
                    self.assertEqual(result["solvable"], m + min(n, m) <= 4)
                    # Real radicals only for the quadratic cases
                    self.assertEqual(result["real_radicals"], m + min(n, m) <= 2)

    def test_degenerate_heavy_neutrinos(self):
        degenerate = Texture(majorana_degenerate=((0, 1, 2),), name="degenerate")
        real = generic_galois_group(3, 3, "real", degenerate, n_trials=6)
        self.assertEqual(real["groups"], "S4xC2")  # = C2 wr S3, three (1,1) blocks
        self.assertEqual(real["decompositions"], ["3o2"])
        self.assertTrue(real["solvable"])
        self.assertTrue(real["group_exact"])
        complex_case = generic_galois_group(3, 3, "complex", degenerate, n_trials=6)
        self.assertEqual(complex_case["groups"], "S6")
        self.assertFalse(complex_case["solvable"])

    def test_named_textures(self):
        diagonal = named_textures(3, 3)[0]
        result = generic_galois_group(3, 3, "real", diagonal, n_trials=6)
        self.assertEqual(result["groups"], "S2 x S2 x S2")
        self.assertTrue(result["real_radicals"])
        two_degenerate = Texture(majorana_degenerate=((0, 1),))
        result = generic_galois_group(3, 2, "real", two_degenerate, n_trials=6)
        self.assertEqual(result["groups"], "D4")
        self.assertTrue(result["real_radicals"])

    def test_texture_scan_two_three(self):
        records = texture_scan(2, 3, "real", n_trials=4)
        solvable = [r for r in records if r["solvable"]]
        # only the pattern that splits the matrix into (1,1) + (1,2) blocks
        self.assertEqual(len(solvable), 1)
        self.assertEqual(solvable[0]["blocks"], 2)
        self.assertTrue(all(r["solvable"] is False for r in records if r["blocks"] == 1))


class TestDecomposition(unittest.TestCase):
    """Closed forms through functional decompositions."""

    def test_composition_roots(self):
        x = sp.Symbol("x")
        # (x^2 - x)^3 - 7 (x^2 - x) + 6, roots from y^3 - 7y + 6 = (y-1)(y-2)(y+3)
        y = x**2 - x
        poly = sp.Poly(sp.expand((y - 1) * (y - 2) * (y - 5)), x)
        roots = solve_reduced_charpoly(poly, try_factor=False)
        self.assertEqual(len(roots), 6)
        numeric = sorted(complex(sp.N(r)).real for r in roots)
        reference = sorted(np.roots([float(c) for c in poly.all_coeffs()]).real)
        np.testing.assert_allclose(numeric, reference, atol=1e-10)

    def test_degree_six_without_decomposition_raises(self):
        x = sp.Symbol("x")
        with self.assertRaises(NotSolvableInRadicals):
            solve_reduced_charpoly(sp.Poly(x**6 - x - 1, x), try_factor=False)

    def test_degenerate_two_two_and_three_three(self):
        for n in (2, 3):
            model = SymbolicSeesawTypeI(n, n, dirac="real", majorana_basis="diagonal")
            M1 = model.M_R_sym[0, 0]
            model.M_R_sym = sp.diag(*[M1] * n)
            model._full_mass_matrix_symbolic = None
            if n == 3:
                result = direct_diagonalization(model)
                check = verify_takagi(result["U"], model.full_mass_matrix_symbolic(),
                                      result["masses"], n_points=2, ranges={M1: (4.0, 5.0)})
                self.assertTrue(check["passed"])
                self.assertTrue(result["factorized"])


class TestNestedDefinitions(unittest.TestCase):
    """Chains of definitions reproduce the original expressions."""

    def test_round_trip(self):
        model = SymbolicSeesawTypeI(2, 2, dirac="real", majorana_basis="diagonal")
        result = direct_diagonalization(model)
        expressions = list(result["masses"]) + [result["U"][0, 0], result["U"][3, 2]]
        names = ["m_1", "m_2", "m_3", "m_4", "U_11", "U_43"]
        chain = nested_definitions(expressions, names, max_inline=20)
        self.assertGreater(chain["n_definitions"], 5)
        self.assertIn("&=", chain["latex"])
        symbols = sorted(set(model.m_D_sym) | {model.M_R_sym[0, 0], model.M_R_sym[1, 1]},
                         key=lambda s: s.name)
        point = dict(zip(symbols, [0.3, -0.2, 0.25, 0.4, 4.2, 7.7]))
        values = {s: sp.Float(v, 30) for s, v in point.items()}
        for symbol, expression in chain["definitions"]:
            values[symbol] = sp.N(expression.xreplace(values), 30)
        outputs = {str(s): complex(sp.N(e.xreplace(values), 30)) for s, e in chain["outputs"]}
        outputs.update({str(s): complex(values[s]) for s, _ in chain["definitions"]
                        if str(s) in names})
        with mpmath.workdps(30):
            for name, expression in zip(names, expressions):
                reference = complex(sp.N(expression.xreplace(values), 30))
                self.assertAlmostEqual(abs(outputs[name] - reference), 0.0, places=12)


class TestExactCasasIbarra(unittest.TestCase):
    """High-precision exact parametrization of the (3,2) minimal seesaw."""

    def setUp(self):
        ev = 1e-9
        self.d = [0.0, 8.6e-3 * ev, 5.0e-2 * ev]
        self.U = pmns_from_experimental("normal")

    def test_exact_reproduces_inputs(self):
        D = [100.0, 150.0]
        with mpmath.workdps(required_precision(self.d[1], D[1], 8.0)):
            O = casas_ibarra_orthogonal(0.3 + 8j)
            model = casas_ibarra_exact_numeric(self.d, D, O, self.U)
            masses, _ = takagi_factorization_mp(model["M"])
            for got, want in zip(masses, sorted(self.d + D)):
                self.assertLess(abs(got - want), mpmath.mpf(10) ** -25 * max(want, 1e-300) + 1e-40)
            observables = light_sector_observables(model["M"])
        self.assertAlmostEqual(observables["s13"], abs(self.U[0, 2]) ** 2, places=12)
        self.assertGreater(observables["U2"], 1e-8)

    def test_leading_order_error_scales_with_mixing(self):
        D = [10.0, 15.0]
        for y in (2.0, 6.0):
            with mpmath.workdps(required_precision(self.d[1], D[1], y)):
                O = casas_ibarra_orthogonal(0.3 + 1j * y)
                exact = light_sector_observables(casas_ibarra_exact_numeric(self.d, D, O, self.U)["M"])
                lo = light_sector_observables(casas_ibarra_leading_order(self.d, D, O, self.U)["M"])
            deviation = abs(lo["dm3l"] / exact["dm3l"] - 1)
            self.assertLess(deviation, 10 * exact["U2"])
            self.assertGreater(deviation, 0.1 * exact["U2"])

    def test_double_precision_fails_at_high_scale(self):
        D = [1e8, 1.5e8]
        with mpmath.workdps(required_precision(self.d[1], D[1])):
            model = casas_ibarra_exact_numeric(self.d, D, casas_ibarra_orthogonal(0.3), self.U)
            matrix = np.array(model["M"].tolist(), dtype=complex)
        masses, _ = takagi_factorization(matrix)
        self.assertGreater(abs(np.sort(masses)[1] / self.d[1] - 1), 1e-2)


if __name__ == "__main__":
    unittest.main()
