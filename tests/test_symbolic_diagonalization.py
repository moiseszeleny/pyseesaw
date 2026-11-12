"""
Test suite for the symbolic diagonalization of the Type I seesaw matrix.

Every symbolic result is checked numerically against an independent Takagi
factorization at random parameter points.
"""

import os
import sys
import unittest

import numpy as np
import sympy as sp

# Add the repository root to the path so that ``src`` is importable as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matrix_utils import (
    seesaw_block_expansion,
    simplify_with_mass_ordering,
    takagi_factorization,
)
from src.seesaw import SymbolicSeesawTypeI
from src.symbolic_diagonalization import (
    NotSolvableInRadicals,
    arrowhead_inverse_parameters,
    arrowhead_mixing_matrix,
    charpoly_matching_equations,
    direct_diagonalization,
    eigenvector_eigenvalue_identity,
    exact_seesaw_parametrization,
    expression_size,
    feasibility_probe,
    lambdify_large,
    majorana_phase_matrix,
    numeric_mass_points,
    orthogonal_matrix,
    reduce_active_space,
    semi_explicit_diagonalization,
    solve_cubic_trigonometric,
    solve_quartic_ferrari,
    solve_reduced_charpoly,
    verify_takagi,
)


def heavy_ranges(model):
    """Sampling ranges that keep the Majorana masses heavy and non-degenerate."""
    return {
        model.M_R_sym[i, i]: (4.0 + 3 * i, 5.0 + 3 * i) for i in range(model.n_sterile)
    }


class TestTakagiFactorization(unittest.TestCase):
    """Numerical Takagi factorization."""

    def test_random_complex_symmetric(self):
        rng = np.random.default_rng(0)
        for dim in (2, 3, 6):
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            M = A + A.T
            masses, U = takagi_factorization(M)
            self.assertTrue(np.all(masses >= 0))
            self.assertTrue(np.all(np.diff(masses) >= 0))
            np.testing.assert_allclose(U.T @ M @ U, np.diag(masses), atol=1e-12)
            np.testing.assert_allclose(U.conj().T @ U, np.eye(dim), atol=1e-12)

    def test_massless_and_degenerate_states(self):
        m_D = np.array([[1.0], [2.0j], [0.5]])
        M = np.block([[np.zeros((3, 3)), m_D], [m_D.T, np.array([[5.0]])]])
        masses, U = takagi_factorization(M)
        np.testing.assert_allclose(masses[:2], 0, atol=1e-12)
        np.testing.assert_allclose(U.T @ M @ U, np.diag(masses), atol=1e-12)

        dirac_pair = np.array([[0, 1], [1, 0]], dtype=complex)
        masses, U = takagi_factorization(dirac_pair)
        np.testing.assert_allclose(masses, [1, 1], atol=1e-12)
        np.testing.assert_allclose(U.T @ dirac_pair @ U, np.eye(2), atol=1e-12)

    def test_rejects_non_symmetric(self):
        with self.assertRaises(ValueError):
            takagi_factorization(np.array([[1, 2], [3, 4]]))


class TestRootSolvers(unittest.TestCase):
    """Closed-form roots of cubics and quartics with real roots."""

    def test_cubic_and_quartic_against_numpy(self):
        rng = np.random.default_rng(1)
        for degree, solver in (
            (3, solve_cubic_trigonometric),
            (4, solve_quartic_ferrari),
        ):
            for _ in range(5):
                roots = np.sort(rng.normal(size=degree))
                coefficients = [sp.Float(c) for c in np.poly(roots)[1:]]
                found = [complex(sp.N(root)) for root in solver(coefficients)]
                np.testing.assert_allclose(found, roots, atol=1e-9)

    def test_biquadratic(self):
        found = [complex(sp.N(r)) for r in solve_quartic_ferrari((0, -5, 0, 4))]
        np.testing.assert_allclose(found, [-2, -1, 1, 2], atol=1e-12)

    def test_factorization_and_degree_limit(self):
        x = sp.Symbol("x", real=True)
        self.assertEqual(
            sorted(solve_reduced_charpoly((x - 1) * (x - 2) * (x**2 - 9), x)),
            [-3, 1, 2, 3],
        )
        a = sp.Symbol("a", real=True)
        with self.assertRaises(NotSolvableInRadicals):
            solve_reduced_charpoly(x**5 + a * x + 1, x)


class TestModelStructure(unittest.TestCase):
    """Counting of massless states and degree of the reduced polynomial."""

    def test_degrees(self):
        for n in (1, 2, 3):
            for m in (1, 2):
                model = SymbolicSeesawTypeI(
                    n, m, dirac="real", majorana_basis="diagonal"
                )
                self.assertEqual(model.n_massless, max(n - m, 0))
                self.assertEqual(model.n_massive, m + min(n, m))
                self.assertEqual(model.reduced_charpoly().degree(), model.n_massive)

    def test_complex_degree(self):
        model = SymbolicSeesawTypeI(2, 1, dirac="polar", majorana_basis="diagonal")
        self.assertEqual(model.reduced_charpoly(hermitian_square=True).degree(), 2)

    def test_polar_dirac_removes_unphysical_phases(self):
        model = SymbolicSeesawTypeI(2, 2, dirac="polar", majorana_basis="diagonal")
        phases = [p for p in model.dirac_phases if p != 0]
        self.assertEqual(len(phases), 2 * (2 - 1))

    def test_default_constructor_unchanged(self):
        model = SymbolicSeesawTypeI(1, 1)
        relations = model.charpoly_relations()
        m_D, M_R = model.m_D_sym[0, 0], model.M_R_sym[0, 0]
        self.assertEqual(relations["coefficients_full_matrix"]["lambda^0"], -(m_D**2))
        self.assertEqual(relations["coefficients_full_matrix"]["lambda^1"], -M_R)
        squared = model.charpoly_relations_MDaggerM()
        self.assertEqual(
            sp.simplify(
                squared["coefficients_full_matrix"]["lambda^1"]
                + M_R**2
                + 2 * m_D * sp.conjugate(m_D)
            ),
            0,
        )


class TestDirectDiagonalization(unittest.TestCase):
    """Masses and mixing from the Lagrangian parameters."""

    def _check(self, n, m, dirac):
        model = SymbolicSeesawTypeI(n, m, dirac=dirac, majorana_basis="diagonal")
        result = direct_diagonalization(model)
        masses = [0] * model.n_massless + result["masses"]
        check = verify_takagi(
            result["U"],
            model.full_mass_matrix_symbolic(),
            masses,
            n_points=3,
            ranges=heavy_ranges(model),
        )
        self.assertTrue(check["passed"], msg=f"({n},{m}) {dirac}: {check}")

    def test_real_cases(self):
        for n, m in ((1, 1), (2, 1), (3, 1), (1, 2)):
            self._check(n, m, "real")

    def test_complex_cases(self):
        for n, m in ((1, 1), (2, 1)):
            self._check(n, m, "polar")

    def test_one_generation_closed_form(self):
        model = SymbolicSeesawTypeI(1, 1, dirac="real", majorana_basis="diagonal")
        result = direct_diagonalization(model)
        m_D, M = model.m_D_sym[0, 0], model.M_R_sym[0, 0]
        light, heavy = result["masses"]
        self.assertEqual(sp.simplify(light * heavy - m_D**2), 0)
        self.assertEqual(sp.simplify(heavy - light - M), 0)

    def test_three_three_has_no_closed_form(self):
        model = SymbolicSeesawTypeI(3, 3, dirac="real", majorana_basis="diagonal")
        with self.assertRaises(NotSolvableInRadicals):
            direct_diagonalization(model, try_factor=False)

    def test_real_quartic(self):
        self._check(2, 2, "real")

    def test_complex_quartic_needs_extended_precision(self):
        """The complex (2,2) closed form is exact but ill-conditioned in doubles."""
        model = SymbolicSeesawTypeI(2, 2, dirac="polar", majorana_basis="diagonal")
        result = direct_diagonalization(model)
        size = expression_size(result["U"])
        self.assertGreater(size["tree"], 10**6)
        self.assertLess(size["dag"], 5000)
        check = verify_takagi(
            result["U"],
            model.full_mass_matrix_symbolic(),
            result["masses"],
            n_points=2,
            ranges=heavy_ranges(model),
            precision=50,
        )
        self.assertTrue(check["passed"], msg=str(check))
        self.assertLess(check["unitarity_error"], 1e-30)

    def test_factorizing_texture(self):
        """Decoupled generations: the sextic factorizes into three quadratics."""
        model = SymbolicSeesawTypeI(3, 3, dirac="real", majorana_basis="diagonal")
        model.set_texture(
            dirac_texture=[(i, j) for i in range(3) for j in range(3) if i != j]
        )
        result = direct_diagonalization(model)
        self.assertTrue(result["factorized"])
        check = verify_takagi(
            result["U"],
            model.full_mass_matrix_symbolic(),
            result["masses"],
            n_points=2,
            ranges=heavy_ranges(model),
        )
        self.assertTrue(check["passed"], msg=str(check))

    def test_semi_explicit_three_three(self):
        """No closed form for the masses, but the mixing matrix is still explicit."""
        model = SymbolicSeesawTypeI(3, 3, dirac="real", majorana_basis="diagonal")
        result = semi_explicit_diagonalization(model)
        check = verify_takagi(
            result["U"],
            model.full_mass_matrix_symbolic(),
            result["masses"],
            points=numeric_mass_points(model, 2),
        )
        self.assertTrue(check["passed"], msg=str(check))

    def test_semi_explicit_two_two(self):
        model = SymbolicSeesawTypeI(2, 2, dirac="real", majorana_basis="diagonal")
        result = semi_explicit_diagonalization(model)
        M_sym = model.full_mass_matrix_symbolic()
        parameters = sorted(M_sym.free_symbols, key=lambda s: s.name)
        rng = np.random.default_rng(3)
        point = {s: rng.uniform(-1, 1) for s in parameters}
        point[model.M_R_sym[0, 0]], point[model.M_R_sym[1, 1]] = 4.0, 7.0
        numeric = np.array(M_sym.subs(point).evalf(), dtype=complex)
        masses, _ = takagi_factorization(numeric)
        point.update(dict(zip(model.massive_masses, masses)))
        check = verify_takagi(result["U"], M_sym, result["masses"], points=[point])
        self.assertTrue(check["passed"], msg=str(check))
        for constraint in result["constraints"]:
            self.assertAlmostEqual(float(constraint.subs(point)), 0, places=8)


class TestHelpers(unittest.TestCase):
    """Phase matrix, reduction of the active space, eigenvector identity."""

    def test_majorana_phase_matrix(self):
        self.assertEqual(majorana_phase_matrix([-1, 1]), sp.diag(sp.I, 1))

    def test_reduce_active_space(self):
        model = SymbolicSeesawTypeI(3, 1, dirac="real", majorana_basis="diagonal")
        V = reduce_active_space(model.m_D_sym)
        reduced = (V.T * model.m_D_sym).applyfunc(sp.simplify)
        self.assertEqual(reduced[0, 0], 0)
        self.assertEqual(reduced[1, 0], 0)
        self.assertEqual((V.T * V).applyfunc(sp.simplify), sp.eye(3))
        norm_squared = sum(x**2 for x in model.m_D_sym)
        self.assertEqual(sp.simplify(reduced[2, 0] ** 2 - norm_squared), 0)

    def test_eigenvector_eigenvalue_identity(self):
        A = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
        eigenvalues, vectors = np.linalg.eigh(np.array(A, dtype=float))
        for i in range(3):
            for a in range(3):
                value = eigenvector_eigenvalue_identity(
                    A, [sp.Float(x) for x in eigenvalues], i, a
                )
                self.assertAlmostEqual(float(value), vectors[a, i] ** 2, places=10)

    def test_simplify_with_mass_ordering(self):
        m1, m2 = sp.symbols("m_1 m_2", positive=True)
        self.assertEqual(
            simplify_with_mass_ordering(sp.sqrt((m1 - m2) ** 2), [m1, m2]), m2 - m1
        )


class TestInverseProblem(unittest.TestCase):
    """Parameters from masses."""

    def test_matching_equations_one_generation(self):
        model = SymbolicSeesawTypeI(1, 1, dirac="real", majorana_basis="diagonal")
        equations = charpoly_matching_equations(model)
        m1, m2 = model.massive_masses
        solution = sp.solve(
            equations, [model.M_R_sym[0, 0], model.m_D_sym[0, 0] ** 2], dict=True
        )[0]
        self.assertEqual(sp.simplify(solution[model.M_R_sym[0, 0]] - (m2 - m1)), 0)
        self.assertEqual(sp.simplify(solution[model.m_D_sym[0, 0] ** 2] - m1 * m2), 0)

    def test_arrowhead_round_trip(self):
        m1 = sp.Symbol("m_1", positive=True)
        for m in (2, 3, 4):
            eigenvalues = [-m1] + list(sp.symbols(f"m_2:{m + 2}", positive=True))
            majorana = list(sp.symbols(f"M_1:{m + 1}", positive=True))
            inverse = arrowhead_inverse_parameters(eigenvalues, majorana)
            dirac = [sp.sqrt(x) for x in inverse["m_D_squared"]]
            M = sp.zeros(m + 1, m + 1)
            for i in range(m):
                M[0, i + 1] = M[i + 1, 0] = dirac[i]
                M[i + 1, i + 1] = majorana[i]
            U = arrowhead_mixing_matrix(eigenvalues, majorana, dirac)

            # Interlaced point satisfying the trace constraint
            values = [-0.1] + [1.0 + 2 * i for i in range(m)]
            fraction = values[-1] / (values[-1] - values[0])
            majorana_values = [
                values[i] + fraction * (values[i + 1] - values[i]) for i in range(m)
            ]
            point = {
                m1: 0.1,
                **dict(zip(eigenvalues[1:], values[1:])),
                **dict(zip(majorana, majorana_values)),
            }
            self.assertAlmostEqual(float(inverse["trace_constraint"].subs(point)), 0)
            check = verify_takagi(U, M, [m1] + eigenvalues[1:], points=[point])
            self.assertTrue(check["passed"], msg=f"m={m}: {check}")

    def test_exact_parametrization_one_generation(self):
        m1, m2 = sp.symbols("m_1 m_2", positive=True)
        result = exact_seesaw_parametrization([m1], [m2], real=True)
        self.assertEqual(sp.simplify(result["m_D"][0, 0] ** 2 - m1 * m2), 0)
        self.assertEqual(sp.simplify(result["M_R"][0, 0] - (m2 - m1)), 0)
        self.assertEqual(result["M"][0, 0], 0)

    def test_exact_parametrization_two_two(self):
        light = sp.symbols("m_1:3", positive=True)
        heavy = sp.symbols("M_1:3", positive=True)
        ranges = {
            light[0]: (0.05, 0.1),
            light[1]: (0.2, 0.3),
            heavy[0]: (2.0, 2.5),
            heavy[1]: (3.0, 3.5),
        }
        theta = sp.Symbol("theta", real=True)
        real_case = exact_seesaw_parametrization(
            light, heavy, orthogonal_matrix(2, [theta]), real=True
        )
        self.assertEqual(real_case["M"][:2, :2], sp.zeros(2, 2))
        check = verify_takagi(
            real_case["U"],
            real_case["M"],
            list(light) + list(heavy),
            n_points=3,
            ranges=ranges,
        )
        self.assertTrue(check["passed"], msg=str(check))
        # m_D is real for a real orthogonal matrix
        point = {
            theta: 0.7,
            light[0]: 0.07,
            light[1]: 0.25,
            heavy[0]: 2.2,
            heavy[1]: 3.1,
        }
        m_D = np.array(real_case["m_D"].subs(point).evalf(), dtype=complex)
        np.testing.assert_allclose(m_D.imag, 0, atol=1e-12)

        z = sp.Symbol("z")
        complex_case = exact_seesaw_parametrization(
            light, heavy, orthogonal_matrix(2, [z])
        )
        check = verify_takagi(
            complex_case["U"],
            complex_case["M"],
            list(light) + list(heavy),
            n_points=3,
            ranges=ranges,
        )
        self.assertTrue(check["passed"], msg=str(check))

    def test_exact_parametrization_three_three(self):
        """Inverse problem beyond the reach of the direct method."""
        light = sp.symbols("m_1:4", positive=True)
        heavy = sp.symbols("M_1:4", positive=True)
        ranges = {
            symbol: (0.05 + 0.1 * i, 0.1 + 0.1 * i) for i, symbol in enumerate(light)
        }
        ranges.update({symbol: (2.0 + i, 2.5 + i) for i, symbol in enumerate(heavy)})
        angles = sp.symbols("theta_1:4", real=True)
        result = exact_seesaw_parametrization(
            light, heavy, orthogonal_matrix(3, angles), real=True
        )
        check = verify_takagi(
            result["U"],
            result["M"],
            list(light) + list(heavy),
            n_points=2,
            ranges=ranges,
        )
        self.assertTrue(check["passed"], msg=str(check))

        complex_angles = sp.symbols("z_1:4")
        result = exact_seesaw_parametrization(
            light, heavy, orthogonal_matrix(3, complex_angles)
        )
        check = verify_takagi(
            result["U"],
            result["M"],
            list(light) + list(heavy),
            n_points=2,
            ranges=ranges,
        )
        self.assertTrue(check["passed"], msg=str(check))

    def test_exact_parametrization_one_active_complex(self):
        """Single light state: X^dagger X is a real number although X is complex."""
        m1, m2, m3 = sp.symbols("m_1 m_2 m_3", positive=True)
        z = sp.Symbol("z")
        ranges = {m1: (0.01, 0.1), m2: (1.0, 2.0), m3: (3.0, 5.0)}
        result = exact_seesaw_parametrization(
            [m1], [m2, m3], sp.Matrix([sp.cos(z), sp.sin(z)])
        )
        check = verify_takagi(result["U"], result["M"], [m1, m2, m3], ranges=ranges)
        self.assertTrue(check["passed"], msg=str(check))

        omega = sp.Symbol("omega", real=True)
        result = exact_seesaw_parametrization(
            [m1], [m2, m3], sp.Matrix([sp.cos(omega), sp.sin(omega)]), real=True
        )
        check = verify_takagi(result["U"], result["M"], [m1, m2, m3], ranges=ranges)
        self.assertTrue(check["passed"], msg=str(check))

    def test_exact_parametrization_with_massless_state(self):
        m2, M1 = sp.symbols("m_2 M_1", positive=True)
        result = exact_seesaw_parametrization([0, m2], [M1], real=True)
        check = verify_takagi(
            result["U"],
            result["M"],
            [0, m2, M1],
            ranges={m2: (0.1, 0.2), M1: (2.0, 3.0)},
        )
        self.assertTrue(check["passed"], msg=str(check))


class TestLargeExpressions(unittest.TestCase):
    """Size measures and evaluation of nested expressions."""

    def test_expression_size(self):
        x, y = sp.symbols("x y")
        shared = sp.sin(x + y)
        expression = shared**2 + shared**3
        size = expression_size(expression)
        self.assertEqual(size["dag"], len(set(sp.preorder_traversal(expression))))
        self.assertEqual(size["tree"], len(list(sp.preorder_traversal(expression))))
        self.assertGreater(size["tree"], size["dag"])

    def test_lambdify_large_matches_lambdify(self):
        x, y = sp.symbols("x y")
        matrix = sp.Matrix([[sp.sqrt(x + y), sp.cos(x) * y], [x / y, sp.exp(sp.I * x)]])
        fast = lambdify_large([x, y], matrix)
        reference = sp.lambdify([x, y], matrix, "numpy")
        np.testing.assert_allclose(
            fast(0.3 + 0j, 1.7 + 0j), reference(0.3 + 0j, 1.7 + 0j)
        )
        vector = lambdify_large([x, y], [x + y, x * y])
        np.testing.assert_allclose(vector(2.0, 3.0), [5.0, 6.0])


class TestSeesawExpansion(unittest.TestCase):
    """Perturbative block diagonalization."""

    def test_leading_order_is_seesaw_formula(self):
        model = SymbolicSeesawTypeI(2, 2)
        expansion = seesaw_block_expansion(model.m_D_sym, model.M_R_sym, order=2)
        expected = -model.m_D_sym * model.M_R_sym.inv() * model.m_D_sym.T
        self.assertEqual(
            (expansion["m_light"] - expected).applyfunc(sp.simplify), sp.zeros(2, 2)
        )

    def test_matches_exact_series_one_generation(self):
        model = SymbolicSeesawTypeI(1, 1, dirac="real", majorana_basis="diagonal")
        m_D = model.m_D_sym[0, 0]
        exact_light, exact_heavy = direct_diagonalization(model)["masses"]
        epsilon = sp.Symbol("epsilon", positive=True)
        expansion = seesaw_block_expansion(model.m_D_sym, model.M_R_sym, order=6)
        for exact, approximate, sign in (
            (exact_light, expansion["m_light"][0, 0], -1),
            (exact_heavy, expansion["m_heavy"][0, 0], 1),
        ):
            series = sp.series(exact.subs(m_D, epsilon * m_D), epsilon, 0, 7).removeO()
            difference = sp.expand(series - sign * approximate.subs(m_D, epsilon * m_D))
            self.assertEqual(sp.simplify(difference), 0)

    def test_convergence_order_complex(self):
        rng = np.random.default_rng(5)
        model = SymbolicSeesawTypeI(2, 2, majorana_basis="diagonal")
        symbols = list(model.m_D_sym) + [model.M_R_sym[0, 0], model.M_R_sym[1, 1]]
        direction = rng.normal(size=4) + 1j * rng.normal(size=4)
        errors = {}
        for order in (2, 4):
            expansion = seesaw_block_expansion(
                model.m_D_sym, model.M_R_sym, order=order
            )
            evaluate = sp.lambdify(symbols, expansion["m_light"], "numpy")
            errors[order] = []
            for epsilon in (0.1, 0.05):
                m_D = epsilon * direction.reshape(2, 2)
                M = np.block([[np.zeros((2, 2)), m_D], [m_D.T, np.diag([3.0, 5.0])]])
                exact, _ = takagi_factorization(M)
                light = np.array(evaluate(*m_D.flatten(), 3.0, 5.0), dtype=complex)
                approximate, _ = takagi_factorization((light + light.T) / 2)
                errors[order].append(np.max(np.abs(approximate - exact[:2])) / exact[1])
        # relative error ~ epsilon^order: halving epsilon gains 2^order
        self.assertGreater(errors[2][0] / errors[2][1], 3.0)
        self.assertGreater(errors[4][0] / errors[4][1], 10.0)
        self.assertLess(errors[4][1], errors[2][1])


class TestFeasibilityProbe(unittest.TestCase):
    """Subprocess-based survey."""

    def test_probe_reports_success_and_obstruction(self):
        ok = feasibility_probe(1, 1, "real", "direct", timeout=60)
        self.assertEqual(ok["status"], "ok")
        self.assertTrue(ok["verified"])
        self.assertEqual(ok["degree"], 2)

        blocked = feasibility_probe(3, 3, "real", "direct", timeout=120)
        self.assertFalse(blocked["solvable_in_radicals"])
        self.assertIn(blocked["status"].split(" (")[0], ("no closed form", "timeout"))


if __name__ == "__main__":
    unittest.main()
