r"""
Galois groups of the Type I seesaw characteristic polynomial.

The masses of the Type I seesaw matrix

.. math:: M = \begin{pmatrix} 0 & m_D \\ m_D^T & M_R \end{pmatrix},
          \qquad M_R = \mathrm{diag}(M_1, \dots, M_m),

are the roots of the reduced characteristic polynomial ``P(lambda)`` of degree
``k = m + min(n, m)`` (``det(lambda - M)`` for real ``m_D``, whose roots are
the signed masses, or ``det(lambda - M^\dagger M)`` for complex ``m_D``, whose
roots are the squared masses), see
:meth:`src.seesaw.SymbolicSeesawTypeI.reduced_charpoly`. Whether the masses
(and with them the mixing matrix) have a closed form in radicals is decided by
the Galois group ``G`` of ``P`` over the field ``Q(parameters)``.

Proof strategy
--------------
The coefficients of ``P`` are polynomials with rational coefficients in the
Lagrangian parameters and ``P`` is monic. For a rational (or, for complex
``m_D``, Gaussian-rational) parameter point at which the specialized
polynomial ``P_0`` stays separable, the Galois group of ``P_0`` over ``Q`` is
isomorphic to a subgroup of ``G`` (decomposition group of a good
specialization; e.g. Lang, *Algebra*, VII.2, or Serre, *Topics in Galois
Theory*, 3.3). Hence:

* **Non-solvability is proven by one specialization**: if the group of any
  irreducible factor of ``P_0`` is not solvable, ``G`` is not solvable and the
  masses admit no formula in radicals. In particular one specialization with
  group ``S_k`` proves ``G = S_k`` (because always ``G <= S_k``).
* **Solvability is proven by the generic factorization**: if ``P`` factorizes
  over ``Q(parameters)`` into factors of degree ``<= 4`` (or ``P`` itself has
  degree ``<= 4``), each factor is solvable in radicals.
* **Real radicals.** All roots of ``P`` are real. By the theorem on real
  radicals (casus irreducibilis generalized; Isaacs, Amer. Math. Monthly 92
  (1985) 571), a root of an irreducible polynomial with only real roots lies
  in a tower of *real* radicals only if the order of its Galois group is a
  power of two. For ``S_3`` and ``S_4`` this fails: the cubic and quartic
  closed forms necessarily pass through complex intermediate quantities (or
  are written trigonometrically, as in
  :func:`src.symbolic_diagonalization.solve_cubic_trigonometric`).

The Galois groups of the specialized factors are computed with
:func:`sympy.polys.numberfields.galoisgroups.galois_group`, which covers
degrees up to 6, i.e. every case up to three active and three sterile
neutrinos.
"""

import itertools
import multiprocessing
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp
from sympy.polys.numberfields.galoisgroups import galois_group

__all__ = [
    "Texture",
    "specialized_mass_matrix",
    "symbolic_mass_matrix",
    "reduced_polynomial",
    "galois_of_poly",
    "generic_galois_group",
    "generic_factorization",
    "texture_scan",
    "named_textures",
    "galois_table",
]

_LAMBDA = sp.Symbol("x")


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Texture:
    r"""
    Special structure of the seesaw parameters.

    Attributes:
    -----------
    dirac_zeros : tuple of (int, int)
        Positions ``(i, j)`` (0-based) with ``(m_D)_ij = 0``
    majorana_degenerate : tuple of tuple of int
        Groups of sterile indices with equal Majorana masses, e.g. ``((0, 1),)``
        for ``M_1 = M_2``
    dirac_equal_rows : tuple of (int, int)
        Pairs of active indices with equal rows of ``m_D`` (e.g. ``(1, 2)``
        for a mu-tau symmetric Dirac matrix with diagonal ``M_R``)
    name : str
        Label used in tables
    """

    dirac_zeros: Tuple[Tuple[int, int], ...] = ()
    majorana_degenerate: Tuple[Tuple[int, ...], ...] = ()
    dirac_equal_rows: Tuple[Tuple[int, int], ...] = ()
    name: str = "generic"

    def zero_pattern(self, n: int, m: int) -> str:
        """Texture of ``m_D`` as rows of ``x`` (free) and ``0`` (zero)."""
        zeros = set(self.dirac_zeros)
        return "/".join(
            "".join("0" if (i, j) in zeros else "x" for j in range(m))
            for i in range(n)
        )


def _majorana_values(m: int, texture: Texture, draw) -> List[sp.Expr]:
    values = [draw() for _ in range(m)]
    for group in texture.majorana_degenerate:
        for index in group[1:]:
            values[index] = values[group[0]]
    return values


def _apply_dirac_relations(m_D: sp.Matrix, texture: Texture) -> sp.Matrix:
    m_D = m_D.copy()
    for i, j in texture.dirac_zeros:
        m_D[i, j] = 0
    for first, second in texture.dirac_equal_rows:
        m_D[second, :] = m_D[first, :]
    return m_D


def _block_matrix(m_D: sp.Matrix, M_R: sp.Matrix) -> sp.Matrix:
    n, m = m_D.shape
    return sp.Matrix.vstack(
        sp.Matrix.hstack(sp.zeros(n, n), m_D), sp.Matrix.hstack(m_D.T, M_R)
    )


def specialized_mass_matrix(
    n: int,
    m: int,
    cp: str = "real",
    texture: Texture = Texture(),
    rng: Optional[random.Random] = None,
    height: int = 9,
) -> sp.Matrix:
    r"""
    Seesaw mass matrix at a random rational parameter point.

    ``m_D`` has random rational entries (``cp='real'``) or Gaussian-rational
    entries ``a + i b`` (``cp='complex'``), ``M_R`` is diagonal with positive
    rational entries; the texture is imposed exactly.

    Parameters:
    -----------
    n, m : int
        Number of active and sterile neutrinos
    cp : {'real', 'complex'}, optional
        Real or complex Dirac matrix
    texture : Texture, optional
        Zeros and relations to impose
    rng : random.Random, optional
        Random generator (default: seeded with 0)
    height : int, optional
        Bound on numerators of the random rationals

    Returns:
    --------
    sp.Matrix
        Exact ``(n+m) x (n+m)`` mass matrix
    """
    rng = rng or random.Random(0)

    def rational() -> sp.Rational:
        numerator = 0
        while numerator == 0:
            numerator = rng.randint(-height, height)
        return sp.Rational(numerator, rng.randint(1, 5))

    def positive() -> sp.Rational:
        return sp.Rational(rng.randint(1, 4 * height), rng.randint(1, 3))

    def entry() -> sp.Expr:
        return rational() + sp.I * rational() if cp == "complex" else rational()

    m_D = _apply_dirac_relations(sp.Matrix(n, m, lambda i, j: entry()), texture)
    M_R = sp.diag(*_majorana_values(m, texture, positive))
    return _block_matrix(m_D, M_R)


def symbolic_mass_matrix(
    n: int, m: int, cp: str = "real", texture: Texture = Texture()
) -> sp.Matrix:
    r"""
    Seesaw mass matrix with symbolic parameters, for the generic factorization.

    Real case: ``(m_D)_ij = a_ij`` real. Complex case:
    ``(m_D)_ij = a_ij + i b_ij`` with ``a_ij, b_ij`` real, so that the
    coefficients of ``det(lambda - M^\dagger M)`` are polynomials with
    rational coefficients in the ``a, b`` and ``M_i``. (Unphysical phases are
    not removed; they do not change the polynomial's Galois group.)
    """

    def entry(i: int, j: int) -> sp.Expr:
        a = sp.Symbol(f"a_{i + 1}{j + 1}", real=True)
        if cp == "complex":
            return a + sp.I * sp.Symbol(f"b_{i + 1}{j + 1}", real=True)
        return a

    m_D = _apply_dirac_relations(sp.Matrix(n, m, entry), texture)
    majorana = iter(sp.Symbol(f"M_{i + 1}", positive=True) for i in range(m))
    M_R = sp.diag(*_majorana_values(m, texture, lambda: next(majorana)))
    return _block_matrix(m_D, M_R)


def reduced_polynomial(
    mass_matrix: sp.Matrix, cp: str = "real", lamb: sp.Symbol = _LAMBDA
) -> Tuple[sp.Poly, int]:
    r"""
    Characteristic polynomial with the factor ``lambda^z`` (massless states) removed.

    Parameters:
    -----------
    mass_matrix : sp.Matrix
        Exact or symbolic seesaw matrix
    cp : {'real', 'complex'}, optional
        ``'real'``: ``det(lambda - M)``; ``'complex'``: ``det(lambda - M^\dagger M)``
    lamb : sp.Symbol, optional
        Polynomial variable

    Returns:
    --------
    (sp.Poly, int)
        Reduced monic polynomial and the number ``z`` of zero roots removed
    """
    operator = mass_matrix
    if cp == "complex":
        operator = (mass_matrix.H * mass_matrix).applyfunc(sp.expand)
    coefficients = [sp.expand(c) for c in operator.charpoly(lamb).all_coeffs()]
    zeros = 0
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
        zeros += 1
    return sp.Poly(coefficients, lamb), zeros


# ---------------------------------------------------------------------------
# Galois groups
# ---------------------------------------------------------------------------


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _factor_group(factor: sp.Poly) -> Dict[str, object]:
    degree = factor.degree()
    if degree == 1:
        return {"degree": 1, "group": "1", "order": 1, "solvable": True}
    group, _ = galois_group(factor, by_name=True)
    permutation_group = group.get_perm_group()
    return {
        "degree": degree,
        "group": group.name,
        "order": int(permutation_group.order()),
        "solvable": bool(permutation_group.is_solvable),
    }


def galois_of_poly(poly: sp.Poly) -> Dict[str, object]:
    r"""
    Factorization over ``Q`` and Galois group of each irreducible factor.

    Parameters:
    -----------
    poly : sp.Poly
        Univariate polynomial with rational coefficients (degree of each
        irreducible factor at most 6)

    Returns:
    --------
    dict
        ``'factors'`` (list of dicts with ``degree``, ``multiplicity``,
        ``group``, ``order``, ``solvable``), ``'separable'``, ``'repeated'``
        (number of repeated roots), ``'solvable'``
        (all factors solvable), ``'real_radicals'`` (every factor group a
        2-group: roots expressible by real radicals),
        ``'factor_degrees'``, ``'groups'`` (e.g. ``'S2 x S2'``) and
        ``'max_order'`` (product of the factor group orders)
    """
    lamb = poly.gen
    _, factor_list = sp.factor_list(poly.as_expr(), lamb)
    factors = []
    for factor, multiplicity in factor_list:
        record = _factor_group(sp.Poly(factor, lamb))
        record["multiplicity"] = multiplicity
        factors.append(record)
    factors.sort(key=lambda f: (-f["degree"], f["group"]))
    order = 1
    for f in factors:
        order *= f["order"]
    return {
        "factors": factors,
        "separable": all(f["multiplicity"] == 1 for f in factors),
        "repeated": sum((f["multiplicity"] - 1) * f["degree"] for f in factors),
        "solvable": all(f["solvable"] for f in factors),
        "real_radicals": all(_is_power_of_two(f["order"]) for f in factors),
        "factor_degrees": tuple(f["degree"] for f in factors),
        "groups": " x ".join(f["group"] for f in factors) or "1",
        "max_order": order,
    }


def _wreath_bound(components: Sequence[int]) -> int:
    r"""
    Order of the iterated wreath product bounding the group of ``g_1(g_2(...))``.

    For ``f = g o h`` with ``deg g = a`` and ``deg h = b`` the roots of ``f``
    are the roots of ``h(x) = y_i``, ``y_i`` the roots of ``g``, so the Galois
    group is a subgroup of ``S_b wr S_a`` of order ``(b!)^a a!``.
    """
    order = 1
    for degree in reversed(components):
        order = sp.factorial(degree) * order**degree
    return int(order)


def _connected_blocks(mass_matrix: sp.Matrix) -> List[List[int]]:
    """
    Index sets of the connected blocks of a mass matrix (graph of its non-zero
    off-diagonal entries). The characteristic polynomial is the product of
    those of the blocks.
    """
    size = mass_matrix.rows
    unvisited = set(range(size))
    blocks = []
    while unvisited:
        stack = [unvisited.pop()]
        block = set(stack)
        while stack:
            i = stack.pop()
            for j in list(unvisited):
                if mass_matrix[i, j] != 0 or mass_matrix[j, i] != 0:
                    unvisited.discard(j)
                    block.add(j)
                    stack.append(j)
        blocks.append(sorted(block))
    return blocks


def _factorization_worker(queue, n, m, cp, texture, decompose) -> None:
    try:
        mass_matrix = symbolic_mass_matrix(n, m, cp, texture)
        factor_list = []
        zeros = 0
        for block in _connected_blocks(mass_matrix):
            polynomial, block_zeros = reduced_polynomial(
                mass_matrix.extract(block, block), cp
            )
            zeros += block_zeros
            if polynomial.degree() > 0:
                factor_list += sp.factor_list(polynomial.as_expr(), _LAMBDA)[1]
        factors = []
        for factor, multiplicity in factor_list:
            factor_poly = sp.Poly(factor, _LAMBDA)
            components: Tuple[int, ...] = (factor_poly.degree(),)
            if decompose and factor_poly.degree() >= 4:
                components = tuple(g.degree() for g in factor_poly.decompose())
            factors.append(components)  # distinct factors only
        factors.sort(key=lambda c: -int(sp.prod(c)))
        queue.put({"factors": factors, "zeros": zeros})
    except Exception as error:  # report, do not crash the scan
        queue.put({"error": f"{type(error).__name__}: {error}"})


def generic_factorization(
    n: int,
    m: int,
    cp: str = "real",
    texture: Texture = Texture(),
    timeout: float = 120.0,
    decompose: bool = True,
) -> Optional[Dict[str, object]]:
    r"""
    Irreducible factors of the reduced polynomial over ``Q(parameters)``.

    Each factor of degree ``>= 4`` is also decomposed functionally,
    ``f = g_1 o g_2 o ...`` (:meth:`sympy.Poly.decompose`): a factor of
    degree 6 that is a cubic of a quadratic is solvable although its degree
    exceeds 4. This happens for degenerate heavy neutrinos, where
    ``P(lambda) = Q(lambda (lambda - M))`` with ``Q`` the characteristic
    polynomial of ``m_D m_D^T``.

    Runs in a subprocess with a time limit (multivariate factorization can be
    slow for many parameters).

    Returns:
    --------
    dict or None
        ``{'factors': list of tuples, 'zeros': int}``, where each tuple lists
        the degrees of the functional components of one distinct irreducible
        factor (outermost first); ``None`` on timeout or error
    """
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    result = None
    # A forked worker occasionally stalls on start-up; one retry is cheap
    # compared with losing the result.
    for _ in range(2):
        queue = context.Queue()
        process = context.Process(
            target=_factorization_worker, args=(queue, n, m, cp, texture, decompose)
        )
        process.start()
        try:
            result = queue.get(timeout=timeout)
        except Exception:  # queue.Empty: timed out
            result = None
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
        if result is not None:
            break
    if result is None or "error" in result:
        return None
    return result


def generic_galois_group(
    n: int,
    m: int,
    cp: str = "real",
    texture: Texture = Texture(),
    n_trials: int = 12,
    seed: int = 0,
    factorize: Optional[bool] = None,
    timeout: float = 120.0,
) -> Dict[str, object]:
    r"""
    Galois group of the reduced seesaw polynomial over ``Q(parameters)``.

    Computes the Galois groups of ``n_trials`` separable specializations
    (lower bounds, see the module docstring) and, if needed, the generic
    factorization (upper bound on the factor degrees).

    Parameters:
    -----------
    n, m : int
        Number of active and sterile neutrinos
    cp : {'real', 'complex'}, optional
        Real Dirac matrix (polynomial ``det(lambda - M)``) or complex one
        (``det(lambda - M^\dagger M)``)
    texture : Texture, optional
        Zeros and relations of the parameters
    n_trials : int, optional
        Number of random specializations
    seed : int, optional
        Seed of the random generator
    factorize : bool, optional
        Whether to compute the generic factorization. Default: only when it
        is needed to decide solvability, i.e. when no specialization shows a
        non-solvable group and the generic degree exceeds 4.
    timeout : float, optional
        Time limit for the generic factorization

    Returns:
    --------
    dict
        ``n``, ``m``, ``cp``, ``texture`` (name), ``pattern``, ``degree``
        (number of distinct roots of the reduced polynomial), ``massless``
        (number of zero roots), ``repeated`` (number of repeated roots),
        ``factor_degrees`` and ``groups`` (distinct irreducible factors of the
        specialization with the
        largest group; a lower bound for the generic group), ``group_order``,
        ``solvable`` and ``real_radicals`` (``True``/``False`` when proven,
        ``None`` when undecided), ``generic_degrees`` (degrees of the generic
        irreducible factors, if known), ``decompositions`` (functional
        decomposition of each generic factor, e.g. ``'3o2'`` for a cubic of a
        quadratic), ``group_exact`` (each generic factor
        has group ``S_d``, so ``groups`` is exact factor by factor),
        ``status`` (``'proven'`` if solvability is decided, else
        ``'lower bound'``) and ``proof`` (a short justification), plus
        ``trials``
    """
    rng = random.Random(seed)
    records = []
    for _ in range(n_trials):
        matrix = specialized_mass_matrix(n, m, cp, texture, rng)
        polynomial, zeros = reduced_polynomial(matrix, cp)
        if polynomial.degree() == 0:
            continue
        analysis = galois_of_poly(polynomial)
        analysis["zeros"] = zeros
        records.append(analysis)

    # Generic numbers of zero and of repeated roots (e.g. decoupled sterile
    # states for degenerate M_R): the minimum over the points. Good
    # specializations have the generic multiplicity structure; the Galois
    # group is that of the distinct irreducible factors.
    generic_zeros = min(r["zeros"] for r in records)
    candidates = [r for r in records if r["zeros"] == generic_zeros]
    generic_repeated = min(r["repeated"] for r in candidates)
    good = [r for r in candidates if r["repeated"] == generic_repeated]
    best = max(good, key=lambda r: (r["max_order"], -len(r["factors"])))
    degree = sum(best["factor_degrees"])  # distinct roots

    result: Dict[str, object] = {
        "n": n,
        "m": m,
        "cp": cp,
        "texture": texture.name,
        "pattern": texture.zero_pattern(n, m),
        "degree": degree,
        "massless": generic_zeros,
        "repeated": generic_repeated,
        "factor_degrees": best["factor_degrees"],
        "groups": best["groups"],
        "group_order": best["max_order"],
        "solvable": best["solvable"],
        "real_radicals": best["real_radicals"],
        "generic_degrees": None,
        "trials": len(good),
    }

    # --- Solvability, real radicals and the exact group, each with its proof.
    # Lower bounds come from the specialization, upper bounds from the generic
    # factors and their functional decompositions (iterated wreath products).
    spec_non_solvable = not best["solvable"]
    spec_not_2_group = not best["real_radicals"]
    irreducible_spec = len(best["factors"]) == 1
    if factorize is None:
        spec_is_bound = irreducible_spec and best["groups"] == f"S{degree}"
        factorize = not (spec_non_solvable or spec_is_bound or degree <= 2)
    generic = _generic_factors(n, m, cp, texture, timeout) if factorize else None
    if generic is None and (irreducible_spec or degree <= 1):
        # an irreducible specialization implies an irreducible generic polynomial
        generic = [best["factor_degrees"]] if degree else []
    result["generic_degrees"] = (
        None if generic is None else tuple(int(sp.prod(c)) for c in generic)
    )
    result["decompositions"] = (
        None if generic is None else ["o".join(map(str, c)) for c in generic]
    )

    # Without the generic factorization the whole polynomial is one "factor"
    # of degree k (group <= S_k), which is still a valid upper bound.
    bound = generic if generic is not None else [(degree,)]
    reasons = []
    if spec_non_solvable:
        solvable: Optional[bool] = False
        reasons.append("non-solvable specialization group")
    elif all(max(c) <= 4 for c in bound):
        solvable = True
        if any(len(c) > 1 and int(sp.prod(c)) > 4 for c in bound):
            reasons.append("functional decomposition into components of degree <= 4")
        else:
            reasons.append("factors of degree <= 4")
    else:
        solvable = None

    if spec_not_2_group:
        real_radicals: Optional[bool] = False
    elif all(max(c) <= 2 for c in bound):
        real_radicals = True
    else:
        real_radicals = None

    exact = False
    if generic is not None and len(generic) == len(best["factors"]):
        # the specialization of each generic factor stays irreducible: pair
        # the factors by degree
        bounds = sorted((int(sp.prod(c)), _wreath_bound(c)) for c in generic)
        orders = sorted((f["degree"], f["order"]) for f in best["factors"])
        exact = bounds == orders
    if exact:
        reasons.append("specialization attains the upper bound")

    result.update(
        solvable=solvable,
        real_radicals=real_radicals,
        group_exact=exact,
        status="proven" if solvable is not None else "lower bound",
        proof="; ".join(reasons) or "specializations only",
    )
    return result


def _generic_factors(n, m, cp, texture, timeout) -> Optional[List[Tuple[int, ...]]]:
    generic = generic_factorization(n, m, cp, texture, timeout)
    return None if generic is None else generic["factors"]


# ---------------------------------------------------------------------------
# Texture scans
# ---------------------------------------------------------------------------


def _canonical_pattern(zeros: Iterable[Tuple[int, int]], n: int, m: int) -> tuple:
    """Smallest form of a zero pattern under row and column permutations."""
    zeros = set(zeros)
    best = None
    for rows in itertools.permutations(range(n)):
        for cols in itertools.permutations(range(m)):
            image = tuple(sorted((rows[i], cols[j]) for i, j in zeros))
            if best is None or image < best:
                best = image
    return best


def _has_decoupled_state(zeros: set, n: int, m: int) -> bool:
    zero_row = any(all((i, j) in zeros for j in range(m)) for i in range(n))
    zero_col = any(all((i, j) in zeros for i in range(n)) for j in range(m))
    return zero_row or zero_col


def texture_scan(
    n: int,
    m: int,
    cp: str = "real",
    max_zeros: Optional[int] = None,
    n_trials: int = 8,
    timeout: float = 60.0,
    seed: int = 0,
) -> List[Dict[str, object]]:
    r"""
    Galois groups of all texture-zero patterns of ``m_D`` (with ``M_R`` diagonal).

    Patterns are classified up to permutations of the active and of the
    sterile states (a sterile permutation only relabels the ``M_i``). Patterns
    with a vanishing row or column of ``m_D`` (a decoupled state) are skipped.

    Parameters:
    -----------
    n, m : int
        Number of active and sterile neutrinos
    cp : {'real', 'complex'}, optional
        Real or complex Dirac matrix
    max_zeros : int, optional
        Largest number of zeros (default: all patterns without decoupled states)
    n_trials : int, optional
        Specializations per pattern
    timeout : float, optional
        Time limit for each generic factorization
    seed : int, optional
        Seed of the random generator

    Returns:
    --------
    list of dict
        One :func:`generic_galois_group` record per pattern, with the extra
        keys ``n_zeros``, ``zeros`` and ``blocks`` (number of connected blocks
        of the mass matrix: a texture that splits the seesaw into independent
        sub-seesaws factorizes the polynomial)
    """
    positions = [(i, j) for i in range(n) for j in range(m)]
    max_zeros = n * m if max_zeros is None else max_zeros
    seen = set()
    records = []
    for count in range(max_zeros + 1):
        for zeros in itertools.combinations(positions, count):
            zero_set = set(zeros)
            if _has_decoupled_state(zero_set, n, m):
                continue
            canonical = _canonical_pattern(zero_set, n, m)
            if canonical in seen:
                continue
            seen.add(canonical)
            texture = Texture(dirac_zeros=canonical, name=f"{count} zeros")
            record = generic_galois_group(
                n, m, cp, texture, n_trials=n_trials, seed=seed, timeout=timeout
            )
            blocks = _connected_blocks(symbolic_mass_matrix(n, m, cp, texture))
            record.update(n_zeros=count, zeros=canonical, blocks=len(blocks))
            records.append(record)
    return records


def named_textures(n: int, m: int) -> List[Texture]:
    r"""
    Physically motivated special structures for an ``(n, m)`` seesaw.

    * ``diagonal m_D`` (``n = m``): decoupled ``(1,1)`` blocks.
    * ``degenerate M_R``: all ``M_i`` equal.
    * ``mu-tau symmetric m_D`` (``n = 3``): equal mu and tau rows.
    * ``FGY`` (``(n, m) = (3, 2)``): the Frampton-Glashow-Yanagida two-zero
      textures, zeros in different rows and columns.
    """
    textures = []
    if n == m and n > 1:
        zeros = tuple((i, j) for i in range(n) for j in range(m) if i != j)
        textures.append(Texture(dirac_zeros=zeros, name="diagonal m_D"))
    if m > 1:
        textures.append(
            Texture(majorana_degenerate=(tuple(range(m)),), name="degenerate M_R")
        )
    if n == 3:
        textures.append(Texture(dirac_equal_rows=((1, 2),), name="mu-tau symmetric m_D"))
    if (n, m) == (3, 2):
        textures.append(Texture(dirac_zeros=((0, 1), (2, 0)), name="FGY (e2, tau1)"))
        textures.append(Texture(dirac_zeros=((0, 0), (1, 1)), name="FGY (e1, mu2)"))
    return textures


def galois_table(
    cases: Sequence[Tuple[int, int]] = (
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ),
    cps: Sequence[str] = ("real", "complex"),
    n_trials: int = 12,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Generic Galois group of every ``(n, m)`` case, as a list of records."""
    return [
        generic_galois_group(n, m, cp, n_trials=n_trials, seed=seed)
        for cp in cps
        for n, m in cases
    ]
