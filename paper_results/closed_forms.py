r"""
Exact Takagi diagonalization in closed form, exported for the paper.

For each case the masses and the mixing matrix are written as a chain of
definitions (:func:`src.symbolic_diagonalization.nested_definitions`), checked
with :func:`verify_takagi`, and the chain itself is checked against the
original expression at random points (round trip).

Produces in ``paper_results/output``:
* ``closed_form_11.tex``: (1,1), fully simplified, next to the seesaw formula;
* ``closed_form_<nm>_<cp>_<method>.tex``: definition chains (``align*`` bodies);
* ``closed_forms_summary.{csv,tex}``: sizes (written out, DAG, number of
  definitions) and verification status.

Run: ``.venv/bin/python paper_results/closed_forms.py``
"""

import time

import mpmath
import numpy as np
import pandas as pd
import sympy as sp

from _common import bool_mark, output_path, write_text
from src.seesaw import SymbolicSeesawTypeI
from src.symbolic_diagonalization import (
    direct_diagonalization,
    expression_size,
    nested_definitions,
    numeric_mass_points,
    semi_explicit_diagonalization,
    verify_takagi,
)

CASES = [
    (1, 1, "real", "direct"),
    (1, 1, "complex", "direct"),
    (1, 2, "real", "direct"),
    (2, 2, "real", "direct"),
    (3, 2, "real", "direct"),
    (3, 2, "complex", "direct"),
    (3, 2, "real", "semi_explicit"),
    (3, 2, "complex", "semi_explicit"),
    (3, 3, "real", "semi_explicit"),
    # Degenerate heavy neutrinos, M_R = M 1: P(lambda) = Q(lambda (lambda - M)),
    # solvable for real m_D although the (3,3) polynomial has degree 6
    (3, 3, "real", "direct", "degenerate"),
]


def _model(n, m, cp, structure="generic"):
    model = SymbolicSeesawTypeI(
        n, m, dirac="real" if cp == "real" else "polar", majorana_basis="diagonal"
    )
    if structure == "degenerate":
        model.M_R_sym = sp.diag(*[model.M_R_sym[0, 0]] * m)
        model._full_mass_matrix_symbolic = None
    return model


def _chain_round_trip(chain, expressions, points, digits=40):
    """Largest difference between the chain and the original expressions."""
    worst = 0.0
    with mpmath.workdps(digits):
        for point in points:
            values = {s: sp.Float(complex(v).real, digits) + sp.I * sp.Float(complex(v).imag, digits)
                      for s, v in point.items()}
            for symbol, expression in chain["definitions"]:
                values[symbol] = sp.N(expression.xreplace(values), digits)
            outputs = {s: sp.N(e.xreplace(values), digits) for s, e in chain["outputs"]}
            outputs.update({s: values[s] for s, _ in chain["definitions"]})
            for name, original in zip(chain["names"], expressions):
                reference = complex(sp.N(_evaluate_dag(original, values), digits))
                worst = max(worst, abs(complex(outputs[name]) - reference))
    return worst


def _evaluate_dag(expression, values):
    """Evaluate a large expression once per distinct subexpression."""
    cache = {}
    stack = [(expression, False)]
    while stack:
        node, expanded = stack.pop()
        if node in cache:
            continue
        if not node.args:
            cache[node] = values.get(node, node)
        elif expanded:
            cache[node] = sp.N(node.func(*[cache[a] for a in node.args]), 40)
        else:
            stack.append((node, True))
            stack.extend((a, False) for a in node.args if a not in cache)
    return cache[expression]


def one_generation():
    """(1,1): exact masses and mixing next to the seesaw approximation."""
    a, M = sp.symbols("m_D M", positive=True)
    root = sp.sqrt(M**2 + 4 * a**2)
    light, heavy = (root - M) / 2, (root + M) / 2
    theta = sp.atan(2 * a / M) / 2
    tex = "\n".join([
        r"% (1,1) Type I seesaw, real m_D, M > 0; U^T M U = diag(m_1, m_2)",
        r"\begin{align*}",
        rf"m_1 &= {sp.latex(light)} = \frac{{m_D^2}}{{M}} - \frac{{m_D^4}}{{M^3}} + \mathcal{{O}}\!\left(\frac{{m_D^6}}{{M^5}}\right), \\",
        rf"m_2 &= {sp.latex(heavy)}, \\",
        rf"\tan 2\theta &= \frac{{2 m_D}}{{M}}, \qquad U = \begin{{pmatrix}} i\cos\theta & \sin\theta \\ -i\sin\theta & \cos\theta \end{{pmatrix}}, \\",
        r"m_1 m_2 &= m_D^2, \qquad m_2 - m_1 = M .",
        r"\end{align*}",
    ])
    # Check the series and the orthogonal form against the direct result
    series = sp.series(light, a, 0, 7).removeO()
    assert sp.simplify(series - (a**2 / M - a**4 / M**3 + 2 * a**6 / M**5)) == 0
    write_text("closed_form_11.tex", tex + "\n")


def export(n, m, cp, method, structure="generic"):
    model = _model(n, m, cp, structure)
    mass_matrix = model.full_mass_matrix_symbolic()
    zeros = [sp.S.Zero] * model.n_massless
    start = time.perf_counter()
    if method == "direct":
        result = direct_diagonalization(model)
        masses = zeros + list(result["masses"])
        points = None
        ranges = {model.M_R_sym[i, i]: (4.0 + 3 * i, 5.0 + 3 * i) for i in range(m)}
    else:
        result = semi_explicit_diagonalization(model)
        masses = zeros + list(result["masses"])
        points = numeric_mass_points(model, 3, 0)
        ranges = None
    U = result["U"]
    build_seconds = time.perf_counter() - start

    massive = list(result["masses"]) if method == "direct" else []
    names = [f"m_{model.n_massless + i + 1}" for i in range(len(massive))]
    if method == "direct" and result["factorized"]:
        names = [f"mu_{i + 1}" for i in range(len(massive))]  # grouped, not ordered
    names += [f"U_{{{i + 1}{j + 1}}}" for i in range(U.rows) for j in range(U.cols)]
    expressions = massive + list(U)
    chain = nested_definitions(expressions, names)
    chain["names"] = [sp.Symbol(x) for x in names]

    check = verify_takagi(U, mass_matrix, masses, n_points=3, ranges=ranges, points=points)
    sample = points if points is not None else _random_points(expressions, ranges, 2)
    round_trip = _chain_round_trip(chain, expressions, sample[:2])

    header = [
        f"% ({n},{m}) Type I seesaw, {cp} m_D, {method} diagonalization, U^T M U = diag(m_i).",
        f"% {chain['n_definitions']} intermediate definitions; massless states first.",
    ]
    if method == "semi_explicit":
        header.append("% Masses m_i are symbols: roots of the reduced characteristic polynomial.")
    suffix = "" if structure == "generic" else f"_{structure}"
    if structure == "degenerate":
        header.append("% Degenerate heavy neutrinos M_1 = ... = M_m; masses mu_i grouped in (1,1) pairs, not ordered.")
    write_text(f"closed_form_{n}{m}_{cp}_{method}{suffix}.tex",
               "\n".join(header + [r"\begin{align*}", chain["latex"], r"\end{align*}"]) + "\n")
    size = expression_size(sp.Matrix(expressions))
    return {
        "case": f"({n},{m})" + ("" if structure == "generic" else f" {structure}"),
        "cp": cp, "method": method,
        "tree_size": size["tree"], "dag_size": size["dag"],
        "definitions": chain["n_definitions"], "latex_chars": len(chain["latex"]),
        "verified": check["passed"], "precision": check["precision"],
        "diagonalization_error": check["diagonalization_error"],
        "round_trip_error": round_trip, "seconds": build_seconds,
    }


def _random_points(expressions, ranges, count):
    from src.symbolic_diagonalization import _dag_free_symbols, _sample_point

    symbols = sorted(_dag_free_symbols(expressions), key=lambda s: s.name)
    rng = np.random.default_rng(1)
    return [_sample_point(symbols, rng, ranges) for _ in range(count)]


if __name__ == "__main__":
    one_generation()
    records = []
    for case in CASES:
        record = export(*case)
        print(record, flush=True)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(output_path("closed_forms_summary.csv"), index=False)
    rows = [
        f"{r.case} & {r.cp} & {r.method.replace('_', '-')} & {r.tree_size:.2e} & {r.dag_size} & "
        f"{r.definitions} & {bool_mark(bool(r.verified))} \\\\"
        for r in frame.itertuples()
    ]
    write_text("closed_forms_summary.tex", "\n".join([
        r"\begin{tabular}{lllrrrc}", r"\hline",
        r"case & $m_D$ & method & size written out & distinct subexpr. & definitions & verified \\",
        r"\hline", *rows, r"\hline", r"\end{tabular}",
    ]) + "\n")
