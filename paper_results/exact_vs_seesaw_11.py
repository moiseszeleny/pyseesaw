r"""
The (1,1) seesaw: exact Takagi diagonalization versus the seesaw formula.

With ``epsilon = m_D / M`` the exact light mass and mixing angle are
``m_1 = M (sqrt(1 + 4 eps^2) - 1) / 2`` and ``tan 2 theta = 2 eps``; the seesaw
formula gives ``m_1 = M eps^2`` and ``theta = eps``. Produces
``exact_vs_seesaw_11.{csv,pdf,png}`` with the relative errors of the leading
and next-to-leading order approximations, checked against the symbolic
closed form of :func:`direct_diagonalization`.

Run: ``.venv/bin/python paper_results/exact_vs_seesaw_11.py``
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402

from _common import output_path  # noqa: E402
from src.seesaw import SymbolicSeesawTypeI  # noqa: E402
from src.symbolic_diagonalization import direct_diagonalization, lambdify_large  # noqa: E402


def main():
    model = SymbolicSeesawTypeI(1, 1, dirac="real", majorana_basis="diagonal")
    result = direct_diagonalization(model)
    a, M = model.m_D_sym[0, 0], model.M_R_sym[0, 0]
    closed_form = lambdify_large([a, M], sp.Matrix([result["masses"][0], sp.Abs(result["U"][1, 0])]),
                                 "mpmath")
    rows = []
    with mpmath.workdps(50):
        for eps in np.logspace(-6, 1, 71):
            e = mpmath.mpf(eps)
            exact_mass = (mpmath.sqrt(1 + 4 * e**2) - 1) / 2  # in units of M
            exact_theta = mpmath.atan(2 * e) / 2
            values = closed_form(e, mpmath.mpf(1))
            rows.append({
                "epsilon": eps,
                "m1_over_M": float(exact_mass),
                "closed_form_check": float(abs(values[0] / exact_mass - 1)),
                "angle_check": float(abs(values[1] / mpmath.sin(exact_theta) - 1)),
                "lo_mass_error": float(abs(e**2 / exact_mass - 1)),
                "nlo_mass_error": float(abs((e**2 - e**4) / exact_mass - 1)),
                "lo_angle_error": float(abs(e / exact_theta - 1)),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path("exact_vs_seesaw_11.csv"), index=False)
    # 50 digits minus the cancellation in sqrt(1 + 4 eps^2) - 1 at eps = 1e-6
    assert frame.closed_form_check.max() < 1e-30 and frame.angle_check.max() < 1e-30

    fig, axis = plt.subplots(figsize=(6, 4.3))
    axis.loglog(frame.epsilon, frame.lo_mass_error, label=r"$m_1\simeq m_D^2/M$")
    axis.loglog(frame.epsilon, frame.nlo_mass_error, label=r"$m_1\simeq m_D^2/M-m_D^4/M^3$")
    axis.loglog(frame.epsilon, frame.lo_angle_error, "--", label=r"$\theta\simeq m_D/M$")
    axis.set_xlabel(r"$\epsilon=m_D/M$")
    axis.set_ylabel("relative error of the approximation")
    axis.set_ylim(1e-25, 10)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path("exact_vs_seesaw_11.pdf"))
    fig.savefig(output_path("exact_vs_seesaw_11.png"), dpi=150)
    plt.close(fig)
    print(frame.iloc[::10].to_string())


if __name__ == "__main__":
    main()
