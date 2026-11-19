r"""
The (3,2) minimal seesaw: exact versus leading-order results.

Inputs: light masses and PMNS matrix from ``experimental_values_2023()``
(``src/pmns_matrix.py``; update these values to the global fit quoted in the
paper), lightest neutrino massless, heavy masses ``D = (M, r M)`` and the
Casas-Ibarra angle ``z = x + i y``. All masses in GeV.

1. ``lo_accuracy_{no,io}.csv`` / ``lo_accuracy.pdf``: the leading-order
   Casas-Ibarra matrix (``m_D = i U^* d^{1/2} O^T D^{1/2}``, ``M_R = D``) is
   diagonalized exactly; relative deviations of the oscillation observables
   from their inputs over the plane ``(M, y)``, with the non-unitarity
   ``max |eta|`` and the total mixing ``U^2 = sum |Theta_{alpha i}|^2``.
   The exact parametrization reproduces the inputs to working precision.
2. ``heavy_splitting.csv`` / ``heavy_splitting.pdf``: quasi-degenerate heavy
   neutrinos ``D = (M, M(1 + delta))``; physical splitting of the
   leading-order model relative to the input splitting.
3. ``double_precision.csv`` / ``double_precision.pdf``: light masses obtained
   from the full ``5 x 5`` matrix with a double-precision Takagi factorization
   versus the high-precision result.
4. ``closed_form_benchmarks.{csv,tex}``: the exact closed-form quartic
   solution (``direct_diagonalization``, real and complex ``m_D``) evaluated
   at physical benchmark points, in double and in extended precision.

Run: ``.venv/bin/python paper_results/minimal_seesaw_32.py``
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402

from _common import EV, output_path, write_text  # noqa: E402
from src.exact_seesaw_numeric import (  # noqa: E402
    casas_ibarra_exact_numeric,
    casas_ibarra_leading_order,
    casas_ibarra_orthogonal,
    light_sector_observables,
    required_precision,
    takagi_factorization_mp,
)
from src.matrix_utils import takagi_factorization  # noqa: E402
from src.pmns_matrix import (  # noqa: E402
    experimental_values_2023,
    pmns_matrix_pdg,
    pmns_matrix_standard,
)
from src.seesaw import SymbolicSeesawTypeI  # noqa: E402
from src.symbolic_diagonalization import direct_diagonalization, lambdify_large  # noqa: E402

RE_Z = 0.3  # real part of the Casas-Ibarra angle
MASS_RATIO = 1.5  # M_2 / M_1 away from the degenerate case
OBSERVABLES = ["dm21", "dm3l", "s12", "s13", "s23"]


def inputs(ordering):
    """Light masses (GeV, ascending by label m_1, m_2, m_3) and PMNS matrix."""
    values = experimental_values_2023()[ordering]
    dm21 = values["dm21_sq_1e5"] * 1e-5
    dm3l = values["dm3l_sq_1e3"] * 1e-3
    if ordering == "normal":
        masses = [0.0, np.sqrt(dm21), np.sqrt(dm3l)]
    else:  # m_3 = 0, dm3l = m_3^2 - m_2^2 < 0
        m2 = np.sqrt(-dm3l)
        masses = [np.sqrt(m2**2 - dm21), m2, 0.0]
    U = pmns_matrix_pdg(*np.radians([values["theta12_deg"], values["theta13_deg"],
                                     values["theta23_deg"], values["delta_cp_deg"]]))
    reference = {
        "dm21": dm21, "dm3l": dm3l,
        "s12": np.sin(np.radians(values["theta12_deg"])) ** 2,
        "s13": np.sin(np.radians(values["theta13_deg"])) ** 2,
        "s23": np.sin(np.radians(values["theta23_deg"])) ** 2,
    }
    return [x * EV for x in masses], U, reference


def _deviations(observables, reference):
    out = {}
    for key in OBSERVABLES:
        value = observables[key] / EV**2 if key.startswith("dm") else observables[key]
        out[key] = value / reference[key] - 1
    return out


def lo_accuracy(ordering, log_masses=np.arange(0, 15.5, 1.0), ys=np.arange(0, 12.5, 1.0)):
    d, U, reference = inputs(ordering)
    lightest_massive = min(x for x in d if x > 0)
    rows = []
    for log_mass in log_masses:
        for y in ys:
            D = [10**log_mass, MASS_RATIO * 10**log_mass]
            with mpmath.workdps(required_precision(lightest_massive, D[1], y)):
                O = casas_ibarra_orthogonal(RE_Z + 1j * y, ordering)
                lo = light_sector_observables(
                    casas_ibarra_leading_order(d, D, O, U)["M"], ordering=ordering)
                exact = light_sector_observables(
                    casas_ibarra_exact_numeric(d, D, O, U)["M"], ordering=ordering)
            lo_dev, exact_dev = _deviations(lo, reference), _deviations(exact, reference)
            rows.append({
                "log10_M": log_mass, "y": y, "U2": exact["U2"], "eta_max": exact["eta_max"],
                **{f"lo_{k}": v for k, v in lo_dev.items()},
                "lo_max": max(abs(v) for v in lo_dev.values()),
                "exact_max": max(abs(v) for v in exact_dev.values()),
                "lo_heavy_rel": abs(lo["heavy_masses"][0] / D[0] - 1),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path(f"lo_accuracy_{ordering[:1]}o.csv"), index=False)
    return frame


def plot_lo_accuracy(frames):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for axis, (ordering, frame) in zip(axes[:2], frames.items()):
        table = frame.pivot(index="y", columns="log10_M", values="lo_max")
        eta = frame.pivot(index="y", columns="log10_M", values="eta_max")
        X, Y = np.meshgrid(table.columns, table.index)
        image = axis.contourf(X, Y, np.log10(table.values + 1e-300),
                              levels=np.arange(-30, 1, 2), cmap="viridis", extend="both")
        lines = axis.contour(X, Y, np.log10(eta.values + 1e-300), levels=[-9, -6, -3],
                             colors="white", linestyles=[":", "--", "-"])
        axis.clabel(lines, fmt={-3: r"$|\eta|=10^{-3}$", -6: r"$10^{-6}$", -9: r"$10^{-9}$"})
        axis.set_xlabel(r"$\log_{10}(M_1/\mathrm{GeV})$")
        axis.set_ylabel(r"$\mathrm{Im}\,z$")
        axis.set_title(f"{ordering} ordering")
        fig.colorbar(image, ax=axis, label=r"$\log_{10}$ max rel. deviation (LO)")
    axis = axes[2]
    for ordering, frame in frames.items():
        mask = frame.U2 < 0.1
        axis.loglog(frame.U2[mask], frame.lo_max[mask], ".", label=f"{ordering}, LO")
        axis.loglog(frame.U2[mask], np.maximum(frame.exact_max[mask], 1e-30), "x",
                    label=f"{ordering}, exact", alpha=0.5)
    grid = np.logspace(-25, -1, 10)
    axis.loglog(grid, grid, "k--", lw=1, label=r"$U^2$")
    axis.set_xlabel(r"$U^2=\sum_{\alpha i}|\Theta_{\alpha i}|^2$")
    axis.set_ylabel("max relative deviation of oscillation observables")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path("lo_accuracy.pdf"))
    fig.savefig(output_path("lo_accuracy.png"), dpi=150)
    plt.close(fig)


def heavy_splitting(ordering="normal"):
    d, U, _ = inputs(ordering)
    lightest_massive = min(x for x in d if x > 0)
    rows = []
    for mass in (1.0, 1e3, 1e6):
        for y in (0.0, 5.0):
            for log_delta in np.arange(-15, -1.5, 1.0):
                delta = 10**log_delta
                dps = required_precision(lightest_massive, 2 * mass, y) + int(-log_delta) + 5
                with mpmath.workdps(dps):
                    # M_2 = M_1 (1 + delta) exactly: in double precision the
                    # rounding of 1 + delta would shift the splitting by ~1e-16 M
                    D = [mpmath.mpf(mass), mpmath.mpf(mass) * (1 + mpmath.mpf(10) ** int(log_delta))]
                    O = casas_ibarra_orthogonal(RE_Z + 1j * y, ordering)
                    lo = light_sector_observables(
                        casas_ibarra_leading_order(d, D, O, U)["M"], ordering=ordering)
                    splitting = mpmath.mpf(mass) * mpmath.mpf(10) ** int(log_delta)
                    shift = lo["heavy_splitting_mp"] - splitting
                rows.append({"M": mass, "y": y, "delta": delta,
                             "splitting_ratio": float(lo["heavy_splitting_mp"] / splitting),
                             "shift_GeV": float(shift)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path("heavy_splitting.csv"), index=False)
    fig, axis = plt.subplots(figsize=(6, 4.3))
    for (mass, y), group in frame.groupby(["M", "y"]):
        axis.loglog(group.delta, np.abs(group.splitting_ratio - 1), "o-", ms=3,
                    label=fr"$M_1=10^{{{np.log10(mass):.0f}}}$ GeV, Im$z$={y:.0f}")
    axis.set_xlabel(r"input splitting $\delta=(M_2-M_1)/M_1$")
    axis.set_ylabel(r"$|\Delta M_\mathrm{phys}/\Delta M_\mathrm{input}-1|$ (LO)")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path("heavy_splitting.pdf"))
    fig.savefig(output_path("heavy_splitting.png"), dpi=150)
    plt.close(fig)
    return frame


def double_precision(ordering="normal"):
    d, U, _ = inputs(ordering)
    lightest_massive = min(x for x in d if x > 0)
    rows = []
    for y in (0.0, 4.0, 8.0):
        for log_mass in np.arange(0, 15.5, 0.5):
            D = [10**log_mass, MASS_RATIO * 10**log_mass]
            with mpmath.workdps(required_precision(lightest_massive, D[1], y)):
                O = casas_ibarra_orthogonal(RE_Z + 1j * y, ordering)
                model = casas_ibarra_exact_numeric(d, D, O, U)
                reference, _ = takagi_factorization_mp(model["M"])
                matrix = np.array(model["M"].tolist(), dtype=complex)
            masses, _ = takagi_factorization(matrix)
            light = np.sort(masses[:3])
            exact = np.sort([float(x) for x in reference[:3]])
            rows.append({"log10_M": log_mass, "y": y,
                         "rel_error_m2": abs(light[1] / exact[1] - 1),
                         "rel_error_m3": abs(light[2] / exact[2] - 1)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path("double_precision.csv"), index=False)
    fig, axis = plt.subplots(figsize=(6, 4.3))
    for y, group in frame.groupby("y"):
        axis.semilogy(group.log10_M, np.maximum(group.rel_error_m2, 1e-17), "o-", ms=3,
                      label=fr"$m_2$, Im$z$={y:.0f}")
    grid = np.arange(0, 15.5, 0.5)
    axis.semilogy(grid, 2.2e-16 * 10**grid * MASS_RATIO / (d[1] if d[1] > 0 else d[0]),
                  "k--", lw=1, label=r"$\epsilon_\mathrm{mach}\,M_2/m_2$")
    axis.set_ylim(1e-17, 1e3)
    axis.set_xlabel(r"$\log_{10}(M_1/\mathrm{GeV})$")
    axis.set_ylabel("relative error, double-precision Takagi")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path("double_precision.pdf"))
    fig.savefig(output_path("double_precision.png"), dpi=150)
    plt.close(fig)
    return frame


# ---------------------------------------------------------------------------
# Closed-form benchmarks
# ---------------------------------------------------------------------------


def _diagonal_sterile_basis(model, real):
    """Rotate the exact parametrization to diagonal M_R (and real or polar m_D)."""
    m_D, M_R = model["m_D"], model["M_R"]
    if real:
        # CP-conserving inputs: m_D = i * real; remove the common phase
        m_D = m_D * mpmath.mpc(0, -1)
        m_D = m_D.apply(mpmath.re)
        M_R = M_R.apply(mpmath.re)
        values, Q = mpmath.eigsy(M_R)
        return m_D * Q, list(values)
    values, W = takagi_factorization_mp(M_R)
    m_D = m_D * W
    # Polar form: remove the phases of the first column by rephasing the actives
    phases = [mpmath.exp(-1j * mpmath.arg(m_D[i, 0])) for i in range(m_D.rows)]
    for i in range(m_D.rows):
        for j in range(m_D.cols):
            m_D[i, j] *= phases[i]
    return m_D, list(values)


def closed_form_benchmarks():
    benchmarks = [  # (label, ordering, M_1 [GeV], Im z, CP)
        ("BP1", "normal", 1e3, 0.0, "real"),
        ("BP2", "normal", 1e6, 0.0, "real"),
        ("BP3", "normal", 10.0, 4.0, "complex"),
        ("BP4", "inverted", 1e3, 3.0, "complex"),
        ("BP5", "normal", 1e10, 1.0, "complex"),
    ]
    closed = {}
    for cp in ("real", "complex"):
        model = SymbolicSeesawTypeI(3, 2, dirac="real" if cp == "real" else "polar",
                                    majorana_basis="diagonal")
        result = direct_diagonalization(model)
        if cp == "real":
            params = list(model.m_D_sym) + [model.M_R_sym[0, 0], model.M_R_sym[1, 1]]
        else:
            params = (list(model.dirac_moduli) + [model.dirac_phases[i, 1] for i in range(3)]
                      + [model.M_R_sym[0, 0], model.M_R_sym[1, 1]])
        masses = sp.Matrix(result["masses"])
        closed[cp] = (params, {mode: lambdify_large(params, masses, mode)
                               for mode in ("numpy", "mpmath")})

    rows = []
    for label, ordering, mass, y, cp in benchmarks:
        d, U, _ = inputs(ordering)
        if cp == "real":
            # CP-conserving inputs (delta = 0, no Majorana phases) and a real
            # Casas-Ibarra angle give a real m_D
            values = experimental_values_2023()[ordering]
            U = pmns_matrix_standard(*np.radians([values["theta12_deg"],
                                                  values["theta13_deg"],
                                                  values["theta23_deg"]]), 0.0)
            if y != 0:
                raise ValueError("a real m_D needs a real Casas-Ibarra angle")
        z = RE_Z + 1j * y
        D = [mass, MASS_RATIO * mass]
        lightest = min(x for x in d if x > 0)
        # The quartic formula for the squared masses cancels ~(M/m)^4: twice
        # the digits a direct diagonalization needs
        dps = 2 * required_precision(lightest, D[1], abs(np.imag(z)))
        with mpmath.workdps(dps):
            O = casas_ibarra_orthogonal(z, ordering)
            model = casas_ibarra_exact_numeric(d, D, O, U)
            m_D, heavy = _diagonal_sterile_basis(model, cp == "real")
            if cp == "real":
                arguments = [m_D[i, j] for i in range(3) for j in range(2)] + heavy
            else:
                arguments = ([abs(m_D[i, j]) for i in range(3) for j in range(2)]
                             + [mpmath.arg(m_D[i, 1]) for i in range(3)] + heavy)
            arguments_mp = [mpmath.mpmathify(a) for a in arguments]
            exact_masses = sorted([x for x in d if x > 0] + D)
            start = time.perf_counter()
            values_mp = closed[cp][1]["mpmath"](*arguments_mp)
            seconds_mp = time.perf_counter() - start
            masses_mp = sorted(abs(mpmath.re(values_mp[i])) for i in range(4))
            error_mp = max(abs(a / b - 1) for a, b in zip(masses_mp, exact_masses))
        with np.errstate(all="ignore"):
            values_np = closed[cp][1]["numpy"](*[complex(a) for a in arguments])
        masses_np = np.sort(np.abs(np.real(np.array(values_np, dtype=complex).flatten())))
        error_np = float(np.max(np.abs(masses_np / np.array(exact_masses, dtype=float) - 1)))
        rows.append({"benchmark": label, "ordering": ordering, "M1_GeV": mass,
                     "Im_z": float(np.imag(z)), "cp": cp, "digits": dps,
                     "m2_eV": float(exact_masses[0]) / EV, "m3_eV": float(exact_masses[1]) / EV,
                     "rel_error_extended": float(error_mp), "rel_error_double": error_np,
                     "seconds_extended": seconds_mp})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path("closed_form_benchmarks.csv"), index=False)
    tex_rows = [
        f"{r.benchmark} & {r.ordering[:1].upper()}O & $10^{{{np.log10(r.M1_GeV):.0f}}}$ & "
        f"{r.Im_z:.0f} & {r.cp} & {r.digits} & {r.rel_error_extended:.1e} & "
        f"{r.rel_error_double:.1e} \\\\"
        for r in frame.itertuples()
    ]
    write_text("closed_form_benchmarks.tex", "\n".join([
        r"\begin{tabular}{lccccccc}", r"\hline",
        r"point & ordering & $M_1$ [GeV] & $\mathrm{Im}\,z$ & $m_D$ & digits & error (extended) & error (double) \\",
        r"\hline", *tex_rows, r"\hline", r"\end{tabular}",
    ]) + "\n")
    return frame


if __name__ == "__main__":
    start = time.perf_counter()
    frames = {ordering: lo_accuracy(ordering) for ordering in ("normal", "inverted")}
    plot_lo_accuracy(frames)
    for ordering, frame in frames.items():
        # below U^2 ~ 1e-12 the deviation reaches the 1e-16 floor of the double-precision inputs
        ratio = (frame.lo_max / frame.U2)[frame.U2.between(1e-12, 1e-2)]
        print(f"{ordering}: exact max deviation {frame.exact_max.max():.1e}; "
              f"LO deviation / U^2 in [{ratio.min():.2f}, {ratio.max():.2f}]", flush=True)
    print(f"LO accuracy: {time.perf_counter() - start:.0f} s", flush=True)
    print(heavy_splitting().groupby(["M", "y"]).splitting_ratio.agg(["min", "max"]), flush=True)
    print(double_precision().pivot(index="log10_M", columns="y", values="rel_error_m2")
          .iloc[::4], flush=True)
    closed_form_benchmarks()
    print(f"total: {time.perf_counter() - start:.0f} s")
