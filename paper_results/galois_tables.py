r"""
Galois groups of the reduced Type I seesaw polynomial.

Produces
* ``galois_generic.{csv,tex}``: generic group for every (n, m) <= (3, 3),
  real and complex m_D;
* ``galois_named_textures.{csv,tex}``: degenerate M_R, diagonal and mu-tau
  symmetric m_D, FGY two-zero textures;
* ``galois_texture_zeros_<n><m>_<cp>.csv`` and ``galois_texture_zeros.tex``:
  all texture-zero patterns of m_D (M_R diagonal) for the cases whose generic
  group is not solvable or is the flagship (3, 2).

Run: ``.venv/bin/python paper_results/galois_tables.py``
"""

import time

import pandas as pd

from _common import bool_mark, output_path, write_text
from src.galois_analysis import galois_table, generic_galois_group, named_textures, texture_scan

COLUMNS = [
    "n", "m", "cp", "texture", "pattern", "degree", "massless", "repeated",
    "groups", "group_order", "decompositions", "solvable", "real_radicals",
    "group_exact", "status", "proof",
]


def _frame(records):
    frame = pd.DataFrame(records)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame["decompositions"] = frame["decompositions"].apply(
        lambda d: None if d is None else ", ".join(d)
    )
    return frame


def _latex_group(name: str) -> str:
    out = []
    for part in name.split(" x "):
        part = part.replace("S4xC2", r"S_4\times C_2")
        if part[:1] in "SACD" and part[1:].isdigit():
            part = f"{part[0]}_{{{part[1:]}}}"
        elif part == "V":
            part = "V_4"
        out.append(part)
    return "$" + r" \times ".join(out) + "$"


def generic_table():
    frame = _frame(galois_table())
    frame[COLUMNS].to_csv(output_path("galois_generic.csv"), index=False)
    rows = []
    for (n, m), group in frame.groupby(["n", "m"], sort=False):
        real = group[group.cp == "real"].iloc[0]
        cplx = group[group.cp == "complex"].iloc[0]
        rows.append(
            f"$({n},{m})$ & {real.massless} & {real.degree} & {_latex_group(real.groups)} & "
            f"{_latex_group(cplx.groups)} & {bool_mark(real.solvable)} & "
            f"{bool_mark(real.real_radicals)} \\\\"
        )
    tex = "\n".join([
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        r"$(n,m)$ & massless & $k$ & $\mathrm{Gal}$ (real $m_D$) & $\mathrm{Gal}$ (complex $m_D$) & radicals & real radicals \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ])
    write_text("galois_generic.tex", tex + "\n")
    return frame


def named_table():
    records = []
    for cp in ("real", "complex"):
        for n, m in [(2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]:
            for texture in named_textures(n, m):
                records.append(generic_galois_group(n, m, cp, texture))
    frame = _frame(records)
    frame[COLUMNS].to_csv(output_path("galois_named_textures.csv"), index=False)
    rows = [
        f"$({r.n},{r.m})$ & {r.texture} & {r.cp} & {r.degree} & {_latex_group(r.groups)} & "
        f"{r.decompositions} & {bool_mark(r.solvable)} & {bool_mark(r.real_radicals)} \\\\"
        for r in frame.itertuples()
    ]
    tex = "\n".join([
        r"\begin{tabular}{llcclccc}",
        r"\hline",
        r"$(n,m)$ & structure & $m_D$ & distinct roots & $\mathrm{Gal}$ & decomposition & radicals & real radicals \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ])
    write_text("galois_named_textures.tex", tex + "\n")
    return frame


def zeros_tables():
    frames = []
    for n, m in [(3, 2), (2, 3), (3, 3)]:
        for cp in ("real", "complex"):
            start = time.perf_counter()
            frame = _frame(texture_scan(n, m, cp, timeout=120))
            frame.to_csv(output_path(f"galois_texture_zeros_{n}{m}_{cp}.csv"), index=False)
            print(f"  texture scan ({n},{m}) {cp}: {len(frame)} patterns, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    # Only the patterns whose group differs from the generic one (zero zeros)
    generic = frame[frame.n_zeros == 0].set_index(["n", "m", "cp"])["groups"]
    differs = [g != generic[(n, m, cp)] for n, m, cp, g in
               zip(frame.n, frame.m, frame.cp, frame.groups)]
    interesting = frame[differs]
    rows = [
        f"$({r.n},{r.m})$ & {r.cp} & \\texttt{{{r.pattern}}} & {r.blocks} & {r.massless} & {r.degree} & "
        f"{_latex_group(r.groups)} & {bool_mark(r.solvable)} & {bool_mark(r.real_radicals)} \\\\"
        for r in interesting.itertuples()
    ]
    tex = "\n".join([
        r"\begin{tabular}{cclccclcc}",
        r"\hline",
        r"$(n,m)$ & $m_D$ & texture & blocks & massless & $k$ & $\mathrm{Gal}$ & radicals & real radicals \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ])
    write_text("galois_texture_zeros.tex", tex + "\n")
    return frame


if __name__ == "__main__":
    for name, job in [("generic", generic_table), ("named textures", named_table),
                      ("texture zeros", zeros_tables)]:
        start = time.perf_counter()
        frame = job()
        print(f"{name}: {len(frame)} rows, {time.perf_counter() - start:.0f} s", flush=True)
        if name != "texture zeros":
            print(frame[["n", "m", "cp", "texture", "groups", "decompositions",
                         "solvable", "real_radicals", "group_exact"]].to_string(), flush=True)
