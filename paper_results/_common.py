"""Shared helpers for the paper result scripts (repo root on sys.path, output dir)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT, exist_ok=True)

#: eV -> GeV
EV = 1e-9


def output_path(name: str) -> str:
    """Path of a result file in ``paper_results/output``."""
    return os.path.join(OUTPUT, name)


def write_text(name: str, text: str) -> str:
    """Write a text result and return its path."""
    path = output_path(name)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def bool_mark(value) -> str:
    """LaTeX mark for a proven yes/no/undecided flag."""
    return {True: r"\checkmark", False: r"$\times$", None: "?"}[value]
