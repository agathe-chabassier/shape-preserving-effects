"""Estimate the elastic treatment effects on your own data.

Install the package first (``pip install -e .`` from the repository root),
then run:

    python3 examples/own_data.py                          # the example file
    python3 examples/own_data.py path/to/your_data.csv    # your file

Expected input: a csv file in "long" format, one row per measurement, with
the columns

    id        identifier of the unit (patient, user, ...)
    day       time of the measurement (any unit; here days since treatment start)
    outcome   measured value
    treated   1 if the unit is treated, 0 otherwise (constant within a unit)
    x         a confounder (constant within a unit); several confounders can
              be given as several columns, listed in COVARIATES below

``examples/example_data.csv`` is a simulated example of this format: 200
units observed on 40 irregular days over one year.

The script interpolates every curve on a common uniform grid, estimates the
populational and local elastic effects, and saves the curves and a figure in
the current directory (``own_data_effects.csv``, ``own_data_effects.png``).

Before using real data:
* every unit must be observed over the whole window [START, END] (the curves
  are interpolated, never extrapolated);
* noisy curves should be smoothed first: the elastic mean registers the
  curves, and registration reads noise as shape (see the section
  "When noise dominates the signal: the template bias" of the paper).
"""

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

from shape_preserving_effects import (
    LocalEffect,
    PopulationalEffect,
    local_effect,
    populational_effect,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

EXAMPLE_CSV = Path(__file__).resolve().parent / "example_data.csv"
TIME, OUTCOME, TREATMENT, ID = "day", "outcome", "treated", "id"
COVARIATES = ["x"]
# Common grid: GRID_SIZE equally spaced times in [START, END].
START, END, GRID_SIZE = 0, 360, 81


@dataclass
class Unit:
    """The measurements of one unit, and its unit-level treatment and covariates."""

    treated: int
    covariates: list[float]
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


def _cell(row: dict[str, str], column: str, line_no: int) -> str:
    """Non-empty text of one cell; raises a ValueError naming the line and column."""
    text = row[column].strip()
    if not text:
        raise ValueError(f"line {line_no}: empty value in column {column!r}")
    return text


def read_long_csv(path: Path) -> dict[str, Unit]:
    """Group the rows of a long-format csv by unit.

    Args:
        path: The csv file.

    Returns:
        The units, keyed by their identifier.

    Raises:
        ValueError: If a value is empty, or if a unit's treatment or
            covariates differ from those of its first row.
    """
    units: dict[str, Unit] = {}
    with path.open(newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            uid = row[ID]
            treated = int(_cell(row, TREATMENT, line_no))
            covariates = [float(_cell(row, c, line_no)) for c in COVARIATES]
            unit = units.setdefault(uid, Unit(treated, covariates))
            if treated != unit.treated or covariates != unit.covariates:
                raise ValueError(
                    f"line {line_no}: unit {uid} has {TREATMENT}={treated}, "
                    f"covariates {covariates}, but its first row has "
                    f"{TREATMENT}={unit.treated}, covariates {unit.covariates}"
                )
            unit.times.append(float(_cell(row, TIME, line_no)))
            unit.values.append(float(_cell(row, OUTCOME, line_no)))
    return units


def to_common_grid(
    units: dict[str, Unit], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Curves on ``grid`` by linear interpolation, with A, X and the unit ids.

    Units not observed over the whole grid are skipped, with a warning.

    Args:
        units: The units read by :func:`read_long_csv`.
        grid: (m,) common time grid.

    Returns:
        ``(Y, A, X, ids)``: the (n, m) curves, the (n,) treatment indicator,
        the (n, d) covariates and the identifiers of the n units kept.
    """
    curves, treatments, covariates, kept_ids = [], [], [], []
    for uid, unit in units.items():
        order = np.argsort(unit.times)
        times, values = np.asarray(unit.times)[order], np.asarray(unit.values)[order]
        if times[0] > grid[0] or times[-1] < grid[-1]:
            logger.warning("unit %s skipped: not observed over the whole window", uid)
            continue
        curves.append(np.interp(grid, times, values))
        treatments.append(unit.treated)
        covariates.append(unit.covariates)
        kept_ids.append(uid)
    return np.array(curves), np.array(treatments), np.array(covariates), kept_ids


def plot_effects(
    grid: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    pop: PopulationalEffect,
    loc: LocalEffect,
    figure_png: Path,
) -> None:
    """Draw the curves, the arm-level elastic means and the two effects."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for curve, treated in zip(Y, A):
        ax[0].plot(grid, curve, color="tab:green" if treated else "tab:orange", lw=0.4, alpha=0.3)
    ax[0].plot(
        grid, pop.arm1.mean, color="tab:green", lw=2.5, label="treated (weighted elastic mean)"
    )
    ax[0].plot(
        grid, pop.arm0.mean, color="tab:orange", lw=2.5, label="control (weighted elastic mean)"
    )
    ax[0].set(title="Curves and arm-level elastic means", xlabel=TIME, ylabel=OUTCOME)
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="0.8", lw=0.8)
    ax[1].plot(
        grid, pop.effect, color="tab:blue", lw=2, label=r"populational effect $\hat\tau_{pop}$"
    )
    ax[1].plot(grid, loc.effect, color="tab:purple", lw=2, label=r"local effect $\hat\tau_{loc}$")
    ax[1].set(title="Estimated treatment effects", xlabel=TIME)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_png, dpi=120)
    logger.info("figure saved to %s", figure_png)


def main(path: Path) -> None:
    """Read the csv, estimate the two elastic effects, save the curves and the figure."""
    grid = np.linspace(START, END, GRID_SIZE)
    Y, A, X, ids = to_common_grid(read_long_csv(path), grid)
    logger.info("%d units (%d treated) on a grid of %d time points", len(ids), A.sum(), GRID_SIZE)

    pop = populational_effect(grid, Y, A, X, n_iter=2)
    loc = local_effect(grid, Y, A, X, k=3, n_iter=2, max_pairs=300, rng=0)

    effects_csv = Path("own_data_effects.csv")
    with effects_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([TIME, "tau_pop", "tau_loc", "mu_treated", "mu_control"])
        for grid_point in zip(grid, pop.effect, loc.effect, pop.arm1.mean, pop.arm0.mean):
            writer.writerow([f"{v:.6g}" for v in grid_point])
    logger.info("effect curves saved to %s", effects_csv)
    plot_effects(grid, Y, A, pop, loc, Path("own_data_effects.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLE_CSV)
