"""Quick start: estimate the elastic treatment effects on a simulated study.

Install the package first (``pip install -e .`` from the repository root),
then run:

    python3 examples/quickstart.py

It simulates one observational study of Scenario 1 of the paper (250 units,
a confounder X that drives both the treatment and the response), estimates

* the populational elastic effect tau_pop (inverse-probability weighting),
* the local elastic effect tau_loc (matching),
* the pointwise FATE (doubly robust DR-FoS estimator), for comparison,

compares them with the causal effects of the simulation model (approximated
on simulated units), and saves a figure ``quickstart.png`` in the current
directory. Takes about half a minute.
"""

import logging

import matplotlib
import numpy as np

from shape_preserving_effects import (
    drop_rebound_trial,
    fate_drfos,
    l2_error,
    local_effect,
    main_dgp,
    population_estimands,
    populational_effect,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Simulate one study, estimate the three effects, log them, save the figure."""
    # 1. Data: n curves observed on a common grid of m time points.
    #    Y is an (n, m) array, A an (n,) array of 0/1, X an (n,) or (n, d) array.
    t = np.linspace(0, 1, 81)
    trial = drop_rebound_trial(n=250, t=t, seed=0)
    Y, A, X = trial.Y, trial.A, trial.X
    logger.info("%d units, %d treated, %d time points", len(A), A.sum(), len(t))

    # 2. The two elastic estimators of the paper.
    pop = populational_effect(t, Y, A, X, n_iter=2)
    loc = local_effect(t, Y, A, X, k=3, n_iter=2, max_pairs=300, rng=0)

    # 3. The pointwise (FATE) estimator, for comparison.
    fate_hat = fate_drfos(t, Y, A, X, rng=0)

    # 4. True causal effects of the simulation model (known in simulation only).
    truth = population_estimands(main_dgp, t, n_pop=400, seed=1, n_iter=2)
    tau_pop, fate = truth["tau_pop"], truth["fate"]

    for name, est in [
        ("populational (IPW)", pop.effect),
        ("local (matching)", loc.effect),
        ("pointwise DR-FoS", fate_hat),
    ]:
        logger.info(
            "%-20s depth %+.2f   L2 distance to tau_pop %.3f",
            name,
            est.min(),
            l2_error(est, tau_pop, t),
        )
    logger.info("%-20s depth %+.2f", "true tau_pop", tau_pop.min())
    logger.info("%-20s depth %+.2f", "true FATE", fate.min())

    # 5. Figure.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i in range(0, len(A), 5):
        ax[0].plot(t, Y[i], color="tab:green" if A[i] else "tab:orange", lw=0.5, alpha=0.5)
    ax[0].plot(t, pop.arm1.mean, color="tab:green", lw=2.5, label="treated arm (elastic mean)")
    ax[0].plot(t, pop.arm0.mean, color="tab:orange", lw=2.5, label="control arm (elastic mean)")
    ax[0].set(title="Observed curves and weighted elastic means", xlabel="time")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, tau_pop, "k--", lw=1.5, label=r"true $\tau_{pop}$")
    ax[1].plot(t, pop.effect, color="tab:blue", lw=2, label="populational estimate")
    ax[1].plot(t, loc.effect, color="tab:purple", lw=2, label="local estimate")
    ax[1].plot(t, fate_hat, color="tab:red", lw=2, label="FATE estimate (pointwise)")
    ax[1].set(title="Treatment-effect curves", xlabel="time")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("quickstart.png", dpi=120)
    logger.info("figure saved to quickstart.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
