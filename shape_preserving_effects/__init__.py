"""Shape-preserving treatment effects for functional outcomes.

Estimators of the elastic causal effects of

    A. Chabassier, J. Olivier, J. Josse and E. Scornet,
    "Shape-Preserving Treatment Effects for Functional Outcomes".

The estimands are built on the phase-centered elastic mean: the average of
the curves registered to their Frechet mean shape and re-timed by the
inverse mean warp (``elastic_mean``). The Karcher mean function, the
alternative curve built from the same Steps 1 and 2, is also available
(``karcher_mean_function``, or ``mean="karcher"`` in the estimators).

Example:
    >>> import numpy as np
    >>> from shape_preserving_effects import (
    ...     drop_rebound_trial, local_effect, populational_effect
    ... )
    >>> t = np.linspace(0, 1, 81)
    >>> trial = drop_rebound_trial(n=250, t=t, seed=0)
    >>> pop = populational_effect(t, trial.Y, trial.A, trial.X)
    >>> loc = local_effect(t, trial.Y, trial.A, trial.X, k=3)
    >>> pop.effect.shape, loc.effect.shape
    ((81,), (81,))
"""

from shape_preserving_effects.competitors import fate_drfos, srvf_kernel_ate
from shape_preserving_effects.datasets import (
    DGP,
    Trial,
    drop_rebound_dgp,
    drop_rebound_trial,
    main_dgp,
    population_estimands,
    sample_trial,
)
from shape_preserving_effects.estimators import (
    LocalEffect,
    PopulationalEffect,
    local_effect,
    logistic_propensity,
    populational_effect,
)
from shape_preserving_effects.mean import (
    MEANS,
    ElasticMeanResult,
    elastic_mean,
    karcher_mean_function,
    mean_curve,
    phase_centre,
    pointwise_mean,
)
from shape_preserving_effects.metrics import (
    METRICS,
    depth,
    drop_speed,
    l2_error,
    onset_time,
    shape_metrics,
)
from shape_preserving_effects.registration import register
from shape_preserving_effects.srsf import (
    from_srsf,
    invert_warp,
    psi_to_warp,
    to_srsf,
    warp_curve,
    warp_srsf,
    warp_to_psi,
)

__all__ = [
    "DGP",
    "MEANS",
    "METRICS",
    "ElasticMeanResult",
    "LocalEffect",
    "PopulationalEffect",
    "Trial",
    "depth",
    "drop_rebound_dgp",
    "drop_rebound_trial",
    "drop_speed",
    "elastic_mean",
    "fate_drfos",
    "from_srsf",
    "invert_warp",
    "karcher_mean_function",
    "l2_error",
    "local_effect",
    "logistic_propensity",
    "main_dgp",
    "mean_curve",
    "onset_time",
    "phase_centre",
    "pointwise_mean",
    "population_estimands",
    "populational_effect",
    "psi_to_warp",
    "register",
    "sample_trial",
    "shape_metrics",
    "srvf_kernel_ate",
    "to_srsf",
    "warp_curve",
    "warp_srsf",
    "warp_to_psi",
]

__version__ = "1.0.0"
__author__ = "Agathe Chabassier"
__license__ = "AGPL-3.0-only"
__copyright__ = "Copyright (c) 2026 Withings, Inria, Sorbonne Université"
