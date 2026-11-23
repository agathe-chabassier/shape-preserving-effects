# Shape-preserving treatment effects for functional outcomes

Code of the paper

> Agathe Chabassier, Joris Olivier, Julie Josse and Erwan Scornet.
> *Shape-Preserving Treatment Effects for Functional Outcomes.* 2026.
> Preprint: <https://hal.science/hal-05838408>.

**Version 1.0.0**, the first public release and the version of the paper
(second version of the preprint). [CHANGELOG.md](CHANGELOG.md) describes it,
and lists what differs from the unpublished code behind the first version
of the preprint.

When the outcome of a study is a **curve** (a weight trajectory after the
start of a treatment, a hormonal cycle, a growth curve), the usual causal
estimand, the functional average treatment effect (FATE), averages the
curves time point by time point. When individual curves share a shape but
not its timing, this pointwise average flattens the shape: the estimated
effect can look like no individual response at all.

This repository provides estimators of **elastic** treatment effects, which
average the curves in the Fisher–Rao (SRSF) geometry and keep both the
typical shape and the typical timing of the effect:

* `populational_effect`: inverse-probability-weighted (IPW) estimator of the
  **populational effect** τ_pop = μ(Y⁽¹⁾) − μ(Y⁽⁰⁾), the difference between
  the elastic means of the treated and control potential outcomes;
* `local_effect`: matching estimator of the **local effect** τ_loc, the
  elastic mean of the differences between comparable treated and control
  units.

**The elastic mean.** All estimands are built on the *phase-centered elastic
mean* (`elastic_mean`): the curves are registered to their mean shape,
re-timed to the typical timing of the sample, and averaged (three steps,
detailed under *elastic mean* in the [glossary](#8-glossary)). It keeps the
average starting and final values of the curves, and without timing variation
it is the pointwise mean. The **Karcher mean function** of `fdasrsf` and
`scikit-fda` is available for comparison (`mean="karcher"`); Appendix J of
the paper compares the two.

It also contains everything needed to reproduce the simulation study of the
paper: the simulation code, the stored results, and the scripts that draw
every simulation figure and table.

The package depends on `numpy` only; `matplotlib` is needed for the figures.

Technical terms are defined in the [glossary](#8-glossary) at the end
of this file.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Using the estimators](#3-using-the-estimators)
4. [Reproducing the simulations of the paper](#4-reproducing-the-simulations-of-the-paper)
5. [Tests](#5-tests)
6. [What is not in this repository](#6-what-is-not-in-this-repository)
7. [Author, citation, licence, contact](#7-author-citation-licence-contact)
8. [Glossary](#8-glossary)

---

## 1. Repository layout

```
shape-preserving-effects/
├── README.md                  this file
├── requirements.txt           Python packages to install
├── pyproject.toml             package metadata (for "pip install")
├── LICENSE                    AGPL-3.0-only licence
├── CITATION.cff               how to cite
├── CHANGELOG.md               changes between versions
├── .pre-commit-config.yaml    formatting and lint checks run before each commit
│
├── shape_preserving_effects/  the Python package
│   ├── estimators.py          populational (IPW) and local (matching) estimators
│   ├── mean.py                weighted phase-centered elastic (Fréchet) mean
│   ├── registration.py        elastic registration of two curves (dynamic programming)
│   ├── srsf.py                square-root slope function (SRSF) transform, warps
│   ├── competitors.py         DR-FoS and SRVF kernel ATE (the two competitors)
│   ├── datasets.py            simulation model of the paper, causal estimands
│   ├── metrics.py             shape metrics (depth, onset time, drop speed), L2 error
│   └── validation.py          input checks shared by the modules
│
├── examples/
│   ├── quickstart.py          estimate the effects on one simulated study
│   ├── own_data.py            estimate the effects from a csv file
│   └── example_data.csv       example csv file (simulated)
│
├── tests/
│   ├── test_package.py        unit tests of the estimators and of the elastic mean
│   ├── test_registration.py   properties of the registration
│   ├── test_inputs_and_matching.py  input checks, units of the curves, matching with ties
│   └── test_fdasrsf_crosscheck.py   comparison with the fdasrsf package (optional)
│
└── simulations/               reproduction of the simulation study
    ├── main_study.py          causal estimands, five scenarios, phase sweep
    ├── sensitivity.py         sensitivity analyses (sample size, propensity model, dimension)
    ├── noise_and_null.py      causal estimands under noise and under no effect
    ├── estimand_gap.py        FATE / tau_pop gap over the (alpha, p_rebound) grid
    ├── estimand_resolution.py Monte-Carlo resolution of tau_pop (800 and 3200 draws)
    ├── make_figures.py        draws every figure and table from the results
    ├── check_reproducibility.py   recomputes a sample of the results and compares
    ├── fig_means_grid.py      functional means across data-generating processes
    ├── functional_means.py    the samples and averaging operators of that figure
    ├── common.py, plotting.py, figures_*.py, fig_intro.py, fig_mean_steps.py
    │                          shared settings, plotting style, figure code
    ├── results/               stored results of the paper (about 4 MB)
    └── figures/               the figures, the table and summary numbers, as produced by make_figures.py
```

---

## 2. Installation

You need **Python 3.10 or later** (`python3 --version`; on Windows, use
`python` instead of `python3` in every command of this file). Download the
repository (**Code** → **Download ZIP** on the GitHub page) or clone it, and
run every command from its folder:

```bash
git clone https://github.com/agathe-chabassier/shape-preserving-effects.git
cd shape-preserving-effects
```

### 2.1 Create an environment and install the dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # NumPy, matplotlib (figures), pytest (tests)
pip install -e .                   # the package itself, from this folder
```

The last command installs the package `shape_preserving_effects` in
*editable* mode: the examples, the tests and the simulation scripts import
it from this folder, and a change in the code is taken into account without
installing again.

In a new terminal, activate the environment again. Then check the
installation: after about a minute, the last line should read
`81 passed, 1 skipped` (the skipped file is the optional comparison of
section 5).

```bash
python3 -m pytest
```

### 2.2 (Optional) Use the package from your own projects

To use the package in another project without downloading the repository,
install it from GitHub (only `shape_preserving_effects/` and NumPy are
installed; add `@v1.0.0` at the end of the address for the version of the
paper):

```bash
pip install "git+https://github.com/agathe-chabassier/shape-preserving-effects.git"
```

```python
from shape_preserving_effects import populational_effect, local_effect

# t: common time grid; Y: (n, m) curves; A: 0/1 treatment; X: confounders
pop = populational_effect(t, Y, A, X)
loc = local_effect(t, Y, A, X, k=3)
```

---

## 3. Using the estimators

### 3.1 A short example

```python
import numpy as np
from shape_preserving_effects import drop_rebound_trial, populational_effect, local_effect

t = np.linspace(0, 1, 81)  # common time grid
trial = drop_rebound_trial(n=250, t=t, seed=0)  # a simulated study
pop = populational_effect(t, trial.Y, trial.A, trial.X)
loc = local_effect(t, trial.Y, trial.A, trial.X, k=3)
```

`drop_rebound_trial` draws a simulated observational study of 250 units
from the simulation model of the paper (Scenario 1): `trial.Y` (observed
curves), `trial.A` (treatment) and `trial.X` (confounder), in the format of
section 3.2, and the potential outcomes `trial.Y1` and `trial.Y0`.
`pop.effect` and `loc.effect` are the estimated effect curves on the grid
`t`:

```python
import matplotlib.pyplot as plt

plt.plot(t, pop.effect, label="populational")
plt.plot(t, loc.effect, label="local")
plt.legend()
plt.show()
```

### 3.2 Data format

The estimators take NumPy arrays:

| argument | shape | content |
|---|---|---|
| `t` | `(m,)` | common time grid, **equally spaced** (any time unit) |
| `Y` | `(n, m)` | outcome curves, one row per unit, evaluated on `t` |
| `A` | `(n,)` | treatment indicator: 1 treated, 0 control |
| `X` | `(n,)` or `(n, d)` | confounders (one row per unit) |

`n` is the number of units, `m` the number of time points and `d` the
number of confounders.

If your units are measured at different times, interpolate each curve on a
common grid first; `examples/own_data.py`
shows how to do it from a csv file in *long format* (one row per
measurement, rather than one row per unit). Every curve must cover the whole
time window.

The estimators check their inputs and stop with an explicit message if the
grid is not equally spaced and increasing, if a curve contains a missing
value (`NaN`), if the shapes do not match, if `A` is not coded 0 / 1, if
an arm is empty, or if an argument is out of range (`clip`, `k`,
`max_pairs`). Each message names the condition that failed.

**Noise.** The elastic mean registers (aligns) the curves before averaging
them, and registration interprets noise as shape. Noisy curves should
therefore be smoothed before estimation, keeping the features that matter
for the application (see the section *When noise dominates the signal: the
template bias* of the paper).

### 3.3 The two estimators

```python
populational_effect(
    t, Y, A, X, clip=0.02, n_iter=3, mean="elastic", estimand="ATE", propensity=None
)
```

Fits a logistic propensity score on the whole sample (or uses the scores
given in `propensity`), clips it to `[clip, 1 - clip]`, and computes in each
arm the phase-centered elastic mean with inverse-propensity weights inside
Steps 1 and 2 (Fréchet mean shape, mean warp) and in the Step-3 average of
the registered curves (normalised to sum to one in each arm). The weights
depend on the population targeted:

| `estimand` | treated units | control units |
|---|---|---|
| `"ATE"` (default): all units | 1 / e(X) | 1 / (1 − e(X)) |
| `"ATT"`: the treated | 1 | e(X) / (1 − e(X)) |
| `"ATC"`: the controls | (1 − e(X)) / e(X) | 1 |

| argument | meaning |
|---|---|
| `t, Y, A, X` | the data (section 3.2) |
| `clip` | the propensity scores (estimated or given) are clipped (truncated) to `[clip, 1 - clip]` before weighting. Without it, a unit whose propensity is close to 0 or 1 would get a huge weight and dominate its arm. With the default 0.02, no weight exceeds 50. Must be in `[0, 0.5)`. |
| `n_iter` | number of register / average alternations used to find the mean shape (Step 1 of the elastic mean, see *elastic mean* in the glossary); the loop stops earlier if the mean shape no longer changes. More alternations bring the result closer to the exact elastic mean; the computing time grows like `n_iter + 1`. Default 3; the paper uses 2. |
| `mean` | `"elastic"` (default): the phase-centered elastic mean, the estimand of the paper; `"karcher"`: the Karcher mean function built from the same Steps 1 and 2, for comparison (see the glossary). |
| `estimand` | population targeted by the weights: `"ATE"`, `"ATT"` or `"ATC"` (table above). |
| `propensity` | propensity scores P(A = 1 \| X) estimated with another model (for instance gradient boosting, preferably cross-fitted), one per unit; `X` is then not used. `None` (default) fits the logistic model on `X`. |

This estimator involves no randomness: the same data always give the same
result.

Returns an object with

* `.effect`: the estimated populational effect (array of length `m`);
* `.arm1.mean`, `.arm0.mean`: the weighted elastic means of the treated and
  control arms (`effect = arm1.mean - arm0.mean`); these two curves are
  often easier to interpret than their difference;
* `.e_hat`: the propensity scores used (after clipping);
* `.weights`: the weight of each unit, before normalisation within its arm.

```python
local_effect(t, Y, A, X, k=5, estimand="ATE", n_iter=3, max_pairs=None, rng=None, mean="elastic")
```

Matches every unit to its k nearest neighbours in the other arm, using the Euclidean distance between their covariates after standardising each covariate (dividing it by its standard deviation in the whole sample), so that a covariate measured on a large scale does not dominate the distance. When several units are tied at the k-th distance (frequent with discrete covariates such as sex or an age band), all of them are kept and share the weight of the k matches, as in Abadie and Imbens (2006), so the estimate does not depend on the order of the rows.

| argument | meaning |
|---|---|
| `t, Y, A, X` | the data (section 3.2) |
| `k` | number of neighbours of each unit in the other arm. Default 5; the paper uses 3. |
| `estimand` | the population over which the effect is averaged: `"ATE"` (all units: every unit is matched), `"ATT"` (for each treated unit, find `k` control units)  or `"ATC"` (for each control unit, find `k` treated units). |
| `n_iter` | as for `populational_effect`. |
| `max_pairs` | maximal number of matched differences (`k × n` for the ATE, `k × n1` for the ATT, `k × n0` for the ATC). Above it, a random subsample of `max_pairs` differences is used, which bounds the computing time and estimates the same quantity. `None` (default) keeps every difference. The paper uses 300. |
| `mean` | as for `populational_effect`. |
| `rng` | seed of the random subsample of `max_pairs` (ignored without subsampling): an integer gives the same estimate at every run, `None` (default) a slightly different one. **Set it to an integer whenever you set `max_pairs`.** |

Returns an object with `.effect` (the estimated local effect),
`.differences` (the matched difference curves whose elastic mean is
`.effect`, after subsampling) and `.mean_result` (the full output of
`elastic_mean` on these differences; its `.weights` are the weights of the
pairs, all equal unless there are ties).

**Computing time.** The cost is driven by the registrations, about 10 ms
each on one core on a grid of 100 points; the elastic mean of `N` curves
needs `(n_iter + 1) × N` of them. With n = 250 units, each estimator takes a
few seconds, and the populational estimator on 5,000 curves a few minutes. On
large samples, give the local estimator a `max_pairs` (a few thousand) and an
integer `rng`. The paper uses

```python
pop = populational_effect(t, Y, A, X, n_iter=2)
loc = local_effect(t, Y, A, X, k=3, n_iter=2, max_pairs=300, rng=0)
```

### 3.4 Examples

```bash
python3 examples/quickstart.py
```

simulates one study, estimates τ_pop, τ_loc and the FATE (with the
DR-FoS estimator), approximates the true τ_pop of the simulation model on
simulated units, prints the L² distance of each estimate to it, and saves the figure `quickstart.png`
in the current folder.

```bash
python3 examples/own_data.py                      # uses examples/example_data.csv
python3 examples/own_data.py my_data.csv          # your own file
```

reads a csv file in long format (columns `id`, `day`, `outcome`, `treated`,
`x`), interpolates the curves on a common grid, and saves the estimated
effects (`own_data_effects.csv`) and a figure (`own_data_effects.png`). It
stops with a message naming the line if a cell is empty or if the treatment
or a confounder of a unit changes from one row to another. To use
other column names, a different time window, or several confounders, edit
the constants at the top of the script `own_data.py` (`TIME`, `OUTCOME`, `TREATMENT`, `ID`,
`COVARIATES`, `START`, `END`).

### 3.5 Other functions

All of them are imported in the same way, for instance
`from shape_preserving_effects import elastic_mean`.

**Elastic geometry.**

| function | purpose |
|---|---|
| `elastic_mean(F, t, weights=None, n_iter=3)` | weighted phase-centered elastic mean of the curves `F` (one per row, shape `(N, m)`); `weights`: one non-negative weight per curve, `None` for equal weights. Returns an object with `.mean` (the mean curve), `.registered` (the phase-centered registered curves, whose weighted average is `.mean`) and `.karcher_mean_function`; the docstring lists the other attributes. |
| `karcher_mean_function(F, t, weights=None, n_iter=3)` | the Karcher mean function of the curves `F`: the curve whose SRSF is the reference representative re-timed by the inverse mean warp, started at the weighted mean initial value (same as `elastic_mean(...).karcher_mean_function`) |
| `pointwise_mean(F, weights=None)` | weighted pointwise (cross-sectional) mean of the rows of `F` |
| `to_srsf(f, t)` | SRSF of the curve `f` |
| `from_srsf(q, t, f0=0.0)` | curve whose SRSF is `q` and whose value at the first time point is `f0` (the SRSF does not record the starting value of the curve, so it must be given back) |
| `register(q1, q2, t)` | optimal warp aligning the SRSF `q2` to the SRSF `q1`; returns the pair `(gamma, dist)`, the warp evaluated on `t` and the distance between `q1` and the aligned `q2` |

**Propensity score and competitors.**

| function | purpose |
|---|---|
| `logistic_propensity(X, A, clip=0.02)` | the propensity score used by `populational_effect` (logistic regression of `A` on `X`, clipped to `[clip, 1 - clip]`) |
| `fate_drfos(t, Y, A, X, n_folds=5, clip=0.02, rng=None)` | DR-FoS estimator of the FATE (Testa et al.), the pointwise competitor of the paper. Its two nuisance models, the propensity score and the outcome regressions of the two arms, are *cross-fitted*: the units are split at random into `n_folds` groups, and the models used for each group are fitted on the other groups. `rng` is the seed of this random split. The function stops if a group leaves too few units of an arm to fit its outcome regression (more covariates than units). |
| `srvf_kernel_ate(t, Y, A, X, lambdas=(1e-3, 1.0, 1e3), n_iter=2, rng=None)` | SRVF operator kernel ATE (Raykov et al.), the elastic competitor of the paper, a kernel ridge regression of the aligned curves on treatment and covariates. The ridge penalty is chosen among `lambdas` by the prediction error on a random 20 % of the units; `rng` is the seed of this random split. It follows the authors' released code (see below). |

With `return_arms=True`, both competitors return the triple
`(effect, arm1, arm0)` instead of `effect` alone.

`srvf_kernel_ate` follows the code released by Raykov et al.
(<https://github.com/JordanRaykov/Kernel-based-estimators-for-Functional-Causal-Effects>,
commit `8a5db6c`). One point differs. The released code stacks the outcomes time by time while
its linear system is ordered unit by unit; here both are ordered unit by
unit, as in Algorithm 1 of their paper.

**Simulation model and evaluation.**

| function | purpose |
|---|---|
| `drop_rebound_dgp(phase=1.0, p_rebound=1.0, noise_sd=0.03)` | the simulation model of the paper (its full definition is in the docstring of `shape_preserving_effects/datasets.py`). `phase` is the phase-variation scale α (how much the timing of the treated response varies between units; 0 = no variation); `p_rebound` is the share of treated responses that rebound after the drop (the others plateau): 1, 0.75, 0.5, 0.25 and 0 in Scenarios 1 to 5; `noise_sd` is the scale σ_ε of the smooth noise. `main_dgp` is Scenario 1. |
| `drop_rebound_trial(n, t, seed=0, phase=1.0, p_rebound=1.0)` | draws one study of `n` units from `drop_rebound_dgp` |
| `sample_trial(dgp, n, t, seed=0)` | draws one study of `n` units from any model `dgp` |
| `population_estimands(dgp, t, n_pop=800, seed=20240, n_iter=3, mean="elastic")` | the true causal effects of a model (`tau_pop`, `tau_ind`, `tau_loc`, `fate`, and the elastic means `mu1`, `mu0` of the two potential outcomes), approximated on `n_pop` simulated units; returns a dictionary, for instance `E = population_estimands(main_dgp, t)` then `E["tau_pop"]` |
| `shape_metrics(tau, t)` | the three shape metrics of a curve `tau`: depth (its minimum), onset time (time at which it reaches half of its depth on the way down) and drop speed (its steepest downward slope) |
| `l2_error(tau_hat, tau_ref, t)` | L² distance between two curves |

Every function is documented in its docstring (`help(populational_effect)`).

### 3.6 Random numbers and seeds

Only the following elements include some randomness: the simulated data (`seed` argument of
`drop_rebound_trial`, `sample_trial` and `population_estimands`), the pair
subsample of `local_effect` (when `max_pairs` is set), the cross-fitting split
of `fate_drfos` and the hold-out split of `srvf_kernel_ate` (`rng` argument).
An integer `seed` or `rng` gives the same result at every run; `None` gives a
slightly different one. All other functions, including `populational_effect`
and `elastic_mean`, are deterministic.

---

## 4. Reproducing the simulations of the paper

All commands are run from the repository folder, with the environment
activated (section 2.1).

### 4.1 Redraw every figure and table from the stored results (about half a minute)

The results of every simulation of the paper are stored in
`simulations/results/`. To redraw the figures and tables from them (no
estimator is run):

```bash
python3 simulations/make_figures.py
```

The figures are written to `simulations/figures/` as PDF files, together
with the sensitivity table (`tab_sensitivity.tex`) and three files of summary
numbers, including those quoted in the text (`numbers_main.json`,
`numbers_noise.json`, `numbers_sensitivity.json`). The table of the appendix *The causal-estimand
gap over the (α, p_rebound) grid* is printed in the terminal.

| paper | content | file | script that computes the results |
|---|---|---|---|
| Figure 1 | FATE vs. shape-preserving effect (introduction) | `fig_intro_fate.pdf` | `fig_intro.py` |
| Figure 3 | steps of the phase-centered elastic mean | `fig_mean_steps.pdf` | `fig_mean_steps.py` |
| Figure 4 | causal estimands across scenarios | `fig_est_grid.pdf` | `main_study.py` |
| Figure 5 | Monte-Carlo distribution of the estimators, Scenarios 1 and 3 | `fig_mc_grid_main.pdf` | `main_study.py` |
| Figure 6 | effect of the phase-variation scale | `fig_phase.pdf` | `main_study.py` |
| Figure 7 | causal estimands as the noise level grows | `fig_bias_grid.pdf` | `noise_and_null.py` |
| Figure 17 | phase-centered elastic mean vs Karcher mean function, on three samples of Figure 21 | `fig_compare_means.pdf` | `figures_compare_means.py` (samples of `fig_means_grid.py`) |
| Figure 18 | the three shape evaluation metrics | `fig_metricschema.pdf` | (schematic) |
| Figure 19 | Monte-Carlo distribution of the estimators, all scenarios | `fig_mc_grid.pdf` | `main_study.py` |
| Figure 20 | absolute errors of the estimators | `fig_boxes_grid.pdf` | `main_study.py` |
| Figure 21 | functional means across data-generating processes | `fig_means_grid.pdf` | `fig_means_grid.py` |
| Figure 22 | causal estimands when the treatment has no effect | `fig_bias_null_grid.pdf` | `noise_and_null.py` |
| Figure 23 | sensitivity to the sample size | `fig_sens_n.pdf` | `sensitivity.py`, `estimand_resolution.py` (dotted line) |
| Figure 24 | sensitivity to propensity-model misspecification | `fig_sens_mis.pdf` | `sensitivity.py` |
| Figure 25 | sensitivity to the covariate dimension | `fig_sens_dim.pdf` | `sensitivity.py` |
| Table 2 | sensitivity analyses, median [IQR] of the L² errors | `tab_sensitivity.tex` | `sensitivity.py` |
| table of Appendix K.2 | causal-estimand gap over the (α, p_rebound) grid | printed in the terminal | `estimand_gap.py` |
| numbers of Appendix K (*Estimands and reference values*) | Monte-Carlo resolution of the estimands | `results/resolution/resolution.json` | `estimand_resolution.py` |

### 4.2 Check that the stored results are reproducible (5 to 10 minutes)

```bash
python3 simulations/check_reproducibility.py
```

recomputes from scratch the causal estimands of Scenario 1, one replication
of each study (Scenarios 1 and 3, phase sweep, three sensitivity analyses),
one cell of the estimand gap table and one row of Figure 21, and compares
them with the stored results. Every line should start with `[ok]` (a
difference below `1e-8`); in practice the differences are at the level of
floating-point round-off, about `1e-14`, and can vary slightly between
computers and library versions.

### 4.3 Recompute every result from scratch (about 13 hours on one core)

The stored results can be fully recomputed. To keep them intact, write the
new results to another folder (here `my_results`) with the option
`--results`:

```bash
python3 simulations/main_study.py      --results my_results --jobs 4
python3 simulations/sensitivity.py     --results my_results --jobs 4
python3 simulations/noise_and_null.py  --results my_results --jobs 4
python3 simulations/estimand_gap.py    --results my_results --jobs 4
python3 simulations/estimand_resolution.py --results my_results --jobs 4
python3 simulations/make_figures.py    --results my_results --figures my_figures
```

* `--jobs 4` runs four simulations in parallel (use at most the number of
  cores of your computer). On one core, the five scripts take about 2 h 30,
  8 h, 10 min, 15 min and 25 min; with `--jobs 4` the total is less than
  4 hours.
* The scripts can be interrupted at any time: every finished replication is
  saved immediately, and running the same command again only computes what
  is missing. `main_study.py` and `sensitivity.py` accept `--status`, which
  prints the progress, for instance
  `python3 simulations/main_study.py --results my_results --status`.
* A *replication* is one simulated study, with its estimates. Every
  replication has its own fixed seed (section 3.6), so the results do not
  depend on the number of jobs or on the order in which the replications
  run. The new
  results are identical to the stored ones up to floating-point round-off,
  and so are the figures in `my_figures/`.

### 4.4 What each simulation script does

All settings are those of the paper; seeds are fixed in the scripts.

**`main_study.py`** (Simulations section; Appendix K). The causal estimands
of the five scenarios (τ_pop, τ_ind, τ_loc, FATE and the arm-level means,
computed on 800 simulated units), 50 replications of a study of n = 250 units
per scenario with the four estimators, and the phase sweep in Scenario 1
(α from 0 to 1, 10 replications per value). Results in `results/main/`.

**`sensitivity.py`** (Appendix K.4). Scenario 1 with one mechanism changed at
a time, the causal effect staying the same:

* sample size: n from 20 to 1000 (configurations `n20` to `n1000`; `base`
  is n = 250);
* misspecified propensity score: the treatment depends on four covariates
  through a mix of a linear part and of interactions (products of
  covariates), with a share λ ∈ {0, 0.5, 1} of interactions
  (configurations `int0`, `int50`, `int100`); the estimators fit a
  logistic model without interactions, which is wrong as soon as λ > 0;
* covariate dimension: d ∈ {2, 10, 25, 50, 100} covariates that all matter
  equally (a *dense* design; configurations `g2` to `g100`).

In the misspecification and dimension analyses, two reference estimators are
added: the unweighted difference of the elastic means of the two arms
(`elastic_unw`, which ignores confounding) and the IPW elastic estimator with
the true propensity scores (`elastic_orc`, an *oracle*, impossible in
practice). The estimand is computed on 3200 simulated units. Results in `results/sensitivity/`.

**`noise_and_null.py`** (Section 6.4; Appendix K.3). The causal estimands of
Scenario 1 for the same 800 units at noise levels σ_ε ∈ {0, 0.03, 0.10, 0.30}
(σ_ε is the scale of the smooth noise, see `drop_rebound_dgp`; 0.03
everywhere else in the paper), and under two designs without treatment effect: both
potential outcomes follow the control law (`null_flat`), or both follow the
treated law (`null_shape`). Results in `results/noise/`.

**`estimand_gap.py`** (Appendix K.2). The L² distance ‖τ_FATE − τ_pop‖
between the two estimands, for α ∈ {0, 0.5, 1} and the five rebound shares,
on 800 units per cell. Results in `results/estimand_gap/`.

**`estimand_resolution.py`** (Appendix K, *Estimands and reference
values*; dotted line of Figure 23). The Monte-Carlo error of the estimands,
from τ_pop recomputed on independent sets of 800 draws (each scenario) and
3200 draws (Scenario 1). Results in `results/resolution/`.

**`figures_compare_means.py`** (Appendix J.3, Figure 17). The phase-centered
elastic mean and the Karcher mean function, both computed from the same run
of Steps 1 and 2 (`elastic_mean` returns both), on three samples of
Figure 21 (sinusoids, chirps, damped oscillations).

**`fig_means_grid.py`** (Appendix K.1, Figure 21). Seven samples of 30
curves averaged by six operators (pointwise, phase-centered elastic,
structural, DBA, soft-DTW, Wasserstein), written with NumPy only in
`functional_means.py`, whose docstring describes them. On another computer,
the soft-DTW average can differ from the stored one by about `1e-3`
(invisible on the figure). Results in `results/means_grid/means.npz`.

**Result files.** The `.csv` files have one row per replication (`rep`) and
estimator, with the L² distance to τ_pop (`l2`) and the signed errors of the
three shape metrics (`e_depth`, `e_onset_time`, `e_drop_speed`). The `.npz`
files are NumPy archives (`numpy.load(path)`) of curves on
`numpy.linspace(0, 1, 81)`.

---

## 5. Tests

```bash
python3 -m pytest
```

The tests check, among other things, that registration recovers a known
warp and returns valid warps on grids of any size, that the elastic mean has
the properties stated in the paper (shape kept under time shifts, mean
initial and final values, mean amplitude) and does not depend on the unit of
the curves, that the IPW estimator removes the confounding that biases the
unweighted one, that matching does not depend on the order of the rows when
covariates are tied, that the elastic and DR-FoS estimates do not change
when a constant is added to every curve, and that invalid inputs are
rejected with a message naming the problem.

The optional comparison with [fdasrsf](https://github.com/jdtuck/fdasrsf_python),
the reference implementation of elastic functional data analysis, requires
that package:

```bash
pip install fdasrsf
python3 -m pytest tests/test_fdasrsf_crosscheck.py
```

It checks that, without weights, our registration, mean warp and
phase-centered elastic mean agree with those of `fdasrsf`, which the
estimators cannot use directly because it does not support weights. Without
`fdasrsf`, these tests are skipped. The agreement is not exact: our registration restricts the warps to piecewise-linear paths whose local slopes take 11 fixed values between 1/4 and 4, whereas fdasrsf allows a finer set of slopes. On the pair of curves of the test the two
warps differ by less than 0.01, and the distance left after alignment is
about 5 % of the initial distance with our warp and 2 % with that of
`fdasrsf` (thresholds calibrated on `fdasrsf` 2.7.4).

**Code style.** The code is formatted and checked with
[ruff](https://docs.astral.sh/ruff/) (configuration in `pyproject.toml`:
Google-style docstrings, type hints, absolute imports) and type-checked with
`mypy`. To run these checks before every commit:

```bash
pip install pre-commit
pre-commit install
```

---

## 6. What is not in this repository

* **The real-data application** (weight trajectories after GLP-1 receptor
  agonist initiation): the individual data cannot be shared (GDPR and
  commercial confidentiality), so neither the data nor the analysis code
  are distributed.

---

## 7. Author, citation, licence, contact

The package was written by Agathe Chabassier (Withings; Inria,
PreMeDICaL team, IDESP (UMR Université de Montpellier - Inserm); Sorbonne
Université, LPSM), during a CIFRE PhD funded by Withings and the
ANRT.

If you use this code, please cite the paper. `CITATION.cff` contains the
reference in a standard format; GitHub shows it under **Cite this
repository** on the page of the repository.

Copyright (C) 2026 Withings, Inria, Sorbonne Université.

License: AGPL-3.0-only; see [LICENSE](LICENSE).

Questions and bug reports: open an issue on GitHub, or write to Agathe
Chabassier (agathe.chabassier@hotmail.fr), corresponding author of the paper.

---

## 8. Glossary

**Causal inference**

* **FATE** (*functional average treatment effect*): the pointwise
  difference of the mean curves of the two potential outcomes,
  E[Y⁽¹⁾(t)] − E[Y⁽⁰⁾(t)] at every time t.
* **DR-FoS**: the doubly robust estimator of the FATE of Testa et al., a
  competitor in the paper.

**Elastic geometry**

* **Pointwise (cross-sectional) mean**: the average of the curves taken
  separately at each time point. When curves have the same shape at
  different times, it flattens the shape.
* **Warp, warping function** γ: an increasing reparametrisation of the time
  interval onto itself; the curve f ∘ γ is f played faster or slower in
  places. The *phase* of a curve is its timing; its *amplitude* is its
  shape once the timing is removed.
* **SRSF** (*square-root slope function*), also called SRVF: the
  transformation q = sign(f′) √|f′| of a curve f. In this representation,
  the usual L² distance becomes the **Fisher–Rao** distance, which does not
  change when two curves are warped in the same way; this makes aligning
  curves well defined.
* **Registration, alignment**: finding the warp that makes one curve as
  close as possible to another (here by dynamic programming).
* **Elastic mean** (*phase-centered elastic mean*): the average used by
  the estimators, computed in three steps. Step 1 (**Fréchet mean shape**):
  starting from one of the curves, repeat *register every curve to the
  current mean shape, then average the aligned SRSFs* (`n_iter` times at
  most); the result is a representative of the mean shape, the **reference
  representative** q̃ (`.reference` in the code). Step 2 (**mean warp**):
  register every curve to the reference representative and average the warps
  (their Fréchet mean γ̄ on the space of warps). Step 3 (**phase-centering
  and averaging**): re-time every registered curve by the inverse of this
  mean warp, so that the curves have the
  typical timing of the sample (**phase-centering**; these are the
  *phase-centered registered curves*), and average these curves time point
  by time point. A **Fréchet mean** is the generalisation of the average to
  spaces in which only distances are defined (called Karcher mean in
  Riemannian geometry; not to be confused with the Karcher mean function
  below).
* **Karcher mean function**: the other curve that Steps 1 and 2 give,
  the curve whose SRSF is the reference representative re-timed by the
  inverse mean warp, started at the average initial value; it is the
  "Karcher mean" of `fdasrsf` and `scikit-fda`. In general it shrinks the
  amplitudes and does not end at the average final value; the package
  computes it for comparison only (`mean="karcher"`).
* **Template**: the curve g of the *template model* (all curves are g
  re-timed, rescaled and shifted); the **template bias** is the bias of the
  elastic mean when the curves are noisy (see the paper).
* **τ_pop** (populational effect): μ(Y⁽¹⁾) − μ(Y⁽⁰⁾), the difference between
  the elastic means μ of the two potential outcomes.
* **τ_loc** (local effect): the elastic mean of the differences between the
  curves of a treated and of a control unit with the same confounders.
* **τ_ind** (individual effect): the elastic mean of the individual effects
  Y⁽¹⁾ − Y⁽⁰⁾ of the units; computable in simulation only.
