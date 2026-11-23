# Changelog

Versions follow [semantic versioning](https://semver.org); a new major
version may change the values returned by the estimators.

## 1.0.0

First public release, accompanying the paper (second version of the
preprint, November 2026).

**Differences with the code behind the first version of the preprint**

The figures of the first version of the preprint (October 2026) were
produced by an earlier state of this code, which was not published. The
elastic estimators of the paper's simulations are unchanged: on the stored
results, `populational_effect`, `local_effect` with one covariate and every
causal estimand are equal bit for bit. What differs:

* `fate_drfos` cross-fits the outcome regressions on the same folds as the
  propensity score, as specified by Testa et al. (they were fitted on the
  whole sample). It raises an error when a fold leaves one arm only, or
  fewer units of an arm than coefficients of the outcome regression.
* `local_effect`: the covariates are scaled by their standard deviation in
  the whole sample for both matching directions (they were scaled by their
  standard deviation in the arm searched). No change with one covariate.
* `local_effect`: units tied at the k-th distance are all kept and weighted to account for k neighbors (Abadie and Imbens, 2006), so the estimate does not
  depend on the order of the rows. No change without ties.
* `srvf_kernel_ate`: when it chooses its ridge penalty, the alignment and
  the output kernel are computed on the 80 % of units used to fit instead of the whole dataset, as in
  the code released by Raykov et al. (all units were aligned first). The
  chosen penalty changes in a few replications only.
* `elastic_mean`: the Step 1 registration and averaging in the shape space stops on a relative change of the reference representative on which we align the curves, instead of an absolute change, so that the mean does not depend on the unit of the curves. No change for curves of order 1 (every simulation of the paper).
* `onset_time` returns `nan` for a curve whose minimum is not negative.
* `simulations/sensitivity.py`: in one replication at n = 20 an arm has two
  units, fewer than the k = 3 neighbours of the local estimator. This
  replication is left out for the local estimator and for DR-FoS, and
  stored as `nan`. DR-FoS has no value at d = 100.
* Consequence for the paper: seven simulation figures and the table of the
  sensitivity analyses differ slightly from those of the first version of
  the preprint.
