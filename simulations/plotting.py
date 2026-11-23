"""Shared plotting style of the paper's figures.

Visual code
-----------
* solid lines: estimates, one colour per method;
* dashed lines: causal estimands, in the colour of the method that targets
  them (blue: tau_pop and the elastic means mu(Y^(a)); red: tau_FATE and the
  pointwise means E[Y^(a)]; brown: tau_ind; purple: tau_loc);
* short dashes: treated arm; dots: control arm;
* shaded bands: interquartile range across replications;
* green / orange thin lines: treated / control potential outcomes;
  light grey: individual differences.

Curve panels have no tick marks, arrow-headed axes and shared vertical
limits; panels with quantitative axes keep their numbers.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

# Methods (solid lines)
BLUE = "#1950A0"  # elastic populational estimator
PURPLE = "#7B3FA0"  # elastic local estimator
YELLOW = "#D9A400"  # SRVF kernel ATE (Raykov et al.)
RED = "#B22D26"  # DR-FoS (Testa et al.)
GREY_M = "#767676"  # unweighted elastic estimator
DGREY_M = "0.25"  # oracle-weighted elastic estimator
# Causal estimands without an estimator of their own
BROWN = "#8C5A2F"  # tau_ind
# Other curves
GREEN = "#167348"  # treated potential outcomes
ORANGE = "#C97B3A"  # control potential outcomes
GREY = "#BDBDBD"  # individual differences
# Annotations of the shape-metric schema
BR_DEPTH, BR_ONSET, BR_SPEED = "#2189BE", "#D14A6A", "#E39320"

# Dash patterns (dashes are reserved for causal estimands)
DASH_EST = (0, (5, 2.2))  # effect-level estimands
DASH_T = (0, (2.4, 1.4))  # treated-arm estimands
DASH_C = (0, (0.8, 1.3))  # control-arm estimands

EST_STYLE = {  # estimator -> (legend label, colour)
    "elastic_pop": ("elastic populational (ours)", BLUE),
    "elastic_loc": ("elastic local (ours)", PURPLE),
    "srvf": ("SRVF kernel ATE", YELLOW),
    "drfos": ("DR-FoS", RED),
    "elastic_unw": ("unweighted elastic (no adj.)", GREY_M),
    "elastic_orc": ("oracle-weighted elastic (true $e$)", DGREY_M),
}
ORDER = ["elastic_pop", "elastic_loc", "srvf", "drfos"]
TITLES = {
    "elastic_pop": "Elastic populational (ours)",
    "elastic_loc": "Elastic local (ours)",
    "srvf": "SRVF kernel ATE",
    "drfos": "DR-FoS",
}
SHORT = {"elastic_pop": "pop.", "elastic_loc": "local", "srvf": "SRVF", "drfos": "DR-FoS"}

XLAB = "Time since treatment"
YLIM = (-1.95, 1.35)  # vertical limits shared by the curve panels

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "TeX Gyre Termes",
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "axes.edgecolor": "0.25",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

# Didactic figures (introduction, elastic-mean steps): bold dark-grey titles.
DIDACTIC_RC = {"axes.titleweight": "bold", "axes.titlecolor": "0.3"}

FS_TITLE = FS_LEG = FS_TXT = 10.0
TEXTWIDTH_IN = 6.22  # text width of the manuscript, in inches
TITLE_PT = 9.0  # target size of the didactic panel titles in the paper


def use_fonts(base):
    """Set the font sizes of the next figure; ``base`` is the label size."""
    global FS_TITLE, FS_LEG, FS_TXT
    FS_TITLE, FS_LEG, FS_TXT = base + 0.5, base - 2.0, base - 1.0
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base + 0.5,
            "axes.labelsize": base,
            "legend.fontsize": base - 2.0,
            "xtick.labelsize": base - 2.0,
            "ytick.labelsize": base - 2.0,
        }
    )


def fit_fonts(fig_width_in, incl_frac=1.0, title_pt=TITLE_PT):
    """Font sizes such that titles render at ``title_pt`` once the figure is
    scaled to ``incl_frac`` times the text width in the paper."""
    use_fonts(title_pt * fig_width_in / (incl_frac * TEXTWIDTH_IN) - 0.5)


def arrows(ax):
    """Arrow heads at the end of the bottom and left axes."""
    ms = plt.rcParams["font.size"] * 0.40
    ax.plot(1, 0, ">", transform=ax.transAxes, clip_on=False, color="0.25", ms=ms, zorder=6)
    ax.plot(0, 1, "^", transform=ax.transAxes, clip_on=False, color="0.25", ms=ms, zorder=6)


def bare(ax, xlabel=XLAB, ylabel=None, ylim=YLIM, xlim=(0.0, 1.0), zero=True):
    """Curve panel: left and bottom axes with arrow heads, no ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    if zero:
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
    arrows(ax)


def quant(ax):
    """Quantitative panel: numbers kept, tick marks removed."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def combo(line, patch):
    """Legend handle drawing a median line over its interquartile band."""
    return (patch, line)


def titled_legend(fig, title, handles, labels, x, y, ncol=1):
    """Framed figure legend with a title, anchored by its top centre at (x, y)."""
    leg = fig.legend(
        handles=handles,
        labels=labels,
        title=title,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(x, y),
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="0.78",
        borderpad=0.6,
        labelspacing=0.45,
        handlelength=1.9,
        handletextpad=0.6,
        columnspacing=1.3,
        borderaxespad=0.0,
        alignment="left",
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
    )
    leg.get_title().set_fontweight("normal")
    leg.get_title().set_fontfamily("serif")
    leg.get_title().set_fontsize(plt.rcParams["legend.fontsize"])
    leg.get_frame().set_linewidth(0.7)
    return leg


def med_band_plot(ax, xs, per_x, estimators, xticks=None, xticklabels=None, logx=False):
    """Median (line) and interquartile band of each estimator along ``xs``."""
    for e in estimators:
        lbl, col = EST_STYLE[e]
        # an estimator without value at x (empty array) leaves a gap there
        med = [np.median(per_x[x][e]) if len(per_x[x][e]) else np.nan for x in xs]
        lo = [np.percentile(per_x[x][e], 25) if len(per_x[x][e]) else np.nan for x in xs]
        hi = [np.percentile(per_x[x][e], 75) if len(per_x[x][e]) else np.nan for x in xs]
        ax.plot(xs, med, color=col, lw=1.6, marker="o", ms=3, label=lbl)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels or [str(x) for x in xticks])
        ax.minorticks_off()
    quant(ax)
    arrows(ax)


def bottom_legend(fig, ax, ncol):
    """Frameless legend centred below the figure, from the handles of ``ax``."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
