"""
Three-panel bar plot of MIMIC-IV model performance.

Panels (left -> right):
    1. Binary Classification   -- AUROC averaged over 6 tasks
    2. Time-to-Event           -- C-index averaged over 3 tasks
    3. Regression              -- R^2 averaged over 4 tasks

Usage
-----
    # Show all available models in each panel (default):
    python bar_plot.py

    # Show only a chosen subset of CC-/Transformer-/Mamba- models.
    # XGBoost (classification), DeepHit (TTE), Most-Recent (regression),
    # and MOTOR (all panels) are always plotted regardless of selection.
    python bar_plot.py --models Transformer-NTP Transformer-TPP Transformer-ORA \\
                                Mamba-NTP     Mamba-TPP     Mamba-ORA

    # Equivalent: edit SELECTED_MODELS at the top of this file and run with no args.

Aggregation across tasks
------------------------
Two valid aggregation choices, controlled by AGG_MODE:

  'within_task' (default)
        Treat each per-task bootstrap distribution as ~Normal(mean_t, std_t^2)
        with near-independent draws across tasks:
            SE_agg = sqrt(sum_t std_t^2) / T
        Tight bars, captures sampling uncertainty within tasks only.

  'across_task'
        Treat each task's mean as one observation:
            SE_agg = SD_t(mean_t) / sqrt(T)
        Wider, but honest about between-task variability -- this is the right
        SE for the claim "ORA is robust across tasks". Note Most-Recent's CI on
        regression will be very wide because its per-task R^2 ranges from
        +0.910 (Platelets) down to -0.431 (PaO2).

If you have raw aligned bootstrap samples (n_bootstrap x n_tasks), the proper
CI is the percentile of {mean_b over tasks of m_{b,t}}. See
`aligned_bootstrap_ci` below.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from raw_data import (
    COLORS,
    HIGHLIGHT_METHODS,
    TASK_SPECIFIC_BASELINES,
    TASK_SPECIFIC_LABEL,
    data_classification,
    data_regression,
    data_tte,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If non-None, this list overrides the CLI --models flag when the script is
# run with no arguments. Useful when you keep editing the same selection.
SELECTED_MODELS = None
# Example:
SELECTED_MODELS = [
    'Transformer-NTP', 'Transformer-TPP', 'Transformer-ORA',
    'Mamba-NTP', 'Mamba-TPP', 'Mamba-ORA',
]

AGG_MODE = 'within_task'        # 'within_task' or 'across_task'
CI_MULTIPLIER = 1.96            # 1.96 = 95% CI; 1.0 = SE bars

ANNOTATE_VALUES = True
ANNOTATION_FONTSIZE = 8.5
ANNOTATION_ROTATION = 0         # horizontal text on top of bars

OUTPUT_DIR = Path('/user/zj2398/femr_chao_meds_v3/src/femr/visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate(data, mode=AGG_MODE):
    """Return (agg_mean, agg_se) of shape (n_methods,)."""
    means = data['mean']
    stds = data['std']
    n_tasks = means.shape[1]

    agg_mean = means.mean(axis=1)

    if mode == 'within_task':
        agg_se = np.sqrt((stds ** 2).sum(axis=1)) / n_tasks
    elif mode == 'across_task':
        if n_tasks < 2:
            raise ValueError("across_task mode needs >= 2 tasks")
        agg_se = means.std(axis=1, ddof=1) / np.sqrt(n_tasks)
    else:
        raise ValueError(f"Unknown AGG_MODE: {mode}")

    return agg_mean, agg_se


def aligned_bootstrap_ci(bootstrap_samples, ci=0.95):
    """
    Optional: compute aggregate CI from raw aligned bootstrap samples.

    Parameters
    ----------
    bootstrap_samples : np.ndarray, shape (n_bootstrap, n_tasks)
        Same bootstrap indices used per task.
    ci : float

    Returns
    -------
    mean, lo, hi : floats
    """
    agg = bootstrap_samples.mean(axis=1)
    alpha = (1 - ci) / 2
    return agg.mean(), np.quantile(agg, alpha), np.quantile(agg, 1 - alpha)


# ---------------------------------------------------------------------------
# Filtering by selected model names
# ---------------------------------------------------------------------------
def filter_panel(data, selected):
    """
    Keep models that are in `selected` OR in this panel's `always_show` list.
    `selected` is a set of method names (or None to keep all).
    """
    if selected is None:
        return data

    must = set(data.get('always_show', []))
    keep = set(selected) | must
    idx = [i for i, m in enumerate(data['methods']) if m in keep]
    if not idx:
        raise ValueError(f"No methods left after filtering panel "
                         f"'{data['panel_label']}'.")

    out = dict(data)
    out['methods'] = [data['methods'][i] for i in idx]
    out['mean'] = data['mean'][idx]
    out['std'] = data['std'][idx]
    out['colors'] = [data['colors'][i] for i in idx]
    return out


def validate_selection(selected):
    """Warn the user if a requested name isn't recognized anywhere."""
    if selected is None:
        return
    unknown = [m for m in selected if m not in COLORS]
    if unknown:
        print(f"[warning] unknown model name(s): {unknown}. "
              f"Valid names: {sorted(COLORS.keys())}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_panel(ax, data, ci_mult=CI_MULTIPLIER, annotate=ANNOTATE_VALUES):
    methods = data['methods']
    colors = data['colors']
    metric = data['metric']

    agg_mean, agg_se = aggregate(data)
    err = ci_mult * agg_se

    x = np.arange(len(methods))
    edgecolors = ['black' if m in HIGHLIGHT_METHODS else '#444444' for m in methods]
    linewidths = [1.4 if m in HIGHLIGHT_METHODS else 0.5 for m in methods]

    ax.bar(
        x, agg_mean, yerr=err, color=colors, width=0.74,
        edgecolor=edgecolors, linewidth=linewidths,
        error_kw=dict(ecolor='black', elinewidth=0.9, capsize=2.5, capthick=0.9),
    )

    if annotate:
        ymax = (agg_mean + err).max()
        ymin = min(0.0, agg_mean.min())
        offset = 0.015 * (ymax - ymin)
        for xi, m, e in zip(x, agg_mean, err):
            ax.text(
                xi, m + e + offset, f'{m:.3f}',
                ha='center', va='bottom',
                fontsize=ANNOTATION_FONTSIZE,
                rotation=ANNOTATION_ROTATION,
            )

    # Y-axis padding (handles negative aggregates if any)
    span = (agg_mean + err).max() - min(0.0, agg_mean.min() - err.max())
    if agg_mean.min() < 0:
        lo = agg_mean.min() - 0.05 * span
    else:
        lo = max(0.0, agg_mean.min() - 0.06 * span)
    hi = (agg_mean + err).max() + 0.18 * span
    ax.set_ylim(lo, hi)

    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(data['panel_label'], fontsize=12, pad=8)


def build_legend(fig, panels):
    """
    Shared legend on right side. Merges XGBoost / DeepHit / Most-Recent into a
    single 'Task-specific Baseline' entry. Preserves the order of first
    appearance across panels.
    """
    seen_labels = []
    label_to_color = {}
    label_to_edge = {}

    for d in panels:
        for m, c in zip(d['methods'], d['colors']):
            if m in TASK_SPECIFIC_BASELINES:
                label = TASK_SPECIFIC_LABEL
            else:
                label = m
            if label not in seen_labels:
                seen_labels.append(label)
                label_to_color[label] = c
                if m in HIGHLIGHT_METHODS:
                    label_to_edge[label] = ('black', 1.4)
                else:
                    label_to_edge[label] = ('#444444', 0.5)

    handles = []
    for label in seen_labels:
        edge_c, lw = label_to_edge[label]
        handles.append(plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=label_to_color[label],
            edgecolor=edge_c, linewidth=lw,
        ))

    fig.legend(
        handles, seen_labels,
        loc='center left', bbox_to_anchor=(0.86, 0.5),
        frameon=False, fontsize=9,
        handlelength=1.4, handleheight=1.0, labelspacing=0.7,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(selected_models=None, suffix=''):
    panels_raw = [data_classification, data_tte, data_regression]
    panels = [filter_panel(d, selected_models) for d in panels_raw]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, data in zip(axes, panels):
        plot_panel(ax, data)

    build_legend(fig, panels)
    fig.subplots_adjust(left=0.05, right=0.85, wspace=0.28, top=0.88, bottom=0.06)

    stem = f'mimic_comparison{suffix}'
    out_pdf = OUTPUT_DIR / f'{stem}.pdf'
    out_png = OUTPUT_DIR / f'{stem}.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MIMIC-IV model comparison.')
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Subset of CC-/Transformer-/Mamba- model names. Always-shown '
             'baselines (XGBoost / DeepHit / Most-Recent / MOTOR) are added '
             'automatically per panel.',
    )
    parser.add_argument(
        '--suffix', default='',
        help='Optional suffix for output filenames.',
    )
    args = parser.parse_args()

    chosen = args.models if args.models is not None else SELECTED_MODELS
    validate_selection(chosen)
    main(selected_models=chosen, suffix=args.suffix)
