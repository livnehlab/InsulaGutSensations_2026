#!/usr/bin/env python3
"""
PyCharm-friendly standalone script to plot CR-opto correlations across multiple mice.

What it does
------------
For each mouse/day pair, loads:
- reward_and_opto_related_averages.pkl

Then creates:
1) One figure per metric with one subplot per mouse
   - cr_opto vs odor_to_prelick
   - cr_opto vs reward_0_2s
   - cr_opto vs swallow_1p5_2s
2) One combined summary figure with rows = target metrics, columns = mice
3) A CSV + PKL with correlation stats for all mice and all metrics

Edit only the USER SETTINGS block and run in PyCharm.
"""

import os
import csv
import pickle
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress


# ===================== USER SETTINGS =====================
MOUSE_DAY_LIST = [
    ('OI151', '240827'),
    ('OI152', '241003'),
    ('OI116', '240529'),
    ('OI124', '240610'),
    ('OI208', '250227'),
]


COMBINED_AVERAGES_TEMPLATE = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis_v3\reward_and_opto_related_averages.pkl"
SAVE_TEMPLATE = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis_v4"

# Optional central output folder. If empty, figures are saved under the first mouse/day SAVE_TEMPLATE folder
GLOBAL_SAVE_LOC = r'Z:\Stavsht\Omer_OXTR\inhibition_correlation_summary_v4'

PREFIX = 'cr_opto_corr_multi_mouse'
X_KEY = 'cr_opto'
X_LABEL = 'Mean activity during CR opto window'

MAKE_PER_METRIC_FIGURES = True
MAKE_COMBINED_GRID = True
SAVE_PDF = True
MAKE_R_VS_P_SUMMARY = True
R_VS_P_CORR_KIND = 'pearson'   # 'pearson' or 'spearman'
ANNOTATE_R_VS_P_POINTS = True
# =========================================================


PRETTY_NAMES = {
    'cr_opto': 'CR opto window',
    'odor_to_prelick': 'Odor onset to 100 ms before first lick',
    'reward_0_2s': '0 to 1 s after reward',
    'swallow_1p5_2s': '1 to 2 s after reward',
}

TARGET_KEYS = ['odor_to_prelick', 'reward_0_2s', 'swallow_1p5_2s']

def build_stats_rows(dataset_list):
    stats_rows = []

    for ds in dataset_list:
        for metric_key in TARGET_KEYS:
            stats = compute_corr_stats(ds['x'], ds['metrics'][metric_key])
            stats_rows.append({
                'mouse': ds['mouse'],
                'day': ds['day'],
                'x_metric': X_KEY,
                'y_metric': metric_key,
                **{k: v for k, v in stats.items() if k not in {'x', 'y'}},
            })

    return stats_rows


def make_r_vs_p_summary(stats_rows, save_loc, corr_kind='pearson', annotate=True):
    if corr_kind == 'pearson':
        r_key = 'pearson_r'
        p_key = 'pearson_p'
        y_label = 'Correlation (r)'
        suffix = 'pearson'
    elif corr_kind == 'spearman':
        r_key = 'spearman_rho'
        p_key = 'spearman_p'
        y_label = 'Correlation (rho)'
        suffix = 'spearman'
    else:
        raise ValueError("corr_kind must be 'pearson' or 'spearman'")

    fig, axes = plt.subplots(1, len(TARGET_KEYS), figsize=(4.8 * len(TARGET_KEYS), 4.5), squeeze=False)
    axes = axes.ravel()

    for ax, metric_key in zip(axes, TARGET_KEYS):
        rows = [
            row for row in stats_rows
            if row['y_metric'] == metric_key
            and np.isfinite(row[r_key])
            and np.isfinite(row[p_key])
        ]

        if not rows:
            ax.set_title(f'{PRETTY_NAMES[metric_key]}\nNo valid data')
            ax.set_xlabel('P-value')
            ax.set_ylabel(y_label)
            continue

        pvals = np.asarray([row[p_key] for row in rows], dtype=float)
        rvals = np.asarray([row[r_key] for row in rows], dtype=float)

        ax.scatter(pvals, rvals, alpha=0.85)
        ax.axvline(0.05, linestyle=':')

        if annotate:
            for row, px, ry in zip(rows, pvals, rvals):
                ax.text(px, ry, row['mouse'], fontsize=8, ha='left', va='bottom')

        ax.set_xlabel('P-value')
        ax.set_ylabel(y_label)
        ax.set_title(PRETTY_NAMES[metric_key])
        ax.set_ylim(-1.05, 1.05)

        xmax = max(0.06, np.nanmax(pvals) * 1.08)
        ax.set_xlim(0, xmax)

    fig.suptitle(f'{PRETTY_NAMES[X_KEY]}: correlation vs p-value across mice', fontsize=14, y=0.99)
    save_figure(fig, join(save_loc, f'{PREFIX}_{suffix}_r_vs_p_summary'))
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_corr_stats(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]

    if len(x) < 3:
        return {
            'n': len(x),
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'slope': np.nan,
            'intercept': np.nan,
            'x': x,
            'y': y,
        }

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    fit = linregress(x, y)

    return {
        'n': len(x),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'slope': fit.slope,
        'intercept': fit.intercept,
        'x': x,
        'y': y,
    }


def plot_on_axis(ax, x, y, xlabel, ylabel, title):
    stats = compute_corr_stats(x, y)
    x = stats['x']
    y = stats['y']

    if len(x) == 0:
        ax.set_title(f'{title}\nNo valid data')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return stats

    ax.scatter(x, y, alpha=0.8)

    if len(x) >= 2 and np.nanmin(x) != np.nanmax(x) and np.isfinite(stats['slope']):
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xx, stats['intercept'] + stats['slope'] * xx)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f'{title}\n'
        f'r={stats["pearson_r"]:.3f}, p={stats["pearson_p"]:.3g} | '
        f'rho={stats["spearman_rho"]:.3f}, p={stats["spearman_p"]:.3g} | n={stats["n"]}'
    )
    return stats


def resolve_output_dir(first_mouse, first_day):
    if GLOBAL_SAVE_LOC:
        os.makedirs(GLOBAL_SAVE_LOC, exist_ok=True)
        return GLOBAL_SAVE_LOC
    out = SAVE_TEMPLATE.format(first_mouse, first_day)
    os.makedirs(out, exist_ok=True)
    return out


# def load_mouse_data(mouse, day):
#     combined_file = COMBINED_AVERAGES_TEMPLATE.format(mouse, day)
#
#     if not os.path.exists(combined_file):
#         raise FileNotFoundError(f'combined averages file not found for {mouse} {day}: {combined_file}')
#
#     combined = load_pickle(combined_file)
#     per_neuron = combined['per_neuron_mean']
#
#     missing = [k for k in [X_KEY] + TARGET_KEYS if k not in per_neuron]
#     if missing:
#         raise KeyError(
#             f'{mouse} {day}: missing keys in per_neuron_mean: {missing}. '
#             f'Available keys: {list(per_neuron.keys())}'
#         )
#
#     x = np.asarray(per_neuron[X_KEY], dtype=float)
#     metrics = {k: np.asarray(per_neuron[k], dtype=float) for k in TARGET_KEYS}
#
#     n_neurons = len(x)
#     for key, values in metrics.items():
#         if len(values) != n_neurons:
#             raise ValueError(
#                 f'{mouse} {day}: {key} has {len(values)} neurons, but {X_KEY} has {n_neurons}.'
#             )
#
#     return {
#         'mouse': mouse,
#         'day': day,
#         'combined_file': combined_file,
#         'x': x,
#         'metrics': metrics,
#     }

def load_mouse_data(mouse, day):
    combined_file = COMBINED_AVERAGES_TEMPLATE.format(mouse, day)

    if not os.path.exists(combined_file):
        raise FileNotFoundError(f'combined averages file not found for {mouse} {day}: {combined_file}')

    combined = load_pickle(combined_file)
    per_neuron = combined['per_neuron_mean']

    missing = [k for k in [X_KEY] + TARGET_KEYS if k not in per_neuron]
    if missing:
        raise KeyError(
            f'{mouse} {day}: missing keys in per_neuron_mean: {missing}. '
            f'Available keys: {list(per_neuron.keys())}'
        )

    x = np.asarray(per_neuron[X_KEY], dtype=float)
    metrics = {k: np.asarray(per_neuron[k], dtype=float) for k in TARGET_KEYS}

    n_neurons_before_filter = len(x)

    for key, values in metrics.items():
        if len(values) != n_neurons_before_filter:
            raise ValueError(
                f'{mouse} {day}: {key} has {len(values)} neurons, but {X_KEY} has {n_neurons_before_filter}.'
            )

    # Keep only inhibitory CR-opto responses:
    # cr_opto < 0
    inhibitory_mask = np.isfinite(x) & (x < 0)

    x = x[inhibitory_mask]
    metrics = {
        key: values[inhibitory_mask]
        for key, values in metrics.items()
    }

    print(
        f'{mouse} {day}: keeping {len(x)} / {n_neurons_before_filter} neurons '
        f'with {X_KEY} < 0'
    )

    return {
        'mouse': mouse,
        'day': day,
        'combined_file': combined_file,
        'x': x,
        'metrics': metrics,
        'n_neurons_before_filter': n_neurons_before_filter,
        'n_neurons_after_filter': len(x),
    }
def save_figure(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + '.png', dpi=300)
    if SAVE_PDF:
        fig.savefig(base_path + '.pdf')
    plt.close(fig)


def make_per_metric_figures(dataset_list, save_loc):
    n_mice = len(dataset_list)
    fig_width = max(5.2 * n_mice, 5.2)

    for metric_key in TARGET_KEYS:
        fig, axes = plt.subplots(1, n_mice, figsize=(fig_width, 4.8), squeeze=False)
        axes = axes.ravel()

        for ax, ds in zip(axes, dataset_list):
            title = f'{ds["mouse"]} | {ds["day"]}'
            plot_on_axis(
                ax,
                ds['x'],
                ds['metrics'][metric_key],
                xlabel=X_LABEL,
                ylabel=PRETTY_NAMES[metric_key],
                title=title,
            )

        fig.suptitle(f'{PRETTY_NAMES[X_KEY]} vs {PRETTY_NAMES[metric_key]}', fontsize=14, y=0.99)
        save_figure(fig, join(save_loc, f'{PREFIX}_{X_KEY}_vs_{metric_key}_all_mice'))

def make_combined_grid(dataset_list, save_loc):
    n_rows = len(TARGET_KEYS)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.5 * n_rows), squeeze=False)

    for row, metric_key in enumerate(TARGET_KEYS):
        for col, ds in enumerate(dataset_list):
            ax = axes[row, col]
            title = f'{ds["mouse"]} | {ds["day"]}'
            plot_on_axis(
                ax,
                ds['x'],
                ds['metrics'][metric_key],
                xlabel=X_LABEL,
                ylabel=PRETTY_NAMES[metric_key],
                title=title,
            )

    fig.suptitle(f'{PRETTY_NAMES[X_KEY]} correlations across mice', fontsize=16, y=0.99)
    save_figure(fig, join(save_loc, f'{PREFIX}_combined_grid'))


def save_stats(stats_rows, save_loc):
    stats_pkl = join(save_loc, f'{PREFIX}_stats.pkl')
    stats_csv = join(save_loc, f'{PREFIX}_stats.csv')

    with open(stats_pkl, 'wb') as f:
        pickle.dump(stats_rows, f)

    with open(stats_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['mouse', 'day', 'x_metric', 'y_metric', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'slope', 'intercept']
        )
        writer.writeheader()
        writer.writerows(stats_rows)


def main():
    if not MOUSE_DAY_LIST:
        raise ValueError('Fill MOUSE_DAY_LIST in the USER SETTINGS block.')

    dataset_list = []
    for mouse, day in MOUSE_DAY_LIST:
        dataset_list.append(load_mouse_data(mouse, day))

    save_loc = resolve_output_dir(*MOUSE_DAY_LIST[0])

    all_stats = build_stats_rows(dataset_list)

    if MAKE_PER_METRIC_FIGURES:
        make_per_metric_figures(dataset_list, save_loc)

    if MAKE_COMBINED_GRID:
        make_combined_grid(dataset_list, save_loc)

    if MAKE_R_VS_P_SUMMARY:
        make_r_vs_p_summary(
            all_stats,
            save_loc,
            corr_kind=R_VS_P_CORR_KIND,
            annotate=ANNOTATE_R_VS_P_POINTS,
        )

    save_stats(all_stats, save_loc)

    print('Done.')
    print(f'Saved outputs to: {save_loc}')


if __name__ == '__main__':
    main()
