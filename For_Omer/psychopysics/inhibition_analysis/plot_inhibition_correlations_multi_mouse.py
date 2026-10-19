#!/usr/bin/env python3
"""
PyCharm-friendly standalone script to plot inhibition-rank correlations across multiple mice.

What it does
------------
For each mouse/day pair, loads:
- reward_related_averages.pkl
- ranked_neurons_by_inhibition.npy (or another inhibition file)

Then creates:
1) One figure per metric with one subplot per mouse
   - odor_to_prelick
   - reward_0_2s
   - swallow_1p5_2s
2) One combined summary figure with rows = metrics, columns = mice
3) A CSV + PKL with correlation stats for all mice and all metrics

Edit only the USER SETTINGS block and run in PyCharm.
"""

import os
import csv
import math
import pickle
from os.path import join, splitext

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

REWARD_RELATED_TEMPLATE = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis\reward_related_averages.pkl"
INHIBITION_TEMPLATE = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis\ranked_neurons_by_inhibition.npy"
SAVE_TEMPLATE = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis"

# Optional central output folder. If empty, figures are saved under the first mouse/day SAVE_TEMPLATE folder
GLOBAL_SAVE_LOC = r'Z:\Stavsht\Omer_OXTR\inhibition_correlation_summary'

PREFIX = 'inhibition_corr_multi_mouse'
X_LABEL = 'Inhibition rank in CR opto window'
INPUT_KIND = 'sort_idx'   # 'values' or 'sort_idx'
PKL_KEY = None            # If inhibition file is a dict in pkl
CSV_COLUMN = None         # If inhibition file is csv with multiple numeric columns

MAKE_PER_METRIC_FIGURES = True
MAKE_COMBINED_GRID = True
SAVE_PDF = True
# =========================================================


PRETTY_NAMES = {
    'odor_to_prelick': 'Odor onset to 100 ms before first lick',
    'reward_0_2s': '0 to 2 s after reward',
    'swallow_1p5_2s': '1.5 to 2 s after reward',
}

METRIC_ORDER = ['odor_to_prelick', 'reward_0_2s', 'swallow_1p5_2s']


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def flatten_to_1d_numeric(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float).ravel()


def load_numeric_vector(path, pkl_key=None, csv_column=None):
    ext = splitext(path)[1].lower()

    if ext == '.npy':
        return flatten_to_1d_numeric(np.load(path, allow_pickle=True))

    if ext == '.pkl':
        obj = load_pickle(path)
        if isinstance(obj, dict):
            if pkl_key is None:
                numeric_keys = [k for k, v in obj.items() if np.asarray(v).ndim >= 1]
                raise ValueError(f'{path} contains a dict. Set PKL_KEY. Candidate keys: {numeric_keys}')
            if pkl_key not in obj:
                raise KeyError(f'Key {pkl_key!r} not found in {path}')
            return flatten_to_1d_numeric(obj[pkl_key])
        return flatten_to_1d_numeric(obj)

    if ext in {'.csv', '.txt'}:
        try:
            data = np.loadtxt(path, delimiter=',')
            return flatten_to_1d_numeric(data)
        except Exception:
            import pandas as pd
            df = pd.read_csv(path)
            if csv_column is None:
                numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                if len(numeric_cols) == 1:
                    return flatten_to_1d_numeric(df[numeric_cols[0]].to_numpy())
                raise ValueError(f'{path} has multiple numeric columns. Set CSV_COLUMN. Numeric columns: {numeric_cols}')
            if csv_column not in df.columns:
                raise KeyError(f'Column {csv_column!r} not found in {path}')
            return flatten_to_1d_numeric(df[csv_column].to_numpy())

    raise ValueError(f'Unsupported file type: {ext}')


def build_rank_array_from_sort_idx(sort_idx):
    """
    sort_idx is assumed to be ordered from most inhibited to least inhibited.
    We convert it so larger rank = stronger inhibition.
    """
    sort_idx = np.asarray(sort_idx, dtype=int)
    n = len(sort_idx)
    ranks = np.empty(n, dtype=float)
    ranks[sort_idx] = np.arange(n, 0, -1, dtype=float)
    return ranks


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


def load_mouse_data(mouse, day):
    reward_related_file = REWARD_RELATED_TEMPLATE.format(mouse, day)
    inhibition_file = INHIBITION_TEMPLATE.format(mouse, day)

    if not os.path.exists(reward_related_file):
        raise FileNotFoundError(f'reward-related file not found for {mouse} {day}: {reward_related_file}')
    if not os.path.exists(inhibition_file):
        raise FileNotFoundError(f'inhibition file not found for {mouse} {day}: {inhibition_file}')

    reward_related = load_pickle(reward_related_file)
    per_neuron = reward_related['per_neuron_mean']

    inhibition_values = load_numeric_vector(inhibition_file, pkl_key=PKL_KEY, csv_column=CSV_COLUMN)
    if INPUT_KIND == 'sort_idx':
        inhibition_values = build_rank_array_from_sort_idx(inhibition_values)

    metrics = {
        'odor_to_prelick': np.asarray(per_neuron['odor_to_prelick'], dtype=float),
        'reward_0_2s': np.asarray(per_neuron['reward_0_2s'], dtype=float),
        'swallow_1p5_2s': np.asarray(per_neuron['swallow_1p5_2s'], dtype=float),
    }

    n_neurons = len(next(iter(metrics.values())))
    if len(inhibition_values) != n_neurons:
        raise ValueError(
            f'{mouse} {day}: inhibition vector has {len(inhibition_values)} values, '
            f'but reward-related averages have {n_neurons} neurons.'
        )

    return {
        'mouse': mouse,
        'day': day,
        'reward_related_file': reward_related_file,
        'inhibition_file': inhibition_file,
        'x': inhibition_values,
        'metrics': metrics,
    }


def save_figure(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + '.png', dpi=300)
    if SAVE_PDF:
        fig.savefig(base_path + '.pdf')
    plt.close(fig)


def make_per_metric_figures(dataset_list, save_loc):
    stats_rows = []
    n_mice = len(dataset_list)

    for metric_key in METRIC_ORDER:
        fig, axes = plt.subplots(1, n_mice, figsize=(5.2 * n_mice, 5.5), squeeze=False)
        axes = axes.ravel()

        for ax, ds in zip(axes, dataset_list):
            title = f'{ds["mouse"]} | {ds["day"]}'
            stats = plot_on_axis(
                ax,
                ds['x'],
                ds['metrics'][metric_key],
                xlabel=X_LABEL,
                ylabel=PRETTY_NAMES[metric_key],
                title=title,
            )
            stats_rows.append({
                'mouse': ds['mouse'],
                'day': ds['day'],
                'metric': metric_key,
                **{k: v for k, v in stats.items() if k not in {'x', 'y'}},
            })

        fig.suptitle(PRETTY_NAMES[metric_key], fontsize=14, y=0.99)
        save_figure(fig, join(save_loc, f'{PREFIX}_{metric_key}_all_mice'))

    return stats_rows


def make_combined_grid(dataset_list, save_loc):
    n_rows = len(METRIC_ORDER)
    n_cols = len(dataset_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.5 * n_rows), squeeze=False)

    for row, metric_key in enumerate(METRIC_ORDER):
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

    fig.suptitle('Inhibition rank correlations across mice', fontsize=16, y=1.01)
    save_figure(fig, join(save_loc, f'{PREFIX}_combined_grid'))


def save_stats(stats_rows, save_loc):
    stats_pkl = join(save_loc, f'{PREFIX}_stats.pkl')
    stats_csv = join(save_loc, f'{PREFIX}_stats.csv')

    with open(stats_pkl, 'wb') as f:
        pickle.dump(stats_rows, f)

    with open(stats_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['mouse', 'day', 'metric', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'slope', 'intercept']
        )
        writer.writeheader()
        writer.writerows(stats_rows)


def main():
    if INPUT_KIND not in {'values', 'sort_idx'}:
        raise ValueError("INPUT_KIND must be 'values' or 'sort_idx'.")
    if not MOUSE_DAY_LIST:
        raise ValueError('Fill MOUSE_DAY_LIST in the USER SETTINGS block.')

    dataset_list = []
    for mouse, day in MOUSE_DAY_LIST:
        dataset_list.append(load_mouse_data(mouse, day))

    save_loc = resolve_output_dir(*MOUSE_DAY_LIST[0])

    all_stats = []
    if MAKE_PER_METRIC_FIGURES:
        all_stats.extend(make_per_metric_figures(dataset_list, save_loc))

    if MAKE_COMBINED_GRID:
        make_combined_grid(dataset_list, save_loc)

    if all_stats:
        save_stats(all_stats, save_loc)

    print('Done.')
    print(f'Saved outputs to: {save_loc}')


if __name__ == '__main__':
    main()
