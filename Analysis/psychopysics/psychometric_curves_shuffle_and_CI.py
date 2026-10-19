import os
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
from scipy.stats import wilcoxon, ttest_rel, shapiro, rankdata
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import curve_fit
from scipy.stats import norm

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

LICK_CODES = {3, 5}

# Map condition to actual stimulation frequency (Hz)
FREQUENCY_MAP = {
    6: 0, 5: 15, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5,
    16: 0, 17: 15, 18: 10, 19: 7.5, 20: 5, 21: 3, 22: 1, 23: 0.5,
}

SORTED_UNIQUE_FREQS = sorted({freq for freq in FREQUENCY_MAP.values()})


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def load_mat_file(path):
    """
    Load MonkeyLogic file.
    Returns:
        cues: trial condition codes
        responses: trial response codes
    """
    mat_data = scipy.io.loadmat(path)
    cues = mat_data['TrialRecord'][0][0][7][0]
    responses = mat_data['TrialRecord'][0][0]['TrialErrors'][0]
    return cues, responses


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def safe_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return np.std(values, ddof=1) / np.sqrt(values.size)


def build_trial_history_masks(responses, hit_codes=None):
    """
    Return boolean masks for trials following a hit versus not.
    """
    if hit_codes is None:
        hit_codes = LICK_CODES

    responses = np.asarray(responses)
    after_hit = np.zeros_like(responses, dtype=bool)
    if responses.size > 1:
        after_hit[1:] = np.isin(responses[:-1], list(hit_codes))

    not_after_hit = ~after_hit
    return after_hit, not_after_hit


def fit_sigmoid_curve(x_data, y_data):
    """
    Fit a sigmoid curve.
    Returns a dict with fit results, or None if fitting fails.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if x_data.size < 4:
        return None

    if np.allclose(y_data, y_data[0]):
        return None

    try:
        p0 = [
            float(np.max(y_data)),
            float(np.median(x_data)),
            1.0,
            float(np.min(y_data))
        ]
        lower_bounds = [0.0, float(np.min(x_data)), -10.0, 0.0]
        upper_bounds = [1.0, float(np.max(x_data)), 10.0, 1.0]

        popt, _ = curve_fit(
            sigmoid,
            x_data,
            y_data,
            p0=p0,
            method='trf',
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )

        y_fit_at_points = sigmoid(x_data, *popt)
        residuals = y_data - y_fit_at_points
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        x_fit = np.linspace(np.min(x_data), np.max(x_data), 200)
        y_fit_dense = sigmoid(x_fit, *popt)

        return {
            'popt': popt,
            'x_fit': x_fit,
            'y_fit': y_fit_dense,
            'r_squared': r_squared,
            'threshold': popt[1],
        }

    except (RuntimeError, ValueError):
        return None


def compute_subset_psychometrics(conditions, responses, frequency_map, mask=None):
    """
    Compute lick fractions and sigmoid fit for a trial subset.
    """
    conditions = np.asarray(conditions)
    responses = np.asarray(responses)

    if mask is None:
        mask = np.ones_like(conditions, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.size != conditions.size:
            raise ValueError("Mask length must match number of trials")

    cond_subset = conditions[mask]
    resp_subset = responses[mask]

    response_dict = {freq: {'licked': 0, 'total': 0} for freq in frequency_map.values()}

    for condition, response in zip(cond_subset, resp_subset):
        if condition not in frequency_map:
            continue
        freq = frequency_map[condition]
        response_dict[freq]['total'] += 1
        if response in LICK_CODES:
            response_dict[freq]['licked'] += 1

    valid_freqs = [
        freq for freq in sorted(response_dict.keys())
        if response_dict[freq]['total'] > 0
    ]

    if not valid_freqs:
        return {
            'frequencies': np.array([]),
            'fractions': np.array([]),
            'response_dict': response_dict,
            'fit': None,
            'max_cr_fraction': np.nan,
        }

    x_data = np.array(valid_freqs, dtype=float)
    y_data = np.array([
        response_dict[f]['licked'] / response_dict[f]['total']
        for f in valid_freqs
    ], dtype=float)

    fit_info = fit_sigmoid_curve(x_data, y_data)

    cr_values = [
        1 - (response_dict[f]['licked'] / response_dict[f]['total'])
        for f in valid_freqs
        if f != 0 and response_dict[f]['total'] > 0
    ]
    max_cr_fraction = max(cr_values) if cr_values else np.nan

    return {
        'frequencies': x_data,
        'fractions': y_data,
        'response_dict': response_dict,
        'fit': fit_info,
        'max_cr_fraction': max_cr_fraction,
    }


# ---------------------------------------------------------------------
# Shuffle Analysis
# ---------------------------------------------------------------------

def compute_shuffled_r2s(conditions, responses, frequency_map, n_shuffles=100, rng=None):
    """
    Shuffle trial-level response labels, recompute fractions, fit sigmoid,
    and collect shuffled R^2 values.

    Returns:
        shuffled_r2s: array of length n_shuffles
        shuffled_subsets: list of subset dicts for plotting examples
    """
    conditions = np.asarray(conditions)
    responses = np.asarray(responses)

    if rng is None:
        rng = np.random.default_rng()

    shuffled_r2s = []
    shuffled_subsets = []

    for _ in range(n_shuffles):
        shuffled_responses = rng.permutation(responses)

        shuffled_subset = compute_subset_psychometrics(
            conditions,
            shuffled_responses,
            frequency_map
        )
        shuffled_subsets.append(shuffled_subset)

        if shuffled_subset['fit'] is not None and np.isfinite(shuffled_subset['fit']['r_squared']):
            shuffled_r2s.append(float(shuffled_subset['fit']['r_squared']))
        else:
            shuffled_r2s.append(np.nan)

    return np.asarray(shuffled_r2s, dtype=float), shuffled_subsets


# ---------------------------------------------------------------------
# Bootstrap confidence intervals for psychometric thresholds
# ---------------------------------------------------------------------

def compute_bootstrap_threshold_ci(
    conditions,
    responses,
    frequency_map,
    n_bootstraps=1000,
    confidence_level=95,
    rng=None,
):
    """
    Estimate a percentile bootstrap confidence interval for the psychometric
    detection threshold.

    Trials are resampled with replacement separately within each stimulation
    frequency, preserving the original number of trials at every frequency.

    Returns:
        bootstrap_thresholds: threshold estimate from each bootstrap sample
        ci_lower: lower percentile confidence limit
        ci_upper: upper percentile confidence limit
    """
    conditions = np.asarray(conditions)
    responses = np.asarray(responses)

    if conditions.size != responses.size:
        raise ValueError("Conditions and responses must have the same length")

    if rng is None:
        rng = np.random.default_rng()

    valid_mask = np.isin(conditions, list(frequency_map.keys()))
    conditions = conditions[valid_mask]
    responses = responses[valid_mask]

    indices_by_frequency = {}
    for freq in sorted(set(frequency_map.values())):
        freq_indices = np.array([
            idx for idx, condition in enumerate(conditions)
            if frequency_map.get(condition) == freq
        ], dtype=int)
        if freq_indices.size > 0:
            indices_by_frequency[freq] = freq_indices

    bootstrap_thresholds = np.full(n_bootstraps, np.nan, dtype=float)

    for bootstrap_idx in range(n_bootstraps):
        sampled_indices = []
        for freq_indices in indices_by_frequency.values():
            sampled_indices.extend(
                rng.choice(freq_indices, size=freq_indices.size, replace=True)
            )

        sampled_indices = np.asarray(sampled_indices, dtype=int)
        bootstrap_subset = compute_subset_psychometrics(
            conditions[sampled_indices],
            responses[sampled_indices],
            frequency_map,
        )

        if bootstrap_subset['fit'] is not None:
            threshold = bootstrap_subset['fit']['threshold']
            if threshold is not None and np.isfinite(threshold):
                bootstrap_thresholds[bootstrap_idx] = float(threshold)

    valid_thresholds = bootstrap_thresholds[np.isfinite(bootstrap_thresholds)]
    if valid_thresholds.size == 0:
        return bootstrap_thresholds, np.nan, np.nan

    alpha = (100 - confidence_level) / 2
    ci_lower, ci_upper = np.percentile(
        valid_thresholds,
        [alpha, 100 - alpha],
    )

    return bootstrap_thresholds, float(ci_lower), float(ci_upper)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_bootstrap_thresholds(bootstrap_records, output_dir):
    """
    Plot each mouse's fitted detection threshold with its bootstrap 95% CI.
    """
    valid_records = [
        record for record in bootstrap_records
        if np.isfinite(record['threshold'])
        and np.isfinite(record['ci_lower'])
        and np.isfinite(record['ci_upper'])
    ]

    if not valid_records:
        print("No valid bootstrap threshold confidence intervals available.")
        return

    mouse_labels = [record['mouse'] for record in valid_records]
    thresholds = np.asarray(
        [record['threshold'] for record in valid_records],
        dtype=float,
    )
    ci_lower = np.asarray(
        [record['ci_lower'] for record in valid_records],
        dtype=float,
    )
    ci_upper = np.asarray(
        [record['ci_upper'] for record in valid_records],
        dtype=float,
    )

    x_positions = np.arange(len(valid_records))

    fig_width = max(8, 0.85 * len(valid_records))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Draw percentile confidence intervals independently of the point estimate,
    # so the plot remains valid even if an interval does not contain the
    # original fitted threshold.
    cap_half_width = 0.08
    ax.vlines(x_positions, ci_lower, ci_upper, linewidth=1.5)
    ax.hlines(
        ci_lower,
        x_positions - cap_half_width,
        x_positions + cap_half_width,
        linewidth=1.5,
    )
    ax.hlines(
        ci_upper,
        x_positions - cap_half_width,
        x_positions + cap_half_width,
        linewidth=1.5,
    )
    ax.scatter(x_positions, thresholds, s=55, zorder=3)

    ax.set_xlim(-0.5, len(valid_records) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mouse_labels, rotation=45, ha='right')
    ax.set_xlabel('Mouse')
    ax.set_ylabel('Detection threshold (Hz)')
    ax.set_title('Psychometric detection thresholds with 95% bootstrap CIs')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    upper_limit = max(np.nanmax(ci_upper) * 1.12, 1.0)
    ax.set_ylim(0, upper_limit)

    fig.tight_layout()
    fig.savefig(output_dir / 'detection_threshold_bootstrap_CI.png', dpi=300)
    fig.savefig(output_dir / 'detection_threshold_bootstrap_CI.pdf', dpi=300)
    plt.close(fig)

    output_path = output_dir / 'detection_threshold_bootstrap_CI.txt'
    with open(output_path, 'w') as file:
        file.write(
            'Mouse\tThreshold_Hz\tCI95_lower_Hz\tCI95_upper_Hz'
            '\tN_valid_bootstraps\tN_bootstraps\n'
        )
        for record in bootstrap_records:
            file.write(
                f"{record['mouse']}\t"
                f"{record['threshold']:.6f}\t"
                f"{record['ci_lower']:.6f}\t"
                f"{record['ci_upper']:.6f}\t"
                f"{record['n_valid_bootstraps']}\t"
                f"{record['n_bootstraps']}\n"
            )


def plot_trial_history_example(mouse, subset_all, subset_not_after, subset_after, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    subset_specs = [
        ('All trials', subset_all, '#4D4D4D'),
        ('Not after hit', subset_not_after, '#1f77b4'),
        ('After hit', subset_after, '#d62728'),
    ]

    threshold_handles = []

    for label, subset, clr in subset_specs:
        if subset['frequencies'].size == 0:
            continue

        detection_points = (1 - subset['fractions']) * 100
        ax.scatter(subset['frequencies'], detection_points, color=clr, alpha=0.6)

        if subset['fit'] is not None:
            ax.plot(
                subset['fit']['x_fit'],
                (1 - subset['fit']['y_fit']) * 100,
                color=clr,
                linewidth=2,
                label=f'{label} fit'
            )
            threshold_val = subset['fit']['threshold']
            ax.axvline(threshold_val, color=clr, linestyle='--', linewidth=1.5)
            threshold_handles.append((label, threshold_val, clr))

    for label, thr_val, clr in threshold_handles:
        ax.plot([], [], color=clr, linestyle='--', linewidth=1.5,
                label=f'{label} threshold: {thr_val:.2f} Hz')

    ax.set_xlabel('Optogenetic Stimulation Frequency (Hz)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title(f'{mouse} Psychometric Curves by Trial History')
    ax.set_xticks(SORTED_UNIQUE_FREQS)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_dir / f'{mouse}_trial_history_comparison.png', dpi=300)
    fig.savefig(output_dir / f'{mouse}_trial_history_comparison.pdf', dpi=300)
    plt.close(fig)


def plot_real_vs_shuffled_example(mouse, real_subset, shuffled_subsets, output_dir, n_example_shuffles=20):
    """
    Plot one example mouse with real fit and several shuffled fits.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if real_subset['frequencies'].size > 0:
        real_detection = (1 - real_subset['fractions']) * 100
        ax.scatter(
            real_subset['frequencies'],
            real_detection,
            color='black',
            s=60,
            zorder=5,
            label='Real data'
        )

    plotted_shuffle_label = False
    n_plotted = 0

    for shuffled_subset in shuffled_subsets:
        if n_plotted >= n_example_shuffles:
            break

        if shuffled_subset['fit'] is None:
            continue

        ax.plot(
            shuffled_subset['fit']['x_fit'],
            (1 - shuffled_subset['fit']['y_fit']) * 100,
            color='gray',
            alpha=0.25,
            linewidth=1.5,
            label='Shuffled fits' if not plotted_shuffle_label else None
        )
        plotted_shuffle_label = True
        n_plotted += 1

    if real_subset['fit'] is not None:
        real_r2 = real_subset['fit']['r_squared']
        ax.plot(
            real_subset['fit']['x_fit'],
            (1 - real_subset['fit']['y_fit']) * 100,
            color='red',
            linewidth=3,
            label=f"Real fit (R$^2$ = {real_r2:.3f})"
        )
        ax.axvline(
            real_subset['fit']['threshold'],
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8
        )

    ax.set_xlabel('Optogenetic Stimulation Frequency (Hz)')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title(f'{mouse}: real fit vs shuffled-label fits')
    ax.set_xticks(SORTED_UNIQUE_FREQS)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_dir / f'{mouse}_real_vs_shuffled_example.png', dpi=300)
    fig.savefig(output_dir / f'{mouse}_real_vs_shuffled_example.pdf', dpi=300)
    plt.close(fig)

def cohens_dz_paired(x, y):
    """
    Cohen's dz for paired samples:
    mean(x - y) / sd(x - y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    diffs = diffs[np.isfinite(diffs)]

    if diffs.size < 2:
        return np.nan

    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return np.nan

    return np.mean(diffs) / sd


def rank_biserial_paired(x, y):
    """
    Rank-biserial correlation for paired data, based on signed ranks.
    Appropriate companion effect size for Wilcoxon signed-rank.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    diffs = diffs[np.isfinite(diffs)]

    # remove zero differences, matching Wilcoxon zero_method='wilcox'
    diffs = diffs[diffs != 0]

    if diffs.size == 0:
        return np.nan

    ranks = rankdata(np.abs(diffs))
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    denom = w_pos + w_neg

    if denom == 0:
        return np.nan

    return (w_pos - w_neg) / denom
def plot_r2_summary(r2_records, output_dir):
    """
    Bar graph of average real R^2 across mice vs average shuffled R^2,
    with individual paired dots connected by lines, paired statistical test,
    and matching effect size.
    """
    real_vals = np.array([rec['real_r2'] for rec in r2_records], dtype=float)
    shuffled_vals = np.array([rec['shuffled_mean_r2'] for rec in r2_records], dtype=float)

    valid_mask = np.isfinite(real_vals) & np.isfinite(shuffled_vals)
    real_vals = real_vals[valid_mask]
    shuffled_vals = shuffled_vals[valid_mask]
    valid_records = [rec for rec, keep in zip(r2_records, valid_mask) if keep]

    if real_vals.size == 0:
        print("No valid R^2 values available for summary plot.")
        return

    means = [np.nanmean(real_vals), np.nanmean(shuffled_vals)]
    sems = [safe_sem(real_vals), safe_sem(shuffled_vals)]

    x = np.array([0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        capsize=6,
        color=['black', 'lightgray'],
        edgecolor='black',
        alpha=0.8
    )

    first_label = True
    for rec in valid_records:
        y_pair = [rec['real_r2'], rec['shuffled_mean_r2']]
        ax.plot(
            x,
            y_pair,
            marker='o',
            color='gray',
            alpha=0.7,
            linewidth=1.5,
            label='Individual mice' if first_label else None
        )
        ax.scatter(x, y_pair, color='gray', s=35, zorder=5)
        first_label = False

    # --------------------------------------------------
    # Paired statistical test + effect size
    # --------------------------------------------------
    diffs = real_vals - shuffled_vals
    test_name = None
    stat = np.nan
    p_value = np.nan
    shapiro_p = np.nan
    effect_name = None
    effect_value = np.nan

    if real_vals.size >= 2:
        if 3 <= diffs.size <= 5000 and not np.allclose(diffs, diffs[0]):
            try:
                shapiro_p = shapiro(diffs).pvalue
            except ValueError:
                shapiro_p = np.nan

        if np.isfinite(shapiro_p) and shapiro_p > 0.05:
            test_name = 'paired t-test'
            stat, p_value = ttest_rel(real_vals, shuffled_vals, nan_policy='omit')
            effect_name = "Cohen's d_z"
            effect_value = cohens_dz_paired(real_vals, shuffled_vals)
        else:
            test_name = 'Wilcoxon signed-rank'
            try:
                stat, p_value = wilcoxon(real_vals, shuffled_vals, zero_method='wilcox')
                effect_name = 'rank-biserial r'
                effect_value = rank_biserial_paired(real_vals, shuffled_vals)
            except ValueError:
                test_name = 'paired t-test'
                stat, p_value = ttest_rel(real_vals, shuffled_vals, nan_policy='omit')
                effect_name = "Cohen's d_z"
                effect_value = cohens_dz_paired(real_vals, shuffled_vals)

    # --------------------------------------------------
    # Bracket and annotation
    # --------------------------------------------------
    ymax_data = max(
        np.nanmax(real_vals) if real_vals.size else 0,
        np.nanmax(shuffled_vals) if shuffled_vals.size else 0,
        max(means) if len(means) else 0
    )
    y_bracket = ymax_data + 0.08
    h = 0.03

    ax.plot(
        [x[0], x[0], x[1], x[1]],
        [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
        color='black',
        linewidth=1.5
    )

    if np.isfinite(p_value):
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
    else:
        p_text = 'p = n/a'

    if np.isfinite(effect_value):
        effect_text = f'{effect_name} = {effect_value:.3f}'
    else:
        effect_text = f'{effect_name} = n/a' if effect_name is not None else 'effect = n/a'

    if test_name is not None:
        ax.text(
            np.mean(x),
            y_bracket + h + 0.01,
            f'{test_name}\n{p_text}\n{effect_text}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels(['Real', 'Shuffled'])
    ax.set_ylabel('R$^2$')
    ax.set_title('Real vs shuffled psychometric fit quality across mice')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    upper_ylim = y_bracket + h + 0.16
    current_min = min(
        np.nanmin(real_vals) if real_vals.size else 0,
        np.nanmin(shuffled_vals) if shuffled_vals.size else 0,
        0
    )
    ax.set_ylim(min(0, current_min - 0.02), upper_ylim)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [bars[0], bars[1]] + handles,
        ['Real mean ± SEM', 'Shuffled mean ± SEM'] + labels,
        loc='best'
    )

    stats_lines = [f'n = {real_vals.size}']
    if np.isfinite(shapiro_p):
        stats_lines.append(f'Shapiro p = {shapiro_p:.3f}')
    if test_name is not None:
        stats_lines.append(f'{test_name}: {p_text}')
    if effect_name is not None:
        stats_lines.append(effect_text)

    ax.text(
        0.02,
        0.98,
        '\n'.join(stats_lines),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    fig.tight_layout()
    fig.savefig(output_dir / 'real_vs_shuffled_r2_summary.png', dpi=300)
    fig.savefig(output_dir / 'real_vs_shuffled_r2_summary.pdf', dpi=300)
    plt.close(fig)

    stats_path = output_dir / 'real_vs_shuffled_r2_stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Real vs shuffled R^2 paired statistics\n')
        f.write(f'n_mice\t{real_vals.size}\n')
        f.write(f'mean_real_R2\t{np.nanmean(real_vals):.6f}\n')
        f.write(f'sem_real_R2\t{safe_sem(real_vals):.6f}\n')
        f.write(f'mean_shuffled_R2\t{np.nanmean(shuffled_vals):.6f}\n')
        f.write(f'sem_shuffled_R2\t{safe_sem(shuffled_vals):.6f}\n')
        f.write(f'shapiro_p\t{shapiro_p:.6f}\n' if np.isfinite(shapiro_p) else 'shapiro_p\tnan\n')
        f.write(f'test\t{test_name}\n')
        f.write(f'statistic\t{stat:.6f}\n' if np.isfinite(stat) else 'statistic\tnan\n')
        f.write(f'p_value\t{p_value:.6f}\n' if np.isfinite(p_value) else 'p_value\tnan\n')
        f.write(f'effect_size_name\t{effect_name}\n')
        f.write(f'effect_size\t{effect_value:.6f}\n' if np.isfinite(effect_value) else 'effect_size\tnan\n')

    print('\nR^2 paired statistics:')
    print(f'n = {real_vals.size}')
    if np.isfinite(shapiro_p):
        print(f'Shapiro p = {shapiro_p:.4f}')
    if test_name is not None:
        print(f'{test_name}: statistic = {stat:.4f}, p = {p_value:.6f}')
    if effect_name is not None:
        print(f'{effect_name} = {effect_value:.4f}')


def plot_threshold_summary(threshold_records, output_dir):
    """
    Plot psychometric thresholds for all trials and by trial history.

    Inferential statistics are restricted to the paired, non-overlapping
    trial-history subsets (Not after hit vs After hit). "All trials" is shown
    descriptively because it is composed of those same two subsets.
    """
    threshold_matrix = np.array([
        [rec['all'], rec['not_after'], rec['after']]
        for rec in threshold_records
    ], dtype=float)

    subset_labels = ['All trials', 'Not after hit', 'After hit']
    bar_colors = ['#4D4D4D', '#1f77b4', '#d62728']
    x_positions = np.arange(len(subset_labels))

    fig, ax = plt.subplots(figsize=(8, 6))

    bar_means = np.nanmean(threshold_matrix, axis=0)
    bar_sems = np.array([
        safe_sem(threshold_matrix[:, i])
        for i in range(threshold_matrix.shape[1])
    ])

    bars = ax.bar(
        x_positions,
        bar_means,
        yerr=bar_sems,
        color=bar_colors,
        alpha=0.7,
        capsize=6,
        linewidth=1.2,
        edgecolor='black'
    )

    line_label_added = False
    for rec in threshold_records:
        y_vals = np.array([rec['all'], rec['not_after'], rec['after']], dtype=float)
        if np.all(np.isnan(y_vals)):
            continue

        ax.plot(
            x_positions,
            y_vals,
            marker='o',
            color='#888888',
            alpha=0.6,
            linewidth=1.5,
            label='Individual mice' if not line_label_added else None
        )
        ax.scatter(x_positions, y_vals, color='#888888', s=30, zorder=5)
        line_label_added = True

    # --------------------------------------------------
    # Paired statistics: Not after hit vs After hit
    # --------------------------------------------------
    not_after_vals = threshold_matrix[:, 1]
    after_vals = threshold_matrix[:, 2]
    valid_pair_mask = np.isfinite(not_after_vals) & np.isfinite(after_vals)
    not_after_paired = not_after_vals[valid_pair_mask]
    after_paired = after_vals[valid_pair_mask]

    diffs = after_paired - not_after_paired
    test_name = None
    stat = np.nan
    p_value = np.nan
    shapiro_p = np.nan
    effect_name = None
    effect_value = np.nan

    if diffs.size >= 2:
        if 3 <= diffs.size <= 5000 and not np.allclose(diffs, diffs[0]):
            try:
                shapiro_p = shapiro(diffs).pvalue
            except ValueError:
                shapiro_p = np.nan

        if np.isfinite(shapiro_p) and shapiro_p > 0.05:
            test_name = 'paired t-test'
            stat, p_value = ttest_rel(after_paired, not_after_paired, nan_policy='omit')
            effect_name = "Cohen's d_z"
            effect_value = cohens_dz_paired(after_paired, not_after_paired)
        else:
            test_name = 'Wilcoxon signed-rank'
            try:
                stat, p_value = wilcoxon(
                    after_paired,
                    not_after_paired,
                    zero_method='wilcox'
                )
                effect_name = 'rank-biserial r'
                effect_value = rank_biserial_paired(after_paired, not_after_paired)
            except ValueError:
                test_name = 'paired t-test'
                stat, p_value = ttest_rel(after_paired, not_after_paired, nan_policy='omit')
                effect_name = "Cohen's d_z"
                effect_value = cohens_dz_paired(after_paired, not_after_paired)

    if np.isfinite(p_value):
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
    else:
        p_text = 'p = n/a'

    if np.isfinite(effect_value):
        effect_text = f'{effect_name} = {effect_value:.3f}'
    else:
        effect_text = (
            f'{effect_name} = n/a'
            if effect_name is not None
            else 'effect = n/a'
        )

    # --------------------------------------------------
    # Bracket and annotation
    # --------------------------------------------------
    finite_thresholds = threshold_matrix[np.isfinite(threshold_matrix)]
    ymax_data = np.max(finite_thresholds) if finite_thresholds.size else 1.0
    y_range = max(ymax_data, 1.0)
    y_bracket = ymax_data + 0.08 * y_range
    h = 0.035 * y_range

    if test_name is not None:
        x1, x2 = x_positions[1], x_positions[2]
        ax.plot(
            [x1, x1, x2, x2],
            [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
            color='black',
            linewidth=1.5
        )
        ax.text(
            (x1 + x2) / 2,
            y_bracket + h + 0.015 * y_range,
            f'{test_name}\n{p_text}\n{effect_text}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(subset_labels)
    ax.set_ylabel('Threshold (Hz)')
    ax.set_title('Psychometric Thresholds by Trial History')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    upper_ylim = y_bracket + h + 0.24 * y_range if test_name is not None else 1.2 * y_range
    ax.set_ylim(0, max(1.0, upper_ylim))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [bars[0], bars[1], bars[2]] + handles,
        [f'{subset_labels[i]} mean ± SEM' for i in range(len(subset_labels))] + labels,
        loc='best'
    )

    stats_lines = [f'paired n = {diffs.size}']
    if np.isfinite(shapiro_p):
        stats_lines.append(f'Shapiro p = {shapiro_p:.3f}')
    if test_name is not None:
        stats_lines.append(f'{test_name}: {p_text}')
    if effect_name is not None:
        stats_lines.append(effect_text)

    ax.text(
        0.02,
        0.98,
        '\n'.join(stats_lines),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    fig.tight_layout()
    fig.savefig(output_dir / 'threshold_comparison_trial_history.png', dpi=300)
    fig.savefig(output_dir / 'threshold_comparison_trial_history.pdf', dpi=300)
    plt.close(fig)

    stats_path = output_dir / 'threshold_comparison_trial_history_stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Psychometric threshold paired statistics\n')
        f.write('Comparison\tAfter hit vs Not after hit\n')
        f.write(f'n_paired_mice\t{diffs.size}\n')
        f.write(f'mean_not_after_Hz\t{np.nanmean(not_after_paired):.6f}\n' if not_after_paired.size else 'mean_not_after_Hz\tnan\n')
        f.write(f'sem_not_after_Hz\t{safe_sem(not_after_paired):.6f}\n' if not_after_paired.size else 'sem_not_after_Hz\tnan\n')
        f.write(f'mean_after_Hz\t{np.nanmean(after_paired):.6f}\n' if after_paired.size else 'mean_after_Hz\tnan\n')
        f.write(f'sem_after_Hz\t{safe_sem(after_paired):.6f}\n' if after_paired.size else 'sem_after_Hz\tnan\n')
        f.write(f'mean_difference_after_minus_not_after_Hz\t{np.nanmean(diffs):.6f}\n' if diffs.size else 'mean_difference_after_minus_not_after_Hz\tnan\n')
        f.write(f'shapiro_p\t{shapiro_p:.6f}\n' if np.isfinite(shapiro_p) else 'shapiro_p\tnan\n')
        f.write(f'test\t{test_name}\n')
        f.write(f'statistic\t{stat:.6f}\n' if np.isfinite(stat) else 'statistic\tnan\n')
        f.write(f'p_value\t{p_value:.6f}\n' if np.isfinite(p_value) else 'p_value\tnan\n')
        f.write(f'effect_size_name\t{effect_name}\n')
        f.write(f'effect_size\t{effect_value:.6f}\n' if np.isfinite(effect_value) else 'effect_size\tnan\n')

    print('\nThreshold paired statistics:')
    print(f'n = {diffs.size}')
    if np.isfinite(shapiro_p):
        print(f'Shapiro p = {shapiro_p:.4f}')
    if test_name is not None:
        print(f'{test_name}: statistic = {stat:.4f}, p = {p_value:.6f}')
    if effect_name is not None:
        print(f'{effect_name} = {effect_value:.4f}')


# ---------------------------------------------------------------------
# Path logic
# ---------------------------------------------------------------------

def determine_path(mouse_id, day_value):
    if mouse_id in {'OI140', 'OI136', 'OI137'}:
        return Path(
            r'\\isi.storwis.weizmann.ac.il\Labs\livneh\omerizh\OI136-140_OXTR_detection_paradigm\detection paradigm\water restricted\4 - psy\{}'.format(mouse_id)
        )
    if mouse_id in {'OI168', 'OI192'}:
        return Path(
            r'\\isi.storwis.weizmann.ac.il\Labs\livneh\omerizh\OI165-169_192_194-195 - OXTR females\detection + psychophysics\psychophysics\{}'.format(mouse_id)
        )
    return Path(r'Z:\2p_data\{}\{}\ML'.format(mouse_id, day_value))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == '__main__':
    colors = [
        '#FFEAEA',
        '#FFD2D2',
        '#FFBBBB',
        '#FFA3A3',
        '#FF8B8B',
        '#FF7474',
        '#FF5C5C',
        '#FF4545',
        '#E03B3B',
        '#C73232'
    ]

    mice = ['OI208', 'OI116', 'OI124', 'OI151', 'OI152', 'OI136', 'OI137', 'OI140', 'OI168', 'OI192']
    day = ['250227', '240529', '240610', '240827', '241003', '', '', '', '', '']

    n_priming_trials = 23
    n_shuffles = 100
    n_bootstraps = 1000
    example_mouse = 'OI151'
    rng = np.random.default_rng(42)
    bootstrap_rng = np.random.default_rng(43)

    base_folder = Path(r'Z:\Stavsht\Omer_OXTR')
    master_output_dir = base_folder / 'psychometric_curve_trial_history_v3_with_shuffle_and_CI'
    master_output_dir.mkdir(parents=True, exist_ok=True)

    example_dir = master_output_dir / 'example_plots'
    example_dir.mkdir(exist_ok=True)

    aggregate_fig, aggregate_ax = plt.subplots(figsize=(20, 10))

    threshold_records = []
    max_cr_records = []
    priming_metrics = []
    r2_records = []
    bootstrap_threshold_records = []

    example_plotted = False

    for mouse, day_value, color in zip(mice, day, colors):
        path_ml = determine_path(mouse, day_value)
        conditions_played = []
        responses = []

        for mat_file in sorted(path_ml.glob('*.mat')):
            cond_run, resp_run = load_mat_file(mat_file)
            conditions_played.extend(cond_run)
            responses.extend(resp_run)

        conditions_played = np.asarray(conditions_played)
        responses = np.asarray(responses)

        if conditions_played.size == 0 or responses.size == 0:
            print(f"Skipping {mouse}: no trials available.")
            continue

        # -------------------------------------------------------------
        # Priming metrics
        # -------------------------------------------------------------
        priming_slice = slice(0, min(n_priming_trials, conditions_played.size))
        priming_responses = responses[priming_slice]

        priming_total = priming_responses.size
        priming_hits = np.isin(priming_responses, list(LICK_CODES)).sum()
        priming_cr = np.isin(priming_responses, [4]).sum()

        priming_hit_rate = priming_hits / priming_total if priming_total > 0 else np.nan
        priming_cr_rate = priming_cr / priming_total if priming_total > 0 else np.nan

        epsilon = 1e-3
        hit_clamped = np.clip(priming_hit_rate, epsilon, 1 - epsilon)
        cr_clamped = np.clip(priming_cr_rate, epsilon, 1 - epsilon)
        priming_dprime = norm.ppf(hit_clamped) - norm.ppf(1 - cr_clamped)

        priming_metrics.append({
            'mouse': mouse,
            'hit_rate': priming_hit_rate,
            'cr_rate': priming_cr_rate,
            'dprime': priming_dprime,
        })

        # Remove priming trials
        conditions_played = conditions_played[n_priming_trials:]
        responses = responses[n_priming_trials:]

        if conditions_played.size == 0 or responses.size == 0:
            print(f"Skipping {mouse}: no trials after removing priming.")
            continue

        # -------------------------------------------------------------
        # Trial-history subsets
        # -------------------------------------------------------------
        after_hit_mask, not_after_hit_mask = build_trial_history_masks(responses)

        subset_all = compute_subset_psychometrics(conditions_played, responses, FREQUENCY_MAP)
        subset_not_after = compute_subset_psychometrics(conditions_played, responses, FREQUENCY_MAP, mask=not_after_hit_mask)
        subset_after = compute_subset_psychometrics(conditions_played, responses, FREQUENCY_MAP, mask=after_hit_mask)

        # -------------------------------------------------------------
        # Bootstrap confidence interval for the detection threshold
        # -------------------------------------------------------------
        bootstrap_thresholds, ci_lower, ci_upper = compute_bootstrap_threshold_ci(
            conditions_played,
            responses,
            FREQUENCY_MAP,
            n_bootstraps=n_bootstraps,
            confidence_level=95,
            rng=bootstrap_rng,
        )

        real_threshold = np.nan
        if subset_all['fit'] is not None and subset_all['fit']['threshold'] is not None:
            real_threshold = float(subset_all['fit']['threshold'])

        bootstrap_threshold_records.append({
            'mouse': mouse,
            'threshold': real_threshold,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_valid_bootstraps': int(np.isfinite(bootstrap_thresholds).sum()),
            'n_bootstraps': n_bootstraps,
        })

        # -------------------------------------------------------------
        # Shuffle R^2
        # -------------------------------------------------------------
        shuffled_r2s, shuffled_subsets = compute_shuffled_r2s(
            conditions_played,
            responses,
            FREQUENCY_MAP,
            n_shuffles=n_shuffles,
            rng=rng
        )

        real_r2 = np.nan
        if subset_all['fit'] is not None and np.isfinite(subset_all['fit']['r_squared']):
            real_r2 = float(subset_all['fit']['r_squared'])

        shuffled_mean_r2 = np.nanmean(shuffled_r2s) if np.isfinite(shuffled_r2s).any() else np.nan

        r2_records.append({
            'mouse': mouse,
            'real_r2': real_r2,
            'shuffled_mean_r2': shuffled_mean_r2,
            'shuffled_r2s': shuffled_r2s.copy(),
        })

        # -------------------------------------------------------------
        # Threshold records
        # -------------------------------------------------------------
        def extract_threshold(subset):
            if subset['fit'] is not None and subset['fit']['threshold'] is not None:
                return float(subset['fit']['threshold'])
            return np.nan

        thresh_all = extract_threshold(subset_all)
        thresh_not_after = extract_threshold(subset_not_after)
        thresh_after = extract_threshold(subset_after)

        threshold_records.append({
            'mouse': mouse,
            'all': thresh_all,
            'not_after': thresh_not_after,
            'after': thresh_after,
        })

        max_cr_records.append({
            'mouse': mouse,
            'max_cr': subset_all['max_cr_fraction'],
        })

        # -------------------------------------------------------------
        # Aggregate psychometric plot
        # -------------------------------------------------------------
        if subset_all['frequencies'].size > 0:
            detection_pct = (1 - subset_all['fractions']) * 100
            aggregate_ax.scatter(subset_all['frequencies'], detection_pct, color=color, alpha=0.5)

            if subset_all['fit'] is not None:
                aggregate_ax.plot(
                    subset_all['fit']['x_fit'],
                    (1 - subset_all['fit']['y_fit']) * 100,
                    color=color,
                    linewidth=2,
                    label=mouse,
                )
                aggregate_ax.axvline(
                    subset_all['fit']['threshold'],
                    color=color,
                    linestyle='--',
                    alpha=0.3
                )
            else:
                aggregate_ax.plot(
                    subset_all['frequencies'],
                    detection_pct,
                    color=color,
                    linewidth=2,
                    label=mouse
                )

        # -------------------------------------------------------------
        # Example plots
        # -------------------------------------------------------------
        if mouse == example_mouse:
            plot_trial_history_example(
                mouse=mouse,
                subset_all=subset_all,
                subset_not_after=subset_not_after,
                subset_after=subset_after,
                output_dir=example_dir
            )

            plot_real_vs_shuffled_example(
                mouse=mouse,
                real_subset=subset_all,
                shuffled_subsets=shuffled_subsets,
                output_dir=example_dir,
                n_example_shuffles=20
            )
            example_plotted = True

    # If requested example mouse was not available, use first valid mouse
    if not example_plotted and len(r2_records) > 0:
        print(f"Example mouse {example_mouse} not plotted. You can set another example_mouse manually.")

    # -----------------------------------------------------------------
    # Finalize aggregate plot
    # -----------------------------------------------------------------
    aggregate_ax.set_xlabel('Optogenetic Stimulation Frequency (Hz)')
    aggregate_ax.set_ylabel('Detection Rate (%)')
    aggregate_ax.set_xticks(SORTED_UNIQUE_FREQS)
    aggregate_ax.set_ylim(0, 105)
    aggregate_ax.grid(axis='y', linestyle='--', alpha=0.3)
    aggregate_ax.legend(loc='lower right')

    valid_real_r2 = np.array(
        [rec['real_r2'] for rec in r2_records if np.isfinite(rec['real_r2'])],
        dtype=float
    )
    if valid_real_r2.size > 0:
        avg_r2 = np.nanmean(valid_real_r2)
        sem_r2 = safe_sem(valid_real_r2)
        aggregate_ax.text(
            0.05,
            0.95,
            f'Avg real R$^2$: {avg_r2:.2f} ± {sem_r2:.2f}',
            transform=aggregate_ax.transAxes,
            fontsize=12,
            va='top',
            bbox=dict(facecolor='white', alpha=0.5),
        )

    aggregate_fig.tight_layout()
    aggregate_fig.savefig(master_output_dir / 'Psychophysics_Plot_All.png', dpi=300)
    aggregate_fig.savefig(master_output_dir / 'Psychophysics_Plot_All.pdf', dpi=300)
    plt.close(aggregate_fig)

    # -----------------------------------------------------------------
    # Summary plots
    # -----------------------------------------------------------------
    plot_threshold_summary(threshold_records, master_output_dir)
    plot_r2_summary(r2_records, master_output_dir)
    plot_bootstrap_thresholds(bootstrap_threshold_records, master_output_dir)

    # -----------------------------------------------------------------
    # Save text outputs
    # -----------------------------------------------------------------
    max_cr_path = master_output_dir / 'max_cr_percentages.txt'
    with open(max_cr_path, 'w') as f:
        f.write('Mouse\tMax CR (%)\n')
        for record in max_cr_records:
            max_cr = record['max_cr']
            percent = max_cr * 100 if max_cr is not None and not np.isnan(max_cr) else float('nan')
            print(f"{record['mouse']}: Max CR = {percent:.2f}%")
            f.write(f"{record['mouse']}\t{percent:.2f}\n")

    priming_path = master_output_dir / 'priming_trial_metrics.txt'
    with open(priming_path, 'w') as f:
        f.write("Mouse\tPriming Hit (%)\tPriming CR (%)\tPriming d'\n")
        for record in priming_metrics:
            hit_pct = record['hit_rate'] * 100 if record['hit_rate'] is not None and not np.isnan(record['hit_rate']) else float('nan')
            cr_pct = record['cr_rate'] * 100 if record['cr_rate'] is not None and not np.isnan(record['cr_rate']) else float('nan')
            dprime_val = record['dprime'] if record['dprime'] is not None and not np.isnan(record['dprime']) else float('nan')
            f.write(f"{record['mouse']}\t{hit_pct:.2f}\t{cr_pct:.2f}\t{dprime_val:.2f}\n")

    r2_out_path = master_output_dir / 'real_vs_shuffled_r2_per_mouse.txt'
    with open(r2_out_path, 'w') as f:
        f.write('Mouse\tReal_R2\tMean_Shuffled_R2\tStd_Shuffled_R2\tN_Valid_Shuffles\n')
        for rec in r2_records:
            shuffled_valid = rec['shuffled_r2s'][np.isfinite(rec['shuffled_r2s'])]
            shuffled_std = np.std(shuffled_valid, ddof=1) if shuffled_valid.size > 1 else np.nan
            f.write(
                f"{rec['mouse']}\t"
                f"{rec['real_r2']:.6f}\t"
                f"{rec['shuffled_mean_r2']:.6f}\t"
                f"{shuffled_std:.6f}\t"
                f"{shuffled_valid.size}\n"
            )

    print("Done.")