import os
import pickle
import random
from os.path import join

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde, pearsonr
import scipy.stats


matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42


###############################################################################
# Helper classes
###############################################################################
class NumpyCoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


###############################################################################
# General helpers
###############################################################################
def sanitize_indices(indices, n_neurons):
    idx = np.asarray(indices).astype(int).ravel()
    idx = idx[(idx >= 0) & (idx < n_neurons)]
    return np.unique(idx)


def baseline_subtract_neurons(norm_data, pre_trial_length, baseline_time):
    import copy
    concat_data = copy.deepcopy(norm_data)
    j = 0
    for trial in concat_data:
        i = 0
        for neuron in trial:
            neuron = np.asarray(neuron)
            baseline = np.mean(neuron[pre_trial_length - baseline_time:pre_trial_length])
            sub_neuron = neuron - baseline
            concat_data[j][i] = sub_neuron
            i += 1
        j += 1
    return concat_data


def add_axvline_opto_seconds(ax, pre_trial_length, rec_hz):
    ax.axvline(x=0, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    ax.axvline(x=1, color='darkorange', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='yellow', linestyle='--', linewidth=1, label='Opto stimulation')
    ax.axvline(x=5, color='yellow', linestyle='--', linewidth=1)
    ax.axvline(x=5, color='blue', linestyle='--', linewidth=1, label='Odor')
    ax.axvline(x=7, color='blue', linestyle='--', linewidth=1)


###############################################################################
# CP histogram
###############################################################################
def choice_probability_hist(choice_prob, save_loc, type, sig_neuron=None):
    """
    Histogram in symmetrized CP' space for a subset of neurons.
    """
    if sig_neuron is None:
        sig_neuron = []

    distribution_key = 'all_distributions'

    n_neurons = len(choice_prob[distribution_key])
    medians = np.array([
        np.nanmean(choice_prob[distribution_key][i])
        for i in range(n_neurons)
    ])

    sig_neuron = np.asarray(sig_neuron).astype(int)
    medians = medians[sig_neuron]

    valid_mask = ~np.isnan(medians)
    medians_clean = medians[valid_mask]

    cp_sym = 2 * (medians_clean - 0.5)

    cp_exc = cp_sym[cp_sym >= 0]
    cp_inh = cp_sym[cp_sym < 0]

    overall_median = np.nanmedian(cp_sym)
    overall_mean = np.nanmean(cp_sym)

    plt.figure(figsize=(6, 4))
    bin_width = 0.02
    bins = np.arange(-1.0, 1.0 + bin_width, bin_width)

    if len(cp_exc) > 0:
        plt.hist(cp_exc, bins=bins, facecolor='salmon',
                 edgecolor='black', alpha=0.8, label='Excitation')

    if len(cp_inh) > 0:
        plt.hist(cp_inh, bins=bins, facecolor='skyblue',
                 edgecolor='black', alpha=0.8, label='Inhibition')

    plt.axvline(overall_median, color='black', linestyle='--', label='Median')
    plt.axvline(overall_mean, color='gray', linestyle='--', label='Mean')

    plt.title(f"Distribution of Symmetrized Choice Probability ({type})")
    plt.xlabel("CP′  (−1 to 1, 0 = chance)")
    plt.ylabel("Count")
    plt.xlim(-1, 1)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_loc, exist_ok=True)
    out_name_pdf = os.path.join(save_loc, f"choice_prob_distribution_sym_{type}.pdf")
    out_name_png = os.path.join(save_loc, f"choice_prob_distribution_sym_{type}.png")
    plt.savefig(out_name_pdf, dpi=300)
    plt.savefig(out_name_png, dpi=300)
    plt.close()
    print(f"Saved histogram to {out_name_pdf}")

    np.save(os.path.join(save_loc, f'overall_median_CP_{type}.npy'), overall_median)
    np.save(os.path.join(save_loc, f'overall_mean_CP_{type}.npy'), overall_mean)

    sig_choice = medians_clean
    total_sig = len(sig_choice)

    if total_sig == 0:
        return {}

    bin_inh_1 = np.sum((sig_choice >= 0.4) & (sig_choice < 0.5))
    bin_inh_2 = np.sum((sig_choice >= 0.3) & (sig_choice < 0.4))
    bin_inh_3 = np.sum((sig_choice >= 0.2) & (sig_choice < 0.3))
    bin_inh_4 = np.sum(sig_choice < 0.2)

    bin_exc_1 = np.sum((sig_choice >= 0.5) & (sig_choice < 0.6))
    bin_exc_2 = np.sum((sig_choice >= 0.6) & (sig_choice < 0.7))
    bin_exc_3 = np.sum((sig_choice >= 0.7) & (sig_choice < 0.8))
    bin_exc_4 = np.sum(sig_choice >= 0.8)

    bins_dict = {
        "0.5-0.4": bin_inh_1,
        "0.5-0.6": bin_exc_1,
        "0.4-0.3": bin_inh_2,
        "0.6-0.7": bin_exc_2,
        "0.3-0.2": bin_inh_3,
        "0.7-0.8": bin_exc_3,
        "<0.2": bin_inh_4,
        ">0.8": bin_exc_4
    }

    percents = {k: (v / total_sig * 100) for k, v in bins_dict.items()}

    bins_df = pd.DataFrame(list(percents.items()), columns=["bin", "percent"])
    out_csv = os.path.join(save_loc, f"choice_prob_bins_{type}.csv")
    bins_df.to_csv(out_csv, index=False)
    print(f"Saved choice probability bin percentages to {out_csv}")

    return percents


###############################################################################
# Mean activity helpers
###############################################################################
def compute_opto_windows_mean_activity(
        all_CR,
        all_FA,
        n_fa,
        n_boot=1000,
        random_state=42
):
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_CR)
    all_distributions = [None] * n_neurons

    mean_means = np.full(n_neurons, np.nan)
    sem_means = np.full(n_neurons, np.nan)
    FA_means = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        arr_2d = all_CR[i]
        if i >= len(all_FA):
            continue

        FA_vals = all_FA[i]
        windowA = arr_2d[:, 0]
        wA_vals = windowA
        FA_data = FA_vals[:, 0]

        FA_means[i] = np.mean(FA_data)
        n_avail = len(wA_vals)

        if n_avail < n_fa:
            mean_means[i] = np.mean(wA_vals)
        else:
            dist_means = np.zeros(n_boot, dtype=float)
            for b in range(n_boot):
                sub_idxs = random.sample(range(n_avail), k=n_fa)
                wA_sub = wA_vals[sub_idxs]
                dist_means[b] = np.mean(wA_sub)

            all_distributions[i] = dist_means
            mean_means[i] = np.mean(dist_means)
            sem_means[i] = np.std(dist_means) / np.sqrt(n_boot)

    results = {
        'all_distributions': all_distributions,
        'mean': mean_means,
        'sem': sem_means,
        'FA_means': FA_means
    }
    return results


###############################################################################
# Representative neuron plotting
###############################################################################
def plot_neurons_by_index(
        dff_run_list_sub,
        cp_med,
        load_loc,
        save_loc,
        rec_hz,
        pre_trial_length,
        neuron_indices,
        tag='selected'
):
    """
    Plot CR vs FA mean ± SEM for specific neurons given by indices.
    """
    os.makedirs(os.path.join(save_loc, "rep_neurons_cp", tag), exist_ok=True)
    out_dir = os.path.join(save_loc, "rep_neurons_cp", tag)

    cp_med = np.asarray(cp_med)

    with open(os.path.join(load_loc, 'correct_reject.pkl'), 'rb') as f:
        correct_reject_trials = pickle.load(f)
    with open(os.path.join(load_loc, 'false_alarm.pkl'), 'rb') as f:
        fa_trials = pickle.load(f)
    with open(os.path.join(load_loc, 'hit.pkl'), 'rb') as f:
        hit_trials = pickle.load(f)

    hit_set = set(hit_trials)

    fa_after_hit_idx = []
    fa_not_after_hit_idx = []
    cr_after_hit_idx = []
    cr_not_after_hit_idx = []

    for idx in fa_trials:
        if idx - 1 in hit_set:
            fa_after_hit_idx.append(idx)
        else:
            fa_not_after_hit_idx.append(idx)

    for idx in correct_reject_trials:
        if idx - 1 in hit_set:
            cr_after_hit_idx.append(idx)
        else:
            cr_not_after_hit_idx.append(idx)

    def mean_sem_over_trials(neuron_idx, trial_inds):
        trs = [dff_run_list_sub[neuron_idx][j] for j in trial_inds if j < len(dff_run_list_sub[neuron_idx])]
        if len(trs) == 0:
            return None, None, None
        df = pd.DataFrame([pd.Series(np.asarray(t).astype(float)) for t in trs])
        m = df.mean(axis=0, skipna=True).values
        s = df.sem(axis=0, skipna=True).values
        T = len(m)
        t_sec = (np.arange(T) - pre_trial_length) / rec_hz
        return t_sec, m, s

    neuron_indices = list(neuron_indices)
    for i in neuron_indices:
        t_cr, m_cr, s_cr = mean_sem_over_trials(i, cr_not_after_hit_idx)
        t_fa, m_fa, s_fa = mean_sem_over_trials(i, fa_not_after_hit_idx)

        if t_cr is None and t_fa is None:
            print(f"[{tag}] neuron {i}: no CR or FA trials, skipping")
            continue

        if t_cr is None:
            t = t_fa
        elif t_fa is None:
            t = t_cr
        else:
            T = min(len(t_cr), len(t_fa))
            t = t_cr[:T]
            m_cr, s_cr = m_cr[:T], s_cr[:T]
            m_fa, s_fa = m_fa[:T], s_fa[:T]

        plt.figure(figsize=(6, 3.5))
        if t_cr is not None:
            plt.plot(t, m_cr[:len(t)], label='CR', linewidth=2)
            plt.fill_between(t, (m_cr - s_cr)[:len(t)], (m_cr + s_cr)[:len(t)], alpha=0.25)
        if t_fa is not None:
            plt.plot(t, m_fa[:len(t)], linestyle='--', label='FA', linewidth=2)
            plt.fill_between(t, (m_fa - s_fa)[:len(t)], (m_fa + s_fa)[:len(t)], alpha=0.25)

        plt.axvline(0, linestyle='--', linewidth=1)
        add_axvline_opto_seconds(plt.gca(), pre_trial_length, rec_hz)

        plt.xlim(t[0], min(t[-1], 20))
        plt.xlabel("Time (s)")
        plt.ylabel("dF/F")
        plt.title(f"{tag} | neuron {i} | CP={cp_med[i]:.2f}")
        plt.legend(loc='best')
        plt.tight_layout()

        fname = os.path.join(out_dir, f"{tag}_neuron_{i}_CP_{cp_med[i]:.2f}.pdf")
        plt.savefig(fname, dpi=300)
        plt.close()
        print("Saved:", fname)


def choose_representative_neurons(cp_values, neuron_indices, cp_bins=None, n_per_bin=3):
    if cp_bins is None:
        cp_bins = [(0.0, 0.4), (0.45, 0.55), (0.6, 1.0)]

    cp_values = np.asarray(cp_values)
    neuron_indices = np.asarray(neuron_indices).astype(int)

    chosen = []
    for low, high in cp_bins:
        in_bin = neuron_indices[(cp_values[neuron_indices] >= low) & (cp_values[neuron_indices] <= high)]
        if len(in_bin) == 0:
            continue

        center = 0.5 * (low + high)
        order = np.argsort(np.abs(cp_values[in_bin] - center))
        chosen.extend(in_bin[order][:min(n_per_bin, len(in_bin))])

    return np.unique(chosen)


###############################################################################
# Group plots
###############################################################################
def raincloud_plot_two_groups_horizontal_overlay(cp_gaussian, cp_sigmoid, cp_others,
                                                 save_fig=False,
                                                 out_path="CP_raincloud_horizontal_overlay_tight.png"):
    cp_gaussian = np.array(cp_gaussian)
    cp_sigmoid = np.array(cp_sigmoid)
    cp_others = np.array(cp_others)

    cp_gaussian = cp_gaussian[np.isfinite(cp_gaussian)]
    cp_sigmoid = cp_sigmoid[np.isfinite(cp_sigmoid)]
    cp_others = cp_others[np.isfinite(cp_others)]

    fig, ax = plt.subplots(figsize=(8, 6))

    density_y_baseline = 0.5
    density_height = 0.2

    x_min = min(np.min(cp_gaussian), np.min(cp_sigmoid), np.min(cp_others))
    x_max = max(np.max(cp_gaussian), np.max(cp_sigmoid), np.max(cp_others))
    x_range = np.linspace(x_min, x_max, 200)

    def plot_density_cloud(data, color):
        if len(data) < 2:
            return
        kde = gaussian_kde(data)
        densities = kde(x_range)
        scaled_dens = densities / np.max(densities) * density_height
        poly_x = np.concatenate([x_range, x_range[::-1]])
        poly_y = np.concatenate([np.full_like(x_range, density_y_baseline),
                                 (density_y_baseline + scaled_dens)[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.4, zorder=1)

    plot_density_cloud(cp_gaussian, color='tan')
    plot_density_cloud(cp_sigmoid, color='lightsteelblue')
    plot_density_cloud(cp_others, color='lightgray')

    summary_y_gaussian = 0.45
    summary_y_others = 0.40
    summary_y_sigmoid = 0.35
    box_height = 0.03

    def plot_boxplot(data, y_center, color):
        if len(data) < 2:
            return
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        rect = plt.Rectangle((q1, y_center - box_height / 2),
                             q3 - q1, box_height,
                             color=color, alpha=0.6, zorder=2)
        ax.add_patch(rect)
        ax.plot([med, med], [y_center - box_height / 2, y_center + box_height / 2],
                color='k', zorder=3)

    def plot_scatter(data, y_center, color):
        jitter = np.random.uniform(-0.005, 0.005, size=len(data))
        ax.scatter(data, np.full_like(data, y_center) + jitter,
                   color=color, alpha=0.7, s=10, zorder=4)

    plot_boxplot(cp_gaussian, summary_y_gaussian, color='tan')
    plot_boxplot(cp_others, summary_y_others, color='lightgray')
    plot_boxplot(cp_sigmoid, summary_y_sigmoid, color='lightsteelblue')

    plot_scatter(cp_gaussian, summary_y_gaussian, color='tan')
    plot_scatter(cp_others, summary_y_others, color='lightgray')
    plot_scatter(cp_sigmoid, summary_y_sigmoid, color='lightsteelblue')

    ax.set_xlabel("Choice Probability")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Gaussian vs Sigmoid vs Other Choice Probability")
    ax.legend(["Gaussian", "Sigmoid", "Other"], loc='upper right')
    plt.tight_layout()
    if save_fig:
        plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cp_group_hist_overlay(cp_all, cp_gaussian, cp_sigmoid, cp_other, save_loc):
    plt.figure(figsize=(10, 5))
    plt.hist(cp_all, bins=20, alpha=0.35, label='All neurons', color='gray', density=True)
    plt.hist(cp_gaussian, bins=20, alpha=0.6, label='Gaussian neurons', color='blue', density=True)
    plt.hist(cp_sigmoid, bins=20, alpha=0.6, label='Sigmoid neurons', color='orange', density=True)
    if len(cp_other) > 0:
        plt.hist(cp_other, bins=20, alpha=0.45, label='Other neurons', color='green', density=True)

    plt.xlabel('Choice Probability')
    plt.ylabel('Density')
    plt.title('CP Distribution: All vs Gaussian vs Sigmoid vs Other')
    plt.legend()
    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(join(save_loc, 'cp_all_vs_gaussian_vs_sigmoid_vs_other.pdf'), dpi=300)
    plt.savefig(join(save_loc, 'cp_all_vs_gaussian_vs_sigmoid_vs_other.png'), dpi=300)
    plt.close()


def plot_cp_vs_response_for_subset(cp_values, cr_values, cr_sem, fa_values, neuron_indices, save_loc, group_name):
    idx = sanitize_indices(neuron_indices, len(cp_values))
    os.makedirs(save_loc, exist_ok=True)

    cp_sub = np.asarray(cp_values)[idx]
    cr_sub = np.asarray(cr_values)[idx]
    cr_sem_sub = np.asarray(cr_sem)[idx]
    fa_sub = np.asarray(fa_values)[idx]

    valid = np.isfinite(cp_sub) & np.isfinite(cr_sub) & np.isfinite(cr_sem_sub)
    if np.sum(valid) >= 2:
        r, p = pearsonr(cp_sub[valid], cr_sub[valid])
    else:
        r, p = np.nan, np.nan

    plt.figure()
    plt.errorbar(cp_sub[valid], cr_sub[valid], yerr=cr_sem_sub[valid],
                 fmt='o', color='gray', alpha=0.5, ecolor='black', capsize=3)
    if np.isfinite(r):
        y_text = np.nanmax(cr_sub[valid]) * 0.9 if np.any(valid) else 0.1
        plt.text(0.1, y_text, f'r = {r:.2f}, p = {p:.3f}', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title(f'Correct Rejection - {group_name}')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, f'CP_vs_CR_response_{group_name}.pdf'))
    plt.savefig(join(save_loc, f'CP_vs_CR_response_{group_name}.png'))
    plt.close()

    valid = np.isfinite(cp_sub) & np.isfinite(fa_sub)
    if np.sum(valid) >= 2:
        r_fa, p_fa = pearsonr(cp_sub[valid], fa_sub[valid])
    else:
        r_fa, p_fa = np.nan, np.nan

    plt.figure()
    plt.scatter(cp_sub[valid], fa_sub[valid], color='gray', alpha=0.5)
    if np.isfinite(r_fa):
        y_text = np.nanmax(fa_sub[valid]) * 0.9 if np.any(valid) else 0.1
        plt.text(0.1, y_text, f'r = {r_fa:.2f}, p = {p_fa:.3f}', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title(f'False Alarm - {group_name}')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, f'CP_vs_FA_response_{group_name}.pdf'))
    plt.savefig(join(save_loc, f'CP_vs_FA_response_{group_name}.png'))
    plt.close()


def summarize_group_threshold_passing(group_indices, comparison_results):
    group_indices = np.asarray(group_indices).astype(int)

    threshold_names = {
        'exc_above_99_5th_idx': 'Exc >= 99.5th',
        'exc_above_99th_idx': 'Exc >= 99th',
        'exc_above_97_5th_idx': 'Exc >= 97.5th',
        'exc_above_95th_idx': 'Exc >= 95th',
        'inh_below_0_5th_idx': 'Inh <= 0.5th',
        'inh_below_1st_idx': 'Inh <= 1st',
        'inh_below_2_5th_idx': 'Inh <= 2.5th',
        'inh_below_5th_idx': 'Inh <= 5th',
    }

    rows = []
    for key, label in threshold_names.items():
        passed_idx = np.asarray(comparison_results['indices_passing_thresholds'][key]).astype(int)
        overlap = np.intersect1d(group_indices, passed_idx)
        rows.append({
            'threshold': label,
            'count': len(overlap),
            'percent_of_group': (len(overlap) / len(group_indices) * 100) if len(group_indices) > 0 else np.nan,
            'indices': ';'.join(map(str, overlap.tolist()))
        })

    return pd.DataFrame(rows)


def plot_group_threshold_summary(df_summary, group_name, save_loc):
    plt.figure(figsize=(10, 5))
    plt.bar(df_summary['threshold'], df_summary['percent_of_group'], color='gray', alpha=0.8)
    plt.ylabel('Percent of group')
    plt.title(f'Neurons passing thresholds - {group_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(join(save_loc, f'threshold_passing_{group_name}.pdf'), dpi=300)
    plt.savefig(join(save_loc, f'threshold_passing_{group_name}.png'), dpi=300)
    plt.close()


###############################################################################
# Real data threshold comparison
###############################################################################
def compare_real_data_to_thresholds(real_data_path, control_thresholds, data_name, save_loc=None):
    print(f"\nLoading real data from: {real_data_path}")

    with open(os.path.join(real_data_path, data_name), 'rb') as f:
        real_choice_prob = NumpyCoreUnpickler(f).load()

    n_neurons_real = len(real_choice_prob['all_distributions'])
    real_choice_medians = np.array([
        np.nanmean(real_choice_prob['all_distributions'][i]) for i in range(n_neurons_real)
    ])

    valid_mask_real = ~np.isnan(real_choice_medians)
    valid_cp_real = real_choice_medians[valid_mask_real]
    idx_valid = np.nonzero(valid_mask_real)[0]

    def _indices(comp, control_value):
        if np.isnan(control_value):
            return np.array([], dtype=int)
        mask = comp(valid_cp_real, control_value)
        return idx_valid[mask]

    exc_99_5th_control = control_thresholds['excitatory_99_5th_percentile']
    exc_99th_control = control_thresholds['excitatory_99th_percentile']
    exc_97_5th_control = control_thresholds['excitatory_97_5th_percentile']
    exc_95th_control = control_thresholds['excitatory_95th_percentile']

    inh_0_5th_control = control_thresholds['inhibitory_0_5th_percentile']
    inh_1st_control = control_thresholds['inhibitory_1st_percentile']
    inh_2_5th_control = control_thresholds['inhibitory_2_5th_percentile']
    inh_5th_control = control_thresholds['inhibitory_5th_percentile']

    exc_above_99_5th_idx = _indices(np.greater_equal, exc_99_5th_control)
    exc_above_99th_idx = _indices(np.greater_equal, exc_99th_control)
    exc_above_97_5th_idx = _indices(np.greater_equal, exc_97_5th_control)
    exc_above_95th_idx = _indices(np.greater_equal, exc_95th_control)

    inh_below_0_5th_idx = _indices(np.less_equal, inh_0_5th_control)
    inh_below_1st_idx = _indices(np.less_equal, inh_1st_control)
    inh_below_2_5th_idx = _indices(np.less_equal, inh_2_5th_control)
    inh_below_5th_idx = _indices(np.less_equal, inh_5th_control)

    n_exc_above_99_5th = len(exc_above_99_5th_idx)
    n_exc_above_99th = len(exc_above_99th_idx)
    n_exc_above_97_5th = len(exc_above_97_5th_idx)
    n_exc_above_95th = len(exc_above_95th_idx)

    n_inh_below_0_5th = len(inh_below_0_5th_idx)
    n_inh_below_1st = len(inh_below_1st_idx)
    n_inh_below_2_5th = len(inh_below_2_5th_idx)
    n_inh_below_5th = len(inh_below_5th_idx)

    pct_exc_above_99_5th = (n_exc_above_99_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_99th = (n_exc_above_99th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_97_5th = (n_exc_above_97_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_exc_above_95th = (n_exc_above_95th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0

    pct_inh_below_0_5th = (n_inh_below_0_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_1st = (n_inh_below_1st / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_2_5th = (n_inh_below_2_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0
    pct_inh_below_5th = (n_inh_below_5th / len(valid_cp_real) * 100) if len(valid_cp_real) > 0 else 0

    comparison_results = {
        'neurons_passing_thresholds': {
            'n_exc_above_99_5th': n_exc_above_99_5th,
            'n_exc_above_99th': n_exc_above_99th,
            'n_exc_above_97_5th': n_exc_above_97_5th,
            'n_exc_above_95th': n_exc_above_95th,
            'n_inh_below_0_5th': n_inh_below_0_5th,
            'n_inh_below_1st': n_inh_below_1st,
            'n_inh_below_2_5th': n_inh_below_2_5th,
            'n_inh_below_5th': n_inh_below_5th
        },
        'percentages_passing_thresholds': {
            'pct_exc_above_99_5th': pct_exc_above_99_5th,
            'pct_exc_above_99th': pct_exc_above_99th,
            'pct_exc_above_97_5th': pct_exc_above_97_5th,
            'pct_exc_above_95th': pct_exc_above_95th,
            'pct_inh_below_0_5th': pct_inh_below_0_5th,
            'pct_inh_below_1st': pct_inh_below_1st,
            'pct_inh_below_2_5th': pct_inh_below_2_5th,
            'pct_inh_below_5th': pct_inh_below_5th
        },
        'indices_passing_thresholds': {
            'exc_above_99_5th_idx': exc_above_99_5th_idx.tolist(),
            'exc_above_99th_idx': exc_above_99th_idx.tolist(),
            'exc_above_97_5th_idx': exc_above_97_5th_idx.tolist(),
            'exc_above_95th_idx': exc_above_95th_idx.tolist(),
            'inh_below_0_5th_idx': inh_below_0_5th_idx.tolist(),
            'inh_below_1st_idx': inh_below_1st_idx.tolist(),
            'inh_below_2_5th_idx': inh_below_2_5th_idx.tolist(),
            'inh_below_5th_idx': inh_below_5th_idx.tolist()
        }
    }

    if save_loc is not None:
        os.makedirs(save_loc, exist_ok=True)

    print(f"\n=== Real Data vs Control Thresholds Comparison ===")
    print(f"  Excitatory above 99.5th: {n_exc_above_99_5th} ({pct_exc_above_99_5th:.2f}%)")
    print(f"  Excitatory above 99th: {n_exc_above_99th} ({pct_exc_above_99th:.2f}%)")
    print(f"  Excitatory above 97.5th: {n_exc_above_97_5th} ({pct_exc_above_97_5th:.2f}%)")
    print(f"  Excitatory above 95th: {n_exc_above_95th} ({pct_exc_above_95th:.2f}%)")
    print(f"  Inhibitory below 0.5th: {n_inh_below_0_5th} ({pct_inh_below_0_5th:.2f}%)")
    print(f"  Inhibitory below 1st: {n_inh_below_1st} ({pct_inh_below_1st:.2f}%)")
    print(f"  Inhibitory below 2.5th: {n_inh_below_2_5th} ({pct_inh_below_2_5th:.2f}%)")
    print(f"  Inhibitory below 5th: {n_inh_below_5th} ({pct_inh_below_5th:.2f}%)")

    plt.figure(figsize=(12, 6))
    thresholds = ['Exc 99.5th', 'Exc 99th', 'Exc 97.5th', 'Exc 95th',
                  'Inh 0.5th', 'Inh 1st', 'Inh 2.5th', 'Inh 5th']
    percentages = [pct_exc_above_99_5th, pct_exc_above_99th, pct_exc_above_97_5th, pct_exc_above_95th,
                   pct_inh_below_0_5th, pct_inh_below_1st, pct_inh_below_2_5th, pct_inh_below_5th]
    colors = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue']
    plt.bar(thresholds, percentages, color=colors, alpha=0.7)
    plt.ylabel('Percentage of neurons')
    plt.title('Neurons passing thresholds')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'neurons_passing_thresholds.png'))
    plt.savefig(os.path.join(save_loc, 'neurons_passing_thresholds.pdf'))
    plt.close()

    with open(os.path.join(save_loc, 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(comparison_results, f)

    rows = []
    for name, idxs in comparison_results['indices_passing_thresholds'].items():
        rows.append({
            'threshold': name,
            'count': len(idxs),
            'indices': ';'.join(map(str, idxs))
        })
    pd.DataFrame(rows).to_csv(os.path.join(save_loc, 'indices_passing_thresholds.csv'), index=False)

    return comparison_results, real_choice_medians, real_choice_prob


###############################################################################
# Subset analysis
###############################################################################
def run_subset_cp_analysis(
        group_name,
        group_indices,
        cp_values_real_data,
        real_choice_prob,
        CR_average_real,
        CR_sem_real,
        FA_average_real,
        dff_run_list_sub,
        real_data_path,
        save_loc,
        rec_hz,
        pre_trial_length
):
    group_indices = np.asarray(group_indices).astype(int)
    group_save = join(save_loc, "subset_analysis", group_name)
    os.makedirs(group_save, exist_ok=True)

    print(f"\nRunning subset analysis for {group_name}: n = {len(group_indices)}")

    if len(group_indices) == 0:
        print(f"No neurons found in group {group_name}. Skipping.")
        return

    choice_probability_hist(
        choice_prob=real_choice_prob,
        save_loc=group_save,
        type=group_name,
        sig_neuron=group_indices
    )

    plot_cp_vs_response_for_subset(
        cp_values=cp_values_real_data,
        cr_values=CR_average_real,
        cr_sem=CR_sem_real,
        fa_values=FA_average_real,
        neuron_indices=group_indices,
        save_loc=group_save,
        group_name=group_name
    )

    rep_idx = choose_representative_neurons(
        cp_values=cp_values_real_data,
        neuron_indices=group_indices,
        cp_bins=[(0.0, 0.4), (0.45, 0.55), (0.6, 1.0)],
        n_per_bin=3
    )

    if len(rep_idx) > 0:
        plot_neurons_by_index(
            dff_run_list_sub=dff_run_list_sub,
            cp_med=cp_values_real_data,
            load_loc=real_data_path,
            save_loc=group_save,
            rec_hz=rec_hz,
            pre_trial_length=pre_trial_length,
            neuron_indices=rep_idx,
            tag=f"rep_{group_name}"
        )

def compare_group_to_thresholds(group_name, group_indices, comparison_results, cp_values_real_data, save_loc):
    """
    Compare one subgroup (Gaussian / Sigmoid / Other / All) to the already-computed
    global control thresholds, and save a group-specific comparison_results.pkl
    with the same structure as the original global comparison.

    The thresholds themselves stay global and fixed.
    """
    group_indices = np.asarray(group_indices).astype(int)
    group_save = join(save_loc, "subset_analysis", group_name)
    os.makedirs(group_save, exist_ok=True)

    # Pull global passing indices
    exc_above_99_5th_global = np.asarray(comparison_results['indices_passing_thresholds']['exc_above_99_5th_idx']).astype(int)
    exc_above_99th_global = np.asarray(comparison_results['indices_passing_thresholds']['exc_above_99th_idx']).astype(int)
    exc_above_97_5th_global = np.asarray(comparison_results['indices_passing_thresholds']['exc_above_97_5th_idx']).astype(int)
    exc_above_95th_global = np.asarray(comparison_results['indices_passing_thresholds']['exc_above_95th_idx']).astype(int)

    inh_below_0_5th_global = np.asarray(comparison_results['indices_passing_thresholds']['inh_below_0_5th_idx']).astype(int)
    inh_below_1st_global = np.asarray(comparison_results['indices_passing_thresholds']['inh_below_1st_idx']).astype(int)
    inh_below_2_5th_global = np.asarray(comparison_results['indices_passing_thresholds']['inh_below_2_5th_idx']).astype(int)
    inh_below_5th_global = np.asarray(comparison_results['indices_passing_thresholds']['inh_below_5th_idx']).astype(int)

    # Intersect with subgroup
    exc_above_99_5th_idx = np.intersect1d(group_indices, exc_above_99_5th_global)
    exc_above_99th_idx = np.intersect1d(group_indices, exc_above_99th_global)
    exc_above_97_5th_idx = np.intersect1d(group_indices, exc_above_97_5th_global)
    exc_above_95th_idx = np.intersect1d(group_indices, exc_above_95th_global)

    inh_below_0_5th_idx = np.intersect1d(group_indices, inh_below_0_5th_global)
    inh_below_1st_idx = np.intersect1d(group_indices, inh_below_1st_global)
    inh_below_2_5th_idx = np.intersect1d(group_indices, inh_below_2_5th_global)
    inh_below_5th_idx = np.intersect1d(group_indices, inh_below_5th_global)

    # Counts
    n_exc_above_99_5th = len(exc_above_99_5th_idx)
    n_exc_above_99th = len(exc_above_99th_idx)
    n_exc_above_97_5th = len(exc_above_97_5th_idx)
    n_exc_above_95th = len(exc_above_95th_idx)

    n_inh_below_0_5th = len(inh_below_0_5th_idx)
    n_inh_below_1st = len(inh_below_1st_idx)
    n_inh_below_2_5th = len(inh_below_2_5th_idx)
    n_inh_below_5th = len(inh_below_5th_idx)

    n_group = len(group_indices)

    # Percentages within subgroup
    pct_exc_above_99_5th = (n_exc_above_99_5th / n_group * 100) if n_group > 0 else 0
    pct_exc_above_99th = (n_exc_above_99th / n_group * 100) if n_group > 0 else 0
    pct_exc_above_97_5th = (n_exc_above_97_5th / n_group * 100) if n_group > 0 else 0
    pct_exc_above_95th = (n_exc_above_95th / n_group * 100) if n_group > 0 else 0

    pct_inh_below_0_5th = (n_inh_below_0_5th / n_group * 100) if n_group > 0 else 0
    pct_inh_below_1st = (n_inh_below_1st / n_group * 100) if n_group > 0 else 0
    pct_inh_below_2_5th = (n_inh_below_2_5th / n_group * 100) if n_group > 0 else 0
    pct_inh_below_5th = (n_inh_below_5th / n_group * 100) if n_group > 0 else 0

    # Same structure as original comparison_results.pkl
    group_comparison_results = {
        'group_name': group_name,
        'n_group_neurons': n_group,
        'group_indices': group_indices.tolist(),
        'cp_mean': np.nanmean(cp_values_real_data[group_indices]) if n_group > 0 else np.nan,
        'cp_median': np.nanmedian(cp_values_real_data[group_indices]) if n_group > 0 else np.nan,
        'cp_std': np.nanstd(cp_values_real_data[group_indices]) if n_group > 0 else np.nan,
        'neurons_passing_thresholds': {
            'n_exc_above_99_5th': n_exc_above_99_5th,
            'n_exc_above_99th': n_exc_above_99th,
            'n_exc_above_97_5th': n_exc_above_97_5th,
            'n_exc_above_95th': n_exc_above_95th,
            'n_inh_below_0_5th': n_inh_below_0_5th,
            'n_inh_below_1st': n_inh_below_1st,
            'n_inh_below_2_5th': n_inh_below_2_5th,
            'n_inh_below_5th': n_inh_below_5th
        },
        'percentages_passing_thresholds': {
            'pct_exc_above_99_5th': pct_exc_above_99_5th,
            'pct_exc_above_99th': pct_exc_above_99th,
            'pct_exc_above_97_5th': pct_exc_above_97_5th,
            'pct_exc_above_95th': pct_exc_above_95th,
            'pct_inh_below_0_5th': pct_inh_below_0_5th,
            'pct_inh_below_1st': pct_inh_below_1st,
            'pct_inh_below_2_5th': pct_inh_below_2_5th,
            'pct_inh_below_5th': pct_inh_below_5th
        },
        'indices_passing_thresholds': {
            'exc_above_99_5th_idx': exc_above_99_5th_idx.tolist(),
            'exc_above_99th_idx': exc_above_99th_idx.tolist(),
            'exc_above_97_5th_idx': exc_above_97_5th_idx.tolist(),
            'exc_above_95th_idx': exc_above_95th_idx.tolist(),
            'inh_below_0_5th_idx': inh_below_0_5th_idx.tolist(),
            'inh_below_1st_idx': inh_below_1st_idx.tolist(),
            'inh_below_2_5th_idx': inh_below_2_5th_idx.tolist(),
            'inh_below_5th_idx': inh_below_5th_idx.tolist()
        }
    }

    # Save the same-named file as in the original
    with open(join(group_save, 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(group_comparison_results, f)

    # Also save a readable CSV
    rows = []
    for key, idxs in group_comparison_results['indices_passing_thresholds'].items():
        count_key = key.replace('_idx', '')
        pct_key = count_key.replace('exc_', 'pct_exc_').replace('inh_', 'pct_inh_')
        rows.append({
            'threshold': key,
            'count': len(idxs),
            'percent_of_group': group_comparison_results['percentages_passing_thresholds'].get(pct_key, np.nan),
            'indices': ';'.join(map(str, idxs))
        })
    pd.DataFrame(rows).to_csv(join(group_save, 'indices_passing_thresholds.csv'), index=False)

    # Plot percent passing in subgroup
    plt.figure(figsize=(10, 5))
    labels = [
        'Exc 99.5th', 'Exc 99th', 'Exc 97.5th', 'Exc 95th',
        'Inh 0.5th', 'Inh 1st', 'Inh 2.5th', 'Inh 5th'
    ]
    percentages = [
        pct_exc_above_99_5th, pct_exc_above_99th, pct_exc_above_97_5th, pct_exc_above_95th,
        pct_inh_below_0_5th, pct_inh_below_1st, pct_inh_below_2_5th, pct_inh_below_5th
    ]
    colors = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue']

    plt.bar(labels, percentages, color=colors, alpha=0.75)
    plt.ylabel('Percentage of neurons')
    plt.title(f'Neurons passing thresholds - {group_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(join(group_save, 'neurons_passing_thresholds.png'), dpi=300)
    plt.savefig(join(group_save, 'neurons_passing_thresholds.pdf'), dpi=300)
    plt.close()

    print(f"\n=== {group_name.upper()} group vs control thresholds ===")
    print(f"  Excitatory above 99.5th: {n_exc_above_99_5th} ({pct_exc_above_99_5th:.2f}%)")
    print(f"  Excitatory above 99th: {n_exc_above_99th} ({pct_exc_above_99th:.2f}%)")
    print(f"  Excitatory above 97.5th: {n_exc_above_97_5th} ({pct_exc_above_97_5th:.2f}%)")
    print(f"  Excitatory above 95th: {n_exc_above_95th} ({pct_exc_above_95th:.2f}%)")
    print(f"  Inhibitory below 0.5th: {n_inh_below_0_5th} ({pct_inh_below_0_5th:.2f}%)")
    print(f"  Inhibitory below 1st: {n_inh_below_1st} ({pct_inh_below_1st:.2f}%)")
    print(f"  Inhibitory below 2.5th: {n_inh_below_2_5th} ({pct_inh_below_2_5th:.2f}%)")
    print(f"  Inhibitory below 5th: {n_inh_below_5th} ({pct_inh_below_5th:.2f}%)")

    return group_comparison_results
###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    ##########################################################################
    # Settings
    ##########################################################################
    day = ['240529']
    mouse = 'OI116'

    base_load_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR'.format(mouse, day[0])
    base_load_loc2 = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR'.format(mouse, day[0])

    gaussian_sigmoid_path = r'Z:\2p_data\{}\{}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v4'.format(mouse, day[0])

    save_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_freq_history_control_CR-CR_rep1\all_reps_with_sigmoid_gaussian'.format(mouse, day[0])
    real_data_path = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_freq_history'.format(mouse, day[0])

    data_name = 'res_cr_subsample_global.pkl'

    os.makedirs(save_loc, exist_ok=True)

    rec_hz = 15
    pre_trial_length = 4 * rec_hz

    repetitions = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5']

    ##########################################################################
    # Load dff for representative plots
    ##########################################################################
    with open(join(real_data_path, 'dff_run_list.pkl'), 'rb') as f:
        dff_run_list = pickle.load(f)

    dff_run_list_sub = baseline_subtract_neurons(
        dff_run_list,
        pre_trial_length,
        baseline_time=4 * rec_hz
    )

    ##########################################################################
    # Collect CP from all repetitions
    ##########################################################################
    all_repetition_results = {}

    for rep in repetitions:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING REPETITION: {rep}")
        print(f"{'=' * 60}")

        load_loc = f"{base_load_loc}_{rep}"

        if not os.path.exists(load_loc):
            print(f"Warning: {load_loc} does not exist. Skipping {rep}.")
            continue

        try:
            with open(os.path.join(load_loc, data_name), 'rb') as f:
                choice_prob = NumpyCoreUnpickler(f).load()
        except FileNotFoundError:
            print(f"Warning: {data_name} not found in {load_loc}. Skipping {rep}.")
            continue

        sig_neuron_indices = np.arange(len(choice_prob['all_distributions']))
        CP_all = [np.nanmean(choice_prob['all_distributions'][i]) for i in sig_neuron_indices]

        all_repetition_results[rep] = {
            'CP_all': CP_all,
            'n_neurons': len(sig_neuron_indices)
        }

        print(f"Repetition {rep}: {len(sig_neuron_indices)} neurons")

    ##########################################################################
    # Average control thresholds across repetitions
    ##########################################################################
    if len(all_repetition_results) <= 1:
        raise RuntimeError("Not enough repetitions found. Need at least 2 repetitions.")

    reps_with_cp_data = [rep for rep, results in all_repetition_results.items() if 'CP_all' in results]
    if len(reps_with_cp_data) <= 1:
        raise RuntimeError("Not enough repetitions with CP data.")

    print(f"\n{'=' * 60}")
    print("CP DISTRIBUTION ANALYSIS AND AVERAGING")
    print(f"{'=' * 60}")

    print("\n1. Testing CP distribution differences (Kolmogorov-Smirnov test):")
    cp_distributions = [all_repetition_results[rep]['CP_all'] for rep in reps_with_cp_data]

    ks_results = []
    for i in range(len(cp_distributions)):
        for j in range(i + 1, len(cp_distributions)):
            stat, p_val = scipy.stats.ks_2samp(cp_distributions[i], cp_distributions[j])
            ks_results.append((reps_with_cp_data[i], reps_with_cp_data[j], stat, p_val))
            print(f"   {reps_with_cp_data[i]} vs {reps_with_cp_data[j]}: D={stat:.3f}, p={p_val:.3f}")

    alpha = 0.05
    significant_differences = [p < alpha for _, _, _, p in ks_results]
    any_significant = any(significant_differences)
    print(f"\nSignificant differences found: {any_significant}")

    print(f"\n2. Averaging CP distributions across {len(reps_with_cp_data)} repetitions:")

    n_neurons = all_repetition_results[reps_with_cp_data[0]]['n_neurons']
    cp_data = np.array([all_repetition_results[rep]['CP_all'] for rep in reps_with_cp_data])

    avg_cp = np.mean(cp_data, axis=0)
    se_cp = np.std(cp_data, axis=0) / np.sqrt(len(reps_with_cp_data))

    print(f"   Averaged {n_neurons} neurons across {len(reps_with_cp_data)} repetitions")
    print(f"   Mean CP: {np.mean(avg_cp):.3f} ± {np.std(avg_cp):.3f}")

    ##########################################################################
    # Debug plots for repetition averaging
    ##########################################################################
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    for i, rep in enumerate(reps_with_cp_data):
        plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.3, label=f'{rep}', density=True)
    plt.xlabel('Choice Probability')
    plt.ylabel('Density')
    plt.title('Individual Repetition Distributions')
    plt.legend()
    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)

    plt.subplot(2, 3, 2)
    plt.hist(avg_cp, bins=20, alpha=0.7, color='red', density=True, label='Averaged')
    plt.xlabel('Choice Probability')
    plt.ylabel('Density')
    plt.title('Averaged Distribution')
    plt.legend()
    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)

    plt.subplot(2, 3, 3)
    for rep in reps_with_cp_data:
        plt.hist(all_repetition_results[rep]['CP_all'], bins=20, alpha=0.2, density=True)
    plt.hist(avg_cp, bins=20, alpha=0.8, color='red', linewidth=2, histtype='step', label='Averaged', density=True)
    plt.xlabel('Choice Probability')
    plt.ylabel('Density')
    plt.title('Overlay Comparison')
    plt.legend()
    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5)

    plt.subplot(2, 3, 4)
    for rep in reps_with_cp_data:
        plt.scatter(all_repetition_results[rep]['CP_all'], avg_cp, alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Individual Repetition CP')
    plt.ylabel('Averaged CP')
    plt.title('Individual vs Averaged CP')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    differences = []
    for rep in reps_with_cp_data:
        diff = np.array(all_repetition_results[rep]['CP_all']) - avg_cp
        differences.extend(diff)
    plt.hist(differences, bins=30, alpha=0.7, color='green')
    plt.xlabel('Difference (Individual - Averaged)')
    plt.ylabel('Count')
    plt.title('Distribution of Differences')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    cp_by_rep = [all_repetition_results[rep]['CP_all'] for rep in reps_with_cp_data]
    plt.boxplot(cp_by_rep, labels=reps_with_cp_data)
    plt.ylabel('Choice Probability')
    plt.title('CP Distribution by Repetition')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(save_loc, 'averaging_debug.pdf'), dpi=300)
    plt.savefig(join(save_loc, 'averaging_debug.png'), dpi=300)
    plt.close()

    ##########################################################################
    # Average thresholds from repetition-specific thresholds
    ##########################################################################
    print(f"\n3. Creating thresholds from averaged repetition thresholds:")

    individual_thresholds = {}
    for rep in reps_with_cp_data:
        rep_cp = all_repetition_results[rep]['CP_all']

        exc_99_5th = np.percentile(rep_cp, 99.5) if len(rep_cp) > 0 else np.nan
        exc_99th = np.percentile(rep_cp, 99) if len(rep_cp) > 0 else np.nan
        exc_97_5th = np.percentile(rep_cp, 97.5) if len(rep_cp) > 0 else np.nan
        exc_95th = np.percentile(rep_cp, 95) if len(rep_cp) > 0 else np.nan

        inh_0_5th = np.percentile(rep_cp, 0.5) if len(rep_cp) > 0 else np.nan
        inh_1st = np.percentile(rep_cp, 1) if len(rep_cp) > 0 else np.nan
        inh_2_5th = np.percentile(rep_cp, 2.5) if len(rep_cp) > 0 else np.nan
        inh_5th = np.percentile(rep_cp, 5) if len(rep_cp) > 0 else np.nan

        individual_thresholds[rep] = {
            'excitatory_99_5th': exc_99_5th,
            'excitatory_99th': exc_99th,
            'excitatory_97_5th': exc_97_5th,
            'excitatory_95th': exc_95th,
            'inhibitory_0_5th': inh_0_5th,
            'inhibitory_1st': inh_1st,
            'inhibitory_2_5th': inh_2_5th,
            'inhibitory_5th': inh_5th
        }

    avg_exc_99_5th = np.mean([individual_thresholds[rep]['excitatory_99_5th'] for rep in reps_with_cp_data])
    avg_exc_99th = np.mean([individual_thresholds[rep]['excitatory_99th'] for rep in reps_with_cp_data])
    avg_exc_97_5th = np.mean([individual_thresholds[rep]['excitatory_97_5th'] for rep in reps_with_cp_data])
    avg_exc_95th = np.mean([individual_thresholds[rep]['excitatory_95th'] for rep in reps_with_cp_data])

    avg_inh_0_5th = np.mean([individual_thresholds[rep]['inhibitory_0_5th'] for rep in reps_with_cp_data])
    avg_inh_1st = np.mean([individual_thresholds[rep]['inhibitory_1st'] for rep in reps_with_cp_data])
    avg_inh_2_5th = np.mean([individual_thresholds[rep]['inhibitory_2_5th'] for rep in reps_with_cp_data])
    avg_inh_5th = np.mean([individual_thresholds[rep]['inhibitory_5th'] for rep in reps_with_cp_data])

    se_exc_99_5th = np.std([individual_thresholds[rep]['excitatory_99_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_exc_99th = np.std([individual_thresholds[rep]['excitatory_99th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_exc_97_5th = np.std([individual_thresholds[rep]['excitatory_97_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_exc_95th = np.std([individual_thresholds[rep]['excitatory_95th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))

    se_inh_0_5th = np.std([individual_thresholds[rep]['inhibitory_0_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_inh_1st = np.std([individual_thresholds[rep]['inhibitory_1st'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_inh_2_5th = np.std([individual_thresholds[rep]['inhibitory_2_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))
    se_inh_5th = np.std([individual_thresholds[rep]['inhibitory_5th'] for rep in reps_with_cp_data]) / np.sqrt(len(reps_with_cp_data))

    cp_thresholds = {
        'excitatory_99_5th_percentile': avg_exc_99_5th,
        'excitatory_99th_percentile': avg_exc_99th,
        'excitatory_97_5th_percentile': avg_exc_97_5th,
        'excitatory_95th_percentile': avg_exc_95th,
        'inhibitory_0_5th_percentile': avg_inh_0_5th,
        'inhibitory_1st_percentile': avg_inh_1st,
        'inhibitory_2_5th_percentile': avg_inh_2_5th,
        'inhibitory_5th_percentile': avg_inh_5th,
        'excitatory_99_5th_se': se_exc_99_5th,
        'excitatory_99th_se': se_exc_99th,
        'excitatory_97_5th_se': se_exc_97_5th,
        'excitatory_95th_se': se_exc_95th,
        'inhibitory_0_5th_se': se_inh_0_5th,
        'inhibitory_1st_se': se_inh_1st,
        'inhibitory_2_5th_se': se_inh_2_5th,
        'inhibitory_5th_se': se_inh_5th,
        'individual_thresholds': individual_thresholds,
        'n_repetitions': len(reps_with_cp_data)
    }

    ##########################################################################
    # Compare real data to control thresholds
    ##########################################################################
    print(f"\n4. Testing real data against averaged thresholds:")

    if not os.path.exists(real_data_path):
        raise RuntimeError(f"Real data path not found: {real_data_path}")

    comparison_results, cp_values_real_data, real_choice_prob = compare_real_data_to_thresholds(
        real_data_path=real_data_path,
        control_thresholds=cp_thresholds,
        data_name=data_name,
        save_loc=save_loc
    )

    ##########################################################################
    # Load Gaussian and Sigmoid neuron groups
    ##########################################################################
    gaussian_neuron_indices = np.load(join(gaussian_sigmoid_path, 'gaussian_neuron_indices.npy'))
    sigmoid_neuron_indices = np.load(join(gaussian_sigmoid_path, 'sigmoid_neuron_indices.npy'))

    n_real_neurons = len(cp_values_real_data)

    gaussian_neuron_indices = sanitize_indices(gaussian_neuron_indices, n_real_neurons)
    sigmoid_neuron_indices = sanitize_indices(sigmoid_neuron_indices, n_real_neurons)

    all_neuron_indices = np.arange(n_real_neurons)
    other_neuron_indices = np.setdiff1d(
        all_neuron_indices,
        np.union1d(gaussian_neuron_indices, sigmoid_neuron_indices)
    )

    groups = {
        'all': all_neuron_indices,
        'gaussian': gaussian_neuron_indices,
        'sigmoid': sigmoid_neuron_indices,
        'other': other_neuron_indices
    }

    group_threshold_results = {}

    for group_name, group_idx in groups.items():
        group_threshold_results[group_name] = compare_group_to_thresholds(
            group_name=group_name,
            group_indices=group_idx,
            comparison_results=comparison_results,
            cp_values_real_data=cp_values_real_data,
            save_loc=save_loc
        )

    with open(join(save_loc, 'group_threshold_results.pkl'), 'wb') as f:
        pickle.dump(group_threshold_results, f)

    cp_gaussian_neurons = cp_values_real_data[gaussian_neuron_indices]
    cp_sigmoid_neurons = cp_values_real_data[sigmoid_neuron_indices]
    cp_other_neurons = cp_values_real_data[other_neuron_indices]

    ##########################################################################
    # Group comparison plots
    ##########################################################################
    plot_cp_group_hist_overlay(
        cp_all=cp_values_real_data,
        cp_gaussian=cp_gaussian_neurons,
        cp_sigmoid=cp_sigmoid_neurons,
        cp_other=cp_other_neurons,
        save_loc=save_loc
    )

    raincloud_plot_two_groups_horizontal_overlay(
        cp_gaussian=cp_gaussian_neurons,
        cp_sigmoid=cp_sigmoid_neurons,
        cp_others=cp_other_neurons,
        save_fig=True,
        out_path=join(save_loc, 'cp_raincloud_gaussian_sigmoid_other.png')
    )

    ##########################################################################
    # Load real-data CR / FA averages for subset analysis
    ##########################################################################
    with open(join(real_data_path, 'cr_light_averages.pkl'), 'rb') as f:
        cr_event_averages_real = pickle.load(f)
    with open(join(real_data_path, 'fa_light_averages.pkl'), 'rb') as f:
        fa_event_averages_real = pickle.load(f)

    N_FA_real = len(fa_event_averages_real[0])

    all_results_real = compute_opto_windows_mean_activity(
        cr_event_averages_real,
        fa_event_averages_real,
        N_FA_real,
        1000
    )

    CR_average_real = np.asarray(all_results_real['mean'])
    CR_sem_real = np.asarray(all_results_real['sem'])
    FA_average_real = np.asarray(all_results_real['FA_means'])

    ##########################################################################
    # Run all, Gaussian, Sigmoid, Other subset analyses
    ##########################################################################
    for group_name, group_idx in groups.items():
        run_subset_cp_analysis(
            group_name=group_name,
            group_indices=group_idx,
            cp_values_real_data=cp_values_real_data,
            real_choice_prob=real_choice_prob,
            CR_average_real=CR_average_real,
            CR_sem_real=CR_sem_real,
            FA_average_real=FA_average_real,
            dff_run_list_sub=dff_run_list_sub,
            real_data_path=real_data_path,
            save_loc=save_loc,
            rec_hz=rec_hz,
            pre_trial_length=pre_trial_length
        )

    ##########################################################################
    # Threshold-passing summaries for each group
    ##########################################################################
    summary_dir = join(save_loc, "subset_analysis", "threshold_group_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    for group_name, group_idx in groups.items():
        df_group_summary = summarize_group_threshold_passing(group_idx, comparison_results)
        df_group_summary.to_csv(join(summary_dir, f"threshold_summary_{group_name}.csv"), index=False)
        plot_group_threshold_summary(df_group_summary, group_name, summary_dir)

    ##########################################################################
    # Save a compact group-level summary table
    ##########################################################################
    group_rows = []
    for group_name, group_idx in groups.items():
        cp_group = cp_values_real_data[group_idx]
        group_rows.append({
            'group': group_name,
            'n_neurons': len(group_idx),
            'cp_mean': np.nanmean(cp_group) if len(group_idx) > 0 else np.nan,
            'cp_median': np.nanmedian(cp_group) if len(group_idx) > 0 else np.nan,
            'cp_std': np.nanstd(cp_group) if len(group_idx) > 0 else np.nan,
            'n_cp_above_0_5': np.sum(cp_group >= 0.5) if len(group_idx) > 0 else 0,
            'n_cp_below_0_5': np.sum(cp_group < 0.5) if len(group_idx) > 0 else 0
        })

    pd.DataFrame(group_rows).to_csv(join(save_loc, 'group_cp_summary.csv'), index=False)

    ##########################################################################
    # Original rep1 backup analysis
    ##########################################################################
    print(f"\n{'=' * 60}")
    print("RUNNING ORIGINAL ANALYSIS FOR REP1 (BACKUP)")
    print(f"{'=' * 60}")

    load_loc = f"{base_load_loc}_rep1"
    load_loc2 = f"{base_load_loc2}_rep1"

    with open(os.path.join(load_loc, data_name), 'rb') as f:
        choice_prob_rep1 = NumpyCoreUnpickler(f).load()

    sig_neuron_indices = np.arange(len(choice_prob_rep1['all_distributions']))

    choice_probability_hist(
        choice_prob_rep1,
        save_loc,
        'control_CR-CR',
        sig_neuron=sig_neuron_indices
    )

    with open(join(load_loc2, 'cr_light_averages.pkl'), 'rb') as f:
        cr_event_averages = pickle.load(f)
    with open(join(load_loc2, 'fa_light_averages.pkl'), 'rb') as f:
        fa_event_averages = pickle.load(f)

    N_FA = len(fa_event_averages[0])
    all_results = compute_opto_windows_mean_activity(cr_event_averages, fa_event_averages, N_FA, 1000)

    FA_average = np.asarray(all_results['FA_means'][sig_neuron_indices])
    CR_average = np.asarray(all_results['mean'][sig_neuron_indices])
    CR_sem = np.asarray(all_results['sem'][sig_neuron_indices])

    CP_all = [np.nanmean(choice_prob_rep1['all_distributions'][i]) for i in sig_neuron_indices]

    r, p = pearsonr(CP_all, CR_average)
    with open(join(save_loc, 'CP_vs_CR_r.pkl'), 'wb') as f:
        pickle.dump(r, f)

    plt.figure()
    plt.errorbar(CP_all, CR_average, yerr=CR_sem, fmt='o', color='gray', alpha=0.5, ecolor='black', capsize=3)
    plt.text(0.1, max(CR_average) * 0.9, f'r = {r:.2f}, p = {p:.3f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title('Correct Rejection')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, 'CP_vs_CR_response.pdf'))
    plt.close()

    r_FA, p_FA = pearsonr(CP_all, FA_average)
    with open(join(save_loc, 'CP_vs_FA_r.pkl'), 'wb') as f:
        pickle.dump(r_FA, f)

    plt.figure()
    plt.scatter(CP_all, np.asarray(FA_average), color='gray', alpha=0.5)
    plt.text(0.1, max(FA_average) * 0.9, f'r = {r_FA:.2f}, p = {p_FA:.3f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Choice Probability')
    plt.ylabel('Mean response')
    plt.title('False Alarm')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(join(save_loc, 'CP_vs_FA_response.pdf'))
    plt.close()

    ##########################################################################
    # Save global results
    ##########################################################################
    averaged_results = {
        'avg_CP': avg_cp,
        'se_CP': se_cp,
        'n_neurons': n_neurons,
        'n_repetitions': len(reps_with_cp_data),
        'repetitions_used': reps_with_cp_data,
        'cp_thresholds': cp_thresholds,
        'ks_test_results': ks_results,
        'any_significant_differences': any_significant,
        'individual_rep_data': cp_data,
        'group_indices': {
            'all': all_neuron_indices.tolist(),
            'gaussian': gaussian_neuron_indices.tolist(),
            'sigmoid': sigmoid_neuron_indices.tolist(),
            'other': other_neuron_indices.tolist()
        }
    }

    with open(join(save_loc, 'averaged_cp_results.pkl'), 'wb') as f:
        pickle.dump(averaged_results, f)

    print("\nDone.")