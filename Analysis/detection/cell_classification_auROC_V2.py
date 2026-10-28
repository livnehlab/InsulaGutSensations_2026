# Classification of neuronal responses using auROC against label-shuffled null distributions.
# Inputs are the per-neuron event data produced by creat_events_data_per_neuron.py.
# Note: variable and label names such as '5th'/'95th' (e.g. 'above_95', 'pre95_auroc') are
# historical; the thresholds computed are the 10th/90th percentiles.

# ---------------------------------------------------------------------------
# Imports and plotting configuration
# ---------------------------------------------------------------------------

import copy
import csv
import math
import os
import pickle
from operator import itemgetter
import h5py
import matplotlib
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import wilcoxon, ttest_rel, ttest_ind, ranksums
from sklearn.metrics import roc_auc_score

matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib_venn import venn2
import pandas as pd
from os.path import join
import seaborn as sns
import random
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000  # MonkeyLogic sampling rate (Hz)
rec_hz = 15  # 2-photon frame rate (Hz)
sync_hz = 6000  # ThorSync sampling rate (Hz)
# Condition 5: 1 s distractor light; 4 s opto stim (15 Hz between 2p frames, 10 ms, max
# intensity); 2 s EB odor; 2 s delay (mouse not supposed to lick).
con1 = [5]
# Condition 6: 1 s distractor light; 4 s delay; 2 s EB odor; 2 s reward window for operant
# licking.
con2 = 6
pre_trial_length = 4 * rec_hz  # frames prepended to each trial
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz  # opto stimulation length (frames)


# ---------------------------------------------------------------------------
# Visualization and significance reporting
# ---------------------------------------------------------------------------


def plot_increase_decrease_bar(sig_neurons, total_neurons, Title):
    """
    Makes a grouped bar plot for each event,
    comparing #increased vs. #decreased cells.
    """
    event_names = list(sig_neurons.keys())
    n_events = len(event_names)

    # Gather data
    inc_counts = []
    dec_counts = []
    for e in event_names:
        inc_counts.append((len(sig_neurons[e]['increase']) / total_neurons) * 100)
        dec_counts.append((len(sig_neurons[e]['decrease']) / total_neurons) * 100)

    x = np.arange(n_events)  # positions

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_inc = ax.bar(x - width / 2, inc_counts, width, label='Increase', color='tomato')
    bar_dec = ax.bar(
        x + width / 2, dec_counts, width, label='Decrease', color='cornflowerblue'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=45, ha='right')
    ax.set_ylabel('% Significant Neurons')
    ax.set_title(Title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_loc, Title))
    plt.show()


def exclude_repeated_significance(significant_neurons, events_order):
    """
    If a neuron is 'increase' in event e1 and 'increase' again in event e2
    (where e2 comes after e1 in time), remove it from e2's set.
    The same is done for 'decrease'.
    significant_neurons[e]['increase'] / ['decrease'] are lists of neuron indices.
    """
    for i in range(len(events_order) - 1):  # consecutive event pairs
        e1 = events_order[i]
        e2 = events_order[i + 1]

        # neurons 'increase' in both events
        inc_e1 = set(significant_neurons[e1]['increase'])
        inc_e2 = set(significant_neurons[e2]['increase'])
        overlap_inc = inc_e1.intersection(inc_e2)

        # remove them from e2
        if len(overlap_inc) > 0:
            new_e2_inc = [
                n for n in significant_neurons[e2]['increase'] if n not in overlap_inc
            ]
            significant_neurons[e2]['increase'] = np.array(new_e2_inc, dtype=int)

        # same for 'decrease'
        dec_e1 = set(significant_neurons[e1]['decrease'])
        dec_e2 = set(significant_neurons[e2]['decrease'])
        overlap_dec = dec_e1.intersection(dec_e2)

        if len(overlap_dec) > 0:
            new_e2_dec = [
                n for n in significant_neurons[e2]['decrease'] if n not in overlap_dec
            ]
            significant_neurons[e2]['decrease'] = np.array(new_e2_dec, dtype=int)

    return significant_neurons


def plot_venn(
    set1,
    set2,
    total,
    labels,
    title,
    filename,
    save_loc,
    color1='gold',
    color2='lightpink',
):
    """Two-set Venn diagram labeled with percentages of `total`."""
    plt.figure()
    v = venn2([set1, set2], set_labels=labels, set_colors=(color1, color2))
    v.get_label_by_id('10').set_text(f"{(len(set1 - set2) / total * 100):.1f}%")
    v.get_label_by_id('01').set_text(f"{(len(set2 - set1) / total * 100):.1f}%")
    # the intersection label exists only if the sets overlap
    if len(set1 & set2) > 0:
        v.get_label_by_id('11').set_text(f"{(len(set1 & set2) / total * 100):.1f}%")
    plt.title(title)
    plt.savefig(join(save_loc, filename))
    plt.show()


# ---------------------------------------------------------------------------
# Permutation tests and event comparisons
# ---------------------------------------------------------------------------


def auroc_permutation_test(event_vals, baseline_vals, n_perm=1000, random_state=None):
    """
    Computes the real auROC (event vs. baseline) for one neuron and classifies
    it against a label-shuffled null distribution.

    Parameters
    ----------
    event_vals : 1D array
        The scalar activity values for the "event" window (one per trial).
    baseline_vals : 1D array
        The scalar activity values for the "baseline" window (one per trial).
    n_perm : int
        Number of permutations for building the null distribution.
    random_state : int or None
        If set, fixes the random seed for reproducible permutations.

    Returns
    -------
    real_auc : float
        The actual auROC (0.0 .. 1.0).
    null_5th, null_95th : float
        10th and 90th percentiles of the null distribution.
    perm_distribution : 1D array
        The array of permuted auROCs (length n_perm).
    classification : str
        'above_95' (real > 90th percentile), 'below_5' (real < 10th percentile) or 'ns'.
        If there is not enough data, (nan, nan, empty array) is returned instead.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Remove NaNs
    event_vals = event_vals[~np.isnan(event_vals)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]

    # If not enough data, return NaN
    if (len(event_vals) < 2) or (len(baseline_vals) < 2):
        return np.nan, np.nan, np.array([])

    # Combine data
    n_event = len(event_vals)
    n_base = len(baseline_vals)
    all_data = np.concatenate([event_vals, baseline_vals])
    labels = np.concatenate([np.ones(n_event), np.zeros(n_base)])

    # ----- 1) Compute real auROC -----
    real_auc = roc_auc_score(labels, all_data)

    # ----- 2) Build null distribution by permutation -----
    perm_distribution = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        # Shuffle the labels
        shuffled_labels = np.random.permutation(labels)
        perm_auc = roc_auc_score(shuffled_labels, all_data)
        perm_distribution[i] = perm_auc

    # ----- 3) Compare the real auROC to the null distribution -----
    # 10th and 90th percentiles (variable names are historical)
    null_5th = np.percentile(perm_distribution, 10)
    null_95th = np.percentile(perm_distribution, 90)

    # Classify
    if real_auc > null_95th:
        classification = "above_95"
    elif real_auc < null_5th:
        classification = "below_5"
    else:
        classification = "ns"  # not significant

    return real_auc, null_5th, null_95th, perm_distribution, classification


def compute_auROC_for_events_with_permutation(
    all_event_data, all_data, n_perm=1000, random_state=None
):
    """
    For each event and neuron, computes the auROC of the trial-wise mean activity
    in the event window vs. the baseline window, and classifies it against a
    label-shuffled null distribution (manual_auroc_with_shuffle).

    Returns
    -------
    results : dict
      results[event_name] = {
         'auroc': np.array of shape (n_neurons,),
         'p_value': list of classifications ('above_95' / 'below_5' / 'ns'),
         'all5': np.array of shape (n_neurons,), lower null percentile,
         'all95': np.array of shape (n_neurons,), upper null percentile
      }
    """
    from sklearn.metrics import roc_auc_score

    results = {}

    for event_name, event_dict in all_event_data.items():
        if 'event' not in event_dict or 'baseline' not in event_dict:
            print(f"Skipping {event_name} - no event/baseline data.")
            continue

        event_data_all_neurons = event_dict['event']
        baseline_data_all_neurons = event_dict['baseline']

        n_neurons = len(event_data_all_neurons)
        aurocs = np.full(n_neurons, np.nan)
        all5 = np.full(n_neurons, np.nan)
        all95 = np.full(n_neurons, np.nan)
        p_values = []

        for neuron_i in range(n_neurons):
            neuron_data = all_data.iloc[neuron_i]
            # build arrays of mean activity across frames for each trial
            event_means = []
            for trial_arr in event_data_all_neurons[neuron_i]:
                if trial_arr.size == 0 or np.isnan(trial_arr).all():
                    event_means.append(np.nan)
                else:
                    event_means.append(np.nanmean(trial_arr))

            baseline_means = []
            for trial_arr in baseline_data_all_neurons[neuron_i]:
                if trial_arr.size == 0 or np.isnan(trial_arr).all():
                    baseline_means.append(np.nan)
                else:
                    baseline_means.append(np.nanmean(trial_arr))

            event_means = np.array(event_means)
            baseline_means = np.array(baseline_means)

            # filter out trials with NaNs
            valid_mask = (~np.isnan(event_means)) & (~np.isnan(baseline_means))
            event_vals = event_means[valid_mask]
            base_vals = baseline_means[valid_mask]

            if len(event_vals) < 2 or len(base_vals) < 2:
                continue

            real_auc, null5, null95, dist, cls = manual_auroc_with_shuffle(
                event_vals,
                base_vals,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state,
            )
            aurocs[neuron_i] = real_auc
            p_values.append(cls)
            all5[neuron_i] = null5
            all95[neuron_i] = null95

        results[event_name] = {
            'auroc': aurocs,
            'p_value': p_values,
            'all5': all5,
            'all95': all95,
        }

    return results


# def compute_opto_windows_vs_baseline(
#         all_averages,  # list of shape (#neurons,) each -> (#trials, 2)
#         all_baseline,  # list of shape (#neurons,) each -> (#trials,)
#         n_perm=1000,
#         random_state=None
# ):
#     """
#     For each neuron i:
#       1) event vs. baseline -> auROC_A, pval_A
#
#     Returns
#     -------
#     results : dict
#       {
#         'aurocA':   np.array of shape (n_neurons,),
#         'pvalA':    np.array of shape (n_neurons,),
#       }
#     """
#     n_neurons = len(all_averages)
#     aurocA = np.full(n_neurons, np.nan)
#     pvalA = []
#
#
#     for i in range(n_neurons):
#         # shape (#trials, 2)
#         arr_2d = all_averages[i]
#         if i >= len(all_baseline):
#             continue  # or raise an error if mismatch
#
#         baseline_vals = all_baseline[i]
#         if arr_2d.shape[0] != len(baseline_vals):
#             print(f"Warning: mismatch in trial counts for neuron {i}!")
#             # handle carefully or skip
#             continue
#
#         # Column 0, column 1
#         windowA = arr_2d[:, 0]
#
#         # Drop trials with NaNs in either distribution
#         # For window A, we also need baseline not NaN
#         maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
#         wA_vals = windowA[maskA]
#         baseA = baseline_vals[maskA]
#
#
#         # If enough data, do the permutation test
#         # --- Window A vs. baseline
#         if len(wA_vals) >= 2 and len(baseA) >= 2:
#             aA, null_5th, null_95th, dist, pA = auroc_permutation_test(wA_vals, baseA, n_perm, random_state)
#             aurocA[i] = aA
#             pvalA.append(pA)
#
#     results = {
#         'aurocA': aurocA,
#         'pvalA': pvalA,
#         'all5': null_5th,
#         'all95': null_95th
#     }
#     return results


# ---------------------------------------------------------------------------
# Sliding-window comparisons
# ---------------------------------------------------------------------------


def compute_opto_windows_vs_baseline_sliding(
    all_averages,  # list of (#trials, 2) arrays, one per neuron
    all_baseline,  # list of (#trials,) arrays, one per neuron
    all_data,
    window_size=40,  # how many trials in each sliding window
    step=10,  # how many trials to jump each time
    n_perm=1000,
    random_state=None,
):
    """
    For each neuron:
      1) Slide a window of 'window_size' valid trials, moving in steps of
         'step' (e.g. 40 trials at a time, shift by 10).
      2) Compute the auROC of each window.
      3) Pick the window whose auROC is farthest from 0.5.
      4) Run the label-shuffle test on that window only.

    Returns
    -------
    results : dict
      {
         'final_auroc':    np.array of shape (n_neurons,),
         'null_5th':       np.array(...),
         'null_95th':      np.array(...),
         'classification': list of strings ('above_95' / 'below_5' / 'ns')
         'best_range':     list of tuples (start_idx, end_idx)
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    final_auroc = np.full(n_neurons, np.nan)
    final_5th = np.full(n_neurons, np.nan)
    final_95th = np.full(n_neurons, np.nan)
    classifications = [None] * n_neurons
    best_ranges = [None] * n_neurons

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, n_windows); column 0 is the event window
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]
        if arr_2d.shape[0] != len(baseline_vals):
            print(f"Warning: mismatch in trial counts for neuron {i}!")
            continue

        event_full = arr_2d[:, 0]

        # Filter out trials with NaNs in event or baseline
        mask = (~np.isnan(event_full)) & (~np.isnan(baseline_vals))
        event_vals = event_full[mask]
        base_vals = baseline_vals[mask]

        n_valid = len(event_vals)
        if n_valid < window_size:
            # not enough trials for one window
            continue

        # 1) Slide over the valid trials in increments of 'step'
        max_auroc_sub = 0
        best_start = 0
        best_end = 0

        # start = 0, step, 2*step, ... while start + window_size <= n_valid
        for start_idx in range(0, n_valid - window_size + 1, step):
            end_idx = start_idx + window_size
            event_chunk = event_vals[start_idx:end_idx]
            base_chunk = base_vals[start_idx:end_idx]

            # auROC of this window (no permutation yet)
            labels = np.concatenate(
                [np.ones(len(event_chunk)), np.zeros(len(base_chunk))]
            )
            scores = np.concatenate([event_chunk, base_chunk])
            this_auroc = roc_auc_score(labels, scores)

            abs_diff = np.abs(this_auroc - 0.5)
            if abs_diff > max_auroc_sub:
                max_auroc_sub = abs_diff
                best_start = start_idx
                best_end = end_idx

        # 2) Label-shuffle test on the selected window only
        if best_end > best_start:
            event_chunk = event_vals[best_start:best_end]
            base_chunk = base_vals[best_start:best_end]

            real_auc, n5, n95, dist, cls = manual_auroc_with_shuffle(
                event_chunk,
                base_chunk,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state,
            )
            final_auroc[i] = real_auc
            final_5th[i] = n5
            final_95th[i] = n95
            classifications[i] = cls
            best_ranges[i] = (best_start, best_end)

    results = {
        'final_auroc': final_auroc,
        'null_5th': final_5th,
        'null_95th': final_95th,
        'classification': classifications,
        'best_range': best_ranges,
    }
    return results


def compute_opto_windows_vs_hit_sliding(
    all_averages,  # list of (#trials, 2) arrays, one per neuron
    all_baseline,  # list of (#trials,) arrays, one per neuron
    all_data,
    best_start=None,
    best_end=None,
    n_perm=1000,
    random_state=None,
):
    """
    For each neuron, runs the label-shuffle test of event (column 0 of
    all_averages[i]) vs. comparison values (column 0 of all_baseline[i], e.g. hit
    trials) on the trial range [best_start[i], best_end[i]). If no ranges are
    given, all trials are used. No NaN filtering is applied here.

    Returns
    -------
    results : dict
      {
         'final_auroc':    np.array of shape (n_neurons,),
         'null_5th':       np.array(...),
         'null_95th':      np.array(...),
         'classification': list of strings ('above_95' / 'below_5' / 'ns')
         'best_range':     list of tuples (start_idx, end_idx)
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    final_auroc = np.full(n_neurons, np.nan)
    final_5th = np.full(n_neurons, np.nan)
    final_95th = np.full(n_neurons, np.nan)
    classifications = [None] * n_neurons
    best_ranges = [None] * n_neurons

    # No ranges given: use all trials
    if best_start is None or best_end is None:
        best_start = [0] * n_neurons
        best_end = [len(all_averages[i]) for i in range(n_neurons)]

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, n_windows); column 0 is the event window
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]
        # if arr_2d.shape[0] != len(baseline_vals):
        #     print(f"Warning: mismatch in trial counts for neuron {i}!")
        #     continue

        event_full = arr_2d[:, 0]

        # NaN filtering (not used):
        # mask = (~np.isnan(event_full)) & (~np.isnan(baseline_vals))
        event_vals = event_full  # [mask]
        base_vals = baseline_vals[:, 0]  # [mask]

        # Label-shuffle test on the given trial range
        if best_end[i] > best_start[i]:
            event_chunk = event_vals[best_start[i] : best_end[i]]
            base_chunk = base_vals[best_start[i] : best_end[i]]

            real_auc, n5, n95, dist, cls = manual_auroc_with_shuffle(
                event_chunk,
                base_chunk,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state,
            )
            final_auroc[i] = real_auc
            final_5th[i] = n5
            final_95th[i] = n95
            classifications[i] = cls
            best_ranges[i] = (best_start[i], best_end[i])

    results = {
        'final_auroc': final_auroc,
        'null_5th': final_5th,
        'null_95th': final_95th,
        'classification': classifications,
        'best_range': best_ranges,
    }
    return results


# ---------------------------------------------------------------------------
# False-alarm comparisons
# ---------------------------------------------------------------------------


def compute_opto_windows_vs_baseline_FA(
    all_averages,  # list of shape (#neurons,) each -> (#trials, 2)
    all_baseline,  # list of shape (#neurons,) each -> (#trials,)
    all_data,
    n_perm=1000,
    random_state=None,
):
    """
    For each neuron i: label-shuffle test (manual_auroc_with_shuffle_FA) of the
    event window (column 0 of all_averages[i]) vs. all_baseline[i].

    Returns
    -------
    results : dict
      {
        'aurocA': np.array of shape (n_neurons,),
        'pvalA':  list of classifications ('above_95' / 'below_5' / 'ns'),
        'all5':   list of lower null percentiles,
        'all95':  list of upper null percentiles
      }
    """
    n_neurons = len(all_averages)
    aurocA = np.full(n_neurons, np.nan)
    pvalA = []
    null_5th_all = []
    null_95th_all = []

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, n_windows)
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]
        if arr_2d.shape[0] != len(baseline_vals):
            print(f"Warning: mismatch in trial counts for neuron {i}!")
            continue

        # event window
        windowA = arr_2d[:, 0]

        # Drop trials with NaNs in the event window or the baseline
        maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        wA_vals = windowA[maskA]
        baseA = baseline_vals[maskA]

        # Label-shuffle test if there is enough data
        if len(wA_vals) >= 2 and len(baseA) >= 2:
            aA, null_5th, null_95th, dist, pA = manual_auroc_with_shuffle_FA(
                wA_vals, baseA, neuron_data, n_perm, random_state
            )
            aurocA[i] = aA
            pvalA.append(pA)
            null_5th_all.append(null_5th)
            null_95th_all.append(null_95th)
    results = {
        'aurocA': aurocA,
        'pvalA': pvalA,
        'all5': null_5th_all,
        'all95': null_95th_all,
    }
    return results


# ---------------------------------------------------------------------------
# Subsampling balanced by trial history
# ---------------------------------------------------------------------------


def compute_balanced_subsample_auroc(
    condA_after,  # list per neuron: after-hit trials for condition A (e.g., CR)
    condA_not,  # list per neuron: not-after-hit trials for condition A
    condB_after,  # list per neuron: after-hit trials for condition B (e.g., FA)
    condB_not,  # list per neuron: not-after-hit trials for condition B
    all_data,  # DataFrame: each row is a neuron, used only to set threshold range
    n_boot=1000,
    n_steps=100,
    sample_frac=0.7,  # 70% of the smaller condition
    random_state=None,
):
    """
    For each neuron:
      1) Count trials in A and B, split by history (after vs. not).
      2) Let n_low = min(total_A, total_B). Target = floor(sample_frac * n_low).
      3) Allocate target between history strata in proportion to the smaller condition,
         but do not exceed what both conditions can supply in each stratum.
      4) Sample *the same counts* from A and B, per stratum, without replacement.
      5) Compute auROC by manual threshold sweep. Also build a label-shuffle null distribution.
    Returns:
      {
        'all_distributions': list of length n_neurons (each is array shape [n_boot]),
        'all_shuffled_distributions': same for shuffled labels,
        'pre95_auroc': np.array of 90th percentiles per neuron,
        'pre05_auroc': np.array of 10th percentiles per neuron,
        'sampled_counts': list of tuples per neuron (after_count, not_after_count, total)
      }
    """
    import numpy as np
    import pandas as pd

    if random_state is not None:
        np.random.seed(random_state)

    def _as_1d(x):
        if x is None:
            return np.array([], dtype=float)
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            arr = np.asarray(x).reshape(-1)
        else:
            arr = np.asarray(x).reshape(-1)
        # drop NaNs
        return arr[~np.isnan(arr)]

    n_neurons = len(condA_after)
    all_distributions = [None] * n_neurons
    all_shuffled_distributions = [None] * n_neurons
    pre95_auroc = np.full(n_neurons, np.nan)
    pre05_auroc = np.full(n_neurons, np.nan)
    sampled_counts = [None] * n_neurons

    for i in range(n_neurons):
        neuron_vals = all_data.iloc[i]

        A_after = _as_1d(condA_after[i]) if i < len(condA_after) else np.array([])
        A_not = _as_1d(condA_not[i]) if i < len(condA_not) else np.array([])
        B_after = _as_1d(condB_after[i]) if i < len(condB_after) else np.array([])
        B_not = _as_1d(condB_not[i]) if i < len(condB_not) else np.array([])

        nA_after, nA_not = len(A_after), len(A_not)
        nB_after, nB_not = len(B_after), len(B_not)
        nA_total = nA_after + nA_not
        nB_total = nB_after + nB_not

        if min(nA_total, nB_total) < 2:
            # not enough data on at least one side
            continue

        # Target total per condition
        target_total = int(np.floor(sample_frac * min(nA_total, nB_total)))
        if target_total < 2:
            continue

        # Caps that both conditions can supply per stratum
        cap_after = min(nA_after, nB_after)
        cap_not = min(nA_not, nB_not)
        if (cap_after + cap_not) < 2:
            continue

        # Proportions from the smaller condition
        if nA_total <= nB_total:
            p_after_small = 0.0 if nA_total == 0 else nA_after / nA_total
        else:
            p_after_small = 0.0 if nB_total == 0 else nB_after / nB_total

        # Initial allocation
        n_after = int(round(p_after_small * target_total))
        n_not = target_total - n_after

        # Clip to caps
        n_after = min(n_after, cap_after)
        n_not = min(n_not, cap_not)

        # If under target due to caps, fill the remainder where there is headroom
        while (n_after + n_not) < target_total:
            add_after = cap_after - n_after
            add_not = cap_not - n_not
            if add_after == 0 and add_not == 0:
                break
            if add_after >= add_not and n_after < cap_after:
                n_after += 1
            elif n_not < cap_not:
                n_not += 1
            else:
                break

        # Final guard
        if n_after + n_not < 2:
            continue

        dist_aurocs = np.zeros(n_boot, dtype=float)
        dist_shuf = np.zeros(n_boot, dtype=float)

        for b in range(n_boot):
            # sample without replacement per stratum and per condition
            if n_after > 0:
                idx_A_after = np.random.choice(nA_after, size=n_after, replace=False)
                idx_B_after = np.random.choice(nB_after, size=n_after, replace=False)
                A_after_s = A_after[idx_A_after]
                B_after_s = B_after[idx_B_after]
            else:
                A_after_s = np.array([])
                B_after_s = np.array([])

            if n_not > 0:
                idx_A_not = np.random.choice(nA_not, size=n_not, replace=False)
                idx_B_not = np.random.choice(nB_not, size=n_not, replace=False)
                A_not_s = A_not[idx_A_not]
                B_not_s = B_not[idx_B_not]
            else:
                A_not_s = np.array([])
                B_not_s = np.array([])

            A_samp = np.concatenate([A_after_s, A_not_s])
            B_samp = np.concatenate([B_after_s, B_not_s])

            # Compute auROC by manual threshold sweep
            au, _, _, _ = manual_threshold_auroc(A_samp, B_samp, neuron_vals, n_steps)

            # Label-shuffle null for this bootstrap draw
            labels = np.concatenate([np.ones(len(A_samp)), np.zeros(len(B_samp))])
            scores = np.concatenate([A_samp, B_samp])
            shuf_labels = np.random.permutation(labels)
            ev_shuf = scores[shuf_labels == 1]
            bs_shuf = scores[shuf_labels == 0]
            au_shuf, _, _, _ = manual_threshold_auroc(
                ev_shuf, bs_shuf, neuron_vals, n_steps
            )

            dist_aurocs[b] = au
            dist_shuf[b] = au_shuf

        all_distributions[i] = dist_aurocs
        all_shuffled_distributions[i] = dist_shuf
        pre95_auroc[i] = np.percentile(dist_aurocs, 90)
        pre05_auroc[i] = np.percentile(dist_aurocs, 10)
        sampled_counts[i] = (n_after, n_not, n_after + n_not)

    return {
        'all_distributions': all_distributions,
        'all_shuffled_distributions': all_shuffled_distributions,
        'pre95_auroc': pre95_auroc,
        'pre05_auroc': pre05_auroc,
        'sampled_counts': sampled_counts,
    }


# ---------------------------------------------------------------------------
# Condition subsampling with optional trial-history matching
# ---------------------------------------------------------------------------


def compute_opto_windows_vs_baseline_subsampling(
    all_averages,  # list of shape (#neurons,) each -> (#trials, 2)
    all_baseline,  # list of shape (#neurons,) each -> (#trials,)
    all_data,  # DataFrame of shape (#neurons, #trials)
    n_fa,  # how many trials to subsample each time
    n_steps=100,  # how many thresholds to sweep
    n_boot=1000,  # how many subsampling iterations
    random_state=42,
    # Trial-history balancing (optional)
    fa_after_hit_data=None,  # list of DataFrames for FA trials after hit
    fa_not_after_hit_data=None,  # list of DataFrames for FA trials not after hit
    cr_after_hit_data=None,  # list of DataFrames for CR trials after hit
    cr_not_after_hit_data=None,  # list of DataFrames for CR trials not after hit
):
    """
    For each neuron (wA_vals = column 0 of all_averages[i], baseline = column 0
    of all_baseline[i]), repeat n_boot times:
      1) take a random subset of n_fa trials of wA_vals (the baseline is used in full)
      2) compute the auROC (manual threshold sweep) of the subset vs. the baseline,
         and the auROC after shuffling the labels
    giving a distribution of n_boot auROCs (and of n_boot shuffled auROCs).

    If trial-history data is provided, the subsampling is instead balanced
    between "after hit" and "not after hit" trials (see the three cases below),
    so that both conditions have the same number of trials of each history type.

    Returns
    -------
    results : dict
      {
        'all_distributions': list of length #neurons,
          each entry is a 1D np.array of shape (n_boot,) with the
          auROCs from each subsample
        'pre95_auroc': np.array of shape (#neurons,), 90th percentile of each distribution
        'pre05_auroc': np.array of shape (#neurons,), 10th percentile of each distribution
        'all_shuffled_distributions': same as all_distributions, with shuffled labels
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    # distribution for each neuron
    all_distributions = [None] * n_neurons
    all_shuffled_distributions = [None] * n_neurons

    pre95_auroc = np.full(n_neurons, np.nan)
    pre05_auroc = np.full(n_neurons, np.nan)

    # If trial-history data is provided, determine how many trials to sample per history type
    use_trial_history = (
        fa_after_hit_data is not None
        and fa_not_after_hit_data is not None
        and cr_after_hit_data is not None
        and cr_not_after_hit_data is not None
    )

    if use_trial_history:
        # Count trials in each category
        n_fa_after_hit = len(fa_after_hit_data[0]) if len(fa_after_hit_data) > 0 else 0
        n_fa_not_after_hit = (
            len(fa_not_after_hit_data[0]) if len(fa_not_after_hit_data) > 0 else 0
        )
        n_cr_after_hit = len(cr_after_hit_data[0]) if len(cr_after_hit_data) > 0 else 0
        n_cr_not_after_hit = (
            len(cr_not_after_hit_data[0]) if len(cr_not_after_hit_data) > 0 else 0
        )

        print(
            f"FA trials: {n_fa_after_hit} after hit, {n_fa_not_after_hit} not after hit"
        )
        print(
            f"CR trials: {n_cr_after_hit} after hit, {n_cr_not_after_hit} not after hit"
        )

        # Determine sampling strategy based on trial counts
        if (
            n_fa_not_after_hit <= n_cr_not_after_hit
            and n_fa_after_hit <= n_cr_after_hit
        ):
            # Case 1: FA has fewer trials in both categories - sample from CR to match FA
            sampling_case = 1
            n_sample_cr_not_after = n_fa_not_after_hit
            n_sample_cr_after = n_fa_after_hit
            print(
                f"Case 1: Sampling from CR to match FA - {n_sample_cr_after} after, {n_sample_cr_not_after} not after"
            )

        elif (
            n_cr_not_after_hit <= n_fa_not_after_hit
            and n_cr_after_hit <= n_fa_after_hit
        ):
            # Case 2: CR has fewer trials in both categories - sample from FA to match CR
            sampling_case = 2
            n_sample_fa_not_after = n_cr_not_after_hit
            n_sample_fa_after = n_cr_after_hit
            print(
                f"Case 2: Sampling from FA to match CR - {n_sample_fa_after} after, {n_sample_fa_not_after} not after"
            )

        else:
            # Case 3: Mixed case - use the smaller number for each category
            sampling_case = 3
            n_sample_cr_not_after = min(n_fa_not_after_hit, n_cr_not_after_hit)
            n_sample_cr_after = min(n_fa_after_hit, n_cr_after_hit)
            n_sample_fa_not_after = min(n_fa_not_after_hit, n_cr_not_after_hit)
            n_sample_fa_after = min(n_fa_after_hit, n_cr_after_hit)
            print(
                f"Case 3: Mixed case - sampling from both CR and FA - {n_sample_cr_after} after, {n_sample_cr_not_after} not after"
            )

        # Calculate total trials to sample
        if sampling_case == 1:
            total_trials_to_sample = n_sample_cr_after + n_sample_cr_not_after
        elif sampling_case == 2:
            total_trials_to_sample = n_sample_fa_after + n_sample_fa_not_after
        else:  # Case 3
            total_trials_to_sample = (
                n_sample_cr_after
                + n_sample_cr_not_after
                + n_sample_fa_after
                + n_sample_fa_not_after
            )
        print(f"Total trials to sample: {total_trials_to_sample}")

    for i in range(n_neurons):
        neuron_vals = all_data.iloc[i]
        arr_2d = all_averages[
            i
        ]  # shape (#trials, n_windows); column 0 is the event window
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]

        # event window
        windowA = arr_2d[:, 0]

        # NaN mask (computed but not applied: all trials are used)
        maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        wA_vals = windowA
        baseA = baseline_vals[:, 0]
        n_avail = len(wA_vals)
        if n_avail < n_fa:
            print(
                f"Neuron {i}: Not enough trials ({n_avail}) to subsample {n_fa}. Skipping."
            )
            continue

        # Build the distribution of auROCs
        dist_aurocs = np.zeros(n_boot, dtype=float)
        dist_shuffled_aurocs = np.zeros(n_boot, dtype=float)

        for b in range(n_boot):
            if use_trial_history:
                # Balanced sampling based on trial history
                wA_sub_list = []
                base_sub_list = []

                if sampling_case == 1:
                    # Sample from CR "after hit" trials
                    if n_sample_cr_after > 0 and n_cr_after_hit > 0:
                        n_actual_sample_after = min(n_sample_cr_after, n_cr_after_hit)
                        after_indices = random.sample(
                            range(n_cr_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            cr_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        wA_sub_list.append(after_samples)

                    # Sample from CR "not after hit" trials
                    if n_sample_cr_not_after > 0 and n_cr_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_cr_not_after, n_cr_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_cr_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            cr_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        wA_sub_list.append(not_after_samples)

                    # Sample from FA "after hit" trials for baseline
                    if n_sample_cr_after > 0 and n_fa_after_hit > 0:
                        n_actual_sample_after = min(n_sample_cr_after, n_fa_after_hit)
                        after_indices = random.sample(
                            range(n_fa_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            fa_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        base_sub_list.append(after_samples)

                    # Sample from FA "not after hit" trials for baseline
                    if n_sample_cr_not_after > 0 and n_fa_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_cr_not_after, n_fa_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_fa_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            fa_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        base_sub_list.append(not_after_samples)

                elif sampling_case == 2:
                    # Sample from FA "after hit" trials
                    if n_sample_fa_after > 0 and n_fa_after_hit > 0:
                        n_actual_sample_after = min(n_sample_fa_after, n_fa_after_hit)
                        after_indices = random.sample(
                            range(n_fa_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            fa_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        wA_sub_list.append(after_samples)

                    # Sample from FA "not after hit" trials
                    if n_sample_fa_not_after > 0 and n_fa_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_fa_not_after, n_fa_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_fa_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            fa_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        wA_sub_list.append(not_after_samples)

                    # Sample from CR "after hit" trials for baseline
                    if n_sample_fa_after > 0 and n_cr_after_hit > 0:
                        n_actual_sample_after = min(n_sample_fa_after, n_cr_after_hit)
                        after_indices = random.sample(
                            range(n_cr_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            cr_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        base_sub_list.append(after_samples)

                    # Sample from CR "not after hit" trials for baseline
                    if n_sample_fa_not_after > 0 and n_cr_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_fa_not_after, n_cr_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_cr_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            cr_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        base_sub_list.append(not_after_samples)

                else:  # Case 3: Mixed case - sample from both CR and FA
                    # For events (CR): sample the smaller number from each category
                    if n_sample_cr_after > 0 and n_cr_after_hit > 0:
                        n_actual_sample_after = min(n_sample_cr_after, n_cr_after_hit)
                        after_indices = random.sample(
                            range(n_cr_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            cr_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        wA_sub_list.append(after_samples)

                    if n_sample_cr_not_after > 0 and n_cr_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_cr_not_after, n_cr_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_cr_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            cr_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        wA_sub_list.append(not_after_samples)

                    # For baseline (FA): sample the smaller number from each category
                    if n_sample_fa_after > 0 and n_fa_after_hit > 0:
                        n_actual_sample_after = min(n_sample_fa_after, n_fa_after_hit)
                        after_indices = random.sample(
                            range(n_fa_after_hit), n_actual_sample_after
                        )
                        after_samples = (
                            fa_after_hit_data[i].iloc[after_indices, 0].values
                        )
                        base_sub_list.append(after_samples)

                    if n_sample_fa_not_after > 0 and n_fa_not_after_hit > 0:
                        n_actual_sample_not_after = min(
                            n_sample_fa_not_after, n_fa_not_after_hit
                        )
                        not_after_indices = random.sample(
                            range(n_fa_not_after_hit), n_actual_sample_not_after
                        )
                        not_after_samples = (
                            fa_not_after_hit_data[i].iloc[not_after_indices, 0].values
                        )
                        base_sub_list.append(not_after_samples)

                # Combine the samples
                if wA_sub_list:
                    wA_sub = np.concatenate(wA_sub_list)
                else:
                    # no history samples: fall back to a random subset of size n_fa
                    sub_idxs = random.sample(range(n_avail), k=n_fa)
                    wA_sub = wA_vals[sub_idxs]

                if base_sub_list:
                    base_sub = np.concatenate(base_sub_list)
                else:
                    # no history samples: use the full baseline
                    base_sub = baseA
            else:
                # Without trial history: random subset of size n_fa
                sub_idxs = random.sample(range(n_avail), k=n_fa)
                wA_sub = wA_vals[sub_idxs]
                base_sub = baseA

            # Both sides must have the same number of trials
            if len(wA_sub) != len(base_sub):
                print(f"Warning: mismatch in trial counts for neuron {i}!")
                continue
            # auROC of the subsample and of the label-shuffled subsample
            labels = np.concatenate([np.ones(len(wA_sub)), np.zeros(len(base_sub))])
            shuffled_labels = np.random.permutation(labels)
            scores = np.concatenate([wA_sub, base_sub])
            ev_shuf = scores[shuffled_labels == 1]
            bs_shuf = scores[shuffled_labels == 0]
            this_auroc, _, _, _ = manual_threshold_auroc(
                wA_sub, base_sub, neuron_vals, n_steps
            )
            shuffled_auroc, _, _, _ = manual_threshold_auroc(
                ev_shuf, bs_shuf, neuron_vals, n_steps
            )

            dist_aurocs[b] = this_auroc
            dist_shuffled_aurocs[b] = shuffled_auroc

        all_distributions[i] = dist_aurocs
        pre95_auroc[i] = np.percentile(dist_aurocs, 90)
        pre05_auroc[i] = np.percentile(dist_aurocs, 10)
        all_shuffled_distributions[i] = dist_shuffled_aurocs

    results = {
        'all_distributions': all_distributions,
        'pre95_auroc': pre95_auroc,
        'pre05_auroc': pre05_auroc,
        'all_shuffled_distributions': all_shuffled_distributions,
    }
    return results


# ---------------------------------------------------------------------------
# Manual ROC calculation and label-shuffle tests
# ---------------------------------------------------------------------------


def manual_threshold_auroc(event_vals, baseline_vals, all_data, n_steps=100):
    """
    Compute auROC by manually sweeping a threshold over the amplitude range of
    the neuron (0.1th to 99.9th percentile of all_data) in n_steps steps.

    event_vals: 1D array of event amplitudes
    baseline_vals: 1D array of baseline amplitudes
    all_data: all values of the neuron, used only to set the threshold range
    n_steps: how many threshold increments to use (default 100)

    Returns
    -------
    auroc: float in [0,1], area under the ROC curve
           (prob(Event>T) vs. prob(Baseline>T)).
    thresholds: array of shape (n_steps+1,), the actual thresholds tested
    tpr: array of shape (n_steps+1,), each TPR = fraction of event_vals > threshold
    fpr: array of shape (n_steps+1,), each FPR = fraction of baseline_vals > threshold
    """
    # threshold range: 0.1th-99.9th percentile of the neuron's values
    min_val = np.nanpercentile(all_data, 0.1, axis=0)
    max_val = np.nanpercentile(all_data, 99.9, axis=0)
    # min_val = np.nanmin(all_data)
    # max_val = np.nanmax(all_data)
    if min_val == max_val:
        # everything is identical => auROC = 0.5
        return 0.5, np.array([min_val]), np.array([0.0]), np.array([0.0])

    # thresholds in n_steps increments
    step = (max_val - min_val) / n_steps
    thresholds = np.arange(n_steps + 1) * step + min_val

    nE = len(event_vals)
    nB = len(baseline_vals)

    tpr = np.zeros(len(thresholds), dtype=float)  # True Positive Rate
    fpr = np.zeros(len(thresholds), dtype=float)  # False Positive Rate

    for i, thr in enumerate(thresholds):
        # Probability that event > thr
        tpr[i] = np.sum(event_vals > thr) / nE
        # Probability that baseline > thr (the "false positive" for that threshold)
        fpr[i] = np.sum(baseline_vals > thr) / nB

    # Integrate TPR over FPR with the trapezoid rule. With thresholds increasing,
    # TPR and FPR decrease, so reverse them to have FPR ascending (0 -> 1).
    if fpr[0] > fpr[-1]:
        tpr = tpr[::-1]
        fpr = fpr[::-1]

    # area under TPR vs FPR
    auroc = np.trapz(tpr, x=fpr)

    return auroc, thresholds, tpr, fpr


def manual_auroc_with_shuffle(
    event_vals, baseline_vals, all_vals, n_perm=1000, n_steps=100, random_state=None
):
    """
    Label-shuffle test of the auROC:
      1) Compute the real auROC by manual threshold sweep
      2) Shuffle event/baseline labels n_perm times -> null distribution
      3) Compare the real auROC to the 10th and 90th percentiles of the null

    Returns
    -------
    real_auc : float
    null_5th : float
       10th percentile of perm_distribution (name is historical)
    null_95th : float
       90th percentile of perm_distribution (name is historical)
    perm_distribution : array of shape (n_perm,)
    classification : str
       'above_95' (real > 90th percentile), 'below_5' (real < 10th percentile) or 'ns'
    If there is not enough data, (nan, empty array, nan, False) is returned instead.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Remove NaNs
    event_vals = event_vals[~np.isnan(event_vals)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    if len(event_vals) < 2 or len(baseline_vals) < 2:
        return np.nan, np.array([]), np.nan, False

    # 1) Real auROC
    real_auc, _, _, _ = manual_threshold_auroc(
        event_vals, baseline_vals, all_vals, n_steps
    )

    # 2) Build label-shuffled null distribution
    nE = len(event_vals)
    nB = len(baseline_vals)
    labels = np.concatenate([np.ones(nE), np.zeros(nB)])
    all_data = np.concatenate([event_vals, baseline_vals])

    perm_distribution = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        # Shuffle event/baseline identity
        shuffled_labels = np.random.permutation(labels)
        # split back into "event" and "baseline"
        event_mask = shuffled_labels == 1
        base_mask = shuffled_labels == 0
        ev_shuf = all_data[event_mask]
        bs_shuf = all_data[base_mask]
        # auROC of the shuffled sets
        auc_i, _, _, _ = manual_threshold_auroc(ev_shuf, bs_shuf, all_vals, n_steps)
        perm_distribution[i] = auc_i

    # 3) Compare real_auc to the 90th / 10th percentiles
    null_95th = np.percentile(perm_distribution, 90)
    null_5th = np.percentile(perm_distribution, 10)
    # Classify
    if real_auc > null_95th:
        classification = "above_95"
    elif real_auc < null_5th:
        classification = "below_5"
    else:
        classification = "ns"  # not significant

    return real_auc, null_5th, null_95th, perm_distribution, classification


def manual_auroc_with_shuffle_FA(
    event_vals, baseline_vals, all_vals, n_perm=1000, n_steps=100, random_state=None
):
    """
    Label-shuffle test of the auROC:
      1) Compute the real auROC by manual threshold sweep
      2) Shuffle event/baseline labels n_perm times -> null distribution
      3) Compare the real auROC to the 10th and 90th percentiles of the null

    Returns
    -------
    real_auc : float
    null_5th : float
       10th percentile of perm_distribution (name is historical)
    null_95th : float
       90th percentile of perm_distribution (name is historical)
    perm_distribution : array of shape (n_perm,)
    classification : str
       'above_95' (real > 90th percentile), 'below_5' (real < 10th percentile) or 'ns'
    If there is not enough data, (nan, empty array, nan, False) is returned instead.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Remove NaNs
    event_vals = event_vals[~np.isnan(event_vals)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    if len(event_vals) < 2 or len(baseline_vals) < 2:
        return np.nan, np.array([]), np.nan, False

    # 1) Real auROC
    real_auc, _, _, _ = manual_threshold_auroc(
        event_vals, baseline_vals, all_vals, n_steps
    )

    # 2) Build label-shuffled null distribution
    nE = len(event_vals)
    nB = len(baseline_vals)
    labels = np.concatenate([np.ones(nE), np.zeros(nB)])
    all_data = np.concatenate([event_vals, baseline_vals])

    perm_distribution = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        # Shuffle event/baseline identity
        shuffled_labels = np.random.permutation(labels)
        # split back into "event" and "baseline"
        event_mask = shuffled_labels == 1
        base_mask = shuffled_labels == 0
        ev_shuf = all_data[event_mask]
        bs_shuf = all_data[base_mask]
        # auROC of the shuffled sets
        auc_i, _, _, _ = manual_threshold_auroc(ev_shuf, bs_shuf, all_vals, n_steps)
        perm_distribution[i] = auc_i

    # 3) Compare real_auc to the 90th / 10th percentiles
    null_95th = np.percentile(perm_distribution, 90)
    null_5th = np.percentile(perm_distribution, 10)
    # Classify
    if real_auc > null_95th:
        classification = "above_95"
    elif real_auc < null_5th:
        classification = "below_5"
    else:
        classification = "ns"  # not significant

    return real_auc, null_5th, null_95th, perm_distribution, classification


# ---------------------------------------------------------------------------
# Session configuration, input loading, and analysis execution
# ---------------------------------------------------------------------------

if '__main__' == __name__:
    # for rep in range(1, 6):
    # Session to analyze
    mouse = 'mouse16'
    day = ['sess14']
    ROOT_PATH = DATA_ROOT + r'\{}\{}\ML\data'.format(mouse, day[0])
    licking_data = DATA_ROOT + r'\{}\{}\ML\licking_analysis_final'.format(mouse, day[0])
    data_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_z_no_sub_history'.format(
        mouse, day[0]
    )
    save_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_z_no_sub_history'.format(
        mouse, day[0]
    )
    os.makedirs(save_loc, exist_ok=True)
    # Load the event data (outputs of creat_events_data_per_neuron.py)
    with open(join(data_loc, "event_data_light.pkl"), 'rb') as f:
        event_data_light = pickle.load(f)
    with open(join(data_loc, "event_data_correct_reject.pkl"), 'rb') as f:
        event_data_correct_reject = pickle.load(f)
    with open(join(data_loc, "event_data_hit.pkl"), 'rb') as f:
        event_data_hit = pickle.load(f)
    with open(join(data_loc, "event_data_false_alarm.pkl"), 'rb') as f:
        event_data_false_alarm = pickle.load(f)
    with open(join(data_loc, "event_data_miss.pkl"), 'rb') as f:
        event_data_miss = pickle.load(f)
    with open(join(data_loc, "light_averages.pkl"), 'rb') as f:
        all_light_averages = pickle.load(f)
    with open(join(data_loc, 'cr_light_averages.pkl'), 'rb') as f:
        cr_event_averages = pickle.load(f)
    with open(join(data_loc, 'fa_light_averages.pkl'), 'rb') as f:
        fa_event_averages = pickle.load(f)
    with open(join(data_loc, 'hit_light_averages.pkl'), 'rb') as f:
        hit_event_averages = pickle.load(f)
    with open(join(data_loc, 'miss_light_averages.pkl'), 'rb') as f:
        miss_event_averages = pickle.load(f)
    with open(join(data_loc, 'all_values.pkl'), 'rb') as f:
        all_values = pickle.load(f)
    with open(join(data_loc, 'correct_reject.pkl'), 'rb') as f:
        correct_reject = pickle.load(f)
    with open(join(data_loc, 'hit.pkl'), 'rb') as f:
        hit = pickle.load(f)
    with open(join(data_loc, 'false_alarm.pkl'), 'rb') as f:
        false_alarm = pickle.load(f)
    with open(join(data_loc, 'miss.pkl'), 'rb') as f:
        miss = pickle.load(f)

    # Load trial history data
    try:
        with open(join(data_loc, 'fa_after_hit_data.pkl'), 'rb') as f:
            fa_after_hit_data = pickle.load(f)
        with open(join(data_loc, 'fa_not_after_hit_data.pkl'), 'rb') as f:
            fa_not_after_hit_data = pickle.load(f)
        with open(join(data_loc, 'cr_after_hit_data.pkl'), 'rb') as f:
            cr_after_hit_data = pickle.load(f)
        with open(join(data_loc, 'cr_not_after_hit_data.pkl'), 'rb') as f:
            cr_not_after_hit_data = pickle.load(f)
        print("Successfully loaded trial history data")
    except FileNotFoundError:
        print("Trial history data files not found. Using original sampling method.")
        fa_after_hit_data = None
        fa_not_after_hit_data = None
        cr_after_hit_data = None
        cr_not_after_hit_data = None

    # Trial-wise mean activity in the reward and lick-onset windows (hit trials)
    avg_activity_reward = []
    for neuron in event_data_hit['reward']['event']:
        avg_activity_reward.append(
            np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1)
        )
    ave_activity_lick_on = []
    for neuron in event_data_hit['lick_onset']['event']:
        ave_activity_lick_on.append(
            np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1)
        )

    # Compare the reward / lick-onset response to the distractor light (control)
    # res_reward_vs_lightHit = compute_opto_windows_vs_hit_sliding(
    #     avg_activity_reward,
    #     hit_event_averages,
    #     all_values,
    #     None,
    #     None,
    #     n_perm=500,
    #     random_state=42
    # )
    # res_lick_on_vs_lightHit = compute_opto_windows_vs_hit_sliding(
    #     ave_activity_lick_on,
    #     hit_event_averages,
    #     all_values,
    #     None,
    #     None,
    #     n_perm=500,
    #     random_state=42
    # )

    # auROC_results_cr = compute_auROC_for_events_with_permutation(
    #     event_data_correct_reject,
    #     all_values,
    #     n_perm=500,
    #     random_state=42
    # )
    # auROC_results_hit = compute_auROC_for_events_with_permutation(
    #     event_data_hit,
    #     all_values,
    #     n_perm=500,
    #     random_state=42
    # )
    # auROC_results_fa = compute_auROC_for_events_with_permutation(
    #     event_data_false_alarm,
    #     all_values,
    #     n_perm=500,
    #     random_state=42
    # )
    # auROC_results_miss = compute_auROC_for_events_with_permutation(
    #     event_data_miss,
    #     all_values,
    #     n_perm=500,
    #     random_state=42
    # )
    #
    # all_baseline_cr = event_data_correct_reject['light_off']['baseline']
    # baseline_mean = [np.asarray(all_baseline_cr[i]) for i in range(len(all_baseline_cr))]
    # baseline_mean = [np.nanmean(baseline_mean[i], axis=1) for i in range(len(baseline_mean))]
    #
    # res = compute_opto_windows_vs_baseline_sliding(
    #     cr_event_averages,
    #     baseline_mean,
    #     all_values,
    #     window_size=int(len(correct_reject)/2),
    #     step=5,
    #     n_perm=1000,
    #     random_state=42
    # )
    # #
    # Light trials: opto window vs. baseline (sliding trial window), then
    # light vs. distractor-light-only (hit) trials on the selected trial range
    all_baseline_light = event_data_light['light_off']['baseline']
    baseline_mean = [
        np.asarray(all_baseline_light[i]) for i in range(len(all_baseline_light))
    ]
    baseline_mean = [
        np.nanmean(baseline_mean[i], axis=1) for i in range(len(baseline_mean))
    ]
    #
    res_all_light = compute_opto_windows_vs_baseline_sliding(
        all_light_averages,
        baseline_mean,
        all_values,
        window_size=int(0.4 * len(all_light_averages[0])),
        step=5,
        n_perm=500,
        random_state=42,
    )
    res_all_light_vs_lightHit = compute_opto_windows_vs_hit_sliding(
        all_light_averages,
        hit_event_averages,
        all_values,
        [
            res_all_light['best_range'][i][0]
            for i in range(len(res_all_light['best_range']))
        ],
        [
            res_all_light['best_range'][i][1]
            for i in range(len(res_all_light['best_range']))
        ],
        n_perm=500,
        random_state=42,
    )
    # # # compare the opto-stimulation period to the distractor light only
    # res_CR_vs_lightHit = compute_opto_windows_vs_hit_sliding(
    #     cr_event_averages,
    #     hit_event_averages,
    #     all_values,
    #     [res['best_range'][i][0] for i in range(len(res['best_range']))],
    #     [res['best_range'][i][1] for i in range(len(res['best_range']))],
    #     n_perm=1000,
    #     random_state=42
    # )

    # for the sated condition
    # res_CR_vs_lightHit = compute_opto_windows_vs_baseline_subsampling(
    #     cr_event_averages,
    #     hit_event_averages,
    #     all_values,
    #     n_fa=len(hit_event_averages[0]),
    #     n_steps=100,
    #     n_boot=1000,
    #     random_state=42
    # )
    #
    # all_baseline_fa = event_data_false_alarm['light_off']['baseline']
    # if len(all_baseline_fa[0]) == 0:
    #     print("No baseline data available for false alarm events.")
    # else:
    #     baseline_mean_fa = [np.asarray(all_baseline_fa[i]) for i in range(len(all_baseline_fa))]
    #     baseline_mean_fa = [np.nanmean(baseline_mean_fa[i], axis=1) for i in range(len(baseline_mean_fa))]
    #     res_fa = compute_opto_windows_vs_baseline_FA(
    #         fa_event_averages,
    #         baseline_mean_fa,
    #         all_values,
    #         n_perm=1000,
    #         random_state=42
    #     )
    # CR vs. FA: subsample CR to the number of FA trials (balanced by trial history:
    # after hit / not after hit) to get the distribution of CR-vs-FA auROCs
    # N_FA = len(fa_event_averages[0])
    # res_cr_subsample = compute_opto_windows_vs_baseline_subsampling(
    #     cr_event_averages,
    #     fa_event_averages,
    #     all_values,
    #     n_fa=N_FA,
    #     n_steps=100,
    #     n_boot=500,
    #     random_state=42,
    #     fa_after_hit_data=fa_after_hit_data,
    #     fa_not_after_hit_data=fa_not_after_hit_data,
    #     cr_after_hit_data=cr_after_hit_data,
    #     cr_not_after_hit_data=cr_not_after_hit_data
    # )
    # res_cr_vs_fa_bal = compute_balanced_subsample_auroc(
    #     cr_after_hit_data,  # A after-hit
    #     cr_not_after_hit_data,  # A not-after
    #     fa_after_hit_data,  # B after-hit
    #     fa_not_after_hit_data,  # B not-after
    #     all_values,
    #     n_boot=100,
    #     n_steps=100,
    #     sample_frac=0.7,
    #     random_state=42
    # )
    # with open(join(data_loc, 'res_fa.pkl'), 'wb') as f:
    #     pickle.dump(res_fa, f)
    # with open(join(save_loc, 'res_cr_subsample_v2.pkl'), 'wb') as f:
    #     pickle.dump(res_cr_subsample, f)
    # #
    # Save the results
    # with open(join(data_loc, 'auROC_results_cr.pkl'), 'wb') as f:
    #     pickle.dump(auROC_results_cr, f)
    # with open(join(data_loc, 'auROC_results_hit.pkl'), 'wb') as f:
    #     pickle.dump(auROC_results_hit, f)
    # with open(join(data_loc, 'auROC_results_fa.pkl'), 'wb') as f:
    #     pickle.dump(auROC_results_fa, f)
    # with open(join(data_loc, 'auROC_results_miss.pkl'), 'wb') as f:
    #     pickle.dump(auROC_results_miss, f)
    # with open(join(save_loc, 'res_all_light.pkl'), 'wb') as f:
    #     pickle.dump(res_all_light, f)
    with open(join(save_loc, 'res_all_light_vs_lightHit.pkl'), 'wb') as f:
        pickle.dump(res_all_light_vs_lightHit, f)
    # with open(join(data_loc, 'res.pkl'), 'wb') as f:
    #     pickle.dump(res, f)

    # with open(join(save_loc, 'res_CR_vs_lightHit.pkl'), 'wb') as f:
    #     pickle.dump(res_CR_vs_lightHit, f)
    # with open(join(data_loc, 'res_reward_vs_lightHit.pkl'), 'wb') as f:
    #     pickle.dump(res_reward_vs_lightHit, f)
    # with open(join(data_loc, 'res_lick_on_vs_lightHit.pkl'), 'wb') as f:
    #     pickle.dump(res_lick_on_vs_lightHit, f)
