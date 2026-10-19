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

start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000
pre_trial_length = 4 * rec_hz
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz

def plot_increase_decrease_bar(sig_neurons, total_neurons,Title):
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
        inc_counts.append((len(sig_neurons[e]['increase'])/total_neurons)*100)
        dec_counts.append((len(sig_neurons[e]['decrease'])/total_neurons)*100)

    x = np.arange(n_events)  # positions

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_inc = ax.bar(x - width / 2, inc_counts, width, label='Increase', color='tomato')
    bar_dec = ax.bar(x + width / 2, dec_counts, width, label='Decrease', color='cornflowerblue')

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
    """
    # We assume significant_neurons[e]['increase'] is a list of neuron indices
    # Similarly for 'decrease'.

    for i in range(len(events_order) - 1):  # from 0 to next-to-last
        e1 = events_order[i]
        e2 = events_order[i+1]

        # find intersection of the 'increase' sets
        inc_e1 = set(significant_neurons[e1]['increase'])
        inc_e2 = set(significant_neurons[e2]['increase'])
        overlap_inc = inc_e1.intersection(inc_e2)

        # remove them from e2
        if len(overlap_inc) > 0:
            # create a new array that excludes overlap_inc
            new_e2_inc = [n for n in significant_neurons[e2]['increase'] if n not in overlap_inc]
            significant_neurons[e2]['increase'] = np.array(new_e2_inc, dtype=int)

        # do the same for 'decrease' if you want:
        dec_e1 = set(significant_neurons[e1]['decrease'])
        dec_e2 = set(significant_neurons[e2]['decrease'])
        overlap_dec = dec_e1.intersection(dec_e2)

        if len(overlap_dec) > 0:
            new_e2_dec = [n for n in significant_neurons[e2]['decrease'] if n not in overlap_dec]
            significant_neurons[e2]['decrease'] = np.array(new_e2_dec, dtype=int)

    return significant_neurons
def plot_venn(set1, set2, total, labels, title, filename, save_loc, color1='gold', color2='lightpink'):
    plt.figure()
    v = venn2([set1, set2], set_labels=labels, set_colors=(color1, color2))
    v.get_label_by_id('10').set_text(f"{(len(set1 - set2) / total * 100):.1f}%")
    v.get_label_by_id('01').set_text(f"{(len(set2 - set1) / total * 100):.1f}%")
    # if there is no intersection
    if len(set1 & set2) > 0:
        v.get_label_by_id('11').set_text(f"{(len(set1 & set2) / total * 100):.1f}%")
    plt.title(title)
    plt.savefig(join(save_loc, filename))
    plt.show()

def auroc_permutation_test(event_vals, baseline_vals, n_perm=1000, random_state=None):
    """
    Computes the real auROC (event vs. baseline) for one neuron
    and a permutation-based p-value by shuffling labels.

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
    p_value : float
        Two-sided p-value for how often permuted auROCs lie at least as far
        from 0.5 as the real auROC.
    perm_distribution : 1D array
        The array of permuted auROCs (length n_perm).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Clean out NaNs just in case
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

    # ----- 3) Compute the distribution of the auROC shuffled values and check where the real is falling -----
    # 5th and 95th percentiles
    null_5th  = np.percentile(perm_distribution, 10)
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
    all_event_data,
    all_data,
    n_perm=1000,
    random_state=None
):
    """
    Similar to compute_auROC_for_events, but also does a permutation-based
    significance test for each neuron/event.

    Returns
    -------
    results : dict
      results[event_name] = {
         'auroc': np.array of shape (n_neurons,),
         'p_value': np.array of shape (n_neurons,)
      }
    """
    from sklearn.metrics import roc_auc_score

    results = {}

    for event_name, event_dict in all_event_data.items():
        if 'event' not in event_dict or 'baseline' not in event_dict:
            print(f"Skipping {event_name} - no event/baseline data.")
            continue

        event_data_all_neurons    = event_dict['event']
        baseline_data_all_neurons = event_dict['baseline']

        n_neurons = len(event_data_all_neurons)
        aurocs   = np.full(n_neurons, np.nan)
        all5     = np.full(n_neurons, np.nan)
        all95    = np.full(n_neurons, np.nan)
        distribution = [None]*n_neurons
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
            base_vals  = baseline_means[valid_mask]

            if len(event_vals) < 2 or len(base_vals) < 2:
                continue

            # call our permutation test function
            real_auc, null5, null95, dist, cls = manual_auroc_with_shuffle(
                event_vals,
                base_vals,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state
            )
            aurocs[neuron_i] = real_auc
            p_values.append(cls)
            all5[neuron_i] = null5
            all95[neuron_i] = null95
            distribution[neuron_i] = dist

        results[event_name] = {
            'auroc': aurocs,
            'p_value': p_values,
            'all5': all5,
            'all95': all95,
            'distribution': distribution
        }

    return results

def compute_opto_windows_vs_hit_sliding_psycho(
    all_averages,     # list of (#trials, 2) arrays, one per neuron
    all_baseline,     # list of (#trials,) arrays, one per neuron
    all_data,
    best_start=None,
    best_end=None,
    n_perm=1000,
    random_state=None
):
    """
    For each neuron:
      1) We'll do a sliding window of 'window_size' across valid trials,
         moving in steps of 'step' (e.g. 40 trials at a time, shift by 10).
      2) Compute auROC for each subrange.
      3) Pick the subrange that yields the highest auROC.
      4) Do a permutation test on that subrange only.
      5) Store final results in 'aurocA[i]', 'classification[i]', etc.

    Returns
    -------
    results : dict
      {
         'final_auroc':    np.array of shape (n_neurons,),
         'null_5th':       np.array(...),
         'null_95th':      np.array(...),
         'classification': list of strings (like 'above_95','ns',etc.)
         'best_range':     list of tuples (start_idx, end_idx)
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    final_auroc = np.full(n_neurons, np.nan)
    final_5th   = np.full(n_neurons, np.nan)
    final_95th  = np.full(n_neurons, np.nan)
    classifications = [None]*n_neurons
    best_ranges = [None]*n_neurons
    distribution = [None]*n_neurons
    # Handle None case: Initialize best_start and best_end if they are None
    if best_start is None or best_end is None:
        best_start = [0] * n_neurons  # Default start index at 0
        best_end = [len(all_averages[i]) for i in range(n_neurons)]  # Full range

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, 2) => col0 = event, col1= ...
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue  # or raise an error

        baseline_vals = all_baseline[i]
        # if arr_2d.shape[0] != len(baseline_vals):
        #     print(f"Warning: mismatch in trial counts for neuron {i}!")
        #     continue

        event_full = arr_2d[:, 0]

        # Filter out trials with NaNs in event or baseline
        # mask = (~np.isnan(event_full)) & (~np.isnan(baseline_vals))
        event_vals = event_full#[mask]
        base_vals  = baseline_vals[:, 0]#[mask]


        # We have the best subrange saved in best_start, best_end
        # 2) Do the permutation test on that subrange only
        if best_end[i] > best_start[i]:
            event_chunk = event_vals[best_start[i]:best_end[i]]
            base_chunk  = base_vals[best_start[i]:best_end[i]]

            real_auc, n5, n95, dist, cls = manual_auroc_with_shuffle(
                event_chunk,
                base_chunk,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state
            )
            final_auroc[i] = real_auc
            final_5th[i]   = n5
            final_95th[i]  = n95
            classifications[i] = cls
            best_ranges[i] = (best_start[i], best_end[i])
            distribution[i] = dist

    results = {
        'final_auroc':    final_auroc,
        'null_5th':       final_5th,
        'null_95th':      final_95th,
        'classification': classifications,
        'best_range':     best_ranges,
        'distribution':   distribution
    }
    return results


def compute_opto_windows_vs_hit_sliding_psycho_FA(
    all_averages,     # list of (#trials, 2) arrays, one per neuron
    all_baseline,     # list of (#trials,) arrays, one per neuron
    all_data,
    best_start,
    best_end,
    n_perm=1000,
    random_state=None
):
    """
    For each neuron:
      1) We'll do a sliding window of 'window_size' across valid trials,
         moving in steps of 'step' (e.g. 40 trials at a time, shift by 10).
      2) Compute auROC for each subrange.
      3) Pick the subrange that yields the highest auROC.
      4) Do a permutation test on that subrange only.
      5) Store final results in 'aurocA[i]', 'classification[i]', etc.

    Returns
    -------
    results : dict
      {
         'final_auroc':    np.array of shape (n_neurons,),
         'null_5th':       np.array(...),
         'null_95th':      np.array(...),
         'classification': list of strings (like 'above_95','ns',etc.)
         'best_range':     list of tuples (start_idx, end_idx)
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    final_auroc = np.full(n_neurons, np.nan)
    final_5th   = np.full(n_neurons, np.nan)
    final_95th  = np.full(n_neurons, np.nan)
    classifications = [None]*n_neurons
    best_ranges = [None]*n_neurons
    distribution = [None]*n_neurons
    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, 2) => col0 = event, col1= ...
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue  # or raise an error

        baseline_vals = all_baseline[i]
        # if arr_2d.shape[0] != len(baseline_vals):
        #     print(f"Warning: mismatch in trial counts for neuron {i}!")
        #     continue

        event_full = arr_2d[:, 0]

        # Filter out trials with NaNs in event or baseline
        # mask = (~np.isnan(event_full)) & (~np.isnan(baseline_vals))
        event_vals = event_full#[mask]
        base_vals  = baseline_vals[:, 0]#[mask]


        # We have the best subrange saved in best_start, best_end
        # 2) Do the permutation test on that subrange only
        if best_end[i] > best_start[i]:
            event_chunk = event_vals[best_start[i]:best_end[i]]
            base_chunk  = base_vals[best_start[i]:best_end[i]]

            real_auc, n5, n95, dist, cls = manual_auroc_with_shuffle(
                event_chunk,
                base_chunk,
                neuron_data,
                n_perm=n_perm,
                random_state=random_state
            )   
            final_auroc[i] = real_auc
            final_5th[i]   = n5
            final_95th[i]  = n95
            classifications[i] = cls
            best_ranges[i] = (best_start[i], best_end[i])
            distribution[i] = dist

    results = {
        'final_auroc':    final_auroc,
        'null_5th':       final_5th,
        'null_95th':      final_95th,
        'classification': classifications,
        'best_range':     best_ranges,
        'distribution':   distribution
    }
    return results
def compute_opto_windows_vs_baseline(
        all_averages,  # list of shape (#neurons,) each -> (#trials, 2)
        all_baseline,  # list of shape (#neurons,) each -> (#trials,)
        all_data,
        n_perm=1000,
        random_state=None
):
    """
    For each neuron i:
      1) event vs. baseline -> auROC_A, pval_A

    Returns
    -------
    results : dict
      {
        'aurocA':   np.array of shape (n_neurons,),
        'pvalA':    np.array of shape (n_neurons,),
      }
    """
    n_neurons = len(all_averages)
    aurocA = np.full(n_neurons, np.nan)
    pvalA = []
    null_5th_all = []
    null_95th_all = []
    distribution = [None]*n_neurons
    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, 2)
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue  # or raise an error if mismatch

        baseline_vals = all_baseline[i]
        baseline_vals = baseline_vals.flatten()

        # Column 0, column 1
        windowA = arr_2d[:, 0]

        # Drop trials with NaNs in either distribution
        # For window A, we also need baseline not NaN
        maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        wA_vals = windowA[maskA]
        baseA = baseline_vals[maskA]
        if wA_vals.shape[0] != len(baseA):
            print(f"Warning: mismatch in trial counts for neuron {i}!")
            continue


        # If enough data, do the permutation test
        # --- Window A vs. baseline
        if len(wA_vals) >= 10 and len(baseA) >= 10:
            aA, null_5th, null_95th, dist, pA = manual_auroc_with_shuffle_FA(wA_vals, baseA, neuron_data, n_perm, random_state)
            aurocA[i] = aA
            pvalA.append(pA)
            null_5th_all.append(null_5th)
            null_95th_all.append(null_95th)
            distribution[i] = dist
        else:
            # Not enough data to compute auROC
            continue

    results = {
        'aurocA': aurocA,
        'pvalA': pvalA,
        'all5': null_5th_all,
        'all95': null_95th_all,
        'distribution': distribution
    }
    return results
# def compute_opto_windows_vs_baseline_subsampling(
#     all_averages,    # list of shape (#neurons,) each -> (#trials, 2)
#     all_baseline,    # list of shape (#neurons,) each -> (#trials,)
#     all_data,        # DataFrame of shape (#neurons, #trials)
#     n_to_sample,            # how many trials to subsample each time
#     n_steps=100,       # how many thresholds to sweep
#     n_boot=1000,     # how many subsampling iterations
#     random_state=42
# ):
#     """
#     For each neuron, do the following:
#       - Let's define wA_vals = column 0 of all_averages[i].
#       - baseline = all_baseline[i].
#       - We assume wA_vals and baseline have the SAME # of valid trials
#         after NaN removal.
#       - For b in [1..n_boot]:
#           1) random subset of size n_fa from the available trials
#           2) compute the real auROC of that subset
#         -> build a distribution of 1000 auROCs
#       - Finally, store that distribution (or a summary) in 'results'.
#
#     Returns
#     -------
#     results : dict
#       {
#         'all_distributions': list of length #neurons,
#           each entry is a 1D np.array of shape (n_boot,) with the
#           auROCs from each subsample
#         'mean_auroc': np.array of shape (#neurons,),
#           the mean of each neuron's distribution
#         'median_auroc': np.array of shape (#neurons,),
#           the median of each neuron's distribution
#       }
#     """
#     if random_state is not None:
#         np.random.seed(random_state)
#
#     n_neurons = len(all_averages)
#     # We'll store the distribution for each neuron
#     all_distributions = [None]*n_neurons
#     all_shuffled_distributions = [None]*n_neurons
#
#     pre95_auroc = np.full(n_neurons, np.nan)
#     pre05_auroc = np.full(n_neurons, np.nan)
#
#     for i in range(n_neurons):
#         neuron_vals = all_data.iloc[i]
#         arr_2d = all_averages[i]   # shape (#trials, 2) => col0 = event, col1 = something else?
#         if i >= len(all_baseline):
#             continue
#
#         baseline_vals = all_baseline[i]
#
#         # Check trial count match
#         # if arr_2d.shape[0] != len(baseline_vals):
#         #     print(f"Warning: mismatch in trial counts for neuron {i}!")
#         #     continue
#
#         # Column 0 is your "event" window
#         windowA = arr_2d[:, 0]
#
#         # Drop NaNs
#         maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
#         # wA_vals = windowA[maskA]
#         # baseA   = baseline_vals[maskA]
#         wA_vals = windowA
#         baseA   = baseline_vals[:, 0]
#         n_avail = len(wA_vals)
#         if n_avail < n_to_sample:
#             print(f"Neuron {i}: Not enough trials ({n_avail}) to subsample {n_to_sample}. Skipping.")
#             continue
#
#         # Build the distribution of auROCs
#         dist_aurocs = np.zeros(n_boot, dtype=float)
#         dist_shuffled_aurocs = np.zeros(n_boot, dtype=float)
#
#         for b in range(n_boot):
#             # 1) random subset of size n_fa
#             # pick indices from 0..n_avail-1
#             # sub_idxs = random.sample(range(n_avail), k=n_fa)
#             sub_idxs = random.sample(range(n_to_sample), k=n_to_sample)
#             wA_sub = wA_vals#[sub_idxs]
#             base_sub = baseA[sub_idxs]
#             # base_sub = baseA
#             # Check trial count match
#             if len(wA_sub) != len(base_sub):
#                 print(f"Warning: mismatch in trial counts for neuron {i}!")
#                 continue
#             # 2) compute auROC
#             this_auroc, _, _, _ = manual_threshold_auroc(wA_sub, base_sub, neuron_vals, n_steps)
#
#             dist_aurocs[b] = this_auroc
#
#             # shuffle the labels
#             labels = np.concatenate([np.ones(len(wA_sub)), np.zeros(len(base_sub))])
#             shuffled_labels = np.random.permutation(labels)
#             scores = np.concatenate([wA_sub, base_sub])
#             ev_shuf = scores[shuffled_labels == 1]
#             bs_shuf = scores[shuffled_labels == 0]
#             auc_i, _, _, _ = manual_threshold_auroc(ev_shuf, bs_shuf, neuron_vals, n_steps)
#
#             dist_shuffled_aurocs[b] = auc_i
#
#         all_distributions[i] = dist_aurocs
#         pre95_auroc[i] = np.percentile(dist_aurocs, 90)
#         pre05_auroc[i] = np.percentile(dist_aurocs, 10)
#         all_shuffled_distributions[i] = dist_shuffled_aurocs
#
#     results = {
#         'all_distributions': all_distributions,
#         'pre95_auroc': pre95_auroc,
#         'pre05_auroc': pre05_auroc,
#         'all_shuffled_distributions': all_shuffled_distributions
#     }
#     return results

import numpy as np
import random

def compute_opto_windows_vs_baseline_subsampling(
    all_averages,
    all_baseline,
    all_data,
    n_to_sample,
    n_steps=100,
    n_boot=1000,
    random_state=42,
    fa_after_hit_data=None,
    fa_not_after_hit_data=None,
    cr_after_hit_data=None,
    cr_not_after_hit_data=None
):
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    all_distributions = [None] * n_neurons
    all_shuffled_distributions = [None] * n_neurons
    pre95_auroc = np.full(n_neurons, np.nan)
    pre05_auroc = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        if i >= len(all_baseline):
            continue

        neuron_vals = all_data.iloc[i]
        arr_2d = all_averages[i]                       # (#trials, 2)
        baseline_vals = all_baseline[i]               # (#trials,)

        windowA = arr_2d[:, 0]

        # Remove NaNs
        good = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        # if not all True save a notice to text file
        if not np.all(good):
            print(f"Neuron {i}: NaNs found in windowA or baseline.")
            with open("nan_warnings.txt", "a") as f:
                f.write(f"Neuron {i}: NaNs found in windowA or baseline.\n")
        wA_vals = windowA#[good]
        wA_vals = wA_vals.flatten()  # Ensure it's 1D
        baseA   = baseline_vals#[good]
        baseA = baseA.flatten()      # Ensure it's 1D

        len_wA   = len(wA_vals)
        len_base = len(baseA)
        if min(len_wA, len_base) == 0:
            continue

        # Sample size cannot exceed the shorter vector
        m = min(n_to_sample, len_wA, len_base)
        if m < n_to_sample:
            print(f"Neuron {i}: using sample size {m} (shorter than requested {n_to_sample}).")

        dist_aurocs = np.zeros(n_boot)
        dist_shuf   = np.zeros(n_boot)

        longer_is_wA = len_wA > len_base

        for b in range(n_boot):
            if longer_is_wA:
                wA_sub   = np.random.choice(wA_vals,  m, replace=False)
                base_sub = baseA if len_base == m else np.random.choice(baseA, m, replace=False)
            elif len_base > len_wA:
                base_sub = np.random.choice(baseA,  m, replace=False)
                wA_sub   = wA_vals if len_wA == m else np.random.choice(wA_vals, m, replace=False)
            # else:  # equal lengths
            #     wA_sub   = np.random.choice(wA_vals,  m, replace=False)
            #     base_sub = np.random.choice(baseA,  m, replace=False)

            # Compute auROC
            roc_val, _, _, _ = manual_threshold_auroc(wA_sub, base_sub, neuron_vals, n_steps)
            dist_aurocs[b] = roc_val

            # Shuffle condition labels
            labels  = np.concatenate([np.ones(m), np.zeros(m)])
            scores  = np.concatenate([wA_sub, base_sub])
            shuffled = np.random.permutation(labels)
            ev_s    = scores[shuffled == 1]
            bs_s    = scores[shuffled == 0]
            roc_shuf, _, _, _ = manual_threshold_auroc(ev_s, bs_s, neuron_vals, n_steps)
            dist_shuf[b] = roc_shuf

        all_distributions[i]        = dist_aurocs
        all_shuffled_distributions[i] = dist_shuf
        pre95_auroc[i] = np.percentile(dist_aurocs, 90)
        pre05_auroc[i] = np.percentile(dist_aurocs, 10)

    return {
        "all_distributions": all_distributions,
        "pre95_auroc": pre95_auroc,
        "pre05_auroc": pre05_auroc,
        "all_shuffled_distributions": all_shuffled_distributions,
    }

def manual_threshold_auroc(event_vals, baseline_vals, all_data, n_steps=100):
    """
    Compute auROC by manually sweeping a threshold over the range of
    min->max amplitude, as described in the paper's Methods section.

    event_vals: 1D array of event (social investigating) amplitudes
    baseline_vals: 1D array of baseline amplitudes
    n_steps: how many threshold increments to use (paper used 100)

    Returns
    -------
    auroc: float in [0,1], area under the ROC curve
           (prob(Event>T) vs. prob(Baseline>T)).
    thresholds: array of shape (n_steps+1,), the actual thresholds tested
    tpr: array of shape (n_steps+1,), each TPR = fraction of event_vals > threshold
    fpr: array of shape (n_steps+1,), each FPR = fraction of baseline_vals > threshold
    """
    # find overall min/max
    min_val = np.nanpercentile(all_data, 0.1, axis=0)
    max_val = np.nanpercentile(all_data, 99.9, axis=0)
    # min_val = np.nanmin(all_data)
    # max_val = np.nanmax(all_data)
    if min_val == max_val:
        print("Warning: min_val == max_val, cannot compute auROC.")
        with open("nan_warnings.txt", "a") as f:
            f.write("Warning: min_val == max_val, cannot compute auROC.\n")
        # everything is identical => auROC = 0.5
        return 0.5, np.array([min_val]), np.array([0.0]), np.array([0.0])

    # We'll build thresholds in n_steps increments
    step = (max_val - min_val) / n_steps
    thresholds = np.arange(n_steps + 1) * step + min_val

    # Precompute lengths
    nE = len(event_vals)
    nB = len(baseline_vals)

    tpr = np.zeros(len(thresholds), dtype=float)  # True Positive Rate
    fpr = np.zeros(len(thresholds), dtype=float)  # False Positive Rate

    for i, thr in enumerate(thresholds):
        # Probability that event > thr
        tpr[i] = np.sum(event_vals > thr) / nE
        # Probability that baseline > thr (the "false positive" for that threshold)
        fpr[i] = np.sum(baseline_vals > thr) / nB

    # Now we have TPR(FPR) param by threshold, integrate via trapezoid rule
    # Sort by fpr ascending if not already (it should be descending in thr).
    # But let's ensure we get a standard shape (some do fpr ascending).
    # Actually, TPR and FPR will be in descending order if thr goes from min->max.
    # Typically, we want FPR from 0->1. Let's *reverse* them if needed.

    # We want to do the standard "FPR ascending" approach. Check if fpr is from 1->0.
    if fpr[0] > fpr[-1]:
        # Reverse them
        tpr = tpr[::-1]
        fpr = fpr[::-1]

    # Compute area under TPR vs FPR
    # np.trapz(y, x) will do the integration
    auroc = np.trapz(tpr, x=fpr)

    return auroc, thresholds, tpr, fpr
def manual_auroc_with_shuffle(event_vals, baseline_vals, all_vals,
                              n_perm=1000,
                              n_steps=100,
                              random_state=None):
    """
    Replicates the paper's approach:
      1) Compute real auROC by manual threshold sweep
      2) Shuffle event/baseline labels 1,000 times -> null distribution
      3) Compare real auROC to 95th percentile => "discriminative"

    Returns
    -------
    real_auc : float
    perm_distribution : array of shape (n_perm,)
    auc_95th : float
       95th percentile of perm_distribution
    is_significant : bool
       True if real_auc > auc_95th
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Clean out NaNs
    event_vals = event_vals[~np.isnan(event_vals)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    if len(event_vals) < 2 or len(baseline_vals) < 2:
        return np.nan, np.array([]), np.nan, False

    # 1) Real auROC
    real_auc, _, _, _ = manual_threshold_auroc(event_vals, baseline_vals, all_vals, n_steps)

    # 2) Build label-shuffled null distribution
    nE = len(event_vals)
    nB = len(baseline_vals)
    labels = np.concatenate([np.ones(nE), np.zeros(nB)])
    all_data = np.concatenate([event_vals, baseline_vals])

    perm_distribution = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        # Shuffle event/baseline identity
        shuffled_labels = np.random.permutation(labels)
        # Then separate them back into "event" vs "baseline"
        event_mask = (shuffled_labels == 1)
        base_mask = (shuffled_labels == 0)
        ev_shuf = all_data[event_mask]
        bs_shuf = all_data[base_mask]
        # Now compute auROC for these newly-labeled sets
        auc_i, _, _, _ = manual_threshold_auroc(ev_shuf, bs_shuf, all_vals, n_steps)
        perm_distribution[i] = auc_i

    # 3) Compare real_auc to 95th percentile
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

def manual_auroc_with_shuffle_FA(event_vals, baseline_vals, all_vals,
                              n_perm=1000,
                              n_steps=100,
                              random_state=None):
    """
    Replicates the paper's approach:
      1) Compute real auROC by manual threshold sweep
      2) Shuffle event/baseline labels 1,000 times -> null distribution
      3) Compare real auROC to 95th percentile => "discriminative"

    Returns
    -------
    real_auc : float
    perm_distribution : array of shape (n_perm,)
    auc_95th : float
       95th percentile of perm_distribution
    is_significant : bool
       True if real_auc > auc_95th
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Clean out NaNs
    event_vals = event_vals[~np.isnan(event_vals)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    if len(event_vals) < 2 or len(baseline_vals) < 2:
        return np.nan, np.array([]), np.nan, False

    # 1) Real auROC
    real_auc, _, _, _ = manual_threshold_auroc(event_vals, baseline_vals, all_vals, n_steps)

    # 2) Build label-shuffled null distribution
    nE = len(event_vals)
    nB = len(baseline_vals)
    labels = np.concatenate([np.ones(nE), np.zeros(nB)])
    all_data = np.concatenate([event_vals, baseline_vals])

    perm_distribution = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        # Shuffle event/baseline identity
        shuffled_labels = np.random.permutation(labels)
        # Then separate them back into "event" vs "baseline"
        event_mask = (shuffled_labels == 1)
        base_mask = (shuffled_labels == 0)
        ev_shuf = all_data[event_mask]
        bs_shuf = all_data[base_mask]
        # Now compute auROC for these newly-labeled sets
        auc_i, _, _, _ = manual_threshold_auroc(ev_shuf, bs_shuf, all_vals, n_steps)
        perm_distribution[i] = auc_i

    # 3) Compare real_auc to 95th percentile
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

def reconstruct_freq_indices(conditions_played, freq):
    """
    Reconstruct trial indices for a specific frequency from conditions_played.
    """
    frequency_map = {7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    
    # Find all codes that map to this frequency
    target_codes = [code for code, mapped_freq in frequency_map.items() if mapped_freq == float(freq)]
    
    # Get trial indices for this frequency
    trial_indices = []
    for i, condition in enumerate(conditions_played):
        if condition in target_codes:
            trial_indices.append(i)
    
    return trial_indices

def select_temporal_hit_trials(hit_event_averages, hit_indices, light_trial_indices):
    """
    Select the closest hit trial for each light trial with no reuse.
    Each hit trial can only be used once.
    
    Parameters:
    - hit_event_averages: all hit trial data
    - hit_indices: indices of all hit trials in the session
    - light_trial_indices: indices of light trials for current frequency
    
    Returns:
    - Filtered hit event averages containing exactly the same number of trials as light trials
    - selected_hit_indices: actual indices of selected hit trials
    """
    # Check if arrays are empty using len() instead of truth value
    if len(light_trial_indices) == 0 or len(hit_indices) == 0:
        return hit_event_averages, []
    
    n_light_trials = len(light_trial_indices)
    n_hit_trials = len(hit_indices)
    
    print(f"  Finding closest hit trials for {n_light_trials} light trials from {n_hit_trials} available hit trials")
    
    # Check if we have enough hit trials
    if n_hit_trials < n_light_trials:
        print(f"  ERROR: Not enough hit trials ({n_hit_trials}) for light trials ({n_light_trials})")
        print(f"  Cannot proceed without reuse. Please check your data or Analysis parameters.")
        return hit_event_averages, []
    
    # For each light trial, find the closest available hit trial
    selected_hit_positions = []  # positions for indexing hit_event_averages
    selected_hit_indices = []    # actual hit trial indices
    available_hit_positions = set(range(n_hit_trials))  # Keep track of available hit trials
    
    # Create a list of (light_trial_idx, light_position) for processing
    light_trials_with_pos = [(light_idx, pos) for pos, light_idx in enumerate(light_trial_indices)]
    
    for light_idx, light_pos in light_trials_with_pos:
        best_hit_pos = None
        best_hit_idx = None
        min_distance = float('inf')
        
        # Find the closest available hit trial to this light trial
        for hit_pos in available_hit_positions:
            hit_idx = hit_indices[hit_pos]  # Get the actual hit trial index
            distance = abs(hit_idx - light_idx)
            
            if distance < min_distance:
                best_hit_pos = hit_pos
                best_hit_idx = hit_idx
                min_distance = distance
        
        if best_hit_pos is not None:
            selected_hit_positions.append(best_hit_pos)  # For indexing data arrays
            selected_hit_indices.append(best_hit_idx)    # Actual trial indices
            available_hit_positions.remove(best_hit_pos)  # Remove from available pool
            print(f"    Light trial {light_idx} -> Hit trial {best_hit_idx}, distance: {min_distance}")
        else:
            print(f"    ERROR: No available hit trials left for light trial {light_idx}")
            break
    
    # Check if we successfully matched all light trials
    if len(selected_hit_positions) != n_light_trials:
        print(f"  ERROR: Only matched {len(selected_hit_positions)} out of {n_light_trials} light trials")
        return hit_event_averages, []
    
    # Save the selected hit indices and light trial indices to a text file
    try:
        with open('selected_hit_positions.txt', 'w') as f:
            f.write("Selected hit trial indices:\n")
            for idx in selected_hit_indices:
                f.write(f"{idx}\n")
            f.write("Light trial indices:\n")
            for idx in light_trial_indices:
                f.write(f"{idx}\n")
    except:
        pass  # Ignore file writing errors
    
    # Extract the selected trials from hit_event_averages using positions
    temporal_hit_averages = []
    for neuron_data in hit_event_averages:
        if isinstance(neuron_data, np.ndarray):
            selected_trials = neuron_data[selected_hit_positions]
        else:
            # Handle DataFrame case
            selected_trials = neuron_data.iloc[selected_hit_positions]
        temporal_hit_averages.append(selected_trials)
    
    print(f"  Successfully selected {len(selected_hit_indices)} unique hit trials to match {n_light_trials} light trials")
    print(f"  Selected hit trial indices: {selected_hit_indices}")
    print(f"  Remaining unused hit trials: {len(available_hit_positions)}")
    
    return temporal_hit_averages, selected_hit_indices  # Return actual indices, not positions

if '__main__' == __name__:
    # Loop over multiple mice/days
    mice_days = [
        ('OI208', '250227'),
        ('OI151', '240827'),
        ('OI152', '241003')
    ]

    for mouse_id, day_str in mice_days:
        print(f"\nProcessing mouse {mouse_id} on day {day_str}")

        licking_data = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\licking_analysis_final'
        load_loc = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auROC_freq_history'
        save_loc = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auROC_freq_psycho_new'
        os.makedirs(save_loc, exist_ok=True)
        priming_trials = 23
        # Load the event data
        with open(join(load_loc, "event_data_all_light.pkl"), 'rb') as f:
            event_data_all_light = pickle.load(f)
        with open(join(load_loc, "all_light_event_averages.pkl"), 'rb') as f:
            all_light_event_averages = pickle.load(f)
        with open(join(load_loc, "event_data_correct_reject.pkl"), 'rb') as f:
            event_data_correct_reject = pickle.load(f)
        with open(join(load_loc, "event_data_hit.pkl"), 'rb') as f:
            event_data_hit = pickle.load(f)
        with open(join(load_loc, "event_data_false_alarm.pkl"), 'rb') as f:
            event_data_false_alarm = pickle.load(f)
        with open(join(load_loc, "event_data_miss.pkl"), 'rb') as f:
            event_data_miss = pickle.load(f)
        with open(join(load_loc, 'cr_light_averages.pkl'), 'rb') as f:
            cr_event_averages = pickle.load(f)
        with open(join(load_loc, 'fa_light_averages.pkl'), 'rb') as f:
            fa_event_averages = pickle.load(f)
        with open(join(load_loc, 'hit_light_averages.pkl'), 'rb') as f:
            hit_event_averages = pickle.load(f)
        with open(join(load_loc, 'miss_light_averages.pkl'), 'rb') as f:
            miss_event_averages = pickle.load(f)
        with open(join(load_loc, 'all_values.pkl'), 'rb') as f:
            all_values = pickle.load(f)
        with open(join(load_loc, 'correct_reject.pkl'), 'rb') as f:
            correct_reject = pickle.load(f)
        with open(join(load_loc, 'hit.pkl'), 'rb') as f:
            hit = pickle.load(f)
        with open(join(load_loc, 'false_alarm.pkl'), 'rb') as f:
            false_alarm = pickle.load(f)
        with open(join(load_loc, 'miss.pkl'), 'rb') as f:
            miss = pickle.load(f)

        # -------------------------------
        # Load Global Trial History Data
        # -------------------------------
        try:
            with open(join(load_loc, 'fa_after_hit_data.pkl'), 'rb') as f:
                fa_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'fa_not_after_hit_data.pkl'), 'rb') as f:
                fa_not_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'cr_after_hit_data.pkl'), 'rb') as f:
                cr_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'cr_not_after_hit_data.pkl'), 'rb') as f:
                cr_not_after_hit_data = pickle.load(f)
            print("Successfully loaded global trial history data")
            use_trial_history_global = True
        except FileNotFoundError:
            print("Global trial history data files not found. Using original sampling method.")
            fa_after_hit_data = None

        # -------------------------------
        # Load Global Trial History Data
        # -------------------------------
        try:
            with open(join(load_loc, 'fa_after_hit_data.pkl'), 'rb') as f:
                fa_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'fa_not_after_hit_data.pkl'), 'rb') as f:
                fa_not_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'cr_after_hit_data.pkl'), 'rb') as f:
                cr_after_hit_data = pickle.load(f)
            with open(join(load_loc, 'cr_not_after_hit_data.pkl'), 'rb') as f:
                cr_not_after_hit_data = pickle.load(f)
            print("Successfully loaded global trial history data")
            use_trial_history_global = True
        except FileNotFoundError:
            print("Global trial history data files not found. Using original sampling method.")
            fa_after_hit_data = None
            fa_not_after_hit_data = None
            cr_after_hit_data = None
            cr_not_after_hit_data = None
            use_trial_history_global = False

        # -------------------------------
        # Global Analysis
        # -------------------------------

        # Check if reward and lick_onset data exist in the global hit data
        if 'reward' in event_data_hit and 'lick_onset' in event_data_hit:
            print("Processing global reward and lick onset Analysis")

            # compute the auROC for the different events
            avg_activity_reward = []
            for neuron in (event_data_hit['reward']['event']):
                avg_activity_reward.append(np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1))
            ave_activity_lick_on = []
            for neuron in (event_data_hit['lick_onset']['event']):
                ave_activity_lick_on.append(np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1))

            # compare the time of the reward\lick onset to the distractor light to make sure its real
            all_baseline_all_light = event_data_all_light['light_off']['baseline']
            baseline_all_light = [np.asarray(all_baseline_all_light[i]) for i in range(len(all_baseline_all_light))]
            baseline_all_light = [np.nanmean(baseline_all_light[i], axis=1) for i in range(len(baseline_all_light))]
            res_all_light = compute_opto_windows_vs_baseline(
                all_light_event_averages,
                baseline_all_light,
                all_values,
                n_perm=500,
                random_state=42
            )

            res_all_light_vs_lightHit = compute_opto_windows_vs_baseline(
                all_light_event_averages,
                hit_event_averages,
                all_values,
                n_perm=500,
                random_state=42
            )
            # res_reward_vs_lightHit = compute_opto_windows_vs_hit_sliding_psycho(
            #     avg_activity_reward,
            #     hit_event_averages,
            #     all_values,
            #     None,
            #     None,
            #     n_perm=500,
            #     random_state=42
            # )
            # res_lick_on_vs_lightHit = compute_opto_windows_vs_hit_sliding_psycho(
            #     ave_activity_lick_on,
            #     hit_event_averages,
            #     all_values,
            #     None,
            #     None,
            #     n_perm=500,
            #     random_state=42
            # )

            # auROC_results_hit = compute_auROC_for_events_with_permutation(
            #     event_data_hit,
            #     all_values,
            #     n_perm=500,
            #     random_state=42
            # )

            # save the global results
            with open(join(save_loc, 'res_all_light.pkl'), 'wb') as f:
                pickle.dump(res_all_light, f)
            with open(join(save_loc, 'res_all_light_vs_lightHit.pkl'), 'wb') as f:
                pickle.dump(res_all_light_vs_lightHit, f)
        #     with open(join(save_loc, 'res_reward_vs_lightHit.pkl'), 'wb') as f:
        #         pickle.dump(res_reward_vs_lightHit, f)
        #     with open(join(save_loc, 'res_lick_on_vs_lightHit.pkl'), 'wb') as f:
        #         pickle.dump(res_lick_on_vs_lightHit, f)
        #     with open(join(save_loc, 'auROC_results_hit.pkl'), 'wb') as f:
        #         pickle.dump(auROC_results_hit, f)

        #     print("Saved global reward and lick onset results")
        # else:
        #     print("No reward/lick_onset data available in global hit data")

        # # -------------------------------
        # # Frequency-Specific Analysis with Trial History
        # # -------------------------------
        # print("\n=== Frequency-Specific Analysis ===")
        
        # for freq in ['0.5', '1', '3', '5', '7.5', '10', '15']:
        #     print(f"\nProcessing frequency {freq} Hz:")
            
        #     # check if the data for the frequency exists
        #     try:
        #         with open(join(load_loc, f'all_light_averages_{freq}.pkl'), 'rb') as f:
        #             all_light_event_freq = pickle.load(f)
        #         with open(join(load_loc, f'event_data_correct_reject_{freq}.pkl'), 'rb') as f:
        #             event_data_all_light_freq = pickle.load(f)
        #     except FileNotFoundError:
        #         print(f'No data for frequency {freq}, skipping...')
        #         continue

        #     # -------------------------------
        #     # Load Frequency-Specific Trial History Data
        #     # -------------------------------
        #     try:
        #         with open(join(load_loc, f'fa_after_hit_data_{freq}.pkl'), 'rb') as f:
        #             freq_fa_after_hit_data = pickle.load(f)
        #         with open(join(load_loc, f'fa_not_after_hit_data_{freq}.pkl'), 'rb') as f:
        #             freq_fa_not_after_hit_data = pickle.load(f)
        #         with open(join(load_loc, f'cr_after_hit_data_{freq}.pkl'), 'rb') as f:
        #             freq_cr_after_hit_data = pickle.load(f)
        #         with open(join(load_loc, f'cr_not_after_hit_data_{freq}.pkl'), 'rb') as f:
        #             freq_cr_not_after_hit_data = pickle.load(f)
        #         print(f"  Successfully loaded trial history data for {freq} Hz")
        #         use_trial_history_freq = True
        #     except FileNotFoundError:
        #         print(f"  Trial history data files not found for {freq} Hz. Using original sampling method.")
        #         freq_fa_after_hit_data = None
        #         freq_fa_not_after_hit_data = None
        #         freq_cr_after_hit_data = None
        #         freq_cr_not_after_hit_data = None
        #         use_trial_history_freq = False

        #     # Load the original conditions_played data to reconstruct trial indices
        #     try:
        #         # Try to load conditions_played from the original data processing
        #         file_loc = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data'
        #         conditions_played = np.load(join(file_loc, 'conditions_played.npy'), allow_pickle=True)
        #         # remove the priming trials
        #         conditions_played = conditions_played[priming_trials:]

        #     except FileNotFoundError:
        #         # If not available, you'll need to load it from the original data source
        #         print(f"  Warning: conditions_played not found")
        #         conditions_played = None

        #     # Reconstruct trial indices for this frequency
        #     if conditions_played is not None:
        #         light_trial_indices = reconstruct_freq_indices(conditions_played, freq)
        #         print(f"  Found {len(light_trial_indices)} light trials for frequency {freq}")
                
        #         # DEBUG: Check for overlap between hit and light trials
        #         light_set = set(light_trial_indices)
        #         hit_set = set(hit)
        #         overlap = light_set.intersection(hit_set)
                
        #         print(f"  DEBUG: Hit trials: {len(hit)} trials")
        #         print(f"  DEBUG: Light trials for {freq}Hz: {len(light_trial_indices)} trials")
        #         print(f"  DEBUG: Overlap (SHOULD BE ZERO): {len(overlap)} trials")
                
        #         if len(overlap) > 0:
        #             print(f"  ERROR: Found {len(overlap)} trials that are both HIT and LIGHT!")
        #             print(f"  Overlapping trial indices: {sorted(list(overlap))}")

        #         # Only proceed if there's no overlap
        #         # Select temporally matched hit trials
        #         temporal_hit_averages, temporal_hit_indices = select_temporal_hit_trials(
        #             hit_event_averages, 
        #             hit,  # hit trial indices from the loaded data
        #             light_trial_indices  # adjust in order to match the temporal window of the light trials
        #         )
        #     else:
        #         # raise an error if conditions_played is not available
        #         print(f"  Warning: conditions_played not found")


        #     # Use temporal_hit_averages instead of hit_event_averages
        #     res_all_light_vs_lightHit = compute_opto_windows_vs_baseline(
        #         all_light_event_freq,
        #         temporal_hit_averages,  # <-- Use temporally matched hit trials
        #         all_values,
        #         n_perm=500,
        #         random_state=42
        #     )
        #     # compute the auROC for the different events
        #     all_baseline_cr_freq = event_data_all_light_freq['light_off']['baseline']
        #     baseline_freq = [np.asarray(all_baseline_cr_freq[i]) for i in range(len(all_baseline_cr_freq))]
        #     baseline_freq = [np.nanmean(baseline_freq[i], axis=1) for i in range(len(baseline_freq))]

        #     res = compute_opto_windows_vs_baseline(
        #         all_light_event_freq,
        #         baseline_freq,
        #         all_values,
        #         n_perm=500,
        #         random_state=42
        #     )

        #     # save the results
        #     with open(join(save_loc, f'res{freq}.pkl'), 'wb') as f:
        #         pickle.dump(res, f)
        #     with open(join(save_loc, f'res_all_light_vs_lightHit_{freq}.pkl'), 'wb') as f:
        #         pickle.dump(res_all_light_vs_lightHit, f)


        # print("\nAll analyses completed!")