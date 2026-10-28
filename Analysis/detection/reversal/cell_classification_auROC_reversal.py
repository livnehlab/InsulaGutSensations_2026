"""
Reversal sessions: auROC classification of light-responsive neurons.

Loads the event data saved by creat_events_data_per_neuron_reversal.py and, for all light
trials, tests the opto window against the baseline (sliding trial window) and against the
no-opto (CR) trials on the selected trial range, using label-shuffled null distributions.
Saves res_all_light_vs_lightCR.pkl to processed_data/auROC of each session.
"""

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

start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000  # MonkeyLogic sampling rate (Hz)
rec_hz = 15  # 2-photon frame rate (Hz)
sync_hz = 6000  # ThorSync sampling rate (Hz)
pre_trial_length = 4 * rec_hz  # frames prepended to each trial
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz  # opto stimulation length (frames)

def plot_increase_decrease_bar(sig_neurons, total_neurons,Title):
    """
    Makes a grouped bar plot for each event,
    comparing #increased vs. #decreased cells.
    """
    event_names = list(sig_neurons.keys())
    n_events = len(event_names)

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
    The same is done for 'decrease'.
    significant_neurons[e]['increase'] / ['decrease'] are lists of neuron indices.
    """
    for i in range(len(events_order) - 1):  # consecutive event pairs
        e1 = events_order[i]
        e2 = events_order[i+1]

        # neurons 'increase' in both events
        inc_e1 = set(significant_neurons[e1]['increase'])
        inc_e2 = set(significant_neurons[e2]['increase'])
        overlap_inc = inc_e1.intersection(inc_e2)

        # remove them from e2
        if len(overlap_inc) > 0:
            new_e2_inc = [n for n in significant_neurons[e2]['increase'] if n not in overlap_inc]
            significant_neurons[e2]['increase'] = np.array(new_e2_inc, dtype=int)

        # same for 'decrease'
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
    # the intersection label exists only if the sets overlap
    if len(set1 & set2) > 0:
        v.get_label_by_id('11').set_text(f"{(len(set1 & set2) / total * 100):.1f}%")
    plt.title(title)
    plt.savefig(join(save_loc, filename))
    plt.show()

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
    For each event and neuron, computes the auROC of the trial-wise mean activity
    in the event window vs. the baseline window, and classifies it against a
    label-shuffled null distribution (manual_auroc_with_shuffle).

    Returns
    -------
    results : dict
      results[event_name] = {
         'auroc': np.array of shape (n_neurons,),
         'p_value': list of classifications ('above_95' / 'below_5' / 'ns'),
         'all5', 'all95': lower / upper null percentiles,
         'distribution': null distribution of each neuron
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

        results[event_name] = {
            'auroc': aurocs,
            'p_value': p_values,
            'all5': all5,
            'all95': all95
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
# #             continue
#
#         # Column 0, column 1
#         windowA = arr_2d[:, 0]
#
#         # Drop trials with NaNs in either distribution
#         # the baseline must not be NaN either
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

def compute_opto_windows_vs_baseline_sliding(
    all_averages,     # list of (#trials, 2) arrays, one per neuron
    all_baseline,     # list of (#trials,) arrays, one per neuron
    all_data,
    window_size=40,   # how many trials in each sliding window
    step=10,          # how many trials to jump each time
    n_perm=1000,
    random_state=None
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
        base_vals  = baseline_vals[mask]

        n_valid = len(event_vals)
        if n_valid < window_size:
            # not enough trials for one window
            continue

        # 1) Slide over the valid trials in increments of 'step'
        max_auroc_sub = 0
        best_start = 0
        best_end   = 0

        # start = 0, step, 2*step, ... while start + window_size <= n_valid
        for start_idx in range(0, n_valid - window_size + 1, step):
            end_idx = start_idx + window_size
            event_chunk = event_vals[start_idx:end_idx]
            base_chunk  = base_vals[start_idx:end_idx]

            # auROC of this window (no permutation yet)
            labels = np.concatenate([np.ones(len(event_chunk)), np.zeros(len(base_chunk))])
            scores = np.concatenate([event_chunk, base_chunk])
            this_auroc = roc_auc_score(labels, scores)

            abs_diff = np.abs(this_auroc - 0.5)
            if abs_diff > max_auroc_sub:
                max_auroc_sub = abs_diff
                best_start = start_idx
                best_end   = end_idx

        # 2) Label-shuffle test on the selected window only
        if best_end > best_start:
            event_chunk = event_vals[best_start:best_end]
            base_chunk  = base_vals[best_start:best_end]

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
            best_ranges[i] = (best_start, best_end)

    results = {
        'final_auroc':    final_auroc,
        'null_5th':       final_5th,
        'null_95th':      final_95th,
        'classification': classifications,
        'best_range':     best_ranges
    }
    return results

def compute_opto_windows_vs_hit_sliding(
    all_averages,     # list of (#trials, 2) arrays, one per neuron
    all_baseline,     # list of (#trials,) arrays, one per neuron
    all_data,
    best_start=None,
    best_end=None,
    n_perm=1000,
    random_state=None
):
    """
    For each neuron, runs the label-shuffle test of event (column 0 of
    all_averages[i]) vs. comparison values (column 0 of all_baseline[i]) on the
    trial range [best_start[i], best_end[i]) (all trials if no ranges are given).
    No NaN filtering is applied here.

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
        event_vals = event_full#[mask]
        base_vals  = baseline_vals[:, 0]#[mask]


        # Label-shuffle test on the given trial range
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

    results = {
        'final_auroc':    final_auroc,
        'null_5th':       final_5th,
        'null_95th':      final_95th,
        'classification': classifications,
        'best_range':     best_ranges
    }
    return results
def compute_opto_windows_vs_baseline_FA(
        all_averages,  # list of shape (#neurons,) each -> (#trials, 2)
        all_baseline,  # list of shape (#neurons,) each -> (#trials,)
        all_data,
        n_perm=1000,
        random_state=None
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
        'all5', 'all95': lists of lower / upper null percentiles,
        'distribution': null distribution of each neuron (when present)
      }
    """
    n_neurons = len(all_averages)
    aurocA = np.full(n_neurons, np.nan)
    pvalA = []


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
            aA, null_5th, null_95th, dist, pA = manual_auroc_with_shuffle_FA(wA_vals, baseA, neuron_data, n_perm, random_state)
            aurocA[i] = aA
            pvalA.append(pA)
    results = {
        'aurocA': aurocA,
        'pvalA': pvalA,
        'all5': null_5th,
        'all95': null_95th
    }
    return results
def compute_opto_windows_vs_baseline_subsampling(
    all_averages,    # list of shape (#neurons,) each -> (#trials, 2)
    all_baseline,    # list of shape (#neurons,) each -> (#trials,)
    all_data,        # DataFrame of shape (#neurons, #trials)
    n_fa,            # how many trials to subsample each time
    n_steps=100,       # how many thresholds to sweep
    n_boot=1000,     # how many subsampling iterations
    random_state=42
):
    """
    For each neuron, do the following:
      - wA_vals = column 0 of all_averages[i]; baseline = all_baseline[i].
      - wA_vals and baseline are assumed to have the SAME # of valid trials
        after NaN removal.
      - For b in [1..n_boot]:
          1) random subset of size n_fa from the available trials
          2) compute the real auROC of that subset
        -> build a distribution of 1000 auROCs
      - Finally, store that distribution (or a summary) in 'results'.

    Returns
    -------
    results : dict
      {
        'all_distributions': list of length #neurons,
          each entry is a 1D np.array of shape (n_boot,) with the
          auROCs from each subsample
        'mean_auroc': np.array of shape (#neurons,),
          the mean of each neuron's distribution
        'median_auroc': np.array of shape (#neurons,),
          the median of each neuron's distribution
      }
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_neurons = len(all_averages)
    # distribution for each neuron
    all_distributions = [None]*n_neurons

    pre95_auroc = np.full(n_neurons, np.nan)
    pre05_auroc = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        neuron_vals = all_data.iloc[i]
        arr_2d = all_averages[i]   # shape (#trials, 2) => col0 = event, col1 = something else?
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]

        # Check trial count match
        # if arr_2d.shape[0] != len(baseline_vals):
        #     print(f"Warning: mismatch in trial counts for neuron {i}!")
        #     continue

        # event window
        windowA = arr_2d[:, 0]

        # Drop NaNs
        maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        # wA_vals = windowA[maskA]
        # baseA   = baseline_vals[maskA]
        wA_vals = windowA
        baseA   = baseline_vals[:, 0]
        n_avail = len(wA_vals)
        if n_avail < n_fa:
            print(f"Neuron {i}: Not enough trials ({n_avail}) to subsample {n_fa}. Skipping.")
            continue

        # Build the distribution of auROCs
        dist_aurocs = np.zeros(n_boot, dtype=float)

        for b in range(n_boot):
            # 1) random subset of size n_fa
            # pick indices from 0..n_avail-1
            sub_idxs = random.sample(range(n_avail), k=n_fa)

            wA_sub = wA_vals[sub_idxs]
            # base_sub = baseA[sub_idxs]
            base_sub = baseA
            # Check trial count match
            if len(wA_sub) != len(base_sub):
                print(f"Warning: mismatch in trial counts for neuron {i}!")
                continue
            # 2) compute auROC
            # labels = np.concatenate([np.ones(len(wA_sub)), np.zeros(len(base_sub))])
            # shuffled_labels = np.random.permutation(labels)
            # scores = np.concatenate([wA_sub, base_sub])
            # ev_shuf = scores[shuffled_labels == 1]
            # bs_shuf = scores[shuffled_labels == 0]
            this_auroc, _, _, _ = manual_threshold_auroc(wA_sub, base_sub, neuron_vals, n_steps)

            dist_aurocs[b] = this_auroc

        all_distributions[i] = dist_aurocs
        pre95_auroc[i] = np.percentile(dist_aurocs, 90)
        pre05_auroc[i] = np.percentile(dist_aurocs, 10)

    results = {
        'all_distributions': all_distributions,
        'pre95_auroc': pre95_auroc,
        'pre05_auroc': pre05_auroc
    }
    return results


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
def manual_auroc_with_shuffle(event_vals, baseline_vals, all_vals,
                              n_perm=1000,
                              n_steps=100,
                              random_state=None):
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
        # split back into "event" and "baseline"
        event_mask = (shuffled_labels == 1)
        base_mask = (shuffled_labels == 0)
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

def manual_auroc_with_shuffle_FA(event_vals, baseline_vals, all_vals,
                              n_perm=1000,
                              n_steps=100,
                              random_state=None):
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
        # split back into "event" and "baseline"
        event_mask = (shuffled_labels == 1)
        base_mask = (shuffled_labels == 0)
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
if '__main__' == __name__:
    mice_days = [
        ('mouse09', 'sess10'),
        ('mouse10', 'sess08'),
        ('mouse14', 'sess13'),
        ('mouse16', 'sess15')
    ]
    for mouse_id, day_str in mice_days:
        if mouse_id in ['mouse09', 'mouse10']:
            ROOT_PATH = DATA_ROOT + fr'\{mouse_id}\{day_str}\ML\data2'
        else:
            ROOT_PATH = DATA_ROOT + fr'\{mouse_id}\{day_str}\ML\data'
        licking_data = DATA_ROOT + fr'\{mouse_id}\{day_str}\ML\licking_analysis_final'
        save_loc = DATA_ROOT + fr'\{mouse_id}\{day_str}\ML\data\processed_data\auROC'
        os.makedirs(save_loc, exist_ok=True)
        # Load the event data
        with open(join(save_loc, "event_data_light.pkl"), 'rb') as f:
            event_data_light = pickle.load(f)
        with open(join(save_loc, "event_data_correct_reject.pkl"), 'rb') as f:
            event_data_correct_reject = pickle.load(f)
        with open(join(save_loc, "event_data_hit.pkl"), 'rb') as f:
            event_data_hit = pickle.load(f)
        with open(join(save_loc, "event_data_false_alarm.pkl"), 'rb') as f:
            event_data_false_alarm = pickle.load(f)
        with open(join(save_loc, "event_data_miss.pkl"), 'rb') as f:
            event_data_miss = pickle.load(f)
        with open(join(save_loc, "light_averages.pkl"), 'rb') as f:
            all_light_averages = pickle.load(f)
        with open(join(save_loc, 'cr_light_averages.pkl'), 'rb') as f:
           cr_event_averages = pickle.load(f)
        with open(join(save_loc, 'fa_light_averages.pkl'), 'rb') as f:
            fa_event_averages = pickle.load(f)
        with open(join(save_loc, 'hit_light_averages.pkl'), 'rb') as f:
            hit_event_averages = pickle.load(f)
        with open(join(save_loc, 'miss_light_averages.pkl'), 'rb') as f:
            miss_event_averages = pickle.load(f)
        with open(join(save_loc, 'all_values.pkl'), 'rb') as f:
            all_values = pickle.load(f)
        with open(join(save_loc, 'correct_reject.pkl'), 'rb') as f:
            correct_reject = pickle.load(f)
        with open(join(save_loc, 'hit.pkl'), 'rb') as f:
            hit = pickle.load(f)
        with open(join(save_loc, 'false_alarm.pkl'), 'rb') as f:
            false_alarm = pickle.load(f)
        with open(join(save_loc, 'miss.pkl'), 'rb') as f:
            miss = pickle.load(f)

        # compute the auROC for the different events
        avg_activity_reward = []
        for neuron in (event_data_hit['reward']['event']):
            avg_activity_reward.append(np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1))
        ave_activity_lick_on = []
        for neuron in (event_data_hit['lick_onset']['event']):
            ave_activity_lick_on.append(np.asarray([np.nanmean(trial) for trial in neuron]).reshape(-1, 1))
        # compare the time of the reward\lick onset to the distractor light to make sure its real
        # res_reward_vs_lightCR = compute_opto_windows_vs_hit_sliding(
        #     avg_activity_reward,
        #     cr_event_averages,
        #     all_values,
        #     None,
        #     None,
        #     n_perm=500,
        #     random_state=42
        # )
        # res_lick_on_vs_lightCR = compute_opto_windows_vs_hit_sliding(
        #     ave_activity_lick_on,
        #     cr_event_averages,
        #     all_values,
        #     None,
        #     None,
        #     n_perm=500,
        #     random_state=42
        # )
        # compute the auROC for the different events
        all_baseline_light = event_data_light['light_off']['baseline']
        baseline_mean = [np.asarray(all_baseline_light[i]) for i in range(len(all_baseline_light))]
        baseline_mean = [np.nanmean(baseline_mean[i], axis=1) for i in range(len(baseline_mean))]

        res_all_light = compute_opto_windows_vs_baseline_sliding(
            all_light_averages,
            baseline_mean,
            all_values,
            window_size=int(0.4 * len(all_light_averages[0])),
            step=5,
            n_perm=500,
            random_state=42
        )

        res_all_light_vs_lightCR = compute_opto_windows_vs_hit_sliding(
            all_light_averages,
            cr_event_averages,
            all_values,
            [res_all_light['best_range'][i][0] for i in range(len(res_all_light['best_range']))],
            [res_all_light['best_range'][i][1] for i in range(len(res_all_light['best_range']))],
            n_perm=500,
            random_state=42
        )
        #
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
        # all_baseline_hit = event_data_hit['light_off']['baseline']
        # baseline_mean = [np.asarray(all_baseline_hit[i]) for i in range(len(all_baseline_hit))]
        # baseline_mean = [np.nanmean(baseline_mean[i], axis=1) for i in range(len(baseline_mean))]
        #
        # res = compute_opto_windows_vs_baseline_sliding(
        #     hit_event_averages,
        #     baseline_mean,
        #     all_values,
        #     window_size=int(len(hit)/2),
        #     step=5,
        #     n_perm=500,
        #     random_state=42
        # )
        # # compare the time of the light between stimulation to only the distractor light
        # res_hit_vs_lightCR = compute_opto_windows_vs_hit_sliding(
        #     hit_event_averages,
        #     cr_event_averages,
        #     all_values,
        #     [res['best_range'][i][0] for i in range(len(res['best_range']))],
        #     [res['best_range'][i][1] for i in range(len(res['best_range']))],
        #     n_perm=500,
        #     random_state=42
        # )
        #
        # all_baseline_miss = event_data_miss['light_off']['baseline']
        # baseline_mean_miss = [np.asarray(all_baseline_miss[i]) for i in range(len(all_baseline_miss))]
        # baseline_mean_miss = [np.nanmean(baseline_mean_miss[i], axis=1) for i in range(len(baseline_mean_miss))]
        # res_miss = compute_opto_windows_vs_baseline_FA(
        #     miss_event_averages,
        #     baseline_mean_miss,
        #     all_values,
        #     n_perm=500,
        #     random_state=42
        # )
        # # # compute opto windows subsampling vs fa in order to get the 95th and 5th percentile for the False Alarm
        # N_miss = len(all_baseline_miss[0])
        # res_hit_subsample = compute_opto_windows_vs_baseline_subsampling(
        #     hit_event_averages,
        #     miss_event_averages,
        #     all_values,
        #     n_fa=N_miss,
        #     n_steps=100,
        #     n_boot=500,
        #     random_state=42
        # )
        #
        # # save the results
        # with open(join(save_loc, 'res_all_light.pkl'), 'wb') as f:
        #     pickle.dump(res_all_light, f)
        with open(join(save_loc, 'res_all_light_vs_lightCR.pkl'), 'wb') as f:
            pickle.dump(res_all_light_vs_lightCR, f)
        # with open(join(save_loc, 'auROC_results_cr.pkl'), 'wb') as f:
        #     pickle.dump(auROC_results_cr, f)
        # with open(join(save_loc, 'auROC_results_hit.pkl'), 'wb') as f:
        #     pickle.dump(auROC_results_hit, f)
        # with open(join(save_loc, 'auROC_results_fa.pkl'), 'wb') as f:
        #     pickle.dump(auROC_results_fa, f)
        # with open(join(save_loc, 'auROC_results_miss.pkl'), 'wb') as f:
        #     pickle.dump(auROC_results_miss, f)
        # with open(join(save_loc, 'res.pkl'), 'wb') as f:
        #     pickle.dump(res, f)
        # with open(join(save_loc, 'res_fa.pkl'), 'wb') as f:
        #     pickle.dump(res_miss, f)
        # with open(join(save_loc, 'res_cr_subsample.pkl'), 'wb') as f:
        #     pickle.dump(res_hit_subsample, f)
        # with open(join(save_loc, 'res_CR_vs_lightHit.pkl'), 'wb') as f:
        #     pickle.dump(res_hit_vs_lightCR, f)
        # with open(join(save_loc, 'res_reward_vs_lightHit.pkl'), 'wb') as f:
        #     pickle.dump(res_reward_vs_lightCR, f)
        # with open(join(save_loc, 'res_lick_on_vs_lightHit.pkl'), 'wb') as f:
        #     pickle.dump(res_lick_on_vs_lightCR, f)
        #
