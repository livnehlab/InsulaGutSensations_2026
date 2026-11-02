"""
Psychophysics CP: CR vs. FA auROC with trial-history-balanced subsampling, all neurons.

Same analysis as cell_classification_auROC_psycho_history.py run on all neurons: for each
neuron, the distribution of CR-vs-FA auROCs over subsamples with the same number of trials
of each history type (plus the label-shuffled distribution) is saved as
res_cr_subsample_global.pkl, which short_after_auroc_control.py and choice_probability.py
load from the real-data folder and from the CR-vs-CR control folders. In the current
configuration the paths point to the ITI-control folder (auROC_freq_history_iti_control).
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
    best_start,
    best_end,
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
def compute_opto_windows_vs_baseline(
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
    null_5th_all = []
    null_95th_all = []

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, n_windows)
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue

        baseline_vals = all_baseline[i]
        baseline_vals = baseline_vals.flatten()

        # event window
        windowA = arr_2d[:, 0]

        # Drop trials with NaNs in the event window or the baseline
        maskA = (~np.isnan(windowA)) & (~np.isnan(baseline_vals))
        wA_vals = windowA[maskA]
        baseA = baseline_vals[maskA]
        if wA_vals.shape[0] != len(baseA):
            print(f"Warning: mismatch in trial counts for neuron {i}!")
            continue


        # Label-shuffle test if there is enough data
        if len(wA_vals) >= 10 and len(baseA) >= 10:
            aA, null_5th, null_95th, dist, pA = manual_auroc_with_shuffle_FA(wA_vals, baseA, neuron_data, n_perm, random_state)
            aurocA[i] = aA
            pvalA.append(pA)
            null_5th_all.append(null_5th)
            null_95th_all.append(null_95th)
        else:
            # Not enough data to compute auROC
            continue

    results = {
        'aurocA': aurocA,
        'pvalA': pvalA,
        'all5': null_5th_all,
        'all95': null_95th_all
    }
    return results

import numpy as np
import random

def compute_opto_windows_vs_baseline_subsampling(
    all_averages,    # list of shape (#neurons,) each -> (#trials, 2)
    all_baseline,    # list of shape (#neurons,) each -> (#trials,)
    all_data,        # DataFrame of shape (#neurons, #trials)
    n_to_sample,            # how many trials to subsample each time
    n_steps=100,       # how many thresholds to sweep
    n_boot=1000,     # how many subsampling iterations
    random_state=42,
    # Trial-history balancing (optional)
    fa_after_hit_data=None,      # list of DataFrames for FA trials after hit
    fa_not_after_hit_data=None,  # list of DataFrames for FA trials not after hit
    cr_after_hit_data=None,      # list of DataFrames for CR trials after hit
    cr_not_after_hit_data=None   # list of DataFrames for CR trials not after hit
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
      
      If trial-history data is provided, sampling is balanced between
      "after hit" and "not after hit" trials to match the FA distribution.

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
    all_shuffled_distributions = [None]*n_neurons

    pre95_auroc = np.full(n_neurons, np.nan)
    pre05_auroc = np.full(n_neurons, np.nan)

    # If trial history data is provided, calculate the balance ratios
    use_trial_history = (fa_after_hit_data is not None and fa_not_after_hit_data is not None and 
                        cr_after_hit_data is not None and cr_not_after_hit_data is not None)
    
    if use_trial_history:
        # Count trials in each category
        n_fa_after_hit = len(fa_after_hit_data[0]) if len(fa_after_hit_data) > 0 else 0
        n_fa_not_after_hit = len(fa_not_after_hit_data[0]) if len(fa_not_after_hit_data) > 0 else 0
        n_cr_after_hit = len(cr_after_hit_data[0]) if len(cr_after_hit_data) > 0 else 0
        n_cr_not_after_hit = len(cr_not_after_hit_data[0]) if len(cr_not_after_hit_data) > 0 else 0
        
        print(f"FA trials: {n_fa_after_hit} after hit, {n_fa_not_after_hit} not after hit")
        print(f"CR trials: {n_cr_after_hit} after hit, {n_cr_not_after_hit} not after hit")
        
        # Determine sampling strategy based on trial counts
        if n_fa_not_after_hit <= n_cr_not_after_hit and n_fa_after_hit <= n_cr_after_hit:
            # Case 1: FA has fewer trials in both categories - sample from CR to match FA
            sampling_case = 1
            n_sample_cr_not_after = n_fa_not_after_hit
            n_sample_cr_after = n_fa_after_hit
            print(f"Case 1: Sampling from CR to match FA - {n_sample_cr_after} after, {n_sample_cr_not_after} not after")
            
        elif n_cr_not_after_hit <= n_fa_not_after_hit and n_cr_after_hit <= n_fa_after_hit:
            # Case 2: CR has fewer trials in both categories - sample from FA to match CR
            sampling_case = 2
            n_sample_fa_not_after = n_cr_not_after_hit
            n_sample_fa_after = n_cr_after_hit
            print(f"Case 2: Sampling from FA to match CR - {n_sample_fa_after} after, {n_sample_fa_not_after} not after")
            
        else:
            # Case 3: Mixed case - use the smaller number for each category
            sampling_case = 3
            n_sample_cr_not_after = min(n_fa_not_after_hit, n_cr_not_after_hit)
            n_sample_cr_after = min(n_fa_after_hit, n_cr_after_hit)
            n_sample_fa_not_after = min(n_fa_not_after_hit, n_cr_not_after_hit)
            n_sample_fa_after = min(n_fa_after_hit, n_cr_after_hit)
            print(f"Case 3: Mixed case - sampling from both CR and FA - {n_sample_cr_after} after, {n_sample_cr_not_after} not after")
            
        # Calculate total trials to sample
        if sampling_case == 1:
            total_trials_to_sample = n_sample_cr_after + n_sample_cr_not_after
        elif sampling_case == 2:
            total_trials_to_sample = n_sample_fa_after + n_sample_fa_not_after
        else:  # Case 3
            total_trials_to_sample = n_sample_cr_after + n_sample_cr_not_after + n_sample_fa_after + n_sample_fa_not_after
        print(f"Total trials to sample: {total_trials_to_sample}")

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
            if use_trial_history:
                # Use trial history balanced sampling
                if sampling_case == 1:
                    # Sample from CR to match FA distribution
                    if n_sample_cr_after > 0:
                        after_samples = cr_after_hit_data[i].iloc[np.random.choice(len(cr_after_hit_data[i]), n_sample_cr_after, replace=False), 0].values
                    else:
                        after_samples = np.array([])
                    
                    if n_sample_cr_not_after > 0:
                        not_after_samples = cr_not_after_hit_data[i].iloc[np.random.choice(len(cr_not_after_hit_data[i]), n_sample_cr_not_after, replace=False), 0].values
                    else:
                        not_after_samples = np.array([])
                    
                    wA_sub = np.concatenate([after_samples, not_after_samples])
                    
                    # Sample from FA to match FA distribution
                    if n_fa_after_hit > 0:
                        fa_after_samples = fa_after_hit_data[i].iloc[np.random.choice(len(fa_after_hit_data[i]), n_fa_after_hit, replace=False), 0].values
                    else:
                        fa_after_samples = np.array([])
                    
                    if n_fa_not_after_hit > 0:
                        fa_not_after_samples = fa_not_after_hit_data[i].iloc[np.random.choice(len(fa_not_after_hit_data[i]), n_fa_not_after_hit, replace=False), 0].values
                    else:
                        fa_not_after_samples = np.array([])
                    
                    base_sub = np.concatenate([fa_after_samples, fa_not_after_samples])
                    
                elif sampling_case == 2:
                    # Sample from FA to match CR distribution
                    if n_sample_fa_after > 0:
                        after_samples = fa_after_hit_data[i].iloc[np.random.choice(len(fa_after_hit_data[i]), n_sample_fa_after, replace=False), 0].values
                    else:
                        after_samples = np.array([])
                    
                    if n_sample_fa_not_after > 0:
                        not_after_samples = fa_not_after_hit_data[i].iloc[np.random.choice(len(fa_not_after_hit_data[i]), n_sample_fa_not_after, replace=False), 0].values
                    else:
                        not_after_samples = np.array([])
                    
                    base_sub = np.concatenate([after_samples, not_after_samples])
                    
                    # Sample from CR to match CR distribution
                    if n_cr_after_hit > 0:
                        cr_after_samples = cr_after_hit_data[i].iloc[np.random.choice(len(cr_after_hit_data[i]), n_cr_after_hit, replace=False), 0].values
                    else:
                        cr_after_samples = np.array([])
                    
                    if n_cr_not_after_hit > 0:
                        cr_not_after_samples = cr_not_after_hit_data[i].iloc[np.random.choice(len(cr_not_after_hit_data[i]), n_cr_not_after_hit, replace=False), 0].values
                    else:
                        cr_not_after_samples = np.array([])
                    
                    wA_sub = np.concatenate([cr_after_samples, cr_not_after_samples])
                    
                else:  # Case 3: Mixed case
                    # Sample from both CR and FA with balanced proportions
                    if n_sample_cr_after > 0:
                        cr_after_samples = cr_after_hit_data[i].iloc[np.random.choice(len(cr_after_hit_data[i]), n_sample_cr_after, replace=False), 0].values
                    else:
                        cr_after_samples = np.array([])
                    
                    if n_sample_cr_not_after > 0:
                        cr_not_after_samples = cr_not_after_hit_data[i].iloc[np.random.choice(len(cr_not_after_hit_data[i]), n_sample_cr_not_after, replace=False), 0].values
                    else:
                        cr_not_after_samples = np.array([])
                    
                    if n_sample_fa_after > 0:
                        fa_after_samples = fa_after_hit_data[i].iloc[np.random.choice(len(fa_after_hit_data[i]), n_sample_fa_after, replace=False), 0].values
                    else:
                        fa_after_samples = np.array([])
                    
                    if n_sample_fa_not_after > 0:
                        fa_not_after_samples = fa_not_after_hit_data[i].iloc[np.random.choice(len(fa_not_after_hit_data[i]), n_sample_fa_not_after, replace=False), 0].values
                    else:
                        fa_not_after_samples = np.array([])
                    
                    wA_sub = np.concatenate([cr_after_samples, cr_not_after_samples])
                    base_sub = np.concatenate([fa_after_samples, fa_not_after_samples])
                
                # Ensure we have enough samples
                if len(wA_sub) < 2 or len(base_sub) < 2:
                    # Fall back to original sampling if trial history sampling fails
                    if longer_is_wA:
                        wA_sub   = np.random.choice(wA_vals,  m, replace=False)
                        base_sub = baseA if len_base == m else np.random.choice(baseA, m, replace=False)
                    elif len_base > len_wA:
                        base_sub = np.random.choice(baseA,  m, replace=False)
                        wA_sub   = wA_vals if len_wA == m else np.random.choice(wA_vals, m, replace=False)
                    else:
                        wA_sub   = np.random.choice(wA_vals,  m, replace=False)
                        base_sub = np.random.choice(baseA,  m, replace=False)
            else:
                # Original sampling method
                if longer_is_wA:
                    wA_sub   = np.random.choice(wA_vals,  m, replace=False)
                    base_sub = baseA if len_base == m else np.random.choice(baseA, m, replace=False)
                elif len_base > len_wA:
                    base_sub = np.random.choice(baseA,  m, replace=False)
                    wA_sub   = wA_vals if len_wA == m else np.random.choice(wA_vals, m, replace=False)
                else:
                    wA_sub   = np.random.choice(wA_vals,  m, replace=False)
                    base_sub = np.random.choice(baseA,  m, replace=False)

            # Compute auROC
            roc_val, _, _, _ = manual_threshold_auroc(wA_sub, base_sub, neuron_vals, n_steps)
            dist_aurocs[b] = roc_val

            # Shuffle condition labels
            labels  = np.concatenate([np.ones(len(wA_sub)), np.zeros(len(base_sub))])
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
        print("Warning: min_val == max_val, cannot compute auROC.")
        with open("nan_warnings.txt", "a") as f:
            f.write("Warning: min_val == max_val, cannot compute auROC.\n")
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
    # for rep in range(1, 6):
    day = ['sess02']
    licking_data = DATA_ROOT + r'\mouse01\{}\ML\licking_analysis_final'.format(day[0])
    save_loc = DATA_ROOT + r'\mouse01\{}\ML\data\processed_data\auROC_freq_history_iti_control'.format(day[0])
    os.makedirs(save_loc, exist_ok=True)

    # Load the event data
    # with open(join(save_loc, "event_data_correct_reject.pkl"), 'rb') as f:
    #     event_data_correct_reject = pickle.load(f)
    # with open(join(save_loc, "event_data_hit.pkl"), 'rb') as f:
    #     event_data_hit = pickle.load(f)
    # with open(join(save_loc, "event_data_false_alarm.pkl"), 'rb') as f:
    #     event_data_false_alarm = pickle.load(f)
    # with open(join(save_loc, "event_data_miss.pkl"), 'rb') as f:
    #     event_data_miss = pickle.load(f)
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

    # -------------------------------
    # Load Global Trial History Data
    # -------------------------------
    try:
        with open(join(save_loc, 'fa_after_hit_data.pkl'), 'rb') as f:
            fa_after_hit_data = pickle.load(f)
        with open(join(save_loc, 'fa_not_after_hit_data.pkl'), 'rb') as f:
            fa_not_after_hit_data = pickle.load(f)
        with open(join(save_loc, 'cr_after_hit_data.pkl'), 'rb') as f:
            cr_after_hit_data = pickle.load(f)
        with open(join(save_loc, 'cr_not_after_hit_data.pkl'), 'rb') as f:
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
    # Global Analysis with Trial History
    # -------------------------------
    print("\n=== Global Analysis (All Frequencies Combined) ===")

    # CR vs. FA subsampling (balanced by trial history) -> distribution of auROCs per neuron
    N_samples = min(len(cr_event_averages[0]), len(fa_event_averages[0]))
    res_cr_subsample = compute_opto_windows_vs_baseline_subsampling(
        cr_event_averages,
        fa_event_averages,
        all_values,
        n_to_sample=N_samples,
        n_steps=100,
        n_boot=500,
        random_state=42,
        # Add trial history parameters
        fa_after_hit_data=fa_after_hit_data,
        fa_not_after_hit_data=fa_not_after_hit_data,
        cr_after_hit_data=cr_after_hit_data,
        cr_not_after_hit_data=cr_not_after_hit_data
    )

    # save the results
    with open(join(save_loc, 'res_cr_subsample_global.pkl'), 'wb') as f:
        pickle.dump(res_cr_subsample, f)

    # -------------------------------
    # Frequency-Specific Analysis with Trial History
    # -------------------------------
    print("\n=== Frequency-Specific Analysis ===")

    for freq in ['0.5', '1', '3', '5', '7.5', '10', '15']:
        print(f"\nProcessing frequency {freq} Hz:")

        # check if the data for the frequency exists
        try:
            with open(join(save_loc, f'all_iti_averages_{freq}.pkl'), 'rb') as f:
                all_light_event_freq = pickle.load(f)
            with open(join(save_loc, f'fa_iti_averages_{freq}.pkl'), 'rb') as f:
                fa_light_event_freq = pickle.load(f)
            with open(join(save_loc, f'cr_iti_averages_{freq}.pkl'), 'rb') as f:
                cr_light_event_freq = pickle.load(f)
            with open(join(save_loc, f'event_data_correct_reject_{freq}.pkl'), 'rb') as f:
                event_data_correct_reject_freq = pickle.load(f)
        except FileNotFoundError:
            print(f'No data for frequency {freq}, skipping...')
            continue

        # -------------------------------
        # Load Frequency-Specific Trial History Data
        # -------------------------------
        try:
            with open(join(save_loc, f'fa_after_hit_data_{freq}.pkl'), 'rb') as f:
                freq_fa_after_hit_data = pickle.load(f)
            with open(join(save_loc, f'fa_not_after_hit_data_{freq}.pkl'), 'rb') as f:
                freq_fa_not_after_hit_data = pickle.load(f)
            with open(join(save_loc, f'cr_after_hit_data_{freq}.pkl'), 'rb') as f:
                freq_cr_after_hit_data = pickle.load(f)
            with open(join(save_loc, f'cr_not_after_hit_data_{freq}.pkl'), 'rb') as f:
                freq_cr_not_after_hit_data = pickle.load(f)
            print(f"  Successfully loaded trial history data for {freq} Hz")
            use_trial_history_freq = True
        except FileNotFoundError:
            print(f"  Trial history data files not found for {freq} Hz. Using original sampling method.")
            freq_fa_after_hit_data = None
            freq_fa_not_after_hit_data = None
            freq_cr_after_hit_data = None
            freq_cr_not_after_hit_data = None
            use_trial_history_freq = False

        # compute the auROC for the different events
        all_baseline_cr_freq = event_data_correct_reject_freq['light_off']['baseline']
        baseline_freq = [np.asarray(all_baseline_cr_freq[i]) for i in range(len(all_baseline_cr_freq))]
        baseline_freq = [np.nanmean(baseline_freq[i], axis=1) for i in range(len(baseline_freq))]
        hit_event_limit = [neuron[10:10 + len(all_light_event_freq[0])] for neuron in hit_event_averages]

        print(f'  Number of trials for the frequency {freq} is {len(baseline_freq[0])}')


        if len(cr_light_event_freq[0]) == 0 or len(fa_light_event_freq[0]) == 0:
            print(f'  No data for frequency {freq}, skipping...')
            continue

        N_samples = min(len(cr_light_event_freq[0]), len(fa_light_event_freq[0]))
        if N_samples < 4:
            print(f'  Not enough samples for frequency {freq}, skipping...')
            continue

        # Frequency-specific Analysis with trial history
        cp_freq = compute_opto_windows_vs_baseline_subsampling(
            cr_light_event_freq,
            fa_light_event_freq,
            all_values,
            n_to_sample=N_samples,
            n_steps=100,
            n_boot=500,
            random_state=42,
            # Add frequency-specific trial history parameters
            fa_after_hit_data=freq_fa_after_hit_data,
            fa_not_after_hit_data=freq_fa_not_after_hit_data,
            cr_after_hit_data=freq_cr_after_hit_data,
            cr_not_after_hit_data=freq_cr_not_after_hit_data
        )

        with open(join(save_loc, f'cp_{freq}.pkl'), 'wb') as f:
            pickle.dump(cp_freq, f)

    # -------------------------------
    # Pooled Frequencies Analysis
    # -------------------------------
    print("\n=== Pooled Frequencies Analysis ===")

    # Initialize lists to store pooled data
    pooled_cr_data = []
    pooled_fa_data = []
    pooled_trial_history_data = {
        'fa_after_hit': [],
        'fa_not_after_hit': [],
        'cr_after_hit': [],
        'cr_not_after_hit': []
    }

    # Collect data from all frequencies that meet the minimum threshold
    for freq in ['0.5', '1', '3', '5', '7.5', '10', '15']:
        print(f"  Collecting data from {freq} Hz...")

        try:
            # Load frequency-specific data
            with open(join(save_loc, f'cr_iti_averages_{freq}.pkl'), 'rb') as f:
                cr_light_event_freq = pickle.load(f)
            with open(join(save_loc, f'fa_iti_averages_{freq}.pkl'), 'rb') as f:
                fa_light_event_freq = pickle.load(f)

            # Check if data exists and meets minimum threshold
            if len(cr_light_event_freq[0]) == 0 or len(fa_light_event_freq[0]) == 0:
                print(f"    No data for frequency {freq}, skipping...")
                continue

            N_samples = min(len(cr_light_event_freq[0]), len(fa_light_event_freq[0]))
            if N_samples < 2:
                print(f"    Not enough samples for frequency {freq} (N={N_samples}), skipping...")
                continue

            print(f"    Adding {N_samples} trials from {freq} Hz")
            # Add data to pooled lists
            pooled_cr_data.append(cr_light_event_freq)
            pooled_fa_data.append(fa_light_event_freq)

            # Try to load trial history data for this frequency
            try:
                with open(join(save_loc, f'fa_after_hit_data_{freq}.pkl'), 'rb') as f:
                    freq_fa_after_hit = pickle.load(f)
                with open(join(save_loc, f'fa_not_after_hit_data_{freq}.pkl'), 'rb') as f:
                    freq_fa_not_after_hit = pickle.load(f)
                with open(join(save_loc, f'cr_after_hit_data_{freq}.pkl'), 'rb') as f:
                    freq_cr_after_hit = pickle.load(f)
                with open(join(save_loc, f'cr_not_after_hit_data_{freq}.pkl'), 'rb') as f:
                    freq_cr_not_after_hit = pickle.load(f)

                # Add trial history data to pooled lists
                pooled_trial_history_data['fa_after_hit'].append(freq_fa_after_hit)
                pooled_trial_history_data['fa_not_after_hit'].append(freq_fa_not_after_hit)
                pooled_trial_history_data['cr_after_hit'].append(freq_cr_after_hit)
                pooled_trial_history_data['cr_not_after_hit'].append(freq_cr_not_after_hit)

            except FileNotFoundError:
                print(f"    Trial history data not found for {freq} Hz, will use None for this frequency")
                # Add None for this frequency's trial history data
                pooled_trial_history_data['fa_after_hit'].append(None)
                pooled_trial_history_data['fa_not_after_hit'].append(None)
                pooled_trial_history_data['cr_after_hit'].append(None)
                pooled_trial_history_data['cr_not_after_hit'].append(None)

        except FileNotFoundError:
            print(f"    Data files not found for frequency {freq}, skipping...")
            continue

    # Check if there is any pooled data
    if len(pooled_cr_data) == 0:
        print("  No valid data found for pooled Analysis!")
    else:
        print(f"  Successfully pooled data from {len(pooled_cr_data)} frequencies")

        # Combine the pooled data across frequencies
        # For each neuron, concatenate trials from all frequencies
        n_neurons = len(pooled_cr_data[0])  # Number of neurons
        print(f"  Number of neurons: {n_neurons}")

        combined_cr_data = []
        combined_fa_data = []

        for neuron_idx in range(n_neurons):
            # Collect all CR trials for this neuron across frequencies
            neuron_cr_trials = []
            for freq_data in pooled_cr_data:
                if neuron_idx < len(freq_data):
                    # Each freq_data[neuron_idx] should be shape (n_trials, 2)
                    # concatenate all trials from all frequencies
                    neuron_cr_trials.append(freq_data[neuron_idx])

            # Collect all FA trials for this neuron across frequencies
            neuron_fa_trials = []
            for freq_data in pooled_fa_data:
                if neuron_idx < len(freq_data):
                    # Each freq_data[neuron_idx] should be shape (n_trials,)
                    # concatenate all trials from all frequencies
                    neuron_fa_trials.append(freq_data[neuron_idx])

            # Concatenate all trials from all frequencies for this neuron
            if neuron_cr_trials:
                combined_cr_data.append(np.vstack(neuron_cr_trials))  # Shape: (total_trials, 2)
            else:
                combined_cr_data.append(np.array([]).reshape(0, 2))
                
            if neuron_fa_trials:
                combined_fa_data.append(np.concatenate(neuron_fa_trials))  # Shape: (total_trials,)
            else:
                combined_fa_data.append(np.array([]))

        # Determine sample size for pooled analysis
        min_cr_trials = min(len(cr) for cr in combined_cr_data if len(cr) > 0)
        min_fa_trials = min(len(fa) for fa in combined_fa_data if len(fa) > 0)
        N_samples_pooled = min(min_cr_trials, min_fa_trials)

        print(f"  Pooled Analysis: {N_samples_pooled} trials per neuron from {len(pooled_cr_data)} frequencies")

        # Combine trial history data across frequencies
        # For each neuron, concatenate trial history data from all frequencies
        combined_trial_history = None
        if any(data is not None for data in pooled_trial_history_data['fa_after_hit']):
            # Initialize combined trial history data structures
            n_neurons = len(pooled_trial_history_data['fa_after_hit'][0]) if pooled_trial_history_data['fa_after_hit'][0] is not None else 0
            
            combined_fa_after_hit = []
            combined_fa_not_after_hit = []
            combined_cr_after_hit = []
            combined_cr_not_after_hit = []

            for neuron_idx in range(n_neurons):
                # Collect trial history data for this neuron across all frequencies
                neuron_fa_after_hit = []
                neuron_fa_not_after_hit = []
                neuron_cr_after_hit = []
                neuron_cr_not_after_hit = []

                for freq_idx, freq_data in enumerate(pooled_trial_history_data['fa_after_hit']):
                    if freq_data is not None and neuron_idx < len(freq_data):
                        # Add trial history data from this frequency
                        neuron_fa_after_hit.append(freq_data[neuron_idx])
                        neuron_fa_not_after_hit.append(pooled_trial_history_data['fa_not_after_hit'][freq_idx][neuron_idx])
                        neuron_cr_after_hit.append(pooled_trial_history_data['cr_after_hit'][freq_idx][neuron_idx])
                        neuron_cr_not_after_hit.append(pooled_trial_history_data['cr_not_after_hit'][freq_idx][neuron_idx])

                # Concatenate trial history data for this neuron across frequencies
                if neuron_fa_after_hit:
                    combined_fa_after_hit.append(pd.concat(neuron_fa_after_hit, ignore_index=True))
                    combined_fa_not_after_hit.append(pd.concat(neuron_fa_not_after_hit, ignore_index=True))
                    combined_cr_after_hit.append(pd.concat(neuron_cr_after_hit, ignore_index=True))
                    combined_cr_not_after_hit.append(pd.concat(neuron_cr_not_after_hit, ignore_index=True))
                else:
                    # If no trial history data for this neuron, create empty DataFrames
                    combined_fa_after_hit.append(pd.DataFrame())
                    combined_fa_not_after_hit.append(pd.DataFrame())
                    combined_cr_after_hit.append(pd.DataFrame())
                    combined_cr_not_after_hit.append(pd.DataFrame())

            combined_trial_history = {
                'fa_after_hit_data': combined_fa_after_hit,
                'fa_not_after_hit_data': combined_fa_not_after_hit,
                'cr_after_hit_data': combined_cr_after_hit,
                'cr_not_after_hit_data': combined_cr_not_after_hit
            }

        # Run the pooled analysis with trial history
        if N_samples_pooled >= 2:
            pooled_results = compute_opto_windows_vs_baseline_subsampling(
                combined_cr_data,
                combined_fa_data,
                all_values,
                n_to_sample=N_samples_pooled,
                n_steps=100,
                n_boot=500,
                random_state=42,
                # Add trial history parameters if available
                fa_after_hit_data=combined_trial_history['fa_after_hit_data'] if combined_trial_history else None,
                fa_not_after_hit_data=combined_trial_history['fa_not_after_hit_data'] if combined_trial_history else None,
                cr_after_hit_data=combined_trial_history['cr_after_hit_data'] if combined_trial_history else None,
                cr_not_after_hit_data=combined_trial_history['cr_not_after_hit_data'] if combined_trial_history else None
            )

            # Save the pooled results
            with open(join(save_loc, 'cp_pooled_frequencies.pkl'), 'wb') as f:
                pickle.dump(pooled_results, f)

            print(f"  Pooled Analysis completed and saved!")
        else:
            print(f"  Not enough samples for pooled Analysis (N={N_samples_pooled})")

print("\nAll analyses completed!")