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
from scipy.interpolate import PchipInterpolator
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

    for i in range(n_neurons):
        neuron_data = all_data.iloc[i]
        # shape (#trials, 2) => col0 = event, col1= ...
        arr_2d = all_averages[i]
        if i >= len(all_baseline):
            continue  # or raise an error

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
            # Not enough data to do a 40-trial subrange
            continue

        # 1) Slide over the valid data in increments of 'step'
        max_auroc_sub = 0
        best_start = 0
        best_end   = 0

        # e.g. from 0..(n_valid-40) stepping by 10
        # ensures "end" = start+window_size <= n_valid
        for start_idx in range(0, n_valid - window_size + 1, step):
            end_idx = start_idx + window_size
            event_chunk = event_vals[start_idx:end_idx]
            base_chunk  = base_vals[start_idx:end_idx]

            # Compute real auROC (no permutation yet)
            labels = np.concatenate([np.ones(len(event_chunk)), np.zeros(len(base_chunk))])
            scores = np.concatenate([event_chunk, base_chunk])
            this_auroc = roc_auc_score(labels, scores)

            abs_diff = np.abs(this_auroc - 0.5)
            if abs_diff > max_auroc_sub:
                max_auroc_sub = abs_diff
                best_start = start_idx
                best_end   = end_idx

        # Now we have the best subrange
        # 2) Do the permutation test on that subrange only
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
# ---------- helpers for ROC panels (unchanged) ----------
def _roc_from_samples(response_vals, baseline_vals, n_thresh=400):
    r = np.asarray(response_vals, float); r = r[np.isfinite(r)]
    b = np.asarray(baseline_vals, float); b = b[np.isfinite(b)]
    if r.size == 0 or b.size == 0:
        return None, None, np.nan
    lo = min(r.min(), b.min()); hi = max(r.max(), b.max())
    thr = np.array([lo]) if (not np.isfinite(lo) or not np.isfinite(hi) or lo == hi) else np.linspace(lo, hi, n_thresh)
    x = np.array([(b > t).mean() for t in thr])   # baseline
    y = np.array([(r > t).mean() for t in thr])   # response
    order = np.argsort(x); x = x[order]; y = y[order]
    au = np.trapz(y, x)
    return x, y, float(au)

def _status_and_color(au, low=0.45, high=0.55):
    if np.isnan(au): return "Non significant", (0.2, 0.2, 0.2)
    if au > high:    return "Excited", (0.75, 0.1, 0.1)
    if au < low:     return "Inhibited", (0.1, 0.35, 0.8)
    return "Non significant", (0.2, 0.2, 0.2)

def _dedup_xy(x, y, reduce="mean"):
    """Sort by x, collapse duplicate x by mean or max of y."""
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.empty_like(xu, dtype=float)
    if reduce == "max":
        for k in range(len(xu)):
            yu[k] = np.max(y[inv == k])
    else:  # mean
        for k in range(len(xu)):
            yu[k] = np.mean(y[inv == k])
    return xu, yu

def _panel_roc(ax, x, y, au, title_txt, frame_color, smooth=True):
    if x is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    if smooth and len(x) > 3:
        # 1) collapse duplicate x to avoid spline errors
        xu, yu = _dedup_xy(x, y, reduce="mean")
        # 2) PCHIP keeps the curve monotone and avoids overshoot
        if len(xu) > 3 and np.all(np.isfinite(xu)) and np.all(np.isfinite(yu)):
            xnew = np.linspace(xu[0], xu[-1], 500)
            ynew = PchipInterpolator(xu, yu)(xnew)
            ax.fill_between(xnew, 0, ynew, alpha=0.15, facecolor="gray")
            ax.plot(xnew, ynew, color="gray", lw=2)
        else:
            ax.fill_between(x, 0, y, alpha=0.15, facecolor="gray")
            ax.plot(x, y, color="gray", lw=2)
    else:
        ax.fill_between(x, 0, y, alpha=0.15, facecolor="gray")
        ax.plot(x, y, color="gray", lw=2)

    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("P (F_baseline > threshold)")
    ax.set_ylabel("P (F_response > threshold)")
    for sp in ax.spines.values():
        sp.set_linewidth(3.0); sp.set_color(frame_color)
    ax.set_title(title_txt, color=frame_color)
    ax.text(0.03, 0.92, f"auROC = {au:.3f}", fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6))
# ---------- end helpers ----------
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
    cr_not_after_hit_data=None,
    make_roc_panels=False,
    roc_panel_neurons=None,
    roc_out_dir=None,
    picked_neurons=None,         # NEW: only run these neuron indices
):
    """
    For each neuron, do the following:
      - Let's define wA_vals = column 0 of all_averages[i].
      - baseline = all_baseline[i].
      - We assume wA_vals and baseline have the SAME # of valid trials
        after NaN removal.
      - For b in [1..n_boot]:
          1) random subset of size n_fa from the available trials
          2) compute the real auROC of that subset
        -> build a distribution of 1000 auROCs
      - Finally, store that distribution (or a summary) in 'results'.
      
      NEW: If trial history data is provided, balance sampling between
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
    # if user gave a list, restrict the loop to those indices
    if picked_neurons is not None and len(picked_neurons) > 0:
        run_indices = list(picked_neurons)
    else:
        run_indices = list(range(n_neurons))

    # preallocate, but we will only fill entries for run_indices
    all_distributions = [None] * n_neurons
    all_shuffled_distributions = [None] * n_neurons
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

    for i in run_indices:
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
    # ---------- create ROC panels if requested ----------
    if make_roc_panels:
        out_dir = roc_out_dir or "roc_panels"
        os.makedirs(out_dir, exist_ok=True)

        if not roc_panel_neurons or len(roc_panel_neurons) == 0:
            # if user did not pass an explicit list, use the picked list
            if picked_neurons and len(picked_neurons) > 0:
                roc_panel_neurons = list(picked_neurons)
            else:
                # fall back to auto-pick from whatever we computed
                med = np.array([np.nanmean(d) if d is not None else np.nan for d in all_distributions])
                chosen = []
                if np.any(np.isfinite(med)):
                    chosen.append(int(np.nanargmax(med)))
                    chosen.append(int(np.nanargmin(med)))
                    finite = np.where(np.isfinite(med))[0]
                    if finite.size:
                        mid_idx = finite[np.nanargmin(np.abs(med[finite] - 0.5))]
                        chosen.append(int(mid_idx))
                roc_panel_neurons = list(dict.fromkeys([ix for ix in chosen if ix is not None]))

        # one PDF per neuron
        for i in roc_panel_neurons:
            r_full = np.asarray(all_averages[i][:, 0], float)
            b_full = np.asarray(all_baseline[i], float)
            x, y, au = _roc_from_samples(r_full, b_full, n_thresh=2000)
            label, color = _status_and_color(au)

            fig, ax = plt.subplots(figsize=(4.0, 3.6))
            _panel_roc(ax, x, y, au, label, color, smooth=True)
            plt.tight_layout()
            pdf_path = os.path.join(out_dir, f"neuron_{i:04d}_roc_panel.pdf")
            plt.savefig(pdf_path)
            plt.close(fig)
            print("Saved ROC panel:", pdf_path)
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
if '__main__' == __name__:
    # for rep in range(1, 1):
    day = ['240827']
    licking_data = r'Z:\2p_data\OI151\{}\ML\licking_analysis_final'.format(day[0])
    save_loc = r'Z:\2p_data\OI151\{}\ML\data\processed_data\auROC_freq_history'.format(day[0])
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

    # compute opto windows subsampling vs fa in order to get the 95th and 5th percentile for the False Alarm
    N_samples = min(len(cr_event_averages[0]), len(fa_event_averages[0]))
    # res_cr_subsample = compute_opto_windows_vs_baseline_subsampling(
    #     cr_event_averages,
    #     fa_event_averages,
    #     all_values,
    #     n_to_sample=N_samples,
    #     n_steps=100,
    #     n_boot=500,
    #     random_state=42,
    #     # Add trial history parameters
    #     fa_after_hit_data=fa_after_hit_data,
    #     fa_not_after_hit_data=fa_not_after_hit_data,
    #     cr_after_hit_data=cr_after_hit_data,
    #     cr_not_after_hit_data=cr_not_after_hit_data
    # )
    picked = [6, 8, 10, 29, 79, 121, 181, 416, 483, 348]  # user provided
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
        cr_not_after_hit_data=cr_not_after_hit_data,
        picked_neurons=picked,  # runs only these indices
        make_roc_panels=True,  # optional
        roc_panel_neurons=picked,  # save one PDF per neuron
        roc_out_dir=os.path.join(save_loc, "roc_panels")
    )
    # save the results
    with open(join(save_loc, 'res_cr_subsample_plots.pkl'), 'wb') as f:
        pickle.dump(res_cr_subsample, f)

        # -------------------------------
        # Frequency-Specific Analysis with Trial History
        # -------------------------------
    #     print("\n=== Frequency-Specific Analysis ===")
    #
    #     for freq in ['0.5', '1', '3', '5', '7.5', '10', '15']:
    #         print(f"\nProcessing frequency {freq} Hz:")
    #
    #         # check if the data for the frequency exists
    #         try:
    #             with open(join(save_loc, f'all_light_averages_{freq}.pkl'), 'rb') as f:
    #                 all_light_event_freq = pickle.load(f)
    #             with open(join(save_loc, f'fa_light_averages_{freq}.pkl'), 'rb') as f:
    #                 fa_light_event_freq = pickle.load(f)
    #             with open(join(save_loc, f'cr_light_averages_{freq}.pkl'), 'rb') as f:
    #                 cr_light_event_freq = pickle.load(f)
    #             with open(join(save_loc, f'event_data_correct_reject_{freq}.pkl'), 'rb') as f:
    #                 event_data_correct_reject_freq = pickle.load(f)
    #         except FileNotFoundError:
    #             print(f'No data for frequency {freq}, skipping...')
    #             continue
    #
    #         # -------------------------------
    #         # Load Frequency-Specific Trial History Data
    #         # -------------------------------
    #         try:
    #             with open(join(save_loc, f'fa_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                 freq_fa_after_hit_data = pickle.load(f)
    #             with open(join(save_loc, f'fa_not_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                 freq_fa_not_after_hit_data = pickle.load(f)
    #             with open(join(save_loc, f'cr_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                 freq_cr_after_hit_data = pickle.load(f)
    #             with open(join(save_loc, f'cr_not_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                 freq_cr_not_after_hit_data = pickle.load(f)
    #             print(f"  Successfully loaded trial history data for {freq} Hz")
    #             use_trial_history_freq = True
    #         except FileNotFoundError:
    #             print(f"  Trial history data files not found for {freq} Hz. Using original sampling method.")
    #             freq_fa_after_hit_data = None
    #             freq_fa_not_after_hit_data = None
    #             freq_cr_after_hit_data = None
    #             freq_cr_not_after_hit_data = None
    #             use_trial_history_freq = False
    #
    #         # compute the auROC for the different events
    #         all_baseline_cr_freq = event_data_correct_reject_freq['light_off']['baseline']
    #         baseline_freq = [np.asarray(all_baseline_cr_freq[i]) for i in range(len(all_baseline_cr_freq))]
    #         baseline_freq = [np.nanmean(baseline_freq[i], axis=1) for i in range(len(baseline_freq))]
    #         hit_event_limit = [neuron[10:10 + len(all_light_event_freq[0])] for neuron in hit_event_averages]
    #
    #         print(f'  Number of trials for the frequency {freq} is {len(baseline_freq[0])}')
    #
    #
    #         if len(cr_light_event_freq[0]) == 0 or len(fa_light_event_freq[0]) == 0:
    #             print(f'  No data for frequency {freq}, skipping...')
    #             continue
    #
    #         N_samples = min(len(cr_light_event_freq[0]), len(fa_light_event_freq[0]))
    #         if N_samples < 4:
    #             print(f'  Not enough samples for frequency {freq}, skipping...')
    #             continue
    #
    #         # Frequency-specific Analysis with trial history
    #         cp_freq = compute_opto_windows_vs_baseline_subsampling(
    #             cr_light_event_freq,
    #             fa_light_event_freq,
    #             all_values,
    #             n_to_sample=N_samples,
    #             n_steps=100,
    #             n_boot=500,
    #             random_state=42,
    #             # Add frequency-specific trial history parameters
    #             fa_after_hit_data=freq_fa_after_hit_data,
    #             fa_not_after_hit_data=freq_fa_not_after_hit_data,
    #             cr_after_hit_data=freq_cr_after_hit_data,
    #             cr_not_after_hit_data=freq_cr_not_after_hit_data
    #         )
    #
    #         with open(join(save_loc, f'cp_{freq}.pkl'), 'wb') as f:
    #             pickle.dump(cp_freq, f)
    #
    #     # -------------------------------
    #     # Pooled Frequencies Analysis
    #     # -------------------------------
    #     print("\n=== Pooled Frequencies Analysis ===")
    #
    #     # Initialize lists to store pooled data
    #     pooled_cr_data = []
    #     pooled_fa_data = []
    #     pooled_trial_history_data = {
    #         'fa_after_hit': [],
    #         'fa_not_after_hit': [],
    #         'cr_after_hit': [],
    #         'cr_not_after_hit': []
    #     }
    #
    #     # Collect data from all frequencies that meet the minimum threshold
    #     for freq in ['0.5', '1', '3', '5', '7.5', '10', '15']:
    #         print(f"  Collecting data from {freq} Hz...")
    #
    #         try:
    #             # Load frequency-specific data
    #             with open(join(save_loc, f'cr_light_averages_{freq}.pkl'), 'rb') as f:
    #                 cr_light_event_freq = pickle.load(f)
    #             with open(join(save_loc, f'fa_light_averages_{freq}.pkl'), 'rb') as f:
    #                 fa_light_event_freq = pickle.load(f)
    #
    #             # Check if data exists and meets minimum threshold
    #             if len(cr_light_event_freq[0]) == 0 or len(fa_light_event_freq[0]) == 0:
    #                 print(f"    No data for frequency {freq}, skipping...")
    #                 continue
    #
    #             N_samples = min(len(cr_light_event_freq[0]), len(fa_light_event_freq[0]))
    #             if N_samples < 2:
    #                 print(f"    Not enough samples for frequency {freq} (N={N_samples}), skipping...")
    #                 continue
    #
    #             print(f"    Adding {N_samples} trials from {freq} Hz")
    #             # Add data to pooled lists
    #             pooled_cr_data.append(cr_light_event_freq)
    #             pooled_fa_data.append(fa_light_event_freq)
    #
    #             # Try to load trial history data for this frequency
    #             try:
    #                 with open(join(save_loc, f'fa_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                     freq_fa_after_hit = pickle.load(f)
    #                 with open(join(save_loc, f'fa_not_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                     freq_fa_not_after_hit = pickle.load(f)
    #                 with open(join(save_loc, f'cr_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                     freq_cr_after_hit = pickle.load(f)
    #                 with open(join(save_loc, f'cr_not_after_hit_data_{freq}.pkl'), 'rb') as f:
    #                     freq_cr_not_after_hit = pickle.load(f)
    #
    #                 # Add trial history data to pooled lists
    #                 pooled_trial_history_data['fa_after_hit'].append(freq_fa_after_hit)
    #                 pooled_trial_history_data['fa_not_after_hit'].append(freq_fa_not_after_hit)
    #                 pooled_trial_history_data['cr_after_hit'].append(freq_cr_after_hit)
    #                 pooled_trial_history_data['cr_not_after_hit'].append(freq_cr_not_after_hit)
    #
    #             except FileNotFoundError:
    #                 print(f"    Trial history data not found for {freq} Hz, will use None for this frequency")
    #                 # Add None for this frequency's trial history data
    #                 pooled_trial_history_data['fa_after_hit'].append(None)
    #                 pooled_trial_history_data['fa_not_after_hit'].append(None)
    #                 pooled_trial_history_data['cr_after_hit'].append(None)
    #                 pooled_trial_history_data['cr_not_after_hit'].append(None)
    #
    #         except FileNotFoundError:
    #             print(f"    Data files not found for frequency {freq}, skipping...")
    #             continue
    #
    #     # Check if we have any pooled data
    #     if len(pooled_cr_data) == 0:
    #         print("  No valid data found for pooled Analysis!")
    #     else:
    #         print(f"  Successfully pooled data from {len(pooled_cr_data)} frequencies")
    #
    #         # Combine the pooled data across frequencies
    #         # For each neuron, concatenate trials from all frequencies
    #         n_neurons = len(pooled_cr_data[0])  # Number of neurons
    #         print(f"  Number of neurons: {n_neurons}")
    #
    #         combined_cr_data = []
    #         combined_fa_data = []
    #
    #         for neuron_idx in range(n_neurons):
    #             # Collect all CR trials for this neuron across frequencies
    #             neuron_cr_trials = []
    #             for freq_data in pooled_cr_data:
    #                 if neuron_idx < len(freq_data):
    #                     # Each freq_data[neuron_idx] should be shape (n_trials, 2)
    #                     # We want to concatenate all trials from all frequencies
    #                     neuron_cr_trials.append(freq_data[neuron_idx])
    #
    #             # Collect all FA trials for this neuron across frequencies
    #             neuron_fa_trials = []
    #             for freq_data in pooled_fa_data:
    #                 if neuron_idx < len(freq_data):
    #                     # Each freq_data[neuron_idx] should be shape (n_trials,)
    #                     # We want to concatenate all trials from all frequencies
    #                     neuron_fa_trials.append(freq_data[neuron_idx])
    #
    #             # Concatenate all trials from all frequencies for this neuron
    #             if neuron_cr_trials:
    #                 combined_cr_data.append(np.vstack(neuron_cr_trials))  # Shape: (total_trials, 2)
    #             else:
    #                 combined_cr_data.append(np.array([]).reshape(0, 2))
    #
    #             if neuron_fa_trials:
    #                 combined_fa_data.append(np.concatenate(neuron_fa_trials))  # Shape: (total_trials,)
    #             else:
    #                 combined_fa_data.append(np.array([]))
    #
    #         # Determine sample size for pooled Analysis
    #         min_cr_trials = min(len(cr) for cr in combined_cr_data if len(cr) > 0)
    #         min_fa_trials = min(len(fa) for fa in combined_fa_data if len(fa) > 0)
    #         N_samples_pooled = min(min_cr_trials, min_fa_trials)
    #
    #         print(f"  Pooled Analysis: {N_samples_pooled} trials per neuron from {len(pooled_cr_data)} frequencies")
    #
    #         # Combine trial history data across frequencies
    #         # For each neuron, concatenate trial history data from all frequencies
    #         combined_trial_history = None
    #         if any(data is not None for data in pooled_trial_history_data['fa_after_hit']):
    #             # Initialize combined trial history data structures
    #             n_neurons = len(pooled_trial_history_data['fa_after_hit'][0]) if pooled_trial_history_data['fa_after_hit'][0] is not None else 0
    #
    #             combined_fa_after_hit = []
    #             combined_fa_not_after_hit = []
    #             combined_cr_after_hit = []
    #             combined_cr_not_after_hit = []
    #
    #             for neuron_idx in range(n_neurons):
    #                 # Collect trial history data for this neuron across all frequencies
    #                 neuron_fa_after_hit = []
    #                 neuron_fa_not_after_hit = []
    #                 neuron_cr_after_hit = []
    #                 neuron_cr_not_after_hit = []
    #
    #                 for freq_idx, freq_data in enumerate(pooled_trial_history_data['fa_after_hit']):
    #                     if freq_data is not None and neuron_idx < len(freq_data):
    #                         # Add trial history data from this frequency
    #                         neuron_fa_after_hit.append(freq_data[neuron_idx])
    #                         neuron_fa_not_after_hit.append(pooled_trial_history_data['fa_not_after_hit'][freq_idx][neuron_idx])
    #                         neuron_cr_after_hit.append(pooled_trial_history_data['cr_after_hit'][freq_idx][neuron_idx])
    #                         neuron_cr_not_after_hit.append(pooled_trial_history_data['cr_not_after_hit'][freq_idx][neuron_idx])
    #
    #                 # Concatenate trial history data for this neuron across frequencies
    #                 if neuron_fa_after_hit:
    #                     combined_fa_after_hit.append(pd.concat(neuron_fa_after_hit, ignore_index=True))
    #                     combined_fa_not_after_hit.append(pd.concat(neuron_fa_not_after_hit, ignore_index=True))
    #                     combined_cr_after_hit.append(pd.concat(neuron_cr_after_hit, ignore_index=True))
    #                     combined_cr_not_after_hit.append(pd.concat(neuron_cr_not_after_hit, ignore_index=True))
    #                 else:
    #                     # If no trial history data for this neuron, create empty DataFrames
    #                     combined_fa_after_hit.append(pd.DataFrame())
    #                     combined_fa_not_after_hit.append(pd.DataFrame())
    #                     combined_cr_after_hit.append(pd.DataFrame())
    #                     combined_cr_not_after_hit.append(pd.DataFrame())
    #
    #             combined_trial_history = {
    #                 'fa_after_hit_data': combined_fa_after_hit,
    #                 'fa_not_after_hit_data': combined_fa_not_after_hit,
    #                 'cr_after_hit_data': combined_cr_after_hit,
    #                 'cr_not_after_hit_data': combined_cr_not_after_hit
    #             }
    #
    #         # Run the pooled Analysis with trial history
    #         if N_samples_pooled >= 2:
    #             pooled_results = compute_opto_windows_vs_baseline_subsampling(
    #                 combined_cr_data,
    #                 combined_fa_data,
    #                 all_values,
    #                 n_to_sample=N_samples_pooled,
    #                 n_steps=100,
    #                 n_boot=500,
    #                 random_state=42,
    #                 # Add trial history parameters if available
    #                 fa_after_hit_data=combined_trial_history['fa_after_hit_data'] if combined_trial_history else None,
    #                 fa_not_after_hit_data=combined_trial_history['fa_not_after_hit_data'] if combined_trial_history else None,
    #                 cr_after_hit_data=combined_trial_history['cr_after_hit_data'] if combined_trial_history else None,
    #                 cr_not_after_hit_data=combined_trial_history['cr_not_after_hit_data'] if combined_trial_history else None
    #             )
    #
    #             # Save the pooled results
    #             with open(join(save_loc, 'cp_pooled_frequencies.pkl'), 'wb') as f:
    #                 pickle.dump(pooled_results, f)
    #
    #             print(f"  Pooled Analysis completed and saved!")
    #         else:
    #             print(f"  Not enough samples for pooled Analysis (N={N_samples_pooled})")
    #
    # print("\nAll analyses completed!")