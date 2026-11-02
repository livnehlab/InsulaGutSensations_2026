#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Psychophysics: SVM perturbation analysis of the 15 Hz inhibited neurons (one session).

The dF/F of each neuron is z-scored and split into trials, and the mean response in the
stimulus window is taken for every trial. Neurons are ranked by their median response on
the 15 Hz correct-reject (CR) trials (most inhibited first). The perturbation adds
multiplier x (median inhibition on 15 Hz CR trials) to the responses of a chosen subset
of neurons on all false-alarm (FA) trials.

A linear SVM is trained on the unperturbed data (15 Hz CR trials = class 0, 0 Hz trials
= class 1) and tested on the perturbed data at every frequency; at 15 Hz the test set is
the left-over CR trials plus all FA trials. Train and test trials never overlap.

Neuron subsets (each run with multipliers 0, 0.5, 1, 2, 5, 10 and 20):
  - five consecutive 20% chunks of the ranking  -> combined_perturb_multiplier_results.pkl
  - five 4% chunks within the 20% most inhibited -> combined_perturb_multiplier_results_4pct_chunks.pkl
  - optional (RUN_PERCENT_STEPS): cumulative top 10-100% -> combined_perturb_percent_results.pkl

The 20% chunk results are then plotted (accuracy per chunk, per multiplier, and heat maps).
Usage: python SVM_classifier_psychophysics_specific_cells.py [run index ...]  (default: all RUNS)
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard Library Imports
import os
import csv
import math
import pickle
import copy
from operator import itemgetter
from os.path import join

# Third-Party Imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit

# Scikit-learn Imports
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION AND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Matplotlib Configuration
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

# Experimental Constants
start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000
pre_trial_length = 4 * rec_hz
baseline_len = 4 * rec_hz

# Analysis Constants
n_priming_trials = 23
stimulus_start_offset = 1 * rec_hz + 1 * rec_hz  # relative to pre_trial_length
stimulus_duration = 5 * rec_hz
frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
target_freq_code = 7  # condition code for 15Hz
target_resp_code = 4   # correct reject
RUN_PERCENT_STEPS = False  # also run the cumulative top 10-100% analysis (combined_perturb_percent_results.pkl)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def divide_to_trials(f_signal, trials_ind, lengths, len_pre_trial):
    """
    Divides the f_signal into trials based on provided indices and run lengths.
    Returns:
      - f_run_list_final: List of divided signals.
      - pre_runs: The signal from the beginning of each run.
    """
    f_run_list_final = []
    for i in range(len(f_signal)):  # for each neuron
        f_run_list = []
        pre_run_list = []
        f_run_list_all = []
        start_ind = 0
        for run_ind in range(len(lengths)):
            end_ind = start_ind + lengths[run_ind]
            f_run = np.split(f_signal.iloc[i][start_ind:end_ind], trials_ind[run_ind], axis=0)
            f_run_list.append(f_run[1:])
            pre_run_list.append(f_run[0])
            start_ind = end_ind
        # add the pre-trial to each trial
        for run in range(len(f_run_list)):
            for j in range(len(f_run_list[run])):
                if j == 0:  # first trial of each run
                    pre_trial = pre_run_list[run][len(pre_run_list[run]) - len_pre_trial:]
                else:
                    pre_trial = f_run_list[run][j - 1][len(f_run_list[run][j - 1]) - len_pre_trial:]
                f_run_list[run][j] = list(pre_trial) + list(f_run_list[run][j])
            f_run_list_all.extend(f_run_list[run])
        f_run_list_final.append(f_run_list_all)
    return f_run_list_final, pre_run_list


def baseline_subtract_neurons(norm_data, pre_trial_length, baseline_time):
    """Subtracts the baseline activity from each neuron in each trial."""
    concat_data = copy.deepcopy(norm_data)
    for j, trial in enumerate(concat_data):
        for i, neuron in enumerate(trial):
            neuron = np.asarray(neuron)
            baseline = np.mean(neuron[pre_trial_length - baseline_time:pre_trial_length])
            concat_data[j][i] = neuron - baseline
    return concat_data


def global_zscore_neurons(data):
    """
    Z-scores each neuron across all trials/time points.
    data is a list of trials, where each trial is a list of neurons (1D np.array).
    """
    zscored_data = copy.deepcopy(data)
    num_trials = len(zscored_data)
    if num_trials == 0:
        return zscored_data
    num_neurons = len(zscored_data[0])
    for neuron_idx in range(num_neurons):
        all_time_points = []
        for trial_idx in range(num_trials):
            all_time_points.append(zscored_data[trial_idx][neuron_idx])
        all_time_points = np.concatenate(all_time_points)
        mu = np.mean(all_time_points)
        sigma = np.std(all_time_points)
        if sigma == 0:
            sigma = 1.0
        for trial_idx in range(num_trials):
            zscored_data[trial_idx][neuron_idx] = (zscored_data[trial_idx][neuron_idx] - mu) / sigma
    return zscored_data


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def main_classifier(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# def main_classifier(X_train, y_train, X_test, y_test):
#     # Replace SVM with Gaussian Naive Bayes
#     clf = GaussianNB()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy

def train_SVM_no_trial_overlap(
    original_pop_vectors,
    perturbed_pop_vectors,
    cr_rows_15hz,               # rows that are CR in the 15 Hz matrix
    fa_rows_15hz,               # rows that are FA in the 15 Hz matrix
    n_iterations=1000,
    train_size=0.7,
    test_size=0.3,
):
    """
    Trains on the original data and tests on the perturbed data, for every frequency.
    Class 0 = 15 Hz trials (CR or FA), class 1 = 0 Hz ('Hit') trials.

    Training draws its 15 Hz samples only from `cr_rows_15hz`.
    The 15 Hz test set is the left-over CR rows plus all FA rows (trimmed to the
    size of the 0 Hz test set). Returns the mean accuracy and SEM per frequency.
    """
    freq_cr  = 15        # 15Hz key in the population dictionaries
    freq_hit = 0        # 0Hz key

    n_cr  = len(cr_rows_15hz)
    n_hit = original_pop_vectors[freq_hit].shape[0]
    n_tr  = int(train_size * min(n_cr, n_hit))
    n_te  = int(test_size  * min(n_cr, n_hit))

    mean_acc, sem_acc = [], []

    for freq in perturbed_pop_vectors:        # loop over every test frequency
        acc = []
        for _ in range(n_iterations):
            # ── TRAIN rows ──────────────────────────────────────
            cr_train = np.random.choice(cr_rows_15hz, n_tr, replace=False)
            hit_rows = np.arange(n_hit)
            hit_tr, hit_te = train_test_split(hit_rows,
                                              train_size=n_tr, test_size=n_te)

            Xtr = np.vstack([original_pop_vectors[freq_cr][cr_train],
                              original_pop_vectors[freq_hit][hit_tr]])
            ytr = np.concatenate([np.zeros(len(cr_train)),
                                  np.ones(len(hit_tr))])
            # ── TEST rows for this 'freq' ──────────────────────
            if freq == freq_cr:
                cr_te = np.setdiff1d(cr_rows_15hz, cr_train)
                test_rows = np.concatenate([cr_te, fa_rows_15hz]).astype(int)

                # balance: randomly trim to match n_te (size of hit_te)
                # (remove the two lines below to keep the larger 15 Hz test set)
                if len(test_rows) > len(hit_te):
                    test_rows = np.random.choice(test_rows, len(hit_te), replace=False)

                Xte = perturbed_pop_vectors[freq][test_rows]
                yte = np.zeros(len(test_rows))

            elif freq == freq_hit:                        # 0Hz panel
                Xte = perturbed_pop_vectors[freq][hit_te]
                yte = np.ones(len(hit_te))

            else:                                         # any other freq
                all_idx = np.arange(perturbed_pop_vectors[freq].shape[0])
                rnd_idx = resample(all_idx, n_samples=n_te, replace=False)
                Xte = perturbed_pop_vectors[freq][rnd_idx]
                yte = np.zeros(len(rnd_idx))

            # shuffle once
            Xtr, ytr = shuffle(Xtr, ytr)
            Xte, yte = shuffle(Xte, yte)

            # ── Sanity checks: ensure no train/test overlap by indices ─────────
            if freq == freq_cr:
                # no overlap between CR train and CR/FA test rows
                overlap_cr = set(map(int, cr_train.tolist())) & set(map(int, test_rows.tolist()))
                if overlap_cr:
                    raise ValueError(f"Overlap between 15Hz CR train and test rows detected: {sorted(list(overlap_cr))[:5]} ...")
            elif freq == freq_hit:
                # no overlap between HIT train and HIT test rows
                overlap_hit = set(map(int, hit_tr.tolist())) & set(map(int, hit_te.tolist()))
                if overlap_hit:
                    raise ValueError(f"Overlap between 0Hz HIT train and test rows detected: {sorted(list(overlap_hit))[:5]} ...")

            acc.append(main_classifier(Xtr, ytr, Xte, yte))

        mean_acc.append(np.mean(acc))
        sem_acc .append(np.std(acc) / np.sqrt(n_iterations))

    return np.array(mean_acc), np.array(sem_acc)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION AND PLOTTING FUNCTIONS  
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_for_freq(freq, orig_pop, pert_pop, conditions_played, responses, frequency_map, save_loc, title_suffix=""):
    """
    2-D PCA of the 0 Hz trials and the CR / FA trials of one frequency
    (FA shown both original and perturbed); saves pca_plot_<freq>Hz<title_suffix>.png.
    """
    # training HIT rows (0Hz)
    freq_hit = 0
    hit_rows = np.arange(orig_pop[freq_hit].shape[0])

    # Build frequency-specific CR and FA indices for this frequency
    correct_reject = np.where(responses == 4)[0]
    false_alarm = np.where(responses == 5)[0]
    
    # Find which condition code corresponds to this frequency
    freq_condition_code = None
    for code, f in frequency_map.items():
        if f == freq:
            freq_condition_code = code
            break
    
    if freq_condition_code is None:
        print(f"Warning: No condition code found for frequency {freq}")
        return
        
    # Build row indices for this specific frequency
    row_counter = 0
    cr_rows_this_freq, fa_rows_this_freq = [], []
    
    for trial_idx, cond in enumerate(conditions_played):
        if frequency_map[cond] == freq:  # only this frequency trials get rows
            if trial_idx in correct_reject:
                cr_rows_this_freq.append(row_counter)
            elif trial_idx in false_alarm:
                fa_rows_this_freq.append(row_counter)
            row_counter += 1  # advance only for this frequency's rows
    
    # Convert to numpy arrays
    cr_rows_this_freq = np.asarray(cr_rows_this_freq, dtype=int)
    fa_rows_this_freq = np.asarray(fa_rows_this_freq, dtype=int)
    
    # Check if we have enough data for meaningful plots
    if len(cr_rows_this_freq) == 0 and len(fa_rows_this_freq) == 0:
        print(f"Warning: No CR or FA trials found for frequency {freq}")
        return
    
    # Extract data based on available trial types
    Xhit = orig_pop[freq_hit][hit_rows]
    
    if len(cr_rows_this_freq) > 0 and len(fa_rows_this_freq) > 0:
        # We have both CR and FA trials
        Xcr = orig_pop[freq][cr_rows_this_freq]
        Xfa_orig = orig_pop[freq][fa_rows_this_freq]
        Xfa_pert = pert_pop[freq][fa_rows_this_freq]
        
        # concatenate -> fit PCA in common space
        Xcat = np.vstack([Xcr, Xhit, Xfa_orig, Xfa_pert])
        pca = PCA(n_components=2)
        X2D = pca.fit_transform(Xcat)
        
        # split back
        nCR = len(Xcr)
        nHit = len(Xhit)
        nFA = len(Xfa_orig)
        
        CR2D = X2D[:nCR]
        HIT2D = X2D[nCR:nCR+nHit]
        FAorig2D = X2D[nCR+nHit:nCR+nHit+nFA]
        FApert2D = X2D[nCR+nHit+nFA:]
        
        plt.figure(figsize=(5,5))
        plt.scatter(CR2D[:,0], CR2D[:,1], c='grey', alpha=0.5, label="CR")
        plt.scatter(HIT2D[:,0], HIT2D[:,1], c='green', alpha=0.5, label="HIT 0Hz")
        plt.scatter(FAorig2D[:,0], FAorig2D[:,1], c='blue', alpha=0.7, label=f"FA {freq}Hz (orig)")
        plt.scatter(FApert2D[:,0], FApert2D[:,1], c='red', alpha=0.7, label=f"FA {freq}Hz (pert.)")
        
    elif len(fa_rows_this_freq) > 0:
        # We only have FA trials (no CR)
        Xfa_orig = orig_pop[freq][fa_rows_this_freq]
        Xfa_pert = pert_pop[freq][fa_rows_this_freq]
        
        Xcat = np.vstack([Xhit, Xfa_orig, Xfa_pert])
        pca = PCA(n_components=2)
        X2D = pca.fit_transform(Xcat)
        
        nHit = len(Xhit)
        nFA = len(Xfa_orig)
        
        HIT2D = X2D[:nHit]
        FAorig2D = X2D[nHit:nHit+nFA]
        FApert2D = X2D[nHit+nFA:]
        
        plt.figure(figsize=(5,5))
        plt.scatter(HIT2D[:,0], HIT2D[:,1], c='green', alpha=0.5, label="HIT 0Hz")
        plt.scatter(FAorig2D[:,0], FAorig2D[:,1], c='blue', alpha=0.7, label=f"FA {freq}Hz (orig)")
        plt.scatter(FApert2D[:,0], FApert2D[:,1], c='red', alpha=0.7, label=f"FA {freq}Hz (pert.)")
        
    elif len(cr_rows_this_freq) > 0:
        # We only have CR trials (no FA), so nothing is perturbed
        Xcr = orig_pop[freq][cr_rows_this_freq]
        
        Xcat = np.vstack([Xcr, Xhit])
        pca = PCA(n_components=2)
        X2D = pca.fit_transform(Xcat)
        
        nCR = len(Xcr)
        nHit = len(Xhit)
        
        CR2D = X2D[:nCR]
        HIT2D = X2D[nCR:]
        
        plt.figure(figsize=(5,5))
        plt.scatter(CR2D[:,0], CR2D[:,1], c='grey', alpha=0.5, label="CR")
        plt.scatter(HIT2D[:,0], HIT2D[:,1], c='green', alpha=0.5, label="HIT 0Hz")
    
    plt.title(f"{freq} Hz  {title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, f"pca_plot_{freq}Hz{title_suffix}.png"), dpi=300)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION AND ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(ROOT_PATH, save_loc):
    """Runs the full perturbation analysis for one session and saves the results and plots to save_loc."""
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ═════════════════════════════════════════════════════════════════════════
    conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
    responses = np.load(join(ROOT_PATH, 'responses.npy'))
    with open(join(ROOT_PATH, 'valve1_open_all.pkl'), 'rb') as f:
        valve_open = pickle.load(f)
    with open(join(ROOT_PATH, 'opto_open_all.pkl'), 'rb') as f:
        opto_on = pickle.load(f)
    with open(join(ROOT_PATH, 'distract_light_on_all.pkl'), 'rb') as f:
        distract_light = pickle.load(f)
    trials_ind = np.load(join(ROOT_PATH, 'trials_ind.npy'))
    run_len = np.load(join(ROOT_PATH, 'run_len.npy'))
    with open(join(ROOT_PATH, 'dff_running_4.5min.pkl'), 'rb') as f:
        dff_4min = pickle.load(f)

    # remove the priming trials
    responses = responses[n_priming_trials:]
    conditions_played = conditions_played[n_priming_trials:]
    trials_ind = [trials_ind[0][n_priming_trials:], trials_ind[1], trials_ind[2]]
    opto_on = opto_on[n_priming_trials:]
    distract_light = distract_light[n_priming_trials:]
    valve_open = valve_open[n_priming_trials:]

    # z-score each neuron over the whole recording, then split into trials
    dff_norm = []
    for i in range(len(dff_4min)):
        neuron_trace = dff_4min.iloc[i]
        mean = np.nanmean(neuron_trace)
        std = np.nanstd(neuron_trace)
        dff_norm.append((neuron_trace - mean) / std)
    dff_norm = pd.DataFrame(dff_norm)
    dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)
    with open(join(save_loc, 'dff_run_list.pkl'), 'wb') as f:
        pickle.dump(dff_run_list, f)


    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: RESPONSE CATEGORIZATION AND BASIC SETUP  
    # ═════════════════════════════════════════════════════════════════════════
    correct_reject = np.where(responses == 4)[0]
    false_alarm    = np.where(responses == 5)[0]
    hit            = np.where(responses == 3)[0]
    miss           = np.where(responses == 2)[0]


    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: PERTURBATION SETUP AND TARGET IDENTIFICATION
    # ═════════════════════════════════════════════════════════════════════════
    con_to_perturb = false_alarm  # trials whose responses are perturbed (all FA trials)
    local_frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    local_target_freq_code = 7  # condition code for specific frequency (15Hz)
    local_target_resp_code = 4   # correct reject
    target_indices = [i for i, (cond, resp) in enumerate(zip(conditions_played, responses))
                      if cond == local_target_freq_code and resp == local_target_resp_code]

    print(f"Found {len(target_indices)} CR trials at 15 Hz.")
    stimu_start = pre_trial_length + stimulus_start_offset
    stimu_end   = pre_trial_length + stimulus_duration

    # inhibition size = median of all negative stimulus-window responses on 15 Hz CR trials
    inhibitory_values = []
    for neuron in dff_run_list:
        for trial_idx in target_indices:
            trial = neuron[trial_idx]
            stim_val = np.mean(trial[stimu_start:stimu_end])
            if stim_val < 0:
                inhibitory_values.append(stim_val)

    inhibitory_values = np.array(inhibitory_values)
    print(f"Collected {len(inhibitory_values)} inhibitory responses.")

    mean_inhibition   = np.mean(inhibitory_values)
    median_inhibition = np.median(inhibitory_values)

    print(f"Mean inhibition (15Hz CR): {mean_inhibition:.3f}")
    print(f"Median inhibition (15Hz CR): {median_inhibition:.3f}")
    addition_value = median_inhibition

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: NEURON RANKING BY INHIBITION STRENGTH
    # ═════════════════════════════════════════════════════════════════════════
    activity_scores = []  # one value per neuron
    for neuron_idx, neuron in enumerate(dff_run_list):
        # gather this neuron's responses on CR trials
        stim_vals = []
        for trial_idx in target_indices:  # CR trials @15Hz
            trial = neuron[trial_idx]
            stim_vals.append(np.mean(trial[stimu_start:stimu_end]))
        # more negative -> more inhibited
        if len(stim_vals) == 0:
            activity_scores.append(0)  # no data -> neutral
        else:
            activity_scores.append(np.median(stim_vals))  # use median
    activity_scores = np.array(activity_scores)


    # Sorted indices: 0 = most inhibited, 1 = 2nd most, ...
    ranked_neurons = np.argsort(activity_scores)  # ascending
    # and descending order
    # ranked_neurons_desc = np.argsort(activity_scores)[::-1]  # descending
    # sorted_inhib_scores = np.sort(activity_scores)  # ascending

    # threshold for the 20% most inhibited neurons (more negative = stronger); used for the histogram
    num_neurons = len(activity_scores)
    n_top20 = int(np.ceil(0.20 * num_neurons)) - 1
    top20_threshold_val = activity_scores[ranked_neurons[n_top20]]  # value at that rank
    top_20_neurons = ranked_neurons[:n_top20 + 1]  # indices of the top 20%
    # CP_PATH = DATA_ROOT + r"\mouse09\sess06\ML\data\processed_data\auROC_freq"
    # cp_idx = np.load(join(CP_PATH, "high_cp_neurons.npy")).astype(int)
    # both = np.intersect1d(cp_idx, top_20_neurons)
    plt.figure(figsize=(6, 4))
    sns.histplot(activity_scores, bins=50, kde=True, color="steelblue")
    plt.axvline(top20_threshold_val, color="red", linestyle="--",
                label=f"20% cut‑off = {top20_threshold_val:.3f}")
    plt.fill_betweenx([0, plt.ylim()[1]], activity_scores.min(), top20_threshold_val,
                      color="red", alpha=0.15)

    plt.xlabel("Median ΔF/F during stim window\n(15Hz CR trials)")
    plt.ylabel("Number of neurons")
    plt.title("")  # Distribution of neuronal inhibition and 20% strongest cut-off
    plt.legend()
    plt.tight_layout()


    # save
    plot_path = os.path.join(save_loc, "inhibition_distribution_top20.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5: PERTURBATION EXPERIMENTS SETUP
    # ═════════════════════════════════════════════════════════════════════════
    multipliers = [0, 0.5, 1, 2, 5, 10, 20]
    # ------------------------------------------------------------------
    # Neuron subsets are selected once from the ranking above, so every
    # multiplier uses the same subsets.
    # ------------------------------------------------------------------
    num_neurons = len(ranked_neurons)

    # ────────────────────────────────────────────────────────────────
    # Row indices of the CR and FA trials inside the 15 Hz population matrix
    # (built once and kept as numpy arrays)
    # ────────────────────────────────────────────────────────────────
    freq_cr = 15  # 15Hz key
    row_counter = 0
    cr_rows_15hz, fa_rows_15hz = [], []

    for trial_idx, cond in enumerate(conditions_played):
        if local_frequency_map[cond] == freq_cr:  # only 15Hz trials get rows
            if trial_idx in correct_reject:
                cr_rows_15hz.append(row_counter)
            elif trial_idx in false_alarm:
                fa_rows_15hz.append(row_counter)
            row_counter += 1  # advance only for 15Hz rows

    # convert to int arrays (ready for indexing)
    cr_rows_15hz = np.asarray(cr_rows_15hz, dtype=int)
    fa_rows_15hz = np.asarray(fa_rows_15hz, dtype=int)
    
    # Validate we have necessary data
    print(f"Found {len(cr_rows_15hz)} CR trials and {len(fa_rows_15hz)} FA trials at 15Hz")
    if len(fa_rows_15hz) == 0:
        print("WARNING: No False Alarm trials found at 15Hz frequency.")
        print("Some analyses (perturbation mechanism) will be skipped.")
    
    # ------------------------------------------------------------------
    # (1) Five disjoint 20% chunks of the ranking (chunk 0 = most inhibited)
    # ------------------------------------------------------------------
    subsets_fixed = {}

    # Integer chunking that exactly partitions the ranking into five contiguous blocks
    for k in range(5):
        start_idx = int(np.floor(k     * num_neurons / 5.0))
        end_idx   = int(np.floor((k+1) * num_neurons / 5.0)) if k < 4 else num_neurons
        key = 0.2 * k if k > 0 else 0  # keep existing keys: 0, 0.2, 0.4, 0.6, 0.8
        subsets_fixed[key] = set(ranked_neurons[start_idx:end_idx])

    # Also define a "baseline" subset that is empty,
    # meaning "no neurons perturbed".
    subsets_fixed["none"] = set()

    # Dictionary that holds the final results
    perturbation_results = {}

    # Loop over all subsets: "none" plus each 20% chunk
    for chunk_key in subsets_fixed:
        perturbed_indices = subsets_fixed[chunk_key]
        print(f"\nUsing subset {chunk_key} with {len(perturbed_indices)} neurons")

        # Then loop over each multiplier
        for mult in multipliers:
            # (A) Build the original and perturbed responses per neuron and frequency
            original_activity_dict_neurons = []
            perturbed_activity_dict_neurons = []

            for neuron_idx in range(num_neurons):
                neuron_activities = dff_run_list[neuron_idx]

                orig_dict = {}
                pert_dict = {}
                for fval in local_frequency_map.values():
                    orig_dict[fval] = []
                    pert_dict[fval] = []

                for trial_idx, (cond, acts) in enumerate(zip(conditions_played, neuron_activities)):
                    freq = local_frequency_map[cond]
                    stim_value = np.mean(acts[stimu_start:stimu_end])

                    # Original is unmodified
                    orig_dict[freq].append(stim_value)

                    # Perturbed only for the selected neurons on the trials to perturb
                    if (neuron_idx in perturbed_indices) and (trial_idx in con_to_perturb):
                        pert_dict[freq].append(stim_value + (addition_value * mult))
                    else:
                        pert_dict[freq].append(stim_value)

                original_activity_dict_neurons.append(orig_dict)
                perturbed_activity_dict_neurons.append(pert_dict)

            # (B) Convert to shape (nTrials, nNeurons)
            freq_keys = sorted(set(local_frequency_map.values()))
            original_population_vectors = {f: [] for f in freq_keys}
            perturbed_population_vectors = {f: [] for f in freq_keys}

            for f in freq_keys:
                tmp_orig_list = []
                tmp_pert_list = []
                for neuron_idx in range(num_neurons):
                    tmp_orig_list.append(original_activity_dict_neurons[neuron_idx][f])
                    tmp_pert_list.append(perturbed_activity_dict_neurons[neuron_idx][f])
                # shape: (nNeurons, nTrials) => transpose => (nTrials, nNeurons)
                original_population_vectors[f] = np.array(tmp_orig_list).T
                perturbed_population_vectors[f] = np.array(tmp_pert_list).T



            # (C) Run classification
            avg_accuracy, sem_all = train_SVM_no_trial_overlap(
                original_population_vectors,
                perturbed_population_vectors,
                cr_rows_15hz,
                fa_rows_15hz,
                n_iterations=500,
            )

            # (D) Store final
            perturbation_results[(chunk_key, mult)] = {
                "avg_accuracy": np.array(avg_accuracy),
                "sem_all": np.array(sem_all),
                "subset": perturbed_indices
            }


    # Save results
    os.makedirs(save_loc, exist_ok=True)
    with open(os.path.join(save_loc, "combined_perturb_multiplier_results.pkl"), "wb") as f:
        pickle.dump(perturbation_results, f)

    print("\nDone. The same subsets are used for each multiplier at a given percentage!\n")

    # ------------------------------------------------------------------
    # (2) Five 4% chunks within the 20% most inhibited neurons
    #     (step = 20% of the neurons; each chunk is 0.2 * step)
    # ------------------------------------------------------------------
    num_neurons = len(ranked_neurons)
    step = num_neurons // 5
    subsets_fixed = {}

    # for i in range(5):
    #     start_idx = i * step
    #     end_idx = (i + 1) * step if i < 4 else num_neurons
    #     subsets_fixed[i] = set(ranked_neurons[start_idx:end_idx])
    for i in np.arange(0.0, 1.0, 0.2):
        start_idx = i * step
        start_idx = int(start_idx)  # round down
        # end_idx = (i + 1) * step if i < 4 else num_neurons
        end_idx = (i + 0.2) * step if i < 4 else step
        end_idx = int(end_idx)  # round down
        subsets_fixed[i] = set(ranked_neurons[start_idx:end_idx])

    # Also define a "baseline" subset that is empty,
    # meaning "no neurons perturbed".
    subsets_fixed["none"] = set()

    # Dictionary that holds the final results
    perturbation_results = {}

    # Loop over all subsets: "none" plus each 4% chunk
    for chunk_key in subsets_fixed:
        perturbed_indices = subsets_fixed[chunk_key]
        print(f"\nUsing subset {chunk_key} with {len(perturbed_indices)} neurons")

        # Then loop over each multiplier
        for mult in multipliers:
            # (A) Build the original and perturbed responses per neuron and frequency
            original_activity_dict_neurons = []
            perturbed_activity_dict_neurons = []

            for neuron_idx in range(num_neurons):
                neuron_activities = dff_run_list[neuron_idx]

                orig_dict = {}
                pert_dict = {}
                for fval in local_frequency_map.values():
                    orig_dict[fval] = []
                    pert_dict[fval] = []

                for trial_idx, (cond, acts) in enumerate(zip(conditions_played, neuron_activities)):
                    freq = local_frequency_map[cond]
                    stim_value = np.mean(acts[stimu_start:stimu_end])

                    # Original is unmodified
                    orig_dict[freq].append(stim_value)

                    # Perturbed only for the selected neurons on the trials to perturb
                    if (neuron_idx in perturbed_indices) and (trial_idx in con_to_perturb):
                        pert_dict[freq].append(stim_value + (addition_value * mult))
                    else:
                        pert_dict[freq].append(stim_value)

                original_activity_dict_neurons.append(orig_dict)
                perturbed_activity_dict_neurons.append(pert_dict)

            # (B) Convert to shape (nTrials, nNeurons)
            freq_keys = sorted(set(local_frequency_map.values()))
            original_population_vectors = {f: [] for f in freq_keys}
            perturbed_population_vectors = {f: [] for f in freq_keys}

            for f in freq_keys:
                tmp_orig_list = []
                tmp_pert_list = []
                for neuron_idx in range(num_neurons):
                    tmp_orig_list.append(original_activity_dict_neurons[neuron_idx][f])
                    tmp_pert_list.append(perturbed_activity_dict_neurons[neuron_idx][f])
                # shape: (nNeurons, nTrials) => transpose => (nTrials, nNeurons)
                original_population_vectors[f] = np.array(tmp_orig_list).T
                perturbed_population_vectors[f] = np.array(tmp_pert_list).T

            # (C) Run classification
            avg_accuracy, sem_all = train_SVM_no_trial_overlap(
                original_population_vectors,
                perturbed_population_vectors,
                cr_rows_15hz,
                fa_rows_15hz,
                n_iterations=500
            )

            # (D) Store final
            perturbation_results[(chunk_key, mult)] = {
                "avg_accuracy": np.array(avg_accuracy),
                "sem_all": np.array(sem_all),
                "subset": perturbed_indices
            }


    # Save results
    os.makedirs(save_loc, exist_ok=True)
    with open(os.path.join(save_loc, "combined_perturb_multiplier_results_4pct_chunks.pkl"), "wb") as f:
        pickle.dump(perturbation_results, f)

    print("\nDone. The same subsets are used for each multiplier at a given percentage!\n")

    if RUN_PERCENT_STEPS:
        # ------------------------------------------------------------------
        # (3) Cumulative top-percent subsets (10%, 20%, ..., 100%), to test
        #     for a plateau as more inhibited neurons are included
        # ------------------------------------------------------------------
        percent_steps = [round(x, 1) for x in np.arange(0.1, 1.0 + 1e-9, 0.1)]
        percent_subsets = {"none": set()}
        for p in percent_steps:
            k = int(np.ceil(p * num_neurons))
            percent_subsets[p] = set(ranked_neurons[:k])

        perturbation_results_percent = {}

        for p_key in percent_subsets:
            perturbed_indices = percent_subsets[p_key]
            print(f"\nUsing cumulative top {p_key if p_key!='none' else 0} with {len(perturbed_indices)} neurons")

            for mult in multipliers:
                original_activity_dict_neurons = []
                perturbed_activity_dict_neurons = []

                for neuron_idx in range(num_neurons):
                    neuron_activities = dff_run_list[neuron_idx]

                    orig_dict = {}
                    pert_dict = {}
                    for fval in local_frequency_map.values():
                        orig_dict[fval] = []
                        pert_dict[fval] = []

                    for trial_idx, (cond, acts) in enumerate(zip(conditions_played, neuron_activities)):
                        freq = local_frequency_map[cond]
                        stim_value = np.mean(acts[stimu_start:stimu_end])

                        # Original
                        orig_dict[freq].append(stim_value)

                        # Perturbed only for selected neurons and trials to perturb
                        if (neuron_idx in perturbed_indices) and (trial_idx in con_to_perturb):
                            pert_dict[freq].append(stim_value + (addition_value * mult))
                        else:
                            pert_dict[freq].append(stim_value)

                    original_activity_dict_neurons.append(orig_dict)
                    perturbed_activity_dict_neurons.append(pert_dict)

                # Convert to population matrices
                freq_keys = sorted(set(local_frequency_map.values()))
                original_population_vectors = {f: [] for f in freq_keys}
                perturbed_population_vectors = {f: [] for f in freq_keys}

                for f in freq_keys:
                    tmp_orig_list = []
                    tmp_pert_list = []
                    for neuron_idx in range(num_neurons):
                        tmp_orig_list.append(original_activity_dict_neurons[neuron_idx][f])
                        tmp_pert_list.append(perturbed_activity_dict_neurons[neuron_idx][f])
                    original_population_vectors[f] = np.array(tmp_orig_list).T
                    perturbed_population_vectors[f] = np.array(tmp_pert_list).T

                # PCA plots
                if mult > 5:
                    # PCA plots for the large multipliers (commented out for speed)
                    # for freq_val in freq_keys:
                    #     plot_pca_for_freq(freq_val,
                    #                       original_population_vectors,
                    #                       perturbed_population_vectors,
                    #                       conditions_played,
                    #                       responses,
                    #                       local_frequency_map,
                    #                       save_loc,
                    #                       title_suffix=f"(mult={mult}, subset={p_key})")
                    pass
                else:
                    # For multipliers <= 5, PCA of the key frequencies only
                    for freq_val in [15, 0]:
                        if freq_val in freq_keys:
                            plot_pca_for_freq(freq_val,
                                              original_population_vectors,
                                              perturbed_population_vectors,
                                              conditions_played,
                                              responses,
                                              local_frequency_map,
                                              save_loc,
                                              title_suffix=f"(baseline, subset={p_key})")

                # Classify
                avg_accuracy, sem_all = train_SVM_no_trial_overlap(
                    original_population_vectors,
                    perturbed_population_vectors,
                    cr_rows_15hz,
                    fa_rows_15hz,
                    n_iterations=500,
                )

                perturbation_results_percent[(p_key, mult)] = {
                    "avg_accuracy": np.array(avg_accuracy),
                    "sem_all": np.array(sem_all),
                    "subset": perturbed_indices,
                }

        # Save cumulative-percent results
        with open(os.path.join(save_loc, "combined_perturb_percent_results.pkl"), "wb") as f:
            pickle.dump(perturbation_results_percent, f)

    # -------------------------------------------------------------------------
    # PLOTTING (results of the 20% chunks)
    # -------------------------------------------------------------------------
    with open(os.path.join(save_loc, "combined_perturb_multiplier_results.pkl"), "rb") as f:
        perturbation_results = pickle.load(f)

    # chunk_list = [0, 1, 2, 3, 4]  # row order: baseline + 5 chunks
    chunk_list = [0, 0.2, 0.4, 0.6000000000000001, 0.8]  # row order: the five 20% chunks (no baseline)
    multipliers = [0, 0.5, 1, 2, 5, 10, 20]

    frequency_map = {6: 0,  7: 15, 8: 10, 9: 7.5,
                     10: 5, 11: 3, 12: 1, 13: 0.5}
    freq_order = sorted(set(frequency_map.values()))  # x-axis order
    xvals = np.arange(len(freq_order))

    colors = sns.color_palette("husl", len(multipliers) + 1)  # +1 for baseline

    # ------------------------------------------------------------------ (D1)
    # One figure per CHUNK (20% group), lines = multipliers
    # ------------------------------------------------------------------
    for chunk in chunk_list:
        plt.figure(figsize=(10, 6))

        for j, mult in enumerate(multipliers):
            res = perturbation_results[(chunk, mult)]
            avg = res["avg_accuracy"].copy()
            sem = res["sem_all"].copy()

            avg[0] = 1 - avg[0]  # flip 0Hz
            plt.errorbar(xvals, avg * 100, yerr=sem * 100,
                         marker="o", linestyle="-",
                         color=colors[j], label=f"×{mult}")

        plt.xticks(xvals, [f"{f}Hz" for f in freq_order])
        plt.ylim(0, 100)
        plt.xlabel("Optogenetic stimulation frequency (Hz)")
        plt.ylabel("Accuracy (%)")
        title = ("NO perturbation" if chunk == "none"
                 else f"Chunk{chunk} (rank {chunk * 20}‑{chunk * 20 + 20}%)")
        plt.title(title)
        plt.legend(title="Multiplier")
        fn = os.path.join(save_loc, f"classifier_mults_chunk_{chunk}.png")
        plt.tight_layout();
        plt.savefig(fn, dpi=300);
        plt.close()

    # ------------------------------------------------------------------ (D2)
    # One figure per MULTIPLIER, lines = chunks
    # ------------------------------------------------------------------
    for mult in multipliers:
        plt.figure(figsize=(10, 6))
        for i, chunk in enumerate(chunk_list):
            res = perturbation_results[(chunk, mult)]
            avg = res["avg_accuracy"].copy()
            sem = res["sem_all"].copy()

            avg[0] = 1 - avg[0]
            label = ("baseline" if chunk == "none"
                     else f"chunk{chunk}")
            plt.errorbar(xvals, avg * 100, yerr=sem * 100,
                         marker="o", linestyle="-",
                         color=colors[i], label=label)

        plt.xticks(xvals, [f"{f}Hz" for f in freq_order])
        plt.ylim(0, 100)
        plt.xlabel("Optogenetic stimulation frequency (Hz)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"All 20% chunks – multiplier ×{mult}")
        plt.legend()
        fn = os.path.join(save_loc, f"classifier_chunks_mult_{mult}.png")
        plt.tight_layout();
        plt.savefig(fn, dpi=300);
        plt.close()

    # ------------------------------------------------------------------ (D3)
    # Heat map: rows = chunk_list, cols = multipliers, one panel per freq
    # ------------------------------------------------------------------
    n_rows, n_cols = len(chunk_list), len(multipliers)
    fig, axs = plt.subplots(1, len(freq_order),
                            figsize=(3 * len(freq_order), 4),
                            sharey=True)

    for fi, f in enumerate(freq_order):
        Z = np.zeros((n_rows, n_cols))
        for ri, chunk in enumerate(chunk_list):
            for cj, mult in enumerate(multipliers):
                val = perturbation_results[(chunk, mult)]["avg_accuracy"][fi]
                if f == 0:  # flip 0Hz
                    val = 1 - val
                Z[ri, cj] = val * 100
        im = axs[fi].imshow(Z, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        axs[fi].set_xticks(np.arange(n_cols))
        axs[fi].set_xticklabels(multipliers, rotation=45, ha="right")
        if fi == 0:
            axs[fi].set_yticks(np.arange(n_rows))
            axs[fi].set_yticklabels(["baseline"] + [f"c{i}" for i in range(4)])
            axs[fi].set_ylabel("subset")
        else:
            axs[fi].set_yticks([])
        axs[fi].set_title(f"{f}Hz")

    cbar = fig.colorbar(im, ax=axs, label="Accuracy (%)",
                        fraction=0.02, pad=0.04)
    plt.suptitle("Accuracy heat‑maps (rows=subset, cols=multiplier)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, "heatmaps_by_frequency.png"), dpi=300)
    plt.close()

    if RUN_PERCENT_STEPS:
        # ------------------------
        # Plot plateau for all frequencies (including 15 Hz)
        # ------------------------
        with open(os.path.join(save_loc, "combined_perturb_percent_results.pkl"), "rb") as f:
            perturbation_results_percent = pickle.load(f)

        freq_order = sorted(set(local_frequency_map.values()))
        perc_plot = [0.0] + percent_steps
        x_perc = np.array(perc_plot) * 100
        colors = sns.color_palette("husl", len(multipliers) + 1)

        for fi, f in enumerate(freq_order):
            plt.figure(figsize=(10, 6))
            for j, mult in enumerate(multipliers):
                y = []
                e = []
                for p in perc_plot:
                    key = "none" if p == 0.0 else round(p, 1)
                    res = perturbation_results_percent[(key, mult)]
                    val = res["avg_accuracy"][fi]
                    sem = res["sem_all"][fi]
                    # Flip 0 Hz panel for consistency with other plots
                    if f == 0:
                        val = 1 - val
                    y.append(val * 100)
                    e.append(sem * 100)
                plt.errorbar(x_perc, y, yerr=e, marker="o", linestyle="-",
                             color=colors[j], label=f"×{mult}")

            plt.xlabel("Top inhibited neurons perturbed (%)")
            plt.ylabel(f"Accuracy at {f} Hz (%)")
            plt.title("Plateau check: accuracy vs. top‑percent inhibited neurons")
            plt.ylim(0, 100)
            plt.legend(title="Multiplier")
            safe_f = ("0" if f == 0 else str(f)).replace(" ", "")
            fn = os.path.join(save_loc, f"plateau_accuracy_{safe_f}Hz_vs_percent.pdf")
            plt.tight_layout(); plt.savefig(fn, dpi=300); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    # (session data folder, output folder); pass run indices on the command line to run only some of them
    RUNS = [
        (DATA_ROOT + r'\mouse10\sess09\ML\data',
         DATA_ROOT + r'\mouse10\sess09\ML\data\processed_data\SVM_classifier_specific_inhibition_15hz_with_inhibition_new_pdf_v2_rerun'),
        (DATA_ROOT + r'\mouse14\sess12\ML\data',
         DATA_ROOT + r'\mouse14\sess12\ML\data\processed_data\SVM_classifier_specific_inhibition_15hz_with_inhibition_new_pdf_v2_rerun'),
    ]
    selected = [RUNS[int(i)] for i in sys.argv[1:]] or RUNS
    for ROOT_PATH, save_loc in selected:
        os.makedirs(save_loc, exist_ok=True)
        main(ROOT_PATH, save_loc)
