"""
Psychophysics: running (sliding-window) classifier of the population activity (one session).

The dF/F of each neuron is z-scored and split into trials (each trial gets the last 4 s of the
previous trial as its pre-trial period, without overlap). A linear SVM (and, separately, a
Gaussian Naive Bayes classifier) is trained on the mean activity 2-5 s after trial start and
tested in a sliding window (round(0.3 * rec_hz) frames, step 1 frame) along the whole trial.
Each bootstrap iteration (n_iter = 500) uses the same number of after-hit and not-after-hit
trials in both classes, with a 70/30 train/test split per class.

Comparisons: HIT vs CR, CR vs FA and HIT vs FA, per frequency, pooled over frequencies
("all") and on all trials ("all_trials"). Saved in <save_loc>:
  classifier_results_by_freq_{HIT_CR, CR_FA, HIT_FA}.pkl            (SVM)
  classifier_results_by_freq_{HIT_CR, CR_FA, HIT_FA}_NB.pkl         (Naive Bayes)
each {freq: {"comb": (mean, sem), "sep": (mean_class1, mean_class2, sem_class1, sem_class2)}},
plus one plot per entry. The across-mice average is made by
ave_results_running_classifier_across_mice.py.
"""
import copy
import csv
import math
import os
import pickle
from operator import itemgetter

import matplotlib
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from sklearn.metrics import roc_auc_score
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd
from os.path import join
import seaborn as sns
import numpy as np
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)


start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000
pre_trial_length = 4 * rec_hz
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz
# ------------------------------------------------------------------
# 1. frequency map (condition code -> Hz) and its reverse (Hz -> codes)
#    (0 Hz, condition code 6, is not included)
# ------------------------------------------------------------------
frequency_map = {
                 7: 15,
                 8: 10,
                 9: 7.5,
                 10: 5,
                 11: 3,
                 12: 1,
                 13: 0.5}

codes_by_freq = {}
for code, freq in frequency_map.items():
    codes_by_freq.setdefault(freq, []).append(code)


def divide_to_trials(f_signal, trials_ind, lengths, len_pre_trial):
    """
    Build trials without overlap: for each trial j>0 we take the last
    len_pre_trial samples from trial j-1 and move them to the front of trial j.
    Trial j-1 is shortened accordingly. Trial 0 gets its pre-trial from the run
    pre-buffer. If a previous trial is shorter than len_pre_trial, we left-pad
    with NaNs to keep timing aligned.
    """
    f_run_list_final = []

    for i in range(len(f_signal)):  # per neuron
        neuron_trace = f_signal.iloc[i].to_numpy()
        all_trials_this_neuron = []

        start_ind = 0
        for run_idx, run_len in enumerate(lengths):
            end_ind = start_ind + run_len
            run_sig = neuron_trace[start_ind:end_ind]

            # split this run: parts[0] is pre-buffer, parts[1:] are trials
            parts = np.split(run_sig, trials_ind[run_idx])
            pre_buf = parts[0]
            raw_trials = [np.asarray(x) for x in parts[1:]]

            if len(raw_trials) == 0:
                start_ind = end_ind
                continue

            corrected = []

            # trial 0: pre from the run pre-buffer
            if len(pre_buf) >= len_pre_trial:
                pre0 = pre_buf[-len_pre_trial:]
            else:
                # pad on the left with NaNs to keep length consistent
                pre0 = np.concatenate([np.full(len_pre_trial - len(pre_buf), np.nan), pre_buf])
            corrected.append(np.concatenate([pre0, raw_trials[0]]))

            # trials 1..N-1: cut from previous and move to current
            for j in range(1, len(raw_trials)):
                prev_raw = raw_trials[j - 1]
                move_len = min(len_pre_trial, len(prev_raw))
                move = prev_raw[-move_len:]
                # pad to full pre length if the previous raw trial is short
                if move_len < len_pre_trial:
                    move = np.concatenate([np.full(len_pre_trial - move_len, np.nan), move])

                # shorten previous corrected trial by move_len taken from its tail
                corrected[j - 1] = corrected[j - 1][:-move_len]

                # current trial gets the moved tail as its pre-trial
                corrected.append(np.concatenate([move, raw_trials[j]]))

            # collect
            all_trials_this_neuron.extend(corrected)
            start_ind = end_ind

        f_run_list_final.append(all_trials_this_neuron)

    # verify no duplicated feature rows at early windows
    def _dupe_check(trials, start, end):
        # Compute a scalar feature per trial (mean over the early window).
        # Use only finite values for duplicate detection to avoid NaN issues.
        feats = [float(np.nanmean(np.asarray(t)[..., start:end])) for t in trials]
        finite_feats = [f for f in feats if not np.isnan(f)]
        return len(finite_feats) != len(set(finite_feats))

    # for one neuron to sanity check
    one_neuron_trials = f_run_list_final[0]
    has_dupes = _dupe_check(one_neuron_trials, 0, pre_trial_length)
    print("Early-window duplicates:", has_dupes)

    return f_run_list_final, None

def _as_nd(x):
    """Return 2-D ndarray whether *x* is a DataFrame or ndarray."""
    return x.values if isinstance(x, pd.DataFrame) else x
def train_svm_classifier_sliding_window_combined_stratified(
        cr_trials_with_idx, fa_trials_with_idx,
        responses,
        pre_trial_length, rec_hz, n_iterations=100):
    """
    Sliding-window SVM, accuracy on the combined test set (class 0 = first list,
    class 1 = second list). Returns the mean and SEM over iterations, per time bin.
    """

    # --- pre-split by history flag -----------------------------
    cr_after, cr_not = split_history(cr_trials_with_idx, responses)
    fa_after, fa_not = split_history(fa_trials_with_idx, responses)

    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end   = pre_trial_length + 5 * rec_hz
    window_size = round(0.3 * rec_hz)
    step_size   = 1
    last_col    = 17 * rec_hz         # assume ≥17s per trial

    accuracies_all = []

    for _ in range(n_iterations):
        # -----------------------------------------------------------
        # pick the minority & majority on-the-fly
        # -----------------------------------------------------------
        n_cr = len(cr_after) + len(cr_not)
        n_fa = len(fa_after) + len(fa_not)

        if n_cr <= n_fa:  # CR is minority (or equal)
            min_after, min_not = cr_after, cr_not
            maj_after, maj_not = fa_after, fa_not
            minority_is_cr = True
        else:  # FA is minority
            min_after, min_not = fa_after, fa_not
            maj_after, maj_not = cr_after, cr_not
            minority_is_cr = False

        # draw a history-balanced sample
        min_block, maj_block = stratified_bootstrap(
            min_after, min_not,
            maj_after, maj_not)

        # map back to (cr_block, fa_block)
        if minority_is_cr:
            cr_block, fa_block = min_block, maj_block
        else:
            cr_block, fa_block = maj_block, min_block

        # SKIP if either block is empty or too small for 70/30 split
        if len(cr_block) < 2 or len(fa_block) < 2:
            continue

        # 70/30 train-test split (separate per class)
        X_cr_train, X_cr_test = train_test_split(cr_block, test_size=0.3)
        X_fa_train, X_fa_test = train_test_split(fa_block, test_size=0.3)

        # ------------------------------------------------------------------
        # 1. build training feature matrix
        # ------------------------------------------------------------------
        X_train = np.vstack([
            _as_nd(trial) for trial in X_cr_train + X_fa_train])
        y_train = np.concatenate([
            np.zeros(len(X_cr_train)),
            np.ones (len(X_fa_train))])

        X_train_feat = np.asarray([
            np.nanmean(_as_nd(trial)[:, stimu_start:stimu_end], axis=1)
            for trial in X_cr_train + X_fa_train])

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_feat, y_train)

        # ------------------------------------------------------------------
        # 2. prepare combined test set
        # ------------------------------------------------------------------
        X_test_comb  = X_cr_test + X_fa_test
        y_test_comb  = np.concatenate([
            np.zeros(len(X_cr_test)),
            np.ones (len(X_fa_test))])

        # ------------------------------------------------------------------
        # 3. sliding-window evaluation
        # ------------------------------------------------------------------
        acc_iter = []

        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size

            X_test_feat = np.asarray([
                np.nanmean(_as_nd(trial)[:, start:end], axis=1)
                for trial in X_test_comb])

            y_pred = clf.predict(X_test_feat)
            acc_iter.append(
                accuracy_score(y_test_comb, y_pred)
                if len(y_pred) else np.nan)

        # Disjointness checks by object identity (no trial appears in both train and test)
        if any(id(x) in {id(t) for t in X_cr_test} for x in X_cr_train):
            raise ValueError("CR trial leakage between train and test (combined)")
        if any(id(x) in {id(t) for t in X_fa_test} for x in X_fa_train):
            raise ValueError("FA trial leakage between train and test (combined)")

        accuracies_all.append(np.asarray(acc_iter))

    # ----------------------------------------------------------------------
    accuracies_all = np.asarray(accuracies_all)
    avg_accuracy   = np.nanmean(accuracies_all, axis=0)
    sem_accuracy   = np.nanstd (accuracies_all, axis=0) / np.sqrt(n_iterations)
    return avg_accuracy, sem_accuracy


def train_svm_classifier_sliding_window_separate_stratified(
        cr_trials_with_idx,         # list of (trial_df, abs_idx)  - CR
        fa_trials_with_idx,         # list of (trial_df, abs_idx)  - FA
        responses,                  # full responses array
        pre_trial_length,
        rec_hz,
        n_iterations=100):
    """
    Same as the combined version, but returns the accuracy on the test trials of each
    class separately: (avg_cr, avg_fa, sem_cr, sem_fa). Each bootstrap sample has the
    same number of after-hit and not-after-hit trials in both classes.
    """

    # --- 1. pre-split by history flag ---------------------------------------
    cr_after, cr_not = split_history(cr_trials_with_idx, responses)
    fa_after, fa_not = split_history(fa_trials_with_idx, responses)

    accuracies_cr_all  = []      # sliding-window accuracies for CR test set
    accuracies_fa_all  = []      # sliding-window accuracies for FA test set

    # fixed parameters
    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end   = pre_trial_length + 5 * rec_hz
    window_size = round(0.3 * rec_hz)
    step_size   = 1
    last_col    = 17 * rec_hz              # assume ≥17s per trial

    for _ in range(n_iterations):

        # -----------------------------------------------------------
        # pick the minority & majority on-the-fly
        # -----------------------------------------------------------
        n_cr = len(cr_after) + len(cr_not)
        n_fa = len(fa_after) + len(fa_not)

        if n_cr <= n_fa:  # CR is minority (or equal)
            min_after, min_not = cr_after, cr_not
            maj_after, maj_not = fa_after, fa_not
            minority_is_cr = True
        else:  # FA is minority
            min_after, min_not = fa_after, fa_not
            maj_after, maj_not = cr_after, cr_not
            minority_is_cr = False

        # draw a history-balanced sample
        min_block, maj_block = stratified_bootstrap(
            min_after, min_not,
            maj_after, maj_not)

        # map back to (cr_block, fa_block)
        if minority_is_cr:
            cr_block, fa_block = min_block, maj_block
        else:
            cr_block, fa_block = maj_block, min_block

        # SKIP if either block is empty
        if len(cr_block) == 0 or len(fa_block) == 0:
            continue

        # 70/30 train-test split (separate per class)
        X_cr_train, X_cr_test = train_test_split(cr_block, test_size=0.3)
        X_fa_train, X_fa_test = train_test_split(fa_block, test_size=0.3)

        # --- 3. build training matrix --------------------------------------
        X_train = np.vstack((X_cr_train, X_fa_train))
        y_train = np.concatenate((np.zeros(len(X_cr_train)),
                                  np.ones(len(X_fa_train))))

        # 3-A  build training feature matrix
        X_train_feat = np.asarray([
            np.nanmean(_as_nd(trial)[:, stimu_start:stimu_end], axis=1)
            for trial in X_train
        ])

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_feat, y_train)

        # --- 4. sliding-window evaluation ----------------------------------
        acc_cr, acc_fa = [], []

        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size

            # features
            X_test_cr = np.asarray([
                np.nanmean(_as_ndarray(trial)[:, start:end], axis=1)
                for trial in X_cr_test
            ])
            X_test_fa = np.asarray([
                np.nanmean(_as_ndarray(trial)[:, start:end], axis=1)
                for trial in X_fa_test
            ])

            # predict
            y_pred_cr = clf.predict(X_test_cr) if len(X_test_cr) else []
            y_pred_fa = clf.predict(X_test_fa) if len(X_test_fa) else []

            # accuracy
            acc_cr.append(
                accuracy_score(np.zeros(len(y_pred_cr)), y_pred_cr)
                if len(y_pred_cr) else np.nan)
            acc_fa.append(
                accuracy_score(np.ones(len(y_pred_fa)),  y_pred_fa)
                if len(y_pred_fa) else np.nan)

        # Disjointness checks by object identity
        if any(id(x) in {id(t) for t in X_cr_test} for x in X_cr_train):
            raise ValueError("CR trial leakage between train and test (separate)")
        if any(id(x) in {id(t) for t in X_fa_test} for x in X_fa_train):
            raise ValueError("FA trial leakage between train and test (separate)")

        accuracies_cr_all.append(np.array(acc_cr))
        accuracies_fa_all.append(np.array(acc_fa))

    # --- 5. aggregate over iterations --------------------------------------
    accuracies_cr_all = np.asarray(accuracies_cr_all)
    accuracies_fa_all = np.asarray(accuracies_fa_all)

    avg_cr  = np.nanmean(accuracies_cr_all, axis=0)
    avg_fa  = np.nanmean(accuracies_fa_all, axis=0)
    sem_cr  = np.nanstd (accuracies_cr_all, axis=0) / np.sqrt(n_iterations)
    sem_fa  = np.nanstd (accuracies_fa_all, axis=0) / np.sqrt(n_iterations)

    return avg_cr, avg_fa, sem_cr, sem_fa



def divide_to_conditions(dff_sub, correct_reject, false_alarm, hit, miss):
    data_correct_reject = []
    data_false_alarm = []
    data_hit = []
    data_miss = []
    for neuron in (dff_sub):
        neuron = pd.DataFrame(neuron)
        neuron_correct_reject = []
        neuron_false_alarm = []
        neuron_hit = []
        neuron_miss = []
        # cond1_dff = neuron.iloc[con1_ind, :]
        # cond2_dff = neuron.iloc[con2_ind, :]
        neuron_correct_reject.append(neuron.iloc[correct_reject, :])
        neuron_false_alarm.append(neuron.iloc[false_alarm, :])
        neuron_hit.append(neuron.iloc[hit, :])
        neuron_miss.append(neuron.iloc[miss, :])
        data_correct_reject.append(neuron_correct_reject[0])
        data_false_alarm.append(neuron_false_alarm[0])
        data_hit.append(neuron_hit[0])
        data_miss.append(neuron_miss[0])
    return data_correct_reject, data_false_alarm, data_hit, data_miss
def convert_list_structure_with_idx(data, abs_indices):
    trials = []
    for rel, abs_idx in enumerate(abs_indices):
        trial_list = []
        for neuron_idx in range(len(data)):
            trial_list.append(data[neuron_idx].iloc[rel].values)
        trials.append((pd.DataFrame(trial_list), abs_idx))
    return trials





def divide_by_history(trial_indices, responses, history_value=3):
    """
    Separates trial indices into two groups:
      - those that immediately follow a trial with response == history_value
      - those that do not.

    Parameters:
      trial_indices (array-like): Indices of trials for a given condition.
      responses (array-like): Array of responses for all trials.
      history_value (int): The response value to look for in the previous trial (default is 3 for "hit").

    Returns:
      after_history (np.array): Trial indices that come after a trial with response==history_value.
      not_after_history (np.array): Trial indices that do not come after a trial with response==history_value.
    """
    after_history = []
    not_after_history = []
    for idx in trial_indices:
        # For the first trial, or if the previous trial is not a hit, add to not_after_history.
        if idx == 0 or responses[idx - 1] != history_value:
            not_after_history.append(idx)
        else:
            after_history.append(idx)
    return np.array(after_history), np.array(not_after_history)

def divide_to_conditions_by_history(dff_sub, correct_reject, false_alarm, hit, miss, responses):
    """
    Separates the dF/F data for each condition into two groups based on trial history:
      (1) Trials that came immediately after a hit (i.e. previous trial had response==3)
      (2) Trials that did not come immediately after a hit.

    Parameters:
      dff_sub (list): List of neurons. Each element is a 2D array or DataFrame where
                      each row corresponds to a trial and each column to a time point.
      correct_reject, false_alarm, hit, miss (array-like): Trial indices (e.g., from np.where)
                      corresponding to each condition.
      responses (array-like): Array of responses for all trials.

    Returns:
      A tuple with eight elements:
         (data_cr_after,  data_cr_not_after,
          data_fa_after,  data_fa_not_after,
          data_hit_after, data_hit_not_after,
          data_miss_after, data_miss_not_after)
      Each element is a list (over neurons) containing a DataFrame with the trials for that group.
    """
    # Split each condition's trial indices by history (here, history_value=3 indicates a hit)
    cr_after, cr_not_after = divide_by_history(correct_reject, responses, history_value=3)
    fa_after, fa_not_after = divide_by_history(false_alarm, responses, history_value=3)
    hit_after, hit_not_after = divide_by_history(hit, responses, history_value=3)
    miss_after, miss_not_after = divide_by_history(miss, responses, history_value=3)

    # For each neuron, extract the corresponding rows from its data.
    data_cr_after = []
    data_cr_not_after = []
    data_fa_after = []
    data_fa_not_after = []
    data_hit_after = []
    data_hit_not_after = []
    data_miss_after = []
    data_miss_not_after = []

    for neuron in dff_sub:
        # Convert to DataFrame if not already
        if not isinstance(neuron, pd.DataFrame):
            neuron_df = pd.DataFrame(neuron)
        else:
            neuron_df = neuron.copy()

        data_cr_after.append(neuron_df.iloc[cr_after, :])
        data_cr_not_after.append(neuron_df.iloc[cr_not_after, :])
        data_fa_after.append(neuron_df.iloc[fa_after, :])
        data_fa_not_after.append(neuron_df.iloc[fa_not_after, :])
        data_hit_after.append(neuron_df.iloc[hit_after, :])
        data_hit_not_after.append(neuron_df.iloc[hit_not_after, :])
        data_miss_after.append(neuron_df.iloc[miss_after, :])
        data_miss_not_after.append(neuron_df.iloc[miss_not_after, :])

    return (data_cr_after, data_cr_not_after,
            data_fa_after, data_fa_not_after,
            data_hit_after, data_hit_not_after,
            data_miss_after, data_miss_not_after)

# ------------------------------------------------------------------
# 3. per-frequency splitter using absolute indices
# ------------------------------------------------------------------

def split_by_frequency_keep_idx(trials_with_idx, cond_vec, wanted_codes):
    """
    Return a list of (trial_df, abs_idx) *with* the abs_idx preserved.
    """
    out = []
    for trial_df, abs_idx in trials_with_idx:
        if cond_vec[abs_idx] in wanted_codes:
            out.append((trial_df, abs_idx))
    return out
def _as_ndarray(x):
    """Return 2-D ndarray regardless of whether *x* is a DataFrame or ndarray."""
    return x.values if isinstance(x, pd.DataFrame) else x
# ------------------------------------------------------------------
# 4. run classifier for all freqs, now index-safe
# ------------------------------------------------------------------
def run_classifier_for_all_freqs(trials_CR_with_idx,
                                 trials_FA_with_idx,
                                 cond_vec,
                                 responses,
                                 pre_trial_len,
                                 hz,
                                 n_iter=100,
                                 min_trials_single=4,
                                 min_trials_pooled=2):
    """
    Runs the SVM for each frequency (needs >= min_trials_single trials of each class),
    for the trials pooled over the frequencies that have >= min_trials_pooled of each
    class ("all"), and for all trials ("all_trials"). Trials whose condition is not in
    frequency_map (e.g. the 0 Hz HIT trials) only enter "all_trials".
    """

    # ------------------------------------------------------------------
    # wrappers that call the two history-aware classifiers
    # ------------------------------------------------------------------
    def run_sep(cr_tuples, fa_tuples):
        return train_svm_classifier_sliding_window_separate_stratified(
            cr_tuples, fa_tuples, responses,
            pre_trial_len, hz, n_iterations=n_iter)

    def run_comb(cr_tuples, fa_tuples):
        return train_svm_classifier_sliding_window_combined_stratified(
            cr_tuples, fa_tuples, responses,
            pre_trial_len, hz, n_iterations=n_iter)

    def _pack(cr_tuples, fa_tuples): # pack results
        avg_CR, avg_FA, sem_CR, sem_FA = run_sep(cr_tuples, fa_tuples)
        avg_all, sem_all               = run_comb(cr_tuples, fa_tuples)
        return {"comb": (avg_all, sem_all),
                "sep" : (avg_CR,  avg_FA, sem_CR, sem_FA)}

    # ------------------------------------------------------------------
    results = {}
    pooled_CR_tuples, pooled_FA_tuples = [], []

    for freq, codes in codes_by_freq.items():

        cr_tuples = split_by_frequency_keep_idx(trials_CR_with_idx, cond_vec, codes)
        fa_tuples = split_by_frequency_keep_idx(trials_FA_with_idx, cond_vec, codes)

        if len(cr_tuples) < min_trials_single or len(fa_tuples) < min_trials_single:
            print(f"Skipping {freq}Hz (not enough CR or FA trials)")
            continue

        results[freq] = _pack(cr_tuples, fa_tuples)
    # ------------------------------------------------------------------
    for freq, codes in codes_by_freq.items():

        cr_tuples_pool = split_by_frequency_keep_idx(trials_CR_with_idx, cond_vec, codes)
        fa_tuples_pool = split_by_frequency_keep_idx(trials_FA_with_idx, cond_vec, codes)

        if len(cr_tuples_pool) < min_trials_pooled or len(fa_tuples_pool) < min_trials_pooled:
            print(f"Skipping {freq}Hz (not enough CR or FA trials)")
            continue
        pooled_CR_tuples.extend(cr_tuples_pool)
        pooled_FA_tuples.extend(fa_tuples_pool)

    # pooled entry ("all")
    if len(pooled_CR_tuples) >= min_trials_single and len(pooled_FA_tuples) >= min_trials_single:
        results["all"] = _pack(pooled_CR_tuples, pooled_FA_tuples)
    else:
        print("No pooled entry created (not enough eligible trials)")

    # --- Always run classifier on all available trials ---
    if len(trials_CR_with_idx) >= min_trials_single and len(trials_FA_with_idx) >= min_trials_single:
        results["all_trials"] = _pack(trials_CR_with_idx, trials_FA_with_idx)
    else:
        print("No all_trials entry created (no eligible trials at all)")

    return results


def train_nb_classifier_sliding_window_combined_stratified(
        cr_trials_with_idx, fa_trials_with_idx,
        responses,
        pre_trial_length, rec_hz, n_iterations=100):
    """Gaussian Naive Bayes version of train_svm_classifier_sliding_window_combined_stratified."""

    cr_after, cr_not = split_history(cr_trials_with_idx, responses)
    fa_after, fa_not = split_history(fa_trials_with_idx, responses)

    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end   = pre_trial_length + 5 * rec_hz
    window_size = round(0.3 * rec_hz)
    step_size   = 1
    last_col    = 17 * rec_hz

    accuracies_all = []

    for _ in range(n_iterations):
        n_cr = len(cr_after) + len(cr_not)
        n_fa = len(fa_after) + len(fa_not)

        if n_cr <= n_fa:
            min_after, min_not = cr_after, cr_not
            maj_after, maj_not = fa_after, fa_not
            minority_is_cr = True
        else:
            min_after, min_not = fa_after, fa_not
            maj_after, maj_not = cr_after, cr_not
            minority_is_cr = False

        min_block, maj_block = stratified_bootstrap(
            min_after, min_not,
            maj_after, maj_not)

        if minority_is_cr:
            cr_block, fa_block = min_block, maj_block
        else:
            cr_block, fa_block = maj_block, min_block

        # SKIP if either block is empty or too small for 70/30 split
        if len(cr_block) < 2 or len(fa_block) < 2:
            continue

        X_cr_train, X_cr_test = train_test_split(cr_block, test_size=0.3)
        X_fa_train, X_fa_test = train_test_split(fa_block, test_size=0.3)

        X_train_feat = np.asarray([
            np.nanmean(_as_nd(trial)[:, stimu_start:stimu_end], axis=1)
            for trial in X_cr_train + X_fa_train
        ])
        y_train = np.concatenate([
            np.zeros(len(X_cr_train)),
            np.ones (len(X_fa_train))
        ])

        clf = GaussianNB()
        clf.fit(X_train_feat, y_train)

        X_test_comb  = X_cr_test + X_fa_test
        y_test_comb  = np.concatenate([
            np.zeros(len(X_cr_test)),
            np.ones (len(X_fa_test))
        ])

        acc_iter = []
        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size
            X_test_feat = np.asarray([
                np.nanmean(_as_nd(trial)[:, start:end], axis=1)
                for trial in X_test_comb
            ])
            y_pred = clf.predict(X_test_feat)
            acc_iter.append(
                accuracy_score(y_test_comb, y_pred)
                if len(y_pred) else np.nan)

        # Disjointness checks by object identity
        if any(id(x) in {id(t) for t in X_cr_test} for x in X_cr_train):
            raise ValueError("CR trial leakage between train and test (NB combined)")
        if any(id(x) in {id(t) for t in X_fa_test} for x in X_fa_train):
            raise ValueError("FA trial leakage between train and test (NB combined)")

        accuracies_all.append(np.asarray(acc_iter))

    accuracies_all = np.asarray(accuracies_all)
    avg_accuracy   = np.nanmean(accuracies_all, axis=0)
    sem_accuracy   = np.nanstd (accuracies_all, axis=0) / np.sqrt(n_iterations)
    return avg_accuracy, sem_accuracy


def train_nb_classifier_sliding_window_separate_stratified(
        cr_trials_with_idx,
        fa_trials_with_idx,
        responses,
        pre_trial_length,
        rec_hz,
        n_iterations=100):
    """Gaussian Naive Bayes version of train_svm_classifier_sliding_window_separate_stratified."""

    cr_after, cr_not = split_history(cr_trials_with_idx, responses)
    fa_after, fa_not = split_history(fa_trials_with_idx, responses)

    accuracies_cr_all  = []
    accuracies_fa_all  = []

    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end   = pre_trial_length + 5 * rec_hz
    window_size = round(0.3 * rec_hz)
    step_size   = 1
    last_col    = 17 * rec_hz

    for _ in range(n_iterations):
        n_cr = len(cr_after) + len(cr_not)
        n_fa = len(fa_after) + len(fa_not)

        if n_cr <= n_fa:
            min_after, min_not = cr_after, cr_not
            maj_after, maj_not = fa_after, fa_not
            minority_is_cr = True
        else:
            min_after, min_not = fa_after, fa_not
            maj_after, maj_not = cr_after, cr_not
            minority_is_cr = False

        min_block, maj_block = stratified_bootstrap(
            min_after, min_not,
            maj_after, maj_not)

        # Guard against empty or too-small blocks that would break train_test_split
        # Need at least 2 samples per block to allow a 70/30 split with non-empty train set
        if len(min_block) < 2 or len(maj_block) < 2:
            continue

        if minority_is_cr:
            X_cr_train, X_cr_test = train_test_split(min_block, test_size=0.3)
            X_fa_train, X_fa_test = train_test_split(maj_block, test_size=0.3)
        else:
            X_cr_train, X_cr_test = train_test_split(maj_block, test_size=0.3)
            X_fa_train, X_fa_test = train_test_split(min_block, test_size=0.3)

        X_train = np.vstack((X_cr_train, X_fa_train))
        y_train = np.concatenate((np.zeros(len(X_cr_train)),
                                  np.ones(len(X_fa_train))))

        X_train_feat = np.asarray([
            np.nanmean(_as_nd(trial)[:, stimu_start:stimu_end], axis=1)
            for trial in X_train
        ])

        clf = GaussianNB()
        clf.fit(X_train_feat, y_train)

        acc_cr, acc_fa = [], []
        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size

            X_test_cr = np.asarray([
                np.nanmean(_as_ndarray(trial)[:, start:end], axis=1)
                for trial in X_cr_test
            ])
            X_test_fa = np.asarray([
                np.nanmean(_as_ndarray(trial)[:, start:end], axis=1)
                for trial in X_fa_test
            ])

            y_pred_cr = clf.predict(X_test_cr) if len(X_test_cr) else []
            y_pred_fa = clf.predict(X_test_fa) if len(X_test_fa) else []

            acc_cr.append(
                accuracy_score(np.zeros(len(y_pred_cr)), y_pred_cr)
                if len(y_pred_cr) else np.nan)
            acc_fa.append(
                accuracy_score(np.ones(len(y_pred_fa)),  y_pred_fa)
                if len(y_pred_fa) else np.nan)

        # Disjointness checks by object identity
        if any(id(x) in {id(t) for t in X_cr_test} for x in X_cr_train):
            raise ValueError("CR trial leakage between train and test (NB separate)")
        if any(id(x) in {id(t) for t in X_fa_test} for x in X_fa_train):
            raise ValueError("FA trial leakage between train and test (NB separate)")

        accuracies_cr_all.append(np.array(acc_cr))
        accuracies_fa_all.append(np.array(acc_fa))

    accuracies_cr_all = np.asarray(accuracies_cr_all)
    accuracies_fa_all = np.asarray(accuracies_fa_all)

    avg_cr  = np.nanmean(accuracies_cr_all, axis=0)
    avg_fa  = np.nanmean(accuracies_fa_all, axis=0)
    sem_cr  = np.nanstd (accuracies_cr_all, axis=0) / np.sqrt(n_iterations)
    sem_fa  = np.nanstd (accuracies_fa_all, axis=0) / np.sqrt(n_iterations)

    return avg_cr, avg_fa, sem_cr, sem_fa


def run_classifier_for_all_freqs_NB(trials_CR_with_idx,
                                    trials_FA_with_idx,
                                    cond_vec,
                                    responses,
                                    pre_trial_len,
                                    hz,
                                    n_iter=100,
                                    min_trials_single=4,
                                    min_trials_pooled=2):
    """Gaussian Naive Bayes version of run_classifier_for_all_freqs."""

    def run_sep(cr_tuples, fa_tuples):
        return train_nb_classifier_sliding_window_separate_stratified(
            cr_tuples, fa_tuples, responses,
            pre_trial_len, hz, n_iterations=n_iter)

    def run_comb(cr_tuples, fa_tuples):
        return train_nb_classifier_sliding_window_combined_stratified(
            cr_tuples, fa_tuples, responses,
            pre_trial_len, hz, n_iterations=n_iter)

    def _pack(cr_tuples, fa_tuples):
        avg_CR, avg_FA, sem_CR, sem_FA = run_sep(cr_tuples, fa_tuples)
        avg_all, sem_all               = run_comb(cr_tuples, fa_tuples)
        return {"comb": (avg_all, sem_all),
                "sep" : (avg_CR,  avg_FA, sem_CR, sem_FA)}

    results = {}
    pooled_CR_tuples, pooled_FA_tuples = [], []

    for freq, codes in codes_by_freq.items():
        cr_tuples = split_by_frequency_keep_idx(trials_CR_with_idx, cond_vec, codes)
        fa_tuples = split_by_frequency_keep_idx(trials_FA_with_idx, cond_vec, codes)

        if len(cr_tuples) < min_trials_single or len(fa_tuples) < min_trials_single:
            print(f"Skipping {freq}Hz (not enough CR or FA trials)")
            continue
        results[freq] = _pack(cr_tuples, fa_tuples)

    for freq, codes in codes_by_freq.items():
        cr_tuples = split_by_frequency_keep_idx(trials_CR_with_idx, cond_vec, codes)
        fa_tuples = split_by_frequency_keep_idx(trials_FA_with_idx, cond_vec, codes)

        if len(cr_tuples) < min_trials_pooled or len(fa_tuples) < min_trials_pooled:
            print(f"Skipping {freq}Hz (not enough CR or FA trials)")
            continue
        pooled_CR_tuples.extend(cr_tuples)
        pooled_FA_tuples.extend(fa_tuples)

    if len(pooled_CR_tuples) >= min_trials_single and len(pooled_FA_tuples) >= min_trials_single:
        results["all"] = _pack(pooled_CR_tuples, pooled_FA_tuples)
    else:
        print("No pooled entry created (not enough eligible trials)")

    if len(trials_CR_with_idx) >= min_trials_single and len(trials_FA_with_idx) >= min_trials_single:
        results["all_trials"] = _pack(trials_CR_with_idx, trials_FA_with_idx)
    else:
        print("No all_trials entry created (no eligible trials at all)")

    return results

def plot_comb_and_sep(results,
                      pre_trial_len,
                      rec_hz,
                      save_dir,
                      base_name="acc_comb_sep",
                      file_name="svm",
                      class_labels=None,
                      title_prefix=None):
    """
    Draw one figure per frequency (including 'all' and 'all_trials') with three curves:
      - combined accuracy (overall test set)
      - accuracy on the first class's test trials
      - accuracy on the second class's test trials
    (class labels are taken from file_name, e.g. 'HIT_CR', unless class_labels is given)
    """
    # shared x-axis
    xlim   = 17 * rec_hz          # trials have ≥17 s
    ticks  = np.arange(0, xlim, step=rec_hz * 2)
    labels = ticks / rec_hz

    # infer model name for title and dynamic class labels
    fn_upper = (file_name or "").upper()
    model_for_title = "Naive Bayes" if "NB" in fn_upper else "SVM"
    title_prefix = title_prefix or f"{model_for_title} accuracy"

    if class_labels and len(class_labels) == 2:
        first_label, second_label = class_labels
    else:
        tokens = [t for t in (file_name or "").split("_") if t]
        if tokens and tokens[-1].upper() == "NB":
            tokens = tokens[:-1]
        if len(tokens) >= 2:
            first_label, second_label = tokens[0], tokens[1]
        else:
            first_label, second_label = "Class 1", "Class 2"

    # iterate over frequencies (keep deterministic order, but show 'all' and 'all_trials' last)
    def sort_key(x):
        if x[0] == 'all':
            return (1e10,)
        if x[0] == 'all_trials':
            return (1e11,)
        return (float(x[0]),)
    for freq, res in sorted(results.items(), key=sort_key):
        if freq == 'all':
            tag = "all_freqs"
        elif freq == 'all_trials':
            tag = "all_trials"
        else:
            tag = f"{freq}Hz"

        # unpack: treat 'sep' as first/second class generically
        avg_comb,  sem_comb                       = res['comb']
        avg_first, avg_second, sem_first, sem_second = res['sep']

        if np.isscalar(avg_comb) or np.isscalar(sem_comb):
            print(f"Skipping {tag} because avg_comb or sem_comb is a scalar.")
            continue

        fig, ax = plt.subplots(figsize=(20, 10))

        # Combined
        ax.plot(avg_comb, lw=2.5, color="black", label="Combined")
        ax.fill_between(range(len(avg_comb)),
                        avg_comb - sem_comb, avg_comb + sem_comb,
                        alpha=0.25, color="black")

        # first class curve
        ax.plot(avg_first,  ls="--", color="tab:blue", label=f"{first_label} only")
        ax.fill_between(range(len(avg_first)),
                        avg_first - sem_first, avg_first + sem_first,
                        alpha=0.25, color="tab:blue")

        # second class curve
        ax.plot(avg_second, ls=":",  color="tab:red",  label=f"{second_label} only")
        ax.fill_between(range(len(avg_second)),
                        avg_second - sem_second, avg_second + sem_second,
                        alpha=0.25, color="tab:red")

        # task timing
        ax.axvline(pre_trial_len,                color="darkorange",
                   ls="--", lw=1, label="Distractor light")
        ax.axvline(pre_trial_len + 9 * rec_hz,   color="darkorange", ls="--", lw=1)
        ax.axvline(pre_trial_len + 1 * rec_hz,   color="yellow",
                   ls="--", lw=1, label="Opto stim")
        ax.axvline(pre_trial_len + 1 * rec_hz + 4 * rec_hz,
                   color="yellow", ls="--", lw=1)
        ax.axvline(pre_trial_len + 5 * rec_hz,   color="blue",
                   ls="--", lw=1, label="Odor")
        ax.axvline(pre_trial_len + 5 * rec_hz + 2 * rec_hz,
                   color="blue", ls="--", lw=1)

        # shading for training window
        ax.axvspan(pre_trial_len + 2 * rec_hz,
                   pre_trial_len + 5 * rec_hz,
                   color="gray", alpha=0.15)

        # cosmetics
        ax.set_xlim(0, xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(0, 1.1, step=0.1))
        ax.set_yticklabels(np.arange(0, 110, step=10))
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("% correct")
        ax.set_title(f"{title_prefix} – {tag}")
        ax.legend(ncol=2)
        fig.tight_layout()

        # save
        png = join(save_dir, f"{base_name}_{tag}_{file_name}.png")
        pdf = png.replace(".png", ".pdf")
        fig.savefig(png)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Saved {png} & {pdf}")

def split_history(trials_with_idx, responses, hit_val=3):
    """
    Split list of (trial_df, abs_idx) into two lists:
        after_hit  – previous trial had response == hit_val
        not_after  – otherwise
    """
    after, not_after = [], []
    for trial_df, abs_idx in trials_with_idx:
        flag = abs_idx > 0 and responses[abs_idx - 1] == hit_val
        (after if flag else not_after).append(trial_df)
    return after, not_after
from sklearn.utils import resample

def stratified_bootstrap(minority_after, minority_not,
                         majority_after, majority_not):
    """
    Return two lists with identical after/not counts, using the maximum possible
    number of trials for each group (after/not after) that both classes can provide.
    """
    # Find the minimum available in both classes for each group
    n_after = min(len(minority_after), len(majority_after))
    n_not   = min(len(minority_not),   len(majority_not))

    # If either group is empty, return empty blocks (the iteration is then skipped)
    if n_after == 0 or n_not == 0:
        print("No trials available for either after or not after")
        return [], [] 

    # Downsample both classes to the minimum for each group
    min_after_sel = resample(minority_after, replace=False, n_samples=n_after)
    min_not_sel   = resample(minority_not,   replace=False, n_samples=n_not)
    maj_after_sel = resample(majority_after, replace=False, n_samples=n_after)
    maj_not_sel   = resample(majority_not,   replace=False, n_samples=n_not)

    minority_block = list(min_after_sel) + list(min_not_sel)
    majority_block = list(maj_after_sel) + list(maj_not_sel)
    return minority_block, majority_block     # order: minority, majority


if '__main__' == __name__:
    # Run once per session. The sessions used:
    #   mouse01/sess02, mouse05/sess04  -> data folder ML\data2
    #   mouse09/sess06, mouse10/sess09, mouse14/sess12  -> data folder ML\data
    day = ['sess04']
    ROOT_PATH = DATA_ROOT + r'\mouse05\{}\ML\data2'.format(day[0])
    save_loc = DATA_ROOT + r'\mouse05\{}\ML\data\classifier_leading_z_no_sub_history_v2'.format(day[0])
    os.makedirs(save_loc, exist_ok=True)
    n_priming_trials = 23
    # load the behavior and imaging data
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

    # z-scoring each neuron's trace (dff_not_norm holds the z-scored traces)
    dff_norm = []
    for i in range(len(dff_4min)):
        neuron_trace = dff_4min.iloc[i]
        mean = np.nanmean(neuron_trace)
        std = np.nanstd(neuron_trace)
        dff_norm.append((neuron_trace - mean) / std)
    dff_not_norm = pd.DataFrame(dff_norm)

    # divide into trials
    dff_run_list, pre_runs = divide_to_trials(dff_not_norm, trials_ind, run_len,
                                              pre_trial_length)

    # opto onset of each trial and reward times (not used by the classifiers)
    opto_on_ind = []
    distract_light_on = []
    distract_light_off = []
    for i in range(len(opto_on)):
        # Find indices where values first exceed 1
        changes = np.diff(np.concatenate(([0], opto_on[i] > 1)))
        ind = np.where(changes == 1)[0]
        opto_on_ind.append(ind[0] if len(ind) > 0 else np.nan)
    # add the pre-trial to each trial
    opto_on_ind = [i + pre_trial_length for i in opto_on_ind]
    opto_off_ind = [i + stim_len for i in opto_on_ind]

    correct_reject = np.where(responses == 4)[0]
    false_alarm = np.where(responses == 5)[0]
    hit = np.where(responses == 3)[0]
    miss = np.where(responses == 2)[0]
    for i in range(len(valve_open)):
        changes = np.diff(np.concatenate(([0], valve_open[i] > 1)))
        valve_open[i] = np.where(changes == 1)[0]
    rewards_times = [valve_open[i] for i in hit]
    # divide the data to the different conditions
    dff_correct_reject, dff_false_alarm, dff_hit, dff_miss = divide_to_conditions(dff_run_list, correct_reject,
                                                                                  false_alarm, hit, miss)
    # Now divide the trials by history:
    (data_cr_after, data_cr_not_after,
     data_fa_after, data_fa_not_after,
     data_hit_after, data_hit_not_after,
     data_miss_after, data_miss_not_after) = divide_to_conditions_by_history(
        dff_run_list, correct_reject, false_alarm, hit, miss, responses)

    correct_rejected_samples = dff_correct_reject
    false_alarm_samples = dff_false_alarm
    hit_samples = dff_hit
    # absolute indices for each class
    idx_CR = correct_reject  # already computed earlier
    idx_FA = false_alarm
    # build (trial_df, abs_idx) lists
    trials_CR = convert_list_structure_with_idx(dff_correct_reject, idx_CR)
    trials_FA = convert_list_structure_with_idx(dff_false_alarm, idx_FA)
    # for hit vs CR
    print("running classifiers for HIT vs CR")
    trials_HIT = convert_list_structure_with_idx(dff_hit, hit)
    clf_results_hit = run_classifier_for_all_freqs(trials_HIT,
                                                   trials_CR,
                                                   conditions_played,
                                                   responses,
                                                   pre_trial_length,
                                                   rec_hz,
                                                   n_iter=500,
                                                   min_trials_single=4,
                                                   min_trials_pooled=2)
    with open(join(save_loc, 'classifier_results_by_freq_HIT_CR.pkl'), 'wb') as f:
        pickle.dump(clf_results_hit, f)

    plot_comb_and_sep(clf_results_hit,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name="HIT_CR")

    # --- run classifiers for each frequency + all together ---
    print(" running classifiers for CR vs. FA")
    clf_results = run_classifier_for_all_freqs(trials_CR,
                                               trials_FA,
                                               conditions_played,
                                               responses,
                                               pre_trial_length,
                                               rec_hz,
                                               n_iter=500,
                                               min_trials_single=4,
                                               min_trials_pooled=2)

    with open(join(save_loc, 'classifier_results_by_freq_CR_FA.pkl'), 'wb') as f:
        pickle.dump(clf_results, f)


    plot_comb_and_sep(clf_results,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name = "CR_FA")


    
    # for hit vs FA
    print("running classifiers for HIT vs FA")
    trials_FA = convert_list_structure_with_idx(dff_false_alarm, false_alarm)
    clf_results_fa = run_classifier_for_all_freqs(trials_HIT,
                                                  trials_FA,
                                                  conditions_played,
                                                  responses,
                                                  pre_trial_length,
                                                  rec_hz,
                                                  n_iter=500,
                                                  min_trials_single=4,
                                                  min_trials_pooled=2)
    with open(join(save_loc, 'classifier_results_by_freq_HIT_FA.pkl'), 'wb') as f:
        pickle.dump(clf_results_fa, f)

    plot_comb_and_sep(clf_results_fa,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name = "HIT_FA")

    # run the same analysis but with NB instead of SVM
    print("running classifiers for HIT vs CR with NB")
    clf_results_hit_nb = run_classifier_for_all_freqs_NB(trials_HIT,
                                                    trials_CR,
                                                    conditions_played,
                                                    responses,
                                                    pre_trial_length,
                                                    rec_hz,
                                                    n_iter=500,
                                                    min_trials_single=4,
                                                    min_trials_pooled=2)
    with open(join(save_loc, 'classifier_results_by_freq_HIT_CR_NB.pkl'), 'wb') as f:
        pickle.dump(clf_results_hit_nb, f)

    plot_comb_and_sep(clf_results_hit_nb,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name = "HIT_CR_NB")

    # run the same analysis but with NB instead of SVM
    print("running classifiers for HIT vs FA with NB")
    clf_results_fa_nb = run_classifier_for_all_freqs_NB(trials_HIT,
                                                        trials_FA,
                                                    conditions_played,
                                                    responses,
                                                    pre_trial_length,
                                                    rec_hz,
                                                    n_iter=500,
                                                    min_trials_single=4,
                                                    min_trials_pooled=2)
    with open(join(save_loc, 'classifier_results_by_freq_HIT_FA_NB.pkl'), 'wb') as f:
        pickle.dump(clf_results_fa_nb, f)

    plot_comb_and_sep(clf_results_fa_nb,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name = "HIT_FA_NB")

    # run the same analysis but with NB instead of SVM
    print("running classifiers for CR vs FA with NB")
    clf_results_cr_nb = run_classifier_for_all_freqs_NB(trials_CR,
                                                    trials_FA,
                                                    conditions_played,
                                                    responses,
                                                    pre_trial_length,
                                                    rec_hz,
                                                    n_iter=500,
                                                    min_trials_single=4,
                                                    min_trials_pooled=2)

    with open(join(save_loc, 'classifier_results_by_freq_CR_FA_NB.pkl'), 'wb') as f:
        pickle.dump(clf_results_cr_nb, f)

    plot_comb_and_sep(clf_results_cr_nb,
                      pre_trial_len=pre_trial_length,
                      rec_hz=rec_hz,
                      save_dir=save_loc,
                      file_name = "CR_FA_NB")
    
