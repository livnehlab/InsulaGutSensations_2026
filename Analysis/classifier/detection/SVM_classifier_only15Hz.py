"""
Detection day: running (sliding-window) linear SVM classifier of the population activity
(one session).

The dF/F of each neuron is z-scored over the whole recording and split into trials (each
trial gets the last 4 s of the previous trial as its pre-trial period, without overlap).
Trials are divided by history (after a hit / not after a hit). The classifier is trained on
the mean activity 2-5 s after trial start and tested in a sliding (leading) window of
round(0.3 * rec_hz) frames up to 17 s; 500 iterations with history-matched sampling and a
70/30 train/test split per class.

Saved in <save_loc> as [mean, SEM] per time bin:
  classifier_CR_FA_not_after_combined.pkl        CR vs FA
  classifier_FA_HIT_combined.pkl                 FA vs HIT
  classifier_CR_HIT_combined_not_balanced.pkl    CR vs HIT
  classifier_CR_HIT_combined_balanced.pkl        CR vs HIT, CR sampled with the FA history counts
plus the plots. The across-mice average is made by plot_classifier_accuracy_combined_all_mice.py.
Requires pandas < 2.0 (see the compatibility patches below).
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
import sys
import pandas
import pandas.core.indexes

# Compatibility patches for unpickling the dF/F tables saved with an older pandas version
# (the Int64Index import below requires pandas < 2.0)
# Patch for pandas.core.indexes.numeric
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes

# Patch for Int64Index
from pandas import Int64Index
setattr(pandas.core.indexes, 'Int64Index', Int64Index)

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
con1 = 5  # 1 sec distractor light; 4 sec Opto stim (15 Hz between 2p frames, 10ms, max intensity); 2 sec EB odor; 2 sec delay (mouse not supposed to lick).
con2 = 6  # 1 sec distractor light; 4 sec delay; 2 sec EB odor; 2 sec reward window for operant licking
pre_trial_length = 4 * rec_hz
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz


def x_lim_calc(df_run_list):
    # Find the third-longest trial length
    longest_length = 0
    second_longest_length = 0
    third_longest_length = 0
    for i in range(0, len(df_run_list)):
        sublist_length = np.count_nonzero(~np.isnan(df_run_list.iloc[i, :]))

        if sublist_length > longest_length:
            third_longest_length = second_longest_length
            second_longest_length = longest_length
            longest_length = sublist_length
        elif sublist_length > second_longest_length:
            third_longest_length = second_longest_length
            second_longest_length = sublist_length
        elif sublist_length > third_longest_length:
            third_longest_length = sublist_length

    print("Length of the longest sublist:", longest_length)
    print("Length of the second longest sublist:", second_longest_length)
    print("Length of the third longest sublist:", third_longest_length)
    xlim = third_longest_length
    return xlim
def add_axvline_opto(axes, pre_trial_length, rec_hz):
    # distract light
    axes.axvline(x=pre_trial_length, color='white', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 9 * rec_hz, color='white', linestyle='--', linewidth=0.7)
    # opto
    axes.axvline(x=pre_trial_length + 1*rec_hz, color='yellow', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 1*rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=0.7)
    # odor
    axes.axvline(x=pre_trial_length + 5 * rec_hz, color='green', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='green', linestyle='--', linewidth=0.7)

def add_axvline(axes, rewards, pre_trial_length, rec_hz, position):
    # distract light
    axes.axvline(x=pre_trial_length, color='white', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 9 * rec_hz, color='white', linestyle='--', linewidth=0.7)
    # odor
    axes.axvline(x=pre_trial_length + 5 * rec_hz, color='green', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='green', linestyle='--', linewidth=0.7)
    # Adding asterisks where rewards occurred
    if position == (0,0):
        for index, reward in enumerate(rewards):
            if len(reward) > 0:
                # Placing the '*' marker on the heatmap at the specified reward index for each trial
                # Adjust coordinates if needed, considering the heatmap's axis configuration
                axes.text(reward + pre_trial_length, index + 0.5, '*', color='white', ha='center', va='center', fontsize=10)

def create_figures(dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, rewards_times, ROOT_PATH, pre_trial_length,
                   rec_hz, folder_name, num_neurons):
    folder = join(ROOT_PATH, folder_name)
    os.makedirs(folder, exist_ok=True)

    for num in range(num_neurons):
        xlim = x_lim_calc(dff_hit[num])
        TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 10) / rec_hz

        # Define global color scale limits based on all conditions to maintain consistency across plots
        data_min = min(df.min().min() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        data_max = max(df.max().max() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        # Configure one figure with four subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 24),
                                  gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        # Plot each condition in a separate subplot
        sns.heatmap(dff_hit[num], ax=axes[0, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_miss[num], ax=axes[1, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_correct_reject[num], ax=axes[0, 1], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_false_alarm[num], ax=axes[1, 1], vmin=data_min, vmax=data_max)

        # Add the event lines
        add_axvline(axes[0, 0], rewards_times, pre_trial_length, rec_hz, position=(0,0))
        add_axvline(axes[1, 0], rewards_times, pre_trial_length, rec_hz, position=(1,0))
        add_axvline_opto(axes[0, 1], pre_trial_length, rec_hz)
        add_axvline_opto(axes[1, 1], pre_trial_length, rec_hz)

        # Set titles for each subplot
        axes[0, 0].set_title('Hit')
        axes[1, 0].set_title('Miss')
        axes[0, 1].set_title('Correct Reject')
        axes[1, 1].set_title('False Alarm')

        # Set common features such as ticks and labels
        for ax in axes.flat:
            ax.set_xlim(0, xlim)
        ax.set_xticks(np.arange(0, xlim, step=rec_hz * 10))
        ax.set_xticklabels(TICKS_LABELS)

        # Save the complete figure
        plt.savefig(join(folder, f'neuron_{num}_combined.png'))
        plt.close(fig)


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

def plot_normalized_data(data, sort_ind, x_lim, data_max, data_min,PATH, filename):
    reordered_data = [data[i] for i in sort_ind]

    axes = sns.heatmap(reordered_data, vmin=data_min, vmax=data_max)
    # Add vertical lines
    axes.axvline(x=300, color='k', linestyle='--')
    axes.axvline(x=360, color='k', linestyle='--')
    # Set the x-axis limit and ticks
    axes.set_xlim(0, x_lim)
    axes.set_xticks(np.arange(0, x_lim, step=rec_hz * 10))
    # axes.set_xticklabels(TICKS_LABELS[0])
    axes.set_xlabel('Time(sec)')
    axes.set_ylabel('neuron')
    plt.savefig(join(PATH, filename))
    plt.show()

def baseline_subtract_neurons(norm_data, pre_trial_length, baseline_time):
    concat_data = copy.deepcopy(norm_data)
    j = 0
    for trial in concat_data:
        i = 0
        for neuron in trial:
            # Calculate baseline activity
            neuron = np.asarray(neuron)
            baseline = np.mean(neuron[pre_trial_length - baseline_time:pre_trial_length])

            # Subtract baseline from all data points
            sub_neuron = neuron - baseline
            concat_data[j][i] = sub_neuron
            i += 1
        j += 1

    return concat_data


def global_zscore_neurons(data):
    """
    data is a list of length (num_trials),
    where data[t] is a list of length (num_neurons),
    and data[t][n] is a 1D np.array of time points for neuron n in trial t.

    This function z-scores each neuron across all trials/time points:
       z = ( x - mean ) / std

    Returns the same structure, but with each neuron z-scored globally.
    """
    # Make a deep copy so we don't modify the original
    zscored_data = copy.deepcopy(data)

    num_trials = len(zscored_data)
    if num_trials == 0:
        return zscored_data

    # Number of neurons = length of zscored_data[0]
    # (assuming each trial has the same number of neurons)
    num_neurons = len(zscored_data[0])

    for neuron_idx in range(num_neurons):
        # 1) Gather all time points for this neuron across all trials
        all_time_points = []
        for trial_idx in range(num_trials):
            all_time_points.append(zscored_data[trial_idx][neuron_idx])
        # Concatenate into one long 1D array
        all_time_points = np.concatenate(all_time_points)

        # 2) Compute mean and std across *all* those points
        mu = np.mean(all_time_points)
        sigma = np.std(all_time_points)

        # Avoid divide-by-zero
        if sigma == 0:
            sigma = 1.0

        # 3) Apply z-score to each trial for this neuron
        for trial_idx in range(num_trials):
            zscored_data[trial_idx][neuron_idx] = (zscored_data[trial_idx][neuron_idx] - mu) / sigma

    return zscored_data

def train_svm_classifier_sliding_window_combined(
    trials_data_CR_not_after, trials_data_FA_not_after,
    trials_data_CR_after, trials_data_FA_after,
    pre_trial_length, rec_hz, n_iterations=100
):
    """
    Sliding-window classifier with history-matched sampling. In every iteration the same
    number of "not after hit" trials and of "after hit" trials is drawn from both classes
    (class 0 = first lists, class 1 = second lists), each class is split 70/30 into
    train/test, the classifier is trained on the mean activity 2-5 s after trial start, and
    it is tested on the combined test set in a leading window of round(0.3 * rec_hz) frames
    up to 17 s. Returns the mean and SEM over iterations, per time bin.
    """
    accuracies_all = []
    stimu_start = pre_trial_length + 2 * rec_hz  # Start of the stimulus window
    stimu_end = pre_trial_length + 5 * rec_hz  # End of the stimulus window
    for iter_idx in range(n_iterations):
        # Sample for "not after"
        CR_not_after_sampled, FA_not_after_sampled = sample_min_trials(
            trials_data_CR_not_after, trials_data_FA_not_after, random_state=None
        )
        # Sample for "after"
        CR_after_sampled, FA_after_sampled = sample_min_trials(
            trials_data_CR_after, trials_data_FA_after, random_state=None
        )
        # Combine for each condition
        CR_combined = CR_not_after_sampled + CR_after_sampled
        FA_combined = FA_not_after_sampled + FA_after_sampled

        # --- Split each condition into training/testing sets separately ---
        X_cr_train, X_cr_test = train_test_split(CR_combined, test_size=0.3, random_state=None)
        X_hit_train, X_hit_test = train_test_split(FA_combined, test_size=0.3, random_state=None)

        # --- Combine the training sets ---
        X_train = np.vstack((X_cr_train, X_hit_train))
        y_train = np.concatenate((np.zeros(len(X_cr_train)), np.ones(len(X_hit_train))))

        # --- Combine the test sets ---
        X_test_cr = np.array(X_cr_test)
        y_test_cr = np.zeros(len(X_test_cr))
        X_test_hit = np.array(X_hit_test)
        y_test_hit = np.ones(len(X_test_hit))

        # Combined test set
        X_test_combined = np.vstack((X_test_cr, X_test_hit))
        y_test_combined = np.concatenate((y_test_cr, y_test_hit))

        # --- Feature extraction for training ---
        X_train_window = np.array([
            np.nanmean(trial[:, stimu_start:stimu_end], axis=1) for trial in X_train
        ])

        # --- Train the SVM classifier ---
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_window, y_train)

        # --- Sliding-window parameters ---
        window_size = round(0.3 * rec_hz)
        step_size   = 1
        last_col    = 17 * rec_hz  # (Assuming trials have at least 17 seconds of data)

        # List to store accuracy for each time point in this iteration
        iteration_accuracies = []

        # --- Sliding-window loop (leading/causal window) ---
        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size

            # Extract features for the combined test set in [start:end]
            X_test_window = np.array([
                np.nanmean(trial[:, start:end], axis=1) for trial in X_test_combined
            ])

            # Predict & compute accuracy
            if len(X_test_window) > 0:
                y_pred = clf.predict(X_test_window)
                accuracy_combined = accuracy_score(y_test_combined, y_pred)
            else:
                accuracy_combined = np.nan

            iteration_accuracies.append(accuracy_combined)

        # Collect the iteration's accuracies
        accuracies_all.append(np.array(iteration_accuracies))

    # Convert to array: shape [n_iterations x num_windows]
    accuracies_all = np.array(accuracies_all)

    # Compute mean & SEM across iterations, for each window
    avg_accuracy_combined = np.nanmean(accuracies_all, axis=0)
    sem_accuracy_combined = np.nanstd(accuracies_all, axis=0) / np.sqrt(n_iterations)

    return avg_accuracy_combined, sem_accuracy_combined
def train_svm_classifier_sliding_window_separate(correct_rejected_samples, hit_samples, pre_trial_length, rec_hz,
                                                 n_iterations=100):
    """
    Trains an SVM classifier on given Correct Rejected and HIT samples using bootstrapping with class balancing.
    The data for each condition is first split into training and testing sets separately, and then the classifier
    is trained on the combined training data and evaluated (using a sliding window) on each condition's test set.

    In this modified version, the sliding window is 'leading' in that for each time point t,
    the classifier is evaluated using the average of the data in the window immediately preceding t.

    Parameters:
        correct_rejected_samples (list or np.ndarray): List/array of trials for Correct Rejected samples.
            Each trial is expected to be a 2D array (n_neurons x n_timepoints).
        hit_samples (list or np.ndarray): List/array of trials for HIT samples.
            Each trial is expected to be a 2D array (n_neurons x n_timepoints).
        pre_trial_length (int): Pre-trial length (in number of samples).
        rec_hz (int): Recording frequency in Hz.
        n_iterations (int): Number of bootstrapping iterations.

    Returns:
        avg_accuracy_cr: Array of average sliding-window accuracies for Correct Rejected samples.
        avg_accuracy_hit: Array of average sliding-window accuracies for HIT samples.
        sem_accuracy_cr: Array of standard errors of the mean for Correct Rejected samples.
        sem_accuracy_hit: Array of standard errors of the mean for HIT samples.
    """
    accuracies_cr_all = []  # will collect sliding-window accuracies for CR test set across iterations
    accuracies_hit_all = []  # will collect sliding-window accuracies for HIT test set across iterations

    # For class balancing, determine the number of trials to use per condition.
    n_samples = min(len(correct_rejected_samples), len(hit_samples))

    # Define the training window (for feature extraction)
    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end = pre_trial_length + 5 * rec_hz

    for iter_idx in range(n_iterations):
        print(f"Iteration {iter_idx + 1} of {n_iterations}")

        # --- Resample each condition independently ---
        X_cr_resampled = resample(correct_rejected_samples, replace=False, n_samples=n_samples, random_state=None)
        X_hit_resampled = resample(hit_samples, replace=False, n_samples=n_samples, random_state=None)

        # --- Split each condition into training and testing sets separately ---
        X_cr_train, X_cr_test = train_test_split(X_cr_resampled, test_size=0.3, random_state=None)
        X_hit_train, X_hit_test = train_test_split(X_hit_resampled, test_size=0.3, random_state=None)

        # --- Combine training sets (with appropriate labels) ---
        X_train = np.vstack((X_cr_train, X_hit_train))
        y_train = np.concatenate((np.zeros(len(X_cr_train)), np.ones(len(X_hit_train))))

        # --- Keep test sets separate ---
        X_test_cr = np.array(X_cr_test)  # test set for Correct Rejected (label 0)
        y_test_cr = np.zeros(len(X_test_cr))
        X_test_hit = np.array(X_hit_test)  # test set for HIT (label 1)
        y_test_hit = np.ones(len(X_test_hit))

        # --- Feature extraction for training ---
        # For each trial in the combined training set, average over the defined time window.
        X_train_window = np.asarray([
            np.nanmean(trial[:, stimu_start:stimu_end], axis=1) for trial in X_train
        ])

        # --- Train the SVM classifier ---
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_window, y_train)

        # --- Sliding-window evaluation with leading (causal) window ---
        # Here, we define a window of a given size (e.g., 0.3 seconds) and for each time point t
        # we compute the feature using the data from [t-window_size, t].
        window_size = round(0.3 * rec_hz)
        step_size = 1

        # Determine the total number of timepoints in a trial.
        last_col = 17 * rec_hz  # (Assuming trials have at least 17 seconds of data)

        accuracies_cr = []
        accuracies_hit = []

        # Loop from time point 'window_size' up to 'last_col'
        # 'end' here represents the current frame (or time point) and the window is the data leading up to it.
        for end in range(window_size, last_col + 1, step_size):
            start = end - window_size

            # Extract features for the current window (the window preceding the current frame)
            X_test_window_cr = np.asarray([
                np.nanmean(trial[:, start:end], axis=1) for trial in X_test_cr
            ])
            X_test_window_hit = np.asarray([
                np.nanmean(trial[:, start:end], axis=1) for trial in X_test_hit
            ])

            # Make predictions and compute accuracy for Correct Rejected
            if X_test_window_cr.shape[0] > 0:
                y_pred_cr = clf.predict(X_test_window_cr)
                accuracy_cr = accuracy_score(y_test_cr, y_pred_cr)
            else:
                accuracy_cr = np.nan

            # Make predictions and compute accuracy for HIT
            if X_test_window_hit.shape[0] > 0:
                y_pred_hit = clf.predict(X_test_window_hit)
                accuracy_hit = accuracy_score(y_test_hit, y_pred_hit)
            else:
                accuracy_hit = np.nan

            accuracies_cr.append(accuracy_cr)
            accuracies_hit.append(accuracy_hit)

        # Append the sliding-window accuracies from this iteration.
        accuracies_cr_all.append(np.array(accuracies_cr))
        accuracies_hit_all.append(np.array(accuracies_hit))

    # Convert collected accuracies into arrays with shape (n_iterations, num_windows)
    accuracies_cr_all = np.array(accuracies_cr_all)
    accuracies_hit_all = np.array(accuracies_hit_all)

    # Compute the average and standard error (SEM) across iterations for each sliding window time point.
    avg_accuracy_cr = np.nanmean(accuracies_cr_all, axis=0)
    avg_accuracy_hit = np.nanmean(accuracies_hit_all, axis=0)
    sem_accuracy_cr = np.nanstd(accuracies_cr_all, axis=0) / np.sqrt(n_iterations)
    sem_accuracy_hit = np.nanstd(accuracies_hit_all, axis=0) / np.sqrt(n_iterations)

    return avg_accuracy_cr, avg_accuracy_hit, sem_accuracy_cr, sem_accuracy_hit

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
def convert_list_strucutre(data):
    new_data = []
    # Iterate over the range of trials
    for trial_index in range(data[0].shape[0]):
        # Initialize an empty list to collect data from all neurons for the current trial
        trial_list = []

        # Iterate over each neuron's dataframe and collect the current trial's data
        for neuron_index in range(len(data)):
            neuron_data = data[neuron_index].iloc[trial_index].values
            trial_list.append(neuron_data)

        # Convert the list of neuron data into a dataframe
        trial_df = pd.DataFrame(trial_list)

        # Append the trial dataframe to the trials_data list
        new_data.append(trial_df)
    return new_data


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

def plot_accuracy(res_con1, res_con2, sem_con1, sem_con2, label_con1, label_con2, file_name):
    xlim = 271  # frames to plot
    TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 2) / rec_hz
    plt.figure(figsize=(20, 10))
    plt.plot(res_con1, label=label_con1, color='black')
    plt.plot(res_con2, label=label_con2, color='r')
    # add the error bars
    plt.fill_between(range(len(res_con1)), res_con1 - sem_con1, res_con1 + sem_con1, alpha=0.2, color='gray')
    plt.fill_between(range(len(res_con2)), res_con2 - sem_con2, res_con2 + sem_con2, alpha=0.2, color='pink')
    plt.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    plt.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    # opto
    plt.axvline(x=pre_trial_length + 1*rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    plt.axvline(x=pre_trial_length + 1*rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    # odor
    plt.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    plt.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    # add shading for the training window
    plt.axvspan(pre_trial_length + 2 * rec_hz, pre_trial_length + 5 * rec_hz, color='gray', alpha=0.2)
    plt.xticks(np.arange(0, xlim, step=rec_hz * 2), TICKS_LABELS)
    plt.yticks(np.arange(0, 1.1, step=0.1), np.arange(0, 110, step=10))
    plt.xlabel('Time(sec)')
    plt.ylabel('Recognized as CR (%)') #Accuracy (%)
    plt.legend()
    plt.savefig(join(save_loc, file_name))
    plt.savefig(join(save_loc, file_name.replace('.png', '.pdf')))
    plt.show()

def norm_to_max(cells_data):
    norm_df = []
    for i in range(len(cells_data)):
        max = np.percentile(cells_data.iloc[i], 99)
        norm_df.append(cells_data.iloc[i] / max)
    norm_df = np.asarray(norm_df)
    return norm_df

def sample_min_trials(list1, list2, n=None, random_state=None):
    """
    Sample n=min(len(list1), len(list2)) from each list (without replacement).
    Returns two lists of length n.
    """
    n1, n2 = len(list1), len(list2)
    if n is None:
        n = min(n1, n2)
    rng = np.random.default_rng(random_state)
    idx1 = rng.choice(n1, n, replace=False)
    idx2 = rng.choice(n2, n, replace=False)
    return [list1[i] for i in idx1], [list2[i] for i in idx2]

if '__main__' == __name__:
    # Run once per session. The detection sessions used:
    #   mouse01/sess01, mouse05/sess03, mouse09/sess05, mouse14/sess11, mouse10/sess07
    # (mouse01 was run with ML\data2, mouse14 with ML\data, its only data folder)
    day = ['sess11']
    ROOT_PATH = DATA_ROOT + r'\mouse14\{}\ML\data'.format(day[0])
    # The results used in the paper were saved in classifier_leading_z_no_sub_history_v2
    # (the folder read by plot_classifier_accuracy_combined_all_mice.py)
    save_loc = DATA_ROOT + r'\mouse14\{}\ML\data\classifier_leading_z_no_sub_history'.format(day[0])
    os.makedirs(save_loc, exist_ok=True)
    # reads pickles written with numpy >= 2 (numpy._core) in an older numpy
    class NumpyCoreUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Patch any module path that starts with 'numpy._core'
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
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
        dff_4min = NumpyCoreUnpickler(f).load()

    # normalize the data
    # dff_norm = norm_to_max(dff_4min)
    # z-score the data
    dff_norm =[]
    for i in range(len(dff_4min)):
        mean = np.nanmean(dff_4min.iloc[i])
        std = np.nanstd(dff_4min.iloc[i])
        dff_norm.append((dff_4min.iloc[i] - mean) / std)
    dff_not_norm = pd.DataFrame(dff_norm)
    # divide into trials
    dff_run_list, pre_runs = divide_to_trials(dff_not_norm, trials_ind, run_len,
                                                                        pre_trial_length)


    dff_sub = dff_run_list

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

    # correct_reject = np.where(np.logical_or(responses == 4, responses == 5))[0]
    # hit = np.where(np.logical_or(responses == 3, responses == 2))[0]
    correct_reject = np.where(responses == 4)[0]
    false_alarm = np.where(responses == 5)[0]
    hit = np.where(responses == 3)[0]
    miss = np.where(responses == 2)[0]
    for i in range(len(valve_open)):
        changes = np.diff(np.concatenate(([0], valve_open[i] > 1)))
        valve_open[i] = np.where(changes == 1)[0]
    rewards_times = [valve_open[i] for i in hit]
    # divide the data to the different conditions
    dff_correct_reject, dff_false_alarm, dff_hit, dff_miss = divide_to_conditions(dff_sub, correct_reject,
                                                                                  false_alarm, hit, miss)
    # Now divide the trials by history:
    (data_cr_after, data_cr_not_after,
     data_fa_after, data_fa_not_after,
     data_hit_after, data_hit_not_after,
     data_miss_after, data_miss_not_after) = divide_to_conditions_by_history(
        dff_sub, correct_reject, false_alarm, hit, miss, responses)

    correct_rejected_samples = dff_correct_reject
    false_alarm_samples = dff_false_alarm
    hit_samples = dff_hit
    # convert the data to a list of trials where each trial is a list of neurons
    trials_data_CR = convert_list_strucutre(correct_rejected_samples)
    trials_data_FA = convert_list_strucutre(false_alarm_samples)
    trials_data_hit = convert_list_strucutre(hit_samples)

    # and for the trials "not after"
    trials_data_CR_not_after = convert_list_strucutre(data_cr_not_after)
    trials_data_FA_not_after = convert_list_strucutre(data_fa_not_after)
    trials_data_hit_not_after = convert_list_strucutre(data_hit_not_after)

    # and for the trials "after"
    trials_data_CR_after = convert_list_strucutre(data_cr_after)
    trials_data_FA_after = convert_list_strucutre(data_fa_after)
    trials_data_hit_after = convert_list_strucutre(data_hit_after)

    # "not after"
    CR_not_after_sampled, FA_not_after_sampled = sample_min_trials(trials_data_CR_not_after, trials_data_FA_not_after, random_state=None)

    # "after"
    CR_after_sampled, FA_after_sampled = sample_min_trials(trials_data_CR_after, trials_data_FA_after, random_state=None)

    # Combine for each condition
    CR_combined = CR_not_after_sampled + CR_after_sampled
    FA_combined = FA_not_after_sampled + FA_after_sampled

    # CR vs FA classifier (the history-matched sampling is redone in every iteration;
    # CR_combined / FA_combined above are not used)
    avg_accuracy_combined, sem_accuracy_combined = train_svm_classifier_sliding_window_combined(
        trials_data_CR_not_after, trials_data_FA_not_after,
        trials_data_CR_after, trials_data_FA_after,
        pre_trial_length, rec_hz, n_iterations=500
    )

    with open(join(save_loc, 'classifier_CR_FA_not_after_combined.pkl'), 'wb') as f:
        pickle.dump([avg_accuracy_combined, sem_accuracy_combined], f)

    # FA vs Hit classifier (combined version)
    avg_accuracy_FA_hit_combined, sem_accuracy_FA_hit_combined = train_svm_classifier_sliding_window_combined(
        trials_data_FA_not_after, trials_data_hit_not_after,
        trials_data_FA_after, trials_data_hit_after,
        pre_trial_length, rec_hz, n_iterations=500
    )

    # Save FA vs Hit classifier results (combined version)
    with open(join(save_loc, 'classifier_FA_HIT_combined.pkl'), 'wb') as f:
        pickle.dump([avg_accuracy_FA_hit_combined, sem_accuracy_FA_hit_combined], f)

    # CR vs Hit classifier (combined version) - not balanced
    avg_accuracy_CR_hit_combined_not_balanced, sem_accuracy_CR_hit_combined_not_balanced = train_svm_classifier_sliding_window_combined(
        trials_data_CR_not_after, trials_data_hit_not_after,
        trials_data_CR_after, trials_data_hit_after,
        pre_trial_length, rec_hz, n_iterations=500
    )
    with open(join(save_loc, 'classifier_CR_HIT_combined_not_balanced.pkl'), 'wb') as f:
        pickle.dump([avg_accuracy_CR_hit_combined_not_balanced, sem_accuracy_CR_hit_combined_not_balanced], f)

    # CR vs Hit classifier (combined version) - balanced with FA proportions
    # Determine the FA trial counts in not_after and after
    n_FA_not_after = len(trials_data_FA_not_after)
    n_FA_after = len(trials_data_FA_after)

    # Sample CR trials to match FA distribution
    n_CR_not_after_to_sample = int(n_FA_not_after)
    n_CR_after_to_sample = int(n_FA_after)

    CR_not_after_sampled_for_hit, _ = sample_min_trials(
        trials_data_CR_not_after, trials_data_CR_not_after,
        n=n_CR_not_after_to_sample, random_state=None
    )
    CR_after_sampled_for_hit, _ = sample_min_trials(
        trials_data_CR_after, trials_data_CR_after,
        n=n_CR_after_to_sample, random_state=None
    )

    avg_accuracy_CR_hit_combined, sem_accuracy_CR_hit_combined = train_svm_classifier_sliding_window_combined(
        CR_not_after_sampled_for_hit, trials_data_hit_not_after,
        CR_after_sampled_for_hit, trials_data_hit_after,
        pre_trial_length, rec_hz, n_iterations=500
    )
    with open(join(save_loc, 'classifier_CR_HIT_combined_balanced.pkl'), 'wb') as f:
        pickle.dump([avg_accuracy_CR_hit_combined, sem_accuracy_CR_hit_combined], f)

    # Plot FA vs Hit combined accuracy
    xlim = 271  # frames to plot
    TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 2) / rec_hz
    plt.figure(figsize=(20, 10))
    plt.plot(avg_accuracy_FA_hit_combined, color='red', label='FA vs Hit Combined Accuracy')
    plt.fill_between(range(len(avg_accuracy_FA_hit_combined)),
                     avg_accuracy_FA_hit_combined - sem_accuracy_FA_hit_combined,
                     avg_accuracy_FA_hit_combined + sem_accuracy_FA_hit_combined,
                     alpha=0.2, color='red')
    plt.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    plt.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    plt.axvline(x=pre_trial_length + 1 * rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    plt.axvline(x=pre_trial_length + 1 * rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    plt.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    plt.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    plt.axvspan(pre_trial_length + 2 * rec_hz, pre_trial_length + 5 * rec_hz, color='gray', alpha=0.2)
    plt.xticks(np.arange(0, xlim, step=rec_hz * 2), TICKS_LABELS)
    plt.yticks(np.arange(0, 1.1, step=0.1), np.arange(0, 110, step=10))
    plt.xlabel('Time(sec)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(join(save_loc, 'running_classifier_FA_HIT_combined_accuracy.png'))
    plt.savefig(join(save_loc, 'running_classifier_FA_HIT_combined_accuracy.pdf'))
    plt.close()

    # Plot CR vs Hit combined accuracy (balanced)
    plt.figure(figsize=(20, 10))
    plt.plot(avg_accuracy_CR_hit_combined, color='green', label='CR vs Hit Combined Accuracy (balanced)')
    plt.fill_between(range(len(avg_accuracy_CR_hit_combined)),
                     avg_accuracy_CR_hit_combined - sem_accuracy_CR_hit_combined,
                     avg_accuracy_CR_hit_combined + sem_accuracy_CR_hit_combined,
                     alpha=0.2, color='green')
    plt.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    plt.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    plt.axvline(x=pre_trial_length + 1 * rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    plt.axvline(x=pre_trial_length + 1 * rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    plt.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    plt.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    plt.axvspan(pre_trial_length + 2 * rec_hz, pre_trial_length + 5 * rec_hz, color='gray', alpha=0.2)
    plt.xticks(np.arange(0, xlim, step=rec_hz * 2), TICKS_LABELS)
    plt.yticks(np.arange(0, 1.1, step=0.1), np.arange(0, 110, step=10))
    plt.xlabel('Time(sec)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(join(save_loc, 'running_classifier_CR_HIT_combined_accuracy_balanced.png'))
    plt.savefig(join(save_loc, 'running_classifier_CR_HIT_combined_accuracy_balanced.pdf'))
    plt.close()







    # accuracy_CR, accuracy_FA, sem_CR, sem_FA = train_svm_classifier_sliding_window_separate(trials_data_CR, trials_data_FA, pre_trial_length, rec_hz, n_iterations=100)
    # accuracy_CR_not_after, accuracy_FA_not_after,  sem_CR_not_after, sem_FA_not_after = train_svm_classifier_sliding_window_separate(trials_data_CR,
    #                                                                                               trials_data_FA,
    #                                                                                               pre_trial_length,
    #                                                                                               rec_hz,
    #                                                                                               n_iterations=100)
    # accuracy_CR_after, accuracy_FA_after, sem_CR_after, sem_FA_after = train_svm_classifier_sliding_window_separate(trials_data_CR_after,
    #                                                                                                 trials_data_FA_after,
    #                                                                                                 pre_trial_length,
    #                                                                                                 rec_hz,
    #                                                                                                 n_iterations=100)
    #
    # # and for the HIT vs CR
    # accuracy_CR_hit, accuracy_hit, sem_CR_hit, sem_hit = train_svm_classifier_sliding_window_separate(trials_data_CR, trials_data_hit, pre_trial_length, rec_hz, n_iterations=100)
    # accuracy_CR_hit_not_after, accuracy_hit_not_after,  sem_CR_hit_not_after, sem_hit_not_after = train_svm_classifier_sliding_window_separate(trials_data_CR_not_after,
    #                                                                                               trials_data_hit_not_after,
    #                                                                                               pre_trial_length,
    #                                                                                               rec_hz,
    #                                                                                               n_iterations=100)
    # accuracy_CR_hit_after, accuracy_hit_after, sem_CR_hit_after, sem_hit_after = train_svm_classifier_sliding_window_separate(trials_data_CR_after,
    #                                                                                                 trials_data_hit_after,
    #                                                                                                 pre_trial_length,
    #                                                                                                 rec_hz,
    #                                                                                                 n_iterations=100)

    # save the accuracy
    # with open(join(data_loc, 'classifier_CR_FA.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_CR, accuracy_FA, sem_CR, sem_FA], f)

    # with open(join(data_loc, 'classifier_CR_FA_after.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_CR_after, accuracy_FA_after, sem_CR_after, sem_FA_after], f)
    # with open(join(data_loc, 'classifier_CR_HIT.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_CR_hit, accuracy_hit, sem_CR_hit, sem_hit], f)
    # with open(join(data_loc, 'classifier_CR_HIT_not_after.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_CR_hit_not_after, accuracy_hit_not_after, sem_CR_hit_not_after, sem_hit_not_after], f)
    # with open(join(data_loc, 'classifier_CR_HIT_after.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_CR_hit_after, accuracy_hit_after, sem_CR_hit_after, sem_hit_after], f)


    # save the data
    # with open(join(data_loc, 'running_classifier.pkl'), 'wb') as f:
    #     pickle.dump([accuracy_0hz, accuracy_15hz, sem_0hz, sem_15hz], f)

    # accuracy_both, sem_both = train_svm_classifier_sliding_window_both(trials_data_15Hz,
    #                                                                                      trials_data_0Hz,
    #                                                                                      pre_trial_length, rec_hz,
    #                                                                                      n_iterations=1000)

    # xlim = 271  # frames to plot
    # TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 2) / rec_hz
    # plt.figure(figsize=(20, 10))
    # plt.plot(accuracy_both, color='black')
    # #add the error bars
    # plt.fill_between(range(len(accuracy_both)), accuracy_both - sem_both, accuracy_both + sem_both, alpha=0.2, color='gray')
    # plt.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    # plt.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    # # opto
    # plt.axvline(x=pre_trial_length + 1*rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    # plt.axvline(x=pre_trial_length + 1*rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    # # odor
    # plt.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    # plt.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    # # add shading for the training window
    # plt.axvspan(pre_trial_length + 3 * rec_hz, pre_trial_length + 5 * rec_hz, color='gray', alpha=0.2)
    # plt.xticks(np.arange(0, xlim, step=rec_hz * 2), TICKS_LABELS)
    # plt.yticks(np.arange(0, 1.1, step=0.1), np.arange(0, 110, step=10))
    # plt.xlabel('Time(sec)')
    # plt.ylabel('% Correct')
    # plt.legend()
    # plt.savefig(join(data_loc, 'running_classifier_both.png'))
    # plt.savefig(join(data_loc, 'running_classifier_both.pdf'))
    # plt.show()


    # accuracy_FA = 1 - accuracy_FA
    # plot_accuracy(accuracy_CR, accuracy_FA, sem_CR, sem_FA, 'CR', 'FA', 'running_classifier_z-entire_trial_CR_FA.png')
    # accuracy_FA_not_after = 1 - accuracy_FA_not_after
    # plot_accuracy(accuracy_CR_not_after, accuracy_FA_not_after, sem_CR_not_after, sem_FA_not_after, 'CR', 'FA', 'running_classifier_z-not_after_CR_FA.png')
    # accuracy_FA_after = 1 - accuracy_FA_after
    # plot_accuracy(accuracy_CR_after, accuracy_FA_after, sem_CR_after, sem_FA_after, 'CR', 'FA', 'running_classifier_z-after_CR_FA.png')
    #
    # # for HIT vs CR
    # accuracy_hit = 1 - accuracy_hit
    # plot_accuracy(accuracy_CR_hit, accuracy_hit, sem_CR_hit, sem_hit, 'CR', 'HIT', 'running_classifier_z-entire_trial_CR_HIT.png')
    # accuracy_hit_not_after = 1 - accuracy_hit_not_after
    # plot_accuracy(accuracy_CR_hit_not_after, accuracy_hit_not_after, sem_CR_hit_not_after, sem_hit_not_after, 'CR', 'HIT', 'running_classifier_z-not_after_CR_HIT.png')
    # accuracy_hit_after = 1 - accuracy_hit_after
    # plot_accuracy(accuracy_CR_hit_after, accuracy_hit_after, sem_CR_hit_after, sem_hit_after, 'CR', 'HIT', 'running_classifier_z-after_CR_HIT.png')
