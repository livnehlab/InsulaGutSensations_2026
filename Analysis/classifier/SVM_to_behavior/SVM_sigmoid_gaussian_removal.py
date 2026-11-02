"""
Psychophysics: SVM classifier after removing the Gaussian or the sigmoid neurons (one session).

The dF/F of each neuron is z-scored, split into trials, and averaged over the stimulus
window. For every training frequency (0.5-15 Hz) a linear SVM is trained to separate the
trials of that frequency (class 0) from the 0 Hz trials (class 1) and tested on every
frequency (500 bootstrap iterations). This is run on four sets of neurons:
  all_cells                     - all neurons
  no_gaussian                   - without the Gaussian-tuned neurons
  no_sigmoid                    - without the sigmoid-tuned neurons
  random_remove_avg_group_size  - 100 repeats, each removing a random set of neurons of the
                                  average size of the two groups (summarized in random_summary.pkl)

The Gaussian / sigmoid neuron indices are read from auroc_neurometric_zscore_no_sub_v4
(gaussian_neuron_indices.npy, sigmoid_neuron_indices.npy, saved by plot_max_responsive_fractions.py).
Outputs (per neuron set): svm_results.pkl, meta.pkl and plots, in
<save_loc>/svm_cell_subset_comparison/<set>/; these are read by
group_classifier_minimal_source_data.py.
"""
import csv
import math
import os
import pickle
import copy
from operator import itemgetter
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd
from os.path import join
import seaborn as sns
from scipy.optimize import curve_fit
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


def find_max_min_values(norm_data_list):
    """Finds the maximum and minimum values of the normalized data."""
    all_max = []
    all_min = []

    for norm_data in norm_data_list:
        max_z = np.nanmax(norm_data)
        min_z = np.nanmin(norm_data)
        all_max.append(max_z)
        all_min.append(min_z)

    max_val = np.max(all_max)
    min_val = np.min(all_min)

    return max_val, min_val


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
    axes.axvline(x=pre_trial_length + 1 * rec_hz, color='yellow', linestyle='--', linewidth=0.7)
    axes.axvline(x=pre_trial_length + 1 * rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=0.7)
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
    if position == (0, 0):
        for index, reward in enumerate(rewards):
            if len(reward) > 0:
                # Placing the '*' marker on the heatmap at the specified reward index for each trial
                # Adjust coordinates if needed, considering the heatmap's axis configuration
                axes.text(reward + pre_trial_length, index + 0.5, '*', color='white', ha='center', va='center',
                          fontsize=10)


def create_figures(dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, rewards_times, ROOT_PATH, pre_trial_length,
                   rec_hz, folder_name, num_neurons):
    folder = join(ROOT_PATH, folder_name)
    os.makedirs(folder, exist_ok=True)

    for num in range(num_neurons):
        xlim = x_lim_calc(dff_hit[num])
        TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 10) / rec_hz

        # Define global color scale limits based on all conditions to maintain consistency across plots
        data_min = min(
            df.min().min() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        data_max = max(
            df.max().max() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        # Configure one figure with four subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 24),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        # Plot each condition in a separate subplot
        sns.heatmap(dff_hit[num], ax=axes[0, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_miss[num], ax=axes[1, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_correct_reject[num], ax=axes[0, 1], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_false_alarm[num], ax=axes[1, 1], vmin=data_min, vmax=data_max)

        # Add the event lines
        add_axvline(axes[0, 0], rewards_times, pre_trial_length, rec_hz, position=(0, 0))
        add_axvline(axes[1, 0], rewards_times, pre_trial_length, rec_hz, position=(1, 0))
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


def concatenate_data(average_data):
    num_conditions = len(average_data)
    num_neurons = len(average_data[0])
    concat_data = []
    for neuron in range(num_neurons):
        concatenated_neuron = []
        for lst_index in range(num_conditions):
            data = average_data[lst_index][neuron]
            data = pd.DataFrame(data).transpose()
            concatenated_neuron.append(data)
        concatenated_neuron = pd.concat(concatenated_neuron, axis=1, ignore_index=True)
        concat_data.append(concatenated_neuron)
    return concat_data


def z_score_normalize(concat_matrix, ave_data):
    """Normalizes a matrix with the average response of each neuron using z-score normalization."""
    # For each neuron, calculate the mean and std of its responses across all data points.
    norm_df = []
    for i in range(len(concat_matrix)):
        mean = np.mean(concat_matrix[i], axis=1)[0]
        std = np.std(concat_matrix[i], axis=1)[0]
        # For each neuron, subtract the mean response from its response in each data point.
        z_df = ave_data[i] - mean
        # For each neuron, divide the resulting value by the standard deviation.
        z_df /= std
        norm_df.append(z_df)
    return norm_df


def min_max_normalize(concat_matrix, ave_data):
    minmax_df = []
    for i in range(len(concat_matrix)):
        min = np.min(concat_matrix[i], axis=1)[0]
        max = np.max(concat_matrix[i], axis=1)[0]
        diff = max - min
        # For each neuron, subtract the minimum from its response in each data point.
        subtract_df = ave_data[i] - min
        # For each neuron, divide the resulting value by the range.
        subtract_df /= diff
        minmax_df.append(subtract_df)

    return minmax_df


def find_min_max(data):
    max = np.nanmax(data)
    min = np.nanmin(data)
    return (max, min)


def plot_normalized_data(data, sort_ind, x_lim, data_max, data_min, PATH, filename):
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


def min_max_normalize(cells_data):
    minmax_df = []
    for i in range(len(cells_data)):
        # find the 1st and 99th percentile of each neuron
        min = np.percentile(cells_data.iloc[i], 1)
        max = np.percentile(cells_data.iloc[i], 99)
        diff = max - min
        # For each neuron, subtract the 1st percentile from its response in each data point.
        subtract_df = cells_data.iloc[i] - min
        # For each neuron, divide the resulting value by the range.
        subtract_df /= diff
        minmax_df.append(subtract_df)
    minmax_df = np.asarray(minmax_df)
    return minmax_df


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def significant_neurons(dff, pre_trial_length, rec_hz):
    baseline_start = 3 * rec_hz
    baseline_end = pre_trial_length
    stimulation_start = pre_trial_length + 1 * rec_hz
    stimulation_end = pre_trial_length + 1 * rec_hz + 4 * rec_hz
    p_values_cells = []
    for neuron in dff:  # for each neuron
        neuron = pd.DataFrame(neuron)
        p_values_cells.append(ttest_ind(np.mean(neuron.iloc[:, stimulation_start:stimulation_end], axis=1),
                                        np.mean(neuron.iloc[:, baseline_start:baseline_end], axis=1))[1])
    return p_values_cells


def fit_neurometric_curve(dff_run_list, stimulation_start, stimulation_end, conditions, save_curves):
    save_curves = join(save_loc, 'neurometric_curves')
    os.makedirs(save_curves, exist_ok=True)
    for i in range(len(dff_run_list)):
        neuron_activities = dff_run_list[i]

        # Frequency map (condition code -> Hz)
        frequency_map = {6: 0, 5: 15, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

        # Create a dictionary to collect average activities
        activity_dict = {freq: [] for freq in frequency_map.values()}

        # Process data
        for condition, activities in zip(conditions, neuron_activities):
            freq = frequency_map[condition]
            # Extract the activity slice for the stimulation period
            stimulation_activities = activities[stimulation_start:stimulation_end]
            # Calculate average activity
            average_activity = np.mean(stimulation_activities)
            activity_dict[freq].append(average_activity)

        # Calculate average activities for each frequency
        averages = {freq: np.mean(vals) if vals else 0 for freq, vals in activity_dict.items()}

        # Min-max normalization of averages
        min_val = min(averages.values())
        max_val = max(averages.values())
        norm_averages = {freq: (val - min_val) / (max_val - min_val) if max_val > min_val else 0 for freq, val in
                         averages.items()}

        # Sorting the averages for plotting
        sorted_frequencies = sorted(norm_averages.keys())
        sorted_averages = [norm_averages[freq] for freq in sorted_frequencies]

        # Plotting
        x_data = sorted_frequencies
        y_data = sorted_averages

        try:
            # Fit the sigmoid function to the data
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='trf')

            # Generate x values for the curve
            x_values = np.linspace(min(x_data), max(x_data), 100)
            # Generate y values from the fitted sigmoid function
            y_values = sigmoid(x_values, *popt)

            plt.figure(figsize=(20, 10))
            plt.plot(x_values, y_values, label='Fit', color='black')  # No scaling needed
            plt.scatter(x_data, np.array(y_data), color='black')  # No scaling needed
            plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
            plt.ylabel('Normalized df/f')
            plt.xticks(x_data, labels=[f"{freq} Hz" for freq in x_data])
            plt.yticks(np.linspace(0, 1, 11))  # Setting y ticks from 0 to 1 in steps of 0.1
            plt.savefig(join(save_curves, f'Neuron {i} - Neurometric Curve.png'))
            plt.show()
        except RuntimeError as e:
            print(f"Curve fitting failed for Neuron {i} due to: {e}")


def auROC_neurometric_curve(auROC, conditions, save_dir):
    responses = []
    for i in range(0, len(auROC)):
        responses.append(np.where(auROC[i] > 0.7, 1, 0))
    # create a neurometric curve
    frequency_map = {6: 0, 5: 15, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    neuron_num = 90
    # Create a dictionary to collect response counts
    response_dict = {freq: {'excited': 0, 'total': 0} for freq in frequency_map.values()}

    # Process data
    for condition, response in zip(conditions, responses[neuron_num]):
        freq = frequency_map[condition]
        if response == 1:
            response_dict[freq]['excited'] += 1
        response_dict[freq]['total'] += 1

    # Fraction of excited trials for each frequency
    fractions_licked = {freq: res['excited'] / res['total'] if res['total'] > 0 else 0 for freq, res in
                        response_dict.items()}

    # Sorting the fractions for plotting
    sorted_frequencies = sorted(fractions_licked.keys())
    sorted_fractions = [fractions_licked[freq] for freq in sorted_frequencies]

    # Plotting
    x_data = sorted_frequencies
    y_data = sorted_fractions

    # Define a sigmoid function
    # Fit the sigmoid function to the data
    p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
    # popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox',maxfev=10000)
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='trf')
    # Generate x values for the curve
    x_values = np.linspace(min(x_data), max(x_data), 100)
    # Generate y values from the fitted sigmoid function
    y_values = sigmoid(x_values, *popt)
    # Create the plot
    plt.figure(figsize=(20, 10))
    plt.plot(x_values, y_values * 100, label='Fit', color='black')  # Scaled by 100 for percentage
    plt.scatter(x_data, np.array(y_data) * 100, color='black')  # Scaled by 100 for percentage
    plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    plt.ylabel('Fraction excited (%)')
    # plt.title('Psychophysics Plot of Licking Response by Stimulation Frequency')
    plt.xticks(x_data, labels=[f"{freq} Hz" for freq in x_data])
    plt.yticks(np.linspace(0, 100, 11))  # Setting y ticks from 0 to 100 in steps of 10
    # plt.grid(True)
    # plt.savefig(join(save_dir, 'neurometric Plot.png'))
    plt.show()


def main_classifier(X_train, y_train, X_test, y_test):
    # Create and train the SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred, target_names=['stimulation', 'HIT'], output_dict=True)

    return accuracy


def main_classifier_nb(X_train, y_train, X_test, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def train_svm_classifier_with_bootstrapping_rest(population_vectors_stim, current_freq,
                                                 n_iterations=1000, train_frac=0.7):
    """
    Trains an SVM classifier using bootstrapping, ensuring balanced training and test sets.

    The training data is always constructed from two classes:
      - class 0 ("CR" in the code): all trials of population_vectors_stim[current_freq]
      - class 1 ("HIT"): trials drawn from population_vectors_stim[0] (0 Hz)

    For each bootstrap iteration, the two classes are each resampled to have
    exactly the same number of trials (n_total) and then split into training and test parts.
    Depending on the frequency branch, the test set is then selected:
      - If freq == current_freq: use the class-0 test data.
      - If freq == 0: use the 0 Hz test data.
      - Else: use data resampled from population_vectors_stim[freq] (labelled class 0).

    Parameters:
        population_vectors_stim (dict): {frequency: array (n_trials, n_neurons)}. Expected keys include:
            - current_freq: class-0 data for training.
            - 0: 0 Hz data.
            - Other frequencies for testing.
        current_freq (int): The key to use for the class-0 training data.
        n_iterations (int): Number of bootstrapping iterations.
        train_frac (float): Fraction of the CR (and HIT) data to use for training. The remainder is used for testing.

    Returns:
        avg_accuracy_all (list): Average accuracy for each frequency (the keys in population_vectors_stim).
        sem_all (list): Standard error of the mean of the accuracy estimates.
    """
    avg_accuracy_all = []
    sem_all = []

    # Determine the total number of trials available for the training frequency.
    # We use this number to sample from both CR and HIT data.
    n_total = len(population_vectors_stim[current_freq])
    n_train = int(np.floor(train_frac * n_total))
    n_test = n_total - n_train

    # Loop over each frequency key in the dictionary.
    for freq in population_vectors_stim.keys():
        accuracies = []
        for iteration in range(n_iterations):
            # ----- Prepare the training data (balanced across CR and HIT) -----
            # Class 0: resample from the current_freq data.
            X_cr_all = resample(population_vectors_stim[current_freq],
                                replace=False, n_samples=n_total, random_state=None)
            # Class 1: resample from the 0 Hz data.
            X_hit_all = resample(population_vectors_stim[0],
                                 replace=False, n_samples=n_total, random_state=None)

            # Split each class into training and test sets.
            X_cr_train, X_cr_test = train_test_split(X_cr_all,
                                                     train_size=n_train, test_size=n_test,
                                                     random_state=None)
            X_hit_train, X_hit_test = train_test_split(X_hit_all,
                                                       train_size=n_train, test_size=n_test,
                                                       random_state=None)

            # Combine training sets from both classes.
            X_train = np.vstack((X_cr_train, X_hit_train))
            y_train = np.concatenate((np.zeros(n_train), np.ones(n_train)))

            # ----- Prepare the test data, depending on the frequency branch -----
            if freq == current_freq:
                # For the training frequency, use its own held-out test data.
                X_test = X_cr_test
                y_test = np.zeros(n_test)
            elif freq == 0:
                # For 0 Hz, use the held-out 0 Hz test data.
                X_test = X_hit_test
                y_test = np.ones(n_test)
            else:
                # For any other frequency, use its trials as class-0 test samples.
                X_test = resample(population_vectors_stim[freq], replace=False, n_samples=n_test, random_state=None)
                y_test = np.zeros(n_test)

            # Shuffle the training and test sets.
            X_train, y_train = shuffle(X_train, y_train, random_state=None)
            X_test, y_test = shuffle(X_test, y_test, random_state=None)

            # Sanity checks: no leakage (same rows both in train and test)
            # We compare exact row bytes which is robust for float arrays that are equal.
            def _row_bytes(a):
                a2 = np.ascontiguousarray(a)
                return a2.view(np.dtype((np.void, a2.dtype.itemsize * a2.shape[1])))

            if freq == current_freq:
                common = np.intersect1d(_row_bytes(X_cr_train), _row_bytes(X_cr_test))
                if common.size > 0:
                    raise ValueError("CR train/test overlap detected")
            elif freq == 0:
                common = np.intersect1d(_row_bytes(X_hit_train), _row_bytes(X_hit_test))
                if common.size > 0:
                    raise ValueError("HIT train/test overlap detected")

            # ----- Train and evaluate the classifier -----
            accuracy = main_classifier(X_train, y_train, X_test, y_test)
            accuracies.append(accuracy)

            print(f"Iteration {iteration + 1} for frequency {freq} completed.")

        # Compute summary statistics for this frequency.
        avg_accuracy = np.mean(accuracies)
        sem = np.std(accuracies) / np.sqrt(n_iterations)
        avg_accuracy_all.append(avg_accuracy)
        sem_all.append(sem)

    return avg_accuracy_all, sem_all


def train_nb_classifier_with_bootstrapping_rest(population_vectors_stim, current_freq,
                                                n_iterations=1000, train_frac=0.7):
    """
    Mirror of train_svm_classifier_with_bootstrapping_rest, but using a GaussianNB classifier.
    See the SVM version for full documentation of the bootstrapping logic and data structure.
    """
    avg_accuracy_all = []
    sem_all = []

    n_total = len(population_vectors_stim[current_freq])
    n_train = int(np.floor(train_frac * n_total))
    n_test = n_total - n_train

    for freq in population_vectors_stim.keys():
        accuracies = []
        for iteration in range(n_iterations):
            X_cr_all = resample(population_vectors_stim[current_freq],
                                replace=False, n_samples=n_total, random_state=None)
            X_hit_all = resample(population_vectors_stim[0],
                                 replace=False, n_samples=n_total, random_state=None)

            X_cr_train, X_cr_test = train_test_split(X_cr_all,
                                                     train_size=n_train, test_size=n_test,
                                                     random_state=None)
            X_hit_train, X_hit_test = train_test_split(X_hit_all,
                                                       train_size=n_train, test_size=n_test,
                                                       random_state=None)

            X_train = np.vstack((X_cr_train, X_hit_train))
            y_train = np.concatenate((np.zeros(n_train), np.ones(n_train)))

            if freq == current_freq:
                X_test = X_cr_test
                y_test = np.zeros(n_test)
            elif freq == 0:
                X_test = X_hit_test
                y_test = np.ones(n_test)
            else:
                X_test = resample(population_vectors_stim[freq], replace=False, n_samples=n_test, random_state=None)
                y_test = np.zeros(n_test)

            X_train, y_train = shuffle(X_train, y_train, random_state=None)
            X_test, y_test = shuffle(X_test, y_test, random_state=None)

            # Sanity checks: no leakage for NB path as well
            def _row_bytes(a):
                a2 = np.ascontiguousarray(a)
                return a2.view(np.dtype((np.void, a2.dtype.itemsize * a2.shape[1])))

            if freq == current_freq:
                common = np.intersect1d(_row_bytes(X_cr_train), _row_bytes(X_cr_test))
                if common.size > 0:
                    raise ValueError("CR train/test overlap detected [NB]")
            elif freq == 0:
                common = np.intersect1d(_row_bytes(X_hit_train), _row_bytes(X_hit_test))
                if common.size > 0:
                    raise ValueError("HIT train/test overlap detected [NB]")

            accuracy = main_classifier_nb(X_train, y_train, X_test, y_test)
            accuracies.append(accuracy)

            print(f"[NB] Iteration {iteration + 1} for frequency {freq} completed.")

        avg_accuracy = np.mean(accuracies)
        sem = np.std(accuracies) / np.sqrt(n_iterations)
        avg_accuracy_all.append(avg_accuracy)
        sem_all.append(sem)

    return avg_accuracy_all, sem_all


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


# =========================
# Cell-subset SVM helpers
# =========================

ORIGINAL_ORDER = [0, 15, 10, 7.5, 5, 3, 1, 0.5]
DESIRED_ORDER = [0, 0.5, 1, 3, 5, 7.5, 10, 15]
INDEX_MAP = [ORIGINAL_ORDER.index(freq) for freq in DESIRED_ORDER]


def build_population_vectors(activity_dict_neurons):
    """
    Build population vectors:
    population_vectors[freq] -> array of shape (n_trials_for_freq, n_neurons)
    """
    first_neuron = activity_dict_neurons[0]
    population_vectors = {freq: [] for freq in first_neuron.keys()}

    for freq in first_neuron.keys():
        freq_samples = []

        for neuron_data in activity_dict_neurons:
            if freq not in neuron_data:
                continue

            neuron_samples = neuron_data[freq]

            if isinstance(neuron_samples, np.ndarray):
                neuron_samples = neuron_samples.tolist()
            elif isinstance(neuron_samples, (int, float, np.float64, np.int64)):
                neuron_samples = [neuron_samples]
            elif not isinstance(neuron_samples, list):
                raise TypeError(f"Unsupported data type: {type(neuron_samples)}")

            freq_samples.append(neuron_samples)

        if len(freq_samples) == 0:
            population_vectors[freq] = np.empty((0, 0))
        else:
            population_vectors[freq] = np.array(list(map(list, zip(*freq_samples))))

    return population_vectors


def get_valid_cell_indices(total_cells, cell_indices, label='cells'):
    """
    Keep only unique, valid indices in [0, total_cells-1].
    """
    idx = np.asarray(cell_indices).astype(int).ravel()
    idx = np.unique(idx)

    invalid = idx[(idx < 0) | (idx >= total_cells)]
    if invalid.size > 0:
        print(f"[WARNING] {label}: found {len(invalid)} invalid indices. They will be ignored.")

    idx = idx[(idx >= 0) & (idx < total_cells)]
    print(f"{label}: {len(idx)} valid cells")

    return idx


def subset_population_vectors(population_vectors, keep_indices):
    """
    Keep only selected neuron columns in every frequency matrix.
    """
    keep_indices = np.asarray(np.unique(keep_indices), dtype=int)
    subset = {}

    for freq, arr in population_vectors.items():
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        subset[freq] = arr[:, keep_indices]

    return subset


def prepare_accuracy_for_plot(avg_accuracy):
    avg_accuracy = np.asarray(avg_accuracy, dtype=float).copy()
    avg_accuracy[0] = 1 - avg_accuracy[0]
    avg_accuracy = avg_accuracy[INDEX_MAP] * 100
    return avg_accuracy


def prepare_sem_for_plot(sem_all):
    sem_all = np.asarray(sem_all, dtype=float).copy()
    sem_all[0] = 0
    sem_all = sem_all[INDEX_MAP] * 100
    return sem_all


def plot_svm_family(results_by_train_freq, save_dir, file_prefix):
    """
    One line per training frequency; x-axis is the tested frequency
    (0 Hz shown as 1 - accuracy).
    """
    freq_list = [0.5, 1, 3, 5, 7.5, 10, 15]
    colors_list = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
    equal_spacing_x_data = list(range(len(DESIRED_ORDER)))

    plt.figure(figsize=(20, 10))

    for i, train_freq in enumerate(freq_list):
        avg_accuracy = prepare_accuracy_for_plot(results_by_train_freq[train_freq]['avg_accuracy'])
        sem_all = prepare_sem_for_plot(results_by_train_freq[train_freq]['sem_all'])

        plt.plot(equal_spacing_x_data, avg_accuracy,
                 label=f'{train_freq} Hz',
                 color=colors_list[i], marker='o', linestyle='-')

        plt.errorbar(equal_spacing_x_data, avg_accuracy, yerr=sem_all,
                     color=colors_list[i], marker='o', linestyle='-', capsize=5)

    plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(equal_spacing_x_data, labels=[f"{freq} Hz" for freq in DESIRED_ORDER])
    plt.yticks(np.linspace(0, 100, 11))
    plt.legend(title='Training Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(join(save_dir, f'{file_prefix}.png'))
    plt.savefig(join(save_dir, f'{file_prefix}.pdf'))
    plt.close()


def plot_subset_comparison(all_results, no_gaussian_results, no_sigmoid_results, random_summary, save_dir):
    """
    For each training frequency, compare:
    - all cells
    - no gaussian
    - no sigmoid
    - random removal control (mean +/- sem across repeats)
    """
    freq_list = [0.5, 1, 3, 5, 7.5, 10, 15]
    x = list(range(len(DESIRED_ORDER)))

    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()

    for i, train_freq in enumerate(freq_list):
        ax = axes[i]

        all_curve = prepare_accuracy_for_plot(all_results[train_freq]['avg_accuracy'])
        all_sem = prepare_sem_for_plot(all_results[train_freq]['sem_all'])

        no_gauss_curve = prepare_accuracy_for_plot(no_gaussian_results[train_freq]['avg_accuracy'])
        no_gauss_sem = prepare_sem_for_plot(no_gaussian_results[train_freq]['sem_all'])

        no_sig_curve = prepare_accuracy_for_plot(no_sigmoid_results[train_freq]['avg_accuracy'])
        no_sig_sem = prepare_sem_for_plot(no_sigmoid_results[train_freq]['sem_all'])

        rand_curve = prepare_accuracy_for_plot(random_summary[train_freq]['avg_accuracy'])
        rand_sem = prepare_sem_for_plot(random_summary[train_freq]['sem_all'])

        ax.errorbar(x, all_curve, yerr=all_sem, label='All cells',
                    color='black', marker='o', linestyle='-', capsize=4)

        ax.errorbar(x, no_gauss_curve, yerr=no_gauss_sem, label='No Gaussian',
                    color='red', marker='o', linestyle='-', capsize=4)

        ax.errorbar(x, no_sig_curve, yerr=no_sig_sem, label='No Sigmoid',
                    color='blue', marker='o', linestyle='-', capsize=4)

        ax.errorbar(x, rand_curve, yerr=rand_sem, label='Random removal',
                    color='gray', marker='o', linestyle='--', capsize=4)

        ax.set_title(f'Train on {train_freq} Hz')
        ax.set_xlabel('Test frequency')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{freq} Hz" for freq in DESIRED_ORDER], rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

    for j in range(len(freq_list), len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(join(save_dir, 'comparison_all_vs_removed_groups.png'))
    plt.savefig(join(save_dir, 'comparison_all_vs_removed_groups.pdf'))
    plt.show()


def run_svm_suite(population_vectors, freq_range, save_dir, n_iterations=500, make_plot=True):
    """
    Run the full SVM classifier suite for one population_vectors object.
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    n_cells = next(iter(population_vectors.values())).shape[1]

    meta = {
        'n_cells': n_cells,
        'freq_range': freq_range,
        'n_iterations': n_iterations
    }

    with open(join(save_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    for train_freq in freq_range:
        print(f'Running SVM: train_freq={train_freq} Hz, n_cells={n_cells}')
        avg_accuracy, sem_all = train_svm_classifier_with_bootstrapping_rest(
            population_vectors,
            train_freq,
            n_iterations=n_iterations
        )

        results[train_freq] = {
            'avg_accuracy': np.asarray(avg_accuracy, dtype=float),
            'sem_all': np.asarray(sem_all, dtype=float)
        }

    with open(join(save_dir, 'svm_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    if make_plot:
        plot_svm_family(results, save_dir, 'svm_family')

    return results


def aggregate_random_control_results(random_results, freq_range):
    """
    Aggregate over random-removal repeats.
    random_results is a list, where each item is the output of run_svm_suite(...).
    """
    summary = {}

    for train_freq in freq_range:
        all_curves = np.stack(
            [rep_results[train_freq]['avg_accuracy'] for rep_results in random_results],
            axis=0
        )

        mean_curve = np.mean(all_curves, axis=0)

        if all_curves.shape[0] > 1:
            sem_curve = np.std(all_curves, axis=0, ddof=1) / np.sqrt(all_curves.shape[0])
        else:
            sem_curve = np.zeros(all_curves.shape[1])

        summary[train_freq] = {
            'avg_accuracy': mean_curve,
            'sem_all': sem_curve
        }

    return summary


def plot_subset_delta_from_all(all_results, no_gaussian_results, no_sigmoid_results, random_summary, save_dir):
    """
    For each training frequency, plot delta relative to all cells:
        delta = condition_accuracy - all_cells_accuracy

    Conditions:
    - No Gaussian
    - No Sigmoid
    - Random removal

    Positive values mean better than all cells.
    Negative values mean worse than all cells.
    """
    freq_list = [0.5, 1, 3, 5, 7.5, 10, 15]
    x = list(range(len(DESIRED_ORDER)))

    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()

    for i, train_freq in enumerate(freq_list):
        ax = axes[i]

        all_curve = prepare_accuracy_for_plot(all_results[train_freq]['avg_accuracy'])
        all_sem = prepare_sem_for_plot(all_results[train_freq]['sem_all'])

        no_gauss_curve = prepare_accuracy_for_plot(no_gaussian_results[train_freq]['avg_accuracy'])
        no_gauss_sem = prepare_sem_for_plot(no_gaussian_results[train_freq]['sem_all'])

        no_sig_curve = prepare_accuracy_for_plot(no_sigmoid_results[train_freq]['avg_accuracy'])
        no_sig_sem = prepare_sem_for_plot(no_sigmoid_results[train_freq]['sem_all'])

        rand_curve = prepare_accuracy_for_plot(random_summary[train_freq]['avg_accuracy'])
        rand_sem = prepare_sem_for_plot(random_summary[train_freq]['sem_all'])

        # Delta relative to all-cells curve
        delta_no_gauss = no_gauss_curve - all_curve
        delta_no_sig = no_sig_curve - all_curve
        delta_rand = rand_curve - all_curve

        # Approximate propagated SEM for visual comparison
        delta_no_gauss_sem = np.sqrt(no_gauss_sem**2 + all_sem**2)
        delta_no_sig_sem = np.sqrt(no_sig_sem**2 + all_sem**2)
        delta_rand_sem = np.sqrt(rand_sem**2 + all_sem**2)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        ax.errorbar(x, delta_no_gauss, yerr=delta_no_gauss_sem,
                    label='No Gaussian - All',
                    color='red', marker='o', linestyle='-', capsize=4)

        ax.errorbar(x, delta_no_sig, yerr=delta_no_sig_sem,
                    label='No Sigmoid - All',
                    color='blue', marker='o', linestyle='-', capsize=4)

        ax.errorbar(x, delta_rand, yerr=delta_rand_sem,
                    label='Random - All',
                    color='gray', marker='o', linestyle='--', capsize=4)

        ax.set_title(f'Train on {train_freq} Hz')
        ax.set_xlabel('Test frequency')
        ax.set_ylabel('Delta accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{freq} Hz" for freq in DESIRED_ORDER], rotation=45)
        ax.grid(alpha=0.3)

    for j in range(len(freq_list), len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(join(save_dir, 'delta_from_all_cells.png'))
    plt.savefig(join(save_dir, 'delta_from_all_cells.pdf'))
    plt.show()


if '__main__' == __name__:
    # Run once per session. The sessions used:
    #   ('mouse01', 'sess02'), ('mouse05', 'sess04')  -> data folder ML\data2
    #   ('mouse09', 'sess06'), ('mouse10', 'sess09'), ('mouse14', 'sess12')  -> data folder ML\data
    # (the data folder must be the same one used for the neurometric fit, so that the
    # Gaussian / sigmoid neuron indices refer to the same neurons)
    mouse = 'mouse05'
    day = ['sess04']

    ROOT_PATH = DATA_ROOT + r'\{}\{}\ML\data2'.format(mouse, day[0])
    save_loc = DATA_ROOT + r'\{}\{}\ML\data\processed_data\new_classifier_gassian_sigmoid_comparison'.format(mouse, day[0])
    os.makedirs(save_loc, exist_ok=True)

    n_priming_trials = 23

    # Parameters for the new comparison
    freq_range = [0.5, 1, 3, 5, 7.5, 10, 15]
    n_iterations_classifier = 500
    n_random_repeats = 100
    random_seed = 12345

    # --------------------
    # Load base data
    # --------------------
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

    # --------------------
    # Load gaussian / sigmoid indices
    # --------------------
    gaussian_sigmoid_path = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v4'.format(
        mouse, day[0]
    )
    gaussian_neuron_indices = np.load(join(gaussian_sigmoid_path, 'gaussian_neuron_indices.npy'))
    sigmoid_neuron_indices = np.load(join(gaussian_sigmoid_path, 'sigmoid_neuron_indices.npy'))

    # --------------------
    # Remove priming trials
    # --------------------
    responses = responses[n_priming_trials:]
    conditions_played = conditions_played[n_priming_trials:]
    trials_ind = [trials_ind[0][n_priming_trials:], trials_ind[1], trials_ind[2]]
    opto_on = opto_on[n_priming_trials:]
    distract_light = distract_light[n_priming_trials:]
    valve_open = valve_open[n_priming_trials:]

    # --------------------
    # Z-score each neuron
    # --------------------
    dff_norm = []
    for i in range(len(dff_4min)):
        neuron_trace = dff_4min.iloc[i]
        mean = np.nanmean(neuron_trace)
        std = np.nanstd(neuron_trace)
        dff_norm.append((neuron_trace - mean) / std)
    dff_norm = pd.DataFrame(dff_norm)

    # --------------------
    # Divide into trials
    # --------------------
    dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)
    with open(join(save_loc, 'dff_run_list.pkl'), 'wb') as f:
        pickle.dump(dff_run_list, f)

    # --------------------
    # Extract event onsets (not used further)
    # --------------------
    for i in range(len(distract_light)):
        changes = np.diff(np.concatenate(([0], distract_light[i] > 1)))
        distract_light[i] = np.where(changes == 1)[0]

        changes = np.diff(np.concatenate(([0], opto_on[i] > 1)))
        opto_on[i] = np.where(changes == 1)[0]

        changes = np.diff(np.concatenate(([0], valve_open[i] > 1)))
        valve_open[i] = np.where(changes == 1)[0]

    # --------------------
    # Build activity dict per neuron
    # --------------------
    stimu_start = pre_trial_length + 2 * rec_hz
    stimu_end = pre_trial_length + 1 * rec_hz + 4 * rec_hz

    activity_dict_neurons = []
    for i in range(len(dff_run_list)):
        neuron_activities = dff_run_list[i]

        frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
        activity_dict = {freq: [] for freq in frequency_map.values()}

        for condition, activities in zip(conditions_played, neuron_activities):
            freq = frequency_map[condition]
            stimulation_activities = np.mean(activities[stimu_start:stimu_end])
            activity_dict[freq].append(stimulation_activities)

        activity_dict_neurons.append(activity_dict)

    # --------------------
    # Build full population vectors
    # --------------------
    population_vectors_all = build_population_vectors(activity_dict_neurons)
    n_total_cells = next(iter(population_vectors_all.values())).shape[1]
    all_cell_indices = np.arange(n_total_cells)

    # --------------------
    # Validate gaussian / sigmoid indices
    # --------------------
    gaussian_neuron_indices = get_valid_cell_indices(
        n_total_cells, gaussian_neuron_indices, label='Gaussian'
    )
    sigmoid_neuron_indices = get_valid_cell_indices(
        n_total_cells, sigmoid_neuron_indices, label='Sigmoid'
    )

    gaussian_keep_indices = np.setdiff1d(all_cell_indices, gaussian_neuron_indices)
    sigmoid_keep_indices = np.setdiff1d(all_cell_indices, sigmoid_neuron_indices)

    avg_random_remove = int(round(
        (len(gaussian_neuron_indices) + len(sigmoid_neuron_indices)) / 2
    ))
    avg_random_remove = min(avg_random_remove, n_total_cells - 1)

    print(f'Total cells: {n_total_cells}')
    print(f'Gaussian cells removed: {len(gaussian_neuron_indices)}')
    print(f'Sigmoid cells removed: {len(sigmoid_neuron_indices)}')
    print(f'Random control removes each time: {avg_random_remove} cells')

    # --------------------
    # Prepare save folders
    # --------------------
    comparison_root = join(save_loc, 'svm_cell_subset_comparison')
    os.makedirs(comparison_root, exist_ok=True)

    all_dir = join(comparison_root, 'all_cells')
    no_gaussian_dir = join(comparison_root, 'no_gaussian')
    no_sigmoid_dir = join(comparison_root, 'no_sigmoid')
    random_dir = join(comparison_root, 'random_remove_avg_group_size')

    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(no_gaussian_dir, exist_ok=True)
    os.makedirs(no_sigmoid_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)

    # --------------------
    # Run SVM on all cells
    # --------------------
    all_results = run_svm_suite(
        population_vectors_all,
        freq_range=freq_range,
        save_dir=all_dir,
        n_iterations=n_iterations_classifier,
        make_plot=True
    )

    # --------------------
    # Run SVM without Gaussian cells
    # --------------------
    population_vectors_no_gaussian = subset_population_vectors(
        population_vectors_all,
        gaussian_keep_indices
    )

    no_gaussian_results = run_svm_suite(
        population_vectors_no_gaussian,
        freq_range=freq_range,
        save_dir=no_gaussian_dir,
        n_iterations=n_iterations_classifier,
        make_plot=True
    )

    # --------------------
    # Run SVM without Sigmoid cells
    # --------------------
    population_vectors_no_sigmoid = subset_population_vectors(
        population_vectors_all,
        sigmoid_keep_indices
    )

    no_sigmoid_results = run_svm_suite(
        population_vectors_no_sigmoid,
        freq_range=freq_range,
        save_dir=no_sigmoid_dir,
        n_iterations=n_iterations_classifier,
        make_plot=True
    )

    # --------------------
    # Random control:
    # remove the average number of cells, repeat 100 times
    # --------------------
    rng = np.random.default_rng(random_seed)
    random_results_all_repeats = []
    random_removed_indices_all_repeats = []

    for rep in range(n_random_repeats):
        print(f'Random control repeat {rep + 1} / {n_random_repeats}')

        removed_idx = np.sort(
            rng.choice(all_cell_indices, size=avg_random_remove, replace=False)
        )
        keep_idx = np.setdiff1d(all_cell_indices, removed_idx)

        population_vectors_random = subset_population_vectors(
            population_vectors_all,
            keep_idx
        )

        rep_dir = join(random_dir, f'repeat_{rep:03d}')
        os.makedirs(rep_dir, exist_ok=True)

        rep_results = run_svm_suite(
            population_vectors_random,
            freq_range=freq_range,
            save_dir=rep_dir,
            n_iterations=n_iterations_classifier,
            make_plot=False
        )

        random_results_all_repeats.append(rep_results)
        random_removed_indices_all_repeats.append(removed_idx)

    # Save all random repeats
    with open(join(random_dir, 'random_removed_indices_all_repeats.pkl'), 'wb') as f:
        pickle.dump(random_removed_indices_all_repeats, f)

    with open(join(random_dir, 'random_results_all_repeats.pkl'), 'wb') as f:
        pickle.dump(random_results_all_repeats, f)

    # Aggregate random control over repeats
    random_summary = aggregate_random_control_results(
        random_results_all_repeats,
        freq_range=freq_range
    )

    with open(join(random_dir, 'random_summary.pkl'), 'wb') as f:
        pickle.dump(random_summary, f)

    # Plot family plot for the random control mean
    plot_svm_family(random_summary, random_dir, 'svm_random_control_mean')

    # --------------------
    # Final comparison plot
    # --------------------
    plot_subset_comparison(
        all_results=all_results,
        no_gaussian_results=no_gaussian_results,
        no_sigmoid_results=no_sigmoid_results,
        random_summary=random_summary,
        save_dir=comparison_root
    )

    # Save combined summary
    combined_summary = {
        'all_cells': all_results,
        'no_gaussian': no_gaussian_results,
        'no_sigmoid': no_sigmoid_results,
        'random_summary': random_summary,
        'gaussian_neuron_indices': gaussian_neuron_indices,
        'sigmoid_neuron_indices': sigmoid_neuron_indices,
        'avg_random_remove': avg_random_remove
    }

    with open(join(comparison_root, 'combined_summary.pkl'), 'wb') as f:
        pickle.dump(combined_summary, f)

    # --------------------
    # Delta from all-cells plot
    # --------------------
    plot_subset_delta_from_all(
        all_results=all_results,
        no_gaussian_results=no_gaussian_results,
        no_sigmoid_results=no_sigmoid_results,
        random_summary=random_summary,
        save_dir=comparison_root
    )

    # Delta summary (kept in memory only; it is not saved)
    delta_summary = {
        'delta_no_gaussian': {
            'avg_accuracy': {freq: no_gaussian_results[freq]['avg_accuracy'] - all_results[freq]['avg_accuracy']
                             for freq in freq_range},
            'sem_all': {freq: np.sqrt(no_gaussian_results[freq]['sem_all']**2 + all_results[freq]['sem_all']**2)
                        for freq in freq_range}
        },
        'delta_no_sigmoid': {
            'avg_accuracy': {freq: no_sigmoid_results[freq]['avg_accuracy'] - all_results[freq]['avg_accuracy']
                             for freq in freq_range},
            'sem_all': {freq: np.sqrt(no_sigmoid_results[freq]['sem_all']**2 + all_results[freq]['sem_all']**2)
                        for freq in freq_range}
        },
        'delta_random': {
            'avg_accuracy': {freq: random_summary[freq]['avg_accuracy'] - all_results[freq]['avg_accuracy']
                             for freq in freq_range},
            'sem_all': {freq: np.sqrt(random_summary[freq]['sem_all']**2 + all_results[freq]['sem_all']**2)
                        for freq in freq_range}
        }
    }