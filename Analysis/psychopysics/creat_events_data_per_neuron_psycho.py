"""
Build per-neuron, per-trial activity data for the psychophysics session (opto stimulation
at different frequencies).

Steps (one imaging session):
    1. Remove the first n_priming_trials (priming) trials.
    2. Z-score each neuron's dF/F trace across the whole session.
    3. Divide the continuous trace into trials, prepending the 4 s preceding each trial.
    4. Split trials by behavioral outcome; light trials = CR + FA trials.
    5. Build event windows (light, reward, licks) and the event data of all light trials.
    6. Average activity in [opto on + 1 s, opto off) for every light trial.
The frequency-specific and trial-history versions are kept commented out at the end.
Outputs are saved as pickles in `save_loc`.
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
# Condition 6: 1 s distractor light; 4 s delay; 2 s EB odor; 2 s reward window for operant licking.
con2 = 6
pre_trial_length = 4 * rec_hz  # frames prepended to each trial
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz  # opto stimulation length (frames)


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
    """Returns the length of the third-longest trial (used as the x-axis limit)."""
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
    """Marks distractor light, opto stimulation and odor on an opto-trial plot."""
    # distractor light
    axes.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1, label='Distractor light')
    axes.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    # opto
    axes.axvline(x=pre_trial_length + 1 * rec_hz, color='yellow', linestyle='--', linewidth=1, label='Opto stim')
    axes.axvline(x=pre_trial_length + 1 * rec_hz + 4 * rec_hz, color='yellow', linestyle='--', linewidth=1)
    # odor
    axes.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1, label='Odor')
    axes.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    # add legend
    axes.legend(loc='upper right')


def add_axvline(axes, rewards, pre_trial_length, rec_hz, position):
    """Marks distractor light and odor; on the top-left (Hit) heatmap also marks reward times."""
    # distractor light
    axes.axvline(x=pre_trial_length, color='darkorange', linestyle='--', linewidth=1)
    axes.axvline(x=pre_trial_length + 9 * rec_hz, color='darkorange', linestyle='--', linewidth=1)
    # odor
    axes.axvline(x=pre_trial_length + 5 * rec_hz, color='blue', linestyle='--', linewidth=1)
    axes.axvline(x=pre_trial_length + 5 * rec_hz + 2 * rec_hz, color='blue', linestyle='--', linewidth=1)
    # asterisk at the reward time of each trial
    if position == (0, 0):
        for index, reward in enumerate(rewards):
            if len(reward) > 0:
                axes.text(reward + pre_trial_length, index + 0.5, '*', color='white', ha='center', va='center',
                          fontsize=12)


def create_figures(dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, rewards_times, ROOT_PATH, pre_trial_length,
                   rec_hz, folder_name, num_neurons):
    """Saves, for each neuron, trial heatmaps and mean +/- SEM traces for the four outcomes."""
    folder = join(ROOT_PATH, folder_name)
    os.makedirs(folder, exist_ok=True)

    for num in range(num_neurons):
        xlim = 20 * rec_hz  # x_lim_calc(dff_hit[num])
        TICKS_LABELS = np.arange(0, xlim, step=rec_hz * 5) / rec_hz

        # Shared color scale across the four conditions
        # percentile_min_value = 40 # 5th percentile for the minimum
        # percentile_max_value = 60  # 95th percentile for the maximum
        data_min = min(
            df.min().min() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]]) + 0.3
        data_max = max(
            df.max().max() for df in [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]]) - 0.2
        # data_min = min(np.percentile(df, percentile_min_value) for df in
        #                           [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        # data_max = max(np.percentile(df, percentile_max_value) for df in
        #                           [dff_hit[num], dff_miss[num], dff_correct_reject[num], dff_false_alarm[num]])
        # norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)
        # Trial heatmaps, one subplot per condition
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 24),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

        # sns.heatmap(dff_hit[num], ax=axes[0, 0], norm=norm, cmap="coolwarm")
        # sns.heatmap(dff_miss[num], ax=axes[1, 0], norm=norm, cmap="coolwarm")
        # sns.heatmap(dff_correct_reject[num], ax=axes[0, 1], norm=norm, cmap="coolwarm")
        # sns.heatmap(dff_false_alarm[num], ax=axes[1, 1], norm=norm, cmap="coolwarm")

        sns.heatmap(dff_hit[num], ax=axes[0, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_miss[num], ax=axes[1, 0], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_correct_reject[num], ax=axes[0, 1], vmin=data_min, vmax=data_max)
        sns.heatmap(dff_false_alarm[num], ax=axes[1, 1], vmin=data_min, vmax=data_max)

        # Task event markers
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
            ax.set_xticks(np.arange(0, xlim, step=rec_hz * 5))
            ax.set_xticklabels(TICKS_LABELS)
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('Trials')

        # Save the heatmap figure (png + pdf)
        plt.savefig(join(folder, f'neuron_{num}_combined.png'))
        folder2 = join(folder, 'pdf_files')
        os.makedirs(folder2, exist_ok=True)
        plt.savefig(join(folder2, f'neuron_{num}_combined.pdf'))

        # Mean activity +/- SEM, one subplot per condition
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 24), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        conditions = [dff_hit, dff_miss, dff_correct_reject, dff_false_alarm]
        titles = ['Hit', 'Miss', 'Correct Reject', 'False Alarm']

        data_min = np.nanmin([np.nanmin((np.nanmean(conditions[i][num], axis=0)) - np.nanmin(
            np.nanstd(conditions[i][num], axis=0) / np.sqrt(np.count_nonzero(~np.isnan(conditions[i][num]))), axis=0))
                              for i in range(4)])
        data_min = data_min
        data_max = np.nanmax([np.nanmax(np.nanmean(conditions[i][num], axis=0) + np.nanmax(
            np.nanstd(conditions[i][num], axis=0) / np.sqrt(np.count_nonzero(~np.isnan(conditions[i][num]))), axis=0))
                              for i in range(4)])
        data_max = data_max
        for i, ax in enumerate(axes.flat):
            mean_activity = np.nanmean(conditions[i][num], axis=0)
            sem = np.nanstd(conditions[i][num], axis=0) / np.sqrt(
                np.count_nonzero(~np.isnan(conditions[i][num]), axis=0))
            time_points = np.arange(len(mean_activity))
            ax.plot(time_points, mean_activity, color='black')
            ax.fill_between(time_points, mean_activity - sem, mean_activity + sem, alpha=0.1, color='lightgray')
            ax.set_title(titles[i])
            ax.set_xlim(0, xlim)
            ax.set_xticks(np.arange(0, xlim, step=rec_hz * 5))
            ax.set_xticklabels(TICKS_LABELS)
            ax.set_ylim(data_min, data_max)
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('Mean Activity')

            # Task event markers (Hit / Miss: no-opto trials; CR / FA: opto trials)
            if i in [0, 1]:
                add_axvline(ax, rewards_times, pre_trial_length, rec_hz, position=(i, 0))
            else:
                add_axvline_opto(ax, pre_trial_length, rec_hz)

        folder3 = join(folder, 'mean_activity_plots')
        os.makedirs(folder3, exist_ok=True)
        plt.savefig(join(folder3, f'neuron_{num}_mean_activity_combined.png'))
        folder4 = join(folder3, 'pdf_files')
        os.makedirs(folder4, exist_ok=True)
        # plt.savefig(join(folder4, f'neuron_{num}_combined.pdf'))
        plt.close(fig)


def divide_to_trials(f_signal, trials_ind, lengths, len_pre_trial):
    """
    Divides each neuron's continuous signal into trials and prepends to each
    trial the `len_pre_trial` frames that precede it.

    Parameters:
    - f_signal (pd.DataFrame): neurons x frames; runs are concatenated along the frames.
    - trials_ind (list): for each run, the trial-onset frames relative to the run start.
    - lengths (list): number of frames in each run.
    - len_pre_trial (int): number of pre-trial frames to prepend to each trial.

    Returns:
    - f_run_list_final (list): f_run_list_final[neuron][trial] = pre-trial + trial frames.
    - pre_run_list: the segment before the first trial of each run (of the last neuron).
    """
    f_run_list_final = []
    for i in range(0, len(f_signal)):  # for each neuron
        f_run_list = []
        pre_run_list = []
        f_run_list_all = []
        start_ind = 0
        for run_ind in range(0, len(lengths)):
            end_ind = start_ind + lengths[run_ind]
            f_run = np.split(f_signal.iloc[i][start_ind:end_ind], trials_ind[run_ind], axis=0)
            f_run_list.append(f_run[1:])
            pre_run_list.append(f_run[0])
            start_ind = end_ind

        # add the pre-trial to each trial
        for run in range(0, len(f_run_list)):
            for j in range(0, len(f_run_list[run])):
                if j == 0:  # first trial of each run
                    pre_trial = pre_run_list[run][len(pre_run_list[run]) - len_pre_trial:]
                else:  # the rest of the trials
                    pre_trial = f_run_list[run][j - 1][len(f_run_list[run][j - 1]) - len_pre_trial:]

                f_run_list[run][j] = (list(pre_trial) + list(f_run_list[run][j]))
            f_run_list_all.extend(f_run_list[run])
        f_run_list_final.append(f_run_list_all)

    return f_run_list_final, pre_run_list


def concatenate_data(average_data):
    """Concatenates, for each neuron, its data from all conditions into one row DataFrame."""
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


# Earlier version of min_max_normalize (not used):
# def min_max_normalize(concat_matrix, ave_data):
#     minmax_df = []
#     for i in range(len(concat_matrix)):
#         min = np.min(concat_matrix[i], axis=1)[0]
#         max = np.max(concat_matrix[i], axis=1)[0]
#         diff = max-min
#         # For each neuron, subtract the mean response from its response in each data point.
#         subtract_df = ave_data[i] - min
#         # For each neuron, divide the resulting value by the range.
#         subtract_df /= diff
#         minmax_df.append(subtract_df)
#
#     return minmax_df


def find_min_max(data):
    """Returns (max, min) of the data, padded by 0.1, for color-scale limits."""
    max = np.nanmax(data) + 0.1
    min = np.nanmin(data) - 0.1
    return (max, min)


def plot_normalized_data(data, sort_ind, x_lim, data_max, data_min, PATH, filename):
    """Heatmap of all neurons (rows ordered by sort_ind)."""
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


def calculate_dff(f_run_list, pre_trial_length, f0_window, ROOT_PATH):
    """dF/F per neuron, with F0 = mean of the f0_window frames before pre_trial_length."""
    df_final_all = []

    for df_neuron in f_run_list:
        df_final = []
        trial_activity = np.asarray(df_neuron)
        f0_trials = np.mean(trial_activity[pre_trial_length - f0_window:pre_trial_length])
        df_final.append((trial_activity - f0_trials) / f0_trials)
        df_final = pd.DataFrame(df_final)
        df_final_all.append(df_final)

    save_location = join(ROOT_PATH, 'dff(2 sec f0)_glu_run1.npy')
    np.save(save_location, df_final_all)

    return df_final_all


def min_max_normalize(cells_data):
    """Scales each neuron to its 1st-99th percentile range."""
    minmax_df = []
    for i in range(len(cells_data)):
        # 1st and 99th percentile of each neuron
        min = np.percentile(cells_data.iloc[i], 1)
        max = np.percentile(cells_data.iloc[i], 99)
        diff = max - min
        # subtract the 1st percentile and divide by the range
        subtract_df = cells_data.iloc[i] - min
        subtract_df /= diff
        minmax_df.append(subtract_df)
    minmax_df = np.asarray(minmax_df)
    return minmax_df

def norm_to_max(cells_data):
    """Divides each neuron by its 99th percentile."""
    norm_df = []
    for i in range(len(cells_data)):
        max = np.percentile(cells_data.iloc[i], 99)
        norm_df.append(cells_data.iloc[i] / max)
    norm_df = np.asarray(norm_df)
    return norm_df

def baseline_subtract_neurons(norm_data, pre_trial_length, baseline_time):
    """Subtracts from each trace the mean of the baseline_time frames before pre_trial_length."""
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

def reassemble_trials_remove_pretrial(dff_run_list_sub, run_len, len_pre_trial):
    """
    Reassembles a continuous array (n_neurons x total_frames) by *removing*
    the first 'len_pre_trial' frames from each trial (those extra pre-trial frames),
    so that the final flattened shape matches sum(run_len).

    Parameters
    ----------
    dff_run_list_sub : list of length n_neurons
        dff_run_list_sub[i] is a list of #trials, each trial is a 1D array
        containing (len_pre_trial + the real trial) frames.
    run_len : list or array of run lengths (frames).
        sum(run_len) = total frames in the original raw data.
    len_pre_trial : int
        How many frames were prepended to each trial for "pre-trial" context.

    Returns
    -------
    df_reassembled : pd.DataFrame
        shape (n_neurons, sum(run_len)) with the original (non-expanded)
        frames placed end-to-end for each neuron.
    """
    n_neurons = len(dff_run_list_sub)
    total_len = sum(run_len)

    # Prepare an array to hold the final flattened data
    reassembled = np.full((n_neurons, total_len), np.nan)

    # copy so that the input is not modified
    dff_copy = copy.deepcopy(dff_run_list_sub)

    for i in range(n_neurons):
        # This is a list of trial arrays for neuron i
        trials = dff_copy[i]

        pointer = 0
        for t_idx in range(len(trials)):
            trial_arr = trials[t_idx]

            # 1) Remove the first len_pre_trial frames
            #    as they were the "pre-trial" chunk
            if len(trial_arr) > len_pre_trial:
                # slice off pre-trial
                trimmed = trial_arr[len_pre_trial:]
            else:
                # trial is shorter than len_pre_trial
                trimmed = np.array([])

            length = len(trimmed)

            # 2) Place the trimmed array into the 'reassembled' output
            reassembled[i, pointer : pointer + length] = trimmed
            pointer += length

        # If pointer < total_len, the remainder stays as NaN
        # If pointer > total_len => mismatch in total frames

    # Convert to DataFrame
    df_reassembled = pd.DataFrame(reassembled)
    return df_reassembled

def load_cellreg_mat(cellreg_path):
    """Loads the cell-to-index map and cell scores from a CellReg .mat file."""
    with h5py.File(cellreg_path, 'r') as file:
        cell_registered_struct = file['cell_registered_struct']
        cell_indexes = np.array(cell_registered_struct['cell_to_index_map'][()]).T
        cell_indexes -= 1
        cell_indexes[cell_indexes == -1] = np.nan
        scores = np.array(cell_registered_struct['cell_scores'][()]).T
        # check that each column in cell_to_index contains all indexes from zero to the number of cells in the column
        _, num_cols = cell_indexes.shape
        # Iterate over each column and check for consecutive integers starting from 0.
        for col in range(num_cols):
            column_data = np.sort(cell_indexes[:, col])  # Sort the column data
            # Generate a range of numbers from zero to the maximum number in the column
            max_val = np.nanmax(column_data)
            expected_range = np.arange(max_val + 1)
            # Find the missing numbers by using set operations
            missing_numbers = set(expected_range) - set(column_data)

            if column_data[0] != 0 or not missing_numbers == set():
                print(f"Column {col} does not contain consecutive integers starting from 0.")
                print(f"Missing numbers in column {col}: {sorted(missing_numbers)}")
            else:
                print(f"Column {col} contains consecutive integers starting from 0.")
    return cell_indexes, scores


def cells_all_days(cell_to_index, cell_scores, corr_threshold):
    """
    Find cells that are present in all days and have a correlation score above the threshold.

    Parameters:
    - cell_to_index: 2D numpy array of shape (n_cells, n_days), containing cell indices per day.
    - cell_scores: 2D numpy array of shape (n_cells, n_days), containing correlation scores per cell per day.
    - corr_threshold: Float, the minimum correlation score threshold.

    Returns:
    - valid_cells: 2D numpy array of the filtered cell indices.
    """

    # Ensure inputs are numpy arrays
    cell_to_index = np.asarray(cell_to_index)
    cell_scores = np.asarray(cell_scores)

    # Filter cells present in all days
    cells_present_all_days = ~np.isnan(cell_to_index).any(axis=1)
    # Filter cells with scores above the threshold in all days
    cells_above_threshold = (cell_scores > corr_threshold).all(axis=1)
    # Combine both conditions
    valid_cells_mask = cells_present_all_days & cells_above_threshold

    # Extract valid cells
    valid_cells = cell_to_index[valid_cells_mask]

    return valid_cells


def divide_to_conditions(dff_sub, correct_reject, false_alarm, hit, miss, light_ind):
    """
    Splits each neuron's trials by behavioral outcome (and light trials).
    Each returned list has one DataFrame (trials x frames) per neuron.
    """
    data_correct_reject = []
    data_false_alarm = []
    data_hit = []
    data_miss = []
    data_light = []
    for neuron in (dff_sub):
        neuron = pd.DataFrame(neuron)
        neuron_correct_reject = []
        neuron_false_alarm = []
        neuron_hit = []
        neuron_miss = []
        neuron_light = []
        neuron_correct_reject.append(neuron.iloc[correct_reject, :])
        neuron_false_alarm.append(neuron.iloc[false_alarm, :])
        neuron_hit.append(neuron.iloc[hit, :])
        neuron_miss.append(neuron.iloc[miss, :])
        neuron_light.append(neuron.iloc[light_ind, :])
        data_correct_reject.append(neuron_correct_reject[0])
        data_false_alarm.append(neuron_false_alarm[0])
        data_hit.append(neuron_hit[0])
        data_miss.append(neuron_miss[0])
        data_light.append(neuron_light[0])
    return data_correct_reject, data_false_alarm, data_hit, data_miss, data_light


def identify_event_responsive_neurons(dff,
                                      events_info,
                                      alpha=0.05,
                                      nan_policy='omit'):
    """
    Identifies neurons that show statistically significant responses
    for each event in the dictionary. Compares mean signal in the
    [start->stop) window vs. the [baseline_start->baseline_stop) window
    for each trial.

    Parameters
    ----------
    dff : list of pd.DataFrame
        dff[i] is a DataFrame (trials x frames) for neuron i.
    events_info : dict
        Built by 'build_events_info(...)'. For each event key, has:
            {
              'start': array of shape (n_trials,),
              'stop':  array of shape (n_trials,),
              'baseline_start': array of shape (n_trials,),
              'baseline_stop':  array of shape (n_trials,)
            }
    alpha : float, optional
        p-value threshold for significance.
    nan_policy : {'omit', 'raise', 'propagate'}, optional
        Behavior for NaNs in ttest_rel.

    Returns
    -------
    significant_neurons : dict
        event_type -> array of neuron indices that passed the significance test
    """

    significant_neurons = {}

    # For each event (e.g., light_on, light_off, reward, lick_onset, lick_offset)
    for event_name, event_dict in events_info.items():
        start_arr = event_dict['start']
        stop_arr = event_dict['stop']
        base_start_arr = event_dict['baseline_start']
        base_stop_arr = event_dict['baseline_stop']

        # neurons significant for this event
        responsive_mask = np.zeros(len(dff), dtype=bool)

        # Iterate over neurons
        for neuron_idx, neuron_df in enumerate(dff):
            # neuron_df is shape (n_trials x n_frames)
            event_vals = []
            baseline_vals = []

            # For each trial in this neuron
            for trial_i in range(neuron_df.shape[0]):
                # --- EVENT window ---
                start_val = start_arr[trial_i]
                stop_val = stop_arr[trial_i]

                # If either is NaN, that means no event window for this trial
                if (np.isnan(start_val)) or (np.isnan(stop_val)):
                    event_vals.append(np.nan)
                else:
                    st = int(max(0, min(start_val, neuron_df.shape[1])))
                    sp = int(max(0, min(stop_val, neuron_df.shape[1])))
                    if sp > st:
                        event_vals.append(neuron_df.iloc[trial_i, st:sp].mean())
                    else:
                        event_vals.append(np.nan)

                # --- BASELINE window ---
                base_start_val = base_start_arr[trial_i]
                base_stop_val = base_stop_arr[trial_i]

                if (np.isnan(base_start_val)) or (np.isnan(base_stop_val)):
                    baseline_vals.append(np.nan)
                else:
                    bs = int(max(0, min(base_start_val, neuron_df.shape[1])))
                    bp = int(max(0, min(base_stop_val, neuron_df.shape[1])))
                    if bp > bs:
                        baseline_vals.append(neuron_df.iloc[trial_i, bs:bp].mean())
                    else:
                        baseline_vals.append(np.nan)

            # Convert to Series and drop NaN trials
            event_series = pd.Series(event_vals).dropna()
            base_series = pd.Series(baseline_vals).dropna()

            # Perform the t-test only if there's at least 5 data points in each
            if (len(event_series) >= 5) and (len(base_series) >= 5):
                t_stat, p_val = ttest_rel(event_series, base_series, nan_policy=nan_policy)
                if p_val < alpha:
                    responsive_mask[neuron_idx] = True

        # Gather which neurons were significant for this event
        significant_neurons[event_name] = np.where(responsive_mask)[0]

    return significant_neurons


def identify_event_increase_decrease_doubletest(dff,
                                                events_info,
                                                alpha=0.05,
                                                nan_policy='omit'):  # ,test_method='ttest'
    """
    Identifies neurons with significantly increased or decreased activity
    for each event, using two paired t-tests per event:
      1) Event window vs. Baseline window
      2) Event window vs. Preceding window (1 sec prior, etc.)

    A neuron is deemed "significant" if p < alpha for BOTH tests,
    and the direction is consistent:
      - 'increase' if (mean_event > mean_baseline) and (mean_event > mean_preceding)
      - 'decrease' if (mean_event < mean_baseline) and (mean_event < mean_preceding)

    Parameters
    ----------
    dff : list of pd.DataFrame
        dff[i] is a DataFrame with shape (n_trials x n_frames) for neuron i.
        Each row is one trial's fluorescence/df/f.

    events_info : dict
        For each event_name, has arrays of length n_trials:
            {
              'start':           array([...]),
              'stop':            array([...]),
              'baseline_start':  array([...]),
              'baseline_stop':   array([...]),
              'preceding_start': array([...]),
              'preceding_stop':  array([...])
            }

    alpha : float
        Significance threshold for both tests.

    nan_policy : {'omit', 'raise', 'propagate'}
        Behavior for NaNs in ttest_rel.

    Returns
    -------
    significant_neurons : dict
        event_name -> {
          'increase': np.array([...]),  # neuron indices that significantly increased
          'decrease': np.array([...])   # neuron indices that significantly decreased
        }
    """

    significant_neurons = {}

    for event_name, event_dict in events_info.items():
        # Unpack the relevant arrays
        start_arr = event_dict['start']
        stop_arr = event_dict['stop']
        base_start_arr = event_dict['baseline_start']
        base_stop_arr = event_dict['baseline_stop']
        prec_start_arr = event_dict['preceding_start']
        prec_stop_arr = event_dict['preceding_stop']

        n_neurons = len(dff)
        increase_mask = np.zeros(n_neurons, dtype=bool)
        decrease_mask = np.zeros(n_neurons, dtype=bool)

        for neuron_idx, neuron_df in enumerate(dff):
            # trial-wise means of the event, baseline and preceding windows
            event_vals = []
            baseline_vals = []
            preceding_vals = []

            n_frames = neuron_df.shape[1]

            # Collect means for each trial
            for trial_i in range(neuron_df.shape[0]):
                st = start_arr[trial_i]
                sp = stop_arr[trial_i]
                bs = base_start_arr[trial_i]
                bp = base_stop_arr[trial_i]
                ps = prec_start_arr[trial_i]
                pp = prec_stop_arr[trial_i]

                # If any are NaN => skip
                if any(np.isnan([st, sp, bs, bp, ps, pp])):
                    event_vals.append(np.nan)
                    baseline_vals.append(np.nan)
                    preceding_vals.append(np.nan)
                    continue

                # Clip
                st = int(np.clip(st, 0, n_frames))
                sp = int(np.clip(sp, 0, n_frames))
                bs = int(np.clip(bs, 0, n_frames))
                bp = int(np.clip(bp, 0, n_frames))
                ps = int(np.clip(ps, 0, n_frames))
                pp = int(np.clip(pp, 0, n_frames))

                # Ensure valid windows
                if sp <= st or bp <= bs or pp <= ps:
                    event_vals.append(np.nan)
                    baseline_vals.append(np.nan)
                    preceding_vals.append(np.nan)
                    continue

                # Means
                ev_mean = neuron_df.iloc[trial_i, st:sp].mean()
                base_mean = neuron_df.iloc[trial_i, bs:bp].mean()
                prec_mean = neuron_df.iloc[trial_i, ps:pp].mean()

                event_vals.append(ev_mean)
                baseline_vals.append(base_mean)
                preceding_vals.append(prec_mean)

            # Convert to Series, drop NaNs
            event_series = pd.Series(event_vals).dropna()
            base_series = pd.Series(baseline_vals).dropna()
            prec_series = pd.Series(preceding_vals).dropna()

            # Paired tests require the same trials in all three windows
            if (len(event_series) != len(base_series)) or (len(event_series) != len(prec_series)):
                raise ValueError(f"Neuron {neuron_idx} has mismatched trial counts for event '{event_name}'.")

            if len(event_series) < 5:
                # Not enough data for a paired test
                continue

            # Test 1: event vs. baseline
            # p_val1, direction1 = paired_test(event_series, base_series, test_method)
            t1_stat, p_val1 = ttest_rel(event_series, base_series, nan_policy=nan_policy)
            # Test 2: event vs. preceding
            t2_stat, p_val2 = ttest_rel(event_series, prec_series, nan_policy=nan_policy)
            # p_val2, direction2 = paired_test(event_series, prec_series, test_method)
            # if (p_val1 < alpha):
            #     # Both significant => check the direction
            #     if direction1 == 'increase':
            #         increase_mask[neuron_idx] = True
            #     elif direction1 == 'decrease':
            #         decrease_mask[neuron_idx] = True
            if (p_val1 < alpha) and (p_val2 < alpha):
                # Both significant => check the direction
                # if direction1 == direction2:
                #     if direction1 == 'increase':
                #         increase_mask[neuron_idx] = True
                #     elif direction1 == 'decrease':
                #         decrease_mask[neuron_idx] = True
                mean_event = event_series.mean()
                mean_base = base_series.mean()
                mean_prec = prec_series.mean()

                # "Increase" if E > B and E > P
                if (mean_event > mean_base) and (mean_event > mean_prec):
                    increase_mask[neuron_idx] = True
                # "Decrease" if E < B and E < P
                elif (mean_event < mean_base) and (mean_event < mean_prec):
                    decrease_mask[neuron_idx] = True
                # Conflicting directions (e.g. E > B but E < P) are not labeled.

        # Store the results
        significant_neurons[event_name] = {
            'increase': np.where(increase_mask)[0],
            'decrease': np.where(decrease_mask)[0],
        }

    return significant_neurons


def identify_light_on_two_windows(
        dff_list,
        light_on_ind,
        light_off_ind,
        baseline_start,
        baseline_stop,
        rec_hz,
        alpha=0.05,
        test_method='ttest',  # or 'wilcoxon'
        do_bonferroni=True
):
    """
    Identify neurons that have a significant change in either of two windows:
      1) First 2s after light_on
      2) Last 2s before light_off

    Compares each window's mean to the baseline [baseline_start : baseline_stop]
    across trials for each neuron. If either window is significant (after
    Bonferroni correction), the neuron is labeled 'increase' or 'decrease'.

    Parameters
    ----------
    dff_list : list of pd.DataFrame
        dff_list[i] is a DataFrame for neuron i, shape = (#trials x #frames).
    light_on_ind : array of shape (#trials,)
        The frame index at which the light turns on (for each trial).
    light_off_ind : array of shape (#trials,)
        The frame index at which the light turns off (for each trial).
    baseline_start : int
        The start index of the baseline window (same for all trials).
    baseline_stop : int
        The stop index of the baseline window (same for all trials).
    rec_hz : int
        The number of frames per second in the data.
    alpha : float
        Significance threshold.
    test_method : {'ttest', 'wilcoxon'}
        Which paired test to run.
    do_bonferroni : bool
        If True, apply Bonferroni correction for the two comparisons.

    Returns
    -------
    results : dict
      results['light_on'] = {
        'increase': np.array([...]),
        'decrease': np.array([...])
      }
      where each array is the set of neuron indices that showed a
      significant increase or decrease (for either window).
    """
    n_neurons = len(dff_list)
    n_trials = dff_list[0].shape[0]
    # Trials without a detected opto on/off get the nominal stimulation window
    light_on_ind = np.asarray(light_on_ind)
    light_off_ind = np.asarray(light_off_ind)
    start_opto = pre_trial_length + 1 * rec_hz  # 1 sec after the trial starts
    end_opto = pre_trial_length + 5 * rec_hz  # 5 sec after the trial starts
    light_on_ind[np.isnan(light_on_ind)] = start_opto
    light_off_ind[np.isnan(light_off_ind)] = end_opto
    # Bonferroni correction across the two tests
    if do_bonferroni:
        alpha_corr = alpha / 2.0
    else:
        alpha_corr = alpha

    increase_mask = np.zeros(n_neurons, dtype=bool)
    decrease_mask = np.zeros(n_neurons, dtype=bool)

    # Iterate over neurons
    for neuron_idx in range(n_neurons):
        neuron_df = dff_list[neuron_idx]  # shape (#trials x #frames)

        # Collect trial-wise means for Window A & Window B, as well as baseline
        wA_vals = []
        wB_vals = []
        base_vals = []

        for trial_i in range(n_trials):
            on_i = light_on_ind[trial_i]
            off_i = light_off_ind[trial_i]

            # If on_i or off_i is NaN or invalid, skip
            if np.isnan(on_i) or np.isnan(off_i):
                wA_vals.append(np.nan)
                wB_vals.append(np.nan)
                base_vals.append(np.nan)
                continue

            on_i = int(on_i)
            off_i = int(off_i)

            # Make sure off_i > on_i for the indexing to make sense
            if off_i <= on_i:
                wA_vals.append(np.nan)
                wB_vals.append(np.nan)
                base_vals.append(np.nan)
                continue

            # Baseline window
            bs = int(baseline_start)
            bp = int(baseline_stop)
            base_mean = neuron_df.iloc[trial_i, bs:bp].mean()

            # Window A: [light_on, light_on + 2*rec_hz)
            A_start = on_i
            A_stop = on_i + 2 * rec_hz

            # Clipping to the valid range (not used):
            # A_start = max(0, min(A_start, neuron_df.shape[1]))
            # A_stop = max(0, min(A_stop, neuron_df.shape[1]))

            if A_stop > A_start:
                wA_mean = neuron_df.iloc[trial_i, A_start:A_stop].mean()
            else:
                wA_mean = np.nan

            # Window B: [light_off - 2*rec_hz, light_off)
            B_stop = off_i
            B_start = off_i - 2 * rec_hz

            # B_start = max(0, min(B_start, neuron_df.shape[1]))
            # B_stop = max(0, min(B_stop, neuron_df.shape[1]))

            if B_stop > B_start:
                wB_mean = neuron_df.iloc[trial_i, B_start:B_stop].mean()
            else:
                wB_mean = np.nan

            wA_vals.append(wA_mean)
            wB_vals.append(wB_mean)
            base_vals.append(base_mean)

        # Convert to pandas Series and drop NaNs
        wA_series = pd.Series(wA_vals).dropna()
        wB_series = pd.Series(wB_vals).dropna()
        base_series = pd.Series(base_vals).dropna()

        # Require at least 5 valid trials
        length = min(len(wA_series), len(wB_series), len(base_series))
        if length < 5:
            # Not enough data for tests
            continue

        # Two tests: (Window A vs. Baseline) and (Window B vs. Baseline)
        p_valA, directionA = paired_test(wA_series, base_series, test_method)
        p_valB, directionB = paired_test(wB_series, base_series, test_method)

        # Apply Bonferroni-corrected alpha
        sigA = (p_valA < alpha_corr)
        sigB = (p_valB < alpha_corr)

        # If either window is significant, label the neuron by the direction(s)
        # of its windows (a neuron can be labeled both 'increase' and 'decrease').
        if sigA or sigB:
            if directionA == 'increase' or directionB == 'increase':
                increase_mask[neuron_idx] = True
            if directionA == 'decrease' or directionB == 'decrease':
                decrease_mask[neuron_idx] = True

    results = {
        'light_on': {
            'increase': np.where(increase_mask)[0],
            'decrease': np.where(decrease_mask)[0]
        }
    }
    return results


def paired_test(series1, series2, method='ttest'):
    """
    Helper function to do a paired test (ttest or wilcoxon)
    and return (p_val, direction).
      direction = 'increase' if mean(series1)>mean(series2)
               or 'decrease' otherwise.
    """
    mean_s1 = series1.mean()
    mean_s2 = series2.mean()
    if method == 'wilcoxon':
        stat, p_val = wilcoxon(series1, series2, alternative='two-sided')
    else:
        stat, p_val = ttest_rel(series1, series2, nan_policy='omit')
    direction = 'increase' if (mean_s1 > mean_s2) else 'decrease'
    return p_val, direction


def identify_light_on_two_windows_boot(
        dff_list,
        light_on_ind,
        light_off_ind,
        baseline_start,
        baseline_stop,
        rec_hz,
        alpha=0.05,
        test_method='ttest',  # or 'wilcoxon'
        do_bonferroni=True,
        # ------------------------
        do_bootstrap=False,
        n_boot=100,
        n_subsample=None,
        min_fraction=0.5
):
    """
    Identify neurons that have a significant change in either of two windows:
      1) First 2s after light_on
      2) Last 2s before light_off

    Compares each window's mean to the baseline [baseline_start : baseline_stop]
    across trials for each neuron. If either window is significant (after
    Bonferroni correction), the neuron is labeled 'increase' or 'decrease'.

    Optionally, use bootstrapping to subsample trials to 'n_subsample' each time,
    and only call neurons significant if they meet the criterion in at least
    'min_fraction' of bootstraps.

    Parameters
    ----------
    dff_list : list of pd.DataFrame
        dff_list[i] is a DataFrame for neuron i, shape = (#trials x #frames).
    light_on_ind : array of shape (#trials,)
        The frame index at which the light turns on (for each trial).
    light_off_ind : array of shape (#trials,)
        The frame index at which the light turns off (for each trial).
    baseline_start : int
        The start index of the baseline window (same for all trials).
    baseline_stop : int
        The stop index of the baseline window (same for all trials).
    rec_hz : int
        The number of frames per second in the data.
    alpha : float
        Significance threshold.
    test_method : {'ttest', 'wilcoxon'}
        Which paired test to run.
    do_bonferroni : bool
        If True, apply Bonferroni correction for the two comparisons.
    do_bootstrap : bool
        If True, run bootstrap approach.
    n_boot : int
        How many bootstrap iterations to run if do_bootstrap=True.
    n_subsample : int or None
        If set (and do_bootstrap=True), randomly pick this many trials
        from the total to run each significance test.
    min_fraction : float
        Neuron must be significant in at least this fraction of bootstrap runs
        to be labeled 'increase' or 'decrease'.

    Returns
    -------
    results : dict
      results['light_on'] = {
        'increase': np.array([...]),
        'decrease': np.array([...])
      }
      where each array is the set of neuron indices that showed a
      significant increase or decrease (for either window).
    """

    # Helper function to do the logic on one set of trials
    def single_run_light_on_two_windows(dff_list_sub, on_ind_sub, off_ind_sub):
        """Returns arrays: inc_mask, dec_mask (bool) for one run (no bootstrapping)."""
        n_neurons_local = len(dff_list_sub)

        inc_mask_local = np.zeros(n_neurons_local, dtype=bool)
        dec_mask_local = np.zeros(n_neurons_local, dtype=bool)

        # For Bonferroni correction across the two comparisons
        alpha_corr = alpha / 2.0 if do_bonferroni else alpha

        # Iterate over neurons
        for neuron_idx in range(n_neurons_local):
            neuron_df = dff_list_sub[neuron_idx]  # shape (#sub-trials x #frames)

            # Collect trial-wise means for Window A & B & baseline
            wA_vals = []
            wB_vals = []
            base_vals = []

            n_subtrials = neuron_df.shape[0]
            for trial_i in range(n_subtrials):
                on_i = on_ind_sub[trial_i]
                off_i = off_ind_sub[trial_i]

                if np.isnan(on_i) or np.isnan(off_i):
                    wA_vals.append(np.nan)
                    wB_vals.append(np.nan)
                    base_vals.append(np.nan)
                    continue

                on_i = int(on_i)
                off_i = int(off_i)
                if off_i <= on_i:
                    wA_vals.append(np.nan)
                    wB_vals.append(np.nan)
                    base_vals.append(np.nan)
                    continue

                # Baseline
                bs = int(baseline_start)
                bp = int(baseline_stop)
                base_mean = neuron_df.iloc[trial_i, bs:bp].mean()

                # Window A: [light_on, light_on+2s)
                A_start = on_i
                A_stop = on_i + 2 * rec_hz
                if A_stop > A_start:
                    wA_mean = neuron_df.iloc[trial_i, A_start:A_stop].mean()
                else:
                    wA_mean = np.nan

                # Window B: [light_off - 2s, light_off)
                B_stop = off_i
                B_start = off_i - 2 * rec_hz
                if B_stop > B_start:
                    wB_mean = neuron_df.iloc[trial_i, B_start:B_stop].mean()
                else:
                    wB_mean = np.nan

                wA_vals.append(wA_mean)
                wB_vals.append(wB_mean)
                base_vals.append(base_mean)

            # Convert to Series and drop NaNs
            wA_series = pd.Series(wA_vals).dropna()
            wB_series = pd.Series(wB_vals).dropna()
            base_series = pd.Series(base_vals).dropna()

            # Paired test requires same length for each comparison
            length = min(len(wA_series), len(wB_series), len(base_series))
            if length < 5:
                continue  # Not enough data for tests

            # trim all three to the same number of trials
            wA_series = wA_series.iloc[:length]
            wB_series = wB_series.iloc[:length]
            base_series = base_series.iloc[:length]

            # Test A: Window A vs Baseline
            p_valA, dirA = paired_test(wA_series, base_series, test_method)
            # Test B: Window B vs Baseline
            p_valB, dirB = paired_test(wB_series, base_series, test_method)

            sigA = (p_valA < alpha_corr)
            sigB = (p_valB < alpha_corr)

            if sigA or sigB:
                # Label by the direction(s) of the windows (can be both)
                if (dirA == 'increase') or (dirB == 'increase'):
                    inc_mask_local[neuron_idx] = True
                if (dirA == 'decrease') or (dirB == 'decrease'):
                    dec_mask_local[neuron_idx] = True

        return inc_mask_local, dec_mask_local

    # ---------------------------------------------------------------------
    # Trials without a detected opto on/off get the nominal stimulation window:
    # start_opto = pre_trial_length + 1 s, end_opto = pre_trial_length + 5 s
    # (pre_trial_length is the module-level constant)
    # ---------------------------------------------------------------------
    global pre_trial_length
    start_opto = pre_trial_length + 1 * rec_hz
    end_opto = pre_trial_length + 5 * rec_hz

    light_on_ind = np.asarray(light_on_ind, dtype=float)
    light_off_ind = np.asarray(light_off_ind, dtype=float)
    light_on_ind[np.isnan(light_on_ind)] = start_opto
    light_off_ind[np.isnan(light_off_ind)] = end_opto

    # ---------------------------------------------------------------------
    # If NOT doing bootstrap => just run once on the full data
    # ---------------------------------------------------------------------
    if not do_bootstrap or (n_subsample is None):
        inc_mask, dec_mask = single_run_light_on_two_windows(
            dff_list,
            light_on_ind,
            light_off_ind
        )
        results = {
            'light_on': {
                'increase': np.where(inc_mask)[0],
                'decrease': np.where(dec_mask)[0]
            }
        }
        return results

    # ---------------------------------------------------------------------
    # Otherwise: bootstrap over trial subsamples
    # ---------------------------------------------------------------------
    n_neurons = len(dff_list)
    n_trials = dff_list[0].shape[0]

    # Tally how many times each neuron is inc/dec
    inc_count = np.zeros(n_neurons, dtype=int)
    dec_count = np.zeros(n_neurons, dtype=int)

    # For each bootstrap iteration:
    for _ in range(n_boot):
        # 1) Randomly pick n_subsample trials from the full set
        if n_subsample > n_trials:
            raise ValueError(
                f"n_subsample={n_subsample} is larger than total trials={n_trials}!"
            )
        subsampled_idxs = random.sample(range(n_trials), k=n_subsample)

        # 2) Subsample the trials of every neuron and the on/off indices
        dff_sub = []
        for neuron_idx in range(n_neurons):
            df_sub = dff_list[neuron_idx].iloc[subsampled_idxs, :].copy()
            dff_sub.append(df_sub)

        on_sub = light_on_ind[subsampled_idxs]
        off_sub = light_off_ind[subsampled_idxs]

        # 3) Run the single-run analysis
        inc_mask_b, dec_mask_b = single_run_light_on_two_windows(dff_sub, on_sub, off_sub)

        # 4) Update tallies
        inc_count[inc_mask_b] += 1
        dec_count[dec_mask_b] += 1

    # ---------------------------------------------------------------------
    # Final label: significant in at least min_fraction of the bootstrap runs
    # ---------------------------------------------------------------------
    inc_thresh = min_fraction * n_boot
    dec_thresh = min_fraction * n_boot

    final_inc_mask = inc_count >= inc_thresh
    final_dec_mask = dec_count >= dec_thresh

    results = {
        'light_on': {
            'increase': np.where(final_inc_mask)[0],
            'decrease': np.where(final_dec_mask)[0]
        }
    }
    return results


# ---------------------------------------------------------------------------
# Event windows and event data
# ---------------------------------------------------------------------------

def build_events_info(opto_on_ind, opto_off_ind,
                      rewards_times, lick_onset_offset,
                      pre_trial_length, rec_hz):
    """
    Builds, for each event (light_on, light_off, reward, lick_onset, lick_offset),
    per-trial frame indices of the event window, the baseline window and the
    preceding window.
    """

    opto_on_ind = np.asarray(opto_on_ind)
    opto_off_ind = np.asarray(opto_off_ind)
    rewards_times = np.asarray(rewards_times)
    lick_onset = np.asarray(lick_onset_offset['onset'])
    lick_offset = np.asarray(lick_onset_offset['offset'])

    # Trials without a detected opto on/off get the nominal stimulation window
    start_opto = pre_trial_length + 1 * rec_hz  # 1 sec after the trial starts
    end_opto = pre_trial_length + 5 * rec_hz  # 5 sec after the trial starts
    opto_on_ind[np.isnan(opto_on_ind)] = start_opto
    opto_off_ind[np.isnan(opto_off_ind)] = end_opto

    n_trials = len(opto_on_ind)

    # For each event: functions of the trial index giving the window boundaries (frames).
    # Baseline for all events: [pre_trial_length, pre_trial_length + 1 s), i.e. the first
    # second of the trial (distractor light, before opto / delay).
    events_config = {
        'light_on': {
            # event: [opto on + 1 s, opto off)
            'event_start': lambda i: opto_on_ind[i] + 1 * rec_hz,
            'event_stop': lambda i: opto_off_ind[i],
            'baseline_start': lambda i: pre_trial_length,
            'baseline_stop': lambda i: pre_trial_length + 1 * rec_hz,
            'preceding_start': lambda i: pre_trial_length, # no separate preceding window for opto: same as baseline
            'preceding_stop': lambda i: pre_trial_length + 1 * rec_hz,
        },
        'light_off': {
            # event: [opto off, opto off + 2 s); preceding: [opto off - 1 s, opto off)
            'event_start': lambda i: opto_off_ind[i],
            'event_stop': lambda i: opto_off_ind[i] + 2 * rec_hz,
            'baseline_start': lambda i: pre_trial_length,
            'baseline_stop': lambda i: pre_trial_length + 1 * rec_hz,
            'preceding_start': lambda i: (opto_off_ind[i] - rec_hz),
            'preceding_stop': lambda i: (opto_off_ind[i]),
        },
        'reward': {
            # event: [reward, reward + 2 s); preceding: [reward - 1 s, reward)
            'event_start': lambda i: rewards_times[i],
            'event_stop': lambda i: rewards_times[i] + 2 * rec_hz,
            'baseline_start': lambda i: pre_trial_length,
            'baseline_stop': lambda i: pre_trial_length + 1 * rec_hz,
            'preceding_start': lambda i: (rewards_times[i] - rec_hz),
            'preceding_stop': lambda i: (rewards_times[i]),
        },
        'lick_onset': {
            # event: [lick onset, lick offset); preceding: [lick onset - 1 s, lick onset)
            'event_start': lambda i: lick_onset[i],
            'event_stop': lambda i: lick_offset[i],
            'baseline_start': lambda i: pre_trial_length,
            'baseline_stop': lambda i: pre_trial_length + 1 * rec_hz,
            'preceding_start': lambda i: (lick_onset[i] - rec_hz),
            'preceding_stop': lambda i: (lick_onset[i]),
        },
        'lick_offset': {
            # event: [lick offset, lick offset + 4 s); preceding: [lick offset - 1 s, lick offset)
            'event_start': lambda i: lick_offset[i],
            'event_stop': lambda i: lick_offset[i] + 4 * rec_hz,
            'baseline_start': lambda i: pre_trial_length,
            'baseline_stop': lambda i: pre_trial_length + 1 * rec_hz,
            'preceding_start': lambda i: (lick_offset[i] - rec_hz),
            'preceding_stop': lambda i: (lick_offset[i]),
        },
    }

    events_info = {}

    for event_name, cfg in events_config.items():
        event_start_arr = np.zeros(n_trials, dtype=float)
        event_stop_arr = np.zeros(n_trials, dtype=float)
        baseline_start_arr = np.zeros(n_trials, dtype=float)
        baseline_stop_arr = np.zeros(n_trials, dtype=float)

        for i in range(n_trials):
            event_start_arr[i] = cfg['event_start'](i)
            event_stop_arr[i] = cfg['event_stop'](i)
            baseline_start_arr[i] = cfg['baseline_start'](i)
            baseline_stop_arr[i] = cfg['baseline_stop'](i)

        events_info[event_name] = {
            'start': event_start_arr,
            'stop': event_stop_arr,
            'baseline_start': baseline_start_arr,
            'baseline_stop': baseline_stop_arr,
        }

        if 'preceding_start' in cfg and 'preceding_stop' in cfg:
            preceding_start_arr = np.zeros(n_trials, dtype=float)
            preceding_stop_arr = np.zeros(n_trials, dtype=float)
            for i in range(n_trials):
                preceding_start_arr[i] = cfg['preceding_start'](i)
                preceding_stop_arr[i] = cfg['preceding_stop'](i)
            events_info[event_name]['preceding_start'] = preceding_start_arr
            events_info[event_name]['preceding_stop'] = preceding_stop_arr

    return events_info


def plot_increase_decrease_bar(sig_neurons, total_neurons, Title):
    """
    Makes a grouped bar plot for each event,
    comparing #increased vs. #decreased cells.
    """
    event_names = list(sig_neurons.keys())
    n_events = len(event_names)

    inc_counts = []
    dec_counts = []
    for e in event_names:
        inc_counts.append((len(sig_neurons[e]['increase']) / total_neurons) * 100)
        dec_counts.append((len(sig_neurons[e]['decrease']) / total_neurons) * 100)

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
        e2 = events_order[i + 1]

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


def extract_two_windows_averages(
        dff_list,
        light_on_inds,
        light_off_inds,
        rec_hz,
        window_s=2
):
    """
    For each neuron in dff_list, return an array of shape (#trials, 1) with the
    average activity of each trial in the window [light_on + 1 s, light_off).

    Parameters
    ----------
    dff_list : list of pd.DataFrame
        dff_list[i] is a DataFrame for neuron i, shape (#trials x #frames)
    light_on_inds : array-like of shape (#trials,)
        Frame indices at which 'light on' occurs
    light_off_inds : array-like of shape (#trials,)
        Frame indices at which 'light off' occurs
    rec_hz : float
        The sampling rate in frames/second (the window starts 1 s after light on)
    window_s : float, optional
        Not used (only by the commented-out second window)

    Returns
    -------
    all_averages : list of np.ndarray
        all_averages[i] has shape (#trials, 1). Trials with a NaN or
        out-of-range window stay NaN.
    """
    n_neurons = len(dff_list)
    all_averages = []

    light_on_inds = np.asarray(light_on_inds, dtype=float)
    light_off_inds = np.asarray(light_off_inds, dtype=float)

    for i in range(n_neurons):
        neuron_df = dff_list[i]  # shape (#trials, #frames)
        n_trials = neuron_df.shape[0]

        arr_means = np.full((n_trials, 1), np.nan)

        for t in range(n_trials):
            on_i = light_on_inds[t]
            off_i = light_off_inds[t]

            if np.isnan(on_i) or np.isnan(off_i):
                continue

            # Window: [on_i + 1 s, off_i)
            A_start = int(on_i) + 1* rec_hz
            A_stop = int(off_i)

            if (A_stop > A_start) and (A_start >= 0) and (A_stop <= neuron_df.shape[1]):
                wA = neuron_df.iloc[t, A_start:A_stop].mean()
                arr_means[t, 0] = wA

            # Second window (not used):
            # # =========== Window B: [off_i - window_s, off_i) ===========
            # B_stop = int(off_i)
            # B_start = B_stop - int(window_s * rec_hz)
            #
            # if (B_stop > B_start) and (B_start >= 0) and (B_stop <= neuron_df.shape[1]):
            #     wB = neuron_df.iloc[t, B_start:B_stop].mean()
            #     arr_means[t, 1] = wB

        all_averages.append(arr_means)

    return all_averages


def two_sample_test(arr1, arr2, method='ttest'):
    """
    Two-sample test returning the p-value.
    - 'ttest':   scipy.stats.ttest_rel
    - 'wilcoxon' or 'ranksums':  scipy.stats.ranksums
    """
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]
    if len(arr1) < 2 or len(arr2) < 2:
        # Not enough data => return p_val=1 so it won't count as "different"
        return 1.0

    if method.lower() in ['wilcoxon', 'ranksums']:
        stat, p_val = ranksums(arr1, arr2)
    else:
        stat, p_val = ttest_rel(arr1, arr2, nan_policy='omit')
    return p_val


def cr_vs_fa_bootstrap_analysis(
        cr_event_averages,  # list of arrays, each shape (#CR_trials, 2)
        fa_event_averages,  # list of arrays, each shape (#FA_trials, 2)
        cr_only_cells,  # list or array of neuron indices
        n_boot=100,
        alpha=0.05,
        test_method='ttest'
):
    """
    For each cell in cr_only_cells, repeatedly subsample CR trials (matching
    the number of FA trials) and compare the window averages (column 0) to FA.
    An iteration counts as "CR_sub looks like FA" if p > alpha.
    (A four-comparison version for two windows is kept commented out below.)

    Returns
    -------
    looks_like_fa_fraction : np.ndarray
      shape (#cr_only_cells,) with the fraction of bootstrap iterations
      in which that cell's subsampled CR is indistinguishable from FA.
    """
    looks_like_fa_fraction = np.zeros(len(cr_only_cells), dtype=float)

    for idx_local, cell_idx in enumerate(cr_only_cells):
        cr_vals_2d = cr_event_averages[cell_idx]  # shape (#CR_trials, 2)
        fa_vals_2d = fa_event_averages[cell_idx]  # shape (#FA_trials, 2)

        n_cr = cr_vals_2d.shape[0]
        n_fa = fa_vals_2d.shape[0]

        count_look_fa = 0

        for _ in range(n_boot):
            # random subset of CR of size n_fa (skipped if there are fewer CR than FA trials)
            if n_fa > n_cr:
                continue

            subsampled_idxs = random.sample(range(n_cr), k=n_fa)
            cr_sub_2d = cr_vals_2d[subsampled_idxs, :]  # shape (n_fa, 2)
            pvals = []

            # CR(A) vs. FA(A)
            pvals.append(two_sample_test(cr_sub_2d[:, 0], fa_vals_2d[:, 0], method=test_method))
            # Four-comparison version for two windows (A, B) (not used):
            # pvals = []
            #
            # # CR(A) vs. FA(A)
            # pvals.append(two_sample_test(cr_sub_2d[:, 0], fa_vals_2d[:, 0], method=test_method))
            # # CR(B) vs. FA(B)
            # pvals.append(two_sample_test(cr_sub_2d[:, 1], fa_vals_2d[:, 1], method=test_method))
            # # CR(A) vs. FA(B)
            # pvals.append(two_sample_test(cr_sub_2d[:, 0], fa_vals_2d[:, 1], method=test_method))
            # # CR(B) vs. FA(A)
            # pvals.append(two_sample_test(cr_sub_2d[:, 1], fa_vals_2d[:, 0], method=test_method))

            # p > alpha => "CR_sub looks like FA"
            if any(p_val > alpha for p_val in pvals):
                count_look_fa += 1

        # fraction of runs that "looks like FA"
        looks_like_fa_fraction[idx_local] = count_look_fa / n_boot

    return looks_like_fa_fraction


def cr_vs_cr_bootstrap_self(
        cr_event_averages,  # list of arrays, each shape (#CR_trials, 2)
        cell_indices,  # which neurons to check
        n_fa,  # the subset size (like the # of FA trials)
        n_boot=100,
        alpha=0.05,
        test_method='ttest'
):
    """
    Control for cr_vs_fa_bootstrap_analysis: for each neuron in cell_indices,
    pick two disjoint subsets (each of size n_fa) of the same CR trials and
    compare them. An iteration counts as "similar" if p > alpha.

    Returns
    -------
    self_sim_fraction : np.ndarray
      shape (#cell_indices,), fraction of bootstrap iterations
      in which the two subsamples are "similar."
    """
    self_sim_fraction = np.zeros(len(cell_indices), dtype=float)

    for i_local, cell_idx in enumerate(cell_indices):
        cr_2d = cr_event_averages[cell_idx]  # shape (#CR_trials, 2)
        n_cr = cr_2d.shape[0]

        count_similar = 0

        for _ in range(n_boot):
            if n_fa * 2 > n_cr:
                # if you want disjoint subsets, do something like:
                # subset_all = random.sample(range(n_cr), k=2*n_fa)
                # subset1 = subset_all[:n_fa]
                # subset2 = subset_all[n_fa:]
                pass

            subset_all = random.sample(range(n_cr), k=2 * n_fa)
            subset1_idx = subset_all[:n_fa]
            subset2_idx = subset_all[n_fa:]

            sub1_2d = cr_2d[subset1_idx, :]
            sub2_2d = cr_2d[subset2_idx, :]

            pvals = []
            pvals.append(two_sample_test(sub1_2d[:, 0], sub2_2d[:, 0], test_method))

            # Four-comparison version for two windows (A, B) (not used):
            # pvals = []
            # # A->A
            # pvals.append(two_sample_test(sub1_2d[:, 0], sub2_2d[:, 0], test_method))
            # # B->B
            # pvals.append(two_sample_test(sub1_2d[:, 1], sub2_2d[:, 1], test_method))
            # # A->B
            # pvals.append(two_sample_test(sub1_2d[:, 0], sub2_2d[:, 1], test_method))
            # # B->A
            # pvals.append(two_sample_test(sub1_2d[:, 1], sub2_2d[:, 0], test_method))

            # p > alpha => "similar"
            if any(p > alpha for p in pvals):
                count_similar += 1

        self_sim_fraction[i_local] = count_similar / n_boot

    return self_sim_fraction


def build_event_data_matrices(dff_list, events_info):
    """
    Extract raw data segments for each event window, baseline window,
    and preceding window (if applicable) for each neuron and each trial.

    Parameters
    ----------
    dff_list : list of pd.DataFrame
        dff_list[i] has shape (#trials, #frames) for neuron i
    events_info : dict
        The dictionary returned by build_events_info(...). For each event_name:
            events_info[event_name] = {
                'start': array of shape (#trials,),
                'stop':  array of shape (#trials,),
                'baseline_start': array (#trials,),
                'baseline_stop':  array (#trials,),
                'preceding_start': array (#trials,),  # optional
                'preceding_stop':  array (#trials,)   # optional
            }

    Returns
    -------
    all_event_data : dict
      Keys = event_name, e.g. 'light_off', 'reward', 'lick_onset', ...
      Values = dict with up to three sub-keys: 'event', 'baseline', 'preceding'.
         all_event_data[event_name]['event'] = a list of length n_neurons
            each neuron entry is a list (length #trials) of 1D numpy arrays
            corresponding to the raw data in that event window for each trial.
         similarly for 'baseline' and 'preceding' (if preceding_... is in the info).
    """
    all_event_data = {}
    n_neurons = len(dff_list)

    for event_name, evt_dict in events_info.items():
        event_data_for_neurons = []
        baseline_data_for_neurons = []
        preceding_data_for_neurons = []

        has_preceding = ('preceding_start' in evt_dict) and ('preceding_stop' in evt_dict)

        event_starts = evt_dict['start']  # shape (#trials,)
        event_stops = evt_dict['stop']  # shape (#trials,)
        base_starts = evt_dict['baseline_start']  # shape (#trials,)
        base_stops = evt_dict['baseline_stop']  # shape (#trials,)
        if has_preceding:
            prec_starts = evt_dict['preceding_start']
            prec_stops = evt_dict['preceding_stop']

        for neuron_idx in range(n_neurons):
            neuron_df = dff_list[neuron_idx]  # shape (#trials, #frames)
            n_trials = neuron_df.shape[0]

            event_segments = []
            baseline_segments = []
            preceding_segments = []

            for t in range(n_trials):
                start_val = event_starts[t]
                stop_val = event_stops[t]

                # --- EVENT window (empty if undefined) ---
                if np.isnan(start_val) or np.isnan(stop_val):
                    seg_event = np.array([])
                else:
                    st_e = max(int(start_val), 0)
                    sp_e = min(int(stop_val), neuron_df.shape[1])
                    if sp_e > st_e:
                        seg_event = neuron_df.iloc[t, st_e:sp_e].values
                    else:
                        seg_event = np.array([])

                event_segments.append(seg_event)

                # --- BASELINE window: from base_starts[t] to base_stops[t] ---
                st_b = int(base_starts[t])
                sp_b = int(base_stops[t])
                st_b = max(st_b, 0)
                sp_b = min(sp_b, neuron_df.shape[1])
                if sp_b > st_b:
                    seg_base = neuron_df.iloc[t, st_b:sp_b].values
                else:
                    seg_base = np.array([])
                baseline_segments.append(seg_base)

                # --- PRECEDING window (empty if undefined) ---
                start_val = prec_starts[t]
                stop_val = prec_stops[t]

                if np.isnan(start_val) or np.isnan(stop_val):
                    seg_prec = np.array([])
                else:
                    st_p = max(int(start_val), 0)
                    sp_p = min(int(stop_val), neuron_df.shape[1])
                    if sp_p > st_p:
                        seg_prec = neuron_df.iloc[t, st_p:sp_p].values
                    else:
                        seg_prec = np.array([])

                preceding_segments.append(seg_prec)
            event_data_for_neurons.append(event_segments)
            baseline_data_for_neurons.append(baseline_segments)
            if has_preceding:
                preceding_data_for_neurons.append(preceding_segments)

        event_dict_store = {
            'event': event_data_for_neurons,
            'baseline': baseline_data_for_neurons
        }
        if has_preceding:
            event_dict_store['preceding'] = preceding_data_for_neurons

        all_event_data[event_name] = event_dict_store

    return all_event_data


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

    # ----- 3) Two-sided p-value: fraction of permuted AUCs at least as far from 0.5 -----
    real_dist_from_05 = abs(real_auc - 0.5)
    perm_dist_from_05 = abs(perm_distribution - 0.5)

    # fraction of permutations with dist >= real_dist_from_05
    num_extreme = np.sum(perm_dist_from_05 >= real_dist_from_05)
    p_value = num_extreme / n_perm

    return real_auc, p_value, perm_distribution


def compute_auROC_for_events_with_permutation(
        all_event_data,
        n_perm=1000,
        random_state=None
):
    """
    Computes, for each event and neuron, the auROC of the trial-wise mean
    activity in the event window vs. the baseline window, with a
    permutation-based p-value.

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

        event_data_all_neurons = event_dict['event']
        baseline_data_all_neurons = event_dict['baseline']

        n_neurons = len(event_data_all_neurons)
        aurocs = np.full(n_neurons, np.nan)
        p_values = np.full(n_neurons, np.nan)

        for neuron_i in range(n_neurons):
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

            real_auc, p_val, _ = auroc_permutation_test(
                event_vals, base_vals,
                n_perm=n_perm,
                random_state=random_state
            )

            aurocs[neuron_i] = real_auc
            p_values[neuron_i] = p_val

        results[event_name] = {
            'auroc': aurocs,
            'p_value': p_values
        }

    return results


def z_score_each_trial(trials_data, pre_trial_length, baseline_time):
    """
    Z-scores each trace using the mean and standard deviation of the whole trace.

    Parameters:
      trials_data: nested list where trials_data[i][j] is a 1D array-like signal.
      pre_trial_length, baseline_time: not used (the whole trace is used; the
          baseline-window version is kept in the comment next to baseline_segment).

    Returns:
      A new nested list with the same structure as trials_data, z-scored.
    """
    import numpy as np
    import copy

    # copy so that the input is not modified
    z_scored_data = copy.deepcopy(trials_data)

    for trial_idx in range(len(z_scored_data)):
        for neuron_idx in range(len(z_scored_data[trial_idx])):
            neuron_signal = np.asarray(z_scored_data[trial_idx][neuron_idx])

            # whole trace (baseline-window version in the comment)
            baseline_segment = neuron_signal[:] #pre_trial_length - baseline_time: pre_trial_length+1*rec_hz
            mean_baseline = np.mean(baseline_segment)
            std_baseline = np.std(baseline_segment)
            if std_baseline == 0:
                std_baseline = 1  # prevent division by zero

            z_scored_data[trial_idx][neuron_idx] = (neuron_signal - mean_baseline) / std_baseline

    return z_scored_data
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

def signed_outlier_counts(dff_4min: pd.DataFrame, thresh: float = 3.5) -> pd.DataFrame:
    """
    Count positive and negative outliers for every neuron.

    Parameters
    ----------
    dff_4min : DataFrame
        Rows = neurons, columns = time points.
    thresh : float
        Robust z-score threshold. Default is 3.5.

    Returns
    -------
    DataFrame with four columns:
        n_pos     - count above +thresh
        n_neg     - count below -thresh
        frac_pos  - fraction above +thresh
        frac_neg  - fraction below -thresh
    """
    med = dff_4min.median(axis=1)
    mad = dff_4min.apply(median_abs_deviation, axis=1, scale="normal").replace(0, np.nan)
    z = dff_4min.sub(med, axis=0).div(mad, axis=0)

    pos_mask = z > thresh
    neg_mask = z < -thresh

    n_pos = pos_mask.sum(axis=1)
    n_neg = neg_mask.sum(axis=1)
    denom = dff_4min.notna().sum(axis=1)
    frac_pos = n_pos / denom
    frac_neg = n_neg / denom

    return pd.DataFrame({
        "n_pos": n_pos,
        "n_neg": n_neg,
        "frac_pos": frac_pos.round(4),
        "frac_neg": frac_neg.round(4)
    })

def plot_signed_outliers(outlier_df: pd.DataFrame, frac_cutoff: float = 0.05):
    """
    Draw two bar plots: one for positive outliers, one for negative.

    Parameters
    ----------
    outlier_df : DataFrame
        Output of signed_outlier_counts.
    frac_cutoff : float
        Horizontal reference line for a chosen fraction threshold.
    """
    ordered_pos = outlier_df.sort_values("frac_pos")
    ordered_neg = outlier_df.sort_values("frac_neg")

    # Positive
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(ordered_pos)), ordered_pos["frac_pos"])
    ax.axhline(frac_cutoff, linestyle="--", color="red")
    ax.set_title("Fraction above +thresh per neuron")
    ax.set_xlabel("Neuron (sorted)")
    ax.set_ylabel("Fraction of frames")
    plt.tight_layout()
    plt.show()

    # Negative
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(ordered_neg)), ordered_neg["frac_neg"])
    ax.axhline(frac_cutoff, linestyle="--", color="red")
    ax.set_title("Fraction below −thresh per neuron")
    ax.set_xlabel("Neuron (sorted)")
    ax.set_ylabel("Fraction of frames")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Session paths
    day = ['sess12']
    alpha = 0.05
    ROOT_PATH = DATA_ROOT + r'\mouse14\{}\ML\data'.format(day[0])
    licking_data = DATA_ROOT + r'\mouse14\{}\ML\licking_analysis_final'.format(day[0])
    save_loc = DATA_ROOT + r'\mouse14\{}\ML\data\processed_data\auROC_freq_history'.format(day[0], alpha)
    os.makedirs(save_loc, exist_ok=True)
    
    # Number of priming trials to exclude
    n_priming_trials = 23
    
    # Load inputs
    conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
    responses = np.load(join(ROOT_PATH, 'responses.npy'))
    with open(join(ROOT_PATH, 'opto_open_all.pkl'), 'rb') as f:
        opto_on = pickle.load(f)
    with open(join(ROOT_PATH, 'distract_light_on_all.pkl'), 'rb') as f:
        distract_light = pickle.load(f)
    trials_ind = np.load(join(ROOT_PATH, 'trials_ind.npy'))
    run_len = np.load(join(ROOT_PATH, 'run_len.npy'))
    with open(join(licking_data, 'lick_onset_offset_results.pkl'), 'rb') as f:
        lick_onset_offset = pickle.load(f)
    with open(join(ROOT_PATH, 'dff_running_4.5min.pkl'), 'rb') as f:
        dff_4min = pickle.load(f)
    
    # Remove priming trials from all data structures
    conditions_played = conditions_played[n_priming_trials:]
    responses = responses[n_priming_trials:]
    opto_on = opto_on[n_priming_trials:]
    distract_light = distract_light[n_priming_trials:]
    # trials_ind holds the trial onsets of each run (one row per run). The priming trials
    # are in the first run, so remove the first n_priming_trials onsets from the first row
    # only (a list is used because the rows now have different lengths).
    trials_ind = [trials_ind[0][n_priming_trials:], trials_ind[1], trials_ind[2]]


    
    # Remove priming trials from lick_onset_offset dictionary
    for key in lick_onset_offset:
        if isinstance(lick_onset_offset[key], list):
            lick_onset_offset[key] = lick_onset_offset[key][n_priming_trials:]
        elif isinstance(lick_onset_offset[key], np.ndarray):
            lick_onset_offset[key] = lick_onset_offset[key][n_priming_trials:]
    
    # Outlier check (not used)
    # outlier_stats = signed_outlier_counts(dff_4min, thresh=3.5)
    # plot_signed_outliers(outlier_stats, frac_cutoff=0.05)
    
    # Z-score each neuron (across all time points, not per trial)
    # Alternative normalizations:
    # dff_norm = min_max_normalize(dff_4min)
    # dff_norm = norm_to_max(dff_4min)
    dff_norm =[]
    for i in range(len(dff_4min)):
        mean = np.nanmean(dff_4min.iloc[i])
        std = np.nanstd(dff_4min.iloc[i])
        dff_norm.append((dff_4min.iloc[i] - mean) / std)
    dff_norm = pd.DataFrame(dff_norm)
    
    # 1) Divide into trials (with pre-trial frames prepended)
    dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)
    
    # 2) No baseline subtraction
    dff_no_sub = dff_run_list
    
    # 3) Reassemble subtracted data back into a single DataFrame (not used)
    # dff_sub_df = reassemble_trials_remove_pretrial(dff_sub, run_len, int(baseline_len))
    # dff_sub_df = pd.DataFrame(dff_sub_df)
    
    # Save the z-scored data and the trial-wise lists
    with open(join(save_loc, 'all_values.pkl'), 'wb') as f:
        pickle.dump(dff_norm, f)

    with open(join(save_loc, 'dff_run_list_no_sub.pkl'), 'wb') as f:
        pickle.dump(dff_no_sub, f)
    
    with open(join(save_loc, 'dff_run_list.pkl'), 'wb') as f:
        pickle.dump(dff_run_list, f)

    # Behavioral outcomes (response codes: 2 = miss, 3 = hit, 4 = correct reject, 5 = false alarm)
    correct_reject = np.where(responses == 4)[0]
    false_alarm = np.where(responses == 5)[0]
    hit = np.where(responses == 3)[0]
    miss = np.where(responses == 2)[0]
    
    # Save the trial indices of each outcome
    with open(join(save_loc, 'correct_reject.pkl'), 'wb') as f:
        pickle.dump(correct_reject, f)
    with open(join(save_loc, 'false_alarm.pkl'), 'wb') as f:
        pickle.dump(false_alarm, f)
    with open(join(save_loc, 'hit.pkl'), 'wb') as f:
        pickle.dump(hit, f)
    with open(join(save_loc, 'miss.pkl'), 'wb') as f:
        pickle.dump(miss, f)

    light_ind = [i for i in range(len(opto_on)) if i in correct_reject or i in false_alarm] # all light trials

    # Opto onset of each trial: first sample where the opto signal exceeds 1
    opto_on_ind = []
    distract_light_on = []
    distract_light_off = []
    for i in range(len(opto_on)):
        changes = np.diff(np.concatenate(([0], opto_on[i] > 1)))
        ind = np.where(changes == 1)[0]
        opto_on_ind.append(ind[0] if len(ind) > 0 else np.nan)
    # shift by the prepended pre-trial frames
    opto_on_ind = [i + pre_trial_length for i in opto_on_ind]
    opto_off_ind = [i + stim_len for i in opto_on_ind]

    # light-on lengths
    distract_light_len = []
    opto_len = []
    for i in range(len(opto_on)):
        # distract_light_len.append(distract_light_off[i] - distract_light_on[i])
        opto_len.append(opto_off_ind[i] - opto_on_ind[i])

    # Split each neuron's trials by condition
    dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, dff_light = divide_to_conditions(dff_no_sub, correct_reject,
                                                                                  false_alarm, hit, miss, light_ind)
    
    # -------------------------------
    # Frequency-Specific Grouping
    # -------------------------------
    # conditions_played holds the condition code of each trial;
    # frequency_map: condition code -> opto stimulation frequency (Hz)
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

    # Trial indices of each frequency
    freq_indices = {freq: [i for i, x in enumerate(conditions_played) if x == freq]
                    for freq in frequency_map.keys()}


    # # For each frequency, intersect its trial indices with the global correct reject indices.
    # freq_correct_reject_indices = {}
    # for freq, idx_list in freq_indices.items():
    #     freq_corr = [i for i in idx_list if i in correct_reject]
    #     if freq_corr:
    #         freq_correct_reject_indices[frequency_map[freq]] = freq_corr

    # Build per-frequency light dF/F data (per neuron)
    light_by_freq = {}
    for freq, trial_indices in freq_indices.items():
        neuron_data = []
        for neuron_trials in dff_no_sub:
            # Ensure index is within range for each neuron
            filtered_trials = [neuron_trials[i] for i in trial_indices if i < len(neuron_trials)]
            neuron_data.append(filtered_trials)
        light_by_freq[freq] = neuron_data

    # -------------------------------
    # Global Event Analysis (All Frequencies Together)
    # -------------------------------
    # Event windows and event data of all light trials
    events_all_light = build_events_info(opto_on_ind= [opto_on_ind[i] for i in light_ind],  
                                            opto_off_ind=[opto_off_ind[i] for i in light_ind],
                                            rewards_times=[lick_onset_offset['rewards_times'][i] for i in light_ind],
                                            lick_onset_offset={
                                                'onset': [lick_onset_offset['onset'][i] for i in light_ind],
                                                'offset': [lick_onset_offset['offset'][i] for i in light_ind]
                                            },
                                            pre_trial_length=pre_trial_length,
                                            rec_hz=rec_hz)

    # events_correct_reject = build_events_info(opto_on_ind=[opto_on_ind[i] for i in correct_reject],
    #                                           # shape = (#hit_trials,)
    #                                           opto_off_ind=[opto_off_ind[i] for i in correct_reject],
    #                                           rewards_times=[lick_onset_offset['rewards_times'][i] for i in
    #                                                          correct_reject],
    #                                           lick_onset_offset={
    #                                               'onset': [lick_onset_offset['onset'][i] for i in correct_reject],
    #                                               'offset': [lick_onset_offset['offset'][i] for i in correct_reject]
    #                                           },
    #                                           pre_trial_length=pre_trial_length,
    #                                           rec_hz=rec_hz)

    # events_hit = build_events_info(opto_on_ind=[opto_on_ind[i] for i in hit],  # shape = (#hit_trials,)
    #                                opto_off_ind=[opto_off_ind[i] for i in hit],
    #                                rewards_times=[lick_onset_offset['rewards_times'][i] for i in hit],
    #                                lick_onset_offset={
    #                                    'onset': [lick_onset_offset['onset'][i] for i in hit],
    #                                    'offset': [lick_onset_offset['offset'][i] for i in hit]
    #                                },
    #                                pre_trial_length=pre_trial_length,
    #                                rec_hz=rec_hz)
    # events_false_alarm = build_events_info(opto_on_ind=[opto_on_ind[i] for i in false_alarm],
    #                                        # shape = (#hit_trials,)
    #                                        opto_off_ind=[opto_off_ind[i] for i in false_alarm],
    #                                        rewards_times=[lick_onset_offset['rewards_times'][i] for i in
    #                                                       false_alarm],
    #                                        lick_onset_offset={
    #                                            'onset': [lick_onset_offset['onset'][i] for i in false_alarm],
    #                                            'offset': [lick_onset_offset['offset'][i] for i in false_alarm]
    #                                        },
    #                                        pre_trial_length=pre_trial_length,
    #                                        rec_hz=rec_hz)
    # events_miss = build_events_info(opto_on_ind=[opto_on_ind[i] for i in miss],  # shape = (#hit_trials,)
    #                                 opto_off_ind=[opto_off_ind[i] for i in miss],
    #                                 rewards_times=[lick_onset_offset['rewards_times'][i] for i in miss],
    #                                 lick_onset_offset={
    #                                     'onset': [lick_onset_offset['onset'][i] for i in miss],
    #                                     'offset': [lick_onset_offset['offset'][i] for i in miss]
    #                                 },
    #                                 pre_trial_length=pre_trial_length,
    #                                 rec_hz=rec_hz)
    #
    # # # create the event data matrices
    event_data_all_light = build_event_data_matrices(dff_light, events_all_light)
    # event_data_correct_reject = build_event_data_matrices(dff_correct_reject, events_correct_reject)
    # event_data_hit = build_event_data_matrices(dff_hit, events_hit)
    # event_data_false_alarm = build_event_data_matrices(dff_false_alarm, events_false_alarm)
    # event_data_miss = build_event_data_matrices(dff_miss, events_miss)
    # Save the event data
    with open(join(save_loc, 'event_data_all_light.pkl'), 'wb') as f:
        pickle.dump(event_data_all_light, f)
    # with open(join(save_loc, 'event_data_correct_reject.pkl'), 'wb') as f:
    #     pickle.dump(event_data_correct_reject, f)
    # with open(join(save_loc, 'event_data_hit.pkl'), 'wb') as f:
    #     pickle.dump(event_data_hit, f)
    # with open(join(save_loc, 'event_data_false_alarm.pkl'), 'wb') as f:
    #     pickle.dump(event_data_false_alarm, f)
    # with open(join(save_loc, 'event_data_miss.pkl'), 'wb') as f:
    #     pickle.dump(event_data_miss, f)


    # Mean activity in [opto on + 1 s, opto off) for every light trial
    all_light_event_averages = extract_two_windows_averages(dff_light, [opto_on_ind[i] for i in light_ind],
                                                            [opto_off_ind[i] for i in light_ind], rec_hz=15,
                                                            window_s=2)
    with open(join(save_loc, 'all_light_event_averages.pkl'), 'wb') as f:
        pickle.dump(all_light_event_averages, f)
    # cr_event_averages = extract_two_windows_averages(dff_correct_reject, [opto_on_ind[i] for i in correct_reject],
    #                                                  [opto_off_ind[i] for i in correct_reject], rec_hz=15,
    #                                                  window_s=2)
    # fa_event_averages = extract_two_windows_averages(dff_false_alarm, [opto_on_ind[i] for i in false_alarm],
    #                                                  [opto_off_ind[i] for i in false_alarm], rec_hz=15, window_s=2)
    # opto_on_fake = np.nanmean(opto_on_ind) * np.ones(len(hit))
    # opto_off_fake = np.nanmean(opto_off_ind) * np.ones(len(hit))
    # Hit_event_averages = extract_two_windows_averages(dff_hit, list(opto_on_fake), list(opto_off_fake), rec_hz=15,
    #                                                   window_s=2)
    # opto_on_fake = np.nanmean(opto_on_ind) * np.ones(len(miss))
    # opto_off_fake = np.nanmean(opto_off_ind) * np.ones(len(miss))
    # miss_event_averages = extract_two_windows_averages(dff_miss, list(opto_on_fake), list(opto_off_fake), rec_hz=15,
    #                                                    window_s=2)

    # # iti_on = 0 * np.ones(len(correct_reject))
    # # iti_off = (iti_on + 3 * rec_hz) * np.ones(len(correct_reject))
    # # cr_event_averages = extract_two_windows_averages(dff_correct_reject, list(iti_on), list(iti_off),
    # #                                                   rec_hz=15,
    # #                                                  window_s=2)
    # # iti_on_fa = 0 * np.ones(len(false_alarm))
    # # iti_off_fa = (iti_on_fa + 3 * rec_hz) * np.ones(len(false_alarm))
    # # fa_event_averages = extract_two_windows_averages(dff_false_alarm, list(iti_on_fa), list(iti_off_fa),
    # #                                                   rec_hz=15, window_s=2)
    # # iti_on_hit = 0 * np.ones(len(hit))
    # # iti_off_hit = (iti_on_hit + 3 * rec_hz) * np.ones(len(hit))
    # # Hit_event_averages = extract_two_windows_averages(dff_hit, list(iti_on_hit), list(iti_off_hit), rec_hz=15,
    # #                                                   window_s=2)
    # # iti_on_miss = 0 * np.ones(len(miss))
    # # iti_off_miss = (iti_on_miss + 3 * rec_hz) * np.ones(len(miss))
    # # miss_event_averages = extract_two_windows_averages(dff_miss, list(iti_on_miss), list(iti_off_miss), rec_hz=15,
    # #                                                    window_s=2)
    # # save
    # with open(join(save_loc, 'cr_light_averages.pkl'), 'wb') as f:
    #     pickle.dump(cr_event_averages, f)
    # with open(join(save_loc, 'fa_light_averages.pkl'), 'wb') as f:
    #     pickle.dump(fa_event_averages, f)
    # with open(join(save_loc, 'hit_light_averages.pkl'), 'wb') as f:
    #     pickle.dump(Hit_event_averages, f)
    # with open(join(save_loc, 'miss_light_averages.pkl'), 'wb') as f:
    #     pickle.dump(miss_event_averages, f)

    # # -------------------------------
    # # Trial History Analysis - After Hit vs Not After Hit
    # # -------------------------------
    # # Create hit_set for trial history analysis
    # hit_set = set(hit)
    
    # # For FA and CR, create "after hit" and "not after hit" lists
    # fa_after_hit_idx = []
    # fa_not_after_hit_idx = []
    # cr_after_hit_idx = []
    # cr_not_after_hit_idx = []

    # for idx in false_alarm:
    #     if idx - 1 in hit_set:
    #         fa_after_hit_idx.append(idx)
    #     else:
    #         fa_not_after_hit_idx.append(idx)

    # for idx in correct_reject:
    #     if idx - 1 in hit_set:
    #         cr_after_hit_idx.append(idx)
    #     else:
    #         cr_not_after_hit_idx.append(idx)

    # # Helper function to extract data by trial indices
    # def extract_by_trial_indices(data_list, all_indices, selected_indices):
    #     # Find the positions of selected_indices within all_indices
    #     selected_positions = []
    #     for i, idx in enumerate(all_indices):
    #         if idx in selected_indices:
    #             selected_positions.append(i)
    #     return [df.iloc[selected_positions, :] for df in data_list]

    # # Convert event averages to DataFrames for easier manipulation
    # fa_event_averages_df = [pd.DataFrame(neuron) for neuron in fa_event_averages]
    # cr_event_averages_df = [pd.DataFrame(neuron) for neuron in cr_event_averages]

    # # Extract data for trial history analysis
    # fa_after_hit_data = extract_by_trial_indices(fa_event_averages_df, false_alarm, fa_after_hit_idx)
    # fa_not_after_hit_data = extract_by_trial_indices(fa_event_averages_df, false_alarm, fa_not_after_hit_idx)
    # cr_after_hit_data = extract_by_trial_indices(cr_event_averages_df, correct_reject, cr_after_hit_idx)
    # cr_not_after_hit_data = extract_by_trial_indices(cr_event_averages_df, correct_reject, cr_not_after_hit_idx)

    # # Save trial history data for later use
    # with open(join(save_loc, 'fa_after_hit_data.pkl'), 'wb') as f:
    #     pickle.dump(fa_after_hit_data, f)
    # with open(join(save_loc, 'fa_not_after_hit_data.pkl'), 'wb') as f:
    #     pickle.dump(fa_not_after_hit_data, f)
    # with open(join(save_loc, 'cr_after_hit_data.pkl'), 'wb') as f:
    #     pickle.dump(cr_after_hit_data, f)
    # with open(join(save_loc, 'cr_not_after_hit_data.pkl'), 'wb') as f:
    #     pickle.dump(cr_not_after_hit_data, f)

    # # Print summary statistics
    # print(f"Trial History Summary:")
    # print(f"FA trials: {len(fa_after_hit_idx)} after hit, {len(fa_not_after_hit_idx)} not after hit")
    # print(f"CR trials: {len(cr_after_hit_idx)} after hit, {len(cr_not_after_hit_idx)} not after hit")

    # # -------------------------------
    # # Frequency-Specific Event Analysis
    # # -------------------------------
    # # Loop over each frequency and build event info, event data matrices,
    # # and extract two-window averages for that frequency.
    # freq_event_data = {}
    # freq_light_event_averages = {}
    # freq_iti_event_averages = {}
    # for freq, trial_indices in freq_indices.items():
    #     save_freq = frequency_map[freq]
    #     print(f"\nProcessing frequency {save_freq} Hz:")
    #     # Build events info for this frequency using only the trials in trial_indices.
    #     events_freq = build_events_info(
    #         opto_on_ind=[opto_on_ind[i] for i in trial_indices],
    #         opto_off_ind=[opto_off_ind[i] for i in trial_indices],
    #         rewards_times=[lick_onset_offset['rewards_times'][i] for i in trial_indices],
    #         lick_onset_offset={
    #             'onset': [lick_onset_offset['onset'][i] for i in trial_indices],
    #             'offset': [lick_onset_offset['offset'][i] for i in trial_indices]
    #         },
    #         pre_trial_length=pre_trial_length,
    #         rec_hz=rec_hz
    #     )
    #     # dff data for this frequency (per neuron)
    #     dff_light_freq = light_by_freq[freq]
    #     # Build event data matrices
    #     dff_light_freq = [pd.DataFrame(neuron) for neuron in dff_light_freq]
    #     event_data_freq = build_event_data_matrices(dff_light_freq, events_freq)
    #     with open(join(save_loc, f'event_data_all_light_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(event_data_freq, f)
    #     freq_event_data[freq] = event_data_freq

    #     # Extract two-window averages for this frequency using ITI instead of opto
    #     # iti_event_avg_freq = extract_two_windows_averages(
    #     #     dff_cr_freq,
    #     #     [0] * len(trial_indices),  # ITI starts at frame 0 for each trial
    #     #     [3 * rec_hz] * len(trial_indices),  # ITI ends at 3 seconds * rec_hz
    #     #     rec_hz=rec_hz,
    #     #     window_s=2
    #     # )

    #     # Extract two-window averages for this frequency.
    #     light_event_avg_freq = extract_two_windows_averages(
    #         dff_light_freq,
    #         [opto_on_ind[i] for i in trial_indices],
    #         [opto_off_ind[i] for i in trial_indices],
    #         rec_hz=rec_hz,
    #         window_s=2
    #     )

    #     # with open(join(save_loc, f'all_iti_averages_{save_freq}.pkl'), 'wb') as f:
    #     #     pickle.dump(iti_event_avg_freq, f)
    #     # freq_iti_event_averages[freq] = iti_event_avg_freq

    #     with open(join(save_loc, f'all_light_averages_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(light_event_avg_freq, f)
    #     freq_light_event_averages[freq] = light_event_avg_freq

    #     # separate to CR and FA for this frequency
    #     cr_freq_trials = [i for i in range(len(trial_indices)) if trial_indices[i] in correct_reject]
    #     fa_freq_trials = [i for i in range(len(trial_indices)) if trial_indices[i] in false_alarm]
    #     data_cr =  [[neuron.iloc[i] for i in cr_freq_trials] for neuron in dff_light_freq]
    #     data_cr = [pd.DataFrame(neuron) for neuron in data_cr]
    #     data_fa = [[neuron.iloc[i] for i in fa_freq_trials] for neuron in dff_light_freq]
    #     data_fa = [pd.DataFrame(neuron) for neuron in data_fa]
    #     if len(cr_freq_trials) == 0 or len(fa_freq_trials) == 0:
    #         print(f"Skipping frequency {save_freq} Hz - no CR or FA trials found.")
    #         continue

    #     # Use ITI windows for CR trials
    #     # cr_event_averages_freq = extract_two_windows_averages(
    #     #     data_cr,
    #     #     [0] * len(cr_freq_trials),  # ITI starts at frame 0
    #     #     [3 * rec_hz] * len(cr_freq_trials),  # ITI ends at 3 seconds * rec_hz
    #     #     rec_hz=rec_hz,
    #     #     window_s=2
    #     # )
    #     # # Use ITI windows for FA trials
    #     # fa_event_averages_freq = extract_two_windows_averages(
    #     #     data_fa,
    #     #     [0] * len(fa_freq_trials),  # ITI starts at frame 0
    #     #     [3 * rec_hz] * len(fa_freq_trials),  # ITI ends at 3 seconds * rec_hz
    #     #     rec_hz=rec_hz,
    #     #     window_s=2
    #     # )
    #     # with open(join(save_loc, f'cr_iti_averages_{save_freq}.pkl'), 'wb') as f:
    #     #     pickle.dump(cr_event_averages_freq, f)
    #     # with open(join(save_loc, f'fa_iti_averages_{save_freq}.pkl'), 'wb') as f:
    #     #     pickle.dump(fa_event_averages_freq, f)
    #     cr_event_averages_freq = extract_two_windows_averages(
    #         data_cr,
    #         [opto_on_ind[trial_indices[i]] for i in range(len(trial_indices)) if trial_indices[i] in correct_reject],
    #         [opto_off_ind[trial_indices[i]] for i in range(len(trial_indices)) if trial_indices[i] in correct_reject],
    #         rec_hz=rec_hz,
    #         window_s=2
    #     )
    #     fa_event_averages_freq = extract_two_windows_averages(
    #         data_fa,
    #         [opto_on_ind[trial_indices[i]] for i in range(len(trial_indices)) if trial_indices[i] in false_alarm],
    #         [opto_off_ind[trial_indices[i]] for i in range(len(trial_indices)) if trial_indices[i] in false_alarm],
    #         rec_hz=rec_hz,
    #         window_s=2
    #     )
    #     with open(join(save_loc, f'cr_light_averages_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(cr_event_averages_freq, f)
    #     with open(join(save_loc, f'fa_light_averages_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(fa_event_averages_freq, f)

    #     # -------------------------------
    #     # Trial History Analysis for This Frequency
    #     # -------------------------------
    #     # Get the actual trial indices (not positions) for this frequency
    #     freq_cr_trial_indices = [trial_indices[i] for i in range(len(trial_indices)) if trial_indices[i] in correct_reject]
    #     freq_fa_trial_indices = [trial_indices[i] for i in range(len(trial_indices)) if trial_indices[i] in false_alarm]
        
    #     # Create "after hit" and "not after hit" lists for this frequency
    #     freq_fa_after_hit_idx = []
    #     freq_fa_not_after_hit_idx = []
    #     freq_cr_after_hit_idx = []
    #     freq_cr_not_after_hit_idx = []

    #     for idx in freq_fa_trial_indices:
    #         if idx - 1 in hit_set:
    #             freq_fa_after_hit_idx.append(idx)
    #         else:
    #             freq_fa_not_after_hit_idx.append(idx)

    #     for idx in freq_cr_trial_indices:
    #         if idx - 1 in hit_set:
    #             freq_cr_after_hit_idx.append(idx)
    #         else:
    #             freq_cr_not_after_hit_idx.append(idx)

    #     # Helper function to extract data by trial indices for frequency-specific data
    #     def extract_by_trial_indices_freq(data_list, all_trial_indices, selected_trial_indices):
    #         # Find the positions of selected_trial_indices within all_trial_indices
    #         selected_positions = []
    #         for i, idx in enumerate(all_trial_indices):
    #             if idx in selected_trial_indices:
    #                 selected_positions.append(i)
    #         return [df.iloc[selected_positions, :] for df in data_list]

    #     # Convert frequency-specific event averages to DataFrames
    #     fa_event_averages_freq_df = [pd.DataFrame(neuron) for neuron in fa_event_averages_freq]
    #     cr_event_averages_freq_df = [pd.DataFrame(neuron) for neuron in cr_event_averages_freq]

    #     # Extract trial history data for this frequency
    #     freq_fa_after_hit_data = extract_by_trial_indices_freq(fa_event_averages_freq_df, freq_fa_trial_indices, freq_fa_after_hit_idx)
    #     freq_fa_not_after_hit_data = extract_by_trial_indices_freq(fa_event_averages_freq_df, freq_fa_trial_indices, freq_fa_not_after_hit_idx)
    #     freq_cr_after_hit_data = extract_by_trial_indices_freq(cr_event_averages_freq_df, freq_cr_trial_indices, freq_cr_after_hit_idx)
    #     freq_cr_not_after_hit_data = extract_by_trial_indices_freq(cr_event_averages_freq_df, freq_cr_trial_indices, freq_cr_not_after_hit_idx)

    #     # Save frequency-specific trial history data
    #     with open(join(save_loc, f'fa_after_hit_data_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(freq_fa_after_hit_data, f)
    #     with open(join(save_loc, f'fa_not_after_hit_data_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(freq_fa_not_after_hit_data, f)
    #     with open(join(save_loc, f'cr_after_hit_data_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(freq_cr_after_hit_data, f)
    #     with open(join(save_loc, f'cr_not_after_hit_data_{save_freq}.pkl'), 'wb') as f:
    #         pickle.dump(freq_cr_not_after_hit_data, f)

    #     # Print frequency-specific trial history summary
    #     print(f"  Frequency {save_freq} Hz Trial History:")
    #     print(f"    FA trials: {len(freq_fa_after_hit_idx)} after hit, {len(freq_fa_not_after_hit_idx)} not after hit")
    #     print(f"    CR trials: {len(freq_cr_after_hit_idx)} after hit, {len(freq_cr_not_after_hit_idx)} not after hit")

    # # -------------------------------
    # # Combined All-Frequencies Light Event Analysis
    # # -------------------------------
    # print("\nProcessing all frequencies combined:")
    
    # # Combine all trial indices from all frequencies
    # all_freq_trial_indices = []
    # for freq, trial_indices in freq_indices.items():
    #     all_freq_trial_indices.extend(trial_indices)
    
    # # Remove duplicates and sort (in case there are overlapping trials)
    # all_freq_trial_indices = sorted(list(set(all_freq_trial_indices)))
    
    # # Combine all frequency data
    # dff_light_all_freq = []
    # for neuron_trials in dff_no_sub:
    #     # Get data for all frequency trials for this neuron
    #     filtered_trials = [neuron_trials[i] for i in all_freq_trial_indices if i < len(neuron_trials)]
    #     dff_light_all_freq.append(filtered_trials)
    
    # # Convert to DataFrame format
    # dff_light_all_freq = [pd.DataFrame(neuron) for neuron in dff_light_all_freq]
    
    # # Extract two-window averages for all frequencies combined
    # light_event_avg_freq_combined = extract_two_windows_averages(
    #     dff_light_all_freq,
    #     [opto_on_ind[i] for i in all_freq_trial_indices],
    #     [opto_off_ind[i] for i in all_freq_trial_indices],
    #     rec_hz=rec_hz,
    #     window_s=2
    # )
    
    # # Save the combined all-frequencies light event averages
    # with open(join(save_loc, 'all_light_averages_all_freq.pkl'), 'wb') as f:
    #     pickle.dump(light_event_avg_freq_combined, f)
    
    # print(f"  Combined Analysis complete. Total trials: {len(all_freq_trial_indices)}")
    
    # # End of script.
    # print("All done.")
