import copy
import math
import os
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
matplotlib.use('Qt5Agg')
import pandas as pd
from os.path import join
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm
from multiprocessing import Pool
from Tools.scripts.dutree import display
# PDF font settings
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
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
        # Assuming x_lim_calc is defined to calculate xlim based on some logic you have
        xlim = x_lim_calc(dff_hit[num])  # Update this function call as necessary
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

        # Add vertical lines using hypothetical functions `add_axvline` and `add_axvline_opto`
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
    Divides the f_signal into trials based on provided indices and run lengths.

    Parameters:
    - f_signal (list): The signal to be divided.
    - trials_ind (list): Indices for trial divisions.
    - run_len (list): Length of runs.

    Returns:
    - f_run_list (list): List of divided signals.
    - pre_runs: The signal from the beginning of each run.
    """
    # divide the f signal to trials

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
        diff = max-min
        # For each neuron, subtract the mean response from its response in each data point.
        subtract_df = ave_data[i] - min
        # For each neuron, divide the resulting value by the range.
        subtract_df /= diff
        minmax_df.append(subtract_df)

    return minmax_df
def find_min_max(data):
    max = np.nanmax(data)
    min = np.nanmin(data)
    return (max, min)

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


def calculate_dff(f_run_list, pre_trial_length, f0_window, ROOT_PATH):
    df_final_all = []

    for df_neuron in f_run_list:
        df_final = []
        trial_activity = np.asarray(df_neuron)
        f0_trials = np.mean(trial_activity[pre_trial_length - f0_window:pre_trial_length])
        # 1 sec (30 frames) before the cue
        df_final.append((trial_activity - f0_trials) / f0_trials)
        df_final = pd.DataFrame(df_final)
        df_final_all.append(df_final)

    save_location = join(ROOT_PATH, 'dff(2 sec f0)_glu_run1.npy')
    np.save(save_location, df_final_all)

    return df_final_all

def min_max_normalize(cells_data):
    minmax_df = []
    for i in range(len(cells_data)):
        # find the 1st and 99th percentile of each neuron
        min = np.percentile(cells_data.iloc[i], 1)
        max = np.percentile(cells_data.iloc[i], 99)
        diff = max-min
        # For each neuron, subtract the mean response from its response in each data point.
        subtract_df = cells_data.iloc[i] - min
        # For each neuron, divide the resulting value by the range.
        subtract_df /= diff
        minmax_df.append(subtract_df)
    minmax_df = np.asarray(minmax_df)
    return minmax_df

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def detection_histogram(dff_trials, stimulation_start, stimulation_end, rec_hz, save_loc):
    half_max_frequencies = []
    for i in range(len(dff_trials)):
        neuron_activities = dff_trials[i]

        # Example frequency map
        frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

        # Create a dictionary to collect average activities
        activity_dict = {freq: [] for freq in frequency_map.values()}

        # Process data
        for condition, activities in zip(conditions_played, neuron_activities):
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

            if min(x_data) <= popt[1] <= max(x_data):  # Check if x0 is within a realistic range
                half_max_frequencies.append(popt[1])
            else:
                print(f"Unrealistic x0 found: {popt[1]}, skipping this neuron.")
        except RuntimeError as e:
            print(f"Curve fitting failed for Neuron {i} due to: {e}")

    # Plotting the histogram of all collected half-max frequencies
    # plt.figure(figsize=(10, 6))
    # plt.hist(half_max_frequencies, bins=20, color='gray', alpha=0.7)
    # plt.axvline(0.91, color='red', linestyle='--', linewidth=1)
    # plt.text(0.91, 6,f'{0.91:.2f} Hz (psychometric curve)', color='red', ha='left', va='center', fontsize=8)
    # plt.xlabel('Stimulation Frequency (Hz) at 50% Max Response')
    # plt.ylabel('Number of Neurons')
    # plt.title('50% Response Frequencies Across Neurons')
    # plt.savefig(join(data_loc, 'half_max_frequencies.pdf'))
    # plt.show()

    plt.figure(figsize=(10, 6))
    density = gaussian_kde(half_max_frequencies)
    xs = np.linspace(0, 15, 100)
    density.covariance_factor = lambda: .2
    density._compute_covariance()
    plt.plot(xs, density(xs), color='black')
    plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    plt.axvline(x=2.1, color='black', linestyle='--')
    plt.text(2.1, 0.02, '2.1 Hz (psychometric curve)', color='red', rotation=90)
    plt.xlabel('Stimulation Frequency (Hz) at 50% Max Response')
    plt.ylabel('Density')
    plt.title('50% Response Frequencies Across Neurons')
    plt.xticks([0, 0.5, 1, 3, 5, 7.5, 10, 15])
    # Example data structure for neuron activities
    plt.savefig(join(save_loc, 'half_max_density.pdf'))
    plt.close()

def fit_neurometric_curve(dff_trials, stimulation_start, stimulation_end, conditions, save_curves):
    save_curves = join(save_loc, 'neurometric_curves_dff')
    save_pdf = join(save_curves, 'pdf')
    os.makedirs(save_curves, exist_ok=True)
    os.makedirs(save_pdf, exist_ok=True)
    all_avg = []
    for i in range(len(dff_trials)):
        neuron_activities = dff_trials[i]

        # Example frequency map
        frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

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
        sems = {freq: np.std(vals) / math.sqrt(len(vals)) if vals else 0 for freq, vals in activity_dict.items()}
        z_score_means = {freq: np.mean(vals) / np.std(vals) if vals else 0 for freq, vals in activity_dict.items()}

        # Min-max normalization of averages
        # min_val = min(averages.values())
        # max_val = max(averages.values())
        # norm_averages = {freq: (val - min_val) / (max_val - min_val) if max_val > min_val else 0 for freq, val in
        #                  averages.items()}

        # Sorting the averages for plotting
        sorted_frequencies = sorted(z_score_means.keys())
        sorted_averages = [z_score_means[freq] for freq in sorted_frequencies]
        all_avg.append(sorted_averages)
        # Creating equally spaced x-axis
        num_points = len(sorted_frequencies)
        equally_spaced_x = np.linspace(0, num_points - 1, num_points)
        # Plotting
        x_data = sorted_frequencies
        y_data = sorted_averages

        try:
            # Fit the sigmoid function to the data
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
            popt, pcov = curve_fit(sigmoid, equally_spaced_x, sorted_averages, p0, method='trf')

            # Generate x values for the curve
            x_values = np.linspace(0, num_points - 1, 100)
            # Generate y values from the fitted sigmoid function
            y_values = sigmoid(x_values, *popt)

            plt.figure(figsize=(20, 10))
            plt.plot(x_values, y_values, label='Fit', color='black')  # No scaling needed
            plt.scatter(equally_spaced_x, sorted_averages, color='black')
            # add error bars
            # plt.errorbar(equally_spaced_x, sorted_averages, yerr=[sems[freq] for freq in sorted_frequencies], fmt='o', color='black')
            plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
            plt.ylabel('Normalized df/f')
            plt.xticks(equally_spaced_x, labels=[f"{freq:.2f} Hz" for freq in sorted_frequencies])
            # plt.yticks(np.linspace(0, 1, 11))  # Setting y ticks from 0 to 1 in steps of 0.1
            plt.savefig(join(save_curves, f'Neuron {i} - Neurometric Curve.png'))
            plt.close()
        except RuntimeError as e:
            print(f"Curve fitting failed for Neuron {i} due to: {e}")
            plt.figure(figsize=(20, 10))
            plt.scatter(equally_spaced_x, sorted_averages, color='black')
            # plt.errorbar(equally_spaced_x, sorted_averages, yerr=[sems[freq] for freq in sorted_frequencies], fmt='o', color='black')
            plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
            plt.ylabel('Normalized df/f')
            plt.xticks(equally_spaced_x, labels=[f"{freq:.2f} Hz" for freq in sorted_frequencies])
            # plt.yticks(np.linspace(0, 1, 11))  # Setting y ticks from 0 to 1 in steps of 0.1
            plt.savefig(join(save_curves, f'Neuron {i} - Neurometric Curve.png'))
            plt.close()
    return all_avg

def compute_goodness_of_fit(x_array, y_array, popt):
    y_fit = sigmoid(x_array, *popt)
    residuals = y_array - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return r_squared

def compute_goodness_of_fit_gaussian(x_array, y_array, popt):
    y_fit = generalized_gaussian(x_array, *popt)
    residuals = y_array - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return r_squared

def gaussian_fwhm(x, A, mu, fwhm, offset):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

def generalized_gaussian(x, a, mu, sigma, beta, offset):
    A = a * (1 - offset)
    return A * np.exp(-np.abs((x - mu) / sigma)**beta) + offset
# This function processes one neuron (index and its data) end-to-end.
def process_neuron_neurometric(args):
    """
    Processes a single neuron: separates responses, fits both excited and inhibited curves,
    chooses the best based on R², runs 1000 shuffles (sequentially), computes thresholds,
    and saves the neuron's neurometric plot.

    Returns a dictionary with key metrics.
    """
    neuron_idx, neuron_auROC, conditions, save_dir, psyco = args

    # Frequency mapping (condition analysis_code -> stimulation frequency in Hz)
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

    # Compute binary responses for excited (> 0.5) and inhibited (< 0.5)
    excited = np.where(neuron_auROC > 0.5, 1, 0)
    inhibited = np.where(neuron_auROC < 0.5, 1, 0)

    # Build response dictionaries for excited and inhibited responses.
    response_dict_excited = {freq: {'count': 0, 'total': 0} for freq in frequency_map.values()}
    response_dict_inhibited = {freq: {'count': 0, 'total': 0} for freq in frequency_map.values()}

    for cond, exc_val, inh_val in zip(conditions, excited, inhibited):
        freq = frequency_map.get(cond, None)
        if freq is not None:
            if exc_val == 1:
                response_dict_excited[freq]['count'] += 1
            response_dict_excited[freq]['total'] += 1

            if inh_val == 1:
                response_dict_inhibited[freq]['count'] += 1
            response_dict_inhibited[freq]['total'] += 1

    # Compute the fraction-of-responses at each frequency.
    fractions_excited = {freq: (res['count'] / res['total'] if res['total'] > 0 else 0)
                         for freq, res in response_dict_excited.items()}
    fractions_inhibited = {freq: (res['count'] / res['total'] if res['total'] > 0 else 0)
                           for freq, res in response_dict_inhibited.items()}

    # Use sorted frequencies as x-data.
    sorted_frequencies = sorted(fractions_excited.keys())
    x_data_all = np.array(sorted_frequencies)
    y_data_excited = np.array([fractions_excited[f] for f in sorted_frequencies])
    y_data_inhibited = np.array([fractions_inhibited[f] for f in sorted_frequencies])

    # Set up initial guesses and bounds for the sigmoid fit.
    bounds = ([0, min(x_data_all), 0, 0], [1, max(x_data_all), 10, 1])
    p0_excited = [np.max(y_data_excited), np.median(x_data_all), 1, np.min(y_data_excited)]
    p0_inhibited = [np.max(y_data_inhibited), np.median(x_data_all), 1, np.min(y_data_inhibited)]

    # Fit the sigmoid to both excited and inhibited responses.
    try:
        popt_excited, _ = curve_fit(sigmoid, x_data_all, y_data_excited, p0=p0_excited,
                                    method='trf', bounds=bounds)
        r_squared_excited = compute_goodness_of_fit(x_data_all, y_data_excited, popt_excited)
    except RuntimeError as e:
        r_squared_excited = 0
        popt_excited = None

    try:
        popt_inhibited, _ = curve_fit(sigmoid, x_data_all, y_data_inhibited, p0=p0_inhibited,
                                      method='trf', bounds=bounds)
        r_squared_inhibited = compute_goodness_of_fit(x_data_all, y_data_inhibited, popt_inhibited)
    except RuntimeError as e:
        r_squared_inhibited = 0
        popt_inhibited = None

    # try to fit also gaussians to the data
    # Define bounds for [A, mu, fwhm, offset]
    # bounds = ([0, min(x_data_all), 0, 0], [1, max(x_data_all), 10, 1])
    # # Update initial guesses for FWHM (using 2.355 if sigma was 1)
    # p0_excited = [np.max(y_data_excited), np.median(x_data_all), 2.355, np.min(y_data_excited)]
    # p0_inhibited = [np.max(y_data_inhibited), np.median(x_data_all), 2.355, np.min(y_data_inhibited)]
    bounds_gauss = ([0, min(x_data_all[1:]), 0, 0, 0],
              [1, max(x_data_all[:-1]), 10, 2, 1])
    # Use beta=2 as an initial guess (which recovers the standard Gaussian)
    p0_excited_gauss = [np.max(y_data_excited), np.median(x_data_all), 1, 2, np.min(y_data_excited)]
    p0_inhibited_gauss = [np.max(y_data_inhibited), np.median(x_data_all), 1, 2, np.min(y_data_inhibited)]
    try:
        popt_excited_gauss, _ = curve_fit(generalized_gaussian, x_data_all, y_data_excited, p0=p0_excited_gauss,
                                    method='trf', bounds=bounds_gauss)
        r_squared_excited_gauss = compute_goodness_of_fit_gaussian(x_data_all, y_data_excited, popt_excited_gauss)
    except RuntimeError as e:
        r_squared_excited_gauss = 0
        popt_excited_gauss = None

    try:
        popt_inhibited_gauss, _ = curve_fit(generalized_gaussian, x_data_all, y_data_inhibited, p0=p0_inhibited_gauss,
                                        method='trf', bounds=bounds_gauss)
        r_squared_inhibited_gauss = compute_goodness_of_fit_gaussian(x_data_all, y_data_inhibited, popt_inhibited_gauss)
    except RuntimeError as e:
        r_squared_inhibited_gauss = 0
        popt_inhibited_gauss = None


    # Choose the best-fit curve (the one with the higher R²).
    # find the best fit
    chosen_r_squared = max(r_squared_excited, r_squared_inhibited, r_squared_excited_gauss, r_squared_inhibited_gauss)
    if r_squared_excited == chosen_r_squared:
    # if r_squared_excited >= r_squared_inhibited:
        chosen_popt = popt_excited
        chosen_r_squared = r_squared_excited
        chosen_y_data = y_data_excited
        chosen_p0 = p0_excited
        chosen_type = 'Excited'
        fit_type = 'sigmoid'
    elif r_squared_inhibited == chosen_r_squared:
        chosen_popt = popt_inhibited
        chosen_r_squared = r_squared_inhibited
        chosen_y_data = y_data_inhibited
        chosen_p0 = p0_inhibited
        chosen_type = 'Inhibited'
        fit_type = 'sigmoid'
    elif r_squared_excited_gauss == chosen_r_squared:
        chosen_popt = popt_excited_gauss
        chosen_r_squared = r_squared_excited_gauss
        chosen_y_data = y_data_excited
        chosen_p0 = p0_excited_gauss
        chosen_type = 'Excited'
        fit_type = 'gaussian'
    elif r_squared_inhibited_gauss == chosen_r_squared:
        chosen_popt = popt_inhibited_gauss
        chosen_r_squared = r_squared_inhibited_gauss
        chosen_y_data = y_data_inhibited
        chosen_p0 = p0_inhibited_gauss
        chosen_type = 'Inhibited'
        fit_type = 'gaussian'

    if chosen_r_squared == r_squared_inhibited or chosen_r_squared == r_squared_excited:
        # Run 1000 shuffles sequentially.
        total_shuffles = 1000
        r_squared_shuffled = []
        for _ in range(total_shuffles):
            shuffled_y_data = np.random.permutation(chosen_y_data)
            try:
                popt_shuffled, _ = curve_fit(sigmoid, x_data_all, shuffled_y_data,
                                             p0=chosen_p0, method='trf', bounds=bounds)
                r_sq = compute_goodness_of_fit(x_data_all, shuffled_y_data, popt_shuffled)
            except RuntimeError:
                r_sq = 0
            r_squared_shuffled.append(r_sq)
        r_squared_shuffled = np.array(r_squared_shuffled)
        percentile_r = np.percentile(r_squared_shuffled, 95)
        # After choosing the best-fit curve, check if a valid solution was obtained:
        if chosen_popt is None:
            print(f"Neuron {neuron_idx}: Curve fitting failed, skipping neuron.")
            return {
                'neuron_idx': neuron_idx,
                'r_squared': 0,
                'half_max_frequency': np.nan,
                'threshold_half_max': np.nan,
                'threshold_dprime': np.nan,
                'significant': False
            }
        # Generate the fitted curve for plotting.
        x_values = np.linspace(min(x_data_all), max(x_data_all), 100)
        y_values = sigmoid(x_values, *chosen_popt)

    else:
        # Run 1000 shuffles sequentially.
        total_shuffles = 1000
        r_squared_shuffled = []
        for _ in range(total_shuffles):
            shuffled_y_data = np.random.permutation(chosen_y_data)
            try:
                popt_shuffled, _ = curve_fit(generalized_gaussian, x_data_all, shuffled_y_data,
                                             p0=chosen_p0, method='trf', bounds=bounds_gauss)
                r_sq = compute_goodness_of_fit_gaussian(x_data_all, shuffled_y_data, popt_shuffled)
            except RuntimeError:
                r_sq = 0
            r_squared_shuffled.append(r_sq)
        r_squared_shuffled = np.array(r_squared_shuffled)
        percentile_r = np.percentile(r_squared_shuffled, 95)
        # After choosing the best-fit curve, check if a valid solution was obtained:
        if chosen_popt is None:
            print(f"Neuron {neuron_idx}: Curve fitting failed, skipping neuron.")
            return {
                'neuron_idx': neuron_idx,
                'r_squared': 0,
                'half_max_frequency': np.nan,
                'threshold_half_max': np.nan,
                'threshold_dprime': np.nan,
                'significant': False
            }
        # Generate the fitted curve for plotting.
        x_values = np.linspace(min(x_data_all), max(x_data_all), 100)
        y_values = generalized_gaussian(x_values, *chosen_popt)

    # Determine the half-max threshold (the frequency at which the response reaches half of its dynamic range).
    half_max = (np.max(y_values) - np.min(y_values)) / 2
    diff_value = np.min(y_values) + half_max
    threshold_for_half_max = x_values[np.where(y_values >= diff_value)[0][0]]

    # Define significance based on whether the neuron's fit R² exceeds the 99th percentile of shuffles.
    significant = chosen_r_squared > percentile_r

    # Compute d' values using the 0 Hz (or lowest frequency) response as the false alarm rate.
    # FA_rate = chosen_y_data[0]
    # epsilon = 1e-4  # To avoid infinite z-scores.
    # FA_rate_clamped = np.clip(FA_rate, epsilon, 1 - epsilon)
    # dprime_vals = []
    # for HR in y_values:
    #     HR_clamped = np.clip(HR, epsilon, 1 - epsilon)
    #     dprime = norm.ppf(HR_clamped) - norm.ppf(FA_rate_clamped)
    #     dprime_vals.append(dprime)
    # dprime_vals = np.array(dprime_vals)
    # desired_dprime = 1.0
    # idx = np.where(dprime_vals >= desired_dprime)[0]
    # if idx.size == 0:
    #     threshold_for_dprime_1 = np.nan
    # else:
    #     threshold_for_dprime_1 = x_values[idx[0]]

    # Save the neurometric curve plot for this neuron.
    neuron_curves_dir = join(save_dir, 'neurometric_curves_auROC')
    neuron_pdf_dir = join(neuron_curves_dir, 'pdf')
    os.makedirs(neuron_curves_dir, exist_ok=True)
    os.makedirs(neuron_pdf_dir, exist_ok=True)
    #
    # plt.figure(figsize=(20, 10))
    # plt.plot(x_values, y_values * 100, label='Fit', color='black')
    # plt.scatter(x_data_all, chosen_y_data * 100, color='black')
    # plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    # plt.ylabel(f'Fraction {chosen_type.lower()} (%)')
    # plt.xticks(x_data_all, labels=[f"{freq:.2f} Hz" for freq in x_data_all])
    # plt.yticks(np.linspace(0, 100, 11))
    # plt.axvline(x=chosen_popt[1], color='red', linestyle='--')
    # plt.text(chosen_popt[1], 50, f'{chosen_popt[1]:.2f} Hz', color='red', ha='right', va='center')
    # plt.axvline(x=threshold_for_half_max, color='green', linestyle='--')
    # plt.text(threshold_for_half_max, 50, f'{threshold_for_half_max:.2f} Hz', color='green', ha='left', va='center')
    # # plt.axvline(x=threshold_for_dprime_1, color='blue', linestyle='--')
    # # plt.text(threshold_for_dprime_1, 50, f'{threshold_for_dprime_1:.2f} Hz', color='blue', ha='right', va='bottom')
    # plt.legend()
    # plt.title(f'Neuron {neuron_idx} Neurometric Curve ({chosen_type})\n'
    #           f'(R² = {chosen_r_squared:.2f}, shuffled R² = {percentile_r:.2f})')
    # plt.savefig(join(neuron_curves_dir, f'Neuron {neuron_idx} - Neurometric Curve.png'))
    # plt.savefig(join(neuron_pdf_dir, f'Neuron {neuron_idx} - Neurometric Curve.pdf'))
    # plt.close()

    # --- Plot ---
    y_values_percent = y_values * 100
    chosen_y_data_percent = chosen_y_data * 100

    plt.plot(x_values, y_values_percent, label='Fit', color='black')
    plt.scatter(x_data_all, chosen_y_data_percent, color='black')

    plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    plt.ylabel(f'Fraction {chosen_type.lower()} (%)')
    plt.xticks(x_data_all, labels=[f"{freq:.2f} Hz" for freq in x_data_all])
    plt.yticks(np.linspace(0, 100, 11))

    # Threshold markers
    plt.axvline(x=chosen_popt[1], color='red', linestyle='--')
    plt.text(chosen_popt[1], 50, f'{chosen_popt[1]:.2f} Hz', color='red', ha='right', va='center')

    plt.axvline(x=threshold_for_half_max, color='green', linestyle='--')
    plt.text(threshold_for_half_max, 50, f'{threshold_for_half_max:.2f} Hz', color='green', ha='left', va='center')

    plt.legend()
    plt.title(f'Neuron {neuron_idx} Neurometric Curve ({chosen_type})\n'
              f'(R² = {chosen_r_squared:.2f}, shuffled R² = {percentile_r:.2f})')

    plt.savefig(join(neuron_curves_dir, f'Neuron {neuron_idx} - Neurometric Curve.png'))
    plt.savefig(join(neuron_pdf_dir, f'Neuron {neuron_idx} - Neurometric Curve.pdf'))
    plt.close()

    # Return key metrics from this neuron.
    return {
        'neuron_idx': neuron_idx,
        'r_squared': chosen_r_squared,
        'half_max_frequency': chosen_popt[1],
        'threshold_half_max': threshold_for_half_max,
        'fit_type': fit_type,
        'significant': significant
    }


# Main function that parallelizes over neurons.
def auROC_neurometric_curve(auROC, conditions, psyco, save_dir):
    """
    Processes all neurons in parallel. For each neuron, it fits a neurometric sigmoid curve
    (separately for excited and inhibited responses), selects the best based on R²,
    performs shuffling to determine significance, saves the neuron's plot, and finally
    generates density plots from neurons with significant fits.

    Parameters:
        auROC (list or array): List/array of auROC arrays (one per neuron), thresholded (NaN for non-significant).
        conditions (list): List of condition codes corresponding to each trial.
        psyco (float): Frequency corresponding to the psychometric curve (for reference in density plots).
        save_dir (str): Directory in which to save all output plots.
    """
    # Build argument list for each neuron.
    args_list = [(i, auROC[i], conditions, save_dir, psyco) for i in range(len(auROC))]

    # Process neurons in parallel.
    with Pool() as pool:
        results = pool.map(process_neuron_neurometric, args_list)

    # Gather overall metrics.
    half_max_frequencies = []
    half_diff = []  # In this example we use threshold_half_max as half-difference.
    d_prime_all = []
    significant_neurons = []
    significant_neurons_gauss = []
    r_squared_all = []

    for res in results:
        r_squared_all.append(res['r_squared'])
        if res['significant'] and res['fit_type'] == 'sigmoid':
            half_diff.append(res['threshold_half_max'])
            # d_prime_all.append(res['threshold_dprime'])
            significant_neurons.append(res['neuron_idx'])
        elif res['significant'] and res['fit_type'] == 'gaussian':
            half_max_frequencies.append(res['half_max_frequency'])
            significant_neurons_gauss.append(res['neuron_idx'])


    # 1. Density of half-max frequencies.
    plt.figure(figsize=(10, 6))
    if len(half_max_frequencies) > 1:
        density = gaussian_kde(half_max_frequencies)
        xs = np.linspace(0, 15, 100)
        density.covariance_factor = lambda: 0.2
        density._compute_covariance()
        plt.plot(xs, density(xs), color='black')
        plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    else:
        print("Not enough half-max frequencies for density estimation.")
    plt.axvline(x=psyco, color='black', linestyle='--')
    plt.text(psyco, 0.02, f'{psyco} Hz (psychometric curve)', color='red', rotation=90)
    plt.xlabel('Stimulation Frequency (Hz) at 50% Max Response')
    plt.ylabel('Density')
    plt.title(f'50% Response Frequencies Across Neurons (n={len(half_max_frequencies)})')
    plt.xticks([0, 0.5, 1, 3, 5, 7.5, 10, 15])
    plt.savefig(join(save_dir, 'half_max_density_auROC.pdf'))
    plt.close()

    # 2. Density of half-difference (threshold for half-max) values.
    plt.figure(figsize=(10, 6))
    if len(half_diff) > 1:
        density = gaussian_kde(half_diff)
        xs = np.linspace(0, 15, 100)
        density.covariance_factor = lambda: 0.2
        density._compute_covariance()
        plt.plot(xs, density(xs), color='black')

        plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    else:
        print("Not enough half-difference values for density estimation.")
    plt.xticks([0, 0.5, 1, 3, 5, 7.5, 10, 15])
    plt.axvline(x=psyco, color='black', linestyle='--')
    plt.text(psyco, 0.02, f'{psyco} Hz (psychometric curve)', color='red', rotation=90)
    plt.xlabel('Half-Difference between Max and Min (%)')
    plt.ylabel('Density')
    plt.title(f'Half-Difference Across Neurons (n={len(half_diff)})')
    plt.savefig(join(save_dir, 'half_diff_density_auROC.pdf'))
    plt.close()

    # plot histogram of the half-difference values normalized by the bahavioural threshold
    half_diff_norm = np.array(half_diff)
    half_diff_norm = half_diff_norm[~np.isnan(half_diff_norm)]
    half_diff_norm = half_diff_norm / psyco
    plt.figure(figsize=(10, 6))
    plt.hist(half_diff_norm, bins=20, color='gray', alpha=0.7)
    plt.xlabel('Half-Difference between Max and Min (%)')
    plt.ylabel('Number of Neurons')
    plt.title('Half-Difference Across Neurons')
    plt.savefig(join(save_dir, 'half_diff_norm_auROC.pdf'))
    plt.close()

    # plot density of the half-difference values normalized by the bahavioural threshold
    plt.figure(figsize=(10, 6))
    if len(half_diff_norm) > 1:
        density = gaussian_kde(half_diff_norm)
        xs = np.linspace(0, max(half_diff_norm)+0.5, 100)
        density.covariance_factor = lambda: 0.2
        density._compute_covariance()
        plt.plot(xs, density(xs), color='black')
        plt.fill_between(xs, 0, density(xs), color='gray', alpha=0.2)
    else:
        print("Not enough half-difference values for density estimation.")
    plt.axvline(x=1, color='black', linestyle='--')
    plt.text(1, 0.02, '1 (psychometric curve)', color='red', rotation=90)
    plt.xlabel('Half-Difference between Max and Min (%)')
    plt.ylabel('Density')
    plt.title(f'Half-Difference Across Neurons Norm (n={len(half_diff_norm)})')
    plt.savefig(join(save_dir, 'half_diff_density_norm_auROC.pdf'))
    plt.close()

    # 3. R² distribution:
    plt.figure(figsize=(10, 6))
    plt.hist(r_squared_all, bins=20, color='gray', alpha=0.7)
    plt.xlabel('R² Value')
    plt.ylabel('Number of Neurons')
    plt.title('R² Values Across Neurons')
    plt.savefig(join(save_dir, 'r_squared_distribution.png'))
    plt.close()

    # save the half-difference values
    np.save(join(save_dir, 'half_diff_values_sigmoid.npy'), half_diff)
    np.save(join(save_dir, 'half_diff_norm_values_sigmoid.npy'), half_diff_norm)
    np.save(join(save_dir, 'half_max_values_gaussian.npy'), half_max_frequencies)
    np.save(join(save_dir, 'r_squared_values.npy'), r_squared_all)
    np.save(join(save_dir, 'significant_neurons.npy'), significant_neurons)
    np.save(join(save_dir, 'significant_neurons_gauss.npy'), significant_neurons_gauss)
    
    # Export excitatory vs inhibitory statistics to CSV
    export_exc_inh_statistics(auROC, significant_neurons, significant_neurons_gauss, save_dir)
    
    # Return the significant neuron lists for further analysis
    return significant_neurons, significant_neurons_gauss


def export_exc_inh_statistics(auROC, significant_neurons, significant_neurons_gauss, save_dir):
    """
    Export statistics of excitatory vs inhibitory neurons for sigmoid and gaussian categories.
    
    Parameters:
        auROC: List of auROC arrays for all neurons
        significant_neurons: List of significant sigmoid neuron indices
        significant_neurons_gauss: List of significant gaussian neuron indices  
        save_dir: Directory to save CSV file
    """
    import pandas as pd
    import csv
    
    print("Calculating excitatory vs inhibitory statistics...")
    
    # Initialize counters
    stats = {
        'sigmoid': {'exc_count': 0, 'inh_count': 0, 'total': 0},
        'gaussian': {'exc_count': 0, 'inh_count': 0, 'total': 0},
        'all_neurons': {'exc_count': 0, 'inh_count': 0, 'total': 0}
    }
    
    # Function to classify neuron as excitatory or inhibitory
    def classify_neuron(neuron_auroc):
        """Classify neuron based on auROC values."""
        valid_auroc = neuron_auroc[~np.isnan(neuron_auroc)]
        if len(valid_auroc) == 0:
            return 'unknown'
        
        mean_auroc = np.mean(valid_auroc)
        if mean_auroc > 0.5:
            return 'excitatory'
        elif mean_auroc < 0.5:
            return 'inhibitory'
        else:
            return 'neutral'
    
    # Analyze all neurons
    for i, neuron_auroc in enumerate(auROC):
        classification = classify_neuron(neuron_auroc)
        
        if classification == 'excitatory':
            stats['all_neurons']['exc_count'] += 1
        elif classification == 'inhibitory':
            stats['all_neurons']['inh_count'] += 1
        
        stats['all_neurons']['total'] += 1
        
        # Check if neuron is in significant categories
        if i in significant_neurons:
            if classification == 'excitatory':
                stats['sigmoid']['exc_count'] += 1
            elif classification == 'inhibitory':
                stats['sigmoid']['inh_count'] += 1
            stats['sigmoid']['total'] += 1
            
        if i in significant_neurons_gauss:
            if classification == 'excitatory':
                stats['gaussian']['exc_count'] += 1
            elif classification == 'inhibitory':
                stats['gaussian']['inh_count'] += 1
            stats['gaussian']['total'] += 1
    
    # Calculate fractions
    results = []
    for category in ['all_neurons', 'sigmoid', 'gaussian']:
        data = stats[category]
        total = data['total']
        
        if total > 0:
            exc_fraction = data['exc_count'] / total
            inh_fraction = data['inh_count'] / total
        else:
            exc_fraction = 0
            inh_fraction = 0
        
        results.append({
            'Category': category.replace('_', ' ').title(),
            'Total_Neurons': total,
            'Excitatory_Count': data['exc_count'],
            'Inhibitory_Count': data['inh_count'],
            'Excitatory_Fraction': round(exc_fraction, 4),
            'Inhibitory_Fraction': round(inh_fraction, 4),
            'Excitatory_Percentage': round(exc_fraction * 100, 2),
            'Inhibitory_Percentage': round(inh_fraction * 100, 2)
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = join(save_dir, 'excitatory_inhibitory_statistics.csv')
    df.to_csv(csv_path, index=False)
    
    # Also save a detailed version with mouse information
    csv_detailed_path = join(save_dir, 'excitatory_inhibitory_detailed.csv')
    with open(csv_detailed_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Analysis', 'Value'])
        writer.writerow(['Total Neurons Analyzed', stats['all_neurons']['total']])
        writer.writerow([''])
        writer.writerow(['SIGMOID NEURONS'])
        writer.writerow(['Total Sigmoid', stats['sigmoid']['total']])
        writer.writerow(['Sigmoid Excitatory', stats['sigmoid']['exc_count']])
        writer.writerow(['Sigmoid Inhibitory', stats['sigmoid']['inh_count']])
        writer.writerow(['Sigmoid Exc Fraction', round(stats['sigmoid']['exc_count']/max(1, stats['sigmoid']['total']), 4)])
        writer.writerow(['Sigmoid Inh Fraction', round(stats['sigmoid']['inh_count']/max(1, stats['sigmoid']['total']), 4)])
        writer.writerow([''])
        writer.writerow(['GAUSSIAN NEURONS'])
        writer.writerow(['Total Gaussian', stats['gaussian']['total']])
        writer.writerow(['Gaussian Excitatory', stats['gaussian']['exc_count']])
        writer.writerow(['Gaussian Inhibitory', stats['gaussian']['inh_count']])
        writer.writerow(['Gaussian Exc Fraction', round(stats['gaussian']['exc_count']/max(1, stats['gaussian']['total']), 4)])
        writer.writerow(['Gaussian Inh Fraction', round(stats['gaussian']['inh_count']/max(1, stats['gaussian']['total']), 4)])
        writer.writerow([''])
        writer.writerow(['INDIVIDUAL NEURON CLASSIFICATIONS'])
        writer.writerow(['Neuron_Index', 'Category', 'Classification', 'Mean_auROC'])
        
        # Add individual neuron data
        for i, neuron_auroc in enumerate(auROC):
            classification = classify_neuron(neuron_auroc)
            valid_auroc = neuron_auroc[~np.isnan(neuron_auroc)]
            mean_auroc = np.mean(valid_auroc) if len(valid_auroc) > 0 else np.nan
            
            category = 'None'
            if i in significant_neurons:
                category = 'Sigmoid'
            elif i in significant_neurons_gauss:
                category = 'Gaussian'
                
            writer.writerow([i, category, classification, round(mean_auroc, 4) if not np.isnan(mean_auroc) else 'NaN'])
    
    print(f"Excitatory/Inhibitory statistics saved to:")
    print(f"  Summary: {csv_path}")
    print(f"  Detailed: {csv_detailed_path}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("EXCITATORY vs INHIBITORY NEURON STATISTICS")
    print("="*50)
    for result in results:
        print(f"\n{result['Category']}:")
        print(f"  Total: {result['Total_Neurons']}")
        print(f"  Excitatory: {result['Excitatory_Count']} ({result['Excitatory_Percentage']}%)")
        print(f"  Inhibitory: {result['Inhibitory_Count']} ({result['Inhibitory_Percentage']}%)")
    print("="*50)


def create_all_neuron_activity_plots(dff_run_list, conditions, significant_neurons, 
                                     significant_neurons_gauss, save_dir, 
                                     pre_trial_length, rec_hz):
    """
    Create heatmaps and line plots of neural activity for ALL neurons with classification labels.
    
    Parameters:
        dff_run_list: List of neural activity data [neuron_idx][trial_idx] -> time series
        conditions: Array of condition codes for each trial
        significant_neurons: List of significant sigmoid neuron indices
        significant_neurons_gauss: List of significant gaussian neuron indices  
        save_dir: Directory to save plots
        pre_trial_length: Length of pre-trial period in frames
        rec_hz: Recording frequency in Hz
    """
    
    print("\n" + "="*60)
    print("CREATING ALL NEURON ACTIVITY VISUALIZATIONS")
    print("="*60)
    
    # Create frequency mapping (excluding 0 Hz)
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    unique_frequencies = sorted([freq for freq in frequency_map.values() if freq != 0])  # Remove 0 Hz
    
    # Define stimulation period (2-5 seconds after trial start)
    stim_start_frame = pre_trial_length + 2 * rec_hz
    stim_end_frame = pre_trial_length + 5 * rec_hz
    
    # Create output directory
    output_dir = join(save_dir, 'all_neuron_activity_plots_baseline_sub')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots for all neurons
    total_neurons = len(dff_run_list)
    print(f"Creating plots for {total_neurons} neurons")
    
    for neuron_idx in range(total_neurons):
        # Determine neuron classification
        if neuron_idx in significant_neurons:
            classification = 'Sigmoid'
        elif neuron_idx in significant_neurons_gauss:
            classification = 'Gaussian'
        else:
            classification = 'Neither'
            
        print(f"Processing neuron {neuron_idx} ({classification})")
        
        # Get neuron data
        neuron_data = dff_run_list[neuron_idx]
        
        # Create the plot
        create_individual_neuron_plot_with_classification(
            neuron_data, conditions, neuron_idx, classification,
            frequency_map, unique_frequencies, stim_start_frame,
            stim_end_frame, output_dir, pre_trial_length, rec_hz
        )
    
    print("All neuron activity plots completed!")


def create_neuron_type_plots(dff_run_list, conditions, neuron_indices, fit_type, 
                            frequency_map, unique_frequencies, stim_start_frame, 
                            stim_end_frame, save_dir, pre_trial_length, rec_hz):
    """Create plots for a specific neuron type (sigmoid or gaussian)."""
    
    # Create output directory
    output_dir = join(save_dir, f'{fit_type}_neuron_activity_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    for neuron_idx in neuron_indices:
        if neuron_idx >= len(dff_run_list):
            print(f"Neuron {neuron_idx} not found in data")
            continue
            
        print(f"Processing {fit_type} neuron {neuron_idx}")
        
        # Get neuron data
        neuron_data = dff_run_list[neuron_idx]
        
        # Create the plot
        create_individual_neuron_plot(neuron_data, conditions, neuron_idx, fit_type,
                                    frequency_map, unique_frequencies, stim_start_frame,
                                    stim_end_frame, output_dir, pre_trial_length, rec_hz)


def create_individual_neuron_plot_with_classification(neuron_data, conditions, neuron_idx, classification,
                                                     frequency_map, unique_frequencies, stim_start_frame,
                                                     stim_end_frame, output_dir, pre_trial_length, rec_hz):
    """Create heatmap and line plot for a single neuron with classification label."""

    # Organize data by frequency (excluding 0 Hz)
    frequency_full_traces = {freq: [] for freq in unique_frequencies}

    for trial_idx, condition in enumerate(conditions):
        if trial_idx < len(neuron_data):
            freq = frequency_map.get(condition, None)
            if freq is not None and freq != 0:  # Exclude 0 Hz trials
                trial_trace = np.array(neuron_data[trial_idx])

                # Store full traces for both heatmap and line plot
                # Ensure trace is long enough and convert to numpy array
                if len(trial_trace) > stim_end_frame:
                    # Convert to numpy array and ensure it's 1D
                    trace_array = np.array(trial_trace).flatten()
                    if len(trace_array) > stim_end_frame:
                        frequency_full_traces[freq].append(trace_array)

    # filter each trial trace to reduce noise using a simple moving average
    def moving_average(data, window_size):
        """Apply a simple moving average to the data."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    for freq in unique_frequencies:
        for i, trace in enumerate(frequency_full_traces[freq]):
            trace = moving_average(trace, window_size=2)
            frequency_full_traces[freq][i] = trace

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # 1. HEATMAP: All trials organized by frequency
    create_activity_heatmap_custom(frequency_full_traces, unique_frequencies, ax1, neuron_idx,
                                  classification, pre_trial_length, rec_hz, stim_start_frame-1 * rec_hz, stim_end_frame)

    # 2. LINE PLOT: Mean activity time course across trials for each frequency
    create_activity_lineplot_custom(frequency_full_traces, unique_frequencies, ax2, neuron_idx, classification,
                                   pre_trial_length, rec_hz, stim_start_frame-1 * rec_hz, stim_end_frame)

    plt.tight_layout()

    # Save plot
    filename = f'neuron_{neuron_idx}_{classification}_activity_analysis.pdf'
    plt.savefig(join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {filename}")


def create_individual_neuron_plot(neuron_data, conditions, neuron_idx, fit_type,
                                 frequency_map, unique_frequencies, stim_start_frame,
                                 stim_end_frame, output_dir, pre_trial_length, rec_hz):
    """Create heatmap and line plot for a single neuron (legacy function)."""
    
    # Organize data by frequency
    frequency_full_traces = {freq: [] for freq in unique_frequencies}
    
    for trial_idx, condition in enumerate(conditions):
        if trial_idx < len(neuron_data):
            freq = frequency_map.get(condition, None)
            if freq is not None:
                trial_trace = np.array(neuron_data[trial_idx])
                
                # Store full traces for both heatmap and line plot
                # Ensure trace is long enough and convert to numpy array
                if len(trial_trace) > stim_end_frame:
                    # Convert to numpy array and ensure it's 1D
                    trace_array = np.array(trial_trace).flatten()
                    if len(trace_array) > stim_end_frame:
                        frequency_full_traces[freq].append(trace_array)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. HEATMAP: All trials organized by frequency
    create_activity_heatmap(frequency_full_traces, unique_frequencies, ax1, neuron_idx, 
                          fit_type, pre_trial_length, rec_hz, stim_start_frame, stim_end_frame)
    
    # 2. LINE PLOT: Mean activity time course across trials for each frequency
    create_activity_lineplot(frequency_full_traces, unique_frequencies, ax2, neuron_idx, fit_type,
                            pre_trial_length, rec_hz, stim_start_frame, stim_end_frame)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{fit_type}_neuron_{neuron_idx}_activity_analysis.pdf'
    plt.savefig(join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def create_activity_heatmap(frequency_full_traces, unique_frequencies, ax, neuron_idx, 
                          fit_type, pre_trial_length, rec_hz, stim_start_frame, stim_end_frame):
    """Create heatmap showing all trials organized by frequency."""
    
    # First, find the minimum length across ALL traces to standardize
    min_length = float('inf')
    total_traces = 0
    
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            for trace in traces:
                min_length = min(min_length, len(trace))
                total_traces += 1
    
    if total_traces == 0 or min_length == float('inf') or min_length < 10:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Collect all traces for heatmap with standardized length and frequency separation
    all_traces = []
    trial_labels = []
    frequency_boundaries = []  # To track where each frequency section ends
    current_row = 0
    
    for freq_idx, freq in enumerate(unique_frequencies):
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            for i, trace in enumerate(traces):
                # Standardize all traces to the same length
                standardized_trace = np.array(trace[:min_length])
                all_traces.append(standardized_trace)
                trial_labels.append(f'{freq}Hz_T{i+1}')
                current_row += 1
            
            # Add frequency boundary
            if freq_idx < len(unique_frequencies) - 1:  # Don't add after last frequency
                frequency_boundaries.append(current_row - 0.5)
    
    if len(all_traces) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create heatmap data matrix - now all traces have the same length
    heatmap_data = np.array(all_traces)
    
    # Create heatmap with reduced y-tick labels for better readability
    # Show only every 3rd trial label or frequency labels
    y_tick_positions = []
    y_tick_labels = []
    
    current_pos = 0
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            # Add frequency label at the middle of its section
            mid_pos = current_pos + len(traces) / 2 - 0.5
            y_tick_positions.append(mid_pos)
            y_tick_labels.append(f'{freq} Hz')
            current_pos += len(traces)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                yticklabels=False,  # Turn off individual trial labels
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Activity (z-score)'},
                ax=ax)
    
    # Add frequency boundaries as horizontal lines
    for boundary in frequency_boundaries:
        ax.axhline(y=boundary, color='white', linewidth=2, alpha=0.8)
    
    # Add custom y-ticks for frequencies
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)
    
    # Add vertical lines for important time points
    time_points = {
        'Trial Start': pre_trial_length,
        'Stim Start': stim_start_frame, 
        'Stim End': stim_end_frame
    }
    
    for label, timepoint in time_points.items():
        if timepoint < heatmap_data.shape[1]:
            ax.axvline(x=timepoint, color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(timepoint, -0.5, label, rotation=45, ha='right', va='top', 
                   fontsize=8, color='black')
    
    # Format x-axis to show time in seconds
    max_frames = heatmap_data.shape[1]
    time_ticks = np.arange(0, max_frames, rec_hz * 2)  # Every 2 seconds
    time_labels = [f'{int(t/rec_hz)}s' for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    
    ax.set_title(f'{fit_type.capitalize()} Neuron {neuron_idx}\nActivity Heatmap: Trials × Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Trials (by Frequency)')


def create_activity_heatmap_custom(frequency_full_traces, unique_frequencies, ax, neuron_idx, 
                                  classification, pre_trial_length, rec_hz, stim_start_frame, stim_end_frame):
    """Create heatmap showing all trials organized by frequency with custom colormap."""
    
    # Custom diverging palette: left=blue (h=240), right=orange (h=20), white center
    cmap = sns.diverging_palette(
        240,   # hue for blue
        20,    # hue for orange
        s=100, # saturation
        l=50,  # lightness
        sep=1,
        center="light", # makes the midpoint white (instead of dark)
        as_cmap=True
    )
    
    # First, find the minimum length across ALL traces to standardize
    min_length = float('inf')
    total_traces = 0
    
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            for trace in traces:
                min_length = min(min_length, len(trace))
                total_traces += 1
    
    if total_traces == 0 or min_length == float('inf') or min_length < 10:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Collect all traces for heatmap with standardized length and frequency separation
    all_traces = []
    trial_labels = []
    frequency_boundaries = []  # To track where each frequency section ends
    current_row = 0
    
    for freq_idx, freq in enumerate(unique_frequencies):
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            for i, trace in enumerate(traces):
                # Standardize all traces to the same length
                standardized_trace = np.array(trace[:min_length])
                all_traces.append(standardized_trace)
                trial_labels.append(f'{freq}Hz_T{i+1}')
                current_row += 1
            
            # Add frequency boundary
            if freq_idx < len(unique_frequencies) - 1:  # Don't add after last frequency
                frequency_boundaries.append(current_row - 0.5)
    
    if len(all_traces) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create heatmap data matrix - now all traces have the same length
    heatmap_data = np.array(all_traces)
    
    # Custom normalization that ensures 0 is always white
    from matplotlib.colors import TwoSlopeNorm
    vmin = -8#np.nanmin(heatmap_data) + 1
    vmax = 1.8#np.nanmax(heatmap_data) - 1.5
    
    # Create asymmetric normalization centered at 0
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    
    # Create heatmap with reduced y-tick labels for better readability
    y_tick_positions = []
    y_tick_labels = []
    
    current_pos = 0
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            # Add frequency label at the middle of its section
            mid_pos = current_pos + len(traces) / 2 - 0.5
            y_tick_positions.append(mid_pos)
            y_tick_labels.append(f'{freq} Hz')
            current_pos += len(traces)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, 
                   aspect='auto',
                   cmap=cmap, 
                   norm=norm,
                   interpolation='nearest')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Activity (z-score)', fraction=0.046, pad=0.04)
    
    # Add frequency boundaries as horizontal lines
    for boundary in frequency_boundaries:
        ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.8)
    
    # Add custom y-ticks for frequencies
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)
    
    # Add vertical lines for important time points
    time_points = {
        'Trial Start': pre_trial_length,
        'Stim Start': stim_start_frame,
        'Stim End': stim_end_frame
    }
    
    for label, timepoint in time_points.items():
        if timepoint < heatmap_data.shape[1]:
            ax.axvline(x=timepoint, color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(timepoint, -0.5, label, rotation=45, ha='right', va='top', 
                   fontsize=8, color='black')
    
    # Format x-axis to show time in seconds
    max_frames = heatmap_data.shape[1]
    time_ticks = np.arange(0, max_frames, rec_hz * 2)  # Every 2 seconds
    time_labels = [f'{int(t/rec_hz)}s' for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    
    ax.set_title(f'Neuron {neuron_idx} ({classification})\nActivity Heatmap: Trials × Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Trials (by Frequency)')


def create_activity_lineplot(frequency_full_traces, unique_frequencies, ax, neuron_idx, fit_type, 
                           pre_trial_length, rec_hz, stim_start_frame, stim_end_frame):
    """Create line plot showing mean activity time course across trials for each frequency."""
    
    # Find the minimum trace length across all frequencies
    min_length = float('inf')
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            freq_min_length = min(len(trace) for trace in traces)
            min_length = min(min_length, freq_min_length)
    
    if min_length == float('inf') or min_length < 10:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate mean time course for each frequency
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_frequencies)))
    
    for i, freq in enumerate(unique_frequencies):
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            # Standardize all traces to minimum length
            standardized_traces = []
            for trace in traces:
                trace_array = np.array(trace).flatten()
                if len(trace_array) >= min_length:
                    standardized_traces.append(trace_array[:min_length])
            
            if len(standardized_traces) == 0:
                continue
                
            trace_array = np.array(standardized_traces)
            
            # Calculate mean and SEM across trials
            mean_trace = np.mean(trace_array, axis=0)
            sem_trace = np.std(trace_array, axis=0) / np.sqrt(len(traces))
            
            # Create time axis in seconds
            time_axis = np.arange(min_length) / rec_hz - pre_trial_length / rec_hz
            
            # Plot mean line with error band
            ax.plot(time_axis, mean_trace, color=colors[i], linewidth=2, 
                   label=f'{freq} Hz (n={len(traces)})')
            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                           color=colors[i], alpha=0.2)
    
    # Add vertical lines for important time points
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1, label='Trial Start')
    stim_start_time = (stim_start_frame - pre_trial_length) / rec_hz
    stim_end_time = (stim_end_frame - pre_trial_length) / rec_hz
    ax.axvline(x=stim_start_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Stim Start')
    ax.axvline(x=stim_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Stim End')
    
    # Add horizontal reference line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Time from Trial Start (s)')
    ax.set_ylabel('Mean Activity (z-score)')
    ax.set_title(f'{fit_type.capitalize()} Neuron {neuron_idx}\nMean Activity Across Trials (± SEM)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def create_activity_lineplot_custom(frequency_full_traces, unique_frequencies, ax, neuron_idx, classification,
                                   pre_trial_length, rec_hz, stim_start_frame, stim_end_frame):
    """Create line plot showing mean activity time course across trials for each frequency (no 0 Hz)."""
    
    # Find the minimum trace length across all frequencies
    min_length = float('inf')
    for freq in unique_frequencies:
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            freq_min_length = min(len(trace) for trace in traces)
            min_length = min(min_length, freq_min_length)
    
    if min_length == float('inf') or min_length < 10:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate mean time course for each frequency
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_frequencies)))
    
    for i, freq in enumerate(unique_frequencies):
        traces = frequency_full_traces[freq]
        if len(traces) > 0:
            # Standardize all traces to minimum length
            standardized_traces = []
            for trace in traces:
                trace_array = np.array(trace).flatten()
                if len(trace_array) >= min_length:
                    standardized_traces.append(trace_array[:min_length])
            
            if len(standardized_traces) == 0:
                continue
                
            trace_array = np.array(standardized_traces)

            smooth_trace = gaussian_filter1d(trace_array, sigma=1, axis=1)
            trace_array = smooth_trace
            # Calculate mean and SEM across trials
            mean_trace = np.mean(trace_array, axis=0)
            sem_trace = np.std(trace_array, axis=0) / np.sqrt(len(traces))
            
            # Create time axis in seconds
            time_axis = np.arange(min_length) / rec_hz - pre_trial_length / rec_hz
            
            # Plot mean line with error band
            ax.plot(time_axis, mean_trace, color=colors[i], linewidth=2, 
                   label=f'{freq} Hz (n={len(traces)})')
            ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                           color=colors[i], alpha=0.2)
    
    # Add vertical lines for important time points
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1, label='Trial Start')
    stim_start_time = (stim_start_frame - pre_trial_length) / rec_hz
    stim_end_time = (stim_end_frame - pre_trial_length) / rec_hz
    ax.axvline(x=stim_start_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Stim Start')
    ax.axvline(x=stim_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Stim End')
    
    # Add horizontal reference line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Time from Trial Start (s)')
    ax.set_ylabel('Mean Activity (z-score)')
    ax.set_title(f'Neuron {neuron_idx} ({classification})\nMean Activity Across Trials (± SEM)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable x-axis limits
    ax.set_xlim(time_axis[0], time_axis[-1])


def find_representative_neurons(auROC, dff_run_list, significant_neurons):
    """
    Find three representative neurons: high auROC, low auROC, and ~0.5 auROC.
    Now uses specific neurons: 121 for high, 518 for low, and closest to 0.5 for neutral.
    
    Returns:
        dict: Dictionary with 'high', 'low', and 'neutral' neuron indices
    """
    representative = {}
    
    # Use specific neurons as requested
    # High auROC: neuron 121
    if 121 < len(auROC) and 121 < len(dff_run_list):
        representative['high'] = 121
    
    # Low auROC: neuron 518
    if 518 < len(auROC) and 518 < len(dff_run_list):
        representative['low'] = 518
    
    # Neutral auROC: find the one closest to 0.5 (keep original logic)
    mean_aurocs = []
    valid_neurons = []
    
    for i, neuron_auroc in enumerate(auROC):
        if i < len(dff_run_list) and len(dff_run_list[i]) > 10:
            valid_auroc = neuron_auroc[~np.isnan(neuron_auroc)]
            if len(valid_auroc) > 5:  # At least 5 valid trials
                mean_auroc = np.mean(valid_auroc)
                mean_aurocs.append(mean_auroc)
                valid_neurons.append(i)
    
    if len(mean_aurocs) > 0:
        mean_aurocs = np.array(mean_aurocs)
        valid_neurons = np.array(valid_neurons)
        
        # Find neutral auROC neuron - closest to 0.5
        neutral_idx = np.argmin(np.abs(mean_aurocs - 0.5))
        representative['neutral'] = valid_neurons[neutral_idx]
    
    return representative


def create_auroc_method_illustration(dff_run_list, auROC, conditions, significant_neurons, 
                                   save_dir, pre_trial_length, rec_hz, 
                                   representative_examples=None):
    """
    Create a comprehensive illustration explaining the auROC method for paper figure.
    Shows three examples: high auROC (excitatory), low auROC (inhibitory), and ~0.5 auROC (no response).
    
    Parameters:
        dff_run_list: Neural activity data
        auROC: auROC values for all neurons
        conditions: Trial conditions  
        significant_neurons: List of significant neuron indices
        save_dir: Directory to save illustration
        pre_trial_length: Pre-trial length in frames
        rec_hz: Recording frequency
        representative_examples: Dict with specific neuron and trial indices for each type
                               Format: {'high': {'neuron': 121, 'trial': 5}, 
                                       'low': {'neuron': 518, 'trial': 12}, 
                                       'neutral': {'neuron': 45, 'trial': 8}}
    """
    print("\nCreating auROC method illustration for paper...")
    
    # Use provided representative examples or find them automatically
    if representative_examples is None:
        example_neurons = find_representative_neurons(auROC, dff_run_list, significant_neurons)
        # Convert to the new format for backward compatibility
        representative_examples = {}
        for neuron_type, neuron_idx in example_neurons.items():
            representative_examples[neuron_type] = {'neuron': neuron_idx, 'trial': 0}  # Default to first trial
    
    if len(representative_examples) == 0:
        print("No suitable example neurons found for illustration")
        return
    
    # Create three different illustrations for different auROC types
    neuron_types = [
        ('high', 'High auROC (Strong Excitatory)', 'darkgreen'),
        ('low', 'Low auROC (Strong Inhibitory)', 'darkred'), 
        ('neutral', 'Neutral auROC (No Response)', 'gray')
    ]
    
    for neuron_type, title_suffix, color in neuron_types:
        if neuron_type not in representative_examples:
            print(f"No suitable {neuron_type} auROC neuron found, skipping...")
            continue
            
        example_info = representative_examples[neuron_type]
        example_neuron = example_info['neuron']
        example_trial = example_info['trial']
        
        # Calculate mean auROC for this neuron (for display purposes)
        valid_auroc = auROC[example_neuron][~np.isnan(auROC[example_neuron])]
        mean_auroc = np.mean(valid_auroc) if len(valid_auroc) > 0 else 0.5
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Panel A: Raw neural activity example
        ax1 = fig.add_subplot(gs[0, :2])
        create_raw_activity_panel_single_trial(dff_run_list, example_neuron, example_trial, 
                                             conditions, ax1, pre_trial_length, rec_hz, neuron_type)
        ax1.set_title('A. Raw Neural Activity Example', fontweight='bold', fontsize=14)
        
        # Panel B: Baseline vs Stimulation distributions
        ax2 = fig.add_subplot(gs[0, 2:])
        create_baseline_stim_distributions_single_trial(dff_run_list, example_neuron, example_trial, 
                                                      ax2, pre_trial_length, rec_hz)
        ax2.set_title('B. Baseline vs Stimulation Distributions', fontweight='bold', fontsize=14)
        
        # Panel C: ROC Curve construction
        ax3 = fig.add_subplot(gs[1, :2])
        create_roc_curve_panel_single_trial(dff_run_list, example_neuron, example_trial, 
                                          ax3, pre_trial_length, rec_hz, auROC, conditions)
        ax3.set_title('C. ROC Curve Construction', fontweight='bold', fontsize=14)
        
        # Panel D: auROC interpretation (same for all)
        ax4 = fig.add_subplot(gs[1, 2:])
        create_auroc_interpretation_panel(ax4)
        ax4.set_title('D. auROC Interpretation', fontweight='bold', fontsize=14)
        
        # Panel E: auROC across frequencies (neurometric curve)
        ax5 = fig.add_subplot(gs[2, :])
        create_neurometric_example(auROC, example_neuron, conditions, ax5)
        ax5.set_title('E. Neurometric Curve: auROC across Frequencies', fontweight='bold', fontsize=14)
        
        # Panel F: Method flowchart (same for all)
        ax6 = fig.add_subplot(gs[3, :])
        create_method_flowchart(ax6)
        ax6.set_title('F. auROC Method Flowchart', fontweight='bold', fontsize=14)
        
        # Add main title with neuron-specific information
        fig.suptitle(f'auROC Method: {title_suffix}\n(Neuron {example_neuron}, Trial {example_trial}, Mean auROC = {mean_auroc:.3f})', 
                    fontsize=16, fontweight='bold', color=color)
        
        # Save illustration
        illustration_path = join(save_dir, f'auROC_method_illustration_{neuron_type}_auROC.pdf')
        plt.savefig(illustration_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{title_suffix} illustration saved to: {illustration_path}")
    
    print("All auROC method illustrations completed!")


def calculate_max_responsive_fraction(auROC, conditions, significant_neurons, significant_neurons_gauss, save_dir):
    """
    Calculate the maximum fraction of responsive trials per neuron and save to Excel.
    
    Parameters:
        auROC: auROC values for all neurons [neuron_idx][trial_idx]
        conditions: Trial condition codes
        significant_neurons: List of significant sigmoid neuron indices
        significant_neurons_gauss: List of significant gaussian neuron indices
        save_dir: Directory to save Excel file
    """
    print("\nCalculating maximum responsive fraction per neuron...")
    
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    
    # Prepare data containers
    sigmoid_data = []
    gaussian_data = []
    
    # Process sigmoid neurons
    for neuron_idx in significant_neurons:
        if neuron_idx < len(auROC):
            max_fraction = calculate_neuron_max_fraction(auROC[neuron_idx], conditions, frequency_map)
            sigmoid_data.append({
                'Neuron_Index': neuron_idx,
                'Max_Responsive_Fraction': max_fraction,
                'Classification': 'Sigmoid'
            })
    
    # Process gaussian neurons
    for neuron_idx in significant_neurons_gauss:
        if neuron_idx < len(auROC):
            max_fraction = calculate_neuron_max_fraction(auROC[neuron_idx], conditions, frequency_map)
            gaussian_data.append({
                'Neuron_Index': neuron_idx,
                'Max_Responsive_Fraction': max_fraction,
                'Classification': 'Gaussian'
            })
    
    # Save to Excel with separate sheets
    excel_path = join(save_dir, 'max_responsive_fraction_analysis.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sigmoid neurons sheet
        if sigmoid_data:
            sigmoid_df = pd.DataFrame(sigmoid_data)
            sigmoid_df.to_excel(writer, sheet_name='Sigmoid_Neurons', index=False)
            print(f"Sigmoid neurons: {len(sigmoid_data)} neurons analyzed")
        
        # Gaussian neurons sheet
        if gaussian_data:
            gaussian_df = pd.DataFrame(gaussian_data)
            gaussian_df.to_excel(writer, sheet_name='Gaussian_Neurons', index=False)
            print(f"Gaussian neurons: {len(gaussian_data)} neurons analyzed")
        
        # Combined summary sheet
        all_data = sigmoid_data + gaussian_data
        if all_data:
            combined_df = pd.DataFrame(all_data)
            combined_df.to_excel(writer, sheet_name='All_Neurons', index=False)
            
            # Add summary statistics
            summary_stats = []
            for classification in ['Sigmoid', 'Gaussian']:
                class_data = combined_df[combined_df['Classification'] == classification]
                if len(class_data) > 0:
                    summary_stats.append({
                        'Classification': classification,
                        'Count': len(class_data),
                        'Mean_Max_Fraction': class_data['Max_Responsive_Fraction'].mean(),
                        'Std_Max_Fraction': class_data['Max_Responsive_Fraction'].std(),
                        'Median_Max_Fraction': class_data['Max_Responsive_Fraction'].median(),
                        'Min_Max_Fraction': class_data['Max_Responsive_Fraction'].min(),
                        'Max_Max_Fraction': class_data['Max_Responsive_Fraction'].max()
                    })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"Max responsive fraction analysis saved to: {excel_path}")
    return excel_path


def calculate_neuron_max_fraction(neuron_auROC, conditions, frequency_map):
    """
    Calculate the maximum fraction of responsive trials for a single neuron across all frequencies.
    Uses the same logic as neurometric curve fitting: calculates excited and inhibited responses
    separately, then returns the maximum fraction from whichever response type is stronger.
    
    Parameters:
        neuron_auROC: auROC values for this neuron [trial_idx]
        conditions: Trial condition codes
        frequency_map: Mapping from condition codes to frequencies
        
    Returns:
        float: Maximum fraction of responsive trials across all frequencies for the dominant response type
    """
    # Organize trials by frequency for excited and inhibited responses separately
    freq_excited = {freq: {'responsive': 0, 'total': 0} for freq in frequency_map.values()}
    freq_inhibited = {freq: {'responsive': 0, 'total': 0} for freq in frequency_map.values()}
    
    # Use the exact same logic as neurometric curve calculation
    # Compute binary responses for excited (> 0.5) and inhibited (< 0.5)
    excited = np.where(neuron_auROC > 0.5, 1, 0)
    inhibited = np.where(neuron_auROC < 0.5, 1, 0)
    
    for trial_idx, condition in enumerate(conditions):
        if trial_idx < len(neuron_auROC):
            freq = frequency_map.get(condition)
            if freq is not None:
                exc_val = excited[trial_idx]
                inh_val = inhibited[trial_idx]
                
                # Count excited responses - exact same logic as neurometric curves
                if exc_val == 1:
                    freq_excited[freq]['responsive'] += 1
                freq_excited[freq]['total'] += 1
                
                # Count inhibited responses - exact same logic as neurometric curves
                if inh_val == 1:
                    freq_inhibited[freq]['responsive'] += 1
                freq_inhibited[freq]['total'] += 1
    
    # Calculate fractions for excited responses
    excited_fractions = []
    for freq, counts in freq_excited.items():
        if counts['total'] > 0:
            fraction = counts['responsive'] / counts['total']
            excited_fractions.append(fraction)
    
    # Calculate fractions for inhibited responses
    inhibited_fractions = []
    for freq, counts in freq_inhibited.items():
        if counts['total'] > 0:
            fraction = counts['responsive'] / counts['total']
            inhibited_fractions.append(fraction)
    
    # Actually fit curves and select based on R² - exactly matching neurometric curve logic
    if not excited_fractions and not inhibited_fractions:
        return 0.0
    
    # Convert to arrays for curve fitting
    sorted_frequencies = sorted(frequency_map.values())
    x_data_all = np.array(sorted_frequencies)
    
    # Prepare y_data arrays
    y_data_excited = np.array([freq_excited[f]['responsive'] / freq_excited[f]['total'] 
                              if freq_excited[f]['total'] > 0 else 0 for f in sorted_frequencies])
    y_data_inhibited = np.array([freq_inhibited[f]['responsive'] / freq_inhibited[f]['total'] 
                                if freq_inhibited[f]['total'] > 0 else 0 for f in sorted_frequencies])
    
    # Set up bounds for sigmoid fitting
    bounds = ([0, min(x_data_all), 0, 0], [1, max(x_data_all), 10, 1])
    
    # Fit sigmoid curves and calculate R²
    r_squared_excited = 0
    r_squared_inhibited = 0
    
    # Try fitting excited sigmoid
    try:
        p0_excited = [np.max(y_data_excited), np.median(x_data_all), 1, np.min(y_data_excited)]
        popt_excited, _ = curve_fit(sigmoid, x_data_all, y_data_excited, p0=p0_excited,
                                   method='trf', bounds=bounds)
        r_squared_excited = compute_goodness_of_fit(x_data_all, y_data_excited, popt_excited)
    except (RuntimeError, ValueError):
        r_squared_excited = 0
    
    # Try fitting inhibited sigmoid
    try:
        p0_inhibited = [np.max(y_data_inhibited), np.median(x_data_all), 1, np.min(y_data_inhibited)]
        popt_inhibited, _ = curve_fit(sigmoid, x_data_all, y_data_inhibited, p0=p0_inhibited,
                                     method='trf', bounds=bounds)
        r_squared_inhibited = compute_goodness_of_fit(x_data_all, y_data_inhibited, popt_inhibited)
    except (RuntimeError, ValueError):
        r_squared_inhibited = 0
    
    # Select the curve with higher R² (exactly matching neurometric curve logic)
    if r_squared_excited >= r_squared_inhibited:
        return max(excited_fractions) if excited_fractions else 0.0
    else:
        return max(inhibited_fractions) if inhibited_fractions else 0.0


def create_raw_activity_panel(dff_run_list, neuron_idx, conditions, ax, pre_trial_length, rec_hz, neuron_type='neutral'):
    """Create panel showing raw neural activity with baseline and stimulation periods."""
    
    # Get neuron data
    neuron_data = dff_run_list[neuron_idx]
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    
    # Find trials for 15 Hz (condition code 7)
    target_freq = 7  # condition code for 15 Hz
    trial_indices = [i for i, cond in enumerate(conditions) if cond == target_freq]
    
    # Select specific trials based on neuron type
    if neuron_type == 'high' and neuron_idx == 121:
        # For high auROC (neuron 121): take first 3 trials of 15 Hz
        selected_trials = trial_indices[:3]
        title_suffix = "First 3 trials of 15 Hz"
    elif neuron_type == 'low' and neuron_idx == 518:
        # For low auROC (neuron 518): take last 3 trials of 15 Hz
        selected_trials = trial_indices[-3:] if len(trial_indices) >= 3 else trial_indices
        title_suffix = "Last 3 trials of 15 Hz"
    else:
        # For neutral: take first 3 trials as before
        selected_trials = trial_indices[:3]
        title_suffix = "First 3 trials of 15 Hz"
    
    # Plot selected trials
    colors = ['blue', 'green', 'red']
    for i, trial_idx in enumerate(selected_trials):
        if trial_idx < len(neuron_data):
            trial_data = neuron_data[trial_idx]
            time_axis = np.arange(len(trial_data)) / rec_hz - pre_trial_length / rec_hz
            ax.plot(time_axis, trial_data, color=colors[i], linewidth=1.5, 
                   label=f'Trial {trial_idx+1}', alpha=0.8)
    
    # Mark baseline and stimulation periods
    baseline_start = -1.0  # 1 sec before trial start
    baseline_end = 0.0
    stim_start = 2.0
    stim_end = 5.0
    
    ax.axvspan(baseline_start, baseline_end, alpha=0.3, color='gray', label='Baseline')
    ax.axvspan(stim_start, stim_end, alpha=0.3, color='orange', label='Stimulation')
    
    ax.set_xlabel('Time from Trial Start (s)')
    ax.set_ylabel('Neural Activity (z-score)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Neuron {neuron_idx}: {title_suffix}')


def create_baseline_stim_distributions(dff_run_list, neuron_idx, ax, pre_trial_length, rec_hz):
    """Create panel showing baseline and stimulation activity distributions."""
    
    neuron_data = dff_run_list[neuron_idx]
    
    # Define periods
    baseline_start_frame = int(pre_trial_length - 1 * rec_hz)  # 1 sec before trial
    baseline_end_frame = pre_trial_length
    stim_start_frame = pre_trial_length + 3 * rec_hz
    stim_end_frame = pre_trial_length + 5 * rec_hz
    
    # Collect baseline and stimulation activities
    baseline_activities = []
    stim_activities = []
    
    for trial_data in neuron_data[:20]:  # Use first 20 trials
        if len(trial_data) > stim_end_frame:
            baseline_act = np.mean(trial_data[baseline_start_frame:baseline_end_frame])
            stim_act = np.mean(trial_data[stim_start_frame:stim_end_frame])
            
            baseline_activities.append(baseline_act)
            stim_activities.append(stim_act)
    
    # Create histograms with thinner bins
    bins = np.linspace(min(min(baseline_activities), min(stim_activities)), 
                      max(max(baseline_activities), max(stim_activities)), 30)  # Increased from 15 to 30
    
    ax.hist(baseline_activities, bins=bins, alpha=0.6, label='Baseline', color='gray', density=True)
    ax.hist(stim_activities, bins=bins, alpha=0.6, label='Stimulation', color='orange', density=True)
    
    ax.set_xlabel('Mean Activity (z-score)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_raw_activity_panel_single_trial(dff_run_list, neuron_idx, trial_idx, conditions, ax, pre_trial_length, rec_hz, neuron_type='neutral'):
    """Create panel showing raw neural activity for a single specific trial."""
    
    # Get neuron data
    neuron_data = dff_run_list[neuron_idx]
    
    # Check if the specified trial exists
    if trial_idx >= len(neuron_data):
        ax.text(0.5, 0.5, f'Trial {trial_idx} not available for neuron {neuron_idx}', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get the specific trial data
    trial_data = neuron_data[trial_idx]
    time_axis = np.arange(len(trial_data)) / rec_hz - pre_trial_length / rec_hz
    
    # Plot the single trial
    ax.plot(time_axis, trial_data, color='darkblue', linewidth=2.0, 
           label=f'Single Trial {trial_idx}', alpha=0.9)
    
    # Mark baseline and stimulation periods
    baseline_start = -1.0  # 1 sec before trial start
    baseline_end = 0.0
    stim_start = 2.0
    stim_end = 5.0
    
    ax.axvspan(baseline_start, baseline_end, alpha=0.3, color='gray', label='Baseline')
    ax.axvspan(stim_start, stim_end, alpha=0.3, color='orange', label='Stimulation')
    
    ax.set_xlabel('Time from Trial Start (s)')
    ax.set_ylabel('Neural Activity (z-score)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Neuron {neuron_idx}: Single Trial {trial_idx}')


def create_baseline_stim_distributions_single_trial(dff_run_list, neuron_idx, trial_idx, ax, pre_trial_length, rec_hz):
    """Create panel showing baseline and stimulation activity for a single trial."""
    
    neuron_data = dff_run_list[neuron_idx]
    
    # Check if the specified trial exists
    if trial_idx >= len(neuron_data):
        ax.text(0.5, 0.5, f'Trial {trial_idx} not available for neuron {neuron_idx}', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get the specific trial data
    trial_data = neuron_data[trial_idx]
    
    # Define periods
    baseline_start_frame = int(pre_trial_length - 1 * rec_hz)  # 1 sec before trial
    baseline_end_frame = pre_trial_length
    stim_start_frame = pre_trial_length + 3 * rec_hz
    stim_end_frame = pre_trial_length + 5 * rec_hz
    
    # Check if trial has enough frames
    if len(trial_data) <= stim_end_frame:
        ax.text(0.5, 0.5, f'Trial {trial_idx} too short for analysis', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract baseline and stimulation activities for this single trial
    baseline_act = np.mean(trial_data[baseline_start_frame:baseline_end_frame])
    stim_act = np.mean(trial_data[stim_start_frame:stim_end_frame])
    
    # Create histograms showing the single trial data
    # Use a small range around the actual values for visualization
    min_val = min(baseline_act, stim_act) - 0.1
    max_val = max(baseline_act, stim_act) + 0.1
    bins = np.linspace(min_val, max_val, 20)
    
    # Plot as vertical lines instead of histograms since we only have one value each
    ax.axvline(baseline_act, color='gray', linewidth=3, label=f'Baseline: {baseline_act:.3f}', alpha=0.8)
    ax.axvline(stim_act, color='orange', linewidth=3, label=f'Stimulation: {stim_act:.3f}', alpha=0.8)
    
    # Add text annotations
    ax.text(0.05, 0.9, f'Baseline: {baseline_act:.3f}', transform=ax.transAxes, 
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.3))
    ax.text(0.05, 0.8, f'Stimulation: {stim_act:.3f}', transform=ax.transAxes, 
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
    
    ax.set_xlabel('Mean Activity (z-score)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Neuron {neuron_idx}, Trial {trial_idx}: Single Trial Activity')


def create_roc_curve_panel(dff_run_list, neuron_idx, ax, pre_trial_length, rec_hz, auROC, conditions):
    """Create ROC curve panel using the same method as main auROC calculation."""
    
    from sklearn.metrics import roc_curve, auc
    
    # Get the actual mean auROC for this neuron from the main calculation
    neuron_auroc = auROC[neuron_idx]
    valid_auroc = neuron_auroc[~np.isnan(neuron_auroc)]
    mean_auroc = np.mean(valid_auroc) if len(valid_auroc) > 0 else 0.5
    
    # Use the same time windows as the main auROC calculation
    neuron_data = dff_run_list[neuron_idx]
    baseline_start_frame = int(pre_trial_length - 1 * rec_hz)  # Same as main calculation
    baseline_end_frame = int(pre_trial_length + 1 * rec_hz)
    stim_start_frame = int(pre_trial_length + 3 * rec_hz)  # Same as main calculation  
    stim_end_frame = int(pre_trial_length + 5 * rec_hz)
    
    # Collect data using the same method as main auROC calculation
    all_activities = []
    all_labels = []
    
    # Use same number of trials as available
    for trial_idx, trial_data in enumerate(neuron_data):
        if len(trial_data) > stim_end_frame:
            baseline_act = np.mean(trial_data[baseline_start_frame:baseline_end_frame])
            stim_act = np.mean(trial_data[stim_start_frame:stim_end_frame])
            
            all_activities.extend([baseline_act, stim_act])
            all_labels.extend([0, 1])  # 0=baseline, 1=stimulation
    
    if len(all_activities) > 0:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_activities)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkblue', linewidth=3, label=f'Example ROC (auROC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Chance Level (auROC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.2, color='darkblue')
        
        # Add interpretation text based on MEAN auROC value
        if mean_auroc > 0.6:
            interpretation = "Strong Excitatory Response"
            text_color = 'darkgreen'
        elif mean_auroc < 0.4:
            interpretation = "Strong Inhibitory Response"
            text_color = 'darkred'
        else:
            interpretation = "Neutral/No Response"
            text_color = 'gray'
            
        ax.text(0.6, 0.2, interpretation, fontsize=12, fontweight='bold', 
               color=text_color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add text showing the mean auROC from main calculation
        ax.text(0.05, 0.95, f'Mean auROC across all trials: {mean_auroc:.3f}', 
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f'Neuron {neuron_idx}: Example ROC Construction')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for ROC curve', ha='center', va='center', 
               transform=ax.transAxes)


def create_roc_curve_panel_single_trial(dff_run_list, neuron_idx, trial_idx, ax, pre_trial_length, rec_hz, auROC, conditions):
    """Create ROC curve panel for a single specific trial."""
    
    from sklearn.metrics import roc_curve, auc
    
    # Get the actual mean auROC for this neuron from the main calculation (for display purposes)
    neuron_auroc = auROC[neuron_idx]
    valid_auroc = neuron_auroc[~np.isnan(neuron_auroc)]
    mean_auroc = np.mean(valid_auroc) if len(valid_auroc) > 0 else 0.5
    
    # Get the specific trial data
    neuron_data = dff_run_list[neuron_idx]
    
    # Check if the specified trial exists
    if trial_idx >= len(neuron_data):
        ax.text(0.5, 0.5, f'Trial {trial_idx} not available for neuron {neuron_idx}', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    trial_data = neuron_data[trial_idx]
    
    # Use the same time windows as the main auROC calculation
    baseline_start_frame = int(pre_trial_length - 1 * rec_hz)  # Same as main calculation
    baseline_end_frame = int(pre_trial_length + 1 * rec_hz)
    stim_start_frame = int(pre_trial_length + 3 * rec_hz)  # Same as main calculation  
    stim_end_frame = int(pre_trial_length + 5 * rec_hz)
    
    # Check if trial has enough frames
    if len(trial_data) <= stim_end_frame:
        ax.text(0.5, 0.5, f'Trial {trial_idx} too short for analysis', 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract baseline and stimulation activities for this single trial
    baseline_act = np.mean(trial_data[baseline_start_frame:baseline_end_frame])
    stim_act = np.mean(trial_data[stim_start_frame:stim_end_frame])
    
    # For single trial, we can't create a proper ROC curve, but we can show the data points
    # and explain the concept
    ax.scatter([0, 1], [baseline_act, stim_act], s=100, 
              color=['gray', 'orange'], alpha=0.8, label='Single Trial Data')
    
    # Add horizontal lines to show the separation
    ax.axhline(baseline_act, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_act:.3f}')
    ax.axhline(stim_act, color='orange', linestyle='--', alpha=0.5, label=f'Stimulation: {stim_act:.3f}')
    
    # Add interpretation text based on the single trial comparison
    if stim_act > baseline_act:
        interpretation = "Excitatory Response (This Trial)"
        text_color = 'darkgreen'
    elif stim_act < baseline_act:
        interpretation = "Inhibitory Response (This Trial)"
        text_color = 'darkred'
    else:
        interpretation = "No Change (This Trial)"
        text_color = 'gray'
        
    ax.text(0.6, 0.2, interpretation, fontsize=12, fontweight='bold', 
           color=text_color, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add text showing the mean auROC from main calculation for context
    ax.text(0.05, 0.95, f'Mean auROC across all trials: {mean_auroc:.3f}', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Add explanation text
    ax.text(0.05, 0.85, f'Single Trial {trial_idx}:\nBaseline vs Stimulation\n(No ROC curve possible\nwith single trial)', 
           transform=ax.transAxes, fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.set_xlabel('Condition (0=Baseline, 1=Stimulation)')
    ax.set_ylabel('Neural Activity (z-score)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.1])
    ax.set_title(f'Neuron {neuron_idx}, Trial {trial_idx}: Single Trial Activity Comparison')


def create_auroc_interpretation_panel(ax):
    """Create panel explaining auROC interpretation."""
    
    ax.axis('off')
    
    # Create text explanation
    interpretation_text = """
auROC Interpretation:

• auROC = 0.5: No discrimination
  (Random/No response)

• auROC > 0.5: Excitatory response
  (Activity increases during stimulation)

• auROC < 0.5: Inhibitory response  
  (Activity decreases during stimulation)

• auROC = 1.0: Perfect discrimination
  (Complete separation of distributions)

Statistical Significance:
• Permutation test determines if auROC
  differs significantly from chance (0.5)
• Only significant neurons included in analysis
    """
    
    ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


def create_neurometric_example(auROC, neuron_idx, conditions, ax):
    """Create example neurometric curve."""
    
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
    
    # Organize auROC by frequency
    freq_auroc = {freq: [] for freq in frequency_map.values()}
    
    neuron_auroc = auROC[neuron_idx]
    for trial_idx, condition in enumerate(conditions):
        if trial_idx < len(neuron_auroc) and not np.isnan(neuron_auroc[trial_idx]):
            freq = frequency_map.get(condition)
            if freq is not None:
                freq_auroc[freq].append(neuron_auroc[trial_idx])
    
    # Calculate means and SEMs
    frequencies = []
    means = []
    sems = []
    
    for freq in sorted(freq_auroc.keys()):
        values = freq_auroc[freq]
        if len(values) > 0:
            frequencies.append(freq)
            means.append(np.mean(values))
            sems.append(np.std(values) / np.sqrt(len(values)))
    
    # Plot neurometric curve
    ax.errorbar(frequencies, means, yerr=sems, marker='o', linewidth=2, 
               markersize=8, capsize=5, color='darkgreen')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance Level')
    ax.set_xlabel('Stimulation Frequency (Hz)')
    ax.set_ylabel('auROC')
    ax.set_title(f'Example Neuron {neuron_idx}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])


def create_method_flowchart(ax):
    """Create method flowchart."""
    
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    
    # Flowchart boxes
    boxes = [
        (1, 2.5, "1. Record\nNeural Activity"),
        (3, 2.5, "2. Define Baseline\n& Stimulation\nPeriods"),
        (5, 2.5, "3. Calculate\nauROC per Trial"),
        (7, 2.5, "4. Statistical\nSignificance\nTesting"),
        (9, 2.5, "5. Neurometric\nCurve Fitting")
    ]
    
    details = [
        (1, 1.5, "• Ca²⁺ imaging\n• Z-score\n  normalization"),
        (3, 1.5, "• Baseline: -1-0s\n• Stimulation:\n  2-5s"),
        (5, 1.5, "• ROC analysis\n• Per frequency\n• Per trial"),
        (7, 1.5, "• Permutation\n  test\n• p < 0.05"),
        (9, 1.5, "• Sigmoid/\n  Gaussian\n  fitting")
    ]
    
    # Draw boxes and text
    for x, y, text in boxes:
        bbox = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        ax.text(x, y, text, ha='center', va='center', fontweight='bold', 
               fontsize=10, bbox=bbox)
    
    for x, y, text in details:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Draw arrows
    for i in range(len(boxes)-1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        y = boxes[i][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))



def process_neuron(
    i, prob_valve2, prob_baseline, n_shuffles, alpha, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    n_trials = prob_valve2.shape[1]
    auROC_per_neuron = np.full(n_trials, np.nan)

    for trial in range(n_trials):
        scores_stim = prob_valve2[i, trial, :]
        scores_base = prob_baseline[i, trial, :]

        # Skip bad trials
        if np.all(np.isnan(scores_stim)) or np.all(np.isnan(scores_base)):
            continue

        all_scores = np.concatenate([scores_base, scores_stim])
        all_labels = np.concatenate([
            np.zeros_like(scores_base),  # 0 = baseline
            np.ones_like(scores_stim)    # 1 = stim
        ])

        # Actual AUC
        try:
            actual_auroc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            continue

        # Trial-specific null by label shuffling
        null_vals = []
        for _ in range(n_shuffles):
            shuf = rng.permutation(all_labels)
            try:
                null_vals.append(roc_auc_score(shuf, all_scores))
            except ValueError:
                pass

        if len(null_vals) == 0:
            continue

        thr_hi = np.percentile(null_vals, 100 * (1 - alpha))
        thr_lo = np.percentile(null_vals, 100 * alpha)

        if actual_auroc > thr_hi or actual_auroc < thr_lo:
            auROC_per_neuron[trial] = actual_auroc

    return i, auROC_per_neuron


# ---------------------------
# 2) Main function
# ---------------------------
def auROC_with_shuffle_parallel(
        dff_bins,
        dff_run_list_valve2,
        folder_name,
        n_shuffles=1000,
        alpha=0.05
):
    """
    Same signature as your original function, but parts are parallelized.
    """

    dff_bins = np.asarray(dff_bins)
    min_dff = np.nanpercentile(dff_bins, 0.1, axis=1)
    max_dff = np.nanpercentile(dff_bins, 99.9, axis=1)
    # criteria = [(max_dff[i] - min_dff[i]) / 100 for i in range(len(dff_bins))]
    criteria_range = [np.linspace(min_dff[i], max_dff[i], num=100) for i in range(len(dff_bins))]

    # For the sake of this example, we assume you have these global or computed:
    baseline_start = int(pre_trial_length - 1 * rec_hz ) #- 2 * rec_hz
    baseline_end = int(pre_trial_length + 1 * rec_hz)
    stimulation_start = int(pre_trial_length + 3 * rec_hz)
    stimulation_end = int(pre_trial_length + 5 * rec_hz)

    n_neurons = len(dff_run_list_valve2)
    n_trials = len(dff_run_list_valve2[0])
    n_crit = len(criteria_range[0])  # 100

    # -------------------
    # Pre-allocate arrays
    # -------------------
    prob_valve2 = np.zeros((n_neurons, n_trials, n_crit))
    prob_baseline = np.zeros((n_neurons, n_trials, n_crit))

    # -------------------------------------------------
    # 2.1) Calculate probabilities for each neuron/trial/criterion
    #      (still in the main process—once per neuron/trial)
    # -------------------------------------------------
    for i in range(n_neurons):
        for trial in range(n_trials):
            stim_data = np.array(dff_run_list_valve2[i][trial][stimulation_start:stimulation_end])
            base_data = np.array(dff_run_list_valve2[i][trial][baseline_start:baseline_end])
            valid_stim = ~np.isnan(stim_data)
            valid_base = ~np.isnan(base_data)

            for j in range(n_crit):
                if np.sum(valid_stim) > 0:
                    prob_valve2[i, trial, j] = (
                            np.sum(stim_data[valid_stim] > criteria_range[i][j]) / np.sum(valid_stim)
                    )
                else:
                    prob_valve2[i, trial, j] = np.nan

                if np.sum(valid_base) > 0:
                    prob_baseline[i, trial, j] = (
                            np.sum(base_data[valid_base] > criteria_range[i][j]) / np.sum(valid_base)
                    )
                else:
                    prob_baseline[i, trial, j] = np.nan

    # -------------------
    # 2.2) Parallel loop
    # -------------------
    # Pre-allocate auROC result array
    auROC_valve2 = np.zeros((n_neurons, n_trials))
    if __name__ == '__main__':
        # Create a pool of workers
        with multiprocessing.Pool() as pool:
            # Map over neurons
            results = pool.starmap(
                process_neuron,
                [
                    (
                        i,
                        prob_valve2,
                        prob_baseline,
                        n_shuffles,
                        alpha
                    )
                    for i in range(n_neurons)
                ]
            )

        # `results` is a list of (i, auROC_for_neuron_i) tuples
        for (i, auROC_for_neuron) in results:
            auROC_valve2[i, :] = auROC_for_neuron

    return auROC_valve2, prob_valve2, prob_baseline

def plot_psychometric(conditions_played, responses, mouse, color, output_dir, save=True):
    """
    Plots the psychometric (sigmoid) function based on the provided data and saves the plot.

    Parameters:
        conditions_played (list): List of condition codes.
        responses (list): List of responses corresponding to each condition.
        mouse (str): Identifier for the subject (used in the plot legend).
        color (str): Color for the plot lines.
        output_dir (str): Directory where the output files will be saved.
    """

    # Map condition to actual stimulation frequency (Hz)
    frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}

    # Initialize a dictionary to store response counts for each actual frequency
    response_dict = {freq: {'licked': 0, 'total': 0} for freq in frequency_map.values()}

    # Process the input data
    for condition, response in zip(conditions_played, responses):
        freq = frequency_map[condition]
        if response == 3 or response == 5:
            response_dict[freq]['licked'] += 1
        response_dict[freq]['total'] += 1

    # Calculate the fraction of "licked" responses for each frequency
    fractions_licked = {
        freq: res['licked'] / res['total'] if res['total'] > 0 else 0
        for freq, res in response_dict.items()
    }

    # Sort the frequencies and corresponding response fractions
    sorted_frequencies = sorted(fractions_licked.keys())
    sorted_fractions = [fractions_licked[freq] for freq in sorted_frequencies]

    # Use the actual frequencies for x_data (no mapping to equally spaced ticks)
    x_data = np.array(sorted_frequencies)
    y_data = np.array(sorted_fractions)

    try:
        # Initial guess for sigmoid parameters: [L, x0, k, b]
        p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='trf')

        # Generate x values for the fitted curve using the actual frequency scale
        x_values = np.linspace(min(x_data), max(x_data), 100)
        y_values = sigmoid(x_values, *popt)

        # Calculate goodness-of-fit metrics (R² and adjusted R²)
        y_fit = sigmoid(x_data, *popt)
        residuals = y_data - y_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R^2: {r_squared:.2f}")
        n = len(x_data)
        k = len(popt)
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        print(f"Adjusted R^2: {r_squared_adj:.2f}")
        # calculate the point of the half-difference between the max and min
        half_max = (np.max(y_values) - np.min(y_values)) / 2
        diff_value = np.min(y_values) + half_max
        threshold_for_half_max = x_values[np.where(1-y_values >= diff_value)[0][0]]
        # Save the R² values
        with open(join(output_dir, 'R^2.txt'), 'w') as f:
            f.write(f"R^2: {r_squared:.2f}\n")
            f.write(f"Adjusted R^2: {r_squared_adj:.2f}\n")

        # Plot the fitted sigmoid curve (scaled to percentage)
        plt.plot(x_values, (1 - y_values) * 100, label=mouse, color=color, linewidth=4)
        # Optionally, plot the original data points:
        # plt.scatter(x_data, (1 - y_data) * 100, color='black')

        # Draw a vertical line at the sigmoid's midpoint (popt[1]) on the actual frequency scale
        plt.axvline(x=popt[1], color=color, linestyle='--')
        plt.text(popt[1], 50, f'{popt[1]:.2f} Hz', color=color, ha='right', va='center')
        plt.axvline(x=threshold_for_half_max, color='black', linestyle='--')
        plt.text(threshold_for_half_max, 50, f'{threshold_for_half_max:.2f} Hz', color='black', ha='left', va='center')


    except RuntimeError as e:
        print(f"Curve fitting failed due to: {e}")

    # Finalize the plot
    plt.legend()
    plt.xlabel('Optogenetic Stimulation Frequency (Hz)')
    plt.ylabel('Detection Rate (%)')
    plt.xticks(x_data, labels=[f"{freq} Hz" for freq in x_data])
    plt.yticks(np.linspace(0, 100, 11))

    # Save the plots to files
    if save == True:
        plt.savefig(join(output_dir, 'Psychophysics Plot.png'))
        plt.savefig(join(output_dir, 'Psychophysics Plot.pdf'))
    plt.close()
    return threshold_for_half_max
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

    # Make a copy so we don't alter dff_run_list_sub in place
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
                # If something weird: the trial is shorter than len_pre_trial
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
def norm_to_max(cells_data):
    norm_df = []
    for i in range(len(cells_data)):
        max = np.percentile(cells_data.iloc[i], 99)
        norm_df.append(cells_data.iloc[i] / max)
    norm_df = np.asarray(norm_df)
    return norm_df
if '__main__' == __name__:
    # Loop over multiple mice/days
    mice_days = [
        # ('OI116', '240529'),
        # ('OI124', '240610'),
        ('OI151', '240827'),
        # ('OI152', '241003'),
        # ('OI208', '250227'),
    ]
    # ('OI151', '240827'),
    for mouse_id, day_str in mice_days:
        print(f"\nProcessing mouse {mouse_id} on day {day_str}")
        ROOT_PATH = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data'
        save_loc = fr'Z:\2p_data\{mouse_id}\{day_str}\ML\data\processed_data\auroc_neurometric_zscore_no_sub_v4'
        os.makedirs(save_loc, exist_ok=True)
        n_priming_trials = 23  # 23 trials of priming, to remove from the data
        # load the values and then make the different plots
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

        # Remove priming trials from all data structures
        conditions_played = conditions_played[n_priming_trials:]
        responses = responses[n_priming_trials:]
        opto_on = opto_on[n_priming_trials:]
        distract_light = distract_light[n_priming_trials:]
        # Remove first 23 columns only from first row of trials_ind
        # trials_ind is a 3x90 array where each row represents different trial information
        # We need to remove the first 20 trials from the first row only
        # Since we can't have different sized rows in a numpy array, we'll create a list
        trials_ind = [trials_ind[0][n_priming_trials:], trials_ind[1], trials_ind[2]]
        # Z-score each neuron (across all timepoints, not per trial)
        dff_norm = []
        for i in range(len(dff_4min)):
            mean = np.nanmean(dff_4min.iloc[i])
            std = np.nanstd(dff_4min.iloc[i])
            dff_norm.append((dff_4min.iloc[i] - mean) / std)
        dff_norm = pd.DataFrame(dff_norm)
        #divide to the different trials
        dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)
        dff_run_list_sub = baseline_subtract_neurons(dff_run_list, pre_trial_length, baseline_time=4 * rec_hz)
        all_values = dff_norm

        # for each trial, find where the light is on and the reward is delivered
        for i in range(len(distract_light)):
            # Find indices where values first exceed 1
            changes = np.diff(np.concatenate(([0], distract_light[i] > 1)))
            distract_light[i] = np.where(changes == 1)[0]
            changes = np.diff(np.concatenate(([0], opto_on[i] > 1)))
            opto_on[i] = np.where(changes == 1)[0]
            changes = np.diff(np.concatenate(([0], valve_open[i] > 1)))

        correct_reject = np.where(responses == 4)[0]
        false_alarm = np.where(responses == 5)[0]
        hit = np.where(responses == 3)[0]
        miss = np.where(responses == 2)[0]

        frequency_map = {6: 0, 7: 15, 8: 10, 9: 7.5, 10: 5, 11: 3, 12: 1, 13: 0.5}
        # get the indices of each frequency
        freq_indices = {freq: [i for i, x in enumerate(conditions_played) if x == freq] for freq in frequency_map.keys()}
        # find the minimum number of trials from all the frequencies
        min_trials = min([len(freq_indices[freq]) for freq in freq_indices.keys()])
        # randomly select the same number of trials from each frequency
        selected_trials = []
        for freq in freq_indices.keys():
            selected_trials.append(np.random.choice(freq_indices[freq], min_trials, replace=False))
        selected_trials = np.concatenate(selected_trials)
        for i in range(len(valve_open)):
            changes = np.diff(np.concatenate(([0], valve_open[i] > 1)))
            valve_open[i] = np.where(changes == 1)[0]
        rewards_times = [valve_open[i] for i in hit]
        # # divide the data to the different conditions
        dff_correct_reject = []
        dff_false_alarm = []
        dff_hit = []
        dff_miss = []
        for neuron in (dff_run_list):
            neuron = pd.DataFrame(neuron)
            neuron_correct_reject = []
            neuron_false_alarm = []
            neuron_hit = []
            neuron_miss = []
            neuron_correct_reject.append(neuron.iloc[correct_reject, :])
            neuron_false_alarm.append(neuron.iloc[false_alarm, :])
            neuron_hit.append(neuron.iloc[hit, :])
            neuron_miss.append(neuron.iloc[miss, :])
            dff_correct_reject.append(neuron_correct_reject[0])
            dff_false_alarm.append(neuron_false_alarm[0])
            dff_hit.append(neuron_hit[0])
            dff_miss.append(neuron_miss[0])

        # create the figures for X examples of neurons
        # create_figures(dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, rewards_times, ROOT_PATH,
        #                pre_trial_length, rec_hz, 'heatmaps_single_cells', num_neurons=497)

        stim_start = pre_trial_length + 2 * rec_hz
        stim_end = pre_trial_length + 5 * rec_hz
        # detection_histogram(dff_sub, stim_start, stim_end, rec_hz, data_loc)
        # avg_responses = fit_neurometric_curve(dff_sub, stim_start, stim_end, conditions_played, data_loc)
        # Normalize each neuron's responses by its maximum.
        # Use keepdims=True to ensure proper broadcasting.
        # normalized_avg_responses = avg_responses / np.max(avg_responses, axis=1, keepdims=True)
        #
        # # Compute the index of the peak response for each neuron.
        # peak_indices = np.argmax(normalized_avg_responses, axis=1)
        #
        # # Compute the peak response value for each neuron.
        # peak_values = np.max(normalized_avg_responses, axis=1)
        #
        # # Use lexsort to sort first by the peak index and then by the peak value.
        # # Note: np.lexsort uses the last key as the primary sort key.
        # sorted_indices = np.lexsort((peak_values, peak_indices))
        #
        # # Reorder the normalized matrix.
        # sorted_normalized_avg_responses = normalized_avg_responses[sorted_indices, :]

        # Plot the sorted normalized responses.
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(sorted_normalized_avg_responses, cmap='viridis', cbar_kws={'label': 'Normalized Response'})
        # plt.xticks(np.arange(0.5, len(conditions_played), 1), (0, 0.5, 1, 3, 5, 7.5, 10, 15), rotation=0)
        # plt.x
        # plt.xlabel('Frequency')
        # plt.ylabel('Neuron Index')
        #

        # calculate the auROC for each trial
        # behave_threshold = plot_psychometric(conditions_played, responses, mouse='Mouse1', color='blue', output_dir=save_loc)
        # auROC, prob_valve, prob_baseline = auROC_with_shuffle_parallel(all_values, dff_run_list, 'auROC', n_shuffles=500, alpha=0.05)
        # significant_neurons, significant_neurons_gauss = auROC_neurometric_curve(auROC, conditions_played, behave_threshold, save_loc)
        # # save the auROC values
        # np.save(join(save_loc, 'auROC.npy'), auROC)
        #
        # # # Calculate and save maximum responsive fraction analysis
        # calculate_max_responsive_fraction(auROC, conditions_played, significant_neurons, significant_neurons_gauss, save_loc)
        # print("Done calculating auROC and neurometric curves.")

        significant_neurons = np.load(join(save_loc, 'significant_neurons.npy'))
        significant_neurons_gauss = np.load(join(save_loc, 'significant_neurons_gauss.npy'))
        # # Create activity plots for all neurons with classification labels
        create_all_neuron_activity_plots(dff_run_list_sub, conditions_played,
                                         significant_neurons, significant_neurons_gauss,
                                         save_loc, pre_trial_length, rec_hz)

        # Create auROC method illustration (only for first mouse to avoid duplicates)
        # if mouse_id == mice_days[0][0]:
        #     # Define specific neuron and trial indices for representative examples
        #     # You can modify these indices to use different neurons and trials for each figure type
        #     # Format: {'type': {'neuron': neuron_index, 'trial': trial_index}}
        #     representative_examples = {
        #         'high': {'neuron': 121, 'trial': 5},      # High auROC neuron, trial 5
        #         'low': {'neuron': 518, 'trial': 12},      # Low auROC neuron, trial 12
        #         'neutral': {'neuron': 45, 'trial': 8}     # Neutral auROC neuron, trial 8
        #     }
        #
        #     create_auroc_method_illustration(dff_run_list, auROC, conditions_played,
        #                                    significant_neurons, save_loc, pre_trial_length, rec_hz,
        #                                    representative_examples)


