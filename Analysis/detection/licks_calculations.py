"""
Lick analysis of one session: lick rate, inter-lick intervals, lick onset / offset and
reward times per trial (lick_onset_offset_results.pkl, used by the events scripts), plus
lick-rate figures and the proportion of false alarms across the session.

All trials are kept (PRIMING_TRIALS = 0); the psychophysics scripts remove the priming
trials themselves.

Condition and response codes:
 - condition 6 = 0 hz and responses marker 2 = didn't lick, 3 = licked
 - condition 5 or condition 7 = 15 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 8 = 10 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 9 = 7.5 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 10 = 5 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 11 = 3 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 12 = 1 hz and responses marker 4 = didn't lick, 5 = licked
 - condition 13 = 0.5 hz and responses marker 4 = didn't lick, 5 = licked
"""

import os
import pickle
from pathlib import Path
import matplotlib
import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
matplotlib.use('Qt5Agg')
matplotlib.rcParams['pdf.fonttype'] = 42


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)

# Constants
start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000
pre_trial_length = rec_hz * 4
space_between_cues = rec_hz * 9
first_space = rec_hz * 2
PRIMING_TRIALS = 0



######################
# Utility Functions
######################
def cues_time(data, n_cues):
    """Find the cue timing in each trial."""
    cues_timing = []
    for trial in range(1, n_cues + 1):
        behavioral_codes = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][1]
        codes_timing = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][0]
        start_cue = np.where(behavioral_codes == start_cue_code)[0]
        stop_cue = np.where(behavioral_codes == stop_cue_code)[0]

        if start_reward_code in behavioral_codes:
            start_reward = np.where(behavioral_codes == start_reward_code)[0]
            reward_num = len(np.where(behavioral_codes == start_reward_code)[0])
            cues_timing.append((codes_timing[start_cue][0][0],
                                codes_timing[stop_cue][0][0],
                                codes_timing[start_reward][0][0],
                                codes_timing[start_reward][-1][0],
                                reward_num))
        else:
            cues_timing.append((codes_timing[start_cue][0][0], codes_timing[stop_cue][0][0]))
    return cues_timing


def identify_cues(signals):
    """Find cue timings (trial starts) in ThorSync data."""
    signals_frames = []
    current_state = 0
    space = 0

    for i in range(np.size(signals, 1)):
        if signals_frames:
            space_cues = space_between_cues
        else:
            space_cues = first_space

        if signals.iloc[:, i][0] > 3:
            if current_state == 0 and space > space_cues:
                signals_frames.append(i)
                current_state = 1
                space = 0
            else:
                current_state = 1
        else:
            if current_state == 1:
                current_state = 0
                space = 0
            else:
                space += 1
    return signals_frames


def load_mat_file(path):
    """Load MonkeyLogic .mat file and extract conditions and responses."""
    mat_data = scipy.io.loadmat(path)
    cues = mat_data['TrialRecord'][0][0][7][0]
    responses = mat_data['TrialRecord'][0][0]['TrialErrors'][0]
    return cues, responses


def reduce_hz(counter, data_a):
    """Downsample data from sync_hz to rec_hz based on frame indices."""
    new_data = np.zeros((1, len(counter)))
    for i in range(1, len(counter)):
        new_data[0, i - 1] = data_a.iloc[counter.iloc[i - 1][0]:counter.iloc[i][0]].max()[0]
    return new_data


# def load_sync_data(data):
#     """Load ThorSync data and downsample signals to rec_hz."""
#     frames_count = pd.DataFrame(data['CI']['FrameCounter'])
#     frames_per_intrval = int(sync_hz / rec_hz)
#     data_points = frames_count[frames_count.index % frames_per_intrval == 0]
#
#     Licks = reduce_hz(data_points, pd.DataFrame(data['AI']['ai6']))
#     valve_1_data = reduce_hz(data_points, pd.DataFrame(data['AI']['ai2']))
#     BC = reduce_hz(data_points, pd.DataFrame(data['AI']['ai3']))
#     opto_data = reduce_hz(data_points, pd.DataFrame(data['AI']['ai4']))
#     distract_light_data = reduce_hz(data_points, pd.DataFrame(data['AI']['ai01']))
#
#     return BC, frames_count, Licks, valve_1_data, opto_data, distract_light_data
# def load_sync_data(run_data, rec_hz=15, lick_hz=30, sync_hz=6000):
#     """
#     run_data: dictionary holding all the signals, e.g. run_data['AI']['ai6'] for licks.
#     rec_hz:   the desired down-sample rate for most signals (15 Hz).
#     lick_hz:  the desired down-sample rate for the Licks channel (30 Hz).
#     sync_hz:  the original sampling rate (e.g. 300 Hz).
#     """
#     # 1) Get the frame counter array
#     frames_count = pd.DataFrame(run_data['CI']['FrameCounter'])
#
#     # 2) Compute different down-sample intervals
#     frames_per_interval_main = int(sync_hz / rec_hz)  # for 15 Hz
#     frames_per_interval_licks = int(sync_hz / lick_hz)  # for 30 Hz
#
#     # 3) Find the first frame that has counter == 1
#     first_frame = (frames_count == 1).idxmax().min()
#
#
#     # 4) Create two sets of 'data_points' indices
#     data_points_main = frames_count[first_frame: len(frames_count): frames_per_interval_main]
#     data_points_licks = frames_count[first_frame: len(frames_count): frames_per_interval_licks]
#
#     highest_frame = data_points_main.max()[0]
#     last_frame_idx = data_points_main.idxmax()[0]
#     data_points_main.drop_duplicates(inplace=True)
#     # 6) Remove excessive duplicates from licks (allow up to 2 total per value)
#     #    6a) Rename column 0 -> "FrameCounter" if needed
#     data_points_licks.columns = ["FrameCounter"]
#
#     #    6b) Group by "FrameCounter" and count occurrences
#     data_points_licks["dup_count"] = data_points_licks.groupby("FrameCounter").cumcount() + 1
#
#     #    6c) Keep only rows where each FrameCounter value has occurred <= 2 times
#     data_points_licks = data_points_licks[data_points_licks["dup_count"] <= 2].copy()
#
#     #    6d) Drop the helper column
#     data_points_licks.drop("dup_count", axis=1, inplace=True)
#
#
#     if set(range(1, highest_frame + 1)) == set(data_points_main[0]):
#         print("Array contains all the frames")
#     else:
#         missing_frames = set(range(1, highest_frame + 1)) - set(data_points_main[0])
#         print("Array is missing the following frames between 1 and highest frame: ", missing_frames)
#
#     if last_frame_idx == data_points_main.idxmax()[0]:
#         print("data points created well")
#     else:
#         print('an error with creating data points')
#
#     # 5) Downsample the signals to 15 Hz
#     valve_1_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai2']))
#     BC = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai3']))
#     opto_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai4']))
#     distract_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai01']))
#
#     # (6) Downsample Licks to 30 Hz
#     Licks_30Hz = reduce_hz(data_points_licks, pd.DataFrame(run_data['AI']['ai6']))
#
#     # (7) "Clean" the 30 Hz Licks
#     Licks_30Hz_flat = Licks_30Hz.squeeze()
#
#     lick_clean_30Hz = np.zeros_like(Licks_30Hz_flat)
#     for l in range(1, len(Licks_30Hz_flat)):
#         if (Licks_30Hz_flat[l] > 3) and (Licks_30Hz_flat[l - 1] <= 3):
#             lick_clean_30Hz[l] = 5
#         else:
#             lick_clean_30Hz[l] = 0
#
#     # (8) downsample the cleaned 30-Hz array to 15 Hz
#     lick_clean_15Hz = []
#     for i in range(0, len(lick_clean_30Hz), 2):
#         lick_clean_15Hz.append(max(lick_clean_30Hz[i: i + 2]))
#     lick_clean_15Hz = np.array(lick_clean_15Hz)
#
#     return BC, frames_count, lick_clean_15Hz, valve_1_data, opto_data, distract_data
def find_rises_indices(signal, min_length=5):
    """
    Find all segments of consecutive 1s that have length >= min_length and
    return their indices.

    Parameters
    ----------
    signal     : array-like of int (0 or 1)
        The digital signal.
    min_length : int
        Minimum number of consecutive 1s to qualify as a rise.

    Returns
    -------
    list_of_indices : list of lists
        Each element is a list that contains the indices of one valid rise.
    """
    signal = np.asarray(signal, dtype=int)
    is_one = (signal == 1)

    list_of_indices = []
    start = None  # Start index of a consecutive run of 1s

    # Iterate through the signal
    for i, val in enumerate(is_one):
        if val:
            # If we see a '1' and haven't started a run, mark start
            if start is None:
                start = i
        else:
            # If we see a '0' and there was a run of 1s, finalize it
            if start is not None:
                run_length = i - start
                # If this run is long enough, record its indices
                if run_length >= min_length:
                    list_of_indices.append(start)
                start = None

    # Edge case: the signal might end with a run of 1s
    if start is not None:
        run_length = len(is_one) - start
        if run_length >= min_length:
            list_of_indices.append(start)

    return list_of_indices

def load_sync_data(run_data, rec_hz=15, lick_hz=30, sync_hz=6000):
    """
    run_data: dictionary holding all the signals, e.g. run_data['AI']['ai6'] for licks.
    rec_hz:   the desired down-sample rate for most signals (15 Hz).
    lick_hz:  the desired down-sample rate for the Licks channel (30 Hz).
    sync_hz:  the sampling rate of the Thorsync (e.g. 6000 Hz).
    """
    # 1) Get the frame out array
    frames_out = pd.DataFrame(run_data['DI']['FrameOut'])
    min_length = 5 # Minimum number of consecutive 1s to qualify as a rise
    # 2) Create two sets of 'data_points' indices
    data_points_main = find_rises_indices(frames_out, min_length)
    # for the licks, we want to double the frame rate
    data_points_licks = []
    for i in range(1, len(data_points_main)):
        data_points_licks.append(data_points_main[i-1])
        data_points_licks.append(data_points_main[i-1] + (data_points_main[i] - data_points_main[i-1])//2)
    data_points_licks.append(data_points_main[-1])
    data_points_licks.append(data_points_main[-1] + sync_hz//lick_hz)
    data_points_licks = pd.DataFrame(data_points_licks)
    data_points_main = pd.DataFrame(data_points_main)

    # print the number of frames to make sure it is correct
    num_frames = len(data_points_main)
    print(f"Number of frames: {num_frames}")

    # 3) Downsample the signals to 15 Hz
    valve_1_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai2']))
    BC = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai3']))
    opto_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai4']))
    distract_data = reduce_hz(data_points_main, pd.DataFrame(run_data['AI']['ai01']))

    # (4) Downsample Licks to 30 Hz
    Licks_30Hz = reduce_hz(data_points_licks, pd.DataFrame(run_data['AI']['ai6']))

    # (5) "Clean" the 30 Hz Licks
    Licks_30Hz_flat = Licks_30Hz.squeeze()

    lick_clean_30Hz = np.zeros_like(Licks_30Hz_flat)
    for l in range(1, len(Licks_30Hz_flat)):
        if (Licks_30Hz_flat[l] > 3) and (Licks_30Hz_flat[l - 1] <= 3):
            lick_clean_30Hz[l] = 5
        else:
            lick_clean_30Hz[l] = 0

    # (6) downsample the cleaned 30-Hz array to 15 Hz
    lick_clean_15Hz = []
    for i in range(0, len(lick_clean_30Hz), 2):
        lick_clean_15Hz.append(max(lick_clean_30Hz[i: i + 2]))
    lick_clean_15Hz = np.array(lick_clean_15Hz)

    return BC, frames_out, lick_clean_15Hz, valve_1_data, opto_data, distract_data

def add_axvline_opto(pre_trial_length, rec_hz):
    # distract light
    # Draw a black rectangle from pre_trial_length to pre_trial_length + 9 * rec_hz
    plt.axvspan(pre_trial_length, pre_trial_length + 9, color='khaki', alpha=0.4, label='Distract Light')

    # Draw a yellow rectangle for 'opto' from pre_trial_length + 16 to pre_trial_length + 16 + 4 * rec_hz
    plt.axvspan(pre_trial_length + 1, pre_trial_length + 1 + 4, color='yellow', alpha=0.4, label='Opto')

    # Draw a green rectangle for 'odor' from pre_trial_length + 5 * rec_hz to pre_trial_length + 5 * rec_hz + 2 * rec_hz
    plt.axvspan(pre_trial_length + 5, pre_trial_length + 5 + 2, color='gray', alpha=0.2, label='Odor')


def add_axvline(rewards, pre_trial_length, rec_hz):
    # distract light
    # Draw a black rectangle from pre_trial_length to pre_trial_length + 9 * rec_hz
    plt.axvspan(pre_trial_length, pre_trial_length + 9, color='khaki', alpha=0.4, label='Distract Light')
    # Draw a green rectangle for 'odor' from pre_trial_length + 5 * rec_hz to pre_trial_length + 5 * rec_hz + 2 * rec_hz
    plt.axvspan(pre_trial_length + 5, pre_trial_length + 5 + 2, color='gray', alpha=0.2, label='Odor')
    # Adding asterisks where rewards occurred
    for index, reward in enumerate(rewards):
        if len(reward) > 0:
            # Placing the '*' marker on the heatmap at the specified reward index for each trial
            # Adjust coordinates if needed, considering the heatmap's axis configuration
            plt.text(reward[0]/rec_hz, index + 0.5, '*', color='blue', ha='center', va='center', fontsize=10, label='Reward')
######################
# Main Analysis
######################
if __name__ == '__main__':
    days = ['sess06']
    for day in days:
        PATH_ML = Path(DATA_ROOT + rf'\mouse09\{day}\ML')
        output_dir = join(PATH_ML, 'licking_analysis_final')
        os.makedirs(output_dir, exist_ok=True)

        # Load conditions and responses
        conditions_played = []
        responses = []
        for mat_file in PATH_ML.glob('*.mat'):
            conditions_played_run, responses_run = load_mat_file(mat_file)
            responses.extend(responses_run)
            conditions_played.extend(conditions_played_run)

        conditions_played = np.asarray(conditions_played)
        responses = np.asarray(responses)

        if len(responses) <= PRIMING_TRIALS:
            raise ValueError(
                f"Not enough trials after removing priming trials: total {len(responses)}, "
                f"required more than {PRIMING_TRIALS}."
            )

        conditions_played = conditions_played[PRIMING_TRIALS:]
        responses = responses[PRIMING_TRIALS:]

        # Load ThorSync data
        licks_output_all = []
        valve1_open_all = []
        trials_ind = []
        for sync_file in PATH_ML.glob('*.h5'):
            data = h5py.File(sync_file)
            cues_data_run, frames_counter_run, Licks_output, valve_1_opens, opto_on, distract_light_on = load_sync_data(data)
            Licks_output = Licks_output.flatten()
            valve_1_opens = valve_1_opens.flatten()
            cues_data_run = pd.DataFrame(cues_data_run)
            trials_ind_run = identify_cues(cues_data_run)
            trials_ind.extend(trials_ind_run)

            # Split into trials
            licks_output_run = np.split(Licks_output, trials_ind_run, axis=0)
            licks_output_all.extend(licks_output_run[1:])
            valve1_open_run = np.split(valve_1_opens, trials_ind_run, axis=0)
            valve1_open_all.extend(valve1_open_run[1:])
            distract_light_on = np.split(distract_light_on, trials_ind_run, axis=0)

        if len(licks_output_all) <= PRIMING_TRIALS:
            raise ValueError(
                f"Not enough trials after removing priming trials: total {len(licks_output_all)}, "
                f"required more than {PRIMING_TRIALS}."
            )

        licks_output_all = licks_output_all[PRIMING_TRIALS:]
        valve1_open_all = valve1_open_all[PRIMING_TRIALS:]


        # Conditions indices
        correct_reject = np.where(responses == 4)[0]
        false_alarm = np.where(responses == 5)[0]
        hit = np.where(responses == 3)[0]
        miss = np.where(responses == 2)[0]

        # test in which portion of the session the false alarm and hit
        num_trials = len(responses)
        all_trials = np.arange(num_trials)

        n_bins = 4
        bin_size = num_trials // n_bins

        bins = []
        for i in range(n_bins - 1):
            bins.append(all_trials[i * bin_size: (i + 1) * bin_size])
        bins.append(all_trials[(n_bins - 1) * bin_size:])

        fa_counts = []
        other_counts = []
        for b in bins:
            fa_in_bin = np.intersect1d(false_alarm, b)
            fa_count = len(fa_in_bin)
            total_bin = len(b)
            other_count = total_bin - fa_count
            fa_counts.append(fa_count)
            other_counts.append(other_count)

        # calculate the proportion of false alarm in each bin
        fa_proportion = np.array(fa_counts) / (np.array(fa_counts) + np.array(other_counts))
        print(fa_proportion)
        #save
        with open(join(output_dir, 'fa_proportion.pkl'), 'wb') as f:
            pickle.dump(fa_proportion, f)



        # Determine max number of frames
        max_number_of_frames = max([len(frames) for frames in licks_output_all[:-1]])
        trials_licks = np.zeros((len(licks_output_all), max_number_of_frames + pre_trial_length))
        trials_licks[:] = np.nan

        # Process rewards
        for i in range(len(valve1_open_all)):
            changes = np.diff(np.concatenate(([0], valve1_open_all[i] > 1)))
            valve1_open_all[i] = np.where(changes == 1)[0]
        rewards_times = valve1_open_all
        # fill missing rewards with nan
        for i in range(len(rewards_times)):
            if len(rewards_times[i]) == 0:
                rewards_times[i] = np.array([np.nan])
        # add the pre_trial_length to the rewards
        for i in range(len(rewards_times)):
            rewards_times[i] = rewards_times[i] + pre_trial_length

        # Identify lick frames per trial
        for i in range(len(licks_output_all)):
            if i > 0:
                pre_cue = licks_output_all[i - 1][len(licks_output_all[i - 1]) - pre_trial_length:]
            else:
                pre_cue = np.zeros(pre_trial_length)

            concat_trial = np.concatenate([pre_cue, licks_output_all[i]])
            licks_output_all[i] = concat_trial
            concat_licks = np.asarray(licks_output_all[i])

            lick_frames = []
            for l in range(1, len(concat_licks)):
                if concat_licks[l] > 3 : #and concat_licks[l - 1] <= 3
                    lick_frames.append(l)

            lick_frames = np.array(lick_frames)
            if len(lick_frames) > 0:
                trials_licks[i, :len(lick_frames)] = lick_frames

        # Compute ILIs
        num_trials, num_frames = trials_licks.shape
        ITI_licks = []
        for trial in range(num_trials):
            lick_times = trials_licks[trial, :]
            lick_times = lick_times[~np.isnan(lick_times)]
            if len(lick_times) > 1:
                ILIs = np.diff(lick_times) / rec_hz * 1000
                ITI_licks.append(ILIs)
            else:
                ITI_licks.append(np.array([]))

        # Flatten ILIs for histogram
        all_ITIs = np.hstack(ITI_licks)
        plt.figure()
        plt.hist(all_ITIs, bins=50, color='blue', alpha=0.7)
        plt.xlabel('Inter-Lick Interval (ms)')
        plt.ylabel('Count')
        plt.title('Inter-Lick Interval')
        plt.show()

        # Save licking data by condition
        licking_data = {
            'correct_reject': trials_licks[correct_reject, :],
            'false_alarm': trials_licks[false_alarm, :],
            'hit': trials_licks[hit, :],
            'miss': trials_licks[miss, :]
        }

        ILI = {
            'correct_reject': [ITI_licks[i] for i in correct_reject],
            'false_alarm': [ITI_licks[i] for i in false_alarm],
            'hit': [ITI_licks[i] for i in hit],
            'miss': [ITI_licks[i] for i in miss]
        }

        with open(join(output_dir, 'licking_data.pkl'), 'wb') as f:
            pickle.dump(licking_data, f)
        with open(join(output_dir, 'ILI.pkl'), 'wb') as f:
            pickle.dump(ILI, f)

        #############################################
        # Add lick rate calculation and onset/offset
        #############################################
        pause_threshold_ms = 500.0
        bin_size = 1.0
        min_consecutive_licks = 5  # lick threshold

        trial_length_s = (max_number_of_frames + pre_trial_length) / rec_hz
        time_vector = np.arange(0, trial_length_s, 1 / rec_hz)

        lick_onsets = []
        lick_offsets = []
        lick_rates_all = []

        for trial in range(num_trials):
            lick_frames = trials_licks[trial, :]
            lick_frames = lick_frames[~np.isnan(lick_frames)].astype(int)

            # ---------------------------------------------
            # If no licks, store NaNs and continue
            # ---------------------------------------------
            if len(lick_frames) == 0:
                lick_onsets.append(np.nan)
                lick_offsets.append(np.nan)
                lick_rates_all.append(np.zeros_like(time_vector))
                continue

            lick_times_s = lick_frames / rec_hz

            # ---------------------------------------------
            # Compute ILIs (in ms)
            # ---------------------------------------------
            if len(lick_times_s) > 1:
                ILIs_ms = np.diff(lick_times_s) * 1000
            else:
                ILIs_ms = np.array([])

            # ---------------------------------------------
            # Compute lick rate (licks/s)
            # ---------------------------------------------
            lick_rate = np.zeros_like(time_vector)
            for i, t in enumerate(time_vector):
                start_t = max(t - bin_size / 2, 0)
                end_t = t + bin_size / 2
                lick_count = np.sum((lick_times_s >= start_t) & (lick_times_s <= end_t))
                lick_rate[i] = lick_count / bin_size
            lick_rates_all.append(lick_rate)

            # ---------------------------------------------
            # Identify all candidate onsets
            # ---------------------------------------------
            trial_start_s = pre_trial_length / rec_hz
            candidate_onsets = []

            # (1) If there's a quiet period >= pause_threshold_ms
            #     before the first lick *after* the trial start
            post_cue_licks = lick_times_s[lick_times_s >= trial_start_s]
            if len(post_cue_licks) > 0:
                first_lick_after_start = post_cue_licks
                silence_before = (first_lick_after_start[0] - trial_start_s) * 1000
                if silence_before >= pause_threshold_ms:
                    candidate_onsets.append(first_lick_after_start)

            # (2) If there's a long pause inside the trial
            # if len(ILIs_ms) > 0:
            #     long_pause_indices = np.where(ILIs_ms > pause_threshold_ms)[0]
            #     for idx_pause in long_pause_indices:
            #         # The "next lick" after a long pause is a candidate
            #         if (idx_pause + 1) < len(lick_times_s):
            #             candidate_onsets.append(lick_times_s[idx_pause + 1])

            # ---------------------------------------------
            # Sort candidates in ascending order
            # ---------------------------------------------
            candidate_onsets = np.array(candidate_onsets)
            # candidate_onsets.sort()

            # ---------------------------------------------
            # Now pick the *earliest valid* onset
            # ---------------------------------------------
            onset_candidate = np.nan
            for cand in candidate_onsets.T:
                # Find index in lick_times_s that corresponds to cand
                idx_cand = np.where(lick_times_s >= cand)[0][0]

                # Count consecutive licks within 1 second after cand
                consecutive_licks = 1  # Count the candidate lick itself
                for i_lick in range(idx_cand + 1, len(lick_times_s)):
                    if (lick_times_s[i_lick] - lick_times_s[i_lick - 1]) <= 0.5: # 1 second
                        consecutive_licks += 1
                    else:
                        break

                # If we have enough consecutive licks, accept this onset and stop
                if consecutive_licks >= min_consecutive_licks:
                    onset_candidate = cand[0]
                    break

            # onset_candidate found (or NaN)
            if np.isnan(onset_candidate):
                lick_onsets.append(np.nan)
                lick_offsets.append(np.nan)
            else:
                # Find onset index
                onset_idx = np.where(lick_times_s >= onset_candidate)[0][0]

                # Default offset is the last lick in the trial
                offset_candidate = lick_times_s[-1]

                # Find the first gap bigger than pause_threshold_ms after onset
                for i in range(onset_idx, len(lick_times_s) - 1):
                    inter_lick_interval_ms = (lick_times_s[i + 1] - lick_times_s[i]) * 1000
                    if inter_lick_interval_ms > pause_threshold_ms:
                        # Set offset to the last lick before the gap
                        offset_candidate = lick_times_s[i]
                        break

                lick_onsets.append(onset_candidate)
                lick_offsets.append(offset_candidate)



        # Save onset/offset results
        lick_analysis_results = {
            'onset':  np.array(lick_onsets) * rec_hz,
            'offset': np.array(lick_offsets) * rec_hz,
            'lick_rate': lick_rates_all,
            'rewards_times': np.array(rewards_times)
        }
        with open(join(output_dir, 'lick_onset_offset_results.pkl'), 'wb') as f:
            pickle.dump(lick_analysis_results, f)

        rewards_times = [valve1_open_all[i] for i in hit]
        # Example trial plot
        trial_idx = 0
        plt.figure()
        plt.plot(time_vector, lick_rates_all[trial_idx], label='Lick Rate (licks/s)')
        if not np.isnan(lick_onsets[trial_idx]):
            plt.axvline(lick_onsets[trial_idx], color='g', linestyle='--', label='Onset')
        if not np.isnan(lick_offsets[trial_idx]):
            plt.axvline(lick_offsets[trial_idx], color='r', linestyle='--', label='Offset')
        plt.xlabel('Time (s)')
        plt.ylabel('Lick Rate (licks/s)')
        plt.title('Example Trial: Lick Rate with Onset/Offset')
        plt.legend()
        plt.savefig(join(output_dir, 'example_trial_onset_offset.png'))
        plt.close()

        ######################
        # Raster Plot by Condition
        ######################
        conditions_list = list(licking_data.keys())  # e.g., ['correct_reject', 'false_alarm', 'hit', 'miss']
        figures = [
            (1, trials_licks[correct_reject, :], 'Correct Reject.pdf', correct_reject),
            (2, trials_licks[false_alarm, :], 'False Alarm.pdf', false_alarm),
            (3, trials_licks[hit, :], 'Hit.pdf', hit),
            (4, trials_licks[miss, :], 'Miss.pdf', miss),
        ]

        # Time labels for x-axis
        TICKS_LABELS = np.arange(0, 25, step=5)  # Adjust step size as needed

        # Example parameters (replace with actual values)
        pre_trial = pre_trial_length / rec_hz  # Duration of pre-trial period in seconds
        reward_times = [10, 15]  # Example reward times (seconds)

        for fig_num, data, filename, trial_indices in figures:
            plt.figure(fig_num, figsize=(20, 15))

            # Convert lick data (in frames) to seconds
            data_in_seconds = data / rec_hz

            # Check if empty
            if data_in_seconds.size == 0:
                print(f"Skipping empty data for {filename}")
                continue
            # Plot raster lines for licks
            plt.eventplot(data_in_seconds, orientation='horizontal', linelengths=0.8, linewidths=0.5, color='black')

            # Plot lick onsets and offsets (already in seconds)
            for i, t_idx in enumerate(trial_indices):
                this_onset = lick_onsets[t_idx]
                this_offset = lick_offsets[t_idx]

                # Plot onset if it's not NaN
                if not np.isnan(this_onset):
                    plt.plot(this_onset, i, marker=5, color='red', markersize=5,
                             label='Onset')
                # Plot offset if it's not NaN
                if not np.isnan(this_offset):
                    plt.plot(this_offset, i, marker=4, color='red', markersize=5, label='Offset')

                # Add reward markers for 'hit' condition (figure 3)
                if fig_num == 1 or fig_num == 2:
                    add_axvline_opto(pre_trial, rec_hz)
                    add_axvline_opto(pre_trial, rec_hz)
                elif fig_num == 3:
                    add_axvline(rewards_times, pre_trial, rec_hz)
                else:
                    plt.axvspan(pre_trial, pre_trial + 9, color='khaki', alpha=0.4, label='Distract Light')
                    # Draw a green rectangle for 'odor' from pre_trial_length + 5 * rec_hz to pre_trial_length + 5 * rec_hz + 2 * rec_hz
                    plt.axvspan(pre_trial + 5, pre_trial + 5 + 2, color='gray',
                                alpha=0.2, label='Odor')
            # Formatting x-axis
            plt.xlim(0, 25)  # Assuming the maximum time is 25 seconds
            plt.xticks(TICKS_LABELS, TICKS_LABELS)

            # Y-axis settings
            plt.ylim(-0.5, data.shape[0] - 0.5)
            plt.yticks(np.arange(0, data.shape[0]), np.arange(1, data.shape[0] + 1))
            plt.gca().invert_yaxis()

            # Labels and title
            plt.xlabel('Time (sec)')
            plt.ylabel('Trial')
            plt.title(filename[:-4])

            # Add legend for onset/offset markers and other annotations
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                unique_legend = dict(zip(labels, handles))  # Remove duplicate legend entries
                plt.legend(unique_legend.values(), unique_legend.keys(), loc='upper right')

            # Save and display the plot
            plt.savefig(join(output_dir, filename))
            plt.close()