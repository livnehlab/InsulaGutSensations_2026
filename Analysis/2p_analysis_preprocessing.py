"""
Preprocessing of one 2-photon session: behavior, sync signals and fluorescence.

Loads the MonkeyLogic .mat files (condition and response code of each trial) and the
ThorSync .h5 files of each run, downsamples the sync signals to the 2-photon frame rate,
detects the trial onsets from the behavioral-code channel and splits the lick, valve,
opto and distractor-light signals into trials. Saves to <PATH_ML>/data:
conditions_played.npy, responses.npy, trials_ind.npy, run_len.npy, licks_all.pkl,
valve1_open_all.pkl, opto_open_all.pkl, distract_light_on_all.pkl, and f_signal.npy
(Suite2p fluorescence of the cells after neuropil subtraction).
Next step: 2p_analysis_df.py (running dF/F).
"""

import os
import pickle
from pathlib import Path
from turtle import clear
import h5py
import matplotlib
import numpy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import plotly.express as px
import seaborn
from numpy import isin

matplotlib.use('Qt5Agg')


# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)

PATH_ML = Path(DATA_ROOT + r'\mouse16\sess16\ML\run1')  # the plots will be saved here
PATH_suite2p = Path(DATA_ROOT + r'\mouse16\sess16\mouse16_run1\suite2p\plane0')

start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000
space_between_cues = rec_hz * 9
first_space = rec_hz * 2
pre_trial_length = 4 * rec_hz


def identify_cues(signals):
    """ Find cues timing in thorsync data """
    signals_frames = []
    current_state = 0
    space = 0
    for i in range(np.size(signals)):
        if signals_frames:
            space_cues = space_between_cues
        else:
            space_cues = first_space
        if signals.iloc[:, i][0] > 3:
            if current_state == 1:
                pass
            elif space > space_cues:
                signals_frames.append(i)
                current_state = 1
                space = 0
            else:
                current_state = 1
        else:
            if current_state == 1:
                space = 0
                current_state = 0
            else:
                space += 1

    return signals_frames


def load_mat_file(path):
    """ Loading MonkeyLogic file.
        Returning cues type (not response)
        """
    mat_data = scipy.io.loadmat(path)
    cues = mat_data['TrialRecord'][0][0][7][0]
    responses = mat_data['TrialRecord'][0][0]['TrialErrors'][0]

    return cues, responses

# def load_sync_data(run_data):
#     frames_count = run_data['CI']['FrameCounter']
#     frames_count = pd.DataFrame(frames_count)
#     frames_per_intrval = int(sync_hz / rec_hz)
#     first_frame = (frames_count == 1).idxmax().min()
#     data_points = frames_count[first_frame:len(frames_count):frames_per_intrval]
#     highest_frame = data_points.max()[0]
#     last_frame_idx = data_points.idxmax()[0]
#     # find the indices in which frames were changed (the recording Hz)
#     data_points.drop_duplicates(inplace=True)
#
#     if set(range(1, highest_frame + 1)) == set(data_points[0]):
#         print("Array contains all the frames")
#     else:
#         missing_frames = set(range(1, highest_frame + 1)) - set(data_points[0])
#         print("Array is missing the following frames between 1 and highest frame: ", missing_frames)
#
#     if last_frame_idx == data_points.idxmax()[0]:
#         print("data points created well")
#     else:
#         print('an error with creating data points')
#     Licks = reduce_hz(data_points, pd.DataFrame(run_data['AI']['ai6']))
#     valve_1_data = reduce_hz(data_points, pd.DataFrame(run_data['AI']['ai2']))
#     BC = reduce_hz(data_points, pd.DataFrame(run_data['AI']['ai3']))  # behavioral analysis_code - to get the different trials
#     opto_data = reduce_hz(data_points, pd.DataFrame(run_data['AI']['ai4']))
#     distract_light_data = reduce_hz(data_points, pd.DataFrame(run_data['AI']['ai01']))
#
#     # plot all the signals
#     # plt.plot(valve_1_data)
#     # plt.plot(opto_data)
#     # plt.plot(distract_light_data)
#     # plt.plot(BC)
#     # plt.legend(['valve1', 'opto', 'distract_light', 'BC'])
#     # plt.show()
#
#     return BC, frames_count, Licks, valve_1_data, opto_data, distract_light_data
#
#
# def reduce_hz(counter, data_a):
#     """ Receiving ThorSync array.
#         Down sampling to 2P HZ based on the frame counter output"""
#     new_data = np.zeros((1, len(counter)))
#     # new_data = data_a.iloc[counter.index]
#     for i in range(0, len(counter)):
#         if i == 0:
#             # new_data[i] = data_a.iloc[0:counter.index[i]].max()
#             continue
#         else:
#             new_data[0, i-1] = data_a.iloc[counter.index[i - 1]:counter.index[i]].max()
#
#     return new_data
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

def reduce_hz(counter, data_a):
    """Downsample data from sync_hz to rec_hz based on frame indices."""
    new_data = np.zeros((1, len(counter)))
    for i in range(1, len(counter)):
        new_data[0, i - 1] = data_a.iloc[counter.iloc[i - 1][0]:counter.iloc[i][0]].max()[0]
    return new_data


def read_suite2p(data_path):
    F_data = np.load(data_path / 'F.npy')
    Fneu_data = np.load(data_path / 'Fneu.npy')
    iscell = np.load(data_path / 'iscell_final.npy')

    return F_data, Fneu_data, iscell


def proccess_Fdata(suite2p_path):
    F, Fneu, iscell = read_suite2p(suite2p_path)
    ROI = np.where(iscell[:, 0] == 1)
    F_cells = F[ROI]
    Fneu_cells = Fneu[ROI]

    # Calculate the median of Fneu_cells along axis 1 (i.e., for each neuron)
    Fneu_cells_median = np.median(Fneu_cells, axis=1)

    # Subtract 70% of Fneu_cells from F_cells for each neuron
    F_cells_processed = F_cells - 0.7 * Fneu_cells

    # Add the median of Fneu_cells for each neuron to F_cells_processed
    for i in range(F_cells_processed.shape[0]):
        F_cells_processed[i, :] += Fneu_cells_median[i]
    return F_cells_processed


def save_data(data, file_path):
    directory = os.path.dirname(file_path)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if '__main__' == __name__:
    # get the conditions played and the licking data divided by trials
    conditions_played = []
    responses = []
    opto_on_all = []
    licks_all = []
    valve1_open_all = []
    distract_light_on_all = []
    cues_data = []
    trials_ind = []
    run_len = []
    df_trials = []
    folder = PATH_ML
    for mat_file in folder.glob('*.mat'):
        print(mat_file)
        run = mat_file.stem[-1]
        conditions_played_run, responses_run = load_mat_file(mat_file)
        conditions_played.extend(conditions_played_run)
        responses.extend(responses_run)
    conditions_played = np.asarray(conditions_played)
    responses = np.asarray(responses)

    for sync_file in folder.glob('*.h5'):
        print(sync_file)
        data = h5py.File(sync_file)
        cues_data_run, frames_counter_run, Licks_output, valve_1_opens, opto_on, distract_light_on = load_sync_data(
            data,
            rec_hz, lick_hz=30, sync_hz=sync_hz)
        Licks_output = Licks_output.flatten()
        valve_1_opens = valve_1_opens.flatten()
        opto_on = opto_on.flatten()
        distract_light_on = distract_light_on.flatten()
        cues_data_run = pd.DataFrame(cues_data_run)
        trials_ind_run = identify_cues(cues_data_run)
        trials_ind_run = np.asarray(trials_ind_run)
        # cues_data_run.reset_index(drop=True, inplace=True)
        cues_data_run = np.asarray(cues_data_run)
        cues_data_run = cues_data_run.flatten()
        plt.plot(cues_data_run)
        cues_data_run = cues_data_run.transpose()
        # mark the trials
        for index in trials_ind_run:
            plt.annotate('*', xy=(index, cues_data_run[index]))
        plt.close()
        trials_ind.append(trials_ind_run)
        # Opto_data_run = np.split(Opto_data, trials_ind_run, axis=0)
        # Opto_data_all.extend(Opto_data_run[1:])
        licks_run = np.split(Licks_output, trials_ind_run, axis=0)
        licks_all.extend(licks_run[1:])
        valve1_open_run = np.split(valve_1_opens, trials_ind_run, axis=0)
        valve1_open_all.extend(valve1_open_run[1:])
        opto_on_run = np.split(opto_on, trials_ind_run, axis=0)
        opto_on_all.extend(opto_on_run[1:])
        distract_light_on_run = np.split(distract_light_on, trials_ind_run, axis=0)
        distract_light_on_all.extend(distract_light_on_run[1:])
        cues_data_run = cues_data_run.transpose()
        run_len.append(len(cues_data_run))

    # # plot the signals of each run
    # for i in range (0, len(conditions_played)):
    #     plt.figure()
    #     plt.plot(distract_light_on_all[i])
    #     plt.plot(valve1_open_all[i])
    #     plt.plot(opto_on_all[i])
    #     plt.plot(licks_all[i])
    #     plt.legend(['distract_light', 'valve1', 'opto', 'licks'])
    #     plt.show()
    #     # wait for the user to press a key
    #     input("Press Enter to continue...")

    # save the licking data and conditions played
    folder = join(PATH_ML, 'data')
    os.makedirs(folder, exist_ok=True)
    # save_data(Opto_data_all, join(folder, 'Opto_data_all.pkl')) # save data that is not numpy array
    save_data(licks_all, join(folder, 'licks_all.pkl'))
    save_data(valve1_open_all, join(folder, 'valve1_open_all.pkl'))
    save_data(opto_on_all, join(folder, 'opto_open_all.pkl'))
    save_data(distract_light_on_all, join(folder, 'distract_light_on_all.pkl'))
    np.save(join(folder, 'conditions_played.npy'), conditions_played)
    np.save(join(folder, 'responses.npy'), responses)
    np.save(join(folder, 'trials_ind.npy'), trials_ind)
    np.save(join(folder, 'run_len.npy'), run_len)

    # get the f signal after subtraction of the neuropil
    f_signal = proccess_Fdata(PATH_suite2p)
    np.save(join(folder, 'f_signal.npy'), f_signal)
    f_signal = pd.DataFrame(f_signal)
    #
