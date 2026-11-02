"""
Running dF/F of one 2-photon session.

For each neuron and run, F0 is the rolling 10th percentile of the fluorescence over a
4.5-min window (270 s), and dF/F = (F - F0) / F0; the runs are concatenated and saved as
dff_running_4.5min.pkl (neurons x frames). Input: the fluorescence after removal of
movement artifacts (clean_movment_artifacts/chosen/clean_data_unz.pkl) and run_len.npy
from 2p_analysis_preprocessing.py.
"""

import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from turtle import clear
import h5py
import numpy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import plotly.express as px
import seaborn
from numpy import isin
import matplotlib
matplotlib.use('Qt5Agg')


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


def save_data(data, file_path):
    directory = os.path.dirname(file_path)
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def runing_dff(signal, run_lengths, window_size, save_location):
    # Convert the signal to a pandas DataFrame if it isn't already
    if not isinstance(signal, pd.DataFrame):
        signal = pd.DataFrame(signal)

    # Create an empty DataFrame to hold the df/f values for each neuron
    dff = pd.DataFrame()

    # Loop over the rows (neurons) in the DataFrame and calculate df/f for each one
    for i in range(signal.shape[0]):
        dff_runs = pd.Series()  # to store concatenated runs for the current neuron

        start_idx = 0
        for length in run_lengths:
            # Extract the signal for the current run
            end_idx = start_idx + length
            signal_run = signal.iloc[i][start_idx:end_idx]

            # Calculate the rolling 10th percentile to serve as F (baseline)
            F_run = signal_run.rolling(window=window_size, min_periods=window_size).quantile(0.1)
            F_run = F_run.fillna(method='bfill')  # fill the values with the next valid value
            dff_run = (signal_run - F_run) / F_run

            # Update start index for the next run and concatenate the df/f for the current run
            start_idx = end_idx
            dff_runs = pd.concat([dff_runs, dff_run])

        # Assign the concatenated runs' df/f to the DataFrame
        dff[i] = dff_runs

    dff = dff.transpose()  # Transpose to match the original format
    save_data(dff, save_location)
    return dff


if '__main__' == __name__:
    days = ['sess16']
    for day in days:
        ROOT_PATH = Path(DATA_ROOT + r'\mouse16\{}\ML\run2\data'.format(day))
        clean_data = Path(join(ROOT_PATH, 'clean_movment_artifacts\chosen'))
        # load the fluorescence (after movement-artifact cleaning) and the run lengths
        # f_signal = np.load(join(ROOT_PATH, 'f_signal.npy'))
        with open(join(clean_data, 'clean_data_unz.pkl'), 'rb') as f:
            f_clean_unz = pickle.load(f)
        run_len = np.load(join(ROOT_PATH, 'run_len.npy'))

        # calculate the df/f for the running time window and save it in a list
        df_f = runing_dff(f_clean_unz, run_len, 270 * rec_hz, save_location=join(ROOT_PATH, 'dff_running_4.5min.pkl'))  # 4.5 min window
