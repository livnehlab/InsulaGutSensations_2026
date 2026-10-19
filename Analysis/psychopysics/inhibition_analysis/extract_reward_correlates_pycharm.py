#!/usr/bin/env python3
"""
PyCharm-friendly standalone script to extract per-neuron reward-related activity averages.

Edit the USER SETTINGS block below and run the script directly.
No command-line arguments are required.

Expected inputs:
- dff_run_list.pkl : list of neurons, each neuron is trials x frames
- hit.pkl          : indices of reward (hit) trials in the full trial set
- lick_onset_offset_results.pkl : dict containing at least
    - 'onset'
    - 'rewards_times'
"""

import os
import csv
import pickle
from os.path import join

import numpy as np
import pandas as pd


# ===================== USER SETTINGS =====================
mouse = 'OI208'
day = '250227'
DATA_DIR = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_freq_history'.format(mouse, day)
# Folder containing dff_run_list.pkl and hit.pkl

LICK_FILE = r'Z:\2p_data\{}\{}\ML\licking_analysis_final\lick_onset_offset_results.pkl'.format(mouse, day)
# Full path to lick_onset_offset_results.pkl

SAVE_LOC = r"Z:\2p_data\{}\{}\ML\data\processed_data\inhibition_analysis".format(mouse, day)
# Output folder. Leave empty to save into DATA_DIR/reward_related_outputs

REC_HZ = 15.0
PRE_TRIAL_LENGTH = 4 * REC_HZ
ODOR_START_OFFSET_S = 5.0
LICK_BUFFER_MS = 100.0
REWARD_WINDOW_S = (0.0, 2.0)
SWALLOW_WINDOW_S = (1.5, 2.0)
PREFIX = 'reward_related'
# =========================================================


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dataframe_list(dff_obj):
    out = []
    for neuron in dff_obj:
        if isinstance(neuron, pd.DataFrame):
            out.append(neuron)
        else:
            out.append(pd.DataFrame(neuron))
    return out


def safe_window_mean(trace, start, stop):
    if np.isnan(start) or np.isnan(stop):
        return np.nan

    start = int(np.round(start))
    stop = int(np.round(stop))

    start = max(0, start)
    stop = min(len(trace), stop)

    if stop <= start:
        return np.nan

    vals = np.asarray(trace[start:stop], dtype=float)
    if vals.size == 0:
        return np.nan
    return np.nanmean(vals)


def subset_hit_trials(dff_run_list, hit_idx):
    dff_run_list = ensure_dataframe_list(dff_run_list)
    hit_idx = np.asarray(hit_idx, dtype=int)

    dff_hit = []
    for neuron_df in dff_run_list:
        dff_hit.append(neuron_df.iloc[hit_idx, :].copy())
    return dff_hit


def extract_reward_related_averages(
    dff_hit,
    reward_times_hit,
    lick_onset_hit,
    rec_hz,
    pre_trial_length,
    save_loc,
    odor_start_offset_s=5.0,
    lick_buffer_ms=100.0,
    reward_window_s=(0.0, 2.0),
    swallow_window_s=(1.5, 2.0),
    prefix='reward_related',
):
    os.makedirs(save_loc, exist_ok=True)

    reward_times_hit = np.asarray(reward_times_hit, dtype=float)
    lick_onset_hit = np.asarray(lick_onset_hit, dtype=float)
    dff_hit = ensure_dataframe_list(dff_hit)

    n_neurons = len(dff_hit)
    if n_neurons == 0:
        raise ValueError('dff_hit is empty')

    n_trials = dff_hit[0].shape[0]
    if reward_times_hit.shape[0] != n_trials:
        raise ValueError(
            f'reward_times_hit has {reward_times_hit.shape[0]} entries, but dff_hit has {n_trials} trials'
        )
    if lick_onset_hit.shape[0] != n_trials:
        raise ValueError(
            f'lick_onset_hit has {lick_onset_hit.shape[0]} entries, but dff_hit has {n_trials} trials'
        )

    odor_start = int(np.round(pre_trial_length + odor_start_offset_s * rec_hz))
    lick_buffer_frames = int(np.round((lick_buffer_ms / 1000.0) * rec_hz))

    reward_start_offset = int(np.round(reward_window_s[0] * rec_hz))
    reward_stop_offset = int(np.round(reward_window_s[1] * rec_hz))
    swallow_start_offset = int(np.round(swallow_window_s[0] * rec_hz))
    swallow_stop_offset = int(np.round(swallow_window_s[1] * rec_hz))

    odor_to_prelick = np.full((n_neurons, n_trials), np.nan, dtype=float)
    reward_0_2s = np.full((n_neurons, n_trials), np.nan, dtype=float)
    swallow_1p5_2s = np.full((n_neurons, n_trials), np.nan, dtype=float)

    for neuron_idx, neuron_df in enumerate(dff_hit):
        for trial_idx in range(n_trials):
            trace = np.asarray(neuron_df.iloc[trial_idx, :], dtype=float)
            lick_on = lick_onset_hit[trial_idx]
            reward_t = reward_times_hit[trial_idx]

            odor_to_prelick[neuron_idx, trial_idx] = safe_window_mean(
                trace,
                odor_start,
                lick_on - lick_buffer_frames,
            )
            reward_0_2s[neuron_idx, trial_idx] = safe_window_mean(
                trace,
                reward_t + reward_start_offset,
                reward_t + reward_stop_offset,
            )
            swallow_1p5_2s[neuron_idx, trial_idx] = safe_window_mean(
                trace,
                reward_t + swallow_start_offset,
                reward_t + swallow_stop_offset,
            )

    results = {
        'per_trial': {
            'odor_to_prelick': odor_to_prelick,
            'reward_0_2s': reward_0_2s,
            'swallow_1p5_2s': swallow_1p5_2s,
        },
        'per_neuron_mean': {
            'odor_to_prelick': np.nanmean(odor_to_prelick, axis=1),
            'reward_0_2s': np.nanmean(reward_0_2s, axis=1),
            'swallow_1p5_2s': np.nanmean(swallow_1p5_2s, axis=1),
        },
        'metadata': {
            'rec_hz': rec_hz,
            'pre_trial_length': pre_trial_length,
            'odor_start_frame': odor_start,
            'lick_buffer_frames': lick_buffer_frames,
            'reward_window_s': tuple(reward_window_s),
            'swallow_window_s': tuple(swallow_window_s),
            'n_neurons': n_neurons,
            'n_trials': n_trials,
        },
    }

    with open(join(save_loc, f'{prefix}_averages.pkl'), 'wb') as f:
        pickle.dump(results, f)

    np.save(join(save_loc, f'{prefix}_odor_to_prelick_per_trial.npy'), odor_to_prelick)
    np.save(join(save_loc, f'{prefix}_reward_0_2s_per_trial.npy'), reward_0_2s)
    np.save(join(save_loc, f'{prefix}_swallow_1p5_2s_per_trial.npy'), swallow_1p5_2s)

    np.save(join(save_loc, f'{prefix}_odor_to_prelick_per_neuron_mean.npy'), results['per_neuron_mean']['odor_to_prelick'])
    np.save(join(save_loc, f'{prefix}_reward_0_2s_per_neuron_mean.npy'), results['per_neuron_mean']['reward_0_2s'])
    np.save(join(save_loc, f'{prefix}_swallow_1p5_2s_per_neuron_mean.npy'), results['per_neuron_mean']['swallow_1p5_2s'])

    csv_path = join(save_loc, f'{prefix}_per_neuron_means.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'neuron_idx',
            'odor_to_prelick',
            'reward_0_2s',
            'swallow_1p5_2s',
        ])
        for neuron_idx in range(n_neurons):
            writer.writerow([
                neuron_idx,
                results['per_neuron_mean']['odor_to_prelick'][neuron_idx],
                results['per_neuron_mean']['reward_0_2s'][neuron_idx],
                results['per_neuron_mean']['swallow_1p5_2s'][neuron_idx],
            ])

    return results


def main():
    if not DATA_DIR:
        raise ValueError('Fill DATA_DIR in the USER SETTINGS block.')
    if not LICK_FILE:
        raise ValueError('Fill LICK_FILE in the USER SETTINGS block.')

    dff_run_list_path = join(DATA_DIR, 'dff_run_list_no_sub.pkl')
    hit_path = join(DATA_DIR, 'hit.pkl')

    if not os.path.exists(dff_run_list_path):
        raise FileNotFoundError(f'dff_run_list.pkl not found: {dff_run_list_path}')
    if not os.path.exists(hit_path):
        raise FileNotFoundError(f'hit.pkl not found: {hit_path}')
    if not os.path.exists(LICK_FILE):
        raise FileNotFoundError(f'lick file not found: {LICK_FILE}')

    save_loc = SAVE_LOC if SAVE_LOC else join(DATA_DIR, 'reward_related_outputs')

    dff_run_list = load_pickle(dff_run_list_path)
    hit = np.asarray(load_pickle(hit_path), dtype=int)
    lick_onset_offset = load_pickle(LICK_FILE)

    dff_hit = subset_hit_trials(dff_run_list, hit)
    reward_times_all = np.asarray(lick_onset_offset['rewards_times'], dtype=float)
    lick_onset_all = np.asarray(lick_onset_offset['onset'], dtype=float)

    reward_times_hit = reward_times_all[hit]
    lick_onset_hit = lick_onset_all[hit]

    results = extract_reward_related_averages(
        dff_hit=dff_hit,
        reward_times_hit=reward_times_hit,
        lick_onset_hit=lick_onset_hit,
        rec_hz=REC_HZ,
        pre_trial_length=PRE_TRIAL_LENGTH,
        save_loc=save_loc,
        odor_start_offset_s=ODOR_START_OFFSET_S,
        lick_buffer_ms=LICK_BUFFER_MS,
        reward_window_s=REWARD_WINDOW_S,
        swallow_window_s=SWALLOW_WINDOW_S,
        prefix=PREFIX,
    )

    print('Done.')
    print(f'Saved outputs to: {save_loc}')
    print(f"Neurons: {results['metadata']['n_neurons']}, reward trials: {results['metadata']['n_trials']}")


if __name__ == '__main__':
    main()
