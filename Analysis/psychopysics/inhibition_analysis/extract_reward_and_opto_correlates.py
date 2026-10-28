#!/usr/bin/env python3
"""
Standalone script to extract per-neuron averages for:
1) reward trials: odor onset to 100 ms before first lick
2) reward trials: 0-2 s after reward
3) reward trials: 1.5-2 s after reward (swallowing)
4) CR trials: mean activity during the exact detected opto window

The timing follows the events pipeline (creat_events_data_per_neuron_psycho.py):
    opto_on_ind is detected from opto_open_all.pkl as the first rising edge > 1
    opto_on_ind = detected_index + pre_trial_length
    opto_off_ind = opto_on_ind + stim_len

Odor onset is then derived from that timing as:
    odor_on_ind = opto_off_ind

For trials with no detected opto onset, the same fallback as in the events pipeline is used:
    default opto start = pre_trial_length + 1 * rec_hz
    default opto stop  = pre_trial_length + 5 * rec_hz
    default odor onset = pre_trial_length + 5 * rec_hz

Edit the USER SETTINGS block below and run the script (no command-line arguments).
"""

import os
import csv
import pickle
from os.path import join

import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Data locations (set these before running)
# ---------------------------------------------------------------------------
DATA_ROOT = r'path\to\2p_data'  # folder with the imaging sessions (<mouse>/<session>/ML/...)

# ===================== USER SETTINGS =====================
mouse = 'mouse01'
day = 'sess02'
DATA_DIR = DATA_ROOT + r'\{}\{}\ML\data\processed_data\auROC_freq_history'.format(mouse, day)
# Folder containing dff_run_list.pkl and hit.pkl

LICK_FILE = DATA_ROOT + r'\{}\{}\ML\licking_analysis_final\lick_onset_offset_results.pkl'.format(mouse, day)
# Full path to lick_onset_offset_results.pkl

OPTO_FILE = DATA_ROOT + r'\{}\{}\ML\data2\opto_open_all.pkl'.format(mouse, day)
# Full path to opto_open_all.pkl


SAVE_LOC = DATA_ROOT + r"\{}\{}\ML\data\processed_data\inhibition_analysis_v4".format(mouse, day)
# Output folder. Leave empty to save into DATA_DIR

REC_HZ = 15.0
PRE_TRIAL_LENGTH = int(4 * REC_HZ)
STIM_LEN = int(4 * REC_HZ)
LICK_BUFFER_MS = 100.0
REWARD_WINDOW_S = (0.0, 2.0)
SWALLOW_WINDOW_S = (1.0, 2.0)
OPTO_THRESHOLD = 1.0

PREFIX = 'reward_and_opto_related'
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


def subset_trials(dff_run_list, trial_idx):
    dff_run_list = ensure_dataframe_list(dff_run_list)
    trial_idx = np.asarray(trial_idx, dtype=int)

    dff_subset = []
    for neuron_df in dff_run_list:
        dff_subset.append(neuron_df.iloc[trial_idx, :].copy())
    return dff_subset


def align_1d_to_trial_count(arr, n_trials, name='array'):
    arr = np.asarray(arr)
    if arr.shape[0] == n_trials:
        return arr
    if arr.shape[0] > n_trials:
        extra = arr.shape[0] - n_trials
        print(f'{name}: trimming first {extra} entries to match processed trial count ({n_trials}).')
        return arr[-n_trials:]
    raise ValueError(f'{name} has length {arr.shape[0]}, which is smaller than processed trial count {n_trials}.')


def align_trial_list_to_trial_count(trial_list, n_trials, name='trial_list'):
    if len(trial_list) == n_trials:
        return trial_list
    if len(trial_list) > n_trials:
        extra = len(trial_list) - n_trials
        print(f'{name}: trimming first {extra} trials to match processed trial count ({n_trials}).')
        return trial_list[-n_trials:]
    raise ValueError(f'{name} has length {len(trial_list)}, which is smaller than processed trial count {n_trials}.')


def detect_opto_on_frames(opto_trials, threshold=1.0):
    opto_on_ind = []
    for trial in opto_trials:
        arr = np.asarray(trial, dtype=float).ravel()
        changes = np.diff(np.concatenate(([0], (arr > threshold).astype(int))))
        rising = np.where(changes == 1)[0]
        opto_on_ind.append(float(rising[0]) if len(rising) > 0 else np.nan)
    return np.asarray(opto_on_ind, dtype=float)


def build_exact_trial_timing(opto_trials, pre_trial_length, stim_len, rec_hz, threshold=1.0):
    detected_on = detect_opto_on_frames(opto_trials, threshold=threshold)
    detected_on = detected_on + pre_trial_length

    default_on = pre_trial_length + 1 * rec_hz
    opto_on = detected_on.copy()
    opto_on[np.isnan(opto_on)] = default_on

    opto_off = opto_on + stim_len
    odor_on = opto_off.copy()

    return {
        'detected_opto_on_frame': detected_on,
        'opto_on_frame': opto_on,
        'opto_off_frame': opto_off,
        'odor_on_frame': odor_on,
        'default_opto_on_frame': float(default_on),
        'default_opto_off_frame': float(default_on + stim_len),
    }


def extract_reward_and_opto_averages(
    dff_hit,
    reward_times_hit,
    lick_onset_hit,
    odor_on_hit,
    dff_cr,
    cr_opto_on,
    cr_opto_off,
    rec_hz,
    pre_trial_length,
    save_loc,
    lick_buffer_ms=100.0,
    reward_window_s=(0.0, 2.0),
    swallow_window_s=(1.5, 2.0),
    prefix='reward_and_opto_related',
):
    os.makedirs(save_loc, exist_ok=True)

    reward_times_hit = np.asarray(reward_times_hit, dtype=float)
    lick_onset_hit = np.asarray(lick_onset_hit, dtype=float)
    odor_on_hit = np.asarray(odor_on_hit, dtype=float)
    cr_opto_on = np.asarray(cr_opto_on, dtype=float)
    cr_opto_off = np.asarray(cr_opto_off, dtype=float)

    dff_hit = ensure_dataframe_list(dff_hit)
    dff_cr = ensure_dataframe_list(dff_cr)

    n_neurons = len(dff_hit)
    if n_neurons == 0:
        raise ValueError('dff_hit is empty')
    if len(dff_cr) != n_neurons:
        raise ValueError('dff_cr and dff_hit do not have the same number of neurons')

    n_hit_trials = dff_hit[0].shape[0]
    n_cr_trials = dff_cr[0].shape[0]

    if reward_times_hit.shape[0] != n_hit_trials:
        raise ValueError(f'reward_times_hit has {reward_times_hit.shape[0]} entries, but dff_hit has {n_hit_trials} trials')
    if lick_onset_hit.shape[0] != n_hit_trials:
        raise ValueError(f'lick_onset_hit has {lick_onset_hit.shape[0]} entries, but dff_hit has {n_hit_trials} trials')
    if odor_on_hit.shape[0] != n_hit_trials:
        raise ValueError(f'odor_on_hit has {odor_on_hit.shape[0]} entries, but dff_hit has {n_hit_trials} trials')
    if cr_opto_on.shape[0] != n_cr_trials or cr_opto_off.shape[0] != n_cr_trials:
        raise ValueError('CR opto timing arrays do not match the number of CR trials')

    lick_buffer_frames = int(np.round((lick_buffer_ms / 1000.0) * rec_hz))
    reward_start_offset = int(np.round(reward_window_s[0] * rec_hz))
    reward_stop_offset = int(np.round(reward_window_s[1] * rec_hz))
    swallow_start_offset = int(np.round(swallow_window_s[0] * rec_hz))
    swallow_stop_offset = int(np.round(swallow_window_s[1] * rec_hz))

    odor_to_prelick = np.full((n_neurons, n_hit_trials), np.nan, dtype=float)
    reward_0_2s = np.full((n_neurons, n_hit_trials), np.nan, dtype=float)
    swallow_1p5_2s = np.full((n_neurons, n_hit_trials), np.nan, dtype=float)
    cr_opto = np.full((n_neurons, n_cr_trials), np.nan, dtype=float)

    for neuron_idx, neuron_df in enumerate(dff_hit):
        for trial_idx in range(n_hit_trials):
            trace = np.asarray(neuron_df.iloc[trial_idx, :], dtype=float)
            odor_on = odor_on_hit[trial_idx]
            lick_on = lick_onset_hit[trial_idx]
            reward_t = reward_times_hit[trial_idx]

            odor_to_prelick[neuron_idx, trial_idx] = safe_window_mean(
                trace,
                odor_on,
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

    for neuron_idx, neuron_df in enumerate(dff_cr):
        for trial_idx in range(n_cr_trials):
            trace = np.asarray(neuron_df.iloc[trial_idx, :], dtype=float)
            cr_opto[neuron_idx, trial_idx] = safe_window_mean(
                trace,
                cr_opto_on[trial_idx],
                cr_opto_off[trial_idx],
            )

    results = {
        'per_trial': {
            'odor_to_prelick': odor_to_prelick,
            'reward_0_2s': reward_0_2s,
            'swallow_1p5_2s': swallow_1p5_2s,
            'cr_opto': cr_opto,
        },
        'per_neuron_mean': {
            'odor_to_prelick': np.nanmean(odor_to_prelick, axis=1),
            'reward_0_2s': np.nanmean(reward_0_2s, axis=1),
            'swallow_1p5_2s': np.nanmean(swallow_1p5_2s, axis=1),
            'cr_opto': np.nanmean(cr_opto, axis=1),
        },
        'timing': {
            'odor_on_hit': odor_on_hit,
            'reward_times_hit': reward_times_hit,
            'lick_onset_hit': lick_onset_hit,
            'cr_opto_on': cr_opto_on,
            'cr_opto_off': cr_opto_off,
        },
        'metadata': {
            'rec_hz': float(rec_hz),
            'pre_trial_length': int(pre_trial_length),
            'stim_len': int(STIM_LEN),
            'lick_buffer_frames': int(lick_buffer_frames),
            'reward_window_s': tuple(reward_window_s),
            'swallow_window_s': tuple(swallow_window_s),
            'n_neurons': int(n_neurons),
            'n_hit_trials': int(n_hit_trials),
            'n_cr_trials': int(n_cr_trials),
            'timing_rule': 'opto_on from first rising edge > threshold in opto_open_all, opto_off = opto_on + stim_len, odor_on = opto_off',
        },
    }

    with open(join(save_loc, f'{prefix}_averages.pkl'), 'wb') as f:
        pickle.dump(results, f)

    np.save(join(save_loc, f'{prefix}_odor_to_prelick_per_trial.npy'), odor_to_prelick)
    np.save(join(save_loc, f'{prefix}_reward_0_2s_per_trial.npy'), reward_0_2s)
    np.save(join(save_loc, f'{prefix}_swallow_1p5_2s_per_trial.npy'), swallow_1p5_2s)
    np.save(join(save_loc, f'{prefix}_cr_opto_per_trial.npy'), cr_opto)

    np.save(join(save_loc, f'{prefix}_odor_to_prelick_per_neuron_mean.npy'), results['per_neuron_mean']['odor_to_prelick'])
    np.save(join(save_loc, f'{prefix}_reward_0_2s_per_neuron_mean.npy'), results['per_neuron_mean']['reward_0_2s'])
    np.save(join(save_loc, f'{prefix}_swallow_1p5_2s_per_neuron_mean.npy'), results['per_neuron_mean']['swallow_1p5_2s'])
    np.save(join(save_loc, f'{prefix}_cr_opto_per_neuron_mean.npy'), results['per_neuron_mean']['cr_opto'])

    csv_path = join(save_loc, f'{prefix}_per_neuron_means.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'neuron_idx',
            'cr_opto',
            'odor_to_prelick',
            'reward_0_2s',
            'swallow_1p5_2s',
        ])
        for neuron_idx in range(n_neurons):
            writer.writerow([
                neuron_idx,
                results['per_neuron_mean']['cr_opto'][neuron_idx],
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
    if not OPTO_FILE:
        raise ValueError('Fill OPTO_FILE in the USER SETTINGS block.')

    dff_run_list_path = join(DATA_DIR, 'dff_run_list.pkl')
    hit_path = join(DATA_DIR, 'hit.pkl')
    cr_path = join(DATA_DIR, 'correct_reject.pkl')

    for p, label in [
        (dff_run_list_path, 'dff_run_list.pkl'),
        (hit_path, 'hit.pkl'),
        (cr_path, 'correct_reject.pkl'),
        (LICK_FILE, 'lick file'),
        (OPTO_FILE, 'opto file'),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'{label} not found: {p}')

    save_loc = SAVE_LOC if SAVE_LOC else DATA_DIR

    dff_run_list = load_pickle(dff_run_list_path)
    dff_run_list = ensure_dataframe_list(dff_run_list)
    n_trials = dff_run_list[0].shape[0]

    hit = np.asarray(load_pickle(hit_path), dtype=int)
    correct_reject = np.asarray(load_pickle(cr_path), dtype=int)
    lick_onset_offset = load_pickle(LICK_FILE)
    opto_trials = load_pickle(OPTO_FILE)

    reward_times_all = align_1d_to_trial_count(lick_onset_offset['rewards_times'], n_trials, name='rewards_times')
    lick_onset_all = align_1d_to_trial_count(lick_onset_offset['onset'], n_trials, name='lick_onset')
    opto_trials = align_trial_list_to_trial_count(opto_trials, n_trials, name='opto_open_all')

    timing = build_exact_trial_timing(
        opto_trials=opto_trials,
        pre_trial_length=PRE_TRIAL_LENGTH,
        stim_len=STIM_LEN,
        rec_hz=REC_HZ,
        threshold=OPTO_THRESHOLD,
    )

    dff_hit = subset_trials(dff_run_list, hit)
    dff_cr = subset_trials(dff_run_list, correct_reject)

    reward_times_hit = reward_times_all[hit]
    lick_onset_hit = lick_onset_all[hit]
    odor_on_hit = timing['odor_on_frame'][hit]
    cr_opto_on = timing['opto_on_frame'][correct_reject]
    cr_opto_off = timing['opto_off_frame'][correct_reject]

    results = extract_reward_and_opto_averages(
        dff_hit=dff_hit,
        reward_times_hit=reward_times_hit,
        lick_onset_hit=lick_onset_hit,
        odor_on_hit=odor_on_hit,
        dff_cr=dff_cr,
        cr_opto_on=cr_opto_on,
        cr_opto_off=cr_opto_off,
        rec_hz=REC_HZ,
        pre_trial_length=PRE_TRIAL_LENGTH,
        save_loc=save_loc,
        lick_buffer_ms=LICK_BUFFER_MS,
        reward_window_s=REWARD_WINDOW_S,
        swallow_window_s=SWALLOW_WINDOW_S,
        prefix=PREFIX,
    )

    with open(join(save_loc, f'{PREFIX}_timing_debug.pkl'), 'wb') as f:
        pickle.dump(timing, f)

    print('Done.')
    print(f'Saved outputs to: {save_loc}')
    print(f"Neurons: {results['metadata']['n_neurons']}, reward trials: {results['metadata']['n_hit_trials']}, CR trials: {results['metadata']['n_cr_trials']}")
    print('Timing used: opto_on from opto_open_all rising edge, opto_off = opto_on + stim_len, odor_on = opto_off.')


if __name__ == '__main__':
    main()
