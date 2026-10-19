
import matplotlib
import copy
import os
import pickle
import random
import h5py
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
from scipy.stats import ttest_rel, wilcoxon
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# Optional backend for script mode
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42

# =========================
# Constants from your code
# =========================
start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
rec_hz = 15
sync_hz = 6000

con1 = [5]   # light trials
con2 = 6

pre_trial_length = 4 * rec_hz
baseline_len = 4 * rec_hz
stim_len = 4 * rec_hz


# # =========================
# # Utility functions
# # =========================
def divide_to_trials(f_signal, trials_ind, lengths, len_pre_trial):
    """
    Divide continuous signal into trials and prepend pre-trial frames.
    f_signal: DataFrame, shape (n_neurons, total_frames)
    Returns:
        f_run_list_final: list of neurons, each entry is a list of trial arrays
        pre_run_list: last pre-run of each run
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

        for run in range(0, len(f_run_list)):
            for j in range(0, len(f_run_list[run])):
                if j == 0:
                    pre_trial = pre_run_list[run][len(pre_run_list[run]) - len_pre_trial:]
                else:
                    pre_trial = f_run_list[run][j - 1][len(f_run_list[run][j - 1]) - len_pre_trial:]

                f_run_list[run][j] = list(pre_trial) + list(f_run_list[run][j])

            f_run_list_all.extend(f_run_list[run])

        f_run_list_final.append(f_run_list_all)

    return f_run_list_final, pre_run_list


def divide_to_conditions(dff_sub, correct_reject, false_alarm, hit, miss, light_general):
    data_correct_reject = []
    data_false_alarm = []
    data_hit = []
    data_miss = []
    data_light_general = []

    for neuron in dff_sub:
        neuron = pd.DataFrame(neuron)

        data_correct_reject.append(neuron.iloc[correct_reject, :])
        data_false_alarm.append(neuron.iloc[false_alarm, :])
        data_hit.append(neuron.iloc[hit, :])
        data_miss.append(neuron.iloc[miss, :])
        data_light_general.append(neuron.iloc[light_general, :])

    return data_correct_reject, data_false_alarm, data_hit, data_miss, data_light_general


def paired_test(series1, series2, method='ttest'):
    mean_s1 = series1.mean()
    mean_s2 = series2.mean()

    if method == 'wilcoxon':
        stat, p_val = wilcoxon(series1, series2, alternative='two-sided')
    else:
        stat, p_val = ttest_rel(series1, series2, nan_policy='omit')

    direction = 'increase' if (mean_s1 > mean_s2) else 'decrease'
    return p_val, direction

#
# def identify_light_on_two_windows(
#         dff_list,
#         light_on_ind,
#         light_off_ind,
#         baseline_start,
#         baseline_stop,
#         rec_hz,
#         alpha=0.05,
#         test_method='ttest',
#         do_bonferroni=True
# ):
#     """
#     Identify neurons with significant light response.
#     Uses two windows:
#       1. first 2s after light onset
#       2. last 2s before light offset
#     """
#     n_neurons = len(dff_list)
#     n_trials = dff_list[0].shape[0]
#
#     light_on_ind = np.asarray(light_on_ind, dtype=float)
#     light_off_ind = np.asarray(light_off_ind, dtype=float)
#
#     start_opto = pre_trial_length + 1 * rec_hz
#     end_opto = pre_trial_length + 5 * rec_hz
#     light_on_ind[np.isnan(light_on_ind)] = start_opto
#     light_off_ind[np.isnan(light_off_ind)] = end_opto
#
#     alpha_corr = alpha / 2.0 if do_bonferroni else alpha
#
#     increase_mask = np.zeros(n_neurons, dtype=bool)
#     decrease_mask = np.zeros(n_neurons, dtype=bool)
#
#     for neuron_idx in range(n_neurons):
#         neuron_df = dff_list[neuron_idx]
#
#         wA_vals = []
#         wB_vals = []
#         base_vals = []
#
#         for trial_i in range(n_trials):
#             on_i = light_on_ind[trial_i]
#             off_i = light_off_ind[trial_i]
#
#             if np.isnan(on_i) or np.isnan(off_i):
#                 wA_vals.append(np.nan)
#                 wB_vals.append(np.nan)
#                 base_vals.append(np.nan)
#                 continue
#
#             on_i = int(on_i)
#             off_i = int(off_i)
#
#             if off_i <= on_i:
#                 wA_vals.append(np.nan)
#                 wB_vals.append(np.nan)
#                 base_vals.append(np.nan)
#                 continue
#
#             bs = int(baseline_start)
#             bp = int(baseline_stop)
#             base_mean = neuron_df.iloc[trial_i, bs:bp].mean()
#
#             A_start = on_i
#             A_stop = min(on_i + 2 * rec_hz, neuron_df.shape[1])
#             wA_mean = neuron_df.iloc[trial_i, A_start:A_stop].mean() if A_stop > A_start else np.nan
#
#             B_stop = off_i
#             B_start = max(0, off_i - 2 * rec_hz)
#             wB_mean = neuron_df.iloc[trial_i, B_start:B_stop].mean() if B_stop > B_start else np.nan
#
#             wA_vals.append(wA_mean)
#             wB_vals.append(wB_mean)
#             base_vals.append(base_mean)
#
#         wA_series = pd.Series(wA_vals).dropna()
#         wB_series = pd.Series(wB_vals).dropna()
#         base_series = pd.Series(base_vals).dropna()
#
#         length = min(len(wA_series), len(wB_series), len(base_series))
#         if length < 5:
#             continue
#
#         wA_series = wA_series.iloc[:length]
#         wB_series = wB_series.iloc[:length]
#         base_series = base_series.iloc[:length]
#
#         p_valA, directionA = paired_test(wA_series, base_series, test_method)
#         p_valB, directionB = paired_test(wB_series, base_series, test_method)
#
#         sigA = (p_valA < alpha_corr)
#         sigB = (p_valB < alpha_corr)
#
#         if sigA or sigB:
#             if directionA == 'increase' or directionB == 'increase':
#                 increase_mask[neuron_idx] = True
#             if directionA == 'decrease' or directionB == 'decrease':
#                 decrease_mask[neuron_idx] = True
#
#     results = {
#         'light_on': {
#             'increase': np.where(increase_mask)[0],
#             'decrease': np.where(decrease_mask)[0]
#         }
#     }
#     return results
#
#
# # =========================
# # Decay analysis functions
# # =========================
# def exp_decay_with_offset(t, A, tau, C):
#     """
#     Exponential decay with offset:
#         y(t) = C + A * exp(-t / tau)
#
#     C is the plateau level the fluorescence decays toward.
#     tau is the fluorescence decay time constant.
#     """
#     return C + A * np.exp(-t / tau)
#
def find_first_significant_peak(trace_sm,
                                on_frame,
                                off_frame,
                                baseline_frames,
                                min_peak_height=0.25,
                                min_peak_prominence=0.2,
                                min_peak_distance=2,
                                baseline_z_thresh=2.0):
    """
    Find the maximal significant local peak inside the opto window.

    Returns
    -------
    peak_idx : int or None
        Index of the maximal significant peak in the full trace.
    """

    n_frames = len(trace_sm)
    on_frame = int(np.clip(on_frame, 0, n_frames - 1))
    off_frame = int(np.clip(off_frame, on_frame + 1, n_frames))

    # baseline from pre-opto period
    bs_start = max(0, on_frame - baseline_frames)
    bs_stop = max(bs_start + 1, on_frame)
    baseline_seg = trace_sm[bs_start:bs_stop]

    baseline_mean = safe_nanmean(baseline_seg)
    baseline_std = np.nanstd(baseline_seg)
    if not np.isfinite(baseline_std) or baseline_std < 1e-8:
        baseline_std = 1e-8

    # search only during opto
    peak_window = trace_sm[on_frame:off_frame]
    if len(peak_window) < 3 or np.all(np.isnan(peak_window)):
        return None

    # replace NaNs for peak finding
    peak_window_clean = peak_window.copy()
    peak_window_clean[np.isnan(peak_window_clean)] = -np.inf

    # threshold above baseline
    abs_height_thresh = max(
        baseline_mean + baseline_z_thresh * baseline_std,
        min_peak_height
    )

    peaks, props = find_peaks(
        peak_window_clean,
        height=abs_height_thresh,
        prominence=min_peak_prominence,
        distance=min_peak_distance
    )

    if len(peaks) == 0:
        return None

    # choose the significant peak with the largest height
    peak_heights = props['peak_heights']
    best_idx = int(np.argmax(peak_heights))
    peak_rel_idx = int(peaks[best_idx])

    peak_idx = on_frame + peak_rel_idx
    return peak_idx
def smooth_trace_nan(x, window=3):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)

def exp_decay_double_anchored(t, tau_fast, tau_slow, w_fast, C, peak_value):
    """
    Double exponential anchored to the measured peak:
    y(0) = peak_value exactly
    """
    return C + (peak_value - C) * (
        w_fast * np.exp(-t / tau_fast) +
        (1.0 - w_fast) * np.exp(-t / tau_slow)
    )
def fit_decay_one_trial(trial_trace,
                        on_frame,
                        off_frame,
                        rec_hz,
                        baseline_frames=None,
                        smooth_window=3):
    """
    Peak is searched anywhere inside the opto window.
    Decay fit starts at that peak and continues until the end of the valid trial.
    Uses a DOUBLE exponential fit only.
    """
    trial_trace = np.asarray(trial_trace, dtype=float)

    if baseline_frames is None:
        baseline_frames = rec_hz

    valid_mask = np.isfinite(trial_trace)
    if not np.any(valid_mask):
        return None

    n_frames = np.where(valid_mask)[0][-1] + 1
    trial_trace = trial_trace[:n_frames]

    on_frame = int(np.clip(on_frame, 0, n_frames - 1))
    off_frame = int(np.clip(off_frame, on_frame + 1, n_frames))

    # Baseline subtraction
    bs_start = max(0, on_frame - baseline_frames)
    bs_stop = max(bs_start + 1, on_frame)
    baseline = safe_nanmean(trial_trace[bs_start:bs_stop])
    trace_bs = trial_trace - baseline

    # Smooth
    trace_sm = smooth_trace_nan(trace_bs, window=smooth_window)

    # Peak = first significant local peak during opto, fallback to global max during opto
    peak_idx = find_first_significant_peak(
        trace_sm=trace_sm,
        on_frame=on_frame,
        off_frame=off_frame,
        baseline_frames=baseline_frames,
        min_peak_height=0.25,
        min_peak_prominence=0.2,
        min_peak_distance=2,
        baseline_z_thresh=2.0
    )

    if peak_idx is None:
        peak_window = trace_sm[on_frame:off_frame]
        if len(peak_window) == 0 or np.all(np.isnan(peak_window)):
            return None
        peak_rel_idx = int(np.nanargmax(peak_window))
        peak_idx = on_frame + peak_rel_idx

    peak_value = trace_sm[peak_idx]

    # Fit until end of valid trial
    fit_end_frame = n_frames
    fit_end_frame = int(np.clip(fit_end_frame, peak_idx + 1, n_frames))

    decay_segment = trace_sm[peak_idx:fit_end_frame]
    valid = np.isfinite(decay_segment)

    if np.sum(valid) < 5:
        return None

    y = decay_segment[valid]
    t = np.arange(len(decay_segment), dtype=float)[valid] / rec_hz

    tail_n = max(3, int(round(0.5 * rec_hz)))
    C0 = safe_nanmean(y[-tail_n:])
    if np.isnan(C0):
        C0 = 0.0

    # Initial guesses
    tau_fast0 = 0.3
    tau_slow0 = 2.0
    w_fast0 = 0.5

    c_upper = peak_value - 1e-6

    try:
        popt, _ = curve_fit(
            lambda t_, tau_fast_, tau_slow_, w_fast_, C_: exp_decay_double_anchored(
                t_, tau_fast_, tau_slow_, w_fast_, C_, peak_value
            ),
            t,
            y,
            p0=[tau_fast0, tau_slow0, w_fast0, C0],
            bounds=([0.03, 0.05, 0.0, -np.inf],
                    [3.0, 30.0, 1.0, c_upper]),
            maxfev=50000
        )

        tau_fast_fit, tau_slow_fit, w_fast_fit, C_fit = popt

        # enforce ordering so tau_fast <= tau_slow
        if tau_fast_fit > tau_slow_fit:
            tau_fast_fit, tau_slow_fit = tau_slow_fit, tau_fast_fit
            w_fast_fit = 1.0 - w_fast_fit

        y_fit = exp_decay_double_anchored(
            t, tau_fast_fit, tau_slow_fit, w_fast_fit, C_fit, peak_value
        )

        ss_res = np.nansum((y - y_fit) ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        fit_r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    except Exception:
        return None

    # 2/3 drop from the observed peak, not from the fit
    threshold = peak_value / 3.0
    below_mask = (decay_segment <= threshold) & np.isfinite(decay_segment)

    crossing_idx = np.nan
    for i in range(len(below_mask) - 1):
        if below_mask[i] and below_mask[i + 1]:
            crossing_idx = i
            break

    t_2third_drop_s = np.nan if np.isnan(crossing_idx) else crossing_idx / rec_hz

    end_win = max(1, int(rec_hz))

    # End of opto
    opto_slice = trace_sm[max(on_frame, off_frame - end_win):off_frame]
    opto_end_mean = safe_nanmean(opto_slice)
    opto_end_to_peak_ratio = np.nan if peak_value == 0 else opto_end_mean / peak_value

    # First 0.5 sec after opto ends = odor start window
    odor_win = max(1, int(round(0.5 * rec_hz)))
    odor_start_slice = trace_sm[off_frame:min(n_frames, off_frame + odor_win)]
    odor_start_mean = safe_nanmean(odor_start_slice)
    odor_start_to_peak_ratio = np.nan if peak_value == 0 else odor_start_mean / peak_value
    odor_start_to_peak_percent = np.nan if np.isnan(odor_start_to_peak_ratio) else odor_start_to_peak_ratio * 100.0
    odor_start_decrease_percent = np.nan if np.isnan(odor_start_to_peak_percent) else 100.0 - odor_start_to_peak_percent

    # End of full fit window
    fit_slice = trace_sm[max(peak_idx, fit_end_frame - end_win):fit_end_frame]
    fit_end_mean = safe_nanmean(fit_slice)
    fit_end_to_peak_ratio = np.nan if peak_value == 0 else fit_end_mean / peak_value

    return {
        'baseline': baseline,
        'peak_idx': peak_idx,
        'peak_time_from_on_s': (peak_idx - on_frame) / rec_hz,
        'peak_value': peak_value,

        # main tau = tau_fast
        'tau_s': tau_fast_fit,
        'tau_fast_s': tau_fast_fit,
        'tau_slow_s': tau_slow_fit,
        'w_fast': w_fast_fit,

        'fit_A': peak_value - C_fit,
        'fit_C': C_fit,
        'fit_r2': fit_r2,
        't_2third_drop_s': t_2third_drop_s,

        'opto_end_mean': opto_end_mean,
        'opto_end_to_peak_ratio': opto_end_to_peak_ratio,

        'odor_start_mean': odor_start_mean,
        'odor_start_to_peak_ratio': odor_start_to_peak_ratio,
        'odor_start_to_peak_percent': odor_start_to_peak_percent,
        'odor_start_decrease_percent': odor_start_decrease_percent,

        'fit_end_mean': fit_end_mean,
        'fit_end_to_peak_ratio': fit_end_to_peak_ratio,

        'trace_bs': trace_bs,
        'trace_sm': trace_sm,
        'fit_end_frame': fit_end_frame,
        'y_fit': y_fit,
    }
def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def max_consecutive_true(mask):
    mask = np.asarray(mask, dtype=bool)
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


# def trial_has_significant_opto_response(trial_trace,
#                                         on_frame,
#                                         off_frame,
#                                         rec_hz,
#                                         baseline_frames=None,
#                                         smooth_window=3,
#                                         z_thresh=2.0,
#                                         min_consecutive_frames=2):
#     """
#     Operational per-trial significance rule:
#     a trial is called opto-responsive if, during the opto window,
#     the smoothed signal crosses baseline + z_thresh*baseline_std
#     for at least min_consecutive_frames.
#
#     This is not a formal p-value, it is a per-trial threshold rule.
#     """
#     trial_trace = np.asarray(trial_trace, dtype=float)
#
#     if baseline_frames is None:
#         baseline_frames = rec_hz
#
#     valid_mask = ~np.isnan(trial_trace)
#     if not np.any(valid_mask):
#         return None
#
#     n_frames = np.where(valid_mask)[0][-1] + 1
#     on_frame = int(np.clip(on_frame, 0, n_frames - 1))
#     off_frame = int(np.clip(off_frame, on_frame + 1, n_frames))
#
#     trace_sm = smooth_trace_nan(trial_trace, window=smooth_window)
#
#     bs_start = max(0, on_frame - baseline_frames)
#     bs_stop = max(bs_start + 1, on_frame)
#     baseline_seg = trace_sm[bs_start:bs_stop]
#
#     if baseline_seg.size < 3 or np.all(np.isnan(baseline_seg)):
#         return {
#             'is_significant': False,
#             'baseline_mean': np.nan,
#             'baseline_std': np.nan,
#             'threshold': np.nan,
#             'peak_z': np.nan,
#             'max_run': 0
#         }
#
#     baseline_mean = safe_nanmean(baseline_seg)
#     baseline_std = np.nanstd(baseline_seg)
#
#     if not np.isfinite(baseline_std) or baseline_std < 1e-8:
#         baseline_std = 1e-8
#
#     opto_seg = trace_sm[on_frame:off_frame]
#     if opto_seg.size == 0 or np.all(np.isnan(opto_seg)):
#         return {
#             'is_significant': False,
#             'baseline_mean': baseline_mean,
#             'baseline_std': baseline_std,
#             'threshold': np.nan,
#             'peak_z': np.nan,
#             'max_run': 0
#         }
#
#     threshold = baseline_mean + z_thresh * baseline_std
#     supra = opto_seg > threshold
#     max_run = max_consecutive_true(supra)
#
#     peak_val = np.nanmax(opto_seg)
#     peak_z = (peak_val - baseline_mean) / baseline_std
#
#     is_significant = (peak_z >= z_thresh) and (max_run >= min_consecutive_frames)
#
#     return {
#         'is_significant': bool(is_significant),
#         'baseline_mean': baseline_mean,
#         'baseline_std': baseline_std,
#         'threshold': threshold,
#         'peak_z': peak_z,
#         'max_run': max_run
#     }
# def summarize_trialwise_decay(per_trial_df,
#                               min_valid_trials_per_neuron=5,
#                               ratio_col='fit_end_to_peak_ratio'):
#     if len(per_trial_df) == 0:
#         return pd.DataFrame()
#
#     summary_rows = []
#
#     for neuron_idx, subdf in per_trial_df.groupby('neuron_idx'):
#         subdf = subdf.copy()
#         subdf = subdf[np.isfinite(subdf['tau_s'])]
#
#         if len(subdf) < min_valid_trials_per_neuron:
#             continue
#
#         row = {
#             'neuron_idx': neuron_idx,
#             'n_valid_trials': len(subdf),
#
#             # primary
#             'median_tau_s': np.nanmedian(subdf['tau_s']),
#             'mean_tau_s': np.nanmean(subdf['tau_s']),
#             'tau_iqr_s': np.nanpercentile(subdf['tau_s'], 75) - np.nanpercentile(subdf['tau_s'], 25),
#
#             'median_peak_value': np.nanmedian(subdf['peak_value']),
#             'median_peak_time_from_on_s': np.nanmedian(subdf['peak_time_from_on_s']),
#             'median_fit_r2': np.nanmedian(subdf['fit_r2']),
#             'median_t_2third_drop_s': np.nanmedian(subdf['t_2third_drop_s']),
#             'median_fit_end_to_peak_ratio': np.nanmedian(subdf[ratio_col]),
#         }
#
#         if 'tau_fast_s' in subdf.columns:
#             row['median_tau_fast_s'] = np.nanmedian(subdf['tau_fast_s'])
#             row['mean_tau_fast_s'] = np.nanmean(subdf['tau_fast_s'])
#
#         if 'tau_slow_s' in subdf.columns:
#             row['median_tau_slow_s'] = np.nanmedian(subdf['tau_slow_s'])
#             row['mean_tau_slow_s'] = np.nanmean(subdf['tau_slow_s'])
#
#         if 'w_fast' in subdf.columns:
#             row['median_w_fast'] = np.nanmedian(subdf['w_fast'])
#             row['mean_w_fast'] = np.nanmean(subdf['w_fast'])
#
#         summary_rows.append(row)
#
#     summary_df = pd.DataFrame(summary_rows)
#     if len(summary_df) > 0:
#         summary_df = summary_df.sort_values('median_tau_s').reset_index(drop=True)
#
#     return summary_df
# def safe_nanmean(x):
#     x = np.asarray(x, dtype=float)
#     if x.size == 0 or np.all(np.isnan(x)):
#         return np.nan
#     return np.nanmean(x)
# def plot_trialwise_decay_summary(per_trial_df, per_neuron_df, save_dir, prefix):
#     if len(per_trial_df) == 0 or len(per_neuron_df) == 0:
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 10))
#
#     axes[0].hist(per_trial_df['tau_s'].dropna(), bins=30, edgecolor='black')
#     axes[0].set_title('All valid trial taus')
#     axes[0].set_xlabel('Tau (s)')
#     axes[0].set_ylabel('Count')
#
#     axes[1].hist(per_neuron_df['median_tau_s'].dropna(), bins=25, edgecolor='black')
#     axes[1].set_title('Median tau per neuron')
#     axes[1].set_xlabel('Median tau (s)')
#     axes[1].set_ylabel('Neuron count')
#
#     # axes[1, 0].scatter(per_neuron_df['median_tau_s'],
#     #                    per_neuron_df['median_fit_end_to_peak_ratio'],
#     #                    alpha=0.7)
#     # axes[1, 0].set_title('Median tau vs fit-end/peak ratio')
#     # axes[1, 0].set_xlabel('Median tau (s)')
#     # axes[1, 0].set_ylabel('Median fit-end/peak ratio')
#     #
#     # axes[1, 1].scatter(per_neuron_df['n_valid_trials'],
#     #                    per_neuron_df['median_tau_s'],
#     #                    alpha=0.7)
#     # axes[1, 1].set_title('Valid trials vs median tau')
#     # axes[1, 1].set_xlabel('Number of valid trials')
#     # axes[1, 1].set_ylabel('Median tau (s)')
#
#     plt.tight_layout()
#     plt.savefig(join(save_dir, f'{prefix}_summary.png'))
#     plt.savefig(join(save_dir, f'{prefix}_summary.pdf'))
#     plt.close(fig)
#
#
#
# def exp_decay_anchored(t, tau, C, peak_value):
#     """
#     Exponential decay anchored to the measured peak.
#     At t=0, y(0) = peak_value exactly.
#     """
#     return C + (peak_value - C) * np.exp(-t / tau)
# def plot_example_fits(dff_light_all,
#                       opto_on_inds_light,
#                       opto_off_inds_light,
#                       per_trial_df,
#                       per_neuron_df,
#                       save_dir,
#                       prefix,
#                       baseline_frames=15,
#                       smooth_window=3,
#                       n_neurons_to_plot=8,
#                       n_trials_per_neuron=4):
#     if len(per_neuron_df) == 0 or len(per_trial_df) == 0:
#         return
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     plot_df = per_neuron_df.sort_values('median_peak_value', ascending=False).head(n_neurons_to_plot)
#     selected_neurons = plot_df['neuron_idx'].values
#
#     for neuron_idx in selected_neurons:
#         neuron_df = dff_light_all[int(neuron_idx)]
#
#         valid_trials_df = per_trial_df[per_trial_df['neuron_idx'] == neuron_idx].copy()
#         if len(valid_trials_df) == 0:
#             continue
#
#         valid_trials_df = valid_trials_df.sort_values('peak_value', ascending=False)
#         trials_to_show = valid_trials_df['trial_idx_local'].values[:n_trials_per_neuron]
#
#         fig, axes = plt.subplots(len(trials_to_show), 1, figsize=(10, 3 * len(trials_to_show)), sharex=True)
#         if len(trials_to_show) == 1:
#             axes = [axes]
#
#         for ax, trial_idx in zip(axes, trials_to_show):
#             trial_trace = neuron_df.iloc[int(trial_idx), :].to_numpy(dtype=float)
#             on_frame = int(opto_on_inds_light[int(trial_idx)])
#             off_frame = int(opto_off_inds_light[int(trial_idx)])
#
#             fit_res = fit_decay_one_trial(
#                 trial_trace=trial_trace,
#                 on_frame=on_frame,
#                 off_frame=off_frame,
#                 rec_hz=rec_hz,
#                 baseline_frames=baseline_frames,
#                 smooth_window=smooth_window
#             )
#
#             # use only the real part of the trial for plotting
#             trial_trace = np.asarray(trial_trace, dtype=float)
#
#             valid_mask = np.isfinite(trial_trace)
#             if not np.any(valid_mask):
#                 continue
#
#             last_valid_idx = np.where(valid_mask)[0][-1]
#             len_valid = last_valid_idx + 1
#
#             trial_trace_plot = trial_trace[:len_valid]
#             t_axis = (np.arange(len_valid) - on_frame) / rec_hz
#
#             # Opto window shading
#             ax.axvspan(0, (off_frame - on_frame) / rec_hz, color='gold', alpha=0.15)
#
#             if fit_res is None:
#                 ax.plot(t_axis, trial_trace, color='black', alpha=0.7)
#                 ax.axvline(0, linestyle='--', color='gold')
#                 ax.axvline((off_frame - on_frame) / rec_hz, linestyle='--', color='orange')
#                 ax.set_title(f'Neuron {neuron_idx}, trial {trial_idx}, fit failed')
#                 ax.set_xlim(t_axis[0], t_axis[-1])
#                 continue
#
#             ax.plot(t_axis, fit_res['trace_bs'][:len_valid], color='gray', alpha=0.5, label='Baseline-subtracted')
#             ax.plot(t_axis, fit_res['trace_sm'][:len_valid], color='black', linewidth=2, label='Smoothed')
#             ax.set_xlim(t_axis[0], t_axis[-1])
#
#             ax.axvline(0, linestyle='--', color='gold', label='Opto on')
#             ax.axvline((off_frame - on_frame) / rec_hz, linestyle='--', color='orange', label='Opto off')
#             ax.axvline((fit_res['peak_idx'] - on_frame) / rec_hz, linestyle='--', color='red', label='Peak')
#
#             peak_idx = fit_res['peak_idx']
#             fit_end_frame = min(fit_res['fit_end_frame'], len_valid)
#             fit_x = (np.arange(peak_idx, fit_end_frame) - on_frame) / rec_hz
#             ax.plot(fit_x, fit_res['y_fit'][:len(fit_x)], color='dodgerblue', linewidth=2, label='Double-exp fit')
#
#             ax.set_title(
#                 f'Neuron {neuron_idx}, trial {trial_idx} | '
#                 f'tau={fit_res["tau_s"]:.3f}s | '
#                 f'R2={fit_res["fit_r2"]:.2f} | '
#                 f'2/3 drop={fit_res["t_2third_drop_s"]:.3f}s'
#             )
#             ax.set_xlim(t_axis[0], t_axis[-1])
#             ax.legend(loc='upper right')
#
#         axes[-1].set_xlabel('Time from opto onset (s)')
#         plt.tight_layout()
#         plt.savefig(join(save_dir, f'{prefix}_neuron_{int(neuron_idx)}_examples.png'))
#         plt.savefig(join(save_dir, f'{prefix}_neuron_{int(neuron_idx)}_examples.pdf'))
#         plt.close(fig)
#
# def analyze_opto_decay_trial_by_trial(dff_light_all,
#                                       opto_on_inds_light,
#                                       opto_off_inds_light,
#                                       rec_hz,
#                                       save_dir,
#                                       neuron_indices=None,
#                                       baseline_frames=None,
#                                       smooth_window=3,
#                                       min_peak=0.25,
#                                       min_fit_r2=0.2,
#                                       min_valid_trials_per_neuron=5,
#                                       prefix='all_neurons',
#                                       require_significant_trials=True,
#                                       trial_sig_z=2.0,
#                                       trial_sig_min_consecutive=2):
#     os.makedirs(save_dir, exist_ok=True)
#
#     if baseline_frames is None:
#         baseline_frames = rec_hz
#
#     if neuron_indices is None:
#         neuron_indices = np.arange(len(dff_light_all))
#
#     neuron_indices = np.asarray(neuron_indices, dtype=int)
#     opto_on_inds_light = np.asarray(opto_on_inds_light)
#     opto_off_inds_light = np.asarray(opto_off_inds_light)
#
#     per_trial_rows = []
#
#     for neuron_idx in neuron_indices:
#         neuron_df = dff_light_all[int(neuron_idx)]
#
#         for trial_local_idx in range(neuron_df.shape[0]):
#             trial_trace = neuron_df.iloc[trial_local_idx, :].to_numpy(dtype=float)
#             on_frame = int(opto_on_inds_light[trial_local_idx])
#             off_frame = int(opto_off_inds_light[trial_local_idx])
#
#             sig_info = trial_has_significant_opto_response(
#                 trial_trace=trial_trace,
#                 on_frame=on_frame,
#                 off_frame=off_frame,
#                 rec_hz=rec_hz,
#                 baseline_frames=baseline_frames,
#                 smooth_window=smooth_window,
#                 z_thresh=trial_sig_z,
#                 min_consecutive_frames=trial_sig_min_consecutive
#             )
#
#             if require_significant_trials and (not sig_info['is_significant']):
#                 continue
#
#             fit_res = fit_decay_one_trial(
#                 trial_trace=trial_trace,
#                 on_frame=on_frame,
#                 off_frame=off_frame,
#                 rec_hz=rec_hz,
#                 baseline_frames=baseline_frames,
#                 smooth_window=smooth_window
#             )
#
#             if fit_res is None:
#                 continue
#
#             if not np.isfinite(fit_res['peak_value']) or fit_res['peak_value'] < min_peak:
#                 continue
#
#             if min_fit_r2 is not None:
#                 if not np.isfinite(fit_res['fit_r2']) or fit_res['fit_r2'] < min_fit_r2:
#                     continue
#
#             per_trial_rows.append({
#                 'neuron_idx': int(neuron_idx),
#                 'trial_idx_local': int(trial_local_idx),
#                 'peak_time_from_on_s': fit_res['peak_time_from_on_s'],
#                 'peak_value': fit_res['peak_value'],
#                 'tau_s': fit_res['tau_s'],
#                 'tau_fast_s': fit_res['tau_fast_s'],
#                 'tau_slow_s': fit_res['tau_slow_s'],
#                 'w_fast': fit_res['w_fast'],
#                 'fit_r2': fit_res['fit_r2'],
#                 't_2third_drop_s': fit_res['t_2third_drop_s'],
#                 'opto_end_mean': fit_res['opto_end_mean'],
#                 'opto_end_to_peak_ratio': fit_res['opto_end_to_peak_ratio'],
#                 'fit_end_mean': fit_res['fit_end_mean'],
#                 'fit_end_to_peak_ratio': fit_res['fit_end_to_peak_ratio'],
#                 'fit_A': fit_res['fit_A'],
#                 'fit_C': fit_res['fit_C'],
#                 'fit_end_frame': fit_res['fit_end_frame'],
#                 'trial_peak_z': sig_info['peak_z'],
#                 'trial_sig_max_run': sig_info['max_run'],
#                 'trial_sig_threshold': sig_info['threshold'],
#             })
#
#     per_trial_df = pd.DataFrame(per_trial_rows)
#
#     per_neuron_df = summarize_trialwise_decay(
#         per_trial_df,
#         min_valid_trials_per_neuron=min_valid_trials_per_neuron,
#         ratio_col='fit_end_to_peak_ratio'
#     )
#
#     per_trial_df.to_csv(join(save_dir, f'{prefix}_per_trial.csv'), index=False)
#     per_neuron_df.to_csv(join(save_dir, f'{prefix}_per_neuron.csv'), index=False)
#
#     with open(join(save_dir, f'{prefix}_per_trial.pkl'), 'wb') as f:
#         pickle.dump(per_trial_df, f)
#
#     with open(join(save_dir, f'{prefix}_per_neuron.pkl'), 'wb') as f:
#         pickle.dump(per_neuron_df, f)
#
#     plot_trialwise_decay_summary(per_trial_df, per_neuron_df, save_dir, prefix)
#
#     plot_example_fits(
#         dff_light_all=dff_light_all,
#         opto_on_inds_light=opto_on_inds_light,
#         opto_off_inds_light=opto_off_inds_light,
#         per_trial_df=per_trial_df,
#         per_neuron_df=per_neuron_df,
#         save_dir=save_dir,
#         prefix=prefix,
#         baseline_frames=baseline_frames,
#         smooth_window=smooth_window,
#         n_neurons_to_plot=8,
#         n_trials_per_neuron=4
#     )
#
#     return per_trial_df, per_neuron_df
# from pathlib import Path
#
#
# def read_suite2p(data_path):
#     data_path = Path(data_path)
#     F_data = np.load(data_path / 'F.npy')
#     Fneu_data = np.load(data_path / 'Fneu.npy')
#     iscell = np.load(data_path / 'iscell_final.npy')
#     spks = np.load(data_path / 'spks.npy')
#     return F_data, Fneu_data, iscell, spks
#
#
# def process_Fdata(suite2p_path):
#     F, Fneu, iscell, _ = read_suite2p(suite2p_path)
#     ROI = np.where(iscell[:, 0] == 1)[0]
#     F_cells = F[ROI]
#     Fneu_cells = Fneu[ROI]
#
#     Fneu_cells_median = np.median(Fneu_cells, axis=1)
#     F_cells_processed = F_cells - 0.7 * Fneu_cells
#
#     for i in range(F_cells_processed.shape[0]):
#         F_cells_processed[i, :] += Fneu_cells_median[i]
#
#     return F_cells_processed
#
#
# def process_spks_data(suite2p_path):
#     _, _, iscell, spks = read_suite2p(suite2p_path)
#     ROI = np.where(iscell[:, 0] == 1)[0]
#     spks_cells = spks[ROI]
#     return spks_cells
#
#
# def zscore_rows(matrix):
#     matrix = np.asarray(matrix, dtype=float)
#     out = np.zeros_like(matrix, dtype=float)
#
#     for i in range(matrix.shape[0]):
#         mu = np.nanmean(matrix[i])
#         sd = np.nanstd(matrix[i])
#         if not np.isfinite(sd) or sd < 1e-8:
#             sd = 1.0
#         out[i] = (matrix[i] - mu) / sd
#
#     return pd.DataFrame(out)
#
#
# def build_light_trial_dataset(signal_df,
#                               trials_ind,
#                               run_len,
#                               pre_trial_length,
#                               correct_reject,
#                               false_alarm,
#                               hit,
#                               miss,
#                               light_ind):
#     signal_run_list, _ = divide_to_trials(signal_df, trials_ind, run_len, pre_trial_length)
#     _, _, _, _, signal_light_all = divide_to_conditions(
#         signal_run_list,
#         correct_reject,
#         false_alarm,
#         hit,
#         miss,
#         light_ind
#     )
#     return signal_run_list, signal_light_all
#
#
# def run_decay_pipeline_for_signal(signal_matrix,
#                                   signal_name,
#                                   save_root,
#                                   neuron_indices,
#                                   trials_ind,
#                                   run_len,
#                                   pre_trial_length,
#                                   correct_reject,
#                                   false_alarm,
#                                   hit,
#                                   miss,
#                                   light_ind,
#                                   opto_on_inds_light,
#                                   opto_off_inds_light,
#                                   rec_hz,
#                                   baseline_frames=15,
#                                   smooth_window=3,
#                                   min_fit_r2=0.2,
#                                   min_valid_trials_per_neuron=5,
#                                   require_significant_trials=True,
#                                   trial_sig_z=2.0,
#                                   trial_sig_min_consecutive=2):
#     """
#     Runs the same decay analysis on any session-level signal matrix
#     shaped (n_neurons, total_frames).
#     """
#
#     signal_df = zscore_rows(signal_matrix)
#
#     signal_run_list, signal_light_all = build_light_trial_dataset(
#         signal_df=signal_df,
#         trials_ind=trials_ind,
#         run_len=run_len,
#         pre_trial_length=pre_trial_length,
#         correct_reject=correct_reject,
#         false_alarm=false_alarm,
#         hit=hit,
#         miss=miss,
#         light_ind=light_ind
#     )
#
#     signal_save_dir = join(save_root, signal_name)
#     os.makedirs(signal_save_dir, exist_ok=True)
#
#     with open(join(signal_save_dir, f'{signal_name}_session_zscore.pkl'), 'wb') as f:
#         pickle.dump(signal_df, f)
#
#     with open(join(signal_save_dir, f'{signal_name}_run_list.pkl'), 'wb') as f:
#         pickle.dump(signal_run_list, f)
#
#     with open(join(signal_save_dir, f'{signal_name}_light_trials.pkl'), 'wb') as f:
#         pickle.dump(signal_light_all, f)
#
#     per_trial_df, per_neuron_df = analyze_opto_decay_trial_by_trial(
#         dff_light_all=signal_light_all,
#         opto_on_inds_light=opto_on_inds_light,
#         opto_off_inds_light=opto_off_inds_light,
#         rec_hz=rec_hz,
#         save_dir=signal_save_dir,
#         neuron_indices=neuron_indices,
#         baseline_frames=baseline_frames,
#         smooth_window=smooth_window,
#         min_fit_r2=min_fit_r2,
#         min_valid_trials_per_neuron=min_valid_trials_per_neuron,
#         prefix=signal_name,
#         require_significant_trials=require_significant_trials,
#         trial_sig_z=trial_sig_z,
#         trial_sig_min_consecutive=trial_sig_min_consecutive
#     )
#
#     return {
#         'signal_df': signal_df,
#         'signal_run_list': signal_run_list,
#         'signal_light_all': signal_light_all,
#         'per_trial_df': per_trial_df,
#         'per_neuron_df': per_neuron_df,
#         'save_dir': signal_save_dir
#     }
#
#
# def compare_modalities(per_neuron_df_fluor,
#                        per_neuron_df_spks,
#                        save_dir,
#                        fluor_label='fluorescence',
#                        spks_label='deconvolved'):
#     os.makedirs(save_dir, exist_ok=True)
#
#     merge_cols = ['neuron_idx']
#     merged = pd.merge(
#         per_neuron_df_fluor,
#         per_neuron_df_spks,
#         on='neuron_idx',
#         how='inner',
#         suffixes=(f'_{fluor_label}', f'_{spks_label}')
#     )
#
#     merged.to_csv(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.csv'), index=False)
#     with open(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.pkl'), 'wb') as f:
#         pickle.dump(merged, f)
#
#     if len(merged) == 0:
#         print('No overlapping neurons between modalities after filtering.')
#         return merged
#
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#
#     # tau
#     x = merged[f'median_tau_s_{fluor_label}']
#     y = merged[f'median_tau_s_{spks_label}']
#     axes[0, 0].scatter(x, y, alpha=0.7)
#     lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
#     axes[0, 0].plot(lims, lims, linestyle='--', color='gray')
#     axes[0, 0].set_title('Median tau')
#     axes[0, 0].set_xlabel(f'{fluor_label}')
#     axes[0, 0].set_ylabel(f'{spks_label}')
#
#     # 2/3 drop
#     x = merged[f'median_t_2third_drop_s_{fluor_label}']
#     y = merged[f'median_t_2third_drop_s_{spks_label}']
#     axes[0, 1].scatter(x, y, alpha=0.7)
#     lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
#     axes[0, 1].plot(lims, lims, linestyle='--', color='gray')
#     axes[0, 1].set_title('Median 2/3 drop time')
#     axes[0, 1].set_xlabel(f'{fluor_label}')
#     axes[0, 1].set_ylabel(f'{spks_label}')
#
#     # end ratio
#     x = merged[f'median_fit_end_to_peak_ratio_{fluor_label}']
#     y = merged[f'median_fit_end_to_peak_ratio_{spks_label}']
#     axes[1, 0].scatter(x, y, alpha=0.7)
#     lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
#     axes[1, 0].plot(lims, lims, linestyle='--', color='gray')
#     axes[1, 0].set_title('Median fit-end to peak ratio')
#     axes[1, 0].set_xlabel(f'{fluor_label}')
#     axes[1, 0].set_ylabel(f'{spks_label}')
#
#     # fit quality
#     x = merged[f'median_fit_r2_{fluor_label}']
#     y = merged[f'median_fit_r2_{spks_label}']
#     axes[1, 1].scatter(x, y, alpha=0.7)
#     lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
#     axes[1, 1].plot(lims, lims, linestyle='--', color='gray')
#     axes[1, 1].set_title('Median fit R2')
#     axes[1, 1].set_xlabel(f'{fluor_label}')
#     axes[1, 1].set_ylabel(f'{spks_label}')
#
#     plt.tight_layout()
#     plt.savefig(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.png'))
#     plt.savefig(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.pdf'))
#     plt.close(fig)
#
#     # Overlay histograms of tau
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.hist(merged[f'median_tau_s_{fluor_label}'].dropna(), bins=20, alpha=0.5, label=fluor_label)
#     ax.hist(merged[f'median_tau_s_{spks_label}'].dropna(), bins=20, alpha=0.5, label=spks_label)
#     ax.set_title('Median tau distributions')
#     ax.set_xlabel('Median tau (s)')
#     ax.set_ylabel('Neuron count')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(join(save_dir, f'tau_hist_{fluor_label}_vs_{spks_label}.png'))
#     plt.savefig(join(save_dir, f'tau_hist_{fluor_label}_vs_{spks_label}.pdf'))
#     plt.close(fig)
#
#     return merged
# def plot_final_histograms_fluor_vs_deconv_separate(
#         fluor_df,
#         deconv_df,
#         save_dir,
#         prefix='final_histograms_fluor_vs_deconv_separate',
#         tau_col='median_tau_s',
#         drop_col='median_t_2third_drop_s',
#         fluor_label='Fluorescence',
#         deconv_label='Deconvolved',
#         bins=20
# ):
#     os.makedirs(save_dir, exist_ok=True)
#
#     fluor_tau = fluor_df[tau_col].to_numpy(dtype=float)
#     deconv_tau = deconv_df[tau_col].to_numpy(dtype=float)
#     fluor_drop = fluor_df[drop_col].to_numpy(dtype=float)
#     deconv_drop = deconv_df[drop_col].to_numpy(dtype=float)
#
#     fluor_tau = fluor_tau[np.isfinite(fluor_tau)]
#     deconv_tau = deconv_tau[np.isfinite(deconv_tau)]
#     fluor_drop = fluor_drop[np.isfinite(fluor_drop)]
#     deconv_drop = deconv_drop[np.isfinite(deconv_drop)]
#
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#
#     axes[0, 0].hist(fluor_tau, bins=bins, edgecolor='black')
#     axes[0, 0].set_title(f'{fluor_label}: median tau')
#     axes[0, 0].set_xlabel('Median tau (s)')
#     axes[0, 0].set_ylabel('Neuron count')
#
#     axes[0, 1].hist(deconv_tau, bins=bins, edgecolor='black')
#     axes[0, 1].set_title(f'{deconv_label}: median tau')
#     axes[0, 1].set_xlabel('Median tau (s)')
#     axes[0, 1].set_ylabel('Neuron count')
#
#     axes[1, 0].hist(fluor_drop, bins=bins, edgecolor='black')
#     axes[1, 0].set_title(f'{fluor_label}: median 2/3-drop time')
#     axes[1, 0].set_xlabel('Median 2/3-drop time (s)')
#     axes[1, 0].set_ylabel('Neuron count')
#
#     axes[1, 1].hist(deconv_drop, bins=bins, edgecolor='black')
#     axes[1, 1].set_title(f'{deconv_label}: median 2/3-drop time')
#     axes[1, 1].set_xlabel('Median 2/3-drop time (s)')
#     axes[1, 1].set_ylabel('Neuron count')
#
#     plt.tight_layout()
#     plt.savefig(join(save_dir, f'{prefix}.png'))
#     plt.savefig(join(save_dir, f'{prefix}.pdf'))
#     plt.close(fig)
# # =========================
# # Main
# # =========================
# if __name__ == '__main__':
#     mouse = 'OI116'
#     day = ['240519']
#     alpha = 0.05
#
#     ROOT_PATH = r'Z:\2p_data\{}\{}\ML\data2'.format(mouse, day[0])
#     licking_data = r'Z:\2p_data\{}\{}\ML\licking_analysis_final'.format(mouse, day[0])
#     auroc_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_z_no_sub_history'.format(mouse, day[0])
#     suite2p_path = r'Z:\2p_data\{}\{}\OI116_run1\suite2p\plane0'.format(mouse, day[0])
#     save_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\opto_decay_trialwise_v2'.format(mouse, day[0])
#
#     os.makedirs(save_loc, exist_ok=True)
#
#     # -------------------------
#     # Load data, same as your code
#     # -------------------------
#     conditions_played = np.load(join(ROOT_PATH, 'conditions_played.npy'))
#     responses = np.load(join(ROOT_PATH, 'responses.npy'))
#
#     with open(join(ROOT_PATH, 'opto_open_all.pkl'), 'rb') as f:
#         opto_on = pickle.load(f)
#
#     with open(join(ROOT_PATH, 'distract_light_on_all.pkl'), 'rb') as f:
#         distract_light = pickle.load(f)
#
#     trials_ind = np.load(join(ROOT_PATH, 'trials_ind.npy'))
#     run_len = np.load(join(ROOT_PATH, 'run_len.npy'))
#
#     with open(join(licking_data, 'lick_onset_offset_results.pkl'), 'rb') as f:
#         lick_onset_offset = pickle.load(f)
#
#     with open(join(ROOT_PATH, 'dff_running_4.5min.pkl'), 'rb') as f:
#         dff_4min = pickle.load(f)
#
#     # -------------------------
#     # load the indexes of light significant neurons from your previous analysis
#     # -------------------------
#
#     with open(join(auroc_loc, 'significant_light_neurons.pkl'), 'rb') as f:
#         sig_light_neurons = pickle.load(f)
#
#     with open(join(auroc_loc, 'significant_light_neurons_increase.pkl'), 'rb') as f:
#         sig_light_increase = pickle.load(f)
#
#     with open(join(auroc_loc, 'significant_light_neurons_decrease.pkl'), 'rb') as f:
#         sig_light_decrease = pickle.load(f)
#
#     with open(join(auroc_loc, 'significant_neurons_reward.pkl'), 'rb') as f:
#         sig_neurons_reward = pickle.load(f)
#
#     with open(join(auroc_loc, 'significant_neurons_both_light_reward.pkl'), 'rb') as f:
#         sig_neurons_both = pickle.load(f)
#
#     # light only populations
#     light_only_neurons = np.setdiff1d(sig_light_neurons, sig_neurons_reward)
#     light_only_increase_neurons = np.setdiff1d(sig_light_increase, sig_neurons_reward)
#     light_only_decrease_neurons = np.setdiff1d(sig_light_decrease, sig_neurons_reward)
#
#     print(f"sig_light_neurons: {len(sig_light_neurons)}")
#     print(f"sig_neurons_reward: {len(sig_neurons_reward)}")
#     print(f"sig_neurons_both: {len(sig_neurons_both)}")
#     print(f"light_only_neurons: {len(light_only_neurons)}")
#     print(f"light_only_increase_neurons: {len(light_only_increase_neurons)}")
#     print(f"light_only_decrease_neurons: {len(light_only_decrease_neurons)}")
#
#     # -------------------------
#     # Normalize exactly like your code:
#     # z-score each neuron across all time points
#     # -------------------------
#     dff_norm = []
#     for i in range(len(dff_4min)):
#         mean = np.nanmean(dff_4min.iloc[i])
#         std = np.nanstd(dff_4min.iloc[i])
#         if std == 0 or np.isnan(std):
#             std = 1.0
#         dff_norm.append((dff_4min.iloc[i] - mean) / std)
#     dff_norm = pd.DataFrame(dff_norm)
#
#     with open(join(save_loc, 'all_values_zscore.pkl'), 'wb') as f:
#         pickle.dump(dff_norm, f)
#
#     # -------------------------
#     # Divide to trials
#     # -------------------------
#     dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)
#
#     with open(join(save_loc, 'dff_run_list.pkl'), 'wb') as f:
#         pickle.dump(dff_run_list, f)
#
#     # -------------------------
#     # Trial indices
#     # -------------------------
#     light_ind = np.where(np.isin(conditions_played[0:len(conditions_played)], con1))[0]
#     no_light_ind = np.where((conditions_played[0:len(conditions_played)] == con2))[0]
#
#     correct_reject = np.where(responses == 4)[0]
#     false_alarm = np.where(responses == 5)[0]
#     hit = np.where(responses == 3)[0]
#     miss = np.where(responses == 2)[0]
#
#     with open(join(save_loc, 'light_ind.pkl'), 'wb') as f:
#         pickle.dump(light_ind, f)
#
#     # -------------------------
#     # Find opto onset and offset per trial
#     # -------------------------
#     opto_on_ind = []
#     for i in range(len(opto_on)):
#         changes = np.diff(np.concatenate(([0], np.asarray(opto_on[i]) > 1)))
#         ind = np.where(changes == 1)[0]
#         opto_on_ind.append(ind[0] if len(ind) > 0 else np.nan)
#
#     opto_on_ind = [i + pre_trial_length if not np.isnan(i) else np.nan for i in opto_on_ind]
#     opto_off_ind = [i + stim_len if not np.isnan(i) else np.nan for i in opto_on_ind]
#
#     with open(join(save_loc, 'opto_on_ind.pkl'), 'wb') as f:
#         pickle.dump(opto_on_ind, f)
#     with open(join(save_loc, 'opto_off_ind.pkl'), 'wb') as f:
#         pickle.dump(opto_off_ind, f)
#
#     # -------------------------
#     # Divide data to conditions
#     # -------------------------
#     dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, dff_light_all = divide_to_conditions(
#         dff_run_list,
#         correct_reject,
#         false_alarm,
#         hit,
#         miss,
#         light_ind
#     )
#
#     with open(join(save_loc, 'dff_light_all.pkl'), 'wb') as f:
#         pickle.dump(dff_light_all, f)
#
#     # -------------------------
#     # Opto indices only for light trials
#     # -------------------------
#     opto_on_inds_light = [opto_on_ind[i] for i in light_ind]
#     opto_off_inds_light = [opto_off_ind[i] for i in light_ind]
#
#     # Fill missing on/off with expected timing
#     default_on = pre_trial_length + 1 * rec_hz
#     default_off = pre_trial_length + 5 * rec_hz
#     opto_on_inds_light = [default_on if np.isnan(x) else int(x) for x in opto_on_inds_light]
#     opto_off_inds_light = [default_off if np.isnan(x) else int(x) for x in opto_off_inds_light]
#
#     # -----------------------------------------
#     # Load Suite2p deconvolved signal
#     # -----------------------------------------
#
#     spks_cells = process_spks_data(suite2p_path)
#
#     # Optional shape check
#     if spks_cells.shape[0] < np.max(light_only_increase_neurons) + 1:
#         raise ValueError('Neuron indices in light_only_increase_neurons exceed spks cell count.')
#
#     # If your fluorescence analysis already used dff_4min in the same cell order,
#     # use dff_4min as the fluorescence signal matrix
#     fluor_signal = dff_4min.to_numpy() if isinstance(dff_4min, pd.DataFrame) else np.asarray(dff_4min)
#     spks_signal = np.asarray(spks_cells)
#
#     if fluor_signal.shape[0] != spks_signal.shape[0]:
#         raise ValueError(
#             f'Fluorescence and spks have different neuron counts: '
#             f'{fluor_signal.shape[0]} vs {spks_signal.shape[0]}. '
#             f'This must match after iscell filtering.'
#         )
#
#     # -----------------------------------------
#     # Choose neuron subset
#     # -----------------------------------------
#     neuron_subset = light_only_increase_neurons
#
#     # -----------------------------------------
#     # Save root
#     # -----------------------------------------
#     multisignal_save_dir = join(save_loc, 'decay_multisignal')
#     os.makedirs(multisignal_save_dir, exist_ok=True)
#
#     # -----------------------------------------
#     # Run fluorescence version
#     # -----------------------------------------
#     fluor_results = run_decay_pipeline_for_signal(
#         signal_matrix=fluor_signal,
#         signal_name='fluorescence',
#         save_root=multisignal_save_dir,
#         neuron_indices=neuron_subset,
#         trials_ind=trials_ind,
#         run_len=run_len,
#         pre_trial_length=pre_trial_length,
#         correct_reject=correct_reject,
#         false_alarm=false_alarm,
#         hit=hit,
#         miss=miss,
#         light_ind=light_ind,
#         opto_on_inds_light=opto_on_inds_light,
#         opto_off_inds_light=opto_off_inds_light,
#         rec_hz=rec_hz,
#         baseline_frames=1 * rec_hz,
#         smooth_window=3,
#         min_fit_r2=0.2,
#         min_valid_trials_per_neuron=5,
#         require_significant_trials=True,
#         trial_sig_z=2.0,
#         trial_sig_min_consecutive=2
#     )
#
#     # -----------------------------------------
#     # Run deconvolved version
#     # For spks, a slightly larger smoothing window is often helpful
#     # -----------------------------------------
#     spks_results = run_decay_pipeline_for_signal(
#         signal_matrix=spks_signal,
#         signal_name='deconvolved_spks',
#         save_root=multisignal_save_dir,
#         neuron_indices=neuron_subset,
#         trials_ind=trials_ind,
#         run_len=run_len,
#         pre_trial_length=pre_trial_length,
#         correct_reject=correct_reject,
#         false_alarm=false_alarm,
#         hit=hit,
#         miss=miss,
#         light_ind=light_ind,
#         opto_on_inds_light=opto_on_inds_light,
#         opto_off_inds_light=opto_off_inds_light,
#         rec_hz=rec_hz,
#         baseline_frames=1 * rec_hz,
#         smooth_window=5,
#         min_fit_r2=0.2,
#         min_valid_trials_per_neuron=5,
#         require_significant_trials=True,
#         trial_sig_z=2.0,
#         trial_sig_min_consecutive=2
#     )
#
#     # -----------------------------------------
#     # Compare together
#     # -----------------------------------------
#     comparison_dir = join(multisignal_save_dir, 'comparison')
#     comparison_df = compare_modalities(
#         per_neuron_df_fluor=fluor_results['per_neuron_df'],
#         per_neuron_df_spks=spks_results['per_neuron_df'],
#         save_dir=comparison_dir,
#         fluor_label='fluorescence',
#         spks_label='deconvolved'
#     )
#
#     plot_final_histograms_fluor_vs_deconv_separate(
#         fluor_df=fluor_results['per_neuron_df'],
#         deconv_df=spks_results['per_neuron_df'],
#         save_dir=join(multisignal_save_dir, 'comparison'),
#         prefix='final_histograms_fluor_vs_deconv',
#         tau_col='median_tau_s',  # or 'median_tau_fast_s' if that is your summary column
#         drop_col='median_t_2third_drop_s'
#     )
#
#     print('Finished fluorescence and deconvolved decay analysis.')
#     print(f'Fluorescence neurons summarized: {len(fluor_results["per_neuron_df"])}')
#     print(f'Deconvolved neurons summarized: {len(spks_results["per_neuron_df"])}')
#     print(f'Overlapping neurons in comparison: {len(comparison_df)}')

def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def max_consecutive_true(mask):
    mask = np.asarray(mask, dtype=bool)
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def trial_has_significant_opto_response(trial_trace,
                                        on_frame,
                                        off_frame,
                                        rec_hz,
                                        baseline_frames=None,
                                        smooth_window=3,
                                        z_thresh=2.0,
                                        min_consecutive_frames=2):
    """
    Operational per-trial significance rule:
    a trial is called opto-responsive if, during the opto window,
    the smoothed signal crosses baseline + z_thresh*baseline_std
    for at least min_consecutive_frames.

    This is not a formal p-value, it is a per-trial threshold rule.
    """
    trial_trace = np.asarray(trial_trace, dtype=float)

    if baseline_frames is None:
        baseline_frames = rec_hz

    valid_mask = ~np.isnan(trial_trace)
    if not np.any(valid_mask):
        return None

    n_frames = np.where(valid_mask)[0][-1] + 1
    on_frame = int(np.clip(on_frame, 0, n_frames - 1))
    off_frame = int(np.clip(off_frame, on_frame + 1, n_frames))

    trace_sm = smooth_trace_nan(trial_trace, window=smooth_window)

    bs_start = max(0, on_frame - baseline_frames)
    bs_stop = max(bs_start + 1, on_frame)
    baseline_seg = trace_sm[bs_start:bs_stop]

    if baseline_seg.size < 3 or np.all(np.isnan(baseline_seg)):
        return {
            'is_significant': False,
            'baseline_mean': np.nan,
            'baseline_std': np.nan,
            'threshold': np.nan,
            'peak_z': np.nan,
            'max_run': 0
        }

    baseline_mean = safe_nanmean(baseline_seg)
    baseline_std = np.nanstd(baseline_seg)

    if not np.isfinite(baseline_std) or baseline_std < 1e-8:
        baseline_std = 1e-8

    opto_seg = trace_sm[on_frame:off_frame]
    if opto_seg.size == 0 or np.all(np.isnan(opto_seg)):
        return {
            'is_significant': False,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'threshold': np.nan,
            'peak_z': np.nan,
            'max_run': 0
        }

    threshold = baseline_mean + z_thresh * baseline_std
    supra = opto_seg > threshold
    max_run = max_consecutive_true(supra)

    peak_val = np.nanmax(opto_seg)
    peak_z = (peak_val - baseline_mean) / baseline_std

    is_significant = (peak_z >= z_thresh) and (max_run >= min_consecutive_frames)

    return {
        'is_significant': bool(is_significant),
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'threshold': threshold,
        'peak_z': peak_z,
        'max_run': max_run
    }
def summarize_trialwise_decay(per_trial_df,
                              min_valid_trials_per_neuron=5,
                              ratio_col='fit_end_to_peak_ratio'):
    if len(per_trial_df) == 0:
        return pd.DataFrame()

    summary_rows = []

    for neuron_idx, subdf in per_trial_df.groupby('neuron_idx'):
        subdf = subdf.copy()
        subdf = subdf[np.isfinite(subdf['tau_s'])]

        if len(subdf) < min_valid_trials_per_neuron:
            continue

        row = {
            'neuron_idx': neuron_idx,
            'n_valid_trials': len(subdf),

            'median_tau_s': np.nanmedian(subdf['tau_s']),
            'mean_tau_s': np.nanmean(subdf['tau_s']),
            'tau_iqr_s': np.nanpercentile(subdf['tau_s'], 75) - np.nanpercentile(subdf['tau_s'], 25),

            'median_peak_value': np.nanmedian(subdf['peak_value']),
            'median_peak_time_from_on_s': np.nanmedian(subdf['peak_time_from_on_s']),
            'median_fit_r2': np.nanmedian(subdf['fit_r2']),
            'median_t_2third_drop_s': np.nanmedian(subdf['t_2third_drop_s']),
            'median_fit_end_to_peak_ratio': np.nanmedian(subdf[ratio_col]),

            'median_odor_start_to_peak_ratio': np.nanmedian(subdf['odor_start_to_peak_ratio']),
            'median_odor_start_to_peak_percent': np.nanmedian(subdf['odor_start_to_peak_percent']),
            'median_odor_start_decrease_percent': np.nanmedian(subdf['odor_start_decrease_percent']),
        }

        if 'tau_fast_s' in subdf.columns:
            row['median_tau_fast_s'] = np.nanmedian(subdf['tau_fast_s'])
            row['mean_tau_fast_s'] = np.nanmean(subdf['tau_fast_s'])

        if 'tau_slow_s' in subdf.columns:
            row['median_tau_slow_s'] = np.nanmedian(subdf['tau_slow_s'])
            row['mean_tau_slow_s'] = np.nanmean(subdf['tau_slow_s'])

        if 'w_fast' in subdf.columns:
            row['median_w_fast'] = np.nanmedian(subdf['w_fast'])
            row['mean_w_fast'] = np.nanmean(subdf['w_fast'])

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values('median_tau_s').reset_index(drop=True)

    return summary_df
def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)
def plot_trialwise_decay_summary(per_trial_df, per_neuron_df, save_dir, prefix):
    if len(per_trial_df) == 0 or len(per_neuron_df) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    axes[0].hist(per_trial_df['tau_s'].dropna(), bins=30, edgecolor='black')
    axes[0].set_title('All valid trial taus')
    axes[0].set_xlabel('Tau (s)')
    axes[0].set_ylabel('Count')

    axes[1].hist(per_neuron_df['median_tau_s'].dropna(), bins=25, edgecolor='black')
    axes[1].set_title('Median tau per neuron')
    axes[1].set_xlabel('Median tau (s)')
    axes[1].set_ylabel('Neuron count')

    # axes[1, 0].scatter(per_neuron_df['median_tau_s'],
    #                    per_neuron_df['median_fit_end_to_peak_ratio'],
    #                    alpha=0.7)
    # axes[1, 0].set_title('Median tau vs fit-end/peak ratio')
    # axes[1, 0].set_xlabel('Median tau (s)')
    # axes[1, 0].set_ylabel('Median fit-end/peak ratio')
    #
    # axes[1, 1].scatter(per_neuron_df['n_valid_trials'],
    #                    per_neuron_df['median_tau_s'],
    #                    alpha=0.7)
    # axes[1, 1].set_title('Valid trials vs median tau')
    # axes[1, 1].set_xlabel('Number of valid trials')
    # axes[1, 1].set_ylabel('Median tau (s)')

    plt.tight_layout()
    plt.savefig(join(save_dir, f'{prefix}_summary.png'))
    plt.savefig(join(save_dir, f'{prefix}_summary.pdf'))
    plt.close(fig)



def exp_decay_anchored(t, tau, C, peak_value):
    """
    Exponential decay anchored to the measured peak.
    At t=0, y(0) = peak_value exactly.
    """
    return C + (peak_value - C) * np.exp(-t / tau)
def plot_example_fits(dff_light_all,
                      opto_on_inds_light,
                      opto_off_inds_light,
                      per_trial_df,
                      per_neuron_df,
                      save_dir,
                      prefix,
                      baseline_frames=15,
                      smooth_window=3,
                      n_neurons_to_plot=8,
                      n_trials_per_neuron=4):
    if len(per_neuron_df) == 0 or len(per_trial_df) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    plot_df = per_neuron_df.sort_values('median_peak_value', ascending=False).head(n_neurons_to_plot)
    selected_neurons = plot_df['neuron_idx'].values

    for neuron_idx in selected_neurons:
        neuron_df = dff_light_all[int(neuron_idx)]

        valid_trials_df = per_trial_df[per_trial_df['neuron_idx'] == neuron_idx].copy()
        if len(valid_trials_df) == 0:
            continue

        valid_trials_df = valid_trials_df.sort_values('peak_value', ascending=False)
        trials_to_show = valid_trials_df['trial_idx_local'].values[:n_trials_per_neuron]

        fig, axes = plt.subplots(len(trials_to_show), 1, figsize=(10, 3 * len(trials_to_show)), sharex=True)
        if len(trials_to_show) == 1:
            axes = [axes]

        for ax, trial_idx in zip(axes, trials_to_show):
            trial_trace = neuron_df.iloc[int(trial_idx), :].to_numpy(dtype=float)
            on_frame = int(opto_on_inds_light[int(trial_idx)])
            off_frame = int(opto_off_inds_light[int(trial_idx)])

            fit_res = fit_decay_one_trial(
                trial_trace=trial_trace,
                on_frame=on_frame,
                off_frame=off_frame,
                rec_hz=rec_hz,
                baseline_frames=baseline_frames,
                smooth_window=smooth_window
            )

            # use only the real part of the trial for plotting
            trial_trace = np.asarray(trial_trace, dtype=float)

            valid_mask = np.isfinite(trial_trace)
            if not np.any(valid_mask):
                continue

            last_valid_idx = np.where(valid_mask)[0][-1]
            len_valid = last_valid_idx + 1

            trial_trace_plot = trial_trace[:len_valid]
            t_axis = (np.arange(len_valid) - on_frame) / rec_hz

            # Opto window shading
            ax.axvspan(0, (off_frame - on_frame) / rec_hz, color='gold', alpha=0.15)

            if fit_res is None:
                ax.plot(t_axis, trial_trace, color='black', alpha=0.7)
                ax.axvline(0, linestyle='--', color='gold')
                ax.axvline((off_frame - on_frame) / rec_hz, linestyle='--', color='orange')
                ax.set_title(f'Neuron {neuron_idx}, trial {trial_idx}, fit failed')
                ax.set_xlim(t_axis[0], t_axis[-1])
                continue

            ax.plot(t_axis, fit_res['trace_bs'][:len_valid], color='gray', alpha=0.5, label='Baseline-subtracted')
            ax.plot(t_axis, fit_res['trace_sm'][:len_valid], color='black', linewidth=2, label='Smoothed')
            ax.set_xlim(t_axis[0], t_axis[-1])

            ax.axvline(0, linestyle='--', color='gold', label='Opto on')
            ax.axvline((off_frame - on_frame) / rec_hz, linestyle='--', color='orange', label='Opto off')
            ax.axvline((fit_res['peak_idx'] - on_frame) / rec_hz, linestyle='--', color='red', label='Peak')

            peak_idx = fit_res['peak_idx']
            fit_end_frame = min(fit_res['fit_end_frame'], len_valid)
            fit_x = (np.arange(peak_idx, fit_end_frame) - on_frame) / rec_hz
            ax.plot(fit_x, fit_res['y_fit'][:len(fit_x)], color='dodgerblue', linewidth=2, label='Double-exp fit')

            ax.set_title(
                f'Neuron {neuron_idx}, trial {trial_idx} | '
                f'tau={fit_res["tau_s"]:.3f}s | '
                f'R2={fit_res["fit_r2"]:.2f} | '
                f'2/3 drop={fit_res["t_2third_drop_s"]:.3f}s'
            )
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time from opto onset (s)')
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}_neuron_{int(neuron_idx)}_examples.png'))
        plt.savefig(join(save_dir, f'{prefix}_neuron_{int(neuron_idx)}_examples.pdf'))
        plt.close(fig)

def analyze_opto_decay_trial_by_trial(dff_light_all,
                                      opto_on_inds_light,
                                      opto_off_inds_light,
                                      rec_hz,
                                      save_dir,
                                      neuron_indices=None,
                                      baseline_frames=None,
                                      smooth_window=3,
                                      min_peak=0.25,
                                      min_fit_r2=0.2,
                                      min_valid_trials_per_neuron=5,
                                      prefix='all_neurons',
                                      require_significant_trials=True,
                                      trial_sig_z=2.0,
                                      trial_sig_min_consecutive=2):
    os.makedirs(save_dir, exist_ok=True)

    if baseline_frames is None:
        baseline_frames = rec_hz

    if neuron_indices is None:
        neuron_indices = np.arange(len(dff_light_all))

    neuron_indices = np.asarray(neuron_indices, dtype=int)
    opto_on_inds_light = np.asarray(opto_on_inds_light)
    opto_off_inds_light = np.asarray(opto_off_inds_light)

    per_trial_rows = []

    for neuron_idx in neuron_indices:
        neuron_df = dff_light_all[int(neuron_idx)]

        for trial_local_idx in range(neuron_df.shape[0]):
            trial_trace = neuron_df.iloc[trial_local_idx, :].to_numpy(dtype=float)
            on_frame = int(opto_on_inds_light[trial_local_idx])
            off_frame = int(opto_off_inds_light[trial_local_idx])

            sig_info = trial_has_significant_opto_response(
                trial_trace=trial_trace,
                on_frame=on_frame,
                off_frame=off_frame,
                rec_hz=rec_hz,
                baseline_frames=baseline_frames,
                smooth_window=smooth_window,
                z_thresh=trial_sig_z,
                min_consecutive_frames=trial_sig_min_consecutive
            )

            if require_significant_trials and (not sig_info['is_significant']):
                continue

            fit_res = fit_decay_one_trial(
                trial_trace=trial_trace,
                on_frame=on_frame,
                off_frame=off_frame,
                rec_hz=rec_hz,
                baseline_frames=baseline_frames,
                smooth_window=smooth_window
            )

            if fit_res is None:
                continue

            if not np.isfinite(fit_res['peak_value']) or fit_res['peak_value'] < min_peak:
                continue

            if min_fit_r2 is not None:
                if not np.isfinite(fit_res['fit_r2']) or fit_res['fit_r2'] < min_fit_r2:
                    continue

            per_trial_rows.append({
                'neuron_idx': int(neuron_idx),
                'trial_idx_local': int(trial_local_idx),
                'peak_time_from_on_s': fit_res['peak_time_from_on_s'],
                'peak_value': fit_res['peak_value'],
                'tau_s': fit_res['tau_s'],
                'tau_fast_s': fit_res['tau_fast_s'],
                'tau_slow_s': fit_res['tau_slow_s'],
                'w_fast': fit_res['w_fast'],
                'fit_r2': fit_res['fit_r2'],
                't_2third_drop_s': fit_res['t_2third_drop_s'],

                'opto_end_mean': fit_res['opto_end_mean'],
                'opto_end_to_peak_ratio': fit_res['opto_end_to_peak_ratio'],

                'odor_start_mean': fit_res['odor_start_mean'],
                'odor_start_to_peak_ratio': fit_res['odor_start_to_peak_ratio'],
                'odor_start_to_peak_percent': fit_res['odor_start_to_peak_percent'],
                'odor_start_decrease_percent': fit_res['odor_start_decrease_percent'],

                'fit_end_mean': fit_res['fit_end_mean'],
                'fit_end_to_peak_ratio': fit_res['fit_end_to_peak_ratio'],

                'fit_A': fit_res['fit_A'],
                'fit_C': fit_res['fit_C'],
                'fit_end_frame': fit_res['fit_end_frame'],
                'trial_peak_z': sig_info['peak_z'],
                'trial_sig_max_run': sig_info['max_run'],
                'trial_sig_threshold': sig_info['threshold'],
            })

    per_trial_df = pd.DataFrame(per_trial_rows)

    per_neuron_df = summarize_trialwise_decay(
        per_trial_df,
        min_valid_trials_per_neuron=min_valid_trials_per_neuron,
        ratio_col='fit_end_to_peak_ratio'
    )

    per_trial_df.to_csv(join(save_dir, f'{prefix}_per_trial.csv'), index=False)
    per_neuron_df.to_csv(join(save_dir, f'{prefix}_per_neuron.csv'), index=False)

    with open(join(save_dir, f'{prefix}_per_trial.pkl'), 'wb') as f:
        pickle.dump(per_trial_df, f)

    with open(join(save_dir, f'{prefix}_per_neuron.pkl'), 'wb') as f:
        pickle.dump(per_neuron_df, f)

    plot_trialwise_decay_summary(per_trial_df, per_neuron_df, save_dir, prefix)

    plot_example_fits(
        dff_light_all=dff_light_all,
        opto_on_inds_light=opto_on_inds_light,
        opto_off_inds_light=opto_off_inds_light,
        per_trial_df=per_trial_df,
        per_neuron_df=per_neuron_df,
        save_dir=save_dir,
        prefix=prefix,
        baseline_frames=baseline_frames,
        smooth_window=smooth_window,
        n_neurons_to_plot=8,
        n_trials_per_neuron=4
    )

    return per_trial_df, per_neuron_df


def read_suite2p(data_path):
    data_path = Path(data_path)
    F_data = np.load(data_path / 'F.npy')
    Fneu_data = np.load(data_path / 'Fneu.npy')
    iscell = np.load(data_path / 'iscell_final.npy')
    spks = np.load(data_path / 'spks.npy')
    return F_data, Fneu_data, iscell, spks


def process_Fdata(suite2p_path):
    F, Fneu, iscell, _ = read_suite2p(suite2p_path)
    ROI = np.where(iscell[:, 0] == 1)[0]
    F_cells = F[ROI]
    Fneu_cells = Fneu[ROI]

    Fneu_cells_median = np.median(Fneu_cells, axis=1)
    F_cells_processed = F_cells - 0.7 * Fneu_cells

    for i in range(F_cells_processed.shape[0]):
        F_cells_processed[i, :] += Fneu_cells_median[i]

    return F_cells_processed


def process_spks_data(suite2p_path):
    _, _, iscell, spks = read_suite2p(suite2p_path)
    ROI = np.where(iscell[:, 0] == 1)[0]
    spks_cells = spks[ROI]
    return spks_cells


def zscore_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    out = np.zeros_like(matrix, dtype=float)

    for i in range(matrix.shape[0]):
        mu = np.nanmean(matrix[i])
        sd = np.nanstd(matrix[i])
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        out[i] = (matrix[i] - mu) / sd

    return pd.DataFrame(out)


def build_light_trial_dataset(signal_df,
                              trials_ind,
                              run_len,
                              pre_trial_length,
                              correct_reject,
                              false_alarm,
                              hit,
                              miss,
                              light_ind):
    signal_run_list, _ = divide_to_trials(signal_df, trials_ind, run_len, pre_trial_length)
    _, _, _, _, signal_light_all = divide_to_conditions(
        signal_run_list,
        correct_reject,
        false_alarm,
        hit,
        miss,
        light_ind
    )
    return signal_run_list, signal_light_all


def run_decay_pipeline_for_signal(signal_matrix,
                                  signal_name,
                                  save_root,
                                  neuron_indices,
                                  trials_ind,
                                  run_len,
                                  pre_trial_length,
                                  correct_reject,
                                  false_alarm,
                                  hit,
                                  miss,
                                  light_ind,
                                  opto_on_inds_light,
                                  opto_off_inds_light,
                                  rec_hz,
                                  baseline_frames=15,
                                  smooth_window=3,
                                  min_fit_r2=0.2,
                                  min_valid_trials_per_neuron=5,
                                  require_significant_trials=True,
                                  trial_sig_z=2.0,
                                  trial_sig_min_consecutive=2):
    """
    Runs the same decay analysis on any session-level signal matrix
    shaped (n_neurons, total_frames).
    """

    signal_df = zscore_rows(signal_matrix)

    signal_run_list, signal_light_all = build_light_trial_dataset(
        signal_df=signal_df,
        trials_ind=trials_ind,
        run_len=run_len,
        pre_trial_length=pre_trial_length,
        correct_reject=correct_reject,
        false_alarm=false_alarm,
        hit=hit,
        miss=miss,
        light_ind=light_ind
    )

    signal_save_dir = join(save_root, signal_name)
    os.makedirs(signal_save_dir, exist_ok=True)

    with open(join(signal_save_dir, f'{signal_name}_session_zscore.pkl'), 'wb') as f:
        pickle.dump(signal_df, f)

    with open(join(signal_save_dir, f'{signal_name}_run_list.pkl'), 'wb') as f:
        pickle.dump(signal_run_list, f)

    with open(join(signal_save_dir, f'{signal_name}_light_trials.pkl'), 'wb') as f:
        pickle.dump(signal_light_all, f)

    per_trial_df, per_neuron_df = analyze_opto_decay_trial_by_trial(
        dff_light_all=signal_light_all,
        opto_on_inds_light=opto_on_inds_light,
        opto_off_inds_light=opto_off_inds_light,
        rec_hz=rec_hz,
        save_dir=signal_save_dir,
        neuron_indices=neuron_indices,
        baseline_frames=baseline_frames,
        smooth_window=smooth_window,
        min_fit_r2=min_fit_r2,
        min_valid_trials_per_neuron=min_valid_trials_per_neuron,
        prefix=signal_name,
        require_significant_trials=require_significant_trials,
        trial_sig_z=trial_sig_z,
        trial_sig_min_consecutive=trial_sig_min_consecutive
    )

    return {
        'signal_df': signal_df,
        'signal_run_list': signal_run_list,
        'signal_light_all': signal_light_all,
        'per_trial_df': per_trial_df,
        'per_neuron_df': per_neuron_df,
        'save_dir': signal_save_dir
    }


def compare_modalities(per_neuron_df_fluor,
                       per_neuron_df_spks,
                       save_dir,
                       fluor_label='fluorescence',
                       spks_label='deconvolved'):
    os.makedirs(save_dir, exist_ok=True)

    merge_cols = ['neuron_idx']
    merged = pd.merge(
        per_neuron_df_fluor,
        per_neuron_df_spks,
        on='neuron_idx',
        how='inner',
        suffixes=(f'_{fluor_label}', f'_{spks_label}')
    )

    merged.to_csv(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.csv'), index=False)
    with open(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.pkl'), 'wb') as f:
        pickle.dump(merged, f)

    if len(merged) == 0:
        print('No overlapping neurons between modalities after filtering.')
        return merged

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # tau
    x = merged[f'median_tau_s_{fluor_label}']
    y = merged[f'median_tau_s_{spks_label}']
    axes[0, 0].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[0, 0].plot(lims, lims, linestyle='--', color='gray')
    axes[0, 0].set_title('Median tau')
    axes[0, 0].set_xlabel(f'{fluor_label}')
    axes[0, 0].set_ylabel(f'{spks_label}')

    # 2/3 drop
    x = merged[f'median_t_2third_drop_s_{fluor_label}']
    y = merged[f'median_t_2third_drop_s_{spks_label}']
    axes[0, 1].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[0, 1].plot(lims, lims, linestyle='--', color='gray')
    axes[0, 1].set_title('Median 2/3 drop time')
    axes[0, 1].set_xlabel(f'{fluor_label}')
    axes[0, 1].set_ylabel(f'{spks_label}')

    # end ratio
    x = merged[f'median_fit_end_to_peak_ratio_{fluor_label}']
    y = merged[f'median_fit_end_to_peak_ratio_{spks_label}']
    axes[1, 0].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[1, 0].plot(lims, lims, linestyle='--', color='gray')
    axes[1, 0].set_title('Median fit-end to peak ratio')
    axes[1, 0].set_xlabel(f'{fluor_label}')
    axes[1, 0].set_ylabel(f'{spks_label}')

    # fit quality
    x = merged[f'median_fit_r2_{fluor_label}']
    y = merged[f'median_fit_r2_{spks_label}']
    axes[1, 1].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[1, 1].plot(lims, lims, linestyle='--', color='gray')
    axes[1, 1].set_title('Median fit R2')
    axes[1, 1].set_xlabel(f'{fluor_label}')
    axes[1, 1].set_ylabel(f'{spks_label}')

    plt.tight_layout()
    plt.savefig(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.png'))
    plt.savefig(join(save_dir, f'comparison_{fluor_label}_vs_{spks_label}.pdf'))
    plt.close(fig)

    # Overlay histograms of tau
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(merged[f'median_tau_s_{fluor_label}'].dropna(), bins=20, alpha=0.5, label=fluor_label)
    ax.hist(merged[f'median_tau_s_{spks_label}'].dropna(), bins=20, alpha=0.5, label=spks_label)
    ax.set_title('Median tau distributions')
    ax.set_xlabel('Median tau (s)')
    ax.set_ylabel('Neuron count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, f'tau_hist_{fluor_label}_vs_{spks_label}.png'))
    plt.savefig(join(save_dir, f'tau_hist_{fluor_label}_vs_{spks_label}.pdf'))
    plt.close(fig)

    return merged
def plot_final_histograms_fluor_vs_deconv_separate(
        fluor_df,
        deconv_df,
        save_dir,
        prefix='final_histograms_fluor_vs_deconv_separate',
        tau_col='median_tau_s',
        drop_col='median_t_2third_drop_s',
        fluor_label='Fluorescence',
        deconv_label='Deconvolved',
        bins=20
):
    os.makedirs(save_dir, exist_ok=True)

    fluor_tau = fluor_df[tau_col].to_numpy(dtype=float)
    deconv_tau = deconv_df[tau_col].to_numpy(dtype=float)
    fluor_drop = fluor_df[drop_col].to_numpy(dtype=float)
    deconv_drop = deconv_df[drop_col].to_numpy(dtype=float)

    fluor_tau = fluor_tau[np.isfinite(fluor_tau)]
    deconv_tau = deconv_tau[np.isfinite(deconv_tau)]
    fluor_drop = fluor_drop[np.isfinite(fluor_drop)]
    deconv_drop = deconv_drop[np.isfinite(deconv_drop)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(fluor_tau, bins=bins, edgecolor='black')
    axes[0, 0].set_title(f'{fluor_label}: median tau')
    axes[0, 0].set_xlabel('Median tau (s)')
    axes[0, 0].set_ylabel('Neuron count')

    axes[0, 1].hist(deconv_tau, bins=bins, edgecolor='black')
    axes[0, 1].set_title(f'{deconv_label}: median tau')
    axes[0, 1].set_xlabel('Median tau (s)')
    axes[0, 1].set_ylabel('Neuron count')

    axes[1, 0].hist(fluor_drop, bins=bins, edgecolor='black')
    axes[1, 0].set_title(f'{fluor_label}: median 2/3-drop time')
    axes[1, 0].set_xlabel('Median 2/3-drop time (s)')
    axes[1, 0].set_ylabel('Neuron count')

    axes[1, 1].hist(deconv_drop, bins=bins, edgecolor='black')
    axes[1, 1].set_title(f'{deconv_label}: median 2/3-drop time')
    axes[1, 1].set_xlabel('Median 2/3-drop time (s)')
    axes[1, 1].set_ylabel('Neuron count')

    plt.tight_layout()
    plt.savefig(join(save_dir, f'{prefix}.png'))
    plt.savefig(join(save_dir, f'{prefix}.pdf'))
    plt.close(fig)
def plot_overlay_fluor_deconv_per_neuron(
        fluor_light_all,
        deconv_light_all,
        fluor_per_trial_df,
        deconv_per_trial_df,
        opto_on_inds_light,
        opto_off_inds_light,
        neuron_indices,
        save_dir,
        baseline_frames=15,
        fluor_smooth_window=5,
        deconv_smooth_window=5,
        max_trials_per_neuron=None
):
    """
    Save one figure per neuron.
    Each subplot is one trial that was included in BOTH fluorescence and deconvolved analyses.
    """
    os.makedirs(save_dir, exist_ok=True)

    for neuron_idx in neuron_indices:
        neuron_idx = int(neuron_idx)

        fluor_trials_df = fluor_per_trial_df[fluor_per_trial_df['neuron_idx'] == neuron_idx].copy()
        deconv_trials_df = deconv_per_trial_df[deconv_per_trial_df['neuron_idx'] == neuron_idx].copy()

        if len(fluor_trials_df) == 0 or len(deconv_trials_df) == 0:
            continue

        fluor_trial_ids = set(fluor_trials_df['trial_idx_local'].astype(int).tolist())
        deconv_trial_ids = set(deconv_trials_df['trial_idx_local'].astype(int).tolist())
        common_trial_ids = sorted(list(fluor_trial_ids & deconv_trial_ids))

        if len(common_trial_ids) == 0:
            continue

        if max_trials_per_neuron is not None:
            common_trial_ids = common_trial_ids[:max_trials_per_neuron]

        fig, axes = plt.subplots(
            len(common_trial_ids), 1,
            figsize=(10, 3 * len(common_trial_ids)),
            sharex=False
        )
        if len(common_trial_ids) == 1:
            axes = [axes]

        for ax, tr in zip(axes, common_trial_ids):
            fluor_trace = fluor_light_all[neuron_idx].iloc[tr, :].to_numpy(dtype=float)
            deconv_trace = deconv_light_all[neuron_idx].iloc[tr, :].to_numpy(dtype=float)

            fluor_valid = np.isfinite(fluor_trace)
            deconv_valid = np.isfinite(deconv_trace)
            if not np.any(fluor_valid) or not np.any(deconv_valid):
                ax.set_visible(False)
                continue

            fluor_len = np.where(fluor_valid)[0][-1] + 1
            deconv_len = np.where(deconv_valid)[0][-1] + 1
            common_len = min(fluor_len, deconv_len)

            fluor_trace = fluor_trace[:common_len]
            deconv_trace = deconv_trace[:common_len]

            on_frame = int(np.clip(opto_on_inds_light[tr], 0, common_len - 1))
            off_frame = int(np.clip(opto_off_inds_light[tr], on_frame + 1, common_len))

            # baseline subtract
            bs_start = max(0, on_frame - baseline_frames)
            bs_stop = max(bs_start + 1, on_frame)

            fluor_baseline = safe_nanmean(fluor_trace[bs_start:bs_stop])
            deconv_baseline = safe_nanmean(deconv_trace[bs_start:bs_stop])

            fluor_trace_bs = fluor_trace - fluor_baseline
            deconv_trace_bs = deconv_trace - deconv_baseline

            fluor_sm = smooth_trace_nan(fluor_trace_bs, window=fluor_smooth_window)
            deconv_sm = smooth_trace_nan(deconv_trace_bs, window=deconv_smooth_window)

            # peaks during opto
            fluor_peak_idx = find_first_significant_peak(
                trace_sm=fluor_sm,
                on_frame=on_frame,
                off_frame=off_frame,
                baseline_frames=baseline_frames,
                min_peak_height=0.25,
                min_peak_prominence=0.2,
                min_peak_distance=2,
                baseline_z_thresh=2.0
            )
            if fluor_peak_idx is None:
                peak_window = fluor_sm[on_frame:off_frame]
                if len(peak_window) > 0 and not np.all(np.isnan(peak_window)):
                    fluor_peak_idx = on_frame + int(np.nanargmax(peak_window))
                else:
                    fluor_peak_idx = on_frame

            deconv_peak_idx = find_first_significant_peak(
                trace_sm=deconv_sm,
                on_frame=on_frame,
                off_frame=off_frame,
                baseline_frames=baseline_frames,
                min_peak_height=0.25,
                min_peak_prominence=0.2,
                min_peak_distance=2,
                baseline_z_thresh=2.0
            )
            if deconv_peak_idx is None:
                peak_window = deconv_sm[on_frame:off_frame]
                if len(peak_window) > 0 and not np.all(np.isnan(peak_window)):
                    deconv_peak_idx = on_frame + int(np.nanargmax(peak_window))
                else:
                    deconv_peak_idx = on_frame

            t_axis = (np.arange(common_len) - on_frame) / rec_hz

            # opto shading
            ax.axvspan(0, (off_frame - on_frame) / rec_hz, color='gold', alpha=0.12)

            # odor-start shading = first 0.5 sec after opto
            odor_end = min(common_len, off_frame + int(round(0.5 * rec_hz)))
            ax.axvspan(
                (off_frame - on_frame) / rec_hz,
                (odor_end - on_frame) / rec_hz,
                color='lightblue',
                alpha=0.12
            )

            ax.plot(t_axis, fluor_trace_bs, alpha=0.35, linewidth=2, color='black', label='Fluorescence raw')
            ax.plot(t_axis, fluor_sm, linewidth=2, label='Fluorescence smoothed')

            # ax.plot(t_axis, deconv_trace_bs, alpha=0.35, linewidth=1, label='Deconvolved raw')
            ax.bar(
                t_axis,
                deconv_trace_bs,
                width=1 / rec_hz,
                alpha=0.4,
                label='Baseline-subtracted deconvolved activity'
            )
            ax.plot(t_axis, deconv_sm, linewidth=2, label='Deconvolved smoothed')

            ax.axvline(0, linestyle='--', color='gold', linewidth=1)
            ax.axvline((off_frame - on_frame) / rec_hz, linestyle='--', color='orange', linewidth=1)

            ax.axvline((fluor_peak_idx - on_frame) / rec_hz, linestyle='--', linewidth=1, label='Fluor peak')
            ax.axvline((deconv_peak_idx - on_frame) / rec_hz, linestyle=':', linewidth=1, label='Deconv peak')

            # trial-level odor quantification from saved tables
            fluor_trial_row = fluor_trials_df[fluor_trials_df['trial_idx_local'] == tr]
            deconv_trial_row = deconv_trials_df[deconv_trials_df['trial_idx_local'] == tr]

            fluor_odor_pct = np.nan
            deconv_odor_pct = np.nan
            if len(fluor_trial_row) > 0 and 'odor_start_to_peak_percent' in fluor_trial_row.columns:
                fluor_odor_pct = fluor_trial_row['odor_start_to_peak_percent'].iloc[0]
            if len(deconv_trial_row) > 0 and 'odor_start_to_peak_percent' in deconv_trial_row.columns:
                deconv_odor_pct = deconv_trial_row['odor_start_to_peak_percent'].iloc[0]

            ax.set_title(
                f'Neuron {neuron_idx}, trial {tr} | '
                f'Fluor odor-start={fluor_odor_pct:.1f}% | '
                f'Deconv odor-start={deconv_odor_pct:.1f}%'
            )
            ax.set_xlabel('Time from opto onset (s)')
            ax.set_ylabel('Baseline-subtracted signal')
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(join(save_dir, f'neuron_{neuron_idx}_fluor_vs_deconv_overlay_all_trials.png'))
        plt.savefig(join(save_dir, f'neuron_{neuron_idx}_fluor_vs_deconv_overlay_all_trials.pdf'))
        plt.close(fig)

def plot_odor_start_quantification(per_neuron_df_fluor,
                                   per_neuron_df_spks,
                                   save_dir,
                                   fluor_label='fluorescence',
                                   spks_label='deconvolved'):
    os.makedirs(save_dir, exist_ok=True)

    merged = pd.merge(
        per_neuron_df_fluor,
        per_neuron_df_spks,
        on='neuron_idx',
        how='inner',
        suffixes=(f'_{fluor_label}', f'_{spks_label}')
    )

    merged.to_csv(join(save_dir, f'odor_start_quantification_{fluor_label}_vs_{spks_label}.csv'), index=False)
    with open(join(save_dir, f'odor_start_quantification_{fluor_label}_vs_{spks_label}.pkl'), 'wb') as f:
        pickle.dump(merged, f)

    if len(merged) == 0:
        print('No overlapping neurons for odor-start quantification.')
        return merged

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # percent of peak left at odor start
    x = merged[f'median_odor_start_to_peak_percent_{fluor_label}']
    y = merged[f'median_odor_start_to_peak_percent_{spks_label}']
    axes[0, 0].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[0, 0].plot(lims, lims, linestyle='--', color='gray')
    axes[0, 0].set_title('Median odor-start % of peak')
    axes[0, 0].set_xlabel(f'{fluor_label}')
    axes[0, 0].set_ylabel(f'{spks_label}')

    # percent decrease by odor start
    x = merged[f'median_odor_start_decrease_percent_{fluor_label}']
    y = merged[f'median_odor_start_decrease_percent_{spks_label}']
    axes[0, 1].scatter(x, y, alpha=0.7)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    axes[0, 1].plot(lims, lims, linestyle='--', color='gray')
    axes[0, 1].set_title('Median odor-start decrease %')
    axes[0, 1].set_xlabel(f'{fluor_label}')
    axes[0, 1].set_ylabel(f'{spks_label}')

    # histogram of percent left
    axes[1, 0].hist(
        merged[f'median_odor_start_to_peak_percent_{fluor_label}'].dropna(),
        bins=20, alpha=0.5, label=fluor_label, edgecolor='black'
    )
    axes[1, 0].hist(
        merged[f'median_odor_start_to_peak_percent_{spks_label}'].dropna(),
        bins=20, alpha=0.5, label=spks_label, edgecolor='black'
    )
    axes[1, 0].set_title('Histogram: odor-start % of peak')
    axes[1, 0].set_xlabel('Percent of peak')
    axes[1, 0].set_ylabel('Neuron count')
    axes[1, 0].legend()

    # histogram of percent decrease
    axes[1, 1].hist(
        merged[f'median_odor_start_decrease_percent_{fluor_label}'].dropna(),
        bins=20, alpha=0.5, label=fluor_label, edgecolor='black'
    )
    axes[1, 1].hist(
        merged[f'median_odor_start_decrease_percent_{spks_label}'].dropna(),
        bins=20, alpha=0.5, label=spks_label, edgecolor='black'
    )
    axes[1, 1].set_title('Histogram: odor-start decrease %')
    axes[1, 1].set_xlabel('Percent decrease from peak')
    axes[1, 1].set_ylabel('Neuron count')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(join(save_dir, f'odor_start_quantification_{fluor_label}_vs_{spks_label}.png'))
    plt.savefig(join(save_dir, f'odor_start_quantification_{fluor_label}_vs_{spks_label}.pdf'))
    plt.close(fig)

    return merged
# =========================
# Main
# =========================
if __name__ == '__main__':
    mouse = 'OI208'
    day = ['250224']

    alpha = 0.05

    ROOT_PATH = r'Z:\2p_data\{}\{}\ML\data'.format(mouse, day[0])
    licking_data = r'Z:\2p_data\{}\{}\ML\licking_analysis_final'.format(mouse, day[0])
    auroc_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\auROC_z_no_sub_history'.format(mouse, day[0])
    suite2p_path = r'Z:\2p_data\{}\{}\OI208_run1\suite2p\plane0'.format(mouse, day[0])
    save_loc = r'Z:\2p_data\{}\{}\ML\data\processed_data\opto_decay_trialwise_v3'.format(mouse, day[0])

    os.makedirs(save_loc, exist_ok=True)

    # -------------------------
    # Load data, same as your code
    # -------------------------
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

    # -------------------------
    # load the indexes of light significant neurons from your previous analysis
    # -------------------------

    with open(join(auroc_loc, 'significant_light_neurons.pkl'), 'rb') as f:
        sig_light_neurons = pickle.load(f)

    with open(join(auroc_loc, 'significant_light_neurons_increase.pkl'), 'rb') as f:
        sig_light_increase = pickle.load(f)

    with open(join(auroc_loc, 'significant_light_neurons_decrease.pkl'), 'rb') as f:
        sig_light_decrease = pickle.load(f)

    with open(join(auroc_loc, 'significant_neurons_reward.pkl'), 'rb') as f:
        sig_neurons_reward = pickle.load(f)

    with open(join(auroc_loc, 'significant_neurons_both_light_reward.pkl'), 'rb') as f:
        sig_neurons_both = pickle.load(f)

    # light only populations
    light_only_neurons = np.setdiff1d(sig_light_neurons, sig_neurons_reward)
    light_only_increase_neurons = np.setdiff1d(sig_light_increase, sig_neurons_reward)
    light_only_decrease_neurons = np.setdiff1d(sig_light_decrease, sig_neurons_reward)

    print(f"sig_light_neurons: {len(sig_light_neurons)}")
    print(f"sig_neurons_reward: {len(sig_neurons_reward)}")
    print(f"sig_neurons_both: {len(sig_neurons_both)}")
    print(f"light_only_neurons: {len(light_only_neurons)}")
    print(f"light_only_increase_neurons: {len(light_only_increase_neurons)}")
    print(f"light_only_decrease_neurons: {len(light_only_decrease_neurons)}")

    # -------------------------
    # Normalize:
    # z-score each neuron across all time points
    # -------------------------
    dff_norm = []
    for i in range(len(dff_4min)):
        mean = np.nanmean(dff_4min.iloc[i])
        std = np.nanstd(dff_4min.iloc[i])
        if std == 0 or np.isnan(std):
            std = 1.0
        dff_norm.append((dff_4min.iloc[i] - mean) / std)
    dff_norm = pd.DataFrame(dff_norm)

    with open(join(save_loc, 'all_values_zscore.pkl'), 'wb') as f:
        pickle.dump(dff_norm, f)

    # -------------------------
    # Divide to trials
    # -------------------------
    dff_run_list, pre_runs = divide_to_trials(dff_norm, trials_ind, run_len, pre_trial_length)

    with open(join(save_loc, 'dff_run_list.pkl'), 'wb') as f:
        pickle.dump(dff_run_list, f)

    # -------------------------
    # Trial indices
    # -------------------------
    light_ind = np.where(np.isin(conditions_played[0:len(conditions_played)], con1))[0]
    no_light_ind = np.where((conditions_played[0:len(conditions_played)] == con2))[0]

    correct_reject = np.where(responses == 4)[0]
    false_alarm = np.where(responses == 5)[0]
    hit = np.where(responses == 3)[0]
    miss = np.where(responses == 2)[0]

    with open(join(save_loc, 'light_ind.pkl'), 'wb') as f:
        pickle.dump(light_ind, f)

    # -------------------------
    # Find opto onset and offset per trial
    # -------------------------
    opto_on_ind = []
    for i in range(len(opto_on)):
        changes = np.diff(np.concatenate(([0], np.asarray(opto_on[i]) > 1)))
        ind = np.where(changes == 1)[0]
        opto_on_ind.append(ind[0] if len(ind) > 0 else np.nan)

    opto_on_ind = [i + pre_trial_length if not np.isnan(i) else np.nan for i in opto_on_ind]
    opto_off_ind = [i + stim_len if not np.isnan(i) else np.nan for i in opto_on_ind]

    with open(join(save_loc, 'opto_on_ind.pkl'), 'wb') as f:
        pickle.dump(opto_on_ind, f)
    with open(join(save_loc, 'opto_off_ind.pkl'), 'wb') as f:
        pickle.dump(opto_off_ind, f)

    # -------------------------
    # Divide data to conditions
    # -------------------------
    dff_correct_reject, dff_false_alarm, dff_hit, dff_miss, dff_light_all = divide_to_conditions(
        dff_run_list,
        correct_reject,
        false_alarm,
        hit,
        miss,
        light_ind
    )

    with open(join(save_loc, 'dff_light_all.pkl'), 'wb') as f:
        pickle.dump(dff_light_all, f)

    # -------------------------
    # Opto indices only for light trials
    # -------------------------
    opto_on_inds_light = [opto_on_ind[i] for i in light_ind]
    opto_off_inds_light = [opto_off_ind[i] for i in light_ind]

    # Fill missing on/off with expected timing
    default_on = pre_trial_length + 1 * rec_hz
    default_off = pre_trial_length + 5 * rec_hz
    opto_on_inds_light = [default_on if np.isnan(x) else int(x) for x in opto_on_inds_light]
    opto_off_inds_light = [default_off if np.isnan(x) else int(x) for x in opto_off_inds_light]

    # -----------------------------------------
    # Load Suite2p deconvolved signal
    # -----------------------------------------

    spks_cells = process_spks_data(suite2p_path)

    # Optional shape check
    if spks_cells.shape[0] < np.max(light_only_increase_neurons) + 1:
        raise ValueError('Neuron indices in light_only_increase_neurons exceed spks cell count.')

    # If your fluorescence analysis already used dff_4min in the same cell order,
    # use dff_4min as the fluorescence signal matrix
    fluor_signal = dff_4min.to_numpy() if isinstance(dff_4min, pd.DataFrame) else np.asarray(dff_4min)
    spks_signal = np.asarray(spks_cells)

    if fluor_signal.shape[0] != spks_signal.shape[0]:
        raise ValueError(
            f'Fluorescence and spks have different neuron counts: '
            f'{fluor_signal.shape[0]} vs {spks_signal.shape[0]}. '
            f'This must match after iscell filtering.'
        )

    # -----------------------------------------
    # Choose neuron subset
    # -----------------------------------------
    neuron_subset = light_only_increase_neurons

    # -----------------------------------------
    # Save root
    # -----------------------------------------
    multisignal_save_dir = join(save_loc, 'decay_multisignal')
    os.makedirs(multisignal_save_dir, exist_ok=True)

    # -----------------------------------------
    # Run fluorescence version
    # -----------------------------------------
    fluor_results = run_decay_pipeline_for_signal(
        signal_matrix=fluor_signal,
        signal_name='fluorescence',
        save_root=multisignal_save_dir,
        neuron_indices=neuron_subset,
        trials_ind=trials_ind,
        run_len=run_len,
        pre_trial_length=pre_trial_length,
        correct_reject=correct_reject,
        false_alarm=false_alarm,
        hit=hit,
        miss=miss,
        light_ind=light_ind,
        opto_on_inds_light=opto_on_inds_light,
        opto_off_inds_light=opto_off_inds_light,
        rec_hz=rec_hz,
        baseline_frames=1 * rec_hz,
        smooth_window=5,
        min_fit_r2=0.2,
        min_valid_trials_per_neuron=5,
        require_significant_trials=True,
        trial_sig_z=2.0,
        trial_sig_min_consecutive=2
    )

    # -----------------------------------------
    # Run deconvolved version
    # For spks, a slightly larger smoothing window is often helpful
    # -----------------------------------------
    spks_results = run_decay_pipeline_for_signal(
        signal_matrix=spks_signal,
        signal_name='deconvolved_spks',
        save_root=multisignal_save_dir,
        neuron_indices=neuron_subset,
        trials_ind=trials_ind,
        run_len=run_len,
        pre_trial_length=pre_trial_length,
        correct_reject=correct_reject,
        false_alarm=false_alarm,
        hit=hit,
        miss=miss,
        light_ind=light_ind,
        opto_on_inds_light=opto_on_inds_light,
        opto_off_inds_light=opto_off_inds_light,
        rec_hz=rec_hz,
        baseline_frames=1 * rec_hz,
        smooth_window=5,
        min_fit_r2=0.2,
        min_valid_trials_per_neuron=5,
        require_significant_trials=True,
        trial_sig_z=2.0,
        trial_sig_min_consecutive=2
    )

    # -----------------------------------------
    # Compare together
    # -----------------------------------------
    comparison_dir = join(multisignal_save_dir, 'comparison')
    comparison_df = compare_modalities(
        per_neuron_df_fluor=fluor_results['per_neuron_df'],
        per_neuron_df_spks=spks_results['per_neuron_df'],
        save_dir=comparison_dir,
        fluor_label='fluorescence',
        spks_label='deconvolved'
    )

    plot_final_histograms_fluor_vs_deconv_separate(
        fluor_df=fluor_results['per_neuron_df'],
        deconv_df=spks_results['per_neuron_df'],
        save_dir=join(multisignal_save_dir, 'comparison'),
        prefix='final_histograms_fluor_vs_deconv',
        tau_col='median_tau_s',  # or 'median_tau_fast_s' if that is your summary column
        drop_col='median_t_2third_drop_s'
    )

    overlay_save_dir = join(multisignal_save_dir, 'fluor_vs_deconv_overlay_per_neuron')

    plot_overlay_fluor_deconv_per_neuron(
        fluor_light_all=fluor_results['signal_light_all'],
        deconv_light_all=spks_results['signal_light_all'],
        fluor_per_trial_df=fluor_results['per_trial_df'],
        deconv_per_trial_df=spks_results['per_trial_df'],
        opto_on_inds_light=opto_on_inds_light,
        opto_off_inds_light=opto_off_inds_light,
        neuron_indices=neuron_subset,
        save_dir=overlay_save_dir,
        baseline_frames=1 * rec_hz,
        fluor_smooth_window=5,
        deconv_smooth_window=5,
        max_trials_per_neuron=None  # set to a number if you want to limit panels
    )

    odor_comparison_df = plot_odor_start_quantification(
        per_neuron_df_fluor=fluor_results['per_neuron_df'],
        per_neuron_df_spks=spks_results['per_neuron_df'],
        save_dir=comparison_dir,
        fluor_label='fluorescence',
        spks_label='deconvolved'
    )

    print('Finished fluorescence and deconvolved decay analysis.')
    print(f'Fluorescence neurons summarized: {len(fluor_results["per_neuron_df"])}')
    print(f'Deconvolved neurons summarized: {len(spks_results["per_neuron_df"])}')
    print(f'Overlapping neurons in comparison: {len(comparison_df)}')
